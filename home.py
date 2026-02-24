# streamlit_app.py
import streamlit as st
import pandas as pd

from cost_proxy import Inputs, build_cost_proxy, PRICE_AREAS, fetch_elhub_norgespris_consumption_comparison

st.set_page_config(page_title="Norgespris og Strømstøtte proxy kostnad ",
                   layout="wide")
st.title("Norgespris og Strømstøtte proxy kostnad ")


# No top-level @st.cache_data needed — granular per-chunk caching
# is now handled inside cost_proxy.py with calendar-month-aligned keys
# and tiered TTLs (7 days for historical months, 1 hour for current).
def fetch_data(start_str: str, end_str: str) -> pd.DataFrame:
    """Thin wrapper — caching is handled at the chunk level inside cost_proxy."""
    start = pd.Timestamp(start_str, tz="Europe/Oslo")
    end = pd.Timestamp(end_str, tz="Europe/Oslo")
    return build_cost_proxy(Inputs(start=start, end=end))


def fetch_consumption_comparison(start_str: str, end_str: str) -> pd.DataFrame:
    """Thin wrapper — caching is handled at the chunk level inside cost_proxy."""
    start = pd.Timestamp(start_str, tz="Europe/Oslo")
    end = pd.Timestamp(end_str, tz="Europe/Oslo")
    return fetch_elhub_norgespris_consumption_comparison(start, end)


# Budget constant (11 billion NOK)
BUDGET_NOK = 11e9

# Sidebar filters
st.sidebar.header("Innstillinger")
start_date_option = st.sidebar.radio(
    "Startdato",
    options=["1. oktober 2025", "1. desember 2025"],
    index=1
)
start_date = pd.to_datetime("2025-10-01").date() if start_date_option == "1. oktober 2025" else pd.to_datetime("2025-12-01").date()
selected_zones = PRICE_AREAS  # Use all zones automatically

if st.sidebar.button("Last data", type="primary"):
    start_str = str(start_date) + " 00:00:00"
    end_str = pd.Timestamp.now(tz="Europe/Oslo").date().isoformat() + " 23:00:00"

    with st.spinner("loading data from Elhub and Entsoe"):
        df = fetch_data(start_str, end_str)
        consumption_comp = fetch_consumption_comparison(start_str, end_str)

    if df.empty:
        st.warning("Ingen data returnert. Sjekk entsoe-tilkobling.")
    else:
        # Store in session state for persistence
        st.session_state["df"] = df
        st.session_state["selected_zones"] = selected_zones
        st.session_state["consumption_comp"] = consumption_comp

# Check if we have data in session state
if "df" in st.session_state:
    df = st.session_state["df"]
    selected_zones = st.session_state.get("selected_zones", PRICE_AREAS)

    # Daily aggregation
    daily = df.groupby("date", as_index=False).agg(
        total_state_cost_nok=("total_state_cost_nok", "sum"),
        np_gain_loss_nok=("np_gain_loss_nok", "sum"),
        support_nok=("support_nok", "sum"),
        np_vat_loss_nok=("np_vat_loss_nok", "sum"),
        support_vat_loss_nok=("support_vat_loss_nok", "sum"),
        volume_kwh=("volume_kwh", "sum"),
        vol_np_kwh=("vol_np_kwh", "sum"),
        norgespris_count=("norgespris_count", "sum"),
        total_count=("total_count", "sum"),
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    # Calculate cumulative sum
    daily["cumulative_cost_nok"] = daily["total_state_cost_nok"].cumsum()

    # Calculate daily share of Norgespris
    daily["share_np_pct"] = (daily["norgespris_count"] / daily["total_count"] * 100).fillna(0)

    # ----- TABS -----
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Daglig utvikling",
        "📈 Kumulativ kostnad",
        "🏠 Norgespris-andel",
        "⚡ Forbruksmultiplikator",
        "📋 Tabell"
    ])

    with tab1:
        st.subheader("Daglig kostnad for staten")
        st.caption("Inkluderer Norgespris (kan være negativ ved lav pris), strømstøtte og tapt MVA. Antar at Norgespris-kunder har samme forbruk som snittet av forbrukere innad i et budområde.")

        chart_data = daily.set_index("date")[[
            "np_gain_loss_nok", "support_nok", "np_vat_loss_nok", "support_vat_loss_nok"
        ]].rename(columns={
            "np_gain_loss_nok": "Norgespris (gevinst/tap)",
            "support_nok": "Strømstøtte",
            "np_vat_loss_nok": "Tapt MVA (Norgespris)",
            "support_vat_loss_nok": "Tapt MVA (Strømstøtte)"
        })
        st.line_chart(chart_data)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_np = daily["np_gain_loss_nok"].sum()
            st.metric(
                "Norgespris totalt",
                f"{total_np / 1e9:.2f} mrd NOK",
                delta="kostnad" if total_np > 0 else "inntekt"
            )
        with col2:
            st.metric("Strømstøtte totalt", f"{daily['support_nok'].sum() / 1e9:.2f} mrd NOK")
        with col3:
            total_vat = daily["np_vat_loss_nok"].sum() + daily["support_vat_loss_nok"].sum()
            st.metric("Tapt MVA totalt", f"{total_vat / 1e9:.2f} mrd NOK")
        with col4:
            st.metric("Total kostnad", f"{daily['total_state_cost_nok'].sum() / 1e9:.2f} mrd NOK")

    with tab2:
        st.subheader("Kumulativ kostnad for staten")
        st.caption(f"Budsjett: 11 mrd NOK")

        # Create cumulative chart with budget line
        cum_data = daily[["date", "cumulative_cost_nok"]].copy()
        cum_data["Budsjett (11 mrd)"] = BUDGET_NOK
        cum_data = cum_data.set_index("date")
        cum_data = cum_data.rename(columns={"cumulative_cost_nok": "Kumulativ kostnad"})

        st.line_chart(cum_data)

        # Show how much of budget is used
        total_cost = daily["cumulative_cost_nok"].iloc[-1] if len(daily) > 0 else 0
        budget_pct = (total_cost / BUDGET_NOK) * 100
        st.progress(min(budget_pct / 100, 1.0))
        st.write(f"**{budget_pct:.1f}%** av budsjett brukt ({total_cost / 1e9:.2f} av 11 mrd NOK)")

    with tab3:
        st.subheader("Andel med Norgespris")
        st.caption("Prosentandel av husholdninger og fritidsboliger som har valgt Norgespris")

        # Chart of share over time
        share_data = daily.set_index("date")[["share_np_pct"]].rename(
            columns={"share_np_pct": "Andel med Norgespris (%)"}
        )
        st.line_chart(share_data)

        # Breakdown by price area
        st.subheader("Norgespris-andel per prisområde")
        area_share = df.drop_duplicates(subset=["date", "price_area", "cons_group"], keep="last")
        area_share = area_share.groupby(["date", "price_area"], as_index=False).agg(
            norgespris_count=("norgespris_count", "sum"),
            total_count=("total_count", "sum"),
        )
        area_share["share_np_pct"] = (area_share["norgespris_count"] / area_share["total_count"] * 100).fillna(0)

        # Pivot for chart
        area_pivot = area_share.pivot(index="date", columns="price_area", values="share_np_pct").fillna(0)
        st.line_chart(area_pivot)

        # Latest share per area
        st.subheader("Siste andel per prisområde")
        latest = area_share.groupby("price_area").last().reset_index()
        for _, row in latest.iterrows():
            st.write(f"**{row['price_area']}**: {row['share_np_pct']:.1f}% ({int(row['norgespris_count']):,} av {int(row['total_count']):,})")

    with tab4:
        st.subheader("Forbruksmultiplikator for Norgespris-kunder")
        st.caption("""
        Viser hvor mye mer (>1) eller mindre (<1) strøm Norgespris-kunder bruker 
        sammenlignet med deres andel av befolkningen. 
        Beregnet som: (Forbruk_NP / Forbruk_Total) / (Antall_NP / Antall_Total)
        """)
        
        
        consumption_comp = st.session_state.get("consumption_comp", pd.DataFrame())
        
        if not consumption_comp.empty:
            consumption_comp["date"] = pd.to_datetime(consumption_comp["date"])
            consumption_comp = consumption_comp.sort_values(["date", "price_area", "cons_group"])
            
            # Auto-exclude last 3 days due to incomplete data from Elhub
            cutoff_date = (pd.Timestamp.now(tz="Europe/Oslo") - pd.Timedelta(days=3)).date()
            consumption_comp = consumption_comp[consumption_comp["date"] < pd.Timestamp(cutoff_date)]
            
            # Add 7-day rolling average option
            show_rolling = st.checkbox("Vis 7-dagers glidende gjennomsnitt", value=True)
            
            # Filter only household consumption group
            filtered_data = consumption_comp[consumption_comp["cons_group"] == "household"].copy()
            
            # Chart: Multiplier per bidding zone for household
            st.subheader("Multiplikator per prisområde - Husholdning")
            
            zone_chart = filtered_data.pivot(index="date", columns="price_area", values="consumption_multiplier").fillna(1.0)
            
            if show_rolling:
                zone_chart_display = zone_chart.rolling(window=7, min_periods=1).mean()
                st.caption("7-dagers glidende gjennomsnitt for å jevne ut daglige variasjoner")
            else:
                zone_chart_display = zone_chart
            st.line_chart(zone_chart_display)
            
            # Summary metrics per price area
            st.subheader("Gjennomsnitt per prisområde")
            cols = st.columns(5)
            for i, pa in enumerate(["NO1", "NO2", "NO3", "NO4", "NO5"]):
                pa_data = filtered_data[filtered_data["price_area"] == pa]
                with cols[i]:
                    if not pa_data.empty:
                        avg_mult = pa_data["consumption_multiplier"].mean()
                        st.metric(
                            pa,
                            f"{avg_mult:.3f}",
                            delta=f"{(avg_mult - 1) * 100:.1f}%"
                        )
            
            # Detailed table per zone
            st.subheader("Detaljer per prisområde (siste 14 dager)")
            recent_data = filtered_data[filtered_data["date"] >= filtered_data["date"].max() - pd.Timedelta(days=14)].copy()
            detail_pivot = recent_data.pivot(index="date", columns="price_area", values="consumption_multiplier")
            detail_pivot = detail_pivot.round(3)
            st.dataframe(detail_pivot, width='stretch')
            
        else:
            st.info("Ingen forbruksdata tilgjengelig.")

    with tab5:
        st.subheader("Siste 14 dager (tabell)")

        display_cols = [
            "date", "total_state_cost_nok", "np_gain_loss_nok", "support_nok",
            "np_vat_loss_nok", "support_vat_loss_nok", "share_np_pct"
        ]
        display_df = daily[display_cols].tail(14).copy()
        display_df.columns = [
            "Dato", "Total kostnad", "Norgespris", "Strømstøtte",
            "Tapt MVA (NP)", "Tapt MVA (Støtte)", "Norgespris-andel %"
        ]

        # Format numbers
        for col in ["Total kostnad", "Norgespris", "Strømstøtte", "Tapt MVA (NP)", "Tapt MVA (Støtte)"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x/1e6:.1f} MNOK")

        st.dataframe(display_df, width='stretch')

        # Download button
        csv = daily.to_csv(index=False)
        st.download_button(
            "Last ned alle data (CSV)",
            csv,
            "norgespris_data.csv",
            "text/csv"
        )
else:
    st.info("Klikk 'Last data' i sidepanelet for å hente data.")

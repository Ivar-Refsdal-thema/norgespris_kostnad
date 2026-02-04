# streamlit_app.py
import streamlit as st
import pandas as pd

from cost_proxy import Inputs, build_cost_proxy, PRICE_AREAS

st.set_page_config(page_title="Norgespris og Strømstøtte proxy kostnad ",
                   layout="wide")
st.title("Norgespris og Strømstøtte proxy kostnad ")


# Cache data for 10 minutes (600 seconds)
@st.cache_data(ttl=600)
def fetch_data_cached(start_str: str, end_str: str) -> pd.DataFrame:
    """Cached wrapper for build_cost_proxy to avoid repeated API calls."""
    start = pd.Timestamp(start_str, tz="Europe/Oslo")
    end = pd.Timestamp(end_str, tz="Europe/Oslo")
    return build_cost_proxy(Inputs(start=start, end=end))


# Budget constant (11 billion NOK)
BUDGET_NOK = 11e9

# Sidebar filters
st.sidebar.header("Innstillinger")
start_date = st.sidebar.date_input("Startdato", value=pd.to_datetime("2025-12-01").date())
selected_zones = st.sidebar.multiselect(
    "Prisområder",
    options=PRICE_AREAS,
    default=PRICE_AREAS,
    help="Velg ett eller flere prisområder"
)

if st.sidebar.button("Last data", type="primary"):
    if not selected_zones:
        st.warning("Velg minst ett prisområde")
    else:
        start_str = str(start_date) + " 00:00:00"
        end_str = pd.Timestamp.now(tz="Europe/Oslo").date().isoformat() + " 23:00:00"

        with st.spinner("Henter data fra Elhub og entsoe..."):
            df = fetch_data_cached(start_str, end_str)

        if df.empty:
            st.warning("Ingen data returnert. Sjekk entsoe-tilkobling.")
        else:
            # Filter by selected zones
            df = df[df["price_area"].isin(selected_zones)].copy()

            if df.empty:
                st.warning("Ingen data for valgte prisområder.")
            else:
                # Store in session state for persistence
                st.session_state["df"] = df
                st.session_state["selected_zones"] = selected_zones

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
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Daglig utvikling",
        "📈 Kumulativ kostnad",
        "🏠 Norgespris-andel",
        "📋 Tabell"
    ])

    with tab1:
        st.subheader("Daglig kostnad for staten")
        st.caption("Inkluderer Norgespris (kan være negativ ved lav pris), strømstøtte og tapt MVA")

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
        st.caption(f"Budsjett: 11 mrd NOK (vist som rød linje)")

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
        area_share = df.groupby(["date", "price_area"], as_index=False).agg(
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

        st.dataframe(display_df, use_container_width=True)

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

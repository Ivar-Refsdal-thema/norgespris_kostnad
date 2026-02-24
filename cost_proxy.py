from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlencode

import pandas as pd
import requests
from entsoe import EntsoePandasClient

import os
import streamlit as st

# Optional: local dev only
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def get_secret(name: str) -> str:
    # 1) Try Streamlit secrets (Cloud or local secrets.toml)
    try:
        if hasattr(st, 'secrets') and name in st.secrets:
            return st.secrets[name]
    except (KeyError, AttributeError, Exception):
        # No secrets.toml / secrets configured locally
        pass

    # 2) Fallback to env vars (incl. .env via dotenv)
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Missing secret '{name}'. "
            f"Set it in Streamlit Secrets (Cloud) or as an environment variable/.env (local)."
        )
    return val


# Placeholder for the initial EUR_TO_NOK (will be fetched dynamically)
EUR_TO_NOK = 11.5  # Fallback value if API fails
NORGESPRIS_CAP_EUR_PER_MWH = 400.0 / EUR_TO_NOK  # ~34.78 EUR/MWh to get 40 øre/kWh
SUPPORT_THRESHOLD_NOK_PER_KWH = 0.77
SUPPORT_RATE = 0.90
VAT_RATE = 0.25  # 25% MVA

# Consumption groups to include
CONSUMPTION_GROUPS = ["household", "cabin"]

# Elhub Energy Data API base URL
ELHUB_API_BASE = "https://api.elhub.no/energy-data/v0"
PRICE_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]

# Zone mapping (price area -> ENTSO-E area code)
# entsoe-py uses codes like 'NO_1', 'NO_2' which it maps to EIC codes internally
ENTSOE_ZONE_MAP = {
    "NO1": "NO_1",
    "NO2": "NO_2",
    "NO3": "NO_3",
    "NO4": "NO_4",
    "NO5": "NO_5",
}


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def fetch_eur_nok_rates(start_str: str, end_str: str) -> pd.DataFrame:
    """
    Fetch daily EUR/NOK exchange rates from Norges Bank API for the entire date range.
    Returns DataFrame with columns: date, eur_to_nok
    Falls back to EUR_TO_NOK constant if API fails.
    """
    rows = []
    start = pd.Timestamp(start_str, tz="Europe/Oslo")
    end = pd.Timestamp(end_str, tz="Europe/Oslo")
    start_date = start.date() if hasattr(start, 'date') else pd.Timestamp(start).date()
    end_date = end.date() if hasattr(end, 'date') else pd.Timestamp(end).date()
    
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()
    
    try:
        # Norges Bank API: fetch entire range at once
        url = f"https://data.norges-bank.no/api/data/EXR/B.EUR.NOK.SP?format=sdmx-json&startPeriod={start_str}&endPeriod={end_str}&locale=no"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if "data" in data and "dataSets" in data["data"] and len(data["data"]["dataSets"]) > 0:
            dataset = data["data"]["dataSets"][0]
            
            # Extract time periods from structure
            time_periods = []
            if "structure" in data["data"]:
                structure = data["data"]["structure"]
                if "dimensions" in structure and "observation" in structure["dimensions"]:
                    obs_dim = structure["dimensions"]["observation"]
                    if len(obs_dim) > 0 and "values" in obs_dim[0]:
                        for val in obs_dim[0]["values"]:
                            time_periods.append(val["id"])
            
            # Extract observations from series
            if "series" in dataset:
                for series_key, series_data in dataset["series"].items():
                    if "observations" in series_data:
                        observations = series_data["observations"]
                        # observations is a dict where keys are indices and values are lists like ['11.771']
                        for obs_idx_str, obs_value_list in observations.items():
                            try:
                                obs_idx = int(obs_idx_str)
                                if obs_idx < len(time_periods):
                                    date_str = time_periods[obs_idx]
                                    rate = float(obs_value_list[0])
                                    rows.append({
                                        "date": pd.Timestamp(date_str).date(),
                                        "eur_to_nok": rate
                                    })
                            except (ValueError, IndexError, TypeError):
                                continue
        
        if rows:
            return pd.DataFrame(rows).drop_duplicates(subset=["date"]).sort_values("date")
    
    except Exception as e:
        print(f"Warning: Failed to fetch EUR/NOK rates from Norges Bank: {e}")
    print("Falling back to constant EUR_TO_NOK rate for missing dates.")
    # Fallback: fill in all dates with constant rate
    current_date = start_date
    while current_date <= end_date:
        rows.append({"date": current_date, "eur_to_nok": EUR_TO_NOK})
        current_date += pd.Timedelta(days=1)
    
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Inputs:
    start: pd.Timestamp  # inclusive
    end: pd.Timestamp  # inclusive


# -----------------------------
# ENTSO-E client helper
# -----------------------------
def _get_entsoe_client() -> EntsoePandasClient:
    """Get ENTSO-E Pandas client using API key from environment."""
    api_key = get_secret("ENTSOE_KEY")
    if not api_key:
        raise ValueError("ENTSOE_KEY not found in environment variables")
    return EntsoePandasClient(api_key=api_key)


# -----------------------------
# Helpers: chunking and cache tiering
# -----------------------------
def _calendar_month_chunks(start: pd.Timestamp, end: pd.Timestamp):
    """Yield (chunk_start, chunk_end) pairs aligned to calendar month boundaries.

    Calendar alignment ensures that shared months (e.g. Dec, Jan, Feb)
    produce identical cache keys regardless of whether the overall query
    starts in October or December.
    """
    current = start
    while current < end:
        next_month = current.replace(day=1) + pd.DateOffset(months=1)
        if current.tz is not None and next_month.tz is None:
            next_month = next_month.tz_localize(current.tz)
        chunk_end = min(next_month, end)
        yield current, chunk_end
        current = chunk_end


def _chunk_is_historical(chunk_start: pd.Timestamp) -> bool:
    """True if chunk covers a completed calendar month (not the current one).

    Historical chunks get a longer cache TTL since their data won't change.
    """
    now = pd.Timestamp.now(tz="Europe/Oslo")
    return (chunk_start.year, chunk_start.month) < (now.year, now.month)


# --- Per-chunk cached fetchers: Elhub Consumption ---

def _fetch_consumption_chunk_raw(start_iso: str, end_iso: str, pa: str, cons_group: str) -> list[dict]:
    """Raw Elhub API call for one consumption chunk (uncached)."""
    params = {
        "dataset": "CONSUMPTION_PER_GROUP_MBA_HOUR",
        "startDate": start_iso,
        "endDate": end_iso,
        "consumptionGroup": cons_group,
    }
    url = f"{ELHUB_API_BASE}/price-areas/{pa}?{urlencode(params)}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    j = r.json()
    rows = []
    for item in j["data"]:
        for rec in item["attributes"].get("consumptionPerGroupMbaHour", []):
            rows.append({
                "start_time": rec["startTime"],
                "price_area": rec["priceArea"],
                "cons_group": cons_group,
                "volume_kwh": rec["quantityKwh"],
            })
    return rows


@st.cache_data(ttl=3600 * 24 * 7, show_spinner=False)
def _fetch_consumption_chunk_hist(start_iso: str, end_iso: str, pa: str, cons_group: str) -> list[dict]:
    """Historical chunk — 7-day TTL (data is final)."""
    return _fetch_consumption_chunk_raw(start_iso, end_iso, pa, cons_group)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_consumption_chunk_current(start_iso: str, end_iso: str, pa: str, cons_group: str) -> list[dict]:
    """Current-month chunk — 1-hour TTL (data still updating)."""
    return _fetch_consumption_chunk_raw(start_iso, end_iso, pa, cons_group)


def fetch_elhub_consumption(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch hourly household + cabin consumption via calendar-month cached chunks."""
    rows: list[dict] = []
    for chunk_start, chunk_end in _calendar_month_chunks(start, end):
        fetch = _fetch_consumption_chunk_hist if _chunk_is_historical(chunk_start) else _fetch_consumption_chunk_current
        cs, ce = str(chunk_start.date()), str(chunk_end.date())
        for pa in PRICE_AREAS:
            for cons_group in CONSUMPTION_GROUPS:
                rows.extend(fetch(cs, ce, pa, cons_group))
    if not rows:
        return pd.DataFrame(columns=["start_time", "price_area", "cons_group", "volume_kwh"])
    df = pd.DataFrame(rows)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    return df


# --- Per-chunk cached fetchers: Elhub Norgespris (unified) ---
#
# Both fetch_elhub_norgespris and fetch_elhub_norgespris_consumption_comparison
# hit the SAME Elhub endpoint (NORGESPRIS_CONSUMPTION_PER_GROUP_EAC_MBA).
# A single cached fetcher eliminates duplicate API calls.

def _fetch_norgespris_chunk_raw(start_iso: str, end_iso: str, pa: str, cons_group: str) -> list[dict]:
    """Raw Elhub API call for one norgespris chunk (uncached).
    Returns all fields so both count and comparison views can derive from it."""
    params = {
        "dataset": "NORGESPRIS_CONSUMPTION_PER_GROUP_EAC_MBA",
        "startDate": start_iso,
        "endDate": end_iso,
        "consumptionGroup": cons_group,
        "granularity": "DAILY",
    }
    url = f"{ELHUB_API_BASE}/price-areas/{pa}?{urlencode(params)}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    j = r.json()
    rows = []
    for item in j.get("data", []):
        for rec in item.get("attributes", {}).get("norgesprisConsumptionPerGroupEacMba", []):
            rows.append({
                "start_time": rec.get("startTime"),
                "price_area": pa,
                "cons_group": cons_group,
                "norgespris_count": rec.get("meteringPointCountNorwayPrice", 0) or 0,
                "total_count": rec.get("totalMeteringPointCount", 0) or 0,
                "forbruk_np": rec.get("quantityKwhNorwayPrice", 0) or 0,
                "forbruk_total": rec.get("totalQuantityKwh", 0) or 0,
            })
    return rows


@st.cache_data(ttl=3600 * 24 * 7, show_spinner=False)
def _fetch_norgespris_chunk_hist(start_iso: str, end_iso: str, pa: str, cons_group: str) -> list[dict]:
    """Historical chunk — 7-day TTL."""
    return _fetch_norgespris_chunk_raw(start_iso, end_iso, pa, cons_group)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_norgespris_chunk_current(start_iso: str, end_iso: str, pa: str, cons_group: str) -> list[dict]:
    """Current-month chunk — 1-hour TTL."""
    return _fetch_norgespris_chunk_raw(start_iso, end_iso, pa, cons_group)


def _fetch_norgespris_unified(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch and cache norgespris data using calendar-month chunks.
    Returns all fields needed by both norgespris counts and consumption comparison."""
    rows: list[dict] = []
    for chunk_start, chunk_end in _calendar_month_chunks(start, end):
        fetch = _fetch_norgespris_chunk_hist if _chunk_is_historical(chunk_start) else _fetch_norgespris_chunk_current
        cs, ce = str(chunk_start.date()), str(chunk_end.date())
        for pa in PRICE_AREAS:
            for cons_group in CONSUMPTION_GROUPS:
                try:
                    rows.extend(fetch(cs, ce, pa, cons_group))
                except Exception as e:
                    print(f"Warning: Failed to fetch norgespris data for {pa}/{cons_group}: {e}")
                    continue
    if not rows:
        return pd.DataFrame(columns=[
            "date", "price_area", "cons_group", "norgespris_count", "total_count",
            "forbruk_np", "forbruk_total"
        ])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["start_time"], utc=True).dt.date
    df = df.drop(columns=["start_time"])
    # Aggregate EAC buckets per day/area/cons_group
    df = df.groupby(["date", "price_area", "cons_group"], as_index=False).agg({
        "norgespris_count": "sum",
        "total_count": "sum",
        "forbruk_np": "sum",
        "forbruk_total": "sum",
    })
    return df


def fetch_elhub_norgespris_consumption_comparison(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Derive consumption comparison from unified cached norgespris data.
    Eliminates duplicate API calls — shares cache with fetch_elhub_norgespris."""
    df = _fetch_norgespris_unified(start, end)
    if df.empty:
        return pd.DataFrame(columns=["date", "price_area", "cons_group", "forbruk_total", "forbruk_np",
                                      "antall_total", "antall_np", "share_consumption", "share_count",
                                      "consumption_multiplier"])
    result = df.rename(columns={"norgespris_count": "antall_np", "total_count": "antall_total"})
    result["share_consumption"] = 0.0
    result["share_count"] = 0.0
    result["consumption_multiplier"] = 1.0
    mask = (result["forbruk_total"] > 0) & (result["antall_total"] > 0) & (result["antall_np"] > 0)
    result.loc[mask, "share_consumption"] = result.loc[mask, "forbruk_np"] / result.loc[mask, "forbruk_total"]
    result.loc[mask, "share_count"] = result.loc[mask, "antall_np"] / result.loc[mask, "antall_total"]
    result.loc[mask, "consumption_multiplier"] = result.loc[mask, "share_consumption"] / result.loc[mask, "share_count"]
    return result


def fetch_elhub_norgespris(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Derive norgespris share from unified cached norgespris data.
    Eliminates duplicate API calls — shares cache with fetch_elhub_norgespris_consumption_comparison."""
    df = _fetch_norgespris_unified(start, end)
    if df.empty:
        return pd.DataFrame(columns=["date", "price_area", "cons_group", "norgespris_count", "total_count", "share_np"])
    result = df[["date", "price_area", "cons_group", "norgespris_count", "total_count"]].copy()
    result["share_np"] = 0.0
    mask = result["total_count"] > 0
    result.loc[mask, "share_np"] = (result.loc[mask, "norgespris_count"] / result.loc[mask, "total_count"]).clip(0, 1)
    return result


# --- Per-chunk cached fetchers: ENTSO-E Spot Prices ---

def _fetch_spot_chunk_raw(start_iso: str, end_iso: str, pa: str) -> list[dict]:
    """Raw ENTSO-E spot price fetch for one price area and one month chunk (uncached)."""
    client = _get_entsoe_client()
    zone_code = ENTSOE_ZONE_MAP[pa]
    start = pd.Timestamp(start_iso)
    end = pd.Timestamp(end_iso)
    if start.tz is None:
        start = start.tz_localize("Europe/Oslo")
    else:
        start = start.tz_convert("Europe/Oslo")
    if end.tz is None:
        end = end.tz_localize("Europe/Oslo")
    else:
        end = end.tz_convert("Europe/Oslo")
    rows = []
    try:
        prices = client.query_day_ahead_prices(zone_code, start=start, end=end)
        if prices is not None and not prices.empty:
            for timestamp, price in prices.items():
                rows.append({
                    "start_time": str(timestamp),
                    "price_area": pa,
                    "spot_eur_mwh": float(price),
                })
    except Exception as e:
        print(f"Failed to fetch ENTSO-E prices for {pa} ({start_iso} to {end_iso}): {e}")
    return rows


@st.cache_data(ttl=3600 * 24 * 7, show_spinner=False)
def _fetch_spot_chunk_hist(start_iso: str, end_iso: str, pa: str) -> list[dict]:
    """Historical chunk — 7-day TTL."""
    return _fetch_spot_chunk_raw(start_iso, end_iso, pa)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_spot_chunk_current(start_iso: str, end_iso: str, pa: str) -> list[dict]:
    """Current-month chunk — 1-hour TTL."""
    return _fetch_spot_chunk_raw(start_iso, end_iso, pa)


def fetch_spot_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch day-ahead spot prices via calendar-month cached chunks."""
    # Ensure timezone awareness
    if start.tz is None:
        start = start.tz_localize("Europe/Oslo")
    if end.tz is None:
        end = end.tz_localize("Europe/Oslo")

    rows: list[dict] = []
    for chunk_start, chunk_end in _calendar_month_chunks(start, end):
        fetch = _fetch_spot_chunk_hist if _chunk_is_historical(chunk_start) else _fetch_spot_chunk_current
        cs, ce = str(chunk_start.date()), str(chunk_end.date())
        for pa in PRICE_AREAS:
            rows.extend(fetch(cs, ce, pa))

    if not rows:
        return pd.DataFrame(columns=["start_time", "price_area", "spot_eur_mwh"])

    df = pd.DataFrame(rows)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)

    # Convert to Europe/Oslo
    df["start_time"] = df["start_time"].dt.tz_convert("Europe/Oslo")

    # ENTSO-E may return 15-min or hourly data; resample to hourly (mean)
    df = df.groupby(
        [pd.Grouper(key="start_time", freq="h"), "price_area"],
        as_index=False,
    ).agg({"spot_eur_mwh": "mean"})

    return df[["start_time", "price_area", "spot_eur_mwh"]]


def eur_mwh_to_nok_kwh(price_eur_per_mwh: pd.Series, eur_to_nok_rate: float | pd.Series) -> pd.Series:
    """
    Convert EUR/MWh to NOK/kWh using exchange rate(s).
    eur_to_nok_rate can be a scalar or a Series aligned with price_eur_per_mwh.
    """
    return (price_eur_per_mwh / 1000.0) * eur_to_nok_rate


# -----------------------------
# Core: bygg timesdatasett med kost
# -----------------------------
def build_cost_proxy(inputs: Inputs) -> pd.DataFrame:
    # 0) Fetch daily EUR/NOK exchange rates (cached by string keys)
    fx_rates = fetch_eur_nok_rates(str(inputs.start), str(inputs.end))
    
    # 1) Elhub: forbruk per time (via Energy Data API) - household + cabin
    cons = fetch_elhub_consumption(inputs.start, inputs.end)
    if cons.empty:
        return pd.DataFrame(columns=[
            "start_time", "price_area", "cons_group", "volume_kwh", "date", "spot_eur_mwh", "eur_to_nok", "spot_nok_kwh",
            "share_np", "norgespris_count", "total_count", "vol_np_kwh", "vol_rest_kwh",
            "price_np_nok_kwh", "support_nok_kwh", "price_rest_nok_kwh",
            "cost_np_nok", "cost_rest_nok", "support_nok",
            "np_gain_loss_nok", "np_vat_loss_nok", "support_vat_loss_nok", "total_state_cost_nok"
        ])

    # Ensure start_time is datetime with proper timezone handling
    cons["start_time"] = pd.to_datetime(cons["start_time"], utc=True).dt.tz_convert("Europe/Oslo")

    # Filtrer periode (API should already filter, but be safe)
    cons = cons[(cons["start_time"] >= inputs.start) & (cons["start_time"] <= inputs.end)].copy()

    if cons.empty:
        return pd.DataFrame(columns=[
            "start_time", "price_area", "cons_group", "volume_kwh", "date", "spot_eur_mwh", "eur_to_nok", "spot_nok_kwh",
            "share_np", "norgespris_count", "total_count", "vol_np_kwh", "vol_rest_kwh",
            "price_np_nok_kwh", "support_nok_kwh", "price_rest_nok_kwh",
            "cost_np_nok", "cost_rest_nok", "support_nok",
            "np_gain_loss_nok", "np_vat_loss_nok", "support_vat_loss_nok", "total_state_cost_nok"
        ])

    # 2) Elhub: norgespris-andel per dag (via Energy Data API)
    np_cnt = fetch_elhub_norgespris(inputs.start, inputs.end)

    # 3) Spotpriser per område
    spot = fetch_spot_prices(inputs.start, inputs.end)

    # 4) Slå sammen: timeforbruk + spot + norgespris-andel (per dag/cons_group)
    cons["date"] = cons["start_time"].dt.date
    df = cons.merge(spot, on=["start_time", "price_area"], how="left")
    df = df.merge(
        np_cnt[["date", "price_area", "cons_group", "share_np", "norgespris_count", "total_count"]],
        on=["date", "price_area", "cons_group"],
        how="left"
    )
    # Merge exchange rates by date
    df = df.merge(fx_rates, on="date", how="left")
    
    df["share_np"] = df["share_np"].fillna(0.0)
    df["norgespris_count"] = df["norgespris_count"].fillna(0)
    df["total_count"] = df["total_count"].fillna(0)
    df["eur_to_nok"] = df["eur_to_nok"].fillna(EUR_TO_NOK)  # Fallback to constant if merge fails

    # Convert spot price to NOK/kWh using daily exchange rates
    df["spot_nok_kwh"] = eur_mwh_to_nok_kwh(df["spot_eur_mwh"], df["eur_to_nok"])

    # 5) Proxy-volum splitt
    df["vol_np_kwh"] = df["volume_kwh"] * df["share_np"]
    df["vol_rest_kwh"] = df["volume_kwh"] * (1.0 - df["share_np"])

    # 6) Prisregler
    # Norgespris: capped at 40 øre/kWg 
    norgespris_cap_nok_per_kwh = (NORGESPRIS_CAP_EUR_PER_MWH / 1000.0) * df["eur_to_nok"]
    df["price_np_nok_kwh"] = df["spot_nok_kwh"].clip(upper=norgespris_cap_nok_per_kwh)

    # Strømstøtte: 90% over 77 øre/kWh - ONLY for household consumption group
    df["support_nok_kwh"] = SUPPORT_RATE * (df["spot_nok_kwh"] - SUPPORT_THRESHOLD_NOK_PER_KWH).clip(lower=0.0)
    # Support only applies to household - set to 0 for other consumption groups
    df.loc[df["cons_group"] != "household", "support_nok_kwh"] = 0.0
    
    df["price_rest_nok_kwh"] = df["spot_nok_kwh"] - df["support_nok_kwh"]

    # 7) Kost for kunder
    df["cost_np_nok"] = df["vol_np_kwh"] * df["price_np_nok_kwh"]
    df["cost_rest_nok"] = df["vol_rest_kwh"] * df["price_rest_nok_kwh"]
    df["support_nok"] = df["vol_rest_kwh"] * df["support_nok_kwh"]

    # 8) State cost calculations (Norgespris is SYMMETRICAL)
    # Norgespris gain/loss: difference between spot and cap (can be negative = gain for state)
    # (spot - cap) * volume: positive = loss for state, negative = gain for state
    df["np_gain_loss_nok"] = df["vol_np_kwh"] * (df["spot_nok_kwh"] - norgespris_cap_nok_per_kwh)

    # VAT loss on Norgespris: when spot > cap, state loses VAT on the difference
    # Lost VAT = (spot - cap) * volume * VAT_RATE (only when spot > cap)
    df["np_vat_loss_nok"] = (
        df["vol_np_kwh"] *
        (df["spot_nok_kwh"] - norgespris_cap_nok_per_kwh).clip(lower=0.0) *
        VAT_RATE
    )

    # VAT loss on strømstøtte: state loses VAT on the support amount
    df["support_vat_loss_nok"] = df["support_nok"] * VAT_RATE

    # Total state cost: norgespris gain/loss + strømstøtte + VAT losses
    df["total_state_cost_nok"] = (
        df["np_gain_loss_nok"] +  # Norgespris subsidy (can be negative = income)
        df["support_nok"] +        # Strømstøtte
        df["np_vat_loss_nok"] +    # Lost VAT on Norgespris
        df["support_vat_loss_nok"] # Lost VAT on strømstøtte
    )

    return df


if __name__ == "__main__":
    # Eksempel: startdato til "i dag"
    start = pd.Timestamp("2025-10-01 00:00:00", tz="Europe/Oslo")
    end = pd.Timestamp(date.today().isoformat() + " 23:00:00", tz="Europe/Oslo")

    inputs = Inputs(start=start, end=end)

    out = build_cost_proxy(inputs)

    # Daglig aggregat (klart for dashboard)
    daily = (
        out.groupby(["date", "price_area"], as_index=False)
        .agg(
            {
                "volume_kwh": "sum",
                "vol_np_kwh": "sum",
                "vol_rest_kwh": "sum",
                "cost_np_nok": "sum",
                "cost_rest_nok": "sum",
                "support_nok": "sum",
                "np_gain_loss_nok": "sum",
                "np_vat_loss_nok": "sum",
                "support_vat_loss_nok": "sum",
                "total_state_cost_nok": "sum",
            }
        )
    )

    print(daily.tail(10))
    print(f"\nTotal state cost: {daily['total_state_cost_nok'].sum() / 1e9:.2f} mrd NOK")

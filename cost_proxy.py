from __future__ import annotations

from operator import ge
import os
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlencode

from arrow import get
import pandas as pd
import requests
from entsoe import EntsoePandasClient

import streamlit as st

# Optional for local dev only
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def get_secret(name: str) -> str:
    # Prefer Streamlit secrets in cloud
    if name in st.secrets:
        return st.secrets[name]
    # Fall back to environment variables (incl. .env locally)
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing secret: {name}")
    return val


# -----------------------------
# Konfig
# -----------------------------
EUR_TO_NOK = 11.5
NORGESPRIS_CAP_EUR_PER_MWH = 400.0 / EUR_TO_NOK  # ~34.78 EUR/MWh to get 40 øre/kWh
NORGESPRIS_CAP_NOK_PER_KWH = (NORGESPRIS_CAP_EUR_PER_MWH / 1000.0) * EUR_TO_NOK  # 0.40 NOK/kWh (40 øre/kWh)
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
# Elhub helpers
# -----------------------------
def _month_chunks(start: pd.Timestamp, end: pd.Timestamp):
    """Yield (chunk_start, chunk_end) pairs of at most ~30 days each."""
    current = start
    while current < end:
        chunk_end = min(current + pd.DateOffset(days=30), end)
        yield current, chunk_end
        current = chunk_end


def fetch_elhub_consumption(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch hourly household and cabin consumption per price area from Elhub Energy Data API.
    Returns DataFrame with columns: start_time, price_area, cons_group, volume_kwh
    Note: API has ~1 month max range, so we chunk requests.
    """
    rows = []
    for chunk_start, chunk_end in _month_chunks(start, end):
        for pa in PRICE_AREAS:
            for cons_group in CONSUMPTION_GROUPS:
                params = {
                    "dataset": "CONSUMPTION_PER_GROUP_MBA_HOUR",
                    "startDate": chunk_start.isoformat(),
                    "endDate": chunk_end.isoformat(),
                    "consumptionGroup": cons_group,
                }
                url = f"{ELHUB_API_BASE}/price-areas/{pa}?{urlencode(params)}"
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                j = r.json()
                for item in j["data"]:
                    for rec in item["attributes"].get("consumptionPerGroupMbaHour", []):
                        rows.append({
                            "start_time": pd.Timestamp(rec["startTime"]),
                            "price_area": rec["priceArea"],
                            "cons_group": cons_group,
                            "volume_kwh": rec["quantityKwh"],
                        })
    return pd.DataFrame(rows)


def fetch_elhub_norgespris(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch daily Norgespris share per price area and consumption group from Elhub Energy Data API.
    Returns DataFrame with columns: date, price_area, cons_group, norgespris_count, total_count, share_np
    Note: API has ~1 year max range for this dataset.
    """
    rows = []
    for pa in PRICE_AREAS:
        for cons_group in CONSUMPTION_GROUPS:
            params = {
                "dataset": "NORGESPRIS_CONSUMPTION_PER_GROUP_EAC_MBA",
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "consumptionGroup": cons_group,
                "granularity": "DAILY",
            }
            url = f"{ELHUB_API_BASE}/price-areas/{pa}?{urlencode(params)}"
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            j = r.json()
            for item in j["data"]:
                for rec in item["attributes"].get("norgesprisConsumptionPerGroupEacMba", []):
                    rows.append({
                        "date": pd.Timestamp(rec["startTime"]).date(),
                        "price_area": pa,
                        "cons_group": cons_group,
                        "norgespris_count": rec.get("meteringPointCountNorwayPrice", 0),
                        "total_count": rec.get("totalMeteringPointCount", 0),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["date", "price_area", "cons_group", "norgespris_count", "total_count", "share_np"])
        return df
    # Aggregate EAC buckets per day/area/cons_group
    df = df.groupby(["date", "price_area", "cons_group"], as_index=False).agg({
        "norgespris_count": "sum",
        "total_count": "sum",
    })
    df["share_np"] = 0.0
    mask = df["total_count"] > 0
    df.loc[mask, "share_np"] = (df.loc[mask, "norgespris_count"] / df.loc[mask, "total_count"]).clip(0, 1)
    return df


# -----------------------------
# Spot price fetch (ENTSO-E API)
# -----------------------------
def fetch_spot_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch day-ahead spot prices from ENTSO-E API for all Norwegian price areas.
    Returns DataFrame with columns: start_time, price_area, spot_eur_mwh
    """
    client = _get_entsoe_client()
    rows = []

    # entsoe-py requires timezone-aware timestamps
    # Convert to Europe/Oslo timezone if not already
    if start.tz is None:
        start = start.tz_localize("Europe/Oslo")
    else:
        start = start.tz_convert("Europe/Oslo")
    
    if end.tz is None:
        end = end.tz_localize("Europe/Oslo")
    else:
        end = end.tz_convert("Europe/Oslo")

    for price_area in PRICE_AREAS:
        try:
            # entsoe-py uses area codes like 'NO_1', 'NO_2', etc.
            zone_code = ENTSOE_ZONE_MAP[price_area]
            
            # query_day_ahead_prices returns a pandas Series with datetime index
            prices = client.query_day_ahead_prices(zone_code, start=start, end=end)
            
            if prices is not None and not prices.empty:
                for timestamp, price in prices.items():
                    rows.append({
                        "start_time": timestamp,
                        "price_area": price_area,
                        "spot_eur_mwh": float(price),
                    })
        except Exception as e:
            print(f"Failed to fetch ENTSO-E prices for {price_area}: {e}")
            continue

    if not rows:
        return pd.DataFrame(columns=["start_time", "price_area", "spot_eur_mwh"])

    df = pd.DataFrame(rows)
    
    # Ensure timezone is Europe/Oslo
    if df["start_time"].dt.tz is None:
        df["start_time"] = pd.to_datetime(df["start_time"]).dt.tz_localize("Europe/Oslo")
    else:
        df["start_time"] = pd.to_datetime(df["start_time"]).dt.tz_convert("Europe/Oslo")

    # ENTSO-E may return 15-min or hourly data; resample to hourly (mean)
    df = df.groupby(
        [pd.Grouper(key="start_time", freq="h"), "price_area"],
        as_index=False,
    ).agg({"spot_eur_mwh": "mean"})

    return df[["start_time", "price_area", "spot_eur_mwh"]]


def eur_mwh_to_nok_kwh(price_eur_per_mwh: pd.Series) -> pd.Series:
    return (price_eur_per_mwh / 1000.0) * EUR_TO_NOK


# -----------------------------
# Core: bygg timesdatasett med kost
# -----------------------------
def build_cost_proxy(inputs: Inputs) -> pd.DataFrame:
    # 1) Elhub: forbruk per time (via Energy Data API) - household + cabin
    cons = fetch_elhub_consumption(inputs.start, inputs.end)

    if cons.empty:
        return pd.DataFrame(columns=[
            "start_time", "price_area", "cons_group", "volume_kwh", "date", "spot_eur_mwh", "spot_nok_kwh",
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
            "start_time", "price_area", "cons_group", "volume_kwh", "date", "spot_eur_mwh", "spot_nok_kwh",
            "share_np", "norgespris_count", "total_count", "vol_np_kwh", "vol_rest_kwh",
            "price_np_nok_kwh", "support_nok_kwh", "price_rest_nok_kwh",
            "cost_np_nok", "cost_rest_nok", "support_nok",
            "np_gain_loss_nok", "np_vat_loss_nok", "support_vat_loss_nok", "total_state_cost_nok"
        ])

    # 2) Elhub: norgespris-andel per dag (via Energy Data API)
    np_cnt = fetch_elhub_norgespris(inputs.start, inputs.end)

    # 3) Spotpriser per område
    spot = fetch_spot_prices(inputs.start, inputs.end)
    spot["spot_nok_kwh"] = eur_mwh_to_nok_kwh(spot["spot_eur_mwh"])

    # 4) Slå sammen: timeforbruk + spot + norgespris-andel (per dag/cons_group)
    cons["date"] = cons["start_time"].dt.date
    df = cons.merge(spot, on=["start_time", "price_area"], how="left")
    df = df.merge(
        np_cnt[["date", "price_area", "cons_group", "share_np", "norgespris_count", "total_count"]],
        on=["date", "price_area", "cons_group"],
        how="left"
    )
    df["share_np"] = df["share_np"].fillna(0.0)
    df["norgespris_count"] = df["norgespris_count"].fillna(0)
    df["total_count"] = df["total_count"].fillna(0)

    # 5) Proxy-volum splitt
    df["vol_np_kwh"] = df["volume_kwh"] * df["share_np"]
    df["vol_rest_kwh"] = df["volume_kwh"] * (1.0 - df["share_np"])

    # 6) Prisregler
    # Norgespris: capped at 40 EUR/MWh (0.46 NOK/kWh)
    df["price_np_nok_kwh"] = df["spot_nok_kwh"].clip(upper=NORGESPRIS_CAP_NOK_PER_KWH)

    # Strømstøtte: 90% over 0.77 NOK/kWh
    df["support_nok_kwh"] = SUPPORT_RATE * (df["spot_nok_kwh"] - SUPPORT_THRESHOLD_NOK_PER_KWH).clip(lower=0.0)
    df["price_rest_nok_kwh"] = df["spot_nok_kwh"] - df["support_nok_kwh"]

    # 7) Kost for kunder
    df["cost_np_nok"] = df["vol_np_kwh"] * df["price_np_nok_kwh"]
    df["cost_rest_nok"] = df["vol_rest_kwh"] * df["price_rest_nok_kwh"]
    df["support_nok"] = df["vol_rest_kwh"] * df["support_nok_kwh"]

    # 8) State cost calculations (Norgespris is SYMMETRICAL)
    # Norgespris gain/loss: difference between spot and cap (can be negative = gain for state)
    # (spot - cap) * volume: positive = loss for state, negative = gain for state
    df["np_gain_loss_nok"] = df["vol_np_kwh"] * (df["spot_nok_kwh"] - NORGESPRIS_CAP_NOK_PER_KWH)

    # VAT loss on Norgespris: when spot > cap, state loses VAT on the difference
    # Lost VAT = (spot - cap) * volume * VAT_RATE (only when spot > cap)
    df["np_vat_loss_nok"] = (
        df["vol_np_kwh"] *
        (df["spot_nok_kwh"] - NORGESPRIS_CAP_NOK_PER_KWH).clip(lower=0.0) *
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

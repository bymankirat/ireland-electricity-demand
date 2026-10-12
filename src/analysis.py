"""Descriptive analysis: growth, peaks, daily patterns, wind and carbon."""

from __future__ import annotations

import numpy as np
import pandas as pd

from holidays import holiday_flags


def yearly(h: pd.DataFrame) -> pd.DataFrame:
    d = h["demand_mw"].dropna()
    g = d.groupby(d.index.year)
    hours = g.size()
    out = pd.DataFrame({
        "Average demand (MW)": g.mean(),
        "Peak hour (MW)": g.max(),
        "Electricity used (TWh)": g.sum() / 1e6,
        "Hours of data": hours,
    })
    w = h[["demand_mw", "wind_mw"]].dropna()
    gw = w.groupby(w.index.year)
    out["Wind share (%)"] = 100 * gw["wind_mw"].sum() / gw["demand_mw"].sum()
    c = h[["co2_g_kwh", "demand_mw"]].dropna()
    gc = c.groupby(c.index.year)
    out["Carbon intensity (g/kWh)"] = (gc.apply(lambda x: np.average(x["co2_g_kwh"], weights=x["demand_mw"])))
    full = [y for y in out.index if 8700 <= out.loc[y, "Hours of data"] <= 8800]
    out["Complete year"] = out.index.isin(full)
    return out


def headline(h: pd.DataFrame, yr: pd.DataFrame) -> dict:
    full = yr[yr["Complete year"]]
    first, last = full.index.min(), full.index.max()
    d = h["demand_mw"].dropna()
    peak_t = d.idxmax()
    daily_peak_hour = d.groupby(d.index.normalize()).idxmax().dt.hour
    winter = daily_peak_hour[daily_peak_hour.index.month.isin([11, 12, 1, 2])]
    last12 = h.loc[h.index >= h.index.max() - pd.DateOffset(years=1)]
    return {
        "first_full_year": int(first), "last_full_year": int(last),
        "growth_pct": 100 * (full.loc[last, "Electricity used (TWh)"] / full.loc[first, "Electricity used (TWh)"] - 1),
        "twh_last": float(full.loc[last, "Electricity used (TWh)"]),
        "twh_first": float(full.loc[first, "Electricity used (TWh)"]),
        "peak_mw": float(d.max()), "peak_time": peak_t,
        "winter_peak_17_19_pct": float(100 * winter.between(17, 19).mean()) if len(winter) else float("nan"),
        "wind_share_last12": float(100 * last12["wind_mw"].sum() / last12["demand_mw"].sum()),
        "hours_wind_over_half_last12": float(100 * ((last12["wind_mw"] / last12["demand_mw"]) >= 0.5).mean()),
        "data_start": h.index.min(), "data_end": d.index.max(),
    }


def profiles(h: pd.DataFrame) -> pd.DataFrame:
    """Average demand by hour for typical day types, last complete 3 years."""
    d = h[["demand_mw"]].dropna()
    d = d[d.index >= d.index.max() - pd.DateOffset(years=3)]
    flags = holiday_flags(d.index)
    wd = (d.index.dayofweek < 5) & (flags["is_holiday"].to_numpy() == 0)
    groups = {
        "Winter weekday (Dec-Feb)": wd & d.index.month.isin([12, 1, 2]),
        "Summer weekday (Jun-Aug)": wd & d.index.month.isin([6, 7, 8]),
        "Sunday or holiday (all year)": (d.index.dayofweek == 6) | (flags["is_holiday"].to_numpy() == 1),
    }
    return pd.DataFrame({k: d.loc[m, "demand_mw"].groupby(d.index[m].hour).mean() for k, m in groups.items()})


def wind_carbon(h: pd.DataFrame) -> dict:
    x = h[["demand_mw", "wind_mw", "co2_g_kwh"]].dropna()
    x = x[x.index >= x.index.max() - pd.DateOffset(years=2)]
    share = (100 * x["wind_mw"] / x["demand_mw"]).clip(0, 100)
    bins = list(range(0, 81, 10)) + [101]
    labels = [f"{b}-{b + 10}%" for b in bins[:-2]] + ["80%+"]
    cut = pd.cut(share, bins=bins, labels=labels, right=False)
    g = x["co2_g_kwh"].groupby(cut, observed=True)
    table = pd.DataFrame({"median": g.median(), "p25": g.quantile(0.25), "p75": g.quantile(0.75),
                          "hours": g.size()})
    slope, intercept = np.polyfit(share, x["co2_g_kwh"], 1)
    r = float(np.corrcoef(share, x["co2_g_kwh"])[0, 1])
    return {"table": table, "slope_per_10pp": float(10 * slope), "intercept": float(intercept), "r": r,
            "low_wind_co2": float(x.loc[share <= share.quantile(0.1), "co2_g_kwh"].median()),
            "high_wind_co2": float(x.loc[share >= share.quantile(0.9), "co2_g_kwh"].median()),
            "period_start": x.index.min(), "period_end": x.index.max()}

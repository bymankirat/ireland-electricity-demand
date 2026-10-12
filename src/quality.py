"""Quality control before any analysis. A FAIL stops the pipeline."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from config import REPORTS_DIR, START

RANGES = {"demand_mw": (1500, 8000), "wind_mw": (0, 6000), "co2_g_kwh": (0, 1000), "temp_c": (-20, 40)}
NAMES = {"demand_mw": "Demand", "wind_mw": "Wind", "co2_g_kwh": "Carbon intensity", "temp_c": "Temperature"}


def _longest_flat(s: pd.Series) -> int:
    v = s.dropna().to_numpy()
    if len(v) < 2:
        return 0
    change = np.r_[True, v[1:] != v[:-1]]
    runs = np.diff(np.r_[np.flatnonzero(change), len(v)])
    return int(runs.max())


def run_checks(h: pd.DataFrame, info: dict) -> dict:
    res: list[tuple[str, str, str]] = []
    add = lambda s, c, d: res.append((s, c, d))  # noqa: E731
    expected = pd.date_range(START, h.index.max().floor("D") + pd.Timedelta(hours=23), freq="h")

    for col in ["demand_mw", "wind_mw", "co2_g_kwh"]:
        cov = 100 * h[col].reindex(expected).notna().mean()
        fail = col == "demand_mw" and cov < 90
        add("FAIL" if fail else "PASS" if cov >= 98 else "WARN", f"{NAMES[col]}: hourly coverage since {START[:4]}",
            f"{cov:.1f}% of hours have data")

    raw = info["raw"]["demand"]
    add("PASS", "Clock changes handled",
        f"{raw['duplicates']} repeated 15-minute times (autumn clock change) averaged; spring gaps filled")
    add("PASS" if info["demand_missing_after_fill"] == 0 else "WARN", "Demand gaps longer than 3 hours",
        f"{info['demand_gaps_hours']} hours missing before filling; {info['demand_missing_after_fill']} still "
        f"missing (left empty, not invented)")

    for col, (lo, hi) in RANGES.items():
        if col not in h:
            continue
        s = h[col].dropna()
        bad = int(((s < lo) | (s > hi)).sum())
        add("FAIL" if col == "demand_mw" and bad > 24 else "WARN" if bad else "PASS",
            f"{NAMES[col]}: values in plausible range ({lo} to {hi})", f"{bad} hours outside")

    flat = _longest_flat(h["demand_mw"])
    add("WARN" if flat > 6 else "PASS", "Demand never stuck at one value", f"longest identical run: {flat} hours")
    jumps = h["demand_mw"].diff().abs()
    big = jumps[jumps > 800]
    add("WARN" if len(big) else "PASS", "No hour-to-hour demand jump above 800 MW",
        f"{len(big)} jumps" + (f", e.g. {', '.join(f'{d:%d %b %Y %H:%M}' for d in big.index[:3])}" if len(big) else ""))

    age = (pd.Timestamp(datetime.now()) - h["demand_mw"].dropna().index.max()).days
    add("WARN" if age > 7 else "PASS", "Data is recent", f"latest hour {h['demand_mw'].dropna().index.max():%d %b %Y %H:%M} ({age} days ago)")

    if info.get("has_temperature"):
        cov = 100 * h["temp_c"].notna().mean()
        add("PASS" if cov >= 98 else "WARN", "Temperature matched to demand hours", f"{cov:.1f}% matched")
    else:
        add("WARN", "Temperature available", "not available: weather-based models skipped")

    n_fail = sum(s == "FAIL" for s, *_ in res)
    n_warn = sum(s == "WARN" for s, *_ in res)
    summary = (f"QC {'FAILED' if n_fail else 'PASSED'}: {len(res) - n_fail - n_warn} passed, "
               f"{n_warn} warnings, {n_fail} failures")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "qc_log.txt").write_text(
        "\n".join([f"Quality control log - {datetime.now():%Y-%m-%d %H:%M}", summary, ""]
                  + [f"[{s}] {c}: {d}" for s, c, d in res]) + "\n")
    return {"passed": n_fail == 0, "summary": summary, "results": res}

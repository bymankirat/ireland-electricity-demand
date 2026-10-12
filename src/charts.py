"""Charts. Each title states the finding."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import FIG_DIR, INTERVAL  # noqa: E402
from forecast import FEATURE_LABELS, MODEL_NAMES  # noqa: E402

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a95"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10.5, "font.family": "DejaVu Sans",
    "legend.frameon": False,
})


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def _sub(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=10, va="bottom")


def growth(yr: pd.DataFrame, hl: dict) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [BLUE if c else "#9fc3ee" for c in yr["Complete year"]]
    bars = ax.bar(yr.index.astype(str), yr["Electricity used (TWh)"], color=colors, width=0.65)
    for b, v, c in zip(bars, yr["Electricity used (TWh)"], yr["Complete year"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}" + ("" if c else "*"), ha="center",
                fontsize=9.5, color=INK_2)
    ax.set_ylabel("Electricity demand (TWh)")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Demand grew {hl['growth_pct']:.0f}% from {hl['first_full_year']} to {hl['last_full_year']}", pad=26)
    _sub(ax, "Republic of Ireland system demand per year (*partial year, not comparable)")
    return _save(fig, "01_growth.png")


def daily_profiles(prof: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for (name, s), c in zip(prof.items(), [BLUE, ORANGE, AQUA]):
        ax.plot(s.index, s.values, color=c, lw=2.2)
        ax.text(23.4, s.iloc[-1], name, color=c, fontsize=9.5, va="center", fontweight="bold")
    w = prof.iloc[:, 0]
    ax.annotate(f"Winter peak {w.max():,.0f} MW at {w.idxmax()}:00", (w.idxmax(), w.max()), xytext=(-62, 4),
                textcoords="offset points", ha="right", va="center", fontsize=9.5, color=INK_2,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
    ax.set_ylim(None, w.max() * 1.03)
    ax.set_xlim(0, 30.5)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_ylabel("Average demand (MW)")
    ax.set_title("Demand peaks on winter weekday evenings", pad=26)
    _sub(ax, "Average demand by hour of day, last 3 years")
    return _save(fig, "02_daily_profile.png")


def wind_co2(wc: dict) -> str:
    t = wc["table"]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(t))
    ax.vlines(x, t["p25"], t["p75"], color=AQUA, lw=6, alpha=0.35)
    ax.plot(x, t["median"], "o-", color=AQUA, lw=2, ms=7)
    for xi, v in zip(x, t["median"]):
        ax.text(xi + 0.12, v, f"{v:.0f}", va="bottom", fontsize=9, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(t.index, fontsize=9.5)
    ax.set_xlabel("Share of demand met by wind in that hour")
    ax.set_ylabel("Carbon intensity (gCO2/kWh)")
    ax.set_ylim(0, None)
    ax.set_title(f"Each extra 10% of wind cuts carbon by about {abs(wc['slope_per_10pp']):.0f} g/kWh", pad=26)
    _sub(ax, "Median carbon intensity per hour (dot) and middle 50% (bar), last 2 years")
    return _save(fig, "03_wind_carbon.png")


def accuracy(fc: dict) -> str:
    order = ["last_week", "yesterday", "linear", "gbm"]
    vals = [fc["test"][k]["mape"] for k in order]
    colors = [GREY, GREY] + [BLUE if k == fc["best_kind"] else "#9fc3ee" for k in order[2:]]
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    bars = ax.barh([MODEL_NAMES[k] for k in order], vals, color=colors, height=0.6)
    for b, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}%", va="center", fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Average hourly error (MAPE, %) on the test period: lower is better")
    title = (f"The chosen model cuts day-ahead error by {fc['skill_pct']:.0f}% versus the standard benchmark"
             if fc["skill_pct"] > 0 else "The standard benchmark was not beaten on the test period")
    ax.set_title(title, pad=26)
    _sub(ax, f"Forecasts made at midnight for the next 24 hours, {fc['test_start']:%b %Y} to "
             f"{fc['test_end']:%b %Y} (never used to build or choose the model)")
    return _save(fig, "04_accuracy.png")


def example_week(fc: dict) -> str:
    o = fc["out"]
    mondays = o.index[(o.index.dayofweek == 0) & (o.index.hour == 0)]
    start = None
    for m in mondays[::-1]:
        if len(o.loc[m: m + pd.Timedelta(hours=167)]) == 168:
            start = m
            break
    w = o.loc[start: start + pd.Timedelta(hours=167)] if start is not None else o.iloc[-168:]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.fill_between(w.index, w["lo"], w["hi"], color=BLUE, alpha=0.15, lw=0)
    ax.plot(w.index, w["actual"], color=INK, lw=1.8)
    ax.plot(w.index, w["forecast"], color=BLUE, lw=1.8, ls="--")
    ax.text(w.index[3], w["actual"].max() * 1.005, "Actual (solid) and forecast (dashed)", color=INK_2, fontsize=9.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.set_ylabel("Demand (MW)")
    mape = w["ape"].mean()
    ax.set_title(f"A recent week: forecasts track the daily pattern within {mape:.1f}% on average", pad=26)
    _sub(ax, f"Day-ahead forecast with {INTERVAL:.0%} prediction interval (shaded)")
    return _save(fig, "05_example_week.png")


def drivers(fc: dict) -> str | None:
    imp = fc["importance"]
    if not imp:
        return None
    s = pd.Series(imp).clip(lower=0).sort_values().tail(8)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh([FEATURE_LABELS.get(k, k) for k in s.index], s.values, color=BLUE, height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Increase in average error (MW) when this input is scrambled")
    top = FEATURE_LABELS.get(s.index[-1], s.index[-1])
    ax.set_title(f"{top} is the most important input", pad=26)
    _sub(ax, "Permutation importance, gradient boosting model, test period")
    return _save(fig, "06_drivers.png")

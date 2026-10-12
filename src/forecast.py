"""Day-ahead electricity demand forecasting, tested honestly.

Set-up: at midnight, forecast each hour of the coming day. Only information available at
midnight is used (demand up to the previous hour, calendar, holidays, temperature).

Design:
  * train <= 2023 (first COVID-19 lockdown left out), choose the model on 2024,
  * retrain on <= 2024, then judge ONCE on 2025 onwards,
  * compare with the standard benchmark "same hour last week",
  * 80% prediction intervals from the model's own past errors: fixed (validation year only) and adaptive
   (recalibrated daily on the previous 90 days of known errors); coverage checked on the test period.

Temperature: the models use the temperature actually observed. A real control room would use a
weather forecast, so results with temperature are a best case; results without it are shown too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import COVID_EXCLUDE, INTERVAL, RANDOM_STATE, TEST_START, TRAIN_END, VALID
from holidays import holiday_flags

DAY_TYPES = {0: "Monday", 1: "Tuesday-Friday", 2: "Saturday", 3: "Sunday or holiday"}
MODEL_NAMES = {
    "last_week": "Same hour last week (benchmark)",
    "yesterday": "Same hour yesterday",
    "linear": "Linear regression",
    "gbm": "Gradient boosting",
}
FEATURE_LABELS = {
    "lag24": "Demand same hour yesterday", "lag168": "Demand same hour last week",
    "yday_mean": "Yesterday's average demand", "yday_max": "Yesterday's peak demand",
    "hour": "Hour of day", "dow": "Day of week", "day_type": "Day type", "is_holiday": "Public holiday",
    "christmas_period": "Christmas period", "doy_sin": "Time of year (sin)", "doy_cos": "Time of year (cos)",
    "temp": "Temperature", "hdd": "Heating need (below 15.5°C)", "temp_yday": "Yesterday's temperature",
    "yday_holiday": "Yesterday was a holiday",
}


def make_features(h: pd.DataFrame) -> pd.DataFrame:
    idx = h.index
    y = h["demand_mw"]
    f = pd.DataFrame(index=idx)
    f["hour"] = idx.hour
    f["dow"] = idx.dayofweek
    flags = holiday_flags(idx)
    f["is_holiday"] = flags["is_holiday"].to_numpy()
    f["christmas_period"] = flags["christmas_period"].to_numpy()
    f["day_type"] = np.select([f["is_holiday"] == 1, f["dow"] == 6, f["dow"] == 5, f["dow"] == 0],
                              [3, 3, 2, 0], default=1)
    doy = idx.dayofyear.to_numpy()
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["lag24"] = y.shift(24)
    f["lag168"] = y.shift(168)
    day = idx.normalize()
    f["yday_mean"] = y.groupby(day).transform("mean").shift(24)
    f["yday_max"] = y.groupby(day).transform("max").shift(24)
    f["yday_holiday"] = f["is_holiday"].shift(24)
    if "temp_c" in h:
        f["temp"] = h["temp_c"]
        f["hdd"] = np.maximum(0, 15.5 - h["temp_c"])
        f["temp_yday"] = h["temp_c"].groupby(day).transform("mean").shift(24)
    f["y"] = y
    return f


def _cols(use_temp: bool) -> list[str]:
    base = ["hour", "dow", "day_type", "is_holiday", "christmas_period", "doy_sin", "doy_cos",
            "lag24", "lag168", "yday_mean", "yday_max", "yday_holiday"]
    return base + (["temp", "hdd", "temp_yday"] if use_temp else [])


def _gbm():
    return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0,
                                         random_state=RANDOM_STATE)


def _linear(cols):
    num = [c for c in cols if c not in ("hour", "dow", "day_type")]
    ct = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["hour_daytype", "dow"]),
        ("num", StandardScaler(), num),
    ])
    return make_pipeline(ct, Ridge(alpha=1.0))


def _lin_frame(X):
    X = X.copy()
    X["hour_daytype"] = X["hour"].astype(str) + "_" + X["day_type"].astype(str)
    return X


def _fit_predict(kind, cols, tr, te):
    if kind == "last_week":
        return te["lag168"].to_numpy()
    if kind == "yesterday":
        return te["lag24"].to_numpy()
    if kind == "linear":
        m = _linear(cols).fit(_lin_frame(tr[cols]), tr["y"])
        return m.predict(_lin_frame(te[cols]))
    m = _gbm().fit(tr[cols], tr["y"])
    return m.predict(te[cols])


def _metrics(y, p, index) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    df = pd.DataFrame({"y": y, "p": p}, index=index)
    daily = df.groupby(index.normalize()).max()
    return {
        "mape": float(100 * np.mean(np.abs(y - p) / y)),
        "mae": float(np.mean(np.abs(y - p))),
        "peak_mae": float(np.mean(np.abs(daily["y"] - daily["p"]))),
        "peak_mape": float(100 * np.mean(np.abs(daily["y"] - daily["p"]) / daily["y"])),
    }


def run(h: pd.DataFrame, has_temp: bool) -> dict:
    f = make_features(h)
    covid = (f.index >= COVID_EXCLUDE[0]) & (f.index <= pd.Timestamp(COVID_EXCLUDE[1]) + pd.Timedelta(hours=23))
    variants = [True, False] if has_temp else [False]

    covid_idx = f.index[covid]

    def frame(lo, hi, cols, drop_covid=False):
        d = f.loc[lo:hi]
        if drop_covid:
            d = d[~d.index.isin(covid_idx)]
        return d.dropna(subset=cols + ["y"])

    # ---- 1. choose the model on the validation year
    val_scores, val_resid = {}, {}
    for use_temp in variants:
        cols = _cols(use_temp)
        tr = frame(None, TRAIN_END, cols, drop_covid=True)
        va = frame(VALID[0], VALID[1], cols)
        for kind in ["linear", "gbm"]:
            p = _fit_predict(kind, cols, tr, va)
            val_scores[(kind, use_temp)] = _metrics(va["y"], p, va.index)["mape"]
            val_resid[(kind, use_temp)] = pd.Series(va["y"].to_numpy() - p, index=va.index)
    best_kind, best_temp = min(val_scores, key=val_scores.get)

    # ---- 2. retrain on everything up to the end of validation; judge once on the test period
    cols = _cols(best_temp)
    tr = frame(None, VALID[1], cols, drop_covid=True)
    te = frame(TEST_START, None, cols)
    preds = {k: _fit_predict(k, cols, tr, te) for k in ["last_week", "yesterday", "linear", "gbm"]}
    test = {k: _metrics(te["y"], p, te.index) for k, p in preds.items()}

    other_temp = None
    if has_temp:  # the same model with the other temperature setting, for transparency
        c2 = _cols(not best_temp)
        tr2, te2 = frame(None, VALID[1], c2, drop_covid=True), frame(TEST_START, None, c2)
        other_temp = _metrics(te2["y"], _fit_predict(best_kind, c2, tr2, te2), te2.index)

    # Intervals: the chosen model's own errors on the validation year, by hour of day
    # (split-conformal style), added to the test forecasts. Nothing from the test period is used.
    lo_q, hi_q = (1 - INTERVAL) / 2, 1 - (1 - INTERVAL) / 2
    r = val_resid[(best_kind, best_temp)]
    q = r.groupby(r.index.hour).quantile([lo_q, hi_q]).unstack()
    best = preds[best_kind]
    hrs = te.index.hour
    lo = best + q[lo_q].reindex(hrs).to_numpy()
    hi = best + q[hi_q].reindex(hrs).to_numpy()
    y_te = te["y"].to_numpy()
    fixed_cov = float(np.mean((y_te >= lo) & (y_te <= hi)))
    fixed_width = float(np.mean(hi - lo))

    # Adaptive intervals: each day, recalibrate from the model's errors over the previous 90 days, all of which
    # are known by then (validation-year errors at first, then the deployed model's own test errors).
    # This keeps the intervals honest when demand grows and the model drifts.
    past = pd.concat([r, pd.Series(y_te - best, index=te.index)]).sort_index()
    past = past[~past.index.duplicated(keep="last")]
    days = te.index.normalize()
    alo, ahi = np.empty(len(te)), np.empty(len(te))
    for d in days.unique():
        win = past.loc[d - pd.Timedelta(days=90): d - pd.Timedelta(hours=1)]
        qa = win.groupby(win.index.hour).quantile([lo_q, hi_q]).unstack()
        qa = qa.reindex(range(24)).fillna(q)
        m = days == d
        alo[m] = best[m] + qa[lo_q].reindex(te.index[m].hour).to_numpy()
        ahi[m] = best[m] + qa[hi_q].reindex(te.index[m].hour).to_numpy()
    coverage = float(np.mean((y_te >= alo) & (y_te <= ahi)))
    width = float(np.mean(ahi - alo))
    lo, hi = alo, ahi

    out = pd.DataFrame({"actual": te["y"], "forecast": best, "lo": lo, "hi": hi,
                        "benchmark": preds["last_week"], "day_type": te["day_type"],
                        "is_holiday": te["is_holiday"]}, index=te.index)
    out["ape"] = 100 * (out["actual"] - out["forecast"]).abs() / out["actual"]
    out["ape_bench"] = 100 * (out["actual"] - out["benchmark"]).abs() / out["actual"]

    by_type = out.groupby("day_type")[["ape", "ape_bench"]].mean().rename(index=DAY_TYPES)
    daily = out.groupby(out.index.normalize()).agg(ape=("ape", "mean"), holiday=("is_holiday", "max"))
    worst = daily.sort_values("ape", ascending=False).head(6)

    imp = {}
    if True:  # what drives the gradient boosting model (shown even if the linear model wins)
        m = _gbm().fit(tr[cols], tr["y"])
        sample = te.sample(min(6000, len(te)), random_state=RANDOM_STATE)
        pi = permutation_importance(m, sample[cols], sample["y"], n_repeats=3,
                                    random_state=RANDOM_STATE, scoring="neg_mean_absolute_error")
        imp = dict(sorted(zip(cols, pi.importances_mean), key=lambda kv: -kv[1]))

    skill = 100 * (1 - test[best_kind]["mape"] / test["last_week"]["mape"])
    return {
        "val_scores": val_scores, "best_kind": best_kind, "best_temp": best_temp,
        "best_name": MODEL_NAMES[best_kind] + (" with temperature" if best_temp else ""),
        "test": test, "other_temp": other_temp, "skill_pct": skill,
        "coverage": coverage, "interval_width": width, "fixed_coverage": fixed_cov, "fixed_width": fixed_width,
        "by_type": by_type, "worst_days": worst, "importance": imp, "out": out,
        "test_start": te.index.min(), "test_end": te.index.max(), "train_rows": int(len(tr)),
    }

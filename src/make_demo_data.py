"""FAKE data in the exact EirGrid and Met Eireann formats, for testing without internet.

Never publish results made from these files: every output is marked DEMO.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import RAW_DIR, START

DEMO_DIR = RAW_DIR.parent / "demo"
DEMO_GRID = DEMO_DIR / "eirgrid"
DEMO_MET = DEMO_DIR / "met_hly532.csv"


def make(end: str = "2026-10-09 23:45", seed: int = 5) -> None:
    rng = np.random.default_rng(seed)
    t = pd.date_range(START, end, freq="15min")
    n = len(t)
    years = (t - t[0]).days / 365.25
    hour = t.hour + t.minute / 60
    doy = t.dayofyear.values
    dow = t.dayofweek.values

    # temperature: annual + daily cycle + weather noise
    temp = (10 - 5 * np.cos(2 * np.pi * (doy - 20) / 365) + 2.5 * np.sin(2 * np.pi * (hour - 9) / 24))
    w = np.zeros(n)
    for i in range(1, n):
        w[i] = 0.995 * w[i - 1] + rng.normal(0, 0.12)
    temp = temp + w

    profile = (0.80 + 0.12 * np.exp(-((hour - 18.0) ** 2) / 6) + 0.10 / (1 + np.exp(-(hour - 7.5) * 2))
               - 0.10 / (1 + np.exp(-(hour - 22.5) * 2)))
    weekend = np.where(dow == 6, 0.88, np.where(dow == 5, 0.92, 1.0))
    demand = (4300 * (1 + 0.03 * years) * profile * weekend
              + 35 * np.maximum(0, 14 - temp) + rng.normal(0, 40, n))
    covid = (t >= "2020-03-28") & (t <= "2020-05-18")
    demand = np.where(covid, demand * 0.88, demand)

    cap = 3600 + 250 * years
    wf = np.zeros(n)
    for i in range(1, n):
        wf[i] = 0.998 * wf[i - 1] + rng.normal(0, 0.03)
    wind = np.clip(cap * (0.32 + 0.25 * np.tanh(wf)), 0, None)
    share = np.clip(wind / demand, 0, 1)
    co2 = np.clip(430 - 330 * share + rng.normal(0, 15, n), 40, None)

    DEMO_GRID.mkdir(parents=True, exist_ok=True)
    fmt = t.strftime("%d-%b-%Y %H:%M:%S")
    for key, name, vals in [("demand", "SYSTEM_DEMAND", demand), ("wind", "WIND_ACTUAL", wind),
                            ("co2", "CO2_INTENSITY", co2)]:
        df = pd.DataFrame({"EffectiveTime": fmt, "FieldName": name, "Region": "ROI",
                           "Value": np.round(vals).astype(int)})
        for m, part in df.groupby(t.strftime("%Y-%m")):
            part.to_csv(DEMO_GRID / f"{key}_{m}.csv", index=False)

    # Met Eireann hourly file (UTC), with the usual notes at the top
    th = pd.date_range(pd.Timestamp(START) - pd.Timedelta(days=2), end, freq="h")
    tl = pd.Series(temp[::4][: len(th)], index=t[::4][: len(th)])
    tt = tl.reindex(th, method="nearest").to_numpy()
    with DEMO_MET.open("w", encoding="latin-1") as f:
        f.write("Station Name: DEMO DATA - NOT REAL\nStation Height: 71 M\n\n")
        f.write("date:  -  Date and Time (utc)\ntemp:  -  Air Temperature (C)\n\n")
        f.write("date,ind,rain,ind,temp,ind,wetb,dewpt\n")
        for d, v in zip(th, tt):
            f.write(f"{d.strftime('%d-%b-%Y %H:%M').lower()},0,0.0,0,{v:.1f},0,{v - 1:.1f},{v - 2:.1f}\n")
    print(f"Demo files written to {DEMO_DIR}")


if __name__ == "__main__":
    make()

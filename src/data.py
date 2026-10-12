"""Download EirGrid and Met Eireann data, then build one clean hourly table.

EirGrid times are Irish local time; Met Eireann hourly times are UTC. Both are put on Irish
local time. At the autumn clock change the repeated hour is averaged; at the spring change the
missing hour is filled from its neighbours.
"""

from __future__ import annotations

import io
import json
import subprocess
import time
import urllib.request
from pathlib import Path

import pandas as pd

from config import AREAS, GRID_API, GRID_DIR, MET_FILE, MET_URL, REGION, START


# --------------------------------------------------------------------------- download
def _fetch(url: str, timeout: int = 120) -> bytes:  # timeout: seconds allowed for the whole download
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:  # noqa: BLE001  some Python installs lack SSL certificates; curl usually works
        return subprocess.run(["curl", "-sSL", "--max-time", str(timeout), url],
                              capture_output=True, check=True).stdout


def _grid_rows(area: str, start: pd.Timestamp, end: pd.Timestamp, tries: int = 5) -> list[dict]:
    """One request, retried with growing pauses: the dashboard sometimes answers with an empty
    page when asked too quickly."""
    url = GRID_API.format(area=area, region=REGION, start=f"{start:%d-%b-%Y}+00:00",
                          end=f"{end:%d-%b-%Y}+23:59")
    for attempt in range(tries):
        try:
            payload = json.loads(_fetch(url))
            time.sleep(0.5)  # be polite to the server
            if payload.get("Status") == "Error" and "no data" in str(payload.get("ErrorMessage", "")).lower():
                return []
            return payload.get("Rows") or []
        except (ValueError, OSError, subprocess.CalledProcessError):
            wait = 3 * (attempt + 1) ** 2  # 3, 12, 27, 48 seconds
            print(f"       (EirGrid busy, retrying in {wait}s ...)")
            time.sleep(wait)
    raise RuntimeError(f"no answer for {area} {start:%d %b %Y}")


def download_grid(folder: Path = GRID_DIR, refresh_recent: bool = True) -> list[str]:
    """One cached CSV per series per month. Completed months are downloaded once."""
    folder.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().normalize()
    months = pd.date_range(START, today, freq="MS")
    total = len(months) * len(AREAS)
    done, failed = 0, []
    for key, area in AREAS.items():
        for m in months:
            done += 1
            f = folder / f"{key}_{m:%Y-%m}.csv"
            recent = m >= (today - pd.DateOffset(months=1)).replace(day=1)
            if f.exists() and not (recent and refresh_recent):
                continue
            end = min(m + pd.offsets.MonthEnd(0), today - pd.Timedelta(days=1))
            if end < m:
                continue
            expected = ((end - m).days + 1) * 96
            try:
                rows = _grid_rows(area, m, end)
                if len(rows) < 0.9 * expected:  # some servers cap long requests: retry week by week
                    rows = []
                    for ws in pd.date_range(m, end, freq="7D"):
                        rows += _grid_rows(area, ws, min(ws + pd.Timedelta(days=6), end))
            except RuntimeError as e:
                failed.append(f"{key} {m:%b %Y}")
                print(f"     {done}/{total}  {key} {m:%b %Y}: skipped ({e}); run again later to fill it")
                continue
            pd.DataFrame(rows).to_csv(f, index=False)
            print(f"     {done}/{total}  {key} {m:%b %Y}: {len(rows):,} readings")
    return failed


def download_met(force: bool = False) -> Path | None:
    if MET_FILE.exists() and not force:
        return MET_FILE
    print("     Downloading Met Éireann hourly weather (large file, can take a few minutes) ...")
    try:
        data = _fetch(MET_URL, timeout=1500)
        if b"date," not in data[:20000].lower():
            raise ValueError("unexpected content")
        MET_FILE.parent.mkdir(parents=True, exist_ok=True)
        MET_FILE.write_bytes(data)
        return MET_FILE
    except Exception as e:  # noqa: BLE001
        print(f"     [!] Temperature download failed ({e}). Models will run without temperature.")
        print(f"         To add it: open {MET_URL} in Chrome, save the file as\n         {MET_FILE}\n"
              f"         and run 'python run_pipeline.py' again.")
        return None


# --------------------------------------------------------------------------- load
def _local_hourly(s: pd.Series) -> pd.Series:
    """Average to hourly, merge the repeated autumn hour, fill the missing spring hour."""
    s = s.groupby(s.index.floor("h")).mean()
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    return s.reindex(full)


def load_grid(folder: Path = GRID_DIR) -> tuple[pd.DataFrame, dict]:
    raw_counts, series = {}, {}
    for key in AREAS:
        files = sorted(folder.glob(f"{key}_*.csv"))
        parts = [pd.read_csv(f) for f in files if f.stat().st_size > 5]
        parts = [p for p in parts if len(p)]
        if not parts:
            raise FileNotFoundError(f"No {key} data in {folder}. Run without --demo to download it.")
        df = pd.concat(parts, ignore_index=True)
        t = pd.to_datetime(df["EffectiveTime"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
        v = pd.to_numeric(df["Value"], errors="coerce")
        s = pd.Series(v.values, index=t).dropna()
        s = s[s.index >= START]
        raw_counts[key] = {"readings": int(len(s)), "duplicates": int(s.index.duplicated().sum()),
                           "first": s.index.min(), "last": s.index.max()}
        series[key] = _local_hourly(s)
    hourly = pd.DataFrame(series)
    hourly.columns = ["demand_mw", "wind_mw", "co2_g_kwh"]
    return hourly, raw_counts


def load_met(path: Path | None) -> pd.Series | None:
    if path is None or not Path(path).exists():
        return None
    text = Path(path).read_text(encoding="latin-1")
    lines = text.splitlines()
    hdr = next(i for i, ln in enumerate(lines) if ln.lower().replace('"', "").startswith("date,"))
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), dtype=str, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    t = pd.to_datetime(df["date"].str.strip(), format="%d-%b-%Y %H:%M", errors="coerce")
    temp = pd.to_numeric(df["temp"].str.strip(), errors="coerce")
    s = pd.Series(temp.values, index=t).dropna()
    s = s[s.index >= pd.Timestamp(START) - pd.Timedelta(days=2)]
    local = s.index.tz_localize("UTC").tz_convert("Europe/Dublin").tz_localize(None)
    s = pd.Series(s.values, index=local)
    return _local_hourly(s).rename("temp_c")


def build_table(grid_folder: Path = GRID_DIR, met_path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    hourly, counts = load_grid(grid_folder)
    temp = load_met(met_path)
    gaps_before = int(hourly["demand_mw"].isna().sum())
    hourly = hourly.interpolate(limit=3, limit_area="inside")  # fill gaps up to 3 hours only
    if temp is not None:
        hourly = hourly.join(temp.interpolate(limit=3, limit_area="inside"), how="left")
    info = {"raw": counts, "demand_gaps_hours": gaps_before,
            "demand_missing_after_fill": int(hourly["demand_mw"].isna().sum()),
            "has_temperature": temp is not None and hourly.get("temp_c", pd.Series()).notna().mean() > 0.9}
    return hourly, info

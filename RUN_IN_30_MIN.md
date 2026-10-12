# Finish this project in 30 minutes

## 1. Run it (10 min, mostly waiting)

Unzip, then in Terminal:

```
cd /Users/mankiratsingh/MSc_Thesis/THESIS_PAPER_DATA_PDF/03_Gut_Microbiota_Dysbiosis/ireland-electricity-demand
python run_pipeline.py
open docs/index.html
```

The first run downloads about 7 years of 15-minute grid data, month by month (you'll see progress),
plus Met Éireann temperature: 5-10 minutes. If it stops part-way, just run it again: months already
downloaded are kept.

Check the report: no orange DEMO banner, QC PASSED. Paste the RESULTS block to Claude to check.

## 2. GitHub (15 min)

1. github.com → **+ → New repository** → `ireland-electricity-demand`, **Public**, no README.
2. **uploading an existing file** → in **Chrome**, drag in `docs`, `reports`, `src` and `README.md`,
   `requirements.txt`, `run_pipeline.py`, `RUN_IN_30_MIN.md`. Do **not** upload `data`.
3. **Commit**, then **Settings → Pages → main → /docs → Save**.

## 3. Keep it current

Once a month: `python run_pipeline.py --refresh`, then upload the new `docs/index.html`.

## Interview talking points (ESB, EirGrid, energy companies)

- **Why "same hour last week" as the benchmark?** It's the standard naive forecast for electricity
  demand: it captures the daily and weekly cycle. A model is only useful if it beats it.
- **How did you avoid leakage?** Forecasts are issued at midnight; every input is known by then.
  Train ≤2023, choose on 2024, test once on 2025+.
- **Why leave out the first lockdown?** Demand behaved abnormally; training on it would teach the wrong pattern.
- **Clock changes?** EirGrid is Irish local time and Met Éireann is UTC; I aligned both, averaged the
  repeated autumn hour and filled the missing spring hour.
- **Holidays?** Computed from the official rules, including weekend substitutions and St Brigid's Day
  from 2023. Holidays and Christmas are where forecasts are hardest.
- **Temperature caveat:** I used observed temperature, so the "with temperature" result is a best case;
  a real system would use a weather forecast. I report both.
- **Are the intervals honest?** Built from 2024 errors, checked on 2025+. If coverage is below 80%,
  that's partly demand growth, and I say so.
- **Next steps:** compare against EirGrid's own forecasts, use weather forecasts, model data-centre
  load separately, add wind-power forecasting.

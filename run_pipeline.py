"""Irish electricity demand: download -> QC -> analysis -> day-ahead forecasts -> report.

    python run_pipeline.py            # downloads EirGrid + Met Eireann data (first run ~5-10 minutes)
    python run_pipeline.py --demo     # simulated data for testing (outputs marked DEMO)
    python run_pipeline.py --refresh  # re-download the last two months (run monthly to stay current)
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import analysis  # noqa: E402
import charts  # noqa: E402
import forecast  # noqa: E402
import quality  # noqa: E402
import report  # noqa: E402
from config import DOCS_DIR, GRID_DIR, REPORTS_DIR  # noqa: E402
from data import build_table, download_grid, download_met  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    print("1/6  Getting data ...")
    if args.demo:
        from make_demo_data import DEMO_GRID, DEMO_MET, make
        make()
        grid, met = DEMO_GRID, DEMO_MET
    else:
        try:
            failed = download_grid(refresh_recent=True)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"\n[!] EirGrid download failed: {e}\n    Check your internet connection and run again "
                     f"(months already downloaded are kept).")
        if failed:
            print(f"\n     [!] {len(failed)} month(s) could not be downloaded this time: {', '.join(failed[:6])}"
                  f"{' ...' if len(failed) > 6 else ''}\n         Run 'python run_pipeline.py' again to fill them "
                  f"(downloaded months are kept).\n")
        grid, met = GRID_DIR, download_met(force=args.refresh)

    print("2/6  Building the hourly table and checking quality ...")
    h, info = build_table(grid, met)
    qc = quality.run_checks(h, info)
    print(f"     {qc['summary']}  (details: reports/qc_log.txt)")
    for s, c, d in qc["results"]:
        if s != "PASS":
            print(f"       [{s}] {c}: {d}")
    if not qc["passed"]:
        sys.exit("\n[!] Quality checks failed, so no report was produced. See reports/qc_log.txt.")
    has_temp = bool(info["has_temperature"])

    print("3/6  Analysing demand, wind and carbon ...")
    yr = analysis.yearly(h)
    hl = analysis.headline(h, yr)
    prof = analysis.profiles(h)
    wc = analysis.wind_carbon(h)

    print("4/6  Training and testing day-ahead forecasts (about a minute) ...")
    fc = forecast.run(h, has_temp)

    print("5/6  Charts ...")
    figs = {"growth": charts.growth(yr, hl), "profile": charts.daily_profiles(prof), "wind": charts.wind_co2(wc),
            "accuracy": charts.accuracy(fc), "week": charts.example_week(fc), "drivers": charts.drivers(fc)}

    print("6/6  Report ...")
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "index.html").write_text(report.build(yr, hl, prof, wc, fc, qc, figs, has_temp, args.demo),
                                         encoding="utf-8")
    REPORTS_DIR.mkdir(exist_ok=True)
    yr.to_csv(REPORTS_DIR / "yearly.csv", float_format="%.2f")
    fc["out"].to_csv(REPORTS_DIR / "test_forecasts.csv", float_format="%.1f")
    (REPORTS_DIR / "results.json").write_text(json.dumps({
        "demo": args.demo, "headline": hl, "wind_co2_slope_per_10pp": wc["slope_per_10pp"],
        "chosen": fc["best_name"], "test": fc["test"], "skill_vs_benchmark_pct": fc["skill_pct"],
        "interval_coverage_adaptive": fc["coverage"], "interval_coverage_fixed": fc["fixed_coverage"],
        "validation": {f"{k}_{'temp' if t else 'notemp'}": v for (k, t), v in fc["val_scores"].items()},
    }, indent=2, default=str))

    bt, lw = fc["test"][fc["best_kind"]], fc["test"]["last_week"]
    test_months = round((fc["test_end"] - fc["test_start"]).days / 30.4)
    cv = (f"**Irish Electricity Demand Forecasting (Python, pandas, scikit-learn):** automated pipeline on "
          f"{hl['data_start']:%Y}-{hl['data_end']:%Y} EirGrid grid data (hourly demand, wind, CO2) and Met Éireann "
          f"weather, with automatic quality checks; day-ahead forecasts accurate to {bt['mape']:.1f}% per hour on "
          f"{test_months} months of unseen data ({fc['skill_pct']:.0f}% better than the industry-standard "
          f"same-hour-last-week benchmark), with adaptive 80% prediction intervals ({100 * fc['coverage']:.0f}% coverage on unseen data); quantified demand "
          f"growth ({hl['growth_pct']:.0f}% since {hl['first_full_year']}) and the wind-carbon link for decision-makers.")
    (REPORTS_DIR / "summary.md").write_text(("> DEMO DATA - do not publish.\n\n" if args.demo else "") + cv + "\n")

    print("\n================ RESULTS ================" + ("  (DEMO DATA - NOT REAL)" if args.demo else ""))
    print(f"Data: {hl['data_start']:%d %b %Y} to {hl['data_end']:%d %b %Y}   temperature: {'yes' if has_temp else 'no'}")
    print(f"Demand {hl['first_full_year']}->{hl['last_full_year']}: {hl['twh_first']:.1f} -> {hl['twh_last']:.1f} TWh "
          f"({hl['growth_pct']:+.0f}%)   record hour {hl['peak_mw']:,.0f} MW ({hl['peak_time']:%d %b %Y %H:%M})")
    print(f"Wind share last 12 months: {hl['wind_share_last12']:.0f}%   CO2 per +10pp wind: {wc['slope_per_10pp']:+.0f} g/kWh")
    print(f"Chosen model: {fc['best_name']}")
    print(f"Test MAPE: chosen {bt['mape']:.2f}% vs benchmark {lw['mape']:.2f}%  (skill {fc['skill_pct']:+.0f}%)   "
          f"daily peak error {bt['peak_mae']:.0f} MW")
    print(f"80% interval coverage on test: adaptive {100 * fc['coverage']:.0f}% (width {fc['interval_width']:.0f} MW), "
          f"fixed {100 * fc['fixed_coverage']:.0f}% (width {fc['fixed_width']:.0f} MW)")
    print(f"\nReport: {DOCS_DIR / 'index.html'}\nCV line: {REPORTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()

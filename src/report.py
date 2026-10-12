"""Self-contained web report, written for an electricity company audience."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd

from config import INTERVAL, TEST_START, VALID
from forecast import MODEL_NAMES

AUTHOR = "Mankirat Singh"
LINKS = {"Portfolio": "https://bymankirat.github.io/myportfolio/", "GitHub": "https://github.com/bymankirat",
         "LinkedIn": "https://linkedin.com/in/mankirat-singh-694440177"}


def _img(p, alt):
    if not p:
        return ""
    return f'<img src="data:image/png;base64,{base64.b64encode(Path(p).read_bytes()).decode()}" alt="{html.escape(alt)}">'


def build(yr, hl, prof, wc, fc, qc, figs, has_temp, demo) -> str:
    t = fc["test"]
    best = fc["best_kind"]
    bt, lw = t[best], t["last_week"]
    winter = prof.iloc[:, 0]
    skill_txt = (f"a {fc['skill_pct']:.0f}% improvement on" if fc["skill_pct"] > 0 else "compared with")

    findings = [
        f"Electricity demand in the Republic of Ireland grew <b>{hl['growth_pct']:.0f}%</b> from "
        f"{hl['first_full_year']} to {hl['last_full_year']} ({hl['twh_first']:.1f} to {hl['twh_last']:.1f} TWh).",
        f"Demand peaks on <b>winter weekday evenings</b>: on an average winter weekday it reaches "
        f"{winter.max():,.0f} MW around {winter.idxmax()}:00. The highest hour on record here was "
        f"{hl['peak_mw']:,.0f} MW on {hl['peak_time']:%d %B %Y}.",
        f"Wind met <b>{hl['wind_share_last12']:.0f}%</b> of demand over the last 12 months, and at least half of "
        f"demand in {hl['hours_wind_over_half_last12']:.0f}% of hours.",
        f"Wind and carbon move together: each extra 10 points of wind share goes with about "
        f"<b>{abs(wc['slope_per_10pp']):.0f} g/kWh</b> less CO2 (median {wc['low_wind_co2']:.0f} g/kWh in the "
        f"least windy tenth of hours against {wc['high_wind_co2']:.0f} g/kWh in the windiest tenth).",
        f"A day-ahead forecast issued at midnight is accurate to <b>{bt['mape']:.2f}%</b> per hour on unseen data, "
        f"{skill_txt} the standard benchmark ({lw['mape']:.2f}%). "
        f"The daily peak is forecast within {bt['peak_mae']:.0f} MW on average.",
    ]

    yr_rows = "".join(
        f"<tr><td>{y}{'' if r['Complete year'] else '*'}</td><td class='num'>{r['Electricity used (TWh)']:.1f}</td>"
        f"<td class='num'>{r['Average demand (MW)']:,.0f}</td><td class='num'>{r['Peak hour (MW)']:,.0f}</td>"
        f"<td class='num'>{r['Wind share (%)']:.0f}</td><td class='num'>{r['Carbon intensity (g/kWh)']:.0f}</td></tr>"
        for y, r in yr.iterrows())
    acc_rows = "".join(
        f"<tr{' class=chosen' if k == best else ''}><td>{MODEL_NAMES[k]}{' <b>(chosen)</b>' if k == best else ''}</td>"
        f"<td class='num'>{t[k]['mape']:.2f}</td><td class='num'>{t[k]['mae']:.0f}</td>"
        f"<td class='num'>{t[k]['peak_mae']:.0f}</td></tr>" for k in ["last_week", "yesterday", "linear", "gbm"])
    val_rows = "".join(
        f"<tr><td>{MODEL_NAMES[k]}, {'with' if tmp else 'without'} temperature</td><td class='num'>{v:.2f}</td></tr>"
        for (k, tmp), v in sorted(fc["val_scores"].items(), key=lambda kv: kv[1]))
    type_rows = "".join(f"<tr><td>{d}</td><td class='num'>{r['ape']:.2f}</td><td class='num'>{r['ape_bench']:.2f}</td></tr>"
                        for d, r in fc["by_type"].iterrows())
    worst_rows = "".join(f"<tr><td>{d:%A %d %B %Y}</td><td class='num'>{r['ape']:.1f}</td>"
                         f"<td>{'Public holiday' if r['holiday'] else ''}</td></tr>" for d, r in fc["worst_days"].iterrows())
    qc_rows = "".join(f"<tr><td class='qc-{s.lower()}'>{s}</td><td>{html.escape(c)}</td><td>{html.escape(d)}</td></tr>"
                      for s, c, d in qc["results"])

    temp_note = ""
    if has_temp and fc["other_temp"] is not None:
        with_t = bt["mape"] if fc["best_temp"] else fc["other_temp"]["mape"]
        without_t = fc["other_temp"]["mape"] if fc["best_temp"] else bt["mape"]
        temp_note = (f"<p>Temperature matters: the same model scores {with_t:.2f}% with temperature and "
                     f"{without_t:.2f}% without. The models use the temperature actually observed at Dublin "
                     f"Airport, so the 'with temperature' result is a best case. In a control room a weather "
                     f"forecast would be used, and accuracy would sit between the two.</p>")

    banner = ('<div class="banner">DEMO DATA: simulated numbers for testing. Run without --demo to download the '
              'real EirGrid and Met Éireann data before publishing.</div>') if demo else ""
    links = " · ".join(f'<a href="{u}">{k}</a>' for k, u in LINKS.items())
    data_to = f"{hl['data_end']:%d %B %Y}"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Irish Electricity Demand</title>
<style>
:root{{--bg:#fcfcfb;--card:#fff;--ink:#0b0b0b;--ink-2:#52514e;--line:#e4e3df;--accent:#2a78d6;--hl:#eef4fc;--warn-bg:#fff4e5;--warn-ink:#7a3d00}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--card:#232322;--ink:#f0efec;--ink-2:#c3c2b7;--line:#3a3a38;--accent:#3987e5;--hl:#1f2a3a;--warn-bg:#3a2a12;--warn-ink:#ffd08a}}:root:not([data-theme="light"]) img{{background:#fcfcfb}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--card:#232322;--ink:#f0efec;--ink-2:#c3c2b7;--line:#3a3a38;--accent:#3987e5;--hl:#1f2a3a;--warn-bg:#3a2a12;--warn-ink:#ffd08a}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
main{{max-width:920px;margin:0 auto;padding:40px 16px 64px}}
.kicker{{color:var(--accent);font-weight:700;letter-spacing:.04em;text-transform:uppercase;font-size:.82rem}}
h1{{font-size:clamp(1.6rem,4vw,2.3rem);line-height:1.2;margin:6px 0 8px}}h2{{font-size:1.3rem;margin:44px 0 8px}}
.byline,.muted{{color:var(--ink-2)}}a{{color:var(--accent)}}
.banner{{background:var(--warn-bg);color:var(--warn-ink);padding:12px 16px;border-radius:8px;font-weight:600;margin:16px 0}}
.note{{border-left:3px solid var(--accent);padding:6px 12px;color:var(--ink-2);font-size:.92rem;margin:14px 0}}
.findings{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 20px 16px 36px;margin:20px 0}}
.findings li{{margin:7px 0}}
img{{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;margin:12px 0}}
.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:.93rem}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}}th{{color:var(--ink-2);font-weight:600}}
.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}tr.chosen{{background:var(--hl)}}
td.qc-pass{{color:#008300;font-weight:700}}td.qc-warn{{color:#b06f00;font-weight:700}}td.qc-fail{{color:#e34948;font-weight:700}}
footer{{margin-top:52px;padding-top:16px;border-top:1px solid var(--line);color:var(--ink-2);font-size:.9rem}}
</style></head><body><main>
{banner}
<div class="kicker">Independent analysis of EirGrid data</div>
<h1>Irish Electricity Demand: Growth, Wind and Day-Ahead Forecasting</h1>
<p class="byline">{AUTHOR} · {links}</p>
<p class="note">Republic of Ireland, hourly, {hl['data_start']:%B %Y} to {data_to}. Data: EirGrid Smart Grid Dashboard
(system demand, wind generation, CO2 intensity) and Met Éireann (Dublin Airport temperature). Independent analysis,
not an EirGrid or ESB publication.</p>

<h2>Key findings</h2>
<ul class="findings">{''.join(f'<li>{f}</li>' for f in findings)}</ul>

<h2>1. Demand is growing</h2>
{_img(figs.get('growth'), 'Annual electricity demand')}
<div class="table-wrap"><table><thead><tr><th>Year</th><th class="num">Demand (TWh)</th><th class="num">Average (MW)</th>
<th class="num">Peak hour (MW)</th><th class="num">Wind share (%)</th><th class="num">Carbon (g/kWh)</th></tr></thead>
<tbody>{yr_rows}</tbody></table></div>
<p class="muted">* Partial year. Carbon intensity is demand-weighted.</p>

<h2>2. When demand peaks</h2>
<p>The system has to be built and operated for the peak, not the average. Winter weekday evenings, when lighting,
heating and cooking coincide, are the hardest hours.</p>
{_img(figs.get('profile'), 'Average daily demand profile')}

<h2>3. Wind and carbon</h2>
<p>Over {wc['period_start']:%B %Y} to {wc['period_end']:%B %Y}, the share of demand met by wind explains most of the
hour-to-hour variation in carbon intensity (correlation {wc['r']:.2f}).</p>
{_img(figs.get('wind'), 'Carbon intensity by wind share')}

<h2>4. Forecasting tomorrow's demand</h2>
<p>Each night at midnight, the model forecasts every hour of the coming day, using only information available then:
recent demand, the calendar, Irish public holidays{', and temperature' if has_temp else ''}. It was trained on data up
to 2023 (leaving out the first COVID-19 lockdown), chosen on {VALID[0][:4]}, and then tested once on
{pd.Timestamp(TEST_START):%B %Y} onwards, a period never used to build or choose it.</p>
{_img(figs.get('accuracy'), 'Forecast accuracy by model')}
<div class="table-wrap"><table><thead><tr><th>Test period</th><th class="num">Hourly error (MAPE %)</th>
<th class="num">Hourly error (MW)</th><th class="num">Daily peak error (MW)</th></tr></thead><tbody>{acc_rows}</tbody></table></div>
{temp_note}
{_img(figs.get('week'), 'Example week of forecasts')}
<p><b>Honest uncertainty.</b> {INTERVAL:.0%} prediction intervals built once from the {VALID[0][:4]} errors contained
the actual demand {100 * fc['fixed_coverage']:.0f}% of the time on the test period{(': too often wrong, because demand kept growing and the model was forecasting further from its training data, so its errors grew' if fc['fixed_coverage'] < INTERVAL - 0.03 else '')}.
Recalibrating the intervals every day from the previous 90 days of known errors (adaptive conformal prediction)
gives <b>{100 * fc['coverage']:.0f}%</b> coverage (target {INTERVAL:.0%}), with an average width of {fc['interval_width']:.0f} MW
against {fc['fixed_width']:.0f} MW. The shaded bands in the chart above are the adaptive ones.</p>

<h3>Where forecasts are hardest</h3>
<div class="table-wrap"><table><thead><tr><th>Day type</th><th class="num">Model error (%)</th>
<th class="num">Benchmark error (%)</th></tr></thead><tbody>{type_rows}</tbody></table></div>
<div class="table-wrap"><table><thead><tr><th>Hardest days in the test period</th><th class="num">Error (%)</th><th></th></tr></thead>
<tbody>{worst_rows}</tbody></table></div>
{_img(figs.get('drivers'), 'What drives the forecast')}

<h3>How the model was chosen (validation year {VALID[0][:4]})</h3>
<div class="table-wrap"><table><thead><tr><th>Candidate</th><th class="num">Hourly error (MAPE %)</th></tr></thead>
<tbody>{val_rows}</tbody></table></div>

<h2>5. What this means for an electricity company</h2>
<ul>
<li><b>Planning:</b> steady demand growth and winter evening peaks drive the need for network capacity and flexible
supply such as batteries and demand response.</li>
<li><b>Operations:</b> a transparent day-ahead forecast with honest uncertainty bands supports scheduling and
trading; its errors are largest around holidays and unusual weather, where human judgement should be added.</li>
<li><b>Decarbonisation:</b> carbon intensity falls sharply as wind rises, so shifting flexible demand (for example
EV charging) into windy hours is one of the cheapest ways to cut emissions.</li>
</ul>

<h2>6. Quality checks</h2>
<p>The data is checked automatically before any analysis, and the run stops on a failure. Result: <b>{qc['summary']}</b>.</p>
<div class="table-wrap"><table><thead><tr><th>Status</th><th>Check</th><th>Detail</th></tr></thead><tbody>{qc_rows}</tbody></table></div>

<h2>Limitations and methods</h2>
<ul>
<li>Republic of Ireland only. System demand as published by EirGrid; it does not separate large users such as data centres.</li>
<li>Temperature is observed, not forecast (see section 4). One weather station (Dublin Airport) represents the country.</li>
<li>EirGrid publishes its own demand forecasts; comparing against them is a natural next step.</li>
<li>Methods: Python (pandas, NumPy, scikit-learn, matplotlib). Ridge regression on hour × day-type, lagged demand and
weather; histogram gradient boosting; permutation importance; fixed and adaptive (rolling 90-day) conformal prediction intervals.
Irish public holidays computed from the official rules, including weekend substitutions and St Brigid's Day from 2023.
Code: <a href="https://github.com/bymankirat/ireland-electricity-demand">GitHub</a>.</li>
</ul>
<footer>Data: EirGrid Smart Grid Dashboard; Met Éireann (CC BY 4.0). Analysis and any errors are the author's own.</footer>
</main></body></html>
"""

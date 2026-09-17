# ⚖️ Peshawar High Court — Case Management Dashboard

A Streamlit dashboard analyzing 765,000+ court cases scraped from publicly
published Peshawar High Court cause lists (2017–2026), covering hearing
schedules, bench composition, case categories, and review-flag trends
across 21 courts and 580+ justices.

## What it does

- **Overview** — headline KPIs, an auto-generated executive summary,
  year-over-year growth, a linear trend forecast with anomaly flagging,
  and category/court/day breakdowns.
- **Court Infrastructure Analysis** — per-court leaderboards, lawyer/judge
  density by court, a court workload-imbalance flag, and a searchable
  case-level detail table.
- **Distribution Analysis by Court** — drill-down view for a single court,
  with section-concentration insight and monthly anomaly flagging.
- **Case Distribution by Judge** — drill-down view for a single justice,
  with workload-vs-peer-average comparison and monthly anomaly flagging.
- **Sidebar** — dataset-wide overview (total cases, courts, date coverage,
  last updated) and a single "reset all filters" control that clears
  every tab's filters at once.

## Architecture

```
cause_lists_combined_*.xlsx  →  data_loader.py  →  app1.py (Streamlit)
                                      │
                                      ├─ first run: parse + vectorize
                                      └─ every run after: read .cache.parquet
```

- **`data_loader.py`** — reads the raw Excel export once, does light
  cleaning (name splitting, day normalization, date construction), and
  writes a `.cache.parquet` file next to the source so every subsequent
  run skips Excel parsing entirely. Wrapped in `@st.cache_data` on top of
  that, so a running session doesn't even hit disk again after first load.
- **`app1.py`** — all presentation logic: theming, KPI cards, chart
  styling, filters, sidebar, and the four analysis tabs. No data
  transformation happens here — it consumes the already-clean DataFrame
  from `data_loader.py`.

## A performance case study: vectorizing the date pipeline

The original date-construction step used `df.apply(build_date, axis=1)` —
a Python-level function called once per row that builds a full pandas
`Series` object just to read three values out of it.

**Benchmark on 200,000 synthetic rows** (same logic, same edge cases —
missing year, non-numeric year, lowercase month, filename-fallback dates):

| Approach | Time | 
|---|---|
| `df.apply(..., axis=1)` | 18.06s |
| Vectorized (`pd.to_datetime` on whole columns) | 0.94s |

**~19x faster**, with output verified identical to the original
row-by-row logic (`old_result.equals(new_result) == True`) across every
edge case the original function handled. On the full 765K-row production
dataset, this turns a multi-minute first load into a matter of seconds —
meaningful because it's the one part of the pipeline the parquet cache
*can't* paper over (it still runs in full whenever the source Excel file
changes).

**Why it's faster:** `axis=1` apply can't use pandas' underlying C-level
vectorized operations — it drops into a Python for-loop under the hood.
Rewriting the same logic as whole-column operations (`pd.to_numeric`,
`.str.extract`, `pd.to_datetime` on a DataFrame of year/month/day columns)
lets pandas do the work in optimized batches instead of one row at a time.

## Analytical features

Two reusable helpers (`forecast_monthly`, `flag_anomalous_months`) power
analytical callouts across every tab, not just Overview:

- **3-month linear trend forecast** on monthly case volume (Overview),
  fit with `numpy.polyfit` — intentionally simple (no seasonality model)
  since the goal is a directional "where is this headed" signal in the
  executive summary, not an operational forecast.
- **Anomaly flagging** — months more than 2 standard deviations from the
  historical mean are automatically surfaced, on Overview *and* on
  whichever court or justice is currently selected in the other two
  drill-down tabs — rather than requiring a human to eyeball every bar
  on every trend chart.
- **Workload comparisons** — a court's caseload vs. the average court
  (Infrastructure tab), and a justice's caseload vs. the average justice
  (Judge tab) — surfaced as plain-language callouts, not just left for
  the reader to infer from a bar chart.

## Tech stack

Python · Streamlit · Pandas · NumPy · Plotly · Parquet (caching layer)

## Data source & disclaimer

Data is compiled from publicly available cause lists published by the
Peshawar High Court, parsed and structured for analysis. This is an
independent analytics project, not an official PHC product.

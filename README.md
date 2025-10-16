# Sales Dashboard (Power BI)

This repository contains a robust Python ETL that cleans and enriches raw sales data and produces analytics-ready CSVs for a Power BI dashboard.

## Quick start

From the repo root (`Sales-Dashboard-powerbi/`):

1) Create a virtual environment and install dependencies

```powershell
python -m venv .venv
".venv\Scripts\pip.exe" install -r requirements.txt
```

2) Run the ETL

Use either module invocation or a script path. Both examples assume you're in the repo root.

```powershell
# As a module (recommended)
".venv\Scripts\python.exe" -m scripts.etl_sales_data --input data/raw/data.csv --output data/cleaned/cleaned_data.csv --margin 0.30 --log-level INFO

# Or via script path
".venv\Scripts\python.exe" "scripts/etl_sales_data.py" --input "data/raw/data.csv" --output "data/cleaned/cleaned_data.csv" --margin 0.30 --log-level INFO
```

### CLI flags

- `--input` Path to the raw CSV (default: `data/raw/data.csv`).
- `--output` Path to write the cleaned CSV (default: `data/cleaned/cleaned_data.csv`).
- `--margin` Gross margin rate used for profit calculations (default: `0.30`).
- `--log-level` Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`).

## What the ETL does

- Reads the raw CSV with robust encoding handling (tries UTF-8, UTF-8-SIG, ISO-8859-1/latin1).
- Normalizes column names and trims string fields.
- Parses dates automatically with day-first detection (e.g., `dd/mm/yyyy`).
- Filters out cancellations (InvoiceNo starting with `C`), non-positive quantities/prices, and duplicates.
- Coerces numeric types safely and guards divide-by-zero/infinite values.
- Derives helpful fields: `total_price`, `year`, `month`, `category` (from `StockCode` prefix), `profit`, `profit_margin`, `revenue_per_customer`.
- Writes cleaned data plus multiple summary tables and a reusable Date dimension for modeling.

## Outputs (in `data/cleaned/`)

- `cleaned_data.csv` — Cleaned row-level dataset for detailed visuals.
- `summary_by_region.csv` — Aggregates by `country`, `year`, `month`.
	- Columns: `country, year, month, total_revenue, avg_unit_price, items_sold, unique_customers`.
- `summary_by_category.csv` — Aggregates by derived `category`, `year`, `month`.
	- Columns: `category, year, month, total_revenue, avg_margin, items_sold`.
- `summary_monthly_totals.csv` — Overall monthly totals for trend visuals.
	- Columns: `year, month, total_revenue, avg_order_value, items_sold, unique_customers`.
- `date_dimension.csv` — Date table for building a calendar model in Power BI.
	- Columns: `date, year, quarter, month, month_name, year_month, day, day_of_week, week`.

## Expected input schema

The ETL is designed for a UK e-commerce style dataset with columns similar to:

```
InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
```

Column names are normalized to snake_case internally (e.g., `InvoiceDate` -> `invoice_date`).

## Power BI modeling tips

- Import all CSVs in `data/cleaned/`.
- Create a relationship between `date_dimension[date]` and `cleaned_data[invoice_date]`.
	- In Power BI, set `cleaned_data[invoice_date]` to Data type `Date` (not Date/Time) to match the Date table key exactly.
- Use `summary_monthly_totals.csv` for KPI cards and monthly trend visuals.
- Use `summary_by_region.csv` and `summary_by_category.csv` for matrices or stacked visuals.

## Troubleshooting

- Encoding errors reading the raw CSV: the ETL auto-retries common encodings; set `--log-level DEBUG` to see which one succeeded.
- Date parsing looks wrong: the ETL infers day-first formats automatically; if your data is strictly month-first, confirm the raw format and share a snippet.
- Profit/margin differences: adjust `--margin` to your business assumption (e.g., `--margin 0.35`).

## Notes

- Dependencies are pinned in `requirements.txt` (pandas 2.x).
- Parquet export is not enabled by default. If you want Parquet for faster refresh, we can add it (requires `pyarrow`).


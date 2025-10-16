"""
ETL to clean raw sales data and produce a cleaned dataset plus summary CSVs.

Adds derived fields: year, month, profit, profit_margin, revenue_per_customer.
Also writes grouped summaries by country (region) and stock category (derived).

Run from repo root:
  python -m scripts.etl_sales_data --input data/raw/data.csv --output data/cleaned/cleaned_data.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _setup_logging(level: str) -> None:
    # Configure a module-level logger for cleaner CI output
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))


def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "ISO-8859-1", "latin1"):
        try:
            logger.info("Reading CSV with encoding=%s: %s", enc, path)
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    logger.info("Reading CSV with default encoding: %s", path)
    return pd.read_csv(path, low_memory=False)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.lower()
    )
    return df


def _infer_dayfirst(date_series: pd.Series) -> bool:
    """Heuristically infer whether dates are day-first.

    - Sample up to 1000 values, parse with dayfirst True/False, pick higher success.
    - Tie-breaker: if first token > 12 often appears, prefer day-first.
    """
    s = date_series.dropna().astype(str).head(1000)
    if s.empty:
        return False
    p_mdy = pd.to_datetime(s, errors="coerce", dayfirst=False)
    p_dmy = pd.to_datetime(s, errors="coerce", dayfirst=True)
    mdy_ok = p_mdy.notna().sum()
    dmy_ok = p_dmy.notna().sum()
    if dmy_ok > mdy_ok:
        return True
    if dmy_ok < mdy_ok:
        return False
    # Tie-break via token > 12 indicating day
    try:
        tokens = s.str.extract(r"^(\d{1,2})[\-/]", expand=False)
        if tokens.dropna().astype(int).gt(12).any():
            return True
    except Exception:
        pass
    return False


def _clean_types_and_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Trim strings
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    # Parse dates with dayfirst auto-detection
    date_col = next((c for c in df.columns if c.lower() in {"invoicedate", "invoice_date"}), None)
    if date_col:
        dayfirst = _infer_dayfirst(df[date_col])
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=dayfirst)
        before = len(df)
        df = df[~df[date_col].isna()].copy()
        if len(df) != before:
            logger.info("Dropped %d rows with invalid %s.", before - len(df), date_col)

    # Remove cancellations and non-numeric invoice numbers
    # - Drop rows where InvoiceNo starts with 'C' (explicit cancellations)
    # - Drop rows where InvoiceNo is not strictly numeric (covers values starting with letters like 'A', blanks, NaN)
    inv_col = next((c for c in df.columns if c.lower() in {"invoiceno", "invoice_no"}), None)
    if inv_col:
        s = df[inv_col].astype("string").str.strip()
        before = len(df)
        mask_c = s.str.upper().str.startswith("C").fillna(False)
        is_numeric = s.str.fullmatch(r"\d+").fillna(False)
        mask_non_numeric = ~is_numeric
        # Count separately without double-counting C* rows in the non-numeric bucket
        canc_removed = int(mask_c.sum())
        nonnum_removed = int((mask_non_numeric & ~mask_c).sum())
        combined = mask_c | mask_non_numeric
        df = df[~combined].copy()
        logger.info(
            "Removed %d cancellation rows and %d rows with non-numeric invoice numbers.",
            canc_removed,
            nonnum_removed,
        )

    # Coerce numerics
    qty_col = next((c for c in df.columns if c.lower() == "quantity"), None)
    price_col = next((c for c in df.columns if c.lower() in {"unitprice", "unit_price"}), None)
    cust_col = next((c for c in df.columns if c.lower() in {"customerid", "customer_id"}), None)

    if qty_col:
        df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce")
    if price_col:
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    if cust_col:
        df[cust_col] = pd.to_numeric(df[cust_col], errors="coerce").astype("Int64")

    # Drop non-positive values
    if qty_col:
        before = len(df)
        df = df[(df[qty_col] > 0)].copy()
    logger.info("Removed %d rows with non-positive quantity.", before - len(df))
    if price_col:
        before = len(df)
        df = df[(df[price_col] > 0)].copy()
    logger.info("Removed %d rows with non-positive unit price.", before - len(df))

    # Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates().copy()
    logger.info("Dropped %d duplicate rows.", before - len(df))

    return df


def _add_derived_fields(df: pd.DataFrame, *, margin: float) -> pd.DataFrame:
    df = df.copy()
    # Identify core cols
    qty_col = next((c for c in df.columns if c.lower() == "quantity"), None)
    price_col = next((c for c in df.columns if c.lower() in {"unitprice", "unit_price"}), None)
    date_col = next((c for c in df.columns if c.lower() in {"invoicedate", "invoice_date"}), None)
    cust_col = next((c for c in df.columns if c.lower() in {"customerid", "customer_id"}), None)

    # total_price as float with NaN (not pd.NA), to be safe in arithmetic
    if qty_col and price_col:
        df["total_price"] = (
            pd.to_numeric(df[qty_col], errors="coerce").astype("float64")
            * pd.to_numeric(df[price_col], errors="coerce").astype("float64")
        )
    else:
        df["total_price"] = np.nan

    # year, month
    if date_col:
        df["year"] = df[date_col].dt.year.astype("Int64")
        df["month"] = df[date_col].dt.month.astype("Int64")
    else:
        df["year"] = pd.Series([np.nan] * len(df), dtype="float64")
        df["month"] = pd.Series([np.nan] * len(df), dtype="float64")

    # Derive a naive 'category' from StockCode prefix (first letter block)
    stock_col = next((c for c in df.columns if c.lower() in {"stockcode", "stock_code"}), None)
    if stock_col:
        df["category"] = df[stock_col].astype("string").str.extract(r"^([A-Za-z]+)", expand=False).fillna("MISC")
    else:
        df["category"] = "MISC"

    # Profit and margin with numeric safety
    denom = pd.to_numeric(df["total_price"], errors="coerce").astype("float64")
    df["profit"] = (denom * float(margin)).astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["profit_margin"] = np.where(denom > 0, df["profit"] / denom, np.nan)

    # revenue_per_customer = total revenue / number of rows for that customer in the month
    if cust_col and date_col:
        df["revenue_per_customer"] = df.groupby([cust_col, "year", "month"], dropna=False)["total_price"].transform("mean").astype("float64")
    elif cust_col:
        df["revenue_per_customer"] = df.groupby([cust_col], dropna=False)["total_price"].transform("mean").astype("float64")
    else:
        df["revenue_per_customer"] = np.nan

    return df


def _write_outputs(df: pd.DataFrame, out_csv: Path, write_summaries: bool = True) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing cleaned CSV to %s", out_csv)
    df.to_csv(out_csv, index=False)

    if not write_summaries:
        return

    # Summary by region (Country)
    # Resolve core columns once
    cols = {c.lower(): c for c in df.columns}
    country_col = cols.get("country")
    qty_col = cols.get("quantity")
    price_col = cols.get("unitprice", cols.get("unit_price"))

    if country_col:
        agg_map = {
            "total_revenue": ("total_price", "sum"),
        }
        if price_col:
            agg_map["avg_unit_price"] = (price_col, "mean")
        if qty_col:
            agg_map["items_sold"] = (qty_col, "sum")
        cust_col = cols.get("customerid") or cols.get("customer_id")
        if isinstance(cust_col, str) and cust_col:
            agg_map["unique_customers"] = (cust_col, "nunique")
        by_region = df.groupby([country_col, "year", "month"], dropna=False).agg(**agg_map).reset_index()
        by_region_path = out_csv.parent / "summary_by_region.csv"
        logger.info("Writing region summary to %s", by_region_path)
        by_region.to_csv(by_region_path, index=False)

    # Summary by category
    if "category" in df.columns:
        agg_map = {
            "total_revenue": ("total_price", "sum"),
            "avg_margin": ("profit_margin", "mean"),
        }
        if qty_col:
            agg_map["items_sold"] = (qty_col, "sum")
        by_cat = df.groupby(["category", "year", "month"], dropna=False).agg(**agg_map).reset_index()
        by_cat_path = out_csv.parent / "summary_by_category.csv"
        logger.info("Writing category summary to %s", by_cat_path)
        by_cat.to_csv(by_cat_path, index=False)

    # Monthly totals for trends
    if "year" in df.columns and "month" in df.columns:
        agg_map = {"total_revenue": ("total_price", "sum"), "avg_order_value": ("total_price", "mean")}
        if qty_col:
            agg_map["items_sold"] = (qty_col, "sum")
        cust_col = cols.get("customerid") or cols.get("customer_id")
        if isinstance(cust_col, str) and cust_col:
            agg_map["unique_customers"] = (cust_col, "nunique")
        monthly = (
            df.groupby(["year", "month"], dropna=False).agg(**agg_map).reset_index().sort_values(["year", "month"]) 
        )
        monthly_path = out_csv.parent / "summary_monthly_totals.csv"
        logger.info("Writing monthly totals to %s", monthly_path)
        monthly.to_csv(monthly_path, index=False)

    # Date dimension for Power BI
    date_col = cols.get("invoicedate", cols.get("invoice_date"))
    if date_col is not None and df[date_col].notna().any():
        dmin = pd.to_datetime(df[date_col]).min()
        dmax = pd.to_datetime(df[date_col]).max()
        if pd.notna(dmin) and pd.notna(dmax):
            dates = pd.date_range(dmin.normalize(), dmax.normalize(), freq="D")
            dim = pd.DataFrame(
                {
                    "date": dates,
                    "year": dates.year,
                    "quarter": dates.quarter,
                    "month": dates.month,
                    "month_name": dates.strftime("%B"),
                    "year_month": dates.strftime("%Y-%m"),
                    "day": dates.day,
                    "day_of_week": dates.dayofweek + 1,  # 1=Mon
                    "week": dates.isocalendar().week.astype(int),
                }
            )
            dim_path = out_csv.parent / "date_dimension.csv"
            logger.info("Writing date dimension to %s", dim_path)
            dim.to_csv(dim_path, index=False)


def run_etl(input_path: Path, output_path: Path, *, margin: float) -> pd.DataFrame:
    raw = _read_csv(input_path)
    logger.info("Raw shape: %s", raw.shape)
    df = _normalize_columns(raw)
    df = _clean_types_and_values(df)
    df = _add_derived_fields(df, margin=margin)
    logger.info("Cleaned shape: %s", df.shape)
    _write_outputs(df, output_path, write_summaries=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw sales CSV and save cleaned dataset + summaries.")
    parser.add_argument("--input", type=str, default=str(Path("data") / "raw" / "data.csv"), help="Path to raw CSV input")
    parser.add_argument("--output", type=str, default=str(Path("data") / "cleaned" / "cleaned_data.csv"), help="Path to cleaned CSV output")
    parser.add_argument("--margin", type=float, default=0.30, help="Gross margin rate for profit calculation (e.g., 0.30)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR")

    args = parser.parse_args()
    _setup_logging(args.log_level)

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    run_etl(input_path, output_path, margin=args.margin)
    logger.info("ETL completed successfully.")


if __name__ == "__main__":
    main()

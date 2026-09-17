"""
Run this ONCE, locally, whenever the source Excel file changes.

It builds the .cache.parquet file next to your source file using the
exact same _prepare() logic the app uses, so the deployed app on
Streamlit Cloud never has to parse the raw .xlsx itself.

Usage:
    python build_cache.py path/to/cause_lists_combined.xlsx

After it finishes, you'll have a file like:
    path/to/cause_lists_combined.cache.parquet

Commit that .cache.parquet file to git alongside the .xlsx and push both.
Next deploy, load_data() will find the parquet and skip the expensive
openpyxl parse entirely.

If you ever update the source data, delete the old .cache.parquet file
and re-run this script to rebuild it before committing again.
"""
import sys
import os

# Make sure we can import data_loader.py from this same folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import _prepare
import pandas as pd


def main():
    if len(sys.argv) != 2:
        print("Usage: python build_cache.py path/to/source_file.xlsx")
        sys.exit(1)

    source_path = sys.argv[1]
    if not os.path.exists(source_path):
        print(f"File not found: {source_path}")
        sys.exit(1)

    parquet_path = os.path.splitext(source_path)[0] + ".cache.parquet"

    ext = os.path.splitext(source_path)[1].lower()
    print(f"Reading {source_path} ... (this can take a few minutes for a large file)")
    if ext == ".csv":
        df = pd.read_csv(source_path, dtype=str, encoding="utf-8-sig")
    else:
        df = pd.read_excel(source_path, dtype=str)

    print(f"Read {len(df):,} rows. Preparing / cleaning...")
    df = _prepare(df)

    # Keep only the last few years' worth of data. Update KEEP_YEARS if the
    # retention window ever needs to change.
    KEEP_YEARS = {2024, 2025, 2026}
    before_trim = len(df)
    df = df[df["Hearing_Date"].dt.year.isin(KEEP_YEARS)].reset_index(drop=True)
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].cat.remove_unused_categories()
    print(f"Trimmed {before_trim - len(df):,} row(s) outside {sorted(KEEP_YEARS)} "
          f"-> {len(df):,} rows kept.")

    print(f"Writing {parquet_path} ...")
    df.to_parquet(parquet_path)

    size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"Done. {parquet_path} is {size_mb:.1f} MB.")
    print("Now commit and push this .cache.parquet file alongside the source file.")


if __name__ == "__main__":
    main()

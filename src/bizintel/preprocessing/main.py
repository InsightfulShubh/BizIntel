"""
Data preprocessing pipeline entry point.

Usage:
    python -m bizintel.processing.main
"""

from __future__ import annotations

import pandas as pd

from bizintel.config.settings import (
    YC_CSV_PATHS,
    CRUNCHBASE_CSV_PATH,
    OUTPUT_DIR,
    YC_OUTPUT_FILENAME,
    CRUNCHBASE_OUTPUT_FILENAME,
    UNIFIED_OUTPUT_FILENAME,
)
from bizintel.processing.data_preprocess import run_preprocessing, write_dataset
from bizintel.processing.validation import add_suspicious_flags


def main() -> None:
    # Raw row counts for stats
    yc_raw_count = sum(
        len(pd.read_csv(p, encoding_errors="replace")) for p in YC_CSV_PATHS
    )
    cb_raw_count = len(
        pd.read_csv(CRUNCHBASE_CSV_PATH, encoding_errors="replace")
    )

    yc_df, cb_df, unified_df = run_preprocessing(
        yc_paths=YC_CSV_PATHS,
        crunchbase_path=CRUNCHBASE_CSV_PATH,
        output_dir=OUTPUT_DIR,
    )

    # Apply validation flags
    yc_df = add_suspicious_flags(yc_df)
    cb_df = add_suspicious_flags(cb_df)
    unified_df = add_suspicious_flags(unified_df)

    # Re-write with suspicious column
    write_dataset(yc_df, OUTPUT_DIR / YC_OUTPUT_FILENAME)
    write_dataset(cb_df, OUTPUT_DIR / CRUNCHBASE_OUTPUT_FILENAME)
    write_dataset(unified_df, OUTPUT_DIR / UNIFIED_OUTPUT_FILENAME)

    # ── Stats ─────────────────────────────────────────────────────────
    yc_cleaned = len(yc_df)
    cb_cleaned = len(cb_df)
    yc_drop_pct = (1 - (yc_cleaned / yc_raw_count)) * 100 if yc_raw_count else 0
    cb_drop_pct = (1 - (cb_cleaned / cb_raw_count)) * 100 if cb_raw_count else 0

    print("YC original rows:", yc_raw_count)
    print("YC cleaned rows:", yc_cleaned)
    print("YC drop %:", f"{yc_drop_pct:.2f}%")
    print("YC suspicious rows:", int(yc_df["is_suspicious"].sum()))

    print("Crunchbase original rows:", cb_raw_count)
    print("Crunchbase cleaned rows:", cb_cleaned)
    print("Crunchbase drop %:", f"{cb_drop_pct:.2f}%")
    print("Crunchbase suspicious rows:", int(cb_df["is_suspicious"].sum()))

    print("\nUnified total rows:", len(unified_df))
    print("Unified suspicious rows:", int(unified_df["is_suspicious"].sum()))


if __name__ == "__main__":
    main()

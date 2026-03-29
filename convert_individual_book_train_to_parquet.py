#!/usr/bin/env python3
"""Convert every CSV in individual_book_train to Parquet in individual_book_train_parquet."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "individual_book_train"
OUTPUT_DIR = ROOT / "individual_book_train_parquet"


def main() -> int:
    try:
        import pandas as pd
    except ImportError:
        print("Install dependencies: pip install pandas pyarrow", file=sys.stderr)
        return 1

    if not INPUT_DIR.is_dir():
        print(f"Not a directory: {INPUT_DIR}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_paths:
        print(f"No CSV files found in {INPUT_DIR}", file=sys.stderr)
        return 1

    for csv_path in csv_paths:
        parquet_path = OUTPUT_DIR / csv_path.with_suffix(".parquet").name
        print(f"Converting {csv_path.name} …")
        df = pd.read_csv(csv_path)
        df.to_parquet(parquet_path, engine="pyarrow", index=False)

    print(f"Done. Parquet files written to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent

files = ["stock_ids.csv", "train.csv"]

for filename in files:
    csv_path = data_dir / filename
    parquet_path = csv_path.with_suffix(".parquet")
    df = pd.read_csv(csv_path)
    df.to_parquet(parquet_path, index=False)
    print(f"Converted {filename} -> {parquet_path.name} ({len(df):,} rows)")

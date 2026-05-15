from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig, normalize_stock_id


BOOK_COLUMNS = [
    "time_id",
    "seconds_in_bucket",
    "bid_price1",
    "ask_price1",
    "bid_price2",
    "ask_price2",
    "bid_size1",
    "ask_size1",
    "bid_size2",
    "ask_size2",
    "stock_id",
]


def list_available_stocks(source_dir: str | Path) -> list[str]:
    source = Path(source_dir)
    return sorted(path.stem for path in source.glob("stock_*.parquet"))


def stock_path(stock: str | int, source_dir: str | Path) -> Path:
    stock_name = normalize_stock_id(stock)
    return Path(source_dir) / f"{stock_name}.parquet"


def load_raw_stock(stock: str | int, data_config: DataConfig) -> pd.DataFrame:
    path = stock_path(stock, data_config.source_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing stock parquet: {path}")
    df = pd.read_parquet(path)
    missing = set(BOOK_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if data_config.max_time_ids_per_stock is not None:
        keep = sorted(df["time_id"].dropna().unique())[: data_config.max_time_ids_per_stock]
        df = df[df["time_id"].isin(keep)]
    return df[BOOK_COLUMNS].copy()


def make_equally_spaced(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for time_id, chunk in df.groupby("time_id", sort=False):
        chunk = chunk.sort_values("seconds_in_bucket").drop_duplicates("seconds_in_bucket")
        chunk = chunk.set_index("seconds_in_bucket").reindex(np.arange(600))
        chunk.index.name = "seconds_in_bucket"
        chunk["time_id"] = time_id
        chunk = chunk.ffill().reset_index()
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True)


def add_wap_and_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["time_id", "seconds_in_bucket"]).copy()
    denominator = (
        out["bid_size1"] + out["ask_size1"] + out["bid_size2"] + out["ask_size2"]
    ).replace(0, np.nan)
    out["weighted_price"] = (
        out["bid_price1"] * out["ask_size1"]
        + out["ask_price1"] * out["bid_size1"]
        + out["bid_price2"] * out["ask_size2"]
        + out["ask_price2"] * out["bid_size2"]
    ) / denominator
    out["mid_price"] = (out["bid_price1"] + out["ask_price1"]) / 2
    out["log_price"] = np.log(out["weighted_price"])
    out["log_price_diff"] = out.groupby("time_id", sort=False)["log_price"].diff()
    return out


def preprocess_book_data(df: pd.DataFrame) -> pd.DataFrame:
    return add_wap_and_returns(make_equally_spaced(df))


def load_processed_stock(stock: str | int, data_config: DataConfig) -> pd.DataFrame:
    stock_name = normalize_stock_id(stock)
    return preprocess_book_data(load_raw_stock(stock_name, data_config))

import os
import pandas as pd
import numpy as np


def load_all_stock_paths(parquet_folder):

    parquet_files = [
        os.path.join(parquet_folder, f)
        for f in os.listdir(parquet_folder)
        if f.endswith(".parquet")
    ]

    return sorted(parquet_files)


def load_stock_data(filepath):

    df = pd.read_parquet(filepath)

    return df


def check_duplicates(df):

    dup_check = (
        df.groupby(["time_id", "seconds_in_bucket"])
        .size()
        .reset_index(name="count")
    )

    dup_check = dup_check[dup_check["count"] > 1]

    if len(dup_check) > 0:
        print("Duplicates found!")
    else:
        print("No duplicates found.")


def make_equally_spaced(df):

    all_time_ids = df["time_id"].unique()

    df_list = []

    for tid in all_time_ids:

        sub = (
            df[df["time_id"] == tid]
            .sort_values("seconds_in_bucket")
            .copy()
        )

        full_index = pd.DataFrame({
            "seconds_in_bucket": np.arange(600)
        })

        sub_full = full_index.merge(
            sub,
            on="seconds_in_bucket",
            how="left"
        )

        sub_full["time_id"] = tid

        sub_full = sub_full.sort_values("seconds_in_bucket")

        # forward fill
        sub_full = sub_full.ffill()

        df_list.append(sub_full)

    return pd.concat(df_list, ignore_index=True)


def preprocess_book_data(df):

    df = make_equally_spaced(df)

    df["weighted_price"] = (
        (
            df["bid_price1"] * df["ask_size1"]
            + df["ask_price1"] * df["bid_size1"]
            + df["bid_price2"] * df["ask_size2"]
            + df["ask_price2"] * df["bid_size2"]
        )
        /
        (
            df["bid_size1"]
            + df["ask_size1"]
            + df["bid_size2"]
            + df["ask_size2"]
        )
    )

    df = df.sort_values(
        ["time_id", "seconds_in_bucket"]
    )

    df["log_price"] = np.log(df["weighted_price"])

    df["log_price_diff"] = (
        df.groupby("time_id")["log_price"]
        .diff()
    )

    return df
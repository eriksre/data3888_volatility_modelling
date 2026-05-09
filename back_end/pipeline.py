import os

from data import (
    load_all_stock_paths,
    load_stock_data,
    check_duplicates,
    preprocess_book_data
)

from artifacts import (
    plot_acf,
    plot_pacf
)


def run_acf_pacf_pipeline(parquet_folder):

    stock_paths = load_all_stock_paths(
        parquet_folder
    )

    print(f"Found {len(stock_paths)} parquet files")

    for filepath in stock_paths:

        stock_name = os.path.basename(filepath)

        print(f"\nProcessing {stock_name}")

        df = load_stock_data(filepath)

        print(
            f"Unique time_ids: {df['time_id'].nunique()}"
        )

        check_duplicates(df)
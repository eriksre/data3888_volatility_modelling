# Main thread where the pipeline is run
from pipeline import run_acf_pacf_pipeline


PARQUET_FOLDER = (
    "/Users/vaniakumar/Desktop/data3888/"
    "data3888_volatility_modelling/"
    "individual_book_train_parquet"
)


if __name__ == "__main__":

    run_acf_pacf_pipeline(
        PARQUET_FOLDER
    )
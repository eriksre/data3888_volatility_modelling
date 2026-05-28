# DATA3888 — Volatility Modelling

## Before you do anything, run this

```bash
pip install -r requirements.txt
```

Now, you are okay to run the report. It uses cached results that have been pre-computed by us to save you 4 hours of computational time :)

---------------------------------------------------------------------------

## Before you run the front end, do this

By default, the app expects CSV files in `individual_book_train/`.

If your CSV files live somewhere else, update `INDIVIDUAL_BOOK_DIR` in `back_end/config.py`.


##  Command to run the front end

DO NOT PRESS RUN ALL MODELS UNLESS YOU WANT TO RE-RUN THE MODEL RESULTS WHICH TAKES BETWEEN 30 MINS AND 4 HRS.

```bash
python precompute_feature_cache.py

streamlit run front_end/app.py
```


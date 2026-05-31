# DATA3888 — Volatility Modelling

## Before you do anything, run this

Make SURE to open RStudio from the working directory, not just the direct final_report.qmd file. It has dependencies in the rest of the project. 

Then run the following commands.
```bash
python -m venv .venv
source .venv/bin/activate # macOS/Linux

pip install -r requirements.txt

quarto check jupyter
quarto render final_report.qmd
```
Ensure you have jupyter and quarto working on your computer. This may require different installation commands depending on whether you're on linux/mac/windows.

Now, you are okay to run the report :)

---------------------------------------------------------------------------

## DO NOT MOVE PAST HERE UNLESS YOU WANT TO RECOMPUTE THE RESULTS

## Before you run the front end, do this

By default, the app expects CSV files in `individual_book_train/`. Use the full dataset when running the frontend or recomputing model results.

If your CSV files live somewhere else, update `INDIVIDUAL_BOOK_DIR` in `back_end/config.py`.


##  Command to run the front end

DO NOT PRESS RUN ALL MODELS UNLESS YOU WANT TO RE-RUN THE MODEL RESULTS WHICH TAKES BETWEEN 30 MINS AND 4 HRS.

```bash
python precompute_feature_cache.py

streamlit run front_end/app.py
```

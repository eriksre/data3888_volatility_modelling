# This file contains info about the back end processes in this codebase

## Structure

* `data.py`: stock registry, file loading, preprocessing
* `features.py`: feature engineering, target construction, feature selection
* `models.py`: supervised models, volatility models, baselines, model registry
* `evaluation.py`: train/test splits and metrics
* `universe.py`: cross-stock summaries, similarity matrices, clustering
* `pipeline.py`: orchestrates full analysis runs
* `artifacts.py`: saves and loads run outputs
* `service.py`: backend functions consumed by the front end

## Front-end contracts

* Individual Stock model table: `service.load_individual_model_metrics(...)` returns `model`, `inference_us`, `rmse`, `qlike`, `pred_target`.
* Individual Stock page payload: `service.load_individual_page_data(...)` returns model metrics, predictions, realised series, and available `time_id` values.
* Universe page payload: `service.load_universe_page_data(...)` returns `(summary_df, corr_df)` matching the current Universe scaffold.

## Artifact layout

* `artifacts/runs/`: complete records of user-triggered pipeline runs

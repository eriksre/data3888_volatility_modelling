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

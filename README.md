# DATA3888 — Volatility Modelling

Predicting short-term realised volatility for 112 anonymous stocks using Optiver limit order book data. The setup is: use the first 5 minutes of each 10-minute trading window to predict volatility in the second 5 minutes.

## Before you run anything

Open `config.py` and change `DATA_DIR` to wherever the parquet files are on your machine:

```python
DATA_DIR = r"path/to/individual_book_train"
```

That's the only path you need to change — everything else is relative to it.

### Dependencies
```bash
pip install numpy pandas scikit-learn xgboost lightgbm arch scipy matplotlib
```

## Project layout

```
config.py              <- paths, feature list, colours (edit this first)
utils/
  features.py          <- WAP, feature extraction, lag features, interactions
  metrics.py           <- qlike loss, temporal train/test split
  models.py            <- RF / XGBoost / LightGBM factory

recluster.py           <- cluster 112 stocks into 2 groups (run first)
cluster_validation.py  <- check the clustering makes sense
cluster_models.py      <- train models on each cluster
individual_models.py   <- train models on each stock individually
timeseries_models.py   <- HAR-RV and GARCH benchmarks
recluster_v2.py        <- alternative clustering methods (GMM, correlation)
```

## Run order

```bash
python recluster.py           # ~10 min
python cluster_validation.py  # ~5 min
python cluster_models.py      # ~15 min
python individual_models.py   # ~20 min
python timeseries_models.py   # ~5 min
```

All outputs go to `plots/`.

## Features

32 features total, defined in `config.py` and computed in `utils/features.py`:

- **Volatility**: RV of first half, std/skew/kurtosis/max/min of log returns, WAP range/std, price change count
- **Spread**: mean/std/range of raw and normalised bid-ask spread
- **Order book**: depth, imbalance mean/std, best bid/ask sizes
- **Activity**: row count, arrival rate, WAP momentum (price direction)
- **Lags**: RV lags 1-2, log(RV) lags 1-3, 5-bucket rolling mean
- **Interactions**: vol/depth, vol×spread, pressure×vol

## Results

| Model | Cluster 1 RMSE | Cluster 2 RMSE |
|-------|---------------|----------------|
| Random Forest | 0.2802 | 0.2296 |
| XGBoost | 0.2773 | 0.2268 |
| LightGBM | 0.2770 | 0.2268 |
| HAR-RV | ~0.57 | ~0.59 |
| GARCH(1,1) | ~0.57 | ~0.59 |

Cluster 1 = 33 high-volatility stocks, Cluster 2 = 79 low-volatility stocks. Individual stock models beat the cluster model on roughly 60% of stocks, most clearly on Cluster 2. ML models outperform HAR-RV and GARCH by ~2x — the gap comes from the richer microstructure features which those models can't use.

## Adding a new model

Add it to the dict in `utils/models.py` — it gets picked up automatically by both `cluster_models.py` and `individual_models.py`.

## Adding new features

- Raw order book features → `extract_features()` in `utils/features.py`
- Temporal lags → `add_lag_features()`
- Interactions/derived → `add_interaction_features()`
- Then add the column name to `FEATURE_COLS` in `config.py`

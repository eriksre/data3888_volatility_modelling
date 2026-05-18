# DATA3888 — Volatility Modelling

Predicting short-term realised volatility for 112 anonymous stocks using Optiver limit order book data. The setup is: use the first 5 minutes of each 10-minute trading window to predict volatility in the second 5 minutes.

## Before you run anything

Open `config.py` and change `DATA_DIR` to wherever the parquet files are on your machine:

```python
DATA_DIR = r"path/to/individual_book_train"
```

That's the only path you need to change — everything else is relative to it.

## Run these once. They install the relevant packages you need and compute the feature cache

```bash
pip install -r requirements.txt
python precompute_feature_cache.py
```

# Command to run the app
```
streamlit run front_end/app.py

```

## Clustering Results # DO NOT USE IN FINAL REPORT

| Model | Cluster 1 RMSE | Cluster 2 RMSE |
|-------|---------------|----------------|
| Random Forest | 0.2802 | 0.2296 |
| XGBoost | 0.2773 | 0.2268 |
| LightGBM | 0.2770 | 0.2268 |
| HAR-RV | ~0.57 | ~0.59 |
| GARCH(1,1) | ~0.57 | ~0.59 |

Cluster 1 = 33 high-volatility stocks, Cluster 2 = 79 low-volatility stocks. Individual stock models beat the cluster model on roughly 60% of stocks, most clearly on Cluster 2. ML models outperform HAR-RV and GARCH by ~2x — the gap comes from the richer microstructure features which those models can't use.
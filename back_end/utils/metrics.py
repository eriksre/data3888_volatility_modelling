import numpy as np


def qlike(y_true, y_pred):
    var_true = np.exp(np.asarray(y_true)) ** 2
    var_pred = np.maximum(np.exp(np.asarray(y_pred)) ** 2, 1e-20)
    ratio    = var_true / var_pred
    return float(np.mean(ratio - np.log(ratio) - 1))


def temporal_split(data, train_frac=0.8, time_col="time_id"):
    """Split at the train_frac-th percentile of time_col — no future leakage."""
    time_ids = sorted(data[time_col].unique())
    cutoff   = time_ids[int(len(time_ids) * train_frac) - 1]
    return data[data[time_col] <= cutoff], data[data[time_col] > cutoff], cutoff

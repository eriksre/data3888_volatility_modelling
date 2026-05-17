import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional



def qlike_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    pred_safe = np.clip(y_pred, eps, None)
    realized_safe = np.clip(y_true, eps, None)
    
    ratio = realized_safe / pred_safe
    loss = ratio - np.log(ratio) - 1
    return float(np.mean(loss))


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

# Define more PM metrics above, and add them into the registry below as needed.
AVAILABLE_METRICS = {
    'QLIKE': qlike_score,
    'RMSE': rmse_score,
    'MAE': mae_score
}


def compute_evaluation_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    metrics_to_run: Optional[List[str]] = None
) -> Dict[str, float]:
   
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {metric: np.nan for metric in (metrics_to_run or AVAILABLE_METRICS.keys())}

    metrics = metrics_to_run if metrics_to_run is not None else list(AVAILABLE_METRICS.keys())
    results = {}
    
    for metric_name in metrics:
        metric_func = AVAILABLE_METRICS.get(metric_name.upper())
        if metric_func:
            results[metric_name.upper()] = metric_func(y_true_clean, y_pred_clean)
        else:
            raise ValueError(f"Performance Metric '{metric_name}' is not supported.")
            
    return results


def evaluate_rv_baseline(
    returns_lookup: Dict[int, np.ndarray],
    time_ids: List[int],
    lookback_window: int = 570,
    target_horizon: int = 30
) -> Dict[str, float]:
   
    rows = []
    
    for tid in time_ids:
        returns = returns_lookup.get(tid)
        if returns is None:
            continue
        
        n = len(returns)
        cutoff = n - target_horizon
        
        if cutoff < lookback_window:
            continue

        pred_var = float(np.mean(returns[cutoff - lookback_window:cutoff] ** 2))
        
        target_var = float(np.mean(returns[cutoff:] ** 2))
        
        if np.isfinite(pred_var) and np.isfinite(target_var):
            rows.append({'target_var': target_var, 'pred_var': pred_var})

    df_forecast = pd.DataFrame(rows)
    
    if df_forecast.empty:
        return {metric: np.nan for metric in AVAILABLE_METRICS.keys()}
        
    return compute_evaluation_metrics(
        y_true=df_forecast['target_var'].to_numpy(),
        y_pred=df_forecast['pred_var'].to_numpy()
    )
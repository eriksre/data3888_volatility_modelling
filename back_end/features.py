from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EPS, FEATURE_LABEL_MAP, FeatureConfig, PIPELINE_VERSION, normalize_stock_id
from .data import load_processed_stock


OB_COLS = ["bid_price1", "ask_price1", "bid_size1", "ask_size1", "bid_size2", "ask_size2"]


def ewma_name(lam: float) -> str:
    return f"EWMA_{str(lam).replace('.', '')}"


def ret_autocorr(values: np.ndarray, lag: int = 1) -> float:
    if len(values) <= lag:
        return np.nan
    lag0, lag1 = values[:-lag], values[lag:]
    denom = np.std(lag0) * np.std(lag1)
    if denom <= 0:
        return 0.0
    return float(np.mean((lag0 - lag0.mean()) * (lag1 - lag1.mean())) / denom)


def expanded_feature_cols(feature_config: FeatureConfig) -> list[str]:
    cols = ["seconds_elapsed", "n_obs", "cutoff", "last_return", "abs_last_return"]
    for window in feature_config.return_windows:
        cols.extend(
            [
                f"RV_{window}",
                f"vol_{window}",
                f"mean_abs_ret_{window}",
                f"mean_ret_{window}",
                f"max_abs_ret_{window}",
                f"prop_zero_ret_{window}",
                f"downside_RV_{window}",
                f"upside_RV_{window}",
                f"jump_count_{window}",
                f"ret_q10_{window}",
                f"ret_q90_{window}",
            ]
        )
    for window in feature_config.acf_windows:
        cols.extend([f"ret_autocorr_{window}", f"ret_autocorr5_{window}"])
    cols.extend(ewma_name(lam) for lam in feature_config.ewma_lambdas)
    cols.append("vol_trend")
    for short, long in [(5, 30), (10, 60), (30, 120), (30, 300), (60, 300), (120, 420), (300, 420)]:
        cols.append(f"RV_ratio_{short}_{long}")
    for window in feature_config.book_windows:
        cols.extend(
            [
                f"mean_spread_{window}",
                f"std_spread_{window}",
                f"mean_imbalance_{window}",
                f"std_imbalance_{window}",
                f"depth_ratio_{window}",
                f"mean_depth_{window}",
                f"std_depth_{window}",
                f"book_pressure_{window}",
            ]
        )
    return cols


def compute_features(returns: np.ndarray, cutoff: int, order_book: pd.DataFrame, feature_config: FeatureConfig) -> dict[str, float]:
    r = returns[:cutoff]
    r2 = r**2
    ob = order_book.iloc[:cutoff]
    feats: dict[str, float] = {
        "seconds_elapsed": float(cutoff) / max(len(returns), 1),
        "n_obs": float(len(returns)),
        "cutoff": float(cutoff),
        "last_return": float(r[-1]) if len(r) else np.nan,
        "abs_last_return": float(abs(r[-1])) if len(r) else np.nan,
    }

    for window in feature_config.return_windows:
        if cutoff >= window:
            win = r[-window:]
            win2 = win**2
            sigma = np.std(win)
            feats[f"RV_{window}"] = float(np.mean(win2))
            feats[f"vol_{window}"] = float(np.sqrt(np.mean(win2)))
            feats[f"mean_abs_ret_{window}"] = float(np.mean(np.abs(win)))
            feats[f"mean_ret_{window}"] = float(np.mean(win))
            feats[f"max_abs_ret_{window}"] = float(np.max(np.abs(win)))
            feats[f"prop_zero_ret_{window}"] = float(np.mean(win == 0))
            feats[f"downside_RV_{window}"] = float(np.mean(win2[win < 0])) if np.any(win < 0) else 0.0
            feats[f"upside_RV_{window}"] = float(np.mean(win2[win > 0])) if np.any(win > 0) else 0.0
            feats[f"jump_count_{window}"] = float(np.sum(np.abs(win) > (3 * sigma + EPS)))
            feats[f"ret_q10_{window}"] = float(np.quantile(win, 0.10))
            feats[f"ret_q90_{window}"] = float(np.quantile(win, 0.90))
        else:
            for name in [
                "RV",
                "vol",
                "mean_abs_ret",
                "mean_ret",
                "max_abs_ret",
                "prop_zero_ret",
                "downside_RV",
                "upside_RV",
                "jump_count",
                "ret_q10",
                "ret_q90",
            ]:
                feats[f"{name}_{window}"] = np.nan

    for window in feature_config.acf_windows:
        if cutoff >= window:
            win = r[-window:]
            feats[f"ret_autocorr_{window}"] = ret_autocorr(win, lag=1)
            feats[f"ret_autocorr5_{window}"] = ret_autocorr(win, lag=5)
        else:
            feats[f"ret_autocorr_{window}"] = np.nan
            feats[f"ret_autocorr5_{window}"] = np.nan

    r2_series = pd.Series(r2)
    for lam in feature_config.ewma_lambdas:
        feats[ewma_name(lam)] = float(r2_series.ewm(alpha=1 - lam, adjust=False).mean().iloc[-1])

    feats["vol_trend"] = feats.get("RV_30", np.nan) / (feats.get("RV_300", np.nan) + EPS)
    for short, long in [(5, 30), (10, 60), (30, 120), (30, 300), (60, 300), (120, 420), (300, 420)]:
        feats[f"RV_ratio_{short}_{long}"] = feats.get(f"RV_{short}", np.nan) / (feats.get(f"RV_{long}", np.nan) + EPS)

    if len(ob):
        mid = (ob["bid_price1"] + ob["ask_price1"]) / 2
        spread = (ob["ask_price1"] - ob["bid_price1"]) / mid.replace(0, np.nan)
        total_l1 = ob["bid_size1"] + ob["ask_size1"]
        total_l2 = (ob["bid_size2"] + ob["ask_size2"]).replace(0, np.nan)
        depth = total_l1 + total_l2
        imbalance = (ob["bid_size1"] - ob["ask_size1"]) / total_l1.replace(0, np.nan)
        pressure = (ob["bid_size1"] + ob["bid_size2"] - ob["ask_size1"] - ob["ask_size2"]) / depth.replace(0, np.nan)
    else:
        spread = total_l1 = total_l2 = depth = imbalance = pressure = pd.Series(dtype=float)

    for window in feature_config.book_windows:
        if len(ob) >= window:
            tail = slice(len(ob) - window, len(ob))
            feats[f"mean_spread_{window}"] = float(spread.iloc[tail].mean())
            feats[f"std_spread_{window}"] = float(spread.iloc[tail].std())
            feats[f"mean_imbalance_{window}"] = float(imbalance.iloc[tail].mean())
            feats[f"std_imbalance_{window}"] = float(imbalance.iloc[tail].std())
            feats[f"depth_ratio_{window}"] = float((total_l1 / total_l2).iloc[tail].mean())
            feats[f"mean_depth_{window}"] = float(depth.iloc[tail].mean())
            feats[f"std_depth_{window}"] = float(depth.iloc[tail].std())
            feats[f"book_pressure_{window}"] = float(pressure.iloc[tail].mean())
        else:
            for name in [
                "mean_spread",
                "std_spread",
                "mean_imbalance",
                "std_imbalance",
                "depth_ratio",
                "mean_depth",
                "std_depth",
                "book_pressure",
            ]:
                feats[f"{name}_{window}"] = np.nan

    return feats


def build_feature_frame(processed: pd.DataFrame, stock: str, feature_config: FeatureConfig) -> pd.DataFrame:
    rows = []
    feature_cols = expanded_feature_cols(feature_config)
    for time_id, chunk in processed.groupby("time_id", sort=False):
        chunk = chunk.sort_values("seconds_in_bucket")
        raw_returns = chunk["log_price_diff"].to_numpy(dtype=float) * 10000
        valid = np.isfinite(raw_returns)
        returns = raw_returns[valid]
        order_book = chunk.loc[valid, OB_COLS].reset_index(drop=True)
        n_obs = len(returns)
        horizon = int(feature_config.forecast_horizon)
        if n_obs < horizon + max(30, min(feature_config.return_windows)):
            continue
        cutoff = n_obs - horizon
        target = float(np.mean(returns[cutoff:] ** 2))
        row = {
            "stock_id": stock,
            "time_id": int(time_id),
            "target_var": target,
            "target_vol": float(np.sqrt(max(target, 0.0))),
        }
        row.update(compute_features(returns, cutoff, order_book, feature_config))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["stock_id", "time_id", "target_var", "target_vol", *feature_cols])
    return frame.dropna(subset=["target_var"]).reset_index(drop=True)


def usable_feature_columns(features: pd.DataFrame, feature_config: FeatureConfig) -> list[str]:
    cols = []
    for col in expanded_feature_cols(feature_config):
        if col not in features.columns:
            continue
        values = features[col]
        if values.notna().all() and np.isfinite(values.to_numpy(dtype=float)).all() and values.nunique(dropna=True) > 1:
            cols.append(col)
    return cols


def load_stock_features(stock: str, data_config, feature_config: FeatureConfig) -> pd.DataFrame:
    stock_name = normalize_stock_id(stock)
    processed = load_processed_stock(stock_name, data_config)
    return build_feature_frame(processed, stock_name, feature_config)


def resolve_manual_feature_labels(labels: tuple[str, ...], available_columns: list[str]) -> list[str]:
    selected: list[str] = []
    for label in labels:
        if label in available_columns:
            selected.append(label)
            continue
        selected.extend(FEATURE_LABEL_MAP.get(label, []))
    return [col for col in dict.fromkeys(selected) if col in available_columns]

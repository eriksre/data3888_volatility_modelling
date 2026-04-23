import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

PLOT_DIR = r"C:\Users\Ayush\Downloads\FINC3017\plots"
import os
os.makedirs(PLOT_DIR, exist_ok=True)

DATA_DIR = r"C:\Users\Ayush\Downloads\DATA3888\data3888 finc6\Optiver_extracted\individual_book_train"
STOCKS = [5, 14]


def compute_wap(df):
    return (df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]) / (
        df["bid_size1"] + df["ask_size1"]
    )


def compute_realized_vol(wap_series):
    log_returns = np.log(wap_series).diff().dropna()
    return np.sqrt((log_returns**2).sum())


def extract_features(half_df):
    """Aggregate first-half features per time_id."""
    half_df = half_df.copy()
    half_df["wap"] = compute_wap(half_df)
    half_df["log_return"] = half_df.groupby("time_id")["wap"].transform(
        lambda x: np.log(x).diff()
    )
    half_df["spread"] = half_df["ask_price1"] - half_df["bid_price1"]

    feats = half_df.groupby("time_id").agg(
        rv_first=("log_return", lambda x: np.sqrt((x**2).sum())),
        mean_bid_price1=("bid_price1", "mean"),
        mean_ask_price1=("ask_price1", "mean"),
        mean_bid_size1=("bid_size1", "mean"),
        mean_ask_size1=("ask_size1", "mean"),
        mean_spread=("spread", "mean"),
        std_spread=("spread", "std"),
        n_rows=("wap", "count"),
    ).reset_index()
    return feats


def compute_target(half_df):
    """Compute log(RV) of second half per time_id."""
    half_df = half_df.copy()
    half_df["wap"] = compute_wap(half_df)

    def rv(group):
        lr = np.log(group["wap"]).diff().dropna()
        return np.sqrt((lr**2).sum())

    targets = half_df.groupby("time_id").apply(rv).reset_index()
    targets.columns = ["time_id", "rv_second"]
    targets["log_rv_second"] = np.log(targets["rv_second"] + 1e-8)
    return targets[["time_id", "log_rv_second"]]


def qlike(y_true, y_pred):
    """QLIKE loss: mean(sigma2_true/sigma2_pred - log(sigma2_true/sigma2_pred) - 1)"""
    # convert log-vol predictions back to variance
    var_true = np.exp(y_true) ** 2
    var_pred = np.exp(y_pred) ** 2
    return np.mean(var_true / var_pred - np.log(var_true / var_pred) - 1)


def run_pipeline(stock_id):
    print(f"\n{'='*50}")
    print(f"Stock {stock_id}")
    print(f"{'='*50}")

    df = pd.read_parquet(f"{DATA_DIR}/stock_{stock_id}.parquet")

    first_half = df[df["seconds_in_bucket"] < 300]
    second_half = df[df["seconds_in_bucket"] >= 300]

    features = extract_features(first_half)
    targets = compute_target(second_half)

    data = features.merge(targets, on="time_id").dropna()
    print(f"Samples: {len(data)}")

    feature_cols = [c for c in data.columns if c not in ["time_id", "log_rv_second"]]
    X = data[feature_cols].values
    y = data["log_rv_second"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["RandomForest"] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        "QLIKE": qlike(y_test, y_pred_rf),
        "model": rf,
        "preds": y_pred_rf,
    }

    # --- LASSO (LassoCV uses default lambda/alpha selection via CV) ---
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    results["LASSO"] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
        "QLIKE": qlike(y_test, y_pred_lasso),
        "model": lasso,
        "preds": y_pred_lasso,
        "alpha": lasso.alpha_,
    }

    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  RMSE  = {res['RMSE']:.6f}")
        print(f"  QLIKE = {res['QLIKE']:.6f}")
        if name == "LASSO":
            print(f"  Selected alpha = {res['alpha']:.6f}")
            coef_df = pd.DataFrame(
                {"feature": feature_cols, "coef": lasso.coef_}
            ).sort_values("coef", key=abs, ascending=False)
            print(f"  Feature coefficients:\n{coef_df.to_string(index=False)}")

    # RF feature importances
    imp_df = pd.DataFrame(
        {"feature": feature_cols, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)
    print(f"\nRandom Forest feature importances:\n{imp_df.to_string(index=False)}")

    plot_results(stock_id, feature_cols, y_test, results, rf, lasso)
    return results, data, feature_cols, y_test, results["RandomForest"]["preds"], results["LASSO"]["preds"]


def plot_results(stock_id, feature_cols, y_test, results, rf, lasso):
    y_pred_rf = results["RandomForest"]["preds"]
    y_pred_lasso = results["LASSO"]["preds"]
    resid_rf = y_test - y_pred_rf
    resid_lasso = y_test - y_pred_lasso
    lim = [min(y_test.min(), y_pred_rf.min(), y_pred_lasso.min()) - 0.1,
           max(y_test.max(), y_pred_rf.max(), y_pred_lasso.max()) + 0.1]
    colors = {"RandomForest": "#2196F3", "LASSO": "#FF5722"}

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"Stock {stock_id} — Volatility Model Diagnostics", fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # --- Row 1: Actual vs Predicted ---
    for col, (name, y_pred, color) in enumerate(zip(
        ["RandomForest", "LASSO"],
        [y_pred_rf, y_pred_lasso],
        [colors["RandomForest"], colors["LASSO"]]
    )):
        ax = fig.add_subplot(gs[0, col * 2: col * 2 + 2])
        ax.scatter(y_test, y_pred, alpha=0.3, s=10, color=color, label="Predictions")
        ax.plot(lim, lim, "k--", linewidth=1, label="Perfect fit")
        rmse = results[name]["RMSE"]
        qlike = results[name]["QLIKE"]
        ax.set_title(f"{name}: Actual vs Predicted\nRMSE={rmse:.4f}  QLIKE={qlike:.4f}", fontsize=10)
        ax.set_xlabel("Actual log(RV)")
        ax.set_ylabel("Predicted log(RV)")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.legend(fontsize=8)

    # --- Row 2: Residual plots ---
    for col, (name, resid, color) in enumerate(zip(
        ["RandomForest", "LASSO"],
        [resid_rf, resid_lasso],
        [colors["RandomForest"], colors["LASSO"]]
    )):
        ax = fig.add_subplot(gs[1, col * 2: col * 2 + 2])
        ax.scatter(y_test, resid, alpha=0.3, s=10, color=color)
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(f"{name}: Residuals vs Actual", fontsize=10)
        ax.set_xlabel("Actual log(RV)")
        ax.set_ylabel("Residual")

    # --- Row 3 left: RF Feature Importance ---
    ax_fi = fig.add_subplot(gs[2, 0:2])
    imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
    imp.plot(kind="barh", ax=ax_fi, color=colors["RandomForest"])
    ax_fi.set_title("Random Forest Feature Importances", fontsize=10)
    ax_fi.set_xlabel("Importance")

    # --- Row 3 right: LASSO Coefficients ---
    ax_lasso = fig.add_subplot(gs[2, 2:4])
    coefs = pd.Series(lasso.coef_, index=feature_cols).sort_values()
    bar_colors = [colors["LASSO"] if v >= 0 else "#9C27B0" for v in coefs]
    coefs.plot(kind="barh", ax=ax_lasso, color=bar_colors)
    ax_lasso.axvline(0, color="black", linewidth=0.8)
    ax_lasso.set_title(f"LASSO Coefficients (α={lasso.alpha_:.4f})", fontsize=10)
    ax_lasso.set_xlabel("Coefficient")

    path = os.path.join(PLOT_DIR, f"stock_{stock_id}_diagnostics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {path}")

    # --- Separate: sorted actual vs predicted overlay ---
    fig2, axes = plt.subplots(1, 2, figsize=(16, 4))
    fig2.suptitle(f"Stock {stock_id} — Sorted Actual vs Predicted (test set)", fontsize=13, fontweight="bold")
    sort_idx = np.argsort(y_test)
    x_range = np.arange(len(y_test))
    for ax, name, y_pred, color in zip(
        axes,
        ["RandomForest", "LASSO"],
        [y_pred_rf, y_pred_lasso],
        [colors["RandomForest"], colors["LASSO"]]
    ):
        ax.plot(x_range, y_test[sort_idx], color="black", linewidth=1, label="Actual", alpha=0.7)
        ax.plot(x_range, y_pred[sort_idx], color=color, linewidth=1, label=name, alpha=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Sorted test index")
        ax.set_ylabel("log(RV)")
        ax.legend(fontsize=8)

    path2 = os.path.join(PLOT_DIR, f"stock_{stock_id}_sorted_predictions.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {path2}")


if __name__ == "__main__":
    all_results = {}
    for sid in STOCKS:
        all_results[sid] = run_pipeline(sid)

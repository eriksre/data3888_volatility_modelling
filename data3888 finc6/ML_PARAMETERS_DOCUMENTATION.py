"""
ML MODELS - INPUT/OUTPUT PARAMETERS DOCUMENTATION
====================================================

This document explains the input and output parameters for fitting ML models 
on the Optiver volatility prediction dataset.
"""

# ============================================================================
# 1. DATA PREPARATION
# ============================================================================

# INPUT DATA:
# -----------
# order_book_feature.parquet: (17,646,119 rows × 11 cols)
#   - stock_id (int64): Stock identifier
#   - time_id (int64): Time window identifier
#   - seconds_in_bucket (float64): Seconds within the time window
#   - bid_price1, ask_price1, bid_price2, ask_price2 (float64): Price levels
#   - bid_size1, ask_size1, bid_size2, ask_size2 (int64): Order sizes

# order_book_target.parquet: (17,911,332 rows × 11 cols)
#   - Same structure as order_book_feature

# trades.parquet: (6,853,535 rows × 6 cols)
#   - time_id (int64): Time identifier
#   - stock_id (int64): Stock identifier
#   - seconds_in_bucket (float64): Seconds within window
#   - price (float64): Trade price
#   - size (float64): Trade size/volume
#   - order_count (float64): Number of orders

# train.parquet: (11,520 rows × 3 cols)
#   - time_id (int64): Time identifier
#   - stock_id (int64): Stock identifier
#   - target (float64): TARGET VARIABLE (Volatility to predict)

# stock_ids.parquet: (10 rows × 2 cols)
#   - instrument (str): Stock name (e.g., 'AAPL XNAS')
#   - stock_id (int64): Stock identifier


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

# INPUT:
# ------
# Raw data from parquet files

# PROCESS:
# --------
# Step 1: Aggregate order book data by (stock_id, time_id)
#   - Compute mean, std, min, max of bid/ask prices
#   - Compute mean and sum of bid/ask sizes
#   OUTPUT: order_book_features DataFrame (11,520 × 12 cols)

# Step 2: Create derived features from order book
#   - bid_ask_spread = ask_price1 - bid_price1
#   - bid_ask_spread_pct = (bid_ask_spread / mid_price) * 100
#   - mid_price = (bid_price1 + ask_price1) / 2
#   - imbalance = (bid_size1 - ask_size1) / (bid_size1 + ask_size1)
#   OUTPUT: Extended order_book_features DataFrame (11,520 × 16 cols)

# Step 3: Aggregate trades data by (stock_id, time_id)
#   - mean, std, min, max of prices
#   - mean, sum, std of sizes
#   - sum, mean of order_count
#   OUTPUT: trades_features DataFrame (11,520 × 11 cols)

# Step 4: Merge all features
#   - Merge order_book_features with trades_features on (stock_id, time_id)
#   - Merge with train target on (stock_id, time_id)
#   OUTPUT: Merged data (11,520 × 32 cols)

# ============================================================================
# 3. FINAL FEATURE MATRIX & TARGET
# ============================================================================

# INPUT TO MODEL TRAINING:
# ========================

# X (Feature Matrix):
#   Shape: (11,520 rows × 29 columns)
#   Content: 29 engineered features (excluding stock_id, time_id, target)
#   Data Type: numpy array (float64), standardized (mean=0, std=1)
#   Features included:
#   - bid_price1_{mean,std,min,max}
#   - ask_price1_{mean,std,min,max}
#   - bid_size1_{mean,sum}
#   - ask_size1_{mean,sum}
#   - bid_price2_mean, ask_price2_mean
#   - bid_size2_mean, ask_size2_mean
#   - bid_ask_spread, bid_ask_spread_pct
#   - mid_price, imbalance
#   - price_{mean,std,min,max}
#   - size_{mean,sum,std}
#   - order_count_{sum,mean}

# y (Target Variable):
#   Shape: (11,520,) - 1D array
#   Content: Volatility prediction targets
#   Data Type: float64
#   Range: 0.000347 to 0.030335
#   Meaning: Price volatility measure to be predicted

# X_train (Training Features):
#   Shape: (9,216 rows × 29 columns) - 80% of data
#   Data Type: numpy array (float64), standardized
#   Scaling: StandardScaler applied (mean=0, std=1)

# y_train (Training Target):
#   Shape: (9,216,) - 1D array
#   Corresponding targets for X_train

# X_test (Test Features):
#   Shape: (2,304 rows × 29 columns) - 20% of data
#   Data Type: numpy array (float64), standardized
#   Scaling: Using same scaler as training data

# y_test (Test Target):
#   Shape: (2,304,) - 1D array
#   Corresponding targets for X_test


# ============================================================================
# 4. MODEL FIT PARAMETERS
# ============================================================================

# 1. LINEAR REGRESSION
# ====================
INPUT_PARAMETERS_LR = {
    "X": "numpy array (9,216 × 29)",
    "y": "numpy array (9,216,)"
}

MODEL_HYPERPARAMETERS_LR = {
    # No hyperparameters to tune
}

OUTPUT_PARAMETERS_LR = {
    "model_attributes": {
        "coef_": "shape (29,) - weights for each feature",
        "intercept_": "scalar - bias term"
    },
    "predictions": "numpy array (2,304,) - predicted volatility for test set"
}

PERFORMANCE_METRICS_LR = {
    "MSE": "Mean Squared Error = 0.000000",
    "RMSE": "Root Mean Squared Error = 0.000675",
    "MAE": "Mean Absolute Error = 0.000532",
    "R2": "R-squared Score = 0.550126"
}


# 2. RIDGE REGRESSION
# ===================
INPUT_PARAMETERS_RIDGE = {
    "X": "numpy array (9,216 × 29)",
    "y": "numpy array (9,216,)"
}

MODEL_HYPERPARAMETERS_RIDGE = {
    "alpha": 1.0,  # Regularization strength (L2 penalty)
}

OUTPUT_PARAMETERS_RIDGE = {
    "model_attributes": {
        "coef_": "shape (29,) - regularized weights for each feature",
        "intercept_": "scalar - bias term"
    },
    "predictions": "numpy array (2,304,) - predicted volatility for test set"
}

PERFORMANCE_METRICS_RIDGE = {
    "MSE": "Mean Squared Error = 0.000000",
    "RMSE": "Root Mean Squared Error = 0.000664",
    "MAE": "Mean Absolute Error = 0.000503",
    "R2": "R-squared Score = 0.564261"  # BEST MODEL
}


# 3. LASSO REGRESSION
# ===================
INPUT_PARAMETERS_LASSO = {
    "X": "numpy array (9,216 × 29)",
    "y": "numpy array (9,216,)"
}

MODEL_HYPERPARAMETERS_LASSO = {
    "alpha": 0.0001,  # Regularization strength (L1 penalty - feature selection)
}

OUTPUT_PARAMETERS_LASSO = {
    "model_attributes": {
        "coef_": "shape (29,) - sparse weights (some features zeroed out)",
        "intercept_": "scalar - bias term"
    },
    "predictions": "numpy array (2,304,) - predicted volatility for test set"
}

PERFORMANCE_METRICS_LASSO = {
    "MSE": "Mean Squared Error = 0.000001",
    "RMSE": "Root Mean Squared Error = 0.000932",
    "MAE": "Mean Absolute Error = 0.000676",
    "R2": "R-squared Score = 0.142048"
}


# 4. RANDOM FOREST
# ================
INPUT_PARAMETERS_RF = {
    "X": "numpy array (9,216 × 29)",
    "y": "numpy array (9,216,)"
}

MODEL_HYPERPARAMETERS_RF = {
    "n_estimators": 100,        # Number of trees in forest
    "max_depth": 15,            # Maximum tree depth
    "random_state": 42,         # Reproducibility seed
    "n_jobs": -1                # Use all CPU cores
}

OUTPUT_PARAMETERS_RF = {
    "model_attributes": {
        "feature_importances_": "shape (29,) - importance score for each feature",
        "n_estimators": 100,    # Actual number of trees
        "trees": "100 decision trees"
    },
    "predictions": "numpy array (2,304,) - predicted volatility for test set"
}

PERFORMANCE_METRICS_RF = {
    "MSE": "Mean Squared Error = 0.000001",
    "RMSE": "Root Mean Squared Error = 0.001045",
    "MAE": "Mean Absolute Error = 0.000932",
    "R2": "R-squared Score = -0.077819"
}


# 5. GRADIENT BOOSTING
# ====================
INPUT_PARAMETERS_GB = {
    "X": "numpy array (9,216 × 29)",
    "y": "numpy array (9,216,)"
}

MODEL_HYPERPARAMETERS_GB = {
    "n_estimators": 100,        # Number of boosting stages
    "learning_rate": 0.1,       # Step shrinkage (controls overfitting)
    "max_depth": 5,             # Maximum tree depth
    "random_state": 42,         # Reproducibility seed
}

OUTPUT_PARAMETERS_GB = {
    "model_attributes": {
        "feature_importances_": "shape (29,) - importance score for each feature",
        "n_estimators": 100,    # Number of trees
        "trees": "Sequential ensemble of 100 trees"
    },
    "predictions": "numpy array (2,304,) - predicted volatility for test set"
}

PERFORMANCE_METRICS_GB = {
    "MSE": "Mean Squared Error = 0.000001",
    "RMSE": "Root Mean Squared Error = 0.001001",
    "MAE": "Mean Absolute Error = 0.000902",
    "R2": "R-squared Score = 0.010783"
}


# 6. XGBOOST
# ==========
INPUT_PARAMETERS_XGB = {
    "X": "numpy array (9,216 × 29)",
    "y": "numpy array (9,216,)"
}

MODEL_HYPERPARAMETERS_XGB = {
    "n_estimators": 100,        # Number of boosting rounds
    "learning_rate": 0.1,       # Step shrinkage
    "max_depth": 5,             # Maximum tree depth
    "random_state": 42,         # Reproducibility seed
    "verbosity": 0              # Silent output
}

OUTPUT_PARAMETERS_XGB = {
    "model_attributes": {
        "feature_importances_": "shape (29,) - importance score for each feature",
        "n_estimators": 100,    # Number of trees
        "best_score": "Float - best validation score (if applicable)"
    },
    "predictions": "numpy array (2,304,) - predicted volatility for test set"
}

PERFORMANCE_METRICS_XGB = {
    "MSE": "Mean Squared Error = 0.000001",
    "RMSE": "Root Mean Squared Error = 0.001030",
    "MAE": "Mean Absolute Error = 0.000916",
    "R2": "R-squared Score = -0.047539"
}


# 7. LIGHTGBM
# ===========
INPUT_PARAMETERS_LGB = {
    "X": "numpy array (9,216 × 29)",
    "y": "numpy array (9,216,)"
}

MODEL_HYPERPARAMETERS_LGB = {
    "n_estimators": 100,        # Number of boosting rounds
    "learning_rate": 0.1,       # Step shrinkage
    "max_depth": 5,             # Maximum tree depth
    "random_state": 42,         # Reproducibility seed
    "verbosity": -1             # Silent output
}

OUTPUT_PARAMETERS_LGB = {
    "model_attributes": {
        "feature_importances_": "shape (29,) - importance score for each feature",
        "n_estimators": 100,    # Number of trees
        "booster_":"LGBMBooster object"
    },
    "predictions": "numpy array (2,304,) - predicted volatility for test set"
}

PERFORMANCE_METRICS_LGB = {
    "MSE": "Mean Squared Error = 0.000001",
    "RMSE": "Root Mean Squared Error = 0.001029",
    "MAE": "Mean Absolute Error = 0.000926",
    "R2": "R-squared Score = -0.044642"
}


# ============================================================================
# 5. EVALUATION METRICS (OUTPUT)
# ============================================================================

EVALUATION_METRICS = {
    "MSE": {
        "formula": "mean((y_true - y_pred)^2)",
        "range": "[0, ∞)",
        "interpretation": "Lower is better. Average squared error.",
        "scaling": "Penalizes large errors more"
    },
    
    "RMSE": {
        "formula": "sqrt(MSE)",
        "range": "[0, ∞)",
        "interpretation": "Lower is better. Same unit as target variable.",
        "scaling": "Root of MSE, more interpretable"
    },
    
    "MAE": {
        "formula": "mean(|y_true - y_pred|)",
        "range": "[0, ∞)",
        "interpretation": "Lower is better. Average absolute error.",
        "scaling": "Treats all errors equally"
    },
    
    "R2_Score": {
        "formula": "1 - (SS_res / SS_tot)",
        "range": "[-∞, 1]",
        "interpretation": "Higher is better. Proportion of variance explained.",
        "scaling": "R²=1: Perfect fit, R²=0: No better than mean baseline, R²<0: Worse than baseline"
    }
}


# ============================================================================
# 6. FEATURE IMPORTANCE (OUTPUT for Tree-based Models)
# ============================================================================

TOP_10_FEATURES = {
    1: "bid_ask_spread_pct (165.68) - Market liquidity indicator",
    2: "order_count_sum (155.59) - Total trading activity",
    3: "order_count_mean (79.01) - Average order count",
    4: "size_sum (67.02) - Total volume traded",
    5: "size_std (54.00) - Volatility in trade sizes",
    6: "ask_size1_sum (44.51) - Total ask volume at level 1",
    7: "bid_size1_sum (43.50) - Total bid volume at level 1",
    8: "bid_size1_mean (39.50) - Average bid volume at level 1",
    9: "ask_size2_mean (39.00) - Average ask volume at level 2",
    10: "bid_price1_mean (36.50) - Average bid price at level 1"
}


# ============================================================================
# 7. SUMMARY TABLE
# ============================================================================

SUMMARY = """
╔════════════════════╦═══════════════════════════════════════════════════╗
║ STAGE              ║ INPUT/OUTPUT DIMENSIONS & DESCRIPTION             ║
╠════════════════════╬═══════════════════════════════════════════════════╣
│ 1. Raw Data        │ Multiple parquet files (millions of rows)         │
├────────────────────┼───────────────────────────────────────────────────┤
│ 2. Aggregation     │ 11,520 rows (unique stock_id × time_id pairs)    │
├────────────────────┼───────────────────────────────────────────────────┤
│ 3. Feature Eng.    │ 29 engineered features + target                   │
├────────────────────┼───────────────────────────────────────────────────┤
│ 4. Scaling         │ StandardScaler (mean=0, std=1)                    │
├────────────────────┼───────────────────────────────────────────────────┤
│ 5. Train/Test      │ Train: (9,216 × 29), Test: (2,304 × 29)         │
├────────────────────┼───────────────────────────────────────────────────┤
│ 6. Model Fit       │ fit(X_train, y_train)                             │
├────────────────────┼───────────────────────────────────────────────────┤
│ 7. Prediction      │ predict(X_test) → (2,304,) array                  │
├────────────────────┼───────────────────────────────────────────────────┤
│ 8. Evaluation      │ Compare with y_test using MSE/RMSE/MAE/R²        │
╚════════════════════╩═══════════════════════════════════════════════════╝
"""

print(SUMMARY)

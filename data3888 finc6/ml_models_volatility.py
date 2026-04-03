import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

#data_dir = Path(r"c:\Users\Ayush\Downloads\DATA3888\data3888 finc6\Optiver_additional_data_extracted\Optiver_additional data")
data_dir  = Path(__file__).parent.parent / "Optiver_additional data"

# Load data
print("Loading data...")
order_book_feature = pd.read_parquet(data_dir / 'order_book_feature.parquet')
order_book_target = pd.read_parquet(data_dir / 'order_book_target.parquet')
trades = pd.read_parquet(data_dir / 'trades.parquet')
train = pd.read_parquet(data_dir / 'train.parquet')
stock_ids = pd.read_parquet(data_dir / 'stock_ids.parquet')

print(f"Data loaded successfully")

# Feature Engineering
print("\nPerforming feature engineering...")

# Aggregate order book features by stock_id and time_id
order_book_features = order_book_feature.groupby(['stock_id', 'time_id']).agg({
    'bid_price1': ['mean', 'std', 'min', 'max'],
    'ask_price1': ['mean', 'std', 'min', 'max'],
    'bid_size1': ['mean', 'sum'],
    'ask_size1': ['mean', 'sum'],
    'bid_price2': ['mean'],
    'ask_price2': ['mean'],
    'bid_size2': ['mean'],
    'ask_size2': ['mean'],
}).reset_index()

# Flatten column names
order_book_features.columns = ['_'.join(col).strip('_') for col in order_book_features.columns.values]
order_book_features.rename(columns={'stock_id_': 'stock_id', 'time_id_': 'time_id'}, inplace=True)

# Create derived features from order book data
order_book_features['bid_ask_spread'] = order_book_features['ask_price1_mean'] - order_book_features['bid_price1_mean']
order_book_features['bid_ask_spread_pct'] = (order_book_features['bid_ask_spread'] / 
                                             ((order_book_features['ask_price1_mean'] + order_book_features['bid_price1_mean']) / 2)) * 100
order_book_features['mid_price'] = (order_book_features['ask_price1_mean'] + order_book_features['bid_price1_mean']) / 2
order_book_features['imbalance'] = (order_book_features['bid_size1_sum'] - order_book_features['ask_size1_sum']) / (order_book_features['bid_size1_sum'] + order_book_features['ask_size1_sum'])

# Aggregate trades by stock_id and time_id
trades_features = trades.groupby(['stock_id', 'time_id']).agg({
    'price': ['mean', 'std', 'min', 'max'],
    'size': ['mean', 'sum', 'std'],
    'order_count': ['sum', 'mean'],
}).reset_index()

# Flatten column names
trades_features.columns = ['_'.join(col).strip('_') for col in trades_features.columns.values]
trades_features.rename(columns={'stock_id_': 'stock_id', 'time_id_': 'time_id'}, inplace=True)

# Merge features
X = order_book_features.merge(trades_features, on=['stock_id', 'time_id'], how='inner')
y = train.copy()

# Merge features with target
data = X.merge(y, on=['stock_id', 'time_id'], how='inner')

# Drop rows with NaN
data = data.dropna()

print(f"Feature matrix shape: {X.shape}")
print(f"Final dataset shape: {data.shape}")

# Prepare X and y
X = data.drop(['target', 'stock_id', 'time_id'], axis=1)
y = data['target']

print(f"Feature set shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Time-based split (important for time series)
split_idx = int(len(X) * 0.8)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y.values[:split_idx], y.values[split_idx:]

print(f"\nTrain set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Dictionary to store results
results = {}

# 1. Linear Regression
print("\n" + "="*50)
print("1. LINEAR REGRESSION")
print("="*50)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results['Linear Regression'] = {
    'mse': mean_squared_error(y_test, y_pred_lr),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    'mae': mean_absolute_error(y_test, y_pred_lr),
    'r2': r2_score(y_test, y_pred_lr)
}
print(f"MSE: {results['Linear Regression']['mse']:.6f}")
print(f"RMSE: {results['Linear Regression']['rmse']:.6f}")
print(f"MAE: {results['Linear Regression']['mae']:.6f}")
print(f"R² Score: {results['Linear Regression']['r2']:.6f}")

# 2. Ridge Regression
print("\n" + "="*50)
print("2. RIDGE REGRESSION")
print("="*50)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
results['Ridge Regression'] = {
    'mse': mean_squared_error(y_test, y_pred_ridge),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
    'mae': mean_absolute_error(y_test, y_pred_ridge),
    'r2': r2_score(y_test, y_pred_ridge)
}
print(f"MSE: {results['Ridge Regression']['mse']:.6f}")
print(f"RMSE: {results['Ridge Regression']['rmse']:.6f}")
print(f"MAE: {results['Ridge Regression']['mae']:.6f}")
print(f"R² Score: {results['Ridge Regression']['r2']:.6f}")

# 3. Lasso Regression
print("\n" + "="*50)
print("3. LASSO REGRESSION")
print("="*50)
lasso = Lasso(alpha=0.0001)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
results['Lasso Regression'] = {
    'mse': mean_squared_error(y_test, y_pred_lasso),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
    'mae': mean_absolute_error(y_test, y_pred_lasso),
    'r2': r2_score(y_test, y_pred_lasso)
}
print(f"MSE: {results['Lasso Regression']['mse']:.6f}")
print(f"RMSE: {results['Lasso Regression']['rmse']:.6f}")
print(f"MAE: {results['Lasso Regression']['mae']:.6f}")
print(f"R² Score: {results['Lasso Regression']['r2']:.6f}")

# 4. Random Forest
print("\n" + "="*50)
print("4. RANDOM FOREST")
print("="*50)
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = {
    'mse': mean_squared_error(y_test, y_pred_rf),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'mae': mean_absolute_error(y_test, y_pred_rf),
    'r2': r2_score(y_test, y_pred_rf)
}
print(f"MSE: {results['Random Forest']['mse']:.6f}")
print(f"RMSE: {results['Random Forest']['rmse']:.6f}")
print(f"MAE: {results['Random Forest']['mae']:.6f}")
print(f"R² Score: {results['Random Forest']['r2']:.6f}")

# 5. Gradient Boosting
print("\n" + "="*50)
print("5. GRADIENT BOOSTING")
print("="*50)
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
results['Gradient Boosting'] = {
    'mse': mean_squared_error(y_test, y_pred_gb),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
    'mae': mean_absolute_error(y_test, y_pred_gb),
    'r2': r2_score(y_test, y_pred_gb)
}
print(f"MSE: {results['Gradient Boosting']['mse']:.6f}")
print(f"RMSE: {results['Gradient Boosting']['rmse']:.6f}")
print(f"MAE: {results['Gradient Boosting']['mae']:.6f}")
print(f"R² Score: {results['Gradient Boosting']['r2']:.6f}")

# 6. XGBoost
print("\n" + "="*50)
print("6. XGBOOST")
print("="*50)
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbosity=0)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
results['XGBoost'] = {
    'mse': mean_squared_error(y_test, y_pred_xgb),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
    'mae': mean_absolute_error(y_test, y_pred_xgb),
    'r2': r2_score(y_test, y_pred_xgb)
}
print(f"MSE: {results['XGBoost']['mse']:.6f}")
print(f"RMSE: {results['XGBoost']['rmse']:.6f}")
print(f"MAE: {results['XGBoost']['mae']:.6f}")
print(f"R² Score: {results['XGBoost']['r2']:.6f}")

# 7. LightGBM
print("\n" + "="*50)
print("7. LIGHTGBM")
print("="*50)
lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbosity=-1)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
results['LightGBM'] = {
    'mse': mean_squared_error(y_test, y_pred_lgb),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lgb)),
    'mae': mean_absolute_error(y_test, y_pred_lgb),
    'r2': r2_score(y_test, y_pred_lgb)
}
print(f"MSE: {results['LightGBM']['mse']:.6f}")
print(f"RMSE: {results['LightGBM']['rmse']:.6f}")
print(f"MAE: {results['LightGBM']['mae']:.6f}")
print(f"R² Score: {results['LightGBM']['r2']:.6f}")

# Summary and Comparison
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)

# Create results dataframe
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('r2', ascending=False)

print("\nRanked by R² Score (Higher is Better):")
print(results_df.to_string())

print("\n" + "="*70)
print(f"BEST MODEL: {results_df.index[0]}")
print(f"R² Score: {results_df['r2'].iloc[0]:.6f}")
print(f"RMSE: {results_df['rmse'].iloc[0]:.6f}")
print(f"MAE: {results_df['mae'].iloc[0]:.6f}")
print("="*70)

# Feature importance for tree-based models
print("\n" + "="*70)
print("FEATURE IMPORTANCE (Top 10 features)")
print("="*70)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'xgb_importance': xgb_model.feature_importances_,
    'lgb_importance': lgb_model.feature_importances_
})

feature_importance['avg_importance'] = (feature_importance['xgb_importance'] + feature_importance['lgb_importance']) / 2
feature_importance = feature_importance.sort_values('avg_importance', ascending=False).head(10)

print(feature_importance.to_string(index=False))

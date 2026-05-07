import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from arch import arch_model

class CrossSectionalModels:
    def __init__(self, model_type="MLR"):
        self.model_type = model_type
        
        # Existing Models
        if model_type == "MLR":
            self.model = LinearRegression()
        elif model_type == "RF":
            # n_estimators: Total number of independent trees to build
            self.model = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=318)
        elif model_type == "Lasso":
            self.model = LassoCV(cv=5, random_state=318)
        elif model_type == "Ridge":
            self.model = RidgeCV(cv=5)
        elif model_type == "DT":
            self.model = DecisionTreeRegressor(max_depth=6, random_state=318)
        elif model_type == "XGB":
            self.model = xgb.XGBRegressor(
                n_estimators=500, 
                learning_rate=0.05, 
                max_depth=6, 
                n_jobs=-1, 
                random_state=318
            )
        else:
            raise ValueError(f"Model type '{model_type}' is not recognized.")
            
    def fit(self, X, y):
        self.feature_names_ = X.columns.tolist()
        self.model.fit(X, y)
        
        extra_info = ""
        if self.model_type == "Lasso":
            extra_info = f" (Optimized L1 Alpha: {self.model.alpha_:.6f})"
        elif self.model_type == "Ridge":
            extra_info = f" (Optimized L2 Alpha: {self.model.alpha_:.6f})"
            
        print(f"Trained {self.model_type} on {len(self.feature_names_)} features{extra_info}.")
        
    def predict(self, X):
        return self.model.predict(X[self.feature_names_])

def run_garch_forecast(returns_sequence, horizon):
    try:
        am = arch_model(returns_sequence, vol='Garch', p=1, q=1, mean='Zero', rescale=True)
        res = am.fit(disp='off')
        forecasts = res.forecast(horizon=horizon)
        
        scale_factor = res.scale
        return np.mean(forecasts.variance.values[-1, :]) / (scale_factor**2)
    except Exception as e:
        return np.nan

def get_dummy_feature_matrix(n_buckets=100, n_features=178):
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X = np.random.rand(n_buckets, n_features)
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = np.random.uniform(0.0001, 0.005, n_buckets)
    return df

if __name__ == "__main__":
    print("||Running Operational Checks||")
    data_dummy = get_dummy_feature_matrix(n_buckets=200, n_features=20)
    X_d = data_dummy.drop('target', axis=1)
    y_d = data_dummy['target']
    
    models_to_test = ["MLR", "Lasso", "Ridge", "RF", "DT", "XGB"]
    
    for m_type in models_to_test:
        model_instance = CrossSectionalModels(model_type=m_type)
        model_instance.fit(X_d, y_d)
    
    dummy_ret = np.random.normal(0, 0.0001, 300)
    g_val = run_garch_forecast(dummy_ret, horizon=300)
    print(f"\nGARCH Convergence Check: Result = {g_val:.10f}")


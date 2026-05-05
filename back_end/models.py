
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from arch import arch_model

def get_dummy_feature_matrix(n_buckets=100, n_features=178):
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X = np.random.rand(n_buckets, n_features)
    df = pd.DataFrame(X, columns=feature_names)
    df['stock_id'] = np.random.randint(1, 20, n_buckets)
    df['time_id'] = np.random.randint(1000, 5000, n_buckets)
    df['target'] = np.random.uniform(0.0001, 0.005, n_buckets)
    return df

class CrossSectionalModels:
    def __init__(self, model_type="MLR"):
        self.model_type = model_type
        if model_type == "MLR":
            self.model = LinearRegression()
        elif model_type == "RF":
            self.model = RandomForestRegressor(n_estimators=100) # Reduced for speed in testing
        # To expand: add elif model_type == "Lasso": self.model = Lasso()
            
    def fit(self, X, y):
        # This handles Alpha or Gamma number of features automatically
        self.model.fit(X, y)
        print(f"Successfully trained {self.model_type} on {X.shape[1]} features.")
        
    def predict(self, X):
        return self.model.predict(X)

def run_garch_forecast(returns_sequence, horizon):
    try:
        am = arch_model(returns_sequence, vol='Garch', p=1, q=1, mean='Zero')
        res = am.fit(disp='off')
        forecasts = res.forecast(horizon=horizon)
        # Taking the last forecast variance and averaging across the horizon
        return np.mean(forecasts.variance.values[-1, :])
    except:
        # GARCH can fail to converge; return a null or average if it crashes
        return np.nan
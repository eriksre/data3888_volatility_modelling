from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from arch import arch_model

def make_models():
    return {
        "MLR": LinearRegression(),
        "Lasso": LassoCV(
            cv=5, random_state=42
            ),
        "Ridge": RidgeCV(
            cv=5
            ),
        "DT": DecisionTreeRegressor(
            max_depth=6, random_state=42
            ),
        "RF": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        ),
        # add new models here, they'll be picked up automatically
    }

def make_garch_models(returns_sequence, horizon=30, p=1, q=1, o=0, vol='Garch', dist='normal'):
    try:
        am = arch_model(
            returns_sequence, 
            vol=vol, 
            p=p, o=o, q=q, 
            dist=dist,
            mean='Zero', 
            rescale=True
        )
        res = am.fit(disp='off')
        forecasts = res.forecast(horizon=horizon)
        scale_factor = res.scale
        return np.mean(forecasts.variance.values[-1, :]) / (scale_factor**2)
    except Exception:
        return np.nan
    

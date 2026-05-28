from __future__ import annotations

import json
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from back_end.config import ModelSpec
from back_end.models import run_garch_on_processed


class _FakeForecast:
    def __init__(self, horizon: int, variance: float):
        self.variance = pd.DataFrame([np.full(horizon, variance)])


class _FakeFit:
    def __init__(self, variance: float, params: pd.Series):
        self._variance = variance
        self.params = params

    def forecast(self, horizon: int, reindex: bool, **kwargs):
        assert horizon == 30
        assert reindex is False
        return _FakeForecast(horizon, self._variance)


class _FakeArchModel:
    train_lengths: list[int] = []
    fit_calls: int = 0
    variance_override: float | None = None
    params_override: pd.Series | None = None

    def __init__(self, returns, **kwargs):
        self._returns = np.asarray(returns, dtype=float)
        self.train_lengths.append(len(self._returns))

    def fit(self, **kwargs):
        _FakeArchModel.fit_calls += 1
        variance = self.variance_override if self.variance_override is not None else float(len(self._returns))
        params = self.params_override
        if params is None:
            params = pd.Series({"omega": 0.1, "alpha[1]": 0.1, "beta[1]": 0.8})
        return _FakeFit(float(variance), params)


class GarchWalkForwardTest(unittest.TestCase):
    def setUp(self):
        self._old_arch = sys.modules.get("arch")
        fake_arch = types.ModuleType("arch")
        fake_arch.arch_model = _FakeArchModel
        sys.modules["arch"] = fake_arch
        _FakeArchModel.train_lengths = []
        _FakeArchModel.fit_calls = 0
        _FakeArchModel.variance_override = None
        _FakeArchModel.params_override = None

    def tearDown(self):
        if self._old_arch is None:
            sys.modules.pop("arch", None)
        else:
            sys.modules["arch"] = self._old_arch

    def test_garch_forecasts_last_30_usable_returns_without_heldout_leakage(self):
        processed = pd.DataFrame(
            {
                "stock_id": ["stock_7"] * 600,
                "time_id": [101] * 600,
                "seconds_in_bucket": np.arange(600),
                "log_price_diff": np.r_[np.nan, np.full(599, 0.0001)],
            }
        )
        spec = ModelSpec(name="GARCH(1,1)", model_type="GARCH(1,1)")

        result = run_garch_on_processed(processed, spec, horizon=30, fold=1, test_ids={101})

        self.assertEqual(len(result), 1)
        self.assertEqual(_FakeArchModel.train_lengths, [569])
        self.assertEqual(json.loads(result.loc[0, "forecast_seconds"]), list(range(570, 600)))
        self.assertEqual(len(json.loads(result.loc[0, "forecast_vol_path"])), 30)
        self.assertEqual(result.loc[0, "stock_id"], "stock_7")
        self.assertEqual(result.loc[0, "time_id"], 101)

    def test_garch_timing_measures_forecast_after_fit_only(self):
        processed = pd.DataFrame(
            {
                "stock_id": ["stock_7"] * 600,
                "time_id": [101] * 600,
                "seconds_in_bucket": np.arange(600),
                "log_price_diff": np.r_[np.nan, np.full(599, 0.0001)],
            }
        )
        spec = ModelSpec(name="GARCH(1,1)", model_type="GARCH(1,1)")
        fit_calls_at_timer = []

        def fake_perf_counter():
            fit_calls_at_timer.append(_FakeArchModel.fit_calls)
            return 10.0 + 0.001 * len(fit_calls_at_timer)

        with patch("back_end.models.time.perf_counter", side_effect=fake_perf_counter):
            result = run_garch_on_processed(processed, spec, horizon=30, fold=1, test_ids={101})

        self.assertEqual(fit_calls_at_timer, [1, 1])
        self.assertAlmostEqual(result.loc[0, "inference_ms"], 1.0)

    def test_garch_marks_explosive_forecasts_as_missing(self):
        _FakeArchModel.variance_override = 1e8
        processed = pd.DataFrame(
            {
                "stock_id": ["stock_7"] * 600,
                "time_id": [101] * 600,
                "seconds_in_bucket": np.arange(600),
                "log_price_diff": np.r_[np.nan, np.full(599, 0.0001)],
            }
        )
        spec = ModelSpec(name="GJR-GARCH(1,1)", model_type="GJR-GARCH(1,1)")

        result = run_garch_on_processed(processed, spec, horizon=30, fold=1, test_ids={101})

        self.assertEqual(len(result), 1)
        self.assertTrue(np.isnan(result.loc[0, "pred_var"]))
        self.assertTrue(np.isnan(result.loc[0, "pred_vol"]))
        self.assertEqual(json.loads(result.loc[0, "forecast_vol_path"]), [])

    def test_garch_marks_non_stationary_fits_as_missing(self):
        _FakeArchModel.params_override = pd.Series({"omega": 0.1, "alpha[1]": 0.6, "beta[1]": 0.5})
        processed = pd.DataFrame(
            {
                "stock_id": ["stock_7"] * 600,
                "time_id": [101] * 600,
                "seconds_in_bucket": np.arange(600),
                "log_price_diff": np.r_[np.nan, np.full(599, 0.0001)],
            }
        )
        spec = ModelSpec(name="GARCH(1,1)", model_type="GARCH(1,1)")

        result = run_garch_on_processed(processed, spec, horizon=30, fold=1, test_ids={101})

        self.assertEqual(len(result), 1)
        self.assertTrue(np.isnan(result.loc[0, "pred_var"]))
        self.assertTrue(np.isnan(result.loc[0, "pred_vol"]))
        self.assertEqual(json.loads(result.loc[0, "forecast_vol_path"]), [])

    def test_gjr_garch_marks_non_stationary_fits_as_missing(self):
        _FakeArchModel.params_override = pd.Series(
            {"omega": 0.0, "alpha[1]": 1.0, "gamma[1]": -1.0, "beta[1]": 1.0}
        )
        processed = pd.DataFrame(
            {
                "stock_id": ["stock_7"] * 600,
                "time_id": [101] * 600,
                "seconds_in_bucket": np.arange(600),
                "log_price_diff": np.r_[np.nan, np.full(599, 0.0001)],
            }
        )
        spec = ModelSpec(name="GJR-GARCH(1,1)", model_type="GJR-GARCH(1,1)")

        result = run_garch_on_processed(processed, spec, horizon=30, fold=1, test_ids={101})

        self.assertEqual(len(result), 1)
        self.assertTrue(np.isnan(result.loc[0, "pred_var"]))
        self.assertTrue(np.isnan(result.loc[0, "pred_vol"]))
        self.assertEqual(json.loads(result.loc[0, "forecast_vol_path"]), [])

    def test_garch_marks_boundary_parameter_fits_as_missing(self):
        _FakeArchModel.params_override = pd.Series({"omega": 0.1, "alpha[1]": 0.99, "beta[1]": 0.0})
        processed = pd.DataFrame(
            {
                "stock_id": ["stock_7"] * 600,
                "time_id": [101] * 600,
                "seconds_in_bucket": np.arange(600),
                "log_price_diff": np.r_[np.nan, np.full(599, 0.0001)],
            }
        )
        spec = ModelSpec(name="GARCH(1,1)", model_type="GARCH(1,1)")

        result = run_garch_on_processed(processed, spec, horizon=30, fold=1, test_ids={101})

        self.assertEqual(len(result), 1)
        self.assertTrue(np.isnan(result.loc[0, "pred_var"]))
        self.assertTrue(np.isnan(result.loc[0, "pred_vol"]))
        self.assertEqual(json.loads(result.loc[0, "forecast_vol_path"]), [])


if __name__ == "__main__":
    unittest.main()

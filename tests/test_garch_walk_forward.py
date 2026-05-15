from __future__ import annotations

import json
import sys
import types
import unittest

import numpy as np
import pandas as pd

from back_end.config import ModelSpec
from back_end.models import run_garch_on_processed


class _FakeForecast:
    def __init__(self, horizon: int, variance: float):
        self.variance = pd.DataFrame([np.full(horizon, variance)])


class _FakeFit:
    def __init__(self, variance: float):
        self._variance = variance

    def forecast(self, horizon: int, reindex: bool, **kwargs):
        assert horizon == 30
        assert reindex is False
        return _FakeForecast(horizon, self._variance)


class _FakeArchModel:
    train_lengths: list[int] = []

    def __init__(self, returns, **kwargs):
        self._returns = np.asarray(returns, dtype=float)
        self.train_lengths.append(len(self._returns))

    def fit(self, **kwargs):
        return _FakeFit(float(len(self._returns)))


class GarchWalkForwardTest(unittest.TestCase):
    def setUp(self):
        self._old_arch = sys.modules.get("arch")
        fake_arch = types.ModuleType("arch")
        fake_arch.arch_model = _FakeArchModel
        sys.modules["arch"] = fake_arch
        _FakeArchModel.train_lengths = []

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


if __name__ == "__main__":
    unittest.main()

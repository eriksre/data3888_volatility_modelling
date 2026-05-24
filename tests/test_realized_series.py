from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from back_end.service import build_prediction_curves, load_window_realized_series


class RealizedSeriesTest(unittest.TestCase):
    def test_window_realized_series_uses_rms_volatility_scale(self):
        returns = np.arange(1, 9, dtype=float)
        processed = pd.DataFrame(
            {
                "time_id": 7,
                "seconds_in_bucket": np.arange(len(returns)),
                "log_price_diff": returns / 10000,
            }
        )

        with patch("back_end.service.load_run_config", return_value={"features": {"forecast_horizon": 2}}), patch(
            "back_end.service.load_processed_stock", return_value=processed
        ):
            realized = load_window_realized_series("run-1", "stock_0", 7)

        first_observed = realized[realized["seconds_in_bucket"] == 4].iloc[0]
        first_heldout = realized[realized["seconds_in_bucket"] == 6].iloc[0]

        self.assertAlmostEqual(first_observed["realized_vol"], np.sqrt(np.mean(returns[:5] ** 2)))
        self.assertAlmostEqual(first_heldout["realized_vol"], np.sqrt(np.mean(returns[:7] ** 2)))
        self.assertEqual(first_heldout["segment"], "heldout_actual")

    def test_garch_prediction_curve_anchors_to_last_observed_realized_volatility(self):
        realized = pd.DataFrame(
            {
                "seconds_in_bucket": [0, 1, 2, 3],
                "realized_vol": [1.0, 1.2, 1.5, 1.7],
                "segment": ["observed", "observed", "heldout_actual", "heldout_actual"],
            }
        )
        predictions = pd.DataFrame(
            {
                "model": ["GARCH(1,1)"],
                "model_type": ["GARCH(1,1)"],
                "pred_vol": [1.65],
                "prediction_kind": ["garch_path"],
                "forecast_seconds": ["[2, 3]"],
                "forecast_vol_path": ["[1.6, 1.55]"],
            }
        )

        curves = build_prediction_curves(realized, predictions)

        self.assertEqual(curves["seconds_in_bucket"].tolist(), [1, 2, 3])
        self.assertEqual(curves["pred_vol"].tolist(), [1.2, 1.6, 1.55])
        self.assertEqual(curves["prediction_kind"].tolist(), ["garch_path", "garch_path", "garch_path"])


if __name__ == "__main__":
    unittest.main()

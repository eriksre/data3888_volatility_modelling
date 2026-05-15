from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from back_end.service import load_individual_model_metrics


class IndividualMetricsTest(unittest.TestCase):
    def test_individual_metrics_omits_mean_5_min_vol(self):
        metrics = pd.DataFrame(
            [
                {
                    "model": "Linear Regression",
                    "rmse": 0.2,
                    "qlike": 0.3,
                }
            ]
        )
        predictions = pd.DataFrame(
            [
                {
                    "model": "Linear Regression",
                    "inference_ms": 0.123,
                }
            ]
        )

        with patch("back_end.service.load_stock_metrics", return_value=metrics), patch(
            "back_end.service.prediction_series", return_value=predictions
        ):
            view = load_individual_model_metrics("run-1", "stock_1", include_scaffold=False)

        self.assertNotIn("pred_target", view.columns)
        self.assertAlmostEqual(view.loc[0, "inference_us"], 123.0)


if __name__ == "__main__":
    unittest.main()

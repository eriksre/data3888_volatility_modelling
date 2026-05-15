from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from back_end.config import FeatureConfig
from back_end.service import load_universe_page_data
from back_end.universe import LOSS_METRICS


class UniverseServiceTest(unittest.TestCase):
    def test_load_universe_page_data_backfills_missing_loss_metrics_from_predictions(self):
        summary = pd.DataFrame(
            [
                {
                    "stock_id": "stock_0",
                    "mean_volatility": 1.5,
                    "best_model": "Model A",
                    "rmse": 0.0,
                    "qlike": 0.0,
                }
            ]
        )
        predictions = pd.DataFrame(
            [
                {
                    "stock_id": "stock_0",
                    "time_id": 1,
                    "model": "Model A",
                    "model_type": "Linear Regression",
                    "pred_var": 1.0,
                    "actual_var": 1.0,
                    "inference_ms": 0.1,
                    "feature_cols": "PC1,PC2",
                },
                {
                    "stock_id": "stock_0",
                    "time_id": 2,
                    "model": "Model A",
                    "model_type": "Linear Regression",
                    "pred_var": 4.0,
                    "actual_var": 4.0,
                    "inference_ms": 0.1,
                    "feature_cols": "PC1,PC2",
                },
                {
                    "stock_id": "stock_0",
                    "time_id": 1,
                    "model": "Model B",
                    "model_type": "Random Forest",
                    "pred_var": 2.0,
                    "actual_var": 1.0,
                    "inference_ms": 0.2,
                    "feature_cols": "PC1,PC2",
                },
                {
                    "stock_id": "stock_0",
                    "time_id": 2,
                    "model": "Model B",
                    "model_type": "Random Forest",
                    "pred_var": 5.0,
                    "actual_var": 4.0,
                    "inference_ms": 0.2,
                    "feature_cols": "PC1,PC2",
                },
            ]
        )

        with patch("back_end.service.load_universe_summary", return_value=summary), patch(
            "back_end.service.load_similarity", return_value=pd.DataFrame()
        ), patch("back_end.service.load_predictions", return_value=predictions), patch(
            "back_end.service.load_metrics", return_value=pd.DataFrame()
        ), patch(
            "back_end.service._load_run_features", return_value=(pd.DataFrame(), FeatureConfig())
        ):
            summary_view, *_ = load_universe_page_data("run-1")

        for metric in LOSS_METRICS:
            self.assertIn(metric, summary_view.columns)
            self.assertEqual(summary_view.loc[0, metric], 0.0)


if __name__ == "__main__":
    unittest.main()

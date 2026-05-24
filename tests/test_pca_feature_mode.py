from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from back_end.config import FeatureConfig, ModelSpec
from back_end.evaluation import summarize_fold_metrics
from back_end.models import MIN_PRED_VOL, run_ml_model, run_model_for_fold
from back_end.universe import LOSS_METRICS, build_model_comparison, build_pca_variance_explained, build_stock_pca, build_universe_summary


class PcaFeatureModeTest(unittest.TestCase):
    def test_pca_feature_mode_records_component_features(self):
        rng = np.random.default_rng(42)
        n_train = 40
        n_test = 8
        feature_names = [f"feature_{idx}" for idx in range(6)]
        train_features = rng.normal(size=(n_train, len(feature_names)))
        test_features = rng.normal(size=(n_test, len(feature_names)))
        train_target = np.square(np.maximum(train_features[:, 0] + 0.5 * train_features[:, 1], 0.0))
        test_target = np.square(np.maximum(test_features[:, 0] + 0.5 * test_features[:, 1], 0.0))

        train = pd.DataFrame(train_features, columns=feature_names)
        train.insert(0, "target_vol", np.sqrt(train_target))
        train.insert(0, "target_var", train_target)
        train.insert(0, "time_id", np.arange(n_train))
        train.insert(0, "stock_id", "stock_0")

        test = pd.DataFrame(test_features, columns=feature_names)
        test.insert(0, "target_vol", np.sqrt(test_target))
        test.insert(0, "target_var", test_target)
        test.insert(0, "time_id", np.arange(n_train, n_train + n_test))
        test.insert(0, "stock_id", "stock_0")

        predictions, importance = run_model_for_fold(
            ModelSpec(
                name="Linear Regression",
                model_type="Linear Regression",
                feature_mode="PCA",
                n_pca_components=2,
            ),
            train,
            test,
            fold=1,
            inherited_feature_mode="PCA",
            inherited_n_pca_components=5,
            inherited_manual_features=(),
        )

        self.assertEqual(set(predictions["feature_cols"]), {"PC1,PC2"})
        self.assertEqual(set(importance["feature"]), {"PC1", "PC2"})
        self.assertTrue(np.isfinite(predictions["pred_vol"]).all())

    def test_ml_predictions_use_positive_volatility_floor(self):
        feature_cols = ["feature"]
        train = pd.DataFrame(
            {
                "stock_id": ["stock_0", "stock_0", "stock_0"],
                "time_id": [1, 2, 3],
                "target_var": [1.0, 0.25, 0.0],
                "target_vol": [1.0, 0.5, 0.0],
                "feature": [0.0, 0.5, 1.0],
            }
        )
        test = pd.DataFrame(
            {
                "stock_id": ["stock_0"],
                "time_id": [4],
                "target_var": [0.25],
                "target_vol": [0.5],
                "feature": [2.0],
            }
        )

        predictions, _ = run_ml_model(
            ModelSpec(name="Linear Regression", model_type="Linear Regression"),
            train,
            test,
            feature_cols,
            fold=1,
        )

        self.assertAlmostEqual(predictions.loc[0, "pred_vol"], MIN_PRED_VOL)
        self.assertAlmostEqual(predictions.loc[0, "pred_var"], MIN_PRED_VOL**2)

    def test_stock_pca_returns_one_coordinate_row_per_stock(self):
        rows = []
        for stock_idx in range(3):
            for time_id in range(5):
                base = stock_idx + time_id / 10
                rows.append(
                    {
                        "stock_id": f"stock_{stock_idx}",
                        "time_id": time_id,
                        "target_var": base + 0.1,
                        "target_vol": np.sqrt(base + 0.1),
                        "RV_30": base,
                        "RV_60": base * 2,
                        "mean_spread_30": 3 - base,
                    }
                )
        features = pd.DataFrame(rows)

        pca_df, explained = build_stock_pca(features, FeatureConfig())

        self.assertEqual(set(pca_df["stock_id"]), {"stock_0", "stock_1", "stock_2"})
        self.assertEqual(len(pca_df), 3)
        self.assertIn("PC1", pca_df.columns)
        self.assertIn("PC2", pca_df.columns)
        self.assertTrue(np.isfinite(pca_df[["PC1", "PC2"]].to_numpy()).all())
        self.assertGreaterEqual(sum(explained), 0.99)

    def test_pca_variance_explained_returns_ratio_per_component(self):
        rows = []
        for idx in range(12):
            rows.append(
                {
                    "stock_id": f"stock_{idx % 3}",
                    "time_id": idx,
                    "target_var": idx + 0.1,
                    "target_vol": np.sqrt(idx + 0.1),
                    "RV_30": float(idx),
                    "RV_60": float(idx * 2),
                    "mean_spread_30": float(12 - idx),
                }
            )
        variance = build_pca_variance_explained(pd.DataFrame(rows), FeatureConfig(), 2)

        self.assertEqual(variance["component"].tolist(), ["PC1", "PC2"])
        self.assertTrue(np.isfinite(variance["explained_variance_ratio"]).all())
        self.assertTrue((variance["explained_variance_ratio"] >= 0).all())
        self.assertLessEqual(float(variance["explained_variance_ratio"].sum()), 1.0)

    def test_model_comparison_counts_stock_level_winners(self):
        predictions = pd.DataFrame(
            [
                {"model": "Model A", "model_type": "Linear Regression", "fold": 1, "stock_id": "stock_0", "pred_var": 1.0, "actual_var": 1.0, "inference_ms": 0.10, "feature_cols": "PC1,PC2"},
                {"model": "Model A", "model_type": "Linear Regression", "fold": 1, "stock_id": "stock_1", "pred_var": 1.1, "actual_var": 1.0, "inference_ms": 0.20, "feature_cols": "PC1,PC2"},
                {"model": "Model B", "model_type": "Random Forest", "fold": 1, "stock_id": "stock_0", "pred_var": 2.0, "actual_var": 1.0, "inference_ms": 0.30, "feature_cols": "PC1,PC2,PC3"},
                {"model": "Model B", "model_type": "Random Forest", "fold": 1, "stock_id": "stock_1", "pred_var": 2.0, "actual_var": 1.0, "inference_ms": 0.40, "feature_cols": "PC1,PC2,PC3"},
            ]
        )
        metrics = summarize_fold_metrics(predictions)

        comparison = build_model_comparison(predictions, metrics)
        model_a = comparison[comparison["model"] == "Model A"].iloc[0]
        model_b = comparison[comparison["model"] == "Model B"].iloc[0]

        self.assertEqual(model_a["best_stocks_rmse"], 2)
        self.assertEqual(model_b["best_stocks_rmse"], 0)
        self.assertEqual(model_a["feature_set"], "2 PCs")
        self.assertEqual(model_b["feature_set"], "3 PCs")
        self.assertAlmostEqual(model_a["mean_inference_ms"], 0.15)

    def test_universe_summary_includes_all_loss_metrics_for_best_model(self):
        features = pd.DataFrame(
            [
                {"stock_id": "stock_0", "time_id": 1, "target_var": 1.0, "target_vol": 1.0},
                {"stock_id": "stock_0", "time_id": 2, "target_var": 4.0, "target_vol": 2.0},
            ]
        )
        predictions = pd.DataFrame(
            [
                {"stock_id": "stock_0", "time_id": 1, "model": "Model A", "pred_var": 1.0, "actual_var": 1.0},
                {"stock_id": "stock_0", "time_id": 2, "model": "Model A", "pred_var": 4.0, "actual_var": 4.0},
                {"stock_id": "stock_0", "time_id": 1, "model": "Model B", "pred_var": 2.0, "actual_var": 1.0},
                {"stock_id": "stock_0", "time_id": 2, "model": "Model B", "pred_var": 5.0, "actual_var": 4.0},
            ]
        )

        summary = build_universe_summary(features, predictions)

        self.assertEqual(summary.loc[0, "best_model"], "Model A")
        for metric in LOSS_METRICS:
            self.assertIn(metric, summary.columns)
            self.assertEqual(summary.loc[0, metric], 0.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import pandas as pd

from front_end.pages.universe import (
    _model_comparison_view,
    _parse_manual_stocks,
    _ranking_chart,
    _select_display_stocks,
    _stock_summary_view,
)


class UniverseViewTest(unittest.TestCase):
    def test_parse_manual_stocks_normalizes_deduplicates_and_reports_missing(self):
        selected, missing = _parse_manual_stocks("0, stock_2\nstock_2 99", ["stock_0", "stock_2", "stock_10"])

        self.assertEqual(selected, ["stock_0", "stock_2"])
        self.assertEqual(missing, ["stock_99"])

    def test_select_display_stocks_adds_manual_stocks_without_duplicates(self):
        ranked_df = pd.DataFrame(
            [
                {"stock_id": "stock_0", "rmse": 0.10},
                {"stock_id": "stock_1", "rmse": 0.20},
                {"stock_id": "stock_2", "rmse": 0.30},
                {"stock_id": "stock_3", "rmse": 0.40},
            ]
        )

        selected = _select_display_stocks(ranked_df, ["stock_1", "stock_3"], top_n=2)

        self.assertEqual(selected["stock_id"].tolist(), ["stock_0", "stock_1", "stock_3"])

    def test_stock_summary_view_orders_stocks_numerically_and_formats_names(self):
        summary_df = pd.DataFrame(
            [
                {"stock_id": "stock_10", "mean_volatility": 0.10, "rmse": 0.30, "qlike": 0.50, "best_model": "Model C"},
                {"stock_id": "stock_2", "mean_volatility": 0.20, "rmse": 0.20, "qlike": 0.40, "best_model": "Model B"},
                {"stock_id": "stock_1", "mean_volatility": 0.30, "rmse": 0.10, "qlike": 0.30, "best_model": "Model A"},
            ]
        )

        view = _stock_summary_view(summary_df)

        self.assertEqual(view["Stock"].tolist(), ["Stock 1", "Stock 2", "Stock 10"])
        self.assertNotIn("stock_id", view.columns)

    def test_ranking_chart_uses_formatted_stock_labels_in_given_numeric_order(self):
        top_df = pd.DataFrame(
            [
                {"stock_id": "stock_1", "mean_volatility": 0.30, "best_model": "Model A"},
                {"stock_id": "stock_2", "mean_volatility": 0.20, "best_model": "Model B"},
                {"stock_id": "stock_10", "mean_volatility": 0.10, "best_model": "Model C"},
            ]
        )

        fig = _ranking_chart(top_df, "mean_volatility", "Mean Volatility")

        self.assertEqual(fig.layout.xaxis.categoryarray, ("Stock 1", "Stock 2", "Stock 10"))
        labels_by_trace = [label for trace in fig.data for label in trace.x]
        self.assertEqual(labels_by_trace, ["Stock 1", "Stock 2", "Stock 10"])

    def test_model_comparison_view_uses_microseconds_and_omits_fold_std(self):
        model_df = pd.DataFrame(
            [
                {
                    "model": "Model A",
                    "model_type": "Linear Regression",
                    "mean_inference_ms": 0.123456,
                    "best_stocks_rmse": 2,
                    "fold_std_rmse": 0.111111,
                    "rmse": 0.222222,
                    "qlike": 0.333333,
                    "pearson_r": 0.444444,
                }
            ]
        )

        view = _model_comparison_view(model_df, "rmse")

        self.assertIn("Mean inference (μs)", view.columns)
        self.assertNotIn("RMSE fold std", view.columns)
        self.assertAlmostEqual(view.loc[0, "Mean inference (μs)"], 123.456)

if __name__ == "__main__":
    unittest.main()

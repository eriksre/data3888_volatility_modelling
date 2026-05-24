from __future__ import annotations

import unittest

import pandas as pd

from front_end.charts import realised_vs_predicted_scatter


class RealisedVsPredictedScatterTest(unittest.TestCase):
    def test_extreme_prediction_does_not_set_axis_range(self):
        predictions = pd.DataFrame(
            [
                {"model": "Stable", "actual_vol": 1.0, "pred_vol": 1.1},
                {"model": "Stable", "actual_vol": 2.0, "pred_vol": 1.8},
                {"model": "Stable", "actual_vol": 4.0, "pred_vol": 3.2},
                {"model": "Exploder", "actual_vol": 1.5, "pred_vol": 1e114},
            ]
        )

        fig = realised_vs_predicted_scatter("stock_0", predictions)

        self.assertEqual(fig.layout.height, 520)
        self.assertEqual(fig.layout.xaxis.range[0], 0.0)
        self.assertIsNone(fig.layout.yaxis.scaleanchor)
        self.assertLess(fig.layout.xaxis.range[1], 10.0)
        self.assertLess(fig.layout.yaxis.range[1], 10.0)
        self.assertFalse(any("extreme prediction" in annotation.text for annotation in fig.layout.annotations))
        marker_points = sum(len(trace.x) for trace in fig.data if trace.mode == "markers")
        self.assertEqual(marker_points, 3)


if __name__ == "__main__":
    unittest.main()

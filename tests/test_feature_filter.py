from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from back_end.config import FeatureConfig
from back_end.features import build_feature_frame


class FeatureFilterTest(unittest.TestCase):
    def test_build_feature_frame_filters_sparse_observed_and_target_windows(self):
        observed_with_19_movements = np.r_[np.ones(19), np.zeros(11)]
        observed_with_20_movements = np.r_[np.ones(20), np.zeros(10)]
        target_with_4_movements = np.r_[np.ones(4), np.zeros(26)]
        target_with_5_movements = np.r_[np.ones(5), np.zeros(25)]
        rows = []

        for time_id, returns in [
            (1, np.r_[observed_with_19_movements, target_with_5_movements]),
            (2, np.r_[observed_with_20_movements, target_with_4_movements]),
            (3, np.r_[observed_with_20_movements, target_with_5_movements]),
        ]:
            for second, log_return in enumerate(returns):
                rows.append(
                    {
                        "time_id": time_id,
                        "seconds_in_bucket": second,
                        "log_price_diff": log_return / 10000,
                        "bid_price1": 1.0,
                        "ask_price1": 1.1,
                        "bid_size1": 100.0,
                        "ask_size1": 100.0,
                        "bid_size2": 100.0,
                        "ask_size2": 100.0,
                    }
                )

        features = build_feature_frame(
            pd.DataFrame(rows),
            "stock_0",
            FeatureConfig(
                forecast_horizon=30,
                return_windows=(5, 30),
                acf_windows=(30,),
                book_windows=(30,),
                ewma_lambdas=(0.9,),
            ),
        )

        self.assertEqual(features["time_id"].tolist(), [3])
        self.assertAlmostEqual(features.loc[0, "target_var"], 5 / 30)


if __name__ == "__main__":
    unittest.main()

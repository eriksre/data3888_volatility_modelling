from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from back_end.config import DataConfig, FeatureConfig
from back_end.feature_cache import load_cached_features, write_feature_cache
from back_end.features import expanded_feature_cols


def _feature_frame(feature_config: FeatureConfig) -> pd.DataFrame:
    rows = []
    for stock in ("stock_0", "stock_1"):
        for time_id in (1, 2):
            row = {
                "stock_id": stock,
                "time_id": time_id,
                "target_var": float(time_id),
                "target_vol": float(time_id) ** 0.5,
            }
            row.update({col: float(time_id) for col in expanded_feature_cols(feature_config)})
            rows.append(row)
    return pd.DataFrame(rows)


class FeatureCacheWriteTest(unittest.TestCase):
    def test_live_features_write_cache_that_can_be_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "features_all_stocks.parquet"
            data_config = DataConfig(
                stocks=("stock_0", "stock_1"),
                source_dir=tmp,
                feature_cache_path=str(cache_path),
            )
            feature_config = FeatureConfig()

            written = write_feature_cache(data_config, feature_config, _feature_frame(feature_config))
            cached, reason = load_cached_features(data_config, feature_config)

        self.assertEqual(written, str(cache_path))
        self.assertEqual(reason, str(cache_path))
        self.assertIsNotNone(cached)
        self.assertEqual(len(cached), 4)

    def test_limited_cache_is_not_used_for_full_data_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "features_all_stocks.parquet"
            feature_config = FeatureConfig()
            limited_config = DataConfig(
                stocks=("stock_0",),
                source_dir=tmp,
                max_time_ids_per_stock=1,
                feature_cache_path=str(cache_path),
            )
            full_config = DataConfig(
                stocks=("stock_0",),
                source_dir=tmp,
                feature_cache_path=str(cache_path),
            )

            write_feature_cache(limited_config, feature_config, _feature_frame(feature_config))
            cached, reason = load_cached_features(full_config, feature_config)

        self.assertIsNone(cached)
        self.assertEqual(reason, "time-id limit does not match cache manifest")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from unittest.mock import patch

from back_end.config import DataConfig
from back_end.pipeline import _resolved_feature_workers


class ParallelFeatureTest(unittest.TestCase):
    def test_default_feature_workers_match_precompute_default_but_cap_to_stock_count(self):
        with patch("back_end.pipeline.os.cpu_count", return_value=8):
            workers = _resolved_feature_workers(DataConfig(feature_workers=None), n_stocks=3)

        self.assertEqual(workers, 3)

    def test_explicit_feature_workers_support_single_worker_deterministic_runs(self):
        workers = _resolved_feature_workers(DataConfig(feature_workers=1), n_stocks=3)

        self.assertEqual(workers, 1)


if __name__ == "__main__":
    unittest.main()

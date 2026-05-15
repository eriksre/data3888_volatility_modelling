from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "front_end"))

from front_end.pages.individual import _sort_stock_options, _sort_time_options


class IndividualSelectorOrderingTest(unittest.TestCase):
    def test_stock_selector_orders_by_stock_number(self):
        stocks = ["stock_1", "stock_10", "stock_2", "stock_100", "stock_11"]

        ordered = _sort_stock_options(stocks)

        self.assertEqual(ordered, ["stock_1", "stock_2", "stock_10", "stock_11", "stock_100"])

    def test_time_selector_orders_ascending(self):
        time_ids = [10, 2, 1, 20, 11]

        ordered = _sort_time_options(time_ids)

        self.assertEqual(ordered, [1, 2, 10, 11, 20])


if __name__ == "__main__":
    unittest.main()

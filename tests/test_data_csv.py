from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from back_end.config import DataConfig
from back_end.data import BOOK_COLUMNS, list_available_stocks, load_raw_stock, stock_path


class CsvStockDataTest(unittest.TestCase):
    def test_stock_discovery_and_loading_use_csv_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            frame = pd.DataFrame(
                [
                    {
                        "time_id": 1,
                        "seconds_in_bucket": 0,
                        "bid_price1": 10.0,
                        "ask_price1": 10.1,
                        "bid_price2": 9.9,
                        "ask_price2": 10.2,
                        "bid_size1": 100,
                        "ask_size1": 120,
                        "bid_size2": 80,
                        "ask_size2": 90,
                        "stock_id": 0,
                    }
                ]
            )
            frame.to_csv(source / "stock_0.csv", index=False)

            self.assertEqual(list_available_stocks(source), ["stock_0"])
            self.assertEqual(stock_path("stock_0", source), source / "stock_0.csv")

            loaded = load_raw_stock("stock_0", DataConfig(source_dir=str(source)))

        self.assertEqual(loaded.columns.tolist(), BOOK_COLUMNS)
        self.assertEqual(len(loaded), 1)


if __name__ == "__main__":
    unittest.main()

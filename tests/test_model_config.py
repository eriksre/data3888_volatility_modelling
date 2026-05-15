from __future__ import annotations

import unittest
from dataclasses import asdict

from back_end.config import model_spec_from_ui


class ModelConfigTest(unittest.TestCase):
    def test_model_spec_from_ui_drops_legacy_custom_loss_payload(self):
        entry = {
            "type": "Linear Regression",
            "feature_mode": "PCA",
            "features": [0, 1, 2],
            "losses": ["RMSE", "Unsafe Custom Loss"],
            "custom_loss": {"name": "Unsafe Custom Loss", "expr": "lambda y, yhat: y - yhat"},
        }

        spec = model_spec_from_ui(entry)
        payload = asdict(spec)

        self.assertNotIn("metrics", payload)
        self.assertNotIn("custom_loss", payload)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from dataclasses import asdict

from back_end.config import SUPPORTED_MODEL_TYPES, model_spec_from_ui
from back_end.models import model_availability_issue


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

    def test_egarch_is_no_longer_supported_or_alias_canonicalized(self):
        spec = model_spec_from_ui({"type": "EGARCH"})

        self.assertNotIn("EGARCH(1,1)", SUPPORTED_MODEL_TYPES)
        self.assertEqual(spec.model_type, "EGARCH")
        self.assertEqual(model_availability_issue(spec.model_type), "EGARCH is not a supported backend model type.")


if __name__ == "__main__":
    unittest.main()

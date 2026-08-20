import os
import unittest
from unittest.mock import patch

from app.core.config import Settings


class SettingsTest(unittest.TestCase):
    def test_runtime_defaults_to_legacy(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()

        self.assertEqual("legacy", settings.agent_runtime_mode)
        self.assertFalse(settings.is_v2_runtime_enabled)
        self.assertFalse(settings.is_v2_shadow_enabled)
        self.assertEqual("2.0", settings.agent_contract_version)

    def test_v2_runtime_is_explicit(self):
        with patch.dict(
            os.environ,
            {
                "AGENT_RUNTIME_MODE": "V2",
                "AGENT_CONTRACT_VERSION": "2.1",
            },
            clear=True,
        ):
            settings = Settings()

        self.assertEqual("v2", settings.agent_runtime_mode)
        self.assertTrue(settings.is_v2_runtime_enabled)
        self.assertEqual("2.1", settings.agent_contract_version)

    def test_rejects_unknown_runtime_mode(self):
        with patch.dict(
            os.environ,
            {"AGENT_RUNTIME_MODE": "experimental"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "AGENT_RUNTIME_MODE"):
                Settings()


if __name__ == "__main__":
    unittest.main()

import unittest

from app.services.openai_client import _response_format


class OpenAiClientTest(unittest.TestCase):
    def test_uses_json_object_without_a_schema(self):
        self.assertEqual(
            _response_format(None, "unused"),
            {"type": "json_object"},
        )

    def test_uses_supplied_json_schema_for_structured_output(self):
        schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        }

        response_format = _response_format(schema, "agent_turn_response_v2")

        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["name"], "agent_turn_response_v2")
        self.assertEqual(response_format["schema"], schema)
        self.assertFalse(response_format["strict"])


if __name__ == "__main__":
    unittest.main()

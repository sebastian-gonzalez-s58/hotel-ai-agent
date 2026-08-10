import unittest

from app.agents.hotel_graphs import generate_room_service_menu_response


class HotelGraphsTest(unittest.TestCase):
    def test_room_service_menu_response_is_plain_text_with_url(self):
        response = generate_room_service_menu_response(
            history=[{"role": "guest", "content": "Muelle 1"}],
            known_context={"language": "es-MX"},
        )

        self.assertIn("https://hotelcristalino.menudigitalonline.com/", response["message"])
        self.assertIn("cantidades", response["message"])
        self.assertIsNone(response["interaction"])


if __name__ == "__main__":
    unittest.main()

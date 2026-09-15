import unittest

from app import app


class LiveTankPageTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index_ok_live_tank(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Live Tank", html)
        self.assertIn("Tracking readout", html)
        self.assertIn('id="stats"', html)

    def test_index_has_stats_and_api_links(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("Fish in view", html)
        self.assertIn("Avg movement", html)
        self.assertIn('data-metric="view"', html)
        self.assertIn('data-metric="movement"', html)
        self.assertIn('class="dock"', html)
        self.assertIn('href="#stats"', html)
        self.assertIn('id="activity"', html)
        self.assertIn('id="log"', html)
        self.assertIn("Movement vs activity", html)
        self.assertIn("Per-fish comparison", html)
        self.assertIn("/api/dashboard", html)
        self.assertIn("/api/stats", html)
        self.assertIn("/api/tracks", html)
        self.assertIn("/api/activity", html)
        self.assertNotIn("View Live Feed", html)
        self.assertNotIn('id="tankFeed"', html)

    def test_index_omits_aquaguard(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertNotIn("AquaGuard", html)
        self.assertNotIn("aquaguard", html.lower())

    def test_predict_routes_gone(self):
        self.assertEqual(self.client.get("/api/predict_latest").status_code, 404)
        self.assertEqual(self.client.post("/api/predict_manual").status_code, 404)


class TrackingApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_dashboard_payload(self):
        res = self.client.get("/api/dashboard")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["source"], "placeholder")
        self.assertIn("updated_at", data)
        self.assertEqual(len(data["cards"]), 5)
        self.assertEqual({card["id"] for card in data["cards"]}, {
            "view", "movement", "peak", "confidence", "tracks"
        })
        self.assertIn("movement", data["metrics"])
        self.assertGreaterEqual(len(data["tracks"]), 1)
        self.assertIn("values", data["comparison"]["movement"])
        self.assertIn("values", data["comparison"]["activity"])
        self.assertGreaterEqual(len(data["comparison"]["by_fish"]), 1)

    def test_stats_cards(self):
        data = self.client.get("/api/stats").get_json()
        view = next(card for card in data["cards"] if card["id"] == "view")
        self.assertEqual(view["value"], 12)
        self.assertEqual(view["display"], "12")

    def test_metric_by_id(self):
        res = self.client.get("/api/metrics/movement")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["id"], "movement")
        self.assertEqual(data["value"], "4.6 cm/s")
        self.assertEqual(len(data["series"]), 12)

    def test_unknown_metric(self):
        res = self.client.get("/api/metrics/not-a-metric")
        self.assertEqual(res.status_code, 404)

    def test_tracks_and_activity(self):
        tracks = self.client.get("/api/tracks").get_json()["tracks"]
        self.assertEqual(tracks[0]["id"], "Fish 03")
        self.assertIn("activity", tracks[0])
        activity = self.client.get("/api/activity").get_json()["comparison"]
        self.assertEqual(len(activity["movement"]["values"]), len(activity["activity"]["values"]))
        self.assertEqual(activity["by_fish"][0]["id"], "Fish 03")

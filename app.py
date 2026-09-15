from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os

import tracking

app = Flask(__name__, static_folder=".")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(".", "live_tracking.html")


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory("assets", filename)


@app.route("/api/dashboard")
def api_dashboard():
    """Full page payload. Frontend loads this once on boot."""
    return jsonify(tracking.get_snapshot())


@app.route("/api/stats")
def api_stats():
    """KPI cards. Swap tracking.get_stats() for live model scores."""
    return jsonify(tracking.get_stats())


@app.route("/api/metrics")
def api_metrics():
    return jsonify(tracking.get_metrics())


@app.route("/api/metrics/<metric_id>")
def api_metric(metric_id):
    metric = tracking.get_metric(metric_id)
    if metric is None:
        return jsonify({"error": "unknown metric", "id": metric_id}), 404
    return jsonify({"id": metric_id, **metric})


@app.route("/api/tracks")
def api_tracks():
    """Per-fish log rows."""
    return jsonify(tracking.get_tracks())


@app.route("/api/activity")
def api_activity():
    """Movement vs activity comparison series."""
    return jsonify(tracking.get_activity())


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        use_reloader=False,
    )

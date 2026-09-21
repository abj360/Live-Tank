"""Live Tank: the tracking dashboard and the live camera, in one Flask app.

Routes
  /                     the dashboard site (live_tracking.html)
  /live                 full-screen camera view with the tracking overlay
  /api/video.mjpg       annotated camera stream
  /api/mask.mjpg        detector's foreground mask
  /api/vision           tracker state for the live panel
  /api/layers           POST: turn overlay layers on and off
  /api/relearn          POST: relearn the empty tank background
  /api/dashboard        whole dashboard payload (live, else placeholder)
  /api/stats            KPI cards
  /api/metrics[/<id>]   chart series per KPI
  /api/tracks           per-fish rows
  /api/activity         movement vs activity comparison
"""
import logging
import os

from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

import tracking
from live_tank.service import service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = Flask(__name__, template_folder="live_tank/web/templates",
            static_folder="live_tank/web/static", static_url_path="/static")
CORS(app)

# Warm the camera up at boot so the first visitor sees live numbers, not the
# sample data. No-op when no camera is configured or LIVE_TANK_OFFLINE=1.
service.start()


# ---- the site ---------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(".", "live_tracking.html")


@app.route("/live")
def live_view():
    return render_template("live.html", label=service.settings.label)


# ---- live camera ------------------------------------------------------------
def _stream(mask):
    return Response(service.frames(mask=mask),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store"})


@app.route("/api/video.mjpg")
def api_video():
    return _stream(mask=False)


@app.route("/api/mask.mjpg")
def api_mask():
    return _stream(mask=True)


@app.route("/api/vision")
def api_vision():
    return jsonify(service.state())


@app.route("/api/layers", methods=["POST"])
def api_layers():
    for name, on in (request.get_json(force=True, silent=True) or {}).items():
        service.set_layer(name, on)
    return jsonify(service.state().get("layers", {}))


@app.route("/api/relearn", methods=["POST"])
def api_relearn():
    service.relearn()
    return jsonify({"ok": True})


# ---- dashboard data ---------------------------------------------------------
@app.route("/api/dashboard")
def api_dashboard():
    """Full page payload. The frontend loads this once on boot."""
    return jsonify(tracking.get_snapshot())


@app.route("/api/stats")
def api_stats():
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
    return jsonify(tracking.get_tracks())


@app.route("/api/activity")
def api_activity():
    return jsonify(tracking.get_activity())


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 5000)),
        threaded=True,
        use_reloader=False,
    )

"""Dashboard payload for the Live Tank site.

Served live from the tank camera when one is reachable (see live_tank/), and
from the placeholder below otherwise, so the site still runs on a host that
cannot see the tank. The keys are identical either way, so the page stays on
/api/dashboard, /api/stats, /api/metrics/<id>, /api/tracks and /api/activity.

  cards[].value / spark     <- counts, speeds, confidence, identities
  metrics[id].series        <- 12-point window for the selected KPI chart
  tracks[]                  <- per-fish rows for the log table
  comparison.movement       <- tank-wide avg speed over the window
  comparison.activity       <- share of fish that are swimming
  comparison.by_fish        <- per-fish movement vs activity bars
"""

import logging
from copy import deepcopy

log = logging.getLogger(__name__)

PLACEHOLDER_SOURCE = "placeholder"

_SNAPSHOT = {
    "source": PLACEHOLDER_SOURCE,
    "updated_at": "2026-09-15T12:00:00Z",
    "updated_label": "Sep 15, 2026 · 12:00",
    "cards": [
        {
            "id": "view",
            "label": "Fish in view",
            "value": 12,
            "display": "12",
            "unit": "",
            "trend_pct": 18.2,
            "trend_dir": "up",
            "spark": [7, 8, 7, 9, 8, 10, 9, 11, 10, 11, 12, 12],
        },
        {
            "id": "movement",
            "label": "Avg movement",
            "value": 4.6,
            "display": "4.6",
            "unit": "cm/s",
            "trend_pct": 12.4,
            "trend_dir": "up",
            "spark": [2.8, 3.1, 3.0, 3.4, 3.6, 4.0, 3.8, 4.2, 4.1, 4.4, 4.5, 4.6],
        },
        {
            "id": "peak",
            "label": "Peak speed",
            "value": 7.2,
            "display": "7.2",
            "unit": "cm/s",
            "trend_pct": 28.6,
            "trend_dir": "up",
            "spark": [4.1, 4.4, 4.8, 5.2, 5.0, 5.8, 6.1, 6.4, 6.2, 6.9, 7.0, 7.2],
        },
        {
            "id": "confidence",
            "label": "Detection confidence",
            "value": 94,
            "display": "94",
            "unit": "%",
            "trend_pct": 2.1,
            "trend_dir": "up",
            "spark": [86, 87, 88, 88, 90, 91, 90, 92, 93, 93, 94, 94],
        },
        {
            "id": "tracks",
            "label": "Unique tracks",
            "value": 11,
            "display": "11",
            "unit": "",
            "trend_pct": 10.0,
            "trend_dir": "up",
            "spark": [6, 7, 7, 8, 8, 9, 9, 10, 10, 10, 11, 11],
        },
    ],
    "metrics": {
        "view": {
            "title": "Fish in view",
            "value": "12",
            "meta": "count, last 30 min",
            "trend_pct": 18.2,
            "trend_dir": "up",
            "series": [7, 8, 7, 9, 8, 10, 9, 11, 10, 11, 12, 12],
            "axis": ["12:00", "12:10", "12:20", "12:30"],
            "detail_title": "Most often in frame",
            "details": [
                {"name": "Fish 03", "sub": "Mid-water", "value": "11 min", "trend": "↑ 18.2%"},
                {"name": "Fish 07", "sub": "Surface", "value": "9 min", "trend": "↑ 14.1%"},
                {"name": "Fish 01", "sub": "Near glass", "value": "8 min", "trend": "↑ 9.4%"},
                {"name": "Fish 11", "sub": "Bottom", "value": "6 min", "trend": "↑ 4.8%"},
            ],
        },
        "movement": {
            "title": "Movement overview",
            "value": "4.6 cm/s",
            "meta": "avg speed, last 30 min",
            "trend_pct": 12.4,
            "trend_dir": "up",
            "series": [2.8, 3.1, 3.0, 3.4, 3.6, 4.0, 3.8, 4.2, 4.1, 4.4, 4.5, 4.6],
            "axis": ["12:00", "12:10", "12:20", "12:30"],
            "detail_title": "Most active in view",
            "details": [
                {"name": "Fish 03", "sub": "Mid-water", "value": "7.2 cm/s", "trend": "↑ 28.6%"},
                {"name": "Fish 07", "sub": "Surface", "value": "6.1 cm/s", "trend": "↑ 21.4%"},
                {"name": "Fish 01", "sub": "Near glass", "value": "5.4 cm/s", "trend": "↑ 18.7%"},
                {"name": "Fish 11", "sub": "Bottom", "value": "3.8 cm/s", "trend": "↑ 9.2%"},
            ],
        },
        "peak": {
            "title": "Peak speed",
            "value": "7.2 cm/s",
            "meta": "fastest burst, last 30 min",
            "trend_pct": 28.6,
            "trend_dir": "up",
            "series": [4.1, 4.4, 4.8, 5.2, 5.0, 5.8, 6.1, 6.4, 6.2, 6.9, 7.0, 7.2],
            "axis": ["12:00", "12:10", "12:20", "12:30"],
            "detail_title": "Fastest bursts",
            "details": [
                {"name": "Fish 03", "sub": "Mid-water", "value": "7.2 cm/s", "trend": "↑ 28.6%"},
                {"name": "Fish 07", "sub": "Surface", "value": "6.8 cm/s", "trend": "↑ 19.0%"},
                {"name": "Fish 01", "sub": "Near glass", "value": "6.1 cm/s", "trend": "↑ 11.4%"},
                {"name": "Fish 09", "sub": "Near glass", "value": "5.5 cm/s", "trend": "↑ 8.2%"},
            ],
        },
        "confidence": {
            "title": "Detection confidence",
            "value": "94%",
            "meta": "model score, last 30 min",
            "trend_pct": 2.1,
            "trend_dir": "up",
            "series": [86, 87, 88, 88, 90, 91, 90, 92, 93, 93, 94, 94],
            "axis": ["12:00", "12:10", "12:20", "12:30"],
            "detail_title": "Highest-confidence IDs",
            "details": [
                {"name": "Fish 01", "sub": "Near glass", "value": "98%", "trend": "↑ 1.2%"},
                {"name": "Fish 03", "sub": "Mid-water", "value": "96%", "trend": "↑ 2.4%"},
                {"name": "Fish 07", "sub": "Surface", "value": "93%", "trend": "↑ 0.8%"},
                {"name": "Fish 11", "sub": "Bottom", "value": "89%", "trend": "↑ 3.1%"},
            ],
        },
        "tracks": {
            "title": "Unique tracks",
            "value": "11",
            "meta": "IDs held this session",
            "trend_pct": 10.0,
            "trend_dir": "up",
            "series": [6, 7, 7, 8, 8, 9, 9, 10, 10, 10, 11, 11],
            "axis": ["12:00", "12:10", "12:20", "12:30"],
            "detail_title": "Longest-held IDs",
            "details": [
                {"name": "Fish 02", "sub": "Bottom", "value": "12.6 min", "trend": "↑ 4.0%"},
                {"name": "Fish 03", "sub": "Mid-water", "value": "11.2 min", "trend": "↑ 8.1%"},
                {"name": "Fish 07", "sub": "Surface", "value": "9.4 min", "trend": "↑ 6.2%"},
                {"name": "Fish 01", "sub": "Near glass", "value": "8.1 min", "trend": "↑ 3.3%"},
            ],
        },
    },
    "tracks": [
        {
            "id": "Fish 03",
            "zone": "Mid-water",
            "speed": 7.2,
            "speed_unit": "cm/s",
            "trend_pct": 28.6,
            "trend_dir": "up",
            "heading": 42,
            "confidence": 96,
            "dwell_min": 11.2,
            "status": "swimming",
            "last_seen": "12:00:04",
            "activity": 92,
        },
        {
            "id": "Fish 07",
            "zone": "Surface",
            "speed": 6.1,
            "speed_unit": "cm/s",
            "trend_pct": 21.4,
            "trend_dir": "up",
            "heading": 118,
            "confidence": 93,
            "dwell_min": 9.4,
            "status": "surface",
            "last_seen": "12:00:03",
            "activity": 88,
        },
        {
            "id": "Fish 01",
            "zone": "Near glass",
            "speed": 5.4,
            "speed_unit": "cm/s",
            "trend_pct": 18.7,
            "trend_dir": "up",
            "heading": 265,
            "confidence": 98,
            "dwell_min": 8.1,
            "status": "swimming",
            "last_seen": "12:00:03",
            "activity": 81,
        },
        {
            "id": "Fish 11",
            "zone": "Bottom",
            "speed": 3.8,
            "speed_unit": "cm/s",
            "trend_pct": 9.2,
            "trend_dir": "up",
            "heading": 310,
            "confidence": 89,
            "dwell_min": 6.0,
            "status": "swimming",
            "last_seen": "12:00:02",
            "activity": 64,
        },
        {
            "id": "Fish 04",
            "zone": "Mid-water",
            "speed": 2.1,
            "speed_unit": "cm/s",
            "trend_pct": 4.1,
            "trend_dir": "down",
            "heading": 188,
            "confidence": 91,
            "dwell_min": 4.7,
            "status": "idle",
            "last_seen": "11:59:58",
            "activity": 28,
        },
        {
            "id": "Fish 09",
            "zone": "Near glass",
            "speed": 4.0,
            "speed_unit": "cm/s",
            "trend_pct": 6.8,
            "trend_dir": "up",
            "heading": 74,
            "confidence": 87,
            "dwell_min": 3.9,
            "status": "swimming",
            "last_seen": "11:59:51",
            "activity": 71,
        },
        {
            "id": "Fish 02",
            "zone": "Bottom",
            "speed": 1.4,
            "speed_unit": "cm/s",
            "trend_pct": 12.0,
            "trend_dir": "down",
            "heading": 21,
            "confidence": 84,
            "dwell_min": 12.6,
            "status": "idle",
            "last_seen": "11:59:44",
            "activity": 18,
        },
        {
            "id": "Fish 08",
            "zone": "Surface",
            "speed": 5.0,
            "speed_unit": "cm/s",
            "trend_pct": 3.3,
            "trend_dir": "up",
            "heading": 350,
            "confidence": 90,
            "dwell_min": 2.8,
            "status": "surface",
            "last_seen": "11:59:40",
            "activity": 76,
        },
    ],
    "comparison": {
        "title": "Movement vs activity",
        "axis": ["12:00", "12:10", "12:20", "12:30"],
        "movement": {
            "label": "Avg movement",
            "unit": "cm/s",
            "values": [2.8, 3.1, 3.0, 3.4, 3.6, 4.0, 3.8, 4.2, 4.1, 4.4, 4.5, 4.6],
        },
        "activity": {
            "label": "Activity index",
            "unit": "%",
            "values": [48, 52, 50, 58, 61, 70, 66, 74, 71, 79, 84, 88],
        },
        "by_fish": [
            {"id": "Fish 03", "zone": "Mid-water", "movement": 7.2, "activity": 92},
            {"id": "Fish 07", "zone": "Surface", "movement": 6.1, "activity": 88},
            {"id": "Fish 01", "zone": "Near glass", "movement": 5.4, "activity": 81},
            {"id": "Fish 08", "zone": "Surface", "movement": 5.0, "activity": 76},
            {"id": "Fish 09", "zone": "Near glass", "movement": 4.0, "activity": 71},
            {"id": "Fish 11", "zone": "Bottom", "movement": 3.8, "activity": 64},
            {"id": "Fish 04", "zone": "Mid-water", "movement": 2.1, "activity": 28},
            {"id": "Fish 02", "zone": "Bottom", "movement": 1.4, "activity": 18},
        ],
    },
}


def get_snapshot():
    """Live payload: from a remote tracker, from a local one, else the sample."""
    import remote

    if remote.enabled():
        data = remote.fetch_json("/api/dashboard")
        if data:
            return data
    else:
        try:
            from live_tank import dashboard

            if dashboard.is_live():
                return dashboard.snapshot()
        except Exception:  # no OpenCV, no camera, nothing tracked yet
            log.exception("Live dashboard unavailable; serving placeholder")
    return deepcopy(_SNAPSHOT)


def get_stats():
    data = get_snapshot()
    return {
        "source": data["source"],
        "updated_at": data["updated_at"],
        "updated_label": data["updated_label"],
        "cards": data["cards"],
    }


def get_metrics():
    data = get_snapshot()
    return {
        "source": data["source"],
        "updated_at": data["updated_at"],
        "metrics": data["metrics"],
    }


def get_metric(metric_id):
    metrics = get_snapshot()["metrics"]
    return metrics.get(metric_id)


def get_tracks():
    data = get_snapshot()
    return {
        "source": data["source"],
        "updated_at": data["updated_at"],
        "tracks": data["tracks"],
    }


def get_activity():
    data = get_snapshot()
    return {
        "source": data["source"],
        "updated_at": data["updated_at"],
        "comparison": data["comparison"],
    }

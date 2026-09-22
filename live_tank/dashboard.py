"""Turns live tracker output into the dashboard payload the page reads.

Same shape as the placeholder in tracking.py: cards, metrics, tracks and the
movement/activity comparison.
"""
import time
from datetime import datetime, timezone

from .service import HISTORY_POINTS, service

ZONES = (("Surface", 0.33), ("Mid-water", 0.66), ("Bottom", 1.01))
IDLE_SPEED = 20            # source pixels per second below which a fish counts as idle
AXIS_POINTS = 4


def zone_of(track):
    y = track["pos"][1]
    return next(name for name, limit in ZONES if y < limit)


def trend(series):
    """Percent change from the start of the window to now."""
    points = [p for p in series if p is not None]
    if len(points) < 2 or not points[0]:
        return 0.0, "up"
    change = (points[-1] - points[0]) / abs(points[0]) * 100
    return round(abs(change), 1), ("up" if change >= 0 else "down")


def axis_labels(now=None):
    """Clock labels across the window covered by the charts."""
    now = now or time.time()
    span = HISTORY_POINTS * 10
    return [datetime.fromtimestamp(now - span + i * span / (AXIS_POINTS - 1)).strftime("%H:%M")
            for i in range(AXIS_POINTS)]


def is_live():
    return service.available


def snapshot():
    """Live dashboard payload; assumes `is_live()` was true."""
    state = service.state()
    settings = service.settings
    unit = settings.speed_unit
    now = time.time()
    axis = axis_labels(now)

    fish = [_fish_row(track, settings) for track in state["tracks"]]
    fish.sort(key=lambda f: f["speed"], reverse=True)
    counts = state["counts"]

    series = {key: service.series(key) for key in ("view", "movement", "peak", "confidence", "tracks", "activity")}
    values = card_values(fish, counts, settings)
    labels = {"view": (f"{settings.label} in view", "", "count"),
              "movement": ("Avg movement", unit, "avg speed"),
              "peak": ("Peak speed", unit, "fastest fish"),
              "confidence": ("Detection confidence", "%", "detector score"),
              "tracks": ("Fish identified", "", "IDs held this session")}

    cards, metrics = [], {}
    for key, (label, card_unit, meaning) in labels.items():
        pct, direction = trend(series[key])
        cards.append({"id": key, "label": label, "value": values[key], "display": f"{values[key]}",
                      "unit": card_unit, "trend_pct": pct, "trend_dir": direction, "spark": series[key]})
        metrics[key] = {
            "title": label,
            "value": f"{values[key]}{(' ' + card_unit) if card_unit and card_unit != '%' else card_unit}",
            "meta": f"{meaning}, last {HISTORY_POINTS * 10 // 60} min",
            "trend_pct": pct, "trend_dir": direction, "series": series[key], "axis": axis,
            "detail_title": _detail_title(key, settings.label),
            "details": _details(key, fish, unit),
        }

    return {
        "source": "live",
        "updated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
        "updated_label": datetime.fromtimestamp(now).strftime("%b %-d, %Y · %H:%M"),
        "camera": {"name": state["source"]["name"], "status": state["source"]["status"],
                   "fps": state["vision"]["fps"], "phase": state["vision"]["phase"]},
        "cards": cards,
        "metrics": metrics,
        "tracks": [{k: v for k, v in f.items() if k != "speed_px"} for f in fish],
        "comparison": {
            "title": "Movement vs activity",
            "axis": axis,
            "movement": {"label": "Avg movement", "unit": unit, "values": series["movement"]},
            "activity": {"label": "Activity index", "unit": "%", "values": series["activity"]},
            "by_fish": [{"id": f["id"], "zone": f["zone"], "movement": f["speed"], "activity": f["activity"]}
                        for f in fish],
        },
    }


def card_values(fish, counts, settings):
    """The five KPI numbers. `fish` rows carry speeds in source px/s
    (`speed_px`), display speeds (`speed`) and confidence already in percent."""
    if not fish:
        return {"view": counts["onscreen"], "movement": 0, "peak": 0,
                "confidence": 0, "tracks": counts["known"]}
    return {
        "view": counts["onscreen"],
        "movement": settings.to_speed(sum(f["speed_px"] for f in fish) / len(fish)),
        "peak": max(f["speed"] for f in fish),
        "confidence": round(sum(f["confidence"] for f in fish) / len(fish)),
        "tracks": counts["known"],
    }


def _fish_row(track, settings):
    speed_px = track["speed"]
    activity = min(100, round(100 * speed_px / (IDLE_SPEED * 5)))
    return {
        "id": f"{settings.label} {track['id']:02d}",
        "zone": zone_of(track),
        "speed": settings.to_speed(speed_px),
        "speed_px": speed_px,
        "speed_unit": settings.speed_unit,
        "trend_pct": 0.0,
        "trend_dir": "up",
        "heading": track["heading"],
        "confidence": round(100 * track["conf"]),
        "dwell_min": round(track["age_s"] / 60, 1),
        "status": "swimming" if speed_px > IDLE_SPEED else "idle",
        "last_seen": datetime.now().strftime("%H:%M:%S") if track["matched"] else "tracking",
        "activity": activity,
    }


def _detail_title(key, label):
    return {"view": "Longest in frame", "movement": "Most active now", "peak": "Fastest right now",
            "confidence": "Highest-confidence IDs", "tracks": f"{label} identified"}[key]


def _details(key, fish, unit):
    if not fish:
        return []
    if key == "confidence":
        rows = sorted(fish, key=lambda f: f["confidence"], reverse=True)
        value = lambda f: f"{f['confidence']}%"
    elif key in ("view", "tracks"):
        rows = sorted(fish, key=lambda f: f["dwell_min"], reverse=True)
        value = lambda f: f"{f['dwell_min']} min"
    else:
        rows = fish
        value = lambda f: f"{f['speed']} {unit}"
    return [{"name": f["id"], "sub": f["zone"], "value": value(f),
             "trend": f"{f['activity']}% active"} for f in rows[:4]]

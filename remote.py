"""Talking to a tracker that runs somewhere else.

The site can be deployed where the tank is unreachable (Vercel, Render). Set
TRACKER_ORIGIN to the public URL of the machine running the tracker and the
site fetches video snapshots and data from it, server side, adding
TRACKER_TOKEN. The token stays in the host's environment and never reaches
the browser.
"""
import json
import logging
import os
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

TIMEOUT_S = 10


def origin():
    return os.environ.get("TRACKER_ORIGIN", "").rstrip("/")


def token():
    return os.environ.get("TRACKER_TOKEN", "")


def enabled():
    return bool(origin())


def fetch(path, timeout=TIMEOUT_S):
    """(body, content_type) from the remote tracker, or (None, None)."""
    url = f"{origin()}{path}"
    request = urllib.request.Request(url, headers={"X-Tank-Token": token()} if token() else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), response.headers.get("Content-Type", "application/octet-stream")
    except urllib.error.HTTPError as exc:
        log.warning("Tracker %s returned %s", path, exc.code)
    except Exception as exc:
        log.warning("Tracker %s unreachable: %s", path, exc)
    return None, None


def fetch_json(path, timeout=TIMEOUT_S):
    body, _type = fetch(path, timeout)
    if body is None:
        return None
    try:
        return json.loads(body)
    except ValueError:
        log.warning("Tracker %s sent something that is not JSON", path)
        return None

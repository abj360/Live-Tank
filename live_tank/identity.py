"""Persistent fish identities on top of DeepSORT (LMB 01, LMB 02, ...).

DeepSORT issues a new track id whenever it loses a fish: fish crossing, blobs
merging, or a fish swimming out of view. This gallery turns those short tracks
into a stable population of fish:

* The population is learned, never configured: a new identity is created only
  when more fish are visible *at the same time* than there are identities, and
  that stays true for a few seconds. Add fish to the tank and they get new
  numbers once they are seen together with the others.
* Any other new track must be a fish we already know. It first tries a
  confident appearance/position match, and otherwise takes the best-matching
  identity that is currently out of view (re-identification by elimination).
* Identities that were only ever seen for a moment and then vanished for a
  long time (a reflection or glitch that slipped through) are retired.
"""
import logging
from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment

log = logging.getLogger(__name__)


def _touching(a, b, pad=0.06):
    """True when boxes (x1, y1, x2, y2), each grown by `pad` of its size, intersect."""
    def grow(box):
        x1, y1, x2, y2 = box
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        return x1 - px, y1 - py, x2 + px, y2 + py
    ax1, ay1, ax2, ay2 = grow(a)
    bx1, by1, bx2, by2 = grow(b)
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def _count_separate(tracks):
    """Number of groups of tracks whose boxes touch (connected components)."""
    parent = list(range(len(tracks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            if _touching(tracks[i]["box"], tracks[j]["box"]):
                parent[find(i)] = find(j)
    return len({find(i) for i in range(len(tracks))})


class Identity:
    def __init__(self, number, now):
        self.number = number
        self.first_seen = now
        self.last_seen = now
        self.lost_at = None
        self.last_pos = None                      # last position actually seen (not predicted)
        self.area = None
        self.features = deque(maxlen=60)
        self.observed_s = 0.0
        self._last_obs = None


class IdentityGallery:
    def __init__(self, match_threshold=0.15, min_hits=20, visible_hits=10, new_fish_s=4.0,
                 crowded_fraction=0.8, appearance_weight=0.25, retire_observed_s=4.0, retire_missing_s=30.0):
        self.match_threshold = match_threshold    # "close match" cost: ~within 10% of the frame diagonal
        self.min_hits = min_hits                  # track hits (~1 s) before it may claim an identity
        self.visible_hits = visible_hits          # track hits (~0.5 s) before it counts as a fish in view
        self.new_fish_s = new_fish_s              # window for "more fish than identities"
        self.crowded_fraction = crowded_fraction  # share of frames in that window that must agree
        self.appearance_weight = appearance_weight
        self.retire_observed_s = retire_observed_s
        self.retire_missing_s = retire_missing_s
        self.identities = {}                      # number -> Identity
        self.by_track = {}                        # DeepSORT track id -> identity number
        self.reids = 0
        self._excess = deque()                    # (time, more fish in view than identities?)

    def update(self, tracks, frame_diag, now):
        """Bind confirmed tracks to identities; returns {track_id: number}.

        `tracks` holds every confirmed track as a dict with `track_id`,
        `center`, `origin` (where the track first appeared), `box`
        (x1, y1, x2, y2), `area`, `feature` (or None), `matched` and `hits`.
        """
        by_id = {t["track_id"]: t for t in tracks}
        for tid in [tid for tid in self.by_track if tid not in by_id]:
            self.identities[self.by_track.pop(tid)].lost_at = now
        self._retire(now)

        # Evidence of a new fish: more separate fish in view than identities,
        # in most frames over a few seconds. It is judged per frame, not per
        # track, because DeepSORT tracks here often last under a second.
        # Touching tracks count once: a fish cut in two by the tank's frame
        # bar, or two tracks on one fish, is still one fish.
        visible = [t for t in tracks if t["matched"] and t["hits"] >= self.visible_hits]
        self._excess.append((now, _count_separate(visible) > len(self.identities)))
        while self._excess and now - self._excess[0][0] > self.new_fish_s:
            self._excess.popleft()
        crowded = (now - self._excess[0][0] >= 0.9 * self.new_fish_s
                   and sum(e for _, e in self._excess) >= self.crowded_fraction * len(self._excess))

        pending = [t for t in tracks if t["track_id"] not in self.by_track and t["matched"]]
        self._confident_matches(pending, frame_diag, now)

        waiting = [t for t in pending if t["track_id"] not in self.by_track and t["hits"] >= self.min_hits]
        self._assign_missing(waiting, frame_diag, now)

        for t in sorted((t for t in waiting if t["track_id"] not in self.by_track), key=lambda t: -t["hits"]):
            if self._take_over_coasting(t, by_id, frame_diag, now):
                continue
            if crowded and not any(_touching(t["box"], o["box"]) for o in tracks if o is not t):
                self._mint(t, now)
                self._excess.clear()  # one new fish per round of evidence
                crowded = False

        self._observe(by_id, now)
        return dict(self.by_track)

    def first_seen(self, number, default):
        identity = self.identities.get(number)
        return identity.first_seen if identity else default

    def missing(self, now):
        bound = set(self.by_track.values())
        return [i for i in self.identities.values() if i.number not in bound and i.lost_at is not None]

    # ---- steps ------------------------------------------------------------------
    def _missing_now(self):
        bound = set(self.by_track.values())
        return [i for i in self.identities.values() if i.number not in bound and i.last_pos is not None]

    def _confident_matches(self, pending, frame_diag, now):
        """Quick re-id: a new track that starts right where an out-of-view fish was last seen."""
        missing = self._missing_now()
        pairs = sorted((self._cost(i, t, frame_diag), i.number, t["track_id"])
                       for i in missing for t in pending)
        used_identities, used_tracks = set(), set()
        for cost, number, tid in pairs:
            if cost > self.match_threshold:
                break
            if number in used_identities or tid in used_tracks:
                continue
            self._bind(tid, number, now, f"close match, cost {cost:.3f}")
            used_identities.add(number)
            used_tracks.add(tid)

    def _assign_missing(self, waiting, frame_diag, now):
        """Give stable unidentified tracks the out-of-view identities.

        Every fish is a fish we already know, so these tracks take the
        missing identities. All of them are matched at once (Hungarian
        assignment) on where each identity was last seen versus where each
        track first appeared, so a fish that left on the left and comes back
        on the left gets its own number back, not the next free one.
        """
        missing = self._missing_now()
        if not waiting or not missing:
            return
        cost = np.array([[self._cost(i, t, frame_diag) for t in waiting] for i in missing])
        for row, col in zip(*linear_sum_assignment(cost)):
            self._bind(waiting[col]["track_id"], missing[row].number, now,
                       f"returning, cost {cost[row, col]:.3f}")

    def _take_over_coasting(self, track, by_id, frame_diag, now):
        """Every identity is bound, but one is only coasting (DeepSORT lost the
        fish this frame) right where this track is: the same fish picked up again."""
        coasting = {number: tid for tid, number in self.by_track.items()
                    if not by_id[tid]["matched"] and _touching(by_id[tid]["box"], track["box"], pad=0.15)}
        if not coasting:
            return False
        cost, number = min((self._cost(self.identities[n], track, frame_diag), n) for n in coasting)
        del self.by_track[coasting[number]]
        self._bind(track["track_id"], number, now, f"picked up again, cost {cost:.3f}")
        return True

    def _bind(self, tid, number, now, why):
        identity = self.identities[number]
        gone = f"{now - identity.lost_at:.1f}s" if identity.lost_at else "coasting"
        log.info("re-id LMB %02d <- track %d (%s, missing %s)", number, tid, why, gone)
        self.by_track[tid] = number
        identity.lost_at = None
        self.reids += 1

    def _mint(self, track, now):
        number = 1
        while number in self.identities:
            number += 1
        self.identities[number] = Identity(number, now)
        self.by_track[track["track_id"]] = number
        log.info("new fish LMB %02d <- track %d (%d identities)", number, track["track_id"], len(self.identities))

    def _observe(self, by_id, now):
        for tid, number in self.by_track.items():
            t = by_id[tid]
            identity = self.identities[number]
            if not t["matched"]:  # coasting: keep the last position actually seen
                identity._last_obs = None
                continue
            identity.last_pos = t["center"]
            if identity._last_obs is not None:
                identity.observed_s += min(now - identity._last_obs, 0.5)
            identity._last_obs = now
            identity.last_seen = now
            identity.area = t["area"] if identity.area is None else identity.area * 0.9 + t["area"] * 0.1
            if t["feature"] is not None and (not identity.features or np.random.random() < 0.25):
                identity.features.append(t["feature"])

    def _retire(self, now):
        bound = set(self.by_track.values())
        for identity in list(self.identities.values()):
            if (identity.number not in bound and identity.lost_at is not None
                    and identity.observed_s < self.retire_observed_s
                    and now - identity.lost_at > self.retire_missing_s):
                log.info("retired LMB %02d (seen only %.1fs)", identity.number, identity.observed_s)
                del self.identities[identity.number]

    def _cost(self, identity, track, frame_diag):
        """Mostly location: where the identity was last seen vs where the track first
        appeared (as a fraction of the frame diagonal). Colour and size break ties."""
        (lx, ly), (ox, oy) = identity.last_pos, track["origin"]
        distance = float(np.hypot(lx - ox, ly - oy)) / frame_diag
        appearance = 0.5
        if identity.features and track["feature"] is not None:
            appearance = 1.0 - float(np.max(np.stack(identity.features) @ track["feature"]))
        size = 0.0
        if identity.area and track["area"]:
            size = min(abs(np.log(track["area"] / identity.area)), 2.0)
        return distance + self.appearance_weight * appearance + 0.05 * size

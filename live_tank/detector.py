"""Background-model fish detector that feeds DeepSORT.

The repository ships no trained YOLO weights, so fish are found as foreground
blobs against a learned background, which suits a fixed underwater camera:

* A temporal median over the first seconds seeds the background, which removes
  fish that swim through during warm-up.
* Only pixels that are darker, more saturated or warmer (redder) than the
  blue-grey water count as fish. Air bubbles, glare and the ghost left where a
  fish used to sit are lighter than the background, so they are ignored.
* Reflections of fish in the tank glass are dimmer copies of a real fish:
  blobs with a weak contrast peak, or that look like a much fainter copy of a
  stronger fish in the same frame, are dropped.
* The aquarium's frame lines are not fish. When the camera sways a little, a
  dark edge shifts and the uncovered strip looks "darker than background".
  Every blob is template-matched against the background within a few pixels;
  if it is just displaced background structure, it is rejected.
* If the camera is bumped, a sustained global shift between frame and
  background is detected and the background is re-seeded immediately.
* The background adapts quickly where no fish are and slowly under fish, so a
  bass hovering in place stays detected for a good while.
"""
from collections import deque, namedtuple

import cv2
import numpy as np

# x, y, w, h: upright box for DeepSORT; corners: rotated box that hugs the fish.
Blob = namedtuple("Blob", "x y w h confidence peak feature corners area")

FEATURE_DIM = 144


class FishDetector:
    def __init__(self, width=640, warmup_frames=40, warmup_stride=5,
                 diff_threshold=20.0, saturation_weight=0.6, warmth_weight=0.8,
                 bg_alpha=0.03, fg_alpha=0.001, tracked_alpha=0.0,
                 min_area_frac=0.004, max_area_frac=0.55,
                 min_peak=55.0, reflection_ratio=0.62, reflection_similarity=0.8,
                 structure_search_px=10, structure_match=0.45, reseed_structure_frac=0.08,
                 core_ratio=1.4, split_core_ratio=2.2, merge_ratio=1.8, ghost_ratio=0.6,
                 min_yellow=0.5, compact_min_yellow=1.5,
                 shift_px=2.5, shift_frames=8, ignore_rects=()):
        self.width = width
        self.warmup_frames = warmup_frames
        self.warmup_stride = warmup_stride
        self.diff_threshold = diff_threshold
        self.saturation_weight = saturation_weight
        self.warmth_weight = warmth_weight
        self.bg_alpha = bg_alpha
        self.fg_alpha = fg_alpha
        self.tracked_alpha = tracked_alpha
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.min_peak = min_peak
        self.reflection_ratio = reflection_ratio
        self.reflection_similarity = reflection_similarity
        self.structure_search_px = structure_search_px
        self.structure_match = structure_match
        self.reseed_structure_frac = reseed_structure_frac
        self.shift_px = shift_px
        self.shift_frames = shift_frames
        self.ignore_rects = ignore_rects
        self.core_ratio = core_ratio
        self.merge_ratio = merge_ratio
        self.split_core_ratio = split_core_ratio
        self.ghost_ratio = ghost_ratio
        self.min_yellow = min_yellow
        self.compact_min_yellow = compact_min_yellow
        self._single_areas = deque(maxlen=300)
        self.rejected = []
        self.reseeds = 0
        self.reset()

    def reset(self):
        self.bg = None
        self.mask = None
        self._warmup = []
        self._frame_i = 0
        self._shifted_for = 0
        self._structure_for = 0

    @property
    def learning(self):
        return self.bg is None

    @property
    def progress(self):
        return 1.0 if self.bg is not None else len(self._warmup) / self.warmup_frames

    def detect(self, small, tracked_shapes=()):
        """Return fish blobs in `small` pixel coordinates.

        `tracked_shapes` are 4x2 rotated-box corners of fish being tracked,
        predicted for this frame. They guide splitting merged fish, and the
        background is barely learned under them, so a bass that stops
        swimming keeps being detected instead of fading into the background.

        Rejected blobs are kept in `self.rejected` as (blob, reason) for display.
        """
        h, w = small.shape[:2]
        blur = cv2.GaussianBlur(small, (5, 5), 0)
        self.rejected = []
        empty = np.zeros((h, w), np.uint8)

        if self.bg is None:
            if self._frame_i % self.warmup_stride == 0:
                self._warmup.append(blur)
            self._frame_i += 1
            if len(self._warmup) >= self.warmup_frames:
                self.bg = np.median(np.stack(self._warmup), axis=0).astype(np.float32)
                self._warmup = []
            self.mask = empty
            return []

        blur_f = blur.astype(np.float32)
        if self._camera_moved(blur, w, h):
            self.bg = blur_f.copy()
            self.reseeds += 1
            self.mask = empty
            return []

        diff, frame_v, bg_v = self._fishness(blur, blur_f)
        for x0, y0, x1, y1 in self.ignore_rects:
            diff[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 0
        mask = np.where(diff > self.diff_threshold, 255, 0).astype(np.uint8)

        # A global change (lights) is not fish: re-adapt fast.
        if cv2.countNonZero(mask) > 0.6 * w * h:
            cv2.accumulateWeighted(blur_f, self.bg, 0.25)
            self.mask = empty
            return []

        close_k = max(3, (w // 120) | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(close_k))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        centers = [_body_axes(corners) for corners in tracked_shapes]
        frame_grad, bg_grad = _gradient(frame_v), _gradient(bg_v)
        # Colour shift vs the water behind: bass are olive (more yellow, a*/b* up);
        # black cups, hoses and frame bars are neutral or bluish.
        lab_now = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_bg = cv2.cvtColor(self.bg.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        tint_a, tint_b = lab_now[..., 1] - lab_bg[..., 1], lab_now[..., 2] - lab_bg[..., 2]
        ghosts = np.zeros_like(mask)
        candidates = []
        structure_area = 0.0
        frame_area = w * h
        for group, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if not self.min_area_frac * frame_area <= area <= self.max_area_frac * frame_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            blob = np.zeros((bh, bw), np.uint8)
            cv2.drawContours(blob, [contour], -1, 255, cv2.FILLED, offset=(-x, -y))
            parts = self._split(blob, x, y, centers, area, diff[y:y + bh, x:x + bw])
            for part in parts:
                found, part_contour = self._describe(part, x, y, diff, hsv, frame_area, split=len(parts) > 1)
                if found is None:
                    continue
                if self._is_structure(frame_v, bg_v, part, x, y):
                    self.rejected.append((found, "TANK EDGE"))
                    structure_area += found.area
                    continue
                if self._is_ghost(frame_grad, bg_grad, part, x, y):
                    self.rejected.append((found, "GHOST"))
                    ghosts[y:y + part.shape[0], x:x + part.shape[1]] |= part
                    continue
                if self._is_dark_object(tint_a, tint_b, part, x, y, found.corners):
                    self.rejected.append((found, "DARK OBJECT"))
                    continue
                if len(parts) == 1:
                    self._single_areas.append(found.area)
                candidates.append((found, part_contour, group))

        filled = np.zeros_like(mask)
        faint = np.zeros_like(mask)  # rejected as reflections: may be a resting fish, learn slowly
        detections = []
        for blob, contour, group in candidates:
            if self._is_reflection(blob, group, candidates):
                self.rejected.append((blob, "REFLECTION"))
                cv2.drawContours(faint, [contour], -1, 255, cv2.FILLED)
                continue
            cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
            detections.append(blob)

        # Lots of tank structure lighting up for a while means the camera has
        # moved further than the edge test can absorb: start the background over.
        if structure_area > self.reseed_structure_frac * frame_area:
            self._structure_for += 1
        else:
            self._structure_for = 0
        if self._structure_for >= self.shift_frames:
            self._structure_for = 0
            self.bg = blur_f.copy()
            self.reseeds += 1

        # Learn the background slowly under fish, quickly elsewhere (tank edges
        # and dark objects included, so a small camera move is absorbed within
        # a second or two). Under a tracked fish nothing is learned at all, so
        # a resting bass stays visible; the ghost test above cleans up after it leaves.
        protect = cv2.dilate(filled | faint, _ellipse(close_k * 3))
        tracked = np.zeros_like(mask)
        for corners in tracked_shapes:
            cv2.fillPoly(tracked, [np.int32(np.round(corners))], 255)
        tracked &= protect  # only the fish's own surroundings, not the whole box
        cv2.accumulateWeighted(blur_f, self.bg, self.bg_alpha, mask=cv2.bitwise_not(protect))
        cv2.accumulateWeighted(blur_f, self.bg, self.fg_alpha, mask=protect & cv2.bitwise_not(tracked))
        if self.tracked_alpha > 0:
            cv2.accumulateWeighted(blur_f, self.bg, self.tracked_alpha, mask=tracked)
        # A ghost is stale background: replace it with what the camera sees now.
        cv2.accumulateWeighted(blur_f, self.bg, 0.5, mask=cv2.dilate(ghosts, _ellipse(close_k)))

        self.mask = filled
        return detections

    def _fishness(self, blur, blur_f):
        frame_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV).astype(np.float32)
        bg_u8 = self.bg.astype(np.uint8)
        bg_hsv = cv2.cvtColor(bg_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
        darker = np.maximum(bg_hsv[..., 2] - frame_hsv[..., 2], 0)
        more_saturated = np.maximum(frame_hsv[..., 1] - bg_hsv[..., 1], 0)
        warmer = np.maximum((blur_f[..., 2] - blur_f[..., 0]) - (self.bg[..., 2] - self.bg[..., 0]), 0)
        diff = darker + self.saturation_weight * more_saturated + self.warmth_weight * warmer
        return diff, frame_hsv[..., 2], bg_hsv[..., 2]

    # ---- one box per fish -----------------------------------------------------------
    @property
    def typical_area(self):
        """Median area of recent single-fish blobs, once enough are seen."""
        if len(self._single_areas) < 30:
            return None
        return float(np.median(self._single_areas))

    def _split(self, blob, x, y, centers, area, roi_diff):
        """Split a blob that covers several fish into one mask per fish.

        1. Fish already tracked separately (crossing, touching, one under the
           other): every blob pixel goes to the predicted fish whose rotated
           body it fits best.
        2. Separate dense cores: nearby fish get merged by their faint, blurred
           edges, but each keeps its own strong core. Pixels go to the nearest core.
        3. Crossing bodies: cut between two deep notches of the outline.
        4. A blob much larger than a typical fish is split along the narrow
           "necks" between bodies (distance-transform watershed).
        """
        h, w = blob.shape
        inside = []
        for cx, cy, ux, uy, half_len, half_wid in _drop_duplicates(centers):
            lx, ly = cx - x, cy - y
            if not (0 <= lx < w and 0 <= ly < h):
                continue
            r = max(2, int(0.5 * half_wid))
            ix, iy = int(lx), int(ly)
            if not blob[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1].any():
                continue
            inside.append((lx, ly, ux, uy, half_len, half_wid))

        if len(inside) >= 2:
            # Distance measured along and across each fish's own body axis, so
            # where one fish lies over another, pixels follow the body they line up with.
            ys, xs = np.nonzero(blob)
            dist = []
            for cx, cy, ux, uy, half_len, half_wid in inside:
                dx, dy = xs - cx, ys - cy
                along = (dx * ux + dy * uy) / half_len
                across = (-dx * uy + dy * ux) / half_wid
                dist.append(along ** 2 + across ** 2)
            dist = np.stack(dist)
            owner = dist.argmin(axis=0)
            parts = []
            for i in range(len(inside)):
                part = np.zeros_like(blob)
                sel = owner == i
                part[ys[sel], xs[sel]] = 255
                parts.append(_largest_component(part))
            return parts

        min_part = max(self.min_area_frac * self.width * self.width * 9 / 16, 0.15 * area)
        core = np.where((blob > 0) & (roi_diff > self.split_core_ratio * self.diff_threshold), 255, 0).astype(np.uint8)
        core = cv2.morphologyEx(core, cv2.MORPH_OPEN, _ellipse(5))
        # Shape-based splits must leave fish-shaped pieces: a dark head and a
        # dark body are two "cores" of one fish, but the head alone is compact.
        parts = _split_by_cores(blob, core, min_core_area=0.12 * area, min_part_area=min_part)
        if len(parts) >= 2 and _fish_shaped(parts):
            return parts

        parts = _notch_split(blob, min_part_area=min_part)
        if len(parts) >= 2 and _fish_shaped(parts):
            return parts

        typical = self.typical_area
        if typical and area > self.merge_ratio * typical:
            parts = _watershed_split(blob, min_part_area=0.5 * typical)
            if len(parts) >= 2 and _fish_shaped(parts):
                return parts
        return [blob]

    def _describe(self, part, x, y, diff, hsv, frame_area, split):
        """Blob for one fish mask; the box hugs the fish's core pixels."""
        contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < self.min_area_frac * frame_area:
            return None, None
        px, py, pw, ph = cv2.boundingRect(contour)
        if max(pw, ph) > 10 * min(pw, ph):
            return None, None
        solidity = area / (cv2.contourArea(cv2.convexHull(contour)) or 1.0)
        if solidity < 0.3 and not split:  # long thin ghost bands from tank structure
            return None, None

        sub = np.zeros((ph, pw), np.uint8)
        cv2.drawContours(sub, [contour], -1, 255, cv2.FILLED, offset=(-px, -py))
        ax, ay = x + px, y + py
        roi_diff = diff[ay:ay + ph, ax:ax + pw]
        inside = sub > 0
        peak = float(np.percentile(roi_diff[inside], 90))

        # Fit to the body: drop the faint halo the blur and morphology add.
        core = inside & (roi_diff > self.core_ratio * self.diff_threshold)
        use = core if core.sum() >= 0.5 * inside.sum() else inside
        ys, xs = np.nonzero(use)
        pts = np.column_stack([xs + ax, ys + ay]).astype(np.float32)
        corners = cv2.boxPoints(cv2.minAreaRect(pts))
        bx, by, bw, bh = cv2.boundingRect(pts)

        confidence = 0.55 * min(1.0, peak / (5 * self.diff_threshold)) + 0.45 * min(1.0, solidity / 0.8)
        feature = _colour_feature(hsv[ay:ay + ph, ax:ax + pw], sub)
        blob = Blob(bx, by, bw, bh, float(np.clip(confidence, 0, 1)), peak, feature, corners, area)
        return blob, contour + np.array([[x, y]])

    def _is_dark_object(self, tint_a, tint_b, blob_mask, x, y, corners):
        """True for dark things that are not fish (suction cups, hoses, corners).

        Measured on this tank: every bass blob is at least slightly more yellow
        than the water behind it, while the black objects are neutral or bluish.
        Compact blobs need a clearer olive tint, since that is what the round
        cups look like; an elongated body is accepted with a fainter tint.
        """
        h, w = blob_mask.shape
        yellow = cv2.mean(tint_b[y:y + h, x:x + w], mask=blob_mask)[0]
        red = cv2.mean(tint_a[y:y + h, x:x + w], mask=blob_mask)[0]
        if yellow < self.min_yellow:
            return True
        e1, e2 = np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[1])
        aspect = max(e1, e2) / max(min(e1, e2), 1.0)
        return aspect < 2.0 and yellow < self.compact_min_yellow and red <= 0.5

    def _is_ghost(self, frame_grad, bg_grad, blob_mask, x, y):
        """True when the blob's outline is in the background, not in the frame.

        A real fish has a sharp outline against the water right now. A ghost
        (where a fish used to be, but the background still remembers it) has
        its outline only in the background image.
        """
        ring = cv2.morphologyEx(blob_mask, cv2.MORPH_GRADIENT, _ellipse(5))
        if cv2.countNonZero(ring) < 20:
            return False
        h, w = blob_mask.shape
        now_edge = cv2.mean(frame_grad[y:y + h, x:x + w], mask=ring)[0]
        bg_edge = cv2.mean(bg_grad[y:y + h, x:x + w], mask=ring)[0]
        return now_edge < self.ghost_ratio * bg_edge

    def _is_structure(self, frame_v, bg_v, blob_mask, x, y):
        """True when the blob's own pixels are explained by background shifted a few pixels.

        A tank frame line that moved when the camera swayed disappears once the
        background is shifted to match; a fish over open water never does.
        """
        m = self.structure_search_px
        img_h, img_w = frame_v.shape
        ys, xs = np.nonzero(blob_mask)
        if ys.size < 20:
            return False
        stride = max(1, ys.size // 1500)  # a sample of the blob's pixels is plenty
        ys, xs = ys[::stride] + y, xs[::stride] + x
        pix = frame_v[ys, xs]
        base = float(np.mean(np.abs(pix - bg_v[ys, xs])))
        if base < 1e-3:
            return True
        best = base
        for dy in range(-m, m + 1, 2):
            sy = np.clip(ys + dy, 0, img_h - 1)
            for dx in range(-m, m + 1, 2):
                if dx == 0 and dy == 0:
                    continue
                sx = np.clip(xs + dx, 0, img_w - 1)
                best = min(best, float(np.mean(np.abs(pix - bg_v[sy, sx]))))
        return best < self.structure_match * base and best < 14.0

    def _is_reflection(self, blob, group, candidates):
        if blob.peak < self.min_peak:
            return True
        for other, _contour, other_group in candidates:
            # Parts split from one blob are touching fish, never a reflection pair.
            if other_group == group or other.peak * self.reflection_ratio < blob.peak:
                continue
            if float(np.dot(other.feature, blob.feature)) >= self.reflection_similarity:
                return True
        return False

    def _camera_moved(self, blur, w, h):
        """True once the frame has been offset from the background for a while.

        Fish moving produce a momentary shift estimate; a bumped camera
        produces one that persists, because the background does not move.
        """
        scale = 0.5
        size = (int(w * scale), int(h * scale))
        cur = cv2.cvtColor(cv2.resize(blur, size), cv2.COLOR_BGR2GRAY).astype(np.float32)
        ref = cv2.cvtColor(cv2.resize(self.bg.astype(np.uint8), size), cv2.COLOR_BGR2GRAY).astype(np.float32)
        window = _hanning(size)
        (dx, dy), response = cv2.phaseCorrelate(ref, cur, window)
        if response > 0.05 and np.hypot(dx, dy) / scale > self.shift_px:
            self._shifted_for += 1
        else:
            self._shifted_for = 0
        if self._shifted_for >= self.shift_frames:
            self._shifted_for = 0
            return True
        return False


def _colour_feature(hsv_roi, blob_mask):
    """Unit-length colour descriptor of a blob's own pixels.

    A 16x8 hue/saturation histogram plus a 16-bin brightness histogram. It
    stands in for the mars-small128 CNN embedding (which needs TensorFlow and
    was trained on people) and plugs into DeepSORT's cosine metric.
    """
    mask = blob_mask if cv2.countNonZero(blob_mask) >= 16 else None
    hue_sat = cv2.calcHist([hsv_roi], [0, 1], mask, [16, 8], [0, 180, 0, 256]).flatten()
    value = cv2.calcHist([hsv_roi], [2], mask, [16], [0, 256]).flatten()
    hue_sat /= np.linalg.norm(hue_sat) or 1.0
    value /= np.linalg.norm(value) or 1.0
    feature = np.concatenate([hue_sat, 0.6 * value])
    norm = np.linalg.norm(feature)
    if norm == 0:
        return np.full(FEATURE_DIM, 1 / np.sqrt(FEATURE_DIM), np.float32)
    return feature / norm


def _gradient(v):
    return cv2.magnitude(cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3))


def _body_axes(corners):
    """(cx, cy, ux, uy, half_length, half_width) of a rotated box."""
    corners = np.asarray(corners, np.float32)
    cx, cy = corners.mean(axis=0)
    e1, e2 = corners[1] - corners[0], corners[2] - corners[1]
    major, minor = (e1, e2) if np.linalg.norm(e1) >= np.linalg.norm(e2) else (e2, e1)
    length = float(np.linalg.norm(major))
    ux, uy = (major / length) if length > 0 else (1.0, 0.0)
    return float(cx), float(cy), float(ux), float(uy), max(length / 2, 2.0), max(float(np.linalg.norm(minor)) / 2, 2.0)


def _drop_duplicates(fish, max_angle_deg=25.0):
    """Drop a second track sitting on the same fish.

    Duplicates share a centre (within half the body width) and a heading.
    A fish passing under another has its own centre or its own heading, so
    both are kept and the blob is split between them.
    """
    cos_limit = np.cos(np.radians(max_angle_deg))
    kept = []
    for f in sorted(fish, key=lambda f: f[4] * f[5], reverse=True):
        cx, cy, ux, uy, _l, hw = f
        duplicate = any(np.hypot(cx - k[0], cy - k[1]) < max(hw, k[5])
                        and abs(ux * k[2] + uy * k[3]) > cos_limit for k in kept)
        if not duplicate:
            kept.append(f)
    return kept


def _fish_shaped(parts, min_aspect=1.7):
    """Every part is elongated like a fish body (long side / short side)."""
    for part in parts:
        pts = cv2.findNonZero(part)
        if pts is None:
            return False
        (_c, (w, h), _a) = cv2.minAreaRect(pts)
        if max(w, h) < min_aspect * max(min(w, h), 1.0):
            return False
    return True


def _split_by_cores(blob, core, min_core_area, min_part_area):
    """Give every blob pixel to its nearest core when there are several cores."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_core_area]
    if len(keep) < 2:
        return [blob]
    seeds = np.isin(labels, keep)
    # Distance to the nearest seed pixel, labelled by which seed component it was.
    _, nearest = cv2.distanceTransformWithLabels(np.where(seeds, 0, 255).astype(np.uint8),
                                                 cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP)
    seed_label = np.zeros(nearest.max() + 1, np.int32)
    seed_label[nearest[seeds]] = labels[seeds]
    owner = seed_label[nearest]
    parts = []
    for i in keep:
        part = np.where((owner == i) & (blob > 0), 255, 0).astype(np.uint8)
        part = _largest_component(part)
        if cv2.countNonZero(part) >= min_part_area:
            parts.append(part)
    return parts if len(parts) >= 2 else [blob]


def _notch_split(blob, min_part_area, min_depth=0.22, max_cut=0.9):
    """Cut crossing or touching bodies between two deep notches in the outline.

    Depth and cut length are relative to sqrt(blob area). A single bent fish
    has one deep notch (inside its curve), so it is never cut.
    """
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [blob]
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 100 or len(contour) < 8:
        return [blob]
    try:
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:
        return [blob]
    if defects is None:
        return [blob]
    scale = np.sqrt(area)
    notches = sorted(((depth / 256.0, tuple(int(v) for v in contour[f].reshape(2)))
                      for _s, _e, f, depth in defects.reshape(-1, 4)
                      if depth / 256.0 > min_depth * scale), reverse=True)
    # Two notches: one fish lying across the end of another (T / L shape), and a
    # cut separates them. Four or more: an X, where any straight cut would give
    # two halves that each mix both fish; that case is split by tracking instead.
    if not 2 <= len(notches) <= 3:
        return [blob]
    pairs = sorted((np.hypot(a[1][0] - b[1][0], a[1][1] - b[1][1]), a[1], b[1])
                   for i, a in enumerate(notches) for b in notches[i + 1:])
    for length, p1, p2 in pairs:
        if length > max_cut * scale:
            break
        cut = blob.copy()
        cv2.line(cut, p1, p2, 0, 3)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cut)
        order = sorted(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
        if len(order) >= 2 and stats[order[1], cv2.CC_STAT_AREA] >= min_part_area:
            return [np.where(labels == i, 255, 0).astype(np.uint8) for i in order[:2]]
    return [blob]


def _largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 2:
        return mask
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == best, 255, 0).astype(np.uint8)


def _watershed_split(blob, min_part_area):
    """Split touching bodies at the narrow necks between them."""
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return [blob]
    seeds = np.where(dist > 0.7 * dist.max(), 255, 0).astype(np.uint8)
    n, seed_labels = cv2.connectedComponents(seeds)
    if n - 1 < 2:
        return [blob]
    markers = seed_labels.astype(np.int32) + 1   # seeds 2..n, background 1
    markers[(blob > 0) & (seeds == 0)] = 0        # unknown: to be flooded
    relief = cv2.cvtColor(cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                          cv2.COLOR_GRAY2BGR)
    cv2.watershed(255 - relief, markers)
    parts = []
    for label in range(2, n + 1):
        part = np.where((markers == label) & (blob > 0), 255, 0).astype(np.uint8)
        if cv2.countNonZero(part) >= min_part_area:
            parts.append(part)
    return parts


_windows = {}


def _hanning(size):
    if size not in _windows:
        _windows[size] = cv2.createHanningWindow(size, cv2.CV_32F)
    return _windows[size]


def _ellipse(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

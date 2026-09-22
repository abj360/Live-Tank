"""Draws the computer-vision layers onto the output frame.

Everything is burned into the frame so boxes can never drift out of sync with
the video. Strokes are doubled (dark navy under, light/electric blue over) so
they stay visible on both the pale water and the dark fish.
"""
import time

import cv2
import numpy as np

NAVY = (74, 31, 11)      # #0B1F4A
ICE = (255, 240, 232)    # #E8F0FF
BLUE = (255, 110, 40)    # #286EFF
FONT = cv2.FONT_HERSHEY_DUPLEX

DEFAULT_LAYERS = {
    "boxes": True,
    "labels": True,
    "trails": True,
    "vectors": True,
    "tentative": True,
    "detections": False,
    "rejected": False,
    "mask": False,
}


def erase_osd_text(img, rect):
    """Inpaint the camera's white OSD text (the date/time) out of `rect`.

    Only the text strokes and their dark outline are repainted, so water or a
    fish behind the clock stays visible.
    """
    h, w = img.shape[:2]
    x0, y0, x1, y1 = int(rect[0] * w), int(rect[1] * h), int(rect[2] * w), int(rect[3] * h)
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    text = cv2.inRange(hsv, (0, 0, 165), (180, 70, 255))
    text = cv2.dilate(text, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_px(w / 1920, 7),) * 2))
    img[y0:y1, x0:x1] = cv2.inpaint(roi, text, 5, cv2.INPAINT_TELEA)


def draw_frame(img, tracks, detections, rejected, mask, layers, label):
    """`detections` and `rejected` hold 4x2 corner arrays of rotated boxes."""
    u = img.shape[1] / 1920.0
    now = time.time()

    if layers["mask"] and mask is not None:
        fg = mask > 0
        tint = img.copy()
        tint[fg] = BLUE
        cv2.addWeighted(tint, 0.38, img, 0.62, 0, img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, ICE, _px(u, 2), cv2.LINE_AA)

    if layers["detections"]:
        for corners in detections:
            _dashed_poly(img, corners, NAVY, _px(u, 4), u)
            _dashed_poly(img, corners, ICE, _px(u, 2), u)

    if layers["rejected"]:
        for corners, reason in rejected:
            _dashed_poly(img, corners, NAVY, _px(u, 2), u)
            bottom = corners[np.argmax(corners[:, 1])]
            _chip(img, int(bottom[0]), int(bottom[1]), [(reason, 0.62, 1)], u, below=True, centered=True)

    for t in tracks:
        corners = t["corners"]
        top = corners[np.argmin(corners[:, 1])]
        if not t["confirmed"]:
            if layers["tentative"]:
                _dashed_poly(img, corners, NAVY, _px(u, 4), u)
                _dashed_poly(img, corners, BLUE, _px(u, 2), u)
                if layers["labels"]:
                    _chip(img, int(top[0]), int(top[1]), [(t["stage"], 0.8, 1)], u, centered=True)
            continue

        if layers["trails"] and len(t["trail"]) > 1:
            _trail(img, t["trail"], u, now, t["id"])

        if layers["boxes"]:
            poly = [np.int32(np.round(corners))]
            cv2.polylines(img, poly, True, NAVY, _px(u, 3), cv2.LINE_AA)
            cv2.polylines(img, poly, True, ICE, _px(u, 1.2), cv2.LINE_AA)
            _brackets(img, corners, u)

        if layers["vectors"]:
            cx, cy = t["center"]
            vx, vy = t["velocity"]
            tip = (int(cx + vx * 12), int(cy + vy * 12))  # Kalman position ~12 frames ahead
            if np.hypot(tip[0] - cx, tip[1] - cy) > 8 * u:
                cv2.arrowedLine(img, (cx, cy), tip, NAVY, _px(u, 7), cv2.LINE_AA, tipLength=0.28)
                cv2.arrowedLine(img, (cx, cy), tip, ICE, _px(u, 3), cv2.LINE_AA, tipLength=0.28)
            cv2.circle(img, (cx, cy), _px(u, 6), NAVY, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), _px(u, 3), ICE, -1, cv2.LINE_AA)

        if layers["labels"]:
            _chip(img, int(top[0]), int(top[1]), [(label, 1.0, 2), (f"{t['id']:02d}", 1.0, 1),
                                                  (f"{round(t['conf'] * 100)}%", 0.8, 1)], u, centered=True)
    return img


def _px(u, px):
    return max(1, int(round(px * u)))


def _brackets(img, corners, u):
    """Corner brackets that follow a rotated box."""
    edges = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    arm = float(np.clip(min(edges) * 0.3, 10 * u, 56 * u))
    polys = []
    for i in range(4):
        c, prev, nxt = corners[i], corners[i - 1], corners[(i + 1) % 4]
        to_prev = (prev - c) / (np.linalg.norm(prev - c) or 1.0)
        to_next = (nxt - c) / (np.linalg.norm(nxt - c) or 1.0)
        polys.append(np.int32(np.round([c + to_prev * arm, c, c + to_next * arm])))
    cv2.polylines(img, polys, False, NAVY, _px(u, 8), cv2.LINE_AA)
    cv2.polylines(img, polys, False, BLUE, _px(u, 3.5), cv2.LINE_AA)


PULSE_PERIOD_S = 1.4   # one pulse travels tail -> head in this time
PULSE_SPAN = 0.14      # fraction of the trail lit by the pulse


def _trail(img, points, u, now, identity):
    """Thin trail with a bright pulse running from tail to head."""
    pts = np.array(points, np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total < 2:
        return
    cum = np.concatenate([[0.0], np.cumsum(seg)]) / total  # 0 at tail, 1 at head
    ipts = np.int32(np.round(pts))

    # Base line: dim at the tail, brighter towards the head.
    cv2.polylines(img, [ipts], False, NAVY, _px(u, 2.5), cv2.LINE_AA)
    for i in range(1, len(ipts)):
        w = cum[i]
        colour = tuple(int(NAVY[c] + (BLUE[c] - NAVY[c]) * (0.35 + 0.65 * w)) for c in range(3))
        cv2.line(img, tuple(ipts[i - 1]), tuple(ipts[i]), colour, _px(u, 1), cv2.LINE_AA)

    # The pulse: a short bright stretch whose position cycles over time.
    phase = ((now / PULSE_PERIOD_S) + identity * 0.37) % 1.0
    lo, hi = phase - PULSE_SPAN, phase
    for i in range(1, len(ipts)):
        a, b = cum[i - 1], cum[i]
        if b < lo or a > hi:
            continue
        glow = 1.0 - min(1.0, abs(hi - b) / PULSE_SPAN)  # brightest at the leading edge
        colour = tuple(int(BLUE[c] + (ICE[c] - BLUE[c]) * glow) for c in range(3))
        cv2.line(img, tuple(ipts[i - 1]), tuple(ipts[i]), colour, _px(u, 1.5 + glow), cv2.LINE_AA)

    # Heartbeat ring at the head.
    beat = (now / PULSE_PERIOD_S + identity * 0.37) % 1.0
    head = tuple(ipts[-1])
    cv2.circle(img, head, _px(u, 3), ICE, -1, cv2.LINE_AA)
    if beat < 0.6:
        cv2.circle(img, head, _px(u, 4 + 14 * beat / 0.6), BLUE, _px(u, 1), cv2.LINE_AA)


def _dashed_poly(img, corners, color, thickness, u):
    dash, gap = max(4, int(16 * u)), max(3, int(10 * u))
    n = len(corners)
    for i in range(n):
        a, b = corners[i], corners[(i + 1) % n]
        length = int(np.hypot(b[0] - a[0], b[1] - a[1]))
        if length == 0:
            continue
        for s in range(0, length, dash + gap):
            e = min(s + dash, length)
            pa = (int(a[0] + (b[0] - a[0]) * s / length), int(a[1] + (b[1] - a[1]) * s / length))
            pb = (int(a[0] + (b[0] - a[0]) * e / length), int(a[1] + (b[1] - a[1]) * e / length))
            cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)


def _chip(img, x, y, parts, u, below=False, centered=False):
    """Pale label chip with dark navy text above (or below) the point (x, y)."""
    scale = 0.78 * u
    pad = max(4, int(9 * u))
    gap = max(4, int(11 * u))
    bar = max(3, int(7 * u))
    sizes = [cv2.getTextSize(text, FONT, scale * m, _px(u, th))[0] for text, m, th in parts]
    text_w = sum(s[0] for s in sizes) + gap * (len(parts) - 1)
    text_h = max(s[1] for s in sizes)
    w = bar + pad * 2 + text_w
    h = text_h + pad * 2

    img_h, img_w = img.shape[:2]
    left = int(np.clip(x - w // 2 if centered else x, 0, max(0, img_w - w)))
    top = y + _px(u, 4) if below else y - h - _px(u, 4)
    if top < 0:  # no room above the box: tuck the chip inside it
        top = y + _px(u, 4)
    top = int(np.clip(top, 0, max(0, img_h - h)))

    # Frosted glass: blur what is behind the chip and wash it with pale ice.
    roi = img[top:top + h, left:left + w]
    if roi.size:
        k = _px(u, 15) | 1
        frosted = cv2.GaussianBlur(roi, (k, k), 0)
        img[top:top + h, left:left + w] = cv2.addWeighted(frosted, 0.42, np.full_like(roi, ICE), 0.58, 0)
    cv2.rectangle(img, (left, top), (left + bar, top + h), NAVY, -1)
    cv2.rectangle(img, (left, top), (left + w, top + h), ICE, _px(u, 1))
    cursor = left + bar + pad
    baseline = top + pad + text_h
    for (text, m, th), (sw, _sh) in zip(parts, sizes):
        cv2.putText(img, text, (cursor, baseline), FONT, scale * m, NAVY, _px(u, th), cv2.LINE_AA)
        cursor += sw + gap

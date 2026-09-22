"""Threaded frame grabber that always exposes the newest frame.

Reading on a dedicated thread and keeping only the latest frame stops RTSP
latency from building up when processing is slower than the camera.
"""
import logging
import threading
import time

import cv2

log = logging.getLogger(__name__)


class FrameSource(threading.Thread):
    def __init__(self, url, label, live):
        super().__init__(daemon=True, name="capture")
        self.url = url
        self.label = label
        self.live = live
        self.status = "CONNECTING"
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.last_frame_at = 0.0
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0

    def _open(self):
        params = []
        for name in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
            if hasattr(cv2, name):
                params += [getattr(cv2, name), 8000]
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def run(self):
        backoff = 1.0
        while True:
            cap = self._open()
            if cap is None:
                self.status = "RECONNECTING" if self.live else "SOURCE ERROR"
                log.warning("Could not open %s, retrying in %.0fs", self.label, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
                continue
            backoff = 1.0
            log.info("Opened %s", self.label)
            period = 0.0 if self.live else 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25.0)
            self.status = "LIVE" if self.live else "PLAYBACK"
            self._pump(cap, period)
            cap.release()
            self.status = "RECONNECTING"
            log.warning("Lost %s, reconnecting", self.label)

    def _pump(self, cap, period):
        last = time.time()
        next_due = last
        rewound = False
        while True:
            ok, frame = cap.read()
            if not ok:
                # Files loop forever; a file that fails right after a rewind is broken.
                if period and not rewound and cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    rewound = True
                    continue
                return
            rewound = False

            now = time.time()
            if period:
                next_due += period
                if next_due > now:
                    time.sleep(next_due - now)
                else:
                    next_due = now
                now = time.time()

            dt = now - last
            last = now
            if dt > 0:
                self.fps = 1.0 / dt if self.fps == 0 else self.fps * 0.9 + 0.1 / dt

            with self._cond:
                self._frame = frame
                self._seq += 1
                self.last_frame_at = now
                self.height, self.width = frame.shape[:2]
                self._cond.notify_all()

    def wait_frame(self, after_seq, timeout):
        """Block until a frame newer than `after_seq` exists; None on timeout."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > after_seq, timeout):
                return None
            return self._seq, self._frame, self.last_frame_at

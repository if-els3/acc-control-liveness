"""
=============================================================
core/liveness.py — Liveness Detection (Anti-Spoofing)
=============================================================
    [A] Eye Blink Detection
            Wajah asli berkedip. Foto tidak.
            → Haarcascade deteksi mata per frame
            → Hitung berapa kali mata hilang-muncul (blink event)

Optimasi untuk kondisi low-light dan pengguna berkacamata:
    - CLAHE + gamma correction pada area mata
    - parameter cascade dibuat configurable dari config.py

Penggunaan:
    liveness = LivenessDetector()
    # Kumpulkan beberapa frame (~2 detik)
    frames = [frame1, frame2, ... , frame_n]
    result = liveness.check(frames, face_box=(x1,y1,x2,y2))
    # result: LivenessResult(is_live, score, detail)
=============================================================
"""
import os
import sys
import logging
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

cv2 = None
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    log.error("OpenCV tidak ada")

# ─── Haarcascade path ────────────────────────────────────
def _find_cascade(filename: str) -> str:
    if not CV2_OK or cv2 is None:
        return ""
    cv_dir = os.path.dirname(cv2.__file__)
    for root, _, files in os.walk(cv_dir):
        if filename in files:
            return os.path.join(root, filename)
    return ""

# ──────────────────────────────────────────────────────────

@dataclass
class LivenessResult:
    is_live:   bool
    score:     float          # 0.0 – 1.0  (makin tinggi makin "hidup")
    votes:     int            # berapa metode vote LIVE
    total:     int            # total metode yang dijalankan
    detail:    dict = field(default_factory=dict)

    def __str__(self):
        status = "LIVE" if self.is_live else "SPOOF"
        return (f"[{status}] score={self.score:.2f} "
                f"votes={self.votes}/{self.total} {self.detail}")


# ══════════════════════════════════════════════════════════
# METODE A — EYE / BLINK DETECTION
# ══════════════════════════════════════════════════════════

class BlinkDetector:
    """
    Deteksi kedipan menggunakan haarcascade mata.
    Blink = mata terdeteksi di frame N, tidak terdeteksi di N+1, terdeteksi lagi N+2.
    Mendukung kacamata dengan fallback cascade.
    """

    def __init__(self):
        if cv2 is None:
            self._cascade = None
            self._history = []
            self._blinks = 0
            self._state = "unknown"
            self._closed_frames = 0
            return

        # Coba kacamata dulu, fallback ke standar
        path = _find_cascade("haarcascade_eye_tree_eyeglasses.xml")
        if not path:
            path = _find_cascade("haarcascade_eye.xml")
        self._cascade = cv2.CascadeClassifier(path) if path else None
        self._history: List[bool] = []    # True = mata terdeteksi
        self._blinks  = 0
        self._state = "unknown"
        self._closed_frames = 0

    @staticmethod
    def _preprocess_for_eyes(face_bgr: np.ndarray) -> np.ndarray:
        """Preprocess area wajah agar deteksi mata lebih stabil di low-light."""
        if cv2 is None:
            return np.zeros((1, 1), dtype=np.uint8)

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=getattr(config, "BLINK_CLAHE_CLIP_LIMIT", 2.0),
            tileGridSize=getattr(config, "BLINK_CLAHE_TILE_GRID", (8, 8)),
        )
        enhanced = clahe.apply(gray)

        gamma = max(0.1, float(getattr(config, "BLINK_GAMMA", 1.15)))
        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, lut)

        return cv2.equalizeHist(enhanced)

    def update(self, face_bgr: np.ndarray) -> bool:
        """Update dengan frame baru. Return True jika mata terdeteksi."""
        if self._cascade is None or cv2 is None:
            return True
        gray = self._preprocess_for_eyes(face_bgr)
        eyes = self._cascade.detectMultiScale(
            gray,
            scaleFactor=float(getattr(config, "BLINK_EYE_SCALE_FACTOR", 1.08)),
            minNeighbors=int(getattr(config, "BLINK_EYE_MIN_NEIGHBORS", 1)),
            minSize=tuple(getattr(config, "BLINK_EYE_MIN_SIZE", (10, 10))),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        eye_present = len(eyes) > 0
        self._history.append(eye_present)

        min_closed = int(getattr(config, "LIVENESS_BLINK_MIN_CLOSED_FRAMES", 1))
        max_closed = int(getattr(config, "LIVENESS_BLINK_MAX_CLOSED_FRAMES", 8))

        if eye_present:
            if self._state == "closed" and min_closed <= self._closed_frames <= max_closed:
                self._blinks += 1
            self._state = "open"
            self._closed_frames = 0
        else:
            if self._state in ("open", "unknown"):
                self._state = "closed"
                self._closed_frames = 1
            else:
                self._closed_frames += 1

        if getattr(config, "DEBUG_EYE_TRACKER", False):
            debug_img = face_bgr.copy()
            for (x, y, w, h) in eyes:
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_img, f"Blinks: {self._blinks} State: {self._state}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            # Resize agar lebih jelas dilihat
            debug_img = cv2.resize(debug_img, (0, 0), fx=3.0, fy=3.0)
            cv2.imshow("Eye Debug Tracker", debug_img)
            cv2.waitKey(1)

        return eye_present

    @property
    def blink_count(self) -> int:
        return self._blinks

    def reset(self):
        self._history.clear()
        self._blinks = 0


def _blink_score(face_frames_bgr: List[np.ndarray]) -> Tuple[float, dict]:
    """
    Hitung skor kedipan dari sequence frame wajah (crop).
    ≥ 1 blink terdeteksi dalam window pengamatan = LIVE.
    """
    detector = BlinkDetector()
    if detector._cascade is None:
        # Cascade tidak tersedia — beri skor netral
        return 0.5, {"blink": "cascade_unavailable"}

    eye_frames = 0
    for f in face_frames_bgr:
        if detector.update(f):
            eye_frames += 1

    blinks  = detector.blink_count
    # Score: 1 blink = pasti live; 0 blink masih mungkin live (pengamatan pendek)
    # Kurangi score jika mata tidak pernah terdeteksi sama sekali
    required_blinks = int(getattr(config, "LIVENESS_BLINK_MIN_COUNT", 1))

    if eye_frames == 0:
        score = 0.1   # mata tidak pernah kelihatan — mungkin foto/sudut salah
    elif blinks >= required_blinks:
        score = 1.0
    else:
        # Belum berkedip tapi mata terlihat — mungkin window pengamatan pendek
        score = float(getattr(config, "LIVENESS_BLINK_NO_EVENT_SCORE", 0.58))

    return score, {
        "blinks":        blinks,
        "required_blinks": required_blinks,
        "eye_frames":    eye_frames,
        "total_frames":  len(face_frames_bgr),
        "blink_score":   round(score, 3),
    }


# ══════════════════════════════════════════════════════════
# MAIN DETECTOR
# ══════════════════════════════════════════════════════════

# Threshold skor blink untuk dinyatakan LIVE
BLINK_LIVE_THRESH = float(getattr(config, "LIVENESS_BLINK_SCORE_THRESH", 0.55))

# Minimum blink votes (blink-only => default 1)
MIN_VOTES = int(getattr(config, "LIVENESS_MIN_VOTES", 1))


class LivenessDetector:
    """
    Blink-only liveness detector.

    Penggunaan dalam mode akses:
        detector = LivenessDetector()
        frames = [frame1, ..., frame_n]   # ≥ 10 frame (~2 detik di 5fps)
        face_box = (x1, y1, x2, y2)
        result = detector.check(frames, face_box)
        if result.is_live:
            # lanjutkan ke face recognition
    """

    def __init__(self):
        self._enabled = CV2_OK

    @staticmethod
    def _crop_face(frame: np.ndarray,
                   box: Tuple[int,int,int,int],
                   pad: float = 0.1) -> np.ndarray:
        """Crop area wajah dengan sedikit padding."""
        pad = float(getattr(config, "LIVENESS_FACE_PAD", pad))
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        pw = int((x2-x1) * pad); ph = int((y2-y1) * pad)
        x1 = max(0, x1-pw); y1 = max(0, y1-ph)
        x2 = min(w, x2+pw); y2 = min(h, y2+ph)
        return frame[y1:y2, x1:x2]

    def check(self,
              frames: List[np.ndarray],
              face_box: Tuple[int,int,int,int]) -> LivenessResult:
        """
        Periksa liveness dari sequence frame.

        Args:
            frames:   list frame BGR (minimal 5, ideal 10–20)
            face_box: (x1, y1, x2, y2) area wajah di frame

        Returns:
            LivenessResult
        """
        if not self._enabled:
            return LivenessResult(True, 1.0, 1, 1,
                                  {"note": "OpenCV tidak ada, skip liveness"})
        if cv2 is None:
            return LivenessResult(True, 1.0, 1, 1,
                                  {"note": "OpenCV tidak ada, skip liveness"})
        if len(frames) < 3:
            return LivenessResult(True, 0.6, 1, 1,
                                  {"note": "Frame tidak cukup, skip liveness"})

        # Crop wajah dari semua frame
        face_crops = []
        for f in frames:
            crop = self._crop_face(f, face_box)
            if crop.size > 0 and crop.shape[0] > 20 and crop.shape[1] > 20:
                face_crops.append(crop)

        if not face_crops:
            return LivenessResult(False, 0.0, 0, 1,
                                  {"note": "Wajah tidak bisa di-crop"})

        detail = {}
        votes  = 0

        # ── METODE: Blink ───────────────────────────────────
        b_score, b_detail = _blink_score(face_crops)
        detail.update(b_detail)
        if b_score >= BLINK_LIVE_THRESH:
            votes += 1

        # ── Keputusan ──────────────────────────────────────
        combined = b_score
        min_score = float(getattr(config, "LIVENESS_MIN_SCORE", BLINK_LIVE_THRESH))
        is_live  = (votes >= MIN_VOTES) and (combined >= min_score)

        log.info(f"Liveness: live={is_live} score={combined:.3f} "
                 f"votes={votes}/1 {detail}")

        return LivenessResult(
            is_live=is_live,
            score=round(combined, 3),
            votes=votes,
            total=1,
            detail=detail,
        )

    def check_realtime(self,
                       cam,              # CameraStream instance
                       face_engine,      # FaceEngine instance
                       duration: float = 3.0) -> LivenessResult:
        """
        Kumpulkan frame secara real-time selama `duration` detik,
        deteksi wajah terbesar di setiap frame, lalu periksa liveness.
        """
        frames   = []
        face_box = None
        t0       = time.time()

        while time.time() - t0 < duration:
            frame = cam.read()
            if frame is None:
                time.sleep(0.05)
                continue

            box = face_engine.detect_largest(frame)
            if box is not None:
                if face_box is None:
                    face_box = box[:4]   # simpan box pertama sebagai referensi
                frames.append(frame)

            time.sleep(0.1)   # ~10 fps collection

        if not frames or face_box is None:
            return LivenessResult(False, 0.0, 0, 1,
                                  {"note": "Tidak ada wajah terdeteksi selama observasi"})

        log.info(f"Liveness: {len(frames)} frame dikumpulkan dalam {duration:.1f}s")
        return self.check(frames, face_box)

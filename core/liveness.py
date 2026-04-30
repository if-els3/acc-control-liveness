"""
=============================================================
core/liveness.py — Liveness Detection (Anti-Spoofing)
=============================================================
Tiga metode dikombinasikan (voting):

  [A] LBP Texture Analysis
      Wajah asli punya tekstur micro (pori, bulu halus)
      berbeda dari foto cetak / layar LCD.
      → Analisis distribusi histogram LBP
      → Hitung variance & high-freq energy

  [B] Optical Flow Motion Analysis
      Wajah asli punya micro-movement natural (nafas,
      kedipan, tremor kecil). Foto = statis.
      → Farneback optical flow antar frame berurutan
      → Ukur std-dev magnitude gerakan

  [C] Eye Blink Detection
      Wajah asli berkedip. Foto tidak.
      → Haarcascade deteksi mata per frame
      → Hitung berapa kali mata hilang-muncul (blink event)

Semua metode murni OpenCV — tidak butuh model tambahan / internet.

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

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    log.error("OpenCV tidak ada")

# ─── Haarcascade path ────────────────────────────────────
def _find_cascade(filename: str) -> str:
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
# METODE A — LBP TEXTURE
# ══════════════════════════════════════════════════════════

def _lbp_map(gray: np.ndarray) -> np.ndarray:
    """
    Hitung LBP (Local Binary Pattern) manual menggunakan numpy.
    Lebih cepat dari loop Python, tidak butuh scikit-image.
    Menggunakan 8 tetangga dengan radius 1.
    """
    h, w    = gray.shape
    lbp     = np.zeros((h-2, w-2), dtype=np.uint8)
    center  = gray[1:-1, 1:-1].astype(np.int16)
    neighbors = [
        gray[0:-2, 0:-2], gray[0:-2, 1:-1], gray[0:-2, 2:],
        gray[1:-1, 2:],   gray[2:,   2:],   gray[2:,   1:-1],
        gray[2:,   0:-2], gray[1:-1, 0:-2],
    ]
    for i, nb in enumerate(neighbors):
        lbp += ((nb.astype(np.int16) >= center) * (1 << i)).astype(np.uint8)
    return lbp


def _texture_score(face_bgr: np.ndarray) -> Tuple[float, dict]:
    """
    Hitung skor tekstur wajah berdasarkan distribusi LBP.
    Wajah asli → distribusi lebih merata (entropy tinggi).
    Foto/layar → distribusi terpusat (entropy rendah, banyak uniform patterns).
    """
    gray   = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))

    lbp    = _lbp_map(resized)
    hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
    hist    = hist.astype(np.float32) + 1e-7
    hist   /= hist.sum()

    # Shannon entropy — lebih tinggi = tekstur lebih kaya
    entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))

    # High-frequency energy via Laplacian
    lap     = cv2.Laplacian(resized, cv2.CV_64F)
    hf_energy = float(lap.var())

    # Normalisasi ke [0,1]
    # Entropy wajah asli biasanya 6.5–7.5 bit; foto 5.0–6.5 bit
    entropy_score = float(np.clip((entropy - 5.0) / 3.0, 0, 1))
    hf_score      = float(np.clip(hf_energy / 500.0, 0, 1))

    combined = 0.6 * entropy_score + 0.4 * hf_score

    return combined, {
        "lbp_entropy":    round(entropy, 3),
        "hf_energy":      round(hf_energy, 1),
        "texture_score":  round(combined, 3),
    }


# ══════════════════════════════════════════════════════════
# METODE B — OPTICAL FLOW MOTION
# ══════════════════════════════════════════════════════════

def _motion_score(frames_gray: List[np.ndarray]) -> Tuple[float, dict]:
    """
    Hitung skor gerakan alami dari sequence frame.
    Wajah hidup → micro-movement terdeteksi (std-dev flow > threshold).
    Foto/layar  → hampir nol gerakan.
    """
    if len(frames_gray) < 2:
        return 0.5, {"motion": "insufficient_frames"}

    magnitudes = []
    for i in range(len(frames_gray) - 1):
        f1 = cv2.resize(frames_gray[i],  (48, 48))
        f2 = cv2.resize(frames_gray[i+1], (48, 48))
        try:
            flow = cv2.calcOpticalFlowFarneback(
                f1, f2, None,
                pyr_scale=0.5, levels=2, winsize=10,
                iterations=2, poly_n=5, poly_sigma=1.1,
                flags=0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(float(mag.mean()))
        except Exception:
            pass

    if not magnitudes:
        return 0.5, {"motion": "flow_error"}

    avg_mag  = float(np.mean(magnitudes))
    std_mag  = float(np.std(magnitudes))

    # Wajah asli: avg 0.3–2.0, std > 0.1
    # Foto diam : avg ~0.0, std ~0.0
    # Layar animasi: avg bisa tinggi tapi pola berbeda
    avg_score = float(np.clip(avg_mag / 1.5, 0, 1))
    std_score = float(np.clip(std_mag / 0.5, 0, 1))
    combined  = 0.5 * avg_score + 0.5 * std_score

    return combined, {
        "flow_avg":     round(avg_mag, 3),
        "flow_std":     round(std_mag, 3),
        "motion_score": round(combined, 3),
    }


# ══════════════════════════════════════════════════════════
# METODE C — EYE / BLINK DETECTION
# ══════════════════════════════════════════════════════════

class BlinkDetector:
    """
    Deteksi kedipan menggunakan haarcascade mata.
    Blink = mata terdeteksi di frame N, tidak terdeteksi di N+1, terdeteksi lagi N+2.
    """

    def __init__(self):
        path = _find_cascade("haarcascade_eye.xml")
        self._cascade = cv2.CascadeClassifier(path) if path else None
        self._history: List[bool] = []    # True = mata terdeteksi
        self._blinks  = 0

    def update(self, face_bgr: np.ndarray) -> bool:
        """Update dengan frame baru. Return True jika mata terdeteksi."""
        if self._cascade is None:
            return True
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        eyes = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(15, 15),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        eye_present = len(eyes) > 0
        self._history.append(eye_present)

        # Deteksi blink: True → False → True
        if len(self._history) >= 3:
            a, b, c = self._history[-3], self._history[-2], self._history[-1]
            if a and not b and c:
                self._blinks += 1

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
    if eye_frames == 0:
        score = 0.1   # mata tidak pernah kelihatan — mungkin foto/sudut salah
    elif blinks >= 1:
        score = 1.0
    else:
        # Belum berkedip tapi mata terlihat — mungkin window pengamatan pendek
        score = 0.55

    return score, {
        "blinks":        blinks,
        "eye_frames":    eye_frames,
        "total_frames":  len(face_frames_bgr),
        "blink_score":   round(score, 3),
    }


# ══════════════════════════════════════════════════════════
# MAIN DETECTOR
# ══════════════════════════════════════════════════════════

# Threshold skor per metode untuk dianggap LIVE
TEXTURE_LIVE_THRESH = 0.40
MOTION_LIVE_THRESH  = 0.15
BLINK_LIVE_THRESH   = 0.50

# Threshold skor gabungan
COMBINED_LIVE_THRESH = 0.25

# Minimum votes dari 3 metode untuk dinyatakan live
MIN_VOTES = 2


class LivenessDetector:
    """
    Multi-method liveness detector.

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
            return LivenessResult(False, 0.0, 0, 3,
                                  {"note": "Wajah tidak bisa di-crop"})

        detail = {}
        scores = []
        votes  = 0

        # ── METODE A: Texture ──────────────────────────────
        # Gunakan frame tengah (kualitas biasanya paling baik)
        mid_frame = face_crops[len(face_crops)//2]
        t_score, t_detail = _texture_score(mid_frame)
        detail.update(t_detail)
        scores.append(t_score)
        if t_score >= TEXTURE_LIVE_THRESH:
            votes += 1

        # ── METODE B: Motion ───────────────────────────────
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in face_crops]
        m_score, m_detail = _motion_score(grays)
        detail.update(m_detail)
        scores.append(m_score)
        if m_score >= MOTION_LIVE_THRESH:
            votes += 1

        # ── METODE C: Blink ────────────────────────────────
        b_score, b_detail = _blink_score(face_crops)
        detail.update(b_detail)
        scores.append(b_score)
        if b_score >= BLINK_LIVE_THRESH:
            votes += 1

        # ── Keputusan ──────────────────────────────────────
        combined = float(np.mean(scores))
        is_live  = (votes >= MIN_VOTES) and (combined >= COMBINED_LIVE_THRESH)

        log.info(f"Liveness: live={is_live} score={combined:.3f} "
                 f"votes={votes}/3 {detail}")

        return LivenessResult(
            is_live=is_live,
            score=round(combined, 3),
            votes=votes,
            total=3,
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
            return LivenessResult(False, 0.0, 0, 3,
                                  {"note": "Tidak ada wajah terdeteksi selama observasi"})

        log.info(f"Liveness: {len(frames)} frame dikumpulkan dalam {duration:.1f}s")
        return self.check(frames, face_box)

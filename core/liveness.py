"""
=============================================================
core/liveness.py — Liveness Detection (Anti-Spoofing)
=============================================================
    [A] Eye Blink Detection via EAR (Eye Aspect Ratio)
            Wajah asli berkedip. Foto tidak.

            Metode utama — MediaPipe Face Mesh:
              → Deteksi 468 landmark wajah per frame
              → Hitung EAR (Eye Aspect Ratio) kiri + kanan
              → EAR < threshold beberapa frame = mata tertutup
              → Mata tertutup lalu terbuka kembali = 1 blink

            EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
            Nilai EAR normal (terbuka) ≈ 0.25–0.35
            Nilai EAR saat berkedip    < BLINK_EAR_THRESHOLD (~0.20)

            Fallback — Haarcascade:
              → Digunakan jika MediaPipe tidak terinstall
              → Kurang akurat, terutama di cahaya tidak seragam

Penggunaan:
    liveness = LivenessDetector()
    frames = [frame1, frame2, ... , frame_n]
    result = liveness.check(frames, face_box=(x1,y1,x2,y2))
    # result: LivenessResult(is_live, score, detail)

OPTIMASI:
    - FaceMesh di-pre-warm sekali di LivenessDetector.__init__()
    - BlinkDetector bisa menerima face_mesh pre-warmed
    - static_image_mode=False (tracking mode, 3-5x lebih cepat)
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

# ─── MediaPipe availability ───────────────────────────────
mp = None
mp_face_mesh = None
MP_OK = False
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    MP_OK = True
    log.info("MediaPipe Face Mesh tersedia — menggunakan metode EAR")
except ImportError:
    log.warning("MediaPipe tidak tersedia — fallback ke Haarcascade")

# ─── MediaPipe Eye Landmark Indices ─────────────────────────
# Indeks landmark MediaPipe Face Mesh untuk 6 titik per mata
# Format: [p1_outer, p2_upper_outer, p3_upper_inner,
#           p4_inner, p5_lower_inner, p6_lower_outer]
_LEFT_EYE_IDX  = [33,  160, 158, 133, 153, 144]
_RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# ─── Haarcascade path (fallback) ─────────────────────────
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
# HELPER: EAR CALCULATION
# ══════════════════════════════════════════════════════════

def _euclidean(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def _compute_ear(landmarks, indices: List[int], img_w: int, img_h: int) -> float:
    """
    Hitung Eye Aspect Ratio (EAR) dari 6 landmark mata.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Args:
        landmarks : daftar landmark MediaPipe (normalized)
        indices   : [p1, p2, p3, p4, p5, p6] index dari landmarks
        img_w/h   : dimensi frame untuk konversi koordinat
    Returns:
        float EAR value (0.0 jika tidak valid)
    """
    pts = [
        (landmarks[i].x * img_w, landmarks[i].y * img_h)
        for i in indices
    ]
    A = _euclidean(pts[1], pts[5])  # p2 – p6  (vertikal atas-bawah luar)
    B = _euclidean(pts[2], pts[4])  # p3 – p5  (vertikal atas-bawah dalam)
    C = _euclidean(pts[0], pts[3])  # p1 – p4  (horizontal)
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


# ══════════════════════════════════════════════════════════
# METODE UTAMA — EAR BLINK DETECTOR (MediaPipe)
# ══════════════════════════════════════════════════════════

class BlinkDetector:
    """
    Deteksi kedipan menggunakan Eye Aspect Ratio (EAR).

    Strategi:
      - Gunakan MediaPipe Face Mesh untuk mendapat 468 landmark per frame
      - Hitung rata-rata EAR kiri dan kanan setiap frame
      - Jika EAR < BLINK_EAR_THRESHOLD selama ≥ BLINK_EAR_CONSEC_FRAMES → mata tertutup
      - Saat EAR naik lagi (mata terbuka) → catat 1 blink

    Fallback ke Haarcascade jika MediaPipe tidak tersedia.

    OPTIMASI: Menerima face_mesh pre-warmed dari LivenessDetector agar tidak
    perlu inisialisasi ulang setiap sesi akses.
    """

    def __init__(self, face_mesh=None):
        """
        Args:
            face_mesh: instance MediaPipe FaceMesh yang sudah diinisialisasi
                       (pre-warmed). Jika None, akan dibuat baru.
        """
        self._blinks        = 0
        self._ear_history: List[float] = []
        self._state         = "unknown"   # "open" | "closed" | "unknown"
        self._closed_frames = 0

        # Baca parameter dari config
        self._ear_thresh    = float(getattr(config, "BLINK_EAR_THRESHOLD", 0.20))
        self._consec_frames = int(getattr(config, "BLINK_EAR_CONSEC_FRAMES", 2))

        # ── Inisialisasi backend ─────────────────────────
        self._mode = "none"  # "mediapipe" | "haar" | "none"
        self._owns_face_mesh = False  # apakah kita yang membuat face_mesh

        if MP_OK and mp_face_mesh is not None:
            if face_mesh is not None:
                # Gunakan instance pre-warmed dari LivenessDetector
                self._face_mesh = face_mesh
                self._mode = "mediapipe"
                log.debug("BlinkDetector: menggunakan FaceMesh pre-warmed")
            else:
                # Buat baru jika tidak ada pre-warmed (fallback)
                try:
                    self._face_mesh = mp_face_mesh.FaceMesh(
                        static_image_mode=False,        # tracking mode — jauh lebih cepat
                        max_num_faces=1,
                        refine_landmarks=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.4,
                    )
                    self._mode = "mediapipe"
                    self._owns_face_mesh = True
                    log.debug("BlinkDetector: MediaPipe Face Mesh baru (tracking mode)")
                except Exception as e:
                    log.warning(f"BlinkDetector: gagal init MediaPipe ({e}), fallback Haar")
                    self._face_mesh = None

        if self._mode != "mediapipe" and CV2_OK and cv2 is not None:
            path = _find_cascade("haarcascade_eye_tree_eyeglasses.xml")
            if not path:
                path = _find_cascade("haarcascade_eye.xml")
            self._cascade = cv2.CascadeClassifier(path) if path else None
            if self._cascade is not None:
                self._mode = "haar"
                log.debug("BlinkDetector: Haarcascade aktif (fallback)")

        log.info(f"BlinkDetector mode: {self._mode}")

    # ── EAR via MediaPipe ────────────────────────────────

    def _ear_from_frame(self, face_bgr: np.ndarray) -> Optional[float]:
        """Hitung rata-rata EAR dari frame wajah menggunakan MediaPipe."""
        if cv2 is None:
            return None
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        h, w = face_bgr.shape[:2]
        result = self._face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0].landmark
        ear_l = _compute_ear(lm, _LEFT_EYE_IDX,  w, h)
        ear_r = _compute_ear(lm, _RIGHT_EYE_IDX, w, h)
        return (ear_l + ear_r) / 2.0

    # ── Haar fallback ────────────────────────────────────

    @staticmethod
    def _preprocess_for_haar(face_bgr: np.ndarray) -> np.ndarray:
        """Preprocessing ringan untuk Haar — hanya CLAHE tanpa equalizeHist."""
        if cv2 is None:
            return np.zeros((1, 1), dtype=np.uint8)
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=float(getattr(config, "BLINK_CLAHE_CLIP_LIMIT", 1.5)),
            tileGridSize=tuple(getattr(config, "BLINK_CLAHE_TILE_GRID", (8, 8))),
        )
        return clahe.apply(gray)

    def _eye_present_haar(self, face_bgr: np.ndarray) -> bool:
        """Deteksi ada/tidaknya mata menggunakan Haarcascade."""
        gray = self._preprocess_for_haar(face_bgr)
        eyes = self._cascade.detectMultiScale(
            gray,
            scaleFactor=float(getattr(config, "BLINK_EYE_SCALE_FACTOR", 1.10)),
            minNeighbors=int(getattr(config, "BLINK_EYE_MIN_NEIGHBORS", 3)),
            minSize=tuple(getattr(config, "BLINK_EYE_MIN_SIZE", (12, 12))),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return len(eyes) > 0

    # ── Main update ──────────────────────────────────────

    def update(self, face_bgr: np.ndarray) -> Optional[float]:
        """
        Proses satu frame wajah.

        Returns:
            EAR value (float) jika MediaPipe, True/False jika Haar, None jika error.
        """
        if self._mode == "mediapipe":
            ear = self._ear_from_frame(face_bgr)

            if ear is None:
                # Face Mesh gagal di frame ini — skip, jangan ubah state
                return None

            self._ear_history.append(ear)
            eye_closed = ear < self._ear_thresh

            if eye_closed:
                if self._state in ("open", "unknown"):
                    self._state = "closed"
                    self._closed_frames = 1
                else:
                    self._closed_frames += 1
            else:
                if self._state == "closed" and self._closed_frames >= self._consec_frames:
                    self._blinks += 1
                    log.debug(f"Blink #{self._blinks} (EAR={ear:.3f}, "
                              f"closed_frames={self._closed_frames})")
                self._state = "open"
                self._closed_frames = 0

            # ── Debug: simpan frame jika diminta ────────
            if getattr(config, "DEBUG_EYE_TRACKER", False) and cv2 is not None:
                self._save_debug(face_bgr, ear)

            return ear

        elif self._mode == "haar":
            eye_present = self._eye_present_haar(face_bgr)

            min_closed = int(getattr(config, "LIVENESS_BLINK_MIN_CLOSED_FRAMES", 2))
            max_closed = int(getattr(config, "LIVENESS_BLINK_MAX_CLOSED_FRAMES", 10))

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

            if getattr(config, "DEBUG_EYE_TRACKER", False) and cv2 is not None:
                self._save_debug(face_bgr, None)

            return float(eye_present)

        return None

    def _save_debug(self, face_bgr: np.ndarray, ear: Optional[float]):
        """Simpan debug frame ke file (tanpa GUI, kompatibel headless)."""
        if cv2 is None:
            return
        debug_img = face_bgr.copy()
        label = (f"EAR={ear:.3f}" if ear is not None else "EAR=N/A")
        label += f"  Blinks:{self._blinks}  [{self._state}]"
        cv2.putText(debug_img, label, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        debug_img = cv2.resize(debug_img, (0, 0), fx=3.0, fy=3.0)
        out_path = os.path.join(getattr(config, "BASE_DIR", "."), "debug_eye.jpg")
        cv2.imwrite(out_path, debug_img)

    @property
    def blink_count(self) -> int:
        return self._blinks

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def ear_history(self) -> List[float]:
        return list(self._ear_history)

    def reset(self):
        """Reset state untuk sesi akses baru (tanpa re-init FaceMesh)."""
        self._blinks = 0
        self._ear_history.clear()
        self._state = "unknown"
        self._closed_frames = 0


# ══════════════════════════════════════════════════════════
# BLINK SCORE
# ══════════════════════════════════════════════════════════

def _blink_score(face_frames_bgr: List[np.ndarray],
                 detector: Optional['BlinkDetector'] = None) -> Tuple[float, dict]:
    """
    Hitung skor kedipan dari sequence frame wajah (crop).
    ≥ 1 blink terdeteksi dalam window pengamatan = LIVE.

    Args:
        face_frames_bgr: list frame wajah (crop)
        detector: BlinkDetector instance (opsional, jika sudah di-warm)
    """
    if detector is None:
        detector = BlinkDetector()

    if detector.mode == "none":
        return 0.5, {"blink": "no_backend_available"}

    valid_frames = 0
    for f in face_frames_bgr:
        result = detector.update(f)
        if result is not None:
            valid_frames += 1

    blinks          = detector.blink_count
    required_blinks = int(getattr(config, "LIVENESS_BLINK_MIN_COUNT", 1))
    ear_vals        = detector.ear_history
    avg_ear         = float(np.mean(ear_vals)) if ear_vals else -1.0

    if valid_frames == 0:
        # Backend gagal total (tidak ada wajah ditemukan di semua frame)
        score = 0.1
    elif blinks >= required_blinks:
        score = 1.0
    else:
        # Mata terdeteksi tapi belum cukup blink — fallback score (harus < threshold)
        score = float(getattr(config, "LIVENESS_BLINK_NO_EVENT_SCORE", 0.45))

    detail = {
        "blinks":          blinks,
        "required_blinks": required_blinks,
        "valid_frames":    valid_frames,
        "total_frames":    len(face_frames_bgr),
        "avg_ear":         round(avg_ear, 4),
        "blink_score":     round(score, 3),
        "blink_method":    detector.mode,
    }
    return score, detail


# ══════════════════════════════════════════════════════════
# MAIN DETECTOR
# ══════════════════════════════════════════════════════════

# Threshold skor blink untuk dinyatakan LIVE
BLINK_LIVE_THRESH = float(getattr(config, "LIVENESS_BLINK_SCORE_THRESH", 0.60))

# Minimum blink votes (blink-only => default 1)
MIN_VOTES = int(getattr(config, "LIVENESS_MIN_VOTES", 1))


class LivenessDetector:
    """
    Blink-only liveness detector.

    OPTIMASI: FaceMesh di-pre-warm sekali saat __init__(), BlinkDetector
    dibuat dan di-reuse antar sesi (reset() dipanggil tiap sesi baru).

    Penggunaan dalam mode akses:
        detector = LivenessDetector()   # pre-warm sekali di awal
        # Per sesi:
        blink_det = detector.create_blink_detector()  # gunakan FaceMesh pre-warmed
        blink_det.reset()
        # ... proses frame ...
        result = detector.check(frames, face_box, blink_detector=blink_det)
    """

    def __init__(self):
        self._enabled = CV2_OK
        self._face_mesh = None   # pre-warmed FaceMesh

        # Pre-warm MediaPipe FaceMesh satu kali
        if MP_OK and mp_face_mesh is not None and CV2_OK:
            try:
                log.info("LivenessDetector: pre-warming FaceMesh...")
                self._face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=False,        # tracking mode — lebih cepat
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.4,
                )
                log.info("LivenessDetector: FaceMesh pre-warmed OK")
            except Exception as e:
                log.warning(f"LivenessDetector: gagal pre-warm FaceMesh ({e})")
                self._face_mesh = None

    def create_blink_detector(self) -> BlinkDetector:
        """
        Buat BlinkDetector yang menggunakan FaceMesh pre-warmed.
        Panggil reset() sebelum setiap sesi akses baru.
        """
        detector = BlinkDetector(face_mesh=self._face_mesh)
        return detector

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
              face_box: Tuple[int,int,int,int],
              blink_detector: Optional[BlinkDetector] = None) -> LivenessResult:
        """
        Periksa liveness dari sequence frame.

        Args:
            frames:         list frame BGR (minimal 5, ideal 10–20)
            face_box:       (x1, y1, x2, y2) area wajah di frame
            blink_detector: BlinkDetector pre-warmed (opsional).
                            Jika None, dibuat baru (tanpa pre-warm).

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

        # ── METODE: Blink (EAR / Haar) ──────────────────
        # Gunakan detector pre-warmed jika tersedia
        b_score, b_detail = _blink_score(face_crops, detector=blink_detector)
        detail.update(b_detail)
        if b_score >= BLINK_LIVE_THRESH:
            votes += 1

        # ── Keputusan ───────────────────────────────────
        combined  = b_score
        min_score = float(getattr(config, "LIVENESS_MIN_SCORE", BLINK_LIVE_THRESH))
        is_live   = (votes >= MIN_VOTES) and (combined >= min_score)

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

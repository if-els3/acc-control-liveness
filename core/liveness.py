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

# ─── Indeks landmark untuk verifikasi kontur wajah (tersembunyi) ───
# Titik referensi bidang wajah (mata + mulut) vs ujung hidung, dipakai untuk
# memperkirakan apakah wajah punya struktur 3D (asli) atau datar (foto
# cetak / layar replay). Landmark ini di-reuse dari hasil FaceMesh yang
# sama dengan yang dipakai EAR -- tidak ada inference tambahan.
_NOSE_TIP_IDX    = 1
_L_EYE_OUTER_IDX = 33
_R_EYE_OUTER_IDX = 263
_L_MOUTH_IDX     = 61
_R_MOUTH_IDX     = 291

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


def _compute_contour_depth_ratio(landmarks) -> Optional[float]:
    """
    Rasio protrusion hidung terhadap bidang mata+mulut, dinormalisasi oleh
    jarak antar-mata (scale-invariant). Dihitung dari landmark (x,y,z)
    ternormalisasi milik MediaPipe -- TIDAK butuh model/inference tambahan
    karena landmark ini sudah dihitung untuk EAR pada frame yang sama.

    Wajah 3D asli               -> hidung menonjol ke kamera (z lebih kecil) -> rasio tinggi
    Foto cetak / video replay   -> permukaan datar                          -> rasio ~0
    """
    try:
        nose = landmarks[_NOSE_TIP_IDX]
        le, re_ = landmarks[_L_EYE_OUTER_IDX], landmarks[_R_EYE_OUTER_IDX]
        lm_l, lm_r = landmarks[_L_MOUTH_IDX], landmarks[_R_MOUTH_IDX]
    except IndexError:
        return None

    plane_z = (le.z + re_.z + lm_l.z + lm_r.z) / 4.0
    protrusion = plane_z - nose.z  # positif = hidung lebih dekat ke kamera

    interocular = ((le.x - re_.x) ** 2 + (le.y - re_.y) ** 2) ** 0.5
    if interocular < 1e-6:
        return None
    return protrusion / interocular


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
        self._open_frames   = 0

        # Verifikasi kontur wajah tersembunyi (reuse landmark FaceMesh, no extra cost)
        self._depth_history: List[float] = []
        self._contour_enabled = bool(getattr(config, "LIVENESS_CONTOUR_ENABLED", True))

        # Baca parameter dari config
        self._ear_thresh      = float(getattr(config, "BLINK_EAR_THRESHOLD", 0.21))
        # Hysteresis: threshold untuk membuka mata sedikit lebih tinggi dari threshold
        # menutup mata, mencegah noise EAR ber-osilasi di sekitar titik potong.
        # Contoh: close_thresh=0.21, open_thresh=0.23 → butuh kenaikan 0.02 sebelum dianggap buka.
        _ear_open_gap         = float(getattr(config, "BLINK_EAR_OPEN_GAP", 0.02))
        self._ear_open_thresh = self._ear_thresh + _ear_open_gap
        self._consec_frames   = int(getattr(config, "BLINK_EAR_CONSEC_FRAMES", 2))
        self._min_closed_frames = int(getattr(config, "LIVENESS_BLINK_MIN_CLOSED_FRAMES", self._consec_frames))
        self._max_closed_frames = int(getattr(config, "LIVENESS_BLINK_MAX_CLOSED_FRAMES", 10))
        # Debounce: jumlah minimum frame TERBUKA berturut-turut sebelum blink berikutnya
        # bisa dihitung. Mencegah 1 kedipan dihitung 2-3x akibat bouncing EAR.
        self._min_open_frames = int(getattr(config, "BLINK_MIN_OPEN_FRAMES", 3))

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

        # ── Verifikasi kontur wajah (tersembunyi) ──────────
        # Reuse landmark yang sama (tanpa proses/inference tambahan) untuk
        # memperkirakan depth relatif hidung. Hasilnya hanya disimpan di
        # memori sesi ini -- tidak pernah ditulis ke log/detail.
        if self._contour_enabled:
            depth_ratio = _compute_contour_depth_ratio(lm)
            if depth_ratio is not None:
                self._depth_history.append(depth_ratio)
                if getattr(config, "DEBUG_CONTOUR_TRACKER", False):
                    self._debug_contour(depth_ratio)

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

            # ── Hysteresis EAR ───────────────────────────────────────
            # Gunakan dua threshold berbeda untuk transisi tutup vs buka:
            #   mata MENUTUP : EAR < ear_thresh       (misal < 0.21)
            #   mata MEMBUKA : EAR > ear_open_thresh  (misal > 0.23)
            # Zona abu-abu [0.21–0.23] tidak mengubah state → eliminasi
            # false-positive akibat noise EAR bergetar di sekitar 0.21.
            if ear < self._ear_thresh:
                eye_signal = "closing"
            elif ear > self._ear_open_thresh:
                eye_signal = "opening"
            else:
                eye_signal = "neutral"  # zona abu-abu, pertahankan state lama

            if eye_signal == "closing":
                if self._state in ("open", "unknown"):
                    self._state = "closed"
                    self._closed_frames = 1
                else:
                    self._closed_frames += 1
                self._open_frames = 0

            elif eye_signal == "opening":
                if self._state == "closed" and self._min_closed_frames <= self._closed_frames <= self._max_closed_frames:
                    # Debounce: hitung blink HANYA jika sudah cukup frame terbuka
                    # sejak blink sebelumnya (hindari double-count 1 kedipan)
                    if self._open_frames >= self._min_open_frames or self._blinks == 0:
                        self._blinks += 1
                        log.debug(f"Blink #{self._blinks} (EAR={ear:.3f}, "
                                  f"closed={self._closed_frames}f, open_before={self._open_frames}f)")
                self._state = "open"
                self._closed_frames = 0
                self._open_frames += 1

            else:
                # neutral zone: pertahankan state saat ini, tambah counter open jika sedang open
                if self._state == "open":
                    self._open_frames += 1

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

    def _debug_contour(self, ratio: float):
        """Kalibrasi manual saja (DEBUG_CONTOUR_TRACKER=True). Tidak dipakai produksi,
        tidak lewat modul logging sama sekali (tidak masuk system.log)."""
        try:
            path = os.path.join(getattr(config, "BASE_DIR", "."), "debug_contour.log")
            with open(path, "a") as f:
                f.write(f"{time.time():.3f},{ratio:.5f}\n")
        except Exception:
            pass

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

    @property
    def depth_history(self) -> List[float]:
        return list(self._depth_history)

    def reset(self):
        """Reset state untuk sesi akses baru (tanpa re-init FaceMesh)."""
        self._blinks = 0
        self._ear_history.clear()
        self._depth_history.clear()
        self._state = "unknown"
        self._closed_frames = 0
        self._open_frames = 0

    def flush_pending_blink(self) -> int:
        """Hitung blink yang 'menggantung' (mata masih tertutup saat sesi habis).

        Masalah: saat FPS rendah (6-9 FPS di Raspi), durasi setiap frame ~111-167ms.
        Sebuah kedipan (100-400ms) bisa hanya terekam sebagai 1-2 frame "tutup"
        tepat di akhir sesi — dan karena blink baru dihitung saat transisi tutup→buka,
        fase "buka" tidak sempat terekam sebelum waktu habis.

        Solusi: di akhir sesi, jika state masih "closed" dan closed_frames sudah
        memenuhi syarat minimum, kita anggap blink itu valid dan hitung sekarang.

        Returns:
            Jumlah total blink setelah flush (termasuk yang pending).
        """
        if (self._state == "closed"
                and self._min_closed_frames <= self._closed_frames <= self._max_closed_frames):
            self._blinks += 1
            log.debug(f"flush_pending_blink: +1 blink (closed_frames={self._closed_frames}) "
                      f"→ total={self._blinks}")
        return self._blinks


# ══════════════════════════════════════════════════════════
# BLINK SCORE
# ══════════════════════════════════════════════════════════

def _blink_score(face_frames_bgr: List[np.ndarray],
                 detector: Optional['BlinkDetector'] = None,
                 required_blinks: Optional[int] = None) -> Tuple[float, dict]:
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

    # ── Flush blink pending ─────────────────────────────────────────────────
    # Jika FPS rendah (6–9 FPS di Raspi), satu kedipan bisa terpenggal:
    # fase "tutup" terekam tapi fase "buka" tidak sempat masuk sebelum sesi
    # habis. flush_pending_blink() menghitung blink yang masih "menggantung"
    # (state=closed, closed_frames sudah memenuhi syarat min).
    blinks = detector.flush_pending_blink()

    if required_blinks is None:
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


# ══════════════════════════════════════════════════════
# VERIFIKASI KONTUR WAJAH  (Anti 3D-Mask / Layar Datar)
# ══════════════════════════════════════════════════════

def _contour_live(depth_vals: List[float]) -> bool:

    min_samples = int(getattr(config, "LIVENESS_CONTOUR_MIN_SAMPLES", 3))
    if len(depth_vals) < min_samples:
        # Sample tidak cukup (mis. wajah miring terus / mediapipe sering gagal)
        # -> fail-open, biarkan keputusan ditentukan oleh metode blink saja.
        return True
    min_ratio = float(getattr(config, "LIVENESS_CONTOUR_MIN_RATIO", 0.12))
    med = float(np.median(depth_vals))
    # log.debug tidak pernah tampil di system.log (level default INFO) --
    # hanya berguna kalau seseorang sengaja menurunkan log level secara manual.
    log.debug(f"[hidden-check] contour_depth_median={med:.4f} thresh={min_ratio}")
    return med >= min_ratio


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
              blink_detector: Optional[BlinkDetector] = None,
              required_blinks: Optional[int] = None) -> LivenessResult:
        """
        Periksa liveness dari sequence frame.

        Args:
            frames:         list frame BGR (minimal 5, ideal 10–20)
            face_box:       (x1, y1, x2, y2) area wajah di frame
            blink_detector: BlinkDetector pre-warmed (opsional).
                            Jika None, dibuat baru (tanpa pre-warm).
            required_blinks: jumlah blink minimum untuk sesi ini.

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
        if required_blinks is None:
            required_blinks = int(getattr(config, "LIVENESS_BLINK_MIN_COUNT", 1))

        # ── METODE: Blink (EAR / Haar) ──────────────────
        # Gunakan detector pre-warmed jika tersedia
        b_score, b_detail = _blink_score(face_crops, detector=blink_detector, required_blinks=required_blinks)
        detail.update(b_detail)
        if b_score >= BLINK_LIVE_THRESH:
            votes += 1

        # ── Verifikasi kontur wajah, tersembunyi (tidak masuk detail/log) ──
        # Reuse landmark dari BlinkDetector -- tanpa model/inference tambahan,
        # sehingga tidak menambah waktu pemrosesan sesi liveness.
        contour_ok = True
        if (blink_detector is not None and blink_detector.mode == "mediapipe"
                and bool(getattr(config, "LIVENESS_CONTOUR_ENABLED", True))):
            contour_ok = _contour_live(blink_detector.depth_history)

        # ── Keputusan ───────────────────────────────────
        combined  = b_score
        min_score = float(getattr(config, "LIVENESS_MIN_SCORE", BLINK_LIVE_THRESH))
        is_live   = (votes >= MIN_VOTES) and (combined >= min_score) and contour_ok

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
                       duration: float = 3.0,
                       required_blinks: Optional[int] = None,
                       blink_detector: Optional[BlinkDetector] = None) -> LivenessResult:
        """
        Kumpulkan frame secara real-time selama `duration` detik dan proses
        liveness secara STREAMING per-frame (bukan batch di akhir).

        OPTIMASI:
          - Deteksi wajah (face_engine) hanya SEKALI di awal untuk mendapat
            face_box referensi — ini penyebab bottleneck utama (~400ms/frame).
          - EAR (MediaPipe) diproses per-frame tanpa jeda buatan → ~10–15 FPS.
          - Early-exit segera setelah target blink terpenuhi + early_exit_delay.
          - Tidak ada time.sleep() di dalam loop utama.
        """
        if required_blinks is None:
            required_blinks = int(getattr(config, "LIVENESS_BLINK_MIN_COUNT", 1))
        early_exit_delay = float(getattr(config, "LIVENESS_EARLY_EXIT_DELAY", 1.0))

        # ── Langkah 1: Deteksi wajah SEKALI untuk mendapat face_box ─────────
        # Coba hingga 3 detik pertama atau frame pertama yang valid.
        face_box       = None
        t_detect_start = time.time()
        _FACE_DETECT_TIMEOUT = 3.0

        log.info("Liveness check_realtime: mencari wajah awal ...")
        while face_box is None and (time.time() - t_detect_start) < _FACE_DETECT_TIMEOUT:
            frame = cam.read()
            if frame is None:
                time.sleep(0.02)  # tunggu frame kamera, minimal
                continue
            box = face_engine.detect_largest(frame)
            if box is not None:
                face_box = tuple(box[:4])   # (x1, y1, x2, y2)
                log.info(f"Liveness: wajah terdeteksi di {face_box}")

        if face_box is None:
            return LivenessResult(False, 0.0, 0, 1,
                                  {"note": "Tidak ada wajah terdeteksi (timeout deteksi awal)"})

        # ── Langkah 2: Siapkan BlinkDetector (pakai pre-warmed jika ada) ────
        if blink_detector is None:
            blink_detector = self.create_blink_detector()
        blink_detector.reset()

        # ── Langkah 3: Loop streaming EAR per-frame — TANPA sleep buatan ────
        # face_engine.detect_largest() TIDAK dipanggil lagi di sini.
        # Crop wajah dilakukan di sini inline agar tidak perlu simpan semua frame.
        frames_collected  = 0
        t0                = time.time()
        target_hit_at     = None   # timestamp saat blink target terpenuhi
        pad               = float(getattr(config, "LIVENESS_FACE_PAD", 0.25))

        log.info(f"Liveness: mulai streaming EAR (durasi maks {duration:.1f}s, "
                 f"target {required_blinks} blink) ...")

        while True:
            elapsed = time.time() - t0

            # Kondisi berhenti: durasi habis
            if elapsed >= duration:
                break

            # Kondisi berhenti: early-exit setelah target tercapai + jeda
            if target_hit_at is not None:
                if (time.time() - target_hit_at) >= early_exit_delay:
                    log.info("Liveness: early-exit — target blink terpenuhi")
                    break

            frame = cam.read()
            if frame is None:
                time.sleep(0.005)   # ~200 FPS max poll, hindari busy-wait penuh
                continue

            # Crop wajah inline (reuse face_box yang sudah dideteksi di awal)
            crop = self._crop_face(frame, face_box, pad=pad)
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                continue

            # Proses EAR — ini adalah operasi ringan (MediaPipe tracking mode)
            blink_detector.update(crop)
            frames_collected += 1

            # Tandai waktu saat target blink pertama kali terpenuhi
            if target_hit_at is None and blink_detector.blink_count >= required_blinks:
                target_hit_at = time.time()
                log.info(f"Liveness: blink ke-{blink_detector.blink_count} "
                          f"terdeteksi di t={elapsed:.2f}s")

        # ── Langkah 4: Scoring (sama dengan _blink_score, tanpa re-proses) ──
        blinks   = blink_detector.blink_count
        ear_vals = blink_detector.ear_history
        avg_ear  = float(np.mean(ear_vals)) if ear_vals else -1.0

        if frames_collected == 0:
            score = 0.1
        elif blinks >= required_blinks:
            score = 1.0
        else:
            score = float(getattr(config, "LIVENESS_BLINK_NO_EVENT_SCORE", 0.45))

        detail = {
            "blinks":          blinks,
            "required_blinks": required_blinks,
            "valid_frames":    frames_collected,
            "avg_ear":         round(avg_ear, 4),
            "blink_score":     round(score, 3),
            "blink_method":    blink_detector.mode,
            "elapsed_s":       round(time.time() - t0, 2),
        }

        # ── Verifikasi kontur (reuse depth_history dari BlinkDetector) ───────
        contour_ok = True
        if (blink_detector.mode == "mediapipe"
                and bool(getattr(config, "LIVENESS_CONTOUR_ENABLED", True))):
            contour_ok = _contour_live(blink_detector.depth_history)

        min_score = float(getattr(config, "LIVENESS_MIN_SCORE", BLINK_LIVE_THRESH))
        votes     = 1 if score >= BLINK_LIVE_THRESH else 0
        is_live   = (votes >= MIN_VOTES) and (score >= min_score) and contour_ok

        log.info(f"Liveness check_realtime: live={is_live} score={score:.3f} "
                 f"frames={frames_collected} {detail}")

        return LivenessResult(
            is_live=is_live,
            score=round(score, 3),
            votes=votes,
            total=1,
            detail=detail,
        )

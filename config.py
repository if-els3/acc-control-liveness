"""
=============================================================
config.py — Konfigurasi Global Sistem Kendali Akses
=============================================================
Sesuaikan nilai di sini dengan setup hardware Anda.
=============================================================
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Database ─────────────────────────────────────────────
DB_PATH = os.path.join(BASE_DIR, "database", "akses.db")
AES_KEY_HEX = "6fe627241f93987be6975820d45e2740"

# Path untuk BlazeFace (PyTorch)
BLAZEFACE_DIR = os.path.join(BASE_DIR, "BlazeFace-PyTorch")
MODEL_DIR = BLAZEFACE_DIR
BLAZEFACE_WEIGHTS = os.path.join(BLAZEFACE_DIR, "blazeface.pth")
BLAZEFACE_MODEL = BLAZEFACE_WEIGHTS
BLAZEFACE_ANCHORS = os.path.join(BLAZEFACE_DIR, "anchors.npy")

# Path untuk MobileFaceNet (TFLite)
MFN_DIR = os.path.join(BASE_DIR, "MobileFaceNet_TF")
MOBILEFACENET_MODEL = os.path.join(MFN_DIR, "mobilefacenet.tflite")
BLAZEFACE_URL     = "https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite"

# ─── Kamera ───────────────────────────────────────────────
CAMERA_INDEX   = 0
CAMERA_WIDTH   = 320
CAMERA_HEIGHT  = 240
CAMERA_FPS     = 30

# ─── Face Recognition ─────────────────────────────────────
DETECT_SIZE       = 128      # input BlazeFace
FACENET_SIZE      = 112      # input MobileFaceNet
TFLITE_THREADS    = 4        # jumlah thread TFLite (sesuai core Raspi)
DETECT_CONFIDENCE = 0.6      # ambang deteksi wajah
FACE_MATCH_THRESH = 0.55     # ambang cosine similarity (lebih tinggi = lebih ketat)
ENROLL_FRAMES     = 5        # jumlah frame diambil saat pendaftaran

# ─── RFID MFRC522 ─────────────────────────────────────────
SPI_BUS    = 0
SPI_DEVICE = 0               # CE0 = pin fisik 24
SPI_SPEED  = 1_000_000
# RFCfgReg gain boost untuk clone chip (0x70 = 48dB max)
RFID_GAIN  = 0x70
# Timeout scan kartu (detik)
RFID_TIMEOUT = 15

# ─── Servo / Pintu ────────────────────────────────────────
# GPIO BCM numbering
SERVO_PIN      = 18          # GPIO18 = pin fisik 12 (PWM0)
SERVO_FREQ     = 50          # Hz
SERVO_OPEN     = 7.5         # duty cycle posisi terbuka  (~90°)
SERVO_CLOSED   = 2.5         # duty cycle posisi tertutup (~0°)
DOOR_OPEN_SEC  = 5           # detik pintu tetap terbuka

# ─── LCD (opsional, I2C) ──────────────────────────────────
LCD_ENABLED    = False       # set True jika LCD terpasang
LCD_I2C_ADDR   = 0x27
LCD_COLS       = 16
LCD_ROWS       = 2

# ─── Log ──────────────────────────────────────────────────
LOG_DIR  = os.path.join(BASE_DIR, "logs_data")
LOG_FILE = os.path.join(LOG_DIR, "system.log")

# ─── Tampilan CLI ─────────────────────────────────────────
APP_NAME    = "Sistem Kendali Akses"
APP_VERSION = "1.0.0"

# ─── Liveness Detection ───────────────────────────────────
LIVENESS_ENABLED       = True    # False = skip liveness, RFID+face saja
DEBUG_EYE_TRACKER      = False   # False = production (tidak simpan debug_eye.jpg)
LIVENESS_DURATION      = 4.0     # detik maks pengambilan frame (dikurangi dari 5.0)
LIVENESS_MIN_SCORE     = 0.60    # threshold skor blink final
LIVENESS_MIN_VOTES     = 1       # cukup 1 vote LIVE
LIVENESS_FACE_PAD      = 0.25    # padding crop wajah
LIVENESS_EARLY_EXIT_DELAY = 1.0  # detik tambahan setelah blink terdeteksi lalu keluar
LIVENESS_MAX_VERIFY_FRAMES = 10  # maks frame yg diproses untuk verifikasi wajah

# ── EAR (Eye Aspect Ratio) — METODE UTAMA ─────────────────
# Digunakan jika MediaPipe terinstall (pip install mediapipe).
# EAR normal saat mata terbuka : ~0.25 – 0.35
# EAR saat berkedip            : < 0.20 (tergantung orang)
# Turunkan BLINK_EAR_THRESHOLD jika terlalu banyak false-positive.
# Naikkan jika blink sulit terdeteksi.
BLINK_EAR_THRESHOLD    = 0.21   # EAR di bawah ini = mata tertutup
BLINK_EAR_CONSEC_FRAMES = 1     # min frame dengan EAR < threshold agar dihitung blink

# ── Blink count & scoring ─────────────────────────────────
# PENTING: LIVENESS_BLINK_NO_EVENT_SCORE HARUS < LIVENESS_BLINK_SCORE_THRESH
# agar wajah diam / foto tidak otomatis lulus.
LIVENESS_BLINK_MIN_COUNT      = 1     # minimal 1 blink event agar dianggap live
LIVENESS_BLINK_SCORE_THRESH   = 0.60  # threshold voting blink
LIVENESS_BLINK_NO_EVENT_SCORE = 0.45  # score fallback jika 0 blink — HARUS < 0.60
LIVENESS_BLINK_MIN_CLOSED_FRAMES = 2  # (Haar fallback) min frame tertutup
LIVENESS_BLINK_MAX_CLOSED_FRAMES = 10 # (Haar fallback) max frame tertutup

# ── Haar fallback (jika MediaPipe tidak ada) ───────────────
BLINK_EYE_SCALE_FACTOR  = 1.10
BLINK_EYE_MIN_NEIGHBORS = 3
BLINK_EYE_MIN_SIZE      = (12, 12)
BLINK_CLAHE_CLIP_LIMIT  = 1.5
BLINK_CLAHE_TILE_GRID   = (8, 8)
BLINK_GAMMA             = 1.0

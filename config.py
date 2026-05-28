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
LIVENESS_DURATION      = 4.5    # detik pengambilan frame untuk liveness (lebih panjang untuk menangkap blink)
LIVENESS_MIN_SCORE     = 0.50   # threshold skor blink final
LIVENESS_MIN_VOTES     = 1      # blink-only: cukup 1 vote LIVE

# Blink-specific tuning (kacamata + low-light)
LIVENESS_BLINK_MIN_COUNT      = 1      # minimal blink event agar dianggap live
LIVENESS_BLINK_SCORE_THRESH   = 0.50   # threshold score dari modul blink
LIVENESS_BLINK_NO_EVENT_SCORE = 0.58   # score fallback saat mata terlihat tapi belum berkedip
LIVENESS_FACE_PAD             = 0.25   # padding crop wajah agar mata tidak terpotong
LIVENESS_BLINK_MIN_CLOSED_FRAMES = 1   # min frame mata tertutup agar dihitung blink
LIVENESS_BLINK_MAX_CLOSED_FRAMES = 8   # batas atas closure agar tidak terlalu panjang dianggap blink

# Haar eye detection tuning
BLINK_EYE_SCALE_FACTOR  = 1.08
BLINK_EYE_MIN_NEIGHBORS = 1
BLINK_EYE_MIN_SIZE      = (8, 8)

# Pre-processing untuk low-light
BLINK_CLAHE_CLIP_LIMIT  = 3.0
BLINK_CLAHE_TILE_GRID   = (8, 8)
BLINK_GAMMA             = 1.25

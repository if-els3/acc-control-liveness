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
DEBUG_EYE_TRACKER      = True    # True = tampilkan window visualisasi tracker mata
LIVENESS_DURATION      = 5.0     # detik pengambilan frame (diperpanjang agar blink punya waktu cukup)
LIVENESS_MIN_SCORE     = 0.60    # threshold skor blink final (dinaikkan agar fallback 0.45 tidak lolos)
LIVENESS_MIN_VOTES     = 1       # blink-only: cukup 1 vote LIVE

# Blink-specific tuning (cahaya normal 195-300 lux / kacamata)
# BUG-FIX: LIVENESS_BLINK_NO_EVENT_SCORE HARUS lebih kecil dari
#           LIVENESS_BLINK_SCORE_THRESH dan LIVENESS_MIN_SCORE agar
#           wajah diam (foto / tidak berkedip) tidak lolos verifikasi.
LIVENESS_BLINK_MIN_COUNT      = 1      # minimal 1 blink event agar dianggap live
LIVENESS_BLINK_SCORE_THRESH   = 0.60   # threshold score modul blink (sama dengan MIN_SCORE)
LIVENESS_BLINK_NO_EVENT_SCORE = 0.45   # fallback saat mata terlihat tapi belum berkedip
                                        # HARUS < LIVENESS_BLINK_SCORE_THRESH agar tidak auto-lulus
LIVENESS_FACE_PAD             = 0.25   # padding crop wajah agar mata tidak terpotong
LIVENESS_BLINK_MIN_CLOSED_FRAMES = 2   # min 2 frame mata tertutup agar dihitung blink (noise filter)
LIVENESS_BLINK_MAX_CLOSED_FRAMES = 10  # batas atas closure (diperlonggar sedikit untuk kedip lambat)

# Haar eye detection tuning — dioptimasi untuk cahaya normal (195–300 lux)
# minNeighbors yang terlalu kecil (1) menyebabkan false-positive di setiap frame
# sehingga state tidak pernah masuk "closed" dan blink tidak tercatat.
BLINK_EYE_SCALE_FACTOR  = 1.10   # sedikit lebih besar agar deteksi lebih stabil
BLINK_EYE_MIN_NEIGHBORS = 3      # naik dari 1 → 3 untuk mengurangi false-positive di cahaya normal
BLINK_EYE_MIN_SIZE      = (12, 12)  # ukuran minimum sedikit lebih besar agar noise tidak terdeteksi

# Pre-processing — dioptimasi untuk cahaya normal / terang (195–300 lux)
# CLAHE clipLimit tinggi + gamma > 1 hanya efektif untuk low-light (<100 lux);
# di cahaya normal malah over-enhance sehingga tepi kelopak mata hilang.
BLINK_CLAHE_CLIP_LIMIT  = 1.5    # diturunkan dari 3.0 → 1.5 untuk cahaya normal
BLINK_CLAHE_TILE_GRID   = (8, 8)
BLINK_GAMMA             = 1.0    # gamma netral (diturunkan dari 1.25) — tidak perlu boost di cahaya normal

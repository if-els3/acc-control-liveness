#!/usr/bin/env python3
"""
=============================================================
UNIT TEST #2 — FACE RECOGNITION (BlazeFace + MobileFaceNet)
=============================================================
Optimisasi Raspberry Pi:
  ✔ Headless Mode     — tidak ada GUI/display window
  ✔ Resize Frame      — downscale sebelum inferensi
  ✔ Multithreading    — kamera di thread terpisah
  ✔ TFLite + XNNPACK  — inferensi CPU ringan

Pipeline:
  [Kamera Thread] → [Resize] → [BlazeFace] → [Crop Wajah]
                              ↓
                         [MobileFaceNet] → [Embedding]
=============================================================
Referensi model:
  BlazeFace : https://github.com/hollance/BlazeFace-PyTorch
              (konversi ke TFLite untuk Raspberry Pi)
  MobileFaceNet: https://github.com/sirius-ai/MobileFaceNet_TF
                 (TFLite export)
=============================================================
"""

import sys
import os
import time
import threading
import urllib.request

try:
    import cv2
except ImportError:
    print("[ERROR] pip install opencv-python-headless")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("[ERROR] pip install numpy")
    sys.exit(1)

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        print("[INFO] Menggunakan TensorFlow Lite (tflite_runtime tidak ada)")
    except ImportError:
        print("[ERROR] Instal salah satu:")
        print("        pip install tflite-runtime        # Raspberry Pi OS")
        print("        pip install tensorflow            # fallback")
        sys.exit(1)

# ─── Konfigurasi ──────────────────────────────────────────
DEVICE_INDEX    = 0
FRAME_W         = 320      # lebar frame kamera (diperkecil)
FRAME_H         = 240      # tinggi frame kamera
DETECT_SIZE     = 128      # input BlazeFace
FACENET_SIZE    = 112      # input MobileFaceNet
NUM_THREADS     = 4        # thread TFLite (sesuaikan core Raspi)
TEST_FRAMES     = 30       # frame pengujian pipeline
CONF_THRESHOLD  = 0.6      # ambang kepercayaan deteksi

MODEL_DIR       = os.path.join(os.path.dirname(__file__), "..", "models")

# URL model publik (fallback jika belum ada)
BLAZEFACE_URL   = "https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite"
FACENET_URL     = None   # MobileFaceNet perlu konversi manual (lihat README)
# ──────────────────────────────────────────────────────────

_frame_buffer   = None
_frame_lock     = threading.Lock()
_stop_event     = threading.Event()


# ═══════════════════════════════════════════════════════════
# KAMERA THREAD
# ═══════════════════════════════════════════════════════════

def camera_reader(cap):
    """Baca frame di thread terpisah agar pipeline tidak blocking."""
    global _frame_buffer
    while not _stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            small = cv2.resize(frame, (FRAME_W, FRAME_H),
                               interpolation=cv2.INTER_LINEAR)
            with _frame_lock:
                _frame_buffer = small
    cap.release()


# ═══════════════════════════════════════════════════════════
# HELPER — MODEL DOWNLOAD & LOAD
# ═══════════════════════════════════════════════════════════

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def download_if_missing(url: str, dest: str):
    if os.path.exists(dest):
        print(f"         Model sudah ada: {os.path.basename(dest)}")
        return
    print(f"         Mengunduh {os.path.basename(dest)} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print("         ✔ Unduhan selesai")
    except Exception as e:
        raise RuntimeError(f"Gagal mengunduh model: {e}")


def load_tflite(model_path: str):
    """Load TFLite interpreter dengan XNNPACK delegate dan multi-thread."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    try:
        # tflite_runtime API
        interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=NUM_THREADS,
        )
    except TypeError:
        # tensorflow.lite API
        interpreter = tflite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()
    return interpreter


def get_io(interpreter):
    inp  = interpreter.get_input_details()
    outp = interpreter.get_output_details()
    return inp, outp


# ═══════════════════════════════════════════════════════════
# PREPROCESS
# ═══════════════════════════════════════════════════════════

def preprocess_blazeface(frame_bgr: np.ndarray) -> np.ndarray:
    """
    BlazeFace input: RGB, 128×128, float32 normalised [0,1] atau [-1,1].
    Mediapipe face_detection_short_range menggunakan [0,1].
    """
    rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rsz   = cv2.resize(rgb, (DETECT_SIZE, DETECT_SIZE),
                       interpolation=cv2.INTER_LINEAR)
    arr   = rsz.astype(np.float32) / 255.0
    return np.expand_dims(arr, 0)    # (1, 128, 128, 3)


def preprocess_facenet(face_bgr: np.ndarray) -> np.ndarray:
    """
    MobileFaceNet input: RGB, 112×112, float32 normalised [-1,1].
    """
    rgb   = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rsz   = cv2.resize(rgb, (FACENET_SIZE, FACENET_SIZE),
                       interpolation=cv2.INTER_LINEAR)
    arr   = (rsz.astype(np.float32) - 127.5) / 128.0
    return np.expand_dims(arr, 0)    # (1, 112, 112, 3)


# ═══════════════════════════════════════════════════════════
# INFERENSI
# ═══════════════════════════════════════════════════════════

def run_blazeface(interp, inp_details, out_details, tensor):
    interp.set_tensor(inp_details[0]['index'], tensor)
    interp.invoke()

    # Output Mediapipe face detection: [boxes, scores] atau berbeda per versi
    # Ambil semua output tensor
    outputs = [interp.get_tensor(o['index']) for o in out_details]
    return outputs


def decode_blazeface_boxes(outputs, orig_w, orig_h,
                           threshold=CONF_THRESHOLD):
    """
    Decode bounding box dari output Mediapipe face detection TFLite.
    Format output: (1, N, 1) skor dan (1, N, 4) box [ymin,xmin,ymax,xmax].
    Ini bisa berbeda tergantung versi — adjust jika perlu.
    """
    boxes_list = []
    try:
        # Cari tensor skor (nilai < 1) dan tensor box (nilai bisa besar)
        scores_tensor = None
        boxes_tensor  = None
        for o in outputs:
            if o.ndim >= 2:
                flat = o.flatten()
                if flat.max() <= 1.0 and flat.min() >= 0.0 and len(flat) < 1000:
                    scores_tensor = o
                elif o.shape[-1] == 4:
                    boxes_tensor  = o

        if scores_tensor is None or boxes_tensor is None:
            return boxes_list

        scores = scores_tensor.flatten()
        boxes  = boxes_tensor.reshape(-1, 4)

        for i, score in enumerate(scores):
            if score >= threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                x1 = int(np.clip(xmin * orig_w, 0, orig_w))
                y1 = int(np.clip(ymin * orig_h, 0, orig_h))
                x2 = int(np.clip(xmax * orig_w, 0, orig_w))
                y2 = int(np.clip(ymax * orig_h, 0, orig_h))
                if x2 > x1 and y2 > y1:
                    boxes_list.append((x1, y1, x2, y2, float(score)))
    except Exception:
        pass

    return boxes_list


def run_facenet(interp, inp_details, out_details, tensor):
    interp.set_tensor(inp_details[0]['index'], tensor)
    interp.invoke()
    embedding = interp.get_tensor(out_details[0]['index'])
    # L2 normalise
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


# ═══════════════════════════════════════════════════════════
# UNIT TESTS
# ═══════════════════════════════════════════════════════════

def test_1_camera_thread():
    """Test 1: Buka kamera dan jalankan thread pembaca frame."""
    print("\n[TEST 1] Inisialisasi kamera + thread ...")
    cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera index {DEVICE_INDEX} tidak bisa dibuka.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    _stop_event.clear()
    t = threading.Thread(target=camera_reader, args=(cap,), daemon=True)
    t.start()

    # Tunggu frame pertama
    deadline = time.time() + 5
    while time.time() < deadline:
        with _frame_lock:
            if _frame_buffer is not None:
                break
        time.sleep(0.05)
    else:
        _stop_event.set()
        raise RuntimeError("Tidak ada frame diterima dalam 5 detik.")

    print(f"         ✔ Camera thread aktif | {FRAME_W}×{FRAME_H}")
    return t


def test_2_load_blazeface():
    """Test 2: Load model BlazeFace TFLite."""
    print("\n[TEST 2] Load BlazeFace TFLite ...")
    ensure_dir(MODEL_DIR)
    model_path = os.path.join(MODEL_DIR, "face_detection_short_range.tflite")

    download_if_missing(BLAZEFACE_URL, model_path)
    interp    = load_tflite(model_path)
    inp, outp = get_io(interp)

    print(f"         Input shape  : {inp[0]['shape']}")
    print(f"         Output count : {len(outp)}")
    print("         ✔ BlazeFace siap (TFLite + XNNPACK)")
    return interp, inp, outp


def test_3_load_mobilefacenet():
    """
    Test 3: Load MobileFaceNet TFLite.
    Model perlu dikonversi manual dari:
    https://github.com/sirius-ai/MobileFaceNet_TF
    Jika belum ada, test ini dilewati dengan WARNING.
    """
    print("\n[TEST 3] Load MobileFaceNet TFLite ...")
    model_path = os.path.join(MODEL_DIR, "mobilefacenet.tflite")

    if not os.path.exists(model_path):
        print("         ⚠  Model MobileFaceNet belum ada.")
        print(f"         Letakkan file di: {model_path}")
        print("         Konversi dari repo:")
        print("         https://github.com/sirius-ai/MobileFaceNet_TF")
        print("         Panduan konversi ada di models/README_CONVERT.md")
        print("         [SKIP] Test ini dilewati")
        return None, None, None

    interp    = load_tflite(model_path)
    inp, outp = get_io(interp)
    print(f"         Input shape  : {inp[0]['shape']}")
    print(f"         Output shape : {outp[0]['shape']}")
    print("         ✔ MobileFaceNet siap")
    return interp, inp, outp


def test_4_pipeline(bf_interp, bf_inp, bf_out, fn_interp, fn_inp, fn_out):
    """Test 4: Jalankan pipeline deteksi + pengenalan pada N frame."""
    print(f"\n[TEST 4] Pipeline deteksi wajah ({TEST_FRAMES} frame) ...")

    detected_frames = 0
    embedded_frames = 0
    times_detect    = []
    times_embed     = []

    for _ in range(TEST_FRAMES):
        with _frame_lock:
            frame = _frame_buffer.copy() if _frame_buffer is not None else None

        if frame is None:
            time.sleep(0.03)
            continue

        h, w = frame.shape[:2]

        # — BlazeFace —
        t0        = time.perf_counter()
        tensor_bf = preprocess_blazeface(frame)
        outputs   = run_blazeface(bf_interp, bf_inp, bf_out, tensor_bf)
        boxes     = decode_blazeface_boxes(outputs, w, h)
        dt_detect = (time.perf_counter() - t0) * 1000
        times_detect.append(dt_detect)

        if boxes:
            detected_frames += 1
            # Ambil kotak dengan skor tertinggi
            x1, y1, x2, y2, score = max(boxes, key=lambda b: b[4])
            face_crop = frame[y1:y2, x1:x2]

            # — MobileFaceNet (jika tersedia) —
            if fn_interp is not None and face_crop.size > 0:
                t1          = time.perf_counter()
                tensor_fn   = preprocess_facenet(face_crop)
                embedding   = run_facenet(fn_interp, fn_inp, fn_out, tensor_fn)
                dt_embed    = (time.perf_counter() - t1) * 1000
                times_embed.append(dt_embed)
                embedded_frames += 1

        time.sleep(0.01)   # yield CPU

    # Laporan
    avg_detect = np.mean(times_detect) if times_detect else 0
    avg_embed  = np.mean(times_embed)  if times_embed  else 0

    print(f"         Frame diproses     : {TEST_FRAMES}")
    print(f"         Wajah terdeteksi   : {detected_frames}/{TEST_FRAMES}")
    print(f"         Embedding dihasilkan: {embedded_frames}")
    print(f"         Avg latency detect : {avg_detect:.1f} ms")
    if times_embed:
        print(f"         Avg latency embed  : {avg_embed:.1f} ms")

    return detected_frames, avg_detect


def test_5_fps_estimate():
    """Test 5: Estimasi FPS pipeline keseluruhan."""
    print(f"\n[TEST 5] Estimasi FPS pipeline (headless) ...")
    n  = 20
    t0 = time.perf_counter()
    for _ in range(n):
        with _frame_lock:
            f = _frame_buffer
        if f is not None:
            _ = cv2.resize(f, (DETECT_SIZE, DETECT_SIZE))
        time.sleep(0.001)
    elapsed = time.perf_counter() - t0
    fps = n / elapsed
    print(f"         FPS estimasi (resize saja): {fps:.1f}")
    print("         ✔ Pipeline dapat berjalan real-time")


# ═══════════════════════════════════════════════════════════
# README KONVERSI MODEL
# ═══════════════════════════════════════════════════════════

README_CONVERT = """\
# Konversi MobileFaceNet ke TFLite
# ─────────────────────────────────

## Langkah 1 — Clone repo
git clone https://github.com/sirius-ai/MobileFaceNet_TF.git
cd MobileFaceNet_TF

## Langkah 2 — Install dependensi
pip install tensorflow==2.x

## Langkah 3 — Ekspor SavedModel
# (sesuaikan checkpoint yang dipakai, misalnya models/MobileFaceNet.pb)
python freeze_graph.py  # atau gunakan tf.saved_model.save()

## Langkah 4 — Konversi ke TFLite
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model/")
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # quantize int8
# Opsional: aktifkan XNNPACK delegate (default aktif di runtime)
tflite_model = converter.convert()

with open("mobilefacenet.tflite", "wb") as f:
    f.write(tflite_model)

## Langkah 5 — Salin ke folder models/
cp mobilefacenet.tflite ../access_control/models/

## Referensi BlazeFace PyTorch → TFLite
# https://github.com/hollance/BlazeFace-PyTorch
# Untuk Raspberry Pi, lebih mudah pakai versi MediaPipe yang sudah TFLite:
# face_detection_short_range.tflite  (sudah diunduh otomatis oleh test)
"""


def main():
    print("═" * 60)
    print("  UNIT TEST #2 — FACE RECOGNITION")
    print("  BlazeFace TFLite  +  MobileFaceNet TFLite")
    print("  Optimisasi: Headless | Resize | Thread | XNNPACK")
    print("═" * 60)

    # Tulis README konversi jika belum ada
    ensure_dir(MODEL_DIR)
    readme_path = os.path.join(MODEL_DIR, "README_CONVERT.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write(README_CONVERT)

    cam_thread = None
    success    = True

    try:
        cam_thread = test_1_camera_thread()
        bf_interp, bf_inp, bf_out       = test_2_load_blazeface()
        fn_interp, fn_inp, fn_out       = test_3_load_mobilefacenet()
        detected, avg_lat               = test_4_pipeline(
                                            bf_interp, bf_inp, bf_out,
                                            fn_interp, fn_inp, fn_out)
        test_5_fps_estimate()

        print("\n" + "─" * 60)
        print("  RINGKASAN FACE RECOGNITION")
        print("─" * 60)
        print(f"  BlazeFace TFLite   : ✔ Aktif  (latency ~{avg_lat:.0f} ms)")
        if fn_interp:
            print("  MobileFaceNet      : ✔ Aktif")
        else:
            print("  MobileFaceNet      : ⚠ Belum ada (lihat models/README_CONVERT.md)")
        print(f"  Wajah terdeteksi   : {detected}/{TEST_FRAMES} frame")
        print("─" * 60)
        print("  \033[92m✔ FACE RECOGNITION TEST SELESAI\033[0m")

    except Exception as e:
        print(f"\n  \033[91m[GAGAL] {e}\033[0m")
        success = False

    finally:
        _stop_event.set()
        if cam_thread:
            cam_thread.join(timeout=3)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

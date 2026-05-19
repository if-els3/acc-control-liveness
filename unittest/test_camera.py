#!/usr/bin/env python3
"""
=============================================================
UNIT TEST #1 — KAMERA USB
=============================================================
Target : Validasi kamera terbuka, resolusi, dan frame rate
Mode   : Headless (tanpa GUI), capture 60 frame lalu hitung FPS
=============================================================
"""

import sys
import time
import threading

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV belum terinstall. Jalankan:")
    print("        pip install opencv-python-headless")
    sys.exit(1)

# ─── Konfigurasi ──────────────────────────────────────────
DEVICE_INDEX  = 0       # /dev/video0 — ubah jika perlu
TARGET_W      = 320     # lebar frame (dikecilkan untuk hemat CPU)
TARGET_H      = 240     # tinggi frame
FRAME_COUNT   = 60      # jumlah frame yang di-capture untuk hitung FPS
MIN_FPS       = 10.0    # ambang batas FPS minimum yang diterima
# ──────────────────────────────────────────────────────────

_latest_frame = None
_lock         = threading.Lock()
_stop_flag    = False


def camera_thread(cap):
    """Thread pembaca frame agar FPS stabil (non-blocking)."""
    global _latest_frame, _stop_flag
    while not _stop_flag:
        ret, frame = cap.read()
        if ret:
            with _lock:
                _latest_frame = frame


def test_open_camera():
    """Test 1: Buka kamera."""
    print("\n[TEST 1] Membuka kamera ...")
    cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera index {DEVICE_INDEX} tidak bisa dibuka. "
                           "Pastikan kamera terhubung dan tidak dipakai proses lain.")
    print(f"         ✔ Kamera terbuka  (index={DEVICE_INDEX})")
    return cap


def test_set_resolution(cap):
    """Test 2: Set resolusi yang diinginkan."""
    print(f"\n[TEST 2] Set resolusi {TARGET_W}×{TARGET_H} ...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"         Resolusi aktual   : {actual_w}×{actual_h}")

    if actual_w == 0 or actual_h == 0:
        raise RuntimeError("Driver melaporkan resolusi 0×0.")
    print("         ✔ Resolusi valid")
    return actual_w, actual_h


def test_fps(cap):
    """Test 3: Ukur FPS nyata menggunakan thread pembaca frame."""
    global _stop_flag, _latest_frame
    print(f"\n[TEST 3] Mengukur FPS ({FRAME_COUNT} frame, thread mode) ...")

    _stop_flag = False
    t = threading.Thread(target=camera_thread, args=(cap,), daemon=True)
    t.start()

    # Tunggu frame pertama
    deadline = time.time() + 5
    while time.time() < deadline:
        with _lock:
            if _latest_frame is not None:
                break
        time.sleep(0.05)
    else:
        _stop_flag = True
        raise RuntimeError("Tidak ada frame diterima dalam 5 detik.")

    # Hitung FPS
    count  = 0
    t0     = time.perf_counter()
    prev_frame = None
    while count < FRAME_COUNT:
        with _lock:
            frame = _latest_frame
        if frame is not prev_frame:
            prev_frame = frame
            count += 1
        else:
            time.sleep(0.001)

    elapsed = time.perf_counter() - t0
    _stop_flag = True
    t.join(timeout=2)

    fps = FRAME_COUNT / elapsed
    print(f"         FPS terukur        : {fps:.2f}")
    if fps < MIN_FPS:
        raise RuntimeError(
            f"FPS terlalu rendah ({fps:.1f} < {MIN_FPS}). "
            "Coba turunkan resolusi atau periksa koneksi USB."
        )
    print(f"         ✔ FPS memenuhi syarat (≥{MIN_FPS})")
    return fps


def test_frame_content():
    """Test 4: Frame tidak kosong / hitam penuh."""
    print("\n[TEST 4] Memeriksa konten frame ...")
    with _lock:
        frame = _latest_frame

    if frame is None:
        raise RuntimeError("Tidak ada frame tersedia.")

    mean_brightness = frame.mean()
    print(f"         Mean brightness   : {mean_brightness:.2f}")
    if mean_brightness < 5:
        raise RuntimeError(
            "Frame terlalu gelap (brightness < 5) — lensa mungkin tertutup."
        )
    print("         ✔ Frame berisi gambar valid")
    return frame.shape


def main():
    print("═" * 56)
    print("  UNIT TEST #1 — KAMERA USB")
    print("═" * 56)

    cap = None
    try:
        cap        = test_open_camera()
        w, h       = test_set_resolution(cap)
        fps        = test_fps(cap)
        shape      = test_frame_content()

        print("\n" + "─" * 56)
        print("  RINGKASAN KAMERA")
        print("─" * 56)
        print(f"  Resolusi   : {w} × {h}")
        print(f"  FPS nyata  : {fps:.2f}")
        print(f"  Shape array: {shape}")
        print("─" * 56)
        print("  \033[92m✔ SEMUA TEST KAMERA LULUS\033[0m")

    except Exception as e:
        print(f"\n  \033[91m[GAGAL] {e}\033[0m")
        sys.exit(1)
    finally:
        if cap is not None:
            cap.release()

    sys.exit(0)


if __name__ == "__main__":
    main()

"""
=============================================================
test_threshold_capture.py — Capture Raw Cosine Similarity Score
=============================================================
Script standalone buat Plan 2 (raw-score sweep threshold testing).
Tidak modifikasi core/menus — baca DB + FaceEngine langsung, log raw
float score ke CSV buat dianalisis pakai analyze_threshold.py

Cara pakai:
  1. Taruh file ini di root project (sejajar main.py, config.py)
  2. python3 test_threshold_capture.py
  3. Input participant_id, label (genuine/impostor), UID target
  4. Hadapkan wajah ke kamera tiap capture
=============================================================
"""
import csv
import time
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from core.database import Database
from core.face_engine import FaceEngine
from core.camera_stream import CameraStream

CSV_PATH = os.path.join(config.BASE_DIR, "threshold_test_scores.csv")


def init_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "participant_id", "label", "target_rfid_uid", "raw_score", "n_frames"])


def capture_score(face_engine, cam, target_embeddings, timeout=5.0, min_frames=3):
    """
    Ambil beberapa frame sampai dapat min_frames wajah valid,
    hitung cosine similarity terbaik terhadap target_embeddings.
    """
    t0 = time.time()
    best_score = 0.0
    frames_used = 0

    while time.time() - t0 < timeout:
        frame = cam.read()
        if frame is None:
            time.sleep(0.05)
            continue

        emb = face_engine.extract_embedding(frame)
        if emb is not None:
            frames_used += 1
            for stored in target_embeddings:
                s_arr = np.array(stored, dtype=np.float32)
                score = float(np.dot(emb, s_arr))  # cosine similarity (embedding sudah L2-normalized)
                if score > best_score:
                    best_score = score
            if frames_used >= min_frames:
                break
        time.sleep(0.1)

    return best_score, frames_used


def main():
    init_csv()
    print("=== Threshold Test — Raw Score Capture (Plan 2) ===\n")

    db = Database()
    face_engine = FaceEngine()

    print("Loading FaceEngine...")
    if not face_engine.load():
        print("[!] FaceEngine gagal load. Cek model BlazeFace/MobileFaceNet.")
        return

    cam = CameraStream()
    if not cam.start():
        print("[!] Kamera gagal dibuka.")
        return

    print(f"FaceEngine mode: {face_engine.mode}")
    print(f"CSV output: {CSV_PATH}")
    print("Ketik 'exit' di Participant ID buat selesai.\n")

    try:
        while True:
            participant_id = input("Participant ID (misal: P01): ").strip()
            if participant_id.lower() == "exit":
                break

            label = input("Label [genuine/impostor]: ").strip().lower()
            if label not in ("genuine", "impostor"):
                print("  [!] Label harus 'genuine' atau 'impostor'. Ulangi.\n")
                continue

            target_uid = input("RFID UID target (user terdaftar yang dibandingkan): ").strip()
            target_embeddings = db.get_embeddings(target_uid)
            if not target_embeddings:
                print(f"  [!] Tidak ada embedding tersimpan untuk UID {target_uid}. Ulangi.\n")
                continue

            input("  Hadapkan wajah ke kamera, tekan Enter buat mulai capture...")
            score, n_frames = capture_score(face_engine, cam, target_embeddings)

            if n_frames == 0:
                print("  [!] Wajah tidak terdeteksi selama timeout. Ulangi percobaan.\n")
                continue

            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    participant_id,
                    label,
                    target_uid,
                    round(score, 6),
                    n_frames,
                ])

            print(f"  ✔ Score tercatat: {score:.4f}  ({n_frames} frame valid)\n")

    finally:
        cam.stop()
        print(f"\nSelesai. Data tersimpan di: {CSV_PATH}")


if __name__ == "__main__":
    main()

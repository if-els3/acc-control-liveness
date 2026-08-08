"""
=============================================================
run_prototype_test.py — Uji Prototipe ACC-Control Liveness
=============================================================
Menjalankan 3 kelompok uji terpisah dan mencetak analisis:

  [A] BlazeFace
      - Akurasi deteksi (detection rate)
      - Rerata inference time & FPS
      - Confidence score per-frame

  [B] Liveness (EAR Blink Detection)
      - Debug visualisasi: EAR timeline per-frame
      - Export CSV untuk analisis APCER / BPCER / ACER
      - Blink count vs actual blink (manual input)

  [C] MobileFaceNet
      - Embedding cosine similarity distribution
      - Placeholder FAR / FRR / EER / Accuracy
        (diisi manual dari hasil uji pengguna)

Jalankan: python run_prototype_test.py [--blazeface] [--liveness]
                                       [--facenet] [--all]
Tanpa argumen → mode interaktif (pilih menu).
=============================================================
"""

import os, sys, time, csv, argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from core.face_engine import FaceEngine
from core.liveness import (
    LivenessDetector, BlinkDetector, MP_OK,
    _compute_ear, _compute_contour_depth_ratio,
    _LEFT_EYE_IDX, _RIGHT_EYE_IDX,
    _NOSE_TIP_IDX, _L_EYE_OUTER_IDX, _R_EYE_OUTER_IDX,
    _L_MOUTH_IDX, _R_MOUTH_IDX,
)

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
except ImportError:
    mp = None
    mp_face_mesh = None

# ─── Output directories ────────────────────────────────────────
OUT_DIR   = "test_output"
CSV_DIR   = os.path.join(OUT_DIR, "csv")
IMG_DIR   = os.path.join(OUT_DIR, "img")
for d in (OUT_DIR, CSV_DIR, IMG_DIR):
    os.makedirs(d, exist_ok=True)

# ─── Drawing constants ─────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_SIMPLEX
CLR_GREEN  = (80,  220,  80)
CLR_CYAN   = (220, 220,   0)
CLR_YELLOW = (0,   220, 220)
CLR_RED    = (60,   60, 255)
CLR_WHITE  = (240, 240, 240)
CLR_GRAY   = (140, 140, 140)
CLR_ORANGE = (0,   165, 255)
SEP        = "─" * 60
SEP2       = "═" * 60

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _txt(img, text, pos, scale=0.5, color=CLR_GREEN, th=1):
    cv2.putText(img, text, pos, FONT, scale, (0, 0, 0), th + 2)
    cv2.putText(img, text, pos, FONT, scale, color, th)

def _header(img, title, sub="", bar=(30, 30, 30)):
    cv2.rectangle(img, (0, 0), (img.shape[1], 50), bar, -1)
    _txt(img, title, (10, 22), 0.65, CLR_WHITE, 2)
    _txt(img, sub,   (10, 42), 0.42, CLR_GRAY,  1)
    cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1), (80, 80, 80), 1)

def _panel(img, x1, y1, x2, y2, border_color=CLR_GRAY):
    cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1)

def capture_frames(n=20, cam_index=0, warmup=2.0):
    """Ambil n frame dari kamera. Fallback ke dummy frame jika gagal."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("  [!] Kamera tidak ditemukan — menggunakan dummy frame.")
        dummy = np.full((240, 320, 3), 40, dtype=np.uint8)
        _txt(dummy, "NO CAMERA", (80, 120), 0.9, CLR_RED, 2)
        return [dummy] * n

    print(f"  Warmup kamera {warmup:.0f}s ...", end="", flush=True)
    time.sleep(warmup)
    print(" OK")

    frames = []
    for _ in range(n):
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()

    if not frames:
        print("  [!] Gagal baca frame — menggunakan dummy.")
        dummy = np.full((240, 320, 3), 40, dtype=np.uint8)
        return [dummy] * n

    return frames

def _save_csv(filename, header, rows):
    path = os.path.join(CSV_DIR, filename)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  CSV saved: {path}")
    return path

# ══════════════════════════════════════════════════════════════
# [A] UJI BLAZEFACE
# ══════════════════════════════════════════════════════════════

def uji_blazeface(engine: FaceEngine, frames: list):
    print(f"\n{SEP2}")
    print("  [A] UJI BLAZEFACE — Akurasi Deteksi & Performa")
    print(SEP2)

    results    = []   # (frame_idx, detected, conf, inf_ms)
    total      = len(frames)
    detected_n = 0
    inf_times  = []

    for i, frame in enumerate(frames):
        t0  = time.perf_counter()
        boxes = engine.detect_with_landmarks(frame)
        inf_ms = (time.perf_counter() - t0) * 1000

        detected = len(boxes) > 0
        best_conf = max(b[4] for b in boxes) if detected else 0.0

        results.append((i, int(detected), round(best_conf, 4), round(inf_ms, 2)))
        inf_times.append(inf_ms)
        if detected:
            detected_n += 1

        bar = "█" * int(best_conf * 20) if detected else "─" * 20
        print(f"  frame {i+1:>2}/{total}  det={'YES' if detected else 'NO ':3}  "
              f"conf={best_conf:.3f}  inf={inf_ms:.1f}ms  [{bar}]")

    # ── Statistik ────────────────────────────────────────────
    det_rate  = detected_n / total * 100
    avg_inf   = np.mean(inf_times)
    fps_est   = 1000 / avg_inf if avg_inf > 0 else 0
    p50       = np.percentile(inf_times, 50)
    p95       = np.percentile(inf_times, 95)

    print(f"\n{SEP}")
    print(f"  HASIL UJI BLAZEFACE")
    print(SEP)
    print(f"  Total frame uji     : {total}")
    print(f"  Frame terdeteksi    : {detected_n}  ({det_rate:.1f}%)")
    print(f"  Inference rerata    : {avg_inf:.2f} ms")
    print(f"  Latency P50 / P95   : {p50:.2f} ms  /  {p95:.2f} ms")
    print(f"  Estimasi FPS proses : {fps_est:.2f} FPS")
    print(f"  Catatan: threshold confidence = {getattr(__import__('config'), 'DETECT_CONFIDENCE', 0.6)}")
    print(SEP)

    # ── CSV ──────────────────────────────────────────────────
    _save_csv("A_blazeface_results.csv",
              ["frame_idx", "detected", "best_conf", "inference_ms"],
              results)

    # ── Visualisasi: detection rate bar chart ────────────────
    CW, CH = 700, 420
    canvas = np.full((CH, CW, 3), 18, dtype=np.uint8)
    _header(canvas, "OUTPUT A — BlazeFace Detection Metrics",
            f"det_rate={det_rate:.1f}%  avg_inf={avg_inf:.1f}ms  fps≈{fps_est:.1f}",
            bar=(10, 30, 20))

    # ── Inference time bar chart ──────────────────────────────
    chart_x, chart_y = 20, 60
    chart_w, chart_h = CW - 40, 160
    _panel(canvas, chart_x, chart_y, chart_x+chart_w, chart_y+chart_h, CLR_GREEN)
    _txt(canvas, "Inference Time per Frame (ms)", (chart_x+5, chart_y+15), 0.42, CLR_GRAY)

    max_ms = max(inf_times) if inf_times else 1
    bar_w  = max(1, (chart_w - 10) // total)
    for i, ms in enumerate(inf_times):
        bh = int((ms / max_ms) * (chart_h - 30))
        bx = chart_x + 5 + i * bar_w
        by = chart_y + chart_h - bh
        color = CLR_GREEN if results[i][1] else CLR_RED
        cv2.rectangle(canvas, (bx, by), (bx + bar_w - 2, chart_y + chart_h), color, -1)

    # Avg line
    avg_y = chart_y + chart_h - int((avg_inf / max_ms) * (chart_h - 30))
    cv2.line(canvas, (chart_x, avg_y), (chart_x+chart_w, avg_y), CLR_YELLOW, 1)
    _txt(canvas, f"avg={avg_inf:.1f}ms", (chart_x + chart_w - 100, avg_y - 4), 0.38, CLR_YELLOW)

    # ── Stats panel ───────────────────────────────────────────
    py = chart_y + chart_h + 15
    _panel(canvas, 20, py, CW-20, py+160, CLR_CYAN)
    col1, col2 = 35, CW // 2 + 10
    _txt(canvas, f"Total Frame Uji        : {total}",          (col1, py+22),  0.48, CLR_WHITE)
    _txt(canvas, f"Frame Terdeteksi       : {detected_n}",     (col1, py+42),  0.48, CLR_GREEN)
    _txt(canvas, f"Detection Rate         : {det_rate:.2f} %", (col1, py+62),  0.52, CLR_CYAN, 2)
    _txt(canvas, f"Frame Tidak Terdeteksi : {total-detected_n}", (col1, py+84), 0.48, CLR_RED)
    _txt(canvas, f"Inference Rerata       : {avg_inf:.2f} ms", (col2, py+22),  0.48, CLR_WHITE)
    _txt(canvas, f"Latency P50            : {p50:.2f} ms",     (col2, py+42),  0.48, CLR_GRAY)
    _txt(canvas, f"Latency P95            : {p95:.2f} ms",     (col2, py+62),  0.48, CLR_GRAY)
    _txt(canvas, f"Estimasi FPS Proses    : {fps_est:.2f} FPS",(col2, py+84),  0.52, CLR_ORANGE, 2)
    _txt(canvas, f"Conf. Threshold        : {getattr(__import__('config'), 'DETECT_CONFIDENCE', 0.6)}",
         (col2, py+106), 0.44, CLR_GRAY)
    _txt(canvas, "Analisis: Green=detected, Red=missed. Yellow line=average latency.",
         (35, py+138), 0.4, CLR_GRAY)

    path = os.path.join(IMG_DIR, "A_blazeface_metrics.jpg")
    cv2.imwrite(path, canvas)
    print(f"  Gambar disimpan: {path}")
    return det_rate, avg_inf, fps_est


# ══════════════════════════════════════════════════════════════
# [B] UJI LIVENESS — EAR DEBUG + APCER/BPCER/ACER SCAFFOLDING
# ══════════════════════════════════════════════════════════════

def uji_liveness_debug(engine: FaceEngine, frames: list):
    """
    Debug liveness: tampilkan EAR per-frame, state machine, blink count,
    dan export CSV untuk kalkulasi APCER / BPCER / ACER manual.
    """
    print(f"\n{SEP2}")
    print("  [B] UJI LIVENESS — EAR Debug & Analisis")
    print(SEP2)

    if not MP_OK:
        print("  [!] MediaPipe tidak tersedia — uji liveness dilewati.")
        return

    # ── Deteksi wajah di frame awal ──────────────────────────
    print("  Mendeteksi wajah (BlazeFace) ...")
    face_box = None
    for frame in frames:
        box = engine.detect_largest(frame)
        if box is not None:
            face_box = box[:4]
            break

    if face_box is None:
        print("  [!] Tidak ada wajah terdeteksi — uji liveness dilewati.")
        return

    x1, y1, x2, y2 = [int(v) for v in face_box]
    print(f"  Wajah: ({x1},{y1})→({x2},{y2})")

    # ── Jalankan BlinkDetector frame-per-frame ────────────────
    import config as cfg
    detector = BlinkDetector()
    ear_rows  = []   # (frame_idx, ear, state, closed_f, open_f, blinks)
    ear_vals  = []

    print(f"\n  {'Frame':>5}  {'EAR':>7}  {'Signal':>8}  {'State':>7}  "
          f"{'ClosedF':>7}  {'OpenF':>6}  {'Blinks':>6}")
    print("  " + "─" * 60)

    for i, frame in enumerate(frames):
        crop = frame[max(0,y1):y2, max(0,x1):x2]
        if crop.size == 0:
            continue
        ear = detector.update(crop)
        if ear is None:
            continue

        ear_vals.append(ear)
        # signal derivation for display
        thresh_c = cfg.BLINK_EAR_THRESHOLD
        thresh_o = thresh_c + getattr(cfg, "BLINK_EAR_OPEN_GAP", 0.02)
        if ear < thresh_c:
            sig = "closing"
        elif ear > thresh_o:
            sig = "opening"
        else:
            sig = "neutral"

        row = (i, round(ear,4), sig, detector._state,
               detector._closed_frames, detector._open_frames, detector.blink_count)
        ear_rows.append(row)

        bar_fill = "▓" * min(int(ear / 0.4 * 20), 20)
        blink_marker = " ← BLINK!" if (i > 0 and detector.blink_count > (ear_rows[-2][6] if len(ear_rows) > 1 else 0)) else ""
        print(f"  {i+1:>5}  {ear:>7.4f}  {sig:>8}  {detector._state:>7}  "
              f"{detector._closed_frames:>7}  {detector._open_frames:>6}  "
              f"{detector.blink_count:>6}{blink_marker}")

    total_blinks = detector.blink_count
    avg_ear = np.mean(ear_vals) if ear_vals else 0.0
    min_ear = np.min(ear_vals) if ear_vals else 0.0
    max_ear = np.max(ear_vals) if ear_vals else 0.0

    print(f"\n{SEP}")
    print(f"  HASIL UJI LIVENESS (EAR Debug)")
    print(SEP)
    print(f"  Total frame valid     : {len(ear_vals)}")
    print(f"  Blink terdeteksi      : {total_blinks}")
    print(f"  EAR rerata            : {avg_ear:.4f}")
    print(f"  EAR min / maks        : {min_ear:.4f}  /  {max_ear:.4f}")
    print(f"  EAR threshold tutup   : {cfg.BLINK_EAR_THRESHOLD}")
    print(f"  EAR threshold buka    : {cfg.BLINK_EAR_THRESHOLD + getattr(cfg,'BLINK_EAR_OPEN_GAP',0.02):.2f}")
    print(f"  Min closed frames     : {cfg.LIVENESS_BLINK_MIN_CLOSED_FRAMES}")
    print(f"  Min open frames       : {getattr(cfg,'BLINK_MIN_OPEN_FRAMES',3)}")
    print(SEP)
    print("  APCER / BPCER / ACER:")
    print("    Isi kolom 'label' di CSV dengan:")
    print("    'live' = percobaan dengan orang asli")
    print("    'spoof'= percobaan dengan foto/video")
    print("    Lalu hitung:")
    print("    APCER = FP_spoof / total_spoof  (foto lolos liveness)")
    print("    BPCER = FN_live  / total_live   (orang asli ditolak liveness)")
    print("    ACER  = (APCER + BPCER) / 2")
    print(SEP)

    # ── CSV ──────────────────────────────────────────────────
    _save_csv("B_liveness_ear_debug.csv",
              ["frame_idx", "ear", "eye_signal", "state",
               "closed_frames", "open_frames", "blink_count"],
              ear_rows)

    # Template CSV untuk APCER/BPCER/ACER (diisi manual)
    template_rows = [
        ["session_id", "label", "required_blinks", "detected_blinks", "liveness_result",
         "score", "notes"],
        ["1", "live",  "1", "", "", "", "isi setelah uji dengan orang asli"],
        ["2", "spoof", "1", "", "", "", "isi setelah uji dengan foto"],
    ]
    apcer_path = os.path.join(CSV_DIR, "B_liveness_apcer_template.csv")
    with open(apcer_path, "w", newline="") as f:
        csv.writer(f).writerows(template_rows)
    print(f"  Template APCER/BPCER: {apcer_path}")

    # ── Visualisasi EAR Timeline ─────────────────────────────
    _save_liveness_img(ear_vals, ear_rows, total_blinks, avg_ear,
                       getattr(cfg, "BLINK_EAR_THRESHOLD", 0.21),
                       getattr(cfg, "BLINK_EAR_OPEN_GAP", 0.02))


def _save_liveness_img(ear_vals, ear_rows, total_blinks, avg_ear,
                       thresh_c, thresh_gap):
    thresh_o = thresh_c + thresh_gap
    CW, CH   = 900, 480
    canvas   = np.full((CH, CW, 3), 18, dtype=np.uint8)

    _header(canvas, "OUTPUT B — Liveness EAR Timeline Debug",
            f"blinks={total_blinks}  avg_ear={avg_ear:.4f}  "
            f"thresh_close={thresh_c:.2f}  thresh_open={thresh_o:.2f}",
            bar=(10, 20, 40))

    # ── EAR sparkline ────────────────────────────────────────
    cx, cy = 20, 60
    cw, ch = CW - 40, 240
    _panel(canvas, cx, cy, cx+cw, cy+ch, CLR_CYAN)
    _txt(canvas, "EAR per Frame", (cx+5, cy+15), 0.42, CLR_GRAY)

    n = len(ear_vals)
    if n > 1:
        for i in range(1, n):
            x_a = cx + int((i-1)/(n-1)*cw)
            x_b = cx + int(i    /(n-1)*cw)
            y_a = cy + ch - int(np.clip(ear_vals[i-1]/0.5,0,1)*ch)
            y_b = cy + ch - int(np.clip(ear_vals[i]  /0.5,0,1)*ch)
            # color by signal zone
            if ear_vals[i] < thresh_c:
                col = CLR_RED
            elif ear_vals[i] > thresh_o:
                col = CLR_GREEN
            else:
                col = CLR_YELLOW  # neutral/hysteresis zone
            cv2.line(canvas, (x_a, y_a), (x_b, y_b), col, 2)

    # threshold lines
    def _thresh_y(v): return cy + ch - int(np.clip(v/0.5,0,1)*ch)
    cv2.line(canvas, (cx, _thresh_y(thresh_c)), (cx+cw, _thresh_y(thresh_c)), CLR_RED,    1)
    cv2.line(canvas, (cx, _thresh_y(thresh_o)), (cx+cw, _thresh_y(thresh_o)), CLR_ORANGE, 1)
    cv2.line(canvas, (cx, _thresh_y(avg_ear)),  (cx+cw, _thresh_y(avg_ear)),  CLR_CYAN,   1)
    _txt(canvas, f"close={thresh_c:.2f}",  (cx+cw-110, _thresh_y(thresh_c)-4), 0.36, CLR_RED)
    _txt(canvas, f"open={thresh_o:.2f}",   (cx+cw-110, _thresh_y(thresh_o)-4), 0.36, CLR_ORANGE)
    _txt(canvas, f"avg={avg_ear:.4f}",     (cx+cw-110, _thresh_y(avg_ear)-4),  0.36, CLR_CYAN)

    # Mark blinks on timeline
    prev_blinks = 0
    for row in ear_rows:
        idx, ear, _, _, _, _, blinks = row
        if blinks > prev_blinks and n > 1:
            bx = cx + int(idx/(n-1)*cw)
            cv2.line(canvas, (bx, cy), (bx, cy+ch), CLR_GREEN, 1)
            _txt(canvas, f"B{blinks}", (bx+2, cy+20), 0.35, CLR_GREEN)
            prev_blinks = blinks

    # Legend
    leg_y = cy + ch + 10
    _txt(canvas, "─── Red: EAR < close_thresh (mata menutup)",  (cx, leg_y+14), 0.38, CLR_RED)
    _txt(canvas, "─── Yellow: neutral zone (hysteresis)",       (cx, leg_y+30), 0.38, CLR_YELLOW)
    _txt(canvas, "─── Green: EAR > open_thresh (mata terbuka)", (cx, leg_y+46), 0.38, CLR_GREEN)
    _txt(canvas, "│ Green vertical lines = blink events counted", (CW//2, leg_y+14), 0.38, CLR_GREEN)

    # ── Stats panel ───────────────────────────────────────────
    py = leg_y + 65
    _panel(canvas, cx, py, CW-20, py+110, CLR_CYAN)
    col1, col2 = cx+15, CW//2+10
    _txt(canvas, f"Total frame valid   : {n}",               (col1, py+22), 0.48, CLR_WHITE)
    _txt(canvas, f"Blink terdeteksi    : {total_blinks}",    (col1, py+42), 0.52, CLR_GREEN, 2)
    _txt(canvas, f"EAR rerata          : {avg_ear:.4f}",     (col1, py+62), 0.48, CLR_CYAN)
    _txt(canvas, f"EAR min / maks      : {min(ear_vals):.4f} / {max(ear_vals):.4f}" if ear_vals else "N/A",
         (col1, py+82), 0.44, CLR_GRAY)
    _txt(canvas, f"Threshold menutup   : {thresh_c:.2f}",    (col2, py+22), 0.48, CLR_WHITE)
    _txt(canvas, f"Threshold membuka   : {thresh_o:.2f}",    (col2, py+42), 0.48, CLR_WHITE)
    _txt(canvas, f"Hysteresis gap      : {thresh_gap:.2f}",  (col2, py+62), 0.48, CLR_ORANGE)
    _txt(canvas, "APCER/BPCER: isi CSV template (diisi manual)", (col2, py+86), 0.38, CLR_GRAY)

    path = os.path.join(IMG_DIR, "B_liveness_ear_debug.jpg")
    cv2.imwrite(path, canvas)
    print(f"  Gambar disimpan: {path}")


# ══════════════════════════════════════════════════════════════
# [C] UJI MOBILEFACENET — FAR / FRR / EER / ACCURACY
# ══════════════════════════════════════════════════════════════

def uji_mobilefacenet(engine: FaceEngine, frames: list):
    """
    Uji MobileFaceNet:
    - Tampilkan distribusi cosine similarity antar frame (genuine pairs)
    - Export CSV untuk kalkulasi FAR / FRR / EER / Accuracy manual
    - Gambar visualisasi embedding dan similarity heatmap
    """
    print(f"\n{SEP2}")
    print("  [C] UJI MOBILEFACENET — Embedding & Similarity")
    print(SEP2)

    # ── Kumpulkan embedding dari semua frame valid ────────────
    embs   = []
    boxes  = []
    failed = 0

    print("  Mengekstrak embedding ...")
    for i, frame in enumerate(frames):
        box = engine.detect_largest(frame)
        if box is None:
            failed += 1
            continue
        x1, y1, x2, y2, _ = box
        crop = frame[max(0,y1):y2, max(0,x1):x2]
        emb  = engine._embed_face(crop)
        if emb is not None:
            embs.append(emb)
            boxes.append(box)
            print(f"  frame {i+1:>2}: OK  dim={emb.shape[0]}  "
                  f"norm={np.linalg.norm(emb):.4f}")
        else:
            failed += 1
            print(f"  frame {i+1:>2}: FAILED (embedding error)")

    if len(embs) < 2:
        print("  [!] Tidak cukup embedding untuk analisis similarity.")
        return

    # ── Hitung pairwise cosine similarity (genuine pairs) ────
    sims = []
    sim_rows = []
    for i in range(len(embs)):
        for j in range(i+1, len(embs)):
            s = float(np.dot(embs[i], embs[j]))
            sims.append(s)
            sim_rows.append((i, j, round(s, 5)))

    avg_sim  = np.mean(sims)
    std_sim  = np.std(sims)
    min_sim  = np.min(sims)
    max_sim  = np.max(sims)

    import config as cfg
    threshold = cfg.FACE_MATCH_THRESH

    print(f"\n{SEP}")
    print(f"  HASIL UJI MOBILEFACENET")
    print(SEP)
    print(f"  Embedding berhasil    : {len(embs)} / {len(frames)}")
    print(f"  Embedding gagal       : {failed}")
    print(f"  Dimensi embedding     : {embs[0].shape[0]}")
    print(f"  Cosine sim rerata     : {avg_sim:.4f}")
    print(f"  Std deviasi           : {std_sim:.4f}")
    print(f"  Sim min / maks        : {min_sim:.4f}  /  {max_sim:.4f}")
    print(f"  Threshold sistem      : {threshold}")
    print(f"  Pasang ≥ threshold    : {sum(1 for s in sims if s >= threshold)} / {len(sims)}")
    print(SEP)
    print("  FAR / FRR / EER / Accuracy:")
    print("    Isi CSV template dengan hasil uji multi-pengguna:")
    print("    'genuine' = pasang wajah orang yang sama (harusnya MATCH)")
    print("    'impostor'= pasang wajah orang berbeda  (harusnya REJECT)")
    print("    FAR = FP_impostor / total_impostor")
    print("    FRR = FN_genuine  / total_genuine")
    print("    EER = titik FAR == FRR (cari dengan sweep threshold)")
    print("    Accuracy = (TP + TN) / total")
    print(SEP)

    # ── CSV ──────────────────────────────────────────────────
    _save_csv("C_mobilefacenet_genuine_pairs.csv",
              ["emb_i", "emb_j", "cosine_similarity"],
              sim_rows)

    # Template FAR/FRR (diisi manual)
    far_rows = [
        ["pair_id", "label", "emb_score", "threshold", "match_result",
         "expected_result", "correct", "notes"],
        ["1", "genuine",  "", str(threshold), "", "MATCH",  "", "pasang orang sama"],
        ["2", "impostor", "", str(threshold), "", "REJECT", "", "pasang orang beda"],
    ]
    far_path = os.path.join(CSV_DIR, "C_mobilefacenet_far_frr_template.csv")
    with open(far_path, "w", newline="") as f:
        csv.writer(f).writerows(far_rows)
    print(f"  Template FAR/FRR: {far_path}")

    # ── Visualisasi ───────────────────────────────────────────
    _save_facenet_img(embs, sims, sim_rows, avg_sim, std_sim, threshold, engine)


def _save_facenet_img(embs, sims, sim_rows, avg_sim, std_sim, threshold, engine):
    CW, CH = 900, 520
    canvas = np.full((CH, CW, 3), 18, dtype=np.uint8)
    _header(canvas, "OUTPUT C — MobileFaceNet Embedding & Similarity",
            f"n_embs={len(embs)}  avg_sim={avg_sim:.4f}  "
            f"std={std_sim:.4f}  thresh={threshold}",
            bar=(30, 10, 30))

    # ── Similarity histogram (top half) ──────────────────────
    hx, hy = 20, 60
    hw, hh  = CW - 40, 180
    _panel(canvas, hx, hy, hx+hw, hy+hh, CLR_ORANGE)
    _txt(canvas, "Genuine Pair Cosine Similarity Distribution", (hx+5, hy+15), 0.42, CLR_GRAY)

    if sims:
        bins   = 30
        lo, hi = max(0.0, min(sims)-0.05), min(1.0, max(sims)+0.05)
        ranges = np.linspace(lo, hi, bins+1)
        counts = np.zeros(bins, dtype=int)
        for s in sims:
            idx = min(int((s-lo)/(hi-lo)*bins), bins-1)
            counts[idx] += 1
        max_c = max(counts) if counts.max() > 0 else 1
        bw = (hw - 10) // bins
        for i, c in enumerate(counts):
            bh_px = int(c / max_c * (hh - 30))
            bx = hx + 5 + i * bw
            mid = (ranges[i] + ranges[i+1]) / 2
            col = CLR_GREEN if mid >= threshold else CLR_RED
            cv2.rectangle(canvas, (bx, hy+hh-bh_px), (bx+bw-1, hy+hh), col, -1)

        # threshold line
        tx = hx + 5 + int((threshold - lo) / (hi - lo) * (hw - 10))
        cv2.line(canvas, (tx, hy+20), (tx, hy+hh), CLR_YELLOW, 2)
        _txt(canvas, f"thresh={threshold}", (tx+3, hy+32), 0.38, CLR_YELLOW)

        # avg line
        ax = hx + 5 + int((avg_sim - lo) / (hi - lo) * (hw - 10))
        cv2.line(canvas, (ax, hy+20), (ax, hy+hh), CLR_CYAN, 1)
        _txt(canvas, f"avg={avg_sim:.3f}", (ax+3, hy+55), 0.36, CLR_CYAN)

    # ── Embedding per-dim preview (first embedding, 128 dims) ─
    py2  = hy + hh + 15
    ex, ey = 20, py2
    ew, eh  = CW // 2 - 30, 140
    _panel(canvas, ex, ey, ex+ew, ey+eh, CLR_ORANGE)
    _txt(canvas, "Embedding dim preview (first 128 of emb[0])", (ex+5, ey+15), 0.38, CLR_GRAY)
    emb0 = embs[0]
    n_show = min(128, emb0.shape[0])
    bw2 = max(1, (ew-10) // n_show)
    for i in range(n_show):
        v = float(emb0[i])
        bh_e = int(min(abs(v)*60, 55))
        bx = ex + 5 + i * bw2
        col = CLR_GREEN if v >= 0 else CLR_RED
        cv2.rectangle(canvas, (bx, ey+eh//2-bh_e), (bx+bw2-1, ey+eh//2), col, -1)

    # ── Stats panel ───────────────────────────────────────────
    sx, sy = CW//2 + 10, py2
    sw, sh = CW//2 - 30, 140
    _panel(canvas, sx, sy, sx+sw, sy+sh, CLR_ORANGE)
    _txt(canvas, f"Embedding dim      : {emb0.shape[0]}",       (sx+10, sy+22), 0.46, CLR_WHITE)
    _txt(canvas, f"Cosine sim rerata  : {avg_sim:.4f}",         (sx+10, sy+42), 0.46, CLR_CYAN)
    _txt(canvas, f"Std deviasi        : {std_sim:.4f}",         (sx+10, sy+62), 0.46, CLR_GRAY)
    _txt(canvas, f"Sim min / maks     : {min(sims):.4f}/{max(sims):.4f}", (sx+10, sy+80), 0.44, CLR_GRAY)
    _txt(canvas, f"Threshold sistem   : {threshold}",           (sx+10, sy+100), 0.46, CLR_YELLOW)
    _txt(canvas, f"Pasang ≥ thresh    : {sum(1 for s in sims if s>=threshold)}/{len(sims)}",
         (sx+10, sy+118), 0.46, CLR_GREEN)

    # ── Footer notes ──────────────────────────────────────────
    fy = sy + sh + 15
    _panel(canvas, 20, fy, CW-20, fy+60, CLR_GRAY)
    _txt(canvas, "FAR / FRR / EER / Accuracy → isi CSV template C_mobilefacenet_far_frr_template.csv",
         (30, fy+22), 0.42, CLR_YELLOW)
    _txt(canvas, "Genuine pairs = orang sama. Impostor pairs = orang berbeda.",
         (30, fy+44), 0.38, CLR_GRAY)

    path = os.path.join(IMG_DIR, "C_mobilefacenet_metrics.jpg")
    cv2.imwrite(path, canvas)
    print(f"  Gambar disimpan: {path}")


# ══════════════════════════════════════════════════════════════
# ORIGINAL VISUAL OUTPUTS (01–05) — tetap tersedia
# ══════════════════════════════════════════════════════════════

def save_01_landmarks(base_frame, face_landmarks_result):
    H, W = base_frame.shape[:2]
    canvas = base_frame.copy()
    lm_count = 0
    if face_landmarks_result and face_landmarks_result.multi_face_landmarks:
        lms = face_landmarks_result.multi_face_landmarks[0].landmark
        lm_count = len(lms)
        for i, lm in enumerate(lms):
            px, py = int(lm.x*W), int(lm.y*H)
            if i in _LEFT_EYE_IDX or i in _RIGHT_EYE_IDX:
                col, r = CLR_CYAN, 3
            elif i in (_L_MOUTH_IDX, _R_MOUTH_IDX):
                col, r = CLR_YELLOW, 3
            elif i == _NOSE_TIP_IDX:
                col, r = CLR_ORANGE, 5
            else:
                col, r = CLR_GREEN, 2
            cv2.circle(canvas, (px, py), r, col, -1)
    _header(canvas, "OUTPUT 01 — Face Landmark Dots",
            f"{lm_count} landmarks  |  Cyan=Eyes  Yellow=Mouth  Orange=Nose",
            bar=(20,20,40))
    path = os.path.join(IMG_DIR, "01_landmark_dots.jpg")
    cv2.imwrite(path, canvas)
    print(f"  01_landmark_dots.jpg ({lm_count} landmarks)")

def save_03_blazeface(base_frame, boxes):
    H, W = base_frame.shape[:2]
    canvas = base_frame.copy()
    for i, box in enumerate(boxes):
        if len(box) == 6:
            x1, y1, x2, y2, score, keypoints = box
        else:
            x1, y1, x2, y2, score = box
            keypoints = []
            
        cv2.rectangle(canvas, (x1,y1), (x2,y2), CLR_GREEN, 2)
        _txt(canvas, f"FACE #{i+1} conf={score:.3f}", (x1+4, y1-5), 0.5, CLR_CYAN)
        
        # Analyze Face Crop for Paper Testing (HVS vs Glossy vs Color)
        face_crop = base_frame[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
        if face_crop.size > 0:
            gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray_crop)
            contrast = np.std(gray_crop)
            hsv_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
            mean_saturation = np.mean(hsv_crop[:, :, 1])
            _txt(canvas, f"B:{mean_brightness:.1f} C:{contrast:.1f} S:{mean_saturation:.1f}", (x1+4, y2+15), 0.4, CLR_YELLOW)
        
        # Draw BlazeFace Simple Landmarks
        for kx, ky in keypoints:
            cv2.circle(canvas, (kx, ky), 3, CLR_ORANGE, -1)
            
    _header(canvas, "OUTPUT 03 — BlazeFace with Landmarks & Paper Metrics",
            f"{len(boxes)} face(s) | Orange=BlazeFace LMs | B=Brightness C=Contrast S=Saturation", bar=(10,30,20))
    path = os.path.join(IMG_DIR, "03_blazeface_detection.jpg")
    cv2.imwrite(path, canvas)
    print(f"  03_blazeface_detection.jpg ({len(boxes)} faces)")

def save_05_face_vector(base_frame, face_crop, engine):
    CW, CH = 900, 400
    canvas = np.full((CH, CW, 3), 18, dtype=np.uint8)
    emb = engine._embed_face(face_crop) if face_crop is not None else None
    if emb is not None:
        dim = emb.shape[0]
        _txt(canvas, f"dim={dim}  norm={np.linalg.norm(emb):.4f}  "
             f"mean={np.mean(emb):.4f}  std={np.std(emb):.4f}",
             (10, 80), 0.5, CLR_CYAN)
        bw = max(1, (CW-20)//min(dim,256))
        for i in range(min(dim,256)):
            v = float(emb[i])
            bh = int(min(abs(v)*80,80))
            bx = 10 + i*bw
            col = CLR_GREEN if v >= 0 else CLR_RED
            cv2.rectangle(canvas, (bx, 200-bh), (bx+bw-1, 200), col, -1)
    else:
        _txt(canvas, "No embedding", (200,200), 1.0, CLR_RED, 2)
    _header(canvas, "OUTPUT 05 — Face Vector (MobileFaceNet)",
            f"engine={engine.mode}", bar=(30,10,30))
    path = os.path.join(IMG_DIR, "05_face_vector.jpg")
    cv2.imwrite(path, canvas)
    print(f"  05_face_vector.jpg")


# ══════════════════════════════════════════════════════════════
# MENU INTERAKTIF
# ══════════════════════════════════════════════════════════════

def interactive_menu(engine, frames, base_frame, boxes, face_crop, face_landmarks_result):
    while True:
        print(f"\n{SEP2}")
        print("  MENU UJI PROTOTIPE")
        print(SEP2)
        print("  [1] Uji BlazeFace          (deteksi, FPS, latency, CSV)")
        print("  [2] Uji Liveness EAR Debug  (EAR timeline, blink count, CSV)")
        print("  [3] Uji MobileFaceNet       (similarity, embedding CSV)")
        print("  [4] Simpan gambar visual    (01 landmark, 03 blazeface, 05 vector)")
        print("  [5] Jalankan semua uji")
        print("  [0] Keluar")
        print(SEP)
        pilih = input("  Pilihan: ").strip()

        if   pilih == "1": uji_blazeface(engine, frames)
        elif pilih == "2": uji_liveness_debug(engine, frames)
        elif pilih == "3": uji_mobilefacenet(engine, frames)
        elif pilih == "4":
            print("\n  Menyimpan gambar visual ...")
            save_01_landmarks(base_frame, face_landmarks_result)
            save_03_blazeface(base_frame, boxes)
            save_05_face_vector(base_frame, face_crop, engine)
            print(f"  Gambar tersimpan di: {IMG_DIR}/")
        elif pilih == "5":
            uji_blazeface(engine, frames)
            uji_liveness_debug(engine, frames)
            uji_mobilefacenet(engine, frames)
            save_01_landmarks(base_frame, face_landmarks_result)
            save_03_blazeface(base_frame, boxes)
            save_05_face_vector(base_frame, face_crop, engine)
        elif pilih == "0":
            print("  Keluar.")
            break
        else:
            print("  [!] Pilihan tidak valid.")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ACC-Control Prototype Test Suite")
    parser.add_argument("--blazeface", action="store_true")
    parser.add_argument("--liveness",  action="store_true")
    parser.add_argument("--facenet",   action="store_true")
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--frames",    type=int, default=20, help="Jumlah frame yang diambil")
    args = parser.parse_args()

    print(f"\n{SEP2}")
    print("  ACC-CONTROL PROTOTYPE TEST SUITE")
    print(SEP2)

    # ── Init FaceEngine ───────────────────────────────────────
    print("\n[INIT] Loading FaceEngine ...")
    engine = FaceEngine()
    if not engine.load():
        print("[!] FaceEngine gagal dimuat — aborting.")
        sys.exit(1)
    print(f"       Mode: {engine.mode}")

    # ── Capture frames ────────────────────────────────────────
    n_frames = args.frames
    print(f"\n[CAPTURE] Mengambil {n_frames} frame dari kamera ...")
    frames = capture_frames(n=n_frames)
    base_frame = frames[-1].copy()
    print(f"          {len(frames)} frame ({base_frame.shape[1]}x{base_frame.shape[0]})")

    # ── Shared detection for visual outputs ───────────────────
    boxes = engine.detect_with_landmarks(base_frame)
    best_box  = max(boxes, key=lambda b:(b[2]-b[0])*(b[3]-b[1])) if boxes else None
    face_crop = None
    if best_box:
        x1,y1,x2,y2 = best_box[0:4]
        face_crop = base_frame[max(0,y1):y2, max(0,x1):x2]

    face_landmarks_result = None
    if MP_OK and mp_face_mesh:
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                    refine_landmarks=False,
                                    min_detection_confidence=0.5) as mesh:
            rgb = cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB)
            face_landmarks_result = mesh.process(rgb)

    # ── CLI mode / interactive ────────────────────────────────
    run_all = args.all
    any_flag = args.blazeface or args.liveness or args.facenet or run_all

    if not any_flag:
        interactive_menu(engine, frames, base_frame, boxes, face_crop, face_landmarks_result)
        return

    if run_all or args.blazeface:
        uji_blazeface(engine, frames)
    if run_all or args.liveness:
        uji_liveness_debug(engine, frames)
    if run_all or args.facenet:
        uji_mobilefacenet(engine, frames)

    if run_all:
        print("\n[SAVE] Menyimpan gambar visual ...")
        save_01_landmarks(base_frame, face_landmarks_result)
        save_03_blazeface(base_frame, boxes)
        save_05_face_vector(base_frame, face_crop, engine)

    print(f"\n{SEP2}")
    print(f"  SELESAI — Output tersimpan di: ./{OUT_DIR}/")
    print(f"  CSV  : ./{CSV_DIR}/")
    print(f"  Gambar: ./{IMG_DIR}/")
    print(SEP2 + "\n")


if __name__ == "__main__":
    main()

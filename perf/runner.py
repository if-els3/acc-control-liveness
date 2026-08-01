"""
=============================================================
perf/runner.py  --  Unified Performance & Prototype Test Runner
=============================================================
Dipanggil oleh main.py dengan flag --perf.

Menggabungkan fungsi dari:
  * profile_performance.py  (CPU/RAM/FPS benchmark)
  * run_prototype_test.py   (BlazeFace / Liveness EAR / MobileFaceNet)

Output yang dihasilkan (disimpan di perf_output/<timestamp>/):
  csv/
    A_blazeface_results.csv
    B_liveness_ear_debug.csv
    B_liveness_apcer_template.csv
    C_mobilefacenet_genuine_pairs.csv
    C_mobilefacenet_far_frr_template.csv
    D_pipeline_profiling.csv
  img/
    A_blazeface_metrics.jpg
    B_liveness_ear_debug.jpg
    C_mobilefacenet_metrics.jpg
    01_landmark_dots.jpg
    03_blazeface_detection.jpg
    05_face_vector.jpg
  profiling_report.md

Penggunaan:
    python main.py --perf [--perf-frames N] [--perf-duration S]
=============================================================
"""

import os
import sys
import time
import csv
import logging
import datetime
import numpy as np

log = logging.getLogger(__name__)

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

FONT       = cv2.FONT_HERSHEY_SIMPLEX if CV2_OK else None
CLR_GREEN  = (80,  220,  80)
CLR_CYAN   = (220, 220,   0)
CLR_YELLOW = (0,   220, 220)
CLR_RED    = (60,   60, 255)
CLR_WHITE  = (240, 240, 240)
CLR_GRAY   = (140, 140, 140)
CLR_ORANGE = (0,   165, 255)
SEP        = "-" * 60
SEP2       = "=" * 60


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


def _save_csv(csv_dir, filename, header, rows):
    path = os.path.join(csv_dir, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  [CSV] {path}")
    return path


def _capture_frames(n=20, cam_index=0, warmup=2.0):
    """Ambil n frame dari kamera."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("  [!] Kamera tidak ditemukan -- menggunakan dummy frame.")
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
        print("  [!] Gagal baca frame -- menggunakan dummy.")
        dummy = np.full((240, 320, 3), 40, dtype=np.uint8)
        return [dummy] * n
    return frames


# ============================================================
# [A] UJI BLAZEFACE
# ============================================================

def uji_blazeface(engine, frames, img_dir, csv_dir):
    import config
    print(f"\n{SEP2}")
    print("  [A] UJI BLAZEFACE -- Akurasi Deteksi & Performa")
    print(SEP2)

    results = []
    total = len(frames)
    detected_n = 0
    inf_times = []

    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        boxes = engine.detect(frame)
        inf_ms = (time.perf_counter() - t0) * 1000

        detected  = len(boxes) > 0
        best_conf = max(b[4] for b in boxes) if detected else 0.0

        results.append((i, int(detected), round(best_conf, 4), round(inf_ms, 2)))
        inf_times.append(inf_ms)
        if detected:
            detected_n += 1

        bar = "#" * int(best_conf * 20) if detected else "-" * 20
        print(f"  frame {i+1:>2}/{total}  det={'YES' if detected else 'NO ':3}  "
              f"conf={best_conf:.3f}  inf={inf_ms:.1f}ms  [{bar}]")

    det_rate = detected_n / total * 100
    avg_inf  = np.mean(inf_times)
    fps_est  = 1000 / avg_inf if avg_inf > 0 else 0
    p50      = np.percentile(inf_times, 50)
    p95      = np.percentile(inf_times, 95)

    print(f"\n{SEP}")
    print(f"  HASIL UJI BLAZEFACE")
    print(SEP)
    print(f"  Total frame uji     : {total}")
    print(f"  Frame terdeteksi    : {detected_n}  ({det_rate:.1f}%)")
    print(f"  Inference rerata    : {avg_inf:.2f} ms")
    print(f"  Latency P50 / P95   : {p50:.2f} ms  /  {p95:.2f} ms")
    print(f"  Estimasi FPS proses : {fps_est:.2f} FPS")
    print(f"  Conf. threshold     : {getattr(config, 'DETECT_CONFIDENCE', 0.6)}")
    print(SEP)

    _save_csv(csv_dir, "A_blazeface_results.csv",
              ["frame_idx", "detected", "best_conf", "inference_ms"], results)

    if CV2_OK:
        CW, CH = 700, 420
        canvas = np.full((CH, CW, 3), 18, dtype=np.uint8)
        _header(canvas, "OUTPUT A -- BlazeFace Detection Metrics",
                f"det_rate={det_rate:.1f}%  avg_inf={avg_inf:.1f}ms  fps={fps_est:.1f}",
                bar=(10, 30, 20))

        cx, cy = 20, 60
        cw, ch = CW - 40, 160
        _panel(canvas, cx, cy, cx+cw, cy+ch, CLR_GREEN)
        _txt(canvas, "Inference Time per Frame (ms)", (cx+5, cy+15), 0.42, CLR_GRAY)
        max_ms = max(inf_times) if inf_times else 1
        bw = max(1, (cw - 10) // total)
        for i, ms in enumerate(inf_times):
            bh = int((ms / max_ms) * (ch - 30))
            bx = cx + 5 + i * bw
            color = CLR_GREEN if results[i][1] else CLR_RED
            cv2.rectangle(canvas, (bx, cy+ch-bh), (bx+bw-2, cy+ch), color, -1)
        avg_y = cy + ch - int((avg_inf / max_ms) * (ch - 30))
        cv2.line(canvas, (cx, avg_y), (cx+cw, avg_y), CLR_YELLOW, 1)
        _txt(canvas, f"avg={avg_inf:.1f}ms", (cx+cw-100, avg_y-4), 0.38, CLR_YELLOW)

        py = cy + ch + 15
        _panel(canvas, 20, py, CW-20, py+160, CLR_CYAN)
        c1, c2 = 35, CW//2+10
        _txt(canvas, f"Total Frame Uji        : {total}",           (c1, py+22), 0.48, CLR_WHITE)
        _txt(canvas, f"Frame Terdeteksi       : {detected_n}",      (c1, py+42), 0.48, CLR_GREEN)
        _txt(canvas, f"Detection Rate         : {det_rate:.2f} %",  (c1, py+62), 0.52, CLR_CYAN, 2)
        _txt(canvas, f"Frame Tidak Terdeteksi : {total-detected_n}",(c1, py+84), 0.48, CLR_RED)
        _txt(canvas, f"Inference Rerata       : {avg_inf:.2f} ms",  (c2, py+22), 0.48, CLR_WHITE)
        _txt(canvas, f"Latency P50            : {p50:.2f} ms",      (c2, py+42), 0.48, CLR_GRAY)
        _txt(canvas, f"Latency P95            : {p95:.2f} ms",      (c2, py+62), 0.48, CLR_GRAY)
        _txt(canvas, f"Estimasi FPS Proses    : {fps_est:.2f} FPS", (c2, py+84), 0.52, CLR_ORANGE, 2)

        path = os.path.join(img_dir, "A_blazeface_metrics.jpg")
        cv2.imwrite(path, canvas)
        print(f"  [IMG] {path}")

    return {"det_rate": det_rate, "avg_inf_ms": avg_inf, "fps_est": fps_est,
            "p50_ms": p50, "p95_ms": p95, "detected_n": detected_n, "total": total}


# ============================================================
# [B] UJI LIVENESS -- EAR DEBUG
# ============================================================

def uji_liveness_debug(engine, frames, img_dir, csv_dir):
    print(f"\n{SEP2}")
    print("  [B] UJI LIVENESS -- EAR Debug & Analisis")
    print(SEP2)

    try:
        from core.liveness import BlinkDetector, MP_OK
    except ImportError:
        print("  [!] core.liveness tidak tersedia -- uji dilewati.")
        return {}

    if not MP_OK:
        print("  [!] MediaPipe tidak tersedia -- uji liveness dilewati.")
        return {}

    import config as cfg

    print("  Mendeteksi wajah (BlazeFace) ...")
    face_box = None
    for frame in frames:
        box = engine.detect_largest(frame)
        if box is not None:
            face_box = box[:4]
            break

    if face_box is None:
        print("  [!] Tidak ada wajah terdeteksi -- uji liveness dilewati.")
        return {}

    x1, y1, x2, y2 = [int(v) for v in face_box]
    print(f"  Wajah: ({x1},{y1})->({x2},{y2})")

    detector = BlinkDetector()
    ear_rows = []
    ear_vals = []

    print(f"\n  {'Frame':>5}  {'EAR':>7}  {'Signal':>8}  {'State':>7}  "
          f"{'ClosedF':>7}  {'OpenF':>6}  {'Blinks':>6}")
    print("  " + "-" * 60)

    for i, frame in enumerate(frames):
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        ear = detector.update(crop)
        if ear is None:
            continue

        ear_vals.append(ear)
        thresh_c = cfg.BLINK_EAR_THRESHOLD
        thresh_o = thresh_c + getattr(cfg, "BLINK_EAR_OPEN_GAP", 0.02)
        sig = "closing" if ear < thresh_c else ("opening" if ear > thresh_o else "neutral")

        row = (i, round(ear, 4), sig, detector._state,
               detector._closed_frames, detector._open_frames, detector.blink_count)
        ear_rows.append(row)

        blink_marker = " <- BLINK!" if (i > 0 and detector.blink_count > (ear_rows[-2][6] if len(ear_rows) > 1 else 0)) else ""
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
    print("  APCER / BPCER / ACER scaffolding:")
    print("    live  = percobaan dengan orang asli")
    print("    spoof = percobaan dengan foto/video")
    print("    APCER = FP_spoof / total_spoof")
    print("    BPCER = FN_live  / total_live")
    print("    ACER  = (APCER + BPCER) / 2")
    print(SEP)

    _save_csv(csv_dir, "B_liveness_ear_debug.csv",
              ["frame_idx", "ear", "eye_signal", "state",
               "closed_frames", "open_frames", "blink_count"], ear_rows)

    template_rows = [
        ["session_id", "label", "required_blinks", "detected_blinks",
         "liveness_result", "score", "notes"],
        ["1", "live",  "1", "", "", "", "isi setelah uji dengan orang asli"],
        ["2", "spoof", "1", "", "", "", "isi setelah uji dengan foto"],
    ]
    apcer_path = os.path.join(csv_dir, "B_liveness_apcer_template.csv")
    with open(apcer_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(template_rows)
    print(f"  [CSV] {apcer_path}")

    if CV2_OK and ear_vals:
        thresh_c   = getattr(cfg, "BLINK_EAR_THRESHOLD", 0.21)
        thresh_gap = getattr(cfg, "BLINK_EAR_OPEN_GAP", 0.02)
        thresh_o   = thresh_c + thresh_gap
        CW, CH = 900, 480
        canvas = np.full((CH, CW, 3), 18, dtype=np.uint8)
        _header(canvas, "OUTPUT B -- Liveness EAR Timeline Debug",
                f"blinks={total_blinks}  avg_ear={avg_ear:.4f}  close={thresh_c:.2f}  open={thresh_o:.2f}",
                bar=(10, 20, 40))

        cx, cy = 20, 60
        cw, ch = CW - 40, 240
        _panel(canvas, cx, cy, cx+cw, cy+ch, CLR_CYAN)
        _txt(canvas, "EAR per Frame", (cx+5, cy+15), 0.42, CLR_GRAY)

        n = len(ear_vals)
        if n > 1:
            for i in range(1, n):
                xa = cx + int((i-1)/(n-1)*cw)
                xb = cx + int(i    /(n-1)*cw)
                ya = cy + ch - int(np.clip(ear_vals[i-1]/0.5, 0, 1)*ch)
                yb = cy + ch - int(np.clip(ear_vals[i]  /0.5, 0, 1)*ch)
                col = CLR_RED if ear_vals[i] < thresh_c else (CLR_GREEN if ear_vals[i] > thresh_o else CLR_YELLOW)
                cv2.line(canvas, (xa, ya), (xb, yb), col, 2)

        def _ty(v): return cy + ch - int(np.clip(v/0.5, 0, 1)*ch)
        cv2.line(canvas, (cx, _ty(thresh_c)), (cx+cw, _ty(thresh_c)), CLR_RED,    1)
        cv2.line(canvas, (cx, _ty(thresh_o)), (cx+cw, _ty(thresh_o)), CLR_ORANGE, 1)
        cv2.line(canvas, (cx, _ty(avg_ear)),  (cx+cw, _ty(avg_ear)),  CLR_CYAN,   1)
        _txt(canvas, f"close={thresh_c:.2f}", (cx+cw-110, _ty(thresh_c)-4), 0.36, CLR_RED)
        _txt(canvas, f"open={thresh_o:.2f}",  (cx+cw-110, _ty(thresh_o)-4), 0.36, CLR_ORANGE)
        _txt(canvas, f"avg={avg_ear:.4f}",    (cx+cw-110, _ty(avg_ear)-4),  0.36, CLR_CYAN)

        prev_blinks = 0
        for row in ear_rows:
            idx, _, _, _, _, _, blinks = row
            if blinks > prev_blinks and n > 1:
                bx = cx + int(idx/(n-1)*cw)
                cv2.line(canvas, (bx, cy), (bx, cy+ch), CLR_GREEN, 1)
                _txt(canvas, f"B{blinks}", (bx+2, cy+20), 0.35, CLR_GREEN)
                prev_blinks = blinks

        leg_y = cy + ch + 10
        _txt(canvas, "Red: EAR < close_thresh (mata menutup)",   (cx, leg_y+14), 0.38, CLR_RED)
        _txt(canvas, "Yellow: neutral zone (hysteresis)",        (cx, leg_y+30), 0.38, CLR_YELLOW)
        _txt(canvas, "Green: EAR > open_thresh (mata terbuka)",  (cx, leg_y+46), 0.38, CLR_GREEN)
        _txt(canvas, "Green vertical = blink events counted",    (CW//2, leg_y+14), 0.38, CLR_GREEN)

        py = leg_y + 65
        _panel(canvas, cx, py, CW-20, py+110, CLR_CYAN)
        c1, c2 = cx+15, CW//2+10
        _txt(canvas, f"Total frame valid   : {n}",                   (c1, py+22), 0.48, CLR_WHITE)
        _txt(canvas, f"Blink terdeteksi    : {total_blinks}",        (c1, py+42), 0.52, CLR_GREEN, 2)
        _txt(canvas, f"EAR rerata          : {avg_ear:.4f}",         (c1, py+62), 0.48, CLR_CYAN)
        _txt(canvas, f"EAR min / maks      : {min_ear:.4f}/{max_ear:.4f}", (c1, py+82), 0.44, CLR_GRAY)
        _txt(canvas, f"Threshold menutup   : {thresh_c:.2f}",        (c2, py+22), 0.48, CLR_WHITE)
        _txt(canvas, f"Threshold membuka   : {thresh_o:.2f}",        (c2, py+42), 0.48, CLR_WHITE)
        _txt(canvas, f"Hysteresis gap      : {thresh_gap:.2f}",      (c2, py+62), 0.48, CLR_ORANGE)
        _txt(canvas, "APCER/BPCER: isi CSV template (diisi manual)", (c2, py+86), 0.38, CLR_GRAY)

        path = os.path.join(img_dir, "B_liveness_ear_debug.jpg")
        cv2.imwrite(path, canvas)
        print(f"  [IMG] {path}")

    return {"blinks": total_blinks, "avg_ear": avg_ear, "valid_frames": len(ear_vals)}


# ============================================================
# [C] UJI MOBILEFACENET
# ============================================================

def uji_mobilefacenet(engine, frames, img_dir, csv_dir):
    import config as cfg
    print(f"\n{SEP2}")
    print("  [C] UJI MOBILEFACENET -- Embedding & Similarity")
    print(SEP2)

    embs   = []
    failed = 0

    print("  Mengekstrak embedding ...")
    for i, frame in enumerate(frames):
        box = engine.detect_largest(frame)
        if box is None:
            failed += 1
            continue
        x1, y1, x2, y2, _ = box
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        emb  = engine._embed_face(crop)
        if emb is not None:
            embs.append(emb)
            print(f"  frame {i+1:>2}: OK  dim={emb.shape[0]}  norm={np.linalg.norm(emb):.4f}")
        else:
            failed += 1
            print(f"  frame {i+1:>2}: FAILED")

    if len(embs) < 2:
        print("  [!] Tidak cukup embedding untuk analisis similarity.")
        return {}

    sims = []
    sim_rows = []
    for i in range(len(embs)):
        for j in range(i+1, len(embs)):
            s = float(np.dot(embs[i], embs[j]))
            sims.append(s)
            sim_rows.append((i, j, round(s, 5)))

    avg_sim   = np.mean(sims)
    std_sim   = np.std(sims)
    min_sim   = np.min(sims)
    max_sim   = np.max(sims)
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
    print(f"  Pasang >= threshold   : {sum(1 for s in sims if s >= threshold)} / {len(sims)}")
    print(SEP)

    _save_csv(csv_dir, "C_mobilefacenet_genuine_pairs.csv",
              ["emb_i", "emb_j", "cosine_similarity"], sim_rows)

    far_rows = [
        ["pair_id", "label", "emb_score", "threshold", "match_result",
         "expected_result", "correct", "notes"],
        ["1", "genuine",  "", str(threshold), "", "MATCH",  "", "pasang orang sama"],
        ["2", "impostor", "", str(threshold), "", "REJECT", "", "pasang orang beda"],
    ]
    far_path = os.path.join(csv_dir, "C_mobilefacenet_far_frr_template.csv")
    with open(far_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(far_rows)
    print(f"  [CSV] {far_path}")

    if CV2_OK and sims:
        CW, CH = 900, 520
        canvas = np.full((CH, CW, 3), 18, dtype=np.uint8)
        _header(canvas, "OUTPUT C -- MobileFaceNet Embedding & Similarity",
                f"n_embs={len(embs)}  avg_sim={avg_sim:.4f}  std={std_sim:.4f}  thresh={threshold}",
                bar=(30, 10, 30))

        hx, hy = 20, 60
        hw, hh = CW - 40, 180
        _panel(canvas, hx, hy, hx+hw, hy+hh, CLR_ORANGE)
        _txt(canvas, "Genuine Pair Cosine Similarity Distribution", (hx+5, hy+15), 0.42, CLR_GRAY)

        lo, hi = max(0.0, min(sims)-0.05), min(1.0, max(sims)+0.05)
        if hi > lo:
            bins   = 30
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
            tx = hx + 5 + int((threshold-lo)/(hi-lo)*(hw-10))
            cv2.line(canvas, (tx, hy+20), (tx, hy+hh), CLR_YELLOW, 2)
            _txt(canvas, f"thresh={threshold}", (tx+3, hy+32), 0.38, CLR_YELLOW)
            ax = hx + 5 + int((avg_sim-lo)/(hi-lo)*(hw-10))
            cv2.line(canvas, (ax, hy+20), (ax, hy+hh), CLR_CYAN, 1)
            _txt(canvas, f"avg={avg_sim:.3f}", (ax+3, hy+55), 0.36, CLR_CYAN)

        py2 = hy + hh + 15
        ex, ey = 20, py2
        ew, eh = CW//2-30, 140
        _panel(canvas, ex, ey, ex+ew, ey+eh, CLR_ORANGE)
        _txt(canvas, "Embedding dim preview (first 128 of emb[0])", (ex+5, ey+15), 0.38, CLR_GRAY)
        emb0   = embs[0]
        n_show = min(128, emb0.shape[0])
        bw2    = max(1, (ew-10) // n_show)
        for i in range(n_show):
            v = float(emb0[i])
            bh_e = int(min(abs(v)*60, 55))
            bx   = ex + 5 + i * bw2
            col  = CLR_GREEN if v >= 0 else CLR_RED
            cv2.rectangle(canvas, (bx, ey+eh//2-bh_e), (bx+bw2-1, ey+eh//2), col, -1)

        sx, sy = CW//2+10, py2
        sw, sh = CW//2-30, 140
        _panel(canvas, sx, sy, sx+sw, sy+sh, CLR_ORANGE)
        _txt(canvas, f"Embedding dim      : {emb0.shape[0]}",        (sx+10, sy+22), 0.46, CLR_WHITE)
        _txt(canvas, f"Cosine sim rerata  : {avg_sim:.4f}",          (sx+10, sy+42), 0.46, CLR_CYAN)
        _txt(canvas, f"Std deviasi        : {std_sim:.4f}",          (sx+10, sy+62), 0.46, CLR_GRAY)
        _txt(canvas, f"Sim min/maks       : {min_sim:.4f}/{max_sim:.4f}", (sx+10, sy+80), 0.44, CLR_GRAY)
        _txt(canvas, f"Threshold sistem   : {threshold}",            (sx+10, sy+100), 0.46, CLR_YELLOW)
        _txt(canvas, f"Pasang >= thresh   : {sum(1 for s in sims if s>=threshold)}/{len(sims)}",
             (sx+10, sy+118), 0.46, CLR_GREEN)

        fy = sy + sh + 15
        _panel(canvas, 20, fy, CW-20, fy+60, CLR_GRAY)
        _txt(canvas, "FAR/FRR/EER/Accuracy -> isi CSV template C_mobilefacenet_far_frr_template.csv",
             (30, fy+22), 0.42, CLR_YELLOW)
        _txt(canvas, "Genuine pairs = orang sama. Impostor pairs = orang berbeda.",
             (30, fy+44), 0.38, CLR_GRAY)

        path = os.path.join(img_dir, "C_mobilefacenet_metrics.jpg")
        cv2.imwrite(path, canvas)
        print(f"  [IMG] {path}")

    return {"avg_sim": avg_sim, "std_sim": std_sim, "threshold": threshold,
            "n_embs": len(embs), "failed": failed}


# ============================================================
# [D] PIPELINE PROFILING
# ============================================================

def uji_pipeline_profiling(engine, duration_seconds=10, max_frames=200,
                            cam_index=0, csv_dir=".", out_dir="."):
    print(f"\n{SEP2}")
    print("  [D] PIPELINE PROFILING -- CPU, RAM & FPS")
    print(SEP2)

    if not PSUTIL_OK:
        print("  [!] Modul 'psutil' tidak ada. pip install psutil untuk CPU/RAM.")

    from core.liveness import LivenessDetector
    from core.camera_stream import CameraStream

    liveness_detector = LivenessDetector()
    blink_detector    = liveness_detector.create_blink_detector()
    blink_detector.reset()

    cam = CameraStream(cam_index=cam_index)
    if not cam.start():
        print("  [!] Gagal membuka kamera!")
        return {}

    if PSUTIL_OK:
        process = psutil.Process(os.getpid())
        process.cpu_percent()
        psutil.cpu_percent()
        time.sleep(0.5)

    print(f"  >> Benchmark ... ({duration_seconds}s atau maks {max_frames} frame)")
    print("  " + "-" * 56)

    times_detect  = []
    times_live    = []
    times_embed   = []
    times_total   = []
    cpu_proc_list = []
    cpu_sys_list  = []
    ram_proc_list = []
    profile_rows  = []
    frame_count   = 0
    face_count    = 0

    t_start = time.perf_counter()
    try:
        while (time.perf_counter() - t_start < duration_seconds) and (frame_count < max_frames):
            t_loop = time.perf_counter()
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1

            t0 = time.perf_counter()
            box = engine.detect_largest(frame)
            det_ms = (time.perf_counter() - t0) * 1000
            times_detect.append(det_ms)

            live_ms = 0.0
            emb_ms  = 0.0
            if box is not None:
                face_count += 1
                x1, y1, x2, y2, _ = box
                crop = frame[max(0, y1):y2, max(0, x1):x2]

                t0 = time.perf_counter()
                if crop.size > 0:
                    blink_detector.update(crop)
                live_ms = (time.perf_counter() - t0) * 1000
                times_live.append(live_ms)

                t0 = time.perf_counter()
                if crop.size > 0:
                    engine._embed_face(crop)
                emb_ms = (time.perf_counter() - t0) * 1000
                times_embed.append(emb_ms)
            else:
                times_live.append(0.0)
                times_embed.append(0.0)

            total_ms = (time.perf_counter() - t_loop) * 1000
            times_total.append(total_ms)

            cpu_p = 0.0
            ram_p = 0.0
            cpu_s = 0.0
            if PSUTIL_OK and frame_count % 5 == 0:
                try:
                    cpu_p = process.cpu_percent() / psutil.cpu_count()
                    ram_p = process.memory_info().rss / (1024 * 1024)
                    cpu_s = psutil.cpu_percent()
                    cpu_proc_list.append(cpu_p)
                    ram_proc_list.append(ram_p)
                    cpu_sys_list.append(cpu_s)
                except Exception:
                    pass

            profile_rows.append((frame_count, round(det_ms, 2), round(live_ms, 2),
                                  round(emb_ms, 2), round(total_ms, 2),
                                  round(cpu_p, 2), round(ram_p, 2)))
            time.sleep(0.02)
    finally:
        t_elapsed = time.perf_counter() - t_start
        cam.stop(force=True)

    avg_fps   = frame_count / t_elapsed if t_elapsed > 0 else 0
    avg_det   = np.mean(times_detect) if times_detect else 0
    avg_live  = np.mean([t for t in times_live  if t > 0]) or 0
    avg_emb   = np.mean([t for t in times_embed if t > 0]) or 0
    avg_total = np.mean(times_total)  if times_total  else 0
    fps_det   = 1000 / avg_det if avg_det > 0 else 0
    cpu_p_avg = np.mean(cpu_proc_list) if cpu_proc_list else 0
    cpu_s_avg = np.mean(cpu_sys_list)  if cpu_sys_list  else 0
    ram_p_avg = np.mean(ram_proc_list) if ram_proc_list else 0

    print(f"  OK - Benchmark selesai!")
    print(SEP)
    print(f"  Total Waktu           : {t_elapsed:.2f} s")
    print(f"  Total Frame           : {frame_count}")
    print(f"  Wajah Terdeteksi      : {face_count}/{frame_count}")
    print(f"  FPS Pipeline          : {avg_fps:.2f}")
    print(f"  FPS Deteksi Wajah     : {fps_det:.2f}")
    print(f"  Latency BlazeFace     : {avg_det:.2f} ms")
    print(f"  Latency Liveness EAR  : {avg_live:.2f} ms")
    print(f"  Latency Face Embedding: {avg_emb:.2f} ms")
    print(f"  Latency Total Loop    : {avg_total:.2f} ms")
    if PSUTIL_OK:
        print(f"  CPU (proses)          : {cpu_p_avg:.2f}%")
        print(f"  CPU (sistem)          : {cpu_s_avg:.2f}%")
        print(f"  RAM (proses)          : {ram_p_avg:.2f} MB")
    print(SEP)

    _save_csv(csv_dir, "D_pipeline_profiling.csv",
              ["frame", "detect_ms", "liveness_ms", "embed_ms", "total_ms",
               "cpu_proc_pct", "ram_proc_mb"],
              profile_rows)

    return {
        "fps_pipeline": avg_fps, "fps_detect": fps_det,
        "avg_det_ms": avg_det, "avg_live_ms": avg_live,
        "avg_emb_ms": avg_emb, "avg_total_ms": avg_total,
        "cpu_proc_pct": cpu_p_avg, "cpu_sys_pct": cpu_s_avg,
        "ram_proc_mb": ram_p_avg, "elapsed_s": t_elapsed,
        "frames": frame_count, "faces": face_count,
    }


# ============================================================
# VISUAL OUTPUTS (landmark, blazeface, face vector)
# ============================================================

def save_visual_outputs(engine, base_frame, img_dir):
    if not CV2_OK:
        return

    try:
        from core.liveness import (
            _LEFT_EYE_IDX, _RIGHT_EYE_IDX,
            _NOSE_TIP_IDX, _L_MOUTH_IDX, _R_MOUTH_IDX,
        )
        import mediapipe as mp
        mp_fm = mp.solutions.face_mesh
        with mp_fm.FaceMesh(static_image_mode=True, max_num_faces=1,
                            refine_landmarks=False, min_detection_confidence=0.5) as mesh:
            rgb = cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB)
            result = mesh.process(rgb)

        canvas = base_frame.copy()
        H, W = canvas.shape[:2]
        lm_count = 0
        if result and result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0].landmark
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
        _header(canvas, "OUTPUT 01 -- Face Landmark Dots",
                f"{lm_count} landmarks  |  Cyan=Eyes  Yellow=Mouth  Orange=Nose",
                bar=(20, 20, 40))
        path = os.path.join(img_dir, "01_landmark_dots.jpg")
        cv2.imwrite(path, canvas)
        print(f"  [IMG] {path} ({lm_count} landmarks)")
    except Exception as e:
        print(f"  [!] Landmark error: {e}")

    try:
        boxes = engine.detect(base_frame)
        canvas = base_frame.copy()
        for i, (x1, y1, x2, y2, score) in enumerate(boxes):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), CLR_GREEN, 2)
            _txt(canvas, f"FACE #{i+1} conf={score:.3f}", (x1+4, y1-5), 0.5, CLR_CYAN)
        _header(canvas, "OUTPUT 03 -- BlazeFace Detection",
                f"{len(boxes)} face(s) detected", bar=(10, 30, 20))
        path = os.path.join(img_dir, "03_blazeface_detection.jpg")
        cv2.imwrite(path, canvas)
        print(f"  [IMG] {path} ({len(boxes)} faces)")
    except Exception as e:
        print(f"  [!] BlazeFace vis error: {e}")

    try:
        boxes = engine.detect(base_frame)
        best_box  = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1])) if boxes else None
        face_crop = None
        if best_box:
            x1, y1, x2, y2, _ = best_box
            face_crop = base_frame[max(0, y1):y2, max(0, x1):x2]

        CW, CH = 900, 400
        canvas = np.full((CH, CW, 3), 18, dtype=np.uint8)
        emb = engine._embed_face(face_crop) if face_crop is not None else None
        if emb is not None:
            dim = emb.shape[0]
            _txt(canvas, f"dim={dim}  norm={np.linalg.norm(emb):.4f}  "
                 f"mean={np.mean(emb):.4f}  std={np.std(emb):.4f}",
                 (10, 80), 0.5, CLR_CYAN)
            bw = max(1, (CW-20)//min(dim, 256))
            for i in range(min(dim, 256)):
                v = float(emb[i])
                bh = int(min(abs(v)*80, 80))
                bx = 10 + i*bw
                col = CLR_GREEN if v >= 0 else CLR_RED
                cv2.rectangle(canvas, (bx, 200-bh), (bx+bw-1, 200), col, -1)
        else:
            _txt(canvas, "No embedding", (200, 200), 1.0, CLR_RED, 2)
        _header(canvas, "OUTPUT 05 -- Face Vector (MobileFaceNet)",
                f"engine={engine.mode}", bar=(30, 10, 30))
        path = os.path.join(img_dir, "05_face_vector.jpg")
        cv2.imwrite(path, canvas)
        print(f"  [IMG] {path}")
    except Exception as e:
        print(f"  [!] Face vector error: {e}")


# ============================================================
# MARKDOWN REPORT
# ============================================================

def _write_report(out_dir, ts_str, res_a, res_b, res_c, res_d):
    import config

    def _f(v, d=2):
        return "—" if (v is None or v == 0) else f"{v:.{d}f}"

    lines = [
        "# Laporan Pengujian Performa Sistem Kendali Akses",
        "",
        f"> Dibuat otomatis oleh `main.py --perf`  ",
        f"> Waktu: {ts_str}",
        "",
        "---",
        "",
        "## 1. Informasi Umum",
        "| Parameter | Nilai |",
        "| --- | --- |",
        f"| Resolusi Kamera | {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT} px |",
        f"| Face Match Threshold | {config.FACE_MATCH_THRESH} |",
        f"| EAR Threshold | {config.BLINK_EAR_THRESHOLD} |",
        f"| Liveness Duration | {config.LIVENESS_DURATION} s |",
        "",
        "---",
        "",
        "## 2. [A] BlazeFace -- Deteksi Wajah",
        "| Metrik | Nilai |",
        "| --- | --- |",
        f"| Detection Rate | {_f(res_a.get('det_rate'))} % |",
        f"| Frame Terdeteksi | {res_a.get('detected_n','—')} / {res_a.get('total','—')} |",
        f"| Inference Rerata | {_f(res_a.get('avg_inf_ms'))} ms |",
        f"| Latency P50 | {_f(res_a.get('p50_ms'))} ms |",
        f"| Latency P95 | {_f(res_a.get('p95_ms'))} ms |",
        f"| Estimasi FPS | {_f(res_a.get('fps_est'))} FPS |",
        "",
        "---",
        "",
        "## 3. [B] Liveness -- EAR Blink Detection",
        "| Metrik | Nilai |",
        "| --- | --- |",
        f"| Blink Terdeteksi | {res_b.get('blinks','—')} |",
        f"| EAR Rerata | {_f(res_b.get('avg_ear'),4)} |",
        f"| Frame Valid | {res_b.get('valid_frames','—')} |",
        "",
        "*Template APCER/BPCER/ACER: `csv/B_liveness_apcer_template.csv` (isi manual)*",
        "",
        "---",
        "",
        "## 4. [C] MobileFaceNet -- Face Recognition",
        "| Metrik | Nilai |",
        "| --- | --- |",
        f"| Embedding Berhasil | {res_c.get('n_embs','—')} |",
        f"| Embedding Gagal | {res_c.get('failed','—')} |",
        f"| Cosine Similarity Rerata | {_f(res_c.get('avg_sim'),4)} |",
        f"| Std Deviasi | {_f(res_c.get('std_sim'),4)} |",
        f"| Threshold Sistem | {res_c.get('threshold','—')} |",
        "",
        "*Template FAR/FRR/EER: `csv/C_mobilefacenet_far_frr_template.csv` (isi manual)*",
        "",
        "---",
        "",
        "## 5. [D] Pipeline Profiling -- Performa Real-time",
        "| Metrik | Nilai |",
        "| --- | --- |",
        f"| FPS Pipeline | {_f(res_d.get('fps_pipeline'))} FPS |",
        f"| FPS Deteksi Wajah | {_f(res_d.get('fps_detect'))} FPS |",
        f"| Latency BlazeFace | {_f(res_d.get('avg_det_ms'))} ms |",
        f"| Latency Liveness EAR | {_f(res_d.get('avg_live_ms'))} ms |",
        f"| Latency Face Embedding | {_f(res_d.get('avg_emb_ms'))} ms |",
        f"| Latency Total Loop | {_f(res_d.get('avg_total_ms'))} ms |",
        f"| CPU Proses | {_f(res_d.get('cpu_proc_pct'))} % |",
        f"| CPU Sistem | {_f(res_d.get('cpu_sys_pct'))} % |",
        f"| RAM Proses | {_f(res_d.get('ram_proc_mb'))} MB |",
        f"| Total Frame | {res_d.get('frames','—')} |",
        f"| Wajah Terdeteksi | {res_d.get('faces','—')} |",
        f"| Durasi Pengujian | {_f(res_d.get('elapsed_s'))} s |",
        "",
        "---",
        "",
        "## 6. Panduan Pengisian Data Manual",
        "",
        "### APCER / BPCER / ACER (Liveness Anti-Spoofing)",
        "Isi `csv/B_liveness_apcer_template.csv`:",
        "",
        "```",
        "APCER = FP_spoof  / total_spoof   (foto/video lolos = False Accept)",
        "BPCER = FN_live   / total_live    (orang asli ditolak = False Reject)",
        "ACER  = (APCER + BPCER) / 2",
        "```",
        "",
        "### FAR / FRR / EER / Accuracy (Face Recognition)",
        "Isi `csv/C_mobilefacenet_far_frr_template.csv`:",
        "",
        "```",
        "FAR      = FP_impostor / total_impostor",
        "FRR      = FN_genuine  / total_genuine",
        "EER      = threshold saat FAR == FRR",
        "Accuracy = (TP + TN) / total_pasang",
        "```",
        "",
    ]
    md_path = os.path.join(out_dir, "profiling_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  [REPORT] {md_path}")
    return md_path


# ============================================================
# MAIN ENTRY POINT (dipanggil dari main.py)
# ============================================================

def run_perf(n_frames=20, duration_seconds=10, cam_index=0):
    """
    Jalankan seluruh pengujian performa dan prototipe.

    Args:
        n_frames         : jumlah frame untuk uji A/B/C
        duration_seconds : durasi profiling pipeline (uji D)
        cam_index        : indeks kamera
    """
    ts        = datetime.datetime.now()
    ts_str    = ts.strftime("%Y-%m-%d %H:%M:%S")
    ts_folder = ts.strftime("%Y%m%d_%H%M%S")

    base_dir = "perf_output"
    out_dir  = os.path.join(base_dir, ts_folder)
    csv_dir  = os.path.join(out_dir, "csv")
    img_dir  = os.path.join(out_dir, "img")
    for d in (out_dir, csv_dir, img_dir):
        os.makedirs(d, exist_ok=True)

    print(f"\n{SEP2}")
    print(f"  PERF MODE -- Sistem Kendali Akses")
    print(f"  Waktu  : {ts_str}")
    print(f"  Output : {out_dir}/")
    print(SEP2)

    if not CV2_OK:
        print("[ERR] OpenCV tidak tersedia. --perf membutuhkan cv2.")
        return

    print("\n[INIT] Loading FaceEngine ...")
    from core.face_engine import FaceEngine
    engine = FaceEngine()
    if not engine.load():
        print("[ERR] FaceEngine gagal dimuat -- aborted.")
        return
    print(f"       Mode: {engine.mode}")

    print(f"\n[CAPTURE] Mengambil {n_frames} frame dari kamera ...")
    frames     = _capture_frames(n=n_frames, cam_index=cam_index)
    base_frame = frames[-1].copy()
    print(f"          {len(frames)} frame ({base_frame.shape[1]}x{base_frame.shape[0]})")

    res_a = uji_blazeface(engine, frames, img_dir, csv_dir)
    res_b = uji_liveness_debug(engine, frames, img_dir, csv_dir)
    res_c = uji_mobilefacenet(engine, frames, img_dir, csv_dir)
    res_d = uji_pipeline_profiling(engine, duration_seconds=duration_seconds,
                                   max_frames=200, cam_index=cam_index,
                                   csv_dir=csv_dir, out_dir=out_dir)

    print(f"\n[VISUAL] Menyimpan gambar output ...")
    save_visual_outputs(engine, base_frame, img_dir)

    _write_report(out_dir, ts_str, res_a, res_b, res_c, res_d)

    print(f"\n{SEP2}")
    print(f"  SELESAI -- Output tersimpan di: ./{out_dir}/")
    print(f"  CSV    : ./{csv_dir}/")
    print(f"  Gambar : ./{img_dir}/")
    print(f"  Laporan: ./{out_dir}/profiling_report.md")
    print(SEP2 + "\n")

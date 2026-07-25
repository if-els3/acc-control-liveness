import os
import cv2
import time
import numpy as np
import sys

# Ensure core imports work
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
    mp_drawing   = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
except ImportError:
    mp = None
    mp_face_mesh = None

# ──────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ──────────────────────────────────────────────────────────────

FONT       = cv2.FONT_HERSHEY_SIMPLEX
CLR_GREEN  = (80,  220, 80)
CLR_CYAN   = (220, 220, 0)
CLR_YELLOW = (0,   220, 220)
CLR_RED    = (60,  60,  255)
CLR_WHITE  = (240, 240, 240)
CLR_GRAY   = (140, 140, 140)
CLR_ORANGE = (0,   165, 255)


def draw_text(img, text, pos=(10, 30), scale=0.55, color=CLR_GREEN, thickness=1):
    cv2.putText(img, text, pos, FONT, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, FONT, scale, color, thickness)


def header_bar(img, title, subtitle="", bar_color=(30, 30, 30)):
    """Draw a dark top bar with title + subtitle."""
    cv2.rectangle(img, (0, 0), (img.shape[1], 50), bar_color, -1)
    draw_text(img, title,    (10, 22), scale=0.7,  color=CLR_WHITE, thickness=2)
    draw_text(img, subtitle, (10, 42), scale=0.45, color=CLR_GRAY,  thickness=1)


def border(img, color=(80, 80, 80), thickness=2):
    cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1), color, thickness)


def status_pill(img, text, x, y, ok=True):
    """Draw a small colored pill label."""
    color  = CLR_GREEN if ok else CLR_RED
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    cv2.rectangle(img, (x-4, y-th-4), (x+tw+4, y+4), color, -1)
    cv2.putText(img, text, (x, y), FONT, 0.45, (0, 0, 0), 2)
    cv2.putText(img, text, (x, y), FONT, 0.45, (0, 0, 0), 1)


# ──────────────────────────────────────────────────────────────
# CAPTURE FRAMES
# ──────────────────────────────────────────────────────────────

def capture_frames(n=15):
    cap = cv2.VideoCapture(0)
    print("[*] Waiting for camera warmup...")
    time.sleep(2)
    frames = []
    for _ in range(n):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        time.sleep(0.08)
    cap.release()

    if not frames:
        print("[!] Camera not found — using synthetic dummy frame.")
        dummy = np.full((480, 640, 3), 40, dtype=np.uint8)
        cv2.putText(dummy, "NO CAMERA", (180, 240), FONT, 1.2, CLR_RED, 2)
        frames = [dummy] * n

    return frames


# ──────────────────────────────────────────────────────────────
# OUTPUT 1 — DOTTED LANDMARK DETECTED IN FACE
# ──────────────────────────────────────────────────────────────

def save_01_landmarks(base_frame, out_dir, face_landmarks_result):
    """Draw every one of 468 MediaPipe face-mesh landmarks as a colored dot."""
    H, W = base_frame.shape[:2]
    canvas = base_frame.copy()

    subtitle = "No MediaPipe landmarks found"
    lm_count = 0

    if face_landmarks_result and face_landmarks_result.multi_face_landmarks:
        lms = face_landmarks_result.multi_face_landmarks[0].landmark
        lm_count = len(lms)

        for i, lm in enumerate(lms):
            px = int(lm.x * W)
            py = int(lm.y * H)
            # Colour-code by region: eyes=cyan, mouth=yellow, rest=green
            if i in _LEFT_EYE_IDX or i in _RIGHT_EYE_IDX:
                dot_color = CLR_CYAN
                r = 3
            elif i in (_L_MOUTH_IDX, _R_MOUTH_IDX, 0, 17, 61, 291, 13, 14):
                dot_color = CLR_YELLOW
                r = 3
            elif i == _NOSE_TIP_IDX:
                dot_color = CLR_ORANGE
                r = 5
            else:
                dot_color = CLR_GREEN
                r = 2
            cv2.circle(canvas, (px, py), r, dot_color, -1)

        # Label key indices
        labels = {
            _NOSE_TIP_IDX: "Nose",
            _L_EYE_OUTER_IDX: "L-Eye",
            _R_EYE_OUTER_IDX: "R-Eye",
            _L_MOUTH_IDX: "L-Mouth",
            _R_MOUTH_IDX: "R-Mouth",
        }
        for idx, name in labels.items():
            lm = lms[idx]
            px, py = int(lm.x * W), int(lm.y * H)
            cv2.line(canvas, (px, py), (px+30, py-20), CLR_WHITE, 1)
            draw_text(canvas, name, (px+32, py-16), scale=0.38, color=CLR_WHITE)

        subtitle = f"468 landmarks detected  |  Cyan=Eyes  Yellow=Mouth  Orange=Nose  Green=Rest"
    else:
        subtitle = "MediaPipe unavailable — no landmarks drawn"

    header_bar(canvas, "OUTPUT 1 — FACE LANDMARK DOTS", subtitle, bar_color=(20, 20, 40))
    draw_text(canvas, f"Total landmarks: {lm_count}", (10, H - 30), scale=0.5, color=CLR_WHITE)
    border(canvas, CLR_CYAN)

    path = os.path.join(out_dir, "01_landmark_dots.jpg")
    cv2.imwrite(path, canvas)
    print(f"   -> Saved: 01_landmark_dots.jpg  ({lm_count} landmarks)")
    return canvas


# ──────────────────────────────────────────────────────────────
# OUTPUT 2 — 3D DEPTH (CONTOUR DEPTH RATIO)
# ──────────────────────────────────────────────────────────────

def save_02_3d_depth(base_frame, out_dir, face_landmarks_result, depth_history):
    H, W = base_frame.shape[:2]
    canvas = base_frame.copy()

    # Dark overlay to make overlaid text readable
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 50), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

    depth_ratio = None
    plane_z     = None
    nose_z      = None

    if face_landmarks_result and face_landmarks_result.multi_face_landmarks:
        lms = face_landmarks_result.multi_face_landmarks[0].landmark
        depth_ratio = _compute_contour_depth_ratio(lms)

        nose  = lms[_NOSE_TIP_IDX]
        le    = lms[_L_EYE_OUTER_IDX]
        re_   = lms[_R_EYE_OUTER_IDX]
        lm_l  = lms[_L_MOUTH_IDX]
        lm_r  = lms[_R_MOUTH_IDX]
        plane_z = (le.z + re_.z + lm_l.z + lm_r.z) / 4.0
        nose_z  = nose.z

        # Draw reference landmarks with depth labels
        for idx, label, color in [
            (_L_EYE_OUTER_IDX, f"L-Eye  z={le.z:.4f}", CLR_CYAN),
            (_R_EYE_OUTER_IDX, f"R-Eye  z={re_.z:.4f}", CLR_CYAN),
            (_NOSE_TIP_IDX,   f"Nose   z={nose.z:.4f}", CLR_ORANGE),
            (_L_MOUTH_IDX,    f"L-Mouth z={lm_l.z:.4f}", CLR_YELLOW),
            (_R_MOUTH_IDX,    f"R-Mouth z={lm_r.z:.4f}", CLR_YELLOW),
        ]:
            lm = lms[idx]
            px, py = int(lm.x * W), int(lm.y * H)
            cv2.circle(canvas, (px, py), 5, color, -1)
            cv2.line(canvas, (px, py), (px+35, py-15), color, 1)
            draw_text(canvas, label, (px+37, py-11), scale=0.38, color=color)

    # Info panel (bottom)
    panel_y = H - 160
    cv2.rectangle(canvas, (10, panel_y), (W-10, H-10), (20, 20, 20), -1)
    cv2.rectangle(canvas, (10, panel_y), (W-10, H-10), CLR_YELLOW, 1)

    if depth_ratio is not None:
        threshold   = 0.12
        is_3d       = depth_ratio >= threshold
        status_txt  = "3D FACE (LIVE)" if is_3d else "FLAT (SPOOF RISK)"
        status_clr  = CLR_GREEN if is_3d else CLR_RED

        draw_text(canvas, f"Nose Z       : {nose_z:.5f}",    (20, panel_y+22),  color=CLR_WHITE)
        draw_text(canvas, f"Plane Z (avg): {plane_z:.5f}",   (20, panel_y+42),  color=CLR_WHITE)
        draw_text(canvas, f"Protrusion   : {plane_z-nose_z:.5f}", (20, panel_y+62), color=CLR_ORANGE)
        draw_text(canvas, f"Depth Ratio  : {depth_ratio:.5f} (thresh={threshold})", (20, panel_y+82), color=CLR_CYAN)
        if depth_history:
            draw_text(canvas, f"History avg  : {np.mean(depth_history):.5f}  ({len(depth_history)} frames)", (20, panel_y+102), color=CLR_GRAY)
        draw_text(canvas, f"Decision     : {status_txt}", (20, panel_y+125), scale=0.6, color=status_clr, thickness=2)
    else:
        draw_text(canvas, "Depth ratio: N/A (MediaPipe not available)", (20, panel_y+22), color=CLR_RED)
        draw_text(canvas, "Formula: (plane_z - nose_z) / interocular_dist", (20, panel_y+48), color=CLR_GRAY)

    subtitle = f"Formula: protrusion = plane_avg_z - nose_z  /  interocular  |  thresh≥0.12 => 3D"
    header_bar(canvas, "OUTPUT 2 — 3D DEPTH (CONTOUR DEPTH RATIO)", subtitle, bar_color=(20, 10, 40))
    border(canvas, CLR_YELLOW)

    path = os.path.join(out_dir, "02_3d_depth.jpg")
    cv2.imwrite(path, canvas)
    depth_str = f"{depth_ratio:.5f}" if depth_ratio is not None else "N/A"
    print(f"   -> Saved: 02_3d_depth.jpg  (depth_ratio={depth_str})")


# ──────────────────────────────────────────────────────────────
# OUTPUT 3 — BLAZEFACE DETECTION
# ──────────────────────────────────────────────────────────────

def save_03_blazeface(base_frame, out_dir, boxes):
    H, W = base_frame.shape[:2]
    canvas = base_frame.copy()

    detected = len(boxes) > 0

    for i, (x1, y1, x2, y2, score) in enumerate(boxes):
        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), CLR_GREEN, 2)
        # Corner accent lines
        corner_len = 15
        for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(canvas, (cx, cy), (cx+dx*corner_len, cy), CLR_ORANGE, 3)
            cv2.line(canvas, (cx, cy), (cx, cy+dy*corner_len), CLR_ORANGE, 3)

        # Score badge
        badge_txt = f"FACE #{i+1}  conf={score:.3f}"
        (bw, bh), _ = cv2.getTextSize(badge_txt, FONT, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1-bh-10), (x1+bw+8, y1), (30,30,30), -1)
        draw_text(canvas, badge_txt, (x1+4, y1-5), scale=0.5, color=CLR_CYAN)

        # Dimensions annotation
        face_w, face_h = x2-x1, y2-y1
        draw_text(canvas, f"{face_w}x{face_h}px", (x1+5, y2-10), scale=0.4, color=CLR_YELLOW)

    if not detected:
        draw_text(canvas, "No Face Detected", (W//2-110, H//2), scale=1.0, color=CLR_RED, thickness=2)

    # Info panel
    panel_y = H - 80
    cv2.rectangle(canvas, (10, panel_y), (W-10, H-10), (20, 20, 20), -1)
    cv2.rectangle(canvas, (10, panel_y), (W-10, H-10), CLR_GREEN, 1)
    draw_text(canvas, f"Faces detected  : {len(boxes)}", (20, panel_y+22), color=CLR_WHITE)
    draw_text(canvas, f"Model           : BlazeFace PyTorch (128x128 input)", (20, panel_y+44), color=CLR_GRAY, scale=0.45)
    draw_text(canvas, f"Output          : [x1,y1,x2,y2, confidence]", (20, panel_y+62), color=CLR_GRAY, scale=0.45)

    subtitle = "BlazeFace: 128x128 input → bounding box + confidence score (detect_largest used)"
    header_bar(canvas, "OUTPUT 3 — BLAZEFACE DETECTION", subtitle, bar_color=(10, 30, 20))
    border(canvas, CLR_GREEN)

    path = os.path.join(out_dir, "03_blazeface_detection.jpg")
    cv2.imwrite(path, canvas)
    print(f"   -> Saved: 03_blazeface_detection.jpg  ({len(boxes)} face(s))")
    return boxes


# ──────────────────────────────────────────────────────────────
# OUTPUT 4 — LIVENESS TRACKER BY EAR
# ──────────────────────────────────────────────────────────────

def save_04_liveness_ear(base_frame, out_dir, frames, face_box, face_landmarks_result):
    H, W = base_frame.shape[:2]
    canvas = base_frame.copy()

    ear_l = ear_r = avg_ear = None
    lm_result_ok = False

    if face_landmarks_result and face_landmarks_result.multi_face_landmarks:
        lms = face_landmarks_result.multi_face_landmarks[0].landmark
        lm_result_ok = True
        ear_l = _compute_ear(lms, _LEFT_EYE_IDX,  W, H)
        ear_r = _compute_ear(lms, _RIGHT_EYE_IDX, W, H)
        avg_ear = (ear_l + ear_r) / 2.0

        # Draw EAR landmark points and connecting lines
        def draw_eye_pts(indices, color):
            pts = [(int(lms[i].x*W), int(lms[i].y*H)) for i in indices]
            for p in pts:
                cv2.circle(canvas, p, 4, color, -1)
            # horizontal axis (p1–p4)
            cv2.line(canvas, pts[0], pts[3], CLR_GRAY, 1)
            # vertical axes (p2–p6, p3–p5)
            cv2.line(canvas, pts[1], pts[5], color, 1)
            cv2.line(canvas, pts[2], pts[4], color, 1)

        draw_eye_pts(_LEFT_EYE_IDX,  CLR_CYAN)
        draw_eye_pts(_RIGHT_EYE_IDX, CLR_YELLOW)

    # Run BlinkDetector over all collected frames
    ear_thresh    = 0.20
    blink_count   = 0
    ear_timeline  = []

    if MP_OK and face_box is not None and len(frames) > 2:
        detector = BlinkDetector()
        box = face_box[:4]
        for f in frames:
            x1, y1, x2, y2 = box
            crop = f[max(0,y1):y2, max(0,x1):x2]
            if crop.size > 0:
                ear_val = detector.update(crop)
                if ear_val is not None:
                    ear_timeline.append(ear_val)
        blink_count = detector.blink_count

    # Draw EAR timeline as a mini sparkline chart
    chart_x, chart_y = 10, H - 140
    chart_w, chart_h = W - 20, 90
    cv2.rectangle(canvas, (chart_x, chart_y), (chart_x+chart_w, chart_y+chart_h), (25,25,25), -1)
    cv2.rectangle(canvas, (chart_x, chart_y), (chart_x+chart_w, chart_y+chart_h), CLR_GRAY, 1)
    draw_text(canvas, "EAR Timeline", (chart_x+5, chart_y+14), scale=0.4, color=CLR_GRAY)
    draw_text(canvas, f"thresh={ear_thresh}", (chart_x+chart_w-90, chart_y+14), scale=0.38, color=CLR_RED)

    if ear_timeline:
        n = len(ear_timeline)
        for i in range(1, n):
            x_a = chart_x + int((i-1) / max(n-1, 1) * chart_w)
            x_b = chart_x + int(i     / max(n-1, 1) * chart_w)
            y_a = chart_y + chart_h - int(np.clip(ear_timeline[i-1]/0.5, 0,1) * chart_h)
            y_b = chart_y + chart_h - int(np.clip(ear_timeline[i]  /0.5, 0,1) * chart_h)
            color = CLR_RED if ear_timeline[i] < ear_thresh else CLR_GREEN
            cv2.line(canvas, (x_a, y_a), (x_b, y_b), color, 2)

        # threshold line
        thresh_y = chart_y + chart_h - int(np.clip(ear_thresh/0.5, 0,1) * chart_h)
        cv2.line(canvas, (chart_x, thresh_y), (chart_x+chart_w, thresh_y), CLR_RED, 1)

    # Info panel above chart
    panel_y = H - 230
    cv2.rectangle(canvas, (10, panel_y), (W-10, H-150), (20, 20, 20), -1)
    cv2.rectangle(canvas, (10, panel_y), (W-10, H-150), CLR_CYAN, 1)

    if lm_result_ok and avg_ear is not None:
        eye_state = "CLOSED" if avg_ear < ear_thresh else "OPEN"
        eye_color = CLR_RED if avg_ear < ear_thresh else CLR_GREEN
        draw_text(canvas, f"EAR Left     : {ear_l:.4f}", (20, panel_y+20), color=CLR_CYAN)
        draw_text(canvas, f"EAR Right    : {ear_r:.4f}", (20, panel_y+40), color=CLR_YELLOW)
        draw_text(canvas, f"EAR Avg      : {avg_ear:.4f}  (thresh={ear_thresh})", (20, panel_y+60), color=CLR_WHITE)
        draw_text(canvas, f"Eye State    : {eye_state}", (20, panel_y+80), color=eye_color, thickness=2)
    else:
        draw_text(canvas, "EAR: MediaPipe not available or no face", (20, panel_y+20), color=CLR_RED)

    frames_n = len(ear_timeline)
    draw_text(canvas, f"Blinks detected: {blink_count}  |  Frames analysed: {frames_n}", (20, panel_y+55 if not lm_result_ok else panel_y+105), color=CLR_ORANGE)

    formula_txt = "EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)"
    draw_text(canvas, formula_txt, (10, H-148), scale=0.42, color=CLR_GRAY)

    subtitle = f"EAR formula  |  cyan=Left eye  yellow=Right eye  |  red line=blink thresh={ear_thresh}"
    header_bar(canvas, "OUTPUT 4 — LIVENESS TRACKER (EAR)", subtitle, bar_color=(10, 20, 40))
    border(canvas, CLR_CYAN)

    path = os.path.join(out_dir, "04_liveness_ear.jpg")
    cv2.imwrite(path, canvas)
    print(f"   -> Saved: 04_liveness_ear.jpg  (EAR_avg={avg_ear:.4f if avg_ear else 'N/A'}, blinks={blink_count})")


# ──────────────────────────────────────────────────────────────
# OUTPUT 5 — FACE VECTOR + MOBILEFACENET PER-DIMENSION
# ──────────────────────────────────────────────────────────────

def save_05_face_vector(base_frame, out_dir, face_crop, engine):
    H_src, W_src = base_frame.shape[:2]
    # Fixed canvas: left=face preview, right=vector analysis
    CW, CH = 900, 560
    canvas = np.full((CH, CW, 3), 18, dtype=np.uint8)

    emb = None
    mode = engine.mode if engine else "Unknown"

    if face_crop is not None and face_crop.size > 0:
        emb = engine._embed_face(face_crop)

    # ── Left panel: face preview ──────────────────────────────
    panel_left = 270
    draw_text(canvas, "Face Input", (10, 64), scale=0.5, color=CLR_GRAY)
    if face_crop is not None and face_crop.size > 0:
        preview = cv2.resize(face_crop, (240, 240))
        canvas[70:310, 15:255] = preview
        cv2.rectangle(canvas, (15, 70), (255, 310), CLR_GREEN, 2)
        # Show preprocessing steps
        face_112 = cv2.resize(face_crop, (112, 112))
        face_norm = (face_112.astype(np.float32) - 127.5) / 128.0
        draw_text(canvas, "Preprocessed (112x112):", (10, 326), scale=0.42, color=CLR_GRAY)
        preview_sm = cv2.resize(face_112, (80, 80))
        canvas[335:415, 15:95] = preview_sm
        cv2.rectangle(canvas, (15, 335), (95, 415), CLR_ORANGE, 1)
        draw_text(canvas, "Normalize:", (100, 350), scale=0.38, color=CLR_GRAY)
        draw_text(canvas, "(px - 127.5) / 128.0", (100, 368), scale=0.38, color=CLR_WHITE)
        draw_text(canvas, f"Range: [{face_norm.min():.3f}, {face_norm.max():.3f}]", (100, 386), scale=0.38, color=CLR_CYAN)
        draw_text(canvas, f"Shape: {face_norm.shape}", (100, 404), scale=0.38, color=CLR_YELLOW)
    else:
        draw_text(canvas, "No face crop", (60, 200), color=CLR_RED)

    # Separator
    cv2.line(canvas, (panel_left-10, 55), (panel_left-10, CH-10), (50,50,50), 1)

    # ── Right panel: vector analysis ─────────────────────────
    px0 = panel_left + 5
    draw_text(canvas, f"Embedding Engine: {mode}", (px0, 64), scale=0.48, color=CLR_ORANGE)

    if emb is not None:
        dim = emb.shape[0]
        norm_val = float(np.linalg.norm(emb))
        mean_val = float(np.mean(emb))
        std_val  = float(np.std(emb))
        pos_pct  = float(np.sum(emb > 0)) / dim * 100

        draw_text(canvas, f"Embedding dimension : {dim}", (px0, 86),  color=CLR_WHITE)
        draw_text(canvas, f"L2 norm (after norm): {norm_val:.6f}",    (px0, 106), color=CLR_GREEN)
        draw_text(canvas, f"Mean value          : {mean_val:.6f}",    (px0, 124), color=CLR_CYAN)
        draw_text(canvas, f"Std deviation       : {std_val:.6f}",     (px0, 142), color=CLR_YELLOW)
        draw_text(canvas, f"Positive dims       : {pos_pct:.1f}%",    (px0, 160), color=CLR_GRAY)

        # ── Full vector string (truncated for readability) ──
        emb_str_parts = [f"{v:.4f}" for v in emb[:20]]
        emb_str = "[ " + ", ".join(emb_str_parts) + (" ... ]" if dim > 20 else " ]")
        draw_text(canvas, "Vector string (first 20 dims):", (px0, 182), scale=0.42, color=CLR_GRAY)
        # Word-wrap into 2 lines
        half = len(emb_str_parts) // 2
        line1 = "[ " + ", ".join(emb_str_parts[:half])
        line2 = "  " + ", ".join(emb_str_parts[half:20]) + (" ... ]" if dim > 20 else " ]")
        draw_text(canvas, line1, (px0, 200), scale=0.36, color=CLR_WHITE)
        draw_text(canvas, line2, (px0, 216), scale=0.36, color=CLR_WHITE)

        # ── Per-dimension bar chart (first 64 dims, 2 rows of 32) ──
        chart_dims = min(dim, 64)
        bar_area_x = px0
        bar_area_y = 232
        bar_area_w = CW - px0 - 15
        bar_area_h = 170
        cv2.rectangle(canvas,
                      (bar_area_x, bar_area_y),
                      (bar_area_x+bar_area_w, bar_area_y+bar_area_h),
                      (28, 28, 28), -1)
        cv2.rectangle(canvas,
                      (bar_area_x, bar_area_y),
                      (bar_area_x+bar_area_w, bar_area_y+bar_area_h),
                      (60, 60, 60), 1)

        draw_text(canvas, f"Per-dim values (first {chart_dims} of {dim}):", (bar_area_x+4, bar_area_y+14), scale=0.4, color=CLR_GRAY)

        cols = 32
        rows = chart_dims // cols
        for row in range(rows):
            for col in range(cols):
                d_idx = row * cols + col
                if d_idx >= chart_dims:
                    break
                val = float(emb[d_idx])
                # Bar: height proportional to |val|, capped at max_bar_h
                bar_max_h = 50
                bar_h = int(np.clip(abs(val) * bar_max_h / 0.4, 1, bar_max_h))
                bx = bar_area_x + 4 + col * (bar_area_w - 8) // cols
                # Row baseline
                base_y = bar_area_y + 30 + row * 75
                color_bar = CLR_GREEN if val >= 0 else CLR_RED
                cv2.rectangle(canvas,
                               (bx, base_y - bar_h),
                               (bx + (bar_area_w - 8)//cols - 1, base_y),
                               color_bar, -1)

            # Row label
            draw_text(canvas, f"d{row*cols}–{(row+1)*cols-1}", (bar_area_x+4, bar_area_y + 30 + row*75 + 12), scale=0.33, color=CLR_GRAY)

        # ── Key individual dimension annotations ──
        TOP5_POS = np.argsort(emb)[-5:][::-1]
        TOP5_NEG = np.argsort(emb)[:5]
        draw_text(canvas, "Top-5 positive dims:", (px0, bar_area_y+bar_area_h+16), scale=0.42, color=CLR_GREEN)
        pos_str = "  ".join([f"d{i}={emb[i]:.3f}" for i in TOP5_POS])
        draw_text(canvas, pos_str, (px0, bar_area_y+bar_area_h+34), scale=0.38, color=CLR_WHITE)
        draw_text(canvas, "Top-5 negative dims:", (px0, bar_area_y+bar_area_h+54), scale=0.42, color=CLR_RED)
        neg_str = "  ".join([f"d{i}={emb[i]:.3f}" for i in TOP5_NEG])
        draw_text(canvas, neg_str, (px0, bar_area_y+bar_area_h+72), scale=0.38, color=CLR_WHITE)

    else:
        draw_text(canvas, "Embedding extraction FAILED", (px0, 100), color=CLR_RED)
        draw_text(canvas, "(No face detected or model error)", (px0, 130), scale=0.45, color=CLR_GRAY)

    subtitle = f"MobileFaceNet: 112x112 normalize → TFLite inference → L2-norm → {emb.shape[0] if emb is not None else '?'}-dim vector"
    header_bar(canvas, "OUTPUT 5 — FACE → VECTOR (MobileFaceNet)", subtitle, bar_color=(30, 10, 30))
    border(canvas, CLR_ORANGE)

    path = os.path.join(out_dir, "05_face_vector.jpg")
    cv2.imwrite(path, canvas)
    dim_str = str(emb.shape[0]) if emb is not None else "N/A"
    print(f"   -> Saved: 05_face_vector.jpg  (embedding dim={dim_str})")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def run_simulation():
    print("\n" + "="*60)
    print("  ACC-CONTROL PROTOTYPE OUTPUT TEST")
    print("="*60)

    out_dir = "test_output"
    os.makedirs(out_dir, exist_ok=True)

    # ── Init engine ──────────────────────────────────────────
    print("\n[INIT] Loading FaceEngine...")
    engine = FaceEngine()
    loaded = engine.load()
    if not loaded:
        print("[!] FaceEngine failed to load — aborting.")
        return
    print(f"       Mode: {engine.mode}")

    # ── Capture frames ───────────────────────────────────────
    print("\n[CAPTURE] Recording 15 frames from webcam...")
    frames = capture_frames(n=15)
    base_frame = frames[-1].copy()
    print(f"          {len(frames)} frame(s) captured  ({base_frame.shape[1]}x{base_frame.shape[0]})")

    # ── BlazeFace detect ─────────────────────────────────────
    print("\n[DETECT] Running BlazeFace...")
    boxes = engine.detect(base_frame)
    best_box = None
    face_crop = None
    if boxes:
        best_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        x1, y1, x2, y2, _ = best_box
        face_crop = base_frame[max(0,y1):y2, max(0,x1):x2]
        print(f"          {len(boxes)} face(s) found  best={best_box[:4]}  conf={best_box[4]:.3f}")
    else:
        print("          No face detected.")

    # ── MediaPipe FaceMesh ───────────────────────────────────
    print("\n[MESH] Running MediaPipe FaceMesh...")
    face_landmarks_result = None
    depth_history = []

    if MP_OK and mp_face_mesh is not None:
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        ) as mesh:
            rgb = cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB)
            face_landmarks_result = mesh.process(rgb)

    if face_landmarks_result and face_landmarks_result.multi_face_landmarks:
        lms = face_landmarks_result.multi_face_landmarks[0].landmark
        depth_r = _compute_contour_depth_ratio(lms)
        if depth_r is not None:
            # Also collect across all frames for history
            for f in frames:
                with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as m2:
                    r2 = m2.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                    if r2 and r2.multi_face_landmarks:
                        dr = _compute_contour_depth_ratio(r2.multi_face_landmarks[0].landmark)
                        if dr is not None:
                            depth_history.append(dr)
            depth_str = f"{depth_r:.5f}" if depth_r is not None else "N/A"
            print(f"          Landmarks: 468  depth_ratio={depth_str}")
        else:
            print("          No landmarks found.")
    else:
        print("          MediaPipe not available.")

    # ── Save all outputs ─────────────────────────────────────
    print("\n[SAVE] Generating output images...")

    print("\n  [1/5] Landmark Dots...")
    save_01_landmarks(base_frame, out_dir, face_landmarks_result)

    print("\n  [2/5] 3D Depth (Contour Depth Ratio)...")
    save_02_3d_depth(base_frame, out_dir, face_landmarks_result, depth_history)

    print("\n  [3/5] BlazeFace Detection...")
    save_03_blazeface(base_frame, out_dir, boxes)

    print("\n  [4/5] Liveness EAR Tracker...")
    save_04_liveness_ear(
        base_frame, out_dir, frames,
        best_box[:4] if best_box else None,
        face_landmarks_result,
    )

    print("\n  [5/5] Face → Vector (MobileFaceNet)...")
    save_05_face_vector(base_frame, out_dir, face_crop, engine)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  DONE! All outputs saved to: ./{out_dir}/")
    print("="*60)
    print("  01_landmark_dots.jpg      → 468 dotted face landmarks")
    print("  02_3d_depth.jpg           → 3D depth / contour depth ratio")
    print("  03_blazeface_detection.jpg→ BlazeFace bounding box + score")
    print("  04_liveness_ear.jpg       → EAR liveness tracker + timeline")
    print("  05_face_vector.jpg        → Face embedding vector + per-dim bars")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_simulation()

# ─────────────────────────────────────────────
#  web/app.py  –  Flask server REST API
# ─────────────────────────────────────────────
import time
import json
import threading
import logging
import traceback
import base64
import hmac
from functools import wraps
from flask import Flask, Response, jsonify, request
import cv2

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.camera_stream import CameraStream

log = logging.getLogger(__name__)

app = Flask(__name__)

# Global instances
_db = None
_face_engine = None
_door = None
_global_camera = CameraStream()

_task_lock = threading.Lock()
_task_status = {
    "task": "none",
    "status": "idle",
    "message": "",
    "result": None,
    "error": None
}
_stop_event = threading.Event()

# ── Shared state (ditulis dari menus/access.py atau endpoints) ───────────
_state_lock = threading.Lock()
_state = {
    "step"            : "Menunggu kartu RFID…",
    "step_code"       : "idle",
    "user_name"       : "",
    "similarity"      : None,
    "blinks"          : 0,
    "liveness_status" : "",
    "message"         : "",
    "ts"              : 0,
}

# ── Real-time overlay for face box in MJPEG stream ──────────────────────────
_rt_lock = threading.Lock()
_rt_overlay = {
    "similarity"      : None,   # float 0-1
    "blinks"          : 0,
    "liveness_status" : "",
    "active"          : False,
}

# ── Resource contention flag ─────────────────────────────────────────────────
# Penghapusan detect_largest() di stream generator karena bbox sudah disuplai
# dari menus/access.py via /api/rt-overlay. Ini menghilangkan FPS drop.


def set_liveness_busy(busy: bool):
    pass

def is_liveness_busy() -> bool:
    return False


# ── Autentikasi ──────────────────────────────────────────────────────────────

def _check_token(req) -> bool:
    """Validasi X-Access-Token header atau ?token= query param.
    Menggunakan hmac.compare_digest untuk mencegah timing attack.
    """
    expected = getattr(config, 'WEB_TOKEN', '')
    if not expected or expected == 'GANTI_TOKEN_INI_SEBELUM_DEPLOY':
        # Token belum dikonfigurasi — log warning tapi izinkan (fail-open untuk dev).
        log.warning("WEB_TOKEN belum dikonfigurasi! Set env var ACCESS_TOKEN.")
        return True
    candidate = (
        req.headers.get("X-Access-Token")
        or req.args.get("token")
        or ""
    )
    return hmac.compare_digest(candidate, expected)


def require_token(f):
    """Decorator — proteksi endpoint administratif dengan API token.

    Cara penggunaan:
        curl -H "X-Access-Token: <token>" http://<ip>:5000/api/enroll
        atau: fetch('/api/enroll', {headers: {'X-Access-Token': token}})
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if not _check_token(request):
            log.warning("Akses ditolak ke %s dari %s (token tidak valid)",
                        request.path, request.remote_addr)
            return jsonify({"error": "Unauthorized",
                            "hint": "Sertakan header X-Access-Token yang valid"}), 401
        return f(*args, **kwargs)
    return decorated


def _check_basic_auth(req) -> bool:
    """Validasi HTTP Basic Auth untuk endpoint stream/display."""
    auth = req.authorization
    if not auth:
        return False
    user_ok = hmac.compare_digest(auth.username or "", getattr(config, 'STREAM_AUTH_USER', ''))
    pass_ok = hmac.compare_digest(auth.password or "", getattr(config, 'STREAM_AUTH_PASS', ''))
    return user_ok and pass_ok


def _basic_auth_challenge():
    """Kembalikan respons 401 dengan header WWW-Authenticate."""
    return Response(
        "Autentikasi diperlukan.", 401,
        {"WWW-Authenticate": 'Basic realm="Access Control Monitor"'}
    )


def require_stream_auth(f):
    """Decorator — proteksi /stream dan / dengan HTTP Basic Auth (opsional).
    Aktifkan dengan STREAM_AUTH_REQUIRED = True di config.py.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if getattr(config, 'STREAM_AUTH_REQUIRED', False):
            if not _check_basic_auth(request):
                return _basic_auth_challenge()
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────────────────────────────────────

def update_state(step: str, step_code: str = "idle",
                 user_name: str = "", similarity=None,
                 blinks=None, liveness_status=None, message: str = ""):
    with _state_lock:
        upd = {
            "step"      : step,
            "step_code" : step_code,
            "user_name" : user_name,
            "similarity": round(similarity, 3) if similarity is not None else None,
            "message"   : message,
            "ts"        : time.time(),
        }
        if blinks is not None:
            upd["blinks"] = blinks
        if liveness_status is not None:
            upd["liveness_status"] = liveness_status
        _state.update(upd)

def get_state() -> dict:
    with _state_lock:
        return dict(_state)

# ── MJPEG Stream ─────────────────────────────────────────────────────────────

def _mjpeg_generator():
    interval = 1.0 / getattr(config, 'STREAM_MAX_FPS', 10)
    if not _global_camera.stream or not _global_camera.stream.isOpened():
        _global_camera.start()

    font = cv2.FONT_HERSHEY_SIMPLEX
    prev_time = time.time()
    fps_val = 0.0
    
    while True:
        t0    = time.time()
        frame = _global_camera.read()
        if frame is not None:
            # Kalkulasi FPS stream
            curr_time = time.time()
            dt = curr_time - prev_time
            if dt > 0:
                fps_val = (0.9 * fps_val) + (0.1 * (1.0 / dt))
            prev_time = curr_time
            
            # Tampilkan FPS di pojok kanan atas
            cv2.putText(frame, f"FPS: {int(fps_val)}", (frame.shape[1] - 70, 20), font, 0.5, (0, 255, 255), 1)

            # Real-time overlay on face box
            with _rt_lock:
                rt = dict(_rt_overlay)

            # Gambar bounding box wajah jika disuplai oleh backend via rt-overlay
            face_box_data = rt.get("face_box")
            if face_box_data:
                x1, y1, x2, y2 = face_box_data
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, "Face", (x1, y1-10), font, 0.5, (255,255,255), 1)

            if rt.get("active") and face_box_data:
                x1, y1, x2, y2 = face_box_data
                lines = []
                sim = rt.get("similarity")
                if sim is not None:
                    pct = int(sim * 100)
                    color = (0,255,0) if pct >= 72 else (0,255,255) if pct >= 55 else (0,0,255)
                    lines.append(f"Sim: {pct}%")
                blinks = rt.get("blinks", 0)
                if blinks:
                    lines.append(f"Blinks: {blinks}")
                lv_status = rt.get("liveness_status", "")
                if lv_status:
                    c = (0,255,0) if lv_status == "LIVE" else (0,0,255)
                    lines.append(f"L: {lv_status}")
                # Draw overlay text below face box
                for i, txt in enumerate(lines):
                    y = y2 + 15 + i * 18
                    color = (0,255,0) if "LIVE" in txt else (0,0,255) if "SPOOF" in txt else (255,255,0)
                    cv2.putText(frame, txt, (x1, y), font, 0.5, color, 1)

            # Overlay similarity from state (top-left)
            state = get_state()
            sim = state.get("similarity")
            if sim is not None:
                pct = int(sim * 100)
                color = (0,255,0) if pct >= 72 else (0,255,255) if pct >= 55 else (0,0,255)
                cv2.putText(frame, f"Similarity: {pct}%", (10, 30), font, 0.7, color, 2)
            step = state.get("step", "")
            if step:
                cv2.putText(frame, step, (10, frame.shape[0]-20), font, 0.5, (255,255,255), 1)

            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg.tobytes() +
                    b"\r\n"
                )
        elapsed = time.time() - t0
        time.sleep(max(0, interval - elapsed))

@app.route("/stream")
@require_stream_auth
def stream():
    """MJPEG stream kamera.
    Dilindungi HTTP Basic Auth jika STREAM_AUTH_REQUIRED=True di config.
    Untuk display HDMI internal (kiosk mode), biarkan STREAM_AUTH_REQUIRED=False.
    """
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ── Read-only Endpoints ───────────────────────────────────────────────────────

@app.route("/api/state", methods=["GET", "POST"])
def api_state():
    if request.method == "POST":
        if not _check_token(request):  # internal write — cek token
            return jsonify({"error": "Unauthorized"}), 401
        data = request.json or {}
        kwargs = {}
        for key in ["step", "step_code", "user_name", "similarity", "message"]:
            if key in data:
                kwargs[key] = data[key]
        if "blinks" in data:
            kwargs["blinks"] = data["blinks"]
        if "liveness_status" in data:
            kwargs["liveness_status"] = data["liveness_status"]
        update_state(**kwargs)
        return jsonify({"status": "ok"})
    return jsonify(get_state())


@app.route("/api/rt-overlay", methods=["GET", "POST"])
def api_rt_overlay():
    if request.method == "POST":
        if not _check_token(request):  # internal write — cek token
            return jsonify({"error": "Unauthorized"}), 401
        data = request.json or {}
        with _rt_lock:
            for k in ["similarity", "blinks", "liveness_status", "active", "face_box"]:
                if k in data:
                    _rt_overlay[k] = data[k]
        return jsonify({"status": "ok"})
    with _rt_lock:
        return jsonify(dict(_rt_overlay))

@app.route("/api/logs")
def api_logs():
    limit = int(request.args.get('limit', 50))
    status = request.args.get('status')
    if _db is None: return jsonify([])
    logs = _db.get_logs(limit=limit, filter_status=status)
    return jsonify([dict(l) for l in logs])

@app.route("/api/stats")
def api_stats():
    if _db is None: return jsonify({})
    return jsonify(_db.statistik())

@app.route("/api/users")
def api_users():
    if _db is None: return jsonify([])
    users = _db.get_all_users()
    return jsonify([dict(u) for u in users])

@app.route("/api/config")
@require_token
def api_config():
    """Ekspos konfigurasi non-sensitif — dilindungi token."""
    SENSITIVE = {"AES_KEY_HEX", "SECRET_KEY", "API_KEY", "DATABASE_URL",
                 "WEB_TOKEN", "STREAM_AUTH_PASS", "STREAM_AUTH_USER"}
    conf = {k: v for k, v in vars(config).items()
            if not k.startswith('__') and k not in SENSITIVE
            and not callable(v)}
    return jsonify(conf)

@app.route("/api/system/info")
def api_system_info():
    return jsonify({
        "app_name": config.APP_NAME,
        "app_version": config.APP_VERSION,
        "liveness_enabled": config.LIVENESS_ENABLED,
        "face_threshold": config.FACE_MATCH_THRESH,
        "fr_mode": _face_engine.mode if _face_engine else "unknown"
    })

@app.route("/api/task/status")
def api_task_status():
    with _task_lock:
        return jsonify(_task_status)

# ── Write Endpoints ───────────────────────────────────────────────────────────

@app.route("/api/users/<int:user_id>", methods=["PUT", "DELETE"])
@require_token
def api_manage_user(user_id):
    if not _db: return jsonify({"error": "DB not ready"}), 500
    if request.method == "DELETE":
        _db.hapus_user(user_id)
        return jsonify({"status": "deleted"})
    elif request.method == "PUT":
        data = request.json or {}
        if "aktif" in data:
            if data["aktif"]: _db.aktifkan_user(user_id)
            else: _db.nonaktifkan_user(user_id)
        return jsonify({"status": "updated"})

# ── Task Runner Helper ────────────────────────────────────────────────────────

def run_task(task_name, func, *args):
    with _task_lock:
        if _task_status["status"] == "running" and task_name != "access_loop":
            return False, "Another task is running"
        _task_status.update({"task": task_name, "status": "running", "message": "Starting...", "error": None, "result": None})
    
    def target():
        try:
            res = func(*args)
            with _task_lock:
                _task_status.update({"status": "completed", "result": res, "message": "Done"})
        except Exception as e:
            with _task_lock:
                _task_status.update({"status": "error", "error": str(e), "message": traceback.format_exc()})
            log.error(f"Task {task_name} error: {e}")
            
    threading.Thread(target=target, daemon=True).start()
    return True, "Started"

# ── Action Endpoints ──────────────────────────────────────────────────────────

@app.route("/api/liveness/toggle", methods=["POST"])
@require_token
def api_toggle_liveness():
    config.LIVENESS_ENABLED = not config.LIVENESS_ENABLED
    log.info("Liveness toggled → %s oleh %s", config.LIVENESS_ENABLED, request.remote_addr)
    return jsonify({"liveness_enabled": config.LIVENESS_ENABLED})

@app.route("/api/enroll", methods=["POST"])
@require_token
def api_enroll():
    data = request.json or {}
    nama = data.get("nama")
    if not nama: return jsonify({"error": "Missing nama"}), 400
    
    def _enroll_task():
        update_state("Scan Kartu RFID", "rfid")
        from core.rfid_reader import RFIDReader
        uid = None
        with RFIDReader() as rfid:
            uid, _ = rfid.scan(timeout=config.RFID_TIMEOUT)
        
        if not uid:
            update_state("Timeout RFID", "error")
            return {"status": "error", "message": "RFID timeout"}
            
        uid_str = str(uid)
        if _db.cek_rfid_terdaftar(uid_str):
            update_state("Kartu sudah terdaftar", "error")
            return {"status": "error", "message": "RFID already registered"}
            
        update_state(f"Wajah {nama} - Hadap Kamera", "verify")
        embeddings = []
        if not _global_camera.stream or not _global_camera.stream.isOpened():
            _global_camera.start()
        
        # Ambil 5 frame
        for i in range(config.ENROLL_FRAMES):
            time.sleep(0.8)
            frame = _global_camera.read()
            if frame is None: continue
            emb = _face_engine.extract_embedding(frame)
            if emb is not None:
                embeddings.append(emb.tolist())
        
        if not embeddings:
            update_state("Wajah tidak terdeteksi", "error")
            return {"status": "error", "message": "No face detected"}
            
        user_id = _db.tambah_user(nama, uid_str, embeddings=embeddings)
        _db.catat_log(uid_str, "ENROLL", f"Pendaftaran berhasil via web", user_id=user_id, nama=nama)
        update_state("Pendaftaran Berhasil", "idle")
        return {"status": "success", "user_id": user_id, "embeddings_count": len(embeddings)}

    started, msg = run_task("enroll", _enroll_task)
    return jsonify({"task_started": started, "message": msg})

@app.route("/api/enroll/update-face", methods=["POST"])
@require_token
def api_update_face():
    def _update_face_task():
        update_state("Scan Kartu RFID untuk Update", "rfid")
        from core.rfid_reader import RFIDReader
        with RFIDReader() as rfid:
            uid, _ = rfid.scan(timeout=config.RFID_TIMEOUT)
        if not uid:
            update_state("Timeout RFID", "error")
            return {"status": "error", "message": "RFID timeout"}
        uid_str = str(uid)
        user = _db.get_user_by_rfid(uid_str)
        if not user:
            update_state("Kartu tidak dikenal", "error")
            return {"status": "error", "message": "User not found"}
            
        update_state(f"Update Wajah {user['nama']}", "verify")
        embeddings = []
        if not _global_camera.stream or not _global_camera.stream.isOpened():
            _global_camera.start()
        for i in range(config.ENROLL_FRAMES):
            time.sleep(0.8)
            frame = _global_camera.read()
            if frame is None: continue
            emb = _face_engine.extract_embedding(frame)
            if emb is not None:
                embeddings.append(emb.tolist())
                
        if not embeddings:
            update_state("Wajah tidak terdeteksi", "error")
            return {"status": "error", "message": "No face detected"}
            
        _db.update_embedding(uid_str, embeddings)
        update_state("Update Berhasil", "idle")
        return {"status": "success", "embeddings_count": len(embeddings)}

    started, msg = run_task("update_face", _update_face_task)
    return jsonify({"task_started": started, "message": msg})

@app.route("/api/liveness/test", methods=["POST"])
@require_token
def api_liveness_test():
    def _liveness_test_task():
        update_state("Uji Liveness", "liveness")
        from core.liveness import LivenessDetector
        liveness = LivenessDetector()
        
        if not _global_camera.stream or not _global_camera.stream.isOpened():
            _global_camera.start()
            
        frames = []; face_box = None; t0 = time.time()
        while time.time() - t0 < getattr(config, 'LIVENESS_DURATION', 3.0):
            frame = _global_camera.read()
            if frame is not None:
                box = _face_engine.detect_largest(frame)
                if box is not None:
                    if face_box is None: face_box = box[:4]
                    frames.append(frame)
            time.sleep(0.1)
            
        if not frames or not face_box:
            update_state("Wajah tidak terdeteksi", "error")
            return {"status": "error", "message": "No face"}
            
        res = liveness.check(frames, face_box)
        status_text = "LIVE" if res.is_live else "SPOOF"
        update_state(f"Liveness: {status_text} ({res.score:.2f})", "idle")
        return {"status": "success", "is_live": res.is_live, "score": res.score, "detail": res.detail}

    started, msg = run_task("liveness_test", _liveness_test_task)
    return jsonify({"task_started": started, "message": msg})

@app.route("/api/access/once", methods=["POST"])
@require_token
def api_access_once():
    def _access_once_task():
        from menus.access import _proses_akses
        from core.liveness import LivenessDetector
        from core.rfid_reader import RFIDReader

        update_state("Menunggu Kartu RFID", "rfid")
        with RFIDReader() as rfid:
            uid, _ = rfid.scan(timeout=60)

        if not uid:
            update_state("Timeout RFID", "idle")
            return {"status": "timeout"}

        liveness = LivenessDetector()
        if not _global_camera.stream or not _global_camera.stream.isOpened():
            _global_camera.start()

        # Define state callback to update web interface
        def _web_state_callback(step: str, step_code: str = "idle",
                               user_name: str = "", similarity=None,
                               blinks=None, liveness_status=None,
                               message: str = ""):
            update_state(step, step_code, user_name, similarity,
                         blinks=blinks, liveness_status=liveness_status,
                         message=message)

        status = _proses_akses(str(uid), _db, _face_engine, liveness, _door, _global_camera, _web_state_callback)

        if status == "GRANTED": update_state("Akses Diberikan", "granted")
        else: update_state(f"Ditolak: {status}", "denied")

        time.sleep(3)
        update_state("Menunggu...", "idle")
        return {"status": status}

    started, msg = run_task("access_once", _access_once_task)
    return jsonify({"task_started": started, "message": msg})

@app.route("/api/access/start", methods=["POST"])
@require_token
def api_access_start():
    def _access_loop():
        from menus.access import _proses_akses
        from core.liveness import LivenessDetector
        from core.rfid_reader import RFIDReader

        liveness = LivenessDetector()
        if not _global_camera.stream or not _global_camera.stream.isOpened():
            _global_camera.start()

        _stop_event.clear()
        rfid = RFIDReader()
        rfid.start()

        # Define state callback to update web interface
        def _web_state_callback(step: str, step_code: str = "idle",
                               user_name: str = "", similarity=None,
                               blinks=None, liveness_status=None,
                               message: str = ""):
            update_state(step, step_code, user_name, similarity,
                         blinks=blinks, liveness_status=liveness_status,
                         message=message)

        try:
            while not _stop_event.is_set():
                update_state("Menunggu RFID...", "idle")
                uid, _ = rfid.scan(timeout=1) # Short timeout to allow stop event checking
                if uid is None: continue

                status = _proses_akses(str(uid), _db, _face_engine, liveness, _door, _global_camera, _web_state_callback)

                if status == "GRANTED": update_state("Akses Diberikan", "granted")
                else: update_state(f"Ditolak: {status}", "denied")

                # Wait before next scan
                for _ in range(30):
                    if _stop_event.is_set(): break
                    time.sleep(0.1)
        finally:
            rfid.stop()
            update_state("Sistem Berhenti", "idle")

        return {"status": "stopped"}

    started, msg = run_task("access_loop", _access_loop)
    return jsonify({"task_started": started, "message": msg})

@app.route("/api/access/stop", methods=["POST"])
@require_token
def api_access_stop():
    _stop_event.set()
    return jsonify({"status": "stopping"})

# ── HTML Display (HDMI / Browser) ─────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Access Control Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0e1a;
    --surface: rgba(12, 18, 35, 0.82);
    --surface-2: rgba(20, 28, 50, 0.9);
    --border: rgba(99, 132, 255, 0.15);
    --border-bright: rgba(99, 132, 255, 0.35);
    --text: #e8eaf6;
    --text-dim: rgba(232, 234, 246, 0.55);
    --green: #22c55e;
    --green-glow: rgba(34, 197, 94, 0.35);
    --red: #ef4444;
    --red-glow: rgba(239, 68, 68, 0.35);
    --yellow: #f59e0b;
    --blue: #3b82f6;
    --purple: #8b5cf6;
    --accent: #6366f1;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', system-ui, sans-serif;
    overflow: hidden;
    height: 100vh; width: 100vw;
    -webkit-font-smoothing: antialiased;
  }

  /* ── Camera ── */
  .cam-wrap {
    position: absolute; top: 0; left: 0;
    width: 100vw; height: 100vh; z-index: 1;
    background: #000;
  }
  .cam-wrap img {
    width: 100%; height: 100%;
    object-fit: contain;
    object-position: center center;
  }

  /* ── Top bar ── */
  .top-bar {
    position: absolute; top: 0; left: 0; right: 0;
    z-index: 20;
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 24px;
    background: linear-gradient(to bottom, rgba(10,14,26,0.9) 0%, transparent 100%);
  }
  .top-bar-left { display: flex; align-items: center; gap: 10px; }
  .live-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 10px var(--green);
    animation: pulse-dot 1.8s ease-in-out infinite;
  }
  @keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(0.8)} }
  .sys-name { font-size: 0.85rem; font-weight: 600; color: var(--text); letter-spacing: 0.8px; text-transform: uppercase; }
  .top-bar-right { display: flex; align-items: center; gap: 14px; }
  .liveness-badge {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.6px;
    padding: 4px 12px; border-radius: 20px;
    background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.4);
    color: var(--green); text-transform: uppercase;
    transition: all 0.4s ease;
  }
  .liveness-badge.off {
    background: rgba(239,68,68,0.15); border-color: rgba(239,68,68,0.4); color: var(--red);
  }
  .clock { font-size: 0.82rem; font-weight: 500; color: var(--text-dim); font-variant-numeric: tabular-nums; }

  /* ── Bottom status card ── */
  .status-card {
    position: absolute; bottom: 0; left: 0; right: 0;
    z-index: 20;
    display: flex; justify-content: center;
    padding: 0 20px 40px;
    pointer-events: none;
    transform: translateY(150px) scale(0.95);
    opacity: 0;
    transition: transform 0.65s cubic-bezier(0.34, 1.56, 0.64, 1), opacity 0.4s ease;
    will-change: transform, opacity;
    perspective: 1000px;
  }
  .status-card.active { 
    transform: translateY(0) scale(1);
    opacity: 1;
  }

  .glass-card {
    background: linear-gradient(135deg, rgba(20, 28, 50, 0.8), rgba(12, 18, 35, 0.6));
    backdrop-filter: blur(24px) saturate(200%);
    -webkit-backdrop-filter: blur(24px) saturate(200%);
    border: 1px solid rgba(255,255,255,0.08);
    border-top: 1px solid rgba(255,255,255,0.15);
    border-radius: 24px;
    padding: 24px 36px 26px;
    min-width: min(480px, 92vw);
    max-width: 600px;
    box-shadow: 0 30px 60px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.02), inset 0 1px 0 rgba(255,255,255,0.1);
    display: flex; flex-direction: column; align-items: center; gap: 14px;
    transform: rotateX(5deg);
    transform-style: preserve-3d;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
  }
  .glass-card:hover {
    transform: rotateX(0deg) translateY(-5px);
    box-shadow: 0 40px 80px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.04), inset 0 1px 0 rgba(255,255,255,0.15);
  }

  /* Step badge */
  .step-badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 20px; border-radius: 30px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; transition: all 0.35s ease;
    background: rgba(99,102,241,0.2); border: 1px solid rgba(99,102,241,0.4); color: #a5b4fc;
  }
  .step-badge.idle    { background: rgba(100,116,139,0.2); border-color: rgba(100,116,139,0.3); color: #94a3b8; }
  .step-badge.rfid    { background: rgba(59,130,246,0.2); border-color: rgba(59,130,246,0.5); color: #93c5fd; }
  .step-badge.verify  { background: rgba(139,92,246,0.2); border-color: rgba(139,92,246,0.5); color: #c4b5fd; }
  .step-badge.liveness{ background: rgba(245,158,11,0.2); border-color: rgba(245,158,11,0.5); color: #fcd34d; }
  .step-badge.liveness_pass{ background: rgba(139,92,246,0.25); border-color: rgba(139,92,246,0.6); color: #ddd6fe; }
  .step-badge.granted { background: rgba(34,197,94,0.25); border-color: rgba(34,197,94,0.6); color: #86efac;
    box-shadow: 0 0 20px rgba(34,197,94,0.25); }
  .step-badge.denied  { background: rgba(239,68,68,0.25); border-color: rgba(239,68,68,0.6); color: #fca5a5;
    box-shadow: 0 0 20px rgba(239,68,68,0.25); }
  .step-badge.error   { background: rgba(239,68,68,0.3); border-color: rgba(239,68,68,0.8); color: #fff; }

  .badge-dot {
    width: 7px; height: 7px; border-radius: 50%; background: currentColor;
    opacity: 0.8;
  }
  .badge-dot.blink-anim { animation: pulse-dot 1s ease-in-out infinite; }

  /* User name */
  .user-name {
    font-size: 1.6rem; font-weight: 800; color: var(--text);
    letter-spacing: -0.3px; display: none;
  }
  .user-name.show { display: block; }

  /* Step message */
  .step-msg {
    font-size: 0.9rem; color: var(--text-dim); font-weight: 400;
    text-align: center; min-height: 1.2em;
    transition: opacity 0.3s ease;
  }

  /* Similarity bar */
  .sim-wrap {
    width: 100%; display: none; flex-direction: column; gap: 5px;
  }
  .sim-wrap.show { display: flex; }
  .sim-label { display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--text-dim); }
  .sim-bar-bg {
    width: 100%; height: 6px; background: rgba(255,255,255,0.1); border-radius: 10px; overflow: hidden;
  }
  .sim-bar-fill {
    height: 100%; border-radius: 10px; transition: width 0.5s ease, background 0.4s ease;
    background: var(--green);
  }

  /* Blink counter */
  .blink-wrap {
    display: none; align-items: center; gap: 8px;
    font-size: 0.82rem; color: var(--yellow);
  }
  .blink-wrap.show { display: flex; }
  .blink-icon { font-size: 1.1rem; animation: pulse-dot 1.2s ease-in-out infinite; }

  /* Divider */
  .divider { width: 100%; height: 1px; background: var(--border); }
</style>
</head>
<body>

  <div class="cam-wrap">
    <img id="stream" src="/stream" alt="Camera stream">
  </div>

  <!-- Top bar -->
  <div class="top-bar">
    <div class="top-bar-left">
      <div class="live-dot"></div>
      <span class="sys-name">Access Control</span>
    </div>
    <div class="top-bar-right">
      <span class="liveness-badge" id="liveness-badge">Liveness: ON</span>
      <span class="clock" id="clock">--:--:--</span>
    </div>
  </div>

  <!-- Bottom status card -->
  <div class="status-card" id="status-card">
    <div class="glass-card">
      <div class="step-badge idle" id="step-badge">
        <span class="badge-dot" id="badge-dot"></span>
        <span id="badge-text">Standby</span>
      </div>

      <div class="user-name" id="user-name"></div>

      <div class="step-msg" id="step-msg">Silakan tap kartu RFID</div>

      <div class="sim-wrap" id="sim-wrap">
        <div class="divider"></div>
        <div class="sim-label">
          <span>Kecocokan Wajah</span>
          <span id="sim-pct">0%</span>
        </div>
        <div class="sim-bar-bg">
          <div class="sim-bar-fill" id="sim-bar" style="width:0%"></div>
        </div>
      </div>

      <div class="blink-wrap" id="blink-wrap">
        <div class="divider" style="margin-bottom:4px"></div>
        <span class="blink-icon">👁</span>
        <span id="blink-text">Kedipan: 0</span>
      </div>
    </div>
  </div>

<script>
const BADGE_CLASSES = ['idle','rfid','verify','liveness','liveness_pass','liveness_skip','granted','denied','error'];
const BADGE_MAP = {
  idle         : { text: 'Standby',          cls: 'idle',          msg: 'Silakan tap kartu RFID' },
  rfid         : { text: 'Tap Kartu RFID',   cls: 'rfid',          msg: 'Tempelkan kartu ke reader...' },
  verify       : { text: 'Verifikasi Wajah', cls: 'verify',        msg: 'Hadapkan wajah ke kamera' },
  liveness     : { text: 'Liveness Check',   cls: 'liveness',      msg: 'Silakan berkedip sesuai instruksi' },
  liveness_pass: { text: 'Liveness Lolos',   cls: 'liveness_pass', msg: 'Melanjutkan ke verifikasi wajah...' },
  liveness_skip: { text: 'Liveness Mati',    cls: 'idle',          msg: 'Liveness dinonaktifkan' },
  granted      : { text: 'Akses Diberikan ✓',cls: 'granted',       msg: 'Selamat datang!' },
  denied       : { text: 'Akses Ditolak ✗',  cls: 'denied',        msg: 'Identitas tidak dikenali' },
  error        : { text: 'Sistem Error ⚠',   cls: 'error',         msg: 'Terjadi kesalahan sistem' },
};

let hideTimer = null;
let lastCode = 'idle';

// Clock
function updateClock() {
  const now = new Date();
  document.getElementById('clock').textContent =
    now.toLocaleTimeString('id-ID', { hour12: false });
}
setInterval(updateClock, 1000);
updateClock();

// Liveness status
async function refreshSystemInfo() {
  try {
    const info = await fetch('/api/system/info').then(r => r.json());
    const badge = document.getElementById('liveness-badge');
    if (info.liveness_enabled) {
      badge.textContent = 'Liveness: ON';
      badge.className = 'liveness-badge';
    } else {
      badge.textContent = 'Liveness: OFF';
      badge.className = 'liveness-badge off';
    }
  } catch(e) {}
}
setInterval(refreshSystemInfo, 5000);
refreshSystemInfo();

// State polling
async function refreshState() {
  try {
    const s = await fetch('/api/state').then(r => r.json());
    const code = s.step_code || 'idle';
    const map  = BADGE_MAP[code] || { text: s.step, cls: 'idle', msg: '' };

    // Badge
    const badge = document.getElementById('step-badge');
    const dot   = document.getElementById('badge-dot');
    BADGE_CLASSES.forEach(c => badge.classList.remove(c));
    badge.classList.add(map.cls);
    document.getElementById('badge-text').textContent = map.text;
    dot.className = 'badge-dot' + (code === 'liveness' ? ' blink-anim' : '');

    // Message — prefer custom step text
    const msg = s.step && !['Menunggu kartu RFID…','Menunggu...'].includes(s.step) ? s.step : map.msg;
    document.getElementById('step-msg').textContent = msg;

    // User name
    const nameEl = document.getElementById('user-name');
    if (s.user_name) {
      nameEl.textContent = s.user_name;
      nameEl.classList.add('show');
    } else {
      nameEl.classList.remove('show');
    }

    // Similarity bar
    const simWrap = document.getElementById('sim-wrap');
    if (s.similarity !== null && s.similarity !== undefined) {
      const pct = Math.round(s.similarity * 100);
      simWrap.classList.add('show');
      document.getElementById('sim-pct').textContent = pct + '%';
      const bar = document.getElementById('sim-bar');
      bar.style.width = pct + '%';
      bar.style.background = pct >= 78 ? '#22c55e' : pct >= 55 ? '#f59e0b' : '#ef4444';
    } else {
      simWrap.classList.remove('show');
    }

    // Blink counter
    const blinkWrap = document.getElementById('blink-wrap');
    if (code === 'liveness' && s.blinks !== undefined) {
      blinkWrap.classList.add('show');
      document.getElementById('blink-text').textContent = 'Kedipan: ' + s.blinks;
    } else {
      blinkWrap.classList.remove('show');
    }

    // Show/hide card
    const card = document.getElementById('status-card');
    if (code === 'idle') {
      if (lastCode !== 'idle') {
        clearTimeout(hideTimer);
        hideTimer = setTimeout(() => card.classList.remove('active'), 4000);
      }
    } else {
      clearTimeout(hideTimer);
      card.classList.add('active');
    }
    lastCode = code;
  } catch(e) {}
}

setInterval(refreshState, 600);
refreshState();
</script>
</body>
</html>"""


@app.route("/")
@app.route("/display")
@require_stream_auth
def display():
    """Halaman monitoring utama — untuk display HDMI / browser.
    Dilindungi HTTP Basic Auth jika STREAM_AUTH_REQUIRED=True di config.
    """
    return _HTML

def init_app(db, face_engine, door):
    global _db, _face_engine, _door
    _db = db
    _face_engine = face_engine
    _door = door

def run_web(db, face_engine, door, host=None, port=None):
    """Jalankan Flask server di thread terpisah.

    Jika WEB_USE_SSL=True di config, server berjalan dengan HTTPS menggunakan
    sertifikat self-signed. Generate sertifikat terlebih dahulu:
        mkdir -p certs
        openssl req -x509 -newkey rsa:2048 -keyout certs/server.key \\
          -out certs/server.crt -days 365 -nodes -subj "/CN=access-control"
    """
    init_app(db, face_engine, door)
    h = host or getattr(config, 'WEB_HOST', '0.0.0.0')
    p = port or getattr(config, 'WEB_PORT', 5000)

    # ── SSL context ──────────────────────────────────────────────────────────
    ssl_ctx = None
    if getattr(config, 'WEB_USE_SSL', False):
        cert = getattr(config, 'SSL_CERT_FILE', '')
        key  = getattr(config, 'SSL_KEY_FILE', '')
        if os.path.exists(cert) and os.path.exists(key):
            ssl_ctx = (cert, key)
            log.info("SSL aktif — menggunakan sertifikat: %s", cert)
        else:
            log.error(
                "WEB_USE_SSL=True tapi file sertifikat tidak ditemukan!\n"
                "  cert : %s\n  key  : %s\n"
                "Jalankan: openssl req -x509 -newkey rsa:2048 "
                "-keyout certs/server.key -out certs/server.crt "
                "-days 365 -nodes -subj '/CN=access-control'",
                cert, key
            )

    # ── Log peringatan token ──────────────────────────────────────────────────
    token = getattr(config, 'WEB_TOKEN', '')
    if not token or token == 'GANTI_TOKEN_INI_SEBELUM_DEPLOY':
        log.warning(
            "[KEAMANAN] WEB_TOKEN belum diubah dari nilai default!\n"
            "  Set env var: export ACCESS_TOKEN='token-rahasia-anda'\n"
            "  atau ubah WEB_TOKEN di config.py sebelum deploy."
        )

    proto = "https" if ssl_ctx else "http"
    t = threading.Thread(
        target=lambda: app.run(
            host=h, port=p, threaded=True,
            use_reloader=False, debug=False,
            ssl_context=ssl_ctx
        ),
        daemon=True,
        name="FlaskThread"
    )
    t.start()
    log.info("Web display: %s://%s:%d", proto, h, p)
    return t

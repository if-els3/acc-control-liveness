# ─────────────────────────────────────────────
#  web/app.py  –  Flask server REST API
# ─────────────────────────────────────────────
import time
import json
import threading
import logging
import traceback
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
    while True:
        t0    = time.time()
        frame = _global_camera.read()
        if frame is not None:
            # Draw face box if face engine available
            face_box_data = None
            if _face_engine and _face_engine.is_loaded():
                box = _face_engine.detect_largest(frame)
                if box:
                    x1, y1, x2, y2, score = [int(v) for v in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Face {score:.2f}", (x1, y1-10), font, 0.5, (0,255,0), 1)
                    face_box_data = (x1, y1, x2, y2)

            # Real-time overlay on face box
            with _rt_lock:
                rt = dict(_rt_overlay)

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
def stream():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ── Read-only Endpoints ───────────────────────────────────────────────────────

@app.route("/api/state", methods=["GET", "POST"])
def api_state():
    if request.method == "POST":
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
        data = request.json or {}
        with _rt_lock:
            for k in ["similarity", "blinks", "liveness_status", "active"]:
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
def api_config():
    SENSITIVE = {"AES_KEY_HEX", "SECRET_KEY", "API_KEY", "DATABASE_URL"}
    conf = {k: v for k, v in vars(config).items() if not k.startswith('__') and k not in SENSITIVE}
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
def api_toggle_liveness():
    config.LIVENESS_ENABLED = not config.LIVENESS_ENABLED
    return jsonify({"liveness_enabled": config.LIVENESS_ENABLED})

@app.route("/api/enroll", methods=["POST"])
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
def api_access_stop():
    _stop_event.set()
    return jsonify({"status": "stopping"})

# ── HTML Display (HDMI / Browser) ─────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Access Control</title>
<style>
  :root {
    --bg: #000;
    --glass-bg: rgba(15, 15, 15, 0.65);
    --glass-border: rgba(255, 255, 255, 0.1);
    --green: #22c55e;
    --red: #ef4444;
    --yellow: #eab308;
    --text: #ffffff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { 
    background: var(--bg); color: var(--text); 
    font-family: 'Segoe UI', system-ui, sans-serif;
    overflow: hidden; 
    height: 100vh; width: 100vw;
  }

  .cam-wrap { 
    position: absolute;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 1;
    background: #000;
  }
  .cam-wrap img { 
    width: 100%; height: 100%; 
    object-fit: contain;
    object-position: center center;
  }

  .overlay-container {
    position: absolute;
    bottom: 8%;
    left: 50%;
    transform: translateX(-50%) translateY(40px);
    z-index: 10;
    opacity: 0;
    transition: all 0.7s cubic-bezier(0.16, 1, 0.3, 1);
    display: flex;
    flex-direction: column;
    align-items: center;
    pointer-events: none;
    will-change: transform, opacity;
  }
  .overlay-container.active {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }

  .glass-panel {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 24px;
    padding: 24px 48px;
    text-align: center;
    min-width: 320px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
  }

  #step-badge {
    display: inline-block; padding: 8px 24px; border-radius: 30px;
    font-size: 1.1rem; font-weight: 600; margin-bottom: 12px;
    background: rgba(255, 255, 255, 0.15); color: #fff;
    transition: background 0.4s ease, box-shadow 0.4s ease;
    letter-spacing: 0.5px;
  }
  #step-badge.granted { background: rgba(34, 197, 94, 0.85); box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3); }
  #step-badge.denied  { background: rgba(239, 68, 68, 0.85); box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3); }
  #step-badge.liveness{ background: rgba(234, 179, 8, 0.85); color: #000; }

  #step-text { font-size: 1.15rem; color: rgba(255,255,255,0.9); font-weight: 500; }
  #user-name { font-size: 1.5rem; font-weight: 700; margin-top: 12px; }

  .flawless-hover {
    position: absolute;
    top: 6%;
    left: 50%;
    transform: translateX(-50%) translateY(-20px);
    background: rgba(20, 20, 20, 0.5);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 10px 24px;
    border-radius: 30px;
    font-size: 0.9rem;
    color: rgba(255,255,255,0.8);
    z-index: 10;
    opacity: 0;
    transition: all 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    pointer-events: none;
    display: flex;
    align-items: center;
    will-change: transform, opacity;
  }
  .flawless-hover.show {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }
  .dot { 
    display: inline-block; width: 8px; height: 8px; 
    border-radius: 50%; background: var(--green); 
    margin-right: 10px; box-shadow: 0 0 10px var(--green);
    animation: pulse 2s infinite; 
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

</style>
</head>
<body>

  <div class="cam-wrap">
    <img id="stream" src="/stream" alt="Camera stream">
  </div>

  <div class="overlay-container" id="overlay">
    <div class="glass-panel">
      <div id="step-badge" class="idle">Menunggu...</div>
      <div id="step-text">Silakan Tap Kartu</div>
      <div id="user-name" style="display:none"></div>
    </div>
  </div>

  <div class="flawless-hover" id="periodic-hover">
    <span class="dot"></span>Sistem Aktif
  </div>

<script>
const STEP_MAP = {
  idle    : ["Standby", "idle"],
  rfid    : ["Tap Kartu RFID", "idle"],
  liveness: ["Liveness Check", "liveness"],
  verify  : ["Verifikasi", "idle"],
  granted : ["Akses Diberikan", "granted"],
  denied  : ["Akses Ditolak", "denied"],
};

let hideTimeout;
let lastStepCode = 'idle';

async function refreshState() {
  try {
    const s = await fetch("/api/state").then(r=>r.json());
    const badge = document.getElementById("step-badge");
    const overlay = document.getElementById("overlay");
    const [label, cls] = STEP_MAP[s.step_code] || [s.step, "idle"];
    
    badge.textContent = label;
    badge.className   = cls;
    document.getElementById("step-text").textContent = s.step;

    const nameEl = document.getElementById("user-name");
    if (s.user_name) {
      nameEl.textContent = s.user_name;
      nameEl.style.display = "block";
    } else {
      nameEl.style.display = "none";
    }

    if (s.step_code === 'idle') {
      if (lastStepCode !== 'idle') {
        clearTimeout(hideTimeout);
        hideTimeout = setTimeout(() => {
          overlay.classList.remove('active');
        }, 4000); 
      }
    } else {
      clearTimeout(hideTimeout);
      overlay.classList.add('active');
    }
    
    lastStepCode = s.step_code;
  } catch(e) {}
}

function showFlawlessHover() {
  const hoverEl = document.getElementById("periodic-hover");
  hoverEl.classList.add("show");
  setTimeout(() => {
    hoverEl.classList.remove("show");
  }, 5000);
}

setInterval(refreshState, 600);
setInterval(showFlawlessHover, 600000); 

setTimeout(() => {
  refreshState();
  showFlawlessHover();
}, 500);

</script>
</body>
</html>"""


@app.route("/")
@app.route("/display")
def display():
    return _HTML

def init_app(db, face_engine, door):
    global _db, _face_engine, _door
    _db = db
    _face_engine = face_engine
    _door = door

def run_web(db, face_engine, door, host=None, port=None):
    init_app(db, face_engine, door)
    h = host or getattr(config, 'WEB_HOST', '0.0.0.0')
    p = port or getattr(config, 'WEB_PORT', 5000)
    t = threading.Thread(
        target=lambda: app.run(host=h, port=p, threaded=True,
                               use_reloader=False, debug=False),
        daemon=True,
        name="FlaskThread"
    )
    t.start()
    log.info("Web display: http://%s:%d", h, p)
    return t

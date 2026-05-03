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
    "step"       : "Menunggu kartu RFID…",
    "step_code"  : "idle",
    "user_name"  : "",
    "similarity" : None,
    "message"    : "",
    "ts"         : 0,
}

def update_state(step: str, step_code: str = "idle",
                 user_name: str = "", similarity=None,
                 message: str = ""):
    with _state_lock:
        _state.update({
            "step"       : step,
            "step_code"  : step_code,
            "user_name"  : user_name,
            "similarity" : round(similarity, 3) if similarity else None,
            "message"    : message,
            "ts"         : time.time(),
        })

def get_state() -> dict:
    with _state_lock:
        return dict(_state)

# ── MJPEG Stream ─────────────────────────────────────────────────────────────

def _mjpeg_generator():
    interval = 1.0 / getattr(config, 'STREAM_MAX_FPS', 10)
    if not _global_camera.stream or not _global_camera.stream.isOpened():
        _global_camera.start()
        
    while True:
        t0    = time.time()
        frame = _global_camera.read()
        if frame is not None:
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

@app.route("/api/state")
def api_state():
    return jsonify(get_state())

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
    conf = {k: v for k, v in vars(config).items() if not k.startswith('__')}
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
                               message: str = ""):
            update_state(step, step_code, user_name, similarity, message)

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
                               message: str = ""):
            update_state(step, step_code, user_name, similarity, message)

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
<title>Access Control Monitor</title>
<style>
  :root {
    --bg: #0d0d0d; --card: #181818; --border: #2a2a2a;
    --green: #22c55e; --red: #ef4444; --yellow: #eab308;
    --blue: #3b82f6; --text: #e5e5e5; --muted: #666;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif;
         display: grid; grid-template-rows: auto 1fr auto; min-height: 100vh; }

  header { padding: 12px 20px; background: var(--card); border-bottom: 1px solid var(--border);
           display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.1rem; font-weight: 600; }
  .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--green);
         animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

  main { display: grid; grid-template-columns: 1fr 360px; gap: 16px; padding: 16px; }

  .cam-wrap { position: relative; background: #000; border-radius: 10px; overflow: hidden;
              aspect-ratio: 4/3; display: flex; align-items: center; justify-content: center; }
  .cam-wrap img { width: 100%; height: 100%; object-fit: contain; }

  .sidebar { display: flex; flex-direction: column; gap: 12px; }

  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
          padding: 14px 16px; }
  .card h2 { font-size: .75rem; text-transform: uppercase; letter-spacing: .08em;
             color: var(--muted); margin-bottom: 10px; }

  #step-badge {
    display: inline-block; padding: 6px 14px; border-radius: 20px;
    font-size: 1rem; font-weight: 700; margin-bottom: 6px;
    background: var(--blue); color: #fff; transition: background .3s;
  }
  #step-badge.granted { background: var(--green); }
  #step-badge.denied  { background: var(--red); }
  #step-badge.liveness{ background: var(--yellow); color: #111; }

  #step-text  { font-size: .9rem; color: var(--muted); }
  #user-name  { font-size: 1.4rem; font-weight: 700; margin-top: 4px; }
  #similarity-bar { height: 6px; background: var(--border); border-radius: 3px; margin-top: 8px; }
  #similarity-fill { height: 100%; border-radius: 3px; background: var(--green);
                     width: 0; transition: width .4s; }
  #similarity-label { font-size: .75rem; color: var(--muted); margin-top: 4px; }

  .log-list { list-style: none; max-height: 220px; overflow-y: auto; }
  .log-list li { display: flex; justify-content: space-between; align-items: center;
                 padding: 5px 0; border-bottom: 1px solid var(--border); font-size: .8rem; }
  .log-list li:last-child { border: none; }
  .tag { padding: 2px 7px; border-radius: 4px; font-size: .7rem; font-weight: 600; }
  .tag.GRANTED { background: #14532d; color: var(--green); }
  .tag.DENIED  { background: #450a0a; color: var(--red); }
  .tag.LIVENESS_FAIL { background: #422006; color: var(--yellow); }
  .tag.UNKNOWN_RFID  { background: #1e1b4b; color: #a5b4fc; }

  .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .stat { background: var(--bg); border-radius: 8px; padding: 10px 12px; }
  .stat .val { font-size: 1.5rem; font-weight: 700; }
  .stat .lbl { font-size: .7rem; color: var(--muted); }

  footer { padding: 8px 20px; background: var(--card); border-top: 1px solid var(--border);
           font-size: .7rem; color: var(--muted); text-align: center; }
</style>
</head>
<body>

<header>
  <div class="dot" id="dot"></div>
  <h1>🔐 Access Control Monitor</h1>
  <span style="margin-left:auto;font-size:.75rem;color:var(--muted)" id="clock"></span>
</header>

<main>
  <!-- Kamera -->
  <div class="cam-wrap">
    <img id="stream" src="/stream" alt="Camera stream">
  </div>

  <!-- Sidebar -->
  <div class="sidebar">

    <!-- Status kartu -->
    <div class="card">
      <h2>Status Saat Ini</h2>
      <span id="step-badge">Idle</span>
      <div id="step-text">Menunggu…</div>
      <div id="user-name" style="display:none"></div>
      <div id="similarity-bar" style="display:none">
        <div id="similarity-fill"></div>
      </div>
      <div id="similarity-label"></div>
    </div>

    <!-- Statistik -->
    <div class="card">
      <h2>Statistik</h2>
      <div class="stats-grid">
        <div class="stat"><div class="val" id="s-users">–</div><div class="lbl">Pengguna</div></div>
        <div class="stat"><div class="val" id="s-total">–</div><div class="lbl">Total Akses</div></div>
        <div class="stat"><div class="val" id="s-granted" style="color:var(--green)">–</div><div class="lbl">Diberikan</div></div>
        <div class="stat"><div class="val" id="s-denied"  style="color:var(--red)">–</div><div class="lbl">Ditolak</div></div>
      </div>
    </div>

    <!-- Log terakhir -->
    <div class="card" style="flex:1">
      <h2>Log Terakhir</h2>
      <ul class="log-list" id="log-list"></ul>
    </div>

  </div>
</main>

<footer>Raspberry Pi Access Control &mdash; AES-128-GCM Encrypted &mdash; BlazeFace + MobileFaceNet</footer>

<script>
const STEP_MAP = {
  idle    : ["Menunggu RFID", ""],
  rfid    : ["Tap Kartu RFID", ""],
  liveness: ["Liveness Detection", "liveness"],
  verify  : ["Verifikasi Wajah", ""],
  granted : ["Akses Diberikan ✅", "granted"],
  denied  : ["Akses Ditolak ❌", "denied"],
};

async function refreshState() {
  try {
    const s = await fetch("/api/state").then(r=>r.json());
    const badge = document.getElementById("step-badge");
    const [label, cls] = STEP_MAP[s.step_code] || [s.step, ""];
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

    const barEl  = document.getElementById("similarity-bar");
    const fillEl = document.getElementById("similarity-fill");
    const lblEl  = document.getElementById("similarity-label");
    if (s.similarity !== null && s.similarity !== undefined) {
      barEl.style.display = "block";
      const pct = Math.round(s.similarity * 100);
      fillEl.style.width = pct + "%";
      fillEl.style.background = pct >= 72 ? "var(--green)" : pct >= 55 ? "var(--yellow)" : "var(--red)";
      lblEl.textContent = `Kemiripan: ${pct}%`;
    } else {
      barEl.style.display = "none";
      lblEl.textContent = "";
    }
  } catch(e) {}
}

async function refreshLogs() {
  try {
    const logs = await fetch("/api/logs").then(r=>r.json());
    const ul = document.getElementById("log-list");
    ul.innerHTML = logs.map(l => `
      <li>
        <span>${l.user_name || l.rfid_uid}</span>
        <span class="tag ${l.result}">${l.result}</span>
      </li>`).join("");
  } catch(e) {}
}

async function refreshStats() {
  try {
    const s = await fetch("/api/stats").then(r=>r.json());
    document.getElementById("s-users"  ).textContent = s.total_users;
    document.getElementById("s-total"  ).textContent = s.total_log;
    document.getElementById("s-granted").textContent = s.granted;
    document.getElementById("s-denied" ).textContent = (s.denied_face || 0) + (s.denied_rfid || 0);
  } catch(e) {}
}

function tick() {
  const now = new Date();
  document.getElementById("clock").textContent =
    now.toLocaleTimeString("id-ID");
}

// Polling intervals
setInterval(refreshState, 800);
setInterval(refreshLogs,  3000);
setInterval(refreshStats, 5000);
setInterval(tick, 1000);

// Initial load
refreshState(); refreshLogs(); refreshStats(); tick();
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

# ─────────────────────────────────────────────
#  web/app.py  –  Flask server untuk:
#    1. MJPEG stream kamera real-time (/stream)
#    2. Halaman status HDMI (/  atau /display)
#    3. JSON state endpoint (/api/state)
#  Hemat memori: tidak ada cv2.imshow,
#  frame dikirim langsung ke browser via SSE/MJPEG
# ─────────────────────────────────────────────
import time
import json
import threading
import logging
from flask import Flask, Response, jsonify, render_template_string

import config
from core.camera_stream import camera
from core.database import get_recent_logs, get_stats

log = logging.getLogger(__name__)

app = Flask(__name__)

# ── Shared state (ditulis dari menus/access.py) ───────────────────────────────
_state_lock = threading.Lock()
_state = {
    "step"       : "Menunggu kartu RFID…",
    "step_code"  : "idle",        # idle | rfid | liveness | verify | granted | denied
    "user_name"  : "",
    "similarity" : None,
    "message"    : "",
    "ts"         : 0,
}


def update_state(step: str, step_code: str = "idle",
                 user_name: str = "", similarity: float | None = None,
                 message: str = ""):
    """Dipanggil dari menus/*.py untuk update tampilan HDMI."""
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
    """
    Generator MJPEG – satu thread per client.
    Throttle ke STREAM_MAX_FPS untuk hemat CPU.
    """
    interval = 1.0 / config.STREAM_MAX_FPS
    while True:
        t0    = time.time()
        jpeg  = camera.get_jpeg_bytes()
        if jpeg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg +
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


# ── JSON state ────────────────────────────────────────────────────────────────

@app.route("/api/state")
def api_state():
    return jsonify(get_state())


@app.route("/api/logs")
def api_logs():
    return jsonify(get_recent_logs(20))


@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())


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
    document.getElementById("s-total"  ).textContent = s.total_access;
    document.getElementById("s-granted").textContent = s.granted;
    document.getElementById("s-denied" ).textContent = s.denied;
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


def run_web(host=None, port=None):
    """Jalankan Flask di thread background (non-blocking)."""
    h = host or config.WEB_HOST
    p = port or config.WEB_PORT
    # Gunakan werkzeug threaded=True, proses tunggal supaya hemat RAM
    t = threading.Thread(
        target=lambda: app.run(host=h, port=p, threaded=True,
                               use_reloader=False, debug=False),
        daemon=True,
        name="FlaskThread"
    )
    t.start()
    log.info("Web display: http://%s:%d", h, p)
    return t

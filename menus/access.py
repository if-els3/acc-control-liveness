"""
=============================================================
menus/access.py — Menu Kontrol Akses (RFID + Liveness + Face)
=============================================================
Alur akses lengkap:
  1. Tunggu tap RFID
  2. Cek UID di database         → DENIED_RFID
  3. Kumpulkan frame 3 detik
  4. Liveness Detection          → DENIED_SPOOF
  5. Face Recognition (voting)   → DENIED_FACE
  6. GRANTED → buka pintu → log
=============================================================
"""
import time, os, sys, logging, json
log = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from core.database      import Database
from core.rfid_reader   import RFIDReader
from core.camera_stream import CameraStream
from core.face_engine   import FaceEngine
from core.liveness      import LivenessDetector, BlinkDetector
from core.servo         import DoorController

try:
    import urllib.request
    _HTTP_OK = True
except Exception:
    _HTTP_OK = False
    urllib = None

SEP  = "─" * 58
SEP2 = "═" * 58
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"; NC = "\033[0m"

def _ok(msg):   print(f"\n  {G}✔ {msg}{NC}")
def _fail(msg): print(f"\n  {R}✘ {msg}{NC}")
def _info(msg): print(f"  {Y}→ {msg}{NC}")

def _http_post_state(host, port, **kwargs):
    if not _HTTP_OK or urllib is None:
        return
    try:
        url = f"http://{host}:{port}/api/state"
        data = json.dumps(kwargs).encode()
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        urllib.request.urlopen(req, timeout=0.5)
    except Exception as e:
        log.debug(f"HTTP state update failed: {e}")

def _make_http_callback(host="localhost", port=None):
    if port is None:
        port = getattr(config, 'WEB_PORT', 5000)
    def callback(step, step_code="idle", user_name="", similarity=None,
                 message="", blinks=None, liveness_status=None):
        _http_post_state(host, port, step=step, step_code=step_code,
                         user_name=user_name, similarity=similarity,
                         message=message, blinks=blinks,
                         liveness_status=liveness_status)
    return callback

def _rt_overlay(host, port, **kwargs):
    if not _HTTP_OK or urllib is None:
        return
    try:
        url = f"http://{host}:{port}/api/rt-overlay"
        data = json.dumps(kwargs).encode()
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        urllib.request.urlopen(req, timeout=0.5)
    except Exception as e:
        log.debug(f"RT overlay update failed: {e}")

def _banner(mode=""):
    print(f"\n{SEP2}")
    print(f"  {config.APP_NAME.upper()}")
    lv = f"{G}ON{NC}" if config.LIVENESS_ENABLED else f"{Y}OFF{NC}"
    print(f"  Liveness: {lv}  |  FR: Aktif  |  {mode}")
    print(f"  Ctrl+C untuk berhenti")
    print(SEP2)

def _proses_akses(uid_str, db, face_engine, liveness, door, cam, state_callback=None):
    """Proses satu siklus akses. Return status string."""
    t_rfid = time.perf_counter()
    t_face_detect = None

    def _report_timing(waktu2_override=None):
        waktu1 = time.perf_counter() - t_rfid
        if waktu2_override is not None:
            waktu2 = waktu2_override
        else:
            waktu2 = (time.perf_counter() - t_face_detect) if t_face_detect is not None else None
        if waktu2 is not None:
            print(f"  Waktu      : waktu1={waktu1:.2f}s | waktu2={waktu2:.2f}s")
        else:
            print(f"  Waktu      : waktu1={waktu1:.2f}s | waktu2=tidak terukur")
        return waktu1, waktu2

    user = db.get_user_by_rfid(uid_str)
    if user is None:
        _fail(f"DITOLAK — Kartu tidak terdaftar ({uid_str})")
        db.catat_log(uid_str, "DENIED_RFID", "UID tidak ada di database")
        return "DENIED_RFID"

    nama = user['nama']
    print(f"\n  {B}Kartu{NC}   : {uid_str}")
    print(f"  {B}Nama{NC}    : {nama}")

    # Update state with user info
    if state_callback:
        state_callback(
            step=f"Kartu terdeteksi: {nama}",
            step_code="rfid",
            user_name=nama,
            similarity=None,
                message="Silakan hadapkan wajah ke kamera"
        )

    stored_embs = db.get_embeddings(uid_str)
    if not stored_embs:
        _ok(f"DITERIMA (RFID only) — {nama}")
        db.catat_log(uid_str, "GRANTED", "Tidak ada data wajah — RFID only",
                     user_id=user['id'], nama=nama)
        door.open(duration=config.DOOR_OPEN_SEC)
        waktu1, waktu2 = _report_timing()
        # Update state for granted access
        if state_callback:
            state_callback(
                step="Akses Diberikan (RFID only)",
                step_code="granted",
                user_name=nama,
                similarity=None,
                message=(
                    "Akses diberikan berdasarkan RFID saja. "
                    f"waktu1={waktu1:.2f}s"
                    + (f", waktu2={waktu2:.2f}s" if waktu2 is not None else ", waktu2=tidak terukur")
                )
            )
        return "GRANTED"

    # --- TAHAP 1: LIVENESS (BLINK PRIORITIZED) ---
    live_blinks = 0
    liveness_frames = []
    face_box = None
    lv_status = ""
    liveness_score = None
    liveness_votes = None
    liveness_total = None

    if config.LIVENESS_ENABLED:
        _info("Silakan BERKEDIP beberapa kali untuk verifikasi liveness...")
        if state_callback:
            state_callback(
                step="Silakan BERKEDIP",
                step_code="liveness",
                user_name=nama,
                similarity=None,
                blinks=0,
                liveness_status="",
                message="Identitas terdeteksi. Sekarang liveness diproses dulu..."
            )
        _rt_overlay(getattr(config, 'WEB_HOST', 'localhost'),
                    getattr(config, 'WEB_PORT', 5000),
                    active=True)

        blink_detector = BlinkDetector()
        t0 = time.time()
        # Kumpulkan frame selama durasi liveness + real-time blink tracking
        while time.time() - t0 < config.LIVENESS_DURATION:
            frame = cam.read()
            if frame is None:
                time.sleep(0.05)
                continue
            box = face_engine.detect_largest(frame)
            if box is not None:
                if face_box is None:
                    face_box = box[:4]
                    t_face_detect = time.perf_counter()
                liveness_frames.append(frame)
                crop = liveness._crop_face(frame, box[:4])
                if crop.size > 0:
                    blink_detector.update(crop)
                    live_blinks = blink_detector.blink_count
                    if state_callback:
                        state_callback(
                            step="Silakan BERKEDIP",
                            step_code="liveness",
                            user_name=nama,
                            similarity=None,
                            blinks=live_blinks,
                            liveness_status="Mengecek...",
                            message=f"Kedipan terdeteksi: {live_blinks}"
                        )
                    _rt_overlay(getattr(config, 'WEB_HOST', 'localhost'),
                                getattr(config, 'WEB_PORT', 5000),
                                blinks=live_blinks,
                                liveness_status="Cek..."
                            )
            time.sleep(0.08)

        if not liveness_frames or face_box is None:
            _fail("GAGAL — Wajah hilang saat deteksi liveness")
            db.catat_log(uid_str, "ERROR", "Wajah hilang saat liveness",
                         user_id=user['id'], nama=nama)
            _rt_overlay(getattr(config, 'WEB_HOST', 'localhost'),
                        getattr(config, 'WEB_PORT', 5000),
                        active=False)
            if state_callback:
                state_callback(
                    step="Wajah Hilang",
                    step_code="error",
                    user_name=nama,
                    similarity=None,
                    blinks=live_blinks,
                    liveness_status="",
                    message="Wajah tidak terdeteksi saat proses liveness"
                )
            return "ERROR"

        res = liveness.check(liveness_frames, face_box)
        td = res.detail
        lv_status = "LIVE" if res.is_live else "SPOOF"
        liveness_score = res.score
        liveness_votes = res.votes
        liveness_total = res.total
        blinks = td.get('blinks', live_blinks)

        print(f"  Liveness  : score={res.score:.2f} votes={res.votes}/{res.total}"
              f"  [blk={td.get('blink_score',0):.2f} blinks={blinks}]")

        waktu2_liveness = (time.perf_counter() - t_face_detect) if t_face_detect is not None else None
        waktu1, waktu2 = _report_timing(waktu2_liveness)

        blink_score = td.get('blink_score', 0)
        cascade_ok = td.get('blink') != 'cascade_unavailable'

        if cascade_ok and blinks == 0 and blink_score < 0.5:
            _fail("DITOLAK — Tidak ada kedipan terdeteksi!")
            db.catat_log(uid_str, "DENIED_SPOOF", "Gagal: Tidak ada kedipan",
                         user_id=user['id'], nama=nama)
            _rt_overlay(getattr(config, 'WEB_HOST', 'localhost'),
                        getattr(config, 'WEB_PORT', 5000),
                        active=False, liveness_status=lv_status)
            if state_callback:
                state_callback(
                    step="Tidak Ada Kedipan",
                    step_code="denied",
                    user_name=nama,
                    similarity=None,
                    blinks=blinks,
                    liveness_status=lv_status,
                    message="Akses ditolak karena tidak ada kedipan terdeteksi"
                )
            return "DENIED_SPOOF"

        if not res.is_live:
            _fail(f"DITOLAK — Liveness gagal (score={res.score:.2f})")
            db.catat_log(uid_str, "DENIED_SPOOF", f"Liveness score {res.score:.2f} rendah",
                         user_id=user['id'], nama=nama)
            _rt_overlay(getattr(config, 'WEB_HOST', 'localhost'),
                        getattr(config, 'WEB_PORT', 5000),
                        active=False, liveness_status=lv_status)
            if state_callback:
                state_callback(
                    step="Liveness Gagal",
                    step_code="denied",
                    user_name=nama,
                    similarity=None,
                    blinks=blinks,
                    liveness_status=lv_status,
                    message=f"Liveness terdeteksi sebagai spoof (score={res.score:.2f})"
                )
            return "DENIED_SPOOF"

        if state_callback:
            state_callback(
                step="Liveness Lolos",
                step_code="liveness_pass",
                user_name=nama,
                similarity=None,
                blinks=blinks,
                liveness_status=lv_status,
                message=(
                    f"Liveness sukses. waktu2={waktu2:.2f}s, lanjut verifikasi wajah"
                )
            )
        _rt_overlay(getattr(config, 'WEB_HOST', 'localhost'),
                    getattr(config, 'WEB_PORT', 5000),
                    active=False, liveness_status=lv_status)
    else:
        waktu2 = None
        if state_callback:
            state_callback(
                step="Liveness Dinonaktifkan",
                step_code="liveness_skip",
                user_name=nama,
                similarity=None,
                blinks=0,
                liveness_status="",
                message="Liveness dinonaktifkan, lanjut verifikasi wajah"
            )

    # --- TAHAP 2: VERIFIKASI WAJAH ---
    _info("Memverifikasi identitas ...")
    if state_callback:
        state_callback(
            step="Verifikasi Wajah",
            step_code="verify",
            user_name=nama,
            similarity=None,
            blinks=live_blinks,
            liveness_status=lv_status,
            message="Mencocokkan wajah dengan database..."
        )

    if config.LIVENESS_ENABLED:
        verify_frames = list(liveness_frames)
    else:
        verify_frames = []
        t0 = time.time()
        target_v = config.ENROLL_FRAMES
        while len(verify_frames) < target_v and time.time() - t0 < 3.0:
            frame = cam.read()
            if frame is None:
                time.sleep(0.05)
                continue
            if face_engine.detect_largest(frame) is not None:
                if t_face_detect is None:
                    t_face_detect = time.perf_counter()
                verify_frames.append(frame)
            time.sleep(0.1)

        if not verify_frames:
            _fail("GAGAL — Wajah tidak terdeteksi untuk verifikasi")
            db.catat_log(uid_str, "ERROR", "Wajah tidak terdeteksi saat verifikasi",
                         user_id=user['id'], nama=nama)
            if state_callback:
                state_callback(
                    step="Wajah tidak terdeteksi",
                    step_code="error",
                    user_name=nama,
                    similarity=None,
                    blinks=live_blinks,
                    liveness_status=lv_status,
                    message="Gagal mendeteksi wajah untuk verifikasi identitas"
                )
            return "ERROR"

    def _sim_callback(sc):
        if state_callback:
            state_callback(
                step="Verifikasi Wajah",
                step_code="verify",
                user_name=nama,
                similarity=sc,
                blinks=live_blinks,
                liveness_status=lv_status,
                message=f"Similarity: {sc*100:.1f}% (threshold: {config.FACE_MATCH_THRESH*100:.0f}%)"
            )
        _rt_overlay(getattr(config, 'WEB_HOST', 'localhost'),
                    getattr(config, 'WEB_PORT', 5000),
                    similarity=sc, active=True)

    match, score = face_engine.verify_multi_frame(verify_frames, stored_embs, min_votes=2, callback=_sim_callback)
    pct = score * 100
    thr = config.FACE_MATCH_THRESH * 100

    if not match:
        _fail(f"AKSES DITOLAK — Wajah tidak cocok ({pct:.1f}% < {thr:.0f}%)")
        db.catat_log(uid_str, "DENIED_FACE", f"Face {pct:.1f}% < {thr:.0f}%",
                     user_id=user['id'], nama=nama)
        if state_callback:
            state_callback(
                step=f"Wajah tidak cocok ({pct:.1f}%)",
                step_code="denied",
                user_name=nama,
                similarity=score,
                blinks=live_blinks,
                liveness_status=lv_status,
                message=f"Identitas tidak terverifikasi (Similarity {pct:.1f}%)"
            )
        return "DENIED_FACE"

    _ok(f"Identitas Terverifikasi ({pct:.1f}%)")

    _ok(f"AKSES DITERIMA — {nama} ({pct:.1f}%)")
    db.catat_log(uid_str, "GRANTED", f"Face {pct:.1f}%, Liveness OK ({live_blinks} blink)",
                 user_id=user['id'], nama=nama)
    door.open(duration=config.DOOR_OPEN_SEC)
    _rt_overlay(getattr(config, 'WEB_HOST', 'localhost'),
                getattr(config, 'WEB_PORT', 5000),
                active=False, liveness_status=lv_status)

    waktu1, _ = _report_timing(waktu2)

    if state_callback:
        state_callback(
            step="Akses Diberikan",
            step_code="granted",
            user_name=nama,
            similarity=score,
            blinks=live_blinks,
            liveness_status=lv_status,
            message=(
                f"Verifikasi sukses! {live_blinks} kedipan terdeteksi. "
                f"waktu1={waktu1:.2f}s"
                + (f", waktu2={waktu2:.2f}s" if waktu2 is not None else ", waktu2=tidak terukur")
            )
        )
    return "GRANTED"


def mode_akses_normal(db, face_engine, door, single_attempt=False, state_callback=None):
    liveness = LivenessDetector()
    cam = CameraStream()
    if not cam.start():
        print(f"  {R}[!] Kamera tidak bisa dibuka.{NC}")
        input("  Tekan Enter ..."); return
    rfid = RFIDReader()
    rfid.start()
    # Build HTTP callback if none provided
    if state_callback is None:
        state_callback = _make_http_callback()
    try:
        while True:
            _banner("Menunggu kartu RFID ...")
            print(f"\n  Tempelkan kartu RFID ...\n")
            t_scan = time.perf_counter()
            uid, _ = rfid.scan(timeout=60)
            if uid is None:
                continue
            print(f"  Tap RFID   : {time.perf_counter() - t_scan:.2f}s")
            status = _proses_akses(str(uid), db, face_engine, liveness, door, cam, state_callback)
            if "DENIED" in status or status == "ERROR":
                print(f"\n  {Y}Coba lagi dalam 3 detik ...{NC}")
                time.sleep(3)
            if single_attempt:
                break
    except KeyboardInterrupt:
        print(f"\n\n  Mode akses dihentikan.")
    finally:
        rfid.stop()
        cam.stop()
    input("\n  Tekan Enter untuk kembali ke menu ...")


def menu_akses_sekali(db, face_engine, door):
    print(f"\n{SEP2}\n  UJI AKSES SEKALI\n  Liveness: {'Aktif' if config.LIVENESS_ENABLED else 'Non-aktif'}\n{SEP2}")
    mode_akses_normal(db, face_engine, door, single_attempt=True)


def menu_akses_kontinu(db, face_engine, door):
    print(f"\n{SEP2}\n  MODE OPERASIONAL — Akses Kontinu\n  Ctrl+C untuk kembali\n{SEP2}")
    mode_akses_normal(db, face_engine, door, single_attempt=False)


def menu_toggle_liveness():
    config.LIVENESS_ENABLED = not config.LIVENESS_ENABLED
    print(f"\n  Liveness: {'AKTIF' if config.LIVENESS_ENABLED else 'NON-AKTIF'}")
    print("  (edit config.py untuk permanen)")
    input("  Tekan Enter ...")


def menu_uji_liveness(face_engine):
    """Test liveness tanpa RFID — berguna untuk tuning threshold."""
    print(f"\n{SEP2}\n  UJI LIVENESS DETECTION\n{SEP2}")
    print(f"\n  Hadapkan wajah ke kamera dan BERKEDIPLAH selama {config.LIVENESS_DURATION:.0f}s")
    input("  Tekan Enter untuk mulai ...")

    liveness = LivenessDetector()
    cam = CameraStream()
    if not cam.start():
        print(f"  {R}Kamera error.{NC}"); input("  Enter ..."); return

    try:
        frames = []; face_box = None; t0 = time.time()
        print("  Mengumpulkan frame", end="", flush=True)
        while time.time() - t0 < config.LIVENESS_DURATION:
            frame = cam.read()
            if frame is not None:
                box = face_engine.detect_largest(frame)
                if box:
                    if face_box is None:
                        face_box = box[:4]
                    frames.append(frame)
                    print(".", end="", flush=True)
            time.sleep(0.1)
        print(f" ({len(frames)} frame)\n")

        if not frames or not face_box:
            print(f"  {R}Wajah tidak terdeteksi.{NC}"); input("  Enter ..."); return

        res = liveness.check(frames, face_box)
        status_str = f"{G}LIVE{NC}" if res.is_live else f"{R}SPOOF{NC}"
        print(f"  {SEP}")
        print(f"  Hasil       : {status_str}")
        print(f"  Score       : {res.score:.3f}  (threshold={config.LIVENESS_MIN_SCORE})")
        print(f"  Votes       : {res.votes}/{res.total}")
        print(f"  {SEP}")
        for k, v in res.detail.items():
            print(f"  {k:<22}: {v}")
        print(f"  {SEP}")
    finally:
        cam.stop()

    input("\n  Tekan Enter ...")

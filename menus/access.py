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
import time, os, sys, logging
log = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from core.database      import Database
from core.rfid_reader   import RFIDReader
from core.camera_stream import CameraStream
from core.face_engine   import FaceEngine
from core.liveness      import LivenessDetector
from core.servo         import DoorController

SEP  = "─" * 58
SEP2 = "═" * 58
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"; NC = "\033[0m"

def _ok(msg):   print(f"\n  {G}✔ {msg}{NC}")
def _fail(msg): print(f"\n  {R}✘ {msg}{NC}")
def _info(msg): print(f"  {Y}→ {msg}{NC}")

def _banner(mode=""):
    print(f"\n{SEP2}")
    print(f"  {config.APP_NAME.upper()}")
    lv = f"{G}ON{NC}" if config.LIVENESS_ENABLED else f"{Y}OFF{NC}"
    print(f"  Liveness: {lv}  |  FR: Aktif  |  {mode}")
    print(f"  Ctrl+C untuk berhenti")
    print(SEP2)

def _proses_akses(uid_str, db, face_engine, liveness, door, cam, state_callback=None):
    """Proses satu siklus akses. Return status string."""
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
        # Update state for granted access
        if state_callback:
            state_callback(
                step="Akses Diberikan (RFID only)",
                step_code="granted",
                user_name=nama,
                similarity=None,
                message="Akses diberikan berdasarkan RFID saja"
            )
        return "GRANTED"

    # Kumpulkan frame
    _info(f"Hadapkan wajah ke kamera ... ({config.LIVENESS_DURATION:.0f}s)")
    # Update state for face verification
    if state_callback:
        state_callback(
            step="Hadapkan wajah ke kamera",
            step_code="verify",
            user_name=nama,
            similarity=None,
                message=f"Mengumpulkan {config.ENROLL_FRAMES if hasattr(config, 'ENROLL_FRAMES') else 5} frame..."
        )
    frames = []; face_box = None
    t0 = time.time()
    while time.time() - t0 < config.LIVENESS_DURATION:
        frame = cam.read()
        if frame is None:
            time.sleep(0.05); continue
        box = face_engine.detect_largest(frame)
        if box is not None:
            if face_box is None:
                face_box = box[:4]
            frames.append(frame)
        time.sleep(0.08)

    if not frames or face_box is None:
        _fail("GAGAL — Wajah tidak terdeteksi")
        db.catat_log(uid_str, "ERROR", "Tidak ada frame wajah",
                     user_id=user['id'], nama=nama)
        # Update state for error
        if state_callback:
            state_callback(
                step="Wajah tidak terdeteksi",
                step_code="error",
                user_name=nama,
                similarity=None,
                message="Tidak ada wajah yang terdeteksi selama pemindaian"
            )
        return "ERROR"

    print(f"  ({len(frames)} frame dikumpulkan)")
    # Update state after frame collection
    if state_callback:
        state_callback(
            step=f"{len(frames)} frame dikumpulkan",
            step_code="verify",
            user_name=nama,
            similarity=None,
                message="Memulai pemrosesan liveness dan wajah..."
        )

    # Liveness
    if config.LIVENESS_ENABLED:
        _info("Memeriksa liveness ...")
        # Update state for liveness check
        if state_callback:
            state_callback(
                step="Memeriksa liveness",
                step_code="liveness",
                user_name=nama,
                similarity=None,
                message="Menganalisis gerakan, tekstur, dan kedipan mata..."
            )
        res = liveness.check(frames, face_box)
        td = res.detail
        print(f"  Liveness  : score={res.score:.2f} votes={res.votes}/{res.total}"
              f"  [tex={td.get('texture_score',0):.2f}"
              f" mot={td.get('motion_score',0):.2f}"
              f" blk={td.get('blink_score',0):.2f}"
              f" blinks={td.get('blinks',0)}]")
        if not res.is_live:
            _fail(f"DITOLAK — Liveness gagal (score={res.score:.2f} < {config.LIVENESS_MIN_SCORE})")
            db.catat_log(uid_str, "DENIED_SPOOF",
                         f"Liveness score={res.score:.2f} votes={res.votes}/3",
                         user_id=user['id'], nama=nama)
            # Update state for liveness failure
            if state_callback:
                state_callback(
                    step=f"Liveness gagal (score={res.score:.2f})",
                    step_code="denied",
                    user_name=nama,
                    similarity=None,
                    message=f"Liveness terdeteksi sebagai spoof dengan score {res.score:.2f}"
                )
            return "DENIED_SPOOF"
        _ok(f"Liveness OK ({res.score:.2f})")
        # Update state for liveness success
        if state_callback:
            state_callback(
                step="Liveness OK",
                step_code="liveness",
                user_name=nama,
                similarity=None,
                message=f"Liveness terdeteksi sebagai hidup dengan score {res.score:.2f}"
            )

    # Face recognition
    _info("Memverifikasi wajah ...")
    # Update state for face verification
    if state_callback:
        state_callback(
            step="Memverifikasi wajah",
            step_code="verify",
            user_name=nama,
            similarity=None,
                message="Membandingkan wajah dengan data terdaftar..."
        )
    match, score = face_engine.verify_multi_frame(frames, stored_embs, min_votes=2)
    pct = score * 100
    thr = config.FACE_MATCH_THRESH * 100
    print(f"  Face score : {pct:.1f}%  (threshold {thr:.0f}%)")
    # Update state with similarity score
    if state_callback:
        state_callback(
            step=f"Face score: {pct:.1f}%",
            step_code="verify",
            user_name=nama,
            similarity=score,
                message=f"Similarity: {pct:.1f}% (threshold: {thr:.0f}%)"
        )

    if match:
        _ok(f"AKSES DITERIMA — {nama}  ({pct:.1f}%)")
        db.catat_log(uid_str, "GRANTED",
                     f"Face {pct:.1f}%, Liveness OK",
                     user_id=user['id'], nama=nama)
        door.open(duration=config.DOOR_OPEN_SEC)
        # Update state for granted access
        if state_callback:
            state_callback(
                step="Akses Diberikan",
                step_code="granted",
                user_name=nama,
                similarity=score,
                message=f"Wajah terverifikasi dengan similarity {pct:.1f}%"
            )
        return "GRANTED"
    else:
        _fail(f"AKSES DITOLAK — Wajah tidak cocok ({pct:.1f}% < {thr:.0f}%)")
        db.catat_log(uid_str, "DENIED_FACE",
                     f"Face {pct:.1f}% < {thr:.0f}%",
                     user_id=user['id'], nama=nama)
        # Update state for denied access
        if state_callback:
            state_callback(
                step=f"Wajah tidak cocok ({pct:.1f}%)",
                step_code="denied",
                user_name=nama,
                similarity=score,
                message=f"Similarity {pct:.1f}% di bawah threshold {thr:.0f}%"
            )
        return "DENIED_FACE"


def mode_akses_normal(db, face_engine, door, single_attempt=False):
    liveness = LivenessDetector()
    cam = CameraStream()
    if not cam.start():
        print(f"  {R}[!] Kamera tidak bisa dibuka.{NC}")
        input("  Tekan Enter ..."); return
    rfid = RFIDReader()
    rfid.start()
    try:
        while True:
            _banner("Menunggu kartu RFID ...")
            print(f"\n  Tempelkan kartu RFID ...\n")
            uid, _ = rfid.scan(timeout=60)
            if uid is None:
                continue
            status = _proses_akses(str(uid), db, face_engine, liveness, door, cam)
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
    print(f"\n  Hadapkan wajah / tempel foto ke kamera selama {config.LIVENESS_DURATION:.0f}s")
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

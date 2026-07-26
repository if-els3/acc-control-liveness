"""
=============================================================
menus/enrollment.py — Menu Pendaftaran Pengguna Baru
=============================================================
Alur:
  1. Scan RFID (tap kartu)
  2. Cek UID belum terdaftar
  3. Input nama pengguna
  4. Capture 5 frame wajah
  5. Ekstrak embedding tiap frame
  6. Simpan ke database
=============================================================
"""
import time
import os
import sys
import logging

log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from core.database      import Database
from core.rfid_reader   import RFIDReader
from core.camera_stream import CameraStream
from core.face_engine   import FaceEngine
from menus.admin        import login_admin

SEP  = "─" * 56
SEP2 = "═" * 56


def _header(title: str):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)


def menu_daftar_pengguna(db: Database, face_engine: FaceEngine):
    """Daftarkan pengguna baru: Admin login, Input ID+Nama, RFID, wajah."""
    _header("PENDAFTARAN PENGGUNA BARU")

    # ── STEP 1: Login Admin ───────────────────────────────
    if not login_admin():
        input("  Tekan Enter untuk kembali ...")
        return

    # ── STEP 2: Input ID dan Nama ─────────────────────────
    print(f"\n  [1/4] Data Pengguna")
    try:
        user_id_input = input("        ID Pengguna  : ").strip()
        user_id_int = int(user_id_input)
    except ValueError:
        print("  [!] ID Pengguna harus berupa angka valid.")
        input("  Tekan Enter untuk kembali ...")
        return

    with db._conn() as conn:
        row = conn.execute("SELECT id FROM users WHERE id=?", (user_id_int,)).fetchone()
        if row:
            print(f"  [!] ID {user_id_int} sudah digunakan.")
            input("  Tekan Enter untuk kembali ...")
            return

    nama = input("        Nama lengkap : ").strip()
    if not nama:
        print("  [!] Nama tidak boleh kosong.")
        input("  Tekan Enter untuk kembali ...")
        return

    # ── STEP 3: Scan RFID ─────────────────────────────────
    print(f"\n  [2/4] Scan Kartu RFID")
    print(f"        Tempelkan kartu ke reader ...")
    print(f"        (timeout {config.RFID_TIMEOUT} detik — Ctrl+C untuk batal)\n")

    uid = None
    with RFIDReader() as rfid:
        uid, _ = rfid.scan(timeout=config.RFID_TIMEOUT)

    if uid is None:
        print("\n  [!] Timeout — tidak ada kartu terdeteksi.")
        input("  Tekan Enter untuk kembali ...")
        return

    uid_str = str(uid)
    print(f"\n  ✔ Kartu terdeteksi : UID = {uid_str}")

    # Cek duplikasi RFID
    if db.cek_rfid_terdaftar(uid_str):
        existing = db.get_user_by_rfid(uid_str)
        print(f"\n  [!] UID ini sudah terdaftar atas nama: {existing['nama']}")
        print("      Gunakan menu 'Kelola Pengguna' untuk mengubah data.")
        input("  Tekan Enter untuk kembali ...")
        return

    # Konfirmasi
    print(f"\n  {SEP}")
    print(f"  ID User: {user_id_int}")
    print(f"  Nama   : {nama}")
    print(f"  UID    : {uid_str}")
    print(f"  {SEP}")
    konfirm = input("  Data sudah benar? [y/N] : ").strip().lower()
    if konfirm != 'y':
        print("  Pendaftaran dibatalkan.")
        input("  Tekan Enter untuk kembali ...")
        return

    # ── STEP 4: Capture wajah ─────────────────────────────
    enroll_frames    = int(getattr(config, "ENROLL_FRAMES", 20))
    enroll_min_valid = int(getattr(config, "ENROLL_MIN_VALID", 12))
    capture_delay    = float(getattr(config, "ENROLL_CAPTURE_DELAY", 0.5))
    quality_floor    = float(getattr(config, "ENROLL_QUALITY_FLOOR", 0.0))

    print(f"\n  [3/4] Pendaftaran Wajah")
    print(f"        Hadapkan wajah ke kamera, pencahayaan cukup.")
    input("        Tekan Enter saat siap ...")

    embeddings = []
    cam = CameraStream()

    if not cam.start():
        print("\n  [!] Kamera tidak bisa dibuka.")
        print("      Simpan data tanpa wajah? ", end="")
        if input("[y/N] : ").strip().lower() == 'y':
            user_id = db.tambah_user(nama, uid_str, embeddings=None, user_id=user_id_int)
            print(f"\n  ✔ Pengguna '{nama}' didaftarkan TANPA data wajah (ID={user_id})")
            print("     Wajah dapat ditambahkan dari menu Kelola Pengguna.")
        input("  Tekan Enter untuk kembali ...")
        return

    try:
        print(f"\n  Mengambil foto wajah ", end="", flush=True)
        for _ in range(enroll_frames):
            time.sleep(capture_delay)
            frame = cam.read()
            if frame is None:
                print("x", end="", flush=True)
                continue

            emb = face_engine.extract_embedding(frame)
            if emb is not None:
                embeddings.append(emb.tolist())
                print("✔", end="", flush=True)
            else:
                print("○", end="", flush=True)

        print(f"  ({len(embeddings)}/{enroll_frames} frame berhasil)\n")

    finally:
        cam.stop()

    if len(embeddings) == 0:
        print("  [!] Tidak ada wajah terdeteksi di semua frame.")
        print("      Pastikan pencahayaan cukup dan wajah terlihat jelas.")
        print("      Simpan data tanpa wajah? ", end="")
        if input("[y/N] : ").strip().lower() == 'y':
            db.tambah_user(nama, uid_str, embeddings=None, user_id=user_id_int)
            print(f"  ✔ Pengguna '{nama}' didaftarkan tanpa wajah.")
        input("  Tekan Enter untuk kembali ...")
        return

    # -- Internal quality filter (hidden) ------------------
    import numpy as _np
    if quality_floor > 0.0 and len(embeddings) >= 3:
        emb_arr  = _np.array(embeddings, dtype=_np.float32)
        mean_emb = emb_arr.mean(axis=0)
        norm_m   = mean_emb / (_np.linalg.norm(mean_emb) + 1e-9)
        embeddings = [
            e for e in embeddings
            if float(_np.dot(_np.array(e, dtype=_np.float32), norm_m)) >= quality_floor
        ]

    if len(embeddings) < enroll_min_valid:
        log.warning("Enrollment: hanya %d embedding valid (min=%d)", len(embeddings), enroll_min_valid)
        print(f"  [!] Jumlah data wajah kurang memadai ({len(embeddings)} foto berhasil).")
        print(f"      Coba lagi dengan pencahayaan lebih baik, atau lanjut simpan?")
        if input("  Lanjut simpan? [y/N] : ").strip().lower() != 'y':
            print("  Pendaftaran dibatalkan.")
            input("  Tekan Enter untuk kembali ...")
            return

    # ── STEP 5: Simpan ke DB ──────────────────────────────
    print(f"  [4/4] Menyimpan ke database ...")
    user_id = db.tambah_user(nama, uid_str, embeddings=embeddings, user_id=user_id_int)
    db.catat_log(uid_str, "ENROLL",
                 f"Pendaftaran berhasil ({len(embeddings)} frame)",
                 user_id=user_id, nama=nama)

    print(f"\n{SEP2}")
    print(f"  ✔ PENDAFTARAN BERHASIL")
    print(SEP2)
    print(f"  Nama     : {nama}")
    print(f"  UID RFID : {uid_str}")
    print(f"  ID User  : {user_id}")
    print(f"  Wajah    : {len(embeddings)} embedding tersimpan")
    print(f"  Mode     : {face_engine.mode}")
    print(SEP2)
    input("\n  Tekan Enter untuk kembali ke menu ...")


def menu_update_wajah(db: Database, face_engine: FaceEngine):
    """Update/tambah data wajah pengguna yang sudah terdaftar."""
    _header("UPDATE DATA WAJAH")

    print("\n  Scan kartu pengguna yang ingin diupdate wajahnya ...")
    uid = None
    with RFIDReader() as rfid:
        uid, _ = rfid.scan(timeout=config.RFID_TIMEOUT)

    if uid is None:
        print("  [!] Timeout."); input("  Enter ..."); return

    uid_str = str(uid)
    user = db.get_user_by_rfid(uid_str)
    if user is None:
        print(f"  [!] UID {uid_str} tidak terdaftar.")
        input("  Tekan Enter ..."); return

    enroll_frames    = int(getattr(config, "ENROLL_FRAMES", 20))
    enroll_min_valid = int(getattr(config, "ENROLL_MIN_VALID", 12))
    capture_delay    = float(getattr(config, "ENROLL_CAPTURE_DELAY", 0.5))
    quality_floor    = float(getattr(config, "ENROLL_QUALITY_FLOOR", 0.0))

    print(f"\n  Pengguna : {user['nama']} (ID={user['id']})")
    print(f"  Mode FR  : {face_engine.mode}")
    input(f"  Hadapkan wajah ke kamera. Tekan Enter saat siap ...")

    embeddings = []
    cam = CameraStream()
    if not cam.start():
        print("  [!] Kamera error."); input("  Enter ..."); return

    try:
        print(f"\n  Mengambil foto wajah ", end="", flush=True)
        for _ in range(enroll_frames):
            time.sleep(capture_delay)
            frame = cam.read()
            if frame is None:
                print("x", end="", flush=True); continue
            emb = face_engine.extract_embedding(frame)
            if emb is not None:
                embeddings.append(emb.tolist())
                print("✔", end="", flush=True)
            else:
                print("○", end="", flush=True)
        print(f"  ({len(embeddings)}/{enroll_frames} berhasil)")
    finally:
        cam.stop()

    if len(embeddings) == 0:
        print("  [!] Tidak ada wajah terdeteksi.")
        input("  Enter ..."); return

    # -- Internal quality filter (hidden) ------------------
    import numpy as _np
    if quality_floor > 0.0 and len(embeddings) >= 3:
        emb_arr  = _np.array(embeddings, dtype=_np.float32)
        mean_emb = emb_arr.mean(axis=0)
        norm_m   = mean_emb / (_np.linalg.norm(mean_emb) + 1e-9)
        embeddings = [
            e for e in embeddings
            if float(_np.dot(_np.array(e, dtype=_np.float32), norm_m)) >= quality_floor
        ]

    if len(embeddings) < enroll_min_valid:
        log.warning("Update wajah: hanya %d embedding valid (min=%d)", len(embeddings), enroll_min_valid)
        print(f"  [!] Jumlah data wajah kurang memadai ({len(embeddings)} foto berhasil).")
        if input("  Lanjut simpan? [y/N] : ").strip().lower() != 'y':
            input("  Enter ..."); return

    db.update_embedding(uid_str, embeddings)
    print(f"\n  ✔ Wajah diupdate: {len(embeddings)} embedding tersimpan.")
    input("  Tekan Enter ...")

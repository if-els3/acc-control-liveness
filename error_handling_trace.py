#!/usr/bin/env python3
"""
=============================================================
error_handling_trace.py — Database Error-Handling Trace
=============================================================
Menguji semua jalur error handling di core/database.py:
  1. Inisialisasi DB (path direktori baru, path tanpa direktori)
  2. Tambah user (RFID duplikat)
  3. Enkripsi / dekripsi embedding
  4. Pencatatan log & statistik
  5. Operasi user (nonaktifkan, aktifkan, hapus)

Jalankan: python error_handling_trace.py
=============================================================
"""
import os, sys, logging, tempfile, shutil
import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("trace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results: list[tuple[str, bool]] = []


def _check(label: str, ok: bool, detail: str = ""):
    tag = PASS if ok else FAIL
    msg = f"  [{tag}] {label}"
    if detail:
        msg += f"  — {detail}"
    print(msg)
    _results.append((label, ok))


# ─── 1. Inisialisasi DB ──────────────────────────────────────────────────────

def test_init_with_new_subdirectory():
    """DB_PATH dengan sub-direktori yang belum ada."""
    tmp = tempfile.mkdtemp()
    try:
        from core.database import Database
        db_path = os.path.join(tmp, "subdir", "test.db")
        db = Database(db_path)
        _check("init: new subdirectory created", os.path.exists(db_path))
    except Exception as e:
        _check("init: new subdirectory created", False, str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_init_with_bare_filename():
    """DB_PATH tanpa komponen direktori (hanya nama file)."""
    orig_cwd = os.getcwd()
    tmp = tempfile.mkdtemp()
    try:
        os.chdir(tmp)
        from core.database import Database
        db = Database("bare_test.db")
        _check("init: bare filename (no directory)", os.path.exists(os.path.join(tmp, "bare_test.db")))
    except Exception as e:
        _check("init: bare filename (no directory)", False, str(e))
    finally:
        os.chdir(orig_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_init_default_path():
    """Inisialisasi dengan DB_PATH default dari config."""
    try:
        from core.database import Database
        db = Database()
        _check("init: default config path", True)
        return db
    except Exception as e:
        _check("init: default config path", False, str(e))
        return None


# ─── 2. Operasi User ────────────────────────────────────────────────────────

def test_user_crud(db):
    """Tambah, baca, nonaktifkan, aktifkan, hapus user."""
    TEST_UID = "TRACE_TEST_UID_001"

    # Tambah user baru
    try:
        uid = db.tambah_user("TraceUser", TEST_UID)
        _check("user: tambah user baru", isinstance(uid, int) and uid > 0, f"id={uid}")
    except Exception as e:
        _check("user: tambah user baru", False, str(e))
        return

    # Cek RFID terdaftar
    try:
        registered = db.cek_rfid_terdaftar(TEST_UID)
        _check("user: cek RFID terdaftar", registered)
    except Exception as e:
        _check("user: cek RFID terdaftar", False, str(e))

    # Duplikat RFID harus ditolak
    try:
        db.tambah_user("DupUser", TEST_UID)
        _check("user: tolak RFID duplikat", False, "tidak ada exception")
    except Exception as e:
        _check("user: tolak RFID duplikat", True, type(e).__name__)

    # Get user by RFID
    try:
        row = db.get_user_by_rfid(TEST_UID)
        _check("user: get_user_by_rfid", row is not None and row["nama"] == "TraceUser")
    except Exception as e:
        _check("user: get_user_by_rfid", False, str(e))

    # Nonaktifkan
    try:
        db.nonaktifkan_user(uid)
        row = db.get_user_by_rfid(TEST_UID)  # hanya kembalikan aktif=1
        _check("user: nonaktifkan (tidak muncul di aktif query)", row is None)
    except Exception as e:
        _check("user: nonaktifkan", False, str(e))

    # Aktifkan kembali
    try:
        db.aktifkan_user(uid)
        row = db.get_user_by_rfid(TEST_UID)
        _check("user: aktifkan kembali", row is not None)
    except Exception as e:
        _check("user: aktifkan kembali", False, str(e))

    # Hapus
    try:
        db.hapus_user(uid)
        _check("user: hapus user", not db.cek_rfid_terdaftar(TEST_UID))
    except Exception as e:
        _check("user: hapus user", False, str(e))


# ─── 3. Embedding enkripsi / dekripsi ───────────────────────────────────────

def test_embedding_roundtrip(db):
    """Simpan + ambil embedding — pastikan roundtrip enkripsi OK."""
    TEST_UID = "TRACE_EMB_UID_001"
    emb = [np.random.rand(512).astype(np.float32) for _ in range(3)]

    try:
        uid = db.tambah_user("EmbUser", TEST_UID, embeddings=emb)
        _check("embedding: simpan dengan enkripsi", True, f"id={uid}")
    except Exception as e:
        _check("embedding: simpan dengan enkripsi", False, str(e))
        return

    try:
        loaded = db.get_embeddings(TEST_UID)
        ok = (
            loaded is not None
            and len(loaded) == len(emb)
            and all(np.allclose(a, b, atol=1e-5) for a, b in zip(emb, loaded))
        )
        _check("embedding: roundtrip decrypt cocok", ok)
    except Exception as e:
        _check("embedding: roundtrip decrypt cocok", False, str(e))

    # Update embedding
    try:
        emb2 = [np.random.rand(512).astype(np.float32) for _ in range(2)]
        db.update_embedding(TEST_UID, emb2)
        loaded2 = db.get_embeddings(TEST_UID)
        ok2 = loaded2 is not None and len(loaded2) == 2
        _check("embedding: update embedding", ok2)
    except Exception as e:
        _check("embedding: update embedding", False, str(e))

    # Bersihkan
    row = db.get_user_by_rfid(TEST_UID)
    if row:
        db.hapus_user(row["id"])


def test_embedding_missing(db):
    """get_embeddings untuk user tanpa wajah harus return None."""
    TEST_UID = "TRACE_NOEMB_UID_001"
    try:
        uid = db.tambah_user("NoEmbUser", TEST_UID, embeddings=None)
        result = db.get_embeddings(TEST_UID)
        _check("embedding: user tanpa wajah → None", result is None)
        db.hapus_user(uid)
    except Exception as e:
        _check("embedding: user tanpa wajah → None", False, str(e))


# ─── 4. Logging & Statistik ─────────────────────────────────────────────────

def test_logging_and_stats(db):
    """catat_log dengan berbagai status, lalu validasi statistik."""
    TEST_UID = "TRACE_LOG_UID_001"
    try:
        uid = db.tambah_user("LogUser", TEST_UID)
    except Exception as e:
        _check("log: setup user", False, str(e))
        return

    statuses = ["GRANTED", "DENIED_FACE", "DENIED_RFID", "DENIED_SPOOF", "ERROR", "ENROLL"]
    for s in statuses:
        try:
            db.catat_log(TEST_UID, s, f"trace test {s}", user_id=uid, nama="LogUser")
            _check(f"log: catat_log status={s}", True)
        except Exception as e:
            _check(f"log: catat_log status={s}", False, str(e))

    # catat_log dengan rfid_uid=None
    try:
        db.catat_log(None, "ERROR", "rfid tidak terbaca")
        _check("log: catat_log rfid=None", True)
    except Exception as e:
        _check("log: catat_log rfid=None", False, str(e))

    # get_logs
    try:
        logs = db.get_logs(limit=10)
        _check("log: get_logs", len(logs) > 0, f"{len(logs)} baris")
    except Exception as e:
        _check("log: get_logs", False, str(e))

    # get_logs dengan filter
    try:
        logs_g = db.get_logs(limit=10, filter_status="GRANTED")
        _check("log: get_logs filter_status=GRANTED", all(r["status"] == "GRANTED" for r in logs_g))
    except Exception as e:
        _check("log: get_logs filter_status", False, str(e))

    # statistik
    try:
        stats = db.statistik()
        required = {"total_user", "total_log", "granted", "denied_face", "denied_rfid"}
        _check("log: statistik keys", required.issubset(stats.keys()), str(stats))
    except Exception as e:
        _check("log: statistik", False, str(e))

    # Bersihkan
    db.hapus_user(uid)


# ─── 5. AES-GCM Self-test ────────────────────────────────────────────────────

def test_aes_gcm_selftest():
    try:
        from core.database import run_aes_gcm_selftest
        result = run_aes_gcm_selftest()
        _check("aes-gcm: self-test", result["failed"] == 0,
               f"passed={result['passed']} skipped={result['skipped']}")
    except Exception as e:
        _check("aes-gcm: self-test", False, str(e))


# ─── Ringkasan ───────────────────────────────────────────────────────────────

def _summary():
    total  = len(_results)
    passed = sum(1 for _, ok in _results if ok)
    failed = total - passed
    print()
    print("─" * 58)
    print(f"  Hasil: {passed}/{total} lulus", end="")
    if failed:
        print(f"  ({failed} GAGAL)")
        for label, ok in _results:
            if not ok:
                print(f"    ✘ {label}")
    else:
        print("  ✔ semua OK")
    print("─" * 58)
    return failed == 0


if __name__ == "__main__":
    print("\n" + "═" * 58)
    print("  DATABASE ERROR-HANDLING TRACE")
    print("═" * 58 + "\n")

    test_aes_gcm_selftest()
    print()

    test_init_with_new_subdirectory()
    test_init_with_bare_filename()
    db = test_init_default_path()
    print()

    if db:
        test_user_crud(db)
        print()
        test_embedding_roundtrip(db)
        test_embedding_missing(db)
        print()
        test_logging_and_stats(db)

    ok = _summary()
    sys.exit(0 if ok else 1)

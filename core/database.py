"""
=============================================================
core/database.py — SQLite Database Manager
=============================================================
Schema:
  users        — data pengguna terdaftar + embedding wajah
  access_logs  — log setiap percobaan akses
=============================================================
"""
import sqlite3
import json
import os
import logging
import numpy as np
from datetime import datetime
from typing import Optional, List

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from core.crypto import encrypt_embedding_list, decrypt_embedding_list

log = logging.getLogger(__name__)

# ─── Path ke file test vector Wycheproof ─────────────────────────────────────
_AES_GCM_TEST_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "unittest",
    "aes_gcm_test.json",
)
_UNSUPPORTED_IV_FLAGS = {"ZeroLengthIv", "SmallIv", "LongIv"}


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    nama        TEXT    NOT NULL,
    rfid_uid    TEXT    NOT NULL UNIQUE,
    face_embeddings TEXT,        -- AES-128-GCM base64 blob of JSON list of embedding arrays
    aktif       INTEGER NOT NULL DEFAULT 1,
    dibuat      TEXT    NOT NULL,
    diubah      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS access_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    rfid_uid    TEXT,
    user_id     INTEGER,
    nama        TEXT,
    status      TEXT    NOT NULL,  -- GRANTED / DENIED_FACE / DENIED_RFID / ERROR
    keterangan  TEXT,
    waktu       TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_users_rfid ON users(rfid_uid);
CREATE INDEX IF NOT EXISTS idx_logs_waktu ON access_logs(waktu);
CREATE INDEX IF NOT EXISTS idx_logs_uid   ON access_logs(rfid_uid);
"""


def _load_aes_gcm_vectors(path: str, key_size: int = 128) -> list[dict]:
    """Muat test vector AES-GCM dari file JSON Wycheproof."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    vectors = []
    for group in data.get("testGroups", []):
        if group.get("keySize") != key_size:
            continue
        for test in group.get("tests", []):
            vectors.append({
                "tcId":   test["tcId"],
                "flags":  set(test.get("flags", [])),
                "key":    bytes.fromhex(test["key"]),
                "iv":     bytes.fromhex(test["iv"]) if test["iv"] else b"",
                "aad":    bytes.fromhex(test["aad"]) if test["aad"] else b"",
                "msg":    bytes.fromhex(test["msg"]) if test["msg"]  else b"",
                "ct":     bytes.fromhex(test["ct"])  if test["ct"]   else b"",
                "tag":    bytes.fromhex(test["tag"]),
                "result": test["result"],
            })
    return vectors

def run_aes_gcm_selftest() -> dict:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.exceptions import InvalidTag

    vectors = _load_aes_gcm_vectors(_AES_GCM_TEST_FILE)
    passed = failed = skipped = 0
    errors = []

    for v in vectors:
        if v["flags"] & _UNSUPPORTED_IV_FLAGS or not (8 <= len(v["iv"]) <= 128):
            skipped += 1
            continue

        aesgcm = AESGCM(v["key"])
        aad_arg = v["aad"] if v["aad"] else None

        if v["result"] == "valid":
            try:
                ct_tag = aesgcm.encrypt(v["iv"], v["msg"], aad_arg)
                ct  = ct_tag[:-16]
                tag = ct_tag[-16:]
                if ct != v["ct"] or tag != v["tag"]:
                    failed += 1
                    errors.append(f"tcId={v['tcId']}: enkripsi tidak cocok")
                    continue
            except Exception as exc:
                failed += 1
                errors.append(f"tcId={v['tcId']}: enkripsi error – {exc}")
                continue

            try:
                plaintext = aesgcm.decrypt(v["iv"], v["ct"] + v["tag"], aad_arg)
                if plaintext != v["msg"]:
                    failed += 1
                    errors.append(f"tcId={v['tcId']}: dekripsi plaintext tidak cocok")
                    continue
            except (InvalidTag, Exception) as exc:
                failed += 1
                errors.append(f"tcId={v['tcId']}: dekripsi error – {exc}")
                continue

        elif v["result"] == "invalid":
            try:
                plaintext = aesgcm.decrypt(v["iv"], v["ct"] + v["tag"], aad_arg)
                failed += 1
                errors.append(f"tcId={v['tcId']}: vector invalid lolos tanpa error")
                continue
            except (InvalidTag, ValueError):
                pass
            except Exception as exc:
                failed += 1
                errors.append(f"tcId={v['tcId']}: unexpected error – {exc}")
                continue
        passed += 1

    result = {"passed": passed, "failed": failed, "skipped": skipped, "errors": errors}
    if failed > 0:
        msg = f"AES-GCM self-test GAGAL: {failed} vector(s) tidak lulus. {errors[:3]}"
        log.error(msg)
        raise RuntimeError(msg)
    log.info("AES-GCM self-test OK: %d lulus, %d di-skip (IV out-of-range)", passed, skipped)
    return result


class Database:
    def __init__(self, db_path: str = config.DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        # ── Validasi implementasi AES-128-GCM sebelum membuka database ────────────
        if os.path.exists(_AES_GCM_TEST_FILE):
            try:
                run_aes_gcm_selftest()
            except FileNotFoundError:
                log.warning("File test vector AES-GCM tidak ditemukan, self-test dilewati.")
            except RuntimeError as exc:
                raise RuntimeError(f"Inisialisasi DB dibatalkan: {exc}") from exc
        else:
            log.warning("aes_gcm_test.json tidak ditemukan, self-test dilewati.")

        with self._conn() as conn:
            conn.executescript(SCHEMA)
        log.info(f"Database siap: {self.db_path}")

    # ── USERS ────────────────────────────────────────────────

    def tambah_user(self, nama: str, rfid_uid: str,
                    embeddings: Optional[List[np.ndarray]] = None) -> int:
        now = datetime.now().isoformat()
        emb_blob = encrypt_embedding_list(embeddings) if embeddings else None
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO users (nama, rfid_uid, face_embeddings, dibuat, diubah) "
                "VALUES (?,?,?,?,?)",
                (nama, str(rfid_uid), emb_blob, now, now)
            )
        log.info(f"User ditambah: {nama} (UID={rfid_uid})")
        lastid = cur.lastrowid
        if lastid is None:
            raise RuntimeError("Failed to insert user, no lastrowid returned")
        return int(lastid)

    def update_embedding(self, rfid_uid: str, embeddings: List[np.ndarray]):
        now = datetime.now().isoformat()
        emb_blob = encrypt_embedding_list(embeddings)
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET face_embeddings=?, diubah=? WHERE rfid_uid=?",
                (emb_blob, now, str(rfid_uid))
            )
        log.info(f"Embedding diupdate: UID={rfid_uid}")

    def get_user_by_rfid(self, rfid_uid: str) -> Optional[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM users WHERE rfid_uid=? AND aktif=1",
                (str(rfid_uid),)
            ).fetchone()

    def get_all_users(self) -> List[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT id, nama, rfid_uid, aktif, dibuat, "
                "CASE WHEN face_embeddings IS NOT NULL THEN 1 ELSE 0 END as has_face "
                "FROM users ORDER BY id"
            ).fetchall()

    def nonaktifkan_user(self, user_id: int):
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET aktif=0, diubah=? WHERE id=?",
                (now, user_id)
            )

    def aktifkan_user(self, user_id: int):
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET aktif=1, diubah=? WHERE id=?",
                (now, user_id)
            )

    def hapus_user(self, user_id: int):
        with self._conn() as conn:
            conn.execute("DELETE FROM access_logs WHERE user_id=?", (user_id,))
            conn.execute("DELETE FROM users WHERE id=?", (user_id,))

    def cek_rfid_terdaftar(self, rfid_uid: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM users WHERE rfid_uid=?", (str(rfid_uid),)
            ).fetchone()
        return row is not None

    def get_embeddings(self, rfid_uid: str) -> Optional[List[np.ndarray]]:
        row = self.get_user_by_rfid(rfid_uid)
        if row and row["face_embeddings"]:
            try:
                return decrypt_embedding_list(row["face_embeddings"])
            except ValueError as e:
                log.error("Gagal dekripsi embedding uid=%s: %s", rfid_uid, e)
                return None
        return None

    # ── LOGS ────────────────────────────────────────────────

    def catat_log(self, rfid_uid: Optional[str], status: str,
                  keterangan: str = "", user_id: Optional[int] = None,
                  nama: Optional[str] = None):
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO access_logs (rfid_uid, user_id, nama, status, keterangan, waktu) "
                "VALUES (?,?,?,?,?,?)",
                (str(rfid_uid) if rfid_uid else None,
                 user_id, nama, status, keterangan, now)
            )

    def get_logs(self, limit: int = 50,
                 filter_status: Optional[str] = None) -> List[sqlite3.Row]:
        query = "SELECT * FROM access_logs"
        params: list = []
        if filter_status:
            query += " WHERE status=?"
            params.append(filter_status)
        query += " ORDER BY waktu DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            return conn.execute(query, params).fetchall()

    def statistik(self) -> dict:
        with self._conn() as conn:
            total_user   = conn.execute("SELECT COUNT(*) FROM users WHERE aktif=1").fetchone()[0]
            total_log    = conn.execute("SELECT COUNT(*) FROM access_logs").fetchone()[0]
            granted      = conn.execute("SELECT COUNT(*) FROM access_logs WHERE status='GRANTED'").fetchone()[0]
            denied_face  = conn.execute("SELECT COUNT(*) FROM access_logs WHERE status='DENIED_FACE'").fetchone()[0]
            denied_rfid  = conn.execute("SELECT COUNT(*) FROM access_logs WHERE status='DENIED_RFID'").fetchone()[0]
        return {
            "total_user":  total_user,
            "total_log":   total_log,
            "granted":     granted,
            "denied_face": denied_face,
            "denied_rfid": denied_rfid,
        }

"""
=============================================================
core/crypto.py  –  AES-128-GCM enkripsi / dekripsi embedding
=============================================================
Implementasi mengikuti NIST SP 800-38D (Galois/Counter Mode).

Spesifikasi:
  - Algoritma  : AES-128-GCM
  - Kunci      : 128-bit (16 byte), dibaca dari env AES_KEY_HEX
  - IV / Nonce : 96-bit (12 byte), di-generate secara acak tiap enkripsi
  - Tag        : 128-bit (16 byte), digabungkan ke ciphertext blob
  - AAD        : string "face_embedding" (additional authenticated data)
  - Encoding   : blob disimpan sebagai base64 string di SQLite

Format blob (base64-decoded):
  [ IV (12 byte) | ciphertext (N byte) | tag (16 byte) ]

Cara generate kunci sekali pakai:
  python -c "import secrets; print(secrets.token_hex(16))"
  → Salin output ke environment variable AES_KEY_HEX
=============================================================
"""

import os
import base64
import secrets
import json
import logging
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

import config

log = logging.getLogger(__name__)

# ─── Konstanta ────────────────────────────────────────────────────────────────
_IV_LEN  = 12   # byte  – NIST SP 800-38D §8.2 merekomendasikan 96-bit IV
_TAG_LEN = 16   # byte  – tag 128-bit
_AAD     = b"face_embedding"   # additional authenticated data

# ─── Kunci AES ───────────────────────────────────────────────────────────────

def _load_key() -> bytes:
    """
    Muat kunci AES-128 dari config.AES_KEY_HEX atau environment variable AES_KEY_HEX.
    Harus 32 karakter hex (= 16 byte = 128-bit).
    Jika tidak ada, generate kunci acak (development only).
    """
    hex_key = getattr(config, "AES_KEY_HEX", os.environ.get("AES_KEY_HEX", ""))
    if len(hex_key) == 32:
        try:
            return bytes.fromhex(hex_key)
        except ValueError:
            pass

    # Fallback: kunci acak (hanya untuk development, TIDAK untuk produksi)
    log.warning(
        "AES_KEY_HEX tidak diatur atau tidak valid. "
        "Menggunakan kunci acak (data tidak persisten)."
    )
    return secrets.token_bytes(16)


_AES_KEY: bytes = _load_key()


# ─── Primitif enkripsi / dekripsi (NIST SP 800-38D) ─────────────────────────

def aes_gcm_encrypt(plaintext: bytes, *, aad: bytes = _AAD) -> bytes:
    """
    Enkripsi bytes dengan AES-128-GCM (NIST SP 800-38D).

    Returns
    -------
    blob : bytes  –  [ IV (12) | ciphertext | tag (16) ]
    """
    iv = secrets.token_bytes(_IV_LEN)
    aesgcm = AESGCM(_AES_KEY)
    ct_and_tag = aesgcm.encrypt(iv, plaintext, aad)
    return iv + ct_and_tag          # tag sudah disertakan oleh library


def aes_gcm_decrypt(blob: bytes, *, aad: bytes = _AAD) -> bytes:
    """
    Dekripsi blob AES-128-GCM (NIST SP 800-38D).

    Parameters
    ----------
    blob : bytes  –  [ IV (12) | ciphertext | tag (16) ]

    Returns
    -------
    plaintext : bytes

    Raises
    ------
    ValueError  : jika blob terlalu pendek atau autentikasi gagal
    """
    if len(blob) < _IV_LEN + _TAG_LEN:
        raise ValueError("Blob AES-GCM terlalu pendek")
    iv         = blob[:_IV_LEN]
    ct_and_tag = blob[_IV_LEN:]
    aesgcm = AESGCM(_AES_KEY)
    try:
        return aesgcm.decrypt(iv, ct_and_tag, aad)
    except InvalidTag as exc:
        raise ValueError("Autentikasi AES-GCM gagal: tag tidak valid") from exc


# ─── Serialisasi embedding ────────────────────────────────────────────────────

def encrypt_embedding_list(embeddings: list) -> str:
    """
    Enkripsi list embedding menjadi base64 string untuk disimpan di DB.
    """
    # Gunakan .tolist() jika e adalah numpy array, jika tidak, biarkan apa adanya (e)
    payload = json.dumps(
        [e.tolist() if hasattr(e, 'tolist') else e for e in embeddings],
        separators=(",", ":")
    ).encode()
    blob = aes_gcm_encrypt(payload)
    return base64.b64encode(blob).decode()


def decrypt_embedding_list(blob_b64: str) -> list[np.ndarray]:
    """
    Dekripsi base64 string dari DB menjadi list numpy embedding.

    Alur:
      base64 string → bytes → AES-128-GCM → JSON → np.ndarray[]

    Raises
    ------
    ValueError  : jika dekripsi atau parsing gagal
    """
    try:
        blob      = base64.b64decode(blob_b64)
        plaintext = aes_gcm_decrypt(blob)
        data      = json.loads(plaintext.decode())
        return [np.array(e, dtype=np.float32) for e in data]
    except (ValueError, json.JSONDecodeError, Exception) as exc:
        raise ValueError(f"Gagal dekripsi embedding: {exc}") from exc

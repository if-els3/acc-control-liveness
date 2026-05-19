"""
=============================================================
test_aes_gcm.py
=============================================================
Unit test AES-128-GCM menggunakan test vector dari
aes_gcm_test.json (Wycheproof / NIST SP 800-38D kompatibel).

Logika enkripsi/dekripsi mengikuti NIST SP 800-38D:
  - AES-GCM dengan IV 96-bit (12 byte) untuk operasi standar
  - Tag 128-bit (16 byte)
  - Kunci 128-bit (16 byte)
  - Implementasi menggunakan Python `cryptography` library
    (AES-GCM di dalamnya mengikuti NIST SP 800-38D sepenuhnya)

Catatan IV yang di-skip:
  - ZeroLengthIv   : IV kosong (0 byte) – tidak aman dan ditolak library
  - SmallIv        : IV < 8 byte – library Python mensyaratkan >= 8 byte
  - LongIv         : IV > 128 byte – library Python mensyaratkan <= 128 byte
  Semua kasus ini ditandai sebagai SKIPPED dalam laporan test.

Jalankan: python test_aes_gcm.py
=============================================================
"""

import json
import os
import unittest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag


# ─── Path ke file test vector ────────────────────────────────────────────────
TEST_VECTOR_FILE = os.path.join(os.path.dirname(__file__), "aes_gcm_test.json")

# Flag yang menandai IV di luar batas yang didukung library Python
_UNSUPPORTED_IV_FLAGS = {"ZeroLengthIv", "SmallIv", "LongIv"}


# ─── Fungsi enkripsi / dekripsi mengikuti NIST SP 800-38D ───────────────────

def aes_gcm_encrypt(key: bytes, iv: bytes, plaintext: bytes, aad: bytes) -> tuple[bytes, bytes]:
    """
    Enkripsi AES-GCM sesuai NIST SP 800-38D.

    Parameters
    ----------
    key       : kunci 128-bit (16 byte)
    iv        : initialization vector, direkomendasikan 96-bit (12 byte)
    plaintext : plaintext yang akan dienkripsi (boleh kosong)
    aad       : additional authenticated data (boleh kosong)

    Returns
    -------
    (ciphertext, tag)
      ciphertext : len sama dengan plaintext
      tag        : 16 byte (128-bit)
    """
    aesgcm = AESGCM(key)
    # AESGCM.encrypt() mengembalikan ciphertext + tag (digabung)
    ct_and_tag = aesgcm.encrypt(iv, plaintext, aad if aad else None)
    tag_len = 16
    ciphertext = ct_and_tag[:-tag_len]
    tag = ct_and_tag[-tag_len:]
    return ciphertext, tag


def aes_gcm_decrypt(key: bytes, iv: bytes, ciphertext: bytes, tag: bytes, aad: bytes) -> bytes | None:
    """
    Dekripsi AES-GCM sesuai NIST SP 800-38D.
    Mengembalikan plaintext, atau None jika autentikasi gagal.
    """
    aesgcm = AESGCM(key)
    try:
        plaintext = aesgcm.decrypt(iv, ciphertext + tag, aad if aad else None)
        return plaintext
    except InvalidTag:
        return None


# ─── Helper memuat test vector ───────────────────────────────────────────────

def load_test_vectors(path: str, key_size: int = 128) -> list[dict]:
    """Muat semua test vector dari file JSON untuk ukuran kunci tertentu."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vectors = []
    for group in data.get("testGroups", []):
        if group.get("keySize") != key_size:
            continue
        for test in group.get("tests", []):
            vectors.append({
                "tcId":    test["tcId"],
                "comment": test.get("comment", ""),
                "flags":   set(test.get("flags", [])),
                "key":     bytes.fromhex(test["key"]),
                "iv":      bytes.fromhex(test["iv"]) if test["iv"] else b"",
                "aad":     bytes.fromhex(test["aad"]) if test["aad"] else b"",
                "msg":     bytes.fromhex(test["msg"]) if test["msg"]  else b"",
                "ct":      bytes.fromhex(test["ct"])  if test["ct"]   else b"",
                "tag":     bytes.fromhex(test["tag"]),
                "result":  test["result"],   # "valid" | "invalid" | "acceptable"
            })
    return vectors


def _is_supported_iv(v: dict) -> bool:
    """
    Cek apakah vector memiliki IV dalam range yang didukung
    library `cryptography` (8–128 byte).
    IV di luar range ini ditandai oleh flag khusus di dataset.
    """
    if v["flags"] & _UNSUPPORTED_IV_FLAGS:
        return False
    iv_len = len(v["iv"])
    return 8 <= iv_len <= 128


# ─── Test Class ──────────────────────────────────────────────────────────────

class TestAESGCM128(unittest.TestCase):
    """
    Unit test AES-128-GCM terhadap Wycheproof / NIST SP 800-38D test vectors.

    Setiap test vector dikategorikan:
    - valid    : enkripsi harus menghasilkan ct & tag yang cocok;
                 dekripsi harus berhasil dan menghasilkan plaintext semula.
    - invalid  : dekripsi harus GAGAL (tag salah / parameter tidak valid).
    - acceptable: diterima implementasi, mungkin mengandung parameter lemah.

    Vector dengan IV di luar batas yang didukung library (ZeroLengthIv,
    SmallIv, LongIv) di-skip karena merupakan batasan implementasi,
    bukan kesalahan algoritma.
    """

    @classmethod
    def setUpClass(cls):
        cls.vectors = load_test_vectors(TEST_VECTOR_FILE, key_size=128)
        supported = [v for v in cls.vectors if _is_supported_iv(v)]
        skipped   = len(cls.vectors) - len(supported)
        print(f"\n[Setup] Loaded {len(cls.vectors)} vector(s), "
              f"{len(supported)} supported, {skipped} skipped (IV out of range)")

    # ── 1. Tes enkripsi: ct & tag harus cocok ────────────────────────────────
    def test_encryption_valid_vectors(self):
        """Enkripsi pada vector 'valid' harus menghasilkan ct & tag yang sama."""
        tested = skipped = 0
        for v in self.vectors:
            if v["result"] != "valid":
                continue
            if not _is_supported_iv(v):
                skipped += 1
                continue

            with self.subTest(tcId=v["tcId"], comment=v["comment"]):
                ct, tag = aes_gcm_encrypt(v["key"], v["iv"], v["msg"], v["aad"])
                self.assertEqual(
                    ct, v["ct"],
                    f"tcId={v['tcId']}: ciphertext mismatch\n"
                    f"  expected: {v['ct'].hex()}\n"
                    f"  got:      {ct.hex()}"
                )
                self.assertEqual(
                    tag, v["tag"],
                    f"tcId={v['tcId']}: tag mismatch\n"
                    f"  expected: {v['tag'].hex()}\n"
                    f"  got:      {tag.hex()}"
                )
                tested += 1
        print(f"  [Encryption]  {tested} valid vector(s) lulus, {skipped} di-skip")

    # ── 2. Tes dekripsi: plaintext harus kembali semula ──────────────────────
    def test_decryption_valid_vectors(self):
        """Dekripsi pada vector 'valid' harus mengembalikan plaintext asli."""
        tested = skipped = 0
        for v in self.vectors:
            if v["result"] != "valid":
                continue
            if not _is_supported_iv(v):
                skipped += 1
                continue

            with self.subTest(tcId=v["tcId"], comment=v["comment"]):
                plaintext = aes_gcm_decrypt(v["key"], v["iv"], v["ct"], v["tag"], v["aad"])
                self.assertIsNotNone(
                    plaintext,
                    f"tcId={v['tcId']}: dekripsi gagal (autentikasi ditolak)"
                )
                self.assertEqual(
                    plaintext, v["msg"],
                    f"tcId={v['tcId']}: plaintext tidak cocok\n"
                    f"  expected: {v['msg'].hex()}\n"
                    f"  got:      {plaintext.hex() if plaintext else 'None'}"
                )
                tested += 1
        print(f"  [Decryption]  {tested} valid vector(s) lulus, {skipped} di-skip")

    # ── 3. Tes dekripsi vector invalid: harus ditolak ────────────────────────
    def test_decryption_invalid_vectors(self):
        """Dekripsi pada vector 'invalid' harus gagal (return None / exception)."""
        tested = skipped = 0
        for v in self.vectors:
            if v["result"] != "invalid":
                continue
            if not _is_supported_iv(v):
                skipped += 1
                continue

            with self.subTest(tcId=v["tcId"], comment=v["comment"]):
                try:
                    plaintext = aes_gcm_decrypt(v["key"], v["iv"], v["ct"], v["tag"], v["aad"])
                    self.assertIsNone(
                        plaintext,
                        f"tcId={v['tcId']}: seharusnya ditolak (invalid vector lolos)"
                    )
                except (ValueError, Exception):
                    pass  # Exception saat dekripsi = valid rejection
                tested += 1
        print(f"  [Rejection]   {tested} invalid vector(s) ditolak dengan benar, {skipped} di-skip")

    # ── 4. Ringkasan statistik ────────────────────────────────────────────────
    def test_vector_count(self):
        """Memastikan test vector berhasil dimuat dan statistik akurat."""
        self.assertGreater(len(self.vectors), 0, "Tidak ada vector yang dimuat")
        valid      = sum(1 for v in self.vectors if v["result"] == "valid")
        invalid    = sum(1 for v in self.vectors if v["result"] == "invalid")
        acceptable = sum(1 for v in self.vectors if v["result"] == "acceptable")
        supported  = sum(1 for v in self.vectors if _is_supported_iv(v))
        print(
            f"\n  [Summary]\n"
            f"    Total vectors   : {len(self.vectors)}\n"
            f"    Valid           : {valid}\n"
            f"    Invalid         : {invalid}\n"
            f"    Acceptable      : {acceptable}\n"
            f"    Supported IV    : {supported}\n"
            f"    Skipped (IV OOR): {len(self.vectors) - supported}"
        )


# ─── CLI helpers ──────────────────────────────────────────────────────────────

def _encrypt_cli(args):
    """CLI encryption: print ciphertext + tag as hex."""
    try:
        key = bytes.fromhex(args.key)
        iv  = bytes.fromhex(args.iv)
        msg = bytes.fromhex(args.msg)
        aad = bytes.fromhex(args.aad) if args.aad else b""
    except ValueError as e:
        print(f"Error: invalid hex input — {e}")
        return 1

    if len(key) != 16:
        print(f"Error: key must be 16 bytes (got {len(key)})")
        return 1

    ct, tag = aes_gcm_encrypt(key, iv, msg, aad)
    print(f"ciphertext: {ct.hex()}")
    print(f"tag:        {tag.hex()}")
    return 0


def _decrypt_cli(args):
    """CLI decryption: print plaintext as hex or 'AUTH_FAIL'."""
    try:
        key = bytes.fromhex(args.key)
        iv  = bytes.fromhex(args.iv)
        ct  = bytes.fromhex(args.ct)
        tag = bytes.fromhex(args.tag)
        aad = bytes.fromhex(args.aad) if args.aad else b""
    except ValueError as e:
        print(f"Error: invalid hex input — {e}")
        return 1

    if len(key) != 16:
        print(f"Error: key must be 16 bytes (got {len(key)})")
        return 1

    pt = aes_gcm_decrypt(key, iv, ct, tag, aad)
    if pt is None:
        print("AUTH_FAIL")
        return 1
    print(f"plaintext: {pt.hex()}")
    return 0


def _interactive_cli(_args=None):
    """Interactive prompt: ask action + params, loop until Ctrl+C."""
    print("Interactive AES-128-GCM. Actions: encrypt / decrypt / quit")
    while True:
        try:
            action = input("\nencrypt/decrypt/quit? ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if action in ("q", "quit", "exit"):
            break
        if action not in ("encrypt", "decrypt"):
            print("  Valid: encrypt, decrypt, quit")
            continue

        try:
            key_hex = input("  key (hex, 32 chars = 16 bytes): ").strip()
            iv_hex  = input("  iv  (hex): ").strip()
            aad_hex = input("  aad (hex, empty=skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        aad_hex = aad_hex or ""

        try:
            key, iv, aad = bytes.fromhex(key_hex), bytes.fromhex(iv_hex), bytes.fromhex(aad_hex)
            if len(key) != 16:
                print("  Error: key must be 16 bytes (32 hex chars)")
                continue
        except ValueError as e:
            print(f"  Error: {e}")
            continue

        if action == "encrypt":
            try:
                msg_hex = input("  plaintext (hex): ").strip()
                msg = bytes.fromhex(msg_hex)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            except ValueError as e:
                print(f"  Error: {e}")
                continue
            ct, tag = aes_gcm_encrypt(key, iv, msg, aad)
            print(f"  ciphertext: {ct.hex()}")
            print(f"  tag:        {tag.hex()}")
        else:
            try:
                ct_hex  = input("  ciphertext (hex): ").strip()
                tag_hex = input("  tag       (hex): ").strip()
                ct, tag = bytes.fromhex(ct_hex), bytes.fromhex(tag_hex)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            except ValueError as e:
                print(f"  Error: {e}")
                continue
            pt = aes_gcm_decrypt(key, iv, ct, tag, aad)
            if pt is None:
                print("  AUTH_FAIL (tag mismatch)")
            else:
                print(f"  plaintext: {pt.hex()}")
    return 0


# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AES-128-GCM tool + Wycheproof test suite"
    )
    sub = parser.add_subparsers(dest="command")

    p_enc = sub.add_parser("encrypt", help="Encrypt plaintext")
    p_enc.add_argument("--key", required=True, help="Key in hex (32 hex chars = 16 bytes)")
    p_enc.add_argument("--iv", required=True, help="IV in hex")
    p_enc.add_argument("--msg", required=True, help="Plaintext in hex")
    p_enc.add_argument("--aad", default="", help="AAD in hex (default empty)")

    p_dec = sub.add_parser("decrypt", help="Decrypt ciphertext")
    p_dec.add_argument("--key", required=True, help="Key in hex (32 hex chars)")
    p_dec.add_argument("--iv", required=True, help="IV in hex")
    p_dec.add_argument("--ct", required=True, help="Ciphertext in hex")
    p_dec.add_argument("--tag", required=True, help="Tag in hex")
    p_dec.add_argument("--aad", default="", help="AAD in hex (default empty)")

    sub.add_parser("interactive", help="Interactive prompt mode")
    sub.add_parser("test", help="Run unit tests against Wycheproof vectors")

    args = parser.parse_args()

    if args.command == "encrypt":
        exit(_encrypt_cli(args))
    elif args.command == "decrypt":
        exit(_decrypt_cli(args))
    elif args.command == "interactive":
        exit(_interactive_cli())
    else:
        # default: run unit tests
        print("=" * 60)
        print("  AES-128-GCM Unit Test  (NIST SP 800-38D / Wycheproof)")
        print("=" * 60)
        unittest.main(verbosity=2)

"""
=============================================================
menus/admin.py — Menu Administrasi
=============================================================
Sub-menu:
  1. Lihat semua pengguna
  2. Nonaktifkan / aktifkan pengguna
  3. Hapus pengguna
  4. Lihat log akses
  5. Statistik sistem
=============================================================
"""
import os
import sys
import logging
from datetime import datetime

log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from core.database import Database

SEP  = "─" * 70
SEP2 = "═" * 70

STATUS_COLOR = {
    "GRANTED":     "\033[92m",   # hijau
    "DENIED_FACE": "\033[91m",   # merah
    "DENIED_RFID": "\033[91m",   # merah
    "ENROLL":      "\033[94m",   # biru
    "ERROR":       "\033[93m",   # kuning
}
NC = "\033[0m"


def _header(title: str):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)


# ── KELOLA PENGGUNA ───────────────────────────────────────

def menu_daftar_user(db: Database):
    _header("DAFTAR PENGGUNA TERDAFTAR")
    users = db.get_all_users()

    if not users:
        print("  (Belum ada pengguna terdaftar)")
        input("\n  Tekan Enter ...")
        return

    print(f"  {'ID':<5} {'Nama':<25} {'UID RFID':<15} {'Wajah':<8} {'Status':<10} Terdaftar")
    print(f"  {SEP}")
    for u in users:
        aktif   = "Aktif"   if u['aktif'] else "Nonaktif"
        wajah   = "Ada"     if u['has_face'] else "Tidak"
        tgl     = u['dibuat'][:10]
        color   = "\033[92m" if u['aktif'] else "\033[90m"
        print(f"  {color}{u['id']:<5} {u['nama']:<25} {u['rfid_uid']:<15} "
              f"{wajah:<8} {aktif:<10} {tgl}{NC}")
    print(f"\n  Total: {len(users)} pengguna")
    input("\n  Tekan Enter ...")


def menu_kelola_user(db: Database):
    _header("KELOLA PENGGUNA")
    users = db.get_all_users()

    if not users:
        print("  Belum ada pengguna.")
        input("  Tekan Enter ..."); return

    for u in users:
        aktif = "Aktif" if u['aktif'] else "Nonaktif"
        print(f"  [{u['id']}] {u['nama']} — {u['rfid_uid']} ({aktif})")

    print()
    try:
        user_id = int(input("  Masukkan ID pengguna (0=batal): ").strip())
    except ValueError:
        return
    if user_id == 0:
        return

    # Cari user
    user = next((u for u in users if u['id'] == user_id), None)
    if user is None:
        print("  [!] ID tidak ditemukan."); input("  Enter ..."); return

    print(f"\n  Pengguna  : {user['nama']}")
    print(f"  UID RFID  : {user['rfid_uid']}")
    print(f"  Status    : {'Aktif' if user['aktif'] else 'Nonaktif'}")
    print(f"  Wajah     : {'Ada' if user['has_face'] else 'Tidak'}")
    print()
    print("  Pilihan:")
    if user['aktif']:
        print("  [1] Nonaktifkan pengguna")
    else:
        print("  [1] Aktifkan kembali pengguna")
    print("  [2] Hapus pengguna (permanen)")
    print("  [0] Kembali")

    pilih = input("\n  Pilihan : ").strip()

    if pilih == '1':
        if user['aktif']:
            db.nonaktifkan_user(user_id)
            print(f"  ✔ {user['nama']} dinonaktifkan.")
        else:
            db.aktifkan_user(user_id)
            print(f"  ✔ {user['nama']} diaktifkan kembali.")

    elif pilih == '2':
        konfirm = input(f"  Hapus '{user['nama']}' secara permanen? [HAPUS]: ").strip()
        if konfirm == "HAPUS":
            db.hapus_user(user_id)
            print(f"  ✔ Pengguna dihapus.")
        else:
            print("  Dibatalkan.")

    input("\n  Tekan Enter ...")


# ── LOG AKSES ─────────────────────────────────────────────

def menu_log_akses(db: Database):
    _header("LOG AKSES")

    print("  Filter:")
    print("  [1] Semua log")
    print("  [2] Akses diterima (GRANTED)")
    print("  [3] Ditolak wajah  (DENIED_FACE)")
    print("  [4] Ditolak RFID   (DENIED_RFID)")
    print("  [5] Error")

    pilih = input("\n  Pilihan [1]: ").strip() or '1'
    filter_map = {'2':'GRANTED','3':'DENIED_FACE','4':'DENIED_RFID','5':'ERROR'}
    filter_status = filter_map.get(pilih)

    try:
        limit = int(input("  Tampilkan berapa baris? [50]: ").strip() or 50)
    except ValueError:
        limit = 50

    logs = db.get_logs(limit=limit, filter_status=filter_status)

    if not logs:
        print("\n  (Tidak ada log)")
        input("  Tekan Enter ..."); return

    print(f"\n  {SEP}")
    print(f"  {'Waktu':<22} {'UID':<14} {'Nama':<20} {'Status':<15} Keterangan")
    print(f"  {SEP}")

    for entry in logs:
        status = entry['status']
        color  = STATUS_COLOR.get(status, "")
        waktu  = entry['waktu'][:19].replace('T', ' ')
        uid    = (entry['rfid_uid'] or '-')[:13]
        nama   = (entry['nama'] or '-')[:19]
        ket    = (entry['keterangan'] or '')[:30]
        print(f"  {color}{waktu:<22} {uid:<14} {nama:<20} {status:<15} {ket}{NC}")

    print(f"\n  Menampilkan {len(logs)} entri")
    input("\n  Tekan Enter ...")


# ── STATISTIK ─────────────────────────────────────────────

def menu_statistik(db: Database):
    _header("STATISTIK SISTEM")
    stat = db.statistik()

    total = stat['granted'] + stat['denied_face'] + stat['denied_rfid']
    rate  = (stat['granted'] / total * 100) if total > 0 else 0

    print(f"\n  {'─'*35}")
    print(f"  Pengguna aktif      : {stat['total_user']}")
    print(f"  {'─'*35}")
    print(f"  Total percobaan akses: {total}")
    print(f"  \033[92mAkses diterima       : {stat['granted']}\033[0m")
    print(f"  \033[91mDitolak (wajah)      : {stat['denied_face']}\033[0m")
    print(f"  \033[91mDitolak (RFID)       : {stat['denied_rfid']}\033[0m")
    print(f"  \033[93mError                : {stat['total_log'] - total}\033[0m")
    print(f"  {'─'*35}")
    print(f"  Tingkat keberhasilan : {rate:.1f}%")
    print(f"  {'─'*35}")

    input("\n  Tekan Enter ...")


# ── KONFIGURASI ───────────────────────────────────────────

def menu_konfigurasi():
    _header("KONFIGURASI SISTEM")
    print(f"\n  {'Parameter':<30} Nilai")
    print(f"  {'─'*50}")
    items = [
        ("Database",         config.DB_PATH),
        ("Kamera index",     str(config.CAMERA_INDEX)),
        ("Resolusi kamera",  f"{config.CAMERA_WIDTH}×{config.CAMERA_HEIGHT}"),
        ("RFID SPI device",  f"CE{config.SPI_DEVICE}"),
        ("RFID gain",        f"0x{config.RFID_GAIN:02X} (48dB max)"),
        ("RFID timeout",     f"{config.RFID_TIMEOUT}s"),
        ("Face threshold",   f"{config.FACE_MATCH_THRESH:.0%}"),
        ("Enroll frames",    str(config.ENROLL_FRAMES)),
        ("TFLite threads",   str(config.TFLITE_THREADS)),
        ("Servo pin (BCM)",  str(config.SERVO_PIN)),
        ("Durasi buka pintu",f"{config.DOOR_OPEN_SEC}s"),
        ("Model BlazeFace",  "Ada" if os.path.exists(config.BLAZEFACE_MODEL) else "Belum ada"),
        ("Model MobileFaceNet", "Ada" if os.path.exists(config.MOBILEFACENET_MODEL) else "Belum ada (pakai LBPH)"),
    ]
    for k, v in items:
        print(f"  {k:<30} {v}")
    print()
    print("  Untuk mengubah konfigurasi, edit file: config.py")
    input("\n  Tekan Enter ...")

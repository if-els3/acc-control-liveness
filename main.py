#!/usr/bin/env python3
"""
=============================================================
main.py — Sistem Kendali Akses RFID + Liveness + Face
=============================================================
Raspberry Pi | MFRC522 via SPI | Kamera USB | Servo | LCD
 
Jalankan: python3 main.py
=============================================================
"""
import os, sys, logging, signal
import config
 
os.makedirs(config.LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
logging.getLogger().handlers[1].setLevel(logging.WARNING)
log = logging.getLogger("main")
 
from core.database      import Database
from core.face_engine   import FaceEngine
from core.servo         import DoorController
from menus.enrollment import menu_daftar_pengguna, menu_update_wajah
from menus.access     import (menu_akses_sekali, menu_akses_kontinu,
                              menu_toggle_liveness, menu_uji_liveness)
from menus.admin      import (menu_daftar_user, menu_kelola_user,
                              menu_log_akses, menu_statistik,
                              menu_konfigurasi)
 
SEP2 = "═" * 60
 
# ─────────────────────────────────────────────────────────
 
def init_system():
    print(f"\n{SEP2}")
    print(f"  {config.APP_NAME}  v{config.APP_VERSION}")
    print(SEP2)
 
    print("  [1/4] Database ...", end=" ", flush=True)
    db = Database()
    print("OK")
 
    print("  [2/4] Face Recognition ...", end=" ", flush=True)
    face_engine = FaceEngine()
    ok = face_engine.load()
    print(f"OK ({face_engine.mode})" if ok else "GAGAL (FR tidak aktif)")
 
    print("  [3/4] Servo/Pintu ...", end=" ", flush=True)
    door = DoorController()
    door.start()
    print("OK")
 
 
    lv = "ON" if config.LIVENESS_ENABLED else "OFF"
    print(f"\n  Liveness Detection : {lv}")
    print(f"  Face threshold     : {config.FACE_MATCH_THRESH:.0%}")
    print(f"  Sistem siap.\n")
    return db, face_engine, door
 
 
def cleanup(door):
    try: door.cleanup()
    except Exception: pass
    log.info("Sistem dihentikan")
 
 
# ─────────────────────────────────────────────────────────
 
def tampilkan_menu(face_engine):
    lv = "\033[92mON\033[0m" if config.LIVENESS_ENABLED else "\033[93mOFF\033[0m"
    fr = face_engine.mode if face_engine.is_loaded else "Tidak aktif"
    print(f"\n{SEP2}")
    print(f"  {config.APP_NAME.upper()}")
    print(f"  v{config.APP_VERSION}  |  FR: {fr}  |  Liveness: {lv}")
    print(SEP2)
    print()
    print("  ── OPERASIONAL ─────────────────────────────────")
    print("  [1] Mode Operasional  (akses kontinu, Ctrl+C stop)")
    print("  [2] Uji Akses Sekali")
    print("  [3] Uji Liveness Saja (tanpa RFID)")
    print("  [4] Toggle Liveness   (ON/OFF sementara)")
    print()
    print("  ── PENDAFTARAN ──────────────────────────────────")
    print("  [5] Daftarkan Pengguna Baru  (RFID + Wajah)")
    print("  [6] Update Data Wajah")
    print()
    print("  ── ADMINISTRASI ─────────────────────────────────")
    print("  [7] Daftar Pengguna")
    print("  [8] Kelola Pengguna   (nonaktif/hapus)")
    print("  [9] Log Akses")
    print("  [a] Statistik Sistem")
    print("  [b] Konfigurasi")
    print()
    print("  [0] Keluar")
    print(SEP2)
 
 
def main():
    db, face_engine, door = init_system()
    
    try:
        from web.app import run_web
        host = getattr(config, 'WEB_HOST', '0.0.0.0')
        port = getattr(config, 'WEB_PORT', 5000)
        run_web(db, face_engine, door, host=host, port=port)
        print(f"  [4/4] Web Interface ...OK")
        print(f"\n  🌐 Akses stream & status:")
        print(f"     http://{host}:{port}/")
        print(f"     http://{host}:{port}/display")
    except Exception as e:
        log.error(f"Gagal memulai web interface: {e}")
 
    def _sig(sig, frame):
        print("\n\n  [!] Ctrl+C — kembali ke menu")
        raise KeyboardInterrupt
 
    signal.signal(signal.SIGINT, _sig)
    input("  Tekan Enter untuk masuk ke menu ...")
 
    try:
        while True:
            tampilkan_menu(face_engine)
            pilih = input("\n  Pilihan: ").strip().lower()
 
            try:
                if   pilih == '1': menu_akses_kontinu(db, face_engine, door)
                elif pilih == '2': menu_akses_sekali(db, face_engine, door)
                elif pilih == '3': menu_uji_liveness(face_engine)
                elif pilih == '4': menu_toggle_liveness()
                elif pilih == '5': menu_daftar_pengguna(db, face_engine)
                elif pilih == '6': menu_update_wajah(db, face_engine)
                elif pilih == '7': menu_daftar_user(db)
                elif pilih == '8': menu_kelola_user(db)
                elif pilih == '9': menu_log_akses(db)
                elif pilih == 'a': menu_statistik(db)
                elif pilih == 'b': menu_konfigurasi()
                elif pilih == '0':
                    if input("\n  Keluar? [y/N]: ").strip().lower() == 'y':
                        break
                else:
                    print("  [!] Pilihan tidak valid.")
            except KeyboardInterrupt:
                print()
 
    finally:
        print(f"\n{SEP2}")
        print("  Menutup sistem ...")
        cleanup(door)
        print("  Sampai jumpa!")
        print(SEP2)
 
 
if __name__ == "__main__":
    main()
 

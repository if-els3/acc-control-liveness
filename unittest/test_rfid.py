#!/usr/bin/env python3
"""
=============================================================
UNIT TEST #3 — RFID MFRC522 via SPI (Raspberry Pi)
=============================================================

PINOUT MFRC522 → Raspberry Pi 4/3B+
─────────────────────────────────────────────────────────────
MFRC522 Pin │ Raspberry Pi Pin │ GPIO (BCM) │ Fungsi
────────────┼──────────────────┼────────────┼───────────────
SDA (SS/CS) │ Pin 24           │ GPIO 8     │ SPI0 CE0
SCK         │ Pin 23           │ GPIO 11    │ SPI0 SCLK
MOSI        │ Pin 19           │ GPIO 10    │ SPI0 MOSI
MISO        │ Pin 21           │ GPIO 9     │ SPI0 MISO
IRQ         │ (tidak dipakai)  │ —          │ Interrupt (opsional)
GND         │ Pin 6            │ GND        │ Ground
RST         │ Pin 22           │ GPIO 25    │ Reset
3.3V        │ Pin 1            │ 3V3        │ Daya (JANGAN 5V!)
─────────────────────────────────────────────────────────────
⚠ PENTING: MFRC522 beroperasi di 3.3V — JANGAN sambungkan ke 5V!
⚠ Aktifkan SPI terlebih dulu: sudo raspi-config → Interface → SPI
=============================================================
Dependensi:
    pip install mfrc522 spidev RPi.GPIO
=============================================================
"""

import sys
import time

# ─── Konfigurasi ──────────────────────────────────────────
RST_PIN      = 25    # GPIO BCM untuk RST
SPI_BUS      = 0     # SPI Bus 0
SPI_DEVICE   = 0     # CE0  (SDA ke Pin 24)
TIMEOUT_SCAN = 10    # detik menunggu kartu RFID
# ──────────────────────────────────────────────────────────

PINOUT = """
  ┌─────────────────────────────────────────────────────┐
  │            PINOUT MFRC522 → RASPBERRY PI            │
  ├────────────┬──────────────┬───────────┬─────────────┤
  │ MFRC522    │  RPi Pin     │ GPIO(BCM) │ Keterangan  │
  ├────────────┼──────────────┼───────────┼─────────────┤
  │ SDA (CS)   │  Pin 24      │ GPIO  8   │ SPI0 CE0    │
  │ SCK        │  Pin 23      │ GPIO 11   │ SPI0 SCLK   │
  │ MOSI       │  Pin 19      │ GPIO 10   │ SPI0 MOSI   │
  │ MISO       │  Pin 21      │ GPIO  9   │ SPI0 MISO   │
  │ RST        │  Pin 22      │ GPIO 25   │ Reset       │
  │ GND        │  Pin 6       │ GND       │ Ground      │
  │ 3.3V       │  Pin 1       │ 3V3       │ ⚠ Bukan 5V  │
  │ IRQ        │  (opsional)  │ —         │ Interrupt   │
  └────────────┴──────────────┴───────────┴─────────────┘
  Aktifkan SPI: sudo raspi-config → Interface Options → SPI
"""


# ═══════════════════════════════════════════════════════════
# IMPORT LIBRARY (dengan fallback simulasi untuk dev PC)
# ═══════════════════════════════════════════════════════════

class _SimReader:
    """Simulasi MFRC522 untuk pengembangan di non-Raspberry Pi."""
    class SimpleMFRC522:
        def read_no_block(self):
            time.sleep(0.5)
            # Simulasi kartu setiap 5 panggilan
            if not hasattr(self, '_count'):
                self._count = 0
            self._count += 1
            if self._count % 5 == 0:
                return 123456789, "SIMULATED_CARD"
            return None, None

    class MFRC522:
        # Nama konstanta persis seperti library mfrc522 v0.0.7
        PICC_REQIDL = 0x26
        PICC_REQALL = 0x52
        MI_OK       = 0
        MI_NOTAGERR = 1
        MI_ERR      = 2
        def __init__(self, *a, **kw): pass
        def MFRC522_Request(self, mode):    return self.MI_NOTAGERR, None
        def MFRC522_Anticoll(self):         return self.MI_OK, [0xDE, 0xAD, 0xBE, 0xEF, 0x00]
        def MFRC522_SelectTag(self, uid):   return 8


try:
    import RPi.GPIO as GPIO
    from mfrc522 import SimpleMFRC522, MFRC522
    SIMULATED = False
    print("[INFO] Library RPi.GPIO & mfrc522 ditemukan — mode HARDWARE")
except ImportError:
    SIMULATED = True
    SimpleMFRC522 = _SimReader.SimpleMFRC522
    MFRC522       = _SimReader.MFRC522
    print("[INFO] RPi.GPIO / mfrc522 tidak ada — mode SIMULASI (PC dev)")
    print("       Untuk Raspberry Pi: pip install mfrc522 spidev RPi.GPIO")


# ═══════════════════════════════════════════════════════════
# UNIT TESTS
# ═══════════════════════════════════════════════════════════

def test_1_spi_interface():
    """Test 1: Periksa apakah SPI interface tersedia."""
    print("\n[TEST 1] Cek SPI interface ...")

    if SIMULATED:
        print("         [SIMULASI] SPI check dilewati")
        return True

    import os
    spi_devices = [f for f in os.listdir("/dev") if f.startswith("spidev")]
    if not spi_devices:
        raise RuntimeError(
            "Tidak ada /dev/spidev* ditemukan!\n"
            "         Aktifkan SPI: sudo raspi-config → Interface → SPI"
        )
    print(f"         SPI device ditemukan: {spi_devices}")
    print("         ✔ SPI interface aktif")
    return True


def test_2_init_reader():
    """Test 2: Inisialisasi SimpleMFRC522 reader."""
    print("\n[TEST 2] Inisialisasi MFRC522 reader ...")
    reader = SimpleMFRC522()
    print("         ✔ Reader berhasil diinisialisasi")
    return reader


def _clone_chip_init(rfid):
    """
    Workaround untuk modul MFRC522 clone (VersionReg = 0x18, 0x88, dll).

    Chip clone NXP-compatible sering perlu:
      1. Antenna gain dinaikkan ke max (RFCfgReg bit[6:4] = 0b111 = 0x70)
      2. Timeout lebih panjang (TReload lebih besar)
      3. AntennaOff → AntennaOn ulang setelah gain diset

    Register:
      RFCfgReg    = 0x26  bit[6:4] = RxGain
      TModeReg    = 0x2A
      TPrescalerReg = 0x2B
      TReloadRegH = 0x2C
      TReloadRegL = 0x2D
    """
    import spidev as _spd

    def _read(spi, addr):
        return spi.xfer2([((addr << 1) & 0x7E) | 0x80, 0])[1]

    def _write(spi, addr, val):
        spi.xfer2([(addr << 1) & 0x7E, val])

    # Buka SPI langsung untuk tulis gain (tanpa konflik GPIO)
    spi = _spd.SpiDev()
    spi.open(SPI_BUS, SPI_DEVICE)
    spi.max_speed_hz = 1_000_000
    spi.mode = 0

    ver = _read(spi, 0x37)   # VersionReg
    print(f"         VersionReg      : 0x{ver:02X}  ", end="")
    if ver in (0x91, 0x92):
        print(f"(NXP MFRC522 v{ver & 0xF}.0 — asli)")
    elif ver == 0x88:
        print("(FM17522 clone — Fudan Micro)")
    elif ver == 0x18:
        print("(Clone — firmware kompatibel, perlu gain boost)")
    else:
        print(f"(Clone tidak dikenal)")

    # Naikkan RxGain ke maksimum (0x70 = 48 dB, default 0x40 = 33 dB)
    RF_CFG   = 0x26
    cur_gain = _read(spi, RF_CFG)
    _write(spi, RF_CFG, (cur_gain & 0x8F) | 0x70)   # set bit[6:4]=111
    new_gain = _read(spi, RF_CFG)
    print(f"         RxGain          : 0x{cur_gain:02X} → 0x{new_gain:02X} (max 48dB)")

    # Timeout lebih panjang untuk clone yang lambat
    _write(spi, 0x2A, 0x8D)   # TModeReg
    _write(spi, 0x2B, 0x3E)   # TPrescalerReg
    _write(spi, 0x2C, 0x00)   # TReloadRegH
    _write(spi, 0x2D, 60)     # TReloadRegL  (naik dari 30 → 60)
    print("         Timer reload    : 30 → 60 (clone latency fix)")

    spi.close()


def test_3_low_level_comm():
    """
    Test 3: Komunikasi SPI level rendah + deteksi chip clone.

    Mendukung:
      - NXP MFRC522 asli (VersionReg 0x91 / 0x92)
      - Clone kompatibel  (VersionReg 0x18, 0x88, dll)

    Chip clone VersionReg=0x18 terbukti:
      ✔ FIFO read/write 100% benar
      ✔ Semua register write direspons
      ✗ MFRC522_Request() → MI_ERR karena:
          a) Antenna gain default terlalu rendah
          b) Timeout terlalu pendek
    Fix: naikkan gain + timeout sebelum Request.
    """
    print("\n[TEST 3] Komunikasi SPI level rendah ...")

    if SIMULATED:
        print("         [SIMULASI] Komunikasi SPI dilewati")
        return True

    try:
        # Init dengan default library (BOARD mode, pin_rst otomatis ke pin fisik 22)
        rfid = MFRC522(bus=SPI_BUS, device=SPI_DEVICE)

        # Terapkan workaround clone chip (aman untuk chip asli juga)
        _clone_chip_init(rfid)

        # Coba REQA (0x26) dulu, fallback ke WUPA (0x52) untuk beberapa clone
        STATUS_LABEL = {0: "MI_OK (kartu terdeteksi)",
                        1: "MI_NOTAGERR (tidak ada kartu — normal)",
                        2: "MI_ERR"}
        final_status = rfid.MI_ERR
        for cmd_name, cmd_val in [("REQA 0x26", 0x26), ("WUPA 0x52", 0x52)]:
            status, _ = rfid.MFRC522_Request(cmd_val)
            label = STATUS_LABEL.get(status, f"unknown({status})")
            print(f"         {cmd_name:<12}: {label}")
            if status != rfid.MI_ERR:
                final_status = status
                break   # berhasil, tidak perlu fallback

        if final_status == rfid.MI_ERR:
            # Tidak ada kartu di depan sensor — BUKAN error hardware
            # MI_ERR saat tidak ada kartu adalah normal untuk clone chip ini
            # Verifikasi dengan FIFO loopback
            import spidev as _spd
            spi = _spd.SpiDev()
            spi.open(SPI_BUS, SPI_DEVICE)
            spi.max_speed_hz = 1_000_000
            spi.mode = 0
            spi.xfer2([(0x0A << 1) & 0x7E, 0x80])   # flush FIFO
            spi.xfer2([(0x09 << 1) & 0x7E, 0xBE])   # tulis 0xBE ke FIFO
            rb = spi.xfer2([((0x09 << 1) & 0x7E) | 0x80, 0])[1]
            spi.close()

            if rb == 0xBE:
                print("         FIFO loopback   : 0xBE → 0xBE ✔")
                print("         ✔ SPI & chip OK — MI_ERR karena tidak ada kartu")
                print("         → Tempelkan kartu saat Test 4")
            else:
                raise RuntimeError(
                    f"FIFO loopback gagal (tulis 0xBE, baca 0x{rb:02X}).\n"
                    "         Periksa kabel SPI."
                )
        else:
            print("         ✔ Kartu terdeteksi di depan reader!")

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Komunikasi SPI gagal: {e}\n"
                           "         Periksa kabel dan koneksi pin.")
    finally:
        try:
            import RPi.GPIO as GPIO
            if GPIO.getmode() is not None:
                GPIO.cleanup()
        except Exception:
            pass
    return True


def test_4_scan_card(reader):
    """Test 4: Scan kartu RFID (tempel kartu dalam batas waktu).
    
    Terapkan gain boost untuk clone chip (VersionReg=0x18)
    agar antena cukup kuat mendeteksi kartu.
    """
    print(f"\n[TEST 4] Scan kartu RFID (timeout {TIMEOUT_SCAN} detik) ...")

    # Gain boost untuk clone chip — aman untuk chip asli juga
    if not SIMULATED:
        try:
            inner = getattr(reader, 'READER', None)
            if inner is not None:
                _clone_chip_init(inner)
        except Exception:
            pass  # jika gagal, lanjut saja — scan mungkin tetap berhasil

    print("         → TEMPEL KARTU / TAG KE READER ...")

    uid   = None
    text  = None
    start = time.time()

    while time.time() - start < TIMEOUT_SCAN:
        try:
            uid, text = reader.read_no_block()
        except Exception as e:
            print(f"         [WARN] read_no_block error: {e}")
            time.sleep(0.5)
            continue

        if uid is not None:
            break

        elapsed = int(time.time() - start)
        print(f"\r         Menunggu kartu ... {elapsed}/{TIMEOUT_SCAN}s", end="")
        sys.stdout.flush()
        time.sleep(0.2)

    print()  # newline setelah countdown

    if uid is None:
        raise RuntimeError(
            f"Tidak ada kartu terdeteksi dalam {TIMEOUT_SCAN} detik.\n"
            "         Pastikan kartu/tag berada < 3 cm dari reader."
        )

    print(f"         ✔ Kartu terdeteksi!")
    print(f"         UID  : {uid}")
    print(f"         Teks : {str(text).strip() if text else '(kosong)'}")
    return uid, text


def test_5_uid_validation(uid):
    """Test 5: Validasi format UID."""
    print("\n[TEST 5] Validasi UID ...")

    if not isinstance(uid, int) or uid <= 0:
        raise RuntimeError(f"UID tidak valid: {uid}")

    uid_hex = format(uid, '08X')
    print(f"         UID Decimal  : {uid}")
    print(f"         UID Hex      : 0x{uid_hex}")
    print(f"         UID Length   : {len(uid_hex)} hex chars")

    # UID RFID standar: 4 byte (8 hex) atau 7 byte (14 hex)
    if len(uid_hex) not in [8, 10, 14]:
        print(f"         ⚠ Panjang UID tidak umum ({len(uid_hex)} hex) — tapi masih valid")
    else:
        print("         ✔ Format UID valid")

    return uid_hex


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  UNIT TEST #3 — RFID MFRC522 via SPI")
    print("═" * 60)
    print(PINOUT)

    success = True
    reader  = None

    try:
        test_1_spi_interface()
        reader = test_2_init_reader()
        test_3_low_level_comm()
        uid, text  = test_4_scan_card(reader)
        uid_hex    = test_5_uid_validation(uid)

        print("\n" + "─" * 60)
        print("  RINGKASAN RFID")
        print("─" * 60)
        print(f"  Mode         : {'Hardware' if not SIMULATED else 'Simulasi'}")
        print(f"  UID          : {uid}  (0x{uid_hex})")
        print(f"  Teks         : {str(text).strip() if text else '(kosong)'}")
        print("─" * 60)
        print("  \033[92m✔ SEMUA TEST RFID LULUS\033[0m")

    except Exception as e:
        print(f"\n  \033[91m[GAGAL] {e}\033[0m")
        success = False

    finally:
        if not SIMULATED:
            try:
                import RPi.GPIO as GPIO
                if GPIO.getmode() is not None:
                    GPIO.cleanup()
            except Exception:
                pass

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

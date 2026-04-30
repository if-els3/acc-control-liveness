"""
=============================================================
core/rfid_reader.py — MFRC522 Reader Wrapper
=============================================================
Mendukung:
  - NXP MFRC522 asli (VersionReg 0x91 / 0x92)
  - Clone chip (VersionReg 0x18, 0x88, dll)

Kabel (GPIO BOARD mode):
  SDA → Pin 24  |  SCK → Pin 23  |  MOSI → Pin 19
  MISO → Pin 21 |  RST → Pin 22  |  3.3V → Pin 1  |  GND → Pin 6
=============================================================
"""
import time
import logging
import os

log = logging.getLogger(__name__)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

SIMULATED = False

try:
    import spidev
    import RPi.GPIO as GPIO
    from mfrc522 import SimpleMFRC522, MFRC522 as _MFRC522
except ImportError:
    SIMULATED = True
    log.warning("RPi.GPIO / mfrc522 tidak ada — mode SIMULASI")

    class _FakeSPI:
        def open(self, *a): pass
        def close(self): pass
        def xfer2(self, data): return [0] * len(data)

    class _FakeMFRC522:
        PICC_REQIDL = 0x26
        MI_OK = 0; MI_NOTAGERR = 1; MI_ERR = 2
        def __init__(self, *a, **kw): pass
        def MFRC522_Request(self, m): return self.MI_NOTAGERR, None

    class _FakeSimple:
        _cnt = 0
        def read_no_block(self):
            self._cnt += 1
            if self._cnt % 6 == 0:
                return 123456789, "SIMULASI"
            return None, None

    SimpleMFRC522 = _FakeSimple
    _MFRC522      = _FakeMFRC522
    spidev        = None
    GPIO          = None


def _apply_clone_fix():
    """Naikkan antenna gain dan timeout untuk clone chip."""
    if SIMULATED or spidev is None:
        return
    try:
        spi = spidev.SpiDev()
        spi.open(config.SPI_BUS, config.SPI_DEVICE)
        spi.max_speed_hz = config.SPI_SPEED
        spi.mode = 0

        def _r(addr): return spi.xfer2([((addr << 1) & 0x7E) | 0x80, 0])[1]
        def _w(addr, val): spi.xfer2([(addr << 1) & 0x7E, val])

        # RFCfgReg: naikkan gain ke max 48dB
        cur = _r(0x26)
        _w(0x26, (cur & 0x8F) | config.RFID_GAIN)
        # Timer: perpanjang timeout
        _w(0x2A, 0x8D); _w(0x2B, 0x3E)
        _w(0x2C, 0x00); _w(0x2D, 60)
        spi.close()
    except Exception as e:
        log.debug(f"clone_fix gagal (aman diabaikan): {e}")


class RFIDReader:
    """
    High-level RFID reader.

    Penggunaan:
        reader = RFIDReader()
        reader.start()
        uid, text = reader.scan(timeout=15)
        reader.stop()

    atau pakai context manager:
        with RFIDReader() as reader:
            uid, text = reader.scan()
    """

    def __init__(self):
        self._reader  = None
        self._started = False

    def start(self):
        if self._started:
            return
        if not SIMULATED:
            # Pastikan GPIO bersih
            try:
                if GPIO.getmode() is not None:
                    GPIO.cleanup()
            except Exception:
                pass
        self._reader  = SimpleMFRC522()
        self._started = True
        # Terapkan fix clone chip setelah reader init
        _apply_clone_fix()
        log.info("RFID reader aktif")

    def stop(self):
        if not SIMULATED and GPIO:
            try:
                if GPIO.getmode() is not None:
                    GPIO.cleanup()
            except Exception:
                pass
        self._started = False
        log.info("RFID reader berhenti")

    def scan(self, timeout: int = config.RFID_TIMEOUT) -> tuple:
        """
        Tunggu kartu RFID.
        Return: (uid: int, text: str) atau (None, None) jika timeout.
        """
        if not self._started:
            self.start()

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                uid, text = self._reader.read_no_block()
            except Exception as e:
                log.warning(f"read_no_block error: {e}")
                time.sleep(0.3)
                continue

            if uid is not None:
                log.info(f"Kartu terdeteksi: UID={uid}")
                return uid, text
            time.sleep(0.1)

        log.info("Scan timeout — tidak ada kartu")
        return None, None

    def read_uid(self, timeout: int = config.RFID_TIMEOUT):
        uid, _ = self.scan(timeout=timeout)
        return uid

    def get_version(self) -> str:
        """Kembalikan string versi chip."""
        if SIMULATED:
            return "SIMULASI"
        try:
            spi = spidev.SpiDev()
            spi.open(config.SPI_BUS, config.SPI_DEVICE)
            spi.max_speed_hz = config.SPI_SPEED
            spi.mode = 0
            ver = spi.xfer2([((0x37 << 1) & 0x7E) | 0x80, 0])[1]
            spi.close()
            known = {0x91: "NXP v1.0", 0x92: "NXP v2.0",
                     0x88: "FM17522", 0x18: "Clone-0x18"}
            return known.get(ver, f"Clone-0x{ver:02X}")
        except Exception:
            return "Unknown"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

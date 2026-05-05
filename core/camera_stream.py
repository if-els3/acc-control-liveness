"""
=============================================================
core/camera_stream.py — Threaded Camera Manager
=============================================================
Membaca frame dari kamera di thread terpisah untuk mencegah
I/O blocking dan frame lag saat inferensi AI berjalan.
=============================================================
"""
import cv2
import threading
import time
import logging
import sys
import os

log = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

class CameraStream:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.src = getattr(config, 'CAMERA_INDEX', getattr(config, 'CAMERA_SRC', 0))
        self.stream = None
        self.grabbed = False
        self.frame = None
        self.stopped = False
        self.thread = None
        self._lock = threading.Lock()
        self._ref_count = 0
        self._initialized = True

    def start(self) -> bool:
        """Inisiasi stream kamera dan jalankan thread latar belakang."""
        with self._lock:
            if self.stream is not None and self.stream.isOpened():
                log.info("Kamera sudah aktif (singleton).")
                self.stopped = False
                self._ref_count += 1
                return True

            log.info(f"Memulai kamera dari sumber: {self.src}")

            # Coba buka kamera sesuai config
            self.stream = cv2.VideoCapture(self.src)

            # Fallback ke index lain jika gagal (0, 1, 2, atau -1)
            if not self.stream.isOpened():
                log.warning(f"Gagal membuka kamera pada index {self.src}. Mencoba fallback...")
                fallback_indices = [0, 1, 2, -1]
                if isinstance(self.src, int) and self.src in fallback_indices:
                    fallback_indices.remove(self.src)

                for idx in fallback_indices:
                    self.stream = cv2.VideoCapture(idx)
                    if self.stream.isOpened():
                        log.info(f"Berhasil membuka kamera fallback pada index {idx}.")
                        self.src = idx
                        break

            if not self.stream.isOpened():
                log.error("Kamera gagal diakses di semua index. Pastikan hardware terhubung.")
                return False

            # Set resolusi jika tersedia di config (optimasi untuk RPi)
            if hasattr(config, 'CAMERA_WIDTH') and hasattr(config, 'CAMERA_HEIGHT'):
                self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
                self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

            # Baca frame pertama untuk memastikan buffer tidak kosong
            self.grabbed, self.frame = self.stream.read()
            if not self.grabbed:
                log.error("Gagal menarik frame dari kamera.")
                return False

            self.stopped = False
            self._ref_count = 1
            # Gunakan daemon thread agar otomatis mati ketika main.py dihentikan (Ctrl+C)
            self.thread = threading.Thread(target=self._update, args=(), daemon=True)
            self.thread.start()

            # Beri jeda sejenak agar sensor kamera auto-adjust white balance & exposure
            time.sleep(1.0)
            return True

    def _update(self):
        """Looping baca frame secara kontinu."""
        consecutive_fail = 0
        while True:
            if self.stopped:
                self.stream.release()
                return

            grabbed, frame = self.stream.read()
            if grabbed:
                self.grabbed = True
                self.frame = frame
                consecutive_fail = 0
            else:
                self.grabbed = False
                consecutive_fail += 1
                if consecutive_fail > 30:
                    log.error("Kamera gagal baca frame >30x berturut-turut")
                    self.stopped = True
                    self.stream.release()
                    return

    def read(self):
        """
        Mengembalikan copy dari frame terbaru.
        Copy digunakan untuk mencegah data terkorupsi jika main thread
        memodifikasi array Numpy saat background thread menimpanya.
        """
        if self.frame is not None and self.grabbed:
            return self.frame.copy()
        return None

    def stop(self, force=False):
        """Hentikan thread dan rilis resource kamera."""
        with self._lock:
            if not force:
                self._ref_count -= 1
                if self._ref_count > 0:
                    log.info(f"Kamera ref_count={self._ref_count}, tidak dihentikan.")
                    return
                log.info("CameraStream: semua ref selesai, hentikan.")
            self.stopped = True
            if self.thread is not None:
                self.thread.join(timeout=2.0)

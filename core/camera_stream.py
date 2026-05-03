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
    def __init__(self):
        self.src = getattr(config, 'CAMERA_INDEX', getattr(config, 'CAMERA_SRC', 0))
        self.stream = None
        self.grabbed = False
        self.frame = None
        self.stopped = False
        self.thread = None

    def start(self) -> bool:
        """Inisiasi stream kamera dan jalankan thread latar belakang."""
        log.info(f"Memulai kamera dari sumber: {self.src}")
        self.stream = cv2.VideoCapture(self.src)

        # Set resolusi jika tersedia di config (optimasi untuk RPi)
        if hasattr(config, 'CAMERA_WIDTH') and hasattr(config, 'CAMERA_HEIGHT'):
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        if not self.stream.isOpened():
            log.error("Kamera gagal diakses.")
            return False

        # Baca frame pertama untuk memastikan buffer tidak kosong
        self.grabbed, self.frame = self.stream.read()
        if not self.grabbed:
            log.error("Gagal menarik frame dari kamera.")
            return False

        self.stopped = False
        # Gunakan daemon thread agar otomatis mati ketika main.py dihentikan (Ctrl+C)
        self.thread = threading.Thread(target=self._update, args=(), daemon=True)
        self.thread.start()
        
        # Beri jeda sejenak agar sensor kamera auto-adjust white balance & exposure
        time.sleep(1.0)
        return True

    def _update(self):
        """Looping baca frame secara kontinu."""
        while True:
            if self.stopped:
                self.stream.release()
                return
            
            # Tarik frame terbaru
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        """
        Mengembalikan copy dari frame terbaru.
        Copy digunakan untuk mencegah data terkorupsi jika main thread 
        memodifikasi array Numpy saat background thread menimpanya.
        """
        if self.frame is not None and self.grabbed:
            return self.frame.copy()
        return None

    def stop(self):
        """Hentikan thread dan rilis resource kamera."""
        self.stopped = True
        if self.thread is not None:
            self.thread.join()

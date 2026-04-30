"""
=============================================================
core/servo.py - Kontrol Pintu (Motor Servo)
=============================================================
"""
import time
import logging
import sys
import os

log = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

try:
    import RPi.GPIO as GPIO
    GPIO_OK = True
except ImportError:
    GPIO_OK = False
    log.warning("RPi.GPIO tidak terdeteksi. Berjalan di mode simulasi servo.")

BCM_TO_BOARD = {
    2: 3, 3: 5, 4: 7, 17: 11, 27: 13, 22: 15, 10: 19, 9: 21, 11: 23,
    0: 27, 5: 29, 6: 31, 13: 33, 19: 35, 26: 37, 14: 8, 15: 10, 18: 12,
    23: 16, 24: 18, 25: 22, 8: 24, 7: 26, 1: 28, 12: 32, 16: 36, 20: 38,
    21: 40
}

class DoorController:
    def __init__(self):
        self.pin = getattr(config, 'SERVO_PIN', 18)
        self.open_duty = getattr(config, 'SERVO_OPEN', 7.5)
        self.close_duty = getattr(config, 'SERVO_CLOSED', 2.5)
        self.pwm = None
        self._pin_gpio = self.pin

    def start(self):
        """Inisialisasi awal saat sistem booting."""
        log.info(f"Menginisiasi Servo pada pin GPIO {self.pin}")
        if GPIO_OK:
            cur = GPIO.getmode()
            if cur is None:
                GPIO.setmode(GPIO.BCM)
                self._pin_gpio = self.pin
            elif cur == GPIO.BCM:
                self._pin_gpio = self.pin
            elif cur == GPIO.BOARD:
                pin_board = getattr(config, "SERVO_PIN_BOARD", None)
                if pin_board is None:
                    pin_board = BCM_TO_BOARD.get(self.pin)
                if not pin_board:
                    raise RuntimeError(
                        "GPIO mode BOARD aktif. Set config.SERVO_PIN_BOARD."
                    )
                self._pin_gpio = pin_board
            else:
                raise RuntimeError(f"GPIO mode tidak dikenali: {cur}")
            GPIO.setwarnings(False)
            # Inisialisasi awal
            GPIO.setup(self._pin_gpio, GPIO.OUT)
            self.pwm = GPIO.PWM(self._pin_gpio, 50)
            self.pwm.start(0)
            self._set_duty(self.close_duty)
        return True

    def _set_duty(self, duty):
        """Set duty cycle langsung dengan proteksi re-setup pin."""
        if not GPIO_OK:
            return
            
        # PENTING: Force setup pin sebagai OUTPUT setiap kali akan digunakan.
        # Ini mencegah error jika modul lain (RFID/LCD) tidak sengaja melakukan cleanup.
        GPIO.setup(self._pin_gpio, GPIO.OUT)
        
        # Aktifkan sinyal
        GPIO.output(self._pin_gpio, True)
        if self.pwm is None: # Jaga-jaga jika objek PWM hilang
            self.pwm = GPIO.PWM(self._pin_gpio, 50)
            self.pwm.start(0)
            
        self.pwm.ChangeDutyCycle(duty)
        time.sleep(0.4)
        
        # Matikan sinyal untuk mencegah servo bergetar (jitter)
        GPIO.output(self._pin_gpio, False)
        self.pwm.ChangeDutyCycle(0)

    def open(self, duration=3):
        log.info(f"Membuka pintu selama {duration} detik.")
        self._set_duty(self.open_duty)
        time.sleep(duration)
        log.info("Menutup pintu kembali.")
        self._set_duty(self.close_duty)

    def cleanup(self):
        if GPIO_OK and self.pwm:
            self.pwm.stop()
            # Gunakan cleanup spesifik ke pin ini saja agar tidak mengganggu pin lain
            GPIO.cleanup(self._pin_gpio)



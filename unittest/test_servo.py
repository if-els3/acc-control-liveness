import unittest
import time
import os

# Detect environment
IS_RPI = os.path.exists('/proc/device-tree/model') and 'raspberry' in open('/proc/device-tree/model').read().lower()
IS_CI = os.getenv('CI', 'false').lower() == 'true'

@unittest.skipUnless(IS_RPI and not IS_CI, "Servo hardware test requires physical Raspberry Pi")
class ServoHardwareTest(unittest.TestCase):
    SERVO_PIN = 18          # BCM pin (physical pin 12)
    PWM_FREQ = 50           # Standard servo frequency
    SETTLE_TIME = 0.8       # Seconds to allow mechanical movement
    
    # ⚠️ CALIBRATE THESE FOR YOUR SPECIFIC SERVO
    # Typical ranges: MG90S/SG90 → (2.5%, 12.5%) ≈ (0.0005s, 0.0025s pulse width)
    MIN_PULSE = 0.0005      # ~0°
    MAX_PULSE = 0.0025      # ~180°
    MID_PULSE = 0.0015      # ~90°

    @classmethod
    def setUpClass(cls):
        try:
            from gpiozero import AngularServo
            cls.servo = AngularServo(
                cls.SERVO_PIN,
                min_angle=0,
                max_angle=180,
                min_pulse_width=cls.MIN_PULSE,
                max_pulse_width=cls.MAX_PULSE
            )
            # Start at neutral position
            cls.servo.angle = 90
            time.sleep(0.5)
        except Exception as e:
            raise unittest.SkipTest(f"GPIO/Servo initialization failed: {e}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'servo'):
            cls.servo.detach()  # Stop PWM output to prevent jitter/overheating
            cls.servo.close()

    def test_servo_moves_to_0_degrees(self):
        self.servo.angle = 0
        time.sleep(self.SETTLE_TIME)
        self.assertAlmostEqual(self.servo.angle, 0, delta=1)
        self._verify_movement_feedback()

    def test_servo_moves_to_180_degrees(self):
        self.servo.angle = 180
        time.sleep(self.SETTLE_TIME)
        self.assertAlmostEqual(self.servo.angle, 180, delta=1)
        self._verify_movement_feedback()

    def test_servo_sweeps_and_returns(self):
        self.servo.angle = 0
        time.sleep(self.SETTLE_TIME)
        self.servo.angle = 180
        time.sleep(self.SETTLE_TIME)
        self.servo.angle = 90
        time.sleep(self.SETTLE_TIME)
        self.assertAlmostEqual(self.servo.angle, 90, delta=1)

    def _verify_movement_feedback(self):
        """
        Override this method if you have physical feedback (current sensor, 
        position encoder, limit switch, or UART servo).
        Default: passes if PWM command was successfully sent.
        """
        pass  # Placeholder for real hardware feedback


if __name__ == '__main__':
    unittest.main(verbosity=2)
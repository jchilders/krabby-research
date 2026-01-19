import os
import serial
import time
import threading
import logging
from typing import Dict, Optional
from serial.tools import list_ports
from interfaces.joint_telemetry import JointTelemetry

import keyboard

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("KrabbySDK")


def _default_port():
    env_port = os.getenv("KRABBY_MCU_PORT")
    if env_port:
        return env_port
    for p in list_ports.comports():
        if "arduino" in (p.description or "").lower() or "arduino" in (p.manufacturer or "").lower():
            return p.device
    return "COM5" if os.name == "nt" else "/dev/ttyACM0"


class KrabbyMCUSDK:
    def __init__(self, port=None, baud=115200):
        self.port = port or _default_port()
        self.baud = baud
        self.ser = None
        self.running = False

        # Structured telemetry per joint
        self.joints: Dict[str, Optional[JointTelemetry]] = {}

        self.last_feedback_ts = None
        self.thread = None
        self._last_debug_log_ts = 0.0
        self.last_error = None
        self.last_cmd: Dict[str, Optional[float]] = {}

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)
            self.running = True
            self.last_error = None
            self.thread = threading.Thread(
                target=self._reader_loop, daemon=True)
            self.thread.start()
            logger.info(f"Connected to {self.port}")
            return True
        except Exception:
            logger.exception("Connection Failed")
            return False

    def _reader_loop(self):
        while self.running and self.ser.is_open:
            try:
                raw = self.ser.readline()
                try:
                    line = raw.decode('utf-8').strip()
                except UnicodeDecodeError as e:
                    logger.warning("Decode error on JT line (len=%d): %s raw=%s", len(raw), e, raw.hex())
                    line = raw.decode('utf-8', errors='ignore').strip()
                except Exception:
                    logger.exception("Decode error")
                    continue
                if not line:
                    continue

                if line.startswith("JT"):
                    self._parse_joint_line(line)
                    self.last_feedback_ts = time.time()
                # --- Log Firmware Messages (like Calibration) ---
                elif "Krabby" in line or "CAL" in line or "Saved" in line:
                    logger.info(f"[MCU] {line}")

            except (serial.SerialException, AttributeError) as e:
                if self.running:
                    logger.exception("Reader loop error")
                else:
                    logger.info("Reader loop stopped: %s", e)
                self.last_error = e
                self.running = False
                break
            except Exception as exc:
                logger.exception("Reader loop error")
                self.last_error = exc
                self.running = False
                break

    def _parse_joint_line(self, line: str):
        jts = JointTelemetry.parse_line(line)
        if not jts:
            return
        for jt in jts:
            self.joints[jt.name] = jt

        # Debug Log
        now = time.time()
        if logger.isEnabledFor(logging.DEBUG) and (now - self._last_debug_log_ts) >= 0.25:
            parts = []
            for name in sorted(self.joints.keys()):
                jt = self.joints.get(name)
                if jt:
                    parts.append(jt.format_compact(self.last_cmd.get(name)))
            if parts:
                logger.debug("JOINTS %s", ";".join(parts))
            self._last_debug_log_ts = now

    def send_command_joints(self, cmds_by_joint: Dict[str, float]):
        """
        Send commands keyed by joint name.
        """
        if not self.ser or not self.ser.is_open:
            return

        seq = []
        for key, raw_val in cmds_by_joint.items():
            val = max(0.0, min(1.0, raw_val))
            seq.append((key, val))
            self.last_cmd[key] = val

        parts = ["T"]
        for name, val in seq:
            parts.append(name)
            parts.append(f"{val:.3f}")

        cmd = " ".join(parts) + "\n"
        self.ser.write(cmd.encode('utf-8'))

        logger.info("CMD -> %s", " ".join(parts))

    def send_command_jog(self, joint_name: str, pwm: int):
        """ Send J<name> <pwm> (-255 to 255) """
        if not self.ser or not self.ser.is_open:
            return
        pwm = max(-255, min(255, int(pwm)))
        cmd = f"J{joint_name} {pwm}\n"
        self.ser.write(cmd.encode('utf-8'))
        # Optional: Uncomment if you want spammy logs for jogging
        # logger.debug(f"JOG -> {joint_name} {pwm}")

    def send_command_calibrate(self):
        if not self.ser or not self.ser.is_open:
            return
        self.ser.write(b"C\n")
        logger.info("CMD -> AUTO-CALIBRATE (C)")

    def send_command_joints_hold(self):
        """
        Send the 'H' command to hold all joints at their current positions.
        """
        if not self.ser or not self.ser.is_open:
            return
        self.ser.write(b"H\n")
        logger.info("CMD -> H")

    def wait_for_move(self, seconds):
        time.sleep(seconds)

    def close(self):
        self.running = False
        if self.ser:
            try:
                # Interrupt any blocking read so the reader thread can exit cleanly
                cancel_read = getattr(self.ser, "cancel_read", None)
                if callable(cancel_read):
                    cancel_read()
            except Exception:
                logger.debug("cancel_read failed during close", exc_info=True)
            try:
                self.ser.close()
            except Exception:
                logger.debug("Serial close failed", exc_info=True)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


if __name__ == "__main__":
    import sys
    # Enable Debug to see telemetry
    if "--debug" in sys.argv:
        logger.setLevel(logging.DEBUG)

    mcu = KrabbyMCUSDK()
    if mcu.connect():
        try:
            print("\n=== Krabby MCU Task 2 ===")
            print("1: Send Neutral (0.5)")
            print("2: AUTO-CALIBRATE (New!)")
            print("3: Manual Jog Mode (Requires 'keyboard' lib)")
            print("q: Quit")

            while True:
                choice = input("\nSelect > ").strip().lower()

                if choice == 'q':
                    break
                elif choice == '1':
                    logger.info("Sending Neutral...")
                    mcu.send_command_joints({
                        "LHY": 0.5, "RHY": 0.5,
                        "LHL": 0.5, "LKL": 0.5,
                        "RHL": 0.5, "RKL": 0.5,
                    })
                elif choice == '2':
                    print("WARNING: This will move ALL limbs to find limits.")
                    if input("Confirm (y/n): ") == 'y':
                        mcu.send_command_calibrate()
                elif choice == '3':
                    joint = input("Enter Joint (e.g. LHY): ").upper()
                    print(f"Holding W/S to move {joint}. ESC to exit.")

                    while True:
                        if keyboard.is_pressed('esc'):
                            mcu.send_command_jog(joint, 0)
                            break
                        elif keyboard.is_pressed('w'):
                            mcu.send_command_jog(joint, 255)  # Max speed
                        elif keyboard.is_pressed('s'):
                            mcu.send_command_jog(joint, -255)
                        else:
                            mcu.send_command_jog(joint, 0)
                        time.sleep(0.05)  # Prevent flooding
        except KeyboardInterrupt:
            mcu.send_command_joints_hold()
        finally:
            mcu.close()
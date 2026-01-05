import os
import serial
import time
import threading
import logging
from typing import Dict, Optional
from serial.tools import list_ports
from interfaces.joint_telemetry import JointTelemetry

JOINT_ORDER = ["LHY", "RHY", "LHL", "RHL", "LKL", "RKL"]

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
        self.joints: Dict[str, Optional[JointTelemetry]] = {key: None for key in JOINT_ORDER}

        self.last_feedback_ts = None
        self.thread = None
        self._last_debug_log_ts = 0.0
        self.last_error = None

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
                elif line.startswith("SAFETY:") or line.startswith("RUNAWAY:"):
                    logger.warning("MCU ALERT: %s", line)
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
            if jt.name in self.joints:
                self.joints[jt.name] = jt

        # Debug Log
        now = time.time()
        if logger.isEnabledFor(logging.DEBUG) and (now - self._last_debug_log_ts) >= 0.25:
            parts = []
            for key in JOINT_ORDER:
                jt = self.joints.get(key)
                if jt:
                    parts.append(jt.format_compact())
            if parts:
                logger.debug("JOINTS %s", ";".join(parts))
            self._last_debug_log_ts = now

    def send_command(self, cmds_by_joint: Dict[str, float]):
        """
        Send commands keyed by joint name.
        Yaw: [-1.0, 1.0]
        Linear: [0.0, 1.0]
        """
        if not self.ser or not self.ser.is_open:
            return

        seq = []
        for key in JOINT_ORDER:
            if key not in cmds_by_joint:
                continue
            val = cmds_by_joint[key]
            if key in ("LHY", "RHY"):
                val = max(-1.0, min(1.0, val))
            else:
                val = max(0.0, min(1.0, val))
            seq.append((key, val))

        count = len(seq)
        parts = ["T", str(count)]
        for name, val in seq:
            parts.append(name)
            parts.append(f"{val:.3f}")

        cmd = " ".join(parts) + "\n"
        self.ser.write(cmd.encode('utf-8'))

        logger.info("CMD -> %s", " ".join(parts))

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

    @property
    def latest_positions(self):
        """
        Returns the last known joint positions in controller order:
        [YawL, YawR, HipL, KneeL, HipR, KneeR].
        Defaults to neutral (0 yaw, 0.5 linear) if no telemetry yet.
        """
        # Mapping from MCU joint name -> index in the control vector
        name_to_idx = {
            "LHY": 0,
            "RHY": 1,
            "LHL": 2,
            "LKL": 3,
            "RHL": 4,
            "RKL": 5,
        }
        # Neutral defaults
        positions = [0.0, 0.0, 0.5, 0.5, 0.5, 0.5]
        for name in JOINT_ORDER:
            jt = self.joints.get(name)
            if not jt:
                continue
            idx = name_to_idx.get(name)
            if idx is not None:
                positions[idx] = jt.pos
        return positions


# --- MAIN TEST ---
if __name__ == "__main__":
    import sys
    # Enable Debug to see telemetry
    if "--debug" in sys.argv or True:  # Force debug for calibration visibility
        logger.setLevel(logging.DEBUG)

    mcu = KrabbyMCUSDK()
    if mcu.connect():
        try:
            logger.info("--- CALIBRATION MODE: Reading Telemetry ---")
            # Send neutral command (Yaw Center, Linear 50%)
            mcu.send_command({
                "LHY": 0.0,
                "RHY": 0.0,
                "LHL": 0.5,
                "LKL": 0.5,
                "RHL": 0.5,
                "RKL": 0.5,
            })

            # Loop to print values
            for _ in range(20):
                mcu.wait_for_move(0.5)

        except KeyboardInterrupt:
            mcu.send_command({
                "LHY": 0.0,
                "RHY": 0.0,
                "LHL": 0.5,
                "LKL": 0.5,
                "RHL": 0.5,
                "RKL": 0.5,
            })
        finally:
            mcu.close()

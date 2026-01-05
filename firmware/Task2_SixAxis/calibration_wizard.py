import time
import logging
import argparse
from krabby_mcu_six_axis import KrabbyMCUSDK

# Configure clean logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("CalibrationWizard")


def get_user_input(prompt):
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return 'q'


def calibration_wizard():
    print("\n=== Krabby-Uno Linear Actuator Calibration Wizard ===")
    print("SAFETY NOTICE: This tool uses 'Pulse Mode'. Motors will only move")
    print("for 0.2 seconds and then auto-stop. This prevents crashing.\n")

    mcu = KrabbyMCUSDK()
    if not mcu.connect():
        print("Error: Could not connect to Arduino.")
        return

    joint_key_by_choice = {
        '1': ("Hip Left", "LHL"),
        '2': ("Knee Left", "LKL"),
        '3': ("Hip Right", "RHL"),
        '4': ("Knee Right", "RKL"),
    }
    send_order = ["LHY", "RHY", "LHL", "LKL", "RHL", "RKL"]

    neutral_cmds = {
        "LHY": 0.0,
        "RHY": 0.0,
        "LHL": 0.5,
        "LKL": 0.5,
        "RHL": 0.5,
        "RKL": 0.5,
    }

    try:
        while True:
            print("\n--- Select Joint to Calibrate ---")
            print("1: Hip Left")
            print("2: Knee Left")
            print("3: Hip Right")
            print("4: Knee Right")
            print("q: Quit")

            choice = get_user_input("Select (1-4): ")
            if choice == 'q':
                break
            if choice not in joint_key_by_choice:
                continue

            joint_name, joint_key = joint_key_by_choice[choice]
            print(f"\n[Calibrating {joint_name}]")
            print("Controls:")
            print("  '+' or 'w' -> Extend (Pulse)")
            print("  '-' or 's' -> Retract (Pulse)")
            print("  'b'        -> Back to Menu")

            while True:
                if mcu.last_error:
                    print(f"\nReader error: {mcu.last_error}")
                    return
                # 1. Read latest Pot Value from telemetry map
                jt = mcu.joints.get(joint_key)
                current_pot = jt.pot if jt else 0

                # 2. Ask for command
                cmd = get_user_input(
                    f"POT VALUE: {current_pot} | Command (+/-): ")

                if cmd == 'b':
                    break

                # 3. Determine Pulse Direction
                # Build default commands keyed by joint name
                cmds_by_joint = {
                    "LHY": 0.0,
                    "RHY": 0.0,
                    "LHL": 0.5,
                    "LKL": 0.5,
                    "RHL": 0.5,
                    "RKL": 0.5,
                }
                # Clamp from latest telemetry where available
                for key in send_order:
                    jt_current = mcu.joints.get(key)
                    if not jt_current:
                        continue
                    if key in ("LHY", "RHY"):
                        cmds_by_joint[key] = max(-1.0, min(1.0, jt_current.pos))
                    else:
                        cmds_by_joint[key] = max(0.0, min(1.0, jt_current.pos))

                # Target gets 0.7 (extend) or 0.3 (retract)
                if cmd in ['+', 'w']:
                    cmds_by_joint[joint_key] = 0.7  # Low speed extend
                    print(f"  -> Extending {joint_name}...")
                elif cmd in ['-', 's']:
                    cmds_by_joint[joint_key] = 0.3  # Low speed retract
                    print(f"  -> Retracting {joint_name}...")
                else:
                    continue

                # 4. EXECUTE PULSE (The Safety Feature)
                # Move for only 0.2 seconds, then STOP.
                mcu.send_command(cmds_by_joint)
                time.sleep(5)
                mcu.send_command(neutral_cmds)  # Hard Stop
                time.sleep(0.1)  # Wait for comms to update pot value

    except KeyboardInterrupt:
        print("\nStopping...")
        mcu.send_command(neutral_cmds)
    finally:
        mcu.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Krabby-Uno Linear Actuator Calibration Wizard")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose MCU logs (including safety/runaway alerts)")
    args = parser.parse_args()

    if args.debug:
        # Elevate root + our loggers so debug actually emits
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        for h in root.handlers:
            h.setLevel(logging.DEBUG)
        logging.getLogger("KrabbySDK").setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    calibration_wizard()

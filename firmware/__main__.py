"""
Interactive MCU menu. Run with: python -m firmware [--debug]
"""
import sys
import logging
import time

import keyboard

from firmware.krabby_mcu import KrabbyMCUSDK, logger

# Joint order: LKL, LHL, LHY, RHY, RHL, RKL
JOINTS = ["LKL", "LHL", "LHY", "RHY", "RHL", "RKL"]
# Extend: Q W E R T Y  |  Retract: A S D F G H
EXTEND_KEYS = ["q", "w", "e", "r", "t", "y"]
RETRACT_KEYS = ["a", "s", "d", "f", "g", "h"]
JOG_PWM = 200


def main():
    if "--debug" in sys.argv:
        logger.setLevel(logging.DEBUG)

    mcu = KrabbyMCUSDK()
    if not mcu.connect():
        return

    try:
        print("\n=== Krabby MCU — Direct key control ===")
        print("Extend: Q W E R T Y  →  LKL LHL LHY RHY RHL RKL")
        print("Retract: A S D F G H  →  LKL LHL LHY RHY RHL RKL")
        print("1: Neutral (0.5)  |  2: Auto-calibrate  |  ESC: Quit")
        print()

        while True:
            if keyboard.is_pressed("esc"):
                break
            if keyboard.is_pressed("1"):
                mcu.send_command_joints({
                    "LHY": 0.5, "RHY": 0.5,
                    "LHL": 0.5, "LKL": 0.5,
                    "RHL": 0.5, "RKL": 0.5,
                })
                time.sleep(0.3)  # debounce
                continue
            if keyboard.is_pressed("2"):
                print("WARNING: This will move ALL limbs to find limits.")
                # Release key and wait for y/n in next iteration is messy; skip for now or use a one-shot
                mcu.send_command_calibrate()
                time.sleep(0.5)
                continue

            for i, joint in enumerate(JOINTS):
                if keyboard.is_pressed(EXTEND_KEYS[i]):
                    mcu.send_command_jog(joint, JOG_PWM)
                elif keyboard.is_pressed(RETRACT_KEYS[i]):
                    mcu.send_command_jog(joint, -JOG_PWM)
                else:
                    mcu.send_command_jog(joint, 0)

            time.sleep(0.04)  # ~25 Hz

    except KeyboardInterrupt:
        mcu.send_command_joints_hold()
    finally:
        mcu.close()


if __name__ == "__main__":
    main()

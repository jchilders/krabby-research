"""
Interactive MCU menu. Run with: python -m firmware [--debug]
"""
import sys
import logging
import time

import keyboard

from firmware.krabby_mcu import KrabbyMCUSDK, logger


def main():
    if "--debug" in sys.argv:
        logger.setLevel(logging.DEBUG)

    mcu = KrabbyMCUSDK()
    if not mcu.connect():
        return

    try:
        print("\n=== Krabby MCU Task 2 ===")
        print("1: Send Neutral (0.5)")
        print("2: AUTO-CALIBRATE (New!)")
        print("3: Manual Jog Mode (Requires 'keyboard' lib)")
        print("q: Quit")

        while True:
            choice = input("\nSelect > ").strip().lower()

            if choice == "q":
                break
            if choice == "1":
                logger.info("Sending Neutral...")
                mcu.send_command_joints({
                    "LHY": 0.5, "RHY": 0.5,
                    "LHL": 0.5, "LKL": 0.5,
                    "RHL": 0.5, "RKL": 0.5,
                })
            elif choice == "2":
                print("WARNING: This will move ALL limbs to find limits.")
                if input("Confirm (y/n): ") == "y":
                    mcu.send_command_calibrate()
            elif choice == "3":
                joint = input("Enter Joint (e.g. LHY): ").upper()
                print(f"Holding W/S to move {joint}. ESC to exit.")

                while True:
                    if keyboard.is_pressed("esc"):
                        mcu.send_command_jog(joint, 0)
                        break
                    if keyboard.is_pressed("w"):
                        mcu.send_command_jog(joint, 255)
                    elif keyboard.is_pressed("s"):
                        mcu.send_command_jog(joint, -255)
                    else:
                        mcu.send_command_jog(joint, 0)
                    time.sleep(0.05)
    except KeyboardInterrupt:
        mcu.send_command_joints_hold()
    finally:
        mcu.close()


if __name__ == "__main__":
    main()

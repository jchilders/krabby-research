"""CLI entry point for InputController - Pygame Test Version.

⚠️  TEMPORARY TEST FILE - WILL BE DELETED LATER ⚠️

This is a temporary test version of the InputController CLI that uses pygame
instead of the inputs library. This is necessary because the inputs library
does not work well with Bluetooth controllers on macOS. The inputs library is
designed for Linux's /dev/input/ interface, which macOS does not use.

This file is for testing purposes only on macOS and will be removed later.

Original file: controller/input/__main__.py
Modified: Uses input_controller_test_pygame instead of input_controller

KEY DIFFERENCES FROM controller/input/__main__.py:
---------------------------------------------------
1. Import:
   - Original: from controller.input.input_controller import InputController
   - This file: from controller.input.pygametemp.input_controller_test_pygame import InputController

2. macOS-Specific Code:
   - This file includes pygame.event.pump() call in monitor loop (required on macOS
     for pygame to update joystick state when called from main thread)

3. Troubleshooting Messages:
   - Original: Linux-focused (bluetoothctl, input group permissions)
   - This file: macOS-focused (System Preferences > Bluetooth, pygame installation)

CORE LOGIC IS IDENTICAL:
------------------------
- list_devices(), monitor_device(), main() functions work the same
- Argument parsing, output formatting, and monitoring loop logic are identical
- The only differences are platform-specific requirements and troubleshooting messages

Prerequisites:
    1. Install pygame library:
       pip install pygame
    
    2. Ensure your Pro Controller is paired via Bluetooth:
       - Press and hold the sync button on the Pro Controller until lights flash
       - On Mac: System Preferences > Bluetooth > Connect to "Pro Controller"

Usage:
    From the krabby-research directory, run:
    
    # List available gamepad devices
    python -m controller.input.pygametemp --list
    
    # Monitor the first available gamepad (default 50 Hz)
    python -m controller.input.pygametemp --monitor
    
    # Monitor a specific device by ID
    python -m controller.input.pygametemp --monitor 0
    
    # Monitor at a custom update rate (e.g., 100 Hz)
    python -m controller.input.pygametemp --monitor --rate 100
    
    # Monitor a specific device at a custom rate
    python -m controller.input.pygametemp --monitor 0 --rate 100

The --monitor command displays:
    - Selected legs when buttons are pressed
    - Control actions (hip up/down, knee out/in, hip yaw) when sticks are moved

Leg selection controls:
    - LT (without LB): Front Left (FL)
    - LB (without LT): Rear Left (RL)
    - LS button: Middle Left (ML)
    - RS button: Middle Right (MR)
    - RT (without RB): Front Right (FR)
    - RB (without RT): Rear Right (RR)
    - LT + LB: FL, RL, MR (tripod combo left)
    - RT + RB: FR, RR, ML (tripod combo right)

Control actions (when legs are selected):
    - Left stick Y: Hip up/down
    - Left stick X: Knee out/in
    - Right stick Y: Hip yaw forward/back
"""
import argparse
import logging
import signal
import sys
import time
from typing import Optional

from controller.input.pygametemp.input_controller_test_pygame import InputController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def list_devices() -> None:
    """List available gamepad devices."""
    devices = InputController.list_devices()
    
    if not devices:
        print("No gamepad devices found.")
        print("\nTroubleshooting:")
        print("1. Make sure your gamepad is connected (USB or Bluetooth)")
        print("2. On macOS, ensure your Pro Controller is paired via Bluetooth:")
        print("   - Press and hold the sync button on the Pro Controller")
        print("   - Go to System Preferences > Bluetooth")
        print("   - Click Connect next to 'Pro Controller'")
        print("3. Make sure pygame is installed: pip install pygame")
        return
    
    print(f"Found {len(devices)} gamepad device(s):\n")
    for i, device in enumerate(devices):
        print(f"  [{i}] {device['name']}")
        print(f"      Path: {device['path']}")
        print()


def monitor_device(device_id: Optional[int], update_rate_hz: float) -> None:
    """Monitor a gamepad device and stream normalized states.
    
    Args:
        device_id: Device ID to monitor (None = first available).
        update_rate_hz: Update rate for monitoring (default: 50.0).
    """
    controller = InputController.get_instance()
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, shutting down...")
        controller.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Check for available devices first
        devices = InputController.list_devices()
        if not devices:
            logger.error("No gamepad devices found. Please connect a gamepad and try again.")
            logger.info("Run with --list to see available devices.")
            sys.exit(1)
        
        # Validate device_id if provided
        if device_id is not None:
            if device_id < 0 or device_id >= len(devices):
                logger.error(
                    f"Invalid device ID: {device_id}. "
                    f"Available devices: 0-{len(devices)-1}"
                )
                sys.exit(1)
            device_name = devices[device_id]["name"]
        else:
            device_id = 0
            device_name = devices[0]["name"] if devices else "Unknown"
        
        # Start controller
        logger.info(f"Starting InputController for device [{device_id}]: {device_name}")
        controller.start(device_id=device_id, update_rate_hz=update_rate_hz)
        
        # Give controller a moment to initialize
        time.sleep(0.1)
        
        # Check if controller thread is still alive
        if controller._thread is None or not controller._thread.is_alive():
            logger.error("Controller thread failed to start or exited immediately.")
            logger.error("Check the logs above for error messages.")
            controller.stop()
            sys.exit(1)
        
        if not controller._running:
            logger.error("Controller is not running. Check for initialization errors above.")
            controller.stop()
            sys.exit(1)
        
        logger.info(f"Monitoring device [{device_id}]: {device_name}")
        logger.info(f"Update rate: {update_rate_hz} Hz")
        logger.info("\nLeg Selection Controls:")
        logger.info("  LT (without LB)     → Front Left (FL)")
        logger.info("  LB (without LT)     → Rear Left (RL)")
        logger.info("  LS button           → Middle Left (ML)")
        logger.info("  RS button           → Middle Right (MR)")
        logger.info("  RT (without RB)     → Front Right (FR)")
        logger.info("  RB (without RT)     → Rear Right (RR)")
        logger.info("  LT + LB             → FL, RL, MR (tripod left)")
        logger.info("  RT + RB             → FR, RR, ML (tripod right)")
        logger.info("\nPress Ctrl+C to stop")
        logger.info("Waiting for controller input...\n")
        
        # Import pygame for event pumping (required on macOS)
        import pygame
        
        # Monitor loop - print only when legs are selected or control actions occur
        check_interval = 1.0 / update_rate_hz  # Check at update rate
        last_legs = set()
        last_hip_up_down = 0.0
        last_knee_out_in = 0.0
        last_hip_yaw = 0.0
        action_threshold = 0.02  # Threshold for detecting control actions
        
        while controller._running and (controller._thread is None or controller._thread.is_alive()):
            # Pump events in main thread to update joystick state (required on macOS)
            pygame.event.pump()
            
            control_data = controller.get_control_data()
            
            # Check if leg selection changed
            legs_changed = control_data.selected_legs != last_legs
            
            # Check if control actions changed (hip up/down, knee out/in, hip yaw)
            hip_up_down = control_data.hip_up_down
            knee_out_in = control_data.knee_out_in
            hip_yaw = control_data.hip_yaw
            
            action_changed = (
                abs(hip_up_down - last_hip_up_down) > action_threshold or
                abs(knee_out_in - last_knee_out_in) > action_threshold or
                abs(hip_yaw - last_hip_yaw) > action_threshold
            )
            
            # Print when legs are selected or control actions occur
            should_print = False
            output_parts = []
            
            # Always show leg selection changes
            if legs_changed:
                if control_data.selected_legs:
                    legs_list = sorted([leg.value for leg in control_data.selected_legs])
                    legs_str = ", ".join(legs_list)
                    output_parts.append(f"Selected Legs: [{legs_str}]")
                else:
                    output_parts.append("Selected Legs: [none]")
                should_print = True
            
            # Show control actions if legs are selected and actions changed
            if control_data.selected_legs and action_changed:
                actions = []
                if abs(hip_up_down) > action_threshold:
                    direction = "up" if hip_up_down > 0 else "down"
                    actions.append(f"Hip {direction}: {abs(hip_up_down):.2f}")
                if abs(knee_out_in) > action_threshold:
                    direction = "out" if knee_out_in > 0 else "in"
                    actions.append(f"Knee {direction}: {abs(knee_out_in):.2f}")
                if abs(hip_yaw) > action_threshold:
                    direction = "forward" if hip_yaw > 0 else "back"
                    actions.append(f"Hip yaw {direction}: {abs(hip_yaw):.2f}")
                
                if actions:
                    action_str = " | ".join(actions)
                    if not legs_changed:
                        legs_list = sorted([leg.value for leg in control_data.selected_legs])
                        legs_str = ", ".join(legs_list)
                        output_parts.append(f"Legs [{legs_str}]")
                    output_parts.append(action_str)
                    should_print = True
            
            if should_print:
                print(" | ".join(output_parts))
            
            # Update tracking variables
            if legs_changed:
                last_legs = control_data.selected_legs.copy()
            if action_changed:
                last_hip_up_down = hip_up_down
                last_knee_out_in = knee_out_in
                last_hip_yaw = hip_yaw
            
            time.sleep(check_interval)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during monitoring: {e}", exc_info=True)
    finally:
        controller.stop()
        print()  # New line after monitoring output
        logger.info("Monitoring stopped")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="InputController CLI for gamepad input handling (Pygame Test Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available gamepad devices
  python -m controller.input.pygametemp --list
  
  # Monitor first available gamepad at 50 Hz
  python -m controller.input.pygametemp --monitor
  
  # Monitor specific device at 100 Hz
  python -m controller.input.pygametemp --monitor 0 --rate 100

Note: This is a temporary test version for macOS. The selected legs will be
clearly displayed when you press buttons on your Pro Controller.
        """,
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available gamepad devices",
    )
    
    # Use a sentinel to distinguish between "not provided" and "provided without value"
    MONITOR_SENTINEL = object()
    
    parser.add_argument(
        "--monitor",
        type=int,
        nargs="?",
        const=MONITOR_SENTINEL,
        default=None,
        metavar="DEVICE_ID",
        help="Monitor a gamepad device and stream normalized states. "
             "If DEVICE_ID is not provided, uses first available device.",
    )
    
    parser.add_argument(
        "--rate",
        type=float,
        default=50.0,
        metavar="HZ",
        help="Update rate in Hz for monitoring (default: 50.0, range: 1-1000)",
    )
    
    args = parser.parse_args()
    
    # Validate rate
    if args.rate < 1.0 or args.rate > 1000.0:
        logger.error("Update rate must be between 1.0 and 1000.0 Hz")
        sys.exit(1)
    
    # Execute command
    if args.list:
        list_devices()
    elif args.monitor is not None:
        # If monitor was provided without a value, it will be MONITOR_SENTINEL
        # If monitor was provided with a value, it will be that int value
        device_id = None if args.monitor is MONITOR_SENTINEL else args.monitor
        monitor_device(device_id, args.rate)
    else:
        # Default to list if no command specified
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
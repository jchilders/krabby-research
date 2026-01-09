"""CLI entry point for InputController.

This module provides a command-line interface for the InputController singleton,
allowing users to list available gamepad devices and monitor controller input
in real-time.

Prerequisites:
    1. Install the inputs library:
       pip install inputs
    
    2. On Linux, ensure proper permissions:
       sudo usermod -a -G input $USER
       # Log out and back in for changes to take effect

Usage:
    From the krabby-research directory, run:
    
    # List available gamepad devices
    python -m controller.input --list
    
    # Monitor the first available gamepad (default 50 Hz)
    python -m controller.input --monitor
    
    # Monitor a specific device by ID
    python -m controller.input --monitor 0
    
    # Monitor at a custom update rate (e.g., 100 Hz)
    python -m controller.input --monitor --rate 100
    
    # Monitor a specific device at a custom rate
    python -m controller.input --monitor 0 --rate 100

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

Note: If you get import errors, ensure you're in the krabby-research directory
or that it's in your PYTHONPATH.
"""
import argparse
import logging
import signal
import sys
import time
from typing import Optional

from controller.input.input_controller import InputController

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
        print("2. On Linux, you may need to pair Bluetooth devices:")
        print("   - Use 'bluetoothctl' to pair your controller")
        print("   - Or use GUI Bluetooth settings")
        print("3. Check device permissions:")
        print("   - Ensure user is in 'input' group: sudo usermod -a -G input $USER")
        print("   - Log out and back in for group changes to take effect")
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
                  Note: Currently, the inputs library reads from all gamepads.
                  This parameter is accepted for future compatibility.
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
        
        # Monitor loop - print only when legs are selected or control actions occur
        check_interval = 1.0 / update_rate_hz  # Check at update rate
        last_legs = set()
        last_hip_up_down = 0.0
        last_knee_out_in = 0.0
        last_hip_yaw = 0.0
        action_threshold = 0.02  # Threshold for detecting control actions
        
        while controller._running and (controller._thread is None or controller._thread.is_alive()):
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
        description="InputController CLI for gamepad input handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available gamepad devices
  python -m controller.input --list
  
  # Monitor first available gamepad at 50 Hz
  python -m controller.input --monitor
  
  # Monitor specific device at 100 Hz
  python -m controller.input --monitor 0 --rate 100
        """,
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available gamepad devices",
    )
    
    parser.add_argument(
        "--monitor",
        type=int,
        nargs="?",
        const=None,
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
        monitor_device(args.monitor, args.rate)
    else:
        # Default to list if no command specified
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()


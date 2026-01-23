"""CLI entry point for InputController.

This module provides a command-line interface for the InputController singleton,
allowing users to list available gamepad devices and monitor controller input
in real-time.

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
        logger.info("Press Ctrl+C to stop\n")
        
        # Monitor loop - print controller state
        check_interval = 1.0 / update_rate_hz
        last_state = None

        # Import pygame for event pumping 
        import pygame
        
        while controller._running and (controller._thread is None or controller._thread.is_alive()):
            # Pump events in main thread to update joystick state 
            pygame.event.pump()

            state = controller.get_state()
            
            # Only print if state changed
            if state != last_state:
                print(f"LT={state.LT} LB={state.LB} LS={state.LS} RS={state.RS} RT={state.RT} RB={state.RB} | "
                      f"LX={state.LX:.3f} LY={state.LY:.3f} RX={state.RX:.3f} RY={state.RY:.3f}")
                last_state = state
            
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

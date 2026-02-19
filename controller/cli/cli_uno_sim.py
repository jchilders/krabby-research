#!/usr/bin/env python3
"""CLI entry point for krabby-uno-sim: gamepad → IsaacSim HAL (simulation path).

Connects to the IsaacSim HAL server via ZMQ TCP and sends joint commands from
the gamepad. Start the IsaacSim HAL server first (e.g. from Isaac or
controller/scripts/demo/test_gamepad_to_isaacsim_hal.py).

Usage:
  krabby-uno-sim
  krabby-uno-sim --observation_endpoint tcp://127.0.0.1:5555 --command_endpoint tcp://127.0.0.1:5556
  krabby-uno-sim --InputController 0 --rate 50
"""

import argparse
import logging
import signal
import sys
import time

from controller.control_loop import ControlLoop, ControlLoopConfig, ControlMode
from hal.client.config import HalClientConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _initialize_pygame() -> None:
    """Initialize Pygame in main thread for InputController (e.g. macOS)."""
    try:
        import pygame
        if not pygame.get_init():
            pygame.init()
            logger.info("Pygame initialized in main thread")
    except Exception:
        pass


def _pump_pygame_events(last_pump_time: list) -> None:
    """Pump Pygame events so joystick state updates (main thread)."""
    try:
        import pygame
        if pygame.get_init():
            t = time.time()
            if t - last_pump_time[0] > 0.016:
                pygame.event.pump()
                last_pump_time[0] = t
    except Exception:
        pass


def main() -> int:
    """Run control loop client (IsaacSim HAL) until Ctrl+C."""
    parser = argparse.ArgumentParser(
        description="krabby-uno-sim: gamepad → IsaacSim HAL (ZMQ TCP). Start IsaacSim HAL server first.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--observation_endpoint",
        type=str,
        default="tcp://127.0.0.1:5555",
        help="HAL observation endpoint (client connects)",
    )
    parser.add_argument(
        "--command_endpoint",
        type=str,
        default="tcp://127.0.0.1:5556",
        help="HAL command endpoint (client connects)",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        metavar="ID",
        dest="device_id",
        help="Gamepad device ID (from python -m controller.input --list). Default: first device.",
    )
    parser.add_argument(
        "--InputController",
        type=int,
        default=None,
        metavar="ID",
        dest="input_controller_id",
        help="Alias for --device-id (gamepad device ID).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=50.0,
        help="Input controller update rate (Hz)",
    )
    args = parser.parse_args()

    device_id = args.device_id if args.device_id is not None else args.input_controller_id

    _initialize_pygame()

    hal_client_config = HalClientConfig(
        observation_endpoint=args.observation_endpoint,
        command_endpoint=args.command_endpoint,
    )
    control_loop_config = ControlLoopConfig(
        mode=ControlMode.INPUT_CONTROLLER_ISAACSIM,
        hal_client_config=hal_client_config,
        input_controller_device_id=device_id,
        input_controller_update_rate_hz=args.rate,
    )

    control_loop = ControlLoop(control_loop_config)
    control_loop.start()

    logger.info(
        "Control loop client started (INPUT_CONTROLLER_ISAACSIM). "
        "observation=%s, command=%s. Press Ctrl+C to stop.",
        args.observation_endpoint,
        args.command_endpoint,
    )
    logger.info(
        "Gamepad: LT/LB/LS/RS/RT/RB = legs; left stick Y/X = hip/knee; right stick Y = hip yaw"
    )

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        logger.info("Received interrupt, stopping...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    last_pump = [0.0]
    try:
        while running:
            _pump_pygame_events(last_pump)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        control_loop.stop()
        logger.info("Control loop client stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())

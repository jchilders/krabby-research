"""Gamepad-only Jetson HAL server (no camera, no inference).

Binds ZMQ TCP for observation and command endpoints. Requires firmware and
MCU hardware; exits with non-zero if firmware is not available or hardware
is not detected. Loop: get_joint_command() -> apply_command() only.

Usage (two-process E2E, from krabby-research):
  Terminal 1: python controller/scripts/jetson/main_gamepad_only.py
  Terminal 2: python controller/scripts/jetson/run_gamepad_to_krabby_client.py
"""

import argparse
import logging
import os
import signal
import sys
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "../../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION
from hal.server import HalServerConfig
from hal.server.jetson import JetsonHalServer
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for gamepad-only HAL server."""
    parser = argparse.ArgumentParser(
        description="Jetson HAL server for gamepad teleop (no camera, no inference). Exits if firmware or MCU not available.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--observation_bind",
        type=str,
        default="tcp://*:6001",
        help="ZMQ observation endpoint (server binds)",
    )
    parser.add_argument(
        "--command_bind",
        type=str,
        default="tcp://*:6002",
        help="ZMQ command endpoint (server binds)",
    )
    parser.add_argument(
        "--mcu-port",
        type=str,
        default=None,
        help="Serial port for MCU (e.g. /dev/ttyACM0). Default from firmware.",
    )
    parser.add_argument(
        "--mcu-baud",
        type=int,
        default=115200,
        help="MCU baud rate",
    )
    args = parser.parse_args()

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        logger.info("Received interrupt signal, stopping...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    hal_server = None

    try:
        config = HalServerConfig(
            observation_bind=args.observation_bind,
            command_bind=args.command_bind,
        )
        observation_dimensions = PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions(
            KRABBY_HEX_DEFINITION
        )
        action_dim = PARKOUR_MODEL_OBSERVATION_DEFINITION.action_dim
        hal_server = JetsonHalServer(
            config,
            observation_dimensions=observation_dimensions,
            action_dim=action_dim,
            robot_definition=KRABBY_HEX_DEFINITION,
            mcu_port=args.mcu_port,
            mcu_baud=args.mcu_baud,
            mcu_auto_connect=True,
        )
        hal_server.initialize()

        # Strict hardware check: terminate if firmware or MCU not available
        if hal_server._mcusdk is None:
            logger.error(
                "Firmware or KrabbyMCUSDK not available. "
                "Install firmware package and ensure MCU hardware is connected. Exiting."
            )
            sys.exit(1)
        if not hal_server._mcusdk.is_connected():
            logger.error(
                "MCU hardware not detected or not connected. "
                "Check serial port and wiring. Exiting."
            )
            sys.exit(1)

        logger.info(
            "Gamepad-only HAL server initialized (ZMQ TCP). "
            "observation=%s, command=%s. Waiting for joint commands...",
            args.observation_bind,
            args.command_bind,
        )

        while running:
            command = hal_server.get_joint_command(timeout_ms=50)
            if command is not None:
                hal_server.apply_command(command)
            time.sleep(0.001)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("Gamepad-only HAL server failed: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        if hal_server is not None:
            hal_server.close()
            logger.info("Gamepad-only HAL server closed")


if __name__ == "__main__":
    main()

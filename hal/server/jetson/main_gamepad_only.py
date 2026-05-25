"""HAL server entry point for gamepad-only operation.

Runs the Jetson HAL server with TCP ZMQ endpoints so that a separate
krabby-uno container (or any HAL client) can connect and send joint
commands without requiring a model checkpoint.

No observation loop — gamepad-only mode only needs get_joint_command()
and apply_command(). Exits with non-zero if MCU hardware is not detected.

Usage:
    krabby-hal-server-gamepad-only [--observation_bind ENDPOINT]
                                   [--command_bind ENDPOINT]
                                   [--mcu-port PORT]
                                   [--mcu-baud BAUD]
"""

import argparse
import logging
import signal
import sys
import time

from hal.server import HalServerConfig
from hal.server.jetson import JetsonHalServer
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION
from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jetson HAL server for gamepad-only operation (no inference checkpoint required)"
    )
    parser.add_argument("--observation_bind", type=str, default="tcp://*:6001")
    parser.add_argument("--command_bind", type=str, default="tcp://*:6002")
    parser.add_argument("--mcu-port", type=str, default=None, help="Serial port for MCU (e.g. /dev/ttyACM0)")
    parser.add_argument("--mcu-baud", type=int, default=115200)
    args = parser.parse_args()

    model_definition = PARKOUR_MODEL_OBSERVATION_DEFINITION
    robot_definition = KRABBY_HEX_DEFINITION
    observation_dimensions = model_definition.get_observation_dimensions(robot_definition)

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        logger.info("Received interrupt signal, stopping...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    hal_server = None
    try:
        hal_server = JetsonHalServer(
            HalServerConfig(
                observation_bind=args.observation_bind,
                command_bind=args.command_bind,
            ),
            observation_dimensions=observation_dimensions,
            action_dim=model_definition.action_dim,
            robot_definition=robot_definition,
            mcu_port=args.mcu_port,
            mcu_baud=args.mcu_baud,
            mcu_auto_connect=True,
        )
        hal_server.initialize()

        if hal_server._mcusdk is None or not hal_server._mcusdk.is_connected():
            logger.error("MCU not available — check firmware and wiring. Exiting.")
            sys.exit(1)

        logger.info(
            "HAL server started in gamepad-only mode "
            "(observation=%s, command=%s). Run `krabby uno` to connect.",
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
        logger.error("Failed to run gamepad-only HAL server: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        if hal_server:
            hal_server.close()


if __name__ == "__main__":
    main()

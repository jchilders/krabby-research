"""IsaacSim MCU SDK interface for applying joint commands.

This module provides a standardized SDK interface for applying joint commands
to IsaacSim environments.
"""

import logging
import statistics

from hal.client.data_structures.hardware import JointCommand

logger = logging.getLogger(__name__)

# Only expose these names for "from hal.server.isaac.isaacsim_mcusdk import *"
__all__ = ["IsaacSimMCUSDK"]

# Match the small-angle neutral threshold used by the Jetson KrabbyMCUSDK
# (see hal/server/jetson/krabby_mcusdk.py::NEUTRAL_RAD_THRESHOLD).
NEUTRAL_RAD_THRESHOLD = 0.01


class IsaacSimMCUSDK:
    """Standardized SDK interface for applying joint commands to IsaacSim.

    Accepts JointCommand (same as Jetson MCU SDK). Order and timestamps come from the command.

    The SDK handles:
    - Processing normalized PWM values (-1.0 to 1.0) from the command
    - Logging commands in Isaac's preferred joint format
    - Validating that command has expected keys
    """

    def __init__(self):
        logger.info("IsaacSimMCUSDK initialized")

    def apply_command(
        self,
        command: JointCommand,
        num_envs: int = 1,
    ) -> dict[str, float]:
        """Apply joint command to IsaacSim environment.

        Processes normalized PWM values from the command for prismatic joints.
        Logs the command in Isaac's preferred joint format for debugging.

        Args:
            command: JointCommand with positions keyed by joint names (from to_positions_dict()).
                     Values are normalized PWM: -1.0 to 1.0 (0.0 = hold).
            num_envs: Number of environments (currently unused, kept for API compatibility).

        Returns:
            Command as dict (command.to_positions_dict()) so the caller can convert to array/tensor.

        Raises:
            ValueError: If command is empty or invalid.
        """
        cmd_dict = command.to_positions_dict()
        if not cmd_dict:
            raise ValueError("command must be non-empty")

        order = command.joint_names
        present = [name for name in order if name in cmd_dict]
        if not present:
            raise ValueError("command must contain at least one joint name from robot definition")

        # Apply a small-angle neutral threshold, mirroring the Jetson KrabbyMCUSDK
        # behaviour: |rad| <= NEUTRAL_RAD_THRESHOLD is treated as zero. This keeps
        # very small stick deflections from producing motion in simulation.
        thresholded_cmd: dict[str, float] = {}
        for name in order:
            if name in cmd_dict:
                rad = cmd_dict[name]
                thresholded_cmd[name] = 0.0 if abs(rad) <= NEUTRAL_RAD_THRESHOLD else rad

        # Joint control is applied by the env via env.step(action); the action term
        # (e.g. DelayedJointPositionAction) converts these values to position targets.
        # We return the (thresholded) command dict for the caller to convert to an array/tensor.

        # Hot path: skip building debug strings unless this logger is actually at DEBUG.
        if logger.isEnabledFor(logging.DEBUG):
            joint_values_str = ", ".join(
                f"{name}={thresholded_cmd[name]:.4f}" for name in order if name in thresholded_cmd
            )
            logger.debug(
                "IsaacSimMCUSDK: Applying joint command "
                "(timestamp_ns=%s, observation_timestamp_ns=%s): %s",
                command.timestamp_ns,
                command.observation_timestamp_ns,
                joint_values_str,
            )
            vals = [thresholded_cmd[name] for name in order if name in thresholded_cmd]
            if vals:
                stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
                logger.debug(
                    "IsaacSimMCUSDK: Joint command stats - "
                    "min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                    min(vals),
                    max(vals),
                    statistics.mean(vals),
                    stdev,
                )

        return thresholded_cmd

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

        # Joint control is applied by the env via env.step(action); the action term
        # (e.g. DelayedJointPositionAction) converts normalized values to position
        # targets. This module returns the normalized command dict for the caller
        # to pass to env.step().

        # Log command in Isaac's preferred joint format
        # Format: joint positions as comma-separated name=value pairs.
        # Order follows robot definition (e.g. hexapod: FL, FR, ML, MR, RL, RR, 3 DOF per leg).
        joint_values_str = ", ".join(
            f"{name}={cmd_dict[name]:.4f}" for name in order if name in cmd_dict
        )
        logger.debug(
            f"IsaacSimMCUSDK: Applying joint command "
            f"(timestamp_ns={command.timestamp_ns}, observation_timestamp_ns={command.observation_timestamp_ns}): "
            f"{joint_values_str}"
        )

        # Log summary statistics
        vals = [cmd_dict[name] for name in order if name in cmd_dict]
        if vals:
            stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
            logger.debug(
                f"IsaacSimMCUSDK: Joint command stats - "
                f"min={min(vals):.4f}, max={max(vals):.4f}, mean={statistics.mean(vals):.4f}, std={stdev:.4f}"
            )

        return cmd_dict

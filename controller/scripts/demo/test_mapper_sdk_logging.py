#!/usr/bin/env python3
"""Simple test to verify mapper → SDK logging.

This script tests the GamepadToIsaacSimHALMapper and IsaacSimMCUSDK integration
without requiring a full HAL server setup. It demonstrates that gamepad control
data is correctly converted to joint commands and logged in Isaac's preferred format.

Usage:
    python controller/scripts/demo/test_mapper_sdk_logging.py

    Requirements:
    - No gamepad required (uses simulated control data)
    
"""

import os
import sys

# Add project root to Python path so we can import controller, hal, etc.
# Script is in controller/scripts/demo/, so go up 3 levels to reach project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "../../..")
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import numpy as np

from controller.input.state import GamepadControlData, LegIdentifier, ControllerState
from controller.mappers.gamepad_to_isaacsim_hal_mapper import GamepadToIsaacSimHALMapper
from hal.server.isaac.isaacsim_mcusdk import IsaacSimMCUSDK

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable DEBUG for IsaacSimMCUSDK to see joint command logs
logging.getLogger('hal.server.isaac.isaacsim_mcusdk').setLevel(logging.DEBUG)
logging.getLogger('controller.mappers.gamepad_to_isaacsim_hal_mapper').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


def main():
    """Main test function."""
    logger.info("=" * 80)
    logger.info("Testing GamepadToIsaacSimHALMapper → IsaacSimMCUSDK logging")
    logger.info("=" * 80)
    
    # Create mapper and SDK
    logger.info("Initializing mapper and SDK...")
    mapper = GamepadToIsaacSimHALMapper(
        hip_up_down_scale=0.3,
        knee_out_in_scale=0.3,
        hip_yaw_scale=0.2,
    )
    sdk = IsaacSimMCUSDK()
    
    # Test Case 1: Single leg (Front Left) with axis movements
    logger.info("\n" + "-" * 80)
    logger.info("Test Case 1: Front Left leg - hip up, knee in, yaw forward")
    logger.info("-" * 80)
    
    state1 = ControllerState()
    control_data1 = GamepadControlData(
        selected_legs={LegIdentifier.FRONT_LEFT},
        hip_up_down=0.5,      # Move hip up
        knee_out_in=-0.3,     # Move knee in
        hip_yaw=0.2,          # Yaw forward
        raw_state=state1,
    )
    
    # Map to joint command
    joint_cmd1 = mapper.map(control_data1)
    
    # Apply via SDK (this will log in Isaac's format)
    logger.info("Applying command via IsaacSimMCUSDK...")
    action1 = sdk.apply_command(joint_cmd1)
    logger.info(f"Action tensor shape: {action1.shape}")
    
    # Test Case 2: Multiple legs (combo selection)
    logger.info("\n" + "-" * 80)
    logger.info("Test Case 2: Combo selection (FL, RL, MR) - hip down, knee out")
    logger.info("-" * 80)
    
    state2 = ControllerState()
    control_data2 = GamepadControlData(
        selected_legs={
            LegIdentifier.FRONT_LEFT,
            LegIdentifier.REAR_LEFT,
            LegIdentifier.MIDDLE_RIGHT,
        },
        hip_up_down=-0.4,     # Move hip down
        knee_out_in=0.6,      # Move knee out
        hip_yaw=-0.1,         # Yaw backward
        raw_state=state2,
    )
    
    joint_cmd2 = mapper.map(control_data2)
    logger.info("Applying command via IsaacSimMCUSDK...")
    action2 = sdk.apply_command(joint_cmd2)
    logger.info(f"Action tensor shape: {action2.shape}")
    
    # Test Case 3: No legs selected (should maintain current positions)
    logger.info("\n" + "-" * 80)
    logger.info("Test Case 3: No legs selected (should maintain positions)")
    logger.info("-" * 80)
    
    state3 = ControllerState()
    control_data3 = GamepadControlData(
        selected_legs=set(),  # No legs selected
        hip_up_down=0.0,
        knee_out_in=0.0,
        hip_yaw=0.0,
        raw_state=state3,
    )
    
    joint_cmd3 = mapper.map(control_data3)
    logger.info("Applying command via IsaacSimMCUSDK...")
    action3 = sdk.apply_command(joint_cmd3)
    logger.info(f"Action tensor shape: {action3.shape}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Test complete! Check logs above for IsaacSimMCUSDK joint command output.")
    logger.info("=" * 80)
    logger.info("\nExpected log format:")
    logger.info("  IsaacSimMCUSDK: Applying joint command (...): ")
    logger.info("    FL_hip_yaw=0.0000, FL_hip_pitch=0.1500, FL_knee=-0.0900, ...")
    logger.info("  IsaacSimMCUSDK: Joint command stats - min=..., max=..., mean=..., std=...")


if __name__ == "__main__":
    main()

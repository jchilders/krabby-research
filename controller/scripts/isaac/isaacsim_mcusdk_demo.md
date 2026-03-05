# Controller Demo Scripts

This directory contains demonstration scripts for testing the controller → HAL → IsaacSimMCUSDK integration pipeline.

**Joystick → Isaac Sim demo:** See [isaacsim_demo_runbook.md](isaacsim_demo_runbook.md) for pairing, launch, and leg selection.

## Scripts


### 1. `test_gamepad_to_isaacsim_hal.py`

**Purpose**: Full end-to-end test that verifies the complete pipeline from gamepad input through the ControlLoop, HAL client/server, to IsaacSimMCUSDK logging.

**What it tests**:
- InputController reading gamepad events. 
- ControlLoop wiring and callback system
- GamepadToIsaacSimHALMapper conversion
- HALClient → HALServer communication
- IsaacSimMCUSDK logging in Isaac's preferred format

**Requirements**:
- A gamepad/joystick connected (Bluetooth or USB)
- The `pygame` library: `pip install pygame`

**Usage**:
```bash
# From the krabby-research root directory
python controller/scripts/isaac/test_gamepad_to_isaacsim_hal.py
```

**Expected Output**: 
When you press buttons on the gamepad, you should see 
 **IsaacSimMCUSDK debug logs** showing joint commands in Isaac's format. When the test is run, it keeps on logging the joint commands in Isaac's format continuously. 
To see specific action, you can press a specific gamepad button(like LB, etc) and then move the joystick(like left stick, right stick, etc) to see the corresponding joint commands in Isaac's format.
   ```
   2026-01-15 11:50:07,234 - hal.server.isaac.isaacsim_mcusdk - DEBUG - IsaacSimMCUSDK: Applying joint command (timestamp_ns=1768495807227837000, observation_timestamp_ns=1768495807227837000): FL_hip_yaw=0.0000, FL_hip_pitch=0.0000, FL_knee=0.0000, FR_hip_yaw=0.0000, FR_hip_pitch=0.0000, FR_knee=0.0000, ML_hip_yaw=0.0000, ML_hip_pitch=0.0000, ML_knee=0.0000, MR_hip_yaw=0.0000, MR_hip_pitch=0.0000, MR_knee=0.0000, RL_hip_yaw=0.0000, RL_hip_pitch=0.0000, RL_knee=0.0000, RR_hip_yaw=0.0000, RR_hip_pitch=0.0000, RR_knee=0.0000
  ...
  -- This is after pressing a button and moving the sticks
  2026-01-15 11:50:51,912 - hal.server.isaac.isaacsim_mcusdk - DEBUG - IsaacSimMCUSDK: Applying joint command (timestamp_ns=1768495851908335000, observation_timestamp_ns=1768495851908335000): FL_hip_yaw=0.0000, FL_hip_pitch=0.0000, FL_knee=0.0000, FR_hip_yaw=0.0000, FR_hip_pitch=0.0000, FR_knee=0.0000, ML_hip_yaw=-0.0160, ML_hip_pitch=-0.5408, ML_knee=-2.0000, MR_hip_yaw=0.0000, MR_hip_pitch=0.0000, MR_knee=0.0000, RL_hip_yaw=-0.1159, RL_hip_pitch=1.7897, RL_knee=0.0174, RR_hip_yaw=0.0000, RR_hip_pitch=0.0000, RR_knee=0.0000
  2026-01-15 11:50:51,912 - hal.server.isaac.isaacsim_mcusdk - DEBUG - IsaacSimMCUSDK: Joint command stats - min=-2.0000, max=1.7897, mean=-0.0481, std=0.6441
   ```

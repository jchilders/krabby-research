# Controller Demo Scripts

This directory contains demonstration scripts for testing the controller → HAL → IsaacSimMCUSDK integration pipeline.

## Scripts

### 1. `test_mapper_sdk_logging.py`

**Purpose**: Simple test to verify that `GamepadToIsaacSimHALMapper` correctly converts gamepad control data to joint commands, and that `IsaacSimMCUSDK` logs them in Isaac's preferred joint format.

**What it tests**:
- Mapper conversion from `GamepadControlData` to `JointCommand`
- SDK logging of joint commands in Isaac's format (e.g., `FL_hip_yaw=0.0000, FL_hip_pitch=0.1500, ...`)
- Multiple test cases: single leg, multiple legs, no legs selected

**Requirements**:
- No gamepad required (uses simulated control data)
- Python dependencies: `numpy`, `torch`

**Usage**:
```bash
# From the krabby-research root directory
python controller/scripts/demo/test_mapper_sdk_logging.py
```

**Expected Output**:
This will print log statements from IsaacSimMCUSDK.

---

### 2. `test_gamepad_to_isaacsim_hal.py`

**Purpose**: Full end-to-end test that verifies the complete pipeline from gamepad input through the ControlLoop, HAL client/server, to IsaacSimMCUSDK logging.

**What it tests**:
- InputController reading gamepad events. 
- ControlLoop wiring and callback system
- GamepadToIsaacSimHALMapper conversion
- HALClient → HALServer communication
- IsaacSimMCUSDK logging in Isaac's preferred format

**Requirements**:
- A gamepad/joystick connected (Bluetooth or USB)
- The `inputs` library: `pip install inputs`
- On macOS, you may need to use pygame instead (see `controller/input/pygametemp/`).

**Usage**:
```bash
# From the krabby-research root directory
python controller/scripts/demo/test_gamepad_to_isaacsim_hal.py
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


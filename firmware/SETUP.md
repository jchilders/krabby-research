# Krabby-Uno Task 2: Six-Axis Leg Controller

## Overview

This firmware drives a full leg pair (Left & Right) consisting of **6 Motors**.

## Prerequisites

- **Hardware:**
  - Arduino Mega 2560
  - **6x** BTS7960 43A H-Bridge Drivers
  - **12x** Resistors (10kΩ) for Current Sense protection
  - 12V Power Supply
- **Software:**
  - Python 3
  - Libraries: `pip install pyserial keyboard`
  - Arduino IDE

---

## 1. Hardware Wiring (New Sequential Layout)

**CRITICAL:** The pin mapping has changed to a clean sequential order (Pins 2-13).
**Polarity Note:**
* **RPWM / R_EN** = Right (Extend/Forward) -> **First Pin** listed.
* **LPWM / L_EN** = Left (Retract/Reverse) -> **Second Pin** listed.

| Leg       | Joint         | Driver PWM (RPWM, LPWM) | Driver EN (R_EN, L_EN) | Potentiometer | Current Sense |
| :-------- | :------------ | :---------------------- | :--------------------- | :------------ | :------------ |
| **Left**  | Yaw (LHY)     | D2 (R),  D3 (L)         | D22 (R), D23 (L)       | A0            | A6            |
|           | Hip (LHL)     | D4 (R),  D5 (L)         | D24 (R), D25 (L)       | A1            | A7            |
|           | Knee (LKL)    | D6 (R),  D7 (L)         | D26 (R), D27 (L)       | A2            | A8            |
| **Right** | Yaw (RHY)     | D8 (R),  D9 (L)         | D28 (R), D29 (L)       | A3            | A9            |
|           | Hip (RHL)     | D10 (R), D11 (L)        | D30 (R), D31 (L)       | A4            | A10           |
|           | Knee (RKL)    | D12 (R), D13 (L)        | D32 (R), D33 (L)       | A5            | A11           |

**Note:** Ensure all Enable (EN) pins are connected and providing 5V, otherwise calibration will get 'lost' as it will not know where joint positions are.

---

## 2. Installation

### 2.1 Serial RX buffer (leader board, 3-board setup)

When using the **leader** board that forwards telemetry from left/right followers, the default 64-byte serial RX buffer can overflow and drop bytes (corrupt or missing actuators in telemetry). Use a **256-byte** RX buffer for Serial2/Serial3 on the leader.

**You do not flash the core separately.** The Arduino “core” is just C++ source that is compiled *with* your sketch into a single firmware image. Change the buffer size, then build and upload as usual.

**Arduino IDE**

- **Option A – One-time edit (survives until you update the AVR board package):**  
  Open the core file (path similar to):
  - Windows: `%LOCALAPPDATA%\Arduino15\packages\arduino\hardware\avr\1.8.7\cores\arduino\HardwareSerial.h`
  - macOS: `~/Library/Arduino15/packages/arduino/hardware/avr/1.8.7/cores/arduino/HardwareSerial.h`  
  Find the block that sets `SERIAL_RX_BUFFER_SIZE` (e.g. `#define SERIAL_RX_BUFFER_SIZE 64`) and change **64** to **256**. Save. Then compile and upload your sketch as usual.

- **Option B – Build flag via platform override:**  
  In the same `avr` package folder (e.g. `.../packages/arduino/hardware/avr/1.8.7/`), create or edit `platform.local.txt` and add:
  ```text
  compiler.c.extra_flags=-DSERIAL_RX_BUFFER_SIZE=256
  compiler.cpp.extra_flags=-DSERIAL_RX_BUFFER_SIZE=256
  ```
  so the define is applied when the core and your sketch are compiled. Then build/upload as usual.

**PlatformIO**

In `platformio.ini` for the board that acts as the leader, add:

```ini
build_flags = -DSERIAL_RX_BUFFER_SIZE=256
```

Then build and upload. No core file edit needed.

**Follower-only boards** do not need this change; only the board that runs `forwardFullLines` (the leader on USB) benefits from the larger buffer.

### 2.2 Telemetry format (wire protocol)

Telemetry is sent as **newline-terminated lines** over serial. The Python side parses each line into a **dict of joint id → values** using `JointTelemetry` in `interfaces/joint_telemetry.py`.

- **Line format:** `<ROLE>; <name> <pos> <pot> <current> <enL> <enR> <pwmL> <pwmR> <saf>; <name> ...; ...`
- **Role prefix:** One of `FRONT`, `UNKNOWN`, `LEFT`, `RIGHT` (no semicolon inside the role).
- **Segment format:** Each joint segment is 9 space-separated values: joint name, position (0–1), pot raw, current raw, enable L/R, PWM L/R, safety.
- **Example:** `FRONT; FLHY 0.723 740 694 0 0 0 0 0;FLHL 0.723 740 691 ...`

On the Arduino side, telemetry is built in **telemetry_manager.h** (struct `JointTelemetry`, `appendTo()`). The old standalone `joint_telemetry.h` was removed; all telemetry formatting and collection lives in `telemetry_manager.h` and `actuator_manager.h`.

### 2.3 Upload firmware and Python

1.  **Upload Firmware:**
    - Open `arduino/arduino.ino` (or `arduino/controller.ino`) in Arduino IDE
    - Upload to each Arduino Mega (all three use the same sketch; role is elected at runtime).
2.  **Setup Python SDK:**
    - Ensure you have the `interfaces/` folder next to your script.
    - Install dependencies: `pip -r requirements.txt`

---

## 3. Usage Guide

Run the interactive MCU menu from the **krabby-research** directory:

```bash
# On Linux/Mac, you may need sudo for keyboard access
python -m firmware
```

For troubleshooting (verbose telemetry):
```bash
python -m firmware --debug
```


### Feature 1: Auto-Calibration (Run Once)
The robot now calibrates itself automatically and saves limits to EEPROM.
 - Select Option 2 (Auto-Calibrate) in the menu.
 - Stand Back: The robot will perform the safety sequence:
    - Yaw Left -> Yaw Right -> Hip Up -> Knee Out -> Knee In -> Hip Down.
 - Result: Limits are saved. You do not need to repeat this after rebooting.

### Feature 2: Manual Jog Mode
 - Select Option 3 (Jog Mode).
 - Type the joint name (e.g., LHY or LKL).
 - Hold 'W' to Extend, Hold 'S' to Retract.
 - Release keys to stop immediately.

### Feature 3: Neutral Pose
 - Select Option 1.
 - Robot moves all joints to center (0.5). Useful to verify calibration accuracy.
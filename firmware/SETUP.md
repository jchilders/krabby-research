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

1.  **Upload Firmware:**
    - Open `arduino/controller.ino` using Arduino IDE
    - Upload to Arduino Mega.
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
# Krabby-Uno Task 2: Six-Axis Leg Controller

## Overview

This firmware drives a full leg pair (Left & Right) consisting of **6 Motors**:

- **2x Hip Yaw Motors:** DC Motor + Encoder
- **4x Linear Actuators:** DC Motor + Potentiometer (Limited travel: Hip/Knee)

## Prerequisites

- **Hardware:**
  - Arduino Mega 2560
  - **6x** BTS7960 43A H-Bridge Drivers
  - **12x** Resistors (4.7kΩ to 10kΩ) for Current Sense protection
  - 12V Power Supply
- **Software:**
  - Python 3 (`pip install pyserial`)
  - Arduino IDE

## 1. Hardware Wiring

**CRITICAL SAFETY WARNING:** You MUST place a resistor (4.7k-10kΩ) between every BTS7960 `IS` pin and the Arduino Analog pin. Connecting 12V drivers directly to Arduino Analog pins can destroy the MCU.

### A. Left Leg

| Joint         | Driver Pin        | Arduino Pin | Function          |
| :------------ | :---------------- | :---------- | :---------------- |
| **Yaw Left**  | **Encoder A**     | **D18**     | Interrupt         |
|               | Encoder B         | D19         | Direction         |
|               | PWM (Fwd/Rev)     | D46, D45    | Drive             |
|               | EN (Fwd/Rev)      | D22, D23    | Enable            |
|               | IS (R/L)          | A4, A5      | Current Sense     |
| **Hip Left**  | **Potentiometer** | **A0**      | Position Feedback |
|               | PWM (Up/Dn)       | D4, D5      | Drive             |
|               | EN (Up/Dn)        | D26, D27    | Enable            |
|               | IS (R/L)          | A8, A9      | Current Sense     |
| **Knee Left** | **Potentiometer** | **A1**      | Position Feedback |
|               | PWM (Out/In)      | D6, D7      | Drive             |
|               | EN (Out/In)       | D28, D29    | Enable            |
|               | IS (R/L)          | A10, A11    | Current Sense     |

### B. Right Leg

| Joint          | Driver Pin        | Arduino Pin | Function          |
| :------------- | :---------------- | :---------- | :---------------- |
| **Yaw Right**  | **Encoder A**     | **D20**     | Interrupt         |
|                | Encoder B         | D21         | Direction         |
|                | PWM (Fwd/Rev)     | D2, D3      | Drive             |
|                | EN (Fwd/Rev)      | D24, D25    | Enable            |
|                | IS (R/L)          | A6, A7      | Current Sense     |
| **Hip Right**  | **Potentiometer** | **A2**      | Position Feedback |
|                | PWM (Up/Dn)       | D8, D9      | Drive             |
|                | EN (Up/Dn)        | D30, D31    | Enable            |
|                | IS (R/L)          | A12, A13    | Current Sense     |
| **Knee Right** | **Potentiometer** | **A3**      | Position Feedback |
|                | PWM (Out/In)      | D10, D11    | Drive             |
|                | EN (Out/In)       | D32, D33    | Enable            |
|                | IS (R/L)          | A14, A15    | Current Sense     |

## 2. Installation

1.  **Configure Firmware:** Open `Six_Axis_Controller.ino`.
2.  **Calibrate:** _Before_ running the robot, follow instructions in `CALIBRATION.md` to set your potentiometer limits.
3.  **Upload:** Flash the code to the Arduino Mega.
4.  **Run SDK:**
    ```bash
    python3 krabby_mcu_six_axis.py --debug
    ```

## 3. TODOs

1. Right now the POT values jump around alot when POT is not moving, causing motor to activate sometimes. Needs to use a running average and discard bad values.
2. Right now the IS values jump around alot (not as much as POT), put running average on this as well.
3. Add an auto calibration that will retract/extend each joint one at a time until it sees no more movement on POT for ~0.25s or whatever is appropriate, then it stores that as the end maxStop value, and make this value persist between reboots. Repeat in this order to safely calibrate all limbs:
Yaw Left -> Yaw Right, move to center
Retract Hip (so hip is upright)
Extend Knee (so leg is pointing in the air)
Retract Knee (so leg is pointing to the ground)
Extend Hip (so leg is tucked under body)
Move all three to 50 (standard 'standing' pose on left side)
Repeat on Right side

4. Tweak manual calibration so you can just hold the +/- and it will move the selected joint, and releasing will stop moving, instead of right now it is awkward to have to do +, then enter, +, enter.
5. Move pins so the limbs are in order on the arduino (i.e can we move 45/46 to D0/1 or 12/13?). Also order all pins so its LHY,LHL,LKL,RHY,RHL,RKL for PWM, IS, POT, and EN
6. Review my changes to see if there any improvements or mistakes
7. I think once these are done Task2 is fully completed.  
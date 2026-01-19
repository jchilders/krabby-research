#pragma once
#include <Arduino.h>
#include <EEPROM.h>
#include "joint_telemetry.h"
#include "command.h"

// Linear actuator controller (w/ potentiometer feedback)
class LinearActuator
{
public:
    struct ControlConfig
    {
        int pwmRampStep = 5;
        int rampIntervalMs = 10;
        int pwmDeadband = 20;
        int pwmErrDeadband = 10;
        float Kp = 2.0;

        ControlConfig() = default;
        ControlConfig(int rampStep, int intervalMs, int deadband, int errDeadband, float kp)
            : pwmRampStep(rampStep), rampIntervalMs(intervalMs), pwmDeadband(deadband), pwmErrDeadband(errDeadband), Kp(kp) {}
    };

    const char *name;
    const int pinPwmR, pinPwmL, pinEnR, pinEnL, pinIS, pinPot;

    // Calibration Limits
    int minStop = 0;
    int maxStop = 1023;

    // State
    int currentPwm = 0;
    int currentTarget = 512; // Start center
    bool manualMode = false; // True if Jogging (TODO 4)
    unsigned long lastRampTime = 0;

    // TODO 1 & 2: Smoothing Variables
    float avgPot = 0.0;
    float avgIS = 0.0;
    const float ALPHA_POT = 0.15; // Smoothing factor (0.1 - 1.0)
    const float ALPHA_IS = 0.10;

    LinearActuator(const char *n, int pR, int pL, int eR, int eL, int isPin, int pot)
        : name(n), pinPwmR(pR), pinPwmL(pL), pinEnR(eR), pinEnL(eL), pinIS(isPin), pinPot(pot) {}

    void setControlConfig(const ControlConfig &cfg) { controlConfig = cfg; }

    void init()
    {
        pinMode(pinPwmR, OUTPUT);
        pinMode(pinPwmL, OUTPUT);
        pinMode(pinEnR, OUTPUT);
        pinMode(pinEnL, OUTPUT);
        pinMode(pinIS, INPUT);
        pinMode(pinPot, INPUT);
        digitalWrite(pinEnR, HIGH);
        digitalWrite(pinEnL, HIGH); // Enable Driver

        // Initialize averaging
        avgPot = analogRead(pinPot);
        avgIS = analogRead(pinIS);
        currentTarget = (int)avgPot;
    }

    // TODO 1: Signal Smoothing
    void updateSensors()
    {
        int rawPot = analogRead(pinPot);
        int rawIS = analogRead(pinIS);

        // Exponential Moving Average
        avgPot = (avgPot * (1.0 - ALPHA_POT)) + (rawPot * ALPHA_POT);
        avgIS = (avgIS * (1.0 - ALPHA_IS)) + (rawIS * ALPHA_IS);
    }

    // Returns smoothed, normalized position [0.0, 1.0]
    float getPos()
    {
        float range = maxStop - minStop;
        if (range == 0)
            return 0.5;
        return ((int)avgPot - minStop) / range;
    }

    int getRawPos() { return (int)avgPot; } // Returns smoothed RAW value

    void setTarget(float val)
    {
        val = constrain(val, 0.0, 1.0);
        currentTarget = minStop + (int)(val * (maxStop - minStop));
        manualMode = false; // Target command cancels Jog
    }

    // TODO 4: Manual Drive (Jog)
    void manualDrive(int pwm)
    {
        manualMode = true;
        // Clamp PWM
        pwm = constrain(pwm, -255, 255);
        // Direct drive, bypassing PID but respecting ramp in next update()
        // For responsiveness, we set currentPwm directly if stopping
        if (pwm == 0)
        {
            currentPwm = 0;
            driveActuator(0, 0);
            // When stopping jog, lock new position as target
            currentTarget = getRawPos();
        }
        else
        {
            // For movement, we let the update() loop handle ramping if desired,
            // but simpler to just drive directly for Jog to feel responsive.
            driveActuator(pwm, 0);
            currentPwm = pwm;
        }
    }

    void update()
    {
        updateSensors(); // Always update sensors

        if (manualMode)
            return; // Skip PID if jogging

        int error = currentTarget - getRawPos();
        if (abs(error) < controlConfig.pwmErrDeadband)
            error = 0;

        int desiredPwm = (int)(error * controlConfig.Kp);
        desiredPwm = constrain(desiredPwm, -255, 255);

        // Ramping Logic
        if (millis() - lastRampTime >= (unsigned long)controlConfig.rampIntervalMs)
        {
            lastRampTime = millis();
            if (currentPwm < desiredPwm)
            {
                currentPwm += controlConfig.pwmRampStep;
                if (currentPwm > desiredPwm)
                    currentPwm = desiredPwm;
            }
            else if (currentPwm > desiredPwm)
            {
                currentPwm -= controlConfig.pwmRampStep;
                if (currentPwm < desiredPwm)
                    currentPwm = desiredPwm;
            }
        }
        driveActuator(currentPwm, controlConfig.pwmDeadband);
    }

    // Helper to detect stall (used in calibration)
    // Returns true if motor is powered but position hasn't changed for 'timeout' ms
    bool isStalled(unsigned long timeout)
    {
        static int lastPos = -1;
        static unsigned long lastMoveTime = 0;

        if (abs(currentPwm) < 50)
        { // Not trying to move
            lastMoveTime = millis();
            return false;
        }

        if (abs(getRawPos() - lastPos) > 2)
        { // Moved
            lastPos = getRawPos();
            lastMoveTime = millis();
            return false;
        }

        if (millis() - lastMoveTime > timeout)
            return true;
        return false;
    }

    JointTelemetry getTelemetry(const char *code)
    {
        JointTelemetry jt;
        jt.name = code;
        jt.pos = getPos();
        jt.pot = (int)avgPot;    // Return Smoothed
        jt.current = (int)avgIS; // Return Smoothed
        jt.enL = digitalRead(pinEnL);
        jt.enR = digitalRead(pinEnR);
        jt.pwmL = currentPwm < 0 ? abs(currentPwm) : 0;
        jt.pwmR = currentPwm > 0 ? currentPwm : 0;
        jt.saf = 0;
        return jt;
    }

private:
    void driveActuator(int pwm, int pwmDeadband)
    {
        if (abs(pwm) < pwmDeadband)
        {
            digitalWrite(pinEnR, LOW);
            digitalWrite(pinEnL, LOW);
            analogWrite(pinPwmR, 0);
            analogWrite(pinPwmL, 0);
        }
        else if (pwm < 0)
        {
            digitalWrite(pinEnR, HIGH);
            digitalWrite(pinEnL, HIGH);
            analogWrite(pinPwmR, 0);
            analogWrite(pinPwmL, abs(pwm));
        }
        else
        {
            digitalWrite(pinEnR, HIGH);
            digitalWrite(pinEnL, HIGH);
            analogWrite(pinPwmR, pwm);
            analogWrite(pinPwmL, 0);
        }
    }

    ControlConfig controlConfig;
};

class ActuatorManager
{
public:
    ActuatorManager(LinearActuator **actsArray, size_t actsCount)
        : actuators(actsArray), count(actsCount) {}

    void initAll()
    {
        for (size_t i = 0; i < count; i++)
            actuators[i]->init();
    }

    // TODO 4: Handle Jog Command
    void handleJog(String name, int pwm)
    {
        // Simple O(N) lookup
        for (size_t i = 0; i < count; i++)
        {
            if (String(actuators[i]->name) == name)
            {
                actuators[i]->manualDrive(pwm);
                return;
            }
        }
    }

    void updateAll()
    {
        if (calState != CAL_IDLE)
        {
            updateCalibration(); // Run calibration logic instead of normal PID
        }
        else
        {
            for (size_t i = 0; i < count; i++)
                actuators[i]->update();
        }
    }

    void applyCommands(const Command *cmds, size_t cmdCount)
    {
        for (size_t i = 0; i < cmdCount; i++)
        {
            const auto &cmd = cmds[i];
            for (size_t j = 0; j < count; j++)
            {
                if (cmd.name == actuators[j]->name)
                {
                    actuators[j]->setTarget(cmd.val);
                    break;
                }
            }
        }
    }

    void holdAll()
    {
        for (size_t i = 0; i < count; i++)
        {
            actuators[i]->setTarget(actuators[i]->getPos());
        }
    }

    String serializeTelemetry() const
    {
        String out;
        out.reserve(256);
        out += "JT ";
        for (size_t i = 0; i < count; i++)
        {
            actuators[i]->getTelemetry(actuators[i]->name).appendTo(out);
            if (i + 1 < count)
                out += ';';
        }
        if (calState != CAL_IDLE)
        {
            out += " CAL_MODE"; // Flag to GUI
        }
        return out;
    }

    // ==================================================
    // TODO 3: AUTO-CALIBRATION & PERSISTENCE
    // ==================================================
    enum CalState
    {
        CAL_IDLE,
        CAL_START,
        CAL_YAW_L_MIN,
        CAL_YAW_L_MAX,
        CAL_YAW_L_CENTER,
        CAL_YAW_R_MIN,
        CAL_YAW_R_MAX,
        CAL_YAW_R_CENTER,
        // Left Leg Sequence
        CAL_LHL_MIN,
        CAL_LKL_MAX,
        CAL_LKL_MIN,
        CAL_LHL_MAX,
        // Right Leg Sequence
        CAL_RHL_MIN,
        CAL_RKL_MAX,
        CAL_RKL_MIN,
        CAL_RHL_MAX,
        CAL_FINISH
    };

    CalState calState = CAL_IDLE;
    unsigned long stateTimer = 0;

    // Struct to save to EEPROM
    struct CalData
    {
        int minVals[6];
        int maxVals[6];
        int magic; // 0xDEADBEEF to check validity
    };

    void startAutoCalibration()
    {
        calState = CAL_START;
        stateTimer = millis();
        Serial.println("Starting Auto-Calibration Sequence...");
    }

    void updateCalibration()
    {
        // Helper lambda to get actuator by index (Hardcoded order: LHY, LHL, LKL, RHY, RHL, RKL)
        // 0=LHY, 1=LHL, 2=LKL, 3=RHY, 4=RHL, 5=RKL
        auto drive = [&](int idx, int pwm)
        { actuators[idx]->manualDrive(pwm); };
        auto isStalled = [&](int idx)
        { return actuators[idx]->isStalled(250); }; // 250ms stall check
        auto saveMin = [&](int idx)
        { actuators[idx]->minStop = actuators[idx]->getRawPos(); };
        auto saveMax = [&](int idx)
        { actuators[idx]->maxStop = actuators[idx]->getRawPos(); };

        // Simple State Machine
        switch (calState)
        {
        case CAL_START:
            calState = CAL_YAW_L_MIN;
            break;

        // --- YAWS FIRST ---
        case CAL_YAW_L_MIN:
            drive(0, -150); // Retract LHY
            if (isStalled(0))
            {
                saveMin(0);
                calState = CAL_YAW_L_MAX;
            }
            break;
        case CAL_YAW_L_MAX:
            drive(0, 150); // Extend LHY
            if (isStalled(0))
            {
                saveMax(0);
                calState = CAL_YAW_L_CENTER;
            }
            break;
        case CAL_YAW_L_CENTER:
            drive(0, 0); // Stop
            calState = CAL_YAW_R_MIN;
            break;

        case CAL_YAW_R_MIN:
            drive(3, -150); // Retract RHY
            if (isStalled(3))
            {
                saveMin(3);
                calState = CAL_YAW_R_MAX;
            }
            break;
        case CAL_YAW_R_MAX:
            drive(3, 150); // Extend RHY
            if (isStalled(3))
            {
                saveMax(3);
                calState = CAL_YAW_R_CENTER;
            }
            break;
        case CAL_YAW_R_CENTER:
            drive(3, 0);
            calState = CAL_LHL_MIN;
            break;

        // --- LEFT LEG SEQUENCE (Hip Up -> Knee Out -> Knee In -> Hip Down) ---
        case CAL_LHL_MIN: // Hip Retract (Up)
            drive(1, -200);
            if (isStalled(1))
            {
                saveMin(1);
                calState = CAL_LKL_MAX;
            }
            break;
        case CAL_LKL_MAX: // Knee Extend (Out)
            drive(2, 200);
            if (isStalled(2))
            {
                saveMax(2);
                calState = CAL_LKL_MIN;
            }
            break;
        case CAL_LKL_MIN: // Knee Retract (In)
            drive(2, -200);
            if (isStalled(2))
            {
                saveMin(2);
                calState = CAL_LHL_MAX;
            }
            break;
        case CAL_LHL_MAX: // Hip Extend (Tuck)
            drive(1, 200);
            if (isStalled(1))
            {
                saveMax(1);
                calState = CAL_RHL_MIN;
            }
            break;

        // --- RIGHT LEG SEQUENCE ---
        case CAL_RHL_MIN:
            drive(4, -200);
            if (isStalled(4))
            {
                saveMin(4);
                calState = CAL_RKL_MAX;
            }
            break;
        case CAL_RKL_MAX:
            drive(5, 200);
            if (isStalled(5))
            {
                saveMax(5);
                calState = CAL_RKL_MIN;
            }
            break;
        case CAL_RKL_MIN:
            drive(5, -200);
            if (isStalled(5))
            {
                saveMin(5);
                calState = CAL_RHL_MAX;
            }
            break;
        case CAL_RHL_MAX:
            drive(4, 200);
            if (isStalled(4))
            {
                saveMax(4);
                calState = CAL_FINISH;
            }
            break;

        case CAL_FINISH:
            // Stop all
            for (int i = 0; i < 6; i++)
                actuators[i]->manualDrive(0);
            saveCalibration(); // Write to EEPROM
            calState = CAL_IDLE;
            Serial.println("CALIBRATION COMPLETE & SAVED.");
            break;

        default:
            calState = CAL_IDLE;
            break;
        }
    }

    void saveCalibration()
    {
        CalData data;
        data.magic = 0xDEADBEEF;
        for (int i = 0; i < count; i++)
        {
            data.minVals[i] = actuators[i]->minStop;
            data.maxVals[i] = actuators[i]->maxStop;
        }
        EEPROM.put(0, data);
        Serial.println("Limits saved to EEPROM.");
    }

    void loadCalibration()
    {
        CalData data;
        EEPROM.get(0, data);
        if (data.magic == 0xDEADBEEF)
        {
            for (int i = 0; i < count && i < 6; i++)
            {
                actuators[i]->minStop = data.minVals[i];
                actuators[i]->maxStop = data.maxVals[i];
            }
            Serial.println("Calibration loaded from EEPROM.");
        }
        else
        {
            Serial.println("No EEPROM calibration found. Using defaults.");
        }
    }

private:
    LinearActuator **actuators;
    size_t count;
};
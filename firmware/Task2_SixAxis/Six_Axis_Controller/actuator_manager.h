#pragma once
#include <Arduino.h>
#include "joint_telemetry.h"
#include "command.h"

// Linear actuator controller (w/ potentiometer feedback)
class LinearActuator
{
public:
    struct ControlConfig
    {
        int pwmRampStep = 5;     // how much to change PWM per ramp step, higher will cause motor to accelerate faster
        int rampIntervalMs = 10; // time in millis between ramp steps
        int pwmDeadband = 20;    // PWM values below this are treated as zero, to avoid motor "creep"
        int pwmErrDeadband = 10; // Position error below which no PWM is applied, to avoid motor oscillation near target
        float Kp = 2.0;          // proportional gain applied to position error to derive desired PWM, PWM is set to max((targetPos - currentPos) * Kp, 255)

        ControlConfig() = default;
        ControlConfig(int rampStep, int intervalMs, int deadband, int errDeadband, float kp)
            : pwmRampStep(rampStep),
              rampIntervalMs(intervalMs),
              pwmDeadband(deadband),
              pwmErrDeadband(errDeadband),
              Kp(kp) {}
    };

    const char *name;  // Short joint name/id (e.g. "LHY" = "Left Hip Yaw")
    // PIN ASSIGNMENTS
    const int pinPwmR; // Sending on PWM_R defines desired motor voltage in the right/extend direction (note: Only send one of PWM_R or PWM_L at a time to avoid motor chatter/damage)
    const int pinPwmL; // Sending on PWM_L defines desired motor voltage in the left/retract direction (note: Only send one of PWM_R or PWM_L at a time to avoid motor chatter/damage)
    const int pinEnR;  // Sending HIGH to EN_R enables 'right' half of H-Bridge (note: both EN_R and EN_L should be HIGH to enable motor drive)
    const int pinEnL;  // Sending HIGH to EN_L enables 'left' half of H-Bridge (note: both EN_R and EN_L should be HIGH to enable motor drive)
    const int pinIS;   // Analog current sense pin, reads motor current from H-Bridge as a value between 0 (0 Amp) and 1023 (Max motor Amps, e.g. ~8A @ 12V for common linear actuators)
    const int pinPot;  // Analog potentiometer pin, reads actuator position as a value between 0 (fully retracted) and 1023 (fully extended)

    // Motion limits (raw ADC 0-1023), setting minStop higher than 0 will limit retraction, setting maxStop lower than 1023 will limit extension
    // Typically set to slightly less than max allowed by mechanical endstops to avoid motor damage
    int minStop = 0;
    int maxStop = 1023;

    // Control state
    int currentPwm = 0;              // Current PWM being applied to motor (-255 to 255)
    int currentTarget = 0;           // Target position (raw ADC)
    bool hasCommand = false;         // True after first external command applied
    unsigned long lastRampTime = 0;  // Last time PWM ramp was updated, in millis, used along with rampIntervalMs to control ramp timing

    LinearActuator(const char *n, int pR, int pL, int eR, int eL, int isPin, int pot)
        : name(n), pinPwmR(pR), pinPwmL(pL), pinEnR(eR), pinEnL(eL), pinIS(isPin), pinPot(pot) {}

    void setControlConfig(const ControlConfig &cfg)
    {
        controlConfig = cfg;
    }

    void init()
    {
        pinMode(pinPwmR, OUTPUT);
        pinMode(pinPwmL, OUTPUT);
        pinMode(pinEnR, OUTPUT);
        pinMode(pinEnL, OUTPUT);
        pinMode(pinIS, INPUT);
        pinMode(pinPot, INPUT);
        // Enable driver; PWM selection drives direction.
        digitalWrite(pinEnR, HIGH);
        digitalWrite(pinEnL, HIGH);
        analogWrite(pinPwmR, 0);
        analogWrite(pinPwmL, 0);
        currentTarget = analogRead(pinPot); // start from current position
    }

    // Sets target position to drive actuator to via update(), as normalized value [0.0,1.0], where 0.0 = minStop, 1.0 = maxStop
    void setTarget(float val)
    {
        if (val > 1.0)
            val = 1.0;
        if (val < 0.0)
            val = 0.0;
        currentTarget = minStop + (int)(val * (maxStop - minStop));
        hasCommand = true;
    }

    // Returns normalized position [0.0,1.0], where 0.0 = minStop, 1.0 = maxStop
    float getPos()
    {
        return (float)(getRawPos() - minStop) / (float)(maxStop - minStop);
    }

    // Returns raw potentiometer reading (0-1023)
    float getRawPos()
    {
        // Sample multiple times to reduce ADC noise; discard first conversion after mux switch.
        const int n = 6;
        analogRead(pinPot); // throw away first read to let mux settle
        int sum = 0;
        int minVal = 1023;
        int maxVal = 0;
        for (int i = 0; i < n; i++)
        {
            int v = analogRead(pinPot);
            sum += v;
            if (v < minVal)
                minVal = v;
            if (v > maxVal)
                maxVal = v;
        }
        // Drop one min and one max to blunt spikes, average the rest.
        return (sum - minVal - maxVal) / float(n - 2);
    }

    // Drives actuator to desired position using controlConfig; call frequently in main loop
    void update()
    {
        if (!hasCommand)
        {
            stop();
            return;
        }
        int error = currentTarget - getRawPos();
        if (abs(error) < controlConfig.pwmErrDeadband)
            error = 0;

        int desiredPwm = (int)(error * controlConfig.Kp);
        if (desiredPwm > 255)
            desiredPwm = 255;
        if (desiredPwm < -255)
            desiredPwm = -255;

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

    // Immediately stops actuator motion, does not reset targetPosition, so motor may be driven again on next update()
    void stop()
    {
        digitalWrite(pinEnR, LOW);
        digitalWrite(pinEnL, LOW);
        analogWrite(pinPwmR, 0);
        analogWrite(pinPwmL, 0);
    }

    // Returns live, non-normalized telemetry for this joint as read from sensors
    JointTelemetry getTelemetry(const char *code)
    {
        JointTelemetry jt;
        jt.name = code;
        jt.pos = getPos();
        jt.pot = analogRead(pinPot);
        jt.current = analogRead(pinIS);
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
            stop();
        }
        else if (pwm < 0)
        {
            retract(pwm);
        }
        else
        {
            extend(pwm);
        }
    }

    void retract(int pwm)
    {
        if (getRawPos() <= minStop)
        {
            stop();
            return;
        }
        digitalWrite(pinEnR, HIGH);
        digitalWrite(pinEnL, HIGH);
        analogWrite(pinPwmR, 0);
        analogWrite(pinPwmL, abs(pwm));
    }

    void extend(int pwm)
    {
        if (getRawPos() >= maxStop)
        {
            stop();
            return;
        }
        digitalWrite(pinEnR, HIGH);
        digitalWrite(pinEnL, HIGH);
        analogWrite(pinPwmR, pwm);
        analogWrite(pinPwmL, 0);
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

    void updateAll()
    {
        for (size_t i = 0; i < count; i++)
            actuators[i]->update();
    }

    size_t size() const
    {
        return count;
    }

    void applyCommands(const Command *cmds, size_t cmdCount)
    {
        // TODO: This is O(N^2), but N is small so probably ok for now. Would need to add a map for larger actuator sets.
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

    String serializeTelemetry() const
    {
        String out;
        out.reserve(3 + (count * JointTelemetry::serializedLengthEstimate()) + count); // 'JT ' + data + separators
        out += "JT ";
        for (size_t i = 0; i < count; i++)
        {
            actuators[i]->getTelemetry(actuators[i]->name).appendTo(out);
            if (i + 1 < count)
                out += ';';
        }
        return out;
    }

private:
    LinearActuator **actuators;
    size_t count;
};

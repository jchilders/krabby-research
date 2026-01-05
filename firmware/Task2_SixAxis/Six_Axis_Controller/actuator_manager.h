#pragma once
#include <Arduino.h>
#include <map>
#include <initializer_list>
#include "joint_telemetry.h"
#include "command.h"

// Linear actuator controller (potentiometer feedback)
class LinearActuator
{
public:
    const char *name;
    const int pinPwmR, pinPwmL, pinEnR, pinEnL, pinIS, pinPot;

    // Motion limits (raw ADC 0-1023)
    int minStop = 0;
    int maxStop = 1023;

    // Proportional gain (higher drives harder toward target)
    float Kp = 2.0;

    // Control state
    int currentPwm = 0;
    int currentTarget = 0;      // Target position (raw ADC)
    unsigned long lastRampTime = 0;

    LinearActuator(const char *n, int pR, int pL, int eR, int eL, int isPin, int pot)
        : name(n), pinPwmR(pR), pinPwmL(pL), pinEnR(eR), pinEnL(eL), pinIS(isPin), pinPot(pot) {}

    void init()
    {
        pinMode(pinPwmR, OUTPUT);
        pinMode(pinPwmL, OUTPUT);
        pinMode(pinEnR, OUTPUT);
        pinMode(pinEnL, OUTPUT);
        pinMode(pinIS, INPUT);
        pinMode(pinPot, INPUT);
        stop();
        currentTarget = analogRead(pinPot); // start from current position
    }

    void setTarget(float val)
    {
        if (val > 1.0)
            val = 1.0;
        if (val < 0.0)
            val = 0.0;
        currentTarget = minStop + (int)(val * (maxStop - minStop));
    }

    float getPos()
    {
        return (float)(getRawPos() - minStop) / (float)(maxStop - minStop);
    }

    float getRawPos()
    {
        return analogRead(pinPot);
    }

    void update(int pwmRampStep, int rampIntervalMs, int pwmDeadband, int pwmErrDeadband)
    {
        int error = currentTarget - getRawPos();
        if (abs(error) < pwmErrDeadband)
            error = 0;

        int desiredPwm = (int)(error * Kp);
        if (desiredPwm > 255)
            desiredPwm = 255;
        if (desiredPwm < -255)
            desiredPwm = -255;

        if (millis() - lastRampTime >= (unsigned long)rampIntervalMs)
        {
            lastRampTime = millis();
            if (currentPwm < desiredPwm)
            {
                currentPwm += pwmRampStep;
                if (currentPwm > desiredPwm)
                    currentPwm = desiredPwm;
            }
            else if (currentPwm > desiredPwm)
            {
                currentPwm -= pwmRampStep;
                if (currentPwm < desiredPwm)
                    currentPwm = desiredPwm;
            }
        }
        driveHardware(currentPwm, pwmDeadband);
    }

    void stop()
    {
        digitalWrite(pinEnR, LOW);
        digitalWrite(pinEnL, LOW);
        analogWrite(pinPwmR, 0);
        analogWrite(pinPwmL, 0);
    }

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
    void driveHardware(int pwm, int pwmDeadband)
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
        digitalWrite(pinEnR, LOW);
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
        digitalWrite(pinEnL, LOW);
        analogWrite(pinPwmR, pwm);
        analogWrite(pinPwmL, 0);
    }
};

class ActuatorManager
{
public:
    ActuatorManager(std::initializer_list<LinearActuator *> list)
    {
        for (auto *act : list)
            actuators[act->name] = act;
    }

    void initAll()
    {
        for (auto &entry : actuators)
            entry.second->init();
    }

    void updateAll(int pwmRampStep, int rampIntervalMs, int pwmDeadband, int pwmErrDeadband)
    {
        for (auto &entry : actuators)
            entry.second->update(pwmRampStep, rampIntervalMs, pwmDeadband, pwmErrDeadband);
    }

    size_t count() const
    {
        return actuators.size();
    }

    void applyCommands(const Command *cmds, size_t count)
    {
        for (size_t i = 0; i < count; i++)
        {
            const auto &cmd = cmds[i];
            auto it = actuators.find(cmd.name);
            if (it != actuators.end())
                it->second->setTarget(cmd.val);
        }
    }

    String serializeTelemetry() const
    {
        String out;
        const size_t count = actuators.size();
        out.reserve(3 + (count * JointTelemetry::serializedLengthEstimate()) + count);
        out += "JT ";
        size_t i = 0;
        for (auto &entry : actuators)
        {
            entry.second->getTelemetry(entry.first.c_str()).appendTo(out);
            if (i + 1 < count)
                out += ';';
            i++;
        }
        return out;
    }

private:
    std::map<String, LinearActuator *> actuators;
};

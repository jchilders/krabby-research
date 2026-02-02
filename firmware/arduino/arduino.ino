/*
 * Krabby-Uno Task 2: Six-Axis Leg Controller (all linear actuators)
 * Joint IDs: LHY, RHY, LHL, LKL, RHL, RKL
 */

#include <Arduino.h>
#include <EEPROM.h> // Added for calibration persistence
#include "joint_telemetry.h"
#include "command.h"
#include "actuator_manager.h"

// ==========================================
// INSTANTIATION (6 LINEAR ACTUATORS)
// ==========================================
// Format: Name, PWM_R, PWM_L, EN_R, EN_L, IS_PIN, POT_PIN

// --- LEFT LEG ---
LinearActuator lhy("LHY", 2, 3, 22, 23, A6, A0);
LinearActuator lhl("LHL", 4, 5, 24, 25, A7, A1);
LinearActuator lkl("LKL", 6, 7, 26, 27, A8, A2);

// --- RIGHT LEG ---
LinearActuator rhy("RHY", 8, 9, 28, 29, A9, A3);
LinearActuator rhl("RHL", 10, 11, 30, 31, A10, A4);
LinearActuator rkl("RKL", 12, 13, 32, 33, A11, A5);

const LinearActuator::ControlConfig ACTUATOR_CONFIG = {
    5,  // PWM_RAMP_STEP
    10, // RAMP_INTERVAL_MS
    20, // PWM_DEADBAND
    10, // PWM_ERROR_DEADBAND
    2.0 // Kp
};

LinearActuator *ACT_LIST[] = {&lhy, &lhl, &lkl, &rhy, &rhl, &rkl};
const size_t ACT_COUNT = sizeof(ACT_LIST) / sizeof(ACT_LIST[0]);
ActuatorManager actuatorManager(ACT_LIST, ACT_COUNT);

// Fixed size command buffer, max one command per joint
const size_t CMD_BUF_SIZE = ACT_COUNT;
Command cmdBuf[CMD_BUF_SIZE];

// ==========================================
// ARDUINO SETUP & LOOP
// ==========================================

// --- TELEMETRY CONFIGURATION ---
const int TELEMETRY_INTERVAL_MS = 50; // 20Hz update
unsigned long lastTelemetry = 0;

void setup()
{
    Serial.begin(115200);

    // Apply control config to each actuator
    for (size_t i = 0; i < ACT_COUNT; i++)
    {
        ACT_LIST[i]->setControlConfig(ACTUATOR_CONFIG);
    }

    actuatorManager.initAll();
    actuatorManager.loadCalibration();

    Serial.println("Krabby Ready. Send T/J/C/H commands to move joints.");
}

void loop()
{
    // 1. PARSE INPUT COMMANDS
    if (Serial.available())
    {
        char cmdType = Serial.peek(); // Peek first to decide handler

        if (cmdType == 'T') // Target Command
        {
            Serial.read(); // Consume 'T'
            String payload = Serial.readStringUntil('\n');
            size_t cmdCount = parseCommands(payload, cmdBuf, CMD_BUF_SIZE);
            if (cmdCount > 0)
            {
                actuatorManager.applyCommands(cmdBuf, cmdCount);
            }
        }
        else if (cmdType == 'J') // TODO 4: Manual Jog Command (J <name> <pwm>)
        {
            Serial.read(); // Consume 'J'
            String name = Serial.readStringUntil(' ');
            int pwm = Serial.readStringUntil('\n').toInt();
            actuatorManager.handleJog(name, pwm);
        }
        else if (cmdType == 'C') // TODO 3: Trigger Auto-Calibration
        {
            Serial.read();                // Consume 'C'
            Serial.readStringUntil('\n'); // Clear line
            actuatorManager.startAutoCalibration();
        }
        else if (cmdType == 'H') // Hold
        {
            Serial.read();
            actuatorManager.holdAll();
            Serial.readStringUntil('\n');
        }
        else
        {
            Serial.read(); // Flush junk
        }
    }

    // 2. UPDATE CONTROL LOOPS
    actuatorManager.updateAll();

    // 3. DUMP TELEMETRY TO SERIAL (one line for all joints)
    unsigned long curMillis = millis();
    if (curMillis - lastTelemetry > TELEMETRY_INTERVAL_MS)
    {
        lastTelemetry = curMillis;

        Serial.println(actuatorManager.serializeTelemetry());
    }
}

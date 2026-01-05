/*
 * Krabby-Uno Task 2: Six-Axis Leg Controller (all linear actuators)
 * Joint IDs: LHY, RHY, LHL, LKL, RHL, RKL
 */

#include <Arduino.h>
#include "joint_telemetry.h"
#include "command.h"
#include "actuator_manager.h"

// --- CONFIGURATION ---
const int TELEMETRY_INTERVAL_MS = 50; // 20Hz update

// Motion profile
const int PWM_RAMP_STEP = 5;
const int RAMP_INTERVAL_MS = 10;
const int PWM_DEADBAND = 20;
const int PWM_ERROR_DEADBAND = 10;

unsigned long lastTelemetry = 0;

// ==========================================
// INSTANTIATION (6 LINEAR ACTUATORS)
// ==========================================
LinearActuator lhy("LHY", 46, 45, 22, 23, A10, A0);
LinearActuator rhy("RHY", 2, 3, 24, 25, A11, A1);
LinearActuator lhl("LHL", 4, 5, 26, 27, A12, A2);
LinearActuator lkl("LKL", 6, 7, 28, 29, A13, A3);
LinearActuator rhl("RHL", 8, 9, 30, 31, A14, A4);
LinearActuator rkl("RKL", 10, 11, 32, 33, A15, A5);
ActuatorManager actuatorManager{&lhy, &rhy, &lhl, &lkl, &rhl, &rkl};
const size_t CMD_BUF_SIZE = actuatorManager.count(); // There's always max one command per joint
Command cmdBuf[CMD_BUF_SIZE];

void setup()
{
    Serial.begin(115200);

    actuatorManager.initAll();
}

void loop()
{
    // 1. PARSE INPUT COMMANDS AND APPLY COMMAND TARGETS: "T <name> <val> [<name> <val>...]"
    if (Serial.available())
    {
        if (Serial.read() == 'T')
        {
            String payload = Serial.readStringUntil('\n');
            size_t cmdCount = parseCommands(payload, cmdBuf, CMD_BUF_SIZE);
            if (cmdCount > 0)
            {
                actuatorManager.applyCommands(cmdBuf, cmdCount);
            }
        }
        // TODO: Should we care about other commands? Or just ignore them?
    }

    // 2. UPDATE CONTROL LOOPS
    actuatorManager.updateAll(PWM_RAMP_STEP, RAMP_INTERVAL_MS, PWM_DEADBAND, PWM_ERROR_DEADBAND);

    // 3. DUMP TELEMETRY TO SERIAL (one line for all joints)
    unsigned long curMillis = millis();
    if (curMillis - lastTelemetry > TELEMETRY_INTERVAL_MS)
    {
        lastTelemetry = curMillis;

        Serial.println(actuatorManager.serializeTelemetry());
    }
}

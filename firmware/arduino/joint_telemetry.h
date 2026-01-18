#pragma once
#include <Arduino.h>
// Shared telemetry format: one line per joint
// JT <name> <pos> <pot> <current> <enL> <enR> <pwmL> <pwmR> <saf>
// Names: LHY, RHY, LHL, LKL, RHL, RKL

struct JointTelemetry
{
    const char *name; // LHY/RHY/LHL/LKL/RHL/RKL
    float pos;        // Yaw: [-1.0,1.0] ; Linear: [0.0,1.0]
    int pot;          // Raw ADC, typically [0,1023] (0 for yaw joints)
    int current;      // Raw ADC current sense, [0,1023]
    bool enL;         // 0/1
    bool enR;         // 0/1
    int pwmL;         // 0-255
    int pwmR;         // 0-255
    bool saf;         // 0/1

    // Append serialized segment to an existing buffer.
    // Format: "<name> <pos> <pot> <current> <enL> <enR> <pwmL> <pwmR> <saf>"
    void appendTo(String &out) const
    {
        out += name;
        out += ' ';
        out += String(pos, 3);
        out += ' ';
        out += pot;
        out += ' ';
        out += current;
        out += ' ';
        out += int(enL);
        out += ' ';
        out += int(enR);
        out += ' ';
        out += pwmL;
        out += ' ';
        out += pwmR;
        out += ' ';
        out += int(saf);
    }

    static constexpr size_t serializedLengthEstimate()
    {
        // Estimate per segment:
        // name (3) + spaces (8) + pos (~6 incl. decimals) +
        // pot (4) + current (4) + enL (1) + enR (1) + pwmL (3) + pwmR (3) + saf (1)
        // Add a little headroom for longer numbers.
        return 40;
    }
};

#pragma once
#include <Arduino.h>

struct Command
{
    String name; // Joint ID: LHY, RHY, LHL, LKL, RHL, RKL
    float val;   // Linear: [0.0,1.0]
};

// Helper to clear command buffer
static inline void clearCommands(Command *cmds, size_t maxCmds)
{
    for (size_t k = 0; k < maxCmds; k++)
    {
        cmds[k].name = "";
        cmds[k].val = 0.0f;
    }
}

// Helper to get next token from line
static inline String nextTok(const String &line, int &idx, int len)
{
    while (idx < len && isspace(line[idx]))
        idx++;
    if (idx >= len)
        return String("");
    int start = idx;
    while (idx < len && !isspace(line[idx]))
        idx++;
    return line.substring(start, idx);
}

// Parse "T <name> <val> [<name> <val>...]" into a caller-provided buffer.
// Returns the number of commands parsed.
inline size_t parseCommands(const String &line, Command *cmds, size_t maxCmds)
{
    if (!cmds || maxCmds == 0)
        return 0;

    clearCommands(cmds, maxCmds);

    const int len = line.length();
    if (len == 0)
        return 0;

    int i = 0;
    // Expect and skip leading 'T'
    String tTok = nextTok(line, i, len);
    if (tTok != "T")
    {
        clearCommands(cmds, maxCmds);
        return 0;
    }
    size_t idx = 0;

    // Parse each <name> <val> pair until buffer full or tokens exhausted.
    while (idx < maxCmds)
    {
        String name = nextTok(line, i, len);
        String valStr = nextTok(line, i, len);
        if (name.length() == 0 || valStr.length() == 0)
        {
            // Problem parsing - clear and return
            clearCommands(cmds, maxCmds);
            return 0;
        }

        cmds[idx].name = name;
        cmds[idx].val = valStr.toFloat();
        idx++;
    }

    return idx;
}

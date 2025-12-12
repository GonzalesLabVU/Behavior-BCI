#ifndef LOGGER_H
#define LOGGER_H

#include <Arduino.h>

#define BAUDRATE 115200

class Logger {
    public:
        Logger();

        void write(const String& data);
        void write(uint16_t data);
        void write(float data);
        String read();
        void ack();
};

#endif

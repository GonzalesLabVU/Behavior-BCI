#ifndef LOGGER_H
#define LOGGER_H

#include <Arduino.h>

class Logger {
    public:
        Logger();

        void write(const String& data);
        void write(int data);
        void write(float data);
        void writeRaw(uint16_t data);
        String read();
        void ack();
};

#endif

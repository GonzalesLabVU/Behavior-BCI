#ifndef BRAKE_H
#define BRAKE_H

#include <Arduino.h>
#include <Servo.h>

// #define BRAKE_PIN 44

#if defined(ARDUINO_AVR_MEGA2560)
    static constexpr uint8_t BRAKE_PIN = 44;
#elif defined(ARDUINO_AVR_UNO)
    static constexpr uint8_t BRAKE_PIN = 9;
#else
    #error "Unsupported board: select Arduino Mega 2560 or Arduino Uno")
#endif

class Brake {
    public:
        Brake();

        void init(unsigned long engage_us, unsigned long release_us);
        void engage();
        void release();

    private:
        Servo servo_;
        unsigned long engage_us_;
        unsigned long release_us_;
        unsigned long hold_ms_;
        int engaged_;
};

#endif

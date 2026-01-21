#ifndef BRAKE_H
#define BRAKE_H

#include <Arduino.h>
#include <Servo.h>


#define BRAKE_PIN 44


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

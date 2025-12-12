#ifndef BRAKE_H
#define BRAKE_H

#include <Arduino.h>
#include <Servo.h>


#define BRAKE_PIN 44
#define RELEASE_US 1200
#define ENGAGE_US 900


class Brake {
    public:
        Brake();

        void engage();
        void release();

    private:
        Servo servo_;
        unsigned long hold_ms_;
        int engaged_;
};

#endif

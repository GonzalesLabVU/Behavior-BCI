#ifndef SPOUT_H
#define SPOUT_H

#include <Arduino.h>

class Spout {
    public:
        Spout();

        void init();
        void pulse();
        void flush();
    
    private:
        static constexpr uint8_t PULSE_PIN = 5;
        static constexpr uint8_t INIT_PIN = 4;
        static constexpr unsigned long INIT_DUR = 10000;
        static constexpr unsigned int PULSE_DUR_US = 5750;
};

#endif

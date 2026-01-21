#ifndef SPOUT_H
#define SPOUT_H

#include <Arduino.h>

class Spout {
    public:
        Spout();

        void init(unsigned long pulse_dur_us);
        void pulse();
        void flush();
    
    private:
        static constexpr uint8_t PULSE_PIN = 5;
        static constexpr uint8_t INIT_PIN = 4;
        static constexpr unsigned long INIT_DUR = 10000;

        unsigned long pulse_dur_us_;
};

#endif

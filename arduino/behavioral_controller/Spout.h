#ifndef SPOUT_H
#define SPOUT_H

#include <Arduino.h>

class Spout {
    public:
        Spout();

        void init(unsigned long pulse_dur_us);
        void pulse();
        void pulse(unsigned long us);
        void flush();
        void flush(unsigned long ms);
    
    private:
        static constexpr uint8_t PULSE_PIN = 5;

        unsigned long pulse_dur_us_ = 0;
};

#endif

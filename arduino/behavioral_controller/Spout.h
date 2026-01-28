#ifndef SPOUT_H
#define SPOUT_H

#include <Arduino.h>

class Spout {
    public:
        Spout();

        void init(unsigned long pulse_dur_us);
        void poll();
        void pulse();
        void pulse(unsigned long us);
        void flush();
        void flush(unsigned long ms);
    
    private:
        static constexpr uint8_t PULSE_PIN = 5;
        static constexpr uint8_t FORCE_PIN = 4;
        static constexpr unsigned long INIT_DUR = 5000;

        unsigned long pulse_dur_us_ = 0;
        unsigned long last_pulse_ms_ = 0;
        unsigned long debounce_ms_ = 250;

        bool forced_ = false;
        bool prev_forced_ = false;
};

#endif

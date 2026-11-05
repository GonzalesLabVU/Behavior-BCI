#ifndef TIMER_H
#define TIMER_H

#include <Arduino.h>

class Timer {
    public:
        Timer();

        void init(unsigned long duration_ms);
        void start();
        void reset();
        bool isRunning() const;
        bool started() const;
        unsigned long timeElapsed();
    
    private:
        unsigned long duration_ms_;
        unsigned long t_start_;
        bool started_;
};

#endif

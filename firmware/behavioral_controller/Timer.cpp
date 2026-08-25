#include "Timer.h"

Timer::Timer():
    duration_ms_(0),
    t_start_(0),
    started_(false)
{}

void Timer::init(unsigned long duration_ms) {
    duration_ms_ = duration_ms;
}

void Timer::start() {
    t_start_ = millis();
    started_ = true;
}

void Timer::reset() {
    started_ = false;
}

bool Timer::started() const {
    return started_;
}

bool Timer::isRunning() const {
    if (!started_) return false;

    return (millis() - t_start_) < duration_ms_;
}

unsigned long Timer::timeElapsed() {
    if (started_) {
        return millis() - t_start_;
    }
    return 0UL;
}

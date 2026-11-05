#include "Brake.h"

Brake::Brake():
    hold_ms_(500),
    engaged_(false),
    state_(IDLE),
    hold_start_ms_(0)
{}

void Brake::init(unsigned long engage_us, unsigned long release_us) {
    engage_us_ = engage_us;
    release_us_ = release_us;
}

void Brake::engage() {
    if (engaged_) return;

    servo_.attach(BRAKE_PIN);
    servo_.writeMicroseconds(engage_us_);

    engaged_ = true;
    state_ = HOLDING;
    hold_start_ms_ = millis();
}

void Brake::release() {
    if (!engaged_) return;

    servo_.attach(BRAKE_PIN);
    servo_.writeMicroseconds(release_us_);

    engaged_ = false;
    state_ = HOLDING;
    hold_start_ms_ = millis();
}

void Brake::update() {
    if (state_ != HOLDING) return;

    if ((unsigned long)(millis() - hold_start_ms_) >= hold_ms_) {
        servo_.detach();
        state_ = IDLE;
    }
}

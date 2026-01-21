#include "Brake.h"

Brake::Brake():
    hold_ms_(500),
    engaged_(false)
{}

void Brake::init(unsigned long engage_us, unsigned long release_us) {
    engage_us_ = engage_us;
    release_us_ = release_us;
}

void Brake::engage() {
    if (engaged_) return;

    servo_.attach(BRAKE_PIN);
    servo_.writeMicroseconds(engage_us_);
    delay(hold_ms_);
    servo_.detach();

    engaged_ = true;
}

void Brake::release() {
    if (!engaged_) return;

    servo_.attach(BRAKE_PIN);
    servo_.writeMicroseconds(release_us_);
    delay(hold_ms_);
    servo_.detach();

    engaged_ = false;
}

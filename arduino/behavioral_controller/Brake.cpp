#include "Brake.h"

Brake::Brake():
    hold_ms_(500),
    engaged_(false)
{}

void Brake::engage() {
    if (engaged_) return;

    servo_.attach(BRAKE_PIN);
    servo_.writeMicroseconds(ENGAGE_US);
    delay(hold_ms_);
    servo_.detach();

    engaged_ = true;
}

void Brake::release() {
    if (!engaged_) return;

    servo_.attach(BRAKE_PIN);
    servo_.writeMicroseconds(RELEASE_US);
    delay(hold_ms_);
    servo_.detach();

    engaged_ = false;
}

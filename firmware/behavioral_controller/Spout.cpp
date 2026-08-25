#include "Spout.h"

Spout::Spout()
{}

void Spout::init(unsigned long pulse_dur_us) {
    pinMode(PULSE_PIN, OUTPUT);
    digitalWrite(PULSE_PIN, LOW);

    pulse_dur_us_ = pulse_dur_us;
}

void Spout::pulse() {
    digitalWrite(PULSE_PIN, HIGH);
    delayMicroseconds(pulse_dur_us_);
    digitalWrite(PULSE_PIN, LOW);
}

void Spout::pulse(unsigned long us) {
    digitalWrite(PULSE_PIN, HIGH);
    delayMicroseconds(us);
    digitalWrite(PULSE_PIN, LOW);
}

void Spout::flush() {
    digitalWrite(PULSE_PIN, HIGH);
    delay(10000);
    digitalWrite(PULSE_PIN, LOW);
}

void Spout::flush(unsigned long ms) {
    digitalWrite(PULSE_PIN, HIGH);
    delay(ms);
    digitalWrite(PULSE_PIN, LOW);
}

#include "Spout.h"

Spout::Spout()
{}

void Spout::init(unsigned long pulse_dur_us) {
    pinMode(PULSE_PIN, OUTPUT);
    digitalWrite(PULSE_PIN, LOW);

    pulse_dur_us_ = pulse_dur_us;
}

void Spout::pulse() {
    pulse(pulse_dur_us_);
}

void Spout::pulse(unsigned long us) {
    static constexpr unsigned long SUBPULSE_US = 2500;
    static constexpr unsigned long GAP_MS = 20;

    unsigned long start_us = micros();

    while ((unsigned long)(micros() - start_us) < us) {
        digitalWrite(PULSE_PIN, HIGH);
        delayMicroseconds(SUBPULSE_US);
        digitalWrite(PULSE_PIN, LOW);
        delay(GAP_MS);
    }
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

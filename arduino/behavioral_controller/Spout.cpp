#include "Spout.h"

Spout::Spout()
{}

void Spout::init() {
    pinMode(PULSE_PIN, OUTPUT);
    digitalWrite(PULSE_PIN, LOW);

    pinMode(INIT_PIN, INPUT);
    
    unsigned long t_start = millis();
    while ((millis() - t_start) < INIT_DUR) {
        if (digitalRead(INIT_PIN) == HIGH) {
            pulse();
            delay(200);
            t_start = millis();
        }
    }
}

void Spout::pulse() {
    digitalWrite(PULSE_PIN, HIGH);
    delayMicroseconds(PULSE_DUR_US);
    digitalWrite(PULSE_PIN, LOW);
}

void Spout::flush() {
    digitalWrite(PULSE_PIN, HIGH);
    delay(30000);
    digitalWrite(PULSE_PIN, LOW);
}

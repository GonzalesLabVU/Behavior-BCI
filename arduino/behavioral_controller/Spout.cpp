#include "Spout.h"

Spout::Spout()
{}

void Spout::init(unsigned long pulse_dur_us) {
    pinMode(PULSE_PIN, OUTPUT);
    digitalWrite(PULSE_PIN, LOW);

    pulse_dur_us_ = pulse_dur_us;

    pinMode(FORCE_PIN, INPUT);

    last_pulse_ms_ = 0;
    forced_ = false;
    prev_forced_ = (digitalRead(FORCE_PIN) == HIGH);
    
    unsigned long t_start = millis();
    while ((millis() - t_start) < INIT_DUR) {
        if (digitalRead(FORCE_PIN) == HIGH) {
            pulse();
            delay(200);

            t_start = millis();
            last_pulse_ms_ = t_start;
        }
    }
}

void Spout::poll() {
    const bool curr_forced = (digitalRead(FORCE_PIN) == HIGH);
    const unsigned long now_ms = millis();

    if (curr_forced && !prev_forced_) {
        if ((now_ms - last_pulse_ms_) >= debounce_ms_) {
            forced_ = true;
        }
    }

    if (forced_) {
        pulse();
        last_pulse_ms_ = now_ms;
        forced_ = false;
    }

    prev_forced_ = curr_forced;
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

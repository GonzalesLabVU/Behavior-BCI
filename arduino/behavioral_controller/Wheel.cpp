#include "Wheel.h"
#include <math.h>

volatile long Wheel::current_pos_ = 0;
volatile uint8_t* Wheel::s_b_in_reg_ = nullptr;
uint8_t Wheel::s_b_mask_ = 0;

Wheel::Wheel():
    displacement(0.0f),
    easy_threshold_counts_(0),
    normal_threshold_counts_(0),
    active_threshold_counts_(0),
    init_pos_(0),
    threshold_reached_(false),
    threshold_missed_(false),
    bidirectional_(false),
    positive_threshold_(true)
{}

void Wheel::init(float easy_threshold, float normal_threshold, bool bidirectional) {
    pinMode(A_PIN, INPUT_PULLUP);
    pinMode(B_PIN, INPUT_PULLUP);

    s_b_in_reg_ = portInputRegister(digitalPinToPort(B_PIN));
    s_b_mask_ = digitalPinToBitMask(B_PIN);

    attachInterrupt(digitalPinToInterrupt(A_PIN), Wheel::isr_, RISING);

    easy_threshold_counts_ = degToCounts_(easy_threshold);
    normal_threshold_counts_ = degToCounts_(normal_threshold);

    bidirectional_ = bidirectional;
    positive_threshold_ = (normal_threshold >= 0.0f);

    reset(true);
}

void Wheel::update() {
    long curr, init;

    noInterrupts();
    curr = current_pos_;
    init = init_pos_;
    interrupts();

    long counts = curr - init;
    long centi = (counts >= 0)
                 ? ((counts * 4500L + 64L) >> 7)
                 : -(((-counts) * 4500L + 64L) >> 7);
    displacement = centi * 0.01f;

    const long th_c = active_threshold_counts_;

    if (bidirectional_) {
        if (counts >= th_c || counts <= -th_c) {
            threshold_reached_ = true;
        }
    } else {
        if (positive_threshold_) {
            if (counts >= th_c) threshold_reached_ = true;
            if (counts <= -th_c) threshold_missed_ = true;
        } else {
            if (counts >= th_c) threshold_missed_ = true;
            if (counts <= -th_c) threshold_reached_ = true;
        }
    }
}

float Wheel::getDisplacement() {
    return displacement;
}

bool Wheel::thresholdReached() {
    if (threshold_reached_) {
        threshold_reached_ = false;
        return true;
    }
    return false;
}

bool Wheel::thresholdMissed() {
    if (bidirectional_) {
        return false;
    }

    if (threshold_missed_) {
        threshold_missed_ = false;
        return true;
    }
    return false;
}

void Wheel::reset(bool easy_trial) {
    active_threshold_counts_ = easy_trial ? easy_threshold_counts_ : normal_threshold_counts_;

    noInterrupts();
    init_pos_ = current_pos_;
    interrupts();

    displacement = 0.0f;
    threshold_reached_ = false;
    threshold_missed_ = false;
}

static long Wheel::degToCounts_(float deg) {
    long c = lroundf(deg * (1024.0f / 360.0f));
    return (c >= 0) ? c : -c;
}

void Wheel::isr_() {
    // bool b_state = digitalRead(B_PIN);
    // int8_t dir = b_state ? -1 : 1;
    // current_pos_ += dir;
    current_pos_ += (*s_b_in_reg_ & s_b_mask_) ? -1 : +1;
}

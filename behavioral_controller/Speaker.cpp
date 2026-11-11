#include "Speaker.h"

Speaker* Speaker::instance_ = nullptr;

void TIMER2_COMPA_vect_func();

Speaker::Speaker():
    mode_(IDLE),
    side_("L"),
    pin_state_(false),
    ocr_val_(124),
    half_us_(500),
    elapsed_us_(0),
    target_us_(0),
    step_us_(0),
    since_step_us_(0),
    rng_(1)
{}

void Speaker::init(const String& side) {
    pinMode(SPEAKER_PIN, OUTPUT);

    drivePin_(false);
    stopTimer2_();

    if (side == "R") {
        side_ = "R";
    } else {
        side_ = "L";
    }

    mode_ = IDLE;
    instance_ = this;

    rng_ = (uint16_t)micros() | 1;

    buildMissTable_();
}

void Speaker::cue() {
    noInterrupts();

    bool can_start = (mode_ == IDLE);

    interrupts();

    if (!can_start) return;

    unsigned int freq_hz =
        (side_ == "R")
        ? RIGHT_CUE_HZ
        : LEFT_CUE_HZ;

    startTone_(CUE, freq_hz, TONE_MS);
}

void Speaker::hit() {
    noInterrupts();

    Mode m = mode_;

    if (m == HIT || m == MISS) {
        interrupts();
        return;
    }

    if (m == CUE) {
        stopTimer2_();
        mode_ = IDLE;
        drivePin_(false);
    }

    interrupts();

    startTone_(HIT, HIT_HZ, TONE_MS);
}

void Speaker::miss() {
    noInterrupts();
    Mode m = mode_;

    if (m == HIT || m == MISS) {
        interrupts();
        return;
    }

    if (m == CUE) {
        stopTimer2_();
        mode_ = IDLE;
        drivePin_(false);
    }

    interrupts();

    startMiss_(TONE_MS);
}

void Speaker::stop() {
    noInterrupts();

    stopTimer2_();
    mode_ = IDLE;

    interrupts();

    drivePin_(false);
}

void Speaker::startTone_(Mode m, unsigned freq_hz, unsigned long duration_ms) {
    noInterrupts();

    mode_ = m;

    updateForFreq_(freq_hz);

    elapsed_us_ = 0;
    target_us_ = duration_ms * 1000UL;

    since_step_us_ = 0;
    step_us_ = 0;

    startTimer2_();

    interrupts();
}

void Speaker::startMiss_(unsigned long duration_ms) {
    nextMissParams_();

    unsigned long target_us = duration_ms * 1000UL;

    noInterrupts();

    mode_ = MISS;

    elapsed_us_ = 0;
    target_us_ = target_us;

    startTimer2_();

    interrupts();
}

void Speaker::buildMissTable_() {
    const unsigned long span_hz = MISS_HZ_MAX - MISS_HZ_MIN;

    for (uint8_t i = 0; i < MISS_TABLE_SIZE; i++) {
        unsigned long freq_hz = MISS_HZ_MIN + (span_hz * i) / (MISS_TABLE_SIZE - 1);
        unsigned long ticks = (500000UL / freq_hz) / 4UL;

        if (ticks < 1UL) ticks = 1UL;
        if (ticks > 255UL) ticks = 255UL;

        miss_ticks_[i] = (uint8_t)ticks;
    }
}

void Speaker::nextMissParams_() {
    uint16_t r = xorshift16_(rng_);

    uint8_t ticks = miss_ticks_[r & (MISS_TABLE_SIZE - 1)];

    ocr_val_ = ticks - 1;
    half_us_ = (uint16_t)ticks << 2;

    step_us_ = MISS_STEP_US_MIN + (uint16_t)(((r >> 8) * MISS_STEP_SPAN) >> 8);
    since_step_us_ = 0;
}

void Speaker::updateForFreq_(unsigned freq_hz) {
    unsigned long half_us = 500000UL / (unsigned long)freq_hz;

    if (half_us < 8UL) half_us = 8UL;

    unsigned long ticks = half_us / 4UL;

    if (ticks < 1UL) ticks = 1UL;
    if (ticks > 255UL) ticks = 255UL;

    half_us_ = ticks * 4UL;
    ocr_val_ = (uint8_t)(ticks - 1UL);
}

inline void Speaker::drivePin_(bool high) {
    digitalWrite(SPEAKER_PIN, high ? HIGH : LOW);
    pin_state_ = high;
}

void Speaker::startTimer2_() {
    TCCR2A = 0;
    TCCR2B = 0;
    TCNT2 = 0;

    OCR2A = ocr_val_;

    TCCR2A |= (1 << WGM21);
    TCCR2B |= (1 << CS22);
    TIMSK2 |= (1 << OCIE2A);
}

void Speaker::stopTimer2_() {
    TIMSK2 &= ~(1 << OCIE2A);
    TCCR2B &= ~((1 << CS22) | (1 << CS21) | (1 << CS20));
    TCNT2 = 0;
}

void Speaker::onTick_() {
    #if defined(ARDUINO_AVR_MEGA2560)
        PINC = _BV(PC4);
    #elif defined(ARDUINO_AVR_UNO)
        PINB = _BV(PB2);
    #else
        digitalWrite(SPEAKER_PIN, !pin_state_);
    #endif

    pin_state_ = !pin_state_;

    elapsed_us_ += half_us_;
    since_step_us_ += half_us_;

    if (mode_ == MISS && since_step_us_ >= step_us_) {
        nextMissParams_();
        OCR2A = ocr_val_;
    }

    if (elapsed_us_ >= target_us_) {
        stopTimer2_();
        mode_ = IDLE;
        drivePin_(false);
    }
}

ISR(TIMER2_COMPA_vect) { TIMER2_COMPA_vect_func(); }

void TIMER2_COMPA_vect_func() {
    if (Speaker::instance_) {
        Speaker::instance_ -> onTick_();
    }
}

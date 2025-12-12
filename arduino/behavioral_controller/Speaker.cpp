#include "Speaker.h"

Speaker* Speaker::instance_ = nullptr;

void TIMER2_COMPA_vect_func();

Speaker::Speaker():
    mode_(IDLE),
    pin_state_(false),
    ocr_val_(124),
    half_us_(500),
    elapsed_us_(0),
    target_us_(0),
    step_us_(0),
    since_step_us_(0)
{}

void Speaker::init() {
    pinMode(SPEAKER_PIN, OUTPUT);

    drivePin_(false);
    stopTimer2_();

    mode_ = IDLE;
    instance_ = this;
    rng_ = (uint32_t)micros() | 1;
}

void Speaker::cue() {
    noInterrupts();
    bool can_start = (mode_ == IDLE);
    interrupts();

    if (!can_start) return;

    startTone_(CUE, CUE_FREQ_HZ, TONE_MS);
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

    startTone_(HIT, HIT_FREQ_HZ, TONE_MS);
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
    noInterrupts();

    mode_ = MISS;

    uint32_t r = xorshift32_(rng_);
    unsigned f0 = MISS_FREQ_HZ_MIN + (r % (MISS_FREQ_HZ_MAX - MISS_FREQ_HZ_MIN + 1));

    updateForFreq_(f0);

    step_us_ = MISS_STEP_US_MIN + (xorshift32_(rng_) % (MISS_STEP_US_MAX - MISS_STEP_US_MIN + 1));
    since_step_us_ = 0;
    elapsed_us_ = 0;
    target_us_ = duration_ms * 1000UL;

    startTimer2_();

    interrupts();
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
    if (high) {
        PORTC |= _BV(4);
    } else {
        PORTC &= ~_BV(4);
    }
    
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
    PINC = _BV(4);
    pin_state_ = !pin_state_;

    elapsed_us_ += half_us_;
    since_step_us_ += half_us_;

    if (mode_ == MISS && since_step_us_ >= step_us_) {
        unsigned f = MISS_FREQ_HZ_MIN + (xorshift32_(rng_) % (MISS_FREQ_HZ_MAX - MISS_FREQ_HZ_MIN + 1));
        updateForFreq_(f);

        OCR2A = ocr_val_;
        step_us_ = MISS_STEP_US_MIN + (xorshift32_(rng_) % (MISS_STEP_US_MAX - MISS_STEP_US_MIN + 1));
        since_step_us_ = 0;
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

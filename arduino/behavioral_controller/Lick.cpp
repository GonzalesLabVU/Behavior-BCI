#include "Lick.h"
#include <math.h>

Lick::Lick():
    just_touched_(false),
    prev_touched_(false),
    read_raw_enabled_(false),
    last_raw_sample_ms_(0)
{}

void Lick::writeReg_(uint8_t reg, uint8_t val) {
    Wire.beginTransmission((uint8_t)ADDR);
    Wire.write((uint8_t)reg);
    Wire.write((uint8_t)val);
    Wire.endTransmission();
}

bool Lick::read2_(uint8_t reg, uint8_t& b0, uint8_t& b1) {
    Wire.beginTransmission(ADDR);
    Wire.write(reg);

    if (Wire.endTransmission(false) != 0) {
        return false;
    }

    if (Wire.requestFrom((int)ADDR, 2) != 2) {
        return false;
    }

    b0 = Wire.read();
    b1 = Wire.read();

    return true;
}

uint16_t Lick::readFiltered_(uint8_t electrode) {
    uint8_t lsb = 0, msb = 0;
    uint8_t reg = (uint8_t)(FILT_DATA_L + 2*electrode);

    if (!read2_(reg, lsb, msb)) {
        return 0;
    }
    return (uint16_t)(((uint16_t)msb << 8) | lsb);
}

void Lick::init(bool read_raw) {
    // begin I2C protocol
    Wire.begin();

    read_raw_enabled_ = read_raw;

    // configure touch sensor
    config_(read_raw_enabled_);

    // initialize touch state
    just_touched_ = false;
    prev_touched_ = false;

    last_raw_sample_ms_ = 0;
}

void Lick::config_(bool read_raw) {
    // reset and stop scanning
    writeReg_(SOFTRESET, 0x63);
    delay(1);
    writeReg_(ELECTRODE_CONFIG, 0x00);
    
    // basic default filter/baseline config
    writeReg_(0x2B, 0x01);
    writeReg_(0x2C, 0x01);
    writeReg_(0x2D, 0x00);
    writeReg_(0x2E, 0x00);
    writeReg_(0x2F, 0x01);
    writeReg_(0x30, 0x01);
    writeReg_(0x31, 0xFF);
    writeReg_(0x32, 0x02);

    // disable electrodes 0-7 (set impossible thresholds)
    uint8_t start_electrode = read_raw ? 1 : 0;

    for (uint8_t i = start_electrode; i < 8; i++) {
        writeReg_((uint8_t)(TOUCH_TH + i*2), 0xFF);
        writeReg_((uint8_t)(RELEASE_TH + i*2), 0x00);
    }

    // set very conservative thresholds for SPOUT_ELECTRODE during warm-up
    writeReg_((uint8_t)(TOUCH_TH + SPOUT_ELECTRODE*2), 30);
    writeReg_((uint8_t)(RELEASE_TH + SPOUT_ELECTRODE*2), 20);

    // disable electrodes 9-11 (set impossible thresholds)
    for (uint8_t i = 9; i < 12; i++) {
        writeReg_((uint8_t)(TOUCH_TH + i*2), 0xFF);
        writeReg_((uint8_t)(RELEASE_TH + i*2), 0x00);
    }

    // disable hardware debounce
    writeReg_(0x5B, 0x00);

    // only enable SPOUT_ELECTRODE
    writeReg_(ELECTRODE_CONFIG, 0x09);

    // warm-up period to let baseline settle
    delay(2500);
}

static void isort16(uint16_t* a, uint16_t len) {
    for (uint16_t i = 1; i < len; ++i) {
        uint16_t key = a[i];
        int j = (int)i - 1;

        while (j >= 0 && a[j] > key) {
            a[j+1] = a[j];
            --j;
        }
        
        a[j+1] = key;
    }
}

void Lick::calibrate() {
    const uint16_t SAMPLE_COUNT = 900;
    const uint16_t SAMPLE_DELAY_MS = 3;
    const float K_SIGMA_TOUCH = 6.0f;
    const uint8_t MIN_TOUCH_TH = 8;
    const uint8_t MAX_TOUCH_TH = 63;
    const uint8_t MIN_RELEASE_TH = 4;
    const float HYST_FRAC = 0.35f;
    const uint8_t PERC_MARGIN = 2;

    static uint16_t buf[900];
    uint16_t n = 0;
    const uint32_t t_end = millis() + (SAMPLE_COUNT * SAMPLE_DELAY_MS) + 10;

    while ((int32_t)(millis() - t_end) < 0 && n < SAMPLE_COUNT) {
        uint16_t v = readFiltered_(SPOUT_ELECTRODE);
        if (v != 0) {
            buf[n++] = v;
        }
        delay(SAMPLE_DELAY_MS);
    }

    if (n < 50) {
        writeReg_((uint8_t)(TOUCH_TH + SPOUT_ELECTRODE*2), DEFAULT_TOUCH_TH);
        writeReg_((uint8_t)(RELEASE_TH + SPOUT_ELECTRODE*2), DEFAULT_RELEASE_TH);
        return;
    }

    isort16(buf, n);
    const uint16_t median = buf[n/2];

    for (uint16_t i = 0; i < n; ++i) {
        uint16_t x = buf[i];
        buf[i] = (x > median) ? (x - median) : (median - x);
    }

    isort16(buf, n);
    const uint16_t mad = buf[n/2];
    const float sigma = 1.4826f * (float)mad;

    const uint16_t idx_p99 = (uint16_t)((99UL * n) / 100UL);
    const uint16_t p99 = buf[min((uint16_t)(idx_p99), (uint16_t)(n-1))];

    uint8_t touch_th = DEFAULT_TOUCH_TH;
    uint8_t thr_sigma = (uint8_t)ceilf(K_SIGMA_TOUCH * sigma);
    uint8_t thr_p99 = (uint8_t)min((uint16_t)(p99 + PERC_MARGIN), (uint16_t)(255));

    if (thr_sigma > touch_th) {
        touch_th = thr_sigma;
    }
    if (thr_p99 > touch_th) {
        touch_th = thr_p99;
    }

    if (touch_th < MIN_TOUCH_TH) {
        touch_th = MIN_TOUCH_TH;
    }
    if (touch_th > MAX_TOUCH_TH) {
        touch_th = MAX_TOUCH_TH;
    }

    uint8_t hysteresis = (uint8_t)ceilf(HYST_FRAC * touch_th);
    if (hysteresis < MIN_RELEASE_TH) {
        hysteresis = MIN_RELEASE_TH;
    }

    uint8_t release_th = (touch_th > hysteresis) ? (uint8_t)(touch_th - hysteresis) : MIN_RELEASE_TH;
    if (release_th < MIN_RELEASE_TH) {
        release_th = MIN_RELEASE_TH;
    }
    if (release_th >= touch_th) {
        release_th = (touch_th > MIN_RELEASE_TH) ? (uint8_t)(touch_th - MIN_RELEASE_TH) : (uint8_t)1;
    }
    if (release_th < DEFAULT_RELEASE_TH) {
        release_th = DEFAULT_RELEASE_TH;
    }

    writeReg_((uint8_t)(TOUCH_TH + SPOUT_ELECTRODE*2), touch_th);
    writeReg_((uint8_t)(RELEASE_TH + SPOUT_ELECTRODE*2), release_th);

    writeReg_(0x5B, 0x00);
}

void Lick::poll() {
    Wire.beginTransmission(ADDR);
    Wire.write(TOUCHSTATUS);
    Wire.endTransmission(false);

    if (Wire.requestFrom((int)ADDR, 2) == 2) {
        uint8_t lsb = Wire.read();
        uint8_t msb = Wire.read();
        uint16_t touched = ((uint16_t)msb << 8) | lsb;

        bool now_touched = ((touched >> SPOUT_ELECTRODE) & 0x01) != 0;

        if (now_touched && !prev_touched_) {
            just_touched_ = true;
        } else {
            just_touched_ = false;
        }

        prev_touched_ = now_touched;
    }
}

bool Lick::justTouched() {
    return just_touched_;
}

uint16_t Lick::getRaw() {
    if (!read_raw_enabled_) {
        return 0;
    }

    uint32_t now = millis();
    bool valid_sample = (uint32_t)(now - last_raw_sample_ms_) >= RAW_SAMPLE_MS;

    if (last_raw_sample_ms_ != 0 && !valid_sample) {
        return 0;
    }

    uint16_t val = readFiltered_(RAW_ELECTRODE);

    if (val != 0) {
        last_raw_sample_ms_ = now;
        return val;
    }

    return 0;
}

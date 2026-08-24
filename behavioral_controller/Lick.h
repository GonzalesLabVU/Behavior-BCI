#ifndef LICK_H
#define LICK_H

#include <Arduino.h>
#include <Wire.h>

class Lick {
    public:
        Lick();

        void init(bool read_raw = false);
        void calibrate();
        void sampleFiltered();
        bool justTouched();
        uint16_t sampleRaw();
    
    private:
        static constexpr uint8_t ADDR = 0x5A;
        static constexpr uint8_t  SOFTRESET = 0x80;
        static constexpr uint8_t ELECTRODE_CONFIG = 0x5E;
        static constexpr uint8_t  TOUCH_TH = 0x41;
        static constexpr uint8_t RELEASE_TH = 0x42;
        static constexpr uint8_t TOUCHSTATUS = 0x00;
        static constexpr uint8_t FILT_DATA_L = 0x04;
        static constexpr uint8_t SPOUT_ELECTRODE = 8;
        static constexpr uint8_t RAW_ELECTRODE = 0;

        static constexpr uint8_t DEFAULT_TOUCH_TH = 12;
        static constexpr uint8_t DEFAULT_RELEASE_TH = 7;

        static constexpr uint32_t RAW_SAMPLE_MS = 50;

        bool prev_touched_;
        bool just_touched_;
        bool read_raw_enabled_;
        uint32_t last_raw_sample_ms_;

        void writeReg_(uint8_t reg, uint8_t val);
        bool read2_(uint8_t reg, uint8_t& b0, uint8_t& b1);
        uint16_t readFiltered_(uint8_t electrode);
        void config_(bool read_raw);
};

#endif


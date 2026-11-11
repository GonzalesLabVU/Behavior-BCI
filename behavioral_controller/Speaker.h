#ifndef SPEAKER_H
#define SPEAKER_H

#include <Arduino.h>

#if defined(ARDUINO_AVR_UNO)
    static constexpr uint8_t SPEAKER_PIN = 10;
#elif defined(ARDUINO_AVR_MEGA2560)
    static constexpr uint8_t SPEAKER_PIN = 33;
#else
    #error "Invalid board: select either Uno or Mega 2560"
#endif

class Speaker {
    public:
        Speaker();

        void init(const String& side);
        void cue();
        void hit();
        void miss();
        void stop();
    
    private:
        static constexpr unsigned int LEFT_CUE_HZ = 2500;
        static constexpr unsigned int RIGHT_CUE_HZ = 2500;
        static constexpr unsigned int HIT_HZ = 4000;

        static constexpr unsigned int MISS_HZ_MIN = 1000;
        static constexpr unsigned int MISS_HZ_MAX = 4000;

        static constexpr unsigned long MISS_STEP_US_MIN = 100;
        static constexpr unsigned long MISS_STEP_US_MAX = 300;
        static constexpr uint16_t MISS_STEP_SPAN =
            (uint16_t)(MISS_STEP_US_MAX - MISS_STEP_US_MIN + 1);
        static constexpr uint8_t MISS_TABLE_SIZE = 64;

        static constexpr unsigned long TONE_MS = 1000;

        enum Mode : uint8_t {
            IDLE,
            CUE,
            HIT,
            MISS
        };

        String side_;

        static Speaker* instance_;

        volatile Mode mode_;
        volatile bool pin_state_;

        volatile uint8_t ocr_val_;

        volatile unsigned long half_us_;
        volatile unsigned long elapsed_us_;
        volatile unsigned long target_us_;

        volatile unsigned long step_us_;
        volatile unsigned long since_step_us_;

        volatile uint16_t rng_;

        uint8_t miss_ticks_[MISS_TABLE_SIZE];

        void startTone_(Mode m,
                        unsigned freq_hz,
                        unsigned long duration_ms);

        void startMiss_(unsigned long duration_ms);

        void buildMissTable_();
        void nextMissParams_();

        void updateForFreq_(unsigned freq_hz);

        void startTimer2_();
        void stopTimer2_();

        void onTick_();

        inline void drivePin_(bool high);

        static inline uint16_t xorshift16_(volatile uint16_t &s) {
            uint16_t x = s;

            x ^= x << 7;
            x ^= x >> 9;
            x ^= x << 8;

            s = x;

            return x;
        }

        friend void TIMER2_COMPA_vect_func();
};

#endif

#ifndef SPEAKER_H
#define SPEAKER_H

#include <Arduino.h>

class Speaker {
    public:
        Speaker();

        void init();
        void cue();
        void hit();
        void miss();
        void stop();
    
    private:
        static constexpr uint8_t SPEAKER_PIN = 33;
        static constexpr unsigned int CUE_FREQ_HZ = 1000;
        static constexpr unsigned int HIT_FREQ_HZ = 4000;
        static constexpr unsigned int MISS_FREQ_HZ_MIN = 1000;
        static constexpr unsigned int MISS_FREQ_HZ_MAX = 4000;
        static constexpr unsigned long MISS_STEP_US_MIN = 100;
        static constexpr unsigned long MISS_STEP_US_MAX = 200;
        static constexpr unsigned long TONE_MS = 1000;

        enum Mode : uint8_t { IDLE, CUE, HIT, MISS };

        static Speaker* instance_;

        volatile Mode mode_;
        volatile bool pin_state_;
        volatile uint8_t ocr_val_;
        volatile unsigned long half_us_;
        volatile unsigned long elapsed_us_;
        volatile unsigned long target_us_;
        volatile unsigned long step_us_;
        volatile unsigned long since_step_us_;
        volatile uint32_t rng_;

        void startTone_(Mode m, unsigned freq_hz, unsigned long duration_ms);
        void startMiss_(unsigned long duration_ms);
        void updateForFreq_(unsigned freq_hz);
        void startTimer2_();
        void stopTimer2_();
        void onTick_();
        inline void drivePin_(bool high);
        static inline uint32_t xorshift32_(volatile uint32_t &s) {
            uint32_t x = s;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            s = x;
            return x;
        }

        friend void TIMER2_COMPA_vect_func();
};

#endif

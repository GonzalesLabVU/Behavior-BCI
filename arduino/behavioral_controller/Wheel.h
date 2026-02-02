#ifndef WHEEL_H
#define WHEEL_H

#include <Arduino.h>

class Wheel {
    public:
        float displacement;

        Wheel();

        void init(float easy_threshold, float normal_threshold, char side, bool reverse);
        void update();
        bool thresholdReached();
        bool thresholdMissed();
        void reset(bool easy, char side);
        inline void reset() { reset(false, 'B'); }
            
    private:
        static constexpr uint8_t A_PIN = 3;
        static constexpr uint8_t B_PIN = 2;

        static volatile uint8_t* s_b_in_reg_;
        static uint8_t s_b_mask_;

        long easy_counts_;
        long normal_counts_;
        long active_counts_;

        static volatile long current_pos_;
        long init_pos_;

        bool threshold_reached_;
        bool threshold_missed_;

        int8_t dir_;
        bool reverse_;

        static int8_t sideToDir_(char side);
        static long degToCounts_(float deg);
        static void isr_();
};

#endif

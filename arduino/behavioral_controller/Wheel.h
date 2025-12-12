#ifndef WHEEL_H
#define WHEEL_H

#include <Arduino.h>

class Wheel {
    public:
        float displacement;

        Wheel();

        void init(float easy_threshold, float normal_threshold, bool bidirectional);
        void update();
        float getDisplacement();
        bool thresholdReached();
        bool thresholdMissed();
        void reset(bool easy_trial);

        inline void reset() { reset(false); }
            
    private:
        static constexpr uint8_t A_PIN = 3;
        static constexpr uint8_t B_PIN = 2;

        static volatile uint8_t* s_b_in_reg_;
        static uint8_t s_b_mask_;

        long easy_threshold_counts_;
        long normal_threshold_counts_;
        long active_threshold_counts_;
        static volatile long current_pos_;
        long init_pos_;
        bool threshold_reached_;
        bool threshold_missed_;
        bool bidirectional_;
        bool positive_threshold_;

        static long degToCounts_(float deg);
        static void isr_();
};

#endif

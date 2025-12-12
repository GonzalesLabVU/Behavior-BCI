// include libraries

#include "Wheel.h"
#include "Brake.h"
#include "Lick.h"
#include "Speaker.h"
#include "Spout.h"
#include "Logger.h"
#include "Timer.h"

#include <math.h>
#include <limits.h>

// macros

#define BAUDRATE 1000000
#define SEED_PIN A0
#define POWER_EN 7

// unit conversion handles

constexpr unsigned long SECONDS(float s) {
    return (unsigned long)(s * 1000.0f);
}
constexpr unsigned long MINUTES(float m) {
    return (unsigned long)(m * 60.0f * 1000.0f);
}

template <typename T>
constexpr float DEGREES(T d) { return static_cast<float>(d); }

// phase_state machine

enum class SessionState {
    MAIN,
    CLEANUP
};
SessionState session_state;

enum class PhaseState {
    IDLE,
    CUE,
    TRIAL,
    HIT,
    MISS,
    DELAY
};
PhaseState phase_state;

// component objects

Brake brake;
Lick lick;
Wheel wheel;
Spout spout;
Speaker speaker;

Timer session_timer;
Timer phase_timer;

Logger logger;

// global parameters

int phase_id;

unsigned long session_T;
unsigned long trial_T;
unsigned long delay_T;
unsigned long tone_T = SECONDS(1);

float easy_threshold = DEGREES(15);
float threshold;
bool bidirectional;

bool session_initialized = false;
bool trial_hit;
bool reward_given = false;

int K;
int trial_num = 0;

long last_disp_mark = LONG_MIN;

// phase logic forward declarations

void run_phase_1();
void run_phase_2();
void run_phase_3_4_5();

// helper functions

inline bool nearMultiple(float x, float step, float tol, float* nearestOut) {
    float q = roundf(x / step);
    float m = q * step;

    if (nearestOut) *nearestOut = m;

    return fabsf(x - m) <= tol;
}

void checkInactivity(float current_disp) {
    static Timer inactivity_timer;
    static bool cue_active = false;
    static unsigned long timeout_T = SECONDS(5);
    static float init_disp = 0.0f;

    const unsigned long inactivity_start_T = SECONDS(5);

    if (phase_state != PhaseState::TRIAL) {
        cue_active = false;
        inactivity_timer.reset();
        init_disp = current_disp;
        return;
    }

    unsigned long elapsed_ms = phase_timer.timeElapsed();

    if (elapsed_ms < inactivity_start_T) {
        cue_active = false;
        inactivity_timer.reset();
        init_disp = current_disp;
        return;
    }

    unsigned long remaining_ms = (elapsed_ms < trial_T) ? (trial_T - elapsed_ms) : 0;

    if (!cue_active) {
        if (!inactivity_timer.started()) {
            init_disp = current_disp;

            inactivity_timer.init(timeout_T);
            inactivity_timer.start();
        }

        if (fabsf(current_disp - init_disp) >= DEGREES(5)) {
            init_disp = current_disp;

            inactivity_timer.init(timeout_T);
            inactivity_timer.start();
        }

        if (!inactivity_timer.isRunning() && (remaining_ms > tone_T)) {
            speaker.cue();
            cue_active = true;

            inactivity_timer.init(tone_T);
            inactivity_timer.start();
        }
    }
    else {
        if (!inactivity_timer.isRunning()) {
            speaker.stop();
            cue_active = false;

            if (remaining_ms > 0) {
                init_disp = current_disp;

                inactivity_timer.init(timeout_T);
                inactivity_timer.start();
            } else {
                inactivity_timer.reset();
            }
        }
    }
}

// main

void setup() {
    // power, serial, RNG initialization
    pinMode(POWER_EN, OUTPUT);
    digitalWrite(POWER_EN, LOW);

    Serial.begin(BAUDRATE);
    randomSeed(analogRead(SEED_PIN));

    // block until host sends phase ID and initial K via serial
    // enable power sources on exit
    String phase_line;
    while (true) {
        if (Serial.available()) {
            phase_line = Serial.readStringUntil('\n');
            phase_line.trim();

            if (phase_line.length() > 0) break;
        }
    }
    phase_id = phase_line.toInt();
    logger.ack();

    String K_line;
    while (true) {
        if (Serial.available()) {
            K_line = Serial.readStringUntil('\n');
            K_line.trim();

            if (K_line.length() > 0) break;
        }
    }
    K = K_line.toInt();
    logger.ack();

    digitalWrite(POWER_EN, HIGH);
    delay(200);

    // initialize external components
    // set session parameters based on the chosen phase ID
    brake.engage();

    speaker.init();

    spout.init();

    lick.init(true);
    lick.calibrate();

    switch (phase_id) {
        case 1: {
            session_T = MINUTES(10);
            trial_T = SECONDS(0);
            delay_T = SECONDS(5);

            threshold = DEGREES(0);
            bidirectional = true;

            break;
        }
        
        case 2: {
            session_T = MINUTES(22.5);
            trial_T = SECONDS(30);
            delay_T = SECONDS(3);

            threshold = DEGREES(0);
            bidirectional = true;

            break;
        }
        
        case 3: {
            session_T = MINUTES(45);
            trial_T = SECONDS(30);
            delay_T = SECONDS(3);

            threshold = DEGREES(30);
            bidirectional = true;

            break;
        }
        
        case 4: {
            session_T = MINUTES(45);
            trial_T = SECONDS(30);
            delay_T = SECONDS(3);

            threshold = DEGREES(60);
            bidirectional = true;
            
            break;
        }
        
        case 5: {
            session_T = MINUTES(45);
            trial_T = SECONDS(30);
            delay_T = SECONDS(3);

            threshold = DEGREES(90);
            bidirectional = true;
            
            break;
        }
    }

    wheel.init(easy_threshold, threshold, bidirectional);

    // initial phase_state -> IDLE
    // initial session_state -> STARTUP
    phase_state = PhaseState::IDLE;
    session_state = SessionState::MAIN;

    // settle delay
    delay(5000);
}

void loop() {
    String received = logger.read();
    if (received.length()) {
        if (received == "E") {
            session_state = SessionState::CLEANUP;
        }
        else {
            K = received.toInt();
        }
    }

    switch (session_state) {
        case SessionState::MAIN: {
            switch (phase_id) {
                case 1: run_phase_1(); break;
                case 2: run_phase_2(); break;
                case 3: run_phase_3_4_5(); break;
                case 4: run_phase_3_4_5(); break;
                case 5: run_phase_3_4_5(); break;
            }

            break;
        }

        case SessionState::CLEANUP: {
            logger.write("S");
            Serial.flush();

            brake.engage();

            phase_timer.reset();
            session_timer.reset();

            delay(500);
            digitalWrite(POWER_EN, LOW);

            for (;;) { delay(1000); }

            break;
        }
    }
}

// phase logic functions
void run_phase_1() {
    switch (phase_state) {
        case PhaseState::IDLE: {
            if (!session_initialized) {
                session_initialized = true;

                session_timer.init(session_T);
                session_timer.start();

                phase_state = PhaseState::HIT;
            }

            break;
        }
        
        case PhaseState::HIT: {
            spout.pulse();
            logger.write("hit");

            phase_state = PhaseState::TRIAL;

            break;
        }
        
        case PhaseState::TRIAL: {
            if (session_timer.isRunning()) {
                uint16_t raw = lick.getRaw();
                if (raw != 0) {
                    logger.write(raw);
                }

                lick.poll();

                if (lick.justTouched()) {
                    logger.write("lick");

                    phase_state = PhaseState::DELAY;
                }
            }
            else {
                spout.pulse();
                logger.write("hit");

                session_state = SessionState::CLEANUP;
            }
            break;
        }
        
        case PhaseState::DELAY: {
            if (session_timer.isRunning()) {
                // entry
                if (!phase_timer.started()) {
                    phase_timer.init(delay_T);
                    phase_timer.start();
                }
                // active
                else {
                    if (phase_timer.isRunning()) {
                        uint16_t raw = lick.getRaw();
                        if (raw != 0) {
                            logger.write(raw);
                        }

                        lick.poll();

                        if (lick.justTouched()) {
                            logger.write("lick");
                        }
                    }
                    else {
                        phase_timer.reset();

                        phase_state = PhaseState::HIT;
                    }
                }
            }
            else {
                session_state = SessionState::CLEANUP;
            }

            break;
        }
    }
}

void run_phase_2() {
    switch (phase_state) {
        case PhaseState::IDLE: {
            // starting IDLE
            if (!session_initialized) {
                // start session timer
                session_timer.init(session_T);
                session_timer.start();

                // flip session initialization flag
                session_initialized = true;

                // IDLE -> CUE
                phase_state = PhaseState::CUE;
            }

            break;
        }
        
        case PhaseState::CUE: {
            // entry
            if (!phase_timer.started()) {
                phase_timer.init(tone_T);
                phase_timer.start();

                logger.write("cue");
                speaker.cue();
            }
            // active
            else {
                // running
                if (phase_timer.isRunning()) {
                    uint16_t raw = lick.getRaw();
                    if (raw != 0) {
                        logger.write(raw);
                    }

                    lick.poll();

                    if (lick.justTouched()) {
                        logger.write("lick");
                    }
                }
                // exit
                else {
                    phase_timer.reset();

                    // CUE -> TRIAL
                    phase_state = PhaseState::TRIAL;
                }
            }

            break;
        }
        
        case PhaseState::TRIAL: {
            // entry
            if (!phase_timer.started()) {
                phase_timer.init(trial_T);
                phase_timer.start();
            }
            // active
            else {
                // running
                if (phase_timer.isRunning()) {
                    uint16_t raw = lick.getRaw();
                    if (raw != 0) {
                        logger.write(raw);
                    }

                    lick.poll();

                    if (lick.justTouched()) {
                        logger.write("lick");
                    }
                }
                // exit
                else {
                    phase_timer.reset();

                    // TRIAL -> HIT
                    phase_state = PhaseState::HIT;
                }
            }

            break;
        }
        
        case PhaseState::HIT: {
            // entry
            if (!phase_timer.started()) {
                phase_timer.init(tone_T);
                phase_timer.start();

                logger.write("hit");
                speaker.hit();
            }
            // active
            else {
                // running
                if (phase_timer.isRunning()) {
                    uint16_t raw = lick.getRaw();
                    if (raw != 0) {
                        logger.write(raw);
                    }

                    lick.poll();

                    if (lick.justTouched()) {
                        logger.write("lick");
                    }
                }
                // exit
                else {
                    phase_timer.reset();
                    spout.pulse();

                    // HIT -> DELAY
                    phase_state = PhaseState::DELAY;
                }
            }
            break;
        }
        
        case PhaseState::DELAY: {
            if (session_timer.isRunning()) {
                // entry
                if (!phase_timer.started()) {
                    phase_timer.init(delay_T);
                    phase_timer.start();
                }
                // active
                else {
                    // running
                    if (phase_timer.isRunning()) {
                        uint16_t raw = lick.getRaw();
                        if (raw != 0) {
                            logger.write(raw);
                        }

                        lick.poll();

                        if (lick.justTouched()) {
                            logger.write("lick");
                        }
                    }
                    // exit
                    else {
                        phase_timer.reset();

                        // DELAY -> CUE
                        phase_state = PhaseState::CUE;
                    }
                }
            }
            else {
                // MAIN -> CLEANUP
                session_state = SessionState::CLEANUP;
            }

            break;
        }
    }
}

void run_phase_3_4_5() {
    switch (phase_state) {        
        case PhaseState::IDLE: {
            if (!session_initialized) {
                // start session timer
                session_timer.init(session_T);
                session_timer.start();

                // flip session initialized flag
                session_initialized = true;

                // IDLE -> CUE
                phase_state = PhaseState::CUE;
            }

            break;
        }
        
        case PhaseState::CUE: {
            // entry
            if (!phase_timer.started()) {
                trial_num += 1;

                logger.write("cue");
                speaker.cue();

                phase_timer.init(tone_T);
                phase_timer.start();
            }
            // active
            else {
                // running
                if (phase_timer.isRunning()) {
                    uint16_t raw = lick.getRaw();
                    if (raw != 0) {
                        logger.write(raw);
                    }

                    lick.poll();

                    if (lick.justTouched()) {
                        logger.write("lick");
                    }
                }
                // exit
                else {
                    phase_timer.reset();
                    brake.release();

                    phase_state = PhaseState::TRIAL;
                }
            }

            break;
        }
        
        case PhaseState::TRIAL: {
            // entry
            if (!phase_timer.started()) {
                bool easy_trial;
                if (trial_num <= 20) {
                    easy_trial = (((trial_num - 1) % 5) == 0);
                } else {
                    easy_trial = (((trial_num - 21) % K) == 0);
                }

                wheel.reset(easy_trial);
                last_disp_mark = LONG_MIN;

                phase_timer.init(trial_T);
                phase_timer.start();
            }
            // active
            else {
                // running
                if (phase_timer.isRunning()) {
                    uint16_t raw = lick.getRaw();
                    if (raw != 0) {
                        logger.write(raw);
                    }

                    lick.poll();
                    
                    if (lick.justTouched()) {
                        logger.write("lick");
                    }

                    wheel.update();
                    float disp = wheel.getDisplacement();
                    float nearest;
                    if (nearMultiple(disp, 0.5f, 0.1f, &nearest)) {
                        long mark = lroundf(nearest);
                        if (mark != last_disp_mark) {
                            logger.write(nearest);
                            last_disp_mark = mark;
                        }
                    }

                    checkInactivity(disp);

                    // success exit
                    if (wheel.thresholdReached()) {
                        phase_timer.reset();

                        // TRIAL -> HIT
                        trial_hit = true;
                        phase_state = PhaseState::HIT;
                    }
                    else if (wheel.thresholdMissed()) {
                        phase_timer.reset();

                        // TRIAL -> MISS
                        trial_hit = false;
                        phase_state = PhaseState::MISS;
                    }
                }
                // failure exit
                else {
                    phase_timer.reset();

                    // TRIAL -> MISS
                    trial_hit = false;
                    phase_state = PhaseState::MISS;
                }
            }

            break;
        }

        case PhaseState::HIT: {
            // entry
            if (!phase_timer.started()) {
                logger.write("hit");
                brake.engage();
                speaker.hit();

                phase_timer.init(tone_T);
                phase_timer.start();
            }
            // active
            else {
                // running
                if (phase_timer.isRunning()) {
                    uint16_t raw = lick.getRaw();
                    if (raw != 0) {
                        logger.write(raw);
                    }

                    lick.poll();

                    if (lick.justTouched()) {
                        logger.write("lick");
                    }

                    // if (phase_timer.timeElapsed() >= (tone_T >> 3)) {
                    //     brake.engage();
                    // }

                    if ((phase_timer.timeElapsed() >= (tone_T >> 1)) && !reward_given) {
                        spout.pulse();
                        reward_given = true;
                    }
                // exit
                } else {
                    phase_timer.reset();
                    reward_given = false;

                    // HIT -> DELAY
                    phase_state = PhaseState::DELAY;
                }
            }

            break;
        }

        case PhaseState::MISS: {
            // entry
            if (!phase_timer.started()) {
                logger.write("miss");
                brake.engage();
                speaker.miss();

                phase_timer.init(tone_T);
                phase_timer.start();
            }
            // active
            else {
                // running
                if (phase_timer.isRunning()) {
                    uint16_t raw = lick.getRaw();
                    if (raw != 0) {
                        logger.write(raw);
                    }

                    lick.poll();

                    if (lick.justTouched()) {
                        logger.write("lick");
                    }

                    // if (phase_timer.timeElapsed() >= (tone_T >> 3)) {
                    //     brake.engage();
                    // }
                }
                // exit
                else {
                    phase_timer.reset();

                    // MISS -> DELAY
                    phase_state = PhaseState::DELAY;
                }
            }

            break;
        }
        
        case PhaseState::DELAY: {
            if (session_timer.isRunning()) {
                // entry
                if (!phase_timer.started()) {
                    phase_timer.init(delay_T);
                    phase_timer.start();
                }
                // active
                else {
                    // running
                    if (phase_timer.isRunning()) {
                        uint16_t raw = lick.getRaw();
                        if (raw != 0) {
                            logger.write(raw);
                        }

                        lick.poll();

                        if (lick.justTouched()) {
                            logger.write("lick");
                        }
                    }
                    // exit
                    else {
                        phase_timer.reset();

                        // DELAY -> CUE
                        phase_state = PhaseState::CUE;
                    }
                }
            }
            else {
                // DELAY -> CLEANUP
                session_state = SessionState::CLEANUP;
            }

            break;
        }
    }
}

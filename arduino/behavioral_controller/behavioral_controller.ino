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

template <typename T>
constexpr unsigned long SECONDS(T s) { return static_cast<unsigned long>(s * 1000.0f); }
template <typename T>
constexpr unsigned long MINUTES(T m) { return static_cast<unsigned long>(m * 60.0f * 1000.0f); }
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
unsigned long blink_T = SECONDS(0.05);
unsigned long wait_T = SECONDS(2.5);

float easy_threshold = DEGREES(15);
float threshold;

bool session_initialized = false;
bool trial_hit;
bool reward_given = false;

long last_disp_mark = LONG_MIN;

bool next_easy_trial = false;
char next_alignment = 'B';

// phase logic forward declarations

void run_phase_1();
void run_phase_2();
void run_phase_3_plus();

// helper functions
void blink(unsigned long blink_ms, unsigned long wait_ms) {
    static unsigned long t0 = 0;
    static bool led_on = false;

    unsigned long now = millis();

    if (led_on) {
        if (now - t0 >= blink_ms) {
            digitalWrite(LED_BUILTIN, LOW);
            
            led_on = false;
            t0 = now;
        }
    } else {
        if (now - t0 >= wait_ms - blink_ms) {
            digitalWrite(LED_BUILTIN, HIGH);

            led_on = true;
            t0 = now;
        }
    }
}

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

static bool parseConfig(const String& line, bool* easy_out, char* align_out) {
    if (!line.startsWith("T")) return false;

    int first_space = line.indexOf(' ');
    if (first_space < 0) return false;

    int second_space = line.indexOf(' ', first_space + 1);
    if (second_space < 0) return false;

    String easy_str = line.substring(first_space + 1, second_space);
    String align_str = line.substring(second_space + 1);
    align_str.trim();

    if (align_str.length() < 1) return false;

    int e = easy_str.toInt();
    char a = align_str.charAt(0);

    if (!(a == 'L' || a == 'l' || a == 'R' || a == 'r' || a == 'B' || a == 'b')) return false;

    if (easy_out) *easy_out = (e != 0);
    if (align_out) *align_out = a;

    return true;
}

static void drainSerial() {
    while (true) {
        String msg = logger.read();
        if (!msg.length()) break;

        if (msg == "E") {
            session_state = SessionState::CLEANUP;
            logger.ack();
            continue;
        }

        if (phase_id >= 3) {
            bool e;
            char a;

            if (parseConfig(msg, &e, &a)) {
                next_easy_trial = e;
                next_alignment = a;

                logger.ack();
            }
        }
    }
}

static void waitForPhase() {
    while (true) {
        String msg = logger.read();
        if (!msg.length()) continue;

        int p = msg.toInt();
        if (p > 0) {
            phase_id = p;
            logger.ack();
            return;
        }
    }
}

static void waitForInitConfig() {
    while (true) {
        String msg = logger.read();
        if (!msg.length()) continue;

        if (msg == "E") {
            session_state = SessionState::CLEANUP;
            logger.ack();
            return;
        }

        bool e;
        char a;

        if (parseConfig(msg, &e, &a)) {
            next_easy_trial = e;
            next_alignment = a;
            
            logger.ack();
            return;
        }
    }
}

// main
void setup() {
    // power, status LED, serial, and RNG initialization
    pinMode(POWER_EN, OUTPUT);
    digitalWrite(POWER_EN, LOW);

    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);

    Serial.begin(BAUDRATE);

    randomSeed(analogRead(SEED_PIN));

    // block until host sends phase ID and initial trial config
    waitForPhase();

    if (phase_id >= 3) {
        waitForInitConfig();
    } else {
        next_easy_trial = false;
        next_alignment = 'B';
    }

    digitalWrite(POWER_EN, HIGH);
    delay(200);

    // initialize external components
    // set session parameters based on phase ID
    brake.engage();

    speaker.init();

    spout.init();

    lick.init();
    lick.calibrate();

    switch (phase_id) {
        case 1: {
            session_T = MINUTES(10);
            trial_T = SECONDS(0);
            delay_T = SECONDS(5);
            threshold = DEGREES(0);
            break;
        }
        case 2: {
            session_T = MINUTES(22.5);
            trial_T = SECONDS(30);
            delay_T = SECONDS(3);
            threshold = DEGREES(0);
            break;
        }
        case 3: {
            session_T = MINUTES(45);
            trial_T = SECONDS(30);
            delay_T = SECONDS(3);
            threshold = DEGREES(30);
            break;
        }
        case 4: {
            session_T = MINUTES(45);
            trial_T = SECONDS(30);
            delay_T = SECONDS(3);
            threshold = DEGREES(60);
            break;
        }
        case 5: {
            session_T = MINUTES(45);
            trial_T = SECONDS(30);
            delay_T = SECONDS(3);
            threshold = DEGREES(90);
            break;
        }
        case 6: {
            session_T = MINUTES(45);
            trial_T = SECONDS(30);
            delay_T = SECONDS(3);
            threshold = DEGREES(30);
            break;
        }
        case 7: {
            session_T = MINUTES(45);
            trial_T = SECONDS(30);
            delay_T = SECONDS(3);
            threshold = DEGREES(60);
            break;
        }
        case 8: {
            session_T = MINUTES(45);
            trial_T = SECONDS(30);
            delay_T = SECONDS(3);
            threshold = DEGREES(90);
            break;
        }
    }

    wheel.init(easy_threshold, threshold, next_alignment);

    // set initial states
    phase_state = PhaseState::IDLE;
    session_state = SessionState::MAIN;

    // settle delay
    delay(5000);
}

void loop() {
    drainSerial();
    blink(blink_T, wait_T);

    switch (session_state) {
        case SessionState::MAIN: {
            switch (phase_id) {
                case 1: run_phase_1(); break;
                case 2: run_phase_2(); break;
                default: run_phase_3_plus(); break;
            }

            break;
        }

        case SessionState::CLEANUP: {
            logger.write("S");

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

void run_phase_3_plus() {
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
                logger.write("cue");
                speaker.cue();

                phase_timer.init(tone_T);
                phase_timer.start();
            }
            // active
            else {
                // running
                if (phase_timer.isRunning()) {
                    lick.poll();
                    if (lick.justTouched()) {
                        logger.write("lick");
                    }

                    // if (phase_timer.timeElapsed() >= (tone_T - (tone_T  >> 2))) {
                    //     brake.release();
                    // }
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
                wheel.reset(next_easy_trial, next_alignment);
                last_disp_mark = LONG_MIN;

                phase_timer.init(trial_T);
                phase_timer.start();
            }
            // active
            else {
                // running
                if (phase_timer.isRunning()) {
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

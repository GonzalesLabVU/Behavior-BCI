// ---------------------------
// IMPORTS
// ---------------------------
#include "Wheel.h"
#include "Brake.h"
#include "Lick.h"
#include "Speaker.h"
#include "Spout.h"
#include "Logger.h"
#include "Timer.h"

#include <math.h>
#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

// ---------------------------
// CONVERSION HANDLES
// ---------------------------
template <typename T>
constexpr unsigned long MILLISECONDS(T s) { return static_cast<unsigned long>(s * 1000.0f); }

template <typename T>
constexpr unsigned long MICROSECONDS(T s) { return static_cast<unsigned long>(s * 1000000.0f); }

template <typename T>
constexpr unsigned long SECONDS(T s) { return static_cast<unsigned long>(s * 1000.0f); }

template <typename T>
constexpr unsigned long MINUTES(T m) { return static_cast<unsigned long>(m * 60.0f * 1000.0f); }

template <typename T>
constexpr float DEGREES(T d) { return static_cast<float>(d); }

// ---------------------------
// CONFIG
// ---------------------------
#define BAUDRATE 1000000
#define RAW_FLAG false
#define SEED_PIN A0
#define POWER_EN 7

static constexpr uint32_t RAW_HZ = 100;
static constexpr uint32_t RAW_US = 1000000UL / RAW_HZ;
static constexpr unsigned long sample_T = MINUTES(5);

// ---------------------------
// STATE
// ---------------------------
enum class SessionState { MAIN, CLEANUP };
enum class PhaseState { IDLE, CUE, TRIAL, HIT, MISS, DELAY };

SessionState session_state = SessionState::MAIN;
PhaseState phase_state = PhaseState::IDLE;

// ---------------------------
// COMPONENTS
// ---------------------------
Brake brake;
Lick lick;
Wheel wheel;
Spout spout;
Speaker speaker;

Timer session_timer;
Timer phase_timer;
Timer raw_timer;

Logger logger;

// ---------------------------
// TRIAL CONFIG
// ---------------------------
struct SessionConfig {
    int phase = 3;
    float engage_ms = 0.0f;
    float release_ms = 0.0f;
    float pulse_ms = 0.0f;
    float threshold = 30.0f;
    char side = 'B';
    bool reverse = false;
};

struct TrialConfig {
    long trial_n = 0;
    bool easy = false;
    char side = 'B';
    bool pending = false;
};

static SessionConfig session_cfg;
static TrialConfig trial_cfg;

static unsigned long session_T = MINUTES(45);
static unsigned long trial_T = SECONDS(30);
static unsigned long delay_T = SECONDS(3);
static unsigned long tone_T = SECONDS(1);

static float easy_threshold = DEGREES(15.0f);

static bool session_initialized = false;
static bool trial_hit = false;
static bool reward_given = false;
static long last_disp_mark = LONG_MIN;

// ---------------------------
// SERIAL HELPERS
// ---------------------------
static bool readLine(char* buf, size_t cap) {
    if (!Serial.available()) return false;

    size_t n = Serial.readBytesUntil('\n', buf, cap - 1);
    if (n == 0) return false;

    buf[n] = '\0';

    while (n && (buf[n - 1] == '\r' || buf[n - 1] == ' ' || buf[n - 1] == '\t')) {
        buf[--n] = '\0';
    }

    char* p = buf;
    while (*p == ' ' || *p == '\t') p++;
    if (p != buf) memmove(buf, p, strlen(p) + 1);

    return buf[0] != '\0';
}

static bool isSideChar(char c) {
    c = (char)toupper((unsigned char)c);
    return (c == 'L' || c == 'R' || c == 'B');
}

static void applyPhaseDefaults(int phase_id) {
    if (phase_id == 1) {
        session_T = MINUTES(10);
        trial_T = SECONDS(0);
        delay_T = SECONDS(5);
    } else if (phase_id == 2) {
        session_T = MINUTES(22.5f);
        trial_T = SECONDS(30);
        delay_T = SECONDS(3);
    } else {
        session_T = MINUTES(2);
        trial_T = SECONDS(30);
        delay_T = SECONDS(3);
    }
}

static bool parseTrialConfig(char* line) {
    if (isalpha((unsigned char)line[0])) return false;

    char* s = line;
    char* t0 = strtok(s, " \t");
    char* t1 = strtok(nullptr, " \t");
    char* t2 = strtok(nullptr, " \t");

    if (!t0 || !t1) return false;

    char* end0 = nullptr;
    long tn = strtol(t0, &end0, 10);
    if (!end0 || *end0 != '\0') return false;

    char* end1 = nullptr;
    long ef = strtol(t1, &end1, 10);
    if (!end1 || *end1 != '\0') return false;

    trial_cfg.trial_n = tn;
    trial_cfg.easy = (ef != 0);
    trial_cfg.side = session_cfg.side;

    if (t2 && t2[0] && isSideChar(t2[0])) {
        trial_cfg.side = (char)toupper((unsigned char)t2[0]);
    }

    trial_cfg.pending = true;
    return true;
}

static bool parseKeyValue(char* line) {
    char* key = strtok(line, " \t");
    char* val = strtok(nullptr, " \t");
    if (!key || !val) return false;

    if (strcmp(key, "engage") == 0) {
        session_cfg.engage_ms = atof(val);
        return (session_cfg.engage_ms > 0.0f);
    }
    if (strcmp(key, "release") == 0) {
        session_cfg.release_ms = atof(val);
        return (session_cfg.release_ms > 0.0f);
    }
    if (strcmp(key, "pulse") == 0) {
        session_cfg.pulse_ms = atof(val);
        return (session_cfg.pulse_ms > 0.0f);
    }
    if (strcmp(key, "threshold") == 0) {
        session_cfg.threshold = DEGREES(atof(val));
        return (session_cfg.threshold >= 0.0f);
    }
    if (strcmp(key, "side") == 0) {
        char c = (char)toupper((unsigned char)val[0]);
        if (!isSideChar(c)) return false;

        session_cfg.side = c;
        return true;
    }
    if (strcmp(key, "reverse") == 0) {
        long v = strtol(val, nullptr, 10);
        if (!(v == 0 || v == 1)) return false;

        session_cfg.reverse = (v == 1);
        return true;
    }
    if (strcmp(key, "phase") == 0) {
        long v = strtol(val, nullptr, 10);
        if (v < 1 || v > 99) return false;

        session_cfg.phase = (int)v;
        return true;
    }

    return false;
}

static void waitForHandshake() {
    bool have_engage = false;
    bool have_release = false;
    bool have_pulse = false;
    bool have_threshold = false;
    bool have_side = false;
    bool have_reverse = false;
    bool have_trial = false;

    session_cfg.phase = 3;

    char line[96];

    for (;;) {
        if (!readLine(line, sizeof(line))) continue;

        // control override messages
        if (strcmp(line, "E") == 0) {
            session_state = SessionState::CLEANUP;
            logger.ack();
            return;
        }

        // trial config settings
        {
            char tmp[96];
            strncpy(tmp, line, sizeof(tmp));
            tmp[sizeof(tmp)-1] = 0;

            if (!have_trial && parseTrialConfig(tmp)) {
                have_trial = true;
                logger.ack();
                goto check_done;
            }
        }

        // session config settings
        {
            char tmp[96];
            strncpy(tmp, line, sizeof(tmp));
            tmp[sizeof(tmp)-1] = 0;

            if (parseKeyValue(tmp)) {
                if (strncmp(line, "engage", 6) == 0) have_engage = (session_cfg.engage_ms > 0.0f);
                if (strncmp(line, "release", 7) == 0) have_release = (session_cfg.release_ms > 0.0f);
                if (strncmp(line, "pulse", 5) == 0) have_pulse = (session_cfg.pulse_ms > 0.0f);
                if (strncmp(line, "threshold", 9) == 0) have_threshold = true;
                if (strncmp(line, "side", 4) == 0) have_side = true;
                if (strncmp(line, "reverse", 7) == 0) have_reverse = true;

                logger.ack();
            }
        }
    
    check_done:
        if (have_engage && have_release && have_pulse && have_threshold && have_side && have_reverse && have_trial) {
            applyPhaseDefaults(session_cfg.phase);
            return;
        }
    }
}

static void drainSerial() {
    char line[96];

    while (readLine(line, sizeof(line))) {
        // control override messages
        if (strcmp(line, "E") == 0) {
            session_state = SessionState::CLEANUP;
            logger.ack();
            continue;
        }

        if (strcmp(line, "H") == 0 || strcmp(line, "M") == 0) {
            logger.ack();

            if (phase_state == PhaseState::TRIAL) {
                phase_timer.reset();
                trial_hit = (line[0] == 'H');
                reward_given = false;
                phase_state = trial_hit ? PhaseState::HIT : PhaseState::MISS;
            }

            continue;
        }

        // trial config settings
        {
            char tmp[96];
            strncpy(tmp, line, sizeof(tmp));
            tmp[sizeof(tmp)-1] = 0;

            if (parseTrialConfig(tmp)) {
                logger.ack();
                continue;
            }
        }

        // session config settings
        {
            char tmp[96];
            strncpy(tmp, line, sizeof(tmp));
            tmp[sizeof(tmp)-1] = 0;

            if (parseKeyValue(tmp)) {
                logger.ack();
                continue;
            }
        }
    }
}

// ---------------------------
// DATA HELPERS
// ---------------------------
inline bool nearMultiple(float x, float step, float tol, float* nearest_out) {
    float q = roundf(x / step);
    float m = q * step;

    if (nearest_out) *nearest_out = m;

    return fabsf(x - m) <= tol;
}

static void checkInactivity(float current_disp) {
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
    } else {
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

// ---------------------------
// TOP LEVEL
// ---------------------------
void run_phase_1();
void run_phase_2();
void run_phase_3_plus();

void setup() {
    pinMode(POWER_EN, OUTPUT);
    digitalWrite(POWER_EN, LOW);

    Serial.begin(BAUDRATE);
    randomSeed(analogRead(SEED_PIN));

    while (!Serial) {}
    waitForHandshake();

    unsigned long engage_us = (unsigned long)(session_cfg.engage_ms * 1000.0f);
    unsigned long release_us = (unsigned long)(session_cfg.release_ms * 1000.0f);
    unsigned long pulse_us = (unsigned long)(session_cfg.pulse_ms * 1000.0f);

    digitalWrite(POWER_EN, HIGH);
    delay(200);

    brake.init(engage_us, release_us);
    brake.engage();

    speaker.init();

    spout.init(pulse_us);

    lick.init(RAW_FLAG);
    lick.calibrate();

    wheel.init(easy_threshold, session_cfg.threshold, trial_cfg.side, session_cfg.reverse);

    delay(2500);

    raw_timer.init(sample_T);
    raw_timer.start();

    session_state = SessionState::MAIN;
    phase_state = PhaseState::IDLE;
}

void loop() {
    drainSerial();

    switch (session_state) {
        case SessionState::MAIN: {
            switch (session_cfg.phase) {
                case 1: run_phase_1(); break;
                case 2: run_phase_2(); break;
                default: run_phase_3_plus(); break;
            }

            static uint32_t last_raw_us = 0;
            if (RAW_FLAG && raw_timer.isRunning()) {
                uint32_t now_us = micros();

                if ((uint32_t)(now_us - last_raw_us) >= RAW_US) {
                    last_raw_us += RAW_US;

                    uint16_t raw_val = lick.sampleRaw();
                    logger.writeRaw(raw_val);
                }
            } else {
                last_raw_us = micros();
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
                lick.sampleFiltered();
                if (lick.justTouched()) {
                    logger.write("lick");

                    phase_state = PhaseState::DELAY;
                }

                spout.poll();
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
                        lick.sampleFiltered();

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
                    lick.sampleFiltered();

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
                    lick.sampleFiltered();
                    if (lick.justTouched()) {
                        logger.write("lick");
                    }

                    spout.poll();
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
                    lick.sampleFiltered();

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
                        lick.sampleFiltered();

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
                session_timer.init(session_T);
                session_timer.start();

                session_initialized = true;
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
                    lick.sampleFiltered();
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
                wheel.reset(trial_cfg.easy, trial_cfg.side);
                trial_cfg.pending = false;

                last_disp_mark = LONG_MIN;

                phase_timer.init(trial_T);
                phase_timer.start();
            }
            // active
            else {
                // running
                if (phase_timer.isRunning()) {
                    lick.sampleFiltered();
                    if (lick.justTouched()) {
                        logger.write("lick");
                    }

                    wheel.update();
                    float disp = wheel.displacement;

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

                    spout.poll();
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
                    lick.sampleFiltered();
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
                    lick.sampleFiltered();
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
                        lick.sampleFiltered();
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

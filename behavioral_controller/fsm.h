#pragma once

// IMPORTS
#include "config.h"
#include "state.h"
#include "comms.h"

// DATA HELPERS
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
            writeSerial("r_cue");
            
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

// STATE MACHINES
void run_phase_0() {
    switch (phase_state) {
        case PhaseState::IDLE: {
            if (!session_initialized) {
                session_initialized = true;

                session_timer.init(session_T);
                session_timer.start();

                phase_state = PhaseState::CUE;
            }

            break;
        }

        case PhaseState::CUE: {
            if (!phase_timer.started()) {
                phase_timer.init(tone_T);
                phase_timer.start();

                writeSerial("cue");
            }
            else {
                if (!phase_timer.isRunning()) {
                    phase_timer.reset();

                    phase_state = PhaseState::TRIAL;
                }
            }

            break;
        }

        case PhaseState::TRIAL: {
            if (!phase_timer.started()) {
                phase_timer.init(trial_T);
                phase_timer.start();
            }
            else {
                if (phase_timer.isRunning()) {
                    lick.sampleFiltered();

                    if (lick.justTouched()) {
                        writeSerial("lick");
                    }
                }
                else {
                    phase_timer.reset();

                    phase_state = PhaseState::HIT;
                }
            }

            break;
        }

        case PhaseState::HIT: {
            if (!phase_timer.started()) {
                phase_timer.init(tone_T);
                phase_timer.start();

                writeSerial("hit");
            }
            else {
                if (!phase_timer.isRunning()) {
                    phase_timer.reset();

                    phase_state = PhaseState::DELAY;
                }
            }

            break;
        }

        case PhaseState::DELAY: {
            if (session_timer.isRunning()) {
                if (!phase_timer.started()) {
                    phase_timer.init(delay_T);
                    phase_timer.start();
                }
                else {
                    if (!phase_timer.isRunning()) {
                        phase_timer.reset();

                        phase_state = PhaseState::CUE;
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

void run_phase_1() {
    switch (phase_state) {
        case PhaseState::IDLE: {
            if (!session_initialized) {
                session_initialized = true;

                session_timer.init(session_T);
                session_timer.start();

                phase_state = PhaseState::CUE;
            }

            break;
        }

        case PhaseState::CUE: {
            // entry
            writeSerial("trial_start");
            phase_state = PhaseState::TRIAL;

            break;
        }
        
        case PhaseState::HIT: {
            // entry
            if (!phase_timer.started()) {
                writeSerial("hit");

                phase_timer.init(tone_T);
                phase_timer.start();
            }
            // active
            else {
                // running
                if (phase_timer.isRunning()) {
                    lick.sampleFiltered();
                    if (lick.justTouched()) {
                        writeSerial("lick");
                    }

                    if ((phase_timer.timeElapsed() >= (tone_T >> 1)) && !reward_given) {
                        spout.pulse();
                        reward_given = true;
                    }
                // exit
                } else {
                    phase_timer.reset();
                    reward_given = false;

                    // HIT -> DELAY
                    writeSerial("trial_stop");
                    phase_state = PhaseState::DELAY;
                }
            }

            break;
        }
        
        case PhaseState::TRIAL: {
            if (session_timer.isRunning()) {
                if (!phase_timer.started()) {
                    phase_timer.init(trial_T);
                    phase_timer.start();
                }

                lick.sampleFiltered();
                if (lick.justTouched()) {
                    writeSerial("lick");

                    phase_timer.reset();
                    phase_state = PhaseState::HIT;
                }
                else if (!phase_timer.isRunning()) {
                    phase_timer.reset();

                    phase_state = PhaseState::HIT;
                }
            }
            else {
                spout.pulse();

                session_state = SessionState::CLEANUP;
            }
            break;
        }
        
        case PhaseState::DELAY: {
            phase_state = PhaseState::CUE;

            break;
        }
    }
}

void run_phase_2() {
    switch (phase_state) {
        case PhaseState::IDLE: {
            if (!session_initialized) {
                brake.release();

                session_timer.init(session_T);
                session_timer.start();

                session_initialized = true;
                phase_state = PhaseState::CUE;
            }

            break;
        }

        case PhaseState::CUE: {
            // entry
            writeSerial("trial_start");
            phase_state = PhaseState::TRIAL;

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
                        writeSerial("lick");
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
                writeSerial("hit");

                spout.pulse();

                phase_timer.init(tone_T);
                phase_timer.start();
            }
            // active
            else {
                // running
                if (phase_timer.isRunning()) {
                    lick.sampleFiltered();
                    if (lick.justTouched()) {
                        writeSerial("lick");
                    }
                // exit
                } else {
                    phase_timer.reset();
                    reward_given = false;

                    // HIT -> DELAY
                    writeSerial("trial_stop");
                    phase_state = PhaseState::DELAY;
                }
            }

            break;
        }

        case PhaseState::MISS: {
            // entry
            if (!phase_timer.started()) {
                writeSerial("miss");

                phase_timer.init(tone_T);
                phase_timer.start();
            }
            // active
            else {
                // running
                if (phase_timer.isRunning()) {
                    lick.sampleFiltered();
                    if (lick.justTouched()) {
                        writeSerial("lick");
                    }
                }
                // exit
                else {
                    phase_timer.reset();

                    // MISS -> DELAY
                    writeSerial("trial_stop");
                    phase_state = PhaseState::DELAY;
                }
            }

            break;
        }

        case PhaseState::DELAY: {
            if (session_timer.isRunning()) {
                phase_state = PhaseState::CUE;
            }
            else {
                // DELAY -> CLEANUP
                session_state = SessionState::CLEANUP;
            }

            break;
        }
    }
}

void run_phase_3() {
    switch (phase_state) {        
        case PhaseState::IDLE: {
            if (!session_initialized) {
                brake.release();

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
                writeSerial("cue");

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
                        writeSerial("lick");
                    }
                }
                // exit
                else {
                    phase_timer.reset();

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
                        writeSerial("lick");
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
                writeSerial("hit");

                speaker.hit();
                spout.pulse();

                phase_timer.init(tone_T);
                phase_timer.start();
            }
            // active
            else {
                // running
                if (phase_timer.isRunning()) {
                    lick.sampleFiltered();
                    if (lick.justTouched()) {
                        writeSerial("lick");
                    }
                // exit
                } else {
                    phase_timer.reset();
                    reward_given = false;

                    // HIT -> DELAY
                    writeSerial("trial_stop");
                    phase_state = PhaseState::DELAY;
                }
            }

            break;
        }

        case PhaseState::MISS: {
            // entry
            if (!phase_timer.started()) {
                writeSerial("miss");

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
                        writeSerial("lick");
                    }
                }
                // exit
                else {
                    phase_timer.reset();

                    // MISS -> DELAY
                    writeSerial("trial_stop");
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
                            writeSerial("lick");
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

void run_phase_4_plus() {
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
                writeSerial("cue");

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
                        writeSerial("lick");
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
                        writeSerial("lick");
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
                writeSerial("hit");

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
                        writeSerial("lick");
                    }

                    if ((phase_timer.timeElapsed() >= (tone_T >> 1)) && !reward_given) {
                        spout.pulse();
                        reward_given = true;
                    }
                // exit
                } else {
                    phase_timer.reset();
                    reward_given = false;

                    // HIT -> DELAY
                    writeSerial("trial_stop");
                    phase_state = PhaseState::DELAY;
                }
            }

            break;
        }

        case PhaseState::MISS: {
            // entry
            if (!phase_timer.started()) {
                writeSerial("miss");

                speaker.stop();
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
                        writeSerial("lick");
                    }
                }
                // exit
                else {
                    phase_timer.reset();

                    // MISS -> DELAY
                    writeSerial("trial_stop");
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
                            writeSerial("lick");
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

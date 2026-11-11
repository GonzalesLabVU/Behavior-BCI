// IMPORTS
#include <avr/wdt.h>

#include "config.h"
#include "state.h"
#include "comms.h"
#include "fsm.h"

// MAIN
void setup() {
    MCUSR = 0;
    wdt_disable();

    pinMode(POWER_EN, OUTPUT);
    digitalWrite(POWER_EN, LOW);

    pinMode(RECORD_TTL, OUTPUT);
    digitalWrite(RECORD_TTL, LOW);
    pinMode(EVENT_TTL, OUTPUT);
    digitalWrite(EVENT_TTL, LOW);

    Serial.begin(BAUDRATE);
    randomSeed(analogRead(SEED_PIN));

    parseFlushCommand();

    if (session_cfg.flush) {
        digitalWrite(POWER_EN, HIGH);
        delay(100);

        spout.init(0);
        spout.flush(10000);

        logger.write("R");
        Serial.flush();

        delay(100);
        digitalWrite(POWER_EN, LOW);

        wdt_enable(WDTO_15MS);
        while (1) {}
    }

    waitForHandshake();

    unsigned long engage_us = (unsigned long)(session_cfg.engage_ms * 1000.0f);
    unsigned long release_us = (unsigned long)(session_cfg.release_ms * 1000.0f);
    unsigned long pulse_us = (unsigned long)(session_cfg.pulse_ms * 1000.0f);

    digitalWrite(POWER_EN, HIGH);
    delay(100);

    brake.init(engage_us, release_us);
    brake.engage();

    String side = (session_cfg.side == 'R') ? "R" : "L";
    speaker.init(side);

    spout.init(pulse_us);

    for (int i = 0; i < 25; i++) {
        spout.flush(10);
        delay(150);
    }

    lick.init(RAW_FLAG);
    // lick.calibrate();

    wheel.init(easy_threshold, session_cfg.threshold, trial_cfg.side, session_cfg.reverse);

    parseStartCommand();

    raw_timer.init(sample_T);
    raw_timer.start();

    session_state = SessionState::MAIN;
    phase_state = PhaseState::IDLE;
}

void loop() {
    drainSerial();
    brake.update();

    switch (session_state) {
        case SessionState::MAIN: {
            switch (session_cfg.phase) {
                case 0: run_phase_0(); break;
                case 1: run_phase_1(); break;
                case 2: run_phase_2(); break;
                case 3: run_phase_3(); break;
                default: run_phase_4_plus(); break;
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
            digitalWrite(RECORD_TTL, LOW);
            digitalWrite(EVENT_TTL, LOW);

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

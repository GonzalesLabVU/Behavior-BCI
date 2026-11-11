#pragma once

// IMPORTS
#include "config.h"
#include "state.h"

// TTL
int eventTypeToPulseCount(const String& eventType) {
    if (eventType == "lick") return 1;
    if (eventType == "cue") return 2;
    if (eventType == "r_cue") return 2;
    if (eventType == "hit") return 3;
    if (eventType == "miss") return 3;
    if (eventType == "img_start") return 4;
    if (eventType == "img_stop") return 5;

    return 0;
}

bool sendPulseTrain(const String& eventType) {
    if (!session_cfg.ephys) return false;

    int count = eventTypeToPulseCount(eventType);
    if (count <= 0) return false;

    for (int i = 0; i < count; i++) {
        digitalWrite(EVENT_TTL, HIGH);
        delayMicroseconds(EVENT_PULSE_US);

        digitalWrite(EVENT_TTL, LOW);

        if (i < count - 1) {
            delayMicroseconds(EVENT_IPI_US);
        }
    }

    delayMicroseconds(EVENT_TRAIN_GAP_US);

    return true;
}

// SERIAL
static bool readLine(char* buf, size_t cap) {
    static char line_buf[96];
    static size_t line_len = 0;

    while (Serial.available()) {
        char c = (char)Serial.read();

        if (c == '\n') {
            size_t n = line_len;

            while (n && (line_buf[n-1] == '\r' || line_buf[n-1] == ' ' || line_buf[n-1] == '\t')) {
                n--;
            }

            char* p = line_buf;
            while (n && (*p == ' ' || *p == '\t')) {
                p++;
                n--;
            }

            size_t out_len = (n < cap - 1) ? n : (cap - 1);
            memcpy(buf, p, out_len);
            buf[out_len] = '\0';

            line_len = 0;

            if (buf[0] == '\0') return false;
            return true;
        }

        if (line_len < sizeof(line_buf) - 1) {
            line_buf[line_len++] = c;
        }
        else {
            line_len = 0;
        }
    }

    return false;
}

static bool isSideChar(char c) {
    c = (char)toupper((unsigned char)c);
    return (c == 'L' || c == 'R' || c == 'B');
}

static void applyPhaseDefaults(int phase_id) {
    if (phase_id == 0) {
        session_T = MINUTES(20);
        trial_T = SECONDS(30);
        delay_T = SECONDS(3);
    } else if (phase_id == 1) {
        // session_T = MINUTES(10);
        session_T = MINUTES(1);
        trial_T = SECONDS(5);
        delay_T = SECONDS(3);
    } else if (phase_id == 2) {
        session_T = MINUTES(20);
        trial_T = SECONDS(30);
        delay_T = SECONDS(1);
    } else if (phase_id == 3) {
        session_T = MINUTES(20);
        trial_T = SECONDS(30);
        delay_T = SECONDS(1);
    } else {
        session_T = MINUTES(30);
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
        if (v < 0 || v > 99) return false;

        session_cfg.phase = (int)v;
        return true;
    }
    if (strcmp(key, "ephys") == 0) {
        long v = strtol(val, nullptr, 10);
        if (!(v == 0 || v == 1)) return false;

        session_cfg.ephys = (v == 1);
        return true;
    }

    return false;
}

static bool parseFlushLine(char* line) {
    char* key = strtok(line, " \t");
    char* val = strtok(nullptr, " \t");
    char* extra = strtok(nullptr, " \t");

    if (!key || !val || extra) return false;
    if (strcmp(key, "flush") != 0) return false;

    char* end = nullptr;
    long v = strtol(val, &end, 10);

    if (!end || *end != '\0') return false;
    if (!(v == 0 || v == 1)) return false;

    session_cfg.flush = (v == 1);
    return true;
}

void parseFlushCommand() {
    char line[96];

    for (;;) {
        if (readLine(line, sizeof(line)) && parseFlushLine(line)) {
            logger.ack();
            return;
        }

        delay(10);
    }
}

static bool parseStartLine(char* line) {
    char* key = strtok(line, " \t");
    char* val = strtok(nullptr, " \t");
    char* extra = strtok(nullptr, " \t");

    if (!key || !val || extra) return false;
    if (strcmp(key, "start") != 0) return false;

    char* end = nullptr;
    long v = strtol(val, &end, 10);

    if (!end || *end != '\0') return false;
    if (!(v == 0 || v == 1)) return false;

    return true;
}

void parseStartCommand() {
    char line[96];

    for (;;) {
        if (readLine(line, sizeof(line))) {
            if (strcmp(line, "R1") == 0) {
                digitalWrite(RECORD_TTL, HIGH);
                logger.ack();
                continue;
            }

            if (strcmp(line, "R2") == 0) {
                digitalWrite(RECORD_TTL, LOW);
                logger.ack();
                continue;
            }

            if (strcmp(line, "img_start") == 0) {
                sendPulseTrain("img_start");
                logger.ack();
                continue;
            }

            if (strcmp(line, "img_stop") == 0) {
                sendPulseTrain("img_stop");
                logger.ack();
                continue;
            }

            if (parseStartLine(line)) {
                logger.ack();
                return;
            }
        }

        delay(10);
    }
}

static void waitForHandshake() {
    bool have_engage = false;
    bool have_release = false;
    bool have_pulse = false;
    bool have_threshold = false;
    bool have_side = false;
    bool have_reverse = false;
    bool have_trial = false;
    bool have_phase = false;
    bool have_ephys = false;

    session_cfg.phase = 0;

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
                if (strncmp(line, "phase", 5) == 0) have_phase = true;
                if (strncmp(line, "ephys", 5) == 0) have_ephys = true;

                logger.ack();
            }
        }

    check_done:
        bool trial_required = (session_cfg.phase != 0 && session_cfg.phase != 1);

        if (have_engage && have_release && have_pulse &&
            have_threshold && have_side && have_reverse && have_phase &&
            have_ephys && (!trial_required || have_trial)) {

            applyPhaseDefaults(session_cfg.phase);

            if (!have_trial) {
                trial_cfg.trial_n = 1;
                trial_cfg.easy = false;
                trial_cfg.side = session_cfg.side;
                trial_cfg.pending = false;
            }

            return;
        }
    }
}

static void drainSerial() {
    char line[96];

    while (readLine(line, sizeof(line))) {
        if (strcmp(line, "R1") == 0) {
            digitalWrite(RECORD_TTL, HIGH);
            logger.ack();
            continue;
        }

        if (strcmp(line, "R2") == 0) {
            digitalWrite(RECORD_TTL, LOW);
            logger.ack();
            continue;
        }

        if (strcmp(line, "img_start") == 0) {
            sendPulseTrain("img_start");
            logger.ack();
            continue;
        }

        if (strcmp(line, "img_stop") == 0) {
            sendPulseTrain("img_stop");
            logger.ack();
            continue;
        }

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

        {
            char tmp[96];
            strncpy(tmp, line, sizeof(tmp));
            tmp[sizeof(tmp)-1] = 0;

            if (parseTrialConfig(tmp)) {
                logger.ack();
                continue;
            }
        }

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

void writeSerial(const char* event) {
    if (event == nullptr) return;

    bool sent = false;

    // trial_start, trial_stop
    if (strcmp(event, "trial_start") == 0) {
        logger.write("trial_start");
        sent = true;
    }
    else if (strcmp(event, "trial_stop") == 0) {
        logger.write("trial_stop");
        sent = true;
    }

    // cue, r_cue
    else if (strcmp(event, "cue") == 0) {
        logger.write("cue");
        sent = true;
    }
    else if (strcmp(event, "r_cue") == 0) {
        logger.write("r_cue");
        sent = true;
    }

    // hit, miss
    else if (strcmp(event, "hit") == 0) {
        logger.write("hit");
        sent = true;
    }
    else if (strcmp(event, "miss") == 0) {
        logger.write("miss");
        sent = true;
    }

    // lick
    else if (strcmp(event, "lick") == 0) {
        logger.write("lick");
        sent = true;
    }

    // process pulse train
    if (sent) {
        sendPulseTrain(event);
    }
}

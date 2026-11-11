#pragma once

#include "Arduino.h"

// IMPORTS
#include <math.h>
#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

// CONVERSION HANDLES
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

// CONSTANTS
#define BAUDRATE 1000000
#define RAW_FLAG false
#define SEED_PIN A0
#define POWER_EN 7

#if defined(ARDUINO_AVR_MEGA2560)
    #define RECORD_TTL 22
    #define EVENT_TTL 23
#elif defined(ARDUINO_AVR_UNO)
    #define RECORD_TTL 10
    #define EVENT_TTL 11
#else
    #error "Unsupported board: select Arduino Mega 2560 or Arduino Uno"
#endif

const unsigned long EVENT_PULSE_US = 11111;
const unsigned long EVENT_IPI_US = 11111;
const unsigned long EVENT_TRAIN_GAP_US = 40000;

static constexpr uint32_t RAW_HZ = 100;
static constexpr uint32_t RAW_US = 1000000UL / RAW_HZ;
static constexpr unsigned long sample_T = MINUTES(5);

// GLOBALS
static unsigned long session_T = MINUTES(45);
static unsigned long trial_T = SECONDS(30);
static unsigned long delay_T = SECONDS(3);
static unsigned long tone_T = SECONDS(1);

static float easy_threshold = DEGREES(15.0f);

struct SessionConfig {
    int phase = 0;
    float engage_ms = 0.0f;
    float release_ms = 0.0f;
    float pulse_ms = 0.0f;
    float threshold = 30.0f;
    char side = 'B';
    bool reverse = false;
    bool flush = false;
    bool ephys = false;
};
struct TrialConfig {
    long trial_n = 0;
    bool easy = false;
    char side = 'B';
    bool pending = false;
};

static SessionConfig session_cfg;
static TrialConfig trial_cfg;

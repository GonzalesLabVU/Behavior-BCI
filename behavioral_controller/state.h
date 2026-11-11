#pragma once

// IMPORTS
#include "config.h"

#include "Wheel.h"
#include "Brake.h"
#include "Lick.h"
#include "Speaker.h"
#include "Spout.h"
#include "Logger.h"
#include "Timer.h"

// STATES
enum class SessionState { MAIN, CLEANUP };
enum class PhaseState { IDLE, CUE, TRIAL, HIT, MISS, DELAY };

SessionState session_state = SessionState::MAIN;
PhaseState phase_state = PhaseState::IDLE;

// CUSTOM OBJECTS
Brake brake;
Lick lick;
Wheel wheel;
Spout spout;
Speaker speaker;

Timer session_timer;
Timer phase_timer;
Timer raw_timer;

Logger logger;

// GLOBALS
static bool session_initialized = false;
static bool trial_hit = false;
static bool reward_given = false;
static long last_disp_mark = LONG_MIN;

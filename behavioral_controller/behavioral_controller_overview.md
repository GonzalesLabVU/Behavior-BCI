# Behavioral Controller — Project Overview

This document describes an Arduino sketch (`behavioral_controller.ino`) that runs a rodent
behavioral-training rig. It controls a wheel-based motor task, a licking/reward spout, a
brake, an auditory speaker, and TTL sync lines for external recording equipment (e.g.
electrophysiology or imaging systems). The sketch communicates with a host computer over
USB serial using a simple line-based text protocol.

Upload this `.md` file together with the `.ino` file to give a new chat session full context
on how the system works, even without the individual utility-class `.h`/`.cpp` files.

---

## 1. Hardware / Target Boards

The firmware supports two boards, selected via Arduino board-specific preprocessor macros:

| Board | Brake servo pin | Speaker pin | RECORD_TTL | EVENT_TTL |
|---|---|---|---|---|
| Arduino Mega 2560 | 44 | 33 | 22 | 23 |
| Arduino Uno | 9 | 10 | 10 | 11 |

Other fixed pins (not board-dependent):
- `POWER_EN` (pin 7): master power-enable line for downstream hardware (brake, spout, etc.)
- `SEED_PIN` (A0): floating analog pin read once at startup to seed the RNG
- Spout pulse pin: 5 (`Spout` class)
- Wheel quadrature encoder: `A_PIN` = 3 (interrupt), `B_PIN` = 2

`#error` directives cause compilation to fail if neither Mega 2560 nor Uno is selected as the
board in the Arduino IDE.

---

## 2. Utility Classes (used by the .ino but not uploaded alongside it)

The sketch is built from several small, single-purpose hardware-wrapper classes. Each is
instantiated once as a global object in the `.ino` (`brake`, `lick`, `wheel`, `spout`,
`speaker`, plus three `Timer` instances and a `Logger`). Their internal implementations live
in separate `.h`/`.cpp` files, summarized below so their behavior can be understood without
those files present.

### 2.1 `Brake`
Controls a servo-actuated wheel brake.
- `init(engage_us, release_us)` — stores the two PWM pulse-width (microseconds) targets that
  correspond to the "engaged" and "released" servo positions.
- `engage()` — attaches the servo, moves it to the engage position, blocks for a fixed hold
  time (500 ms) to let it physically reach that position, then detaches the servo (to avoid
  jitter/heat when idle). No-op if already engaged.
- `release()` — same as above but moves to the release position. No-op if already released.
- Internally tracks `engaged_` state so repeated calls are safe/idempotent.

### 2.2 `Lick`
Wraps an MPR121-style capacitive touch sensor (I2C, address `0x5A`) used to detect licks on
the reward spout electrode.
- `init(read_raw)` — starts I2C, resets/configures the sensor, disables all electrodes except
  a single "spout" electrode (electrode 8), sets conservative touch/release thresholds, and
  runs a baseline warm-up delay. If `read_raw` is true, one additional electrode (electrode 0)
  is left enabled for raw-signal streaming/debugging.
- `calibrate()` — an optional (currently unused/commented-out in `setup()`) routine that
  samples the filtered electrode signal for ~2.7 s, computes a robust noise estimate (median +
  MAD-based sigma) and a 99th-percentile-based threshold, and writes adaptive touch/release
  thresholds back to the sensor. Falls back to fixed defaults if too few samples are captured.
- `sampleFiltered()` — polls the touch-status register and updates an edge-detected
  "just touched" flag (only true on the same loop iteration the touch begins).
- `justTouched()` — returns/clears that edge-triggered flag; the main sketch calls this every
  loop iteration during almost every phase to detect and log licks.
- `sampleRaw()` — returns the raw filtered electrode-0 reading, only if `read_raw` was enabled
  at init (used for the currently-disabled raw data-streaming feature, gated by `RAW_FLAG`).

### 2.3 `Wheel`
Reads a quadrature rotary encoder (running wheel) via a hardware interrupt on pin 3 and
computes cumulative angular displacement, used as the main behavioral response in the task.
- `init(easy_threshold, normal_threshold, side, reverse)` — sets up pins/interrupt, converts
  the two threshold angles (degrees) into encoder counts, and stores whether counts should be
  interpreted as reversed and which rotation direction ("L"/"R"/"B") counts as "correct."
- Interrupt service routine (`isr_`) increments/decrements a shared `current_pos_` counter on
  each rising edge of channel A, using channel B's level to determine direction.
- `update()` — recomputes `displacement` (in degrees, as a public float) from the difference
  between current and initial encoder position, and checks whether the (direction-corrected)
  displacement has reached the currently active threshold (`thresholdReached()`) or moved the
  wrong way past the threshold (`thresholdMissed()`, only meaningful when a side is enforced).
- `reset(easy, side)` — re-zeroes the reference position, selects which threshold (easy vs.
  normal) and direction requirement apply to the upcoming trial, and clears the
  reached/missed flags. A no-arg `reset()` overload resets with normal difficulty and no side
  requirement.
- `thresholdReached()` / `thresholdMissed()` — each is a one-shot "consume" read: returns true
  once, then clears itself, so the main loop can detect the event exactly once per trial.

### 2.4 `Speaker`
Drives a piezo speaker on Timer2 (hardware CTC interrupt) to generate square-wave tones
without blocking the main loop, so behavior/serial handling can continue during a tone.
- `init(side)` — configures the pin and remembers which side ("L"/"R") the speaker belongs to
  (used to select the cue frequency), seeds an internal xorshift RNG from `micros()`.
- `cue()` — plays a fixed-frequency "cue" tone (2500 Hz) for 1 second; ignored if a tone is
  already playing.
- `hit()` — plays a fixed "hit" tone (4000 Hz) for 1 second; will preempt an in-progress `cue`
  but not an in-progress `hit`/`miss`.
- `miss()` — plays a "miss" sound: a randomly frequency-hopping tone (1000–4000 Hz, hopping
  every 100–200 µs) for 1 second — an audibly harsh/noisy buzz, distinct from the pure-tone
  cue/hit sounds.
- `stop()` — immediately silences the speaker and returns it to idle (used to cut short an
  inactivity re-cue tone).
- Internally, `startTimer2_()` configures Timer2 in CTC mode to toggle the speaker pin at the
  frequency-derived half-period; the ISR (`onTick_`, wired through a global trampoline function
  and a static `instance_` pointer) toggles the pin, advances elapsed/step counters, randomly
  re-picks frequency during `miss` mode, and stops itself once the configured duration elapses.

### 2.5 `Spout`
Controls a solenoid/valve on pin 5 that delivers liquid reward.
- `init(pulse_dur_us)` — sets the default reward pulse duration (microseconds) used by the
  no-argument `pulse()` overload.
- `pulse()` / `pulse(us)` — opens the valve for the configured (or given) duration, blocking,
  then closes it. Used to deliver a single reward.
- `flush()` / `flush(ms)` — opens the valve for a long duration (10 s default, or a given
  number of milliseconds) to prime/flush the lines. Used during setup and for the manual
  "flush" startup routine.

### 2.6 `Timer`
A simple non-blocking millis()-based countdown/stopwatch helper used throughout the state
machine to time phases and sessions without blocking `loop()`.
- `init(duration_ms)` — sets the duration for the next run.
- `start()` — records the current `millis()` as the start time and marks the timer "started."
- `isRunning()` — true while elapsed time is less than the configured duration (false once
  expired). This is the "still active" check most state-machine code branches on.
- `started()` — true once `start()` has been called, independent of whether time has expired
  (used to distinguish "not yet begun" vs. "began and finished").
- `reset()` — clears the started flag, effectively returning the timer to its unstarted state
  (does not clear the configured duration).
- `timeElapsed()` — milliseconds since `start()`, or 0 if never started.

### 2.7 `Logger`
A thin wrapper around `Serial` that standardizes the outgoing message format and reads
incoming command lines.
- `write(const String&)` — prints session start/stop markers `"S"`/`"R"` bare (no prefix);
  everything else is printed as `"[EVT] <text>"` (used for behavioral events like cue/hit/
  miss/lick, and for wheel-displacement milestone values passed as text).
- `write(int)` / `write(float)` — prints as `"[ENC] <value>"` (encoder/displacement data).
- `writeRaw(uint16_t)` — prints as `"[RAW] <value>"` (raw touch-sensor stream, currently
  unused since `RAW_FLAG` is `false`).
- `read()` — reads and trims one newline-terminated line from `Serial`, or returns `""` if
  nothing is available (note: the main `.ino` actually uses its own `readLine()` helper built
  directly on `Serial`, rather than this method, for the handshake/config parsing).
- `ack()` — sends a bare `"A"` acknowledgment line, used to confirm receipt of each
  handshake/config command from the host.

---

## 3. Serial Communication Protocol

Baud rate: **1,000,000**. All host↔device communication is line-based ASCII terminated by
`\n`.

### 3.1 Output (device → host)
- `"S"` — session start marker at power-up... (actually printed at the *end*, see Cleanup)
- `"R"` — printed after a manual flush routine completes
- `"[EVT] <event>"` — behavioral events: `cue`, `r_cue` (re-cue during inactivity), `hit`,
  `miss`, `lick`
- `"[ENC] <value>"` — wheel displacement milestones (nearest 0.5° marks reached during a
  trial)
- `"[RAW] <value>"` — raw capacitive sensor stream (only if `RAW_FLAG` were `true`; currently
  disabled)
- `"A"` — acknowledgment of a received handshake/config line

### 3.2 Input (host → device)

**Startup flush gate** (`parseFlushCommand`, blocks until received):
- `flush 0` / `flush 1` — whether to run the priming/flush-only routine and halt (if `1`) or
  proceed to normal handshake (if `0`)

**Handshake** (`waitForHandshake`, blocks until all required keys received):
- `engage <ms>`, `release <ms>`, `pulse <ms>` — brake engage/release servo timing and reward
  pulse duration
- `threshold <deg>` — wheel-rotation threshold (degrees) for the current session
- `side <L|R|B>` — required rotation direction (or `B` = either)
- `reverse <0|1>` — whether encoder direction is inverted
- `phase <0-99>` — training phase/stage number (see Section 4)
- `ephys <0|1>` — whether to emit TTL sync pulses for each event
- `<trial_n> <easy 0|1> [side]` — optional numeric trial-config line (only required when
  `phase` is 2 or higher); pre-loads the first trial's difficulty/side
- `E` — abort/end session immediately, skip straight to cleanup

**Post-setup start gate** (`parseStartCommand`, blocks until a `start 0|1` line):
- `R1` / `R2` — set/clear the `RECORD_TTL` line (external recorder start/stop) at any time
  while waiting
- `img_start` / `img_stop` — emit a distinct TTL pulse-train code for imaging start/stop
- `start 0` or `start 1` — releases the gate and begins the main loop

**During the main loop** (`drainSerial`, non-blocking, processed every loop iteration):
- `R1` / `R2`, `img_start` / `img_stop`, `E` — same as above, usable at any time
- `H` / `M` — manual override: force the current trial to end as a Hit or Miss (only takes
  effect while in the `TRIAL` phase state)
- Any handshake-style key/value or trial-config line — accepted and applied live (e.g. to
  change `threshold`, `side`, `reverse`, `ephys`, or queue a new trial's easy/side setting)

---

## 4. Session / Phase State Machine

Two nested state machines drive the sketch:

- **`SessionState`**: `MAIN` (normal operation) → `CLEANUP` (graceful shutdown, then halts
  forever).
- **`PhaseState`**: `IDLE` → `CUE` → `TRIAL` → `HIT` or `MISS` → `DELAY` → back to `CUE`
  (looping until the session timer expires, then falling through to `CLEANUP`).

`loop()` first calls `drainSerial()` to process any pending commands, then dispatches to one
of five phase-specific functions based on `session_cfg.phase`:

| phase | function | Session length (default) | Trial length | Delay |
|---|---|---|---|---|
| 0 | `run_phase_0()` | 20 min | 30 s | 3 s |
| 1 | `run_phase_1()` | 10 min | 5 s | 3 s |
| 2 | `run_phase_2()` | 20 min | 30 s | 1 s |
| 3 | `run_phase_3()` | 20 min | 30 s | 1 s |
| ≥4 | `run_phase_4_plus()` | 30 min | 30 s | 3 s |

(Defaults are set by `applyPhaseDefaults()` right after the handshake completes, and can be
overridden by sending explicit `phase` and other config values.)

Each phase function is a hand-rolled state machine using the `Timer` "started but not
running" pattern to detect phase entry (do setup once) vs. exit (timer expired), rather than
blocking delays — this keeps `loop()` responsive to serial input and lick sampling throughout.

### Phase-by-phase behavior

**Phase 0 — passive habituation / pairing.**
Brake stays engaged throughout (never released). Cycles CUE (tone + log) → TRIAL (just waits
out `trial_T` while sampling licks — wheel is irrelevant since it's locked) → HIT (always a
"hit," logged, no reward or brake action) → DELAY → repeat. Essentially a fixed-interval
cue/tone exposure phase with the wheel braked.

**Phase 1 — spout/reward association, no wheel or cue tone.**
Skips `CUE` entirely; goes straight to `HIT` after `IDLE`. In `HIT`, logs "hit", starts a
1-second timer, delivers a spout reward pulse halfway through, then moves to `DELAY`, which
immediately loops back into `TRIAL`. `TRIAL` here is actually a lick-triggered re-entry into
`HIT` — it waits either for a lick (immediate transition to another rewarded `HIT`) or for
`trial_T` to elapse, then always advances to `HIT` again. When the session timer expires it
fires one final `spout.pulse()` and moves to cleanup. Net effect: simple, no-cue,
lick-triggered (or timed) repeated reward delivery.

**Phase 2 — wheel task, no cue tone, no brake release logged as an action, no speaker.**
`IDLE` releases the brake once. `CUE` is a pass-through (no tone) straight into `TRIAL`.
`TRIAL` resets the wheel for the queued trial (easy/side), then waits for the animal to
rotate the wheel far enough (`wheel.thresholdReached()` → `HIT`) or the wrong way past
threshold (`thresholdMissed()` → `MISS`), logging displacement milestones and licks along the
way; timing out with neither counts as a `MISS`. `HIT` delivers a spout reward (no tone).
`MISS` has no consequence beyond logging. Both proceed to `DELAY`, which — if the session
timer is still running — loops back to `CUE` (i.e., straight back into `TRIAL` since `CUE` is
a pass-through here); otherwise moves to cleanup.

**Phase 3 — full wheel task with cue tone and hit tone, still no brake engage/release per
trial (brake released once at start and left released).**
`IDLE` releases the brake once. `CUE` plays the cue tone and waits it out (licks logged).
`TRIAL` behaves as in Phase 2 (wheel-threshold based hit/miss, with displacement-milestone
logging), but on time-out with no threshold crossed it's still routed to `MISS`. `HIT` plays
the hit tone and delivers reward simultaneously, then a `DELAY`. `MISS` plays the miss tone
(no reward). `DELAY` waits out `delay_T` (sampling licks) then loops back to `CUE`, or exits
to cleanup once the session timer ends. This is the closest lower-effort predecessor of the
"full" task below, distinguished from phase ≥4 mainly by not re-engaging the brake on Hit/
Miss and not including the inactivity re-cue logic.

**Phase ≥4 ("full" task) — the complete trained-animal paradigm.**
`IDLE` starts the session timer and moves to `CUE`. `CUE` plays the cue tone, and on
expiry releases the brake and enters `TRIAL`. `TRIAL` resets the wheel for the (possibly
just-updated) trial config, then on each loop: samples licks, updates the wheel, logs
displacement milestones, and calls `checkInactivity()` (see below) to nag the animal with a
re-cue tone if it stalls; a threshold-reached rotation ends the trial as `HIT`, a wrong-way
threshold-miss (only relevant if a side was required) ends it as `MISS`, and simply timing out
without either also counts as `MISS`. `HIT` re-engages the brake, plays the hit tone, and
delivers a reward pulse halfway through the tone. `MISS` re-engages the brake and plays the
miss (frequency-hopping buzz) tone, no reward. `DELAY` waits out `delay_T` (still sampling
licks) then loops back to `CUE`, or — once the session timer has expired — transitions to
`CLEANUP`.

### Inactivity re-cue (`checkInactivity`, phase ≥4 only)
While in `TRIAL` and past the first 5 seconds of the trial, if the wheel hasn't moved at
least 5° from its last-checked position within a 5-second window, the speaker plays another
cue tone (logged as `r_cue`) as a "keep going" prompt — but only if there's more than 1 second
left in the trial. The re-cue tone is cut short (`speaker.stop()`) once its own inactivity
timer lapses, and the whole check resets whenever displacement moves enough or a new trial
begins.

---

## 5. TTL Sync Pulses (`sendPulseTrain` / ephys mode)

When `session_cfg.ephys` is `1`, every logged behavioral event also emits a coded burst of TTL
pulses on `EVENT_TTL` for an external recording system to timestamp, via
`sendPulseTrain(eventType)`:

| Event | Pulse count |
|---|---|
| `lick` | 1 |
| `cue` / `r_cue` | 2 |
| `hit` / `miss` | 3 |
| `img_start` | 4 |
| `img_stop` | 5 |

Each pulse is ~11.1 ms high (`EVENT_PULSE_US` = 11111 µs) with ~11.1 ms between pulses in a
train (`EVENT_IPI_US`), and a ~40 ms gap (`EVENT_TRAIN_GAP_US`) is enforced after the whole
train finishes. `img_start`/`img_stop` pulse trains are sent unconditionally as soon as those
commands are received (independent of the `ephys` flag), since they're also used to gate the
separate `RECORD_TTL` line via `R1`/`R2`.

---

## 6. Startup Sequence (`setup()`)

1. Clear watchdog status/disable watchdog (in case of a prior watchdog-triggered reset).
2. Set `POWER_EN`, `RECORD_TTL`, `EVENT_TTL` low; begin `Serial` at 1 Mbaud; seed RNG from a
   floating analog pin.
3. Block on `parseFlushCommand()` waiting for a `flush 0|1` line.
   - If `flush 1`: power on, run a 10-second spout flush, print `"R"`, power off, then arm a
     15 ms watchdog and spin forever to force a hardware reset — a dedicated "prime the
     lines and reboot" utility mode, not part of a real session.
4. Otherwise, block on `waitForHandshake()` until all session-config fields (and, if needed,
   the first trial config) have been received.
5. Convert ms-based config values to microseconds, power on peripherals, initialize `brake`
   (engaged), `speaker` (side-tagged), `spout` (with a 5-pulse/150 ms-interval priming
   sequence), `lick` sensor, and `wheel` (with easy/normal thresholds, side, reverse flag).
6. Block on `parseStartCommand()` until a `start 0|1` line is received (while still servicing
   `R1`/`R2`/`img_start`/`img_stop` commands).
7. Start the (currently inert, since `RAW_FLAG` is `false`) raw-sampling timer, and enter the
   main `MAIN`/`IDLE` state to begin `loop()`.

## 7. Shutdown Sequence (`CLEANUP`)
Sets `RECORD_TTL`/`EVENT_TTL` low, logs `"S"` (session-end marker), engages the brake, resets
both timers, waits 500 ms, powers everything down via `POWER_EN`, then spins forever
(session is over; a physical reset/re-upload is required to run another session).

---

## 8. Notable Implementation Details Worth Knowing
- All phase transitions are driven by polling `Timer::started()`/`isRunning()` each loop
  iteration rather than blocking `delay()`, so serial commands and lick sampling remain
  responsive during tones/trials/delays (the `Brake` and `Spout` classes are the main
  exceptions — their `engage()`/`release()`/`pulse()`/`flush()` calls do block briefly).
- `Speaker` tone generation runs entirely off a Timer2 CTC hardware interrupt, so tones do not
  block `loop()` at all, even while the brake or spout is separately blocking.
- The wheel's ISR only fires on channel-A rising edges and reads channel B's level directly
  from the AVR port register (not `digitalRead`) for speed.
- Raw touch-sensor streaming (`RAW_FLAG`) and `Lick::calibrate()` are present in the code but
  currently disabled/unused (`RAW_FLAG` is `false`, `calibrate()` call is commented out) —
  worth knowing if asked to debug or extend those code paths.
- The trial-config side (`trial_cfg.side`) can be updated live mid-session via the serial
  protocol even while a trial is in progress; it only takes effect the next time `TRIAL` is
  entered (`wheel.reset()` is called at trial entry, not mid-trial).

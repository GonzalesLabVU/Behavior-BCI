# Behavioral Master — Python-Side System Overview

This document describes the Python-side software that drives the rodent behavioral-training
rig (the counterpart to the Arduino `behavioral_controller.ino` firmware). The centerpiece is
`behavioral_master.py`, a single script that a trainer runs on a lab PC once per session. It
talks to the Arduino over serial, optionally drives a visual "cursor" feedback task (via
`cursor_utils.py`), optionally synchronizes with a two-photon imaging rig, optionally emits
electrophysiology sync pulses, logs a status dashboard (via `dashboard_utils.py`), and saves
all collected session data to Google Sheets (with local JSON fallbacks).

Upload this `.md` file together with `behavioral_master.py` to give a new chat session full
context on how the system works, even without `cursor_utils.py`, `dashboard_utils.py`, or
`TCPClient.py` present.

---

## 1. File Map

| File | Role |
|---|---|
| `behavioral_master.py` | Main entry point. Orchestrates the whole session: prompts the trainer, talks to the Arduino, runs the event loop, and saves data. |
| `cursor_utils.py` | Runs a full-screen (or windowed) `pygame` visual-feedback widget ("cursor") that mirrors wheel position on screen during wheel-based phases, in its own thread. |
| `dashboard_utils.py` | Thin helper that writes live session status (running/idle, animal, phase) to a shared "Dashboard" Google Sheet so multiple rigs' status can be monitored centrally. |
| `TCPClient.py` | Raw-socket TCP client (`PrairieClient`) that drives a Bruker/Prairie View two-photon imaging rig over the local network, used to start/stop imaging in sync with trial outcomes. |

### External files/services this system expects at runtime
- `.env` (same directory as the script) — environment variables such as `BRAKE_ENGAGE_MS`,
  `BRAKE_RELEASE_MS`, `SPOUT_PULSE_MS`, `CLIENT_ID`, `DASHBOARD_ID`, `SMTP_USERNAME`,
  `SMTP_PASSWORD`, `SMTP_TO_ADDR`, `GITHUB_TOKEN`, per-cohort `<COHORT>_ID` Google Sheets
  workbook IDs, and optionally `GOOGLE_SERVICE_ACCOUNT_JSON`.
- `credentials.json` — Google service-account credentials for the Sheets/Drive APIs.
- `animal_map.json` — a dict mapping underscore-joined cohort keys (e.g. `"M1_M2_M3"`) to a
  cohort name, used to look up which workbook an animal's data belongs to.
- `errors.log` — local append-only error log, optionally pushed to a GitHub repo
  (`GonzalesLabVU/Behavior-BCI`, `pc/config/errors.log`) if `GITHUB_TOKEN` is set.

---

## 2. `behavioral_master.py` — Architecture

The script is organized as a set of small **`InterfaceObject`** subclasses, each wrapping one
hardware/API/human boundary, plus a handful of free functions and two data-holding classes
(`SessionData`, `FileLock`). This mirrors an inversion-of-control style: `setup()` and
`main()` take an `interfaces` bundle (a `BehaviorInterfaces` instance) so that any interface
can be swapped out (e.g. for testing) without changing the control flow.

### 2.1 Interface classes

- **`SystemInterface`** — environment/filesystem/hardware glue: loads `.env`, reads
  `animal_map.json`, resolves an animal's Google Sheets workbook ID, finds and opens the
  Arduino's serial port (`_find_arduino_port` looks for "arduino" or "usb serial" in the port
  description), and builds phase-specific config dicts (brake/spout timing, wheel threshold,
  side, reverse flag) by merging `.env` values with the hard-coded `PHASE_CONFIG` table.
- **`ConsoleInterface`** — all trainer-facing console output: session start banner, a
  live-updating trial-status table (trial count, elapsed time, hit/miss block indicators, hit
  rate), a duration summary, and formatted display of any exceptions collected in the global
  `EXC_STACK` at the end of the run.
- **`TrainerInterface`** — all interactive `input()` prompts: whether to flush the reward
  line, animal ID (blank → `"DEV"` debug animal), training phase, target side (phase 4 only),
  whether imaging/ephys are active, and confirmation prompts around saving/overwriting.
- **`DashboardInterface`** — pushes `running`/`finished`/`idle` status (plus animal/phase) to
  the shared Dashboard sheet via `dashboard_utils.write_fields`, failing safely (a warning,
  not a crash) if the write fails. `notify_finish()` also monkey-patches `keyboard.read_key`
  so that the dashboard is marked `idle` at the exact moment the trainer dismisses the final
  "press any key" prompt.
- **`ExceptionInterface`** — collects exceptions into the module-level `EXC_STACK` (for
  console display), appends formatted tracebacks to the local `errors.log`, and can commit/push
  that log to a GitHub repo using a throwaway shallow clone (via `git` subprocess calls) when
  `GITHUB_TOKEN` is configured.
- **`EmailInterface`** — sends a plain-text/HTML session-summary email (date, start/stop
  time, duration, trial count, hit rate) via Gmail SMTP once a real (non-`DEV`) session
  finishes.
- **`SaveInterface`** — the most complex interface; writes session data into a per-animal
  Google Sheets workbook (see §5), plus a JSON local backup of raw capacitive-sensor data, and
  a full local JSON fallback if the Sheets write fails.
- **`PrairieInterface`** — thin wrapper for connecting to / finishing an imaging session via
  the (not-uploaded) `PrairieClient`.
- **`CursorInterface`** — starts/updates/stops the `cursor_utils.BCI` visual-feedback task for
  wheel-based phases (phase ≥ 4 only; phases ≤ 3 get `cursor=None`).
- **`BehaviorInterfaces`** — a simple container that instantiates one of each interface above;
  this is the single object passed into `setup()` and `main()`.

### 2.2 `ArduinoLink` — the serial protocol client
This is the Python-side mirror of the Arduino's serial protocol (see the Arduino-side `.md`
for the firmware side of this exchange).
- Wraps a `pyserial` `Serial` object; if no Arduino was found, `active=False` and every
  send/wait call becomes a harmless no-op (lets the whole system run in a keyboard-only "DEV"
  mode without hardware).
- Runs a background reader thread (`_reader_loop`) that classifies each incoming line and
  pushes typed messages onto `msg_q`: `"RESTART"` (bare `R`), `"END"` (bare `S`), `"EVT"`
  (`[EVT] ...`), `"ENC"` (`[ENC] ...`), `"RAW"` (`[RAW] ...`), or an `"ERR"` message if the
  reader itself throws. A bare `"A"` line is treated specially — it sets `ack_evt` rather than
  being queued, since it's the handshake acknowledgment.
- `send_and_wait(text)` writes a line and blocks (with a 5 s default timeout) until the next
  `"A"` ack arrives — used for every handshake/config command, mirroring the Arduino's
  blocking `waitForHandshake`/`parseStartCommand`/`drainSerial` ack behavior.
- `send(text)` fires and forgets (used for the `"E"` abort command during cleanup).
- `send_after(cmd, delay_s)` spins up a daemon thread to send (and wait-ack) a command after a
  delay — used to schedule imaging start/stop TTL pulses relative to trial outcomes without
  blocking the main loop.
- Convenience wrappers: `send_config()` (sends the full `engage`/`release`/`pulse`/
  `threshold`/`side`/`reverse`/`phase` handshake in order), `send_ephys()`, `send_flush()`
  (also handles the Arduino's forced-reboot-after-flush behavior by raising `SystemExit(0)`
  when it sees the `RESTART` line), `send_start()`, `start_ephys()`/`stop_ephys()` (send `R1`/
  `R2`), `start_imaging()`/`stop_imaging()` (send `img_start`/`img_stop`, optionally delayed).

### 2.3 `SessionData` — in-memory session record
A plain data container accumulated throughout the session and eventually serialized:
- `meta` — animal/phase/date, timing, imaging/ephys flags, adaptive-difficulty parameters
  (`K1`, `K2`), and per-trial config log (`trial_config`: trial number, easy/normal, side).
- `evt` — parallel timestamp/value lists of all logged behavioral events (`cue`, `r_cue`,
  `hit`, `miss`, `lick`, plus synthetic `setK <value>` calibration events).
- `enc` — parallel timestamp/value lists of wheel-displacement milestone samples.
- `img` — lists of imaging start/stop timestamps (filled in from the Prairie client at the
  end of the session).
- `raw` — two sub-logs: `evt` (a raw duplicate log of `hit`/`lick` events) and `cap` (raw
  capacitive touch-sensor stream, only populated if the Arduino were built with `RAW_FLAG`
  enabled).
- Helper methods: `log_trial_config`, `add_evt`/`add_enc`/`add_raw_cap`/`add_raw_evt`,
  `any_data()` (used to decide whether there's anything worth saving), `to_dict()` (JSON-safe
  serialization for the local fallback save path), and `is_finished` (both start/stop
  timestamps present).

### 2.4 `FileLock` — distributed Google Sheets lock
Because multiple rigs may write to the same per-cohort workbook, `SaveInterface.save_data()`
takes out a **cooperative distributed lock implemented entirely inside the Google Sheet
itself** (there is no external lock server):
- To acquire, a client creates its own uniquely-named worksheet tab (named after its `owner`
  ID) stamped with a lock "tag" cell and an owner/token/created/expires metadata row, then
  re-scans all sheets for other lock tabs. Among all currently-non-expired lock tabs, the one
  with the earliest `created` timestamp (tie-broken by token, then sheet ID) "wins"; if that's
  the caller's own tab, the lock is acquired. Otherwise the caller deletes its own attempt (if
  any) and polls again after a jittered sleep.
- Expired lock tabs (past their `expires` timestamp) are proactively deleted by whichever
  client notices them first.
- `update()`/`reset()` are called periodically during a long save to refresh/extend the lease
  (`reset_s`/`lease_s` thresholds) and to detect if the lock has been lost (e.g. deleted by
  another client after expiry) — in which case a `RuntimeError` aborts the save.
- `release()` deletes the lock worksheet, retrying a few times with jittered backoff, and
  treats "lock already gone / not owned" as a successful release (idempotent).
- `_get_client_id()` generates a `hostname:pid:random` owner string so concurrent runs (even
  on the same machine) don't collide.

### 2.5 Adaptive trial difficulty
- **`PHASE_CONFIG`** hard-codes wheel-rotation thresholds, target side, and reverse-encoder
  flag per phase (phases 2–7: wheel association → tone association → easy/normal/harder/
  hardest wheel tasks). Phases `"0"` and `"1"` have no wheel requirement and aren't in this
  table (handled as special-cased "no cfg" phases throughout).
- **`_get_easy(phase, trial_n, K)`** decides whether a given trial should use the easy
  (smaller) wheel threshold instead of the phase's normal threshold: always easy below phase
  5; for the first 20 trials, every 5th trial is easy; after that, every `K`-th trial is easy,
  where `K` is adaptively set.
- **`_update_easy_rate(session_data, trial_stack)`** (only used above phase 4, gated by
  `do_calibration`) looks at the animal's hit count over the last `N` trials (`N = 4*K`) and
  sets a new `K` (fewer hits → more frequent easy trials: `K=3`; exactly 10 hits → `K=5`; more
  → `K=7`), logging a synthetic `"setK <K>"` event and locking in `calibrated = True` for the
  rest of the session.
- **`_is_early_exit(...)`** contains a (currently dead — the function returns `False`
  unconditionally on its very first line) more elaborate implementation that would have
  tracked recent trials-per-minute and ended the session early if the animal's trial rate
  dropped too low for too long. Worth knowing about if asked to debug/re-enable early-exit
  behavior, since the "real" logic already exists below the early `return False`.

---

## 3. Program Flow

### 3.1 `setup(interfaces)`
Runs once at program start, before the main event loop:
1. Finds/opens the Arduino serial port (continues in "no hardware" mode with a warning if
   none is found) and wraps it in an `ArduinoLink`.
2. Prompts whether to flush the reward line; if so, sends `flush 1` and lets the Arduino's
   own flush-and-reboot routine run (which raises `SystemExit(0)` back on this side once the
   Arduino signals `RESTART`).
3. Prompts for animal ID (blank defaults to `"DEV"`) and training phase; requires a detected
   Arduino if the animal isn't `"DEV"`/phase-agnostic. `"DEV"` also turns on verbose serial
   logging.
4. For phase `"4"` specifically, additionally prompts for a target side override.
5. Prompts whether imaging and/or ephys are active.
6. Builds the phase config (brake/spout timings from `.env`, wheel threshold/side/reverse from
   `PHASE_CONFIG`), sends the full config + ephys handshake to the Arduino, connects to
   Prairie View if imaging is active, and — for phases with a wheel component — starts the
   `CursorInterface` visual feedback task with the first trial's easy/side settings.
7. For phase > 1, sends the first trial's config line (`"1 <0|1>"`) to the Arduino.
8. Builds the `SessionData` object, records config into its metadata, starts ephys TTL if
   requested, sends the final `start 1` handshake command, and notifies the dashboard that the
   session is `running`.
9. On any exception during setup, caches it, best-effort closes the link/serial port, and
   re-raises (so `__main__` can log/report it without crashing ungracefully).

### 3.2 `main(link, session_data, cursor, client, interfaces)`
The live session event loop, driven entirely by messages arriving on `link.msg_q` from the
Arduino reader thread:
- On the very first message, records the session start time and prints the console header.
- **`EVT` messages** are the behavioral event stream and drive almost everything:
  - Every event is also pushed to the global `EVT_QUEUE` so the `cursor_utils` pygame thread
    can react to it (new trial cue, hit/miss) independently of this loop.
  - `cue`: logs the event; if imaging is active and this is the very first trial, sends the
    initial imaging-start command and starts the TTL sync line together, then begins tracking
    trial number/timing.
  - `r_cue`: logged as an event (no other side effects here — the re-cue tone is purely an
    Arduino/`cursor_utils` phenomenon).
  - `hit`/`miss`: de-duplicated (an identical repeated outcome for the same trial is ignored);
    on the real transition it logs the event, and if imaging is active it schedules (via
    `send_after`) a stop-then-restart of both the Prairie client and the imaging TTL line
    around the inter-trial interval (stop at +1 s, restart at +3 s) so each trial gets its own
    imaging bout. It then updates the rolling trial-outcome window, prints the console
    trial-status row, and — if calibration is active — updates the trial-difficulty stack and
    (once `N` trials have accumulated) computes a new `K`. For phase ≥ 4, once calibrated it
    checks the (currently disabled) early-exit condition, and otherwise computes the next
    trial's easy/side settings, sends the next trial-config line to the Arduino, and updates
    the on-screen cursor's target settings to match.
  - `hit`/`lick` events are additionally mirrored into `session_data.raw.evt` as a redundant
    raw log.
- **`ENC` messages** — parsed as a float wheel-displacement sample, appended to
  `session_data.enc`, and also pushed onto the global `ENC_QUEUE` for the cursor task to
  render live.
- **`RAW` messages** — appended to `session_data.raw.cap` (only produced if the firmware's raw
  capacitive stream is enabled).
- **`END`** (Arduino's `"S"` cleanup marker) — marks ephys as stopped in metadata and breaks
  out of the loop, ending the session normally.
- **`ERR`** — re-raises the underlying exception from the reader thread (or wraps a
  non-exception payload in a `RuntimeError`).
- On `KeyboardInterrupt`, marks the session aborted, safely stops ephys, and calls `_cleanup`
  (which sends the Arduino an `"E"` abort command and waits, with a timeout, for its `END`
  acknowledgment) before re-raising.
- The `finally` block always stamps `t_stop`/`duration_sec`, safely stops ephys, stops the
  Prairie client if present, and notifies the dashboard that the session has finished.

### 3.3 `if __name__ == "__main__":` driver
Wraps `setup()` + `main()` in a top-level try/except/finally that:
- Tracks `animal_id_for_log`/`phase_id_for_log` for error-log context even if setup fails
  early.
- Treats `SystemExit`/`KeyboardInterrupt` as clean exits; caches and displays anything else.
- Always shows a run summary, finishes the Prairie client and cursor task, and closes the
  Arduino link (each independently guarded so one failure doesn't block the other cleanup
  steps; each failure is logged/committed via `ExceptionInterface`).
- If the session actually finished (`is_finished`) and wasn't the `DEV` animal, emails a
  session summary.
- If there's any data at all and it isn't a `DEV` run, prompts to confirm saving, resolves the
  duplicate-session overwrite protocol (`SaveInterface.resolve_protocol`), and — if
  confirmed — saves the raw capacitive data locally and the full session to Google Sheets
  (with local-JSON fallback on failure).
- Finally re-displays any collected exceptions and waits for a keypress before exiting (so a
  double-clicked script window doesn't vanish immediately).

---

## 4. `cursor_utils.py` — On-Screen Visual Feedback Task

Provides a `pygame`-based full-screen (or windowed) "cursor" widget shown on a second monitor
during wheel-based training phases (phase ≥ 4), giving the animal (and trainer) real-time
visual feedback of wheel rotation alongside the physical brake/tone/reward hardware.

- **Module-level shared state**: `TRIAL_CONFIG` (a `(is_easy, alignment)` tuple, guarded by
  `TRIAL_LOCK`) lets the main script update the next trial's difficulty/side from a different
  thread; `ABORT_EVT` lets the cursor thread signal the main script that the trainer closed
  the window or hit Escape/Ctrl-C, which `main()` checks each loop iteration to raise a clean
  `KeyboardInterrupt`.
- **`cursor_fcn(threshold, evt_queue, enc_queue, ...)`** is the actual pygame render loop, run
  in its own thread:
  - Initializes pygame and picks a display/fullscreen mode on first call (or reuses an
    existing surface on subsequent calls within the same process).
  - Consumes `evt_queue` (the same events `behavioral_master.py` pushes to `EVT_QUEUE`): a
    `cue` event resets/arms the cursor for a new trial and adopts the latest `TRIAL_CONFIG`; a
    `hit`/`miss` event (while a trial is active and not already in a delay) freezes the cursor
    at its current position and starts a fixed post-outcome delay before blacking out the
    screen until the next `cue`.
  - While an active, non-delayed trial is running, consumes `enc_queue` (mirroring the wheel
    displacement pushed from `main()`'s `ENC` handling) to update the drawn cursor position,
    scaled so that the phase's easy/normal threshold always maps to a fixed target position on
    screen regardless of the actual threshold magnitude (`gain = target_deg / threshold`), and
    draws left/right/both target boxes depending on the trial's required side.
  - Watches for the window being closed, Escape, or Ctrl-C, in which case it sets `ABORT_EVT`
    and returns `'quit'`; a `stop_evt` (set from `BCI.stop()`) causes it to quit and return
    `'stopped'` instead.
- **`BCI` class** — the thread-lifecycle wrapper `behavioral_master.py` actually uses:
  - `__init__` reads the current phase's config (`bidirectional`, `threshold`) out of the same
    `PHASE_CONFIG` dict used on the main-script side; `enabled` is `False` for phases without
    a config entry (i.e., phases ≤ 3).
  - `start()` launches `cursor_fcn` in a daemon thread if enabled and not already running.
  - `update_config(is_easy, alignment)` updates the shared `TRIAL_CONFIG` under `TRIAL_LOCK`
    for the next `cue` event to pick up.
  - `stop(timeout)` signals `_stop_evt`, posts a synthetic pygame `QUIT` event as a wake-up
    nudge, and joins the render thread.

---

## 5. `dashboard_utils.py` — Shared Status Dashboard

A small, independent Google Sheets writer used purely for a live multi-rig status dashboard
(distinct from the per-animal data workbooks that `SaveInterface` writes to).

- Lazily loads `.env` (its own private cache, separate from `behavioral_master.py`'s
  `dotenv`-based loading) and lazily builds/caches a `gspread` client from either
  `credentials.json` or a `GOOGLE_SERVICE_ACCOUNT_JSON` environment variable.
- `CLIENT_START_COLS` maps three fixed client types (`BEHAVIOR`, `IMAGING`, `DEVELOPMENT` —
  with `BEH`/`IMG`/`DEV` shorthand accepted) to fixed starting columns on a single `"Dashboard"`
  worksheet; `FIELD_ROWS` maps `status`/`animal`/`phase` to fixed rows.
- `write_fields(client_id, fields, timestamp=None)` — normalizes the client ID, resolves (or
  creates, if missing) the `"Dashboard"` worksheet, and batch-updates a `[timestamp, value]`
  pair into the two-column block for each requested field/row. This is the function
  `DashboardInterface._safe_write()` calls on the `behavioral_master.py` side to report
  `running`/`finished`/`idle` status plus the current animal/phase.

---

## 6. `TCPClient.py` — Two-Photon Imaging (Prairie View) Client

Implements `PrairieClient`, a raw-TCP client that talks to a fixed local-network endpoint
(client bound to `192.168.2.1`, server at `192.168.2.2:5005`) presumed to be a bridge/listener
in front of Bruker's Prairie View imaging software. This is what `PrairieInterface` in
`behavioral_master.py` wraps.

- **Connection & wire protocol.** On construction, it opens a `TCP_NODELAY` blocking socket,
  connects immediately (so `PrairieClient()` itself can raise if the imaging rig is
  unreachable — which is why `PrairieInterface.connect()` wraps instantiation in a try/except),
  and wraps the socket in a buffered file object for line-based reads. Every command is a
  single newline-terminated ASCII line (`CONFIG`, `START`, `STOP`, `FINISH`); every reply is a
  single line starting with `OK` (optionally followed by JSON payload data for `FINISH`) or
  anything else, which is treated as failure. `_hexdump`/`_dump` provide optional verbose
  wire-level logging.
- **Single background network thread.** All actual socket I/O happens on one dedicated
  `_net_loop` thread reading off an internal command queue (`_q`); every public method
  (`configure`, `start`, `stop`, `finish`) funnels through `_enqueue()`, which posts a
  `(cmd, want_data, response_queue)` tuple and then blocks (with a timeout) on a private
  one-slot response queue. This serializes all traffic over the single socket regardless of
  which caller/thread invoked the method, which matters because `start_after`/`stop_after`
  fire from separate `threading.Timer` threads.
- **`configure()`** — sends `CONFIG`; returns whether the rig acknowledged.
- **`start(wait_s=None)` / `stop(wait_s=None)`** — send `START`/`STOP`, optionally sleeping
  first (this sleep happens synchronously on the *caller's* thread, not the network thread).
  Each is idempotent/no-op-safe: `start()` no-ops (returns `True`) if already imaging, `stop()`
  no-ops if not currently imaging, and both refuse (`return False`) once `finish()` has been
  called. Internal `_imaging` state tracks which is currently true.
- **`start_after(delay_s)` / `stop_after(delay_s)`** — schedule a `start()`/`stop()` on a
  daemon `threading.Timer` after `delay_s`, cancelling any previously pending timer of the
  same kind first. This is what `behavioral_master.py`'s `link.start_imaging(delay_s=...)` /
  `link.stop_imaging(delay_s=...)` (via `ArduinoLink.send_after`) ultimately compose with —
  note there are actually two independent scheduling mechanisms in play across the whole
  system: `ArduinoLink.send_after` schedules the Arduino TTL command, while
  `PrairieClient.start_after`/`stop_after` independently schedules the imaging-software
  command, and `main()` fires both roughly together so the TTL pulse and the imaging
  start/stop land at (approximately) the same time.
- **`restart_after(stop_delay_s, start_delay_s)`** — a convenience chained scheduler (stop,
  wait, start) that exists but is not currently invoked anywhere in `behavioral_master.py`
  (which instead separately calls `stop_after` and `start_after`/`start` back-to-back).
- **Timer `.join()` patch.** `_patch_join` monkey-patches each `threading.Timer` instance with
  an additional `isr_join(interval=0.2)` method that polls `is_alive()` in a loop rather than
  doing a single blocking join — a defensive pattern presumably intended for use from
  interrupt-sensitive contexts (name suggests "ISR-safe join"), though nothing in the uploaded
  code currently calls `.isr_join()` instead of the normal `.join()`.
- **`finish()`** — cancels any pending timers, then repeatedly sends `FINISH` (which the
  server apparently answers with successive JSON chunks containing `start_ts`/`stop_ts`
  arrays and a `done` flag) until the server signals `done`, accumulating the full list of
  imaging start/stop timestamps for the session. Sets `_finished = True`, after which all
  other methods become permanent no-ops. `PrairieInterface.finish()` (on the
  `behavioral_master.py` side) copies `client.start_ts`/`client.stop_ts` into
  `session_data.img` right after calling this.
- **Pending-error propagation.** Because `start_after`/`stop_after`/`restart_after` run on
  background timer threads, any exception they raise can't surface to the caller directly;
  instead `_set_pending_error` stashes the first such exception, and the (currently unused by
  `behavioral_master.py`) `raise_pending_error()` method exists to let a caller poll for and
  re-raise it later on the main thread. Since nothing currently calls
  `raise_pending_error()`, scheduled start/stop failures are effectively silent unless
  `verbose=True` wire logging is enabled — worth knowing if asked to debug why a scheduled
  imaging start/stop silently failed.
- **`disconnect()`** — stops the network thread, closes the read-file and shuts down/closes
  the socket; not currently called anywhere in `behavioral_master.py` (which calls `finish()`
  but never `disconnect()` on the shared `client`), so the socket is left open until process
  exit — only relevant if asked to debug resource cleanup or add explicit disconnect handling.
- **`if __name__ == "__main__":` block** — a standalone smoke-test/demo harness (configure,
  start, a long stop, then loop start/stop) not used by the main behavioral system; useful
  only as a reference for expected command sequencing against the real imaging rig.

---

## 7. Notable Cross-Cutting Details Worth Knowing
- The Python side treats the Arduino connection as optional almost everywhere: if no Arduino
  is found, `ArduinoLink.active` is `False` and every `send*`/`start_ephys`/`start_imaging`
  call becomes a no-op that still returns success, letting a trainer run the whole flow (minus
  real hardware) for development (`animal_id == "DEV"`).
- The global `EVT_QUEUE`/`ENC_QUEUE` (defined in `cursor_utils.py`, imported by
  `behavioral_master.py`) are the sole communication channel between the main serial-reading
  loop and the independent pygame cursor thread — there's no shared session-state object
  passed directly between them.
- Error handling is deliberately layered and non-fatal wherever possible: almost every cleanup
  step (closing the link, stopping the cursor, finishing Prairie, sending the session email,
  saving data) is wrapped so a failure in one doesn't prevent the others from running, with
  failures instead cached to `EXC_STACK` and optionally persisted to `errors.log`/GitHub via
  `ExceptionInterface`.
- Session data is saved in *pairs of columns* on shared per-cohort Google Sheets workbooks
  (one 2-column block per animal/phase/date, located by matching date/animal/phase header
  text), which is why the `FileLock` distributed-locking mechanism exists — concurrent writes
  from multiple rigs to the same workbook would otherwise race.
- `main()`'s hit/miss handling checks `client.stop_after(...)` and `link.stop_imaging(...)`
  (etc.) return values and raises if either reports failure, but those calls only report
  whether the *scheduling* succeeded, not whether the *scheduled* action itself later
  succeeds — a delayed `PrairieClient` start/stop failure is only recorded via
  `_set_pending_error` and, since `raise_pending_error()` is never polled, will not surface
  as a visible error during the session.

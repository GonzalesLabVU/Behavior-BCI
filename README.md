# Behavioral Master — Python-Side System Overview

This document describes the software that drives the rodent behavioral-training rig, with the
emphasis on the Python side (the counterpart to the Arduino `behavioral_controller.ino`
firmware). The centerpiece is `behavioral_master.py`, a single script that a trainer runs on
the primary training PC once per session. It talks to the Arduino over serial, optionally
drives a visual "cursor" feedback task (via `cursor_utils.py`), optionally records per-trial
behavior-camera video (via `camera_utils.py`), optionally synchronizes with a two-photon
imaging rig (via `TCPClient.py`, which talks to `TCPServer.py` on the microscope PC),
optionally emits electrophysiology sync pulses (decoded by the custom TTL Record Panel Open
Ephys plugin on the ephys PC), logs a status dashboard (via `dashboard_utils.py`), and saves
all collected session data to Google Sheets (with an incremental local JSON backup that is
kept if the Sheets save fails).

Upload this `.md` file together with `behavioral_master.py` to give a new chat session full
context on how the system works, even without `cursor_utils.py`, `camera_utils.py`,
`dashboard_utils.py`, `sheets_utils.py`, `TCPClient.py`, `TCPServer.py`, the launcher scripts, or
the TTL Record Panel plugin sources present.

---

## 1. File Map

| File | Role |
|---|---|
| `behavioral_master.py` | Main entry point. Orchestrates the whole session: prompts the trainer, talks to the Arduino, runs the event loop, and saves data. |
| `cursor_utils.py` | Runs a full-screen (or windowed) `pygame` visual-feedback widget ("cursor") that mirrors wheel position on screen during wheel-based phases, in its own thread. |
| `dashboard_utils.py` | Thin helper that writes live session status (running/idle, animal, phase) to a shared "Dashboard" Google Sheet so multiple rigs' status can be monitored centrally. |
| `sheets_utils.py` | `ResilientHTTPClient` — the `gspread` HTTP client used by both `behavioral_master.py` and `dashboard_utils.py`. Throttles requests per process and retries 429/5xx/timeouts with exponential backoff plus jitter (see §5.1). |
| `TCPClient.py` | Raw-socket TCP client (`PrairieClient`) that drives a Bruker/Prairie View two-photon imaging rig over the local network, used to start/stop imaging in sync with trial outcomes. |
| `camera_utils.py` | `FLIRCamera` — records one `.mp4` per trial from a FLIR (Spinnaker/PySpin) camera on a background thread, plus a `metadata.xml` of per-trial frame timestamps (see §7). |
| `TCPServer.py` | Runs on the **microscope PC**. Listens for `PrairieClient`'s commands and drives the Prairie View GUI via UI automation (see §8). |
| `behavioral_controller.ino` | Arduino Mega 2560 firmware: trial state machine, hardware I/O, and the record/event TTL lines (serial protocol in §2.2, TTL encoding in §9). |
| `TTLRecordPanel*.h/.cpp`, `OpenEphysLib.cpp` → `ttl-record-panel.dll` | Custom Open Ephys plugin ("TTL Record Panel") on the **ephys PC**: gates recording from one TTL line and decodes pulse-count event trains from another (see §9). |
| `run.bat`, `launcher.ps1` | Setup/launch scripts for the primary training PC: pull the latest code from GitHub (`git pull --ff-only`), build the Python venv, install dependencies, compile/upload the Arduino sketch, then run `behavioral_master` (see §10). |

### Rig computers
| Computer | Runs |
|---|---|
| Primary training PC (used by the trainer) | `behavioral_master.py`, `cursor_utils.py`, `camera_utils.py`, `dashboard_utils.py`, `sheets_utils.py`, `TCPClient.py`, `run.bat`/`launcher.ps1`. Connected by USB to the Arduino Mega 2560 and by a direct Ethernet cable to the microscope PC. |
| Microscope PC (Prairie View / Bruker two-photon) | `TCPServer.py` |
| Ephys PC (Open Ephys GUI) | `ttl-record-panel.dll` in the signal chain, receiving the Arduino's record/event TTL lines |

### Repository layout
The project folder on every machine is the root of the GitHub repo
`GonzalesLabVU/Behavior-BCI` (branch `main`), so the layout is flat: the `.py` files,
`run.bat`, `launcher.ps1`, `requirements.txt`, `animal_map.json`, `workbooks.json`,
`behavioral_controller/` (the Arduino sketch), and `open-ephys/ttl-record-panel.dll`.
`.gitignore` is an allow-list (`/*` ignores everything, then the paths above are re-included),
so secrets, `.venv/`, `backup/`, logs, the Spinnaker wheel, and the Open Ephys build trees can
never be committed by accident. Runtime errors are pushed by `ExceptionInterface` to the
separate `error-logs` branch so they never interfere with `git push`/`git pull` on `main`.
See §12 for the update workflow.

### External files/services this system expects at runtime
- `.env` (same directory as the script; **untracked, secrets only**) — `SMTP_USERNAME`,
  `SMTP_PASSWORD`, `SMTP_TO_ADDR`, `GITHUB_TOKEN` (a fine-grained token with Contents
  read/write on this repo only), `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN`, and
  optionally `NTP_SERVER` (defaults to `pool.ntp.org`) and `GOOGLE_SERVICE_ACCOUNT_JSON`.
- `rig.env` (same directory; **untracked, per machine**) — everything that differs between
  rigs: `CLIENT_ID` (`BEHAVIOR`, `IMAGING`, or `DEVELOPMENT`), `SPOUT_PULSE_MS`,
  `BRAKE_ENGAGE_MS`, `BRAKE_RELEASE_MS`, and optionally `SHEETS_MAX_REQ_PER_MIN` (per-process
  Sheets request budget, default 30; use roughly 90 divided by the number of rigs). It is
  loaded with `load_dotenv` right after `.env`, so `git pull` never overwrites it.
- `workbooks.json` (**tracked, shared by all rigs**) — maps `"DASHBOARD"` and each cohort name
  to a Google Sheets workbook ID, e.g. `{"DASHBOARD": "<id>", "COHORT1": "<id>", ...}`. A key
  must be exactly the cohort name used as a value in `animal_map.json`. Sheet IDs are
  identifiers, not credentials: the workbooks are "anyone with the link can view", and only
  the service account and named people have edit access.
- `credentials.json` — Google service-account credentials for the Sheets/Drive APIs
  (untracked). All rigs share the same service account, so they share one per-user Sheets API
  quota.
- `animal_map.json` (**tracked**) — a dict mapping underscore-joined animal keys (e.g.
  `"M1_M2_M3"`) to a cohort name (e.g. `"COHORT6"`), which `workbooks.json` then maps to a
  workbook ID.
- `errors.log` — local append-only error log, optionally pushed to the `error-logs` branch of
  the GitHub repo (`GonzalesLabVU/Behavior-BCI`) as `logs/errors_<client>.log` if
  `GITHUB_TOKEN` is set.
- `backup/` — per-session incremental JSON backup files (see `SaveInterface`/`BackupWriter`),
  created at setup and deleted after a successful save.
- `raw_cap_Animal=<id>_Phase=<phase>_<MM-DD-YYYY>.json` — local raw capacitive-sensor export
  written next to the script by `SaveInterface.save_raw()` (only if raw data exists).
- `D:\Behavioral\DeepLabCut\<YYYY-MM-DD>\Animal <id> - Phase <phase>\` — video output folder
  (`VIDEO_BASE_DIR`) used when camera recording is enabled.
- Launcher inputs (see §10): `requirements.txt`, a `behavioral_controller/` sketch folder
  containing `behavioral_controller.ino` and `sketch.yaml`, and (for video) a
  `spinnaker_python-*.whl` PySpin wheel matching Python 3.12.

---

## 2. `behavioral_master.py` — Architecture

The script is organized as a set of small **`InterfaceObject`** subclasses, each wrapping one
hardware/API/human boundary, plus a handful of free functions and two data-holding classes
(`SessionData`, `FileLock`). This mirrors an inversion-of-control style: `setup()` and
`main()` take an `interfaces` bundle (a `BehaviorInterfaces` instance) so that any interface
can be swapped out (e.g. for testing) without changing the control flow.

### 2.1 Interface classes

- **`SystemInterface`** — environment/filesystem/hardware glue: loads `.env` and `rig.env`,
  reads `animal_map.json` and `workbooks.json`, resolves an animal's Google Sheets workbook ID
  (animal → cohort name via `animal_map.json`, cohort name → ID via `workbooks.json`), finds and
  opens the Arduino's serial port (`_find_arduino_port` looks for "arduino" or "usb serial" in
  the port description), and builds phase-specific config dicts (brake/spout timing from
  `rig.env`; wheel threshold, side, and reverse flag from the hard-coded `PHASE_CONFIG` table).
- **`ConsoleInterface`** — all trainer-facing console output: session start banner, a
  live-updating trial-status table (trial count, elapsed time, hit/miss block indicators, hit
  rate), a duration summary, and formatted display of any exceptions collected in the global
  `EXC_STACK` at the end of the run.
- **`TrainerInterface`** — all interactive `input()` prompts: whether to flush the reward
  line, animal ID (blank → `"DEV"` debug animal), training phase, target side (phases 4 and 5
  only), whether video/imaging/ephys are active, the continuous-imaging start/stop window
  (phase 2 with imaging only), and confirmation prompts around saving/overwriting.
- **`DashboardInterface`** — pushes `running`/`finished`/`idle` status (plus animal/phase) and
  the live success-rate string (`performance`, updated after every hit/miss) to the shared
  Dashboard sheet via `dashboard_utils.write_fields`. It also appends a "started training"
  line to the Notifications sheet (`record_notification`) at session start and a row to the
  Recent Sessions sheet (`record_session`) at the end. Every call just puts a job on an
  internal `Queue`; a single daemon worker thread (`_run`) performs all the Sheets I/O, so the
  main event loop never waits on the network. The worker merges queued `fields` updates
  (latest value wins) and holds back performance-only updates until at least
  `MIN_PERF_INTERVAL_S` (5 s) after the previous write, while status changes go out
  immediately. Every write is failure-safe (a printed warning, not a crash). `flush()` blocks
  (default 90 s) until the queue is drained and is called at the very end of the `__main__`
  driver. `notify_idle()` (status `idle`, animal/phase cleared) is called from the `__main__`
  driver after the session has been saved/logged.
- **`ExceptionInterface`** — collects exceptions into the module-level `EXC_STACK` (for console
  display), appends formatted tracebacks to the local `errors.log`, and can commit/push that log
  to the `error-logs` branch of the GitHub repo (as `logs/errors_<client>.log`, one file per
  rig) using a throwaway shallow clone (via `git` subprocess calls) when `GITHUB_TOKEN` is
  configured.
- **`EmailInterface`** — sends a plain-text/HTML session-summary email (date, start/stop
  time, duration, trial count, hit rate) via Gmail SMTP once a real (non-`DEV`) session
  finishes.
- **`SaveInterface`** — the most complex interface; writes session data into a per-animal Google
  Sheets workbook (see §11), plus a local raw capacitive-sensor JSON export (`save_raw`). During
  the session, `start_backup`/`append_backup` stream an incremental per-trial JSON backup
  (`backup/`) through a background `BackupWriter` thread: each trial stores only the
  `SessionData.diff()` slice added since the previous trial, and the file is rewritten
  atomically via a temp file. The backup is deleted after a successful Sheets save and kept if
  the save fails (it replaces the old full-session fallback file). `save_session()` calls
  `save_data()` up to 3 times (waiting about 20 s, then 40 s, plus jitter) before giving up; a
  retry is safe because `resolve_protocol()` has already set `overwrite_confirmed`, so a
  partially written block is simply found and overwritten. `resolve_protocol()` handles same-day
  duplicate sessions: it prompts to overwrite, or auto-overwrites when the earlier session was
  under 5 minutes and the new one is longer. Note: `_build_rows` doesn't currently write a
  `duration_sec` row to the Metadata sheet, so the existing-duration lookup behind that
  auto-overwrite rule finds nothing for sessions saved with the current code.
- **`PrairieInterface`** — wrapper for connecting to / finishing an imaging session via
  `PrairieClient`. `connect()` instantiates the client and sends `CONFIG`; if the connection
  or `CONFIG` fails it prints a warning and the session continues without imaging
  (`meta['imaging_active']` ends up `False`). `finish()` calls `client.finish()` and copies
  the returned start/stop timestamps into `session_data.img`.
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
- `send_and_wait(text)` writes a line and blocks (5 s default timeout per attempt, 2 attempts,
  then `TimeoutError`) until the next
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
  `R2`; `stop_ephys(safe=True)` is idempotent and caches rather than raises failures),
  `start_imaging()`/`stop_imaging()` (send `img_start`/`img_stop`, optionally delayed; the
  Arduino only emits the matching `EVENT_TTL` pulse trains when its `ephys` flag is 1, see §9).

### 2.3 `SessionData` — in-memory session record
A plain data container accumulated throughout the session and eventually serialized:
- `meta` — animal/phase/date, timing, imaging/ephys/video flags, the optional
  continuous-imaging window (`img_start_t`/`img_stop_t`), side override, adaptive-difficulty
  parameters (`K1`, `K2`), internal bookkeeping (`_img_tcp_state`, `_backup_path`), and
  per-trial config log (`trial_config`: trial number, easy/normal, side).
- `evt` — parallel timestamp/value lists of the behavioral events written to the Event sheet
  (`cue` or, in phases 1–2, `trial_start`; `r_cue`; `hit`; `miss`; `trial_stop`; plus
  synthetic `setK <value>` calibration events). `lick` events are not logged here (see
  `raw.evt`).
- `enc` — parallel timestamp/value lists of wheel-displacement milestone samples.
- `img` — lists of imaging start/stop timestamps (filled in from the Prairie client at the
  end of the session).
- `raw` — two sub-logs: `evt` (a raw log of `hit`/`lick` events; this is the only place licks
  are recorded, and it goes to the local backup but is not written to Google Sheets) and
  `cap` (raw capacitive touch-sensor stream, only populated if the Arduino were built with
  `RAW_FLAG` enabled; exported locally by `save_raw`).
- Helper methods: `log_trial_config`, `add_evt`/`add_enc`/`add_raw_cap`/`add_raw_evt`,
  `img_mark_*`/`img_can_start`/`img_can_stop` (a small configured→started→stopped state
  machine that keeps duplicate imaging start/stop commands from being sent), `any_data()`
  (used to decide whether there's anything worth saving), `diff()` (returns only the entries
  added since the last call, for the incremental backup), `to_dict()` (JSON-safe
  serialization; currently not called anywhere), and `is_finished` (both start/stop
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
- `update()` returns immediately, with no API calls, while at least `reset_s` seconds of lease
  remain (this removes about 40 reads from a typical save); once less remains it re-reads the
  lock tab to confirm ownership. `update()`/`reset()` are called periodically during a long save
  to refresh/extend the lease (`reset_s`/`lease_s` thresholds) and to detect if the lock has
  been lost (e.g. deleted by another client after expiry) — in which case a `RuntimeError`
  aborts the save.
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
  `do_calibration`) looks at the animal's hit count over the last `N` trials (initially 20,
  then `N = 4*K`) and sets a new `K` (fewer than 9 hits → more frequent easy trials: `K=3`;
  9–11 hits → `K=5`; 12 or more → `K=7`), logging a synthetic `"setK <K>"` event and locking
  in `calibrated = True` for the rest of the session.
- **`_is_early_exit(...)`** contains a (currently dead — the function returns `False`
  unconditionally on its very first line) more elaborate implementation that would have
  tracked recent trials-per-minute and ended the session early if the animal's trial rate
  dropped too low for too long. Worth knowing about if asked to debug/re-enable early-exit
  behavior, since the "real" logic already exists below the early `return False`.

### 2.6 Timestamps (NTP-corrected clock)
All behavioral/encoder timestamps are `HH:MM:SS.mmm` strings from `_get_ts()`, which adds a
module-level NTP offset (`_NTP_OFFSET`) to the local clock. `setup()` measures the offset once
(`_sync_ntp_offset()`, server from `NTP_SERVER`, default `pool.ntp.org`) and starts a daemon
thread that re-syncs every 10 minutes; if a sync fails it prints a warning and keeps the last
offset (or the plain local clock). The same function is passed to `FLIRCamera` as `ts_fcn`, so
video frame timestamps share this clock. Imaging start/stop timestamps come from
`TCPServer.py` on the microscope PC, which stamps them with that machine's own system clock
(`datetime.now()`), not this NTP-corrected offset.

---

## 3. Program Flow

### 3.1 `setup(interfaces)`
Runs once at program start, before the main event loop:
1. Syncs the NTP clock offset (and starts the background re-sync thread), then finds/opens
   the Arduino serial port (prints a warning if none is found, though setup aborts in step 3
   without one) and wraps it in an `ArduinoLink`.
2. Prompts whether to flush the reward line; if so, sends `flush 1` and lets the Arduino's
   own flush-and-reboot routine run (which raises `SystemExit(0)` back on this side once the
   Arduino signals `RESTART`).
3. Prompts for animal ID (blank defaults to `"DEV"`) and training phase; raises if no Arduino
   was detected (this applies to every animal, including `"DEV"`). `"DEV"` also turns on
   verbose serial logging.
4. For phases `"4"` and `"5"`, additionally prompts for a target side override.
5. Prompts whether video, imaging, and/or ephys are active (plus the continuous-imaging
   start/stop minutes when imaging is on in phase `"2"`).
6. Builds the phase config (brake/spout timings from `rig.env`, wheel threshold/side/reverse from
   `PHASE_CONFIG`), sends the full config + ephys handshake to the Arduino, creates the
   `SessionData` and starts its incremental backup, connects to Prairie View if imaging is
   active (continuing without imaging on failure), initializes/configures the `FLIRCamera` if
   video is active (continuing without video on failure), and — for phases with a wheel
   component — starts the `CursorInterface` visual feedback task with the first trial's
   easy/side settings.
7. For phase > 1, sends the first trial's config line (`"1 <0|1>"`) to the Arduino.
8. Records the first trial's config, starts ephys TTL if requested, sends the final
   `start 1` handshake command, and notifies the dashboard that the session is `running`.
9. On any exception during setup, caches it, deletes the just-created backup file, best-effort
   closes the camera and the link/serial port, and re-raises (so `__main__` can log/report it
   without crashing ungracefully).

### 3.2 `main(link, session_data, cursor, client, interfaces)`
The live session event loop, driven entirely by messages arriving on `link.msg_q` from the
Arduino reader thread:
- On the very first message, records the session start time and prints the console header.
- **`EVT` messages** are the behavioral event stream and drive almost everything:
  - Every event is also pushed to the global `EVT_QUEUE` so the `cursor_utils` pygame thread
    can react to it (new trial cue, hit/miss) independently of this loop.
  - `cue` (or `trial_start`, which the firmware sends instead of a tone cue in phases 1–2):
    logs the event, starts the next per-trial video file if video is active, and — if imaging
    is active, this is the very first trial, and the phase isn't the continuous-imaging
    phase 2 — sends the initial imaging-start command and starts the TTL sync line together,
    then begins tracking trial number/timing. This is also when any queued next-trial config
    is actually sent to the Arduino (see `hit`/`miss` below).
  - `trial_stop`: logged as an event, queues the trial's new data slice to the incremental
    backup (`append_backup`), and stops the current video file if video is active.
  - `r_cue`: logged as an event (no other side effects here — the re-cue tone is purely an
    Arduino/`cursor_utils` phenomenon).
  - `hit`/`miss`: de-duplicated (an identical outcome arriving less than 100 ms after the
    previous one is ignored); on the real transition it logs the event, and — in every phase
    except the continuous-imaging phase 2 — if imaging is active it schedules (via
    `send_after`) a stop-then-restart of both the Prairie client and the imaging TTL line
    around the inter-trial interval (stop at +1 s, restart at +3 s) so each trial gets its own
    imaging bout. It then updates the rolling trial-outcome window, prints the console
    trial-status row, pushes the running success rate to the dashboard, and — if calibration
    is active — updates the trial-difficulty stack and (once `N` trials have accumulated)
    computes a new `K`. For phase ≥ 4, once calibrated it checks the (currently disabled)
    early-exit condition, and otherwise computes the next trial's easy/side settings and
    queues them as `pending_next_cfg`; the config line is sent to the Arduino (and the
    on-screen cursor's target settings updated to match) when the next `cue`/`trial_start`
    event arrives, not immediately.
  - `hit`/`lick` events are additionally mirrored into `session_data.raw.evt` (the only place
    `lick` events are recorded).
- **`ENC` messages** — parsed as a float wheel-displacement sample, appended to
  `session_data.enc`, and also pushed onto the global `ENC_QUEUE` for the cursor task to
  render live.
- **`RAW` messages** — appended to `session_data.raw.cap` (only produced if the firmware's raw
  capacitive stream is enabled).
- **`END`** (Arduino's `"S"` cleanup marker) — marks ephys as stopped in metadata and breaks
  out of the loop, ending the session normally.
- **`ERR`** — re-raises the underlying exception from the reader thread (or wraps a
  non-exception payload in a `RuntimeError`).
- **Continuous imaging (phase 2 only)** — instead of per-trial imaging bouts, when imaging is
  requested the trainer's start/stop minutes (`img_start_t`/`img_stop_t`) become two daemon
  `Timer`s armed on the first Arduino message. They fire `_img_window_start` /
  `_img_window_stop`, each of which sends the Prairie command and the matching imaging TTL
  command once. The timers are cancelled, and a still-running imaging window is stopped, in
  the `finally` block. `SessionData`'s `_img_tcp_state` guards every start/stop so duplicates
  are skipped.
- On `KeyboardInterrupt`, marks the session aborted, safely stops ephys, and calls `_cleanup`
  (which sends the Arduino an `"E"` abort command and waits, with a timeout, for its `END`
  acknowledgment) before re-raising.
- The `finally` block always stamps `t_stop`/`duration_sec`, safely stops ephys, cancels any
  continuous-imaging timers, stops the Prairie client if imaging is still running, queues one
  last backup slice, notifies the dashboard that the session has finished, and then sleeps
  6 s before restoring the default Ctrl-C handler.

### 3.3 `if __name__ == "__main__":` driver
Wraps `setup()` + `main()` in a top-level try/except/finally that:
- Tracks `animal_id_for_log`/`phase_id_for_log` for error-log context even if setup fails
  early.
- Treats `SystemExit`/`KeyboardInterrupt` as clean exits; caches and displays anything else.
- Always shows a run summary; then finishes the Prairie client (queueing a final `"img"`
  backup slice with the imaging timestamps), stops the cursor task, closes the Arduino link,
  and closes the camera (each independently guarded so one failure doesn't block the other
  cleanup steps; each failure is logged/committed via `ExceptionInterface`).
- If there's no data or it's a `DEV` run, the backup file is simply deleted. Otherwise it
  prompts to confirm saving, resolves the duplicate-session overwrite protocol
  (`SaveInterface.resolve_protocol`), and — if confirmed — saves the raw capacitive data
  locally and the full session to Google Sheets; if the Sheets save fails, a warning is
  printed and the local backup file is kept.
- Records the session's final success rate to the dashboard's Recent Sessions log, marks the
  dashboard `idle`, and — if the session actually finished (`is_finished`) and wasn't the `DEV`
  animal — emails a session summary. Just before the exceptions are re-displayed it calls
  `interfaces.dashboard.flush()` so the queued dashboard writes (status, Recent Sessions row)
  finish uploading while the email is being sent.
- Finally re-displays any collected exceptions along with a `[Process exited with code N]`
  line. The script itself no longer waits for a keypress; `launcher.ps1` pauses on "Press
  Enter to continue" so a double-clicked window doesn't vanish immediately.

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

A small, independent Google Sheets writer used for a live multi-rig status dashboard plus its
recent-session and notification logs (distinct from the per-animal data workbooks that
`SaveInterface` writes to).

- Lazily loads `.env` (its own private cache, separate from `behavioral_master.py`'s
  `dotenv`-based loading) only for the optional `GOOGLE_SERVICE_ACCOUNT_JSON` fallback, and
  lazily builds/caches a `gspread` client (using `ResilientHTTPClient`) from either
  `credentials.json` or that variable. The dashboard workbook ID is the `"DASHBOARD"` entry in
  `workbooks.json`.
- The workbook and each worksheet are cached in module-level dicts (`_workbook()`,
  `_get_worksheet()`), so steady-state dashboard updates make no read requests. Any failed write
  calls `_invalidate_cache()` so the next call re-opens everything (e.g. after a sheet is
  renamed or deleted). All calls are made from `DashboardInterface`'s single worker thread.
- `CLIENT_START_COLS` maps three fixed client types (`BEHAVIOR`, `IMAGING`, `DEVELOPMENT` —
  with `BEH`/`IMG`/`DEV` shorthand accepted) to fixed starting columns on a single `"Dashboard"`
  worksheet; `FIELD_ROWS` maps `status`/`animal`/`phase`/`performance` to fixed rows.
- `write_fields(client_id, fields, timestamp=None)` — normalizes the client ID, resolves (from
  cache, or creates if missing) the `"Dashboard"` worksheet, and batch-updates a
  `[timestamp, value]` pair into the two-column block for each requested field/row. This is the function
  `DashboardInterface._safe_write()` calls on the `behavioral_master.py` side to report
  `running`/`finished`/`idle` status plus the current animal/phase.

- `log_session(client_id, animal, phase, performance, start_utc=None)` — inserts a row (date,
  platform, animal, phase, performance, local start time) at the top of a `"Recent Sessions"`
  worksheet, creating it with headers if missing; skipped when `performance` is blank.
- `log_notification(message)` — inserts a `[timestamp, message]` row at the top of a
  `"Notifications"` worksheet (also auto-created). `DashboardInterface.notify_start()` uses it
  to announce each session start.

### 5.1 `sheets_utils.py` — Rate-limit-safe Sheets client

`ResilientHTTPClient` subclasses `gspread`'s `HTTPClient` and is passed as `http_client=` to
`gspread.authorize()` in both `behavioral_master.py` (`API_CLIENT`) and `dashboard_utils.py`.
- **Throttle.** `_throttle()` keeps a sliding 60-second window of request timestamps for the
  process and sleeps (with jitter) when it already holds `SHEETS_MAX_REQ_PER_MIN` (from
  `rig.env`, default 30) requests. Google's Sheets quotas are per user per minute, and every
  rig authenticates as the same service account, so size this as roughly 90 divided by the
  number of rigs.
- **Retry.** Up to 7 attempts; HTTP 408, 429, and >=500 responses and connection
  errors/timeouts are retried after 2, 4, 8, ... up to 60 s plus jitter. Other errors (e.g.
  404) are raised immediately.
- Dashboard writes are made on a background thread (see `DashboardInterface`), so backoff
  waits never stall the trial loop. Save-time calls (`SaveInterface`, `FileLock`) run on the
  main thread after the session has ended.

---

## 6. `TCPClient.py` — Two-Photon Imaging (Prairie View) Client

Implements `PrairieClient`, a raw-TCP client that talks to a fixed local-network endpoint
(client bound to `192.168.2.1`, server at `192.168.2.2:5005`), i.e. the primary training PC
talking over a direct Ethernet cable to `TCPServer.py` (§8) on the microscope PC that controls
Prairie View. This is what `PrairieInterface` in `behavioral_master.py` wraps.

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
- **`finish()`** — cancels any pending timers, then repeatedly sends `FINISH` (which
  `TCPServer.py` answers with one `{start_ts, stop_ts, done}` JSON object per call, walking
  its recorded timestamps in order) until the server signals `done`, accumulating the full
  list of
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

## 7. `camera_utils.py` — Behavior-Camera Video

`FLIRCamera` records per-trial video from the first FLIR camera found through the Spinnaker SDK
(`PySpin`, which `launcher.ps1` installs from a local wheel). It is created directly in
`setup()` (there is no `InterfaceObject` wrapper) only when the trainer answers "yes" to the
video prompt; if the camera can't be initialized, setup prints a warning and continues without
video (`meta['video_active'] = False`).

- **Output.** Files go to `VIDEO_BASE_DIR / <YYYY-MM-DD> / "Animal <id> - Phase <phase>"`
  (`D:\Behavioral\DeepLabCut` by default), one file per trial named
  `<date>_animal-<id>_phase-<phase>_trial-<n>.mp4`.
- **Configuration (`configure()`).** Full-sensor field of view, 4×4 averaged binning, manual
  exposure (2000 µs) and gain (15), `Mono8` pixels, 30 fps, maximum USB throughput; it then
  does a throwaway `cv2.VideoWriter` open/close to warm up the `avc1` encoder.
- **Recording.** `start()` spawns a daemon acquisition thread that writes frames with
  `cv2.VideoWriter` and logs, per trial, the first/last frame timestamp (via the injected
  `ts_fcn`, i.e. `_get_ts`) and the frame count. `stop()` signals the thread and joins it
  (5 s timeout), re-raising any exception captured on the thread. `behavioral_master.main()`
  calls `start()` on each `cue`/`trial_start` and `stop()` on each `trial_stop`.
- **Metadata.** `close()` (also registered with `atexit`) stops any active recording, writes
  `metadata.xml` into the output folder with the measured `frameRate`, `frameCount`, `tStart`,
  and `tStop` for every trial, and releases the camera.

---

## 8. `TCPServer.py` — Microscope-PC Server (Prairie View Automation)

Runs on the computer that controls the Bruker two-photon microscope and is the other half of
`TCPClient.py`. It binds to `192.168.2.2:5005`, accepts a single client connection, and
executes newline-terminated commands by driving the Prairie View 5.8 GUI through `pyautogui`/
`pynput`/`mss` UI automation (if those imports fail it runs network-only and the GUI actions
are no-ops).

- **Startup.** `main()` looks for a window titled `Prairie View 5.8` (error dialog if it can't
  be found), draws a transparent red border overlay around the monitor holding that window,
  starts the server, and polls every 500 ms; the process exits if Prairie View's window
  disappears. Answering `y` to the "development mode" prompt instead prints the mouse position
  relative to the Prairie View window, for re-calibrating the hard-coded click targets.
- **Commands** (`VALID_CMD`). Each is answered with `OK` (plus a JSON payload for `FINISH`) or
  `ERR`, and is only accepted in the `IDLE` state:
  - `CONFIG` — clicks the T-Series controls and types the period (`0.001`) and duration
    (`35`) values. Repeats are ignored once configured.
  - `START` — increments the server's trial counter, appends a timestamp to `start_ts`, types
    the counter into the iteration field, and clicks the trigger button.
  - `STOP` — appends a timestamp to `stop_ts`, clicks the trigger button again, then closes
    the "Playback Controls" window if it appears.
  - `FINISH` — returns one recorded `{start_ts, stop_ts, done}` entry per call; after the last
    entry it reports `done` and shuts the server down.
- **Fragility.** Click targets are fixed pixel offsets (`PrairieUser.pixel_map`) relative to
  the Prairie View window, so any change to its layout or display scaling requires
  re-calibration. Timestamps use this PC's local clock (`HH:MM:SS.mmm`).

---

## 9. TTL Record Panel (Open Ephys Plugin) and the Arduino TTL Lines

`ttl-record-panel.dll` is built from `OpenEphysLib.cpp`, `TTLRecordPanel.h/.cpp`, and
`TTLRecordPanelEditor.h/.cpp` and loaded into the Open Ephys signal chain on the ephys PC as a
`UTILITY` processor named "TTL Record Panel" (library version 0.1.0). It consumes TTL events
from the Arduino's two digital outputs, `RECORD_TTL` and `EVENT_TTL` (Mega: pins 22/23;
Uno: 10/11).

- **Record gate (`record_line`, default 1-based line 5).** A rising edge calls
  `CoreServices::setRecordingStatus(true)` and a falling edge stops recording. The Python side
  drives this line with `ArduinoLink.start_ephys()`/`stop_ephys()` (`R1`/`R2` → `RECORD_TTL`
  HIGH/LOW); the firmware also forces both lines LOW during cleanup.
- **Event trains (`event_line`, default 6).** The firmware's `sendPulseTrain()` marks each
  event with a number of pulses (11.111 ms high, 11.111 ms gap, then a 40 ms train gap). The
  plugin counts rising edges and decodes a train once the line has been quiet for
  `decode_gap_ms` (default 33 ms). A decoded count `N` is written as a one-sample pulse on the
  plugin's own `"TTL"` channel at line index `N-1` (so `states.npy` shows ±`N`); counts above 8
  are ignored. Pulse counts: 1 = `lick`, 2 = `cue` (or `r_cue`), 3 = `hit` (or `miss`),
  4 = `img_start`, 5 = `img_stop`. `trial_start`/`trial_stop` and all other events send no
  pulses, so hit vs. miss and cue vs. re-cue can't be told apart from the TTL record alone.
- **Only when `ephys` is on.** `sendPulseTrain()` returns immediately unless the session
  config has `ephys 1`, so the event and imaging pulse trains (including those triggered by
  `img_start`/`img_stop`) are only emitted for ephys sessions.
- **Editor/parameters.** Combo boxes for `record_line`/`event_line` (1-based GUI numbering,
  converted from the 0-based `TTLEvent::getLine()`), a `decode_gap_ms` box, and
  `enable_record_control`/`enable_event_decode` checkboxes. A status-bar warning is sent if
  both lines are set to the same number. With Binary recording, decoded events are saved under
  `events/TTL_Record_Panel-<node id>.Rhythm Data/TTL/`.

---

## 10. Launcher (`run.bat` / `launcher.ps1`)

`run.bat` unblocks and runs `launcher.ps1` (PowerShell 5.1+) from its own folder; this is the
normal way to start a session on the primary training PC. Each run:
0. If the folder is a git checkout and `git` is on `PATH`, runs `git pull --ff-only` (via
   `Update-FromGit`) so the Python code, `animal_map.json`, `workbooks.json`,
   `requirements.txt`, and the Arduino sketch are current before anything else happens. A
   missing `git`, no network, or a failed pull only prints a warning and the launch continues
   with the local files. (Avoid editing `run.bat` after rollout: a batch file that is replaced
   while it is running can misbehave. `launcher.ps1` is safe because PowerShell parses the whole
   script first.)
1. Ensures a Python 3.12 virtual environment (`.venv`), installing Python via `winget` if the
   `py` launcher doesn't have it, then upgrades pip and installs `requirements.txt`.
2. Installs the PySpin wheel (`spinnaker_python-*.whl`, `cp312`/`win_amd64`) if PySpin isn't
   importable (only a warning if it's missing, since video is optional).
3. Ensures `arduino-cli` and the `arduino:avr` core, plus the libraries listed in
   `behavioral_controller/sketch.yaml`.
4. Detects an Arduino Mega 2560 (or Uno) on USB, compiles the `behavioral_controller` sketch
   (auto-installing a missing library header such as `Servo.h` and retrying), and uploads it,
   so the firmware is re-flashed on every launch.
5. Runs `python -m behavioral_master` from the project folder (which is why
   `SCRIPT_DIR = Path.cwd()` finds `.env`, `rig.env`, `credentials.json`, `animal_map.json`,
   `workbooks.json`, and `backup/`), then waits for Enter before closing.

---

## 11. Notable Cross-Cutting Details Worth Knowing
- `ArduinoLink` still turns every `send*`/`start_ephys`/`start_imaging` call into a no-op when
  its serial port is missing (`active` is `False`), but `setup()` now raises if no Arduino is
  detected (even for the `DEV` animal), so the old keyboard-only "no hardware" dev mode is
  effectively gone. The other optional pieces degrade gracefully instead: a failed Prairie
  connection or camera init just prints a warning and the session continues without
  imaging/video.
- The global `EVT_QUEUE`/`ENC_QUEUE` (defined in `behavioral_master.py` and passed into
  `cursor_utils.BCI`) are the sole communication channel between the main serial-reading
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
- **Google Sheets rate limits.** The Sheets API allows roughly 60 reads and 60 writes per minute
  per user, and all rigs share one service account. Three things keep the system under that:
  cached dashboard worksheets plus coalesced, throttled dashboard writes; `FileLock.update()`
  skipping the network while the lease is healthy; and `ResilientHTTPClient` throttling and
  retrying every request. If saves still fail repeatedly, the local `backup/` file is kept,
  exactly as before.
- Session timestamps are stamped in `ArduinoLink`'s reader thread (not the main loop), so a
  slow main loop would not corrupt timestamps, but it would delay imaging scheduling and
  cursor updates. This is why all dashboard I/O is kept off the main loop.

---

## 12. Repository, Deployment, and Updating Rigs

- **One repo, flat layout.** Each machine's project folder is a clone of
  `https://github.com/GonzalesLabVU/Behavior-BCI` (branch `main`). Tracked: code,
  `animal_map.json`, `workbooks.json`, `requirements.txt`, `run.bat`, `launcher.ps1`,
  `behavioral_controller/`, `open-ephys/ttl-record-panel.dll`. Untracked (copy by hand once per
  machine): `.env`, `rig.env`, `credentials.json`, and `spinnaker_python-*.whl`.
- **Developer workflow** (from the project folder):
  `git add -A`, `git commit -m "..."`, `git push`.
- **Rigs** pick up changes by relaunching (`launcher.ps1` runs `git pull --ff-only`) or by
  running `git pull --ff-only` manually. Firmware is re-flashed on every launch, so Arduino
  changes arrive the same way.
- **Adding a cohort:** add its animals to `animal_map.json` (`"X1_X2_X3": "COHORT9"`) and its
  workbook ID to `workbooks.json` (`"COHORT9": "<id>"`) in one commit; no per-rig edits are
  needed. Share the new workbook with the service account (Editor) and set general access to
  "Anyone with the link: Viewer" (never Editor).
- **Per-rig tuning:** edit that rig's `rig.env` only.
- **Setting up a new rig:** clone the repo (or run `git init -b main`,
  `git remote add origin <url>`, `git fetch origin`, `git checkout -f -B main origin/main` in an
  existing folder), then create `.env`, `rig.env`, and `credentials.json`, and copy in the
  Spinnaker wheel if the rig records video.
- **Error logs** go to the `error-logs` branch (`logs/errors_<client>.log`), never to `main`.
  The `GITHUB_TOKEN` in `.env` only needs Contents read/write on this repo.
- If `git pull` warns about local changes that were never intended (usually CRLF/LF line-ending
  noise), run `git status` on that rig; for pure line-ending changes, `git checkout -- .`
  restores the committed versions.
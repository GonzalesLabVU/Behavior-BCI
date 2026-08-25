"""
In-memory session state and local disk recording: tracks trial/event
data as it happens and writes the append-only, crash-safe session record
"""

import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from collections import deque

from behavior_rig.timeutil import _get_ts, _ts_to_ms


class SessionData:
    def __init__(self, animal_id, phase_id, date_str):
        """
        Initialize containers for one behavioral session.

        Args:
            animal_id: Animal identifier for the session.
            phase_id: Training phase identifier.
            date_str: Session date string.
        """
        self.session_id = uuid.uuid4().hex
        self.schema_version = "1"
        self.started_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

        self.meta = {
            "client": None,
            "workbook_id": None,
            "date": date_str,
            "animal": animal_id,
            "phase": phase_id,
            "session_id": self.session_id,
            "schema_version": self.schema_version,
            "started_at_utc": self.started_at_utc,
            "aborted": False,
            "t_start": None,
            "t_stop": None,
            "duration_sec": None,
            "imaging_active": False,
            "ephys_active": False,
            "K1": 5,
            "K2": None,
            "easy_trials": [],
            "normal_trials": [],
            "left_targets": [],
            "right_targets": [],
            "both_targets": []
            }
        
        self.trial_config = []

        self.evt = {"timestamps": [], "values": []}
        self.enc = {"timestamps": [], "values": []}
        self.img = {"start_ts": [], "stop_ts": []}
        self.raw = {
            "evt": {"timestamps": [], "values": []},
            "cap": {"timestamps": [], "values": []}
            }

    def _ensure_session_tracking(self):
        self.meta.setdefault("trial_config", [])
        self.meta.setdefault("K1", 5)
        self.meta.setdefault("K2", None)

    def log_trial_config(self, trial_n, type, side):
        self._ensure_session_tracking()

        self.meta['trial_config'].append({
            "trial": int(trial_n),
            "is_easy": bool(type),
            "side": str(side)
            })

    def add_evt(self, ts, payload):
        """Append a parsed behavioral event.

        Args:
            ts: Event timestamp string.
            payload: Event label or payload value.
        """
        self.evt["timestamps"].append(ts)
        self.evt["values"].append(payload)

    def add_enc(self, ts, payload):
        """Append an encoder sample.

        Args:
            ts: Encoder timestamp string.
            payload: Encoder value.
        """
        self.enc["timestamps"].append(ts)
        self.enc["values"].append(payload)

    def add_raw_cap(self, ts, payload):
        """Append a raw capacitive sensor sample when it parses as an integer.

        Args:
            ts: Sample timestamp string.
            payload: Raw sample payload.
        """
        try:
            v = int(str(payload).strip())
        except Exception:
            return

        self.raw["cap"]["timestamps"].append(ts)
        self.raw["cap"]["values"].append(v)
    
    def add_raw_evt(self, ts, payload):
        """Append a raw event marker.

        Args:
            ts: Event timestamp string.
            payload: Raw event payload.
        """
        self.raw["evt"]["timestamps"].append(ts)
        self.raw["evt"]["values"].append(str(payload))

    def any_data(self, field=None):
        """Check whether session data has been collected.

        Args:
            field: Optional field name to check: evt, enc, img, or raw.

        Returns:
            True when the selected data exists, otherwise False.
        """
        if field is None:
            return (
                bool(self.evt["timestamps"]) or
                bool(self.enc["timestamps"]) or
                bool(self.img['start_ts']) or
                bool(self.raw["evt"]["timestamps"]) or
                bool(self.raw["cap"]["timestamps"])
                )
        
        match field:
            case "evt":
                return bool(self.evt['timestamps'])
            case "enc":
                return bool(self.enc['timestamps'])
            case "img":
                return bool(self.img['start_ts'])
            case "raw":
                return bool(self.raw['cap']['timestamps'])
        
        raise ValueError(f"Invalid field: {field!r} (Expected one of: None, 'evt', 'enc', 'raw')")

    def to_dict(self):
        """
        Convert the session data to JSON-safe dictionaries.

        Returns:
            Dictionary containing metadata, event, encoder, imaging, and raw data.
        """
        def _json_safe(x):
            """
            Recursively convert values to JSON-safe primitives.

            Args:
                x: Value to convert.

            Returns:
                JSON-safe representation of the value.
            """
            if x is None or isinstance(x, (str, int, float, bool)):
                return x
            
            if isinstance(x, dict):
                return {str(k): _json_safe(v) for k, v in x.items()}

            if isinstance(x, (list, tuple)):
                return [_json_safe(v) for v in x]
            
            return str(x)
        
        meta_out = dict(self.meta)
        cfg = meta_out.get('trial_config', []) or []

        try:
            easy_trials = [c['trial'] for c in cfg if c.get('is_easy') is True]
            normal_trials = [c['trial'] for c in cfg if c.get('is_easy') is False]

            left_targets = [c['trial'] for c in cfg if c.get('side') == "L"]
            right_targets = [c['trial'] for c in cfg if c.get('side') == "R"]
            both_targets = [c['trial'] for c in cfg if c.get('side') == "B"]
        except Exception:
            easy_trials, normal_trials = [], []
            left_targets, right_targets, both_targets = [], [], []

        meta_out['easy_trials'] = list(easy_trials)
        meta_out['normal_trials'] = list(normal_trials)
        meta_out['left_targets'] = list(left_targets)
        meta_out['right_targets'] = list(right_targets)
        meta_out['both_targets'] = list(both_targets)

        return {
            'session_id': self.session_id,
            'schema_version': self.schema_version,
            'started_at_utc': self.started_at_utc,
            'meta': _json_safe(meta_out),
            'evt': _json_safe(self.evt),
            'enc': _json_safe(self.enc),
            'img': _json_safe(self.img),
            'raw': _json_safe(self.raw)
            }

    @property
    def is_finished(self):
        """Return whether the session has both start and stop timestamps."""
        return (self.meta['t_start'] is not None) and (self.meta['t_stop'] is not None)


class SessionWriter:
    """
    Append-only local recorder

    This is the canonical session record — Google Sheets is an export target, not the source of truth
    """
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self._fh = None
        self._path = None
        self._buffer_count = 0
        self._buffer_limit = 25

    def open(self, session_data):
        self.base_dir.mkdir(parents=True, exist_ok=True)

        animal = session_data.meta.get("animal", "UNKNOWN")
        phase = session_data.meta.get("phase", "0")
        date = str(session_data.meta.get("date", "")).replace("/", "-")
        fname = f"{date}_{animal}_{phase}_{session_data.session_id}.jsonl"

        self._path = self.base_dir / fname

        self._fh = open(self._path, "a", encoding="utf-8")
        self._fh.write(json.dumps({"type": "header", **session_data.meta}) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

        return self._path

    def append(self, record, force_fsync=False):
        if self._fh is None:
            return

        self._fh.write(json.dumps(record) + "\n")
        self._buffer_count += 1

        if force_fsync or self._buffer_count >= self._buffer_limit:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._buffer_count = 0

    def finalize(self, session_data):
        payload = session_data.to_dict()
        final_path = self._path.with_suffix(".json")

        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        if self._fh is not None:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
            self._fh = None

        return self._path, final_path

    def emergency_dump(self, session_data):
        try:
            path = self.base_dir / f"EMERGENCY_{session_data.session_id}.json"

            with open(path, "w", encoding="utf-8") as f:
                json.dump(session_data.to_dict(), f, indent=2)

            return path
        except Exception:
            return None


def _get_easy(phase, trial_n, K):
    """Determine whether a trial should use the easy threshold.

    Args:
        phase: Integer training phase.
        trial_n: One-based trial number.
        K: Easy-trial spacing after the initial block.

    Returns:
        True when the trial should be easy, otherwise False.
    """
    if phase < 5:
        return True

    if trial_n <= 20:
        return ((trial_n - 1) % 5) == 0

    K = max(1, int(K))
    return ((trial_n - 21) % K) == 0


def _update_easy_rate(session_data, trial_stack):
    """Update the adaptive easy-trial spacing from recent hit count.

    Args:
        session_data: SessionData instance used to record the K change event.
        trial_stack: Recent trial outcomes used for calibration.

    Returns:
        Tuple of new K value, new calibration window N, and hit count.
    """
    n_hits = sum(1 for x in trial_stack if x == "hit")

    if n_hits < 10:
        K = 3
    elif n_hits == 10:
        K = 5
    else:
        K = 7

    N = 4 * K

    trial_stack.clear()
    session_data.add_evt(_get_ts(), f"setK {K}")

    return K, N, n_hits


def _is_early_exit(evt, index, end_ms, min_duration=20*60, min_trials=150):
    """
    Determine whether low recent trial rate should end the session early.

    Args:
        evt: Event dictionary containing timestamps and values.
        index: Current trial index.
        end_ms: Current trial end time in milliseconds since midnight.
        min_duration: Minimum elapsed session seconds before early exit.
        min_trials: Minimum trial count before early exit.

    Returns:
        True when early-exit criteria are met, otherwise False.
    """
    buf = getattr(_is_early_exit, '_buf', None)
    if buf is None:
        buf = deque(maxlen=11)
        setattr(_is_early_exit, '_buf', buf)
    
    new_xy = (None, None)

    t0_ms = None
    elapsed_s = None

    try:
        ts_list = evt.get('timestamps', []) if isinstance(evt, dict) else []
        vals_list = evt.get('values', []) if isinstance(evt, dict) else []

        for ts, val in zip(ts_list, vals_list):
            if val == "cue":
                t0_ms = _ts_to_ms(ts)
                break
    except Exception:
        t0_ms = None
    
    prev_t0 = getattr(_is_early_exit, '_t0_ms', None)
    curr_t0 = int(t0_ms) if t0_ms is not None else None

    if curr_t0 is not None and (prev_t0 is None or prev_t0 != curr_t0):
        setattr(_is_early_exit, '_t0_ms', curr_t0)

        buf = deque(maxlen=11)
        setattr(_is_early_exit, '_buf', buf)
    
    if t0_ms is not None:
        try:
            dt_ms = int(end_ms) - int(t0_ms)
            if dt_ms < 0:
                dt_ms += 24 * 3600 * 1000
            
            elapsed_s = max(0.0, dt_ms / 1000.0)

            x = max(0.0, dt_ms / 60000.0)
            y = int(index)

            new_xy = (x, y) if int(index) >= min_trials else (None, None)
        except Exception:
            new_xy = (None, None)
    
    buf.append(new_xy)

    exit_valid = not (index < min_trials
                      or t0_ms is None
                      or new_xy == (None, None)
                      or len(buf) < 11
                      or elapsed_s < float(min_duration))

    if not exit_valid:
        return False
    
    buf = [xy for xy in buf if None not in xy]
    if len(buf) < 11:
        return False
    
    rates = []
    prev_xy = None
    
    for curr_xy in buf[-11:]:
        if prev_xy is None:
            prev_xy = curr_xy
            continue

        x1, y1 = prev_xy
        x2, y2 = curr_xy

        dx = float(x2) - float(x1)
        dy = float(y2) - float(y1)

        rates.append(float('inf') if dx <= 0.0 else (dy / dx))
        prev_xy = curr_xy
    
    return sum(1 for r in rates if r < 4.0) >= 5

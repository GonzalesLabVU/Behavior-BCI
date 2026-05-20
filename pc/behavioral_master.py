import os
import sys
import warnings
import traceback

import keyboard
import math
import random
import time
import socket
import uuid
import json
from itertools import zip_longest
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue, Empty
from collections import deque
from threading import Thread, Event, Lock

import serial
import serial.tools.list_ports
from cursor_utils import BCI, ABORT_EVT
from redis_utils import add_entry, remove_entry
from TCPClient import PrairieClient

import gspread
from gspread.utils import rowcol_to_a1
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv

import html
import smtplib
from email.message import EmailMessage

import subprocess
import shutil
import tempfile


# ---------------------------
# BASIC CONFIG
# ---------------------------
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore",
                        category=UserWarning,
                        message="pkg_resources is deprecated as an API.*",
                        )

SCRIPT_DIR = Path.cwd()
ANIMAL_MAP_PATH = SCRIPT_DIR / "animal_map.json"
ERROR_LOG_PATH = SCRIPT_DIR / "errors.log"

load_dotenv(SCRIPT_DIR / ".env")

BAUDRATE = 1_000_000
EARLY_STRING = "E"

PHASE_CONFIG = {
    '2': {'threshold': 15.0, 'side': 'B', 'reverse': False}, # wheel association
    '3': {'threshold': 15.0, 'side': 'B', 'reverse': False}, # tone association
    '4': {'threshold': 15.0, 'side': 'L', 'reverse': False}, # easy wheel
    '5': {'threshold': 30.0, 'side': 'L', 'reverse': False}, # normal wheel
    '6': {'threshold': 60.0, 'side': 'L', 'reverse': False}, # harder wheel
    '7': {'threshold': 90.0, 'side': 'L', 'reverse': False}, # hardest wheel
    }

MAX_STREAK = 4
LAST_SIDE = None
SIDE_STREAK = 0

EVT_QUEUE: "Queue[tuple[str, str]]" = Queue()
ENC_QUEUE: "Queue[tuple[str, object]]" = Queue()
EXC_STACK: "deque[dict[str, object]]" = deque()


# ---------------------------
# INTERFACE OBJECTS
# ---------------------------
class InterfaceObject:
    """Small base class for hardware/API/operator boundaries used by
    the training system"""
    interface_name = "generic"

    @property
    def ready(self):
        """Return whether this interface is ready for use."""
        return True


class SystemInterface(InterfaceObject):
    interface_name = "environment"

    def __init__(self, script_dir=SCRIPT_DIR):
        """Initialize paths and load environment settings for the script.

        Args:
            script_dir: Directory containing runtime assets and the .env file.
        """
        self.script_dir = Path(script_dir)
        self.animal_map_path = self.script_dir / "animal_map.json"
        self.credentials_path = self.script_dir / "credentials.json"
        self.env_path = self.script_dir / ".env"

        load_dotenv(self.env_path)

    @property
    def animal_map(self):
        with open(self.animal_map_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("animal_map.json must be a dict")

        for k, v in data.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError("animal_map.json keys and values must be strings")

        return data

    def require(self, name):
        """Read a required environment variable.

        Args:
            name: Environment variable name.

        Returns:
            The configured environment value.
        """
        v = os.getenv(name)
        if not v:
            raise RuntimeError(f"{name} not found in .env file")

        return v

    def _env_float(self, name):
        """Read a required environment variable as a float.

        Args:
            name: Environment variable name.

        Returns:
            Parsed floating-point value.
        """
        v = self.require(name).strip()

        if (len(v) >= 2) and (v[0] == v[-1]) and v[0] in {"'", '"'}:
            v = v[1:-1].strip()

        return float(v)

    def validate_assets(self):
        """Verify that required local asset files are present.

        Returns:
            True when validation succeeds.
        """
        if not self.animal_map_path.exists():
            raise FileNotFoundError("animal_map.json not found in the script directory")
        
        if not self.credentials_path.exists():
            raise FileNotFoundError("credentials.json not found in the script directory")

        return True

    @staticmethod
    def _cohort_tokens(map_key):
        return [t.strip() for t in str(map_key).split("_") if t.strip()]

    def animal_exists(self, animal_id, animal_map=None):
        """
        Check whether an animal ID exists in the animal map.

        Args:
            animal_id: Animal identifier to validate.
            animal_map: Optional preloaded animal map.

        Returns:
            True when the animal is present, otherwise False.
        """
        animal_id = str(animal_id).strip()
        animal_map = animal_map or self.animal_map

        return any(animal_id in self._cohort_tokens(key)
                   for key in animal_map.keys())

    def get_workbook_id(self, animal_id, animal_map=None):
        """
        Resolve the Google Sheets workbook ID for an animal.

        Args:
            animal_id: Animal identifier to resolve.
            animal_map: Optional preloaded animal map.

        Returns:
            Workbook ID string, or None if the DEV animal.
        """
        animal_id = str(animal_id).strip()
        animal_map = animal_map or self.animal_map

        if animal_id.upper() == "DEV":
            return None

        self.validate_assets()

        try:
            map_key = next(key for key in animal_map.keys()
                           if animal_id in self._cohort_tokens(key))
        except StopIteration:
            raise ValueError(f"No cohort assigned for animal {animal_id!r}")

        cohort_name = animal_map[map_key]
        workbook_id = f"{cohort_name}_ID"

        return self.require(workbook_id)

    def _find_arduino_port(self):
        ports = serial.tools.list_ports.comports()

        for port in ports:
            dsc = (port.description or "").lower()
            if "arduino" in dsc or "usb serial" in dsc:
                return port.device

        return None

    def serial_connect(self):
        port = self._find_arduino_port()
        if not port:
            print("\n[WARNING] No Arduino detected (continuing anyway)", flush=True)
            return None, False

        try:
            ser = serial.Serial(port, BAUDRATE, timeout=0.05)
            time.sleep(2)

            if not ser.is_open:
                print(f"\n[WARNING] {port} port is not open after initialization (continuing anyway)", flush=True)
                return None, False

            print(f"\nConnected to {port} port\n", flush=True)
            return ser, True
        except Exception as e:
            print(f"\n[WARNING] Could not open Arduino port: {e}", flush=True)
            return None, False

    def get_arduino(self, ser):
        link = ArduinoLink(ser)

        try:
            link.start()
        except Exception:
            pass

        return link

    def get_config(self, phase_id):
        cfg = PHASE_CONFIG.get(str(phase_id))
        if cfg is None and str(phase_id) not in {"0", "1"}:
            raise ValueError(f"No PHASE_CONFIG entry for phase {phase_id}")
        
        return {
            "cfg": cfg,
            "engage_ms": self._env_float("BRAKE_ENGAGE_MS"),
            "release_ms": self._env_float("BRAKE_RELEASE_MS"),
            "pulse_ms": self._env_float("SPOUT_PULSE_MS"),
            "threshold": float(cfg.get('threshold', 0.0)) if cfg else 0.0,
            "side": str(cfg.get('side', 'B')).upper() if cfg else "B",
            "reverse": bool(cfg.get('reverse', False)) if cfg else False,
            }

    @property
    def client_id(self):
        """Return the CLIENT_ID environment value as a string."""
        return str(os.getenv("CLIENT_ID"))


class ConsoleInterface(InterfaceObject):
    interface_name = "console"

    INFO_CFG = {
        "labels": ["TRIAL", "ELAPSED", "SUCCESS", "FAILURE", "RATE"],
        "label_pads": (5, 5, 4, 4, 5),
        "value_pads": (6, 5, None, None, 3)
        }
    BLOCK = "\u2588"

    def clear(self):
        """Clear the console window."""
        _cmd_run('cls')

    def line(self, text="", **kwargs):
        """Print a line of text to the console.

        Args:
            text: Text to print.
            **kwargs: Additional keyword arguments passed to print().
        """
        print(text, **kwargs)

    def warning(self, text):
        """Print a formatted warning message.

        Args:
            text: Warning text to display.
        """
        print(f'[WARNING] {text}', flush=True)

    def show_start(self):
        """Display the session start time."""
        print(f'\nSession started at {datetime.now().strftime("%I:%M %p")}\n', flush=True)

    def show_header(self):
        """Display the trial status table header."""
        labels = self.INFO_CFG["labels"]
        label_pads = self.INFO_CFG["label_pads"]

        cells = [f'{" " * sz}{txt}{" " * sz}'
                 for txt, sz in zip(labels, label_pads)]
        header = "|".join(cells)
        hline = "|".join("—" * len(cell) for cell in cells)

        print(header)
        print(hline)

    def show_trial_info(self, dt, n_hit, n_miss, outcome):
        """Display a single row of trial progress information.

        Args:
            dt: Trial elapsed time in seconds.
            n_hit: Number of recent hit outcomes.
            n_miss: Number of recent miss outcomes.
            outcome: Outcome label for the current trial.
        """
        labels = self.INFO_CFG["labels"]
        label_pads = self.INFO_CFG["label_pads"]
        value_pads = self.INFO_CFG["value_pads"]

        n_total = n_hit + n_miss

        trial_str = n_total
        elapsed_str = f"{(dt - 1.5):.2f} s"

        col_w = [len(label) + (2 * pad)
                 for label, pad in zip(labels, label_pads)]
    
        success_w = col_w[labels.index("SUCCESS")]
        failure_w = col_w[labels.index("FAILURE")]

        success_str = (self.BLOCK * success_w if outcome == "hit"
                       else " " * success_w)
        failure_str = (self.BLOCK * failure_w if outcome == "miss"
                       else " " * failure_w)

        rate = 100.0 * (n_hit / n_total) if n_total else 0.0
        rate_str = f"{rate:.1f} %"

        all_str = [trial_str, elapsed_str, success_str, failure_str, rate_str]
        values = {label: s for label, s in zip(labels, all_str)}

        cells = []
        for i, label in enumerate(labels):
            val = str(values[label])
            total_w = col_w[i]
            rpad = value_pads[i]

            if rpad is None:
                cells.append(val)
                continue

            free_w = max(0, total_w - rpad - len(val))
            cell = (" " * free_w) + val + (" " * rpad)
            cells.append(cell)

        print("|".join(cells))

    def show_summary(self, session_data):
        """Display a brief session duration summary.

        Args:
            session_data: SessionData instance containing metadata.
        """
        if not session_data:
            return
        
        dur = session_data.meta.get('duration_sec')
        if dur is None:
            return
        
        m, s = divmod(int(max(0, dur)), 60)
        print(f"\nSession duration: {m}:{s:02d}\n", flush=True)

    def show_exceptions(self):
        hline = 100 * "—"

        if not EXC_STACK:
            _cmd_run("echo.")
            print(f"{hline}\n")
            print(f"{hline}\n")
            print("[Process exited with code 0]")
            return

        print(hline + "\nEXCEPTION STACK (in order of occurrence):\n" + hline, flush=True)

        for i, info in enumerate(EXC_STACK, start=1):
            print(f"\n[{i}] {info['type']} in {info['caller']}:", flush=True)

            exc = info['exc']
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

            print(f"\n{tb}", flush=True)
            print(hline, flush=True)

        print("\n[Process exited with code 1]\n")

    def wait_for_key(self):
        """Wait for one keyboard press before exiting."""
        print('\nPress any key to continue . . .', end="", flush=True)
        time.sleep(0.25)
        keyboard.read_key()
        _cmd_run('echo.', 'echo.')


class TrainerInterface(InterfaceObject):
    interface_name = "trainer"

    def __init__(self, system=None):
        """Initialize trainer prompts with an environment interface.

        Args:
            env: Optional SystemInterface instance.
        """
        self.system = system or SystemInterface()

    def prompt_flush(self):
        """Prompt whether to flush the spout before the session.

        Returns:
            True when the trainer confirms flushing, otherwise False.
        """
        flush_raw = input("\nFlush spout for 5 seconds? [y/N]:  ")
        flush_choice = _is_affirmative(flush_raw)

        if flush_choice:
            flush_raw = input("This operation will restart the program. Continue? [y/N]:  ")
            return _is_affirmative(flush_raw)

        return flush_choice

    def prompt_animal(self):
        """
        Prompt for and validate the animal ID.

        Returns:
            Tuple of the selected animal ID and loaded animal map.
        """
        animal_map = self.system.animal_map

        while True:
            print("\nAnimal ID:  ", end="", flush=True)

            animal_raw = sys.stdin.readline()
            if animal_raw == "":
                raise EOFError
            
            animal_raw = animal_raw.rstrip("\n").upper()

            if not animal_raw:
                sys.stdout.write('\x1b[1A')
                sys.stdout.write('\x1b[2K')
                sys.stdout.write('Animal ID:  DEV\n')
                sys.stdout.flush()
                animal_id = "DEV"
            else:
                if not self.system.animal_exists(animal_raw, animal_map):
                    print('Please enter a valid animal')
                    continue

                animal_id = animal_raw

            return animal_id, animal_map

    def prompt_phase(self):
        """Prompt for a valid training phase.

        Returns:
            Selected phase ID string.
        """
        valid_phases = _valid_phases()

        while True:
            phase_id = input('Training Phase:  ').strip()
            if phase_id in valid_phases:
                return phase_id

            print('Please enter a valid phase\n', flush=True)

    def prompt_imaging(self):
        """Prompt whether imaging is active.

        Returns:
            True when imaging is active, otherwise False.
        """
        imaging_raw = input('\nImaging active? [y/N]:  ')
        return _is_affirmative(imaging_raw)

    def prompt_ephys(self):
        """Prompt whether electrophysiology recording is active.

        Returns:
            True when ephys is active, otherwise False.
        """
        ephys_raw = input("Ephys active? [y/N]:  ")
        return _is_affirmative(ephys_raw)

    def confirm_meta(self, session_data):
        animal_map = self.system.animal_map

        while True:
            raw = input("Enter the correct animal/phase to use for this session:  ").strip().upper()
            if not raw:
                continue

            if raw.isdigit():
                if raw in _valid_phases():
                    session_data.meta['phase'] = raw
                    return True

                print("Please enter a valid phase")
                continue

            if raw.isalnum():
                if self.system.animal_exists(raw, animal_map):
                    session_data.meta['animal'] = raw
                    session_data.meta['workbook_id'] = self.system.get_workbook_id(raw, animal_map)
                    return True

                print("Please enter a valid animal")
                continue

            print("Please enter a valid animal or phase")

    def confirm_save(self):
        """Prompt whether to save the completed session.

        Returns:
            True when the trainer accepts saving, otherwise False.
        """
        save_choice = input('\nSave current session? [Y/n]:  ').strip().lower()
        _cmd_run('echo.')

        return save_choice in {"", "y", "yes"}


class RedisInterface(InterfaceObject):
    interface_name = "redis"

    def __init__(self, client_id=None):
        """Initialize Redis status publishing for a client.

        Args:
            client_id: Optional client identifier; defaults to CLIENT_ID.
        """
        self.client_id = str(client_id or os.getenv("CLIENT_ID"))

    @staticmethod
    def _utc_iso():
        """Return the current UTC time as an ISO-8601 string."""
        return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')

    def _write(self, key, value):
        """Replace a Redis-style key with a string value.

        Args:
            key: Key to update.
            value: Value to write.
        """
        remove_entry(key)
        add_entry(key, str(value))

    def session_key(self, field):
        """Build a Redis session key for a field.

        Args:
            field: Session field name.

        Returns:
            Fully qualified Redis key.
        """
        return f"client:{self.client_id}:session:{field}"

    def clear_session(self):
        """Remove active session metadata keys for this client."""
        for field in ("animal", "phase", "start_utc", "stop_utc"):
            remove_entry(self.session_key(field))

    def notify_start(self, session_data):
        """Publish session start metadata and running state.

        Args:
            session_data: SessionData instance for the active run.
        """
        start_utc = self._utc_iso()

        self.clear_session()

        self._write(self.session_key("animal"), session_data.meta.get("animal", ""))
        self._write(self.session_key("phase"), session_data.meta.get("phase", ""))
        self._write(self.session_key("start_utc"), start_utc)

        self._write(f"client:{self.client_id}:state", "running")

    def notify_finish(self):
        """Publish session finish state and defer idle cleanup until key press."""
        self._write(self.session_key("stop_utc"), self._utc_iso())
        self._write(f"client:{self.client_id}:state", "finished")

        original_read_key = keyboard.read_key

        def _read_key_and_set_idle(*args, **kwargs):
            """Wrap keyboard.read_key to set Redis state idle after the key press.

            Args:
                *args: Positional arguments forwarded to keyboard.read_key.
                **kwargs: Keyword arguments forwarded to keyboard.read_key.

            Returns:
                The original keyboard.read_key return value.
            """
            try:
                return original_read_key(*args, **kwargs)
            finally:
                self._write(f"client:{self.client_id}:state", "idle")
                self.clear_session()
                keyboard.read_key = original_read_key
        
        keyboard.read_key = _read_key_and_set_idle


class ExceptionInterface(InterfaceObject):
    interface_name = "exception"

    def __init__(self, animal_id="UNKNOWN", phase_id="0"):
        """Initialize exception logging context.

        Args:
            animal_id: Animal identifier for log records.
            phase_id: Phase identifier for log records.
        """
        self.animal_id = animal_id
        self.phase_id = phase_id

    def set_session(self, animal_id, phase_id):
        """Update the animal and phase used for exception logs.

        Args:
            animal_id: Animal identifier for log records.
            phase_id: Phase identifier for log records.
        """
        self.animal_id = animal_id
        self.phase_id = phase_id

    def cache(self, exc, caller):
        """Cache an exception for console display.

        Args:
            exc: Exception instance to cache.
            caller: Name of the caller where the exception occurred.
        """
        EXC_STACK.append({
            "type": type(exc).__name__,
            "caller": caller,
            "exc": exc
            })

    def log(self, exc):
        """
        Append an exception to the local error log.

        Args:
            exc: Exception or error value to log.
        """
        global ERROR_LOGGED
        ERROR_LOGGED = True

        try:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")

            client = os.getenv("CLIENT_ID", "UNKNOWN_CLIENT")
            level = "UNKNOWN"

            if isinstance(exc, BaseException) and exc.__traceback__ is not None:
                tb = exc.__traceback__
                while tb.tb_next:
                    tb = tb.tb_next

                level = tb.tb_frame.f_code.co_name

            animal = str(self.animal_id)
            phase = str(self.phase_id)

            header = [
                f"TIMESTAMP={date_str} {time_str}",
                f"LEVEL={level}",
                f"CLIENT={client}",
                f"ANIMAL={animal}",
                f"PHASE={phase}"
                ]
            hline = ["-" * 40]
            body = []

            if isinstance(exc, BaseException):
                tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)

                for line in "".join(tb_lines).rstrip('\n').splitlines():
                    body.append(f"  {line}")
            else:
                body.append(f"  {type(exc).__name__}: {exc!r}")

            with open(ERROR_LOG_PATH, 'a', encoding='utf-8') as f:
                for line in hline + header + hline + body:
                    f.write(line + '\n')

                f.write('\n')
        except Exception:
            pass

    def commit(self):
        """
        Commit the local error log to the remote repository.

        Returns:
            True when a commit and push occurred, otherwise False.
        """
        global ERROR_LOGGED
        global LOG_COMMIT_FAIL

        if not ERROR_LOGGED:
            return False

        token = os.getenv("GITHUB_TOKEN")
        if not token:
            print("[WARNING] GITHUB_TOKEN not set (skipping errors.log push)", flush=True)
            return False

        if not ERROR_LOG_PATH.exists():
            return False

        remote_url = f"https://x-access-token:{token}@github.com/{REPO_SLUG}.git"

        try:
            with tempfile.TemporaryDirectory(prefix='behavior_bci_repo_') as td:
                repo_dir = Path(td) / "repo"

                _git_run(['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, remote_url, str(repo_dir)])

                dest_path = repo_dir / REPO_REL_PATH
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(ERROR_LOG_PATH, dest_path)

                st = _git_run(['git', 'status', '--porcelain', str(REPO_REL_PATH)], cwd=repo_dir).stdout.strip()
                if not st:
                    return False

                _git_run(['git', 'config', 'user.name', 'behavior-bci-bot'], cwd=repo_dir)
                _git_run(['git', 'config', 'user.email', 'behavior-bci-bot@users.noreply.github.com'], cwd=repo_dir)

                _git_run(['git', 'add', str(REPO_REL_PATH)], cwd=repo_dir)

                msg = f"Update errors.log (animal={self.animal_id}, phase={self.phase_id})"
                c = subprocess.run(
                    ['git', 'commit', '-m', msg],
                    cwd=repo_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                    )

                if c.returncode != 0:
                    return False

                _git_run(['git', 'push', 'origin', REPO_BRANCH], cwd=repo_dir, check=True)

                return True
        except Exception as e:
            if not LOG_COMMIT_FAIL:
                LOG_COMMIT_FAIL = True
                print(f"[WARNING] Failed to commit errors.log: {type(e).__name__}", flush=True)

            return False

    def log_and_commit(self, exc):
        """Log an exception and attempt to commit the error log.

        Args:
            exc: Exception or error value to log.

        Returns:
            Result of the commit attempt when applicable.
        """
        if isinstance(exc, KeyboardInterrupt):
            return
        
        try:
            self.log(exc)
        finally:
            try:
                self.commit()
            except Exception:
                pass


class EmailInterface(InterfaceObject):
    interface_name = "email"

    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587

    def send_session_summary(self, session_data):
        """Email a summary of the completed session.

        Args:
            session_data: SessionData instance containing event and timing data.
        """
        def format_subject(animal, phase):
            """Build the email subject from animal and phase identifiers.

            Args:
                animal: Animal identifier.
                phase: Training phase identifier.

            Returns:
                Formatted subject string.
            """
            animal_str = f'Animal {animal}'
            phase_str = f'Phase {phase}'
            return f'{animal_str}  |  {phase_str}'

        def format_body(date, t_start, t_stop, dur_s, evt):
            """Build the plain-text session summary email body.

            Args:
                date: Session date string in MM/DD/YYYY format.
                t_start: Human-readable start time.
                t_stop: Human-readable stop time.
                dur_s: Session duration in seconds.
                evt: Event data dictionary with values.

            Returns:
                Formatted email body string.
            """
            date_str = datetime.strptime(date, '%m/%d/%Y').strftime('%b-%d')

            m, s = divmod(int(dur_s or 0), 60)
            t_elapsed = f"{m}m {s}s"

            n_hits = sum(1 for e in evt['values'] if e == 'hit')
            n_total = sum(1 for e in evt['values'] if e == 'cue')
            hit_rate = ((n_hits / n_total) * 100) if n_total else 0.0

            lines = [
                ("Date", date_str),
                ("", ""),
                ("Started", str(t_start)),
                ("Finished", str(t_stop)),
                ("Duration", str(t_elapsed)),
                ("", ""),
                ("Total Trials", str(n_total)),
                ("Success Rate", f"{hit_rate:.1f}%"),
            ]

            out = []
            for label, value in lines:
                if not label and not value:
                    out.append("")
                else:
                    out.append(f"{label:<13}{value:>13}")

            return "\n".join(out)

        def ms_to_12h(ms):
            """Convert milliseconds since midnight to a 12-hour clock string.

            Args:
                ms: Milliseconds since midnight.

            Returns:
                Time string formatted as H:MM AM/PM.
            """
            ms = int(ms)
            total_s = ms // 1000
            h24 = (total_s // 3600) % 24
            m = (total_s % 3600) // 60

            am_pm = "AM" if h24 < 12 else "PM"
            h12 = h24 % 12
            if h12 == 0:
                h12 = 12

            return f'{h12}:{m:02d} {am_pm}'

        smtp_username = SystemInterface().require("SMTP_USERNAME")
        smtp_password = SystemInterface().require("SMTP_PASSWORD")
        smtp_to_addr = SystemInterface().require("SMTP_TO_ADDR")

        date = session_data.meta['date']
        animal = session_data.meta['animal']
        phase = session_data.meta['phase']

        subject = format_subject(animal, phase)

        start_ms = session_data.meta.get('t_start')
        stop_ms = session_data.meta.get('t_stop')

        t_start = ms_to_12h(start_ms) if start_ms is not None else "?"
        t_stop = ms_to_12h(stop_ms) if stop_ms is not None else "?"
        dur_s = session_data.meta.get('duration_sec', 0)
        evt = session_data.evt

        body = format_body(date, t_start, t_stop, dur_s, evt)

        try:
            recipients = json.loads(smtp_to_addr)
            if isinstance(recipients, str):
                recipients = [recipients]
        except Exception:
            recipients = [r.strip() for r in smtp_to_addr.split(",") if r.strip()]

        to_addr = ", ".join(recipients)

        msg = EmailMessage()
        msg['From'] = smtp_username
        msg['To'] = to_addr
        msg['Subject'] = subject

        msg.set_content(body)
        msg.add_alternative(
            f"<pre style=\"font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;\">"
            f"{html.escape(body)}"
            f"</pre>",
            subtype="html",
        )

        with smtplib.SMTP(self.SMTP_SERVER, self.SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)


class SaveInterface(InterfaceObject):
    interface_name = "save"

    VALID_SESSION_S = 5 * 60

    def _build_rows(self, session_data, dtype):
        """
        Build worksheet rows for metadata or imaging output.

        Args:
            session_data: SessionData instance.
            dtype: Row type to build. Expected values are "meta" or "img".

        Returns:
            List of two-column rows.
        """
        if dtype == "meta":
            session_data._ensure_session_tracking()

            client_id = str(os.getenv("CLIENT_ID"))

            cfg = session_data.meta.get('trial_config', []) or []
            easy_trials = [c['trial'] for c in cfg if c.get('is_easy') is True]
            normal_trials = [c['trial'] for c in cfg if c.get('is_easy') is False]

            left_targets = [c['trial'] for c in cfg if c.get('side') == "L"]
            right_targets = [c['trial'] for c in cfg if c.get('side') == "R"]
            both_targets = [c['trial'] for c in cfg if c.get('side') == "B"]

            meta_pairs = [
                ("client", client_id),
                ("imaging_active", session_data.meta.get('imaging_active', False)),
                ("ephys_active", session_data.meta.get('ephys_active', False)),
                ("K1", session_data.meta.get('K1', 5)),
                ("K2", session_data.meta.get('K2', None)),
                ("easy_trials", easy_trials),
                ("normal_trials", normal_trials),
                ("left_targets", left_targets),
                ("right_targets", right_targets),
                ("both_targets", both_targets)
                ]

            out = []
            for key, value in meta_pairs:
                if isinstance(value, (list, tuple)):
                    if len(value) == 0:
                        out.append([key, "None"])
                    else:
                        out.append([key, value[0]])

                        for v in value[1:]:
                            out.append(["", v])
                else:
                    out.append([key, "" if value is None else value])

            return out

        if dtype == "img":
            starts = session_data.img.get('start_ts') or []
            stops = session_data.img.get('stop_ts') or []

            out = []
            for t1, t2 in zip_longest(starts, stops, fillvalue=None):
                if t1 is not None:
                    out.append([str(t1), "start"])
                if t2 is not None:
                    out.append([str(t2), "stop"])

            return out

        raise ValueError(f"Unsupported row type: {dtype!r}")

    def _align_cells(self, wb, ws, r1, c1, r2, c2):
        sheet_id = ws._properties["sheetId"]
        req = {
            "requests": [{
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": r1 - 1,
                        "endRowIndex": r2,
                        "startColumnIndex": c1 - 1,
                        "endColumnIndex": c2
                        },
                    "cell": {
                        "userEnteredFormat": {
                            "horizontalAlignment": "LEFT"
                            }
                        },
                    "fields": "userEnteredFormat.horizontalAlignment"
                    }
                }]
            }
        
        wb.batch_update(req)

    def resolve_protocol(self, session_data):
        """
        Resolve duplicate-session handling before saving.

        Args:
            session_data: SessionData instance to save.

        Returns:
            True when saving should continue, otherwise False.
        """
        def _norm(x):
            return (x or "").strip()
        
        def _get_existing_duration(ws, start_col):
            try:
                vals = ws.get(f"{rowcol_to_a1(4, start_col)}:{rowcol_to_a1(ws.row_count, start_col + 1)}")
            except Exception:
                return None

            pending_key = None

            for row in vals:
                key = str(row[0]).strip() if len(row) > 0 else ""
                value = row[1] if len(row) > 1 else ""

                if key:
                    pending_key = key

                if pending_key == "duration_sec":
                    try:
                        return float(value)
                    except Exception:
                        return None

            return None
        
        def _find_existing_block(wb):
            target_date = _norm(session_data.meta.get('date', ""))
            target_animal = _norm(f"Animal {session_data.meta.get('animal', '')}")
            target_phase = _norm(f"Phase {session_data.meta.get('phase', '')}")

            try:
                ws = wb.worksheet("Metadata")
            except Exception:
                return None

            max_col = len(ws.row_values(2))
            if max_col <= 0:
                return None

            header_rng = f"A1:{rowcol_to_a1(2, max_col)}"
            header = ws.get(header_rng)

            row1 = header[0] if len(header) > 0 else []
            row2 = header[1] if len(header) > 1 else []

            for c in range(1, max_col + 1, 2):
                date_val = _norm(row1[c-1] if (c - 1) < len(row1) else "")
                animal_val = _norm(row2[c-1] if (c - 1) < len(row2) else "")
                phase_val = _norm(row2[c] if c < len(row2) else "")

                if (date_val == target_date) and (animal_val == target_animal) and (phase_val == target_phase):
                    return {
                        "worksheet": ws,
                        "start_col": c,
                        "duration_sec": _get_existing_duration(ws, c)
                        }

            return None

        session_data.overwrite_confirmed = False

        while True:
            workbook_id = session_data.meta.get('workbook_id')
            if not workbook_id:
                session_data.overwrite_confirmed = True
                return True

            wb = API_CLIENT.open_by_key(workbook_id)
            existing = _find_existing_block(wb)
            if existing is None:
                session_data.overwrite_confirmed = True
                return True

            prev_duration = existing.get('duration_sec')
            curr_duration = float(session_data.meta.get('duration_sec') or 0)

            auto_overwrite = (prev_duration is not None
                              and prev_duration < self.VALID_SESSION_S
                              and curr_duration > self.VALID_SESSION_S)
            if auto_overwrite:
                session_data.overwrite_confirmed = True
                return True

            overwrite_raw = input("A training session has already been recorded for this animal/phase today.\n"
                                  "Do you want to overwrite the earlier session with this session's data? [y/N]:  ")
            if _is_affirmative(overwrite_raw):
                session_data.overwrite_confirmed = True
                return True

            exit_raw = input("Exit this session without saving? [y/N]:  ")
            if _is_affirmative(exit_raw):
                session_data.overwrite_confirmed = False
                return False

            TrainerInterface().confirm_meta(session_data)

    def save_data(self, session_data):
        """
        Save session data into the configured Google Sheets workbook.

        Args:
            session_data: SessionData instance to write.

        Returns:
            True when saving completes, or None when no workbook is configured.
        """
        workbook_id = session_data.meta.get("workbook_id")

        if not workbook_id:
            print('[WARNING] No data recorded (skipping save)')
            return
        
        client_id = FileLock._get_client_id()

        def _norm(x):
            """Normalize a header value for comparison.

            Args:
                x: Value to normalize.

            Returns:
                Stripped string value.
            """
            return (x or "").strip()
        
        def _target_headers():
            """Build the date, animal, and phase headers for the session.

            Returns:
                Tuple of normalized date, animal, and phase header strings.
            """
            d = _norm(session_data.meta.get("date", ""))
            a = _norm(f"Animal {session_data.meta.get('animal', '')}")
            p = _norm(f"Phase {session_data.meta.get('phase', '')}")

            return d, a, p
        
        def _find_cols(ws):
            """Find or allocate the two-column block for this session.

            Args:
                ws: Worksheet to inspect.

            Returns:
                Tuple of starting column and whether it overwrites existing data.
            """
            target_d, target_a, target_p = _target_headers()

            max_col = len(ws.row_values(2))
            if max_col <= 0:
                return 1, False
            
            header_rng = f'A1:{rowcol_to_a1(2, max_col)}'
            header = ws.get(header_rng)
            row1 = header[0] if len(header) > 0 else []
            row2 = header[1] if len(header) > 1 else []

            for c in range(1, max_col + 1, 2):
                d_val = _norm(row1[c-1] if (c - 1) < len(row1) else "")
                a_val = _norm(row2[c-1] if (c-1) < len(row2) else "")
                p_val = _norm(row2[c] if c < len(row2) else "")

                if (d_val == target_d) and (a_val == target_a) and (p_val == target_p):
                    return c, True
            
            new_col = (((max_col + 1) // 2) * 2) + 1
            return new_col, False

        def _batch_write_cols(ws, start_row, start_col, data, chunk_rows=2000, group_chunks=10):
            """Write two-column data to a worksheet in grouped batches.

            Args:
                ws: Worksheet to update.
                start_row: Starting row, one-based.
                start_col: Starting column, one-based.
                data: Rows to write.
                chunk_rows: Maximum rows per value range.
                group_chunks: Maximum ranges per batch update request.
            """
            sheet = ws.spreadsheet
            name = ws.title

            def _rng(r1, c1, r2, c2):
                """Build an A1 range for a worksheet rectangle.

                Args:
                    r1: Starting row, one-based.
                    c1: Starting column, one-based.
                    r2: Ending row, one-based.
                    c2: Ending column, one-based.

                Returns:
                    A1 notation range string.
                """
                return f'{name}!{rowcol_to_a1(r1, c1)}:{rowcol_to_a1(r2, c2)}'
            
            req = []
            n = len(data)

            for i in range(0, n, chunk_rows):
                chunk = data[i:i+chunk_rows]
                r1 = start_row + i
                r2 = r1 + len(chunk) - 1
                c1 = start_col
                c2 = start_col + 1

                req.append({'range': _rng(r1, c1, r2, c2), 'values': chunk})

                if len(req) >= group_chunks:
                    sheet.values_batch_update(body={'valueInputOption': 'RAW', 'data': req})
                    req.clear()
            
            if req:
                sheet.values_batch_update(body={'valueInputOption': 'RAW', 'data': req})

        lock = None

        try:
            lock = FileLock(workbook_id, owner=client_id).acquire()

            wb = API_CLIENT.open_by_key(workbook_id)
            lock.wb = wb

            sheet_map = (
                ("evt", "Event"),
                ("enc", "Encoder"),
                ("img", "Imaging"),
                ("meta", "Metadata")
                )

            for dtype, sheet_name in sheet_map:
                match dtype:
                    case "meta":
                        data_rows = self._build_rows(session_data, "meta")
                        data = data_rows
                        n_rows = len(data_rows)
                        label = "metadata"
                    case "img":
                        data_rows = self._build_rows(session_data, "img")
                        data = data_rows
                        n_rows = len(data_rows)
                        label = 'imaging'
                    case _:
                        d = getattr(session_data, dtype)
                        n_rows = len(d['timestamps'])
                        data = [[ts, val] for ts, val in zip(d['timestamps'], d['values'])]
                        label = sheet_name.lower()

                if n_rows == 0:
                    continue

                if dtype != 'meta':
                    print(f'Writing {label} data...', flush=True)
                else:
                    print('Writing metadata...', flush=True)

                lock.update()
                lock.reset()

                try:
                    ws = wb.worksheet(sheet_name)
                except Exception:
                    ws = wb.add_worksheet(title=sheet_name, rows=200, cols=26)
                
                lock.update()
                lock.reset()

                start_col, overwrite = _find_cols(ws)
                if overwrite and not getattr(session_data, "overwrite_confirmed", False):
                    raise RuntimeError("Refusing to overwrite existing session data without save-protocol confirmation")

                needed_cols = start_col + 1

                if ws.col_count < needed_cols:
                    ws.add_cols(needed_cols - ws.col_count)

                if overwrite:
                    clear_rng = f'{rowcol_to_a1(1, start_col)}:{rowcol_to_a1(ws.row_count, start_col + 1)}'
                    
                    lock.update()
                    lock.reset()

                    ws.batch_clear([clear_rng])
                
                header_rng = f'{rowcol_to_a1(1, start_col)}:{rowcol_to_a1(2, start_col + 1)}'
                skip_rng = f'{rowcol_to_a1(3, start_col)}:{rowcol_to_a1(3, start_col + 1)}'

                header = [
                    [session_data.meta['date'], ""],
                    [f"Animal {session_data.meta['animal']}", f"Phase {session_data.meta['phase']}"]
                    ]
                
                lock.update()
                lock.reset()

                ws.batch_update([
                    {'range': header_rng, 'values': header},
                    {'range': skip_rng, 'values': [["", ""]]}
                    ])

                needed_rows = 3 + n_rows
                if ws.row_count < needed_rows:
                    ws.add_rows(needed_rows - ws.row_count)

                lock.update()
                lock.reset()

                _batch_write_cols(ws, start_row=4, start_col=start_col, data=data)

                if dtype == 'meta':
                    r1 = 1
                    r2 = 3 + n_rows
                    c1 = start_col
                    c2 = start_col + 1

                    lock.update()
                    lock.reset()

                    self._align_cells(wb, ws, r1, c1, r2, c2)

                print("\r\033[2K", end="", flush=True)
            
            return True
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception as e:
                    exc_proxy = ExceptionInterface(session_data.meta.get("animal", "UNKNOWN"),
                                                   session_data.meta.get("phase", "0"))
                    exc_proxy.log_and_commit(e)

    def save_raw(self, session_data):
        """Save raw capacitive sensor data to a local JSON file.

        Args:
            session_data: SessionData instance containing raw cap data.

        Returns:
            Path to the saved file, or None when no raw data is available.
        """
        if session_data is None:
            return None

        cap = session_data.raw.get('cap', {})
        ts_list = cap.get('timestamps', [])
        val_list = cap.get('values', [])

        if not ts_list:
            return None

        animal_id = str(session_data.meta.get("animal", "UNKNOWN"))
        animal_str = f'Animal={animal_id}'

        phase_id = str(session_data.meta.get("phase", "0"))
        phase_str = f'Phase={phase_id}'

        date_str = str(session_data.meta.get("date", "")).strip()
        try:
            mm_dd_yyyy = datetime.strptime(date_str, "%m/%d/%Y").strftime("%m-%d-%Y")
        except Exception:
            mm_dd_yyyy = datetime.now().strftime("%m-%d-%Y")

        out_name = f'raw_cap_{animal_str}_{phase_str}_{mm_dd_yyyy}.json'
        out_path = SCRIPT_DIR / out_name

        payload = {
            "meta": {
                "animal": animal_id,
                "phase": str(session_data.meta.get("phase", "")),
                "date": date_str,
            },
            "data": {
                "timestamps": ts_list,
                "values": val_list,
            },
        }

        with open(out_path, 'w', encoding='utf-8') as out_file:
            json.dump(payload, out_file, indent=4)

        print(f'Saved raw data locally to {out_path.name}', flush=True)
        return out_path

    def save_session(self, session_data):
        """
        Save session data to Google Sheets with local fallback on failure.

        Args:
            session_data: SessionData instance to persist.

        Returns:
            True when the primary save succeeds, otherwise False.
        """
        animal = session_data.meta.get('animal', 'UNKNOWN') if session_data else 'UNKNOWN'
        phase = session_data.meta.get('phase', '0') if session_data else '0'

        exc_proxy = ExceptionInterface(animal, phase)

        try:
            self.save_data(session_data)
            return True
        except Exception as e:
            try:
                self.fallback_save(session_data)
            except Exception as e2:
                exc_proxy.log(e2)

            exc_proxy.log(e)
            return False
        finally:
            try:
                exc_proxy.commit()
            except Exception:
                pass

    def fallback_save(self, session_data):
        """
        Save session data using the local fallback path.

        Args:
            session_data: SessionData instance to persist.

        Returns:
            Path to the local fallback file.
        """
        animal = str(session_data.meta.get('animal', 'UNKNOWN'))
        phase = str(session_data.meta.get('phase', '0'))
        date = str(session_data.meta.get('date', '0000-00-00')).replace('/', '.')
        rand = uuid.uuid4().hex[:6]

        out_path = SCRIPT_DIR / f"date={date}_animal={animal}_phase={phase}_id={rand}.json"
        payload = session_data.to_dict()

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=4)

        print("\r\033[2K", end="", flush=True)
        print(f"[WARNING] Saved session data locally to {out_path.name}", flush=True)

        return out_path


class PrairieInterface(InterfaceObject):
    interface_name = "prairie"

    def connect(self, imaging_active):
        """
        Connect to Prairie View when imaging is active.

        Args:
            imaging_active: Whether imaging should be initialized.

        Returns:
            PrairieClient instance, or None when imaging is inactive/unavailable.
        """
        if not imaging_active:
            return None

        try:
            client = PrairieClient()
        except Exception as e:
            print(f"\n[WARNING] Imaging requested, but Prairie View connection could not be established "
                  f"({type(e).__name__}: {e}). Continuing without imaging...",
                  flush=True)
            return None

        try:
            configured = client.configure()
        except Exception as e:
            print(f"\n[WARNING] Prairie View CONFIG failed "
                  f"({type(e).__name__}: {e}). Continuing without imaging...",
                  flush=True)
            return None

        if not configured:
            print("\n[WARNING] Prairie View CONFIG returned false. Continuing without imaging...",
                  flush=True)
            return None

        return client

    def finish(self, client, session_data):
        """Finish Prairie imaging and copy timestamps into session data.

        Args:
            client: PrairieClient instance or None.
            session_data: SessionData instance to receive imaging timestamps.
        """
        if client is None:
            return
        
        client.finish()

        if session_data is not None:
            session_data.img["start_ts"] = list(getattr(client, "start_ts", []) or [])
            session_data.img["stop_ts"] = list(getattr(client, "stop_ts", []) or [])


class CursorInterface(InterfaceObject):
    interface_name = "cursor"

    def connect(self, phase_id, side):
        """Start the cursor task for wheel phases.

        Args:
            phase_id: Training phase identifier.
            side: Target side configuration.

        Returns:
            Tuple of cursor instance or None, and the initial easy-trial flag.
        """
        if int(phase_id) <= 3:
            return None, True

        easy = _get_easy(phase=int(phase_id), trial_n=1, K=5)

        cursor = BCI(
            phase_id=phase_id,
            evt_queue=EVT_QUEUE,
            enc_queue=ENC_QUEUE,
            config=PHASE_CONFIG,
            display_idx=1,
            fullscreen=False,
            easy_threshold=15.0,
        )
        cursor.update_config(easy, side)
        cursor.start()

        return cursor, easy

    def update_trial(self, cursor, is_easy, side):
        """Update cursor task settings for the next trial.

        Args:
            cursor: Active cursor object or None.
            is_easy: Whether the next trial is easy.
            side: Target side for the next trial.
        """
        if cursor is not None:
            cursor.update_config(is_easy, side)

    def stop(self, cursor):
        """Stop the cursor task if it is running.

        Args:
            cursor: Active cursor object or None.

        Returns:
            Cursor stop result, or True when no cursor is active.
        """
        if cursor is not None:
            return cursor.stop()
        
        return True


class BehaviorInterfaces:
    def __init__(self):
        """
        Create the default runtime interfaces used by setup and main.
        """
        self.system = SystemInterface()
        self.user = TrainerInterface(self.system)
        self.console = ConsoleInterface()
        self.redis = RedisInterface()
        self.exceptions = ExceptionInterface()
        self.email = EmailInterface()
        self.saving = SaveInterface()
        self.prairie = PrairieInterface()
        self.cursor = CursorInterface()


# ---------------------------
# LOGGING
# ---------------------------
REPO_SLUG = "GonzalesLabVU/Behavior-BCI"
REPO_BRANCH = "main"
REPO_REL_PATH = Path("pc") / "config" / "errors.log"

ERROR_LOGGED = False
LOG_COMMIT_FAIL = False


# ---------------------------
# SERIAL INTERFACE
# ---------------------------
class ArduinoLink:
    EPHYS_START_STRING = "R1"
    EPHYS_STOP_STRING = "R2"
    FINISH_STRING = "S"
    RESTART_STRING = "R"
    ACK_STRING = "A"

    def __init__(self, ser, verbose=False):
        """
        Initialize the serial link wrapper and reader thread.

        Args:
            ser: Open pyserial Serial object, or None for inactive mode.
        """
        self.ser = ser
        self.verbose = bool(verbose)
        self.active = ser is not None and ser.is_open
        self.stop_evt = Event()
        self.ack_evt = Event()
        self.write_lock = Lock()
        self.msg_q = Queue()
        self._reader = Thread(target=self._reader_loop, daemon=True)

    def _reader_loop(self):
        try:
            while not self.stop_evt.is_set() and self.ser and self.ser.is_open:
                raw = self.ser.readline()
                if not raw:
                    continue

                try:
                    line = raw.decode('utf-8', errors='strict').strip()
                except UnicodeDecodeError:
                    line = raw.decode('latin1', errors='ignore').strip()

                if not line:
                    continue

                if self.verbose:
                    print(f"[RECV]  {line!r}", flush=True)

                if line == self.ACK_STRING:
                    self.ack_evt.set()
                    continue

                ts = _get_ts()

                if line == self.RESTART_STRING:
                    self.msg_q.put(("RESTART", ts, None))
                    continue

                if line == self.FINISH_STRING:
                    self.msg_q.put(("END", ts, None))
                    continue

                if line.startswith("[EVT]"):
                    payload = line.split("]", 1)[1].strip()
                    self.msg_q.put(("EVT", ts, payload))
                    continue

                if line.startswith("[ENC]"):
                    payload = line.split("]", 1)[1].strip()
                    self.msg_q.put(("ENC", ts, payload))
                    continue

                if line.startswith("[RAW]"):
                    payload = line.split("]", 1)[1].strip()
                    self.msg_q.put(("RAW", ts, payload))
                    continue

        except Exception as e:
            try:
                self.msg_q.put(("ERR", _get_ts(), e))
            except Exception:
                pass

    def send_config(self, phase_id, params):
        try:
            self.send_and_wait(f"engage {params['engage_ms']:.4f}")
            self.send_and_wait(f"release {params['release_ms']:.4f}")
            self.send_and_wait(f"pulse {params['pulse_ms']:.4f}")
            self.send_and_wait(f"threshold {params['threshold']:.4f}")
            self.send_and_wait(f"side {params['side']}")
            self.send_and_wait(f"reverse {'1' if params['reverse'] else '0'}")
            self.send_and_wait(f"phase {phase_id}")
        except Exception as e:
            self.close()
            raise RuntimeError(f"[ERROR] Failed during Arduino setup handshake: {e}") from e

    def send_ephys(self, ephys_active):
        try:
            self.send_and_wait(f"ephys {'1' if ephys_active else '0'}")
        except Exception as e:
            self.close()
            raise RuntimeError(f"[ERROR] Failed during Arduino ephys handshake: {e}") from e

    def send_flush(self, flush_active):
        try:
            self.send_and_wait(f"flush {'1' if flush_active else '0'}")
        except Exception as e:
            self.close()
            raise RuntimeError(f"[ERROR] Failed during Arduino flush handshake: {e}") from e

        if not flush_active:
            return True

        print()
        deadline = time.time() + 5.5

        while True:
            remaining = math.floor(deadline - time.time())
            if remaining > 0:
                print(f"\rFlushing...{remaining}s", end="", flush=True)

            try:
                typ, _, payload = self.msg_q.get(timeout=1.0)
            except Empty:
                continue

            if typ == "RESTART":
                print("\rFlushing...Done", flush=True)
                self.close()
                raise SystemExit(0)

            if typ == "ERR":
                if isinstance(payload, BaseException):
                    raise payload

                raise RuntimeError(f"ArduinoLink reader error during flush handshake: {payload!r}")

    def send_start(self):
        try:
            self.send_and_wait("start 1")
        except Exception as e:
            self.close()
            raise RuntimeError(f"[ERROR] Failed during Arduino start command handshake: {e}") from e

    def start(self):
        """Start the background serial reader when the link is active."""
        if self.active:
            self._reader.start()

    def close(self):
        """Stop the reader and close the serial port if open."""
        self.stop_evt.set()

        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

    def send_and_wait(self, text, timeout_s=5.0):
        if not self.active:
            return True
        
        if self.verbose:
            print(f"[SEND]  {text!r}", flush=True)

        with self.write_lock:
            self.ack_evt.clear()
            self.ser.write((str(text).strip() + "\n").encode('utf-8'))
            self.ser.flush()

            if not self.ack_evt.wait(timeout=float(timeout_s)):
                raise TimeoutError(f"No ACK after sending: {text!r}")

        return True

    def send(self, text):
        if not self.active:
            return True
        
        if self.verbose:
            print(f"[SEND]  {text!r}")

        with self.write_lock:
            self.ser.write((str(text).strip() + "\n").encode('utf-8'))
            self.ser.flush()

        return True

    def send_after(self, cmd, delay_s=0.0, timeout_s=5.0):
        """Schedule a command to be sent after an optional delay.

        Args:
            cmd: Command text to send.
            delay_s: Delay before sending in seconds.
            timeout_s: Maximum seconds to wait for ACK.

        Returns:
            True after the background send worker is started.
        """
        def worker():
            """Delay and send the scheduled command from a background thread."""
            if delay_s > 0:
                time.sleep(float(delay_s))

            self.send_and_wait(cmd, timeout_s=timeout_s)
        
        Thread(target=worker, daemon=True).start()
        return True

    def start_ephys(self, timeout_s=5.0):
        """Send the ephys start command.

        Args:
            timeout_s: Maximum seconds to wait for ACK.

        Returns:
            True when the command succeeds or the link is inactive.
        """
        return self.send_and_wait(self.EPHYS_START_STRING, timeout_s=timeout_s)

    def stop_ephys(self, session_data=None, timeout_s=5.0, safe=False):
        """
        Stop ephys recording.

        Args:
            session_data: Optional SessionData instance used for idempotent safe-stop tracking.
            timeout_s: Maximum seconds to wait for ACK.
            safe: If True, cache failures instead of raising and skip duplicate stops.

        Returns:
            True when stopped, skipped, or inactive.
        """
        if safe:
            if session_data is None:
                return True

            if not session_data.meta.get("ephys_active", False):
                return True

            if session_data.meta.get("_ephys_stopped", False):
                return True

            try:
                self.send_and_wait(self.EPHYS_STOP_STRING, timeout_s=timeout_s)
                session_data.meta['_ephys_stopped'] = True
            except Exception as e:
                ExceptionInterface().cache(e, "main.ephys_stop")

            return True

        return self.send_and_wait(self.EPHYS_STOP_STRING, timeout_s=timeout_s)

    def start_imaging(self, delay_s=0.0, timeout_s=5.0):
        """Send or schedule the imaging start TTL command.

        Args:
            delay_s: Delay before sending in seconds.
            timeout_s: Maximum seconds to wait for ACK.

        Returns:
            True after the command is scheduled.
        """
        return self.send_after("img_start", delay_s=delay_s, timeout_s=timeout_s)

    def stop_imaging(self, delay_s=0.0, timeout_s=5.0):
        """Send or schedule the imaging stop TTL command.

        Args:
            delay_s: Delay before sending in seconds.
            timeout_s: Maximum seconds to wait for ACK.

        Returns:
            True after the command is scheduled.
        """
        return self.send_after("img_stop", delay_s=delay_s, timeout_s=timeout_s)


class SessionData:
    def __init__(self, animal_id, phase_id, date_str):
        """
        Initialize containers for one behavioral session.

        Args:
            animal_id: Animal identifier for the session.
            phase_id: Training phase identifier.
            date_str: Session date string.
        """
        self.meta = {
            "client": None,
            "workbook_id": None,
            "date": date_str,
            "animal": animal_id,
            "phase": phase_id,
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
        """Convert the session data to JSON-safe dictionaries.

        Returns:
            Dictionary containing metadata, event, encoder, imaging, and raw data.
        """
        def _json_safe(x):
            """Recursively convert values to JSON-safe primitives.

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


# ---------------------------
# DATA SAVING
# ---------------------------
API_SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
API_CREDS = Credentials.from_service_account_file(str(SCRIPT_DIR / 'credentials.json'), scopes=API_SCOPES)
API_CLIENT = gspread.authorize(API_CREDS)
API_DRIVE = build('drive', 'v3', credentials=API_CREDS, cache_discovery=False)


class FileLock:
    POLL_S = 5.0
    RETRY_S = 5.0
    LEASE_S = 180
    RESET_S = 60
    TIMEOUT_S = 300

    TAG = "------ LOCK ------"
    TAG_RANGE = "A1"
    META_RANGE = "A2:D2"

    def __init__(self, workbook_id, owner):
        """Initialize a Google Sheets worksheet lock.

        Args:
            workbook_id: Google Sheets workbook ID to protect.
            owner: Unique owner string for this lock attempt.
        """
        self.poll_s = float(self.POLL_S)
        self.retry_s = float(self.RETRY_S)
        self.lease_s = int(self.LEASE_S)
        self.reset_s = int(self.RESET_S)
        self.timeout_s = int(self.TIMEOUT_S)

        self.client = API_CLIENT
        self.workbook_id = workbook_id
        self.owner = owner
        self.token = uuid.uuid4().hex

        self.sheet_name = None
        self.created = 0
        self.expires = 0

        self.wb = None
        self.ws = None

    def _confirm_ws(self, ws, err_msg='Lock lost'):
        """Confirm that a worksheet still represents this lock.

        Args:
            ws: Worksheet to validate.
            err_msg: Error message used if validation fails.
        """
        meta = self._get_resource("meta", ws=ws, err_msg=err_msg)
        self._ensure_control(meta['owner'], meta['token'], err_msg=err_msg)

    @staticmethod
    def _get_client_id():
        return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

    def _get_resource(self, resource="workbook", ws=None, err_msg = "Lock tag missing (lock lost)"):
        """Open or read lock-related Google Sheets resources.

        Args:
            resource: One of "workbook", "worksheet", or "meta".
            ws: Worksheet to read metadata from when resource is "meta".
            err_msg: Error message used when lock metadata is missing or invalid.

        Returns:
            Workbook, worksheet, or metadata dictionary depending on resource.
        """
        if resource == "workbook":
            self.wb = self.client.open_by_key(self.workbook_id)
            return self.wb
        
        if resource == "worksheet":
            if self.sheet_name is None:
                raise RuntimeError("Lock not acquired (sheet_name is None)")
            
            if self.wb is None:
                self._get_resource("workbook")

            try:
                self.ws = self.wb.worksheet(self.sheet_name)
            except Exception:
                self._get_resource("workbook")
                self.ws = self.wb.worksheet(self.sheet_name)

            return self.ws
        
        if resource == "meta":
            if ws is None:
                ws = self._get_resource("worksheet")

            try:
                vals = ws.get('A1:D2')
            except Exception as e:
                raise RuntimeError(err_msg) from e
            
            tag = (vals[0][0] if vals and vals[0] else "") if vals else ""
            if (tag or "") != self.TAG:
                raise RuntimeError(err_msg)
            
            row = vals[1] if len(vals) > 1 and vals[1] else ["", "", "0", "0"]
            owner = str(row[0] or "")
            token = str(row[1] or "")

            try:
                created_ts = int(str(row[2] or "0"))
            except Exception:
                created_ts = 0

            try:
                expires_ts = int(str(row[3] or "0"))
            except Exception:
                expires_ts = 0

            return {
                "tag": tag,
                "owner": owner,
                "token": token,
                "created": created_ts,
                "expires": expires_ts,
                "info": row
                }
        
        raise ValueError(f"Unknown lock resource: {resource!r}")

    def _ensure_control(self, owner, token, err_msg='Lock lost'):
        """Verify that the supplied owner and token match this lock.

        Args:
            owner: Owner string read from the lock sheet.
            token: Token string read from the lock sheet.
            err_msg: Error message used if ownership does not match.
        """
        if owner != self.owner or token != self.token:
            raise RuntimeError(err_msg)

    def sleep(self, dur_s, jitter_ms=1000):
        """Sleep with optional random jitter.

        Args:
            dur_s: Base sleep duration in seconds.
            jitter_ms: Maximum additional jitter in milliseconds.
        """
        if dur_s < 0:
            dur_s = 0.0
        
        if jitter_ms and jitter_ms > 0:
            dur_s += random.random() * (jitter_ms / 1000.0)
        
        time.sleep(dur_s)

    def acquire(self):
        """Acquire the workbook lock, waiting until this client owns it.

        Returns:
            This FileLock instance after acquisition.
        """
        wb = self._get_resource("workbook")

        deadline = time.monotonic() + self.timeout_s
        attempt = 0
        created_ts = _now()

        print('Acquiring lock...', end='\r', flush=True)

        def q_sheet(title):
            """Quote a worksheet title for a Sheets range.

            Args:
                title: Worksheet title.

            Returns:
                Quoted worksheet title.
            """
            return "'" + title.replace("'", "''") + "'"
        
        def to_int(x, default=0):
            """Convert a value to int with a fallback.

            Args:
                x: Value to convert.
                default: Value returned when conversion fails.

            Returns:
                Converted integer or default.
            """
            try:
                return int(x)
            except Exception:
                return default
            
        def scan_locks():
            """Scan workbook sheets for active lock records.

            Returns:
                List of lock metadata dictionaries.
            """
            meta = wb.fetch_sheet_metadata(params={'fields': 'sheets(properties(sheetId,title))'})
            sheets = meta.get('sheets', [])
            
            if not sheets:
                return []
            
            props = [s.get('properties', {}) for s in sheets]
            titles = [p.get('title', '') for p in props]
            ids = [p.get('sheetId', 0) for p in props]

            ranges = [f'{q_sheet(t)}!A1:D2' for t in titles]
            resp = wb.values_batch_get(ranges)
            vrs = resp.get('valueRanges', [])

            assert len(titles) == len(ids) == len(vrs)

            locks = []

            for title, id, vr in zip(titles, ids, vrs):
                values = vr.get('values', [])
                if not values or not values[0]:
                    continue

                tag = values[0][0] if values[0] else ""
                if (tag or "") != self.TAG:
                    continue

                row = values[1] if len(values) > 1 and values[1] else []
                owner = str(row[0]) if len(row) > 0 else ""
                token = str(row[1]) if len(row) > 1 else ""
                created = to_int(row[2], 0) if len(row) > 2 else 0
                expires = to_int(row[3], 0) if len(row) > 3 else 0

                locks.append({
                    'sheetId': id,
                    'title': title,
                    'owner': owner,
                    'token': token,
                    'created': created,
                    'expires': expires
                    })
            
            return locks

        def batch_delete(ids):
            """Delete worksheets by sheet ID, ignoring delete failures.

            Args:
                ids: Iterable of sheet IDs to delete.
            """
            if not ids:
                return
            
            req = [{'deleteSheet': {'sheetId': id}} for id in ids]

            try:
                wb.batch_update({'requests': req})
            except Exception:
                pass

        def is_mine(lock):
            """Check whether a scanned lock belongs to this FileLock.

            Args:
                lock: Lock metadata dictionary.

            Returns:
                True when owner and token match this instance.
            """
            return lock.get('owner') == self.owner and lock.get('token') == self.token

        while time.monotonic() < deadline:
            attempt += 1
            print(f'Acquiring lock...[TRIES={attempt}]', flush=True)

            now = _now()

            try:
                locks = scan_locks()
            except Exception:
                self.sleep(self.retry_s, jitter_ms=750)
                wb = self._get_resource("workbook")
                continue

            expired_ids = [lock['sheetId'] for lock in locks if lock['expires'] and now >= lock['expires']]
            if expired_ids:
                batch_delete(expired_ids)
                self.sleep(0.1, jitter_ms=100)
                continue

            active = [lock for lock in locks if lock['expires'] and now < lock['expires']]
            if active:
                winner = min(active, key=lambda lock: (lock['created'], lock['token'], lock['sheetId']))

                if is_mine(winner):
                    self.sheet_name = winner['title']
                    self.created = int(winner['created'] or created_ts)
                    self.expires = int(winner['expires'] or 0)
                    self.wb = wb
                    self.ws = None

                    print("\r\033[2KLock acquired", flush=True)
                    return self
                
                remaining = int(winner['expires'] or 0) - now
                sleep_s = self.poll_s if remaining > self.poll_s else max(0.2, float(remaining))

                self.sleep(sleep_s, jitter_ms=350)
                continue

            try:
                my_lock = wb.add_worksheet(title=self.owner, rows=10, cols=10)
            except Exception:
                self.sleep(self.poll_s, jitter_ms=750)
                wb = self._get_resource("workbook")
                continue

            try:
                expires_ts = _now() + self.lease_s
                my_meta = [self.owner, self.token, str(created_ts), str(expires_ts)]

                my_lock.batch_update([
                    {'range': self.TAG_RANGE, 'values': [[self.TAG]]},
                    {'range': self.META_RANGE, 'values': [my_meta]}
                    ])
            except Exception:
                try:
                    wb.del_worksheet(my_lock)
                except Exception:
                    pass

                self.sleep(self.poll_s, jitter_ms=750)
                wb = self._get_resource("workbook")
                continue

            try:
                locks2 = scan_locks()
            except Exception:
                self.sleep(self.poll_s, jitter_ms=750)
                continue

            now2 = _now()

            expired2 = [lock['sheetId'] for lock in locks2 if lock['expires'] and now2 >= lock['expires']]
            if expired2:
                batch_delete(expired2)
                continue

            active2 = [lock for lock in locks2 if lock['expires'] and now2 < lock['expires']]
            if not active2:
                self.sleep(0.2, jitter_ms=200)
                continue

            winner2 = min(active2, key=lambda lock: (lock['created'], lock['token'], lock['sheetId']))

            if is_mine(winner2):
                self.sheet_name = winner2['title']
                self.created = int(winner2['created'] or created_ts)
                self.expires = int(winner2['expires'] or 0)
                self.wb = wb
                self.ws = None

                print("\r\033[2KLock acquired", flush=True)
                return self
            
            my_id = None

            for lock in active2:
                if is_mine(lock):
                    my_id = lock['sheetId']
                    break
            
            if my_id:
                batch_delete([my_id])
            
            self.sleep(0.5, jitter_ms=500)
        
        raise TimeoutError('Timed out during lock acquisition')

    def update(self):
        """Refresh local lock state from the worksheet.

        Returns:
            Remaining lock lease time in seconds.
        """
        ws = self._get_resource("worksheet")
        meta = self._get_resource("meta", ws=ws)

        owner = meta['owner']
        token = meta['token']
        created_ts = meta['created']
        expires_ts = meta['expires']

        self._ensure_control(owner, token, err_msg='Lock lost during update')

        self.created = int(created_ts or self.created)
        self.expires = int(expires_ts or 0)

        return int(self.expires or 0) - _now()

    def reset(self):
        """Extend the lock lease when it is near expiration.

        Returns:
            Remaining lock lease time in seconds.
        """
        remaining = int(self.expires or 0) - _now()
        if remaining >= self.reset_s:
            return remaining
        
        ws = self._get_resource("worksheet")
        meta = self._get_resource("meta", ws=ws)

        owner = meta['owner']
        token = meta['token']
        created_ts = meta['created']
        expires_ts = meta['expires']

        if not created_ts:
            created_ts = self.created or _now()
        
        self._ensure_control(owner, token, err_msg='Lock lost before reset')

        remaining = expires_ts - _now()
        if remaining >= self.reset_s:
            self.created = int(created_ts or self.created)
            self.expires = int(expires_ts or 0)

            return remaining
        
        new_expires = _now() + self.lease_s
        new_meta = [self.owner, self.token, str(created_ts or _now()), str(new_expires)]

        try:
            ws.update(self.META_RANGE, [new_meta])
        except Exception as e:
            raise RuntimeError('Failed to reset lock') from e
        
        self._confirm_ws(ws, err_msg='Lock lost after reset')

        meta2 = self._get_resource("meta", ws=ws, err_msg="Lock lost after reset")

        created_ts2 = meta2['created']
        expires_ts2 = meta2['expires']
        
        self.created = int(created_ts2 or created_ts or self.created)
        self.expires = int(expires_ts2 or new_expires)
        
        return int(self.expires or 0) - _now()

    def release(self, retries=5):
        """
        Release the workbook lock by deleting its worksheet.

        Args:
            retries: Number of release attempts before raising.

        Returns:
            True when the lock is released or no longer owned.
        """
        last_e = RuntimeError('Lock release failed\n')

        for attempt in range(retries):
            print(f'Releasing lock...[TRIES={attempt + 1}]', flush=True)

            try:
                wb = self.client.open_by_key(self.workbook_id)
                ws = wb.worksheet(self.sheet_name or self.owner)

                try:
                    meta = self._get_resource("meta", ws=ws)

                    owner = meta['owner']
                    token = meta['token']
                except RuntimeError:
                    print("\r\033[2KLock released", flush=True)
                    return True
                
                try:
                    self._ensure_control(owner, token, err_msg='Lock released (not owned)\n')
                except RuntimeError:
                    print("\r\033[2KLock released\n", flush=True)
                    return True
                
                wb.del_worksheet(ws)

                print("\r\033[2KLock released\n", flush=True)
                return True
            except Exception as e:
                last_e = e
                self.sleep(self.retry_s, jitter_ms=2500)
        
        raise last_e


# ---------------------------
# SHARED
# ---------------------------
def _get_ts():
    """Return the current local time as an HH:MM:SS.mmm timestamp string."""
    t = time.time()
    base = time.strftime("%H:%M:%S", time.localtime(t))
    ms = int((t - int(t)) * 1000)

    return f"{base}.{ms:03d}"


def _ts_to_ms(ts):
    """Convert an HH:MM:SS.mmm timestamp string to milliseconds since midnight.

    Args:
        ts: Timestamp value to parse.

    Returns:
        Integer milliseconds since midnight, or None if parsing fails.
    """
    try:
        ts = str(ts).strip()
        if not ts:
            return None
        
        hms, ms = ts.split(".", 1)
        h, m, s = hms.split(":")

        return ((3600*int(h) + 60*int(m) + int(s)) * 1000) + int(ms[:3])
    except Exception:
        return None


def _now():
    """Return the current Unix timestamp as integer seconds."""
    return int(time.time())


def _valid_phases():
    """Return the set of valid phase identifiers."""
    return {"0", "1"} | set(PHASE_CONFIG.keys())


def _cmd_run(*args):
    """Run one or more shell commands joined for Windows command execution.

    Args:
        *args: Command strings to execute in sequence.
    """
    cmd = " & ".join(args)
    os.system(cmd)


def _git_run(cmd, cwd=None, check=True):
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
        )


def _is_affirmative(text):
    """Interpret a user-entered yes/no response.

    Args:
        text: Raw response text.

    Returns:
        True for y/yes responses, otherwise False.
    """
    return str(text).strip().lower() in {"y", "yes"}


# ---------------------------
# SETUP
# ---------------------------
def _get_date():
    """Return today's date formatted for session metadata."""
    return datetime.now().strftime("%m/%d/%Y")


def setup(interfaces=None):
    """
    Initialize hardware, prompts, and session state before running.

    Args:
        interfaces: Optional BehaviorInterfaces instance for runtime boundaries.

    Returns:
        Tuple of ArduinoLink, SessionData, cursor object, and Prairie client.
    """
    interfaces = interfaces or BehaviorInterfaces()

    system_proxy = interfaces.system
    user_proxy = interfaces.user
    prairie_proxy = interfaces.prairie
    cursor_proxy = interfaces.cursor
    redis_proxy = interfaces.redis

    ser = None
    link = None
    client = None
    cursor = None

    animal_id = "DEV"
    phase_id = "3"

    try:
        ser, arduino_found = system_proxy.serial_connect()
        link = system_proxy.get_arduino(ser)

        flush_active = user_proxy.prompt_flush()
        if link.active:
            link.send_flush(flush_active)

        animal_id, animal_map = user_proxy.prompt_animal()
        workbook_id = system_proxy.get_workbook_id(animal_id, animal_map)

        if animal_id == "DEV":
            link.verbose = True

        phase_id = user_proxy.prompt_phase()

        if not arduino_found:
            raise RuntimeError(f'No Arduino detected (required for phase {phase_id})')

        imaging_active = user_proxy.prompt_imaging()
        ephys_active = user_proxy.prompt_ephys()

        print('\nInitializing resources...', flush=True)

        settings = system_proxy.get_config(phase_id)
        cfg = settings['cfg']
        side = settings['side']
            
        link.send_config(phase_id, settings)
        link.send_ephys(ephys_active)
        
        client = prairie_proxy.connect(imaging_active)

        is_easy = True
        if cfg and link.active:
            cursor, is_easy = cursor_proxy.connect(phase_id, side)
        
        if int(phase_id) > 1:
            try:
                link.send_and_wait(f"1 {'1' if is_easy else '0'}")
            except Exception as e:
                raise RuntimeError(f'[ERROR] Failed during initial trial config handshake: {e}') from e
        
        session_data = SessionData(animal_id, str(phase_id), _get_date())

        session_data.meta['workbook_id'] = workbook_id
        session_data.meta['imaging_active'] = bool(client is not None)
        session_data.meta['ephys_active'] = bool(ephys_active)

        session_data.log_trial_config(trial_n=1, type=is_easy, side=side)

        if ephys_active:
            link.start_ephys()

        print('Running session...\n', flush=True)
        link.send_start()

        redis_proxy.notify_start(session_data)

        return link, session_data, cursor, client
    except Exception as e:
        interfaces.exceptions.cache(e, 'setup')

        if link is not None:
            try:
                link.close()
            except Exception as e2:
                interfaces.exceptions.cache(e2, 'setup._cleanup')
        
        if ser is not None:
            try:
                ser.close()
            except Exception as e2:
                interfaces.exceptions.cache(e2, 'setup._cleanup')
        
        raise


# ---------------------------
# MAIN
# ---------------------------
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
    """Determine whether low recent trial rate should end the session early.

    Args:
        evt: Event dictionary containing timestamps and values.
        index: Current trial index.
        end_ms: Current trial end time in milliseconds since midnight.
        min_duration: Minimum elapsed session seconds before early exit.
        min_trials: Minimum trial count before early exit.

    Returns:
        True when early-exit criteria are met, otherwise False.
    """
    width = 5

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


def _cleanup(link, msg, timeout_s=30.0):
    """Request early termination, wait for Arduino END, and close the link.

    Args:
        link: ArduinoLink to stop and close.
        msg: Message printed when cleanup completes or times out.
        timeout_s: Maximum seconds to wait for END.
    """
    try:
        try:
            link.send(EARLY_STRING)
        except Exception:
            pass

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                typ, _, _ = link.msg_q.get(timeout=0.05)

                if typ == "END":
                    print(f'{msg}', flush=True)
                    return
            except Empty:
                pass
        
        print(f'{msg}', flush=True)
    finally:
        link.close()


def main(link, session_data, cursor, client=None, interfaces=None):
    """
    Run the behavioral session event loop.

    Args:
        link: ArduinoLink providing serial messages and commands.
        session_data: SessionData instance to populate.
        cursor: Cursor task object, or None.
        client: Optional PrairieClient for imaging control.
        interfaces: Optional BehaviorInterfaces instance.
    """
    interfaces = interfaces or BehaviorInterfaces()

    console_proxy = interfaces.console
    redis_proxy = interfaces.redis
    cursor_proxy = interfaces.cursor

    do_calibration = int(session_data.meta['phase']) > 4
    imaging_active = (bool(session_data.meta.get('imaging_active', False))
                      and (client is not None))
    ephys_active = bool(session_data.meta.get('ephys_active', False))

    K = 5
    N = 20
    trial_n = 0
    phase_id = str(session_data.meta['phase'])

    trial_stack = []
    calibrated = not do_calibration
    last_outcome = None

    trial_start_ms = None
    trial_dt = 0.0
    recent_outcomes = deque()

    def _get_msg(timeout_s=0.05):
        """
        Read one message from the Arduino queue.

        Args:
            timeout_s: Maximum seconds to wait for a message.

        Returns:
            Tuple of message type, timestamp, and payload.
        """
        typ, ts, payload = link.msg_q.get(timeout=timeout_s)

        return typ, ts, payload

    started = False
    first_trial = True

    try:
        while link.ser and link.ser.is_open:
            if cursor is not None and ABORT_EVT.is_set():
                raise KeyboardInterrupt

            try:
                typ, ts, payload = _get_msg(timeout_s=0.05)
            except Empty:
                continue

            if not started:
                started = True
                session_data.meta["t_start"] = _ts_to_ms(ts)

                console_proxy.show_start()
                console_proxy.show_header()

            if typ == "ERR":
                if isinstance(payload, BaseException):
                    _cmd_run('echo.')
                    raise payload
                
                raise RuntimeError(f"\nArduinoLink reader error: {payload!r}")

            if typ == "END":
                if ephys_active:
                    session_data.meta['_ephys_stopped'] = True

                break

            if typ == "RAW":
                session_data.add_raw_cap(ts, payload)

            if typ == "EVT":
                p = str(payload)

                try:
                    EVT_QUEUE.put_nowait((ts, p))
                except Exception:
                    pass

                if p == 'cue':
                    session_data.add_evt(ts, p)

                    if imaging_active and first_trial:
                        client_ok = client.start()
                        ttl_ok = link.start_imaging(delay_s=0.0)

                        if not (client_ok and ttl_ok):
                            raise RuntimeError('Initial START command failed')

                        first_trial = False
                    
                    trial_n += 1
                    last_outcome = None
                    trial_start_ms = _ts_to_ms(ts)
                
                if p == 'r_cue':
                    session_data.add_evt(ts, p)
                
                if p in {'hit', 'miss'}:
                    if last_outcome == p:
                        continue

                    last_outcome = p

                    session_data.add_evt(ts, p)

                    if imaging_active:
                        client_stop_ok = client.stop_after(delay_s=1.0)
                        ttl_stop_ok = link.stop_imaging(delay_s=1.0)

                        client_start_ok = client.start_after(delay_s=3.0)
                        ttl_start_ok = link.start_imaging(delay_s=3.0)

                        if not (client_stop_ok and ttl_stop_ok and client_start_ok and ttl_start_ok):
                            raise RuntimeError('Failed to schedule imaging restart')

                    end_ms = _ts_to_ms(ts)
                    if trial_start_ms is None or end_ms is None:
                        trial_dt = 0.0
                    else:
                        trial_dt = max(0.0, (end_ms - trial_start_ms) / 1000.0)

                    recent_outcomes.append(p)

                    n_hit = sum(1 for o in recent_outcomes if o == 'hit')
                    n_miss = sum(1 for o in recent_outcomes if o == 'miss')

                    console_proxy.show_trial_info(trial_dt, n_hit, n_miss, p)

                    if do_calibration:
                        trial_stack.insert(0, p)
                        if len(trial_stack) > N:
                            trial_stack.pop()
                        
                        if not calibrated:
                            if len(trial_stack) >= N:
                                K, N, calibration_hits = _update_easy_rate(session_data, trial_stack)

                                session_data.meta['K2'] = K
                                calibrated = True

                                print(f'\nCalibration finished [hits={calibration_hits}/20, K={K}, N={N}]\n', flush=True)
                                console_proxy.show_header()

                    if int(phase_id) >= 4:
                        if calibrated:
                            early_exit = _is_early_exit(session_data.evt, trial_n, end_ms)

                            if early_exit:
                                if imaging_active:
                                    client.stop()
                                    time.sleep(1)
                                    client.finish()

                                link.stop_ephys(session_data, safe=True)

                                _cleanup(link, 'Terminated by early exit')
                                break

                        next_trial_n = trial_n + 1
                        next_easy = _get_easy(int(phase_id), next_trial_n, K)
                        next_side = PHASE_CONFIG[phase_id]['side']

                        time.sleep(0.05)
                        link.send_and_wait(f'{next_trial_n} {"1" if next_easy else "0"}')
                        session_data.log_trial_config(next_trial_n, next_easy, next_side)

                        if cursor is not None:
                            cursor_proxy.update_trial(cursor, next_easy, next_side)

                if p in {"hit", "lick"}:
                    session_data.add_raw_evt(ts, p)

            if typ == "ENC":
                p = str(payload)
                
                try:
                    pos = float(p)
                    session_data.add_enc(ts, str(pos))

                    try:
                        ENC_QUEUE.put_nowait(("WHEEL", pos))
                    except Exception:
                        pass
                except Exception:
                    pass
    except KeyboardInterrupt:
        session_data.meta["aborted"] = True

        link.stop_ephys(session_data, safe=True)
        _cleanup(link, "\nTerminated by KeyboardInterrupt")
        raise
    except Exception as e:
        interfaces.exceptions.cache(e, 'main')
    finally:
        if session_data.meta["t_start"] is None:
            session_data.meta["t_start"] = _ts_to_ms(_get_ts())

        session_data.meta["t_stop"] = _ts_to_ms(_get_ts())

        t0 = session_data.meta['t_start']
        t1 = session_data.meta['t_stop']
        dt = 0 if (t0 is None or t1 is None) else max(0, (t1 - t0) // 1000)
        session_data.meta["duration_sec"] = int(dt)

        link.stop_ephys(session_data, safe=True)

        if client is not None:
            client.stop()

        redis_proxy.notify_finish()


if __name__ == "__main__":
    interfaces = BehaviorInterfaces()
    interfaces.console.clear()

    link = None
    session_data = None
    cursor = None
    prairie = None

    animal_id_for_log = "UNKNOWN"
    phase_id_for_log = "0"

    run_exc = None

    try:
        link, session_data, cursor, prairie = setup(interfaces)

        if session_data is not None:
            animal_id_for_log = session_data.meta.get("animal", "UNKNOWN")
            phase_id_for_log = session_data.meta.get("phase", "0")

        main(link, session_data, cursor, prairie, interfaces)
    except SystemExit as e:
        pass
    except KeyboardInterrupt as e:
        pass
    except BaseException as e:
        run_exc = e
        interfaces.exceptions.cache(e, '__main__')
        interfaces.console.show_exceptions()
    finally:
        interfaces.console.show_summary(session_data)
        run_info = (animal_id_for_log, phase_id_for_log)

        if prairie is not None:
            try:
                interfaces.prairie.finish(prairie, session_data)
            except Exception as e:
                interfaces.exceptions.cache(e, "__main__.prairie_finish")
                ExceptionInterface(*run_info).log_and_commit(e)

        if cursor is not None:
            try:
                interfaces.cursor.stop(cursor)
            except Exception as e:
                interfaces.exceptions.cache(e, "__main__.cursor_stop")
                ExceptionInterface(*run_info).log_and_commit(e)
        
        if link is not None:
            try:
                link.close()
            except Exception as e:
                interfaces.exceptions.cache(e, '__main__.link_close')
                ExceptionInterface(*run_info).log_and_commit(e)

        if session_data is not None and session_data.is_finished:
            if session_data.meta.get('animal', None) not in {None, "DEV"}:
                try:
                    interfaces.email.send_session_summary(session_data)
                except Exception as e:
                    interfaces.exceptions.cache(e, '__main__.send_email')
                    ExceptionInterface(*run_info).log_and_commit(e)
        
        if session_data is not None and session_data.any_data():
            if session_data.meta.get('animal', None) not in {None, "DEV"}:
                try:
                    if interfaces.user.confirm_save():
                        if interfaces.saving.resolve_protocol(session_data):
                            interfaces.saving.save_raw(session_data)

                            ok = interfaces.saving.save_session(session_data)
                            if not ok:
                                interfaces.console.warning("Google Sheets save failed (saving locally instead)")
                        else:
                            print("Session exited without saving", flush=True)
                except Exception as e:
                    interfaces.exceptions.cache(e, '__main__.safe_save')

        interfaces.console.show_exceptions()
        interfaces.console.wait_for_key()

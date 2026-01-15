import os
import sys
import warnings
import traceback

import random
import time
import socket
import uuid
import json
import functools
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from collections import deque
from threading import Thread, Event

import serial
import serial.tools.list_ports
from cursor_utils import BCI

import gspread
from gspread.utils import rowcol_to_a1
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv

import smtplib
from email.message import EmailMessage

import subprocess
import shutil
import tempfile


# ---------------------------
# BASIC CONFIG
# ---------------------------
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="pkg_resources is deprecated as an API.*",
)

SCRIPT_DIR = Path(__file__).resolve().parent
ANIMAL_MAP_PATH = SCRIPT_DIR / "animal_map.json"
ERROR_LOG_PATH = SCRIPT_DIR / "errors.log"

BAUDRATE = 1_000_000
EARLY_END_STRING = "E"
SESSION_END_STRING = "S"

PHASE_CONFIG = {
    "3": {"bidirectional": True, "threshold": 30.0},
    "4": {"bidirectional": True, "threshold": 60.0},
    "5": {"bidirectional": True, "threshold": 90.0},
    "6": {"bidirectional": False, "threshold": 30.0},
    "7": {"bidirectional": False, "threshold": 60.0},
    "8": {"bidirectional": False, "threshold": 90.0}
    }

MAX_STREAK = 4
LAST_SIDE = None
SIDE_STREAK = 0

EVT_QUEUE: "Queue[tuple[str, str]]" = Queue()
ENC_QUEUE: "Queue[tuple[str, str]]" = Queue()
DEV_QUEUE: "Queue[tuple[str, str, object]]" = Queue()


# ---------------------------
# FORMAT HELPERS
# ---------------------------
def _get_date():
    return datetime.now().strftime("%m/%d/%Y")


def _get_ts():
    t = time.time()
    base = time.strftime("%H:%M:%S", time.localtime(t))
    ms = int((t - int(t)) * 1000)

    return f"{base}.{ms:03d}"


def _now():
    return int(time.time())


# ---------------------------
# TRIAL INFO
# ---------------------------
INFO_HEADER = "   TRIAL   |    AVG_DT    |    N_HIT    |    N_MISS    |     RATE"
INFO_HLINE  = "----------------------------------------------------------------------"

INFO_COL_WIDTH = (11, 14, 13, 14, 14)
INFO_COL_PAD = [(" " * n) for n in (4, 4, 5, 6, 4)]
INFO_HEADER_FLAG = False

INFO_WIN_SIZE = 20


def _status_cell(value, pad, width):
    entry = pad + str(value)

    if len(entry) < width:
        entry = entry + (" " * (width - len(entry)))
    else:
        entry = entry[:width]
    
    return entry


def show_trial_header():
    print(INFO_HEADER, flush=True)
    print(INFO_HLINE, flush=True)


def show_trial_info(trial_n, avg_dt, n_hit, n_miss):
    n_total = n_hit + n_miss
    rate = ((n_hit / n_total) * 100.0) if n_total else 0.0

    trial_str = f'{trial_n}'
    dt_str = f'{avg_dt:.2f}s'
    hit_str = f'{n_hit}'
    miss_str = f'{n_miss}'
    rate_str = f'{rate:.1f}%'

    row = ""

    for i, s in enumerate([trial_str, dt_str, hit_str, miss_str, rate_str]):
        row += _status_cell(s, INFO_COL_PAD[i], INFO_COL_WIDTH[i]) + "|"
    
    row = row[:-1]
    print(row, flush=True)


# ---------------------------
# LOGGING
# ---------------------------
REPO_SLUG = "GonzalesLabVU/Behavior-BCI"
REPO_BRANCH = "main"
REPO_REL_PATH = Path("pc") / "config" / "errors.log"

ERROR_LOGGED = False


def _ensure_session_tracking(session_data):
    session_data.meta.setdefault('trial_config', [])
    session_data.meta.setdefault('K1', 5)
    session_data.meta.setdefault('K2', None)


def log_trial_config(session_data, trial_n, type, side):
    _ensure_session_tracking(session_data)

    session_data.meta['trial_config'].append({
        'trial': int(trial_n),
        'is_easy': bool(type),
        'side': str(side)
        })


def time_this(fcn):
    @functools.wraps(fcn)

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fcn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        units = 's'

        if elapsed < 0.001:
            elapsed = elapsed * 1000
            units = 'ms'

        print(f'{fcn.__name__} executed in {elapsed:.3f} {units}\n')
        return result
    
    return wrapper


def log_error(animal_id, phase_id, exc):
    global ERROR_LOGGED
    ERROR_LOGGED = True

    try:
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')

        client = os.getenv('USER_ID', 'UNKNOWN_USER')
        script_name = Path(__file__).name

        animal = str(animal_id)
        phase = str(phase_id)

        header = [
            f'TIME = {date_str} {time_str}',
            f'USER = {client}',
            f'ANIMAL = {animal}',
            f'PHASE = {phase}',
            f'SOURCE = {script_name}'
            ]
        hline = ['-' * 40]
        body = []

        if isinstance(exc, BaseException):
            tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)

            for line in "".join(tb_lines).rstrip('\n').splitlines():
                body.append(f'  {line}')
        else:
            body.append(f'  {type(exc).__name__}: {exc!r}')
        
        with open(ERROR_LOG_PATH, 'a', encoding='utf-8') as f:
            for line in header + hline + body:
                f.write(line + '\n')
            
            f.write('\n')
    except Exception:
        pass


def commit_error_log(animal_id='UNKNOWN', phase_id='0'):
    global ERROR_LOGGED

    if not ERROR_LOGGED:
        return False
    
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print('[WARNING] GITHUB_TOKEN not set, skipping errors.log push', flush=True)
        return False
    
    if not ERROR_LOG_PATH.exists():
        return False
    
    remote_url = f'https://x-access-token:{token}@github.com/{REPO_SLUG}.git'

    def git_run(cmd, cwd=None, check=True):
        return subprocess.run(cmd,
                              cwd=cwd,
                              check=check,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              text=True)
    
    try:
        with tempfile.TemporaryDirectory(prefix='behavior_bci_repo_') as td:
            repo_dir = Path(td) / "repo"

            git_run(['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, remote_url, str(repo_dir)])

            dest_path = repo_dir / REPO_REL_PATH
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(ERROR_LOG_PATH, dest_path)

            st = git_run(['git', 'status', '--porcelain', str(REPO_REL_PATH)], cwd=repo_dir).stdout.strip()
            if not st:
                return False
            
            git_run(['git', 'config', 'user.name', 'behavior-bci-bot'], cwd=repo_dir)
            git_run(['git', 'config', 'user.email', 'behavior-bci-bot@users.noreply.github.com'], cwd=repo_dir)

            git_run(['git', 'add', str(REPO_REL_PATH)], cwd=repo_dir)

            msg = f'Update errors.log (animal={animal_id}, phase={phase_id})'
            c = subprocess.run(['git', 'commit', '-m', msg],
                               cwd=repo_dir,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)
            
            if c.returncode != 0:
                return False
            
            git_run(['git', 'push', 'origin', REPO_BRANCH], cwd=repo_dir, check=True)

            return True
    except Exception as e:
        print(f'[WARNING] Failed to commit errors.log: {type(e).__name__}', flush=True)
        return False


# ---------------------------
# RESOURCE LOADING
# ---------------------------
def load_animal_map(path=ANIMAL_MAP_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, dict):
        raise ValueError('animal_map.json must be a dict')
    
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError('animal_map.json keys and values must be strings')
    
    return data


def validate_animal(animal_id, animal_map):
    if not any(animal_id in key for key in animal_map.keys()):
        raise ValueError('Animal not found in animal_map.json')


def validate_resources():
    map_file = SCRIPT_DIR / 'animal_map.json'
    if not map_file.exists():
        raise FileNotFoundError('animal_map.json not found in the script directory')

    creds_file = SCRIPT_DIR / 'credentials.json'
    if not creds_file.exists():
        raise FileNotFoundError('credentials.json not found in the script directory')

    load_dotenv(SCRIPT_DIR / ".env")


def _require_env(name):
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f'{name} not found in .env')
    
    return v


# ---------------------------
# SERIAL COMMUNICATION
# ---------------------------
def find_arduino_port():
    ports = serial.tools.list_ports.comports()

    for port in ports:
        dsc = (port.description or "").lower()
        if "arduino" in dsc or "usb serial" in dsc:
            return port.device
        
    return None


def update_easy_rate(session_data, trial_stack):
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


class ArduinoLink:
    def __init__(self, ser):
        self.ser = ser
        self.stop_evt = Event()
        self.ack_evt = Event()
        self.msg_q: "Queue[tuple[str, str, object]]" = Queue()
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

                if line == "A":
                    self.ack_evt.set()
                    continue

                ts = _get_ts()

                if line == SESSION_END_STRING:
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

                self.msg_q.put(("RAW", ts, line))

        except Exception as e:
            try:
                self.msg_q.put(("ERR", _get_ts(), e))
            except Exception:
                pass

    def start(self):
        self._reader.start()

    def close(self):
        self.stop_evt.set()

        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

    def send_and_wait(self, text, timeout=5.0):
        self.ack_evt.clear()
        self.ser.write((text.strip() + "\n").encode("utf-8"))
        self.ser.flush()

        if not self.ack_evt.wait(timeout=timeout):
            raise TimeoutError(f"No ACK after sending: {text!r}")

    def send(self, text):
        self.ser.write((text.strip() + "\n").encode("utf-8"))
        self.ser.flush()


class SessionData:
    def __init__(self, animal_id, phase_id, date_str):
        self.meta = {
            "date": date_str,
            "animal": animal_id,
            "phase": phase_id,
            "start_wall": None,
            "end_wall": None,
            "duration_sec": None,
            "aborted": False,
            }
        self.evt = {"timestamps": [], "values": []}
        self.enc = {"timestamps": [], "values": []}

    def add_evt(self, ts, payload):
        self.evt["timestamps"].append(ts)
        self.evt["values"].append(payload)

    def add_enc(self, ts, payload):
        self.enc["timestamps"].append(ts)
        self.enc["values"].append(payload)

    def any_data(self):
        return bool(self.evt["timestamps"]) or bool(self.enc["timestamps"])

    def to_dict(self):
        def _json_safe(x):
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
            easy_trials = []
            normal_trials = []

            left_targets = []
            right_targets = []
            both_targets = []
        
        meta_out.setdefault('K1', 5)
        meta_out.setdefault('K2', None)

        meta_out['easy_trials'] = list(easy_trials)
        meta_out['normal_trials'] = list(normal_trials)
        meta_out['left_targets'] = list(left_targets)
        meta_out['right_targets'] = list(right_targets)
        meta_out['both_targets'] = list(both_targets)

        return {
            'meta': _json_safe(meta_out),
            'evt': _json_safe(self.evt),
            'enc': _json_safe(self.enc)
            }


def choose_trial_type(trial_n, K):
    if trial_n <= 20:
        return ((trial_n - 1) % 5) == 0

    K = max(1, int(K))
    return ((trial_n - 21) % K) == 0


def choose_target_side(phase_id):
    global LAST_SIDE, SIDE_STREAK

    cfg = PHASE_CONFIG.get(str(phase_id))
    if not cfg:
        return "B"
    
    if cfg.get('bidirectional', False):
        return "B"
    
    if LAST_SIDE in {"L", "R"} and SIDE_STREAK >= MAX_STREAK:
        side = "R" if LAST_SIDE == "L" else "L"
    else:
        side = random.choice(["L", "R"])
    
    if side == LAST_SIDE:
        SIDE_STREAK += 1
    else:
        LAST_SIDE = side
        SIDE_STREAK = 1
    
    return side


def send_next_config(link, next_type, next_side, timeout=3.0):
    link.send_and_wait(f'T {1 if next_type else 0} {next_side}', timeout=timeout)


# ---------------------------
# DATA SAVING
# ---------------------------
API_SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
API_CREDS = Credentials.from_service_account_file(str(SCRIPT_DIR / 'credentials.json'),
                                                  scopes=API_SCOPES)
API_CLIENT = gspread.authorize(API_CREDS)
API_DRIVE = build('drive', 'v3', credentials=API_CREDS, cache_discovery=False)

LOCK_POLL_S = 5.0
LOCK_RETRY_S = 5.0
LOCK_LEASE_S = 180
LOCK_RESET_S = 60
LOCK_TIMEOUT_S = 300

LOCK_TAG = "------ LOCK ------"
LOCK_TAG_RANGE = "A1"
LOCK_META_RANGE = "A2:D2"


def _build_meta_rows(session_data):
    _ensure_session_tracking(session_data)

    cfg = session_data.meta.get('trial_config', []) or []
    easy_trials = [c['trial'] for c in cfg if c.get('is_easy') is True]
    normal_trials = [c['trial'] for c in cfg if c.get('is_easy') is False]

    left_targets = [c['trial'] for c in cfg if c.get('side') == "L"]
    right_targets = [c['trial'] for c in cfg if c.get('side') == "R"]
    both_targets = [c['trial'] for c in cfg if c.get('side') == "B"]

    meta_pairs = [
        ('K1', session_data.meta.get('K1', 5)),
        ('K2', session_data.meta.get('K2', None)),
        ('easy_trials', easy_trials),
        ('normal_trials', normal_trials),
        ('left_targets', left_targets),
        ('right_targets', right_targets),
        ('both_targets', both_targets)
        ]
    
    out = []
    for key, value in meta_pairs:
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                out.append([key, "None"])
            else:
                out.append([key, value[0]])

                for val in value[1:]:
                    out.append(["", val])
        else:
            out.append([key, "" if value is None else value])
    
    return out


def _get_workbook_id(animal_id, animal_map):
    try:
        map_key = next(key for key in animal_map.keys() if animal_id in key)
    except StopIteration:
        raise ValueError(f'No cohort mapping found for Animal {animal_id!r}')
    
    cohort_name = animal_map[map_key]
    env_var = f'{cohort_name}_ID'

    return _require_env(env_var)


def _get_client_id():
    return f'{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}'


class FileLock:
    def __init__(self, workbook_id, owner):
        self.poll_s = float(LOCK_POLL_S)
        self.retry_s = float(LOCK_RETRY_S)
        self.lease_s = int(LOCK_LEASE_S)
        self.reset_s = int(LOCK_RESET_S)
        self.timeout_s = int(LOCK_TIMEOUT_S)

        self.client = API_CLIENT
        self.workbook_id = workbook_id
        self.owner = owner
        self.token = uuid.uuid4().hex

        self.sheet_name = None
        self.created = 0
        self.expires = 0

        self.wb = None
        self.ws = None

    def _open_wb(self):
        self.wb = self.client.open_by_key(self.workbook_id)
        return self.wb
    
    def _get_ws(self):
        if self.sheet_name is None:
            raise RuntimeError('Lock not acquired (sheet_name is None)')
        
        if self.wb is None:
            self._open_wb()
        
        try:
            self.ws = self.wb.worksheet(self.sheet_name)
        except Exception:
            self._open_wb()
            self.ws = self.wb.worksheet(self.sheet_name)
        
        return self.ws

    def _confirm_ws(self, ws, err_msg='Lock lost'):
        meta = self._get_meta(ws, err_msg=err_msg)
        self._ensure_control(meta['owner'], meta['token'], err_msg=err_msg)

    def _is_lock(self, ws):
        try:
            return (ws.acell(LOCK_TAG_RANGE).value or "") == LOCK_TAG
        except Exception:
            return False

    def _get_meta(self, ws, err_msg='Lock tag missing (lock lost)'):
        try:
            vals = ws.get('A1:D2')
        except Exception as e:
            raise RuntimeError(err_msg) from e
        
        tag = (vals[0][0] if vals and vals[0] else "") if vals else ""
        if (tag or "") != LOCK_TAG:
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

        meta = {
            'tag': tag,
            'owner': owner,
            'token': token,
            'created': created_ts,
            'expires': expires_ts,
            'info': row
            }
        
        return meta

    def _ensure_control(self, owner, token, err_msg='Lock lost'):
        if owner != self.owner or token != self.token:
            raise RuntimeError(err_msg)
        
    def sleep(self, dur_s, jitter_ms=1000):
        if dur_s < 0:
            dur_s = 0.0
        
        if jitter_ms and jitter_ms > 0:
            dur_s += random.random() * (jitter_ms / 1000.0)
        
        time.sleep(dur_s)

    def acquire(self):
        wb = self._open_wb()

        deadline = time.monotonic() + self.timeout_s
        attempt = 0
        created_ts = _now()

        print('Acquiring lock...', end='\r', flush=True)

        def q_sheet(title):
            return "'" + title.replace("'", "''") + "'"
        
        def to_int(x, default=0):
            try:
                return int(x)
            except Exception:
                return default
            
        def scan_locks():
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
                if (tag or "") != LOCK_TAG:
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
            if not ids:
                return
            
            req = [{'deleteSheet': {'sheetId': id}} for id in ids]

            try:
                wb.batch_update({'requests': req})
            except Exception:
                pass

        def is_mine(lock):
            return lock.get('owner') == self.owner and lock.get('token') == self.token

        while time.monotonic() < deadline:
            attempt += 1
            print(f'Acquiring lock...[TRIES={attempt}]', end='\r', flush=True)

            now = _now()

            try:
                locks = scan_locks()
            except Exception:
                self.sleep(self.retry_s, jitter_ms=750)
                wb = self._open_wb()
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
                wb = self._open_wb()
                continue

            try:
                expires_ts = _now() + self.lease_s
                my_meta = [self.owner, self.token, str(created_ts), str(expires_ts)]

                my_lock.batch_update([
                    {'range': LOCK_TAG_RANGE, 'values': [[LOCK_TAG]]},
                    {'range': LOCK_META_RANGE, 'values': [my_meta]}
                    ])
            except Exception:
                try:
                    wb.del_worksheet(my_lock)
                except Exception:
                    pass

                self.sleep(self.poll_s, jitter_ms=750)
                wb = self._open_wb()
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
        ws = self._get_ws()
        meta = self._get_meta(ws)

        owner = meta['owner']
        token = meta['token']
        created_ts = meta['created']
        expires_ts = meta['expires']

        self._ensure_control(owner, token, err_msg='Lock lost during update')

        self.created = int(created_ts or self.created)
        self.expires = int(expires_ts or 0)

        return int(self.expires or 0) - _now()

    def reset(self):
        remaining = int(self.expires or 0) - _now()
        if remaining >= self.reset_s:
            return remaining
        
        ws = self._get_ws()
        meta = self._get_meta(ws)

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
            ws.update(LOCK_META_RANGE, [new_meta])
        except Exception as e:
            raise RuntimeError('Failed to reset lock') from e
        
        self._confirm_ws(ws, err_msg='Lock lost after reset')

        meta2 = self._get_meta(ws, err_msg='Lock lost after reset')

        created_ts2 = meta2['created']
        expires_ts2 = meta2['expires']
        
        self.created = int(created_ts2 or created_ts or self.created)
        self.expires = int(expires_ts2 or new_expires)
        
        return int(self.expires or 0) - _now()

    def release(self, retries=5):
        last_e = RuntimeError('Lock release failed')

        for attempt in range(retries):
            print(f'Releasing lock...[TRIES={attempt + 1}]', end='\r', flush=True)

            try:
                wb = self.client.open_by_key(self.workbook_id)
                ws = wb.worksheet(self.sheet_name or self.owner)

                try:
                    meta = self._get_meta(ws)

                    owner = meta['owner']
                    token = meta['token']
                except RuntimeError:
                    print("\r\033[2KLock released", flush=True)
                    return True
                
                try:
                    self._ensure_control(owner, token, err_msg='Lock released (not owned)')
                except RuntimeError:
                    print("\r\033[2KLock released", flush=True)
                    return True
                
                wb.del_worksheet(ws)

                print("\r\033[2KLock released", flush=True)
                return True
            except Exception as e:
                last_e = e
                self.sleep(self.retry_s, jitter_ms=2500)
        
        raise last_e


def save_data(session_data):
    workbook_id = session_data.meta["workbook_id"]
    client_id = _get_client_id()

    def _batch_write_cols(ws, start_row, start_col, data, chunk_rows=2000, group_chunks=10):
        sheet = ws.spreadsheet
        name = ws.title

        _get_range = lambda r1, c1, r2, c2: f'{name}!{rowcol_to_a1(r1, c1)}:{rowcol_to_a1(r2, c2)}'

        req = []
        n = len(data)

        for i in range(0, n, chunk_rows):
            chunk = data[i:i+chunk_rows]
            r1 = start_row + i
            r2 = r1 + len(chunk) - 1
            c1 = start_col
            c2 = start_col + 1

            req.append({'range': _get_range(r1, c1, r2, c2), 'values': chunk})

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

        for dtype, sheet_name in (('evt', 'Event'), ('enc', 'Encoder'), ('meta', 'Metadata')):
            if dtype == 'meta':
                data_rows = _build_meta_rows(session_data)
                n_rows = len(data_rows)
                label = 'metadata'
            else:
                d = getattr(session_data, dtype)
                n_rows= len(d['timestamps'])
                label = sheet_name.lower()

            if dtype != 'meta':
                print(f'Writing {label} data...', flush=True)
            else:
                print('Writing metadata...', flush=True)

            if n_rows == 0:
                continue

            lock.update()
            lock.reset()

            try:
                ws = wb.worksheet(sheet_name)
            except Exception:
                ws = wb.add_worksheet(title=sheet_name, rows=200, cols=26)

            max_col = len(ws.row_values(2))
            needed_cols = max_col + 2
            if ws.col_count < needed_cols:
                ws.add_cols(needed_cols - ws.col_count)
            
            header_rng = f'{rowcol_to_a1(1, max_col+1)}:{rowcol_to_a1(2, max_col+2)}'
            skip_rng = f'{rowcol_to_a1(3, max_col+1)}:{rowcol_to_a1(3, max_col+2)}'

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
            
            if dtype == 'meta':
                data = data_rows
            else:
                data = [[ts, val] for ts, val in zip(d['timestamps'], d['values'])]
            
            first_row = 4
            first_col = max_col + 1

            lock.update()
            lock.reset()

            _batch_write_cols(ws, first_row, first_col, data)
            print("\r\033[2K", end="", flush=True)

        
        return True
    except KeyboardInterrupt as e:
        try:
            fallback_save(session_data, exc=e)
        except Exception:
            pass

        return False
    except Exception as e:
        try:
            fallback_save(session_data, exc=e)
        except Exception as e2:
            log_error(session_data.meta.get('animal', 'UNKNOWN'), session_data.meta.get('phase', '0'), e2)
        
        log_error(session_data.meta.get('animal', 'UNKNOWN'), session_data.meta.get('phase', '0'), e)
        return False
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception as e:
                try:
                    fallback_save(session_data, exc=e)
                except Exception as e2:
                    log_error(session_data.meta.get('animal', 'UNKNOWN'), session_data.meta.get('phase', '0'), e2)
                
                log_error(session_data.meta.get('animal', 'UNKNOWN'), session_data.meta.get('phase', '0'), e)


def fallback_save(session_data, exc=None):
    animal = str(session_data.meta.get('animal', 'UNKNOWN'))
    phase = str(session_data.meta.get('phase', '0'))
    date = str(session_data.meta.get('date', '0000-00-00')).replace('/', '.')
    rand = uuid.uuid4().hex[:6]

    out_path = SCRIPT_DIR / f'{date}_animal={animal}_phase={phase}_id={rand}.json'
    payload = session_data.to_dict()

    if exc is not None:
        payload.setdefault('meta', {})
        payload['meta']['fallback_saved'] = True
        payload['meta']['fallback_reason'] = f'{type(exc).__name__}: {exc}'
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=4)

    try:
        commit_error_log(animal, phase)
    except Exception:
        pass
    
    print("\r\033[2K", end="", flush=True)
    print(f"[WARNING] Saved session data locally to {out_path.name}", flush=True)

    return out_path


# ---------------------------
# EMAIL SMTP
# ---------------------------
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587


def _format_subject(date, animal, phase):
    date_str = datetime.strptime(date, '%m/%d/%Y').strftime('%b-%d')
    animal_str = f'Animal {animal}'
    phase_str = f'Phase {phase}'

    return f'{date_str}  |  {animal_str}  |  {phase_str}'


def _format_body(t_start, t_stop, dur_s, evt):
    m, s = divmod(int(dur_s or 0), 60)
    t_elapsed = f'{m}:{s:02d}'

    n_hits = sum(1 for e in evt['values'] if e == 'hit')
    n_total = sum(1 for e in evt['values'] if e == 'cue')
    hit_rate = ((n_hits / n_total) * 100) if n_total else 0.0

    lines = [
        ("Started", str(t_start)),
        ("Finished", str(t_stop)),
        ("Elapsed", str(t_elapsed)),
        ("", ""),
        ("Total Trials", str(n_total)),
        ("Success Rate", f"{hit_rate:.1f}%")
        ]
    width = max(len(label) for label, _ in lines)

    out = []
    for label, value in lines:
        if not label and not value:
            out.append("")
        else:
            out.append(f"{label + ':':<{width + 1}}  {value}")
    
    return "\n".join(out)


def send_email(session_data):
    smtp_username = _require_env("SMTP_USERNAME")
    smtp_password = _require_env("SMTP_PASSWORD")
    smtp_to_addr = _require_env("SMTP_TO_ADDR")

    date = session_data.meta['date']
    animal = session_data.meta['animal']
    phase = session_data.meta['phase']

    subject = _format_subject(date, animal, phase)

    t_start = datetime.fromisoformat(session_data.meta['start_wall']).strftime('%I:%M %p').lstrip('0')
    t_stop = datetime.fromisoformat(session_data.meta['end_wall']).strftime('%I:%M %p').lstrip('0')
    dur_s = session_data.meta['duration_sec']
    evt = session_data.evt

    body = _format_body(t_start, t_stop, dur_s, evt)

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

    try:
        print('Notifying users of session completion...', end='\r', flush=True)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()

            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        print('Notifying users of session completion...done')
    except Exception:
        raise


# ---------------------------
# TERMINATION / CLEANUP
# ---------------------------
def is_early_exit(trial_stack, N):
    if len(trial_stack) < N:
        return False
    
    n_hits = sum(1 for x in trial_stack[:N] if x == "hit")
    return n_hits < (N / 4)


def end_session(link, msg):
    try:
        link.send_and_wait(EARLY_END_STRING, timeout=2.0)
        deadline = time.time() + 2.0

        while time.time() < deadline:
            try:
                typ, _, _ = link.msg_q.get(timeout=0.05)
                if typ == "END":
                    print(f"{msg}", flush=True)
                    return
            except Empty:
                pass
    finally:
        link.close()


# ---------------------------
# TOP LEVEL
# ---------------------------
def _dev_start(evt_queue, enc_queue, dev_queue, session_data, state, stop_evt, cursor, key_speed=50.0, trial_ms=30_000, delay_ms=3000):
    t = Thread(target=_dev_loop,
               args=(
                   evt_queue,
                   enc_queue,
                   dev_queue,
                   session_data,
                   state,
                   stop_evt,
                   cursor,
                   float(key_speed),
                   int(trial_ms),
                   int(delay_ms)
                   ),
               daemon=True)
    t.start()

    return t


def _dev_loop(evt_queue, enc_queue, dev_queue, session_data, state, stop_evt, cursor, key_speed, trial_ms, delay_ms):
    try:
        import pygame as pg
    except Exception:
        return
    
    MIN_DEG = -90.0
    MAX_DEG = +90.0

    def _emit_evt(e):
        t = _get_ts()

        try:
            evt_queue.put_nowait((t, e))
        except Exception:
            pass

        try:
            dev_queue.put_nowait(("EVT", t, e))
        except Exception:
            pass

        try:
            session_data.add_evt(t, e)
        except Exception:
            pass
    
    def _emit_enc(d):
        t = _get_ts()
        v = f'{d:.2f}'

        try:
            enc_queue.put_nowait((t, v))
        except Exception:
            pass

        try:
            session_data.add_enc(t, v)
        except Exception:
            pass

    def _is_hit(d):
        try:
            th = float(getattr(cursor, 'threshold'))
            easy_th = float(getattr(cursor, 'easy_threshold', 15.0))
        except Exception:
            th = 30.0
            easy_th = 15.0
        
        try:
            from cursor_utils import TRIAL_CONFIG, TRIAL_LOCK

            with TRIAL_LOCK:
                is_easy, alignment = TRIAL_CONFIG
        except Exception:
            is_easy, alignment = (False, "B")
        
        alignment = (alignment or "B").upper()
        if alignment not in {"L", "R", "B"}:
            alignment = "B"
        
        T = easy_th if bool(is_easy) else th

        match alignment:
            case "B":
                return abs(d) >= abs(T)
            case "L":
                return d <= -abs(T)
            case "R":
                return d >= +abs(T)
            case _:
                return False
    
    def _is_miss(d):
        try:
            th = float(getattr(cursor, 'threshold'))
            easy_th = float(getattr(cursor, 'easy_threshold', 15.0))
        except Exception:
            th = 30.0
            easy_th = 15.0
        
        try:
            from cursor_utils import TRIAL_CONFIG, TRIAL_LOCK

            with TRIAL_LOCK:
                is_easy, alignment = TRIAL_CONFIG
        except Exception:
            is_easy, alignment = (False, "B")
        
        alignment = (alignment or "B").upper()
        if alignment not in {"L", "R", "B"}:
            alignment = "B"
        
        T = easy_th if bool(is_easy) else th

        match alignment:
            case "L":
                return d >= +abs(T)
            case "R":
                return d <= -abs(T)
            case _:
                return False

    def _now_ms():
        return int(time.time() * 1000)
   
    try:
        if not pg.get_init():
            pg.init()
    except Exception:
        pass

    deg = 0.0
    last_t = time.time()

    trial_active = False
    trial_start_ms = None
    outcome_sent = False
    next_cue_ms = _now_ms()

    state['trial_active'] = False

    while not stop_evt.is_set():
        now_s = time.time()
        dt = max(0.0, now_s - last_t)
        last_t = now_s

        try:
            pg.event.pump()
        except Exception:
            pass

        now_ms = _now_ms()

        if (not trial_active) and (now_ms >= next_cue_ms):
            trial_active = True
            state['trial_active'] = True

            trial_start_ms = now_ms
            outcome_sent = False
            deg = 0.0

            _emit_evt("cue")
        
        if not trial_active:
            time.sleep(0.01)
            continue

        try:
            keys = pg.key.get_pressed() if pg.get_init() else None
        except Exception:
            keys = None
        
        if keys:
            dx = 0.0

            if keys[pg.K_LEFT]:
                dx -= key_speed * dt
            if keys[pg.K_RIGHT]:
                dx += key_speed * dt
            
            if dx:
                deg = max(MIN_DEG, min(MAX_DEG, deg + dx))
        
        _emit_enc(deg)

        elapsed_ms = 0 if trial_start_ms is None else max(0, now_ms - trial_start_ms)
        hit = _is_hit(deg)
        miss = _is_miss(deg)
        timeout = elapsed_ms >= int(trial_ms)

        if (not outcome_sent) and (hit or miss or timeout):
            outcome_sent = True

            if hit:
                _emit_evt("hit")
            if miss or timeout:
                _emit_evt("miss")

            trial_active = False
            state['trial_active'] = False

            trial_start_ms = None
            deg = 0.0

            next_cue_ms = _now_ms() + int(delay_ms)
        
        time.sleep(0.003)


def setup():
    port = find_arduino_port()
    if not port:
        raise RuntimeError("No Arduino detected")

    ser = None
    link = None

    try:
        ser = serial.Serial(port, BAUDRATE, timeout=0.05)
        time.sleep(2)

        if not ser.is_open:
            raise RuntimeError(f"{port} port is not open after initialization")
        
        print(f"Connected to {port} port\n", flush=True)
        try:
            print("Animal ID:  ", end="", flush=True)

            animal_raw = sys.stdin.readline()
            if animal_raw == "":
                raise EOFError
            
            animal_raw = animal_raw.rstrip("\n").upper()
        except KeyboardInterrupt:
            print('\n\nTerminated by KeyboardInterrupt')
            raise

        if not animal_raw:
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')
            sys.stdout.write('Animal ID:  DEV\n')
            sys.stdout.flush()
            
            animal_id = "DEV"
        else:
            animal_id = animal_raw

        dev_mode = (animal_id == "DEV")
        workbook_id = None

        if not dev_mode:
            validate_resources()
            animal_map = load_animal_map()
            validate_animal(animal_id, animal_map)

            workbook_id = _get_workbook_id(animal_id, animal_map)            

        valid_phases = {"1", "2"} | set(PHASE_CONFIG.keys())

        while True:
            try:
                phase_id = input("Training Phase:  ").strip()
            except KeyboardInterrupt:
                print('\n\nTerminated by KeyboardInterrupt')
                raise
            
            if phase_id in valid_phases:
                break

            print("Please enter a valid phase", flush=True)
        
        global LAST_SIDE, SIDE_STREAK
        LAST_SIDE = None
        SIDE_STREAK = 0

        link = ArduinoLink(ser)
        link.start()

        try:
            link.send_and_wait(phase_id)
        except Exception as e:
            link.close()
            raise RuntimeError(f"Failed to send training phase to Arduino: {e}") from e

        cursor = None
        type_1 = None
        side_1 = None

        if phase_id in PHASE_CONFIG:
            type_1 = choose_trial_type(trial_n=1, K=5)
            side_1 = choose_target_side(phase_id=phase_id)

            try:
                send_next_config(link, type_1, side_1)
            except Exception as e:
                print(f'[ERROR] Unhandled exception during initial phase hand-off: {e}')

            cursor = BCI(phase_id=phase_id,
                         evt_queue=EVT_QUEUE,
                         enc_queue=ENC_QUEUE,
                         display_idx=1,
                         fullscreen=True,
                         easy_threshold=15.0)
            
            cursor.update_config(type_1, side_1)
            cursor.start()

        session_data = SessionData(
            animal_id=animal_id,
            phase_id=phase_id,
            date_str=_get_date()
            )
        
        if not dev_mode:
            session_data.meta["workbook_id"] = workbook_id
        
        _ensure_session_tracking(session_data)
        if type_1 is not None and side_1 is not None:
            log_trial_config(session_data, trial_n=1, type=type_1, side=side_1)

        return link, session_data, cursor, dev_mode
    except Exception:
        if link is not None:
            try:
                link.close()
            except Exception:
                pass
        elif ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        
        raise


def main(link, session_data, cursor, dev_mode):
    do_calibration = str(session_data.meta["phase"]) not in {"1", "2"}

    K = 5
    N = 20
    trial_n = 0

    trial_stack = []
    calibrated = not do_calibration
    last_outcome = None

    trial_start_t = None
    recent_dt = deque(maxlen=INFO_WIN_SIZE)
    recent_outcomes = deque(maxlen=INFO_WIN_SIZE)

    dev_stop_evt = Event()
    dev_thread = None
    dev_state = {'trial_active': False}

    started = False
    t0 = None

    try:
        if dev_mode and cursor is not None:
            dev_stop_evt.clear()
            dev_thread = _dev_start(EVT_QUEUE, ENC_QUEUE, DEV_QUEUE, session_data, dev_state, dev_stop_evt, cursor)
        
        msg_q = DEV_QUEUE if (dev_mode and cursor is not None) else link.msg_q

        while link.ser and link.ser.is_open:
            try:
                typ, ts, payload = msg_q.get(timeout=0.05)
            except Empty:
                continue

            if not started:
                started = True
                t0 = time.time()
                session_data.meta["start_wall"] = datetime.now().isoformat(timespec="seconds")

                print(f"\nSession started at {datetime.now().strftime('%I:%M %p')}", flush=True)
                print()
                show_trial_header()

            if typ == "ERR":
                if isinstance(payload, BaseException):
                    print()
                    raise payload
                
                raise RuntimeError(f"\nArduinoLink reader error: {payload!r}")

            if typ == "END":
                break

            if typ == "EVT":
                p = str(payload)

                if not (dev_mode and cursor is not None):
                    session_data.add_evt(ts, p)

                try:
                    if not (dev_mode and cursor is not None):
                        EVT_QUEUE.put_nowait((ts, p))
                except Exception:
                    pass

                if p == 'cue':
                    trial_n += 1
                    last_outcome = None
                    trial_start_t = time.time()
                    dev_state['trial_active'] = True
                
                if p in {'hit', 'miss'}:
                    dev_state['trial_active'] = False

                    if last_outcome == p:
                        continue
                    last_outcome = p

                    dt = 0.0 if trial_start_t is None else max(0.0, time.time() - trial_start_t)                    
                    recent_dt.append(dt)
                    avg_dt = (sum(recent_dt) / len(recent_dt)) if recent_dt else 0.0

                    recent_outcomes.append(p)

                    n_hit = sum(1 for o in recent_outcomes if o == 'hit')
                    n_miss = sum(1 for o in recent_outcomes if o == 'miss')

                    show_trial_info(trial_n=trial_n, avg_dt=avg_dt, n_hit=n_hit, n_miss=n_miss)

                    if do_calibration:
                        trial_stack.insert(0, p)
                        if len(trial_stack) > N:
                            trial_stack.pop()
                        
                        if calibrated:
                            if len(trial_stack) >= N and is_early_exit(trial_stack, N):
                                end_session(link, '\nTerminated by early exit')
                                session_data.meta['aborted'] = True
                                break
                        else:
                            if len(trial_stack) >= N:
                                K, N, calibration_hits = update_easy_rate(session_data, trial_stack)
                                
                                session_data.meta['K2'] = K
                                calibrated = True
                                print(f'\nCalibration finished [hits={calibration_hits}/20, K={K}, N={N}]\n', flush=True)
                                show_trial_header()
                    
                    if cursor is not None:
                        next_trial_n = trial_n + 1
                        next_type = choose_trial_type(next_trial_n, K)
                        next_side = choose_target_side(session_data.meta['phase'])

                        if not dev_mode:
                            send_next_config(link, next_type, next_side)
                            log_trial_config(session_data, trial_n=next_trial_n, type=next_type, side=next_side)
                        
                        cursor.update_config(next_type, next_side)
            elif typ == "ENC":
                if dev_mode and cursor is not None:
                    continue

                p = str(payload)
                session_data.add_enc(ts, p)

                try:
                    ENC_QUEUE.put_nowait((ts, p))
                except Exception:
                    pass
    except KeyboardInterrupt:
        session_data.meta["aborted"] = True
        end_session(link, "\nTerminated by KeyboardInterrupt")
        raise
    finally:
        if dev_thread is not None:
            dev_stop_evt.set()

            try:
                dev_thread.join(timeout=2.0)
            except Exception:
                pass

        if session_data.meta["start_wall"] is None:
            session_data.meta["start_wall"] = datetime.now().isoformat(timespec="seconds")

        session_data.meta["end_wall"] = datetime.now().isoformat(timespec="seconds")

        if t0 is None:
            t0 = time.time()

        dt = int(max(0, time.time() - t0))
        session_data.meta["duration_sec"] = dt
        m, s = divmod(dt, 60)
        print(f"Session duration: {m}:{s:02d}", flush=True)

        if not dev_mode:
            try:
                send_email(session_data)
            except Exception as e:
                log_error(session_data.meta['animal'], session_data.meta['phase'], e)

    return session_data


if __name__ == "__main__":
    link = None
    session_data = None
    cursor = None
    dev_mode = True
    animal_id_for_log = "UNKNOWN"
    phase_id_for_log = "0"

    print()

    try:
        link, session_data, cursor, dev_mode = setup()

        if session_data is not None:
            animal_id_for_log = session_data.meta.get("animal", "UNKNOWN")
            phase_id_for_log = session_data.meta.get("phase", "0")

        main(link, session_data, cursor, dev_mode)

        if not dev_mode:
            if not session_data.any_data():
                raise ValueError('No data collected during session')

            print()

            save_ok = save_data(session_data)
            if not save_ok:
                print('[WARNING] Failed to save data using Google Sheets API', flush=True)
    except KeyboardInterrupt:
        pass
    except SystemExit as e:
        if getattr(e, "code", 0) not in {0, None}:
            log_error(animal_id_for_log, phase_id_for_log, e)

        raise
    except Exception as e:
        log_error(animal_id_for_log, phase_id_for_log, e)
        raise
    finally:
        if cursor is not None:
            try:
                cursor.stop(timeout=2.0)
            except Exception:
                pass

        if link is not None:
            try:
                link.close()
            except Exception:
                pass
        
        try:
            commit_error_log(animal_id_for_log, phase_id_for_log)
        except Exception:
            pass

        print()

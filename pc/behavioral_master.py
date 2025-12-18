import os
import sys
import warnings
import time
import json
import re
import shutil
import tempfile
import subprocess
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from queue import Queue, Empty
from threading import Thread, Event

import requests
import serial
import serial.tools.list_ports

from cursor_utils import cursor_fcn


# ---------------------------
# BASIC CONFIG
# ---------------------------
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
try:
    os.system('cls' if os.name == 'nt' else 'clear')
except Exception:
    pass

warnings.filterwarnings('ignore',
                        category=UserWarning,
                        message='pkg_resources is deprecated as an API.*')

BAUDRATE = 1_000_000
SESSION_END_STRING = 'S'

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_MAP_PATH = SCRIPT_DIR / 'animal_map.json'

EVT_QUEUE: 'Queue[tuple[str, str]]' = Queue()
ENC_QUEUE: 'Queue[tuple[str, str]]' = Queue()


# ---------------------------
# REPOSITORY CONFIG
# ---------------------------
REPO_URL = 'https://github.com/GonzalesLabVU/Behavior-BCI.git'
REPO_OWNER = 'GonzalesLabVU'
REPO_NAME = 'Behavior-BCI'
REPO_BRANCH = 'main'

REPO_DATA_DIR = Path('pc/data')
REPO_MAP_PATH = Path('pc/config/animal_map.json')

MAP_LOCK_REF = 'refs/heads/animal_map_write_flag'
LOCK_POLL_SEC = 30
LOCK_MAX_WAIT_SEC = 600


# ---------------------------
# EMAIL-TO-SMS
# ---------------------------
GATEWAYS = {
    'AT&T': 'txt.att.net',
    'Verizon': 'vtext.com'
    }

RECIPIENTS = [
    {'number': '2033213509', 'carrier': 'Verizon'},
    {'number': '5167849929', 'carrier': 'AT&T'}
    ]


def _phone_to_email(number, carrier):
    digits = ''.join(ch for ch in str(number) if ch.isdigit())

    if carrier not in GATEWAYS:
        raise ValueError(f'Unsupported carrier: {carrier}')
    
    return f'{digits}@{GATEWAYS[carrier]}'


def _format_sms(animal_id, phase_id, t_start, t_stop):
    today = datetime.now().date()
    m, _ = divmod(round(t_stop - t_start), 60)
    dt = datetime.fromtimestamp(t_stop)

    time_str = dt.strftime('%I:%M %p').lstrip('0')
    head_str = f'Session finished at {time_str}'
    animal_str = f'Animal:  {animal_id}'
    phase_str = f'Phase:  {phase_id}'
    date_str = f'Date:  {today.month}-{today.day}-{today.year}'
    runtime_str = f'Runtime:  {m} min'

    return f'{head_str}\n\n{animal_str}\n{phase_str}\n{date_str}\n{runtime_str}'


def send_sms(animal_id, phase_id, t_start, t_stop):
    smtp_user = os.environ['SMTP_USER']
    smtp_pass = os.environ['SMTP_APP_PASSWORD']

    text = _format_sms(animal_id, phase_id, t_start, t_stop)
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
        server.login(smtp_user, smtp_pass)

        for r in RECIPIENTS:
            to_addr = _phone_to_email(r['number'], r['carrier'])

            msg = EmailMessage()
            msg['From'] = smtp_user
            msg['To'] = to_addr
            msg['Subject'] = ''
            msg.set_content(text)

            server.send_message(msg)


# ---------------------------
# FORMAT HELPERS
# ---------------------------
def _get_date():
    return datetime.now().strftime('%m/%d/%Y')


def _get_ts():
    t = time.time()
    base = time.strftime("%H:%M:%S", time.localtime(t))
    ms = int((t - int(t)) * 1000)

    return f"{base}.{ms:03d}"


def _safe_slug(s):
    s = s.strip()
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', s)

    return s or 'UNKNOWN'


def json_filename(animal_id, phase_id, date_str):
    a = _safe_slug(str(animal_id))
    p = _safe_slug(str(phase_id))
    d = str(date_str).replace('/', '-')

    return f'{a}{p}_{d}.json'


# ---------------------------
# GITHUB
# ---------------------------
def _gh_headers():
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        raise RuntimeError('Missing GITHUB_TOKEN env var')
    
    header = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28'
        }
    return header


def _gh_api(url, method='GET', json_body=None):
    return requests.request(method, url, 
                            headers=_gh_headers(),
                            json=json_body,
                            timeout=20)


def _acquire_write_lock(lock_ref):
    base = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/git'
    url_create = f'{base}/refs'

    start = time.time()
    while True:
        r_main = _gh_api(f'{base}/ref/heads/{REPO_BRANCH}')
        if r_main.status_code != 200:
            raise RuntimeError(f'Failed to read branch ref: {r_main.status_code} {r_main.text}')
        
        sha = r_main.json()['object']['sha']
        resp = _gh_api(url_create,
                       method='POST',
                       json_body={'ref': lock_ref, 'sha': sha})
        
        if resp.status_code == 201:
            return
        
        if resp.status_code == 422:
            if time.time() - start > LOCK_MAX_WAIT_SEC:
                raise TimeoutError('Timed out waiting for write lock')
            
            time.sleep(LOCK_POLL_SEC)
            continue

        raise RuntimeError(f'Lock acquire failed: {resp.status_code} {resp.text}')


def _release_write_lock(lock_ref):
    base = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/git'
    url_delete = f'{base}/ref/{lock_ref.replace("refs/", "")}'

    resp = _gh_api(url_delete, method='DELETE')

    if resp.status_code in {204, 404}:
        return
    
    raise RuntimeError(f'Lock release failed: {resp.status_code} {resp.text}')


@contextmanager
def github_write_lock(lock_ref):
    _acquire_write_lock(lock_ref)

    try:
        yield
    finally:
        try:
            _release_write_lock(lock_ref)
        except Exception as e:
            print(f'[WARNING] Lock release failed: {e}', flush=True)


def _git_run(args, cwd, *, capture=False):
    env = os.environ.copy()
    env['GIT_TERMINAL_PROMPT'] = '0'

    return subprocess.run(['git', *args],
                          cwd=str(cwd),
                          check=True,
                          text=True,
                          capture_output=capture,
                          env=env)


@contextmanager
def _repo_clone(parent_dir):
    tmp = Path(tempfile.mkdtemp(prefix='repo_tmp_', dir=str(parent_dir)))

    token = os.environ.get('GITHUB_TOKEN')
    url = REPO_URL

    if token and url.startswith('https://github.com/'):
        url = url.replace('https://github.com/', f'https://x-access-token:{token}@github.com/', 1)
    
    try:
        _git_run(['clone', '--depth', '1', '--branch', REPO_BRANCH, url, str(tmp)], cwd=parent_dir)
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def repo_pull_file(repo_rel_path, local_abs_path):
    with _repo_clone(SCRIPT_DIR) as clone_dir:
        src = clone_dir / repo_rel_path
        if not src.exists():
            return False
        
        local_abs_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_abs_path)

        return True


def repo_push_file(local_abs_path, repo_rel_path, *, commit_msg, retries=3):
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        raise RuntimeError('Missing GITHUB_TOKEN env var')
    
    if not local_abs_path.exists():
        raise FileNotFoundError(f'Local file not found: {local_abs_path}')
    
    push_url = f"https://x-access-token:{token}@github.com/{REPO_OWNER}/{REPO_NAME}.git"

    with _repo_clone(SCRIPT_DIR) as clone_dir:
        target = clone_dir / repo_rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_abs_path, target)

        rel = str(target.relative_to(clone_dir))
        _git_run(['add', rel], cwd=clone_dir)

        st = _git_run(['status', '--porcelain', '--', rel], cwd=clone_dir, capture=True)
        if not st.stdout.strip():
            return False
        
        _git_run(['config', 'user.email', 'bci-bot@users.noreply.github.com'], cwd=clone_dir)
        _git_run(['config', 'user.name', 'Behavior-BCI bot'], cwd=clone_dir)
        _git_run(['commit', '-m', commit_msg], cwd=clone_dir)

        for attempt in range(1, retries + 1):
            try:
                _git_run(['pull', '--rebase', 'origin', REPO_BRANCH], cwd=clone_dir)
                _git_run(['push', push_url, f'HEAD:{REPO_BRANCH}'], cwd=clone_dir)

                return True
            except subprocess.CalledProcessError:
                if attempt >= retries:
                    raise

                time.sleep(2.0)
    
    return False


# ---------------------------
# ANIMAL MAP
# ---------------------------
def load_animal_map(path=LOCAL_MAP_PATH):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            raise ValueError('animal_map.json must map cohort -> list of animals')
    
        cohorts = {c: set(a) for (c, a) in data.items()}
    except FileNotFoundError:
        cohorts = {}
    except Exception as e:
        print(f'[WARNING] Failed to load {path}: {e}\nUsing empty map', flush=True)
        cohorts = {}
    
    animal_to_cohort = {animal: cohort for (cohort, animals) in cohorts.items() for animal in animals}

    return cohorts, animal_to_cohort


def save_animal_map(cohorts, path=LOCAL_MAP_PATH):
    data = {cohort: sorted(list(animals)) for (cohort, animals) in cohorts.items()}

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def pull_animal_map():
    try:
        ok = repo_pull_file(REPO_MAP_PATH, LOCAL_MAP_PATH)
        if ok:
            print(f'[git] Pulled {REPO_MAP_PATH} -> {LOCAL_MAP_PATH}', flush=True)
    except Exception as e:
        print(f'[WARNING] Failed to pull animal_map.json: {e}', flush=True)


def assign_animal_to_cohort(animal_id):
    animal_id = animal_id.upper().strip()
    if not animal_id:
        return False
    
    _, animal_to_cohort = load_animal_map()

    if animal_id in animal_to_cohort:
        print(f'Animal {animal_id} is already assigned to {animal_to_cohort[animal_id]}', flush=True)
        return True
    
    print(f'Animal {animal_id} is not currently assigned to a cohort', flush=True)
    choice = input('Assign it now? [y/N]:  ').strip().lower()
    if choice not in {'y', 'yes'}:
        return False
    
    cohort_id = input('Enter cohort number (leave blank to cancel):  ').strip()
    if not cohort_id:
        print('Cohort assignment cancelled', flush=True)
        return False
    
    cohort_name = f'cohort{cohort_id}'

    try:
        with github_write_lock(MAP_LOCK_REF):
            repo_pull_file(REPO_MAP_PATH, LOCAL_MAP_PATH)
            cohorts, animal_to_cohort = load_animal_map()

            if animal_id in animal_to_cohort:
                print(f'Animal {animal_id} is already assigned to {animal_to_cohort[animal_id]} (no changes)', flush=True)
                return True
            
            if cohort_name not in cohorts:
                create = input(f'{cohort_name} does not exist. Create it? [y/N]:  ').strip().lower()
                if create not in {'y', 'yes'}:
                    print('Cohort not created (assignment cancelled)', flush=True)
                    return False
                
                cohorts[cohort_name] = set()
            
            cohorts[cohort_name].add(animal_id)
            save_animal_map(cohorts, LOCAL_MAP_PATH)

            commit_msg = 'Update cohort assignments in animal_map.json'
            pushed = repo_push_file(LOCAL_MAP_PATH, REPO_MAP_PATH, commit_msg=commit_msg)
            if pushed:
                print(f'[git] Updated {REPO_MAP_PATH}', flush=True)
            else:
                print('[git] No changes to push for animal_map.json')

        return True
    except Exception as e:
        print(f'[WARNING] Failed to update animal_map.json: {e}', flush=True)
        return False


# ---------------------------
# SERIAL
# ---------------------------
def _decode_line(raw):
    if not raw:
        return ''
    
    try:
        return raw.decode('utf-8', errors='strict').strip()
    except UnicodeDecodeError:
        return raw.decode('latin1', errors='ignore').strip()
    

class ArduinoLink:
    def __init__(self, ser):
        self.ser = ser
        self.stop_evt = Event()
        self.ack_evt = Event()
        self.msg_q: "Queue[tuple[str, str, object]]" = Queue()
        self._reader = Thread(target=self._reader_loop, daemon=True)

    def start(self):
        self._reader.start()

    def close(self):
        self.stop_evt.set()
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

    def send_and_wait(self, text, timeout=2.0):
        self.ack_evt.clear()
        self.ser.write((text.strip() + "\n").encode("utf-8"))
        self.ser.flush()
        if not self.ack_evt.wait(timeout=timeout):
            raise TimeoutError(f"No ACK after sending: {text!r}")

    def send(self, text):
        self.ser.write((text.strip() + "\n").encode("utf-8"))
        self.ser.flush()

    def _reader_loop(self):
        try:
            while not self.stop_evt.is_set() and self.ser and self.ser.is_open:
                raw = self.ser.readline()
                if not raw:
                    continue

                line = _decode_line(raw)
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


def _find_arduino_port():
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        dsc = (port.description or '').lower()

        if 'arduino' in dsc or 'usb serial' in dsc:
            return port.device
    
    return None


# ---------------------------
# DATA
# ---------------------------
class SessionData:
    def __init__(self, animal_id, phase_id, date_str):
        self.meta = {
            'date': date_str,
            'animal': animal_id,
            'phase': phase_id,
            'start_wall': None,
            'end_wall': None,
            'duration_sec': None,
            'aborted': False
            }
        self.evt = {'timestamps': [], 'values': []}
        self.enc = {'timestamps': [], 'values': []}
    

    def add_evt(self, ts, payload):
        self.evt['timestamps'].append(ts)
        self.evt['values'].append(payload)
    

    def add_enc(self, ts, payload):
        self.enc['timestamps'].append(ts)
        self.enc['values'].append(payload)
    

    def any_data(self):
        return bool(self.evt['timestamps']) or bool(self.enc['timestamps'])
    

    def to_dict(self):
        return {'meta': self.meta, 'event': self.evt, 'encoder': self.enc}


def write_json(session_data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(session_data.to_dict(), f, indent=4)


def push_json(path, *, repo_filename):
    commit_msg = f'Add session {repo_filename} ({datetime.now().strftime("%Y-%m-%d %H:%M:%S")})'
    repo_path = REPO_DATA_DIR / repo_filename

    repo_push_file(path, repo_path, commit_msg=commit_msg)
    print(f'[git] Pushed {repo_path} to {REPO_BRANCH}', flush=True)


# ---------------------------
# CALIBRATION / EARLY EXIT
# ---------------------------
def _set_easy_rate(link, session_data, trial_stack):
    n_hits = sum(1 for x in trial_stack if x == 'hit')

    if n_hits < 10:
        K = 3
    elif n_hits == 10:
        K = 5
    else:
        K = 7
    
    N = 4 * K

    trial_stack.clear()
    link.send_and_wait(str(K))
    session_data.add_evt(_get_ts(), f'setK {K}')

    return K, N, n_hits


def _is_early_exit(trial_stack, N):
    if len(trial_stack) < N:
        return False
    
    n_hits = sum(1 for x in trial_stack[:N] if x == 'hit')

    return n_hits < (N / 4)


def _terminate_session(link, msg):
    try:
        link.send('E')

        deadline = time.time() + 2.0
        while time.time() < deadline:
            try:
                typ, _, _ = link.msg_q.get(timeout=0.05)
                if typ == 'END':
                    print(f'\n{msg}\n', flush=True)
                    return
            except Empty:
                pass
    finally:
        link.close()


# ---------------------------
# TOP LEVEL
# ---------------------------
def setup():
    pull_animal_map()
    cohorts, animal_to_cohort = load_animal_map()

    port = _find_arduino_port()
    if not port:
        print('No Arduino detected', flush=True)
        sys.exit(1)
    
    try:
        ser = serial.Serial(port, BAUDRATE, timeout=0.05)
        time.sleep(2)
        print(f'Connected to {port}', flush=True)
    except serial.SerialException as e:
        print(f'Serial error: {e}', flush=True)
        sys.exit(1)
    
    dev_mode = False

    while True:
        try:
            animal_raw = input('Animal ID:  ').strip().upper()
            if animal_raw == '':
                dev_mode = True
                animal_id = 'DEV'
                print('\nDEV MODE\n', flush=True)
                break

            animal_id = animal_raw
            if animal_id in animal_to_cohort:
                break

            if assign_animal_to_cohort(animal_id):
                cohorts, animal_to_cohort = load_animal_map()
                if animal_id in animal_to_cohort:
                    break
            
            print('Please enter a valid ID', flush=True)
        except KeyboardInterrupt:
            print('\nProcess terminated by user', flush=True)
            sys.exit(0)
    
    while True:
        try:
            phase_id = input('Training Phase:  ').strip()
            if phase_id in {'1', '2', '3', '4', '5'}:
                break

            print('Please enter a valid phase', flush=True)
        except KeyboardInterrupt:
            print('\nProcess terminated by user', flush=True)
            sys.exit(0)
    
    link = ArduinoLink(ser)
    link.start()

    try:
        link.send_and_wait(phase_id)
    except Exception as e:
        print(f'Failed to send training phase to Arduino: {e}', flush=True)
        link.close()
        sys.exit(1)
    
    cursor_thread = None

    if str(phase_id) not in {'1', '2'}:
        threshold = {'3': 30.0, '4': 60.0, '5': 90.0}[str(phase_id)]

        cursor_thread = Thread(
            target=cursor_fcn,
            args=(threshold, EVT_QUEUE, ENC_QUEUE),
            kwargs={'display_index': 1, 'fullscreen': True},
            daemon=True
            )
        cursor_thread.start()
    
    date_str = _get_date()
    fname = json_filename(animal_id, phase_id, date_str)
    path = SCRIPT_DIR / fname

    session_data = SessionData(
        animal_id=animal_id,
        phase_id=phase_id,
        date_str=date_str)
    
    return link, session_data, dev_mode, cursor_thread, path, fname


def main(link, session_data):
    do_calibration = str(session_data.meta['phase']) not in {'1', '2'}

    K = 5
    N = 20

    try:
        link.send_and_wait(str(K))
    except Exception as e:
        print(f'Failed to send initial K to Arduino: {e}', flush=True)
        link.close()
        sys.exit(1)
    
    session_data.add_evt(_get_ts(), f'setK {K}')

    trial_stack = []
    calibrated = not do_calibration
    last_outcome = None

    started = False
    t0 = None

    try:
        while link.ser and link.ser.is_open:
            try:
                typ, ts, payload = link.msg_q.get(timeout=0.05)
            except Empty:
                continue

            if not started:
                started = True
                t0 = time.time()
                session_data.meta['start_wall'] = datetime.now().isoformat(timespec='seconds')
                print(f'Session start: {datetime.now().strftime("%I:%M %p")}', flush=True)
                print('Running session protocol...', flush=True)
            
            if typ == 'ERR':
                continue

            if typ == 'END':
                break

            if typ == 'EVT':
                p = str(payload)
                session_data.add_evt(ts, p)

                try:
                    EVT_QUEUE.put_nowait((ts, p))
                except Exception:
                    pass

                if do_calibration and p in {'hit', 'miss'}:
                    if last_outcome == p:
                        continue

                    last_outcome = p

                    trial_stack.insert(0, p)
                    if len(trial_stack) > N:
                        trial_stack.pop()

                    if calibrated:
                        if len(trial_stack) >= N and _is_early_exit(trial_stack, N):
                            _terminate_session(link, 'Session terminated by early exit')
                            session_data.meta['aborted'] = True
                            break
                    else:
                        if len(trial_stack) >= N:
                            K, N, calibration_hits = _set_easy_rate(link, session_data, trial_stack)
                            calibrated = True
                            print(f'Calibration finished [hits={calibration_hits}/20, K={K}, N={N}]', flush=True)
            elif typ == 'ENC':
                p = str(payload)
                session_data.add_enc(ts, p)

                try:
                    ENC_QUEUE.put_nowait((ts, p))
                except Exception:
                    pass
            else:
                pass
    except KeyboardInterrupt:
        session_data.meta['aborted'] = True
        _terminate_session(link, 'Session terminated by user')
    finally:
        try:
            link.close()
        except Exception:
            pass

        if session_data.meta['start_wall'] is None:
            session_data.meta['start_wall'] = datetime.now().isoformat(timespec='seconds')
        
        session_data.meta['end_wall'] = datetime.now().isoformat(timespec='seconds')

        if t0 is None:
            t0 = time.time()
        
        dt = int(max(0, time.time() - t0))
        session_data.meta['duration_sec'] = dt
        m, s = divmod(dt, 60)
        print(f'Session duration: {m}:{s:02d}', flush=True)
    
    return session_data


if __name__ == '__main__':
    link = None
    session_data = None
    cursor_thread = None
    dev_mode = True

    path = None
    repo_filename = None

    try:
        link, session_data, dev_mode, cursor_thread, path, repo_filename = setup()
        session_data = main(link, session_data)

        if session_data and session_data.any_data():
            write_json(session_data, path)
            print(f'Saved session data to: {path}', flush=True)

            if not dev_mode:
                push_json(path, repo_filename=repo_filename)
    except SystemExit:
        if session_data and session_data.any_data() and path and repo_filename:
            try:
                write_json(session_data, path)
                print(f'Saved session data to: {path}', flush=True)

                if not dev_mode:
                    push_json(path, repo_filename=repo_filename)
            except Exception as e:
                print(f'[WARNING] Failed to save/push on SystemExit: {e}', flush=True)
        
        raise
    except Exception as e:
        print(f'[ERROR] Unhandled exception: {e}', flush=True)

        if session_data and session_data.any_data() and path:
            try:
                write_json(session_data, path)
                print(f'Saved session data to: {path}', flush=True)
            except Exception as e2:
                print(f'[WARNING] Failed to save after exception: {e2}', flush=True)
    finally:
        if cursor_thread is not None:
            cursor_thread.join(timeout=2.0)
        
        if link is not None:
            try:
                link.close()
            except Exception:
                pass

        try:
            if session_data is not None:
                if session_data.meta.get('start_wall'):
                    t_start = datetime.fromisoformat(session_data.meta['start_wall']).timestamp()
                else:
                    t_start = time.time()
                
                if session_data.meta.get('end_wall'):
                    t_stop = datetime.fromisoformat(session_data.meta['end_wall']).timestamp()
                else:
                    t_stop = time.time()
                
                animal_id = session_data.meta.get('animal', 'UNKNOWN')
                phase_id = session_data.meta.get('phase', 'UNKNOWN')

                send_sms(animal_id, phase_id, t_start, t_stop)
        except Exception as sms_e:
            print(f'[WARNING] Failed to send SMS notification: {sms_e}', flush=True)

import sys
import os
os.system('cls')
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='pkg_resources is deprecated as an API.*')

import time
from datetime import datetime
from collections import deque
from queue import Queue
import serial
import serial.tools.list_ports
from threading import Thread
from openpyxl import load_workbook, Workbook
import json
from cursor_utils import cursor_fcn

BAUDRATE = 1000000

SESSION_END_STRING = "S"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANIMAL_MAP_PATH = os.path.join(SCRIPT_DIR, 'animal_map.json')

EVT_QUEUE: "Queue[tuple[str, str]]" = Queue()
ENC_QUEUE: "Queue[tuple[str, str]]" = Queue()

RAW_LOG = deque()


# ----- STARTUP -----
def _load_animal_map(path=ANIMAL_MAP_PATH):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("'animal_map.json' file must contain an object mapping cohort -> list of animals")
    
        cohorts = {name: set(animals) for (name, animals) in data.items()}
    except FileNotFoundError:
        cohorts = {}
    except Exception as e:
        print(f'Warning: failed to load {path}: {e}. Starting with empty map')
        cohorts = {}
    
    return cohorts


def _save_animal_map(cohorts, path=ANIMAL_MAP_PATH):
    data = {name: sorted(list(animals)) for (name, animals) in cohorts.items()}

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


COHORTS = _load_animal_map()
ANIMAL_TO_COHORT = {
    animal: cohort_name
    for (cohort_name, animals) in COHORTS.items()
    for animal in animals
    }


# ----- HELPERS -----
def _find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        dsc = (port.description or "").lower()

        if 'arduino' in dsc or 'usb serial' in dsc:
            return port.device
    return None


def _update_cohorts(animal_id):
    global COHORTS, ANIMAL_TO_COHORT

    print(f'\nAnimal {animal_id} is not currently assigned to a cohort.')
    choice = input('Would you like to assign it now? [y/N]:  ').strip().lower()

    if choice not in {'y', 'yes'}:
        return False
    
    while True:
        cohort_id = input('\nEnter cohort number for this animal (leave blank to cancel):  ')

        if not cohort_id:
            print('Cohort assignment cancelled\n')
            return False
        else:
            cohort_name = 'cohort' + cohort_id
        
        if cohort_name not in COHORTS:
            create = input(f'\nCohort {cohort_id} does not currently exist. Create it now? [y/N]:  ').strip().lower()

            if create not in {'y', 'yes'}:
                print('Cohort not created\n')
                continue

            COHORTS[cohort_name] = set()
        
        old_cohort = ANIMAL_TO_COHORT.get(animal_id)
        old_cohort_animals_before = set(COHORTS.get(old_cohort, set())) if old_cohort else None

        new_cohort_animals_before = set(COHORTS.get(cohort_name, set()))

        if old_cohort and old_cohort != cohort_name:
            COHORTS[old_cohort].discard(animal_id)
        
        COHORTS[cohort_name].add(animal_id)
        ANIMAL_TO_COHORT[animal_id] = cohort_name

        _save_animal_map(COHORTS)
        print(f'\nAssigned animal {animal_id} to cohort {cohort_id}\n')

        if old_cohort and old_cohort != cohort_name and old_cohort_animals_before is not None:
            _update_workbook_filename(old_cohort, old_cohort_animals_before, COHORTS.get(old_cohort, set()))

        _update_workbook_filename(cohort_name, new_cohort_animals_before, COHORTS.get(cohort_name, set()))
        
        return True


def _open_or_create_workbook(path):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    
    if os.path.exists(path):
        try:
            wb = load_workbook(path)
        except Exception:
            wb = Workbook()
    else:
        wb = Workbook()
    
    _validate_sheets(wb)
    return wb


def _is_sheet_empty(ws):
    for row in ws.iter_rows():
        for cell in row:
            if cell.value not in (None, ''):
                return False
    
    return True


def _validate_sheets(wb):
    if 'Event' not in wb.sheetnames:
        wb.create_sheet('Event')
    if 'Encoder' not in wb.sheetnames:
        wb.create_sheet('Encoder')
    
    for name in list(wb.sheetnames):
        if name in ('Event', 'Encoder'):
            continue

        ws = wb[name]
        if _is_sheet_empty(ws):
            wb.remove(ws)


def _find_workbook_for_animal(animal_id):
    animal_id = animal_id.upper()

    cohort_name = ANIMAL_TO_COHORT.get(animal_id)
    if cohort_name is None:
        filename = 'backup_data.xlsx'
        print(f'\n[WARNING] Unknown animal {animal_id} or cohort {cohort_name}\n')
    else:
        animals_in_cohort = COHORTS.get(cohort_name, set())
        if animals_in_cohort:
            label = ''.join(sorted(animals_in_cohort))
        else:
            label = cohort_name
        
        filename = f'{label}_data.xlsx'
    
    return os.path.join(SCRIPT_DIR, filename)


def _update_workbook_filename(cohort_name, old_animals, new_animals):
    old_animals = set(old_animals or [])
    new_animals = set(new_animals or [])

    if not old_animals:
        return
    
    old_label = ''.join(sorted(old_animals))
    new_label = ''.join(sorted(new_animals)) if new_animals else cohort_name

    if old_label == new_label:
        return
    
    old_path = os.path.join(SCRIPT_DIR, f'{old_label}_data.xlsx')
    new_path = os.path.join(SCRIPT_DIR, f'{new_label}_data.xlsx')

    if not os.path.exists(old_path):
        return
    
    if os.path.exists(new_path):
        timestamp = int(time.time())
        alt_path = os.path.join(SCRIPT_DIR, f'{new_label}_data_{timestamp}.xlsx')

        new_base = os.path.basename(new_path)
        alt_base = os.path.basename(alt_path)

        os.rename(old_path, alt_path)
        print(f'\n[WARNING] {new_base} exists in the current folder; renamed old workbook to {alt_base}\n', flush=True)
    else:
        new_base = os.path.basename(new_path)
        old_base = os.path.basename(old_path)

        os.rename(old_path, new_path)
        print(f'\nRenamed workbook {old_base} to {new_base}\n', flush=True)


def _write_to_sheet(ws, date_str, animal, phase, data):
    col = 1
    while ws.cell(row=1, column=col).value is not None:
        col += 2
    
    animal_str = 'Animal ' + animal
    phase_str = 'Phase ' + phase
    
    ws.cell(row=1, column=col, value=date_str)
    ws.cell(row=2, column=col, value=animal_str)
    ws.cell(row=2, column=col+1, value=phase_str)

    for r, (ts, payload) in enumerate(data, start=4):
        try:
            val = float(payload)
        except Exception:
            val = str(payload)
        
        ws.cell(row=r, column=col, value=ts)
        ws.cell(row=r, column=col+1, value=val)


def _save_data(data, animal_id, phase_id):
    if animal_id not in ANIMAL_TO_COHORT:
        raise ValueError(f'Unknown animal ID: {animal_id}\n')
    
    save_path = _find_workbook_for_animal(animal_id)

    session_date = datetime.now().date()
    session_date_str = f'{session_date.month}/{session_date.day}/{session_date.year}'

    event_data = data['EVT']
    encoder_data = data['ENC']

    wb = _open_or_create_workbook(save_path)

    if event_data:
        if 'Event' in wb.sheetnames:
            ws = wb['Event']
        else:
            ws = wb.create_sheet('Event')
        
        _write_to_sheet(ws, session_date_str, animal_id, phase_id, event_data)
    
    if encoder_data:
        if 'Encoder' in wb.sheetnames:
            ws = wb['Encoder']
        else:
            ws = wb.create_sheet('Encoder')
        
        _write_to_sheet(ws, session_date_str, animal_id, phase_id, encoder_data)
    
    try:
        wb.save(save_path)
        print(f'\nSaved data to {save_path}\n', flush=True)
    except PermissionError:
        base, ext = os.path.splitext(os.path.basename(save_path))
        alt = os.path.join(SCRIPT_DIR, f'{base}_{int(time.time())}{ext}')

        wb.save(alt)
        print(f'\nWorkbook was locked; saved data to {alt}', flush=True)
    
    if len(RAW_LOG) > 0:
        t_lick = [t for (t, e) in event_data if e == 'lick']
        t_hit = [t for (t, e) in event_data if e == 'hit']

        log = {
            'cap': {
                'time': [t for (t, _) in RAW_LOG],
                'value': [float(v) for (_, v) in RAW_LOG]
                },
            't_lick': t_lick,
            't_hit': t_hit
            }
        
        fname = f'Animal_{animal_id}_raw_cap.json'
        path = os.path.join(SCRIPT_DIR, fname)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=4)


def _save_on_error(session_data, animal_id=None, phase_id=None):
    if not isinstance(session_data, dict):
        return
    
    try:
        evt = list(session_data.get("EVT", []))
        enc = list(session_data.get("ENC", []))
    except Exception:
        return
    
    if not evt and not enc:
        print('\nNo data collected\n')
        return
    
    backup_path = os.path.join(SCRIPT_DIR, 'backup_data.xlsx')
    wb = _open_or_create_workbook(backup_path)

    session_date = datetime.now().date()
    session_date_str = f'{session_date.month}/{session_date.day}/{session_date.year}'

    if evt:
        if 'Event' in wb.sheetnames:
            ws = wb['Event']
        else:
            ws = wb.create_sheet('Event')
        
        _write_to_sheet(ws, session_date_str,
                        animal_id if animal_id is not None else '[unknown]',
                        phase_id if phase_id is not None else '[unknown]',
                        evt)
    
    if enc:
        if 'Encoder' in wb.sheetnames:
            ws = wb['Encoder']
        else:
            ws = wb.create_sheet('Encoder')
        
        _write_to_sheet(ws, session_date_str,
                        animal_id if animal_id is not None else '[unknown]',
                        phase_id if phase_id is not None else '[unknown]',
                        enc)

    try:
        wb.save(backup_path)
        print(f'\n[ERROR] Saved session data to {backup_path}\n', flush=True)
    except Exception as e:
        print(f'\n[ERROR] Failed to save session data: {e}\n', flush=True)
    
    if len(RAW_LOG) > 0:
        t_lick = [t for (t, e) in evt if e == 'lick']
        t_hit = [t for (t, e) in evt if e == 'hit']

        log = {
            'cap': {
                'time': [t for (t, _) in RAW_LOG],
                'value': [float(v) for (_, v) in RAW_LOG]
                },
            't_lick': t_lick,
            't_hit': t_hit
            }
        
        if animal_id is not None:
            fname = f'Animal_{animal_id}_raw_cap.json'
        else:
            fname = 'Animal_unknown_raw_cap.json'
        
        path = os.path.join(SCRIPT_DIR, fname)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=4)


def _now_ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _decode_line(raw):
    if not raw:
        return ""
    
    try:
        return raw.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError:
        return raw.decode("latin1", errors="ignore").strip()


def _handle_line(line, session_data, N=None, trial_stack=None):
    if not line:
        return None
    
    if line == SESSION_END_STRING:
        return 'S'
    
    if line.startswith("[EVT]"):
        parts = line.split("]", 1)
        payload = parts[1].strip() if len(parts) > 1 else ""
        ts = _now_ts()

        session_data["EVT"].append([ts, payload])
        EVT_QUEUE.put((ts, payload))

        if payload in {"hit", "miss"}:
            if len(session_data['EVT']) >= 2:
                prev_payload = session_data['EVT'][-2][1]
                if prev_payload in {"hit", "miss"}:
                    return None
            
            if trial_stack is not None:
                if len(trial_stack) >= N:
                    trial_stack.pop()
                trial_stack.appendleft(payload)

        return None
    
    if line.startswith("[ENC]"):
        parts = line.split("]", 1)
        payload = parts[1].strip() if len(parts) > 1 else ""
        ts = _now_ts()
        
        session_data["ENC"].append([ts, payload])
        ENC_QUEUE.put((ts, payload))

        return None
    
    if line.startswith("[RAW]"):
        parts = line.split("]", 1)
        payload = parts[1].strip() if len(parts) > 1 else ""
        ts = _now_ts()

        RAW_LOG.append([ts, payload])

        return None

    return None


def _wait_for_ack(ser, session_data, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ser.in_waiting > 0:
            raw = ser.readline()
            line = _decode_line(raw)
            if not line:
                continue

            if line == 'A':
                return True
            
            _handle_line(line, session_data)
        else:
            time.sleep(0.005)
    
    return False


def _send_to_serial(ser, session_data, text, timeout=2.0):
    ser.write((text.strip() + "\n").encode("utf-8"))
    ser.flush()

    ack_received = _wait_for_ack(ser, session_data, timeout=timeout)
    if not ack_received:
        raise TimeoutError(f'No ACK after sending: {text!r}')


def _set_easy_rate(ser, session_data, trial_stack):
    n_hits = sum(1 for x in trial_stack if x == 'hit')
    if n_hits < 10:
        K = 3
    elif n_hits == 10:
        K = 5
    else:
        K = 7
    N = 4 * K

    trial_stack.clear()

    try:
        _send_to_serial(ser, session_data, str(K))
    except Exception as e:
        print(f"Failed to send calibrated easy-trial rate to Arduino: {e}\n")
        ser.close()
        sys.exit(1)
    
    EVT_QUEUE.put((_now_ts(), f'setK {K}'))

    return K, N, n_hits


def _is_early_exit(trial_stack, N):
    if len(trial_stack) < N:
        return False

    n_hits = sum(1 for x in trial_stack if x == 'hit')
    return n_hits < 7


def _print_window(trial_stack, N, trial_index):
    if N is None or N <= 0:
        return
    
    if len(trial_stack) < N:
        return
    
    n_hits = sum(1 for x in trial_stack if x == 'hit')
    hit_rate = 100.0 * (n_hits / N)

    end_trial = trial_index
    start_trial = (end_trial - N + 1)

    print(f'(Trials {start_trial}-{end_trial}):  {n_hits}/{N} ({hit_rate:.1f}%)', flush=True)


def _terminate_session(ser, session_data, msg=None):
    try:
        ser.write(("E\n").encode("utf-8"))
        ser.flush()

        end_deadline = time.time() + 2.0
        while time.time() < end_deadline:
            if ser.in_waiting > 0:
                raw = ser.readline()
                line = _decode_line(raw)
                flag = _handle_line(line, session_data)

                if flag == 'S':
                    print(f"\n\n{msg or 'Session terminated'}\n")
                    break
            else:
                time.sleep(0.01)
    except Exception as e:
        print(f'\n\nFailed to terminate session: {e}\n')
    finally:
        if ser and ser.is_open:
            ser.close()


# -------- TOP LEVEL --------
def setup(session_data):
    # initialize serial connection with Arduino
    port = _find_arduino_port()
    if not port:
        print("No Arduino detected\n", flush=True)
        sys.exit(1)
    
    try:
        ser = serial.Serial(port, BAUDRATE, timeout=0.05)
        time.sleep(2)
        print(f"\nConnected to {port}\n", flush=True)
    except serial.SerialException as e:
        print(f"\nSerial error: {e}\n")
        sys.exit(1)
    
    # ask for animal ID
    dev_mode = False

    while True:
        try:
            animal_id_raw = input("\nAnimal ID: ").strip().upper()

            if animal_id_raw == "":
                animal_id = 'DEV'
                dev_mode = True
                print("\n------------\n  DEV MODE \n------------\n")
                break
            else:
                animal_id = animal_id_raw

            if animal_id and animal_id in ANIMAL_TO_COHORT:
                break

            if animal_id:
                if _update_cohorts(animal_id):
                    break

            print("Please enter a valid ID\n")
        except KeyboardInterrupt:
            print('\nProcess terminated by user\n')
            sys.exit(0)
    
    # ask for phase ID
    while True:
        try:
            phase_id = input("Training Phase: ").strip()

            if phase_id and phase_id in {"1", "2", "3", "4", "5"}:
                break
            print("Please enter a valid training phase\n")
        except KeyboardInterrupt:
            print('\nProcess terminated by user\n')
            sys.exit(0)
    
    # send phase ID to Arduino
    try:
        _send_to_serial(ser, session_data, phase_id)
    except Exception as e:
        print(f"Failed to send training phase to Arduino: {e}\n")
        ser.close()
        sys.exit(1)
    
    # start cursor thread if needed
    cursor_thread = None

    if str(phase_id) not in {'1', '2'}:
        threshold = {
            '3': 30.0,
            '4': 60.0,
            '5': 90.0
            }[str(phase_id)]
        cursor_thread = Thread(
            target=cursor_fcn,
            args=(threshold, EVT_QUEUE, ENC_QUEUE),
            kwargs={'display_index': 1, 'fullscreen': True},
            daemon=True
            )
        
        cursor_thread.start()
    
    return ser, animal_id, phase_id, session_data, cursor_thread, dev_mode


def main(ser, session_data, phase_id):
    K = 5
    N = 20

    try:
        _send_to_serial(ser, session_data, str(K))
    except Exception as e:
        print(f"Failed to send initial easy-trial rate to Arduino: {e}\n")
        ser.close()
        sys.exit(1)
    
    EVT_QUEUE.put((_now_ts(), f'setK {K}'))

    trial_stack = deque()
    calibrated = False
    finished = False
    t_start = None
    t_stop = None
    trial_idx = 0

    try:
        while ser and ser.is_open and not finished:
            while ser.in_waiting > 0:
                if t_start is None:
                    t_start = time.time()

                    print(f"\nSession start time: {datetime.now().strftime('%I:%M %p')}")
                    print("\nRunning session protocol...", end="", flush=True)

                raw = ser.readline()
                line = _decode_line(raw)
                flag = _handle_line(line, session_data, N, trial_stack)

                if phase_id not in {'1', '2'} and calibrated:
                    trial_idx += 1
                    _print_window(trial_stack, N, trial_idx)

                if phase_id not in {'1', '2'}:
                    if not calibrated:
                        if len(trial_stack) >= N:
                            K, N, calibration_hits = _set_easy_rate(ser, session_data, trial_stack)
                            calibrated = True

                            print(f'\nCalibration period finished [hits={calibration_hits}/20, K={K}, N={N}]\n', flush=True)
                    else:
                        if _is_early_exit(trial_stack, N):
                            _terminate_session(ser, session_data, msg='Session terminated by early exit')

                            t_stop = time.time()
                            finished = True
                            break
                
                if flag == 'S':
                    t_stop = time.time()
                    finished = True
                    break
            
            time.sleep(0.01)
    finally:
        if ser and ser.is_open:
            ser.close()
    
    if t_stop is None:
        t_stop = time.time()
    
    if t_start is None:
        t_start = t_stop

    elapsed = int(t_stop - t_start)
    m, s = divmod(elapsed, 60)
    print(f'\nSession duration: {m:02d}:{s:02d}')

    return session_data


if __name__ == "__main__":
    session_data = {"EVT": deque(), "ENC": deque()}
    animal_id = None
    phase_id = None
    dev_mode = True
    cursor_thread = None
    ser = None

    try:
        ser, animal_id, phase_id, session_data, cursor_thread, dev_mode = setup(session_data)
        session_data = main(ser, session_data, phase_id)

        try:
            import pygame

            if pygame.get_init():
                pygame.event.post(pygame.event.Event(pygame.QUIT))
        except Exception:
            pass

        if cursor_thread is not None:
            cursor_thread.join(timeout=2.0)
        
        if not dev_mode and animal_id in ANIMAL_TO_COHORT:
            _save_data(session_data, animal_id, phase_id)
            
        # [plotting logic]

    except KeyboardInterrupt:
        try:
            if ser is not None and ser.is_open:
                _terminate_session(ser, session_data, msg='Session terminated by keyboard interrupt')
        except Exception as e:
            print(f'\n[WARNING] Failed to terminate session cleanly after KeyboardInterrupt: {e}\n')

        _save_on_error(session_data, animal_id, phase_id)

    except Exception as e:
        print(f'\n[ERROR] Unhandled exception: {e}\n')
        _save_on_error(session_data, animal_id, phase_id)

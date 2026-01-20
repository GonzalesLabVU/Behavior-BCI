import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import warnings
warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API')

import time
import threading
from queue import Queue, Empty
import pygame as pg

# ---------------------------
# BASIC CONFIG
# ---------------------------
MIN_DEG = -90.0
MAX_DEG = +90.0
DELAY_MS = 1000

BLUE = (50, 50, 255)
BLACK = (0, 0, 0)

PHASE_CONFIG = {
    '3': {'bidirectional': True, 'threshold': 30.0},
    '4': {'bidirectional': True, 'threshold': 60.0},
    '5': {'bidirectional': True, 'threshold': 90.0},
    '6': {'bidirectional': False, 'threshold': 30.0},
    '7': {'bidirectional': False, 'threshold': 60.0},
    '8': {'bidirectional': False, 'threshold': 90.0},
    }

TRIAL_CONFIG = (False, "B")
TRIAL_LOCK = threading.Lock()

ABORT_EVT = threading.Event()


# ---------------------------
# HELPERS
# ---------------------------
def _parse_event(payload):
    if not payload:
        return ""
    
    return str(payload).strip().split()[0].lower()


def cursor_fcn(threshold, evt_queue, enc_queue, *, display_idx=None, fullscreen=True, easy_threshold=15.0, stop_evt=None, blackout_evt=None):
    th = abs(float(threshold))
    easy_th = abs(float(easy_threshold))
    base_target_deg = 30.0

    if not pg.get_init():
        pg.init()

        try:
            n_displays = pg.display.get_num_displays()
        except Exception:
            n_displays = 1
        
        if display_idx is None:
            display_idx = 1 if n_displays >= 2 else 0

        display_idx = max(0, min(display_idx, max(0, n_displays - 1)))

        try:
            screen_sizes = pg.display.get_desktop_sizes()
            WIDTH, HEIGHT = screen_sizes[display_idx]
        except Exception:
            info = pg.display.Info()
            WIDTH, HEIGHT = info.current_w, info.current_h
        
        flags = 0

        if fullscreen:
            flags |= pg.FULLSCREEN
        else:
            flags |= pg.NOFRAME
        
        try:
            screen = pg.display.set_mode((WIDTH, HEIGHT), flags, display=display_idx, vsync=1)
        except TypeError:
            try:
                screen = pg.display.set_mode((WIDTH, HEIGHT), flags)
            except TypeError:
                screen = pg.display.set_mode((WIDTH, HEIGHT))
        
        pg.display.flip()
    else:
        screen = pg.display.get_surface()
        WIDTH, HEIGHT = screen.get_size()
    
    clock = pg.time.Clock()

    cursor_w = int(HEIGHT * 0.0825)
    target_sz = int(round(cursor_w * 1.25))
    y_center = HEIGHT // 2

    s = 180.0 / max(1.0, (WIDTH - (2.0 * cursor_w)))
    MIN_X = MIN_DEG - (s * cursor_w)
    MAX_X = MAX_DEG + (s * cursor_w)
    X_SPAN = MAX_X - MIN_X

    def _clamp_deg(d):
        return max(MIN_DEG, min(MAX_DEG, d))
    
    def _deg_to_x(d):
        frac = (d - MIN_X) / X_SPAN
        return int(round(frac * WIDTH))
    
    latest_disp = 0.0
    trial_active = False

    while True:
        if stop_evt is not None and stop_evt.is_set():
            try:
                pg.quit()
            except Exception:
                pass

            return 'stopped'
        
        if blackout_evt is not None and blackout_evt.is_set():
            screen.fill(BLACK)
            pg.display.flip()
            clock.tick(240)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                ABORT_EVT.set()
                pg.quit()
                return 'quit'
            
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    ABORT_EVT.set()
                    pg.quit()
                    return 'quit'

                mods = pg.key.get_mods()
                if event.key == pg.K_c and (mods & (pg.KMOD_LCTRL | pg.KMOD_RCTRL)):
                    ABORT_EVT.set()
                    pg.quit()
                    return 'quit'

        while True:
            try:
                _, payload = evt_queue.get_nowait()
            except Empty:
                break

            evt = _parse_event(payload)

            if evt == 'cue':
                latest_disp = 0.0

                while True:
                    try:
                        enc_queue.get_nowait()
                    except Empty:
                        break
                
                trial_active = True
            elif evt in {'hit', 'miss'}:
                trial_active = False
        
        if not trial_active:
            clock.tick(240)
            continue

        last_disp = None
        while True:
            try:
                _, payload = enc_queue.get_nowait()
                last_disp = payload
            except Empty:
                break
        
        if last_disp is not None:
            try:
                latest_disp = float(last_disp)
            except (TypeError, ValueError):
                pass
        
        screen.fill(BLACK)

        with TRIAL_LOCK:
            is_easy, alignment = TRIAL_CONFIG
        
        alignment = (alignment or "B").upper()
        if alignment not in {"L", "R", "B"}:
            alignment = "B"
        
        T = easy_th if is_easy else th
        target_deg = easy_th if is_easy else base_target_deg
        gain = target_deg / max(1e-6, T)

        cx = _deg_to_x(_clamp_deg(latest_disp * gain))
        cursor = pg.Rect(cx - (cursor_w // 2), 0, cursor_w, HEIGHT)

        pg.draw.rect(screen, BLUE, cursor)

        rx = _deg_to_x(+target_deg)
        lx = _deg_to_x(-target_deg)

        half_t = target_sz // 2
        half_c = cursor_w // 2
        top = (y_center - half_t) * 0.85

        if alignment in {"B", "L"}:
            left_target = pg.Rect(lx - half_c - half_t, top, target_sz, target_sz)

            pg.draw.rect(screen, BLUE, left_target)
            pg.draw.rect(screen, BLACK, left_target, width=5)
        
        if alignment in {"B", "R"}:
            right_target = pg.Rect(rx + half_c - half_t, top, target_sz, target_sz)

            pg.draw.rect(screen, BLUE, right_target)
            pg.draw.rect(screen, BLACK, right_target, width=5)

        pg.display.flip()
        clock.tick(240)


# ---------------------------
# BCI CLASS
# ---------------------------
class BCI:
    def __init__(self, *, phase_id, evt_queue, enc_queue, display_idx=1, fullscreen=True, easy_threshold=15.0):
        self.phase_id = str(phase_id)
        self.evt_q = evt_queue
        self.enc_q = enc_queue

        cfg = PHASE_CONFIG.get(self.phase_id)

        self.enabled = cfg is not None
        self.bidirectional = bool(cfg.get('bidirectional')) if cfg else False
        self.threshold = float(cfg.get('threshold')) if cfg else None

        self.display_idx = display_idx
        self.fullscreen = bool(fullscreen)
        self.easy_threshold = easy_threshold

        self._evt_q_internal = Queue()
        self._blackout_evt = threading.Event()
        self._blackout_timer = None
        self._evt_forward_thread = None

        self._stop_evt = threading.Event()
        self._thread = None

    def _forward_events(self):
        while not self._stop_evt.is_set():
            try:
                ts, evt = self.evt_q.get(timeout=0.5)
            except Empty:
                continue
            except Exception:
                continue

            e = str(evt).strip().lower()

            if e == "cue":
                self._blackout_evt.clear()
                t = self._blackout_timer
                self._blackout_timer = None

                if t is not None:
                    try:
                        t.cancel()
                    except Exception:
                        pass
            elif e in {"hit", "miss"}:
                t = self._blackout_timer

                if t is not None:
                    try:
                        t.cancel()
                    except Exception:
                        pass
                
                self._blackout_timer = threading.Timer(1.0, self._blackout_evt.set)
                self._blackout_timer.daemon = True
                self._blackout_timer.start()
            
            try:
                self._evt_q_internal.put_nowait((ts, evt))
            except Exception:
                pass

    def start(self):
        if not self.enabled:
            return False
        
        if self._thread and self._thread.is_alive():
            return True
        
        self._stop_evt.clear()

        if not self._evt_forward_thread or not self.evt_forward_thread.is_alive():
            self._evt_forward_thread = threading.Thread(target=self._forward_events, daemon=True)
            self._evt_forward_thread.start()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        return True
    
    def update_config(self, is_easy, alignment):
        a = (alignment or "B").upper()
        if a not in {"L", "R", "B"}:
            a = "B"
        
        global TRIAL_CONFIG
        with TRIAL_LOCK:
            TRIAL_CONFIG = (bool(is_easy), a)

    def stop(self, timeout=2.0):
        self._stop_evt.set()

        try:
            self._blackout_evt.clear()
            t = self._blackout_timer
            self._blackout_timer = None

            if t is not None:
                t.cancel()
        except Exception:
            pass

        try:
            if pg.get_init():
                pg.event.post(pg.event.Event(pg.QUIT))
        except Exception:
            pass

        t = self._thread
        if t and t.is_alive():
            t.join(timeout=float(timeout))
        
        return True
    
    def _run(self):
        cursor_fcn(threshold=self.threshold,
                   evt_queue=self._evt_q_internal,
                   enc_queue=self.enc_q,
                   display_idx=self.display_idx,
                   fullscreen=self.fullscreen,
                   easy_threshold=self.easy_threshold,
                   stop_evt=self._stop_evt,
                   blackout_evt=self._blackout_evt)


# ---------------------------
# TOP LEVEL (TESTING)
# ---------------------------
KEY_SPEED = 50.0
TRIAL_MS = 30_000


def _choose_alignment(phase_id, trial_n):
    cfg = PHASE_CONFIG.get(str(phase_id))
    if not cfg:
        return "B"
    
    if cfg.get('bidirectional', False):
        return "B"
    
    return "L" if (trial_n % 2 == 1) else "R"


def _choose_is_easy(trial_n, K):
    if trial_n <= 20:
        return ((trial_n - 1) % 5) == 0
    
    K = max(1, int(K))
    return ((trial_n - 21) % K) == 0


def _driver(evt_q, enc_q, phase_id, threshold, easy_threshold, cursor, stop_evt):
    deg = 0.0
    trial_n = 0
    last_frame_ms = int(time.time() * 1000)

    def _now_ms():
        return int(time.time() * 1000)
    
    def _send_evt(evt):
        try:
            evt_q.put_nowait((time.time(), evt))
        except Exception:
            pass
    
    def _send_enc(enc):
        try:
            enc_q.put_nowait((time.time(), enc))
        except Exception:
            pass

    while not stop_evt.is_set():
        trial_n += 1
        is_easy = _choose_is_easy(trial_n, K=5)
        alignment = _choose_alignment(str(phase_id), trial_n)

        try:
            cursor.update_config(is_easy, alignment)
        except Exception:
            pass

        _send_evt('cue')
        trial_start_ms = _now_ms()

        while not stop_evt.is_set():
            keys = pg.key.get_pressed() if pg.get_init() else None

            now = _now_ms()
            dt = max(0.0, (now - last_frame_ms) / 1000.0)
            last_frame_ms = now

            if keys:
                dx = 0.0

                if keys[pg.K_LEFT]:
                    dx -= KEY_SPEED * dt
                if keys[pg.K_RIGHT]:
                    dx += KEY_SPEED * dt
                
                if dx != 0.0:
                    deg = max(MIN_DEG, min(MAX_DEG, deg + dx))
            
            _send_enc(deg)

            T = float(easy_threshold) if bool(is_easy) else float(threshold)
            elapsed = now - trial_start_ms
            hit = abs(deg) >= abs(T)
            timeout = elapsed >= TRIAL_MS

            if hit or timeout:
                time.sleep(1.0)
                _send_evt('hit' if hit else 'miss')
                time.sleep((DELAY_MS / 1000.0) + 0.05)

                deg = 0.0
                break

            time.sleep(0.003)


def run_as_main(phase_id="3", fullscreen=False, display_idx=None, easy_threshold=15.0):
    from queue import Queue

    evt_q = Queue()
    enc_q = Queue()
    stop_evt = threading.Event()

    cursor = BCI(phase_id=str(phase_id),
                 evt_queue=evt_q,
                 enc_queue=enc_q,
                 display_idx=display_idx,
                 fullscreen=fullscreen,
                 easy_threshold=easy_threshold)
    
    if not cursor.enabled:
        raise RuntimeError(f'Phase {phase_id!r} not configured for BCI (no threshold set)')
    
    cursor.start()

    driver = threading.Thread(
        target=_driver,
        args=(evt_q, enc_q, str(phase_id), float(cursor.threshold), float(cursor.easy_threshold), cursor, stop_evt),
        daemon=True
        )
    driver.start()

    try:
        while cursor._thread and cursor._thread.is_alive():
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()

        try:
            cursor.stop(timeout=2.0)
        except Exception:
            pass


if __name__ == '__main__':
    run_as_main(phase_id="4", fullscreen=False, display_idx=1, easy_threshold=15.0)

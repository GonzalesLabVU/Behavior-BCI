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

TRIAL_CONFIG = (False, "B")
TRIAL_LOCK = threading.Lock()
ABORT_EVT = threading.Event()

ENC_TAG = "WHEEL"
SIM_TAG = "SIM"
TAGS = {ENC_TAG, SIM_TAG}


# ---------------------------
# HELPERS
# ---------------------------
def _parse_event(payload):
    if not payload:
        return ""
    
    return str(payload).strip().split()[0].lower()


def cursor_fcn(threshold, evt_queue, enc_queue, *, display_idx=None, fullscreen=True,
               easy_threshold=15.0, stop_evt=None):
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
    
    current_disp = 0.0
    trial_active = False
    is_blackout = False
    delay_until = None
    freeze_disp = None
    active_is_easy, active_alignment = TRIAL_CONFIG

    while True:
        if stop_evt is not None and stop_evt.is_set():
            try:
                pg.quit()
            except Exception:
                pass

            return 'stopped'

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

            if evt == "cue":
                is_blackout = False
                delay_until = None
                freeze_disp = None
                current_disp = 0.0

                with TRIAL_LOCK:
                    active_is_easy, active_alignment = TRIAL_CONFIG

                while True:
                    try:
                        enc_queue.get_nowait()
                    except Empty:
                        break
                
                trial_active = True
            elif evt in {"hit", "miss"}:
                if trial_active and (delay_until is None) and (not is_blackout):
                    freeze_disp = current_disp
                    delay_until = time.monotonic() + (DELAY_MS / 1000.0)
        
        if delay_until is not None and time.monotonic() >= delay_until:
            is_blackout = True
            delay_until = None
        
        if is_blackout:
            while True:
                try:
                    enc_queue.get_nowait()
                except Empty:
                    break

            screen.fill(BLACK)
            pg.display.flip()
            clock.tick(240)
            continue

        if not trial_active:
            clock.tick(240)
            continue

        delay_active = (delay_until is not None)

        if not delay_active:
            new_disp = None

            while True:
                try:
                    tag, payload = enc_queue.get_nowait()
                except Empty:
                    break
            
                tag = str(tag).upper().strip()
                if tag not in TAGS:
                    continue

                new_disp = payload
            
            if new_disp is not None:
                try:
                    current_disp = float(new_disp)
                except (TypeError, ValueError):
                    pass
        
        screen.fill(BLACK)

        is_easy, alignment = active_is_easy, active_alignment
        
        alignment = (alignment or "B").upper()
        if alignment not in {"L", "R", "B"}:
            alignment = "B"
        
        T = easy_th if is_easy else th
        target_deg = easy_th if is_easy else base_target_deg
        gain = target_deg / max(1e-6, T)

        enc_val = (freeze_disp if (delay_until is not None and freeze_disp is not None)
                     else current_disp)

        cx = _deg_to_x(_clamp_deg(enc_val * gain))
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
    def __init__(self, *, phase_id, evt_queue, enc_queue, config, display_idx=1, fullscreen=True,
                 easy_threshold=15.0):
        self.phase_id = str(phase_id)
        self.evt_q = evt_queue
        self.enc_q = enc_queue

        cfg = config.get(self.phase_id)

        self.enabled = cfg is not None
        self.bidirectional = bool(cfg.get('bidirectional')) if cfg else False
        self.threshold = float(cfg.get('threshold')) if cfg else None

        self.display_idx = display_idx
        self.fullscreen = bool(fullscreen)
        self.easy_threshold = easy_threshold

        self._stop_evt = threading.Event()
        self._thread = None

    def start(self):
        if not self.enabled:
            return False
        
        if self._thread and self._thread.is_alive():
            return True
        
        self._stop_evt.clear()

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
                   evt_queue=self.evt_q,
                   enc_queue=self.enc_q,
                   display_idx=self.display_idx,
                   fullscreen=self.fullscreen,
                   easy_threshold=self.easy_threshold,
                   stop_evt=self._stop_evt)

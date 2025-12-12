import os

os.system('cls')
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import time
import threading
import warnings
import pygame as pg
from queue import Queue, Empty

warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API')

MIN_DEGREES = -90.0
MAX_DEGREES = +90.0
DELAY_MS = 1000

BLUE = (50, 50, 255)
BLACK = (0, 0, 0)


def _parse_event(payload):
    if not payload:
        return ''
    return str(payload).strip().split()[0].lower()


def cursor_fcn(threshold, evt_queue, enc_queue, *, display_index=None, fullscreen=True, easy_threshold=15.0):
    th = abs(float(threshold))
    easy_th = abs(float(easy_threshold))
    base_target_deg = 30.0

    if not pg.get_init():
        pg.init()

        try:
            num_displays = pg.display.get_num_displays()
        except Exception:
            num_displays = 1
        
        if display_index is None:
            display_index = 1 if num_displays >= 2 else 0
        display_index = max(0, min(display_index, max(0, num_displays - 1)))

        try:
            screen_sizes = pg.display.get_desktop_sizes()
            WIDTH, HEIGHT = screen_sizes[display_index]
        except Exception:
            info = pg.display.Info()
            WIDTH, HEIGHT = info.current_w, info.current_h

        flags = 0
        if fullscreen:
            flags |= pg.FULLSCREEN
        else:
            flags |= pg.NOFRAME
        
        try:
            screen = pg.display.set_mode(
                (WIDTH, HEIGHT), flags,
                display=display_index,
                vsync=1
                )
        except TypeError:
            try:
                screen = pg.display.set_mode((WIDTH, HEIGHT), flags)
            except TypeError:
                screen = pg.display.set_mode((WIDTH, HEIGHT))
        
        pg.display.set_caption('Cursor Movement')
        pg.display.flip()
    else:
        screen = pg.display.get_surface()
        WIDTH, HEIGHT = screen.get_size()
    
    clock = pg.time.Clock()

    cursor_color = BLUE
    cursor_w = int(HEIGHT * 0.07)
    target_size = cursor_w
    y_center = HEIGHT // 2

    s = 180.0 / max(1.0, (WIDTH - 2.0*cursor_w))
    MIN_X = MIN_DEGREES - s*cursor_w
    MAX_X = MAX_DEGREES + s*cursor_w
    X_SPAN = MAX_X - MIN_X

    ##################
    def _clamp_deg(d):
        return max(MIN_DEGREES, min(MAX_DEGREES, d))
    
    def _deg_to_x(d):
        frac = (d - MIN_X) / X_SPAN
        return int(round(frac * WIDTH))
    ##################

    latest_disp = 0.0
    trial_idx = 0
    K_updated = False
    K = 5
    trial_active = False

    ##################
    def _show_easy_for(trial_num):
        if trial_num <= 20 or not K_updated:
            return ((trial_num - 1) % 5) == 0
        else:
            step = max(1, int(K))
            return trial_num >= 21 and ((trial_num - 21) % step) == 0
    ##################
    
    show_easy = _show_easy_for(1)

    ##################
    def _start_next_trial():
        nonlocal trial_idx, show_easy, latest_disp, trial_active

        trial_idx += 1
        show_easy = _show_easy_for(trial_idx)
        latest_disp = 0.0

        while True:
            try:
                enc_queue.get_nowait()
            except Empty:
                break

        trial_active = True
    ##################

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return 'quit'
            
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.quit()
                    return 'quit'
                
                mods = pg.key.get_mods()
                if event.key == pg.K_c and (mods & (pg.KMOD_LCTRL | pg.KMOD_RCTRL)):
                    pg.quit()
                    return 'quit'
        
        while True:
            try:
                _, payload = evt_queue.get_nowait()
            except Empty:
                break

            evt = _parse_event(payload)    

            if evt == 'cue':
                _start_next_trial()
            elif evt in {'hit', 'miss'}:
                trial_active = False
            else:
                txt = (payload or "").strip().lower().replace("=", " ").replace(":", " ")
                parts = txt.split()
                if len(parts) >= 2 and parts[0] == 'setk':
                    try:
                        K = max(1, int(float(parts[1])))
                        K_updated = True
                    except ValueError:
                        pass
        
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

        T = easy_th if show_easy else th
        visual_target_deg = easy_th if show_easy else base_target_deg
        gain = visual_target_deg / max(1e-6, T)

        rx = _deg_to_x(+visual_target_deg)
        lx = _deg_to_x(-visual_target_deg)
        left_target = pg.Rect(lx - target_size, y_center - target_size // 2, target_size, target_size)
        right_target = pg.Rect(rx, y_center - target_size // 2, target_size, target_size)

        cx = _deg_to_x(_clamp_deg(latest_disp * gain))
        cursor = pg.Rect(cx - cursor_w // 2, 0, cursor_w, HEIGHT)

        pg.draw.rect(screen, cursor_color, cursor)

        pg.draw.rect(screen, BLUE, left_target)
        pg.draw.rect(screen, BLACK, left_target, width=5)
        pg.draw.rect(screen, BLUE, right_target)
        pg.draw.rect(screen, BLACK, right_target, width=5)

        pg.display.flip()
        clock.tick(240)
    
    return 'running'


KEY_SPEED = 30.0
TRIAL_MS = 30_000


def _show_easy_for_external(trial_num, K=5, K_updated=False):
    if trial_num <= 20 or not K_updated:
        return ((trial_num - 1) % 5) == 0
    else:
        step = max(1, int(K))
        return trial_num >= 21 and ((trial_num - 21) % step) == 0
    

def _driver(evt_queue, enc_queue, threshold, easy_threshold, stop_evt):
    deg = 0.0
    trial_idx = 0
    K = 5
    K_updated = False
    last_frame_ms = pg.time.get_ticks() if pg.get_init() else int(time.time() * 1000)

    ##################
    def _now_ms():
        return pg.time.get_ticks() if pg.get_init() else int(time.time() * 1000)
    
    def _send(evt):
        try:
            evt_queue.put_nowait((time.time(), evt))
        except Exception:
            pass
    ##################
    
    trial_idx += 1
    _send('cue')
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
                deg = max(MIN_DEGREES, min(MAX_DEGREES, deg + dx))
        
        try:
            enc_queue.put_nowait((time.time(), deg))
        except Exception:
            pass

        T = easy_threshold if _show_easy_for_external(trial_idx, K, K_updated) else threshold
        elapsed = now - trial_start_ms
        hit = abs(deg) >= abs(T)
        timeout = elapsed >= TRIAL_MS

        if hit or timeout:
            time.sleep(1.0)
            _send('hit' if hit else 'miss')
            time.sleep((DELAY_MS / 1000.0) + 0.05)
            
            trial_idx += 1
            deg = 0.0
            _send('cue')
            trial_start_ms = _now_ms()
        
        time.sleep(0.003)


def _run_as_main(threshold=30.0, easy_threshold=15.0, fullscreen=False, display_index=None):
    evt_q = Queue()
    enc_q = Queue()
    stop_evt = threading.Event()

    driver = threading.Thread(
        target=_driver,
        args=(evt_q, enc_q, float(threshold), float(easy_threshold), stop_evt),
        daemon=True
        )
    driver.start()

    try:
        cursor_fcn(
            threshold=threshold,
            evt_queue = evt_q,
            enc_queue=enc_q,
            display_index=display_index,
            fullscreen=fullscreen,
            easy_threshold=easy_threshold
            )
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()

        try:
            if pg.get_init():
                pg.event.post(pg.event.Event(pg.QUIT))
        except Exception:
            pass


if __name__ == '__main__':
    _run_as_main(
        threshold=30.0,
        easy_threshold=15.0,
        fullscreen=False,
        display_index=None
        )

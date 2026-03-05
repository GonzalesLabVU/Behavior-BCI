"""
TCP SERVER  ->  MICROSCOPE COMPUTER
"""

import os, sys, traceback, warnings
warnings.filterwarnings('ignore',
                        message=".*pin_memory' argument is set as true but no accelerator is found.*",
                        category=UserWarning)

import json, re, socket, threading, time
import tkinter as tk
from tkinter import messagebox
from datetime import datetime

WINDOWS = sys.platform == 'win32'

try:
    import pynput, pyautogui, mss
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE    = 0
    WIN_IMPORTS_OK = True
except Exception:
    WIN_IMPORTS_OK = False


# --------------------------
# GLOBALS
# --------------------------
SERVER_PORT    = 5005
DISCOVERY_PORT = 5007


# --------------------------
# HELPERS
# --------------------------
def _now_ts():
    return datetime.now().strftime('%H:%M:%S.%f')[:-3]


def _get_screen(win_bbox):
    if not WIN_IMPORTS_OK or win_bbox is None:
        return None

    wl, wt, ww, wh = win_bbox
    cx, cy = wl + ww / 2.0, wt + wh / 2.0

    with mss.mss() as sct:
        monitors = sct.monitors[1:]

    for m in monitors:
        if m['left'] <= cx < m['left'] + m['width'] and m['top'] <= cy < m['top'] + m['height']:
            return m['left'], m['top'], m['width'], m['height']

    with mss.mss() as sct:
        v = sct.monitors[0]
    return v['left'], v['top'], v['width'], v['height']


# --------------------------
# DISCOVERY
# --------------------------
class DiscoveryService:
    def __init__(self, server_port, discovery_port=DISCOVERY_PORT,
                 verbose=False):
        self.server_port    = int(server_port)
        self.discovery_port = int(discovery_port)
        self._verbose       = bool(verbose)
        self._stop_event    = threading.Event()
        self._thread        = None
        self._sock          = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout=1.0):
        self._stop_event.set()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as w:
                w.sendto(b'{}', ('127.0.0.1', self.discovery_port))
        except Exception:
            pass
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = self._sock = None

    def _run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock = s
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', self.discovery_port))
            s.settimeout(0.5)
            reply = json.dumps({"id": "SERVER_HERE",
                                "server_port": self.server_port}).encode()
            while not self._stop_event.is_set():
                try:
                    data, addr = s.recvfrom(2048)
                    _dump(f'[SERVER] UDP_RECV from {addr}', data, verbose=self._verbose)
                except socket.timeout:
                    continue
                except OSError as e:
                    if "timed out" in str(e).lower():
                        continue

                    break
                try:
                    if json.loads(data.decode('utf-8', errors='ignore')).get('id') == "DISCOVER_SERVER":
                        s.sendto(reply, addr)
                        _dump(f'[SERVER] UDP_SEND to {addr}', reply, verbose=self._verbose)
                except Exception:
                    continue
        finally:
            try:
                s.close()
            except Exception:
                pass


# --------------------------
# UI AUTOMATION
# --------------------------
class PrairieUser:
    pixel_map = {
        "tSeries":  (210, 514),
        "period":   (164, 580),
        "duration": (244, 580),
        "basename": (388, 866),
        "savePath": (284, 866),
        "iter":     (512, 870),
        "trig":     (606, 870)
        }

    def __init__(self, target="Prairie View 5.8", points=None):
        self.target   = target
        self.points   = points or self.pixel_map
        self._last_xy = None
        self._panel   = None

    def activate(self, timeout=5.0):
        if not WIN_IMPORTS_OK:
            return False
        t0 = time.time()
        while time.time() - t0 < timeout:
            windows = pyautogui.getWindowsWithTitle(self.target)
            if windows:
                try:
                    windows[0].activate()
                except Exception:
                    pass
                self._show_bbox()
                return True
            time.sleep(0.1)
        return False

    def deactivate(self):
        panel = self._panel
        if panel is None:
            return
        self._panel = None
        try:
            panel['alive'] = False
        except Exception:
            pass
        for widget in (panel.get('win'), panel.get('root')):
            if widget:
                try:
                    widget.after(0, widget.destroy)
                except Exception:
                    pass
                try:
                    widget.destroy()
                except Exception:
                    pass
        t = panel.get('thread')
        if t and t.is_alive() and threading.current_thread() is not t:
            try:
                t.join(timeout=1.0)
            except Exception:
                pass

    def get_bbox(self, target=None):
        if not WIN_IMPORTS_OK:
            return None
        try:
            windows = pyautogui.getWindowsWithTitle(target or self.target)
            if not windows:
                return None
            w = windows[0]
            left, top, width, height = int(w.left), int(w.top), int(w.width), int(w.height)
            return (left, top, width, height) if width > 0 and height > 0 else None
        except Exception:
            return None

    def ref_xy(self):
        bbox = self.get_bbox()
        return (bbox[0], bbox[1]) if bbox else (0, 0)

    def norm_xy(self, *args, mode="rel"):
        assert mode in {"abs", "rel"}
        if len(args) == 1 and isinstance(args[0], str):
            label = args[0]
            if label not in self.points:
                raise KeyError(f'Unknown point label: {label!r}')
            x1, y1 = self.points[label]
        elif len(args) == 1 and isinstance(args[0], (tuple, list)):
            x1, y1 = args[0]
        elif len(args) == 2:
            x1, y1 = args
        else:
            raise ValueError(f'Expected label, (x,y), or x,y but got {args!r}')
        if mode == "abs":
            return int(x1), int(y1)
        x0, y0 = self.ref_xy()
        return int(x0 + x1), int(y0 + y1)

    def poll(self, mode="abs"):
        if not WIN_IMPORTS_OK:
            return
        assert mode in {"abs", "rel"}
        pos    = pyautogui.position()
        mx, my = int(pos.x), int(pos.y)
        if self._last_xy == (mx, my):
            return
        self._last_xy = (mx, my)
        bbox = self.get_bbox()
        if bbox is None:
            return
        left, top, width, height = bbox
        rx, ry = mx - left, my - top
        if rx < 0 or ry < 0 or rx >= width or ry >= height:
            return
        print(f'({mx}, {my})' if mode == "abs" else f'({rx}, {ry})')

    def click(self, *args, clicks=1, mode="rel"):
        if not WIN_IMPORTS_OK:
            return
        x, y = self.norm_xy(*args, mode=mode)
        pyautogui.click(x, y, clicks=clicks)

    def move(self, *args, mode="rel"):
        if not WIN_IMPORTS_OK:
            return
        x, y = self.norm_xy(*args, mode=mode)
        pyautogui.moveTo(x, y)

    def scroll(self, target="Browse For Folder", dist=-5):
        if not WIN_IMPORTS_OK:
            return
        try:
            windows = pyautogui.getWindowsWithTitle(target)
            if windows:
                try:
                    windows[0].activate()
                except Exception:
                    pass
        except Exception:
            pass
        bbox = self.get_bbox(target=target)
        if bbox:
            left, top, width, height = bbox
            self.move(left + width // 2, top + height // 2)
        else:
            pos = pyautogui.position()
            self.move(int(pos.x), int(pos.y))
        pynput.mouse.Controller().scroll(dx=0, dy=int(dist))

    def write(self, text):
        if not WIN_IMPORTS_OK:
            return
        pyautogui.write(str(text))

    def read(self, topleft, bottomright):
        pass

    def configure(self):
        if not WIN_IMPORTS_OK:
            return
        time.sleep(0.1)
        x1, y1 = self.pixel_map['tSeries']
        self.click(x1, y1)
        time.sleep(0.1)

        x2, y2 = self.pixel_map['period']
        self.click(x2, y2, clicks=2)
        self.write(1 / 1000)
        time.sleep(0.1)

        x3, y3 = self.pixel_map['duration']
        self.click(x3, y3, clicks=2)
        self.write(35)

    def start_tseries(self, trial_n):
        if not WIN_IMPORTS_OK:
            return
        x0, y0 = self.pixel_map['iter']
        self.click(x0, y0, clicks=2)
        self.write(trial_n)
        time.sleep(0.1)
        x1, y1 = self.pixel_map['trig']
        self.click(x1, y1)

    def stop_tseries(self):
        if not WIN_IMPORTS_OK:
            return
        x0, y0 = self.pixel_map['trig']
        self.click(x0, y0)

        target = "Playback Controls"
        t0, bbox = time.time(), None
        while time.time() - t0 < 0.25:
            bbox = self.get_bbox(target=target)
            if bbox:
                break
            time.sleep(0.01)

        if bbox is None:
            return

        try:
            windows = pyautogui.getWindowsWithTitle(target)
            if windows:
                windows[0].activate()
        except Exception:
            pass

        left, top, width, _ = bbox
        self.click((left + width - 10, top + 10), clicks=1, mode='abs')

    def _show_bbox(self):
        if not WIN_IMPORTS_OK or self._panel:
            return
        if not (win_bbox := self.get_bbox()):
            return
        if not (screen_bbox := _get_screen(win_bbox)):
            return

        left, top, width, height = screen_bbox
        panel = {"thread": None, "alive": True, "ready": threading.Event(),
                 "root": None, "win": None, "canvas": None,
                 "rect_id": None, "poll_ms": 50, "last_bbox": None}
        self._panel = panel

        t = threading.Thread(target=self._bbox_thread,
                             args=(left, top, width, height), daemon=True)
        panel['thread'] = t
        t.start()

        for _ in range(50):
            if panel['ready'].wait(timeout=0.1):
                break

        if not panel['alive'] or panel['win'] is None:
            self._panel = None

    def _bbox_thread(self, left, top, width, height):
        panel = self._panel
        if panel is None:
            return
        try:
            root = tk.Tk()
            root.withdraw()
            win = tk.Toplevel(root)
            win.overrideredirect(True)
            win.attributes('-topmost', True)
            try:
                win.configure(bg='magenta')
                win.wm_attributes('-transparentcolor', 'magenta')
            except Exception:
                win.configure(bg='magenta')
            win.geometry(f'{width}x{height}+{left}+{top}')

            canvas  = tk.Canvas(win, width=width, height=height,
                                highlightthickness=0, bd=0, bg='magenta')
            canvas.pack(fill='both', expand=True)
            rect_id = canvas.create_rectangle(1, 1, width-2, height-2,
                                              outline="#ff0000", width=10, fill="")

            panel.update({"root": root, "win": win, "canvas": canvas,
                          "rect_id": rect_id, "last_bbox": (left, top, width, height)})
            panel['ready'].set()

            def _tick():
                p = self._panel
                if p is None or not p.get('alive'):
                    for w in (win, root):
                        try:
                            w.destroy()
                        except Exception:
                            pass
                    return

                if (sb := _get_screen(self.get_bbox())) and sb != p.get('last_bbox'):
                    l2, t2, w2, h2 = sb
                    p['last_bbox'] = sb
                    try:
                        win.geometry(f'{w2}x{h2}+{l2}+{t2}')
                        canvas.config(width=w2, height=h2)
                        canvas.coords(p['rect_id'], 1, 1, w2-2, h2-2)
                    except Exception:
                        pass
                try:
                    win.after(p['poll_ms'], _tick)
                except Exception:
                    if self._panel:
                        self._panel['alive'] = False
                    _tick()

            win.after(panel['poll_ms'], _tick)
            root.mainloop()
        except Exception:
            if self._panel:
                self._panel['alive'] = False
                self._panel['ready'].set()


# --------------------------
# SERVER
# --------------------------
def _hexdump(b, maxlen=512):
    if b is None:
        return "<None>"
    
    bb = b[:maxlen]
    hx = " ".join(f'{x:02x}' for x in bb)
    tail = "" if len(b) <= maxlen else f' ... (+{len(b) - maxlen} bytes)'

    return f"len={len(b)} hex={hx}{tail}"


def _dump(tag, b, verbose=False):
    if not verbose:
        return
    
    print(f'{tag} {repr(b)} | {_hexdump(b)}', flush=True)


class PrairieServer:
    VALID_CMD = {"CONFIG", "START", "STOP", "FINISH"}

    def __init__(self, local_ip, local_port, user,
                 timeout=35.0, verbose=False):
        
        self.user        = user
        self.state       = "IDLE"
        self._configured = False
        self.trial_n     = 0
        self.start_ts    = []
        self.stop_ts     = []
        self._finish_idx = 0

        self._local_ip   = local_ip
        self._local_port = local_port
        self._timeout    = timeout
        self._verbose    = bool(verbose)
        self._sock       = None
        self._connected  = False
        self._running    = threading.Event()
        self._thread     = None
        self._lock       = threading.Lock()
        self._discovery  = DiscoveryService(server_port=local_port,
                                            verbose=self._verbose)

    def connect(self):
        assert not self._connected

        if self._verbose:
            print(f'[SERVER] Binding to socket...', flush=True)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self._local_ip, self._local_port))
        self._sock.listen(1)
        self._sock.settimeout(self._timeout)

        self._connected = True
        self._running.set()

        if self._verbose:
            print(f'[SERVER] Starting server process thread...', flush=True)

        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        self._discovery.start()

        if self._verbose:
            print('\n[SERVER] (Use Ctrl+C to kill process)')
            print(f'[SERVER] Listening on {self._local_ip}:{self._local_port}')

    def disconnect(self):
        if not self._connected and not self._running.is_set():
            return
        
        if self._verbose:
            print('[SERVER] Disconnecting from client...', flush=True)

        self._connected = False
        self._running.clear()
        try:
            self._discovery.stop()
        except Exception:
            pass
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        t, self._thread = self._thread, None
        if t and t.is_alive() and threading.current_thread() is not t:
            try:
                t.join(timeout=2.0)
            except Exception:
                pass

    def _accept_loop(self):
        while self._running.is_set():
            try:
                conn, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
            conn.settimeout(self._timeout)
            self._reader_loop(conn, addr)
            break
        try:
            self.disconnect()
        except Exception:
            pass

    def _reader_loop(self, conn, addr):
        if self._verbose:
            print(f'[SERVER] Client connected from {addr[0]}:{addr[1]}\n')
        
        try:
            self._discovery.stop(timeout=1.0)
        except Exception:
            pass

        rfile = conn.makefile('rb')
        try:
            while self._running.is_set():
                try:
                    line = rfile.readline()
                except socket.timeout:
                    continue
                except OSError:
                    break

                if not line:
                    break

                _dump('[SERVER] TCP_RECV', line, verbose=self._verbose)

                cmd = line.decode('utf-8', errors='ignore').strip().upper()
                if cmd not in self.VALID_CMD:
                    continue

                extra = None
                try:
                    extra = self._dispatch(cmd)
                    self._ack(conn, extra)
                except Exception as e:
                    print(f'[SERVER] Unexpected exception during {cmd!r}:\n\n{e}\n')
                    traceback.print_exc()
                    self._err(conn)
                    continue

                if cmd == "FINISH" and isinstance(extra, dict) and extra.get('done'):
                    return
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _monitor_window(self, root, interval_ms=500):
        if WIN_IMPORTS_OK and self.user.get_bbox() is None:
            try:
                root.quit()
            except Exception:
                pass
            return
        root.after(interval_ms, self._monitor_window, root, interval_ms)

    def _dispatch(self, cmd):
        match cmd:
            case "CONFIG":
                with self._lock:
                    self._require_state('IDLE')
                    if self._configured:
                        if self._verbose:
                            print('[SERVER] Configuration already done, ignoring command')
                        return None
                    self.state = "CONFIG"
                try:
                    self.user.configure()
                    self._configured = True
                finally:
                    with self._lock:
                        self.state = "IDLE"
                print('[SERVER] CONFIG')
                return None

            case "START":
                with self._lock:
                    self._require_state('IDLE')
                    self.state = "START"
                    self.trial_n += 1
                    self.start_ts.append(_now_ts())
                try:
                    self.user.start_tseries(self.trial_n)
                finally:
                    with self._lock:
                        self.state = "IDLE"
                print(f'[SERVER] START  trial={self.trial_n}  ts={self.start_ts[-1]}')
                return None

            case "STOP":
                with self._lock:
                    self._require_state('IDLE')
                    self.state = "STOP"
                    self.stop_ts.append(_now_ts())
                try:
                    self.user.stop_tseries()
                finally:
                    with self._lock:
                        self.state = "IDLE"
                print(f'[SERVER] STOP  trial={self.trial_n}  ts={self.stop_ts[-1]}')
                return None

            case "FINISH":
                with self._lock:
                    self._require_state('IDLE')
                    self.state    = "FINISH"
                    i             = self._finish_idx
                    total         = max(len(self.start_ts), len(self.stop_ts), 1)
                    start         = self.start_ts[i] if i < len(self.start_ts) else None
                    stop          = self.stop_ts[i]  if i < len(self.stop_ts)  else None
                    done          = (i + 1) >= total
                    self._finish_idx += 1
                    self.state    = "IDLE"
                    if done:
                        self._running.clear()
                print('[SERVER] FINISH')
                return {"start_ts": str(start) if start else "",
                        "stop_ts":  str(stop)  if stop  else "",
                        "done":     done}

            case _:
                raise RuntimeError('Unreachable')

    def _ack(self, conn, data=None):
        line = 'OK'
        if data:
            line += ' ' + json.dumps(data)
        try:
            out = (line + '\n').encode('utf-8')
            _dump('[SERVER] TCP_SEND', out, verbose=self._verbose)
            conn.sendall(out)
        except OSError:
            pass

    def _err(self, conn):
        try:
            out = b'ERR\n'
            _dump('[SERVER] TCP_SEND', out, verbose=self._verbose)
            conn.sendall(out)
        except OSError:
            pass

    def _require_state(self, expected):
        if self.state != expected:
            raise RuntimeError(
                f'Command requires state {expected!r}, current state is {self.state!r}')


# --------------------------
# MAIN
# --------------------------
def main(verbose=False):
    root   = tk.Tk()
    root.withdraw()
    server = None
    user   = PrairieUser()

    if WIN_IMPORTS_OK:
        if not user.activate(timeout=5.0):
            try:
                r = tk.Tk()
                r.withdraw()
                messagebox.showerror(
                    title='Prairie Server Startup',
                    message=(f'Could not find the target window:\n\n"{user.target}"\n\n'
                             'Make sure PrairieView is running and the window title matches, '
                             'then try again'),
                    parent=r)
                r.destroy()
            except Exception:
                pass
            try:
                root.quit()
                root.destroy()
            except Exception:
                pass
            return
    else:
        if verbose:
            print(f'[SERVER] WIN_IMPORTS_OK=False — UI automation disabled, running network-only', flush=True)
        

    server = PrairieServer(local_ip='0.0.0.0',
                           local_port=SERVER_PORT,
                           user=user,
                           verbose=verbose)
    server.connect()
    root.after(500, server._monitor_window, root, 500)

    try:
        root.mainloop()
    finally:
        try:
            if server:
                server.disconnect()
        except Exception:
            pass
        try:
            if user:
                user.deactivate()
        except Exception:
            pass
        try:
            root.quit()
            root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    if WINDOWS:
        os.system('cls')
    main(verbose=False)

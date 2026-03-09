"""
TCP CLIENT  ->  TRAINING COMPUTER
"""

import json
import time
import socket
import threading
import types
import queue

CLIENT_IP = "192.168.2.1"
CLIENT_PORT = 5006
SERVER_IP = "192.168.2.2"
SERVER_PORT = 5005

RECV_TIMEOUT = 5.0


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


class PrairieClient:
    def __init__(self, verbose=True):
        self.verbose = verbose
        
        self.start_ts = []
        self.stop_ts = []

        self._imaging = False
        self._finished = False

        self.start_timer = None
        self.stop_timer = None

        self._q = queue.Queue()
        self._net_thread = None
        self._net_stop = threading.Event()
        self._sock_lock = threading.Lock()
        
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(RECV_TIMEOUT)
        self._sock.bind((CLIENT_IP, CLIENT_PORT))
        self._sock.connect((SERVER_IP, SERVER_PORT))

        self._rfile = self._sock.makefile('rb')

        self._net_thread = threading.Thread(target=self._net_loop, daemon=True)
        self._net_thread.start()
    
    @staticmethod
    def _patch_join(t):
        if t is None:
            return None
        
        orig_join = t.join

        def isr_join(timer, interval=0.2):
            while timer.is_alive():
                orig_join(timeout=interval)
        
        t.isr_join = types.MethodType(isr_join, t)
        return t
    
    def _send(self, cmd, want_data=False):
        with self._sock_lock:
            out = (cmd + '\n').encode('utf-8')
            _dump('[CLIENT] TCP_SEND', out, verbose=self.verbose)
            self._sock.sendall(out)

            line = self._rfile.readline()
            _dump('[CLIENT] TCP_RECV', line, verbose=self.verbose)
        
        if not line:
            return None if want_data else False
        
        line = line.decode('utf-8', errors='ignore').strip()
        if not line.startswith('OK'):
            return None if want_data else False
        
        if want_data:
            rest = line[2:].strip()

            try:
                return json.loads(rest) if rest else {}
            except Exception:
                return {}
        
        return True
    
    def _net_loop(self):
        while not self._net_stop.is_set():
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                break

            cmd, want_data, resp_q = item
            try:
                resp = self._send(cmd, want_data=want_data)
            except Exception:
                resp = None if want_data else False
            
            if resp_q is not None:
                try:
                    resp_q.put(resp, timeout=0.5)
                except Exception:
                    pass
    
    def _enqueue(self, cmd, want_data=False, wait_reply=True):
        if self._finished:
            return None if want_data else False
        
        resp_q = queue.Queue(maxsize=1) if wait_reply else None
        self._q.put((cmd, want_data, resp_q))

        if not wait_reply:
            return True
        
        try:
            return resp_q.get(timeout=RECV_TIMEOUT + 1.0)
        except queue.Empty:
            return None if want_data else False
    
    def _cancel_timer(self, t):
        if t is not None:
            try:
                t.cancel()
            except Exception:
                pass
    
    def configure(self):
        return bool(self._enqueue('CONFIG', want_data=False, wait_reply=True))
    
    def start(self, wait_s=None):
        if self._finished:
            return False
        if self._imaging:
            return True
        
        if wait_s is not None:
            time.sleep(wait_s)

        self._imaging = True

        return bool(self._enqueue('START', want_data=False, wait_reply=True))
    
    def stop(self, wait_s=None):
        if self._finished:
            return False
        if not self._imaging:
            return True
        
        if wait_s is not None:
            time.sleep(wait_s)
        
        self._imaging = False
            
        return bool(self._enqueue('STOP', want_data=False, wait_reply=True))
        
        # self._cancel_timer(self.stop_timer)
        # self.stop_timer = Timer(float(wait_s), lambda: self._enqueue('STOP', False, True))
        # self.stop_timer.daemon = True
        # self._patch_join(self.stop_timer)
        # self.stop_timer.start()

    def finish(self):
        if self._finished:
            return

        self._cancel_timer(self.stop_timer)
        self._cancel_timer(self.start_timer)
        self.stop_timer = None
        self.start_timer = None

        start_ts = []
        stop_ts = []

        while True:
            data = self._enqueue('FINISH', want_data=True, wait_reply=True)
            if not isinstance(data, dict):
                break

            if data.get('start_ts'):
                start_ts.append(data['start_ts'])
            if data.get('stop_ts'):
                stop_ts.append(data['stop_ts'])
            
            if data.get('done'):
                break
        
        self.start_ts = [str(v) for v in start_ts]
        self.stop_ts = [str(v) for v in stop_ts]

        self._finished = True

        self._net_stop.set()
        self._q.put(None)

        try:
            self._net_thread.join(timeout=1.0)
        except Exception:
            pass

        try:
            self._rfile.close()
        except Exception:
            pass

        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass

        try:
            self._sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    client = PrairieClient(verbose=True)

    try:
        client.configure()
        print('[CLIENT] CONFIG')

        client.start()
        print('[CLIENT] START')

        client.stop(wait_s=31.0)
        print('[CLIENT] STOP')

        while True:
            client.start(wait_s=1.0)
            print('[CLIENT] START')

            client.stop(wait_s=32.0)
            print('[CLIENT] STOP')
    except KeyboardInterrupt:
        pass
    finally:
        client.finish()
        print('[CLIENT] FINISH')

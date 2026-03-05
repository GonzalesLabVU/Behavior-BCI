"""
TCP CLIENT  ->  TRAINING COMPUTER
"""

import json
import socket
import threading
import types
import queue
from threading import Timer

SERVER_IP = "192.168.2.2"

SERVER_PORT = 5005
SEARCH_PORT = 5007

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
    def __init__(self, server_ip=None, server_port=None,
                 search_timeout=1.0, search_attempts=3,
                 verbose=False):
        
        self.verbose = verbose
        
        self.start_ts = []
        self.stop_ts = []

        self._finished = False

        self.start_timer = None
        self.stop_timer = None

        self._q = queue.Queue()
        self._net_thread = None
        self._net_stop = threading.Event()
        self._sock_lock = threading.Lock()

        if server_ip is None or server_port is None:
            found_ip, found_port = self._search(search_timeout, search_attempts)
            server_ip = found_ip or SERVER_IP
            server_port = found_port or SERVER_PORT
        
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.settimeout(RECV_TIMEOUT)
        self._sock.connect((server_ip, int(server_port)))

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

    def _search(self, timeout=1.0, attempts=2):
        msg = json.dumps({"id": "DISCOVER_SERVER"}).encode()

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as disc:
            disc.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            disc.settimeout(timeout)

            for _ in range(attempts):
                try:
                    _dump('[CLIENT], UDP_SEND broadcast', msg, verbose=self.verbose)
                    disc.sendto(msg, ('255.255.255.255', SEARCH_PORT))
                    
                    data, addr = disc.recvfrom(2048)
                    reply = json.loads(data.decode('utf-8', errors='ignore'))
                    _dump('[CLIENT] UDP_RECV from {addr}', data, verbose=self.verbose)

                    if reply.get('id') == "SERVER_HERE":
                        return addr[0], int(reply.get('server_port', SERVER_PORT))
                except Exception:
                    continue
        
        return None, None
    
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
        
        if wait_s is None:
            return bool(self._enqueue('START', want_data=False, wait_reply=True))
        
        self._cancel_timer(self.start_timer)
        self.start_timer = Timer(float(wait_s), lambda: self._enqueue('START', False, True))
        self.start_timer.daemon = True
        self._patch_join(self.start_timer)
        self.start_timer.start()

        return True
    
    def stop(self, wait_s=None):
        if self._finished:
            return False
        
        if wait_s is None:
            return bool(self._enqueue('STOP', want_data=False, wait_reply=True))
        
        self._cancel_timer(self.stop_timer)
        self.stop_timer = Timer(float(wait_s), lambda: self._enqueue('STOP', False, True))
        self.stop_timer.daemon = True
        self._patch_join(self.stop_timer)
        self.stop_timer.start()

        return True

    def finish(self):
        if self._finished:
            return
        
        self._finished = True

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

        self._net_stop.set()
        self._q.put(None)

        try:
            self._net_thread.join(timeout=1.0)
        except Exception:
            pass

        try:
            self._sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    client = PrairieClient(verbose=False)

    client.configure()
    print('[CLIENT] CONFIG')

    client.start()
    print('[CLIENT] START')

    client.stop(wait_s=31.0)
    client.stop_timer.isr_join()
    print('[CLIENT] STOP')

    try:
        while True:
            client.start(wait_s=1.0)
            client.start_timer.isr_join()
            print('[CLIENT] START')

            client.stop(wait_s=32.0)
            client.stop_timer.isr_join()
            print('[CLIENT] STOP')
    except KeyboardInterrupt:
        client.finish()
        print('[CLIENT] FINISH')

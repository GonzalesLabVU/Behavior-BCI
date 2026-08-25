"""
Serial protocol layer for the Arduino behavioral controller: framing,
acks, and the read/write threads that talk to the physical rig
"""

import time
import math
from threading import Thread, Event, Lock
from queue import Queue, Empty

from behavior_rig.timeutil import _get_ts


class ArduinoLink:
    EPHYS_START_STRING = "R1"
    EPHYS_STOP_STRING = "R2"
    FINISH_STRING = "S"
    RESTART_STRING = "R"
    ACK_STRING = "A"

    def __init__(self, ser, verbose=False, exceptions=None):
        """
        Initialize the serial link wrapper and reader thread.

        Args:
            ser: Open pyserial Serial object, or None for inactive mode.
        """
        self.ser = ser
        self.verbose = bool(verbose)
        self.active = ser is not None and ser.is_open
        self.exceptions = exceptions
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
        """
        Start the background serial reader when the link is active
        """
        if self.active:
            self._reader.start()

    def close(self):
        """
        Stop the reader and close the serial port if open
        """
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
        """
        Schedule a command to be sent after an optional delay

        Args:
            cmd: Command text to send
            delay_s: Delay before sending in seconds
            timeout_s: Maximum seconds to wait for ACK

        Returns:
            True after the background send worker is started
        """
        def worker():
            """
            Delay and send the scheduled command from a background thread
            """
            if delay_s > 0:
                time.sleep(float(delay_s))

            self.send_and_wait(cmd, timeout_s=timeout_s)
        
        Thread(target=worker, daemon=True).start()
        return True

    def start_ephys(self, timeout_s=5.0):
        """
        Send the ephys start command

        Args:
            timeout_s: Maximum seconds to wait for ACK

        Returns:
            True when the command succeeds or the link is inactive
        """
        return self.send_and_wait(self.EPHYS_START_STRING, timeout_s=timeout_s)

    def stop_ephys(self, session_data=None, timeout_s=5.0, safe=False):
        """
        Stop ephys recording

        Args:
            session_data: Optional SessionData instance used for idempotent safe-stop tracking
            timeout_s: Maximum seconds to wait for ACK
            safe: If True, cache failures instead of raising and skip duplicate stops

        Returns:
            True when stopped, skipped, or inactive
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
                if self.exceptions is not None:
                    self.exceptions.cache(e, "main.ephys_stop")

            return True

        return self.send_and_wait(self.EPHYS_STOP_STRING, timeout_s=timeout_s)

    def start_imaging(self, delay_s=0.0, timeout_s=5.0):
        """
        Send or schedule the imaging start TTL command

        Args:
            delay_s: Delay before sending in seconds
            timeout_s: Maximum seconds to wait for ACK

        Returns:
            True after the command is scheduled
        """
        return self.send_after("img_start", delay_s=delay_s, timeout_s=timeout_s)

    def stop_imaging(self, delay_s=0.0, timeout_s=5.0):
        """
        Send or schedule the imaging stop TTL command

        Args:
            delay_s: Delay before sending in seconds
            timeout_s: Maximum seconds to wait for ACK

        Returns:
            True after the command is scheduled
        """
        return self.send_after("img_stop", delay_s=delay_s, timeout_s=timeout_s)

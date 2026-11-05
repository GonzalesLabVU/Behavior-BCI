import os
import random
import threading
import time
from collections import deque

import requests
from gspread.exceptions import APIError
from gspread.http_client import HTTPClient

MAX_TRIES = 7
RETRY_CODES = {408, 429}

_stamps = deque()
_stamps_lock = threading.Lock()


def _throttle():
    limit = int(os.getenv("SHEETS_MAX_REQ_PER_MIN", "30"))

    while True:
        with _stamps_lock:
            now = time.monotonic()

            while _stamps and now - _stamps[0] >= 60.0:
                _stamps.popleft()

            if len(_stamps) < limit:
                _stamps.append(now)
                return

            wait = 60.0 - (now - _stamps[0])

        time.sleep(wait + random.uniform(0.05, 0.5))


class ResilientHTTPClient(HTTPClient):
    def request(self, *args, **kwargs):
        delay = 2.0

        for attempt in range(1, MAX_TRIES + 1):
            _throttle()

            try:
                return super().request(*args, **kwargs)
            except APIError as e:
                if attempt == MAX_TRIES or not (e.code in RETRY_CODES or e.code >= 500):
                    raise
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt == MAX_TRIES:
                    raise

            time.sleep(delay + random.uniform(0, delay / 2))
            delay = min(delay * 2, 60.0)

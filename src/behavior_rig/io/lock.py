"""
Cross-process locking around the shared Sheets workbook, preventing
concurrent writers from corrupting a save
"""

import os
import time
import random
import socket
import uuid

from behavior_rig.config import get_sheets_client
from behavior_rig.timeutil import _now


class FileLock:
    POLL_S = 5.0
    RETRY_S = 5.0
    LEASE_S = 180
    RESET_S = 60
    TIMEOUT_S = 300

    TAG = "------ LOCK ------"
    TAG_RANGE = "A1"
    META_RANGE = "A2:D2"

    def __init__(self, workbook_id, owner):
        """Initialize a Google Sheets worksheet lock.

        Args:
            workbook_id: Google Sheets workbook ID to protect.
            owner: Unique owner string for this lock attempt.
        """
        self.poll_s = float(self.POLL_S)
        self.retry_s = float(self.RETRY_S)
        self.lease_s = int(self.LEASE_S)
        self.reset_s = int(self.RESET_S)
        self.timeout_s = int(self.TIMEOUT_S)

        self.client = get_sheets_client()
        self.workbook_id = workbook_id
        self.owner = owner
        self.token = uuid.uuid4().hex

        self.sheet_name = None
        self.created = 0
        self.expires = 0

        self.wb = None
        self.ws = None

    def _confirm_ws(self, ws, err_msg='Lock lost'):
        """Confirm that a worksheet still represents this lock.

        Args:
            ws: Worksheet to validate.
            err_msg: Error message used if validation fails.
        """
        meta = self._get_resource("meta", ws=ws, err_msg=err_msg)
        self._ensure_control(meta['owner'], meta['token'], err_msg=err_msg)

    @staticmethod
    def _get_client_id():
        return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

    def _get_resource(self, resource="workbook", ws=None, err_msg = "Lock tag missing (lock lost)"):
        """Open or read lock-related Google Sheets resources.

        Args:
            resource: One of "workbook", "worksheet", or "meta".
            ws: Worksheet to read metadata from when resource is "meta".
            err_msg: Error message used when lock metadata is missing or invalid.

        Returns:
            Workbook, worksheet, or metadata dictionary depending on resource.
        """
        if resource == "workbook":
            self.wb = self.client.open_by_key(self.workbook_id)
            return self.wb
        
        if resource == "worksheet":
            if self.sheet_name is None:
                raise RuntimeError("Lock not acquired (sheet_name is None)")
            
            if self.wb is None:
                self._get_resource("workbook")

            try:
                self.ws = self.wb.worksheet(self.sheet_name)
            except Exception:
                self._get_resource("workbook")
                self.ws = self.wb.worksheet(self.sheet_name)

            return self.ws
        
        if resource == "meta":
            if ws is None:
                ws = self._get_resource("worksheet")

            try:
                vals = ws.get('A1:D2')
            except Exception as e:
                raise RuntimeError(err_msg) from e
            
            tag = (vals[0][0] if vals and vals[0] else "") if vals else ""
            if (tag or "") != self.TAG:
                raise RuntimeError(err_msg)
            
            row = vals[1] if len(vals) > 1 and vals[1] else ["", "", "0", "0"]
            owner = str(row[0] or "")
            token = str(row[1] or "")

            try:
                created_ts = int(str(row[2] or "0"))
            except Exception:
                created_ts = 0

            try:
                expires_ts = int(str(row[3] or "0"))
            except Exception:
                expires_ts = 0

            return {
                "tag": tag,
                "owner": owner,
                "token": token,
                "created": created_ts,
                "expires": expires_ts,
                "info": row
                }
        
        raise ValueError(f"Unknown lock resource: {resource!r}")

    def _ensure_control(self, owner, token, err_msg='Lock lost'):
        """Verify that the supplied owner and token match this lock.

        Args:
            owner: Owner string read from the lock sheet.
            token: Token string read from the lock sheet.
            err_msg: Error message used if ownership does not match.
        """
        if owner != self.owner or token != self.token:
            raise RuntimeError(err_msg)

    def sleep(self, dur_s, jitter_ms=1000):
        """Sleep with optional random jitter.

        Args:
            dur_s: Base sleep duration in seconds.
            jitter_ms: Maximum additional jitter in milliseconds.
        """
        if dur_s < 0:
            dur_s = 0.0
        
        if jitter_ms and jitter_ms > 0:
            dur_s += random.random() * (jitter_ms / 1000.0)
        
        time.sleep(dur_s)

    def acquire(self):
        """Acquire the workbook lock, waiting until this client owns it.

        Returns:
            This FileLock instance after acquisition.
        """
        wb = self._get_resource("workbook")

        deadline = time.monotonic() + self.timeout_s
        attempt = 0
        created_ts = _now()

        print('Acquiring lock...', end='\r', flush=True)

        def q_sheet(title):
            """Quote a worksheet title for a Sheets range.

            Args:
                title: Worksheet title.

            Returns:
                Quoted worksheet title.
            """
            return "'" + title.replace("'", "''") + "'"
        
        def to_int(x, default=0):
            """Convert a value to int with a fallback.

            Args:
                x: Value to convert.
                default: Value returned when conversion fails.

            Returns:
                Converted integer or default.
            """
            try:
                return int(x)
            except Exception:
                return default
            
        def scan_locks():
            """Scan workbook sheets for active lock records.

            Returns:
                List of lock metadata dictionaries.
            """
            meta = wb.fetch_sheet_metadata(params={'fields': 'sheets(properties(sheetId,title))'})
            sheets = meta.get('sheets', [])
            
            if not sheets:
                return []
            
            props = [s.get('properties', {}) for s in sheets]
            titles = [p.get('title', '') for p in props]
            ids = [p.get('sheetId', 0) for p in props]

            ranges = [f'{q_sheet(t)}!A1:D2' for t in titles]
            resp = wb.values_batch_get(ranges)
            vrs = resp.get('valueRanges', [])

            assert len(titles) == len(ids) == len(vrs)

            locks = []

            for title, id, vr in zip(titles, ids, vrs):
                values = vr.get('values', [])
                if not values or not values[0]:
                    continue

                tag = values[0][0] if values[0] else ""
                if (tag or "") != self.TAG:
                    continue

                row = values[1] if len(values) > 1 and values[1] else []
                owner = str(row[0]) if len(row) > 0 else ""
                token = str(row[1]) if len(row) > 1 else ""
                created = to_int(row[2], 0) if len(row) > 2 else 0
                expires = to_int(row[3], 0) if len(row) > 3 else 0

                locks.append({
                    'sheetId': id,
                    'title': title,
                    'owner': owner,
                    'token': token,
                    'created': created,
                    'expires': expires
                    })
            
            return locks

        def batch_delete(ids):
            """Delete worksheets by sheet ID, ignoring delete failures.

            Args:
                ids: Iterable of sheet IDs to delete.
            """
            if not ids:
                return
            
            req = [{'deleteSheet': {'sheetId': id}} for id in ids]

            try:
                wb.batch_update({'requests': req})
            except Exception:
                pass

        def is_mine(lock):
            """Check whether a scanned lock belongs to this FileLock.

            Args:
                lock: Lock metadata dictionary.

            Returns:
                True when owner and token match this instance.
            """
            return lock.get('owner') == self.owner and lock.get('token') == self.token

        while time.monotonic() < deadline:
            attempt += 1
            print(f'Acquiring lock...[TRIES={attempt}]', flush=True)

            now = _now()

            try:
                locks = scan_locks()
            except Exception:
                self.sleep(self.retry_s, jitter_ms=750)
                wb = self._get_resource("workbook")
                continue

            expired_ids = [lock['sheetId'] for lock in locks if lock['expires'] and now >= lock['expires']]
            if expired_ids:
                batch_delete(expired_ids)
                self.sleep(0.1, jitter_ms=100)
                continue

            active = [lock for lock in locks if lock['expires'] and now < lock['expires']]
            if active:
                winner = min(active, key=lambda lock: (lock['created'], lock['token'], lock['sheetId']))

                if is_mine(winner):
                    self.sheet_name = winner['title']
                    self.created = int(winner['created'] or created_ts)
                    self.expires = int(winner['expires'] or 0)
                    self.wb = wb
                    self.ws = None

                    print("\r\033[2KLock acquired", flush=True)
                    return self
                
                remaining = int(winner['expires'] or 0) - now
                sleep_s = self.poll_s if remaining > self.poll_s else max(0.2, float(remaining))

                self.sleep(sleep_s, jitter_ms=350)
                continue

            try:
                my_lock = wb.add_worksheet(title=self.owner, rows=10, cols=10)
            except Exception:
                self.sleep(self.poll_s, jitter_ms=750)
                wb = self._get_resource("workbook")
                continue

            try:
                expires_ts = _now() + self.lease_s
                my_meta = [self.owner, self.token, str(created_ts), str(expires_ts)]

                my_lock.batch_update([
                    {'range': self.TAG_RANGE, 'values': [[self.TAG]]},
                    {'range': self.META_RANGE, 'values': [my_meta]}
                    ])
            except Exception:
                try:
                    wb.del_worksheet(my_lock)
                except Exception:
                    pass

                self.sleep(self.poll_s, jitter_ms=750)
                wb = self._get_resource("workbook")
                continue

            try:
                locks2 = scan_locks()
            except Exception:
                self.sleep(self.poll_s, jitter_ms=750)
                continue

            now2 = _now()

            expired2 = [lock['sheetId'] for lock in locks2 if lock['expires'] and now2 >= lock['expires']]
            if expired2:
                batch_delete(expired2)
                continue

            active2 = [lock for lock in locks2 if lock['expires'] and now2 < lock['expires']]
            if not active2:
                self.sleep(0.2, jitter_ms=200)
                continue

            winner2 = min(active2, key=lambda lock: (lock['created'], lock['token'], lock['sheetId']))

            if is_mine(winner2):
                self.sheet_name = winner2['title']
                self.created = int(winner2['created'] or created_ts)
                self.expires = int(winner2['expires'] or 0)
                self.wb = wb
                self.ws = None

                print("\r\033[2KLock acquired", flush=True)
                return self
            
            my_id = None

            for lock in active2:
                if is_mine(lock):
                    my_id = lock['sheetId']
                    break
            
            if my_id:
                batch_delete([my_id])
            
            self.sleep(0.5, jitter_ms=500)
        
        raise TimeoutError('Timed out during lock acquisition')

    def update(self):
        """Refresh local lock state from the worksheet.

        Returns:
            Remaining lock lease time in seconds.
        """
        ws = self._get_resource("worksheet")
        meta = self._get_resource("meta", ws=ws)

        owner = meta['owner']
        token = meta['token']
        created_ts = meta['created']
        expires_ts = meta['expires']

        self._ensure_control(owner, token, err_msg='Lock lost during update')

        self.created = int(created_ts or self.created)
        self.expires = int(expires_ts or 0)

        return int(self.expires or 0) - _now()

    def reset(self):
        """Extend the lock lease when it is near expiration.

        Returns:
            Remaining lock lease time in seconds.
        """
        remaining = int(self.expires or 0) - _now()
        if remaining >= self.reset_s:
            return remaining
        
        ws = self._get_resource("worksheet")
        meta = self._get_resource("meta", ws=ws)

        owner = meta['owner']
        token = meta['token']
        created_ts = meta['created']
        expires_ts = meta['expires']

        if not created_ts:
            created_ts = self.created or _now()
        
        self._ensure_control(owner, token, err_msg='Lock lost before reset')

        remaining = expires_ts - _now()
        if remaining >= self.reset_s:
            self.created = int(created_ts or self.created)
            self.expires = int(expires_ts or 0)

            return remaining
        
        new_expires = _now() + self.lease_s
        new_meta = [self.owner, self.token, str(created_ts or _now()), str(new_expires)]

        try:
            ws.update(self.META_RANGE, [new_meta])
        except Exception as e:
            raise RuntimeError('Failed to reset lock') from e
        
        self._confirm_ws(ws, err_msg='Lock lost after reset')

        meta2 = self._get_resource("meta", ws=ws, err_msg="Lock lost after reset")

        created_ts2 = meta2['created']
        expires_ts2 = meta2['expires']
        
        self.created = int(created_ts2 or created_ts or self.created)
        self.expires = int(expires_ts2 or new_expires)
        
        return int(self.expires or 0) - _now()

    def release(self, retries=5):
        """
        Release the workbook lock by deleting its worksheet.

        Args:
            retries: Number of release attempts before raising.

        Returns:
            True when the lock is released or no longer owned.
        """
        last_e = RuntimeError('Lock release failed\n')

        for attempt in range(retries):
            print(f'Releasing lock...[TRIES={attempt + 1}]', flush=True)

            try:
                wb = self.client.open_by_key(self.workbook_id)
                ws = wb.worksheet(self.sheet_name or self.owner)

                try:
                    meta = self._get_resource("meta", ws=ws)

                    owner = meta['owner']
                    token = meta['token']
                except RuntimeError:
                    print("\r\033[2KLock released", flush=True)
                    return True
                
                try:
                    self._ensure_control(owner, token, err_msg='Lock released (not owned)\n')
                except RuntimeError:
                    print("\r\033[2KLock released\n", flush=True)
                    return True
                
                wb.del_worksheet(ws)

                print("\r\033[2KLock released\n", flush=True)
                return True
            except Exception as e:
                last_e = e
                self.sleep(self.retry_s, jitter_ms=2500)
        
        raise last_e

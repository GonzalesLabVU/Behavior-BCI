"""
Google Sheets export: formats and uploads a finalized session record to
the lab's shared spreadsheet, independent of local saving
"""

import os
import json
import uuid
from itertools import zip_longest

from gspread.utils import rowcol_to_a1

from behavior_rig.config import get_sheets_client, DATA_DIR
from behavior_rig.errors import ExceptionInterface
from behavior_rig.hardware.system import InterfaceObject
from behavior_rig.io.lock import FileLock
from behavior_rig.ui.trainer import TrainerInterface, _is_affirmative


class SaveInterface(InterfaceObject):
    interface_name = "save"

    VALID_SESSION_S = 5 * 60

    def __init__(self, trainer=None, exceptions=None):
        self.trainer = trainer or TrainerInterface()
        self.exceptions = exceptions or ExceptionInterface()

    def _build_rows(self, session_data, dtype):
        """
        Build worksheet rows for metadata or imaging output.

        Args:
            session_data: SessionData instance.
            dtype: Row type to build. Expected values are "meta" or "img".

        Returns:
            List of two-column rows.
        """
        if dtype == "meta":
            session_data._ensure_session_tracking()

            client_id = str(os.getenv("CLIENT_ID"))

            cfg = session_data.meta.get('trial_config', []) or []
            easy_trials = [c['trial'] for c in cfg if c.get('is_easy') is True]
            normal_trials = [c['trial'] for c in cfg if c.get('is_easy') is False]

            left_targets = [c['trial'] for c in cfg if c.get('side') == "L"]
            right_targets = [c['trial'] for c in cfg if c.get('side') == "R"]
            both_targets = [c['trial'] for c in cfg if c.get('side') == "B"]

            meta_pairs = [
                ("client", client_id),
                ("imaging_active", session_data.meta.get('imaging_active', False)),
                ("ephys_active", session_data.meta.get('ephys_active', False)),
                ("K1", session_data.meta.get('K1', 5)),
                ("K2", session_data.meta.get('K2', None)),
                ("easy_trials", easy_trials),
                ("normal_trials", normal_trials),
                ("left_targets", left_targets),
                ("right_targets", right_targets),
                ("both_targets", both_targets)
                ]

            out = []
            for key, value in meta_pairs:
                if isinstance(value, (list, tuple)):
                    if len(value) == 0:
                        out.append([key, "None"])
                    else:
                        out.append([key, value[0]])

                        for v in value[1:]:
                            out.append(["", v])
                else:
                    out.append([key, "" if value is None else value])

            return out

        if dtype == "img":
            starts = session_data.img.get('start_ts') or []
            stops = session_data.img.get('stop_ts') or []

            out = []
            for t1, t2 in zip_longest(starts, stops, fillvalue=None):
                if t1 is not None:
                    out.append([str(t1), "start"])
                if t2 is not None:
                    out.append([str(t2), "stop"])

            return out

        raise ValueError(f"Unsupported row type: {dtype!r}")

    def _align_cells(self, wb, ws, r1, c1, r2, c2):
        sheet_id = ws._properties["sheetId"]
        req = {
            "requests": [{
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": r1 - 1,
                        "endRowIndex": r2,
                        "startColumnIndex": c1 - 1,
                        "endColumnIndex": c2
                        },
                    "cell": {
                        "userEnteredFormat": {
                            "horizontalAlignment": "LEFT"
                            }
                        },
                    "fields": "userEnteredFormat.horizontalAlignment"
                    }
                }]
            }
        
        wb.batch_update(req)

    def resolve_protocol(self, session_data):
        """
        Resolve duplicate-session handling before saving.

        Args:
            session_data: SessionData instance to save.

        Returns:
            True when saving should continue, otherwise False.
        """
        def _norm(x):
            return (x or "").strip()
        
        def _get_existing_duration(ws, start_col):
            try:
                vals = ws.get(f"{rowcol_to_a1(4, start_col)}:{rowcol_to_a1(ws.row_count, start_col + 1)}")
            except Exception:
                return None

            pending_key = None

            for row in vals:
                key = str(row[0]).strip() if len(row) > 0 else ""
                value = row[1] if len(row) > 1 else ""

                if key:
                    pending_key = key

                if pending_key == "duration_sec":
                    try:
                        return float(value)
                    except Exception:
                        return None

            return None
        
        def _find_existing_block(wb):
            target_date = _norm(session_data.meta.get('date', ""))
            target_animal = _norm(f"Animal {session_data.meta.get('animal', '')}")
            target_phase = _norm(f"Phase {session_data.meta.get('phase', '')}")

            try:
                ws = wb.worksheet("Metadata")
            except Exception:
                return None

            max_col = len(ws.row_values(2))
            if max_col <= 0:
                return None

            header_rng = f"A1:{rowcol_to_a1(2, max_col)}"
            header = ws.get(header_rng)

            row1 = header[0] if len(header) > 0 else []
            row2 = header[1] if len(header) > 1 else []

            for c in range(1, max_col + 1, 2):
                date_val = _norm(row1[c-1] if (c - 1) < len(row1) else "")
                animal_val = _norm(row2[c-1] if (c - 1) < len(row2) else "")
                phase_val = _norm(row2[c] if c < len(row2) else "")

                if (date_val == target_date) and (animal_val == target_animal) and (phase_val == target_phase):
                    return {
                        "worksheet": ws,
                        "start_col": c,
                        "duration_sec": _get_existing_duration(ws, c)
                        }

            return None

        session_data.overwrite_confirmed = False

        while True:
            workbook_id = session_data.meta.get('workbook_id')
            if not workbook_id:
                session_data.overwrite_confirmed = True
                return True

            wb = get_sheets_client.open_by_key(workbook_id)
            existing = _find_existing_block(wb)
            if existing is None:
                session_data.overwrite_confirmed = True
                return True

            prev_duration = existing.get('duration_sec')
            curr_duration = float(session_data.meta.get('duration_sec') or 0)

            auto_overwrite = (prev_duration is not None
                              and prev_duration < self.VALID_SESSION_S
                              and curr_duration > self.VALID_SESSION_S)
            if auto_overwrite:
                session_data.overwrite_confirmed = True
                return True

            overwrite_raw = input("A training session has already been recorded for this animal/phase today.\n"
                                  "Do you want to overwrite the earlier session with this session's data? [y/N]:  ")
            if _is_affirmative(overwrite_raw):
                session_data.overwrite_confirmed = True
                return True

            exit_raw = input("Exit this session without saving? [y/N]:  ")
            if _is_affirmative(exit_raw):
                session_data.overwrite_confirmed = False
                return False

            self.trainer.confirm_meta(session_data)

    def save_data(self, session_data):
        """
        Save session data into the configured Google Sheets workbook.

        Args:
            session_data: SessionData instance to write.

        Returns:
            True when saving completes, or None when no workbook is configured.
        """
        workbook_id = session_data.meta.get("workbook_id")

        if not workbook_id:
            print('[WARNING] No data recorded (skipping save)')
            return
        
        client_id = FileLock._get_client_id()

        def _norm(x):
            """Normalize a header value for comparison.

            Args:
                x: Value to normalize.

            Returns:
                Stripped string value.
            """
            return (x or "").strip()
        
        def _target_headers():
            """Build the date, animal, and phase headers for the session.

            Returns:
                Tuple of normalized date, animal, and phase header strings.
            """
            d = _norm(session_data.meta.get("date", ""))
            a = _norm(f"Animal {session_data.meta.get('animal', '')}")
            p = _norm(f"Phase {session_data.meta.get('phase', '')}")

            return d, a, p
        
        def _find_cols(ws):
            """Find or allocate the two-column block for this session.

            Args:
                ws: Worksheet to inspect.

            Returns:
                Tuple of starting column and whether it overwrites existing data.
            """
            target_d, target_a, target_p = _target_headers()

            max_col = len(ws.row_values(2))
            if max_col <= 0:
                return 1, False
            
            header_rng = f'A1:{rowcol_to_a1(2, max_col)}'
            header = ws.get(header_rng)
            row1 = header[0] if len(header) > 0 else []
            row2 = header[1] if len(header) > 1 else []

            for c in range(1, max_col + 1, 2):
                d_val = _norm(row1[c-1] if (c - 1) < len(row1) else "")
                a_val = _norm(row2[c-1] if (c-1) < len(row2) else "")
                p_val = _norm(row2[c] if c < len(row2) else "")

                if (d_val == target_d) and (a_val == target_a) and (p_val == target_p):
                    return c, True
            
            new_col = (((max_col + 1) // 2) * 2) + 1
            return new_col, False

        def _batch_write_cols(ws, start_row, start_col, data, chunk_rows=2000, group_chunks=10):
            """Write two-column data to a worksheet in grouped batches.

            Args:
                ws: Worksheet to update.
                start_row: Starting row, one-based.
                start_col: Starting column, one-based.
                data: Rows to write.
                chunk_rows: Maximum rows per value range.
                group_chunks: Maximum ranges per batch update request.
            """
            sheet = ws.spreadsheet
            name = ws.title

            def _rng(r1, c1, r2, c2):
                """Build an A1 range for a worksheet rectangle.

                Args:
                    r1: Starting row, one-based.
                    c1: Starting column, one-based.
                    r2: Ending row, one-based.
                    c2: Ending column, one-based.

                Returns:
                    A1 notation range string.
                """
                return f'{name}!{rowcol_to_a1(r1, c1)}:{rowcol_to_a1(r2, c2)}'
            
            req = []
            n = len(data)

            for i in range(0, n, chunk_rows):
                chunk = data[i:i+chunk_rows]
                r1 = start_row + i
                r2 = r1 + len(chunk) - 1
                c1 = start_col
                c2 = start_col + 1

                req.append({'range': _rng(r1, c1, r2, c2), 'values': chunk})

                if len(req) >= group_chunks:
                    sheet.values_batch_update(body={'valueInputOption': 'RAW', 'data': req})
                    req.clear()
            
            if req:
                sheet.values_batch_update(body={'valueInputOption': 'RAW', 'data': req})

        lock = None

        try:
            lock = FileLock(workbook_id, owner=client_id).acquire()

            wb = get_sheets_client.open_by_key(workbook_id)
            lock.wb = wb

            sheet_map = (
                ("evt", "Event"),
                ("enc", "Encoder"),
                ("img", "Imaging"),
                ("meta", "Metadata")
                )

            for dtype, sheet_name in sheet_map:
                match dtype:
                    case "meta":
                        data_rows = self._build_rows(session_data, "meta")
                        data = data_rows
                        n_rows = len(data_rows)
                        label = "metadata"
                    case "img":
                        data_rows = self._build_rows(session_data, "img")
                        data = data_rows
                        n_rows = len(data_rows)
                        label = 'imaging'
                    case _:
                        d = getattr(session_data, dtype)
                        n_rows = len(d['timestamps'])
                        data = [[ts, val] for ts, val in zip(d['timestamps'], d['values'])]
                        label = sheet_name.lower()

                if n_rows == 0:
                    continue

                if dtype != 'meta':
                    print(f'Writing {label} data...', flush=True)
                else:
                    print('Writing metadata...', flush=True)

                lock.update()
                lock.reset()

                try:
                    ws = wb.worksheet(sheet_name)
                except Exception:
                    ws = wb.add_worksheet(title=sheet_name, rows=200, cols=26)
                
                lock.update()
                lock.reset()

                start_col, overwrite = _find_cols(ws)
                if overwrite and not getattr(session_data, "overwrite_confirmed", False):
                    raise RuntimeError("Refusing to overwrite existing session data without save-protocol confirmation")

                needed_cols = start_col + 1

                if ws.col_count < needed_cols:
                    ws.add_cols(needed_cols - ws.col_count)

                if overwrite:
                    clear_rng = f'{rowcol_to_a1(1, start_col)}:{rowcol_to_a1(ws.row_count, start_col + 1)}'
                    
                    lock.update()
                    lock.reset()

                    ws.batch_clear([clear_rng])
                
                header_rng = f'{rowcol_to_a1(1, start_col)}:{rowcol_to_a1(2, start_col + 1)}'
                skip_rng = f'{rowcol_to_a1(3, start_col)}:{rowcol_to_a1(3, start_col + 1)}'

                header = [
                    [session_data.meta['date'], ""],
                    [f"Animal {session_data.meta['animal']}", f"Phase {session_data.meta['phase']}"]
                    ]
                
                lock.update()
                lock.reset()

                ws.batch_update([
                    {'range': header_rng, 'values': header},
                    {'range': skip_rng, 'values': [["", ""]]}
                    ])

                needed_rows = 3 + n_rows
                if ws.row_count < needed_rows:
                    ws.add_rows(needed_rows - ws.row_count)

                lock.update()
                lock.reset()

                _batch_write_cols(ws, start_row=4, start_col=start_col, data=data)

                if dtype == 'meta':
                    r1 = 1
                    r2 = 3 + n_rows
                    c1 = start_col
                    c2 = start_col + 1

                    lock.update()
                    lock.reset()

                    self._align_cells(wb, ws, r1, c1, r2, c2)

                print("\r\033[2K", end="", flush=True)
            
            return True
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception as e:
                    self.exceptions.set_session(session_data.meta.get("animal", "UNKNOWN"),
                                                session_data.meta.get("phase", "0"))
                    self.exceptions.log_and_commit(e)

    def save_session(self, session_data):
        """
        Save session data to Google Sheets with local fallback on failure.

        Args:
            session_data: SessionData instance to persist.

        Returns:
            True when the primary save succeeds, otherwise False.
        """
        animal = session_data.meta.get('animal', 'UNKNOWN') if session_data else 'UNKNOWN'
        phase = session_data.meta.get('phase', '0') if session_data else '0'

        self.exceptions.set_session(animal, phase)

        try:
            self.save_data(session_data)
            return True
        except Exception as e:
            try:
                self.fallback_save(session_data)
            except Exception as e2:
                self.exceptions.log(e2)

            self.exceptions.log(e)
            return False
        finally:
            try:
                self.exceptions.commit()
            except Exception:
                pass

    def fallback_save(self, session_data):
        """
        Save session data using the local fallback path.

        Args:
            session_data: SessionData instance to persist.

        Returns:
            Path to the local fallback file.
        """
        animal = str(session_data.meta.get('animal', 'UNKNOWN'))
        phase = str(session_data.meta.get('phase', '0'))
        date = str(session_data.meta.get('date', '0000-00-00')).replace('/', '.')
        rand = uuid.uuid4().hex[:6]

        out_path = DATA_DIR / f"date={date}_animal={animal}_phase={phase}_id={rand}.json"
        payload = session_data.to_dict()

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=4)

        print("\r\033[2K", end="", flush=True)
        print(f"[WARNING] Saved session data locally to {out_path.name}", flush=True)

        return out_path

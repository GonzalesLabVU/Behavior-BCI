import os
import keyboard
from datetime import datetime, timezone
from gspread.utils import rowcol_to_a1

from behavior_rig.config import CONFIG_DIR, get_sheets_client
from behavior_rig.hardware.system import InterfaceObject


ENV_PATH = CONFIG_DIR / ".env"
CREDENTIALS_PATH = CONFIG_DIR / "credentials.json"

API_SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]

DB_SHEET_NAME = "Dashboard"
CLIENT_START_COLS = {
    "BEHAVIOR": 1,
    "IMAGING": 3,
    "DEVELOPMENT": 5
    }
FIELD_ROWS = {
    "status": 1,
    "animal": 2,
    "phase": 3
    }

_ENV_CACHE = None


class DashboardInterface(InterfaceObject):
    interface_name = "dashboard"

    def __init__(self, client_id=None):
        self.client_id = str(client_id or os.getenv("CLIENT_ID")).strip().upper()

    @staticmethod
    def _utc_iso():
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def _safe_write(self, fields, timestamp=None):
        try:
            return write_fields(self.client_id,
                                fields,
                                timestamp=timestamp or self._utc_iso())
        except Exception as exc:
            print(f"[WARNING] Dashboard update failed: {type(exc).__name__}: {exc}",
                  flush=True)
            return None

    def notify_start(self, session_data):
        start_utc = self._utc_iso()

        self._safe_write({
            "status": "running",
            "animal": session_data.meta.get("animal", ""),
            "phase": session_data.meta.get("phase", "")
            }, timestamp=start_utc)

    def notify_finish(self):
        self._safe_write({"status": "finished"}, timestamp=self._utc_iso())

        original_read_key = keyboard.read_key

        def _read_key_and_set_idle(*args, **kwargs):
            try:
                return original_read_key(*args, **kwargs)
            finally:
                self._safe_write({
                    "status": "idle",
                    "animal": "",
                    "phase": ""
                    }, timestamp=self._utc_iso())

                keyboard.read_key = original_read_key

        keyboard.read_key = _read_key_and_set_idle


def _load_env():
    global _ENV_CACHE

    if _ENV_CACHE is not None:
        return _ENV_CACHE

    data = {}

    if ENV_PATH.exists():
        for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()

            if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
                val = val[1:-1]

            data[key] = val

    _ENV_CACHE = data
    return data


def _get_env(name):
    value = os.getenv(name)
    if value:
        return value

    value = _load_env().get(name)
    if value:
        return value

    raise RuntimeError(f"{name} not found in {ENV_PATH.name}")


def _utc_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _norm_client_id(client_id):
    client_id = str(client_id or "").strip().upper()

    if client_id in {"BEH", "BEHAVIOR"}:
        return "BEHAVIOR"
    if client_id in {"IMG", "IMAGING"}:
        return "IMAGING"
    if client_id in {"DEV", "DEVELOPMENT"}:
        return "DEVELOPMENT"

    raise ValueError(f"Unknown dashboard client ID: {client_id!r}")


def _dashboard_worksheet():
    dashboard_id = _get_env("DASHBOARD_ID")
    workbook = get_sheets_client().open_by_key(dashboard_id)

    try:
        worksheet = workbook.worksheet(DB_SHEET_NAME)
    except Exception:
        worksheet = workbook.add_worksheet(title=DB_SHEET_NAME,
                                           rows=13,
                                           cols=6)

    if worksheet.row_count < 3:
        worksheet.add_rows(3 - worksheet.row_count)

    if worksheet.col_count < 6:
        worksheet.add_cols(6 - worksheet.col_count)

    return worksheet


def write_fields(client_id, fields, timestamp=None):
    client_id = _norm_client_id(client_id)
    start_col = CLIENT_START_COLS[client_id]
    timestamp = timestamp or _utc_iso()

    updates = []

    for key, value in fields.items():
        key = str(key).strip().lower()
        if key not in FIELD_ROWS:
            raise ValueError(f"Unsupported dashboard field: {key!r}")

        row = FIELD_ROWS[key]
        rng_name = (f"{rowcol_to_a1(row, start_col)}:"
                    f"{rowcol_to_a1(row, start_col + 1)}")

        updates.append({
            "range": rng_name,
            "values": [[timestamp, "" if value is None else str(value)]]
            })

    if not updates:
        return None

    worksheet = _dashboard_worksheet()
    worksheet.batch_update(updates)

    return {
        "ok": True,
        "client_id": client_id,
        "timestamp": timestamp,
        "fields": fields
        }

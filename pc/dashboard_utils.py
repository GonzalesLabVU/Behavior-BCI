import json
import os
from datetime import datetime, timezone
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials
from gspread.utils import rowcol_to_a1


ENV_PATH = Path(__file__).resolve().parent / ".env"
CREDENTIALS_PATH = Path(__file__).resolve().parent / "credentials.json"
API_SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]

SESSION_LOG_SHEET_NAME = "Recent Sessions"
SESSION_LOG_HEADERS = ["Date", "Platform", "Animal", "Phase", "Performance", "Time"]
SESSION_LOG_PLATFORM_NAMES = {
    "BEHAVIOR": "Behavior",
    "IMAGING": "Imaging",
    "DEVELOPMENT": "Development"
    }

NOTIFICATIONS_SHEET_NAME = "Notifications"
NOTIFICATIONS_HEADERS = ["Timestamp", "Message"]

DB_SHEET_NAME = "Dashboard"
CLIENT_START_COLS = {
    "BEHAVIOR": 1,
    "IMAGING": 3,
    "DEVELOPMENT": 5
    }
FIELD_ROWS = {
    "status": 1,
    "animal": 2,
    "phase": 3,
    "performance": 4
    }

_REQUIRED_ROW_COUNT = max(FIELD_ROWS.values())
_ENV_CACHE = None
_CLIENT_CACHE = None


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


def _format_utc(start_utc):
    if not start_utc:
        return ""

    try:
        when = datetime.fromisoformat(str(start_utc).replace("Z", "+00:00"))
    except ValueError:
        return ""

    when = when.astimezone()
    hour12 = when.strftime("%I").lstrip("0") or "12"

    return f"{hour12}:{when.strftime('%M %p')}"


def _norm_client_id(client_id):
    client_id = str(client_id or "").strip().upper()

    if client_id in {"BEH", "BEHAVIOR"}:
        return "BEHAVIOR"
    if client_id in {"IMG", "IMAGING"}:
        return "IMAGING"
    if client_id in {"DEV", "DEVELOPMENT"}:
        return "DEVELOPMENT"

    raise ValueError(f"Unknown dashboard client ID: {client_id!r}")


def _build_client():
    global _CLIENT_CACHE

    if _CLIENT_CACHE is not None:
        return _CLIENT_CACHE

    if CREDENTIALS_PATH.exists():
        creds = Credentials.from_service_account_file(CREDENTIALS_PATH,
                                                      scopes=API_SCOPES)
    else:
        raw = _get_env("GOOGLE_SERVICE_ACCOUNT_JSON")
        creds = Credentials.from_service_account_info(json.loads(raw),
                                                      scopes=API_SCOPES)

    _CLIENT_CACHE = gspread.authorize(creds)
    return _CLIENT_CACHE


def _dashboard_worksheet():
    dashboard_id = _get_env("DASHBOARD_ID")
    workbook = _build_client().open_by_key(dashboard_id)

    try:
        worksheet = workbook.worksheet(DB_SHEET_NAME)
    except Exception:
        worksheet = workbook.add_worksheet(title=DB_SHEET_NAME,
                                           rows=13,
                                           cols=6)

    if worksheet.row_count < _REQUIRED_ROW_COUNT:
        worksheet.add_rows(_REQUIRED_ROW_COUNT - worksheet.row_count)

    if worksheet.col_count < 6:
        worksheet.add_cols(6 - worksheet.col_count)

    return worksheet


def _session_log_worksheet():
    dashboard_id = _get_env("DASHBOARD_ID")
    workbook = _build_client().open_by_key(dashboard_id)

    try:
        worksheet = workbook.worksheet(SESSION_LOG_SHEET_NAME)
    except Exception:
        worksheet = workbook.add_worksheet(title=SESSION_LOG_SHEET_NAME,
                                           rows=200,
                                           cols=len(SESSION_LOG_HEADERS))
        worksheet.append_row(SESSION_LOG_HEADERS, value_input_option="USER_ENTERED")

    return worksheet


def _notifications_worksheet():
    dashboard_id = _get_env("DASHBOARD_ID")
    workbook = _build_client().open_by_key(dashboard_id)

    try:
        worksheet = workbook.worksheet(NOTIFICATIONS_SHEET_NAME)
    except Exception:
        worksheet = workbook.add_worksheet(title=NOTIFICATIONS_SHEET_NAME,
                                           rows=200,
                                           cols=len(NOTIFICATIONS_HEADERS))
        worksheet.append_row(NOTIFICATIONS_HEADERS, value_input_option="USER_ENTERED")

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


def log_session(client_id, animal, phase, performance, start_utc=None, timestamp=None):
    performance = str(performance or "").strip()
    if not performance:
        return None

    client_id = _norm_client_id(client_id)
    platform = SESSION_LOG_PLATFORM_NAMES.get(client_id, client_id.title())

    when = timestamp or datetime.now()
    date_str = f"{when.month}/{when.day}/{when.strftime('%y')}"
    time_str = _format_utc(start_utc)

    row = [date_str, platform, str(animal or ""), str(phase or ""), performance, time_str]

    worksheet = _session_log_worksheet()
    worksheet.insert_row(row, index=2, value_input_option="USER_ENTERED")

    return {"ok": True, "client_id": client_id, "row": row}


def log_notification(message, timestamp=None):
    message = str(message or "").strip()
    if not message:
        return None

    when = timestamp or datetime.now()
    hour12 = when.strftime("%I").lstrip("0") or "12"
    display_ts = f"{when.month}/{when.day}/{when.year}, {hour12}:{when.strftime('%M:%S %p')}"

    row = [display_ts, message]

    worksheet = _notifications_worksheet()
    worksheet.insert_row(row, index=2, value_input_option="USER_ENTERED")

    return {"ok": True, "row": row}

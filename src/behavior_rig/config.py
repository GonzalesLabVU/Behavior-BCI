
"""
Static configuration: phase definitions, serial/timeout constants, path
resolution, and lazy Google Sheets/Drive client construction
"""

import os
from pathlib import Path
from queue import Queue

import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.getenv("BEHAVIOR_CONFIG_DIR", str(REPO_ROOT / "config")))
DATA_DIR = REPO_ROOT / "data"

load_dotenv(CONFIG_DIR / ".env")

ANIMAL_MAP_PATH = CONFIG_DIR / "animal_map.json"
ERROR_LOG_PATH = CONFIG_DIR / "errors.log"

PHASE_CONFIG = {
    '2': {'threshold': 15.0, 'side': 'B', 'reverse': False}, # wheel association
    '3': {'threshold': 15.0, 'side': 'B', 'reverse': False}, # tone association
    '4': {'threshold': 15.0, 'side': 'L', 'reverse': False}, # easy wheel
    '5': {'threshold': 30.0, 'side': 'L', 'reverse': False}, # normal wheel
    '6': {'threshold': 60.0, 'side': 'L', 'reverse': False}, # harder wheel
    '7': {'threshold': 90.0, 'side': 'L', 'reverse': False}, # hardest wheel
    }

BAUDRATE = 1_000_000
EARLY_STRING = "E"

EVT_QUEUE: "Queue[tuple[str, str]]" = Queue()
ENC_QUEUE: "Queue[tuple[str, object]]" = Queue()


_api_cache = dict()


def _valid_phases():
    """
    Return the set of valid phase identifiers
    """
    return {"0", "1"} | set(PHASE_CONFIG.keys())


def _credentials_path():
    return CONFIG_DIR / "credentials.json"


def get_sheets_client():
    """
    Return the system's Google Sheets API client object
    """
    if "client" not in _api_cache:
        creds_path = _credentials_path()
        if not creds_path.exists():
            raise RuntimeError(f"Google service-account credentials not found at {creds_path}; see docs/install.md for how to provision one")

        creds = Credentials.from_service_account_file(str(creds_path),
                                                      scopes=[
                                                          "https://www.googleapis.com/auth/spreadsheets",
                                                          "https://www.googleapis.com/auth/drive"
                                                            ]
                                                        )
        _api_cache["client"] = gspread.authorize(creds)

    return _api_cache["client"]


def get_drive_client():
    """
    Return the system's Google Drive API client object
    """
    if "drive" not in _api_cache:
        creds = Credentials.from_service_account_file(str(_credentials_path()),
                                                      scopes=["https://www.googleapis.com/auth/drive"]
                                                      )
        _api_cache["drive"] = build("drive", "v3", credentials=creds, cache_discovery=False)

    return _api_cache["drive"]

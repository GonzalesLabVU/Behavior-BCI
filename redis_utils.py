import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


ENV_PATH = Path(__file__).resolve().parent / ".env"
_ENV_CACHE = None


def _load_local_env():
    global _ENV_CACHE

    if _ENV_CACHE is not None:
        return _ENV_CACHE

    data = {}

    if ENV_PATH.exists():
        for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]

            data[key] = value

    _ENV_CACHE = data
    return data


def _get_env_value(name):
    value = os.getenv(name)
    if value:
        return value

    value = _load_local_env().get(name)
    if value:
        return value

    raise RuntimeError(f"{name} not found in {ENV_PATH.name}")


def _redis_request(command):
    base_url = _get_env_value("UPSTASH_REDIS_REST_URL").rstrip("/")
    token = _get_env_value("UPSTASH_REDIS_REST_TOKEN")

    request = Request(
        f"{base_url}/{command.lstrip('/')}",
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )

    with urlopen(request) as response:
        payload = response.read().decode("utf-8")

    data = json.loads(payload) if payload else {}
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(data["error"])

    return data


def add_entry(key, value):
    key = str(key)
    value = str(value)

    if not key:
        raise ValueError("Redis key cannot be empty")

    return _redis_request(f"set/{quote(key, safe='')}/{quote(value, safe='')}")


def remove_entry(key):
    key = str(key)

    if not key:
        raise ValueError("Redis key cannot be empty")

    return _redis_request(f"del/{quote(key, safe='')}")

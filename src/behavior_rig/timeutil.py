"""
Shared timestamp helpers: current time formatting, millisecond
conversion, and the session date string used across the package
"""

import time
from datetime import datetime


def _get_ts():
    """Return the current local time as an HH:MM:SS.mmm timestamp string."""
    t = time.time()
    base = time.strftime("%H:%M:%S", time.localtime(t))
    ms = int((t - int(t)) * 1000)

    return f"{base}.{ms:03d}"


def _ts_to_ms(ts):
    """Convert an HH:MM:SS.mmm timestamp string to milliseconds since midnight.

    Args:
        ts: Timestamp value to parse.

    Returns:
        Integer milliseconds since midnight, or None if parsing fails.
    """
    try:
        ts = str(ts).strip()
        if not ts:
            return None
        
        hms, ms = ts.split(".", 1)
        h, m, s = hms.split(":")

        return ((3600*int(h) + 60*int(m) + int(s)) * 1000) + int(ms[:3])
    except Exception:
        return None


def _now():
    """Return the current Unix timestamp as integer seconds."""
    return int(time.time())


def _get_date():
    """Return today's date formatted for session metadata."""
    return datetime.now().strftime("%m/%d/%Y")

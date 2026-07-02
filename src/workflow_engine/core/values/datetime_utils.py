from __future__ import annotations

from datetime import datetime, timezone


def parse_iso8601_datetime(value: str) -> datetime:
    """Parse a strict ISO 8601 datetime string into UTC."""
    text = value.strip()
    if not text:
        raise ValueError("Empty datetime string")

    # Reject opaque numeric strings (e.g. Slack message timestamps).
    numeric = text.replace(".", "", 1)
    if numeric.isdigit():
        raise ValueError(f"Expected ISO 8601 datetime, got numeric string: {value!r}")

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError("Naive datetimes are not allowed")
    return dt.astimezone(timezone.utc)


def to_utc_datetime(value: datetime | int | float | str) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("Naive datetimes are not allowed")
        return value.astimezone(timezone.utc)
    if isinstance(value, bool):
        raise TypeError("bool is not a valid datetime")
    if isinstance(value, int):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, float):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        return parse_iso8601_datetime(value)
    raise TypeError(f"Cannot convert {type(value).__name__} to datetime")

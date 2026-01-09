"""
Utility functions for standardized timestamp handling across the application.
All timestamps are in UTC timezone with microseconds set to 0.
"""

from datetime import datetime, timezone


def get_utc_timestamp() -> datetime:
    """
    Get current UTC timestamp with microseconds set to 0.

    Returns:
        datetime: Current UTC datetime with microseconds=0
    """
    now = datetime.now(timezone.utc)
    return now.replace(microsecond=0)


def to_iso_format(dt: datetime) -> str:
    """
    Convert datetime to ISO format string with microseconds=0 and UTC timezone.

    Args:
        dt: datetime object to convert

    Returns:
        str: ISO format string with microseconds=0
    """
    # Ensure timezone is UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo != timezone.utc:
        dt = dt.astimezone(timezone.utc)

    # Set microseconds to 0 and return ISO format
    return dt.replace(microsecond=0).isoformat()


def get_utc_timestamp_iso() -> str:
    """
    Get current UTC timestamp as ISO format string with microseconds=0.

    Returns:
        str: Current UTC datetime in ISO format with microseconds=0
    """
    return to_iso_format(get_utc_timestamp())


def get_utc_timestamp_for_db() -> str:
    """
    Get current UTC timestamp as ISO format string for database storage.
    This is an alias for get_utc_timestamp_iso() for clarity in database operations.

    Returns:
        str: Current UTC datetime in ISO format with microseconds=0
    """
    return get_utc_timestamp_iso()


def parse_iso_datetime(iso_string: str) -> datetime:
    """
    Parse an ISO format datetime string back to a datetime object.

    Args:
        iso_string: ISO format datetime string

    Returns:
        datetime: Parsed datetime object in UTC timezone
    """
    return datetime.fromisoformat(iso_string)


def calculate_duration_minutes(start_iso: str, end_iso: str) -> int:
    """
    Calculate duration in minutes between two ISO format datetime strings.

    Args:
        start_iso: Start time as ISO format string
        end_iso: End time as ISO format string

    Returns:
        int: Duration in minutes
    """
    start_dt = parse_iso_datetime(start_iso)
    end_dt = parse_iso_datetime(end_iso)
    duration_seconds = (end_dt - start_dt).total_seconds()
    return int(duration_seconds // 60)
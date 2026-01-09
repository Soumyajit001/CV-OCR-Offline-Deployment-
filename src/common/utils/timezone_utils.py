"""
Utility functions for timezone conversion in API responses.
Converts UTC timestamps to client timezone for display.
"""

from datetime import datetime, timezone
from typing import Optional, Union
import pytz
from loguru import logger


def convert_utc_to_timezone(utc_dt: Union[datetime, str], target_timezone: str) -> str:
    """
    Convert UTC datetime to target timezone and return as ISO string.

    Args:
        utc_dt: UTC datetime object or ISO string
        target_timezone: Target timezone string (e.g., 'Asia/Kolkata', 'America/New_York')

    Returns:
        str: Datetime in target timezone as ISO string
    """
    try:
        # Parse string to datetime if needed
        if isinstance(utc_dt, str):
            dt = datetime.fromisoformat(utc_dt.replace("Z", "+00:00"))
        else:
            dt = utc_dt

        # Ensure it's UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        elif dt.tzinfo != timezone.utc:
            dt = dt.astimezone(timezone.utc)

        # Convert to target timezone
        target_tz = pytz.timezone(target_timezone)
        local_dt = dt.astimezone(target_tz)

        # Return as ISO string with timezone info
        return local_dt.isoformat()

    except Exception as e:
        logger.warning(f"Failed to convert timezone {target_timezone}: {e}")
        # Fallback to UTC
        if isinstance(utc_dt, str):
            return utc_dt
        return utc_dt.isoformat()


def convert_datetime_fields(
    data: dict, datetime_fields: list, target_timezone: str
) -> dict:
    """
    Convert multiple datetime fields in a dictionary to target timezone.

    Args:
        data: Dictionary containing datetime fields
        datetime_fields: List of field names that contain datetime values
        target_timezone: Target timezone string

    Returns:
        dict: Dictionary with converted datetime fields
    """
    result = data.copy()

    for field in datetime_fields:
        if field in result and result[field] is not None:
            try:
                result[field] = convert_utc_to_timezone(result[field], target_timezone)
            except Exception as e:
                logger.warning(
                    f"Failed to convert field {field} to timezone {target_timezone}: {e}"
                )

    return result


async def get_client_timezone(client_id: str) -> str:
    """
    Get client timezone from organization settings.
    Falls back to UTC if not found or invalid.

    Args:
        client_id: Client identifier

    Returns:
        str: Client timezone string
    """
    try:
        from src.database.client.lib import OrganizationSettingsLib
        from src.database.common.types import PyObjectId

        org_lib = OrganizationSettingsLib()
        org_settings = await org_lib.get_org_settings(str(PyObjectId(client_id)))

        if org_settings and "timezone" in org_settings:
            timezone_str = org_settings["timezone"]
            # Validate timezone
            try:
                pytz.timezone(timezone_str)
                return timezone_str
            except pytz.exceptions.UnknownTimeZoneError:
                logger.warning(
                    f"Invalid timezone {timezone_str} for client {client_id}"
                )

        return "UTC"

    except Exception as e:
        logger.warning(f"Failed to get timezone for client {client_id}: {e}")
        return "UTC"


async def convert_response_datetimes(
    response_data: Union[dict, list], client_id: str
) -> Union[dict, list]:
    """
    Convert datetime fields in API response to client timezone.

    Args:
        response_data: Response data (dict or list of dicts)
        client_id: Client identifier

    Returns:
        Union[dict, list]: Response data with converted datetime fields
    """
    # Get client timezone
    client_timezone = await get_client_timezone(client_id)

    # Common datetime fields across all schemas
    datetime_fields = [
        "created_at",
        "updated_at",
        "deleted_at",
        "detected_at",
        "assigned_at",
        "completed_at",
        "due_date",
        "archived_at",
        "timestamp",
        "last_event_at",
    ]

    if isinstance(response_data, list):
        return [
            convert_datetime_fields(item, datetime_fields, client_timezone)
            for item in response_data
        ]
    else:
        return convert_datetime_fields(response_data, datetime_fields, client_timezone)

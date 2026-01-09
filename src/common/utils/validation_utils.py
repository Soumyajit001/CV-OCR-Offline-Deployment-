"""
Validation utilities for the Configurator service.
"""

from typing import Optional
from bson import ObjectId
from bson.errors import InvalidId
from src.common.exceptions import InvalidObjectIdError


def validate_object_id(value: str, field_name: str = "id") -> ObjectId:
    """
    Validate and convert a string to ObjectId.

    Args:
        value: String value to convert
        field_name: Name of the field for error messages

    Returns:
        ObjectId instance

    Raises:
        InvalidObjectIdError: If the value is not a valid ObjectId
    """
    if not value:
        raise InvalidObjectIdError(field_name, "empty string")

    try:
        return ObjectId(value)
    except (InvalidId, TypeError, ValueError) as e:
        raise InvalidObjectIdError(field_name, value) from e


def validate_optional_object_id(value: Optional[str], field_name: str = "id") -> Optional[ObjectId]:
    """
    Validate and convert an optional string to ObjectId.

    Args:
        value: Optional string value to convert
        field_name: Name of the field for error messages

    Returns:
        ObjectId instance or None

    Raises:
        InvalidObjectIdError: If the value is not a valid ObjectId
    """
    if value is None:
        return None
    return validate_object_id(value, field_name)

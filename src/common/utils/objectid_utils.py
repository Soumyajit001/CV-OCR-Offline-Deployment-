from bson import ObjectId
from bson.errors import InvalidId
from fastapi.exceptions import HTTPException
from typing import Any, Dict, List, Union
from datetime import datetime

def parse_objectid(value: str) -> ObjectId:
    """Validate and convert a string to MongoDB ObjectId, raising HTTP 400 if invalid."""
    try:
        return ObjectId(value)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid id")

def convert_objectids_for_validation(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively convert all ObjectIds to strings before Pydantic validation.
    Works for nested dicts and lists too.
    """
    def convert_value(v):
        if isinstance(v, ObjectId):
            return str(v)
        elif isinstance(v, dict):
            return {k: convert_value(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [convert_value(i) for i in v]
        return v

    return convert_value(data)


def convert_objectids_for_mongo(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively convert all string values that are valid ObjectId hex strings
    back into bson.ObjectId before MongoDB insert or update.
    """
    def convert_value(v):
        if isinstance(v, str) and ObjectId.is_valid(v):
            return ObjectId(v)
        elif isinstance(v, dict):
            return {k: convert_value(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [convert_value(i) for i in v]
        return v

    return convert_value(data)

def normalize_for_response(data: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
    """
    Recursively convert ObjectIds and datetimes to JSON-safe strings for responses.
    Works for nested dicts and lists.
    """
    def convert_value(v):
        if isinstance(v, ObjectId):
            return str(v)
        elif isinstance(v, datetime):
            return v.isoformat()
        elif isinstance(v, dict):
            return {k: convert_value(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [convert_value(i) for i in v]
        return v

    return convert_value(data)
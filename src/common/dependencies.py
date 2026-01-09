from fastapi import Path, HTTPException, Depends, Header
from typing import Optional, Dict
import httpx
from loguru import logger
from src.common.utils.validation_utils import validate_object_id
from src.common.exceptions import InvalidObjectIdError
from src.common.config.settings import get_settings

settings = get_settings()

AUTH_BASE_URL = "http://13.204.81.137:8000"

def get_client_id(
    client_id: str = Path(..., min_length=1, description="Client identifier"),
    # user=Depends(verify_auth_token),
):
    """
    Validate and return client_id from path parameter.

    Raises:
        HTTPException: If client_id is not a valid ObjectId
    """
    try:
        validate_object_id(client_id, "client_id")
        # if user["client_id"] != client_id:
        #     raise HTTPException(status_code=403, detail="Client mismatch")
        return client_id
    except InvalidObjectIdError as e:
        raise HTTPException(status_code=400, detail=str(e))

async def verify_auth_token(
    authorization: Optional[str] = Header(None)
)-> Dict:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization Header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token format")
    # return True
    token = authorization.removeprefix("Bearer ").strip()

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.post(
                f"{AUTH_BASE_URL}/oauth/introspect",
                json={
                    "token": token,
                    "token_type_hint": "access_token",
                },
            )

        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Auth service unavailable")

        data = response.json()

        if not data.get("active"):
            raise HTTPException(status_code=401, detail="Token inactive or expired")
        
        if not data.get("email"):
             logger.warning("Auth failed: Token missing email claim")
             raise HTTPException(status_code=401, detail="Invalid token claims")

        return data

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail="Authentication check failed")

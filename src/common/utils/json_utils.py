# src/common/utils/json_utils.py
from fastapi.encoders import jsonable_encoder
from bson import ObjectId

def bson_to_json(data):
    return jsonable_encoder(data, custom_encoder={ObjectId: str})

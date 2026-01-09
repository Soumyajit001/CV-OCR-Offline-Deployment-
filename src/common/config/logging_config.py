"""
Logging configuration with JSON format support.
"""

import sys
import json
from loguru import logger
from datetime import datetime
from src.common.config.settings import get_settings

settings = get_settings()


def serialize_log(record: dict) -> str:
    """
    Serialize log record to JSON format.
    """
    log_entry = {
        "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }

    if record["exception"]:
        log_entry["exception"] = {
            "type": record["exception"].type.__name__,
            "value": str(record["exception"].value),
        }

    if record.get("extra"):
        log_entry["extra"] = record["extra"]

    return json.dumps(log_entry)


def configure_logging():
    """
    Configure loguru logger with colorized text in debug mode
    and JSON logging in production.
    """
    logger.remove()

    if settings.debug:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            level=settings.log_level,
            colorize=True,
        )
    else:
        # ✅ Use a custom sink for JSON output
        logger.add(
            lambda msg: sys.stderr.write(serialize_log(msg.record) + "\n"),
            level=settings.log_level,
        )

    logger.info(
        f"Logging configured: level={settings.log_level}, debug={settings.debug}, json_format={not settings.debug}"
    )

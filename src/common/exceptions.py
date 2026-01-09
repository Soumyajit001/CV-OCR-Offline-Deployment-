"""
Custom exceptions for the Configurator service.
"""

from typing import Optional


class ConfiguratorException(Exception):
    """Base exception for all configurator errors."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ValidationError(ConfiguratorException):
    """Raised when data validation fails."""

    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class NotFoundError(ConfiguratorException):
    """Raised when a resource is not found."""

    def __init__(self, resource: str, resource_id: str):
        message = f"{resource} with id={resource_id} not found"
        super().__init__(message, status_code=404)


class DatabaseError(ConfiguratorException):
    """Raised when database operations fail."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(f"Database error: {message}", status_code=500)
        self.original_error = original_error


class InvalidObjectIdError(ValidationError):
    """Raised when an invalid ObjectId is provided."""

    def __init__(self, field_name: str, value: str):
        super().__init__(f"Invalid ObjectId for {field_name}: {value}")


class ClientNotFoundError(NotFoundError):
    """Raised when a client is not found."""

    def __init__(self, client_id: str):
        super().__init__("Client", client_id)

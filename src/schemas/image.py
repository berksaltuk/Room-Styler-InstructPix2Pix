from pydantic import BaseModel
from typing import Optional


class StyleTransferResponse(BaseModel):
    """
    Model to represent the response for a style transfer operation.

    Attributes:
        success (bool): Indicates whether the style transfer was successful.
        message (Optional[str]): An optional message providing additional details or errors.
    """
    success: bool
    message: Optional[str] = None

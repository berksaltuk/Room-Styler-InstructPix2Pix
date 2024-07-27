from pydantic import BaseModel
from typing import Optional


class StyleTransferResponse(BaseModel):
    success: bool
    message: Optional[str] = None

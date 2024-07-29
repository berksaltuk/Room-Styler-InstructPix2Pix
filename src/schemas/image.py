from pydantic import BaseModel, Field
from fastapi import Response
from io import BytesIO

class StyleTransferRequest(BaseModel):
    """
    Model to represent the request for room styler endpoint.

    Attributes:
        prompt (str): the style prompt that will be used during inference.
        number_of_variants (int): number of variants that .
    """
    prompt: str = Field(..., description="Style prompt of your room", example="turn it into a Japanese style living room")
    number_of_variants: int = Field(..., description="Number of variants you want to create")

class ZipFileResponse(Response):
    """
    Custom response for serving zip files.

    Args:
        content (BytesIO): The zip file data.
        filename (str): The name of the zip file.
        status_code (int, optional): HTTP status code. Defaults to 200.
        **kwargs: Additional arguments for the base Response class.

    Attributes:
        media_type (str): Always set to "application/zip".
    """
    # configure the zip file response
    media_type = "application/zip"

    def __init__(self, content: BytesIO, filename: str, status_code: int = 200, **kwargs):
        headers = {
            "Content-Disposition": f"attachment; filename={filename}"
        }
        super().__init__(content=content.getvalue(), headers=headers, media_type=self.media_type, status_code=status_code, **kwargs)

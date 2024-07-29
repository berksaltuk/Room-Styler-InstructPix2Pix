from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import ValidationError
from typing import Annotated

from src.controllers.image import ImageController
from src.schemas.image import StyleTransferRequest, ZipFileResponse

router = APIRouter()

@router.post("/images/room-styler",
             description="Takes prompt and room image to apply style to given image",
             response_class=ZipFileResponse)
async def transfer_style(
        prompt: Annotated[str, Form(..., description="Style prompt of your room", example="turn it into a Japanese style living room")],
        n: Annotated[int, Form(..., description="Number of variants you want to create")],
        source_image: UploadFile = File(..., description="Content source image of your room")):
    """
    Endpoint to transfer style to a given room image based on a prompt.

    Args:
        prompt (str): Style prompt describing how the room should look.
        n (int): Number of style variations to generate.
        source_image (UploadFile): The uploaded image file of the room to which the style will be applied.

    Returns:
        Response: The response from the ImageController after applying the style transfer.
    """
    try:
        # Validate the form data using the Pydantic model
        _ = StyleTransferRequest(prompt=prompt, number_of_variants=n)
    except ValidationError as e:
        return HTTPException(
            status_code=422,
            detail=e.errors()
          )
    
    # get the zip file response
    zip_buffer = await ImageController.transfer_style(prompt, n, image=source_image)
    response = ZipFileResponse(content=zip_buffer, filename="generated_images.zip")

    # close the zip buffer
    zip_buffer.close()

    return response

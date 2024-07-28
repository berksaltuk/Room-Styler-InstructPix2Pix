from fastapi import APIRouter, UploadFile, File, Form
from src.controllers.image import ImageController
from typing import Annotated

router = APIRouter()


@router.post("/images/room-styler",
             description="Takes prompt and room image to apply style to given image")
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
    response = await ImageController.transfer_style(prompt, n, image=source_image)
    return response

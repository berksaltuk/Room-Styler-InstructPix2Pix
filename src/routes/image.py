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
    response = await ImageController.transfer_style(prompt, n, image=source_image)
    return response

from fastapi import APIRouter, UploadFile, File, Form
from src.controllers.image import ImageController
from typing import Annotated

router = APIRouter()


@router.post("/images/transfer",
             description="Takes prompt and image to apply style prompt to given image")
async def transfer_style(
        prompt: Annotated[str, Form(...)],
        n: Annotated[int, Form(...)],
        source_image: UploadFile = File(...)):
    response = await ImageController.transfer_style(prompt, n, image=source_image)
    return response

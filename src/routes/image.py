from fastapi import APIRouter, UploadFile, File, Form
from src.schemas.image import StyleTransferResponse
from src.controllers.image import ImageController
from typing import Annotated

router = APIRouter()

@router.post("/images/transfer", response_model=StyleTransferResponse)
async def transfer_style(prompt: Annotated[str, Form(...)], image: UploadFile = File(...)):
    response = await ImageController.transfer_style(prompt, image)
    return response
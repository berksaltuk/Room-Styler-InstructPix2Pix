from fastapi import UploadFile
from src.schemas.image import StyleTransferResponse
from src.services.image import ImageService


class ImageController:
    @staticmethod
    async def transfer_style(prompt: str,
                             image: UploadFile) -> StyleTransferResponse:
        response = await ImageService.transfer_style(prompt, image)
        return StyleTransferResponse(success=True, message=response)

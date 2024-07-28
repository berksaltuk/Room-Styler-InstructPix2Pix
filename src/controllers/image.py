from fastapi import UploadFile, Response
from src.services.image import ImageService


class ImageController:
    @staticmethod
    async def transfer_style(prompt: str,
                             n: int,
                             image: UploadFile) -> Response:
        image_bytes = await image.read()
        response = await ImageService.transfer_style(prompt, n, image=image_bytes)
        return response

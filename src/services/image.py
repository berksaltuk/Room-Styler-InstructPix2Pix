from fastapi import UploadFile


class ImageService:
    @staticmethod
    async def transfer_style(prompt: str, image: UploadFile) -> str:
        return image.filename

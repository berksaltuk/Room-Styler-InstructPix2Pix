from fastapi import UploadFile, Response
from src.services.image import ImageService


class ImageController:
    @staticmethod
    async def transfer_style(prompt: str,
                             n: int,
                             image: UploadFile) -> Response:
        """
        Static method to handle the transfer style request.

        Args:
            prompt (str): The text prompt guiding the style transfer.
            n (int): The number of style variations to generate.
            image (UploadFile): The uploaded image file to which the style will be applied.

        Returns:
            Response: The response from the ImageService after applying the style transfer.
        """

        # Read the image bytes and call the image service
        image_bytes = await image.read()
        response = await ImageService.transfer_style(prompt, n, image=image_bytes)
        return response

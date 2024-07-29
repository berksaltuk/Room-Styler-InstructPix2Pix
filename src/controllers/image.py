from fastapi import UploadFile, HTTPException
from io import BytesIO

from src.services.image import ImageService
from src.utils.image import is_image

class ImageController:
    @staticmethod
    async def transfer_style(prompt: str,
                             n: int,
                             image: UploadFile) -> BytesIO:
        """
        Static method to handle the transfer style request.

        Args:
            prompt (str): The text prompt guiding the style transfer.
            n (int): The number of style variations to generate.
            image (UploadFile): The uploaded image file to which the style will be applied.

        Returns:
            Response: The response from the ImageService after applying the style transfer.
        """
        # to make this endpoint efficient, limit n to be 5 at most
        if not (1 <= n <= 5):
          raise HTTPException(
              status_code=422,
              detail="The number of variants must be between 1 and 5"
          )
        
        image_bytes = await image.read()
        image_buffer = BytesIO(image_bytes)

        # check if input is an image
        if not is_image(image_buffer):
          raise HTTPException(
              status_code=422,
              detail="The uploaded file is not a valid image."
          )
    
        zip_buffer = await ImageService.transfer_style(prompt, n, image=image_bytes)
        return zip_buffer

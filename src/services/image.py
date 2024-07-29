from fastapi import HTTPException
from PIL import Image, ImageOps
from io import BytesIO
import torch
import zipfile
import gc

from src.models.diffusers import diffusers_instance


class ImageService:
    @staticmethod
    async def transfer_style(prompt: str, n: int, image: bytes) -> BytesIO:
        """
        Applies style transfer to the given image based on the prompt and generates a zip file with the resulting images.

        Args:
            prompt (str): The style prompt for the style transfer.
            n (int): The number of style variants to generate.
            image (bytes): The image file in bytes to which the style will be applied.

        Returns:
            BytesIO: A zip_buffer containing images
        """
        zip_buffer = BytesIO()
        try:
           
            source_image = Image.open(BytesIO(image)).convert("RGB")
            source_image = ImageOps.exif_transpose(source_image)
           
            source_image.thumbnail((512, 512))

            pipe = diffusers_instance.get_pipeline()

            with torch.no_grad():
                generated_images = pipe(
                    prompt=prompt,
                    image=source_image,
                    guidance_scale=7.66,
                    image_guidance_scale=1.87,
                    num_images_per_prompt=n,
                    num_inference_steps=50
                ).images

            gc.collect()
            torch.cuda.empty_cache()

            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                img_byte_arr = BytesIO()
                source_image.save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)
                zip_file.writestr(
                    "source_image.jpg",
                    img_byte_arr.getvalue()
                )
                for idx, img in enumerate(generated_images):
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format='JPEG')
                    img_byte_arr.seek(0)
                    zip_file.writestr(
                        f"image_{idx+1}.jpg",
                        img_byte_arr.getvalue()
                    )

            zip_buffer.seek(0)
            return zip_buffer
        except (IOError, OSError) as e:
            raise HTTPException(
                detail=f'Error processing the images: {e}',
                status_code=400
            )
        except torch.error as e:
            raise HTTPException(
                detail=f'Torch operation error: {e}',
                status_code=500
            )
        except zipfile.BadZipFile as e:
            # Handle errors related to zip file creation
            raise HTTPException(
                detail=f'Error creating zip file: {e}',
                status_code=500
            )
        except Exception as e:
            # Catch all other exceptions
            raise HTTPException(
                detail=f'An unexpected error occurred: {e}',
                status_code=500
            )

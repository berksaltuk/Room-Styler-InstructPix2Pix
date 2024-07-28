from fastapi import HTTPException, Response
from PIL import Image, ImageOps
from io import BytesIO
import torch
import zipfile

from src.models.diffusers import diffusers_instance


class ImageService:
    @staticmethod
    async def transfer_style(prompt: str,
                             n: int,
                             image: bytes) -> Response:
        zip_buffer = BytesIO()
        try:
            source_image = Image.open(BytesIO(image)).convert("RGB")
            source_image = ImageOps.exif_transpose(source_image)
            source_image = source_image.convert("RGB")
            source_image.thumbnail((512, 512))
            pipe = diffusers_instance.get_pipeline()
            with torch.no_grad():
              generated_images = pipe(prompt=prompt,
                                      image=source_image,
                                      guidance_scale=7.66,
                                      image_guidance_scale=1.87,
                                      num_images_per_prompt=n,
                                      num_inference_steps=30).images
            
            # Create a zip file by iterating through generated images
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for idx, img in enumerate(generated_images):
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format='JPEG')
                    img_byte_arr.seek(0)
                    zip_file.writestr(
                        f"image_{idx+1}.jpg",
                        img_byte_arr.getvalue())

            zip_buffer.seek(0)
            headers = {
                "Content-Disposition": "attachment; filename=generated_images.zip"}
            return Response(
                zip_buffer.getvalue(),
                headers=headers,
                media_type="application/zip")
        except BaseException:
            raise HTTPException(
                detail='There was an error creating the zip file',
                status_code=400)
        finally:
            zip_buffer.close()

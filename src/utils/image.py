from io import BytesIO
from PIL import Image
import magic

def is_image(file: BytesIO) -> bool:
    """
    Checks if the provided file-like object is a valid image.

    Args:
        file (BytesIO): A file-like object containing the image data.

    Returns:
        bool: True if the file is a valid image of an acceptable type (JPEG, PNG, GIF, BMP, TIFF), False otherwise.

    Raises:
        IOError: If the file cannot be opened or is not a valid image.
        SyntaxError: If the file is not a valid image format or contains corrupted data.

    Notes:
        - The function first uses the `magic` library to check the MIME type of the file based on its initial bytes.
        - It then attempts to open the file as an image using PIL and verifies it.
        - The MIME types considered valid are: 'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff'.
        - If the MIME type is not among the valid types or if the image cannot be verified, the function returns False.
        - This function is useful for validating image uploads or checking image file integrity before processing.
    """
    try:
        mime = magic.Magic(mime=True)
        file_content = file.read(2048)
        mime_type = mime.from_buffer(file_content)
        file.seek(0)

        if mime_type not in ["image/jpeg", "image/png", "image/gif", "image/bmp", "image/tiff"]:
            return False
        
        with Image.open(file) as img:
            img.verify()
        return True
    except (IOError, SyntaxError) as e:
        print(f"Error: {e}")
        return False

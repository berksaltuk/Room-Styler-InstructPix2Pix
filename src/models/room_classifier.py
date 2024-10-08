from transformers import pipeline
from io import BytesIO
from PIL import Image
import torch

class RoomImageClassifier:
    def __init__(self, model_name="berksaltuk/room-classifier"):
        """
        Initialize the ImageClassifier class.
        This sets up the image classification pipeline with the specified model.

        Args:
            model_name (str): The name of the model to use for classification.
        """

        # set the device, if cuda available the inference is way faster
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # I trained this classifier model for room type classification
        # This model has classes bedroom, living room, kitchen and bathroom
        self.pipe = pipeline(
            "image-classification",
            model=model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device=self.device
            )
    def get_pipeline(self):
        """
        Get the configured StableDiffusionInstructPix2PixPipeline.

        Returns:
            pipe (StableDiffusionInstructPix2PixPipeline): The configured pipeline instance.
        """
        return self.pipe
    
    def classify_image(self, image_file):
        """
        Classify an image using the configured pipeline.

        Args:
            image_path (str): The path to the image file to be classified.

        Returns:
            results (list): A list of classification results with labels and scores.
        """
        image = Image.open(BytesIO(image_file))
        results = self.pipe(image)
        return results

room_classifier_instance = RoomImageClassifier()
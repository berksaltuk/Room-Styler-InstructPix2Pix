from transformers import pipeline
from io import BytesIO
from PIL import Image

class RoomImageClassifier:
    def __init__(self, model_name="berksaltuk/room-classifier"):
        """
        Initialize the ImageClassifier class.
        This sets up the image classification pipeline with the specified model.

        Args:
            model_name (str): The name of the model to use for classification.
        """
        self.pipe = pipeline("image-classification", model=model_name)

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
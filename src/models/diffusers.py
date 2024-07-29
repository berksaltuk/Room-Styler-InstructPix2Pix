import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


from dotenv import load_dotenv

load_dotenv()


class DiffusersModel:
    def __init__(self):
        """
        Initialize the DiffusersModel class.
        This sets up the StableDiffusionInstructPix2PixPipeline with appropriate settings.
        """

        # set the device, if cuda available the inference is way faster
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # use InstructPix2Pix pre-trained model for inference
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None).to(
            self.device)

        # set the euler ancestral scheduler for stability and robustness
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        # enable attention slicing to reduce memory usage during inference
        self.pipe.enable_attention_slicing()

    def get_pipeline(self):
        """
        Get the configured StableDiffusionInstructPix2PixPipeline.

        Returns:
            pipe (StableDiffusionInstructPix2PixPipeline): The configured pipeline instance.
        """
        return self.pipe


diffusers_instance = DiffusersModel()

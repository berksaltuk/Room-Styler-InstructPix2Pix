import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


from dotenv import load_dotenv

load_dotenv()


class DiffusersModel:
    def __init__(self):
        # token = os.getenv("HUGGING_FACE_TOKEN") # May use this later

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None).to(
            self.device)

        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_attention_slicing()

    def get_pipeline(self):
        return self.pipe


diffusers_instance = DiffusersModel()

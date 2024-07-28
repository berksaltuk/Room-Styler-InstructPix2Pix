import torch
import os
from diffusers import StableDiffusionImg2ImgPipeline, LMSDiscreteScheduler


from dotenv import load_dotenv

load_dotenv()


class DiffusersModel:
    def __init__(self):
        # token = os.getenv("HUGGING_FACE_TOKEN") # May use this later

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = torch.Generator(device=self.device).manual_seed(1024)
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "nitrosocke/Ghibli-Diffusion",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(
            self.device)

        self.lms = LMSDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler = self.lms
        self.pipe.enable_attention_slicing()

    def get_pipeline(self):
        return self.pipe

    def get_generator(self):
        return self.generator


diffusers_instance = DiffusersModel()

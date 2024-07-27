import torch
import os
from diffusers import StableDiffusionImg2ImgPipeline
from dotenv import load_dotenv

load_dotenv()

class DiffusersModel:
    def __init__(self):
        token = os.getenv("HUGGING_FACE_TOKEN")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "nitrosocke/Ghibli-Diffusion",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_auth_token=token
        ).to(device)
    
    def get_pipeline(self):
        return self.pipe

diffusers_instance = DiffusersModel()

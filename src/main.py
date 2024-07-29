from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.routes.image import router as image_router
from src.models.diffusers import diffusers_instance
from src.models.room_classifier import room_classifier_instance
@asynccontextmanager
async def lifespan(app: FastAPI):
    # init pipeline
    _ = diffusers_instance.get_pipeline()
    _ = room_classifier_instance.get_pipeline()
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(image_router, prefix="/api")

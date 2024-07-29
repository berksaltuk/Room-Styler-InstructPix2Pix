from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.routes.image import router as image_router
from src.models.diffusers import diffusers_instance
from src.models.room_classifier import room_classifier_instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for FastAPI app lifespan.

    Initializes pipelines for diffusers and room classifier
    at the start of the app and ensures proper cleanup at shutdown.
    """
    # init pipelines to load the components at the application startup 
    _ = diffusers_instance.get_pipeline()
    _ = room_classifier_instance.get_pipeline()
    yield

# initialize the FastAPI app
app = FastAPI(lifespan=lifespan)

# include the image router
app.include_router(image_router, prefix="/api")

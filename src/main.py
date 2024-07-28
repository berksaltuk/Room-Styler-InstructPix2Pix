from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.routes.image import router as image_router
from src.models.diffusers import diffusers_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    # init pipeline
    _ = diffusers_instance.get_pipeline()
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(image_router, prefix="/api")

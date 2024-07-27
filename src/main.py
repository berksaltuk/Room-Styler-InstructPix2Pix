from fastapi import FastAPI
from src.routes.image import router as image_router

app = FastAPI()

app.include_router(image_router, prefix="/api")



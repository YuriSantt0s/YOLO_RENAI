from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from Backend.app.controllers.predict_controller import router

app = FastAPI(title="YOLO Medical Detector")

BASE_DIR = Path(__file__).resolve().parent.parent

# Static files
app.mount(
    "/static",
    StaticFiles(directory="Frontend"),
    name="static"
)

# Rotas
app.include_router(router)

@app.get("/")
async def root():
    return FileResponse("Frontend/pages/index.html")
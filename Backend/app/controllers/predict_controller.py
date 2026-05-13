from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from Backend.app.services.predict_service import predict_image

router = APIRouter()

@router.get("/health")
def health():
    return {
        "status": "ok"
    }

@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = 0.15,
    iou: float = 0.50,
):

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Arquivo deve ser imagem")

    contents = await file.read()

    result = predict_image(contents, conf, iou)

    return JSONResponse(result)
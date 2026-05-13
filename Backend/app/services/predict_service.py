import base64
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"

model = YOLO(str(MODEL_PATH))
model.to("cpu")

CLASS_NAMES = {
    0: "rim",
    1: "tumor",
    2: "cisto",
}

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (255, 255, 0),
}

def predict_image(contents, conf, iou):

    nparr = np.frombuffer(contents, np.uint8)

    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.predict(
        source=img_np,
        conf=conf,
        iou=iou,
        device="cpu"
    )[0]

    img_out = img_np.copy()

    deteccoes = []

    if results.boxes is not None:

        for box in results.boxes:

            cls_id = int(box.cls.item())

            conf_v = float(box.conf.item())

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            nome = CLASS_NAMES.get(cls_id)

            cor = CLASS_COLORS.get(cls_id)

            cor_bgr = (cor[2], cor[1], cor[0])

            cv2.rectangle(
                img_out,
                (x1, y1),
                (x2, y2),
                cor_bgr,
                2
            )

            deteccoes.append({
                "classe": nome,
                "confianca": round(conf_v, 4),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            })

    img_rgb = cv2.cvtColor(
        img_out,
        cv2.COLOR_BGR2RGB
    )

    _, buffer = cv2.imencode(".png", img_rgb)

    img_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "total_deteccoes": len(deteccoes),
        "deteccoes": deteccoes,
        "imagem_anotada": f"data:image/png;base64,{img_b64}",
    }
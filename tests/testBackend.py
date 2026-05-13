# ── Cole isso no terminal do VS Code para testar o modelo direto ──────────────
# Abre um novo terminal (sem o uvicorn rodando) e roda:
# python debug.py

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

MODEL_PATH = "C:/Faculdade/TCC/yolo_detector/models/best.pt"
IMG_PATH   = "C:/Users/yurin/Downloads/drive-download-20260511T234816Z-3-001/case_00588_slice_125.png"   # coloque o nome de uma imagem de teste aqui

# ── 1. Verifica se o modelo carrega ──────────────────────────────────────────
print("Carregando modelo...")
model = YOLO(MODEL_PATH)
print(f"✓ Modelo carregado")
print(f"  Classes: {model.names}")

# ── 2. Roda com threshold mínimo para ver o que o modelo detecta ─────────────
print(f"\nRodando predição com conf=0.01 (mínimo possível)...")
img     = cv2.imread(IMG_PATH)
results = model.predict(
    source  = img,
    conf    = 0.01,   # threshold mínimo — mostra TUDO que o modelo detecta
    iou     = 0.3,
    device  = "cpu",
    verbose = True,
)[0]

# ── 3. Mostra o que foi detectado ─────────────────────────────────────────────
print(f"\n── Resultado ──────────────────────────────────────")
if results.boxes is None or len(results.boxes) == 0:
    print("  NENHUMA detecção — mesmo com conf=0.01")
    print("  Causas prováveis:")
    print("  1. Imagem em formato errado (16-bit, DICOM, etc)")
    print("  2. Modelo treinado com classes diferentes")
    print("  3. Imagem muito diferente do dataset de treino")
else:
    print(f"  {len(results.boxes)} detecções encontradas:")
    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf_v = float(box.conf.item())
        nome   = model.names[cls_id]
        print(f"    {nome:<10} confiança: {conf_v:.4f} ({conf_v:.1%})")

# ── 4. Verifica a imagem ──────────────────────────────────────────────────────
print(f"\n── Info da imagem ─────────────────────────────────")
print(f"  Shape : {img.shape}")
print(f"  dtype : {img.dtype}")
print(f"  min   : {img.min()}")
print(f"  max   : {img.max()}")
print(f"  mean  : {img.mean():.1f}")

if img.dtype != np.uint8:
    print(f"\n  ⚠ PROBLEMA: imagem é {img.dtype} — modelo espera uint8 (8-bit)")
    print(f"  Converta com: img_8bit = (img / 256).astype(np.uint8)")

if img.mean() < 5:
    print(f"\n  ⚠ PROBLEMA: imagem muito escura (mean={img.mean():.1f})")
    print(f"  Pode ser janelamento HU errado")

if img.mean() > 250:
    print(f"\n  ⚠ PROBLEMA: imagem saturada (mean={img.mean():.1f})")
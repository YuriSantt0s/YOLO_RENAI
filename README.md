# 🩺 RENAI - YOLO Medical Detector

> O presente projeto é um Sistema inteligente de detecção renal utilizando YOLO em tumográfias computadorizadas.

Detecção automática de:

- 🟢 Rim
- 🔴 Tumor renal
- 🟠 Cisto renal

---

## 📸 Preview

![Preview](https://dummyimage.com/1200x650/e8f1fb/1a2332&text=YOLO+Medical+Detector)

---

# 📌 Sobre o projeto

O **RENAI** é uma aplicação web para análise de imagens médicas utilizando modelos YOLO treinados para identificação de estruturas renais.O modelo foi treinado utilizando a base de dados publica XXXX, disponivel
na plataforma kagle e desenvolvido no google colab (para vizualizar mais sobre a pipeline de tratamento de dados e treinamento do modelo acesse gitXXXX). O projeto organiza-se sendo uma API rest, 
unindo front e backend em uma estrutura monolitica 
monolitica 

A plataforma permite:

- ✅ Upload de imagens médicas
- ✅ Detecção automática em tempo real
- ✅ Visualização de bounding boxes
- ✅ Resumo estatístico das classes detectadas
- ✅ Interface moderna e responsiva
- ✅ Ajuste de confiança e IoU

---
# ⚙️ Tecnologias utilizadas

## Frontend
- HTML5
- CSS3
- JavaScript Vanilla

## Backend
- Python
- FastAPI
- Uvicorn

## Inteligência Artificial
- YOLO
- OpenCV
- NumPy
- Pillow

---

# 🚀 Instalação
## 1. Clone o repositório

```bash
git clone https://github.com/YuriSantt0s/yolo-medical-detector.git
```

---
## 2. Acesse o diretório
```bash
cd yolo-medical-detector
```

---
## 3. Crie um ambiente virtual

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / MacOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---
## 4. Instale as dependências

```bash
pip install -r requirements.txt
```

---

# ▶️ Executando o projeto

## Inicie o servidor

```bash
uvicorn app:app --reload
```

---

# 🌐 Acesse

```txt
http://127.0.0.1:8000
```

---

# 📦 Formatos suportados

- PNG
- JPG
- JPEG
- TIFF
- DICOM (.dcm)

# 🧬 Modelo YOLO

O modelo utilizado pode ser substituído em:

```bash
model/best.pt
```

---

# 🔒 Aviso importante

> Este projeto possui finalidade acadêmica e experimental.
>
> Não substitui diagnóstico médico profissional.

---

# 👨‍💻 Autor

## Yuri Nascimento dos Santos

GitHub:
👉 https://github.com/YuriSantt0s

---

# ⭐ Apoie o projeto

Se este projeto foi útil, considere deixar uma estrela no repositório.

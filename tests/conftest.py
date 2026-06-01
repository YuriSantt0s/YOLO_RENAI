# tests/conftest.py
# ═══════════════════════════════════════════════════════════════
# Fixtures compartilhadas — carregadas automaticamente pelo pytest
# ═══════════════════════════════════════════════════════════════

import io
import pytest
import numpy as np

from PIL import Image
from fastapi.testclient import TestClient

from Backend.app.main import app


# ── Cliente de teste ──────────────────────────────────────────────────────────
# TestClient simula requisições HTTP sem subir um servidor real
# Muito mais rápido que chamar localhost

@pytest.fixture(scope="session")
def client():
    """
    Cliente HTTP reutilizado em toda a sessão de testes.
    scope="session" → criado uma vez, não a cada teste.
    """
    with TestClient(app) as c:
        yield c


# ── Imagens de teste geradas em memória ──────────────────────────────────────
# Evita depender de arquivos no disco — testes são portáteis

def _make_png(width=512, height=512, mode="L", value=128) -> bytes:
    """Gera um PNG em memória com valor de pixel uniforme."""
    arr = np.full((height, width), value, dtype=np.uint8)
    if mode == "RGB":
        arr = np.stack([arr, arr, arr], axis=-1)
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_png_16bit(width=512, height=512) -> bytes:
    """Gera PNG 16-bit (uint16) — formato KiTS23."""
    arr = np.full((height, width), 32768, dtype=np.uint16)
    img = Image.fromarray(arr, mode="I;16")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpg(width=512, height=512) -> bytes:
    """Gera um JPEG em memória."""
    arr = np.full((height, width, 3), 100, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ── Fixtures de imagem ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def png_gray():
    """PNG grayscale 8-bit 512×512."""
    return _make_png(mode="L", value=80)


@pytest.fixture(scope="session")
def png_rgb():
    """PNG RGB 8-bit 512×512."""
    return _make_png(mode="RGB", value=80)


@pytest.fixture(scope="session")
def png_16bit():
    """PNG 16-bit uint16 — simula imagem de TC."""
    return _make_png_16bit()


@pytest.fixture(scope="session")
def jpg_image():
    """JPEG RGB."""
    return _make_jpg()


@pytest.fixture(scope="session")
def tiny_png():
    """PNG muito pequeno 32×32 — testa robustez com imagens fora do padrão."""
    return _make_png(width=32, height=32, mode="L", value=50)


@pytest.fixture(scope="session")
def corrupt_bytes():
    """Bytes aleatórios — simula arquivo corrompido."""
    rng = np.random.default_rng(42)
    return bytes(rng.integers(0, 256, 512, dtype=np.uint8))


@pytest.fixture(scope="session")
def empty_bytes():
    """Arquivo completamente vazio."""
    return b""


@pytest.fixture(scope="session")
def text_file():
    """Arquivo de texto — formato inválido enviado como imagem."""
    return b"isso nao e uma imagem, e um arquivo de texto qualquer"

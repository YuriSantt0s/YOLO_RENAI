# tests/unit/test_image_utils.py
# ═══════════════════════════════════════════════════════════════
# Testes UNITÁRIOS — Backend/app/utils/image_utils.py
# Testa cada função de conversão isoladamente, sem HTTP,
# sem modelo YOLO, sem banco de dados.
# ═══════════════════════════════════════════════════════════════

import io
import pytest
import numpy as np
from PIL import Image

from Backend.app.utils.image_utils import (
    detectar_formato,
    ler_png_jpg,
    ler_tiff,
    carregar_imagem,
    imagem_para_base64,
    desenhar_deteccoes,
)


# ── Helpers para gerar bytes de imagem ────────────────────────────────────────

def _png_gray_8bit(w=512, h=512, valor=80) -> bytes:
    arr = np.full((h, w), valor, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "L").save(buf, "PNG")
    return buf.getvalue()


def _png_gray_16bit(w=512, h=512, valor=32768) -> bytes:
    arr = np.full((h, w), valor, dtype=np.uint16)
    buf = io.BytesIO()
    Image.fromarray(arr, "I;16").save(buf, "PNG")
    return buf.getvalue()


def _png_rgb(w=512, h=512) -> bytes:
    arr = np.full((h, w, 3), 100, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, "PNG")
    return buf.getvalue()


def _png_rgba(w=64, h=64) -> bytes:
    arr = np.full((h, w, 4), 128, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def _jpg(w=512, h=512) -> bytes:
    arr = np.full((h, w, 3), 100, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _tiff_8bit(w=256, h=256) -> bytes:
    arr = np.full((h, w), 60, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "L").save(buf, "TIFF")
    return buf.getvalue()


# ════════════════════════════════════════════════════
# DETECTAR FORMATO
# ════════════════════════════════════════════════════

@pytest.mark.unit
class TestDetectarFormato:
    """Testa a detecção de formato pelos magic bytes."""

    def test_detecta_png(self):
        assert detectar_formato(_png_gray_8bit()) == "png"

    def test_detecta_jpg(self):
        assert detectar_formato(_jpg()) == "jpg"

    def test_detecta_tiff(self):
        assert detectar_formato(_tiff_8bit()) == "tiff"

    def test_detecta_bytes_vazios(self):
        """Bytes vazios devem retornar 'desconhecido', não lançar exceção."""
        resultado = detectar_formato(b"")
        assert resultado == "desconhecido"

    def test_detecta_bytes_aleatorios(self):
        """Bytes aleatórios sem magic bytes válidos → desconhecido."""
        resultado = detectar_formato(b"\x00\x01\x02\x03" * 100)
        assert resultado == "desconhecido"

    def test_detecta_texto_como_desconhecido(self):
        resultado = detectar_formato(b"isso nao e uma imagem")
        assert resultado == "desconhecido"


# ════════════════════════════════════════════════════
# LER PNG / JPG
# ════════════════════════════════════════════════════

@pytest.mark.unit
class TestLerPngJpg:
    """Testa a leitura e conversão de PNG e JPEG."""

    # ── Tipo de saída ─────────────────────────────────────────────────────────

    def test_retorna_ndarray(self):
        img = ler_png_jpg(_png_gray_8bit())
        assert isinstance(img, np.ndarray)

    def test_saida_sempre_uint8(self):
        """Saída deve ser sempre uint8 independente da entrada."""
        img = ler_png_jpg(_png_gray_8bit())
        assert img.dtype == np.uint8

    def test_saida_sempre_3_canais(self):
        """Saída deve ser sempre BGR (3 canais) para o YOLO."""
        img = ler_png_jpg(_png_gray_8bit())
        assert len(img.shape) == 3
        assert img.shape[2] == 3

    # ── PNG 8-bit grayscale ───────────────────────────────────────────────────

    def test_png_gray_8bit_dimensoes(self):
        img = ler_png_jpg(_png_gray_8bit(w=512, h=512))
        assert img.shape == (512, 512, 3)

    def test_png_gray_8bit_range_valido(self):
        img = ler_png_jpg(_png_gray_8bit(valor=80))
        assert img.min() >= 0
        assert img.max() <= 255

    # ── PNG 16-bit → 8-bit ───────────────────────────────────────────────────

    def test_png_16bit_convertido_para_8bit(self):
        """PNG 16-bit deve ser convertido para uint8."""
        img = ler_png_jpg(_png_gray_16bit())
        assert img.dtype == np.uint8

    def test_png_16bit_range_valido(self):
        img = ler_png_jpg(_png_gray_16bit(valor=32768))
        assert img.min() >= 0
        assert img.max() <= 255

    def test_png_16bit_maximo_nao_estoura(self):
        """Valor máximo uint16 (65535) deve resultar em 255."""
        img = ler_png_jpg(_png_gray_16bit(valor=65535))
        assert img.max() == 255

    def test_png_16bit_minimo_zero(self):
        """Valor mínimo (0) deve resultar em 0."""
        img = ler_png_jpg(_png_gray_16bit(valor=0))
        assert img.max() == 0

    # ── PNG RGB ───────────────────────────────────────────────────────────────

    def test_png_rgb_mantém_3_canais(self):
        img = ler_png_jpg(_png_rgb())
        assert img.shape[2] == 3

    def test_png_rgb_uint8(self):
        img = ler_png_jpg(_png_rgb())
        assert img.dtype == np.uint8

    # ── PNG RGBA → BGR ────────────────────────────────────────────────────────

    def test_png_rgba_converte_para_bgr(self):
        """RGBA (4 canais) deve ser convertido para BGR (3 canais)."""
        img = ler_png_jpg(_png_rgba())
        assert img.shape[2] == 3

    # ── JPEG ─────────────────────────────────────────────────────────────────

    def test_jpg_lido_corretamente(self):
        img = ler_png_jpg(_jpg())
        assert img.dtype == np.uint8
        assert img.shape[2] == 3

    # ── Entrada inválida ──────────────────────────────────────────────────────

    def test_bytes_corrompidos_levanta_erro(self):
        """Bytes inválidos devem lançar ValueError."""
        with pytest.raises((ValueError, Exception)):
            ler_png_jpg(b"\x00\x01\x02\x03" * 100)

    def test_bytes_vazios_levanta_erro(self):
        with pytest.raises((ValueError, Exception)):
            ler_png_jpg(b"")


# ════════════════════════════════════════════════════
# LER TIFF
# ════════════════════════════════════════════════════

@pytest.mark.unit
class TestLerTiff:
    """Testa leitura de arquivos TIFF."""

    def test_tiff_retorna_ndarray(self):
        img = ler_tiff(_tiff_8bit())
        assert isinstance(img, np.ndarray)

    def test_tiff_saida_uint8(self):
        img = ler_tiff(_tiff_8bit())
        assert img.dtype == np.uint8

    def test_tiff_saida_3_canais(self):
        img = ler_tiff(_tiff_8bit())
        assert len(img.shape) == 3
        assert img.shape[2] == 3

    def test_tiff_range_valido(self):
        img = ler_tiff(_tiff_8bit())
        assert img.min() >= 0
        assert img.max() <= 255


# ════════════════════════════════════════════════════
# CARREGAR IMAGEM — função principal
# ════════════════════════════════════════════════════

@pytest.mark.unit
class TestCarregarImagem:
    """Testa a função de entrada única que detecta e lê qualquer formato."""

    def test_retorna_tupla(self):
        img, info = carregar_imagem(_png_gray_8bit(), "tc.png")
        assert isinstance(img, np.ndarray)
        assert isinstance(info, dict)

    def test_info_contem_campos_obrigatorios(self):
        _, info = carregar_imagem(_png_gray_8bit(), "tc.png")
        campos = {"formato", "shape", "dtype", "mean", "min", "max"}
        assert campos.issubset(info.keys()), (
            f"Campos ausentes: {campos - info.keys()}"
        )

    def test_info_formato_png(self):
        _, info = carregar_imagem(_png_gray_8bit(), "tc.png")
        assert info["formato"] == "png"

    def test_info_formato_jpg(self):
        _, info = carregar_imagem(_jpg(), "foto.jpg")
        assert info["formato"] == "jpg"

    def test_info_mean_coerente(self):
        """Mean deve estar entre 0 e 255."""
        _, info = carregar_imagem(_png_gray_8bit(valor=80), "tc.png")
        assert 0 <= info["mean"] <= 255

    def test_fallback_pelo_nome_do_arquivo(self):
        """
        Se magic bytes não identificarem o formato,
        deve usar a extensão do nome do arquivo.
        """
        dados_tiff = _tiff_8bit()
        img, info = carregar_imagem(dados_tiff, "histologia.tif")
        assert img is not None

    def test_formato_invalido_levanta_erro(self):
        """Bytes de texto devem lançar ValueError."""
        with pytest.raises(ValueError, match="[Ff]ormato"):
            carregar_imagem(b"nao sou uma imagem", "arquivo.txt")

    def test_bytes_vazios_levanta_erro(self):
        with pytest.raises((ValueError, Exception)):
            carregar_imagem(b"", "vazio.png")


# ════════════════════════════════════════════════════
# IMAGEM PARA BASE64
# ════════════════════════════════════════════════════

@pytest.mark.unit
class TestImagemParaBase64:
    """Testa a conversão de imagem para string base64."""

    def _img_bgr(self) -> np.ndarray:
        return np.full((64, 64, 3), 100, dtype=np.uint8)

    def test_retorna_string(self):
        result = imagem_para_base64(self._img_bgr())
        assert isinstance(result, str)

    def test_comeca_com_data_uri(self):
        result = imagem_para_base64(self._img_bgr())
        assert result.startswith("data:image/png;base64,"), (
            f"Data URI incorreto: {result[:40]}"
        )

    def test_base64_nao_vazio(self):
        result = imagem_para_base64(self._img_bgr())
        b64_parte = result.split(",")[1]
        assert len(b64_parte) > 0

    def test_base64_decodificavel(self):
        """String base64 deve ser decodificável sem erro."""
        import base64
        result   = imagem_para_base64(self._img_bgr())
        b64_parte = result.split(",")[1]
        decoded  = base64.b64decode(b64_parte)
        assert len(decoded) > 0

    def test_imagens_diferentes_geram_base64_diferentes(self):
        img1 = np.full((64, 64, 3), 50,  dtype=np.uint8)
        img2 = np.full((64, 64, 3), 200, dtype=np.uint8)
        assert imagem_para_base64(img1) != imagem_para_base64(img2)


# ════════════════════════════════════════════════════
# DESENHAR DETECÇÕES
# ════════════════════════════════════════════════════

@pytest.mark.unit
class TestDesenharDeteccoes:
    """Testa a função que desenha bounding boxes na imagem."""

    # ── Mock de box para simular saída do YOLO ───────────────────────────────
    class MockBox:
        def __init__(self, cls_id, conf, x1, y1, x2, y2):
            import torch
            self.cls  = torch.tensor([cls_id], dtype=torch.float32)
            self.conf = torch.tensor([conf],   dtype=torch.float32)
            self.xyxy = [torch.tensor([x1, y1, x2, y2], dtype=torch.float32)]

    CLASS_NAMES  = {0: "rim", 1: "tumor", 2: "cisto"}
    CLASS_COLORS = {0: (0,255,0), 1: (255,0,0), 2: (255,255,0)}

    def _img(self) -> np.ndarray:
        return np.full((512, 512, 3), 80, dtype=np.uint8)

    def test_retorna_ndarray(self):
        img = self._img()
        resultado = desenhar_deteccoes(img, [], self.CLASS_NAMES, self.CLASS_COLORS)
        assert isinstance(resultado, np.ndarray)

    def test_sem_deteccoes_nao_altera_shape(self):
        img = self._img()
        resultado = desenhar_deteccoes(img, [], self.CLASS_NAMES, self.CLASS_COLORS)
        assert resultado.shape == img.shape

    def test_nao_modifica_imagem_original(self):
        """A função deve retornar uma cópia, não modificar o original."""
        img    = self._img()
        copia  = img.copy()
        desenhar_deteccoes(img, [], self.CLASS_NAMES, self.CLASS_COLORS)
        assert np.array_equal(img, copia), "Imagem original foi modificada"

    def test_com_deteccao_altera_pixels(self):
        """Com pelo menos uma detecção, a imagem de saída deve ser diferente."""
        img   = self._img()
        boxes = [self.MockBox(cls_id=0, conf=0.9,
                              x1=100, y1=100, x2=300, y2=300)]
        resultado = desenhar_deteccoes(img, boxes, self.CLASS_NAMES, self.CLASS_COLORS)
        assert not np.array_equal(resultado, img), (
            "Imagem com detecção deveria ser diferente da original"
        )

    def test_shape_preservado_com_deteccoes(self):
        img   = self._img()
        boxes = [self.MockBox(0, 0.9, 50, 50, 200, 200)]
        resultado = desenhar_deteccoes(img, boxes, self.CLASS_NAMES, self.CLASS_COLORS)
        assert resultado.shape == (512, 512, 3)

    def test_multiplas_deteccoes(self):
        img   = self._img()
        boxes = [
            self.MockBox(0, 0.95, 10,  10,  200, 200),
            self.MockBox(1, 0.80, 220, 220, 400, 400),
            self.MockBox(2, 0.60, 50,  300, 150, 450),
        ]
        resultado = desenhar_deteccoes(img, boxes, self.CLASS_NAMES, self.CLASS_COLORS)
        assert resultado.shape == img.shape

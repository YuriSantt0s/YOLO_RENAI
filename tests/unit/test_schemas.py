# tests/unit/test_schemas.py
# ═══════════════════════════════════════════════════════════════
# Testes UNITÁRIOS — Backend/app/models/schemas.py
# Valida os modelos Pydantic de entrada e saída da API
# sem nenhuma dependência de HTTP ou modelo YOLO
# ═══════════════════════════════════════════════════════════════

import pytest
from pydantic import ValidationError

from Backend.app.models.schemas import (
    DeteccaoSchema,
    BboxSchema,
    PredictResponseSchema,
    ImageInfoSchema,
)


# ════════════════════════════════════════════════════
# BBOX SCHEMA
# ════════════════════════════════════════════════════

@pytest.mark.unit
class TestBboxSchema:
    """Testa o schema de bounding box."""

    def test_bbox_valido(self):
        bbox = BboxSchema(x1=10, y1=20, x2=300, y2=400)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 300
        assert bbox.y2 == 400

    def test_bbox_coordenadas_zero(self):
        """Coordenadas zero são válidas (objeto na borda)."""
        bbox = BboxSchema(x1=0, y1=0, x2=0, y2=0)
        assert bbox.x1 == 0

    def test_bbox_campos_obrigatorios(self):
        """Todos os campos são obrigatórios."""
        with pytest.raises(ValidationError) as exc:
            BboxSchema(x1=10, y1=20)
        erros = exc.value.errors()
        campos_faltando = {e["loc"][0] for e in erros}
        assert "x2" in campos_faltando
        assert "y2" in campos_faltando

    def test_bbox_rejeita_string(self):
        """Coordenadas devem ser numéricas."""
        with pytest.raises(ValidationError):
            BboxSchema(x1="abc", y1=0, x2=100, y2=100)

    def test_bbox_serializa_para_dict(self):
        bbox = BboxSchema(x1=10, y1=20, x2=300, y2=400)
        d = bbox.model_dump()
        assert set(d.keys()) == {"x1", "y1", "x2", "y2"}


# ════════════════════════════════════════════════════
# DETECÇÃO SCHEMA
# ════════════════════════════════════════════════════

@pytest.mark.unit
class TestDeteccaoSchema:
    """Testa o schema de uma detecção individual."""

    def _bbox(self):
        return {"x1": 10, "y1": 20, "x2": 300, "y2": 400}

    def test_deteccao_valida(self):
        d = DeteccaoSchema(
            classe="rim",
            confianca=0.95,
            confianca_pct="95.0%",
            bbox=self._bbox(),
            area_px=84100,
        )
        assert d.classe == "rim"
        assert d.confianca == 0.95

    def test_confianca_entre_0_e_1(self):
        """Confiança deve estar entre 0 e 1."""
        with pytest.raises(ValidationError):
            DeteccaoSchema(
                classe="rim",
                confianca=1.5,       # inválido — acima de 1
                confianca_pct="150%",
                bbox=self._bbox(),
                area_px=100,
            )

    def test_confianca_negativa_invalida(self):
        with pytest.raises(ValidationError):
            DeteccaoSchema(
                classe="rim",
                confianca=-0.1,
                confianca_pct="-10%",
                bbox=self._bbox(),
                area_px=100,
            )

    def test_classes_validas(self):
        """Todas as classes do modelo devem ser aceitas."""
        for classe in ["rim", "tumor", "cisto"]:
            d = DeteccaoSchema(
                classe=classe,
                confianca=0.8,
                confianca_pct="80%",
                bbox=self._bbox(),
                area_px=500,
            )
            assert d.classe == classe

    def test_area_px_positiva(self):
        """Área em pixels deve ser não-negativa."""
        with pytest.raises(ValidationError):
            DeteccaoSchema(
                classe="rim",
                confianca=0.9,
                confianca_pct="90%",
                bbox=self._bbox(),
                area_px=-100,        # inválido
            )

    def test_campo_bbox_obrigatorio(self):
        with pytest.raises(ValidationError):
            DeteccaoSchema(
                classe="rim",
                confianca=0.9,
                confianca_pct="90%",
                area_px=100,
                # bbox ausente
            )

    def test_serializa_corretamente(self):
        d = DeteccaoSchema(
            classe="tumor",
            confianca=0.75,
            confianca_pct="75.0%",
            bbox=self._bbox(),
            area_px=2500,
        )
        data = d.model_dump()
        assert data["classe"] == "tumor"
        assert data["confianca"] == 0.75
        assert "bbox" in data


# ════════════════════════════════════════════════════
# IMAGE INFO SCHEMA
# ════════════════════════════════════════════════════

@pytest.mark.unit
class TestImageInfoSchema:
    """Testa o schema de metadados da imagem."""

    def _info_valido(self):
        return dict(
            formato="png",
            shape=(512, 512, 3),
            dtype="uint8",
            mean=60.3,
            min=0,
            max=255,
        )

    def test_info_valido(self):
        info = ImageInfoSchema(**self._info_valido())
        assert info.formato == "png"
        assert info.mean == 60.3

    def test_formatos_aceitos(self):
        for fmt in ["png", "jpg", "tiff", "dicom"]:
            info = ImageInfoSchema(**{**self._info_valido(), "formato": fmt})
            assert info.formato == fmt

    def test_mean_entre_0_e_255(self):
        """Média dos pixels deve estar no range uint8."""
        with pytest.raises(ValidationError):
            ImageInfoSchema(**{**self._info_valido(), "mean": 300.0})

    def test_campos_obrigatorios(self):
        with pytest.raises(ValidationError):
            ImageInfoSchema(formato="png")   # faltam shape, dtype, mean, min, max


# ════════════════════════════════════════════════════
# PREDICT RESPONSE SCHEMA
# ════════════════════════════════════════════════════

@pytest.mark.unit
class TestPredictResponseSchema:
    """Testa o schema completo da resposta do /predict."""

    def _deteccao(self, classe="rim", conf=0.9):
        return {
            "classe": classe,
            "confianca": conf,
            "confianca_pct": f"{int(conf*100)}%",
            "bbox": {"x1": 10, "y1": 10, "x2": 300, "y2": 300},
            "area_px": 84100,
        }

    def _info(self):
        return {
            "formato": "png",
            "shape": (512, 512, 3),
            "dtype": "uint8",
            "mean": 60.0,
            "min": 0,
            "max": 255,
        }

    def test_resposta_valida_sem_deteccoes(self):
        r = PredictResponseSchema(
            total_deteccoes=0,
            deteccoes=[],
            imagem_anotada="data:image/png;base64,abc123",
            info_imagem=self._info(),
        )
        assert r.total_deteccoes == 0
        assert r.deteccoes == []

    def test_resposta_valida_com_deteccoes(self):
        r = PredictResponseSchema(
            total_deteccoes=2,
            deteccoes=[
                self._deteccao("rim",   0.95),
                self._deteccao("tumor", 0.80),
            ],
            imagem_anotada="data:image/png;base64,abc123",
            info_imagem=self._info(),
        )
        assert r.total_deteccoes == 2
        assert len(r.deteccoes) == 2

    def test_total_deteccoes_nao_negativo(self):
        with pytest.raises(ValidationError):
            PredictResponseSchema(
                total_deteccoes=-1,
                deteccoes=[],
                imagem_anotada="data:image/png;base64,abc",
                info_imagem=self._info(),
            )

    def test_imagem_anotada_obrigatoria(self):
        with pytest.raises(ValidationError):
            PredictResponseSchema(
                total_deteccoes=0,
                deteccoes=[],
                # imagem_anotada ausente
                info_imagem=self._info(),
            )

    def test_serializa_para_dict(self):
        r = PredictResponseSchema(
            total_deteccoes=1,
            deteccoes=[self._deteccao()],
            imagem_anotada="data:image/png;base64,xyz",
            info_imagem=self._info(),
        )
        data = r.model_dump()
        assert "total_deteccoes" in data
        assert "deteccoes"        in data
        assert "imagem_anotada"   in data
        assert "info_imagem"      in data

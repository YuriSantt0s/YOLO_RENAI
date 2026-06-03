# tests/integration/test_predict.py
# ═══════════════════════════════════════════════════════════════
# Testes de INTEGRAÇÃO — endpoint POST /predict
# Testa o fluxo completo: HTTP → backend → YOLO → resposta
# ═══════════════════════════════════════════════════════════════

import pytest


@pytest.mark.integration
class TestPredictEndpoint:
    """Testes do endpoint POST /predict com imagens válidas."""

    # ── Status HTTP ───────────────────────────────────────────────────────────

    def test_predict_retorna_200_com_png_gray(self, client, png_gray):
        """PNG grayscale 8-bit deve retornar HTTP 200."""
        r = client.post(
            "/predict",
            files={"file": ("tc.png", png_gray, "image/png")},
        )
        assert r.status_code == 200, (
            f"Esperado 200, recebido {r.status_code}\n{r.text}"
        )

    def test_predict_retorna_200_com_png_rgb(self, client, png_rgb):
        """PNG RGB deve retornar HTTP 200."""
        r = client.post(
            "/predict",
            files={"file": ("img.png", png_rgb, "image/png")},
        )
        assert r.status_code == 200

    def test_predict_retorna_200_com_jpg(self, client, jpg_image):
        """JPEG deve retornar HTTP 200."""
        r = client.post(
            "/predict",
            files={"file": ("foto.jpg", jpg_image, "image/jpeg")},
        )
        assert r.status_code == 200

    def test_predict_retorna_200_com_png_16bit(self, client, png_16bit):
        """PNG 16-bit (formato KiTS23) deve retornar HTTP 200."""
        r = client.post(
            "/predict",
            files={"file": ("tc_16bit.png", png_16bit, "image/png")},
        )
        assert r.status_code == 200

    # ── Estrutura da resposta ─────────────────────────────────────────────────

    def test_predict_retorna_json(self, client, png_gray):
        """Resposta deve ser JSON válido."""
        r = client.post(
            "/predict",
            files={"file": ("tc.png", png_gray, "image/png")},
        )
        assert r.headers["content-type"].startswith("application/json")
        data = r.json()
        assert data is not None

    def test_predict_campos_obrigatorios(self, client, png_gray):
        """Resposta deve conter todos os campos esperados."""
        r    = client.post(
            "/predict",
            files={"file": ("tc.png", png_gray, "image/png")},
        )
        data = r.json()
        campos = {"total_deteccoes", "deteccoes", "imagem_anotada", "info_imagem"}
        assert campos.issubset(data.keys()), (
            f"Campos ausentes: {campos - data.keys()}"
        )

    def test_predict_total_deteccoes_e_inteiro(self, client, png_gray):
        data = client.post(
            "/predict",
            files={"file": ("tc.png", png_gray, "image/png")},
        ).json()
        assert isinstance(data["total_deteccoes"], int)
        assert data["total_deteccoes"] >= 0

    def test_predict_deteccoes_e_lista(self, client, png_gray):
        data = client.post(
            "/predict",
            files={"file": ("tc.png", png_gray, "image/png")},
        ).json()
        assert isinstance(data["deteccoes"], list)

    def test_predict_total_coerente_com_lista(self, client, png_gray):
        """total_deteccoes deve bater com o tamanho da lista."""
        data = client.post(
            "/predict",
            files={"file": ("tc.png", png_gray, "image/png")},
        ).json()
        assert data["total_deteccoes"] == len(data["deteccoes"]), (
            f"total={data['total_deteccoes']} mas "
            f"len(deteccoes)={len(data['deteccoes'])}"
        )

    def test_predict_imagem_anotada_e_data_uri(self, client, png_gray):
        """imagem_anotada deve ser um data URI base64 válido."""
        data = client.post(
            "/predict",
            files={"file": ("tc.png", png_gray, "image/png")},
        ).json()
        assert data["imagem_anotada"].startswith("data:image/"), (
            f"Esperado data URI, recebido: {data['imagem_anotada'][:50]}"
        )

    # ── Estrutura de cada detecção ────────────────────────────────────────────

    def test_predict_campos_de_cada_deteccao(self, client, png_gray):
        """Cada detecção deve ter os campos obrigatórios."""
        data = client.post(
            "/predict",
            params={"conf": 0.01},     # conf mínimo para forçar detecções
            files={"file": ("tc.png", png_gray, "image/png")},
        ).json()

        for det in data["deteccoes"]:
            assert "classe"        in det, f"Campo 'classe' ausente: {det}"
            assert "confianca"     in det, f"Campo 'confianca' ausente: {det}"
            assert "confianca_pct" in det, f"Campo 'confianca_pct' ausente: {det}"
            assert "bbox"          in det, f"Campo 'bbox' ausente: {det}"
            assert "area_px"       in det, f"Campo 'area_px' ausente: {det}"

    def test_predict_bbox_tem_4_campos(self, client, png_gray):
        data = client.post(
            "/predict",
            params={"conf": 0.01},
            files={"file": ("tc.png", png_gray, "image/png")},
        ).json()

        for det in data["deteccoes"]:
            bbox = det["bbox"]
            assert set(bbox.keys()) == {"x1", "y1", "x2", "y2"}, (
                f"bbox com campos inesperados: {bbox.keys()}"
            )

    def test_predict_confianca_entre_0_e_1(self, client, png_gray):
        data = client.post(
            "/predict",
            params={"conf": 0.01},
            files={"file": ("tc.png", png_gray, "image/png")},
        ).json()

        for det in data["deteccoes"]:
            assert 0 <= det["confianca"] <= 1, (
                f"Confiança fora do range [0,1]: {det['confianca']}"
            )

    def test_predict_classe_valida(self, client, png_gray):
        """Classes retornadas devem ser rim, tumor ou cisto."""
        data = client.post(
            "/predict",
            params={"conf": 0.01},
            files={"file": ("tc.png", png_gray, "image/png")},
        ).json()

        classes_validas = {"rim", "tumor", "cisto"}
        for det in data["deteccoes"]:
            assert det["classe"] in classes_validas, (
                f"Classe inválida retornada: '{det['classe']}'"
            )

    def test_predict_area_px_positiva(self, client, png_gray):
        data = client.post(
            "/predict",
            params={"conf": 0.01},
            files={"file": ("tc.png", png_gray, "image/png")},
        ).json()

        for det in data["deteccoes"]:
            assert det["area_px"] >= 0, (
                f"area_px negativa: {det['area_px']}"
            )

    # ── Parâmetros de threshold ───────────────────────────────────────────────

    def test_predict_conf_alto_menos_deteccoes(self, client, png_gray):
        """Confiança alta deve retornar igual ou menos detecções que conf baixa."""
        baixo = client.post(
            "/predict", params={"conf": 0.01},
            files={"file": ("tc.png", png_gray, "image/png")},
        ).json()["total_deteccoes"]

        alto = client.post(
            "/predict", params={"conf": 0.90},
            files={"file": ("tc.png", png_gray, "image/png")},
        ).json()["total_deteccoes"]

        assert alto <= baixo, (
            f"conf=0.90 retornou mais detecções ({alto}) "
            f"que conf=0.01 ({baixo})"
        )

    def test_predict_conf_minimo_aceito(self, client, png_gray):
        """conf=0.05 deve retornar HTTP 200."""
        r = client.post(
            "/predict", params={"conf": 0.05},
            files={"file": ("tc.png", png_gray, "image/png")},
        )
        assert r.status_code == 200

    def test_predict_iou_minimo_aceito(self, client, png_gray):
        """iou=0.10 deve retornar HTTP 200."""
        r = client.post(
            "/predict", params={"conf": 0.15, "iou": 0.10},
            files={"file": ("tc.png", png_gray, "image/png")},
        )
        assert r.status_code == 200

    def test_predict_iou_maximo_aceito(self, client, png_gray):
        """iou=0.90 deve retornar HTTP 200."""
        r = client.post(
            "/predict", params={"conf": 0.15, "iou": 0.90},
            files={"file": ("tc.png", png_gray, "image/png")},
        )
        assert r.status_code == 200

    # ── info_imagem ───────────────────────────────────────────────────────────

    def test_predict_info_imagem_presente(self, client, png_gray):
        data = client.post(
            "/predict",
            files={"file": ("tc.png", png_gray, "image/png")},
        ).json()
        assert "info_imagem" in data

    def test_predict_info_formato_correto(self, client, png_gray):
        data = client.post(
            "/predict",
            files={"file": ("tc.png", png_gray, "image/png")},
        ).json()
        assert data["info_imagem"]["formato"] == "png"

    def test_predict_info_mean_valido(self, client, png_gray):
        data = client.post(
            "/predict",
            files={"file": ("tc.png", png_gray, "image/png")},
        ).json()
        mean = data["info_imagem"]["mean"]
        assert 0 <= mean <= 255, f"Mean fora do range: {mean}"

    # ── Imagem pequena ────────────────────────────────────────────────────────

    def test_predict_imagem_pequena_nao_crasha(self, client, tiny_png):
        """Imagem 32×32 não deve causar erro interno."""
        r = client.post(
            "/predict",
            files={"file": ("mini.png", tiny_png, "image/png")},
        )
        assert r.status_code in (200, 422), (
            f"Status inesperado para imagem pequena: {r.status_code}"
        )

    # ── Método HTTP errado ────────────────────────────────────────────────────

    def test_predict_nao_aceita_get(self, client):
        r = client.get("/predict")
        assert r.status_code == 405

    def test_predict_sem_arquivo_retorna_422(self, client):
        """Chamada sem arquivo deve retornar 422 Unprocessable Entity."""
        r = client.post("/predict")
        assert r.status_code == 422

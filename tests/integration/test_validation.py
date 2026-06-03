# tests/integration/test_validation.py
# ═══════════════════════════════════════════════════════════════
# Testes de INTEGRAÇÃO — validação de entradas inválidas
# Garante que o backend rejeita corretamente:
# - Formatos de arquivo não suportados
# - Arquivos corrompidos ou vazios
# - Parâmetros fora do range permitido
# - Ausência de campos obrigatórios
# ═══════════════════════════════════════════════════════════════

import pytest


@pytest.mark.integration
class TestValidacaoFormatos:
    """Testa rejeição de formatos de arquivo inválidos."""

    def test_arquivo_texto_retorna_400(self, client, text_file):
        """Arquivo de texto enviado como imagem deve retornar 400."""
        r = client.post(
            "/predict",
            files={"file": ("documento.txt", text_file, "text/plain")},
        )
        assert r.status_code == 400, (
            f"Arquivo de texto deveria retornar 400, recebeu {r.status_code}"
        )

    def test_arquivo_corrompido_retorna_4xx(self, client, corrupt_bytes):
        """
        Bytes aleatórios devem retornar 400 ou 422.
        Nunca deve retornar 200 ou 500.
        """
        r = client.post(
            "/predict",
            files={"file": ("corrompido.png", corrupt_bytes, "image/png")},
        )
        assert r.status_code in (400, 422), (
            f"Arquivo corrompido deveria retornar 400/422, "
            f"recebeu {r.status_code}"
        )

    def test_arquivo_vazio_retorna_4xx(self, client, empty_bytes):
        """Arquivo completamente vazio deve ser rejeitado."""
        r = client.post(
            "/predict",
            files={"file": ("vazio.png", empty_bytes, "image/png")},
        )
        assert r.status_code in (400, 422), (
            f"Arquivo vazio deveria retornar 400/422, "
            f"recebeu {r.status_code}"
        )

    def test_pdf_retorna_400(self, client):
        """PDF não é formato suportado."""
        pdf_bytes = b"%PDF-1.4 fake pdf content"
        r = client.post(
            "/predict",
            files={"file": ("exame.pdf", pdf_bytes, "application/pdf")},
        )
        assert r.status_code == 400, (
            f"PDF deveria retornar 400, recebeu {r.status_code}"
        )

    def test_json_retorna_400(self, client):
        """Arquivo JSON enviado como imagem deve ser rejeitado."""
        json_bytes = b'{"deteccoes": []}'
        r = client.post(
            "/predict",
            files={"file": ("dados.json", json_bytes, "application/json")},
        )
        assert r.status_code == 400, (
            f"JSON deveria retornar 400, recebeu {r.status_code}"
        )

    def test_html_retorna_400(self, client):
        """Arquivo HTML não é imagem."""
        html_bytes = b"<html><body>nao sou imagem</body></html>"
        r = client.post(
            "/predict",
            files={"file": ("pagina.html", html_bytes, "text/html")},
        )
        assert r.status_code == 400

    def test_arquivo_sem_extensao_corrompido_retorna_4xx(self, client):
        """Arquivo sem extensão com bytes inválidos deve ser rejeitado."""
        r = client.post(
            "/predict",
            files={"file": ("semextensao", b"\x00\x01\x02", "application/octet-stream")},
        )
        assert r.status_code in (400, 422)


@pytest.mark.integration
class TestValidacaoParametros:
    """Testa rejeição de parâmetros fora do range."""

    def test_conf_maior_que_1_retorna_422(self, client, png_gray):
        """conf > 1.0 é inválido."""
        r = client.post(
            "/predict",
            params={"conf": 1.5},
            files={"file": ("tc.png", png_gray, "image/png")},
        )
        assert r.status_code == 422, (
            f"conf=1.5 deveria retornar 422, recebeu {r.status_code}"
        )

    def test_conf_negativo_retorna_422(self, client, png_gray):
        """conf negativo é inválido."""
        r = client.post(
            "/predict",
            params={"conf": -0.1},
            files={"file": ("tc.png", png_gray, "image/png")},
        )
        assert r.status_code == 422, (
            f"conf=-0.1 deveria retornar 422, recebeu {r.status_code}"
        )

    def test_iou_maior_que_1_retorna_422(self, client, png_gray):
        """iou > 1.0 é inválido."""
        r = client.post(
            "/predict",
            params={"iou": 2.0},
            files={"file": ("tc.png", png_gray, "image/png")},
        )
        assert r.status_code == 422, (
            f"iou=2.0 deveria retornar 422, recebeu {r.status_code}"
        )

    def test_iou_negativo_retorna_422(self, client, png_gray):
        """iou negativo é inválido."""
        r = client.post(
            "/predict",
            params={"iou": -0.5},
            files={"file": ("tc.png", png_gray, "image/png")},
        )
        assert r.status_code == 422

    def test_conf_zero_aceito(self, client, png_gray):
        """conf=0.0 é limite válido — deve retornar 200."""
        r = client.post(
            "/predict",
            params={"conf": 0.0},
            files={"file": ("tc.png", png_gray, "image/png")},
        )
        assert r.status_code == 200

    def test_conf_um_aceito(self, client, png_gray):
        """conf=1.0 é limite válido — deve retornar 200."""
        r = client.post(
            "/predict",
            params={"conf": 1.0},
            files={"file": ("tc.png", png_gray, "image/png")},
        )
        assert r.status_code == 200

    def test_iou_zero_aceito(self, client, png_gray):
        """iou=0.0 é limite válido."""
        r = client.post(
            "/predict",
            params={"conf": 0.15, "iou": 0.0},
            files={"file": ("tc.png", png_gray, "image/png")},
        )
        assert r.status_code == 200

    def test_parametros_tipo_string_retorna_422(self, client, png_gray):
        """Parâmetros não numéricos devem ser rejeitados pelo FastAPI."""
        r = client.post(
            "/predict",
            params={"conf": "alto", "iou": "medio"},
            files={"file": ("tc.png", png_gray, "image/png")},
        )
        assert r.status_code == 422


@pytest.mark.integration
class TestValidacaoCamposObrigatorios:
    """Testa ausência de campos obrigatórios na requisição."""

    def test_sem_arquivo_retorna_422(self, client):
        """Requisição sem arquivo deve retornar 422."""
        r = client.post("/predict")
        assert r.status_code == 422, (
            f"Sem arquivo deveria retornar 422, recebeu {r.status_code}"
        )

    def test_campo_errado_retorna_422(self, client, png_gray):
        """
        Arquivo enviado com nome de campo errado ('imagem' em vez de 'file')
        deve retornar 422.
        """
        r = client.post(
            "/predict",
            files={"imagem": ("tc.png", png_gray, "image/png")},
        )
        assert r.status_code == 422, (
            f"Campo errado deveria retornar 422, recebeu {r.status_code}"
        )


@pytest.mark.integration
class TestRespostasDeErro:
    """Testa a estrutura das respostas de erro."""

    def test_erro_400_contem_detail(self, client, text_file):
        """Respostas 400 devem conter o campo 'detail'."""
        r = client.post(
            "/predict",
            files={"file": ("doc.txt", text_file, "text/plain")},
        )
        if r.status_code == 400:
            data = r.json()
            assert "detail" in data, (
                f"Resposta 400 sem 'detail': {data}"
            )

    def test_erro_422_contem_detail(self, client, png_gray):
        """Respostas 422 devem conter o campo 'detail' com os erros."""
        r = client.post(
            "/predict",
            params={"conf": 99.9},
            files={"file": ("tc.png", png_gray, "image/png")},
        )
        if r.status_code == 422:
            data = r.json()
            assert "detail" in data, (
                f"Resposta 422 sem 'detail': {data}"
            )

    def test_erro_nao_retorna_500(self, client, corrupt_bytes):
        """
        Entradas inválidas nunca devem causar erro interno (500).
        500 indica bug no backend, não erro do usuário.
        """
        r = client.post(
            "/predict",
            files={"file": ("corrompido.bin", corrupt_bytes, "image/png")},
        )
        assert r.status_code != 500, (
            f"Arquivo corrompido causou erro interno 500. "
            f"Resposta: {r.text[:200]}"
        )

    def test_endpoint_inexistente_retorna_404(self, client):
        """Endpoint não existente deve retornar 404."""
        r = client.get("/endpoint_que_nao_existe")
        assert r.status_code == 404

    def test_rota_errada_nao_retorna_500(self, client):
        """Rotas erradas não devem causar erro interno."""
        r = client.post("/predic")   # typo intencional
        assert r.status_code in (404, 405, 422)
        assert r.status_code != 500

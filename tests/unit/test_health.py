# tests/integration/test_health.py
# ═══════════════════════════════════════════════════════════════
# Testes de integração — endpoint GET /health
# Verifica: status HTTP, estrutura da resposta e tempo de resposta
# ═══════════════════════════════════════════════════════════════

import time
import pytest


# ── Limites de performance ────────────────────────────────────────────────────
MAX_RESPONSE_TIME_MS = 200   # health deve responder em menos de 200ms


@pytest.mark.integration
class TestHealthEndpoint:
    """Testes do endpoint GET /health."""

    # ── Status HTTP ───────────────────────────────────────────────────────────

    def test_health_retorna_200(self, client):
        """Endpoint deve retornar HTTP 200."""
        r = client.get("/health")
        assert r.status_code == 200, (
            f"Esperado 200, recebido {r.status_code}"
        )

    def test_health_retorna_json(self, client):
        """Resposta deve ser JSON válido."""
        r = client.get("/health")
        assert r.headers["content-type"].startswith("application/json"), (
            "Content-Type deveria ser application/json"
        )
        # Não deve lançar exceção
        data = r.json()
        assert data is not None

    # ── Estrutura da resposta ─────────────────────────────────────────────────

    def test_health_campo_status(self, client):
        """Resposta deve conter o campo 'status'."""
        data = client.get("/health").json()
        assert "status" in data, (
            f"Campo 'status' ausente na resposta: {data}"
        )

    def test_health_status_ok(self, client):
        """Campo status deve ser 'ok'."""
        data = client.get("/health").json()
        assert data["status"] == "ok", (
            f"Esperado status='ok', recebido '{data['status']}'"
        )

    # ── Idempotência ─────────────────────────────────────────────────────────

    def test_health_multiplas_chamadas(self, client):
        """
        Endpoint deve ser idempotente — múltiplas chamadas
        devem retornar o mesmo resultado.
        """
        respostas = [client.get("/health").json() for _ in range(5)]
        for r in respostas:
            assert r["status"] == "ok", (
                f"Resposta inconsistente na chamada múltipla: {r}"
            )

    # ── Performance ───────────────────────────────────────────────────────────

    def test_health_tempo_resposta(self, client):
        """
        Health check deve responder em menos de 200ms.
        Um health lento indica problema na inicialização do servidor.
        """
        inicio = time.perf_counter()
        client.get("/health")
        elapsed_ms = (time.perf_counter() - inicio) * 1000

        assert elapsed_ms < MAX_RESPONSE_TIME_MS, (
            f"Health muito lento: {elapsed_ms:.1f}ms "
            f"(limite: {MAX_RESPONSE_TIME_MS}ms)"
        )

    def test_health_tempo_resposta_media(self, client):
        """
        Média de 10 chamadas deve ficar abaixo do limite.
        Detecta lentidão intermitente.
        """
        tempos = []
        for _ in range(10):
            inicio = time.perf_counter()
            client.get("/health")
            tempos.append((time.perf_counter() - inicio) * 1000)

        media = sum(tempos) / len(tempos)
        assert media < MAX_RESPONSE_TIME_MS, (
            f"Média de resposta alta: {media:.1f}ms "
            f"(limite: {MAX_RESPONSE_TIME_MS}ms)\n"
            f"Tempos individuais: {[f'{t:.1f}' for t in tempos]}"
        )

    # ── Método HTTP errado ────────────────────────────────────────────────────

    def test_health_nao_aceita_post(self, client):
        """
        GET /health não deve aceitar POST.
        FastAPI retorna 405 Method Not Allowed.
        """
        r = client.post("/health")
        assert r.status_code == 405, (
            f"POST em /health deveria retornar 405, recebeu {r.status_code}"
        )

    def test_health_nao_aceita_put(self, client):
        """GET /health não deve aceitar PUT."""
        r = client.put("/health")
        assert r.status_code in (405, 404), (
            f"PUT em /health deveria retornar 405/404, recebeu {r.status_code}"
        )

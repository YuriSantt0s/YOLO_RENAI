# tests/e2e/test_performance.py
# ═══════════════════════════════════════════════════════════════
# Testes E2E — Performance e carga
# Mede tempo de resposta por formato, requisições sequenciais
# e comportamento sob múltiplas chamadas consecutivas.
#
# Limites definidos para CPU (sem GPU):
#   - /health        < 200ms
#   - /predict PNG   < 15s  (inferência YOLO na CPU)
#   - /predict JPG   < 15s
#   - Sequencial 5x  < 75s total (5 × 15s)
# ═══════════════════════════════════════════════════════════════

import time
import statistics
import pytest


# ── Limites de tempo (segundos) ───────────────────────────────────────────────
HEALTH_MAX_S   = 0.20    # 200ms
PREDICT_MAX_S  = 15.0    # 15s por predição na CPU
SEQ_TOTAL_MAX  = 75.0    # 5 chamadas sequenciais
DEGRADATION_MAX = 2.0    # última chamada não pode ser 2x mais lenta que a 1ª


def _predict(client, img_bytes, filename, content_type,
             conf=0.15, iou=0.50):
    """Helper: envia uma imagem e retorna (status_code, elapsed_s, data)."""
    inicio = time.perf_counter()
    r = client.post(
        "/predict",
        params={"conf": conf, "iou": iou},
        files={"file": (filename, img_bytes, content_type)},
    )
    elapsed = time.perf_counter() - inicio
    data = r.json() if r.status_code == 200 else {}
    return r.status_code, elapsed, data


# ════════════════════════════════════════════════════
# TEMPO DE RESPOSTA POR ENDPOINT
# ════════════════════════════════════════════════════

@pytest.mark.e2e
class TestTempoResposta:
    """Tempo de resposta individual por endpoint e formato."""

    def test_health_abaixo_de_200ms(self, client):
        inicio   = time.perf_counter()
        r        = client.get("/health")
        elapsed  = time.perf_counter() - inicio

        assert r.status_code == 200
        assert elapsed < HEALTH_MAX_S, (
            f"/health demorou {elapsed*1000:.1f}ms "
            f"(limite: {HEALTH_MAX_S*1000:.0f}ms)"
        )

    def test_predict_png_gray_dentro_do_limite(self, client, png_gray):
        status, elapsed, _ = _predict(
            client, png_gray, "tc.png", "image/png"
        )
        assert status == 200
        assert elapsed < PREDICT_MAX_S, (
            f"PNG grayscale: {elapsed:.2f}s (limite: {PREDICT_MAX_S}s)"
        )

    def test_predict_png_rgb_dentro_do_limite(self, client, png_rgb):
        status, elapsed, _ = _predict(
            client, png_rgb, "img.png", "image/png"
        )
        assert status == 200
        assert elapsed < PREDICT_MAX_S, (
            f"PNG RGB: {elapsed:.2f}s (limite: {PREDICT_MAX_S}s)"
        )

    def test_predict_jpg_dentro_do_limite(self, client, jpg_image):
        status, elapsed, _ = _predict(
            client, jpg_image, "foto.jpg", "image/jpeg"
        )
        assert status == 200
        assert elapsed < PREDICT_MAX_S, (
            f"JPEG: {elapsed:.2f}s (limite: {PREDICT_MAX_S}s)"
        )

    def test_predict_png_16bit_dentro_do_limite(self, client, png_16bit):
        """PNG 16-bit tem etapa extra de conversão — deve continuar no limite."""
        status, elapsed, _ = _predict(
            client, png_16bit, "tc_16bit.png", "image/png"
        )
        assert status == 200
        assert elapsed < PREDICT_MAX_S, (
            f"PNG 16-bit: {elapsed:.2f}s (limite: {PREDICT_MAX_S}s)"
        )


# ════════════════════════════════════════════════════
# CONSISTÊNCIA — múltiplas chamadas sequenciais
# ════════════════════════════════════════════════════

@pytest.mark.e2e
class TestConsistenciaSequencial:
    """
    Testa que o backend mantém performance estável em
    chamadas sequenciais — detecta memory leak e degradação.
    """

    N = 5   # número de chamadas sequenciais

    def test_n_chamadas_retornam_200(self, client, png_gray):
        """Todas as chamadas sequenciais devem retornar 200."""
        for i in range(self.N):
            status, _, _ = _predict(client, png_gray, "tc.png", "image/png")
            assert status == 200, (
                f"Chamada {i+1}/{self.N} retornou {status}"
            )

    def test_tempo_total_sequencial(self, client, png_gray):
        """N chamadas sequenciais devem caber no limite total."""
        inicio = time.perf_counter()
        for _ in range(self.N):
            _predict(client, png_gray, "tc.png", "image/png")
        total = time.perf_counter() - inicio

        assert total < SEQ_TOTAL_MAX, (
            f"{self.N} chamadas levaram {total:.2f}s "
            f"(limite: {SEQ_TOTAL_MAX}s)"
        )

    def test_sem_degradacao_progressiva(self, client, png_gray):
        """
        A última chamada não deve ser DEGRADATION_MAX vezes
        mais lenta que a primeira — indica memory leak ou warmup.
        """
        tempos = []
        for _ in range(self.N):
            _, elapsed, _ = _predict(client, png_gray, "tc.png", "image/png")
            tempos.append(elapsed)

        primeiro = tempos[0]
        ultimo   = tempos[-1]

        assert ultimo < primeiro * DEGRADATION_MAX, (
            f"Degradação detectada: "
            f"1ª={primeiro:.2f}s → {self.N}ª={ultimo:.2f}s "
            f"(fator {ultimo/primeiro:.1f}x, limite {DEGRADATION_MAX}x)"
        )

    def test_resultados_consistentes(self, client, png_gray):
        """
        O número de detecções deve ser idêntico para a mesma imagem
        com os mesmos parâmetros — o modelo é determinístico.
        """
        totais = []
        for _ in range(3):
            _, _, data = _predict(
                client, png_gray, "tc.png", "image/png",
                conf=0.15, iou=0.50,
            )
            if data:
                totais.append(data.get("total_deteccoes", -1))

        if len(totais) > 1:
            assert len(set(totais)) == 1, (
                f"Detecções inconsistentes entre chamadas: {totais}"
            )


# ════════════════════════════════════════════════════
# ESTATÍSTICAS DE PERFORMANCE
# ════════════════════════════════════════════════════

@pytest.mark.e2e
class TestEstatisticasPerformance:
    """
    Coleta estatísticas detalhadas de tempo.
    Não falha o build — apenas reporta os números.
    Útil para monitorar regressões ao longo do tempo.
    """

    N = 5

    def test_relatorio_de_performance(self, client, png_gray, capsys):
        """
        Roda N predições e imprime média, mediana, min e max.
        Falha apenas se a média ultrapassar o limite.
        """
        tempos = []
        for i in range(self.N):
            _, elapsed, _ = _predict(
                client, png_gray, "tc.png", "image/png"
            )
            tempos.append(elapsed)

        media   = statistics.mean(tempos)
        mediana = statistics.median(tempos)
        desvio  = statistics.stdev(tempos) if len(tempos) > 1 else 0
        minimo  = min(tempos)
        maximo  = max(tempos)

        with capsys.disabled():
            print(f"\n{'─'*50}")
            print(f"  RELATÓRIO DE PERFORMANCE — /predict (CPU)")
            print(f"{'─'*50}")
            print(f"  Amostras : {self.N}")
            print(f"  Média    : {media:.2f}s")
            print(f"  Mediana  : {mediana:.2f}s")
            print(f"  Desvio   : {desvio:.2f}s")
            print(f"  Mínimo   : {minimo:.2f}s")
            print(f"  Máximo   : {maximo:.2f}s")
            print(f"  Limite   : {PREDICT_MAX_S}s")
            status = "✓ DENTRO DO LIMITE" if media < PREDICT_MAX_S else "✗ ACIMA DO LIMITE"
            print(f"  Status   : {status}")
            print(f"{'─'*50}")

        assert media < PREDICT_MAX_S, (
            f"Média de {media:.2f}s acima do limite de {PREDICT_MAX_S}s"
        )

    def test_health_relatorio(self, client, capsys):
        """Estatísticas do /health."""
        N = 20
        tempos = []
        for _ in range(N):
            inicio  = time.perf_counter()
            client.get("/health")
            tempos.append(time.perf_counter() - inicio)

        media  = statistics.mean(tempos) * 1000
        maximo = max(tempos) * 1000

        with capsys.disabled():
            print(f"\n{'─'*50}")
            print(f"  RELATÓRIO DE PERFORMANCE — /health")
            print(f"{'─'*50}")
            print(f"  Amostras : {N}")
            print(f"  Média    : {media:.1f}ms")
            print(f"  Máximo   : {maximo:.1f}ms")
            print(f"  Limite   : {HEALTH_MAX_S*1000:.0f}ms")
            print(f"{'─'*50}")

        assert media < HEALTH_MAX_S * 1000, (
            f"Média do /health: {media:.1f}ms "
            f"(limite: {HEALTH_MAX_S*1000:.0f}ms)"
        )


# ════════════════════════════════════════════════════
# CONF THRESHOLD — impacto no tempo
# ════════════════════════════════════════════════════

@pytest.mark.e2e
class TestImpactoThreshold:
    """
    Verifica que diferentes thresholds não impactam
    significativamente o tempo de inferência.
    O YOLO roda a inferência completa independente do conf —
    o filtro é aplicado depois.
    """

    def test_conf_baixo_nao_e_mais_lento(self, client, png_gray):
        """conf=0.01 não deve ser significativamente mais lento que conf=0.90."""
        _, t_baixo, _ = _predict(
            client, png_gray, "tc.png", "image/png", conf=0.01
        )
        _, t_alto, _ = _predict(
            client, png_gray, "tc.png", "image/png", conf=0.90
        )

        # Tolerância de 3x — diferença esperada é mínima
        assert t_baixo < t_alto * 3, (
            f"conf=0.01 muito mais lento: {t_baixo:.2f}s vs {t_alto:.2f}s"
        )

    def test_iou_baixo_nao_e_mais_lento(self, client, png_gray):
        """iou=0.10 não deve ser significativamente mais lento que iou=0.90."""
        _, t_baixo, _ = _predict(
            client, png_gray, "tc.png", "image/png", iou=0.10
        )
        _, t_alto, _ = _predict(
            client, png_gray, "tc.png", "image/png", iou=0.90
        )

        assert t_baixo < t_alto * 3, (
            f"iou=0.10 muito mais lento: {t_baixo:.2f}s vs {t_alto:.2f}s"
        )

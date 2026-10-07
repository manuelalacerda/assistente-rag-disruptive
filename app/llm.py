"""Único ponto do projeto que conversa com a API do Gemini.

Por que isolar: (1) se trocar de modelo/provedor, só este arquivo muda;
(2) nos testes substituímos esta classe por uma versão falsa, sem gastar cota
nem precisar de internet; (3) o tratamento de erros/retentativas fica num lugar só.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Callable, Optional

import numpy as np

from app.config import Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Falha ao falar com o modelo (cota, rede, resposta vazia...)."""


class LLMQuotaDiaria(LLMError):
    """A cota DIÁRIA gratuita do modelo acabou. Esperar segundos não adianta: só amanhã (ou outro modelo)."""


# Formatos de texto recomendados no Lab 4 para o modelo gemini-embedding-2:
# a pergunta e o documento recebem "rótulos" diferentes, o que melhora a busca.
def formatar_consulta(pergunta: str) -> str:
    return f"task: question answering | query: {pergunta}"


def formatar_documento(titulo: str, texto: str) -> str:
    return f"title: {titulo} | text: {texto}"


def _normalizar(matriz: np.ndarray) -> np.ndarray:
    """Vetores de tamanho 1: assim o produto escalar vira a similaridade de cosseno."""
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    return (matriz / normas).astype(np.float32)


def _com_retentativas(fn: Callable, esperas_cota: tuple = (3, 8)):
    """Repete a chamada quando a API reclama de cota (429), instabilidade (503) ou rede.

    `esperas_cota`: segundos de espera a cada nova tentativa em caso de 429. No chat usamos
    esperas curtas (o aluno está olhando a tela); no build do índice, esperas longas, que
    atravessam a janela de 1 minuto do limite gratuito.
    """
    tentativas = len(esperas_cota) + 1
    for i in range(tentativas):
        try:
            return fn()
        except Exception as e:  # o SDK levanta tipos diferentes; olhamos a mensagem
            msg = str(e)
            cota = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            if cota and "PerDay" in msg:   # ex.: GenerateRequestsPerDayPerProjectPerModel-FreeTier
                modelo = re.search(r"model: ([\w.\-]+)", msg)
                espera = re.search(r"retry in ([\w.]+)", msg)
                raise LLMQuotaDiaria(
                    f"Cota diária gratuita esgotada (modelo {modelo.group(1) if modelo else '?'}); "
                    f"renova em ~{espera.group(1) if espera else 'algumas horas'}."
                ) from e
            temporario = cota or any(c in msg for c in (
                "500", "503", "504", "UNAVAILABLE", "10060", "timed out", "Timeout", "Connection"))
            if not temporario or i == tentativas - 1:
                raise LLMError(f"Falha na API do Gemini: {msg[:300]}") from e
            espera = esperas_cota[i] if cota else 2.0 * 2 ** i
            log.warning("Gemini temporariamente indisponível (%s). Nova tentativa em %.0fs", msg[:60], espera)
            time.sleep(espera)


class GeminiClient:
    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY não definida. Copie .env.example para .env e preencha.")
        from google import genai  # import tardio: os testes não precisam do SDK instalado
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self.s = settings
        self.esperas_cota = (3, 8)       # o build do índice aumenta isso
        self.pausa_embedding = 0.1       # segundos entre embeddings individuais

    # ---------------------------------------------------------------- embeddings
    def _chamar_embedding(self, textos: list[str]) -> list:
        cfg = None
        if self.s.embedding_dim:
            cfg = self._types.EmbedContentConfig(output_dimensionality=self.s.embedding_dim)
        resp = _com_retentativas(
            lambda: self._client.models.embed_content(model=self.s.embedding_model, contents=textos, config=cfg),
            self.esperas_cota,
        )
        return [e.values for e in resp.embeddings]

    def _embed(self, textos: list[str]) -> np.ndarray:
        """Devolve UM vetor por texto.

        Alguns modelos (ex.: gemini-embedding-2, multimodal) tratam uma lista como partes de um
        ÚNICO conteúdo e devolvem 1 vetor só. Se o número de vetores não bater com o de textos,
        passamos a embedar um texto por chamada (mais lento, porém sempre correto).
        """
        if len(textos) > 1 and getattr(self, "_lote_funciona", True):
            vetores = self._chamar_embedding(textos)
            if len(vetores) == len(textos):
                return _normalizar(np.array(vetores, dtype=np.float32))
            log.warning("Modelo devolveu %d vetores para %d textos; passando a embedar um por vez.",
                        len(vetores), len(textos))
            self._lote_funciona = False

        vetores = []
        for texto in textos:
            v = self._chamar_embedding([texto])
            if len(v) != 1:
                raise LLMError(f"Embedding inesperado: {len(v)} vetores para 1 texto.")
            vetores.append(v[0])
            if len(textos) > 1:
                time.sleep(self.pausa_embedding)
        return _normalizar(np.array(vetores, dtype=np.float32))

    def embed_consulta(self, pergunta: str) -> np.ndarray:
        return self._embed([formatar_consulta(pergunta)])[0]

    def embed_documentos(self, pares: list[tuple[str, str]], lote: int = 50,
                         progresso: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
        """pares = [(título, texto), ...]. Envia em lotes para respeitar limites da API."""
        partes = []
        for i in range(0, len(pares), lote):
            textos = [formatar_documento(t, x) for t, x in pares[i:i + lote]]
            partes.append(self._embed(textos))
            if progresso:
                progresso(min(i + lote, len(pares)), len(pares))
            time.sleep(0.5)  # gentileza com o limite de requisições do plano gratuito
        return np.vstack(partes)

    # ---------------------------------------------------------------- geração
    def gerar(self, prompt: str, system: str, modelo: Optional[str] = None,
              temperatura: Optional[float] = None) -> str:
        """Gera texto. Se a cota DIÁRIA do modelo acabar, tenta o `fallback_model` (cada modelo tem cota própria)."""
        modelo = modelo or self.s.generation_model
        try:
            return self._gerar(prompt, system, modelo, temperatura)
        except LLMQuotaDiaria:
            reserva = self.s.fallback_model
            if not reserva or reserva == modelo:
                raise
            log.warning("Cota diária de %s esgotada; usando o modelo de reserva %s", modelo, reserva)
            return self._gerar(prompt, system, reserva, temperatura)

    def _gerar(self, prompt: str, system: str, modelo: str, temperatura: Optional[float]) -> str:
        cfg = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.s.temperature if temperatura is None else temperatura,
        )
        resp = _com_retentativas(
            lambda: self._client.models.generate_content(model=modelo, contents=prompt, config=cfg),
            self.esperas_cota,
        )
        texto = (resp.text or "").strip()
        if not texto:
            raise LLMError("O modelo devolveu uma resposta vazia (possível bloqueio de segurança).")
        return texto

"""Peças reutilizáveis pelos testes: um Gemini falso e um mini-repositório de disciplina."""

import json
import re
import zlib
from dataclasses import replace

import numpy as np
import pytest

from app.config import Settings
from app.prompts import SYSTEM_REESCRITA


class FakeLLM:
    """Imita o GeminiClient sem rede: embeddings por "saco de palavras" e respostas roteirizadas."""

    DIM = 128

    def __init__(self, resposta: str = "Resposta de teste [1].", falhar_embedding: bool = False):
        self.resposta = resposta
        self.falhar_embedding = falhar_embedding
        self.chamadas: list[dict] = []

    def _vetor(self, texto: str) -> np.ndarray:
        v = np.zeros(self.DIM, dtype=np.float32)
        for t in re.findall(r"\w+", texto.lower()):
            v[zlib.crc32(t.encode()) % self.DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_consulta(self, pergunta: str) -> np.ndarray:
        if self.falhar_embedding:
            raise RuntimeError("API fora do ar")
        return self._vetor(pergunta)

    def embed_documentos(self, pares, lote=50, progresso=None) -> np.ndarray:
        return np.vstack([self._vetor(f"{t} {x}") for t, x in pares])

    def gerar(self, prompt, system, modelo=None, temperatura=None) -> str:
        self.chamadas.append({"prompt": prompt, "system": system, "modelo": modelo})
        if system == SYSTEM_REESCRITA:
            return "consulta reescrita sobre MQTT"
        return self.resposta


@pytest.fixture
def settings(tmp_path) -> Settings:
    return replace(Settings(), gemini_api_key="x", chat_db_path=str(tmp_path / "chat.db"),
                   index_db_path=str(tmp_path / "index.db"), rate_limit_per_min=1000,
                   rate_limit_global_per_min=1000, min_similarity=0.0)


MKDOCS = """
site_name: Teste
site_url: https://exemplo.github.io/curso/
extra: {}
markdown_extensions:
  - pymdownx.emoji:
      emoji_index: !!python/name:material.extensions.emoji.twemoji
nav:
  - Home: index.md
  - 1º Semestre - IoT:
    - aulas/iot/index.md
    - ESP32:
      - Lab09 - MQTT: aulas/iot/mqtt/index.md
  - 2º Semestre - IA:
    - Lab4 - RAG: aulas/genAI/lab4/lab4.md
"""

PAGINAS = {
    "index.md": "# Início\n\nBem-vindos à disciplina de arquiteturas disruptivas.",
    "aulas/iot/index.md": "# IoT\n\nIntrodução à Internet das Coisas com sensores e atuadores conectados.",
    "aulas/iot/mqtt/index.md": (
        "## MQTT com ESP32\n\nMQTT é um protocolo de mensagens baseado em publicação e assinatura. "
        "O broker recebe as mensagens e entrega aos clientes inscritos nos tópicos.\n\n"
        "## Código\n\n```cpp\nconst char* MQTT_BROKER = \"broker.hivemq.com\";\nconst int MQTT_PORT = 1883;\n```\n"
    ),
    "aulas/genAI/lab4/lab4.md": (
        "# RAG\n\nRAG significa geração aumentada por recuperação. Embeddings representam textos como vetores.\n\n"
        "## Chunks\n\nDocumentos extensos são divididos em trechos chamados chunks, cada um com seu embedding."
    ),
    "aulas/antiga/obsoleta.md": "# Antiga\n\nConteúdo esquecido que NÃO está no menu do site.",
}

NOTEBOOK = {"cells": [
    {"cell_type": "markdown", "source": ["## Embeddings\n", "Gere vetores com o Gemini."]},
    {"cell_type": "code", "source": ["resultado = client.models.embed_content(model=M, contents=t)"], "outputs": [{"text": "ruido"}]},
    {"cell_type": "code", "source": []},
]}


@pytest.fixture
def repo_disciplina(tmp_path):
    """Cria <tmp>/repo/{mkdocs.yml, material/...} imitando a estrutura real."""
    raiz = tmp_path / "repo"
    (raiz / "material").mkdir(parents=True)
    (raiz / "mkdocs.yml").write_text(MKDOCS, encoding="utf-8")
    for caminho, texto in PAGINAS.items():
        arq = raiz / "material" / caminho
        arq.parent.mkdir(parents=True, exist_ok=True)
        arq.write_text(texto, encoding="utf-8")
    (raiz / "material/aulas/genAI/lab4/lab4_rag.ipynb").write_text(json.dumps(NOTEBOOK), encoding="utf-8")
    return raiz


@pytest.fixture
def fake_llm():
    return FakeLLM()

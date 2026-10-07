"""O "cérebro" do sistema: orquestra o fluxo completo de uma pergunta.

    pergunta ──► (1) reescrever (se há histórico) ──► (2) buscar trechos
             ──► (3) montar prompt ──► (4) gerar resposta ──► (5) extrair fontes citadas

Este módulo não sabe nada de HTTP nem de banco de dados de conversa: recebe a
pergunta + histórico e devolve um `Resultado`. Isso o torna fácil de testar e de
reaproveitar (API, linha de comando, avaliação).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from app.config import Settings
from app.prompts import SYSTEM_PROMPT, SYSTEM_REESCRITA, prompt_reescrita, prompt_resposta
from app.retriever import Hit, Retriever

log = logging.getLogger(__name__)

NAO_ENCONTREI = "Não encontrei isso no material da disciplina."
FORA_DO_ESCOPO = "Só respondo sobre o conteúdo da disciplina."
_PREFIXOS_RECUSA = ("Não encontrei", "Só respondo")   # mesmas frases definidas no SYSTEM_PROMPT


def eh_recusa(resposta: str) -> bool:
    return resposta.lstrip().startswith(_PREFIXOS_RECUSA)


def remover_citacoes(texto: str) -> str:
    """Numa recusa não mostramos fontes; então os marcadores [n] ficariam soltos e sem link."""
    return re.sub(r"[ \t]*(?:\[\d+(?:\s*,\s*\d+)*\])+", "", texto)


@dataclass
class Fonte:
    n: int            # número usado na citação [n]
    titulo: str
    trilha: str
    secao: str
    url: str
    tipo: str
    trecho: str       # prévia curta do trecho usado


@dataclass
class Resultado:
    resposta: str
    fontes: list[Fonte]
    consulta: str                       # consulta efetivamente usada na busca
    hits: list[Hit] = field(default_factory=list)
    tempos_ms: dict = field(default_factory=dict)
    modelo: str = ""


def extrair_citacoes(resposta: str, maximo: int) -> list[int]:
    """Números [n] citados, na ordem em que aparecem, ignorando colchetes dentro de código."""
    sem_codigo = re.sub(r"```.*?```", "", resposta, flags=re.DOTALL)
    sem_codigo = re.sub(r"`[^`]*`", "", sem_codigo)
    vistos: list[int] = []
    for grupo in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", sem_codigo):
        for n in (int(x) for x in re.split(r"\s*,\s*", grupo)):
            if 1 <= n <= maximo and n not in vistos:
                vistos.append(n)
    return vistos


def _previa(texto: str, limite: int = 260) -> str:
    plano = re.sub(r"^#+\s*", "", texto, flags=re.MULTILINE)
    plano = re.sub(r"\s+", " ", plano).strip()
    return plano if len(plano) <= limite else plano[:limite].rsplit(" ", 1)[0] + "…"


class RagService:
    def __init__(self, retriever: Retriever, llm, settings: Settings):
        self.retriever = retriever
        self.llm = llm
        self.s = settings

    def _reescrever(self, pergunta: str, historico: list[dict]) -> str:
        """Perguntas de continuação ("e no lab 3?") não funcionam na busca. Torna-as autônomas."""
        if not historico:
            return pergunta
        try:
            consulta = self.llm.gerar(
                prompt_reescrita(historico, pergunta), SYSTEM_REESCRITA,
                modelo=self.s.rewrite_model, temperatura=0.0,
            )
            return consulta.strip().strip('"')[:400] or pergunta
        except Exception as e:  # reescrever é um "extra": se falhar, usamos a pergunta original
            log.warning("Falha ao reescrever pergunta: %s", e)
            return pergunta

    def responder(self, pergunta: str, historico: list[dict] | None = None) -> Resultado:
        historico = (historico or [])[-2 * self.s.history_turns:]
        tempos: dict[str, int] = {}

        t0 = time.perf_counter()
        consulta = self._reescrever(pergunta, historico)
        tempos["reescrita"] = int((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        hits = self.retriever.buscar(consulta, k=self.s.top_k, candidatos=self.s.candidates)
        tempos["busca"] = int((time.perf_counter() - t0) * 1000)

        # Portão de qualidade: sem trecho relevante, nem chamamos o LLM (mais barato e sem alucinação).
        melhor_sim = max((h.similaridade or 0.0 for h in hits), default=0.0)
        sem_base = not hits or (
            self.s.min_similarity > 0 and self.retriever.tem_vetores and melhor_sim < self.s.min_similarity
        )
        if sem_base:
            return Resultado(NAO_ENCONTREI, [], consulta, hits, tempos, modelo="")

        t0 = time.perf_counter()
        resposta = self.llm.gerar(prompt_resposta(pergunta, hits, historico), SYSTEM_PROMPT)
        tempos["geracao"] = int((time.perf_counter() - t0) * 1000)

        fontes: list[Fonte] = []
        if eh_recusa(resposta):
            resposta = remover_citacoes(resposta)
        else:
            citados = extrair_citacoes(resposta, len(hits)) or [1, 2]  # sem citação: mostra os 2 melhores
            for n in citados:
                if n <= len(hits):
                    h = hits[n - 1]
                    fontes.append(Fonte(n, h.titulo, h.trilha, h.secao, h.url, h.tipo, _previa(h.texto)))

        return Resultado(resposta, fontes, consulta, hits, tempos, modelo=self.s.generation_model)

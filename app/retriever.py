"""Busca híbrida: junta DOIS jeitos de procurar e combina os resultados.

1. Busca semântica (vetores): acha trechos com o mesmo SIGNIFICADO, mesmo com
   palavras diferentes. ("até quando posso desistir da compra?" -> "devolução em 7 dias")
2. Busca por palavras (BM25, via SQLite FTS5): acha termos exatos e raros — "MQTT",
   "ESP32", "Serial.begin", "CP2". É onde embeddings costumam errar.

A combinação usa RRF (Reciprocal Rank Fusion): cada busca dá uma posição (1º, 2º...)
a cada trecho, e somamos 1/(60 + posição). Um trecho bem colocado nas duas listas
sobe no ranking. O RRF dispensa calibrar escalas de nota diferentes.

Se a API de embeddings estiver fora do ar, a busca por palavras continua funcionando.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

RRF_K = 60
MAX_POR_PAGINA = 5   # evita que uma única página ocupe TODOS os resultados (mas deixa a página certa ir fundo)

_STOPWORDS = set("""
a o as os um uma uns umas de do da dos das em no na nos nas por para com sem sobre ao aos à às e ou mas que
qual quais quem como onde quando porque por que se é são foi ser era tem têm ter há eu me meu minha seu sua
isso isto esse essa esses essas este esta estes estas aquele aquela lá aqui já mais menos muito muita também
só pode posso podem faz fazer fazem preciso quero gostaria explique explica diga fale
quanto quantos quanta quantas usa usar usado existe existem gero gera dá dar vai vão
""".split())


@dataclass
class Hit:
    id: int
    titulo: str
    trilha: str
    secao: str
    url: str
    tipo: str
    texto: str
    score: float = 0.0                 # nota final (RRF)
    similaridade: Optional[float] = None   # cosseno com a pergunta (se houve busca vetorial)
    posicao_vetorial: Optional[int] = None
    posicao_palavras: Optional[int] = None


def consulta_fts(pergunta: str) -> str:
    """Transforma a pergunta numa consulta FTS5 segura: palavras úteis ligadas por OR."""
    tokens = re.findall(r"\w+", pergunta.lower())
    uteis = [t for t in tokens if t not in _STOPWORDS and (len(t) > 1 or t.isdigit())]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(uteis))


class Retriever:
    def __init__(self, db_path: str, embed_consulta: Optional[Callable[[str], np.ndarray]] = None):
        self.db_path = db_path
        self.embed_consulta = embed_consulta
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        linhas = con.execute(
            "SELECT id, titulo, trilha, secao, url, tipo, texto, embedding FROM chunks ORDER BY id"
        ).fetchall()
        self.meta = dict(con.execute("SELECT chave, valor FROM meta").fetchall())
        n_colunas_fts = len(con.execute("PRAGMA table_info(chunks_fts)").fetchall())
        con.close()
        # pesos do BM25 por coluna (titulo, secao, texto[, resumo]); índices antigos não têm "resumo"
        self._pesos_bm25 = ", ".join(str(p) for p in (3.0, 2.0, 1.0, 2.0)[:n_colunas_fts])

        self.chunks: dict[int, Hit] = {}
        vetores, ids = [], []
        for r in linhas:
            self.chunks[r["id"]] = Hit(r["id"], r["titulo"], r["trilha"], r["secao"], r["url"], r["tipo"], r["texto"])
            if r["embedding"] is not None:
                ids.append(r["id"])
                vetores.append(np.frombuffer(r["embedding"], dtype=np.float32))
        # matriz (n_chunks x dimensão): a busca vetorial vira uma única multiplicação
        self.ids_vetoriais = np.array(ids)
        self.matriz = np.vstack(vetores) if vetores else None

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    @property
    def tem_vetores(self) -> bool:
        return self.matriz is not None

    # ------------------------------------------------------------ as duas buscas
    def _busca_vetorial(self, pergunta: str, n: int) -> list[tuple[int, float]]:
        if self.matriz is None or self.embed_consulta is None:
            return []
        try:
            q = self.embed_consulta(pergunta)
        except Exception as e:  # API fora do ar: seguimos só com palavras
            log.warning("Busca vetorial indisponível, usando só palavras: %s", e)
            return []
        sims = self.matriz @ q
        melhores = np.argsort(-sims)[:n]
        return [(int(self.ids_vetoriais[i]), float(sims[i])) for i in melhores]

    def _busca_palavras(self, pergunta: str, n: int) -> list[int]:
        consulta = consulta_fts(pergunta)
        if not consulta:
            return []
        con = sqlite3.connect(self.db_path)
        try:
            # pesos: palavra no título (3x), na seção (2x) ou no resumo (2x) vale mais que no corpo
            linhas = con.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                f"ORDER BY bm25(chunks_fts, {self._pesos_bm25}) LIMIT ?",
                (consulta, n),
            ).fetchall()
        finally:
            con.close()
        return [r[0] for r in linhas]

    # ------------------------------------------------------------ API pública
    def buscar(self, pergunta: str, k: int = 6, candidatos: int = 25) -> list[Hit]:
        vet = self._busca_vetorial(pergunta, candidatos)
        pal = self._busca_palavras(pergunta, candidatos)

        scores: dict[int, float] = {}
        info: dict[int, dict] = {}
        for pos, (cid, sim) in enumerate(vet, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + pos)
            info.setdefault(cid, {}).update(similaridade=sim, posicao_vetorial=pos)
        for pos, cid in enumerate(pal, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + pos)
            info.setdefault(cid, {}).update(posicao_palavras=pos)

        resultado: list[Hit] = []
        por_pagina: dict[str, int] = {}
        for cid in sorted(scores, key=scores.get, reverse=True):
            base = self.chunks[cid]
            if por_pagina.get(base.url, 0) >= MAX_POR_PAGINA:
                continue
            por_pagina[base.url] = por_pagina.get(base.url, 0) + 1
            resultado.append(Hit(**{**base.__dict__, "score": scores[cid], **info[cid]}))
            if len(resultado) == k:
                break
        return resultado

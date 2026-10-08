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
import unicodedata
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


_STOPWORDS_SEM_ACENTO = {unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode("ascii") for w in _STOPWORDS}


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


def _sem_acento(texto: str) -> str:
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii").lower()


def _palavras_de_assunto(texto: str) -> set[str]:
    """Palavras que dizem DO QUE se trata (sem acento, sem 'lab', números e palavras comuns)."""
    tokens = re.findall(r"\w+", _sem_acento(texto))
    return {t for t in tokens if t not in _STOPWORDS_SEM_ACENTO and t not in ("lab", "labs") and not t.isdigit()}


def filtrar_por_assunto(pergunta: str, paginas: dict) -> dict:
    """Se a pergunta traz o assunto ("lab 9 MQTT"), fica só a página cujo título/menu o cita.

    `paginas`: {url: Hit}. Só filtra quando o assunto distingue as páginas (algumas casam, outras não).
    """
    if len(paginas) < 2:
        return paginas
    assunto = _palavras_de_assunto(pergunta)
    casadas = {}
    for url, h in paginas.items():
        titulo_menu = re.sub(r"\bLab\s*0*\d+(?:[.,_]\d+)?", "", f"{h.titulo} {h.trilha}", flags=re.IGNORECASE)
        if assunto & _palavras_de_assunto(titulo_menu):
            casadas[url] = h
    return casadas if 0 < len(casadas) < len(paginas) else paginas


def eh_codigo(texto: str, limite: float = 0.5) -> bool:
    codigo = sum(len(b.split()) for b in re.findall(r"```.*?```", texto, flags=re.DOTALL))
    return codigo / max(len(texto.split()), 1) >= limite


def labs_citados(pergunta: str) -> list[str]:
    """Números de lab citados na pergunta: 'lab 3' -> ['3'], 'Lab3.5' -> ['3.5'], 'labs 8 e 9' -> ['8']."""
    achados = re.findall(r"\blab\s*0*(\d+(?:[.,_]\d+)?)", pergunta, flags=re.IGNORECASE)
    return list(dict.fromkeys(a.replace(",", ".").replace("_", ".") for a in achados))


def titulo_e_do_lab(titulo: str, numero: str) -> bool:
    """'Lab03 - Serial' e 'Lab3 - Ferramentas' são o lab 3; 'Lab3.5 - ...' NÃO é."""
    n = re.escape(numero)
    return re.search(rf"\bLab\s*0*{n}(?![\d])(?![.,_]\d)", titulo, flags=re.IGNORECASE) is not None


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
        resultado = self._garantir_paginas_do_lab(pergunta, resultado)
        return self._incluir_vizinhos_de_codigo(resultado)

    def _garantir_paginas_do_lab(self, pergunta: str, resultado: list[Hit]) -> list[Hit]:
        """O curso repete números de lab (ex.: "Lab 3" existe em IoT e em GenAI). Se a pergunta cita
        "lab N" sem dizer o assunto, o início de TODA página com esse número entra no contexto, para o
        modelo poder avisar da ambiguidade. Apenas ACRESCENTA: nunca tira um resultado que a busca achou."""
        numeros = labs_citados(pergunta)
        if not numeros:
            return resultado
        primeiro_por_pagina: dict[str, Hit] = {}
        for cid in sorted(self.chunks):                       # menor id = começo da página
            h = self.chunks[cid]
            if any(titulo_e_do_lab(h.titulo, n) for n in numeros):
                primeiro_por_pagina.setdefault(h.url, h)
        presentes = {h.url for h in resultado}
        for url, base in filtrar_por_assunto(pergunta, primeiro_por_pagina).items():
            if url not in presentes:
                resultado.append(Hit(**{**base.__dict__, "score": 0.0}))
                presentes.add(url)
        return resultado

    def _incluir_vizinhos_de_codigo(self, resultado: list[Hit], topo: int = 3, maximo: int = 3) -> list[Hit]:
        """Código longo é dividido em vários trechos (ex.: a constante `MQTT_PORT = 1883` num e o uso dela
        no seguinte). Se um dos melhores resultados é código, trazemos os vizinhos imediatos da mesma
        seção: sem eles o modelo vê o uso da variável mas não o valor."""
        presentes = {h.id for h in resultado}
        saida: list[Hit] = []
        adicionados = 0

        def vizinho(h: Hit, delta: int):
            v = self.chunks.get(h.id + delta)
            if v and v.id not in presentes and v.url == h.url and v.secao == h.secao:   # seção vazia = abertura da página
                presentes.add(v.id)
                return Hit(**{**v.__dict__, "score": 0.0})
            return None

        for pos, h in enumerate(resultado):
            antes = depois = None
            if pos < topo and adicionados < maximo and eh_codigo(h.texto):
                antes = vizinho(h, -1)
                adicionados += antes is not None
                if adicionados < maximo:
                    depois = vizinho(h, +1)
                    adicionados += depois is not None
            saida.extend(x for x in (antes, h, depois) if x is not None)
        return saida
        primeiro_por_pagina: dict[str, Hit] = {}
        for cid in sorted(self.chunks):                       # menor id = começo da página
            h = self.chunks[cid]
            if any(titulo_e_do_lab(h.titulo, n) for n in numeros):
                primeiro_por_pagina.setdefault(h.url, h)
        protegidas = set(primeiro_por_pagina)
        presentes = {h.url for h in resultado}
        for url, base in primeiro_por_pagina.items():
            if url in presentes:
                continue
            novo = Hit(**{**base.__dict__, "score": 0.0})
            if len(resultado) < k:
                resultado.append(novo)
            else:  # troca o pior resultado que não seja de uma das páginas do lab
                idx = next((i for i in range(len(resultado) - 1, -1, -1) if resultado[i].url not in protegidas), None)
                if idx is None:
                    break
                resultado[idx] = novo
            presentes.add(url)
        return resultado

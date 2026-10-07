"""Etapa 3 da ingestão: gerar o arquivo `data/index.db` (a "base de conhecimento").

Como rodar (na raiz do projeto):

    python -m ingest.build_index                      # baixa o repositório da disciplina do GitHub
    python -m ingest.build_index --local-dir ../DisruptiveArchitectures-master
    python -m ingest.build_index --sem-embeddings     # só texto (busca por palavra-chave), sem chave de API

O que o index.db contém:
    chunks      -> cada trecho + de onde veio + seu resumo + seu vetor (embedding)
    chunks_fts  -> o mesmo texto no formato do FTS5 (busca por palavras, ranking BM25)
    meta        -> modelo de embedding, dimensão, quantidade de chunks...

Cache retomável: cada resumo e cada vetor é guardado em `data/build_cache.json` assim que
é gerado. Se a cota gratuita do Gemini acabar no meio (erro 429), é só esperar alguns
minutos e rodar de novo: o build continua de onde parou, sem gastar cota repetida.
Apague o arquivo de cache para forçar tudo do zero (ex.: depois de mudar o prompt de resumo).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np

from app.config import get_settings
from ingest.chunker import chunkar
from ingest.sources import Documento, carregar_documentos, obter_repositorio

SCHEMA = """
CREATE TABLE meta (chave TEXT PRIMARY KEY, valor TEXT);
CREATE TABLE chunks (
    id        INTEGER PRIMARY KEY,
    fonte     TEXT NOT NULL,      -- arquivo de origem (ex.: aulas/genAI/lab4/lab4.md)
    titulo    TEXT NOT NULL,      -- título da página no menu do site
    trilha    TEXT NOT NULL,      -- seções do menu (ex.: "2º Semestre - IA > Laboratórios de GenAI")
    secao     TEXT NOT NULL,      -- título interno da página
    url       TEXT NOT NULL,      -- link público para citar
    tipo      TEXT NOT NULL,      -- pagina | notebook
    texto     TEXT NOT NULL,
    resumo    TEXT NOT NULL DEFAULT '',  -- descrição gerada por LLM (ajuda a achar código e tabelas)
    embedding BLOB                -- float32 normalizado (pode ser NULL se --sem-embeddings)
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    titulo, secao, texto, resumo,
    tokenize = 'unicode61 remove_diacritics 2'   -- "avaliação" casa com "avaliacao"
);
"""

LIMITE_CODIGO = 0.6   # trecho com mais de 60% de código recebe resumo
FATIA_EMBEDDINGS = 10  # quantos vetores geramos entre cada gravação do cache


def _apelido_lab(titulo: str) -> str:
    """'Lab04 - MLP' também deve ser achado por quem digita 'lab 4'. Acrescentamos esse apelido."""
    achados = re.findall(r"\bLab\s*0*(\d+(?:[.,]\d+)?)", titulo, flags=re.IGNORECASE)
    return " ".join(f"lab {n}" for n in achados)


def contexto_do_chunk(doc: Documento, secao: str) -> str:
    """Cabeçalho que acompanha o chunk na busca: onde ele está no site."""
    partes = [*doc.trilha, doc.titulo] + ([secao] if secao else [])
    return " > ".join(partes)


def precisa_resumo(texto: str) -> bool:
    """Só vale gastar uma chamada de LLM em trechos que a pergunta em português dificilmente
    "casa" sozinha: os dominados por código ou que contêm tabela. Prosa já é achada normalmente."""
    codigo = sum(len(b.split()) for b in re.findall(r"```.*?```", texto, flags=re.DOTALL))
    tabela = re.search(r"^\|.*\|\s*\n\|[\s:|-]+\|", texto, flags=re.MULTILINE) is not None
    return tabela or codigo / max(len(texto.split()), 1) >= LIMITE_CODIGO


# ------------------------------------------------------------------ cache retomável

def _chave(*partes: str) -> str:
    return hashlib.sha256("\x1f".join(partes).encode("utf-8")).hexdigest()


class _Cache:
    """Dicionário em disco: chave -> resumo (str) ou vetor (base64). Sem caminho, não persiste."""

    def __init__(self, caminho: str | None):
        self.caminho, self.dados, self._novos = caminho, {}, 0
        if caminho and Path(caminho).exists():
            try:
                self.dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.dados = {}   # cache corrompido: recomeça

    def get(self, chave: str):
        return self.dados.get(chave)

    def set(self, chave: str, valor: str) -> None:
        self.dados[chave] = valor
        self._novos += 1

    def salvar(self) -> None:
        if not self.caminho or not self._novos:
            return
        Path(self.caminho).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{self.caminho}.tmp"
        Path(tmp).write_text(json.dumps(self.dados), encoding="utf-8")
        os.replace(tmp, self.caminho)
        self._novos = 0


def _codificar(v: np.ndarray) -> str:
    return base64.b64encode(v.astype(np.float32).tobytes()).decode("ascii")


def _decodificar(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float32)


# ------------------------------------------------------------------ construção do índice

def construir_indice(documentos: list[Documento], saida: str, embedder=None,
                     embedding_model: str = "", resumidor=None, cache_path: str | None = None,
                     filtro_resumo=precisa_resumo) -> dict:
    """Cria o index.db.

    `embedder`: objeto com `embed_documentos(pares, lote=..., progresso=...)`.
    `resumidor`: função (contexto, texto) -> descrição curta. Opcional; falhas viram resumo vazio
                 (e NÃO entram no cache, para serem tentadas de novo na próxima execução).
    `cache_path`: arquivo JSON que permite retomar o build após uma falha de cota.
    """
    linhas = [(doc, ch) for doc in documentos for ch in chunkar(doc.texto)]
    cache = _Cache(cache_path)

    # --- resumos (só para código/tabelas) ---
    resumos = [""] * len(linhas)
    alvo = [i for i, (_, c) in enumerate(linhas) if resumidor is not None and filtro_resumo(c.texto)]
    do_cache = falhas = 0
    for n, i in enumerate(alvo, start=1):
        d, c = linhas[i]
        contexto = contexto_do_chunk(d, c.secao)
        chave = _chave("resumo", contexto, c.texto)
        if (guardado := cache.get(chave)) is not None:
            resumos[i], do_cache = guardado, do_cache + 1
        else:
            try:
                resumos[i] = resumidor(contexto, c.texto).strip()
                cache.set(chave, resumos[i])
                cache.salvar()
            except Exception as e:  # um resumo que falha não derruba a indexação
                falhas += 1
                print(f"\n  [aviso] sem resumo para o chunk {i + 1}: {str(e)[:90]}")
        print(f"  resumos: {n}/{len(alvo)}", end="\r")
    if alvo:
        print(f"\n  resumos: {len(alvo)} chunks de código/tabela ({do_cache} do cache, {falhas} falharam)")

    # --- embeddings (o resumo entra no texto que gera o vetor, mas NÃO no texto mostrado ao aluno) ---
    vetores = None
    if embedder is not None and linhas:
        pares = [(contexto_do_chunk(d, c.secao), f"{resumos[i]}\n\n{c.texto}".strip())
                 for i, (d, c) in enumerate(linhas)]
        chaves = [_chave("emb", embedding_model, t, x) for t, x in pares]
        lista: list = [None] * len(linhas)
        faltando = []
        for i, chave in enumerate(chaves):
            guardado = cache.get(chave)
            if guardado is not None:
                lista[i] = _decodificar(guardado)
            else:
                faltando.append(i)
        print(f"  embeddings: {len(linhas) - len(faltando)} do cache, {len(faltando)} a gerar")
        for ini in range(0, len(faltando), FATIA_EMBEDDINGS):
            idxs = faltando[ini:ini + FATIA_EMBEDDINGS]
            novos = embedder.embed_documentos([pares[i] for i in idxs], lote=FATIA_EMBEDDINGS)
            for i, v in zip(idxs, novos):
                lista[i] = v
                cache.set(chaves[i], _codificar(v))
            cache.salvar()   # progresso protegido: se a cota acabar agora, nada se perde
            print(f"  embeddings: {len(linhas) - len(faltando) + ini + len(idxs)}/{len(linhas)}", end="\r")
        print()
        vetores = np.vstack(lista)

    # --- gravação do SQLite (arquivo temporário + troca atômica) ---
    Path(saida).parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=Path(saida).parent, suffix=".tmp")
    tmp.close()
    con = sqlite3.connect(tmp.name)
    con.executescript(SCHEMA)
    for i, (doc, ch) in enumerate(linhas, start=1):
        emb = vetores[i - 1].astype(np.float32).tobytes() if vetores is not None else None
        con.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?)",
            (i, doc.fonte, doc.titulo, " > ".join(doc.trilha), ch.secao, doc.url, doc.tipo, ch.texto,
             resumos[i - 1], emb),
        )
        titulo_fts = f"{contexto_do_chunk(doc, '').replace(' > ', ' ')} {_apelido_lab(doc.titulo)}"
        con.execute("INSERT INTO chunks_fts(rowid, titulo, secao, texto, resumo) VALUES (?,?,?,?,?)",
                    (i, titulo_fts, ch.secao, ch.texto, resumos[i - 1]))

    dim = int(vetores.shape[1]) if vetores is not None else 0
    meta = {"embedding_model": embedding_model if vetores is not None else "",
            "dimensao": str(dim), "n_chunks": str(len(linhas)), "n_documentos": str(len(documentos))}
    con.executemany("INSERT INTO meta VALUES (?,?)", meta.items())
    con.commit()
    con.close()
    os.replace(tmp.name, saida)  # troca atômica: nunca deixa um index.db pela metade
    return {"chunks": len(linhas), "documentos": len(documentos), "dimensao": dim,
            "sem_resumo": sum(1 for i in alvo if not resumos[i])}


def main() -> int:
    ap = argparse.ArgumentParser(description="Gera data/index.db a partir do site da disciplina.")
    ap.add_argument("--local-dir", help="pasta do repositório da disciplina (senão baixa do GitHub)")
    ap.add_argument("--out", default=get_settings().index_db_path)
    ap.add_argument("--cache", default="data/build_cache.json", help="arquivo de progresso retomável")
    ap.add_argument("--sem-cache", action="store_true")
    ap.add_argument("--sem-embeddings", action="store_true", help="não chama a API (só busca por palavras)")
    ap.add_argument("--sem-notebooks", action="store_true")
    ap.add_argument("--sem-resumos", action="store_true", help="não gera resumos por LLM")
    ap.add_argument("--pausa-resumo", type=float, default=4.0, help="segundos entre resumos (limite por minuto)")
    ap.add_argument("--pausa-embedding", type=float, default=1.0, help="segundos entre embeddings")
    args = ap.parse_args()

    print("1/3 Obtendo o material da disciplina...")
    raiz = obter_repositorio(args.local_dir)
    docs = carregar_documentos(raiz, incluir_notebooks=not args.sem_notebooks)
    print(f"    {len(docs)} documentos ({sum(d.tipo == 'notebook' for d in docs)} notebooks)")

    embedder = resumidor = None
    settings = get_settings()
    if not args.sem_embeddings:
        from app.llm import GeminiClient, LLMError
        from app.prompts import SYSTEM_RESUMO, prompt_resumo

        embedder = GeminiClient(settings)
        embedder.esperas_cota = (10, 30, 65, 65)      # atravessa a janela de 1 minuto do plano gratuito
        embedder.pausa_embedding = args.pausa_embedding
        if not args.sem_resumos:
            def resumidor(contexto, texto):
                time.sleep(args.pausa_resumo)
                return embedder.gerar(prompt_resumo(contexto, texto), SYSTEM_RESUMO,
                                      modelo=settings.rewrite_model, temperatura=0.0)

    print("2/3 Quebrando em chunks, resumindo código/tabelas (LLM) e gerando embeddings...")
    try:
        info = construir_indice(docs, args.out, embedder, settings.embedding_model, resumidor,
                                None if args.sem_cache else args.cache)
    except Exception as e:
        if type(e).__name__ != "LLMError":
            raise
        print(f"\nO build parou: {e}\n"
              "O progresso já está salvo. Espere alguns minutos (a cota do plano gratuito é por minuto/dia) "
              "e rode o MESMO comando de novo: ele continua de onde parou.")
        return 1

    print(f"3/3 Pronto: {info['chunks']} chunks, dimensão {info['dimensao']} -> {args.out}")
    if info["sem_resumo"]:
        print(f"    Atenção: {info['sem_resumo']} chunks de código ficaram sem resumo (cota). "
              "Rode de novo para completar; os já feitos vêm do cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

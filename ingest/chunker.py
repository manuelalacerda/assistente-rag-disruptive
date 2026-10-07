"""Etapa 2 da ingestão: quebrar cada documento em "chunks" (trechos) para a busca.

Por que isso importa: o chunk é a unidade que o RAG recupera. Se ele for cortado
no meio de uma explicação, ou perder o código, a resposta final sai pior — não
importa quão bom seja o LLM. Por isso este chunker:

- quebra preferencialmente nas fronteiras de títulos (##, ###);
- NUNCA corta um bloco de código ao meio (exceto blocos gigantes);
- guarda o "caminho" do título (ex.: "Embeddings > Trechos e similaridade"),
  que depois serve de contexto para a busca e para a citação;
- repete o último parágrafo no começo do chunk seguinte (sobreposição), para não
  perder a ligação entre ideias vizinhas;
- evita chunks minúsculos e títulos "órfãos" no fim de um chunk.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass

MAX_PALAVRAS = 260      # tamanho alvo máximo de um chunk
MIN_PALAVRAS = 90       # só fechamos um chunk numa fronteira de título se ele já tem isso
OVERLAP_MAX = 60        # tamanho máximo do parágrafo repetido no chunk seguinte
LIMITE_BLOCO = 1.6      # blocos maiores que MAX * LIMITE_BLOCO são divididos

_TAGS_HTML = r"br|span|div|p|b|i|u|em|strong|small|sub|sup|center|img|a|details|summary|font|hr|mark|iframe|video|source|table|thead|tbody|tr|td|th"


@dataclass
class Chunk:
    secao: str   # caminho de títulos, ex.: "Arquitetura do RAG"
    texto: str


@dataclass
class _Bloco:
    tipo: str          # "titulo" | "codigo" | "texto"
    texto: str
    trilha: str = ""   # caminho de títulos onde o bloco está


def _palavras(texto: str) -> int:
    return len(texto.split())


# ------------------------------------------------------------------ limpeza do markdown

def _limpar_texto(trecho: str) -> str:
    """Limpa só trechos que NÃO são código (nunca mexemos dentro de ``` ```)."""
    trecho = re.sub(r"<!--.*?-->", "", trecho, flags=re.DOTALL)           # comentários HTML
    trecho = re.sub(r"!\[[^\]]*\]\([^)]*\)(\{[^}]*\})?", "", trecho)       # imagens
    trecho = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)",                  # link externo: mantém a URL
                    lambda m: m.group(1) if m.group(1).startswith("http") else f"{m.group(1)} ({m.group(2)})",
                    trecho)
    trecho = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", trecho)               # link interno: só o texto
    trecho = re.sub(r"\{\s*[.#:][^}\n]*\}", "", trecho)                     # atributos {: .md-button }
    trecho = re.sub(r"<br\s*/?>", " ", trecho, flags=re.IGNORECASE)
    trecho = re.sub(rf"</?(?:{_TAGS_HTML})\b[^>]*>", "", trecho, flags=re.IGNORECASE)
    return trecho


def _converter_admonicoes(linhas: list[str]) -> list[str]:
    """`!!! note "Título"` vira `Nota: Título` e o conteúdo indentado é desindentado."""
    saida, dentro = [], False
    for linha in linhas:
        m = re.match(r"^\s*(?:!!!|\?\?\?\+?)\s+(\w+)(?:\s+\"([^\"]*)\")?\s*$", linha)
        t = re.match(r"^\s*===\s+\"([^\"]*)\"\s*$", linha)
        if m:
            tipo, titulo = m.group(1), m.group(2) or ""
            saida.append(f"{tipo.capitalize()}: {titulo}".strip().rstrip(":") if titulo else f"{tipo.capitalize()}:")
            dentro = True
        elif t:
            saida.append(f"Aba: {t.group(1)}")
            dentro = True
        elif dentro and linha.startswith("    "):
            saida.append(linha[4:])
        elif dentro and not linha.strip():
            saida.append("")
        else:
            dentro = False
            saida.append(linha)
    return saida


# ------------------------------------------------------------------ do texto aos blocos

_CERCA = re.compile(r"(?ms)^([ \t]*```[^\n]*\n.*?^[ \t]*```[ \t]*)$")


def _blocos(texto: str) -> list[_Bloco]:
    texto = re.sub(r"\A---\n.*?\n---\s*", "", texto, flags=re.DOTALL)  # front-matter yaml
    segmentos = _CERCA.split(texto)  # índices ímpares = blocos de código

    pilha: list[tuple[int, str, bool]] = []   # (nível, título, visível)
    primeiro_titulo = True
    blocos: list[_Bloco] = []

    def trilha() -> str:
        return " > ".join(t for _, t, visivel in pilha if visivel)

    for i, seg in enumerate(segmentos):
        if i % 2 == 1:  # bloco de código
            codigo = textwrap.dedent(seg).strip("\n")
            if codigo.strip():
                blocos.append(_Bloco("codigo", codigo, trilha()))
            continue

        linhas = _converter_admonicoes(_limpar_texto(seg).split("\n"))
        paragrafo: list[str] = []

        def fechar_paragrafo():
            if paragrafo:
                texto_p = "\n".join(paragrafo).strip()
                if texto_p:
                    blocos.append(_Bloco("texto", texto_p, trilha()))
                paragrafo.clear()

        for linha in linhas:
            h = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", linha)
            if h:
                fechar_paragrafo()
                nivel, titulo = len(h.group(1)), h.group(2).strip()
                while pilha and pilha[-1][0] >= nivel:
                    pilha.pop()
                # o primeiro título do documento é o título da página: já vem no nome da fonte
                pilha.append((nivel, titulo, not primeiro_titulo))
                primeiro_titulo = False
                blocos.append(_Bloco("titulo", f"{'#' * nivel} {titulo}", trilha()))
            elif not linha.strip() or re.match(r"^\s*([-*_])\1{2,}\s*$", linha):  # vazio ou régua ---
                fechar_paragrafo()
            else:
                paragrafo.append(linha.rstrip())
        fechar_paragrafo()
    return blocos


def _dividir_bloco(b: _Bloco, max_palavras: int) -> list[_Bloco]:
    """Divide blocos gigantes (código longo, tabelas, parágrafos enormes)."""
    if _palavras(b.texto) <= max_palavras * LIMITE_BLOCO or b.tipo == "titulo":
        return [b]

    if b.tipo == "codigo":
        linhas = b.texto.split("\n")
        abre, fecha = linhas[0], linhas[-1]
        corpo = linhas[1:-1]
        partes, atual, n = [], [], 0
        for linha in corpo:
            if n + _palavras(linha) > max_palavras and atual:
                partes.append(atual)
                atual, n = [], 0
            atual.append(linha)
            n += _palavras(linha)
        if atual:
            partes.append(atual)
        return [_Bloco("codigo", "\n".join([abre, *p, fecha]), b.trilha) for p in partes]

    unidades = b.texto.split("\n") if "\n" in b.texto else re.split(r"(?<=[.!?])\s+", b.texto)
    partes, atual, n = [], [], 0
    for u in unidades:
        if n + _palavras(u) > max_palavras and atual:
            partes.append("\n".join(atual))
            atual, n = [], 0
        atual.append(u)
        n += _palavras(u)
    if atual:
        partes.append("\n".join(atual))
    return [_Bloco(b.tipo, p, b.trilha) for p in partes]


# ------------------------------------------------------------------ empacotamento

def chunkar(texto: str, max_palavras: int = MAX_PALAVRAS, min_palavras: int = MIN_PALAVRAS) -> list[Chunk]:
    blocos: list[_Bloco] = []
    for b in _blocos(texto):
        blocos.extend(_dividir_bloco(b, max_palavras))

    chunks: list[list[_Bloco]] = []
    atual: list[_Bloco] = []
    n = 0
    base = 0   # quantos blocos do início de `atual` foram só "emprestados" (títulos órfãos / sobreposição)

    def fechar(proximo_eh_titulo: bool) -> None:
        """Fecha o chunk atual e prepara o início do próximo (títulos órfãos + sobreposição)."""
        nonlocal atual, n, base
        orfaos: list[_Bloco] = []
        while atual and atual[-1].tipo == "titulo":
            orfaos.insert(0, atual.pop())
        if atual:
            chunks.append(atual)
        inicio = list(orfaos)
        if not proximo_eh_titulo and not orfaos and atual and atual[-1].tipo == "texto" \
                and _palavras(atual[-1].texto) <= OVERLAP_MAX:
            inicio = [atual[-1]]  # sobreposição: repete o último parágrafo
        atual = inicio
        base = len(atual)
        n = sum(_palavras(b.texto) for b in atual)

    for b in blocos:
        w = _palavras(b.texto)
        eh_fronteira = b.tipo == "titulo" and len(b.texto.split()[0]) <= 3  # #, ## ou ###
        # só fecha se já existe conteúdo NOVO no chunk (evita chunk feito só de sobreposição)
        if len(atual) > base and (n + w > max_palavras or (eh_fronteira and n >= min_palavras)):
            fechar(proximo_eh_titulo=eh_fronteira)
        atual.append(b)
        n += w
    if len(atual) > base or not chunks:
        chunks.append(atual)

    # junta uma "sobra" muito pequena no chunk anterior
    if len(chunks) >= 2 and sum(_palavras(b.texto) for b in chunks[-1]) < min_palavras // 2:
        total = sum(_palavras(b.texto) for b in chunks[-2] + chunks[-1])
        if total <= max_palavras * 1.3:
            chunks[-2] = chunks[-2] + chunks[-1]
            chunks.pop()

    resultado = []
    for grupo in chunks:
        corpo = "\n\n".join(b.texto for b in grupo).strip()
        sem_titulos = " ".join(b.texto for b in grupo if b.tipo != "titulo")
        if _palavras(sem_titulos) >= 8:
            primeiro_conteudo = next((b for b in grupo if b.tipo != "titulo"), grupo[0])
            resultado.append(Chunk(secao=primeiro_conteudo.trilha, texto=corpo))
    return resultado

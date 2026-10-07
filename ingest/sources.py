"""Etapa 1 da ingestão: descobrir QUAIS documentos entram na base e COMO citá-los.

Decisões importantes (todas pensadas para a qualidade da resposta):

1. Usamos o `nav` do mkdocs.yml como "lista oficial" do que está publicado no site.
   Assim o chat não responde com base em material antigo que ficou esquecido na pasta.
2. O título que aparece na citação é o do menu do site (ex.: "Lab4 - RAG e Bases de
   Conhecimento"), e não o nome do arquivo.
3. Os notebooks (.ipynb) ao lado de cada página são incluídos: nos labs de GenAI é
   ali que mora o código de verdade. Eles são citados com o link da página do lab.
"""

from __future__ import annotations

import io
import json
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import requests
import yaml

REPO_ZIP_PADRAO = "https://codeload.github.com/arnaldojr/DisruptiveArchitectures/zip/refs/heads/master"


@dataclass
class Documento:
    fonte: str                 # caminho relativo dentro de material/ (ex.: aulas/genAI/lab4/lab4.md)
    titulo: str                # título "de menu" da página
    trilha: list[str]          # seções do menu até a página (ex.: ["2º Semestre - IA", "Laboratórios de GenAI"])
    url: str                   # URL pública da página no site
    tipo: str                  # "pagina" ou "notebook"
    texto: str = field(repr=False, default="")


# ---------------------------------------------------------------- obter o repositório

def obter_repositorio(local_dir: str | None = None, zip_url: str = REPO_ZIP_PADRAO) -> Path:
    """Devolve a pasta raiz do repositório da disciplina (a que contém mkdocs.yml).

    - `local_dir`: usa um clone/descompactação que você já tem.
    - senão: baixa o .zip da branch master no GitHub (sempre a versão mais nova).
    """
    if local_dir:
        raiz = Path(local_dir).resolve()
    else:
        destino = Path(tempfile.mkdtemp(prefix="disruptive_"))
        resposta = requests.get(zip_url, timeout=60)
        resposta.raise_for_status()
        zipfile.ZipFile(io.BytesIO(resposta.content)).extractall(destino)
        raiz = next(p for p in destino.iterdir() if p.is_dir())

    if not (raiz / "mkdocs.yml").exists():
        # aceita também a pasta que contém o repositório dentro dela
        filhos = [p for p in raiz.iterdir() if p.is_dir() and (p / "mkdocs.yml").exists()]
        if len(filhos) == 1:
            return filhos[0]
        raise FileNotFoundError(f"mkdocs.yml não encontrado em {raiz}")
    return raiz


# ---------------------------------------------------------------- mkdocs.yml

class _Loader(yaml.SafeLoader):
    """O mkdocs.yml usa tags do tipo `!!python/name:...` que o SafeLoader recusa. Ignoramos."""


_Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)


def _percorrer_nav(no, trilha: list[str], saida: list[tuple[str | None, str, list[str]]]) -> None:
    """Achata o nav em tuplas (título_do_menu, caminho.md, trilha)."""
    if isinstance(no, list):
        for item in no:
            _percorrer_nav(item, trilha, saida)
    elif isinstance(no, dict):
        for nome, valor in no.items():
            if isinstance(valor, str):
                saida.append((nome, valor, trilha))
            else:  # seção com filhos
                _percorrer_nav(valor, trilha + [nome], saida)
    elif isinstance(no, str):
        saida.append((None, no, trilha))


def ler_nav(raiz: Path) -> tuple[str, list[tuple[str | None, str, list[str]]]]:
    cfg = yaml.load((raiz / "mkdocs.yml").read_text(encoding="utf-8"), Loader=_Loader)
    saida: list = []
    _percorrer_nav(cfg.get("nav", []), [], saida)
    site_url = (cfg.get("site_url") or "").rstrip("/")
    return site_url, [t for t in saida if t[1].endswith(".md")]


# ---------------------------------------------------------------- URL pública (igual ao mkdocs)

def caminho_para_url(caminho_md: str, site_url: str) -> str:
    partes = caminho_md.split("/")
    if partes[-1] == "index.md":
        partes = partes[:-1]
    else:
        partes[-1] = partes[-1].removesuffix(".md")
    return f"{site_url}/{'/'.join(partes)}/" if partes else f"{site_url}/"


# ---------------------------------------------------------------- notebooks

def notebook_para_markdown(caminho: Path) -> str:
    """Converte um .ipynb em markdown: células de texto viram texto, de código viram ```python.

    Saídas (outputs) são ignoradas: costumam ser longas, repetitivas e cheias de ruído.
    """
    nb = json.loads(caminho.read_text(encoding="utf-8"))
    blocos = []
    for celula in nb.get("cells", []):
        fonte = celula.get("source", "")
        fonte = "".join(fonte) if isinstance(fonte, list) else fonte
        fonte = fonte.strip()
        if not fonte:
            continue
        if celula.get("cell_type") == "markdown":
            blocos.append(fonte)
        elif celula.get("cell_type") == "code":
            blocos.append(f"```python\n{fonte}\n```")
    return "\n\n".join(blocos)


# ---------------------------------------------------------------- montagem dos documentos

def _titulo_do_arquivo(texto: str, caminho: str) -> str:
    achou = re.search(r"^#{1,2}\s+(.+)$", texto, flags=re.MULTILINE)
    if achou:
        return achou.group(1).strip()
    p = Path(caminho)
    return p.parent.name if p.stem == "index" else p.stem


def carregar_documentos(raiz: Path, incluir_notebooks: bool = True) -> list[Documento]:
    site_url, itens = ler_nav(raiz)
    material = raiz / "material"
    documentos: list[Documento] = []
    vistos: set[str] = set()
    notebooks_vistos: set[Path] = set()

    for titulo_menu, caminho, trilha in itens:
        arquivo = material / caminho
        if caminho in vistos:
            continue
        vistos.add(caminho)
        if not arquivo.exists():
            print(f"  [aviso] listado no nav mas não existe: {caminho}")
            continue

        texto = arquivo.read_text(encoding="utf-8")
        titulo = titulo_menu or _titulo_do_arquivo(texto, caminho)
        url = caminho_para_url(caminho, site_url)
        documentos.append(Documento(caminho, titulo, trilha, url, "pagina", texto))

        if incluir_notebooks:
            for nb in sorted(arquivo.parent.glob("*.ipynb")):
                if nb in notebooks_vistos:  # pasta compartilhada por 2 páginas: indexa uma vez só
                    continue
                notebooks_vistos.add(nb)
                texto_nb = notebook_para_markdown(nb)
                if texto_nb:
                    rel = str(nb.relative_to(material))
                    documentos.append(
                        Documento(rel, f"{titulo} (notebook)", trilha, url, "notebook", texto_nb)
                    )
    return documentos

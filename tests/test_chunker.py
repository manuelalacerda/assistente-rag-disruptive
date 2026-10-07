from ingest.chunker import MAX_PALAVRAS, chunkar


def _fences_balanceadas(texto: str) -> bool:
    return sum(1 for l in texto.split("\n") if l.strip().startswith("```")) % 2 == 0


def _doc_longo() -> str:
    paragrafo = " ".join(["palavra"] * 70)
    partes = []
    for i in range(6):
        partes.append(f"## Seção {i}\n\n{paragrafo}\n\n{paragrafo}\n")
    return "# Título da página\n\n" + "\n".join(partes)


def test_documento_vazio_nao_gera_chunks():
    assert chunkar("") == []
    assert chunkar("---\ntitulo: x\n---\n") == []


def test_chunks_respeitam_tamanho_e_nao_terminam_em_titulo():
    chunks = chunkar(_doc_longo())
    assert len(chunks) > 3
    for c in chunks:
        assert len(c.texto.split()) <= MAX_PALAVRAS * 1.6
        ultima = c.texto.strip().split("\n")[-1]
        assert not ultima.startswith("#"), "título órfão no fim do chunk"


def test_secao_registra_caminho_de_titulos():
    chunks = chunkar("# Página\n\n## Teoria\n\n### Embeddings\n\nUm embedding é um vetor numérico que representa significado.")
    assert chunks[0].secao == "Teoria > Embeddings"  # o 1º título (da página) não entra no caminho


def test_codigo_nunca_e_cortado_ao_meio():
    codigo = "\n".join(f"x{i} = {i}  # linha de código número {i}" for i in range(60))
    doc = f"## Exemplo\n\nAntes do código.\n\n```python\n{codigo}\n```\n\nDepois do código."
    for c in chunkar(doc):
        assert _fences_balanceadas(c.texto)
    assert any("x0 = 0" in c.texto for c in chunkar(doc))


def test_bloco_de_codigo_gigante_e_dividido_em_pedacos_validos():
    codigo = "\n".join(f"valor_{i} = calcular({i}, {i + 1}, {i + 2})" for i in range(400))
    chunks = chunkar(f"## Código\n\n```python\n{codigo}\n```")
    assert len(chunks) > 1
    assert all(_fences_balanceadas(c.texto) for c in chunks)
    assert all(c.texto.count("```python") == 1 for c in chunks)


def test_limpeza_preserva_includes_cpp_e_remove_ruido():
    doc = (
        "## Bibliotecas\n\n"
        "### `#include <WiFi.h>`\n\n"
        "<!-- comentário que não deve aparecer -->\n\n"
        "![figura](img.png)\n\n"
        "Veja o [Google AI Studio](https://aistudio.google.com/) e a [agenda](../agenda.md).{ .md-button }\n\n"
        "!!! note \"Atenção\"\n\n    O schema não garante verdade.\n"
    )
    texto = chunkar(doc)[0].texto
    assert "<WiFi.h>" in texto
    assert "comentário" not in texto and "img.png" not in texto and "md-button" not in texto
    assert "https://aistudio.google.com/" in texto      # link externo mantém a URL
    assert "../agenda.md" not in texto                  # link interno só o texto
    assert "Note: Atenção" in texto and "O schema não garante verdade." in texto


def test_sobreposicao_repete_ultimo_paragrafo_curto():
    p = " ".join(["alfa"] * 100)
    curto = "Este parágrafo curto liga as ideias vizinhas."
    doc = f"## S\n\n{p}\n\n{p}\n\n{curto}\n\n{p}\n\n{p}"
    chunks = chunkar(doc)
    assert sum(curto in c.texto for c in chunks) >= 2

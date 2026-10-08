import sqlite3

import pytest

from app.retriever import Retriever, consulta_fts
from ingest.build_index import construir_indice
from ingest.sources import caminho_para_url, carregar_documentos, ler_nav, notebook_para_markdown

SITE = "https://exemplo.github.io/curso"


# ------------------------------------------------------------------ fontes

def test_urls_seguem_a_regra_do_mkdocs():
    assert caminho_para_url("aulas/iot/mqtt/index.md", SITE) == f"{SITE}/aulas/iot/mqtt/"
    assert caminho_para_url("aulas/genAI/lab4/lab4.md", SITE) == f"{SITE}/aulas/genAI/lab4/lab4/"
    assert caminho_para_url("index.md", SITE) == f"{SITE}/"


def test_nav_le_titulos_trilhas_e_ignora_tags_python(repo_disciplina):
    site, itens = ler_nav(repo_disciplina)
    assert site == SITE
    por_caminho = {c: (t, trilha) for t, c, trilha in itens}
    assert por_caminho["aulas/iot/mqtt/index.md"] == ("Lab09 - MQTT", ["1º Semestre - IoT", "ESP32"])
    assert por_caminho["aulas/iot/index.md"][0] is None            # item sem título no menu
    assert "aulas/antiga/obsoleta.md" not in por_caminho


def test_notebook_vira_markdown_sem_outputs(repo_disciplina):
    nb = repo_disciplina / "material/aulas/genAI/lab4/lab4_rag.ipynb"
    md = notebook_para_markdown(nb)
    assert "## Embeddings" in md and "```python\nresultado = client" in md
    assert "ruido" not in md


def test_somente_paginas_do_menu_entram_e_notebook_cita_a_pagina(repo_disciplina):
    docs = carregar_documentos(repo_disciplina)
    fontes = {d.fonte for d in docs}
    assert "aulas/antiga/obsoleta.md" not in fontes
    nb = next(d for d in docs if d.tipo == "notebook")
    assert nb.url == f"{SITE}/aulas/genAI/lab4/lab4/"
    assert nb.titulo == "Lab4 - RAG (notebook)"
    assert next(d for d in docs if d.fonte == "aulas/iot/index.md").titulo == "IoT"  # veio do 1º título


# ------------------------------------------------------------------ índice + busca

@pytest.fixture
def indice(tmp_path, repo_disciplina, fake_llm):
    caminho = str(tmp_path / "idx.db")
    construir_indice(carregar_documentos(repo_disciplina), caminho, fake_llm, "fake-embed")
    return caminho


def test_indice_tem_chunks_vetores_e_meta(indice):
    con = sqlite3.connect(indice)
    assert con.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0] > 3
    assert dict(con.execute("SELECT chave, valor FROM meta"))["embedding_model"] == "fake-embed"


def test_busca_hibrida_encontra_a_pagina_certa(indice, fake_llm):
    r = Retriever(indice, fake_llm.embed_consulta)
    assert r.tem_vetores
    hits = r.buscar("qual porta do broker MQTT?", k=3)
    assert hits[0].titulo == "Lab09 - MQTT"
    assert hits[0].posicao_vetorial and hits[0].posicao_palavras   # as duas buscas concordaram
    assert "1883" in hits[0].texto or any("1883" in h.texto for h in hits)


def test_busca_so_por_palavras_quando_nao_ha_vetores(tmp_path, repo_disciplina):
    caminho = str(tmp_path / "sem_vetores.db")
    construir_indice(carregar_documentos(repo_disciplina), caminho, embedder=None)
    r = Retriever(caminho)
    assert not r.tem_vetores
    assert r.buscar("o que são chunks e embeddings?", k=2)[0].url.endswith("lab4/lab4/")


def test_se_a_api_de_embeddings_cair_a_busca_por_palavras_continua(indice, fake_llm):
    fake_llm.falhar_embedding = True
    r = Retriever(indice, fake_llm.embed_consulta)
    hits = r.buscar("MQTT broker", k=2)
    assert hits and hits[0].similaridade is None


def test_apelido_lab_permite_buscar_por_lab_numero(indice):
    r = Retriever(indice)  # só palavras
    assert r.buscar("lab 4", k=3)[0].titulo.startswith("Lab4")


def test_consulta_fts_e_segura_com_caracteres_especiais():
    q = consulta_fts('Como uso "aspas", (parênteses), * e AND/OR no MQTT?')
    assert "mqtt" in q and "(" not in q and "*" not in q
    assert consulta_fts("o que é de?") == ""      # só stopwords


def test_limite_de_trechos_por_pagina(tmp_path, repo_disciplina, fake_llm):
    # página única e longa: nenhum resultado pode passar do limite de trechos da mesma URL
    (repo_disciplina / "material/aulas/iot/mqtt/index.md").write_text(
        "\n\n".join(f"## Parte {i}\n\nmqtt broker tópico {' '.join(['detalhe'] * 120)}" for i in range(8)),
        encoding="utf-8")
    caminho = str(tmp_path / "longo.db")
    construir_indice(carregar_documentos(repo_disciplina), caminho, fake_llm)
    hits = Retriever(caminho, fake_llm.embed_consulta).buscar("mqtt broker tópico", k=8)
    from collections import Counter
    from app.retriever import MAX_POR_PAGINA
    assert max(Counter(h.url for h in hits).values()) <= MAX_POR_PAGINA


def test_resumo_gerado_na_indexacao_ajuda_a_achar_o_trecho_de_codigo(tmp_path, repo_disciplina, fake_llm):
    """O trecho de código não diz 'porta' em português; o resumo diz, e a busca por palavras passa a achá-lo."""
    def resumidor(contexto, texto):
        return "Configura o endereço do servidor e a porta de conexão do ESP32." if "MQTT_PORT" in texto else ""

    caminho = str(tmp_path / "com_resumo.db")
    construir_indice(carregar_documentos(repo_disciplina), caminho, fake_llm, "fake", resumidor,
                     filtro_resumo=lambda t: True)
    con = sqlite3.connect(caminho)
    assert con.execute("SELECT COUNT(*) FROM chunks WHERE resumo != ''").fetchone()[0] == 1
    hits = Retriever(caminho).buscar("qual a porta de conexão?", k=3)   # 'porta' só existe no resumo
    assert "MQTT_PORT" in hits[0].texto
    assert "porta de conexão" not in hits[0].texto                      # o resumo não aparece para o aluno


def test_falha_no_resumo_nao_derruba_a_indexacao(tmp_path, repo_disciplina, fake_llm):
    def resumidor(contexto, texto):
        raise RuntimeError("429")
    info = construir_indice(carregar_documentos(repo_disciplina), str(tmp_path / "x.db"), fake_llm, "fake", resumidor,
                          filtro_resumo=lambda t: True)
    assert info["chunks"] > 0


# ------------------------------------------------------------------ resumos seletivos e cache retomável

def test_so_codigo_e_tabela_recebem_resumo():
    from ingest.build_index import precisa_resumo
    codigo = "```cpp\n" + "\n".join(f"int x{i} = {i};" for i in range(30)) + "\n```"
    assert precisa_resumo(codigo)
    assert precisa_resumo("Preços:\n\n| Serviço | Valor |\n|---|---|\n| Banho | R$ 60,00 |")
    assert not precisa_resumo("Texto explicativo " * 40 + "\n\n```py\nx = 1\n```")   # prosa com um exemplo pequeno


def test_build_retoma_do_cache_sem_repetir_chamadas(tmp_path, repo_disciplina, fake_llm):
    chamadas = {"resumo": 0, "emb": 0}
    original = fake_llm.embed_documentos

    def conta_emb(pares, lote=50, progresso=None):
        chamadas["emb"] += len(pares)
        return original(pares, lote, progresso)
    fake_llm.embed_documentos = conta_emb

    def resumidor(ctx, texto):
        chamadas["resumo"] += 1
        return "resumo"

    docs, cache = carregar_documentos(repo_disciplina), str(tmp_path / "cache.json")
    kw = dict(embedder=fake_llm, embedding_model="fake", resumidor=resumidor, cache_path=cache,
              filtro_resumo=lambda t: True)
    construir_indice(docs, str(tmp_path / "a.db"), **kw)
    primeira = dict(chamadas)
    assert primeira["resumo"] > 0 and primeira["emb"] > 0

    construir_indice(docs, str(tmp_path / "b.db"), **kw)    # 2ª execução: tudo vem do cache
    assert chamadas == primeira


def test_build_interrompido_por_cota_continua_de_onde_parou(tmp_path, repo_disciplina, fake_llm):
    from app.llm import LLMError
    docs, cache = carregar_documentos(repo_disciplina), str(tmp_path / "cache.json")
    gerados = []
    original = fake_llm.embed_documentos

    def instavel(pares, lote=50, progresso=None):
        if len(gerados) >= 3:                       # "a cota acabou" depois de alguns vetores
            raise LLMError("429")
        gerados.extend(pares)
        return original(pares, lote, progresso)
    fake_llm.embed_documentos = instavel

    import ingest.build_index as bi
    bi.FATIA_EMBEDDINGS_ORIGINAL, bi.FATIA_EMBEDDINGS = bi.FATIA_EMBEDDINGS, 3
    try:
        with pytest.raises(LLMError):
            construir_indice(docs, str(tmp_path / "x.db"), fake_llm, "fake", cache_path=cache)
        feitos = len(gerados)
        assert feitos == 3 and not (tmp_path / "x.db").exists()      # não deixou índice pela metade

        fake_llm.embed_documentos = original
        restantes = []
        fake_llm.embed_documentos = lambda p, lote=50, progresso=None: (restantes.extend(p), original(p, lote, progresso))[1]
        info = construir_indice(docs, str(tmp_path / "x.db"), fake_llm, "fake", cache_path=cache)
        assert len(restantes) == info["chunks"] - feitos             # só gerou o que faltava
    finally:
        bi.FATIA_EMBEDDINGS = bi.FATIA_EMBEDDINGS_ORIGINAL


# ------------------------------------------------------------------ perguntas sintéticas

def test_gera_perguntas_sinteticas_e_retoma_sem_repetir(tmp_path, repo_disciplina, fake_llm):
    from eval.gerar_perguntas import gerar
    from eval.run_eval import avaliar_sintetico

    caminho = str(tmp_path / "idx.db")
    longo = " ".join(["mqtt broker topico publicacao assinatura mensagem"] * 12)
    (repo_disciplina / "material/aulas/iot/mqtt/index.md").write_text(f"## MQTT\n\n{longo}", encoding="utf-8")
    construir_indice(carregar_documentos(repo_disciplina), caminho, fake_llm, "fake")

    # o "LLM" devolve a própria palavra-chave do trecho, para o teste ser determinístico
    fake_llm.resposta = "Como funciona o broker mqtt de publicacao e assinatura?"
    saida = str(tmp_path / "sint.jsonl")
    n1 = gerar(caminho, fake_llm, n=5, saida=saida, pausa=0)
    assert n1 >= 1
    assert gerar(caminho, fake_llm, n=5, saida=saida, pausa=0) == 0      # 2ª vez: nada novo

    r = Retriever(caminho, fake_llm.embed_consulta)
    m = avaliar_sintetico(r, saida, k=6, candidatos=25)
    assert m["n"] == n1 and m["trecho_hit6"] == 1.0 and m["pagina_hit6"] == 1.0


# ------------------------------------------------------------------ "lab N" existe em mais de uma página

def test_labs_citados_e_titulos():
    from app.retriever import labs_citados, titulo_e_do_lab
    assert labs_citados("Do que trata o Lab 3?") == ["3"]
    assert labs_citados("e o LAB03?") == ["3"] and labs_citados("lab 3.5 e lab 4") == ["3.5", "4"]
    assert labs_citados("explique MQTT") == []
    assert titulo_e_do_lab("Lab03 - Serial", "3") and titulo_e_do_lab("Lab3 - Ferramentas (notebook)", "3")
    assert not titulo_e_do_lab("Lab3.5 - Do protótipo ao produto", "3")      # outro lab
    assert not titulo_e_do_lab("Lab13 - Algo", "3") and not titulo_e_do_lab("Lab30 - Algo", "3")


def test_pergunta_sobre_lab_traz_todas_as_paginas_com_aquele_numero(tmp_path, fake_llm):
    from ingest.sources import Documento
    longo = " ".join(["ferramentas funcoes schema pydantic saidas estruturadas lab"] * 30)
    docs = [
        Documento("a.md", "Lab3 - Ferramentas e Saídas Estruturadas", ["IA"], "https://x/genai/lab3/", "pagina",
                  "\n\n".join(f"## Parte {i}\n\n{longo}" for i in range(6))),
        Documento("b.md", "Lab03 - Serial", ["IoT"], "https://x/iot/lab3/", "pagina",
                  "## Comunicação serial\n\nO Arduino envia dados ao computador pela porta USB usando o monitor."),
        Documento("c.md", "Lab3.5 - Do protótipo ao produto", ["IA"], "https://x/genai/lab3_5/", "pagina",
                  "## Produto\n\nTransforme o assistente em uma API com banco de dados e README."),
    ]
    caminho = str(tmp_path / "labs.db")
    construir_indice(docs, caminho, fake_llm, "fake")
    r = Retriever(caminho, fake_llm.embed_consulta)

    urls = [h.url for h in r.buscar("Do que trata o Lab 3?", k=4, candidatos=25)]
    assert "https://x/genai/lab3/" in urls and "https://x/iot/lab3/" in urls     # as DUAS páginas do lab 3
    assert len(urls) == 4

    sem_lab = [h.url for h in r.buscar("ferramentas funcoes schema pydantic", k=4, candidatos=25)]
    assert "https://x/iot/lab3/" not in sem_lab        # sem citar "lab N", nada é forçado


# ------------------------------------------------------------------ vizinhos de código e assunto

def _doc_codigo_dividido():
    """Código longo que vira 2 trechos: o valor (1883) fica no 1º; o uso da variável, no 2º."""
    parte1 = "\n".join(f"const int CONFIG_{i} = {i};  // parametro numero {i} da placa" for i in range(25))
    parte1 += "\nconst int PORTA_BROKER = 1883;\n" + "\n".join(f"const int EXTRA_{i} = {i};" for i in range(25))
    parte2 = "\n".join(f"int calculo_{i}(int x) {{ return x + {i}; }}  // funcao auxiliar {i}" for i in range(25))
    parte2 += "\nclient.setServer(BROKER, PORTA_BROKER);\n"
    return "## Código-base\n\n```cpp\n" + parte1 + "\n" + parte2 + "```"


def test_codigo_dividido_traz_o_trecho_vizinho_com_o_valor(tmp_path, fake_llm):
    from ingest.chunker import chunkar
    from ingest.sources import Documento
    texto = _doc_codigo_dividido()
    assert len(chunkar(texto)) >= 2                        # confirma o cenário: o código foi dividido
    caminho = str(tmp_path / "cod.db")
    construir_indice([Documento("m.md", "Lab09 - MQTT", ["IoT"], "https://x/mqtt/", "pagina", texto)], caminho, fake_llm, "fake")
    r = Retriever(caminho)                                  # só palavras

    hits = r.buscar("setServer", k=1)                       # a busca acha SÓ o trecho do uso da variável...
    assert any("1883" in h.texto for h in hits)             # ...e o vizinho com o valor vem junto
    assert len(hits) <= 1 + 3                               # e o aumento do contexto é limitado


def test_vizinho_nao_vem_de_outra_secao_nem_para_texto_comum(tmp_path, fake_llm):
    from ingest.sources import Documento
    prosa = " ".join(["explicacao detalhada sobre mqtt e topicos"] * 40)
    docs = [Documento("m.md", "Lab09 - MQTT", ["IoT"], "https://x/mqtt/", "pagina",
                      f"## Teoria\n\n{prosa}\n\n## Prática\n\n{prosa} praticaunica")]
    caminho = str(tmp_path / "prosa.db")
    construir_indice(docs, caminho, fake_llm, "fake")
    assert len(Retriever(caminho).buscar("praticaunica", k=1)) == 1       # prosa: sem expansão


def test_assunto_desambigua_o_lab_repetido():
    from app.retriever import Hit, filtrar_por_assunto
    mqtt = Hit(1, "Lab09 - MQTT", "1º Semestre - IoT > ESP32", "", "u1", "pagina", "t")
    nodered = Hit(2, "Lab09 - Fluxos e Dashboards", "1º Semestre - IoT > Node-RED", "", "u2", "pagina", "t")
    pags = {"u1": mqtt, "u2": nodered}
    assert list(filtrar_por_assunto("qual a porta do MQTT no lab 9?", pags)) == ["u1"]
    assert list(filtrar_por_assunto("lab 9 do node-red", pags)) == ["u2"]
    assert list(filtrar_por_assunto("Do que trata o Lab 9?", pags)) == ["u1", "u2"]        # sem assunto: continua ambíguo
    assert list(filtrar_por_assunto("lab 9 na parte de IoT", pags)) == ["u1", "u2"]         # 'IoT' está nas duas: não distingue


def test_garantia_de_labs_nunca_expulsa_resultados_da_busca(tmp_path, fake_llm):
    from ingest.sources import Documento
    docs = [Documento("a.md", "Lab3 - Ferramentas", ["IA"], "https://x/ia/lab3/", "pagina",
                      "\n\n".join(f"## P{i}\n\n" + " ".join(["funcoes schema pydantic"] * 60) for i in range(4))),
            Documento("b.md", "Lab03 - Serial", ["IoT"], "https://x/iot/lab3/", "pagina",
                      "## Serial\n\nO Arduino envia dados ao computador pelo monitor serial.")]
    caminho = str(tmp_path / "g.db")
    construir_indice(docs, caminho, fake_llm, "fake")
    r = Retriever(caminho)
    sem_lab = {h.id for h in r.buscar("funcoes schema pydantic", k=3)}
    com_lab = {h.id for h in r.buscar("funcoes schema pydantic lab 3", k=3)}
    assert sem_lab <= com_lab                               # tudo que a busca achou continua lá
    assert any(h.url == "https://x/iot/lab3/" for h in r.buscar("funcoes schema pydantic lab 3", k=3))

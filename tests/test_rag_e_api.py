from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.db import ChatDB
from app.llm import LLMError
from app.main import create_app
from app.prompts import SYSTEM_PROMPT, SYSTEM_REESCRITA
from app.rag import FORA_DO_ESCOPO, NAO_ENCONTREI, RagService, extrair_citacoes
from app.retriever import Retriever
from ingest.build_index import construir_indice
from ingest.sources import carregar_documentos
from tests.conftest import FakeLLM


@pytest.fixture
def rag(settings, repo_disciplina, fake_llm):
    construir_indice(carregar_documentos(repo_disciplina), settings.index_db_path, fake_llm)
    return RagService(Retriever(settings.index_db_path, fake_llm.embed_consulta), fake_llm, settings)


# ------------------------------------------------------------------ citações

def test_extrair_citacoes_ordem_limites_e_codigo():
    texto = "MQTT usa broker [2]. Também [1, 3] e [9]. Em código `lista[1]` e ```\nx[3]\n``` não contam. De novo [2]."
    assert extrair_citacoes(texto, maximo=3) == [2, 1, 3]


# ------------------------------------------------------------------ serviço RAG

def test_resposta_traz_apenas_as_fontes_citadas(rag, fake_llm):
    fake_llm.resposta = "O broker recebe e distribui as mensagens [1]."
    res = rag.responder("O que faz o broker MQTT?")
    assert [f.n for f in res.fontes] == [1]
    assert res.fontes[0].url == res.hits[0].url
    prompt = fake_llm.chamadas[-1]["prompt"]
    assert "<contexto>" in prompt and "[1]" in prompt and "Pergunta do aluno" in prompt
    assert fake_llm.chamadas[-1]["system"] == SYSTEM_PROMPT


@pytest.mark.parametrize("resposta", [f"{NAO_ENCONTREI} Procure no Lab 4.", f"{FORA_DO_ESCOPO} Pergunte sobre os labs."])
def test_recusa_do_modelo_nao_mostra_fontes(rag, fake_llm, resposta):
    fake_llm.resposta = resposta
    assert rag.responder("Qual a capital da França?").fontes == []


def test_recusa_nao_deixa_marcadores_de_citacao_soltos(rag, fake_llm):
    fake_llm.resposta = f"{NAO_ENCONTREI} O material trata dos checkpoints [3][6], [8] mas não da média."
    res = rag.responder("Qual a média mínima?")
    assert res.fontes == [] and "[" not in res.resposta
    assert "checkpoints, mas não da média" in res.resposta.replace(" ,", ",")


def test_sem_historico_nao_reescreve_e_com_historico_reescreve(rag, fake_llm):
    rag.responder("O que é MQTT?")
    assert all(c["system"] != SYSTEM_REESCRITA for c in fake_llm.chamadas)

    historico = [{"role": "user", "content": "Fale do MQTT"}, {"role": "assistant", "content": "É um protocolo."}]
    res = rag.responder("e a porta?", historico)
    assert res.consulta == "consulta reescrita sobre MQTT"
    reescrita = next(c for c in fake_llm.chamadas if c["system"] == SYSTEM_REESCRITA)
    assert reescrita["modelo"] == rag.s.rewrite_model
    assert "conversa_anterior" in fake_llm.chamadas[-1]["prompt"]


def test_falha_na_reescrita_usa_a_pergunta_original(rag, fake_llm):
    original = fake_llm.gerar
    def gerar(prompt, system, modelo=None, temperatura=None):
        if system == SYSTEM_REESCRITA:
            raise LLMError("429")
        return original(prompt, system, modelo, temperatura)
    fake_llm.gerar = gerar
    res = rag.responder("e a porta?", [{"role": "user", "content": "MQTT"}])
    assert res.consulta == "e a porta?"


def test_sem_trechos_nao_chama_o_llm(rag, fake_llm):
    rag.retriever.matriz = None  # simula índice sem vetores
    fake_llm.chamadas.clear()
    res = rag.responder("de?")
    assert res.resposta == NAO_ENCONTREI and res.fontes == [] and not fake_llm.chamadas


def test_portao_de_similaridade_minima(rag, fake_llm):
    rag.s = replace(rag.s, min_similarity=0.99)
    res = rag.responder("O que é MQTT?")
    assert res.resposta == NAO_ENCONTREI and not fake_llm.chamadas


# ------------------------------------------------------------------ API

@pytest.fixture
def cliente(settings, rag):
    app = create_app(rag=rag, db=ChatDB(settings.chat_db_path), settings=settings)
    with TestClient(app) as c:
        yield c


def test_health_e_pagina_inicial(cliente):
    h = cliente.get("/health").json()
    assert h["status"] == "ok" and h["chunks"] > 0 and h["busca_vetorial"] is True
    r = cliente.get("/")
    assert r.status_code == 200 and "Assistente da disciplina" in r.text


def test_chat_devolve_resposta_fontes_e_mantem_sessao(cliente, rag, fake_llm):
    fake_llm.resposta = "MQTT usa um broker [1]."
    r1 = cliente.post("/api/chat", json={"pergunta": "O que é MQTT?"})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["resposta"].startswith("MQTT usa") and d1["fontes"][0]["n"] == 1 and d1["session_id"]

    r2 = cliente.post("/api/chat", json={"pergunta": "e a porta?", "session_id": d1["session_id"]})
    assert r2.status_code == 200
    # na 2ª pergunta o histórico da 1ª foi enviado ao RAG (gerou a etapa de reescrita)
    assert any(c["system"] == SYSTEM_REESCRITA for c in fake_llm.chamadas)
    assert cliente.get("/api/stats").json()["perguntas"] == 2


@pytest.mark.parametrize("corpo", [{}, {"pergunta": "   "}, {"pergunta": "x" * 501}])
def test_validacao_rejeita_pergunta_invalida(cliente, corpo):
    assert cliente.post("/api/chat", json=corpo).status_code == 422


def test_feedback_so_para_respostas_existentes(cliente):
    d = cliente.post("/api/chat", json={"pergunta": "O que é MQTT?"}).json()
    assert cliente.post("/api/feedback", json={"message_id": d["message_id"], "nota": "util"}).status_code == 200
    assert cliente.post("/api/feedback", json={"message_id": 9999, "nota": "util"}).status_code == 404
    assert cliente.post("/api/feedback", json={"message_id": d["message_id"], "nota": "talvez"}).status_code == 422
    assert cliente.get("/api/stats").json()["feedback_util"] == 1


def test_limite_de_taxa_por_ip(settings, rag):
    app = create_app(rag=rag, db=ChatDB(settings.chat_db_path), settings=replace(settings, rate_limit_per_min=2))
    with TestClient(app) as c:
        codigos = [c.post("/api/chat", json={"pergunta": "MQTT?"}).status_code for _ in range(3)]
    assert codigos == [200, 200, 429]


def test_erro_do_llm_vira_502_amigavel(cliente, fake_llm):
    def quebra(*a, **k):
        raise LLMError("cota")
    fake_llm.gerar = quebra
    r = cliente.post("/api/chat", json={"pergunta": "O que é MQTT?"})
    assert r.status_code == 502 and "indisponível" in r.json()["detail"]


def test_app_sobe_degradado_sem_indice(settings, monkeypatch):
    monkeypatch.setattr("app.main._construir_rag", lambda s: (_ for _ in ()).throw(FileNotFoundError("sem índice")))
    with TestClient(create_app(settings=settings)) as c:
        assert c.get("/health").json()["status"] == "degradado"
        assert c.post("/api/chat", json={"pergunta": "oi"}).status_code == 503


# ------------------------------------------------------------------ embeddings em lote

def test_embedding_cai_para_um_por_vez_quando_o_modelo_agrega_a_lista():
    """Reproduz o bug real: o modelo devolve 1 vetor para uma lista de N textos."""
    from types import SimpleNamespace
    from app.config import Settings
    from app.llm import GeminiClient

    class ClienteFalso:
        class models:
            @staticmethod
            def embed_content(model, contents, config=None):
                n = 1 if len(contents) > 1 else len(contents)   # agrega listas em um único vetor
                return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 2.0, 3.0])] * n)

    g = GeminiClient.__new__(GeminiClient)
    g.s, g._types, g._client = Settings(), SimpleNamespace(EmbedContentConfig=None), ClienteFalso()
    g.esperas_cota, g.pausa_embedding = (3, 8), 0
    vetores = g.embed_documentos([("t", f"texto {i}") for i in range(7)], lote=5)
    assert vetores.shape == (7, 3)


# ------------------------------------------------------------------ retentativas

def test_retentativas_esperam_o_tempo_configurado_e_nao_repetem_erro_permanente(monkeypatch):
    from app import llm
    esperas = []
    monkeypatch.setattr(llm.time, "sleep", esperas.append)

    estado = {"n": 0}
    def instavel():
        estado["n"] += 1
        if estado["n"] < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "ok"
    assert llm._com_retentativas(instavel, esperas_cota=(10, 65, 65)) == "ok"
    assert esperas == [10, 65]

    esperas.clear()
    def permanente():
        raise RuntimeError("404 model not found")
    with pytest.raises(LLMError):
        llm._com_retentativas(permanente)
    assert esperas == []     # erro permanente: falha na hora, sem esperar


# ------------------------------------------------------------------ cota diária

MSG_COTA_DIARIA = ("429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generate_content_free_tier_requests, limit: 20, "
                   "model: gemini-3.5-flash\nPlease retry in 1h12m38s. quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")


def test_cota_diaria_falha_na_hora_sem_esperar(monkeypatch):
    from app import llm
    esperas = []
    monkeypatch.setattr(llm.time, "sleep", esperas.append)

    def estourou():
        raise RuntimeError(MSG_COTA_DIARIA)
    with pytest.raises(llm.LLMQuotaDiaria) as e:
        llm._com_retentativas(estourou, esperas_cota=(10, 30, 65))
    assert esperas == [] and "gemini-3.5-flash" in str(e.value) and "1h12m38s" in str(e.value)


def test_modelo_de_reserva_assume_quando_a_cota_diaria_acaba():
    from dataclasses import replace
    from types import SimpleNamespace
    from app.config import Settings
    from app.llm import GeminiClient

    usados = []

    class ClienteFalso:
        class models:
            @staticmethod
            def generate_content(model, contents, config=None):
                usados.append(model)
                if model == "principal":
                    raise RuntimeError(MSG_COTA_DIARIA)
                return SimpleNamespace(text="resposta da reserva")

    g = GeminiClient.__new__(GeminiClient)
    g.s = replace(Settings(), generation_model="principal", fallback_model="reserva")
    g._types = SimpleNamespace(GenerateContentConfig=lambda **k: None)
    g._client, g.esperas_cota = ClienteFalso(), (3, 8)
    assert g.gerar("oi", "sys") == "resposta da reserva"
    assert usados == ["principal", "reserva"]

    g.s = replace(g.s, fallback_model="")               # sem reserva configurada: erro claro
    with pytest.raises(LLMError):
        g.gerar("oi", "sys")


def test_api_avisa_limite_diario_com_mensagem_clara(cliente, fake_llm):
    from app.llm import LLMQuotaDiaria
    def estourou(*a, **k):
        raise LLMQuotaDiaria("acabou")
    fake_llm.gerar = estourou
    r = cliente.post("/api/chat", json={"pergunta": "O que é MQTT?"})
    assert r.status_code == 503 and "limite diário" in r.json()["detail"]


def test_avaliacao_para_limpo_quando_a_cota_diaria_acaba(monkeypatch):
    from app.llm import LLMQuotaDiaria
    from eval.run_eval import avaliar_respostas
    from app.rag import Resultado

    class RagFalso:
        n = 0
        def responder(self, pergunta, historico=None):
            self.n += 1
            if self.n == 3:
                raise LLMQuotaDiaria("acabou")
            return Resultado(NAO_ENCONTREI, [], pergunta)

    golden = [{"id": f"q{i}", "categoria": "x", "pergunta": "?", "recusa": True} for i in range(5)]
    r = avaliar_respostas(RagFalso(), golden, usar_juiz=False, pausa=0)
    assert r["total"] == 2 and r["interrompido"] and r["aprovadas"] == 2


# ------------------------------------------------------------------ certificados do sistema (TLS)

@pytest.mark.parametrize("valor, plataforma, esperado", [
    ("auto", "win32", True), ("auto", "darwin", True), ("auto", "linux", False),   # Linux = servidor: não muda
    ("1", "linux", True), ("0", "win32", False), ("nao", "win32", False),
])
def test_certificados_do_sistema_ligam_so_onde_faz_sentido(monkeypatch, valor, plataforma, esperado):
    from app import tls
    monkeypatch.setenv("USE_SYSTEM_CERTS", valor)
    monkeypatch.setattr(tls.sys, "platform", plataforma)
    monkeypatch.setattr(tls, "load_dotenv", lambda: None)
    assert tls.ativo() is esperado


def test_cliente_gemini_recebe_contexto_ssl_quando_ativo(monkeypatch):
    pytest.importorskip("truststore")      # só roda se o pacote opcional estiver instalado
    import ssl
    from app import tls
    from app.config import Settings
    from app.llm import GeminiClient
    from dataclasses import replace

    monkeypatch.setenv("USE_SYSTEM_CERTS", "1")
    monkeypatch.setattr(tls, "load_dotenv", lambda: None)
    assert isinstance(tls.contexto(), ssl.SSLContext)      # truststore instalado no ambiente de teste
    cliente = GeminiClient(replace(Settings(), gemini_api_key="chave-falsa"))   # constrói sem chamar a rede
    assert cliente._client is not None


# ------------------------------------------------------------------ acabamento da resposta

@pytest.mark.parametrize("entrada, esperado", [
    ("porta `1883`` do broker", "porta `1883` do broker"),
    ("porta ``1883`` do broker", "porta `1883` do broker"),
    ("porta `1883` do broker", "porta `1883` do broker"),                       # já correto: não muda
    ("use `a` e ``b`` aqui", "use `a` e `b` aqui"),
    ("antes\n```python\nx = `y``\n```\ndepois `z``", "antes\n```python\nx = `y``\n```\ndepois `z`"),   # bloco intacto
])
def test_normaliza_codigo_inline_sem_tocar_nos_blocos(entrada, esperado):
    from app.rag import normalizar_codigo_inline
    assert normalizar_codigo_inline(entrada) == esperado


def test_resposta_final_chega_com_crases_corrigidas(rag, fake_llm):
    fake_llm.resposta = "A porta é `1883`` [1]."
    assert "`1883`" in rag.responder("porta do MQTT?").resposta and "``" not in rag.responder("porta do MQTT?").resposta


def test_instrucao_de_continuidade_so_aparece_com_historico(rag, fake_llm):
    rag.responder("O que é MQTT?")
    assert "CONTINUA a conversa anterior" not in fake_llm.chamadas[-1]["prompt"]
    rag.responder("e a porta?", [{"role": "user", "content": "MQTT"}, {"role": "assistant", "content": "ok"}])
    assert "CONTINUA a conversa anterior" in fake_llm.chamadas[-1]["prompt"]


def test_modelo_de_reserva_tambem_assume_em_falha_passageira_do_principal(monkeypatch):
    """Como na demo: o principal devolve 503 (sobrecarga) e o aluno não deve ver erro."""
    from dataclasses import replace
    from types import SimpleNamespace
    from app import llm
    from app.config import Settings

    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    usados = []

    class ClienteFalso:
        class models:
            @staticmethod
            def generate_content(model, contents, config=None):
                usados.append(model)
                if model == "principal":
                    raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand")
                return SimpleNamespace(text="resposta da reserva")

    g = llm.GeminiClient.__new__(llm.GeminiClient)
    g.s = replace(Settings(), generation_model="principal", fallback_model="reserva")
    g._types = SimpleNamespace(GenerateContentConfig=lambda **k: None)
    g._client, g.esperas_cota = ClienteFalso(), (3, 8)
    assert g.gerar("oi", "sys") == "resposta da reserva"
    assert usados.count("principal") == 3 and usados[-1] == "reserva"   # 3 tentativas no principal, depois a reserva

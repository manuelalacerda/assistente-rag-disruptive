"""A API HTTP: a "porta de entrada" do sistema (exigência do Lab 3.5).

Rotas:
    GET  /              -> a página do chat (app/static/index.html)
    POST /api/chat      -> recebe a pergunta, executa o RAG, devolve resposta + fontes
    POST /api/feedback  -> registra "útil" / "não ajudou" para uma resposta
    GET  /api/stats     -> contadores simples (perguntas, feedback)
    GET  /health        -> usado pelo servidor de deploy para saber se o app está de pé
    GET  /docs          -> documentação interativa gerada automaticamente

Rodar localmente:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import Settings, get_settings
from app.db import ChatDB
from app.llm import LLMError, LLMQuotaDiaria
from app.schemas import ChatRequest, ChatResponse, FeedbackRequest, FonteOut

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

PAGINA = Path(__file__).parent / "static" / "index.html"


class LimiteTaxa:
    """Janela deslizante em memória: no máximo `limite` chamadas por `janela` segundos por chave."""

    def __init__(self, limite: int, janela: float = 60.0):
        self.limite, self.janela = limite, janela
        self._chamadas: dict[str, deque] = defaultdict(deque)

    def permitido(self, chave: str) -> bool:
        agora = time.monotonic()
        fila = self._chamadas[chave]
        while fila and agora - fila[0] > self.janela:
            fila.popleft()
        if len(fila) >= self.limite:
            return False
        fila.append(agora)
        return True


def _construir_rag(settings: Settings):
    """Monta Retriever + Gemini + RagService. Separado para podermos injetar um falso nos testes."""
    from app.llm import GeminiClient
    from app.rag import RagService
    from app.retriever import Retriever

    if not Path(settings.index_db_path).exists():
        raise FileNotFoundError(
            f"Índice não encontrado em {settings.index_db_path}. Rode: python -m ingest.build_index"
        )
    llm = GeminiClient(settings)
    return RagService(Retriever(settings.index_db_path, llm.embed_consulta), llm, settings)


def create_app(rag=None, db: ChatDB | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.erro_init = None
        app.state.rag = rag
        if rag is None:
            try:
                app.state.rag = _construir_rag(settings)
                r = app.state.rag.retriever
                log.info("Índice carregado: %d chunks (vetores: %s)", r.n_chunks, r.tem_vetores)
            except Exception as e:  # o app sobe mesmo assim e /health explica o problema
                app.state.erro_init = str(e)
                log.error("Falha ao iniciar o RAG: %s", e)
        yield

    app = FastAPI(title="Assistente da disciplina Disruptive Architectures", version="1.0.0", lifespan=lifespan)
    app.state.db = db or ChatDB(settings.chat_db_path)
    limite_ip = LimiteTaxa(settings.rate_limit_per_min)
    limite_global = LimiteTaxa(settings.rate_limit_global_per_min)

    if settings.allowed_origins:  # só necessário se o chat for embutido em OUTRO site (ex.: o do mkdocs)
        app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins,
                           allow_methods=["POST", "GET"], allow_headers=["Content-Type"])

    @app.get("/", include_in_schema=False)
    def pagina():
        return FileResponse(PAGINA)

    @app.get("/health")
    def health():
        rag_ = app.state.rag
        if rag_ is None:
            return {"status": "degradado", "erro": app.state.erro_init}
        r = rag_.retriever
        return {"status": "ok", "chunks": r.n_chunks, "busca_vetorial": r.tem_vetores,
                "embedding_model": r.meta.get("embedding_model", ""), "modelo": settings.generation_model}

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, request: Request):
        ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip()
              or (request.client.host if request.client else "?"))
        if not (limite_ip.permitido(ip) and limite_global.permitido("global")):
            raise HTTPException(429, "Muitas perguntas em pouco tempo. Aguarde um minuto e tente de novo.")
        if len(req.pergunta) > settings.max_question_chars:
            raise HTTPException(422, f"Pergunta muito longa (máximo {settings.max_question_chars} caracteres).")
        if app.state.rag is None:
            raise HTTPException(503, "O assistente ainda não está pronto. Tente novamente em instantes.")

        inicio = time.perf_counter()
        session_id = req.session_id or uuid.uuid4().hex
        historico = app.state.db.historico(session_id, 2 * settings.history_turns)
        try:
            res = app.state.rag.responder(req.pergunta, historico)
        except LLMQuotaDiaria as e:
            log.error("Cota diária esgotada: %s", e)
            raise HTTPException(503, "O limite diário gratuito da IA foi atingido. Tente novamente mais tarde.")
        except LLMError as e:
            log.error("Erro do LLM: %s", e)
            raise HTTPException(502, "O serviço de IA está indisponível ou no limite de uso. Tente em instantes.")
        except Exception:
            log.exception("Erro inesperado ao responder")
            raise HTTPException(500, "Erro interno ao processar a pergunta.")

        latencia = int((time.perf_counter() - inicio) * 1000)
        fontes = [FonteOut(**f.__dict__) for f in res.fontes]
        app.state.db.salvar_mensagem(session_id, "user", req.pergunta)
        message_id = app.state.db.salvar_mensagem(
            session_id, "assistant", res.resposta,
            meta={"fontes": [f.model_dump() for f in fontes], "consulta": res.consulta,
                  "tempos_ms": res.tempos_ms, "modelo": res.modelo, "latencia_ms": latencia,
                  "recuperados": [{"url": h.url, "secao": h.secao, "score": round(h.score, 5),
                                   "similaridade": h.similaridade} for h in res.hits]},
        )
        log.info("pergunta=%r consulta=%r latencia=%dms fontes=%d", req.pergunta[:80], res.consulta[:80],
                 latencia, len(fontes))
        return ChatResponse(resposta=res.resposta, fontes=fontes, session_id=session_id,
                            message_id=message_id, latencia_ms=latencia)

    @app.post("/api/feedback")
    def feedback(req: FeedbackRequest):
        if not app.state.db.salvar_feedback(req.message_id, req.nota, req.comentario):
            raise HTTPException(404, "Resposta não encontrada.")
        return {"ok": True}

    @app.get("/api/stats")
    def stats():
        return app.state.db.estatisticas()

    return app


app = create_app()

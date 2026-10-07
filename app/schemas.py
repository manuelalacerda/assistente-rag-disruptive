"""Contratos da API (o que entra e o que sai), validados pelo Pydantic.

Por que existe: o FastAPI usa estas classes para rejeitar automaticamente pedidos
malformados (erro 422) e para gerar a documentação interativa em /docs.
Mesma ideia da "saída estruturada" do Lab 3, agora aplicada à própria API.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    pergunta: str = Field(..., description="Pergunta do aluno sobre o conteúdo da disciplina")
    session_id: Optional[str] = Field(None, max_length=64, description="Identifica a conversa; omita na primeira pergunta")

    @field_validator("pergunta")
    @classmethod
    def nao_vazia(cls, v: str) -> str:
        v = " ".join(v.split())  # tira espaços/quebras de linha repetidos
        if not v:
            raise ValueError("A pergunta não pode ser vazia.")
        return v


class FonteOut(BaseModel):
    n: int
    titulo: str
    trilha: str
    secao: str
    url: str
    tipo: str
    trecho: str


class ChatResponse(BaseModel):
    resposta: str
    fontes: list[FonteOut]
    session_id: str
    message_id: int
    latencia_ms: int


class FeedbackRequest(BaseModel):
    message_id: int
    nota: Literal["util", "nao_util"]
    comentario: Optional[str] = Field(None, max_length=500)

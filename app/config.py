"""Configurações do projeto, lidas de variáveis de ambiente.

Por que existe: o código nunca deve ter chave de API nem caminhos "chumbados".
Tudo que muda entre o seu computador e o servidor (chave, modelos, limites)
fica aqui, e é definido por variáveis de ambiente (ou pelo arquivo .env local).
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()  # lê o arquivo .env (se existir). No servidor, as variáveis já vêm prontas.


def _lista(valor: str) -> list[str]:
    return [v.strip() for v in valor.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    # --- Gemini (mesma stack usada nos labs da disciplina) ---
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    generation_model: str = field(default_factory=lambda: os.getenv("GENERATION_MODEL", "gemini-3.5-flash"))
    rewrite_model: str = field(default_factory=lambda: os.getenv("REWRITE_MODEL", "gemini-3.5-flash-lite"))
    # modelo usado quando a cota DIÁRIA do generation_model acaba (cada modelo tem cota própria). "" = desliga
    fallback_model: str = field(default_factory=lambda: os.getenv("FALLBACK_MODEL", "gemini-3.5-flash-lite"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "gemini-embedding-2"))
    # 0 = usar a dimensão padrão do modelo
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "0")))

    # --- Bancos SQLite ---
    index_db_path: str = field(default_factory=lambda: os.getenv("INDEX_DB_PATH", "data/index.db"))
    chat_db_path: str = field(default_factory=lambda: os.getenv("CHAT_DB_PATH", "data/chat.db"))

    # --- Qualidade da busca / resposta ---
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "8")))
    candidates: int = field(default_factory=lambda: int(os.getenv("CANDIDATES", "25")))
    # Similaridade mínima (cosseno) do melhor trecho. 0 = desligado.
    # Calibre com `python -m eval.run_eval` antes de ligar.
    min_similarity: float = field(default_factory=lambda: float(os.getenv("MIN_SIMILARITY", "0")))
    history_turns: int = field(default_factory=lambda: int(os.getenv("HISTORY_TURNS", "3")))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.2")))

    # --- Proteção da API pública ---
    max_question_chars: int = field(default_factory=lambda: int(os.getenv("MAX_QUESTION_CHARS", "500")))
    rate_limit_per_min: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MIN", "12")))
    # teto para o servidor inteiro (protege a cota da API mesmo se alguém forjar o IP)
    rate_limit_global_per_min: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_GLOBAL_PER_MIN", "120")))
    allowed_origins: list[str] = field(default_factory=lambda: _lista(os.getenv("ALLOWED_ORIGINS", "")))


def get_settings() -> Settings:
    return Settings()

"""Persistência das conversas em SQLite (exigência do Lab 3.5: banco de dados).

Guardamos:
- cada mensagem (do aluno e do assistente) — dá memória à conversa e serve de log;
- no `meta` da resposta: fontes usadas, consulta de busca, tempos e modelo — essencial
  para investigar respostas ruins depois;
- o feedback "útil / não ajudou" — a base para melhorar o sistema com dados reais.

Este banco é DIFERENTE do index.db (a base de conhecimento, somente leitura).
Misturar os dois seria um erro: ao reindexar o site, você perderia o histórico.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS mensagens (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    conteudo   TEXT NOT NULL,
    criado_em  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    meta       TEXT
);
CREATE INDEX IF NOT EXISTS idx_mensagens_sessao ON mensagens (session_id, id);

CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL REFERENCES mensagens (id),
    nota       TEXT NOT NULL CHECK (nota IN ('util', 'nao_util')),
    comentario TEXT,
    criado_em  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""


class ChatDB:
    def __init__(self, caminho: str):
        self.caminho = caminho
        if caminho != ":memory:":
            Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        with self._conexao() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def _conexao(self):
        # uma conexão por operação: simples e seguro com várias threads do FastAPI
        con = sqlite3.connect(self.caminho, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def salvar_mensagem(self, session_id: str, role: str, conteudo: str, meta: Optional[dict] = None) -> int:
        with self._conexao() as con:
            cur = con.execute(
                "INSERT INTO mensagens (session_id, role, conteudo, meta) VALUES (?, ?, ?, ?)",
                (session_id, role, conteudo, json.dumps(meta, ensure_ascii=False) if meta else None),
            )
            return cur.lastrowid

    def historico(self, session_id: str, max_mensagens: int = 6) -> list[dict]:
        with self._conexao() as con:
            linhas = con.execute(
                "SELECT role, conteudo FROM mensagens WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, max_mensagens),
            ).fetchall()
        return [{"role": r["role"], "content": r["conteudo"]} for r in reversed(linhas)]

    def salvar_feedback(self, message_id: int, nota: str, comentario: Optional[str]) -> bool:
        with self._conexao() as con:
            existe = con.execute(
                "SELECT 1 FROM mensagens WHERE id = ? AND role = 'assistant'", (message_id,)
            ).fetchone()
            if not existe:
                return False
            con.execute(
                "INSERT INTO feedback (message_id, nota, comentario) VALUES (?, ?, ?)",
                (message_id, nota, comentario),
            )
            return True

    def estatisticas(self) -> dict:
        with self._conexao() as con:
            perguntas = con.execute("SELECT COUNT(*) FROM mensagens WHERE role = 'user'").fetchone()[0]
            sessoes = con.execute("SELECT COUNT(DISTINCT session_id) FROM mensagens").fetchone()[0]
            notas = dict(con.execute("SELECT nota, COUNT(*) FROM feedback GROUP BY nota").fetchall())
        return {"perguntas": perguntas, "conversas": sessoes,
                "feedback_util": notas.get("util", 0), "feedback_nao_util": notas.get("nao_util", 0)}

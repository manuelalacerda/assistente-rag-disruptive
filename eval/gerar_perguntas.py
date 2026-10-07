"""Gera perguntas de teste AUTOMATICAMENTE, a partir dos próprios trechos do material.

Por que existe: o golden.jsonl foi escrito à mão por quem já conhecia o material, então seu
vocabulário é parecido com o dos textos (resultado otimista). Aqui o LLM lê um trecho sorteado
e escreve a pergunta que um aluno faria, SEM copiar as palavras do trecho. Sabemos qual trecho
responde cada pergunta, então medimos se a busca o encontra: `python -m eval.run_eval --sintetico`.

    python -m eval.gerar_perguntas              # 50 perguntas -> eval/sintetico.jsonl
    python -m eval.gerar_perguntas --n 80

É retomável: se a cota do Gemini acabar, rode de novo e ele continua de onde parou.
Limite honesto: a pergunta nasce do mesmo texto que a responde (há algum viés a favor da busca).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import time
from pathlib import Path

from app.config import get_settings

SYSTEM = "Você é um aluno universitário curioso. Responda apenas com a pergunta, sem aspas nem explicações."

PROMPT = """Leia o trecho do material de uma disciplina e escreva UMA pergunta que um aluno faria e que este trecho responde.

Regras:
- Use palavras do dia a dia; NÃO copie frases nem termos raros do trecho quando houver um jeito mais simples de dizer.
- A pergunta deve fazer sentido sozinha (mencione o assunto: tecnologia, lab ou tema).
- Não escreva "trecho", "texto" ou "material".
- Uma única frase, no máximo 25 palavras.

Página: {titulo} > {secao}

<trecho>
{texto}
</trecho>"""


def hash_texto(texto: str) -> str:
    """Identifica um trecho pelo conteúdo (continua valendo se o índice for reconstruído)."""
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()


def amostrar(db_path: str, n: int, seed: int) -> list[tuple]:
    con = sqlite3.connect(db_path)
    linhas = con.execute("SELECT id, url, titulo, secao, texto FROM chunks").fetchall()
    con.close()
    elegiveis = [r for r in linhas if len(r[4].split()) >= 40]   # trechos minúsculos não geram boa pergunta
    random.Random(seed).shuffle(elegiveis)
    return elegiveis[:n]


def gerar(db_path: str, llm, n: int = 50, saida: str = "eval/sintetico.jsonl", seed: int = 7,
          pausa: float = 4.0, modelo: str | None = None) -> int:
    arquivo = Path(saida)
    feitos = set()
    if arquivo.exists():
        feitos = {json.loads(l)["texto_hash"] for l in arquivo.read_text(encoding="utf-8").splitlines() if l.strip()}

    novos = 0
    for _, url, titulo, secao, texto in amostrar(db_path, n, seed):
        h = hash_texto(texto)
        if h in feitos:
            continue
        try:
            bruto = llm.gerar(PROMPT.format(titulo=titulo, secao=secao or "-", texto=texto[:2500]),
                              SYSTEM, modelo=modelo, temperatura=0.7)
        except Exception as e:
            print(f"  [aviso] pulei um trecho: {str(e)[:90]}")
            continue
        pergunta = bruto.strip().split("\n")[0].strip().strip('"“”')
        if len(pergunta.split()) < 4:
            continue
        with arquivo.open("a", encoding="utf-8") as f:   # grava a cada pergunta: nada se perde
            f.write(json.dumps({"pergunta": pergunta, "texto_hash": h, "url": url, "titulo": titulo,
                                "secao": secao}, ensure_ascii=False) + "\n")
        novos += 1
        print(f"  {len(feitos) + novos}/{n}  {pergunta}")
        time.sleep(pausa)
    return novos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="eval/sintetico.jsonl")
    ap.add_argument("--pausa", type=float, default=4.0, help="segundos entre chamadas (limite por minuto)")
    args = ap.parse_args()

    from app.llm import GeminiClient
    s = get_settings()
    llm = GeminiClient(s)
    llm.esperas_cota = (10, 30, 65, 65)
    try:
        novos = gerar(s.index_db_path, llm, args.n, args.out, args.seed, args.pausa, s.rewrite_model)
    except KeyboardInterrupt:
        print("\nInterrompido. Rode de novo para continuar.")
        return 1
    print(f"\n{novos} perguntas novas em {args.out}. Agora: python -m eval.run_eval --sintetico")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

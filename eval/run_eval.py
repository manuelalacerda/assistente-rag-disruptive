"""Avaliação da qualidade: transforma "acho que está bom" em números.

Duas etapas, da mais barata para a mais completa:

  python -m eval.run_eval
      Avalia SÓ A BUSCA. Para cada pergunta de eval/golden.jsonl, vê se a página certa
      aparece entre os k primeiros trechos. Mede hit@1, hit@3, hit@k e MRR.
      Não gasta geração; usa embeddings se houver GEMINI_API_KEY (senão, só palavras).

  python -m eval.run_eval --respostas [--juiz]
      Roda o sistema INTEIRO e confere cada resposta:
        - contém os fatos esperados (`deve_conter`)?
        - cita uma fonte da página certa?
        - recusou quando o assunto NÃO está no material (`recusa`)?
      Com --juiz, um segundo LLM dá nota de 1 a 5 para a FIDELIDADE ao contexto
      (a resposta só afirma o que está nos trechos?).

Use antes e depois de cada mudança (chunking, prompt, modelo, k) e compare os números.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from app.config import get_settings
from app.llm import LLMError, LLMQuotaDiaria
from app.rag import RagService, eh_recusa
from app.retriever import Retriever

PROMPT_JUIZ = """Você avalia a FIDELIDADE de uma resposta de um assistente a trechos de contexto.

<contexto>
{contexto}
</contexto>

Pergunta: {pergunta}
Resposta do assistente: {resposta}

Nota de 1 a 5:
5 = toda afirmação da resposta está sustentada pelo contexto;
3 = mistura afirmações sustentadas com algumas não verificáveis no contexto;
1 = a resposta contradiz ou inventa informação que não está no contexto.
Responda apenas em JSON: {{"nota": <1-5>, "motivo": "<uma frase>"}}"""


def carregar_golden(caminho: str) -> list[dict]:
    return [json.loads(l) for l in Path(caminho).read_text(encoding="utf-8").splitlines() if l.strip()]


def casa_url(url: str, padroes: list[str]) -> bool:
    return any(p in url for p in padroes)


# ------------------------------------------------------------------ etapa 1: busca

def avaliar_busca(retriever: Retriever, golden: list[dict], k: int, candidatos: int) -> dict:
    itens = [g for g in golden if g.get("url_contem") and not g.get("anteriores")]
    posicoes, falhas, falhas_fato = [], [], []
    com_fato = achou_fato = 0
    for g in itens:
        hits = retriever.buscar(g["pergunta"], k=k, candidatos=candidatos)
        pos = next((i for i, h in enumerate(hits, 1) if casa_url(h.url, g["url_contem"])), None)
        posicoes.append(pos)
        if pos is None:
            falhas.append(g)
        # métrica mais exigente: o FATO que responde à pergunta está em algum trecho recuperado?
        if g.get("texto_contem"):
            com_fato += 1
            if any(re.search(g["texto_contem"], h.texto, re.IGNORECASE) for h in hits):
                achou_fato += 1
            else:
                falhas_fato.append(g)

    n = len(itens)
    hit = lambda m: sum(1 for p in posicoes if p is not None and p <= m) / n
    mrr = sum(1 / p for p in posicoes if p) / n
    print(f"\nBUSCA ({n} perguntas, k={k}, vetores={'sim' if retriever.tem_vetores else 'não'})")
    print(f"  hit@1 = {hit(1):.0%}   hit@3 = {hit(3):.0%}   hit@{k} = {hit(k):.0%}   MRR = {mrr:.3f}")
    fato = achou_fato / com_fato if com_fato else 0.0
    print(f"  fato@{k} = {fato:.0%}  (o trecho com a informação exata chegou ao prompt; {com_fato} perguntas)")
    for g in falhas:
        print(f"  ✗ não achou a página certa: [{g['id']}] {g['pergunta']}  (esperado: {g['url_contem']})")
    for g in falhas_fato:
        print(f"  ✗ página certa, mas sem o fato: [{g['id']}] {g['pergunta']}  (procurado: {g['texto_contem']})")
    return {"n": n, "hit1": hit(1), "hit3": hit(3), f"hit{k}": hit(k), "mrr": mrr, f"fato{k}": fato}


# ------------------------------------------------------------------ busca com perguntas sintéticas

def avaliar_sintetico(retriever: Retriever, caminho: str, k: int, candidatos: int) -> dict:
    """Para cada pergunta gerada, confere se o trecho de ORIGEM (e a página dele) voltou na busca."""
    itens = [json.loads(l) for l in Path(caminho).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not itens:
        raise SystemExit(f"{caminho} está vazio. Rode: python -m eval.gerar_perguntas")
    pos_trecho, pos_pagina, falhas = [], [], []
    for it in itens:
        hits = retriever.buscar(it["pergunta"], k=k, candidatos=candidatos)
        pt = next((i for i, h in enumerate(hits, 1) if hashlib.sha1(h.texto.encode()).hexdigest() == it["texto_hash"]), None)
        pp = next((i for i, h in enumerate(hits, 1) if h.url == it["url"]), None)
        pos_trecho.append(pt)
        pos_pagina.append(pp)
        if pt is None:
            falhas.append(it)

    n = len(itens)
    hit = lambda pos, m: sum(1 for p in pos if p is not None and p <= m) / n
    mrr = sum(1 / p for p in pos_trecho if p) / n
    print(f"\nBUSCA COM PERGUNTAS SINTÉTICAS ({n} perguntas geradas pelo LLM, k={k})")
    print(f"  trecho exato:  hit@1 = {hit(pos_trecho, 1):.0%}   hit@3 = {hit(pos_trecho, 3):.0%}   hit@{k} = {hit(pos_trecho, k):.0%}   MRR = {mrr:.3f}")
    print(f"  página certa:  hit@1 = {hit(pos_pagina, 1):.0%}   hit@{k} = {hit(pos_pagina, k):.0%}")
    for it in falhas[:8]:
        print(f"  ✗ {it['pergunta']}\n      (origem: {it['titulo']} > {it['secao'] or '-'})")
    if len(falhas) > 8:
        print(f"  ... e mais {len(falhas) - 8} sem o trecho de origem")
    return {"n": n, "trecho_hit1": hit(pos_trecho, 1), "trecho_hit3": hit(pos_trecho, 3),
            f"trecho_hit{k}": hit(pos_trecho, k), "trecho_mrr": mrr,
            "pagina_hit1": hit(pos_pagina, 1), f"pagina_hit{k}": hit(pos_pagina, k)}


# ------------------------------------------------------------------ etapa 2: respostas

def julgar(rag: RagService, pergunta: str, res) -> int | None:
    from app.prompts import montar_contexto
    prompt = PROMPT_JUIZ.format(contexto=montar_contexto(res.hits), pergunta=pergunta, resposta=res.resposta)
    try:
        bruto = rag.llm.gerar(prompt, "Você é um avaliador rigoroso e objetivo.", temperatura=0.0)
        return int(json.loads(re.search(r"\{.*\}", bruto, re.DOTALL).group(0))["nota"])
    except Exception:
        return None


def avaliar_respostas(rag: RagService, golden: list[dict], usar_juiz: bool, pausa: float) -> dict:
    linhas, interrompido = [], None
    for g in golden:
        try:
            historico: list[dict] = []
            for anterior in g.get("anteriores", []):
                r0 = rag.responder(anterior, historico)
                historico += [{"role": "user", "content": anterior}, {"role": "assistant", "content": r0.resposta}]
                time.sleep(pausa)
            res = rag.responder(g["pergunta"], historico)
            time.sleep(pausa)
        except LLMQuotaDiaria as e:       # cota do dia acabou: parar já, guardando o que foi feito
            interrompido = str(e)
            print(f"\n  ⚠ Avaliação interrompida em [{g['id']}]: {e}")
            break
        except LLMError as e:             # erro isolado (rede/503): registra e segue
            print(f"  ! {g['id']:<22} erro da API: {str(e)[:80]}")
            continue

        if g.get("recusa"):
            ok_conteudo, ok_fonte = eh_recusa(res.resposta), not res.fontes
        else:
            ok_conteudo = all(re.search(p, res.resposta, re.IGNORECASE) for p in g.get("deve_conter", []))
            ok_fonte = casa_url(" ".join(f.url for f in res.fontes), g.get("url_contem", [""])) if g.get("url_contem") else True
        nota = julgar(rag, g["pergunta"], res) if usar_juiz and not g.get("recusa") else None
        linhas.append({"id": g["id"], "categoria": g["categoria"], "conteudo": ok_conteudo, "fonte": ok_fonte,
                       "fidelidade": nota, "resposta": res.resposta, "consulta": res.consulta})
        marca = lambda b: "✓" if b else "✗"
        print(f"  {marca(ok_conteudo and ok_fonte)} {g['id']:<22} conteúdo={marca(ok_conteudo)} fonte={marca(ok_fonte)}"
              + (f" fidelidade={nota}" if nota else ""))

    total = len(linhas)
    if not total:
        print("\nNenhuma pergunta foi avaliada.")
        return {"aprovadas": 0, "total": 0, "interrompido": interrompido, "itens": []}
    aprovadas = sum(l["conteudo"] and l["fonte"] for l in linhas)
    notas = [l["fidelidade"] for l in linhas if l["fidelidade"]]
    parcial = f"  (PARCIAL: {total} de {len(golden)} perguntas; interrompida por cota)" if interrompido else ""
    print(f"\nRESPOSTAS: {aprovadas}/{total} aprovadas ({aprovadas / total:.0%}){parcial}")
    por_cat: dict[str, list[bool]] = {}
    for l in linhas:
        por_cat.setdefault(l["categoria"], []).append(l["conteudo"] and l["fonte"])
    for cat, v in sorted(por_cat.items()):
        print(f"  {cat:<16} {sum(v)}/{len(v)}")
    if notas:
        print(f"  fidelidade média (juiz): {sum(notas) / len(notas):.2f} / 5")
    return {"aprovadas": aprovadas, "total": total, "interrompido": interrompido,
            "fidelidade_media": sum(notas) / len(notas) if notas else None, "itens": linhas}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden.jsonl")
    ap.add_argument("--k", type=int, default=get_settings().top_k)
    ap.add_argument("--respostas", action="store_true", help="roda o sistema inteiro (gasta chamadas de LLM)")
    ap.add_argument("--juiz", action="store_true", help="com --respostas: nota de fidelidade por um LLM juiz")
    ap.add_argument("--sintetico", nargs="?", const="eval/sintetico.jsonl", metavar="ARQUIVO",
                    help="também avalia a busca com perguntas geradas por eval.gerar_perguntas")
    ap.add_argument("--sem-vetores", action="store_true", help="força busca só por palavras")
    ap.add_argument("--pausa", type=float, default=1.0, help="segundos entre chamadas (limite do plano gratuito)")
    ap.add_argument("--out", default="eval/resultados.json")
    args = ap.parse_args()

    s = get_settings()
    golden = carregar_golden(args.golden)
    llm = None
    if s.gemini_api_key and not args.sem_vetores:
        from app.llm import GeminiClient
        llm = GeminiClient(s)
    elif args.respostas:
        raise SystemExit("--respostas precisa de GEMINI_API_KEY (veja .env.example).")

    retriever = Retriever(s.index_db_path, llm.embed_consulta if llm else None)
    resultado = {"busca": avaliar_busca(retriever, golden, args.k, s.candidates)}

    if args.sintetico:
        resultado["sintetico"] = avaliar_sintetico(retriever, args.sintetico, args.k, s.candidates)

    if args.respostas:
        print("\nRODANDO O SISTEMA COMPLETO...")
        resultado["respostas"] = avaliar_respostas(RagService(retriever, llm, s), golden, args.juiz, args.pausa)

    Path(args.out).write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados salvos em {args.out}")


if __name__ == "__main__":
    main()

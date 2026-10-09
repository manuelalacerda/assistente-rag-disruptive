# 🤖 Assistente RAG - Disruptive Architectures IoT, IoB e Generative AI
 
Chat com IA que responde perguntas sobre o site da disciplina **usando só o conteúdo publicado** (aulas, labs, notebooks, checkpoints e agenda) e **cita a página de origem** de cada resposta.
 
🔗 **Deploy:** `https://assistente-rag-disruptive.onrender.com/`  
📦 **Repositório:** `https://github.com/manuelalacerda/assistente-rag-disruptive`
 
---

## Sobre o projeto
 
Projeto da disciplina **Disruptive Architectures IoT, IoB e Generative AI**, um **sistema completo: assistente virtual de IA que usa **RAG** (*Retrieval-Augmented Generation*) sobre o site da matéria, com atenção especial à **qualidade das respostas**.
  
**O que o assistente faz?**
 
- Responde somente com base no conteúdo da disciplina e cita a fonte de cada informação (`[1]`, `[2]`...), com link para a página.
- Recusa o que não está no material (*"Não encontrei isso no material da disciplina."*) e o que não tem relação com a disciplina (*"Só respondo sobre o conteúdo da disciplina."*).
- Entende perguntas de continuação usando o histórico da conversa.
- Avisa quando um termo é repetido. Exemplo: "Lab 3" existe em IoT e em GenAI, então o assistente apresenta as duas possibilidades.
- Coleta feedback do usuário (👍 ajudou / 👎 não ajudou) para melhorar o sistema com dados reais.

---

## 🌐 Tecnologias

* Python
* FastAPI
* Gemini API (geração e embeddings)
* SQLite
* HTML/JS (front-end)
* Docker
* Render
 
---
 
## 👥 Integrantes

<table>
  <tr>
    <td width="130">
      <img src="https://github.com/manuelalacerda.png" width="120" style="border-radius: 50%;"/>
    </td>
    <td>
      <b>Manuela de Lacerda Soares</b><br/>
      <b>RM:</b> 564887 &nbsp;&nbsp;|&nbsp;&nbsp;<b>Turma:</b> 2TDSPG - FIAP <br/>
    </td>
  </tr>
  <tr>
    <td width="130">
      <img src="https://github.com/sSofia-s.png" width="120" style="border-radius: 50%;"/>
    </td>
    <td>
      <b>Sofia Siqueira Fontes</b><br/>
      <b>RM:</b> 563829 &nbsp;&nbsp;|&nbsp;&nbsp;<b>Turma:</b> 2TDSPG - FIAP <br/>
    </td>
  </tr>
</table>
 
---

## 🛠️ Arquitetura

```
INGESTÃO (offline, você roda)                 SERVIDOR (online)
GitHub da disciplina                          Navegador ──► FastAPI (/api/chat)
  └ mkdocs.yml (nav) + .md + .ipynb                          │ 1. reescreve a pergunta (se há histórico)
      │ sources.py  → documentos + URL                       │ 2. busca híbrida: vetores + BM25 → RRF
      │ chunker.py  → chunks por título/código               │ 3. prompt com trechos numerados [1][2]
      │ build_index → embeddings (Gemini)                    │ 4. Gemini gera a resposta
      ▼                                                      │ 5. devolve texto + fontes citadas
  data/index.db  ───────── lido pelo servidor ───────────────┘
                                          chat.db (SQLite): mensagens, fontes usadas, feedback
```

---

## 👨‍💻 Como testar

### Opção A - Aplicação no ar
1. Abra `https://assistente-rag-disruptive.onrender.com/` e faça as perguntas que desejar.
2. Clique em 👍/👎 numa resposta e confira `https://assistente-rag-disruptive.onrender.com/api/stats`.

### Também é possível testar a API diretamente:
Pelo terminal:

```bash
curl -X POST https://assistente-rag-disruptive.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "Quantos checkpoints existem por semestre?"}'
```


### Opção B - Rodando localmente
Pré-requisitos: Python 3.12+ e uma chave gratuita do Gemini em <https://aistudio.google.com/apikey>.
Clone o repositório e execute os comandos:

```bash
git clone https://github.com/manuelalacerda/assistente-rag-disruptive.git
cd assistente-rag-disruptive
 
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
 
copy .env.example .env
# edite o .env e preencha sua GEMINI_API_KEY
 
python -m ingest.build_index       # baixa o material e gera data/index.db
uvicorn app.main:app --reload      # abra http://localhost:8000
```

---
   
## 🧪 Testes e qualidade

```bash
python -m pytest -q                          # testes automatizados (sem internet, sem chave)
python -m eval.run_eval                      # qualidade da BUSCA: hit@k e MRR
python -m eval.run_eval --respostas --juiz   # sistema inteiro + nota de fidelidade e (precisa da chave)
```

---

## 📦 Deploy (Render)

1. Suba o projeto no GitHub **com `data/index.db` commitado** e sem o `.env`.
2. No Render: New → Blueprint → escolha o repositório (usa `render.yaml`).
3. No painel, defina `GEMINI_API_KEY`.
4. Teste `https://SEU-APP.onrender.com/health`.

O plano gratuito "dorme" após inatividade: abra o site 1 min antes de apresentar.

---

## ⚖️ Decisões técnicas

| Decisão | Por quê |
|---|---|
| Busca híbrida (vetor + BM25 + RRF) | Vetores acertam paráfrases; BM25 acerta termos exatos (MQTT, Serial.begin). |
| Chunks por título, código intacto | O pipeline ingênuo corta código e perde contexto. |
| Notebooks indexados | Nos labs de GenAI o conteúdo real está nos `.ipynb`. |
| Só páginas do `nav` | Evita responder com material antigo esquecido (ex.: agenda de 2025). |
| Reescrita da pergunta | "e a porta?" só faz sentido com o histórico. |
| Prompt restritivo + citações [n] | Reduz alucinação e permite conferir a fonte. |
| SQLite | Zero infraestrutura; índice (somente leitura) separado do log de conversas. |
| Rate limit | Protege a cota da sua chave numa API pública. |

---

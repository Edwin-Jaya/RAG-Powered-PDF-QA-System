# RAG-Powered PDF QA System (Streamlit · Qdrant · LM Studio · GPT-OSS · Poetry)

An end-to-end Retrieval-Augmented Generation (RAG) system that ingests PDFs, generates embeddings using LM Studio (text-embedding-nomic-embed-text-v1.5), stores vector representations in Qdrant, and provides a Streamlit interface that queries a cloud LLM (GPT-OSS) for context-grounded answers. The prompt is designed to prevent the model from responding with information outside the retrieved context.

# Table of contents
1. Highlights
2. Demo
3. Tech stack
4. Repository layout
5. Quick start (local)
6. Flow
7.  Troubleshooting & gotchas
8.  Tests & CI suggestions
9.  Contact
10. License

# Highlights

- Ingest multiple PDFs → extract text → chunk with overlap → compute embeddings (LM Studio) → upsert to Qdrant.
- Retrieval + prompt composition → Cloud LLM (GPT-OSS) answer (configurable to OpenAI/Azure/Gemini/etc).
- Clean, modular code: services/ wrappers for extractor, chunker, embeddings, vectorstore, rag pipeline.
- Streamlit UI for easy demo and non-technical users.

# Demo

![Demo](https://github.com/Edwin-Jaya/RAG-Powered-PDF-QA-System/blob/main/readme_assets/demo.gif)

# Tech stack

- Python 3.12
- Streamlit (frontend)
- Qdrant (vector store)
- LM Studio — local embedding server (text-embedding-nomic-embed-text-v1.5)
- Cloud LLM (GPT-OSS)
- pdfplumber for PDF extraction
- poetry (recommended) or pip for dependency management

# Repository layout

```
RAG-Powered PDF QA System/
├── configs/
│   └── settings.py               # .env loading and constants
├── services/
│   ├── extractor.py              # PDF -> text
│   ├── chunker.py                # chunk_text()
│   ├── embeddings.py             # LM Studio wrapper (get_embedding/get_embeddings)
│   ├── vectorstore.py            # Qdrant client wrapper (create_collection, upsert, search)
│   ├── rag_pipeline.py           # retrieval + prompt assembly + generate
│   └── llm_client.py             # cloud LLM wrapper
├── app.py                        # Streamlit UI (entrypoint)
├── .env
├── requirements.txt / pyproject.toml
├── README.md
└── README_assets/                # screenshots / demo GIF
```

# Quick start (local)
1) Download this project then open it.

2) **If you prefer Poetry:** <br>
<code>poetry install
poetry shell</code>

3) **If you prefer pip:** <br>
<code>python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt</code>

3) Create .env (see example below)
4) Start services (Qdrant & LM Studio)
5) Run Streamlit UI
<code>streamlit run app.py</code>


Open localhost and follow UI: upload PDF → ingest → ask questions.

## Environment (.env) example

Create .env in project root:
```
**Qdrant**
QDRANT_URL=""
COLLECTION_NAME="rag_one"
QDRANT_API_KEY=

**LM Studio**
LM_STUDIO_URL="http://localhost:1234/v1"
EMBED_MODEL="text-embedding-nomic-embed-text-v1.5"

**Cloud LLM (I use GPT-OSS from Groq)**
LLM_API=
```

## LM Studio

1. Open LM Studio GUI.
2. Download text-embedding-nomic-embed-text-v1.5 from Models.
3. In LM Studio Server tab choose Embedding API (not Chat) and start it.
4. Confirm the API (default) http://localhost:1234/v1.
5. Run the app (Streamlit)
   <code>streamlit run app.py</code>


# Flow And Architecture
The diagram below illustrates the high-level architecture of the Retrieval-Augmented Generation (RAG) system implemented in this project.
![General Architecture](https://github.com/Edwin-Jaya/RAG-Powered-PDF-QA-System/blob/main/readme_assets/general_architecture.png)

# Troubleshooting & gotchas

- Model is not embedding → ensure in LM Studio you started Embedding API for text-embedding-nomic-embed-text-v1.5.
- Vectors must be list[float] — do not index into the embedding vector accidentally (e.g., embedding[0]).
- points format — Qdrant expects a list of dicts like {"id":..., "vector":[...], "payload":{...}}.
- Empty upsert — validate len(embeddings) == len(texts) and non-empty lists before calling upsert.
- Vector size mismatch — set VECTOR_SIZE (e.g., 768) when creating collection.

# Deployment suggestions

Containerize the Streamlit app; use a separate VM/container for LM Studio or host embeddings in production. For scale, use managed Qdrant or a Kubernetes cluster. Use incremental ingestion with UUIDs and idempotency keys.

# Contact

- Edwin Liona Jaya — AI Engineer
- LinkedIn: https://www.linkedin.com/in/edwinjaya
- Email: edwinlionaj@gmail.com 

Open to: Full-time roles, contract & freelance (AI Engineering, ML Infrastructure, Data Engineering).

# License

MIT — see LICENSE for details.









# Sanskrit NLP Research Assistant — RAG Pipeline

A production-inspired Retrieval-Augmented Generation (RAG) system that answers research questions over a corpus of Sanskrit NLP papers. Built to demonstrate end-to-end ML engineering: document ingestion, semantic search, LLM integration, evaluation, and a deployment-ready API.

---

## Pipeline Overview

```
PDF Papers → Chunking → Embeddings → ChromaDB → Query → LLM → Answer
                                                      ↓
                                             Evaluation Metrics
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Orchestration | LangChain |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| LLM | `mistralai/Mistral-7B-Instruct-v0.2` (via HuggingFace Inference API) |
| PDF Parsing | PyPDF |
| API | FastAPI + Uvicorn |
| Evaluation | ROUGE, cosine similarity, keyword hit rate |
| Environment | Google Colab (GPU: T4) |

---

## Project Structure

```
sanskrit-nlp-rag/
├── notebook/
│   └── Sanskrit_NLP_RAG_Pipeline.ipynb   ← main notebook
├── api/
│   └── app.py                             ← extracted FastAPI app
├── evaluation/
│   ├── eval_report.json
│   └── eval_results.png
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Prerequisites

- A Google Colab account (free tier with T4 GPU recommended)
- A HuggingFace account and API token → [get one here](https://huggingface.co/settings/tokens)

### 2. Open the Notebook

Upload `Sanskrit_NLP_RAG_Pipeline.ipynb` to Google Colab, or open it directly from GitHub.

### 3. Set Up Your HuggingFace Token

Go to **Secrets** (🔑 icon in the left sidebar) and add:

```
Name:  HF_TOKEN
Value: hf_your_token_here
```

The notebook will read it automatically. If you skip this step, it will prompt you to paste the token at runtime.

### 4. Add Your PDFs

Upload Sanskrit NLP research papers to the `/content/papers/` directory in Colab, or let the notebook auto-download sample ArXiv papers as a fallback.

### 5. Run All Cells

Steps are clearly labeled 1–11. Run them in order:

| Step | What it does |
|---|---|
| 1 | Installs all dependencies |
| 2 | HuggingFace login |
| 3 | Loads PDFs (upload or auto-download) |
| 4 | Chunks documents (512 tokens, 64 overlap) |
| 5 | Generates embeddings and builds ChromaDB vector store |
| 6 | Configures MMR retriever (top-4 chunks) |
| 7 | Connects Mistral-7B LLM with a custom RAG prompt |
| 8 | Interactive Q&A interface |
| 9 | Runs evaluation framework with metrics + charts |
| 10 | Starts a local FastAPI server with `/query`, `/health`, `/stats` endpoints |
| 11 | Saves and downloads the eval report |

---

## API Endpoints

Once Step 10 runs, a REST API is available at `http://localhost:8000`.

**POST** `/query`
```json
{
  "question": "What are the key challenges in Sanskrit machine translation?",
  "top_k": 4
}
```

**GET** `/health` — liveness check

**GET** `/stats` — vector store statistics

Auto-generated docs: `http://localhost:8000/docs`

---

## Evaluation Framework

The notebook runs a systematic evaluation over a curated question set, measuring:

- **Retrieval cosine similarity** — semantic closeness of retrieved chunks to the query
- **Keyword hit rate** — presence of expected terms in the generated answer
- **ROUGE-L** — longest common subsequence overlap against reference answers

Results are saved to `eval_report.json` and visualized as `eval_results.png`.

---

## Requirements

```
langchain
langchain-community
langchain-huggingface
chromadb
sentence-transformers
huggingface_hub
pypdf
fastapi
uvicorn
nest-asyncio
rouge-score
scikit-learn
matplotlib
```

Install all at once:
```bash
pip install langchain langchain-community langchain-huggingface chromadb \
    sentence-transformers huggingface_hub pypdf fastapi uvicorn \
    nest-asyncio rouge-score scikit-learn matplotlib
```

---

## Configuration

Key parameters you can tune in the notebook:

| Parameter | Default | Where |
|---|---|---|
| Chunk size | 512 tokens | Step 4 |
| Chunk overlap | 64 tokens | Step 4 |
| Retrieval top-k | 4 | Step 6 |
| MMR lambda (diversity vs. relevance) | 0.7 | Step 6 |
| LLM max new tokens | 512 | Step 7 |
| LLM temperature | 0.2 | Step 7 |

---

## Notes

- The LLM runs via the HuggingFace **Inference API** (free tier) — no local GPU needed for the generation step. The embedding model runs on the Colab T4 GPU.
- ChromaDB is persisted to `/content/chroma_db/`. If you restart the runtime, re-run Step 5 to rebuild it, or modify the code to load from the persisted directory.
- The RAG prompt instructs the model to answer strictly from retrieved context and flag when information is insufficient — reducing hallucination.

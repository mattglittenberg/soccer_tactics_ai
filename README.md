# Soccer Tactical AI Assistant

A fully local and free AI-powered soccer coaching assistant built with Python. Ask questions about formations, pressing systems, set pieces, and tactical strategy, all running on your machine with no API costs, no rate limits, and no data leaving your device.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?logo=ollama)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20DB-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

- **Fully local** — runs on your machine via Ollama, no cloud API required.
- **RAG pipeline** — answers grounded in your own curated soccer knowledge base.
- **Multi-turn conversation** — maintains session history across questions.
- **FastAPI backend** — REST API with auto-generated interactive docs.
- **Optional Gradio UI** — browser-based chat interface in three lines of code.
- **Secure by default** — localhost-only, input validation, rate limiting.

---

## Architecture

```
User Question
     │
     ▼
FastAPI Backend  ──►  RAG Pipeline  ──►  ChromaDB (local)
     │                    │                    │
     │              Relevant Chunks       Soccer Knowledge
     │                    │               (.txt files)
     ▼                    ▼
  Ollama (local)  ◄──  Augmented Prompt
  llama3.1:8b
     │
     ▼
 Expert Tactical Answer
```

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| LLM | [Ollama](https://ollama.com) + `llama3.1:8b` | Runs locally, free |
| Embeddings | `sentence-transformers` | Local, no API key needed |
| Vector DB | ChromaDB | Persisted to disk |
| Backend | FastAPI | Auto-generated OpenAPI docs |
| UI (optional) | Gradio | Local browser interface |

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed
- 8GB+ free RAM (16GB recommended)
- macOS, Linux, or Windows (WSL)

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/your-username/soccer-ai-coach.git
cd soccer-ai-coach
```

### 2. Set up Python environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install Ollama and pull the model

```bash
# Install Ollama from https://ollama.com or via Homebrew:
brew install ollama

# Start the Ollama server
ollama serve

# Pull the model (4.9GB, one-time download)
ollama pull llama3.1:8b

# Verify GPU acceleration is active (should show "Metal" on Apple Silicon)
ollama ps
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env if needed — defaults work for local development
```

### 5. Build the knowledge base

Add your `.txt` soccer knowledge files to the `knowledge/` folder, then run:

```bash
python ingest.py
```

### 6. Run the assistant

**CLI mode:**
```bash
python assistant.py
```

**API mode:**
```bash
uvicorn api:app --reload
# Visit http://localhost:8000/docs for interactive API documentation
```

**UI mode (optional):**
```bash
python ui.py
# Visit http://localhost:7860
```

---

## Project Structure

```
soccer-ai-coach/
├── .env.example            # Environment variable template
├── .gitignore
├── requirements.txt
│
├── knowledge/              # Your curated soccer content (.txt files)
│   ├── formations_overview.txt
│   ├── pressing_systems.txt
│   ├── positional_play.txt
│   ├── set_pieces.txt
│   ├── transitions.txt
│   └── manager_philosophies.txt
│
├── chroma_db/              # Vector database (auto-generated, gitignored)
│
├── config.py               # Model name, chunk size, retrieval settings
├── prompts.py              # System prompt
├── ingest.py               # Load knowledge/ → ChromaDB (run once)
├── rag.py                  # Retrieval and prompt augmentation
├── assistant.py            # CLI chat interface
├── api.py                  # FastAPI backend
├── ui.py                   # Optional Gradio web UI
│
└── evals/
    └── questions.json      # Evaluation question set
```

---

## Configuration

All key settings live in `config.py`:

```python
MODEL = "llama3.1:8b"
CHUNK_SIZE = 350
CHUNK_OVERLAP = 50
N_RETRIEVAL_RESULTS = 3
```

---

## API Reference

Once the server is running, full interactive docs are available at `http://localhost:8000/docs`.

### `POST /chat`

Send a message and receive a tactical response.

**Request:**
```json
{
  "message": "What are the trade-offs of a high defensive line?",
  "session_id": "optional-session-id",
  "use_rag": true
}
```

**Response:**
```json
{
  "reply": "A high defensive line compresses space in midfield...",
  "session_id": "abc-123",
  "sources": ["pressing_systems.txt", "formations_overview.txt"]
}
```

### `GET /health`
Check API and Ollama server status.

### `DELETE /session/{session_id}`
Clear a conversation session.

### `GET /models`
List all locally available Ollama models.

---

## Building Your Knowledge Base

The quality of the `knowledge/` files is the biggest lever on answer quality. Each `.txt` file should cover one topic in plain, detailed prose — the more specific, the better.

Suggested topics to start:
- `formations_overview.txt` — 4-3-3, 4-2-3-1, 3-5-2, 4-4-2 diamond, etc.
- `pressing_systems.txt` — gegenpressing, high press, mid-block, low block
- `positional_play.txt` — Juego de Posición, half-spaces, third-man runs
- `set_pieces.txt` — attacking and defensive corners, free kicks
- `transitions.txt` — counter-pressing, defensive transitions
- `manager_philosophies.txt` — Guardiola, Klopp, Ancelotti, Arteta, etc.

You can also ingest PDFs (coaching books, match analysis reports) by using the `ingest_pdf()` function in `ingest.py`.

After adding new files, re-run:
```bash
python ingest.py
```

---

## Security

This project is designed for local development. Key security measures already in place:

- Ollama bound to `127.0.0.1` only (no external network access)
- FastAPI CORS restricted to localhost origins
- Rate limiting on the `/chat` endpoint (10 requests/minute by default)
- Input validation via Pydantic (message length capped at 2,000 characters)
- `.env` for all configuration; secrets never hardcoded

Before running, verify Ollama is only listening locally:
```bash
lsof -i :11434
# Should show 127.0.0.1:11434, not *:11434
```

---

## Extending the Project

A few directions worth exploring once the core system is working:

**Add a reranker** — retrieve more candidates, rerank by true relevance before sending to the model:
```bash
pip install sentence-transformers  # already installed
# Use CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
```

**Enable streaming responses** — real-time token streaming via the Ollama stream API makes the assistant feel significantly more responsive.

**Build an eval set** — add 20-30 questions with known good answers to `evals/questions.json` and script an automated scorer. This is how you measure whether prompt or RAG changes actually help.

**Swap models instantly** — change `MODEL` in `config.py` to compare `qwen3.5:9b`, `qwen3.5:4b`, or `deepseek-r1:14b` without touching any other code.

---

## Learning Goals

This project is designed as a hands-on introduction to AI engineering. By building it, you'll understand:

- How local LLMs work and why unified memory (Apple Silicon) matters for inference speed
- LLM API patterns — stateless calls, role-based message history, system prompts
- Prompt engineering — how system prompts shape model behavior
- Embeddings — what they are and why semantic similarity captures meaning
- Vector databases — indexing, cosine similarity search, chunking strategy
- RAG architecture — the industry-standard pattern for grounding LLMs in custom knowledge
- FastAPI — async REST APIs, Pydantic validation, session management, OpenAPI docs

---

## Contributing

Contributions are welcome. If you have soccer knowledge files, improved prompts, or eval questions to add, please open a pull request.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/add-set-piece-knowledge`)
3. Commit your changes (`git commit -m 'Add set piece knowledge base file'`)
4. Push to the branch (`git push origin feature/add-set-piece-knowledge`)
5. Open a Pull Request

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Ollama](https://ollama.com) for making local LLM inference accessible
- [Qwen team at Alibaba](https://github.com/QwenLM/Qwen) for the Qwen3.5 model family
- [ChromaDB](https://www.trychroma.com) for the vector database
- [sentence-transformers](https://www.sbert.net) for local embeddings

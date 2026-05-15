from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager
import uuid
from rag import retrieve, build_augmented_prompt, collection
from config import N_RETRIEVAL_RESULTS, MODEL
from prompts import SYSTEM_PROMPT
import ollama


# --- Startup Check ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        count = collection.count()
        print(f"ChromaDB connected - {count} chunks in knowledge base")
        if count == 0:
            print("Knowledge base is empty - run ingest.py first")
    except Exception as e:
        print(f"ChromaDB error on startup: {e}")
    yield


# --- App Setup ---

app = FastAPI(
    title="Soccer Tactical AI Assistant",
    description="Local LLM-powered soccer coaching assistant with RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Session storage
sessions: dict[str, list] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:7860"],  # only your own frontends
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
    allow_credentials = False
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- Request / Response Models ---

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(default=None, max_length=100)
    use_rag: bool = True

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    sources: list[str]

class SessionInfo(BaseModel):
    session_id: str
    message_count: int


# --- Endpoints ---

@app.get("/health")
async def health():
    """Check if the API and Ollama are running."""
    try:
        ollama.list()
        return {"status": "ok", "ollama": "connected", "model": MODEL}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama not available: {e}")
    

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, body: ChatRequest):
    """Main chat endpoint. Supports multi-turn conversations via session_id"""

    # Create new session if none provided
    session_id = body.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []
    history = sessions[session_id]

    # RAG retrieval
    sources = []
    if body.use_rag:
        chunks = retrieve(body.message, n_results=N_RETRIEVAL_RESULTS)
        message_to_send = build_augmented_prompt(body.message, chunks)
        sources = [c["source"] for c in chunks]
    else:
        message_to_send = body.message

    history.append({"role": "user", "content": message_to_send})

    # Call local Ollama model
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history
        )
        reply = response["message"]["content"]
    except Exception as e:
        history.pop()
        sessions[session_id] = history
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {str(e)}. Is Ollama running?"
        )

    # Store non-augmented message in history
    history[-1] = {"role": "user", "content": body.message}
    history.append({"role": "assistant", "content": reply})
    sessions[session_id] = history

    return ChatResponse(reply=reply, session_id=session_id, sources=sources)


@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Check how many messages are in a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionInfo(
        session_id=session_id,
        message_count=len(sessions[session_id])
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a session's conversation history."""
    sessions.pop(session_id, None)
    return {"message": f"Session {session_id} cleared"}

@app.get("/models")
async def list_models():
    """List all locally available Ollama models."""
    models = ollama.list()
    return {"models": [m.model for m in models.models]}
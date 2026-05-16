from pathlib import Path
from datetime import datetime
import chromadb
import re
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_PATH

SECTION_SEPARATOR = "\n\n---\n\n"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="soccer_tactics",
    metadata={
        "description": "Soccer Tactics knowledgebase",
        "created": str(datetime.now()),
        "hnsw:space": "cosine"
        }  # cosine similarity for text
)

def chunk_text(text: str, chunk_size: int = 350, overlap: int = 50) -> list[str]:
    """
    Splits text on section separator then word count into overlapping chunks.
    """
    if SECTION_SEPARATOR in text:
        sections = text.split(SECTION_SEPARATOR)
    else:
        sections = [text]   # treat whole file as one section

    all_chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        words = section.split()

        # If the section fits in one chunk, keep it whole
        if len(words) <= chunk_size:
            if len(words) > 30:
                all_chunks.append(section)
        else:
            # Otherwise split by word count with overlap
            step = chunk_size - overlap
            for i in range(0, len(words), step):
                chunk = " ".join(words[i : i + chunk_size])
                if len(chunk.split()) > 30:
                    all_chunks.append(chunk)

    return all_chunks

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract raw text from PDF. Works poorly on scanned image pdfs.
    """

    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append(text)
    
    full_text = "\n\n".join(pages)

    # Clean up PDF extraction artifacts
    full_text = re.sub(r'\s+', ' ', full_text)        # collapse whitespace
    full_text = re.sub(r'(\w)-\n(\w)', r'\1\2', full_text)  # fix hyphenation
    
    return full_text.strip()

def ingest_txt(txt_path: str):
    """Ingest a single text file into ChromaDB"""
    
    txt = Path(txt_path)
    print(f"Reading {txt.name}...")

    text = txt.read_text(encoding="utf-8")
        
    # Skip empty files
    if not text:
        print(f"Skipping {txt.name} — file is empty")
        return

    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    # Skip files that produced no chunks
    if not chunks:
        print(f"Skipping {txt.name} — no chunks produced (content too short?)")
        print(f"Word count: {len(text.split())} words")
        return

    # Generate embeddings for all chunks at once
    embeddings = embedder.encode(chunks, show_progress_bar=True).tolist()

    # Store in ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{txt.stem}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": txt.name, "chunk_index": i} for i in range(len(chunks))]
    )
    print(f"Ingested {txt.name} → {len(chunks)} chunks")

def ingest_pdf(pdf_path: str):
    """Ingest a single PDF into ChromaDB"""

    path = Path(pdf_path)
    print(f"Reading {path.name}...")

    text = extract_pdf_text(pdf_path)

    if not text:
        print(f"Could not extract text from {path.name}")
        print(" (PDF may be scanned/image-based - try OCR)")
        return
    
    print(f"Extracted {len(text.split())} words")
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    
    if not chunks:
        print(f"No chunks produced from {path.name}")
        return

    embeddings = embedder.encode(chunks, show_progress_bar=True).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{path.stem}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{
            "source": path.name,
            "type": "pdf", 
            "chunk_index": i
            } for i in range(len(chunks))]
    )
    print(f"Ingested PDF: {path.name} → {len(chunks)} chunks")

def ingest_folder(folder_path: str):
    """Ingest text and PDF files from folder"""
    folder = Path(folder_path)
    
    txt_files = list(folder.glob("*.txt"))
    pdf_files = list(folder.glob("*.pdf"))

    print(f"Found {len(txt_files)} .txt files and {len(pdf_files)} .pdf files\n")

    if not txt_files and pdf_files:
        print(f"No .txt or .pdf files found in {folder_path}")
        return

    for file in txt_files:
        ingest_txt(str(file))

    for file in pdf_files:
        ingest_pdf(str(file))

    print(f"\nKnowledge base ready. Total chunks: {collection.count()}")


if __name__ == "__main__":
    ingest_folder("knowledge/")
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from config import CHUNK_SIZE, CHUNK_OVERLAP

SECTION_SEPARATOR = "\n\n---\n\n"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="soccer_tactics",
    metadata={"hnsw:space": "cosine"}  # cosine similarity for text
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

def ingest_folder(folder_path: str):
    folder = Path(folder_path)
    txt_files = list(folder.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {folder_path}")
        return

    for file in txt_files:
        text = file.read_text(encoding="utf-8")
        
                # Skip empty files
        if not text:
            print(f"Skipping {file.name} — file is empty")
            continue

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        # Skip files that produced no chunks
        if not chunks:
            print(f"Skipping {file.name} — no chunks produced (content too short?)")
            print(f"Word count: {len(text.split())} words")
            continue

        # Generate embeddings for all chunks at once
        embeddings = embedder.encode(chunks, show_progress_bar=True).tolist()

        # Store in ChromaDB
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"{file.stem}_chunk_{i}" for i in range(len(chunks))],
            metadatas=[{"source": file.name, "chunk_index": i} for i in range(len(chunks))]
        )
        print(f"Ingested {file.name} → {len(chunks)} chunks")

    print(f"\nKnowledge base ready. Total chunks: {collection.count()}")

def ingest_pdf(pdf_path: str):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    embeddings = embedder.encode(chunks, show_progress_bar=True).tolist()

    filename = Path(pdf_path).name
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{Path(pdf_path).stem}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": filename, "chunk_index": i} for i in range(len(chunks))]
    )
    print(f"Ingested PDF: {filename} → {len(chunks)} chunks")


if __name__ == "__main__":
    ingest_folder("knowledge/")
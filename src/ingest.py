from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer


embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="soccer_tactics",
    metadata={"hnsw:space": "cosine"}  # cosine similarity for text
)

def chunk_text(text: str, chunk_size: int = 350, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.
    overlap ensures context isn't lost at chunk boundaries.
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.split()) > 30:  # skip tiny trailing chunks
            chunks.append(chunk)
    return chunks

def ingest_folder(folder_path: str):
    folder = Path(folder_path)
    txt_files = list(folder.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {folder_path}")
        return

    for file in txt_files:
        text = file.read_text(encoding="utf-8")
        chunks = chunk_text(text)

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


if __name__ == "__main__":
    ingest_folder("knowledge/")
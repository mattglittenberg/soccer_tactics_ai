import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import CHROMA_PATH

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection("soccer_tactics")


def retrieve(query: str, n_results: int = 3) -> list[dict]:
    """
    Embed the query and find the most semantically similar chunks.
    Then rerank with cross encoder to improve relevence.
    Returns list of dicts with 'text' and 'source' keys.
    """
    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results*2,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "relevance": round(1 - dist, 3)
        })

    pairs = [(query, c["text"]) for c in chunks]
    rerank_scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, rerank_scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    return reranked[:n_results]


def build_augmented_prompt(query: str, chunks: list[dict]) -> str:
    """
    Inject retrieved context into the user's question.
    The model uses this context to ground its answer.
    """
    context_blocks = []
    for chunk in chunks:
        context_blocks.append(
            f"[Source: {chunk['source']} | Relevance: {chunk['relevance']}]\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    return f"""Use the following soccer knowledge base to answer the question. 
    If the knowledge base doesn't cover something, supplement with your general expertise.
    Always prioritize the knowledge base content when it's relevant.

=== KNOWLEDGE BASE ===
{context}
=== END KNOWLEDGE BASE ===

Question: {query}"""
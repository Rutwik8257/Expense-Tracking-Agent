"""
ChromaDB-backed memory for user corrections.

How it works:
- Each correction is stored as a document: the transaction description (text to embed)
  with metadata: {corrected_category, amount, merchant}
- On lookup, ChromaDB computes cosine similarity between the new tx description
  and all stored corrections, returning the closest match.
- If similarity > 0.85, categorize_tx biases toward that category.
"""
import chromadb
from chromadb.config import Settings

# Persistent local storage — survives restarts
_client = chromadb.PersistentClient(
    path="db/chroma",
    settings=Settings(anonymized_telemetry=False),
)
_collection = _client.get_or_create_collection(
    name="corrections",
    metadata={"hnsw:space": "cosine"},  # use cosine distance
)


def save_correction(transaction: dict, corrected_category: str) -> None:
    """Store a user correction so future categorization can learn from it."""
    doc_id = f"tx_{transaction.get('id', transaction['description'])}_{corrected_category}"
    _collection.upsert(
        ids=[doc_id],
        documents=[transaction["description"]],
        metadatas=[{
            "corrected_category": corrected_category,
            "amount": str(transaction.get("amount", "")),
            "merchant": transaction.get("description", ""),
        }],
    )


def find_similar_correction(description: str, n_results: int = 1) -> list[dict]:
    """
    Return up to n_results past corrections similar to `description`.
    Each result: {"corrected_category": str, "distance": float, "merchant": str}
    distance is cosine distance (0 = identical, 1 = orthogonal).
    Similarity = 1 - distance, so similarity > 0.85 means distance < 0.15.
    """
    count = _collection.count()
    if count == 0:
        return []

    results = _collection.query(
        query_texts=[description],
        n_results=min(n_results, count),
        include=["metadatas", "distances"],
    )

    output = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        similarity = 1.0 - dist
        output.append({
            "corrected_category": meta["corrected_category"],
            "merchant": meta["merchant"],
            "similarity": round(similarity, 4),
        })
    return output


def clear_memory() -> None:
    """Wipe all corrections — used in eval baseline runs."""
    global _collection
    _client.delete_collection("corrections")
    _collection = _client.get_or_create_collection(
        name="corrections",
        metadata={"hnsw:space": "cosine"},
    )

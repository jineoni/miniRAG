from src.config import EMBEDDING_MODEL, TOP_K


def search_similar_chunks(question, model, collection, top_k=TOP_K):
    if not question.strip():
        raise ValueError("Enter a question.")
    if top_k < 1:
        raise ValueError("Top-K must be positive.")
    if collection.metadata.get("embedding_model") != EMBEDDING_MODEL:
        raise ValueError("Embedding model changed. Run python build_index.py again.")
    count = collection.count()
    if not count:
        return []
    vector = model.encode(question, normalize_embeddings=True)
    results = collection.query(
        query_embeddings=[vector.tolist()],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            **metadata,
            "source_id": rank,
            "chunk_id": chunk_id,
            "text": text,
            "similarity": 1 - distance,
        }
        for rank, (chunk_id, text, metadata, distance) in enumerate(zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0],
        ), start=1)
    ]

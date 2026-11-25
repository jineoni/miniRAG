import chromadb
from sentence_transformers import SentenceTransformer

from src.config import COLLECTION_NAME, DB_PATH, EMBEDDING_MODEL


def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def open_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_collection(COLLECTION_NAME, embedding_function=None)


def run_indexing(chunks):
    """Rebuild the collection to remove obsolete chunks on subsequent runs."""
    if not chunks:
        raise ValueError("No text extracted. The existing index has not been changed.")
    model = load_embedding_model()
    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    client = chromadb.PersistentClient(path=str(DB_PATH))
    if COLLECTION_NAME in [collection.name for collection in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        COLLECTION_NAME,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": EMBEDDING_MODEL},
    )
    for start in range(0, len(chunks), 100):
        batch = chunks[start:start + 100]
        collection.add(
            ids=[chunk["chunk_id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            embeddings=vectors[start:start + 100].tolist(),
            metadatas=[
                {key: value for key, value in chunk.items() if key not in ("text", "chunk_id")}
                for chunk in batch
            ],
        )
    return collection.count()

import numpy as np
from core.vector_store import VectorStore


def simple_embed(text: str, dim=128):
    """
    Very simple embedding using character-level hashing.
    Deterministic and API-free.
    """
    vector = np.zeros(dim)

    for char in text.lower():
        vector[ord(char) % dim] += 1

    norm = np.linalg.norm(vector)
    if norm != 0:
        vector = vector / norm

    return vector


def build_vector_store(chunks):
    vectors = []
    for chunk in chunks:
        vec = simple_embed(chunk["text"])
        vectors.append(vec)

    vectors = np.array(vectors).astype("float32")

    store = VectorStore(dim=vectors.shape[1])
    store.add(vectors, chunks)
    return store


def retrieve(query: str, store: VectorStore, top_k=5):
    query_vector = simple_embed(query).reshape(1, -1).astype("float32")
    return store.search(query_vector, top_k=top_k)

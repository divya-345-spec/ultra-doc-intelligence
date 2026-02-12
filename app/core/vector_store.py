import faiss
import numpy as np
import os
import json


STORE_DIR = "data/vector_store"


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.text_chunks = []

    def add(self, vectors, chunks):
        self.index.add(vectors)
        self.text_chunks.extend(chunks)

    def search(self, query_vector, top_k=5):
        if self.index.ntotal == 0:
            return []

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.text_chunks):
                results.append({
                    "chunk": self.text_chunks[idx],
                    "distance": float(dist)
                })
        return results

    def save(self, path=STORE_DIR):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(self.text_chunks, f)

    @classmethod
    def load(cls, path=STORE_DIR):
        index_path = os.path.join(path, "index.faiss")
        chunks_path = os.path.join(path, "chunks.json")

        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            return None

        index = faiss.read_index(index_path)
        store = cls(dim=index.d)
        store.index = index

        with open(chunks_path, "r", encoding="utf-8") as f:
            store.text_chunks = json.load(f)

        return store

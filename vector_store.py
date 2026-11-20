import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStoreManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # BERT-based embedding model initialize karna
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build_index(self, text_chunks):
        """
        Text chunks ke embeddings bana kar FAISS index build karta hai.
        """
        self.chunks = text_chunks
        if not self.chunks:
            return False

        # Dense vector embeddings generate karna
        embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True)
        dimension = embeddings.shape[1]

        # FAISS L2 Flat Index setup
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings, dtype=np.float32))
        return True

    def search_similar(self, query, k=3):
        """
        User query ke basis par top-k relevant text chunks retrieve karta hai.
        """
        if self.index is None or not self.chunks:
            return []

        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(
            np.array(query_embedding, dtype=np.float32), k
        )

        retrieved = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        return retrieved
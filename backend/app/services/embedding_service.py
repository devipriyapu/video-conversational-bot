from __future__ import annotations

import os


class EmbeddingService:
    def __init__(self, model_name: str, device: str) -> None:
        # Stabilize torch/sentence-transformers runtime on macOS.
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def embed_text(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

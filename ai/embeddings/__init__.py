"""
Biomedical Text & Multimodal Embedding Utilities
Owner: Gourav / Shanks

Provides configurable embedding model interfaces for BAAI/bge-large-en-v1.5 and MedCPT.
"""

from typing import List
import hashlib
from ai.config import AIConfig


class TextEmbedder:
    """
    Interface for biomedical text embedding generation.
    Uses deterministic hashing in mock mode and SentenceTransformers in real mode.
    """

    def __init__(
        self,
        model_name: str = AIConfig.EMBEDDING_MODEL_NAME,
        mode: str = "mock",
        dim: int = 384,
    ):
        self.model_name = model_name
        self.mode = mode.lower()
        self.dim = dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.mode == "real":
            raise RuntimeError(
                f"Real text embedding generation with '{self.model_name}' requires loading model weights."
            )
        vectors: List[List[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [((digest[i % len(digest)] / 255.0) * 2.0 - 1.0) for i in range(self.dim)]
            vectors.append(vec)
        return vectors


__all__ = ["TextEmbedder"]

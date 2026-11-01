"""
Vision Feature Extraction & Embeddings
Owner: Shanks (Feature Extraction)

Generates visual feature vector embeddings from preprocessed Chest X-Ray images.
"""

from typing import List, Any
import numpy as np
from ai.common.schemas import PreprocessedImage
from ai.vision.model import MedicalVisionModel


class VisionEmbedder:
    """
    Converts preprocessed medical images into dense feature vectors.
    """

    def __init__(self, embedding_dim: int = 512, mode: str = "mock"):
        self.embedding_dim = embedding_dim
        self.vision_model = MedicalVisionModel(mode=mode)

    def extract_features(self, image_input: Any) -> List[float]:
        """
        Extract feature embedding vector from a PreprocessedImage or array.
        In mock mode, produces a deterministic vector derived from image statistics.
        """
        if isinstance(image_input, PreprocessedImage):
            base_stats = self.vision_model.extract_features(image_input)
            repeats = (self.embedding_dim // len(base_stats)) + 1
            tiled = np.tile(base_stats, repeats)[: self.embedding_dim]
            return [round(float(x), 5) for x in tiled]

        return [0.0] * self.embedding_dim

"""
AI Module Configuration
Owner: Gourav / Shared AI Team

Defines central settings for model paths, retrieval parameters, vector DB settings, and vision pipeline defaults.
"""

class AIConfig:
    # Vision Model Settings
    VISION_MODEL_NAME: str = "MedGemma-7B"  # Options: MedGemma, BiomedCLIP, Rad-DINO
    IMAGE_SIZE: tuple = (512, 512)
    ALLOW_DICOM: bool = True

    # Vector DB & Retrieval Settings
    VECTOR_DB_TYPE: str = "chroma"  # Options: chroma, qdrant
    TOP_K_RETRIEVAL: int = 3
    CHUNKING_SIZE: int = 500
    CHUNKING_OVERLAP: int = 50

    # RAG & LLM Settings
    LLM_MODEL: str = "gpt-4o-mini"
    TEMPERATURE: float = 0.2

    # Disease Classification Settings
    CONFIDENCE_THRESHOLD: float = 0.70
    NUM_CLASSES: int = 14  # e.g. NIH ChestX-ray14 labels

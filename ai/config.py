"""
AI Module Configuration
Owner: Gourav / Shared AI Team

Defines central settings for AI execution mode (mock vs real), Chest X-Ray vision pipeline defaults,
pathology label lists, vector DB settings, and LLM parameters.
"""

import os
from typing import List, Tuple, Dict, Any

# Standard NIH ChestX-ray14 pathology labels
DEFAULT_CHEST_XRAY_LABELS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


class AIConfig:
    # Execution Mode: "mock" (default, zero external dependencies) or "real"
    AI_MODE: str = os.getenv("AI_MODE", "mock").strip().lower()

    # Modality Scope: Focused strictly on Chest X-Ray (DX / CR) for initial prototype
    SUPPORTED_MODALITY: str = "DX"
    ALLOW_DICOM: bool = os.getenv("ALLOW_DICOM", "false").lower() == "true"

    # Vision Model Settings
    VISION_MODEL_NAME: str = os.getenv("VISION_MODEL_NAME", "densenet121-cxr")
    VISION_WEIGHTS_PATH: str = os.getenv("VISION_WEIGHTS_PATH", "")
    IMAGE_SIZE: Tuple[int, int] = (224, 224)

    # Disease Classification Settings
    CHEST_XRAY_LABELS: List[str] = DEFAULT_CHEST_XRAY_LABELS
    NUM_CLASSES: int = len(DEFAULT_CHEST_XRAY_LABELS)
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))

    # Vector DB, Embeddings & Retrieval Settings
    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "chroma")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "data/processed/chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "chest_xray_literature")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
    SAMPLE_GUIDELINES_PATH: str = os.getenv(
        "SAMPLE_GUIDELINES_PATH", "data/samples/chest_xray_guidelines.json"
    )
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "3"))
    CHUNKING_SIZE: int = int(os.getenv("CHUNKING_SIZE", "500"))
    CHUNKING_OVERLAP: int = int(os.getenv("CHUNKING_OVERLAP", "50"))

    # RAG & LLM Settings
    LLM_MODEL: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    @classmethod
    def get_ai_mode(cls) -> str:
        """Return current AI_MODE dynamically from environment or class default."""
        mode = os.getenv("AI_MODE", cls.AI_MODE).strip().lower()
        if mode not in ("mock", "real"):
            raise ValueError(f"Invalid AI_MODE '{mode}'. Expected 'mock' or 'real'.")
        return mode

    @classmethod
    def validate_real_mode_status(cls) -> Dict[str, Any]:
        """
        Inspect availability of real AI integrations without downloading models
        or silently falling back to mock mode.
        """
        weights_path = os.getenv("VISION_WEIGHTS_PATH", cls.VISION_WEIGHTS_PATH)
        vision_ready = bool(weights_path and os.path.isfile(weights_path))

        api_key = os.getenv("OPENAI_API_KEY", cls.OPENAI_API_KEY)
        llm_ready = bool(
            api_key
            and api_key.strip()
            and api_key.strip() != "your_openai_api_key_here"
        )

        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", cls.CHROMA_PERSIST_DIR)
        rag_ready = bool(chroma_dir and os.path.isdir(chroma_dir))

        missing: List[str] = []
        if not vision_ready:
            missing.append(
                "VISION_WEIGHTS_PATH is not configured or weight file does not exist on disk"
            )
        if not rag_ready:
            missing.append(
                f"ChromaDB persistence directory '{chroma_dir}' is not initialized"
            )
        if not llm_ready:
            missing.append("OPENAI_API_KEY is not configured for real LLM synthesis")

        return {
            "vision_ready": vision_ready,
            "rag_ready": rag_ready,
            "llm_ready": llm_ready,
            "all_ready": vision_ready and rag_ready and llm_ready,
            "missing_integrations": missing,
        }

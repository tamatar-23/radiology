"""
Document Ingestion, Metadata Extraction & Text Chunking
Owner: Gourav (Agentic RAG & Vector DB)

Provides clean interfaces and working implementations for ingesting medical literature,
extracting citation metadata, and splitting documents into overlapping chunks.
"""

import json
import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from ai.config import AIConfig


class DocumentChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    document_title: str = Field(..., description="Source reference title")
    source_url: str = Field(..., description="Verifiable source URL or DOI")
    keywords: List[str] = Field(default_factory=list)
    text: str = Field(..., description="Chunk text passage")


class MedicalDocumentIngestor:
    """
    Loads medical reference documents, extracts metadata, and performs overlapping
    character chunking suitable for dense biomedical embedding models.
    """

    def __init__(
        self,
        chunk_size: int = AIConfig.CHUNKING_SIZE,
        chunk_overlap: int = AIConfig.CHUNKING_OVERLAP,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_metadata(self, raw_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Extract standardized citation metadata from a raw document dictionary."""
        return {
            "document_id": str(raw_doc.get("document_id") or raw_doc.get("id") or "doc_unknown"),
            "document_title": str(raw_doc.get("document_title") or raw_doc.get("title") or "Untitled Reference"),
            "source_url": str(raw_doc.get("source_url") or raw_doc.get("url") or ""),
            "keywords": [str(k).lower() for k in raw_doc.get("keywords", [])],
        }

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping segments of at most `chunk_size` characters,
        preferring sentence or word boundaries where possible.
        """
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return []
        if len(cleaned) <= self.chunk_size:
            return [cleaned]

        chunks: List[str] = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        while start < len(cleaned):
            end = min(start + self.chunk_size, len(cleaned))
            if end < len(cleaned):
                # Prefer breaking at a period or space
                last_period = cleaned.rfind(". ", start, end)
                last_space = cleaned.rfind(" ", start, end)
                if last_period > start + (self.chunk_size // 2):
                    end = last_period + 1
                elif last_space > start + (self.chunk_size // 2):
                    end = last_space
            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(cleaned):
                break
            start = max(start + 1, end - self.chunk_overlap)
        return chunks

    def ingest_documents(self, raw_documents: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """Convert a list of raw documents into typed DocumentChunk objects."""
        all_chunks: List[DocumentChunk] = []
        for doc in raw_documents:
            meta = self.extract_metadata(doc)
            content = str(doc.get("content") or doc.get("text") or "")
            text_segments = self.chunk_text(content)
            for idx, segment in enumerate(text_segments):
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=f"{meta['document_id']}#chunk-{idx}",
                        document_id=meta["document_id"],
                        document_title=meta["document_title"],
                        source_url=meta["source_url"],
                        keywords=meta["keywords"],
                        text=segment,
                    )
                )
        return all_chunks

    def load_corpus_file(self, corpus_path: str) -> List[DocumentChunk]:
        """Load and chunk a JSON corpus file from disk."""
        if not os.path.isfile(corpus_path):
            return []
        with open(corpus_path, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)
        if not isinstance(raw_docs, list):
            raise ValueError(f"Expected a JSON array of documents in '{corpus_path}'.")
        return self.ingest_documents(raw_docs)

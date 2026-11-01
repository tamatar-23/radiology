"""
Medical Vector Database Store Interface (ChromaDB)
Owner: Gourav (Agentic RAG & Vector DB)

Provides a ChromaDB wrapper with lazy client initialization so mock mode
does not require a running Chroma server or heavy embedding dependencies.
"""

import os
from typing import List, Dict, Any, Optional
from ai.config import AIConfig
from ai.rag.ingestion import DocumentChunk


class MedicalVectorStore:
    """
    ChromaDB vector database interface for storing and querying medical literature chunks.
    """

    def __init__(
        self,
        collection_name: str = AIConfig.CHROMA_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        embedding_model_name: str = AIConfig.EMBEDDING_MODEL_NAME,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory or AIConfig.CHROMA_PERSIST_DIR
        self.embedding_model_name = embedding_model_name
        self._client: Any = None
        self._collection: Any = None

    def _ensure_collection(self) -> Any:
        """Lazily initialize ChromaDB PersistentClient and collection when requested."""
        if self._collection is not None:
            return self._collection

        try:
            import chromadb  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is required for vector store operations in real mode."
            ) from exc

        os.makedirs(self.persist_directory, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"embedding_model": self.embedding_model_name},
        )
        return self._collection

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: Optional[List[List[float]]] = None) -> int:
        """Index a list of DocumentChunks into the ChromaDB collection."""
        if not chunks:
            return 0
        collection = self._ensure_collection()
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "document_id": c.document_id,
                "document_title": c.document_title,
                "source_url": c.source_url,
                "keywords": ",".join(c.keywords),
            }
            for c in chunks
        ]
        kwargs: Dict[str, Any] = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        collection.upsert(**kwargs)
        return len(chunks)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Backward-compatible method accepting raw document dictionaries."""
        from ai.rag.ingestion import MedicalDocumentIngestor

        ingestor = MedicalDocumentIngestor()
        chunks = ingestor.ingest_documents(documents)
        if chunks:
            self.add_chunks(chunks)

    def search(
        self,
        query_embedding: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = AIConfig.TOP_K_RETRIEVAL,
    ) -> List[Dict[str, Any]]:
        """Execute similarity search against the ChromaDB collection."""
        if not os.path.isdir(self.persist_directory):
            return []

        collection = self._ensure_collection()
        if collection.count() == 0:
            return []

        n_results = min(top_k, collection.count())
        if query_embedding is not None:
            results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
        elif query_text is not None:
            results = collection.query(query_texts=[query_text], n_results=n_results)
        else:
            return []

        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        items: List[Dict[str, Any]] = []
        for idx, chunk_id in enumerate(ids):
            meta = metas[idx] if idx < len(metas) and metas[idx] else {}
            dist = float(distances[idx]) if idx < len(distances) else 0.25
            score = max(0.0, min(1.0, 1.0 / (1.0 + dist)))
            items.append(
                {
                    "chunk_id": chunk_id,
                    "document_title": meta.get("document_title", "Indexed Medical Document"),
                    "source_url": meta.get("source_url", ""),
                    "passage": docs[idx] if idx < len(docs) else "",
                    "relevance_score": round(score, 3),
                }
            )
        return items

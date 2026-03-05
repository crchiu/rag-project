import os
import uuid
from typing import List, Dict, Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from sentence_transformers import SentenceTransformer, CrossEncoder


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


class RAG:
    """
    RAG core:
    - Embedding: BAAI/bge-m3 (SentenceTransformer) -> Qdrant vector search
    - Reranker: jinaai/jina-reranker-v2-base-multilingual (CrossEncoder) -> rerank top candidates
    - Isolation: scope_id filter (avoid cross-user mixing)
    """

    def __init__(self):
        # --- Qdrant ---
        self.qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.collection = os.getenv("QDRANT_COLLECTION", "project_rag")
        self.client = QdrantClient(url=self.qdrant_url)

        # --- Models ---
        self.embed_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.reranker_model = os.getenv("RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual")

        # --- Retrieval ---
        self.top_k = env_int("TOP_K", 10)
        self.search_candidates = env_int("SEARCH_CANDIDATES", 40)

        # --- Runtime / perf ---
        # DEVICE=cuda / cpu / auto
        self.device = os.getenv("DEVICE", "auto").lower()
        # Reranker batch size
        self.rerank_batch = env_int("RERANK_BATCH", 32)
        # Embedding batch size
        self.embed_batch = env_int("EMBED_BATCH", 32)

        resolved_device = self._resolve_device(self.device)

        # Embedding model (SentenceTransformer supports device=...)
        self.embedder = SentenceTransformer(self.embed_model, device=resolved_device)

        # Reranker (CrossEncoder supports device=...; Jina reranker requires trust_remote_code)
        self.reranker = CrossEncoder(
            self.reranker_model,
            device=resolved_device,
            trust_remote_code=True
        )

        self._ensure_collection()

        print(
            f"[RAG] device={resolved_device} embed_model={self.embed_model} "
            f"reranker_model={self.reranker_model} top_k={self.top_k} "
            f"candidates={self.search_candidates} rerank_batch={self.rerank_batch}"
        )

    def _resolve_device(self, device: str) -> str:
        """
        Resolve device string to 'cuda' or 'cpu'.
        If DEVICE=auto, pick cuda when available.
        """
        if device in ("cuda", "cpu"):
            return device

        # auto
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _ensure_collection(self):
        dim = self.embedder.get_sentence_embedding_dimension()
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            # (Optional) Could validate dim here; keep it simple for now
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

        # Payload indexes for filtering / metadata lookup
        for f in ["scope_id", "doc_id", "filename", "section_path"]:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=f,
                field_schema=qm.PayloadSchemaType.KEYWORD
            )

    # -------- Embedding / Upsert --------

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.embed_batch,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return vecs.astype(np.float32)

    def delete_scope(self, scope_id: str) -> None:
        flt = qm.Filter(
            must=[qm.FieldCondition(key="scope_id", match=qm.MatchValue(value=scope_id))]
        )
        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(filter=flt),
            wait=True
        )

    def upsert(self, scope_id: str, doc_id: str, filename: str, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0

        vectors = self.embed([c["text"] for c in chunks])

        points: List[qm.PointStruct] = []
        for c, v in zip(chunks, vectors):
            payload = {
                "scope_id": scope_id,
                "doc_id": doc_id,
                "filename": filename,
                "chunk_id": int(c["chunk_id"]),
                "section_path": c.get("section_path"),
                "page": c.get("page"),
                "text": c["text"],
            }
            points.append(
                qm.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=v.tolist(),
                    payload=payload
                )
            )

        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    # -------- Search / Rerank --------

    def _vector_search(self, scope_id: str, query: str, limit: int) -> List[Dict[str, Any]]:
        qv = self.embed([query])[0].tolist()

        flt = qm.Filter(
            must=[qm.FieldCondition(key="scope_id", match=qm.MatchValue(value=scope_id))]
        )

        res = self.client.search(
            collection_name=self.collection,
            query_vector=qv,
            query_filter=flt,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        hits: List[Dict[str, Any]] = []
        for r in res:
            p = r.payload or {}
            hits.append({
                "doc_id": p.get("doc_id", ""),
                "filename": p.get("filename", ""),
                "chunk_id": int(p.get("chunk_id", -1)),
                "score": float(r.score),  # vector similarity
                "text": p.get("text", ""),
                "meta": {k: v for k, v in p.items() if k != "text"},
            })
        return hits

    def _rerank(self, query: str, hits: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not hits:
            return hits

        # CrossEncoder expects (query, doc_text) pairs
        pairs = [(query, (h.get("text") or "")) for h in hits]

        # Batch prediction is critical for performance
        rr_scores = self.reranker.predict(pairs, batch_size=self.rerank_batch)

        ranked: List[Dict[str, Any]] = []
        for h, s in zip(hits, rr_scores):
            h2 = dict(h)
            # overwrite score with reranker relevance score
            h2["score"] = float(s)
            ranked.append(h2)

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    def search(self, scope_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        cand = max(top_k, self.search_candidates)
        hits = self._vector_search(scope_id, query, limit=cand)
        return self._rerank(query, hits, top_k=top_k)
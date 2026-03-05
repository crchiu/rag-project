import os
import uuid
from typing import List, Dict, Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

class RAG:
    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.collection = os.getenv("QDRANT_COLLECTION", "project_rag")
        self.model_name = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v3")

        self.top_k = env_int("TOP_K", 10)
        self.dom_ratio = env_float("DOMINANT_SECTION_RATIO", 0.6)
        self.min_top_score = env_float("MIN_TOP_SCORE", 0.35)

        self.client = QdrantClient(url=self.qdrant_url)

        # ✅ Jina v3 需要 trust_remote_code
        self.embedder = SentenceTransformer(self.model_name, trust_remote_code=True)

        self._ensure_collection()

    def _ensure_collection(self):
        dim = self.embedder.get_sentence_embedding_dimension()
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

        for f in ["scope_id", "doc_id", "filename", "section_path"]:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=f,
                field_schema=qm.PayloadSchemaType.KEYWORD
            )

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
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

    def search(self, scope_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        qv = self.embed([query])[0].tolist()

        flt = qm.Filter(
            must=[qm.FieldCondition(key="scope_id", match=qm.MatchValue(value=scope_id))]
        )

        res = self.client.search(
            collection_name=self.collection,
            query_vector=qv,
            query_filter=flt,
            limit=top_k,
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
                "score": float(r.score),
                "text": p.get("text", ""),
                "meta": {k: v for k, v in p.items() if k != "text"},
            })
        return hits
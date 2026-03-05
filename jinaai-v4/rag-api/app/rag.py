import os
import uuid
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def resolve_device() -> torch.device:
    d = (os.getenv("DEVICE", "auto") or "auto").lower()
    if d == "cuda":
        return torch.device("cuda")
    if d == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(env_name: str, default: str) -> torch.dtype:
    v = (os.getenv(env_name, default) or default).lower()
    if v == "bf16":
        return torch.bfloat16
    if v == "fp16":
        return torch.float16
    return torch.float32


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden: [B, T, H], mask: [B, T]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


class JinaV4Embedder:
    """
    Native Transformers embedder for jina-embeddings-v4.
    - Uses mean pooling + L2 normalize.
    - Supports truncate_dim (Matryoshka) by slicing.
    """
    def __init__(self, model_id: str, device: torch.device, dtype: torch.dtype, truncate_dim: int, batch_size: int):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.truncate_dim = truncate_dim
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=(dtype if device.type == "cuda" else None),
            device_map=("auto" if device.type == "cuda" else None),
        )
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> np.ndarray:
        vecs: List[np.ndarray] = []
        bs = self.batch_size

        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            tok = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            tok = {k: v.to(self.device) for k, v in tok.items()}

            out = self.model(**tok)
            pooled = mean_pool(out.last_hidden_state, tok["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)

            if self.truncate_dim and pooled.shape[1] > self.truncate_dim:
                pooled = pooled[:, : self.truncate_dim]

            vecs.append(pooled.detach().float().cpu().numpy())

        return np.vstack(vecs).astype(np.float32)

    def dim(self) -> int:
        # probe dimension
        v = self.encode(["__dim_probe__"])
        return int(v.shape[1])


class JinaReranker:
    """
    Native Transformers reranker for jina-reranker-v2-base-multilingual.
    - Uses SequenceClassification model logits as relevance score.
    """
    def __init__(self, model_id: str, device: torch.device, dtype: torch.dtype, batch_size: int):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=(dtype if device.type == "cuda" else None),
            device_map=("auto" if device.type == "cuda" else None),
        )
        self.model.eval()

    @torch.inference_mode()
    def score(self, query: str, passages: List[str]) -> List[float]:
        scores: List[float] = []
        bs = self.batch_size

        for i in range(0, len(passages), bs):
            batch = passages[i:i+bs]
            tok = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            tok = {k: v.to(self.device) for k, v in tok.items()}
            out = self.model(**tok)

            # logits: [B, 1] or [B, 2] depending on head; handle both
            logits = out.logits
            if logits.dim() == 2 and logits.size(-1) == 1:
                batch_scores = logits.squeeze(-1)
            elif logits.dim() == 2 and logits.size(-1) >= 2:
                # take "relevant" logit as last class (common pattern)
                batch_scores = logits[:, -1]
            else:
                batch_scores = logits.reshape(-1)

            scores.extend(batch_scores.detach().float().cpu().tolist())

        return scores


class RAG:
    def __init__(self):
        # Qdrant
        self.qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.collection = os.getenv("QDRANT_COLLECTION", "project_rag_v4_native")
        self.client = QdrantClient(url=self.qdrant_url)

        # Models
        self.embed_model = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v4")
        self.reranker_model = os.getenv("RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual")

        # Params
        self.top_k = env_int("TOP_K", 8)
        self.search_candidates = env_int("SEARCH_CANDIDATES", 20)
        self.embed_batch = env_int("EMBED_BATCH", 8)
        self.rerank_batch = env_int("RERANK_BATCH", 16)
        self.truncate_dim = env_int("EMBED_TRUNCATE_DIM", 1024)

        self.device = resolve_device()
        self.embed_dtype = resolve_dtype("EMBED_DTYPE", "bf16")
        self.rerank_dtype = resolve_dtype("RERANK_DTYPE", "fp16")

        # Init
        self.embedder = JinaV4Embedder(
            model_id=self.embed_model,
            device=self.device,
            dtype=self.embed_dtype,
            truncate_dim=self.truncate_dim,
            batch_size=self.embed_batch
        )
        self.reranker = JinaReranker(
            model_id=self.reranker_model,
            device=self.device,
            dtype=self.rerank_dtype,
            batch_size=self.rerank_batch
        )

        self._ensure_collection()

        print(
            f"[RAG] device={self.device} embed_model={self.embed_model} dim={self.embedder.dim()} "
            f"reranker_model={self.reranker_model} top_k={self.top_k} cand={self.search_candidates} "
            f"truncate_dim={self.truncate_dim} embed_batch={self.embed_batch} rerank_batch={self.rerank_batch}"
        )

    def _ensure_collection(self):
        dim = self.embedder.dim()
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

    # ---- ops ----

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

        vectors = self.embedder.encode([c["text"] for c in chunks])

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

    def _vector_search(self, scope_id: str, query: str, limit: int) -> List[Dict[str, Any]]:
        qv = self.embedder.encode([query])[0].tolist()

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
                "score": float(r.score),
                "text": p.get("text", ""),
                "meta": {k: v for k, v in p.items() if k != "text"},
            })
        return hits

    def _rerank(self, query: str, hits: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not hits:
            return hits

        passages = [(h.get("text") or "") for h in hits]
        scores = self.reranker.score(query, passages)

        ranked: List[Dict[str, Any]] = []
        for h, s in zip(hits, scores):
            h2 = dict(h)
            h2["score"] = float(s)
            ranked.append(h2)

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    def search(self, scope_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        cand = max(top_k, self.search_candidates)
        hits = self._vector_search(scope_id, query, limit=cand)
        return self._rerank(query, hits, top_k=top_k)
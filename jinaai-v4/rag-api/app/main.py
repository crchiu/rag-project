import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .rag import RAG
from .ingest import parse_file
from .schemas import RunRequest, RunResponse, RunResult, RetrievedChunk
from .compose import dominant_section, summarize_section

UPLOAD_DIR = "/data/uploads"

app = FastAPI(title="Project RAG Service (jina-v4 + jina reranker + OCR + scope isolation + GPU)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # prod 請收斂
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAG()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    scope_id: str = Form(...),
    reset_scope: bool = Form(True),
):
    if not scope_id:
        raise HTTPException(status_code=400, detail="scope_id is required")

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    if reset_scope:
        rag.delete_scope(scope_id)

    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="filename is required")

    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".docx"]:
        raise HTTPException(status_code=400, detail="Only PDF/DOCX supported")

    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(await file.read())

    chunk_size = int(os.getenv("CHUNK_SIZE", "900"))
    overlap = int(os.getenv("CHUNK_OVERLAP", "150"))

    try:
        parsed = parse_file(path, filename, chunk_size, overlap)
        doc_id = parsed["doc_id"]
        chunks = parsed["chunks"]
        meta = parsed.get("meta", {})
        indexed = rag.upsert(scope_id=scope_id, doc_id=doc_id, filename=filename, chunks=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parse/Index failed: {e}")

    return {
        "ok": True,
        "scope_id": scope_id,
        "reset_scope": reset_scope,
        "doc_id": doc_id,
        "filename": filename,
        "chunks_indexed": indexed,
        "need_ocr": bool(meta.get("need_ocr", False)),
    }

@app.post("/rag/run", response_model=RunResponse)
async def rag_run(req: RunRequest):
    task = (req.task or "").strip()
    if not req.scope_id:
        raise HTTPException(status_code=400, detail="scope_id is required")
    if not task:
        raise HTTPException(status_code=400, detail="task is required")

    top_k = req.top_k or rag.top_k
    hits = rag.search(scope_id=req.scope_id, query=task, top_k=top_k)
    evidence = [RetrievedChunk(**h) for h in hits]

    output_schema = req.output_schema
    if output_schema == "auto":
        output_schema = "section_summary" if req.section_hint else "qa"

    if req.compose == "never" or output_schema == "evidence_only":
        return RunResponse(
            scope_id=req.scope_id,
            mode="evidence_only",
            task=task,
            section=req.section_hint,
            evidence=evidence,
            result=RunResult(type="none", text="")
        )

    dom = dominant_section(hits, ratio_threshold=0.6)
    section = req.section_hint or dom

    if output_schema == "section_summary" and hits:
        text = summarize_section(hits, section=section)
        return RunResponse(
            scope_id=req.scope_id,
            mode="composed",
            task=task,
            section=section,
            evidence=evidence,
            result=RunResult(type="section_summary", text=text)
        )

    if hits:
        top = hits[0]
        doc_id = top.get("doc_id", "")
        cid = top.get("chunk_id", -1)
        page = (top.get("meta") or {}).get("page")
        cite = f"{doc_id}#{cid}" + (f"(p.{page})" if page else "")
        snippet = (top.get("text") or "").strip().replace("\n", " ")
        snippet = snippet[:280] + ("…" if len(snippet) > 280 else "")
        return RunResponse(
            scope_id=req.scope_id,
            mode="composed",
            task=task,
            section=section,
            evidence=evidence,
            result=RunResult(type="qa", text=f"{snippet}（{cite}）")
        )

    return RunResponse(
        scope_id=req.scope_id,
        mode="evidence_only",
        task=task,
        section=section,
        evidence=[],
        result=RunResult(type="none", text="")
    )

@app.delete("/scopes/{scope_id}")
def clear_scope(scope_id: str):
    rag.delete_scope(scope_id)
    return {"ok": True, "scope_id": scope_id}
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

class RetrievedChunk(BaseModel):
    doc_id: str
    filename: str
    chunk_id: int
    score: float
    text: str
    meta: Dict[str, Any] = {}

class RunRequest(BaseModel):
    scope_id: str = Field(..., description="Isolation scope (e.g. chat_id or user_id:chat_id)")
    task: str = Field(..., description="User question/instruction")
    top_k: Optional[int] = None
    compose: Literal["auto", "always", "never"] = "auto"
    output_schema: Literal["auto", "qa", "section_summary", "evidence_only"] = "auto"
    section_hint: Optional[str] = None

class RunResult(BaseModel):
    type: Literal["qa", "section_summary", "none"]
    text: str = ""

class RunResponse(BaseModel):
    scope_id: str
    mode: Literal["evidence_only", "composed"]
    task: str
    section: Optional[str] = None
    evidence: List[RetrievedChunk]
    result: RunResult
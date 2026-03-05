from typing import List, Dict, Any, Optional
from collections import Counter

def dominant_section(hits: List[Dict[str, Any]], ratio_threshold: float = 0.6) -> Optional[str]:
    secs = []
    for h in hits:
        sec = (h.get("meta") or {}).get("section_path")
        if sec:
            secs.append(sec)
    if not secs:
        return None
    c = Counter(secs)
    sec, cnt = c.most_common(1)[0]
    return sec if (cnt / max(1, len(secs))) >= ratio_threshold else None

def summarize_section(hits: List[Dict[str, Any]], section: Optional[str], max_points: int = 6) -> str:
    if not hits:
        return "（未檢索到足夠佐證內容）"

    picked = []
    for h in hits:
        sec = (h.get("meta") or {}).get("section_path")
        if section and sec and sec != section:
            continue
        t = (h.get("text") or "").strip().replace("\n", " ")
        if not t:
            continue
        doc_id = h.get("doc_id", "")
        cid = h.get("chunk_id", -1)
        page = (h.get("meta") or {}).get("page")
        cite = f"{doc_id}#{cid}" + (f"(p.{page})" if page else "")
        picked.append(f"- {t[:220]}{'…' if len(t) > 220 else ''}（{cite}）")
        if len(picked) >= max_points:
            break

    if not picked:
        top = hits[0]
        doc_id = top.get("doc_id", "")
        cid = top.get("chunk_id", -1)
        page = (top.get("meta") or {}).get("page")
        cite = f"{doc_id}#{cid}" + (f"(p.{page})" if page else "")
        t = (top.get("text") or "").strip().replace("\n", " ")
        return f"{t[:280]}{'…' if len(t) > 280 else ''}（{cite}）"

    return "\n".join(picked)
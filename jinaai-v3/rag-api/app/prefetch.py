import os
import shutil
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

def rm_rf(p: str):
    try:
        shutil.rmtree(p)
    except FileNotFoundError:
        pass

def main():
    hf_home = os.getenv("HF_HOME", "/data/hf_cache")
    os.makedirs(hf_home, exist_ok=True)

    model_id = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v3")

    # --- 0) 先清掉 dynamic modules（避免舊 commit 半套殘留）---
    rm_rf(os.path.join(hf_home, "modules", "transformers_modules", "jinaai", "xlm-roberta-flash-implementation"))
    rm_rf(os.path.join(hf_home, "modules", "transformers_modules", "jinaai", "jina-embeddings-v3"))

    # --- 1) 先確保 hub snapshots 下載完整（force_download 讓它重抓）---
    snapshot_download(repo_id=model_id, cache_dir=hf_home, force_download=True)
    snapshot_download(repo_id="jinaai/xlm-roberta-flash-implementation", cache_dir=hf_home, force_download=True)

    # --- 2) 實際載入一次（觸發 transformers dynamic modules 正確生成到 HF_HOME/modules）---
    SentenceTransformer(model_id, trust_remote_code=True)

    # --- 3) sanity check：確保 flash-implementation 的目錄裡至少有 mha.py / mlp.py ---
    base = os.path.join(hf_home, "modules", "transformers_modules", "jinaai", "xlm-roberta-flash-implementation")
    found = []
    if os.path.isdir(base):
        for root, _, files in os.walk(base):
            if "mha.py" in files:
                found.append(os.path.join(root, "mha.py"))
            if "mlp.py" in files:
                found.append(os.path.join(root, "mlp.py"))
    if not found:
        raise RuntimeError(f"Dynamic modules still missing expected files under: {base}")

    print("✅ Prefetch OK:", model_id)

if __name__ == "__main__":
    main()
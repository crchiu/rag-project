# rag-project
一套 本地端文件檢索式生成（Retrieval-Augmented Generation, RAG）系統，整合 文件解析、OCR、向量搜尋與語意重排序（reranking），用於提升文件查詢與知識檢索的準確性。
系統架構以 FastAPI + Qdrant 向量資料庫 + HuggingFace 模型為核心，並支援 GPU 推論加速，可部署於地端伺服器或私有雲環境。

## 主要功能
 - PDF / DOCX 文件解析
 - OCR 掃描文件辨識
 - Chunk 切分與結構化 metadata
 - 向量搜尋（Vector Retrieval）
 - 語意重排序（Semantic Reranking）
 - Scope 隔離機制（避免不同文件互相干擾）
 - GPU 推論加速
 - REST API（FastAPI）


## Embedding Model：BGE-M3
採用 BAAI / bge-m3 作為主要 embedding 模型。

### 模型特色
BGE-M3（BAAI General Embedding Model v3）
為北京智源研究院（BAAI）推出的多語言 embedding 模型，專為 RAG 與搜尋任務設計。
主要特性：
 - 支援 100+ 語言
 - 支援 長文本 embedding
 - 同時支援
 - Dense Retrieval
 - Multi-Vector Retrieval
 - Sparse Retrieval
 - 向量維度：1024

## Reranker Model
為提升檢索結果準確性，本系統加入 Cross-Encoder Reranker：jinaai/jina-reranker-v2-base-multilingual


## Embedding Model：jina-embeddings-v4
jina-embeddings-v4 為 Jina 最新一代 embedding 模型。

### 主要特性：
 - 基於 Qwen2.5-VL-3B
 - 支援 Multimodal（文字 + 圖像）
 - 支援 32k token 長文本
 - 向量維度：2048
 - 支援 multi-vector retrieval

### 適合：
 - 複雜 RAG
 - 多模態文件搜尋
 - 長文本知識庫

### 然而該模型：
 - 模型規模較大（約 3B+）
 - GPU 需求較高
 - 推論成本較高
因此目前未納入本專案主要架構。


## Embedding Model：jina-embeddings-v3（暫停使用）
本專案原先評估 jina-embeddings-v3 作為 embedding 模型。
在實際部署過程中發現以下問題：
 - 依賴 Remote Code：該模組透過 HuggingFace dynamic module loader 動態下載。
 - Docker 環境穩定性問題：在容器環境中容易出現：FileNotFoundError


## Jina Embedding v4 導入測試紀錄與暫停開發說明
原規劃將原先使用之嵌入模型替換為 Jina Embeddings v4，並搭配 Jina Reranker v2，以期提升多語語意檢索能力與跨語言向量表現。然而在實際導入與系統整合過程中，發現該模型目前仍高度依賴 Hugging Face transformers 套件之 動態載入（trust_remote_code）機制，並透過遠端程式碼引入自訂模型結構（如 qwen2_5_vl 相關模組）。
在 Docker 化部署與 GPU 環境整合過程中，系統多次出現 套件版本相依衝突與 API 不相容問題，包含 SlidingWindowCache 等快取類別於不同 transformers 版本中的定義差異，導致模型初始化時無法正確載入。即使透過更新套件版本、清理快取、重新建置容器等方式嘗試排除，仍存在上游模型程式碼與套件版本耦合過深、且可能隨 Hugging Face 遠端程式碼更新而改變的情形，使系統穩定性與可重現性難以確保。
考量目前專案仍以 RAG 檢索服務穩定運作為優先目標，若持續投入調整底層依賴套件與相容層，將增加系統維運風險並影響開發進度。因此本階段決定 暫停 Jina Embeddings v4 的整合開發，先行維持既有 BGE-M3 + Jina Reranker 的組合架構，以確保檢索品質與系統穩定性。
後續將視 Jina 官方模型與 transformers 套件相容性成熟度，以及相關套件版本穩定後，再評估重新導入 Jina Embeddings v4 之可行性。

## 模型選型結論
目前專案採用：
 - Embedding  : BAAI/bge-m3
 - Reranker   : jina-reranker-v2-base-multilingual
 - Vector DB  : Qdrant

### 執行方式：
docker compose down
docker compose build --no-cache rag-api
docker compose up -d
docker logs -f rag-api
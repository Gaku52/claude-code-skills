# RAG — 検索拡張生成・チャンク分割・リランキング

> RAG (Retrieval-Augmented Generation) は LLM の生成時に外部知識ベースから関連情報を検索・注入する手法であり、ハルシネーション低減と最新情報への対応を同時に実現する、LLM プロダクション運用の中核パターンである。

## この章で学ぶこと

1. **RAG の基本アーキテクチャ** — Indexing、Retrieval、Generation の3フェーズ設計
2. **チャンク分割とベクトル化の最適化** — 分割戦略、Embedding モデル選定、メタデータ活用
3. **高度な検索・リランキング手法** — ハイブリッド検索、リランカー、クエリ変換
4. **Agentic RAG とマルチステップ推論** — ツール統合、自律的検索、ルーティング
5. **本番環境での運用と評価** — モニタリング、キャッシュ、継続的改善パイプライン

---

## 1. RAG の基本アーキテクチャ

```
┌──────────────────────────────────────────────────────────┐
│              RAG パイプライン全体像                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  【インデックス構築 (オフライン)】                          │
│                                                          │
│  文書群 → チャンク分割 → Embedding → ベクトルDB格納       │
│  ┌────┐   ┌──────┐    ┌────────┐   ┌──────────┐        │
│  │ PDF│   │chunk1│    │[0.1,..]│   │Pinecone  │        │
│  │ Web│──▶│chunk2│──▶│[0.3,..]│──▶│Weaviate  │        │
│  │ DB │   │chunk3│    │[0.7,..]│   │pgvector  │        │
│  └────┘   └──────┘    └────────┘   └──────────┘        │
│                                                          │
│  【検索・生成 (オンライン)】                               │
│                                                          │
│  ユーザー                                                 │
│  クエリ ──▶ Embedding ──▶ ベクトル検索                    │
│    │                          │                          │
│    │                   関連チャンク (top-k)               │
│    │                          │                          │
│    └────────┐                 │                          │
│             ▼                 ▼                          │
│         ┌──────────────────────────┐                    │
│         │   プロンプト構築           │                    │
│         │   [システム指示]           │                    │
│         │   [検索結果コンテキスト]    │                    │
│         │   [ユーザー質問]           │                    │
│         └──────────┬───────────────┘                    │
│                    ▼                                     │
│              LLM で回答生成                               │
│              (引用元付き)                                  │
└──────────────────────────────────────────────────────────┘
```

### 1.1 RAG と他の知識注入手法の比較

| 手法 | 知識更新コスト | 精度 | レイテンシ | 適用シーン |
|------|-------------|------|----------|----------|
| RAG | 低 (DB更新のみ) | 高 | 中〜高 | 頻繁に更新される知識、社内文書 |
| ファインチューニング | 高 (再学習必要) | 高 | 低 | 安定した専門知識、文体・形式の変更 |
| プロンプトエンジニアリング | 最低 | 中 | 低 | 少量の固定知識、フォーマット指定 |
| Long Context | 低 | 中〜高 | 高 | 少数の長文書、セッション内の参照 |
| Knowledge Graph + RAG | 中 | 最高 | 高 | 構造化された関係性が重要な場合 |

### 1.2 RAG の成熟度モデル

```
┌─────────────────────────────────────────────────────────────────┐
│                RAG 成熟度モデル (5 段階)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Level 1: Naive RAG                                            │
│  ├── 単純なベクトル検索 + LLM 生成                               │
│  ├── 固定チャンクサイズ、単一 Embedding モデル                     │
│  └── 評価なし、フィードバックループなし                            │
│                                                                 │
│  Level 2: Advanced RAG                                         │
│  ├── ハイブリッド検索、リランキング                               │
│  ├── セマンティックチャンキング、メタデータフィルタリング             │
│  └── 基本的な評価指標 (Recall@k, MRR)                           │
│                                                                 │
│  Level 3: Modular RAG                                          │
│  ├── パイプラインの各コンポーネントが交換可能                      │
│  ├── クエリ変換、Multi-Query、HyDE                              │
│  └── 自動評価 (RAGAS) + A/B テスト                              │
│                                                                 │
│  Level 4: Agentic RAG                                          │
│  ├── LLM がツールとして検索を自律的に実行                        │
│  ├── マルチステップ推論、反復検索                                │
│  └── 自己修正、検索結果の信頼度判定                              │
│                                                                 │
│  Level 5: Adaptive RAG                                         │
│  ├── クエリの複雑さに応じて戦略を動的に選択                      │
│  ├── 継続的学習によるパイプライン最適化                          │
│  └── ユーザーフィードバック駆動の自動改善                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. チャンク分割戦略

### 2.1 分割手法の比較

```python
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

document = "... 長い文書テキスト ..."

# 方法1: 固定長分割 (最もシンプル)
fixed_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n"
)

# 方法2: 再帰的分割 (推奨)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "、", " ", ""]
    # → 段落 > 行 > 文 > 句 > 単語 の順で分割を試みる
)

# 方法3: トークンベース分割
token_splitter = TokenTextSplitter(
    chunk_size=256,     # トークン数で指定
    chunk_overlap=32,
)

chunks = recursive_splitter.split_text(document)
```

### 2.2 セマンティックチャンキング

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# 意味的な区切りで分割 (Embedding の類似度変化点で分割)
semantic_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
)

chunks = semantic_splitter.split_text(document)
# → 意味的に一貫したチャンクが得られる
```

### 2.3 ドキュメントタイプ別チャンク戦略

```python
from langchain.text_splitter import (
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    HTMLHeaderTextSplitter,
)

# Markdown 文書 — ヘッダー階層を尊重した分割
markdown_splitter = MarkdownTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# Python コード — 関数・クラス単位で分割
code_splitter = PythonCodeTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

# HTML — ヘッダータグ (h1, h2, h3) で階層的に分割
html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
)

# テーブルデータ — 行単位で分割し、ヘッダーを各チャンクに付加
def split_table_document(text: str, max_rows_per_chunk: int = 20) -> list[str]:
    """テーブルを含む文書を行単位で分割"""
    lines = text.strip().split("\n")
    header = lines[0] if lines else ""
    separator = lines[1] if len(lines) > 1 and set(lines[1].strip()) <= {"-", "|", " "} else None

    data_start = 2 if separator else 1
    data_lines = lines[data_start:]

    chunks = []
    for i in range(0, len(data_lines), max_rows_per_chunk):
        chunk_lines = data_lines[i:i + max_rows_per_chunk]
        if separator:
            chunk = f"{header}\n{separator}\n" + "\n".join(chunk_lines)
        else:
            chunk = f"{header}\n" + "\n".join(chunk_lines)
        chunks.append(chunk)

    return chunks
```

### 2.4 Parent-Child チャンキング

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Parent: 大きなチャンク (文脈保持用)
# Child: 小さなチャンク (検索精度向上用)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

vectorstore = Chroma(
    collection_name="child_chunks",
    embedding_function=OpenAIEmbeddings(),
)
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# ドキュメントの追加
retriever.add_documents(documents)

# 検索: Child で精密検索 → Parent チャンクを返却
# → 検索精度と文脈の両立
results = retriever.get_relevant_documents("RAGの利点は？")
```

### 2.5 チャンク分割パラメータの最適化

| パラメータ | 推奨範囲 | 小さい場合 | 大きい場合 |
|-----------|---------|-----------|-----------|
| chunk_size | 256-1024 tokens | 精密な検索、文脈不足 | 文脈豊富、ノイズ混入 |
| chunk_overlap | 10-20% of size | 情報欠落リスク | 重複・コスト増 |
| top_k | 3-10 | 情報不足 | コンテキスト長消費 |

### 2.6 メタデータ戦略

```python
from datetime import datetime
from typing import Any

def create_chunk_with_metadata(
    text: str,
    source: str,
    document_title: str,
    section_hierarchy: list[str],
    page_number: int | None = None,
    created_at: datetime | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """チャンクにリッチなメタデータを付与"""
    return {
        "text": text,
        "metadata": {
            # 基本情報
            "source": source,                          # ファイルパス / URL
            "document_title": document_title,          # 文書タイトル
            "section_hierarchy": section_hierarchy,     # ["第1章", "1.2節", "概要"]

            # 位置情報
            "page_number": page_number,
            "chunk_index": None,  # 後で設定

            # 時間情報
            "created_at": (created_at or datetime.now()).isoformat(),
            "indexed_at": datetime.now().isoformat(),

            # 分類情報
            "tags": tags or [],
            "document_type": _detect_doc_type(source),  # pdf, html, md, etc.
            "language": "ja",

            # 検索最適化
            "summary": None,        # LLM で事前生成
            "questions": None,      # チャンクに対する想定質問 (後述)
        }
    }


def enrich_chunk_with_llm(chunk: dict, llm_client) -> dict:
    """LLM でチャンクのメタデータを強化"""

    text = chunk["text"]

    # 1. 要約を生成
    summary_response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"以下のテキストを1文で要約してください:\n\n{text}"
        }],
    )
    chunk["metadata"]["summary"] = summary_response.choices[0].message.content

    # 2. 想定質問を生成 (Hypothetical Questions)
    questions_response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"以下のテキストに対して、ユーザーが尋ねそうな質問を3つ生成してください。1行に1つずつ出力:\n\n{text}"
        }],
    )
    chunk["metadata"]["questions"] = questions_response.choices[0].message.content.strip().split("\n")

    return chunk
```

---

## 3. Embedding とベクトル検索

### 3.1 Embedding モデルの選択

```python
# OpenAI Embedding
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-large",  # 3072次元
    input="RAGとは何ですか？",
    dimensions=1024,  # 次元削減オプション (コスト削減)
)
vector = response.data[0].embedding

# Cohere Embed v3 (多言語に強い)
import cohere
co = cohere.Client("YOUR_API_KEY")

response = co.embed(
    texts=["RAGとは何ですか？"],
    model="embed-multilingual-v3.0",
    input_type="search_query",  # クエリ用とドキュメント用で使い分け
)
vector = response.embeddings[0]

# BGE-M3 (OSS, ローカル実行可能)
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
embeddings = model.encode(
    ["RAGとは何ですか？"],
    return_dense=True,
    return_sparse=True,     # スパースベクトルも同時生成
    return_colbert_vecs=True,  # ColBERT ベクトルも生成
)
dense_vector = embeddings["dense_vecs"][0]
sparse_vector = embeddings["lexical_weights"][0]
```

### 3.2 Embedding モデル比較

| モデル | 次元数 | 日本語 | 料金 ($/1M tokens) | MTEB スコア |
|--------|-------|--------|-------------------|------------|
| text-embedding-3-large | 3072 | 良 | $0.13 | 64.6 |
| text-embedding-3-small | 1536 | 中 | $0.02 | 62.3 |
| Cohere embed-v3 | 1024 | 優 | $0.10 | 64.5 |
| Voyage-3 | 1024 | 良 | $0.06 | 67.1 |
| BGE-M3 (OSS) | 1024 | 優 | 無料 | 65.0 |
| multilingual-e5-large (OSS) | 1024 | 優 | 無料 | 61.5 |

### 3.3 Embedding のバッチ処理と最適化

```python
import asyncio
from typing import AsyncIterator

async def batch_embed(
    texts: list[str],
    client: OpenAI,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    max_concurrent: int = 5,
) -> list[list[float]]:
    """大量テキストを効率的にバッチ Embedding"""

    semaphore = asyncio.Semaphore(max_concurrent)
    all_embeddings = [None] * len(texts)

    async def embed_batch(start_idx: int, batch: list[str]):
        async with semaphore:
            response = await asyncio.to_thread(
                client.embeddings.create,
                model=model,
                input=batch,
            )
            for i, item in enumerate(response.data):
                all_embeddings[start_idx + i] = item.embedding

    tasks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tasks.append(embed_batch(i, batch))

    await asyncio.gather(*tasks)
    return all_embeddings


def deduplicate_by_embedding(
    chunks: list[dict],
    embeddings: list[list[float]],
    similarity_threshold: float = 0.95,
) -> list[dict]:
    """Embedding の類似度で重複チャンクを除去"""
    import numpy as np

    vectors = np.array(embeddings)
    # コサイン類似度行列の計算
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / norms
    similarity_matrix = normalized @ normalized.T

    keep_indices = []
    removed = set()

    for i in range(len(chunks)):
        if i in removed:
            continue
        keep_indices.append(i)
        for j in range(i + 1, len(chunks)):
            if similarity_matrix[i][j] > similarity_threshold:
                removed.add(j)

    return [chunks[i] for i in keep_indices]
```

---

## 4. RAG パイプラインの実装

### 4.1 基本的な RAG 実装

```python
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

client = OpenAI()
qdrant = QdrantClient(":memory:")  # ローカル in-memory

# 1. コレクション作成
qdrant.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# 2. ドキュメントのインデックス
def index_documents(documents: list[dict]):
    points = []
    for i, doc in enumerate(documents):
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=doc["text"],
        ).data[0].embedding

        points.append(PointStruct(
            id=i,
            vector=embedding,
            payload={"text": doc["text"], "source": doc["source"]},
        ))

    qdrant.upsert(collection_name="docs", points=points)

# 3. 検索 + 生成
def rag_query(query: str, top_k: int = 5) -> str:
    # クエリをベクトル化
    query_vector = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    ).data[0].embedding

    # ベクトル検索
    results = qdrant.search(
        collection_name="docs",
        query_vector=query_vector,
        limit=top_k,
    )

    # コンテキスト構築
    context = "\n\n".join([
        f"[出典: {r.payload['source']}]\n{r.payload['text']}"
        for r in results
    ])

    # LLM で回答生成
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """
あなたは質問応答アシスタントです。
提供されたコンテキストのみに基づいて回答してください。
情報が不足する場合は「提供された情報では回答できません」と述べてください。
回答には必ず出典を引用してください。
"""},
            {"role": "user", "content": f"""
コンテキスト:
{context}

質問: {query}
"""},
        ],
    )

    return response.choices[0].message.content
```

### 4.2 ハイブリッド検索

```python
# ベクトル検索 + キーワード検索の組み合わせ
from qdrant_client.models import Filter, FieldCondition, MatchValue

def hybrid_search(query: str, top_k: int = 10) -> list:
    """ベクトル検索 + BM25 のハイブリッド"""

    # 1. ベクトル検索 (意味的類似度)
    query_vector = embed(query)
    vector_results = qdrant.search(
        collection_name="docs",
        query_vector=query_vector,
        limit=top_k,
    )

    # 2. キーワード検索 (BM25 / 全文検索)
    keyword_results = keyword_search(query, top_k=top_k)

    # 3. Reciprocal Rank Fusion (RRF) でスコア統合
    rrf_scores = {}
    k = 60  # RRF 定数

    for rank, result in enumerate(vector_results):
        doc_id = result.id
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)

    for rank, result in enumerate(keyword_results):
        doc_id = result["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # スコア順にソート
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return sorted_ids[:top_k]
```

### 4.3 メタデータフィルタリング付き検索

```python
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    DatetimeRange,
)

def filtered_search(
    query: str,
    department: str | None = None,
    doc_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    tags: list[str] | None = None,
    top_k: int = 10,
) -> list:
    """メタデータフィルタ付きベクトル検索"""

    conditions = []

    if department:
        conditions.append(
            FieldCondition(key="department", match=MatchValue(value=department))
        )

    if doc_type:
        conditions.append(
            FieldCondition(key="document_type", match=MatchValue(value=doc_type))
        )

    if date_from or date_to:
        conditions.append(
            FieldCondition(
                key="created_at",
                range=DatetimeRange(
                    gte=date_from,
                    lte=date_to,
                ),
            )
        )

    if tags:
        for tag in tags:
            conditions.append(
                FieldCondition(key="tags", match=MatchValue(value=tag))
            )

    query_filter = Filter(must=conditions) if conditions else None

    query_vector = embed(query)
    results = qdrant.search(
        collection_name="docs",
        query_vector=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )

    return results


# 使用例: 人事部の PDF ドキュメントのみから検索
results = filtered_search(
    query="有給休暇の申請方法を教えてください",
    department="HR",
    doc_type="pdf",
    date_from="2025-01-01",
    top_k=5,
)
```

### 4.4 ストリーミング RAG

```python
from openai import OpenAI

def rag_query_streaming(query: str, top_k: int = 5):
    """ストリーミング対応の RAG クエリ"""

    client = OpenAI()

    # 1. 検索
    query_vector = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    ).data[0].embedding

    results = qdrant.search(
        collection_name="docs",
        query_vector=query_vector,
        limit=top_k,
    )

    context = "\n\n".join([
        f"[出典: {r.payload['source']}]\n{r.payload['text']}"
        for r in results
    ])

    # 2. ストリーミング生成
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "提供されたコンテキストに基づいて回答してください。"},
            {"role": "user", "content": f"コンテキスト:\n{context}\n\n質問: {query}"},
        ],
        stream=True,
    )

    # 3. チャンクごとに yield
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response += token
            yield {
                "type": "token",
                "content": token,
            }

    # 4. 最後に出典情報を付加
    yield {
        "type": "sources",
        "content": [
            {"source": r.payload["source"], "score": r.score}
            for r in results
        ],
    }
```

---

## 5. リランキング

```
┌──────────────────────────────────────────────────────────┐
│              リランキングパイプライン                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  クエリ                                                   │
│    │                                                     │
│    ▼                                                     │
│  初期検索 (top-50)  ← 高速だが精度は中程度                │
│  (ベクトル検索 / BM25)                                    │
│    │                                                     │
│    ▼                                                     │
│  リランカー (top-50 → top-5)  ← 低速だが高精度            │
│  (Cross-Encoder / Cohere Rerank / LLM)                   │
│    │                                                     │
│    ▼                                                     │
│  上位5件をコンテキストとして LLM に渡す                    │
│                                                          │
│  効果: 検索精度 +10-25% 向上                              │
└──────────────────────────────────────────────────────────┘
```

### 5.1 リランカーの実装

```python
# Cohere Rerank
import cohere

co = cohere.Client("YOUR_API_KEY")

def rerank(query: str, documents: list[str], top_n: int = 5) -> list:
    response = co.rerank(
        model="rerank-multilingual-v3.0",
        query=query,
        documents=documents,
        top_n=top_n,
    )
    return [
        {"index": r.index, "score": r.relevance_score, "text": documents[r.index]}
        for r in response.results
    ]

# Cross-Encoder (OSS)
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

def cross_encoder_rerank(query: str, documents: list[str], top_n: int = 5):
    pairs = [[query, doc] for doc in documents]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, documents), reverse=True)
    return [{"score": s, "text": d} for s, d in ranked[:top_n]]
```

### 5.2 LLM ベースリランキング

```python
import json

def llm_rerank(
    query: str,
    documents: list[dict],
    client: OpenAI,
    top_n: int = 5,
) -> list[dict]:
    """LLM を使った高精度リランキング"""

    doc_list = "\n".join([
        f"[{i}] {doc['text'][:300]}"
        for i, doc in enumerate(documents)
    ])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""以下のクエリに対して、各ドキュメントの関連性を0-10で評価してください。
JSON配列で返してください: [{{"index": 0, "score": 8, "reason": "理由"}}]

クエリ: {query}

ドキュメント:
{doc_list}
""",
        }],
        response_format={"type": "json_object"},
    )

    rankings = json.loads(response.choices[0].message.content)
    scored = sorted(rankings["results"], key=lambda x: x["score"], reverse=True)

    return [
        {**documents[item["index"]], "rerank_score": item["score"]}
        for item in scored[:top_n]
    ]
```

### 5.3 リランカー比較

| リランカー | 精度 | 速度 | コスト | 日本語対応 | 導入容易性 |
|-----------|------|------|--------|----------|----------|
| Cohere Rerank v3 | 高 | 高速 | 有料 ($1/1K queries) | 優秀 | 容易 |
| Cross-Encoder (ms-marco) | 中〜高 | 中 | 無料 | 限定的 | 中 |
| BGE-Reranker-v2 | 高 | 中 | 無料 | 良好 | 中 |
| LLM Rerank (GPT-4o-mini) | 最高 | 低速 | 中 | 優秀 | 容易 |
| FlashRank (軽量) | 中 | 最速 | 無料 | 限定的 | 容易 |

---

## 6. 高度な RAG テクニック

### 6.1 クエリ変換

```python
async def multi_query_rag(original_query: str) -> str:
    """クエリを複数の観点に分解して検索精度を向上"""

    # Step 1: クエリを複数バリエーションに変換
    expansion_prompt = f"""
以下の質問を、異なる観点から3つの検索クエリに言い換えてください。
各クエリは1行ずつ出力してください。

質問: {original_query}
"""
    response = await call_llm(expansion_prompt)
    queries = [original_query] + response.strip().split("\n")

    # Step 2: 各クエリで検索
    all_results = set()
    for query in queries:
        results = vector_search(query, top_k=5)
        all_results.update(results)

    # Step 3: リランクして上位を取得
    reranked = rerank(original_query, list(all_results), top_n=5)

    # Step 4: LLM で回答生成
    return await generate_answer(original_query, reranked)
```

### 6.2 HyDE (Hypothetical Document Embedding)

```python
async def hyde_search(query: str, top_k: int = 5) -> list:
    """仮想ドキュメントを生成してから検索"""

    # Step 1: クエリに対する仮想的な回答を LLM で生成
    hyde_prompt = f"""
以下の質問に対する詳細な回答を生成してください。
実際の知識に基づかなくても構いません。
もっともらしい回答を作成してください。

質問: {query}
"""
    hypothetical_doc = await call_llm(hyde_prompt)

    # Step 2: 仮想ドキュメントを Embedding
    # → 実際の正解ドキュメントに近いベクトルが得られる
    hyde_vector = embed(hypothetical_doc)

    # Step 3: 仮想ドキュメントのベクトルで検索
    results = qdrant.search(
        collection_name="docs",
        query_vector=hyde_vector,
        limit=top_k,
    )

    return results

# HyDE が有効なケース:
# - 専門用語が多いドメイン (クエリとドキュメントの語彙ギャップが大きい)
# - 短いクエリ (情報量が少ない)
# - 質問形式と文書形式のギャップが大きい場合
```

### 6.3 Step-Back Prompting + RAG

```python
async def step_back_rag(query: str) -> str:
    """抽象的な質問に変換してから検索 (Step-Back Prompting)"""

    # Step 1: 具体的な質問を抽象化
    step_back_prompt = f"""
以下の具体的な質問を、より一般的・抽象的な質問に変換してください。
元の質問に答えるために必要な背景知識を得るための質問にしてください。

具体的な質問: {query}
抽象的な質問:
"""
    abstract_query = await call_llm(step_back_prompt)

    # Step 2: 元のクエリと抽象クエリの両方で検索
    original_results = vector_search(query, top_k=3)
    abstract_results = vector_search(abstract_query, top_k=3)

    # Step 3: 両方の結果を統合してコンテキスト構築
    all_context = merge_and_deduplicate(original_results + abstract_results)

    # Step 4: 回答生成
    return await generate_answer(query, all_context)

# 例:
# 元の質問: "GPT-4 の context window は何トークンですか？"
# 抽象化: "大規模言語モデルの context window とは何か、各モデルの比較"
# → より包括的な情報を検索できる
```

### 6.4 CRAG (Corrective RAG)

```python
async def corrective_rag(query: str) -> str:
    """検索結果の品質を自己評価し、必要に応じて修正"""

    # Step 1: 初期検索
    results = vector_search(query, top_k=5)

    # Step 2: 各結果の関連性を LLM で評価
    evaluation_prompt = f"""
以下の検索結果がクエリに対して十分に関連しているか評価してください。
各結果について "relevant", "ambiguous", "irrelevant" のいずれかを返してください。

クエリ: {query}

検索結果:
{format_results(results)}

JSON形式で返してください: [{{"index": 0, "judgment": "relevant"}}]
"""
    evaluations = await call_llm_json(evaluation_prompt)

    # Step 3: 評価結果に基づいてアクション分岐
    relevant_count = sum(1 for e in evaluations if e["judgment"] == "relevant")

    if relevant_count >= 3:
        # 十分な関連結果がある → そのまま回答生成
        context = [results[e["index"]] for e in evaluations if e["judgment"] == "relevant"]
        return await generate_answer(query, context)

    elif relevant_count >= 1:
        # 部分的に関連 → 追加検索で補完
        additional = await web_search(query, top_k=3)  # Web 検索で補完
        combined = [results[e["index"]] for e in evaluations if e["judgment"] != "irrelevant"]
        combined.extend(additional)
        return await generate_answer(query, combined)

    else:
        # 関連結果なし → Web 検索にフォールバック
        web_results = await web_search(query, top_k=5)
        return await generate_answer(query, web_results)
```

### 6.5 Self-RAG (自己反省型RAG)

```python
async def self_rag(query: str) -> str:
    """自己反省トークンで生成品質を制御"""

    # Step 1: 検索が必要か判定
    need_retrieval = await judge_retrieval_need(query)

    if not need_retrieval:
        # 検索不要 → 直接回答
        return await direct_answer(query)

    # Step 2: 検索実行
    results = vector_search(query, top_k=5)

    # Step 3: 各チャンクについて個別に回答候補を生成
    candidates = []
    for result in results:
        # 回答を生成
        answer = await generate_with_single_context(query, result)

        # 忠実度チェック: 回答がコンテキストに忠実か？
        is_faithful = await check_faithfulness(answer, result)

        # 有用性チェック: 回答がクエリに対して有用か？
        is_useful = await check_usefulness(answer, query)

        candidates.append({
            "answer": answer,
            "context": result,
            "is_faithful": is_faithful,
            "is_useful": is_useful,
            "score": (2 if is_faithful else 0) + (1 if is_useful else 0),
        })

    # Step 4: 最もスコアの高い回答を選択
    best = max(candidates, key=lambda c: c["score"])

    if best["score"] == 0:
        return "申し訳ございませんが、信頼性の高い回答を生成できませんでした。"

    return best["answer"]
```

### 6.6 技法比較表

| 技法 | 精度向上 | 複雑度 | レイテンシ | 用途 |
|------|---------|--------|----------|------|
| Naive RAG | 基準 | 低 | 低 | MVP/プロトタイプ |
| ハイブリッド検索 | +10-15% | 中 | 中 | 汎用 |
| リランキング | +10-25% | 中 | 高 | 精度重視 |
| Multi-Query | +5-15% | 中 | 高 | 曖昧なクエリ |
| Parent-Child Chunk | +5-10% | 高 | 中 | 長文書 |
| HyDE | +5-15% | 中 | 高 | 専門ドメイン |
| Step-Back | +5-10% | 中 | 高 | 抽象的な質問 |
| CRAG | +10-20% | 高 | 高 | 信頼性重視 |
| Self-RAG | +15-25% | 最高 | 最高 | 高精度要求 |
| Agentic RAG | +20-30% | 高 | 最高 | 複雑な質問 |

---

## 7. Agentic RAG

### 7.1 アーキテクチャ

```
┌────────────────────────────────────────────────────────────────┐
│                     Agentic RAG                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ユーザークエリ                                                │
│       │                                                       │
│       ▼                                                       │
│  ┌──────────────────┐                                         │
│  │  Router Agent     │ ← クエリの種類を判定                    │
│  │  (クエリ分析)     │                                         │
│  └──────┬───────────┘                                         │
│         │                                                     │
│    ┌────┼────────────────┐                                    │
│    ▼    ▼                ▼                                    │
│  [検索]  [計算]     [Web検索]                                  │
│  ベクトルDB  コード実行   外部API                               │
│    │    │                │                                    │
│    └────┼────────────────┘                                    │
│         ▼                                                     │
│  ┌──────────────────┐                                         │
│  │  Synthesizer      │ ← 結果を統合・回答生成                 │
│  │  (回答生成)       │                                         │
│  └──────┬───────────┘                                         │
│         │                                                     │
│         ▼                                                     │
│  ┌──────────────────┐                                         │
│  │  Self-Check       │ ← 回答の品質を自己評価                 │
│  │  (品質チェック)    │                                        │
│  └──────┬───────────┘                                         │
│         │                                                     │
│    十分か？── No ──▶ 追加検索・修正 (ループ)                   │
│         │                                                     │
│        Yes                                                    │
│         ▼                                                     │
│     最終回答                                                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 7.2 LangGraph による実装

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    query: str
    search_results: list
    web_results: list
    answer: str
    quality_score: float
    iteration: int
    messages: Annotated[list, operator.add]


def route_query(state: AgentState) -> str:
    """クエリの種類に応じてルーティング"""
    query = state["query"]

    # LLM でルーティング判定
    response = llm.invoke(f"""
以下のクエリを分類してください:
- "vector_search": 社内文書の検索が必要
- "web_search": 最新のWeb情報が必要
- "direct": 検索不要で直接回答可能
- "multi_step": 複数の検索ステップが必要

クエリ: {query}
分類:
""")

    return response.strip()


def vector_search_node(state: AgentState) -> AgentState:
    """ベクトル検索ノード"""
    results = vector_search(state["query"], top_k=10)
    reranked = rerank(state["query"], results, top_n=5)
    return {"search_results": reranked}


def web_search_node(state: AgentState) -> AgentState:
    """Web 検索ノード"""
    results = web_search(state["query"], top_k=5)
    return {"web_results": results}


def generate_answer_node(state: AgentState) -> AgentState:
    """回答生成ノード"""
    context = state.get("search_results", []) + state.get("web_results", [])
    answer = generate_answer(state["query"], context)
    return {"answer": answer, "iteration": state.get("iteration", 0) + 1}


def quality_check_node(state: AgentState) -> AgentState:
    """品質チェックノード"""
    score = evaluate_answer_quality(state["query"], state["answer"])
    return {"quality_score": score}


def should_retry(state: AgentState) -> str:
    """リトライ判定"""
    if state["quality_score"] >= 0.8:
        return "end"
    if state["iteration"] >= 3:
        return "end"  # 最大3回でリトライ停止
    return "retry"


# グラフ構築
graph = StateGraph(AgentState)

graph.add_node("route", route_query)
graph.add_node("vector_search", vector_search_node)
graph.add_node("web_search", web_search_node)
graph.add_node("generate", generate_answer_node)
graph.add_node("quality_check", quality_check_node)

graph.set_entry_point("route")

graph.add_conditional_edges("route", route_query, {
    "vector_search": "vector_search",
    "web_search": "web_search",
    "direct": "generate",
    "multi_step": "vector_search",
})

graph.add_edge("vector_search", "generate")
graph.add_edge("web_search", "generate")
graph.add_edge("generate", "quality_check")

graph.add_conditional_edges("quality_check", should_retry, {
    "end": END,
    "retry": "vector_search",
})

app = graph.compile()

# 実行
result = app.invoke({"query": "当社の2024年度の売上推移と業界動向を比較してください"})
```

### 7.3 マルチドキュメント RAG

```python
from dataclasses import dataclass

@dataclass
class DocumentSource:
    name: str
    collection: str
    priority: int
    filters: dict | None = None

class MultiSourceRAG:
    """複数のドキュメントソースを横断して検索"""

    def __init__(self, sources: list[DocumentSource]):
        self.sources = sorted(sources, key=lambda s: s.priority)

    async def search(self, query: str, top_k: int = 10) -> list:
        """全ソースから並列検索"""
        import asyncio

        tasks = [
            self._search_source(query, source, top_k)
            for source in self.sources
        ]
        all_results = await asyncio.gather(*tasks)

        # 全結果をフラット化してリランク
        flat_results = [r for results in all_results for r in results]
        reranked = rerank(query, flat_results, top_n=top_k)

        return reranked

    async def _search_source(self, query: str, source: DocumentSource, top_k: int):
        """個別ソースの検索"""
        query_vector = embed(query)
        results = qdrant.search(
            collection_name=source.collection,
            query_vector=query_vector,
            query_filter=build_filter(source.filters) if source.filters else None,
            limit=top_k,
        )
        # ソース情報を付加
        for r in results:
            r.payload["source_name"] = source.name
            r.payload["source_priority"] = source.priority
        return results


# 使用例
rag = MultiSourceRAG([
    DocumentSource("社内Wiki", "wiki_docs", priority=1),
    DocumentSource("技術ブログ", "blog_posts", priority=2),
    DocumentSource("製品マニュアル", "manuals", priority=1, filters={"status": "active"}),
    DocumentSource("FAQ", "faq_docs", priority=3),
])

results = await rag.search("デプロイ手順を教えてください")
```

---

## 8. 本番運用の設計パターン

### 8.1 キャッシュ戦略

```python
import hashlib
import json
from datetime import datetime, timedelta
from redis import Redis

class RAGCache:
    """RAG クエリのセマンティックキャッシュ"""

    def __init__(self, redis_client: Redis, ttl_hours: int = 24):
        self.redis = redis_client
        self.ttl = timedelta(hours=ttl_hours)

    def _query_hash(self, query: str) -> str:
        """クエリの正規化ハッシュ"""
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> dict | None:
        """完全一致キャッシュ"""
        key = f"rag:exact:{self._query_hash(query)}"
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    def set(self, query: str, result: dict):
        """結果をキャッシュ"""
        key = f"rag:exact:{self._query_hash(query)}"
        self.redis.setex(key, self.ttl, json.dumps(result, ensure_ascii=False))

    async def get_semantic(self, query: str, threshold: float = 0.95) -> dict | None:
        """セマンティックキャッシュ (類似クエリのヒット)"""
        query_vector = embed(query)

        # キャッシュ用ベクトルDBから類似クエリを検索
        results = qdrant.search(
            collection_name="query_cache",
            query_vector=query_vector,
            limit=1,
            score_threshold=threshold,  # 95%以上の類似度
        )

        if results:
            cached_key = f"rag:semantic:{results[0].id}"
            cached = self.redis.get(cached_key)
            if cached:
                return json.loads(cached)

        return None

    async def set_semantic(self, query: str, result: dict):
        """セマンティックキャッシュにも保存"""
        query_vector = embed(query)
        point_id = self._query_hash(query)

        qdrant.upsert(
            collection_name="query_cache",
            points=[PointStruct(
                id=point_id,
                vector=query_vector,
                payload={"query": query, "cached_at": datetime.now().isoformat()},
            )],
        )

        key = f"rag:semantic:{point_id}"
        self.redis.setex(key, self.ttl, json.dumps(result, ensure_ascii=False))
```

### 8.2 インデックス更新パイプライン

```python
from datetime import datetime
from enum import Enum

class UpdateStrategy(Enum):
    FULL_REBUILD = "full_rebuild"
    INCREMENTAL = "incremental"
    UPSERT = "upsert"

class IndexManager:
    """RAG インデックスのライフサイクル管理"""

    def __init__(self, qdrant_client, embedding_client):
        self.qdrant = qdrant_client
        self.embedder = embedding_client
        self.collection = "production_docs"

    async def incremental_update(
        self,
        new_documents: list[dict],
        updated_documents: list[dict],
        deleted_ids: list[str],
    ):
        """差分更新 (最も効率的)"""

        # 1. 削除
        if deleted_ids:
            self.qdrant.delete(
                collection_name=self.collection,
                points_selector=deleted_ids,
            )

        # 2. 更新 (既存ポイントを上書き)
        if updated_documents:
            await self._upsert_documents(updated_documents)

        # 3. 新規追加
        if new_documents:
            await self._upsert_documents(new_documents)

        # 4. キャッシュ無効化
        await self._invalidate_cache()

    async def _upsert_documents(self, documents: list[dict]):
        """ドキュメントをチャンク化して upsert"""
        for doc in documents:
            chunks = chunk_document(doc)
            embeddings = await batch_embed([c["text"] for c in chunks])

            points = [
                PointStruct(
                    id=f"{doc['id']}_{i}",
                    vector=emb,
                    payload={
                        "text": chunk["text"],
                        "source": doc["source"],
                        "document_id": doc["id"],
                        "updated_at": datetime.now().isoformat(),
                        **chunk.get("metadata", {}),
                    },
                )
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
            ]

            self.qdrant.upsert(collection_name=self.collection, points=points)

    async def full_rebuild(self, documents: list[dict]):
        """フルリビルド (非推奨だが確実)"""
        # 新しいコレクションを作成
        temp_collection = f"{self.collection}_temp_{int(datetime.now().timestamp())}"

        self.qdrant.create_collection(
            collection_name=temp_collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

        # 全ドキュメントをインデックス
        for doc in documents:
            chunks = chunk_document(doc)
            embeddings = await batch_embed([c["text"] for c in chunks])
            # ... upsert ...

        # アトミックに切り替え (ダウンタイムなし)
        self.qdrant.update_collection_aliases(
            change_aliases_operations=[
                {"create_alias": {"collection_name": temp_collection, "alias_name": self.collection}},
            ]
        )
```

### 8.3 モニタリングとオブザーバビリティ

```python
import time
from dataclasses import dataclass, field
from typing import Any

@dataclass
class RAGMetrics:
    """RAG パイプラインのメトリクス"""
    query: str
    total_latency_ms: float = 0
    embedding_latency_ms: float = 0
    search_latency_ms: float = 0
    rerank_latency_ms: float = 0
    generation_latency_ms: float = 0
    num_results_retrieved: int = 0
    num_results_after_rerank: int = 0
    top_score: float = 0
    avg_score: float = 0
    tokens_used: int = 0
    cache_hit: bool = False
    error: str | None = None
    metadata: dict = field(default_factory=dict)

class RAGMonitor:
    """RAG パイプラインのモニタリング"""

    def __init__(self, metrics_backend):
        self.backend = metrics_backend

    def track_query(self, metrics: RAGMetrics):
        """メトリクスを記録"""
        self.backend.histogram("rag.latency.total", metrics.total_latency_ms)
        self.backend.histogram("rag.latency.embedding", metrics.embedding_latency_ms)
        self.backend.histogram("rag.latency.search", metrics.search_latency_ms)
        self.backend.histogram("rag.latency.rerank", metrics.rerank_latency_ms)
        self.backend.histogram("rag.latency.generation", metrics.generation_latency_ms)

        self.backend.histogram("rag.results.count", metrics.num_results_retrieved)
        self.backend.histogram("rag.results.top_score", metrics.top_score)

        self.backend.counter("rag.queries.total", 1)
        if metrics.cache_hit:
            self.backend.counter("rag.cache.hits", 1)
        if metrics.error:
            self.backend.counter("rag.errors.total", 1, tags={"error": metrics.error})

    def alert_low_relevance(self, metrics: RAGMetrics, threshold: float = 0.5):
        """低関連性スコアのアラート"""
        if metrics.top_score < threshold:
            self.backend.alert(
                "rag.low_relevance",
                f"Query '{metrics.query[:50]}...' had low relevance score: {metrics.top_score}",
            )


class InstrumentedRAG:
    """計測機能付き RAG パイプライン"""

    def __init__(self, rag_pipeline, monitor: RAGMonitor):
        self.rag = rag_pipeline
        self.monitor = monitor

    async def query(self, query: str, **kwargs) -> dict:
        metrics = RAGMetrics(query=query)
        start = time.time()

        try:
            # Embedding
            t0 = time.time()
            query_vector = await self.rag.embed(query)
            metrics.embedding_latency_ms = (time.time() - t0) * 1000

            # Search
            t0 = time.time()
            results = await self.rag.search(query_vector, **kwargs)
            metrics.search_latency_ms = (time.time() - t0) * 1000
            metrics.num_results_retrieved = len(results)

            if results:
                metrics.top_score = results[0].score
                metrics.avg_score = sum(r.score for r in results) / len(results)

            # Rerank
            t0 = time.time()
            reranked = await self.rag.rerank(query, results)
            metrics.rerank_latency_ms = (time.time() - t0) * 1000
            metrics.num_results_after_rerank = len(reranked)

            # Generate
            t0 = time.time()
            answer = await self.rag.generate(query, reranked)
            metrics.generation_latency_ms = (time.time() - t0) * 1000
            metrics.tokens_used = answer.get("usage", {}).get("total_tokens", 0)

            metrics.total_latency_ms = (time.time() - start) * 1000

            return {"answer": answer["text"], "sources": reranked, "metrics": metrics}

        except Exception as e:
            metrics.error = str(e)
            metrics.total_latency_ms = (time.time() - start) * 1000
            raise
        finally:
            self.monitor.track_query(metrics)
```

---

## 9. RAG の評価

### 9.1 RAGAS フレームワーク

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from datasets import Dataset

# 評価データセットの準備
eval_data = {
    "question": [
        "有給休暇の申請方法は？",
        "リモートワークの規定は？",
    ],
    "answer": [
        "有給休暇はHRシステムから申請できます。上司の承認が必要です。",
        "週3日までリモートワークが可能です。事前申請が必要です。",
    ],
    "contexts": [
        ["有給休暇の申請はHRシステムの「休暇申請」メニューから行います。申請後、直属の上司の承認が必要です。"],
        ["リモートワーク規定: 週3日まで在宅勤務が可能。前日までにシステムで申請。"],
    ],
    "ground_truth": [
        "有給休暇はHRシステムの「休暇申請」メニューから申請し、直属の上司の承認を得る必要がある。",
        "リモートワークは週3日まで可能で、前日までにシステムで事前申請が必要。",
    ],
}

dataset = Dataset.from_dict(eval_data)

# 評価実行
results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,         # 回答がコンテキストに忠実か (0-1)
        answer_relevancy,     # 回答がクエリに関連しているか (0-1)
        context_precision,    # 検索結果の精度 (0-1)
        context_recall,       # 検索結果の網羅性 (0-1)
        answer_correctness,   # 回答の正確性 (0-1)
    ],
)

print(results)
# {'faithfulness': 0.92, 'answer_relevancy': 0.88,
#  'context_precision': 0.85, 'context_recall': 0.90,
#  'answer_correctness': 0.87}
```

### 9.2 検索精度の評価

```python
from typing import Any

def evaluate_retrieval(
    queries: list[str],
    ground_truth_docs: list[list[str]],  # 各クエリの正解ドキュメントID
    retrieval_fn,
    k_values: list[int] = [1, 3, 5, 10],
) -> dict[str, float]:
    """検索精度の評価"""

    metrics = {f"recall@{k}": 0.0 for k in k_values}
    metrics.update({f"precision@{k}": 0.0 for k in k_values})
    metrics["mrr"] = 0.0  # Mean Reciprocal Rank
    metrics["ndcg@10"] = 0.0  # Normalized Discounted Cumulative Gain

    for query, truth in zip(queries, ground_truth_docs):
        results = retrieval_fn(query, top_k=max(k_values))
        retrieved_ids = [r.id for r in results]

        # Recall@k, Precision@k
        for k in k_values:
            top_k_ids = set(retrieved_ids[:k])
            truth_set = set(truth)

            recall = len(top_k_ids & truth_set) / len(truth_set) if truth_set else 0
            precision = len(top_k_ids & truth_set) / k

            metrics[f"recall@{k}"] += recall / len(queries)
            metrics[f"precision@{k}"] += precision / len(queries)

        # MRR
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in truth:
                metrics["mrr"] += 1.0 / rank / len(queries)
                break

    return metrics


# 使用例
results = evaluate_retrieval(
    queries=["有給休暇の申請方法", "リモートワーク規定"],
    ground_truth_docs=[["doc_001", "doc_002"], ["doc_015"]],
    retrieval_fn=lambda q, top_k: vector_search(q, top_k=top_k),
)
print(results)
# {'recall@1': 0.75, 'recall@5': 0.95, 'precision@5': 0.40, 'mrr': 0.83, ...}
```

### 9.3 E2E 回答品質の自動評価

```python
async def auto_evaluate_answer(
    query: str,
    answer: str,
    context: list[str],
    ground_truth: str | None = None,
    evaluator_llm: str = "gpt-4o",
) -> dict[str, Any]:
    """LLM-as-Judge による回答品質の自動評価"""

    evaluation_prompt = f"""以下の質問に対する回答を評価してください。

質問: {query}

回答: {answer}

提供されたコンテキスト:
{chr(10).join(context)}

{"正解: " + ground_truth if ground_truth else ""}

以下の観点で1-5のスコアをつけてください:
1. faithfulness (忠実度): 回答がコンテキストに忠実か？幻覚はないか？
2. relevance (関連性): 回答が質問に適切に答えているか？
3. completeness (完全性): 必要な情報を網羅しているか？
4. conciseness (簡潔性): 冗長でなく、要点を押さえているか？
5. citation_quality (引用品質): 出典が適切に示されているか？

JSON形式で返してください:
{{"faithfulness": 4, "relevance": 5, "completeness": 3, "conciseness": 4, "citation_quality": 3, "overall": 3.8, "feedback": "改善点..."}}
"""

    response = await call_llm(evaluation_prompt, model=evaluator_llm)
    return json.loads(response)
```

---

## 10. ドキュメント前処理パイプライン

### 10.1 マルチフォーマット対応

```python
from pathlib import Path
from abc import ABC, abstractmethod

class DocumentParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> list[dict]:
        pass

class PDFParser(DocumentParser):
    def parse(self, file_path: str) -> list[dict]:
        import pymupdf  # PyMuPDF

        doc = pymupdf.open(file_path)
        pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            # テーブルの抽出
            tables = page.find_tables()
            table_texts = [t.to_pandas().to_markdown() for t in tables]

            pages.append({
                "text": text,
                "tables": table_texts,
                "page_number": page_num + 1,
                "source": file_path,
            })
        return pages

class HTMLParser(DocumentParser):
    def parse(self, file_path: str) -> list[dict]:
        from bs4 import BeautifulSoup

        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        # 不要な要素を除去
        for tag in soup.find_all(["script", "style", "nav", "footer"]):
            tag.decompose()

        return [{
            "text": soup.get_text(separator="\n", strip=True),
            "title": soup.title.string if soup.title else "",
            "source": file_path,
        }]

class MarkdownParser(DocumentParser):
    def parse(self, file_path: str) -> list[dict]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return [{
            "text": content,
            "source": file_path,
        }]

class DocxParser(DocumentParser):
    def parse(self, file_path: str) -> list[dict]:
        from docx import Document

        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        # テーブルも抽出
        tables = []
        for table in doc.tables:
            rows = []
            for row in table.rows:
                cells = [cell.text for cell in row.cells]
                rows.append(cells)
            tables.append(rows)

        return [{
            "text": "\n".join(paragraphs),
            "tables": tables,
            "source": file_path,
        }]


class UniversalDocumentPipeline:
    """あらゆる形式のドキュメントを処理するパイプライン"""

    PARSERS: dict[str, type[DocumentParser]] = {
        ".pdf": PDFParser,
        ".html": HTMLParser,
        ".htm": HTMLParser,
        ".md": MarkdownParser,
        ".docx": DocxParser,
    }

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def process(self, file_path: str) -> list[dict]:
        ext = Path(file_path).suffix.lower()
        parser_cls = self.PARSERS.get(ext)
        if not parser_cls:
            raise ValueError(f"Unsupported format: {ext}")

        # 1. パース
        documents = parser_cls().parse(file_path)

        # 2. チャンク分割
        all_chunks = []
        for doc in documents:
            chunks = self.splitter.split_text(doc["text"])
            for i, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source": doc["source"],
                        "chunk_index": i,
                        "page_number": doc.get("page_number"),
                    },
                })

        return all_chunks
```

---

## 11. アンチパターンとベストプラクティス

### アンチパターン 1: チャンクサイズの不適切な設定

```python
# NG: チャンクが大きすぎる
splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
# → 検索精度が低下 (無関係な情報が大量に混入)
# → コンテキストウィンドウを浪費

# NG: チャンクが小さすぎる
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
# → 文脈が失われる
# → 断片的すぎて LLM が理解できない

# OK: 適切なサイズ + オーバーラップ
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,  # 約12%のオーバーラップ
)
```

### アンチパターン 2: 検索結果の無検証利用

```python
# NG: 検索結果をそのまま全部投入
results = vector_search(query, top_k=20)
context = "\n".join([r.text for r in results])
# → 無関係な結果がノイズとなり回答品質低下

# OK: 関連性スコアの閾値チェック + リランク
results = vector_search(query, top_k=20)
relevant = [r for r in results if r.score > 0.7]  # 閾値フィルタ
reranked = rerank(query, relevant, top_n=5)  # リランク

if not reranked:
    return "関連する情報が見つかりませんでした。"
```

### アンチパターン 3: Embedding モデルの不一致

```python
# NG: インデックス時と検索時で異なる Embedding モデルを使用
# インデックス時
doc_embedding = embed_with_model_a(document)  # モデル A
# 検索時
query_embedding = embed_with_model_b(query)    # モデル B (不一致!)
# → ベクトル空間が異なるため、まともな検索結果が得られない

# OK: 同一モデルを必ず使用、バージョンも固定
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_VERSION = "2024-01-01"

# 設定で一元管理
config = {
    "embedding_model": EMBEDDING_MODEL,
    "embedding_dimensions": 1536,
}
```

### アンチパターン 4: プロンプトの不備

```python
# NG: コンテキストの利用指示が不明確
prompt = f"質問: {query}\n参考: {context}\n回答:"
# → LLM が自身の知識で回答し、ハルシネーション発生

# OK: 明確な指示とガードレール
prompt = f"""あなたは社内文書に基づいて回答するアシスタントです。

重要なルール:
1. 「提供されたコンテキスト」のみに基づいて回答してください
2. コンテキストに含まれない情報で推測しないでください
3. 情報が不十分な場合は「この情報はコンテキストに含まれていません」と明示してください
4. 回答には必ず出典 [出典: ファイル名] を含めてください
5. 矛盾する情報がある場合は、両方の情報を提示してください

コンテキスト:
{context}

質問: {query}

回答:"""
```

### アンチパターン 5: スケーラビリティ無視

```python
# NG: 全ドキュメントをメモリにロード
all_docs = load_all_documents()  # 数GB のドキュメント
embeddings = embed_all(all_docs)  # OOM リスク

# OK: バッチ処理 + ストリーミング
async def index_documents_streaming(doc_paths: list[str], batch_size: int = 50):
    """メモリ効率の良いバッチインデックス"""
    for i in range(0, len(doc_paths), batch_size):
        batch_paths = doc_paths[i:i + batch_size]

        # バッチ単位で処理
        chunks = []
        for path in batch_paths:
            doc_chunks = parse_and_chunk(path)
            chunks.extend(doc_chunks)

        # バッチ Embedding
        embeddings = await batch_embed([c["text"] for c in chunks])

        # バッチ upsert
        points = [
            PointStruct(id=f"doc_{i}_{j}", vector=emb, payload=chunk)
            for j, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        qdrant.upsert(collection_name="docs", points=points)

        # メモリ解放
        del chunks, embeddings, points
```

---

## 12. 実務ユースケース

### 12.1 社内ヘルプデスク RAG

```python
class HelpDeskRAG:
    """社内ヘルプデスク向け RAG システム"""

    def __init__(self):
        self.collections = {
            "hr": "人事関連文書",
            "it": "IT サポート文書",
            "legal": "法務・コンプライアンス文書",
            "general": "一般業務文書",
        }

    async def answer(self, query: str, user_department: str) -> dict:
        """部署コンテキストを考慮した回答"""

        # 1. クエリのカテゴリを自動分類
        category = await self._classify_query(query)

        # 2. 該当コレクションから検索 (部署フィルタ付き)
        results = await filtered_search(
            query=query,
            collection=category,
            filters={"accessible_departments": user_department},
            top_k=5,
        )

        # 3. エスカレーション判定
        if not results or all(r.score < 0.5 for r in results):
            return {
                "answer": "この質問については担当部署にお問い合わせください。",
                "escalation": True,
                "suggested_department": category,
            }

        # 4. 回答生成
        answer = await self._generate_answer(query, results)

        return {
            "answer": answer,
            "sources": [r.payload["source"] for r in results],
            "confidence": results[0].score,
            "escalation": False,
        }
```

### 12.2 コードベース RAG

```python
class CodebaseRAG:
    """コードベースに対する質問応答"""

    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def index_codebase(self):
        """コードベースをインデックス"""
        from tree_sitter_languages import get_parser

        # ファイルごとに処理
        for file_path in glob.glob(f"{self.repo_path}/**/*.py", recursive=True):
            with open(file_path) as f:
                code = f.read()

            # AST ベースの分割 (関数・クラス単位)
            parser = get_parser("python")
            tree = parser.parse(code.encode())

            functions = self._extract_functions(tree, code)
            classes = self._extract_classes(tree, code)

            for item in functions + classes:
                # コードの要約を LLM で生成
                summary = self._summarize_code(item["code"])

                # Embedding はコード + 要約のハイブリッド
                chunk = {
                    "text": f"{summary}\n\n```python\n{item['code']}\n```",
                    "metadata": {
                        "file_path": file_path,
                        "type": item["type"],  # function / class
                        "name": item["name"],
                        "line_start": item["line_start"],
                        "line_end": item["line_end"],
                    },
                }
                self._index_chunk(chunk)

    async def query(self, question: str) -> dict:
        """コードベースに関する質問に回答"""
        results = vector_search(question, top_k=5)

        answer = await generate_answer(
            question,
            results,
            system_prompt="""あなたはコードベースに精通したシニアエンジニアです。
提供されたコードスニペットに基づいて質問に回答してください。
回答にはファイルパスと行番号を含めてください。""",
        )

        return answer
```

### 12.3 法務・コンプライアンス RAG

```python
class LegalRAG:
    """法務文書向け RAG (正確性が最重要)"""

    async def answer(self, query: str) -> dict:
        """法務クエリへの回答 (多重検証付き)"""

        # 1. 複数の検索戦略で候補を収集
        vector_results = await vector_search(query, top_k=10)
        keyword_results = await keyword_search(query, top_k=10)
        hybrid_results = rrf_merge(vector_results, keyword_results)

        # 2. 法務特化リランキング
        reranked = await self._legal_rerank(query, hybrid_results)

        # 3. 回答生成 (保守的なプロンプト)
        answer = await generate_answer(
            query, reranked,
            system_prompt="""あなたは法務アドバイザーです。

重要なルール:
1. 提供された文書のみに基づいて回答してください
2. 法的判断を含む場合は必ず免責事項を付けてください
3. 曖昧な場合は「法務部門への確認をお勧めします」と付記してください
4. 条文番号・規定名を正確に引用してください
5. 複数の解釈が可能な場合はすべての解釈を提示してください""",
        )

        # 4. ファクトチェック (回答とコンテキストの整合性検証)
        fact_check = await self._verify_facts(answer, reranked)

        # 5. 信頼度スコア
        confidence = self._calculate_confidence(reranked, fact_check)

        return {
            "answer": answer,
            "confidence": confidence,
            "sources": [r.payload for r in reranked],
            "fact_check": fact_check,
            "disclaimer": "本回答は参考情報であり、法的助言ではありません。正式な判断には法務部門にご相談ください。",
        }
```

---

## 13. FAQ

### Q1: RAG とファインチューニングはどう使い分ける?

RAG は外部知識の注入に適し、ファインチューニングはモデルの行動パターン (文体、フォーマット、判断基準) の変更に適する。
頻繁に更新される情報 → RAG、安定した専門知識 → ファインチューニング。
実務では RAG + ファインチューニングの併用が最も効果的。

### Q2: ベクトル DB のインデックス更新頻度はどうすべき?

ドキュメントの更新頻度に依存する。リアルタイム性が必要なら差分インデックス更新を実装する。
バッチ更新の場合は日次〜週次が一般的。
古い情報と新しい情報が混在する場合は、メタデータにタイムスタンプを付与して検索時に重み付けする。

### Q3: RAG の評価指標は何を使うべき?

主要指標: (1) Faithfulness — 回答がコンテキストに忠実か、(2) Relevancy — 検索結果が質問に関連しているか、(3) Answer Correctness — 最終回答の正確性。
RAGAS フレームワークや DeepEval で自動評価可能。
Chunk 精度の評価も重要で、正解チャンクが top-k に含まれているかを測定する。

### Q4: 日本語 RAG で特に注意すべき点は?

日本語はトークン効率が英語の 2-3 倍悪い (同じ文章で多くのトークンを消費)。
チャンクサイズは文字数ではなくトークン数で管理すべき。
Embedding モデルは多言語対応 (Cohere, BGE-M3) を選択する。
形態素解析ベースのキーワード検索との併用 (ハイブリッド) が効果的。

### Q5: RAG のレイテンシを改善するには?

主要な改善ポイント: (1) セマンティックキャッシュの導入 (類似クエリの再利用)、(2) Embedding の非同期バッチ処理、(3) ストリーミング生成 (体感レイテンシの改善)、(4) ベクトル DB のインデックス最適化 (HNSW パラメータ調整)、(5) リランカーの選択 (軽量モデルでの妥協点)。典型的な目標: E2E で 2 秒以内。

### Q6: チャンク間の文脈喪失にどう対処する?

(1) Parent-Child チャンキングで検索は小チャンク、LLM への入力は大チャンクを使用。(2) チャンクにセクション階層情報をメタデータとして付与。(3) チャンク先頭に「この文書は○○について述べています」というコンテキストプレフィックスを自動付加。(4) Sliding Window で適度なオーバーラップを確保。

### Q7: マルチモーダル RAG (画像・図表を含む) はどう実装する?

(1) 画像は VLM (GPT-4o, Claude) でテキスト記述に変換してからインデックス。(2) 図表は OCR + テーブル抽出で構造化テキストに変換。(3) CLIP などのマルチモーダル Embedding を使えば画像を直接ベクトル化可能。(4) PDF 内の図はページ画像として切り出し、VLM で説明文を生成してチャンクに含める。

---

## まとめ

| 項目 | 推奨 |
|------|------|
| チャンク分割 | RecursiveCharacterTextSplitter (512 tokens, 12% overlap) |
| Embedding | text-embedding-3-large / Cohere v3 / BGE-M3 |
| ベクトル DB | Qdrant / pgvector (自前) / Pinecone (マネージド) |
| 検索方式 | ハイブリッド検索 (ベクトル + BM25) |
| リランキング | Cohere Rerank v3 / Cross-Encoder |
| クエリ変換 | Multi-Query / HyDE / Step-Back |
| 評価 | RAGAS フレームワーク / LLM-as-Judge |
| キャッシュ | セマンティックキャッシュ (Redis + ベクトル DB) |
| モニタリング | レイテンシ、関連性スコア、キャッシュヒット率 |
| 高度手法 | Agentic RAG (LangGraph) / Self-RAG / CRAG |

---

## 次に読むべきガイド

- [02-function-calling.md](./02-function-calling.md) — RAG と Function Calling の連携
- [03-embeddings.md](./03-embeddings.md) — Embedding の詳細技術
- [../03-infrastructure/01-vector-databases.md](../03-infrastructure/01-vector-databases.md) — ベクトル DB の選定と運用

---

## 参考文献

1. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS 2020
2. Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey," arXiv:2312.10997, 2023
3. LangChain, "RAG Documentation," https://python.langchain.com/docs/tutorials/rag/
4. RAGAS, "Evaluation framework for RAG," https://docs.ragas.io/
5. Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection," ICLR 2024
6. Yan et al., "Corrective Retrieval Augmented Generation," arXiv:2401.15884, 2024
7. Zheng et al., "Step-Back Prompting Enables Reasoning via Abstraction," ICLR 2024
8. Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)," ACL 2023

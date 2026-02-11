# RAG — 検索拡張生成・チャンク分割・リランキング

> RAG (Retrieval-Augmented Generation) は LLM の生成時に外部知識ベースから関連情報を検索・注入する手法であり、ハルシネーション低減と最新情報への対応を同時に実現する、LLM プロダクション運用の中核パターンである。

## この章で学ぶこと

1. **RAG の基本アーキテクチャ** — Indexing、Retrieval、Generation の3フェーズ設計
2. **チャンク分割とベクトル化の最適化** — 分割戦略、Embedding モデル選定、メタデータ活用
3. **高度な検索・リランキング手法** — ハイブリッド検索、リランカー、クエリ変換

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

### 2.3 チャンク分割パラメータの最適化

| パラメータ | 推奨範囲 | 小さい場合 | 大きい場合 |
|-----------|---------|-----------|-----------|
| chunk_size | 256-1024 tokens | 精密な検索、文脈不足 | 文脈豊富、ノイズ混入 |
| chunk_overlap | 10-20% of size | 情報欠落リスク | 重複・コスト増 |
| top_k | 3-10 | 情報不足 | コンテキスト長消費 |

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

### 6.2 技法比較表

| 技法 | 精度向上 | 複雑度 | レイテンシ | 用途 |
|------|---------|--------|----------|------|
| Naive RAG | 基準 | 低 | 低 | MVP/プロトタイプ |
| ハイブリッド検索 | +10-15% | 中 | 中 | 汎用 |
| リランキング | +10-25% | 中 | 高 | 精度重視 |
| Multi-Query | +5-15% | 中 | 高 | 曖昧なクエリ |
| Parent-Child Chunk | +5-10% | 高 | 中 | 長文書 |
| HyDE | +5-15% | 中 | 高 | 専門ドメイン |
| Agentic RAG | +20-30% | 高 | 最高 | 複雑な質問 |

---

## 7. アンチパターン

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

---

## 8. FAQ

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

---

## まとめ

| 項目 | 推奨 |
|------|------|
| チャンク分割 | RecursiveCharacterTextSplitter (512 tokens, 12% overlap) |
| Embedding | text-embedding-3-large / Cohere v3 / BGE-M3 |
| ベクトル DB | Qdrant / pgvector (自前) / Pinecone (マネージド) |
| 検索方式 | ハイブリッド検索 (ベクトル + BM25) |
| リランキング | Cohere Rerank v3 / Cross-Encoder |
| クエリ変換 | Multi-Query / HyDE |
| 評価 | RAGAS フレームワーク |

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

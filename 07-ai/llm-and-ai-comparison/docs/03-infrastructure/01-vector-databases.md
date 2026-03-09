# ベクトル DB — Pinecone・Weaviate・pgvector・Qdrant

> ベクトルデータベースは高次元ベクトルの保存と近似最近傍検索 (ANN) に特化したデータストアであり、RAG・セマンティック検索・レコメンデーションなど Embedding ベースのアプリケーションの基盤インフラである。

## この章で学ぶこと

1. **ベクトル DB の基本原理** — ANN アルゴリズム (HNSW, IVF)、インデックス構造、距離関数
2. **主要ベクトル DB の比較と選定** — Pinecone、Weaviate、Qdrant、pgvector の特徴と使い分け
3. **プロダクション運用** — スケーリング、バックアップ、モニタリング、コスト最適化


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [API 統合 — SDK・ストリーミング・リトライ戦略](./00-api-integration.md) の内容を理解していること

---

## 1. ベクトル検索の基本原理

```
┌──────────────────────────────────────────────────────────┐
│         ANN (近似最近傍) 検索アルゴリズム                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  問題: N個のベクトルから、クエリに最も近い k 個を見つける  │
│  全探索: O(N×d) → 100万件×1024次元では遅すぎる            │
│                                                          │
│  解決策: ANN (Approximate Nearest Neighbor)               │
│                                                          │
│  1. HNSW (Hierarchical Navigable Small World)            │
│     ┌─ Layer 2: ○───○           (粗い探索)              │
│     ├─ Layer 1: ○─○─○─○─○       (中間)                 │
│     └─ Layer 0: ○○○○○○○○○○○○○○ (全ノード)              │
│     → グラフを上から下に辿って近傍を探索                  │
│     → 検索: O(log N), メモリ: O(N×M)                    │
│                                                          │
│  2. IVF (Inverted File Index)                            │
│     → ベクトルをクラスタに分割                            │
│     → クエリに近いクラスタのみ探索                        │
│     → 検索: O(N/K×d), 構築が高速                        │
│                                                          │
│  3. Product Quantization (PQ)                            │
│     → ベクトルを部分空間に分割して量子化                  │
│     → メモリ使用量を大幅削減                              │
│     → 精度は若干低下                                     │
│                                                          │
│  性能目安 (100万ベクトル, 1024次元):                      │
│  ├── 全探索: ~500ms                                      │
│  ├── IVF:    ~10ms                                       │
│  └── HNSW:   ~1ms                                        │
└──────────────────────────────────────────────────────────┘
```

### 1.1 HNSW アルゴリズムの詳細

```
┌──────────────────────────────────────────────────────────┐
│         HNSW (Hierarchical Navigable Small World) 詳解    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  基本アイデア: Skip List + Small World Graph              │
│                                                          │
│  構築フェーズ:                                            │
│  1. 各ノードにランダムなレイヤーを割り当て               │
│     (確率的に上位レイヤーほどノード数が少ない)            │
│  2. 各レイヤーで最近傍のノード M 個とエッジを接続         │
│  3. 上位レイヤーは「高速道路」として機能                  │
│                                                          │
│  検索フェーズ:                                            │
│  1. 最上位レイヤーのエントリポイントから開始              │
│  2. 現在のレイヤーで最近傍に移動 (Greedy Search)          │
│  3. これ以上近づけなくなったら下のレイヤーに降りる        │
│  4. 最下層 (Layer 0) で候補集合を返す                     │
│                                                          │
│  パラメータの影響:                                        │
│  ┌────────────────┬──────────┬──────────┬──────────┐     │
│  │ パラメータ      │ 検索精度 │ 検索速度 │ メモリ   │     │
│  ├────────────────┼──────────┼──────────┼──────────┤     │
│  │ M ↑ (接続数)   │ ↑       │ ↓       │ ↑       │     │
│  │ ef_construct ↑ │ ↑       │ 構築↓   │ -       │     │
│  │ ef_search ↑    │ ↑       │ ↓       │ -       │     │
│  └────────────────┴──────────┴──────────┴──────────┘     │
│                                                          │
│  推奨設定:                                                │
│  ├── M = 16 (バランス) / 32-64 (高精度)                  │
│  ├── ef_construction = 200-400                           │
│  └── ef_search = top_k × 2-4                            │
└──────────────────────────────────────────────────────────┘
```

### 1.2 距離関数の選択

```python
import numpy as np

# ベクトル DB で使用される主な距離関数

def cosine_distance(a, b):
    """コサイン距離: 0 (同一) ～ 2 (正反対)
    テキスト Embedding で最も一般的"""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a, b):
    """ユークリッド距離 (L2): 0 ～ ∞
    画像特徴量やクラスタリングに適する"""
    return np.linalg.norm(a - b)

def dot_product_distance(a, b):
    """内積: 正規化済みベクトルではコサインと等価
    最も計算が速い"""
    return -np.dot(a, b)  # 負にして距離化

# 選択ガイド:
# - テキスト検索 → コサイン距離 or 内積
# - 画像検索 → ユークリッド距離
# - 正規化済み → 内積 (最速)
```

---

## 2. 主要ベクトル DB

### 2.1 Pinecone (マネージドサービス)

```python
from pinecone import Pinecone, ServerlessSpec

# 初期化
pc = Pinecone(api_key="YOUR_API_KEY")

# インデックス作成
pc.create_index(
    name="products",
    dimension=1024,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

index = pc.Index("products")

# ベクトルの挿入 (Upsert)
index.upsert(
    vectors=[
        {
            "id": "prod-001",
            "values": [0.1, 0.2, ...],  # 1024次元
            "metadata": {
                "name": "ワイヤレスイヤホン",
                "category": "electronics",
                "price": 15000,
            },
        },
    ],
    namespace="jp-products",
)

# クエリ (メタデータフィルタ付き)
results = index.query(
    vector=[0.15, 0.22, ...],
    top_k=10,
    filter={
        "category": {"$eq": "electronics"},
        "price": {"$lte": 20000},
    },
    include_metadata=True,
    namespace="jp-products",
)

for match in results.matches:
    print(f"ID: {match.id}, Score: {match.score:.4f}")
    print(f"  {match.metadata}")
```

### 2.2 Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
)

# 接続
client = QdrantClient(url="http://localhost:6333")

# コレクション作成
client.create_collection(
    collection_name="products",
    vectors_config=VectorParams(
        size=1024,
        distance=Distance.COSINE,
    ),
)

# ベクトル挿入
client.upsert(
    collection_name="products",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],
            payload={
                "name": "ワイヤレスイヤホン",
                "category": "electronics",
                "price": 15000,
            },
        ),
    ],
)

# フィルタ付き検索
results = client.search(
    collection_name="products",
    query_vector=[0.15, 0.22, ...],
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="electronics")),
            FieldCondition(key="price", range=Range(lte=20000)),
        ]
    ),
    limit=10,
)
```

### 2.3 Weaviate

```python
import weaviate
from weaviate.classes.config import Property, DataType, Configure

# 接続
client = weaviate.connect_to_local()

# コレクション作成 (スキーマ定義)
collection = client.collections.create(
    name="Product",
    vectorizer_config=Configure.Vectorizer.none(),  # 外部Embedding使用
    properties=[
        Property(name="name", data_type=DataType.TEXT),
        Property(name="category", data_type=DataType.TEXT),
        Property(name="price", data_type=DataType.INT),
    ],
)

# データ挿入
collection.data.insert(
    properties={"name": "ワイヤレスイヤホン", "category": "electronics", "price": 15000},
    vector=[0.1, 0.2, ...],
)

# ハイブリッド検索 (ベクトル + キーワード)
results = collection.query.hybrid(
    query="高音質イヤホン",
    vector=[0.15, 0.22, ...],
    alpha=0.5,  # 0=キーワードのみ, 1=ベクトルのみ
    limit=10,
)

client.close()
```

### 2.4 pgvector (PostgreSQL 拡張)

```python
import psycopg2
from pgvector.psycopg2 import register_vector

# 接続
conn = psycopg2.connect("postgresql://localhost/mydb")
register_vector(conn)

cur = conn.cursor()

# 拡張とテーブル作成
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
cur.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT,
        price INTEGER,
        embedding vector(1024)
    )
""")

# HNSW インデックス作成
cur.execute("""
    CREATE INDEX ON products
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200)
""")

# データ挿入
cur.execute(
    "INSERT INTO products (name, category, price, embedding) VALUES (%s, %s, %s, %s)",
    ("ワイヤレスイヤホン", "electronics", 15000, [0.1, 0.2, ...]),
)

# 検索 (SQL で記述可能)
cur.execute("""
    SELECT name, price, 1 - (embedding <=> %s::vector) AS similarity
    FROM products
    WHERE category = 'electronics' AND price <= 20000
    ORDER BY embedding <=> %s::vector
    LIMIT 10
""", ([0.15, 0.22, ...], [0.15, 0.22, ...]))

for row in cur.fetchall():
    print(f"{row[0]}: ¥{row[1]:,} (類似度: {row[2]:.4f})")

conn.commit()
```

### 2.5 Chroma (軽量・ローカル開発向け)

```python
import chromadb
from chromadb.utils import embedding_functions

# クライアント作成 (インメモリ or 永続化)
client = chromadb.Client()  # インメモリ
# client = chromadb.PersistentClient(path="./chroma_db")  # ディスク永続化

# Embedding 関数の設定 (内蔵 or 外部)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="YOUR_API_KEY",
    model_name="text-embedding-3-small",
)

# コレクション作成
collection = client.create_collection(
    name="documents",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},  # 距離関数を指定
)

# ドキュメント追加 (自動的に Embedding が生成される)
collection.add(
    documents=[
        "Pythonは汎用プログラミング言語です",
        "機械学習はAIの一分野です",
        "寿司は日本料理です",
    ],
    metadatas=[
        {"category": "programming"},
        {"category": "ai"},
        {"category": "food"},
    ],
    ids=["doc1", "doc2", "doc3"],
)

# 検索 (テキストクエリで自動 Embedding)
results = collection.query(
    query_texts=["AIプログラミング"],
    n_results=2,
    where={"category": {"$ne": "food"}},
)
print(results)

# Chroma の利点:
# - セットアップが最も簡単 (pip install chromadb)
# - Embedding 関数を内蔵
# - 開発・プロトタイプに最適
# - LangChain/LlamaIndex と統合が容易
```

---

## 3. ベクトル DB 比較

### 3.1 機能比較

| 機能 | Pinecone | Qdrant | Weaviate | pgvector | Chroma |
|------|----------|--------|----------|---------|--------|
| 提供形態 | マネージドのみ | OSS + Cloud | OSS + Cloud | PostgreSQL拡張 | OSS |
| ANN アルゴリズム | 独自 | HNSW | HNSW | HNSW / IVFFlat | HNSW |
| メタデータフィルタ | 高性能 | 高性能 | 高性能 | SQL (最強) | 基本的 |
| ハイブリッド検索 | N/A | Sparse vector | BM25 内蔵 | pg_trgm 併用 | N/A |
| マルチテナント | Namespace | Collection/Payload | マルチテナント | スキーマ/RLS | Collection |
| 最大ベクトル数 | 無制限 | 数十億 | 数十億 | PostgreSQL依存 | 〜数百万 |
| 言語 | - | Rust | Go | C | Python |
| ライセンス | 商用 | Apache 2.0 | BSD-3 | PostgreSQL | Apache 2.0 |
| 用途 | 本番 | 本番 | 本番 | 本番 | 開発/小規模 |

### 3.2 ユースケース別推奨

| ユースケース | 推奨 DB | 理由 |
|-------------|--------|------|
| スタートアップ MVP | Pinecone | フルマネージド、学習コスト低 |
| 既存 PostgreSQL 環境 | pgvector | インフラ追加不要 |
| 高性能検索サービス | Qdrant | Rust 実装、低レイテンシ |
| ハイブリッド検索重視 | Weaviate | BM25 内蔵 |
| エッジ/組み込み | Qdrant (in-memory) | 軽量、依存少 |
| エンタープライズ | Pinecone / Weaviate Cloud | SLA、サポート |
| プロトタイプ/PoC | Chroma | セットアップ最速 |
| LangChain 統合 | Chroma / pgvector | 公式サポート充実 |

### 3.3 コスト比較

```
┌──────────────────────────────────────────────────────────┐
│          ベクトル DB コスト比較 (100万ベクトル, 1024次元)   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Pinecone Serverless:                                    │
│  ├── ストレージ: $0.33/GB/月 ≈ $1.3/月                   │
│  ├── 読み取り: $8.25/100万 Read Units                    │
│  └── 合計 (低負荷): ~$30-50/月                           │
│                                                          │
│  Qdrant Cloud:                                           │
│  ├── 4GB RAM ノード: ~$50/月〜                            │
│  └── 合計: ~$50-100/月                                   │
│                                                          │
│  Weaviate Cloud:                                         │
│  ├── Sandbox: 無料 (14日)                                │
│  └── Production: ~$50-200/月                             │
│                                                          │
│  pgvector (自前):                                        │
│  ├── RDS db.r6g.large: ~$200/月                          │
│  └── 他のワークロードと共有可能                           │
│                                                          │
│  Qdrant / Weaviate (自前デプロイ):                       │
│  ├── EC2 r6i.large: ~$100/月                             │
│  └── 運用コスト: 要考慮                                  │
│                                                          │
│  Chroma:                                                 │
│  └── 無料 (ローカル実行)                                 │
│                                                          │
│  コスト最適化のポイント:                                  │
│  1. 量子化 (int8): メモリ 1/4、コスト 1/4                │
│  2. 次元削減 (1024→512): メモリ半減                      │
│  3. Serverless (Pinecone): 使った分だけ課金              │
└──────────────────────────────────────────────────────────┘
```

---

## 4. パフォーマンス最適化

```
┌──────────────────────────────────────────────────────────┐
│          HNSW パラメータチューニング                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  構築時パラメータ:                                        │
│  ├── M (接続数): 大きい → 精度↑ メモリ↑ 構築速度↓       │
│  │   推奨: 16 (デフォルト) 〜 64 (高精度)                │
│  └── ef_construction: 大きい → 精度↑ 構築速度↓          │
│      推奨: 128 〜 256                                    │
│                                                          │
│  検索時パラメータ:                                        │
│  └── ef (探索幅): 大きい → 精度↑ 速度↓                  │
│      推奨: top_k の 2-4 倍                               │
│                                                          │
│  トレードオフ:                                            │
│  ┌─────────┬──────────┬──────────┬──────────┐           │
│  │ 設定    │ レイテンシ│ Recall   │ メモリ   │           │
│  ├─────────┼──────────┼──────────┼──────────┤           │
│  │ 低品質  │  0.5ms   │  90%     │  少      │           │
│  │ バランス│  1ms     │  95%     │  中      │           │
│  │ 高品質  │  3ms     │  99%     │  多      │           │
│  └─────────┴──────────┴──────────┴──────────┘           │
└──────────────────────────────────────────────────────────┘
```

### 4.1 バッチ処理の最適化

```python
# Qdrant でのバッチアップサート
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

client = QdrantClient(url="http://localhost:6333")

def batch_upsert(collection: str, data: list[dict], batch_size: int = 100):
    """大量データの効率的な挿入"""
    points = [
        PointStruct(id=d["id"], vector=d["vector"], payload=d["metadata"])
        for d in data
    ]

    # バッチに分割して挿入
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection,
            points=batch,
            wait=False,  # 非同期挿入 (高速化)
        )

    # 最後に同期を待つ
    client.upsert(
        collection_name=collection,
        points=[],
        wait=True,
    )
```

### 4.2 pgvector のインデックスチューニング

```sql
-- pgvector のパフォーマンス最適化

-- 1. HNSW インデックスの作成 (推奨)
CREATE INDEX idx_documents_embedding ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- 検索時の ef を設定
SET hnsw.ef_search = 100;  -- top_k の 2-4 倍

-- 2. IVFFlat インデックス (構築が速い、大量データ向け)
CREATE INDEX idx_documents_ivf ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 1000);  -- sqrt(N) が目安

-- 検索時のプローブ数
SET ivfflat.probes = 10;  -- lists の 1-5%

-- 3. パフォーマンス確認
EXPLAIN ANALYZE
SELECT name, 1 - (embedding <=> $1::vector) AS similarity
FROM documents
ORDER BY embedding <=> $1::vector
LIMIT 10;

-- 4. 部分インデックス (フィルタが多い場合)
CREATE INDEX idx_electronics_embedding ON documents
USING hnsw (embedding vector_cosine_ops)
WHERE category = 'electronics';

-- 5. メモリ設定
SET maintenance_work_mem = '2GB';  -- インデックス構築時のメモリ
SET work_mem = '256MB';             -- 検索時のメモリ
```

### 4.3 Qdrant の Named Vectors (マルチベクトル)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, NamedVector,
)

client = QdrantClient(url="http://localhost:6333")

# マルチベクトルコレクション
client.create_collection(
    collection_name="products_multi",
    vectors_config={
        "title": VectorParams(size=384, distance=Distance.COSINE),
        "description": VectorParams(size=1024, distance=Distance.COSINE),
        "image": VectorParams(size=512, distance=Distance.COSINE),
    },
)

# 複数ベクトルでのデータ挿入
client.upsert(
    collection_name="products_multi",
    points=[
        PointStruct(
            id=1,
            vector={
                "title": [0.1, 0.2, ...],       # タイトル Embedding
                "description": [0.3, 0.4, ...],  # 説明文 Embedding
                "image": [0.5, 0.6, ...],         # 画像 Embedding
            },
            payload={"name": "ワイヤレスイヤホン", "price": 15000},
        ),
    ],
)

# 特定のベクトルフィールドで検索
results = client.search(
    collection_name="products_multi",
    query_vector=NamedVector(
        name="description",  # 説明文ベクトルで検索
        vector=[0.35, 0.45, ...],
    ),
    limit=10,
)
```

---

## 5. RAG パイプラインでの活用

### 5.1 LangChain + pgvector

```python
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Embedding モデル
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# pgvector ストアの作成
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="rag_documents",
    connection="postgresql://localhost/rag_db",
)

# ドキュメントの読み込みと分割
from langchain_community.document_loaders import TextLoader
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(documents)

# ベクトル DB に挿入
vectorstore.add_documents(chunks)

# RAG チェーンの構築
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    chain_type="stuff",
    retriever=retriever,
)

# 質問応答
answer = qa_chain.invoke("RAGの仕組みを説明してください")
print(answer["result"])
```

### 5.2 LlamaIndex + Qdrant

```python
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Qdrant クライアント
qdrant_client = QdrantClient(url="http://localhost:6333")

# ベクトルストア設定
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="llama_docs",
    enable_hybrid=True,  # ハイブリッド検索を有効化
)

# 設定
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ドキュメントの読み込みとインデックス作成
documents = SimpleDirectoryReader("./docs/").load_data()
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,
)

# クエリエンジン
query_engine = index.as_query_engine(
    similarity_top_k=5,
    vector_store_query_mode="hybrid",  # ハイブリッド検索
    alpha=0.5,
)

response = query_engine.query("ベクトルDBの選び方は？")
print(response)
```

---

## 6. スケーリング戦略

```
┌──────────────────────────────────────────────────────────┐
│          ベクトル DB スケーリングパターン                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  データ規模        推奨構成                               │
│  ────────         ────────                               │
│  〜100万件         単一ノード (メモリ内)                   │
│  〜1000万件        単一ノード (SSD + メモリマッピング)     │
│  〜1億件          シャーディング (複数ノード)              │
│  〜10億件以上      分散クラスタ + PQ 圧縮                 │
│                                                          │
│  メモリ見積もり (1024次元, float32):                      │
│  100万件: ~4GB (ベクトルのみ)                             │
│  1000万件: ~40GB                                         │
│  1億件: ~400GB                                           │
│                                                          │
│  コスト削減策:                                            │
│  1. 次元削減 (1024 → 512): メモリ半減                    │
│  2. 量子化 (float32 → int8): メモリ 1/4                  │
│  3. PQ 圧縮: メモリ 1/8-1/16                             │
│  4. ディスクインデックス: メモリ不要 (速度は低下)         │
└──────────────────────────────────────────────────────────┘
```

### 6.1 Qdrant のシャーディングとレプリケーション

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, OptimizersConfigDiff,
    CollectionParams,
)

client = QdrantClient(url="http://localhost:6333")

# シャーディング付きコレクション作成
client.create_collection(
    collection_name="large_scale",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    shard_number=4,           # 4つのシャードに分割
    replication_factor=2,     # 2重レプリケーション
    write_consistency_factor=1,
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=20000,  # インデックス構築の閾値
        memmap_threshold=50000,    # メモリマッピングの閾値
    ),
)
```

---

## 7. モニタリングと運用

### 7.1 Qdrant のメトリクス監視

```python
import requests
from datetime import datetime

def monitor_qdrant(base_url: str = "http://localhost:6333"):
    """Qdrant のヘルスチェックとメトリクス取得"""

    # ヘルスチェック
    health = requests.get(f"{base_url}/healthz").json()
    print(f"ステータス: {health}")

    # コレクション情報
    collections = requests.get(f"{base_url}/collections").json()
    for col in collections["result"]["collections"]:
        name = col["name"]
        info = requests.get(f"{base_url}/collections/{name}").json()
        result = info["result"]

        print(f"\nコレクション: {name}")
        print(f"  ベクトル数: {result['vectors_count']:,}")
        print(f"  ポイント数: {result['points_count']:,}")
        print(f"  セグメント数: {len(result.get('segments', []))}")
        print(f"  ディスク使用: {result.get('disk_data_size', 0) / 1024**2:.1f} MB")
        print(f"  RAM 使用: {result.get('ram_data_size', 0) / 1024**2:.1f} MB")

    # テレメトリ (Prometheus 形式)
    metrics = requests.get(f"{base_url}/metrics").text
    return metrics

# 定期監視
monitor_qdrant()
```

### 7.2 pgvector の監視クエリ

```sql
-- pgvector のパフォーマンス監視

-- 1. インデックスの利用状況
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE indexname LIKE '%embedding%';

-- 2. 遅いクエリの検出
SELECT
    query,
    calls,
    mean_exec_time,
    max_exec_time,
    total_exec_time
FROM pg_stat_statements
WHERE query LIKE '%<=>%'  -- ベクトル検索クエリ
ORDER BY mean_exec_time DESC
LIMIT 10;

-- 3. テーブルサイズの確認
SELECT
    pg_size_pretty(pg_total_relation_size('products')) AS total_size,
    pg_size_pretty(pg_relation_size('products')) AS table_size,
    pg_size_pretty(pg_indexes_size('products')) AS index_size;
```

---

## 8. トラブルシューティング

```
┌──────────────────────────────────────────────────────────┐
│          ベクトル DB トラブルシューティング                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  問題 1: 検索結果の品質が低い                              │
│  ├── 原因 1: Embedding モデルの選択ミス                   │
│  │   └── 解決: BGE-M3 など多言語モデルに変更              │
│  ├── 原因 2: メタデータフィルタが不適切                   │
│  │   └── 解決: Pre-filtering vs Post-filtering を確認     │
│  ├── 原因 3: HNSW パラメータが低すぎる                    │
│  │   └── 解決: ef_search を top_k × 4 に増やす           │
│  └── 原因 4: チャンクサイズが不適切                       │
│      └── 解決: 256-512 トークンに調整                    │
│                                                          │
│  問題 2: 検索速度が遅い                                   │
│  ├── 原因 1: インデックスが作成されていない               │
│  │   └── 解決: HNSW インデックスを作成                    │
│  ├── 原因 2: データがメモリに収まっていない               │
│  │   └── 解決: 量子化 or メモリ増設                      │
│  ├── 原因 3: ef_search が大きすぎる                       │
│  │   └── 解決: 精度とのトレードオフを見直し               │
│  └── 原因 4: メタデータフィルタが非効率                   │
│      └── 解決: ペイロードインデックスを追加               │
│                                                          │
│  問題 3: メモリ不足                                       │
│  ├── 原因 1: 全データがメモリにロードされている           │
│  │   └── 解決: memmap / ディスクモードを有効化            │
│  ├── 原因 2: HNSW の M パラメータが大きすぎる             │
│  │   └── 解決: M=16 に下げる (精度低下は 1-2%)           │
│  └── 原因 3: ベクトル次元数が大きい                       │
│      └── 解決: 次元削減 (3072→1024) or 量子化            │
│                                                          │
│  問題 4: データ整合性の問題                               │
│  ├── 原因: Embedding モデルの変更                         │
│  │   └── 解決: 全ベクトルの再計算が必要                   │
│  └── 原因: 部分更新による不整合                          │
│      └── 解決: バージョニング + 一括再構築                │
└──────────────────────────────────────────────────────────┘
```

---

## 9. アンチパターン

### アンチパターン 1: 全データをメモリに載せようとする

```python
# NG: 1億件を全てメモリに格納
client.create_collection(
    collection_name="huge_data",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    # → 1536 * 4bytes * 100M = 614GB のメモリが必要!
)

# OK: 量子化 + ディスクバックのインデックス
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig

client.create_collection(
    collection_name="huge_data",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(type="int8", always_ram=False),
    ),
    on_disk_payload=True,  # ペイロードもディスクに
)
```

### アンチパターン 2: インデックスなしの大規模検索

```sql
-- NG: pgvector でインデックスなし
SELECT * FROM documents
ORDER BY embedding <=> $1
LIMIT 10;
-- → 100万件でフルスキャン: 数秒かかる

-- OK: HNSW インデックスを作成
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);
-- → 同じクエリが 1ms 以下に
```

### アンチパターン 3: 単一ベクトル空間にすべてを混在

```python
# NG: 異なる種類のコンテンツを同一コレクションに
collection.upsert([
    # 商品説明と技術文書を混在 → 検索精度が低下
    {"id": 1, "vector": product_embedding, "type": "product"},
    {"id": 2, "vector": doc_embedding, "type": "documentation"},
])

# OK: コンテンツタイプ別にコレクションを分割
product_collection = client.create_collection("products", ...)
doc_collection = client.create_collection("documentation", ...)
# または Named Vectors で分離
```

### アンチパターン 4: バックアップ戦略なしでの運用

```python
# NG: ベクトル DB のバックアップを取っていない
# → 障害時にすべてのベクトルを再計算する必要がある

# OK: 定期的なスナップショット取得
# Qdrant の場合
snapshot = requests.post(
    f"{base_url}/collections/products/snapshots"
).json()

# pgvector の場合
# 通常の pg_dump がそのまま使える
# pg_dump -Fc mydb > mydb_backup.dump

# 元データの保持が最重要
# ベクトルは元テキスト + Embedding モデルから再構築可能
```

---

## 10. ベストプラクティス

### 10.1 設計チェックリスト

```
┌──────────────────────────────────────────────────────────┐
│          ベクトル DB 設計チェックリスト                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  □ データ設計                                            │
│    ├── ベクトル次元数: 1024 推奨                          │
│    ├── 距離関数: コサイン (テキスト) / L2 (画像)          │
│    ├── メタデータ: 検索フィルタに使うフィールドを定義     │
│    └── ID 体系: UUID or 意味のある ID                     │
│                                                          │
│  □ インデックス設計                                      │
│    ├── HNSW: M=16, ef_construction=200                   │
│    ├── メタデータインデックス: フィルタ対象に作成          │
│    └── ef_search: top_k × 2-4                            │
│                                                          │
│  □ 運用設計                                              │
│    ├── バックアップ: 定期スナップショット                 │
│    ├── モニタリング: レイテンシ、メモリ、Recall            │
│    ├── スケーリング: 水平分割の閾値を決めておく           │
│    └── 更新戦略: Embedding モデル変更時の再計算手順       │
│                                                          │
│  □ セキュリティ                                          │
│    ├── 認証: API キー or mTLS                             │
│    ├── 暗号化: at-rest + in-transit                       │
│    └── アクセス制御: テナント分離                         │
└──────────────────────────────────────────────────────────┘
```


---

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## 実務での適用シナリオ

### シナリオ1: スタートアップでのMVP開発

**状況:** 限られたリソースで素早くプロダクトをリリースする必要がある

**アプローチ:**
- シンプルなアーキテクチャを選択
- 必要最小限の機能に集中
- 自動テストはクリティカルパスのみ
- モニタリングは早期から導入

**学んだ教訓:**
- 完璧を求めすぎない（YAGNI原則）
- ユーザーフィードバックを早期に取得
- 技術的負債は意識的に管理する

### シナリオ2: レガシーシステムのモダナイゼーション

**状況:** 10年以上運用されているシステムを段階的に刷新する

**アプローチ:**
- Strangler Fig パターンで段階的に移行
- 既存のテストがない場合はCharacterization Testを先に作成
- APIゲートウェイで新旧システムを共存
- データ移行は段階的に実施

| フェーズ | 作業内容 | 期間目安 | リスク |
|---------|---------|---------|--------|
| 1. 調査 | 現状分析、依存関係の把握 | 2-4週間 | 低 |
| 2. 基盤 | CI/CD構築、テスト環境 | 4-6週間 | 低 |
| 3. 移行開始 | 周辺機能から順次移行 | 3-6ヶ月 | 中 |
| 4. コア移行 | 中核機能の移行 | 6-12ヶ月 | 高 |
| 5. 完了 | 旧システム廃止 | 2-4週間 | 中 |

### シナリオ3: 大規模チームでの開発

**状況:** 50人以上のエンジニアが同一プロダクトを開発する

**アプローチ:**
- ドメイン駆動設計で境界を明確化
- チームごとにオーナーシップを設定
- 共通ライブラリはInner Source方式で管理
- APIファーストで設計し、チーム間の依存を最小化

```python
# チーム間のAPI契約定義
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """チーム間のAPI契約"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # レスポンスタイムSLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """SLA準拠の確認"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """OpenAPI形式で出力"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# 使用例
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### シナリオ4: パフォーマンスクリティカルなシステム

**状況:** ミリ秒単位のレスポンスが求められるシステム

**最適化ポイント:**
1. キャッシュ戦略（L1: インメモリ、L2: Redis、L3: CDN）
2. 非同期処理の活用
3. コネクションプーリング
4. クエリ最適化とインデックス設計

| 最適化手法 | 効果 | 実装コスト | 適用場面 |
|-----------|------|-----------|---------|
| インメモリキャッシュ | 高 | 低 | 頻繁にアクセスされるデータ |
| CDN | 高 | 低 | 静的コンテンツ |
| 非同期処理 | 中 | 中 | I/O待ちが多い処理 |
| DB最適化 | 高 | 高 | クエリが遅い場合 |
| コード最適化 | 低-中 | 高 | CPU律速の場合 |

---

## チーム開発での活用

### コードレビューのチェックリスト

このトピックに関連するコードレビューで確認すべきポイント:

- [ ] 命名規則が一貫しているか
- [ ] エラーハンドリングが適切か
- [ ] テストカバレッジは十分か
- [ ] パフォーマンスへの影響はないか
- [ ] セキュリティ上の問題はないか
- [ ] ドキュメントは更新されているか

### ナレッジ共有のベストプラクティス

| 方法 | 頻度 | 対象 | 効果 |
|------|------|------|------|
| ペアプログラミング | 随時 | 複雑なタスク | 即時のフィードバック |
| テックトーク | 週1回 | チーム全体 | 知識の水平展開 |
| ADR (設計記録) | 都度 | 将来のメンバー | 意思決定の透明性 |
| 振り返り | 2週間ごと | チーム全体 | 継続的改善 |
| モブプログラミング | 月1回 | 重要な設計 | 合意形成 |

### 技術的負債の管理

```
優先度マトリクス:

        影響度 高
          │
    ┌─────┼─────┐
    │ 計画 │ 即座 │
    │ 的に │ に   │
    │ 対応 │ 対応 │
    ├─────┼─────┤
    │ 記録 │ 次の │
    │ のみ │ Sprint│
    │     │ で   │
    └─────┼─────┘
          │
        影響度 低
    発生頻度 低  発生頻度 高
```
---

## 11. FAQ

### Q1: pgvector と専用ベクトル DB のどちらを選ぶべき?

既に PostgreSQL を使っていて、ベクトル数が 1000 万件以下なら pgvector が最も合理的。
追加インフラ不要、SQL でフィルタリングできる、トランザクション対応という利点がある。
1000 万件超、サブミリ秒のレイテンシ要件、高度な ANN チューニングが必要なら専用 DB を検討。

### Q2: ベクトル DB のバックアップと障害復旧は?

Pinecone はマネージドなので自動バックアップ。Qdrant/Weaviate はスナップショット API でバックアップ可能。
pgvector は通常の PostgreSQL バックアップ (pg_dump) がそのまま使える。
ベクトルデータは元テキスト + Embedding モデルから再構築できるため、元データのバックアップが最重要。

### Q3: マルチモーダル検索 (テキスト+画像) はどう実装する?

CLIP などのマルチモーダル Embedding で画像とテキストを同一ベクトル空間に埋め込む。
あるいは、テキストベクトルと画像ベクトルを別々のフィールドに格納し、
Named Vectors (Qdrant) や複数ベクトルフィールド (Weaviate) で管理する。

### Q4: ベクトル DB のデータ更新はどうするのが最適?

リアルタイム更新: Upsert API で個別更新。遅延は最小だがスループットは低い。
バッチ更新: 定期的に一括 Upsert。スループットが高い。
全量再構築: Embedding モデル変更時。新コレクション作成 → 切り替え (Blue-Green)。

### Q5: 検索精度 (Recall) はどう測定・改善する?

測定: 全探索結果を Ground Truth とし、ANN 結果との一致率を計算。
改善: ef_search を上げる (最も効果的)、M を上げる (構築時)、量子化パラメータを調整。
目標: Recall 95% 以上をレイテンシ制約内で達成。

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## まとめ

| 項目 | 推奨 |
|------|------|
| マネージド推奨 | Pinecone (最も手軽) |
| OSS 推奨 | Qdrant (Rust, 高速) |
| 既存 PG 環境 | pgvector (追加インフラ不要) |
| ハイブリッド検索 | Weaviate (BM25 内蔵) |
| プロトタイプ | Chroma (セットアップ最速) |
| ANN アルゴリズム | HNSW (精度と速度のバランス最良) |
| 推奨パラメータ | M=16, ef_construction=200, ef=top_k×4 |
| スケーリング | 量子化 + シャーディング |
| コスト最適化 | 次元削減 + 量子化 + Serverless |

---

## 次に読むべきガイド

- [02-local-llm.md](./02-local-llm.md) — ローカル LLM とベクトル DB の組み合わせ
- [../02-applications/01-rag.md](../02-applications/01-rag.md) — RAG パイプラインでのベクトル DB 活用
- [../02-applications/03-embeddings.md](../02-applications/03-embeddings.md) — Embedding モデルの選定

---

## 参考文献

1. Pinecone, "Documentation," https://docs.pinecone.io/
2. Qdrant, "Documentation," https://qdrant.tech/documentation/
3. Weaviate, "Documentation," https://weaviate.io/developers/weaviate
4. pgvector, "GitHub Repository," https://github.com/pgvector/pgvector
5. Malkov & Yashunin, "Efficient and Robust Approximate Nearest Neighbor using Hierarchical Navigable Small World Graphs," IEEE TPAMI, 2020
6. Chroma, "Documentation," https://docs.trychroma.com/

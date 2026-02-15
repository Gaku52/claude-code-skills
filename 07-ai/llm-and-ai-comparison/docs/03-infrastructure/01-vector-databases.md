# ベクトル DB — Pinecone・Weaviate・pgvector・Qdrant

> ベクトルデータベースは高次元ベクトルの保存と近似最近傍検索 (ANN) に特化したデータストアであり、RAG・セマンティック検索・レコメンデーションなど Embedding ベースのアプリケーションの基盤インフラである。

## この章で学ぶこと

1. **ベクトル DB の基本原理** — ANN アルゴリズム (HNSW, IVF)、インデックス構造、距離関数
2. **主要ベクトル DB の比較と選定** — Pinecone、Weaviate、Qdrant、pgvector の特徴と使い分け
3. **プロダクション運用** — スケーリング、バックアップ、モニタリング、コスト最適化

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

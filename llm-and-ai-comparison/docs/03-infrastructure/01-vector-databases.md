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

---

## 3. ベクトル DB 比較

### 3.1 機能比較

| 機能 | Pinecone | Qdrant | Weaviate | pgvector |
|------|----------|--------|----------|---------|
| 提供形態 | マネージドのみ | OSS + Cloud | OSS + Cloud | PostgreSQL拡張 |
| ANN アルゴリズム | 独自 | HNSW | HNSW | HNSW / IVFFlat |
| メタデータフィルタ | 高性能 | 高性能 | 高性能 | SQL (最強) |
| ハイブリッド検索 | N/A | Sparse vector | BM25 内蔵 | pg_trgm 併用 |
| マルチテナント | Namespace | Collection/Payload | マルチテナント | スキーマ/RLS |
| 最大ベクトル数 | 無制限 | 数十億 | 数十億 | PostgreSQL依存 |
| 言語 | - | Rust | Go | C |
| ライセンス | 商用 | Apache 2.0 | BSD-3 | PostgreSQL |

### 3.2 ユースケース別推奨

| ユースケース | 推奨 DB | 理由 |
|-------------|--------|------|
| スタートアップ MVP | Pinecone | フルマネージド、学習コスト低 |
| 既存 PostgreSQL 環境 | pgvector | インフラ追加不要 |
| 高性能検索サービス | Qdrant | Rust 実装、低レイテンシ |
| ハイブリッド検索重視 | Weaviate | BM25 内蔵 |
| エッジ/組み込み | Qdrant (in-memory) | 軽量、依存少 |
| エンタープライズ | Pinecone / Weaviate Cloud | SLA、サポート |

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

---

## 5. スケーリング戦略

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

---

## 6. アンチパターン

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

---

## 7. FAQ

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

---

## まとめ

| 項目 | 推奨 |
|------|------|
| マネージド推奨 | Pinecone (最も手軽) |
| OSS 推奨 | Qdrant (Rust, 高速) |
| 既存 PG 環境 | pgvector (追加インフラ不要) |
| ハイブリッド検索 | Weaviate (BM25 内蔵) |
| ANN アルゴリズム | HNSW (精度と速度のバランス最良) |
| 推奨パラメータ | M=16, ef_construction=200, ef=top_k×4 |
| スケーリング | 量子化 + シャーディング |

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

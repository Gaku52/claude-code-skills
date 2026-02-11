# Embeddings — ベクトル表現・類似度検索・クラスタリング

> Embedding はテキスト・画像等のデータを高次元ベクトル空間に射影する技術であり、意味的類似度の計算、検索、分類、クラスタリングなど LLM エコシステムの基盤を支える数学的表現手法である。

## この章で学ぶこと

1. **Embedding の数学的基礎** — ベクトル空間、距離関数、次元削減の原理
2. **Embedding モデルの選定と利用** — API・OSS モデルの比較、多言語対応、ファインチューニング
3. **実践的な応用パターン** — セマンティック検索、クラスタリング、異常検知、分類

---

## 1. Embedding の基本概念

```
┌──────────────────────────────────────────────────────────┐
│           Embedding の直感的理解                           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  テキスト              ベクトル空間                        │
│                                                          │
│  "猫が眠る"    ──▶    [0.82, 0.15, -0.33, ...]          │
│  "犬が寝る"    ──▶    [0.79, 0.18, -0.31, ...]  ← 近い │
│  "経済が成長"  ──▶    [-0.21, 0.67, 0.44, ...]  ← 遠い │
│                                                          │
│               y                                          │
│               ^    犬が寝る                               │
│               |   * *猫が眠る                             │
│               |                                          │
│               |                                          │
│               |          *経済が成長                      │
│               |                                          │
│               └──────────────────▶ x                     │
│                                                          │
│  意味が近い → ベクトルが近い (コサイン類似度が高い)       │
│  意味が遠い → ベクトルが遠い (コサイン類似度が低い)       │
└──────────────────────────────────────────────────────────┘
```

### 1.1 距離関数

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """コサイン類似度: -1 ～ 1 (1に近いほど類似)"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """ユークリッド距離: 0 ～ ∞ (0に近いほど類似)"""
    return np.linalg.norm(a - b)

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """内積: 正規化済みベクトルではコサイン類似度と等価"""
    return np.dot(a, b)

# 使用例
vec_cat  = np.array([0.82, 0.15, -0.33])
vec_dog  = np.array([0.79, 0.18, -0.31])
vec_econ = np.array([-0.21, 0.67, 0.44])

print(f"猫-犬: {cosine_similarity(vec_cat, vec_dog):.4f}")   # → 0.9987 (高い)
print(f"猫-経済: {cosine_similarity(vec_cat, vec_econ):.4f}") # → -0.2341 (低い)
```

---

## 2. Embedding モデルの利用

### 2.1 OpenAI Embedding API

```python
from openai import OpenAI

client = OpenAI()

# 単一テキストの埋め込み
response = client.embeddings.create(
    model="text-embedding-3-large",
    input="機械学習とは何ですか？",
    dimensions=1024,  # 次元削減 (3072 → 1024): コスト・速度改善
)
embedding = response.data[0].embedding
print(f"次元数: {len(embedding)}")  # 1024

# バッチ処理 (最大2048テキスト)
texts = [
    "Pythonは汎用プログラミング言語です",
    "機械学習は人工知能の一分野です",
    "寿司は日本の伝統的な料理です",
]
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts,
)
embeddings = [d.embedding for d in response.data]
```

### 2.2 OSS Embedding モデル (Sentence Transformers)

```python
from sentence_transformers import SentenceTransformer

# BGE-M3: 多言語対応の高性能OSSモデル
model = SentenceTransformer("BAAI/bge-m3")

texts = [
    "機械学習の基礎を学ぶ",
    "ディープラーニング入門",
    "今日のランチは何にする？",
]

embeddings = model.encode(texts, normalize_embeddings=True)
print(f"形状: {embeddings.shape}")  # (3, 1024)

# 類似度行列
from sentence_transformers.util import cos_sim
similarity_matrix = cos_sim(embeddings, embeddings)
print(similarity_matrix)
```

### 2.3 Cohere Embed v3

```python
import cohere

co = cohere.Client("YOUR_API_KEY")

# input_type が重要: クエリとドキュメントで異なる埋め込みを生成
query_embed = co.embed(
    texts=["日本の人口は？"],
    model="embed-multilingual-v3.0",
    input_type="search_query",       # 検索クエリ用
).embeddings[0]

doc_embeds = co.embed(
    texts=[
        "日本の人口は約1億2500万人です。",
        "東京は日本の首都です。",
    ],
    model="embed-multilingual-v3.0",
    input_type="search_document",    # 文書用
).embeddings
```

---

## 3. Embedding モデル比較

### 3.1 性能・スペック比較

| モデル | 次元数 | 最大入力 | 日本語 | MTEB | 料金 | ライセンス |
|--------|-------|---------|--------|------|------|-----------|
| text-embedding-3-large | 3072 | 8191 tok | 良 | 64.6 | $0.13/1M | API |
| text-embedding-3-small | 1536 | 8191 tok | 中 | 62.3 | $0.02/1M | API |
| Cohere embed-v3 | 1024 | 512 tok | 優 | 64.5 | $0.10/1M | API |
| Voyage-3 | 1024 | 32000 tok | 良 | 67.1 | $0.06/1M | API |
| BGE-M3 | 1024 | 8192 tok | 優 | 65.0 | 無料 | MIT |
| multilingual-e5-large | 1024 | 512 tok | 優 | 61.5 | 無料 | MIT |
| nomic-embed-text | 768 | 8192 tok | 中 | 62.4 | 無料 | Apache 2.0 |

### 3.2 用途別推奨

| 用途 | 推奨モデル | 理由 |
|------|-----------|------|
| 日本語検索 | BGE-M3 / Cohere v3 | 多言語性能最高 |
| 低コスト大量処理 | text-embedding-3-small | 最安 + 十分な品質 |
| 最高精度 | Voyage-3 / BGE-M3 | MTEB 上位 |
| 長文書対応 | Voyage-3 | 32K トークン対応 |
| オンプレミス | BGE-M3 | OSS + 高性能 |
| エッジデバイス | nomic-embed-text | 軽量 768次元 |

---

## 4. 実践的な応用パターン

### 4.1 セマンティック検索

```python
import numpy as np
from openai import OpenAI

client = OpenAI()

def semantic_search(query: str, documents: list[str], top_k: int = 5) -> list:
    """セマンティック検索の基本実装"""

    # 全テキストを一括で埋め込み
    all_texts = [query] + documents
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=all_texts,
    )

    query_vec = np.array(response.data[0].embedding)
    doc_vecs = np.array([d.embedding for d in response.data[1:]])

    # コサイン類似度を計算
    similarities = np.dot(doc_vecs, query_vec) / (
        np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec)
    )

    # 上位 k 件を返す
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [
        {"text": documents[i], "score": float(similarities[i])}
        for i in top_indices
    ]

# 使用例
docs = [
    "Pythonは汎用的なプログラミング言語で、機械学習で広く使われています",
    "JavaScriptはWebブラウザで動作するスクリプト言語です",
    "深層学習はニューラルネットワークを多層にした機械学習手法です",
    "寿司は酢飯と魚介類を組み合わせた日本料理です",
]
results = semantic_search("AIに使われる言語は？", docs, top_k=2)
for r in results:
    print(f"[{r['score']:.4f}] {r['text']}")
```

### 4.2 クラスタリング

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def cluster_texts(texts: list[str], n_clusters: int = 3):
    """テキストをクラスタリング"""
    # Embedding 取得
    embeddings = get_embeddings(texts)  # [N, dim]

    # K-means クラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # PCA で2次元に次元削減して可視化
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    # クラスタごとにグループ化
    clusters = {}
    for text, label in zip(texts, labels):
        clusters.setdefault(int(label), []).append(text)

    return clusters

# 使用例
texts = [
    "Pythonで機械学習", "TensorFlowの使い方", "PyTorchチュートリアル",
    "東京の観光地", "京都の寺院", "大阪の食べ歩き",
    "確定申告の方法", "住民税の計算", "年末調整の手順",
]
clusters = cluster_texts(texts, n_clusters=3)
for label, items in clusters.items():
    print(f"\nクラスタ {label}:")
    for item in items:
        print(f"  - {item}")
```

### 4.3 異常検知

```python
import numpy as np

def detect_anomalies(
    reference_texts: list[str],
    test_texts: list[str],
    threshold: float = 0.5,
) -> list[dict]:
    """Embedding ベースの異常検知"""

    ref_embeddings = np.array(get_embeddings(reference_texts))
    test_embeddings = np.array(get_embeddings(test_texts))

    # 参照テキストの重心 (セントロイド) を計算
    centroid = ref_embeddings.mean(axis=0)
    centroid /= np.linalg.norm(centroid)  # 正規化

    results = []
    for text, emb in zip(test_texts, test_embeddings):
        emb_norm = emb / np.linalg.norm(emb)
        similarity = np.dot(centroid, emb_norm)
        is_anomaly = similarity < threshold
        results.append({
            "text": text,
            "similarity": float(similarity),
            "is_anomaly": is_anomaly,
        })

    return results
```

### 4.4 テキスト分類 (ゼロショット)

```python
def zero_shot_classify(text: str, categories: list[str]) -> dict:
    """Embedding ベースのゼロショット分類"""

    # テキストとカテゴリ全てを埋め込み
    all_inputs = [text] + categories
    embeddings = get_embeddings(all_inputs)

    text_emb = np.array(embeddings[0])
    cat_embs = np.array(embeddings[1:])

    # 各カテゴリとの類似度を計算
    similarities = np.dot(cat_embs, text_emb) / (
        np.linalg.norm(cat_embs, axis=1) * np.linalg.norm(text_emb)
    )

    # ソフトマックスで確率に変換
    exp_sim = np.exp(similarities * 10)  # temperature=0.1
    probs = exp_sim / exp_sim.sum()

    return {cat: float(prob) for cat, prob in zip(categories, probs)}

# 使用例
result = zero_shot_classify(
    "新しいGPUが発表され、AI処理が3倍高速化",
    ["テクノロジー", "スポーツ", "政治", "エンタメ"]
)
# → {'テクノロジー': 0.89, 'スポーツ': 0.03, '政治': 0.04, 'エンタメ': 0.04}
```

---

## 5. 次元削減とパフォーマンス最適化

```
┌──────────────────────────────────────────────────────────┐
│          Embedding 最適化のトレードオフ                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  次元数      精度        ストレージ     検索速度          │
│  3072       最高        ×3            遅い               │
│  1536       高い        ×1.5          普通               │
│  1024       良好        ×1            速い  ← 推奨       │
│  512        中程度      ×0.5          最速               │
│  256        低下        ×0.25         最速               │
│                                                          │
│  推奨: 1024次元がコスパ最適                               │
│  理由: 精度低下が1-2%に対し、ストレージ・速度が大幅改善   │
│                                                          │
│  Matryoshka Embedding:                                   │
│  text-embedding-3 は任意の次元数に切り詰め可能            │
│  dimensions パラメータで指定するだけ                      │
└──────────────────────────────────────────────────────────┘
```

### 5.1 バッチ処理と並列化

```python
import asyncio
from openai import AsyncOpenAI

async def batch_embed(texts: list[str], batch_size: int = 100) -> list:
    """大量テキストの効率的な埋め込み"""
    client = AsyncOpenAI()
    all_embeddings = []

    # バッチに分割
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

    # 並列実行 (レート制限に注意)
    semaphore = asyncio.Semaphore(5)  # 同時5リクエストまで

    async def embed_batch(batch):
        async with semaphore:
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
            )
            return [d.embedding for d in response.data]

    results = await asyncio.gather(*[embed_batch(b) for b in batches])

    for batch_result in results:
        all_embeddings.extend(batch_result)

    return all_embeddings
```

---

## 6. アンチパターン

### アンチパターン 1: Embedding モデルの混在

```python
# NG: インデックス時と検索時で異なるモデルを使用
index_embeddings = openai_embed(documents)    # text-embedding-3-large
query_embedding = cohere_embed(query)          # embed-v3
# → ベクトル空間が異なるため、類似度計算が無意味

# OK: 同一モデルで統一
index_embeddings = openai_embed(documents, model="text-embedding-3-small")
query_embedding = openai_embed(query, model="text-embedding-3-small")
```

### アンチパターン 2: 巨大テキストの直接埋め込み

```python
# NG: 10万文字のドキュメントをそのまま埋め込み
embedding = embed(huge_document)  # 情報が圧縮されすぎて精度低下

# OK: 適切にチャンク分割してから埋め込み
chunks = split_text(huge_document, chunk_size=512)
chunk_embeddings = [embed(chunk) for chunk in chunks]
# → 各チャンクの意味が保持される
```

---

## 7. FAQ

### Q1: Embedding の次元数はどう選ぶべき?

RAG や検索用途では 1024 次元が精度とコストのバランスが良い。
大規模データ (数百万件以上) では 256-512 次元に削減してストレージ・速度を優先。
精度が最重要なら 1536-3072 次元を使い、ANN インデックス (HNSW等) で速度を補う。

### Q2: Embedding モデルのファインチューニングは効果的か?

ドメイン固有の用語や概念が多い場合 (医療、法律、特定業界)、ファインチューニングで 5-15% の精度向上が期待できる。
Sentence Transformers の `SentenceTransformerTrainer` で対比学習が可能。
ただし、汎用用途では最新のプリトレインモデルの方が良い場合も多い。

### Q3: 日本語 Embedding で特に注意する点は?

トークナイザの日本語対応品質が性能に直結する。
BGE-M3、Cohere embed-v3、multilingual-e5 は日本語で高性能。
日本語特化モデル (intfloat/multilingual-e5-large 等) は JSTS/JSICK ベンチマークで評価する。
「クエリ」と「ドキュメント」で異なるプレフィックスを付けるモデル (e5系) では、この使い分けが精度に大きく影響する。

---

## まとめ

| 項目 | 推奨 |
|------|------|
| API 推奨モデル | text-embedding-3-small (コスパ) / Voyage-3 (精度) |
| OSS 推奨モデル | BGE-M3 (多言語) / nomic-embed-text (軽量) |
| 日本語推奨 | BGE-M3 / Cohere embed-v3 |
| 推奨次元数 | 1024 (バランス型) |
| 距離関数 | コサイン類似度 (正規化済みなら内積と等価) |
| バッチサイズ | 100-500 テキスト/リクエスト |
| 主要用途 | 検索、分類、クラスタリング、異常検知、RAG |

---

## 次に読むべきガイド

- [01-rag.md](./01-rag.md) — Embedding を活用した RAG パイプライン
- [04-multimodal.md](./04-multimodal.md) — マルチモーダル Embedding
- [../03-infrastructure/01-vector-databases.md](../03-infrastructure/01-vector-databases.md) — ベクトル DB の選定と運用

---

## 参考文献

1. Muennighoff et al., "MTEB: Massive Text Embedding Benchmark," EACL 2023
2. OpenAI, "Embeddings Guide," https://platform.openai.com/docs/guides/embeddings
3. Xiao et al., "C-Pack: Packaged Resources To Advance General Chinese Embedding," arXiv:2309.07597, 2023
4. Sentence Transformers, "Documentation," https://www.sbert.net/

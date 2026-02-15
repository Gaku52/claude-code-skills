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

### 1.2 距離関数の使い分け

```
┌──────────────────────────────────────────────────────────┐
│           距離関数の選択ガイド                              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  コサイン類似度 (Cosine Similarity)                       │
│  ├── 範囲: -1 ～ 1                                       │
│  ├── ベクトルの方向のみを比較 (大きさは無視)              │
│  ├── テキスト Embedding で最も一般的                       │
│  └── 推奨: ほとんどの検索・類似度タスク                   │
│                                                          │
│  ユークリッド距離 (L2 Distance)                          │
│  ├── 範囲: 0 ～ ∞                                        │
│  ├── ベクトルの大きさも考慮                               │
│  ├── クラスタリングで使用                                 │
│  └── 推奨: 正規化されていないベクトル                     │
│                                                          │
│  内積 (Dot Product / Inner Product)                      │
│  ├── 範囲: -∞ ～ ∞                                       │
│  ├── 正規化済みベクトルではコサイン類似度と等価            │
│  ├── 計算が最も高速                                       │
│  └── 推奨: 正規化済みベクトル (多くの API がデフォルト)    │
│                                                          │
│  マンハッタン距離 (L1 Distance)                          │
│  ├── 各次元の絶対差の合計                                 │
│  ├── 外れ値に対してユークリッドより頑健                   │
│  └── 推奨: スパースなベクトル                             │
└──────────────────────────────────────────────────────────┘
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

### 2.4 Google Vertex AI Embedding

```python
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel

# Google の多言語 Embedding モデル
model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")

embeddings = model.get_embeddings(
    texts=["機械学習の基礎", "ディープラーニング入門"],
    auto_truncate=True,
)

for emb in embeddings:
    print(f"次元数: {len(emb.values)}")  # 768
    print(f"統計: {emb.statistics}")
```

### 2.5 Embedding モデルのローカル実行

```python
# ONNX ランタイムで高速推論
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import numpy as np

# ONNX 最適化済みモデルをロード
model = ORTModelForFeatureExtraction.from_pretrained(
    "BAAI/bge-m3",
    export=True,  # 初回は PyTorch → ONNX 変換
)
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

# 推論
inputs = tokenizer(
    ["機械学習の基礎を学ぶ"],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="np",
)

outputs = model(**inputs)
# [CLS] トークンの出力を Embedding として使用
embedding = outputs.last_hidden_state[:, 0, :]
embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
print(f"形状: {embedding.shape}")  # (1, 1024)
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

### 3.3 Embedding モデル選定フローチャート

```
┌──────────────────────────────────────────────────────────┐
│          Embedding モデル選定フロー                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  START: 要件確認                                         │
│    │                                                     │
│    ├── データをクラウドに送れない?                         │
│    │   YES → OSS モデル                                  │
│    │         ├── 多言語 → BGE-M3                         │
│    │         ├── 軽量   → nomic-embed-text               │
│    │         └── 日本語特化 → multilingual-e5-large       │
│    │                                                     │
│    NO ↓                                                  │
│    ├── 長文書 (8K+ トークン) を扱う?                      │
│    │   YES → Voyage-3 (32K 対応)                         │
│    │                                                     │
│    NO ↓                                                  │
│    ├── コスト重視?                                       │
│    │   YES → text-embedding-3-small ($0.02/1M)           │
│    │                                                     │
│    NO ↓                                                  │
│    ├── 日本語重視?                                       │
│    │   YES → Cohere embed-v3 / BGE-M3                    │
│    │                                                     │
│    NO ↓                                                  │
│    └── 最高精度                                          │
│         → text-embedding-3-large (dimensions=1024)       │
└──────────────────────────────────────────────────────────┘
```

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

### 4.2 ハイブリッド検索 (ベクトル + キーワード)

```python
import numpy as np
from rank_bm25 import BM25Okapi
import MeCab

def hybrid_search(
    query: str,
    documents: list[str],
    alpha: float = 0.5,  # 0=キーワードのみ, 1=ベクトルのみ
    top_k: int = 5,
) -> list[dict]:
    """ハイブリッド検索: BM25 + Embedding"""

    # 1. BM25 (キーワード検索)
    mecab = MeCab.Tagger("-Owakati")
    tokenized_docs = [mecab.parse(doc).strip().split() for doc in documents]
    tokenized_query = mecab.parse(query).strip().split()

    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(tokenized_query)
    # 正規化 (0-1)
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()

    # 2. Embedding (セマンティック検索)
    all_texts = [query] + documents
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=all_texts,
    )
    query_vec = np.array(response.data[0].embedding)
    doc_vecs = np.array([d.embedding for d in response.data[1:]])

    cosine_scores = np.dot(doc_vecs, query_vec) / (
        np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec)
    )
    # 正規化 (0-1)
    cosine_scores = (cosine_scores + 1) / 2

    # 3. スコア統合
    hybrid_scores = alpha * cosine_scores + (1 - alpha) * bm25_scores

    # 上位 k 件を返す
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    return [
        {
            "text": documents[i],
            "hybrid_score": float(hybrid_scores[i]),
            "vector_score": float(cosine_scores[i]),
            "bm25_score": float(bm25_scores[i]),
        }
        for i in top_indices
    ]
```

### 4.3 クラスタリング

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

### 4.4 異常検知

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

### 4.5 テキスト分類 (ゼロショット)

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

### 4.6 重複検出 (Deduplication)

```python
import numpy as np
from itertools import combinations

def find_near_duplicates(
    texts: list[str],
    threshold: float = 0.95,
) -> list[tuple[int, int, float]]:
    """Embedding ベースの重複検出"""

    embeddings = np.array(get_embeddings(texts))
    # 正規化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms

    # 類似度行列を計算
    similarity_matrix = np.dot(normalized, normalized.T)

    # 閾値以上のペアを抽出
    duplicates = []
    for i, j in combinations(range(len(texts)), 2):
        sim = similarity_matrix[i, j]
        if sim >= threshold:
            duplicates.append((i, j, float(sim)))

    return sorted(duplicates, key=lambda x: -x[2])

# 使用例
texts = [
    "Pythonは人気のプログラミング言語です",
    "Pythonは広く使われるプログラミング言語です",  # ほぼ重複
    "JavaScriptはWebで使われる言語です",
    "今日は天気が良いです",
]

duplicates = find_near_duplicates(texts, threshold=0.9)
for i, j, sim in duplicates:
    print(f"[{sim:.4f}] '{texts[i]}' ≈ '{texts[j]}'")
```

### 4.7 レコメンデーション

```python
import numpy as np

class EmbeddingRecommender:
    """Embedding ベースのレコメンデーションエンジン"""

    def __init__(self):
        self.items: list[dict] = []
        self.embeddings: np.ndarray | None = None

    def add_items(self, items: list[dict]):
        """アイテムを追加 (title, description, metadata)"""
        self.items = items
        texts = [f"{item['title']}: {item['description']}" for item in items]
        self.embeddings = np.array(get_embeddings(texts))
        # 正規化
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / norms

    def recommend_by_text(self, query: str, top_k: int = 5) -> list[dict]:
        """テキストクエリに基づくレコメンド"""
        query_emb = np.array(get_embeddings([query])[0])
        query_emb = query_emb / np.linalg.norm(query_emb)

        scores = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {**self.items[i], "score": float(scores[i])}
            for i in top_indices
        ]

    def recommend_similar(self, item_index: int, top_k: int = 5) -> list[dict]:
        """類似アイテムのレコメンド"""
        scores = np.dot(self.embeddings, self.embeddings[item_index])
        top_indices = np.argsort(scores)[::-1][1:top_k+1]  # 自分自身を除外

        return [
            {**self.items[i], "score": float(scores[i])}
            for i in top_indices
        ]

    def recommend_by_history(
        self, viewed_indices: list[int], top_k: int = 5
    ) -> list[dict]:
        """閲覧履歴に基づくレコメンド"""
        # 閲覧アイテムの平均ベクトル
        viewed_embs = self.embeddings[viewed_indices]
        profile = viewed_embs.mean(axis=0)
        profile = profile / np.linalg.norm(profile)

        scores = np.dot(self.embeddings, profile)

        # 既に閲覧したアイテムを除外
        for idx in viewed_indices:
            scores[idx] = -1

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {**self.items[i], "score": float(scores[i])}
            for i in top_indices
        ]
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

### 5.1 Matryoshka Representation Learning (MRL)

```python
from openai import OpenAI

client = OpenAI()

# 同じテキストを異なる次元数で埋め込み
text = "機械学習の基礎を学ぶ"

for dim in [256, 512, 1024, 3072]:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        dimensions=dim,
    )
    emb = response.data[0].embedding
    print(f"次元数: {dim:4d}, メモリ: {dim * 4:>5d} bytes/vector")

# 出力:
# 次元数:  256, メモリ:  1024 bytes/vector
# 次元数:  512, メモリ:  2048 bytes/vector
# 次元数: 1024, メモリ:  4096 bytes/vector
# 次元数: 3072, メモリ: 12288 bytes/vector

# MRL の利点:
# - 同一モデルで精度 vs コストを柔軟に調整
# - 先頭の次元ほど重要な情報を保持
# - 段階的な検索 (粗い→細かい) が可能
```

### 5.2 バッチ処理と並列化

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

### 5.3 Embedding のキャッシュ戦略

```python
import hashlib
import json
import sqlite3
import numpy as np
from typing import Optional

class EmbeddingCache:
    """SQLite ベースの Embedding キャッシュ"""

    def __init__(self, db_path: str = "embedding_cache.db", model: str = "text-embedding-3-small"):
        self.model = model
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT PRIMARY KEY,
                model TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def _hash(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()

    def get(self, text: str) -> Optional[list[float]]:
        """キャッシュから Embedding を取得"""
        row = self.conn.execute(
            "SELECT embedding FROM embeddings WHERE text_hash = ?",
            (self._hash(text),),
        ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def put(self, text: str, embedding: list[float]):
        """Embedding をキャッシュに保存"""
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings (text_hash, model, embedding) VALUES (?, ?, ?)",
            (self._hash(text), self.model, json.dumps(embedding)),
        )
        self.conn.commit()

    def get_or_create(self, texts: list[str]) -> list[list[float]]:
        """キャッシュミスのテキストのみ API を呼び出し"""
        results = [None] * len(texts)
        uncached_indices = []

        for i, text in enumerate(texts):
            cached = self.get(text)
            if cached:
                results[i] = cached
            else:
                uncached_indices.append(i)

        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            response = client.embeddings.create(
                model=self.model, input=uncached_texts
            )
            for idx, emb_data in zip(uncached_indices, response.data):
                results[idx] = emb_data.embedding
                self.put(texts[idx], emb_data.embedding)

        return results

# 使用例
cache = EmbeddingCache()
embeddings = cache.get_or_create(["Hello", "World", "Hello"])  # "Hello" は1回だけAPI呼び出し
```

---

## 6. Embedding のファインチューニング

### 6.1 Sentence Transformers でのファインチューニング

```python
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
)
from torch.utils.data import DataLoader

# ベースモデルをロード
model = SentenceTransformer("BAAI/bge-m3")

# 訓練データの準備 (正例ペア)
train_examples = [
    InputExample(texts=["Pythonの使い方", "Pythonプログラミング入門"], label=0.9),
    InputExample(texts=["Pythonの使い方", "Java言語の基礎"], label=0.3),
    InputExample(texts=["機械学習とは", "ディープラーニングの基礎"], label=0.8),
    InputExample(texts=["機械学習とは", "今日の天気"], label=0.05),
    # 1000+ ペアを推奨
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# CosineSimilarity Loss (回帰型)
train_loss = losses.CosineSimilarityLoss(model)

# 訓練
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="./finetuned-embedding",
)

# ファインチューニング済みモデルの利用
finetuned = SentenceTransformer("./finetuned-embedding")
embeddings = finetuned.encode(["テスト文書"])
```

### 6.2 対比学習 (Contrastive Learning) によるファインチューニング

```python
from sentence_transformers import InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# Triplet Loss (アンカー, 正例, 負例)
triplet_examples = [
    InputExample(texts=[
        "Pythonでのデータ分析",     # アンカー
        "pandas を使ったデータ処理",  # 正例 (類似)
        "JavaScriptフレームワーク",   # 負例 (非類似)
    ]),
    # ...
]

triplet_loader = DataLoader(triplet_examples, shuffle=True, batch_size=16)
triplet_loss = losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE)

# Multiple Negatives Ranking Loss (大規模データに最適)
mnrl_examples = [
    InputExample(texts=["クエリ1", "関連文書1"]),
    InputExample(texts=["クエリ2", "関連文書2"]),
    # バッチ内の他ペアが自動的に負例になる
]

mnrl_loader = DataLoader(mnrl_examples, shuffle=True, batch_size=32)
mnrl_loss = losses.MultipleNegativesRankingLoss(model)

# 評価器の設定
evaluator = InformationRetrievalEvaluator(
    queries={"q1": "Python データ分析", "q2": "機械学習 入門"},
    corpus={"d1": "pandasの使い方...", "d2": "scikit-learnチュートリアル..."},
    relevant_docs={"q1": {"d1"}, "q2": {"d2"}},
)

model.fit(
    train_objectives=[(mnrl_loader, mnrl_loss)],
    evaluator=evaluator,
    epochs=5,
    evaluation_steps=500,
    output_path="./finetuned-retrieval",
)
```

---

## 7. チャンク分割戦略

```
┌──────────────────────────────────────────────────────────┐
│          テキストチャンク分割の戦略                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. 固定長分割 (Fixed Size)                              │
│     ├── 実装が最も簡単                                    │
│     ├── 文の途中で切れるリスク                            │
│     └── 推奨チャンクサイズ: 256-512 トークン              │
│                                                          │
│  2. セマンティック分割 (Semantic Chunking)                │
│     ├── 文/段落境界で分割                                 │
│     ├── Embedding の類似度が急変する点で分割              │
│     └── 意味の一貫性が高い                               │
│                                                          │
│  3. オーバーラップ分割 (Sliding Window)                   │
│     ├── チャンク間に重複区間を設定                        │
│     ├── 境界付近の情報損失を軽減                          │
│     └── 推奨オーバーラップ: 50-100 トークン               │
│                                                          │
│  4. 再帰的分割 (Recursive)                               │
│     ├── LangChain の RecursiveCharacterTextSplitter       │
│     ├── 段落 → 文 → 単語の順で階層的に分割               │
│     └── 最も汎用的で品質が良い                            │
│                                                          │
│  5. ドキュメント構造ベース                                │
│     ├── Markdown: ヘッダーで分割                          │
│     ├── HTML: タグ構造で分割                              │
│     └── コード: 関数/クラス単位で分割                     │
└──────────────────────────────────────────────────────────┘
```

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 再帰的分割 (最も一般的)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # トークン数ではなく文字数
    chunk_overlap=50,      # オーバーラップ文字数
    separators=["\n\n", "\n", "。", "、", " ", ""],  # 分割優先順
    length_function=len,
)

chunks = splitter.split_text(long_document)

# セマンティック分割 (Embedding 類似度ベース)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

semantic_splitter = SemanticChunker(
    OpenAIEmbeddings(model="text-embedding-3-small"),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90,  # 類似度の低い箇所で分割
)

semantic_chunks = semantic_splitter.split_text(long_document)
```

---

## 8. トラブルシューティング

### 8.1 よくある問題と解決策

```
┌──────────────────────────────────────────────────────────┐
│          Embedding のトラブルシューティング                 │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  問題 1: 検索精度が低い                                   │
│  ├── 原因 1: チャンクサイズが不適切                       │
│  │   └── 解決: 256-512 トークンに調整                    │
│  ├── 原因 2: クエリとドキュメントの形式が異なる           │
│  │   └── 解決: e5系は "query:" "passage:" プレフィックス  │
│  ├── 原因 3: モデルの言語対応が弱い                      │
│  │   └── 解決: 多言語モデル (BGE-M3) に変更              │
│  └── 原因 4: ドメイン特化用語が多い                      │
│      └── 解決: ファインチューニングを検討                 │
│                                                          │
│  問題 2: 速度が遅い                                       │
│  ├── 原因 1: 毎回 API を呼んでいる                       │
│  │   └── 解決: キャッシュ層を追加                         │
│  ├── 原因 2: バッチ処理していない                         │
│  │   └── 解決: 100件/バッチで一括処理                    │
│  └── 原因 3: 次元数が大きすぎる                           │
│      └── 解決: 1024次元に削減                             │
│                                                          │
│  問題 3: コストが高い                                     │
│  ├── 原因 1: 大きいモデルを使っている                    │
│  │   └── 解決: text-embedding-3-small に変更              │
│  ├── 原因 2: 重複テキストを再計算している                 │
│  │   └── 解決: ハッシュベースのキャッシュ                 │
│  └── 原因 3: 不必要に長いテキストを埋め込んでいる         │
│      └── 解決: チャンク分割で最適長に                     │
│                                                          │
│  問題 4: 類似度スコアが直感と合わない                     │
│  ├── 原因 1: 距離関数の選択ミス                          │
│  │   └── 解決: コサイン類似度を使用                       │
│  └── 原因 2: 正規化されていない                          │
│      └── 解決: normalize_embeddings=True                 │
└──────────────────────────────────────────────────────────┘
```

### 8.2 Embedding の品質検証

```python
import numpy as np
from itertools import combinations

def validate_embeddings(
    test_pairs: list[tuple[str, str, float]],
    model_name: str = "text-embedding-3-small",
) -> dict:
    """Embedding モデルの品質検証

    test_pairs: [(text_a, text_b, expected_similarity), ...]
    expected_similarity: 0.0 (無関係) ～ 1.0 (同義)
    """
    texts_a = [p[0] for p in test_pairs]
    texts_b = [p[1] for p in test_pairs]
    expected = [p[2] for p in test_pairs]

    embs_a = np.array(get_embeddings(texts_a, model=model_name))
    embs_b = np.array(get_embeddings(texts_b, model=model_name))

    # コサイン類似度を計算
    actual = []
    for a, b in zip(embs_a, embs_b):
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        actual.append(float(sim))

    # 相関係数 (期待値との相関)
    from scipy.stats import spearmanr
    correlation, p_value = spearmanr(expected, actual)

    # 分類精度 (閾値0.5で正/負を判別)
    correct = sum(
        1 for e, a in zip(expected, actual)
        if (e >= 0.5 and a >= 0.5) or (e < 0.5 and a < 0.5)
    )
    accuracy = correct / len(test_pairs)

    return {
        "model": model_name,
        "spearman_correlation": correlation,
        "p_value": p_value,
        "classification_accuracy": accuracy,
        "mean_actual_similarity": np.mean(actual),
    }
```

---

## 9. アンチパターン

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

### アンチパターン 3: キャッシュなしの大量処理

```python
# NG: 同じテキストを何度も API に送信
for query in user_queries:  # 多くは過去のクエリと同一
    embedding = embed(query)  # 毎回 API 呼び出し → コスト膨大

# OK: キャッシュを使用
cache = EmbeddingCache()
for query in user_queries:
    embedding = cache.get_or_create([query])[0]
```

### アンチパターン 4: Embedding の可視化なしでの運用

```python
# NG: Embedding の分布を確認せずにデプロイ
# → 想定外のクラスタリング結果やバイアスに気づかない

# OK: 定期的に可視化して品質を確認
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings, labels, title="Embedding Space"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(np.array(embeddings))

    plt.figure(figsize=(10, 8))
    for label in set(labels):
        mask = [l == label for l in labels]
        points = reduced[mask]
        plt.scatter(points[:, 0], points[:, 1], label=label, alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.savefig("embedding_visualization.png")
```

---

## 10. FAQ

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

### Q4: Embedding モデルを変更する際の注意点は?

モデル変更時は全ベクトルの再計算が必要。異なるモデルのベクトル空間は互換性がない。
段階的な移行: 新旧モデルを並行稼働し、品質比較した上で切り替える。
バージョン管理: モデル名とバージョンをメタデータに記録しておく。

### Q5: Sparse Embedding と Dense Embedding の違いは?

Dense Embedding (本章): 全次元に値が入る (1024次元に1024個の値)。意味的類似度に強い。
Sparse Embedding (BM25, SPLADE等): ほとんどの次元が0。キーワード一致に強い。
ハイブリッド検索: 両方を組み合わせると最高の検索精度が得られる。
BGE-M3 は Dense + Sparse の両方を生成できる唯一のモデルの一つ。

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
| チャンクサイズ | 256-512 トークン (オーバーラップ 50-100) |
| キャッシュ | 必須 (SQLite / Redis) |
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
5. Kusupati et al., "Matryoshka Representation Learning," NeurIPS 2022
6. Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering," EMNLP 2020

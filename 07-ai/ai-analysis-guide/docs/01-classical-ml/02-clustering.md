# クラスタリング — K-means、DBSCAN、階層的

> ラベルなしデータからグループ構造を自動発見するクラスタリング手法を比較・実装する

## この章で学ぶこと

1. **K-means** — セントロイドベースのクラスタリングとクラスタ数の決定法
2. **DBSCAN** — 密度ベースのクラスタリングで任意形状のクラスタを検出
3. **階層的クラスタリング** — デンドログラムによるクラスタ構造の可視化

---

## 1. K-means クラスタリング

### K-meansのアルゴリズム

```
K-means のイテレーション:

Step 1: 初期化           Step 2: 割り当て         Step 3: 更新
 (ランダムにK個の          (最近接セントロイドに     (各クラスタの
  セントロイド配置)         各点を割り当て)          重心を再計算)

  ○  ○    ○              ●  ●    ○              ●  ●    ○
    ★       ○               ★       ○               ★     ○
  ○    ○                  ●    ●                  ●    ●
         ★    ○                  ★    ○                ★    ○
    ○  ○     ○               ○  ○     ○             ○  ○     ○
  ○       ○    ○           ○       ○    ○         ○       ○    ○

  ★ = セントロイド          ● = クラスタ1           ★ = 移動後の
                           ○ = クラスタ2             セントロイド

  → Step 2, 3 を収束するまで繰り返す
```

### コード例1: K-meansとクラスタ数決定

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.datasets import make_blobs

# データ生成
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=1.0,
                        random_state=42)

# エルボー法 + シルエットスコアで最適K探索
K_range = range(2, 11)
inertias = []
silhouettes = []

for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, km.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# エルボー法
ax1.plot(K_range, inertias, "bo-")
ax1.set_xlabel("クラスタ数 K")
ax1.set_ylabel("慣性 (Inertia)")
ax1.set_title("エルボー法")
ax1.grid(True, alpha=0.3)

# シルエットスコア
ax2.plot(K_range, silhouettes, "ro-")
ax2.set_xlabel("クラスタ数 K")
ax2.set_ylabel("シルエットスコア")
ax2.set_title("シルエット分析")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/kmeans_selection.png", dpi=150)
plt.close()

best_k = K_range[np.argmax(silhouettes)]
print(f"最適クラスタ数: {best_k}")
```

---

## 2. DBSCAN — 密度ベースクラスタリング

### DBSCANの動作原理

```
DBSCAN のパラメータ:
  eps: 近傍の半径
  min_samples: コア点となるための最小近傍数

点の分類:
  ┌──────────────────────────────────────┐
  │                                      │
  │  コア点 (Core)                       │
  │  ├── eps内にmin_samples個以上の点    │
  │  └── クラスタの中心を形成           │
  │                                      │
  │  ボーダー点 (Border)                 │
  │  ├── コア点のeps内にあるが           │
  │  └── 自身はコア点の条件を満たさない │
  │                                      │
  │  ノイズ点 (Noise)                    │
  │  ├── どのコア点のeps内にもない       │
  │  └── ラベル = -1                    │
  │                                      │
  └──────────────────────────────────────┘

  図示 (eps=1, min_samples=3):

       ●───●      ← コア点同士が接続 → 同一クラスタ
      / \   \
     ●   ●   ◐   ← ◐ = ボーダー点

         ✕        ← ✕ = ノイズ点（孤立）
```

### コード例2: DBSCAN実装とパラメータ探索

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_moons

# 非凸形状のデータ
X, y_true = make_moons(n_samples=500, noise=0.1, random_state=42)
X = StandardScaler().fit_transform(X)

# eps の自動決定（k-距離グラフ）
k = 5
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X)
distances, _ = nn.kneighbors(X)
k_distances = np.sort(distances[:, -1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(k_distances)
ax1.set_xlabel("点のインデックス（ソート済み）")
ax1.set_ylabel(f"{k}-距離")
ax1.set_title(f"k-距離グラフ（k={k}）→ 肘の位置がeps候補")
ax1.grid(True, alpha=0.3)

# 最適eps付近で実行
eps_optimal = 0.3
db = DBSCAN(eps=eps_optimal, min_samples=k)
labels = db.fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=20, alpha=0.7)
ax2.scatter(X[labels == -1, 0], X[labels == -1, 1],
            c="red", marker="x", s=50, label=f"ノイズ ({n_noise})")
ax2.set_title(f"DBSCAN: {n_clusters}クラスタ, ノイズ{n_noise}点")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/dbscan_result.png", dpi=150)
plt.close()
```

---

## 3. 階層的クラスタリング

### コード例3: デンドログラムと階層的クラスタリング

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=150, centers=4, cluster_std=0.8,
                        random_state=42)

# SciPyでデンドログラム生成
Z = linkage(X, method="ward")  # Ward法（分散最小化）

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# デンドログラム
dendrogram(Z, truncate_mode="lastp", p=30, ax=ax1,
           leaf_rotation=90, leaf_font_size=8)
ax1.set_title("デンドログラム（Ward法）")
ax1.set_xlabel("サンプル")
ax1.set_ylabel("距離")
ax1.axhline(y=15, color="r", linestyle="--", label="カット位置")
ax1.legend()

# クラスタリング結果
agg = AgglomerativeClustering(n_clusters=4, linkage="ward")
labels = agg.fit_predict(X)

ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=30, alpha=0.7)
ax2.set_title("階層的クラスタリング結果（K=4）")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/hierarchical_clustering.png", dpi=150)
plt.close()
```

### コード例4: 連結法の比較

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np

linkages = ["ward", "complete", "average", "single"]
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

print(f"{'連結法':12s} {'ARI':>8s} {'NMI':>8s}")
print("-" * 32)
for link in linkages:
    if link == "ward":
        agg = AgglomerativeClustering(n_clusters=4, linkage=link)
    else:
        agg = AgglomerativeClustering(n_clusters=4, linkage=link)
    labels = agg.fit_predict(X)
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    print(f"{link:12s} {ari:8.4f} {nmi:8.4f}")
```

### コード例5: クラスタリング結果の実践的活用

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 顧客セグメンテーションの例
np.random.seed(42)
n = 1000
customers = pd.DataFrame({
    "customer_id": range(n),
    "purchase_amount": np.random.exponential(5000, n),
    "purchase_frequency": np.random.poisson(5, n),
    "recency_days": np.random.exponential(30, n),
    "avg_session_minutes": np.random.exponential(10, n),
})

features = ["purchase_amount", "purchase_frequency",
            "recency_days", "avg_session_minutes"]

# スケーリング → K-means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customers[features])

km = KMeans(n_clusters=4, n_init=10, random_state=42)
customers["segment"] = km.fit_predict(X_scaled)

# セグメントごとの特性を分析
segment_profile = (
    customers
    .groupby("segment")[features]
    .agg(["mean", "median", "std"])
    .round(1)
)

# セグメントにラベル付け
segment_names = {
    0: "休眠顧客", 1: "優良顧客",
    2: "新規顧客", 3: "一般顧客"
}

customers["segment_name"] = customers["segment"].map(segment_names)

print("=== セグメント別プロファイル ===")
summary = customers.groupby("segment_name")[features].mean().round(1)
summary["顧客数"] = customers.groupby("segment_name").size()
print(summary)
```

---

## 比較表

### クラスタリング手法の比較

| 手法 | クラスタ形状 | K指定 | 外れ値処理 | 計算量 | スケーラビリティ | 適用場面 |
|---|---|---|---|---|---|---|
| K-means | 球状 | 必要 | 弱い | O(nKt) | 高い | 均一サイズの球状クラスタ |
| Mini-Batch K-means | 球状 | 必要 | 弱い | O(nK) | 非常に高い | 大規模データ |
| DBSCAN | 任意形状 | 不要 | 強い (ノイズ検出) | O(n log n) | 中程度 | 密度差のあるデータ |
| HDBSCAN | 任意形状 | 不要 | 強い | O(n log n) | 中程度 | 密度が不均一 |
| 階層的 (Ward) | 球状 | 後決め可 | 弱い | O(n²) | 低い | 小〜中規模、構造探索 |
| GMM | 楕円形 | 必要 | 中程度 | O(nK²d) | 中程度 | 確率的クラスタ割り当て |

### クラスタ評価指標の使い分け

| 指標 | 正解ラベル | 範囲 | 解釈 | 用途 |
|---|---|---|---|---|
| シルエットスコア | 不要 | [-1, 1] | 高いほどクラスタが明確 | K選択、品質評価 |
| Calinski-Harabasz | 不要 | [0, ∞) | 高いほど良い | K選択 |
| Davies-Bouldin | 不要 | [0, ∞) | 低いほど良い | K選択 |
| ARI (調整ランド指数) | 必要 | [-1, 1] | 1=完全一致、0=ランダム | 正解比較 |
| NMI (正規化相互情報量) | 必要 | [0, 1] | 1=完全一致 | 正解比較 |

---

## アンチパターン

### アンチパターン1: スケーリングなしのK-means

```python
# BAD: 単位の異なる特徴量をそのままクラスタリング
# 年収（万円: 300〜2000）と年齢（歳: 20〜70）→ 年収が支配的になる
km = KMeans(n_clusters=3)
km.fit(df[["income", "age"]])  # 距離が年収に引きずられる

# GOOD: StandardScalerで正規化してからクラスタリング
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[["income", "age"]])
km = KMeans(n_clusters=3, n_init=10, random_state=42)
km.fit(X_scaled)
```

### アンチパターン2: 「とりあえずK=3」

```python
# BAD: 根拠なくK=3を選択
km = KMeans(n_clusters=3).fit(X)

# GOOD: 複数の指標でKを選定
from sklearn.metrics import silhouette_score, calinski_harabasz_score

for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sil = silhouette_score(X, km.labels_)
    ch = calinski_harabasz_score(X, km.labels_)
    print(f"K={k:2d}  シルエット={sil:.4f}  CH={ch:.0f}  慣性={km.inertia_:.0f}")
```

---

## FAQ

### Q1: K-meansとGMMの使い分けは？

**A:** K-meansは各点を1つのクラスタに「ハード」に割り当てるが、GMMは確率的な「ソフト」割り当てが可能。クラスタが重なり合う場合やクラスタの形が楕円形の場合はGMMが適している。K-meansは高速で大規模データに向く。

### Q2: DBSCANのepsとmin_samplesの決め方は？

**A:** min_samplesは次元数×2が目安。epsはk-距離グラフ（k=min_samples）の「肘」の位置から読み取る。データのドメイン知識がある場合は、「同一クラスタとみなせる最大距離」から設定する。

### Q3: クラスタリング結果の「正しさ」はどう判断する？

**A:** 正解ラベルがない場合、単一の正解はない。(1) シルエットスコアなどの内部指標、(2) ドメイン専門家によるクラスタの解釈可能性、(3) 下流タスク（マーケティング施策等）での有効性、の3軸で総合的に判断する。

---

## まとめ

| 項目 | 要点 |
|---|---|
| K-means | 高速・シンプル。球状クラスタ向け。Kはエルボー法+シルエットで選定 |
| DBSCAN | 任意形状対応。ノイズ検出可能。パラメータはk-距離グラフで決定 |
| 階層的 | デンドログラムで構造可視化。小〜中規模データ向け |
| 評価 | 内部指標（シルエット等）+ ドメイン知識で総合判断 |
| 前処理 | スケーリングは必須（距離ベースの手法全般） |

---

## 次に読むべきガイド

- [03-dimensionality-reduction.md](./03-dimensionality-reduction.md) — 次元削減でクラスタ構造を可視化
- [../03-applied/00-nlp.md](../03-applied/00-nlp.md) — テキストクラスタリングの応用

---

## 参考文献

1. **Martin Ester et al.** "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise" KDD 1996
2. **scikit-learn** "Clustering" — https://scikit-learn.org/stable/modules/clustering.html
3. **Lior Rokach, Oded Maimon** "Clustering Methods" in Data Mining and Knowledge Discovery Handbook, Springer, 2005

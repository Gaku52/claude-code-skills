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

### K-meansの数学的基礎

K-meansは以下の目的関数（慣性、Within-Cluster Sum of Squares: WCSS）を最小化する。

```
目的関数:
  J = Σ_{k=1}^{K} Σ_{x∈C_k} ||x - μ_k||²

  C_k: クラスタkに属するデータ点の集合
  μ_k: クラスタkのセントロイド（重心）
  K: クラスタ数

収束性:
  ・各イテレーションでJは単調減少する（証明可能）
  ・有限ステップで局所最適解に収束する
  ・ただし大域最適解は保証されない → 複数回の初期化が重要
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

### コード例1b: K-meansのフルスクラッチ実装

```python
import numpy as np

class KMeansFromScratch:
    """K-meansアルゴリズムのフル実装"""

    def __init__(self, n_clusters: int = 3, max_iter: int = 300,
                 tol: float = 1e-4, n_init: int = 10,
                 random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.rng = np.random.RandomState(random_state)

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """K-means++による初期化"""
        n_samples = X.shape[0]
        centroids = [X[self.rng.randint(n_samples)]]

        for _ in range(1, self.n_clusters):
            # 各点から最も近いセントロイドまでの距離の二乗
            distances = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in centroids],
                axis=0
            )
            # 距離に比例する確率で次のセントロイドを選択
            probs = distances / distances.sum()
            idx = self.rng.choice(n_samples, p=probs)
            centroids.append(X[idx])

        return np.array(centroids)

    def _assign_clusters(self, X: np.ndarray,
                          centroids: np.ndarray) -> np.ndarray:
        """各点を最近接セントロイドに割り当て"""
        distances = np.array([
            np.sum((X - c) ** 2, axis=1) for c in centroids
        ])
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X: np.ndarray,
                           labels: np.ndarray) -> np.ndarray:
        """各クラスタの重心を再計算"""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centroids[k] = X[mask].mean(axis=0)
        return centroids

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray,
                          centroids: np.ndarray) -> float:
        """WCSS（慣性）を計算"""
        inertia = 0.0
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                inertia += np.sum((X[mask] - centroids[k]) ** 2)
        return inertia

    def fit(self, X: np.ndarray) -> "KMeansFromScratch":
        """複数回の初期化で最良の結果を選択"""
        best_inertia = float("inf")
        best_centroids = None
        best_labels = None

        for _ in range(self.n_init):
            centroids = self._init_centroids(X)

            for iteration in range(self.max_iter):
                labels = self._assign_clusters(X, centroids)
                new_centroids = self._update_centroids(X, labels)

                # 収束判定
                shift = np.sum((new_centroids - centroids) ** 2)
                centroids = new_centroids
                if shift < self.tol:
                    break

            inertia = self._compute_inertia(X, labels, centroids)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels

        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """新しいデータ点をクラスタに割り当て"""
        return self._assign_clusters(X, self.cluster_centers_)

# 使用例
X, y_true = make_blobs(n_samples=500, centers=4, random_state=42)
km = KMeansFromScratch(n_clusters=4, n_init=10)
km.fit(X)
print(f"慣性: {km.inertia_:.2f}")
print(f"セントロイド形状: {km.cluster_centers_.shape}")
```

### コード例1c: シルエット分析の詳細可視化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs
import matplotlib.cm as cm

def plot_silhouette_analysis(X, k_range=range(2, 7)):
    """各Kでのシルエットプロットを並べて描画"""
    fig, axes = plt.subplots(1, len(k_range), figsize=(5 * len(k_range), 6))

    for ax, k in zip(axes, k_range):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)

        sil_avg = silhouette_score(X, labels)
        sil_samples = silhouette_samples(X, labels)

        y_lower = 10
        for i in range(k):
            cluster_sil = sil_samples[labels == i]
            cluster_sil.sort()
            size = cluster_sil.shape[0]
            y_upper = y_lower + size

            color = cm.nipy_spectral(float(i) / k)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0, cluster_sil,
                facecolor=color, edgecolor=color, alpha=0.7
            )
            ax.text(-0.05, y_lower + 0.5 * size, str(i))
            y_lower = y_upper + 10

        ax.set_title(f"K={k}, Avg={sil_avg:.3f}")
        ax.set_xlabel("シルエット係数")
        ax.axvline(x=sil_avg, color="red", linestyle="--")
        ax.set_xlim([-0.2, 1.0])
        ax.set_yticks([])

    plt.suptitle("シルエット分析によるクラスタ数選定", fontsize=14)
    plt.tight_layout()
    plt.savefig("reports/silhouette_analysis.png", dpi=150)
    plt.close()

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)
plot_silhouette_analysis(X)
```

### コード例1d: Mini-Batch K-means（大規模データ対応）

```python
from sklearn.cluster import MiniBatchKMeans, KMeans
import numpy as np
import time

def compare_kmeans_scalability(n_samples_list, n_clusters=5):
    """標準K-meansとMini-Batch K-meansの速度比較"""
    results = []

    for n in n_samples_list:
        X = np.random.randn(n, 10)

        # 標準K-means
        start = time.time()
        km = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
        km.fit(X)
        km_time = time.time() - start

        # Mini-Batch K-means
        start = time.time()
        mbkm = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000,
                                n_init=3, random_state=42)
        mbkm.fit(X)
        mb_time = time.time() - start

        results.append({
            "n_samples": n,
            "kmeans_time": km_time,
            "mbkmeans_time": mb_time,
            "speedup": km_time / mb_time,
            "inertia_ratio": mbkm.inertia_ / km.inertia_,
        })
        print(f"n={n:>8,}: KMeans={km_time:.2f}s  "
              f"MiniBatch={mb_time:.2f}s  "
              f"高速化={km_time/mb_time:.1f}x  "
              f"慣性比={mbkm.inertia_/km.inertia_:.4f}")

    return results

# 使用例
results = compare_kmeans_scalability([1000, 5000, 10000, 50000, 100000])
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

### DBSCANアルゴリズムの詳細フロー

```
DBSCAN の処理ステップ:

1. 全データ点を「未訪問」に設定
2. 未訪問の点pを選択
3. pのeps近傍を取得（N(p)）
4. |N(p)| >= min_samples なら:
   a. pを「コア点」としてマーク
   b. 新しいクラスタCを作成し、pをCに追加
   c. N(p)の各点qについて:
      - qが未訪問なら:
        - qを「訪問済み」に設定
        - qのeps近傍N(q)を取得
        - |N(q)| >= min_samples ならN(p)にN(q)を追加
      - qがどのクラスタにも属していないなら:
        - qをCに追加（ボーダー点）
5. |N(p)| < min_samples なら:
   a. pを一時的に「ノイズ」とマーク
   b. 後のステップでコア点のeps内にあれば「ボーダー点」になりうる
6. 全ての点が訪問済みになるまで2-5を繰り返す

計算量:
  ・空間インデックス（kd-tree, ball-tree）使用時: O(n log n)
  ・ブルートフォース: O(n²)
  ・メモリ: O(n)
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

### コード例2b: DBSCAN のパラメータグリッド探索

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_moons

X, y_true = make_moons(n_samples=500, noise=0.1, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

eps_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
min_samples_range = [3, 5, 7, 10, 15]

fig, axes = plt.subplots(len(min_samples_range), len(eps_range),
                          figsize=(4 * len(eps_range),
                                   3 * len(min_samples_range)))

for i, ms in enumerate(min_samples_range):
    for j, eps in enumerate(eps_range):
        db = DBSCAN(eps=eps, min_samples=ms)
        labels = db.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()

        ax = axes[i][j]
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1],
                   c=labels, cmap="tab10", s=5, alpha=0.7)
        ax.set_title(f"eps={eps}, ms={ms}\n"
                     f"C={n_clusters}, noise={n_noise}",
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle("DBSCAN パラメータ感度分析", fontsize=14)
plt.tight_layout()
plt.savefig("reports/dbscan_param_grid.png", dpi=150)
plt.close()
```

### コード例2c: HDBSCAN（階層的DBSCAN）

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
import hdbscan

def compare_dbscan_hdbscan(X, y_true=None):
    """DBSCANとHDBSCANの比較"""
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    X_scaled = StandardScaler().fit_transform(X)

    # DBSCAN
    db = DBSCAN(eps=0.3, min_samples=5)
    db_labels = db.fit_predict(X_scaled)

    # HDBSCAN（epsの指定不要）
    hdb = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
    hdb_labels = hdb.fit_predict(X_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # DBSCAN
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise_db = (db_labels == -1).sum()
    ax1.scatter(X_scaled[:, 0], X_scaled[:, 1],
                c=db_labels, cmap="tab10", s=10, alpha=0.7)
    ax1.set_title(f"DBSCAN: {n_clusters_db}クラスタ, ノイズ{n_noise_db}")

    # HDBSCAN
    n_clusters_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    n_noise_hdb = (hdb_labels == -1).sum()
    ax2.scatter(X_scaled[:, 0], X_scaled[:, 1],
                c=hdb_labels, cmap="tab10", s=10, alpha=0.7)
    ax2.set_title(f"HDBSCAN: {n_clusters_hdb}クラスタ, ノイズ{n_noise_hdb}")

    plt.tight_layout()
    plt.savefig("reports/dbscan_vs_hdbscan.png", dpi=150)
    plt.close()

    # HDBSCANの信頼度情報
    print(f"HDBSCAN クラスタリング確率統計:")
    print(f"  平均確率: {hdb.probabilities_.mean():.3f}")
    print(f"  最小確率: {hdb.probabilities_.min():.3f}")
    print(f"  確率>0.5のサンプル: {(hdb.probabilities_ > 0.5).sum()}")

    return db_labels, hdb_labels

# 密度が不均一なデータでテスト
np.random.seed(42)
X1, _ = make_blobs(n_samples=200, centers=[[0, 0]], cluster_std=0.5)
X2, _ = make_blobs(n_samples=50, centers=[[5, 5]], cluster_std=0.3)
X3, _ = make_blobs(n_samples=300, centers=[[3, -2]], cluster_std=1.5)
X = np.vstack([X1, X2, X3])

compare_dbscan_hdbscan(X)
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

### コード例4b: 連結法の視覚的比較

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons, make_circles, make_blobs

datasets = {
    "Blobs": make_blobs(n_samples=300, centers=3, cluster_std=0.8,
                         random_state=42),
    "Moons": make_moons(n_samples=300, noise=0.1, random_state=42),
    "Circles": make_circles(n_samples=300, noise=0.05, factor=0.5,
                             random_state=42),
    "Aniso": (None, None),  # 異方性データ
}

# 異方性データの生成
np.random.seed(42)
X_aniso, y_aniso = make_blobs(n_samples=300, centers=3, random_state=42)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X_aniso, transformation)
datasets["Aniso"] = (X_aniso, y_aniso)

linkages = ["ward", "complete", "average", "single"]

fig, axes = plt.subplots(len(datasets), len(linkages) + 1,
                          figsize=(4 * (len(linkages) + 1),
                                   3.5 * len(datasets)))

for row, (name, (X, y)) in enumerate(datasets.items()):
    # 元データ
    axes[row][0].scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", s=10)
    axes[row][0].set_title(f"{name}\n(正解)")

    for col, link in enumerate(linkages, 1):
        try:
            agg = AgglomerativeClustering(
                n_clusters=3, linkage=link,
                connectivity=None
            )
            labels = agg.fit_predict(X)
            axes[row][col].scatter(X[:, 0], X[:, 1],
                                    c=labels, cmap="tab10", s=10)
        except Exception:
            axes[row][col].text(0.5, 0.5, "Error",
                                transform=axes[row][col].transAxes)

        axes[row][col].set_title(f"{link}")
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])

plt.suptitle("データセット形状 x 連結法 の比較", fontsize=14)
plt.tight_layout()
plt.savefig("reports/linkage_comparison.png", dpi=150)
plt.close()
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

## 4. Gaussian Mixture Model（GMM）

### GMMの仕組み

```
K-means vs GMM の違い:

K-means:
  ・各点を1つのクラスタに「ハード」割り当て
  ・クラスタ形状は球状のみ
  ・パラメータ: セントロイド μ_k

GMM:
  ・各点の各クラスタへの所属確率を「ソフト」計算
  ・クラスタ形状は楕円形（共分散行列で表現）
  ・パラメータ: μ_k, Σ_k, π_k

  確率モデル:
  p(x) = Σ_{k=1}^{K} π_k × N(x | μ_k, Σ_k)

  π_k: 混合比率（各クラスタの重み）
  μ_k: 平均ベクトル
  Σ_k: 共分散行列

  推定方法: EM（Expectation-Maximization）アルゴリズム
  ・E step: 各点の各クラスタへの所属確率を計算
  ・M step: 所属確率に基づいてパラメータを更新
```

### コード例5b: GMMの実装と活用

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse

def plot_gmm_with_ellipses(X, n_components, covariance_type="full"):
    """GMMの結果を楕円付きで可視化"""
    gmm = GaussianMixture(n_components=n_components,
                           covariance_type=covariance_type,
                           random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ハード割り当て
    ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=10, alpha=0.7)
    ax1.set_title("GMM ハード割り当て")

    # ソフト割り当て（不確実性を色で表現）
    uncertainty = 1 - probs.max(axis=1)
    scatter = ax2.scatter(X[:, 0], X[:, 1], c=uncertainty,
                           cmap="YlOrRd", s=10, alpha=0.7)
    plt.colorbar(scatter, ax=ax2, label="不確実性")
    ax2.set_title("GMM 不確実性マップ")

    # 楕円の描画
    for mean, cov in zip(gmm.means_, gmm.covariances_):
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        for nsig in [1, 2, 3]:
            width, height = 2 * nsig * np.sqrt(eigenvalues)
            ellipse = Ellipse(
                xy=mean, width=width, height=height,
                angle=angle, fill=False, edgecolor="black",
                linewidth=1, alpha=0.5 - nsig * 0.1
            )
            ax1.add_patch(ellipse)

    plt.tight_layout()
    plt.savefig("reports/gmm_analysis.png", dpi=150)
    plt.close()

    # BIC/AIC によるモデル選択
    print(f"BIC: {gmm.bic(X):.2f}")
    print(f"AIC: {gmm.aic(X):.2f}")
    print(f"対数尤度: {gmm.score(X):.4f}")

    return gmm, labels

# 使用例
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=[1.0, 1.5, 0.5, 1.2],
                   random_state=42)
gmm, labels = plot_gmm_with_ellipses(X, n_components=4)
```

### コード例5c: BIC/AICによるクラスタ数選定

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def select_n_components_gmm(X, max_components=10):
    """BIC/AICでGMMの最適コンポーネント数を選定"""
    n_range = range(1, max_components + 1)
    bics = []
    aics = []
    log_likelihoods = []

    for n in n_range:
        gmm = GaussianMixture(n_components=n, random_state=42, n_init=5)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))
        log_likelihoods.append(gmm.score(X) * X.shape[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(n_range, bics, "bo-", label="BIC")
    ax1.plot(n_range, aics, "ro-", label="AIC")
    ax1.set_xlabel("コンポーネント数")
    ax1.set_ylabel("情報量基準")
    ax1.set_title("BIC/AIC によるモデル選択")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 最適コンポーネント数
    best_n_bic = n_range[np.argmin(bics)]
    best_n_aic = n_range[np.argmin(aics)]
    ax1.axvline(best_n_bic, color="blue", linestyle="--", alpha=0.5,
                label=f"BIC最適: {best_n_bic}")
    ax1.axvline(best_n_aic, color="red", linestyle="--", alpha=0.5,
                label=f"AIC最適: {best_n_aic}")
    ax1.legend()

    ax2.plot(n_range, log_likelihoods, "go-")
    ax2.set_xlabel("コンポーネント数")
    ax2.set_ylabel("対数尤度")
    ax2.set_title("対数尤度の推移")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reports/gmm_model_selection.png", dpi=150)
    plt.close()

    print(f"BIC最適コンポーネント数: {best_n_bic}")
    print(f"AIC最適コンポーネント数: {best_n_aic}")
    return best_n_bic, best_n_aic

# 使用例
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=500, centers=4, random_state=42)
select_n_components_gmm(X)
```

---

## 5. クラスタリングの応用パターン

### コード例6: テキストクラスタリング

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from collections import Counter

def text_clustering(documents, n_clusters=5, n_top_terms=10):
    """TF-IDF + SVD + K-meansによるテキストクラスタリング"""

    # TF-IDFベクトル化
    vectorizer = TfidfVectorizer(
        max_df=0.5, min_df=2, max_features=10000,
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    print(f"TF-IDF行列: {tfidf_matrix.shape}")

    # SVDで次元削減（潜在意味解析）
    svd = TruncatedSVD(n_components=50, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa_pipeline = make_pipeline(svd, normalizer)
    X_lsa = lsa_pipeline.fit_transform(tfidf_matrix)
    print(f"説明分散比: {svd.explained_variance_ratio_.sum():.2%}")

    # K-meansクラスタリング
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X_lsa)

    # 各クラスタの特徴的な単語を抽出
    terms = vectorizer.get_feature_names_out()
    original_centroids = svd.inverse_transform(km.cluster_centers_)

    print(f"\n=== {n_clusters}クラスタの特徴語 ===")
    for i in range(n_clusters):
        top_term_indices = original_centroids[i].argsort()[::-1][:n_top_terms]
        top_terms = [terms[idx] for idx in top_term_indices]
        cluster_size = (labels == i).sum()
        print(f"  クラスタ{i} ({cluster_size}文書): {', '.join(top_terms)}")

    return labels, vectorizer, km

# 使用例（20 Newsgroupsデータセット）
# from sklearn.datasets import fetch_20newsgroups
# data = fetch_20newsgroups(subset="train", remove=("headers", "footers"))
# labels, vec, km = text_clustering(data.data, n_clusters=5)
```

### コード例7: 画像セグメンテーション（K-means）

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

def segment_image_kmeans(image_array, n_segments=5):
    """
    画像をK-meansでセグメンテーション

    Parameters:
        image_array: (H, W, 3) のNumPy配列（RGB画像）
        n_segments: セグメント数
    """
    h, w, c = image_array.shape

    # ピクセルを特徴ベクトルに変換
    # 特徴: [R, G, B, x_normalized, y_normalized]
    pixels = image_array.reshape(-1, c).astype(np.float32)

    # 位置情報を追加（空間的な一貫性のため）
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    positions = np.column_stack([
        xx.ravel() / w,  # 正規化x座標
        yy.ravel() / h   # 正規化y座標
    ])

    # 色と位置を結合
    features = np.hstack([pixels / 255.0, positions * 0.5])

    # Mini-Batch K-means（大画像対応）
    km = MiniBatchKMeans(n_clusters=n_segments, batch_size=1000,
                          random_state=42)
    labels = km.fit_predict(features)

    # セグメンテーション結果を画像に変換
    segmented = km.cluster_centers_[:, :3][labels] * 255
    segmented = segmented.reshape(h, w, c).astype(np.uint8)

    # ラベルマップ
    label_map = labels.reshape(h, w)

    return segmented, label_map

# # 使用例
# from PIL import Image
# img = np.array(Image.open("sample.jpg"))
# segmented, labels = segment_image_kmeans(img, n_segments=8)
```

### コード例8: 異常検知としてのクラスタリング

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

class ClusterBasedAnomalyDetector:
    """クラスタリングベースの異常検知"""

    def __init__(self, n_clusters: int = 5, percentile: float = 95):
        self.n_clusters = n_clusters
        self.percentile = percentile
        self.scaler = StandardScaler()
        self.km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        self.threshold = None

    def fit(self, X: np.ndarray):
        """正常データで学習"""
        X_scaled = self.scaler.fit_transform(X)
        self.km.fit(X_scaled)

        # 各点のセントロイドからの距離を計算
        distances = self._compute_distances(X_scaled)

        # 閾値を設定（パーセンタイル）
        self.threshold = np.percentile(distances, self.percentile)
        print(f"異常検知閾値: {self.threshold:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """異常スコアと予測を返す"""
        X_scaled = self.scaler.transform(X)
        distances = self._compute_distances(X_scaled)

        # 閾値を超えたら異常（-1）
        predictions = np.where(distances > self.threshold, -1, 1)
        return predictions, distances

    def _compute_distances(self, X_scaled: np.ndarray) -> np.ndarray:
        """各点の最近接セントロイドからの距離"""
        labels = self.km.predict(X_scaled)
        centroids = self.km.cluster_centers_
        distances = np.sqrt(
            np.sum((X_scaled - centroids[labels]) ** 2, axis=1)
        )
        return distances

# 使用例
np.random.seed(42)
X_normal = np.random.randn(1000, 5)
X_anomaly = np.random.randn(50, 5) * 3 + 5
X_all = np.vstack([X_normal, X_anomaly])
y_true = np.array([1] * 1000 + [-1] * 50)

detector = ClusterBasedAnomalyDetector(n_clusters=5, percentile=95)
detector.fit(X_normal)
predictions, scores = detector.predict(X_all)

# 評価
from sklearn.metrics import classification_report
print(classification_report(y_true, predictions, target_names=["異常", "正常"]))
```

---

## 6. トラブルシューティング

### よくある問題と解決策

```
問題1: K-meansが収束しない / 結果が毎回異なる
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原因:
  ・ランダムシードが固定されていない
  ・初期化回数（n_init）が不足
  ・データにスケーリングが適用されていない

解決策:
  ・random_state を固定する
  ・n_init を 10〜20 に設定する（デフォルト10）
  ・StandardScaler でスケーリングを必ず適用
  ・K-means++ 初期化を使用（scikit-learnのデフォルト）

問題2: DBSCANが1つの巨大クラスタしか返さない
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原因:
  ・eps が大きすぎる
  ・min_samples が小さすぎる
  ・スケーリングされていないデータ

解決策:
  ・k-距離グラフで適切な eps を推定する
  ・min_samples を次元数 × 2 程度に設定する
  ・StandardScaler でスケーリングする

問題3: クラスタ数の決定に困る
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原因:
  ・データに明確なクラスタ構造がない場合もある
  ・単一の指標に頼っている

解決策:
  ・複数の指標を組み合わせる（エルボー法 + シルエット + BIC/AIC）
  ・ドメイン知識を活用する
  ・Gap統計量を使う
  ・HDBSCANで自動推定する

問題4: 高次元データでクラスタリングがうまくいかない
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原因:
  ・「次元の呪い」：高次元ではユークリッド距離が意味をなさなくなる
  ・ノイズ特徴量がクラスタ構造を覆い隠す

解決策:
  ・PCA / UMAP で次元削減してからクラスタリング
  ・特徴量選択でノイズ特徴量を除去
  ・Subspace clustering を検討
  ・コサイン距離など適切な距離尺度を使用
```

### コード例9: クラスタリング品質の総合評価

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score, adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

def comprehensive_clustering_evaluation(X, y_true=None, n_clusters=4):
    """複数の手法・指標でクラスタリングを総合評価"""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    methods = {
        "K-means": KMeans(n_clusters=n_clusters, n_init=10, random_state=42),
        "K-means (Mini)": KMeans(n_clusters=n_clusters, n_init=10,
                                  random_state=42, algorithm="elkan"),
        "Hierarchical (Ward)": AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward"),
        "GMM": GaussianMixture(n_components=n_clusters, random_state=42),
    }

    results = []
    for name, model in methods.items():
        if hasattr(model, "fit_predict"):
            labels = model.fit_predict(X_scaled)
        else:
            model.fit(X_scaled)
            labels = model.predict(X_scaled)

        row = {
            "手法": name,
            "シルエット": silhouette_score(X_scaled, labels),
            "CH指数": calinski_harabasz_score(X_scaled, labels),
            "DB指数": davies_bouldin_score(X_scaled, labels),
        }

        if y_true is not None:
            row["ARI"] = adjusted_rand_score(y_true, labels)
            row["NMI"] = normalized_mutual_info_score(y_true, labels)

        results.append(row)

    df = pd.DataFrame(results)
    print("=== クラスタリング手法の総合比較 ===")
    print(df.to_string(index=False, float_format="%.4f"))
    return df

# 使用例
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=1.0,
                         random_state=42)
evaluation = comprehensive_clustering_evaluation(X, y_true, n_clusters=4)
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
| 階層的 (Ward) | 球状 | 後決め可 | 弱い | O(n^2) | 低い | 小〜中規模、構造探索 |
| GMM | 楕円形 | 必要 | 中程度 | O(nK^2d) | 中程度 | 確率的クラスタ割り当て |
| Spectral | 任意形状 | 必要 | 弱い | O(n^3) | 低い | グラフ構造のデータ |

### クラスタ評価指標の使い分け

| 指標 | 正解ラベル | 範囲 | 解釈 | 用途 |
|---|---|---|---|---|
| シルエットスコア | 不要 | [-1, 1] | 高いほどクラスタが明確 | K選択、品質評価 |
| Calinski-Harabasz | 不要 | [0, +inf) | 高いほど良い | K選択 |
| Davies-Bouldin | 不要 | [0, +inf) | 低いほど良い | K選択 |
| ARI (調整ランド指数) | 必要 | [-1, 1] | 1=完全一致、0=ランダム | 正解比較 |
| NMI (正規化相互情報量) | 必要 | [0, 1] | 1=完全一致 | 正解比較 |
| V-measure | 必要 | [0, 1] | 均質性×完全性の調和平均 | 正解比較 |
| Gap統計量 | 不要 | 実数 | 最大値のKが最適 | K選択（統計的） |

### 距離尺度の選択ガイド

| 距離尺度 | 数式 | 適用データ | 特徴 |
|---|---|---|---|
| ユークリッド | sqrt(sum((x-y)^2)) | 低〜中次元の連続値 | 最も一般的 |
| マンハッタン | sum(abs(x-y)) | 高次元、外れ値がある場合 | 外れ値に強い |
| コサイン | 1 - cos(x,y) | テキスト、高次元スパース | 方向のみ考慮 |
| マハラノビス | sqrt((x-y)^T S^-1 (x-y)) | 相関のある特徴量 | 共分散を考慮 |
| ワード法距離 | WCSS増加量 | 階層的クラスタリング | クラスタの分散を最小化 |

---

## ベストプラクティス

### クラスタリングプロジェクトのワークフロー

```
1. データ理解・前処理
   ├── 欠損値の処理
   ├── 外れ値の検出・処理
   ├── スケーリング（StandardScaler推奨）
   └── 次元削減の検討（高次元の場合）

2. 探索的分析
   ├── ペアプロットで2D構造を確認
   ├── 相関分析で冗長な特徴量を特定
   └── 分布の確認（歪度、多峰性）

3. クラスタリング実行
   ├── 複数の手法を試す（K-means, DBSCAN, GMM等）
   ├── クラスタ数を複数の指標で決定
   └── パラメータの感度分析

4. 結果の評価
   ├── 内部指標（シルエット、CH指数等）
   ├── 可視化（2D射影、特徴量の箱ひげ図）
   ├── ドメイン専門家によるレビュー
   └── クラスタの安定性検証（ブートストラップ）

5. 解釈・活用
   ├── 各クラスタのプロファイリング
   ├── ビジネスアクションへの変換
   └── 定期的な再クラスタリングの計画
```

### チェックリスト

```
前処理:
  [ ] スケーリングを適用したか
  [ ] 外れ値を処理したか
  [ ] 欠損値を処理したか
  [ ] カテゴリ変数をエンコーディングしたか
  [ ] 高次元の場合、次元削減を検討したか

クラスタリング:
  [ ] ランダムシードを固定したか
  [ ] 複数の手法を比較したか
  [ ] クラスタ数を複数指標で決定したか
  [ ] パラメータの感度分析を行ったか

評価:
  [ ] 複数の評価指標を確認したか
  [ ] 可視化で結果を確認したか
  [ ] クラスタの安定性を検証したか
  [ ] ドメイン知識と整合するか確認したか

デプロイ:
  [ ] 新規データへの適用方法を決定したか（predict vs 再学習）
  [ ] 再クラスタリングの頻度を決定したか
  [ ] クラスタの変化をモニタリングする仕組みがあるか
```

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

### アンチパターン3: クラスタの過信

```python
# BAD: クラスタリング結果をそのまま事実として扱う
clusters = km.fit_predict(X)
df["customer_type"] = clusters
# "クラスタ0は優良顧客です" → 本当に？

# GOOD: 安定性と解釈可能性を検証する
from sklearn.utils import resample

n_bootstrap = 50
cluster_stability = np.zeros((len(X), n_bootstrap))

for i in range(n_bootstrap):
    X_boot, idx = resample(X, return_indices=True, random_state=i)
    km_boot = KMeans(n_clusters=k, n_init=5, random_state=42)
    km_boot.fit(X_boot)
    labels_boot = km_boot.predict(X)
    cluster_stability[:, i] = labels_boot

# 各サンプルのクラスタ割り当ての安定性を計算
from scipy.stats import mode
stability_scores = []
for j in range(len(X)):
    most_common = mode(cluster_stability[j], keepdims=True)[1][0]
    stability_scores.append(most_common / n_bootstrap)

avg_stability = np.mean(stability_scores)
print(f"平均クラスタ安定性: {avg_stability:.3f}")
# 0.8以上なら安定、0.5以下なら不安定
```

### アンチパターン4: カテゴリ変数への直接K-means適用

```python
# BAD: カテゴリ変数をワンホットエンコードしてK-means
# → ユークリッド距離がカテゴリ変数に対して不適切

# GOOD: K-modes（カテゴリ用）or K-prototypes（混合型）を使用
# pip install kmodes
from kmodes.kprototypes import KPrototypes

# 数値列とカテゴリ列が混在
X_mixed = df[["age", "income", "city", "education"]].values
categorical_indices = [2, 3]  # カテゴリ列のインデックス

kp = KPrototypes(n_clusters=4, init="Cao", random_state=42)
labels = kp.fit_predict(X_mixed, categorical=categorical_indices)
```

---

## FAQ

### Q1: K-meansとGMMの使い分けは？

**A:** K-meansは各点を1つのクラスタに「ハード」に割り当てるが、GMMは確率的な「ソフト」割り当てが可能。クラスタが重なり合う場合やクラスタの形が楕円形の場合はGMMが適している。K-meansは高速で大規模データに向く。ただしGMMはEMアルゴリズムの初期化に敏感で、局所最適に陥りやすいため、n_initを大きめに設定する。

### Q2: DBSCANのepsとmin_samplesの決め方は？

**A:** min_samplesは次元数*2が目安。epsはk-距離グラフ（k=min_samples）の「肘」の位置から読み取る。データのドメイン知識がある場合は、「同一クラスタとみなせる最大距離」から設定する。epsが分からない場合はHDBSCANを使うとeps指定が不要になる。

### Q3: クラスタリング結果の「正しさ」はどう判断する？

**A:** 正解ラベルがない場合、単一の正解はない。(1) シルエットスコアなどの内部指標、(2) ドメイン専門家によるクラスタの解釈可能性、(3) 下流タスク（マーケティング施策等）での有効性、の3軸で総合的に判断する。

### Q4: 大規模データ（数百万行以上）にはどの手法が適しているか？

**A:** Mini-Batch K-meansが第一候補。O(n)の計算量で、バッチサイズを調整することでメモリ消費も制御可能。密度ベースが必要な場合はHDBSCANのapproximate_predict機能を使う。大規模データでは、まずサンプリングで手法を比較し、最適な手法を全データに適用するアプローチも有効。Daskやsparkmlにも分散クラスタリングの実装がある。

### Q5: 時系列データのクラスタリングはどうすればよいか？

**A:** 時系列データには通常のユークリッド距離が不適切な場合が多い。(1) DTW（Dynamic Time Warping）距離を使ったクラスタリング、(2) 時系列から特徴量を抽出（tsfreshなど）してから標準的クラスタリング、(3) TimeSeriesKMeans（tslearnライブラリ）を使用。時系列の長さが不均一な場合はDTWが特に有効。

### Q6: クラスタリングの結果をどうビジネスに活用するか？

**A:** 主なパターン: (1) 顧客セグメンテーション→セグメント別マーケティング施策、(2) 異常検知→クラスタから外れたデータ点を異常とみなす、(3) データの圧縮→各クラスタの代表点でデータを要約、(4) ラベリング支援→クラスタ内サンプリングで効率的なアノテーション。ビジネスインパクトを明確にするには、クラスタの特性を非技術者にも分かりやすく説明することが重要。

---

## まとめ

| 項目 | 要点 |
|---|---|
| K-means | 高速・シンプル。球状クラスタ向け。Kはエルボー法+シルエットで選定 |
| DBSCAN | 任意形状対応。ノイズ検出可能。パラメータはk-距離グラフで決定 |
| HDBSCAN | DBSCAN改良版。eps不要。密度が不均一なデータに強い |
| 階層的 | デンドログラムで構造可視化。小〜中規模データ向け |
| GMM | 楕円形クラスタ対応。確率的割り当て。BIC/AICでモデル選択 |
| 評価 | 内部指標（シルエット等）+ ドメイン知識で総合判断 |
| 前処理 | スケーリングは必須（距離ベースの手法全般） |
| 応用 | テキスト、画像、異常検知、顧客セグメンテーション |

---

## 次に読むべきガイド

- [03-dimensionality-reduction.md](./03-dimensionality-reduction.md) — 次元削減でクラスタ構造を可視化
- [../03-applied/00-nlp.md](../03-applied/00-nlp.md) — テキストクラスタリングの応用

---

## 参考文献

1. **Martin Ester et al.** "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise" KDD 1996
2. **scikit-learn** "Clustering" -- https://scikit-learn.org/stable/modules/clustering.html
3. **Lior Rokach, Oded Maimon** "Clustering Methods" in Data Mining and Knowledge Discovery Handbook, Springer, 2005
4. **Leland McInnes et al.** "hdbscan: Hierarchical density based clustering" JOSS, 2017
5. **Arthur, D. and Vassilvitskii, S.** "k-means++: The advantages of careful seeding" SODA 2007

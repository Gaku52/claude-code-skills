# 次元削減 — PCA、t-SNE、UMAP

> 高次元データを低次元に圧縮し、可視化・ノイズ除去・計算効率化を実現する手法を解説する

## この章で学ぶこと

1. **PCA（主成分分析）** — 線形射影によるデータの分散最大化と次元削減
2. **t-SNE** — 非線形な局所構造保存による高次元データの可視化
3. **UMAP** — 大域構造も保持する高速な非線形次元削減

---

## 1. PCA（主成分分析）

### PCAの幾何学的直感

```
元の2D空間:                 PCA後（第1主成分方向）:

  y │    *   *                 PC1
    │   * * *  *              ←──────────────────────→
    │  *  *  *                  * * * * * * * *
    │ *  * *                    (分散が最大の方向に射影)
    │*  *
    └──────────── x

  PCAの手順:
  1. データの共分散行列 C = (1/n) X^T X を計算
  2. Cの固有値分解 → 固有ベクトル（主成分方向）
  3. 大きい固有値の方向がデータの分散を多く説明
  4. 上位k個の固有ベクトルで射影

  寄与率:
  ┌─────┬─────┬─────┬─────┬─────┬────┐
  │ PC1 │ PC2 │ PC3 │ PC4 │ PC5 │... │
  │ 45% │ 25% │ 15% │ 8%  │ 4%  │... │
  │█████│████ │███  │██   │█    │    │
  └─────┴─────┴─────┴─────┴─────┴────┘
  累積 45%   70%   85%   93%   97%
  → 上位3成分で85%の情報を保持
```

### コード例1: PCA実装と寄与率分析

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

# 手書き数字データ（64次元）
digits = load_digits()
X, y = digits.data, digits.target

# スケーリング
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA: 全成分で固有値分析
pca_full = PCA()
pca_full.fit(X_scaled)

# 累積寄与率プロット
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
        pca_full.explained_variance_ratio_, alpha=0.6, label="個別寄与率")
ax1.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
         np.cumsum(pca_full.explained_variance_ratio_), "ro-", label="累積寄与率")
ax1.axhline(y=0.95, color="g", linestyle="--", label="95%ライン")
ax1.set_xlabel("主成分番号")
ax1.set_ylabel("寄与率")
ax1.set_title("PCA 寄与率分析")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 95%を満たす次元数
n_components_95 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95) + 1
print(f"95%の分散を保持する次元数: {n_components_95} / {X.shape[1]}")

# 2Dに射影して可視化
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

scatter = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=5, alpha=0.6)
ax2.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})")
ax2.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})")
ax2.set_title("PCA 2D射影（手書き数字）")
plt.colorbar(scatter, ax=ax2)

plt.tight_layout()
plt.savefig("reports/pca_analysis.png", dpi=150)
plt.close()
```

---

## 2. t-SNE

### t-SNEの動作原理

```
t-SNEの2ステップ:

Step 1: 高次元空間でのペアワイズ類似度
  ┌──────────────────────────────────┐
  │ 各点ペア(i,j)について             │
  │ ガウス分布で条件付き確率を計算:   │
  │                                  │
  │ p(j|i) = exp(-||xi-xj||²/2σi²) │
  │           / Σk exp(...)          │
  │                                  │
  │ → 近い点ほど高い確率              │
  └──────────────────────────────────┘
              │
              v
Step 2: 低次元空間で再配置
  ┌──────────────────────────────────┐
  │ t分布（裾が重い）で類似度を定義: │
  │                                  │
  │ q(j|i) = (1+||yi-yj||²)^(-1)   │
  │           / Σk (...)             │
  │                                  │
  │ KL(P||Q) を最小化する y を探索   │
  │ → 近い点は近く、遠い点は遠く     │
  └──────────────────────────────────┘

  なぜt分布？ → 「Crowding Problem」の解決
  高次元の「中距離」の点が低次元で潰れるのを防ぐ
```

### コード例2: t-SNEの実装とperplexity比較

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

# perplexityの影響を比較
perplexities = [5, 15, 30, 50, 100]
fig, axes = plt.subplots(1, len(perplexities), figsize=(25, 5))

for ax, perp in zip(axes, perplexities):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                n_iter=1000, learning_rate="auto", init="pca")
    X_tsne = tsne.fit_transform(X)

    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y,
                         cmap="tab10", s=5, alpha=0.6)
    ax.set_title(f"perplexity={perp}")
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("t-SNE: perplexityの影響", fontsize=14)
plt.tight_layout()
plt.savefig("reports/tsne_perplexity.png", dpi=150)
plt.close()
```

---

## 3. UMAP

### コード例3: UMAP実装と比較

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.datasets import load_digits
import time

digits = load_digits()
X, y = digits.data, digits.target

methods = {
    "PCA": lambda: PCA(n_components=2).fit_transform(X),
    "t-SNE": lambda: TSNE(n_components=2, random_state=42,
                           init="pca", learning_rate="auto").fit_transform(X),
    "UMAP": lambda: umap.UMAP(n_components=2, random_state=42).fit_transform(X),
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, method) in zip(axes, methods.items()):
    start = time.time()
    X_2d = method()
    elapsed = time.time() - start

    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10",
                         s=5, alpha=0.6)
    ax.set_title(f"{name} ({elapsed:.2f}秒)")
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("次元削減手法の比較（手書き数字データ）", fontsize=14)
plt.colorbar(scatter, ax=axes, shrink=0.8)
plt.tight_layout()
plt.savefig("reports/dim_reduction_comparison.png", dpi=150)
plt.close()
```

### コード例4: UMAPのハイパーパラメータ探索

```python
import umap
import numpy as np
import matplotlib.pyplot as plt

n_neighbors_list = [5, 15, 50, 200]
min_dist_list = [0.0, 0.1, 0.5, 0.99]

fig, axes = plt.subplots(len(n_neighbors_list), len(min_dist_list),
                          figsize=(20, 20))

for i, nn in enumerate(n_neighbors_list):
    for j, md in enumerate(min_dist_list):
        reducer = umap.UMAP(n_neighbors=nn, min_dist=md,
                            n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X)

        axes[i][j].scatter(X_umap[:, 0], X_umap[:, 1], c=y,
                           cmap="tab10", s=3, alpha=0.5)
        axes[i][j].set_title(f"nn={nn}, md={md}", fontsize=9)
        axes[i][j].set_xticks([])
        axes[i][j].set_yticks([])

    axes[i][0].set_ylabel(f"n_neighbors={nn}")

for j, md in enumerate(min_dist_list):
    axes[0][j].set_xlabel(f"min_dist={md}")

plt.suptitle("UMAP パラメータグリッド", fontsize=14)
plt.tight_layout()
plt.savefig("reports/umap_params.png", dpi=150)
plt.close()
```

### コード例5: 次元削減を前処理として活用

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# 高次元データに対してPCAで前処理
from sklearn.datasets import fetch_openml
# mnist = fetch_openml("mnist_784", version=1, as_frame=False)
# X, y = mnist.data[:10000], mnist.target[:10000]

# サンプル: 手書き数字データ
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data, digits.target

# PCA前処理の有無で比較
configs = [
    ("RF (全次元)", make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=100, random_state=42)
    )),
    ("PCA(95%) + RF", make_pipeline(
        StandardScaler(),
        PCA(n_components=0.95),  # 95%の分散を保持
        RandomForestClassifier(n_estimators=100, random_state=42)
    )),
    ("PCA(10) + RF", make_pipeline(
        StandardScaler(),
        PCA(n_components=10),
        RandomForestClassifier(n_estimators=100, random_state=42)
    )),
]

import time
for name, pipe in configs:
    start = time.time()
    scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    elapsed = time.time() - start
    print(f"{name:20s}  Acc={scores.mean():.4f}±{scores.std():.4f}  "
          f"時間={elapsed:.2f}秒")
```

---

## 比較表

### 次元削減手法の特性比較

| 手法 | 種類 | 保持する構造 | 計算量 | 新データ適用 | 主な用途 |
|---|---|---|---|---|---|
| PCA | 線形 | 大域的分散 | O(min(n,d)²×max(n,d)) | 可（transform） | 前処理、ノイズ除去 |
| Kernel PCA | 非線形 | 非線形大域構造 | O(n³) | 可（近似） | 非線形構造の発見 |
| t-SNE | 非線形 | 局所構造 | O(n²) → O(n log n) | 不可 | 2D/3D可視化 |
| UMAP | 非線形 | 局所+大域 | O(n^1.14) | 可（transform） | 可視化、前処理 |
| LDA | 線形（教師あり） | クラス間分離 | O(nd²) | 可 | 分類前処理 |
| Autoencoder | 非線形 | 学習された表現 | NN依存 | 可 | 特徴抽出、生成 |

### パラメータ選択ガイド

| 手法 | 主要パラメータ | 推奨範囲 | 効果 |
|---|---|---|---|
| PCA | n_components | 0.95（寄与率）or 次元数 | 保持する情報量 |
| t-SNE | perplexity | 5〜50（デフォルト30） | 局所/大域バランス |
| t-SNE | learning_rate | "auto" or n/12 | 収束の安定性 |
| t-SNE | n_iter | 1000〜5000 | 収束品質 |
| UMAP | n_neighbors | 5〜200（デフォルト15） | 局所/大域バランス |
| UMAP | min_dist | 0.0〜0.99（デフォルト0.1） | クラスタの密集度 |

---

## アンチパターン

### アンチパターン1: t-SNEの結果を距離として解釈

```
# BAD: t-SNEの距離でクラスタ間の類似性を語る
"t-SNEの図でクラスタAとBが近いから、AとBは似ている"
→ t-SNEは大域的な距離関係を保存しない。
  同じデータでもperplexityを変えると配置が大きく変わる。

# GOOD: t-SNEは「局所的な近傍関係の保存」として解釈
- 同じクラスタ内の点が近い → 局所構造が保存されている
- クラスタ間の距離 → 信頼できない
- クラスタのサイズ → 意味がない（密度が歪む）
- 定量的な分析にはPCAやUMAPを使う
```

### アンチパターン2: 可視化のためだけにPCAで2Dに射影

```python
# BAD: PCA(n_components=2) で寄与率30%しかないのに結論を出す
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
print(f"寄与率: {sum(pca.explained_variance_ratio_):.1%}")
# → 30%の情報しかない2D図で「クラスタが分離していない」と結論

# GOOD: まず寄与率を確認し、適切な手法を選択
pca_full = PCA()
pca_full.fit(X)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
print(f"2D寄与率: {cumvar[1]:.1%}")

if cumvar[1] < 0.5:
    print("PCA 2Dでは情報が不足。t-SNE or UMAPで可視化推奨")
    # t-SNE or UMAP で可視化
```

---

## FAQ

### Q1: PCA、t-SNE、UMAPのどれを使うべき？

**A:** 目的による。(1) 前処理・ノイズ除去 → PCA（線形、transform可能）、(2) 可視化（小〜中規模） → t-SNE（局所構造の保存が最良）、(3) 可視化＋前処理（大規模） → UMAP（高速、transform可能、大域構造も保持）。まずPCAで概観し、必要に応じてt-SNE/UMAPを試す。

### Q2: PCAの成分数はどう決める？

**A:** (1) 累積寄与率が95%以上になる成分数、(2) スクリープロット（固有値の減衰曲線）の「肘」、(3) 下流タスク（分類等）の交差検証スコアで選択。実用的には `PCA(n_components=0.95)` で自動決定できる。

### Q3: t-SNEは再現性がないのか？

**A:** random_stateを固定すれば同じデータで同じ結果が得られる。ただし初期値への依存が強いため、異なるrandom_stateで複数回実行し、結果が安定しているか確認するのが良い。scikit-learn 1.2以降では `init="pca"` がデフォルトとなり再現性が向上している。

---

## まとめ

| 項目 | 要点 |
|---|---|
| PCA | 線形次元削減。分散最大化。前処理・ノイズ除去に最適 |
| t-SNE | 非線形。局所構造保存。2D/3D可視化専用。大域距離は不正確 |
| UMAP | 非線形。局所+大域保持。高速。可視化+前処理に両用可 |
| 選択基準 | 前処理→PCA、可視化→t-SNE/UMAP、大規模→UMAP |
| 注意点 | スケーリング必須。寄与率/パラメータを適切に設定 |

---

## 次に読むべきガイド

- [../02-deep-learning/00-neural-networks.md](../02-deep-learning/00-neural-networks.md) — ニューラルネットワークの基礎
- [../03-applied/01-computer-vision.md](../03-applied/01-computer-vision.md) — 画像データの次元削減応用

---

## 参考文献

1. **Laurens van der Maaten, Geoffrey Hinton** "Visualizing Data using t-SNE" JMLR, 2008
2. **Leland McInnes, John Healy** "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction" 2018 — https://arxiv.org/abs/1802.03426
3. **scikit-learn** "Decomposing signals in components" — https://scikit-learn.org/stable/modules/decomposition.html

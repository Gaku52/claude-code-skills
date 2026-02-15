# 次元削減 — PCA、t-SNE、UMAP

> 高次元データを低次元に圧縮し、可視化・ノイズ除去・計算効率化を実現する手法を解説する

## この章で学ぶこと

1. **PCA（主成分分析）** — 線形射影によるデータの分散最大化と次元削減
2. **t-SNE** — 非線形な局所構造保存による高次元データの可視化
3. **UMAP** — 大域構造も保持する高速な非線形次元削減
4. **Kernel PCA** — カーネルトリックによる非線形 PCA
5. **LDA（線形判別分析）** — 教師あり次元削減によるクラス分離の最大化
6. **Autoencoder** — ニューラルネットワークによる非線形次元削減

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

### 1.1 PCAの数学的基礎

PCA は共分散行列の固有値問題に帰着する。データ行列 X (n x d) を中心化した後、共分散行列 C = (1/n) X^T X の固有値分解を行い、固有値の大きい順に対応する固有ベクトルを取り出す。

```
数学的定式化:

  入力: X ∈ ℝ^(n×d) （中心化済みデータ）
  目標: W ∈ ℝ^(d×k) を見つけて Z = XW (n×k) に射影

  最大化問題:
    max_W  tr(W^T C W)
    s.t.   W^T W = I_k

  解: C の上位 k 個の固有ベクトルを並べた行列 W

  共分散行列の固有値分解:
    C = V Λ V^T

    V = [v1, v2, ..., vd]  固有ベクトル行列
    Λ = diag(λ1, λ2, ..., λd)  固有値（λ1 ≥ λ2 ≥ ... ≥ λd）

  寄与率:
    第 i 主成分の寄与率 = λi / Σλj

  SVD との関係:
    X = U Σ V^T  （特異値分解）
    C = (1/n) V Σ^2 V^T
    → SVD の右特異ベクトル V が主成分方向
    → 特異値の二乗 / n が固有値に対応

  SVD を使う利点:
    - 共分散行列を明示的に計算しない → メモリ効率
    - 数値的に安定
    - scikit-learn の PCA は内部で SVD を使用
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

### コード例1b: PCAをゼロから実装

```python
import numpy as np

class PCAFromScratch:
    """
    PCA をゼロから実装する。

    なぜ自作するのか:
    - 共分散行列 → 固有値分解の流れを理解する
    - scikit-learn の PCA が内部で何をしているかを把握する
    - SVD ベースの実装との違いを体感する
    """

    def __init__(self, n_components: int = 2, method: str = "eigen"):
        """
        Parameters:
            n_components: 保持する主成分数
            method: "eigen"（固有値分解）or "svd"（特異値分解）
        """
        self.n_components = n_components
        self.method = method
        self.components_ = None       # 主成分方向 (k x d)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X: np.ndarray) -> "PCAFromScratch":
        n_samples, n_features = X.shape

        # Step 1: 中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        if self.method == "eigen":
            # Step 2a: 共分散行列の計算
            cov_matrix = (1 / (n_samples - 1)) * X_centered.T @ X_centered

            # Step 3a: 固有値分解
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Step 4a: 固有値の降順でソート
            sorted_idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_idx]
            eigenvectors = eigenvectors[:, sorted_idx]

            # Step 5a: 上位 k 個を選択
            self.components_ = eigenvectors[:, :self.n_components].T
            self.explained_variance_ = eigenvalues[:self.n_components]

        elif self.method == "svd":
            # Step 2b: SVD（共分散行列を作らない）
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

            # Step 3b: 特異値から固有値を計算
            eigenvalues = (S ** 2) / (n_samples - 1)

            # Step 4b: 上位 k 個を選択
            self.components_ = Vt[:self.n_components]
            self.explained_variance_ = eigenvalues[:self.n_components]

        # 寄与率の計算
        total_variance = np.sum(self.explained_variance_)
        # 全分散は全固有値の合計
        if self.method == "eigen":
            total_var_all = np.sum(eigenvalues)
        else:
            total_var_all = np.sum((S ** 2) / (n_samples - 1))

        self.explained_variance_ratio_ = self.explained_variance_ / total_var_all

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """データを主成分空間に射影"""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """主成分空間から元の空間に逆射影（近似復元）"""
        return Z @ self.components_ + self.mean_

    def reconstruction_error(self, X: np.ndarray) -> float:
        """再構成誤差（情報損失の指標）"""
        Z = self.transform(X)
        X_reconstructed = self.inverse_transform(Z)
        return np.mean((X - X_reconstructed) ** 2)

# 使用例: scikit-learn との比較
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data

# ゼロから実装（SVD版）
pca_scratch = PCAFromScratch(n_components=10, method="svd")
Z_scratch = pca_scratch.fit_transform(X)

# scikit-learn
pca_sklearn = SklearnPCA(n_components=10)
Z_sklearn = pca_sklearn.fit_transform(X)

# 寄与率の比較（符号の違いを除けば一致するはず）
print("=== 寄与率の比較 ===")
for i in range(5):
    print(f"  PC{i+1}: 自作={pca_scratch.explained_variance_ratio_[i]:.6f}, "
          f"sklearn={pca_sklearn.explained_variance_ratio_[i]:.6f}")

# 再構成誤差
error = pca_scratch.reconstruction_error(X)
print(f"\n再構成誤差 (10成分): {error:.4f}")
print(f"累積寄与率: {sum(pca_scratch.explained_variance_ratio_):.4f}")
```

### 1.2 PCA の逆変換と画像再構成

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data

# 異なる成分数での再構成を可視化
n_components_list = [2, 5, 10, 20, 40, 64]
sample_idx = 42  # 表示するサンプル

fig, axes = plt.subplots(2, len(n_components_list) + 1, figsize=(20, 6))

# 元画像
for ax in axes[:, 0]:
    ax.imshow(X[sample_idx].reshape(8, 8), cmap="gray")
    ax.set_title("Original")
    ax.axis("off")

for col, n_comp in enumerate(n_components_list, 1):
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)

    # 再構成画像
    axes[0][col].imshow(X_reconstructed[sample_idx].reshape(8, 8), cmap="gray")
    axes[0][col].set_title(f"k={n_comp}")
    axes[0][col].axis("off")

    # 差分画像
    diff = np.abs(X[sample_idx] - X_reconstructed[sample_idx])
    axes[1][col].imshow(diff.reshape(8, 8), cmap="hot")
    axes[1][col].set_title(f"Error: {np.mean(diff**2):.2f}")
    axes[1][col].axis("off")

axes[1][0].set_visible(False)

plt.suptitle("PCA 再構成: 成分数と画質の関係", fontsize=14)
plt.tight_layout()
plt.savefig("reports/pca_reconstruction.png", dpi=150)
plt.close()

# 再構成誤差の推移をプロット
errors = []
cumvar = []
for k in range(1, 65):
    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X)
    X_rec = pca.inverse_transform(X_pca)
    errors.append(np.mean((X - X_rec) ** 2))
    cumvar.append(sum(pca.explained_variance_ratio_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(1, 65), errors, "b-")
ax1.set_xlabel("主成分数")
ax1.set_ylabel("平均二乗再構成誤差")
ax1.set_title("再構成誤差 vs 主成分数")
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, 65), cumvar, "r-")
ax2.axhline(y=0.95, color="g", linestyle="--", label="95%")
ax2.set_xlabel("主成分数")
ax2.set_ylabel("累積寄与率")
ax2.set_title("累積寄与率 vs 主成分数")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/pca_error_vs_components.png", dpi=150)
plt.close()
```

### 1.3 Incremental PCA（大規模データ向け）

```python
from sklearn.decomposition import IncrementalPCA
import numpy as np

# 大規模データをバッチ処理で PCA（メモリ節約）
n_samples = 100000
n_features = 500
batch_size = 5000

# データを分割して処理
ipca = IncrementalPCA(n_components=50)

# バッチごとに partial_fit
for i in range(0, n_samples, batch_size):
    # 実際にはファイルから読み込むなど
    X_batch = np.random.randn(min(batch_size, n_samples - i), n_features)
    ipca.partial_fit(X_batch)

print(f"IncrementalPCA 学習完了")
print(f"累積寄与率 (50成分): {sum(ipca.explained_variance_ratio_):.4f}")

# 通常の PCA との比較
# 全データをメモリに載せる場合
# from sklearn.decomposition import PCA
# pca = PCA(n_components=50)
# pca.fit(X_all)  # ← X_all が大きすぎるとメモリエラー

# IncrementalPCA なら OK
# データベースやファイルから chunk ごとに読み込み可能
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

### 2.1 t-SNE の数学的詳細

```
Crowding Problem の詳細:

  高次元空間:
    - 点 xi のまわりに「近い」点、「中距離」の点、「遠い」点がある
    - d 次元の球体の体積は r^d に比例
    - d が大きいほど「中距離」の点の数が膨大になる

  2D空間に射影すると:
    - 「近い」点を近くに配置 ← OK
    - 「中距離」の大量の点を適切な距離に配置 ← 困難
    - 2D の面積が足りない → 点が重なる = Crowding

  t 分布の解決策:
    ┌─────────────────────────────────────┐
    │   ガウス分布 vs t 分布（自由度1）   │
    │                                     │
    │   ガウス: ∝ exp(-d²/2)             │
    │   t分布:  ∝ (1 + d²)^(-1)          │
    │                                     │
    │   ←── 中心からの距離 d ──→          │
    │   ▓▓                                │
    │   ▓▓▓▓                  ← ガウス    │
    │   ▓▓▓▓▓▓                            │
    │   ████████████████████  ← t分布     │
    │                                     │
    │   t 分布は裾が重い                  │
    │   → 「やや遠い」点を押し出す力が強い│
    │   → Crowding を緩和する             │
    └─────────────────────────────────────┘

  perplexity パラメータ:
    - 各点の「近傍」の有効な数を制御
    - perplexity ≈ 2^(エントロピー)
    - 小さい perplexity → 局所構造を重視
    - 大きい perplexity → より大域的な構造も考慮
    - 推奨: 5 ≤ perplexity ≤ 50（データサイズの 1/3 以下）
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

### コード例2b: t-SNE の収束過程を可視化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

# 異なる iteration 数での結果を比較
n_iters = [50, 100, 250, 500, 1000, 2000]
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for ax, n_iter in zip(axes.flatten(), n_iters):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=n_iter,
        random_state=42,
        init="pca",
        learning_rate="auto",
        method="barnes_hut",  # O(n log n) 近似
    )
    X_tsne = tsne.fit_transform(X)

    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y,
                         cmap="tab10", s=5, alpha=0.6)
    ax.set_title(f"n_iter={n_iter}, KL={tsne.kl_divergence_:.4f}")
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("t-SNE 収束過程: イテレーション数の影響", fontsize=14)
plt.tight_layout()
plt.savefig("reports/tsne_convergence.png", dpi=150)
plt.close()
```

### 2.2 t-SNE の落とし穴と正しい解釈

```
t-SNE の結果を解釈する際の注意点:

  ✓ 信頼できる情報:
    - 同じクラスタ内の点が近い → 局所的な類似性がある
    - 明確に分離したクラスタ → 高次元でも異なるグループ

  ✗ 信頼できない情報:
    - クラスタ間の距離 → 大域的距離関係は保存されない
    - クラスタのサイズ → 密度が歪む（密集 → 膨張、散在 → 圧縮）
    - クラスタの形状 → 高次元での実際の形状とは無関係

  よくある誤解:

  誤: 「t-SNE で 2 つのクラスタが近い → 似ている」
  正: t-SNE は大域的距離を保存しない。perplexity を変えると
      クラスタの相対位置は大きく変わる。

  誤: 「t-SNE で分離しない → クラスタが存在しない」
  正: t-SNE のパラメータが適切でない可能性がある。
      perplexity を変えて試すべき。

  誤: 「t-SNE の軸に意味がある」
  正: t-SNE の出力の軸はランダムな方向。回転や反転に不変。
      PCA と異なり軸の解釈は不可能。
```

---

## 3. UMAP

### 3.1 UMAP のアルゴリズム概要

```
UMAP (Uniform Manifold Approximation and Projection):

  理論的基盤: 位相幾何学 + ファジー集合論

  Step 1: 高次元での近傍グラフ構築
  ┌─────────────────────────────────────┐
  │ 各点 xi について:                    │
  │ 1. k-近傍を見つける                  │
  │ 2. 局所的な距離をスケーリング        │
  │ 3. ファジー集合として近傍関係を定義  │
  │                                     │
  │ 類似度: w(xi, xj) =                 │
  │   exp(-(d(xi,xj) - ρi) / σi)       │
  │                                     │
  │ ρi: xi の最近傍までの距離            │
  │ σi: 正規化パラメータ                 │
  └─────────────────────────────────────┘
              │
              v
  Step 2: 低次元での最適化
  ┌─────────────────────────────────────┐
  │ クロスエントロピーを最小化:           │
  │                                     │
  │ CE = Σ w_h log(w_h/w_l)             │
  │    + Σ (1-w_h) log((1-w_h)/(1-w_l)) │
  │                                     │
  │ w_h: 高次元の重み                    │
  │ w_l: 低次元の重み                    │
  │ w_l = (1 + a||yi-yj||^(2b))^(-1)   │
  │                                     │
  │ 引力: 近い点を引きつける             │
  │ 斥力: 遠い点を押し離す              │
  └─────────────────────────────────────┘

  t-SNE との違い:
  ┌──────────────────┬──────────────────┐
  │     t-SNE        │     UMAP         │
  ├──────────────────┼──────────────────┤
  │ KL ダイバージェンス│ クロスエントロピー│
  │ 対称化 p_ij      │ ファジー和集合   │
  │ ガウス→t分布     │ ファジー→パラメトリック│
  │ 大域構造 ✗       │ 大域構造 ○       │
  │ transform 不可   │ transform 可能   │
  │ O(n²) or O(n log n)│ O(n^1.14)     │
  └──────────────────┴──────────────────┘
```

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

### 3.2 UMAP の transform（新規データへの適用）

```python
import umap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

# 訓練/テスト分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# UMAP を訓練データで学習
reducer = umap.UMAP(n_components=2, random_state=42)
X_train_2d = reducer.fit_transform(X_train)

# テストデータに transform を適用（t-SNE ではこれが不可能）
X_test_2d = reducer.transform(X_test)

# 可視化
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train,
            cmap="tab10", s=5, alpha=0.5)
ax1.set_title("訓練データ (fit_transform)")
ax1.set_xticks([])
ax1.set_yticks([])

ax2.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test,
            cmap="tab10", s=5, alpha=0.5)
ax2.set_title("テストデータ (transform)")
ax2.set_xticks([])
ax2.set_yticks([])

plt.suptitle("UMAP: 新規データへの適用", fontsize=14)
plt.tight_layout()
plt.savefig("reports/umap_transform.png", dpi=150)
plt.close()

# UMAP + 分類器のパイプライン
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

pipe = make_pipeline(
    umap.UMAP(n_components=10, random_state=42),
    RandomForestClassifier(n_estimators=100, random_state=42)
)

scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
print(f"UMAP(10) + RF: Accuracy = {scores.mean():.4f} ± {scores.std():.4f}")
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

## 4. Kernel PCA（非線形 PCA）

### 4.1 カーネルトリックによる拡張

```
通常のPCA:
  線形射影 → 線形構造しか捉えられない

Kernel PCA:
  1. カーネル関数で暗黙的に高次元空間に写像
  2. その高次元空間で PCA を実行
  3. 結果を低次元に射影

  カーネル関数の例:
  ┌──────────────────────────────────────────┐
  │ RBF (Gaussian):                          │
  │   K(x, x') = exp(-γ||x - x'||²)        │
  │   → 無限次元空間への写像               │
  │                                          │
  │ Polynomial:                              │
  │   K(x, x') = (γ x·x' + r)^d            │
  │   → d次の多項式空間への写像            │
  │                                          │
  │ Sigmoid:                                 │
  │   K(x, x') = tanh(γ x·x' + r)          │
  │   → ニューラルネットの活性化に類似     │
  └──────────────────────────────────────────┘

  いつ使うか:
  - データに非線形構造がある（例: 同心円、螺旋）
  - PCA で分離できないクラスが Kernel PCA で分離できる場合
  - 計算量: O(n²) なので大規模データには不向き
```

### コード例6: Kernel PCA の実装と比較

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_moons, make_circles

# 非線形データ: 半月型と同心円
datasets = {
    "半月型": make_moons(n_samples=500, noise=0.1, random_state=42),
    "同心円": make_circles(n_samples=500, noise=0.05, factor=0.3, random_state=42),
}

fig, axes = plt.subplots(len(datasets), 4, figsize=(20, 10))

for row, (name, (X, y)) in enumerate(datasets.items()):
    # 元データ
    axes[row][0].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=10)
    axes[row][0].set_title(f"{name}: 元データ")

    # 通常の PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    axes[row][1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", s=10)
    axes[row][1].set_title("PCA")

    # Kernel PCA (RBF)
    kpca_rbf = KernelPCA(n_components=2, kernel="rbf", gamma=10)
    X_kpca_rbf = kpca_rbf.fit_transform(X)
    axes[row][2].scatter(X_kpca_rbf[:, 0], X_kpca_rbf[:, 1], c=y,
                          cmap="coolwarm", s=10)
    axes[row][2].set_title("Kernel PCA (RBF)")

    # Kernel PCA (Polynomial)
    kpca_poly = KernelPCA(n_components=2, kernel="poly", degree=3, gamma=1)
    X_kpca_poly = kpca_poly.fit_transform(X)
    axes[row][3].scatter(X_kpca_poly[:, 0], X_kpca_poly[:, 1], c=y,
                          cmap="coolwarm", s=10)
    axes[row][3].set_title("Kernel PCA (Poly)")

plt.tight_layout()
plt.savefig("reports/kernel_pca_comparison.png", dpi=150)
plt.close()
```

### コード例6b: Kernel PCA の gamma パラメータ最適化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=1000, noise=0.1, factor=0.3, random_state=42)

# gamma のグリッドサーチ
gammas = [0.01, 0.1, 1, 5, 10, 50, 100]

fig, axes = plt.subplots(1, len(gammas), figsize=(28, 4))

for ax, gamma in zip(axes, gammas):
    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=gamma)
    X_kpca = kpca.fit_transform(X)
    ax.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap="coolwarm", s=5)
    ax.set_title(f"γ={gamma}")
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("Kernel PCA: gamma の影響 (RBF)", fontsize=14)
plt.tight_layout()
plt.savefig("reports/kpca_gamma.png", dpi=150)
plt.close()

# パイプラインでの最適 gamma 探索
pipe = make_pipeline(
    KernelPCA(kernel="rbf"),
    LogisticRegression()
)

param_grid = {
    "kernelpca__n_components": [2, 5, 10],
    "kernelpca__gamma": [0.01, 0.1, 1, 10],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")
grid.fit(X, y)

print(f"最良パラメータ: {grid.best_params_}")
print(f"最良スコア: {grid.best_score_:.4f}")
```

---

## 5. LDA（線形判別分析）

### 5.1 LDA の原理

```
LDA (Linear Discriminant Analysis):
  教師あり次元削減 — クラスラベルを使って最適な射影方向を見つける

  目標: クラス間分散を最大化 & クラス内分散を最小化

  ┌──────────────────────────────────────────┐
  │                                          │
  │  元の空間:        LDA 射影後:            │
  │                                          │
  │   ○ ○         ← クラスA                 │
  │  ○ ○ ○        ← クラスA    LDA          │
  │     △ △       ← クラスB    ─→  ○○○○ △△△ │
  │    △ △ △      ← クラスB        (分離!)  │
  │                                          │
  │  PCA: 分散が最大の方向                    │
  │  LDA: クラスが最も分離する方向            │
  │                                          │
  └──────────────────────────────────────────┘

  数学的定式化:
    クラス内分散行列: S_W = Σ_c Σ_{x∈c} (x - μ_c)(x - μ_c)^T
    クラス間分散行列: S_B = Σ_c n_c (μ_c - μ)(μ_c - μ)^T

    最大化: J(w) = (w^T S_B w) / (w^T S_W w)
    解: S_W^(-1) S_B の固有ベクトル

    最大次元数: min(d, C-1)  （C: クラス数）
    → 10クラス分類なら最大9次元まで
```

### コード例7: LDA と PCA の比較

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Wine データセット（13次元、3クラス）
wine = load_wine()
X, y = wine.data, wine.target

X_scaled = StandardScaler().fit_transform(X)

# PCA vs LDA の 2D 射影を比較
pca = PCA(n_components=2)
lda = LinearDiscriminantAnalysis(n_components=2)

X_pca = pca.fit_transform(X_scaled)
X_lda = lda.fit_transform(X_scaled, y)  # LDA はラベルが必要

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for cls in np.unique(y):
    mask = y == cls
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Class {cls}", s=30, alpha=0.7)
    ax2.scatter(X_lda[mask, 0], X_lda[mask, 1], label=f"Class {cls}", s=30, alpha=0.7)

ax1.set_title(f"PCA 2D (寄与率: {sum(pca.explained_variance_ratio_):.1%})")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_title("LDA 2D (教師あり)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle("PCA vs LDA: Wine データセット", fontsize=14)
plt.tight_layout()
plt.savefig("reports/pca_vs_lda.png", dpi=150)
plt.close()

# 分類精度の比較
for name, reducer in [("PCA(2)", PCA(n_components=2)),
                       ("PCA(5)", PCA(n_components=5)),
                       ("LDA(2)", LinearDiscriminantAnalysis(n_components=2))]:
    pipe = make_pipeline(StandardScaler(), reducer, LogisticRegression())
    scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    print(f"{name:10s}  Accuracy = {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 6. Autoencoder による次元削減

### 6.1 Autoencoder の構造

```
Autoencoder:
  ニューラルネットワークによる非線形次元削減

  入力 → エンコーダ → ボトルネック → デコーダ → 復元
  (d)     (d→h1→...→k)    (k)    (k→...→h1→d)  (d)

  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │  入力層    隠れ層    ボトルネック   隠れ層  出力層│
  │  (784)    (256)      (32)         (256)  (784) │
  │                                                 │
  │   ○ ○      ○         ○             ○      ○ ○  │
  │   ○ ○      ○         ○             ○      ○ ○  │
  │   ○ ○      ○         ○             ○      ○ ○  │
  │   ○ ○      ○                       ○      ○ ○  │
  │   ○ ○                                    ○ ○  │
  │                                                 │
  │  エンコーダ ──→  潜在表現  ──→  デコーダ        │
  │                  (z ∈ ℝ^k)                     │
  │                                                 │
  │  損失: L = ||x - x̂||²  （再構成誤差）          │
  │                                                 │
  │  PCA との関係:                                  │
  │  - 線形活性化のAE = PCA と等価                  │
  │  - 非線形活性化 → より強力な次元削減           │
  └─────────────────────────────────────────────────┘
```

### コード例8: PyTorch による Autoencoder 実装

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# データ準備
digits = load_digits()
X = StandardScaler().fit_transform(digits.data).astype(np.float32)
y = digits.target

X_tensor = torch.tensor(X)
dataset = TensorDataset(X_tensor, X_tensor)  # 入力 = 出力
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Autoencoder モデル
class Autoencoder(nn.Module):
    def __init__(self, input_dim=64, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

# 学習
model = Autoencoder(input_dim=64, latent_dim=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

losses = []
for epoch in range(100):
    epoch_loss = 0
    for x_batch, _ in loader:
        x_hat, z = model(x_batch)
        loss = criterion(x_hat, x_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    losses.append(epoch_loss / len(loader))
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: Loss = {losses[-1]:.6f}")

# 潜在空間の可視化
model.eval()
with torch.no_grad():
    _, Z = model(X_tensor)
    Z = Z.numpy()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Autoencoder 2D
scatter = ax1.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=5, alpha=0.6)
ax1.set_title("Autoencoder 2D")
ax1.set_xticks([])
ax1.set_yticks([])
plt.colorbar(scatter, ax=ax1)

# PCA 2D（比較用）
from sklearn.decomposition import PCA
X_pca = PCA(n_components=2).fit_transform(X)
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", s=5, alpha=0.6)
ax2.set_title("PCA 2D")
ax2.set_xticks([])
ax2.set_yticks([])

# 学習曲線
ax3.plot(losses)
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Reconstruction Loss")
ax3.set_title("学習曲線")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/autoencoder_vs_pca.png", dpi=150)
plt.close()
```

---

## 7. 応用: 次元削減の実践的ユースケース

### 7.1 テキストデータの可視化

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import umap
import matplotlib.pyplot as plt

# テキストデータ（サンプル）
texts = [
    "machine learning is a subset of artificial intelligence",
    "deep learning uses neural networks with many layers",
    "natural language processing deals with text data",
    "computer vision focuses on image recognition",
    "reinforcement learning learns from rewards",
    "soccer is a popular sport worldwide",
    "basketball requires good physical fitness",
    "tennis is played on different surfaces",
    "swimming is both a sport and exercise",
    "baseball has nine players on a team",
    "python is a programming language",
    "javascript runs in web browsers",
    "rust focuses on memory safety",
    "go is designed for concurrency",
    "java is used in enterprise applications",
]
categories = ["AI"] * 5 + ["Sports"] * 5 + ["Programming"] * 5

# TF-IDF ベクトル化
vectorizer = TfidfVectorizer(stop_words="english")
X_tfidf = vectorizer.fit_transform(texts)
print(f"TF-IDF 行列の形状: {X_tfidf.shape}")

# 高次元スパース → 低次元密ベクトルに変換
# Step 1: TruncatedSVD で 50 次元に削減（スパース行列対応）
svd = TruncatedSVD(n_components=min(10, X_tfidf.shape[1] - 1))
X_svd = svd.fit_transform(X_tfidf)

# Step 2: UMAP で 2D に可視化
reducer = umap.UMAP(n_components=2, n_neighbors=5, random_state=42)
X_2d = reducer.fit_transform(X_svd)

# 可視化
color_map = {"AI": "red", "Sports": "blue", "Programming": "green"}
colors = [color_map[c] for c in categories]

plt.figure(figsize=(10, 8))
for cat in color_map:
    mask = [c == cat for c in categories]
    plt.scatter(X_2d[np.array(mask), 0], X_2d[np.array(mask), 1],
                c=color_map[cat], label=cat, s=100)

for i, text in enumerate(texts):
    short_text = text[:25] + "..." if len(text) > 25 else text
    plt.annotate(short_text, (X_2d[i, 0], X_2d[i, 1]),
                 fontsize=7, alpha=0.7)

plt.title("テキストデータの次元削減可視化")
plt.legend()
plt.tight_layout()
plt.savefig("reports/text_dim_reduction.png", dpi=150)
plt.close()
```

### 7.2 画像特徴量の可視化（CNN + UMAP）

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import numpy as np
import umap
import matplotlib.pyplot as plt

# 事前学習済み CNN で特徴量を抽出
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # 最終FC層を除去
model.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# CIFAR-10 のサブセット
dataset = CIFAR10(root="./data", train=False, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=128, shuffle=False)

# 特徴量抽出
features = []
labels = []
with torch.no_grad():
    for images, targets in loader:
        feat = model(images).squeeze()
        features.append(feat.numpy())
        labels.append(targets.numpy())
        if len(features) * 128 >= 2000:
            break

features = np.vstack(features)[:2000]
labels = np.concatenate(labels)[:2000]
print(f"特徴量行列: {features.shape}")  # (2000, 512)

# UMAP で 2D に射影
reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
X_2d = reducer.fit_transform(features)

# 可視化
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(12, 10))
for i, name in enumerate(class_names):
    mask = labels == i
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=name, s=5, alpha=0.6)

plt.legend(markerscale=5, loc="best")
plt.title("CIFAR-10: CNN特徴量の UMAP 可視化")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig("reports/cnn_features_umap.png", dpi=150)
plt.close()
```

### 7.3 異常検知への応用

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 正常データ + 異常データの生成
np.random.seed(42)
n_normal = 500
n_anomaly = 20
n_features = 20

# 正常データ: 低ランク構造（実際には5次元の構造）
true_dim = 5
W = np.random.randn(true_dim, n_features)
Z_normal = np.random.randn(n_normal, true_dim)
X_normal = Z_normal @ W + np.random.randn(n_normal, n_features) * 0.3

# 異常データ: 正常データの構造から外れたパターン
X_anomaly = np.random.randn(n_anomaly, n_features) * 3

X = np.vstack([X_normal, X_anomaly])
labels = np.array([0] * n_normal + [1] * n_anomaly)  # 0=正常, 1=異常

# PCA で再構成誤差を計算
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=5)  # 正常データの真の次元数
X_pca = pca.fit_transform(X_scaled)
X_reconstructed = pca.inverse_transform(X_pca)

# 再構成誤差 = 異常スコア
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

# 閾値: 正常データの 99 パーセンタイル
threshold = np.percentile(reconstruction_error[:n_normal], 99)
predictions = (reconstruction_error > threshold).astype(int)

# 可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 再構成誤差の分布
ax1.hist(reconstruction_error[labels == 0], bins=30, alpha=0.5, label="正常", color="blue")
ax1.hist(reconstruction_error[labels == 1], bins=10, alpha=0.5, label="異常", color="red")
ax1.axvline(threshold, color="green", linestyle="--", label=f"閾値={threshold:.2f}")
ax1.set_xlabel("再構成誤差")
ax1.set_ylabel("頻度")
ax1.set_title("PCA 再構成誤差による異常検知")
ax1.legend()

# PCA 2D 可視化
X_2d = PCA(n_components=2).fit_transform(X_scaled)
ax2.scatter(X_2d[labels == 0, 0], X_2d[labels == 0, 1],
            c="blue", s=10, alpha=0.5, label="正常")
ax2.scatter(X_2d[labels == 1, 0], X_2d[labels == 1, 1],
            c="red", s=50, alpha=0.8, label="異常", marker="x")
ax2.set_title("PCA 2D 可視化")
ax2.legend()

plt.tight_layout()
plt.savefig("reports/pca_anomaly_detection.png", dpi=150)
plt.close()

# 検出精度
from sklearn.metrics import precision_score, recall_score, f1_score
print(f"Precision: {precision_score(labels, predictions):.4f}")
print(f"Recall: {recall_score(labels, predictions):.4f}")
print(f"F1: {f1_score(labels, predictions):.4f}")
```

---

## 8. トラブルシューティング

### 8.1 よくある問題と解決策

```
問題1: PCA の寄与率が低い（2成分で 30% 未満）

  原因:
    - データが高次元で分散が多くの方向に均等に分布
    - 非線形構造が支配的

  解決策:
    1. まずスケーリングが適切か確認
    2. 非線形手法（t-SNE, UMAP）を可視化に使用
    3. Kernel PCA で非線形構造を捕捉
    4. PCA の次元数を増やして下流タスクの精度を確認

問題2: t-SNE でクラスタが分離しない

  原因:
    - perplexity が不適切
    - データに明確なクラスタ構造がない
    - 前処理不足

  解決策:
    1. perplexity を 5, 15, 30, 50, 100 で試す
    2. learning_rate = "auto" を使用
    3. n_iter を 1000 以上に増やす
    4. 事前に PCA で 50 次元程度に削減してから t-SNE を適用

問題3: UMAP が遅い

  原因:
    - データ量が大きい
    - n_neighbors が大きすぎる

  解決策:
    1. 事前に PCA で次元削減
    2. n_neighbors を小さくする（5-15 程度）
    3. metric="cosine" の方が高速な場合がある
    4. サンプリングしてから UMAP を適用

問題4: Kernel PCA で結果が不安定

  原因:
    - gamma パラメータが不適切
    - カーネル行列の条件数が悪い

  解決策:
    1. gamma のグリッドサーチを実施
    2. RBF カーネルの場合、1/(n_features * X.var()) を初期値にする
    3. fit_inverse_transform=True で逆変換を有効化
    4. データのスケーリングを確認
```

### 8.2 パフォーマンスチューニング

```python
import numpy as np
import time
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
import umap

# ベンチマーク: データサイズ別の実行時間
data_sizes = [1000, 5000, 10000, 50000]
results = []

for n in data_sizes:
    X = np.random.randn(n, 100)

    # PCA
    start = time.time()
    PCA(n_components=2).fit_transform(X)
    pca_time = time.time() - start

    # t-SNE (5000以下のみ)
    if n <= 5000:
        start = time.time()
        TSNE(n_components=2, random_state=42).fit_transform(X)
        tsne_time = time.time() - start
    else:
        tsne_time = float("nan")

    # UMAP
    start = time.time()
    umap.UMAP(n_components=2, random_state=42).fit_transform(X)
    umap_time = time.time() - start

    results.append({
        "n": n,
        "PCA": f"{pca_time:.2f}s",
        "t-SNE": f"{tsne_time:.2f}s" if not np.isnan(tsne_time) else "N/A",
        "UMAP": f"{umap_time:.2f}s",
    })
    print(f"n={n:>6d}  PCA={pca_time:.2f}s  t-SNE={tsne_time:.2f}s  "
          f"UMAP={umap_time:.2f}s")
```

---

## 比較表

### 次元削減手法の特性比較

| 手法 | 種類 | 保持する構造 | 計算量 | 新データ適用 | 主な用途 |
|---|---|---|---|---|---|
| PCA | 線形 | 大域的分散 | O(min(n,d)^2 x max(n,d)) | 可（transform） | 前処理、ノイズ除去 |
| Kernel PCA | 非線形 | 非線形大域構造 | O(n^3) | 可（近似） | 非線形構造の発見 |
| t-SNE | 非線形 | 局所構造 | O(n^2) → O(n log n) | 不可 | 2D/3D可視化 |
| UMAP | 非線形 | 局所+大域 | O(n^1.14) | 可（transform） | 可視化、前処理 |
| LDA | 線形（教師あり） | クラス間分離 | O(nd^2) | 可 | 分類前処理 |
| Autoencoder | 非線形 | 学習された表現 | NN依存 | 可 | 特徴抽出、生成 |
| Incremental PCA | 線形 | 大域的分散 | O(batch x d^2) | 可 | 大規模データ |
| TruncatedSVD | 線形 | 分散 | O(n x d x k) | 可 | スパースデータ |

### パラメータ選択ガイド

| 手法 | 主要パラメータ | 推奨範囲 | 効果 |
|---|---|---|---|
| PCA | n_components | 0.95（寄与率）or 次元数 | 保持する情報量 |
| t-SNE | perplexity | 5〜50（デフォルト30） | 局所/大域バランス |
| t-SNE | learning_rate | "auto" or n/12 | 収束の安定性 |
| t-SNE | n_iter | 1000〜5000 | 収束品質 |
| UMAP | n_neighbors | 5〜200（デフォルト15） | 局所/大域バランス |
| UMAP | min_dist | 0.0〜0.99（デフォルト0.1） | クラスタの密集度 |
| Kernel PCA | gamma | 1/(n_features * X.var()) | RBF の幅 |
| LDA | n_components | 1〜(C-1) | 射影次元数 |
| Autoencoder | latent_dim | タスク依存 | ボトルネック幅 |

### ユースケース別推奨手法

| ユースケース | 第一候補 | 第二候補 | 理由 |
|---|---|---|---|
| EDA（探索的データ分析） | UMAP | t-SNE | 高速かつ大域構造も保持 |
| 前処理（次元削減） | PCA | UMAP | 線形で安定、transform 可能 |
| 可視化（小規模、< 5000） | t-SNE | UMAP | 局所構造の保存が最良 |
| 可視化（大規模、> 10000） | UMAP | PCA → t-SNE | UMAP が圧倒的に高速 |
| ノイズ除去 | PCA | Autoencoder | 低ランク近似でノイズ除去 |
| 異常検知 | PCA | Autoencoder | 再構成誤差 = 異常スコア |
| テキストデータ | TruncatedSVD | UMAP | スパース行列に対応 |
| 分類前処理（教師あり） | LDA | PCA | クラス分離を最大化 |
| 非線形構造の発見 | Kernel PCA | UMAP | 明示的な射影が可能 |
| 生成モデルの潜在空間 | Autoencoder | VAE | デコーダで再構成可能 |

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

### アンチパターン3: スケーリングなしで PCA を適用

```python
# BAD: 単位の異なる特徴量をそのまま PCA
# 面積(m²: 10-200) と 部屋数(1-5) → 面積の分散が圧倒的に大きい
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_raw)
# → PC1 ≈ 面積の方向（単位の影響）

# GOOD: StandardScaler で正規化してから PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(), PCA(n_components=2))
X_pca = pipe.fit_transform(X_raw)
# → 全特徴量が等しくスケーリングされた状態で PCA
```

### アンチパターン4: 次元削減後のデータでハイパーパラメータ選択

```python
# BAD: 全データで次元削減 → 分割 → モデル評価
# → テストデータの情報が次元削減に漏洩（データリーク）
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)  # 全データで fit（リーク!）
X_train, X_test = train_test_split(X_reduced)

# GOOD: Pipeline 内で次元削減 → CV 内で fit/transform が分離される
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

pipe = make_pipeline(PCA(n_components=10), LogisticRegression())
scores = cross_val_score(pipe, X, y, cv=5)
# → 各 fold で PCA の fit は訓練データのみに適用される
```

---

## 演習

### 演習1（基礎）: PCA の実装と解釈

```
課題:
  1. Iris データセット（4次元）に PCA を適用し、2D で可視化せよ
  2. 各主成分の寄与率と累積寄与率を計算せよ
  3. 第1主成分の方向ベクトル（loadings）から、
     どの特徴量が PC1 に最も寄与しているかを分析せよ

ヒント:
  - pca.components_ に主成分方向ベクトルが格納されている
  - loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
```

### 演習2（応用）: t-SNE と UMAP の比較実験

```
課題:
  1. MNIST（784次元）のサブセット（5000サンプル）を使用
  2. PCA(50) → t-SNE(2) のパイプラインを実装
  3. PCA(50) → UMAP(2) のパイプラインを実装
  4. 実行時間と可視化品質を比較せよ
  5. UMAP の n_neighbors を変えて結果がどう変わるかを記録せよ

評価基準:
  - 同じ数字の点が近くに配置されているか（局所構造）
  - 異なる数字のクラスタが分離しているか
  - 実行時間の差
```

### 演習3（発展）: Autoencoder vs PCA の比較

```
課題:
  1. 手書き数字データに対して以下を実装:
     a. PCA (n_components=k) での再構成
     b. Autoencoder (latent_dim=k) での再構成
  2. k = 2, 5, 10, 20 で再構成誤差を比較せよ
  3. k=2 の潜在空間を 2D で可視化し、クラス分離性を比較せよ
  4. 非線形 Autoencoder が PCA を上回る条件を考察せよ

発展課題:
  - Variational Autoencoder (VAE) を実装し、
    潜在空間の連続性を確認せよ
```

---

## FAQ

### Q1: PCA、t-SNE、UMAPのどれを使うべき？

**A:** 目的による。(1) 前処理・ノイズ除去 → PCA（線形、transform可能）、(2) 可視化（小〜中規模） → t-SNE（局所構造の保存が最良）、(3) 可視化＋前処理（大規模） → UMAP（高速、transform可能、大域構造も保持）。まずPCAで概観し、必要に応じてt-SNE/UMAPを試す。

### Q2: PCAの成分数はどう決める？

**A:** (1) 累積寄与率が95%以上になる成分数、(2) スクリープロット（固有値の減衰曲線）の「肘」、(3) 下流タスク（分類等）の交差検証スコアで選択。実用的には `PCA(n_components=0.95)` で自動決定できる。

### Q3: t-SNEは再現性がないのか？

**A:** random_stateを固定すれば同じデータで同じ結果が得られる。ただし初期値への依存が強いため、異なるrandom_stateで複数回実行し、結果が安定しているか確認するのが良い。scikit-learn 1.2以降では `init="pca"` がデフォルトとなり再現性が向上している。

### Q4: 大規模データ（100万行以上）に次元削減を適用するには？

**A:** PCA なら IncrementalPCA でバッチ処理が可能。UMAP は内部で近似近傍探索を使うため 100 万行でも実用的な時間で動作する。t-SNE は O(n^2) のため大規模データには不向き。代替として openTSNE ライブラリの Barnes-Hut 近似や FIt-SNE（FFT 加速）がある。

### Q5: 次元削減後のデータで距離計算は有効か？

**A:** PCA 後のユークリッド距離は、元の空間での距離の近似（情報量の損失分だけ劣化）。UMAP 後の距離は局所的には有効だが大域的には信頼できない。t-SNE 後の距離は定量分析には使えない。距離ベースの分析には PCA を使うのが安全。

### Q6: PCA で負の主成分荷重（loading）はどう解釈する？

**A:** 負の荷重は、その特徴量が増加すると主成分の値が減少することを意味する。例えば PC1 の荷重が [0.8, -0.6] なら、特徴量1が増加すると PC1 が増加し、特徴量2が増加すると PC1 が減少する。これは「対比」を表しており、2つの特徴量のトレードオフ関係を反映している。

---

## ベストプラクティス

### 次元削減ワークフロー

```
Step 1: データの前処理
  ├── 欠損値の処理
  ├── 外れ値の処理
  └── StandardScaler でスケーリング（PCA/Kernel PCA では必須）

Step 2: 概観の把握
  ├── PCA で全成分の寄与率を確認
  ├── 累積寄与率プロットで情報量の分布を理解
  └── 2D PCA 射影で大まかな構造を確認

Step 3: 目的に応じた手法選択
  ├── 前処理 → PCA（n_components=0.95）
  ├── 可視化 → UMAP or t-SNE
  ├── 教師あり → LDA
  └── 非線形構造 → Kernel PCA or Autoencoder

Step 4: パラメータチューニング
  ├── PCA: 寄与率 or CV で n_components を決定
  ├── t-SNE: perplexity を複数試す
  ├── UMAP: n_neighbors と min_dist を調整
  └── Kernel PCA: gamma のグリッドサーチ

Step 5: 評価と検証
  ├── 可視化の場合: 複数の手法・パラメータで比較
  ├── 前処理の場合: 下流タスクの精度で評価
  ├── 再構成誤差の確認
  └── データリークがないことを確認（Pipeline 使用）
```

### チェックリスト

```
□ スケーリングを適用したか（特に PCA、Kernel PCA）
□ PCA の寄与率を確認したか
□ t-SNE の結果を大域的な距離として解釈していないか
□ 複数の perplexity / n_neighbors で結果を比較したか
□ Pipeline を使ってデータリークを防いでいるか
□ 新規データへの適用が必要な場合、transform 可能な手法を選んだか
□ 計算時間とデータサイズのバランスを考慮したか
□ 次元削減の目的（可視化 vs 前処理 vs 異常検知）を明確にしたか
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| PCA | 線形次元削減。分散最大化。前処理・ノイズ除去に最適 |
| t-SNE | 非線形。局所構造保存。2D/3D可視化専用。大域距離は不正確 |
| UMAP | 非線形。局所+大域保持。高速。可視化+前処理に両用可 |
| Kernel PCA | 非線形 PCA。カーネルで暗黙的に高次元空間で PCA |
| LDA | 教師あり次元削減。クラス分離を最大化。最大 C-1 次元 |
| Autoencoder | NN による非線形次元削減。再構成ベース。柔軟性が最も高い |
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
4. **Jolliffe, I. T.** "Principal Component Analysis" 2nd Edition, Springer, 2002
5. **Wattenberg, M., Viegas, F., Johnson, I.** "How to Use t-SNE Effectively" Distill, 2016 — https://distill.pub/2016/misread-tsne/
6. **Kingma, D. P., Welling, M.** "Auto-Encoding Variational Bayes" ICLR, 2014 — https://arxiv.org/abs/1312.6114

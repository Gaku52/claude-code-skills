# 回帰 — 線形/多項式/Ridge/Lasso

> 連続値を予測する回帰手法の理論と実装を、正則化まで含めて体系的に理解する

## この章で学ぶこと

1. **線形回帰の数理** — 最小二乗法、正規方程式、勾配降下法の原理
2. **正則化回帰** — Ridge（L2）、Lasso（L1）、ElasticNetによる過学習抑制
3. **多項式回帰と非線形拡張** — 特徴量の非線形変換と適切な次数選択
4. **高度な回帰手法** — ロバスト回帰、ベイズ回帰、量子回帰
5. **実務的な回帰分析** — 残差分析、特徴量エンジニアリング、パイプライン構築

---

## 1. 線形回帰の基礎

### 1.1 最小二乗法の幾何学的解釈

```
y (目的変数)
│
│         *
│       /  *
│     /   *     ŷ = β₀ + β₁x
│   / *
│ / *            残差 (y - ŷ) の二乗和を最小化
│*
└──────────────── x (説明変数)

正規方程式: β = (X^T X)^(-1) X^T y

    残差の視覚化:
    │    *
    │    |   ← 残差 e = y - ŷ
    │    ŷ
    │   /
    │  /
```

### 1.2 線形回帰の仮定（ガウス・マルコフの定理）

```
最小二乗推定量（OLS）が最良線形不偏推定量（BLUE）であるための条件:

1. 線形性:      y = Xβ + ε  （パラメータに対して線形）
2. 不偏性:      E[ε] = 0    （誤差の期待値がゼロ）
3. 等分散性:    Var(ε) = σ²I （誤差の分散が一定）
4. 無相関:      Cov(εᵢ, εⱼ) = 0  （i≠j, 誤差間に相関なし）
5. 外生性:      Cov(X, ε) = 0    （説明変数と誤差が無相関）

追加の仮定（推論・検定用）:
6. 正規性:      ε ~ N(0, σ²I)  （誤差が正規分布に従う）
7. 非多重共線性: rank(X) = p    （特徴量間に完全な線形従属がない）

仮定が崩れた場合の影響:
  ・ 等分散性違反 → 不均一分散 → 重み付き最小二乗法(WLS)を使用
  ・ 無相関違反 → 自己相関 → 一般化最小二乗法(GLS)を使用
  ・ 多重共線性 → 係数の不安定化 → Ridge回帰/VIF分析
  ・ 正規性違反 → 検定結果が不正確 → ブートストラップ法
```

### コード例1: 線形回帰の実装（ゼロから）

```python
import numpy as np

class LinearRegressionFromScratch:
    """最小二乗法による線形回帰のフル実装"""

    def __init__(self, method: str = "normal_equation"):
        self.method = method
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            lr: float = 0.01, n_iter: int = 1000) -> "LinearRegressionFromScratch":
        n, m = X.shape

        if self.method == "normal_equation":
            # 正規方程式: β = (X^T X)^(-1) X^T y
            X_b = np.c_[np.ones((n, 1)), X]
            theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
            self.bias = theta[0]
            self.weights = theta[1:]

        elif self.method == "gradient_descent":
            # 勾配降下法
            self.weights = np.zeros(m)
            self.bias = 0.0
            self.history = []

            for i in range(n_iter):
                y_pred = X @ self.weights + self.bias
                error = y_pred - y

                # 勾配の計算
                dw = (2 / n) * (X.T @ error)
                db = (2 / n) * np.sum(error)

                # パラメータ更新
                self.weights -= lr * dw
                self.bias -= lr * db

                # MSE記録
                mse = np.mean(error ** 2)
                self.history.append(mse)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

    def r2_score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

# 使用例
np.random.seed(42)
X = np.random.randn(100, 3)
y = 3 * X[:, 0] + 2 * X[:, 1] - 1 * X[:, 2] + 5 + np.random.randn(100) * 0.5

model = LinearRegressionFromScratch(method="normal_equation")
model.fit(X, y)
print(f"重み: {model.weights.round(3)}")  # ≈ [3, 2, -1]
print(f"バイアス: {model.bias:.3f}")       # ≈ 5
print(f"R²: {model.r2_score(X, y):.4f}")
```

### コード例1b: 勾配降下法のバリエーション

```python
import numpy as np
import matplotlib.pyplot as plt

class GradientDescentVariants:
    """勾配降下法の各種バリエーション"""

    def __init__(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def batch_gd(self, X, y, lr=0.01, n_iter=1000):
        """バッチ勾配降下法（全データを使用）"""
        n = X.shape[0]
        history = []

        for i in range(n_iter):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            self.weights -= lr * (2 / n) * (X.T @ error)
            self.bias -= lr * (2 / n) * np.sum(error)

            history.append(np.mean(error ** 2))

        return history

    def stochastic_gd(self, X, y, lr=0.01, n_iter=1000):
        """確率的勾配降下法（1サンプルずつ更新）"""
        n = X.shape[0]
        history = []

        for i in range(n_iter):
            # ランダムにシャッフル
            indices = np.random.permutation(n)

            for idx in indices:
                xi = X[idx:idx+1]
                yi = y[idx:idx+1]
                y_pred = xi @ self.weights + self.bias
                error = y_pred - yi

                self.weights -= lr * 2 * (xi.T @ error).ravel()
                self.bias -= lr * 2 * error[0]

            # エポック終了時のMSE
            y_pred_all = X @ self.weights + self.bias
            history.append(np.mean((y_pred_all - y) ** 2))

        return history

    def mini_batch_gd(self, X, y, lr=0.01, n_iter=1000, batch_size=32):
        """ミニバッチ勾配降下法"""
        n = X.shape[0]
        history = []

        for i in range(n_iter):
            indices = np.random.permutation(n)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                batch_n = X_batch.shape[0]

                y_pred = X_batch @ self.weights + self.bias
                error = y_pred - y_batch

                self.weights -= lr * (2 / batch_n) * (X_batch.T @ error)
                self.bias -= lr * (2 / batch_n) * np.sum(error)

            # エポック終了時のMSE
            y_pred_all = X @ self.weights + self.bias
            history.append(np.mean((y_pred_all - y) ** 2))

        return history


# 比較実験
np.random.seed(42)
X = np.random.randn(500, 5)
true_w = np.array([3.0, -2.0, 1.5, 0.0, -1.0])
y = X @ true_w + 2.0 + np.random.randn(500) * 0.5

fig, ax = plt.subplots(figsize=(12, 6))

for method_name, method_fn in [
    ("Batch GD", lambda m: m.batch_gd(X, y, lr=0.01, n_iter=100)),
    ("SGD", lambda m: m.stochastic_gd(X, y, lr=0.001, n_iter=100)),
    ("Mini-Batch GD (32)", lambda m: m.mini_batch_gd(X, y, lr=0.01, n_iter=100)),
]:
    model = GradientDescentVariants(5)
    history = method_fn(model)
    ax.plot(history, label=method_name, linewidth=2)

ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title("勾配降下法のバリエーション比較")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale("log")
plt.tight_layout()
plt.savefig("reports/gd_variants.png", dpi=150)
plt.close()
```

---

## 2. 正則化回帰

### 2.1 正則化の効果

```
正則化なし (OLS)        Ridge (L2)              Lasso (L1)
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 損失 = MSE   │    │ 損失 = MSE   │    │ 損失 = MSE   │
│              │    │  + λΣβ²      │    │  + λΣ|β|     │
│              │    │              │    │              │
│ 重みに制約   │    │ 重みを縮小   │    │ 重みをゼロに │
│ なし         │    │ (全体的に)   │    │ (スパース)   │
│              │    │              │    │              │
│ 過学習リスク │    │ 多重共線性   │    │ 特徴量選択   │
│ 高い         │    │ に強い       │    │ 効果あり     │
└──────────────┘    └──────────────┘    └──────────────┘

ElasticNet = L1 + L2 の組み合わせ
損失 = MSE + α×ρΣ|β| + α×(1-ρ)Σβ²
          ρ: L1比率 (0〜1)
```

### 2.2 正則化の数学的理解

```
■ 制約付き最適化としての解釈

  Ridge:  min ||y - Xβ||²   subject to  ||β||₂ ≤ t
  Lasso:  min ||y - Xβ||²   subject to  ||β||₁ ≤ t

  幾何学的解釈:
    Ridge → L2制約 = 円（楕円）→ 解が軸上に来にくい → スパースにならない
    Lasso → L1制約 = ひし形 → 角（軸上）に解が来やすい → スパースになる

  ┌──────────────────────────────────┐
  │         Ridge (L2)               │
  │    等高線（MSE）                  │
  │      ____                        │
  │     /    \    ○ ← L2制約（円）    │
  │    |  *   |  │                    │
  │     \____/   │                    │
  │              │                    │
  │         Lasso (L1)               │
  │      ____                        │
  │     /    \  ◇ ← L1制約（ひし形）  │
  │    |  *   | / \                   │
  │     \____/ │                     │
  │         角に解 → βⱼ = 0          │
  └──────────────────────────────────┘

■ ベイズ的解釈
  Ridge → 事前分布: β ~ N(0, 1/λ)  （正規分布 → MAP推定）
  Lasso → 事前分布: β ~ Laplace(0, 1/λ)  （ラプラス分布 → MAP推定）
```

### コード例2: 正則化回帰の比較実験

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# 高次元で多重共線性のあるデータ
np.random.seed(42)
n_samples, n_features = 100, 50
X = np.random.randn(n_samples, n_features)
# 真のモデル: 最初の5変数だけが重要
true_coef = np.zeros(n_features)
true_coef[:5] = [3, -2, 1.5, -1, 0.5]
y = X @ true_coef + np.random.randn(n_samples) * 0.5

models = {
    "線形回帰 (OLS)": LinearRegression(),
    "Ridge (α=1.0)": Ridge(alpha=1.0),
    "Ridge (α=10.0)": Ridge(alpha=10.0),
    "Lasso (α=0.1)": Lasso(alpha=0.1),
    "Lasso (α=1.0)": Lasso(alpha=1.0),
    "ElasticNet (α=0.1)": ElasticNet(alpha=0.1, l1_ratio=0.5),
}

print(f"{'モデル':25s} {'CV R²':>10s} {'非ゼロ係数':>10s}")
print("-" * 50)
for name, model in models.items():
    pipe = make_pipeline(StandardScaler(), model)
    scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")
    pipe.fit(X, y)
    coef = pipe.named_steps[type(model).__name__.lower()].coef_ \
           if hasattr(model, "coef_") else pipe[-1].coef_
    n_nonzero = np.sum(np.abs(coef) > 1e-6)
    print(f"{name:25s} {scores.mean():10.4f} {n_nonzero:10d}")
```

### コード例2b: 正則化回帰のスクラッチ実装

```python
import numpy as np

class RegularizedRegression:
    """Ridge/Lasso/ElasticNetのスクラッチ実装"""

    def __init__(self, alpha=1.0, l1_ratio=0.0, method='ridge'):
        """
        method: 'ridge', 'lasso', 'elasticnet'
        l1_ratio: ElasticNetにおけるL1の比率（0=Ridge, 1=Lasso）
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.method = method
        self.weights = None
        self.bias = None

    def _ridge_fit(self, X, y):
        """Ridge回帰の閉形式解"""
        n, m = X.shape
        X_b = np.c_[np.ones((n, 1)), X]
        # (X^T X + αI)^(-1) X^T y （バイアスには正則化適用しない）
        I = np.eye(m + 1)
        I[0, 0] = 0  # バイアス項は正則化しない
        theta = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
        self.bias = theta[0]
        self.weights = theta[1:]

    def _lasso_fit(self, X, y, n_iter=1000, tol=1e-6):
        """Lasso回帰の座標降下法"""
        n, m = X.shape
        self.weights = np.zeros(m)
        self.bias = np.mean(y)

        for iteration in range(n_iter):
            weights_old = self.weights.copy()

            for j in range(m):
                # j番目の特徴量以外の予測値
                residual = y - self.bias - X @ self.weights + X[:, j] * self.weights[j]
                rho = X[:, j] @ residual / n

                # ソフト閾値処理（Soft Thresholding）
                if rho > self.alpha / 2:
                    self.weights[j] = (rho - self.alpha / 2) / (X[:, j] @ X[:, j] / n)
                elif rho < -self.alpha / 2:
                    self.weights[j] = (rho + self.alpha / 2) / (X[:, j] @ X[:, j] / n)
                else:
                    self.weights[j] = 0.0

            self.bias = np.mean(y - X @ self.weights)

            # 収束判定
            if np.max(np.abs(self.weights - weights_old)) < tol:
                break

    def _elasticnet_fit(self, X, y, n_iter=1000, tol=1e-6):
        """ElasticNet回帰の座標降下法"""
        n, m = X.shape
        self.weights = np.zeros(m)
        self.bias = np.mean(y)

        l1_penalty = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1 - self.l1_ratio)

        for iteration in range(n_iter):
            weights_old = self.weights.copy()

            for j in range(m):
                residual = y - self.bias - X @ self.weights + X[:, j] * self.weights[j]
                rho = X[:, j] @ residual / n

                denominator = (X[:, j] @ X[:, j] / n) + l2_penalty

                if rho > l1_penalty / 2:
                    self.weights[j] = (rho - l1_penalty / 2) / denominator
                elif rho < -l1_penalty / 2:
                    self.weights[j] = (rho + l1_penalty / 2) / denominator
                else:
                    self.weights[j] = 0.0

            self.bias = np.mean(y - X @ self.weights)

            if np.max(np.abs(self.weights - weights_old)) < tol:
                break

    def fit(self, X, y, **kwargs):
        if self.method == 'ridge':
            self._ridge_fit(X, y)
        elif self.method == 'lasso':
            self._lasso_fit(X, y, **kwargs)
        elif self.method == 'elasticnet':
            self._elasticnet_fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return X @ self.weights + self.bias


# 使用例
np.random.seed(42)
X = np.random.randn(200, 20)
true_coef = np.zeros(20)
true_coef[:5] = [3.0, -2.0, 1.5, -1.0, 0.5]
y = X @ true_coef + 1.0 + np.random.randn(200) * 0.3

# Ridge
ridge = RegularizedRegression(alpha=1.0, method='ridge')
ridge.fit(X, y)
print(f"Ridge 非ゼロ係数: {np.sum(np.abs(ridge.weights) > 1e-4)}")

# Lasso
lasso = RegularizedRegression(alpha=0.1, method='lasso')
lasso.fit(X, y)
print(f"Lasso 非ゼロ係数: {np.sum(np.abs(lasso.weights) > 1e-4)}")

# ElasticNet
enet = RegularizedRegression(alpha=0.1, l1_ratio=0.5, method='elasticnet')
enet.fit(X, y)
print(f"ElasticNet 非ゼロ係数: {np.sum(np.abs(enet.weights) > 1e-4)}")
```

### コード例3: 正則化パラメータの最適化

```python
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
import numpy as np
import matplotlib.pyplot as plt

# RidgeCV: 最適な α を交差検証で探索
alphas = np.logspace(-4, 4, 100)

ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X, y)
print(f"Ridge 最適α: {ridge_cv.alpha_:.4f}")

# LassoCV: 正則化パスで効率的に探索
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X, y)
print(f"Lasso 最適α: {lasso_cv.alpha_:.4f}")
print(f"Lasso 非ゼロ係数: {np.sum(np.abs(lasso_cv.coef_) > 1e-6)}")

# ElasticNetCV: αとl1_ratioの同時最適化
enet_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
    cv=5, random_state=42, max_iter=10000
)
enet_cv.fit(X, y)
print(f"ElasticNet 最適α: {enet_cv.alpha_:.4f}, l1_ratio: {enet_cv.l1_ratio_:.2f}")

# 正則化パスの可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Ridgeの係数パス
coefs_ridge = []
for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X, y)
    coefs_ridge.append(ridge.coef_)

ax1.semilogx(alphas, coefs_ridge)
ax1.axvline(ridge_cv.alpha_, color="r", linestyle="--", label=f"最適α={ridge_cv.alpha_:.3f}")
ax1.set_xlabel("α (正則化強度)")
ax1.set_ylabel("係数値")
ax1.set_title("Ridge 正則化パス")
ax1.legend()

# Lassoの係数パス
coefs_lasso = []
for a in alphas:
    lasso = Lasso(alpha=a, max_iter=10000)
    lasso.fit(X, y)
    coefs_lasso.append(lasso.coef_)

ax2.semilogx(alphas, coefs_lasso)
ax2.axvline(lasso_cv.alpha_, color="r", linestyle="--", label=f"最適α={lasso_cv.alpha_:.3f}")
ax2.set_xlabel("α (正則化強度)")
ax2.set_ylabel("係数値")
ax2.set_title("Lasso 正則化パス")
ax2.legend()

plt.tight_layout()
plt.savefig("reports/regularization_paths.png", dpi=150)
plt.close()
```

---

## 3. 多項式回帰

### コード例4: 多項式回帰と次数選択

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# 非線形データの生成
np.random.seed(42)
X = np.sort(np.random.uniform(-3, 3, 50)).reshape(-1, 1)
y = 0.5 * X.ravel()**3 - 2 * X.ravel()**2 + X.ravel() + np.random.randn(50) * 3

# 各次数でのフィットを比較
degrees = [1, 2, 3, 5, 10, 20]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)

for ax, degree in zip(axes.flatten(), degrees):
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    model.fit(X, y)
    y_plot = model.predict(X_plot)

    ax.scatter(X, y, s=20, alpha=0.6, label="データ")
    ax.plot(X_plot, y_plot, "r-", linewidth=2, label=f"次数={degree}")
    ax.set_title(f"次数={degree}, CV-MSE={-scores.mean():.2f}")
    ax.set_ylim(-40, 40)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/polynomial_degrees.png", dpi=150)
plt.close()
```

### コード例4b: バイアス-バリアンストレードオフの可視化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

def bias_variance_decomposition(X, y, degrees, n_bootstraps=100):
    """ブートストラップによるバイアス-バリアンス分解"""

    X_test = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    n = len(X)

    results = {"degree": [], "bias_sq": [], "variance": [], "mse": []}

    for degree in degrees:
        predictions = np.zeros((n_bootstraps, len(X_test)))

        for b in range(n_bootstraps):
            # ブートストラップサンプル
            idx = np.random.choice(n, size=n, replace=True)
            X_boot, y_boot = X[idx], y[idx]

            model = make_pipeline(
                PolynomialFeatures(degree),
                Ridge(alpha=0.001)
            )
            model.fit(X_boot, y_boot)
            predictions[b] = model.predict(X_test).ravel()

        # 真の関数（既知の場合）
        y_true = 0.5 * X_test.ravel()**3 - 2 * X_test.ravel()**2 + X_test.ravel()

        mean_pred = predictions.mean(axis=0)
        bias_sq = np.mean((mean_pred - y_true) ** 2)
        variance = np.mean(predictions.var(axis=0))
        mse = bias_sq + variance

        results["degree"].append(degree)
        results["bias_sq"].append(bias_sq)
        results["variance"].append(variance)
        results["mse"].append(mse)

    return results


# 実行
np.random.seed(42)
X = np.sort(np.random.uniform(-3, 3, 50)).reshape(-1, 1)
y = 0.5 * X.ravel()**3 - 2 * X.ravel()**2 + X.ravel() + np.random.randn(50) * 3

degrees = range(1, 16)
results = bias_variance_decomposition(X, y, degrees)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(results["degree"], results["bias_sq"], "b-o", label="Bias²", linewidth=2)
ax.plot(results["degree"], results["variance"], "r-o", label="Variance", linewidth=2)
ax.plot(results["degree"], results["mse"], "g--o", label="MSE (Bias²+Variance)", linewidth=2)
ax.set_xlabel("多項式の次数")
ax.set_ylabel("誤差")
ax.set_title("バイアス-バリアンストレードオフ")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reports/bias_variance_tradeoff.png", dpi=150)
plt.close()
```

---

## 4. 高度な回帰手法

### 4.1 ロバスト回帰

```python
import numpy as np
from sklearn.linear_model import (
    HuberRegressor, RANSACRegressor, TheilSenRegressor,
    LinearRegression
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# 外れ値を含むデータ
np.random.seed(42)
n = 100
X = np.random.randn(n, 1)
y = 3 * X.ravel() + 2 + np.random.randn(n) * 0.5

# 外れ値を追加（10%）
outlier_idx = np.random.choice(n, size=10, replace=False)
y[outlier_idx] += np.random.randn(10) * 20

models = {
    "OLS（通常の線形回帰）": LinearRegression(),
    "Huber回帰（ε=1.35）": HuberRegressor(epsilon=1.35),
    "RANSAC": RANSACRegressor(random_state=42),
    "Theil-Sen": TheilSenRegressor(random_state=42),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)

for ax, (name, model) in zip(axes.flatten(), models.items()):
    model.fit(X, y)
    y_pred = model.predict(X_plot)

    ax.scatter(X, y, alpha=0.5, s=30)
    ax.scatter(X[outlier_idx], y[outlier_idx], color='red', s=50,
               marker='x', label='外れ値')
    ax.plot(X_plot, y_pred, 'r-', linewidth=2, label=name)
    ax.set_title(name)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/robust_regression.png", dpi=150)
plt.close()

# 性能比較
print(f"{'モデル':30s} {'傾き':>8s} {'切片':>8s}")
print("-" * 50)
for name, model in models.items():
    model.fit(X, y)
    coef = model.coef_[0] if hasattr(model, 'coef_') else model.estimator_.coef_[0]
    intercept = model.intercept_ if hasattr(model, 'intercept_') else model.estimator_.intercept_
    print(f"{name:30s} {coef:8.3f} {intercept:8.3f}")
print(f"{'（真の値）':30s} {'3.000':>8s} {'2.000':>8s}")
```

### 4.2 ベイズ線形回帰

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ベイズ線形回帰 — 予測の不確実性も出力
np.random.seed(42)
X = np.sort(np.random.uniform(0, 10, 30)).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.randn(30) * 0.3

X_test = np.linspace(0, 10, 200).reshape(-1, 1)

# ベイズRidge（多項式特徴量付き）
model = make_pipeline(
    PolynomialFeatures(degree=7),
    BayesianRidge(
        alpha_1=1e-6, alpha_2=1e-6,
        lambda_1=1e-6, lambda_2=1e-6,
        compute_score=True
    )
)
model.fit(X, y)

# 予測と不確実性
y_mean, y_std = model.predict(X_test, return_std=True)

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(X, y, color='navy', s=40, label='訓練データ')
ax.plot(X_test, y_mean, 'r-', linewidth=2, label='予測平均')
ax.fill_between(
    X_test.ravel(),
    y_mean - 2 * y_std,
    y_mean + 2 * y_std,
    alpha=0.2, color='red',
    label='95%信頼区間'
)
ax.plot(X_test, np.sin(X_test.ravel()), 'g--', linewidth=1.5, label='真の関数')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("ベイズ線形回帰 — 予測と不確実性")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reports/bayesian_regression.png", dpi=150)
plt.close()
```

### 4.3 量子回帰（Quantile Regression）

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import QuantileRegressor

# 不均一分散を持つデータ
np.random.seed(42)
n = 200
X = np.random.uniform(0, 10, n).reshape(-1, 1)
y = 2 * X.ravel() + 3 + np.random.randn(n) * (0.5 + 0.3 * X.ravel())

X_test = np.linspace(0, 10, 100).reshape(-1, 1)

quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
colors = ['blue', 'cyan', 'red', 'cyan', 'blue']
linestyles = ['--', '-.', '-', '-.', '--']

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(X, y, alpha=0.3, s=15, color='gray')

for q, color, ls in zip(quantiles, colors, linestyles):
    model = QuantileRegressor(quantile=q, alpha=0.01, solver='highs')
    model.fit(X, y)
    y_pred = model.predict(X_test)
    ax.plot(X_test, y_pred, color=color, linestyle=ls,
            linewidth=2, label=f'Q{int(q*100)}')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("量子回帰 — 予測区間の推定")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reports/quantile_regression.png", dpi=150)
plt.close()
```

---

## 5. 多重共線性の診断と対策

### 5.1 VIF（分散拡大要因）による診断

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def calculate_vif(X, feature_names=None):
    """VIF（Variance Inflation Factor）の計算"""
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    vif_data = []
    for i in range(X.shape[1]):
        # i番目の特徴量を他の全特徴量で回帰
        X_others = np.delete(X, i, axis=1)
        y_i = X[:, i]

        model = LinearRegression()
        model.fit(X_others, y_i)
        r2 = model.score(X_others, y_i)

        vif = 1 / (1 - r2) if r2 < 1 else float('inf')
        vif_data.append({
            "特徴量": feature_names[i],
            "VIF": vif,
            "R²": r2,
            "判定": "問題なし" if vif < 5 else ("要注意" if vif < 10 else "多重共線性あり")
        })

    return pd.DataFrame(vif_data)


# 使用例: 多重共線性のあるデータ
np.random.seed(42)
x1 = np.random.randn(100)
x2 = 2 * x1 + np.random.randn(100) * 0.1  # x1とほぼ同じ
x3 = np.random.randn(100)                   # 独立
x4 = x1 + x3 + np.random.randn(100) * 0.5  # x1とx3の組み合わせ
x5 = np.random.randn(100)                   # 独立

X = np.column_stack([x1, x2, x3, x4, x5])
feature_names = ["x1", "x2 (≈2*x1)", "x3 (独立)", "x4 (x1+x3)", "x5 (独立)"]

vif_result = calculate_vif(X, feature_names)
print("=== VIF分析結果 ===")
print(vif_result.to_string(index=False))

print("\n判定基準:")
print("  VIF < 5   : 問題なし")
print("  5 ≤ VIF < 10 : 要注意（相関が高い特徴量がある）")
print("  VIF ≥ 10  : 多重共線性あり（対策が必要）")
```

### 5.2 条件数による診断

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

def condition_number_analysis(X, feature_names=None):
    """条件数による多重共線性の診断"""
    X_scaled = StandardScaler().fit_transform(X)

    # 特異値分解
    U, S, Vt = np.linalg.svd(X_scaled)
    condition_number = S[0] / S[-1]

    print(f"条件数: {condition_number:.2f}")
    print(f"  < 30  : 問題なし")
    print(f"  30-100: 中程度の多重共線性")
    print(f"  > 100 : 深刻な多重共線性")

    # 各特異値の情報
    print(f"\n特異値:")
    for i, s in enumerate(S):
        print(f"  σ_{i+1} = {s:.4f}  (寄与率: {s**2/np.sum(S**2)*100:.1f}%)")

    return condition_number
```

---

## 6. 特徴量エンジニアリングと前処理

### 6.1 回帰における特徴量変換

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer,
    PolynomialFeatures, SplineTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

def create_feature_engineering_pipeline(X, y, feature_names):
    """回帰のための特徴量エンジニアリングパイプライン"""

    # 各変換の効果を比較
    transformers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
        "PowerTransformer (Yeo-Johnson)": PowerTransformer(method='yeo-johnson'),
        "QuantileTransformer (Normal)": QuantileTransformer(output_distribution='normal'),
    }

    print(f"{'変換手法':40s} {'CV R²':>10s}")
    print("-" * 55)

    for name, transformer in transformers.items():
        pipe = Pipeline([
            ("transform", transformer),
            ("model", Ridge(alpha=1.0))
        ])
        scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")
        print(f"{name:40s} {scores.mean():10.4f} (+/- {scores.std():.4f})")


# 非線形変換の例
def add_nonlinear_features(X, feature_names):
    """手動での非線形特徴量追加"""
    df = pd.DataFrame(X, columns=feature_names)

    # 対数変換（正の値のみ）
    for col in feature_names:
        if (df[col] > 0).all():
            df[f"log_{col}"] = np.log(df[col])

    # 平方根変換
    for col in feature_names:
        if (df[col] >= 0).all():
            df[f"sqrt_{col}"] = np.sqrt(df[col])

    # 交互作用項
    for i, col1 in enumerate(feature_names):
        for col2 in feature_names[i+1:]:
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]

    return df


# スプライン変換の例
np.random.seed(42)
X = np.random.uniform(0, 10, (200, 1))
y = np.sin(X.ravel()) + 0.5 * X.ravel() + np.random.randn(200) * 0.3

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. 線形
model_linear = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=0.1))
])
model_linear.fit(X, y)

# 2. 多項式
model_poly = Pipeline([
    ("poly", PolynomialFeatures(degree=5)),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=0.1))
])
model_poly.fit(X, y)

# 3. スプライン
model_spline = Pipeline([
    ("spline", SplineTransformer(n_knots=8, degree=3)),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=0.1))
])
model_spline.fit(X, y)

X_test = np.linspace(0, 10, 200).reshape(-1, 1)

for ax, (name, model) in zip(axes, [
    ("線形", model_linear),
    ("多項式 (degree=5)", model_poly),
    ("Bスプライン (knots=8)", model_spline)
]):
    y_pred = model.predict(X_test)
    cv_score = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

    ax.scatter(X, y, alpha=0.3, s=10)
    ax.plot(X_test, y_pred, 'r-', linewidth=2)
    ax.set_title(f"{name}\nCV R² = {cv_score:.4f}")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/feature_transformations.png", dpi=150)
plt.close()
```

---

## 7. 実践的な回帰パイプライン

### コード例5: 本番品質の回帰パイプライン

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def build_regression_pipeline(
    numeric_features: list,
    categorical_features: list,
    poly_degree: int = 1
) -> Pipeline:
    """本番品質の回帰パイプラインを構築"""

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
    ])

    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge()),
    ])

    return pipeline

# 使用例
df = pd.DataFrame({
    "sqft": np.random.uniform(500, 3000, 200),
    "bedrooms": np.random.choice([1, 2, 3, 4, 5], 200),
    "location": np.random.choice(["urban", "suburban", "rural"], 200),
    "age": np.random.uniform(0, 50, 200),
})
df["price"] = (
    200 * df["sqft"] + 50000 * df["bedrooms"]
    - 1000 * df["age"]
    + np.where(df["location"] == "urban", 100000, 0)
    + np.random.randn(200) * 20000
)

X = df.drop(columns=["price"])
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = build_regression_pipeline(
    numeric_features=["sqft", "bedrooms", "age"],
    categorical_features=["location"],
    poly_degree=2
)

param_grid = {
    "preprocessor__num__poly__degree": [1, 2, 3],
    "regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_root_mean_squared_error")
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)
print(f"最良パラメータ: {grid.best_params_}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):,.0f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")
```

### コード例5b: 回帰モデルの包括的な評価

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)
from scipy import stats

def comprehensive_regression_evaluation(y_true, y_pred, feature_names=None, model=None):
    """回帰モデルの包括的な評価レポート"""

    residuals = y_true - y_pred

    # メトリクス
    metrics = {
        "R²": r2_score(y_true, y_pred),
        "Adjusted R²": None,  # 後で計算
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "Max Error": np.max(np.abs(residuals)),
        "Median AE": np.median(np.abs(residuals)),
    }

    print("=== 回帰モデル評価レポート ===\n")
    for name, value in metrics.items():
        if value is not None:
            print(f"  {name:15s}: {value:,.4f}")

    # 残差分析
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. 予測値 vs 実際値
    ax = axes[0, 0]
    ax.scatter(y_pred, y_true, alpha=0.5, s=20)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=2)
    ax.set_xlabel("予測値")
    ax.set_ylabel("実際値")
    ax.set_title("予測値 vs 実際値")
    ax.grid(True, alpha=0.3)

    # 2. 残差 vs 予測値（均一分散の確認）
    ax = axes[0, 1]
    ax.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel("予測値")
    ax.set_ylabel("残差")
    ax.set_title("残差 vs 予測値")
    ax.grid(True, alpha=0.3)

    # 3. QQプロット（正規性の確認）
    ax = axes[0, 2]
    stats.probplot(residuals, plot=ax)
    ax.set_title("Q-Qプロット（残差の正規性）")

    # 4. 残差のヒストグラム
    ax = axes[1, 0]
    ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7, density=True)
    # 正規分布のフィット
    mu, std = residuals.mean(), residuals.std()
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x_norm, stats.norm.pdf(x_norm, mu, std), 'r-', linewidth=2)
    ax.set_xlabel("残差")
    ax.set_ylabel("密度")
    ax.set_title(f"残差の分布 (平均={mu:.2f}, 標準偏差={std:.2f})")

    # 5. スケール-位置プロット（等分散性）
    ax = axes[1, 1]
    standardized_residuals = residuals / std
    ax.scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.5, s=20)
    ax.set_xlabel("予測値")
    ax.set_ylabel("√|標準化残差|")
    ax.set_title("スケール-位置プロット")
    ax.grid(True, alpha=0.3)

    # 6. 残差の自己相関
    ax = axes[1, 2]
    from statsmodels.graphics.tsaplots import plot_acf
    try:
        plot_acf(residuals, ax=ax, lags=20)
        ax.set_title("残差の自己相関 (ACF)")
    except ImportError:
        ax.bar(range(20), [np.corrcoef(residuals[:-i], residuals[i:])[0, 1]
                           if i > 0 else 1.0
                           for i in range(20)], alpha=0.7)
        ax.set_title("残差の自己相関")

    plt.tight_layout()
    plt.savefig("reports/regression_evaluation.png", dpi=150)
    plt.close()

    # 統計的検定
    print("\n=== 統計的検定 ===")

    # 正規性検定（Shapiro-Wilk）
    if len(residuals) <= 5000:
        stat, p_value = stats.shapiro(residuals)
        print(f"  Shapiro-Wilk検定: W={stat:.4f}, p={p_value:.4f}",
              "→ 正規性あり" if p_value > 0.05 else "→ 正規性なし")

    # Durbin-Watson検定（自己相関）
    dw = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
    print(f"  Durbin-Watson統計量: {dw:.4f}",
          "(≈2で自己相関なし, <1.5で正の自己相関, >2.5で負の自己相関)")

    return metrics


# 使用例
# comprehensive_regression_evaluation(y_test, y_pred)
```

---

## 8. 回帰指標の詳細

### 8.1 評価指標一覧

```
指標            式                              特徴
─────────────────────────────────────────────────────────────────────
MSE            Σ(y-ŷ)²/n                       二乗誤差、外れ値に敏感
RMSE           √(MSE)                          MSEと同じ単位
MAE            Σ|y-ŷ|/n                        絶対誤差、外れ値に頑健
MAPE           Σ|(y-ŷ)/y|/n × 100              百分率誤差、yが0に近いと不安定
R²             1 - SS_res/SS_tot               決定係数、1に近いほど良い
Adjusted R²    1 - (1-R²)(n-1)/(n-p-1)         特徴量数で補正したR²
AIC            n·ln(SS_res/n) + 2p             モデル選択（小さいほど良い）
BIC            n·ln(SS_res/n) + p·ln(n)         AICよりペナルティが大きい

R²の注意点:
  ・ R² = 0.9 は「予測精度が高い」とは限らない
  ・ 特徴量を追加するとR²は必ず増加する → Adjusted R²を使う
  ・ 外れ値の影響を大きく受ける
  ・ 非線形モデルの評価にはR²が不適切な場合がある
```

### 8.2 回帰指標の使い分け

```
場面                        推奨指標            理由
───────────────────────────────────────────────────────────
一般的な回帰評価            RMSE + R²           標準的
外れ値が多いデータ          MAE                 頑健性
ビジネス報告                MAPE                理解しやすい
モデル比較（異なる変数数）   Adjusted R²         公平な比較
モデル選択                  AIC / BIC           情報量基準
時系列予測                  RMSE + MAE          両方見る
住宅価格予測                RMSLE               対数スケール
需要予測                    MAPE + WMAPE        ビジネスKPI
```

---

## 比較表

### 回帰手法の選択ガイド

| 手法 | 正則化 | 特徴量選択 | 多重共線性 | 計算コスト | 適用場面 |
|---|---|---|---|---|---|
| OLS (線形回帰) | なし | 不可 | 弱い | O(n*m^2) | ベースライン、解釈重視 |
| Ridge (L2) | L2 | 不可 | 強い | O(n*m^2) | 多重共線性、全特徴量が重要 |
| Lasso (L1) | L1 | 可能 | 中程度 | 反復法 | スパースモデル、特徴量選択 |
| ElasticNet | L1+L2 | 可能 | 強い | 反復法 | 高次元、グループ化された特徴量 |
| 多項式回帰 | - | - | - | O(n*m^d) | 非線形関係がある場合 |
| Huber回帰 | L2 | 不可 | 中程度 | 反復法 | 外れ値が少数ある場合 |
| RANSAC | なし | 不可 | 弱い | 反復法 | 外れ値が多い場合 |
| ベイズ回帰 | 事前分布 | 可能 | 強い | O(n*m^2) | 不確実性の推定が必要 |
| 量子回帰 | - | - | - | 線形計画 | 予測区間の推定 |
| スプライン回帰 | - | - | - | O(n*k) | 滑らかな非線形関係 |

### 正則化パラメータ α の効果

| αの大きさ | バイアス | バリアンス | モデルの複雑度 | 係数の大きさ | 過学習リスク |
|---|---|---|---|---|---|
| α → 0 | 低い | 高い | 高い | 大きい | 高い |
| α が小さい | やや低い | やや高い | やや高い | やや大きい | やや高い |
| α が適切 | 中程度 | 中程度 | 適切 | 適切 | 低い |
| α が大きい | 高い | 低い | 低い | 小さい | 低い（過少適合） |
| α → ∞ | 最大 | 最小 | ゼロモデル | ≈ 0 | なし（過少適合） |

### スケーリング手法の比較

| 手法 | 変換式 | 出力範囲 | 外れ値への耐性 | 使い所 |
|---|---|---|---|---|
| StandardScaler | (x-μ)/σ | 概ね[-3, 3] | 低い | 正則化回帰（標準） |
| MinMaxScaler | (x-min)/(max-min) | [0, 1] | 低い | NN、距離ベース |
| RobustScaler | (x-Q2)/(Q3-Q1) | 可変 | 高い | 外れ値が多い |
| PowerTransformer | Box-Cox / Yeo-Johnson | 概ね正規 | 中程度 | 歪んだ分布 |
| QuantileTransformer | 分位点変換 | [0,1] or N(0,1) | 高い | 非線形関係 |

---

## アンチパターン

### アンチパターン1: スケーリングなしの正則化

```python
# BAD: スケーリングせずに正則化 → 単位の大きい特徴量が不当に罰される
from sklearn.linear_model import Lasso

# 面積(m², 10~200) vs 部屋数(1~5) → 面積の係数が小さくなりやすい
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)  # 不公平な正則化

# GOOD: StandardScalerで統一してから正則化
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), Lasso(alpha=1.0))
pipe.fit(X_train, y_train)  # 公平な正則化
```

### アンチパターン2: R²だけで回帰モデルを評価

```python
# BAD: R²が高い = 良いモデルとは限らない
# R² = 0.95 でも残差に自己相関や不均一分散があれば不適切

# GOOD: 残差分析を必ず実施
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
residuals = y_test - y_pred

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. 残差 vs 予測値（均一分散の確認）
axes[0].scatter(y_pred, residuals, alpha=0.5)
axes[0].axhline(y=0, color="r", linestyle="--")
axes[0].set_xlabel("予測値")
axes[0].set_ylabel("残差")
axes[0].set_title("残差 vs 予測値")

# 2. QQプロット（正規性の確認）
from scipy import stats
stats.probplot(residuals, plot=axes[1])
axes[1].set_title("Q-Qプロット")

# 3. 残差のヒストグラム
axes[2].hist(residuals, bins=30, edgecolor="black")
axes[2].set_title("残差の分布")

plt.tight_layout()
plt.savefig("reports/residual_analysis.png", dpi=150)
```

### アンチパターン3: 目的変数の変換を忘れる

```python
# BAD: 右に歪んだ目的変数（価格、所得）をそのまま使う
model = Ridge()
model.fit(X_train, y_train_skewed)  # 外れ値に引っ張られる

# GOOD: 対数変換してから学習
import numpy as np

y_train_log = np.log1p(y_train_skewed)  # log(1 + y)
model.fit(X_train, y_train_log)

# 予測値を元のスケールに戻す
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # exp(y) - 1
```

### アンチパターン4: テストデータでのフィッティング

```python
# BAD: テストデータを含めてfit_transform
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)  # テストデータの情報がリーク

# GOOD: 訓練データだけでfitし、テストデータにtransform
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 訓練データでfit
X_test_scaled = scaler.transform(X_test)          # テストデータはtransformのみ

# BEST: Pipelineを使って自動管理
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
pipe.fit(X_train, y_train)        # 内部で正しくfit_transform
pipe.predict(X_test)               # 内部で正しくtransform
```

### アンチパターン5: 多重共線性を無視する

```python
# BAD: 相関の高い特徴量をそのまま投入
# 身長(cm) と 身長(inch) を両方使う → 係数が不安定

# GOOD: VIF分析で確認し、対策を講じる
# 方法1: 相関の高い片方を削除
# 方法2: PCAで次元削減
# 方法3: Ridge回帰で多重共線性に対処

from sklearn.decomposition import PCA
pipe = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.95),  # 分散の95%を保持
    Ridge(alpha=1.0)
)
```

---

## FAQ

### Q1: RidgeとLassoのどちらを使うべき？

**A:** 不要な特徴量が多いと推測される場合はLasso（特徴量選択効果あり）。全ての特徴量がある程度重要と考えられる場合はRidge。相関の高い特徴量グループがある場合は、Lassoはグループ内の1つだけを選び他を0にする傾向があるため、ElasticNetの方が安定する。迷ったらElasticNetでL1/L2の比率も含めてCVで最適化するのが安全。

### Q2: 線形回帰で非線形な関係は捉えられないのか？

**A:** 特徴量変換（多項式、対数、平方根等）を適用すれば、パラメータに対しては線形のまま非線形関係を表現できる。「パラメータに対する線形性」と「入力に対する線形性」は別概念。PolynomialFeaturesで交互作用や多項式項を自動生成できる。SplineTransformerを使えば区分的な非線形性も柔軟に捉えられる。

### Q3: 正則化の強さ（α）はどう決めるのか？

**A:** 交差検証が標準的手法。scikit-learnのRidgeCV / LassoCV / ElasticNetCVは内部で効率的に交差検証を行う。αの探索範囲は対数スケール（10^-4〜10^4）で設定するのが一般的。LassoCVは正則化パスアルゴリズムを使うため、個別にαを試すよりも高速。

### Q4: 特徴量が多い場合、Lassoで自動選択に任せて良い？

**A:** Lassoは便利だが万能ではない。(1) 相関の高い特徴量がある場合、恣意的に1つだけ選ぶ（再現性が低い）、(2) αの値によって選択される特徴量が大きく変わる、(3) Stability Selection（ランダムサブサンプリング + Lasso を繰り返し、頻繁に選ばれる特徴量を最終選択）を使うとより安定する。

### Q5: 目的変数が正の値だけの場合（価格、件数など）、何か注意が必要？

**A:** (1) 対数変換（log1p）してから回帰するのが定石。分布の歪みが改善され、等分散性の仮定にも近づく。(2) 予測値に負の値が出ないようにする工夫が必要。(3) ポアソン回帰やガンマ回帰といった一般化線形モデル（GLM）の使用も検討。scikit-learnの`PoissonRegressor`、`GammaRegressor`が使える。

### Q6: サンプル数が少ない場合の回帰モデルはどう選ぶ？

**A:** (1) 特徴量数 > サンプル数の場合、OLSは不可（ランク落ち） → Ridge/Lassoが必須、(2) Leave-One-Out CVが有効（n-1個で学習、1個でテスト）、(3) ベイズ回帰で事前知識を導入すると安定する、(4) 正則化のαは大きめに設定する（過学習を防ぐ）。

---

## まとめ

| 項目 | 要点 |
|---|---|
| 線形回帰 | 最小二乗法。正規方程式 or 勾配降下法でパラメータ推定 |
| Ridge | L2正則化。多重共線性に強い。係数を縮小するがゼロにはしない |
| Lasso | L1正則化。不要な特徴量の係数をゼロにする（スパース性） |
| ElasticNet | L1+L2。LassoとRidgeの長所を兼ね備える |
| 多項式回帰 | 非線形関係を特徴量変換で捕捉。次数が高すぎると過学習 |
| ロバスト回帰 | 外れ値に頑健（Huber, RANSAC, Theil-Sen） |
| ベイズ回帰 | 予測の不確実性も出力。事前分布による正則化 |
| 前処理 | スケーリング必須、VIF分析、対数変換、スプライン変換 |
| 評価 | R²だけでなく残差分析、RMSE、MAE等を総合的に判断 |

---

## 次に読むべきガイド

- [01-classification.md](./01-classification.md) — 分類モデルの理論と実装
- [02-clustering.md](./02-clustering.md) — 教師なし学習（クラスタリング）

---

## 参考文献

1. **Trevor Hastie, Robert Tibshirani** "Statistical Learning with Sparsity: The Lasso and Generalizations" CRC Press, 2015
2. **scikit-learn** "Generalized Linear Models" — https://scikit-learn.org/stable/modules/linear_model.html
3. **Andrew Ng** "Machine Learning (Stanford CS229)" Lecture Notes — https://cs229.stanford.edu/
4. **Jerome Friedman, Trevor Hastie, Robert Tibshirani** "The Elements of Statistical Learning" Springer, 2009
5. **Peter J. Huber** "Robust Statistics" Wiley, 2004
6. **Christopher M. Bishop** "Pattern Recognition and Machine Learning" Springer, 2006
7. **Nicolai Meinshausen, Peter Buhlmann** "Stability Selection" JRSS Series B, 2010

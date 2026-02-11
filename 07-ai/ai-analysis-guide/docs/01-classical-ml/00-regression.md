# 回帰 — 線形/多項式/Ridge/Lasso

> 連続値を予測する回帰手法の理論と実装を、正則化まで含めて体系的に理解する

## この章で学ぶこと

1. **線形回帰の数理** — 最小二乗法、正規方程式、勾配降下法の原理
2. **正則化回帰** — Ridge（L2）、Lasso（L1）、ElasticNetによる過学習抑制
3. **多項式回帰と非線形拡張** — 特徴量の非線形変換と適切な次数選択

---

## 1. 線形回帰の基礎

### 最小二乗法の幾何学的解釈

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

---

## 2. 正則化回帰

### 正則化の効果

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

# 正則化パスの可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Ridgeの係数パス
from sklearn.linear_model import ridge_regression
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

### コード例5: 実践的な回帰パイプライン

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

---

## 比較表

### 回帰手法の選択ガイド

| 手法 | 正則化 | 特徴量選択 | 多重共線性 | 計算コスト | 適用場面 |
|---|---|---|---|---|---|
| OLS (線形回帰) | なし | × | 弱い | O(n×m²) | ベースライン、解釈重視 |
| Ridge (L2) | L2 | × | 強い | O(n×m²) | 多重共線性、全特徴量が重要 |
| Lasso (L1) | L1 | ○ | 中程度 | 反復法 | スパースモデル、特徴量選択 |
| ElasticNet | L1+L2 | ○ | 強い | 反復法 | 高次元、グループ化された特徴量 |
| 多項式回帰 | - | - | - | O(n×m^d) | 非線形関係がある場合 |

### 正則化パラメータ α の効果

| α の大きさ | バイアス | バリアンス | モデルの複雑度 | 係数の大きさ | 過学習リスク |
|---|---|---|---|---|---|
| α → 0 | 低い | 高い | 高い | 大きい | 高い |
| α が小さい | やや低い | やや高い | やや高い | やや大きい | やや高い |
| α が適切 | 中程度 | 中程度 | 適切 | 適切 | 低い |
| α が大きい | 高い | 低い | 低い | 小さい | 低い（過少適合） |
| α → ∞ | 最大 | 最小 | ゼロモデル | ≈ 0 | なし（過少適合） |

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

---

## FAQ

### Q1: RidgeとLassoのどちらを使うべき？

**A:** 不要な特徴量が多いと推測される場合はLasso（特徴量選択効果あり）。全ての特徴量がある程度重要と考えられる場合はRidge。迷ったらElasticNetでL1/L2の比率も含めてCVで最適化するのが安全。

### Q2: 線形回帰で非線形な関係は捉えられないのか？

**A:** 特徴量変換（多項式、対数、平方根等）を適用すれば、パラメータに対しては線形のまま非線形関係を表現できる。「パラメータに対する線形性」と「入力に対する線形性」は別概念。PolynomialFeaturesで交互作用や多項式項を自動生成できる。

### Q3: 正則化の強さ（α）はどう決めるのか？

**A:** 交差検証が標準的手法。scikit-learnのRidgeCV / LassoCV / ElasticNetCVは内部で効率的に交差検証を行う。αの探索範囲は対数スケール（10⁻⁴〜10⁴）で設定するのが一般的。

---

## まとめ

| 項目 | 要点 |
|---|---|
| 線形回帰 | 最小二乗法。正規方程式 or 勾配降下法でパラメータ推定 |
| Ridge | L2正則化。多重共線性に強い。係数を縮小するがゼロにはしない |
| Lasso | L1正則化。不要な特徴量の係数をゼロにする（スパース性） |
| ElasticNet | L1+L2。LassoとRidgeの長所を兼ね備える |
| 多項式回帰 | 非線形関係を特徴量変換で捕捉。次数が高すぎると過学習 |

---

## 次に読むべきガイド

- [01-classification.md](./01-classification.md) — 分類モデルの理論と実装
- [02-clustering.md](./02-clustering.md) — 教師なし学習（クラスタリング）

---

## 参考文献

1. **Trevor Hastie, Robert Tibshirani** "Statistical Learning with Sparsity: The Lasso and Generalizations" CRC Press, 2015
2. **scikit-learn** "Generalized Linear Models" — https://scikit-learn.org/stable/modules/linear_model.html
3. **Andrew Ng** "Machine Learning (Stanford CS229)" Lecture Notes — https://cs229.stanford.edu/

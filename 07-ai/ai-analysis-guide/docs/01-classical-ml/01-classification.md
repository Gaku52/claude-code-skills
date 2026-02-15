# 分類 — ロジスティック回帰、SVM、ランダムフォレスト

> 離散値（クラスラベル）を予測する分類手法の理論・実装・選択基準を網羅的に理解する

## この章で学ぶこと

1. **ロジスティック回帰** — シグモイド関数、最尤推定、確率的分類の原理
2. **サポートベクターマシン（SVM）** — マージン最大化、カーネルトリック、ソフトマージン
3. **決定木** — 情報利得、ジニ不純度、剪定
4. **アンサンブル学習** — ランダムフォレスト、勾配ブースティングの仕組みと使い分け
5. **評価指標と閾値最適化** — 混同行列、ROC/PR曲線、クラス不均衡対策

---

## 1. ロジスティック回帰

### 1.1 決定境界の仕組み

```
ロジスティック回帰の構造:

  入力 x₁, x₂, ..., xₘ
       │
       v
  線形結合: z = w₁x₁ + w₂x₂ + ... + wₘxₘ + b
       │
       v
  シグモイド: σ(z) = 1 / (1 + e^(-z))
       │
       v
  確率出力: P(y=1|x) = σ(z) ∈ [0, 1]
       │
       v
  閾値判定: ŷ = 1 if σ(z) ≥ 0.5 else 0

  σ(z) のグラフ:
  P(y=1)
  1.0 │              ___________
      │            /
  0.5 │-----------/
      │          /
  0.0 │_________/
      └──────────────────────── z
         -4  -2   0   2   4
```

### 1.2 最尤推定の数理

```
■ 尤度関数
  L(w) = Π P(yᵢ|xᵢ; w)
       = Π σ(wᵀxᵢ)^yᵢ × (1 - σ(wᵀxᵢ))^(1-yᵢ)

■ 対数尤度（最大化対象）
  ℓ(w) = Σ [yᵢ log σ(wᵀxᵢ) + (1-yᵢ) log(1 - σ(wᵀxᵢ))]

■ 交差エントロピー損失（最小化対象） = -ℓ(w)/n

■ 勾配
  ∂ℓ/∂w = Σ (yᵢ - σ(wᵀxᵢ)) xᵢ

■ 正則化付きの場合
  L1正則化: ℓ(w) - λΣ|wⱼ|  → スパースな解
  L2正則化: ℓ(w) - λΣwⱼ²   → 滑らかな解

  scikit-learnでは C = 1/λ （Cが大きい = 正則化が弱い）
```

### コード例1: ロジスティック回帰のスクラッチ実装

```python
import numpy as np

class LogisticRegressionScratch:
    """ロジスティック回帰のフルスクラッチ実装"""

    def __init__(self, learning_rate=0.01, n_iter=1000, l2_lambda=0.0):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.l2_lambda = l2_lambda
        self.weights = None
        self.bias = None
        self.history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n, m = X.shape
        self.weights = np.zeros(m)
        self.bias = 0.0

        for i in range(self.n_iter):
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)

            # 交差エントロピー損失
            loss = -np.mean(
                y * np.log(y_pred + 1e-8) +
                (1 - y) * np.log(1 - y_pred + 1e-8)
            )
            if self.l2_lambda > 0:
                loss += self.l2_lambda / (2 * n) * np.sum(self.weights ** 2)

            # 勾配計算
            error = y_pred - y
            dw = (1 / n) * (X.T @ error) + (self.l2_lambda / n) * self.weights
            db = (1 / n) * np.sum(error)

            # パラメータ更新
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            self.history.append(loss)

        return self

    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


# 使用例
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X, y = data.data, data.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegressionScratch(learning_rate=0.1, n_iter=500, l2_lambda=0.01)
model.fit(X_train, y_train)

print(f"訓練精度: {model.accuracy(X_train, y_train):.4f}")
print(f"テスト精度: {model.accuracy(X_test, y_test):.4f}")
```

### コード例1b: ロジスティック回帰の詳細分析（scikit-learn）

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.datasets import load_breast_cancer

# データ準備
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 正則化強度の比較
print(f"{'C':>8s} {'Penalty':>8s} {'AUC':>10s} {'非ゼロ':>6s}")
print("-" * 40)
for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    for penalty in ['l1', 'l2']:
        solver = 'saga' if penalty == 'l1' else 'lbfgs'
        lr = LogisticRegression(
            C=C, penalty=penalty, solver=solver,
            max_iter=5000, random_state=42
        )
        scores = cross_val_score(lr, X_train_s, y_train, cv=5, scoring="roc_auc")
        lr.fit(X_train_s, y_train)
        n_nonzero = np.sum(np.abs(lr.coef_[0]) > 1e-4)
        print(f"{C:8.3f} {penalty:>8s} {scores.mean():10.4f} {n_nonzero:6d}")

# 最良モデルの特徴量重要度
best_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
best_lr.fit(X_train_s, y_train)

importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": best_lr.coef_[0],
    "abs_coef": np.abs(best_lr.coef_[0]),
    "odds_ratio": np.exp(best_lr.coef_[0])
}).sort_values("abs_coef", ascending=False).head(10)

print("\n--- 重要特徴量 Top 10 ---")
print(importance.to_string(index=False))

# テスト評価
y_pred = best_lr.predict(X_test_s)
y_prob = best_lr.predict_proba(X_test_s)[:, 1]
print(f"\nAUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=data.target_names))
```

### 1.3 多クラス分類への拡張

```
■ One-vs-Rest (OvR)
  K個のクラスがある場合、K個の二値分類器を学習
  クラスk: 「クラスkか否か」の二値分類
  予測: P(y=k|x) が最大のクラスを選択

■ One-vs-One (OvO)
  K(K-1)/2 個の二値分類器を学習
  各ペア (i, j) について分類器を構築
  予測: 多数決で決定

■ Softmax回帰（多クラスロジスティック回帰）
  P(y=k|x) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx)
  全クラスを同時に学習
  scikit-learn: multi_class="multinomial"

実務上の選択:
  ・ クラス数 ≤ 10: Softmax（推奨）
  ・ クラス数が多い場合: OvR（計算効率）
  ・ SVMの場合: OvO（scikit-learnのデフォルト）
```

---

## 2. サポートベクターマシン (SVM)

### 2.1 マージン最大化とカーネルトリック

```
線形SVM（ハードマージン）:

     クラス +1: ●        決定境界: w·x + b = 0
     クラス -1: ○
                         マージン
  x₂ │                 ←────→
     │  ●  ●         /  ____  \
     │    ●  ●      / /    \ \
     │      ●      / /      \ \  ← サポートベクター
     │            / /        \ \    (境界に最も近い点)
     │    ───────/ /──────────\ \────
     │          / /            \ \
     │   ○  ○ / /    ○         \ \
     │  ○   ○                   ○
     └─────────────────────────────── x₁

カーネルトリック（非線形分類）:

  入力空間（線形分離不可）      特徴空間（線形分離可能）
  ┌──────────────┐           ┌──────────────┐
  │  ○ ● ○       │           │         ●    │
  │ ● ● ● ○      │  φ(x)    │   ●   ● ●   │
  │ ○ ● ● ○      │ ──────>  │  ────────── │
  │  ○ ○ ○       │           │ ○   ○   ○   │
  │              │           │○     ○      │
  └──────────────┘           └──────────────┘
  カーネル K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ)
```

### 2.2 SVMの数学的定式化

```
■ ハードマージンSVM（線形分離可能な場合）
  最小化: (1/2)||w||²
  制約:  yᵢ(wᵀxᵢ + b) ≥ 1,  ∀i

■ ソフトマージンSVM（線形分離不可能な場合）
  最小化: (1/2)||w||² + C·Σξᵢ
  制約:  yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ,  ξᵢ ≥ 0

  C: 誤分類のペナルティ
    C大 → マージン小、誤分類少ない（過学習リスク）
    C小 → マージン大、誤分類許容（汎化性能重視）

■ 主要カーネル
  線形:     K(x, z) = xᵀz
  多項式:   K(x, z) = (γxᵀz + r)^d
  RBF:      K(x, z) = exp(-γ||x - z||²)
  シグモイド: K(x, z) = tanh(γxᵀz + r)

■ RBFカーネルの γ パラメータ
  γ大 → 各サンプルの影響範囲が狭い → 複雑な境界（過学習リスク）
  γ小 → 各サンプルの影響範囲が広い → 滑らかな境界（過少適合リスク）
```

### コード例2: SVM カーネルの比較

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 非線形データの生成
X_moon, y_moon = make_moons(n_samples=300, noise=0.2, random_state=42)
X_circ, y_circ = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)

datasets = [("Moons", X_moon, y_moon), ("Circles", X_circ, y_circ)]
kernels = ["linear", "poly", "rbf", "sigmoid"]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for row, (name, X, y) in enumerate(datasets):
    for col, kernel in enumerate(kernels):
        ax = axes[row][col]
        pipe = make_pipeline(StandardScaler(), SVC(kernel=kernel, gamma="auto"))
        scores = cross_val_score(pipe, X, y, cv=5)
        pipe.fit(X, y)

        # 決定境界の描画
        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200),
            np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 200)
        )
        Z = pipe.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=20)
        ax.set_title(f"{name} / {kernel}\nCV Acc={scores.mean():.3f}")

plt.tight_layout()
plt.savefig("reports/svm_kernels.png", dpi=150)
plt.close()
```

### コード例2b: SVMのハイパーパラメータチューニング

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target

# C と gamma の同時最適化（RBFカーネル）
param_grid = {
    'svc__C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'svc__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
}

pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
grid = GridSearchCV(
    pipe, param_grid,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='roc_auc', n_jobs=-1, verbose=0
)
grid.fit(X, y)

print(f"最良パラメータ: {grid.best_params_}")
print(f"最良AUC: {grid.best_score_:.4f}")

# C-gamma のヒートマップ
import pandas as pd
results = pd.DataFrame(grid.cv_results_)
# C と gamma が数値の場合のみヒートマップ化
numeric_results = results[results['param_svc__gamma'].apply(lambda x: isinstance(x, float))]
if len(numeric_results) > 0:
    pivot = numeric_results.pivot_table(
        values='mean_test_score',
        index='param_svc__C',
        columns='param_svc__gamma'
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    import seaborn as sns
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='viridis', ax=ax)
    ax.set_xlabel('gamma')
    ax.set_ylabel('C')
    ax.set_title('SVM RBFカーネル: C x gamma のAUCスコア')
    plt.tight_layout()
    plt.savefig("reports/svm_heatmap.png", dpi=150)
    plt.close()
```

---

## 3. 決定木

### 3.1 決定木のアルゴリズム

```
■ 分割基準

  ジニ不純度 (Gini Impurity):
    G(t) = 1 - Σₖ pₖ²
    0 = 完全にピュア、0.5 = 最大不純度（2クラスの場合）

  エントロピー (Information Gain):
    H(t) = -Σₖ pₖ log₂(pₖ)
    0 = 完全にピュア、1 = 最大不純度（2クラスの場合）

  情報利得:
    IG = H(親) - Σ (|子ノードi| / |親|) × H(子ノードi)

■ 決定木の可視化例
                      [全データ: 100件]
                     feature_A ≤ 5.0 ?
                    /              \
           [左: 60件]          [右: 40件]
         feature_B ≤ 3.2 ?    feature_C ≤ 7.0 ?
          /        \            /        \
    [30件]      [30件]     [25件]     [15件]
   Class=0    Class=1    Class=1    Class=0

■ 剪定（Pruning）
  ・ 事前剪定: max_depth, min_samples_split, min_samples_leaf
  ・ 事後剪定: ccp_alpha（Cost-Complexity Pruning）
  ・ 事前剪定の方が計算コストが低く、一般的
```

### コード例2c: 決定木の可視化と剪定

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target

# 剪定パラメータの影響
fig, axes = plt.subplots(2, 3, figsize=(24, 14))

configs = [
    ("max_depth=2", {"max_depth": 2}),
    ("max_depth=5", {"max_depth": 5}),
    ("max_depth=None（制限なし）", {}),
    ("min_samples_leaf=10", {"min_samples_leaf": 10}),
    ("min_samples_split=20", {"min_samples_split": 20}),
    ("ccp_alpha=0.02", {"ccp_alpha": 0.02}),
]

for ax, (title, params) in zip(axes.flatten(), configs):
    tree = DecisionTreeClassifier(random_state=42, **params)
    scores = cross_val_score(tree, X, y, cv=5, scoring="accuracy")
    tree.fit(X, y)

    plot_tree(tree, ax=ax, feature_names=data.feature_names,
              class_names=data.target_names, filled=True,
              fontsize=7, max_depth=3)
    ax.set_title(f"{title}\nCV Acc={scores.mean():.3f} (depth={tree.get_depth()}, "
                 f"leaves={tree.get_n_leaves()})")

plt.tight_layout()
plt.savefig("reports/decision_tree_pruning.png", dpi=150)
plt.close()

# Cost-Complexity Pruning Path
tree_full = DecisionTreeClassifier(random_state=42)
tree_full.fit(X, y)

path = tree_full.cost_complexity_pruning_path(X, y)
ccp_alphas = path.ccp_alphas[:-1]  # 最後は単一ノードなので除外

train_scores = []
test_scores = []
for alpha in ccp_alphas:
    tree = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    scores = cross_val_score(tree, X, y, cv=5, scoring="accuracy")
    tree.fit(X, y)
    train_scores.append(tree.score(X, y))
    test_scores.append(scores.mean())

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ccp_alphas, train_scores, "b-o", label="Train Accuracy", markersize=3)
ax.plot(ccp_alphas, test_scores, "r-o", label="CV Accuracy", markersize=3)
ax.set_xlabel("ccp_alpha")
ax.set_ylabel("Accuracy")
ax.set_title("Cost-Complexity Pruning Path")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reports/ccp_pruning_path.png", dpi=150)
plt.close()
```

---

## 4. アンサンブル学習

### 4.1 バギング vs ブースティング

```
バギング (Bagging) — Random Forest:

  元データ D
  ┌──────────┐
  │ ブートストラップサンプリング
  ├──────────┼──────────┼──────────┐
  │  D₁      │  D₂      │  D₃      │  ... Dₙ
  │  ↓       │  ↓       │  ↓       │
  │ Tree₁    │ Tree₂    │ Tree₃    │  ... Treeₙ
  │  ↓       │  ↓       │  ↓       │
  │ pred₁    │ pred₂    │ pred₃    │  ... predₙ
  └──────────┴──────────┴──────────┘
              │ 多数決 (分類) / 平均 (回帰)
              v
          最終予測

ブースティング (Boosting) — GBM / XGBoost:

  弱学習器を逐次的に追加:
  ┌────────┐   ┌────────┐   ┌────────┐
  │ Tree₁  │──>│ Tree₂  │──>│ Tree₃  │──> ... → 最終予測
  │        │   │ 残差1を │   │ 残差2を │
  │ 全体を │   │ 学習    │   │ 学習    │
  │ 学習   │   │        │   │        │
  └────────┘   └────────┘   └────────┘
     ↓             ↓             ↓
  残差₁ =      残差₂ =      残差₃ = ...
  y - ŷ₁      残差₁ - ŷ₂   残差₂ - ŷ₃

スタッキング (Stacking):
  ┌──────────────────────────────────┐
  │ レベル0: 複数のベースモデル      │
  │  LR   RF   XGB   SVM   KNN     │
  │  ↓    ↓    ↓     ↓     ↓       │
  │ p₁   p₂   p₃    p₄    p₅      │
  │  └────┴────┴─────┴─────┘       │
  │           ↓                      │
  │ レベル1: メタモデル（LR等）      │
  │           ↓                      │
  │       最終予測                   │
  └──────────────────────────────────┘
```

### コード例3: ランダムフォレストの実装と分析

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# データ準備
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ハイパーパラメータの影響分析
results = []
for n_est in [10, 50, 100, 200, 500]:
    for max_depth in [3, 5, 10, None]:
        rf = RandomForestClassifier(
            n_estimators=n_est, max_depth=max_depth,
            random_state=42, n_jobs=-1
        )
        scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="f1")
        results.append({
            "n_estimators": n_est,
            "max_depth": max_depth or "None",
            "f1_mean": scores.mean(),
            "f1_std": scores.std()
        })

results_df = pd.DataFrame(results).sort_values("f1_mean", ascending=False)
print(results_df.head(10).to_string(index=False))

# 最良モデルの特徴量重要度
best_rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
best_rf.fit(X_train, y_train)

importance = pd.DataFrame({
    "feature": data.feature_names,
    "importance": best_rf.feature_importances_
}).sort_values("importance", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance["feature"], importance["importance"])
ax.set_xlabel("重要度 (Gini Importance)")
ax.set_title("ランダムフォレスト 特徴量重要度 Top 15")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("reports/rf_feature_importance.png", dpi=150)
plt.close()
```

### コード例3b: Permutation ImportanceとSHAP値

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Permutation Importance（テストデータで計算 → より信頼性が高い）
perm_imp = permutation_importance(
    rf, X_test, y_test,
    n_repeats=30, random_state=42, n_jobs=-1
)

# Gini Importance vs Permutation Importance の比較
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Gini Importance
sorted_idx = rf.feature_importances_.argsort()[-15:]
ax1.barh(data.feature_names[sorted_idx], rf.feature_importances_[sorted_idx])
ax1.set_title("Gini Importance (MDI)")
ax1.set_xlabel("重要度")

# Permutation Importance
sorted_idx_perm = perm_imp.importances_mean.argsort()[-15:]
ax2.barh(
    data.feature_names[sorted_idx_perm],
    perm_imp.importances_mean[sorted_idx_perm]
)
ax2.errorbar(
    perm_imp.importances_mean[sorted_idx_perm],
    range(15),
    xerr=perm_imp.importances_std[sorted_idx_perm],
    fmt='none', color='black', capsize=3
)
ax2.set_title("Permutation Importance (テストデータ)")
ax2.set_xlabel("精度低下量")

plt.tight_layout()
plt.savefig("reports/importance_comparison.png", dpi=150)
plt.close()

# SHAP値（shap パッケージが必要）
try:
    import shap

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)

    # Summary Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values[1], X_test,
                      feature_names=data.feature_names, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png", dpi=150)
    plt.close()

    print("SHAP値の計算完了")
except ImportError:
    print("shapパッケージがインストールされていません: pip install shap")
```

### コード例4: 勾配ブースティングの実装

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb
import lightgbm as lgb

# scikit-learn GBM
gb_sklearn = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, random_state=42
)

# XGBoost
gb_xgb = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="logloss",
    random_state=42
)

# LightGBM
gb_lgb = lgb.LGBMClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1
)

models = {
    "sklearn GBM": gb_sklearn,
    "XGBoost": gb_xgb,
    "LightGBM": gb_lgb,
}

import time
for name, model in models.items():
    start = time.time()
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
    elapsed = time.time() - start
    print(f"{name:15s}  F1={scores.mean():.4f}+/-{scores.std():.4f}  "
          f"時間={elapsed:.2f}秒")
```

### コード例4b: LightGBMの詳細チューニング

```python
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
import optuna

def objective(trial):
    """OptunaによるLightGBMのハイパーパラメータ最適化"""

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 8, 256),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbose': -1,
    }

    model = lgb.LGBMClassifier(**params)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        from sklearn.metrics import roc_auc_score
        y_pred = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, y_pred))

    return np.mean(scores)


# 最適化の実行
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, timeout=300)

print(f"\n最良AUC: {study.best_value:.4f}")
print(f"最良パラメータ:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
```

### コード例4c: スタッキングアンサンブル

```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb

# ベースモデル
estimators = [
    ('lr', make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=100, use_label_encoder=False,
                                eval_metric='logloss', random_state=42)),
    ('svm', make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))),
    ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
]

# スタッキング
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1
)

# 各モデルとスタッキングの比較
print(f"{'モデル':20s} {'CV AUC':>10s}")
print("-" * 35)

for name, model in estimators:
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"{name:20s} {scores.mean():10.4f}")

scores = cross_val_score(stacking, X_train, y_train, cv=5, scoring='roc_auc')
print(f"{'Stacking':20s} {scores.mean():10.4f}")
```

---

## 5. 評価指標と閾値最適化

### 5.1 混同行列と主要指標

```
                予測: Positive    予測: Negative
実際: Positive    TP (True Pos)    FN (False Neg)
実際: Negative    FP (False Pos)   TN (True Neg)

精度 (Accuracy)    = (TP + TN) / (TP + TN + FP + FN)
適合率 (Precision) = TP / (TP + FP)  → 「陽性と予測した中で本当に陽性の割合」
再現率 (Recall)    = TP / (TP + FN)  → 「実際の陽性の中で正しく検出した割合」
F1スコア          = 2 × P × R / (P + R)  → PrecisionとRecallの調和平均
特異度 (Specificity) = TN / (TN + FP)  → 「実際の陰性の中で正しく除外した割合」

■ タスクに応じた重要指標
  ・ スパム検出: Precision重視（正常メールを誤ってスパム判定したくない）
  ・ 癌検診: Recall重視（癌患者を見逃したくない）
  ・ 不正検知: Recall重視 + 高Precision → F1スコア
  ・ 一般的な分類: F1スコア or AUC-ROC
```

### 5.2 ROC曲線とPR曲線

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, RocCurveDisplay,
    PrecisionRecallDisplay
)

def plot_roc_and_pr_curves(models_dict, X_test, y_test):
    """複数モデルのROC曲線とPR曲線を比較描画"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for name, model in models_dict.items():
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        # ROC曲線
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={roc_auc:.3f})')

        # PR曲線
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax2.plot(recall, precision, linewidth=2, label=f'{name} (AP={ap:.3f})')

    # ROC曲線の仕上げ
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC曲線')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # PR曲線の仕上げ
    baseline = y_test.sum() / len(y_test)
    ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
                label=f'ランダム (AP={baseline:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall曲線')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reports/roc_pr_curves.png", dpi=150)
    plt.close()
```

### コード例5: 閾値最適化

```python
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score
import matplotlib.pyplot as plt

def optimize_threshold(y_true, y_prob, metric="f1"):
    """最適な分類閾値を探索"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # 各閾値でのF1スコアを計算
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    # 最適閾値
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]

    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(thresholds, precisions[:-1], label="Precision")
    ax1.plot(thresholds, recalls[:-1], label="Recall")
    ax1.plot(thresholds, f1_scores[:-1], label="F1", linewidth=2)
    ax1.axvline(best_threshold, color="r", linestyle="--",
                label=f"最適閾値={best_threshold:.3f}")
    ax1.set_xlabel("閾値")
    ax1.set_ylabel("スコア")
    ax1.set_title("閾値 vs 指標")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(recalls, precisions)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall曲線")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reports/threshold_optimization.png", dpi=150)
    plt.close()

    print(f"最適閾値: {best_threshold:.4f}")
    print(f"最適F1: {best_f1:.4f}")
    return best_threshold

# 使用例
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
y_prob = model.predict_proba(X_test_s)[:, 1]
best_th = optimize_threshold(y_test, y_prob)

# 最適閾値で予測
y_pred_opt = (y_prob >= best_th).astype(int)
print(f"\nデフォルト閾値(0.5) F1: {f1_score(y_test, model.predict(X_test_s)):.4f}")
print(f"最適閾値({best_th:.3f}) F1: {f1_score(y_test, y_pred_opt):.4f}")
```

---

## 6. クラス不均衡への対策

### 6.1 サンプリング手法

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

# 不均衡データの生成（1:10の比率）
X_imb, y_imb = make_classification(
    n_samples=10000, n_features=20,
    n_informative=10, n_redundant=5,
    weights=[0.9, 0.1],  # 90% vs 10%
    random_state=42
)

print(f"クラス比率: {np.bincount(y_imb)}")

# 各対策の比較
from sklearn.utils.class_weight import compute_class_weight

results = {}

# 1. 何もしない
lr_base = LogisticRegression(max_iter=1000)
scores = cross_val_score(lr_base, X_imb, y_imb, cv=5, scoring='f1')
results["何もしない"] = scores.mean()

# 2. class_weight='balanced'
lr_balanced = LogisticRegression(max_iter=1000, class_weight='balanced')
scores = cross_val_score(lr_balanced, X_imb, y_imb, cv=5, scoring='f1')
results["class_weight=balanced"] = scores.mean()

# 3. SMOTE（imblearn が必要）
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    smote_pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('lr', LogisticRegression(max_iter=1000))
    ])
    scores = cross_val_score(smote_pipe, X_imb, y_imb, cv=5, scoring='f1')
    results["SMOTE"] = scores.mean()
except ImportError:
    print("imbalanced-learn未インストール: pip install imbalanced-learn")

# 4. ランダムフォレスト + class_weight
rf_balanced = RandomForestClassifier(
    n_estimators=200, class_weight='balanced_subsample', random_state=42
)
scores = cross_val_score(rf_balanced, X_imb, y_imb, cv=5, scoring='f1')
results["RF+balanced_subsample"] = scores.mean()

print(f"\n{'対策':30s} {'F1':>8s}")
print("-" * 42)
for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:30s} {score:8.4f}")
```

---

## 7. k近傍法（k-NN）とナイーブベイズ

### 7.1 k近傍法

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# kの最適化
k_range = range(1, 31)
scores = []

for k in k_range:
    pipe = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k, weights='distance')
    )
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
    scores.append(cv_scores.mean())

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_range, scores, 'b-o', markersize=5)
best_k = k_range[np.argmax(scores)]
ax.axvline(best_k, color='r', linestyle='--', label=f'最適 k={best_k}')
ax.set_xlabel('k (近傍数)')
ax.set_ylabel('CV Accuracy')
ax.set_title('k-NN: kの最適化')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reports/knn_optimization.png", dpi=150)
plt.close()
```

### 7.2 ナイーブベイズ

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

# ナイーブベイズの種類と適用場面
nb_models = {
    "GaussianNB": GaussianNB(),  # 連続値特徴量
    "MultinomialNB": make_pipeline(
        MinMaxScaler(), MultinomialNB()  # カウントデータ（テキスト分類）
    ),
    "BernoulliNB": BernoulliNB(),  # 二値特徴量
}

print(f"{'モデル':20s} {'CV Accuracy':>12s}")
print("-" * 35)
for name, model in nb_models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name:20s} {scores.mean():12.4f}")
```

---

## 比較表

### 分類アルゴリズムの特性比較

| アルゴリズム | 学習速度 | 予測速度 | 解釈性 | 非線形対応 | スケーリング要否 | 欠損値対応 |
|---|---|---|---|---|---|---|
| ロジスティック回帰 | 速い | 極速 | 高い | 不可 (特徴量変換で可) | 要 | 不可 |
| SVM (線形) | 速い | 極速 | 中程度 | 不可 | 要 | 不可 |
| SVM (RBF) | 遅い | 遅い | 低い | 可能 | 要 | 不可 |
| 決定木 | 速い | 極速 | 高い | 可能 | 不要 | 一部可 |
| ランダムフォレスト | 中程度 | 中程度 | 中程度 | 可能 | 不要 | 一部可 |
| XGBoost / LightGBM | 中程度 | 速い | 低い | 可能 | 不要 | 可能 |
| k-NN | 不要 | 遅い | 中程度 | 可能 | 要 | 不可 |
| ナイーブベイズ | 極速 | 極速 | 高い | 不可 | 場合による | 不可 |

### クラス不均衡への対処法

| 手法 | カテゴリ | 説明 | メリット | デメリット |
|---|---|---|---|---|
| class_weight="balanced" | アルゴリズム側 | 少数クラスの重みを増加 | 簡単 | 過学習リスク |
| SMOTE | オーバーサンプリング | 少数クラスの合成サンプル生成 | データ量増加 | ノイズ増加の可能性 |
| ADASYN | オーバーサンプリング | 困難なサンプル付近で多く生成 | 適応的 | 計算コスト |
| RandomUnderSampler | アンダーサンプリング | 多数クラスをランダム削減 | 高速化 | 情報喪失 |
| TomekLinks | アンダーサンプリング | 境界付近のノイズを除去 | ノイズ除去 | 効果が限定的 |
| 閾値調整 | 後処理 | 分類閾値を最適化 | モデル変更不要 | 検証データが必要 |
| Focal Loss | 損失関数 | 簡単な例の重みを下げる | 効果的 | カスタム実装が必要 |
| コスト敏感学習 | 損失関数 | 誤分類コストを非対称に設定 | 柔軟 | コスト設定が難しい |

### 評価指標の選択ガイド

| 場面 | 推奨指標 | 理由 |
|---|---|---|
| 均衡データの一般分類 | Accuracy, F1 | 標準的 |
| 不均衡データ | F1, AUC-PR | Accuracyは多数クラス偏重 |
| 医療診断（見逃し防止） | Recall, Sensitivity | FNの最小化が最重要 |
| スパム検出 | Precision | FPの最小化が重要 |
| ランキング・情報検索 | AUC-ROC, MAP | 順序の正しさが重要 |
| 多クラス分類 | Macro F1, Weighted F1 | クラス別の性能バランス |
| 確率出力の校正 | Brier Score, Log Loss | 確率の正確さ |
| モデル比較 | AUC-ROC | 閾値非依存 |

---

## アンチパターン

### アンチパターン1: 多クラス分類でのAccuracy偏重

```python
# BAD: マクロ平均を見ない
from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# → 多数クラスを当てるだけで高スコアになりうる

# GOOD: クラスごとの性能を確認
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
# → precision, recall, f1をクラスごとに確認
# → macro avg と weighted avg の差が大きい場合は不均衡の影響あり
```

### アンチパターン2: SVMに大規模データを適用

```python
# BAD: 100万件のデータにRBF SVMを適用（O(n²)〜O(n³)で非現実的）
from sklearn.svm import SVC
svc = SVC(kernel="rbf")
svc.fit(X_large, y_large)  # メモリ不足 or 数時間かかる

# GOOD: 大規模データには線形SVMかGBMを使用
from sklearn.svm import LinearSVC
# または
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss="hinge", random_state=42)  # 線形SVMと等価
sgd.fit(X_large, y_large)  # O(n) で高速

# RBF的な非線形が必要なら
from sklearn.kernel_approximation import RBFSampler
rbf_feature = RBFSampler(gamma=1.0, n_components=100, random_state=42)
X_rbf = rbf_feature.fit_transform(X_large)
sgd.fit(X_rbf, y_large)  # 近似カーネルで高速化
```

### アンチパターン3: 特徴量重要度の誤った解釈

```python
# BAD: Gini Importanceだけで特徴量の重要性を判断
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print("重要度:", rf.feature_importances_)
# → カーディナリティの高い特徴量（カテゴリ数が多い）が過大評価される
# → 相関のある特徴量間で重要度が分散する

# GOOD: Permutation Importanceを併用
from sklearn.inspection import permutation_importance
perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=30)
# → テストデータで計算するため、過学習の影響を受けにくい
# → SHAP値も併用するとより信頼性が高い
```

### アンチパターン4: 交差検証の中でデータリーク

```python
# BAD: 交差検証の外でSMOTEを適用
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
scores = cross_val_score(model, X_resampled, y_resampled, cv=5)
# → 検証フォールドに合成サンプルの情報がリーク

# GOOD: imblearnのPipelineで交差検証の中でSMOTEを適用
from imblearn.pipeline import Pipeline as ImbPipeline
pipe = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', LogisticRegression(max_iter=1000))
])
scores = cross_val_score(pipe, X, y, cv=5, scoring='f1')
```

---

## FAQ

### Q1: ロジスティック回帰の C パラメータとは？

**A:** CはLassoの alpha の逆数（C = 1/alpha）。C が大きいほど正則化が弱く、モデルが複雑になる。C が小さいほど正則化が強く、単純なモデルになる。CVで最適値を探索するのが標準。scikit-learnのデフォルトは C=1.0。L1ペナルティ（solver='saga'）を使えばスパースな解が得られ、特徴量選択の効果もある。

### Q2: ランダムフォレストの木の数（n_estimators）はいくつが良い？

**A:** 一般にn_estimatorsを増やすと性能は単調に改善し、ある地点で飽和する（過学習しにくい）。100〜500が一般的。計算時間とのトレードオフで決定する。OOB（Out-of-Bag）スコアの推移を監視して飽和点を見極める方法もある。max_features（各分割で考慮する特徴量数）の方が性能への影響が大きいことが多い。

### Q3: XGBoostとLightGBMの違いは？

**A:** XGBoostはレベルワイズ（層ごと）の木成長、LightGBMはリーフワイズ（葉ごと）の木成長。LightGBMの方が一般に高速で、大規模データに適する。精度は同等かLightGBMがやや優位な場合が多い。カテゴリ変数の直接サポートはLightGBMが優れている。CatBoostはカテゴリ変数の扱いに特化しており、前処理なしで使える。

### Q4: 分類モデルの選び方のフローチャートは？

**A:** (1) まずロジスティック回帰でベースラインを確立、(2) 特徴量間に非線形関係がありそうならランダムフォレスト、(3) 精度を追求するならLightGBM/XGBoost、(4) 解釈性が必要なら決定木 or ロジスティック回帰 + SHAP、(5) 小規模データ（<1000件）ならSVM(RBF)も検討、(6) テキスト分類ならナイーブベイズがベースライン。最終的には複数モデルを比較し、ドメイン知識と合わせて判断する。

### Q5: ROC-AUCとPR-AUCのどちらを使うべき？

**A:** クラスが均衡している場合はROC-AUC、不均衡な場合はPR-AUC（Average Precision）が推奨。ROC-AUCは不均衡データで過度に楽観的なスコアを示す傾向がある。例えば99:1の不均衡で全て多数クラスと予測してもROC-AUCは0.5だが、PR-AUCは0.01程度になる（不均衡を反映）。

### Q6: 確率出力が信頼できるモデルはどれ？

**A:** ロジスティック回帰の確率出力は最も校正されている（calibrated）。ランダムフォレストは過度に0/1に寄る傾向、SVMのdecision_functionは確率ではない。GBMの確率出力はある程度校正されているが完璧ではない。確率の校正が重要な場合は `CalibratedClassifierCV` を使うか、Plattスケーリング/Isotonic回帰で事後的に校正する。

---

## まとめ

| 項目 | 要点 |
|---|---|
| ロジスティック回帰 | 確率を出力。解釈性が高い。ベースラインに最適 |
| SVM | マージン最大化。カーネルで非線形対応。中規模データ向け |
| 決定木 | 解釈性が最も高い。アンサンブルのベースに |
| ランダムフォレスト | バギング+ランダム特徴量選択。過学習しにくい。並列化可能 |
| 勾配ブースティング | 逐次的に残差を学習。精度が最も高いことが多い |
| k-NN | シンプルで直感的。スケーリング必須。次元の呪いに注意 |
| ナイーブベイズ | 極めて高速。テキスト分類のベースライン |
| 閾値調整 | デフォルト0.5が最適とは限らない。PR曲線で最適化 |
| クラス不均衡 | class_weight、SMOTE、Focal Loss等で対処 |

---

## 次に読むべきガイド

- [02-clustering.md](./02-clustering.md) — 教師なし学習のクラスタリング手法
- [03-dimensionality-reduction.md](./03-dimensionality-reduction.md) — 次元削減

---

## 参考文献

1. **Christopher M. Bishop** "Pattern Recognition and Machine Learning" Springer, 2006
2. **Tianqi Chen, Carlos Guestrin** "XGBoost: A Scalable Tree Boosting System" KDD 2016
3. **Guolin Ke et al.** "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" NeurIPS 2017
4. **scikit-learn** "Supervised Learning" — https://scikit-learn.org/stable/supervised_learning.html
5. **Leo Breiman** "Random Forests" Machine Learning, 2001
6. **Corinna Cortes, Vladimir Vapnik** "Support-Vector Networks" Machine Learning, 1995
7. **Nitesh V. Chawla et al.** "SMOTE: Synthetic Minority Over-sampling Technique" JAIR, 2002
8. **Scott M. Lundberg, Su-In Lee** "A Unified Approach to Interpreting Model Predictions" NeurIPS 2017

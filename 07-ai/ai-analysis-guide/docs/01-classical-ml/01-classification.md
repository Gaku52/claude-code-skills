# 分類 — ロジスティック回帰、SVM、ランダムフォレスト

> 離散値（クラスラベル）を予測する分類手法の理論・実装・選択基準を網羅的に理解する

## この章で学ぶこと

1. **ロジスティック回帰** — シグモイド関数、最尤推定、確率的分類の原理
2. **サポートベクターマシン（SVM）** — マージン最大化、カーネルトリック、ソフトマージン
3. **アンサンブル学習** — ランダムフォレスト、勾配ブースティングの仕組みと使い分け

---

## 1. ロジスティック回帰

### 決定境界の仕組み

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

### コード例1: ロジスティック回帰の詳細分析

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
for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    lr = LogisticRegression(C=C, max_iter=1000, random_state=42)
    scores = cross_val_score(lr, X_train_s, y_train, cv=5, scoring="roc_auc")
    lr.fit(X_train_s, y_train)
    n_nonzero = np.sum(np.abs(lr.coef_[0]) > 1e-4)
    print(f"C={C:6.2f}  AUC={scores.mean():.4f}±{scores.std():.4f}  "
          f"非ゼロ係数={n_nonzero}")

# 最良モデルの特徴量重要度
best_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
best_lr.fit(X_train_s, y_train)

importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": best_lr.coef_[0],
    "abs_coef": np.abs(best_lr.coef_[0])
}).sort_values("abs_coef", ascending=False).head(10)

print("\n--- 重要特徴量 Top 10 ---")
print(importance.to_string(index=False))

# テスト評価
y_pred = best_lr.predict(X_test_s)
y_prob = best_lr.predict_proba(X_test_s)[:, 1]
print(f"\nAUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=data.target_names))
```

---

## 2. サポートベクターマシン (SVM)

### マージン最大化とカーネルトリック

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

---

## 3. アンサンブル学習

### バギング vs ブースティング

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

### コード例4: 勾配ブースティングの実装

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
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
    print(f"{name:15s}  F1={scores.mean():.4f}±{scores.std():.4f}  "
          f"時間={elapsed:.2f}秒")
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
model = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
y_prob = model.predict_proba(X_test_s)[:, 1]
best_th = optimize_threshold(y_test, y_prob)

# 最適閾値で予測
y_pred_opt = (y_prob >= best_th).astype(int)
print(f"\nデフォルト閾値(0.5) F1: {f1_score(y_test, model.predict(X_test_s)):.4f}")
print(f"最適閾値({best_th:.3f}) F1: {f1_score(y_test, y_pred_opt):.4f}")
```

---

## 比較表

### 分類アルゴリズムの特性比較

| アルゴリズム | 学習速度 | 予測速度 | 解釈性 | 非線形対応 | スケーリング要否 | 欠損値対応 |
|---|---|---|---|---|---|---|
| ロジスティック回帰 | 速い | 極速 | 高い | × (特徴量変換で可) | 要 | × |
| SVM (線形) | 速い | 極速 | 中程度 | × | 要 | × |
| SVM (RBF) | 遅い | 遅い | 低い | ○ | 要 | × |
| 決定木 | 速い | 極速 | 高い | ○ | 不要 | △ |
| ランダムフォレスト | 中程度 | 中程度 | 中程度 | ○ | 不要 | △ |
| XGBoost / LightGBM | 中程度 | 速い | 低い | ○ | 不要 | ○ |
| k-NN | 不要 | 遅い | 中程度 | ○ | 要 | × |

### クラス不均衡への対処法

| 手法 | カテゴリ | 説明 | メリット | デメリット |
|---|---|---|---|---|
| class_weight="balanced" | アルゴリズム側 | 少数クラスの重みを増加 | 簡単 | 過学習リスク |
| SMOTE | オーバーサンプリング | 少数クラスの合成サンプル生成 | データ量増加 | ノイズ増加の可能性 |
| RandomUnderSampler | アンダーサンプリング | 多数クラスをランダム削減 | 高速化 | 情報喪失 |
| 閾値調整 | 後処理 | 分類閾値を最適化 | モデル変更不要 | 検証データが必要 |
| Focal Loss | 損失関数 | 簡単な例の重みを下げる | 効果的 | カスタム実装が必要 |

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

---

## FAQ

### Q1: ロジスティック回帰の C パラメータとは？

**A:** CはLassoの α の逆数（C = 1/α）。C が大きいほど正則化が弱く、モデルが複雑になる。C が小さいほど正則化が強く、単純なモデルになる。CVで最適値を探索するのが標準。scikit-learnのデフォルトは C=1.0。

### Q2: ランダムフォレストの木の数（n_estimators）はいくつが良い？

**A:** 一般にn_estimatorsを増やすと性能は単調に改善し、ある地点で飽和する（過学習しにくい）。100〜500が一般的。計算時間とのトレードオフで決定する。OOB（Out-of-Bag）スコアの推移を監視して飽和点を見極める方法もある。

### Q3: XGBoostとLightGBMの違いは？

**A:** XGBoostはレベルワイズ（層ごと）の木成長、LightGBMはリーフワイズ（葉ごと）の木成長。LightGBMの方が一般に高速で、大規模データに適する。精度は同等かLightGBMがやや優位な場合が多い。カテゴリ変数の直接サポートはLightGBMが優れている。

---

## まとめ

| 項目 | 要点 |
|---|---|
| ロジスティック回帰 | 確率を出力。解釈性が高い。ベースラインに最適 |
| SVM | マージン最大化。カーネルで非線形対応。中規模データ向け |
| ランダムフォレスト | バギング+ランダム特徴量選択。過学習しにくい。並列化可能 |
| 勾配ブースティング | 逐次的に残差を学習。精度が最も高いことが多い |
| 閾値調整 | デフォルト0.5が最適とは限らない。PR曲線で最適化 |

---

## 次に読むべきガイド

- [02-clustering.md](./02-clustering.md) — 教師なし学習のクラスタリング手法
- [03-dimensionality-reduction.md](./03-dimensionality-reduction.md) — 次元削減

---

## 参考文献

1. **Christopher M. Bishop** "Pattern Recognition and Machine Learning" Springer, 2006
2. **Tianqi Chen, Carlos Guestrin** "XGBoost: A Scalable Tree Boosting System" KDD 2016
3. **scikit-learn** "Supervised Learning" — https://scikit-learn.org/stable/supervised_learning.html

# 機械学習基礎 — 教師あり/なし、評価指標

> 機械学習の基本概念を理論と実装の両面から体系的に理解する

## この章で学ぶこと

1. **学習パラダイム** — 教師あり、教師なし、半教師あり、強化学習の原理と使い分け
2. **バイアス-バリアンストレードオフ** — 過学習・過少適合のメカニズムと対策
3. **評価指標** — 分類・回帰それぞれの評価指標と適切な選択基準

---

## 1. 教師あり学習の基礎

### 学習の仕組み

```
教師あり学習のフロー:

  訓練データ                    予測
  ┌─────────────┐           ┌─────────┐
  │ 特徴量 (X)  │           │ 新データ │
  │ ラベル (y)  │           │ (X_new) │
  └──────┬──────┘           └────┬────┘
         │                       │
         v                       v
  ┌──────┴──────┐         ┌─────┴─────┐
  │  学習       │         │  推論     │
  │  f(X) ≈ y  │────────>│  ŷ = f(X) │
  │  パラメータ │  モデル │           │
  │  最適化     │         │           │
  └─────────────┘         └───────────┘

  損失関数: L(y, ŷ) を最小化するパラメータ θ を探す
  θ* = argmin_θ Σ L(y_i, f(x_i; θ)) + λR(θ)
                                        ↑ 正則化項
```

### コード例1: 教師あり学習のワークフロー全体

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. データ準備
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target

# 2. 訓練/テスト分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 前処理
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 4. モデル学習
models = {
    "ロジスティック回帰": LogisticRegression(max_iter=1000, random_state=42),
    "ランダムフォレスト": RandomForestClassifier(n_estimators=100, random_state=42),
}

for name, model in models.items():
    # 交差検証
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="f1")
    print(f"\n{name}")
    print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # テスト評価
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    print(classification_report(y_test, y_pred, target_names=data.target_names))
```

---

## 2. バイアス-バリアンストレードオフ

### 概念図

```
予測誤差の分解:

  総誤差 = バイアス² + バリアンス + ノイズ（削減不可）

  モデル複雑度 →  低い                            高い
                   │                                │
  バイアス:        │████████████████░░░░░░░░░░░░░░░│  高→低
  バリアンス:      │░░░░░░░░░░░░░░░████████████████│  低→高
                   │                                │
  訓練誤差:        │████████░░░░░░░░░░░░░░░░░░░░░░░│  高→極低
  テスト誤差:      │████████░░░░░░░░░░░░░████████░░│  U字型
                   │         ↑                      │
                   │     最適点                     │
                   └────────────────────────────────┘

  過少適合                最適             過学習
  (Underfitting)         (Best)         (Overfitting)
```

### コード例2: 過学習の可視化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier

def plot_learning_curve(estimator, X, y, title="学習曲線"):
    """学習曲線をプロットして過学習/過少適合を診断"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy", n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.1, color="blue")
    ax.fill_between(train_sizes, test_mean - test_std,
                    test_mean + test_std, alpha=0.1, color="orange")
    ax.plot(train_sizes, train_mean, "o-", color="blue", label="訓練スコア")
    ax.plot(train_sizes, test_mean, "o-", color="orange", label="検証スコア")
    ax.set_xlabel("訓練サンプル数")
    ax.set_ylabel("精度")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/learning_curve.png", dpi=150)
    plt.close()

# 過学習する深い決定木
deep_tree = DecisionTreeClassifier(max_depth=None, random_state=42)
plot_learning_curve(deep_tree, X_train_s, y_train, "深い決定木（過学習傾向）")

# 正則化された浅い決定木
shallow_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
plot_learning_curve(shallow_tree, X_train_s, y_train, "浅い決定木（適切な複雑度）")
```

---

## 3. 交差検証

### 交差検証の種類

```
K-Fold 交差検証 (K=5):

  Fold 1: [TEST] [Train] [Train] [Train] [Train]
  Fold 2: [Train] [TEST] [Train] [Train] [Train]
  Fold 3: [Train] [Train] [TEST] [Train] [Train]
  Fold 4: [Train] [Train] [Train] [TEST] [Train]
  Fold 5: [Train] [Train] [Train] [Train] [TEST]

  最終スコア = 5つのFoldのスコアの平均 ± 標準偏差

Stratified K-Fold（クラス不均衡時）:
  各Foldでクラスの比率を維持

Time Series Split（時系列データ）:
  Fold 1: [Train] → [TEST]
  Fold 2: [Train][Train] → [TEST]
  Fold 3: [Train][Train][Train] → [TEST]
  ※ 未来のデータで訓練しない
```

### コード例3: 交差検証戦略の使い分け

```python
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit,
    cross_validate, RepeatedStratifiedKFold
)
from sklearn.ensemble import GradientBoostingClassifier

def comprehensive_cv(model, X, y, cv_strategy="stratified"):
    """包括的な交差検証を実行"""

    strategies = {
        "kfold": KFold(n_splits=5, shuffle=True, random_state=42),
        "stratified": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        "repeated": RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42),
        "timeseries": TimeSeriesSplit(n_splits=5),
    }

    cv = strategies[cv_strategy]
    scoring = ["accuracy", "f1", "precision", "recall", "roc_auc"]

    results = cross_validate(
        model, X, y, cv=cv, scoring=scoring,
        return_train_score=True, n_jobs=-1
    )

    print(f"交差検証戦略: {cv_strategy}")
    print("-" * 50)
    for metric in scoring:
        train_key = f"train_{metric}"
        test_key = f"test_{metric}"
        print(f"  {metric:12s}: "
              f"Train={results[train_key].mean():.4f} ± {results[train_key].std():.4f}  "
              f"Test={results[test_key].mean():.4f} ± {results[test_key].std():.4f}")

    return results

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
results = comprehensive_cv(model, X_train_s, y_train, "stratified")
```

---

## 4. 評価指標

### コード例4: 分類指標の詳細計算

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, roc_curve, average_precision_score
)
import matplotlib.pyplot as plt

class ClassificationEvaluator:
    """分類モデルの包括的評価"""

    def __init__(self, y_true, y_pred, y_prob=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob

    def print_metrics(self):
        """主要指標を一覧表示"""
        print("=" * 50)
        print("分類評価指標")
        print("=" * 50)
        print(f"  Accuracy:  {accuracy_score(self.y_true, self.y_pred):.4f}")
        print(f"  Precision: {precision_score(self.y_true, self.y_pred):.4f}")
        print(f"  Recall:    {recall_score(self.y_true, self.y_pred):.4f}")
        print(f"  F1-Score:  {f1_score(self.y_true, self.y_pred):.4f}")
        if self.y_prob is not None:
            print(f"  AUC-ROC:   {roc_auc_score(self.y_true, self.y_prob):.4f}")
            print(f"  AP:        {average_precision_score(self.y_true, self.y_prob):.4f}")

        cm = confusion_matrix(self.y_true, self.y_pred)
        print(f"\n  混同行列:")
        print(f"           予測=0  予測=1")
        print(f"  実際=0   {cm[0,0]:5d}   {cm[0,1]:5d}  (TN, FP)")
        print(f"  実際=1   {cm[1,0]:5d}   {cm[1,1]:5d}  (FN, TP)")

    def plot_roc_pr(self):
        """ROC曲線とPR曲線を並べて描画"""
        if self.y_prob is None:
            print("確率予測が必要です")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # ROC曲線
        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
        auc = roc_auc_score(self.y_true, self.y_prob)
        ax1.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax1.set_xlabel("偽陽性率 (FPR)")
        ax1.set_ylabel("真陽性率 (TPR)")
        ax1.set_title("ROC曲線")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # PR曲線
        prec, rec, _ = precision_recall_curve(self.y_true, self.y_prob)
        ap = average_precision_score(self.y_true, self.y_prob)
        ax2.plot(rec, prec, label=f"AP = {ap:.3f}")
        ax2.set_xlabel("再現率 (Recall)")
        ax2.set_ylabel("適合率 (Precision)")
        ax2.set_title("PR曲線")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("reports/roc_pr_curves.png", dpi=150)
        plt.close()

# 使用例
model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]

evaluator = ClassificationEvaluator(y_test, y_pred, y_prob)
evaluator.print_metrics()
evaluator.plot_roc_pr()
```

### コード例5: 回帰指標の計算

```python
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)

class RegressionEvaluator:
    """回帰モデルの包括的評価"""

    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

    def all_metrics(self) -> dict:
        """全指標を計算"""
        mse = mean_squared_error(self.y_true, self.y_pred)
        return {
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(self.y_true, self.y_pred),
            "MAPE(%)": mean_absolute_percentage_error(self.y_true, self.y_pred) * 100,
            "R²": r2_score(self.y_true, self.y_pred),
            "Adjusted R²": self._adjusted_r2(n_features=10),
        }

    def _adjusted_r2(self, n_features: int) -> float:
        """自由度調整済みR²"""
        n = len(self.y_true)
        r2 = r2_score(self.y_true, self.y_pred)
        return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    def print_metrics(self):
        metrics = self.all_metrics()
        print("=" * 40)
        print("回帰評価指標")
        print("=" * 40)
        for name, value in metrics.items():
            print(f"  {name:15s}: {value:.4f}")

# 使用例
# evaluator = RegressionEvaluator(y_test, y_pred)
# evaluator.print_metrics()
```

---

## 比較表

### 分類評価指標の使い分け

| 指標 | 数式 | 重視する場面 | クラス不均衡への耐性 |
|---|---|---|---|
| Accuracy | (TP+TN)/N | クラス均衡時の全体評価 | 弱い |
| Precision | TP/(TP+FP) | 偽陽性のコストが高い（スパム検出） | 中程度 |
| Recall | TP/(TP+FN) | 偽陰性のコストが高い（がん検診） | 中程度 |
| F1-Score | 2×P×R/(P+R) | PrecisionとRecallのバランス | 中程度 |
| AUC-ROC | ROC曲線下面積 | 閾値に依存しない総合評価 | 中程度 |
| Average Precision | PR曲線下面積 | クラス不均衡時の陽性検出能力 | 強い |
| Log Loss | -Σ y log(p) | 確率予測の精度 | 中程度 |

### 回帰評価指標の比較

| 指標 | 数式 | スケール依存 | 外れ値耐性 | 解釈性 |
|---|---|---|---|---|
| MSE | Σ(y-ŷ)²/n | あり | 弱い（二乗） | 低い |
| RMSE | √MSE | あり（元の単位） | 弱い | 高い |
| MAE | Σ|y-ŷ|/n | あり（元の単位） | 中程度 | 高い |
| MAPE | Σ|y-ŷ|/y /n | なし（%） | 弱い | 高い |
| R² | 1-SS_res/SS_tot | なし（0〜1） | 弱い | 高い |

---

## アンチパターン

### アンチパターン1: 不均衡データでのAccuracy信仰

```python
# BAD: 99%が正常、1%が不正のデータで Accuracy を使う
# "全て正常と予測" → Accuracy 99% だが不正検出能力ゼロ

# GOOD: 不均衡データでは適切な指標を選択
from sklearn.metrics import classification_report

# class_weight でクラス不均衡に対処
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# F1, Recall, PR-AUC で評価
print(classification_report(y_test, y_pred))
```

### アンチパターン2: テストデータでのハイパーパラメータ調整

```python
# BAD: テストデータでスコアを見ながら調整 → 情報リーク
for max_depth in [3, 5, 10, 20]:
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # テストで評価 → NG
    print(f"depth={max_depth}: {score:.4f}")

# GOOD: 検証データ or 交差検証で調整、テストは最後に1回だけ
from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": [3, 5, 10, 20]}
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid, cv=5, scoring="f1"
)
grid.fit(X_train, y_train)  # 訓練+検証のみ
print(f"最良パラメータ: {grid.best_params_}")
print(f"テストスコア: {grid.score(X_test, y_test):.4f}")  # 最後に1回
```

---

## FAQ

### Q1: 教師なし学習の「正解」はどう評価するのか？

**A:** 外部評価（正解ラベルがある場合）と内部評価（ラベルなし）の2種類がある。内部評価ではシルエットスコア（クラスタの分離度）、エルボー法（慣性の減少率）等を使う。ただし、最終的にはドメイン知識による定性的評価が不可欠である。

### Q2: 交差検証の K はいくつが良い？

**A:** 一般的には K=5 または K=10 が標準。データが少ない場合は K を大きく（Leave-One-Outまで）、計算コストが高い場合は K=3〜5 で十分。RepeatedKFold（繰り返し交差検証）で分散を安定させる手法もある。

### Q3: 過学習を防ぐ方法にはどんなものがある？

**A:** 主な対策: (1) 正則化（L1/L2、Dropout）、(2) 早期停止（Early Stopping）、(3) データ拡張、(4) モデルの複雑度制限（max_depth等）、(5) アンサンブル学習（Bagging、Boosting）、(6) 交差検証による適切な評価。最も効果的なのは「訓練データを増やす」こと。

---

## まとめ

| 項目 | 要点 |
|---|---|
| 教師あり学習 | 入力Xとラベルyからf(X)≈yを学習。回帰と分類に大別 |
| バイアス-バリアンス | 総誤差=バイアス²+バリアンス+ノイズ。モデル複雑度で制御 |
| 交差検証 | K-Fold/Stratified/TimeSeries。テストデータは最後に1回だけ |
| 分類指標 | 不均衡ではF1/PR-AUC。閾値に依存しない評価はROC-AUC |
| 回帰指標 | RMSE（元の単位）、R²（説明率）、MAPE（%で比較可能） |

---

## 次に読むべきガイド

- [03-python-ml-stack.md](./03-python-ml-stack.md) — Python ML開発環境の詳細
- [../01-classical-ml/00-regression.md](../01-classical-ml/00-regression.md) — 回帰モデルの実装

---

## 参考文献

1. **Trevor Hastie, Robert Tibshirani, Jerome Friedman** "The Elements of Statistical Learning" 2nd Edition, Springer, 2009
2. **scikit-learn** "Model evaluation: quantifying the quality of predictions" — https://scikit-learn.org/stable/modules/model_evaluation.html
3. **Google Developers** "Machine Learning Crash Course" — https://developers.google.com/machine-learning/crash-course

# Python MLスタック — NumPy、pandas、scikit-learn

> Python機械学習エコシステムの中核ライブラリを実践的に習得する

## この章で学ぶこと

1. **NumPy** — 高速な多次元配列演算とブロードキャスティング
2. **pandas** — 表形式データの読み込み・加工・集計の全操作
3. **scikit-learn** — 前処理→学習→評価のパイプライン構築

---

## 1. NumPy — 数値計算の基盤

### NumPyのアーキテクチャ

```
Python リスト              NumPy ndarray
┌───┬───┬───┬───┐         ┌───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │         │ 1 │ 2 │ 3 │ 4 │
└─┬─┴─┬─┴─┬─┴─┬─┘         └───┴───┴───┴───┘
  │   │   │   │              連続メモリブロック
  v   v   v   v              (C言語配列と同等)
 obj obj obj obj
 (各要素が別オブジェクト)    → ベクトル化演算で高速
 → ループが必要で低速        → BLAS/LAPACK連携
```

### コード例1: NumPy高速演算

```python
import numpy as np
import time

# --- ベクトル化 vs ループの速度比較 ---
n = 1_000_000
a = np.random.randn(n)
b = np.random.randn(n)

# BAD: Pythonループ
start = time.time()
result_loop = [a[i] + b[i] for i in range(n)]
print(f"ループ: {time.time() - start:.4f}秒")

# GOOD: ベクトル化演算
start = time.time()
result_vec = a + b
print(f"ベクトル化: {time.time() - start:.4f}秒")
# → 100倍以上高速

# --- ブロードキャスティング ---
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
row = np.array([10, 20, 30])

# 行ベクトルが自動的にブロードキャスト
result = matrix + row
# [[11, 22, 33],
#  [14, 25, 36],
#  [17, 28, 39]]

# --- 線形代数 ---
A = np.random.randn(3, 3)
b = np.random.randn(3)

# 連立方程式 Ax = b を解く
x = np.linalg.solve(A, b)
print(f"解: {x}")
print(f"検証 Ax - b ≈ 0: {np.allclose(A @ x, b)}")

# 固有値分解
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"固有値: {eigenvalues}")
```

---

## 2. pandas — データ操作の標準ツール

### コード例2: pandas基本操作とメソッドチェーン

```python
import pandas as pd
import numpy as np

# --- DataFrameの作成と基本操作 ---
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "age": [28, 35, 42, 31, 27],
    "department": ["Engineering", "Marketing", "Engineering", "Sales", "Marketing"],
    "salary": [85000, 72000, 95000, 68000, 71000],
    "join_date": pd.to_datetime(["2020-03-15", "2019-07-01", "2018-01-20",
                                  "2021-06-10", "2022-02-28"])
})

# メソッドチェーンでデータ加工
result = (
    df
    .assign(
        tenure_years=lambda x: (pd.Timestamp.now() - x["join_date"]).dt.days / 365,
        salary_rank=lambda x: x["salary"].rank(ascending=False).astype(int)
    )
    .query("age >= 30")
    .sort_values("salary", ascending=False)
    .reset_index(drop=True)
)
print(result)

# --- GroupBy 集計 ---
summary = (
    df
    .groupby("department")
    .agg(
        人数=("name", "count"),
        平均年齢=("age", "mean"),
        平均給与=("salary", "mean"),
        最高給与=("salary", "max"),
    )
    .round(0)
    .sort_values("平均給与", ascending=False)
)
print(summary)
```

### コード例3: 大規模データの効率的な読み込み

```python
import pandas as pd

# --- メモリ最適化読み込み ---
def read_optimized(filepath: str, sample_rows: int = 10000) -> pd.DataFrame:
    """メモリ効率の良いCSV読み込み"""

    # まずサンプルで型を推定
    sample = pd.read_csv(filepath, nrows=sample_rows)

    # 型の最適化マップを作成
    dtype_map = {}
    for col in sample.columns:
        col_type = sample[col].dtype
        if col_type == "int64":
            if sample[col].min() >= 0 and sample[col].max() <= 255:
                dtype_map[col] = "uint8"
            elif sample[col].min() >= -128 and sample[col].max() <= 127:
                dtype_map[col] = "int8"
            elif sample[col].min() >= -32768 and sample[col].max() <= 32767:
                dtype_map[col] = "int16"
            else:
                dtype_map[col] = "int32"
        elif col_type == "float64":
            dtype_map[col] = "float32"
        elif col_type == "object":
            if sample[col].nunique() / len(sample) < 0.5:
                dtype_map[col] = "category"

    # 最適化した型で読み込み
    df = pd.read_csv(filepath, dtype=dtype_map)

    original_mb = sample.memory_usage(deep=True).sum() / 1e6
    optimized_mb = df.head(sample_rows).memory_usage(deep=True).sum() / 1e6
    print(f"メモリ削減: {original_mb:.1f}MB → {optimized_mb:.1f}MB "
          f"({(1-optimized_mb/original_mb)*100:.0f}%削減)")

    return df
```

---

## 3. scikit-learn — MLパイプライン

### scikit-learn API設計

```
scikit-learn の一貫したAPI:

  すべての推定器 (Estimator)
  ├── fit(X, y)           # 学習
  ├── predict(X)          # 予測
  ├── score(X, y)         # 評価
  └── get_params()        # パラメータ取得

  変換器 (Transformer) は追加で:
  ├── transform(X)        # 変換
  └── fit_transform(X)    # 学習+変換

  Pipeline で連結:
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ Scaler   │──>│ PCA      │──>│ Model    │
  │(変換器)  │   │(変換器)  │   │(推定器)  │
  │fit       │   │fit       │   │fit       │
  │transform │   │transform │   │predict   │
  └──────────┘   └──────────┘   └──────────┘
```

### コード例4: scikit-learnパイプライン構築

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

# サンプルデータ
df = pd.DataFrame({
    "area": [50, 70, 90, 120, 60, np.nan, 80, 100],
    "rooms": [2, 3, 3, 4, 2, 3, np.nan, 4],
    "location": ["都心", "郊外", "都心", "都心", "郊外", "郊外", "都心", "郊外"],
    "age_years": [5, 10, 3, 1, 20, 15, 8, 12],
    "price": [5000, 4000, 7000, 9000, 3500, 3000, 6000, 4500],
})

X = df.drop(columns=["price"])
y = df["price"]

# 数値列とカテゴリ列で異なる前処理
numeric_features = ["area", "rooms", "age_years"]
categorical_features = ["location"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# 前処理 + モデルの統合パイプライン
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(random_state=42)),
])

# ハイパーパラメータ探索
param_grid = {
    "regressor__n_estimators": [50, 100, 200],
    "regressor__max_depth": [3, 5, 7],
    "regressor__learning_rate": [0.01, 0.1, 0.3],
}

grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="neg_mean_squared_error")
grid.fit(X, y)

print(f"最良パラメータ: {grid.best_params_}")
print(f"最良スコア (neg MSE): {grid.best_score_:.2f}")
```

### コード例5: カスタム変換器の作成

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class OutlierClipper(BaseEstimator, TransformerMixin):
    """IQRベースの外れ値クリッピング変換器"""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.array(X)
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_ = Q1 - self.factor * IQR
        self.upper_ = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        X = np.array(X).copy()
        X = np.clip(X, self.lower_, self.upper_)
        return X

class FeatureInteraction(BaseEstimator, TransformerMixin):
    """特徴量の交互作用を追加する変換器"""

    def __init__(self, interaction_pairs=None):
        self.interaction_pairs = interaction_pairs

    def fit(self, X, y=None):
        if self.interaction_pairs is None:
            n_features = X.shape[1]
            from itertools import combinations
            self.interaction_pairs = list(combinations(range(n_features), 2))
        return self

    def transform(self, X):
        X = np.array(X)
        interactions = []
        for i, j in self.interaction_pairs:
            interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
        return np.hstack([X] + interactions)

# パイプラインで使用
pipeline = Pipeline([
    ("clipper", OutlierClipper(factor=1.5)),
    ("interaction", FeatureInteraction()),
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor()),
])
```

### コード例6: モデルの保存と読み込み

```python
import joblib
import json
from datetime import datetime
from pathlib import Path

def save_model(pipeline, metrics: dict, output_dir: str = "models/"):
    """モデルと付随情報を保存"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{output_dir}/model_{timestamp}.joblib"
    meta_path = f"{output_dir}/model_{timestamp}_meta.json"

    # モデル本体
    joblib.dump(pipeline, model_path)

    # メタデータ
    meta = {
        "timestamp": timestamp,
        "model_type": type(pipeline.named_steps.get("regressor",
                          pipeline.named_steps.get("model"))).__name__,
        "metrics": metrics,
        "sklearn_version": __import__("sklearn").__version__,
        "python_version": __import__("platform").python_version(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"モデル保存: {model_path}")
    print(f"メタデータ: {meta_path}")
    return model_path

def load_model(model_path: str):
    """モデルの読み込みと検証"""
    pipeline = joblib.load(model_path)
    meta_path = model_path.replace(".joblib", "_meta.json")

    if Path(meta_path).exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"モデル種別: {meta['model_type']}")
        print(f"学習日時: {meta['timestamp']}")
        print(f"評価指標: {meta['metrics']}")

    return pipeline
```

---

## 比較表

### NumPy vs pandas vs Polars

| 項目 | NumPy | pandas | Polars |
|---|---|---|---|
| データ型 | 同型多次元配列 | 異型表形式 | 異型表形式 |
| 速度 | 極速（C/Fortran） | 中速 | 高速（Rust） |
| メモリ効率 | 高い | 中程度 | 高い |
| 遅延評価 | なし | なし | あり（LazyFrame） |
| API | 低レベル | 高レベル | 高レベル |
| 主な用途 | 数値計算、線形代数 | データ加工、EDA | 大規模データ処理 |
| 学習コスト | 中程度 | 低い | 中程度 |

### scikit-learn モデル選択チートシート

| データ条件 | 推奨モデル | 訓練速度 | 解釈性 | 精度 |
|---|---|---|---|---|
| 小規模・線形関係 | LinearRegression / LogisticRegression | 極速 | 高い | 中 |
| 中規模・非線形 | RandomForest | 速い | 中程度 | 高い |
| 中規模・高精度 | GradientBoosting | 中程度 | 低い | 高い |
| 大規模・高次元 | SGDClassifier | 極速 | 中程度 | 中 |
| テキスト分類 | MultinomialNB | 極速 | 中程度 | 中 |
| 少量・高精度 | SVM (RBFカーネル) | 遅い | 低い | 高い |
| 外れ値検出 | IsolationForest | 速い | 低い | 中〜高 |

---

## アンチパターン

### アンチパターン1: pandas のループ処理

```python
# BAD: iterrows で1行ずつ処理（極端に遅い）
for idx, row in df.iterrows():
    df.loc[idx, "new_col"] = row["a"] * row["b"] + row["c"]

# GOOD: ベクトル演算を使用
df["new_col"] = df["a"] * df["b"] + df["c"]

# GOOD: 複雑な条件は np.where か apply
df["category"] = np.where(df["value"] > 100, "high", "low")
```

### アンチパターン2: Pipeline を使わない前処理

```python
# BAD: 前処理とモデルが分離 → テスト時に変換忘れのリスク
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
model = RandomForestClassifier()
model.fit(X_train_s, y_train)
# テスト時に scaler.transform() を忘れる可能性大

# GOOD: Pipeline で一体化
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), RandomForestClassifier())
pipe.fit(X_train, y_train)        # 前処理+学習が一括
score = pipe.score(X_test, y_test) # 前処理+予測が一括
```

---

## FAQ

### Q1: pandasとPolarsのどちらを使うべき？

**A:** 2024年時点ではpandasが圧倒的にエコシステムが広く、scikit-learnやmatplotlibとの連携もシームレス。Polarsはデータが100万行を超える場合や速度が重要な場面で威力を発揮する。新規プロジェクトでデータサイズが大きいならPolarsを検討し、それ以外はpandasが安全な選択。

### Q2: scikit-learnのPipelineはどこまでカスタマイズできる？

**A:** BaseEstimator + TransformerMixinを継承すれば任意の変換器を作成可能。ColumnTransformerで列ごとに異なる処理を適用でき、FeatureUnionで特徴量を結合できる。NestedCVやカスタムスコアラーも組み合わせれば、ほぼ全てのMLワークフローをPipelineで表現できる。

### Q3: JupyterノートブックとPythonスクリプトの使い分けは？

**A:** 探索・可視化・レポーティングにはNotebook、本番パイプラインやテストコードにはスクリプト。Notebookで試行錯誤した後、確定したコードをsrc/以下のモジュールに移すのが理想。NotebookはGit管理しにくいため、nbstripoutでセル出力を除去してからコミットする。

---

## まとめ

| 項目 | 要点 |
|---|---|
| NumPy | ベクトル化演算で高速化。ループを避けブロードキャストを活用 |
| pandas | メソッドチェーンで可読性を高める。大規模データは型最適化 |
| scikit-learn | Pipeline + ColumnTransformer で再現性のあるワークフローを構築 |
| モデル保存 | joblibで保存。メタデータ（バージョン、指標）を併せて記録 |
| カスタム変換器 | BaseEstimator + TransformerMixin で独自の前処理をPipeline統合 |

---

## 次に読むべきガイド

- [../01-classical-ml/00-regression.md](../01-classical-ml/00-regression.md) — 回帰モデルの実装と評価
- [../01-classical-ml/01-classification.md](../01-classical-ml/01-classification.md) — 分類モデルの実装と評価

---

## 参考文献

1. **Jake VanderPlas** "Python Data Science Handbook" O'Reilly Media, 2016 — https://jakevdp.github.io/PythonDataScienceHandbook/
2. **scikit-learn Documentation** "API Reference" — https://scikit-learn.org/stable/modules/classes.html
3. **Wes McKinney** "Python for Data Analysis" 3rd Edition, O'Reilly Media, 2022

# データ前処理 — 欠損値、正規化、特徴量エンジニアリング

> 生データを機械学習モデルが消化できる形に変換する全手法を網羅的に解説する

## この章で学ぶこと

1. **欠損値処理** — 欠損のパターン分析と適切な補完戦略の選択
2. **スケーリングと正規化** — StandardScaler、MinMaxScaler、RobustScaler の使い分け
3. **特徴量エンジニアリング** — ドメイン知識を活かした特徴量設計と自動生成

---

## 1. 欠損値の分析と処理

### 欠損パターンの分類

```
欠損のメカニズム（Rubin, 1976）
┌──────────────────────────────────────────────────────┐
│                                                      │
│  MCAR (Missing Completely at Random)                 │
│  ├── 欠損は完全にランダム                            │
│  ├── 例: センサーの一時的な故障                      │
│  └── 対処: リストワイズ削除でも偏りなし              │
│                                                      │
│  MAR (Missing at Random)                             │
│  ├── 欠損は他の観測変数に依存                        │
│  ├── 例: 高齢者ほど収入を回答しない                  │
│  └── 対処: 多重代入法が有効                          │
│                                                      │
│  MNAR (Missing Not at Random)                        │
│  ├── 欠損は欠損値自身に依存                          │
│  ├── 例: 高収入者ほど収入を隠す                      │
│  └── 対処: ドメイン知識に基づくモデル化が必要        │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### コード例1: 欠損値の可視化と分析

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_missing(df: pd.DataFrame) -> pd.DataFrame:
    """欠損値の詳細分析レポートを生成"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    report = pd.DataFrame({
        "欠損数": missing,
        "欠損率(%)": missing_pct.round(2),
        "データ型": df.dtypes,
        "ユニーク数": df.nunique()
    })
    report = report[report["欠損数"] > 0].sort_values("欠損率(%)", ascending=False)
    return report

def plot_missing_heatmap(df: pd.DataFrame) -> None:
    """欠損パターンのヒートマップを描画"""
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap="viridis", ax=ax)
    ax.set_title("欠損値パターン（黄色 = 欠損）")
    plt.tight_layout()
    plt.savefig("reports/missing_heatmap.png", dpi=150)
    plt.close()

# 使用例
df = pd.DataFrame({
    "age": [25, np.nan, 30, 45, np.nan, 60],
    "income": [50000, 60000, np.nan, np.nan, 55000, np.nan],
    "city": ["東京", "大阪", "東京", np.nan, "福岡", "大阪"]
})

report = analyze_missing(df)
print(report)
```

### コード例2: 欠損値補完戦略

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class MissingValueHandler:
    """欠損値処理の統合クラス"""

    def __init__(self, strategy: str = "auto"):
        self.strategy = strategy
        self.imputers = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue

            missing_rate = df[col].isnull().mean()

            # 欠損率80%以上 → 列削除を推奨
            if missing_rate > 0.8:
                print(f"⚠ {col}: 欠損率{missing_rate:.0%} → 列削除を推奨")
                result.drop(columns=[col], inplace=True)
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                if missing_rate < 0.05:
                    # 少量欠損 → 中央値で補完
                    median_val = df[col].median()
                    result[col].fillna(median_val, inplace=True)
                    self.imputers[col] = ("median", median_val)
                else:
                    # 多めの欠損 → KNN補完
                    imputer = KNNImputer(n_neighbors=5)
                    numeric_cols = df.select_dtypes(include="number").columns
                    result[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    self.imputers[col] = ("knn", imputer)
            else:
                # カテゴリ変数 → 最頻値 or "Unknown"
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                result[col].fillna(mode_val, inplace=True)
                self.imputers[col] = ("mode", mode_val)

        return result

handler = MissingValueHandler()
df_clean = handler.fit_transform(df)
print(df_clean)
```

---

## 2. スケーリングと正規化

### スケーリング手法の処理フロー

```
生データ                   変換後
┌─────────┐
│ x=1000  │  StandardScaler     z = (x - μ) / σ
│ x=2000  │ ──────────────────> 平均0, 標準偏差1
│ x=5000  │
│ x=100   │  MinMaxScaler       z = (x - min) / (max - min)
│         │ ──────────────────> 範囲 [0, 1]
│         │
│ 外れ値  │  RobustScaler       z = (x - Q2) / (Q3 - Q1)
│ あり    │ ──────────────────> 中央値0, IQR基準
│         │
│ 正の歪み│  Log変換            z = log(1 + x)
│         │ ──────────────────> 歪度を緩和
└─────────┘
```

### コード例3: スケーリング手法の比較

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, PowerTransformer
)

def compare_scalers(data: np.ndarray) -> pd.DataFrame:
    """各スケーリング手法の結果を比較"""
    scalers = {
        "元データ": None,
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
        "MaxAbsScaler": MaxAbsScaler(),
        "PowerTransformer": PowerTransformer(method="yeo-johnson"),
    }

    results = {}
    for name, scaler in scalers.items():
        if scaler is None:
            scaled = data
        else:
            scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        results[name] = {
            "平均": np.mean(scaled).round(3),
            "標準偏差": np.std(scaled).round(3),
            "最小": np.min(scaled).round(3),
            "最大": np.max(scaled).round(3),
            "中央値": np.median(scaled).round(3),
        }

    return pd.DataFrame(results).T

# 外れ値を含むデータ
data = np.array([10, 20, 30, 25, 15, 22, 28, 1000])
print(compare_scalers(data))
```

---

## 3. カテゴリ変数のエンコーディング

### コード例4: エンコーディング手法の実装

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

class CategoryEncoder:
    """カテゴリ変数エンコーディングの統合クラス"""

    def __init__(self):
        self.encoders = {}

    def label_encode(self, series: pd.Series) -> pd.Series:
        """ラベルエンコーディング（順序なし二値 or 順序あり）"""
        le = LabelEncoder()
        encoded = le.fit_transform(series)
        self.encoders[series.name] = ("label", le)
        return pd.Series(encoded, name=series.name)

    def onehot_encode(self, df: pd.DataFrame, col: str,
                      drop_first: bool = True) -> pd.DataFrame:
        """ワンホットエンコーディング（名義変数）"""
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
        result = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        self.encoders[col] = ("onehot", None)
        return result

    def target_encode(self, df: pd.DataFrame, col: str,
                      target: str, smoothing: float = 10.0) -> pd.DataFrame:
        """ターゲットエンコーディング（高カーディナリティ向け）"""
        global_mean = df[target].mean()
        agg = df.groupby(col)[target].agg(["mean", "count"])

        # スムージング: サンプル数が少ないカテゴリを全体平均に近づける
        smooth = (agg["count"] * agg["mean"] + smoothing * global_mean) / \
                 (agg["count"] + smoothing)

        result = df.copy()
        result[f"{col}_encoded"] = result[col].map(smooth)
        self.encoders[col] = ("target", smooth.to_dict())
        return result

# 使用例
df = pd.DataFrame({
    "city": ["東京", "大阪", "東京", "福岡", "大阪", "東京"],
    "size": ["S", "M", "L", "M", "S", "L"],
    "price": [100, 80, 120, 70, 85, 130]
})

encoder = CategoryEncoder()
df_encoded = encoder.onehot_encode(df, "city")
df_encoded = encoder.target_encode(df_encoded, "size", "price")
print(df_encoded)
```

---

## 4. 特徴量エンジニアリング

### コード例5: 時系列特徴量の自動生成

```python
import pandas as pd
import numpy as np

def create_datetime_features(df: pd.DataFrame,
                              date_col: str) -> pd.DataFrame:
    """日付列から特徴量を自動生成"""
    result = df.copy()
    dt = pd.to_datetime(result[date_col])

    result[f"{date_col}_year"] = dt.dt.year
    result[f"{date_col}_month"] = dt.dt.month
    result[f"{date_col}_day"] = dt.dt.day
    result[f"{date_col}_dayofweek"] = dt.dt.dayofweek
    result[f"{date_col}_hour"] = dt.dt.hour
    result[f"{date_col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    result[f"{date_col}_quarter"] = dt.dt.quarter

    # 周期的エンコーディング（月を円形に変換）
    result[f"{date_col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    result[f"{date_col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)

    # 曜日の周期的エンコーディング
    result[f"{date_col}_dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    result[f"{date_col}_dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    return result

def create_lag_features(df: pd.DataFrame, col: str,
                        lags: list = [1, 7, 30]) -> pd.DataFrame:
    """ラグ特徴量と移動平均を生成"""
    result = df.copy()
    for lag in lags:
        result[f"{col}_lag_{lag}"] = result[col].shift(lag)
        result[f"{col}_rolling_mean_{lag}"] = result[col].rolling(lag).mean()
        result[f"{col}_rolling_std_{lag}"] = result[col].rolling(lag).std()
    return result

# 使用例
df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=100, freq="D"),
    "sales": np.random.randint(50, 200, 100)
})

df = create_datetime_features(df, "date")
df = create_lag_features(df, "sales", lags=[1, 7, 14])
print(df.head(15))
```

### コード例6: 交互作用特徴量とビニング

```python
import pandas as pd
import numpy as np
from itertools import combinations

def create_interaction_features(df: pd.DataFrame,
                                 columns: list) -> pd.DataFrame:
    """数値列の交互作用特徴量を生成"""
    result = df.copy()
    for col1, col2 in combinations(columns, 2):
        result[f"{col1}_x_{col2}"] = result[col1] * result[col2]
        # ゼロ除算対策付き比率
        result[f"{col1}_div_{col2}"] = result[col1] / (result[col2] + 1e-8)
    return result

def create_bins(df: pd.DataFrame, col: str,
                n_bins: int = 5, strategy: str = "quantile") -> pd.DataFrame:
    """ビニング（離散化）"""
    result = df.copy()
    if strategy == "quantile":
        result[f"{col}_bin"] = pd.qcut(result[col], q=n_bins,
                                        labels=False, duplicates="drop")
    elif strategy == "uniform":
        result[f"{col}_bin"] = pd.cut(result[col], bins=n_bins, labels=False)
    return result
```

---

## 比較表

### 欠損値補完手法の比較

| 手法 | 計算コスト | 精度 | 適用場面 | 注意点 |
|---|---|---|---|---|
| 削除（listwise） | 極低 | - | MCAR・欠損率<5% | データ量が減る |
| 平均値/中央値 | 低 | 低〜中 | 数値・欠損率<10% | 分散を過小評価 |
| 最頻値 | 低 | 低〜中 | カテゴリ変数 | 分布を歪める |
| KNN補完 | 中 | 中〜高 | MAR・数値 | スケーリング必須 |
| 多重代入法(MICE) | 高 | 高 | MAR・多変量 | 収束確認が必要 |
| モデルベース | 高 | 高 | 複雑な欠損パターン | 実装コスト大 |

### スケーリング手法の選択ガイド

| 手法 | 数式 | 外れ値耐性 | 適用場面 | 代表的アルゴリズム |
|---|---|---|---|---|
| StandardScaler | (x-μ)/σ | 弱い | 正規分布に近いデータ | SVM, ロジスティック回帰 |
| MinMaxScaler | (x-min)/(max-min) | 弱い | NN（特にCNN） | ニューラルネットワーク |
| RobustScaler | (x-Q2)/(Q3-Q1) | 強い | 外れ値が多いデータ | 汎用 |
| MaxAbsScaler | x/|max| | 弱い | スパースデータ | テキスト分類 |
| Log変換 | log(1+x) | 中程度 | 右に歪んだ分布 | 汎用 |

---

## アンチパターン

### アンチパターン1: テストデータでの fit

```python
# BAD: テストデータでも fit してしまう
from sklearn.preprocessing import StandardScaler

# 訓練データ
scaler_train = StandardScaler()
X_train = scaler_train.fit_transform(X_train)

# テストデータでも fit_transform ← 情報リーク!
scaler_test = StandardScaler()
X_test = scaler_test.fit_transform(X_test)

# GOOD: 訓練データの統計量でテストデータを変換
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)       # transform のみ
```

### アンチパターン2: 高カーディナリティ変数のワンホットエンコーディング

```python
# BAD: ユニーク値10,000のカテゴリをワンホット → 次元爆発
df_onehot = pd.get_dummies(df["zip_code"])  # 10,000列が追加される!

# GOOD: ターゲットエンコーディング or 埋め込み層を使用
from category_encoders import TargetEncoder
te = TargetEncoder(smoothing=10)
df["zip_code_encoded"] = te.fit_transform(df["zip_code"], df["target"])

# または頻度エンコーディング
freq = df["zip_code"].value_counts(normalize=True)
df["zip_code_freq"] = df["zip_code"].map(freq)
```

---

## FAQ

### Q1: 欠損値がある列を削除するか補完するかの判断基準は？

**A:** 一般的な基準: (1) 欠損率80%以上 → 削除を検討、(2) 欠損率5%未満 → 単純補完で十分、(3) 5〜80% → KNN/MICEなど高度な補完。ただし、ドメイン的に重要な列は欠損率が高くても補完する価値がある。欠損自体が情報を持つ場合（例：回答拒否）は「欠損フラグ列」を追加する。

### Q2: 特徴量は多いほど良いのか？

**A:** No。「次元の呪い」により、特徴量が多すぎると汎化性能が劣化する。特に訓練データが少ない場合は顕著。特徴量選択（SelectKBest、LASSO、相互情報量）で不要な特徴量を除去するか、PCAで次元削減する。目安として「サンプル数 / 特徴量数 > 10」を保つ。

### Q3: 正規化とスケーリングの違いは？

**A:** 厳密には、正規化（Normalization）は各サンプルのベクトルを単位長にする操作（L2正規化等）、スケーリング（Scaling）は各特徴量の範囲を揃える操作。ただし実務では混同されることが多い。MinMaxScalerは「正規化」と呼ばれることもあるが、正確にはスケーリングである。

---

## まとめ

| 項目 | 要点 |
|---|---|
| 欠損値処理 | まず欠損パターンを分析し、MCAR/MAR/MNARに応じて戦略を選択 |
| スケーリング | アルゴリズムに応じて選択。外れ値がある場合はRobustScaler |
| エンコーディング | 低カーディナリティ→ワンホット、高カーディナリティ→ターゲットエンコーディング |
| 特徴量設計 | ドメイン知識 + 自動生成（交互作用、時系列特徴量）を組み合わせる |
| 注意点 | テストデータへのリークを防ぐ。全変換は訓練データのみで fit |

---

## 次に読むべきガイド

- [02-ml-basics.md](./02-ml-basics.md) — 前処理済みデータで機械学習モデルを構築する方法
- [../01-classical-ml/00-regression.md](../01-classical-ml/00-regression.md) — 回帰モデルの実装

---

## 参考文献

1. **Stef van Buuren** "Flexible Imputation of Missing Data" 2nd Edition, CRC Press, 2018
2. **scikit-learn Documentation** "Preprocessing data" — https://scikit-learn.org/stable/modules/preprocessing.html
3. **Feature Engine** "Feature Engineering for Machine Learning" — https://feature-engine.trainindata.com/

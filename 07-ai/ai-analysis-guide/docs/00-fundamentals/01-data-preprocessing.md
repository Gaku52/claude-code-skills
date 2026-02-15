# データ前処理 — 欠損値、正規化、特徴量エンジニアリング

> 生データを機械学習モデルが消化できる形に変換する全手法を網羅的に解説する

## この章で学ぶこと

1. **欠損値処理** — 欠損のパターン分析と適切な補完戦略の選択
2. **スケーリングと正規化** — StandardScaler、MinMaxScaler、RobustScaler の使い分け
3. **カテゴリ変数のエンコーディング** — ワンホット、ターゲット、頻度エンコーディングの実装
4. **特徴量エンジニアリング** — ドメイン知識を活かした特徴量設計と自動生成
5. **外れ値処理** — 検出・除去・変換の実践手法
6. **データ品質管理** — バリデーション、パイプライン構築、再現性の確保

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

MCAR の検定:
  Little の MCAR テスト:
    H0: データは MCAR である
    p > 0.05 → MCAR と判断（削除可能）
    p ≤ 0.05 → MAR または MNAR（補完が望ましい）
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
        "ユニーク数": df.nunique(),
        "非欠損数": df.notnull().sum(),
    })
    report = report[report["欠損数"] > 0].sort_values("欠損率(%)", ascending=False)

    # 推奨アクションの追加
    actions = []
    for _, row in report.iterrows():
        pct = row["欠損率(%)"]
        if pct > 80:
            actions.append("列削除推奨")
        elif pct > 50:
            actions.append("モデルベース補完 or 削除")
        elif pct > 10:
            actions.append("KNN/MICE補完")
        elif pct > 5:
            actions.append("中央値/最頻値補完")
        else:
            actions.append("単純補完で十分")
    report["推奨アクション"] = actions

    return report

def plot_missing_heatmap(df: pd.DataFrame) -> None:
    """欠損パターンのヒートマップを描画"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # 左: 欠損パターンのヒートマップ
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False,
                cmap="viridis", ax=axes[0])
    axes[0].set_title("欠損値パターン（黄色 = 欠損）")

    # 右: 列ごとの欠損率バーチャート
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    missing_pct = missing_pct[missing_pct > 0]
    if len(missing_pct) > 0:
        missing_pct.plot(kind="barh", ax=axes[1], color="coral")
        axes[1].set_xlabel("欠損率 (%)")
        axes[1].set_title("列ごとの欠損率")
        axes[1].axvline(x=50, color="red", linestyle="--", alpha=0.5, label="50%ライン")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "欠損値なし", ha="center", va="center",
                     fontsize=14, transform=axes[1].transAxes)

    plt.tight_layout()
    plt.savefig("reports/missing_analysis.png", dpi=150)
    plt.close()

def analyze_missing_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """欠損値間の相関を分析（共起パターンの発見）"""
    missing_cols = df.columns[df.isnull().any()].tolist()
    if len(missing_cols) < 2:
        print("欠損値のある列が2つ未満のため相関分析不可")
        return pd.DataFrame()

    missing_indicator = df[missing_cols].isnull().astype(int)
    corr = missing_indicator.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title("欠損値の相関行列")
    plt.tight_layout()
    plt.savefig("reports/missing_correlation.png", dpi=150)
    plt.close()

    return corr

# 使用例
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    "age": np.random.normal(40, 15, n),
    "income": np.random.lognormal(10, 1, n),
    "education_years": np.random.randint(6, 20, n).astype(float),
    "satisfaction": np.random.normal(3.5, 1, n),
    "city": np.random.choice(["東京", "大阪", "福岡", "名古屋"], n),
})

# 現実的な欠損パターンの作成
# MAR: 若い人ほど収入を回答しない
mask_income = (df["age"] < 30) & (np.random.random(n) < 0.3)
df.loc[mask_income, "income"] = np.nan

# MCAR: ランダムな欠損
mask_age = np.random.random(n) < 0.05
df.loc[mask_age, "age"] = np.nan

# 連動した欠損: satisfactionが欠損するとeducation_yearsも欠損しやすい
mask_sat = np.random.random(n) < 0.1
df.loc[mask_sat, "satisfaction"] = np.nan
df.loc[mask_sat & (np.random.random(n) < 0.7), "education_years"] = np.nan

report = analyze_missing(df)
print(report)
plot_missing_heatmap(df)
```

### コード例2: 欠損値補完戦略

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

class MissingValueHandler:
    """欠損値処理の統合クラス"""

    def __init__(self, strategy: str = "auto"):
        self.strategy = strategy
        self.imputers = {}
        self.missing_flags = []

    def fit_transform(self, df: pd.DataFrame,
                      add_indicator: bool = True) -> pd.DataFrame:
        """欠損値を分析して適切な補完を実行

        Parameters
        ----------
        df : pd.DataFrame
            入力データフレーム
        add_indicator : bool
            欠損フラグ列を追加するか（デフォルト: True）
        """
        result = df.copy()

        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue

            missing_rate = df[col].isnull().mean()

            # 欠損フラグ列の追加（欠損自体が情報を持つ場合に有用）
            if add_indicator and missing_rate > 0.01:
                result[f"{col}_is_missing"] = df[col].isnull().astype(int)
                self.missing_flags.append(f"{col}_is_missing")

            # 欠損率80%以上 → 列削除を推奨
            if missing_rate > 0.8:
                print(f"WARNING: {col}: 欠損率{missing_rate:.0%} → 列削除を推奨")
                result.drop(columns=[col], inplace=True)
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                if missing_rate < 0.05:
                    # 少量欠損 → 中央値で補完
                    median_val = df[col].median()
                    result[col].fillna(median_val, inplace=True)
                    self.imputers[col] = ("median", median_val)
                elif missing_rate < 0.30:
                    # 中程度の欠損 → KNN補完
                    imputer = KNNImputer(n_neighbors=5, weights="distance")
                    numeric_cols = df.select_dtypes(include="number").columns.tolist()
                    result[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    self.imputers[col] = ("knn", imputer)
                else:
                    # 多めの欠損 → 反復的補完（MICE相当）
                    imputer = IterativeImputer(
                        estimator=RandomForestRegressor(
                            n_estimators=50, random_state=42
                        ),
                        max_iter=10, random_state=42
                    )
                    numeric_cols = df.select_dtypes(include="number").columns.tolist()
                    result[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    self.imputers[col] = ("iterative", imputer)
            else:
                # カテゴリ変数 → 最頻値 or "Unknown"
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                result[col].fillna(mode_val, inplace=True)
                self.imputers[col] = ("mode", mode_val)

        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """学習済みの補完パラメータで新しいデータを変換"""
        result = df.copy()

        for col, (strategy, param) in self.imputers.items():
            if col not in df.columns:
                continue

            if strategy == "median":
                result[col].fillna(param, inplace=True)
            elif strategy == "mode":
                result[col].fillna(param, inplace=True)
            elif strategy in ("knn", "iterative"):
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                result[numeric_cols] = param.transform(df[numeric_cols])

        # 欠損フラグ列の追加
        for flag_col in self.missing_flags:
            orig_col = flag_col.replace("_is_missing", "")
            if orig_col in df.columns:
                result[flag_col] = df[orig_col].isnull().astype(int)

        return result

    def report(self) -> None:
        """補完戦略のレポートを表示"""
        print("=" * 60)
        print("欠損値補完レポート")
        print("=" * 60)
        for col, (strategy, param) in self.imputers.items():
            if strategy == "median":
                print(f"  {col}: 中央値補完 (値={param:.2f})")
            elif strategy == "mode":
                print(f"  {col}: 最頻値補完 (値={param})")
            elif strategy == "knn":
                print(f"  {col}: KNN補完 (k=5, 距離加重)")
            elif strategy == "iterative":
                print(f"  {col}: 反復的補完 (RandomForest, max_iter=10)")
        if self.missing_flags:
            print(f"\n  欠損フラグ列: {len(self.missing_flags)}個追加")

handler = MissingValueHandler()
df_clean = handler.fit_transform(df)
handler.report()
print(f"\n補完前: {df.shape}, 補完後: {df_clean.shape}")
print(f"残りの欠損: {df_clean.isnull().sum().sum()}")
```

### コード例2b: 多重代入法（Multiple Imputation）の実装

```python
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def multiple_imputation(df, n_imputations=5, random_state=42):
    """多重代入法による補完と推論の不確実性の推定

    複数回の補完を行い、結果のばらつきから推論の不確実性を評価する
    """
    imputed_datasets = []

    for i in range(n_imputations):
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=20,
            random_state=random_state + i,
            sample_posterior=True  # 事後分布からサンプリング
        )
        numeric_cols = df.select_dtypes(include="number").columns
        imputed = df.copy()
        imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        imputed_datasets.append(imputed)

    return imputed_datasets

def pool_results(imputed_datasets, col):
    """Rubinのルールで結果を統合

    複数の補完結果を統合して、点推定値と信頼区間を計算する
    """
    m = len(imputed_datasets)
    estimates = [ds[col].values for ds in imputed_datasets]

    # 点推定: 各補完データセットの平均の平均
    Q_bar = np.mean([np.mean(est) for est in estimates])

    # Within-imputation variance（各補完内の分散の平均）
    W = np.mean([np.var(est) for est in estimates])

    # Between-imputation variance（補完間の分散）
    B = np.var([np.mean(est) for est in estimates])

    # Total variance (Rubinのルール)
    T = W + (1 + 1/m) * B

    # 信頼区間
    se = np.sqrt(T)
    ci_lower = Q_bar - 1.96 * se
    ci_upper = Q_bar + 1.96 * se

    print(f"列 '{col}' の多重代入結果:")
    print(f"  点推定値: {Q_bar:.4f}")
    print(f"  標準誤差: {se:.4f}")
    print(f"  95%信頼区間: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Within-variance: {W:.4f}")
    print(f"  Between-variance: {B:.4f}")

    return Q_bar, se, (ci_lower, ci_upper)

# 使用例
# imputed_datasets = multiple_imputation(df, n_imputations=10)
# pool_results(imputed_datasets, "income")
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
│         │
│ 極端な  │  PowerTransformer   z = ((x^λ - 1) / λ)  (Box-Cox)
│ 非正規  │ ──────────────────> 正規分布に近づける
│         │
│ ベクトル│  Normalizer         z = x / ||x||₂
│ 長統一  │ ──────────────────> L2ノルム=1（行方向）
└─────────┘

各手法の適用判断フロー:
  外れ値がある? → Yes → RobustScaler
                → No  → 分布が歪んでいる?
                          → Yes → PowerTransformer / Log変換
                          → No  → モデルが距離ベース?
                                    → Yes → StandardScaler
                                    → No  → MinMaxScaler or そのまま
```

### コード例3: スケーリング手法の比較

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, PowerTransformer, QuantileTransformer,
    Normalizer
)

def compare_scalers(data: np.ndarray, feature_name: str = "特徴量") -> pd.DataFrame:
    """各スケーリング手法の結果を比較"""
    scalers = {
        "元データ": None,
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
        "MaxAbsScaler": MaxAbsScaler(),
        "PowerTransformer\n(Yeo-Johnson)": PowerTransformer(method="yeo-johnson"),
        "QuantileTransformer\n(正規分布)": QuantileTransformer(
            output_distribution="normal", random_state=42
        ),
    }

    results = {}
    fig, axes = plt.subplots(len(scalers), 1, figsize=(12, 3 * len(scalers)))

    for idx, (name, scaler) in enumerate(scalers.items()):
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
            "歪度": pd.Series(scaled).skew().round(3),
        }

        axes[idx].hist(scaled, bins=50, edgecolor="black", alpha=0.7)
        axes[idx].set_title(f"{name}: mean={np.mean(scaled):.2f}, "
                           f"std={np.std(scaled):.2f}")
        axes[idx].axvline(x=np.mean(scaled), color="red", linestyle="--")

    plt.tight_layout()
    plt.savefig("reports/scaler_comparison.png", dpi=150)
    plt.close()

    return pd.DataFrame(results).T

# 外れ値を含む右に歪んだデータ
np.random.seed(42)
data_skewed = np.concatenate([
    np.random.lognormal(3, 1, 1000),  # 通常のデータ
    np.array([5000, 8000, 10000])     # 外れ値
])
print(compare_scalers(data_skewed, "収入"))
```

### コード例3b: スケーリングの選択を自動化

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
)

class AutoScaler:
    """データの特性に基づいてスケーリング手法を自動選択"""

    def __init__(self, outlier_threshold: float = 3.0,
                 skew_threshold: float = 1.0):
        self.outlier_threshold = outlier_threshold
        self.skew_threshold = skew_threshold
        self.scalers = {}
        self.selected_methods = {}

    def _has_outliers(self, x: np.ndarray) -> bool:
        """IQR法で外れ値を検出"""
        q1, q3 = np.percentile(x[~np.isnan(x)], [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return np.any((x < lower) | (x > upper))

    def _is_skewed(self, x: np.ndarray) -> bool:
        """歪度をチェック"""
        skewness = stats.skew(x[~np.isnan(x)])
        return abs(skewness) > self.skew_threshold

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """各列に最適なスケーリングを適用"""
        result = df.copy()
        numeric_cols = df.select_dtypes(include="number").columns

        for col in numeric_cols:
            x = df[col].dropna().values

            if len(x) == 0:
                continue

            has_outliers = self._has_outliers(x)
            is_skewed = self._is_skewed(x)

            if has_outliers and is_skewed:
                scaler = PowerTransformer(method="yeo-johnson")
                method = "PowerTransformer (外れ値+歪み対策)"
            elif has_outliers:
                scaler = RobustScaler()
                method = "RobustScaler (外れ値対策)"
            elif is_skewed:
                scaler = PowerTransformer(method="yeo-johnson")
                method = "PowerTransformer (歪み対策)"
            else:
                scaler = StandardScaler()
                method = "StandardScaler (標準)"

            result[col] = scaler.fit_transform(
                df[col].values.reshape(-1, 1)
            ).flatten()

            self.scalers[col] = scaler
            self.selected_methods[col] = method

        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """学習済みのスケーラーで変換"""
        result = df.copy()
        for col, scaler in self.scalers.items():
            if col in df.columns:
                result[col] = scaler.transform(
                    df[col].values.reshape(-1, 1)
                ).flatten()
        return result

    def report(self):
        """選択されたスケーリング手法のレポート"""
        print("=" * 60)
        print("AutoScaler 選択レポート")
        print("=" * 60)
        for col, method in self.selected_methods.items():
            print(f"  {col:25s}: {method}")

# 使用例
auto_scaler = AutoScaler()
df_scaled = auto_scaler.fit_transform(df_clean.select_dtypes(include="number"))
auto_scaler.report()
```

---

## 3. カテゴリ変数のエンコーディング

### エンコーディング手法の体系

```
カテゴリ変数のエンコーディング手法:

  ┌────────────────────────────────────────────┐
  │ カーディナリティ（ユニーク値の数）          │
  └──────────────────┬───────────────────────┘
                     │
        ┌────────────┴────────────┐
        │ 低い (< 10-15)          │ 高い (≥ 15)
        ├─────────────────────────┤─────────────────────────
        │                        │
        │ ┌────────────────┐     │ ┌─────────────────────┐
        │ │ OneHotEncoding │     │ │ TargetEncoding      │
        │ │ 安全・標準的   │     │ │ 目的変数の統計量    │
        │ └────────────────┘     │ └─────────────────────┘
        │                        │
        │ ┌────────────────┐     │ ┌─────────────────────┐
        │ │ OrdinalEncoding│     │ │ FrequencyEncoding   │
        │ │ 順序がある場合 │     │ │ 出現頻度で置換      │
        │ └────────────────┘     │ └─────────────────────┘
        │                        │
        │ ┌────────────────┐     │ ┌─────────────────────┐
        │ │ BinaryEncoding │     │ │ HashEncoding        │
        │ │ 2進数表現      │     │ │ ハッシュで固定次元  │
        │ └────────────────┘     │ └─────────────────────┘
        │                        │
        │                        │ ┌─────────────────────┐
        │                        │ │ Embedding           │
        │                        │ │ NN用の低次元表現    │
        │                        │ └─────────────────────┘
```

### コード例4: エンコーディング手法の実装

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import KFold

class CategoryEncoder:
    """カテゴリ変数エンコーディングの統合クラス"""

    def __init__(self):
        self.encoders = {}

    def label_encode(self, series: pd.Series) -> pd.Series:
        """ラベルエンコーディング（順序なし二値 or 順序あり）"""
        le = LabelEncoder()
        encoded = le.fit_transform(series.astype(str))
        self.encoders[series.name] = ("label", le)
        return pd.Series(encoded, name=series.name, index=series.index)

    def ordinal_encode(self, series: pd.Series,
                       order: list) -> pd.Series:
        """順序付きエンコーディング（明示的な順序を指定）

        Parameters
        ----------
        series : pd.Series
            エンコード対象
        order : list
            順序のリスト（例: ["low", "medium", "high"]）
        """
        mapping = {val: idx for idx, val in enumerate(order)}
        encoded = series.map(mapping)
        self.encoders[series.name] = ("ordinal", mapping)
        return encoded

    def onehot_encode(self, df: pd.DataFrame, col: str,
                      drop_first: bool = True,
                      max_categories: int = 15) -> pd.DataFrame:
        """ワンホットエンコーディング（名義変数）"""
        n_unique = df[col].nunique()
        if n_unique > max_categories:
            print(f"WARNING: {col} のカーディナリティ={n_unique}が高い。"
                  f"TargetEncoding推奨。")

        dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
        result = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        self.encoders[col] = ("onehot", None)
        return result

    def target_encode(self, df: pd.DataFrame, col: str,
                      target: str, smoothing: float = 10.0) -> pd.DataFrame:
        """ターゲットエンコーディング（高カーディナリティ向け）

        スムージングでサンプル数が少ないカテゴリの過学習を防止
        """
        global_mean = df[target].mean()
        agg = df.groupby(col)[target].agg(["mean", "count"])

        # ベイジアンスムージング
        smooth = (agg["count"] * agg["mean"] + smoothing * global_mean) / \
                 (agg["count"] + smoothing)

        result = df.copy()
        result[f"{col}_target_enc"] = result[col].map(smooth)
        # 未知カテゴリへの対処
        result[f"{col}_target_enc"].fillna(global_mean, inplace=True)
        self.encoders[col] = ("target", {
            "mapping": smooth.to_dict(),
            "global_mean": global_mean
        })
        return result

    def target_encode_cv(self, df: pd.DataFrame, col: str,
                         target: str, n_splits: int = 5) -> pd.DataFrame:
        """CVベースのターゲットエンコーディング（リーク防止版）

        交差検証の各Foldで別々にターゲットエンコーディングを行い、
        データリークを防止する。
        """
        result = df.copy()
        result[f"{col}_target_enc_cv"] = np.nan

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        global_mean = df[target].mean()

        for train_idx, val_idx in kf.split(df):
            train = df.iloc[train_idx]
            agg = train.groupby(col)[target].mean()
            result.iloc[val_idx, result.columns.get_loc(f"{col}_target_enc_cv")] = \
                df.iloc[val_idx][col].map(agg)

        result[f"{col}_target_enc_cv"].fillna(global_mean, inplace=True)
        return result

    def frequency_encode(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """頻度エンコーディング（出現頻度で置換）"""
        freq = df[col].value_counts(normalize=True)
        result = df.copy()
        result[f"{col}_freq_enc"] = result[col].map(freq)
        result[f"{col}_freq_enc"].fillna(0, inplace=True)
        self.encoders[col] = ("frequency", freq.to_dict())
        return result

    def binary_encode(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """バイナリエンコーディング（2進数表現）"""
        le = LabelEncoder()
        encoded = le.fit_transform(df[col].astype(str))
        n_bits = int(np.ceil(np.log2(len(le.classes_) + 1)))

        binary_cols = []
        for bit in range(n_bits):
            col_name = f"{col}_bin_{bit}"
            df_copy = df.copy() if bit == 0 else result
            binary_cols.append(col_name)

        result = df.copy()
        for bit in range(n_bits):
            col_name = f"{col}_bin_{bit}"
            result[col_name] = ((encoded >> bit) & 1).astype(int)

        result.drop(columns=[col], inplace=True)
        return result

# 使用例
df = pd.DataFrame({
    "city": ["東京", "大阪", "東京", "福岡", "大阪", "東京", "名古屋", "福岡"],
    "size": ["S", "M", "L", "M", "S", "L", "XL", "M"],
    "price": [100, 80, 120, 70, 85, 130, 140, 75]
})

encoder = CategoryEncoder()

# 名義変数: ワンホット
df_encoded = encoder.onehot_encode(df, "city")

# 順序変数: 順序付き
df_encoded["size_ord"] = encoder.ordinal_encode(
    df["size"], order=["S", "M", "L", "XL"]
)

# 高カーディナリティ: ターゲット
df_encoded = encoder.target_encode(df_encoded, "size", "price")

# 頻度エンコーディング
df_freq = encoder.frequency_encode(df, "city")

print("エンコード結果:")
print(df_encoded.head())
print(f"\n頻度エンコーディング:")
print(df_freq[["city", "city_freq_enc"]].head())
```

---

## 4. 外れ値の検出と処理

### 外れ値検出手法

```
外れ値検出手法の分類:

  統計的手法:
    ├── Z-Score法: |z| > 3 を外れ値とする（正規分布前提）
    ├── IQR法: Q1 - 1.5*IQR 未満、Q3 + 1.5*IQR 超を外れ値
    ├── Modified Z-Score法: MAD（中央絶対偏差）ベース（頑健）
    └── Grubbs検定: 帰無仮説検定による外れ値検出

  機械学習手法:
    ├── Isolation Forest: ランダムな分割による孤立度
    ├── Local Outlier Factor (LOF): 局所密度に基づく検出
    ├── One-Class SVM: 正常データの超平面からの距離
    └── DBSCAN: ノイズ点を外れ値として検出

  ドメイン知識:
    └── ビジネスルールに基づく閾値（年齢<0 or >150 等）
```

### コード例: 外れ値検出と処理

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

class OutlierHandler:
    """外れ値の検出と処理"""

    def __init__(self):
        self.outlier_info = {}

    def detect_iqr(self, series: pd.Series,
                    multiplier: float = 1.5) -> pd.Series:
        """IQR法による外れ値検出"""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        is_outlier = (series < lower) | (series > upper)
        n_outliers = is_outlier.sum()

        self.outlier_info[series.name] = {
            "method": "IQR",
            "lower_bound": lower,
            "upper_bound": upper,
            "n_outliers": n_outliers,
            "pct_outliers": n_outliers / len(series) * 100
        }

        print(f"  {series.name}: {n_outliers}個の外れ値 "
              f"({n_outliers/len(series)*100:.1f}%), "
              f"範囲=[{lower:.2f}, {upper:.2f}]")

        return is_outlier

    def detect_zscore(self, series: pd.Series,
                       threshold: float = 3.0) -> pd.Series:
        """Z-Score法による外れ値検出"""
        z = (series - series.mean()) / series.std()
        is_outlier = z.abs() > threshold
        return is_outlier

    def detect_modified_zscore(self, series: pd.Series,
                                threshold: float = 3.5) -> pd.Series:
        """Modified Z-Score法（MADベース、頑健）"""
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z = 0.6745 * (series - median) / (mad + 1e-10)
        is_outlier = modified_z.abs() > threshold
        return is_outlier

    def detect_isolation_forest(self, df: pd.DataFrame,
                                 contamination: float = 0.05) -> np.ndarray:
        """Isolation Forestによる多変量外れ値検出"""
        numeric_cols = df.select_dtypes(include="number").columns
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42, n_jobs=-1
        )
        labels = iso_forest.fit_predict(df[numeric_cols].fillna(0))
        is_outlier = labels == -1
        print(f"  Isolation Forest: {is_outlier.sum()}個の外れ値 "
              f"({is_outlier.sum()/len(df)*100:.1f}%)")
        return is_outlier

    def handle_outliers(self, df: pd.DataFrame, col: str,
                         method: str = "clip",
                         detection: str = "iqr") -> pd.DataFrame:
        """外れ値の処理

        Parameters
        ----------
        method : str
            "clip": 上下限にクリッピング
            "remove": 外れ値行を削除
            "nan": 外れ値をNaNに置換（後で補完）
            "winsorize": Winsorization（パーセンタイルに収める）
            "log": 対数変換（正の歪みのある場合）
        """
        result = df.copy()

        if detection == "iqr":
            is_outlier = self.detect_iqr(result[col])
        elif detection == "zscore":
            is_outlier = self.detect_zscore(result[col])
        elif detection == "modified_zscore":
            is_outlier = self.detect_modified_zscore(result[col])
        else:
            raise ValueError(f"未知の検出方法: {detection}")

        if method == "clip":
            info = self.outlier_info.get(col, {})
            lower = info.get("lower_bound", result[col].quantile(0.01))
            upper = info.get("upper_bound", result[col].quantile(0.99))
            result[col] = result[col].clip(lower=lower, upper=upper)
        elif method == "remove":
            result = result[~is_outlier]
        elif method == "nan":
            result.loc[is_outlier, col] = np.nan
        elif method == "winsorize":
            lower = result[col].quantile(0.01)
            upper = result[col].quantile(0.99)
            result[col] = result[col].clip(lower=lower, upper=upper)
        elif method == "log":
            result[col] = np.log1p(result[col].clip(lower=0))

        return result

    def visualize_outliers(self, df: pd.DataFrame,
                           columns: list = None) -> None:
        """外れ値の可視化"""
        if columns is None:
            columns = df.select_dtypes(include="number").columns.tolist()

        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.ravel()

        for idx, col in enumerate(columns):
            if idx >= len(axes):
                break
            axes[idx].boxplot(df[col].dropna(), vert=True)
            axes[idx].set_title(f"{col}")
            axes[idx].grid(True, alpha=0.3)

        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig("reports/outlier_boxplots.png", dpi=150)
        plt.close()

# 使用例
outlier_handler = OutlierHandler()
# df_clean = outlier_handler.handle_outliers(df, "income", method="clip")
# outlier_handler.visualize_outliers(df, ["age", "income", "satisfaction"])
```

---

## 5. 特徴量エンジニアリング

### 特徴量エンジニアリングの体系

```
特徴量エンジニアリングの分類:

  1. 基本変換
     ├── 数学的変換: log, sqrt, 二乗, 逆数
     ├── ビニング: 連続値の離散化
     └── クリッピング: 範囲の制限

  2. 集約特徴量
     ├── 統計量: 平均, 中央値, 最大, 最小, 標準偏差
     ├── カウント: 出現回数, 一意値数
     └── 比率: 対全体比率, 対グループ比率

  3. 時系列特徴量
     ├── ラグ特徴量: 過去N期間の値
     ├── 移動統計量: 移動平均, 移動標準偏差
     ├── 差分特徴量: 前期比, 前年同期比
     └── 周期特徴量: 曜日, 月, 季節のsin/cos変換

  4. テキスト特徴量
     ├── 文字数, 単語数, 文数
     ├── TF-IDF
     └── 埋め込み (Word2Vec, BERT)

  5. 交互作用特徴量
     ├── 乗算: A × B
     ├── 除算: A / B
     └── 多項式: A², A × B, B²

  6. ドメイン固有特徴量
     └── ビジネスKPI, 医療指標, 金融指標等
```

### コード例5: 時系列特徴量の自動生成

```python
import pandas as pd
import numpy as np

def create_datetime_features(df: pd.DataFrame,
                              date_col: str) -> pd.DataFrame:
    """日付列から特徴量を自動生成"""
    result = df.copy()
    dt = pd.to_datetime(result[date_col])

    # 基本的な時間特徴量
    result[f"{date_col}_year"] = dt.dt.year
    result[f"{date_col}_month"] = dt.dt.month
    result[f"{date_col}_day"] = dt.dt.day
    result[f"{date_col}_dayofweek"] = dt.dt.dayofweek
    result[f"{date_col}_hour"] = dt.dt.hour
    result[f"{date_col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    result[f"{date_col}_quarter"] = dt.dt.quarter
    result[f"{date_col}_dayofyear"] = dt.dt.dayofyear
    result[f"{date_col}_weekofyear"] = dt.dt.isocalendar().week.astype(int)

    # 周期的エンコーディング（月を円形に変換）
    result[f"{date_col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    result[f"{date_col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)

    # 曜日の周期的エンコーディング
    result[f"{date_col}_dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    result[f"{date_col}_dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    # 時間帯の周期的エンコーディング
    if dt.dt.hour.max() > 0:
        result[f"{date_col}_hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        result[f"{date_col}_hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)

    # 月初・月末フラグ
    result[f"{date_col}_is_month_start"] = dt.dt.is_month_start.astype(int)
    result[f"{date_col}_is_month_end"] = dt.dt.is_month_end.astype(int)

    # 祝日フラグ（簡易版、実務ではjpholidayパッケージ等を使用）
    result[f"{date_col}_is_holiday"] = 0  # 拡張可能

    return result

def create_lag_features(df: pd.DataFrame, col: str,
                        lags: list = [1, 7, 30],
                        group_col: str = None) -> pd.DataFrame:
    """ラグ特徴量と移動平均を生成

    Parameters
    ----------
    group_col : str, optional
        グループ単位でラグを計算する場合の列名
    """
    result = df.copy()

    for lag in lags:
        if group_col:
            result[f"{col}_lag_{lag}"] = result.groupby(group_col)[col].shift(lag)
            result[f"{col}_rolling_mean_{lag}"] = result.groupby(group_col)[col] \
                .transform(lambda x: x.rolling(lag, min_periods=1).mean())
            result[f"{col}_rolling_std_{lag}"] = result.groupby(group_col)[col] \
                .transform(lambda x: x.rolling(lag, min_periods=1).std())
        else:
            result[f"{col}_lag_{lag}"] = result[col].shift(lag)
            result[f"{col}_rolling_mean_{lag}"] = result[col].rolling(
                lag, min_periods=1
            ).mean()
            result[f"{col}_rolling_std_{lag}"] = result[col].rolling(
                lag, min_periods=1
            ).std()

        # 変化率
        result[f"{col}_pct_change_{lag}"] = result[col].pct_change(periods=lag)

    # EWMA（指数加重移動平均）
    for span in [7, 14, 30]:
        result[f"{col}_ewma_{span}"] = result[col].ewm(span=span).mean()

    return result

def create_diff_features(df: pd.DataFrame, col: str,
                          periods: list = [1, 7]) -> pd.DataFrame:
    """差分特徴量を生成"""
    result = df.copy()
    for period in periods:
        result[f"{col}_diff_{period}"] = result[col].diff(period)
        result[f"{col}_diff_pct_{period}"] = result[col].pct_change(period)
    return result

# 使用例
df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=365, freq="D"),
    "sales": np.random.randint(50, 200, 365) + \
             (np.sin(np.arange(365) * 2 * np.pi / 365) * 30).astype(int),
    "store_id": np.random.choice(["A", "B", "C"], 365),
})

df = create_datetime_features(df, "date")
df = create_lag_features(df, "sales", lags=[1, 7, 14, 28])
df = create_diff_features(df, "sales", periods=[1, 7])
print(f"特徴量数: {df.shape[1]}")
print(df.head(30))
```

### コード例6: 交互作用特徴量とビニング

```python
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

def create_interaction_features(df: pd.DataFrame,
                                 columns: list,
                                 operations: list = None) -> pd.DataFrame:
    """数値列の交互作用特徴量を生成

    Parameters
    ----------
    operations : list, optional
        実行する演算のリスト。デフォルトは ["multiply", "divide"]
        選択肢: "multiply", "divide", "add", "subtract"
    """
    if operations is None:
        operations = ["multiply", "divide"]

    result = df.copy()
    for col1, col2 in combinations(columns, 2):
        if "multiply" in operations:
            result[f"{col1}_x_{col2}"] = result[col1] * result[col2]
        if "divide" in operations:
            result[f"{col1}_div_{col2}"] = result[col1] / (result[col2] + 1e-8)
            result[f"{col2}_div_{col1}"] = result[col2] / (result[col1] + 1e-8)
        if "add" in operations:
            result[f"{col1}_plus_{col2}"] = result[col1] + result[col2]
        if "subtract" in operations:
            result[f"{col1}_minus_{col2}"] = result[col1] - result[col2]

    return result

def create_polynomial_features(df: pd.DataFrame, columns: list,
                                degree: int = 2) -> pd.DataFrame:
    """多項式特徴量の生成"""
    poly = PolynomialFeatures(degree=degree, include_bias=False,
                               interaction_only=False)
    poly_features = poly.fit_transform(df[columns])
    poly_names = poly.get_feature_names_out(columns)

    result = df.copy()
    for i, name in enumerate(poly_names):
        if name not in columns:  # 元の特徴量は除外
            result[f"poly_{name}"] = poly_features[:, i]

    return result

def create_bins(df: pd.DataFrame, col: str,
                n_bins: int = 5, strategy: str = "quantile",
                labels: list = None) -> pd.DataFrame:
    """ビニング（離散化）

    Parameters
    ----------
    strategy : str
        "quantile": 等頻度ビニング（各ビンのサンプル数が均等）
        "uniform": 等幅ビニング（各ビンの幅が均等）
        "kmeans": K-meansクラスタリングによるビニング
    """
    result = df.copy()

    if strategy == "quantile":
        result[f"{col}_bin"] = pd.qcut(
            result[col], q=n_bins, labels=labels or False, duplicates="drop"
        )
    elif strategy == "uniform":
        result[f"{col}_bin"] = pd.cut(
            result[col], bins=n_bins, labels=labels or False
        )
    elif strategy == "kmeans":
        from sklearn.preprocessing import KBinsDiscretizer
        kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="kmeans")
        result[f"{col}_bin"] = kbd.fit_transform(
            result[col].values.reshape(-1, 1)
        ).astype(int)

    return result

def create_aggregation_features(df: pd.DataFrame,
                                 group_col: str,
                                 agg_col: str) -> pd.DataFrame:
    """グループ集約特徴量の生成"""
    result = df.copy()

    agg = df.groupby(group_col)[agg_col].agg([
        "mean", "std", "min", "max", "median", "count"
    ])
    agg.columns = [f"{agg_col}_by_{group_col}_{stat}" for stat in agg.columns]

    result = result.merge(agg, left_on=group_col, right_index=True, how="left")

    # 偏差特徴量: 個体値 - グループ平均
    mean_col = f"{agg_col}_by_{group_col}_mean"
    result[f"{agg_col}_dev_from_{group_col}_mean"] = result[agg_col] - result[mean_col]

    # 比率特徴量: 個体値 / グループ平均
    result[f"{agg_col}_ratio_to_{group_col}_mean"] = \
        result[agg_col] / (result[mean_col] + 1e-8)

    return result
```

### コード例7: テキスト特徴量の生成

```python
import pandas as pd
import numpy as np
import re

def create_text_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """テキスト列から基本的な統計特徴量を生成"""
    result = df.copy()
    text = df[text_col].fillna("")

    # 基本統計
    result[f"{text_col}_length"] = text.str.len()
    result[f"{text_col}_word_count"] = text.str.split().str.len().fillna(0)
    result[f"{text_col}_sentence_count"] = text.str.count(r'[。.!?]') + 1
    result[f"{text_col}_avg_word_length"] = \
        text.apply(lambda x: np.mean([len(w) for w in x.split()]) if x else 0)

    # 特殊文字の統計
    result[f"{text_col}_n_digits"] = text.str.count(r'\d')
    result[f"{text_col}_n_uppercase"] = text.str.count(r'[A-Z]')
    result[f"{text_col}_n_special"] = text.str.count(r'[!@#$%^&*()]')
    result[f"{text_col}_n_urls"] = text.str.count(r'https?://\S+')
    result[f"{text_col}_n_emails"] = text.str.count(r'\S+@\S+\.\S+')

    # 感嘆符・疑問符の数（感情分析の手がかり）
    result[f"{text_col}_n_exclamation"] = text.str.count("!")
    result[f"{text_col}_n_question"] = text.str.count(r'\?')

    # ユニーク単語比率（語彙の豊かさ）
    result[f"{text_col}_unique_word_ratio"] = text.apply(
        lambda x: len(set(x.split())) / (len(x.split()) + 1e-8) if x else 0
    )

    return result
```

---

## 6. 特徴量選択

### 特徴量選択手法

```
特徴量選択の3つのアプローチ:

  1. フィルター法（統計的テスト）
     ├── 分散フィルター: 分散が低い特徴量を除去
     ├── 相関フィルター: 目的変数との相関が低い特徴量を除去
     ├── 相互情報量: 非線形な関係も捉える
     └── カイ二乗検定: カテゴリ変数の独立性検定
     → 高速、モデル非依存

  2. ラッパー法（モデルベース）
     ├── 前進選択法: 1つずつ特徴量を追加
     ├── 後退消去法: 1つずつ特徴量を削除
     └── 再帰的特徴量除去 (RFE): モデルの重要度で逐次削除
     → 精度高、計算コスト大

  3. 埋め込み法（学習過程で選択）
     ├── L1正則化 (Lasso): 不要な係数を0にする
     ├── 木ベースの重要度: RandomForest, XGBoost
     └── Permutation Importance
     → 精度とコストのバランスが良い
```

### コード例8: 特徴量選択の実装

```python
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif,
    mutual_info_classif, RFE
)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

class FeatureSelector:
    """特徴量選択の統合クラス"""

    def __init__(self, X, y, feature_names=None):
        self.X = X
        self.y = y
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.scores = {}

    def variance_filter(self, threshold: float = 0.01) -> list:
        """分散フィルター: 分散が閾値以下の特徴量を除去"""
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(self.X)
        mask = selector.get_support()
        removed = [f for f, m in zip(self.feature_names, mask) if not m]
        print(f"分散フィルター: {len(removed)}個除去 (閾値={threshold})")
        return [f for f, m in zip(self.feature_names, mask) if m]

    def correlation_filter(self, threshold: float = 0.95) -> list:
        """高相関フィルター: 相関が高いペアの一方を除去"""
        corr_matrix = pd.DataFrame(self.X, columns=self.feature_names).corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        print(f"相関フィルター: {len(to_drop)}個除去 (閾値={threshold})")
        return [f for f in self.feature_names if f not in to_drop]

    def statistical_test(self, k: int = 10,
                          method: str = "f_classif") -> list:
        """統計的テストによる選択"""
        if method == "f_classif":
            selector = SelectKBest(f_classif, k=k)
        elif method == "mutual_info":
            selector = SelectKBest(mutual_info_classif, k=k)

        selector.fit(self.X, self.y)
        mask = selector.get_support()
        self.scores[method] = selector.scores_

        selected = [f for f, m in zip(self.feature_names, mask) if m]
        print(f"{method}: 上位{k}個を選択")
        return selected

    def rfe_selection(self, n_features: int = 10) -> list:
        """再帰的特徴量除去"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe = RFE(model, n_features_to_select=n_features, step=1)
        rfe.fit(self.X, self.y)
        mask = rfe.support_

        selected = [f for f, m in zip(self.feature_names, mask) if m]
        print(f"RFE: {n_features}個を選択")
        return selected

    def importance_based(self, threshold: float = 0.01) -> list:
        """モデルの重要度に基づく選択"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(self.X, self.y)
        importances = model.feature_importances_

        mask = importances > threshold
        selected = [f for f, m in zip(self.feature_names, mask) if m]
        print(f"重要度ベース: {len(selected)}個を選択 (閾値={threshold})")

        # 可視化
        sorted_idx = np.argsort(importances)[::-1][:20]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(sorted_idx)),
                importances[sorted_idx], align="center")
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([self.feature_names[i] for i in sorted_idx])
        ax.set_xlabel("重要度")
        ax.set_title("特徴量重要度 (Top 20)")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig("reports/feature_importance_selection.png", dpi=150)
        plt.close()

        return selected

# 使用例
# selector = FeatureSelector(X_train, y_train, feature_names)
# selected = selector.importance_based(threshold=0.01)
```

---

## 7. データ品質管理パイプライン

### コード例9: 統合的な前処理パイプライン

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder, PowerTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
import joblib

class DataPreprocessingPipeline:
    """実務向けデータ前処理パイプライン

    使い方:
        1. fit(X_train, y_train) で学習
        2. transform(X_test) で変換
        3. save/load でパイプラインの永続化
    """

    def __init__(self, numeric_features, categorical_features,
                 ordinal_features=None, ordinal_categories=None,
                 scaler_type="standard"):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.ordinal_features = ordinal_features or []
        self.ordinal_categories = ordinal_categories or []
        self.scaler_type = scaler_type
        self.pipeline = None
        self._build_pipeline()

    def _get_scaler(self):
        """スケーラーの取得"""
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "power": PowerTransformer(method="yeo-johnson"),
        }
        return scalers.get(self.scaler_type, StandardScaler())

    def _build_pipeline(self):
        """パイプラインの構築"""
        # 数値特徴量の処理
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", self._get_scaler()),
        ])

        # カテゴリ特徴量の処理
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                drop="first", sparse_output=False,
                handle_unknown="ignore", min_frequency=0.01
            )),
        ])

        transformers = [
            ("num", numeric_transformer, self.numeric_features),
            ("cat", categorical_transformer, self.categorical_features),
        ]

        # 順序変数の処理（オプション）
        if self.ordinal_features:
            ordinal_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(
                    categories=self.ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                )),
            ])
            transformers.append(
                ("ord", ordinal_transformer, self.ordinal_features)
            )

        self.pipeline = Pipeline(steps=[
            ("preprocessor", ColumnTransformer(
                transformers=transformers,
                remainder="drop"
            )),
            ("variance_filter", VarianceThreshold(threshold=0.0)),
        ])

    def fit(self, X, y=None):
        """パイプラインの学習"""
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        """データの変換"""
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        """学習と変換を同時に実行"""
        return self.pipeline.fit_transform(X, y)

    def get_feature_names(self):
        """変換後の特徴量名を取得"""
        preprocessor = self.pipeline.named_steps["preprocessor"]
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                names.extend(cols)
            elif name == "cat":
                encoder = trans.named_steps["encoder"]
                if hasattr(encoder, "get_feature_names_out"):
                    names.extend(encoder.get_feature_names_out(cols))
                else:
                    names.extend(cols)
            elif name == "ord":
                names.extend(cols)
        return names

    def save(self, path: str):
        """パイプラインの保存"""
        joblib.dump(self.pipeline, path)
        print(f"パイプラインを保存: {path}")

    def load(self, path: str):
        """パイプラインの読込"""
        self.pipeline = joblib.load(path)
        print(f"パイプラインを読込: {path}")

# 使用例
pipeline = DataPreprocessingPipeline(
    numeric_features=["age", "income", "satisfaction"],
    categorical_features=["city"],
    ordinal_features=["education"],
    ordinal_categories=[["high_school", "bachelor", "master", "phd"]],
    scaler_type="robust"
)

# X_train_processed = pipeline.fit_transform(X_train)
# X_test_processed = pipeline.transform(X_test)
# pipeline.save("models/preprocessing_pipeline.joblib")
```

### コード例10: データバリデーションチェック

```python
import pandas as pd
import numpy as np

class DataValidator:
    """データ品質のバリデーションチェック"""

    def __init__(self):
        self.rules = []
        self.results = []

    def add_rule(self, name: str, check_fn, severity: str = "error"):
        """バリデーションルールの追加

        Parameters
        ----------
        severity : str
            "error": 修正必須
            "warning": 注意
            "info": 情報
        """
        self.rules.append({
            "name": name, "check_fn": check_fn, "severity": severity
        })

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """全ルールでバリデーション実行"""
        self.results = []

        for rule in self.rules:
            passed, message = rule["check_fn"](df)
            self.results.append({
                "ルール": rule["name"],
                "重要度": rule["severity"],
                "結果": "PASS" if passed else "FAIL",
                "詳細": message
            })

        results_df = pd.DataFrame(self.results)
        n_fail = (results_df["結果"] == "FAIL").sum()
        n_error = ((results_df["結果"] == "FAIL") &
                   (results_df["重要度"] == "error")).sum()

        print("=" * 60)
        print(f"バリデーション結果: {len(self.results)}ルール中 "
              f"{n_fail}個FAIL (うちエラー{n_error}個)")
        print("=" * 60)
        for _, row in results_df.iterrows():
            status = "PASS" if row["結果"] == "PASS" else "FAIL"
            print(f"  [{row['重要度']:7s}] [{status}] {row['ルール']}: {row['詳細']}")

        return results_df

# 使用例
validator = DataValidator()

# ルールの定義
validator.add_rule(
    "欠損率チェック",
    lambda df: (
        df.isnull().mean().max() < 0.5,
        f"最大欠損率: {df.isnull().mean().max()*100:.1f}%"
    ),
    severity="warning"
)

validator.add_rule(
    "重複行チェック",
    lambda df: (
        df.duplicated().sum() == 0,
        f"重複行: {df.duplicated().sum()}行"
    ),
    severity="warning"
)

validator.add_rule(
    "データ型チェック",
    lambda df: (
        df.select_dtypes(include="object").shape[1] < df.shape[1],
        f"object型: {df.select_dtypes(include='object').shape[1]}列"
    ),
    severity="info"
)

validator.add_rule(
    "無限値チェック",
    lambda df: (
        not np.isinf(df.select_dtypes(include="number")).any().any(),
        f"無限値を含む列: {df.select_dtypes(include='number').columns[np.isinf(df.select_dtypes(include='number')).any()].tolist()}"
    ),
    severity="error"
)

validator.add_rule(
    "サンプル数チェック",
    lambda df: (
        len(df) >= 100,
        f"サンプル数: {len(df)}"
    ),
    severity="error"
)

# results = validator.validate(df)
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
| 定数補完（-999等） | 極低 | 低 | 木ベースモデル限定 | 距離ベースモデルに悪影響 |

### スケーリング手法の選択ガイド

| 手法 | 数式 | 外れ値耐性 | 適用場面 | 代表的アルゴリズム |
|---|---|---|---|---|
| StandardScaler | (x-μ)/σ | 弱い | 正規分布に近いデータ | SVM, ロジスティック回帰, PCA |
| MinMaxScaler | (x-min)/(max-min) | 弱い | 固定範囲が必要な場合 | ニューラルネットワーク |
| RobustScaler | (x-Q2)/(Q3-Q1) | 強い | 外れ値が多いデータ | 汎用 |
| MaxAbsScaler | x/|max| | 弱い | スパースデータ | テキスト分類, スパースSVM |
| PowerTransformer | Yeo-Johnson/Box-Cox | 中程度 | 歪んだ分布 | 汎用 |
| QuantileTransformer | 分位数→正規/一様 | 強い | 任意の分布 | 汎用 |
| Normalizer | x/||x|| | - | サンプル単位の正規化 | テキスト、TF-IDF |
| Log変換 | log(1+x) | 中程度 | 右に歪んだ分布 | 汎用 |

### エンコーディング手法の比較

| 手法 | 次元増加 | 順序保持 | カーディナリティ | リーク危険性 | 適用場面 |
|---|---|---|---|---|---|
| OneHot | 大 | No | 低(≤15) | なし | 名義変数 |
| Ordinal | なし | Yes | 任意 | なし | 順序変数 |
| Label | なし | No | 任意 | なし | 木ベースモデル |
| Target | なし | No | 高 | あり(CV必要) | 高カーディナリティ |
| Frequency | なし | No | 高 | なし | 出現頻度が意味を持つ場合 |
| Binary | log₂(k) | No | 中〜高 | なし | 次元を抑えたい場合 |
| Hash | 固定 | No | 極高 | なし | テキスト、Web特徴量 |
| Embedding | 固定 | No | 極高 | なし | NN、大規模データ |

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

### アンチパターン3: 全体での分割前に前処理

```python
# BAD: 分割前にスケーリング → テストデータの情報がリーク
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X)  # 全データでfit
X_train, X_test = train_test_split(X_all_scaled, ...)

# GOOD: 分割後にパイプラインで処理
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(X, ...)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SomeModel()),
])
pipeline.fit(X_train, y_train)  # パイプライン内でfitされる
```

### アンチパターン4: ターゲットエンコーディングのリーク

```python
# BAD: 訓練データ全体でターゲットエンコーディング → リーク
mean_by_city = df.groupby("city")["target"].mean()
df["city_target_enc"] = df["city"].map(mean_by_city)

# GOOD: CVベースのターゲットエンコーディング
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
df["city_target_enc_cv"] = np.nan

for train_idx, val_idx in kf.split(df):
    mean_by_city = df.iloc[train_idx].groupby("city")["target"].mean()
    df.loc[df.index[val_idx], "city_target_enc_cv"] = \
        df.iloc[val_idx]["city"].map(mean_by_city)

# 欠損値はグローバル平均で補完
df["city_target_enc_cv"].fillna(df["target"].mean(), inplace=True)
```

### アンチパターン5: 特徴量の多重共線性の無視

```python
# BAD: 高相関の特徴量をそのまま使用 → 線形モデルが不安定に
# 例: "身長(cm)" と "身長(inch)" の両方を含める

# GOOD: VIF（分散膨張因子）で多重共線性を検出・除去
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_multicollinearity(X, feature_names, threshold=10):
    """VIFで多重共線性をチェック"""
    vif_data = pd.DataFrame()
    vif_data["feature"] = feature_names
    vif_data["VIF"] = [
        variance_inflation_factor(X, i) for i in range(X.shape[1])
    ]
    vif_data = vif_data.sort_values("VIF", ascending=False)

    high_vif = vif_data[vif_data["VIF"] > threshold]
    if len(high_vif) > 0:
        print(f"WARNING: VIF > {threshold} の特徴量:")
        for _, row in high_vif.iterrows():
            print(f"  {row['feature']}: VIF={row['VIF']:.1f}")

    return vif_data
```

---

## トラブルシューティング

| 問題 | 症状 | 対処法 |
|---|---|---|
| テストスコアが異常に高い | CV=0.99超え | データリークを疑う。前処理のfit順序を確認 |
| 欠損値補完後にモデルが悪化 | 補完前より精度低下 | 補完手法を変更。欠損フラグ列を追加 |
| ワンホット後にメモリ不足 | MemoryError | sparse=True指定 or TargetEncoding |
| 特徴量数が多すぎる | 訓練が遅い | 特徴量選択を実施。PCAで次元削減 |
| 外れ値がモデルを歪める | 訓練誤差は低いがテスト悪い | RobustScaler or 外れ値のクリッピング |
| カテゴリ値が本番で未知 | KeyError or NaN | handle_unknown="ignore" or デフォルト値 |
| スケーリングしたのに精度が変わらない | 木ベースモデル使用 | 木ベースはスケーリング不要 |
| MICE補完が収束しない | 警告メッセージ | max_iterを増加 or 特徴量を減らす |

---

## FAQ

### Q1: 欠損値がある列を削除するか補完するかの判断基準は？

**A:** 一般的な基準: (1) 欠損率80%以上 → 削除を検討、(2) 欠損率5%未満 → 単純補完で十分、(3) 5〜80% → KNN/MICEなど高度な補完。ただし、ドメイン的に重要な列は欠損率が高くても補完する価値がある。欠損自体が情報を持つ場合（例：回答拒否）は「欠損フラグ列」を追加する。

### Q2: 特徴量は多いほど良いのか？

**A:** No。「次元の呪い」により、特徴量が多すぎると汎化性能が劣化する。特に訓練データが少ない場合は顕著。特徴量選択（SelectKBest、LASSO、相互情報量）で不要な特徴量を除去するか、PCAで次元削減する。目安として「サンプル数 / 特徴量数 > 10」を保つ。

### Q3: 正規化とスケーリングの違いは？

**A:** 厳密には、正規化（Normalization）は各サンプルのベクトルを単位長にする操作（L2正規化等）、スケーリング（Scaling）は各特徴量の範囲を揃える操作。ただし実務では混同されることが多い。MinMaxScalerは「正規化」と呼ばれることもあるが、正確にはスケーリングである。

### Q4: 木ベースモデルでもスケーリングは必要か？

**A:** 基本的に不要。決定木、ランダムフォレスト、XGBoost、LightGBMなどの木ベースモデルは分割点の探索にスケールの影響を受けない。ただし、(1) 正則化を使う場合、(2) PCAと組み合わせる場合、(3) KNNや線形モデルとのアンサンブルの場合はスケーリングが必要。

### Q5: ターゲットエンコーディングでリークを防ぐには？

**A:** (1) CVベースのターゲットエンコーディングを使う（各Foldで別々に計算）、(2) スムージングパラメータを適切に設定する（サンプル数が少ないカテゴリの過学習防止）、(3) テストデータのエンコーディングは訓練データの統計量のみで行う。

### Q6: 前処理の順序はどうすべきか？

**A:** 推奨順序: (1) 型変換（日付のパース等）、(2) 外れ値処理（必要な場合）、(3) 欠損値補完、(4) カテゴリエンコーディング、(5) 特徴量エンジニアリング、(6) スケーリング、(7) 特徴量選択。この順序をパイプラインに含めることで再現性を確保する。

---

## まとめ

| 項目 | 要点 |
|---|---|
| 欠損値処理 | まず欠損パターンを分析し、MCAR/MAR/MNARに応じて戦略を選択 |
| スケーリング | アルゴリズムに応じて選択。外れ値がある場合はRobustScaler |
| エンコーディング | 低カーディナリティ→ワンホット、高カーディナリティ→ターゲットエンコーディング |
| 外れ値処理 | IQR法 or Isolation Forestで検出、clip/remove/nanで処理 |
| 特徴量設計 | ドメイン知識 + 自動生成（交互作用、時系列特徴量）を組み合わせる |
| 特徴量選択 | フィルター法→ラッパー法→埋め込み法の順で計算コストが増加 |
| パイプライン | 全変換をPipelineに含め、fit/transform/saveで一貫管理 |
| データリーク防止 | テストデータへのリークを防ぐ。全変換は訓練データのみで fit |

---

## 次に読むべきガイド

- [02-ml-basics.md](./02-ml-basics.md) — 前処理済みデータで機械学習モデルを構築する方法
- [../01-classical-ml/00-regression.md](../01-classical-ml/00-regression.md) — 回帰モデルの実装

---

## 参考文献

1. **Stef van Buuren** "Flexible Imputation of Missing Data" 2nd Edition, CRC Press, 2018
2. **scikit-learn Documentation** "Preprocessing data" — https://scikit-learn.org/stable/modules/preprocessing.html
3. **Feature Engine** "Feature Engineering for Machine Learning" — https://feature-engine.trainindata.com/
4. **Alice Zheng, Amanda Casari** "Feature Engineering for Machine Learning", O'Reilly, 2018
5. **category_encoders** "A set of scikit-learn-style transformers for encoding categorical variables" — https://contrib.scikit-learn.org/category_encoders/

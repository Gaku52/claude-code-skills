# AI解析概要 — データサイエンスとMLの全体像

> データサイエンスと機械学習の全体像を俯瞰し、AI解析プロジェクトの進め方を体系的に理解する

## この章で学ぶこと

1. **データサイエンスのライフサイクル** — ビジネス課題定義からモデルデプロイまでの全工程
2. **機械学習の分類体系** — 教師あり・教師なし・強化学習の位置づけと使い分け
3. **AI解析プロジェクトの設計原則** — 再現性・スケーラビリティ・倫理を考慮した設計

---

## 1. データサイエンスのライフサイクル

AI解析プロジェクトは、直線的ではなく反復的なプロセスで進行する。CRISP-DM（Cross-Industry Standard Process for Data Mining）が最も広く採用されているフレームワークである。

### ライフサイクル全体像

```
+-------------------+
|  ビジネス理解     |
|  (Business        |
|   Understanding)  |
+--------+----------+
         |
         v
+--------+----------+     +-------------------+
|  データ理解       |<--->|  データ準備       |
|  (Data            |     |  (Data            |
|   Understanding)  |     |   Preparation)    |
+--------+----------+     +--------+----------+
         |                         |
         +----------+--------------+
                    |
                    v
         +----------+----------+
         |  モデリング         |
         |  (Modeling)         |
         +----------+----------+
                    |
                    v
         +----------+----------+
         |  評価               |
         |  (Evaluation)       |
         +----------+----------+
                    |
                    v
         +----------+----------+
         |  デプロイ           |
         |  (Deployment)       |
         +---------------------+
```

### コード例1: プロジェクト構成テンプレート

```python
# AI解析プロジェクトの標準ディレクトリ構成
"""
my-ml-project/
├── data/
│   ├── raw/              # 生データ（変更不可）
│   ├── processed/        # 前処理済みデータ
│   └── external/         # 外部データソース
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data/             # データ処理モジュール
│   ├── features/         # 特徴量エンジニアリング
│   ├── models/           # モデル定義・学習
│   └── visualization/    # 可視化ユーティリティ
├── models/               # 学習済みモデル保存
├── reports/              # 分析レポート
├── tests/                # テストコード
├── config.yaml           # 設定ファイル
├── requirements.txt      # 依存パッケージ
└── README.md
"""

import os

def create_project_structure(project_name: str) -> None:
    """AI解析プロジェクトの標準構造を作成"""
    dirs = [
        "data/raw", "data/processed", "data/external",
        "notebooks", "src/data", "src/features",
        "src/models", "src/visualization",
        "models", "reports", "tests"
    ]
    for d in dirs:
        os.makedirs(os.path.join(project_name, d), exist_ok=True)

    # .gitkeep を配置して空ディレクトリをGit管理
    for d in dirs:
        gitkeep = os.path.join(project_name, d, ".gitkeep")
        open(gitkeep, "w").close()

    print(f"プロジェクト '{project_name}' を作成しました")

create_project_structure("fraud-detection")
```

---

## 2. 機械学習の分類体系

### 学習パラダイムの全体図

```
              機械学習 (Machine Learning)
              ┌──────────┼──────────┐
              │          │          │
         教師あり学習  教師なし学習  強化学習
         (Supervised)  (Unsupervised) (Reinforcement)
              │          │          │
         ┌────┴────┐  ┌──┴──┐    ┌──┴──┐
         │         │  │     │    │     │
        回帰    分類  クラスタ 次元  方策   価値
       (Regr) (Cls) リング  削減  勾配   関数
                     (Clust) (DR) (PG)  (VF)
              │          │          │
         ┌────┴────┐  ┌──┴──┐    ┌──┴──┐
        線形  決定木  K-means PCA  Q学習  SARSA
        SVM   RF     DBSCAN tSNE PPO    A3C
        NN    GBM    GMM    UMAP DQN    SAC
```

### コード例2: 問題タイプ自動判定

```python
import pandas as pd
import numpy as np

def identify_problem_type(target: pd.Series) -> dict:
    """ターゲット変数から問題タイプを自動判定する"""
    result = {
        "dtype": str(target.dtype),
        "n_unique": target.nunique(),
        "n_samples": len(target),
    }

    # 数値型で多くのユニーク値 → 回帰
    if pd.api.types.is_numeric_dtype(target):
        ratio = target.nunique() / len(target)
        if ratio > 0.05 and target.nunique() > 20:
            result["problem_type"] = "回帰 (Regression)"
            result["suggested_metrics"] = ["RMSE", "MAE", "R²"]
        else:
            result["problem_type"] = "分類 (Classification)"
            result["suggested_metrics"] = ["Accuracy", "F1", "AUC-ROC"]
    else:
        result["problem_type"] = "分類 (Classification)"
        if target.nunique() == 2:
            result["sub_type"] = "二値分類"
        else:
            result["sub_type"] = "多クラス分類"
        result["suggested_metrics"] = ["Accuracy", "F1-macro", "AUC-ROC"]

    return result

# 使用例
df = pd.DataFrame({
    "price": [100.5, 200.3, 150.0, 300.7, 250.1],
    "category": ["A", "B", "A", "C", "B"],
    "is_fraud": [0, 1, 0, 0, 1]
})

print(identify_problem_type(df["price"]))
# {'dtype': 'float64', 'n_unique': 5, 'n_samples': 5,
#  'problem_type': '回帰 (Regression)', 'suggested_metrics': [...]}
```

---

## 3. データの種類と前処理の考え方

### コード例3: データ品質チェック

```python
import pandas as pd
import numpy as np

def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """データフレームの品質レポートを生成"""
    report = pd.DataFrame({
        "型": df.dtypes,
        "非null数": df.count(),
        "null数": df.isnull().sum(),
        "null率(%)": (df.isnull().sum() / len(df) * 100).round(2),
        "ユニーク数": df.nunique(),
        "ユニーク率(%)": (df.nunique() / len(df) * 100).round(2),
    })

    # 数値列の統計
    for col in df.select_dtypes(include=[np.number]).columns:
        report.loc[col, "平均"] = df[col].mean()
        report.loc[col, "標準偏差"] = df[col].std()
        report.loc[col, "最小"] = df[col].min()
        report.loc[col, "最大"] = df[col].max()

    return report

# 使用例
df = pd.read_csv("data/raw/sample.csv")  # 実際のデータ
report = data_quality_report(df)
print(report.to_string())
```

### コード例4: 探索的データ分析（EDA）パイプライン

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDARunner:
    """探索的データ分析の自動化ランナー"""

    def __init__(self, df: pd.DataFrame, target_col: str = None):
        self.df = df
        self.target = target_col
        self.numeric_cols = df.select_dtypes(include="number").columns.tolist()
        self.categorical_cols = df.select_dtypes(include="object").columns.tolist()

    def summary(self) -> None:
        """基本統計量の表示"""
        print("=" * 60)
        print(f"データ形状: {self.df.shape}")
        print(f"数値列: {len(self.numeric_cols)}")
        print(f"カテゴリ列: {len(self.categorical_cols)}")
        print(f"欠損値あり列: {self.df.isnull().any().sum()}")
        print("=" * 60)

    def correlation_matrix(self) -> None:
        """相関行列のヒートマップ"""
        if len(self.numeric_cols) < 2:
            print("数値列が2列未満のため相関行列を生成できません")
            return
        corr = self.df[self.numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        plt.title("相関行列")
        plt.tight_layout()
        plt.savefig("reports/correlation_matrix.png", dpi=150)
        plt.close()

    def distribution_plots(self) -> None:
        """数値列の分布プロット"""
        n_cols = min(len(self.numeric_cols), 12)
        fig, axes = plt.subplots(
            nrows=(n_cols + 2) // 3, ncols=3, figsize=(15, 4 * ((n_cols + 2) // 3))
        )
        axes = axes.flatten() if n_cols > 1 else [axes]

        for i, col in enumerate(self.numeric_cols[:n_cols]):
            self.df[col].hist(bins=30, ax=axes[i], edgecolor="black")
            axes[i].set_title(col)

        plt.tight_layout()
        plt.savefig("reports/distributions.png", dpi=150)
        plt.close()

# 使用例
# eda = EDARunner(df, target_col="price")
# eda.summary()
# eda.correlation_matrix()
# eda.distribution_plots()
```

---

## 4. AI解析プロジェクトのワークフロー

### ワークフロー詳細図

```
データ取得          前処理             特徴量             モデリング          評価
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ DB接続   │    │ 欠損値   │    │ エンコード│    │ 学習     │    │ 交差検証 │
│ API取得  │───>│ 外れ値   │───>│ スケーリング│──>│ ハイパー │───>│ 指標計算 │
│ CSV読込  │    │ 重複除去 │    │ 選択     │    │ パラメータ│    │ 可視化   │
│ Web取得  │    │ 型変換   │    │ 生成     │    │ 調整     │    │ 解釈     │
└─────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
      │              │              │              │              │
      v              v              v              v              v
  data/raw/     data/processed/ src/features/  models/       reports/
```

### コード例5: 設定駆動型パイプライン

```python
import yaml
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ExperimentConfig:
    """実験設定をYAMLから読み込む"""
    name: str
    data_path: str
    target_column: str
    feature_columns: List[str]
    model_type: str = "random_forest"
    test_size: float = 0.2
    random_state: int = 42
    hyperparams: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def to_yaml(self, path: str) -> None:
        import dataclasses
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)

# config.yaml の例:
# name: "house_price_prediction"
# data_path: "data/processed/house_prices.csv"
# target_column: "price"
# feature_columns: ["area", "rooms", "age", "location"]
# model_type: "gradient_boosting"
# test_size: 0.2
# random_state: 42
# hyperparams:
#   n_estimators: 500
#   max_depth: 6
#   learning_rate: 0.01
```

---

## 比較表

### AI解析手法の選択ガイド

| データの性質 | 推奨手法 | 代表的アルゴリズム | 出力 | 典型的なユースケース |
|---|---|---|---|---|
| ラベル付き・連続値 | 教師あり回帰 | 線形回帰, XGBoost | 数値 | 価格予測, 需要予測 |
| ラベル付き・離散値 | 教師あり分類 | ロジスティック回帰, RF | カテゴリ | 不正検知, 診断 |
| ラベルなし・構造発見 | 教師なしクラスタリング | K-means, DBSCAN | クラスタID | 顧客セグメント |
| ラベルなし・次元圧縮 | 教師なし次元削減 | PCA, t-SNE | 低次元表現 | 可視化, ノイズ除去 |
| 逐次的意思決定 | 強化学習 | Q学習, PPO | 行動方策 | ゲームAI, ロボット制御 |
| 大量テキスト | NLP | BERT, GPT | テキスト | 翻訳, 要約, 分類 |
| 画像・動画 | コンピュータビジョン | CNN, ViT | 検出/分類 | 自動運転, 医療画像 |

### プロジェクト規模別ツール選択

| 項目 | 小規模 (〜1万行) | 中規模 (〜100万行) | 大規模 (1億行〜) |
|---|---|---|---|
| データ処理 | pandas | pandas + Dask | Spark / Polars |
| モデル学習 | scikit-learn | XGBoost / LightGBM | 分散学習 (Horovod) |
| 実験管理 | ノートブック | MLflow | Kubeflow / Vertex AI |
| デプロイ | Flask / FastAPI | Docker + Cloud Run | Kubernetes + Seldon |
| 計算環境 | ローカル | GPU (Colab / SageMaker) | マルチGPUクラスタ |
| コスト目安 | 無料〜数千円/月 | 数千〜数万円/月 | 数十万円/月〜 |

---

## アンチパターン

### アンチパターン1: リーク（Data Leakage）

```python
# BAD: テストデータを含めてスケーリングしている
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # 全データで fit → リーク!
X_train, X_test = train_test_split(X_scaled)

# GOOD: 訓練データのみで fit し、テストデータには transform のみ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 訓練のみで fit
X_test_scaled = scaler.transform(X_test)         # transform のみ
```

**なぜ危険か**: テストデータの統計量が訓練に混入し、モデルの汎化性能を過大評価する。本番環境で期待通りの性能が出ない原因の大半がリークである。

### アンチパターン2: 「まずディープラーニング」症候群

```
問題: 1000件の表形式データで売上を予測したい

BAD な思考:
  "最新のTransformerモデルを使おう" → 過学習 → 性能劣化

GOOD な思考:
  "まず線形回帰 + 特徴量エンジニアリング" → ベースライン確立
  → 改善が必要ならXGBoost → それでも不足ならNN検討

判断基準:
  ┌─────────────────────────────────────────────────────┐
  │ データ量 < 10,000   → 古典ML（勾配ブースティング）  │
  │ データ量 10K〜100K  → 古典ML or 浅いNN              │
  │ データ量 > 100K     → DLも選択肢に入る              │
  │ 画像/音声/テキスト  → DL推奨（データ量に依らず）     │
  └─────────────────────────────────────────────────────┘
```

### アンチパターン3: 再現性の欠如

```python
# BAD: ランダムシードを固定していない
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, y)  # 実行ごとに結果が異なる

# GOOD: ランダムシードを固定し、環境情報も記録
import random, numpy as np, platform

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# 実行環境を記録
print(f"Python: {platform.python_version()}")
print(f"NumPy: {np.__version__}")
print(f"OS: {platform.system()} {platform.release()}")
```

---

## FAQ

### Q1: データサイエンティストとMLエンジニアの違いは？

**A:** データサイエンティストは「分析・仮説検証・洞察の発見」に重点を置き、MLエンジニアは「モデルの本番運用・スケーラビリティ・信頼性」に重点を置く。小規模チームでは兼務することが多いが、大規模組織では分業する。どちらも統計学、プログラミング、ドメイン知識が必要である。

### Q2: AI解析プロジェクトの成功率はどれくらい？

**A:** Gartnerの調査（2023年）では、AI/MLプロジェクトの約85%が本番環境に到達しないと報告されている。主な失敗要因は、(1) 問題定義が不明確、(2) データ品質が低い、(3) ビジネスKPIとの紐づけ不足。技術的失敗よりも組織的・プロセス的失敗が多い。

### Q3: PoCからプロダクション化へ移行するコツは？

**A:** (1) PoCの段階から本番を意識したコード品質を保つ、(2) データパイプラインの自動化を早期に構築する、(3) モニタリング指標を最初から定義する、(4) ステークホルダーと定期的にレビューを行い期待値を管理する。特に「Notebookをそのまま本番に持ち込まない」ことが重要である。

### Q4: GPUは必要か？

**A:** 表形式データの古典的ML（XGBoost等）ではCPUで十分。ディープラーニング（画像、テキスト、音声）ではGPUが事実上必須。Google ColabやAWS SageMakerなどのクラウドGPUを活用すれば初期投資を抑えられる。

---

## まとめ

| 項目 | 要点 |
|---|---|
| ライフサイクル | CRISP-DM: ビジネス理解→データ理解→準備→モデリング→評価→デプロイの反復 |
| 学習パラダイム | 教師あり（回帰・分類）、教師なし（クラスタ・次元削減）、強化学習 |
| プロジェクト設計 | 再現性（シード固定）、モジュール性（設定駆動）、品質（データ検証） |
| ツール選択 | 規模に応じて段階的にスケールアップ。最小限のツールから始める |
| 成功の鍵 | 技術より問題定義とデータ品質。ベースラインから始めて段階的に改善 |

---

## 次に読むべきガイド

- [01-data-preprocessing.md](./01-data-preprocessing.md) — データ前処理の具体的手法
- [02-ml-basics.md](./02-ml-basics.md) — 機械学習の基礎理論と評価指標
- [03-python-ml-stack.md](./03-python-ml-stack.md) — Python ML開発環境の構築

---

## 参考文献

1. **Aurélien Géron** "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" 3rd Edition, O'Reilly Media, 2022
2. **Pete Chapman et al.** "CRISP-DM 1.0: Step-by-step data mining guide" SPSS Inc., 2000 — https://www.datascience-pm.com/crisp-dm-2/
3. **Google** "Rules of Machine Learning: Best Practices for ML Engineering" — https://developers.google.com/machine-learning/guides/rules-of-ml
4. **MLOps Community** "MLOps Principles" — https://ml-ops.org/content/mlops-principles

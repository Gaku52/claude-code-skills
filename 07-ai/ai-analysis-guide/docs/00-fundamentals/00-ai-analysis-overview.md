# AI解析概要 — データサイエンスとMLの全体像

> データサイエンスと機械学習の全体像を俯瞰し、AI解析プロジェクトの進め方を体系的に理解する

## この章で学ぶこと

1. **データサイエンスのライフサイクル** — ビジネス課題定義からモデルデプロイまでの全工程
2. **機械学習の分類体系** — 教師あり・教師なし・強化学習の位置づけと使い分け
3. **AI解析プロジェクトの設計原則** — 再現性・スケーラビリティ・倫理を考慮した設計
4. **EDA（探索的データ分析）** — データの理解を深める系統的なアプローチ
5. **実験管理とバージョニング** — MLflowやDVCを使った実験の追跡と再現
6. **プロジェクト推進のベストプラクティス** — チーム運営、ステークホルダー管理、リスク対策

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

### 各フェーズの詳細と成果物

```
CRISP-DM 各フェーズの詳細:

Phase 1: ビジネス理解
  ├── 入力: ビジネス課題、ステークホルダーの要件
  ├── 活動:
  │   ├── KPIの定義（売上向上5%、解約率10%削減等）
  │   ├── 成功基準の明確化（精度閾値、レイテンシ要件）
  │   ├── データ利用可能性の調査
  │   └── ROIの概算
  └── 成果物: プロジェクト計画書、成功基準文書

Phase 2: データ理解
  ├── 入力: 利用可能なデータソース
  ├── 活動:
  │   ├── データ収集とアクセス確保
  │   ├── EDA（探索的データ分析）
  │   ├── データ品質評価
  │   └── 初期仮説の構築
  └── 成果物: EDAレポート、データ辞書、品質レポート

Phase 3: データ準備
  ├── 入力: 生データ、EDA結果
  ├── 活動:
  │   ├── データクリーニング（欠損値、外れ値、重複）
  │   ├── 特徴量エンジニアリング
  │   ├── データ統合（複数ソースの結合）
  │   └── 訓練/検証/テスト分割
  └── 成果物: 前処理済みデータセット、前処理パイプライン

Phase 4: モデリング
  ├── 入力: 前処理済みデータ
  ├── 活動:
  │   ├── ベースラインモデルの構築
  │   ├── 複数アルゴリズムの比較
  │   ├── ハイパーパラメータ最適化
  │   └── アンサンブル手法の検討
  └── 成果物: 学習済みモデル、実験ログ

Phase 5: 評価
  ├── 入力: 学習済みモデル、テストデータ
  ├── 活動:
  │   ├── テストデータでの性能評価
  │   ├── ビジネスKPIとの照合
  │   ├── 公平性・バイアスのチェック
  │   └── A/Bテスト計画の策定
  └── 成果物: 評価レポート、デプロイ判定

Phase 6: デプロイ
  ├── 入力: 承認済みモデル
  ├── 活動:
  │   ├── モデルのパッケージング
  │   ├── API/バッチ推論の構築
  │   ├── モニタリングの設定
  │   └── 再学習パイプラインの構築
  └── 成果物: 本番サービス、モニタリングダッシュボード
```

### コード例1: プロジェクト構成テンプレート

```python
# AI解析プロジェクトの標準ディレクトリ構成
"""
my-ml-project/
├── data/
│   ├── raw/              # 生データ（変更不可）
│   ├── processed/        # 前処理済みデータ
│   ├── interim/          # 中間データ
│   └── external/         # 外部データソース
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data/             # データ処理モジュール
│   │   ├── __init__.py
│   │   ├── loader.py     # データ読込
│   │   └── validation.py # データ検証
│   ├── features/         # 特徴量エンジニアリング
│   │   ├── __init__.py
│   │   ├── builder.py    # 特徴量生成
│   │   └── selector.py   # 特徴量選択
│   ├── models/           # モデル定義・学習
│   │   ├── __init__.py
│   │   ├── train.py      # 学習スクリプト
│   │   ├── predict.py    # 推論スクリプト
│   │   └── evaluate.py   # 評価スクリプト
│   └── visualization/    # 可視化ユーティリティ
│       ├── __init__.py
│       └── plots.py
├── models/               # 学習済みモデル保存
├── reports/              # 分析レポート・図表
│   └── figures/
├── tests/                # テストコード
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── configs/              # 設定ファイル
│   ├── config.yaml
│   └── hyperparams.yaml
├── scripts/              # ユーティリティスクリプト
│   ├── train.sh
│   └── deploy.sh
├── Makefile              # タスクランナー
├── pyproject.toml        # プロジェクト設定
├── requirements.txt      # 依存パッケージ
├── .env.example          # 環境変数テンプレート
├── .gitignore
└── README.md
"""

import os
import json
from datetime import datetime

def create_project_structure(project_name: str,
                              include_dvc: bool = False,
                              include_docker: bool = False) -> None:
    """AI解析プロジェクトの標準構造を作成

    Parameters
    ----------
    project_name : str
        プロジェクト名
    include_dvc : bool
        DVCファイルを含めるか
    include_docker : bool
        Dockerfileを含めるか
    """
    dirs = [
        "data/raw", "data/processed", "data/interim", "data/external",
        "notebooks", "src/data", "src/features",
        "src/models", "src/visualization",
        "models", "reports/figures", "tests",
        "configs", "scripts"
    ]
    for d in dirs:
        os.makedirs(os.path.join(project_name, d), exist_ok=True)

    # .gitkeep を配置して空ディレクトリをGit管理
    for d in dirs:
        gitkeep = os.path.join(project_name, d, ".gitkeep")
        open(gitkeep, "w").close()

    # __init__.py の作成
    for d in ["src", "src/data", "src/features", "src/models", "src/visualization"]:
        init_file = os.path.join(project_name, d, "__init__.py")
        open(init_file, "w").close()

    # .gitignore の作成
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
.venv/
env/

# Data
data/raw/*
data/processed/*
data/interim/*
!data/**/.gitkeep

# Models
models/*.joblib
models/*.pkl
models/*.h5

# Environment
.env

# IDE
.vscode/
.idea/

# Jupyter
.ipynb_checkpoints/

# MLflow
mlruns/
"""
    with open(os.path.join(project_name, ".gitignore"), "w") as f:
        f.write(gitignore_content)

    # Makefile の作成
    makefile_content = """.PHONY: install train test lint clean

install:
\tpip install -r requirements.txt

train:
\tpython -m src.models.train --config configs/config.yaml

test:
\tpytest tests/ -v

lint:
\truff check src/ tests/
\tmypy src/

clean:
\trm -rf __pycache__ .pytest_cache models/*.joblib
"""
    with open(os.path.join(project_name, "Makefile"), "w") as f:
        f.write(makefile_content)

    # プロジェクトメタデータ
    metadata = {
        "name": project_name,
        "created_at": datetime.now().isoformat(),
        "version": "0.1.0",
    }
    with open(os.path.join(project_name, "project_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if include_docker:
        dockerfile = """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY models/ models/
COPY configs/ configs/
CMD ["python", "-m", "src.models.predict"]
"""
        with open(os.path.join(project_name, "Dockerfile"), "w") as f:
            f.write(dockerfile)

    print(f"プロジェクト '{project_name}' を作成しました")
    print(f"  ディレクトリ数: {len(dirs)}")

create_project_structure("fraud-detection", include_docker=True)
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

追加パラダイム:
  ┌──────────────────────────────────────────┐
  │ 半教師あり学習 (Semi-Supervised)          │
  │   少量のラベル付きデータ + 大量のラベルなし│
  │   例: Self-Training, Label Propagation    │
  ├──────────────────────────────────────────┤
  │ 自己教師あり学習 (Self-Supervised)        │
  │   データ自体から学習信号を生成            │
  │   例: BERT (MLM), GPT (NTP), SimCLR     │
  ├──────────────────────────────────────────┤
  │ 転移学習 (Transfer Learning)             │
  │   事前学習済みモデルを別タスクに適用      │
  │   例: Fine-tuning, Feature Extraction    │
  ├──────────────────────────────────────────┤
  │ メタ学習 (Meta-Learning)                 │
  │   「学ぶ方法を学ぶ」                     │
  │   例: MAML, Prototypical Networks        │
  └──────────────────────────────────────────┘
```

### 各パラダイムの適用場面

```
タスクとデータに応じたパラダイム選択:

  ┌─────────────────────────┐
  │ ラベル付きデータがある？ │
  └────────┬────────────────┘
       Yes │          No
           │            │
  ┌────────┴──────┐  ┌──┴──────────────────┐
  │ 十分な量ある？│  │ 構造を発見したい？  │
  └────┬──────────┘  └──┬───────────────────┘
   Yes │   No          Yes │        No
       │     │             │          │
  教師あり  半教師あり  教師なし   自己教師あり
  学習     学習       学習      学習
       │                            │
  ┌────┴────┐                  ┌────┴────┐
  │目的変数 │                  │事前学習 │
  │の型は？ │                  │→Fine-tune│
  └────┬────┘                  └─────────┘
  連続│   離散│
      │       │
   回帰    分類
```

### コード例2: 問題タイプ自動判定

```python
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def identify_problem_type(target: pd.Series,
                           threshold_unique_ratio: float = 0.05,
                           threshold_unique_count: int = 20) -> Dict[str, Any]:
    """ターゲット変数から問題タイプを自動判定する

    Parameters
    ----------
    target : pd.Series
        ターゲット変数
    threshold_unique_ratio : float
        ユニーク値比率の閾値（これ以上なら回帰）
    threshold_unique_count : int
        ユニーク値数の閾値（これ以上なら回帰候補）
    """
    result = {
        "dtype": str(target.dtype),
        "n_unique": target.nunique(),
        "n_samples": len(target),
        "missing_rate": target.isnull().mean(),
        "unique_ratio": target.nunique() / len(target),
    }

    # 数値型で多くのユニーク値 → 回帰
    if pd.api.types.is_numeric_dtype(target):
        ratio = target.nunique() / len(target)
        if ratio > threshold_unique_ratio and target.nunique() > threshold_unique_count:
            result["problem_type"] = "回帰 (Regression)"
            result["suggested_metrics"] = ["RMSE", "MAE", "R²", "MAPE"]
            result["suggested_models"] = [
                "LinearRegression", "Ridge", "Lasso",
                "RandomForestRegressor", "XGBRegressor", "LGBMRegressor"
            ]
            result["baseline_model"] = "平均値予測 (DummyRegressor)"
        else:
            result["problem_type"] = "分類 (Classification)"
            if target.nunique() == 2:
                result["sub_type"] = "二値分類 (Binary)"
                result["suggested_metrics"] = ["F1", "AUC-ROC", "Precision", "Recall"]
            else:
                result["sub_type"] = f"多クラス分類 ({target.nunique()}クラス)"
                result["suggested_metrics"] = ["F1-macro", "AUC-ROC (OVR)", "Accuracy"]
            result["suggested_models"] = [
                "LogisticRegression", "RandomForestClassifier",
                "XGBClassifier", "LGBMClassifier"
            ]
            result["baseline_model"] = "最頻値予測 (DummyClassifier)"

            # クラス不均衡の検出
            value_counts = target.value_counts(normalize=True)
            imbalance_ratio = value_counts.max() / value_counts.min()
            if imbalance_ratio > 5:
                result["warning"] = (
                    f"クラス不均衡検出 (比率={imbalance_ratio:.1f}x). "
                    f"SMOTE/class_weight の使用を推奨"
                )
                result["suggested_metrics"].extend(["PR-AUC", "MCC"])
    else:
        result["problem_type"] = "分類 (Classification)"
        if target.nunique() == 2:
            result["sub_type"] = "二値分類"
        else:
            result["sub_type"] = f"多クラス分類 ({target.nunique()}クラス)"
        result["suggested_metrics"] = ["F1-macro", "AUC-ROC", "Accuracy"]
        result["suggested_models"] = [
            "LogisticRegression", "RandomForestClassifier",
            "XGBClassifier", "LGBMClassifier"
        ]

    return result

def suggest_approach(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """データフレームからAI解析アプローチを提案"""
    n_samples, n_features = df.shape
    target = df[target_col]
    problem_info = identify_problem_type(target)

    suggestion = {
        **problem_info,
        "n_samples": n_samples,
        "n_features": n_features - 1,  # ターゲット列を除く
        "data_size_category": (
            "小規模" if n_samples < 10000
            else "中規模" if n_samples < 1000000
            else "大規模"
        ),
    }

    # データ量に応じたツール推奨
    if n_samples < 10000:
        suggestion["recommended_framework"] = "scikit-learn"
        suggestion["compute"] = "CPU (ローカル)"
    elif n_samples < 1000000:
        suggestion["recommended_framework"] = "XGBoost / LightGBM"
        suggestion["compute"] = "CPU or GPU (クラウド推奨)"
    else:
        suggestion["recommended_framework"] = "Spark / Dask + LightGBM"
        suggestion["compute"] = "分散処理クラスタ"

    return suggestion

# 使用例
df = pd.DataFrame({
    "price": [100.5, 200.3, 150.0, 300.7, 250.1],
    "category": ["A", "B", "A", "C", "B"],
    "is_fraud": [0, 1, 0, 0, 1]
})

print("=== 回帰タスク ===")
print(identify_problem_type(df["price"]))
print("\n=== 分類タスク ===")
print(identify_problem_type(df["is_fraud"]))
```

---

## 3. データの種類と前処理の考え方

### データの種類と特性

```
データ種別と前処理の対応:

  構造化データ（表形式）:
    ├── 数値データ: 連続値（身長、価格）、離散値（年齢、回数）
    │   └── 前処理: スケーリング、欠損値補完、外れ値処理
    ├── カテゴリデータ: 名義（色、地域）、順序（学歴、満足度）
    │   └── 前処理: エンコーディング（OneHot, Target, Ordinal）
    └── 時系列データ: タイムスタンプ付き連続観測
        └── 前処理: ラグ特徴量、移動平均、差分、周期特徴量

  非構造化データ:
    ├── テキスト: 自然言語文書
    │   └── 前処理: トークン化、TF-IDF、Word2Vec、BERT埋め込み
    ├── 画像: ピクセルデータ
    │   └── 前処理: リサイズ、正規化、データ拡張（回転、反転）
    ├── 音声: 波形データ
    │   └── 前処理: MFCC、スペクトログラム変換
    └── 動画: フレーム列
        └── 前処理: キーフレーム抽出、オプティカルフロー
```

### コード例3: データ品質チェック

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

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
        report.loc[col, "歪度"] = df[col].skew()
        report.loc[col, "尖度"] = df[col].kurtosis()

    # カテゴリ列のトップ値
    for col in df.select_dtypes(include=["object", "category"]).columns:
        top_val = df[col].mode()
        if len(top_val) > 0:
            report.loc[col, "最頻値"] = top_val.iloc[0]
            report.loc[col, "最頻値の割合(%)"] = (
                df[col].value_counts(normalize=True).iloc[0] * 100
            )

    return report

def detect_data_issues(df: pd.DataFrame) -> List[Dict[str, str]]:
    """データの潜在的な問題を検出"""
    issues = []

    # 1. 高い欠損率
    for col in df.columns:
        missing_rate = df[col].isnull().mean()
        if missing_rate > 0.5:
            issues.append({
                "列": col, "問題": "高い欠損率",
                "詳細": f"欠損率={missing_rate:.1%}",
                "推奨": "列削除またはモデルベース補完"
            })

    # 2. 定数列
    for col in df.columns:
        if df[col].nunique() <= 1:
            issues.append({
                "列": col, "問題": "定数列",
                "詳細": f"ユニーク値={df[col].nunique()}",
                "推奨": "列を除去"
            })

    # 3. 高カーディナリティのカテゴリ列
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() > 100:
            issues.append({
                "列": col, "問題": "高カーディナリティ",
                "詳細": f"ユニーク値={df[col].nunique()}",
                "推奨": "TargetEncoding or Hash"
            })

    # 4. 重複行
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        issues.append({
            "列": "全体", "問題": "重複行",
            "詳細": f"{n_dup}行 ({n_dup/len(df)*100:.1f}%)",
            "推奨": "重複を確認し、必要に応じて除去"
        })

    # 5. 疑わしい外れ値（数値列）
    for col in df.select_dtypes(include=[np.number]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outlier_count = ((df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)).sum()
        if outlier_count > 0:
            issues.append({
                "列": col, "問題": "外れ値",
                "詳細": f"{outlier_count}個 (3×IQR基準)",
                "推奨": "クリッピングまたは変換を検討"
            })

    if not issues:
        print("潜在的な問題は検出されませんでした。")
    else:
        print(f"検出された問題: {len(issues)}件")
        for issue in issues:
            print(f"  [{issue['列']}] {issue['問題']}: {issue['詳細']} → {issue['推奨']}")

    return issues

# 使用例
# df = pd.read_csv("data/raw/sample.csv")
# report = data_quality_report(df)
# print(report.to_string())
# issues = detect_data_issues(df)
```

### コード例4: 探索的データ分析（EDA）パイプライン

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

class EDARunner:
    """探索的データ分析の自動化ランナー"""

    def __init__(self, df: pd.DataFrame, target_col: str = None):
        self.df = df
        self.target = target_col
        self.numeric_cols = df.select_dtypes(include="number").columns.tolist()
        self.categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    def summary(self) -> None:
        """基本統計量の表示"""
        print("=" * 60)
        print(f"データ形状: {self.df.shape}")
        print(f"数値列: {len(self.numeric_cols)}")
        print(f"カテゴリ列: {len(self.categorical_cols)}")
        print(f"欠損値あり列: {self.df.isnull().any().sum()}")
        print(f"重複行: {self.df.duplicated().sum()}")
        print(f"メモリ使用量: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        print("=" * 60)

        if self.target:
            print(f"\nターゲット変数: {self.target}")
            if self.target in self.numeric_cols:
                print(f"  平均: {self.df[self.target].mean():.4f}")
                print(f"  標準偏差: {self.df[self.target].std():.4f}")
                print(f"  中央値: {self.df[self.target].median():.4f}")
            else:
                print(f"  クラス分布:")
                for cls, count in self.df[self.target].value_counts().items():
                    pct = count / len(self.df) * 100
                    print(f"    {cls}: {count} ({pct:.1f}%)")

    def correlation_matrix(self, method: str = "pearson",
                           threshold: float = 0.7) -> None:
        """相関行列のヒートマップ"""
        if len(self.numeric_cols) < 2:
            print("数値列が2列未満のため相関行列を生成できません")
            return

        corr = self.df[self.numeric_cols].corr(method=method)

        fig, ax = plt.subplots(figsize=(max(10, len(self.numeric_cols) * 0.8),
                                        max(8, len(self.numeric_cols) * 0.6)))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    mask=mask, ax=ax, vmin=-1, vmax=1, center=0)
        plt.title(f"相関行列 ({method})")
        plt.tight_layout()
        plt.savefig("reports/correlation_matrix.png", dpi=150)
        plt.close()

        # 高相関ペアの報告
        high_corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > threshold:
                    high_corr_pairs.append(
                        (corr.columns[i], corr.columns[j], corr.iloc[i, j])
                    )

        if high_corr_pairs:
            print(f"\n高相関ペア (|r| > {threshold}):")
            for col1, col2, r in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"  {col1} <-> {col2}: r = {r:.3f}")

    def distribution_plots(self) -> None:
        """数値列の分布プロット"""
        n_cols = min(len(self.numeric_cols), 16)
        if n_cols == 0:
            return

        n_grid_cols = 4
        n_grid_rows = (n_cols + n_grid_cols - 1) // n_grid_cols

        fig, axes = plt.subplots(
            nrows=n_grid_rows, ncols=n_grid_cols,
            figsize=(4 * n_grid_cols, 3.5 * n_grid_rows)
        )
        if n_grid_rows == 1 and n_grid_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, col in enumerate(self.numeric_cols[:n_cols]):
            self.df[col].hist(bins=30, ax=axes[i], edgecolor="black", alpha=0.7)
            axes[i].set_title(f"{col}\n(mean={self.df[col].mean():.1f}, "
                              f"std={self.df[col].std():.1f})", fontsize=9)
            axes[i].axvline(self.df[col].mean(), color="red", linestyle="--", alpha=0.5)

        for i in range(n_cols, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle("数値特徴量の分布", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig("reports/distributions.png", dpi=150, bbox_inches="tight")
        plt.close()

    def target_analysis(self) -> None:
        """ターゲット変数と各特徴量の関係を可視化"""
        if self.target is None:
            print("ターゲット列が指定されていません")
            return

        # 数値特徴量 vs ターゲット
        n_cols = min(len(self.numeric_cols), 12)
        if n_cols > 0 and self.target in self.numeric_cols:
            feature_cols = [c for c in self.numeric_cols if c != self.target][:n_cols]
            n_grid_cols = 3
            n_grid_rows = (len(feature_cols) + n_grid_cols - 1) // n_grid_cols

            fig, axes = plt.subplots(n_grid_rows, n_grid_cols,
                                      figsize=(5 * n_grid_cols, 4 * n_grid_rows))
            axes = axes.flatten() if n_grid_rows > 1 else [axes] if n_grid_rows == 1 and n_grid_cols == 1 else axes.flatten()

            for i, col in enumerate(feature_cols):
                if i < len(axes):
                    axes[i].scatter(self.df[col], self.df[self.target],
                                    alpha=0.3, s=10)
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel(self.target)
                    corr_val = self.df[col].corr(self.df[self.target])
                    axes[i].set_title(f"r = {corr_val:.3f}")

            for i in range(len(feature_cols), len(axes)):
                axes[i].set_visible(False)

            plt.suptitle(f"特徴量 vs {self.target}", fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig("reports/target_analysis.png", dpi=150, bbox_inches="tight")
            plt.close()

    def categorical_analysis(self) -> None:
        """カテゴリ変数の分布を可視化"""
        n_cats = min(len(self.categorical_cols), 8)
        if n_cats == 0:
            return

        fig, axes = plt.subplots(2, min(4, n_cats), figsize=(5 * min(4, n_cats), 8))
        if n_cats == 1:
            axes = np.array([[axes]])
        elif n_cats <= 4:
            axes = axes.reshape(2, -1)

        for i, col in enumerate(self.categorical_cols[:n_cats]):
            row, col_idx = i // 4, i % 4
            if col_idx < axes.shape[1] and row < axes.shape[0]:
                top_n = self.df[col].value_counts().head(10)
                top_n.plot(kind="barh", ax=axes[row][col_idx])
                axes[row][col_idx].set_title(f"{col} (Top 10)")

        plt.tight_layout()
        plt.savefig("reports/categorical_analysis.png", dpi=150)
        plt.close()

    def full_report(self) -> None:
        """全てのEDA分析を実行"""
        print("EDA レポート生成中...")
        self.summary()
        self.correlation_matrix()
        self.distribution_plots()
        self.target_analysis()
        self.categorical_analysis()
        print("レポート生成完了。reports/ ディレクトリを確認してください。")

# 使用例
# eda = EDARunner(df, target_col="price")
# eda.full_report()
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
from typing import List, Optional, Dict, Any

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
    cv_folds: int = 5
    scoring: str = "f1"
    hyperparams: dict = field(default_factory=dict)
    preprocessing: dict = field(default_factory=lambda: {
        "scaler": "standard",
        "imputer": "median",
        "encoder": "onehot"
    })

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def to_yaml(self, path: str) -> None:
        import dataclasses
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)

    def validate(self) -> List[str]:
        """設定の妥当性を検証"""
        errors = []
        if not self.name:
            errors.append("name は必須です")
        if self.test_size <= 0 or self.test_size >= 1:
            errors.append("test_size は 0〜1 の間で指定してください")
        if self.cv_folds < 2:
            errors.append("cv_folds は 2 以上を指定してください")
        valid_models = ["random_forest", "gradient_boosting", "logistic_regression",
                        "xgboost", "lightgbm", "svm", "neural_network"]
        if self.model_type not in valid_models:
            errors.append(f"model_type は {valid_models} のいずれかを指定してください")
        return errors

# config.yaml の例:
"""
name: "house_price_prediction"
data_path: "data/processed/house_prices.csv"
target_column: "price"
feature_columns: ["area", "rooms", "age", "location"]
model_type: "gradient_boosting"
test_size: 0.2
random_state: 42
cv_folds: 5
scoring: "neg_root_mean_squared_error"
hyperparams:
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.01
preprocessing:
  scaler: "robust"
  imputer: "knn"
  encoder: "target"
"""
```

---

## 5. 実験管理

### 実験管理の重要性

```
実験管理なしの場合の問題:

  "先週のモデル精度 92% だったけど、
   どのパラメータだったか覚えていない..."

  ├── パラメータの記録漏れ
  ├── データバージョンの不一致
  ├── コード変更の追跡不能
  └── 結果の再現不能

実験管理ありの場合:

  Experiment: fraud_detection_v3
  ├── Run ID: abc123
  ├── Parameters: {max_depth: 5, lr: 0.01, ...}
  ├── Metrics: {f1: 0.923, auc: 0.968, ...}
  ├── Artifacts: model.joblib, confusion_matrix.png
  ├── Data Version: data/processed/v2.3
  ├── Git Commit: e4f5a6b
  └── Tags: ["production_candidate", "v3"]
```

### コード例6: MLflowによる実験管理

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report
)
import numpy as np
import json
import platform
from datetime import datetime

class ExperimentTracker:
    """MLflowベースの実験管理クラス"""

    def __init__(self, experiment_name: str,
                 tracking_uri: str = "mlruns"):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def log_environment(self):
        """実行環境情報をログ"""
        import sklearn
        mlflow.log_param("python_version", platform.python_version())
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("os", f"{platform.system()} {platform.release()}")
        mlflow.log_param("timestamp", datetime.now().isoformat())

    def run_experiment(self, model, model_name, X_train, X_test,
                       y_train, y_test, params=None):
        """実験を実行してMLflowにログ"""
        with mlflow.start_run(run_name=model_name):
            # 環境情報
            self.log_environment()

            # パラメータのログ
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            if params:
                mlflow.log_params(params)

            # 交差検証
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=5, scoring="f1")
            mlflow.log_metric("cv_f1_mean", cv_scores.mean())
            mlflow.log_metric("cv_f1_std", cv_scores.std())

            # モデル学習
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] \
                     if hasattr(model, "predict_proba") else None

            # メトリクスのログ
            metrics = {
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_f1": f1_score(y_test, y_pred),
            }
            if y_prob is not None:
                metrics["test_auc_roc"] = roc_auc_score(y_test, y_prob)

            mlflow.log_metrics(metrics)

            # モデルの保存
            mlflow.sklearn.log_model(model, "model")

            # 分類レポートのアーティファクト
            report = classification_report(y_test, y_pred, output_dict=True)
            with open("classification_report.json", "w") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact("classification_report.json")

            print(f"\n{model_name}:")
            print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

            return metrics

    def compare_models(self, models_dict, X_train, X_test, y_train, y_test):
        """複数モデルを比較実行"""
        results = {}
        for name, (model, params) in models_dict.items():
            metrics = self.run_experiment(
                model, name, X_train, X_test, y_train, y_test, params
            )
            results[name] = metrics

        # 結果の比較表
        print("\n" + "=" * 60)
        print("モデル比較")
        print("=" * 60)
        for name, metrics in results.items():
            print(f"  {name:30s}: F1={metrics['test_f1']:.4f}, "
                  f"Acc={metrics['test_accuracy']:.4f}")

        return results

# 使用例
# tracker = ExperimentTracker("fraud_detection")
# models = {
#     "RandomForest": (
#         RandomForestClassifier(n_estimators=100, random_state=42),
#         {"n_estimators": 100}
#     ),
#     "GradientBoosting": (
#         GradientBoostingClassifier(n_estimators=200, random_state=42),
#         {"n_estimators": 200}
#     ),
# }
# results = tracker.compare_models(models, X_train, X_test, y_train, y_test)
```

### コード例7: 再現性を確保するシード管理

```python
import random
import numpy as np
import os

def set_global_seed(seed: int = 42) -> None:
    """全てのランダムシードを一括設定"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch（インストール済みの場合）
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # TensorFlow（インストール済みの場合）
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    print(f"Global seed set to {seed}")

def log_environment_info() -> dict:
    """実行環境の情報を収集"""
    import platform
    import sys

    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    # 主要ライブラリのバージョン
    for lib in ["numpy", "pandas", "sklearn", "xgboost", "lightgbm",
                "torch", "tensorflow"]:
        try:
            mod = __import__(lib)
            info[f"{lib}_version"] = mod.__version__
        except ImportError:
            pass

    return info

# プロジェクトの冒頭で呼び出す
set_global_seed(42)
env_info = log_environment_info()
print("環境情報:")
for key, value in env_info.items():
    print(f"  {key}: {value}")
```

---

## 6. モデルデプロイメント

### デプロイメントパターン

```
モデルデプロイメントの選択肢:

  1. バッチ推論
     ├── 定期的にまとめて推論（日次/時間次）
     ├── 適用場面: レコメンド、レポート生成
     └── ツール: Airflow, Cloud Functions, cron

  2. リアルタイム推論 (REST API)
     ├── HTTPリクエストに対してリアルタイムで応答
     ├── 適用場面: 不正検知、チャットボット
     └── ツール: FastAPI, Flask, Seldon

  3. エッジ推論
     ├── デバイス上で推論（モバイル、IoT）
     ├── 適用場面: 自動運転、スマートフォンアプリ
     └── ツール: TensorFlow Lite, ONNX Runtime

  4. ストリーミング推論
     ├── データストリームに対して連続的に推論
     ├── 適用場面: ログ解析、センサーデータ
     └── ツール: Kafka + ML, Flink, Spark Streaming
```

### コード例8: FastAPIによるモデルサービング

```python
"""
モデルサービングの基本実装

実行方法:
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Model Serving API",
    description="機械学習モデルの推論API",
    version="1.0.0"
)

# モデルの読込
try:
    model = joblib.load("models/production_model.joblib")
    logger.info("モデルを読み込みました")
except FileNotFoundError:
    logger.warning("モデルファイルが見つかりません。ダミーモデルを使用します。")
    model = None

class PredictionRequest(BaseModel):
    """推論リクエストのスキーマ"""
    features: List[float] = Field(..., description="特徴量のリスト")
    request_id: Optional[str] = Field(None, description="リクエストID（追跡用）")

    class Config:
        json_schema_extra = {
            "example": {
                "features": [25.0, 50000.0, 3.5, 1.0],
                "request_id": "req-001"
            }
        }

class PredictionResponse(BaseModel):
    """推論レスポンスのスキーマ"""
    prediction: int
    probability: float
    request_id: Optional[str]
    model_version: str
    timestamp: str

@app.get("/health")
def health_check():
    """ヘルスチェックエンドポイント"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """推論エンドポイント"""
    if model is None:
        raise HTTPException(status_code=503, detail="モデルが読み込まれていません")

    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features).max())

        response = PredictionResponse(
            prediction=prediction,
            probability=probability,
            request_id=request.request_id,
            model_version="1.0.0",
            timestamp=datetime.now().isoformat()
        )

        logger.info(f"Prediction: {prediction}, Prob: {probability:.4f}, "
                     f"Request: {request.request_id}")
        return response

    except Exception as e:
        logger.error(f"推論エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(requests: List[PredictionRequest]):
    """バッチ推論エンドポイント"""
    if model is None:
        raise HTTPException(status_code=503, detail="モデルが読み込まれていません")

    features_batch = np.array([req.features for req in requests])
    predictions = model.predict(features_batch)
    probabilities = model.predict_proba(features_batch).max(axis=1)

    return [
        PredictionResponse(
            prediction=int(pred),
            probability=float(prob),
            request_id=req.request_id,
            model_version="1.0.0",
            timestamp=datetime.now().isoformat()
        )
        for pred, prob, req in zip(predictions, probabilities, requests)
    ]
```

---

## 比較表

### AI解析手法の選択ガイド

| データの性質 | 推奨手法 | 代表的アルゴリズム | 出力 | 典型的なユースケース |
|---|---|---|---|---|
| ラベル付き・連続値 | 教師あり回帰 | 線形回帰, XGBoost | 数値 | 価格予測, 需要予測 |
| ラベル付き・離散値 | 教師あり分類 | ロジスティック回帰, RF | カテゴリ | 不正検知, 診断 |
| ラベルなし・構造発見 | 教師なしクラスタリング | K-means, DBSCAN | クラスタID | 顧客セグメント |
| ラベルなし・次元圧縮 | 教師なし次元削減 | PCA, t-SNE, UMAP | 低次元表現 | 可視化, ノイズ除去 |
| 逐次的意思決定 | 強化学習 | Q学習, PPO | 行動方策 | ゲームAI, ロボット制御 |
| 大量テキスト | NLP | BERT, GPT | テキスト | 翻訳, 要約, 分類 |
| 画像・動画 | コンピュータビジョン | CNN, ViT | 検出/分類 | 自動運転, 医療画像 |
| 時系列 | 時系列予測 | ARIMA, LSTM, Prophet | 将来値 | 需要予測, 株価予測 |
| 少量ラベル | 半教師あり/転移学習 | Self-Training, Fine-tune | 分類/回帰 | ラベルコスト高の場面 |

### プロジェクト規模別ツール選択

| 項目 | 小規模 (〜1万行) | 中規模 (〜100万行) | 大規模 (1億行〜) |
|---|---|---|---|
| データ処理 | pandas | pandas + Dask / Polars | Spark / Polars |
| モデル学習 | scikit-learn | XGBoost / LightGBM | 分散学習 (Horovod) |
| 実験管理 | ノートブック | MLflow | Kubeflow / Vertex AI |
| デプロイ | Flask / FastAPI | Docker + Cloud Run | Kubernetes + Seldon |
| 計算環境 | ローカル | GPU (Colab / SageMaker) | マルチGPUクラスタ |
| データバージョン管理 | Git | DVC | Delta Lake / Iceberg |
| 特徴量ストア | なし | Feast (ローカル) | Feast / Tecton |
| モニタリング | 手動 | Prometheus + Grafana | Evidently + 専用ツール |
| コスト目安 | 無料〜数千円/月 | 数千〜数万円/月 | 数十万円/月〜 |

### MLフレームワークの比較

| フレームワーク | 得意分野 | 学習曲線 | スケーラビリティ | コミュニティ |
|---|---|---|---|---|
| scikit-learn | 古典的ML全般 | 低い | 中程度 | 非常に大きい |
| XGBoost | 勾配ブースティング | 低い | 高い | 大きい |
| LightGBM | 高速勾配ブースティング | 低い | 非常に高い | 大きい |
| CatBoost | カテゴリ特徴量 | 低い | 高い | 中程度 |
| PyTorch | 深層学習 | 中程度 | 非常に高い | 非常に大きい |
| TensorFlow | 深層学習・本番運用 | 高い | 非常に高い | 非常に大きい |
| JAX | 高性能数値計算 | 高い | 非常に高い | 成長中 |
| statsmodels | 統計モデル | 中程度 | 低い | 中程度 |

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

### アンチパターン4: ベースラインなしの開発

```python
# BAD: いきなり複雑なモデルから始める
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(256, 128, 64))
model.fit(X_train, y_train)

# GOOD: まずベースラインを確立してから改善
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

# Step 1: 最も単純なベースライン
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_score = baseline.score(X_test, y_test)
print(f"ベースライン (最頻値予測): {baseline_score:.4f}")

# Step 2: シンプルな線形モデル
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_score = lr.score(X_test, y_test)
print(f"ロジスティック回帰: {lr_score:.4f}")

# Step 3: 必要に応じて複雑なモデルへ
# （ベースラインを上回ることを確認しながら進める）
```

### アンチパターン5: 本番環境の考慮不足

```
PoC環境と本番環境のギャップ:

  PoC                          本番
  ├── バッチ処理               ├── リアルタイム処理
  ├── 静的データ               ├── ストリーミングデータ
  ├── ノートブック             ├── API/マイクロサービス
  ├── ローカルファイル         ├── データベース/クラウド
  ├── 手動実行                 ├── 自動パイプライン
  └── 精度のみ評価             └── レイテンシ/スループットも

対策:
  - PoCの段階からパイプラインを意識したコード設計
  - モデルの推論速度のベンチマーク
  - データドリフトの検出計画
  - ロールバック手順の準備
```

---

## トラブルシューティング

### よくある問題と対処法

| 問題 | 症状 | 原因 | 対処法 |
|---|---|---|---|
| モデルが学習しない | 精度がベースラインと同じ | 特徴量に情報がない | EDAで特徴量の有用性を確認 |
| 過学習 | Train高/Test低 | モデルが複雑すぎる | 正則化、データ増加、特徴量削減 |
| スコアが不安定 | CVのFold間でばらつき大 | データ量不足 or 分布偏り | データ増加、Stratified CV |
| メモリ不足 | MemoryError | データが大きすぎる | Dask使用、分割処理、dtype最適化 |
| 再現性がない | 毎回結果が異なる | シード未固定 | set_global_seed()を使用 |
| デプロイ後に精度低下 | 本番で性能劣化 | データドリフト | モニタリング、再学習パイプライン |
| 訓練が遅い | 数時間かかる | ハイパーパラメータ過多 | Optuna + Pruning、LightGBM |

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

### Q5: どのプログラミング言語を使うべきか？

**A:** Pythonが事実上の標準。エコシステム（scikit-learn, PyTorch, TensorFlow, pandas等）が圧倒的に充実している。R は統計分析に強みがあるが、MLエンジニアリングではPythonが優位。本番システムのパフォーマンスが重要な場合はRustやGoでの推論部分の実装も選択肢。

### Q6: クラウドとオンプレミスの選択基準は？

**A:** クラウド推奨のケース: (1) スケーラビリティが必要、(2) GPU利用が一時的、(3) チームが分散。オンプレミス推奨のケース: (1) データのセキュリティ要件が厳しい、(2) 常時GPUが必要、(3) ランニングコストが重要。ハイブリッドアプローチ（開発はクラウド、本番はオンプレ）も一般的。

---

## まとめ

| 項目 | 要点 |
|---|---|
| ライフサイクル | CRISP-DM: ビジネス理解→データ理解→準備→モデリング→評価→デプロイの反復 |
| 学習パラダイム | 教師あり（回帰・分類）、教師なし（クラスタ・次元削減）、強化学習 |
| プロジェクト設計 | 再現性（シード固定）、モジュール性（設定駆動）、品質（データ検証） |
| EDA | 基本統計、相関分析、分布確認、ターゲットとの関係性を系統的に調査 |
| 実験管理 | MLflowで実験を追跡。パラメータ、メトリクス、アーティファクトを記録 |
| ツール選択 | 規模に応じて段階的にスケールアップ。最小限のツールから始める |
| デプロイ | バッチ/リアルタイム/エッジの選択。FastAPIが入門に最適 |
| 成功の鍵 | 技術より問題定義とデータ品質。ベースラインから始めて段階的に改善 |

---

## 次に読むべきガイド

- [01-data-preprocessing.md](./01-data-preprocessing.md) — データ前処理の具体的手法
- [02-ml-basics.md](./02-ml-basics.md) — 機械学習の基礎理論と評価指標
- [03-python-ml-stack.md](./03-python-ml-stack.md) — Python ML開発環境の構築

---

## 参考文献

1. **Aurelien Geron** "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" 3rd Edition, O'Reilly Media, 2022
2. **Pete Chapman et al.** "CRISP-DM 1.0: Step-by-step data mining guide" SPSS Inc., 2000 — https://www.datascience-pm.com/crisp-dm-2/
3. **Google** "Rules of Machine Learning: Best Practices for ML Engineering" — https://developers.google.com/machine-learning/guides/rules-of-ml
4. **MLOps Community** "MLOps Principles" — https://ml-ops.org/content/mlops-principles
5. **MLflow** "Open source platform for the machine learning lifecycle" — https://mlflow.org/
6. **FastAPI** "Modern, fast web framework for building APIs" — https://fastapi.tiangolo.com/

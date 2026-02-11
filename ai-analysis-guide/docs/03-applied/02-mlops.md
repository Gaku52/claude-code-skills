# MLOps — 機械学習の本番運用基盤

> 実験管理からモデルデプロイ、本番監視まで、ML モデルを継続的に価値へ変換するためのエンジニアリングプラクティスを体系的に学ぶ。

---

## この章で学ぶこと

1. **実験管理** — パラメータ・メトリクス・アーティファクトを再現可能な形で記録し、チームで共有する手法
2. **モデルデプロイ** — コンテナ化、サービングパターン、CI/CD パイプラインによる安全なリリース戦略
3. **本番監視** — データドリフト・モデル劣化を検知し、再学習トリガーを自動化するフィードバックループ

---

## 1. MLOps の全体像

### 1.1 MLOps 成熟度モデル

```
+-------------------------------------------------------------------+
|  Level 0: 手動            全工程を手作業で実行                      |
|  Level 1: パイプライン化   学習・評価を自動パイプラインで実行        |
|  Level 2: CI/CD           モデルの継続的学習・デプロイを自動化       |
|  Level 3: フルループ       監視→再学習→デプロイの完全自動化          |
+-------------------------------------------------------------------+
```

### 1.2 MLOps ライフサイクル全体図

```
+----------+     +----------+     +----------+     +----------+
|  データ   | --> |  実験     | --> |  デプロイ  | --> |  監視    |
|  収集/    |     |  管理/    |     |  サービング |     |  ドリフト |
|  前処理   |     |  学習     |     |  CI/CD    |     |  検知    |
+----------+     +----------+     +----------+     +----------+
     ^                                                    |
     |              フィードバックループ                     |
     +----------------------------------------------------+
```

### 1.3 DevOps と MLOps の比較

| 観点 | DevOps | MLOps |
|------|--------|-------|
| バージョン管理対象 | コード | コード + データ + モデル |
| テスト | ユニット/統合テスト | + データ検証 + モデル品質テスト |
| デプロイ対象 | アプリケーション | モデル + サービングインフラ |
| 監視対象 | レイテンシ、エラー率 | + データドリフト、モデル劣化 |
| 再デプロイトリガー | コード変更 | コード変更 + データ変化 + 精度低下 |

---

## 2. 実験管理

### 2.1 MLflow による実験トラッキング

```python
# コード例 1: MLflow で実験を記録する
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# 実験を開始
mlflow.set_experiment("churn-prediction-v2")

with mlflow.start_run(run_name="rf-baseline"):
    # ハイパーパラメータを記録
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42
    }
    mlflow.log_params(params)

    # モデル学習
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # メトリクスを記録
    y_pred = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    # モデルをアーティファクトとして保存
    mlflow.sklearn.log_model(model, "model")

    # 特徴量重要度のプロットを保存
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.barh(feature_names, model.feature_importances_)
    fig.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
```

### 2.2 DVC によるデータバージョン管理

```bash
# コード例 2: DVC でデータとモデルをバージョン管理する

# 初期化
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"

# リモートストレージの設定
dvc remote add -d myremote s3://my-bucket/dvc-store

# データファイルをDVC管理下に追加
dvc add data/train.csv
git add data/train.csv.dvc data/.gitignore
git commit -m "Add training data v1"

# パイプラインを定義 (dvc.yaml)
# stages:
#   preprocess:
#     cmd: python src/preprocess.py
#     deps: [data/train.csv, src/preprocess.py]
#     outs: [data/processed/]
#   train:
#     cmd: python src/train.py
#     deps: [data/processed/, src/train.py]
#     outs: [models/model.pkl]
#     metrics: [metrics.json]

# パイプライン実行
dvc repro

# 実験の比較
dvc metrics diff
```

### 2.3 実験管理ツール比較

| 機能 | MLflow | Weights & Biases | DVC | Neptune |
|------|--------|-------------------|-----|---------|
| 実験トラッキング | ○ | ○ | ○ | ○ |
| モデルレジストリ | ○ | ○ | △ | ○ |
| データバージョン管理 | △ | △ | ○ | △ |
| 可視化ダッシュボード | ○ | ○ (高機能) | △ | ○ |
| チームコラボ | ○ | ○ | ○ (Git連携) | ○ |
| セルフホスト | ○ | △ (有料) | ○ | △ |
| コスト | 無料 (OSS) | フリーミアム | 無料 (OSS) | フリーミアム |

---

## 3. モデルデプロイ

### 3.1 サービングパターン

```
パターン A: バッチ推論
+--------+     +----------+     +---------+     +--------+
| データ  | --> | バッチ    | --> | 結果    | --> | DB /   |
| ストア  |     | ジョブ    |     | テーブル |     | キャッシュ|
+--------+     +----------+     +---------+     +--------+
                  (定期実行)

パターン B: リアルタイム推論
+--------+     +-----------+     +--------+
| クライ  | --> | 推論API   | --> | レスポ  |
| アント  |     | (REST/gRPC)|    | ンス    |
+--------+     +-----------+     +--------+
                  |
              +--------+
              | モデル  |
              | サーバ  |
              +--------+

パターン C: ストリーミング推論
+--------+     +--------+     +-----------+     +--------+
| イベント | --> | Kafka  | --> | 推論      | --> | 出力   |
| ソース  |     | 等     |     | ワーカー   |     | トピック |
+--------+     +--------+     +-----------+     +--------+
```

### 3.2 コンテナ化とサービング

```python
# コード例 3: FastAPI + Docker でモデルをサービングする

# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Churn Prediction API", version="1.0.0")

# 起動時にモデルをロード
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = joblib.load("/app/models/model.pkl")

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array(request.features).reshape(1, -1)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0].max()

    return PredictionResponse(
        prediction=int(prediction),
        probability=float(probability)
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}
```

```dockerfile
# コード例 4: Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.3 CI/CD パイプライン (GitHub Actions)

```yaml
# コード例 5: .github/workflows/mlops-pipeline.yml
name: MLOps Pipeline

on:
  push:
    paths:
      - 'src/**'
      - 'data/**'
      - 'dvc.yaml'

jobs:
  train-and-evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Pull DVC data
        run: dvc pull
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Run pipeline
        run: dvc repro

      - name: Evaluate model
        run: |
          python src/evaluate.py --threshold 0.85
          # 精度が閾値を下回ったら失敗

      - name: Register model
        if: success()
        run: |
          python src/register_model.py \
            --model-name churn-prediction \
            --stage Production

  deploy:
    needs: train-and-evaluate
    runs-on: ubuntu-latest
    steps:
      - name: Build and push Docker image
        run: |
          docker build -t myregistry/churn-api:${{ github.sha }} .
          docker push myregistry/churn-api:${{ github.sha }}

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/churn-api \
            churn-api=myregistry/churn-api:${{ github.sha }}
```

---

## 4. 本番監視

### 4.1 データドリフト検知

```python
# コード例 6: Evidently でデータドリフトを検知する
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import pandas as pd

# 学習時のデータ (リファレンス) と本番データ (カレント)
reference_data = pd.read_csv("data/train.csv")
current_data = pd.read_csv("data/production_latest.csv")

# ドリフトレポートを生成
report = Report(metrics=[
    DataDriftPreset(),
    TargetDriftPreset(),
])
report.run(
    reference_data=reference_data,
    current_data=current_data
)

# HTML レポートとして保存
report.save_html("drift_report.html")

# プログラムから結果を取得
result = report.as_dict()
drift_detected = result["metrics"][0]["result"]["dataset_drift"]

if drift_detected:
    print("WARNING: データドリフトを検知しました")
    # Slack通知、再学習トリガーなど
    trigger_retraining_pipeline()
```

### 4.2 モデル劣化の監視ダッシュボード

```python
# コード例 7: Prometheus + Grafana 用のメトリクスを公開する
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# メトリクス定義
PREDICTION_COUNT = Counter(
    'model_predictions_total',
    'Total number of predictions',
    ['model_version', 'prediction_class']
)
PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)
MODEL_ACCURACY = Gauge(
    'model_accuracy_rolling',
    'Rolling accuracy over last N predictions'
)
DATA_DRIFT_SCORE = Gauge(
    'data_drift_score',
    'Data drift score (0=no drift, 1=full drift)',
    ['feature_name']
)

# メトリクスサーバー起動 (ポート 8001)
start_http_server(8001)

def predict_with_monitoring(model, features, model_version="v1.0"):
    start_time = time.time()

    prediction = model.predict(features)

    # レイテンシを記録
    latency = time.time() - start_time
    PREDICTION_LATENCY.observe(latency)

    # 予測カウントを記録
    PREDICTION_COUNT.labels(
        model_version=model_version,
        prediction_class=str(prediction)
    ).inc()

    return prediction
```

---

## 5. アンチパターン

### アンチパターン 1: 「ノートブック本番投入」

```
[誤り] Jupyter Notebook をそのまま本番環境で cron 実行する

問題点:
- 再現性がない（セルの実行順序依存、グローバル変数汚染）
- テスト不可能
- バージョン管理が困難（JSON差分が読めない）
- エラーハンドリングが不十分

[正解] ノートブックは探索・プロトタイプ用途に限定し、
       本番コードは .py モジュールに変換する

  notebook (探索) --> Python モジュール --> テスト --> パイプライン --> デプロイ
```

### アンチパターン 2: 「モデルを置いて放置」

```
[誤り] モデルを一度デプロイしたら監視せずに放置する

実際に起きる問題:
1. データドリフト: ユーザー行動が変化し、学習時と入力分布が乖離
2. コンセプトドリフト: 予測対象自体の定義が変化
3. 無言の劣化: エラーは出ないが精度が徐々に低下

  デプロイ時精度: 92% --> 3ヶ月後: 85% --> 6ヶ月後: 72%
  (誰も気づかないまま劣化)

[正解] 監視 + 自動再学習のフィードバックループを構築する
  - 定期的なドリフト検知
  - 精度閾値を設定し、下回ったらアラート
  - 自動再学習パイプラインをトリガー
```

---

## 6. FAQ

### Q1: 小規模チームでも MLOps は必要ですか？

**A:** はい、ただし成熟度レベルを段階的に上げることが重要です。最低限として以下を推奨します。

- **Level 0 からの脱却**: MLflow などで実験を記録する（数時間で導入可能）
- **データバージョン管理**: DVC で学習データを管理する
- **モデルレジストリ**: どのモデルが本番にあるか追跡する

小規模チームでは Level 1（パイプライン化）まで到達できれば十分な場合が多いです。

### Q2: モデルの A/B テストはどう実装しますか？

**A:** 一般的なアプローチは以下の通りです。

1. **トラフィック分割**: ロードバランサーやサービスメッシュ（Istio 等）でトラフィックを分割
2. **シャドーモード**: 新モデルに本番トラフィックのコピーを流し、結果を記録するだけ（ユーザーには影響なし）
3. **カナリアリリース**: 段階的にトラフィック比率を増やす（5% → 25% → 50% → 100%）

統計的に有意な差が確認できるまでテストを継続し、メトリクス（精度、ビジネス KPI）で判断します。

### Q3: GPU サーバーのコストを最適化するには？

**A:** 以下の戦略が有効です。

- **スポットインスタンス**: 学習ジョブにスポット/プリエンプティブルインスタンスを使用（最大 90% コスト削減）
- **オートスケーリング**: 推論サーバーをトラフィックに応じてスケールイン/アウト
- **モデル最適化**: 量子化（INT8）、蒸留、プルーニングで推論コストを削減
- **バッチ推論**: リアルタイム性が不要な場合はバッチ処理で GPU 利用効率を向上

---

## 7. まとめ

| カテゴリ | ポイント | 代表ツール |
|----------|----------|------------|
| 実験管理 | パラメータ・メトリクス・アーティファクトを記録 | MLflow, W&B |
| データ管理 | データのバージョン管理と再現性確保 | DVC, LakeFS |
| モデルレジストリ | モデルのライフサイクル管理 | MLflow, Vertex AI |
| パイプライン | 学習・評価の自動化 | Kubeflow, Airflow |
| サービング | リアルタイム/バッチ推論の提供 | TF Serving, Triton |
| CI/CD | モデルの継続的デプロイ | GitHub Actions, Jenkins |
| 監視 | ドリフト検知・精度監視 | Evidently, Grafana |
| フィードバック | 自動再学習トリガー | Kubeflow Pipelines |

---

## 次に読むべきガイド

- [責任ある AI](./03-responsible-ai.md) — 公平性・説明可能性・プライバシーの実装
- [データ前処理と特徴量エンジニアリング](../02-core/01-feature-engineering.md) — MLOps パイプラインに組み込む前処理設計
- [システム設計ガイド](../../../system-design-guide/docs/01-fundamentals/00-overview.md) — MLOps インフラの設計原則

---

## 参考文献

1. Sculley, D. et al. (2015). "Hidden Technical Debt in Machine Learning Systems." *Advances in Neural Information Processing Systems 28 (NIPS 2015)*. Google. https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems
2. Google Cloud. (2023). "MLOps: Continuous delivery and automation pipelines in machine learning." *Google Cloud Architecture Center*. https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
3. Kreuzberger, D., Kühl, N., & Hirschl, S. (2023). "Machine Learning Operations (MLOps): Overview, Definition, and Architecture." *IEEE Access, 11*, 31866-31879. https://ieeexplore.ieee.org/document/10081336

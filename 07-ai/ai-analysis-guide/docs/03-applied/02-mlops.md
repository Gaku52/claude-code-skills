# MLOps — 機械学習の本番運用基盤

> 実験管理からモデルデプロイ、本番監視まで、ML モデルを継続的に価値へ変換するためのエンジニアリングプラクティスを体系的に学ぶ。

---

## この章で学ぶこと

1. **実験管理** — パラメータ・メトリクス・アーティファクトを再現可能な形で記録し、チームで共有する手法
2. **モデルデプロイ** — コンテナ化、サービングパターン、CI/CD パイプラインによる安全なリリース戦略
3. **本番監視** — データドリフト・モデル劣化を検知し、再学習トリガーを自動化するフィードバックループ
4. **フィーチャーストア** — 特徴量の一元管理と再利用性の確保
5. **インフラストラクチャ** — Kubernetes、クラウドマネージドサービスの活用パターン

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

各レベルの詳細と到達に必要な要素を以下に示す。

| レベル | 特徴 | 必要なツール/プラクティス | チーム規模 |
|--------|------|--------------------------|-----------|
| Level 0 | Jupyter Notebookで手動実験、手動デプロイ | Git、手動テスト | 1-2人 |
| Level 1 | 学習パイプラインの自動化、実験トラッキング | MLflow、DVC、Airflow | 3-5人 |
| Level 2 | CI/CDでモデルを自動テスト・デプロイ | GitHub Actions、Docker、K8s | 5-10人 |
| Level 3 | ドリフト検知→自動再学習→自動デプロイの完全ループ | Evidently、Kubeflow、Feature Store | 10人以上 |

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
| パイプライン複雑度 | 比較的単純（ビルド→テスト→デプロイ） | 複雑（データ取得→前処理→学習→評価→デプロイ→監視） |
| 成果物の特性 | 決定論的（同じコード=同じ結果） | 確率的（同じコードでもデータで結果が変わる） |
| ロールバック | コードを戻す | モデル+データ+設定の整合性を維持して戻す |

### 1.4 MLOps の技術スタック全体像

```
┌─────────────────────────────────────────────────────────────┐
│                     MLOps プラットフォーム                     │
├─────────────┬─────────────┬─────────────┬───────────────────┤
│ データ管理   │ 実験管理     │ モデル管理   │ 本番運用          │
├─────────────┼─────────────┼─────────────┼───────────────────┤
│ DVC         │ MLflow      │ Model       │ Monitoring        │
│ LakeFS      │ W&B         │ Registry    │ (Evidently,       │
│ Delta Lake  │ Neptune     │ (MLflow,    │  Grafana,         │
│ Great       │ CometML     │  Vertex AI) │  Prometheus)      │
│ Expectations│             │             │                   │
├─────────────┼─────────────┼─────────────┼───────────────────┤
│ Feature     │ パイプライン  │ サービング   │ フィードバック     │
│ Store       │             │             │                   │
├─────────────┼─────────────┼─────────────┼───────────────────┤
│ Feast       │ Kubeflow    │ TF Serving  │ Auto Retrain      │
│ Tecton      │ Airflow     │ Triton      │ Pipeline          │
│ Hopsworks   │ Prefect     │ TorchServe  │ A/B Testing       │
│             │ Dagster     │ BentoML     │ Shadow Deploy     │
└─────────────┴─────────────┴─────────────┴───────────────────┘
```

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

### 2.2 MLflow モデルレジストリの活用

```python
# コード例 2: MLflow Model Registry でモデルのライフサイクルを管理する
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# モデルを登録
model_name = "churn-prediction"
result = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=model_name
)

# モデルのステージを遷移
# None → Staging → Production → Archived
client.transition_model_version_stage(
    name=model_name,
    version=result.version,
    stage="Staging",
    archive_existing_versions=False
)

# Staging 環境でテストを実施
staging_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/Staging"
)
staging_predictions = staging_model.predict(X_staging_test)
staging_accuracy = accuracy_score(y_staging_test, staging_predictions)

print(f"Staging モデルの精度: {staging_accuracy:.4f}")

# 精度が閾値を超えたら Production に昇格
if staging_accuracy >= 0.85:
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True  # 旧バージョンをアーカイブ
    )
    print(f"モデル v{result.version} を Production に昇格")
else:
    print(f"精度不足: {staging_accuracy:.4f} < 0.85")

# モデルのメタデータを追加
client.update_model_version(
    name=model_name,
    version=result.version,
    description="RandomForest baseline with 100 estimators. "
                "Trained on 2024-01 data. F1=0.87"
)

# 全バージョンの一覧を取得
for mv in client.search_model_versions(f"name='{model_name}'"):
    print(f"  v{mv.version}: stage={mv.current_stage}, "
          f"created={mv.creation_timestamp}")
```

### 2.3 Weights & Biases (W&B) による高度な実験管理

```python
# コード例 3: W&B でハイパーパラメータスイープを実行する
import wandb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# スイープの設定
sweep_config = {
    "method": "bayes",  # ベイズ最適化
    "metric": {
        "name": "val_f1",
        "goal": "maximize"
    },
    "parameters": {
        "n_estimators": {"min": 50, "max": 500},
        "max_depth": {"values": [3, 5, 7, 10, 15]},
        "learning_rate": {"distribution": "log_uniform_values",
                          "min": 0.001, "max": 0.3},
        "subsample": {"min": 0.5, "max": 1.0},
        "min_samples_split": {"values": [2, 5, 10, 20]},
    }
}

def train_sweep():
    """W&B スイープのトレーニング関数"""
    with wandb.init() as run:
        config = wandb.config

        model = GradientBoostingClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            min_samples_split=config.min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)

        # 検証メトリクス
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        wandb.log({
            "val_accuracy": accuracy_score(y_val, y_pred),
            "val_f1": f1_score(y_val, y_pred),
            "val_auc": roc_auc_score(y_val, y_prob),
        })

        # 特徴量重要度テーブル
        importance_table = wandb.Table(
            columns=["feature", "importance"],
            data=[[name, imp] for name, imp in
                  zip(feature_names, model.feature_importances_)]
        )
        wandb.log({"feature_importance": importance_table})

# スイープを実行
sweep_id = wandb.sweep(sweep_config, project="churn-prediction")
wandb.agent(sweep_id, function=train_sweep, count=50)  # 50回試行
```

### 2.4 DVC によるデータバージョン管理

```bash
# コード例 4: DVC でデータとモデルをバージョン管理する

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

### 2.5 DVC パイプライン定義の詳細

```yaml
# コード例 5: dvc.yaml の完全なパイプライン定義
stages:
  data_validation:
    cmd: python src/validate_data.py
    deps:
      - data/raw/
      - src/validate_data.py
    outs:
      - reports/data_validation.html
    metrics:
      - metrics/data_quality.json:
          cache: false

  preprocess:
    cmd: python src/preprocess.py --config configs/preprocess.yaml
    deps:
      - data/raw/
      - src/preprocess.py
      - configs/preprocess.yaml
    params:
      - preprocess.normalize
      - preprocess.feature_selection
    outs:
      - data/processed/train.parquet
      - data/processed/test.parquet
      - artifacts/preprocessor.pkl

  feature_engineering:
    cmd: python src/feature_engineering.py
    deps:
      - data/processed/train.parquet
      - data/processed/test.parquet
      - src/feature_engineering.py
    outs:
      - data/features/train_features.parquet
      - data/features/test_features.parquet
    params:
      - features.window_sizes
      - features.aggregations

  train:
    cmd: python src/train.py --config configs/model.yaml
    deps:
      - data/features/train_features.parquet
      - src/train.py
      - configs/model.yaml
    params:
      - model.type
      - model.hyperparameters
    outs:
      - models/model.pkl
      - models/model_metadata.json
    metrics:
      - metrics/train_metrics.json:
          cache: false
    plots:
      - plots/training_curve.csv:
          x: epoch
          y: loss

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - data/features/test_features.parquet
      - models/model.pkl
      - src/evaluate.py
    metrics:
      - metrics/eval_metrics.json:
          cache: false
    plots:
      - plots/confusion_matrix.csv:
          template: confusion
          x: predicted
          y: actual
      - plots/roc_curve.csv:
          x: fpr
          y: tpr
```

### 2.6 実験管理ツール比較

| 機能 | MLflow | Weights & Biases | DVC | Neptune | CometML |
|------|--------|-------------------|-----|---------|---------|
| 実験トラッキング | ○ | ○ | ○ | ○ | ○ |
| モデルレジストリ | ○ | ○ | △ | ○ | ○ |
| データバージョン管理 | △ | △ | ○ | △ | △ |
| 可視化ダッシュボード | ○ | ○ (高機能) | △ | ○ | ○ |
| チームコラボ | ○ | ○ | ○ (Git連携) | ○ | ○ |
| セルフホスト | ○ | △ (有料) | ○ | △ | △ |
| ハイパーパラメータ探索 | △ | ○ (Sweep) | △ | ○ | ○ |
| コスト | 無料 (OSS) | フリーミアム | 無料 (OSS) | フリーミアム | フリーミアム |

---

## 3. フィーチャーストア

### 3.1 フィーチャーストアの概念

```
フィーチャーストアの役割:

  データソース        フィーチャーストア            利用者
  +----------+      +------------------+       +-----------+
  | DB       | ---> |                  | ----> | 学習      |
  | ログ     |      | 特徴量定義       |       | パイプライン|
  | API      | ---> | (Feature View)   | ----> +-----------+
  | ストリーム|      |                  |       | 推論      |
  +----------+      | オフラインストア  |       | サーバー   |
                    | (バッチ)         | ----> +-----------+
                    |                  |
                    | オンラインストア  |  ← 低レイテンシ
                    | (リアルタイム)   |       アクセス
                    +------------------+

  主要メリット:
  1. 学習と推論で同じ特徴量定義を共有（Training-Serving Skew の防止）
  2. 特徴量の再利用（チーム間でのシェア）
  3. ポイントインタイム結合（データリーク防止）
  4. オンライン/オフラインの自動同期
```

### 3.2 Feast によるフィーチャーストア構築

```python
# コード例 6: Feast でフィーチャーストアを構築する
from feast import FeatureStore, Entity, Feature, FeatureView
from feast import FileSource, ValueType
from feast.types import Float32, Int64, String
from datetime import timedelta

# データソースの定義
customer_source = FileSource(
    path="data/customer_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# エンティティの定義
customer = Entity(
    name="customer_id",
    value_type=ValueType.INT64,
    description="顧客を一意に識別するID",
)

# フィーチャービューの定義
customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=90),  # 特徴量の有効期限
    schema=[
        Feature(name="total_purchases", dtype=Int64),
        Feature(name="avg_order_value", dtype=Float32),
        Feature(name="days_since_last_purchase", dtype=Int64),
        Feature(name="login_count_30d", dtype=Int64),
        Feature(name="support_tickets_count", dtype=Int64),
        Feature(name="customer_segment", dtype=String),
    ],
    source=customer_source,
    online=True,
    tags={"team": "data-science", "version": "v2"},
)

# フィーチャーストアの初期化
store = FeatureStore(repo_path="feature_repo/")

# オフラインでの学習用データ取得（ポイントインタイム結合）
import pandas as pd

entity_df = pd.DataFrame({
    "customer_id": [1001, 1002, 1003, 1004, 1005],
    "event_timestamp": pd.to_datetime([
        "2024-01-15", "2024-01-15", "2024-01-15",
        "2024-01-15", "2024-01-15"
    ]),
})

training_data = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "customer_features:total_purchases",
        "customer_features:avg_order_value",
        "customer_features:days_since_last_purchase",
        "customer_features:login_count_30d",
    ],
).to_df()

print(f"学習データ形状: {training_data.shape}")
print(training_data.head())

# オンラインでのリアルタイム推論用データ取得
online_features = store.get_online_features(
    features=[
        "customer_features:total_purchases",
        "customer_features:avg_order_value",
        "customer_features:days_since_last_purchase",
    ],
    entity_rows=[{"customer_id": 1001}],
).to_dict()

print(f"リアルタイム特徴量: {online_features}")
```

---

## 4. モデルデプロイ

### 4.1 サービングパターン

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

パターン D: エッジ推論
+--------+     +-----------+     +--------+
| センサー | --> | エッジ    | --> | ローカル |
| カメラ  |     | デバイス   |     | アクション|
+--------+     | (TFLite/  |     +--------+
               | ONNX)     |         |
               +-----------+         v
                                  クラウドに
                                  結果送信
```

### 4.2 コンテナ化とサービング

```python
# コード例 7: FastAPI + Docker でモデルをサービングする

# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import logging
import time
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

logger = logging.getLogger(__name__)

# メトリクス定義
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["status"]
)
REQUEST_LATENCY = Histogram(
    "prediction_request_latency_seconds",
    "Prediction request latency",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# モデルのライフサイクル管理
model = None
model_metadata = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションの起動/終了処理"""
    global model, model_metadata
    logger.info("モデルをロード中...")
    model = joblib.load("/app/models/model.pkl")
    model_metadata = {
        "model_name": "churn-prediction",
        "version": "v2.1",
        "trained_at": "2024-01-15T10:30:00Z",
        "features": ["feature_0", "feature_1", "feature_2"]
    }
    logger.info(f"モデルをロード完了: {model_metadata['version']}")
    yield
    logger.info("アプリケーションを終了")

app = FastAPI(
    title="Churn Prediction API",
    version="2.1.0",
    lifespan=lifespan
)

class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=1,
        description="入力特徴量のリスト"
    )
    request_id: str = Field(
        default=None,
        description="リクエストの追跡用ID"
    )

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    request_id: str = None
    latency_ms: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        X = np.array(request.features).reshape(1, -1)
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0].max()
        latency_ms = (time.time() - start_time) * 1000

        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(latency_ms / 1000)

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            model_version=model_metadata["version"],
            request_id=request.request_id,
            latency_ms=round(latency_ms, 2)
        )
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        logger.error(f"推論エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(requests: list[PredictionRequest]):
    """バッチ推論エンドポイント"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    X = np.array([r.features for r in requests])
    predictions = model.predict(X)
    probabilities = model.predict_proba(X).max(axis=1)
    latency_ms = (time.time() - start_time) * 1000

    return {
        "predictions": [
            {
                "prediction": int(pred),
                "probability": float(prob),
                "request_id": req.request_id,
            }
            for pred, prob, req in zip(predictions, probabilities, requests)
        ],
        "total_latency_ms": round(latency_ms, 2),
        "count": len(requests),
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_metadata.get("version", "unknown"),
    }

@app.get("/metrics")
async def metrics():
    """Prometheus メトリクスエンドポイント"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/model/info")
async def model_info():
    """モデルのメタデータを返す"""
    return model_metadata
```

```dockerfile
# コード例 8: マルチステージ Dockerfile
# ---- ビルドステージ ----
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- 実行ステージ ----
FROM python:3.11-slim

# セキュリティ: 非rootユーザーで実行
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app
COPY --from=builder /install /usr/local
COPY app/ ./app/
COPY models/ ./models/

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER appuser
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--log-level", "info"]
```

### 4.3 Kubernetes デプロイメント

```yaml
# コード例 9: Kubernetes マニフェスト
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-prediction-api
  labels:
    app: churn-prediction
    version: v2.1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: churn-prediction
  template:
    metadata:
      labels:
        app: churn-prediction
        version: v2.1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: churn-api
          image: myregistry/churn-api:v2.1
          ports:
            - containerPort: 8000
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "1000m"
              memory: "1Gi"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          env:
            - name: MODEL_PATH
              value: "/app/models/model.pkl"
            - name: LOG_LEVEL
              value: "info"
---
apiVersion: v1
kind: Service
metadata:
  name: churn-prediction-service
spec:
  selector:
    app: churn-prediction
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: churn-prediction-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: churn-prediction-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: prediction_request_latency_seconds
        target:
          type: AverageValue
          averageValue: "200m"  # 200msを超えたらスケールアウト
```

### 4.4 CI/CD パイプライン (GitHub Actions)

```yaml
# コード例 10: .github/workflows/mlops-pipeline.yml
name: MLOps Pipeline

on:
  push:
    paths:
      - 'src/**'
      - 'data/**'
      - 'dvc.yaml'

jobs:
  data-validation:
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

      - name: Validate data with Great Expectations
        run: python src/validate_data.py

  train-and-evaluate:
    needs: data-validation
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

      - name: Upload metrics as artifact
        uses: actions/upload-artifact@v4
        with:
          name: model-metrics
          path: metrics/

  deploy:
    needs: train-and-evaluate
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Build and push Docker image
        run: |
          docker build -t myregistry/churn-api:${{ github.sha }} .
          docker push myregistry/churn-api:${{ github.sha }}

      - name: Deploy to Kubernetes (Canary)
        run: |
          # カナリアデプロイ: 10%のトラフィックを新バージョンに
          kubectl set image deployment/churn-api-canary \
            churn-api=myregistry/churn-api:${{ github.sha }}

          # 5分間の監視
          sleep 300

          # メトリクスを確認してフル展開を決定
          python src/check_canary_metrics.py

      - name: Full rollout
        if: success()
        run: |
          kubectl set image deployment/churn-api \
            churn-api=myregistry/churn-api:${{ github.sha }}
```

### 4.5 BentoML によるモデルパッケージング

```python
# コード例 11: BentoML でモデルをパッケージ化する
import bentoml
from bentoml.io import JSON, NumpyNdarray
import numpy as np

# モデルを BentoML に保存
saved_model = bentoml.sklearn.save_model(
    "churn_classifier",
    model,
    signatures={
        "predict": {"batchable": True, "batch_dim": 0},
        "predict_proba": {"batchable": True, "batch_dim": 0},
    },
    metadata={
        "accuracy": 0.92,
        "dataset_version": "v2.1",
        "training_date": "2024-01-15",
    },
    custom_objects={
        "preprocessor": preprocessor,
    }
)
print(f"保存: {saved_model}")

# Bentoサービスの定義
runner = bentoml.sklearn.get("churn_classifier:latest").to_runner()
svc = bentoml.Service("churn_prediction_service", runners=[runner])

@svc.api(input=NumpyNdarray(), output=JSON())
async def predict(input_array: np.ndarray) -> dict:
    prediction = await runner.predict.async_run(input_array)
    probability = await runner.predict_proba.async_run(input_array)
    return {
        "prediction": prediction.tolist(),
        "probability": probability.max(axis=1).tolist(),
    }

# bentofile.yaml
"""
service: "service:svc"
include:
  - "*.py"
python:
  requirements_txt: "./requirements.txt"
docker:
  python_version: "3.11"
  distro: "debian"
"""

# ビルドとデプロイ
# bentoml build
# bentoml containerize churn_prediction_service:latest
```

---

## 5. 本番監視

### 5.1 データドリフト検知

```python
# コード例 12: Evidently でデータドリフトを検知する
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset, TargetDriftPreset,
    DataQualityPreset, ClassificationPreset
)
from evidently.metrics import (
    DataDriftTable, DatasetDriftMetric,
    ColumnDriftMetric, DatasetMissingValuesMetric
)
import pandas as pd

# 学習時のデータ (リファレンス) と本番データ (カレント)
reference_data = pd.read_csv("data/train.csv")
current_data = pd.read_csv("data/production_latest.csv")

# 詳細なドリフトレポートを生成
report = Report(metrics=[
    DataDriftPreset(),
    TargetDriftPreset(),
    DataQualityPreset(),
    ColumnDriftMetric(column_name="age"),
    ColumnDriftMetric(column_name="income"),
    DatasetMissingValuesMetric(),
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
drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]

print(f"データセットドリフト: {'検知' if drift_detected else '未検知'}")
print(f"ドリフトした特徴量の割合: {drift_share:.1%}")

if drift_detected:
    print("WARNING: データドリフトを検知しました")
    # ドリフトした特徴量の詳細を取得
    drifted_columns = [
        col for col, info in
        result["metrics"][0]["result"]["drift_by_columns"].items()
        if info["drift_detected"]
    ]
    print(f"ドリフトした特徴量: {drifted_columns}")
    trigger_retraining_pipeline()
```

### 5.2 Evidently によるリアルタイム監視

```python
# コード例 13: Evidently のテストスイートで品質ゲートを構築する
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestShareOfDriftedColumns,
    TestColumnDrift,
    TestMeanInNSigmas,
    TestShareOfMissingValues,
    TestNumberOfColumnsWithMissingValues,
    TestColumnShareOfMissingValues,
)

# テストスイートの定義
test_suite = TestSuite(tests=[
    # ドリフトテスト
    TestShareOfDriftedColumns(lt=0.3),  # 30%未満の特徴量がドリフト
    TestColumnDrift(column_name="age"),
    TestColumnDrift(column_name="income"),

    # データ品質テスト
    TestShareOfMissingValues(lt=0.05),  # 欠損率5%未満
    TestColumnShareOfMissingValues(
        column_name="customer_id", eq=0  # customer_id は欠損不可
    ),

    # 統計テスト
    TestMeanInNSigmas(column_name="age", n=3),  # 平均が3σ以内
    TestMeanInNSigmas(column_name="income", n=3),
])

test_suite.run(
    reference_data=reference_data,
    current_data=current_data
)

# 結果の判定
test_results = test_suite.as_dict()
all_passed = all(
    test["status"] == "SUCCESS"
    for test in test_results["tests"]
)

if not all_passed:
    failed_tests = [
        test for test in test_results["tests"]
        if test["status"] == "FAIL"
    ]
    print(f"失敗したテスト数: {len(failed_tests)}")
    for test in failed_tests:
        print(f"  - {test['name']}: {test['description']}")

    # CI/CD パイプラインを停止するか、アラートを発火
    send_alert_to_slack(failed_tests)
```

### 5.3 モデル劣化の監視ダッシュボード

```python
# コード例 14: Prometheus + Grafana 用のメトリクスを公開する
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import numpy as np
from collections import deque

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
FEATURE_DISTRIBUTION = Histogram(
    'feature_distribution',
    'Distribution of input features',
    ['feature_name'],
    buckets=[-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
)

# メトリクスサーバー起動 (ポート 8001)
start_http_server(8001)

# ローリングウィンドウで精度を追跡
class RollingAccuracyTracker:
    def __init__(self, window_size=1000):
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)

    def update(self, prediction, actual):
        self.predictions.append(prediction)
        self.actuals.append(actual)
        if len(self.predictions) >= 100:  # 最低100件で計算
            accuracy = np.mean(
                np.array(list(self.predictions)) == np.array(list(self.actuals))
            )
            MODEL_ACCURACY.set(accuracy)

tracker = RollingAccuracyTracker(window_size=1000)

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

    # 特徴量の分布を記録
    for i, feature_name in enumerate(feature_names):
        FEATURE_DISTRIBUTION.labels(
            feature_name=feature_name
        ).observe(float(features[0][i]))

    return prediction
```

### 5.4 自動再学習パイプライン

```python
# コード例 15: ドリフト検知から自動再学習までのパイプライン
import schedule
import time
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AutoRetrainingPipeline:
    """ドリフト検知ベースの自動再学習パイプライン"""

    def __init__(self, config):
        self.config = config
        self.last_retrain_date = None
        self.consecutive_drift_count = 0

    def check_and_retrain(self):
        """定期的にドリフトを検知し、必要に応じて再学習する"""
        logger.info(f"ドリフトチェック開始: {datetime.now()}")

        # 1. 直近の本番データを取得
        current_data = self.fetch_production_data(
            since=datetime.now() - timedelta(days=1)
        )

        # 2. ドリフト検知
        drift_result = self.detect_drift(current_data)

        if drift_result["drift_detected"]:
            self.consecutive_drift_count += 1
            logger.warning(
                f"ドリフト検知 (連続{self.consecutive_drift_count}回): "
                f"ドリフト特徴量={drift_result['drifted_features']}"
            )

            # 3. 連続で閾値を超えたら再学習をトリガー
            if self.consecutive_drift_count >= self.config["retrain_threshold"]:
                logger.info("自動再学習をトリガーします")
                self.trigger_retraining()
                self.consecutive_drift_count = 0
        else:
            self.consecutive_drift_count = 0
            logger.info("ドリフト未検知")

        # 4. 精度ベースのチェック
        if drift_result.get("accuracy_below_threshold", False):
            logger.warning(
                f"精度低下検知: {drift_result['current_accuracy']:.4f} "
                f"< {self.config['accuracy_threshold']}"
            )
            self.trigger_retraining()

    def trigger_retraining(self):
        """再学習パイプラインを実行する"""
        logger.info("再学習パイプラインを開始")

        # Kubeflow Pipeline をトリガー
        import kfp
        client = kfp.Client(host=self.config["kubeflow_host"])
        run = client.create_run_from_pipeline_func(
            self.retrain_pipeline,
            arguments={
                "data_start_date": (
                    datetime.now() - timedelta(days=90)
                ).strftime("%Y-%m-%d"),
                "data_end_date": datetime.now().strftime("%Y-%m-%d"),
                "model_name": self.config["model_name"],
            },
        )
        logger.info(f"再学習ジョブ開始: run_id={run.run_id}")
        self.last_retrain_date = datetime.now()

    def fetch_production_data(self, since):
        """本番データベースから直近データを取得"""
        import pandas as pd
        query = f"""
        SELECT * FROM predictions
        WHERE created_at >= '{since.isoformat()}'
        ORDER BY created_at DESC
        LIMIT 10000
        """
        return pd.read_sql(query, self.config["db_connection"])

    def detect_drift(self, current_data):
        """ドリフト検知を実行"""
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )
        result = report.as_dict()
        return {
            "drift_detected": result["metrics"][0]["result"]["dataset_drift"],
            "drifted_features": [
                col for col, info in
                result["metrics"][0]["result"]["drift_by_columns"].items()
                if info["drift_detected"]
            ],
            "drift_share": result["metrics"][0]["result"]["share_of_drifted_columns"],
        }

# 定期実行のスケジューリング
pipeline = AutoRetrainingPipeline(config={
    "retrain_threshold": 3,  # 3回連続ドリフトで再学習
    "accuracy_threshold": 0.85,
    "model_name": "churn-prediction",
    "kubeflow_host": "https://kubeflow.example.com",
    "db_connection": "postgresql://...",
})

# 毎日9時にドリフトチェック
schedule.every().day.at("09:00").do(pipeline.check_and_retrain)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 6. データ検証とテスト

### 6.1 Great Expectations によるデータ検証

```python
# コード例 16: Great Expectations でデータ品質ゲートを構築する
import great_expectations as gx

# データコンテキストの初期化
context = gx.get_context()

# データソースの設定
datasource = context.sources.add_pandas("my_datasource")
data_asset = datasource.add_dataframe_asset("training_data")

# Expectation Suite の定義
suite = context.add_expectation_suite("training_data_quality")

# Expectation の追加
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(column="customer_id")
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="age", min_value=0, max_value=150
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="income", min_value=0, max_value=10000000
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeInSet(
        column="gender", value_set=["M", "F", "Other", "Unknown"]
    )
)
suite.add_expectation(
    gx.expectations.ExpectTableRowCountToBeBetween(
        min_value=1000, max_value=10000000
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnProportionOfUniqueValuesToBeBetween(
        column="customer_id", min_value=0.99, max_value=1.0
    )
)

# バリデーション実行
batch_request = data_asset.build_batch_request(dataframe=training_df)
results = context.run_validation_operator(
    "action_list_operator",
    assets_to_validate=[batch_request],
)

if not results["success"]:
    failed = [r for r in results["results"] if not r["success"]]
    print(f"データ検証失敗: {len(failed)}件")
    for f in failed:
        print(f"  - {f['expectation_config']['expectation_type']}: "
              f"{f['result']['unexpected_count']}件の違反")
    raise ValueError("データ品質基準を満たしていません")
```

### 6.2 モデルテスト戦略

```python
# コード例 17: ML モデルの包括的テスト
import pytest
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score

class TestModelQuality:
    """モデルの品質を検証するテストスイート"""

    @pytest.fixture
    def model(self):
        return joblib.load("models/model.pkl")

    @pytest.fixture
    def test_data(self):
        import pandas as pd
        df = pd.read_csv("data/test.csv")
        X = df.drop("target", axis=1)
        y = df["target"]
        return X, y

    def test_minimum_accuracy(self, model, test_data):
        """最低精度の確認"""
        X, y = test_data
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        assert accuracy >= 0.85, f"精度が閾値未満: {accuracy:.4f} < 0.85"

    def test_minimum_f1_score(self, model, test_data):
        """最低F1スコアの確認"""
        X, y = test_data
        y_pred = model.predict(X)
        f1 = f1_score(y, y_pred, average="weighted")
        assert f1 >= 0.80, f"F1が閾値未満: {f1:.4f} < 0.80"

    def test_no_data_leakage(self, model, test_data):
        """テストデータに対して過度に高い精度でないことを確認"""
        X, y = test_data
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        assert accuracy < 0.99, f"精度が高すぎる（データリークの疑い）: {accuracy:.4f}"

    def test_prediction_latency(self, model, test_data):
        """推論レイテンシの確認"""
        import time
        X, y = test_data
        single_sample = X.iloc[[0]]

        start = time.time()
        for _ in range(100):
            model.predict(single_sample)
        avg_latency = (time.time() - start) / 100

        assert avg_latency < 0.01, f"推論が遅すぎる: {avg_latency:.4f}s > 0.01s"

    def test_fairness_across_groups(self, model, test_data):
        """グループ間の公平性を確認"""
        X, y = test_data
        y_pred = model.predict(X)

        # グループ別精度の確認（例: 性別）
        if "gender" in X.columns:
            for group in X["gender"].unique():
                mask = X["gender"] == group
                group_acc = accuracy_score(y[mask], y_pred[mask])
                assert group_acc >= 0.75, (
                    f"グループ '{group}' の精度が低すぎる: {group_acc:.4f}"
                )

    def test_model_robustness(self, model, test_data):
        """ノイズに対するロバスト性を確認"""
        X, y = test_data
        baseline_pred = model.predict(X)

        # 小さなノイズを追加
        noise = np.random.normal(0, 0.01, X.shape)
        X_noisy = X + noise
        noisy_pred = model.predict(X_noisy)

        # 予測の安定性を確認
        stability = np.mean(baseline_pred == noisy_pred)
        assert stability >= 0.95, (
            f"ノイズに対する安定性が低い: {stability:.4f} < 0.95"
        )

    def test_prediction_distribution(self, model, test_data):
        """予測分布が想定範囲内か確認"""
        X, y = test_data
        y_pred = model.predict(X)

        # 各クラスの予測割合をチェック
        for cls in np.unique(y):
            pred_rate = np.mean(y_pred == cls)
            true_rate = np.mean(y == cls)
            assert abs(pred_rate - true_rate) < 0.2, (
                f"クラス{cls}の予測率が大きく乖離: "
                f"予測={pred_rate:.3f}, 実際={true_rate:.3f}"
            )
```

---

## 7. A/Bテストとカナリアデプロイ

### 7.1 モデルの A/B テスト実装

```python
# コード例 18: Istio を使ったトラフィック分割
"""
# Istio VirtualService でトラフィックを分割

apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: churn-prediction-vs
spec:
  hosts:
    - churn-prediction
  http:
    - match:
        - headers:
            x-model-version:
              exact: "v2"
      route:
        - destination:
            host: churn-prediction
            subset: v2
    - route:
        - destination:
            host: churn-prediction
            subset: v1
          weight: 90
        - destination:
            host: churn-prediction
            subset: v2
          weight: 10
"""

# A/B テストの統計的判定
import numpy as np
from scipy import stats

class ABTestAnalyzer:
    """モデル A/B テストの統計分析"""

    def __init__(self, alpha=0.05, min_sample_size=1000):
        self.alpha = alpha
        self.min_sample_size = min_sample_size

    def calculate_sample_size(self, baseline_rate, mde, alpha=0.05, power=0.8):
        """
        必要なサンプルサイズを計算する。
        mde: 最小検出可能効果 (Minimum Detectable Effect)
        """
        from statsmodels.stats.power import NormalIndPower
        analysis = NormalIndPower()
        effect_size = mde / np.sqrt(baseline_rate * (1 - baseline_rate))
        sample_size = analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            alternative="two-sided"
        )
        return int(np.ceil(sample_size))

    def analyze(self, control_conversions, control_total,
                treatment_conversions, treatment_total):
        """A/B テストの結果を分析する"""
        control_rate = control_conversions / control_total
        treatment_rate = treatment_conversions / treatment_total

        # z検定
        pooled_rate = (control_conversions + treatment_conversions) / \
                      (control_total + treatment_total)
        se = np.sqrt(pooled_rate * (1 - pooled_rate) *
                     (1/control_total + 1/treatment_total))
        z_stat = (treatment_rate - control_rate) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # 効果量
        lift = (treatment_rate - control_rate) / control_rate

        # 信頼区間
        ci_95 = stats.norm.interval(
            0.95,
            loc=treatment_rate - control_rate,
            scale=se
        )

        return {
            "control_rate": control_rate,
            "treatment_rate": treatment_rate,
            "lift": lift,
            "lift_pct": f"{lift:.2%}",
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "confidence_interval_95": ci_95,
            "recommendation": (
                "Treatment を採用" if p_value < self.alpha and lift > 0
                else "Control を維持"
            ),
        }

# 使用例
analyzer = ABTestAnalyzer(alpha=0.05)

# 必要サンプルサイズの計算
n = analyzer.calculate_sample_size(
    baseline_rate=0.05,  # 基準コンバージョン率 5%
    mde=0.005,           # 0.5%の改善を検出したい
)
print(f"必要サンプルサイズ（各群）: {n:,}")

# A/B テスト結果の分析
result = analyzer.analyze(
    control_conversions=520, control_total=10000,   # 旧モデル
    treatment_conversions=580, treatment_total=10000  # 新モデル
)
print(f"コントロール率: {result['control_rate']:.4f}")
print(f"トリートメント率: {result['treatment_rate']:.4f}")
print(f"リフト: {result['lift_pct']}")
print(f"p値: {result['p_value']:.4f}")
print(f"統計的有意: {result['significant']}")
print(f"推奨: {result['recommendation']}")
```

---

## 8. アンチパターン

### アンチパターン 1: 「ノートブック本番投入」

```
[誤り] Jupyter Notebook をそのまま本番環境で cron 実行する

問題点:
- 再現性がない（セルの実行順序依存、グローバル変数汚染）
- テスト不可能
- バージョン管理が困難（JSON差分が読めない）
- エラーハンドリングが不十分
- メモリリークのリスク

[正解] ノートブックは探索・プロトタイプ用途に限定し、
       本番コードは .py モジュールに変換する

  notebook (探索) --> Python モジュール --> テスト --> パイプライン --> デプロイ

具体的な移行手順:
  1. ノートブックのコードを関数/クラスに分割
  2. 設定値を外部ファイル (YAML/JSON) に分離
  3. ユニットテストを追加
  4. ロギングとエラーハンドリングを追加
  5. CLIインターフェースを作成 (argparse/click)
  6. CI/CDパイプラインに組み込み
```

### アンチパターン 2: 「モデルを置いて放置」

```
[誤り] モデルを一度デプロイしたら監視せずに放置する

実際に起きる問題:
1. データドリフト: ユーザー行動が変化し、学習時と入力分布が乖離
2. コンセプトドリフト: 予測対象自体の定義が変化
3. 無言の劣化: エラーは出ないが精度が徐々に低下
4. 季節性: 年末年始やセール時期で入力パターンが変化

  デプロイ時精度: 92% --> 3ヶ月後: 85% --> 6ヶ月後: 72%
  (誰も気づかないまま劣化)

[正解] 監視 + 自動再学習のフィードバックループを構築する
  - 定期的なドリフト検知
  - 精度閾値を設定し、下回ったらアラート
  - 自動再学習パイプラインをトリガー
```

### アンチパターン 3: 「Training-Serving Skew」

```
[誤り] 学習時と推論時で異なる前処理を適用する

具体例:
  学習時: StandardScaler → model.fit(scaled_data)
  推論時: MinMaxScaler → model.predict(scaled_data)  # 異なるスケーラー!

  学習時: 特徴量 A,B,C,D を使用
  推論時: 特徴量 A,B,C を使用（D を忘れた）

  学習時: Python 3.10 + pandas 1.5
  推論時: Python 3.11 + pandas 2.0（挙動が微妙に異なる）

[正解]
  1. 前処理をモデルと一緒に保存する（sklearn Pipeline, preprocessor pickle）
  2. フィーチャーストアで特徴量定義を統一する
  3. Docker イメージでランタイム環境を固定する
  4. テストで学習/推論の一致を検証する
```

### アンチパターン 4: 「巨大モデルの無計画デプロイ」

```
[誤り] 10GBのモデルをそのまま推論APIにデプロイする

問題点:
- コールドスタートに数分かかる
- メモリ使用量が膨大
- スケールアウトのコストが高い
- レイテンシが要件を満たさない

[正解] モデル最適化を行ってからデプロイする
  - 量子化 (INT8/FP16): サイズ50-75%削減、速度2-4倍向上
  - プルーニング: 不要なパラメータを削除
  - 蒸留 (Distillation): 小さいモデルに知識を転写
  - ONNX変換: フレームワーク最適化の恩恵
  - バッチ推論: リアルタイム性不要なら一括処理
```

---

## 9. FAQ

### Q1: 小規模チームでも MLOps は必要ですか？

**A:** はい、ただし成熟度レベルを段階的に上げることが重要です。最低限として以下を推奨します。

- **Level 0 からの脱却**: MLflow などで実験を記録する（数時間で導入可能）
- **データバージョン管理**: DVC で学習データを管理する
- **モデルレジストリ**: どのモデルが本番にあるか追跡する

小規模チームでは Level 1（パイプライン化）まで到達できれば十分な場合が多いです。

**段階的導入ロードマップ（推奨）:**

| 期間 | 施策 | 工数 |
|------|------|------|
| 1週目 | MLflow 導入、実験記録を開始 | 4時間 |
| 2週目 | DVC でデータバージョン管理 | 8時間 |
| 1ヶ月目 | モデルの Docker コンテナ化 | 16時間 |
| 2ヶ月目 | CI/CD で自動テスト・デプロイ | 24時間 |
| 3ヶ月目 | Evidently でドリフト監視 | 16時間 |

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
- **マルチテナンシー**: NVIDIA Triton の Dynamic Batching で GPU 利用率を最大化
- **Serverless GPU**: Modal、RunPod 等のサーバーレス GPU で使用分のみ課金

### Q4: オンプレミスとクラウドのどちらで MLOps を構築すべきですか？

**A:** 判断基準は以下の通りです。

| 観点 | オンプレミス | クラウド |
|------|-------------|---------|
| 初期コスト | 高い（GPU 購入） | 低い（従量課金） |
| 運用コスト | 低い（長期運用） | 使用量依存 |
| スケーラビリティ | 限定的 | 高い |
| データセキュリティ | 完全制御 | 共有責任 |
| 立ち上げ速度 | 遅い | 速い |

多くのチームは「クラウドで始めて、規模が大きくなったらハイブリッド」が現実的です。

### Q5: MLOps プラットフォームを自前構築するべきですか？

**A:** 基本的にはマネージドサービスの活用を推奨します。自前構築は運用コストが高く、本来の ML 開発に充てるべきリソースを消費します。

- **AWS**: SageMaker（学習・デプロイ統合）
- **GCP**: Vertex AI（AutoML + カスタムモデル）
- **Azure**: Azure ML（Enterprise 統合）
- **OSS**: Kubeflow + MLflow + Feast の組み合わせ

---

## 10. まとめ

| カテゴリ | ポイント | 代表ツール |
|----------|----------|------------|
| 実験管理 | パラメータ・メトリクス・アーティファクトを記録 | MLflow, W&B |
| データ管理 | データのバージョン管理と再現性確保 | DVC, LakeFS |
| データ検証 | データ品質ゲートの自動化 | Great Expectations, Evidently |
| フィーチャーストア | 特徴量の一元管理と再利用 | Feast, Tecton |
| モデルレジストリ | モデルのライフサイクル管理 | MLflow, Vertex AI |
| パイプライン | 学習・評価の自動化 | Kubeflow, Airflow, Dagster |
| サービング | リアルタイム/バッチ推論の提供 | TF Serving, Triton, BentoML |
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
3. Kreuzberger, D., Kuhl, N., & Hirschl, S. (2023). "Machine Learning Operations (MLOps): Overview, Definition, and Architecture." *IEEE Access, 11*, 31866-31879. https://ieeexplore.ieee.org/document/10081336
4. Breck, E. et al. (2019). "Data Validation for Machine Learning." *MLSys 2019*. Google. https://mlsys.org/Conferences/2019/doc/2019/167.pdf
5. Polyzotis, N. et al. (2019). "Data Lifecycle Challenges in Production Machine Learning." *SIGMOD Record, 47*(2). https://doi.org/10.1145/3299887.3299891

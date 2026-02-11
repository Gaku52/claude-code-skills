# AWS Lambda 基礎

> サーバーを一切管理せずにコードを実行できる AWS Lambda の基本概念、関数の作成方法、トリガー設定、IAM ロール、環境変数、レイヤーまでを体系的に学ぶ。

---

## この章で学ぶこと

1. **Lambda 関数の作成とデプロイ** -- ランタイム選択からコードのアップロード、テスト実行までの一連の流れを理解する
2. **トリガーと IAM ロールの設計** -- API Gateway・S3・SQS などのイベントソースと、最小権限の実行ロールを正しく構成する
3. **環境変数とレイヤーの活用** -- 設定の外部化と共通ライブラリの再利用でメンテナンス性を高める手法を身につける

---

## 1. Lambda とは何か

### 1.1 サーバーレスコンピューティングの位置づけ

```
従来型 (EC2)              コンテナ (ECS/EKS)          サーバーレス (Lambda)
+-----------------+      +-----------------+      +-----------------+
| アプリケーション |      | アプリケーション |      | アプリケーション |
+-----------------+      +-----------------+      +-----------------+
| ミドルウェア     |      | コンテナランタイム|      |                 |
+-----------------+      +-----------------+      |  AWS が全て管理  |
| OS              |      |  OS (共有)       |      |                 |
+-----------------+      +-----------------+      +-----------------+
| ハードウェア     |      | ハードウェア     |      | ハードウェア     |
+-----------------+      +-----------------+      +-----------------+
  ユーザー管理範囲:広      ユーザー管理範囲:中      ユーザー管理範囲:狭
```

Lambda はイベント駆動型のコンピューティングサービスであり、以下の特徴を持つ。

- **プロビジョニング不要** -- サーバーの起動・停止・スケーリングは AWS が自動管理
- **実行時間課金** -- リクエスト数と実行時間(1ms 単位)で課金
- **自動スケーリング** -- 同時実行数は需要に応じて自動的に増減
- **幅広い言語サポート** -- Python、Node.js、Java、Go、.NET、Ruby、カスタムランタイム

### 1.2 Lambda の実行モデル

```
イベントソース           Lambda サービス              実行環境
+------------+        +------------------+        +------------------+
|            |  呼出  |                  |  配置  |  実行環境 (MicroVM)|
| API Gateway| -----> |  Lambda Control  | -----> |  +-------------+ |
| S3         |        |  Plane           |        |  | ランタイム  | |
| SQS        |        |                  |        |  | + ユーザー  | |
| EventBridge|        +------------------+        |  |   コード    | |
+------------+               |                    |  +-------------+ |
                             |  ログ送信          +------------------+
                             v                           |
                    +------------------+                  | メトリクス
                    |  CloudWatch Logs |                  v
                    +------------------+          +------------------+
                                                  | CloudWatch       |
                                                  | Metrics          |
                                                  +------------------+
```

---

## 2. Lambda 関数の作成

### 2.1 対応ランタイム一覧

| ランタイム | 識別子 | サポート状況 | 主な用途 |
|-----------|--------|-------------|---------|
| Python 3.12 | `python3.12` | GA | データ処理、API バックエンド |
| Node.js 20.x | `nodejs20.x` | GA | API バックエンド、リアルタイム処理 |
| Java 21 | `java21` | GA | エンタープライズ、バッチ処理 |
| Go (provided.al2023) | `provided.al2023` | GA | 高性能処理 |
| .NET 8 | `dotnet8` | GA | Windows 連携、エンタープライズ |
| Ruby 3.3 | `ruby3.3` | GA | スクリプト、Webhook |
| カスタムランタイム | `provided.al2023` | GA | Rust、PHP など任意の言語 |

### 2.2 Python で Hello World

```python
# lambda_function.py
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    Lambda のエントリポイント。

    Parameters:
        event (dict): トリガーから渡されるイベントデータ
        context (LambdaContext): 実行コンテキスト情報

    Returns:
        dict: API Gateway 互換のレスポンス
    """
    logger.info(f"Received event: {json.dumps(event)}")

    name = event.get("queryStringParameters", {}).get("name", "World")

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({
            "message": f"Hello, {name}!",
            "requestId": context.aws_request_id,
            "remainingTime": context.get_remaining_time_in_millis()
        })
    }
```

### 2.3 Node.js での実装例

```javascript
// index.mjs (ES Modules)
export const handler = async (event, context) => {
  console.log("Event:", JSON.stringify(event, null, 2));

  const name = event.queryStringParameters?.name || "World";

  return {
    statusCode: 200,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
    },
    body: JSON.stringify({
      message: `Hello, ${name}!`,
      requestId: context.awsRequestId,
    }),
  };
};
```

### 2.4 AWS CLI による関数作成

```bash
# 1. デプロイパッケージ作成
zip function.zip lambda_function.py

# 2. Lambda 関数の作成
aws lambda create-function \
  --function-name my-hello-function \
  --runtime python3.12 \
  --role arn:aws:iam::123456789012:role/lambda-execution-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://function.zip \
  --timeout 30 \
  --memory-size 256 \
  --description "Hello World Lambda function"

# 3. テスト呼び出し
aws lambda invoke \
  --function-name my-hello-function \
  --payload '{"queryStringParameters": {"name": "AWS"}}' \
  --cli-binary-format raw-in-base64-out \
  output.json

cat output.json
```

### 2.5 Lambda のメモリとタイムアウト設定

| 設定項目 | 最小値 | 最大値 | デフォルト | 備考 |
|---------|--------|--------|-----------|------|
| メモリ | 128 MB | 10,240 MB | 128 MB | CPU は比例配分 |
| タイムアウト | 1 秒 | 900 秒 (15分) | 3 秒 | API Gateway 経由は 29 秒制限 |
| エフェメラルストレージ | 512 MB | 10,240 MB | 512 MB | /tmp 領域 |
| デプロイパッケージ | - | 50 MB (zip) / 250 MB (展開後) | - | レイヤー含む |
| コンテナイメージ | - | 10 GB | - | ECR イメージ使用時 |

```
メモリとCPUの関係:

メモリ        vCPU相当      適用シーン
128 MB   -->  ~0.08 vCPU    軽量なAPI応答
512 MB   -->  ~0.33 vCPU    一般的なAPI処理
1,024 MB -->  ~0.58 vCPU    データ変換
1,769 MB -->  1 vCPU        計算処理
3,008 MB -->  2 vCPU        画像処理
10,240 MB --> 6 vCPU        ML推論、大規模バッチ
```

---

## 3. トリガーとイベントソース

### 3.1 主要なイベントソース

```
+------------------+     同期呼出      +--------+
| API Gateway      | ----------------> |        |
| ALB              |                   |        |
| Lambda URL       |                   |        |
+------------------+                   |        |
                                       | Lambda |
+------------------+     非同期呼出    |  関数  |
| S3               | ----------------> |        |
| SNS              |                   |        |
| EventBridge      |                   |        |
| IoT              |                   |        |
+------------------+                   |        |
                                       |        |
+------------------+   ポーリングベース |        |
| SQS              | ----------------> |        |
| DynamoDB Streams | (Event Source     |        |
| Kinesis          |  Mapping)         |        |
+------------------+                   +--------+
```

### 3.2 呼び出しモデルの比較

| 呼び出しモデル | イベントソース例 | リトライ動作 | エラーハンドリング |
|---------------|----------------|-------------|-----------------|
| 同期 (RequestResponse) | API Gateway, ALB | 呼び出し元が制御 | 即座にエラー応答 |
| 非同期 (Event) | S3, SNS, EventBridge | 最大2回リトライ | DLQ / Destinations |
| ポーリング (Event Source Mapping) | SQS, Kinesis, DynamoDB | ソースにより異なる | バッチ失敗時の制御 |

### 3.3 API Gateway トリガーの設定

```bash
# REST API の作成と Lambda 統合
aws apigateway create-rest-api \
  --name "HelloAPI" \
  --description "Hello World API"

# Lambda パーミッション追加
aws lambda add-permission \
  --function-name my-hello-function \
  --statement-id apigateway-invoke \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:ap-northeast-1:123456789012:abc123/*"
```

---

## 4. IAM ロールの設計

### 4.1 実行ロールの構成要素

Lambda 関数に必要な IAM ロールは2つの部分から成る。

1. **信頼ポリシー (Trust Policy)** -- Lambda サービスがこのロールを引き受ける許可
2. **アクセス許可ポリシー (Permission Policy)** -- 関数が AWS リソースにアクセスする許可

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

### 4.2 最小権限のポリシー例

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:ap-northeast-1:123456789012:log-group:/aws/lambda/my-hello-function:*"
    },
    {
      "Sid": "DynamoDBAccess",
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:Query"
      ],
      "Resource": "arn:aws:dynamodb:ap-northeast-1:123456789012:table/MyTable"
    }
  ]
}
```

---

## 5. 環境変数

### 5.1 環境変数の設定と利用

```bash
# 環境変数の設定
aws lambda update-function-configuration \
  --function-name my-hello-function \
  --environment "Variables={
    DB_TABLE=users-table,
    LOG_LEVEL=INFO,
    REGION=ap-northeast-1,
    FEATURE_FLAG_NEW_UI=true
  }"
```

```python
# lambda_function.py -- 環境変数の読み取り
import os

TABLE_NAME = os.environ.get("DB_TABLE", "default-table")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING")
REGION = os.environ.get("REGION", "ap-northeast-1")
FEATURE_FLAG = os.environ.get("FEATURE_FLAG_NEW_UI", "false") == "true"

def lambda_handler(event, context):
    # TABLE_NAME を使って DynamoDB にアクセス
    import boto3
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(TABLE_NAME)
    # ...
```

### 5.2 環境変数の暗号化

```
環境変数の暗号化フロー:

設定時:
  平文 --> AWS KMS で暗号化 --> 暗号化された環境変数を保存

実行時:
  暗号化された環境変数 --> Lambda ランタイムが自動復号 --> 平文で利用可能

カスタムKMS利用時:
  Lambda 実行ロールに kms:Decrypt 権限が必要
```

| 暗号化方式 | 説明 | 追加設定 |
|-----------|------|---------|
| デフォルト暗号化 | AWS 管理キーで自動暗号化 | 不要 |
| カスタム KMS キー | 顧客管理キーで暗号化 | KMS キー ARN を指定 |
| ヘルパーによる暗号化 | 転送中の暗号化を追加 | Lambda コンソールで設定 |

---

## 6. Lambda レイヤー

### 6.1 レイヤーの仕組み

```
Lambda 関数のファイルシステム:

/opt/                      <-- レイヤーの展開先
  ├── python/              <-- Python ライブラリ
  │   └── lib/
  │       └── python3.12/
  │           └── site-packages/
  │               ├── requests/
  │               └── boto3/
  ├── nodejs/              <-- Node.js ライブラリ
  │   └── node_modules/
  └── bin/                 <-- カスタムバイナリ

/var/task/                 <-- 関数コード
  └── lambda_function.py

/tmp/                      <-- エフェメラルストレージ (512MB-10GB)
```

### 6.2 レイヤーの作成とアタッチ

```bash
# 1. レイヤー用のディレクトリ構造を作成
mkdir -p layer/python
pip install requests -t layer/python/

# 2. ZIP パッケージ作成
cd layer && zip -r ../my-layer.zip python/

# 3. レイヤーの公開
aws lambda publish-layer-version \
  --layer-name my-common-libs \
  --description "共通ライブラリ (requests等)" \
  --zip-file fileb://my-layer.zip \
  --compatible-runtimes python3.12 python3.11

# 4. 関数にレイヤーをアタッチ
aws lambda update-function-configuration \
  --function-name my-hello-function \
  --layers arn:aws:lambda:ap-northeast-1:123456789012:layer:my-common-libs:1
```

### 6.3 レイヤーの制限事項

| 項目 | 制限 |
|------|-----|
| 関数あたりの最大レイヤー数 | 5 |
| レイヤー含む合計展開サイズ | 250 MB |
| レイヤーバージョン数 | 無制限 |
| レイヤー共有 | 同一リージョン内、クロスアカウント可 |

---

## 7. アンチパターン

### 7.1 モノリシック Lambda

```
[悪い例] 1つのLambda関数に全機能を詰め込む

def lambda_handler(event, context):
    path = event["path"]
    if path == "/users":
        return handle_users(event)
    elif path == "/orders":
        return handle_orders(event)
    elif path == "/products":
        return handle_products(event)
    elif path == "/payments":
        return handle_payments(event)
    # ... 数十のルート
```

**問題点**: デプロイパッケージが巨大化し、コールドスタートが遅くなる。1つの変更で全機能に影響し、テストが困難になる。

**改善**: 機能ごとに個別の Lambda 関数を作成し、API Gateway のルーティングで振り分ける。

### 7.2 Lambda 内での同期的な待機

```python
# [悪い例] Lambda 内で長時間の同期待機
import time

def lambda_handler(event, context):
    # 外部APIを呼び、結果が出るまでポーリング
    job_id = start_external_job()
    while True:
        status = check_job_status(job_id)
        if status == "COMPLETE":
            break
        time.sleep(10)  # 10秒ごとにポーリング -- 実行時間を浪費
    return get_job_result(job_id)
```

**問題点**: 実行時間が長くなりコストが増大。タイムアウトのリスクも高まる。

**改善**: Step Functions でステートマシンを構成するか、コールバックパターンを利用する。

---

## 8. FAQ

### Q1. Lambda のコールドスタートとは何ですか？

Lambda 関数が初めて呼び出されるとき、または実行環境がリサイクルされた後に、新しい実行環境の初期化が必要になる。この初期化時間を「コールドスタート」と呼ぶ。Python/Node.js で数百ミリ秒、Java/.NET で数秒かかることがある。対策としては、Provisioned Concurrency の利用や、デプロイパッケージの軽量化が有効である。

### Q2. Lambda 関数の同時実行数に制限はありますか？

デフォルトではリージョンあたり 1,000 同時実行がソフトリミットとして設定されている。Service Quotas から引き上げをリクエストできる。また、関数単位で `ReservedConcurrentExecutions` を設定して、特定の関数が他の関数のキャパシティを奪わないよう制御できる。

### Q3. Lambda でデータベース接続をどう管理すべきですか？

RDS を利用する場合は、RDS Proxy を経由して接続プーリングを行うのが推奨される。Lambda 関数のハンドラ外(グローバルスコープ)でコネクションを初期化し、実行環境の再利用時にコネクションを使い回すパターンが基本となる。DynamoDB のような HTTP ベースのサービスであればコネクション管理の問題は発生しない。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| Lambda とは | サーバー管理不要のイベント駆動型コンピューティング |
| ランタイム | Python, Node.js, Java, Go, .NET, Ruby, カスタム |
| トリガー | 同期(API Gateway)、非同期(S3, SNS)、ポーリング(SQS, Kinesis) |
| IAM ロール | 信頼ポリシー + 最小権限のアクセス許可ポリシー |
| 環境変数 | 設定の外部化、KMS による暗号化サポート |
| レイヤー | 共通ライブラリの再利用、最大5レイヤーまでアタッチ可能 |
| 課金 | リクエスト数 + 実行時間(GB-秒) |

---

## 次に読むべきガイド

- [Lambda 応用](./01-lambda-advanced.md) -- コールドスタート最適化、Provisioned Concurrency、Step Functions
- [サーバーレスパターン](./02-serverless-patterns.md) -- API+Lambda+DynamoDB、イベント駆動アーキテクチャ
- [IAM 詳解](../08-security/00-iam-deep-dive.md) -- Lambda 実行ロールの高度な設計

---

## 参考文献

1. AWS 公式ドキュメント「AWS Lambda デベロッパーガイド」 https://docs.aws.amazon.com/lambda/latest/dg/
2. AWS Well-Architected Framework「サーバーレスアプリケーションレンズ」 https://docs.aws.amazon.com/wellarchitected/latest/serverless-applications-lens/
3. Jeremy Daly「Serverless Architectures on AWS, 2nd Edition」Manning Publications, 2024
4. AWS ブログ「Operating Lambda: Performance optimization」 https://aws.amazon.com/blogs/compute/operating-lambda-performance-optimization-part-1/

# S3 応用

> バージョニング、レプリケーション、S3 Select、Transfer Acceleration でプロダクションレベルの S3 運用を実現する

## この章で学ぶこと

1. バージョニングを活用してデータの世代管理と誤削除からの復旧ができる
2. クロスリージョンレプリケーション (CRR) で災害復旧と低レイテンシアクセスを実現できる
3. S3 Select、Transfer Acceleration、コスト最適化戦略を実装できる

---

## 1. バージョニング

### 1.1 バージョニングの仕組み

```
バージョニング有効時のオブジェクト管理

  PUT report.pdf (v1)
  +-------------------+
  | Key: report.pdf   |
  | Version: aaa111   | ← 現在のバージョン
  +-------------------+

  PUT report.pdf (v2) → 上書きではなく新バージョン作成
  +-------------------+
  | Key: report.pdf   |
  | Version: bbb222   | ← 現在のバージョン
  +-------------------+
  +-------------------+
  | Key: report.pdf   |
  | Version: aaa111   | ← 旧バージョン（保持）
  +-------------------+

  DELETE report.pdf → 削除マーカーを追加（データは残る）
  +-------------------+
  | Key: report.pdf   |
  | Delete Marker      | ← 現在のバージョン
  +-------------------+
  +-------------------+
  | Key: report.pdf   |
  | Version: bbb222   | ← 復元可能
  +-------------------+
  +-------------------+
  | Key: report.pdf   |
  | Version: aaa111   | ← 復元可能
  +-------------------+
```

### 1.2 コード例: バージョニングの設定と操作

```bash
# バージョニングを有効化
aws s3api put-bucket-versioning \
  --bucket my-app-bucket \
  --versioning-configuration Status=Enabled

# バージョン一覧の確認
aws s3api list-object-versions \
  --bucket my-app-bucket \
  --prefix report.pdf \
  --query '{Versions: Versions[].[Key,VersionId,LastModified,IsLatest], DeleteMarkers: DeleteMarkers[].[Key,VersionId]}'

# 特定バージョンの取得
aws s3api get-object \
  --bucket my-app-bucket \
  --key report.pdf \
  --version-id aaa111 \
  ./report-v1.pdf

# 削除マーカーを削除して復元
aws s3api delete-object \
  --bucket my-app-bucket \
  --key report.pdf \
  --version-id "DELETE_MARKER_VERSION_ID"

# 特定バージョンを完全削除
aws s3api delete-object \
  --bucket my-app-bucket \
  --key report.pdf \
  --version-id aaa111
```

### 1.3 コード例: Python でバージョン管理

```python
import boto3

s3 = boto3.client('s3')

def list_versions(bucket, key):
    """オブジェクトの全バージョンを一覧表示"""
    response = s3.list_object_versions(Bucket=bucket, Prefix=key)
    for version in response.get('Versions', []):
        print(f"Version: {version['VersionId']} | "
              f"Size: {version['Size']} | "
              f"Date: {version['LastModified']} | "
              f"Latest: {version['IsLatest']}")
    for marker in response.get('DeleteMarkers', []):
        print(f"Delete Marker: {marker['VersionId']} | "
              f"Date: {marker['LastModified']}")

def restore_version(bucket, key, version_id):
    """特定バージョンを復元（コピーして最新に）"""
    s3.copy_object(
        Bucket=bucket,
        Key=key,
        CopySource={'Bucket': bucket, 'Key': key, 'VersionId': version_id}
    )
    print(f"Restored {key} to version {version_id}")
```

---

## 2. レプリケーション

### 2.1 レプリケーションの種類

```
+--------------------------------------------+
| CRR (Cross-Region Replication)              |
| ソースとデスティネーションが異なるリージョン    |
| → 災害復旧、低レイテンシアクセス              |
+--------------------------------------------+

+--------------------------------------------+
| SRR (Same-Region Replication)               |
| ソースとデスティネーションが同じリージョン      |
| → ログ集約、テスト環境へのデータコピー         |
+--------------------------------------------+

  東京リージョン              大阪リージョン
  +----------------+         +------------------+
  | Source Bucket  | ------> | Destination      |
  | (ap-northeast-1)|  CRR   | Bucket           |
  | versioning: ON |         | (ap-northeast-3) |
  +----------------+         | versioning: ON   |
                             +------------------+
```

### 2.2 コード例: CRR の設定

```bash
# 前提: 両バケットでバージョニングが有効であること

# レプリケーション用 IAM ロールを作成
cat > replication-trust.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "s3.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

aws iam create-role \
  --role-name S3ReplicationRole \
  --assume-role-policy-document file://replication-trust.json

# レプリケーションルールを設定
aws s3api put-bucket-replication \
  --bucket source-bucket \
  --replication-configuration '{
  "Role": "arn:aws:iam::123456789012:role/S3ReplicationRole",
  "Rules": [
    {
      "ID": "ReplicateAll",
      "Status": "Enabled",
      "Priority": 1,
      "Filter": {"Prefix": ""},
      "Destination": {
        "Bucket": "arn:aws:s3:::destination-bucket",
        "StorageClass": "STANDARD_IA",
        "ReplicationTime": {
          "Status": "Enabled",
          "Time": {"Minutes": 15}
        },
        "Metrics": {
          "Status": "Enabled",
          "EventThreshold": {"Minutes": 15}
        }
      },
      "DeleteMarkerReplication": {"Status": "Enabled"}
    }
  ]
}'
```

### 2.3 レプリケーション比較表

| 機能 | CRR | SRR |
|------|-----|-----|
| リージョン | 異なるリージョン | 同一リージョン |
| ユースケース | DR、低レイテンシ | ログ集約、環境コピー |
| バージョニング | 必須 | 必須 |
| 既存オブジェクト | S3 Batch Replication | S3 Batch Replication |
| 削除マーカー | 選択可能 | 選択可能 |
| 料金 | リクエスト + データ転送 | リクエストのみ |

---

## 3. S3 Select と Glacier Select

### 3.1 S3 Select の仕組み

```
従来の方式:
  S3 ---全データ取得---> アプリ ---フィルタ---> 結果
  (100MB転送)

S3 Select:
  S3 ---SQLでフィルタ---> 結果のみ転送---> アプリ
  (1MB転送)

  → データ転送量を最大 99% 削減
  → 対応フォーマット: CSV, JSON, Parquet
```

### 3.2 コード例: S3 Select (Python)

```python
import boto3
import json

s3 = boto3.client('s3')

# CSV ファイルから特定条件のデータを抽出
def query_csv(bucket, key, sql):
    response = s3.select_object_content(
        Bucket=bucket,
        Key=key,
        ExpressionType='SQL',
        Expression=sql,
        InputSerialization={
            'CSV': {
                'FileHeaderInfo': 'USE',
                'RecordDelimiter': '\n',
                'FieldDelimiter': ','
            },
            'CompressionType': 'GZIP'
        },
        OutputSerialization={'JSON': {'RecordDelimiter': '\n'}}
    )

    records = []
    for event in response['Payload']:
        if 'Records' in event:
            records.append(event['Records']['Payload'].decode('utf-8'))
    return records

# 使用例: 2024年の東京のデータのみ取得
results = query_csv(
    'analytics-bucket',
    'data/sales-2024.csv.gz',
    "SELECT s.date, s.product, s.amount FROM s3object s WHERE s.region = 'Tokyo' AND s.amount > '10000'"
)

# JSON ファイルからクエリ
response = s3.select_object_content(
    Bucket='data-bucket',
    Key='logs/events.json',
    ExpressionType='SQL',
    Expression="SELECT s.timestamp, s.level, s.message FROM s3object s WHERE s.level = 'ERROR'",
    InputSerialization={'JSON': {'Type': 'LINES'}},
    OutputSerialization={'JSON': {'RecordDelimiter': '\n'}}
)
```

---

## 4. Transfer Acceleration

### 4.1 仕組み

```
通常のアップロード:
  クライアント (ブラジル) --インターネット--> S3 (東京)
  遅延: 高い (多数のホップ)

Transfer Acceleration:
  クライアント (ブラジル) --> CloudFront Edge (サンパウロ)
                                    |
                              AWS バックボーン (最適化)
                                    |
                                    v
                               S3 (東京)
  遅延: 低い (AWS 内部ネットワーク)
```

### 4.2 コード例: Transfer Acceleration の設定

```bash
# Transfer Acceleration を有効化
aws s3api put-bucket-accelerate-configuration \
  --bucket my-global-bucket \
  --accelerate-configuration Status=Enabled

# Acceleration エンドポイントでアップロード
aws s3 cp large-file.zip \
  s3://my-global-bucket/uploads/ \
  --endpoint-url https://my-global-bucket.s3-accelerate.amazonaws.com

# 速度比較ツール
# https://s3-accelerate-speedtest.s3-accelerate.amazonaws.com/en/accelerate-speed-comparsion.html
```

### 4.3 コード例: Python で Transfer Acceleration を使用

```python
import boto3
from boto3.s3.transfer import TransferConfig

# Acceleration エンドポイントを使用
s3 = boto3.client(
    's3',
    endpoint_url='https://s3-accelerate.amazonaws.com',
    config=boto3.session.Config(s3={'use_accelerate_endpoint': True})
)

# マルチパート設定（大容量ファイル向け）
config = TransferConfig(
    multipart_threshold=100 * 1024 * 1024,   # 100MB
    max_concurrency=10,
    multipart_chunksize=100 * 1024 * 1024,
    use_threads=True
)

s3.upload_file(
    'large-dataset.tar.gz',
    'my-global-bucket',
    'datasets/large-dataset.tar.gz',
    Config=config
)
```

---

## 5. コスト最適化

### 5.1 コスト最適化チェックリスト

```
S3 コスト最適化ピラミッド

           /\
          /  \  不要データ削除
         /    \  (ライフサイクル Expiration)
        /------\
       /        \  ストレージクラス最適化
      /          \  (Intelligent-Tiering / Glacier)
     /------------\
    /              \  リクエスト最適化
   /                \  (S3 Select, プレフィックス設計)
  /------------------\
   転送コスト削減
   (CloudFront, VPC エンドポイント)
```

### 5.2 コード例: S3 Storage Lens で分析

```bash
# Storage Lens 設定を作成
aws s3control put-storage-lens-configuration \
  --account-id 123456789012 \
  --config-id my-storage-lens \
  --storage-lens-configuration '{
    "Id": "my-storage-lens",
    "AccountLevel": {
      "BucketLevel": {
        "ActivityMetrics": {"IsEnabled": true},
        "PrefixLevel": {
          "StorageMetrics": {
            "IsEnabled": true,
            "SelectionCriteria": {
              "MaxDepth": 3,
              "MinStorageBytesPercentage": 1.0
            }
          }
        }
      }
    },
    "IsEnabled": true
  }'
```

### 5.3 コード例: VPC エンドポイントでデータ転送コスト削減

```bash
# S3 用 Gateway エンドポイント（無料）
aws ec2 create-vpc-endpoint \
  --vpc-id vpc-xxx \
  --service-name com.amazonaws.ap-northeast-1.s3 \
  --route-table-ids rtb-xxx \
  --vpc-endpoint-type Gateway

# エンドポイント経由のアクセスは NAT Gateway の料金が不要
# → データ転送コストを大幅に削減
```

---

## 6. イベント通知

### 6.1 コード例: S3 イベント通知の設定

```bash
# Lambda へのイベント通知
aws s3api put-bucket-notification-configuration \
  --bucket my-app-bucket \
  --notification-configuration '{
  "LambdaFunctionConfigurations": [
    {
      "Id": "ProcessUploadedImages",
      "LambdaFunctionArn": "arn:aws:lambda:ap-northeast-1:123456789012:function:image-processor",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [
            {"Name": "prefix", "Value": "uploads/"},
            {"Name": "suffix", "Value": ".jpg"}
          ]
        }
      }
    }
  ],
  "EventBridgeConfiguration": {}
}'
```

---

## 7. アンチパターン

### アンチパターン 1: バージョニング有効化後にライフサイクルを設定しない

バージョニングを有効にすると全バージョンが保持されるため、ストレージコストが際限なく増加する。NoncurrentVersionExpiration を必ず設定すべきである。

```json
{
  "Rules": [{
    "ID": "ExpireOldVersions",
    "Status": "Enabled",
    "Filter": {"Prefix": ""},
    "NoncurrentVersionExpiration": {"NoncurrentDays": 90}
  }]
}
```

### アンチパターン 2: S3 をデータベースのように使う

S3 はオブジェクトストレージであり、ランダムアクセスやトランザクション処理には向かない。頻繁な更新が必要なデータには DynamoDB や RDS を使用すべきである。

```
# 悪い例
S3 に JSON ファイルを格納して毎秒更新
→ 結果整合性の問題、PUT リクエスト料金が高額に

# 良い例
リアルタイムデータ → DynamoDB
集計結果・レポート → S3
```

---

## 8. FAQ

### Q1. S3 バッチオペレーションとは何か？

大量のオブジェクト（数十億件）に対して一括操作を実行するサービス。ストレージクラスの一括変更、タグの追加、ACL の更新、Lambda 関数の実行などが可能。S3 インベントリレポートを入力として使用する。

### Q2. S3 Object Lock と Glacier Vault Lock の違いは？

S3 Object Lock は WORM (Write Once Read Many) モデルでオブジェクトの削除・上書きを防止する。Governance モード（特権ユーザーは解除可能）と Compliance モード（誰も解除不可）がある。Glacier Vault Lock は Glacier 専用の同様の機能。コンプライアンス要件で使い分ける。

### Q3. Requester Pays バケットとは？

通常はバケット所有者がデータ転送料金を負担するが、Requester Pays を有効にするとリクエスト元が料金を負担する。大規模な公開データセット（ゲノムデータ等）で利用される。

---

## 9. まとめ

| 項目 | ポイント |
|------|---------|
| バージョニング | 誤削除防止、ただしライフサイクルで旧バージョンを期限管理 |
| CRR / SRR | 災害復旧・低レイテンシ / ログ集約・環境コピー |
| S3 Select | SQL でフィルタし転送量を最大 99% 削減 |
| Transfer Acceleration | CloudFront Edge 経由でグローバルアップロード高速化 |
| コスト最適化 | Intelligent-Tiering + ライフサイクル + VPC エンドポイント |
| イベント通知 | Lambda/SQS/SNS/EventBridge と連携 |

---

## 次に読むべきガイド

- [02-cloudfront.md](./02-cloudfront.md) — CloudFront CDN の設定
- [../03-database/00-rds-basics.md](../03-database/00-rds-basics.md) — RDS の基礎

---

## 参考文献

1. S3 バージョニング — https://docs.aws.amazon.com/AmazonS3/latest/userguide/Versioning.html
2. S3 レプリケーション — https://docs.aws.amazon.com/AmazonS3/latest/userguide/replication.html
3. S3 Select ユーザーガイド — https://docs.aws.amazon.com/AmazonS3/latest/userguide/selecting-content-from-objects.html
4. S3 Transfer Acceleration — https://docs.aws.amazon.com/AmazonS3/latest/userguide/transfer-acceleration.html

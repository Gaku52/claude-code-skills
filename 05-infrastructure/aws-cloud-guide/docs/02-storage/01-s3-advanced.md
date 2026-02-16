# S3 応用

> バージョニング、レプリケーション、S3 Select、Transfer Acceleration でプロダクションレベルの S3 運用を実現する

## この章で学ぶこと

1. バージョニングを活用してデータの世代管理と誤削除からの復旧ができる
2. クロスリージョンレプリケーション (CRR) で災害復旧と低レイテンシアクセスを実現できる
3. S3 Select、Transfer Acceleration、コスト最適化戦略を実装できる
4. S3 イベント通知と EventBridge を連携したイベント駆動アーキテクチャを構築できる
5. S3 Object Lock とコンプライアンス対応、バッチオペレーションによる大規模運用ができる

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

### 1.2 バージョニングの状態遷移

```
バージョニングの3つの状態
===========================

  Unversioned（未設定）
      |
      | PUT Bucket Versioning: Enabled
      v
  Enabled（有効）
      |
      | PUT Bucket Versioning: Suspended
      v
  Suspended（一時停止）
      |
      | PUT Bucket Versioning: Enabled
      v
  Enabled（再有効化）

※ 一度有効化すると Unversioned には戻せない
※ Suspended 中の PUT は VersionId = "null" で保存
※ Suspended 中も既存のバージョンは保持される
```

### 1.3 コード例: バージョニングの設定と操作

```bash
# バージョニングを有効化
aws s3api put-bucket-versioning \
  --bucket my-app-bucket \
  --versioning-configuration Status=Enabled

# バージョニングの状態を確認
aws s3api get-bucket-versioning \
  --bucket my-app-bucket

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

# バージョニングの一時停止
aws s3api put-bucket-versioning \
  --bucket my-app-bucket \
  --versioning-configuration Status=Suspended
```

### 1.4 コード例: Python でバージョン管理

```python
import boto3
from datetime import datetime, timezone

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

def delete_all_versions(bucket, key):
    """オブジェクトの全バージョンと削除マーカーを完全削除"""
    response = s3.list_object_versions(Bucket=bucket, Prefix=key)

    # バージョンを削除
    for version in response.get('Versions', []):
        s3.delete_object(
            Bucket=bucket,
            Key=key,
            VersionId=version['VersionId']
        )
        print(f"Deleted version: {version['VersionId']}")

    # 削除マーカーを削除
    for marker in response.get('DeleteMarkers', []):
        s3.delete_object(
            Bucket=bucket,
            Key=key,
            VersionId=marker['VersionId']
        )
        print(f"Deleted marker: {marker['VersionId']}")

def get_version_at_time(bucket, key, target_time):
    """指定時刻のバージョンを取得"""
    response = s3.list_object_versions(Bucket=bucket, Prefix=key)
    versions = sorted(
        response.get('Versions', []),
        key=lambda x: x['LastModified'],
        reverse=True
    )
    for v in versions:
        if v['LastModified'] <= target_time:
            return v
    return None
```

### 1.5 コード例: ライフサイクルルールとバージョニングの連携

```bash
# 旧バージョンを90日後に Glacier に移行、365日後に削除
aws s3api put-bucket-lifecycle-configuration \
  --bucket my-app-bucket \
  --lifecycle-configuration '{
    "Rules": [
      {
        "ID": "ManageOldVersions",
        "Status": "Enabled",
        "Filter": {"Prefix": ""},
        "NoncurrentVersionTransitions": [
          {
            "NoncurrentDays": 90,
            "StorageClass": "GLACIER"
          },
          {
            "NoncurrentDays": 180,
            "StorageClass": "DEEP_ARCHIVE"
          }
        ],
        "NoncurrentVersionExpiration": {
          "NoncurrentDays": 365,
          "NewerNoncurrentVersions": 5
        }
      },
      {
        "ID": "CleanupDeleteMarkers",
        "Status": "Enabled",
        "Filter": {"Prefix": ""},
        "Expiration": {
          "ExpiredObjectDeleteMarker": true
        }
      },
      {
        "ID": "AbortIncompleteMultipart",
        "Status": "Enabled",
        "Filter": {"Prefix": ""},
        "AbortIncompleteMultipartUpload": {
          "DaysAfterInitiation": 7
        }
      }
    ]
  }'
```

### 1.6 コード例: CloudFormation でバージョニング対応バケットを定義

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: S3 Bucket with Versioning and Lifecycle

Resources:
  AppBucket:
    Type: AWS::S3::Bucket
    DeletionPolicy: Retain
    Properties:
      BucketName: !Sub '${AWS::StackName}-app-data'
      VersioningConfiguration:
        Status: Enabled
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: aws:kms
              KMSMasterKeyID: !Ref BucketKmsKey
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true
      LifecycleConfiguration:
        Rules:
          - Id: ManageVersions
            Status: Enabled
            NoncurrentVersionTransitions:
              - StorageClass: GLACIER
                TransitionInDays: 90
            NoncurrentVersionExpiration:
              NoncurrentDays: 365
              NewerNoncurrentVersions: 3
          - Id: CleanupIncomplete
            Status: Enabled
            AbortIncompleteMultipartUpload:
              DaysAfterInitiation: 7

  BucketKmsKey:
    Type: AWS::KMS::Key
    Properties:
      Description: KMS key for S3 bucket encryption
      EnableKeyRotation: true
      KeyPolicy:
        Version: '2012-10-17'
        Statement:
          - Sid: AllowRootAccount
            Effect: Allow
            Principal:
              AWS: !Sub 'arn:aws:iam::${AWS::AccountId}:root'
            Action: 'kms:*'
            Resource: '*'
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

### 2.2 レプリケーション詳細アーキテクチャ

```
レプリケーションの動作フロー
================================

  1. オブジェクト PUT → ソースバケット
  2. S3 が非同期でレプリケーションキューに登録
  3. IAM ロールで認証してデスティネーションに PUT
  4. レプリケーションメトリクス（CloudWatch）で状態監視

  レプリケーション対象:
  ✅ 新規オブジェクト (PUT)
  ✅ メタデータの変更
  ✅ ACL の変更
  ✅ タグの変更（Replica modification sync 有効時）
  ✅ 削除マーカー（設定による）

  レプリケーション非対象:
  ❌ 既存オブジェクト（S3 Batch Replication で対応）
  ❌ SSE-C で暗号化されたオブジェクト
  ❌ バケット設定（ライフサイクル、通知等）
  ❌ レプリカからの再レプリケーション（デフォルト）

  双方向レプリケーション:
  Source A <---> Source B
  ※ レプリカ修正同期を有効にして双方向に設定
  ※ ループ防止のためレプリカは再レプリケーション対象外
```

### 2.3 コード例: CRR の設定

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

# レプリケーション用 IAM ポリシーを作成
cat > replication-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetReplicationConfiguration",
        "s3:ListBucket"
      ],
      "Resource": "arn:aws:s3:::source-bucket"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObjectVersionForReplication",
        "s3:GetObjectVersionAcl",
        "s3:GetObjectVersionTagging"
      ],
      "Resource": "arn:aws:s3:::source-bucket/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:ReplicateObject",
        "s3:ReplicateDelete",
        "s3:ReplicateTags"
      ],
      "Resource": "arn:aws:s3:::destination-bucket/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt"
      ],
      "Resource": "arn:aws:kms:ap-northeast-1:123456789012:key/source-key-id",
      "Condition": {
        "StringLike": {
          "kms:ViaService": "s3.ap-northeast-1.amazonaws.com"
        }
      }
    },
    {
      "Effect": "Allow",
      "Action": [
        "kms:Encrypt"
      ],
      "Resource": "arn:aws:kms:ap-northeast-3:123456789012:key/dest-key-id",
      "Condition": {
        "StringLike": {
          "kms:ViaService": "s3.ap-northeast-3.amazonaws.com"
        }
      }
    }
  ]
}
EOF

aws iam put-role-policy \
  --role-name S3ReplicationRole \
  --policy-name S3ReplicationPolicy \
  --policy-document file://replication-policy.json

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
        },
        "EncryptionConfiguration": {
          "ReplicaKmsKeyID": "arn:aws:kms:ap-northeast-3:123456789012:key/dest-key-id"
        }
      },
      "SourceSelectionCriteria": {
        "SseKmsEncryptedObjects": {
          "Status": "Enabled"
        }
      },
      "DeleteMarkerReplication": {"Status": "Enabled"}
    }
  ]
}'

# レプリケーションの状態確認
aws s3api get-bucket-replication --bucket source-bucket

# レプリケーションメトリクスの確認（S3 Replication Time Control）
aws cloudwatch get-metric-statistics \
  --namespace AWS/S3 \
  --metric-name ReplicationLatency \
  --dimensions Name=SourceBucket,Value=source-bucket \
              Name=DestinationBucket,Value=destination-bucket \
              Name=RuleId,Value=ReplicateAll \
  --start-time 2026-02-14T00:00:00Z \
  --end-time 2026-02-15T00:00:00Z \
  --period 3600 \
  --statistics Average
```

### 2.4 コード例: S3 Batch Replication（既存オブジェクトのレプリケーション）

```bash
# S3 インベントリレポートの設定（バッチレプリケーションの入力）
aws s3api put-bucket-inventory-configuration \
  --bucket source-bucket \
  --id weekly-inventory \
  --inventory-configuration '{
    "Destination": {
      "S3BucketDestination": {
        "AccountId": "123456789012",
        "Bucket": "arn:aws:s3:::inventory-reports-bucket",
        "Format": "CSV",
        "Prefix": "inventory"
      }
    },
    "IsEnabled": true,
    "Id": "weekly-inventory",
    "IncludedObjectVersions": "All",
    "Schedule": {"Frequency": "Weekly"},
    "OptionalFields": [
      "Size", "LastModifiedDate", "StorageClass",
      "ReplicationStatus", "EncryptionStatus"
    ]
  }'

# バッチレプリケーションジョブの作成
aws s3control create-job \
  --account-id 123456789012 \
  --operation '{"S3ReplicateObject":{}}' \
  --manifest '{
    "Spec": {
      "Format": "S3InventoryReport_CSV_20211130",
      "Fields": ["Bucket","Key","VersionId"]
    },
    "Location": {
      "ObjectArn": "arn:aws:s3:::inventory-reports-bucket/inventory/source-bucket/weekly-inventory/data/manifest.json",
      "ETag": "abc123"
    }
  }' \
  --report '{
    "Bucket": "arn:aws:s3:::batch-reports-bucket",
    "Format": "Report_CSV_20180820",
    "Enabled": true,
    "Prefix": "batch-replication",
    "ReportScope": "AllTasks"
  }' \
  --priority 42 \
  --role-arn arn:aws:iam::123456789012:role/S3BatchRole \
  --confirmation-required
```

### 2.5 レプリケーション比較表

| 機能 | CRR | SRR |
|------|-----|-----|
| リージョン | 異なるリージョン | 同一リージョン |
| ユースケース | DR、低レイテンシ | ログ集約、環境コピー |
| バージョニング | 必須 | 必須 |
| 既存オブジェクト | S3 Batch Replication | S3 Batch Replication |
| 削除マーカー | 選択可能 | 選択可能 |
| 料金 | リクエスト + データ転送 | リクエストのみ |
| RTC (S3 Replication Time Control) | 対応（99.99% を 15分以内） | 対応 |
| SSE-KMS 暗号化 | 対応（異なる KMS キー指定可） | 対応 |
| 双方向レプリケーション | 対応 | 対応 |

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

S3 Select SQL の制約:
  - SELECT / FROM / WHERE のみ（JOIN 不可）
  - 集約関数: COUNT, SUM, AVG, MIN, MAX
  - LIKE 演算子、BETWEEN、IN 対応
  - LIMIT 句対応
  - サブクエリ非対応
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
        elif 'Stats' in event:
            stats = event['Stats']['Details']
            print(f"Scanned: {stats['BytesScanned']} bytes, "
                  f"Processed: {stats['BytesProcessed']} bytes, "
                  f"Returned: {stats['BytesReturned']} bytes")
    return records

# 使用例: 2024年の東京のデータのみ取得
results = query_csv(
    'analytics-bucket',
    'data/sales-2024.csv.gz',
    "SELECT s.date, s.product, s.amount FROM s3object s WHERE s.region = 'Tokyo' AND CAST(s.amount AS INT) > 10000"
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

# Parquet ファイルからクエリ（列指向のため、列選択で最大の効果）
response = s3.select_object_content(
    Bucket='analytics-bucket',
    Key='data/events.parquet',
    ExpressionType='SQL',
    Expression="SELECT event_type, COUNT(*) as cnt FROM s3object s GROUP BY event_type",
    InputSerialization={'Parquet': {}},
    OutputSerialization={'JSON': {'RecordDelimiter': '\n'}}
)
```

### 3.3 コード例: S3 Select の実践的ラッパークラス

```python
import boto3
import json
from typing import List, Dict, Any, Optional

class S3SelectQuery:
    """S3 Select のラッパークラス"""

    def __init__(self, region_name='ap-northeast-1'):
        self.s3 = boto3.client('s3', region_name=region_name)

    def query_csv(
        self,
        bucket: str,
        key: str,
        sql: str,
        compression: str = 'NONE',
        delimiter: str = ',',
        header: str = 'USE'
    ) -> List[Dict[str, Any]]:
        """CSV ファイルに対して S3 Select クエリを実行"""
        response = self.s3.select_object_content(
            Bucket=bucket,
            Key=key,
            ExpressionType='SQL',
            Expression=sql,
            InputSerialization={
                'CSV': {
                    'FileHeaderInfo': header,
                    'RecordDelimiter': '\n',
                    'FieldDelimiter': delimiter
                },
                'CompressionType': compression
            },
            OutputSerialization={'JSON': {'RecordDelimiter': '\n'}}
        )
        return self._parse_response(response)

    def query_json(
        self,
        bucket: str,
        key: str,
        sql: str,
        json_type: str = 'LINES',
        compression: str = 'NONE'
    ) -> List[Dict[str, Any]]:
        """JSON ファイルに対して S3 Select クエリを実行"""
        response = self.s3.select_object_content(
            Bucket=bucket,
            Key=key,
            ExpressionType='SQL',
            Expression=sql,
            InputSerialization={
                'JSON': {'Type': json_type},
                'CompressionType': compression
            },
            OutputSerialization={'JSON': {'RecordDelimiter': '\n'}}
        )
        return self._parse_response(response)

    def query_parquet(
        self,
        bucket: str,
        key: str,
        sql: str
    ) -> List[Dict[str, Any]]:
        """Parquet ファイルに対して S3 Select クエリを実行"""
        response = self.s3.select_object_content(
            Bucket=bucket,
            Key=key,
            ExpressionType='SQL',
            Expression=sql,
            InputSerialization={'Parquet': {}},
            OutputSerialization={'JSON': {'RecordDelimiter': '\n'}}
        )
        return self._parse_response(response)

    def _parse_response(self, response) -> List[Dict[str, Any]]:
        """レスポンスをパースして辞書のリストとして返す"""
        records = []
        stats = {}
        for event in response['Payload']:
            if 'Records' in event:
                payload = event['Records']['Payload'].decode('utf-8')
                for line in payload.strip().split('\n'):
                    if line:
                        records.append(json.loads(line))
            elif 'Stats' in event:
                details = event['Stats']['Details']
                stats = {
                    'bytes_scanned': details['BytesScanned'],
                    'bytes_processed': details['BytesProcessed'],
                    'bytes_returned': details['BytesReturned'],
                    'compression_ratio': (
                        1 - details['BytesReturned'] / details['BytesScanned']
                    ) if details['BytesScanned'] > 0 else 0
                }
                print(f"S3 Select Stats: scanned={stats['bytes_scanned']}, "
                      f"returned={stats['bytes_returned']}, "
                      f"compression={stats['compression_ratio']:.1%}")
        return records

# 使用例
sq = S3SelectQuery()

# 大容量 CSV からの高速フィルタリング
errors = sq.query_csv(
    'log-bucket',
    'access-logs/2026/02/access.csv.gz',
    "SELECT s.timestamp, s.status, s.path FROM s3object s WHERE CAST(s.status AS INT) >= 500",
    compression='GZIP'
)

# JSON Lines ログからの抽出
slow_queries = sq.query_json(
    'app-logs',
    'db-logs/slow-queries.json',
    "SELECT s.query, s.duration, s.timestamp FROM s3object s WHERE s.duration > 1000"
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

効果が高いケース:
  - 地理的に遠い場所からのアップロード
  - 大容量ファイルの転送
  - インターネット経路が不安定な場合

効果が低いケース:
  - S3 と同じリージョンからのアクセス
  - 小容量ファイルの転送
  - ネットワーク帯域が十分な場合
```

### 4.2 コード例: Transfer Acceleration の設定

```bash
# Transfer Acceleration を有効化
aws s3api put-bucket-accelerate-configuration \
  --bucket my-global-bucket \
  --accelerate-configuration Status=Enabled

# 有効化の確認
aws s3api get-bucket-accelerate-configuration \
  --bucket my-global-bucket

# Acceleration エンドポイントでアップロード
aws s3 cp large-file.zip \
  s3://my-global-bucket/uploads/ \
  --endpoint-url https://my-global-bucket.s3-accelerate.amazonaws.com

# マルチパートアップロードと組み合わせ（大容量ファイル）
aws s3 cp large-dataset.tar.gz \
  s3://my-global-bucket/datasets/ \
  --endpoint-url https://my-global-bucket.s3-accelerate.amazonaws.com \
  --expected-size 10737418240

# 速度比較ツール
# https://s3-accelerate-speedtest.s3-accelerate.amazonaws.com/en/accelerate-speed-comparsion.html
```

### 4.3 コード例: Python で Transfer Acceleration を使用

```python
import boto3
from boto3.s3.transfer import TransferConfig
import time

# Acceleration エンドポイントを使用
s3 = boto3.client(
    's3',
    config=boto3.session.Config(s3={'use_accelerate_endpoint': True})
)

# マルチパート設定（大容量ファイル向け）
config = TransferConfig(
    multipart_threshold=100 * 1024 * 1024,   # 100MB
    max_concurrency=10,
    multipart_chunksize=100 * 1024 * 1024,
    use_threads=True
)

# プログレスコールバック付きアップロード
class ProgressTracker:
    def __init__(self, filename, filesize):
        self.filename = filename
        self.filesize = filesize
        self.uploaded = 0
        self.start_time = time.time()

    def __call__(self, bytes_amount):
        self.uploaded += bytes_amount
        percentage = (self.uploaded / self.filesize) * 100
        elapsed = time.time() - self.start_time
        speed = self.uploaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
        print(f"\r{self.filename}: {percentage:.1f}% ({speed:.1f} MB/s)", end='')

import os
filepath = 'large-dataset.tar.gz'
filesize = os.path.getsize(filepath)
progress = ProgressTracker(filepath, filesize)

s3.upload_file(
    filepath,
    'my-global-bucket',
    'datasets/large-dataset.tar.gz',
    Config=config,
    Callback=progress
)
print(f"\nUpload complete in {time.time() - progress.start_time:.1f}s")
```

### 4.4 コード例: Transfer Acceleration 速度テスト

```python
import boto3
import time
import os

def benchmark_transfer(bucket, key, filepath, use_acceleration=False):
    """通常転送と Acceleration 転送の速度を比較"""
    config_kwargs = {}
    if use_acceleration:
        config_kwargs['config'] = boto3.session.Config(
            s3={'use_accelerate_endpoint': True}
        )

    s3 = boto3.client('s3', **config_kwargs)
    filesize = os.path.getsize(filepath)

    start = time.time()
    s3.upload_file(filepath, bucket, key)
    elapsed = time.time() - start

    speed_mbps = (filesize / 1024 / 1024) / elapsed
    print(f"{'Accelerated' if use_acceleration else 'Standard'}: "
          f"{elapsed:.2f}s ({speed_mbps:.2f} MB/s)")
    return elapsed

# ベンチマーク実行
test_file = 'test-100mb.bin'
print("Transfer speed comparison:")
standard_time = benchmark_transfer('my-bucket', 'test/std', test_file, False)
accel_time = benchmark_transfer('my-bucket', 'test/accel', test_file, True)
improvement = ((standard_time - accel_time) / standard_time) * 100
print(f"Improvement: {improvement:.1f}%")
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

### 5.2 ストレージクラス詳細比較表

| ストレージクラス | 保存料金(GB/月) | 取得料金(GB) | 最低保存期間 | 取得時間 | 可用性 | ユースケース |
|---|---|---|---|---|---|---|
| STANDARD | $0.025 | 無料 | なし | 即時 | 99.99% | アクティブデータ |
| INTELLIGENT_TIERING | $0.025 | 無料 | なし | 即時 | 99.9% | アクセスパターン不明 |
| STANDARD_IA | $0.0138 | $0.01 | 30日 | 即時 | 99.9% | 低頻度アクセス |
| ONE_ZONE_IA | $0.011 | $0.01 | 30日 | 即時 | 99.5% | 再生成可能データ |
| GLACIER_IR | $0.005 | $0.03 | 90日 | 即時 | 99.9% | アーカイブ即時アクセス |
| GLACIER_FLEXIBLE | $0.0045 | $0.01-0.03 | 90日 | 1分〜12時間 | 99.99% | アーカイブ |
| DEEP_ARCHIVE | $0.002 | $0.02 | 180日 | 12〜48時間 | 99.99% | 長期保存 |

※ 料金は東京リージョン基準の概算

### 5.3 コード例: Intelligent-Tiering の設定

```bash
# Intelligent-Tiering アーカイブアクセス層を有効化
aws s3api put-bucket-intelligent-tiering-configuration \
  --bucket my-data-bucket \
  --id archive-config \
  --intelligent-tiering-configuration '{
    "Id": "archive-config",
    "Status": "Enabled",
    "Tierings": [
      {
        "AccessTier": "ARCHIVE_ACCESS",
        "Days": 90
      },
      {
        "AccessTier": "DEEP_ARCHIVE_ACCESS",
        "Days": 180
      }
    ],
    "Filter": {
      "Prefix": "data/"
    }
  }'

# ライフサイクルで Intelligent-Tiering に自動移行
aws s3api put-bucket-lifecycle-configuration \
  --bucket my-data-bucket \
  --lifecycle-configuration '{
    "Rules": [
      {
        "ID": "MoveToIT",
        "Status": "Enabled",
        "Filter": {"Prefix": "data/"},
        "Transitions": [
          {
            "Days": 0,
            "StorageClass": "INTELLIGENT_TIERING"
          }
        ]
      }
    ]
  }'
```

### 5.4 コード例: S3 Storage Lens で分析

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
        "AdvancedCostOptimizationMetrics": {"IsEnabled": true},
        "AdvancedDataProtectionMetrics": {"IsEnabled": true},
        "DetailedStatusCodesMetrics": {"IsEnabled": true},
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
    "DataExport": {
      "S3BucketDestination": {
        "AccountId": "123456789012",
        "Arn": "arn:aws:s3:::storage-lens-reports",
        "Format": "CSV",
        "OutputSchemaVersion": "V_1",
        "Encryption": {
          "SSES3": {}
        }
      },
      "CloudWatchMetrics": {
        "IsEnabled": true
      }
    },
    "IsEnabled": true
  }'
```

### 5.5 コード例: VPC エンドポイントでデータ転送コスト削減

```bash
# S3 用 Gateway エンドポイント（無料）
aws ec2 create-vpc-endpoint \
  --vpc-id vpc-xxx \
  --service-name com.amazonaws.ap-northeast-1.s3 \
  --route-table-ids rtb-xxx \
  --vpc-endpoint-type Gateway

# エンドポイントポリシーでアクセスを制限
aws ec2 modify-vpc-endpoint \
  --vpc-endpoint-id vpce-xxx \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Sid": "AllowSpecificBuckets",
        "Effect": "Allow",
        "Principal": "*",
        "Action": [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ],
        "Resource": [
          "arn:aws:s3:::my-app-bucket",
          "arn:aws:s3:::my-app-bucket/*",
          "arn:aws:s3:::my-logs-bucket",
          "arn:aws:s3:::my-logs-bucket/*"
        ]
      }
    ]
  }'

# エンドポイント経由のアクセスは NAT Gateway の料金が不要
# → データ転送コストを大幅に削減
```

### 5.6 コード例: コスト分析スクリプト

```python
import boto3
from datetime import datetime, timedelta

def analyze_s3_costs(bucket_name):
    """S3 バケットのコスト最適化レポートを生成"""
    s3 = boto3.client('s3')
    cloudwatch = boto3.client('cloudwatch')

    # バケットサイズの取得
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=1)

    size_response = cloudwatch.get_metric_statistics(
        Namespace='AWS/S3',
        MetricName='BucketSizeBytes',
        Dimensions=[
            {'Name': 'BucketName', 'Value': bucket_name},
            {'Name': 'StorageType', 'Value': 'StandardStorage'}
        ],
        StartTime=start_time,
        EndTime=end_time,
        Period=86400,
        Statistics=['Average']
    )

    # オブジェクト数の取得
    count_response = cloudwatch.get_metric_statistics(
        Namespace='AWS/S3',
        MetricName='NumberOfObjects',
        Dimensions=[
            {'Name': 'BucketName', 'Value': bucket_name},
            {'Name': 'StorageType', 'Value': 'AllStorageTypes'}
        ],
        StartTime=start_time,
        EndTime=end_time,
        Period=86400,
        Statistics=['Average']
    )

    # ストレージクラス別の分析
    paginator = s3.get_paginator('list_objects_v2')
    storage_classes = {}
    total_size = 0
    for page in paginator.paginate(Bucket=bucket_name, MaxKeys=1000):
        for obj in page.get('Contents', []):
            sc = obj.get('StorageClass', 'STANDARD')
            if sc not in storage_classes:
                storage_classes[sc] = {'count': 0, 'size': 0}
            storage_classes[sc]['count'] += 1
            storage_classes[sc]['size'] += obj['Size']
            total_size += obj['Size']

    # レポート出力
    print(f"=== S3 Cost Analysis: {bucket_name} ===")
    print(f"Total Size: {total_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"\nStorage Class Distribution:")
    for sc, info in storage_classes.items():
        pct = (info['size'] / total_size * 100) if total_size > 0 else 0
        print(f"  {sc}: {info['count']} objects, "
              f"{info['size'] / 1024 / 1024:.1f} MB ({pct:.1f}%)")

    # 最適化の推奨
    print(f"\n=== Recommendations ===")
    std_size = storage_classes.get('STANDARD', {}).get('size', 0)
    if std_size > 100 * 1024 * 1024 * 1024:  # 100GB以上
        print("- Consider Intelligent-Tiering for large STANDARD storage")
    if 'STANDARD' in storage_classes and storage_classes['STANDARD']['count'] > 10000:
        print("- Enable S3 Inventory for detailed analysis")
    print("- Check for incomplete multipart uploads")
    print("- Review lifecycle rules for old objects")

# 実行
analyze_s3_costs('my-app-bucket')
```

---

## 6. イベント通知

### 6.1 イベント通知アーキテクチャ

```
S3 イベント通知の配信先
========================

                    +---> Lambda (画像リサイズ、動画変換)
                    |
S3 Event -----+--> +---> SQS (メッセージキュー)
  (ObjectCreated,  |
   ObjectRemoved,  +---> SNS (通知 → メール、SMS)
   Restore,        |
   Replication)    +---> EventBridge (高度なルーティング)
                           |
                           +---> Step Functions
                           +---> Lambda
                           +---> ECS タスク
                           +---> 他の AWS サービス

EventBridge の利点:
  - 複数のルール/ターゲット
  - コンテンツベースのフィルタリング
  - アーカイブ & リプレイ
  - クロスアカウント配信
```

### 6.2 コード例: S3 イベント通知の設定

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
            {"Name": "prefix", "Value": "uploads/images/"},
            {"Name": "suffix", "Value": ".jpg"}
          ]
        }
      }
    },
    {
      "Id": "ProcessUploadedPNG",
      "LambdaFunctionArn": "arn:aws:lambda:ap-northeast-1:123456789012:function:image-processor",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [
            {"Name": "prefix", "Value": "uploads/images/"},
            {"Name": "suffix", "Value": ".png"}
          ]
        }
      }
    }
  ],
  "QueueConfigurations": [
    {
      "Id": "LogDeletion",
      "QueueArn": "arn:aws:sqs:ap-northeast-1:123456789012:s3-deletion-queue",
      "Events": ["s3:ObjectRemoved:*"]
    }
  ],
  "EventBridgeConfiguration": {}
}'
```

### 6.3 コード例: EventBridge を使ったイベント駆動処理

```python
# Lambda: S3 イベントを処理する関数
import boto3
import json
import urllib.parse

s3 = boto3.client('s3')

def handler(event, context):
    """S3 イベント通知を処理する Lambda ハンドラ"""

    # S3 イベント通知の場合
    if 'Records' in event:
        for record in event['Records']:
            bucket = record['s3']['bucket']['name']
            key = urllib.parse.unquote_plus(
                record['s3']['object']['key'], encoding='utf-8'
            )
            size = record['s3']['object'].get('size', 0)
            event_name = record['eventName']

            print(f"Event: {event_name}, Bucket: {bucket}, "
                  f"Key: {key}, Size: {size}")

            if event_name.startswith('ObjectCreated:'):
                process_new_object(bucket, key, size)
            elif event_name.startswith('ObjectRemoved:'):
                handle_deletion(bucket, key)

    # EventBridge 経由の場合
    elif event.get('source') == 'aws.s3':
        detail = event['detail']
        bucket = detail['bucket']['name']
        key = detail['object']['key']
        process_new_object(bucket, key, detail['object']['size'])

    return {'statusCode': 200}

def process_new_object(bucket, key, size):
    """新規オブジェクトの処理"""
    if key.endswith(('.jpg', '.jpeg', '.png', '.webp')):
        # 画像処理パイプラインを起動
        generate_thumbnails(bucket, key)
    elif key.endswith('.csv'):
        # CSV データの ETL 処理
        trigger_etl_job(bucket, key)
    elif key.endswith('.mp4'):
        # 動画トランスコーディング
        start_transcode_job(bucket, key)

def handle_deletion(bucket, key):
    """オブジェクト削除時の処理"""
    print(f"Object deleted: s3://{bucket}/{key}")
    # CDN キャッシュの無効化
    # 検索インデックスからの削除
    # 関連リソースのクリーンアップ
```

### 6.4 コード例: CloudFormation でイベント通知を設定

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  UploadBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub '${AWS::StackName}-uploads'
      NotificationConfiguration:
        LambdaConfigurations:
          - Event: 's3:ObjectCreated:*'
            Function: !GetAtt ImageProcessorFunction.Arn
            Filter:
              S3Key:
                Rules:
                  - Name: prefix
                    Value: images/
        EventBridgeConfiguration:
          EventBridgeEnabled: true

  ImageProcessorFunction:
    Type: AWS::Serverless::Function
    Properties:
      Runtime: python3.12
      Handler: index.handler
      MemorySize: 1024
      Timeout: 300
      Policies:
        - S3ReadPolicy:
            BucketName: !Sub '${AWS::StackName}-uploads'
        - S3CrudPolicy:
            BucketName: !Sub '${AWS::StackName}-processed'

  ImageProcessorPermission:
    Type: AWS::Lambda::Permission
    Properties:
      FunctionName: !Ref ImageProcessorFunction
      Action: lambda:InvokeFunction
      Principal: s3.amazonaws.com
      SourceArn: !GetAtt UploadBucket.Arn

  # EventBridge ルール（高度なフィルタリング）
  LargeFileRule:
    Type: AWS::Events::Rule
    Properties:
      EventPattern:
        source:
          - aws.s3
        detail-type:
          - 'Object Created'
        detail:
          bucket:
            name:
              - !Ref UploadBucket
          object:
            size:
              - numeric:
                  - '>='
                  - 104857600  # 100MB以上
      Targets:
        - Arn: !GetAtt LargeFileProcessorFunction.Arn
          Id: LargeFileTarget
```

---

## 7. S3 Object Lock とコンプライアンス

### 7.1 Object Lock の概要

```
S3 Object Lock モード
=====================

Governance モード:
  - 特別な権限 (s3:BypassGovernanceRetention) を持つユーザーは解除可能
  - テスト環境や柔軟な保持期間が必要な場合に使用
  - IAM ポリシーで解除権限を管理

Compliance モード:
  - Root ユーザーを含め、誰も解除不可
  - 規制要件（FINRA、SEC 等）を満たす場合に使用
  - 保持期間中は絶対にデータを削除できない

Legal Hold:
  - 保持期間とは独立して設定可能
  - s3:PutObjectLegalHold 権限で ON/OFF を切り替え
  - 訴訟ホールド等の法的要件に対応
```

### 7.2 コード例: Object Lock の設定

```bash
# Object Lock 有効なバケットを作成（作成時のみ設定可能）
aws s3api create-bucket \
  --bucket compliance-bucket \
  --region ap-northeast-1 \
  --create-bucket-configuration LocationConstraint=ap-northeast-1 \
  --object-lock-enabled-for-bucket

# デフォルトの Object Lock 設定
aws s3api put-object-lock-configuration \
  --bucket compliance-bucket \
  --object-lock-configuration '{
    "ObjectLockEnabled": "Enabled",
    "Rule": {
      "DefaultRetention": {
        "Mode": "COMPLIANCE",
        "Days": 365
      }
    }
  }'

# 個別オブジェクトに Object Lock を設定
aws s3api put-object-retention \
  --bucket compliance-bucket \
  --key financial-reports/2025-annual.pdf \
  --retention '{
    "Mode": "COMPLIANCE",
    "RetainUntilDate": "2030-12-31T00:00:00Z"
  }'

# Legal Hold の設定
aws s3api put-object-legal-hold \
  --bucket compliance-bucket \
  --key contracts/nda-2025.pdf \
  --legal-hold '{"Status": "ON"}'

# Legal Hold の解除
aws s3api put-object-legal-hold \
  --bucket compliance-bucket \
  --key contracts/nda-2025.pdf \
  --legal-hold '{"Status": "OFF"}'
```

---

## 8. S3 バッチオペレーション

### 8.1 バッチオペレーションの概要

```
S3 バッチオペレーションのフロー
================================

  1. マニフェスト作成
     ├── S3 インベントリレポート（自動生成）
     └── CSV ファイル（手動作成）

  2. ジョブ作成 → オペレーション指定
     ├── オブジェクトのコピー
     ├── ストレージクラスの変更
     ├── タグの追加/削除
     ├── ACL の更新
     ├── Object Lock の設定
     ├── S3 Batch Replication
     └── Lambda 関数の実行

  3. ジョブ実行 → 完了レポート
```

### 8.2 コード例: バッチオペレーション

```bash
# ストレージクラスの一括変更ジョブ
aws s3control create-job \
  --account-id 123456789012 \
  --operation '{
    "S3PutObjectCopy": {
      "TargetResource": "arn:aws:s3:::my-app-bucket",
      "StorageClass": "INTELLIGENT_TIERING",
      "MetadataDirective": "COPY"
    }
  }' \
  --manifest '{
    "Spec": {
      "Format": "S3InventoryReport_CSV_20211130",
      "Fields": ["Bucket", "Key", "VersionId"]
    },
    "Location": {
      "ObjectArn": "arn:aws:s3:::inventory-bucket/manifest.json",
      "ETag": "abc123def456"
    }
  }' \
  --report '{
    "Bucket": "arn:aws:s3:::reports-bucket",
    "Format": "Report_CSV_20180820",
    "Enabled": true,
    "Prefix": "batch-reports/storage-class-change",
    "ReportScope": "FailedTasksOnly"
  }' \
  --priority 10 \
  --role-arn arn:aws:iam::123456789012:role/S3BatchRole \
  --confirmation-required

# タグの一括追加ジョブ
aws s3control create-job \
  --account-id 123456789012 \
  --operation '{
    "S3PutObjectTagging": {
      "TagSet": [
        {"Key": "Department", "Value": "Finance"},
        {"Key": "Classification", "Value": "Internal"}
      ]
    }
  }' \
  --manifest '{
    "Spec": {
      "Format": "S3BatchOperations_CSV_20180820",
      "Fields": ["Bucket", "Key"]
    },
    "Location": {
      "ObjectArn": "arn:aws:s3:::manifests/tag-objects.csv",
      "ETag": "xyz789"
    }
  }' \
  --report '{
    "Bucket": "arn:aws:s3:::reports-bucket",
    "Format": "Report_CSV_20180820",
    "Enabled": true,
    "ReportScope": "AllTasks"
  }' \
  --priority 20 \
  --role-arn arn:aws:iam::123456789012:role/S3BatchRole \
  --no-confirmation-required

# ジョブの状態確認
aws s3control describe-job \
  --account-id 123456789012 \
  --job-id "job-id-here"

# ジョブの一覧
aws s3control list-jobs \
  --account-id 123456789012 \
  --job-statuses Active Complete
```

---

## 9. S3 のセキュリティ設定

### 9.1 暗号化の選択

```
S3 暗号化方式の比較
====================

SSE-S3 (デフォルト):
  - Amazon が管理する AES-256 キー
  - 追加コストなし
  - 鍵の管理不要
  → 一般的なワークロードに推奨

SSE-KMS:
  - AWS KMS のカスタマーマネージドキー
  - 鍵のローテーション管理
  - CloudTrail でキー使用を監査
  - KMS API 呼び出しコスト
  → コンプライアンス要件がある場合に推奨

SSE-C:
  - 顧客が提供する暗号化キー
  - AWS はキーを保存しない
  - リクエストごとにキーを送信
  → 独自のキー管理が必要な場合

クライアントサイド暗号化:
  - アプリケーション側で暗号化してから S3 に PUT
  - AWS は暗号化に一切関与しない
  → 最高レベルのセキュリティ要件
```

### 9.2 コード例: バケットポリシーのセキュリティ強化

```bash
# HTTPS 強制 + VPC エンドポイント制限 + 暗号化強制
aws s3api put-bucket-policy --bucket secure-bucket --policy '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ForceHTTPS",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::secure-bucket",
        "arn:aws:s3:::secure-bucket/*"
      ],
      "Condition": {
        "Bool": {
          "aws:SecureTransport": "false"
        }
      }
    },
    {
      "Sid": "RestrictToVPC",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::secure-bucket",
        "arn:aws:s3:::secure-bucket/*"
      ],
      "Condition": {
        "StringNotEquals": {
          "aws:SourceVpce": "vpce-xxxxxxxx"
        }
      }
    },
    {
      "Sid": "ForceSSEKMS",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::secure-bucket/*",
      "Condition": {
        "StringNotEquals": {
          "s3:x-amz-server-side-encryption": "aws:kms"
        }
      }
    },
    {
      "Sid": "DenyUnencryptedTransport",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::secure-bucket/*",
      "Condition": {
        "Null": {
          "s3:x-amz-server-side-encryption": "true"
        }
      }
    }
  ]
}'

# S3 Access Point の作成（アクセス制御の簡素化）
aws s3control create-access-point \
  --account-id 123456789012 \
  --name app-readonly-ap \
  --bucket secure-bucket \
  --vpc-configuration VpcId=vpc-xxx \
  --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

### 9.3 コード例: S3 Access Analyzer

```bash
# IAM Access Analyzer で外部アクセスを検出
aws accessanalyzer create-analyzer \
  --analyzer-name s3-analyzer \
  --type ACCOUNT

# 検出結果の確認
aws accessanalyzer list-findings \
  --analyzer-arn arn:aws:access-analyzer:ap-northeast-1:123456789012:analyzer/s3-analyzer \
  --filter '{
    "resourceType": {
      "eq": ["AWS::S3::Bucket"]
    },
    "status": {
      "eq": ["ACTIVE"]
    }
  }'
```

---

## 10. プレフィックス設計とパフォーマンス

### 10.1 S3 のパフォーマンス特性

```
S3 パフォーマンス上限（プレフィックスあたり）
=============================================

  読み取り: 5,500 GET/HEAD リクエスト/秒
  書き込み: 3,500 PUT/POST/DELETE リクエスト/秒

  プレフィックス例:
    s3://bucket/images/   → 1つのプレフィックス
    s3://bucket/videos/   → 別のプレフィックス

  並列プレフィックスでスループット向上:
    s3://bucket/images/2026/02/15/aa/
    s3://bucket/images/2026/02/15/ab/
    ...
    → プレフィックス数 × 5,500 GET/秒

  ※ 以前はランダムプレフィックスが推奨されていたが、
    2018年の改善により、日付ベースのプレフィックスでも
    自動的にパーティション分割される
```

### 10.2 コード例: 高スループット S3 アクセス

```python
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

s3 = boto3.client('s3')

def parallel_upload(bucket, prefix, file_list, max_workers=20):
    """並列アップロードで高スループットを実現"""
    results = []

    def upload_one(filepath):
        key = f"{prefix}/{os.path.basename(filepath)}"
        s3.upload_file(filepath, bucket, key)
        return {'file': filepath, 'key': key, 'status': 'success'}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(upload_one, f): f for f in file_list
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({
                    'file': futures[future],
                    'status': 'error',
                    'error': str(e)
                })

    success = sum(1 for r in results if r['status'] == 'success')
    print(f"Uploaded {success}/{len(file_list)} files")
    return results

def parallel_download(bucket, prefix, dest_dir, max_workers=20):
    """並列ダウンロードで高スループットを実現"""
    paginator = s3.get_paginator('list_objects_v2')
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            keys.append(obj['Key'])

    os.makedirs(dest_dir, exist_ok=True)

    def download_one(key):
        local_path = os.path.join(dest_dir, os.path.basename(key))
        s3.download_file(bucket, key, local_path)
        return {'key': key, 'local': local_path, 'status': 'success'}

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_one, k): k for k in keys
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                results.append({
                    'key': futures[future],
                    'status': 'error',
                    'error': str(e)
                })

    success = sum(1 for r in results if r['status'] == 'success')
    print(f"Downloaded {success}/{len(keys)} files")
    return results
```

---

## 11. S3 と他のサービスとの連携

### 11.1 Athena との連携

```bash
# S3 上のデータを Athena でクエリ
aws athena start-query-execution \
  --query-string "
    CREATE EXTERNAL TABLE IF NOT EXISTS access_logs (
      request_time string,
      remote_ip string,
      request_method string,
      request_path string,
      status_code int,
      bytes_sent bigint,
      user_agent string
    )
    ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.RegexSerDe'
    WITH SERDEPROPERTIES (
      'input.regex' = '([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([0-9]*) ([0-9]*) (.*)$'
    )
    LOCATION 's3://my-logs-bucket/access-logs/'
  " \
  --result-configuration OutputLocation=s3://athena-results/
```

### 11.2 AWS CDK による S3 バケット定義

```typescript
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as s3n from 'aws-cdk-lib/aws-s3-notifications';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

export class S3AdvancedStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // メインバケット（バージョニング + ライフサイクル）
    const mainBucket = new s3.Bucket(this, 'MainBucket', {
      bucketName: `${this.stackName}-main-data`,
      versioned: true,
      encryption: s3.BucketEncryption.KMS_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      lifecycleRules: [
        {
          id: 'TransitionToIA',
          transitions: [
            {
              storageClass: s3.StorageClass.INFREQUENT_ACCESS,
              transitionAfter: cdk.Duration.days(30),
            },
            {
              storageClass: s3.StorageClass.INTELLIGENT_TIERING,
              transitionAfter: cdk.Duration.days(60),
            },
          ],
          noncurrentVersionTransitions: [
            {
              storageClass: s3.StorageClass.GLACIER,
              transitionAfter: cdk.Duration.days(90),
            },
          ],
          noncurrentVersionExpiration: cdk.Duration.days(365),
          abortIncompleteMultipartUploadAfter: cdk.Duration.days(7),
        },
      ],
      intelligentTieringConfigurations: [
        {
          name: 'archive-config',
          archiveAccessTierTime: cdk.Duration.days(90),
          deepArchiveAccessTierTime: cdk.Duration.days(180),
        },
      ],
    });

    // DR 用レプリカバケット
    const replicaBucket = new s3.Bucket(this, 'ReplicaBucket', {
      bucketName: `${this.stackName}-replica`,
      versioned: true,
      encryption: s3.BucketEncryption.KMS_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
    });

    // イベント通知: 画像アップロード時に Lambda を実行
    const imageProcessor = new lambda.Function(this, 'ImageProcessor', {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'index.handler',
      code: lambda.Code.fromAsset('lambda/image-processor'),
      memorySize: 1024,
      timeout: cdk.Duration.minutes(5),
    });

    mainBucket.addEventNotification(
      s3.EventType.OBJECT_CREATED,
      new s3n.LambdaDestination(imageProcessor),
      { prefix: 'uploads/images/', suffix: '.jpg' }
    );

    mainBucket.grantRead(imageProcessor);

    // S3 Access Point
    new s3.CfnAccessPoint(this, 'ReadOnlyAccessPoint', {
      bucket: mainBucket.bucketName,
      name: 'readonly-ap',
      vpcConfiguration: {
        vpcId: 'vpc-xxx',
      },
      publicAccessBlockConfiguration: {
        blockPublicAcls: true,
        blockPublicPolicy: true,
        ignorePublicAcls: true,
        restrictPublicBuckets: true,
      },
    });
  }
}
```

---

## 12. アンチパターン

### アンチパターン 1: バージョニング有効化後にライフサイクルを設定しない

バージョニングを有効にすると全バージョンが保持されるため、ストレージコストが際限なく増加する。NoncurrentVersionExpiration を必ず設定すべきである。

```json
{
  "Rules": [{
    "ID": "ExpireOldVersions",
    "Status": "Enabled",
    "Filter": {"Prefix": ""},
    "NoncurrentVersionTransitions": [
      {"NoncurrentDays": 30, "StorageClass": "STANDARD_IA"},
      {"NoncurrentDays": 90, "StorageClass": "GLACIER"}
    ],
    "NoncurrentVersionExpiration": {
      "NoncurrentDays": 365,
      "NewerNoncurrentVersions": 3
    }
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
ログデータ → S3（追記のみ）
```

### アンチパターン 3: パブリックバケットの放置

S3 バケットのパブリックアクセスを有効にしたまま放置すると、データ漏洩のリスクがある。Block Public Access を必ず有効にする。

```bash
# 全アカウントレベルでパブリックアクセスをブロック
aws s3control put-public-access-block \
  --account-id 123456789012 \
  --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

### アンチパターン 4: 大量の小さなオブジェクトを個別にアップロード

数百万の小さなファイル（1KB 未満）を個別に PUT すると、リクエスト料金が保存料金を大幅に上回る場合がある。

```
# 悪い例
100万ファイル × 1KB = 1GB (保存: ~$0.025)
100万 PUT リクエスト = $5.00 (リクエスト料金の方が200倍高い)

# 良い例
小ファイルを tarball にまとめてから S3 に PUT
アプリケーション側で S3 Select や Athena を使って個別アクセス
```

---

## 13. FAQ

### Q1. S3 バッチオペレーションとは何か？

大量のオブジェクト（数十億件）に対して一括操作を実行するサービス。ストレージクラスの一括変更、タグの追加、ACL の更新、Lambda 関数の実行などが可能。S3 インベントリレポートを入力として使用する。

### Q2. S3 Object Lock と Glacier Vault Lock の違いは？

S3 Object Lock は WORM (Write Once Read Many) モデルでオブジェクトの削除・上書きを防止する。Governance モード（特権ユーザーは解除可能）と Compliance モード（誰も解除不可）がある。Glacier Vault Lock は Glacier 専用の同様の機能。コンプライアンス要件で使い分ける。

### Q3. Requester Pays バケットとは？

通常はバケット所有者がデータ転送料金を負担するが、Requester Pays を有効にするとリクエスト元が料金を負担する。大規模な公開データセット（ゲノムデータ等）で利用される。

### Q4. S3 マルチパートアップロードのベストプラクティスは？

100MB 以上のファイルにはマルチパートアップロードを使用する。パートサイズは 5MB〜5GB、最大パート数は 10,000。失敗したパートの再試行が可能で、並列アップロードで高スループットを実現できる。AbortIncompleteMultipartUpload ライフサイクルルールで不完全なアップロードを自動クリーンアップする。

### Q5. S3 の結果整合性モデルはどうなったか？

2020年12月以降、S3 は全ての操作（PUT、DELETE、LIST を含む）で強い読み取り整合性（strong read-after-write consistency）を提供する。新しいオブジェクトの PUT 直後の GET、DELETE 直後の LIST など、全てのケースで最新のデータが返る。追加コストやパフォーマンス影響なしで提供される。

### Q6. S3 Express One Zone とは何か？

2023年に発表された高性能ストレージクラス。単一 AZ に配置され、一桁ミリ秒のレイテンシを提供する。ディレクトリバケットという新しいバケットタイプを使用し、通常の S3 API と互換性がある。ML トレーニングデータ、リアルタイム分析、HPC ワークロードに適している。

---

## 14. まとめ

| 項目 | ポイント |
|------|---------|
| バージョニング | 誤削除防止、ただしライフサイクルで旧バージョンを期限管理 |
| CRR / SRR | 災害復旧・低レイテンシ / ログ集約・環境コピー |
| S3 Select | SQL でフィルタし転送量を最大 99% 削減 |
| Transfer Acceleration | CloudFront Edge 経由でグローバルアップロード高速化 |
| コスト最適化 | Intelligent-Tiering + ライフサイクル + VPC エンドポイント |
| イベント通知 | Lambda/SQS/SNS/EventBridge と連携 |
| Object Lock | WORM モデルでコンプライアンス対応 |
| バッチオペレーション | 数十億オブジェクトの一括処理 |
| セキュリティ | Block Public Access + SSE-KMS + VPC エンドポイント + バケットポリシー |
| パフォーマンス | プレフィックス設計 + 並列アクセス + マルチパートアップロード |

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
5. S3 Object Lock — https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock.html
6. S3 Batch Operations — https://docs.aws.amazon.com/AmazonS3/latest/userguide/batch-ops.html
7. S3 セキュリティベストプラクティス — https://docs.aws.amazon.com/AmazonS3/latest/userguide/security-best-practices.html
8. S3 Storage Lens — https://docs.aws.amazon.com/AmazonS3/latest/userguide/storage_lens.html

# S3 基礎

> AWS のオブジェクトストレージ S3 の基本概念 — バケット、オブジェクト、アクセス制御、ライフサイクル、静的ホスティング

## この章で学ぶこと

1. S3 のバケットとオブジェクトの概念を理解し、基本的な CRUD 操作ができる
2. バケットポリシー、ACL、パブリックアクセスブロックを使った適切なアクセス制御を設計できる
3. ライフサイクルルールによるコスト最適化と静的 Web サイトホスティングを設定できる

---

## 1. S3 とは

Amazon Simple Storage Service (S3) は、99.999999999% (イレブンナイン) の耐久性を持つオブジェクトストレージサービスである。

### 1.1 S3 の基本構造

```
S3 のデータモデル

  +------------------------------------------+
  |  AWS Account                              |
  |                                           |
  |  +------------------------------------+   |
  |  |  Bucket: my-app-images             |   |
  |  |  (グローバルに一意な名前)             |   |
  |  |                                    |   |
  |  |  +------------------------------+  |   |
  |  |  | Object                       |  |   |
  |  |  | Key: photos/2024/cat.jpg     |  |   |
  |  |  | Value: (バイナリデータ)        |  |   |
  |  |  | Metadata: Content-Type, etc.  |  |   |
  |  |  | Size: 最大 5TB               |  |   |
  |  |  +------------------------------+  |   |
  |  |                                    |   |
  |  |  +------------------------------+  |   |
  |  |  | Object                       |  |   |
  |  |  | Key: docs/report.pdf         |  |   |
  |  |  +------------------------------+  |   |
  |  +------------------------------------+   |
  +------------------------------------------+

  ※ S3 にディレクトリの概念はない
    "photos/2024/cat.jpg" はフラットなキー名
    コンソールではプレフィックスをフォルダとして表示
```

### 1.2 S3 の特性

```
+--------------------------------------------------+
|              S3 の主要特性                          |
+--------------------------------------------------+
| 耐久性:  99.999999999% (11 nines)                 |
| 可用性:  99.99% (Standard)                        |
| 容量:    無制限（オブジェクトあたり最大 5TB）          |
| 整合性:  強い読み取り整合性 (2020年12月~)            |
| 暗号化:  サーバーサイド/クライアントサイド対応          |
| バージョニング: オブジェクト単位で全バージョン保持     |
+--------------------------------------------------+
```

---

## 2. バケットとオブジェクトの操作

### 2.1 コード例: AWS CLI での基本操作

```bash
# バケットの作成
aws s3 mb s3://my-app-bucket-2024 --region ap-northeast-1

# ファイルのアップロード
aws s3 cp ./index.html s3://my-app-bucket-2024/
aws s3 cp ./images/ s3://my-app-bucket-2024/images/ --recursive

# ファイルの同期（差分のみ転送）
aws s3 sync ./dist/ s3://my-app-bucket-2024/ --delete

# ファイルのダウンロード
aws s3 cp s3://my-app-bucket-2024/report.pdf ./

# オブジェクトの一覧
aws s3 ls s3://my-app-bucket-2024/
aws s3 ls s3://my-app-bucket-2024/ --recursive --summarize

# オブジェクトの削除
aws s3 rm s3://my-app-bucket-2024/old-file.txt
aws s3 rm s3://my-app-bucket-2024/temp/ --recursive
```

### 2.2 コード例: Python (boto3) での操作

```python
import boto3
import json

s3 = boto3.client('s3', region_name='ap-northeast-1')

# バケット作成
s3.create_bucket(
    Bucket='my-app-bucket-2024',
    CreateBucketConfiguration={'LocationConstraint': 'ap-northeast-1'}
)

# ファイルアップロード
s3.upload_file(
    Filename='./report.pdf',
    Bucket='my-app-bucket-2024',
    Key='docs/report.pdf',
    ExtraArgs={
        'ContentType': 'application/pdf',
        'ServerSideEncryption': 'AES256',
        'Metadata': {'author': 'tanaka', 'version': '1.0'}
    }
)

# JSON データの書き込み
s3.put_object(
    Bucket='my-app-bucket-2024',
    Key='data/config.json',
    Body=json.dumps({'env': 'production', 'debug': False}),
    ContentType='application/json'
)

# ファイルダウンロード
s3.download_file('my-app-bucket-2024', 'docs/report.pdf', './downloaded.pdf')

# オブジェクト一覧（ページネーション対応）
paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket='my-app-bucket-2024', Prefix='images/'):
    for obj in page.get('Contents', []):
        print(f"{obj['Key']} - {obj['Size']} bytes")
```

### 2.3 コード例: JavaScript (SDK v3) での操作

```javascript
import {
  S3Client,
  PutObjectCommand,
  GetObjectCommand,
} from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import { readFile } from 'fs/promises';

const s3 = new S3Client({ region: 'ap-northeast-1' });

// ファイルアップロード
async function upload(bucket, key, filePath) {
  const body = await readFile(filePath);
  await s3.send(new PutObjectCommand({
    Bucket: bucket,
    Key: key,
    Body: body,
    ContentType: 'image/jpeg',
    ServerSideEncryption: 'AES256',
  }));
}

// Presigned URL の生成（期限付きアクセス）
async function getPresignedUrl(bucket, key) {
  const command = new GetObjectCommand({ Bucket: bucket, Key: key });
  const url = await getSignedUrl(s3, command, { expiresIn: 3600 });
  return url;
}
```

---

## 3. ストレージクラス

### 3.1 ストレージクラス比較

| クラス | 可用性 | 最小保存期間 | 取り出し料金 | ユースケース |
|--------|--------|------------|------------|------------|
| Standard | 99.99% | なし | なし | 頻繁なアクセス |
| Intelligent-Tiering | 99.9% | なし | なし | アクセスパターン不明 |
| Standard-IA | 99.9% | 30日 | あり | 低頻度アクセス |
| One Zone-IA | 99.5% | 30日 | あり | 再作成可能なデータ |
| Glacier Instant Retrieval | 99.9% | 90日 | あり | 四半期に1回アクセス |
| Glacier Flexible Retrieval | 99.99% | 90日 | あり | 年に1-2回アクセス |
| Glacier Deep Archive | 99.99% | 180日 | あり | コンプライアンス保管 |

### 3.2 料金比較 (東京リージョン概算)

| クラス | 保存料金 (GB/月) | 取り出し (GB) |
|--------|-----------------|-------------|
| Standard | $0.025 | 無料 |
| Standard-IA | $0.019 | $0.01 |
| One Zone-IA | $0.015 | $0.01 |
| Glacier Instant | $0.005 | $0.03 |
| Glacier Flexible | $0.0045 | $0.01-$0.03 |
| Deep Archive | $0.002 | $0.02-$0.05 |

```
ストレージクラス選択フロー

  アクセス頻度は？
  ├── 毎日/毎週 → Standard
  ├── 不明 → Intelligent-Tiering
  ├── 月に数回 → Standard-IA
  ├── 四半期に1回 → Glacier Instant Retrieval
  ├── 年に1-2回 → Glacier Flexible Retrieval
  └── ほぼアクセスしない → Glacier Deep Archive

  再作成可能？
  ├── Yes → One Zone-IA (IA より安い)
  └── No  → Standard-IA (マルチ AZ)
```

---

## 4. アクセス制御

### 4.1 アクセス制御の層

```
S3 アクセス制御の4層

  +------------------------------------------+
  | 1. パブリックアクセスブロック (最優先)       |
  |    アカウント/バケットレベルで公開を防止     |
  +------------------------------------------+
              ↓
  +------------------------------------------+
  | 2. バケットポリシー (リソースベース)         |
  |    JSON で許可/拒否ルールを定義            |
  +------------------------------------------+
              ↓
  +------------------------------------------+
  | 3. IAM ポリシー (ID ベース)                |
  |    ユーザー/ロールに S3 権限を付与          |
  +------------------------------------------+
              ↓
  +------------------------------------------+
  | 4. ACL (レガシー、非推奨)                   |
  |    オブジェクト単位の読み書き権限            |
  +------------------------------------------+
```

### 4.2 コード例: パブリックアクセスブロック

```bash
# パブリックアクセスを完全にブロック（推奨）
aws s3api put-public-access-block \
  --bucket my-app-bucket-2024 \
  --public-access-block-configuration '{
    "BlockPublicAcls": true,
    "IgnorePublicAcls": true,
    "BlockPublicPolicy": true,
    "RestrictPublicBuckets": true
  }'

# 確認
aws s3api get-public-access-block --bucket my-app-bucket-2024
```

### 4.3 コード例: バケットポリシー

```bash
# 特定 IAM ロールからのみアクセス可能なポリシー
aws s3api put-bucket-policy --bucket my-app-bucket-2024 --policy '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowAppRoleAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::123456789012:role/AppServerRole"
      },
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-app-bucket-2024",
        "arn:aws:s3:::my-app-bucket-2024/*"
      ]
    },
    {
      "Sid": "DenyUnencryptedUploads",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::my-app-bucket-2024/*",
      "Condition": {
        "StringNotEquals": {
          "s3:x-amz-server-side-encryption": "AES256"
        }
      }
    }
  ]
}'
```

---

## 5. ライフサイクルルール

### 5.1 ライフサイクルの遷移パス

```
オブジェクトのライフサイクル遷移

  Standard
     |
     | 30日後
     v
  Standard-IA / Intelligent-Tiering
     |
     | 60日後
     v
  Glacier Instant Retrieval
     |
     | 90日後
     v
  Glacier Flexible Retrieval
     |
     | 180日後
     v
  Glacier Deep Archive
     |
     | 365日後
     v
  削除 (Expiration)
```

### 5.2 コード例: ライフサイクルルールの設定

```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket my-app-bucket-2024 \
  --lifecycle-configuration '{
  "Rules": [
    {
      "ID": "ArchiveAndExpire",
      "Status": "Enabled",
      "Filter": {"Prefix": "logs/"},
      "Transitions": [
        {"Days": 30, "StorageClass": "STANDARD_IA"},
        {"Days": 90, "StorageClass": "GLACIER"},
        {"Days": 365, "StorageClass": "DEEP_ARCHIVE"}
      ],
      "Expiration": {"Days": 730}
    },
    {
      "ID": "CleanupIncompleteUploads",
      "Status": "Enabled",
      "Filter": {"Prefix": ""},
      "AbortIncompleteMultipartUpload": {
        "DaysAfterInitiation": 7
      }
    },
    {
      "ID": "ExpireOldVersions",
      "Status": "Enabled",
      "Filter": {"Prefix": ""},
      "NoncurrentVersionTransitions": [
        {"NoncurrentDays": 30, "StorageClass": "STANDARD_IA"},
        {"NoncurrentDays": 90, "StorageClass": "GLACIER"}
      ],
      "NoncurrentVersionExpiration": {"NoncurrentDays": 365}
    }
  ]
}'
```

---

## 6. 静的 Web サイトホスティング

### 6.1 アーキテクチャ

```
S3 静的ホスティング構成

  ユーザー
    |
    v
  Route 53 (DNS)
    |
    v
  CloudFront (CDN, HTTPS)
    |
    v
  S3 Bucket (OAC 経由)
  ├── index.html
  ├── error.html
  ├── css/
  ├── js/
  └── images/
```

### 6.2 コード例: 静的ホスティングの設定

```bash
# バケットの静的ホスティングを有効化
aws s3 website s3://my-website-bucket \
  --index-document index.html \
  --error-document error.html

# ファイルをアップロード
aws s3 sync ./build/ s3://my-website-bucket/ \
  --cache-control "public, max-age=31536000" \
  --exclude "index.html" \
  --exclude "*.json"

# index.html はキャッシュしない
aws s3 cp ./build/index.html s3://my-website-bucket/ \
  --cache-control "no-cache, no-store, must-revalidate"

# エンドポイント: http://my-website-bucket.s3-website-ap-northeast-1.amazonaws.com
```

### 6.3 コード例: CloudFront + OAC 構成（推奨）

```bash
# OAC (Origin Access Control) を作成
OAC_ID=$(aws cloudfront create-origin-access-control \
  --origin-access-control-config '{
    "Name": "S3-OAC",
    "OriginAccessControlOriginType": "s3",
    "SigningBehavior": "always",
    "SigningProtocol": "sigv4"
  }' --query 'OriginAccessControl.Id' --output text)

# バケットポリシーで CloudFront からのアクセスのみ許可
aws s3api put-bucket-policy --bucket my-website-bucket --policy '{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "AllowCloudFrontOAC",
    "Effect": "Allow",
    "Principal": {"Service": "cloudfront.amazonaws.com"},
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::my-website-bucket/*",
    "Condition": {
      "StringEquals": {
        "AWS:SourceArn": "arn:aws:cloudfront::123456789012:distribution/EXXXXXXXXX"
      }
    }
  }]
}'
```

---

## 7. サーバーサイド暗号化

| 方式 | キー管理 | コスト | ユースケース |
|------|---------|--------|------------|
| SSE-S3 (AES256) | AWS 管理 | 無料 | デフォルト推奨 |
| SSE-KMS | KMS で管理 | KMS 料金 | 監査・キーローテーション |
| SSE-KMS (CMK) | ユーザー管理 | KMS 料金 | クロスアカウント |
| SSE-C | ユーザー提供 | 無料 | 独自キー管理 |

```bash
# デフォルト暗号化を有効化
aws s3api put-bucket-encryption --bucket my-app-bucket-2024 \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      },
      "BucketKeyEnabled": true
    }]
  }'
```

---

## 8. アンチパターン

### アンチパターン 1: バケットを公開設定のままにする

S3 バケットの公開設定は過去に大規模なデータ漏洩を引き起こしてきた。パブリックアクセスブロックを必ず有効にし、公開が必要な場合は CloudFront + OAC 経由とすべきである。

```bash
# 悪い例 — パブリック読み取りを許可
aws s3api put-bucket-acl --bucket my-bucket --acl public-read

# 良い例 — パブリックアクセスをブロックし、CloudFront 経由で配信
aws s3api put-public-access-block --bucket my-bucket \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

### アンチパターン 2: マルチパートアップロードの未完了を放置する

失敗したマルチパートアップロードの断片が残り続けると、ストレージコストが無駄に発生する。ライフサイクルルールで自動クリーンアップすべきである。

---

## 9. FAQ

### Q1. S3 のバケット名に制約はあるか？

グローバルに一意である必要があり、3-63文字、小文字・数字・ハイフンのみ使用可能。ピリオドは SSL 証明書の問題を起こすため避けるべきである。`my-company-app-prod` のような命名規則が推奨される。

### Q2. 5GB 以上のファイルをアップロードするには？

マルチパートアップロードを使用する。AWS CLI の `aws s3 cp` は自動的にマルチパートアップロードを行う（閾値はデフォルト 8MB）。SDK でも `upload` メソッドが自動分割する。最大 5TB まで対応。

### Q3. S3 のコストを削減するには？

(1) Intelligent-Tiering でアクセスパターンに応じた自動階層化、(2) ライフサイクルルールで古いデータを Glacier に移行、(3) 不完全なマルチパートアップロードの削除、(4) S3 Storage Lens でコスト分析を実施する。

---

## 10. まとめ

| 項目 | ポイント |
|------|---------|
| データモデル | バケット（名前空間）+ オブジェクト（キー + 値） |
| 耐久性 | 99.999999999% (イレブンナイン) |
| ストレージクラス | アクセス頻度に応じて Standard → IA → Glacier |
| アクセス制御 | パブリックアクセスブロック + バケットポリシー + IAM |
| 暗号化 | SSE-S3 をデフォルトで有効化 |
| ライフサイクル | 自動遷移 + 自動削除でコスト最適化 |
| 静的ホスティング | CloudFront + OAC が推奨構成 |

---

## 次に読むべきガイド

- [01-s3-advanced.md](./01-s3-advanced.md) — バージョニング、レプリケーション、S3 Select
- [02-cloudfront.md](./02-cloudfront.md) — CloudFront CDN 設定

---

## 参考文献

1. Amazon S3 ユーザーガイド — https://docs.aws.amazon.com/AmazonS3/latest/userguide/
2. S3 セキュリティベストプラクティス — https://docs.aws.amazon.com/AmazonS3/latest/userguide/security-best-practices.html
3. S3 料金 — https://aws.amazon.com/s3/pricing/
4. S3 ストレージクラス — https://aws.amazon.com/s3/storage-classes/

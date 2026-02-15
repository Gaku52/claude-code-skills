# S3 基礎

> AWS のオブジェクトストレージ S3 の基本概念 — バケット、オブジェクト、アクセス制御、ライフサイクル、静的ホスティング

## この章で学ぶこと

1. S3 のバケットとオブジェクトの概念を理解し、基本的な CRUD 操作ができる
2. バケットポリシー、ACL、パブリックアクセスブロックを使った適切なアクセス制御を設計できる
3. ライフサイクルルールによるコスト最適化と静的 Web サイトホスティングを設定できる
4. ストレージクラスの特性を理解し、ユースケースに応じた選択ができる
5. サーバーサイド暗号化とデータ保護の実装方法を習得する
6. S3 イベント通知やメトリクスを活用した運用監視ができる

---

## 1. S3 とは

Amazon Simple Storage Service (S3) は、99.999999999% (イレブンナイン) の耐久性を持つオブジェクトストレージサービスである。2006年のサービス開始以来、AWS の中核サービスとして数兆のオブジェクトを保管し、毎秒数百万のリクエストを処理している。

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

### 1.2 S3 の主要特性

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
| リージョン: バケットは特定リージョンに作成            |
| スケーリング: 自動スケーリング、プロビジョニング不要   |
+--------------------------------------------------+
```

### 1.3 S3 の整合性モデル

2020年12月以降、S3 は強い読み取り整合性（Strong Read-After-Write Consistency）を提供している。これにより、PUT/DELETE 直後の GET リクエストで最新のデータが返される。

```
S3 の整合性モデル（2020年12月以降）

  PUT Object (新規作成)
    → 直後の GET で新しいオブジェクトが返る ✓

  PUT Object (上書き)
    → 直後の GET で更新後のデータが返る ✓

  DELETE Object
    → 直後の GET で 404 が返る ✓

  LIST Objects
    → PUT/DELETE 直後の LIST に反映される ✓

  ※ 以前の「結果整合性」の問題は解消された
  ※ パフォーマンスへの影響もなし
```

### 1.4 S3 のリクエスト処理とパフォーマンス

S3 は、プレフィックスごとに毎秒 3,500 の PUT/COPY/POST/DELETE リクエストと 5,500 の GET/HEAD リクエストを処理できる。

```
パフォーマンスの考え方

  ■ プレフィックスの分散（パフォーマンス最適化）

  悪い例: すべてのキーが同じプレフィックス
    logs/2024-01-01.json
    logs/2024-01-02.json
    logs/2024-01-03.json
    → logs/ プレフィックスに負荷集中

  良い例: ハッシュプレフィックスで分散
    a1b2/logs/2024-01-01.json
    c3d4/logs/2024-01-02.json
    e5f6/logs/2024-01-03.json
    → 異なるパーティションに分散

  ※ 現在の S3 は内部的にキー名でパーティションを
    自動最適化するため、多くの場合この対策は不要
    （2018年以降のパフォーマンス改善による）
```

---

## 2. バケットとオブジェクトの操作

### 2.1 バケットの命名規則

```
バケット名のルール
├── 長さ: 3-63文字
├── 使用可能文字: 小文字、数字、ハイフン
├── 先頭: 小文字または数字
├── ピリオド: 使用可能だが SSL で問題あり（非推奨）
├── 一意性: グローバルに一意（全 AWS アカウント共通）
└── 予約名: "xn--" プレフィックスは使用不可

命名規則のベストプラクティス
├── {会社名}-{環境}-{用途}-{リージョン}
│   例: acme-prod-assets-ap-northeast-1
├── {プロジェクト}-{環境}-{サービス}
│   例: myapp-staging-uploads
└── 避けるべき名前
    ├── 汎用的な名前 (data, backup, files)
    ├── 個人情報を含む名前
    └── AWS アカウントIDを含む名前
```

### 2.2 AWS CLI での基本操作

```bash
# バケットの作成
aws s3 mb s3://my-app-bucket-2024 --region ap-northeast-1

# バケットの一覧表示
aws s3 ls

# ファイルのアップロード
aws s3 cp ./index.html s3://my-app-bucket-2024/
aws s3 cp ./images/ s3://my-app-bucket-2024/images/ --recursive

# Content-Type を指定してアップロード
aws s3 cp ./data.json s3://my-app-bucket-2024/data/ \
  --content-type "application/json" \
  --content-encoding "utf-8"

# メタデータ付きでアップロード
aws s3 cp ./report.pdf s3://my-app-bucket-2024/reports/ \
  --metadata '{"author":"tanaka","version":"2.0"}'

# ファイルの同期（差分のみ転送）
aws s3 sync ./dist/ s3://my-app-bucket-2024/ --delete

# 特定のファイルを除外して同期
aws s3 sync ./dist/ s3://my-app-bucket-2024/ \
  --exclude "*.log" \
  --exclude ".git/*" \
  --exclude "node_modules/*" \
  --delete

# ドライラン（実際にはコピーしない）
aws s3 sync ./dist/ s3://my-app-bucket-2024/ --dryrun

# ファイルのダウンロード
aws s3 cp s3://my-app-bucket-2024/report.pdf ./
aws s3 cp s3://my-app-bucket-2024/images/ ./local-images/ --recursive

# オブジェクトの一覧
aws s3 ls s3://my-app-bucket-2024/
aws s3 ls s3://my-app-bucket-2024/ --recursive --summarize
aws s3 ls s3://my-app-bucket-2024/ --recursive --human-readable --summarize

# 特定のプレフィックス配下を一覧
aws s3 ls s3://my-app-bucket-2024/logs/2024/01/

# オブジェクトの削除
aws s3 rm s3://my-app-bucket-2024/old-file.txt
aws s3 rm s3://my-app-bucket-2024/temp/ --recursive

# バケットの削除（空の場合のみ）
aws s3 rb s3://my-app-bucket-2024

# バケットの強制削除（中身ごと削除）
aws s3 rb s3://my-app-bucket-2024 --force

# オブジェクトのメタデータ確認
aws s3api head-object --bucket my-app-bucket-2024 --key report.pdf

# プレサインドURL の生成（1時間有効）
aws s3 presign s3://my-app-bucket-2024/private/report.pdf --expires-in 3600
```

### 2.3 s3api コマンドによる詳細操作

```bash
# バケット作成（s3api は低レベル API）
aws s3api create-bucket \
  --bucket my-app-bucket-2024 \
  --region ap-northeast-1 \
  --create-bucket-configuration LocationConstraint=ap-northeast-1

# オブジェクトのアップロード（詳細パラメータ指定）
aws s3api put-object \
  --bucket my-app-bucket-2024 \
  --key config/settings.json \
  --body ./settings.json \
  --content-type "application/json" \
  --server-side-encryption AES256 \
  --metadata '{"environment":"production"}' \
  --tagging "project=myapp&env=prod"

# オブジェクトの取得
aws s3api get-object \
  --bucket my-app-bucket-2024 \
  --key config/settings.json \
  ./downloaded-settings.json

# 条件付き取得（変更されていなければダウンロードしない）
aws s3api get-object \
  --bucket my-app-bucket-2024 \
  --key config/settings.json \
  --if-modified-since "2024-01-01T00:00:00Z" \
  ./downloaded-settings.json

# オブジェクトのタグ設定
aws s3api put-object-tagging \
  --bucket my-app-bucket-2024 \
  --key reports/monthly.pdf \
  --tagging '{
    "TagSet": [
      {"Key": "department", "Value": "finance"},
      {"Key": "classification", "Value": "confidential"}
    ]
  }'

# タグの取得
aws s3api get-object-tagging \
  --bucket my-app-bucket-2024 \
  --key reports/monthly.pdf

# オブジェクト一覧（ページネーション対応、最大1000件ずつ）
aws s3api list-objects-v2 \
  --bucket my-app-bucket-2024 \
  --prefix logs/ \
  --max-keys 100

# 続きを取得（ContinuationToken を使用）
aws s3api list-objects-v2 \
  --bucket my-app-bucket-2024 \
  --prefix logs/ \
  --continuation-token "TOKEN_FROM_PREVIOUS_RESPONSE"

# バケットのリージョン確認
aws s3api get-bucket-location --bucket my-app-bucket-2024
```

### 2.4 Python (boto3) での操作

```python
import boto3
import json
import os
from botocore.exceptions import ClientError
from datetime import datetime

# S3 クライアントの作成
s3_client = boto3.client('s3', region_name='ap-northeast-1')
s3_resource = boto3.resource('s3', region_name='ap-northeast-1')

# ===========================================
# バケット操作
# ===========================================

# バケット作成
def create_bucket(bucket_name: str, region: str = 'ap-northeast-1') -> bool:
    """S3 バケットを作成する"""
    try:
        s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={'LocationConstraint': region}
        )
        print(f"バケット '{bucket_name}' を作成しました")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            print(f"バケット '{bucket_name}' は既に存在します")
            return True
        print(f"エラー: {e}")
        return False

# バケット一覧
def list_buckets() -> list:
    """全バケットの一覧を取得する"""
    response = s3_client.list_buckets()
    buckets = []
    for bucket in response['Buckets']:
        buckets.append({
            'name': bucket['Name'],
            'created': bucket['CreationDate'].isoformat()
        })
    return buckets

# ===========================================
# ファイルアップロード
# ===========================================

# 基本的なファイルアップロード
def upload_file(file_path: str, bucket: str, key: str,
                content_type: str = None, metadata: dict = None) -> bool:
    """ファイルをS3にアップロードする"""
    extra_args = {
        'ServerSideEncryption': 'AES256',
    }
    if content_type:
        extra_args['ContentType'] = content_type
    if metadata:
        extra_args['Metadata'] = metadata

    try:
        s3_client.upload_file(
            Filename=file_path,
            Bucket=bucket,
            Key=key,
            ExtraArgs=extra_args
        )
        print(f"アップロード完了: {key}")
        return True
    except ClientError as e:
        print(f"アップロードエラー: {e}")
        return False

# JSON データの直接書き込み
def put_json(bucket: str, key: str, data: dict) -> bool:
    """JSON データを S3 に直接書き込む"""
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, ensure_ascii=False, indent=2),
            ContentType='application/json; charset=utf-8',
            ServerSideEncryption='AES256'
        )
        return True
    except ClientError as e:
        print(f"書き込みエラー: {e}")
        return False

# バイナリデータのアップロード
def upload_with_progress(file_path: str, bucket: str, key: str) -> bool:
    """プログレスバー付きでアップロードする"""
    file_size = os.path.getsize(file_path)
    uploaded = 0

    def progress_callback(bytes_transferred):
        nonlocal uploaded
        uploaded += bytes_transferred
        percentage = (uploaded / file_size) * 100
        print(f"\r進捗: {percentage:.1f}% ({uploaded}/{file_size} bytes)", end='')

    try:
        s3_client.upload_file(
            file_path, bucket, key,
            Callback=progress_callback,
            ExtraArgs={'ServerSideEncryption': 'AES256'}
        )
        print(f"\n完了: {key}")
        return True
    except ClientError as e:
        print(f"\nエラー: {e}")
        return False

# ===========================================
# ファイルダウンロード
# ===========================================

def download_file(bucket: str, key: str, local_path: str) -> bool:
    """S3 からファイルをダウンロードする"""
    try:
        # ディレクトリが存在しなければ作成
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(bucket, key, local_path)
        print(f"ダウンロード完了: {local_path}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"オブジェクトが見つかりません: {key}")
        else:
            print(f"エラー: {e}")
        return False

def get_json(bucket: str, key: str) -> dict:
    """S3 から JSON を読み込む"""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except ClientError as e:
        print(f"読み込みエラー: {e}")
        return None

# ===========================================
# オブジェクト一覧（ページネーション対応）
# ===========================================

def list_objects(bucket: str, prefix: str = '',
                max_keys: int = None) -> list:
    """オブジェクト一覧を取得する（ページネーション自動処理）"""
    paginator = s3_client.get_paginator('list_objects_v2')
    params = {'Bucket': bucket, 'Prefix': prefix}

    objects = []
    for page in paginator.paginate(**params):
        for obj in page.get('Contents', []):
            objects.append({
                'key': obj['Key'],
                'size': obj['Size'],
                'last_modified': obj['LastModified'].isoformat(),
                'storage_class': obj.get('StorageClass', 'STANDARD')
            })
            if max_keys and len(objects) >= max_keys:
                return objects
    return objects

def get_total_size(bucket: str, prefix: str = '') -> dict:
    """特定プレフィックス配下の合計サイズとオブジェクト数を算出する"""
    paginator = s3_client.get_paginator('list_objects_v2')
    total_size = 0
    total_count = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            total_size += obj['Size']
            total_count += 1

    return {
        'total_size_bytes': total_size,
        'total_size_mb': round(total_size / (1024 * 1024), 2),
        'total_size_gb': round(total_size / (1024 * 1024 * 1024), 4),
        'total_count': total_count
    }

# ===========================================
# オブジェクトの存在確認と情報取得
# ===========================================

def object_exists(bucket: str, key: str) -> bool:
    """オブジェクトの存在を確認する"""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise

def get_object_info(bucket: str, key: str) -> dict:
    """オブジェクトのメタデータを取得する"""
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        return {
            'content_type': response['ContentType'],
            'content_length': response['ContentLength'],
            'last_modified': response['LastModified'].isoformat(),
            'etag': response['ETag'],
            'metadata': response.get('Metadata', {}),
            'server_side_encryption': response.get('ServerSideEncryption'),
            'storage_class': response.get('StorageClass', 'STANDARD')
        }
    except ClientError as e:
        print(f"情報取得エラー: {e}")
        return None

# ===========================================
# 使用例
# ===========================================

if __name__ == '__main__':
    BUCKET = 'my-app-bucket-2024'

    # バケット作成
    create_bucket(BUCKET)

    # JSON データの保存
    config = {
        'app_name': 'MyApp',
        'version': '2.0',
        'features': {'dark_mode': True, 'notifications': True}
    }
    put_json(BUCKET, 'config/app-settings.json', config)

    # 読み込み
    loaded = get_json(BUCKET, 'config/app-settings.json')
    print(f"読み込んだ設定: {loaded}")

    # オブジェクト一覧
    objects = list_objects(BUCKET, prefix='config/')
    for obj in objects:
        print(f"  {obj['key']} ({obj['size']} bytes)")

    # 合計サイズ
    stats = get_total_size(BUCKET)
    print(f"合計: {stats['total_count']} ファイル, {stats['total_size_mb']} MB")
```

### 2.5 JavaScript (SDK v3) での操作

```javascript
import {
  S3Client,
  PutObjectCommand,
  GetObjectCommand,
  DeleteObjectCommand,
  ListObjectsV2Command,
  HeadObjectCommand,
  CopyObjectCommand,
} from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import { Upload } from '@aws-sdk/lib-storage';
import { readFile, writeFile, mkdir } from 'fs/promises';
import { createReadStream, createWriteStream } from 'fs';
import { pipeline } from 'stream/promises';
import path from 'path';

const s3 = new S3Client({ region: 'ap-northeast-1' });

// ====================================
// ファイルアップロード
// ====================================

/** 小さなファイルのアップロード */
async function uploadFile(bucket, key, filePath, contentType) {
  const body = await readFile(filePath);
  await s3.send(new PutObjectCommand({
    Bucket: bucket,
    Key: key,
    Body: body,
    ContentType: contentType || 'application/octet-stream',
    ServerSideEncryption: 'AES256',
  }));
  console.log(`アップロード完了: ${key}`);
}

/** JSON データの直接書き込み */
async function putJson(bucket, key, data) {
  await s3.send(new PutObjectCommand({
    Bucket: bucket,
    Key: key,
    Body: JSON.stringify(data, null, 2),
    ContentType: 'application/json; charset=utf-8',
    ServerSideEncryption: 'AES256',
  }));
}

/** 大きなファイルのマルチパートアップロード（プログレス付き） */
async function uploadLargeFile(bucket, key, filePath) {
  const stream = createReadStream(filePath);

  const upload = new Upload({
    client: s3,
    params: {
      Bucket: bucket,
      Key: key,
      Body: stream,
      ServerSideEncryption: 'AES256',
    },
    // 5MB パートサイズ（最小値）
    partSize: 5 * 1024 * 1024,
    // 並行アップロード数
    queueSize: 4,
  });

  upload.on('httpUploadProgress', (progress) => {
    const percentage = ((progress.loaded / progress.total) * 100).toFixed(1);
    process.stdout.write(`\r進捗: ${percentage}%`);
  });

  await upload.done();
  console.log(`\n完了: ${key}`);
}

// ====================================
// ファイルダウンロード
// ====================================

/** ファイルのダウンロード */
async function downloadFile(bucket, key, localPath) {
  const response = await s3.send(new GetObjectCommand({
    Bucket: bucket,
    Key: key,
  }));

  // ディレクトリ作成
  await mkdir(path.dirname(localPath), { recursive: true });

  // ストリームで保存
  const writeStream = createWriteStream(localPath);
  await pipeline(response.Body, writeStream);
  console.log(`ダウンロード完了: ${localPath}`);
}

/** JSON の読み込み */
async function getJson(bucket, key) {
  const response = await s3.send(new GetObjectCommand({
    Bucket: bucket,
    Key: key,
  }));
  const body = await response.Body.transformToString();
  return JSON.parse(body);
}

// ====================================
// Presigned URL の生成
// ====================================

/** ダウンロード用 Presigned URL */
async function getDownloadUrl(bucket, key, expiresIn = 3600) {
  const command = new GetObjectCommand({ Bucket: bucket, Key: key });
  return await getSignedUrl(s3, command, { expiresIn });
}

/** アップロード用 Presigned URL */
async function getUploadUrl(bucket, key, contentType, expiresIn = 3600) {
  const command = new PutObjectCommand({
    Bucket: bucket,
    Key: key,
    ContentType: contentType,
    ServerSideEncryption: 'AES256',
  });
  return await getSignedUrl(s3, command, { expiresIn });
}

// ====================================
// オブジェクト一覧
// ====================================

/** 全オブジェクト一覧（ページネーション自動処理） */
async function listAllObjects(bucket, prefix = '') {
  const objects = [];
  let continuationToken;

  do {
    const response = await s3.send(new ListObjectsV2Command({
      Bucket: bucket,
      Prefix: prefix,
      ContinuationToken: continuationToken,
    }));

    if (response.Contents) {
      objects.push(...response.Contents);
    }
    continuationToken = response.NextContinuationToken;
  } while (continuationToken);

  return objects;
}

// ====================================
// オブジェクトのコピーと移動
// ====================================

/** オブジェクトのコピー */
async function copyObject(bucket, sourceKey, destKey) {
  await s3.send(new CopyObjectCommand({
    Bucket: bucket,
    CopySource: `${bucket}/${sourceKey}`,
    Key: destKey,
    ServerSideEncryption: 'AES256',
  }));
  console.log(`コピー完了: ${sourceKey} → ${destKey}`);
}

/** オブジェクトの移動（コピー＋削除） */
async function moveObject(bucket, sourceKey, destKey) {
  await copyObject(bucket, sourceKey, destKey);
  await s3.send(new DeleteObjectCommand({
    Bucket: bucket,
    Key: sourceKey,
  }));
  console.log(`移動完了: ${sourceKey} → ${destKey}`);
}

// ====================================
// 使用例
// ====================================

async function main() {
  const bucket = 'my-app-bucket-2024';

  // JSON 保存・読み込み
  await putJson(bucket, 'config/settings.json', {
    theme: 'dark',
    language: 'ja',
  });

  const settings = await getJson(bucket, 'config/settings.json');
  console.log('設定:', settings);

  // Presigned URL 生成
  const url = await getDownloadUrl(bucket, 'private/report.pdf');
  console.log('ダウンロードURL:', url);
}

main().catch(console.error);
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
| Express One Zone | 99.95% | なし | なし | 超低レイテンシ |

### 3.2 料金比較 (東京リージョン概算)

| クラス | 保存料金 (GB/月) | 取り出し (GB) | 最小課金サイズ |
|--------|-----------------|-------------|-------------|
| Standard | $0.025 | 無料 | なし |
| Intelligent-Tiering | $0.025 (高頻度) | 無料 | なし |
| Standard-IA | $0.019 | $0.01 | 128KB |
| One Zone-IA | $0.015 | $0.01 | 128KB |
| Glacier Instant | $0.005 | $0.03 | 128KB |
| Glacier Flexible | $0.0045 | $0.01-$0.03 | 40KB |
| Deep Archive | $0.002 | $0.02-$0.05 | 40KB |

```
ストレージクラス選択フロー

  アクセス頻度は？
  ├── ミリ秒レイテンシ必須 → Express One Zone
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

### 3.3 Intelligent-Tiering 詳細

S3 Intelligent-Tiering は、アクセスパターンに基づいてオブジェクトを自動的に最適なアクセス層に移動するストレージクラスである。

```
Intelligent-Tiering のアクセス層

  +--------------------------------------+
  | 高頻度アクセス層 (Frequent Access)      |
  | → Standard と同じ料金                  |
  | → アクセス直後にここに移動             |
  +--------------------------------------+
         |
         | 30日間アクセスなし
         v
  +--------------------------------------+
  | 低頻度アクセス層 (Infrequent Access)    |
  | → Standard-IA と同じ料金               |
  +--------------------------------------+
         |
         | 90日間アクセスなし（オプトイン）
         v
  +--------------------------------------+
  | アーカイブアクセス層 (Archive Access)    |
  | → Glacier Flexible と同じ料金          |
  +--------------------------------------+
         |
         | 180日間アクセスなし（オプトイン）
         v
  +--------------------------------------+
  | ディープアーカイブアクセス層              |
  | → Deep Archive と同じ料金              |
  +--------------------------------------+

  ※ 監視・自動階層化料金: オブジェクトあたり $0.0025/月
  ※ 128KB 未満のオブジェクトは常に高頻度アクセス層
  ※ 取り出し料金なし（アーカイブ層からの取得を除く）
```

```bash
# Intelligent-Tiering の設定
# アーカイブアクセス層を有効化
aws s3api put-bucket-intelligent-tiering-configuration \
  --bucket my-app-bucket-2024 \
  --id "ArchiveConfig" \
  --intelligent-tiering-configuration '{
    "Id": "ArchiveConfig",
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

# ストレージクラスを指定してアップロード
aws s3 cp ./file.txt s3://my-app-bucket-2024/ \
  --storage-class INTELLIGENT_TIERING
```

### 3.4 Glacier からのデータ復元

```bash
# Glacier Flexible Retrieval からの復元
# Standard (3-5時間)
aws s3api restore-object \
  --bucket my-app-bucket-2024 \
  --key archives/old-data.tar.gz \
  --restore-request '{
    "Days": 7,
    "GlacierJobParameters": {
      "Tier": "Standard"
    }
  }'

# Expedited (1-5分、追加料金あり)
aws s3api restore-object \
  --bucket my-app-bucket-2024 \
  --key archives/urgent-data.tar.gz \
  --restore-request '{
    "Days": 1,
    "GlacierJobParameters": {
      "Tier": "Expedited"
    }
  }'

# Bulk (5-12時間、最安)
aws s3api restore-object \
  --bucket my-app-bucket-2024 \
  --key archives/bulk-data.tar.gz \
  --restore-request '{
    "Days": 30,
    "GlacierJobParameters": {
      "Tier": "Bulk"
    }
  }'

# 復元状態の確認
aws s3api head-object \
  --bucket my-app-bucket-2024 \
  --key archives/old-data.tar.gz \
  --query 'Restore'
# 出力例: "ongoing-request=\"false\", expiry-date=\"Sun, 01 Jan 2025 00:00:00 GMT\""
```

```python
# Python での Glacier 復元と状態監視
import boto3
import time
from datetime import datetime

s3 = boto3.client('s3', region_name='ap-northeast-1')

def restore_from_glacier(bucket: str, key: str, days: int = 7,
                         tier: str = 'Standard') -> dict:
    """Glacier からオブジェクトを復元する"""
    try:
        s3.restore_object(
            Bucket=bucket,
            Key=key,
            RestoreRequest={
                'Days': days,
                'GlacierJobParameters': {'Tier': tier}
            }
        )
        return {'status': 'initiated', 'key': key, 'tier': tier}
    except s3.exceptions.ClientError as e:
        if 'RestoreAlreadyInProgress' in str(e):
            return {'status': 'already_in_progress', 'key': key}
        raise

def check_restore_status(bucket: str, key: str) -> dict:
    """復元の進捗を確認する"""
    response = s3.head_object(Bucket=bucket, Key=key)
    restore = response.get('Restore', '')

    if not restore:
        return {'status': 'not_restored', 'storage_class': response.get('StorageClass')}
    elif 'ongoing-request="true"' in restore:
        return {'status': 'in_progress'}
    elif 'ongoing-request="false"' in restore:
        return {'status': 'completed', 'restore_info': restore}
    return {'status': 'unknown', 'raw': restore}

def wait_for_restore(bucket: str, key: str,
                     check_interval: int = 300, max_wait: int = 43200):
    """復元完了まで待機する（デフォルト最大12時間）"""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        status = check_restore_status(bucket, key)
        print(f"[{datetime.now().isoformat()}] 状態: {status['status']}")

        if status['status'] == 'completed':
            return status
        elif status['status'] == 'not_restored':
            raise Exception(f"復元が開始されていません: {key}")

        time.sleep(check_interval)

    raise TimeoutError(f"復元がタイムアウトしました: {key}")
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

  アクセス判定フロー:
  1. パブリックアクセスブロックでブロックされるか？
     → Yes: アクセス拒否
  2. 明示的な Deny があるか？
     → Yes: アクセス拒否
  3. 明示的な Allow があるか？
     → Yes: アクセス許可
  4. デフォルト: アクセス拒否（暗黙の Deny）
```

### 4.2 パブリックアクセスブロック

```bash
# アカウントレベルでパブリックアクセスをブロック（推奨）
aws s3control put-public-access-block \
  --account-id 123456789012 \
  --public-access-block-configuration '{
    "BlockPublicAcls": true,
    "IgnorePublicAcls": true,
    "BlockPublicPolicy": true,
    "RestrictPublicBuckets": true
  }'

# バケットレベルでパブリックアクセスをブロック
aws s3api put-public-access-block \
  --bucket my-app-bucket-2024 \
  --public-access-block-configuration '{
    "BlockPublicAcls": true,
    "IgnorePublicAcls": true,
    "BlockPublicPolicy": true,
    "RestrictPublicBuckets": true
  }'

# 設定の確認
aws s3api get-public-access-block --bucket my-app-bucket-2024
```

```
パブリックアクセスブロックの4つの設定

  BlockPublicAcls:
    → 新しいパブリック ACL の設定をブロック
    → PUT Object で public-read ACL を指定するとエラー

  IgnorePublicAcls:
    → 既存のパブリック ACL を無視
    → すでに設定済みの public-read ACL が無効になる

  BlockPublicPolicy:
    → パブリックアクセスを許可するバケットポリシーの設定をブロック
    → Principal: "*" のポリシーを PUT するとエラー

  RestrictPublicBuckets:
    → パブリックポリシーが設定されたバケットへの
      クロスアカウントアクセスを制限
```

### 4.3 バケットポリシーの実践パターン

```bash
# パターン1: 特定 IAM ロールからのみアクセス可能
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
    }
  ]
}'

# パターン2: 暗号化されていないアップロードを拒否
aws s3api put-bucket-policy --bucket my-app-bucket-2024 --policy '{
  "Version": "2012-10-17",
  "Statement": [
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

# パターン3: HTTPS のみ許可
aws s3api put-bucket-policy --bucket my-app-bucket-2024 --policy '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyInsecureTransport",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::my-app-bucket-2024",
        "arn:aws:s3:::my-app-bucket-2024/*"
      ],
      "Condition": {
        "Bool": {
          "aws:SecureTransport": "false"
        }
      }
    }
  ]
}'

# パターン4: 特定 VPC エンドポイントからのみアクセス許可
aws s3api put-bucket-policy --bucket my-app-bucket-2024 --policy '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowVPCEndpointOnly",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::my-app-bucket-2024",
        "arn:aws:s3:::my-app-bucket-2024/*"
      ],
      "Condition": {
        "StringNotEquals": {
          "aws:SourceVpce": "vpce-1234567890abcdef0"
        }
      }
    }
  ]
}'

# パターン5: IP アドレスによる制限
aws s3api put-bucket-policy --bucket my-app-bucket-2024 --policy '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowFromSpecificIP",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::my-app-bucket-2024",
        "arn:aws:s3:::my-app-bucket-2024/*"
      ],
      "Condition": {
        "NotIpAddress": {
          "aws:SourceIp": [
            "203.0.113.0/24",
            "198.51.100.0/24"
          ]
        }
      }
    }
  ]
}'

# パターン6: クロスアカウントアクセス
aws s3api put-bucket-policy --bucket my-app-bucket-2024 --policy '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowCrossAccountAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::987654321098:root"
      },
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-app-bucket-2024",
        "arn:aws:s3:::my-app-bucket-2024/shared/*"
      ]
    }
  ]
}'
```

### 4.4 CORS 設定

```bash
# CORS の設定（Web アプリからの直接アクセス用）
aws s3api put-bucket-cors --bucket my-app-bucket-2024 --cors-configuration '{
  "CORSRules": [
    {
      "AllowedHeaders": ["*"],
      "AllowedMethods": ["GET", "PUT", "POST"],
      "AllowedOrigins": ["https://myapp.example.com", "https://admin.example.com"],
      "ExposeHeaders": ["ETag", "x-amz-request-id"],
      "MaxAgeSeconds": 3600
    }
  ]
}'

# CORS 確認
aws s3api get-bucket-cors --bucket my-app-bucket-2024
```

### 4.5 ACL の無効化（推奨設定）

```bash
# オブジェクト所有権を BucketOwnerEnforced に設定
# → ACL が無効化され、バケット所有者が全オブジェクトを所有
aws s3api put-bucket-ownership-controls \
  --bucket my-app-bucket-2024 \
  --ownership-controls '{
    "Rules": [{"ObjectOwnership": "BucketOwnerEnforced"}]
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

  ※ 遷移の制約:
    - Standard-IA: 最低30日経過後
    - Glacier: 最低90日経過後
    - 最小オブジェクトサイズ: 128KB（IA系）/ 40KB（Glacier系）
    - One Zone-IA への遷移後、他の IA への遷移は不可
```

### 5.2 ライフサイクルルールの設定例

```bash
# 実践的なライフサイクル設定
aws s3api put-bucket-lifecycle-configuration \
  --bucket my-app-bucket-2024 \
  --lifecycle-configuration '{
  "Rules": [
    {
      "ID": "ArchiveLogs",
      "Status": "Enabled",
      "Filter": {"Prefix": "logs/"},
      "Transitions": [
        {"Days": 30, "StorageClass": "STANDARD_IA"},
        {"Days": 90, "StorageClass": "GLACIER"},
        {"Days": 365, "StorageClass": "DEEP_ARCHIVE"}
      ],
      "Expiration": {"Days": 2555}
    },
    {
      "ID": "CleanupTempFiles",
      "Status": "Enabled",
      "Filter": {"Prefix": "tmp/"},
      "Expiration": {"Days": 7}
    },
    {
      "ID": "ArchiveReports",
      "Status": "Enabled",
      "Filter": {
        "And": {
          "Prefix": "reports/",
          "Tags": [
            {"Key": "archive", "Value": "true"}
          ]
        }
      },
      "Transitions": [
        {"Days": 90, "StorageClass": "GLACIER_IR"}
      ]
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
    },
    {
      "ID": "ExpireDeleteMarkers",
      "Status": "Enabled",
      "Filter": {"Prefix": ""},
      "Expiration": {
        "ExpiredObjectDeleteMarker": true
      }
    },
    {
      "ID": "SmallObjectFilter",
      "Status": "Enabled",
      "Filter": {
        "And": {
          "Prefix": "data/",
          "ObjectSizeGreaterThan": 131072
        }
      },
      "Transitions": [
        {"Days": 60, "StorageClass": "STANDARD_IA"}
      ]
    }
  ]
}'

# ライフサイクル設定の確認
aws s3api get-bucket-lifecycle-configuration --bucket my-app-bucket-2024
```

### 5.3 コスト最適化シミュレーション

```python
# ライフサイクルルールによるコスト削減シミュレーション
def calculate_storage_cost(
    total_gb: float,
    access_pattern: str = 'standard',
    months: int = 12
) -> dict:
    """ストレージコストを概算する（東京リージョン）"""
    prices = {
        'STANDARD':     0.025,
        'STANDARD_IA':  0.019,
        'ONE_ZONE_IA':  0.015,
        'GLACIER_IR':   0.005,
        'GLACIER':      0.0045,
        'DEEP_ARCHIVE': 0.002,
    }

    # ライフサイクルなし（全期間 Standard）
    cost_no_lifecycle = total_gb * prices['STANDARD'] * months

    # ライフサイクルあり
    cost_with_lifecycle = 0
    for month in range(1, months + 1):
        if month <= 1:
            cost_with_lifecycle += total_gb * prices['STANDARD']
        elif month <= 3:
            cost_with_lifecycle += total_gb * prices['STANDARD_IA']
        elif month <= 6:
            cost_with_lifecycle += total_gb * prices['GLACIER_IR']
        elif month <= 12:
            cost_with_lifecycle += total_gb * prices['GLACIER']
        else:
            cost_with_lifecycle += total_gb * prices['DEEP_ARCHIVE']

    savings = cost_no_lifecycle - cost_with_lifecycle
    savings_pct = (savings / cost_no_lifecycle) * 100

    return {
        'without_lifecycle': round(cost_no_lifecycle, 2),
        'with_lifecycle': round(cost_with_lifecycle, 2),
        'savings': round(savings, 2),
        'savings_percentage': round(savings_pct, 1),
    }

# 使用例
result = calculate_storage_cost(total_gb=1000, months=12)
print(f"ライフサイクルなし: ${result['without_lifecycle']}")
print(f"ライフサイクルあり: ${result['with_lifecycle']}")
print(f"削減額: ${result['savings']} ({result['savings_percentage']}%)")
# ライフサイクルなし: $300.00
# ライフサイクルあり: $83.50
# 削減額: $216.50 (72.2%)
```

---

## 6. 静的 Web サイトホスティング

### 6.1 アーキテクチャ

```
S3 静的ホスティング構成（推奨）

  ユーザー
    |
    v
  Route 53 (DNS)
    |  example.com → CloudFront
    v
  CloudFront (CDN, HTTPS)
    |  キャッシュ、圧縮、WAF 統合
    v
  S3 Bucket (OAC 経由、非公開)
  ├── index.html
  ├── error.html
  ├── css/
  │   ├── main.abc123.css
  │   └── vendor.def456.css
  ├── js/
  │   ├── app.ghi789.js
  │   └── vendor.jkl012.js
  └── images/
      ├── logo.png
      └── hero.webp

  ※ S3 のバケットは非公開のまま
  ※ CloudFront OAC 経由でのみアクセス可能
  ※ バケットの静的ホスティング機能は不要
    （CloudFront がオリジンとして直接 S3 API を使用）
```

### 6.2 静的ホスティングの設定

```bash
# === 方法1: S3 単体での静的ホスティング（開発環境向け） ===

# バケットの静的ホスティングを有効化
aws s3 website s3://my-website-bucket \
  --index-document index.html \
  --error-document error.html

# SPA (Single Page Application) 用のリダイレクトルール
aws s3api put-bucket-website \
  --bucket my-website-bucket \
  --website-configuration '{
    "IndexDocument": {"Suffix": "index.html"},
    "ErrorDocument": {"Key": "index.html"},
    "RoutingRules": [
      {
        "Condition": {
          "HttpErrorCodeReturnedEquals": "404"
        },
        "Redirect": {
          "ReplaceKeyWith": "index.html",
          "HttpRedirectCode": "200"
        }
      }
    ]
  }'

# ファイルのアップロード（キャッシュ戦略付き）
# ハッシュ付きアセット（長期キャッシュ）
aws s3 sync ./build/static/ s3://my-website-bucket/static/ \
  --cache-control "public, max-age=31536000, immutable" \
  --exclude "*.map"

# index.html（キャッシュなし）
aws s3 cp ./build/index.html s3://my-website-bucket/ \
  --cache-control "no-cache, no-store, must-revalidate" \
  --content-type "text/html; charset=utf-8"

# その他の HTML ファイル
aws s3 sync ./build/ s3://my-website-bucket/ \
  --exclude "static/*" \
  --exclude "*.map" \
  --cache-control "public, max-age=0, must-revalidate"

# エンドポイント確認
echo "http://my-website-bucket.s3-website-ap-northeast-1.amazonaws.com"
```

### 6.3 CloudFront + OAC 構成（本番推奨）

```bash
# Step 1: OAC (Origin Access Control) を作成
OAC_ID=$(aws cloudfront create-origin-access-control \
  --origin-access-control-config '{
    "Name": "S3-Website-OAC",
    "Description": "OAC for S3 static website",
    "OriginAccessControlOriginType": "s3",
    "SigningBehavior": "always",
    "SigningProtocol": "sigv4"
  }' --query 'OriginAccessControl.Id' --output text)

echo "OAC ID: ${OAC_ID}"

# Step 2: CloudFront ディストリビューション作成
DIST_ID=$(aws cloudfront create-distribution \
  --distribution-config '{
    "CallerReference": "my-website-2024",
    "Comment": "Static website distribution",
    "Enabled": true,
    "DefaultRootObject": "index.html",
    "Origins": {
      "Quantity": 1,
      "Items": [{
        "Id": "S3Origin",
        "DomainName": "my-website-bucket.s3.ap-northeast-1.amazonaws.com",
        "OriginAccessControlId": "'${OAC_ID}'",
        "S3OriginConfig": {
          "OriginAccessIdentity": ""
        }
      }]
    },
    "DefaultCacheBehavior": {
      "TargetOriginId": "S3Origin",
      "ViewerProtocolPolicy": "redirect-to-https",
      "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6",
      "Compress": true,
      "AllowedMethods": {
        "Quantity": 2,
        "Items": ["GET", "HEAD"]
      }
    },
    "CustomErrorResponses": {
      "Quantity": 1,
      "Items": [{
        "ErrorCode": 404,
        "ResponsePagePath": "/index.html",
        "ResponseCode": "200",
        "ErrorCachingMinTTL": 0
      }]
    },
    "PriceClass": "PriceClass_200"
  }' --query 'Distribution.Id' --output text)

echo "Distribution ID: ${DIST_ID}"

# Step 3: バケットポリシーで CloudFront からのアクセスのみ許可
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
        "AWS:SourceArn": "arn:aws:cloudfront::123456789012:distribution/'${DIST_ID}'"
      }
    }
  }]
}'

# Step 4: キャッシュの無効化（デプロイ後）
aws cloudfront create-invalidation \
  --distribution-id ${DIST_ID} \
  --paths "/*"

# 特定パスのみ無効化（コスト削減）
aws cloudfront create-invalidation \
  --distribution-id ${DIST_ID} \
  --paths "/index.html" "/manifest.json"
```

### 6.4 デプロイスクリプト（実践例）

```bash
#!/bin/bash
# deploy-static-site.sh - S3 + CloudFront への静的サイトデプロイ

set -euo pipefail

BUCKET="my-website-bucket"
DIST_ID="E1234567890ABC"
BUILD_DIR="./build"

echo "=== ビルド ==="
npm run build

echo "=== ハッシュ付きアセットのアップロード（長期キャッシュ） ==="
aws s3 sync "${BUILD_DIR}/static/" "s3://${BUCKET}/static/" \
  --cache-control "public, max-age=31536000, immutable" \
  --exclude "*.map" \
  --size-only

echo "=== HTML ファイルのアップロード（キャッシュなし） ==="
aws s3 cp "${BUILD_DIR}/index.html" "s3://${BUCKET}/index.html" \
  --cache-control "no-cache, no-store, must-revalidate" \
  --content-type "text/html; charset=utf-8"

echo "=== その他ファイルのアップロード ==="
aws s3 sync "${BUILD_DIR}/" "s3://${BUCKET}/" \
  --exclude "static/*" \
  --exclude "*.map" \
  --exclude "index.html" \
  --cache-control "public, max-age=3600"

echo "=== CloudFront キャッシュ無効化 ==="
INVALIDATION_ID=$(aws cloudfront create-invalidation \
  --distribution-id "${DIST_ID}" \
  --paths "/index.html" "/manifest.json" "/service-worker.js" \
  --query 'Invalidation.Id' --output text)

echo "無効化ID: ${INVALIDATION_ID}"

echo "=== 無効化完了を待機 ==="
aws cloudfront wait invalidation-completed \
  --distribution-id "${DIST_ID}" \
  --id "${INVALIDATION_ID}"

echo "=== デプロイ完了 ==="
```

---

## 7. サーバーサイド暗号化

### 7.1 暗号化方式の比較

| 方式 | キー管理 | コスト | ユースケース | API 呼び出し制限 |
|------|---------|--------|------------|----------------|
| SSE-S3 (AES256) | AWS 管理 | 無料 | デフォルト推奨 | なし |
| SSE-KMS (aws/s3) | AWS 管理 KMS キー | KMS 料金 | 監査ログ必要時 | KMS クォータ |
| SSE-KMS (CMK) | ユーザー管理キー | KMS 料金 | クロスアカウント | KMS クォータ |
| SSE-C | ユーザー提供キー | 無料 | 独自キー管理 | HTTPS 必須 |
| CSE | クライアント側 | なし | 完全制御 | なし |

```
暗号化方式の選択フロー

  暗号化が必要？
  ├── デフォルトで暗号化したい → SSE-S3 (AES256)
  ├── キーの利用を監査したい → SSE-KMS (aws/s3)
  ├── キーのローテーションを制御したい → SSE-KMS (CMK)
  ├── クロスアカウントでキーを共有したい → SSE-KMS (CMK)
  ├── AWS にキーを預けたくない → SSE-C
  └── データをアップロード前に暗号化したい → CSE
```

### 7.2 暗号化の設定

```bash
# デフォルト暗号化を SSE-S3 (AES256) に設定
aws s3api put-bucket-encryption --bucket my-app-bucket-2024 \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      },
      "BucketKeyEnabled": true
    }]
  }'

# デフォルト暗号化を SSE-KMS に設定
aws s3api put-bucket-encryption --bucket my-app-bucket-2024 \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "aws:kms",
        "KMSMasterKeyID": "arn:aws:kms:ap-northeast-1:123456789012:key/12345678-1234-1234-1234-123456789012"
      },
      "BucketKeyEnabled": true
    }]
  }'

# 暗号化設定の確認
aws s3api get-bucket-encryption --bucket my-app-bucket-2024

# KMS キーでアップロード
aws s3 cp ./sensitive-data.csv s3://my-app-bucket-2024/data/ \
  --sse aws:kms \
  --sse-kms-key-id "arn:aws:kms:ap-northeast-1:123456789012:key/12345678-1234-1234-1234-123456789012"
```

```python
# Python での SSE-C（クライアント提供キー）によるアップロード/ダウンロード
import boto3
import hashlib
import base64
import os

s3 = boto3.client('s3', region_name='ap-northeast-1')

# 256ビットの暗号化キーを生成
encryption_key = os.urandom(32)
key_b64 = base64.b64encode(encryption_key).decode('utf-8')
key_md5 = base64.b64encode(
    hashlib.md5(encryption_key).digest()
).decode('utf-8')

# SSE-C でアップロード
s3.put_object(
    Bucket='my-app-bucket-2024',
    Key='encrypted/secret-data.bin',
    Body=b'This is secret data',
    SSECustomerAlgorithm='AES256',
    SSECustomerKey=key_b64,
    SSECustomerKeyMD5=key_md5
)

# SSE-C でダウンロード（同じキーが必要）
response = s3.get_object(
    Bucket='my-app-bucket-2024',
    Key='encrypted/secret-data.bin',
    SSECustomerAlgorithm='AES256',
    SSECustomerKey=key_b64,
    SSECustomerKeyMD5=key_md5
)
data = response['Body'].read()
print(f"復号データ: {data}")
```

---

## 8. S3 イベント通知

### 8.1 イベント通知のアーキテクチャ

```
S3 イベント通知の宛先

  S3 バケット
    |
    | イベント発生（PUT, DELETE, etc.）
    |
    ├── → SNS トピック → Email, SMS, HTTP
    ├── → SQS キュー → バッチ処理
    ├── → Lambda 関数 → リアルタイム処理
    └── → EventBridge → 複雑なルーティング

  主要なイベントタイプ:
  ├── s3:ObjectCreated:* (Put, Post, Copy, CompleteMultipartUpload)
  ├── s3:ObjectRemoved:* (Delete, DeleteMarkerCreated)
  ├── s3:ObjectRestore:* (Post, Completed)
  ├── s3:ReducedRedundancyLostObject
  ├── s3:Replication:*
  └── s3:LifecycleTransition
```

### 8.2 イベント通知の設定

```bash
# Lambda 関数へのイベント通知設定
aws s3api put-bucket-notification-configuration \
  --bucket my-app-bucket-2024 \
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
      }
    ],
    "QueueConfigurations": [
      {
        "Id": "LogProcessingQueue",
        "QueueArn": "arn:aws:sqs:ap-northeast-1:123456789012:log-processing",
        "Events": ["s3:ObjectCreated:*"],
        "Filter": {
          "Key": {
            "FilterRules": [
              {"Name": "prefix", "Value": "logs/"},
              {"Name": "suffix", "Value": ".gz"}
            ]
          }
        }
      }
    ],
    "EventBridgeConfiguration": {}
  }'
```

```python
# Lambda 関数でのイベント処理例
import boto3
import json
import urllib.parse

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """S3 イベントを処理する Lambda 関数"""
    for record in event['Records']:
        # イベント情報の取得
        bucket = record['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(record['s3']['object']['key'])
        size = record['s3']['object']['size']
        event_name = record['eventName']
        event_time = record['eventTime']

        print(f"イベント: {event_name}")
        print(f"バケット: {bucket}")
        print(f"キー: {key}")
        print(f"サイズ: {size} bytes")
        print(f"時刻: {event_time}")

        # オブジェクトの処理
        if event_name.startswith('ObjectCreated'):
            process_new_object(bucket, key)
        elif event_name.startswith('ObjectRemoved'):
            handle_deletion(bucket, key)

def process_new_object(bucket: str, key: str):
    """新しいオブジェクトを処理する"""
    # メタデータの取得
    response = s3.head_object(Bucket=bucket, Key=key)
    content_type = response['ContentType']

    # 画像の場合はサムネイル生成
    if content_type.startswith('image/'):
        generate_thumbnail(bucket, key)

    # ログファイルの場合は解析
    elif key.endswith('.log') or key.endswith('.gz'):
        analyze_log(bucket, key)

def generate_thumbnail(bucket: str, key: str):
    """サムネイルを生成する（簡略化）"""
    print(f"サムネイル生成: {key}")
    # Pillow 等を使ったサムネイル生成処理
    # ...

def handle_deletion(bucket: str, key: str):
    """削除イベントを処理する"""
    print(f"オブジェクト削除を検知: {key}")
    # 関連データのクリーンアップ等
```

---

## 9. S3 のモニタリングとメトリクス

### 9.1 CloudWatch メトリクス

```
S3 の標準メトリクス（無料）

  ├── BucketSizeBytes: バケットの合計サイズ
  ├── NumberOfObjects: オブジェクト数
  └── ※ 日次で更新（リアルタイムではない）

S3 リクエストメトリクス（有料、フィルタ設定が必要）

  ├── AllRequests: 全リクエスト数
  ├── GetRequests: GET リクエスト数
  ├── PutRequests: PUT リクエスト数
  ├── DeleteRequests: DELETE リクエスト数
  ├── HeadRequests: HEAD リクエスト数
  ├── ListRequests: LIST リクエスト数
  ├── 4xxErrors: クライアントエラー数
  ├── 5xxErrors: サーバーエラー数
  ├── FirstByteLatency: 最初のバイトまでのレイテンシ
  ├── TotalRequestLatency: 合計リクエストレイテンシ
  ├── BytesDownloaded: ダウンロードバイト数
  └── BytesUploaded: アップロードバイト数
```

```bash
# リクエストメトリクスの有効化
aws s3api put-bucket-metrics-configuration \
  --bucket my-app-bucket-2024 \
  --id AllRequests \
  --metrics-configuration '{
    "Id": "AllRequests",
    "Filter": {}
  }'

# 特定プレフィックスのメトリクス
aws s3api put-bucket-metrics-configuration \
  --bucket my-app-bucket-2024 \
  --id ApiRequests \
  --metrics-configuration '{
    "Id": "ApiRequests",
    "Filter": {
      "Prefix": "api/"
    }
  }'

# CloudWatch でメトリクスを取得
aws cloudwatch get-metric-statistics \
  --namespace AWS/S3 \
  --metric-name BucketSizeBytes \
  --dimensions Name=BucketName,Value=my-app-bucket-2024 \
               Name=StorageType,Value=StandardStorage \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-31T23:59:59Z \
  --period 86400 \
  --statistics Average
```

### 9.2 S3 アクセスログ

```bash
# アクセスログの有効化
# まずログ保存先バケットのポリシーを設定
aws s3api put-bucket-policy --bucket my-s3-access-logs --policy '{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "S3ServerAccessLogsPolicy",
    "Effect": "Allow",
    "Principal": {"Service": "logging.s3.amazonaws.com"},
    "Action": "s3:PutObject",
    "Resource": "arn:aws:s3:::my-s3-access-logs/*",
    "Condition": {
      "StringEquals": {
        "aws:SourceAccount": "123456789012"
      }
    }
  }]
}'

# アクセスログを有効化
aws s3api put-bucket-logging --bucket my-app-bucket-2024 --bucket-logging-status '{
  "LoggingEnabled": {
    "TargetBucket": "my-s3-access-logs",
    "TargetPrefix": "my-app-bucket-2024/"
  }
}'
```

### 9.3 S3 Storage Lens

```bash
# Storage Lens ダッシュボードの作成
aws s3control put-storage-lens-configuration \
  --account-id 123456789012 \
  --config-id my-storage-lens \
  --storage-lens-configuration '{
    "Id": "my-storage-lens",
    "AccountLevel": {
      "BucketLevel": {
        "ActivityMetrics": {
          "IsEnabled": true
        },
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

---

## 10. マルチパートアップロード

### 10.1 マルチパートアップロードの仕組み

```
マルチパートアップロードのフロー

  ファイル (100MB)
    |
    | 分割
    v
  +--------+--------+--------+--------+
  | Part 1 | Part 2 | Part 3 | ... N  |
  | 10MB   | 10MB   | 10MB   |        |
  +--------+--------+--------+--------+
    |         |         |         |
    | 並行     | 並行     | 並行     | 並行
    v         v         v         v
  S3 (一時パート保管)
    |
    | Complete Multipart Upload
    v
  完成したオブジェクト (100MB)

  制約:
  ├── 最小パートサイズ: 5MB（最後のパートを除く）
  ├── 最大パートサイズ: 5GB
  ├── 最大パート数: 10,000
  ├── 最大オブジェクトサイズ: 5TB
  └── 推奨: 100MB 以上のファイルでマルチパート使用
```

### 10.2 Python でのマルチパートアップロード

```python
import boto3
import os
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

s3 = boto3.client('s3', region_name='ap-northeast-1')

def multipart_upload(file_path: str, bucket: str, key: str,
                     part_size: int = 50 * 1024 * 1024):
    """マルチパートアップロードを実行する"""
    file_size = os.path.getsize(file_path)
    total_parts = math.ceil(file_size / part_size)

    print(f"ファイルサイズ: {file_size / (1024*1024):.1f} MB")
    print(f"パート数: {total_parts}")

    # Step 1: マルチパートアップロードの開始
    response = s3.create_multipart_upload(
        Bucket=bucket,
        Key=key,
        ServerSideEncryption='AES256'
    )
    upload_id = response['UploadId']
    print(f"Upload ID: {upload_id}")

    parts = []
    try:
        # Step 2: パートの並行アップロード
        def upload_part(part_number, start, end):
            with open(file_path, 'rb') as f:
                f.seek(start)
                data = f.read(end - start)

                response = s3.upload_part(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=data
                )
                return {
                    'PartNumber': part_number,
                    'ETag': response['ETag']
                }

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for i in range(total_parts):
                part_num = i + 1
                start = i * part_size
                end = min(start + part_size, file_size)
                future = executor.submit(upload_part, part_num, start, end)
                futures[future] = part_num

            for future in as_completed(futures):
                part = future.result()
                parts.append(part)
                print(f"  パート {part['PartNumber']}/{total_parts} 完了")

        # パート番号順にソート
        parts.sort(key=lambda x: x['PartNumber'])

        # Step 3: マルチパートアップロードの完了
        s3.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={'Parts': parts}
        )
        print(f"アップロード完了: {key}")

    except Exception as e:
        # エラー時はアップロードを中止
        s3.abort_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id
        )
        print(f"アップロード中止: {e}")
        raise

# 未完了のマルチパートアップロードを一覧・クリーンアップ
def cleanup_incomplete_uploads(bucket: str):
    """未完了のマルチパートアップロードを削除する"""
    response = s3.list_multipart_uploads(Bucket=bucket)
    uploads = response.get('Uploads', [])

    for upload in uploads:
        key = upload['Key']
        upload_id = upload['UploadId']
        initiated = upload['Initiated']

        print(f"未完了: {key} (開始: {initiated})")

        s3.abort_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id
        )
        print(f"  → 中止しました")

    print(f"合計 {len(uploads)} 件の未完了アップロードをクリーンアップしました")
```

---

## 11. S3 バージョニング

### 11.1 バージョニングの基本

```bash
# バージョニングの有効化
aws s3api put-bucket-versioning \
  --bucket my-app-bucket-2024 \
  --versioning-configuration Status=Enabled

# バージョニングの状態確認
aws s3api get-bucket-versioning --bucket my-app-bucket-2024

# 全バージョンの一覧
aws s3api list-object-versions \
  --bucket my-app-bucket-2024 \
  --prefix config/settings.json

# 特定バージョンの取得
aws s3api get-object \
  --bucket my-app-bucket-2024 \
  --key config/settings.json \
  --version-id "abc123def456" \
  ./settings-old.json

# 特定バージョンの削除
aws s3api delete-object \
  --bucket my-app-bucket-2024 \
  --key config/settings.json \
  --version-id "abc123def456"

# 削除マーカーの削除（オブジェクトの復元）
aws s3api delete-object \
  --bucket my-app-bucket-2024 \
  --key config/settings.json \
  --version-id "DELETE_MARKER_VERSION_ID"
```

```
バージョニングの動作

  バージョニング有効時の PUT:
  settings.json v1 (最初のアップロード)
  settings.json v2 (上書きアップロード → v1 は保持)
  settings.json v3 (上書きアップロード → v1, v2 は保持)

  バージョニング有効時の DELETE:
  settings.json に削除マーカーが付与
  → GET すると 404 が返る
  → 削除マーカーを削除すると v3 が最新に戻る
  → 全バージョンは保持されたまま

  注意:
  ├── バージョニングは一度有効にすると無効化できない
  │   （Suspended にはできるが、既存バージョンは残る）
  ├── 全バージョンがストレージ料金の対象
  └── ライフサイクルルールで古いバージョンを自動削除推奨
```

---

## 12. Presigned URL（署名付き URL）

### 12.1 Presigned URL の用途と仕組み

```
Presigned URL のフロー

  ■ ダウンロード用
  クライアント → API サーバー → S3 (Presigned URL 生成)
       ↑                              |
       +--- Presigned URL を返す ------+
       |
       +--- URL で直接 S3 からダウンロード --------→ S3

  ■ アップロード用
  クライアント → API サーバー → S3 (Presigned URL 生成)
       ↑                              |
       +--- Presigned URL を返す ------+
       |
       +--- URL で直接 S3 にアップロード ---------→ S3

  メリット:
  ├── クライアントに AWS 認証情報を渡す必要がない
  ├── サーバーを経由せず直接 S3 にアクセスできる
  ├── 有効期限を設定できる（最大7日間）
  └── 特定のオブジェクトのみアクセスを許可できる
```

### 12.2 Presigned URL の実装

```python
import boto3
from botocore.config import Config

# Presigned URL 用のクライアント（署名バージョン指定）
s3 = boto3.client(
    's3',
    region_name='ap-northeast-1',
    config=Config(signature_version='s3v4')
)

def generate_download_url(bucket: str, key: str,
                          expires_in: int = 3600,
                          filename: str = None) -> str:
    """ダウンロード用 Presigned URL を生成する"""
    params = {
        'Bucket': bucket,
        'Key': key,
    }
    # ダウンロード時のファイル名を指定
    if filename:
        params['ResponseContentDisposition'] = f'attachment; filename="{filename}"'

    url = s3.generate_presigned_url(
        'get_object',
        Params=params,
        ExpiresIn=expires_in
    )
    return url

def generate_upload_url(bucket: str, key: str,
                        content_type: str = 'application/octet-stream',
                        max_size: int = None,
                        expires_in: int = 3600) -> dict:
    """アップロード用 Presigned URL を生成する"""
    params = {
        'Bucket': bucket,
        'Key': key,
        'ContentType': content_type,
        'ServerSideEncryption': 'AES256',
    }

    url = s3.generate_presigned_url(
        'put_object',
        Params=params,
        ExpiresIn=expires_in
    )

    return {
        'url': url,
        'fields': {
            'Content-Type': content_type,
            'x-amz-server-side-encryption': 'AES256',
        }
    }

def generate_presigned_post(bucket: str, key_prefix: str,
                            content_type: str = 'image/jpeg',
                            max_size_mb: int = 10,
                            expires_in: int = 3600) -> dict:
    """POST 用の Presigned URL（フォームアップロード向け）"""
    conditions = [
        {'bucket': bucket},
        ['starts-with', '$key', key_prefix],
        {'Content-Type': content_type},
        ['content-length-range', 1, max_size_mb * 1024 * 1024],
        {'x-amz-server-side-encryption': 'AES256'},
    ]

    fields = {
        'Content-Type': content_type,
        'x-amz-server-side-encryption': 'AES256',
    }

    response = s3.generate_presigned_post(
        Bucket=bucket,
        Key=f'{key_prefix}/${{filename}}',
        Fields=fields,
        Conditions=conditions,
        ExpiresIn=expires_in
    )

    return response

# 使用例
if __name__ == '__main__':
    BUCKET = 'my-app-bucket-2024'

    # ダウンロード URL
    download_url = generate_download_url(
        BUCKET, 'reports/monthly.pdf',
        filename='月次レポート.pdf'
    )
    print(f"ダウンロード URL: {download_url}")

    # アップロード URL
    upload_info = generate_upload_url(
        BUCKET, 'uploads/images/photo.jpg',
        content_type='image/jpeg'
    )
    print(f"アップロード URL: {upload_info['url']}")

    # POST 用 URL
    post_info = generate_presigned_post(
        BUCKET, 'uploads/avatars',
        max_size_mb=5
    )
    print(f"POST URL: {post_info['url']}")
    print(f"POST Fields: {post_info['fields']}")
```

```javascript
// フロントエンドからの Presigned URL 利用例

// ダウンロード
async function downloadFile(presignedUrl, filename) {
  const response = await fetch(presignedUrl);
  const blob = await response.blob();

  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
}

// PUT によるアップロード
async function uploadFile(presignedUrl, file, contentType) {
  const response = await fetch(presignedUrl, {
    method: 'PUT',
    headers: {
      'Content-Type': contentType,
      'x-amz-server-side-encryption': 'AES256',
    },
    body: file,
  });

  if (!response.ok) {
    throw new Error(`アップロード失敗: ${response.status}`);
  }
  console.log('アップロード完了');
}

// POST によるアップロード（フォームデータ）
async function uploadWithPost(presignedPost, file) {
  const formData = new FormData();

  // Presigned POST のフィールドを追加
  Object.entries(presignedPost.fields).forEach(([key, value]) => {
    formData.append(key, value);
  });

  // ファイルは最後に追加（重要）
  formData.append('file', file);

  const response = await fetch(presignedPost.url, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`アップロード失敗: ${response.status}`);
  }
  console.log('アップロード完了');
}
```

---

## 13. アンチパターン

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

```bash
# 未完了のマルチパートアップロードの確認
aws s3api list-multipart-uploads --bucket my-app-bucket-2024

# ライフサイクルルールで7日後に自動削除
aws s3api put-bucket-lifecycle-configuration \
  --bucket my-app-bucket-2024 \
  --lifecycle-configuration '{
    "Rules": [{
      "ID": "AbortIncompleteMultipartUploads",
      "Status": "Enabled",
      "Filter": {"Prefix": ""},
      "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7}
    }]
  }'
```

### アンチパターン 3: バージョニングなしで本番データを保管する

バージョニングが無効な場合、誤った上書きや削除が復元不可能になる。本番環境のバケットでは必ずバージョニングを有効にすべきである。

### アンチパターン 4: ライフサイクルルールなしで大量データを保管する

すべてのデータを Standard ストレージクラスに保持し続けると、不要なコストが発生する。アクセスパターンに応じたライフサイクルルールを設定すべきである。

### アンチパターン 5: バケットポリシーとIAMポリシーの二重管理

アクセス制御をバケットポリシーと IAM ポリシーの両方で管理すると、意図しないアクセス許可や拒否が発生しやすくなる。原則として IAM ポリシーで管理し、クロスアカウントアクセスなど IAM だけでは実現できないケースでのみバケットポリシーを使用すべきである。

### アンチパターン 6: 暗号化設定の漏れ

デフォルト暗号化を設定せず、個別のアップロード時に暗号化を指定する運用はヒューマンエラーのリスクが高い。バケットレベルでデフォルト暗号化を有効にし、暗号化されていないアップロードを拒否するバケットポリシーも併用すべきである。

---

## 14. FAQ

### Q1. S3 のバケット名に制約はあるか？

グローバルに一意である必要があり、3-63文字、小文字・数字・ハイフンのみ使用可能。ピリオドは SSL 証明書の問題を起こすため避けるべきである。`my-company-app-prod` のような命名規則が推奨される。

### Q2. 5GB 以上のファイルをアップロードするには？

マルチパートアップロードを使用する。AWS CLI の `aws s3 cp` は自動的にマルチパートアップロードを行う（閾値はデフォルト 8MB）。SDK でも `upload` メソッドが自動分割する。最大 5TB まで対応。

### Q3. S3 のコストを削減するには？

(1) Intelligent-Tiering でアクセスパターンに応じた自動階層化、(2) ライフサイクルルールで古いデータを Glacier に移行、(3) 不完全なマルチパートアップロードの削除、(4) S3 Storage Lens でコスト分析を実施する。

### Q4. S3 Select と Athena の違いは？

S3 Select は単一オブジェクト内の CSV/JSON/Parquet データから特定のカラムや行をフィルタリングして取得する機能。Athena は複数オブジェクトにまたがる SQL クエリを実行するサーバーレス分析サービス。小規模な単一ファイルの検索には S3 Select、大規模なデータ分析には Athena が適している。

### Q5. S3 のデータ転送料金はどうなっているか？

インバウンド（S3 へのアップロード）は無料。アウトバウンド（S3 からのダウンロード）は最初の 100GB/月が無料、それ以降は $0.114/GB（東京リージョン）。同一リージョン内の EC2 からのアクセスは無料。CloudFront 経由のアクセスは S3 からの転送料は無料（CloudFront の転送料がかかる）。

### Q6. S3 のリクエスト料金はどれくらいか？

Standard の場合、PUT/COPY/POST/LIST リクエストは $0.0047/1,000リクエスト、GET/SELECT/HEAD リクエストは $0.00037/1,000リクエスト。大量のリクエストが発生するアプリケーションでは、CloudFront でキャッシュすることでリクエスト数とコストを削減できる。

### Q7. バケットを別のリージョンに移動できるか？

バケットのリージョンは変更できない。別リージョンにデータを移動するには、新しいバケットを作成して S3 クロスリージョンレプリケーション（CRR）を設定するか、`aws s3 sync` でコピーする。

### Q8. S3 オブジェクトロックとは何か？

WORM（Write Once Read Many）モデルでオブジェクトの削除や上書きを防止する機能。コンプライアンス要件（SEC Rule 17a-4、FINRA など）で必要になることが多い。Governance モード（特権ユーザーは解除可能）と Compliance モード（誰も解除不可）がある。

---

## 15. まとめ

| 項目 | ポイント |
|------|---------|
| データモデル | バケット（名前空間）+ オブジェクト（キー + 値） |
| 耐久性 | 99.999999999% (イレブンナイン) |
| 整合性 | 強い読み取り整合性（Read-After-Write） |
| ストレージクラス | アクセス頻度に応じて Standard → IA → Glacier |
| アクセス制御 | パブリックアクセスブロック + バケットポリシー + IAM |
| 暗号化 | SSE-S3 をデフォルトで有効化 |
| ライフサイクル | 自動遷移 + 自動削除でコスト最適化 |
| バージョニング | 本番環境では必ず有効化 |
| 静的ホスティング | CloudFront + OAC が推奨構成 |
| イベント通知 | Lambda/SQS/SNS/EventBridge と連携 |
| マルチパートアップロード | 100MB 以上のファイルで推奨 |
| Presigned URL | クライアントへの一時的なアクセス権付与 |
| モニタリング | CloudWatch メトリクス + S3 Storage Lens |

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
5. S3 パフォーマンス最適化 — https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html
6. S3 暗号化ガイド — https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingEncryption.html
7. S3 ライフサイクルルール — https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html

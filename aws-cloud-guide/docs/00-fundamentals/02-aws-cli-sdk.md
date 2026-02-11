# AWS CLI / SDK

> コマンドラインと各種プログラミング言語から AWS を操作するための基盤ツールをマスターする

## この章で学ぶこと

1. AWS CLI v2 をインストール・設定し、プロファイルを使い分けて複数アカウントを操作できる
2. JavaScript (AWS SDK v3) と Python (boto3) で AWS サービスを操作するコードを書ける
3. 認証情報を安全に管理し、環境変数・IAM ロール・SSO を適切に使い分けられる

---

## 1. AWS CLI v2 のインストールと設定

### 1.1 インストール

```bash
# macOS
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /

# Linux (x86_64)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
  -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# バージョン確認
aws --version
# aws-cli/2.x.x Python/3.x.x ...
```

### 1.2 初期設定

```bash
# インタラクティブ設定
aws configure
# AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
# AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
# Default region name [None]: ap-northeast-1
# Default output format [None]: json

# 設定ファイルの確認
cat ~/.aws/credentials
cat ~/.aws/config
```

### 1.3 設定ファイルの構造

```
~/.aws/
├── credentials    # 認証情報（アクセスキー）
│   [default]
│   aws_access_key_id = AKIA...
│   aws_secret_access_key = wJal...
│
│   [dev]
│   aws_access_key_id = AKIA...
│   aws_secret_access_key = xxxx...
│
└── config         # リージョン、出力形式、ロール設定
    [default]
    region = ap-northeast-1
    output = json

    [profile dev]
    region = ap-northeast-1
    output = yaml

    [profile prod]
    role_arn = arn:aws:iam::111111111111:role/Admin
    source_profile = default
    region = ap-northeast-1
```

---

## 2. プロファイル管理

### 2.1 名前付きプロファイル

```bash
# 名前付きプロファイルを作成
aws configure --profile dev
aws configure --profile staging
aws configure --profile prod

# プロファイルを指定してコマンド実行
aws s3 ls --profile dev
aws ec2 describe-instances --profile prod

# 環境変数でデフォルトプロファイルを切り替え
export AWS_PROFILE=dev
aws s3 ls  # dev プロファイルで実行される
```

### 2.2 認証情報の解決順序

```
AWS CLI / SDK の認証情報解決順序（優先度順）

  +-----------------------------------+
  | 1. コマンドラインオプション         |  --profile, --region
  +-----------------------------------+
              ↓ (未設定なら)
  +-----------------------------------+
  | 2. 環境変数                        |  AWS_ACCESS_KEY_ID
  |                                   |  AWS_SECRET_ACCESS_KEY
  |                                   |  AWS_SESSION_TOKEN
  +-----------------------------------+
              ↓ (未設定なら)
  +-----------------------------------+
  | 3. 共有認証情報ファイル             |  ~/.aws/credentials
  +-----------------------------------+
              ↓ (未設定なら)
  +-----------------------------------+
  | 4. 共有設定ファイル                 |  ~/.aws/config
  +-----------------------------------+
              ↓ (未設定なら)
  +-----------------------------------+
  | 5. ECS コンテナ認証情報             |  タスクロール
  +-----------------------------------+
              ↓ (未設定なら)
  +-----------------------------------+
  | 6. EC2 インスタンスメタデータ       |  インスタンスプロファイル
  +-----------------------------------+
```

### 2.3 AssumeRole によるクロスアカウントアクセス

```bash
# ~/.aws/config でロールを設定
# [profile prod]
# role_arn = arn:aws:iam::111111111111:role/AdminRole
# source_profile = default
# mfa_serial = arn:aws:iam::999999999999:mfa/my-user

# MFA 付きでロールを引き受ける
aws sts assume-role \
  --role-arn arn:aws:iam::111111111111:role/AdminRole \
  --role-session-name my-session \
  --serial-number arn:aws:iam::999999999999:mfa/my-user \
  --token-code 123456

# 上記の結果を環境変数に設定
export AWS_ACCESS_KEY_ID=ASIAXXXXXXXX
export AWS_SECRET_ACCESS_KEY=XXXXXXXX
export AWS_SESSION_TOKEN=XXXXXXXX
```

---

## 3. AWS CLI 実践テクニック

### 3.1 出力フォーマットと --query

```bash
# JSON 出力（デフォルト）
aws ec2 describe-instances --output json

# テーブル形式（人間が読みやすい）
aws ec2 describe-instances --output table

# YAML 形式
aws ec2 describe-instances --output yaml

# --query で JMESPath フィルタ
aws ec2 describe-instances \
  --query 'Reservations[].Instances[].[InstanceId,State.Name,InstanceType]' \
  --output table

# 特定タグのインスタンスだけ抽出
aws ec2 describe-instances \
  --filters "Name=tag:Environment,Values=production" \
  --query 'Reservations[].Instances[].{
    ID: InstanceId,
    Type: InstanceType,
    State: State.Name,
    IP: PublicIpAddress
  }' \
  --output table
```

### 3.2 便利なワンライナー集

```bash
# 全リージョンの EC2 インスタンス一覧
for region in $(aws ec2 describe-regions --query 'Regions[].RegionName' --output text); do
  echo "=== $region ==="
  aws ec2 describe-instances --region $region \
    --query 'Reservations[].Instances[].[InstanceId,State.Name]' \
    --output table
done

# S3 バケットサイズの確認
aws cloudwatch get-metric-statistics \
  --namespace AWS/S3 \
  --metric-name BucketSizeBytes \
  --dimensions Name=BucketName,Value=my-bucket Name=StorageType,Value=StandardStorage \
  --start-time $(date -u -v-1d +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 86400 \
  --statistics Average
```

---

## 4. AWS SDK for JavaScript (v3)

### 4.1 セットアップ

```bash
# パッケージインストール（必要なサービスだけ）
npm install @aws-sdk/client-s3
npm install @aws-sdk/client-dynamodb
npm install @aws-sdk/lib-dynamodb  # DocumentClient
```

### 4.2 S3 操作

```javascript
import { S3Client, PutObjectCommand, GetObjectCommand, ListObjectsV2Command } from '@aws-sdk/client-s3';

const s3 = new S3Client({ region: 'ap-northeast-1' });

// ファイルアップロード
async function uploadFile(bucket, key, body) {
  const command = new PutObjectCommand({
    Bucket: bucket,
    Key: key,
    Body: body,
    ContentType: 'application/json',
  });
  const response = await s3.send(command);
  console.log('Upload success:', response.$metadata.httpStatusCode);
}

// ファイルダウンロード
async function downloadFile(bucket, key) {
  const command = new GetObjectCommand({ Bucket: bucket, Key: key });
  const response = await s3.send(command);
  const body = await response.Body.transformToString();
  return JSON.parse(body);
}

// オブジェクト一覧
async function listObjects(bucket, prefix) {
  const command = new ListObjectsV2Command({
    Bucket: bucket,
    Prefix: prefix,
    MaxKeys: 100,
  });
  const response = await s3.send(command);
  return response.Contents.map(obj => ({
    key: obj.Key,
    size: obj.Size,
    lastModified: obj.LastModified,
  }));
}
```

### 4.3 DynamoDB 操作

```javascript
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, PutCommand, GetCommand, QueryCommand } from '@aws-sdk/lib-dynamodb';

const client = new DynamoDBClient({ region: 'ap-northeast-1' });
const docClient = DynamoDBDocumentClient.from(client);

// アイテム書き込み
async function putItem(tableName, item) {
  const command = new PutCommand({
    TableName: tableName,
    Item: item,
  });
  await docClient.send(command);
}

// アイテム取得
async function getItem(tableName, key) {
  const command = new GetCommand({
    TableName: tableName,
    Key: key,
  });
  const response = await docClient.send(command);
  return response.Item;
}

// クエリ
async function queryItems(tableName, pk) {
  const command = new QueryCommand({
    TableName: tableName,
    KeyConditionExpression: 'PK = :pk',
    ExpressionAttributeValues: { ':pk': pk },
  });
  const response = await docClient.send(command);
  return response.Items;
}

// 使用例
await putItem('Users', { PK: 'USER#001', SK: 'PROFILE', name: '田中太郎', age: 30 });
const user = await getItem('Users', { PK: 'USER#001', SK: 'PROFILE' });
```

---

## 5. AWS SDK for Python (boto3)

### 5.1 セットアップ

```bash
pip install boto3
```

### 5.2 S3 操作

```python
import boto3
import json

s3 = boto3.client('s3', region_name='ap-northeast-1')

# ファイルアップロード
def upload_file(bucket, key, file_path):
    s3.upload_file(file_path, bucket, key)
    print(f"Uploaded: s3://{bucket}/{key}")

# JSON データアップロード
def upload_json(bucket, key, data):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False),
        ContentType='application/json'
    )

# ファイルダウンロード
def download_json(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response['Body'].read().decode('utf-8')
    return json.loads(body)

# Presigned URL 生成（期限付き公開 URL）
def generate_presigned_url(bucket, key, expiration=3600):
    url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=expiration
    )
    return url
```

### 5.3 EC2 操作

```python
import boto3

ec2 = boto3.resource('ec2', region_name='ap-northeast-1')

# インスタンス一覧
def list_instances():
    instances = ec2.instances.filter(
        Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
    )
    for instance in instances:
        name = next(
            (tag['Value'] for tag in (instance.tags or []) if tag['Key'] == 'Name'),
            'N/A'
        )
        print(f"{instance.id} | {instance.instance_type} | {name} | {instance.public_ip_address}")

# インスタンス停止
def stop_instances(instance_ids):
    ec2.instances.filter(InstanceIds=instance_ids).stop()
    print(f"Stopping: {instance_ids}")
```

---

## 6. SSO (IAM Identity Center) との連携

### 6.1 SSO プロファイルの設定

```bash
# SSO 設定
aws configure sso
# SSO session name: my-sso
# SSO start URL: https://my-org.awsapps.com/start
# SSO region: ap-northeast-1
# SSO registration scopes: sso:account:access

# SSO ログイン
aws sso login --profile my-sso-profile

# ~/.aws/config に追記される設定例
# [profile my-sso-profile]
# sso_session = my-sso
# sso_account_id = 123456789012
# sso_role_name = AdministratorAccess
# region = ap-northeast-1
#
# [sso-session my-sso]
# sso_start_url = https://my-org.awsapps.com/start
# sso_region = ap-northeast-1
# sso_registration_scopes = sso:account:access
```

---

## 7. 認証情報管理のベストプラクティス

### 7.1 環境別の推奨方式

| 環境 | 推奨方式 | 理由 |
|------|---------|------|
| ローカル開発 | IAM Identity Center (SSO) | 一時認証情報、MFA 統合 |
| CI/CD | OIDC (GitHub Actions 等) | アクセスキー不要 |
| EC2 上 | インスタンスプロファイル | 自動ローテーション |
| ECS 上 | タスクロール | コンテナ単位の権限分離 |
| Lambda | 実行ロール | 自動付与 |

### 7.2 やってはいけない認証情報管理

```
+---------------------------------------------+
|  絶対にやってはいけないこと                     |
+---------------------------------------------+
| x ソースコードにアクセスキーをハードコード       |
| x .env ファイルを Git にコミット                |
| x アクセスキーを Slack/メールで共有             |
| x 全員が同じアクセスキーを共有                  |
| x アクセスキーをローテーションしない             |
+---------------------------------------------+
|  代わりにやるべきこと                           |
+---------------------------------------------+
| o IAM ロール/一時認証情報を使う                 |
| o AWS Secrets Manager でシークレット管理       |
| o .gitignore に .env, credentials を追加      |
| o git-secrets で漏洩を検出                     |
| o 90日ごとにキーをローテーション                |
+---------------------------------------------+
```

---

## 8. アンチパターン

### アンチパターン 1: アクセスキーをソースコードに直接埋め込む

```python
# 悪い例 — 絶対にやってはいけない
s3 = boto3.client('s3',
    aws_access_key_id='AKIAIOSFODNN7EXAMPLE',
    aws_secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
)

# 良い例 — 環境変数または IAM ロールを使用
s3 = boto3.client('s3', region_name='ap-northeast-1')
# 認証情報は環境変数 or ~/.aws/credentials or IAM ロールから自動解決
```

### アンチパターン 2: 全操作で AdministratorAccess を使う

開発者全員に `AdministratorAccess` を付与すると、誤ったリソース削除やセキュリティ事故のリスクが高まる。最小権限のカスタムポリシーを作成すべきである。

```bash
# 悪い例
aws iam attach-user-policy \
  --user-name developer \
  --policy-arn arn:aws:iam::aws:policy/AdministratorAccess

# 良い例 — 必要な権限だけのカスタムポリシー
aws iam attach-user-policy \
  --user-name developer \
  --policy-arn arn:aws:iam::123456789012:policy/DeveloperLimitedAccess
```

---

## 9. FAQ

### Q1. AWS CLI v1 と v2 の違いは？

v2 は v1 の後継で、SSO 統合、自動ページネーション、AWS CloudShell 対応などが追加されている。新規プロジェクトでは v2 を使用すべき。v1 は 2024年以降メンテナンスモードに移行。

### Q2. SDK v2 と v3 (JavaScript) の違いは？

v3 はモジュラーアーキテクチャを採用し、必要なサービスのみインポートできる。バンドルサイズの削減、Tree-shaking 対応、ミドルウェアスタックのカスタマイズが利点。新規プロジェクトでは v3 を使用する。

### Q3. 認証情報が漏洩した場合の対処法は？

(1) 該当アクセスキーを即座に無効化・削除、(2) CloudTrail で不正アクティビティを確認、(3) 影響を受けたリソースを特定・修復、(4) 新しいキーを生成（可能なら IAM ロールに移行）、(5) git-secrets や GuardDuty を導入して再発防止。

---

## 10. まとめ

| 項目 | ポイント |
|------|---------|
| CLI インストール | v2 を使用、`aws configure` で初期設定 |
| プロファイル | 環境ごとに名前付きプロファイルを分離 |
| 認証情報解決 | コマンドライン → 環境変数 → ファイル → ロールの順 |
| SDK (JavaScript) | v3 のモジュラーインポートを使用 |
| SDK (Python) | boto3 は自動で認証情報を解決 |
| セキュリティ | IAM ロール推奨、アクセスキーのハードコード厳禁 |
| SSO | マルチアカウント運用では IAM Identity Center 推奨 |

---

## 次に読むべきガイド

- [../01-compute/00-ec2-basics.md](../01-compute/00-ec2-basics.md) — EC2 インスタンスの基礎
- [../02-storage/00-s3-basics.md](../02-storage/00-s3-basics.md) — S3 の基礎

---

## 参考文献

1. AWS CLI v2 ユーザーガイド — https://docs.aws.amazon.com/cli/latest/userguide/
2. AWS SDK for JavaScript v3 Developer Guide — https://docs.aws.amazon.com/sdk-for-javascript/v3/developer-guide/
3. Boto3 Documentation — https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
4. AWS Security Credentials Best Practices — https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html

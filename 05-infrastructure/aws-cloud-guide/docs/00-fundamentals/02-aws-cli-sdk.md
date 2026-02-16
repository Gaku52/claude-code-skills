# AWS CLI / SDK

> コマンドラインと各種プログラミング言語から AWS を操作するための基盤ツールをマスターする

## この章で学ぶこと

1. AWS CLI v2 をインストール・設定し、プロファイルを使い分けて複数アカウントを操作できる
2. JavaScript (AWS SDK v3) と Python (boto3) で AWS サービスを操作するコードを書ける
3. 認証情報を安全に管理し、環境変数・IAM ロール・SSO を適切に使い分けられる
4. AWS CLI の高度なテクニック（JMESPath、ページネーション、ウェイター）を活用できる
5. CI/CD 環境での認証情報管理（OIDC、Secrets Manager）を実装できる

---

## 1. AWS CLI v2 のインストールと設定

### 1.1 インストール

```bash
# macOS
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /

# macOS (Homebrew)
brew install awscli

# Linux (x86_64)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
  -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Linux (ARM64 / Graviton)
curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" \
  -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Docker
docker run --rm -it amazon/aws-cli --version
# エイリアス設定
alias aws='docker run --rm -it -v ~/.aws:/root/.aws -v $(pwd):/aws amazon/aws-cli'

# バージョン確認
aws --version
# aws-cli/2.x.x Python/3.x.x ...

# アップデート
sudo ./aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update
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

# 設定値の個別確認
aws configure get region
aws configure get profile.dev.region
aws configure get default.output
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

### 1.4 環境変数による設定

```bash
# 認証情報の環境変数
export AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"
export AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
export AWS_SESSION_TOKEN="FwoGZXIvYXdzEBYaD..."  # 一時認証情報の場合

# プロファイルの切り替え
export AWS_PROFILE=dev

# リージョンのオーバーライド
export AWS_DEFAULT_REGION=us-east-1

# デフォルト出力形式
export AWS_DEFAULT_OUTPUT=json

# エンドポイント URL（LocalStack 等で使用）
export AWS_ENDPOINT_URL=http://localhost:4566

# 設定の確認
aws configure list
# 出力例:
#       Name                    Value             Type    Location
#       ----                    -----             ----    --------
#    profile                <not set>             None    None
# access_key     ****************MPLE shared-credentials-file
# secret_key     ****************EKEY shared-credentials-file
#     region           ap-northeast-1      config-file    ~/.aws/config
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

# プロファイル一覧の確認
aws configure list-profiles
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
  | 3. Web Identity Token              |  AWS_WEB_IDENTITY_TOKEN_FILE
  |                                   |  (EKS, GitHub Actions)
  +-----------------------------------+
              ↓ (未設定なら)
  +-----------------------------------+
  | 4. 共有認証情報ファイル             |  ~/.aws/credentials
  +-----------------------------------+
              ↓ (未設定なら)
  +-----------------------------------+
  | 5. 共有設定ファイル                 |  ~/.aws/config
  +-----------------------------------+
              ↓ (未設定なら)
  +-----------------------------------+
  | 6. ECS コンテナ認証情報             |  タスクロール
  +-----------------------------------+
              ↓ (未設定なら)
  +-----------------------------------+
  | 7. EC2 インスタンスメタデータ       |  インスタンスプロファイル
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

### 2.4 プロファイル切り替えスクリプト

```bash
#!/bin/bash
# aws-switch-profile.sh
# 使い方: source aws-switch-profile.sh

echo "利用可能なプロファイル:"
aws configure list-profiles | nl

read -p "プロファイル番号を選択: " num
PROFILE=$(aws configure list-profiles | sed -n "${num}p")

if [ -z "$PROFILE" ]; then
  echo "無効な番号です"
  return 1
fi

export AWS_PROFILE="$PROFILE"
echo "プロファイルを '$PROFILE' に切り替えました"

# 現在の ID を確認
aws sts get-caller-identity --output table
```

### 2.5 MFA 付き一時認証情報の取得スクリプト

```bash
#!/bin/bash
# aws-mfa.sh - MFA 認証して一時認証情報を取得
# 使い方: eval $(./aws-mfa.sh 123456)

MFA_CODE=$1
MFA_SERIAL="arn:aws:iam::123456789012:mfa/my-user"
DURATION=43200  # 12時間

if [ -z "$MFA_CODE" ]; then
  echo "Usage: eval \$(./aws-mfa.sh <MFA_CODE>)" >&2
  exit 1
fi

# 一時認証情報を取得
CREDS=$(aws sts get-session-token \
  --serial-number "$MFA_SERIAL" \
  --token-code "$MFA_CODE" \
  --duration-seconds "$DURATION" \
  --output json)

# 環境変数として出力
echo "export AWS_ACCESS_KEY_ID=$(echo $CREDS | jq -r '.Credentials.AccessKeyId')"
echo "export AWS_SECRET_ACCESS_KEY=$(echo $CREDS | jq -r '.Credentials.SecretAccessKey')"
echo "export AWS_SESSION_TOKEN=$(echo $CREDS | jq -r '.Credentials.SessionToken')"

EXPIRY=$(echo $CREDS | jq -r '.Credentials.Expiration')
echo "# 有効期限: $EXPIRY" >&2
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

# テキスト形式（スクリプト向け）
aws ec2 describe-instances --output text

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

### 3.2 JMESPath 詳細ガイド

```bash
# 基本的なフィルタリング
# 配列から特定フィールドを抽出
aws ec2 describe-instances \
  --query 'Reservations[].Instances[].InstanceId'

# オブジェクトの構築
aws ec2 describe-instances \
  --query 'Reservations[].Instances[].{
    ID: InstanceId,
    Type: InstanceType,
    AZ: Placement.AvailabilityZone,
    State: State.Name,
    LaunchTime: LaunchTime
  }' --output table

# 条件付きフィルタリング（running のインスタンスのみ）
aws ec2 describe-instances \
  --query 'Reservations[].Instances[?State.Name==`running`].{
    ID: InstanceId,
    Type: InstanceType
  }' --output table

# ソート
aws ec2 describe-instances \
  --query 'sort_by(Reservations[].Instances[], &LaunchTime)[].{
    ID: InstanceId,
    LaunchTime: LaunchTime
  }' --output table

# 最初の N 件を取得
aws ec2 describe-instances \
  --query 'Reservations[].Instances[][:5].InstanceId'

# パイプ演算子
aws ec2 describe-instances \
  --query 'Reservations[].Instances[] | length(@)'

# ネストされた配列のフラット化
aws ec2 describe-security-groups \
  --query 'SecurityGroups[].{
    GroupName: GroupName,
    InboundRules: IpPermissions[].{
      Protocol: IpProtocol,
      Port: ToPort,
      Source: IpRanges[].CidrIp | join(`, `, @)
    }
  }' --output yaml

# タグからの値取得
aws ec2 describe-instances \
  --query 'Reservations[].Instances[].{
    ID: InstanceId,
    Name: Tags[?Key==`Name`].Value | [0]
  }' --output table
```

### 3.3 便利なワンライナー集

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

# 停止中のインスタンスを一括起動
aws ec2 describe-instances \
  --filters "Name=instance-state-name,Values=stopped" "Name=tag:Environment,Values=dev" \
  --query 'Reservations[].Instances[].InstanceId' \
  --output text | xargs -n 1 aws ec2 start-instances --instance-ids

# 未アタッチの EBS ボリュームを検出
aws ec2 describe-volumes \
  --filters "Name=status,Values=available" \
  --query 'Volumes[].{
    ID: VolumeId,
    Size: Size,
    AZ: AvailabilityZone,
    Created: CreateTime
  }' --output table

# セキュリティグループで 0.0.0.0/0 に SSH を公開しているものを検出
aws ec2 describe-security-groups \
  --filters "Name=ip-permission.from-port,Values=22" \
    "Name=ip-permission.cidr,Values=0.0.0.0/0" \
  --query 'SecurityGroups[].{
    GroupId: GroupId,
    GroupName: GroupName,
    VpcId: VpcId
  }' --output table

# Lambda 関数のメモリとタイムアウト一覧
aws lambda list-functions \
  --query 'Functions[].{
    Name: FunctionName,
    Runtime: Runtime,
    Memory: MemorySize,
    Timeout: Timeout,
    LastModified: LastModified
  }' --output table

# IAM ユーザーのアクセスキー最終使用日を確認
for user in $(aws iam list-users --query 'Users[].UserName' --output text); do
  echo "--- $user ---"
  aws iam list-access-keys --user-name "$user" --query 'AccessKeyMetadata[].{
    KeyId: AccessKeyId,
    Status: Status,
    Created: CreateDate
  }' --output table
done
```

### 3.4 ページネーションと自動ページング

```bash
# AWS CLI v2 はデフォルトで自動ページネーション
aws s3api list-objects-v2 --bucket my-bucket
# → 1000件を超えても自動的に全件取得

# ページネーションを手動で制御
aws s3api list-objects-v2 --bucket my-bucket --max-items 100
# → NextToken が返る場合、次のページを取得
aws s3api list-objects-v2 --bucket my-bucket --starting-token "TOKEN..."

# ページネーションを無効化（パフォーマンス向上）
aws s3api list-objects-v2 --bucket my-bucket --no-paginate --max-items 100

# server-side ページサイズを指定
aws s3api list-objects-v2 --bucket my-bucket --page-size 500
```

### 3.5 ウェイター（非同期リソースの完了待ち）

```bash
# EC2 インスタンスの起動完了を待つ
aws ec2 run-instances --image-id ami-xxx --instance-type t3.micro \
  --query 'Instances[0].InstanceId' --output text
# → i-0123456789abcdef0

aws ec2 wait instance-running --instance-ids i-0123456789abcdef0
echo "インスタンスが running になりました"

# EBS ボリュームの利用可能を待つ
aws ec2 wait volume-available --volume-ids vol-xxx

# RDS インスタンスの起動完了を待つ
aws rds wait db-instance-available --db-instance-identifier my-db

# スナップショットの完了を待つ
aws ec2 wait snapshot-completed --snapshot-ids snap-xxx

# CloudFormation スタックの作成完了を待つ
aws cloudformation wait stack-create-complete --stack-name my-stack

# カスタムタイムアウト設定
aws ec2 wait instance-running \
  --instance-ids i-xxx \
  --cli-read-timeout 600
```

### 3.6 S3 の高度な操作

```bash
# 高速同期（マルチパートアップロード設定）
aws configure set default.s3.max_concurrent_requests 20
aws configure set default.s3.multipart_threshold 64MB
aws configure set default.s3.multipart_chunksize 16MB

# ディレクトリの同期
aws s3 sync ./build s3://my-bucket/static \
  --delete \
  --exclude "*.tmp" \
  --include "*.html" \
  --cache-control "max-age=86400" \
  --acl private

# プレサインド URL の生成
aws s3 presign s3://my-bucket/report.pdf --expires-in 3600

# バケット間コピー（クロスリージョン）
aws s3 sync s3://source-bucket s3://dest-bucket \
  --source-region ap-northeast-1 \
  --region us-east-1

# 大容量ファイルのマルチパートアップロード
aws s3 cp large-file.tar.gz s3://my-bucket/ \
  --expected-size 10737418240 \
  --storage-class INTELLIGENT_TIERING

# S3 Select でデータをクエリ
aws s3api select-object-content \
  --bucket my-bucket \
  --key data.csv \
  --expression "SELECT s.name, s.age FROM S3Object s WHERE s.age > '30'" \
  --expression-type SQL \
  --input-serialization '{"CSV": {"FileHeaderInfo": "USE"}}' \
  --output-serialization '{"CSV": {}}' \
  output.csv
```

### 3.7 AWS CLI のカスタマイズ

```bash
# ~/.aws/config でのカスタマイズ
# [default]
# region = ap-northeast-1
# output = json
# cli_pager = less      # ページャーの設定
# cli_auto_prompt = on  # 自動補完を有効化
# retry_mode = adaptive # リトライモード

# ページャーを無効化（スクリプト向け）
export AWS_PAGER=""
# または
aws ec2 describe-instances --no-cli-pager

# エイリアスの設定 (~/.aws/cli/alias)
# [toplevel]
# whoami = sts get-caller-identity
# running-instances = ec2 describe-instances \
#   --filters "Name=instance-state-name,Values=running" \
#   --query 'Reservations[].Instances[].[InstanceId,InstanceType,Tags[?Key==`Name`].Value|[0]]' \
#   --output table
# sg-open-ssh = ec2 describe-security-groups \
#   --filters "Name=ip-permission.from-port,Values=22" "Name=ip-permission.cidr,Values=0.0.0.0/0" \
#   --query 'SecurityGroups[].{ID:GroupId,Name:GroupName}' \
#   --output table

# エイリアスの使用
aws whoami
aws running-instances
aws sg-open-ssh
```

---

## 4. AWS SDK for JavaScript (v3)

### 4.1 セットアップ

```bash
# パッケージインストール（必要なサービスだけ）
npm install @aws-sdk/client-s3
npm install @aws-sdk/client-dynamodb
npm install @aws-sdk/lib-dynamodb  # DocumentClient
npm install @aws-sdk/client-lambda
npm install @aws-sdk/client-sqs
npm install @aws-sdk/client-ses

# 共通ユーティリティ
npm install @aws-sdk/credential-providers
npm install @aws-sdk/middleware-retry
npm install @aws-sdk/s3-request-presigner
```

### 4.2 S3 操作

```javascript
import {
  S3Client,
  PutObjectCommand,
  GetObjectCommand,
  ListObjectsV2Command,
  DeleteObjectCommand,
  CopyObjectCommand,
} from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';

const s3 = new S3Client({ region: 'ap-northeast-1' });

// ファイルアップロード
async function uploadFile(bucket, key, body) {
  const command = new PutObjectCommand({
    Bucket: bucket,
    Key: key,
    Body: body,
    ContentType: 'application/json',
    ServerSideEncryption: 'AES256',
    Metadata: {
      'uploaded-by': 'my-app',
      'upload-time': new Date().toISOString(),
    },
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

// オブジェクト一覧（ページネーション対応）
async function listAllObjects(bucket, prefix) {
  const allObjects = [];
  let continuationToken = undefined;

  do {
    const command = new ListObjectsV2Command({
      Bucket: bucket,
      Prefix: prefix,
      MaxKeys: 1000,
      ContinuationToken: continuationToken,
    });
    const response = await s3.send(command);

    if (response.Contents) {
      allObjects.push(...response.Contents.map(obj => ({
        key: obj.Key,
        size: obj.Size,
        lastModified: obj.LastModified,
      })));
    }
    continuationToken = response.NextContinuationToken;
  } while (continuationToken);

  return allObjects;
}

// プレサインド URL の生成
async function generatePresignedUrl(bucket, key, expiresIn = 3600) {
  const command = new GetObjectCommand({ Bucket: bucket, Key: key });
  const url = await getSignedUrl(s3, command, { expiresIn });
  return url;
}

// ストリーミングアップロード
import { Upload } from '@aws-sdk/lib-storage';
import { createReadStream } from 'fs';

async function uploadLargeFile(bucket, key, filePath) {
  const upload = new Upload({
    client: s3,
    params: {
      Bucket: bucket,
      Key: key,
      Body: createReadStream(filePath),
    },
    queueSize: 4,         // 並列アップロード数
    partSize: 5 * 1024 * 1024,  // パートサイズ: 5MB
  });

  upload.on('httpUploadProgress', (progress) => {
    console.log(`Progress: ${progress.loaded}/${progress.total}`);
  });

  await upload.done();
}
```

### 4.3 DynamoDB 操作

```javascript
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import {
  DynamoDBDocumentClient,
  PutCommand,
  GetCommand,
  QueryCommand,
  UpdateCommand,
  DeleteCommand,
  BatchWriteCommand,
  TransactWriteCommand,
} from '@aws-sdk/lib-dynamodb';

const client = new DynamoDBClient({ region: 'ap-northeast-1' });
const docClient = DynamoDBDocumentClient.from(client, {
  marshallOptions: {
    removeUndefinedValues: true,
    convertClassInstanceToMap: true,
  },
});

// アイテム書き込み
async function putItem(tableName, item) {
  const command = new PutCommand({
    TableName: tableName,
    Item: item,
    ConditionExpression: 'attribute_not_exists(PK)',  // 重複防止
  });
  await docClient.send(command);
}

// アイテム取得
async function getItem(tableName, key) {
  const command = new GetCommand({
    TableName: tableName,
    Key: key,
    ConsistentRead: true,  // 強い整合性
  });
  const response = await docClient.send(command);
  return response.Item;
}

// クエリ（ページネーション対応）
async function queryAllItems(tableName, pk, skPrefix) {
  const allItems = [];
  let lastKey = undefined;

  do {
    const command = new QueryCommand({
      TableName: tableName,
      KeyConditionExpression: 'PK = :pk AND begins_with(SK, :skPrefix)',
      ExpressionAttributeValues: {
        ':pk': pk,
        ':skPrefix': skPrefix,
      },
      ExclusiveStartKey: lastKey,
      Limit: 100,
    });
    const response = await docClient.send(command);
    allItems.push(...response.Items);
    lastKey = response.LastEvaluatedKey;
  } while (lastKey);

  return allItems;
}

// 条件付き更新
async function updateItem(tableName, key, updates) {
  const command = new UpdateCommand({
    TableName: tableName,
    Key: key,
    UpdateExpression: 'SET #name = :name, #age = :age, updatedAt = :now',
    ExpressionAttributeNames: {
      '#name': 'name',
      '#age': 'age',
    },
    ExpressionAttributeValues: {
      ':name': updates.name,
      ':age': updates.age,
      ':now': new Date().toISOString(),
    },
    ReturnValues: 'ALL_NEW',
  });
  const response = await docClient.send(command);
  return response.Attributes;
}

// バッチ書き込み（25件ずつ）
async function batchWriteItems(tableName, items) {
  const BATCH_SIZE = 25;
  for (let i = 0; i < items.length; i += BATCH_SIZE) {
    const batch = items.slice(i, i + BATCH_SIZE);
    const command = new BatchWriteCommand({
      RequestItems: {
        [tableName]: batch.map(item => ({
          PutRequest: { Item: item },
        })),
      },
    });
    await docClient.send(command);
  }
}

// トランザクション
async function transferPoints(fromUser, toUser, points) {
  const command = new TransactWriteCommand({
    TransactItems: [
      {
        Update: {
          TableName: 'Users',
          Key: { PK: fromUser, SK: 'PROFILE' },
          UpdateExpression: 'SET points = points - :points',
          ConditionExpression: 'points >= :points',
          ExpressionAttributeValues: { ':points': points },
        },
      },
      {
        Update: {
          TableName: 'Users',
          Key: { PK: toUser, SK: 'PROFILE' },
          UpdateExpression: 'SET points = points + :points',
          ExpressionAttributeValues: { ':points': points },
        },
      },
    ],
  });
  await docClient.send(command);
}

// 使用例
await putItem('Users', {
  PK: 'USER#001', SK: 'PROFILE',
  name: '田中太郎', age: 30, points: 1000,
});
const user = await getItem('Users', { PK: 'USER#001', SK: 'PROFILE' });
```

### 4.4 Lambda 呼び出し

```javascript
import { LambdaClient, InvokeCommand } from '@aws-sdk/client-lambda';

const lambda = new LambdaClient({ region: 'ap-northeast-1' });

// 同期呼び出し
async function invokeLambdaSync(functionName, payload) {
  const command = new InvokeCommand({
    FunctionName: functionName,
    InvocationType: 'RequestResponse',
    Payload: JSON.stringify(payload),
  });
  const response = await lambda.send(command);
  const result = JSON.parse(new TextDecoder().decode(response.Payload));
  return result;
}

// 非同期呼び出し
async function invokeLambdaAsync(functionName, payload) {
  const command = new InvokeCommand({
    FunctionName: functionName,
    InvocationType: 'Event',
    Payload: JSON.stringify(payload),
  });
  await lambda.send(command);
}
```

### 4.5 SQS 操作

```javascript
import {
  SQSClient,
  SendMessageCommand,
  ReceiveMessageCommand,
  DeleteMessageCommand,
} from '@aws-sdk/client-sqs';

const sqs = new SQSClient({ region: 'ap-northeast-1' });
const QUEUE_URL = 'https://sqs.ap-northeast-1.amazonaws.com/123456789012/my-queue';

// メッセージ送信
async function sendMessage(body, groupId) {
  const command = new SendMessageCommand({
    QueueUrl: QUEUE_URL,
    MessageBody: JSON.stringify(body),
    MessageGroupId: groupId,
    MessageDeduplicationId: `${Date.now()}-${Math.random()}`,
  });
  await sqs.send(command);
}

// メッセージ受信と処理
async function processMessages() {
  const command = new ReceiveMessageCommand({
    QueueUrl: QUEUE_URL,
    MaxNumberOfMessages: 10,
    WaitTimeSeconds: 20,  // ロングポーリング
    VisibilityTimeout: 60,
  });
  const response = await sqs.send(command);

  for (const message of response.Messages || []) {
    try {
      const body = JSON.parse(message.Body);
      await handleMessage(body);

      // 正常処理後にメッセージを削除
      await sqs.send(new DeleteMessageCommand({
        QueueUrl: QUEUE_URL,
        ReceiptHandle: message.ReceiptHandle,
      }));
    } catch (error) {
      console.error('Message processing failed:', error);
      // メッセージは削除せず、VisibilityTimeout 後に再処理される
    }
  }
}
```

### 4.6 エラーハンドリングとリトライ

```javascript
import { S3Client, GetObjectCommand } from '@aws-sdk/client-s3';
import { NodeHttpHandler } from '@smithy/node-http-handler';

// リトライ設定付きクライアント
const s3 = new S3Client({
  region: 'ap-northeast-1',
  maxAttempts: 5,
  retryMode: 'adaptive',
  requestHandler: new NodeHttpHandler({
    connectionTimeout: 5000,
    socketTimeout: 30000,
  }),
});

// エラーハンドリング
async function getObjectSafely(bucket, key) {
  try {
    const command = new GetObjectCommand({ Bucket: bucket, Key: key });
    const response = await s3.send(command);
    return await response.Body.transformToString();
  } catch (error) {
    switch (error.name) {
      case 'NoSuchKey':
        console.log(`Object not found: ${key}`);
        return null;
      case 'NoSuchBucket':
        throw new Error(`Bucket does not exist: ${bucket}`);
      case 'AccessDenied':
        throw new Error(`Access denied to ${bucket}/${key}`);
      case 'ThrottlingException':
      case 'TooManyRequestsException':
        console.log('Rate limited, retrying...');
        await new Promise(resolve => setTimeout(resolve, 2000));
        return getObjectSafely(bucket, key);
      default:
        throw error;
    }
  }
}
```

---

## 5. AWS SDK for Python (boto3)

### 5.1 セットアップ

```bash
pip install boto3
pip install boto3-stubs[essential]  # 型ヒント（開発時）
```

### 5.2 S3 操作

```python
import boto3
import json
from botocore.config import Config

# リトライ設定付きクライアント
config = Config(
    region_name='ap-northeast-1',
    retries={'max_attempts': 5, 'mode': 'adaptive'},
    max_pool_connections=50,
)

s3 = boto3.client('s3', config=config)

# ファイルアップロード
def upload_file(bucket, key, file_path):
    s3.upload_file(
        file_path, bucket, key,
        ExtraArgs={
            'ServerSideEncryption': 'AES256',
            'Metadata': {'uploaded-by': 'my-app'},
        },
        Callback=lambda bytes_transferred: print(f'Transferred: {bytes_transferred} bytes'),
    )
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

# ページネーション対応のオブジェクト一覧
def list_all_objects(bucket, prefix=''):
    paginator = s3.get_paginator('list_objects_v2')
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            objects.append({
                'key': obj['Key'],
                'size': obj['Size'],
                'last_modified': obj['LastModified'],
            })
    return objects

# バケット間のコピー
def copy_between_buckets(src_bucket, src_key, dest_bucket, dest_key):
    s3.copy_object(
        CopySource={'Bucket': src_bucket, 'Key': src_key},
        Bucket=dest_bucket,
        Key=dest_key,
        ServerSideEncryption='AES256',
    )
```

### 5.3 EC2 操作

```python
import boto3
from datetime import datetime, timedelta

ec2 = boto3.resource('ec2', region_name='ap-northeast-1')
ec2_client = boto3.client('ec2', region_name='ap-northeast-1')

# インスタンス一覧
def list_instances(state='running'):
    instances = ec2.instances.filter(
        Filters=[{'Name': 'instance-state-name', 'Values': [state]}]
    )
    for instance in instances:
        name = next(
            (tag['Value'] for tag in (instance.tags or []) if tag['Key'] == 'Name'),
            'N/A'
        )
        print(f"{instance.id} | {instance.instance_type} | "
              f"{name} | {instance.public_ip_address}")

# インスタンス停止
def stop_instances(instance_ids):
    ec2.instances.filter(InstanceIds=instance_ids).stop()
    print(f"Stopping: {instance_ids}")

# タグでインスタンスを操作
def stop_dev_instances():
    """開発環境のインスタンスを夜間停止"""
    instances = ec2.instances.filter(
        Filters=[
            {'Name': 'instance-state-name', 'Values': ['running']},
            {'Name': 'tag:Environment', 'Values': ['development']},
        ]
    )
    ids = [i.id for i in instances]
    if ids:
        ec2.instances.filter(InstanceIds=ids).stop()
        print(f"Stopped {len(ids)} dev instances: {ids}")

# 古いスナップショットの削除
def cleanup_old_snapshots(days=30):
    cutoff = datetime.now(tz=datetime.now().astimezone().tzinfo) - timedelta(days=days)
    snapshots = ec2_client.describe_snapshots(OwnerIds=['self'])['Snapshots']
    for snap in snapshots:
        if snap['StartTime'] < cutoff:
            ec2_client.delete_snapshot(SnapshotId=snap['SnapshotId'])
            print(f"Deleted: {snap['SnapshotId']} ({snap['StartTime']})")

# ウェイターの使用
def launch_and_wait(ami_id, instance_type, key_name, sg_ids, subnet_id):
    instances = ec2.create_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        KeyName=key_name,
        SecurityGroupIds=sg_ids,
        SubnetId=subnet_id,
        MinCount=1, MaxCount=1,
        TagSpecifications=[{
            'ResourceType': 'instance',
            'Tags': [{'Key': 'Name', 'Value': 'my-server'}],
        }],
    )
    instance = instances[0]
    print(f"Launching: {instance.id}")

    # running 状態まで待機
    instance.wait_until_running()
    instance.reload()
    print(f"Running: {instance.public_ip_address}")
    return instance
```

### 5.4 DynamoDB 操作

```python
import boto3
from boto3.dynamodb.conditions import Key, Attr
from decimal import Decimal

dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-1')
table = dynamodb.Table('Users')

# アイテム書き込み
def put_item(pk, sk, data):
    item = {'PK': pk, 'SK': sk, **data}
    table.put_item(Item=item)

# アイテム取得
def get_item(pk, sk):
    response = table.get_item(Key={'PK': pk, 'SK': sk})
    return response.get('Item')

# クエリ
def query_items(pk, sk_prefix=None):
    if sk_prefix:
        response = table.query(
            KeyConditionExpression=Key('PK').eq(pk) & Key('SK').begins_with(sk_prefix)
        )
    else:
        response = table.query(
            KeyConditionExpression=Key('PK').eq(pk)
        )
    return response['Items']

# バッチ書き込み
def batch_write(items):
    with table.batch_writer() as batch:
        for item in items:
            batch.put_item(Item=item)

# トランザクション
def transfer_points(from_user, to_user, points):
    client = boto3.client('dynamodb', region_name='ap-northeast-1')
    client.transact_write_items(
        TransactItems=[
            {
                'Update': {
                    'TableName': 'Users',
                    'Key': {'PK': {'S': from_user}, 'SK': {'S': 'PROFILE'}},
                    'UpdateExpression': 'SET points = points - :pts',
                    'ConditionExpression': 'points >= :pts',
                    'ExpressionAttributeValues': {':pts': {'N': str(points)}},
                }
            },
            {
                'Update': {
                    'TableName': 'Users',
                    'Key': {'PK': {'S': to_user}, 'SK': {'S': 'PROFILE'}},
                    'UpdateExpression': 'SET points = points + :pts',
                    'ExpressionAttributeValues': {':pts': {'N': str(points)}},
                }
            },
        ]
    )
```

### 5.5 セッション管理とマルチアカウント

```python
import boto3

# デフォルトセッション
default_session = boto3.Session(region_name='ap-northeast-1')

# プロファイル指定セッション
dev_session = boto3.Session(profile_name='dev')
prod_session = boto3.Session(profile_name='prod')

# AssumeRole でクロスアカウントアクセス
def get_cross_account_session(role_arn, session_name='cross-account'):
    sts = boto3.client('sts')
    response = sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName=session_name,
        DurationSeconds=3600,
    )
    credentials = response['Credentials']
    return boto3.Session(
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
    )

# 使用例
prod_session = get_cross_account_session(
    'arn:aws:iam::111111111111:role/AdminRole'
)
prod_s3 = prod_session.client('s3')
prod_s3.list_buckets()
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

### 6.2 複数アカウントの SSO 設定

```
# ~/.aws/config
[sso-session my-org]
sso_start_url = https://my-org.awsapps.com/start
sso_region = ap-northeast-1
sso_registration_scopes = sso:account:access

[profile dev]
sso_session = my-org
sso_account_id = 111111111111
sso_role_name = PowerUserAccess
region = ap-northeast-1

[profile staging]
sso_session = my-org
sso_account_id = 222222222222
sso_role_name = PowerUserAccess
region = ap-northeast-1

[profile prod]
sso_session = my-org
sso_account_id = 333333333333
sso_role_name = ReadOnlyAccess
region = ap-northeast-1

[profile prod-admin]
sso_session = my-org
sso_account_id = 333333333333
sso_role_name = AdministratorAccess
region = ap-northeast-1
```

---

## 7. CI/CD での認証情報管理

### 7.1 GitHub Actions + OIDC（推奨）

```yaml
# .github/workflows/deploy.yml
name: Deploy to AWS
on:
  push:
    branches: [main]

permissions:
  id-token: write
  contents: read

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/GitHubActionsRole
          aws-region: ap-northeast-1
          role-session-name: github-actions-${{ github.run_id }}

      - name: Deploy
        run: |
          aws s3 sync ./build s3://my-app-bucket --delete
          aws cloudfront create-invalidation \
            --distribution-id EDFDVBD6EXAMPLE \
            --paths "/*"
```

### 7.2 GitHub Actions 用 IAM ロール

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:my-org/my-repo:ref:refs/heads/main"
        }
      }
    }
  ]
}
```

### 7.3 Terraform での OIDC プロバイダー設定

```hcl
# GitHub Actions OIDC プロバイダー
resource "aws_iam_openid_connect_provider" "github_actions" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["ffffffffffffffffffffffffffffffffffffffff"]
}

# GitHub Actions 用ロール
resource "aws_iam_role" "github_actions" {
  name = "github-actions-deploy-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = aws_iam_openid_connect_provider.github_actions.arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            "token.actions.githubusercontent.com:sub" = "repo:my-org/my-repo:*"
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "github_actions_s3" {
  role       = aws_iam_role.github_actions.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}
```

---

## 8. 認証情報管理のベストプラクティス

### 8.1 環境別の推奨方式

| 環境 | 推奨方式 | 理由 |
|------|---------|------|
| ローカル開発 | IAM Identity Center (SSO) | 一時認証情報、MFA 統合 |
| CI/CD | OIDC (GitHub Actions 等) | アクセスキー不要 |
| EC2 上 | インスタンスプロファイル | 自動ローテーション |
| ECS 上 | タスクロール | コンテナ単位の権限分離 |
| Lambda | 実行ロール | 自動付与 |
| EKS 上 | IRSA (IAM Roles for Service Accounts) | Pod 単位の権限分離 |
| ローカル（SSO 不可） | aws-vault + 一時認証情報 | 暗号化ストレージ |

### 8.2 やってはいけない認証情報管理

```
+---------------------------------------------+
|  絶対にやってはいけないこと                     |
+---------------------------------------------+
| x ソースコードにアクセスキーをハードコード       |
| x .env ファイルを Git にコミット                |
| x アクセスキーを Slack/メールで共有             |
| x 全員が同じアクセスキーを共有                  |
| x アクセスキーをローテーションしない             |
| x ルートユーザーのアクセスキーを作成             |
+---------------------------------------------+
|  代わりにやるべきこと                           |
+---------------------------------------------+
| o IAM ロール/一時認証情報を使う                 |
| o AWS Secrets Manager でシークレット管理       |
| o .gitignore に .env, credentials を追加      |
| o git-secrets で漏洩を検出                     |
| o 90日ごとにキーをローテーション                |
| o CI/CD は OIDC 連携を使う                     |
+---------------------------------------------+
```

### 8.3 git-secrets のセットアップ

```bash
# インストール
brew install git-secrets  # macOS
# または
git clone https://github.com/awslabs/git-secrets.git
cd git-secrets && make install

# リポジトリに設定
cd /path/to/repo
git secrets --install
git secrets --register-aws

# グローバル設定（全リポジトリに適用）
git secrets --install ~/.git-templates/git-secrets
git config --global init.templateDir ~/.git-templates/git-secrets
git secrets --register-aws --global

# テスト
echo "AKIAIOSFODNN7EXAMPLE" > test.txt
git add test.txt
git commit -m "test"
# → ERROR: Matched one or more prohibited patterns
```

### 8.4 AWS Secrets Manager との連携

```python
import boto3
import json

def get_secret(secret_name, region='ap-northeast-1'):
    """Secrets Manager からシークレットを取得"""
    client = boto3.client('secretsmanager', region_name=region)
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# 使用例
db_creds = get_secret('prod/database')
connection = psycopg2.connect(
    host=db_creds['host'],
    port=db_creds['port'],
    dbname=db_creds['dbname'],
    user=db_creds['username'],
    password=db_creds['password'],
)
```

```javascript
import {
  SecretsManagerClient,
  GetSecretValueCommand,
} from '@aws-sdk/client-secrets-manager';

const client = new SecretsManagerClient({ region: 'ap-northeast-1' });

async function getSecret(secretName) {
  const command = new GetSecretValueCommand({ SecretId: secretName });
  const response = await client.send(command);
  return JSON.parse(response.SecretString);
}

// 使用例
const dbCreds = await getSecret('prod/database');
```

---

## 9. AWS CloudShell

### 9.1 CloudShell の概要

AWS CloudShell は、AWS マネジメントコンソールからブラウザベースのシェル環境にアクセスできるサービスである。

```
CloudShell の特徴
+----------------------------------------------------------+
|  ✓ AWS CLI v2 がプリインストール済み                       |
|  ✓ 認証情報はコンソールログインセッションから自動取得        |
|  ✓ 1GB の永続ストレージ ($HOME)                           |
|  ✓ Python, Node.js, Java, PowerShell 等がプリインストール  |
|  ✓ pip, npm 等でパッケージ追加可能                         |
|  ✓ 無料（コンソールアクセス権限があれば利用可能）            |
|                                                           |
|  制限事項:                                                 |
|  × 20分間操作がないとタイムアウト                           |
|  × 同時セッション数制限あり                                |
|  × 一部リージョンでは利用不可                              |
|  × アウトバウンド通信のみ（インバウンド不可）               |
+----------------------------------------------------------+
```

---

## 10. アンチパターン

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

### アンチパターン 3: エラーハンドリングなしで SDK を使用する

```python
# 悪い例 — エラーハンドリングなし
s3.get_object(Bucket='my-bucket', Key='data.json')

# 良い例 — 適切なエラーハンドリング
from botocore.exceptions import ClientError

try:
    response = s3.get_object(Bucket='my-bucket', Key='data.json')
except ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == 'NoSuchKey':
        print("Object not found")
    elif error_code == 'NoSuchBucket':
        print("Bucket not found")
    elif error_code == 'AccessDenied':
        print("Access denied - check IAM permissions")
    else:
        raise
```

### アンチパターン 4: ページネーションを考慮しない

```python
# 悪い例 — 最初の1000件しか取得できない
response = s3.list_objects_v2(Bucket='my-bucket')
objects = response['Contents']

# 良い例 — ページネーターで全件取得
paginator = s3.get_paginator('list_objects_v2')
objects = []
for page in paginator.paginate(Bucket='my-bucket'):
    objects.extend(page.get('Contents', []))
```

---

## 11. LocalStack でのローカル開発

### 11.1 LocalStack のセットアップ

```bash
# Docker で起動
docker run -d \
  --name localstack \
  -p 4566:4566 \
  -e SERVICES=s3,dynamodb,sqs,lambda \
  -e DEFAULT_REGION=ap-northeast-1 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  localstack/localstack

# AWS CLI のエンドポイントを LocalStack に向ける
alias awslocal='aws --endpoint-url=http://localhost:4566'

# S3 バケットを作成
awslocal s3 mb s3://my-test-bucket

# DynamoDB テーブルを作成
awslocal dynamodb create-table \
  --table-name Users \
  --attribute-definitions \
    AttributeName=PK,AttributeType=S \
    AttributeName=SK,AttributeType=S \
  --key-schema \
    AttributeName=PK,KeyType=HASH \
    AttributeName=SK,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST
```

### 11.2 Python での LocalStack 使用

```python
import boto3

# LocalStack 用のクライアント
def get_localstack_client(service):
    return boto3.client(
        service,
        endpoint_url='http://localhost:4566',
        region_name='ap-northeast-1',
        aws_access_key_id='test',
        aws_secret_access_key='test',
    )

s3 = get_localstack_client('s3')
dynamodb = get_localstack_client('dynamodb')

# テストコードで使用
def test_upload_and_download():
    s3.put_object(
        Bucket='my-test-bucket',
        Key='test.json',
        Body='{"message": "hello"}',
    )
    response = s3.get_object(Bucket='my-test-bucket', Key='test.json')
    body = response['Body'].read().decode('utf-8')
    assert '"hello"' in body
```

---

## 12. FAQ

### Q1. AWS CLI v1 と v2 の違いは？

v2 は v1 の後継で、SSO 統合、自動ページネーション、AWS CloudShell 対応、自動プロンプト (`--cli-auto-prompt`) などが追加されている。新規プロジェクトでは v2 を使用すべき。v1 は 2024年以降メンテナンスモードに移行。

### Q2. SDK v2 と v3 (JavaScript) の違いは？

v3 はモジュラーアーキテクチャを採用し、必要なサービスのみインポートできる。バンドルサイズの削減、Tree-shaking 対応、ミドルウェアスタックのカスタマイズが利点。新規プロジェクトでは v3 を使用する。

### Q3. 認証情報が漏洩した場合の対処法は？

(1) 該当アクセスキーを即座に無効化・削除、(2) CloudTrail で不正アクティビティを確認、(3) 影響を受けたリソースを特定・修復、(4) 新しいキーを生成（可能なら IAM ロールに移行）、(5) git-secrets や GuardDuty を導入して再発防止。

### Q4. boto3 の client と resource の違いは？

client は低レベルの AWS API を直接呼び出す薄いラッパー。resource は高レベルのオブジェクト指向インターフェース。resource は一部サービスのみ対応。新しいサービスは client のみ対応していることが多い。パフォーマンスが重要な場合は client を使用する。

### Q5. AWS SDK のリトライ戦略はどう設定するか？

デフォルトでは standard モードで3回リトライ。adaptive モードは API のレスポンスヘッダーに基づいてリトライ間隔を調整する。スロットリングが頻発する場合は adaptive モードと maxAttempts の増加を検討する。

---

## 13. まとめ

| 項目 | ポイント |
|------|---------|
| CLI インストール | v2 を使用、`aws configure` で初期設定 |
| プロファイル | 環境ごとに名前付きプロファイルを分離 |
| 認証情報解決 | コマンドライン → 環境変数 → ファイル → ロールの順 |
| JMESPath | --query で必要なデータだけを抽出 |
| ウェイター | 非同期リソースの完了を安全に待機 |
| SDK (JavaScript) | v3 のモジュラーインポート、エラーハンドリング必須 |
| SDK (Python) | boto3 は自動で認証情報を解決、ページネーター活用 |
| セキュリティ | IAM ロール推奨、アクセスキーのハードコード厳禁 |
| SSO | マルチアカウント運用では IAM Identity Center 推奨 |
| CI/CD | OIDC 連携でアクセスキー不要のデプロイ |
| ローカル開発 | LocalStack で AWS サービスをエミュレーション |

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
5. JMESPath Specification — https://jmespath.org/specification.html
6. LocalStack Documentation — https://docs.localstack.cloud/
7. GitHub Actions OIDC with AWS — https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services

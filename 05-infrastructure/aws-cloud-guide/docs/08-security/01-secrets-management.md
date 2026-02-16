# シークレット管理 — Secrets Manager / Parameter Store / KMS

> AWS におけるシークレット（機密情報）のライフサイクルを安全に管理し、アプリケーションからハードコードされた認証情報を排除するための実践ガイド。

---

## この章で学ぶこと

1. **AWS Secrets Manager** によるシークレットの自動ローテーションと取得パターン
2. **Systems Manager Parameter Store** との使い分けと階層型パラメータ設計
3. **AWS KMS（Key Management Service）** によるエンベロープ暗号化と鍵ポリシー設計
4. **マルチアカウント・マルチリージョン** のシークレット管理戦略
5. **シークレットの監査・監視・自動修復** のベストプラクティス

---

## 1. シークレット管理の全体像

### 1.1 なぜシークレット管理が必要か

```
┌──────────────────────────────────────────────────────┐
│              アンチパターン: ハードコード              │
│                                                      │
│   app.py                                             │
│   ┌──────────────────────────────────────┐           │
│   │ DB_PASSWORD = "P@ssw0rd123"          │ ← 危険!   │
│   │ API_KEY     = "sk-abc123..."         │           │
│   └──────────────────────────────────────┘           │
│        │                                             │
│        ▼                                             │
│   Git リポジトリに push → 漏洩リスク                  │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│              ベストプラクティス: 外部管理              │
│                                                      │
│   app.py                                             │
│   ┌──────────────────────────────────────┐           │
│   │ secret = get_secret("prod/db/pass")  │ ← 安全    │
│   └──────────────────┬───────────────────┘           │
│                      │ API Call                      │
│                      ▼                               │
│   ┌──────────────────────────────────────┐           │
│   │  AWS Secrets Manager / SSM           │           │
│   │  (暗号化・監査・ローテーション)       │           │
│   └──────────────────────────────────────┘           │
└──────────────────────────────────────────────────────┘
```

### 1.2 サービス選択フロー

```
シークレットを管理したい
        │
        ├─ 自動ローテーションが必要？
        │       │
        │       ├─ Yes → Secrets Manager
        │       │
        │       └─ No ─┐
        │               │
        ├─ 設定値か機密値か？
        │       │
        │       ├─ 設定値（Feature Flag 等） → Parameter Store (String)
        │       │
        │       └─ 機密値（パスワード等）   → Parameter Store (SecureString)
        │                                     または Secrets Manager
        │
        └─ 暗号鍵の管理が必要？ → KMS
```

### 1.3 シークレット管理のライフサイクル

```
┌──────────────────────────────────────────────────────────┐
│              シークレットライフサイクル                      │
│                                                          │
│  1. 生成 (Generation)                                    │
│     ├── Secrets Manager の GenerateSecretString           │
│     ├── KMS の GenerateRandom                            │
│     └── 強力なパスワードポリシーの適用                     │
│                                                          │
│  2. 保管 (Storage)                                       │
│     ├── Secrets Manager (KMS 暗号化)                     │
│     ├── Parameter Store SecureString (KMS 暗号化)        │
│     └── バージョニングによる履歴管理                       │
│                                                          │
│  3. 配布 (Distribution)                                  │
│     ├── SDK/CLI での動的取得                               │
│     ├── ECS/Lambda の環境変数注入                         │
│     └── VPC エンドポイント経由のアクセス                   │
│                                                          │
│  4. ローテーション (Rotation)                             │
│     ├── Secrets Manager の自動ローテーション               │
│     ├── Lambda 関数によるカスタムローテーション             │
│     └── マルチユーザーローテーション戦略                    │
│                                                          │
│  5. 監査 (Auditing)                                      │
│     ├── CloudTrail によるアクセスログ                      │
│     ├── Config Rules によるコンプライアンス監視             │
│     └── EventBridge による異常検知アラート                  │
│                                                          │
│  6. 廃棄 (Revocation)                                    │
│     ├── スケジュール削除 (7-30日の復旧期間)                │
│     ├── 即座のアクセス無効化 (リソースポリシー変更)        │
│     └── KMS キーの無効化                                  │
└──────────────────────────────────────────────────────────┘
```

---

## 2. AWS Secrets Manager

### 2.1 シークレットの作成（CLI）

```bash
# シークレット作成
aws secretsmanager create-secret \
  --name "prod/myapp/database" \
  --description "Production DB credentials" \
  --secret-string '{"username":"admin","password":"S3cur3P@ss!","host":"db.example.com","port":5432}' \
  --kms-key-id "alias/prod-database-key" \
  --tags Key=Environment,Value=Production Key=Application,Value=myapp

# シークレット取得
aws secretsmanager get-secret-value \
  --secret-id "prod/myapp/database" \
  --query 'SecretString' \
  --output text

# 特定バージョンの取得
aws secretsmanager get-secret-value \
  --secret-id "prod/myapp/database" \
  --version-stage "AWSPREVIOUS"

# シークレットの更新
aws secretsmanager update-secret \
  --secret-id "prod/myapp/database" \
  --secret-string '{"username":"admin","password":"N3wS3cur3P@ss!","host":"db.example.com","port":5432}'

# シークレットの一覧取得（フィルタ付き）
aws secretsmanager list-secrets \
  --filters Key=name,Values=prod/ \
  --query 'SecretList[*].{Name:Name,Description:Description,LastRotated:LastRotatedDate}'

# シークレットの削除（復旧期間あり）
aws secretsmanager delete-secret \
  --secret-id "prod/myapp/database" \
  --recovery-window-in-days 7

# 削除の取り消し
aws secretsmanager restore-secret \
  --secret-id "prod/myapp/database"
```

### 2.2 Python からの取得

```python
import json
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name: str, region: str = "ap-northeast-1") -> dict:
    """Secrets Manager からシークレットを取得する"""
    client = boto3.client("secretsmanager", region_name=region)

    try:
        response = client.get_secret_value(SecretId=secret_name)
        secret = json.loads(response["SecretString"])
        return secret
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "ResourceNotFoundException":
            raise ValueError(f"シークレット '{secret_name}' が見つかりません")
        elif error_code == "DecryptionFailureException":
            raise PermissionError("KMS 復号に失敗しました")
        raise

# 使用例
creds = get_secret("prod/myapp/database")
connection_string = (
    f"postgresql://{creds['username']}:{creds['password']}"
    f"@{creds['host']}:{creds['port']}/mydb"
)
```

### 2.3 クライアント側キャッシュの実装

```python
from aws_secretsmanager_caching import SecretCache, SecretCacheConfig
import boto3

# キャッシュ設定
cache_config = SecretCacheConfig(
    max_cache_size=1000,           # キャッシュする最大シークレット数
    exception_retry_delay_base=1,   # リトライ間隔（秒）
    exception_retry_growth_factor=2,
    exception_retry_delay_max=3600,
    default_secret_version_stage="AWSCURRENT",
    secret_refresh_interval=3600,   # キャッシュの TTL（秒）
    secret_version_stage_refresh_interval=3600,
)

# キャッシュの初期化
client = boto3.client("secretsmanager", region_name="ap-northeast-1")
cache = SecretCache(config=cache_config, client=client)

# キャッシュ経由でシークレット取得（TTL内はAPI呼び出しなし）
secret_string = cache.get_secret_string("prod/myapp/database")
secret_dict = json.loads(secret_string)

# バイナリシークレットの場合
binary_secret = cache.get_secret_binary("prod/myapp/certificate")
```

### 2.4 Lambda Extensions によるシークレット取得

```python
# Lambda Extension を使ったシークレット取得（コールドスタート最適化）
import urllib3
import json
import os

# AWS Parameters and Secrets Lambda Extension のエンドポイント
SECRETS_EXTENSION_HTTP_PORT = 2773
SECRETS_EXTENSION_ENDPOINT = f"http://localhost:{SECRETS_EXTENSION_HTTP_PORT}"

http = urllib3.PoolManager()

def get_secret_from_extension(secret_id: str) -> dict:
    """Lambda Extension 経由でシークレットを取得（キャッシュ付き）"""
    headers = {
        "X-Aws-Parameters-Secrets-Token": os.environ.get("AWS_SESSION_TOKEN", ""),
    }
    url = (
        f"{SECRETS_EXTENSION_ENDPOINT}/secretsmanager/get"
        f"?secretId={secret_id}"
    )

    response = http.request("GET", url, headers=headers)
    body = json.loads(response.data.decode("utf-8"))
    return json.loads(body["SecretString"])

def handler(event, context):
    """Lambda ハンドラー"""
    # Extension が自動的にキャッシュするため、毎回 API を呼ばない
    db_creds = get_secret_from_extension("prod/myapp/database")
    api_key = get_secret_from_extension("prod/myapp/api-key")

    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Success"})
    }
```

```yaml
# SAM テンプレートで Lambda Extension を追加
AWSTemplateFormatVersion: "2010-09-09"
Transform: AWS::Serverless-2016-10-31
Resources:
  MyFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: index.handler
      Runtime: python3.12
      Layers:
        # AWS Parameters and Secrets Lambda Extension
        - arn:aws:lambda:ap-northeast-1:133490724326:layer:AWS-Parameters-and-Secrets-Lambda-Extension:11
      Environment:
        Variables:
          PARAMETERS_SECRETS_EXTENSION_CACHE_ENABLED: "true"
          PARAMETERS_SECRETS_EXTENSION_CACHE_SIZE: "1000"
          PARAMETERS_SECRETS_EXTENSION_HTTP_PORT: "2773"
          SECRETS_MANAGER_TTL: "300"  # 5分キャッシュ
      Policies:
        - Version: "2012-10-17"
          Statement:
            - Effect: Allow
              Action: secretsmanager:GetSecretValue
              Resource: "arn:aws:secretsmanager:ap-northeast-1:*:secret:prod/myapp/*"
            - Effect: Allow
              Action: kms:Decrypt
              Resource: "arn:aws:kms:ap-northeast-1:*:key/*"
```

### 2.5 自動ローテーション（CloudFormation）

```yaml
AWSTemplateFormatVersion: "2010-09-09"
Resources:
  DBSecret:
    Type: AWS::SecretsManager::Secret
    Properties:
      Name: prod/myapp/database
      GenerateSecretString:
        SecretStringTemplate: '{"username": "admin"}'
        GenerateStringKey: password
        PasswordLength: 32
        ExcludeCharacters: '"@/\'

  DBSecretRotation:
    Type: AWS::SecretsManager::RotationSchedule
    Properties:
      SecretId: !Ref DBSecret
      RotationLambdaARN: !GetAtt RotationFunction.Arn
      RotationRules:
        AutomaticallyAfterDays: 30    # 30日ごとに自動ローテーション

  RotationFunction:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: secret-rotation-handler
      Runtime: python3.12
      Handler: index.handler
      Role: !GetAtt RotationRole.Arn
      Timeout: 60
      VpcConfig:
        SubnetIds:
          - !Ref PrivateSubnet1
          - !Ref PrivateSubnet2
        SecurityGroupIds:
          - !Ref RotationSecurityGroup
      Code:
        ZipFile: |
          import boto3
          import json
          import logging

          logger = logging.getLogger()
          logger.setLevel(logging.INFO)

          def handler(event, context):
              """Secrets Manager ローテーション Lambda"""
              step = event["Step"]
              secret_id = event["SecretId"]
              token = event["ClientRequestToken"]

              client = boto3.client("secretsmanager")

              if step == "createSecret":
                  create_secret(client, secret_id, token)
              elif step == "setSecret":
                  set_secret(client, secret_id, token)
              elif step == "testSecret":
                  test_secret(client, secret_id, token)
              elif step == "finishSecret":
                  finish_secret(client, secret_id, token)

          def create_secret(client, secret_id, token):
              """新パスワードを生成してPENDINGバージョンとして保存"""
              current = client.get_secret_value(
                  SecretId=secret_id, VersionStage="AWSCURRENT"
              )
              current_secret = json.loads(current["SecretString"])

              # 新しいパスワードを生成
              new_password = client.get_random_password(
                  PasswordLength=32,
                  ExcludeCharacters='"@/\\'
              )["RandomPassword"]

              current_secret["password"] = new_password
              client.put_secret_value(
                  SecretId=secret_id,
                  ClientRequestToken=token,
                  SecretString=json.dumps(current_secret),
                  VersionStages=["AWSPENDING"]
              )
              logger.info(f"createSecret: New secret version created for {secret_id}")

          def set_secret(client, secret_id, token):
              """データベースのパスワードを新しい値に変更"""
              pending = client.get_secret_value(
                  SecretId=secret_id, VersionId=token, VersionStage="AWSPENDING"
              )
              secret = json.loads(pending["SecretString"])

              # PostgreSQL のパスワード変更
              import psycopg2
              conn = psycopg2.connect(
                  host=secret["host"],
                  port=secret["port"],
                  user="admin_master",  # マスターユーザーで接続
                  password=get_master_password(client),
                  database="mydb"
              )
              conn.autocommit = True
              with conn.cursor() as cur:
                  cur.execute(
                      f"ALTER USER {secret['username']} WITH PASSWORD %s",
                      (secret["password"],)
                  )
              conn.close()
              logger.info(f"setSecret: DB password updated for {secret_id}")

          def test_secret(client, secret_id, token):
              """新パスワードで接続テスト"""
              pending = client.get_secret_value(
                  SecretId=secret_id, VersionId=token, VersionStage="AWSPENDING"
              )
              secret = json.loads(pending["SecretString"])

              import psycopg2
              conn = psycopg2.connect(
                  host=secret["host"],
                  port=secret["port"],
                  user=secret["username"],
                  password=secret["password"],
                  database="mydb"
              )
              conn.close()
              logger.info(f"testSecret: Connection test passed for {secret_id}")

          def finish_secret(client, secret_id, token):
              """AWSCURRENT ラベルを新バージョンに移動"""
              metadata = client.describe_secret(SecretId=secret_id)
              current_version = None
              for version_id, stages in metadata["VersionIdsToStages"].items():
                  if "AWSCURRENT" in stages:
                      current_version = version_id
                      break

              client.update_secret_version_stage(
                  SecretId=secret_id,
                  VersionStage="AWSCURRENT",
                  MoveToVersionId=token,
                  RemoveFromVersionId=current_version
              )
              logger.info(f"finishSecret: AWSCURRENT moved to {token}")

  RotationRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole
      Policies:
        - PolicyName: RotationPolicy
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Effect: Allow
                Action:
                  - secretsmanager:GetSecretValue
                  - secretsmanager:PutSecretValue
                  - secretsmanager:UpdateSecretVersionStage
                  - secretsmanager:DescribeSecret
                  - secretsmanager:GetRandomPassword
                Resource: !Ref DBSecret
              - Effect: Allow
                Action: kms:Decrypt
                Resource: "*"
```

### 2.6 ローテーションの流れ

```
┌─────────────────────────────────────────────────────────┐
│            Secrets Manager ローテーション 4 Step          │
│                                                         │
│  Step 1: createSecret                                   │
│  ┌───────────┐    新パスワード生成    ┌──────────────┐  │
│  │ Secrets   │ ──────────────────── → │ AWSPENDING   │  │
│  │ Manager   │                        │ (新バージョン)│  │
│  └───────────┘                        └──────────────┘  │
│       │                                                 │
│  Step 2: setSecret                                      │
│       │    DB のパスワードを新しい値に変更               │
│       ▼                                                 │
│  ┌───────────┐                        ┌──────────────┐  │
│  │ RDS / DB  │ ← ALTER USER ... ──── │ Lambda       │  │
│  └───────────┘                        └──────────────┘  │
│       │                                                 │
│  Step 3: testSecret                                     │
│       │    新パスワードで接続テスト                      │
│       ▼                                                 │
│  Step 4: finishSecret                                   │
│       │    AWSCURRENT ラベルを新バージョンに移動         │
│       ▼                                                 │
│  ┌──────────────┐                                       │
│  │ AWSCURRENT   │  ← ラベル移動完了                     │
│  │ (新バージョン)│                                       │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘
```

### 2.7 マルチユーザーローテーション戦略

```
┌──────────────────────────────────────────────────────────┐
│      マルチユーザーローテーション (Alternating Users)       │
│                                                          │
│  初期状態:                                                │
│  ┌──────────────────────┐                                │
│  │ app_user_1 (CURRENT) │  ← アプリはこのユーザーで接続   │
│  │ app_user_2 (STANDBY) │  ← 待機中                      │
│  └──────────────────────┘                                │
│                                                          │
│  ローテーション後:                                        │
│  ┌──────────────────────┐                                │
│  │ app_user_1 (STANDBY) │  ← パスワード変更済み・待機     │
│  │ app_user_2 (CURRENT) │  ← アプリはこのユーザーに切替   │
│  └──────────────────────┘                                │
│                                                          │
│  利点:                                                   │
│  - ローテーション中のダウンタイムなし                      │
│  - ロールバックが容易                                     │
│  - 古い接続が有効なまま新接続を開始可能                    │
└──────────────────────────────────────────────────────────┘
```

### 2.8 Secrets Manager のリソースポリシー

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowCrossAccountRead",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::222222222222:role/AppRole"
      },
      "Action": "secretsmanager:GetSecretValue",
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "secretsmanager:VersionStage": "AWSCURRENT"
        }
      }
    },
    {
      "Sid": "DenyNonVPCEndpoint",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "secretsmanager:*",
      "Resource": "*",
      "Condition": {
        "StringNotEquals": {
          "aws:sourceVpce": "vpce-xxxxxxxxxxxx"
        }
      }
    }
  ]
}
```

```bash
# リソースポリシーの設定
aws secretsmanager put-resource-policy \
  --secret-id "prod/myapp/database" \
  --resource-policy file://secret-policy.json

# リソースポリシーの検証
aws secretsmanager validate-resource-policy \
  --resource-policy file://secret-policy.json
```

### 2.9 シークレットのレプリケーション

```bash
# マルチリージョンレプリケーション
aws secretsmanager replicate-secret-to-regions \
  --secret-id "prod/myapp/database" \
  --add-replica-regions '[
    {"Region": "us-west-2", "KmsKeyId": "alias/prod-key-usw2"},
    {"Region": "eu-west-1", "KmsKeyId": "alias/prod-key-euw1"}
  ]'

# レプリカの状態確認
aws secretsmanager describe-secret \
  --secret-id "prod/myapp/database" \
  --query 'ReplicationStatus'

# レプリカの削除
aws secretsmanager remove-regions-from-replication \
  --secret-id "prod/myapp/database" \
  --remove-replica-regions "eu-west-1"
```

---

## 3. Systems Manager Parameter Store

### 3.1 パラメータの階層設計

```bash
# 階層型パラメータの作成
aws ssm put-parameter \
  --name "/myapp/prod/database/host" \
  --value "db.example.com" \
  --type String

aws ssm put-parameter \
  --name "/myapp/prod/database/password" \
  --value "S3cur3P@ss!" \
  --type SecureString \
  --key-id "alias/myapp-key"    # KMS キーを指定

aws ssm put-parameter \
  --name "/myapp/prod/database/port" \
  --value "5432" \
  --type String \
  --tags Key=Environment,Value=Production

# 階層ごとの一括取得
aws ssm get-parameters-by-path \
  --path "/myapp/prod/database" \
  --with-decryption \
  --recursive

# 複数パラメータの同時取得
aws ssm get-parameters \
  --names "/myapp/prod/database/host" "/myapp/prod/database/port" \
  --with-decryption

# パラメータの履歴取得
aws ssm get-parameter-history \
  --name "/myapp/prod/database/password" \
  --with-decryption \
  --query 'Parameters[*].{Version:Version,Value:Value,LastModifiedDate:LastModifiedDate}'
```

### 3.2 階層設計のベストプラクティス

```
推奨する階層構造:

/
├── myapp/                          # アプリケーション名
│   ├── shared/                     # 全環境共通
│   │   ├── log-level               # String: "INFO"
│   │   └── feature-flags/          # Feature Flags
│   │       ├── dark-mode           # String: "true"
│   │       └── new-ui              # String: "false"
│   │
│   ├── prod/                       # 本番環境
│   │   ├── database/
│   │   │   ├── host                # String: "prod-db.xxx.rds.amazonaws.com"
│   │   │   ├── port                # String: "5432"
│   │   │   └── password            # SecureString: "xxx"
│   │   ├── redis/
│   │   │   ├── endpoint            # String: "prod-redis.xxx.cache.amazonaws.com"
│   │   │   └── auth-token          # SecureString: "xxx"
│   │   └── api-keys/
│   │       ├── stripe              # SecureString: "sk_live_xxx"
│   │       └── sendgrid            # SecureString: "SG.xxx"
│   │
│   └── staging/                    # ステージング環境
│       ├── database/
│       │   ├── host                # String: "staging-db.xxx.rds.amazonaws.com"
│       │   └── password            # SecureString: "xxx"
│       └── api-keys/
│           └── stripe              # SecureString: "sk_test_xxx"

IAM ポリシーでの階層制御:
- /myapp/prod/*    → 本番チームのみ
- /myapp/staging/* → 開発チームも可
- /myapp/shared/*  → 全チーム読み取り可
```

### 3.3 Python での階層的なパラメータ取得

```python
import boto3
from typing import Any

class ParameterStoreClient:
    """Parameter Store の階層型パラメータをdictとして取得するクライアント"""

    def __init__(self, region: str = "ap-northeast-1"):
        self.client = boto3.client("ssm", region_name=region)

    def get_parameters_by_path(
        self, path: str, decrypt: bool = True
    ) -> dict[str, Any]:
        """指定パス以下のパラメータを辞書として返す"""
        parameters = {}
        paginator = self.client.get_paginator("get_parameters_by_path")

        for page in paginator.paginate(
            Path=path,
            Recursive=True,
            WithDecryption=decrypt,
        ):
            for param in page["Parameters"]:
                # パスの末尾部分をキーにする
                key = param["Name"].replace(path, "").lstrip("/")
                parameters[key] = param["Value"]

        return parameters

    def get_config(self, app: str, env: str) -> dict:
        """アプリケーション設定を環境ごとに取得"""
        # 共通設定
        shared = self.get_parameters_by_path(f"/{app}/shared")
        # 環境固有設定
        env_specific = self.get_parameters_by_path(f"/{app}/{env}")

        # 環境固有が共通を上書き
        config = {**shared, **env_specific}
        return config

# 使用例
ssm = ParameterStoreClient()
config = ssm.get_config("myapp", "prod")
# → {"log-level": "INFO", "database/host": "prod-db.xxx", ...}
```

### 3.4 ECS タスク定義での参照

```json
{
  "containerDefinitions": [
    {
      "name": "myapp",
      "image": "123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/myapp:latest",
      "secrets": [
        {
          "name": "DB_PASSWORD",
          "valueFrom": "arn:aws:ssm:ap-northeast-1:123456789012:parameter/myapp/prod/database/password"
        },
        {
          "name": "API_KEY",
          "valueFrom": "arn:aws:secretsmanager:ap-northeast-1:123456789012:secret:prod/myapp/api-key-AbCdEf"
        }
      ],
      "environment": [
        {
          "name": "DB_HOST",
          "value": "db.example.com"
        }
      ]
    }
  ]
}
```

### 3.5 CloudFormation での動的参照

```yaml
AWSTemplateFormatVersion: "2010-09-09"
Resources:
  # Parameter Store の値を参照
  MyRDSInstance:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceIdentifier: myapp-db
      Engine: postgres
      MasterUsername: "{{resolve:ssm:/myapp/prod/database/username}}"
      MasterUserPassword: "{{resolve:ssm-secure:/myapp/prod/database/password}}"
      DBInstanceClass: db.r6g.large

  # Secrets Manager の値を参照
  MyRDSInstanceV2:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceIdentifier: myapp-db-v2
      Engine: postgres
      MasterUsername: "{{resolve:secretsmanager:prod/myapp/database:SecretString:username}}"
      MasterUserPassword: "{{resolve:secretsmanager:prod/myapp/database:SecretString:password}}"
      DBInstanceClass: db.r6g.large

  # Secrets Manager + バージョン指定
  MySecret:
    Type: AWS::SecretsManager::Secret
    Properties:
      Name: prod/myapp/database

  # RDS と Secrets Manager の直接統合
  SecretRDSAttachment:
    Type: AWS::SecretsManager::SecretTargetAttachment
    Properties:
      SecretId: !Ref MySecret
      TargetId: !Ref MyRDSInstanceV2
      TargetType: AWS::RDS::DBInstance
```

---

## 4. AWS KMS（Key Management Service）

### 4.1 エンベロープ暗号化の仕組み

```
┌──────────────────────────────────────────────────────────┐
│              エンベロープ暗号化                            │
│                                                          │
│  ┌─────────────┐                                         │
│  │ CMK (Master)│  KMS 内部に保管（エクスポート不可）      │
│  │  Key        │                                         │
│  └──────┬──────┘                                         │
│         │ GenerateDataKey API                            │
│         ▼                                                │
│  ┌─────────────────────────────────┐                     │
│  │ Data Key (平文)   │ Data Key    │                     │
│  │                   │ (暗号化済み) │                     │
│  └────────┬──────────┴──────┬──────┘                     │
│           │                 │                            │
│    平文 Data Key で         │ 暗号化済み Data Key を      │
│    データを暗号化            │ データと一緒に保存          │
│           │                 │                            │
│           ▼                 ▼                            │
│  ┌──────────────────────────────┐                        │
│  │ 暗号化データ + 暗号化 Data Key │  ← S3 等に保存        │
│  └──────────────────────────────┘                        │
│                                                          │
│  ※ 平文 Data Key はメモリから即座に削除                   │
└──────────────────────────────────────────────────────────┘
```

### 4.2 KMS キーの作成とポリシー

```bash
# カスタマーマネージドキー作成
aws kms create-key \
  --description "MyApp encryption key" \
  --key-usage ENCRYPT_DECRYPT \
  --key-spec SYMMETRIC_DEFAULT \
  --tags TagKey=Environment,TagValue=Production TagKey=Application,TagValue=myapp

# エイリアス設定
aws kms create-alias \
  --alias-name alias/myapp-key \
  --target-key-id "arn:aws:kms:ap-northeast-1:123456789012:key/xxxx-xxxx"

# キーの自動ローテーション有効化（1年ごと）
aws kms enable-key-rotation \
  --key-id alias/myapp-key

# ローテーション状態の確認
aws kms get-key-rotation-status \
  --key-id alias/myapp-key

# キーポリシーの設定
aws kms put-key-policy \
  --key-id alias/myapp-key \
  --policy-name default \
  --policy '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Sid": "EnableRootAccountFullAccess",
        "Effect": "Allow",
        "Principal": {"AWS": "arn:aws:iam::123456789012:root"},
        "Action": "kms:*",
        "Resource": "*"
      },
      {
        "Sid": "AllowKeyAdministration",
        "Effect": "Allow",
        "Principal": {"AWS": "arn:aws:iam::123456789012:role/KeyAdminRole"},
        "Action": [
          "kms:Create*", "kms:Describe*", "kms:Enable*", "kms:List*",
          "kms:Put*", "kms:Update*", "kms:Revoke*", "kms:Disable*",
          "kms:Get*", "kms:Delete*", "kms:TagResource", "kms:UntagResource",
          "kms:ScheduleKeyDeletion", "kms:CancelKeyDeletion"
        ],
        "Resource": "*"
      },
      {
        "Sid": "AllowKeyUsage",
        "Effect": "Allow",
        "Principal": {"AWS": "arn:aws:iam::123456789012:role/MyAppRole"},
        "Action": [
          "kms:Encrypt", "kms:Decrypt", "kms:ReEncrypt*",
          "kms:GenerateDataKey*", "kms:DescribeKey"
        ],
        "Resource": "*"
      },
      {
        "Sid": "AllowServiceIntegration",
        "Effect": "Allow",
        "Principal": {"AWS": "arn:aws:iam::123456789012:role/MyAppRole"},
        "Action": [
          "kms:CreateGrant", "kms:ListGrants", "kms:RevokeGrant"
        ],
        "Resource": "*",
        "Condition": {
          "Bool": {"kms:GrantIsForAWSResource": "true"}
        }
      }
    ]
  }'
```

### 4.3 Python でのエンベロープ暗号化

```python
import boto3
from cryptography.fernet import Fernet
import base64

kms = boto3.client("kms", region_name="ap-northeast-1")

def encrypt_data(plaintext: str, key_id: str) -> dict:
    """エンベロープ暗号化でデータを暗号化"""
    # 1. データキーを生成
    response = kms.generate_data_key(
        KeyId=key_id,
        KeySpec="AES_256"
    )
    plaintext_key = response["Plaintext"]        # 平文データキー
    encrypted_key = response["CiphertextBlob"]    # 暗号化データキー

    # 2. 平文データキーでデータを暗号化
    fernet_key = base64.urlsafe_b64encode(plaintext_key)
    f = Fernet(fernet_key)
    encrypted_data = f.encrypt(plaintext.encode())

    # 3. 平文データキーをメモリから削除
    del plaintext_key, fernet_key

    return {
        "encrypted_data": base64.b64encode(encrypted_data).decode(),
        "encrypted_key": base64.b64encode(encrypted_key).decode()
    }

def decrypt_data(encrypted_payload: dict) -> str:
    """エンベロープ暗号化されたデータを復号"""
    encrypted_key = base64.b64decode(encrypted_payload["encrypted_key"])
    encrypted_data = base64.b64decode(encrypted_payload["encrypted_data"])

    # 1. KMS でデータキーを復号
    response = kms.decrypt(CiphertextBlob=encrypted_key)
    plaintext_key = response["Plaintext"]

    # 2. 復号されたデータキーでデータを復号
    fernet_key = base64.urlsafe_b64encode(plaintext_key)
    f = Fernet(fernet_key)
    decrypted = f.decrypt(encrypted_data)

    del plaintext_key, fernet_key
    return decrypted.decode()
```

### 4.4 AWS Encryption SDK の活用

```python
# AWS Encryption SDK を使ったより堅牢な暗号化
import aws_encryption_sdk
from aws_encryption_sdk import CommitmentPolicy

# クライアント初期化
client = aws_encryption_sdk.EncryptionSDKClient(
    commitment_policy=CommitmentPolicy.REQUIRE_ENCRYPT_REQUIRE_DECRYPT
)

# KMS Key Provider
kms_key_provider = aws_encryption_sdk.StrictAwsKmsMasterKeyProvider(
    key_ids=[
        "arn:aws:kms:ap-northeast-1:123456789012:key/xxxx-xxxx",
        "arn:aws:kms:us-west-2:123456789012:key/yyyy-yyyy",  # マルチリージョンキー
    ]
)

def encrypt_with_sdk(plaintext: str, context: dict) -> bytes:
    """Encryption SDK でデータを暗号化（暗号化コンテキスト付き）"""
    ciphertext, encryptor_header = client.encrypt(
        source=plaintext.encode(),
        key_provider=kms_key_provider,
        encryption_context=context,  # 改ざん検知用のコンテキスト
    )
    return ciphertext

def decrypt_with_sdk(ciphertext: bytes, expected_context: dict) -> str:
    """Encryption SDK でデータを復号"""
    plaintext, decryptor_header = client.decrypt(
        source=ciphertext,
        key_provider=kms_key_provider,
    )
    # 暗号化コンテキストの検証
    for key, value in expected_context.items():
        assert decryptor_header.encryption_context.get(key) == value, \
            f"Encryption context mismatch for key: {key}"

    return plaintext.decode()

# 使用例
context = {
    "purpose": "user-data-encryption",
    "tenant": "company-a",
    "data-type": "pii",
}
encrypted = encrypt_with_sdk("個人情報データ", context)
decrypted = decrypt_with_sdk(encrypted, context)
```

### 4.5 KMS キーの用途別分離設計

```
┌──────────────────────────────────────────────────────────┐
│              KMS キー分離設計                               │
│                                                          │
│  用途別キー:                                              │
│  ┌──────────────────┐                                    │
│  │ alias/prod-db    │ → RDS, Secrets Manager (DB認証情報) │
│  │ alias/prod-s3    │ → S3 バケット暗号化                 │
│  │ alias/prod-ebs   │ → EBS ボリューム暗号化              │
│  │ alias/prod-logs  │ → CloudWatch Logs 暗号化            │
│  │ alias/prod-sqs   │ → SQS メッセージ暗号化              │
│  │ alias/prod-sign  │ → 署名用 (RSA/ECC)                 │
│  └──────────────────┘                                    │
│                                                          │
│  利点:                                                   │
│  - キーポリシーで最小権限を実現                            │
│  - 1つのキー無効化が全サービスに影響しない                 │
│  - 監査ログでアクセス目的を特定しやすい                    │
│  - コンプライアンス要件（PCI DSS 等）への対応が容易        │
└──────────────────────────────────────────────────────────┘
```

---

## 5. CDK によるシークレット管理

### 5.1 CDK でのシークレット定義

```typescript
import * as cdk from 'aws-cdk-lib';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as rds from 'aws-cdk-lib/aws-rds';
import { Construct } from 'constructs';

export class SecretsStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // KMS キーの作成
    const dbEncryptionKey = new kms.Key(this, 'DBEncryptionKey', {
      alias: 'prod-database-key',
      description: 'Encryption key for database secrets',
      enableKeyRotation: true,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // Secrets Manager でシークレット作成
    const dbSecret = new secretsmanager.Secret(this, 'DBSecret', {
      secretName: 'prod/myapp/database',
      description: 'Production database credentials',
      encryptionKey: dbEncryptionKey,
      generateSecretString: {
        secretStringTemplate: JSON.stringify({ username: 'admin' }),
        generateStringKey: 'password',
        passwordLength: 32,
        excludeCharacters: '"@/\\',
      },
    });

    // RDS インスタンスとの統合
    const dbInstance = new rds.DatabaseInstance(this, 'Database', {
      engine: rds.DatabaseInstanceEngine.postgres({
        version: rds.PostgresEngineVersion.VER_15_4,
      }),
      instanceType: cdk.aws_ec2.InstanceType.of(
        cdk.aws_ec2.InstanceClass.R6G,
        cdk.aws_ec2.InstanceSize.LARGE,
      ),
      credentials: rds.Credentials.fromSecret(dbSecret),
      storageEncryptionKey: dbEncryptionKey,
    });

    // 自動ローテーションの設定
    dbSecret.addRotationSchedule('RotationSchedule', {
      automaticallyAfter: cdk.Duration.days(30),
      hostedRotation: secretsmanager.HostedRotation.postgreSqlSingleUser({
        functionName: 'db-secret-rotation',
      }),
    });

    // Parameter Store パラメータ
    new ssm.StringParameter(this, 'DBHost', {
      parameterName: '/myapp/prod/database/host',
      stringValue: dbInstance.instanceEndpoint.hostname,
      tier: ssm.ParameterTier.STANDARD,
    });

    // シークレット ARN の出力
    new cdk.CfnOutput(this, 'SecretArn', {
      value: dbSecret.secretArn,
      description: 'Database secret ARN',
    });
  }
}
```

---

## 6. VPC エンドポイント経由のシークレットアクセス

### 6.1 プライベートサブネットからのアクセス設計

```
┌──────────────────────────────────────────────────────────┐
│              VPC エンドポイント経由のアクセス                │
│                                                          │
│  VPC (10.0.0.0/16)                                       │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Private Subnet                                     │  │
│  │  ┌───────────────┐    VPC Endpoint     ┌─────────┐ │  │
│  │  │ ECS / Lambda  │ ─── (Interface) ──→ │ Secrets │ │  │
│  │  │ (No IGW)      │    vpce-xxxx        │ Manager │ │  │
│  │  └───────────────┘                     └─────────┘ │  │
│  │                                                     │  │
│  │  ┌───────────────┐    VPC Endpoint     ┌─────────┐ │  │
│  │  │ ECS / Lambda  │ ─── (Interface) ──→ │ SSM     │ │  │
│  │  │               │    vpce-yyyy        │ (Param) │ │  │
│  │  └───────────────┘                     └─────────┘ │  │
│  │                                                     │  │
│  │  ┌───────────────┐    VPC Endpoint     ┌─────────┐ │  │
│  │  │ ECS / Lambda  │ ─── (Interface) ──→ │ KMS     │ │  │
│  │  │               │    vpce-zzzz        │         │ │  │
│  │  └───────────────┘                     └─────────┘ │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  トラフィックはインターネットを経由しない                   │
└──────────────────────────────────────────────────────────┘
```

```bash
# VPC エンドポイントの作成
aws ec2 create-vpc-endpoint \
  --vpc-id vpc-xxxx \
  --service-name com.amazonaws.ap-northeast-1.secretsmanager \
  --vpc-endpoint-type Interface \
  --subnet-ids subnet-aaaa subnet-bbbb \
  --security-group-ids sg-xxxx \
  --private-dns-enabled

aws ec2 create-vpc-endpoint \
  --vpc-id vpc-xxxx \
  --service-name com.amazonaws.ap-northeast-1.ssm \
  --vpc-endpoint-type Interface \
  --subnet-ids subnet-aaaa subnet-bbbb \
  --security-group-ids sg-xxxx \
  --private-dns-enabled

aws ec2 create-vpc-endpoint \
  --vpc-id vpc-xxxx \
  --service-name com.amazonaws.ap-northeast-1.kms \
  --vpc-endpoint-type Interface \
  --subnet-ids subnet-aaaa subnet-bbbb \
  --security-group-ids sg-xxxx \
  --private-dns-enabled
```

---

## 7. シークレットの監査と監視

### 7.1 CloudTrail によるアクセス監視

```python
import boto3
import json
from datetime import datetime, timedelta

def audit_secret_access(secret_name: str, hours: int = 24) -> list[dict]:
    """特定シークレットへのアクセス履歴を取得"""
    ct = boto3.client("cloudtrail", region_name="ap-northeast-1")

    events = ct.lookup_events(
        LookupAttributes=[
            {"AttributeKey": "ResourceName", "AttributeValue": secret_name}
        ],
        StartTime=datetime.utcnow() - timedelta(hours=hours),
        EndTime=datetime.utcnow(),
    )

    results = []
    for event in events.get("Events", []):
        detail = json.loads(event["CloudTrailEvent"])
        results.append({
            "Time": str(event["EventTime"]),
            "Event": event["EventName"],
            "User": detail.get("userIdentity", {}).get("arn", "Unknown"),
            "SourceIP": detail.get("sourceIPAddress", "Unknown"),
            "Success": "errorCode" not in detail,
        })

    return results
```

### 7.2 Config Rules によるコンプライアンス監視

```yaml
# AWS Config Rules for Secrets Manager
AWSTemplateFormatVersion: "2010-09-09"
Resources:
  # シークレットのローテーションが有効か確認
  SecretRotationEnabled:
    Type: AWS::Config::ConfigRule
    Properties:
      ConfigRuleName: secretsmanager-rotation-enabled
      Source:
        Owner: AWS
        SourceIdentifier: SECRETSMANAGER_ROTATION_ENABLED_CHECK
      InputParameters:
        maximumAllowedRotationFrequency: 90  # 最大90日間隔

  # シークレットが未使用でないか確認
  SecretUnused:
    Type: AWS::Config::ConfigRule
    Properties:
      ConfigRuleName: secretsmanager-secret-unused
      Source:
        Owner: AWS
        SourceIdentifier: SECRETSMANAGER_SECRET_UNUSED
      InputParameters:
        unusedForDays: 90

  # シークレットが自動ローテーション対象か確認
  SecretScheduledRotation:
    Type: AWS::Config::ConfigRule
    Properties:
      ConfigRuleName: secretsmanager-scheduled-rotation
      Source:
        Owner: AWS
        SourceIdentifier: SECRETSMANAGER_SCHEDULED_ROTATION_SUCCESS_CHECK

  # KMS キーのローテーションが有効か確認
  KMSKeyRotation:
    Type: AWS::Config::ConfigRule
    Properties:
      ConfigRuleName: kms-key-rotation-enabled
      Source:
        Owner: AWS
        SourceIdentifier: CMK_BACKING_KEY_ROTATION_ENABLED
```

### 7.3 EventBridge によるシークレット異常検知

```yaml
# シークレットへの異常アクセスを検知
AWSTemplateFormatVersion: "2010-09-09"
Resources:
  SecretAccessAlert:
    Type: AWS::Events::Rule
    Properties:
      Name: secret-access-anomaly-alert
      EventPattern:
        source:
          - "aws.secretsmanager"
        detail-type:
          - "AWS API Call via CloudTrail"
        detail:
          eventName:
            - "GetSecretValue"
            - "PutSecretValue"
            - "DeleteSecret"
            - "UpdateSecret"
          errorCode:
            - "AccessDeniedException"
            - "DecryptionFailureException"
      Targets:
        - Arn: !Ref AlertTopic
          Id: secret-alert

  SecretRotationFailure:
    Type: AWS::Events::Rule
    Properties:
      Name: secret-rotation-failure-alert
      EventPattern:
        source:
          - "aws.secretsmanager"
        detail-type:
          - "AWS API Call via CloudTrail"
        detail:
          eventName:
            - "RotationFailed"
      Targets:
        - Arn: !Ref AlertTopic
          Id: rotation-failure

  AlertTopic:
    Type: AWS::SNS::Topic
    Properties:
      TopicName: secrets-security-alerts
```

---

## 8. サービス比較

### 8.1 Secrets Manager vs Parameter Store

| 機能 | Secrets Manager | Parameter Store (Standard) | Parameter Store (Advanced) |
|------|----------------|---------------------------|---------------------------|
| **料金** | $0.40/シークレット/月 + API料 | 無料 | $0.05/パラメータ/月 |
| **最大サイズ** | 64 KB | 4 KB | 8 KB |
| **自動ローテーション** | 組み込みサポート | Lambda で自前実装 | Lambda で自前実装 |
| **クロスアカウント共有** | IAM ポリシーで可能 | 不可 | 不可 |
| **バージョニング** | あり（ステージラベル） | あり（番号のみ） | あり（番号のみ） |
| **暗号化** | KMS 必須 | SecureString で KMS | SecureString で KMS |
| **CloudFormation 動的参照** | `{{resolve:secretsmanager:...}}` | `{{resolve:ssm:...}}` | `{{resolve:ssm:...}}` |
| **レプリケーション** | マルチリージョン対応 | 不可 | 不可 |
| **リソースポリシー** | あり | なし | なし |
| **推奨用途** | DB 認証情報, API キー | 設定値, Feature Flag | 大きめの設定値 |
| **Lambda Extension** | あり | あり | あり |
| **パラメータ数上限** | 制限なし (API制限あり) | 10,000 | 100,000 |

### 8.2 KMS キータイプ比較

| キータイプ | 管理者 | コスト | ローテーション | ユースケース |
|-----------|--------|--------|---------------|-------------|
| **AWS マネージドキー** (`aws/xxx`) | AWS | 無料 | 自動（3年） | サービスデフォルト暗号化 |
| **カスタマーマネージドキー** | ユーザー | $1/月 + API料 | 手動 or 自動設定 | 細かいアクセス制御が必要 |
| **カスタマー提供キー** (BYOK) | ユーザー | API料のみ | ユーザー管理 | コンプライアンス要件 |
| **外部キーストア** | ユーザー | $1/月 + API料 | ユーザー管理 | HSM 連携, 規制対応 |
| **マルチリージョンキー** | ユーザー | $1/月/リージョン | 自動(設定時) | DR, マルチリージョン暗号化 |

### 8.3 暗号化コンテキストの活用比較

| 用途 | コンテキストキー | 値の例 | 効果 |
|------|-----------------|--------|------|
| **テナント分離** | `tenant-id` | `company-abc` | テナント間のデータ混在防止 |
| **データ分類** | `data-classification` | `pii`, `confidential` | データ種別の追跡 |
| **アクセス制御** | `purpose` | `backup`, `analytics` | IAM Condition での制御 |
| **監査** | `request-id` | `req-12345` | CloudTrail での追跡 |

---

## 9. アンチパターン

### 9.1 環境変数にシークレットを平文で設定

```yaml
# NG: docker-compose.yml にパスワードをハードコード
services:
  app:
    environment:
      - DB_PASSWORD=P@ssw0rd123    # Git に commit される
      - API_KEY=sk-live-abc123     # docker inspect で閲覧可能

# OK: Secrets Manager から動的取得
services:
  app:
    environment:
      - SECRET_ARN=arn:aws:secretsmanager:ap-northeast-1:123456789012:secret:prod/db
    # アプリ起動時に SDK で取得
```

**問題点**: 環境変数は `docker inspect`、`/proc/<pid>/environ`、ログ出力で容易に漏洩する。

### 9.2 全てのシークレットに同一 KMS キーを使う

```
NG:
  全シークレット → aws/secretsmanager (デフォルトキー)
    → 鍵の取り消しで全シークレットが使用不能

OK:
  本番DB → alias/prod-database-key
  API連携 → alias/prod-api-key
  監査ログ → alias/prod-audit-key
    → 影響範囲を限定
```

**問題点**: キーポリシー変更の影響が全シークレットに波及し、障害時の影響範囲が広がる。

### 9.3 ローテーションなしで長期間使用

```
NG:
  DB パスワードを作成時から一度も変更せず2年間使用
  → 退職者が古いパスワードを知っている可能性
  → 侵害されていても検知できない

OK:
  Secrets Manager の自動ローテーションを30日間隔で設定
  → 定期的にパスワードが自動変更される
  → 仮に漏洩しても有効期間が限定される
```

### 9.4 シークレットのログ出力

```python
# NG: シークレットをログに出力
import logging
logger = logging.getLogger()

secret = get_secret("prod/myapp/database")
logger.info(f"DB connection: {secret}")  # パスワードがログに残る

# OK: マスキングしてログ出力
def mask_secret(secret_dict: dict, mask_keys: list[str]) -> dict:
    """機密フィールドをマスクして返す"""
    masked = secret_dict.copy()
    for key in mask_keys:
        if key in masked:
            masked[key] = "***MASKED***"
    return masked

logger.info(f"DB connection: {mask_secret(secret, ['password', 'api_key'])}")
```

### 9.5 VPC エンドポイントなしでプライベートサブネットからアクセス

```
NG:
  Private Subnet → NAT Gateway → Internet → Secrets Manager
  → インターネットを経由するため通信経路が長い
  → NAT Gateway のコストが発生
  → セキュリティリスクが増加

OK:
  Private Subnet → VPC Endpoint → Secrets Manager
  → AWS ネットワーク内で完結
  → インターネット非経由でコスト削減
  → VPC エンドポイントポリシーで追加制御
```

---

## 10. FAQ

### Q1. Secrets Manager と Parameter Store SecureString、どちらを使うべき？

**A.** 自動ローテーションが必要なら Secrets Manager。コスト重視で手動管理可能なら Parameter Store SecureString。RDS/Redshift/DocumentDB のローテーションは Secrets Manager に組み込みテンプレートがあり、設定が容易。クロスアカウント共有が必要な場合も Secrets Manager を選択する。

### Q2. KMS の API コール料金が心配。キャッシュは可能？

**A.** Secrets Manager SDK にはクライアント側キャッシュライブラリがある。`aws-secretsmanager-caching` (Python) や `aws-secretsmanager-caching-java` を使えば TTL ベースでキャッシュされ、API コール数を大幅に削減できる。Lambda Extension も同様のキャッシュ機能を提供する。

```python
from aws_secretsmanager_caching import SecretCache

cache = SecretCache()
secret = cache.get_secret_string("prod/myapp/database")  # TTL 内はキャッシュ
```

### Q3. Lambda からシークレットにアクセスする最も安全な方法は？

**A.** Lambda の実行ロールに最小権限の IAM ポリシーをアタッチし、VPC エンドポイント経由でアクセスする。Lambda Extensions を使えばランタイム外でシークレットを取得・キャッシュできる。

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": "arn:aws:secretsmanager:ap-northeast-1:123456789012:secret:prod/myapp/*"
    },
    {
      "Effect": "Allow",
      "Action": "kms:Decrypt",
      "Resource": "arn:aws:kms:ap-northeast-1:123456789012:key/specific-key-id"
    }
  ]
}
```

### Q4. シークレットのローテーション中にアプリケーションが影響を受けないようにするには？

**A.** (1) マルチユーザーローテーション戦略を使い、2つのユーザーを交互に切り替える。(2) アプリケーション側でシークレット取得時にリトライロジックを実装する。(3) Secrets Manager のバージョンステージ（AWSCURRENT/AWSPREVIOUS）を活用し、古い認証情報でも一時的にアクセス可能にする。(4) コネクションプールの再接続ロジックを実装する。

### Q5. git-secrets などのツールでシークレットの漏洩を防ぐ方法は？

**A.** (1) `git-secrets` をプリコミットフックに設定し、AWS アクセスキーパターンをブロック。(2) GitHub の Secret Scanning を有効化。(3) `trufflehog` や `gitleaks` で定期的にリポジトリをスキャン。(4) AWS の IAM Access Analyzer で公開されたシークレットを検出。これらを CI/CD パイプラインに組み込むことで多層防御を実現する。

```bash
# git-secrets のセットアップ
git secrets --install
git secrets --register-aws

# プリコミットフックで検査
git secrets --scan
```

---

## 11. まとめ

| 項目 | ポイント |
|------|---------|
| **Secrets Manager** | 自動ローテーション対応、RDS 統合、$0.40/月/シークレット |
| **Parameter Store** | 無料枠あり、階層構造、設定値と機密値の両方に対応 |
| **KMS** | エンベロープ暗号化の基盤、キーポリシーで細かいアクセス制御 |
| **設計原則** | シークレットはコードに含めない、最小権限、用途別にキーを分離 |
| **キャッシュ** | クライアント側キャッシュや Lambda Extension で API コスト削減 |
| **ローテーション** | 30日以内の自動ローテーション、マルチユーザー戦略でダウンタイム回避 |
| **ネットワーク** | VPC エンドポイント経由でインターネット非経由のアクセス |
| **監査** | CloudTrail + Config Rules + EventBridge で監視・コンプライアンス |
| **IaC** | CDK/CloudFormation でシークレット管理をコード化 |

---

## 次に読むべきガイド

- [02-waf-shield.md](./02-waf-shield.md) — WAF/Shield によるアプリケーション保護
- [00-iam-deep-dive.md](./00-iam-deep-dive.md) — IAM ポリシー設計とロール管理
- VPC エンドポイント設計 — プライベートネットワークからのサービスアクセス

---

## 参考文献

1. **AWS公式ドキュメント** — "AWS Secrets Manager User Guide" — https://docs.aws.amazon.com/secretsmanager/latest/userguide/
2. **AWS公式ドキュメント** — "AWS Systems Manager Parameter Store User Guide" — https://docs.aws.amazon.com/systems-manager/latest/userguide/systems-manager-parameter-store.html
3. **AWS公式ドキュメント** — "AWS Key Management Service Developer Guide" — https://docs.aws.amazon.com/kms/latest/developerguide/
4. **AWS Well-Architected Framework** — Security Pillar — "Protect data at rest" — https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/
5. **AWS Encryption SDK** — "AWS Encryption SDK Developer Guide" — https://docs.aws.amazon.com/encryption-sdk/latest/developer-guide/
6. **AWS Lambda Extensions** — "Using AWS Parameters and Secrets Lambda Extension" — https://docs.aws.amazon.com/secretsmanager/latest/userguide/retrieving-secrets_lambda.html

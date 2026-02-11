# シークレット管理 — Secrets Manager / Parameter Store / KMS

> AWS におけるシークレット（機密情報）のライフサイクルを安全に管理し、アプリケーションからハードコードされた認証情報を排除するための実践ガイド。

---

## この章で学ぶこと

1. **AWS Secrets Manager** によるシークレットの自動ローテーションと取得パターン
2. **Systems Manager Parameter Store** との使い分けと階層型パラメータ設計
3. **AWS KMS（Key Management Service）** によるエンベロープ暗号化と鍵ポリシー設計

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

---

## 2. AWS Secrets Manager

### 2.1 シークレットの作成（CLI）

```bash
# シークレット作成
aws secretsmanager create-secret \
  --name "prod/myapp/database" \
  --description "Production DB credentials" \
  --secret-string '{"username":"admin","password":"S3cur3P@ss!","host":"db.example.com","port":5432}'

# シークレット取得
aws secretsmanager get-secret-value \
  --secret-id "prod/myapp/database" \
  --query 'SecretString' \
  --output text
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

### 2.3 自動ローテーション（CloudFormation）

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
      Code:
        ZipFile: |
          import boto3
          import json

          def handler(event, context):
              """Secrets Manager ローテーション Lambda"""
              step = event["Step"]
              secret_id = event["SecretId"]
              token = event["ClientRequestToken"]

              client = boto3.client("secretsmanager")

              if step == "createSecret":
                  # 新パスワード生成
                  client.get_random_password(PasswordLength=32)
              elif step == "setSecret":
                  # DB 側のパスワード変更
                  pass
              elif step == "testSecret":
                  # 新パスワードで接続テスト
                  pass
              elif step == "finishSecret":
                  # AWSCURRENT ラベルを移動
                  client.update_secret_version_stage(
                      SecretId=secret_id,
                      VersionStage="AWSCURRENT",
                      MoveToVersionId=token
                  )
```

### 2.4 ローテーションの流れ

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

# 階層ごとの一括取得
aws ssm get-parameters-by-path \
  --path "/myapp/prod/database" \
  --with-decryption \
  --recursive
```

### 3.2 ECS タスク定義での参照

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
  --tags TagKey=Environment,TagValue=Production

# エイリアス設定
aws kms create-alias \
  --alias-name alias/myapp-key \
  --target-key-id "arn:aws:kms:ap-northeast-1:123456789012:key/xxxx-xxxx"
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

---

## 5. サービス比較

### 5.1 Secrets Manager vs Parameter Store

| 機能 | Secrets Manager | Parameter Store (Standard) | Parameter Store (Advanced) |
|------|----------------|---------------------------|---------------------------|
| **料金** | $0.40/シークレット/月 + API料 | 無料 | $0.05/パラメータ/月 |
| **最大サイズ** | 64 KB | 4 KB | 8 KB |
| **自動ローテーション** | 組み込みサポート | Lambda で自前実装 | Lambda で自前実装 |
| **クロスアカウント共有** | IAM ポリシーで可能 | 不可 | 不可 |
| **バージョニング** | あり（ステージラベル） | あり（番号のみ） | あり（番号のみ） |
| **暗号化** | KMS 必須 | SecureString で KMS | SecureString で KMS |
| **CloudFormation 動的参照** | `{{resolve:secretsmanager:...}}` | `{{resolve:ssm:...}}` | `{{resolve:ssm:...}}` |
| **推奨用途** | DB 認証情報, API キー | 設定値, Feature Flag | 大きめの設定値 |

### 5.2 KMS キータイプ比較

| キータイプ | 管理者 | コスト | ローテーション | ユースケース |
|-----------|--------|--------|---------------|-------------|
| **AWS マネージドキー** (`aws/xxx`) | AWS | 無料 | 自動（3年） | サービスデフォルト暗号化 |
| **カスタマーマネージドキー** | ユーザー | $1/月 + API料 | 手動 or 自動設定 | 細かいアクセス制御が必要 |
| **カスタマー提供キー** (BYOK) | ユーザー | API料のみ | ユーザー管理 | コンプライアンス要件 |
| **外部キーストア** | ユーザー | $1/月 + API料 | ユーザー管理 | HSM 連携, 規制対応 |

---

## 6. アンチパターン

### 6.1 環境変数にシークレットを平文で設定

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

### 6.2 全てのシークレットに同一 KMS キーを使う

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

---

## 7. FAQ

### Q1. Secrets Manager と Parameter Store SecureString、どちらを使うべき？

**A.** 自動ローテーションが必要なら Secrets Manager。コスト重視で手動管理可能なら Parameter Store SecureString。RDS/Redshift/DocumentDB のローテーションは Secrets Manager に組み込みテンプレートがあり、設定が容易。

### Q2. KMS の API コール料金が心配。キャッシュは可能？

**A.** Secrets Manager SDK にはクライアント側キャッシュライブラリがある。`aws-secretsmanager-caching` (Python) や `aws-secretsmanager-caching-java` を使えば TTL ベースでキャッシュされ、API コール数を大幅に削減できる。

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

---

## 8. まとめ

| 項目 | ポイント |
|------|---------|
| **Secrets Manager** | 自動ローテーション対応、RDS 統合、$0.40/月/シークレット |
| **Parameter Store** | 無料枠あり、階層構造、設定値と機密値の両方に対応 |
| **KMS** | エンベロープ暗号化の基盤、キーポリシーで細かいアクセス制御 |
| **設計原則** | シークレットはコードに含めない、最小権限、用途別にキーを分離 |
| **運用** | ローテーションスケジュール設定、CloudTrail で監査、キャッシュで API コスト削減 |

---

## 次に読むべきガイド

- [02-waf-shield.md](./02-waf-shield.md) — WAF/Shield によるアプリケーション保護
- IAM ベストプラクティスガイド — ロールとポリシーの設計パターン
- VPC エンドポイント設計 — プライベートネットワークからのサービスアクセス

---

## 参考文献

1. **AWS公式ドキュメント** — "AWS Secrets Manager User Guide" — https://docs.aws.amazon.com/secretsmanager/latest/userguide/
2. **AWS公式ドキュメント** — "AWS Systems Manager Parameter Store User Guide" — https://docs.aws.amazon.com/systems-manager/latest/userguide/systems-manager-parameter-store.html
3. **AWS公式ドキュメント** — "AWS Key Management Service Developer Guide" — https://docs.aws.amazon.com/kms/latest/developerguide/
4. **AWS Well-Architected Framework** — Security Pillar — "Protect data at rest" — https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/

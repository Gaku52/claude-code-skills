# Amazon API Gateway

> AWS のフルマネージド API サービスを理解し、REST API / HTTP API の構築・Lambda 統合・認証認可を実装する

## この章で学ぶこと

1. **API Gateway の基本概念** — REST API と HTTP API の違い、エンドポイントタイプの選択
2. **Lambda 統合とプロキシ統合** — サーバーレス API の構築パターン
3. **認証・認可の実装** — Cognito、IAM、Lambda オーソライザーの活用

---

## 1. API Gateway とは

API Gateway は、REST、HTTP、WebSocket API を作成・公開・管理するためのフルマネージドサービスである。バックエンドとして Lambda、EC2、任意の HTTP エンドポイントを統合できる。

### 図解 1: API Gateway のアーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway                              │
│                                                             │
│  Client ──→ [カスタムドメイン] ──→ [ステージ: prod]          │
│             api.example.com        │                        │
│                                    ▼                        │
│                              ┌──────────┐                   │
│                              │ API 定義 │                   │
│                              └────┬─────┘                   │
│                                   │                         │
│         ┌─────────────────────────┼──────────────────┐      │
│         ▼                         ▼                  ▼      │
│  ┌─────────────┐          ┌─────────────┐    ┌───────────┐ │
│  │ GET /users  │          │POST /users  │    │GET /health│ │
│  │             │          │             │    │           │ │
│  │ Lambda Fn   │          │ Lambda Fn   │    │ Mock      │ │
│  │ (list)      │          │ (create)    │    │ 統合      │ │
│  └─────────────┘          └─────────────┘    └───────────┘ │
│         │                         │                         │
│         ▼                         ▼                         │
│  ┌──────────────────────────────────────┐                   │
│  │        バックエンド                   │                   │
│  │  Lambda / EC2 / ECS / HTTP           │                   │
│  └──────────────────────────────────────┘                   │
│                                                             │
│  機能: スロットリング、キャッシュ、認証、                    │
│        CORS、WAF 統合、CloudWatch ログ                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. REST API vs HTTP API

### 比較表 1: REST API vs HTTP API

| 項目 | REST API | HTTP API |
|------|----------|----------|
| **プロトコル** | REST | HTTP (REST 互換) |
| **コスト** | $3.50/100 万リクエスト | $1.00/100 万リクエスト |
| **レイテンシ** | やや高い | 低い (最大 60% 削減) |
| **Lambda 統合** | プロキシ / 非プロキシ | プロキシのみ |
| **認証** | Cognito, IAM, Lambda Auth | Cognito, IAM, JWT |
| **API キー / 使用量プラン** | あり | なし |
| **キャッシュ** | あり | なし |
| **WAF 統合** | あり | なし |
| **リクエスト変換** | あり (VTL) | なし |
| **WebSocket** | あり (別タイプ) | なし |
| **推奨** | 高機能が必要な場合 | シンプルな API (推奨) |

---

## 3. API 構築

### コード例 1: AWS CLI で REST API を作成

```bash
# REST API の作成
aws apigateway create-rest-api \
  --name "MyApp-API" \
  --description "Production REST API" \
  --endpoint-configuration types=REGIONAL

# API ID を取得
API_ID="abc123def4"

# ルートリソース ID を取得
ROOT_ID=$(aws apigateway get-resources \
  --rest-api-id $API_ID \
  --query 'items[?path==`/`].id' \
  --output text)

# /users リソースの作成
aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $ROOT_ID \
  --path-part users

# GET /users メソッドの作成
aws apigateway put-method \
  --rest-api-id $API_ID \
  --resource-id res-users \
  --http-method GET \
  --authorization-type COGNITO_USER_POOLS \
  --authorizer-id auth-cognito

# Lambda プロキシ統合
aws apigateway put-integration \
  --rest-api-id $API_ID \
  --resource-id res-users \
  --http-method GET \
  --type AWS_PROXY \
  --integration-http-method POST \
  --uri "arn:aws:apigateway:ap-northeast-1:lambda:path/2015-03-31/functions/arn:aws:lambda:ap-northeast-1:123456789012:function:listUsers/invocations"

# デプロイ
aws apigateway create-deployment \
  --rest-api-id $API_ID \
  --stage-name prod \
  --stage-description "Production stage"
```

### コード例 2: SAM テンプレートでサーバーレス API を構築

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: Serverless API with SAM

Globals:
  Function:
    Runtime: python3.12
    Timeout: 30
    MemorySize: 256
    Environment:
      Variables:
        TABLE_NAME: !Ref UsersTable
        STAGE: !Ref Stage

Parameters:
  Stage:
    Type: String
    Default: prod

Resources:
  # HTTP API (推奨)
  HttpApi:
    Type: AWS::Serverless::HttpApi
    Properties:
      StageName: !Ref Stage
      CorsConfiguration:
        AllowOrigins:
          - "https://example.com"
        AllowMethods:
          - GET
          - POST
          - PUT
          - DELETE
        AllowHeaders:
          - Authorization
          - Content-Type
      Auth:
        DefaultAuthorizer: CognitoAuthorizer
        Authorizers:
          CognitoAuthorizer:
            AuthorizationScopes:
              - email
            IdentitySource: $request.header.Authorization
            JwtConfiguration:
              issuer: !Sub "https://cognito-idp.ap-northeast-1.amazonaws.com/${UserPool}"
              audience:
                - !Ref UserPoolClient

  # Lambda 関数
  ListUsersFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: app.list_users
      CodeUri: src/
      Events:
        ListUsers:
          Type: HttpApi
          Properties:
            ApiId: !Ref HttpApi
            Path: /users
            Method: GET

  CreateUserFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: app.create_user
      CodeUri: src/
      Events:
        CreateUser:
          Type: HttpApi
          Properties:
            ApiId: !Ref HttpApi
            Path: /users
            Method: POST

  GetUserFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: app.get_user
      CodeUri: src/
      Events:
        GetUser:
          Type: HttpApi
          Properties:
            ApiId: !Ref HttpApi
            Path: /users/{userId}
            Method: GET

  # DynamoDB テーブル
  UsersTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Sub "${Stage}-Users"
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: PK
          AttributeType: S
        - AttributeName: SK
          AttributeType: S
      KeySchema:
        - AttributeName: PK
          KeyType: HASH
        - AttributeName: SK
          KeyType: RANGE

Outputs:
  ApiUrl:
    Value: !Sub "https://${HttpApi}.execute-api.ap-northeast-1.amazonaws.com/${Stage}"
```

### コード例 3: Lambda ハンドラーの実装

```python
# src/app.py
import json
import os
import boto3
from datetime import datetime
import uuid

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(os.environ["TABLE_NAME"])


def _response(status_code: int, body: dict) -> dict:
    """API Gateway プロキシ統合のレスポンス形式"""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "https://example.com",
        },
        "body": json.dumps(body, default=str),
    }


def list_users(event, context):
    """GET /users"""
    try:
        # クエリパラメータ
        params = event.get("queryStringParameters") or {}
        limit = int(params.get("limit", 20))

        resp = table.scan(Limit=limit)  # 本番では query を使用
        return _response(200, {
            "users": resp["Items"],
            "count": resp["Count"],
        })
    except Exception as e:
        return _response(500, {"error": str(e)})


def create_user(event, context):
    """POST /users"""
    try:
        body = json.loads(event.get("body", "{}"))
        user_id = str(uuid.uuid4())

        item = {
            "PK": f"USER#{user_id}",
            "SK": "PROFILE",
            "userId": user_id,
            "name": body["name"],
            "email": body["email"],
            "createdAt": datetime.utcnow().isoformat(),
        }
        table.put_item(Item=item)
        return _response(201, {"user": item})
    except KeyError as e:
        return _response(400, {"error": f"Missing field: {e}"})
    except Exception as e:
        return _response(500, {"error": str(e)})


def get_user(event, context):
    """GET /users/{userId}"""
    try:
        user_id = event["pathParameters"]["userId"]
        resp = table.get_item(
            Key={"PK": f"USER#{user_id}", "SK": "PROFILE"}
        )
        item = resp.get("Item")
        if not item:
            return _response(404, {"error": "User not found"})
        return _response(200, {"user": item})
    except Exception as e:
        return _response(500, {"error": str(e)})
```

---

## 4. 認証・認可

### 図解 2: 認証方式の比較

```
1. Cognito User Pools (JWT):
   Client ──→ Cognito (ログイン) ──→ JWT Token
   Client ──→ API Gateway ──→ JWT 検証 ──→ Lambda
                │
                └─ Authorization: Bearer <jwt>

2. IAM 認証:
   Client ──→ SigV4 署名 ──→ API Gateway ──→ IAM ポリシー検証 ──→ Lambda
                │
                └─ AWS SDK が自動署名

3. Lambda Authorizer:
   Client ──→ API Gateway ──→ Lambda Authorizer ──→ ポリシー生成
                │                    │
                │                    ├─ Token ベース (JWT/OAuth)
                │                    └─ Request ベース (Header/Query)
                │
                └─ キャッシュ可能 (TTL 設定)

4. API Key:
   Client ──→ API Gateway ──→ API Key 検証 ──→ 使用量プラン確認
                │
                └─ x-api-key: <key>
                   ※ 認証ではなくスロットリング/計測用
```

### コード例 4: Lambda Authorizer の実装

```python
# authorizer.py
import json
import jwt
import os
import boto3
from typing import Optional

# JWKS キャッシュ
_jwks_cache = None

def handler(event, context):
    """Lambda Authorizer (Token ベース)"""
    try:
        token = event.get("authorizationToken", "")
        if token.startswith("Bearer "):
            token = token[7:]

        # JWT を検証
        claims = verify_jwt(token)
        if not claims:
            raise Exception("Invalid token")

        # IAM ポリシーを生成
        policy = generate_policy(
            principal_id=claims["sub"],
            effect="Allow",
            resource=event["methodArn"],
            context={
                "userId": claims["sub"],
                "email": claims.get("email", ""),
                "role": claims.get("custom:role", "user"),
            },
        )
        return policy

    except Exception as e:
        print(f"Authorization failed: {e}")
        raise Exception("Unauthorized")


def verify_jwt(token: str) -> Optional[dict]:
    """JWT トークンを検証"""
    try:
        # Cognito の JWKS URL
        issuer = os.environ["TOKEN_ISSUER"]
        audience = os.environ["TOKEN_AUDIENCE"]

        claims = jwt.decode(
            token,
            options={"verify_signature": True},
            algorithms=["RS256"],
            issuer=issuer,
            audience=audience,
        )
        return claims
    except jwt.InvalidTokenError:
        return None


def generate_policy(
    principal_id: str,
    effect: str,
    resource: str,
    context: dict = None,
) -> dict:
    """IAM ポリシードキュメントを生成"""
    # ARN からワイルドカードリソースを生成
    arn_parts = resource.split(":")
    api_gateway_arn = ":".join(arn_parts[:5])
    api_id_stage = arn_parts[5].split("/")
    resource_arn = f"{api_gateway_arn}:{api_id_stage[0]}/{api_id_stage[1]}/*"

    policy = {
        "principalId": principal_id,
        "policyDocument": {
            "Version": "2012-10-17",
            "Statement": [{
                "Action": "execute-api:Invoke",
                "Effect": effect,
                "Resource": resource_arn,
            }],
        },
    }

    if context:
        policy["context"] = context

    return policy
```

---

## 5. カスタムドメインと CORS

### コード例 5: カスタムドメインの設定

```bash
# ACM 証明書の取得（us-east-1 が必要な場合もある）
aws acm request-certificate \
  --domain-name "api.example.com" \
  --validation-method DNS \
  --region ap-northeast-1

# カスタムドメインの作成
aws apigatewayv2 create-domain-name \
  --domain-name "api.example.com" \
  --domain-name-configurations \
    CertificateArn=arn:aws:acm:ap-northeast-1:123456789012:certificate/xxx,EndpointType=REGIONAL

# API マッピングの作成
aws apigatewayv2 create-api-mapping \
  --domain-name "api.example.com" \
  --api-id abc123 \
  --stage prod

# Route 53 に Alias レコードを追加
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "api.example.com",
        "Type": "A",
        "AliasTarget": {
          "HostedZoneId": "Z2FDTNDATAQYW2",
          "DNSName": "d-xxx.execute-api.ap-northeast-1.amazonaws.com",
          "EvaluateTargetHealth": false
        }
      }
    }]
  }'
```

### 図解 3: API Gateway のステージとデプロイメント

```
API Gateway API
  │
  ├─ Stage: dev
  │   ├─ URL: https://abc123.execute-api.ap-ne-1.amazonaws.com/dev
  │   ├─ Stage Variables: {TABLE: "dev-Users", LOG_LEVEL: "DEBUG"}
  │   └─ Deployment: deploy-001
  │
  ├─ Stage: staging
  │   ├─ URL: https://abc123.execute-api.ap-ne-1.amazonaws.com/staging
  │   ├─ Stage Variables: {TABLE: "stg-Users", LOG_LEVEL: "INFO"}
  │   └─ Deployment: deploy-002
  │
  └─ Stage: prod
      ├─ URL: https://abc123.execute-api.ap-ne-1.amazonaws.com/prod
      │   → カスタムドメイン: api.example.com
      ├─ Stage Variables: {TABLE: "prod-Users", LOG_LEVEL: "WARN"}
      ├─ Deployment: deploy-003
      ├─ Canary: 10% → deploy-004 (新バージョン)
      ├─ Throttle: 10,000 req/s (バースト: 5,000)
      └─ Cache: 0.5 GB, TTL 300s
```

### 比較表 2: エンドポイントタイプ

| 項目 | Regional | Edge-Optimized | Private |
|------|----------|----------------|---------|
| **配置** | リージョン内 | CloudFront 経由 | VPC 内 |
| **レイテンシ** | リージョン近接で最小 | グローバル最適化 | VPC 内で最小 |
| **カスタムドメイン** | ACM (同一リージョン) | ACM (us-east-1) | なし |
| **WAF** | 直接アタッチ | CloudFront 経由 | なし |
| **推奨** | 単一リージョン API | グローバル API | 内部 API |

---

## 6. アンチパターン

### アンチパターン 1: Lambda のコールドスタートを無視する

```
[悪い例]
  API Gateway → Lambda (VPC 内、メモリ 128MB)
  → コールドスタート: 5-10 秒
  → API タイムアウト (29 秒) に近づく

[良い例]
  対策 1: Provisioned Concurrency
    aws lambda put-provisioned-concurrency-config \
      --function-name myFunction \
      --qualifier prod \
      --provisioned-concurrent-executions 10

  対策 2: メモリを増やす（CPU も比例して増加）
    MemorySize: 1024  # 128MB → 1024MB

  対策 3: VPC 外に配置（可能な場合）
    → VPC Lambda の ENI 作成時間を回避

  対策 4: SnapStart（Java の場合）
    SnapStart:
      ApplyOn: PublishedVersions
```

### アンチパターン 2: 全てを 1 つの Lambda 関数に集約

```
[悪い例]
  API Gateway → 1 つの Lambda (全エンドポイント処理)
  → デプロイが全エンドポイントに影響
  → メモリ/タイムアウトが最大公約数
  → 権限が過剰 (全リソースへのアクセス)

  def handler(event, context):
      path = event["path"]
      method = event["httpMethod"]
      if path == "/users" and method == "GET":
          return list_users()
      elif path == "/users" and method == "POST":
          return create_user()
      elif path.startswith("/orders"):
          return handle_orders()
      # ... 数十のルーティング

[良い例]
  API Gateway → 個別の Lambda 関数
  GET  /users  → listUsersFunction  (128MB, 5s timeout)
  POST /users  → createUserFunction (256MB, 10s timeout)
  GET  /orders → listOrdersFunction (512MB, 30s timeout)

  メリット:
  - 個別のメモリ/タイムアウト設定
  - 最小権限の IAM ロール
  - 個別のデプロイとロールバック
  - 個別のメトリクスと監視
```

---

## 7. FAQ

### Q1: REST API と HTTP API のどちらを選ぶべきですか？

**A:** 新規プロジェクトでは HTTP API を推奨する。コストが 70% 安く、レイテンシも低い。REST API は API キー管理、使用量プラン、リクエスト変換 (VTL)、キャッシュ、WAF 直接統合が必要な場合に選択する。既存の REST API を HTTP API に移行することも可能。

### Q2: API Gateway のレート制限はどう設定しますか？

**A:** REST API ではデフォルトで 10,000 req/s（バースト 5,000）。使用量プランと API キーで個別のスロットリングが可能。HTTP API ではルートごとにスロットリングを設定する。Lambda の同時実行数制限も考慮し、API Gateway のスロットルと Lambda の Reserved Concurrency を合わせて設計する。

### Q3: WebSocket API はどのような場面で使いますか？

**A:** リアルタイム双方向通信が必要な場合に使用する。チャットアプリ、ライブダッシュボード、IoT デバイス通信、オンラインゲームなどが典型例。WebSocket API は $connect、$disconnect、$default のルートと、カスタムルートを定義して Lambda で処理する。接続管理には DynamoDB を使い、@connections API でサーバーからメッセージをプッシュする。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| API タイプ | 新規は HTTP API 推奨。高機能が必要なら REST API |
| 統合タイプ | Lambda プロキシ統合が最もシンプル |
| 認証 | Cognito JWT が標準。カスタムロジックは Lambda Authorizer |
| カスタムドメイン | ACM 証明書 + Route 53 Alias で設定 |
| ステージ | dev/staging/prod で環境分離。Stage Variables で設定切替 |
| 監視 | CloudWatch Logs + X-Ray でリクエスト追跡 |
| コスト | HTTP API は $1/100 万リクエスト。REST API は $3.5 |

---

## 次に読むべきガイド

- [01-route53.md](./01-route53.md) — API Gateway のカスタムドメイン設定
- [00-iam-deep-dive.md](../08-security/00-iam-deep-dive.md) — API Gateway の IAM 認証
- [02-codepipeline.md](../07-devops/02-codepipeline.md) — API のCI/CD パイプライン

---

## 参考文献

1. **AWS 公式ドキュメント** — Amazon API Gateway 開発者ガイド
   https://docs.aws.amazon.com/apigateway/latest/developerguide/
2. **AWS SAM ドキュメント** — サーバーレスアプリケーションモデル
   https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/
3. **HTTP API vs REST API** — 選定ガイド
   https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-vs-rest.html

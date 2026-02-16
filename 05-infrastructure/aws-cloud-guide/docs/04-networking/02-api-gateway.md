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

## 6. WebSocket API

### 図解 4: WebSocket API のアーキテクチャ

```
WebSocket API のルーティング:
==============================

Client ──→ WebSocket API ──→ Route Selection
               │
    ┌──────────┼──────────────────────┐
    │          │                      │
    ▼          ▼                      ▼
 $connect   $default              $disconnect
 (Lambda)   (Lambda)              (Lambda)
    │                                 │
    ▼                                 ▼
 DynamoDB                          DynamoDB
 (接続管理)                       (接続削除)

カスタムルート:
  sendMessage → Lambda → @connections API → 他クライアントに送信
  joinRoom    → Lambda → DynamoDB (ルーム管理)
  typing      → Lambda → @connections API → タイピング通知
```

### コード例 5b: WebSocket API の Lambda ハンドラー

```python
# websocket_handler.py
import json
import os
import boto3
from datetime import datetime, timezone

dynamodb = boto3.resource("dynamodb")
connections_table = dynamodb.Table(os.environ["CONNECTIONS_TABLE"])
api_gateway = boto3.client("apigatewaymanagementapi",
    endpoint_url=os.environ["WEBSOCKET_ENDPOINT"])


def connect_handler(event, context):
    """$connect ルート: WebSocket 接続時"""
    connection_id = event["requestContext"]["connectionId"]
    user_id = event.get("queryStringParameters", {}).get("userId", "anonymous")

    connections_table.put_item(Item={
        "connectionId": connection_id,
        "userId": user_id,
        "connectedAt": datetime.now(timezone.utc).isoformat(),
        "ttl": int(datetime.now(timezone.utc).timestamp()) + 86400,
    })

    return {"statusCode": 200}


def disconnect_handler(event, context):
    """$disconnect ルート: WebSocket 切断時"""
    connection_id = event["requestContext"]["connectionId"]
    connections_table.delete_item(Key={"connectionId": connection_id})
    return {"statusCode": 200}


def send_message_handler(event, context):
    """sendMessage カスタムルート: メッセージの送信"""
    connection_id = event["requestContext"]["connectionId"]
    body = json.loads(event.get("body", "{}"))
    message = body.get("message", "")
    room_id = body.get("roomId", "general")

    # ルーム内の全接続を取得
    response = connections_table.scan(
        FilterExpression="roomId = :room",
        ExpressionAttributeValues={":room": room_id},
    )

    # 各接続にメッセージを送信
    payload = json.dumps({
        "action": "message",
        "message": message,
        "senderId": connection_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }).encode("utf-8")

    stale_connections = []
    for item in response["Items"]:
        target_id = item["connectionId"]
        try:
            api_gateway.post_to_connection(
                ConnectionId=target_id,
                Data=payload,
            )
        except api_gateway.exceptions.GoneException:
            stale_connections.append(target_id)

    # 切断済み接続を削除
    for conn_id in stale_connections:
        connections_table.delete_item(Key={"connectionId": conn_id})

    return {"statusCode": 200}
```

### コード例 5c: WebSocket API の SAM テンプレート

```yaml
Resources:
  WebSocketApi:
    Type: AWS::ApiGatewayV2::Api
    Properties:
      Name: ChatWebSocket
      ProtocolType: WEBSOCKET
      RouteSelectionExpression: "$request.body.action"

  ConnectRoute:
    Type: AWS::ApiGatewayV2::Route
    Properties:
      ApiId: !Ref WebSocketApi
      RouteKey: $connect
      AuthorizationType: NONE
      Target: !Sub "integrations/${ConnectIntegration}"

  DisconnectRoute:
    Type: AWS::ApiGatewayV2::Route
    Properties:
      ApiId: !Ref WebSocketApi
      RouteKey: $disconnect
      Target: !Sub "integrations/${DisconnectIntegration}"

  SendMessageRoute:
    Type: AWS::ApiGatewayV2::Route
    Properties:
      ApiId: !Ref WebSocketApi
      RouteKey: sendMessage
      Target: !Sub "integrations/${SendMessageIntegration}"

  Stage:
    Type: AWS::ApiGatewayV2::Stage
    Properties:
      ApiId: !Ref WebSocketApi
      StageName: prod
      AutoDeploy: true

Outputs:
  WebSocketUrl:
    Value: !Sub "wss://${WebSocketApi}.execute-api.${AWS::Region}.amazonaws.com/prod"
  ConnectionsUrl:
    Value: !Sub "https://${WebSocketApi}.execute-api.${AWS::Region}.amazonaws.com/prod"
```

---

## 7. スロットリングとキャッシュ

### REST API のキャッシュ設定

```bash
# ステージのキャッシュを有効化
aws apigateway update-stage \
  --rest-api-id abc123 \
  --stage-name prod \
  --patch-operations \
    op=replace,path=/cacheClusterEnabled,value=true \
    op=replace,path=/cacheClusterSize,value=0.5

# メソッドレベルのキャッシュ設定
aws apigateway update-method \
  --rest-api-id abc123 \
  --resource-id res-users \
  --http-method GET \
  --patch-operations \
    op=replace,path=/cacheKeyParameters/method.request.querystring.page,value=true \
    op=replace,path=/cacheTtlInSeconds,value=300
```

### スロットリングの設定

```
スロットリングの階層:
====================

1. アカウントレベル (デフォルト)
   - 10,000 req/s (リージョンあたり)
   - バースト: 5,000

2. ステージレベル
   api.example.com/prod → 5,000 req/s

3. ルートレベル（REST API）
   GET /users → 1,000 req/s
   POST /orders → 500 req/s

4. 使用量プラン + API キー（REST API のみ）
   Free プラン: 100 req/日, 10 req/s
   Pro プラン: 10,000 req/日, 100 req/s
   Enterprise プラン: 100,000 req/日, 1,000 req/s
```

```bash
# 使用量プランの作成（REST API）
aws apigateway create-usage-plan \
  --name "Free" \
  --description "Free tier usage plan" \
  --api-stages apiId=abc123,stage=prod \
  --throttle burstLimit=10,rateLimit=10 \
  --quota limit=100,period=DAY

# API キーの作成
aws apigateway create-api-key \
  --name "customer-001" \
  --enabled

# API キーを使用量プランに関連付け
aws apigateway create-usage-plan-key \
  --usage-plan-id plan-001 \
  --key-id key-001 \
  --key-type API_KEY

# HTTP API のルートレベルスロットリング
aws apigatewayv2 update-stage \
  --api-id http-abc123 \
  --stage-name prod \
  --route-settings '{
    "GET /users": {
      "ThrottlingBurstLimit": 100,
      "ThrottlingRateLimit": 50
    },
    "POST /orders": {
      "ThrottlingBurstLimit": 50,
      "ThrottlingRateLimit": 20
    }
  }'
```

---

## 8. 監視とログ

### CloudWatch ログの設定

```bash
# REST API のアクセスログ設定
aws apigateway update-stage \
  --rest-api-id abc123 \
  --stage-name prod \
  --patch-operations \
    op=replace,path=/accessLogSettings/destinationArn,value="arn:aws:logs:ap-northeast-1:123456789012:log-group:/api-gateway/prod" \
    op=replace,path=/accessLogSettings/format,value='{"requestId":"$context.requestId","ip":"$context.identity.sourceIp","caller":"$context.identity.caller","user":"$context.identity.user","requestTime":"$context.requestTime","httpMethod":"$context.httpMethod","resourcePath":"$context.resourcePath","status":"$context.status","protocol":"$context.protocol","responseLength":"$context.responseLength","integrationLatency":"$context.integrationLatency"}'

# HTTP API のアクセスログ設定
aws apigatewayv2 update-stage \
  --api-id http-abc123 \
  --stage-name prod \
  --access-log-settings '{
    "DestinationArn": "arn:aws:logs:ap-northeast-1:123456789012:log-group:/api-gateway/http/prod",
    "Format": "{\"requestId\":\"$context.requestId\",\"ip\":\"$context.identity.sourceIp\",\"requestTime\":\"$context.requestTime\",\"httpMethod\":\"$context.httpMethod\",\"path\":\"$context.path\",\"status\":\"$context.status\",\"latency\":\"$context.responseLatency\",\"integrationLatency\":\"$context.integrationLatency\"}"
  }'

# X-Ray トレーシングの有効化
aws apigateway update-stage \
  --rest-api-id abc123 \
  --stage-name prod \
  --patch-operations \
    op=replace,path=/tracingEnabled,value=true
```

### 主要メトリクス

| メトリクス | 説明 | アラーム閾値 |
|---|---|---|
| Count | リクエスト数 | 異常な急増/急減 |
| 4XXError | クライアントエラー率 | > 5% |
| 5XXError | サーバーエラー率 | > 1% |
| Latency | エンドツーエンドレイテンシ | p99 > 3秒 |
| IntegrationLatency | バックエンドレイテンシ | p99 > 2秒 |
| CacheHitCount | キャッシュヒット数 | 監視（ヒット率計算用） |
| CacheMissCount | キャッシュミス数 | 監視（ヒット率計算用） |

### CloudWatch アラームの設定

```bash
# 5xx エラー率アラーム
aws cloudwatch put-metric-alarm \
  --alarm-name "APIGateway-5xx-Error" \
  --alarm-description "API Gateway 5xx error rate exceeds 1%" \
  --metric-name 5XXError \
  --namespace AWS/ApiGateway \
  --statistic Average \
  --period 300 \
  --threshold 0.01 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 3 \
  --dimensions Name=ApiName,Value=MyApp-API \
  --alarm-actions arn:aws:sns:ap-northeast-1:123456789012:alerts

# レイテンシアラーム
aws cloudwatch put-metric-alarm \
  --alarm-name "APIGateway-Latency" \
  --alarm-description "API Gateway p99 latency exceeds 3 seconds" \
  --metric-name Latency \
  --namespace AWS/ApiGateway \
  --extended-statistic p99 \
  --period 300 \
  --threshold 3000 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 3 \
  --dimensions Name=ApiName,Value=MyApp-API \
  --alarm-actions arn:aws:sns:ap-northeast-1:123456789012:alerts
```

---

## 9. Terraform による API Gateway 構成

```hcl
# HTTP API
resource "aws_apigatewayv2_api" "http" {
  name          = "my-http-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["https://example.com"]
    allow_methods = ["GET", "POST", "PUT", "DELETE"]
    allow_headers = ["Authorization", "Content-Type"]
    max_age       = 3600
  }
}

resource "aws_apigatewayv2_stage" "prod" {
  api_id      = aws_apigatewayv2_api.http.id
  name        = "prod"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_logs.arn
    format = jsonencode({
      requestId        = "$context.requestId"
      ip               = "$context.identity.sourceIp"
      requestTime      = "$context.requestTime"
      httpMethod       = "$context.httpMethod"
      path             = "$context.path"
      status           = "$context.status"
      responseLatency  = "$context.responseLatency"
    })
  }

  default_route_settings {
    throttling_burst_limit = 1000
    throttling_rate_limit  = 500
  }
}

# Lambda 統合
resource "aws_apigatewayv2_integration" "list_users" {
  api_id             = aws_apigatewayv2_api.http.id
  integration_type   = "AWS_PROXY"
  integration_uri    = aws_lambda_function.list_users.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "list_users" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "GET /users"
  target    = "integrations/${aws_apigatewayv2_integration.list_users.id}"

  authorization_type = "JWT"
  authorizer_id      = aws_apigatewayv2_authorizer.cognito.id
}

# JWT Authorizer (Cognito)
resource "aws_apigatewayv2_authorizer" "cognito" {
  api_id           = aws_apigatewayv2_api.http.id
  authorizer_type  = "JWT"
  identity_sources = ["$request.header.Authorization"]
  name             = "cognito-authorizer"

  jwt_configuration {
    audience = [aws_cognito_user_pool_client.app.id]
    issuer   = "https://${aws_cognito_user_pool.main.endpoint}"
  }
}

# カスタムドメイン
resource "aws_apigatewayv2_domain_name" "api" {
  domain_name = "api.example.com"

  domain_name_configuration {
    certificate_arn = aws_acm_certificate.api.arn
    endpoint_type   = "REGIONAL"
    security_policy = "TLS_1_2"
  }
}

resource "aws_apigatewayv2_api_mapping" "api" {
  api_id      = aws_apigatewayv2_api.http.id
  domain_name = aws_apigatewayv2_domain_name.api.id
  stage       = aws_apigatewayv2_stage.prod.id
}

# Route 53 Alias
resource "aws_route53_record" "api" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "api.example.com"
  type    = "A"

  alias {
    name                   = aws_apigatewayv2_domain_name.api.domain_name_configuration[0].target_domain_name
    zone_id                = aws_apigatewayv2_domain_name.api.domain_name_configuration[0].hosted_zone_id
    evaluate_target_health = false
  }
}
```

---

## 10. アンチパターン

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

## 11. リクエストバリデーション

REST API では、バックエンド（Lambda）に到達する前にリクエストの検証が可能。不正なリクエストを早期に排除することで、Lambda 呼び出し回数を削減しコストを抑える。

### コード例 10: リクエストモデルとバリデーターの定義

```yaml
# SAM テンプレート: リクエストモデルとバリデーター
Resources:
  RestApi:
    Type: AWS::ApiGateway::RestApi
    Properties:
      Name: validated-api

  # リクエストモデル (JSON Schema)
  CreateUserModel:
    Type: AWS::ApiGateway::Model
    Properties:
      RestApiId: !Ref RestApi
      ContentType: application/json
      Name: CreateUserModel
      Schema:
        $schema: "http://json-schema.org/draft-04/schema#"
        title: CreateUserRequest
        type: object
        required:
          - email
          - name
        properties:
          email:
            type: string
            format: email
            pattern: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
          name:
            type: string
            minLength: 1
            maxLength: 100
          age:
            type: integer
            minimum: 0
            maximum: 150
          role:
            type: string
            enum:
              - admin
              - editor
              - viewer

  # バリデーター
  RequestValidator:
    Type: AWS::ApiGateway::RequestValidator
    Properties:
      RestApiId: !Ref RestApi
      Name: body-and-params-validator
      ValidateRequestBody: true
      ValidateRequestParameters: true

  # メソッドにバリデーターとモデルを適用
  CreateUserMethod:
    Type: AWS::ApiGateway::Method
    Properties:
      RestApiId: !Ref RestApi
      ResourceId: !Ref UsersResource
      HttpMethod: POST
      AuthorizationType: COGNITO_USER_POOLS
      RequestValidatorId: !Ref RequestValidator
      RequestModels:
        application/json: !Ref CreateUserModel
      RequestParameters:
        method.request.header.Authorization: true
        method.request.querystring.tenant: true
      Integration:
        Type: AWS_PROXY
        IntegrationHttpMethod: POST
        Uri: !Sub "arn:aws:apigateway:${AWS::Region}:lambda:path/2015-03-31/functions/${CreateUserFunction.Arn}/invocations"
```

### コード例 11: HTTP API のパラメータバリデーション (OpenAPI)

```yaml
# openapi.yaml — HTTP API 用
openapi: "3.0.1"
info:
  title: "User API"
  version: "1.0"
paths:
  /users/{userId}:
    get:
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
            pattern: "^usr_[a-zA-Z0-9]{12}$"
        - name: fields
          in: query
          required: false
          schema:
            type: string
            enum: [basic, full, minimal]
      x-amazon-apigateway-integration:
        type: aws_proxy
        httpMethod: POST
        uri: "arn:aws:apigateway:ap-northeast-1:lambda:path/2015-03-31/functions/${GetUserFn}/invocations"
        payloadFormatVersion: "2.0"
```

バリデーションエラー時は 400 Bad Request が自動返却される。

```json
{
  "message": "Invalid request body",
  "errors": [
    {
      "path": "/email",
      "message": "string does not match pattern"
    }
  ]
}
```

---

## 12. WAF 統合

REST API は AWS WAF を直接アタッチでき、SQL インジェクション、XSS、ボット対策などを API レベルで適用できる。

### コード例 12: WAF WebACL の作成と API Gateway へのアタッチ

```bash
# WAF WebACL の作成
aws wafv2 create-web-acl \
  --name "api-gateway-waf" \
  --scope REGIONAL \
  --default-action Allow={} \
  --rules '[
    {
      "Name": "AWSManagedRulesCommonRuleSet",
      "Priority": 1,
      "Statement": {
        "ManagedRuleGroupStatement": {
          "VendorName": "AWS",
          "Name": "AWSManagedRulesCommonRuleSet"
        }
      },
      "OverrideAction": {"None": {}},
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "CommonRuleSet"
      }
    },
    {
      "Name": "AWSManagedRulesSQLiRuleSet",
      "Priority": 2,
      "Statement": {
        "ManagedRuleGroupStatement": {
          "VendorName": "AWS",
          "Name": "AWSManagedRulesSQLiRuleSet"
        }
      },
      "OverrideAction": {"None": {}},
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "SQLiRuleSet"
      }
    },
    {
      "Name": "RateLimit",
      "Priority": 3,
      "Statement": {
        "RateBasedStatement": {
          "Limit": 2000,
          "AggregateKeyType": "IP"
        }
      },
      "Action": {"Block": {}},
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "RateLimit"
      }
    }
  ]' \
  --visibility-config \
    SampledRequestsEnabled=true,CloudWatchMetricsEnabled=true,MetricName=api-gateway-waf

# WAF を API Gateway ステージにアタッチ
aws wafv2 associate-web-acl \
  --web-acl-arn arn:aws:wafv2:ap-northeast-1:123456789012:regional/webacl/api-gateway-waf/xxx \
  --resource-arn arn:aws:apigateway:ap-northeast-1::/restapis/abc123/stages/prod
```

### コード例 13: Terraform WAF + API Gateway

```hcl
# WAF WebACL
resource "aws_wafv2_web_acl" "api" {
  name  = "api-gateway-waf"
  scope = "REGIONAL"

  default_action {
    allow {}
  }

  # AWS マネージドルール: Common Rule Set
  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 1

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesCommonRuleSet"
      }
    }

    visibility_config {
      sampled_requests_enabled   = true
      cloudwatch_metrics_enabled = true
      metric_name                = "CommonRuleSet"
    }
  }

  # IP ベースレート制限
  rule {
    name     = "RateLimit"
    priority = 10

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      sampled_requests_enabled   = true
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimit"
    }
  }

  # Geo ブロック（特定国からのアクセスを拒否）
  rule {
    name     = "GeoBlock"
    priority = 20

    action {
      block {}
    }

    statement {
      geo_match_statement {
        country_codes = ["CN", "RU"]
      }
    }

    visibility_config {
      sampled_requests_enabled   = true
      cloudwatch_metrics_enabled = true
      metric_name                = "GeoBlock"
    }
  }

  visibility_config {
    sampled_requests_enabled   = true
    cloudwatch_metrics_enabled = true
    metric_name                = "api-gateway-waf"
  }
}

# WAF と REST API ステージの関連付け
resource "aws_wafv2_web_acl_association" "api" {
  resource_arn = aws_api_gateway_stage.prod.arn
  web_acl_arn  = aws_wafv2_web_acl.api.arn
}
```

### 図解 5: WAF による多層防御

```
クライアント
    │
    ▼
┌──────────────────────────────────────────────┐
│  AWS WAF                                      │
│  ┌────────────────────────────────────────┐   │
│  │ Rule 1: AWS Managed Common Rules       │   │
│  │  - XSS 検知 → Block                   │   │
│  │  - サイズ制限超過 → Block              │   │
│  ├────────────────────────────────────────┤   │
│  │ Rule 2: SQLi Rule Set                  │   │
│  │  - SQL インジェクション → Block        │   │
│  ├────────────────────────────────────────┤   │
│  │ Rule 3: Rate Limit (2000 req/5min/IP)  │   │
│  │  - 超過 → Block (429)                  │   │
│  ├────────────────────────────────────────┤   │
│  │ Rule 4: Geo Block                      │   │
│  │  - 特定国 → Block                      │   │
│  └────────────────────────────────────────┘   │
│  Default Action: Allow                        │
└──────────────────────────┬───────────────────┘
                           │ 通過
                           ▼
               ┌──────────────────┐
               │  API Gateway     │
               │  (REST API)      │
               │  Throttling +    │
               │  Validation      │
               └────────┬─────────┘
                        │
                        ▼
               ┌──────────────────┐
               │  Lambda Backend  │
               └──────────────────┘
```

---

## 13. FAQ

### Q1: REST API と HTTP API のどちらを選ぶべきですか？

**A:** 新規プロジェクトでは HTTP API を推奨する。コストが 70% 安く、レイテンシも低い。REST API は API キー管理、使用量プラン、リクエスト変換 (VTL)、キャッシュ、WAF 直接統合が必要な場合に選択する。既存の REST API を HTTP API に移行することも可能。

### Q2: API Gateway のレート制限はどう設定しますか？

**A:** REST API ではデフォルトで 10,000 req/s（バースト 5,000）。使用量プランと API キーで個別のスロットリングが可能。HTTP API ではルートごとにスロットリングを設定する。Lambda の同時実行数制限も考慮し、API Gateway のスロットルと Lambda の Reserved Concurrency を合わせて設計する。

### Q3: WebSocket API はどのような場面で使いますか？

**A:** リアルタイム双方向通信が必要な場合に使用する。チャットアプリ、ライブダッシュボード、IoT デバイス通信、オンラインゲームなどが典型例。WebSocket API は $connect、$disconnect、$default のルートと、カスタムルートを定義して Lambda で処理する。接続管理には DynamoDB を使い、@connections API でサーバーからメッセージをプッシュする。

### Q4: CORS エラーの原因と対処法は？

**A:** CORS エラーの主な原因は以下の 3 つ。(1) API Gateway で CORS が未設定 — HTTP API では `CorsConfiguration` を、REST API では OPTIONS メソッドに Mock 統合を追加。(2) Lambda のレスポンスに CORS ヘッダーが欠落 — プロキシ統合では Lambda 側で `Access-Control-Allow-Origin` を返す必要がある。(3) Cognito 認証ヘッダーが `AllowHeaders` に含まれていない — `Authorization` ヘッダーを明示的に許可する。

```python
# Lambda プロキシ統合での CORS ヘッダー付与
def lambda_handler(event, context):
    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin": "https://example.com",
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type,Authorization,X-Api-Key",
            "Access-Control-Max-Age": "86400"
        },
        "body": json.dumps({"message": "success"})
    }
```

### Q5: API Gateway の 29 秒タイムアウトを回避する方法は？

**A:** API Gateway の統合タイムアウト上限は 29 秒（変更不可）。長時間処理には以下のパターンを採用する。

```
非同期処理パターン:
1. POST /jobs → Lambda が SQS にジョブ登録 → 即座に jobId を返却 (< 1秒)
2. バックエンド Lambda が SQS からジョブを取得して処理 (制限なし)
3. GET /jobs/{jobId} → 処理結果をポーリングで取得

Step Functions パターン:
1. POST /jobs → Step Functions 実行を開始 → executionArn を返却
2. Step Functions 内で長時間処理を実行
3. GET /jobs/{executionArn} → DescribeExecution で状態を取得
```

### Q6: API Gateway のコスト最適化のポイントは？

**A:** (1) HTTP API を選択（REST API の約 30% のコスト）。(2) REST API を使う場合はキャッシュを有効化してバックエンド呼び出しを削減。(3) 使用量プランで API キーごとにクォータを設定し、過剰利用を防止。(4) CloudFront を前段に配置してキャッシュヒット率を向上させる。(5) Lambda の Provisioned Concurrency とのバランスを取り、不要な事前ウォームアップを避ける。

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

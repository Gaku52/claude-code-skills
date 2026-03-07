# APIゲートウェイ

> APIゲートウェイはマイクロサービスの統一エントリポイントである。ルーティング、認証の一元化、レート制限、リクエスト変換、サーキットブレーカーまで、APIゲートウェイの設計・構築・運用を体系的に習得する。Kong、AWS API Gateway、Nginx、Envoy を中心に、プロダクション環境で求められる構成パターン、セキュリティ統合、サービスメッシュ連携を包括的に扱う。

---

## この章で学ぶこと

- [ ] APIゲートウェイの役割とアーキテクチャを理解する
- [ ] 主要なゲートウェイ製品（Kong、AWS API Gateway、Nginx、Envoy）を比較し選定できる
- [ ] BFF（Backend for Frontend）パターンを設計できる
- [ ] レート制限アルゴリズム（Token Bucket、Sliding Window）を実装できる
- [ ] 認証統合（JWT、OAuth 2.0、API Key）をゲートウェイ層で構成できる
- [ ] サーキットブレーカーとリトライ戦略を適切に設定できる
- [ ] サービスメッシュ（Istio + Envoy）との連携を理解する
- [ ] 本番環境におけるモニタリングとトラブルシューティングを実践できる

---

## 1. APIゲートウェイの役割とアーキテクチャ

### 1.1 なぜAPIゲートウェイが必要か

マイクロサービスアーキテクチャにおいて、クライアントが個々のサービスに直接通信する場合、以下の問題が生じる。

```
┌─────────────────────────────────────────────────────────┐
│ ゲートウェイなしの構成（問題パターン）                        │
│                                                         │
│  Browser ─── https://user.api.example.com/users         │
│          ├── https://order.api.example.com/orders        │
│          ├── https://payment.api.example.com/pay         │
│          └── https://notify.api.example.com/notifications│
│                                                         │
│  問題点:                                                 │
│   - クライアントが全サービスのエンドポイントを知る必要がある   │
│   - CORS設定がサービスごとに分散する                       │
│   - 認証ロジックが各サービスに重複する                      │
│   - レート制限を統一的に適用できない                        │
│   - サービスの追加・統合・分割がクライアントに影響する         │
│   - TLS証明書を各サービスで管理する必要がある                │
└─────────────────────────────────────────────────────────┘
```

APIゲートウェイはこれらの問題を解決する「正面玄関」として機能する。

```
┌─────────────────────────────────────────────────────────────┐
│ ゲートウェイありの構成（推奨パターン）                          │
│                                                             │
│           ┌──────────────────┐                               │
│  Browser ─┤  API Gateway     ├─── User Service (内部)        │
│  Mobile  ─┤  (単一URL)       ├─── Order Service (内部)       │
│  3rd App ─┤  api.example.com ├─── Payment Service (内部)     │
│           └──────────────────┘                               │
│                  │                                           │
│                  ├── TLS終端                                  │
│                  ├── 認証・認可                                │
│                  ├── レート制限                                │
│                  ├── ルーティング                              │
│                  ├── リクエスト/レスポンス変換                   │
│                  ├── ロードバランシング                         │
│                  ├── キャッシュ                                │
│                  ├── ログ・メトリクス                           │
│                  └── サーキットブレーカー                       │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 APIゲートウェイの8つの主要機能

```
APIゲートウェイ = マイクロサービスへの単一エントリポイント

  クライアント → API Gateway → User Service
                             → Order Service
                             → Payment Service
                             → Notification Service

主要機能:
  (1) ルーティング:
     → /users/* → User Service
     → /orders/* → Order Service
     → パスベース、ヘッダーベース、クエリパラメータベースのルーティング
     → バージョニング: /v1/* → Service v1, /v2/* → Service v2

  (2) 認証・認可の一元化:
     → JWT検証をゲートウェイで実施
     → 各サービスは認証済みリクエストのみ受信
     → API Key の検証
     → OAuth 2.0 トークンイントロスペクション
     → mTLS によるサービス間認証

  (3) レート制限:
     → グローバルなレート制限
     → クライアント/プラン別の制限
     → Token Bucket / Sliding Window アルゴリズム
     → DDoS防御の第一層

  (4) リクエスト/レスポンス変換:
     → ヘッダーの追加/削除
     → リクエストボディの変換（XML → JSON 等）
     → レスポンスの集約（API Composition）
     → GraphQL → REST 変換

  (5) ロードバランシング:
     → サービスインスタンス間の分散
     → ラウンドロビン、最小接続数、重み付き
     → ヘルスチェック（アクティブ/パッシブ）

  (6) キャッシュ:
     → レスポンスキャッシュ（TTL制御）
     → CDN統合（CloudFront, Fastly）
     → 条件付きリクエスト（ETag, Last-Modified）

  (7) 監視・ログ:
     → アクセスログ（構造化ログ）
     → メトリクス収集（レイテンシ、エラー率、スループット）
     → 分散トレーシング（OpenTelemetry, Jaeger）
     → アラート統合（PagerDuty, Slack）

  (8) サーキットブレーカー:
     → 障害サービスへのリクエストを遮断
     → フォールバックレスポンス
     → 障害の伝播を防止（カスケード障害の回避）
```

### 1.3 デプロイメントパターン

APIゲートウェイのデプロイには複数のパターンがある。各パターンの特性を理解して適切に選択する。

```
┌────────────────────────────────────────────────────────────────┐
│ パターン1: 集中型ゲートウェイ                                     │
│                                                                │
│   Client → [API Gateway] → Service A                           │
│                           → Service B                           │
│                           → Service C                           │
│                                                                │
│   特徴: 単一のゲートウェイが全トラフィックを処理                     │
│   利点: シンプル、一元管理                                        │
│   欠点: 単一障害点、チーム間のボトルネック                           │
├────────────────────────────────────────────────────────────────┤
│ パターン2: BFF（Backend for Frontend）                           │
│                                                                │
│   Web    → [Web BFF]    → Service A / B / C                    │
│   Mobile → [Mobile BFF] → Service A / B / C                    │
│   3rd    → [Public GW]  → Service A / B / C                    │
│                                                                │
│   特徴: クライアント種別ごとに専用ゲートウェイ                      │
│   利点: クライアント最適化、チーム独立                              │
│   欠点: 管理コスト増大、ロジック重複リスク                          │
├────────────────────────────────────────────────────────────────┤
│ パターン3: 2層ゲートウェイ                                        │
│                                                                │
│   Client → [Edge GW] → [Internal GW A] → Service A            │
│                       → [Internal GW B] → Service B            │
│                       → [Internal GW C] → Service C            │
│                                                                │
│   特徴: Edge層（外部向け）+ 内部層（ドメイン別）                    │
│   利点: 関心の分離、ドメインチームの自律性                          │
│   欠点: レイテンシ増加、構成の複雑化                               │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. ゲートウェイ製品の詳細比較

### 2.1 主要製品の機能比較

| 項目 | AWS API Gateway | Kong | Nginx (Plus) | Envoy | Traefik | Apigee |
|------|----------------|------|-------------|-------|---------|--------|
| タイプ | マネージド | OSS/商用 | OSS/商用 | OSS | OSS/商用 | マネージド |
| デプロイ | サーバーレス | セルフホスト/K8s | セルフホスト/K8s | サイドカー/K8s | セルフホスト/K8s | クラウド |
| プロトコル | HTTP, WebSocket | HTTP, gRPC, WebSocket | HTTP, TCP, UDP | HTTP, gRPC, TCP | HTTP, gRPC, TCP | HTTP, gRPC |
| プラグイン数 | Lambda連携中心 | 300+ (Hub) | モジュール式 | フィルタチェーン | ミドルウェア | ポリシー |
| 設定方式 | Console/CloudFormation/CDK | Admin API/declarative YAML | conf ファイル | xDS API / YAML | YAML / Label | Console/API |
| K8s 統合 | なし | Kong Ingress Controller | Ingress Controller | Istio / Gateway API | Ingress / CRD | Apigee Adapter |
| サービスメッシュ | なし | Kong Mesh (Kuma) | なし | Istio データプレーン | Traefik Mesh | Apigee Service Mesh |
| コスト | 従量課金 | OSS無料/Enterprise有料 | OSS無料/Plus有料 | 無料 | OSS無料/Enterprise有料 | 従量課金 |
| 学習コスト | 低 | 中 | 低 | 高 | 低〜中 | 中〜高 |
| 最適用途 | AWS環境 | 汎用 | 高性能リバースプロキシ | サービスメッシュ | コンテナ/動的環境 | エンタープライズAPI管理 |

### 2.2 パフォーマンス特性の比較

| 指標 | AWS API Gateway | Kong | Nginx | Envoy |
|------|----------------|------|-------|-------|
| レイテンシ（P99） | 10-30ms 追加 | 1-5ms 追加 | < 1ms 追加 | 1-3ms 追加 |
| スループット | 10,000 RPS (デフォルト) | 50,000+ RPS | 100,000+ RPS | 50,000+ RPS |
| メモリ使用量 | マネージド | 200-500MB | 50-100MB | 100-300MB |
| 水平スケーリング | 自動 | 手動/K8s HPA | 手動/K8s HPA | Istio で自動 |
| ウォームアップ | コールドスタートあり | 不要 | 不要 | 不要 |

### 2.3 選定フローチャート

```
                       ┌─────────────────┐
                       │ ゲートウェイ選定  │
                       └────────┬────────┘
                                │
                     ┌──────────▼──────────┐
                     │ AWS 環境のみ？        │
                     └─────┬─────────┬─────┘
                       Yes │         │ No
                           │         │
                   ┌───────▼───┐     │
                   │ AWS API GW │     │
                   └───────────┘     │
                              ┌──────▼──────────┐
                              │ K8s を使用？      │
                              └──┬──────────┬───┘
                             Yes │          │ No
                                 │          │
                        ┌────────▼────┐  ┌──▼──────────┐
                        │サービスメッシュ│  │ Nginx       │
                        │が必要？      │  │ (シンプル)   │
                        └──┬─────┬───┘  └─────────────┘
                       Yes │     │ No
                           │     │
                  ┌────────▼──┐ ┌▼───────────┐
                  │ Envoy +   │ │ Kong       │
                  │ Istio     │ │ (プラグイン)│
                  └───────────┘ └────────────┘
```

---

## 3. AWS API Gateway 詳細

### 3.1 APIタイプの選択

```
AWS API Gateway の3種類:

  (1) HTTP API（推奨・低コスト）:
     → REST API の 70% 安い料金
     → 低レイテンシ（REST APIより高速）
     → JWT Authorizer（ネイティブサポート）
     → Lambda プロキシ統合
     → CORS 自動設定
     → 制限: リクエスト/レスポンス変換なし、Usage Plan なし

  (2) REST API（全機能）:
     → リクエスト/レスポンス変換（VTL テンプレート）
     → APIキー + Usage Plan（API課金管理）
     → AWS WAF 統合
     → キャッシュ機能（0.5GB〜237GB）
     → リクエストバリデーション
     → カナリアリリース対応

  (3) WebSocket API:
     → 双方向リアルタイム通信
     → $connect / $disconnect / $default ルート
     → チャット、ゲーム、金融リアルタイムフィード
     → Lambda / DynamoDB バックエンド
     → Connection管理API（@connections）

選択基準:
  HTTP API ← シンプルなREST + JWT認証で十分な場合
  REST API ← Usage Plan、WAF、変換が必要な場合
  WebSocket API ← リアルタイム双方向通信が必要な場合
```

### 3.2 AWS SAM によるデプロイ

```yaml
# AWS SAM テンプレート（本番品質）
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: Production API Gateway with Authentication and Rate Limiting

Parameters:
  Environment:
    Type: String
    Default: production
    AllowedValues: [development, staging, production]
  JwtIssuer:
    Type: String
    Description: JWT token issuer URL
  JwtAudience:
    Type: String
    Description: JWT token audience

Globals:
  Function:
    Runtime: nodejs20.x
    Timeout: 30
    MemorySize: 256
    Environment:
      Variables:
        ENVIRONMENT: !Ref Environment
        LOG_LEVEL: !If [IsProd, "warn", "debug"]
    Tracing: Active
  Api:
    Cors:
      AllowOrigin: "'https://app.example.com'"
      AllowHeaders: "'Authorization,Content-Type,X-Request-ID'"
      AllowMethods: "'GET,POST,PUT,DELETE,OPTIONS'"
      MaxAge: "'86400'"

Conditions:
  IsProd: !Equals [!Ref Environment, production]

Resources:
  # HTTP API Gateway
  ApiGateway:
    Type: AWS::Serverless::HttpApi
    Properties:
      StageName: !Ref Environment
      Description: !Sub "API Gateway - ${Environment}"
      Auth:
        DefaultAuthorizer: JwtAuthorizer
        Authorizers:
          JwtAuthorizer:
            JwtConfiguration:
              issuer: !Ref JwtIssuer
              audience:
                - !Ref JwtAudience
      AccessLogSettings:
        DestinationArn: !GetAtt ApiAccessLogGroup.Arn
        Format: >-
          {"requestId":"$context.requestId",
           "ip":"$context.identity.sourceIp",
           "method":"$context.httpMethod",
           "path":"$context.path",
           "status":"$context.status",
           "latency":"$context.responseLatency",
           "userAgent":"$context.identity.userAgent",
           "integrationLatency":"$context.integrationLatency"}
      DefaultRouteSettings:
        ThrottlingBurstLimit: 200
        ThrottlingRateLimit: 100

  # CloudWatch Log Group
  ApiAccessLogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub "/aws/apigateway/${Environment}/access-logs"
      RetentionInDays: !If [IsProd, 90, 14]

  # Users API
  ListUsersFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: handlers/users.list
      Description: List all users with pagination
      Events:
        ListUsers:
          Type: HttpApi
          Properties:
            ApiId: !Ref ApiGateway
            Path: /users
            Method: GET

  GetUserFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: handlers/users.get
      Description: Get user by ID
      Events:
        GetUser:
          Type: HttpApi
          Properties:
            ApiId: !Ref ApiGateway
            Path: /users/{userId}
            Method: GET

  CreateUserFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: handlers/users.create
      Description: Create a new user
      Events:
        CreateUser:
          Type: HttpApi
          Properties:
            ApiId: !Ref ApiGateway
            Path: /users
            Method: POST

  # Health Check (認証なし)
  HealthCheckFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: handlers/health.check
      Description: Health check endpoint
      Events:
        HealthCheck:
          Type: HttpApi
          Properties:
            ApiId: !Ref ApiGateway
            Path: /health
            Method: GET
            Auth:
              Authorizer: NONE

Outputs:
  ApiEndpoint:
    Description: API Gateway endpoint URL
    Value: !Sub "https://${ApiGateway}.execute-api.${AWS::Region}.amazonaws.com/${Environment}"
  ApiId:
    Description: API Gateway ID
    Value: !Ref ApiGateway
```

### 3.3 カスタムドメインとステージ管理

```yaml
# カスタムドメイン設定
Resources:
  ApiDomainName:
    Type: AWS::ApiGatewayV2::DomainName
    Properties:
      DomainName: api.example.com
      DomainNameConfigurations:
        - CertificateArn: !Ref AcmCertificate
          EndpointType: REGIONAL
          SecurityPolicy: TLS_1_2

  ApiMapping:
    Type: AWS::ApiGatewayV2::ApiMapping
    Properties:
      DomainName: api.example.com
      ApiId: !Ref ApiGateway
      Stage: !Ref Environment
      ApiMappingKey: v1

  # Route 53 レコード
  DnsRecord:
    Type: AWS::Route53::RecordSet
    Properties:
      HostedZoneId: !Ref HostedZoneId
      Name: api.example.com
      Type: A
      AliasTarget:
        DNSName: !GetAtt ApiDomainName.RegionalDomainName
        HostedZoneId: !GetAtt ApiDomainName.RegionalHostedZoneId
```

### 3.4 AWS API Gateway の制限事項

```
AWS API Gateway の主要制限:

  HTTP API:
  ├── ペイロードサイズ: 10 MB
  ├── タイムアウト: 30 秒
  ├── リクエスト/秒: 10,000（デフォルト、引き上げ可能）
  ├── ルート数: 300
  ├── ステージ数: 10
  └── Authorizer: JWT のみ（Lambda Authorizer なし → REST API を検討）

  REST API:
  ├── ペイロードサイズ: 10 MB
  ├── タイムアウト: 29 秒
  ├── リクエスト/秒: 10,000（デフォルト、引き上げ可能）
  ├── リソース数: 300
  ├── ステージ数: 10
  ├── API Key 数: 10,000
  └── Usage Plan 数: 300

  WebSocket API:
  ├── メッセージサイズ: 128 KB（送信）/ 32 KB（受信フレーム）
  ├── 接続時間: 最大 2 時間
  ├── アイドルタイムアウト: 10 分
  └── 同時接続数: デフォルト制限あり

  共通の注意点:
  → コールドスタート: 初回リクエストでレイテンシが増加する場合がある
  → VPC Link: VPC内リソースへの接続にはVPC Linkが必要
  → バイナリデータ: Content-Type のマッピングが必要
  → CORS: HTTP API は自動、REST API は手動設定が必要
```

---

## 4. Kong Gateway 詳細

### 4.1 アーキテクチャ

Kong はコントロールプレーンとデータプレーンに分離されたアーキテクチャを持つ。

```
┌────────────────────────────────────────────────────────────┐
│ Kong アーキテクチャ                                          │
│                                                            │
│  ┌─────────────────────────────────────┐                   │
│  │ Control Plane                       │                   │
│  │  ┌─────────┐  ┌──────────────────┐  │                   │
│  │  │ Admin API│  │ Database         │  │                   │
│  │  │ :8001    │  │ (PostgreSQL)     │  │                   │
│  │  └─────────┘  └──────────────────┘  │                   │
│  └──────────────────┬──────────────────┘                   │
│                     │ 設定同期                              │
│  ┌──────────────────▼──────────────────┐                   │
│  │ Data Plane (Proxy)                  │                   │
│  │  ┌─────────┐  ┌──────────────────┐  │                   │
│  │  │ Proxy   │  │ Plugin Chain     │  │                   │
│  │  │ :8000   │  │ Auth → Rate Limit│  │                   │
│  │  │ :8443   │  │ → Log → Transform│  │                   │
│  │  └─────────┘  └──────────────────┘  │                   │
│  └─────────────────────────────────────┘                   │
│                                                            │
│  DB-less Mode: YAML設定のみでDB不要（推奨: K8s環境）          │
│  Hybrid Mode: CP/DP分離でセキュリティ強化                      │
└────────────────────────────────────────────────────────────┘
```

### 4.2 宣言的設定（DB-less モード）

```yaml
# kong.yml - 本番品質の宣言的設定
_format_version: "3.0"
_transform: true

# サービス定義
services:
  # ユーザーサービス
  - name: user-service
    url: http://user-service:3000
    connect_timeout: 5000
    read_timeout: 30000
    write_timeout: 30000
    retries: 3
    routes:
      - name: users-route
        paths:
          - /api/v1/users
        strip_path: false
        protocols:
          - https
        methods:
          - GET
          - POST
          - PUT
          - DELETE
        headers:
          x-api-version:
            - v1
    plugins:
      # JWT 認証
      - name: jwt
        config:
          secret_is_base64: false
          claims_to_verify:
            - exp
          header_names:
            - Authorization
          key_claim_name: iss
      # レート制限
      - name: rate-limiting
        config:
          minute: 100
          hour: 5000
          policy: redis
          redis:
            host: redis
            port: 6379
            timeout: 2000
          fault_tolerant: true
          hide_client_headers: false
      # CORS
      - name: cors
        config:
          origins:
            - https://app.example.com
            - https://staging.example.com
          methods:
            - GET
            - POST
            - PUT
            - DELETE
            - OPTIONS
          headers:
            - Authorization
            - Content-Type
            - X-Request-ID
          exposed_headers:
            - X-RateLimit-Remaining
            - X-RateLimit-Limit
          max_age: 86400
          credentials: true
      # リクエスト変換
      - name: request-transformer
        config:
          add:
            headers:
              - "X-Gateway-Version:kong-3.x"
              - "X-Forwarded-Service:user-service"
      # レスポンス変換
      - name: response-transformer
        config:
          remove:
            headers:
              - X-Powered-By
              - Server
          add:
            headers:
              - "X-Content-Type-Options:nosniff"
              - "X-Frame-Options:DENY"
              - "Strict-Transport-Security:max-age=31536000; includeSubDomains"

  # 注文サービス
  - name: order-service
    url: http://order-service:3000
    connect_timeout: 5000
    read_timeout: 60000
    write_timeout: 60000
    retries: 2
    routes:
      - name: orders-route
        paths:
          - /api/v1/orders
        strip_path: false
        protocols:
          - https
    plugins:
      - name: rate-limiting
        config:
          minute: 50
          hour: 2000
          policy: redis
          redis:
            host: redis
            port: 6379
      - name: request-size-limiting
        config:
          allowed_payload_size: 5
          size_unit: megabytes

  # ヘルスチェック（認証不要）
  - name: health-service
    url: http://health-aggregator:3000
    routes:
      - name: health-route
        paths:
          - /health
        methods:
          - GET

# グローバルプラグイン
plugins:
  # 全サービス共通のログ
  - name: tcp-log
    config:
      host: log-collector
      port: 5140
      tls: false
      keepalive: 60000
  # Prometheus メトリクス
  - name: prometheus
    config:
      per_consumer: true
      status_code_metrics: true
      latency_metrics: true
      bandwidth_metrics: true
  # Bot 検出
  - name: bot-detection
    config:
      deny:
        - "curl"
        - "wget"

# コンシューマー定義（API利用者）
consumers:
  - username: mobile-app
    keyauth_credentials:
      - key: mobile-app-key-xxxxx
    plugins:
      - name: rate-limiting
        config:
          minute: 200
          hour: 10000
          policy: redis
          redis:
            host: redis
            port: 6379

  - username: partner-api
    keyauth_credentials:
      - key: partner-key-yyyyy
    plugins:
      - name: rate-limiting
        config:
          minute: 50
          hour: 1000
          policy: redis
          redis:
            host: redis
            port: 6379

# アップストリーム定義（ロードバランシング）
upstreams:
  - name: user-service
    algorithm: round-robin
    healthchecks:
      active:
        type: http
        http_path: /health
        healthy:
          interval: 10
          successes: 3
        unhealthy:
          interval: 5
          http_failures: 3
          tcp_failures: 3
          timeouts: 3
      passive:
        type: http
        healthy:
          successes: 5
        unhealthy:
          http_failures: 5
          tcp_failures: 3
          timeouts: 3
    targets:
      - target: user-service-1:3000
        weight: 100
      - target: user-service-2:3000
        weight: 100
```

### 4.3 Kong on Kubernetes

```yaml
# Kong Ingress Controller を使った K8s 設定
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: rate-limiting-plugin
  namespace: api
config:
  minute: 100
  policy: local
  fault_tolerant: true
plugin: rate-limiting
---
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: jwt-plugin
  namespace: api
plugin: jwt
---
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: cors-plugin
  namespace: api
config:
  origins:
    - "https://app.example.com"
  methods:
    - GET
    - POST
    - PUT
    - DELETE
  headers:
    - Authorization
    - Content-Type
  credentials: true
  max_age: 86400
plugin: cors
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: user-api-ingress
  namespace: api
  annotations:
    konghq.com/plugins: rate-limiting-plugin,jwt-plugin,cors-plugin
    konghq.com/strip-path: "false"
    konghq.com/protocols: "https"
spec:
  ingressClassName: kong
  tls:
    - secretName: api-tls-secret
      hosts:
        - api.example.com
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /api/v1/users
            pathType: Prefix
            backend:
              service:
                name: user-service
                port:
                  number: 3000
          - path: /api/v1/orders
            pathType: Prefix
            backend:
              service:
                name: order-service
                port:
                  number: 3000
```

---

## 5. Nginx によるAPIゲートウェイ構成

### 5.1 基本構成

```nginx
# /etc/nginx/nginx.conf - APIゲートウェイ構成
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 4096;
    multi_accept on;
    use epoll;
}

http {
    # 基本設定
    include /etc/nginx/mime.types;
    default_type application/json;
    charset utf-8;

    # ログフォーマット（JSON構造化ログ）
    log_format json_combined escape=json
        '{'
            '"time":"$time_iso8601",'
            '"remote_addr":"$remote_addr",'
            '"request_method":"$request_method",'
            '"request_uri":"$request_uri",'
            '"status":"$status",'
            '"body_bytes_sent":"$body_bytes_sent",'
            '"request_time":"$request_time",'
            '"upstream_response_time":"$upstream_response_time",'
            '"http_user_agent":"$http_user_agent",'
            '"http_x_request_id":"$http_x_request_id",'
            '"upstream_addr":"$upstream_addr"'
        '}';

    access_log /var/log/nginx/access.log json_combined;

    # パフォーマンスチューニング
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    keepalive_requests 1000;

    # バッファ設定
    client_body_buffer_size 16k;
    client_max_body_size 10m;
    proxy_buffer_size 16k;
    proxy_buffers 4 32k;
    proxy_busy_buffers_size 64k;

    # Gzip圧縮
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types application/json application/xml text/plain;

    # セキュリティヘッダー
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # レート制限ゾーン
    limit_req_zone $binary_remote_addr zone=api_global:10m rate=100r/s;
    limit_req_zone $http_x_api_key zone=api_key:10m rate=50r/s;
    limit_req_zone $binary_remote_addr zone=auth_endpoint:10m rate=10r/s;

    # 接続数制限
    limit_conn_zone $binary_remote_addr zone=conn_limit:10m;

    # アップストリーム定義
    upstream user_service {
        least_conn;
        server user-service-1:3000 weight=5 max_fails=3 fail_timeout=30s;
        server user-service-2:3000 weight=5 max_fails=3 fail_timeout=30s;
        server user-service-3:3000 weight=3 max_fails=3 fail_timeout=30s backup;
        keepalive 32;
    }

    upstream order_service {
        least_conn;
        server order-service-1:3000 max_fails=3 fail_timeout=30s;
        server order-service-2:3000 max_fails=3 fail_timeout=30s;
        keepalive 16;
    }

    upstream payment_service {
        server payment-service-1:3000 max_fails=2 fail_timeout=60s;
        server payment-service-2:3000 max_fails=2 fail_timeout=60s;
        keepalive 8;
    }

    # キャッシュ設定
    proxy_cache_path /var/cache/nginx/api_cache
        levels=1:2
        keys_zone=api_cache:10m
        max_size=1g
        inactive=60m
        use_temp_path=off;

    # SSL設定
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
    ssl_session_tickets off;

    # HTTPS サーバー
    server {
        listen 443 ssl http2;
        server_name api.example.com;

        ssl_certificate /etc/nginx/ssl/fullchain.pem;
        ssl_certificate_key /etc/nginx/ssl/privkey.pem;

        # リクエストID生成
        set $request_id $http_x_request_id;
        if ($request_id = '') {
            set $request_id $request_id;
        }

        # ヘルスチェック（認証不要）
        location /health {
            access_log off;
            return 200 '{"status":"healthy"}';
        }

        # CORS Preflight
        location ~ ^/api/ {
            if ($request_method = 'OPTIONS') {
                add_header 'Access-Control-Allow-Origin' 'https://app.example.com';
                add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS';
                add_header 'Access-Control-Allow-Headers' 'Authorization, Content-Type, X-Request-ID';
                add_header 'Access-Control-Max-Age' 86400;
                add_header 'Content-Length' 0;
                return 204;
            }

            # 以下のlocationブロックへフォールスルー
        }

        # ユーザーAPI
        location /api/v1/users {
            limit_req zone=api_global burst=20 nodelay;
            limit_conn conn_limit 50;

            proxy_pass http://user_service;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-Request-ID $request_id;

            proxy_connect_timeout 5s;
            proxy_read_timeout 30s;
            proxy_send_timeout 10s;

            # レスポンスキャッシュ（GETリクエストのみ）
            proxy_cache api_cache;
            proxy_cache_methods GET;
            proxy_cache_valid 200 5m;
            proxy_cache_valid 404 1m;
            proxy_cache_key "$scheme$request_method$host$request_uri";
            proxy_cache_bypass $http_cache_control;
            add_header X-Cache-Status $upstream_cache_status;
        }

        # 注文API
        location /api/v1/orders {
            limit_req zone=api_global burst=10 nodelay;

            proxy_pass http://order_service;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-Request-ID $request_id;

            proxy_connect_timeout 5s;
            proxy_read_timeout 60s;
            proxy_send_timeout 10s;
        }

        # 決済API（厳しいレート制限）
        location /api/v1/payments {
            limit_req zone=auth_endpoint burst=5 nodelay;
            limit_conn conn_limit 10;

            proxy_pass http://payment_service;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-Request-ID $request_id;

            proxy_connect_timeout 5s;
            proxy_read_timeout 120s;
            proxy_send_timeout 30s;

            # キャッシュ無効
            proxy_no_cache 1;
            proxy_cache_bypass 1;
        }

        # エラーハンドリング
        error_page 429 = @rate_limited;
        location @rate_limited {
            default_type application/json;
            return 429 '{"error":"rate_limit_exceeded","message":"Too many requests. Please retry after some time.","retry_after":60}';
        }

        error_page 502 503 504 = @service_unavailable;
        location @service_unavailable {
            default_type application/json;
            return 503 '{"error":"service_unavailable","message":"The service is temporarily unavailable. Please try again later."}';
        }
    }

    # HTTP → HTTPS リダイレクト
    server {
        listen 80;
        server_name api.example.com;
        return 301 https://$server_name$request_uri;
    }
}
```

---

## 6. レート制限の設計と実装

### 6.1 レート制限アルゴリズムの比較

| アルゴリズム | 仕組み | 精度 | メモリ | バースト許容 | 実装難度 |
|------------|--------|------|--------|------------|---------|
| Fixed Window | 固定時間窓でカウント | 低 | 小 | 窓境界で2倍 | 低 |
| Sliding Window Log | リクエスト時刻を全記録 | 高 | 大 | なし | 中 |
| Sliding Window Counter | 前窓と現窓の加重平均 | 中-高 | 小 | 小 | 中 |
| Token Bucket | トークン消費方式 | 高 | 小 | 制御可能 | 中 |
| Leaky Bucket | 一定レートで流出 | 高 | 小 | なし | 中 |

### 6.2 Token Bucket アルゴリズムの実装

```javascript
// Token Bucket レート制限（Redis ベース）
const Redis = require('ioredis');

class TokenBucketRateLimiter {
  /**
   * @param {Object} options
   * @param {number} options.maxTokens - バケットの最大容量
   * @param {number} options.refillRate - 1秒あたりのトークン補充数
   * @param {number} options.tokensPerRequest - 1リクエストで消費するトークン数
   */
  constructor(options) {
    this.redis = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT) || 6379,
      maxRetriesPerRequest: 3,
      retryDelayOnFailover: 100,
    });
    this.maxTokens = options.maxTokens || 100;
    this.refillRate = options.refillRate || 10;
    this.tokensPerRequest = options.tokensPerRequest || 1;
    this.keyPrefix = 'ratelimit:token_bucket:';
  }

  /**
   * レート制限チェック
   * @param {string} identifier - クライアント識別子（IP, API Key, User ID）
   * @returns {Object} { allowed, remaining, retryAfter, limit }
   */
  async checkLimit(identifier) {
    const key = `${this.keyPrefix}${identifier}`;
    const now = Date.now();

    // Luaスクリプトでアトミックに実行
    const luaScript = `
      local key = KEYS[1]
      local max_tokens = tonumber(ARGV[1])
      local refill_rate = tonumber(ARGV[2])
      local tokens_per_request = tonumber(ARGV[3])
      local now = tonumber(ARGV[4])

      local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
      local tokens = tonumber(bucket[1])
      local last_refill = tonumber(bucket[2])

      -- 初回アクセスの場合
      if tokens == nil then
        tokens = max_tokens
        last_refill = now
      end

      -- トークン補充
      local elapsed = (now - last_refill) / 1000
      local new_tokens = elapsed * refill_rate
      tokens = math.min(max_tokens, tokens + new_tokens)

      -- トークン消費判定
      local allowed = 0
      local remaining = tokens
      local retry_after = 0

      if tokens >= tokens_per_request then
        tokens = tokens - tokens_per_request
        allowed = 1
        remaining = tokens
      else
        retry_after = math.ceil((tokens_per_request - tokens) / refill_rate)
        remaining = tokens
      end

      -- 更新
      redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
      redis.call('EXPIRE', key, math.ceil(max_tokens / refill_rate) + 10)

      return {allowed, math.floor(remaining), retry_after}
    `;

    const result = await this.redis.eval(
      luaScript, 1, key,
      this.maxTokens, this.refillRate, this.tokensPerRequest, now
    );

    return {
      allowed: result[0] === 1,
      remaining: result[1],
      retryAfter: result[2],
      limit: this.maxTokens,
    };
  }

  /**
   * Express ミドルウェア
   */
  middleware(identifierFn) {
    return async (req, res, next) => {
      const identifier = identifierFn
        ? identifierFn(req)
        : req.headers['x-api-key'] || req.ip;

      try {
        const result = await this.checkLimit(identifier);

        // レート制限ヘッダーを設定
        res.set('X-RateLimit-Limit', String(result.limit));
        res.set('X-RateLimit-Remaining', String(result.remaining));

        if (!result.allowed) {
          res.set('Retry-After', String(result.retryAfter));
          return res.status(429).json({
            error: 'rate_limit_exceeded',
            message: 'Too many requests',
            retryAfter: result.retryAfter,
            limit: result.limit,
          });
        }

        next();
      } catch (error) {
        // Redis障害時はリクエストを許可（fault-tolerant）
        console.error('Rate limiter error:', error.message);
        next();
      }
    };
  }
}

// 利用例
const rateLimiter = new TokenBucketRateLimiter({
  maxTokens: 100,     // バケット容量
  refillRate: 10,      // 毎秒10トークン補充
  tokensPerRequest: 1, // 1リクエスト = 1トークン
});

// Express での利用
const express = require('express');
const app = express();

// グローバルレート制限
app.use(rateLimiter.middleware());

// エンドポイント別のレート制限
const strictLimiter = new TokenBucketRateLimiter({
  maxTokens: 10,
  refillRate: 1,
  tokensPerRequest: 1,
});

app.post('/api/v1/auth/login',
  strictLimiter.middleware((req) => `login:${req.ip}`),
  (req, res) => { /* ... */ }
);
```

### 6.3 Sliding Window Counter の実装

```javascript
// Sliding Window Counter（精度とメモリ効率のバランス型）
class SlidingWindowRateLimiter {
  constructor(options) {
    this.redis = new Redis(options.redisUrl || 'redis://localhost:6379');
    this.windowSize = options.windowSize || 60; // 秒
    this.maxRequests = options.maxRequests || 100;
    this.keyPrefix = 'ratelimit:sliding:';
  }

  async checkLimit(identifier) {
    const key = `${this.keyPrefix}${identifier}`;
    const now = Math.floor(Date.now() / 1000);
    const currentWindow = Math.floor(now / this.windowSize);
    const previousWindow = currentWindow - 1;
    const windowProgress = (now % this.windowSize) / this.windowSize;

    const luaScript = `
      local current_key = KEYS[1] .. ':' .. ARGV[1]
      local previous_key = KEYS[1] .. ':' .. ARGV[2]
      local max_requests = tonumber(ARGV[3])
      local window_progress = tonumber(ARGV[4])
      local window_size = tonumber(ARGV[5])

      local current_count = tonumber(redis.call('GET', current_key) or '0')
      local previous_count = tonumber(redis.call('GET', previous_key) or '0')

      -- 加重平均による推定レート
      local estimated_count = previous_count * (1 - window_progress) + current_count

      if estimated_count >= max_requests then
        local retry_after = math.ceil(window_size * (1 - window_progress))
        return {0, math.floor(max_requests - estimated_count), retry_after}
      end

      -- カウントインクリメント
      redis.call('INCR', current_key)
      redis.call('EXPIRE', current_key, window_size * 2)

      local remaining = math.floor(max_requests - estimated_count - 1)
      return {1, remaining, 0}
    `;

    const result = await this.redis.eval(
      luaScript, 1, key,
      currentWindow, previousWindow,
      this.maxRequests, windowProgress, this.windowSize
    );

    return {
      allowed: result[0] === 1,
      remaining: Math.max(0, result[1]),
      retryAfter: result[2],
      limit: this.maxRequests,
    };
  }
}
```

---

## 7. 認証統合パターン

### 7.1 ゲートウェイ層での認証フロー

```
┌───────────────────────────────────────────────────────────────────┐
│ ゲートウェイ認証フロー                                               │
│                                                                   │
│  (1) API Key 認証                                                 │
│  Client ──[X-API-Key: xxx]──→ Gateway ──[検証]──→ Backend         │
│                                  │                                │
│                                  ├── Redis/DB で Key 検証          │
│                                  ├── プラン/クォータ確認             │
│                                  └── X-Consumer-ID ヘッダー付与     │
│                                                                   │
│  (2) JWT Bearer Token 認証                                        │
│  Client ──[Authorization: Bearer xxx]──→ Gateway ──→ Backend      │
│                                            │                      │
│                                            ├── 署名検証（RS256）    │
│                                            ├── exp / iss 確認      │
│                                            ├── scope 確認          │
│                                            └── X-User-ID 付与      │
│                                                                   │
│  (3) OAuth 2.0 Token Introspection                                │
│  Client ──[Token]──→ Gateway ──→ Auth Server ──→ Gateway → Backend│
│                                      │                            │
│                                      ├── /introspect エンドポイント │
│                                      ├── active / scope 確認       │
│                                      └── キャッシュ（5分）           │
│                                                                   │
│  (4) mTLS（サービス間通信）                                         │
│  Service A ──[Client Cert]──→ Gateway ──[証明書検証]──→ Service B  │
│                                  │                                │
│                                  ├── CA 証明書チェーン検証           │
│                                  ├── CN/SAN によるサービス識別        │
│                                  └── 証明書ローテーション対応         │
└───────────────────────────────────────────────────────────────────┘
```

### 7.2 JWT認証ミドルウェアの実装

```javascript
// ゲートウェイ用 JWT 認証ミドルウェア
const jwt = require('jsonwebtoken');
const jwksClient = require('jwks-rsa');

class JwtAuthenticator {
  constructor(options) {
    this.issuer = options.issuer;
    this.audience = options.audience;
    this.algorithms = options.algorithms || ['RS256'];
    this.clockTolerance = options.clockTolerance || 30;

    // JWKS クライアント（公開鍵の動的取得）
    this.jwks = jwksClient({
      jwksUri: `${this.issuer}/.well-known/jwks.json`,
      cache: true,
      cacheMaxEntries: 5,
      cacheMaxAge: 600000, // 10分キャッシュ
      rateLimit: true,
      jwksRequestsPerMinute: 10,
    });

    // パス除外設定
    this.excludePaths = new Set(options.excludePaths || [
      '/health',
      '/metrics',
      '/api/v1/auth/login',
      '/api/v1/auth/register',
    ]);
  }

  /**
   * 署名検証用の公開鍵を取得
   */
  getSigningKey(header) {
    return new Promise((resolve, reject) => {
      this.jwks.getSigningKey(header.kid, (err, key) => {
        if (err) return reject(err);
        resolve(key.getPublicKey());
      });
    });
  }

  /**
   * Express ミドルウェア
   */
  middleware() {
    return async (req, res, next) => {
      // 除外パスのチェック
      if (this.excludePaths.has(req.path)) {
        return next();
      }

      // OPTIONS リクエストはスキップ
      if (req.method === 'OPTIONS') {
        return next();
      }

      const authHeader = req.headers.authorization;
      if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return res.status(401).json({
          error: 'unauthorized',
          message: 'Missing or invalid Authorization header',
        });
      }

      const token = authHeader.substring(7);

      try {
        // ヘッダーを先にデコードして kid を取得
        const decoded = jwt.decode(token, { complete: true });
        if (!decoded || !decoded.header) {
          return res.status(401).json({
            error: 'invalid_token',
            message: 'Token could not be decoded',
          });
        }

        // 公開鍵を取得して検証
        const signingKey = await this.getSigningKey(decoded.header);
        const payload = jwt.verify(token, signingKey, {
          algorithms: this.algorithms,
          issuer: this.issuer,
          audience: this.audience,
          clockTolerance: this.clockTolerance,
        });

        // 検証済み情報をリクエストに付与
        req.user = {
          id: payload.sub,
          email: payload.email,
          roles: payload.roles || [],
          scopes: payload.scope ? payload.scope.split(' ') : [],
        };

        // バックエンドサービス向けヘッダーを設定
        req.headers['x-user-id'] = payload.sub;
        req.headers['x-user-email'] = payload.email || '';
        req.headers['x-user-roles'] = (payload.roles || []).join(',');

        next();
      } catch (error) {
        if (error.name === 'TokenExpiredError') {
          return res.status(401).json({
            error: 'token_expired',
            message: 'Token has expired',
            expiredAt: error.expiredAt,
          });
        }
        if (error.name === 'JsonWebTokenError') {
          return res.status(401).json({
            error: 'invalid_token',
            message: error.message,
          });
        }
        console.error('JWT verification error:', error);
        return res.status(500).json({
          error: 'internal_error',
          message: 'Authentication service unavailable',
        });
      }
    };
  }

  /**
   * スコープベースの認可ミドルウェア
   */
  requireScopes(...requiredScopes) {
    return (req, res, next) => {
      if (!req.user || !req.user.scopes) {
        return res.status(403).json({
          error: 'forbidden',
          message: 'Insufficient permissions',
        });
      }

      const hasAllScopes = requiredScopes.every(
        scope => req.user.scopes.includes(scope)
      );

      if (!hasAllScopes) {
        return res.status(403).json({
          error: 'insufficient_scope',
          message: `Required scopes: ${requiredScopes.join(', ')}`,
          requiredScopes,
          currentScopes: req.user.scopes,
        });
      }

      next();
    };
  }
}

// 利用例
const auth = new JwtAuthenticator({
  issuer: 'https://auth.example.com',
  audience: 'https://api.example.com',
  algorithms: ['RS256'],
  excludePaths: ['/health', '/api/v1/auth/login'],
});

app.use(auth.middleware());

app.get('/api/v1/admin/users',
  auth.requireScopes('admin:read', 'users:list'),
  (req, res) => {
    // req.user.id, req.user.roles でアクセス可能
    res.json({ users: [] });
  }
);
```

---

## 8. サーキットブレーカーパターン

### 8.1 状態遷移モデル

```
┌───────────────────────────────────────────────────────────┐
│ サーキットブレーカー 状態遷移図                                │
│                                                           │
│    ┌──────────┐   失敗閾値超過   ┌──────────┐              │
│    │  CLOSED  │ ──────────────→ │   OPEN   │              │
│    │ (通常)   │                 │ (遮断)   │              │
│    │          │ ←────────────── │          │              │
│    └──────────┘   成功(3回)     └────┬─────┘              │
│         ↑                           │                     │
│         │  成功                      │ タイムアウト          │
│         │                           ↓                     │
│         │                    ┌──────────┐                 │
│         └─────────────────── │HALF-OPEN │                 │
│                    成功      │ (試行)   │                  │
│                              └────┬─────┘                 │
│                                   │                       │
│                                   │ 失敗                   │
│                                   ↓                       │
│                              ┌──────────┐                 │
│                              │   OPEN   │                 │
│                              │ (再遮断) │                  │
│                              └──────────┘                 │
│                                                           │
│  パラメータ設定ガイドライン:                                   │
│   失敗閾値: 5回連続失敗 or 50%エラー率（直近10リクエスト）       │
│   タイムアウト: 30秒後に Half-Open へ遷移                      │
│   成功閾値: Half-Open で3回連続成功したら Closed へ復帰         │
│   監視ウィンドウ: 直近10リクエストでエラー率を計算                │
└───────────────────────────────────────────────────────────┘
```

### 8.2 プロダクション品質のサーキットブレーカー実装

```javascript
// プロダクション品質のサーキットブレーカー
const EventEmitter = require('events');

class CircuitBreaker extends EventEmitter {
  constructor(options = {}) {
    super();
    this.name = options.name || 'default';
    this.failureThreshold = options.failureThreshold || 5;
    this.successThreshold = options.successThreshold || 3;
    this.resetTimeout = options.resetTimeout || 30000;
    this.monitorWindow = options.monitorWindow || 10;
    this.halfOpenMaxConcurrent = options.halfOpenMaxConcurrent || 1;

    this.state = 'CLOSED';
    this.failures = 0;
    this.successes = 0;
    this.lastFailureTime = null;
    this.halfOpenRequests = 0;
    this.requestHistory = [];

    // メトリクス
    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      rejectedRequests: 0,
      stateChanges: [],
    };
  }

  getState() {
    if (this.state === 'OPEN') {
      if (Date.now() - this.lastFailureTime >= this.resetTimeout) {
        this._transitionTo('HALF_OPEN');
      }
    }
    return this.state;
  }

  _transitionTo(newState) {
    const oldState = this.state;
    this.state = newState;

    if (newState === 'CLOSED') {
      this.failures = 0;
      this.successes = 0;
      this.halfOpenRequests = 0;
      this.requestHistory = [];
    }
    if (newState === 'HALF_OPEN') {
      this.halfOpenRequests = 0;
      this.successes = 0;
    }

    this.metrics.stateChanges.push({
      from: oldState,
      to: newState,
      timestamp: new Date().toISOString(),
    });

    this.emit('stateChange', { from: oldState, to: newState, name: this.name });
  }

  _getErrorRate() {
    if (this.requestHistory.length < this.monitorWindow) {
      return 0;
    }
    const recentRequests = this.requestHistory.slice(-this.monitorWindow);
    const failures = recentRequests.filter(r => !r.success).length;
    return failures / recentRequests.length;
  }

  async execute(fn, fallback) {
    this.metrics.totalRequests++;
    const currentState = this.getState();

    // OPEN状態: 即座に拒否
    if (currentState === 'OPEN') {
      this.metrics.rejectedRequests++;
      this.emit('rejected', { name: this.name, state: currentState });

      if (fallback) {
        return fallback(new Error(`Circuit ${this.name} is OPEN`));
      }
      throw new Error(`Circuit ${this.name} is OPEN`);
    }

    // HALF_OPEN状態: 同時リクエスト数を制限
    if (currentState === 'HALF_OPEN') {
      if (this.halfOpenRequests >= this.halfOpenMaxConcurrent) {
        this.metrics.rejectedRequests++;
        if (fallback) {
          return fallback(new Error(`Circuit ${this.name} is HALF_OPEN (max concurrent)`));
        }
        throw new Error(`Circuit ${this.name} is HALF_OPEN (max concurrent reached)`);
      }
      this.halfOpenRequests++;
    }

    try {
      const result = await fn();
      this._onSuccess();
      return result;
    } catch (error) {
      this._onFailure(error);
      if (fallback) return fallback(error);
      throw error;
    }
  }

  _onSuccess() {
    this.metrics.successfulRequests++;
    this.requestHistory.push({ success: true, timestamp: Date.now() });

    if (this.state === 'HALF_OPEN') {
      this.successes++;
      if (this.successes >= this.successThreshold) {
        this._transitionTo('CLOSED');
        this.emit('recovery', { name: this.name });
      }
    }

    // CLOSED 状態での連続失敗カウントをリセット
    if (this.state === 'CLOSED') {
      this.failures = Math.max(0, this.failures - 1);
    }
  }

  _onFailure(error) {
    this.metrics.failedRequests++;
    this.requestHistory.push({ success: false, timestamp: Date.now(), error: error.message });
    this.lastFailureTime = Date.now();

    if (this.state === 'HALF_OPEN') {
      this._transitionTo('OPEN');
      this.emit('trip', { name: this.name, error: error.message });
      return;
    }

    if (this.state === 'CLOSED') {
      this.failures++;
      const errorRate = this._getErrorRate();

      if (this.failures >= this.failureThreshold || errorRate >= 0.5) {
        this._transitionTo('OPEN');
        this.emit('trip', { name: this.name, error: error.message, errorRate });
      }
    }
  }

  getMetrics() {
    return {
      ...this.metrics,
      currentState: this.getState(),
      errorRate: this._getErrorRate(),
      consecutiveFailures: this.failures,
    };
  }

  reset() {
    this._transitionTo('CLOSED');
  }
}

// サーキットブレーカーレジストリ（複数サービス管理）
class CircuitBreakerRegistry {
  constructor() {
    this.breakers = new Map();
  }

  get(name, options = {}) {
    if (!this.breakers.has(name)) {
      const breaker = new CircuitBreaker({ name, ...options });

      breaker.on('stateChange', (event) => {
        console.log(`[CircuitBreaker] ${event.name}: ${event.from} → ${event.to}`);
      });
      breaker.on('trip', (event) => {
        console.error(`[CircuitBreaker] ${event.name} TRIPPED: ${event.error}`);
      });
      breaker.on('recovery', (event) => {
        console.log(`[CircuitBreaker] ${event.name} RECOVERED`);
      });

      this.breakers.set(name, breaker);
    }
    return this.breakers.get(name);
  }

  getAllMetrics() {
    const metrics = {};
    for (const [name, breaker] of this.breakers) {
      metrics[name] = breaker.getMetrics();
    }
    return metrics;
  }
}

// 使用例
const registry = new CircuitBreakerRegistry();

async function getUser(id) {
  const breaker = registry.get('user-service', {
    failureThreshold: 5,
    resetTimeout: 30000,
    successThreshold: 3,
  });

  return breaker.execute(
    async () => {
      const response = await fetch(`http://user-service:3000/users/${id}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    },
    (error) => ({
      id,
      name: 'Unknown User',
      _fallback: true,
      _error: error.message,
    })
  );
}

// メトリクスエンドポイント
app.get('/metrics/circuit-breakers', (req, res) => {
  res.json(registry.getAllMetrics());
});
```

---

## 9. サービスメッシュとの統合

### 9.1 サービスメッシュの基本概念

サービスメッシュは、マイクロサービス間の通信を管理するインフラストラクチャ層である。APIゲートウェイが「南北」（外部→内部）トラフィックを管理するのに対し、サービスメッシュは「東西」（内部サービス間）トラフィックを管理する。

```
┌───────────────────────────────────────────────────────────────────┐
│ APIゲートウェイ vs サービスメッシュ                                    │
│                                                                   │
│  南北トラフィック（North-South）                                     │
│  ─────────────────────────────                                    │
│  外部クライアント → APIゲートウェイ → 内部サービス                       │
│                                                                   │
│  ・外部からのリクエストを処理                                         │
│  ・認証、レート制限、TLS終端                                         │
│  ・API管理、開発者ポータル                                           │
│                                                                   │
│  東西トラフィック（East-West）                                       │
│  ─────────────────────────────                                    │
│  内部サービス ↔ サイドカープロキシ ↔ 内部サービス                       │
│                                                                   │
│  ・サービス間通信の暗号化（mTLS）                                     │
│  ・サービスディスカバリ                                               │
│  ・負荷分散、リトライ、サーキットブレーカー                              │
│  ・トラフィック制御（カナリア、A/Bテスト）                              │
│  ・オブザーバビリティ（メトリクス、トレース、ログ）                       │
└───────────────────────────────────────────────────────────────────┘
```

### 9.2 Istio + Envoy アーキテクチャ

```
┌───────────────────────────────────────────────────────────────────┐
│ Istio アーキテクチャ                                                │
│                                                                   │
│  ┌─────────────────── Control Plane ──────────────────┐           │
│  │                                                     │           │
│  │   istiod (Pilot + Citadel + Galley 統合)            │           │
│  │   ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │           │
│  │   │  Pilot   │ │ Citadel  │ │   Configuration  │   │           │
│  │   │ (Config) │ │ (Cert)   │ │   (Validation)   │   │           │
│  │   └────┬─────┘ └────┬─────┘ └────────┬─────────┘   │           │
│  └────────┼─────────────┼───────────────┼─────────────┘           │
│           │ xDS API     │ mTLS証明書     │ 設定配布                │
│  ┌────────▼─────────────▼───────────────▼─────────────┐           │
│  │                 Data Plane                          │           │
│  │                                                     │           │
│  │  ┌──────────────┐    ┌──────────────┐               │           │
│  │  │  Pod A       │    │  Pod B       │               │           │
│  │  │ ┌──────────┐ │    │ ┌──────────┐ │               │           │
│  │  │ │ Service A│ │    │ │ Service B│ │               │           │
│  │  │ └────┬─────┘ │    │ └────┬─────┘ │               │           │
│  │  │      │       │    │      │       │               │           │
│  │  │ ┌────▼─────┐ │    │ ┌────▼─────┐ │               │           │
│  │  │ │  Envoy   │◄├────┤►│  Envoy   │ │               │           │
│  │  │ │ (Sidecar)│ │mTLS│ │ (Sidecar)│ │               │           │
│  │  │ └──────────┘ │    │ └──────────┘ │               │           │
│  │  └──────────────┘    └──────────────┘               │           │
│  └─────────────────────────────────────────────────────┘           │
└───────────────────────────────────────────────────────────────────┘
```

### 9.3 Istio の設定例

```yaml
# VirtualService: トラフィックルーティング
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: user-service-vs
  namespace: api
spec:
  hosts:
    - user-service
  http:
    # カナリアリリース: 10%のトラフィックを v2 に振り分け
    - match:
        - headers:
            x-canary:
              exact: "true"
      route:
        - destination:
            host: user-service
            subset: v2
          weight: 100
    - route:
        - destination:
            host: user-service
            subset: v1
          weight: 90
        - destination:
            host: user-service
            subset: v2
          weight: 10
      timeout: 10s
      retries:
        attempts: 3
        perTryTimeout: 3s
        retryOn: 5xx,reset,connect-failure,retriable-4xx
      fault:
        delay:
          percentage:
            value: 0.1
          fixedDelay: 5s
---
# DestinationRule: サブセット定義とサーキットブレーカー
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: user-service-dr
  namespace: api
spec:
  host: user-service
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
        connectTimeout: 5s
      http:
        h2UpgradePolicy: DEFAULT
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
        maxRequestsPerConnection: 10
        maxRetries: 3
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 10s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
    loadBalancer:
      simple: LEAST_REQUEST
  subsets:
    - name: v1
      labels:
        version: v1
    - name: v2
      labels:
        version: v2
---
# PeerAuthentication: mTLS 設定
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: api
spec:
  mtls:
    mode: STRICT
---
# AuthorizationPolicy: サービス間の認可
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: user-service-authz
  namespace: api
spec:
  selector:
    matchLabels:
      app: user-service
  rules:
    - from:
        - source:
            principals:
              - "cluster.local/ns/api/sa/order-service"
              - "cluster.local/ns/api/sa/api-gateway"
      to:
        - operation:
            methods: ["GET", "POST"]
            paths: ["/users/*"]
    - from:
        - source:
            principals:
              - "cluster.local/ns/api/sa/api-gateway"
      to:
        - operation:
            methods: ["DELETE"]
            paths: ["/users/*"]
---
# RequestAuthentication: JWT認証（Istio Ingress Gateway）
apiVersion: security.istio.io/v1beta1
kind: RequestAuthentication
metadata:
  name: jwt-auth
  namespace: istio-system
spec:
  selector:
    matchLabels:
      istio: ingressgateway
  jwtRules:
    - issuer: "https://auth.example.com/"
      jwksUri: "https://auth.example.com/.well-known/jwks.json"
      audiences:
        - "https://api.example.com"
      forwardOriginalToken: true
      outputPayloadToHeader: "x-jwt-payload"
```

### 9.4 Istio Gateway + APIゲートウェイの2層構成

```yaml
# Istio Ingress Gateway（外部トラフィック受け口）
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: api-gateway
  namespace: istio-system
spec:
  selector:
    istio: ingressgateway
  servers:
    - port:
        number: 443
        name: https
        protocol: HTTPS
      tls:
        mode: SIMPLE
        credentialName: api-tls-credential
      hosts:
        - "api.example.com"
---
# VirtualService: Istio Ingress → Kong Gateway
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: api-routing
  namespace: api
spec:
  hosts:
    - "api.example.com"
  gateways:
    - istio-system/api-gateway
  http:
    - match:
        - uri:
            prefix: /api/
      route:
        - destination:
            host: kong-proxy.kong-system.svc.cluster.local
            port:
              number: 80
      corsPolicy:
        allowOrigins:
          - exact: "https://app.example.com"
        allowMethods:
          - GET
          - POST
          - PUT
          - DELETE
          - OPTIONS
        allowHeaders:
          - Authorization
          - Content-Type
          - X-Request-ID
        maxAge: "86400s"
```

---

## 10. BFF（Backend for Frontend）パターン詳細

### 10.1 BFF の設計原則

```
BFF パターンの全体像:

  ┌───────────────────────────────────────────────────────────────┐
  │                                                               │
  │  Web Browser ─── Web BFF ───┐                                 │
  │                              │                                │
  │  iOS App ────── Mobile BFF ──┼──→ User Service                │
  │  Android App ──┘              │    Order Service               │
  │                              │    Payment Service             │
  │  Partner API ── Public GW ───┘    Notification Service        │
  │                                                               │
  └───────────────────────────────────────────────────────────────┘

  各 BFF の責務:
  ┌────────────────────────────────────────────────────────────┐
  │ Web BFF:                                                   │
  │  ├── フルフィーチャーのレスポンス                               │
  │  ├── SSR 用のデータ集約                                      │
  │  ├── Cookie ベース認証（HttpOnly, Secure, SameSite）          │
  │  ├── CSRF 対策                                              │
  │  ├── HTML メタデータ生成（OGP, SEO）                          │
  │  └── WebSocket 接続管理                                      │
  │                                                            │
  │ Mobile BFF:                                                │
  │  ├── 軽量レスポンス（フィールド選択、帯域節約）                   │
  │  ├── プッシュ通知トークン管理                                  │
  │  ├── Bearer Token 認証（OAuth 2.0）                          │
  │  ├── オフラインサポート（差分同期API）                          │
  │  ├── アプリバージョン別の互換性レイヤー                         │
  │  └── デバイス情報ヘッダー処理                                  │
  │                                                            │
  │ Public API GW:                                             │
  │  ├── REST + ページネーション（カーソルベース）                   │
  │  ├── API Key 認証 + Usage Plan                              │
  │  ├── 厳格なレート制限（プランごと）                             │
  │  ├── OpenAPI 仕様書の自動生成                                 │
  │  ├── Webhook 配信                                           │
  │  └── API バージョニング（URL / Header）                       │
  └────────────────────────────────────────────────────────────┘
```

### 10.2 BFF 実装例（Node.js / Express）

```javascript
// Web BFF - API Composition パターン
const express = require('express');
const axios = require('axios');

const app = express();

// サービスクライアント（サーキットブレーカー付き）
const serviceClients = {
  user: createServiceClient('http://user-service:3000', 'user-service'),
  order: createServiceClient('http://order-service:3000', 'order-service'),
  product: createServiceClient('http://product-service:3000', 'product-service'),
};

function createServiceClient(baseURL, name) {
  const client = axios.create({
    baseURL,
    timeout: 5000,
    headers: { 'Content-Type': 'application/json' },
  });
  const breaker = registry.get(name, {
    failureThreshold: 5,
    resetTimeout: 30000,
  });

  return {
    async get(path, options = {}) {
      return breaker.execute(
        () => client.get(path, options).then(r => r.data),
        () => options.fallback || null
      );
    },
    async post(path, data, options = {}) {
      return breaker.execute(
        () => client.post(path, data, options).then(r => r.data),
        () => options.fallback || null
      );
    },
  };
}

// ダッシュボードAPI: 複数サービスからデータを集約
app.get('/bff/dashboard', async (req, res) => {
  const userId = req.headers['x-user-id'];

  try {
    // 並行リクエスト（Promise.allSettled で部分障害に対応）
    const [userResult, ordersResult, recommendationsResult] = await Promise.allSettled([
      serviceClients.user.get(`/users/${userId}`, {
        fallback: { id: userId, name: 'User', _fallback: true },
      }),
      serviceClients.order.get(`/orders?userId=${userId}&limit=5`, {
        fallback: { items: [], total: 0, _fallback: true },
      }),
      serviceClients.product.get(`/recommendations?userId=${userId}&limit=10`, {
        fallback: { items: [], _fallback: true },
      }),
    ]);

    // レスポンス集約
    const dashboard = {
      user: userResult.status === 'fulfilled' ? userResult.value : null,
      recentOrders: ordersResult.status === 'fulfilled' ? ordersResult.value : { items: [] },
      recommendations: recommendationsResult.status === 'fulfilled'
        ? recommendationsResult.value
        : { items: [] },
      _metadata: {
        timestamp: new Date().toISOString(),
        partial: [userResult, ordersResult, recommendationsResult]
          .some(r => r.status === 'rejected' || (r.value && r.value._fallback)),
      },
    };

    // 部分障害の場合は 206 Partial Content
    const statusCode = dashboard._metadata.partial ? 206 : 200;
    res.status(statusCode).json(dashboard);
  } catch (error) {
    console.error('Dashboard aggregation error:', error);
    res.status(500).json({ error: 'internal_error' });
  }
});

// Mobile BFF: 軽量レスポンス + フィールド選択
app.get('/bff/mobile/feed', async (req, res) => {
  const userId = req.headers['x-user-id'];
  const fields = req.query.fields ? req.query.fields.split(',') : null;
  const appVersion = req.headers['x-app-version'] || '1.0.0';

  try {
    const [orders, notifications] = await Promise.all([
      serviceClients.order.get(`/orders?userId=${userId}&limit=20`),
      serviceClients.user.get(`/users/${userId}/notifications?unread=true`),
    ]);

    let feed = {
      orders: orders?.items || [],
      unreadCount: notifications?.total || 0,
      notifications: (notifications?.items || []).slice(0, 5),
    };

    // フィールド選択（帯域節約）
    if (fields) {
      const filteredFeed = {};
      for (const field of fields) {
        if (feed[field] !== undefined) {
          filteredFeed[field] = feed[field];
        }
      }
      feed = filteredFeed;
    }

    // アプリバージョン互換性
    if (compareVersions(appVersion, '2.0.0') < 0) {
      // v1 形式への変換
      feed = transformToV1Format(feed);
    }

    res.json(feed);
  } catch (error) {
    console.error('Mobile feed error:', error);
    res.status(500).json({ error: 'internal_error' });
  }
});

function compareVersions(a, b) {
  const pa = a.split('.').map(Number);
  const pb = b.split('.').map(Number);
  for (let i = 0; i < 3; i++) {
    if (pa[i] > pb[i]) return 1;
    if (pa[i] < pb[i]) return -1;
  }
  return 0;
}

function transformToV1Format(feed) {
  return {
    data: feed.orders || [],
    badge: feed.unreadCount || 0,
  };
}
```

---

## 11. リトライ戦略とタイムアウト設計

### 11.1 リトライパターンの比較

| パターン | 説明 | 適用場面 | 注意点 |
|---------|------|---------|--------|
| Immediate Retry | 即座にリトライ | 一時的なネットワークエラー | サービスに負荷を与える |
| Fixed Interval | 固定間隔でリトライ | 短時間の障害 | Thundering Herd 問題 |
| Exponential Backoff | 指数関数的に間隔拡大 | 一般的なリトライ | 最大間隔の設定が必要 |
| Exponential + Jitter | 指数 + ランダム揺らぎ | 分散システム推奨 | 標準的なベストプラクティス |
| Circuit Breaker + Retry | CB内でリトライ | 障害検知と組み合わせ | 複雑だが最も堅牢 |

### 11.2 Exponential Backoff with Jitter の実装

```javascript
// プロダクション品質のリトライユーティリティ
class RetryPolicy {
  constructor(options = {}) {
    this.maxRetries = options.maxRetries || 3;
    this.baseDelay = options.baseDelay || 1000;  // 1秒
    this.maxDelay = options.maxDelay || 30000;    // 30秒
    this.jitterFactor = options.jitterFactor || 0.5;
    this.retryableErrors = options.retryableErrors || [
      'ECONNRESET', 'ECONNREFUSED', 'ETIMEDOUT', 'EPIPE',
      'EAI_AGAIN', 'EHOSTUNREACH',
    ];
    this.retryableStatusCodes = options.retryableStatusCodes || [
      408, 429, 500, 502, 503, 504,
    ];
  }

  /**
   * リトライ対象かどうかを判定
   */
  isRetryable(error) {
    // ネットワークエラー
    if (error.code && this.retryableErrors.includes(error.code)) {
      return true;
    }
    // HTTP ステータスコード
    if (error.response && this.retryableStatusCodes.includes(error.response.status)) {
      return true;
    }
    return false;
  }

  /**
   * 次のリトライまでの待機時間を計算
   * Exponential Backoff with Full Jitter
   */
  calculateDelay(attempt) {
    // 指数バックオフ
    const exponentialDelay = Math.min(
      this.maxDelay,
      this.baseDelay * Math.pow(2, attempt)
    );
    // Full Jitter: [0, exponentialDelay] の範囲でランダム
    const jitter = Math.random() * exponentialDelay * this.jitterFactor;
    return Math.floor(exponentialDelay * (1 - this.jitterFactor) + jitter);
  }

  /**
   * リトライ付きで関数を実行
   */
  async execute(fn, context = {}) {
    let lastError;

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        const result = await fn(attempt);
        return result;
      } catch (error) {
        lastError = error;

        // 最終試行 or リトライ不可のエラー
        if (attempt === this.maxRetries || !this.isRetryable(error)) {
          throw error;
        }

        const delay = this.calculateDelay(attempt);

        // 429 の場合は Retry-After ヘッダーを尊重
        const retryAfterHeader = error.response?.headers?.['retry-after'];
        const actualDelay = retryAfterHeader
          ? Math.max(delay, parseInt(retryAfterHeader) * 1000)
          : delay;

        console.warn(
          `[Retry] ${context.name || 'request'} attempt ${attempt + 1}/${this.maxRetries}`,
          `delay=${actualDelay}ms`,
          `error=${error.message}`
        );

        await new Promise(resolve => setTimeout(resolve, actualDelay));
      }
    }

    throw lastError;
  }
}

// タイムアウト階層の設計
class TimeoutConfig {
  /**
   * タイムアウト階層:
   *   クライアント > ゲートウェイ > サービス > DB/外部API
   *
   *   Client: 30s → Gateway: 25s → Service: 20s → DB: 5s
   *   各層で余裕を持たせることで、適切なエラーレスポンスを返せる
   */
  static getConfig(tier) {
    const configs = {
      // 外部クライアントに面するゲートウェイ
      gateway: {
        connectTimeout: 5000,
        readTimeout: 25000,
        writeTimeout: 10000,
        idleTimeout: 60000,
      },
      // 内部サービス間通信
      service: {
        connectTimeout: 3000,
        readTimeout: 20000,
        writeTimeout: 5000,
        idleTimeout: 30000,
      },
      // データベース接続
      database: {
        connectTimeout: 2000,
        queryTimeout: 5000,
        poolTimeout: 10000,
      },
      // 外部API呼び出し
      externalApi: {
        connectTimeout: 5000,
        readTimeout: 15000,
        writeTimeout: 5000,
      },
    };
    return configs[tier];
  }
}

// 利用例: ゲートウェイでのサービス呼び出し
const retryPolicy = new RetryPolicy({
  maxRetries: 3,
  baseDelay: 500,
  maxDelay: 5000,
});

async function callUserService(userId) {
  const timeouts = TimeoutConfig.getConfig('service');

  return retryPolicy.execute(
    async (attempt) => {
      const response = await axios.get(
        `http://user-service:3000/users/${userId}`,
        {
          timeout: timeouts.readTimeout,
          headers: {
            'X-Retry-Attempt': String(attempt),
            'X-Request-Timeout': String(timeouts.readTimeout),
          },
        }
      );
      return response.data;
    },
    { name: `getUser(${userId})` }
  );
}
```

---

## 12. モニタリングとオブザーバビリティ

### 12.1 APIゲートウェイのメトリクス設計

ゲートウェイで収集すべきメトリクスは RED メソッド（Rate, Error, Duration）を基本とする。

```
APIゲートウェイ メトリクス体系:

  (1) Rate（リクエスト率）
      ├── requests_total: リクエスト総数
      ├── requests_per_second: RPS（秒間リクエスト数）
      └── requests_by_route: ルート別リクエスト数

  (2) Error（エラー率）
      ├── errors_total: エラー総数
      ├── error_rate: エラー率（4xx + 5xx）/ total
      ├── errors_by_status: ステータスコード別エラー数
      └── circuit_breaker_trips: サーキットブレーカー発動回数

  (3) Duration（レイテンシ）
      ├── request_duration_seconds: リクエスト処理時間
      │   ├── P50（中央値）
      │   ├── P95
      │   ├── P99
      │   └── P99.9
      ├── upstream_response_time: アップストリーム応答時間
      └── gateway_processing_time: ゲートウェイ自体の処理時間

  (4) Saturation（飽和度）
      ├── active_connections: アクティブ接続数
      ├── connection_pool_usage: コネクションプール使用率
      ├── rate_limit_remaining: レート制限残量
      └── memory_usage: メモリ使用量
```

### 12.2 Prometheus メトリクス収集（Express ミドルウェア）

```javascript
// Prometheus メトリクス収集ミドルウェア
const promClient = require('prom-client');

// デフォルトメトリクス（CPU, メモリ, イベントループ）
promClient.collectDefaultMetrics({
  prefix: 'api_gateway_',
  gcDurationBuckets: [0.001, 0.01, 0.1, 1, 2, 5],
});

// カスタムメトリクス
const httpRequestDuration = new promClient.Histogram({
  name: 'api_gateway_http_request_duration_seconds',
  help: 'HTTP request duration in seconds',
  labelNames: ['method', 'route', 'status_code', 'service'],
  buckets: [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
});

const httpRequestTotal = new promClient.Counter({
  name: 'api_gateway_http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code', 'service'],
});

const activeConnections = new promClient.Gauge({
  name: 'api_gateway_active_connections',
  help: 'Number of active connections',
});

const circuitBreakerState = new promClient.Gauge({
  name: 'api_gateway_circuit_breaker_state',
  help: 'Circuit breaker state (0=closed, 1=half-open, 2=open)',
  labelNames: ['service'],
});

const rateLimitHits = new promClient.Counter({
  name: 'api_gateway_rate_limit_hits_total',
  help: 'Total number of rate limit hits',
  labelNames: ['identifier_type', 'endpoint'],
});

const upstreamResponseTime = new promClient.Histogram({
  name: 'api_gateway_upstream_response_time_seconds',
  help: 'Upstream service response time',
  labelNames: ['service'],
  buckets: [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],
});

// メトリクス収集ミドルウェア
function metricsMiddleware() {
  return (req, res, next) => {
    activeConnections.inc();
    const startTime = process.hrtime.bigint();

    // レスポンス完了時にメトリクス記録
    res.on('finish', () => {
      activeConnections.dec();
      const duration = Number(process.hrtime.bigint() - startTime) / 1e9;
      const route = req.route?.path || req.path;
      const service = req.headers['x-target-service'] || 'unknown';

      httpRequestDuration.observe(
        { method: req.method, route, status_code: res.statusCode, service },
        duration
      );

      httpRequestTotal.inc(
        { method: req.method, route, status_code: res.statusCode, service }
      );
    });

    next();
  };
}

// メトリクスエンドポイント
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', promClient.register.contentType);
  res.send(await promClient.register.metrics());
});

// Grafana ダッシュボード用の PromQL クエリ例
const grafanaQueries = {
  // RPS（リクエスト/秒）
  rps: 'rate(api_gateway_http_requests_total[5m])',

  // エラー率（5xx）
  errorRate: 'sum(rate(api_gateway_http_requests_total{status_code=~"5.."}[5m])) / sum(rate(api_gateway_http_requests_total[5m]))',

  // P99 レイテンシ
  p99Latency: 'histogram_quantile(0.99, sum(rate(api_gateway_http_request_duration_seconds_bucket[5m])) by (le, route))',

  // サービス別アップストリーム応答時間
  upstreamP95: 'histogram_quantile(0.95, sum(rate(api_gateway_upstream_response_time_seconds_bucket[5m])) by (le, service))',

  // サーキットブレーカー状態
  cbState: 'api_gateway_circuit_breaker_state',

  // レート制限発動率
  rateLimitRate: 'rate(api_gateway_rate_limit_hits_total[5m])',
};
```

### 12.3 分散トレーシング統合

```javascript
// OpenTelemetry による分散トレーシング
const { NodeSDK } = require('@opentelemetry/sdk-node');
const { OTLPTraceExporter } = require('@opentelemetry/exporter-trace-otlp-grpc');
const { HttpInstrumentation } = require('@opentelemetry/instrumentation-http');
const { ExpressInstrumentation } = require('@opentelemetry/instrumentation-express');
const { Resource } = require('@opentelemetry/resources');
const {
  SEMRESATTRS_SERVICE_NAME,
  SEMRESATTRS_SERVICE_VERSION,
  SEMRESATTRS_DEPLOYMENT_ENVIRONMENT,
} = require('@opentelemetry/semantic-conventions');

// SDK初期化
const sdk = new NodeSDK({
  resource: new Resource({
    [SEMRESATTRS_SERVICE_NAME]: 'api-gateway',
    [SEMRESATTRS_SERVICE_VERSION]: '1.0.0',
    [SEMRESATTRS_DEPLOYMENT_ENVIRONMENT]: process.env.NODE_ENV || 'development',
  }),
  traceExporter: new OTLPTraceExporter({
    url: process.env.OTEL_EXPORTER_OTLP_ENDPOINT || 'http://jaeger:4317',
  }),
  instrumentations: [
    new HttpInstrumentation({
      requestHook: (span, request) => {
        span.setAttribute('http.route', request.path);
        span.setAttribute('gateway.client_ip', request.headers['x-real-ip'] || request.ip);
      },
    }),
    new ExpressInstrumentation(),
  ],
});

sdk.start();

// トレースコンテキスト伝播ミドルウェア
function tracePropagation() {
  return (req, res, next) => {
    // W3C Trace Context ヘッダーを上流サービスに転送
    const traceParent = req.headers['traceparent'];
    const traceState = req.headers['tracestate'];

    if (traceParent) {
      req.traceContext = { traceParent, traceState };
    }

    // リクエストIDの設定（トレースIDと紐づけ）
    const requestId = req.headers['x-request-id'] || generateRequestId();
    req.headers['x-request-id'] = requestId;
    res.set('X-Request-ID', requestId);

    next();
  };
}

function generateRequestId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).substring(2, 10)}`;
}
```

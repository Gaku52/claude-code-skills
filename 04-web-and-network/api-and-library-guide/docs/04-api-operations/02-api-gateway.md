# APIゲートウェイ

> APIゲートウェイはマイクロサービスの統一エントリポイント。ルーティング、認証の一元化、レート制限、リクエスト変換、サーキットブレーカーまで、APIゲートウェイの設計・構築・運用を体系的に習得する。

## この章で学ぶこと

- [ ] APIゲートウェイの役割とアーキテクチャを理解する
- [ ] 主要なゲートウェイ製品の比較と選定基準を把握する
- [ ] BFF（Backend for Frontend）パターンを学ぶ

---

## 1. APIゲートウェイの役割

```
APIゲートウェイ = マイクロサービスへの単一エントリポイント

  クライアント → API Gateway → User Service
                             → Order Service
                             → Payment Service
                             → Notification Service

主要機能:
  ① ルーティング:
     → /users/* → User Service
     → /orders/* → Order Service
     → パスベース、ヘッダーベースのルーティング

  ② 認証・認可の一元化:
     → JWT検証をゲートウェイで実施
     → 各サービスは認証済みリクエストのみ受信
     → API Key の検証

  ③ レート制限:
     → グローバルなレート制限
     → クライアント/プラン別の制限

  ④ リクエスト/レスポンス変換:
     → ヘッダーの追加/削除
     → リクエストボディの変換
     → レスポンスの集約

  ⑤ ロードバランシング:
     → サービスインスタンス間の分散
     → ヘルスチェック

  ⑥ キャッシュ:
     → レスポンスキャッシュ
     → CDN統合

  ⑦ 監視・ログ:
     → メトリクス収集
     → アクセスログ
     → 分散トレーシング

  ⑧ サーキットブレーカー:
     → 障害サービスへのリクエストを遮断
     → フォールバックレスポンス
```

---

## 2. ゲートウェイ製品比較

```
                  AWS API GW    Kong         Nginx        Envoy
──────────────────────────────────────────────────────────────
タイプ            マネージド    OSS/商用      OSS/商用     OSS
デプロイ          サーバーレス  セルフホスト   セルフホスト  サイドカー
プロトコル        HTTP,WS       HTTP,gRPC     HTTP         HTTP,gRPC,TCP
プラグイン        Lambda連携    豊富(300+)    モジュール    フィルタ
設定              Console/CF    Admin API     conf         xDS API
K8s統合           ×             ○             Ingress      Istio
コスト            従量課金      Free/有料     Free/有料     Free
学習コスト        低            中            低           高
適用              AWS環境       汎用          汎用         メッシュ

選定基準:
  AWS環境 → AWS API Gateway
  K8s + プラグイン重視 → Kong
  シンプル + 高性能 → Nginx
  サービスメッシュ → Envoy + Istio
```

---

## 3. AWS API Gateway

```
AWS API Gateway の種類:

  ① HTTP API（推奨）:
     → 低コスト（REST API の 70% 安い）
     → 低レイテンシ
     → JWT認証、CORS、Lambda統合

  ② REST API:
     → 全機能（APIキー、Usage Plan、WAF統合）
     → リクエスト/レスポンス変換
     → キャッシュ機能

  ③ WebSocket API:
     → 双方向通信
     → チャット、リアルタイム通知

構成例（HTTP API + Lambda）:
  クライアント → CloudFront → API Gateway → Lambda → DynamoDB

  利点:
  ✓ インフラ管理不要
  ✓ 自動スケーリング
  ✓ 従量課金（リクエスト数ベース）
  ✓ IAM / Cognito 認証統合

  制限:
  → ペイロードサイズ: 10MB
  → タイムアウト: 30秒（HTTP API）、29秒（REST API）
  → リクエスト/秒: 10,000（デフォルト、引き上げ可能）
```

```yaml
# AWS SAM テンプレート
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Globals:
  Api:
    Cors:
      AllowOrigin: "'https://app.example.com'"
      AllowHeaders: "'Authorization,Content-Type'"
      AllowMethods: "'GET,POST,PUT,DELETE'"

Resources:
  ApiGateway:
    Type: AWS::Serverless::HttpApi
    Properties:
      StageName: v1
      Auth:
        DefaultAuthorizer: JwtAuthorizer
        Authorizers:
          JwtAuthorizer:
            JwtConfiguration:
              issuer: https://auth.example.com/
              audience:
                - https://api.example.com

  ListUsersFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: handlers/users.list
      Runtime: nodejs20.x
      Events:
        ListUsers:
          Type: HttpApi
          Properties:
            ApiId: !Ref ApiGateway
            Path: /users
            Method: GET
```

---

## 4. Kong Gateway

```yaml
# Kong の宣言的設定（kong.yml）
_format_version: "3.0"

services:
  - name: user-service
    url: http://user-service:3000
    routes:
      - name: users-route
        paths:
          - /api/v1/users
        strip_path: false
    plugins:
      - name: jwt
        config:
          secret_is_base64: false
      - name: rate-limiting
        config:
          minute: 100
          policy: redis
          redis_host: redis
      - name: cors
        config:
          origins:
            - https://app.example.com
          methods:
            - GET
            - POST
            - PUT
            - DELETE
          headers:
            - Authorization
            - Content-Type

  - name: order-service
    url: http://order-service:3000
    routes:
      - name: orders-route
        paths:
          - /api/v1/orders
    plugins:
      - name: rate-limiting
        config:
          minute: 50
```

---

## 5. サーキットブレーカー

```
サーキットブレーカーパターン:

  Closed（通常状態）:
  → リクエストを通常通り転送
  → 失敗をカウント

  Open（遮断状態）:
  → 一定数の失敗後に遷移
  → リクエストを即座に拒否（フォールバック返却）
  → ダウンストリームの回復を待つ

  Half-Open（半開状態）:
  → タイムアウト後に遷移
  → 限定的にリクエストを通す
  → 成功 → Closed に復帰
  → 失敗 → Open に戻る

  状態遷移:
  Closed --[失敗閾値超過]--> Open
  Open --[タイムアウト]--> Half-Open
  Half-Open --[成功]--> Closed
  Half-Open --[失敗]--> Open

パラメータ:
  → 失敗閾値: 5回連続失敗 or 50%エラー率
  → タイムアウト: 30秒後に Half-Open
  → 監視ウィンドウ: 直近10リクエスト
```

```javascript
// サーキットブレーカーの実装
class CircuitBreaker {
  constructor(options = {}) {
    this.failureThreshold = options.failureThreshold || 5;
    this.resetTimeout = options.resetTimeout || 30000;
    this.monitorWindow = options.monitorWindow || 10;
    this.state = 'CLOSED';
    this.failures = 0;
    this.lastFailureTime = null;
    this.successCount = 0;
  }

  async execute(fn, fallback) {
    if (this.state === 'OPEN') {
      if (Date.now() - this.lastFailureTime > this.resetTimeout) {
        this.state = 'HALF_OPEN';
      } else {
        return fallback ? fallback() : Promise.reject(new Error('Circuit is OPEN'));
      }
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      if (fallback) return fallback();
      throw error;
    }
  }

  onSuccess() {
    if (this.state === 'HALF_OPEN') {
      this.successCount++;
      if (this.successCount >= 3) {
        this.state = 'CLOSED';
        this.failures = 0;
        this.successCount = 0;
      }
    }
    this.failures = Math.max(0, this.failures - 1);
  }

  onFailure() {
    this.failures++;
    this.lastFailureTime = Date.now();
    this.successCount = 0;
    if (this.failures >= this.failureThreshold) {
      this.state = 'OPEN';
    }
  }
}

// 使用例
const userServiceBreaker = new CircuitBreaker({
  failureThreshold: 5,
  resetTimeout: 30000,
});

async function getUser(id) {
  return userServiceBreaker.execute(
    () => fetch(`http://user-service:3000/users/${id}`).then(r => r.json()),
    () => ({ id, name: 'Unknown', cached: true }), // フォールバック
  );
}
```

---

## 6. BFF（Backend for Frontend）

```
BFF パターン:
  → クライアントの種類ごとに専用のゲートウェイを配置

  Web Browser → Web BFF → User Service
  Mobile App  → Mobile BFF → Order Service
  Third Party → Public API GW → Payment Service

  Web BFF:
  → フル機能、リッチなレスポンス
  → SSR対応
  → Cookie認証

  Mobile BFF:
  → 軽量レスポンス（帯域節約）
  → プッシュ通知統合
  → Bearer Token認証

  Public API:
  → REST + ページネーション
  → API Key認証
  → レート制限が厳しい

利点:
  ✓ クライアントの要件に最適化
  ✓ フロントエンドチームが所有
  ✓ バックエンドサービスの変更を吸収

欠点:
  ✗ BFF自体の管理コスト
  ✗ ロジックの重複リスク
  → 共通ロジックはライブラリに抽出
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 役割 | ルーティング、認証、レート制限、監視 |
| AWS | HTTP API（低コスト）、REST API（全機能） |
| Kong | プラグイン豊富、K8s統合 |
| サーキットブレーカー | Closed → Open → Half-Open |
| BFF | クライアント種類別のゲートウェイ |

---

## 参考文献
1. Kong. "Kong Gateway Documentation." docs.konghq.com, 2024.
2. AWS. "API Gateway Documentation." docs.aws.amazon.com, 2024.
3. Richardson, C. "Microservices Patterns." Manning, 2018.

# API First 設計

> API First設計は実装前にAPIの契約を定義するアプローチ。OpenAPI仕様でAPIを先に設計し、フロントエンド・バックエンドが並行開発できる体制を構築する。スキーマ駆動開発によって型安全性・テスト自動化・ドキュメント生成を一体化し、チーム全体の生産性を飛躍的に向上させる手法である。

## この章で学ぶこと

- [ ] API First設計の哲学と利点を理解する
- [ ] OpenAPI（Swagger）仕様の書き方を把握する
- [ ] モックサーバーを活用した並行開発を学ぶ
- [ ] コード生成ツールチェーンの構築方法を習得する
- [ ] API設計レビューのプロセスと品質基準を理解する
- [ ] 実務プロジェクトでの導入ステップを把握する
- [ ] Contract Testing の実践方法を学ぶ
- [ ] Design-First ワークフローの組織的展開を理解する

---

## 1. API First とは

### 1.1 基本概念

```
API First = 「コード実装の前にAPIの設計を確定させる」

  従来のアプローチ（Code First）:
  バックエンド実装 → API仕様が確定 → フロントエンド開発
  → フロントエンドが待ちになる

  API First:
  API仕様を定義 → モックサーバー立ち上げ
  → バックエンド: 仕様に沿って実装
  → フロントエンド: モックサーバーで並行開発
  → 両者が合流して統合テスト

  利点:
  ✓ フロントエンドとバックエンドの並行開発
  ✓ 仕様書 = 唯一の信頼できる情報源（Single Source of Truth）
  ✓ API設計のレビューが容易
  ✓ コード生成による型安全なクライアント
  ✓ テストの自動生成
  ✓ ドキュメントの自動生成と常時最新化
  ✓ マイクロサービス間の契約の明確化
  ✓ 組織横断的なAPI標準の統一

  ツールチェーン:
  設計:      Stoplight Studio, Swagger Editor, Redocly
  仕様:      OpenAPI 3.1 (YAML/JSON)
  モック:    Prism, MSW, WireMock, Microcks
  コード生成: openapi-generator, orval, openapi-typescript
  ドキュメント: Redoc, Swagger UI, Scalar, Elements
  テスト:    Dredd, Schemathesis, Pact, Specmatic
  リント:    Spectral, Redocly CLI, vacuum
  ガバナンス: Optic, Bump.sh
```

### 1.2 Code First vs API First の詳細比較

```
┌─────────────────────┬──────────────────────┬──────────────────────┐
│ 観点                │ Code First           │ API First            │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ 開発開始            │ バックエンド実装から  │ API仕様定義から       │
│ フロントエンド開始  │ バックエンド完了後    │ モックで即座に開始    │
│ 仕様書管理          │ コードから自動生成    │ 仕様書がマスター      │
│ 設計レビュー        │ コードレビューに混在  │ 独立した設計レビュー  │
│ 型安全性            │ 手動定義が必要        │ 自動生成で保証        │
│ 変更管理            │ 実装差分から追跡      │ 仕様差分で明確        │
│ 学習コスト          │ 低い                  │ 中程度（OpenAPI習得） │
│ 初期コスト          │ 低い                  │ 中程度                │
│ 長期メンテナンス    │ 高い                  │ 低い                  │
│ チーム間合意        │ 曖昧になりがち        │ 明確な契約            │
│ テスト自動化        │ 手動セットアップ      │ 仕様から自動生成      │
│ ドキュメント鮮度    │ 乖離しがち            │ 常に最新              │
│ 適用規模            │ 小規模プロジェクト    │ 中〜大規模            │
│ マイクロサービス    │ 調整が困難            │ 契約駆動で最適        │
└─────────────────────┴──────────────────────┴──────────────────────┘
```

### 1.3 API First が解決する課題

```
問題1: フロントエンド・バックエンドの待ち合わせ
─────────────────────────────────────────
  Code First:
  Week 1-3: バックエンド実装
  Week 4-6: フロントエンド開発（バックエンドが終わるまで待機）
  Week 7:   統合テスト
  合計: 7週間

  API First:
  Week 1:   API仕様を共同設計
  Week 2-4: バックエンド実装 ←→ フロントエンド開発（並行）
  Week 5:   統合テスト
  合計: 5週間（約30%短縮）

問題2: 仕様とコードの乖離
─────────────────────────────────────────
  Code First:
  コード変更 → ドキュメント更新忘れ → 仕様書が古い → バグの温床

  API First:
  仕様書変更 → CI/CDで検証 → コード生成で反映 → 常に同期

問題3: マイクロサービス間の契約不整合
─────────────────────────────────────────
  Code First:
  サービスAが変更 → サービスBが壊れる → 本番障害

  API First:
  仕様変更をPR → Contract Test → 依存サービスへ通知 → 安全に移行

問題4: API設計の品質のバラつき
─────────────────────────────────────────
  Code First:
  開発者ごとに異なるAPI設計 → 一貫性がない

  API First:
  スタイルガイド + Linter → 設計レビュー → 統一されたAPI品質
```

### 1.4 API First の成熟度モデル

```
Level 0: Ad Hoc（場当たり的）
  - API仕様なし
  - 口頭やSlackでの仕様伝達
  - ドキュメントは手動で後から作成

Level 1: Design First（設計先行）
  - OpenAPIで仕様を先に書く
  - 仕様書からドキュメント生成
  - 手動でのコード実装

Level 2: Contract Driven（契約駆動）
  - モックサーバーで並行開発
  - コード生成の活用
  - Contract Testの導入

Level 3: Automated（自動化）
  - CI/CDでの仕様検証
  - Breaking Change自動検出
  - ドキュメント・SDK自動公開

Level 4: Governed（ガバナンス）
  - 組織全体のAPIスタイルガイド
  - Design System for APIs
  - API Catalog管理
  - メトリクスに基づく品質改善

目標: 新規プロジェクトはLevel 2以上で開始し、
      6ヶ月以内にLevel 3到達を目指す
```

---

## 2. OpenAPI 仕様

### 2.1 基本構造

```yaml
# openapi.yaml - OpenAPI 3.1 仕様書の完全な例
openapi: '3.1.0'
info:
  title: User Management API
  version: '1.0.0'
  description: |
    ユーザー管理のためのRESTful API。

    ## 概要
    このAPIは、ユーザーの登録・認証・プロフィール管理を提供します。

    ## 認証
    Bearer Token（JWT）による認証が必要です。
    `/auth/login` エンドポイントでトークンを取得してください。

    ## レート制限
    - 認証済みユーザー: 1000 req/min
    - 未認証: 100 req/min

    ## エラーハンドリング
    全てのエラーレスポンスは RFC 7807 Problem Details 形式に従います。
  contact:
    name: API Support
    email: api-support@example.com
    url: https://developer.example.com/support
  license:
    name: Apache 2.0
    url: https://www.apache.org/licenses/LICENSE-2.0.html
  termsOfService: https://example.com/terms

externalDocs:
  description: 詳細なAPI開発者ガイド
  url: https://developer.example.com/guide

servers:
  - url: https://api.example.com/v1
    description: Production
  - url: https://staging-api.example.com/v1
    description: Staging
  - url: http://localhost:3000/v1
    description: Local Development

tags:
  - name: Users
    description: ユーザー管理操作
  - name: Auth
    description: 認証・認可操作
  - name: Admin
    description: 管理者専用操作

paths:
  /users:
    get:
      summary: ユーザー一覧の取得
      description: |
        ページネーション付きのユーザー一覧を返します。
        フィルタリングとソートに対応しています。
      operationId: listUsers
      tags: [Users]
      parameters:
        - name: page
          in: query
          description: ページ番号（1始まり）
          schema:
            type: integer
            default: 1
            minimum: 1
        - name: per_page
          in: query
          description: 1ページあたりの件数
          schema:
            type: integer
            default: 20
            minimum: 1
            maximum: 100
        - name: sort
          in: query
          description: ソートフィールド
          schema:
            type: string
            enum: [name, email, created_at, updated_at]
            default: created_at
        - name: order
          in: query
          description: ソート順
          schema:
            type: string
            enum: [asc, desc]
            default: desc
        - name: search
          in: query
          description: 名前・メールでの検索（部分一致）
          schema:
            type: string
            maxLength: 100
        - name: role
          in: query
          description: ロールでフィルタ
          schema:
            type: string
            enum: [user, admin, moderator]
        - name: status
          in: query
          description: ステータスでフィルタ
          schema:
            type: string
            enum: [active, inactive, suspended]
      responses:
        '200':
          description: ユーザー一覧
          headers:
            X-Total-Count:
              description: 総件数
              schema:
                type: integer
            X-RateLimit-Remaining:
              description: 残りリクエスト数
              schema:
                type: integer
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserListResponse'
              examples:
                default:
                  summary: 標準レスポンス例
                  value:
                    data:
                      - id: "550e8400-e29b-41d4-a716-446655440000"
                        name: "田中太郎"
                        email: "tanaka@example.com"
                        role: "admin"
                        status: "active"
                        createdAt: "2024-01-15T09:00:00Z"
                    meta:
                      total: 150
                      page: 1
                      per_page: 20
                      total_pages: 8
        '401':
          $ref: '#/components/responses/Unauthorized'
        '429':
          $ref: '#/components/responses/TooManyRequests'

    post:
      summary: ユーザーの作成
      description: |
        新しいユーザーを作成します。
        メールアドレスはシステム全体で一意である必要があります。
      operationId: createUser
      tags: [Users]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
            examples:
              basic:
                summary: 基本的なユーザー作成
                value:
                  name: "山田花子"
                  email: "yamada@example.com"
              withRole:
                summary: ロール指定でユーザー作成
                value:
                  name: "佐藤次郎"
                  email: "sato@example.com"
                  role: "moderator"
                  profile:
                    bio: "エンジニアリングマネージャー"
                    avatarUrl: "https://example.com/avatars/sato.png"
      responses:
        '201':
          description: ユーザー作成成功
          headers:
            Location:
              description: 作成されたリソースのURL
              schema:
                type: string
                format: uri
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserResponse'
        '409':
          description: メールアドレスが既に使用されている
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '422':
          $ref: '#/components/responses/ValidationError'

  /users/{userId}:
    get:
      summary: ユーザー詳細の取得
      operationId: getUser
      tags: [Users]
      parameters:
        - $ref: '#/components/parameters/UserId'
      responses:
        '200':
          description: ユーザー詳細
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserResponse'
        '404':
          $ref: '#/components/responses/NotFound'

    put:
      summary: ユーザー情報の更新
      description: ユーザー情報を完全に置換します。
      operationId: updateUser
      tags: [Users]
      parameters:
        - $ref: '#/components/parameters/UserId'
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UpdateUserRequest'
      responses:
        '200':
          description: 更新成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserResponse'
        '404':
          $ref: '#/components/responses/NotFound'
        '409':
          description: メールアドレスが既に使用されている
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '422':
          $ref: '#/components/responses/ValidationError'

    patch:
      summary: ユーザー情報の部分更新
      description: 指定されたフィールドのみを更新します。
      operationId: patchUser
      tags: [Users]
      parameters:
        - $ref: '#/components/parameters/UserId'
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/PatchUserRequest'
      responses:
        '200':
          description: 部分更新成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserResponse'
        '404':
          $ref: '#/components/responses/NotFound'

    delete:
      summary: ユーザーの削除
      description: |
        ユーザーを論理削除します。
        削除後30日以内であれば復元可能です。
      operationId: deleteUser
      tags: [Users]
      parameters:
        - $ref: '#/components/parameters/UserId'
      responses:
        '204':
          description: 削除成功
        '404':
          $ref: '#/components/responses/NotFound'

  /auth/login:
    post:
      summary: ログイン
      operationId: login
      tags: [Auth]
      security: []  # 認証不要
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [email, password]
              properties:
                email:
                  type: string
                  format: email
                password:
                  type: string
                  minLength: 8
      responses:
        '200':
          description: ログイン成功
          content:
            application/json:
              schema:
                type: object
                properties:
                  access_token:
                    type: string
                  refresh_token:
                    type: string
                  expires_in:
                    type: integer
                    description: アクセストークンの有効期限（秒）
                  token_type:
                    type: string
                    enum: [Bearer]
        '401':
          description: 認証失敗
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /auth/refresh:
    post:
      summary: トークンリフレッシュ
      operationId: refreshToken
      tags: [Auth]
      security: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [refresh_token]
              properties:
                refresh_token:
                  type: string
      responses:
        '200':
          description: トークン更新成功
          content:
            application/json:
              schema:
                type: object
                properties:
                  access_token:
                    type: string
                  expires_in:
                    type: integer

  /users/{userId}/avatar:
    put:
      summary: アバター画像のアップロード
      operationId: uploadAvatar
      tags: [Users]
      parameters:
        - $ref: '#/components/parameters/UserId'
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                file:
                  type: string
                  format: binary
                  description: "画像ファイル（JPEG, PNG, WebP）最大5MB"
            encoding:
              file:
                contentType: image/jpeg, image/png, image/webp
      responses:
        '200':
          description: アップロード成功
          content:
            application/json:
              schema:
                type: object
                properties:
                  url:
                    type: string
                    format: uri
        '413':
          description: ファイルサイズ超過
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

components:
  parameters:
    UserId:
      name: userId
      in: path
      required: true
      description: ユーザーのUUID
      schema:
        type: string
        format: uuid
      example: "550e8400-e29b-41d4-a716-446655440000"

  schemas:
    User:
      type: object
      required: [id, name, email, role, status, createdAt]
      properties:
        id:
          type: string
          format: uuid
          readOnly: true
        name:
          type: string
          minLength: 1
          maxLength: 100
          description: ユーザーの表示名
        email:
          type: string
          format: email
          description: メールアドレス（一意）
        role:
          type: string
          enum: [user, admin, moderator]
          default: user
          description: ユーザーのロール
        status:
          type: string
          enum: [active, inactive, suspended]
          default: active
          description: アカウントのステータス
        profile:
          $ref: '#/components/schemas/UserProfile'
        createdAt:
          type: string
          format: date-time
          readOnly: true
        updatedAt:
          type: string
          format: date-time
          readOnly: true

    UserProfile:
      type: object
      properties:
        bio:
          type: string
          maxLength: 500
          description: 自己紹介文
        avatarUrl:
          type: string
          format: uri
          description: アバター画像のURL
        location:
          type: string
          maxLength: 100
        website:
          type: string
          format: uri
        socialLinks:
          type: object
          properties:
            twitter:
              type: string
            github:
              type: string
            linkedin:
              type: string

    CreateUserRequest:
      type: object
      required: [name, email]
      properties:
        name:
          type: string
          minLength: 1
          maxLength: 100
        email:
          type: string
          format: email
        role:
          type: string
          enum: [user, admin, moderator]
          default: user
        profile:
          $ref: '#/components/schemas/UserProfile'

    UpdateUserRequest:
      type: object
      required: [name, email]
      properties:
        name:
          type: string
          minLength: 1
          maxLength: 100
        email:
          type: string
          format: email
        role:
          type: string
          enum: [user, admin, moderator]
        profile:
          $ref: '#/components/schemas/UserProfile'

    PatchUserRequest:
      type: object
      properties:
        name:
          type: string
          minLength: 1
          maxLength: 100
        email:
          type: string
          format: email
        role:
          type: string
          enum: [user, admin, moderator]
        status:
          type: string
          enum: [active, inactive, suspended]
        profile:
          $ref: '#/components/schemas/UserProfile'
      minProperties: 1

    UserResponse:
      type: object
      properties:
        data:
          $ref: '#/components/schemas/User'

    UserListResponse:
      type: object
      properties:
        data:
          type: array
          items:
            $ref: '#/components/schemas/User'
        meta:
          $ref: '#/components/schemas/PaginationMeta'
        links:
          $ref: '#/components/schemas/PaginationLinks'

    PaginationMeta:
      type: object
      properties:
        total:
          type: integer
          description: 総件数
        page:
          type: integer
          description: 現在のページ番号
        per_page:
          type: integer
          description: 1ページあたりの件数
        total_pages:
          type: integer
          description: 総ページ数

    PaginationLinks:
      type: object
      properties:
        self:
          type: string
          format: uri
        first:
          type: string
          format: uri
        last:
          type: string
          format: uri
        prev:
          type: string
          format: uri
          nullable: true
        next:
          type: string
          format: uri
          nullable: true

    Error:
      type: object
      required: [type, title, status]
      properties:
        type:
          type: string
          format: uri
          description: エラータイプを識別するURI
        title:
          type: string
          description: エラーの概要
        status:
          type: integer
          description: HTTPステータスコード
        detail:
          type: string
          description: エラーの詳細説明
        instance:
          type: string
          format: uri
          description: エラーが発生した具体的なリソース
        errors:
          type: array
          items:
            type: object
            properties:
              field:
                type: string
              message:
                type: string
              code:
                type: string
          description: フィールド単位のバリデーションエラー

  responses:
    Unauthorized:
      description: 認証エラー
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            type: "https://api.example.com/errors/unauthorized"
            title: "Unauthorized"
            status: 401
            detail: "認証トークンが無効または期限切れです"

    NotFound:
      description: リソースが見つからない
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            type: "https://api.example.com/errors/not-found"
            title: "Not Found"
            status: 404
            detail: "指定されたリソースは存在しません"

    ValidationError:
      description: バリデーションエラー
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            type: "https://api.example.com/errors/validation"
            title: "Validation Error"
            status: 422
            detail: "入力データにエラーがあります"
            errors:
              - field: "email"
                message: "有効なメールアドレスを入力してください"
                code: "invalid_format"

    TooManyRequests:
      description: レート制限超過
      headers:
        Retry-After:
          description: 再試行までの秒数
          schema:
            type: integer
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            type: "https://api.example.com/errors/rate-limit"
            title: "Too Many Requests"
            status: 429
            detail: "レート制限を超えました。60秒後に再試行してください"

  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: |
        JWTベースの認証。`/auth/login` でトークンを取得してください。
        トークンは1時間で期限切れとなります。
    apiKey:
      type: apiKey
      in: header
      name: X-API-Key
      description: サービス間通信用のAPIキー

security:
  - bearerAuth: []
```

### 2.2 OpenAPI 3.1 の重要な機能

```yaml
# 1. JSON Schema との完全互換
# OpenAPI 3.1 は JSON Schema Draft 2020-12 と完全互換
components:
  schemas:
    # if/then/else が使える
    Payment:
      type: object
      properties:
        method:
          type: string
          enum: [credit_card, bank_transfer, crypto]
        cardNumber:
          type: string
        bankAccount:
          type: string
      if:
        properties:
          method:
            const: credit_card
      then:
        required: [cardNumber]
      else:
        if:
          properties:
            method:
              const: bank_transfer
        then:
          required: [bankAccount]

    # prefixItems (旧 tuple validation)
    Coordinate:
      type: array
      prefixItems:
        - type: number
          description: 緯度
        - type: number
          description: 経度
      minItems: 2
      maxItems: 2

    # contentEncoding, contentMediaType
    FileUpload:
      type: object
      properties:
        content:
          type: string
          contentEncoding: base64
          contentMediaType: image/png

# 2. Webhooks
webhooks:
  userCreated:
    post:
      summary: ユーザー作成時のWebhook
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                event:
                  type: string
                  const: user.created
                data:
                  $ref: '#/components/schemas/User'
                timestamp:
                  type: string
                  format: date-time
      responses:
        '200':
          description: Webhook受信確認

  userDeleted:
    post:
      summary: ユーザー削除時のWebhook
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                event:
                  type: string
                  const: user.deleted
                data:
                  type: object
                  properties:
                    userId:
                      type: string
                      format: uuid
                timestamp:
                  type: string
                  format: date-time
      responses:
        '200':
          description: Webhook受信確認

# 3. パスアイテムの $ref
paths:
  /users:
    $ref: './paths/users.yaml'
  /users/{userId}:
    $ref: './paths/users-by-id.yaml'
```

### 2.3 仕様ファイルの分割管理

```
プロジェクト規模が大きくなると、1ファイルでの管理は困難になる。
ファイル分割のベストプラクティス:

api/
├── openapi.yaml          # ルートファイル（$refで各ファイルを参照）
├── info.yaml             # API情報（title, description, version）
├── paths/
│   ├── users.yaml        # /users パス定義
│   ├── users-by-id.yaml  # /users/{userId} パス定義
│   ├── auth.yaml         # /auth/* パス定義
│   └── admin.yaml        # /admin/* パス定義
├── schemas/
│   ├── user.yaml         # User関連スキーマ
│   ├── auth.yaml         # Auth関連スキーマ
│   ├── common.yaml       # 共通スキーマ（Error, Pagination）
│   └── admin.yaml        # Admin関連スキーマ
├── parameters/
│   ├── path.yaml         # パスパラメータ
│   └── query.yaml        # クエリパラメータ
├── responses/
│   └── errors.yaml       # 共通エラーレスポンス
└── examples/
    ├── users.yaml        # ユーザー関連の例
    └── errors.yaml       # エラーレスポンスの例
```

```yaml
# api/openapi.yaml（ルートファイル）
openapi: '3.1.0'
info:
  $ref: './info.yaml'
servers:
  - url: https://api.example.com/v1
    description: Production
paths:
  /users:
    $ref: './paths/users.yaml'
  /users/{userId}:
    $ref: './paths/users-by-id.yaml'
  /auth/login:
    $ref: './paths/auth.yaml#/login'
components:
  schemas:
    User:
      $ref: './schemas/user.yaml#/User'
    Error:
      $ref: './schemas/common.yaml#/Error'
```

```yaml
# api/schemas/user.yaml（分割されたスキーマファイル）
User:
  type: object
  required: [id, name, email, createdAt]
  properties:
    id:
      type: string
      format: uuid
    name:
      type: string
      minLength: 1
      maxLength: 100
    email:
      type: string
      format: email
    role:
      type: string
      enum: [user, admin, moderator]
    createdAt:
      type: string
      format: date-time

CreateUserRequest:
  type: object
  required: [name, email]
  properties:
    name:
      type: string
      minLength: 1
      maxLength: 100
    email:
      type: string
      format: email

UserResponse:
  type: object
  properties:
    data:
      $ref: '#/User'
```

```bash
# 分割ファイルの結合（バンドル）
# Redocly CLIを使用
npx @redocly/cli bundle api/openapi.yaml -o dist/openapi.yaml

# Swagger CLI を使用
npx swagger-cli bundle api/openapi.yaml -o dist/openapi.yaml -t yaml

# バリデーション
npx @redocly/cli lint api/openapi.yaml
npx swagger-cli validate api/openapi.yaml
```

---

## 3. コード生成

### 3.1 TypeScript 型生成

```bash
# openapi-typescript: OpenAPIからTypeScript型を生成
npm install -D openapi-typescript

# 型生成の実行
npx openapi-typescript openapi.yaml -o src/api/types.ts

# watchモード（仕様変更を自動検知）
npx openapi-typescript openapi.yaml -o src/api/types.ts --watch
```

```typescript
// 生成された型の使用例（src/api/types.ts から）
import type { paths, components } from './types';

// リクエスト型の取得
type CreateUserBody = paths['/users']['post']['requestBody']['content']['application/json'];
// => { name: string; email: string; role?: 'user' | 'admin' | 'moderator'; }

// レスポンス型の取得
type UserListResponse = paths['/users']['get']['responses']['200']['content']['application/json'];

// スキーマ型の直接参照
type User = components['schemas']['User'];
type Error = components['schemas']['Error'];

// クエリパラメータ型
type ListUsersParams = paths['/users']['get']['parameters']['query'];
```

```typescript
// openapi-fetch: 型安全なFetchクライアント
import createClient from 'openapi-fetch';
import type { paths } from './types';

const client = createClient<paths>({
  baseUrl: 'https://api.example.com/v1',
  headers: {
    Authorization: `Bearer ${token}`,
  },
});

// 完全に型安全なAPIコール
// パス、メソッド、パラメータ、レスポンスすべてが型チェックされる
const { data, error } = await client.GET('/users', {
  params: {
    query: {
      page: 1,
      per_page: 20,
      sort: 'created_at',  // enum から選択
      role: 'admin',       // enum から選択
    },
  },
});

if (data) {
  // data は UserListResponse 型として推論される
  data.data.forEach(user => {
    console.log(user.name);   // string
    console.log(user.email);  // string
    console.log(user.role);   // 'user' | 'admin' | 'moderator'
  });
}

// ユーザー作成（リクエストボディも型チェック）
const { data: newUser, error: createError } = await client.POST('/users', {
  body: {
    name: '田中太郎',
    email: 'tanaka@example.com',
    // role: 'invalid' // ← コンパイルエラー！
  },
});

// パスパラメータも型安全
const { data: user } = await client.GET('/users/{userId}', {
  params: {
    path: { userId: '550e8400-e29b-41d4-a716-446655440000' },
  },
});
```

### 3.2 orval によるクライアント生成

```typescript
// orval.config.ts
import { defineConfig } from 'orval';

export default defineConfig({
  userApi: {
    input: {
      target: './openapi.yaml',
      validation: true,
    },
    output: {
      target: './src/api/generated.ts',
      client: 'react-query',  // TanStack Query のフック生成
      mode: 'tags-split',     // タグごとにファイル分割
      schemas: './src/api/models',
      mock: true,             // MSW用モックも同時生成
      override: {
        mutator: {
          path: './src/api/custom-fetch.ts',
          name: 'customFetch',
        },
        query: {
          useQuery: true,
          useMutation: true,
          signal: true,
        },
        // Zodバリデーションスキーマも生成
        zod: {
          strict: {
            response: true,
            body: true,
          },
        },
      },
    },
    hooks: {
      afterAllFilesWrite: 'prettier --write',
    },
  },
});
```

```bash
# orval の実行
npx orval

# watch モード
npx orval --watch
```

```typescript
// 生成されたReact Queryフックの使用例
import { useListUsers, useCreateUser, useGetUser } from './api/generated';

function UserList() {
  // 自動生成されたフック（キャッシュキー、型すべて自動）
  const { data, isLoading, error } = useListUsers({
    page: 1,
    per_page: 20,
    role: 'admin',
  });

  const createUser = useCreateUser();

  const handleCreate = async () => {
    await createUser.mutateAsync({
      data: {
        name: '新規ユーザー',
        email: 'new@example.com',
      },
    });
  };

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div>
      {data?.data?.map(user => (
        <UserCard key={user.id} user={user} />
      ))}
      <button onClick={handleCreate}>追加</button>
    </div>
  );
}

function UserDetail({ userId }: { userId: string }) {
  // パスパラメータも型安全
  const { data } = useGetUser(userId);

  return <div>{data?.data?.name}</div>;
}
```

### 3.3 サーバーサイドコード生成

```bash
# Go サーバースタブ生成（oapi-codegen）
go install github.com/deepmap/oapi-codegen/cmd/oapi-codegen@latest

oapi-codegen \
  -generate types,server,spec \
  -package api \
  -o server/api/api.gen.go \
  openapi.yaml
```

```go
// 生成されたインターフェースの実装（Go）
package api

import (
    "net/http"
    "github.com/labstack/echo/v4"
)

// 生成されたインターフェース
type ServerInterface interface {
    ListUsers(ctx echo.Context, params ListUsersParams) error
    CreateUser(ctx echo.Context) error
    GetUser(ctx echo.Context, userId string) error
    UpdateUser(ctx echo.Context, userId string) error
    DeleteUser(ctx echo.Context, userId string) error
}

// 実装
type UserHandler struct {
    userService UserService
}

func (h *UserHandler) ListUsers(ctx echo.Context, params ListUsersParams) error {
    users, total, err := h.userService.List(ctx.Request().Context(), ListOptions{
        Page:    params.Page,
        PerPage: params.PerPage,
        Sort:    params.Sort,
        Order:   params.Order,
        Search:  params.Search,
        Role:    params.Role,
    })
    if err != nil {
        return ctx.JSON(http.StatusInternalServerError, Error{
            Type:   "https://api.example.com/errors/internal",
            Title:  "Internal Server Error",
            Status: 500,
        })
    }

    totalPages := (total + *params.PerPage - 1) / *params.PerPage
    return ctx.JSON(http.StatusOK, UserListResponse{
        Data: users,
        Meta: PaginationMeta{
            Total:      total,
            Page:       *params.Page,
            PerPage:    *params.PerPage,
            TotalPages: totalPages,
        },
    })
}

func (h *UserHandler) CreateUser(ctx echo.Context) error {
    var req CreateUserRequest
    if err := ctx.Bind(&req); err != nil {
        return ctx.JSON(http.StatusUnprocessableEntity, Error{
            Type:   "https://api.example.com/errors/validation",
            Title:  "Validation Error",
            Status: 422,
            Detail: err.Error(),
        })
    }

    user, err := h.userService.Create(ctx.Request().Context(), req)
    if err != nil {
        return handleServiceError(ctx, err)
    }

    return ctx.JSON(http.StatusCreated, UserResponse{Data: user})
}
```

```bash
# Python サーバースタブ生成（FastAPI）
pip install openapi-generator-cli

openapi-generator-cli generate \
  -i openapi.yaml \
  -g python-fastapi \
  -o server/ \
  --additional-properties=packageName=user_api
```

```python
# 生成されたFastAPIサーバーの拡張例
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel, EmailStr
from typing import Optional
from uuid import UUID

# 生成されたモデル
class User(BaseModel):
    id: UUID
    name: str
    email: EmailStr
    role: str = "user"
    status: str = "active"
    created_at: str
    updated_at: Optional[str] = None

class CreateUserRequest(BaseModel):
    name: str
    email: EmailStr
    role: Optional[str] = "user"

class UserResponse(BaseModel):
    data: User

class PaginationMeta(BaseModel):
    total: int
    page: int
    per_page: int
    total_pages: int

class UserListResponse(BaseModel):
    data: list[User]
    meta: PaginationMeta

# エンドポイント実装
app = FastAPI(title="User Management API", version="1.0.0")

@app.get("/v1/users", response_model=UserListResponse)
async def list_users(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    sort: Optional[str] = Query("created_at", enum=["name", "email", "created_at"]),
    order: Optional[str] = Query("desc", enum=["asc", "desc"]),
    search: Optional[str] = Query(None, max_length=100),
    role: Optional[str] = Query(None, enum=["user", "admin", "moderator"]),
    user_service: UserService = Depends(get_user_service),
):
    users, total = await user_service.list_users(
        page=page, per_page=per_page, sort=sort,
        order=order, search=search, role=role,
    )
    return UserListResponse(
        data=users,
        meta=PaginationMeta(
            total=total,
            page=page,
            per_page=per_page,
            total_pages=(total + per_page - 1) // per_page,
        ),
    )

@app.post("/v1/users", response_model=UserResponse, status_code=201)
async def create_user(
    body: CreateUserRequest,
    user_service: UserService = Depends(get_user_service),
):
    user = await user_service.create_user(body)
    return UserResponse(data=user)
```

### 3.4 コード生成のCI/CDパイプライン

```yaml
# .github/workflows/api-codegen.yml
name: API Code Generation

on:
  push:
    paths:
      - 'api/openapi.yaml'
      - 'api/**/*.yaml'
  pull_request:
    paths:
      - 'api/openapi.yaml'
      - 'api/**/*.yaml'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Validate OpenAPI Spec
        run: |
          npx @redocly/cli lint api/openapi.yaml

      - name: Check for breaking changes
        run: |
          npx @opticdev/optic diff api/openapi.yaml \
            --base origin/main \
            --check
        if: github.event_name == 'pull_request'

  generate:
    needs: validate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Generate TypeScript types
        run: |
          npx openapi-typescript api/openapi.yaml \
            -o frontend/src/api/types.ts

      - name: Generate API client
        run: |
          cd frontend && npx orval

      - name: Generate Go server stubs
        run: |
          go install github.com/deepmap/oapi-codegen/cmd/oapi-codegen@latest
          oapi-codegen -generate types,server \
            -package api \
            -o backend/api/api.gen.go \
            api/openapi.yaml

      - name: Commit generated code
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add -A
          git diff --staged --quiet || \
            git commit -m "chore: regenerate API code from OpenAPI spec"
          git push
```

---

## 4. モックサーバー

### 4.1 Prism によるモックサーバー

```bash
# Prismのインストールと起動
npm install -D @stoplight/prism-cli

# モックサーバー起動（OpenAPI仕様からレスポンスを自動生成）
npx prism mock openapi.yaml
# → http://localhost:4010 でモックAPIが起動

# 動的モック（リクエストに基づいてレスポンスを変化）
npx prism mock openapi.yaml --dynamic

# バリデーションプロキシモード
# 実際のAPIに対するリクエスト/レスポンスが仕様に準拠しているか検証
npx prism proxy openapi.yaml https://api.example.com/v1
```

```bash
# Prism モックサーバーへのリクエスト例
# ユーザー一覧取得
curl http://localhost:4010/users?page=1&per_page=10

# ユーザー作成
curl -X POST http://localhost:4010/users \
  -H "Content-Type: application/json" \
  -d '{"name": "テスト", "email": "test@example.com"}'

# バリデーションエラーの確認
curl -X POST http://localhost:4010/users \
  -H "Content-Type: application/json" \
  -d '{"name": ""}'
# → 422 Validation Error が返る

# 特定のレスポンス例を指定
curl http://localhost:4010/users \
  -H "Prefer: example=empty_list"
```

### 4.2 MSW（Mock Service Worker）

```typescript
// msw/handlers.ts - フロントエンド開発用のモックハンドラ
import { http, HttpResponse, delay } from 'msw';

// OpenAPI仕様に基づいたモックデータ
const mockUsers = [
  {
    id: '550e8400-e29b-41d4-a716-446655440000',
    name: '田中太郎',
    email: 'tanaka@example.com',
    role: 'admin',
    status: 'active',
    createdAt: '2024-01-15T09:00:00Z',
    updatedAt: '2024-06-01T12:00:00Z',
  },
  {
    id: '550e8400-e29b-41d4-a716-446655440001',
    name: '山田花子',
    email: 'yamada@example.com',
    role: 'user',
    status: 'active',
    createdAt: '2024-02-20T10:30:00Z',
    updatedAt: '2024-05-15T08:00:00Z',
  },
  {
    id: '550e8400-e29b-41d4-a716-446655440002',
    name: '佐藤次郎',
    email: 'sato@example.com',
    role: 'moderator',
    status: 'inactive',
    createdAt: '2024-03-10T14:00:00Z',
    updatedAt: null,
  },
];

export const handlers = [
  // ユーザー一覧
  http.get('https://api.example.com/v1/users', async ({ request }) => {
    await delay(200); // リアルな遅延をシミュレート

    const url = new URL(request.url);
    const page = parseInt(url.searchParams.get('page') || '1');
    const perPage = parseInt(url.searchParams.get('per_page') || '20');
    const search = url.searchParams.get('search');
    const role = url.searchParams.get('role');

    let filtered = [...mockUsers];

    // フィルタリング
    if (search) {
      filtered = filtered.filter(u =>
        u.name.includes(search) || u.email.includes(search)
      );
    }
    if (role) {
      filtered = filtered.filter(u => u.role === role);
    }

    // ページネーション
    const total = filtered.length;
    const start = (page - 1) * perPage;
    const paged = filtered.slice(start, start + perPage);

    return HttpResponse.json({
      data: paged,
      meta: {
        total,
        page,
        per_page: perPage,
        total_pages: Math.ceil(total / perPage),
      },
      links: {
        self: `/users?page=${page}&per_page=${perPage}`,
        first: `/users?page=1&per_page=${perPage}`,
        last: `/users?page=${Math.ceil(total / perPage)}&per_page=${perPage}`,
        prev: page > 1 ? `/users?page=${page - 1}&per_page=${perPage}` : null,
        next: page < Math.ceil(total / perPage)
          ? `/users?page=${page + 1}&per_page=${perPage}`
          : null,
      },
    });
  }),

  // ユーザー作成
  http.post('https://api.example.com/v1/users', async ({ request }) => {
    await delay(300);

    const body = await request.json() as { name: string; email: string; role?: string };

    // バリデーション
    if (!body.name || body.name.length === 0) {
      return HttpResponse.json(
        {
          type: 'https://api.example.com/errors/validation',
          title: 'Validation Error',
          status: 422,
          detail: '入力データにエラーがあります',
          errors: [
            { field: 'name', message: '名前は必須です', code: 'required' },
          ],
        },
        { status: 422 }
      );
    }

    // 重複チェック
    if (mockUsers.some(u => u.email === body.email)) {
      return HttpResponse.json(
        {
          type: 'https://api.example.com/errors/conflict',
          title: 'Conflict',
          status: 409,
          detail: 'このメールアドレスは既に使用されています',
        },
        { status: 409 }
      );
    }

    const newUser = {
      id: crypto.randomUUID(),
      name: body.name,
      email: body.email,
      role: body.role || 'user',
      status: 'active' as const,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    mockUsers.push(newUser);

    return HttpResponse.json(
      { data: newUser },
      {
        status: 201,
        headers: {
          Location: `/users/${newUser.id}`,
        },
      }
    );
  }),

  // ユーザー詳細
  http.get('https://api.example.com/v1/users/:userId', async ({ params }) => {
    await delay(150);

    const user = mockUsers.find(u => u.id === params.userId);
    if (!user) {
      return HttpResponse.json(
        {
          type: 'https://api.example.com/errors/not-found',
          title: 'Not Found',
          status: 404,
          detail: '指定されたユーザーは存在しません',
        },
        { status: 404 }
      );
    }

    return HttpResponse.json({ data: user });
  }),

  // ユーザー削除
  http.delete('https://api.example.com/v1/users/:userId', async ({ params }) => {
    await delay(200);

    const index = mockUsers.findIndex(u => u.id === params.userId);
    if (index === -1) {
      return HttpResponse.json(
        {
          type: 'https://api.example.com/errors/not-found',
          title: 'Not Found',
          status: 404,
          detail: '指定されたユーザーは存在しません',
        },
        { status: 404 }
      );
    }

    mockUsers.splice(index, 1);
    return new HttpResponse(null, { status: 204 });
  }),

  // 認証
  http.post('https://api.example.com/v1/auth/login', async ({ request }) => {
    await delay(500);

    const body = await request.json() as { email: string; password: string };

    if (body.email === 'admin@example.com' && body.password === 'password123') {
      return HttpResponse.json({
        access_token: 'mock-jwt-token-xxxxx',
        refresh_token: 'mock-refresh-token-xxxxx',
        expires_in: 3600,
        token_type: 'Bearer',
      });
    }

    return HttpResponse.json(
      {
        type: 'https://api.example.com/errors/unauthorized',
        title: 'Unauthorized',
        status: 401,
        detail: 'メールアドレスまたはパスワードが正しくありません',
      },
      { status: 401 }
    );
  }),
];
```

```typescript
// msw/browser.ts - ブラウザ環境でのセットアップ
import { setupWorker } from 'msw/browser';
import { handlers } from './handlers';

export const worker = setupWorker(...handlers);

// main.tsx でのセットアップ
async function enableMocking() {
  if (import.meta.env.DEV) {
    const { worker } = await import('./msw/browser');
    return worker.start({
      onUnhandledRequest: 'warn',
    });
  }
}

enableMocking().then(() => {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App />
    </StrictMode>
  );
});
```

```typescript
// msw/server.ts - テスト環境でのセットアップ
import { setupServer } from 'msw/node';
import { handlers } from './handlers';

export const server = setupServer(...handlers);

// vitest.setup.ts
import { beforeAll, afterEach, afterAll } from 'vitest';
import { server } from './msw/server';

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
```

### 4.3 WireMock による高度なモック

```json
// wiremock/mappings/get-users.json
{
  "request": {
    "method": "GET",
    "urlPathPattern": "/v1/users",
    "queryParameters": {
      "page": {
        "matches": "[0-9]+"
      }
    }
  },
  "response": {
    "status": 200,
    "headers": {
      "Content-Type": "application/json"
    },
    "jsonBody": {
      "data": [
        {
          "id": "550e8400-e29b-41d4-a716-446655440000",
          "name": "田中太郎",
          "email": "tanaka@example.com",
          "role": "admin"
        }
      ],
      "meta": {
        "total": 1,
        "page": 1,
        "per_page": 20,
        "total_pages": 1
      }
    }
  }
}
```

```json
// wiremock/mappings/create-user-validation-error.json
{
  "request": {
    "method": "POST",
    "urlPath": "/v1/users",
    "bodyPatterns": [
      {
        "matchesJsonPath": {
          "expression": "$.name",
          "absent": true
        }
      }
    ]
  },
  "response": {
    "status": 422,
    "headers": {
      "Content-Type": "application/json"
    },
    "jsonBody": {
      "type": "https://api.example.com/errors/validation",
      "title": "Validation Error",
      "status": 422,
      "errors": [
        {
          "field": "name",
          "message": "名前は必須です",
          "code": "required"
        }
      ]
    }
  },
  "priority": 1
}
```

```bash
# WireMock の起動
docker run -d \
  --name wiremock \
  -p 8080:8080 \
  -v $(pwd)/wiremock:/home/wiremock \
  wiremock/wiremock:latest

# 動作確認
curl http://localhost:8080/v1/users?page=1
```

---

## 5. API リンティングとスタイルガイド

### 5.1 Spectral によるリンティング

```yaml
# .spectral.yaml - APIリンティングルール
extends:
  - spectral:oas

rules:
  # カスタムルール: オペレーションIDの命名規則
  operation-id-casing:
    given: "$.paths[*][*].operationId"
    then:
      function: casing
      functionOptions:
        type: camel
    severity: error
    message: "operationIdはcamelCaseで記述してください"

  # カスタムルール: レスポンスにはdataラッパーを使う
  response-data-wrapper:
    given: "$.paths[*][get,post,put,patch].responses[200,201].content.application/json.schema"
    then:
      field: properties.data
      function: truthy
    severity: warn
    message: "レスポンスはdata プロパティでラップしてください"

  # カスタムルール: エラーレスポンスはRFC 7807形式
  error-response-format:
    given: "$.paths[*][*].responses[4XX,5XX].content.application/json.schema"
    then:
      - field: properties.type
        function: truthy
      - field: properties.title
        function: truthy
      - field: properties.status
        function: truthy
    severity: error
    message: "エラーレスポンスはRFC 7807 Problem Details形式にしてください"

  # カスタムルール: パスは複数形名詞
  path-plural-resource:
    given: "$.paths"
    then:
      function: pattern
      functionOptions:
        match: "^/[a-z]+s(/\\{[^}]+\\}(/[a-z]+s)?)*$"
    severity: warn
    message: "リソースパスは複数形の名詞を使ってください"

  # すべてのエンドポイントにタグを付ける
  operation-tag-defined:
    given: "$.paths[*][get,post,put,patch,delete]"
    then:
      field: tags
      function: length
      functionOptions:
        min: 1
    severity: error
    message: "すべてのオペレーションにタグを付けてください"

  # 説明文の必須化
  operation-description:
    given: "$.paths[*][get,post,put,patch,delete]"
    then:
      field: description
      function: truthy
    severity: warn
    message: "オペレーションにdescriptionを付けてください"

  # セキュリティの定義チェック
  security-defined:
    given: "$"
    then:
      field: security
      function: truthy
    severity: error
    message: "グローバルセキュリティを定義してください"

  # プロパティ名のケーシング
  property-casing:
    given: "$.components.schemas[*].properties[*]~"
    then:
      function: casing
      functionOptions:
        type: camel
    severity: error
    message: "プロパティ名はcamelCaseで記述してください"
```

```bash
# Spectralの実行
npx @stoplight/spectral-cli lint openapi.yaml

# 出力例:
# openapi.yaml
#  45:17  warning  operation-description   オペレーションにdescriptionを付けてください  paths./users.get
#  78:21  error    operation-id-casing     operationIdはcamelCaseで記述してください     paths./users.post
#
# ✖ 2 problems (1 error, 1 warning, 0 infos, 0 hints)
```

### 5.2 組織のAPIスタイルガイド定義

```yaml
# api-style-guide.yaml - 組織全体のAPIスタイルガイド
extends:
  - spectral:oas
  - .spectral.yaml  # プロジェクト固有ルール

rules:
  # === 命名規則 ===
  # URLはケバブケース
  paths-kebab-case:
    given: "$.paths[*]~"
    then:
      function: pattern
      functionOptions:
        match: "^(/[a-z][a-z0-9-]*(/\\{[a-zA-Z]+\\})?)+$"
    severity: error

  # === バージョニング ===
  # URLにバージョンを含める
  path-version:
    given: "$.servers[*].url"
    then:
      function: pattern
      functionOptions:
        match: "/v[0-9]+"
    severity: error
    message: "サーバーURLにバージョンを含めてください（例: /v1）"

  # === ページネーション ===
  # GETリストにはページネーション必須
  pagination-required:
    given: "$.paths[*].get.parameters"
    then:
      function: schema
      functionOptions:
        schema:
          type: array
          contains:
            type: object
            properties:
              name:
                const: page
    severity: warn
    message: "一覧エンドポイントにはpageパラメータを含めてください"

  # === セキュリティ ===
  # HTTPSのみ
  https-only:
    given: "$.servers[*].url"
    then:
      function: pattern
      functionOptions:
        match: "^https://|^http://localhost"
    severity: error
    message: "本番環境ではHTTPSを使用してください"

  # === レスポンス ===
  # 成功レスポンスの型定義必須
  response-schema-required:
    given: "$.paths[*][*].responses[2XX].content.application/json"
    then:
      field: schema
      function: truthy
    severity: error
```

---

## 6. Contract Testing

### 6.1 Pact によるConsumer-Driven Contract Testing

```typescript
// consumer/tests/user-api.pact.spec.ts
import { PactV4, MatchersV3 } from '@pact-foundation/pact';
import { UserApiClient } from '../src/api/user-client';

const { like, eachLike, uuid, iso8601DateTimeWithMillis } = MatchersV3;

const provider = new PactV4({
  consumer: 'FrontendApp',
  provider: 'UserService',
  dir: './pacts',
});

describe('User API Contract', () => {
  describe('GET /users', () => {
    it('ユーザー一覧を取得できる', async () => {
      await provider
        .addInteraction()
        .given('ユーザーが3人存在する')
        .uponReceiving('ユーザー一覧の取得リクエスト')
        .withRequest('GET', '/v1/users', (builder) => {
          builder.query({ page: '1', per_page: '20' });
          builder.headers({ Authorization: 'Bearer valid-token' });
        })
        .willRespondWith(200, (builder) => {
          builder.headers({ 'Content-Type': 'application/json' });
          builder.jsonBody({
            data: eachLike({
              id: uuid(),
              name: like('田中太郎'),
              email: like('tanaka@example.com'),
              role: like('user'),
              status: like('active'),
              createdAt: iso8601DateTimeWithMillis(),
            }),
            meta: {
              total: like(3),
              page: like(1),
              per_page: like(20),
              total_pages: like(1),
            },
          });
        })
        .executeTest(async (mockServer) => {
          const client = new UserApiClient(mockServer.url);
          const result = await client.listUsers({ page: 1, perPage: 20 });

          expect(result.data).toHaveLength(1);
          expect(result.meta.total).toBe(3);
        });
    });

    it('ユーザーを作成できる', async () => {
      await provider
        .addInteraction()
        .uponReceiving('ユーザー作成リクエスト')
        .withRequest('POST', '/v1/users', (builder) => {
          builder.headers({
            'Content-Type': 'application/json',
            Authorization: 'Bearer valid-token',
          });
          builder.jsonBody({
            name: '山田花子',
            email: 'yamada@example.com',
          });
        })
        .willRespondWith(201, (builder) => {
          builder.headers({
            'Content-Type': 'application/json',
            Location: like('/users/550e8400-e29b-41d4-a716-446655440000'),
          });
          builder.jsonBody({
            data: {
              id: uuid(),
              name: '山田花子',
              email: 'yamada@example.com',
              role: 'user',
              status: 'active',
              createdAt: iso8601DateTimeWithMillis(),
            },
          });
        })
        .executeTest(async (mockServer) => {
          const client = new UserApiClient(mockServer.url);
          const result = await client.createUser({
            name: '山田花子',
            email: 'yamada@example.com',
          });

          expect(result.data.name).toBe('山田花子');
          expect(result.data.email).toBe('yamada@example.com');
        });
    });
  });
});
```

```typescript
// provider/tests/user-api.pact-provider.spec.ts
import { Verifier } from '@pact-foundation/pact';
import { app } from '../src/app';

describe('Provider Verification', () => {
  let server: any;

  beforeAll(async () => {
    server = app.listen(3001);
  });

  afterAll(() => server.close());

  it('Pact契約を満たしている', async () => {
    const verifier = new Verifier({
      providerBaseUrl: 'http://localhost:3001',
      pactUrls: ['./pacts/FrontendApp-UserService.json'],
      // または Pact Broker から取得
      // pactBrokerUrl: 'https://pact-broker.example.com',
      // providerVersion: process.env.GIT_SHA,
      stateHandlers: {
        'ユーザーが3人存在する': async () => {
          // テストデータのセットアップ
          await seedTestUsers(3);
        },
      },
      requestFilter: (req, res, next) => {
        // テスト用のAuth headerを追加
        req.headers['authorization'] = 'Bearer test-token';
        next();
      },
    });

    await verifier.verifyProvider();
  });
});
```

### 6.2 Schemathesis による仕様ベーステスト

```bash
# Schemathesis: OpenAPI仕様から自動テスト生成
pip install schemathesis

# 基本的なテスト実行
schemathesis run http://localhost:3000/v1/openapi.yaml

# 詳細オプション
schemathesis run http://localhost:3000/v1/openapi.yaml \
  --checks all \
  --hypothesis-max-examples 100 \
  --auth "Bearer test-token" \
  --base-url http://localhost:3000/v1 \
  --workers 4

# 特定のエンドポイントのみテスト
schemathesis run http://localhost:3000/v1/openapi.yaml \
  --endpoint "/users" \
  --method GET

# stateful テスト（APIの状態遷移をテスト）
schemathesis run http://localhost:3000/v1/openapi.yaml \
  --stateful=links
```

```python
# Schemathesis のPythonテストとしての使用
import schemathesis

schema = schemathesis.from_url("http://localhost:3000/v1/openapi.yaml")

@schema.parametrize()
def test_api(case):
    """OpenAPI仕様に基づいた自動テスト"""
    response = case.call()
    case.validate_response(response)

# 特定のエンドポイントのテスト
@schema.parametrize(endpoint="/users", method="POST")
def test_create_user(case):
    response = case.call()
    case.validate_response(response)

    if response.status_code == 201:
        data = response.json()
        assert "data" in data
        assert "id" in data["data"]

# カスタムチェック
@schema.parametrize()
def test_response_time(case):
    """レスポンスタイムが500ms以内であること"""
    import time
    start = time.time()
    response = case.call()
    elapsed = time.time() - start

    assert elapsed < 0.5, f"Response took {elapsed:.2f}s (max: 0.5s)"
    case.validate_response(response)
```

### 6.3 Dredd によるAPI仕様テスト

```bash
# Dreddのインストール
npm install -D dredd

# 基本的な実行
npx dredd openapi.yaml http://localhost:3000/v1

# 設定ファイルを使った実行
npx dredd
```

```yaml
# dredd.yml
dry-run: false
hookfiles:
  - "./test/dredd-hooks.ts"
language: typescript
server: npm start
server-wait: 5
reporter:
  - apiary
  - html
output:
  - ./test-results/dredd-report.html
header:
  - "Authorization: Bearer test-token"
  - "Content-Type: application/json"
names: false
only: []
sorted: false
```

```typescript
// test/dredd-hooks.ts
import { Hooks } from 'dredd-hooks';
const hooks = new Hooks();

// テスト前のセットアップ
hooks.beforeAll((transactions, done) => {
  // テストデータベースの初期化
  console.log('テストデータベースを初期化中...');
  done();
});

// 特定のエンドポイントのフック
hooks.before('Users > User Collection > List Users', (transaction, done) => {
  // 事前にテストユーザーを作成
  transaction.request.headers['Authorization'] = 'Bearer test-admin-token';
  done();
});

hooks.before('Users > User Resource > Create User', (transaction, done) => {
  // リクエストボディの調整
  const body = JSON.parse(transaction.request.body);
  body.email = `test-${Date.now()}@example.com`;
  transaction.request.body = JSON.stringify(body);
  done();
});

// レスポンス後の検証
hooks.after('Users > User Resource > Get User', (transaction, done) => {
  const body = JSON.parse(transaction.real.body);
  if (!body.data.id) {
    transaction.fail = 'Response missing user ID';
  }
  done();
});

// スキップするエンドポイント
hooks.before('Admin > Admin Operations > Delete All Users', (transaction, done) => {
  transaction.skip = true;
  done();
});

export default hooks;
```

---

## 7. ドキュメント生成

### 7.1 Redoc

```html
<!-- index.html - Redocによるドキュメント表示 -->
<!DOCTYPE html>
<html>
<head>
  <title>User Management API - Documentation</title>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { margin: 0; padding: 0; }
  </style>
</head>
<body>
  <redoc spec-url='./openapi.yaml'
    hide-hostname
    expand-responses="200,201"
    required-props-first
    sort-props-alphabetically
    path-in-middle-panel
    theme='{
      "colors": {
        "primary": { "main": "#4f46e5" }
      },
      "typography": {
        "fontSize": "15px",
        "fontFamily": "Inter, sans-serif"
      },
      "sidebar": {
        "width": "280px"
      }
    }'
  ></redoc>
  <script src="https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js"></script>
</body>
</html>
```

```bash
# Redocで静的HTMLを生成
npx @redocly/cli build-docs openapi.yaml -o docs/index.html

# カスタムテーマ付き
npx @redocly/cli build-docs openapi.yaml \
  -o docs/index.html \
  --theme.openapi.colors.primary.main="#4f46e5" \
  --theme.openapi.typography.fontSize="15px"
```

### 7.2 Swagger UI

```typescript
// Express.js での Swagger UI 設定
import express from 'express';
import swaggerUi from 'swagger-ui-express';
import YAML from 'yamljs';

const app = express();
const swaggerDocument = YAML.load('./openapi.yaml');

const options = {
  customCss: '.swagger-ui .topbar { display: none }',
  customSiteTitle: "User Management API",
  swaggerOptions: {
    persistAuthorization: true,
    displayRequestDuration: true,
    filter: true,
    tryItOutEnabled: true,
  },
};

app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument, options));

// Swagger JSON エンドポイント
app.get('/api-docs.json', (req, res) => {
  res.json(swaggerDocument);
});
```

### 7.3 Scalar

```typescript
// Scalar - モダンなAPIドキュメントUI
import { apiReference } from '@scalar/express-api-reference';

app.use('/docs', apiReference({
  spec: {
    url: '/api-docs.json',
  },
  theme: 'default',
  layout: 'modern',
  darkMode: true,
  customCss: `
    .darklight { display: none; }
  `,
}));
```

---

## 8. 設計レビューチェックリスト

### 8.1 包括的なレビュー項目

```
API設計レビュー包括チェックリスト:

━━━ 命名規則 ━━━
□ リソース名は名詞・複数形か（/users, /orders, /products）
□ URLはケバブケースか（/user-profiles）
□ プロパティ名は一貫したケーシングか（camelCase推奨）
□ operationIdはcamelCaseか
□ enumの値は一貫しているか（snake_case推奨）
□ 日時フィールド名は統一されているか（createdAt/created_at）

━━━ HTTPメソッド ━━━
□ GET: データ取得のみ、副作用なし
□ POST: リソース作成、または非冪等操作
□ PUT: リソースの完全置換（冪等）
□ PATCH: リソースの部分更新
□ DELETE: リソースの削除（冪等）
□ 冪等性が正しいか（PUT/DELETE は冪等）
□ 安全性が正しいか（GET/HEAD/OPTIONS は安全）

━━━ ステータスコード ━━━
□ 200: 成功（GET, PUT, PATCH）
□ 201: 作成成功（POST）+ Locationヘッダ
□ 204: 成功・レスポンスボディなし（DELETE）
□ 400: リクエスト不正
□ 401: 認証エラー
□ 403: 認可エラー（権限不足）
□ 404: リソース未発見
□ 409: 競合（重複など）
□ 422: バリデーションエラー
□ 429: レート制限超過 + Retry-Afterヘッダ
□ 500: サーバー内部エラー

━━━ レスポンス設計 ━━━
□ エラーレスポンスが統一されているか（RFC 7807推奨）
□ 一覧レスポンスにページネーション情報があるか
□ レスポンスのdataラッパーが統一されているか
□ null可能フィールドが明示されているか
□ 日時はISO 8601形式か
□ IDはUUID形式か

━━━ セキュリティ ━━━
□ 認証方式が定義されているか
□ 入力バリデーションが定義されているか（minLength, maxLength, pattern）
□ レート制限が考慮されているか
□ CORS設定が適切か
□ センシティブデータがURLに含まれていないか
□ 適切な権限チェックがあるか

━━━ 互換性 ━━━
□ 破壊的変更がないか
□ オプショナルフィールドの追加は後方互換か
□ バージョニング戦略が決まっているか
□ 廃止予定のエンドポイントにDeprecatedマーキングがあるか
□ Sunset ヘッダが設定されているか

━━━ パフォーマンス ━━━
□ 大量データのエンドポイントにページネーションがあるか
□ N+1問題を回避するinclude/expandパラメータがあるか
□ キャッシュ戦略（ETag, Cache-Control）が考慮されているか
□ 不要なデータのフィルタリング（fields パラメータ）があるか

━━━ ドキュメント ━━━
□ すべてのエンドポイントにsummaryがあるか
□ リクエスト/レスポンスのexamplesがあるか
□ エラーケースが文書化されているか
□ 認証方法の説明があるか
□ レート制限の説明があるか
```

### 8.2 設計レビュープロセス

```
API設計レビューのワークフロー:

Step 1: 設計提案
───────────────────
  - 開発者がOpenAPI仕様のPRを作成
  - PR説明にAPIの目的・ユースケースを記載
  - 仕様変更の理由を明記

Step 2: 自動チェック（CI）
───────────────────
  - Spectralによるリンティング
  - Breaking Change検出
  - 型定義の生成テスト
  - モックサーバーの起動テスト

Step 3: 人によるレビュー
───────────────────
  - APIアーキテクトまたはテックリード
  - セキュリティエンジニア（認証/認可関連）
  - フロントエンド開発者（使い勝手の確認）
  - チェックリストに基づく確認

Step 4: フィードバック反映
───────────────────
  - レビューコメントに基づく修正
  - 再度自動チェック

Step 5: 承認とマージ
───────────────────
  - 最低2名の承認
  - CI全通過
  - APIカタログへの自動登録
```

---

## 9. 実務での導入ステップ

### 9.1 段階的な導入計画

```
Phase 1: 基盤整備（1-2週間）
─────────────────────────────
  □ OpenAPI仕様書のテンプレート作成
  □ Spectralルールの初期設定
  □ CI/CDパイプラインへのリンティング追加
  □ チームへのOpenAPIトレーニング
  □ ツールチェーンの選定と導入

  成果物:
  - .spectral.yaml
  - openapi-template.yaml
  - CI設定ファイル
  - トレーニング資料

Phase 2: パイロットプロジェクト（2-4週間）
─────────────────────────────────────
  □ 1つの新規APIをAPI Firstで設計
  □ モックサーバーの活用
  □ コード生成の導入
  □ 並行開発の実践
  □ 振り返りとプロセス改善

  成果物:
  - パイロットAPIの仕様書
  - コード生成設定
  - MSWハンドラ
  - 振り返りレポート

Phase 3: 展開（4-8週間）
─────────────────────────
  □ 既存APIのOpenAPI仕様書化
  □ 全新規APIのAPI First適用
  □ Contract Testingの導入
  □ APIカタログの構築
  □ スタイルガイドの策定

  成果物:
  - 既存APIの仕様書
  - APIカタログ
  - APIスタイルガイド
  - Contract Testスイート

Phase 4: 成熟化（継続的）
─────────────────────────
  □ Breaking Change自動検出
  □ SDKの自動生成・公開
  □ APIメトリクスの収集
  □ 定期的なスタイルガイド更新
  □ API設計のナレッジ共有

  成果物:
  - 自動化されたCI/CDパイプライン
  - メトリクスダッシュボード
  - ナレッジベース
```

### 9.2 プロジェクト構成テンプレート

```
project/
├── api/
│   ├── openapi.yaml          # API仕様書（ルート）
│   ├── paths/                 # パス定義
│   ├── schemas/               # スキーマ定義
│   ├── parameters/            # パラメータ定義
│   ├── responses/             # レスポンス定義
│   └── examples/              # レスポンス例
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   ├── types.ts       # ← 自動生成
│   │   │   ├── client.ts      # ← 自動生成
│   │   │   └── custom-fetch.ts
│   │   └── msw/
│   │       ├── handlers.ts    # モックハンドラ
│   │       ├── browser.ts
│   │       └── server.ts
│   └── orval.config.ts
├── backend/
│   ├── api/
│   │   └── api.gen.go         # ← 自動生成
│   └── internal/
│       └── handler/
│           └── user.go        # ハンドラ実装
├── tests/
│   ├── contract/
│   │   ├── consumer.spec.ts   # Consumer契約テスト
│   │   └── provider.spec.ts   # Provider検証テスト
│   └── pacts/                 # 生成されたPactファイル
├── docs/
│   └── index.html             # ← Redocで自動生成
├── .spectral.yaml             # リンティングルール
├── .github/
│   └── workflows/
│       ├── api-lint.yml       # API仕様のリンティング
│       ├── api-codegen.yml    # コード生成
│       └── api-docs.yml       # ドキュメント生成
└── Makefile
```

```makefile
# Makefile - API開発タスク
.PHONY: api-lint api-bundle api-mock api-codegen api-docs api-test

# API仕様のリンティング
api-lint:
	npx @stoplight/spectral-cli lint api/openapi.yaml
	npx @redocly/cli lint api/openapi.yaml

# API仕様のバンドル（分割ファイルの結合）
api-bundle:
	npx @redocly/cli bundle api/openapi.yaml -o dist/openapi.yaml

# モックサーバーの起動
api-mock:
	npx @stoplight/prism-cli mock api/openapi.yaml --port 4010

# コード生成
api-codegen: api-bundle
	npx openapi-typescript dist/openapi.yaml -o frontend/src/api/types.ts
	cd frontend && npx orval

# ドキュメント生成
api-docs: api-bundle
	npx @redocly/cli build-docs dist/openapi.yaml -o docs/index.html

# Contract Test
api-test:
	cd tests/contract && npm test

# Breaking Change検出
api-breaking:
	npx @opticdev/optic diff api/openapi.yaml --base origin/main --check

# 全タスク実行
api-all: api-lint api-bundle api-codegen api-docs api-test
```

### 9.3 package.json のスクリプト設定

```json
{
  "name": "user-management-api",
  "scripts": {
    "api:lint": "spectral lint api/openapi.yaml",
    "api:bundle": "redocly bundle api/openapi.yaml -o dist/openapi.yaml",
    "api:mock": "prism mock api/openapi.yaml --port 4010",
    "api:mock:dynamic": "prism mock api/openapi.yaml --port 4010 --dynamic",
    "api:codegen": "openapi-typescript api/openapi.yaml -o src/api/types.ts && orval",
    "api:codegen:watch": "openapi-typescript api/openapi.yaml -o src/api/types.ts --watch",
    "api:docs": "redocly build-docs api/openapi.yaml -o docs/index.html",
    "api:docs:preview": "redocly preview-docs api/openapi.yaml",
    "api:breaking": "optic diff api/openapi.yaml --base origin/main --check",
    "api:test": "schemathesis run http://localhost:3000/v1/openapi.yaml --checks all",
    "api:validate": "redocly lint api/openapi.yaml && spectral lint api/openapi.yaml",
    "precommit:api": "npm run api:lint && npm run api:codegen",
    "dev": "concurrently \"npm run api:mock\" \"npm run dev:frontend\" \"npm run dev:backend\"",
    "dev:frontend": "vite",
    "dev:backend": "go run ./cmd/server"
  },
  "devDependencies": {
    "@stoplight/prism-cli": "^5.8.0",
    "@stoplight/spectral-cli": "^6.11.0",
    "@redocly/cli": "^1.25.0",
    "@opticdev/optic": "^0.54.0",
    "openapi-typescript": "^7.4.0",
    "orval": "^7.1.0",
    "swagger-ui-express": "^5.0.0",
    "concurrently": "^9.0.0"
  }
}
```

---

## 10. 高度なパターン

### 10.1 APIゲートウェイとの統合

```yaml
# Kong Gateway の宣言的設定（OpenAPIから生成）
_format_version: "3.0"

services:
  - name: user-service
    url: http://user-service:3000
    routes:
      - name: users-list
        paths:
          - /v1/users
        methods:
          - GET
          - POST
        plugins:
          - name: rate-limiting
            config:
              minute: 100
              policy: redis
          - name: jwt
            config:
              claims_to_verify:
                - exp
          - name: request-validator
            config:
              body_schema: '{"type":"object","required":["name","email"]}'

  - name: auth-service
    url: http://auth-service:3001
    routes:
      - name: auth-login
        paths:
          - /v1/auth/login
        methods:
          - POST
        plugins:
          - name: rate-limiting
            config:
              minute: 10
              policy: redis
```

```typescript
// OpenAPIからAPI Gateway設定を生成するスクリプト
import { parse } from 'yaml';
import { readFileSync, writeFileSync } from 'fs';

interface OpenAPISpec {
  paths: Record<string, Record<string, any>>;
  components: {
    securitySchemes: Record<string, any>;
  };
}

function generateGatewayConfig(spec: OpenAPISpec) {
  const services: any[] = [];

  for (const [path, methods] of Object.entries(spec.paths)) {
    for (const [method, operation] of Object.entries(methods)) {
      if (['get', 'post', 'put', 'patch', 'delete'].includes(method)) {
        const tag = operation.tags?.[0] || 'default';

        let service = services.find(s => s.name === `${tag}-service`);
        if (!service) {
          service = {
            name: `${tag}-service`,
            url: `http://${tag}-service:3000`,
            routes: [],
          };
          services.push(service);
        }

        const route = {
          name: operation.operationId,
          paths: [path.replace(/{(\w+)}/g, ':$1')],
          methods: [method.toUpperCase()],
          plugins: [],
        };

        // セキュリティ設定
        if (operation.security !== undefined) {
          if (operation.security.length > 0) {
            route.plugins.push({
              name: 'jwt',
              config: { claims_to_verify: ['exp'] },
            });
          }
        }

        // レート制限
        route.plugins.push({
          name: 'rate-limiting',
          config: { minute: 100, policy: 'redis' },
        });

        service.routes.push(route);
      }
    }
  }

  return { _format_version: '3.0', services };
}

const spec = parse(readFileSync('openapi.yaml', 'utf-8'));
const config = generateGatewayConfig(spec);
writeFileSync('kong.yaml', JSON.stringify(config, null, 2));
```

### 10.2 マイクロサービスでのAPI First

```
マイクロサービスアーキテクチャでのAPI First運用:

┌──────────────────────────────────────────────┐
│                API Registry                   │
│  (すべてのサービスのOpenAPI仕様を集約管理)      │
│                                               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐        │
│  │ User API│ │Order API│ │Payment  │  ...    │
│  │  v1.2.0 │ │  v2.0.0 │ │ API v1  │        │
│  └─────────┘ └─────────┘ └─────────┘        │
└──────────────────────────────────────────────┘
         ↓              ↓             ↓
┌────────────┐  ┌────────────┐  ┌────────────┐
│ User       │  │ Order      │  │ Payment    │
│ Service    │←→│ Service    │←→│ Service    │
│            │  │            │  │            │
│ ・仕様を先  │  │ ・依存先の  │  │ ・Contract │
│   に定義    │  │   仕様参照  │  │   Test実施 │
│ ・Contract  │  │ ・型安全な  │  │ ・Breaking │
│   Test公開  │  │   クライアント │  │   Change  │
│ ・Mock提供  │  │   生成      │  │   検出     │
└────────────┘  └────────────┘  └────────────┘
```

```yaml
# サービス間通信の仕様定義
# order-service/api/internal/user-client.yaml
# （User Serviceの仕様から必要な部分を参照）
openapi: '3.1.0'
info:
  title: User Service Client（Order Serviceが使用する部分）
  version: '1.0.0'

paths:
  /users/{userId}:
    get:
      operationId: getUser
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: object
                    properties:
                      id:
                        type: string
                        format: uuid
                      name:
                        type: string
                      email:
                        type: string
                        format: email
```

### 10.3 イベント駆動APIの設計

```yaml
# AsyncAPI 仕様（イベント駆動API）
asyncapi: '2.6.0'
info:
  title: User Events API
  version: '1.0.0'
  description: ユーザー関連イベントの非同期API仕様

channels:
  user.created:
    publish:
      operationId: onUserCreated
      summary: ユーザー作成イベント
      message:
        name: UserCreatedEvent
        contentType: application/json
        payload:
          type: object
          required: [eventId, eventType, timestamp, data]
          properties:
            eventId:
              type: string
              format: uuid
            eventType:
              type: string
              const: user.created
            timestamp:
              type: string
              format: date-time
            data:
              type: object
              properties:
                userId:
                  type: string
                  format: uuid
                name:
                  type: string
                email:
                  type: string
                  format: email
        examples:
          - payload:
              eventId: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
              eventType: "user.created"
              timestamp: "2024-06-01T12:00:00Z"
              data:
                userId: "550e8400-e29b-41d4-a716-446655440000"
                name: "田中太郎"
                email: "tanaka@example.com"

  user.updated:
    publish:
      operationId: onUserUpdated
      summary: ユーザー更新イベント
      message:
        name: UserUpdatedEvent
        contentType: application/json
        payload:
          type: object
          required: [eventId, eventType, timestamp, data]
          properties:
            eventId:
              type: string
              format: uuid
            eventType:
              type: string
              const: user.updated
            timestamp:
              type: string
              format: date-time
            data:
              type: object
              properties:
                userId:
                  type: string
                  format: uuid
                changes:
                  type: object
                  additionalProperties: true

  user.deleted:
    publish:
      operationId: onUserDeleted
      summary: ユーザー削除イベント
      message:
        name: UserDeletedEvent
        contentType: application/json
        payload:
          type: object
          properties:
            eventId:
              type: string
              format: uuid
            eventType:
              type: string
              const: user.deleted
            timestamp:
              type: string
              format: date-time
            data:
              type: object
              properties:
                userId:
                  type: string
                  format: uuid
                deletedAt:
                  type: string
                  format: date-time
```

### 10.4 API ライフサイクル管理

```
API ライフサイクルの各フェーズ:

┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Design  │ → │  Build  │ → │  Test   │ → │ Deploy  │ → │ Retire  │
│ 設計    │   │  構築   │   │ テスト  │   │ 運用    │   │ 廃止    │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
     │              │              │              │              │
     ▼              ▼              ▼              ▼              ▼
  OpenAPI       コード生成     Contract      モニタリング   Deprecation
  Spectral      モック生成     Schemathesis   メトリクス    Sunset Header
  レビュー      型安全実装     Pact           アラート      移行ガイド

各フェーズの詳細:

1. Design（設計）
   - ユースケース分析
   - OpenAPI仕様の作成
   - Spectralによるリンティング
   - 設計レビュー（最低2名承認）
   - Breaking Change検出

2. Build（構築）
   - コード生成（型、クライアント、サーバースタブ）
   - モックサーバー構築
   - ハンドラ実装
   - 並行開発の実施

3. Test（テスト）
   - Contract Testing
   - Property-based Testing（Schemathesis）
   - Integration Testing
   - Performance Testing
   - Security Testing

4. Deploy（運用）
   - ドキュメント公開
   - SDK配布
   - メトリクス収集
   - エラー率モニタリング
   - SLA管理

5. Retire（廃止）
   - Deprecationマーキング
   - Sunset Headerの追加
   - 移行ガイドの提供
   - 利用者への通知
   - 段階的な廃止
```

```typescript
// API廃止のための実装例
import express from 'express';

// Deprecation ミドルウェア
function deprecationMiddleware(
  sunsetDate: string,
  alternativeUrl: string,
) {
  return (req: express.Request, res: express.Response, next: express.NextFunction) => {
    res.setHeader('Deprecation', 'true');
    res.setHeader('Sunset', new Date(sunsetDate).toUTCString());
    res.setHeader('Link', `<${alternativeUrl}>; rel="successor-version"`);

    // メトリクスに記録
    metrics.counter('api.deprecated.usage', 1, {
      path: req.path,
      method: req.method,
      consumer: req.headers['x-consumer-id'] as string,
    });

    next();
  };
}

// 廃止予定エンドポイントの設定
app.get('/v1/users/search',
  deprecationMiddleware('2025-06-01', '/v2/users?search='),
  async (req, res) => {
    // 旧実装
    const results = await userService.search(req.query.q as string);
    res.json({ data: results });
  }
);
```

---

## 11. トラブルシューティング

### 11.1 よくある問題と解決策

```
問題1: 仕様とコードの乖離
──────────────────────────
  症状: 実装がOpenAPI仕様と一致しない
  原因: 手動実装で仕様の変更が反映されていない
  解決:
  - CI/CDでDredd/Schemathesisによる自動検証
  - コード生成の活用で乖離を防ぐ
  - Prismのproxyモードで実APIを検証

問題2: OpenAPI仕様が肥大化
──────────────────────────
  症状: 1ファイルが数千行になり管理困難
  原因: すべての定義を1ファイルに記述
  解決:
  - ファイル分割（paths/, schemas/, responses/）
  - $refによる参照
  - redocly bundleで統合
  - タグによる論理的な分類

問題3: コード生成の型が不正確
──────────────────────────
  症状: 生成された型がnull許容やオプショナルの扱いが不正確
  原因: OpenAPI仕様のnullable/required指定が不完全
  解決:
  - nullableフィールドの明示的指定
  - requiredフィールドの正確なリスト
  - 生成された型のスナップショットテスト

問題4: モックと実装の不一致
──────────────────────────
  症状: モックサーバーではOKだが実APIで動かない
  原因: モックが仕様に基づかないカスタム実装
  解決:
  - Prismで仕様ベースのモック使用
  - Contract Testの導入
  - E2Eテストの追加

問題5: Breaking Changeの検出漏れ
──────────────────────────
  症状: APIの変更がクライアントを壊す
  原因: Breaking Changeの自動検出がない
  解決:
  - Opticの導入
  - CIでの自動チェック
  - セマンティックバージョニング
  - 変更ログの自動生成
```

### 11.2 パフォーマンス考慮事項

```typescript
// APIレスポンスのキャッシュ制御
import express from 'express';

function cacheControl(maxAge: number, isPublic: boolean = false) {
  return (req: express.Request, res: express.Response, next: express.NextFunction) => {
    const directive = isPublic ? 'public' : 'private';
    res.setHeader('Cache-Control', `${directive}, max-age=${maxAge}`);
    next();
  };
}

// ETagベースの条件付きリクエスト
function conditionalRequest() {
  return (req: express.Request, res: express.Response, next: express.NextFunction) => {
    const originalJson = res.json.bind(res);

    res.json = (body: any) => {
      const etag = generateETag(JSON.stringify(body));
      res.setHeader('ETag', etag);

      if (req.headers['if-none-match'] === etag) {
        return res.status(304).end();
      }

      return originalJson(body);
    };

    next();
  };
}

// ユーザー一覧（キャッシュ付き）
app.get('/v1/users',
  cacheControl(60, false),  // 60秒のプライベートキャッシュ
  conditionalRequest(),
  async (req, res) => {
    const users = await userService.list(req.query);
    res.json({ data: users });
  }
);

// 静的リソース（長いキャッシュ）
app.get('/v1/users/:id/avatar',
  cacheControl(86400, true),  // 24時間のパブリックキャッシュ
  async (req, res) => {
    const avatar = await userService.getAvatar(req.params.id);
    res.type('image/png').send(avatar);
  }
);
```

---

## 12. 実践演習

### 演習1: ECサイトAPIの設計

```
要件:
- 商品カタログのCRUD
- カートの管理
- 注文の作成・取得
- ユーザーレビュー

課題:
1. OpenAPI仕様を設計してください
2. Spectralルールを設定してください
3. Prismでモックサーバーを起動してください
4. openapi-typescriptで型を生成してください
5. MSWでフロントエンド用モックを作成してください
```

```yaml
# 演習1の回答例（商品カタログ部分）
openapi: '3.1.0'
info:
  title: E-Commerce API
  version: '1.0.0'

paths:
  /products:
    get:
      operationId: listProducts
      tags: [Products]
      parameters:
        - name: page
          in: query
          schema: { type: integer, default: 1 }
        - name: per_page
          in: query
          schema: { type: integer, default: 20, maximum: 100 }
        - name: category
          in: query
          schema: { type: string }
        - name: min_price
          in: query
          schema: { type: number, minimum: 0 }
        - name: max_price
          in: query
          schema: { type: number, minimum: 0 }
        - name: sort
          in: query
          schema:
            type: string
            enum: [price_asc, price_desc, newest, popular]
      responses:
        '200':
          description: 商品一覧
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/Product'
                  meta:
                    $ref: '#/components/schemas/PaginationMeta'

  /products/{productId}:
    get:
      operationId: getProduct
      tags: [Products]
      parameters:
        - name: productId
          in: path
          required: true
          schema: { type: string, format: uuid }
      responses:
        '200':
          description: 商品詳細
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    $ref: '#/components/schemas/ProductDetail'

  /cart:
    get:
      operationId: getCart
      tags: [Cart]
      responses:
        '200':
          description: カートの内容
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    $ref: '#/components/schemas/Cart'

    post:
      operationId: addToCart
      tags: [Cart]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [productId, quantity]
              properties:
                productId:
                  type: string
                  format: uuid
                quantity:
                  type: integer
                  minimum: 1
                  maximum: 99
      responses:
        '200':
          description: カートに追加成功
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    $ref: '#/components/schemas/Cart'

  /orders:
    post:
      operationId: createOrder
      tags: [Orders]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [shippingAddressId, paymentMethodId]
              properties:
                shippingAddressId:
                  type: string
                  format: uuid
                paymentMethodId:
                  type: string
                  format: uuid
                note:
                  type: string
                  maxLength: 500
      responses:
        '201':
          description: 注文作成成功
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    $ref: '#/components/schemas/Order'

components:
  schemas:
    Product:
      type: object
      properties:
        id: { type: string, format: uuid }
        name: { type: string }
        price: { type: number }
        currency: { type: string, default: "JPY" }
        thumbnailUrl: { type: string, format: uri }
        category: { type: string }
        inStock: { type: boolean }

    ProductDetail:
      allOf:
        - $ref: '#/components/schemas/Product'
        - type: object
          properties:
            description: { type: string }
            images:
              type: array
              items: { type: string, format: uri }
            specifications:
              type: object
              additionalProperties: { type: string }
            averageRating: { type: number, minimum: 0, maximum: 5 }
            reviewCount: { type: integer }

    Cart:
      type: object
      properties:
        items:
          type: array
          items:
            type: object
            properties:
              product: { $ref: '#/components/schemas/Product' }
              quantity: { type: integer }
              subtotal: { type: number }
        totalItems: { type: integer }
        totalPrice: { type: number }

    Order:
      type: object
      properties:
        id: { type: string, format: uuid }
        status:
          type: string
          enum: [pending, confirmed, shipped, delivered, cancelled]
        items:
          type: array
          items:
            type: object
            properties:
              productId: { type: string, format: uuid }
              productName: { type: string }
              quantity: { type: integer }
              unitPrice: { type: number }
        totalPrice: { type: number }
        createdAt: { type: string, format: date-time }

    PaginationMeta:
      type: object
      properties:
        total: { type: integer }
        page: { type: integer }
        per_page: { type: integer }
        total_pages: { type: integer }
```

### 演習2: API仕様のリファクタリング

```
課題: 以下の問題がある既存API仕様を改善してください

問題のある仕様:
- エラーレスポンスが統一されていない
- ページネーションがない
- 認証が定義されていない
- operationIdがない
- examplesがない
- nullable指定が漏れている

改善のポイント:
1. RFC 7807 形式のエラーレスポンスを統一定義
2. ページネーションパラメータとメタ情報を追加
3. Bearer Token認証を追加
4. 全エンドポイントにoperationIdを付与
5. リクエスト/レスポンスのexamplesを追加
6. nullable: true を必要なフィールドに追加
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| API First | 実装前に仕様を確定、並行開発を実現 |
| OpenAPI 3.1 | 業界標準のAPI仕様記述形式、JSON Schema完全互換 |
| コード生成 | 型安全なクライアント/サーバーを自動生成 |
| モックサーバー | Prism/MSW等で仕様からモックを自動生成 |
| リンティング | Spectralで設計品質を自動チェック |
| Contract Testing | Pact/Schemathesisで仕様準拠を検証 |
| ドキュメント | Redoc/Scalar/Swagger UIで自動生成 |
| ライフサイクル | 設計→構築→テスト→運用→廃止の全フェーズ管理 |
| ガバナンス | 組織全体のスタイルガイドと品質基準 |
| Breaking Change | Opticで破壊的変更を自動検出 |

---

## 次に読むべきガイド
-> [[01-naming-and-conventions.md]] -- 命名規則と慣例
-> [[02-versioning-strategy.md]] -- バージョニング戦略
-> [[03-pagination-and-filtering.md]] -- ページネーションとフィルタリング

---

## 参考文献
1. OpenAPI Initiative. "OpenAPI Specification 3.1." 2024.
2. Stoplight. "API Design Guide." 2024.
3. RFC 7807. "Problem Details for HTTP APIs." IETF, 2016.
4. SmartBear. "Swagger / OpenAPI Best Practices." 2024.
5. Pact Foundation. "Consumer-Driven Contract Testing." 2024.
6. Schemathesis. "Property-Based Testing for APIs." 2024.
7. Redocly. "API Documentation Best Practices." 2024.
8. Optic. "API Change Management." 2024.
9. AsyncAPI Initiative. "AsyncAPI Specification." 2024.
10. Kong. "API Gateway Configuration." 2024.

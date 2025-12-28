# API Documentation完全ガイド

> **対象読者**: バックエンド開発者、API設計者、フロントエンド開発者
> **難易度**: 中級
> **推定読了時間**: 50分

---

## 📋 目次

1. [API Documentationとは](#api-documentationとは)
2. [なぜ重要か](#なぜ重要か)
3. [ドキュメント形式の選択](#ドキュメント形式の選択)
4. [OpenAPI (Swagger)](#openapi-swagger)
5. [エンドポイントの記述](#エンドポイントの記述)
6. [認証・認可](#認証認可)
7. [エラーハンドリング](#エラーハンドリング)
8. [バージョニング](#バージョニング)
9. [コード生成](#コード生成)
10. [実践例](#実践例)
11. [ツールとベストプラクティス](#ツールとベストプラクティス)
12. [FAQ](#faq)

---

## API Documentationとは

### 定義

**API Documentation（API仕様書）** は、APIの使い方、エンドポイント、リクエスト/レスポンス形式、認証方法などを記述した技術文書です。

### 対象読者

```
👥 主な読者:
├── フロントエンド開発者（APIを使う側）
├── バックエンド開発者（APIを実装する側）
├── モバイル開発者（APIを使う側）
├── パートナー企業（外部API利用者）
└── 将来の自分（数ヶ月後）
```

### ドキュメントに含めるべき情報

```markdown
✅ 必須:
- エンドポイント（URL、HTTPメソッド）
- リクエストパラメータ
- レスポンス形式
- エラーコード
- 認証方法

✅ 推奨:
- 使用例（curl、コード例）
- レート制限
- バージョニング情報
- 変更履歴

✅ オプション:
- SDKリンク
- Postmanコレクション
- インタラクティブなAPI試行環境
```

---

## なぜ重要か

### 1. 開発効率の向上

**ドキュメントなし:**
```
開発者A: 「このエンドポイントのパラメータは？」
開発者B: 「コード見て」
開発者A: 「コード読むのに30分かかった...」
```

**ドキュメントあり:**
```
開発者A: API仕様書を確認
→ 5分で理解
→ すぐに実装開始
```

**効果:**
- 質問が80%減少
- 実装時間が50%短縮
- オンボーディング時間が1週間→1日に

### 2. エラーの削減

```
❌ ドキュメントなし:
→ パラメータ名を間違える（userName vs user_name）
→ 必須/オプションが不明
→ 試行錯誤で時間を浪費

✅ ドキュメントあり:
→ 正しいパラメータ名が明記
→ バリデーションルールが明確
→ 1回で正しく実装
```

### 3. 外部連携の促進

```
外部パートナー企業との連携:

良いAPI仕様書 → 1週間で連携完了
悪いAPI仕様書 → 1ヶ月かかる + 何度も質問
```

---

## ドキュメント形式の選択

### 形式の比較

| 形式 | メリット | デメリット | 推奨用途 |
|------|---------|-----------|---------|
| **OpenAPI (Swagger)** | 標準形式、ツール豊富、コード生成可能 | YAML/JSON記述が必要 | RESTful API全般 |
| **Markdown** | 書きやすい、Git管理しやすい | ツールサポート少ない | 小規模API、内部API |
| **Postman** | 試行しながら作成、共有簡単 | バージョン管理しにくい | プロトタイプ、テスト |
| **GraphQL Schema** | 型安全、自己文書化 | REST APIには使えない | GraphQL専用 |
| **gRPC Proto** | 高速、型安全 | REST APIには使えない | gRPC専用 |

### 推奨: OpenAPI 3.0

**理由:**
```
✅ 業界標準
✅ ツールが豊富（Swagger UI、Redoc、Postmanなど）
✅ コード生成可能（クライアントSDK、サーバースタブ）
✅ バリデーション可能
✅ Gitで管理可能
```

---

## OpenAPI (Swagger)

### 基本構造

```yaml
openapi: 3.0.0
info:
  title: User API
  version: 1.0.0
  description: ユーザー管理API
  contact:
    name: API Support
    email: support@example.com
  license:
    name: MIT
    url: https://opensource.org/licenses/MIT

servers:
  - url: https://api.example.com/v1
    description: 本番環境
  - url: https://staging-api.example.com/v1
    description: ステージング環境

paths:
  /users:
    get:
      summary: ユーザー一覧取得
      description: 登録されているユーザーの一覧を取得します
      tags:
        - Users
      parameters:
        - name: page
          in: query
          description: ページ番号（1から開始）
          required: false
          schema:
            type: integer
            minimum: 1
            default: 1
        - name: limit
          in: query
          description: 1ページあたりの件数
          required: false
          schema:
            type: integer
            minimum: 1
            maximum: 100
            default: 20
      responses:
        '200':
          description: 成功
          content:
            application/json:
              schema:
                type: object
                properties:
                  users:
                    type: array
                    items:
                      $ref: '#/components/schemas/User'
                  pagination:
                    $ref: '#/components/schemas/Pagination'
              example:
                users:
                  - id: "user-123"
                    name: "山田太郎"
                    email: "yamada@example.com"
                    createdAt: "2025-01-15T10:00:00Z"
                pagination:
                  page: 1
                  limit: 20
                  total: 150
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'

components:
  schemas:
    User:
      type: object
      required:
        - id
        - name
        - email
      properties:
        id:
          type: string
          description: ユーザーID（UUID形式）
          example: "user-123"
        name:
          type: string
          description: ユーザー名
          minLength: 2
          maxLength: 50
          example: "山田太郎"
        email:
          type: string
          format: email
          description: メールアドレス
          example: "yamada@example.com"
        createdAt:
          type: string
          format: date-time
          description: 作成日時（ISO 8601形式）
          example: "2025-01-15T10:00:00Z"

    Pagination:
      type: object
      properties:
        page:
          type: integer
          description: 現在のページ番号
          example: 1
        limit:
          type: integer
          description: 1ページあたりの件数
          example: 20
        total:
          type: integer
          description: 総件数
          example: 150

  responses:
    BadRequest:
      description: リクエストが不正です
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: VALIDATION_ERROR
            message: "パラメータが不正です"
            details:
              - field: "email"
                message: "メールアドレス形式が不正です"

    Unauthorized:
      description: 認証エラー
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: UNAUTHORIZED
            message: "認証に失敗しました"

  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: JWT形式のアクセストークン

security:
  - BearerAuth: []
```

### セクション別解説

#### 1. infoセクション

```yaml
info:
  title: User API              # APIの名前
  version: 1.0.0              # バージョン（Semantic Versioning）
  description: |              # 説明（Markdown可）
    ユーザー管理APIです。

    ## 機能
    - ユーザーの作成・取得・更新・削除
    - ユーザー検索
    - ユーザー認証

  contact:
    name: API Support
    email: support@example.com
    url: https://support.example.com
  license:
    name: MIT
```

#### 2. serversセクション

```yaml
servers:
  - url: https://api.example.com/v1
    description: 本番環境

  - url: https://staging-api.example.com/v1
    description: ステージング環境

  - url: http://localhost:3000/v1
    description: ローカル開発環境
    variables:
      port:
        default: '3000'
        description: ポート番号
```

#### 3. pathsセクション（エンドポイント定義）

```yaml
paths:
  /users:
    get:
      summary: ユーザー一覧取得
      description: 登録されているユーザーの一覧を取得します
      operationId: listUsers    # 一意なID（コード生成で使用）
      tags:
        - Users
      parameters: [...]
      responses: [...]

    post:
      summary: ユーザー作成
      description: 新しいユーザーを作成します
      operationId: createUser
      tags:
        - Users
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserInput'
      responses: [...]
```

---

## エンドポイントの記述

### GET（取得）

```yaml
/users/{userId}:
  get:
    summary: ユーザー情報取得
    description: 指定されたIDのユーザー情報を取得します
    tags:
      - Users
    parameters:
      - name: userId
        in: path                  # パスパラメータ
        required: true
        description: ユーザーID
        schema:
          type: string
          example: "user-123"

      - name: include
        in: query                 # クエリパラメータ
        required: false
        description: 含める関連データ（カンマ区切り）
        schema:
          type: string
          enum:
            - posts
            - comments
            - both
          example: "posts,comments"

    responses:
      '200':
        description: 成功
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/User'
            example:
              id: "user-123"
              name: "山田太郎"
              email: "yamada@example.com"
              posts:
                - id: "post-1"
                  title: "最初の投稿"

      '404':
        description: ユーザーが見つかりません
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/Error'
            example:
              error: NOT_FOUND
              message: "ユーザーが見つかりません"
```

### POST（作成）

```yaml
/users:
  post:
    summary: ユーザー作成
    description: 新しいユーザーを作成します
    tags:
      - Users
    requestBody:
      required: true
      description: ユーザー情報
      content:
        application/json:
          schema:
            type: object
            required:
              - name
              - email
              - password
            properties:
              name:
                type: string
                description: ユーザー名
                minLength: 2
                maxLength: 50
                example: "山田太郎"
              email:
                type: string
                format: email
                description: メールアドレス（ユニーク）
                example: "yamada@example.com"
              password:
                type: string
                format: password
                description: パスワード（8文字以上）
                minLength: 8
                example: "SecurePass123!"
          example:
            name: "山田太郎"
            email: "yamada@example.com"
            password: "SecurePass123!"

    responses:
      '201':
        description: 作成成功
        headers:
          Location:
            description: 作成されたリソースのURL
            schema:
              type: string
              example: "/users/user-123"
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/User'
            example:
              id: "user-123"
              name: "山田太郎"
              email: "yamada@example.com"
              createdAt: "2025-01-15T10:00:00Z"

      '400':
        description: バリデーションエラー
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ValidationError'
            example:
              error: VALIDATION_ERROR
              message: "入力値が不正です"
              details:
                - field: "email"
                  message: "このメールアドレスは既に使用されています"

      '409':
        description: メールアドレス重複
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/Error'
            example:
              error: CONFLICT
              message: "メールアドレスは既に使用されています"
```

### PUT（更新）

```yaml
/users/{userId}:
  put:
    summary: ユーザー情報更新
    description: ユーザー情報を更新します（全体更新）
    tags:
      - Users
    parameters:
      - name: userId
        in: path
        required: true
        schema:
          type: string
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - name
              - email
            properties:
              name:
                type: string
              email:
                type: string
                format: email
    responses:
      '200':
        description: 更新成功
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/User'
```

### PATCH（部分更新）

```yaml
/users/{userId}:
  patch:
    summary: ユーザー情報部分更新
    description: ユーザー情報の一部を更新します
    tags:
      - Users
    parameters:
      - name: userId
        in: path
        required: true
        schema:
          type: string
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              name:
                type: string
              email:
                type: string
                format: email
            # required なし（どれか1つでOK）
          example:
            name: "鈴木一郎"  # emailは更新しない
    responses:
      '200':
        description: 更新成功
```

### DELETE（削除）

```yaml
/users/{userId}:
  delete:
    summary: ユーザー削除
    description: 指定されたユーザーを削除します
    tags:
      - Users
    parameters:
      - name: userId
        in: path
        required: true
        schema:
          type: string
    responses:
      '204':
        description: 削除成功（レスポンスボディなし）

      '404':
        description: ユーザーが見つかりません
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/Error'
```

---

## 認証・認可

### JWT Bearer認証

```yaml
components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: |
        JWT形式のアクセストークンを使用します。

        ## 取得方法
        POST /auth/login でトークンを取得してください。

        ## 使用方法
        ```
        Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
        ```

        ## 有効期限
        24時間（期限切れの場合は401エラー）

# 全体に適用
security:
  - BearerAuth: []

# または個別のエンドポイントに適用
paths:
  /users:
    get:
      security:
        - BearerAuth: []

  /public/info:
    get:
      security: []  # 認証不要
```

### APIキー認証

```yaml
components:
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
      description: |
        APIキーを使用した認証です。

        ## 取得方法
        管理画面からAPIキーを発行してください。

        ## 使用方法
        ```
        X-API-Key: your-api-key-here
        ```

security:
  - ApiKeyAuth: []
```

### OAuth 2.0

```yaml
components:
  securitySchemes:
    OAuth2:
      type: oauth2
      description: OAuth 2.0認証
      flows:
        authorizationCode:
          authorizationUrl: https://example.com/oauth/authorize
          tokenUrl: https://example.com/oauth/token
          scopes:
            read:users: ユーザー情報の読み取り
            write:users: ユーザー情報の書き込み
            delete:users: ユーザーの削除

security:
  - OAuth2:
      - read:users
      - write:users
```

### 複数の認証方法

```yaml
# BearerまたはApiKeyのどちらか
security:
  - BearerAuth: []
  - ApiKeyAuth: []

# BearerとApiKey両方必要
security:
  - BearerAuth: []
    ApiKeyAuth: []
```

---

## エラーハンドリング

### 標準エラーレスポンス

```yaml
components:
  schemas:
    Error:
      type: object
      required:
        - error
        - message
      properties:
        error:
          type: string
          description: エラーコード
          example: "VALIDATION_ERROR"
        message:
          type: string
          description: エラーメッセージ（ユーザー向け）
          example: "入力値が不正です"
        details:
          type: array
          description: 詳細なエラー情報
          items:
            type: object
            properties:
              field:
                type: string
                description: エラーが発生したフィールド
                example: "email"
              message:
                type: string
                description: フィールド固有のエラーメッセージ
                example: "メールアドレス形式が不正です"
        requestId:
          type: string
          description: リクエストID（サポート問い合わせ用）
          example: "req-abc123"
```

### HTTPステータスコード一覧

```yaml
responses:
  '200':
    description: 成功

  '201':
    description: 作成成功

  '204':
    description: 成功（レスポンスボディなし）

  '400':
    description: |
      リクエストが不正です

      原因:
      - パラメータ形式エラー
      - バリデーションエラー
      - 必須パラメータ不足

  '401':
    description: |
      認証エラー

      原因:
      - トークンがない
      - トークンが無効
      - トークンの期限切れ

  '403':
    description: |
      権限エラー

      原因:
      - 必要な権限がない
      - アクセスが禁止されているリソース

  '404':
    description: |
      リソースが見つかりません

      原因:
      - 存在しないID
      - 削除済みのリソース

  '409':
    description: |
      競合エラー

      原因:
      - ユニーク制約違反
      - 同時更新の競合

  '422':
    description: |
      処理できないエンティティ

      原因:
      - ビジネスロジックエラー
      - 状態遷移エラー

  '429':
    description: |
      レート制限超過

      ヘッダー:
      - X-RateLimit-Limit: 上限
      - X-RateLimit-Remaining: 残り回数
      - X-RateLimit-Reset: リセット時刻

  '500':
    description: |
      サーバーエラー

      原因:
      - 予期しないエラー
      - システム障害

  '503':
    description: |
      サービス利用不可

      原因:
      - メンテナンス中
      - 過負荷
```

### エラーコード体系

```yaml
components:
  schemas:
    ErrorCode:
      type: string
      enum:
        # バリデーションエラー（400）
        - VALIDATION_ERROR
        - REQUIRED_FIELD_MISSING
        - INVALID_FORMAT
        - OUT_OF_RANGE

        # 認証エラー（401）
        - UNAUTHORIZED
        - TOKEN_EXPIRED
        - TOKEN_INVALID
        - CREDENTIALS_INVALID

        # 権限エラー（403）
        - FORBIDDEN
        - INSUFFICIENT_PERMISSIONS

        # リソースエラー（404）
        - NOT_FOUND
        - RESOURCE_NOT_FOUND

        # 競合エラー（409）
        - CONFLICT
        - DUPLICATE_ENTRY
        - VERSION_CONFLICT

        # ビジネスロジックエラー（422）
        - UNPROCESSABLE_ENTITY
        - INVALID_STATE_TRANSITION
        - BUSINESS_RULE_VIOLATION

        # レート制限（429）
        - RATE_LIMIT_EXCEEDED

        # サーバーエラー（500）
        - INTERNAL_SERVER_ERROR
        - DATABASE_ERROR

      example: "VALIDATION_ERROR"
```

---

## バージョニング

### URLパスバージョニング（推奨）

```yaml
servers:
  - url: https://api.example.com/v1
    description: バージョン1（現行）

  - url: https://api.example.com/v2
    description: バージョン2（最新）

# v1とv2で異なる仕様を持つ
```

**メリット:**
- 明確
- キャッシュしやすい
- 複数バージョン並行運用が容易

**デメリット:**
- URLが変わる

### ヘッダーバージョニング

```yaml
paths:
  /users:
    get:
      parameters:
        - name: API-Version
          in: header
          schema:
            type: string
            enum:
              - '1.0'
              - '2.0'
            default: '2.0'
```

**メリット:**
- URLが変わらない

**デメリット:**
- キャッシュが複雑
- テストしにくい

### 非推奨（Deprecated）の扱い

```yaml
/users:
  get:
    deprecated: true
    description: |
      **⚠️ 非推奨: このエンドポイントは2025年12月31日に廃止されます**

      代わりに `/v2/users` を使用してください。

      移行ガイド: https://docs.example.com/migration/v1-to-v2
```

### CHANGELOG

```markdown
## API CHANGELOG

### v2.0.0 - 2025-02-01

#### Breaking Changes
- `/users` エンドポイントのレスポンス形式変更
  - 変更前: `{ data: [...] }`
  - 変更後: `{ users: [...], pagination: {...} }`

#### New Features
- `/users/search` エンドポイント追加

#### Deprecations
- `/v1/users` は2025年12月31日に廃止予定

### v1.2.0 - 2025-01-15

#### New Features
- `/users/{id}/avatar` エンドポイント追加
```

---

## コード生成

### クライアントSDK生成

```bash
# OpenAPI GeneratorでTypeScript SDKを生成
npx @openapitools/openapi-generator-cli generate \
  -i openapi.yaml \
  -g typescript-axios \
  -o ./generated/typescript-client

# 使用例
import { UsersApi } from './generated/typescript-client';

const api = new UsersApi({
  basePath: 'https://api.example.com/v1',
  accessToken: 'your-token'
});

const users = await api.listUsers({ page: 1, limit: 20 });
```

### サーバースタブ生成

```bash
# Node.js Express サーバースタブ生成
npx @openapitools/openapi-generator-cli generate \
  -i openapi.yaml \
  -g nodejs-express-server \
  -o ./generated/server
```

### 対応言語

```
クライアント:
- TypeScript / JavaScript
- Python
- Java
- Swift (iOS)
- Kotlin (Android)
- Go
- Ruby
- PHP

サーバー:
- Node.js (Express, Fastify)
- Python (FastAPI, Flask, Django)
- Java (Spring Boot)
- Go (Gin, Echo)
```

---

## 実践例

### 完全なRESTful API仕様

```yaml
openapi: 3.0.0
info:
  title: Todo API
  version: 1.0.0
  description: |
    シンプルなTodo管理APIです。

    ## 認証
    全てのエンドポイントでJWT認証が必要です。

    ## レート制限
    - 認証済みユーザー: 1000リクエスト/時間
    - 未認証: 100リクエスト/時間

servers:
  - url: https://api.example.com/v1

paths:
  /todos:
    get:
      summary: Todo一覧取得
      tags:
        - Todos
      parameters:
        - name: status
          in: query
          schema:
            type: string
            enum:
              - todo
              - in_progress
              - done
        - name: sort
          in: query
          schema:
            type: string
            enum:
              - created_at
              - updated_at
              - due_date
            default: created_at
        - name: order
          in: query
          schema:
            type: string
            enum:
              - asc
              - desc
            default: desc
      responses:
        '200':
          description: 成功
          content:
            application/json:
              schema:
                type: object
                properties:
                  todos:
                    type: array
                    items:
                      $ref: '#/components/schemas/Todo'

    post:
      summary: Todo作成
      tags:
        - Todos
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - title
              properties:
                title:
                  type: string
                  minLength: 1
                  maxLength: 200
                description:
                  type: string
                  maxLength: 1000
                dueDate:
                  type: string
                  format: date-time
                priority:
                  type: string
                  enum:
                    - low
                    - medium
                    - high
                  default: medium
      responses:
        '201':
          description: 作成成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Todo'

  /todos/{todoId}:
    parameters:
      - name: todoId
        in: path
        required: true
        schema:
          type: string

    get:
      summary: Todo詳細取得
      tags:
        - Todos
      responses:
        '200':
          description: 成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Todo'
        '404':
          $ref: '#/components/responses/NotFound'

    patch:
      summary: Todo更新
      tags:
        - Todos
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                title:
                  type: string
                description:
                  type: string
                status:
                  type: string
                  enum:
                    - todo
                    - in_progress
                    - done
                dueDate:
                  type: string
                  format: date-time
                priority:
                  type: string
                  enum:
                    - low
                    - medium
                    - high
      responses:
        '200':
          description: 更新成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Todo'

    delete:
      summary: Todo削除
      tags:
        - Todos
      responses:
        '204':
          description: 削除成功
        '404':
          $ref: '#/components/responses/NotFound'

components:
  schemas:
    Todo:
      type: object
      required:
        - id
        - title
        - status
        - createdAt
        - updatedAt
      properties:
        id:
          type: string
          description: TodoのID
          example: "todo-123"
        title:
          type: string
          description: Todoのタイトル
          example: "APIドキュメント作成"
        description:
          type: string
          description: 詳細説明
          example: "OpenAPI形式でAPI仕様書を作成する"
        status:
          type: string
          enum:
            - todo
            - in_progress
            - done
          description: ステータス
          example: "in_progress"
        dueDate:
          type: string
          format: date-time
          description: 期限
          example: "2025-01-31T23:59:59Z"
        priority:
          type: string
          enum:
            - low
            - medium
            - high
          description: 優先度
          example: "high"
        createdAt:
          type: string
          format: date-time
          description: 作成日時
          example: "2025-01-15T10:00:00Z"
        updatedAt:
          type: string
          format: date-time
          description: 更新日時
          example: "2025-01-15T15:30:00Z"

    Error:
      type: object
      required:
        - error
        - message
      properties:
        error:
          type: string
        message:
          type: string
        details:
          type: array
          items:
            type: object

  responses:
    NotFound:
      description: リソースが見つかりません
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: "NOT_FOUND"
            message: "Todoが見つかりません"

  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

security:
  - BearerAuth: []
```

---

## ツールとベストプラクティス

### 推奨ツール

#### 1. Swagger Editor

```bash
# Docker で起動
docker run -p 8080:8080 swaggerapi/swagger-editor

# ブラウザで開く
open http://localhost:8080
```

**用途:**
- OpenAPI仕様書の編集
- リアルタイムプレビュー
- バリデーション

#### 2. Swagger UI

```bash
# Docker で起動
docker run -p 8080:8080 \
  -e SWAGGER_JSON=/openapi.yaml \
  -v $(pwd)/openapi.yaml:/openapi.yaml \
  swaggerapi/swagger-ui

# ブラウザで開く
open http://localhost:8080
```

**用途:**
- APIドキュメントの公開
- インタラクティブなAPI試行

#### 3. Redoc

```bash
# Docker で起動
docker run -p 8080:80 \
  -e SPEC_URL=openapi.yaml \
  -v $(pwd)/openapi.yaml:/usr/share/nginx/html/openapi.yaml \
  redocly/redoc

# ブラウザで開く
open http://localhost:8080
```

**用途:**
- 美しいドキュメント表示
- 印刷・PDF出力

#### 4. Postman

```
1. Postmanを開く
2. Import → OpenAPI 3.0ファイルを選択
3. コレクションが自動生成される
```

**用途:**
- API テスト
- チーム共有
- モック サーバー

#### 5. Stoplight Studio

```
GUI でOpenAPI仕様書を作成・編集
https://stoplight.io/studio
```

**用途:**
- ビジュアルエディタ
- デザインファースト開発

### バリデーション

```bash
# OpenAPIバリデーション
npx @openapitools/openapi-generator-cli validate -i openapi.yaml

# Spectral（より詳細なリント）
npm install -g @stoplight/spectral-cli
spectral lint openapi.yaml
```

### CI/CDでの自動化

```yaml
# .github/workflows/api-docs.yml
name: API Documentation

on:
  push:
    paths:
      - 'openapi.yaml'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Validate OpenAPI
        run: |
          npx @openapitools/openapi-generator-cli validate \
            -i openapi.yaml

      - name: Lint with Spectral
        run: |
          npm install -g @stoplight/spectral-cli
          spectral lint openapi.yaml

  deploy-docs:
    runs-on: ubuntu-latest
    needs: validate
    steps:
      - uses: actions/checkout@v3

      - name: Generate docs
        run: |
          npx redoc-cli bundle openapi.yaml \
            -o docs/index.html

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
```

### ベストプラクティス

```markdown
✅ DO:
- 全てのエンドポイントに説明を付ける
- 実際の例を含める（example）
- エラーケースを網羅する
- バージョニングを明示
- 認証方法を明記
- レート制限を文書化

❌ DON'T:
- コードと乖離させない
- 曖昧な説明
- 例を省略
- エラーレスポンスを省略
- 認証を説明せずに必須にする
```

---

## FAQ

### Q1: OpenAPIとSwaggerの違いは？

**A**: Swaggerは旧名称、OpenAPIは新しい名称です。

```
Swagger 2.0 → OpenAPI 2.0
OpenAPI 3.0 → 最新の標準

推奨: OpenAPI 3.0以降を使用
```

### Q2: YAMLとJSON、どちらで書くべき？

**A**: YAML推奨です。

```
✅ YAML:
- 読みやすい
- コメントが書ける
- 手書きしやすい

❌ JSON:
- 読みにくい
- コメント不可
- 手書きしにくい

ただし、最終的にはJSONに変換可能
```

### Q3: スキーマ定義はどこまで詳細にすべき？

**A**: バリデーションに必要な全ての情報を含めます。

```yaml
✅ Good:
properties:
  email:
    type: string
    format: email
    minLength: 5
    maxLength: 255
    example: "user@example.com"
    description: "ユーザーのメールアドレス（ユニーク制約）"

❌ Bad:
properties:
  email:
    type: string
```

### Q4: 既存APIからOpenAPI仕様書を生成できる？

**A**: 可能です（ただし手動調整が必要）。

```typescript
// Express + Swagger JSDoc
/**
 * @swagger
 * /users:
 *   get:
 *     summary: ユーザー一覧
 *     responses:
 *       200:
 *         description: 成功
 */
app.get('/users', getUsers);

// swagger-jsdoc で自動生成
const swaggerSpec = swaggerJsdoc(options);
```

**他の方法:**
- Postman → Export as OpenAPI
- APIリクエスト記録 → 変換ツール

### Q5: ドキュメントが古くなるのを防ぐには？

**A**: 自動化とプロセス統合が重要です。

```markdown
✅ 対策:
1. コードファースト
   - コードからスキーマ生成（FastAPI、NestJSなど）

2. CI/CDでバリデーション
   - PRでOpenAPIファイル変更を必須化

3. 契約テスト
   - スキーマとAPIレスポンスの一致を自動テスト

4. レビュープロセス
   - API変更時はドキュメント更新もレビュー
```

### Q6: 大規模APIの管理方法は？

**A**: 分割とタグ付けを活用します。

```yaml
# ファイル分割
openapi.yaml
├── paths/
│   ├── users.yaml
│   ├── posts.yaml
│   └── comments.yaml
└── components/
    ├── schemas/
    │   ├── User.yaml
    │   ├── Post.yaml
    │   └── Comment.yaml
    └── responses/
        └── Errors.yaml

# メインファイル
$ref: './paths/users.yaml#/paths/~1users'

# タグで整理
tags:
  - name: Users
    description: ユーザー管理
  - name: Posts
    description: 投稿管理
  - name: Comments
    description: コメント管理
```

### Q7: 認証トークンの取得方法も記載すべき？

**A**: 必ず記載してください。

```yaml
paths:
  /auth/login:
    post:
      summary: ログイン
      description: |
        メールアドレスとパスワードでログインし、
        アクセストークンを取得します。

        取得したトークンは他のエンドポイントで使用してください。
      tags:
        - Authentication
      security: []  # このエンドポイントは認証不要
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - email
                - password
              properties:
                email:
                  type: string
                  format: email
                password:
                  type: string
                  format: password
      responses:
        '200':
          description: ログイン成功
          content:
            application/json:
              schema:
                type: object
                properties:
                  accessToken:
                    type: string
                    description: アクセストークン（有効期限24時間）
                  refreshToken:
                    type: string
                    description: リフレッシュトークン（有効期限30日）
                  expiresIn:
                    type: integer
                    description: トークンの有効期限（秒）
              example:
                accessToken: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
                refreshToken: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
                expiresIn: 86400
```

### Q8: WebSocket APIも記載できる？

**A**: OpenAPI 3.0では部分的に可能です。

```yaml
# HTTP Callbacksとして記載
paths:
  /subscribe:
    post:
      summary: WebSocket接続開始
      description: |
        WebSocketエンドポイント: ws://api.example.com/ws

        接続後、以下の形式でメッセージを送受信します。
      callbacks:
        onMessage:
          '{$request.body#/callbackUrl}':
            post:
              requestBody:
                content:
                  application/json:
                    schema:
                      type: object
                      properties:
                        type:
                          type: string
                          enum:
                            - message
                            - notification
                        data:
                          type: object
```

**より詳細には:**
- AsyncAPIを使用（WebSocket/Kafka専用）

---

## まとめ

### API Documentation成功の5原則

```
1. 完全性
   → 全てのエンドポイントを記載

2. 正確性
   → コードと一致させる

3. 分かりやすさ
   → 例を豊富に含める

4. 鮮度
   → 常に最新に保つ

5. テスト可能性
   → 実際に試せる環境を提供
```

### 次のステップ

```markdown
□ OpenAPI 3.0の基本構文を学ぶ
□ 既存APIのドキュメントを作成
□ Swagger UIで公開
□ CI/CDでバリデーション自動化
□ クライアントSDK生成を試す
```

### 参考リンク

- [OpenAPI Specification](https://spec.openapis.org/oas/v3.0.0)
- [Swagger Editor](https://editor.swagger.io/)
- [Redoc](https://redocly.com/redoc/)
- [OpenAPI Generator](https://openapi-generator.tech/)

---

**このガイドは実際のプロジェクトで進化していきます。**
**フィードバックをお待ちしています！**

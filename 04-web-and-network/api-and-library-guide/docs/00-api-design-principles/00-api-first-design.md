# API First 設計

> API First設計は実装前にAPIの契約を定義するアプローチ。OpenAPI仕様でAPIを先に設計し、フロントエンド・バックエンドが並行開発できる体制を構築する。

## この章で学ぶこと

- [ ] API First設計の哲学と利点を理解する
- [ ] OpenAPI（Swagger）仕様の書き方を把握する
- [ ] モックサーバーを活用した並行開発を学ぶ

---

## 1. API First とは

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

  ツールチェーン:
  設計:      Stoplight Studio, Swagger Editor
  仕様:      OpenAPI 3.1 (YAML/JSON)
  モック:    Prism, MSW, WireMock
  コード生成: openapi-generator, orval, openapi-typescript
  ドキュメント: Redoc, Swagger UI, Scalar
  テスト:    Dredd, Schemathesis
```

---

## 2. OpenAPI 仕様

```yaml
# openapi.yaml
openapi: '3.1.0'
info:
  title: User Management API
  version: '1.0.0'
  description: ユーザー管理のためのRESTful API

servers:
  - url: https://api.example.com/v1
    description: Production
  - url: https://staging-api.example.com/v1
    description: Staging

paths:
  /users:
    get:
      summary: ユーザー一覧の取得
      operationId: listUsers
      tags: [Users]
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: per_page
          in: query
          schema:
            type: integer
            default: 20
            maximum: 100
      responses:
        '200':
          description: ユーザー一覧
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserListResponse'
        '401':
          $ref: '#/components/responses/Unauthorized'

    post:
      summary: ユーザーの作成
      operationId: createUser
      tags: [Users]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
      responses:
        '201':
          description: ユーザー作成成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserResponse'
        '422':
          $ref: '#/components/responses/ValidationError'

  /users/{userId}:
    get:
      summary: ユーザー詳細の取得
      operationId: getUser
      tags: [Users]
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: ユーザー詳細
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserResponse'
        '404':
          $ref: '#/components/responses/NotFound'

components:
  schemas:
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
          enum: [user, admin]
          default: user
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

    PaginationMeta:
      type: object
      properties:
        total:
          type: integer
        page:
          type: integer
        per_page:
          type: integer
        total_pages:
          type: integer

    Error:
      type: object
      properties:
        type:
          type: string
        title:
          type: string
        status:
          type: integer
        detail:
          type: string

  responses:
    Unauthorized:
      description: 認証エラー
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
    NotFound:
      description: リソースが見つからない
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
    ValidationError:
      description: バリデーションエラー
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'

  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

security:
  - bearerAuth: []
```

---

## 3. コード生成

```bash
# TypeScript の型生成（openapi-typescript）
$ npx openapi-typescript openapi.yaml -o src/api/types.ts

# クライアント生成（orval）
$ npx orval --input openapi.yaml --output src/api/client.ts

# サーバースタブ生成（openapi-generator）
$ npx openapi-generator-cli generate \
    -i openapi.yaml \
    -g typescript-express-server \
    -o server/

# モックサーバー起動（Prism）
$ npx @stoplight/prism-cli mock openapi.yaml
# → http://localhost:4010 でモックAPIが起動
```

---

## 4. 設計レビューチェックリスト

```
API設計レビュー項目:

  命名:
  □ リソース名は名詞・複数形か
  □ 一貫したケーシングか（camelCase or snake_case）
  □ URLはケバブケースか

  メソッド:
  □ 適切なHTTPメソッドを使用しているか
  □ 冪等性が正しいか（PUT/DELETE は冪等）

  レスポンス:
  □ 適切なステータスコードを返しているか
  □ エラーレスポンスが統一されているか（RFC 7807）
  □ ページネーションが実装されているか

  セキュリティ:
  □ 認証方式が定義されているか
  □ 入力バリデーションが定義されているか
  □ レート制限が考慮されているか

  互換性:
  □ 破壊的変更がないか
  □ オプショナルフィールドの追加は後方互換か
  □ バージョニング戦略が決まっているか
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| API First | 実装前に仕様を確定、並行開発を実現 |
| OpenAPI | 業界標準のAPI仕様記述形式 |
| コード生成 | 型安全なクライアント/サーバーを自動生成 |
| モックサーバー | Prism等で仕様からモックを自動生成 |

---

## 次に読むべきガイド
→ [[01-naming-and-conventions.md]] — 命名規則と慣例

---

## 参考文献
1. OpenAPI Initiative. "OpenAPI Specification 3.1." 2024.
2. Stoplight. "API Design Guide." 2024.

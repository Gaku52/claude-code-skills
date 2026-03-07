# APIドキュメンテーション

> APIドキュメントはAPIの「顔」であり、開発者が最初に触れるインターフェースである。OpenAPI/Swagger、Redoc、Scalar、自動生成ツール、インタラクティブドキュメント、コード例の設計、ドキュメント駆動開発（Design-First）まで、開発者に愛されるドキュメント作成の全技法を体系的に習得する。

## この章で学ぶこと

- [ ] OpenAPI 3.0/3.1 仕様の詳細構造とドキュメント自動生成の仕組みを理解する
- [ ] Swagger UI、Redoc、Scalar 等の主要レンダリングツールの特性と使い分けを把握する
- [ ] ドキュメント駆動開発（Design-First）のワークフローを実践できるようになる
- [ ] インタラクティブドキュメントの構築方法とカスタマイズ手法を学ぶ
- [ ] 良いドキュメントの構成要素・設計原則・品質メトリクスを体得する
- [ ] コード例の設計原則と多言語対応のベストプラクティスを実装できるようになる

---

## 1. APIドキュメンテーションの全体像

### 1.1 なぜAPIドキュメントが重要なのか

APIドキュメントは、API提供者と利用者の間の「契約書」であると同時に「ユーザーマニュアル」でもある。Postmanの2023年度調査によると、API選定時に「ドキュメントの質」を最重要視する開発者は全体の52%に達し、「APIの機能」(48%)を上回った。つまり、どれほど優れた機能を持つAPIであっても、ドキュメントが貧弱であれば採用されないのが現実である。

```
APIドキュメントの価値チェーン:

  +------------------+     +-------------------+     +------------------+
  |  ドキュメント品質  | --> |  開発者体験 (DX)   | --> |  API採用率向上    |
  +------------------+     +-------------------+     +------------------+
          |                         |                         |
          v                         v                         v
  +------------------+     +-------------------+     +------------------+
  | サポート問い合わせ  | --> |  オンボーディング   | --> |  ビジネス成長     |
  | コスト削減        |     |  時間の短縮        |     |  パートナー拡大   |
  +------------------+     +-------------------+     +------------------+

  ドキュメント品質が高い API:
    - Time to First Call (TTFC): 平均 5分以下
    - サポートチケット: 40% 削減
    - 開発者離脱率: 60% 改善
    - SDK 利用開始率: 3倍向上
```

### 1.2 ドキュメントの4層モデル

APIドキュメントは単一の文書ではなく、利用者の習熟度とユースケースに応じた複数の層で構成される。

```
APIドキュメントの4層構造:

  ┌─────────────────────────────────────────────────┐
  │  Layer 4: コンセプト（Concept）                    │
  │  ・アーキテクチャ設計思想                           │
  │  ・Webhook の仕組み、レート制限の考え方             │
  │  ・セキュリティモデル、データモデル                  │
  │  対象: 設計者・アーキテクト                         │
  ├─────────────────────────────────────────────────┤
  │  Layer 3: チュートリアル（Tutorial）                │
  │  ・「決済システムを作ろう」等の実装ガイド            │
  │  ・ステップバイステップの手順書                     │
  │  ・完成品のソースコード付き                         │
  │  対象: 初級〜中級の開発者                          │
  ├─────────────────────────────────────────────────┤
  │  Layer 2: ガイド（Guide）                          │
  │  ・Quick Start、認証方法、ページネーション          │
  │  ・「○○をするには」の How-to ドキュメント          │
  │  ・ベストプラクティス集                             │
  │  対象: 中級の開発者                                │
  ├─────────────────────────────────────────────────┤
  │  Layer 1: リファレンス（Reference）                 │
  │  ・全エンドポイント一覧                             │
  │  ・パラメータ、レスポンス、エラーコード              │
  │  ・OpenAPI から自動生成が可能                       │
  │  対象: 全ての開発者（最頻利用層）                   │
  └─────────────────────────────────────────────────┘

  良いドキュメント = 4層全てが揃っている
  優れたドキュメント = 4層が相互にリンクされている
```

各層の代表例と特徴を比較する。

| 層 | 目的 | 代表例 | 更新頻度 | 自動生成 |
|---|---|---|---|---|
| リファレンス | エンドポイントの仕様把握 | Stripe API Reference | API変更時 | OpenAPI から可能 |
| ガイド | 特定タスクの実現方法 | Stripe Docs の "Accept a payment" | 機能追加時 | 一部テンプレート化可能 |
| チュートリアル | 学習目的の段階的実装 | Twilio Quest, Plaid Quickstart | 定期的 | サンプルコードの自動テスト可能 |
| コンセプト | 設計思想・アーキテクチャ理解 | Stripe のアーキテクチャ解説 | 大規模変更時 | 手動執筆が主体 |

### 1.3 ドキュメント駆動開発（Design-First / Documentation-Driven Development）

従来の「Code-First」アプローチでは、実装後にドキュメントを作成するため、仕様とドキュメントの乖離が生じやすい。「Design-First」アプローチでは、OpenAPI 仕様書をまず設計し、それを基にコード生成とドキュメント生成を同時に行う。

```
Code-First vs Design-First の比較:

  === Code-First ===

  実装 --> テスト --> OpenAPI 生成 --> ドキュメント生成
    ↑                                       |
    |          仕様とドキュメントの乖離       |
    +---------------------------------------+

  問題点:
  - ドキュメントが後回しになり、陳腐化しやすい
  - API設計のレビューが実装後になる
  - 破壊的変更に気づきにくい

  === Design-First（推奨） ===

                  ┌─→ サーバースタブ生成
  OpenAPI 設計 ──┼─→ クライアントSDK生成
                  ├─→ ドキュメント自動生成
                  ├─→ モックサーバー起動
                  └─→ テスト自動生成

  利点:
  - 仕様が Single Source of Truth
  - フロントエンド・バックエンド並行開発が可能
  - 破壊的変更の早期検出
  - ドキュメントが常に最新
```

Design-First のワークフロー実装例を以下に示す。

```yaml
# .github/workflows/api-design-first.yml
# Design-First ワークフローの CI/CD パイプライン
name: API Design-First Pipeline
on:
  push:
    paths:
      - 'api/openapi.yaml'
      - 'api/components/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # OpenAPI 仕様の構文チェック
      - name: Validate OpenAPI spec
        run: npx @redocly/cli lint api/openapi.yaml

      # 破壊的変更の検出
      - name: Check breaking changes
        run: npx oasdiff breaking api/openapi.yaml --base origin/main

  generate:
    needs: validate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # サーバースタブ生成
      - name: Generate server stub
        run: |
          npx @openapitools/openapi-generator-cli generate \
            -i api/openapi.yaml \
            -g typescript-express-server \
            -o generated/server

      # クライアントSDK生成
      - name: Generate TypeScript SDK
        run: |
          npx @openapitools/openapi-generator-cli generate \
            -i api/openapi.yaml \
            -g typescript-axios \
            -o generated/sdk-typescript

      # Python SDK 生成
      - name: Generate Python SDK
        run: |
          npx @openapitools/openapi-generator-cli generate \
            -i api/openapi.yaml \
            -g python \
            -o generated/sdk-python

  docs:
    needs: validate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Redoc でドキュメント生成
      - name: Build API docs
        run: npx @redocly/cli build-docs api/openapi.yaml -o docs/index.html

      # GitHub Pages にデプロイ
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
```

---

## 2. OpenAPI 仕様の詳細

### 2.1 OpenAPI 3.0 と 3.1 の違い

OpenAPI は REST API の仕様を記述するための標準フォーマットである。バージョン 3.1 は JSON Schema Draft 2020-12 との完全互換を実現した重要なアップデートである。

| 特徴 | OpenAPI 3.0 | OpenAPI 3.1 |
|---|---|---|
| JSON Schema 互換性 | 部分的（独自拡張あり） | 完全互換（Draft 2020-12） |
| nullable の扱い | `nullable: true` | `type: ["string", "null"]` |
| 排他的キーワード | `exclusiveMinimum: true` + `minimum` | `exclusiveMinimum: 値` |
| Webhook 定義 | paths で代用 | `webhooks` キーワード追加 |
| $ref と他キーワードの併用 | 不可 | 可能 |
| if/then/else | 非対応 | 対応 |
| ライセンス identifier | 非対応 | `identifier` フィールド追加 |
| ツールサポート | 広範 | 拡大中（2024年時点で主要ツール対応済み） |

### 2.2 OpenAPI 仕様の構造化設計

大規模なAPIでは、単一の YAML ファイルに全ての仕様を書くとメンテナンスが困難になる。$ref を活用したマルチファイル構成が推奨される。

```yaml
# ディレクトリ構成例
# api/
# ├── openapi.yaml           # ルートファイル
# ├── info.yaml               # API基本情報
# ├── paths/
# │   ├── users.yaml          # /users エンドポイント
# │   ├── orders.yaml         # /orders エンドポイント
# │   └── products.yaml       # /products エンドポイント
# ├── components/
# │   ├── schemas/
# │   │   ├── User.yaml
# │   │   ├── Order.yaml
# │   │   └── Error.yaml
# │   ├── parameters/
# │   │   ├── pagination.yaml
# │   │   └── filtering.yaml
# │   ├── responses/
# │   │   ├── NotFound.yaml
# │   │   └── ValidationError.yaml
# │   └── securitySchemes/
# │       └── bearerAuth.yaml
# └── examples/
#     ├── user-create.yaml
#     └── user-list.yaml

# --- openapi.yaml（ルートファイル） ---
openapi: '3.1.0'
info:
  title: Example Commerce API
  version: '2.0.0'
  description: |
    Example Commerce API は EC プラットフォームの中核機能を提供します。

    ## 認証
    全てのリクエストに Bearer トークンが必要です。
    トークンは [ダッシュボード](https://dashboard.example.com) から取得できます。

    ## レート制限
    | プラン     | リクエスト/分 | バースト上限 |
    |-----------|-------------|------------|
    | Free      | 60          | 10         |
    | Pro       | 600         | 50         |
    | Enterprise| 6000        | 200        |

    ## エラーハンドリング
    全てのエラーは統一された形式で返されます。
    詳細は [エラーリファレンス](#tag/Errors) を参照してください。
  contact:
    name: API Support
    email: api-support@example.com
    url: https://support.example.com
  license:
    name: MIT
    identifier: MIT
  x-logo:
    url: https://example.com/logo.png
    altText: Example Commerce API

servers:
  - url: https://api.example.com/v2
    description: 本番環境
  - url: https://sandbox.api.example.com/v2
    description: サンドボックス環境（テスト用）
  - url: http://localhost:3000/v2
    description: ローカル開発環境

tags:
  - name: Users
    description: |
      ユーザーの作成・取得・更新・削除を行います。
      ユーザーは全てのリソースの所有者となる基本エンティティです。
    externalDocs:
      description: ユーザー管理ガイド
      url: https://docs.example.com/guides/users
  - name: Orders
    description: |
      注文の作成・取得・管理を行います。
      注文ライフサイクルの詳細はコンセプトガイドを参照してください。
  - name: Products
    description: 商品の作成・取得・管理を行います。

security:
  - bearerAuth: []

paths:
  /users:
    $ref: './paths/users.yaml'
  /orders:
    $ref: './paths/orders.yaml'
  /products:
    $ref: './paths/products.yaml'

webhooks:
  orderCompleted:
    post:
      summary: 注文完了通知
      description: 注文が完了した際に送信される Webhook イベント
      operationId: onOrderCompleted
      tags: [Webhooks]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: './components/schemas/OrderCompletedEvent.yaml'
      responses:
        '200':
          description: Webhook の受信確認

components:
  securitySchemes:
    bearerAuth:
      $ref: './components/securitySchemes/bearerAuth.yaml'
```

### 2.3 スキーマ設計のベストプラクティス

OpenAPI のスキーマ定義は、ドキュメントの品質に直結する。以下に充実したスキーマ定義の例を示す。

```yaml
# components/schemas/User.yaml
# ユーザースキーマの詳細定義
type: object
title: User
description: |
  ユーザーはシステムの基本エンティティです。
  全てのリソース（注文、レビュー等）はユーザーに紐づきます。
required:
  - id
  - email
  - name
  - status
  - createdAt
properties:
  id:
    type: string
    format: uuid
    description: ユーザーの一意識別子（UUID v4）
    example: "550e8400-e29b-41d4-a716-446655440000"
    readOnly: true
  email:
    type: string
    format: email
    description: |
      ユーザーのメールアドレス。
      システム全体で一意である必要があります。
      変更後は確認メールが送信されます。
    example: "taro.yamada@example.com"
    maxLength: 254
  name:
    type: string
    description: ユーザーの表示名（2〜100文字）
    example: "Taro Yamada"
    minLength: 2
    maxLength: 100
  status:
    type: string
    description: ユーザーのアカウント状態
    enum:
      - active
      - inactive
      - suspended
      - pending_verification
    example: "active"
    x-enum-descriptions:
      active: 有効なアカウント
      inactive: 無効化されたアカウント
      suspended: 利用規約違反等で停止されたアカウント
      pending_verification: メールアドレス確認待ち
  role:
    type: string
    description: ユーザーの権限レベル
    enum:
      - admin
      - manager
      - user
    default: user
    example: "user"
  avatar:
    type:
      - string
      - "null"
    format: uri
    description: プロフィール画像の URL（未設定の場合は null）
    example: "https://cdn.example.com/avatars/550e8400.jpg"
  metadata:
    type: object
    description: |
      任意のキー・バリューペアを格納できるメタデータ。
      最大 50 個のキーを設定可能。各キーは 40 文字以内、値は 500 文字以内。
    additionalProperties:
      type: string
      maxLength: 500
    maxProperties: 50
    example:
      department: "Engineering"
      employee_id: "EMP-12345"
  createdAt:
    type: string
    format: date-time
    description: アカウント作成日時（ISO 8601 形式、UTC）
    example: "2024-01-15T09:30:00Z"
    readOnly: true
  updatedAt:
    type: string
    format: date-time
    description: 最終更新日時（ISO 8601 形式、UTC）
    example: "2024-06-20T14:22:00Z"
    readOnly: true
```

### 2.4 エンドポイント定義の充実化

ドキュメント品質を左右するのは、エンドポイント定義の詳細さである。

```yaml
# paths/users.yaml - ユーザーエンドポイントの完全な定義例
get:
  summary: ユーザー一覧の取得
  description: |
    登録済みユーザーの一覧を取得します。
    Cursor ベースのページネーションに対応しています。

    ### 権限
    - `users:read` スコープが必要

    ### レート制限
    - 100 リクエスト/分

    ### ソート
    `sort` パラメータで並び順を指定できます。
    `-` プレフィックスで降順になります（例: `-createdAt`）。

    ### フィルタリング
    `status` パラメータで特定の状態のユーザーのみ取得できます。
    複数指定する場合はカンマ区切りにします（例: `status=active,inactive`）。
  operationId: listUsers
  tags: [Users]
  parameters:
    - name: cursor
      in: query
      description: |
        ページネーションカーソル。
        前回のレスポンスの `meta.nextCursor` の値を指定してください。
        初回リクエスト時は省略します。
      schema:
        type: string
      example: "eyJpZCI6MTAwfQ"
    - name: limit
      in: query
      description: 1ページあたりの取得件数
      schema:
        type: integer
        minimum: 1
        maximum: 100
        default: 20
      example: 20
    - name: sort
      in: query
      description: |
        ソート順を指定します。
        利用可能なフィールド: `name`, `email`, `createdAt`, `updatedAt`
        降順の場合は `-` をプレフィックスにします。
      schema:
        type: string
        enum:
          - name
          - -name
          - email
          - -email
          - createdAt
          - -createdAt
          - updatedAt
          - -updatedAt
        default: -createdAt
    - name: status
      in: query
      description: ユーザー状態によるフィルタリング（カンマ区切りで複数指定可能）
      schema:
        type: string
      example: "active,inactive"
    - name: search
      in: query
      description: 名前またはメールアドレスによる部分一致検索
      schema:
        type: string
        minLength: 2
      example: "yamada"
  responses:
    '200':
      description: ユーザー一覧の取得に成功
      headers:
        X-RateLimit-Limit:
          description: レート制限の上限値
          schema:
            type: integer
          example: 100
        X-RateLimit-Remaining:
          description: 残りリクエスト数
          schema:
            type: integer
          example: 95
        X-RateLimit-Reset:
          description: レート制限リセット日時（Unix タイムスタンプ）
          schema:
            type: integer
          example: 1719900000
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/UserListResponse'
          examples:
            default:
              summary: 基本的なレスポンス
              value:
                data:
                  - id: "550e8400-e29b-41d4-a716-446655440000"
                    name: "Taro Yamada"
                    email: "taro@example.com"
                    status: "active"
                    role: "user"
                    createdAt: "2024-01-15T09:30:00Z"
                  - id: "660f9500-f30c-52e5-b827-557766551111"
                    name: "Hanako Sato"
                    email: "hanako@example.com"
                    status: "active"
                    role: "manager"
                    createdAt: "2024-02-20T11:00:00Z"
                meta:
                  total: 150
                  hasNextPage: true
                  nextCursor: "eyJpZCI6MTIwfQ"
            empty:
              summary: 検索結果が空の場合
              value:
                data: []
                meta:
                  total: 0
                  hasNextPage: false
                  nextCursor: null
    '400':
      description: リクエストパラメータが不正
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
          example:
            error:
              code: "INVALID_PARAMETER"
              message: "limit は 1 以上 100 以下で指定してください"
              details:
                - field: "limit"
                  value: 200
                  constraint: "maximum: 100"
    '401':
      $ref: '#/components/responses/Unauthorized'
    '429':
      $ref: '#/components/responses/RateLimited'

post:
  summary: ユーザーの作成
  description: |
    新しいユーザーを作成します。
    作成後、確認メールが送信されます。

    ### 権限
    - `users:write` スコープが必要

    ### 冪等性
    `Idempotency-Key` ヘッダーを指定することで、
    ネットワークエラー時の重複作成を防止できます。
  operationId: createUser
  tags: [Users]
  parameters:
    - name: Idempotency-Key
      in: header
      description: |
        冪等性キー（UUID v4 を推奨）。
        同一キーで 24 時間以内に再リクエストした場合、
        最初のリクエストの結果が返されます。
      schema:
        type: string
        format: uuid
      example: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
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
              name: "Taro Yamada"
              email: "taro@example.com"
          with_metadata:
            summary: メタデータ付きユーザー作成
            value:
              name: "Taro Yamada"
              email: "taro@example.com"
              role: "manager"
              metadata:
                department: "Engineering"
                employee_id: "EMP-12345"
  responses:
    '201':
      description: ユーザーの作成に成功
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/User'
    '409':
      description: メールアドレスが既に使用されている
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
          example:
            error:
              code: "DUPLICATE_EMAIL"
              message: "指定されたメールアドレスは既に登録されています"
    '422':
      description: バリデーションエラー
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ValidationErrorResponse'
          example:
            error:
              code: "VALIDATION_ERROR"
              message: "入力値に誤りがあります"
              details:
                - field: "email"
                  code: "INVALID_FORMAT"
                  message: "有効なメールアドレスを入力してください"
                - field: "name"
                  code: "TOO_SHORT"
                  message: "名前は2文字以上で入力してください"
```

---

## 3. ドキュメント生成ツール比較

### 3.1 主要ツールの詳細比較

OpenAPI 仕様からドキュメントを生成するツールは複数存在する。プロジェクトの要件に応じた適切な選択が重要である。

| 特性 | Swagger UI | Redoc | Scalar | Stoplight Elements |
|---|---|---|---|---|
| レイアウト | 1カラム | 3カラム | 3カラム | 3カラム |
| Try it out 機能 | 標準搭載 | 有料プラン | 標準搭載 | 標準搭載 |
| ダークモード | プラグイン必要 | 対応 | 標準搭載 | 対応 |
| SEO 対応 | 弱い（SPA） | 強い（SSR可） | 強い | 普通 |
| バンドルサイズ | 約 1.5MB | 約 500KB | 約 300KB | 約 700KB |
| カスタマイズ性 | CSS/プラグイン | 限定的 | テーマシステム | React コンポーネント |
| コード例自動生成 | なし | なし | 多言語対応 | なし |
| React 統合 | @swagger-api/swagger-ui-react | redoc の React ラッパー | @scalar/api-reference-react | @stoplight/elements |
| 料金 | 無料（OSS） | 基本無料/Pro有料 | 無料（OSS） | 基本無料/Pro有料 |
| GitHub Stars (2024) | 25k+ | 22k+ | 8k+ | 4k+ |

### 3.2 Swagger UI の設定と拡張

```html
<!-- Swagger UI の基本セットアップ -->
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>API Documentation - Swagger UI</title>
  <link rel="stylesheet"
    href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
  <style>
    /* カスタムスタイル */
    .swagger-ui .topbar { display: none; } /* ヘッダーバーを非表示 */
    .swagger-ui .info .title {
      font-size: 2rem;
      color: #1a1a2e;
    }
    .swagger-ui .opblock.opblock-get {
      border-color: #61affe;
      background: rgba(97, 175, 254, 0.05);
    }
    .swagger-ui .opblock.opblock-post {
      border-color: #49cc90;
      background: rgba(73, 204, 144, 0.05);
    }
    .swagger-ui .opblock.opblock-delete {
      border-color: #f93e3e;
      background: rgba(249, 62, 62, 0.05);
    }
    /* レスポンシブ対応 */
    @media (max-width: 768px) {
      .swagger-ui .wrapper { padding: 0 10px; }
    }
  </style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    SwaggerUIBundle({
      url: '/api/openapi.yaml',
      dom_id: '#swagger-ui',
      deepLinking: true,
      // 認証情報のプリセット（開発環境用）
      onComplete: function() {
        // テスト用トークンを自動設定
        if (window.location.hostname === 'localhost') {
          ui.preauthorizeApiKey('bearerAuth', 'sk_test_abc123');
        }
      },
      // レイアウト設定
      layout: 'BaseLayout',
      // フィルタ機能を有効化
      filter: true,
      // Try it out をデフォルトで有効化
      tryItOutEnabled: true,
      // リクエスト/レスポンスの表示形式
      defaultModelsExpandDepth: 2,
      defaultModelExpandDepth: 2,
      // バリデーション
      validatorUrl: null, // 外部バリデータを無効化
      // 操作のソート
      operationsSorter: 'alpha',
      tagsSorter: 'alpha',
    });
  </script>
</body>
</html>
```

### 3.3 Redoc の設定とカスタマイズ

```html
<!-- Redoc の高度なセットアップ -->
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>API Documentation - Redoc</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { margin: 0; padding: 0; }
  </style>
</head>
<body>
  <div id="redoc-container"></div>
  <script src="https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js"></script>
  <script>
    Redoc.init('/api/openapi.yaml', {
      // テーマのカスタマイズ
      theme: {
        colors: {
          primary: { main: '#5B21B6' },        // メインカラー
          success: { main: '#059669' },         // 成功色
          warning: { main: '#D97706' },         // 警告色
          error: { main: '#DC2626' },           // エラー色
          text: { primary: '#1F2937' },         // テキスト色
        },
        typography: {
          fontSize: '15px',
          fontFamily: '"Noto Sans JP", "Helvetica Neue", sans-serif',
          headings: {
            fontFamily: '"Noto Sans JP", "Helvetica Neue", sans-serif',
            fontWeight: '700',
          },
          code: {
            fontSize: '13px',
            fontFamily: '"JetBrains Mono", "Fira Code", monospace',
          },
        },
        sidebar: {
          width: '280px',
          backgroundColor: '#F9FAFB',
          textColor: '#374151',
          activeTextColor: '#5B21B6',
        },
        rightPanel: {
          backgroundColor: '#1F2937',
          textColor: '#F3F4F6',
        },
      },
      // 機能設定
      scrollYOffset: 0,
      hideDownloadButton: false,
      hideHostname: false,
      hideLoading: false,
      nativeScrollbars: true,
      pathInMiddlePanel: true,
      requiredPropsFirst: true,
      sortPropsAlphabetically: false,
      expandResponses: '200',
      jsonSampleExpandLevel: 3,
      // SEO 設定
      generateTagDescriptions: true,
    }, document.getElementById('redoc-container'));
  </script>
</body>
</html>
```

### 3.4 Scalar の設定（モダンな選択肢）

```typescript
// Express.js での Scalar 統合例
import express from 'express';
import { apiReference } from '@scalar/express-api-reference';

const app = express();

// OpenAPI 仕様の配信
app.get('/api/openapi.json', (req, res) => {
  res.sendFile('./api/openapi.json', { root: __dirname });
});

// Scalar API Reference の設定
app.use(
  '/docs',
  apiReference({
    spec: {
      url: '/api/openapi.json',
    },
    theme: 'purple',          // テーマ: default, alternate, moon, purple, solarized
    layout: 'modern',         // レイアウト: modern, classic
    darkMode: true,            // ダークモード初期状態
    hideModels: false,         // スキーマモデルの表示
    hideDownloadButton: false, // ダウンロードボタン
    hideTestRequestButton: false,
    // 認証設定
    authentication: {
      preferredSecurityScheme: 'bearerAuth',
      // テスト用トークン（開発環境のみ）
      apiKey: {
        token: process.env.NODE_ENV === 'development'
          ? 'sk_test_abc123'
          : '',
      },
    },
    // メタデータ
    metaData: {
      title: 'Example Commerce API',
      description: 'EC プラットフォーム API リファレンス',
      ogDescription: 'Example Commerce API の開発者向けドキュメント',
    },
    // カスタム CSS
    customCss: `
      .scalar-app {
        --scalar-font: 'Noto Sans JP', sans-serif;
      }
    `,
  })
);

app.listen(3000, () => {
  console.log('API docs available at http://localhost:3000/docs');
});
```

---

## 4. コード例の設計原則

### 4.1 良いコード例の要件

APIドキュメントにおけるコード例は、開発者が最も参照する部分である。以下の原則を厳守する。

```
コード例の6原則:

  ┌─────────────────────────────────────────────────────┐
  │ 1. 即座に実行可能                                     │
  │    コピー&ペーストでそのまま動作すること               │
  │    必要な import / require を省略しない               │
  ├─────────────────────────────────────────────────────┤
  │ 2. 現実的な値を使用                                   │
  │    foo, bar, test ではなく具体的な値                  │
  │    "Taro Yamada", "taro@example.com" 等              │
  ├─────────────────────────────────────────────────────┤
  │ 3. エラーハンドリングを含む                            │
  │    成功パスだけでなく失敗パスも示す                    │
  │    try-catch / error callback を含める               │
  ├─────────────────────────────────────────────────────┤
  │ 4. 多言語対応                                         │
  │    最低限: curl + JavaScript + Python                │
  │    理想: + Go + Ruby + Java + PHP                    │
  ├─────────────────────────────────────────────────────┤
  │ 5. プログレッシブ・ディスクロージャー                  │
  │    基本例 → 詳細例 → 高度な例の段階的開示             │
  │    初心者が圧倒されないよう配慮する                    │
  ├─────────────────────────────────────────────────────┤
  │ 6. 出力結果を含む                                     │
  │    実行結果のレスポンス例を添える                      │
  │    ステータスコード、ヘッダーも示す                    │
  └─────────────────────────────────────────────────────┘
```

### 4.2 多言語コード例の実装

```bash
# === curl ===
# ユーザーの作成
curl -X POST https://api.example.com/v2/users \
  -H "Authorization: Bearer sk_test_abc123" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: $(uuidgen)" \
  -d '{
    "name": "Taro Yamada",
    "email": "taro@example.com",
    "role": "user",
    "metadata": {
      "department": "Engineering"
    }
  }'

# レスポンス例（HTTP 201 Created）
# {
#   "id": "550e8400-e29b-41d4-a716-446655440000",
#   "name": "Taro Yamada",
#   "email": "taro@example.com",
#   "status": "pending_verification",
#   "role": "user",
#   "avatar": null,
#   "metadata": { "department": "Engineering" },
#   "createdAt": "2024-07-01T10:00:00Z",
#   "updatedAt": "2024-07-01T10:00:00Z"
# }
```

```javascript
// === JavaScript / TypeScript (SDK) ===
import { ExampleClient, ValidationError, RateLimitError } from '@example/sdk';

const client = new ExampleClient({
  apiKey: process.env.EXAMPLE_API_KEY,
  // オプション: カスタム設定
  timeout: 30000,          // タイムアウト: 30秒
  maxRetries: 3,           // 最大リトライ回数
  baseURL: 'https://api.example.com/v2',
});

// ユーザーの作成
async function createUser() {
  try {
    const user = await client.users.create({
      name: 'Taro Yamada',
      email: 'taro@example.com',
      role: 'user',
      metadata: {
        department: 'Engineering',
      },
    }, {
      idempotencyKey: crypto.randomUUID(),
    });

    console.log('Created user:', user.id);
    // => "550e8400-e29b-41d4-a716-446655440000"
    return user;

  } catch (error) {
    if (error instanceof ValidationError) {
      // バリデーションエラー（422）
      console.error('Validation failed:', error.errors);
      // => [{ field: "email", code: "INVALID_FORMAT", message: "..." }]
    } else if (error instanceof RateLimitError) {
      // レート制限（429）
      console.error(`Rate limited. Retry after ${error.retryAfter}s`);
    } else {
      // その他のエラー
      console.error('Unexpected error:', error.message);
    }
    throw error;
  }
}
```

```python
# === Python (SDK) ===
import os
import uuid
from example_sdk import ExampleClient
from example_sdk.errors import ValidationError, RateLimitError

client = ExampleClient(
    api_key=os.environ["EXAMPLE_API_KEY"],
    timeout=30.0,        # タイムアウト: 30秒
    max_retries=3,       # 最大リトライ回数
)

# ユーザーの作成
def create_user():
    try:
        user = client.users.create(
            name="Taro Yamada",
            email="taro@example.com",
            role="user",
            metadata={
                "department": "Engineering",
            },
            idempotency_key=str(uuid.uuid4()),
        )
        print(f"Created user: {user.id}")
        # => "550e8400-e29b-41d4-a716-446655440000"
        return user

    except ValidationError as e:
        # バリデーションエラー（422）
        print(f"Validation failed: {e.errors}")
        raise
    except RateLimitError as e:
        # レート制限（429）
        print(f"Rate limited. Retry after {e.retry_after}s")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    create_user()
```

```go
// === Go ===
package main

import (
    "context"
    "fmt"
    "log"
    "os"
    "time"

    "github.com/example/sdk-go"
    "github.com/google/uuid"
)

func main() {
    client := example.NewClient(
        os.Getenv("EXAMPLE_API_KEY"),
        example.WithTimeout(30*time.Second),
        example.WithMaxRetries(3),
    )

    ctx := context.Background()

    // ユーザーの作成
    user, err := client.Users.Create(ctx, &example.CreateUserParams{
        Name:  "Taro Yamada",
        Email: "taro@example.com",
        Role:  example.RoleUser,
        Metadata: map[string]string{
            "department": "Engineering",
        },
    }, example.WithIdempotencyKey(uuid.New().String()))

    if err != nil {
        var validationErr *example.ValidationError
        var rateLimitErr *example.RateLimitError

        switch {
        case errors.As(err, &validationErr):
            log.Printf("Validation failed: %v", validationErr.Errors)
        case errors.As(err, &rateLimitErr):
            log.Printf("Rate limited. Retry after %ds", rateLimitErr.RetryAfter)
        default:
            log.Fatalf("Unexpected error: %v", err)
        }
        return
    }

    fmt.Printf("Created user: %s\n", user.ID)
    // => "550e8400-e29b-41d4-a716-446655440000"
}
```

---

## 5. Quick Start ガイドの設計

### 5.1 効果的な Quick Start の構造

Quick Start ガイドは「5分以内に最初のAPIコールを成功させる」ことを目標とする。以下に完全な Quick Start の例を示す。

```markdown
# Quick Start

## 前提条件
- Node.js 18 以上
- Example アカウント（[無料登録](https://dashboard.example.com/signup)）

## ステップ 1: API キーの取得（1分）

[ダッシュボード](https://dashboard.example.com/api-keys) にログインし、
テスト用 API キーを取得します。

テスト用キーは `sk_test_` から始まります。
本番用キーは `sk_live_` から始まります。

> **注意**: テスト用キーで作成されたデータは本番環境には影響しません。

## ステップ 2: SDK のインストール（30秒）

npm install @example/sdk
# または
yarn add @example/sdk
# または
pnpm add @example/sdk

## ステップ 3: 最初の API コール（2分）

import { ExampleClient } from '@example/sdk';

const client = new ExampleClient({
  apiKey: process.env.EXAMPLE_API_KEY, // sk_test_abc123
});

// ユーザーの作成
const user = await client.users.create({
  name: 'Taro Yamada',
  email: 'taro@example.com',
});
console.log('Created:', user.id);

// ユーザーの取得
const fetched = await client.users.get(user.id);
console.log('Name:', fetched.name); // => "Taro Yamada"

// ユーザー一覧の取得
const { data: users } = await client.users.list({ limit: 10 });
console.log('Total users:', users.length);

## ステップ 4: エラーハンドリング（1分）

import { ExampleClient, ExampleError, ValidationError } from '@example/sdk';

try {
  await client.users.create({ name: '', email: 'invalid' });
} catch (error) {
  if (error instanceof ValidationError) {
    // フィールドごとのエラー内容を確認
    for (const detail of error.errors) {
      console.error(`${detail.field}: ${detail.message}`);
    }
  } else if (error instanceof ExampleError) {
    console.error(`API Error [${error.code}]: ${error.message}`);
  }
}

## 次のステップ
- [認証ガイド](/guides/authentication) - OAuth 2.0 の設定
- [ページネーション](/guides/pagination) - 大量データの取得
- [Webhook](/guides/webhooks) - リアルタイム通知の設定
- [API リファレンス](/reference) - 全エンドポイントの詳細
```

---

## 6. Changelog と Migration Guide

### 6.1 効果的な Changelog の構成

Changelog は単なる変更履歴ではなく、開発者がバージョンアップの影響範囲と対応方法を判断するための重要文書である。Keep a Changelog フォーマットを基盤とし、API 固有の要素を追加する。

```markdown
# Changelog

全ての注目すべき変更はこのファイルに記録されます。
形式は [Keep a Changelog](https://keepachangelog.com/) に準拠しています。

## [2.0.0] - 2024-07-01 — メジャーアップデート

### 破壊的変更 (BREAKING CHANGES)

#### メソッド名の変更
リソースベースの命名規則に統一しました。

| v1.x (旧) | v2.x (新) |
|---|---|
| `client.getUser(id)` | `client.users.get(id)` |
| `client.listUsers(params)` | `client.users.list(params)` |
| `client.createUser(data)` | `client.users.create(data)` |
| `client.updateUser(id, data)` | `client.users.update(id, data)` |
| `client.deleteUser(id)` | `client.users.delete(id)` |

#### エラー型のリネーム
- `ApiError` → `ExampleError`
- `HttpError` → `ExampleHttpError`
- `TimeoutError` → `ExampleTimeoutError`

#### ランタイム要件
- Node.js 16 のサポートを終了（Node.js 18+ が必要）
- Python 3.8 のサポートを終了（Python 3.9+ が必要）

### 追加 (Added)
- `client.users.listAll()` で自動ページネーション（AsyncIterator）
- リトライ設定のカスタマイズ（`maxRetries`, `retryDelay`）
- Webhook 署名検証ヘルパー `client.webhooks.verify(payload, signature)`
- TypeScript: 全レスポンス型のエクスポート

### 変更 (Changed)
- デフォルトタイムアウトを 10秒 → 30秒に変更
- ページネーションのデフォルト件数を 10 → 20に変更

### 修正 (Fixed)
- タイムアウト時のメモリリーク (#234)
- 大量の並行リクエスト時のコネクションプール枯渇 (#256)
- 日本語文字列のエンコーディング問題 (#271)

### 非推奨 (Deprecated)
- `client.users.find(query)` は v3.0 で削除予定
  → `client.users.list({ search: query })` を使用してください

## [1.5.0] - 2024-04-15

### 追加
- `client.orders.refund(orderId, params)` メソッド
- リクエストログのカスタムハンドラ設定
```

### 6.2 Migration Guide の設計

バージョン間の移行ガイドは、機械的な差分だけでなく、移行戦略と検証手順を含めるべきである。

```markdown
# v1.x から v2.x への移行ガイド

## 移行の概要

| 項目 | 詳細 |
|---|---|
| 推定作業時間 | 小規模プロジェクト: 30分、大規模: 2時間 |
| 破壊的変更の数 | 8 件 |
| 自動移行ツール | あり（`npx @example/migrate v1-to-v2`） |
| v1.x のサポート期限 | 2025-01-01（セキュリティパッチのみ） |

## 自動移行ツール

npx @example/migrate v1-to-v2 --dry-run  # プレビュー
npx @example/migrate v1-to-v2            # 実行

## 手動移行手順

### ステップ 1: SDK のアップデート

npm install @example/sdk@2

### ステップ 2: メソッド呼び出しの更新

// Before (v1.x)
const user = await client.getUser('user_123');
const users = await client.listUsers({ page: 1 });

// After (v2.x)
const user = await client.users.get('user_123');
const users = await client.users.list({ cursor: null });

### ステップ 3: エラーハンドリングの更新

// Before (v1.x)
import { ApiError } from '@example/sdk';
try { ... } catch (e) {
  if (e instanceof ApiError) { ... }
}

// After (v2.x)
import { ExampleError } from '@example/sdk';
try { ... } catch (e) {
  if (e instanceof ExampleError) { ... }
}

### ステップ 4: ページネーションの更新

// Before (v1.x) - オフセットベース
const page1 = await client.listUsers({ page: 1, perPage: 20 });
const page2 = await client.listUsers({ page: 2, perPage: 20 });

// After (v2.x) - カーソルベース
const page1 = await client.users.list({ limit: 20 });
const page2 = await client.users.list({
  limit: 20,
  cursor: page1.meta.nextCursor,
});

// v2.x 推奨: 自動ページネーション
for await (const user of client.users.listAll()) {
  console.log(user.name);
}

## 検証チェックリスト
- [ ] 全てのAPIコールが正常に動作する
- [ ] エラーハンドリングが正しく機能する
- [ ] ページネーションが期待通り動作する
- [ ] Webhook の受信が正常に処理される
- [ ] TypeScript の型エラーがない
```

---

## 7. ドキュメント品質メトリクスと評価

### 7.1 定量的品質指標

ドキュメントの品質を主観的な評価に頼らず、定量的に計測するためのフレームワークを導入する。

```
ドキュメント品質スコアカード:

  ┌─────────────────────────────────────────────────┐
  │  カテゴリ A: 完全性（Completeness）  配点: 30   │
  ├─────────────────────────────────────────────────┤
  │  □ 全エンドポイントが文書化されている     (5)    │
  │  □ 全パラメータに説明がある               (5)    │
  │  □ 全レスポンスコードが説明されている      (5)    │
  │  □ 認証方法が説明されている                (5)    │
  │  □ エラーコード一覧がある                  (5)    │
  │  □ Quick Start ガイドがある                (5)    │
  ├─────────────────────────────────────────────────┤
  │  カテゴリ B: 正確性（Accuracy）      配点: 25   │
  ├─────────────────────────────────────────────────┤
  │  □ コード例が実際に動作する               (10)    │
  │  □ レスポンス例が実API出力と一致する       (5)    │
  │  □ パラメータ制約が正確                    (5)    │
  │  □ 最終更新日が6ヶ月以内                   (5)    │
  ├─────────────────────────────────────────────────┤
  │  カテゴリ C: 利便性（Usability）     配点: 25   │
  ├─────────────────────────────────────────────────┤
  │  □ 検索機能がある                          (5)    │
  │  □ Try it out 機能がある                   (5)    │
  │  □ 多言語コード例がある                    (5)    │
  │  □ モバイル対応                            (5)    │
  │  □ ダークモード対応                        (5)    │
  ├─────────────────────────────────────────────────┤
  │  カテゴリ D: 開発者体験（DX）        配点: 20   │
  ├─────────────────────────────────────────────────┤
  │  □ TTFC が 5分以内                         (5)    │
  │  □ SDK のインストール手順がある             (5)    │
  │  □ Changelog が維持されている               (5)    │
  │  □ 移行ガイドがある                        (5)    │
  └─────────────────────────────────────────────────┘

  評価基準:
    90-100: 優秀（Stripe, Twilio レベル）
    70-89:  良好（多くの商用APIのレベル）
    50-69:  改善必要
    0-49:   重大な問題あり
```

### 7.2 ドキュメント品質チェックリスト

```
API ドキュメント出荷前チェックリスト:

  === 必須要素 ===
  □ Quick Start（5分以内に最初のAPIコール成功）
  □ 認証方法の説明（APIキー取得方法を含む）
  □ 全エンドポイントのリファレンス
  □ 各エンドポイントのリクエスト/レスポンス例
  □ エラーコードの一覧と対処法
  □ レート制限の説明（プランごとの上限値）
  □ SDK のインストールと初期化手順
  □ ページネーションの使い方
  □ Webhook の設定方法（該当する場合）
  □ Changelog（Keep a Changelog 形式）

  === 品質基準 ===
  □ コード例がコピー&ペーストで動作する
  □ 全てのパラメータに説明・型・制約がある
  □ 成功/エラーの両方のレスポンス例がある
  □ 複数言語のコード例（最低 curl + 1 SDK）
  □ 検索機能がある
  □ レスポンシブデザイン（モバイル対応）
  □ ダークモード対応
  □ 定期的に更新されている（最終更新日が明記）

  === 高度な要素（推奨） ===
  □ インタラクティブな Try it out 機能
  □ サンドボックス環境の提供
  □ OpenAPI 仕様ファイルのダウンロード
  □ SDK の自動生成設定
  □ Postman Collection の提供
  □ GraphQL Playground（GraphQL の場合）
  □ 変更通知の仕組み（RSS, メール等）
```

---

## 8. インタラクティブドキュメントの実装

### 8.1 Try it out 機能の設計

インタラクティブドキュメントの核心は「Try it out」機能である。開発者がブラウザ上で直接APIを呼び出して動作を確認できるこの機能は、ドキュメントの理解を飛躍的に向上させる。

```
Try it out 機能のアーキテクチャ:

  ┌──────────────────────────────────────────────────┐
  │                   ブラウザ                        │
  │                                                   │
  │  ┌─────────────┐  ┌──────────────────────────┐   │
  │  │ パラメータ    │  │  レスポンス表示           │   │
  │  │ 入力フォーム  │  │  - ステータスコード       │   │
  │  │              │  │  - ヘッダー              │   │
  │  │  name: [...] │  │  - ボディ (JSON)         │   │
  │  │  email:[...] │  │  - レスポンス時間         │   │
  │  │              │  │                          │   │
  │  │ [Execute]    │  │  200 OK  (142ms)         │   │
  │  └──────┬───────┘  │  { "id": "user_123", ... │   │
  │         │          │  }                        │   │
  │         v          └──────────────────────────┘   │
  │  ┌──────────────┐                                 │
  │  │ CORS Proxy   │  ← 本番APIへの直接アクセスが     │
  │  │ (必要に応じて) │    不可能な場合に必要            │
  │  └──────┬───────┘                                 │
  └─────────┼─────────────────────────────────────────┘
            │
            v
  ┌──────────────────┐
  │  API サーバー      │
  │  (sandbox 環境)   │
  │                   │
  │  重要: Try it out │
  │  は sandbox に    │
  │  接続すること      │
  └──────────────────┘
```

### 8.2 サンドボックス環境の設計

Try it out 機能を安全に提供するには、本番環境とは分離されたサンドボックス環境が不可欠である。

```typescript
// サンドボックス環境のミドルウェア実装例
import express from 'express';
import rateLimit from 'express-rate-limit';

const sandboxApp = express();

// サンドボックス固有のミドルウェア
sandboxApp.use((req, res, next) => {
  // サンドボックスであることを明示するヘッダー
  res.setHeader('X-Environment', 'sandbox');
  res.setHeader('X-Sandbox-Warning',
    'This is a test environment. Data is reset daily.');
  next();
});

// サンドボックス用の厳格なレート制限
const sandboxLimiter = rateLimit({
  windowMs: 60 * 1000,    // 1分
  max: 30,                  // 30リクエスト/分
  message: {
    error: {
      code: 'SANDBOX_RATE_LIMIT',
      message: 'サンドボックス環境のレート制限に達しました（30リクエスト/分）',
      retryAfter: 60,
    },
  },
  standardHeaders: true,
  legacyHeaders: false,
});

sandboxApp.use(sandboxLimiter);

// サンドボックスデータの自動リセット（毎日 UTC 0:00）
import cron from 'node-cron';

cron.schedule('0 0 * * *', async () => {
  console.log('Resetting sandbox data...');
  await resetSandboxDatabase();
  console.log('Sandbox data reset complete');
});

// テスト用APIキーの自動発行
sandboxApp.post('/sandbox/api-keys', async (req, res) => {
  const key = generateSandboxApiKey();
  res.json({
    apiKey: key,
    expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
    limits: {
      requestsPerMinute: 30,
      dataRetention: '24 hours',
    },
    note: 'このキーはサンドボックス環境専用です。本番環境では使用できません。',
  });
});
```

### 8.3 Postman Collection の自動生成

多くの開発者は Postman を日常的に使用しているため、Postman Collection の提供は有効である。

```javascript
// OpenAPI から Postman Collection を生成するスクリプト
// scripts/generate-postman-collection.js
import { readFileSync, writeFileSync } from 'fs';
import Converter from 'openapi-to-postmanv2';

const openapiSpec = readFileSync('./api/openapi.yaml', 'utf-8');

const options = {
  schemaFaker: true,
  requestNameSource: 'Fallback',
  indentCharacter: '  ',
  folderStrategy: 'Tags',
  includeAuthInfoInExample: true,
  parametersResolution: 'Example',
};

Converter.convert(
  { type: 'string', data: openapiSpec },
  options,
  (err, result) => {
    if (err) {
      console.error('Conversion failed:', err);
      process.exit(1);
    }

    if (!result.result) {
      console.error('Conversion failed:', result.reason);
      process.exit(1);
    }

    const collection = result.output[0].data;

    // 環境変数の追加
    collection.variable = [
      { key: 'baseUrl', value: 'https://sandbox.api.example.com/v2' },
      { key: 'apiKey', value: 'sk_test_your_key_here' },
    ];

    writeFileSync(
      './docs/example-api.postman_collection.json',
      JSON.stringify(collection, null, 2)
    );

    console.log('Postman collection generated successfully');
  }
);
```

---

## 9. エラードキュメンテーション

### 9.1 エラーレスポンスの設計と文書化

エラードキュメントは、開発者がトラブルシューティングを行う際の最重要リソースである。全てのエラーコードに対して、原因と対処法を明確に記載する。

```yaml
# components/schemas/ErrorResponse.yaml
type: object
title: ErrorResponse
description: |
  全ての API エラーは統一された形式で返されます。
  `error.code` でエラーの種類を判別できます。
required:
  - error
properties:
  error:
    type: object
    required:
      - code
      - message
    properties:
      code:
        type: string
        description: |
          機械可読なエラーコード。
          アプリケーション内での分岐処理に使用してください。
        enum:
          - INVALID_PARAMETER
          - VALIDATION_ERROR
          - AUTHENTICATION_REQUIRED
          - INSUFFICIENT_PERMISSIONS
          - RESOURCE_NOT_FOUND
          - DUPLICATE_RESOURCE
          - RATE_LIMIT_EXCEEDED
          - INTERNAL_ERROR
          - SERVICE_UNAVAILABLE
      message:
        type: string
        description: 人間可読なエラーメッセージ（日本語または英語）
      details:
        type: array
        description: エラーの詳細情報（バリデーションエラー時に使用）
        items:
          type: object
          properties:
            field:
              type: string
              description: エラーが発生したフィールド名
            code:
              type: string
              description: フィールド固有のエラーコード
            message:
              type: string
              description: フィールド固有のエラーメッセージ
      requestId:
        type: string
        description: |
          リクエスト追跡用の一意識別子。
          サポートへの問い合わせ時にこの ID を共有してください。
        example: "req_a1b2c3d4e5f6"
```

### 9.2 エラーコード一覧と対処法

```
エラーコードリファレンス:

  ┌────────────────────────────┬──────┬────────────────────────────┐
  │ コード                      │ HTTP │ 対処法                      │
  ├────────────────────────────┼──────┼────────────────────────────┤
  │ INVALID_PARAMETER          │ 400  │ パラメータの値・型を確認     │
  │ VALIDATION_ERROR           │ 422  │ details[] で各フィールド確認 │
  │ AUTHENTICATION_REQUIRED    │ 401  │ Authorization ヘッダーを確認 │
  │ INSUFFICIENT_PERMISSIONS   │ 403  │ API キーのスコープを確認     │
  │ RESOURCE_NOT_FOUND         │ 404  │ リソース ID の存在を確認     │
  │ DUPLICATE_RESOURCE         │ 409  │ 一意制約に違反するフィールド確認│
  │ RATE_LIMIT_EXCEEDED        │ 429  │ Retry-After ヘッダーに従う   │
  │ INTERNAL_ERROR             │ 500  │ リトライ、サポートに連絡      │
  │ SERVICE_UNAVAILABLE        │ 503  │ ステータスページを確認        │
  └────────────────────────────┴──────┴────────────────────────────┘
```

### 9.3 エラーハンドリングのベストプラクティス

```typescript
// 包括的なエラーハンドリング実装例
import {
  ExampleClient,
  ExampleError,
  ValidationError,
  AuthenticationError,
  RateLimitError,
  NotFoundError,
  InternalError,
} from '@example/sdk';

const client = new ExampleClient({
  apiKey: process.env.EXAMPLE_API_KEY,
  maxRetries: 3,
  // リトライ可能なエラーの自動リトライ設定
  retryOn: [429, 500, 503],
  retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 30000),
});

async function robustApiCall<T>(
  operation: () => Promise<T>,
  context: string
): Promise<T> {
  try {
    return await operation();
  } catch (error) {
    if (error instanceof ValidationError) {
      // 422: 入力値の修正が必要
      console.error(`[${context}] Validation errors:`);
      for (const detail of error.errors) {
        console.error(`  - ${detail.field}: ${detail.message}`);
      }
      throw error; // リトライ不要

    } else if (error instanceof AuthenticationError) {
      // 401: API キーの確認が必要
      console.error(`[${context}] Authentication failed. Check your API key.`);
      throw error; // リトライ不要

    } else if (error instanceof RateLimitError) {
      // 429: SDK の自動リトライに任せる（maxRetries 超過時のみここに到達）
      console.error(
        `[${context}] Rate limit exceeded after ${client.maxRetries} retries. ` +
        `Retry after ${error.retryAfter}s`
      );
      throw error;

    } else if (error instanceof NotFoundError) {
      // 404: リソースが存在しない
      console.warn(`[${context}] Resource not found: ${error.message}`);
      return null as T; // アプリケーション要件に応じて null を返す

    } else if (error instanceof InternalError) {
      // 500: サーバー側の問題（SDK 自動リトライ超過時）
      console.error(
        `[${context}] Internal server error (requestId: ${error.requestId}). ` +
        `Please contact support with this request ID.`
      );
      throw error;

    } else if (error instanceof ExampleError) {
      // その他の API エラー
      console.error(
        `[${context}] API Error [${error.code}]: ${error.message} ` +
        `(requestId: ${error.requestId})`
      );
      throw error;

    } else {
      // ネットワークエラー等の非 API エラー
      console.error(`[${context}] Unexpected error:`, error);
      throw error;
    }
  }
}

// 使用例
const user = await robustApiCall(
  () => client.users.create({
    name: 'Taro Yamada',
    email: 'taro@example.com',
  }),
  'createUser'
);
```

---

## 10. ドキュメントのテスト自動化

### 10.1 コード例の自動テスト

ドキュメント内のコード例が実際に動作することを保証するために、自動テストを導入する。

```typescript
// tests/docs-examples.test.ts
// ドキュメント内コード例の動作確認テスト
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { ExampleClient, ValidationError } from '@example/sdk';

const client = new ExampleClient({
  apiKey: process.env.EXAMPLE_TEST_API_KEY,
  baseURL: 'https://sandbox.api.example.com/v2',
});

describe('Quick Start ガイドのコード例', () => {
  let createdUserId: string;

  it('ステップ 3: ユーザーの作成', async () => {
    // ドキュメントのコード例と同一
    const user = await client.users.create({
      name: 'Taro Yamada',
      email: `test-${Date.now()}@example.com`,
    });

    expect(user.id).toBeDefined();
    expect(user.name).toBe('Taro Yamada');
    createdUserId = user.id;
  });

  it('ステップ 3: ユーザーの取得', async () => {
    const fetched = await client.users.get(createdUserId);
    expect(fetched.name).toBe('Taro Yamada');
  });

  it('ステップ 3: ユーザー一覧の取得', async () => {
    const { data: users } = await client.users.list({ limit: 10 });
    expect(Array.isArray(users)).toBe(true);
    expect(users.length).toBeLessThanOrEqual(10);
  });

  it('ステップ 4: バリデーションエラー', async () => {
    try {
      await client.users.create({ name: '', email: 'invalid' });
      expect.unreachable('Should have thrown');
    } catch (error) {
      expect(error).toBeInstanceOf(ValidationError);
      expect((error as ValidationError).errors.length).toBeGreaterThan(0);
    }
  });

  afterAll(async () => {
    // テストデータのクリーンアップ
    if (createdUserId) {
      await client.users.delete(createdUserId);
    }
  });
});

describe('エラーハンドリングガイドのコード例', () => {
  it('認証エラー', async () => {
    const badClient = new ExampleClient({ apiKey: 'invalid_key' });
    try {
      await badClient.users.list();
      expect.unreachable('Should have thrown');
    } catch (error) {
      expect(error).toBeInstanceOf(Error);
    }
  });

  it('404 エラー', async () => {
    try {
      await client.users.get('nonexistent_id');
      expect.unreachable('Should have thrown');
    } catch (error) {
      expect(error).toHaveProperty('code', 'RESOURCE_NOT_FOUND');
    }
  });
});
```

### 10.2 OpenAPI 仕様の検証自動化

```yaml
# .github/workflows/api-docs-ci.yml
# ドキュメント品質の CI チェック
name: API Documentation CI
on:
  pull_request:
    paths:
      - 'api/**'
      - 'docs/**'

jobs:
  lint-openapi:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Lint OpenAPI spec
        run: |
          npx @redocly/cli lint api/openapi.yaml \
            --config api/.redocly.yaml

      - name: Check for breaking changes
        if: github.event_name == 'pull_request'
        run: |
          npx oasdiff breaking \
            --base <(git show origin/main:api/openapi.yaml) \
            --revision api/openapi.yaml \
            --fail-on ERR

  test-examples:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20

      - name: Install dependencies
        run: npm ci

      - name: Run documentation example tests
        env:
          EXAMPLE_TEST_API_KEY: ${{ secrets.SANDBOX_API_KEY }}
        run: npx vitest run tests/docs-examples.test.ts

  validate-links:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check documentation links
        run: |
          npx markdown-link-check docs/**/*.md \
            --config .markdown-link-check.json

  spell-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Spell check documentation
        run: |
          npx cspell "docs/**/*.md" --config .cspell.json
```

### 10.3 ドキュメントカバレッジの計測

```typescript
// scripts/check-doc-coverage.ts
// OpenAPI 仕様のドキュメントカバレッジを計測
import { readFileSync } from 'fs';
import yaml from 'js-yaml';

interface CoverageReport {
  total: number;
  documented: number;
  missing: string[];
  coverage: number;
}

function checkCoverage(spec: any): Record<string, CoverageReport> {
  const reports: Record<string, CoverageReport> = {};

  // エンドポイント記述のカバレッジ
  const endpoints: CoverageReport = {
    total: 0, documented: 0, missing: [], coverage: 0
  };

  for (const [path, methods] of Object.entries(spec.paths || {})) {
    for (const [method, operation] of Object.entries(methods as any)) {
      if (['get', 'post', 'put', 'patch', 'delete'].includes(method)) {
        endpoints.total++;
        const op = operation as any;
        if (op.description && op.description.length > 20) {
          endpoints.documented++;
        } else {
          endpoints.missing.push(`${method.toUpperCase()} ${path}`);
        }
      }
    }
  }
  endpoints.coverage = Math.round(
    (endpoints.documented / endpoints.total) * 100
  );
  reports['endpoints'] = endpoints;

  // パラメータ記述のカバレッジ
  const params: CoverageReport = {
    total: 0, documented: 0, missing: [], coverage: 0
  };

  for (const [path, methods] of Object.entries(spec.paths || {})) {
    for (const [method, operation] of Object.entries(methods as any)) {
      const op = operation as any;
      for (const param of op.parameters || []) {
        params.total++;
        if (param.description) {
          params.documented++;
        } else {
          params.missing.push(
            `${method.toUpperCase()} ${path} -> ${param.name}`
          );
        }
      }
    }
  }
  params.coverage = Math.round(
    (params.documented / params.total) * 100
  );
  reports['parameters'] = params;

  // レスポンス例のカバレッジ
  const examples: CoverageReport = {
    total: 0, documented: 0, missing: [], coverage: 0
  };

  for (const [path, methods] of Object.entries(spec.paths || {})) {
    for (const [method, operation] of Object.entries(methods as any)) {
      const op = operation as any;
      for (const [code, response] of Object.entries(op.responses || {})) {
        examples.total++;
        const resp = response as any;
        const hasExample = resp.content?.['application/json']?.example
          || resp.content?.['application/json']?.examples;
        if (hasExample) {
          examples.documented++;
        } else {
          examples.missing.push(
            `${method.toUpperCase()} ${path} -> ${code}`
          );
        }
      }
    }
  }
  examples.coverage = Math.round(
    (examples.documented / examples.total) * 100
  );
  reports['examples'] = examples;

  return reports;
}

// 実行
const spec = yaml.load(readFileSync('./api/openapi.yaml', 'utf-8'));
const reports = checkCoverage(spec);

console.log('=== Documentation Coverage Report ===\n');
for (const [category, report] of Object.entries(reports)) {
  const status = report.coverage >= 90 ? 'PASS' : 'WARN';
  console.log(`[${status}] ${category}: ${report.coverage}% ` +
    `(${report.documented}/${report.total})`);
  if (report.missing.length > 0) {
    console.log(`  Missing:`);
    report.missing.forEach(m => console.log(`    - ${m}`));
  }
}
```

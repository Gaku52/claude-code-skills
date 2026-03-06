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

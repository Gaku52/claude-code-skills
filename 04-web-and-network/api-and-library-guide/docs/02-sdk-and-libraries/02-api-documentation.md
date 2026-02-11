# APIドキュメンテーション

> APIドキュメントはAPIの「顔」であり、開発者が最初に触れるインターフェース。OpenAPI/Swagger、自動生成ツール、インタラクティブドキュメント、コード例の設計まで、開発者に愛されるドキュメント作成の全技法を習得する。

## この章で学ぶこと

- [ ] OpenAPI仕様の活用とドキュメント自動生成を理解する
- [ ] インタラクティブドキュメントの構築方法を把握する
- [ ] 良いドキュメントの構成要素と設計原則を学ぶ

---

## 1. ドキュメントの種類

```
APIドキュメントの4層:

  ① リファレンス（Reference）:
     → エンドポイント一覧、パラメータ、レスポンス
     → OpenAPI → Swagger UI / Redoc / Scalar で自動生成
     → 例: Stripe API Reference

  ② ガイド（Guide）:
     → 「○○をするには」の手順書
     → Quick Start, Authentication, Pagination 等
     → 例: Stripe Docs

  ③ チュートリアル（Tutorial）:
     → ステップバイステップの実装例
     → 「決済システムを作ろう」等
     → 例: Twilio Quest

  ④ コンセプト（Concept）:
     → アーキテクチャ、設計思想の説明
     → Webhooks の仕組み、レート制限の考え方等
     → 例: Stripe のアーキテクチャ解説

良いドキュメント = 4層全てが揃っている
```

---

## 2. OpenAPI からの自動生成

```yaml
# openapi.yaml の充実化（ドキュメント品質向上）
paths:
  /users:
    get:
      summary: ユーザー一覧の取得
      description: |
        登録済みユーザーの一覧を取得します。
        Cursor ベースのページネーションに対応しています。

        ### 権限
        - `users:read` スコープが必要

        ### レート制限
        - 100リクエスト/分
      operationId: listUsers
      tags: [Users]
      parameters:
        - name: cursor
          in: query
          description: ページネーションカーソル。前回のレスポンスの `nextCursor` を指定。
          schema:
            type: string
          example: "eyJpZCI6MTAwfQ"
        - name: limit
          in: query
          description: 1ページあたりの取得件数（1-100）
          schema:
            type: integer
            minimum: 1
            maximum: 100
            default: 20
      responses:
        '200':
          description: ユーザー一覧の取得に成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserListResponse'
              examples:
                default:
                  summary: 基本的なレスポンス
                  value:
                    data:
                      - id: "user_123"
                        name: "Taro Yamada"
                        email: "taro@example.com"
                        role: "user"
                    meta:
                      hasNextPage: true
                      nextCursor: "eyJpZCI6MTIwfQ"
                empty:
                  summary: 空の結果
                  value:
                    data: []
                    meta:
                      hasNextPage: false
                      nextCursor: null
```

```
ドキュメント生成ツール:

  Swagger UI:
  → 最も一般的
  → インタラクティブ（Try it out 機能）
  → カスタマイズ可能

  Redoc:
  → 3カラムレイアウト（概要・パラメータ・レスポンス）
  → 美しいデザイン
  → SEO対応

  Scalar:
  → モダンなUI
  → ダークモード対応
  → コード例の自動生成

  Stoplight Elements:
  → Reactコンポーネント
  → カスタマイズ性が高い
  → 自社サイトに埋め込み可能
```

---

## 3. コード例の設計

```
コード例の原則:
  ✓ コピー&ペーストで動くこと
  ✓ 複数言語対応（curl, JS, Python, Ruby, Go）
  ✓ エラーハンドリングを含むこと
  ✓ 実際的な値を使うこと（foo/bar ではなく具体的な値）

良い例:
  curl -X POST https://api.example.com/v1/users \
    -H "Authorization: Bearer sk_test_abc123" \
    -H "Content-Type: application/json" \
    -d '{
      "name": "Taro Yamada",
      "email": "taro@example.com"
    }'

  // JavaScript (SDK)
  const client = new ExampleClient({ apiKey: 'sk_test_abc123' });
  const user = await client.users.create({
    name: 'Taro Yamada',
    email: 'taro@example.com',
  });
  console.log(user.id); // "user_456"

  # Python (SDK)
  client = ExampleClient(api_key="sk_test_abc123")
  user = client.users.create(
      name="Taro Yamada",
      email="taro@example.com",
  )
  print(user.id)  # "user_456"

悪い例:
  // 動かない、エラーハンドリングなし、抽象的な値
  fetch('/api/users', { body: { name: 'foo' } })
```

---

## 4. Quick Start ガイド

```markdown
## Quick Start

### 1. インストール

npm install @example/sdk
# or
yarn add @example/sdk

### 2. クライアントの初期化

import { ExampleClient } from '@example/sdk';

const client = new ExampleClient({
  apiKey: process.env.EXAMPLE_API_KEY,
});

### 3. 最初のAPI呼び出し

// ユーザーの作成
const user = await client.users.create({
  name: 'Taro Yamada',
  email: 'taro@example.com',
});
console.log('Created user:', user.id);

// ユーザーの取得
const fetched = await client.users.get(user.id);
console.log('User name:', fetched.name);

### 4. エラーハンドリング

import { ExampleError, ValidationError } from '@example/sdk';

try {
  await client.users.create({ name: '', email: 'invalid' });
} catch (error) {
  if (error instanceof ValidationError) {
    console.error('Validation errors:', error.errors);
  }
}

→ ここまで5分以内で完了
```

---

## 5. Changelog と Migration Guide

```markdown
## Changelog

### v2.0.0 (2024-07-01) — Breaking Changes

#### 破壊的変更
- `client.getUser(id)` → `client.users.get(id)` に変更
- エラーの型を `ApiError` → `ExampleError` にリネーム
- Node.js 16 のサポートを終了

#### 移行方法
// Before (v1.x)
const user = await client.getUser('123');

// After (v2.x)
const user = await client.users.get('123');

#### 新機能
- `client.users.listAll()` で自動ページネーション
- リトライ設定のカスタマイズ

#### バグ修正
- タイムアウト時のメモリリークを修正
```

---

## 6. ドキュメント品質チェックリスト

```
必須要素:
  □ Quick Start（5分以内に最初のAPIコール）
  □ 認証方法の説明
  □ 全エンドポイントのリファレンス
  □ リクエスト/レスポンスの例
  □ エラーコードの一覧と対処法
  □ レート制限の説明
  □ SDKのインストールと初期化
  □ ページネーションの使い方
  □ Webhook の設定方法（該当する場合）
  □ Changelog

品質基準:
  □ コード例がコピー&ペーストで動く
  □ 全てのパラメータに説明がある
  □ 成功/エラーの両方のレスポンス例がある
  □ 複数言語のコード例（最低 curl + 1言語）
  □ 検索機能がある
  □ モバイル対応
  □ ダークモード対応
  □ 定期的に更新されている
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 4層構造 | リファレンス、ガイド、チュートリアル、コンセプト |
| 自動生成 | OpenAPI → Redoc / Scalar / Swagger UI |
| コード例 | コピペで動く、複数言語、具体的な値 |
| Quick Start | 5分で最初のAPI呼び出し |
| Changelog | 破壊的変更 + 移行方法を必ず記載 |

---

## 次に読むべきガイド
→ [[00-authentication-patterns.md]] — 認証パターン

---

## 参考文献
1. Stripe. "Stripe Documentation." stripe.com/docs, 2024.
2. Redocly. "Redoc." github.com/Redocly/redoc, 2024.
3. ReadMe. "What Makes Great API Documentation." readme.com, 2024.

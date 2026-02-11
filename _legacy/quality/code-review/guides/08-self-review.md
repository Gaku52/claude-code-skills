# セルフレビューガイド

## 概要

セルフレビューは、PR作成前に自分のコードを客観的にレビューするプロセスです。他人にレビューを依頼する前に問題を発見し、修正することで、レビュー効率を大幅に向上させます。

## 目次

1. [セルフレビューの重要性](#セルフレビューの重要性)
2. [PR作成前チェックリスト](#pr作成前チェックリスト)
3. [diffレビュー](#diffレビュー)
4. [自動化ツール活用](#自動化ツール活用)
5. [コミットメッセージ](#コミットメッセージ)
6. [PR説明文](#pr説明文)

---

## セルフレビューの重要性

### なぜセルフレビューが必要か

1. **レビュー時間の短縮**: 自分で見つけられる問題を事前に修正
2. **品質向上**: 細かいミスを減らす
3. **学習機会**: 自分のコードを客観的に見る習慣
4. **レビュワーの負担軽減**: より本質的なフィードバックに集中できる

### セルフレビューのタイミング

```bash
# ❌ Bad: すぐにPR作成
git add .
git commit -m "Add feature"
git push origin feature/new-feature
# GitHub でPR作成

# ✅ Good: セルフレビュー後にPR作成
git add .
git commit -m "Add user authentication feature"

# セルフレビュー
git diff main...HEAD
# コードを確認、問題を修正

# 自動チェック
npm run lint
npm run test
npm run build

# 問題なければPush
git push origin feature/new-feature
```

---

## PR作成前チェックリスト

### 基本チェック

```markdown
## PR作成前セルフレビューチェックリスト

### コード品質
- [ ] 変更内容を git diff で確認した
- [ ] デバッグコード・console.log を削除した
- [ ] コメントアウトされたコードを削除した
- [ ] TODOコメントを確認した
- [ ] 命名が適切か確認した
- [ ] マジックナンバーを定数化した

### 機能性
- [ ] 要件を満たしている
- [ ] エッジケースを考慮した
- [ ] エラーハンドリングが適切
- [ ] 境界値をテストした

### テスト
- [ ] 新しいテストを追加した
- [ ] 既存のテストが通る
- [ ] カバレッジが維持・向上している
- [ ] E2Eテストを実行した（必要に応じて）

### パフォーマンス
- [ ] N+1問題がない
- [ ] 不要なループがない
- [ ] メモリリークがない
- [ ] 大量データを考慮した

### セキュリティ
- [ ] 入力値を検証している
- [ ] 認証・認可が適切
- [ ] 機密情報がログに出ない
- [ ] SQLインジェクション対策済み

### ドキュメント
- [ ] コードコメントを追加した
- [ ] APIドキュメントを更新した
- [ ] READMEを更新した（必要に応じて）
- [ ] マイグレーションガイドを作成した（破壊的変更の場合）

### ビルド・デプロイ
- [ ] ビルドが通る
- [ ] Linterが通る
- [ ] 型チェックが通る
- [ ] CI/CDが通る

### Git
- [ ] コミットメッセージが適切
- [ ] コミットが論理的に分割されている
- [ ] ブランチ名が規約に従っている
```

---

## diffレビュー

### git diff での確認方法

```bash
# ✅ 変更全体を確認
git diff main...HEAD

# ✅ ファイルごとに確認
git diff main...HEAD src/services/user.service.ts

# ✅ 統計情報を確認
git diff --stat main...HEAD

# ✅ 単語単位のdiff
git diff --word-diff main...HEAD

# ✅ GUIツールを使用
git difftool main...HEAD

# ✅ GitHub CLI
gh pr diff
```

### 見るべきポイント

```typescript
// ❌ Bad: デバッグコードが残っている
function processUser(user: User): void {
  console.log('DEBUG: user =', user);  // ❌ 削除し忘れ
  debugger;  // ❌ 削除し忘れ

  // 処理
  updateUserStatus(user);

  console.log('DEBUG: done');  // ❌ 削除し忘れ
}

// ❌ Bad: コメントアウトされたコード
function calculatePrice(item: Item): number {
  // const oldPrice = item.basePrice * 1.1;
  // return oldPrice;

  return item.basePrice * 1.08;
}

// ❌ Bad: 未使用のインポート
import { useState, useEffect, useCallback } from 'react';  // useCallbackは未使用
import { format } from 'date-fns';  // 未使用
import { debounce } from 'lodash';  // 未使用

// ✅ Good: クリーンなコード
function processUser(user: User): void {
  updateUserStatus(user);
}

function calculatePrice(item: Item): number {
  return item.basePrice * 1.08;
}

import { useState, useEffect } from 'react';
```

### diff確認の自動化

```typescript
// .git/hooks/pre-commit
#!/bin/bash

# デバッグコードのチェック
if git diff --cached | grep -E "console\.(log|debug|info)" > /dev/null; then
    echo "Error: console.log found in staged files"
    echo "Please remove debug code before committing"
    exit 1
fi

# debugger文のチェック
if git diff --cached | grep -E "debugger" > /dev/null; then
    echo "Error: debugger statement found"
    exit 1
fi

# TODOコメントのチェック
if git diff --cached | grep -E "TODO|FIXME|XXX" > /dev/null; then
    echo "Warning: TODO/FIXME comments found"
    echo "Please review and create issues if needed"
fi

echo "Pre-commit checks passed ✓"
```

---

## 自動化ツール活用

### ESLint

```javascript
// .eslintrc.js
module.exports = {
  rules: {
    // デバッグコード禁止
    'no-console': 'error',
    'no-debugger': 'error',

    // コメントアウトされたコード禁止
    'no-commented-out-code': 'error',

    // 未使用変数禁止
    'no-unused-vars': 'error',

    // 複雑度制限
    'complexity': ['error', 10],

    // 関数の行数制限
    'max-lines-per-function': ['error', 50],
  },
};

// セルフレビュー
npm run lint
npm run lint:fix  // 自動修正
```

### Prettier

```javascript
// .prettierrc
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "es5"
}

// セルフレビュー
npm run format
npm run format:check
```

### TypeScript

```bash
# 型チェック
tsc --noEmit

# strict モード推奨
# tsconfig.json
{
  "compilerOptions": {
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true
  }
}
```

### テストカバレッジ

```bash
# カバレッジ確認
npm run test:coverage

# カバレッジが下がっていないか確認
# package.json
{
  "jest": {
    "coverageThreshold": {
      "global": {
        "branches": 80,
        "functions": 80,
        "lines": 80,
        "statements": 80
      }
    }
  }
}
```

### Git hooks（Husky）

```bash
# Husky設定
npm install -D husky

# pre-commit hook
# .husky/pre-commit
#!/bin/sh
. "$(dirname "$0")/_/husky.sh"

# Lint
npm run lint

# Test（変更されたファイルのみ）
npm run test:changed

# 型チェック
npm run type-check

echo "✓ All pre-commit checks passed"
```

---

## コミットメッセージ

### Conventional Commits

```bash
# ✅ Good: Conventional Commits形式
git commit -m "feat: add user authentication"
git commit -m "fix: resolve memory leak in image loader"
git commit -m "docs: update API documentation"
git commit -m "refactor: extract user validation logic"
git commit -m "test: add tests for payment processing"

# ✅ より詳細な説明
git commit -m "feat: add user authentication

- Implement JWT-based authentication
- Add login/logout endpoints
- Add password hashing with bcrypt
- Add authentication middleware

Closes #123"

# ❌ Bad: 不明確なメッセージ
git commit -m "fix bug"
git commit -m "update"
git commit -m "wip"
git commit -m "asdfasdf"
```

### 形式

```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Type
- `feat`: 新機能
- `fix`: バグ修正
- `docs`: ドキュメント
- `style`: フォーマット
- `refactor`: リファクタリング
- `test`: テスト
- `chore`: ビルド・ツール

#### 例

```bash
# Feature
git commit -m "feat(auth): add two-factor authentication

Implement TOTP-based 2FA using speakeasy library.
Users can enable 2FA in their profile settings.

Closes #456"

# Bug fix
git commit -m "fix(api): prevent SQL injection in user search

Use parameterized queries instead of string concatenation.

Security issue reported in #789"

# Breaking change
git commit -m "feat(api): change user endpoint response format

BREAKING CHANGE: User API now returns camelCase instead of snake_case

Migration guide:
- Update client code to use camelCase field names
- Use the provided migration script: npm run migrate:user-api"
```

---

## PR説明文

### テンプレート

```markdown
## 概要
この変更は何をするものですか？なぜ必要ですか？

## 変更内容
- [ ] 新機能A
- [ ] バグ修正B
- [ ] リファクタリングC

## スクリーンショット
（UIの変更がある場合）

### Before
![before](before.png)

### After
![after](after.png)

## テスト方法
1. ローカル環境を起動
2. `/users`にアクセス
3. 新規ユーザーを作成
4. ログインできることを確認

## 影響範囲
- `UserService`: ユーザー作成ロジック追加
- `UserController`: 新しいエンドポイント追加
- `user.test.ts`: テストケース追加

## Breaking Changes
- [ ] なし
- [ ] あり（詳細は下記）

## マイグレーション
（破壊的変更がある場合の移行手順）

## チェックリスト
- [x] テストを追加した
- [x] ドキュメントを更新した
- [x] Lintが通る
- [x] ビルドが通る
- [x] 既存のテストが通る

## 関連Issue
Closes #123
Related to #456

## レビュワーへのメモ
- `UserService.ts`の234行目のロジックについて意見をください
- パフォーマンスへの影響を確認してください
```

### 良いPR説明の例

```markdown
## 概要
ユーザー認証機能を追加します。JWT ベースの認証を実装し、ログイン・ログアウトをサポートします。

## 背景
Issue #123 で報告された通り、現在のシステムには認証機能がなく、すべてのエンドポイントが公開されています。これはセキュリティ上の大きな問題です。

## 変更内容
- [x] JWT 生成・検証機能
- [x] ログイン API (`POST /api/auth/login`)
- [x] ログアウト API (`POST /api/auth/logout`)
- [x] 認証ミドルウェア
- [x] パスワードハッシュ化（bcrypt）
- [x] リフレッシュトークン機能

## 技術的詳細

### 使用ライブラリ
- `jsonwebtoken`: JWT 生成・検証
- `bcrypt`: パスワードハッシュ化

### アーキテクチャ
\`\`\`
Client
  ↓ (email, password)
AuthController
  ↓
AuthService (パスワード検証、JWT生成)
  ↓
UserRepository (ユーザー取得)
  ↓
Database
\`\`\`

### セキュリティ考慮事項
- パスワードはbcryptでハッシュ化（コスト: 10）
- JWTは環境変数から読み込んだ秘密鍵で署名
- アクセストークンの有効期限: 15分
- リフレッシュトークンの有効期限: 7日

## テスト方法

### 1. ユーザー登録
\`\`\`bash
curl -X POST http://localhost:3000/api/users \\
  -H "Content-Type: application/json" \\
  -d '{"email":"test@example.com","password":"SecurePass123"}'
\`\`\`

### 2. ログイン
\`\`\`bash
curl -X POST http://localhost:3000/api/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{"email":"test@example.com","password":"SecurePass123"}'
\`\`\`

レスポンス:
\`\`\`json
{
  "accessToken": "eyJhbGc...",
  "refreshToken": "eyJhbGc...",
  "user": {
    "id": "123",
    "email": "test@example.com"
  }
}
\`\`\`

### 3. 認証が必要なエンドポイントにアクセス
\`\`\`bash
curl -X GET http://localhost:3000/api/users/me \\
  -H "Authorization: Bearer eyJhbGc..."
\`\`\`

## パフォーマンス
- bcryptのハッシュ化: ~100ms
- JWT生成: <1ms
- 認証ミドルウェア: <1ms

## 影響範囲
- 新規ファイル:
  - `src/auth/auth.service.ts`
  - `src/auth/auth.controller.ts`
  - `src/auth/auth.middleware.ts`
  - `src/auth/auth.dto.ts`
- 変更ファイル:
  - `src/app.ts`: ルート追加
  - `src/users/user.entity.ts`: `passwordHash`フィールド追加

## Breaking Changes
なし。既存のAPIはすべて動作します。認証が必要なエンドポイントは次のPRで追加予定です。

## 今後の予定
- [ ] 2要素認証 (Issue #124)
- [ ] パスワードリセット (Issue #125)
- [ ] ソーシャルログイン (Issue #126)

## チェックリスト
- [x] ユニットテスト追加（カバレッジ: 95%）
- [x] 統合テスト追加
- [x] APIドキュメント更新
- [x] セキュリティレビュー実施
- [x] パフォーマンステスト実施

## レビュワーへのメモ
特に以下の点について確認をお願いします：
1. JWT秘密鍵の管理方法は適切か？
2. トークンの有効期限（15分/7日）は適切か？
3. bcryptのコスト（10）は適切か？

Closes #123
```

---

## セルフレビューのコツ

### 時間を置く

```
❌ コード書いてすぐレビュー → ミスを見逃しやすい
✅ 翌日レビュー → 客観的に見られる

# コーディング
git commit -m "feat: add user feature"

# 一晩寝かせる

# 翌日セルフレビュー
git diff main...HEAD
# 問題を発見・修正
git commit --amend
```

### 他人の視点で見る

```
自分: このコードわかりやすい！
↓
3ヶ月後の自分: これ何してるんだっけ？
↓
他の開発者: ??? 意味不明

→ コメント追加、リファクタリング
```

### 声に出して説明

```
1. コードを声に出して読む
2. 何をしているか説明する
3. 説明できない部分を改善

「この部分は...えーっと...まず...」
→ 複雑すぎる → リファクタリング
```

---

## まとめ

セルフレビューは最も効果的な品質向上手段です。

### 重要ポイント

1. **チェックリストを使う**
2. **diff を必ず確認**
3. **自動化ツールを活用**
4. **時間を置いて見直す**
5. **他人の視点で考える**

### 次のステップ

- [レビュー実施ガイド](09-reviewing.md)
- [フィードバック対応](10-feedback-response.md)
- [自動化ガイド](12-automation.md)

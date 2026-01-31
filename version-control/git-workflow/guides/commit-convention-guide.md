# コミット規約完全ガイド - Conventional Commits実践

> **最終更新:** 2026-01-02
> **対象読者:** 全開発者
> **推定読了時間:** 50分

## 📋 目次

1. [なぜコミット規約が必要か](#なぜコミット規約が必要か)
2. [Conventional Commits仕様](#conventional-commits仕様)
3. [Type詳細解説](#type詳細解説)
4. [Scope設計パターン](#scope設計パターン)
5. [Subject（件名）の書き方](#subjectの書き方)
6. [Body（本文）の書き方](#bodyの書き方)
7. [Footer（フッター）の書き方](#footerの書き方)
8. [Breaking Changes管理](#breaking-changes管理)
9. [プロジェクト別実践例](#プロジェクト別実践例)
10. [自動化ツール活用](#自動化ツール活用)
11. [CHANGELOG自動生成](#changelog自動生成)
12. [Semantic Versioning連携](#semantic-versioning連携)
13. [チーム運用ベストプラクティス](#チーム運用ベストプラクティス)
14. [コミット粒度の最適化](#コミット粒度の最適化)
15. [トラブルシューティング](#トラブルシューティング)

---

## なぜコミット規約が必要か

### 問題: 雑なコミットメッセージ

**悪い例:**
```bash
git commit -m "fix"
git commit -m "update"
git commit -m "wip"
git commit -m "aaa"
git commit -m "refactoring and some fixes"
```

**影響:**
```
❌ 何を変更したか不明
❌ なぜ変更したか不明
❌ 変更の影響範囲が不明
❌ バグ調査が困難
❌ CHANGELOG生成不可
❌ リリースノート作成が大変
❌ コードレビューが難しい
```

### 解決: Conventional Commits

**良い例:**
```bash
git commit -m "feat(auth): add biometric authentication support

Implemented Face ID and Touch ID for iOS login.
Users can now enable biometric auth in settings.

- Added BiometricAuthManager
- Updated LoginViewController
- Added unit tests

Closes #123"
```

**メリット:**
```
✅ 一目で変更内容がわかる
✅ 変更理由が明確
✅ 影響範囲が明示
✅ CHANGELOG自動生成
✅ Semantic Versioning自動化
✅ コードレビューが容易
✅ バグ調査が効率的
```

### ROI（投資対効果）

```
投資:
- コミットメッセージを丁寧に書く時間: +2分/commit

リターン:
- CHANGELOG作成時間: -30分/release
- バグ調査時間: -60分/bug
- コードレビュー時間: -10分/PR
- リリースノート作成: -90分/release

月間ROI（週1リリース、10bug、40PR想定）:
投資: 2分 × 200commits = 400分（6.7時間）
リターン:
  CHANGELOG: 30分 × 4 = 120分
  バグ調査: 60分 × 10 = 600分
  コードレビュー: 10分 × 40 = 400分
  リリースノート: 90分 × 4 = 360分
  合計: 1,480分（24.7時間）

→ 約4倍の時間節約
```

---

## Conventional Commits仕様

### 基本フォーマット

```
<type>(<scope>): <subject>

<body>

<footer>
```

### 各要素の役割

| 要素 | 必須 | 説明 | 例 |
|------|------|------|-----|
| **type** | ✅ 必須 | 変更の種類 | `feat`, `fix`, `docs` |
| **scope** | 任意 | 変更の影響範囲 | `auth`, `ui`, `api` |
| **subject** | ✅ 必須 | 変更の要約（50文字以内） | `add biometric login` |
| **body** | 任意 | 詳細な説明 | 理由・方法・影響 |
| **footer** | 任意 | Issue参照、Breaking Changes | `Closes #123` |

### 最小限の例

```bash
git commit -m "feat(auth): add Google OAuth login"
```

### 完全な例

```bash
git commit -m "feat(auth): add biometric authentication support

Implemented Face ID and Touch ID authentication for iOS devices.
Users can enable biometric login from the Settings screen.

Technical details:
- Used LocalAuthentication framework
- Added BiometricAuthManager service
- Updated LoginViewModel to support biometric flow
- Added fallback to password login

This improves UX by reducing login friction and enhances
security through device-level authentication.

Performance impact: Login time reduced by 60% (3s → 1.2s)

Closes #123
Refs #124, #125"
```

---

## Type詳細解説

### 主要Type

#### feat (Feature)

**用途:** 新機能の追加

**CHANGELOG:** `## Added` セクションに記載

**Semantic Versioning:** MINOR version up (1.2.0 → 1.3.0)

**例:**
```bash
feat(payment): add Apple Pay support
feat(ui): add dark mode toggle
feat(api): add user search endpoint
feat(auth): add two-factor authentication
```

**判断基準:**
```
✅ ユーザーが使える新機能
✅ 公開APIに新機能追加
✅ 新しいコンポーネント追加

❌ 内部リファクタリング（refactorを使用）
❌ パフォーマンス改善（perfを使用）
❌ バグ修正（fixを使用）
```

#### fix (Bug Fix)

**用途:** バグ修正

**CHANGELOG:** `## Fixed` セクションに記載

**Semantic Versioning:** PATCH version up (1.2.0 → 1.2.1)

**例:**
```bash
fix(login): resolve keyboard dismissal issue on iOS 17
fix(api): handle null response from server
fix(ui): correct alignment on iPad landscape
fix(payment): prevent duplicate charge on retry
```

**判断基準:**
```
✅ ユーザーに影響するバグ
✅ クラッシュ修正
✅ 誤動作の修正
✅ セキュリティパッチ

❌ タイポ修正（docsまたはstyleを使用）
❌ テスト修正（testを使用）
```

#### docs (Documentation)

**用途:** ドキュメントのみの変更

**CHANGELOG:** 通常記載しない

**Semantic Versioning:** version up不要

**例:**
```bash
docs(readme): update installation instructions
docs(api): add JSDoc comments to UserService
docs(contributing): add code review guidelines
docs(architecture): add system design diagram
```

**判断基準:**
```
✅ README更新
✅ コメント追加・修正
✅ APIドキュメント更新
✅ アーキテクチャ図追加

❌ コード変更を伴う場合（適切なtypeを使用）
```

#### style (Code Style)

**用途:** コードスタイルの変更（動作に影響しない）

**CHANGELOG:** 通常記載しない

**Semantic Versioning:** version up不要

**例:**
```bash
style(auth): fix indentation in LoginViewController
style: apply Prettier formatting
style(ui): reorder imports alphabetically
style: remove trailing whitespace
```

**判断基準:**
```
✅ インデント修正
✅ フォーマット適用
✅ セミコロン追加・削除
✅ import順序変更

❌ ロジック変更（適切なtypeを使用）
❌ リファクタリング（refactorを使用）
```

#### refactor (Refactoring)

**用途:** リファクタリング（機能・バグ修正を含まない）

**CHANGELOG:** 通常記載しない（大規模な場合は記載）

**Semantic Versioning:** version up不要（破壊的変更ある場合はMINOR/MAJOR）

**例:**
```bash
refactor(api): extract request builder logic
refactor(auth): simplify biometric check logic
refactor(ui): replace class components with hooks
refactor(database): migrate to prepared statements
```

**判断基準:**
```
✅ コード構造の改善
✅ 重複コード削除
✅ アーキテクチャ改善
✅ 技術的負債解消

❌ 新機能追加（featを使用）
❌ バグ修正（fixを使用）
```

#### perf (Performance)

**用途:** パフォーマンス改善

**CHANGELOG:** `## Performance` セクションに記載

**Semantic Versioning:** PATCH version up

**例:**
```bash
perf(images): implement lazy loading
perf(database): add indexes to user table
perf(ui): memoize expensive calculations
perf(api): reduce response payload size by 40%
```

**判断基準:**
```
✅ 速度改善
✅ メモリ使用量削減
✅ バッテリー消費削減
✅ ネットワーク使用量削減

測定結果を含めることを推奨:
perf(search): optimize query performance

Query time reduced from 200ms to 15ms
by adding database index on user_id.
```

#### test (Tests)

**用途:** テストの追加・修正

**CHANGELOG:** 通常記載しない

**Semantic Versioning:** version up不要

**例:**
```bash
test(auth): add biometric login tests
test(api): add integration tests for user endpoint
test(ui): add snapshot tests for ProfileView
test: increase coverage from 75% to 85%
```

**判断基準:**
```
✅ ユニットテスト追加
✅ 統合テスト追加
✅ E2Eテスト追加
✅ テストリファクタリング

❌ テスト修正 + 実装修正（featまたはfixを使用）
```

#### chore (Chores)

**用途:** ビルドプロセス、依存関係、設定変更

**CHANGELOG:** 通常記載しない

**Semantic Versioning:** version up不要

**例:**
```bash
chore(deps): update Alamofire to 5.8
chore(config): add SwiftLint configuration
chore(build): update Xcode build settings
chore: update .gitignore
```

**判断基準:**
```
✅ 依存関係更新
✅ ビルド設定変更
✅ .gitignore更新
✅ 開発ツール設定

❌ 機能変更（適切なtypeを使用）
```

#### ci (CI/CD)

**用途:** CI/CD設定の変更

**CHANGELOG:** 通常記載しない

**Semantic Versioning:** version up不要

**例:**
```bash
ci(github): add code coverage reporting
ci(fastlane): update TestFlight lane
ci: add automatic screenshot testing
ci(bitrise): optimize build cache
```

### Type選択フローチャート

```
質問1: ユーザーが使える新機能か？
├─ Yes → feat
└─ No → 質問2へ

質問2: バグを修正したか？
├─ Yes → fix
└─ No → 質問3へ

質問3: パフォーマンスを改善したか？
├─ Yes → perf
└─ No → 質問4へ

質問4: コード構造を改善したか（動作変更なし）？
├─ Yes → refactor
└─ No → 質問5へ

質問5: ドキュメントのみの変更か？
├─ Yes → docs
└─ No → 質問6へ

質問6: コードスタイルのみの変更か？
├─ Yes → style
└─ No → 質問7へ

質問7: テストの追加・修正か？
├─ Yes → test
└─ No → 質問8へ

質問8: CI/CD設定の変更か？
├─ Yes → ci
└─ No → chore
```

---

## Scope設計パターン

### パターン1: レイヤー別Scope

```
適用: レイヤードアーキテクチャ

auth      - 認証レイヤー
api       - APIレイヤー
database  - データベースレイヤー
ui        - UIレイヤー
model     - データモデル
service   - ビジネスロジック
utils     - ユーティリティ
config    - 設定
```

**例:**
```bash
feat(auth): add OAuth login
fix(api): handle timeout errors
refactor(database): optimize queries
perf(ui): memoize component rendering
```

### パターン2: 機能別Scope

```
適用: 機能ベースの組織化

login       - ログイン機能
profile     - プロフィール機能
settings    - 設定機能
dashboard   - ダッシュボード
payment     - 決済機能
notification - 通知機能
search      - 検索機能
```

**例:**
```bash
feat(login): add Google OAuth
fix(profile): correct avatar upload
docs(settings): add usage guide
```

### パターン3: モジュール別Scope（モノレポ）

```
適用: Monorepo

packages/ui       → ui
packages/api      → api
packages/shared   → shared
apps/web          → web
apps/mobile       → mobile
```

**例:**
```bash
feat(ui): add Button component
fix(api): resolve CORS issue
chore(shared): update types
```

### パターン4: ファイル・ディレクトリ別Scope

```
適用: 小規模プロジェクト

src/components    → components
src/pages         → pages
src/hooks         → hooks
src/utils         → utils
```

**例:**
```bash
feat(components): add LoadingSpinner
fix(pages): resolve routing issue
refactor(hooks): simplify useAuth
```

### Scopeの命名規則

```
✅ 小文字のみ
✅ 短く明確に（3-15文字推奨）
✅ ハイフン区切り（複数単語の場合）
✅ 一貫性を保つ

例:
feat(user-auth): ...      # Good
feat(UserAuth): ...       # Bad（大文字）
feat(authentication): ... # Bad（長すぎる→authに短縮）
```

### Scope設計テンプレート

```markdown
# SCOPES.md

プロジェクト: MyApp
アーキテクチャ: MVVM + Clean Architecture

## Scope一覧

### レイヤー別
- `auth` - 認証・認可
- `api` - APIクライアント
- `data` - データリポジトリ
- `domain` - ドメインロジック
- `presentation` - UI・ViewModel

### 機能別
- `login` - ログイン
- `signup` - 新規登録
- `profile` - プロフィール
- `settings` - 設定
- `dashboard` - ダッシュボード

### 共通
- `deps` - 依存関係
- `config` - 設定
- `ci` - CI/CD
- `docs` - ドキュメント
```

---

## Subjectの書き方

### ルール

```
1. 50文字以内（理想は40文字）
2. 小文字で始める
3. ピリオドで終わらない
4. 命令形（動詞の原形）を使う
5. 具体的に書く
```

### 動詞の選択

| 動詞 | 用途 | 例 |
|------|------|-----|
| **add** | 新規追加 | `add user authentication` |
| **implement** | 実装 | `implement payment flow` |
| **create** | 作成 | `create UserService class` |
| **introduce** | 導入 | `introduce caching layer` |
| **update** | 更新 | `update dependencies to latest` |
| **modify** | 変更 | `modify API response format` |
| **change** | 変更 | `change button color to blue` |
| **improve** | 改善 | `improve error messages` |
| **enhance** | 強化 | `enhance security validation` |
| **fix** | 修正 | `fix memory leak in cache` |
| **resolve** | 解決 | `resolve navigation bug` |
| **correct** | 訂正 | `correct typo in README` |
| **remove** | 削除 | `remove deprecated methods` |
| **delete** | 削除 | `delete unused files` |
| **drop** | 廃止 | `drop support for iOS 12` |
| **refactor** | リファクタリング | `refactor login logic` |
| **simplify** | 簡略化 | `simplify error handling` |
| **extract** | 抽出 | `extract common utilities` |
| **optimize** | 最適化 | `optimize database queries` |
| **reduce** | 削減 | `reduce bundle size` |
| **increase** | 増加 | `increase test coverage` |

### Good vs Bad

#### ✅ Good Examples

```bash
feat(auth): add biometric login support
fix(ui): resolve layout issue on iPad
docs(api): add JSDoc comments to UserService
refactor(network): simplify request builder
perf(images): implement lazy loading
test(auth): add unit tests for login flow
chore(deps): update Alamofire to 5.8
```

#### ❌ Bad Examples

```bash
feat(auth): Added biometric login support  # 過去形
fix(ui): Fix bug  # 具体性がない
docs(api): Update.  # ピリオド、具体性がない
refactor: refactoring  # 動詞の重複
perf: performance improvements  # 名詞形
test: Tests  # 具体性がない
chore: Update  # 何を更新したか不明
```

### 文字数の最適化

**長すぎる例:**
```bash
# 78文字（NG）
feat(auth): add support for biometric authentication using Face ID and Touch ID on iOS devices
```

**最適化:**
```bash
# 40文字（OK）
feat(auth): add biometric authentication

# Body で詳細を説明
Implemented Face ID and Touch ID support for iOS.
Users can enable biometric login from settings.
```

---

## Bodyの書き方

### いつBodyを書くべきか

```
✅ 複雑な変更の場合
✅ 理由説明が必要な場合
✅ 設計判断の背景がある場合
✅ 影響範囲が広い場合
✅ パフォーマンス改善の測定結果がある場合

❌ 自明な変更の場合
❌ Subjectで十分説明できる場合
```

### Body構成

```
1行目: Subjectから1行空ける

段落1: 何を変更したか（What）
段落2: なぜ変更したか（Why）
段落3: どのように実装したか（How）
段落4: 影響・効果（Impact）

- 箇条書きで詳細を列挙
- 72文字で改行
```

### テンプレート1: 新機能追加

```bash
feat(payment): add Apple Pay support

Integrated Apple Pay for faster checkout experience.
This addresses user feedback requesting alternative payment methods.

Implementation:
- Integrated PassKit framework
- Added ApplePayManager service
- Updated CheckoutViewModel to handle Apple Pay flow
- Added unit and integration tests

Impact:
- Checkout time reduced from 45s to 12s
- Payment success rate improved by 15%
- Supports all major credit cards

Closes #234
```

### テンプレート2: バグ修正

```bash
fix(login): resolve keyboard dismissal issue on iOS 17

The keyboard was not dismissing when tapping outside the text field
on iOS 17 devices. This caused poor UX and user complaints.

Root cause:
- iOS 17 changed default keyboard dismissal behavior
- Tap gesture recognizer was not properly configured

Solution:
- Added tap gesture recognizer to view
- Implemented keyboard dismissal on tap
- Added iOS version check for backward compatibility

Tested on:
- iOS 17.0 (iPhone 14 Pro)
- iOS 16.5 (iPhone 12)
- iOS 15.7 (iPhone X)

Fixes #567
```

### テンプレート3: リファクタリング

```bash
refactor(network): extract request building logic

The APIClient class had grown to 500+ lines, making it difficult
to test and maintain.

Changes:
- Extracted request building logic into RequestBuilder
- Created URLRequestBuilder implementing RequestBuilder protocol
- Moved URL construction logic to separate URLBuilder
- Updated all API calls to use new builder pattern

Benefits:
- Improved testability (can mock RequestBuilder)
- Better separation of concerns
- Easier to add new request types
- Reduced APIClient from 500 to 150 lines

No functional changes. All existing tests pass.
```

### テンプレート4: パフォーマンス改善

```bash
perf(database): optimize user query performance

User list queries were taking 200-500ms, causing noticeable lag
in the UI.

Analysis:
- Profiled database queries
- Identified missing index on user_id field
- Found N+1 query problem in user relationships

Optimizations:
- Added compound index on (user_id, created_at)
- Implemented eager loading for user relationships
- Added query result caching (5min TTL)

Results:
- Average query time: 200ms → 15ms (93% reduction)
- 99th percentile: 500ms → 30ms
- Database CPU usage: -40%

Benchmarked with 10,000 users over 1000 requests.

Refs #789
```

### Body書き方のコツ

```
✅ 72文字で改行（読みやすさ）
✅ 箇条書きを活用
✅ 測定結果を具体的に
✅ ビフォー・アフターを明示
✅ テスト結果を記載
✅ Issue番号を参照

❌ 曖昧な表現（"いくつか"、"多少"など）
❌ 感情的な表現
❌ コードスニペットの大量貼り付け
```

---

## Footerの書き方

### Issue参照

```bash
# 1つのIssueをクローズ
Closes #123

# 複数のIssueをクローズ
Closes #123, #456, #789

# 関連Issue（クローズしない）
Refs #111
Related to #222

# バグ修正
Fixes #567
Resolves #890
```

### Breaking Changes

```bash
# BREAKING CHANGEセクション
BREAKING CHANGE: API response format changed

The /users endpoint now returns { users: [...] }
instead of [...] directly.

Migration guide:
- Update API client to access response.users
- Update TypeScript types
- Run migration script: npm run migrate-api-types
```

### Co-authored-by

```bash
# ペアプログラミング時
Co-authored-by: John Doe <john@example.com>
Co-authored-by: Jane Smith <jane@example.com>
```

### その他のFooter

```bash
# レビュワー
Reviewed-by: Tech Lead <lead@example.com>

# 署名
Signed-off-by: Developer <dev@example.com>

# デプロイメモ
Deployment-note: Requires database migration
```

---

## Breaking Changes管理

### Breaking Changeとは

```
既存のユーザーやAPI利用者に影響を与える変更

例:
❌ API レスポンス形式の変更
❌ 関数シグネチャの変更
❌ デフォルト動作の変更
❌ 必須パラメータの追加
❌ 公開APIの削除
```

### 表記方法1: Type + `!`

```bash
feat(api)!: change user endpoint response format

BREAKING CHANGE: /api/users now returns paginated response

Before:
{
  "users": [...]
}

After:
{
  "data": [...],
  "pagination": {
    "page": 1,
    "total": 100
  }
}
```

### 表記方法2: Footerのみ

```bash
refactor(auth): simplify authentication flow

BREAKING CHANGE: AuthService.login() now returns Promise

Migration:
- Change from callback style to async/await
- Update all login calls to use await

Before:
AuthService.login(credentials, (error, user) => {})

After:
const user = await AuthService.login(credentials)
```

### Semantic Versioning への影響

```
BREAKING CHANGE → MAJOR version up

1.2.0 → 2.0.0
```

### Breaking Change チェックリスト

```markdown
Breaking Change 作成時:

- [ ] `!` または `BREAKING CHANGE:` を明記
- [ ] 変更理由を説明
- [ ] Before/After を明示
- [ ] Migration Guide を提供
- [ ] ドキュメント更新
- [ ] 該当Issue番号を参照
- [ ] レビュー承認
- [ ] チームに事前通知
```

---

## プロジェクト別実践例

### iOS (Swift) プロジェクト

```bash
# Feature追加
feat(auth): add Face ID authentication
feat(ui): add dark mode support
feat(network): implement retry logic with exponential backoff

# Bug修正
fix(login): resolve crash on iOS 17
fix(ui): correct layout on iPad landscape
fix(network): handle timeout errors properly

# Performance
perf(images): implement image caching with Kingfisher
perf(database): optimize Core Data fetch requests

# Refactoring
refactor(auth): migrate to Combine framework
refactor(ui): replace UIKit with SwiftUI

# Tests
test(auth): add unit tests for biometric login
test(ui): add snapshot tests for ProfileView

# Chores
chore(deps): update Alamofire to 5.8
chore(project): update Xcode build settings
```

### React / TypeScript プロジェクト

```bash
# Feature
feat(ui): add user profile component
feat(api): add authentication hooks
feat(state): implement Redux Toolkit slices

# Bug Fix
fix(form): resolve validation error display
fix(routing): correct navigation state
fix(api): handle 401 unauthorized properly

# Performance
perf(rendering): memoize expensive calculations
perf(bundle): code-split routes for faster load

# Refactoring
refactor(components): convert to functional components
refactor(types): migrate to strict TypeScript

# Tests
test(components): add React Testing Library tests
test(hooks): add unit tests for custom hooks

# Chores
chore(deps): update React to 18.2
chore(config): update ESLint rules
```

### Python / FastAPI プロジェクト

```bash
# Feature
feat(api): add user registration endpoint
feat(auth): implement JWT authentication
feat(database): add SQLAlchemy models

# Bug Fix
fix(api): resolve CORS configuration
fix(validation): handle null values properly
fix(database): correct migration script

# Performance
perf(api): add response caching with Redis
perf(database): optimize query with indexes

# Refactoring
refactor(api): extract validation logic
refactor(auth): simplify token generation

# Tests
test(api): add pytest for endpoints
test(auth): add integration tests

# Chores
chore(deps): update FastAPI to 0.100
chore(docker): optimize Docker image size
```

---

## 自動化ツール活用

### commitlint

**インストール:**
```bash
npm install --save-dev @commitlint/cli @commitlint/config-conventional
```

**設定:**
```javascript
// commitlint.config.js
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      [
        'feat',
        'fix',
        'docs',
        'style',
        'refactor',
        'perf',
        'test',
        'chore',
        'ci',
        'revert'
      ]
    ],
    'subject-case': [0], // 大文字小文字を許容
    'subject-max-length': [2, 'always', 50],
  }
};
```

**Git Hook統合:**
```bash
# Huskyインストール
npm install --save-dev husky

# Git Hookセットアップ
npx husky install
npx husky add .husky/commit-msg 'npx --no -- commitlint --edit $1'
```

### commitizen

**インストール:**
```bash
npm install --save-dev commitizen cz-conventional-changelog
```

**設定:**
```json
// package.json
{
  "scripts": {
    "commit": "cz"
  },
  "config": {
    "commitizen": {
      "path": "cz-conventional-changelog"
    }
  }
}
```

**使用:**
```bash
# 通常のgit commitの代わりに
npm run commit

# 対話形式でコミットメッセージ作成
? Select the type of change: feat
? What is the scope: auth
? Write a short description: add Google OAuth
? Provide a longer description: (optional)
? Are there any breaking changes? No
? Does this close any issues? #123
```

### Git Template

**テンプレート作成:**
```bash
# ~/.gitmessage.txt
# <type>(<scope>): <subject>

# <body>

# <footer>

# Type: feat, fix, docs, style, refactor, perf, test, chore, ci
# Scope: auth, ui, api, database, etc.
# Subject: imperative, lowercase, no period, max 50 chars
# Body: what, why, how (wrap at 72 chars)
# Footer: Closes #123, BREAKING CHANGE
```

**設定:**
```bash
git config --global commit.template ~/.gitmessage.txt
```

---

## CHANGELOG自動生成

### conventional-changelog

**インストール:**
```bash
npm install --save-dev conventional-changelog-cli
```

**生成:**
```bash
# CHANGELOG.mdを生成
npx conventional-changelog -p angular -i CHANGELOG.md -s
```

**結果（CHANGELOG.md）:**
```markdown
# Changelog

## [1.3.0](https://github.com/user/repo/compare/v1.2.0...v1.3.0) (2026-01-02)

### Features

* **auth:** add biometric authentication ([abc123](https://github.com/user/repo/commit/abc123))
* **ui:** add dark mode support ([def456](https://github.com/user/repo/commit/def456))

### Bug Fixes

* **login:** resolve keyboard dismissal on iOS 17 ([ghi789](https://github.com/user/repo/commit/ghi789))

### Performance Improvements

* **images:** implement lazy loading ([jkl012](https://github.com/user/repo/commit/jkl012))
```

### GitHub Release Notes自動生成

**GitHub Actions:**
```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Generate changelog
        run: |
          npx conventional-changelog -p angular -i CHANGELOG.md -s
          git add CHANGELOG.md
          git commit -m "docs: update CHANGELOG.md"

      - name: Create GitHub Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          body_path: CHANGELOG.md
```

---

## Semantic Versioning連携

### semantic-release

**インストール:**
```bash
npm install --save-dev semantic-release
```

**設定:**
```json
// .releaserc.json
{
  "branches": ["main"],
  "plugins": [
    "@semantic-release/commit-analyzer",
    "@semantic-release/release-notes-generator",
    "@semantic-release/changelog",
    "@semantic-release/npm",
    "@semantic-release/github",
    "@semantic-release/git"
  ]
}
```

**GitHub Actions:**
```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    branches: [main]

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3

      - name: Semantic Release
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          NPM_TOKEN: ${{ secrets.NPM_TOKEN }}
        run: npx semantic-release
```

### バージョニングルール

```
コミットメッセージ → バージョン変更

fix: ...              → 1.2.0 → 1.2.1 (PATCH)
feat: ...             → 1.2.0 → 1.3.0 (MINOR)
feat!: ... or
BREAKING CHANGE: ...  → 1.2.0 → 2.0.0 (MAJOR)

docs/style/test/chore → バージョン変更なし
```

---

## チーム運用ベストプラクティス

### ルール文書化

```markdown
# CONTRIBUTING.md

## コミットメッセージ規約

本プロジェクトは Conventional Commits を採用しています。

### フォーマット

<type>(<scope>): <subject>

### Type

- `feat`: 新機能
- `fix`: バグ修正
- `docs`: ドキュメント
- `style`: フォーマット
- `refactor`: リファクタリング
- `perf`: パフォーマンス改善
- `test`: テスト
- `chore`: ビルド・設定

### Scope

- `auth`: 認証
- `ui`: UI
- `api`: API
- `database`: データベース

### 例

```bash
feat(auth): add Google OAuth login
fix(ui): resolve layout issue on iPad
docs(api): add endpoint documentation
```

### 自動チェック

コミット時に commitlint が自動的にチェックします。
```

### オンボーディング

```markdown
# 新メンバー向けガイド

## コミットメッセージの書き方

1. Commitizen を使う（推奨）:
   ```bash
   npm run commit
   ```

2. テンプレートを使う:
   ```bash
   git config commit.template .gitmessage.txt
   git commit  # エディタが開く
   ```

3. 手動で書く:
   ```bash
   git commit -m "feat(auth): add Google OAuth"
   ```

## よくある間違い

❌ `git commit -m "fix"`
✅ `git commit -m "fix(ui): resolve button alignment"`

❌ `git commit -m "Added new feature"`
✅ `git commit -m "feat(api): add user search"`
```

### レビュープロセス

```markdown
## PR作成前チェックリスト

- [ ] 全てのコミットメッセージがConventional Commitsに準拠
- [ ] Scopeが適切に設定されている
- [ ] Subjectが50文字以内
- [ ] 複雑な変更にはBodyが記載されている
- [ ] 関連IssueがFooterに記載されている
- [ ] Breaking Changeがある場合、適切にマークされている

## レビュワーの確認事項

- [ ] コミット履歴がきれい（squash不要）
- [ ] 各コミットが原子的（1commit = 1change）
- [ ] コミットメッセージで変更内容が理解できる
```

---

## コミット粒度の最適化

### 原子的コミット（Atomic Commits）

**原則:**
```
1コミット = 1つの論理的な変更
```

**Good Example:**
```bash
# ✅ 3つの独立したコミット
git commit -m "feat(auth): add login UI components"
git commit -m "feat(auth): add login business logic"
git commit -m "test(auth): add login flow tests"
```

**Bad Example:**
```bash
# ❌ 1つの巨大なコミット
git commit -m "feat(auth): add complete login feature"
# （UI + ロジック + テスト全てを含む）
```

### コミット分割戦略

```bash
# 機能を段階的にコミット

# 1. データモデル
git add src/models/User.ts
git commit -m "feat(model): add User model"

# 2. API クライアント
git add src/api/UserApi.ts
git commit -m "feat(api): add user API client"

# 3. ViewModel
git add src/viewmodels/UserViewModel.ts
git commit -m "feat(viewmodel): add UserViewModel"

# 4. UI コンポーネント
git add src/components/UserProfile.tsx
git commit -m "feat(ui): add UserProfile component"

# 5. テスト
git add src/__tests__/UserProfile.test.tsx
git commit -m "test(ui): add UserProfile tests"
```

### git add -p（部分的ステージング）

```bash
# 1つのファイル内の変更を分割してコミット

git add -p src/auth/LoginService.ts

# 対話的に選択:
# y - このhunkをstageする
# n - このhunkをskipする
# s - このhunkを分割する

# 最初のコミット（バグ修正）
git commit -m "fix(auth): resolve null pointer exception"

# 2番目のコミット（リファクタリング）
git add -p src/auth/LoginService.ts
git commit -m "refactor(auth): simplify error handling"
```

---

## トラブルシューティング

### Q1: コミットメッセージを間違えた

**直前のコミットの場合:**
```bash
# コミットメッセージ修正
git commit --amend -m "correct message"

# 既にpushしている場合（単独作業）
git push --force-with-lease
```

**過去のコミットの場合:**
```bash
# インタラクティブrebase
git rebase -i HEAD~3

# エディタで:
# pick → reword に変更
# 保存して終了
# 新しいメッセージを入力
```

### Q2: commitlintでエラーが出る

**エラー例:**
```
⧗   input: feat add login
✖   subject may not be empty [subject-empty]
✖   type may not be empty [type-empty]
```

**修正:**
```bash
# 正しいフォーマット
git commit -m "feat(auth): add login"
```

### Q3: 複数のTypeが該当する

```
例: バグ修正 + リファクタリング

悪い例:
git commit -m "fix/refactor: resolve bug and refactor code"

良い例（分割）:
git commit -m "fix(ui): resolve layout bug"
git commit -m "refactor(ui): simplify component structure"
```

### Q4: Scopeが不明確

```
例: 複数のレイヤーにまたがる変更

選択肢1: より広いScope
git commit -m "feat(auth): add complete login flow"

選択肢2: 分割
git commit -m "feat(ui): add login UI"
git commit -m "feat(api): add login API"
```

---

## まとめ

### 重要ポイント

```
1. フォーマット遵守
   <type>(<scope>): <subject>

2. Type選択
   feat/fix/docs/style/refactor/perf/test/chore/ci

3. Subject
   - 50文字以内
   - 命令形
   - 小文字で始める

4. Body（必要に応じて）
   - What/Why/How を説明
   - 72文字で改行

5. Footer
   - Issue参照（Closes #123）
   - Breaking Change明記
```

### チェックリスト

```markdown
□ Type は適切か
□ Scope は設定されているか
□ Subject は50文字以内か
□ 命令形で書かれているか
□ 複雑な変更にBody が記載されているか
□ Issue番号を参照しているか
□ Breaking Change がある場合マークしたか
□ commitlint エラーなしか
```

### 次のステップ

```
□ チームでConventional Commits採用を合意
□ CONTRIBUTING.md作成
□ commitlintセットアップ
□ commitizen導入
□ CI/CDでチェック自動化
□ CHANGELOG自動生成設定
□ semantic-release導入検討
```

---

**このガイドで一貫性のあるコミット履歴を構築しましょう！**

**文字数:** 約25,000文字

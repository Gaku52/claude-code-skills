# コミットメッセージ規約

## Conventional Commits完全ガイド

良いコミットメッセージは：
- ✅ 変更の意図が明確
- ✅ 自動的にCHANGELOG生成可能
- ✅ セマンティックバージョニングと連携
- ✅ コードレビューが容易

---

## 基本フォーマット

```
<type>(<scope>): <subject>

<body>

<footer>
```

### 例

```
feat(auth): add biometric authentication support

Implemented Face ID and Touch ID authentication for iOS.
Users can now enable biometric login from settings.

- Added BiometricAuthManager
- Updated LoginViewController
- Added unit tests for biometric flow

Closes #123
```

---

## Type（必須）

| Type | 説明 | CHANGELOG | 例 |
|------|------|-----------|-----|
| **feat** | 新機能 | ✅ ADDED | `feat(ui): add dark mode support` |
| **fix** | バグ修正 | ✅ FIXED | `fix(network): resolve timeout issue` |
| **docs** | ドキュメントのみ | ❌ | `docs(readme): update installation steps` |
| **style** | コードスタイル（動作変更なし） | ❌ | `style(login): fix indentation` |
| **refactor** | リファクタリング | ❌ | `refactor(api): simplify request builder` |
| **perf** | パフォーマンス改善 | ✅ IMPROVED | `perf(images): implement lazy loading` |
| **test** | テスト追加・修正 | ❌ | `test(auth): add biometric tests` |
| **chore** | ビルド、設定等 | ❌ | `chore(deps): update Alamofire to 5.8` |
| **ci** | CI設定 | ❌ | `ci(github): add code coverage report` |
| **revert** | コミット取り消し | ❌ | `revert: feat(auth): add biometric` |

---

## Scope（オプションだが推奨）

変更の影響範囲を示す：

### iOS開発での一般的なScope

```
auth        - 認証関連
ui          - UI全般
network     - ネットワーク層
database    - データ永続化
api         - APIクライアント
model       - データモデル
utils       - ユーティリティ
config      - 設定
deps        - 依存関係
test        - テスト関連
```

### 機能別Scope

```
login       - ログイン画面
profile     - プロフィール画面
settings    - 設定画面
cart        - カート機能
payment     - 決済機能
```

---

## Subject（必須）

簡潔な変更内容（50文字以内）：

### ✅ Good

```
add biometric authentication
fix memory leak in image cache
update README installation steps
refactor network layer for testability
```

### ❌ Bad

```
Added biometric authentication feature with Face ID and Touch ID support for user login  # 長すぎ
fix bug  # 具体性がない
Update  # 何を更新したか不明
WIP  # 作業中コミットはNG
```

### ルール

- ✅ 動詞の原形で始める（add, fix, update, remove）
- ✅ 小文字で始める
- ✅ ピリオドなし
- ✅ 50文字以内
- ❌ 過去形は使わない（added, fixed）
- ❌ 「〜した」「〜を実装」などの日本語的表現

---

## Body（オプション）

詳細な説明が必要な場合：

### 含めるべき内容

- **What**: 何を変更したか
- **Why**: なぜ変更したか
- **How**: どのように実装したか（複雑な場合）

### 例

```
feat(cache): implement LRU cache for images

The app was consuming too much memory when displaying
image-heavy content. Implemented an LRU (Least Recently Used)
cache to automatically evict old images.

- Cache size limit: 100MB
- Eviction policy: LRU
- Persists to disk for offline support

This reduces peak memory usage by approximately 40%.
```

### ルール

- 72文字で改行
- Subjectから1行空ける
- 箇条書きOK（-, *, +）

---

## Footer（オプション）

### Breaking Changes（重大な変更）

```
feat(api)!: change authentication response format

BREAKING CHANGE: API response format changed from
{ token: string } to { accessToken: string, refreshToken: string }

Migration guide:
- Update APIClient to use new response format
- Store both access and refresh tokens
```

`!` を type の後に追加 + BREAKING CHANGE: で明示

### Issue参照

```
Closes #123
Fixes #456, #789
Refs #111
```

---

## 実践例集

### 新機能追加

```
feat(payment): add Apple Pay support

Integrated PassKit for Apple Pay payments.
Users can now complete purchases using Apple Pay
from the checkout screen.

- Added PaymentManager
- Updated CheckoutViewController
- Added unit and integration tests

Closes #234
```

### バグ修正

```
fix(login): resolve keyboard dismissal issue on iOS 17

The keyboard was not dismissing when tapping outside
the text field on iOS 17 devices. Added tap gesture
recognizer to handle this case.

Fixes #567
```

### リファクタリング

```
refactor(network): extract request building logic

Extracted request building logic from APIClient into
separate RequestBuilder class for better testability
and reusability.

- Created RequestBuilder protocol
- Implemented URLRequestBuilder
- Updated all API calls to use new builder
```

### パフォーマンス改善

```
perf(database): optimize query performance with indexes

Added database indexes on frequently queried fields
(user_id, created_at). Reduced average query time
from 200ms to 15ms.
```

### ドキュメント

```
docs(architecture): add system design documentation

Added comprehensive architecture documentation covering:
- MVVM pattern usage
- Dependency injection strategy
- Module boundaries
```

### 依存関係更新

```
chore(deps): update dependencies to latest versions

- Alamofire 5.7 → 5.8
- Realm 10.40 → 10.42
- Kingfisher 7.9 → 7.10

All updates are backward compatible.
```

### CI/CD

```
ci(github): add automated screenshot testing

Added workflow to generate and upload screenshots
on every PR for visual regression testing.
```

---

## 複数の変更がある場合

### ❌ 避けるべき

```
feat(ui): add dark mode and fix login bug and update deps
```

### ✅ 推奨：コミットを分ける

```
feat(ui): add dark mode support
fix(login): resolve authentication timeout
chore(deps): update Alamofire to 5.8
```

---

## 特殊なケース

### マージコミット

```
Merge branch 'feature/user-auth' into develop
```

自動生成されるのでそのまま使用

### Revertコミット

```
revert: feat(auth): add biometric authentication

This reverts commit abc123def456.

Reason: Causing crashes on iOS 16.
```

---

## ツールによる自動化

### Git Hook（commit-msg）

```bash
#!/bin/sh
# .git/hooks/commit-msg

commit_msg=$(cat "$1")

# Conventional Commits形式チェック
if ! echo "$commit_msg" | grep -qE "^(feat|fix|docs|style|refactor|perf|test|chore|ci|revert)(\(.+\))?: .+"; then
    echo "Error: Commit message does not follow Conventional Commits format"
    echo ""
    echo "Format: <type>(<scope>): <subject>"
    echo "Example: feat(auth): add biometric support"
    exit 1
fi
```

### テンプレート設定

```bash
git config commit.template ~/.claude/skills/git-workflow/templates/commit-message-template.md
```

---

## コミットメッセージから学ぶ

### 良い例（実際のOSSから）

```
feat(router): add support for named routes

fix(ssr): prevent hydration mismatch in async components

perf(compiler): optimize template compilation by 30%

refactor(reactivity)!: simplify ref implementation

BREAKING CHANGE: ref() now returns a readonly ref by default
```

### 悪い例と改善

| ❌ Bad | ✅ Good |
|-------|--------|
| `Update` | `docs(readme): update installation steps` |
| `Fix bug` | `fix(login): resolve timeout on slow networks` |
| `WIP` | (そもそもコミットしない) |
| `Misc changes` | 複数のコミットに分割 |
| `Add feature` | `feat(payment): add Apple Pay support` |

---

## チェックリスト

コミット前に確認：

- [ ] Type は適切か？
- [ ] Scope は明確か？
- [ ] Subject は50文字以内か？
- [ ] 動詞の原形で始まっているか？
- [ ] 複数の変更を1つのコミットにしていないか？
- [ ] Breaking Change がある場合、`!` と `BREAKING CHANGE:` を記載したか？
- [ ] Issue番号を参照したか？（該当する場合）

---

## 参考リンク

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Angular Commit Guidelines](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit)
- [Semantic Versioning](https://semver.org/)

---

## 過去の失敗事例

→ [../incidents/commit-message-mistakes.md](../incidents/commit-message-mistakes.md)

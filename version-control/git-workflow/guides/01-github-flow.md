# GitHub Flow完全ガイド

> **最終更新:** 2025-12-27
> **対象読者:** Git初心者〜中級者、小〜中規模チーム開発者
> **推定読了時間:** 45分

## 📋 目次

1. [GitHub Flowとは](#github-flowとは)
2. [基本概念](#基本概念)
3. [完全なワークフロー](#完全なワークフロー)
4. [実践例：Feature開発](#実践例feature開発)
5. [実践例：Hotfix対応](#実践例hotfix対応)
6. [ベストプラクティス](#ベストプラクティス)
7. [よくある失敗と対策](#よくある失敗と対策)
8. [チーム運用のポイント](#チーム運用のポイント)
9. [ツール活用](#ツール活用)
10. [トラブルシューティング](#トラブルシューティング)

---

## GitHub Flowとは

### 定義

GitHub Flowは、**シンプルで軽量なブランチ戦略**です。GitHubが提唱し、多くのチームで採用されています。

**核心原則:**
- `main`ブランチは常にデプロイ可能な状態
- 全ての開発は`main`から分岐したfeatureブランチで行う
- Pull Request（PR）でレビュー・マージ
- マージ後は即座にデプロイ

### Git Flowとの違い

| 項目 | GitHub Flow | Git Flow |
|------|-------------|----------|
| **ブランチ数** | 少ない（main + feature） | 多い（main/develop/feature/release/hotfix） |
| **複雑さ** | シンプル | 複雑 |
| **リリース頻度** | 高頻度（継続的デプロイ） | 定期リリース |
| **適用ケース** | Webサービス、SaaS | パッケージソフト、大規模プロジェクト |
| **学習コスト** | 低い | 高い |

### 適用ケース

**GitHub Flowが最適:**
- ✅ 継続的デプロイ（CD）を実践している
- ✅ リリースサイクルが短い（週次・日次）
- ✅ 小〜中規模チーム（2-20人）
- ✅ Webアプリケーション・SaaS
- ✅ シンプルさを重視

**GitHub Flowが不向き:**
- ❌ 複数バージョンを並行サポート
- ❌ 定期リリース（月次・四半期）
- ❌ リリース前の長期テスト期間
- ❌ モバイルアプリ（App Store審査待ち）

---

## 基本概念

### 1. mainブランチ

**役割:**
- プロダクション環境と同じ状態
- 常にデプロイ可能
- 全ての変更の起点

**ルール:**
- ❌ 直接コミット禁止
- ✅ PRを通してのみマージ
- ✅ CI/CDが全てパス
- ✅ レビュー承認必須

### 2. featureブランチ

**命名規則:**
```
<type>/<ticket-id>-<short-description>

例:
feature/USER-123-add-login-page
bugfix/BUG-456-fix-validation-error
hotfix/CRITICAL-789-fix-payment-crash
```

**ライフサイクル:**
1. `main`から分岐
2. 開発・コミット
3. PR作成
4. レビュー・修正
5. `main`へマージ
6. ブランチ削除

### 3. Pull Request（PR）

**目的:**
- コードレビュー
- 自動テスト実行
- デプロイ前チェック
- ナレッジ共有

**PR作成タイミング:**
- 🟢 **Early PR（推奨）**: 開発開始直後にDraft PRを作成
- 🟡 **Ready for Review**: 実装完了後、レビュー可能になったら
- 🔴 **Late PR（非推奨）**: マージ直前に作成

---

## 完全なワークフロー

### ステップ1: ブランチ作成

```bash
# 1. mainブランチを最新化
git checkout main
git pull origin main

# 2. featureブランチ作成
git checkout -b feature/USER-123-add-profile-page

# 3. リモートにpush（Early PR作成のため）
git push -u origin feature/USER-123-add-profile-page
```

**ポイント:**
- 必ず最新の`main`から分岐
- ブランチ名は説明的に
- チケットIDを含める

### ステップ2: 開発

```bash
# 1. ファイル編集
vim src/ProfilePage.tsx

# 2. 変更を確認
git status
git diff

# 3. ステージング
git add src/ProfilePage.tsx

# 4. コミット（Conventional Commits形式）
git commit -m "feat(profile): add user profile page

- Add ProfilePage component
- Implement avatar upload
- Add bio editing feature

Refs: USER-123"

# 5. プッシュ
git push
```

**コミット頻度:**
- 🟢 **小さく頻繁に**: 1機能1コミット
- 🟡 **適度に**: 1日1-3コミット
- 🔴 **大きなコミット**: 複数機能を1コミット（避ける）

### ステップ3: PR作成

**GitHub上で:**

1. **Draft PR作成**（開発開始時）
   ```markdown
   ## 🚧 Work in Progress

   プロフィールページを実装中です。

   ### TODO
   - [ ] UIコンポーネント実装
   - [ ] APIエンドポイント接続
   - [ ] テスト追加
   - [ ] スタイリング調整
   ```

2. **Ready for Review**（実装完了時）
   ```markdown
   ## 概要
   ユーザープロフィールページを追加しました。

   ## 変更内容
   - プロフィール表示コンポーネント
   - アバター画像アップロード機能
   - 自己紹介文編集機能

   ## 動作確認
   - [x] 単体テスト実行（カバレッジ95%）
   - [x] E2Eテスト実行
   - [x] 実機での動作確認（iOS/Android）
   - [x] アクセシビリティチェック

   ## スクリーンショット
   ![profile-page](./screenshots/profile-page.png)

   ## 関連Issue
   Closes #123
   ```

3. **レビュワー指定**
   - コード担当者（必須）
   - デザイナー（UI変更時）
   - QA担当（重要機能時）

### ステップ4: レビュー対応

**レビューコメントへの対応:**

```bash
# 1. レビューコメントを確認
# （GitHubのPRページで）

# 2. 修正実装
vim src/ProfilePage.tsx

# 3. コミット
git add src/ProfilePage.tsx
git commit -m "fix(profile): address review comments

- Improve error handling
- Add loading state
- Fix accessibility issues"

# 4. プッシュ（自動的にPRが更新される）
git push
```

**良いレビュー対応:**
- ✅ コメントに返信して理解を示す
- ✅ 修正内容を明確にコミットメッセージに記載
- ✅ 大きな変更は再レビュー依頼
- ❌ 無言で修正してpush

### ステップ5: CI/CDチェック

**自動実行される項目:**
- Lint（ESLint, SwiftLint等）
- Unit Tests
- Integration Tests
- E2E Tests
- ビルド
- セキュリティスキャン

**全てパスするまで:**
```bash
# ローカルでCIと同じチェックを実行
npm run lint
npm run test
npm run build

# 修正してpush
git add .
git commit -m "test: fix failing tests"
git push
```

### ステップ6: マージ

**マージ前最終チェック:**
- [ ] CI/CD全てパス
- [ ] レビュー承認済み（最低1人）
- [ ] コンフリクトなし
- [ ] mainブランチの最新変更を取り込み済み

**マージ方法:**

```bash
# オプション1: GitHub上でマージ（推奨）
# 「Squash and merge」または「Merge commit」を選択

# オプション2: ローカルでマージ
git checkout main
git pull origin main
git merge --no-ff feature/USER-123-add-profile-page
git push origin main
```

**マージ戦略:**
- 🟢 **Squash and merge**: 1つのコミットにまとめる（推奨）
- 🟡 **Merge commit**: コミット履歴を保持
- 🔴 **Rebase and merge**: 履歴を線形に（慎重に）

### ステップ7: デプロイ

```bash
# 自動デプロイ（CD設定済みの場合）
# → mainへのマージで自動的にデプロイ

# 手動デプロイ
git checkout main
git pull origin main
./deploy.sh production
```

### ステップ8: ブランチ削除

```bash
# GitHub上で自動削除（設定済みの場合）

# または手動削除
git branch -d feature/USER-123-add-profile-page
git push origin --delete feature/USER-123-add-profile-page
```

---

## 実践例：Feature開発

### シナリオ

「ユーザーがお気に入りボタンを押せる機能」を追加

### 完全な手順

#### 1. チケット確認

```
[USER-456] お気に入り機能の追加

要件:
- 各記事にお気に入りボタンを表示
- ログインユーザーのみ使用可能
- お気に入り数を表示
- お気に入り一覧ページを追加
```

#### 2. ブランチ作成

```bash
git checkout main
git pull origin main
git checkout -b feature/USER-456-add-favorite-button
git push -u origin feature/USER-456-add-favorite-button
```

#### 3. Draft PR作成

GitHubで即座にDraft PR作成:

```markdown
## 🚧 WIP: お気に入り機能の実装

### 実装予定
- [ ] フロントエンド: お気に入りボタンコンポーネント
- [ ] バックエンド: お気に入りAPI
- [ ] データベース: favoritesテーブル
- [ ] テスト: Unit + E2E
- [ ] ドキュメント更新

### 進捗
- [x] DB設計完了
- [ ] API実装中...
```

#### 4. 開発（小さくコミット）

```bash
# コミット1: データベーススキーマ
git add db/migrations/20250127_create_favorites.sql
git commit -m "feat(db): add favorites table schema

- Create favorites table
- Add user_id and article_id foreign keys
- Add unique constraint

Refs: USER-456"
git push

# コミット2: APIエンドポイント
git add src/api/favorites.ts
git commit -m "feat(api): add favorites endpoints

- POST /api/favorites
- DELETE /api/favorites/:id
- GET /api/favorites (list user favorites)

Refs: USER-456"
git push

# コミット3: フロントエンドコンポーネント
git add src/components/FavoriteButton.tsx
git commit -m "feat(ui): add favorite button component

- Toggle favorite state
- Show favorite count
- Handle loading/error states

Refs: USER-456"
git push

# コミット4: テスト
git add src/__tests__/FavoriteButton.test.tsx
git commit -m "test(ui): add favorite button tests

- Test toggle functionality
- Test loading states
- Test error handling

Refs: USER-456"
git push
```

#### 5. Ready for Review

Draft PRを「Ready for Review」に変更:

```markdown
## 概要
お気に入り機能を実装しました。

## 変更内容
- フロントエンド: FavoriteButtonコンポーネント
- バックエンド: Favorites API (POST/DELETE/GET)
- データベース: favoritesテーブル追加
- テスト: Unit + E2E（カバレッジ92%）

## 動作確認
- [x] ローカル環境で動作確認
- [x] ログイン/非ログイン時の挙動確認
- [x] エラーハンドリング確認
- [x] パフォーマンステスト（1000件のお気に入りで問題なし）

## スクリーンショット
![favorite-button](./screenshots/favorite-button.gif)

## 関連Issue
Closes #456
```

#### 6. レビュー・修正

**レビューコメント例:**

> **Reviewer:** エラー時のメッセージが英語ですが、日本語にしませんか？

```bash
# 修正
vim src/components/FavoriteButton.tsx

git add src/components/FavoriteButton.tsx
git commit -m "fix(ui): change error messages to Japanese

As per review comment by @reviewer"
git push
```

> **Reviewer:** LGTM! 👍

#### 7. マージ

GitHub上で「Squash and merge」:

```
feat(favorites): add favorite button feature (#456)

ユーザーがお気に入りボタンを押せる機能を追加

- フロントエンド: FavoriteButtonコンポーネント
- バックエンド: Favorites API
- データベース: favoritesテーブル
- テスト: カバレッジ92%
```

#### 8. デプロイ確認

```bash
# CI/CDが自動デプロイ
# → プロダクション環境で動作確認

# Slackに通知
# 「✅ feature/USER-456-add-favorite-button がプロダクションにデプロイされました」
```

---

## 実践例：Hotfix対応

### シナリオ

「本番環境でログイン機能が動作しない」緊急バグ

### 完全な手順

#### 1. 緊急対応開始

```bash
# 即座にブランチ作成
git checkout main
git pull origin main
git checkout -b hotfix/CRITICAL-789-fix-login-crash
git push -u origin hotfix/CRITICAL-789-fix-login-crash
```

#### 2. 原因調査

```bash
# ローカルで再現
npm run dev

# ログ確認
tail -f logs/production.log

# 原因特定: 環境変数が未設定
```

#### 3. 修正

```bash
# 修正実装
vim src/auth/login.ts

# コミット
git add src/auth/login.ts
git commit -m "fix(auth): fix login crash due to missing env var

- Add fallback for AUTH_SECRET
- Add validation for required env vars
- Improve error logging

Fixes: CRITICAL-789"
git push
```

#### 4. PR作成（緊急）

```markdown
## 🚨 Hotfix: ログイン機能の修正

### 問題
本番環境でログイン時にクラッシュ

### 原因
AUTH_SECRET環境変数が未設定

### 修正内容
- 環境変数のフォールバック追加
- バリデーション強化
- エラーログ改善

### 動作確認
- [x] ローカルで修正確認
- [x] ステージング環境で確認
- [x] CI/CDパス

### 緊急度
🔴 **Critical** - 即座にデプロイ必要
```

#### 5. 即座レビュー

```bash
# Slackで緊急レビュー依頼
# 「@team 緊急PRレビューお願いします！本番ログインが止まっています」

# 5分後にApprove
```

#### 6. 即座マージ・デプロイ

```bash
# GitHub上でマージ
# → 自動デプロイ

# 3分後にプロダクション反映

# 動作確認
curl https://api.example.com/auth/login -d '{"email":"test@example.com"}'
# → 正常動作確認
```

#### 7. 事後対応

```bash
# インシデントレポート作成
vim incidents/2025-01-27-login-crash.md

git add incidents/2025-01-27-login-crash.md
git commit -m "docs(incident): add login crash incident report"
git push
```

---

## ベストプラクティス

### 1. ブランチは小さく、短命に

**推奨:**
- ブランチのライフタイム: 1-3日
- 変更行数: 200-500行
- 1PR = 1機能

**理由:**
- レビューしやすい
- コンフリクトが少ない
- デプロイリスク低減

### 2. Early PR（Draft PR）

**タイミング:**
開発開始直後にDraft PRを作成

**メリット:**
- 進捗の可視化
- 早期フィードバック
- CI/CDの早期実行

**例:**
```markdown
## 🚧 WIP: ユーザー検索機能

### 実装予定
- [ ] 検索UI
- [ ] APIエンドポイント
- [ ] テスト

### 進捗: 30%
```

### 3. コミットは原子的に

**原子的コミット:**
1コミット = 1つの変更

**良い例:**
```bash
git commit -m "feat(search): add search input component"
git commit -m "feat(search): add search API endpoint"
git commit -m "test(search): add search component tests"
```

**悪い例:**
```bash
git commit -m "add search feature and fix some bugs"
```

### 4. mainは常にデプロイ可能

**ルール:**
- ❌ 壊れたコードをマージしない
- ✅ CI/CD全パス必須
- ✅ レビュー承認必須

**CI/CD設定例:**
```yaml
# .github/workflows/ci.yml
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: npm test
      - run: npm run lint
      - run: npm run build
```

### 5. デプロイは小さく頻繁に

**推奨頻度:**
- 🟢 1日複数回
- 🟡 1日1回
- 🔴 週1回（リスク高）

**メリット:**
- 問題の早期発見
- ロールバック容易
- ユーザーへの価値提供が早い

### 6. コミュニケーション重視

**PR説明は詳細に:**
```markdown
## 概要
短く要約

## なぜこの変更が必要か
背景・理由

## 何を変更したか
具体的な変更内容

## どうやってテストしたか
テスト手順

## スクリーンショット
視覚的な確認
```

### 7. mainブランチ保護

**GitHub設定:**
```
Settings > Branches > Branch protection rules

✅ Require pull request reviews before merging
✅ Require status checks to pass before merging
✅ Require branches to be up to date before merging
✅ Include administrators
```

---

## よくある失敗と対策

### 失敗1: 大きすぎるPR

**症状:**
- 1000行以上の変更
- 複数機能を1PRに
- レビューに数日かかる

**対策:**
```bash
# PRを分割
git checkout main
git checkout -b feature/USER-123-part1-ui
# UIのみ実装・PR

git checkout main
git checkout -b feature/USER-123-part2-api
# API実装・PR（UIマージ後）
```

### 失敗2: mainから分岐し忘れ

**症状:**
```bash
# 間違い: feature/oldから新featureを分岐
git checkout feature/old-feature
git checkout -b feature/new-feature
```

**対策:**
```bash
# 正しい: 常にmainから分岐
git checkout main
git pull origin main
git checkout -b feature/new-feature
```

### 失敗3: コンフリクト放置

**症状:**
- PRに「Conflicts」表示
- マージできない

**対策:**
```bash
# mainの最新を取り込む
git checkout main
git pull origin main
git checkout feature/USER-123
git merge main
# コンフリクト解決
git add .
git commit -m "merge: resolve conflicts with main"
git push
```

### 失敗4: CI/CD失敗を無視

**症状:**
- テスト失敗してるのにマージ
- 本番環境が壊れる

**対策:**
```bash
# ローカルでCIと同じチェック実行
npm run ci

# 全てパスするまで修正
npm run lint:fix
npm run test:fix
git add .
git commit -m "fix: resolve CI failures"
git push
```

### 失敗5: レビュー待ち時間の無駄

**症状:**
- レビュー待ちで1日待つ
- 生産性低下

**対策:**
1. **Early PR**: 早めにDraft PR作成
2. **Slack通知**: レビュー依頼を通知
3. **並行作業**: レビュー待ちの間に別タスク
4. **セルフレビュー**: PR作成前に自分でレビュー

### 失敗6: ブランチ削除忘れ

**症状:**
- マージ済みブランチが大量に残る
- どれが有効か不明

**対策:**
```bash
# GitHub設定で自動削除
Settings > General > Automatically delete head branches

# または手動削除
git branch -d feature/USER-123
git push origin --delete feature/USER-123
```

### 失敗7: コミットメッセージが雑

**症状:**
```bash
git commit -m "fix"
git commit -m "update"
git commit -m "wip"
```

**対策:**
```bash
# Conventional Commits形式
git commit -m "feat(auth): add Google OAuth login

- Integrate Google OAuth SDK
- Add login button to UI
- Handle authentication flow

Refs: USER-456"
```

---

## チーム運用のポイント

### 1. ブランチ命名規則の統一

**ルール文書化:**
```markdown
# BRANCHING.md

## ブランチ命名規則

<type>/<ticket-id>-<description>

### Type
- feature: 新機能
- bugfix: バグ修正
- hotfix: 緊急修正
- refactor: リファクタリング
- docs: ドキュメント

### 例
feature/USER-123-add-profile-page
bugfix/BUG-456-fix-validation
hotfix/CRITICAL-789-fix-crash
```

### 2. PRテンプレート

```markdown
<!-- .github/pull_request_template.md -->

## 概要
<!-- 変更内容を簡潔に -->

## 変更理由
<!-- なぜこの変更が必要か -->

## 変更内容
<!-- 具体的な変更点 -->
-
-

## テスト
<!-- どうやってテストしたか -->
- [ ] Unit Tests
- [ ] Integration Tests
- [ ] Manual Testing

## スクリーンショット
<!-- UI変更がある場合 -->

## チェックリスト
- [ ] コードレビュー完了
- [ ] CI/CDパス
- [ ] ドキュメント更新
- [ ] マイグレーションスクリプト（必要な場合）

## 関連Issue
Closes #
```

### 3. レビュー文化

**良いレビューの例:**
```
👍 コードが綺麗でわかりやすいです！

💡 Suggestion: ここはuseMemo使うとパフォーマンス改善できそうです
```js
const filtered = useMemo(() => items.filter(...), [items]);
```

❓ Question: この部分の意図を教えてください
```

**避けるべきレビュー:**
```
❌ 「これ間違ってる」（理由なし）
❌ 「全部書き直して」（具体性なし）
❌ 「前のやり方の方が良かった」（代案なし）
```

### 4. リリースフロー

```bash
# 1. mainにマージ
# → 自動的にステージング環境へデプロイ

# 2. ステージング環境で動作確認
npm run test:staging

# 3. 問題なければプロダクションへデプロイ
# → 手動承認 or 自動デプロイ（設定次第）

# 4. プロダクション動作確認
npm run test:production

# 5. 問題あればロールバック
git revert HEAD
git push origin main
```

### 5. コミュニケーションツール統合

**Slack通知設定:**
```yaml
# .github/workflows/pr-notification.yml
name: PR Notification
on:
  pull_request:
    types: [opened, ready_for_review]

jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
      - name: Notify Slack
        env:
          SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }}
        run: |
          curl -X POST $SLACK_WEBHOOK \
            -d '{"text":"New PR: ${{ github.event.pull_request.title }}"}'
```

---

## ツール活用

### 1. GitHub CLI

```bash
# インストール
brew install gh

# PR作成
gh pr create --title "feat: add feature" --body "Description"

# PR一覧
gh pr list

# PRレビュー
gh pr review 123 --approve
gh pr review 123 --comment --body "LGTM!"

# PRマージ
gh pr merge 123 --squash
```

### 2. Git Hooks

```bash
# .git/hooks/pre-commit
#!/bin/sh
npm run lint
npm run test
```

### 3. GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI
on: [pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm ci
      - run: npm test
      - run: npm run lint
      - run: npm run build
```

### 4. PR分析ツール

**Danger JS:**
```js
// dangerfile.js
import { danger, warn, fail } from 'danger';

// PRが大きすぎる
if (danger.github.pr.additions > 500) {
  warn('このPRは大きすぎます。分割を検討してください。');
}

// テストが追加されていない
if (!danger.git.modified_files.includes('test')) {
  fail('テストを追加してください。');
}
```

---

## トラブルシューティング

### Q1: コンフリクトが解決できない

**解決手順:**
```bash
# 1. mainの最新を取得
git checkout main
git pull origin main

# 2. featureブランチでmainをマージ
git checkout feature/USER-123
git merge main

# 3. コンフリクトファイルを確認
git status

# 4. VSCodeなどでコンフリクト解決

# 5. 解決後コミット
git add .
git commit -m "merge: resolve conflicts with main"
git push
```

### Q2: 間違ったブランチにコミットしてしまった

**解決手順:**
```bash
# 1. 正しいブランチを作成
git checkout -b correct-branch

# 2. 間違ったブランチから最新コミットを取得
git cherry-pick <commit-hash>

# 3. 間違ったブランチのコミットを削除
git checkout wrong-branch
git reset --hard HEAD~1

# 4. 正しいブランチにpush
git checkout correct-branch
git push -u origin correct-branch
```

### Q3: mainに直接コミットしてしまった

**解決手順:**
```bash
# 1. featureブランチを作成
git branch feature/USER-123

# 2. mainを1つ前に戻す
git reset --hard HEAD~1

# 3. featureブランチに切り替え
git checkout feature/USER-123

# 4. push
git push -u origin feature/USER-123

# 5. 通常のPRフローへ
```

### Q4: CI/CDが永遠に失敗する

**デバッグ手順:**
```bash
# 1. ローカルでCIと同じコマンド実行
npm run lint
npm run test
npm run build

# 2. エラーログ確認
cat logs/test.log

# 3. 修正
vim src/broken-file.ts

# 4. 再テスト
npm run test

# 5. push
git add .
git commit -m "fix: resolve CI failures"
git push
```

### Q5: レビュー依頼したのに反応がない

**対策:**
```bash
# 1. Slackで直接依頼
@reviewer PRレビューお願いします！ https://github.com/...

# 2. 別のレビュワーを追加
# （GitHub PRページで）

# 3. チーム会議で確認

# 4. 緊急の場合はペアプログラミング形式でレビュー
```

---

## まとめ

### GitHub Flowの要点

1. **シンプル**: mainブランチ + featureブランチのみ
2. **高頻度デプロイ**: mainマージ = デプロイ
3. **PR中心**: 全ての変更はPRを通す
4. **常にデプロイ可能**: mainは常に本番品質

### 成功の鍵

- ✅ 小さく頻繁にPR
- ✅ Early PR（Draft PR）
- ✅ コミュニケーション重視
- ✅ CI/CD完全自動化
- ✅ レビュー文化の醸成

### 次のステップ

1. [PRレビューガイド](./06-pr-review.md) でレビュースキル向上
2. [コンフリクト解決ガイド](./07-conflict-resolution.md) でトラブル対応力強化
3. [Git Hooks活用](./09-git-hooks.md) で自動化推進

---

**関連ドキュメント:**
- [Git Flow完全ガイド](./02-git-flow.md)
- [Trunk-Based Development](./03-trunk-based.md)
- [コミットメッセージ規約](./05-commit-messages.md)

**外部リソース:**
- [GitHub Flow公式ドキュメント](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Understanding the GitHub flow](https://guides.github.com/introduction/flow/)

---

## 付録A: GitHub Flow vs Git Flow 詳細比較

### プロジェクトタイプ別推奨

| プロジェクトタイプ | 推奨戦略 | 理由 |
|------------------|---------|------|
| WebアプリケーションSaaS | GitHub Flow | 継続的デプロイに最適 |
| モバイルアプリ（App Store） | Git Flow | 審査期間を考慮した定期リリース |
| オープンソースライブラリ | Git Flow | 複数バージョン並行サポート |
| 社内ツール | GitHub Flow | シンプルで運用しやすい |
| エンタープライズソフトウェア | Git Flow | 厳格なリリースプロセス |
| スタートアップMVP | GitHub Flow | スピード重視 |

### チームサイズ別推奨

| チームサイズ | 推奨戦略 | 理由 |
|------------|---------|------|
| 1-5人 | GitHub Flow | シンプル、学習コスト低 |
| 6-20人 | GitHub Flow | PR中心の協調作業に適している |
| 21-50人 | Git Flow | 複数チーム並行開発の制御 |
| 51人以上 | Git Flow + Monorepo | 大規模プロジェクト管理 |

### 技術スタック別考慮事項

**Next.js / React:**
```bash
# Vercel自動デプロイと相性良好
# → GitHub Flow推奨

git checkout -b feature/new-page
# ... 開発 ...
git push
# → 自動的にプレビューデプロイ
# → mainマージで本番デプロイ
```

**iOS / Android:**
```bash
# App Store審査期間を考慮
# → Git Flow推奨

git checkout -b release/1.2.0
# ... リリース準備 ...
# → ステージング環境テスト
# → App Store申請
# → 審査通過後mainへマージ
```

**マイクロサービス:**
```bash
# サービスごとに独立してデプロイ
# → GitHub Flow推奨（サービスごと）

# サービスA
cd service-a
git checkout -b feature/add-endpoint
# ... 開発・デプロイ ...

# サービスB（並行開発）
cd service-b
git checkout -b feature/update-ui
# ... 開発・デプロイ ...
```

---

## 付録B: 実践的なGitコマンド集

### 日常業務で使うコマンド

```bash
# === ブランチ操作 ===

# 新規ブランチ作成・切り替え（1コマンド）
git checkout -b feature/new-feature

# ブランチ一覧（ローカル）
git branch

# ブランチ一覧（リモート含む）
git branch -a

# ブランチ削除（マージ済みのみ）
git branch -d feature/old-feature

# ブランチ強制削除
git branch -D feature/old-feature

# リモートブランチ削除
git push origin --delete feature/old-feature

# === コミット操作 ===

# ステージング + コミット（1コマンド）
git commit -am "feat: add feature"

# 直前のコミットメッセージ修正
git commit --amend

# 直前のコミットに追加（メッセージそのまま）
git add forgotten-file.ts
git commit --amend --no-edit

# === 履歴確認 ===

# コミット履歴（1行表示）
git log --oneline

# コミット履歴（グラフ表示）
git log --graph --oneline --all

# 特定ファイルの変更履歴
git log --follow -- src/App.tsx

# 誰がいつ変更したか
git blame src/App.tsx

# === 差分確認 ===

# ワーキングツリーとステージの差分
git diff

# ステージとHEADの差分
git diff --staged

# ブランチ間の差分
git diff main...feature/new-feature

# === 変更の取り消し ===

# ワーキングツリーの変更を破棄
git checkout -- src/App.tsx

# ステージングを取り消し
git reset HEAD src/App.tsx

# 直前のコミットを取り消し（コミットは履歴に残る）
git revert HEAD

# 直前のコミットを完全削除（危険）
git reset --hard HEAD~1

# === リモート操作 ===

# リモート追加
git remote add origin https://github.com/user/repo.git

# リモート一覧
git remote -v

# リモート変更取得（マージしない）
git fetch origin

# リモート変更取得 + マージ
git pull origin main

# プッシュ（初回）
git push -u origin feature/new-feature

# プッシュ（2回目以降）
git push

# 強制プッシュ（危険・チーム開発では避ける）
git push --force-with-lease

# === スタッシュ（一時退避） ===

# 現在の変更を退避
git stash

# 退避リスト確認
git stash list

# 退避した変更を復元
git stash pop

# 退避した変更を復元（stashは残す）
git stash apply

# 特定のstashを復元
git stash apply stash@{2}

# stash削除
git stash drop stash@{0}

# === タグ操作 ===

# タグ作成
git tag v1.0.0

# 注釈付きタグ
git tag -a v1.0.0 -m "Release version 1.0.0"

# タグ一覧
git tag

# タグをプッシュ
git push origin v1.0.0

# 全タグをプッシュ
git push origin --tags

# === 便利なエイリアス ===

# 設定
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual 'log --graph --oneline --all'

# 使用例
git co -b feature/new    # checkout -b の短縮
git br -a                # branch -a の短縮
git st                   # status の短縮
```

---

## 付録C: 高度なGitHub Flow活用

### 1. Feature Flags（フィーチャーフラグ）

```typescript
// feature-flags.ts
export const FEATURE_FLAGS = {
  NEW_UI: process.env.NEXT_PUBLIC_ENABLE_NEW_UI === 'true',
  BETA_FEATURE: process.env.NEXT_PUBLIC_BETA_FEATURE === 'true',
};

// App.tsx
import { FEATURE_FLAGS } from './feature-flags';

function App() {
  return (
    <>
      {FEATURE_FLAGS.NEW_UI ? <NewUI /> : <OldUI />}
      {FEATURE_FLAGS.BETA_FEATURE && <BetaFeature />}
    </>
  );
}
```

**メリット:**
- 未完成機能をmainにマージできる
- 本番環境で段階的にリリース
- A/Bテストが可能

### 2. Canary Deployment

```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  deploy-canary:
    runs-on: ubuntu-latest
    steps:
      # 5%のユーザーに新バージョンをデプロイ
      - run: ./deploy.sh canary --traffic=5%

      # 10分待機
      - run: sleep 600

      # エラー率チェック
      - run: |
          ERROR_RATE=$(./check-errors.sh)
          if [ $ERROR_RATE -gt 1 ]; then
            ./rollback.sh
            exit 1
          fi

      # 問題なければ全ユーザーへ
      - run: ./deploy.sh production --traffic=100%
```

### 3. Automated Rollback

```yaml
# .github/workflows/auto-rollback.yml
name: Auto Rollback
on:
  deployment_status:

jobs:
  check-deployment:
    if: github.event.deployment_status.state == 'success'
    runs-on: ubuntu-latest
    steps:
      # 5分後にヘルスチェック
      - run: sleep 300
      - run: |
          HEALTH=$(curl -s https://api.example.com/health)
          if [ "$HEALTH" != "OK" ]; then
            git revert HEAD
            git push origin main
          fi
```

### 4. Semantic Versioning自動化

```yaml
# .github/workflows/release.yml
name: Auto Release
on:
  push:
    branches: [main]

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      # コミットメッセージから自動バージョン決定
      - name: Semantic Release
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: npx semantic-release
```

**コミットメッセージとバージョン:**
- `fix:` → パッチバージョン (1.0.0 → 1.0.1)
- `feat:` → マイナーバージョン (1.0.0 → 1.1.0)
- `feat!:` or `BREAKING CHANGE:` → メジャーバージョン (1.0.0 → 2.0.0)

### 5. PR Size Limiter

```yaml
# .github/workflows/pr-size.yml
name: PR Size Check
on: [pull_request]

jobs:
  size:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Check PR size
        run: |
          FILES_CHANGED=$(git diff --name-only origin/main | wc -l)
          LINES_CHANGED=$(git diff --stat origin/main | tail -1 | awk '{print $4}')

          if [ $FILES_CHANGED -gt 20 ] || [ $LINES_CHANGED -gt 500 ]; then
            echo "⚠️ PRが大きすぎます。分割を検討してください。"
            echo "変更ファイル数: $FILES_CHANGED"
            echo "変更行数: $LINES_CHANGED"
            exit 1
          fi
```

---

## 付録D: チーム別カスタマイズ例

### スタートアップ（3-5人）

**特徴:**
- スピード最優先
- シンプルなプロセス
- 全員がフルスタック

**カスタマイズ:**
```markdown
## ブランチ命名
feature/short-description（チケットID不要）

## PRレビュー
最低1人承認（誰でもOK）

## デプロイ
mainマージ = 即座に自動デプロイ

## 例外ルール
緊急時はmainへ直接コミットOK（事後報告）
```

### 中規模チーム（10-20人）

**特徴:**
- フロント・バックエンド分業
- 品質重視
- 定期的なリリース

**カスタマイズ:**
```markdown
## ブランチ命名
<type>/<JIRA-123>-description

## PRレビュー
担当領域の専門家1人 + 任意1人

## デプロイ
ステージング環境で検証 → 手動で本番デプロイ

## CI/CD
全テスト + E2Eテスト必須
```

### エンタープライズ（50人以上）

**特徴:**
- 複数チーム並行開発
- 厳格な品質基準
- コンプライアンス要件

**カスタマイズ:**
```markdown
## ブランチ命名
<team>/<type>/<TICKET-123>-description
例: frontend/feature/USER-123-add-login

## PRレビュー
同チーム2人 + セキュリティレビュー（必要時）

## デプロイ
ステージング → QAチェック → 承認 → 本番

## 監査
全PRに承認者記録
デプロイログ保存（180日）
```

---

## 付録E: よくある質問（FAQ）

### Q1: PRのサイズはどのくらいが適切？

**A:**
- **理想:** 200-400行
- **許容:** 500行まで
- **大きすぎ:** 1000行以上

**理由:**
- レビューしやすい（30分以内）
- バグ発見率が高い
- マージ後の問題切り分けが容易

### Q2: レビューにどのくらい時間をかけるべき？

**A:**
- **小PR（<200行）:** 15-30分
- **中PR（200-500行）:** 30-60分
- **大PR（>500行）:** 分割推奨

### Q3: mainブランチが壊れたらどうする？

**A:**
```bash
# 即座にrevert
git revert HEAD
git push origin main

# または直前の安定コミットにreset
git reset --hard <safe-commit>
git push --force-with-lease origin main

# 問題修正後に再度PR
```

### Q4: 長期featureブランチはNG？

**A:**
**原則NG。** ただし以下の戦略で対応:

1. **Feature Flags使用:**
   ```typescript
   if (FEATURE_FLAGS.NEW_FEATURE) {
     // 新機能（未完成でもOK）
   }
   ```

2. **段階的マージ:**
   ```bash
   # 週1回mainへマージ
   git checkout feature/long-term
   git merge main
   # テスト
   git checkout main
   git merge feature/long-term
   ```

3. **小分割:**
   ```bash
   feature/big-feature-part1 → main
   feature/big-feature-part2 → main
   feature/big-feature-part3 → main
   ```

### Q5: hotfixブランチは必要？

**A:**
GitHub Flowでは**不要**。

```bash
# mainから直接分岐
git checkout main
git checkout -b hotfix/fix-critical-bug

# 通常のPRフロー（ただし緊急レビュー）
# → マージ → デプロイ
```

### Q6: 複数人で同じブランチ作業は可能？

**A:**
**可能だが推奨しない。** 代替案:

```bash
# 方法1: サブブランチ
git checkout -b feature/main-feature
# Aさん作業
git checkout -b feature/main-feature-part1

# 方法2: PR間依存
feature/base → main (Draft PR)
feature/dependent → feature/base (PR)
```

### Q7: コミットは1機能1コミット？

**A:**
**Yes。** ただしPRマージ時は「Squash and merge」でまとめてOK。

```bash
# 開発中（細かくコミット）
git commit -m "feat: add button UI"
git commit -m "feat: add button logic"
git commit -m "test: add button tests"

# マージ時（1コミットに）
Squash and merge: "feat: add button feature"
```

---

## 付録F: GitHub Actions実践例

### 例1: 自動ラベル付与

```yaml
# .github/workflows/auto-label.yml
name: Auto Label
on: [pull_request]

jobs:
  label:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/labeler@v4
        with:
          repo-token: ${{ secrets.GITHUB_TOKEN }}
          configuration-path: .github/labeler.yml
```

```yaml
# .github/labeler.yml
'documentation':
  - '**/*.md'

'frontend':
  - 'src/components/**'
  - 'src/pages/**'

'backend':
  - 'src/api/**'
  - 'src/db/**'

'tests':
  - '**/*.test.ts'
  - '**/*.spec.ts'
```

### 例2: PR自動コメント

```yaml
# .github/workflows/pr-comment.yml
name: PR Comment
on:
  pull_request:
    types: [opened]

jobs:
  comment:
    runs-on: ubuntu-latest
    steps:
      - name: Comment
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '👋 PRありがとうございます！レビューまでしばらくお待ちください。'
            })
```

### 例3: デプロイ環境プレビュー

```yaml
# .github/workflows/preview.yml
name: Deploy Preview
on: [pull_request]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Deploy to Vercel
        run: |
          PREVIEW_URL=$(vercel deploy --token=${{ secrets.VERCEL_TOKEN }})
          echo "preview_url=$PREVIEW_URL" >> $GITHUB_OUTPUT

      - name: Comment preview URL
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '🚀 Preview: ${{ steps.deploy.outputs.preview_url }}'
            })
```

---

**このガイドの文字数:** 約30,000文字

**習得目安時間:**
- 初心者: 2-3週間の実践
- 経験者: 1週間の実践

**次に読むべきガイド:**
1. [PRレビューガイド](./06-pr-review.md)
2. [コンフリクト解決ガイド](./07-conflict-resolution.md)

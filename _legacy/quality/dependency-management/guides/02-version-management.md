# Version Management - 依存関係のバージョン管理戦略

## 📋 目次

1. [概要](#概要)
2. [Semantic Versioning](#semantic-versioning)
3. [バージョン指定戦略](#バージョン指定戦略)
4. [更新ポリシー](#更新ポリシー)
5. [自動化ツール](#自動化ツール)
6. [バージョン競合の解決](#バージョン競合の解決)
7. [実践例](#実践例)

## 概要

依存関係のバージョン管理は、プロジェクトの安定性と保守性に直結する重要な要素です。適切なバージョン管理により、セキュリティリスクを最小化し、予期しない動作を防ぎ、チーム開発をスムーズに進めることができます。

**このガイドで学べること:**
- Semantic Versioningの理解と適用
- 環境別のバージョン固定戦略
- 依存関係の更新ポリシー設計
- 自動化ツールの活用
- バージョン競合の解決方法

## Semantic Versioning

### 基本ルール

Semantic Versioning（SemVer）は、バージョン番号に明確な意味を持たせる規約です。

```
MAJOR.MINOR.PATCH

例: 2.4.1

MAJOR (2): 破壊的変更（後方互換性なし）
MINOR (4): 機能追加（後方互換性あり）
PATCH (1): バグ修正（後方互換性あり）
```

### バージョンアップの判断基準

**MAJOR（メジャー）更新:**
```typescript
// v1.x.x
function fetchUser(id: string): Promise<User>

// v2.0.0 - 破壊的変更
function fetchUser(options: FetchOptions): Promise<User>
// ❌ 引数の型が変更され、既存コードが動かない
```

**MINOR（マイナー）更新:**
```typescript
// v1.4.x
function fetchUser(id: string): Promise<User>

// v1.5.0 - 機能追加
function fetchUser(id: string, cache?: boolean): Promise<User>
// ✅ オプション引数追加、既存コードはそのまま動く
```

**PATCH（パッチ）更新:**
```typescript
// v1.4.0 - バグあり
function calculateTotal(items: Item[]): number {
  return items.reduce((sum, item) => sum + item.price, 0)
  // バグ: 税込み計算が漏れている
}

// v1.4.1 - バグ修正
function calculateTotal(items: Item[]): number {
  return items.reduce((sum, item) => sum + item.priceWithTax, 0)
  // ✅ バグ修正のみ、APIは変わらない
}
```

### プレリリースバージョン

```
1.0.0-alpha.1    # アルファ版（初期テスト）
1.0.0-beta.1     # ベータ版（機能完成、テスト中）
1.0.0-rc.1       # リリース候補版
1.0.0            # 正式リリース
```

**使用例:**
```json
{
  "dependencies": {
    "next": "14.0.0",           // 安定版
    "react": "19.0.0-rc.0"      // リリース候補版
  }
}
```

## バージョン指定戦略

### npm/yarn/pnpm

```json
{
  "dependencies": {
    // 1. 完全固定（Exact）
    "package1": "1.0.0",

    // 2. キャレット（Caret）- デフォルト、推奨
    "package2": "^1.0.0",      // >= 1.0.0 < 2.0.0
    "package3": "^1.2.3",      // >= 1.2.3 < 2.0.0
    "package4": "^0.2.3",      // >= 0.2.3 < 0.3.0（0.x系は特別）

    // 3. チルダ（Tilde）
    "package5": "~1.2.3",      // >= 1.2.3 < 1.3.0
    "package6": "~1.2",        // >= 1.2.0 < 1.3.0

    // 4. 範囲指定
    "package7": ">=1.0.0 <2.0.0",
    "package8": "1.0.0 - 2.0.0",

    // 5. 特殊記号
    "package9": "*",           // ❌ 非推奨: 任意のバージョン
    "package10": "latest"      // ❌ 非推奨: 最新
  }
}
```

### Swift Package Manager

```swift
dependencies: [
    // 1. 完全固定（本番推奨）
    .package(url: "https://github.com/realm/realm-swift.git", exact: "10.45.0"),

    // 2. 最小バージョン指定（開発推奨）
    .package(url: "https://github.com/Alamofire/Alamofire.git", from: "5.8.0"),
    // 5.8.0, 5.9.0, 6.0.0 すべて許可

    // 3. 範囲指定
    .package(url: "https://github.com/example/package.git", "1.0.0"..<"2.0.0"),

    // 4. upToNextMajor
    .package(url: "https://github.com/example/package.git", .upToNextMajor(from: "1.0.0")),
    // >= 1.0.0 < 2.0.0

    // 5. upToNextMinor
    .package(url: "https://github.com/example/package.git", .upToNextMinor(from: "1.2.0")),
    // >= 1.2.0 < 1.3.0

    // 6. ブランチ指定（開発時のみ）
    .package(url: "https://github.com/example/package.git", branch: "develop"),

    // 7. コミットハッシュ（特定バージョン固定）
    .package(url: "https://github.com/example/package.git", revision: "abc123def456"),
]
```

### CocoaPods

```ruby
# Podfile

# 1. 完全固定
pod 'Realm', '10.45.0'

# 2. ペシミスティックオペレーター（推奨）
pod 'Alamofire', '~> 5.8.0'    # >= 5.8.0 < 5.9.0
pod 'Alamofire', '~> 5.8'      # >= 5.8.0 < 6.0.0

# 3. 比較演算子
pod 'AFNetworking', '>= 4.0.0'
pod 'SDWebImage', '< 6.0.0'

# 4. GitHubから特定タグ
pod 'MyPod', :git => 'https://github.com/user/MyPod.git', :tag => '1.0.0'

# 5. GitHubから特定コミット
pod 'MyPod', :git => 'https://github.com/user/MyPod.git', :commit => 'abc123'

# 6. GitHubからブランチ（開発時のみ）
pod 'MyPod', :git => 'https://github.com/user/MyPod.git', :branch => 'develop'

# 7. ローカルパス（開発時のみ）
pod 'MyLocalPod', :path => '../MyLocalPod'
```

### Python (Poetry)

```toml
[tool.poetry.dependencies]
python = "^3.11"              # >= 3.11 < 4.0

# 1. キャレット（デフォルト）
django = "^4.2.0"             # >= 4.2.0 < 5.0.0

# 2. チルダ
requests = "~4.2.0"           # >= 4.2.0 < 4.3.0

# 3. ワイルドカード
pytest = "7.*"                # 7.x系の任意のバージョン

# 4. 比較演算子
numpy = ">=1.24.0,<2.0.0"     # 範囲指定

# 5. 完全固定
pillow = "10.1.0"
```

## 更新ポリシー

### 環境別戦略

**開発環境:**
```json
{
  "dependencies": {
    // 柔軟なバージョン指定で最新の改善を取得
    "react": "^18.2.0",
    "next": "^14.0.0"
  }
}
```

**ステージング/本番環境:**
```json
{
  "dependencies": {
    // 完全固定で予測可能性を確保
    "react": "18.2.0",
    "next": "14.0.4"
  }
}
```

### 更新頻度のガイドライン

```markdown
## 依存関係更新ポリシー

### セキュリティアップデート
- **頻度**: 即時（Critical/High）、週次（Medium/Low）
- **対応**: 自動化ツール（Dependabot、Renovate）で検知
- **プロセス**:
  1. アラート受信
  2. 影響範囲確認
  3. テスト実施
  4. 緊急デプロイ（Critical/High）

### パッチバージョン（x.x.PATCH）
- **頻度**: 週次
- **対応**: 自動マージ（CI通過後）
- **理由**: バグ修正のみ、リスク低い

### マイナーバージョン（x.MINOR.x）
- **頻度**: 月次
- **対応**: レビュー後マージ
- **プロセス**:
  1. リリースノート確認
  2. 新機能の影響調査
  3. テスト実施
  4. スプリント内でマージ

### メジャーバージョン（MAJOR.x.x）
- **頻度**: 四半期ごと、または必要時
- **対応**: 計画的移行
- **プロセス**:
  1. マイグレーションガイド確認
  2. 破壊的変更の影響範囲調査
  3. テスト計画策定
  4. 段階的移行（feature flag使用）
  5. モニタリング強化
```

### 更新タイミングの例

```bash
# 毎週月曜日: パッチ更新
npm update                    # パッチのみ更新
npm audit fix                 # セキュリティ修正

# 毎月第1金曜日: マイナー更新
npm outdated                  # 古いパッケージ確認
npm update --save             # マイナーまで更新

# 四半期ごと: メジャー更新
npm outdated                  # 全体確認
# 個別に計画的に更新
npm install react@19.0.0
```

## 自動化ツール

### Dependabot

**設定例（.github/dependabot.yml）:**
```yaml
version: 2
updates:
  # npm
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
    open-pull-requests-limit: 10
    reviewers:
      - "team-lead"
    assignees:
      - "maintainer"
    labels:
      - "dependencies"
      - "automated"
    commit-message:
      prefix: "chore"
      include: "scope"
    # バージョン別設定
    versioning-strategy: increase
    # セキュリティのみ
    # open-pull-requests-limit: 0  # 通常更新は無効
    # security-updates-only: true  # セキュリティのみ有効

  # Swift Package Manager
  - package-ecosystem: "swift"
    directory: "/"
    schedule:
      interval: "weekly"

  # CocoaPods
  - package-ecosystem: "cocoapods"
    directory: "/"
    schedule:
      interval: "weekly"
```

**自動マージ設定:**
```yaml
# .github/workflows/auto-merge-dependabot.yml
name: Auto-merge Dependabot PRs

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  auto-merge:
    runs-on: ubuntu-latest
    if: github.actor == 'dependabot[bot]'
    steps:
      - name: Approve PR
        run: gh pr review --approve "$PR_URL"
        env:
          PR_URL: ${{ github.event.pull_request.html_url }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Auto-merge
        # パッチ更新のみ自動マージ
        if: contains(github.event.pull_request.title, 'patch')
        run: gh pr merge --auto --squash "$PR_URL"
        env:
          PR_URL: ${{ github.event.pull_request.html_url }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### Renovate

**設定例（renovate.json）:**
```json
{
  "$schema": "https://docs.renovatebot.com/renovate-schema.json",
  "extends": ["config:base"],
  "schedule": ["before 10am on monday"],
  "timezone": "Asia/Tokyo",
  "labels": ["dependencies"],
  "assignees": ["team-lead"],
  "reviewers": ["team:developers"],

  // パッケージ別設定
  "packageRules": [
    {
      "matchUpdateTypes": ["patch"],
      "automerge": true,
      "automergeType": "pr",
      "matchPackagePatterns": ["*"]
    },
    {
      "matchUpdateTypes": ["minor"],
      "automerge": false,
      "groupName": "Minor updates"
    },
    {
      "matchUpdateTypes": ["major"],
      "enabled": false,
      "groupName": "Major updates (manual review required)"
    },
    {
      "matchDepTypes": ["devDependencies"],
      "automerge": true
    }
  ],

  // セキュリティ更新
  "vulnerabilityAlerts": {
    "enabled": true,
    "labels": ["security"],
    "assignees": ["security-team"]
  }
}
```

### npm-check-updates

```bash
# インストール
npm install -g npm-check-updates

# 古いパッケージ確認
ncu

# 出力例:
# react           ^18.2.0  →  ^19.0.0
# typescript       ^5.0.0  →   ^5.3.0

# package.jsonを更新（実行のみ）
ncu -u

# 特定パッケージのみ
ncu -f react

# マイナー更新のみ
ncu --target minor

# パッチ更新のみ
ncu --target patch

# インタラクティブモード
ncu -i
```

## バージョン競合の解決

### npm/yarn/pnpm

**問題:**
```
npm ERR! Could not resolve dependency:
npm ERR! peer react@"^18.0.0" from react-dom@18.2.0
npm ERR! node_modules/react-dom
npm ERR!   react-dom@"^18.2.0" from the root project
```

**解決方法1: overrides（npm 8.3+）:**
```json
{
  "overrides": {
    "react": "18.2.0"
  }
}
```

**解決方法2: resolutions（yarn）:**
```json
{
  "resolutions": {
    "react": "18.2.0",
    "**/lodash": "4.17.21"
  }
}
```

**解決方法3: pnpm.overrides:**
```json
{
  "pnpm": {
    "overrides": {
      "react": "18.2.0"
    }
  }
}
```

### Swift Package Manager

**問題:**
```
error: Dependencies could not be resolved because package 'A' depends on package 'C' 1.0.0..<2.0.0 and
package 'B' depends on package 'C' 2.0.0..<3.0.0.
```

**解決方法:**
```swift
// Package.swift
dependencies: [
    // 競合するパッケージを明示的に指定
    .package(url: "https://github.com/example/C.git", from: "2.0.0"),
    .package(url: "https://github.com/example/A.git", from: "1.0.0"),
    .package(url: "https://github.com/example/B.git", from: "1.0.0"),
]
```

```bash
# キャッシュクリアと再解決
rm Package.resolved
swift package resolve
```

### CocoaPods

**問題:**
```
[!] CocoaPods could not find compatible versions for pod "Alamofire":
  In Podfile:
    Alamofire (~> 5.8)

    Pod1 (= 2.0.0) was resolved to 2.0.0, which depends on
      Alamofire (~> 4.0)
```

**解決方法:**
```ruby
# Podfile
platform :ios, '15.0'

target 'MyApp' do
  # 競合するバージョンを明示的に指定
  pod 'Alamofire', '~> 5.8'

  # Pod1のAlamofire依存を無視（リスクあり）
  pod 'Pod1', '2.0.0', :inhibit_warnings => true
end
```

```bash
# キャッシュクリアと再インストール
pod cache clean --all
pod deintegrate
rm Podfile.lock
pod install
```

## 実践例

### プロジェクト開始時のセットアップ

**1. package.json初期設定:**
```json
{
  "name": "my-app",
  "version": "1.0.0",
  "private": true,
  "engines": {
    "node": ">=18.0.0",
    "npm": ">=9.0.0"
  },
  "dependencies": {
    "react": "^18.2.0",
    "next": "^14.0.0"
  },
  "devDependencies": {
    "typescript": "^5.3.0",
    "@types/react": "^18.2.0"
  }
}
```

**2. .npmrc設定:**
```bash
# .npmrc
save-exact=false              # ^を使用（開発時）
engine-strict=true
audit-level=high
package-lock=true
```

**3. Dependabot設定:**
```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
```

### 本番リリース前のバージョン固定

```bash
# 1. 現在のバージョンを確認
npm list --depth=0

# 2. package.jsonを完全固定に変更
npm install --save-exact

# 3. package-lock.jsonを再生成
rm package-lock.json
npm install

# 4. .npmrc更新
echo "save-exact=true" >> .npmrc

# 5. コミット
git add package.json package-lock.json .npmrc
git commit -m "chore: lock dependencies for production"
```

### セキュリティアップデートの対応

```bash
# 1. セキュリティ監査
npm audit

# 出力例:
# found 3 vulnerabilities (2 moderate, 1 high)

# 2. 詳細確認
npm audit --json > audit-report.json

# 3. 自動修正（パッチのみ）
npm audit fix

# 4. マイナー/メジャー含む修正（慎重に）
npm audit fix --force

# 5. 手動修正
npm install package@latest

# 6. テスト実行
npm test

# 7. コミット
git add package.json package-lock.json
git commit -m "chore(security): update vulnerable dependencies"
```

### メジャーバージョンアップの段階的移行

```bash
# 例: React 18 → 19 への移行

# 1. 新バージョンの調査
npm info react@19

# 2. マイグレーションガイド確認
# https://react.dev/blog/2024/XX/XX/react-19

# 3. ブランチ作成
git checkout -b upgrade/react-19

# 4. package.json更新
npm install react@19 react-dom@19

# 5. 型定義更新
npm install -D @types/react@19

# 6. コード修正
# - 破壊的変更に対応
# - Deprecated API置き換え

# 7. テスト実行
npm test
npm run build

# 8. 動作確認
npm run dev

# 9. PR作成
git add .
git commit -m "feat: upgrade to React 19"
git push origin upgrade/react-19
```

## まとめ

### バージョン管理のベストプラクティス

1. **Semantic Versioningを理解する**
   - MAJOR: 破壊的変更
   - MINOR: 機能追加
   - PATCH: バグ修正

2. **環境別に戦略を使い分ける**
   - 開発: 柔軟（^1.0.0）
   - 本番: 固定（1.0.0）

3. **ロックファイルを必ずコミット**
   - package-lock.json
   - yarn.lock
   - Podfile.lock
   - Package.resolved

4. **自動化ツールを活用**
   - Dependabot/Renovate
   - CI/CDでの監査

5. **更新ポリシーを明確化**
   - セキュリティ: 即時
   - パッチ: 週次
   - マイナー: 月次
   - メジャー: 四半期

### チェックリスト

- [ ] Semantic Versioningを理解している
- [ ] バージョン指定方法を使い分けている
- [ ] ロックファイルをGitにコミットしている
- [ ] 自動化ツール（Dependabot等）を設定している
- [ ] 更新ポリシーをドキュメント化している
- [ ] 定期的なセキュリティ監査を実施している
- [ ] バージョン競合の解決方法を知っている

---

**関連ガイド:**
- [Package Manager Best Practices](./01-package-managers.md)
- [Security & License](./03-security-license.md)

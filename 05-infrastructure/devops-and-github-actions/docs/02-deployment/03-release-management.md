# リリース管理

> セマンティックバージョニング、CHANGELOG の自動生成、リリースプロセスの自動化により、予測可能で安全なリリースサイクルを確立する

## この章で学ぶこと

1. **セマンティックバージョニングの原則と実践** — MAJOR.MINOR.PATCH の判断基準と自動バージョニング
2. **CHANGELOG とリリースノートの自動生成** — Conventional Commits から自動的にドキュメントを生成する仕組み
3. **リリースプロセスの自動化** — GitHub Releases、npm publish、タグ管理の CI/CD パイプライン
4. **Feature Flag によるリリース制御** — 機能のデプロイとリリースを分離し、安全なロールアウトを実現する
5. **モノレポにおけるリリース管理** — Changesets を活用したパッケージ間の依存関係を考慮したバージョン管理

---

## 1. リリース管理の全体像

```
┌────────────────────────────────────────────────────────┐
│              リリース管理パイプライン                      │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Conventional    セマンティック     CHANGELOG    GitHub   │
│  Commits         バージョニング     自動生成     Release  │
│  ┌──────┐       ┌──────────┐     ┌────────┐  ┌──────┐│
│  │feat: │──────►│ v1.2.0   │────►│ 変更   │─►│ Tag  ││
│  │fix:  │       │ → v1.3.0 │     │ 履歴   │  │ +    ││
│  │BREAK│       │ or v2.0.0│     │ 生成   │  │ Note ││
│  └──────┘       └──────────┘     └────────┘  └──────┘│
│                                                        │
│  ┌──────────────────────────────────────────────┐     │
│  │ ツールチェーン                                 │     │
│  │ commitlint → semantic-release → GitHub CLI   │     │
│  │ or                                            │     │
│  │ commitlint → changesets → npm publish         │     │
│  └──────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────┘
```

### リリース管理の基本原則

リリース管理で最も重要なのは**予測可能性**と**再現性**である。以下の原則を基礎に設計する。

1. **バージョンの一意性**: 同じバージョン番号は一度しか使わない。リリースされたバージョンの内容を変更しない
2. **自動化の徹底**: バージョン決定、CHANGELOG 生成、タグ作成、公開を全自動化する
3. **トレーサビリティ**: どのコミットがどのバージョンに含まれるかを常に追跡可能にする
4. **ロールバック可能性**: 問題発生時に前のバージョンへ即座に戻せる仕組みを維持する

```
リリースプロセスの成熟度モデル:

  Level 0: 完全手動
  ┌──────────────────────────────────────────────┐
  │ - package.json を手動で変更                     │
  │ - CHANGELOG を手動で記述                        │
  │ - Git タグを手動で作成                          │
  │ - デプロイを手動で実行                          │
  │ リスク: 高い (人為的ミス多発)                    │
  └──────────────────────────────────────────────┘

  Level 1: 半自動
  ┌──────────────────────────────────────────────┐
  │ - Conventional Commits で変更種別を明示         │
  │ - CHANGELOG は自動生成                         │
  │ - Git タグは手動で作成                          │
  │ - デプロイスクリプトで自動化                     │
  │ リスク: 中 (タグ付け忘れ、バージョン不整合)      │
  └──────────────────────────────────────────────┘

  Level 2: 完全自動 (目指すべき姿)
  ┌──────────────────────────────────────────────┐
  │ - Conventional Commits で変更種別を明示         │
  │ - semantic-release でバージョン自動決定         │
  │ - CHANGELOG, タグ, GitHub Release を自動生成   │
  │ - CI/CD でデプロイまで自動化                    │
  │ リスク: 低い (プロセスが標準化・自動化)          │
  └──────────────────────────────────────────────┘
```

---

## 2. セマンティックバージョニング (SemVer)

```
セマンティックバージョニングの構造:

  v 2 . 1 . 3 - beta.1 + build.456
    │   │   │     │          │
    │   │   │     │          └── ビルドメタデータ (比較に使わない)
    │   │   │     └───────────── プレリリース識別子
    │   │   └──────────────────── PATCH: 後方互換性のあるバグ修正
    │   └─────────────────────── MINOR: 後方互換性のある機能追加
    └──────────────────────────── MAJOR: 破壊的変更

バージョン判定フローチャート:

  変更内容は？
     │
     ├─ API の削除/変更 (破壊的) ──► MAJOR++  (1.2.3 → 2.0.0)
     │
     ├─ 新機能追加 (後方互換) ────► MINOR++  (1.2.3 → 1.3.0)
     │
     └─ バグ修正 (後方互換) ─────► PATCH++  (1.2.3 → 1.2.4)
```

### 2.1 SemVer の詳細ルール

| ルール | 説明 | 例 |
|--------|------|-----|
| 初期開発 | 0.x.y は不安定 API を示す。いつでも破壊的変更が可能 | 0.1.0, 0.2.0 |
| MAJOR 0 | 0.y.z では MINOR が破壊的変更、PATCH がバグ修正 | 0.1.0 → 0.2.0 |
| PATCH | バグ修正のみ。内部実装の変更で API 契約は不変 | 1.2.3 → 1.2.4 |
| MINOR | 後方互換の新機能。既存機能の非推奨化もここ | 1.2.3 → 1.3.0 |
| MAJOR | 後方互換性を壊す変更。MINOR と PATCH は 0 にリセット | 1.2.3 → 2.0.0 |
| プレリリース | ハイフン区切りで識別子を付与。正式版より優先度が低い | 2.0.0-alpha.1 |
| ビルドメタデータ | プラス記号で付与。バージョン比較では無視される | 2.0.0+build.123 |

### 2.2 破壊的変更の判断基準

```typescript
// 破壊的変更の具体例

// ケース 1: 関数シグネチャの変更 → MAJOR
// Before:
function getUser(id: string): User { ... }
// After:
function getUser(id: string, options: GetUserOptions): User { ... }
// 既存の呼び出しコードが壊れる → MAJOR

// ケース 2: 任意パラメータの追加 → MINOR
// Before:
function getUser(id: string): User { ... }
// After:
function getUser(id: string, options?: GetUserOptions): User { ... }
// 既存の呼び出しコードは影響なし → MINOR

// ケース 3: レスポンス形式の変更 → MAJOR
// Before:
// { "name": "Alice", "email": "alice@example.com" }
// After:
// { "data": { "name": "Alice", "email": "alice@example.com" } }
// 既存のパーサーが壊れる → MAJOR

// ケース 4: レスポンスへのフィールド追加 → MINOR
// Before:
// { "name": "Alice", "email": "alice@example.com" }
// After:
// { "name": "Alice", "email": "alice@example.com", "avatar": "url" }
// 既存のパーサーは影響なし (未知のフィールドを無視する前提) → MINOR

// ケース 5: デフォルト値の変更 → ケースバイケース
// 動作が変わるがシグネチャは同じ → 通常 MAJOR（ユーザーの期待を裏切る場合）
```

---

## 3. Conventional Commits

### 3.1 基本フォーマット

```bash
# Conventional Commits のフォーマット
# <type>[optional scope]: <description>
#
# [optional body]
#
# [optional footer(s)]

# 例: 新機能
git commit -m "feat(auth): ソーシャルログインを追加

Google と GitHub による OAuth2 認証を実装。
既存のメール/パスワード認証に影響なし。

Closes #123"

# 例: バグ修正
git commit -m "fix(api): ページネーションのオフセット計算を修正

0始まりのインデックスが1始まりとして計算されていた問題を修正。

Fixes #456"

# 例: 破壊的変更
git commit -m "feat(api)!: レスポンス形式をJSON:API仕様に変更

BREAKING CHANGE: API レスポンスの構造が変更されました。
data プロパティでラップされるようになります。
移行ガイド: docs/migration-v3.md"
```

### 3.2 コミットタイプの一覧と使い分け

| タイプ | バージョン影響 | 説明 | 使用例 |
|--------|--------------|------|--------|
| `feat` | MINOR | 新機能の追加 | 新しい API エンドポイント、UI コンポーネント |
| `fix` | PATCH | バグ修正 | 計算ロジックの修正、クラッシュの修正 |
| `docs` | なし | ドキュメントのみの変更 | README 更新、JSDoc 追加 |
| `style` | なし | コードの意味に影響しないスタイル変更 | フォーマット、セミコロン追加 |
| `refactor` | なし | バグ修正でも新機能でもないコード変更 | 内部構造の改善、変数名変更 |
| `perf` | PATCH | パフォーマンス改善 | クエリ最適化、キャッシュ導入 |
| `test` | なし | テストの追加・修正 | ユニットテスト、E2E テスト |
| `build` | なし | ビルドシステムや外部依存の変更 | webpack 設定、npm パッケージ更新 |
| `ci` | なし | CI 設定の変更 | GitHub Actions、CircleCI |
| `chore` | なし | その他の変更 | .gitignore、設定ファイル |
| `revert` | 元の変更に依存 | コミットの取り消し | `revert: feat(auth): ...` |

### 3.3 commitlint の設定

```json
// commitlint.config.js に相当する設定
// package.json 内
{
  "commitlint": {
    "extends": ["@commitlint/config-conventional"],
    "rules": {
      "type-enum": [2, "always", [
        "feat", "fix", "docs", "style", "refactor",
        "perf", "test", "build", "ci", "chore", "revert"
      ]],
      "subject-max-length": [2, "always", 72],
      "body-max-line-length": [2, "always", 100]
    }
  }
}
```

```yaml
# .github/workflows/commitlint.yml — PR のコミットメッセージを検証
name: Commitlint

on:
  pull_request:
    branches: [main]

jobs:
  commitlint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-node@v4
        with:
          node-version: 20

      - name: Install commitlint
        run: npm install --save-dev @commitlint/cli @commitlint/config-conventional

      - name: Validate PR commits
        run: npx commitlint --from ${{ github.event.pull_request.base.sha }} --to ${{ github.event.pull_request.head.sha }} --verbose
```

### 3.4 Husky + commitlint でローカル検証

```json
// package.json
{
  "scripts": {
    "prepare": "husky"
  },
  "devDependencies": {
    "@commitlint/cli": "^19.0.0",
    "@commitlint/config-conventional": "^19.0.0",
    "husky": "^9.0.0"
  }
}
```

```bash
# .husky/commit-msg
npx --no -- commitlint --edit ${1}
```

```bash
# コミットメッセージのインタラクティブ作成 (commitizen)
# インストール
npm install --save-dev commitizen cz-conventional-changelog

# package.json に追加
# "config": {
#   "commitizen": {
#     "path": "cz-conventional-changelog"
#   }
# }

# 使用
npx cz
# ? Select the type of change: feat
# ? What is the scope: auth
# ? Short description: ソーシャルログインを追加
# ? Longer description: Google と GitHub による OAuth2 認証を実装
# ? Breaking changes? No
# ? Issues closed: #123
```

---

## 4. semantic-release による自動リリース

### 4.1 基本ワークフロー

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    branches: [main]

permissions:
  contents: write
  issues: write
  pull-requests: write
  packages: write

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          persist-credentials: false

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - run: npm ci

      - name: Run tests
        run: npm test

      - name: Semantic Release
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          NPM_TOKEN: ${{ secrets.NPM_TOKEN }}
        run: npx semantic-release
```

### 4.2 semantic-release 設定

```json
// .releaserc.json — semantic-release 設定
{
  "branches": [
    "main",
    { "name": "next", "prerelease": true },
    { "name": "beta", "prerelease": true }
  ],
  "plugins": [
    "@semantic-release/commit-analyzer",
    "@semantic-release/release-notes-generator",
    [
      "@semantic-release/changelog",
      { "changelogFile": "CHANGELOG.md" }
    ],
    [
      "@semantic-release/npm",
      { "npmPublish": true }
    ],
    [
      "@semantic-release/github",
      {
        "assets": [
          { "path": "dist/**", "label": "Distribution" }
        ]
      }
    ],
    [
      "@semantic-release/git",
      {
        "assets": ["CHANGELOG.md", "package.json"],
        "message": "chore(release): ${nextRelease.version} [skip ci]\n\n${nextRelease.notes}"
      }
    ]
  ]
}
```

### 4.3 カスタムリリースルールの設定

```json
// .releaserc.json — カスタム commit-analyzer 設定
{
  "plugins": [
    [
      "@semantic-release/commit-analyzer",
      {
        "preset": "conventionalcommits",
        "releaseRules": [
          { "type": "feat", "release": "minor" },
          { "type": "fix", "release": "patch" },
          { "type": "perf", "release": "patch" },
          { "type": "refactor", "release": "patch" },
          { "type": "docs", "scope": "api", "release": "patch" },
          { "type": "build", "scope": "deps", "release": "patch" },
          { "breaking": true, "release": "major" }
        ],
        "parserOpts": {
          "noteKeywords": ["BREAKING CHANGE", "BREAKING CHANGES", "BREAKING"]
        }
      }
    ],
    [
      "@semantic-release/release-notes-generator",
      {
        "preset": "conventionalcommits",
        "presetConfig": {
          "types": [
            { "type": "feat", "section": "Features" },
            { "type": "fix", "section": "Bug Fixes" },
            { "type": "perf", "section": "Performance" },
            { "type": "refactor", "section": "Refactoring", "hidden": false },
            { "type": "docs", "section": "Documentation", "hidden": false },
            { "type": "chore", "hidden": true },
            { "type": "style", "hidden": true },
            { "type": "test", "hidden": true }
          ]
        }
      }
    ]
  ]
}
```

### 4.4 Docker イメージのリリース連携

```yaml
# .github/workflows/release-with-docker.yml
name: Release with Docker

on:
  push:
    branches: [main]

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      packages: write

    outputs:
      new-release-published: ${{ steps.semantic.outputs.new_release_published }}
      new-release-version: ${{ steps.semantic.outputs.new_release_version }}

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          persist-credentials: false

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - run: npm ci && npm test

      - name: Semantic Release
        id: semantic
        uses: cycjimmy/semantic-release-action@v4
        with:
          semantic_version: 23
          extra_plugins: |
            @semantic-release/changelog
            @semantic-release/git
            conventional-changelog-conventionalcommits
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  docker:
    needs: release
    if: needs.release.outputs.new-release-published == 'true'
    runs-on: ubuntu-latest
    permissions:
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and Push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:v${{ needs.release.outputs.new-release-version }}
            ghcr.io/${{ github.repository }}:latest
```

---

## 5. Changesets によるモノレポ対応リリース

### 5.1 基本設定

```json
// .changeset/config.json
{
  "$schema": "https://unpkg.com/@changesets/config/schema.json",
  "changelog": "@changesets/cli/changelog",
  "commit": false,
  "fixed": [],
  "linked": [["@myorg/core", "@myorg/utils"]],
  "access": "public",
  "baseBranch": "main",
  "updateInternalDependencies": "patch"
}
```

### 5.2 Changeset ファイルの例

```markdown
<!-- .changeset/happy-dogs-dance.md (自動生成されるテンプレート) -->
---
"@myorg/core": minor
"@myorg/utils": patch
---

ユーザープロフィール API にアバター画像アップロード機能を追加。
utils パッケージに画像リサイズヘルパーを追加。
```

### 5.3 Changesets ワークフロー

```
Changesets のワークフロー:

  開発者                 CI                     npm
    │                    │                       │
    │── npx changeset    │                       │
    │   (変更内容を記述)  │                       │
    │                    │                       │
    │── git push ──────► │                       │
    │                    │── Changeset Bot       │
    │                    │   PR 作成             │
    │                    │   (Version Packages)  │
    │                    │                       │
    │── PR マージ ──────►│                       │
    │                    │── changeset version   │
    │                    │   (バージョン更新)     │
    │                    │── changeset publish ──►│
    │                    │   (npm publish)        │
    │                    │── GitHub Release 作成  │
```

### 5.4 Changesets の GitHub Actions ワークフロー

```yaml
# .github/workflows/changesets.yml
name: Changesets

on:
  push:
    branches: [main]

concurrency: ${{ github.workflow }}-${{ github.ref }}

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
      packages: write

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - run: npm ci

      - name: Create Release PR or Publish
        id: changesets
        uses: changesets/action@v1
        with:
          publish: npm run release
          version: npm run version
          title: 'chore: version packages'
          commit: 'chore: version packages'
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          NPM_TOKEN: ${{ secrets.NPM_TOKEN }}

      - name: Create GitHub Releases
        if: steps.changesets.outputs.published == 'true'
        run: |
          PACKAGES='${{ steps.changesets.outputs.publishedPackages }}'
          echo "$PACKAGES" | jq -r '.[] | "\(.name)@\(.version)"' | while read pkg; do
            NAME=$(echo "$pkg" | cut -d@ -f1-2)
            VERSION=$(echo "$pkg" | rev | cut -d@ -f1 | rev)
            gh release create "$NAME@v$VERSION" \
              --title "$NAME v$VERSION" \
              --generate-notes
          done
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### 5.5 モノレポでのパッケージ連携設定

```json
// .changeset/config.json — 高度な設定
{
  "$schema": "https://unpkg.com/@changesets/config/schema.json",
  "changelog": [
    "@changesets/changelog-github",
    { "repo": "myorg/myapp" }
  ],
  "commit": false,
  "fixed": [
    ["@myorg/react-components", "@myorg/react-icons"]
  ],
  "linked": [
    ["@myorg/core", "@myorg/utils", "@myorg/types"]
  ],
  "access": "public",
  "baseBranch": "main",
  "updateInternalDependencies": "patch",
  "ignore": ["@myorg/docs", "@myorg/examples"],
  "snapshot": {
    "useCalculatedVersion": true,
    "prereleaseTemplate": "{tag}-{datetime}-{commit}"
  }
}
```

```
linked と fixed の違い:

  linked (連動):
  - パッケージ A が minor → パッケージ B も minor にバンプ
  - 個別に changeset を作成可能
  - バージョン番号は常に同じ

  fixed (固定):
  - パッケージ群が常に同じバージョン番号を持つ
  - 1つの changeset で全パッケージが更新
  - React のようなパッケージ群に適する

  linked の例:
  @myorg/core: 1.2.0  ←── 同じ minor バージョン
  @myorg/utils: 1.2.0 ←── 同じ minor バージョン

  fixed の例:
  @myorg/react-components: 3.5.0 ←── 完全に同じバージョン
  @myorg/react-icons: 3.5.0      ←── 完全に同じバージョン
```

---

## 6. release-please による Google スタイルのリリース管理

### 6.1 release-please の基本設定

```yaml
# .github/workflows/release-please.yml
name: Release Please

on:
  push:
    branches: [main]

permissions:
  contents: write
  pull-requests: write

jobs:
  release-please:
    runs-on: ubuntu-latest
    outputs:
      release_created: ${{ steps.release.outputs.release_created }}
      tag_name: ${{ steps.release.outputs.tag_name }}

    steps:
      - uses: googleapis/release-please-action@v4
        id: release
        with:
          release-type: node
          token: ${{ secrets.GITHUB_TOKEN }}
```

```json
// release-please-config.json — モノレポ対応設定
{
  "$schema": "https://raw.githubusercontent.com/googleapis/release-please/main/schemas/config.json",
  "packages": {
    "packages/core": {
      "release-type": "node",
      "component": "core",
      "changelog-path": "CHANGELOG.md"
    },
    "packages/cli": {
      "release-type": "node",
      "component": "cli",
      "changelog-path": "CHANGELOG.md"
    },
    "packages/server": {
      "release-type": "node",
      "component": "server",
      "changelog-path": "CHANGELOG.md"
    }
  },
  "plugins": [
    {
      "type": "node-workspace",
      "updateAllPackages": true
    },
    {
      "type": "linked-versions",
      "groupName": "myapp",
      "components": ["core", "cli", "server"]
    }
  ]
}
```

---

## 7. CHANGELOG 自動生成

### 7.1 自動生成例

```markdown
<!-- CHANGELOG.md (自動生成例) -->
# Changelog

## [2.1.0](https://github.com/myorg/myapp/compare/v2.0.0...v2.1.0) (2025-03-15)

### Features

* **auth:** ソーシャルログインを追加 ([#123](https://github.com/myorg/myapp/issues/123)) ([a1b2c3d](https://github.com/myorg/myapp/commit/a1b2c3d))
* **dashboard:** リアルタイム通知を実装 ([#130](https://github.com/myorg/myapp/issues/130)) ([e4f5g6h](https://github.com/myorg/myapp/commit/e4f5g6h))

### Bug Fixes

* **api:** ページネーションのオフセット計算を修正 ([#456](https://github.com/myorg/myapp/issues/456)) ([i7j8k9l](https://github.com/myorg/myapp/commit/i7j8k9l))

### Performance Improvements

* **query:** N+1 クエリを解消しレスポンス時間を 40% 改善 ([#140](https://github.com/myorg/myapp/issues/140)) ([m0n1o2p](https://github.com/myorg/myapp/commit/m0n1o2p))

## [2.0.0](https://github.com/myorg/myapp/compare/v1.5.0...v2.0.0) (2025-02-01)

### BREAKING CHANGES

* **api:** レスポンス形式を JSON:API 仕様に変更 ([#100](https://github.com/myorg/myapp/issues/100))
```

### 7.2 カスタム CHANGELOG テンプレート

```javascript
// changelog-template.js — カスタム CHANGELOG テンプレート
const changelogConfig = {
  writerOpts: {
    transform: (commit, context) => {
      // commit type の日本語マッピング
      const typeMap = {
        feat: '新機能',
        fix: 'バグ修正',
        perf: 'パフォーマンス改善',
        refactor: 'リファクタリング',
        docs: 'ドキュメント',
        build: 'ビルド',
        ci: 'CI/CD',
      };

      if (typeMap[commit.type]) {
        commit.type = typeMap[commit.type];
      } else {
        return null; // 表示しない
      }

      // scope の日本語化 (任意)
      if (commit.scope) {
        commit.scope = commit.scope.replace(/auth/g, '認証')
                                    .replace(/api/g, 'API')
                                    .replace(/ui/g, 'UI');
      }

      return commit;
    },
    groupBy: 'type',
    commitGroupsSort: (a, b) => {
      const order = ['新機能', 'バグ修正', 'パフォーマンス改善', 'リファクタリング'];
      return order.indexOf(a.title) - order.indexOf(b.title);
    },
  },
};

module.exports = changelogConfig;
```

---

## 8. Feature Flag によるリリース制御

### 8.1 Feature Flag の基本実装

```typescript
// feature-flags.ts — Feature Flag の実装
interface FeatureFlag {
  name: string;
  enabled: boolean;
  rolloutPercentage?: number;
  allowedUsers?: string[];
  enabledEnvironments?: string[];
  description: string;
  createdAt: string;
  expiresAt?: string;
}

class FeatureFlagService {
  private flags: Map<string, FeatureFlag> = new Map();

  constructor(private readonly config: FeatureFlag[]) {
    for (const flag of config) {
      this.flags.set(flag.name, flag);
    }
  }

  isEnabled(flagName: string, context?: {
    userId?: string;
    environment?: string;
  }): boolean {
    const flag = this.flags.get(flagName);
    if (!flag) return false;

    // グローバルに無効
    if (!flag.enabled) return false;

    // 有効期限チェック
    if (flag.expiresAt && new Date(flag.expiresAt) < new Date()) {
      return false;
    }

    // 環境チェック
    if (flag.enabledEnvironments && context?.environment) {
      if (!flag.enabledEnvironments.includes(context.environment)) {
        return false;
      }
    }

    // ユーザー固有のアクセス
    if (flag.allowedUsers && context?.userId) {
      if (flag.allowedUsers.includes(context.userId)) {
        return true;
      }
    }

    // ロールアウト率による制御
    if (flag.rolloutPercentage !== undefined && context?.userId) {
      const hash = this.hashUserId(context.userId);
      return hash < flag.rolloutPercentage;
    }

    return flag.enabled;
  }

  private hashUserId(userId: string): number {
    let hash = 0;
    for (let i = 0; i < userId.length; i++) {
      const char = userId.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash) % 100;
  }
}

// 使用例
const featureFlags = new FeatureFlagService([
  {
    name: 'new-checkout-flow',
    enabled: true,
    rolloutPercentage: 10,  // 10% のユーザーに展開
    enabledEnvironments: ['production', 'staging'],
    description: '新しいチェックアウトフロー',
    createdAt: '2025-03-01',
    expiresAt: '2025-06-01',
  },
  {
    name: 'dark-mode',
    enabled: true,
    allowedUsers: ['user-001', 'user-002'],  // 特定ユーザーのみ
    description: 'ダークモード対応',
    createdAt: '2025-02-15',
  },
]);

// API エンドポイントでの使用
app.get('/checkout', (req, res) => {
  if (featureFlags.isEnabled('new-checkout-flow', {
    userId: req.user.id,
    environment: process.env.NODE_ENV,
  })) {
    return renderNewCheckout(req, res);
  }
  return renderLegacyCheckout(req, res);
});
```

### 8.2 Feature Flag ツールの比較

| ツール | 形態 | 料金 | 特徴 |
|--------|------|------|------|
| LaunchDarkly | SaaS | 有料 | エンタープライズ向け、SDK が豊富 |
| Unleash | OSS/SaaS | 無料枠あり | セルフホスト可能、API ベース |
| Flagsmith | OSS/SaaS | 無料枠あり | セルフホスト可能、セグメント機能 |
| ConfigCat | SaaS | 無料枠あり | シンプル、設定ファイルベース |
| 環境変数 | 自前 | 無料 | 最もシンプルだが動的変更不可 |
| リモートConfig | Firebase | 無料枠あり | モバイルアプリ向け |

### 8.3 Feature Flag を使ったリリース戦略

```
Feature Flag によるデプロイとリリースの分離:

  従来のアプローチ:
  ┌──────────────────────────────────────┐
  │ デプロイ = リリース                     │
  │ コードをデプロイ → 全ユーザーに公開     │
  │ 問題発生 → ロールバック (ダウンタイム)   │
  └──────────────────────────────────────┘

  Feature Flag アプローチ:
  ┌──────────────────────────────────────┐
  │ デプロイ ≠ リリース                    │
  │                                      │
  │ 1. コードをデプロイ (Flag OFF)         │
  │    → ユーザーには影響なし              │
  │                                      │
  │ 2. 内部テスター向けに Flag ON          │
  │    → 内部検証                         │
  │                                      │
  │ 3. 10% ロールアウト                    │
  │    → メトリクス監視                    │
  │                                      │
  │ 4. 50% → 100% ロールアウト             │
  │    → 段階的に全ユーザーに公開          │
  │                                      │
  │ 問題発生 → Flag OFF (即座に無効化)      │
  │    → ダウンタイムなし                   │
  └──────────────────────────────────────┘
```

---

## 9. ホットフィックスとロールバック

### 9.1 ホットフィックスプロセス

```bash
# ホットフィックスの手順

# 1. main から hotfix ブランチを作成
git checkout main
git pull origin main
git checkout -b hotfix/fix-payment-timeout

# 2. 修正をコミット
git commit -m "fix(payment): タイムアウト値を30秒に延長

決済API のタイムアウトが10秒で、ネットワーク遅延時に
タイムアウトエラーが頻発していた問題を修正。

Fixes #789"

# 3. テスト実行
npm test

# 4. main にマージ (PR 経由)
gh pr create --base main --title "fix(payment): タイムアウト値を修正" --body "..."

# 5. semantic-release が自動で PATCH バージョンアップ
# 例: v1.2.3 → v1.2.4

# 6. develop ブランチにもマージ (Git Flow の場合)
git checkout develop
git merge main
git push origin develop
```

### 9.2 ロールバック戦略

```yaml
# .github/workflows/rollback.yml — 緊急ロールバック
name: Emergency Rollback

on:
  workflow_dispatch:
    inputs:
      target-version:
        description: 'Version to rollback to (e.g., v1.2.3)'
        required: true
      reason:
        description: 'Reason for rollback'
        required: true
      environment:
        description: 'Target environment'
        required: true
        type: choice
        options:
          - production
          - staging

jobs:
  rollback:
    runs-on: ubuntu-latest
    environment: ${{ inputs.environment }}

    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ inputs.target-version }}

      - name: Verify target version exists
        run: |
          if ! git tag | grep -q "^${{ inputs.target-version }}$"; then
            echo "Error: Tag ${{ inputs.target-version }} not found"
            exit 1
          fi

      - name: Deploy rollback version
        run: |
          echo "Rolling back to ${{ inputs.target-version }}"
          echo "Reason: ${{ inputs.reason }}"
          # デプロイコマンド (環境に応じて変更)
          # aws ecs update-service ...
          # argocd app set myapp --revision ...

      - name: Notify rollback
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "ROLLBACK executed: ${{ inputs.environment }} → ${{ inputs.target-version }}\nReason: ${{ inputs.reason }}\nBy: ${{ github.actor }}"
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}

      - name: Create incident issue
        run: |
          gh issue create \
            --title "Rollback: ${{ inputs.environment }} → ${{ inputs.target-version }}" \
            --body "## Rollback Details
          - **Environment**: ${{ inputs.environment }}
          - **From**: latest
          - **To**: ${{ inputs.target-version }}
          - **Reason**: ${{ inputs.reason }}
          - **Executed by**: ${{ github.actor }}
          - **Time**: $(date -u '+%Y-%m-%d %H:%M:%S UTC')

          ## Action Items
          - [ ] Root cause analysis
          - [ ] Fix PR created
          - [ ] Post-mortem scheduled" \
            --label "incident,rollback"
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## 10. リリースの品質ゲート

### 10.1 リリース前の自動チェック

```yaml
# .github/workflows/release-gate.yml — リリース品質ゲート
name: Release Quality Gate

on:
  pull_request:
    branches: [main]

jobs:
  quality-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - run: npm ci

      # 1. ユニットテスト
      - name: Unit Tests
        run: npm test -- --coverage
        env:
          CI: true

      # 2. カバレッジチェック
      - name: Coverage Check
        run: |
          COVERAGE=$(npx c8 check-coverage --lines 80 --functions 80 --branches 70 2>&1) || {
            echo "Coverage below threshold"
            exit 1
          }

      # 3. リンティング
      - name: Lint
        run: npm run lint

      # 4. 型チェック
      - name: Type Check
        run: npx tsc --noEmit

      # 5. セキュリティ監査
      - name: Security Audit
        run: npm audit --audit-level=high

      # 6. ライセンスチェック
      - name: License Check
        run: npx license-checker --failOn "GPL-3.0;AGPL-3.0"

      # 7. バンドルサイズチェック
      - name: Bundle Size Check
        run: |
          npm run build
          SIZE=$(du -sb dist/ | cut -f1)
          MAX_SIZE=5242880  # 5MB
          if [ "$SIZE" -gt "$MAX_SIZE" ]; then
            echo "Bundle size ($SIZE bytes) exceeds limit ($MAX_SIZE bytes)"
            exit 1
          fi

      # 8. Changeset チェック (変更がある場合)
      - name: Check for changeset
        run: |
          if git diff --name-only ${{ github.event.pull_request.base.sha }} | grep -qE '\.(ts|tsx|js|jsx)$'; then
            if ! ls .changeset/*.md 2>/dev/null | grep -v README; then
              echo "Warning: Source files changed but no changeset found"
              echo "Run 'npx changeset' to add a changeset"
            fi
          fi
```

---

## 11. 比較表

| 特性 | semantic-release | Changesets | release-please |
|------|-----------------|------------|----------------|
| バージョン決定 | コミットメッセージから自動 | 開発者が明示的に記述 | コミットメッセージから自動 |
| モノレポ対応 | 限定的 | 優れている | 対応 |
| CHANGELOG 生成 | 自動 | 自動 | 自動 |
| npm publish | 対応 | 対応 | 対応 |
| GitHub Release | 対応 | 要追加設定 | 対応 |
| 学習コスト | 中 | 低い | 低い |
| カスタマイズ性 | プラグインで高い | 中 | 中 |
| Release PR | なし (直接リリース) | あり | あり |
| プレリリース | ブランチベース | コマンドベース | ブランチベース |
| メンテナー | semantic-release org | Changesets org | Google |

| ブランチ戦略 | Git Flow | GitHub Flow | Trunk-Based |
|-------------|----------|-------------|-------------|
| ブランチ数 | 多い | 少ない | 最少 |
| リリース頻度 | 低い | 中 | 高い |
| 複雑さ | 高い | 低い | 中 |
| 適用規模 | 大規模 | 中規模 | 全規模 |
| Feature Flag 必要性 | 低い | 中 | 高い |
| リリースブランチ | あり | なし | なし |
| ホットフィックス | 専用ブランチ | main から直接 | main から直接 |

| Feature Flag ツール | LaunchDarkly | Unleash | Flagsmith | 環境変数 |
|---------------------|-------------|---------|-----------|----------|
| 動的変更 | 対応 | 対応 | 対応 | 要再デプロイ |
| ターゲティング | 高度 | 中 | 中 | なし |
| A/B テスト | 対応 | 対応 | 対応 | 不可 |
| 監査ログ | あり | あり | あり | なし |
| セルフホスト | 不可 | 可能 | 可能 | - |
| SDK 対応言語 | 多数 | 多数 | 多数 | - |
| 無料枠 | なし | あり | あり | 無料 |

---

## 12. アンチパターン

### アンチパターン 1: 手動バージョン管理

```
[悪い例]
- package.json のバージョンを手動で変更
- CHANGELOG を手動で記述 (漏れや誤記が頻発)
- リリースタグを手動で作成 (付け忘れ)
- 「次のバージョン番号は何？」という議論に時間を浪費

[良い例]
- Conventional Commits + semantic-release で全自動化
- コミットメッセージの型 (feat/fix/BREAKING) でバージョンが自動決定
- CHANGELOG、タグ、GitHub Release が CI で自動生成
- 開発者はコミットメッセージに集中するだけ
```

### アンチパターン 2: リリースとホットフィックスの混在

```
[悪い例]
- 本番障害発生 → main ブランチに直接修正をプッシュ
- 開発中の未完成機能が一緒にリリースされてしまう
- ホットフィックスの手順が定まっておらず毎回混乱

[良い例]
- ホットフィックスブランチの手順を明文化:
  1. main から hotfix/xxx ブランチを作成
  2. 修正をコミット (fix: ...)
  3. テスト通過を確認
  4. main にマージ → 自動リリース (PATCH バージョンアップ)
  5. develop ブランチにもマージ (Git Flow の場合)
- Feature Flag で未完成機能を隠蔽しておけば、main からの修正が安全
```

### アンチパターン 3: Feature Flag の放置

```
[悪い例]
- Feature Flag を作成したが、全ユーザーに展開後も残し続ける
- 古い Flag が数十個残り、コードが条件分岐だらけに
- どの Flag が有効でどれが無効か把握できない
- Flag 同士の組み合わせで予期しない動作が発生

[良い例]
- Flag の有効期限を必ず設定 (最大90日)
- 全ユーザー展開後は次のスプリントで Flag を削除
- Flag 棚卸しを月次で実施 (未使用 Flag の削除)
- Flag のライフサイクルを Jira/GitHub Issues で追跡
- Flag 削除の PR テンプレートを用意:
  1. Flag のコードパスを削除
  2. テストを更新
  3. 設定ファイルから Flag 定義を削除
```

### アンチパターン 4: CHANGELOG の軽視

```
[悪い例]
- CHANGELOG が存在しない、または更新されていない
- 利用者が「何が変わったか」を把握できない
- 破壊的変更に気づかずアップデートして障害発生
- サポートへの問い合わせが増加

[良い例]
- CHANGELOG を自動生成し、リリースと同時に公開
- BREAKING CHANGES セクションを目立たせる
- 移行ガイドへのリンクを含める
- GitHub Release のボディに CHANGELOG を含める
- RSS/Atom フィードでリリース通知を配信
```

---

## 13. FAQ

### Q1: semantic-release と Changesets、どちらを選ぶべきですか？

**単一パッケージ**なら semantic-release が楽です。コミットメッセージから全自動でバージョンが決まります。**モノレポ**（複数パッケージ）なら Changesets が適しています。パッケージごとに変更内容を明示できるため、連動するバージョン管理が容易です。チームメンバーが Conventional Commits に慣れていない場合も、Changesets の方が学習コストが低いです。release-please は Google が開発・メンテしており、モノレポ対応も良好で、Release PR というワークフローが特徴です。3つとも成熟したツールなので、チームの好みとプロジェクト構成で選択してください。

### Q2: プレリリースバージョン（alpha/beta/rc）はどう管理しますか？

semantic-release では `branches` 設定で `next` や `beta` ブランチを定義します。これらのブランチへのマージで `v2.0.0-beta.1` のようなプレリリースバージョンが自動生成されます。Changesets では `npx changeset pre enter beta` コマンドでプレリリースモードに入り、`npx changeset pre exit` で通常モードに戻します。release-please ではブランチ名にプレリリースタイプを含めることで制御します。

### Q3: CHANGELOG はどのくらい遡って維持すべきですか？

全履歴を維持するのが理想ですが、現実的にはメジャーバージョン単位で分割することを推奨します。v1.x の CHANGELOG は `CHANGELOG-v1.md` としてアーカイブし、`CHANGELOG.md` には現行メジャーバージョンの履歴のみを記載します。Git タグとGitHub Releases が存在するため、古い変更履歴はそちらから参照できます。

### Q4: Conventional Commits のルールをチームに浸透させるには？

以下の段階的なアプローチを推奨します:

1. **まず commitlint + Husky を導入**: ローカルでコミット時にバリデーションする
2. **commitizen (cz) を導入**: 対話的にコミットメッセージを作成できるようにする
3. **PR テンプレートにルールを記載**: コミットメッセージの書き方をリマインドする
4. **CI でも検証**: PR のコミットメッセージを CI でチェックする
5. **チーム勉強会**: 実例を交えて30分程度の勉強会を実施する

### Q5: リリース頻度はどのくらいが適切ですか？

プロジェクトの成熟度とリスク許容度によります。一般的な目安:

- **SaaS (B2C)**: 日次〜週次。Trunk-Based + Feature Flag が適する
- **SaaS (B2B)**: 週次〜隔週。顧客通知のリードタイムを確保
- **ライブラリ (npm)**: 変更があれば随時。semantic-release で自動化
- **モバイルアプリ**: 隔週〜月次。アプリストアの審査時間を考慮
- **オンプレミス**: 月次〜四半期。顧客のアップデート作業を考慮

リリース頻度を上げるほど各リリースのリスクが下がり（変更が小さいため）、問題発生時の原因特定も容易になります。

---

## まとめ

| 項目 | 要点 |
|------|------|
| SemVer | MAJOR(破壊的変更).MINOR(新機能).PATCH(バグ修正) |
| Conventional Commits | feat:/fix:/BREAKING CHANGE で変更種別を明示 |
| semantic-release | コミットメッセージから自動バージョニング・リリース |
| Changesets | モノレポ対応。開発者が変更内容を明示的に記述 |
| release-please | Google 製。Release PR ベースのワークフロー |
| CHANGELOG | 自動生成。手動管理は漏れと労力の無駄 |
| Feature Flag | デプロイとリリースを分離。段階的ロールアウトを実現 |
| ホットフィックス | 専用ブランチと明文化された手順で対応 |
| プレリリース | alpha/beta/rc をブランチ戦略で管理 |
| 品質ゲート | テスト・カバレッジ・セキュリティ監査をリリース前に自動チェック |

---

## 次に読むべきガイド

- [00-deployment-strategies.md](./00-deployment-strategies.md) — デプロイ戦略の基本
- [01-cloud-deployment.md](./01-cloud-deployment.md) — クラウドデプロイの実践
- [02-container-deployment.md](./02-container-deployment.md) — コンテナデプロイとレジストリ管理
- [../03-monitoring/00-observability.md](../03-monitoring/00-observability.md) — オブザーバビリティの基礎

---

## 参考文献

1. **Semantic Versioning 2.0.0** — https://semver.org/ — SemVer の公式仕様
2. **Conventional Commits** — https://www.conventionalcommits.org/ — コミットメッセージ規約の公式サイト
3. **semantic-release Documentation** — https://semantic-release.gitbook.io/ — 自動リリースツールの公式ドキュメント
4. **Changesets Documentation** — https://github.com/changesets/changesets — モノレポ対応リリース管理ツール
5. **release-please Documentation** — https://github.com/googleapis/release-please — Google 製リリース管理ツール
6. **Martin Fowler - Feature Toggles** — https://martinfowler.com/articles/feature-toggles.html — Feature Flag の設計パターン
7. **Trunk Based Development** — https://trunkbaseddevelopment.com/ — Trunk-Based 開発の解説サイト

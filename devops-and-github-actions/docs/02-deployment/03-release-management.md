# リリース管理

> セマンティックバージョニング、CHANGELOG の自動生成、リリースプロセスの自動化により、予測可能で安全なリリースサイクルを確立する

## この章で学ぶこと

1. **セマンティックバージョニングの原則と実践** — MAJOR.MINOR.PATCH の判断基準と自動バージョニング
2. **CHANGELOG とリリースノートの自動生成** — Conventional Commits から自動的にドキュメントを生成する仕組み
3. **リリースプロセスの自動化** — GitHub Releases、npm publish、タグ管理の CI/CD パイプライン

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

---

## 3. Conventional Commits

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

---

## 4. semantic-release による自動リリース

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

---

## 5. Changesets によるモノレポ対応リリース

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

```markdown
<!-- .changeset/happy-dogs-dance.md (自動生成されるテンプレート) -->
---
"@myorg/core": minor
"@myorg/utils": patch
---

ユーザープロフィール API にアバター画像アップロード機能を追加。
utils パッケージに画像リサイズヘルパーを追加。
```

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

---

## 6. CHANGELOG 自動生成

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

---

## 7. 比較表

| 特性 | semantic-release | Changesets | release-please |
|------|-----------------|------------|----------------|
| バージョン決定 | コミットメッセージから自動 | 開発者が明示的に記述 | コミットメッセージから自動 |
| モノレポ対応 | 限定的 | 優れている | 対応 |
| CHANGELOG 生成 | 自動 | 自動 | 自動 |
| npm publish | 対応 | 対応 | 対応 |
| GitHub Release | 対応 | 要追加設定 | 対応 |
| 学習コスト | 中 | 低い | 低い |
| カスタマイズ性 | プラグインで高い | 中 | 中 |

| ブランチ戦略 | Git Flow | GitHub Flow | Trunk-Based |
|-------------|----------|-------------|-------------|
| ブランチ数 | 多い | 少ない | 最少 |
| リリース頻度 | 低い | 中 | 高い |
| 複雑さ | 高い | 低い | 中 |
| 適用規模 | 大規模 | 中規模 | 全規模 |
| Feature Flag 必要性 | 低い | 中 | 高い |

---

## 8. アンチパターン

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

---

## 9. FAQ

### Q1: semantic-release と Changesets、どちらを選ぶべきですか？

**単一パッケージ**なら semantic-release が楽です。コミットメッセージから全自動でバージョンが決まります。**モノレポ**（複数パッケージ）なら Changesets が適しています。パッケージごとに変更内容を明示できるため、連動するバージョン管理が容易です。チームメンバーが Conventional Commits に慣れていない場合も、Changesets の方が学習コストが低いです。

### Q2: プレリリースバージョン（alpha/beta/rc）はどう管理しますか？

semantic-release では `branches` 設定で `next` や `beta` ブランチを定義します。これらのブランチへのマージで `v2.0.0-beta.1` のようなプレリリースバージョンが自動生成されます。Changesets では `npx changeset pre enter beta` コマンドでプレリリースモードに入り、`npx changeset pre exit` で通常モードに戻します。

### Q3: CHANGELOG はどのくらい遡って維持すべきですか？

全履歴を維持するのが理想ですが、現実的にはメジャーバージョン単位で分割することを推奨します。v1.x の CHANGELOG は `CHANGELOG-v1.md` としてアーカイブし、`CHANGELOG.md` には現行メジャーバージョンの履歴のみを記載します。Git タグとGitHub Releases が存在するため、古い変更履歴はそちらから参照できます。

---

## まとめ

| 項目 | 要点 |
|------|------|
| SemVer | MAJOR(破壊的変更).MINOR(新機能).PATCH(バグ修正) |
| Conventional Commits | feat:/fix:/BREAKING CHANGE で変更種別を明示 |
| semantic-release | コミットメッセージから自動バージョニング・リリース |
| Changesets | モノレポ対応。開発者が変更内容を明示的に記述 |
| CHANGELOG | 自動生成。手動管理は漏れと労力の無駄 |
| ホットフィックス | 専用ブランチと明文化された手順で対応 |
| プレリリース | alpha/beta/rc をブランチ戦略で管理 |

---

## 次に読むべきガイド

- [00-deployment-strategies.md](./00-deployment-strategies.md) — デプロイ戦略の基本
- [01-cloud-deployment.md](./01-cloud-deployment.md) — クラウドデプロイの実践
- [../03-monitoring/00-observability.md](../03-monitoring/00-observability.md) — オブザーバビリティの基礎

---

## 参考文献

1. **Semantic Versioning 2.0.0** — https://semver.org/ — SemVer の公式仕様
2. **Conventional Commits** — https://www.conventionalcommits.org/ — コミットメッセージ規約の公式サイト
3. **semantic-release Documentation** — https://semantic-release.gitbook.io/ — 自動リリースツールの公式ドキュメント
4. **Changesets Documentation** — https://github.com/changesets/changesets — モノレポ対応リリース管理ツール

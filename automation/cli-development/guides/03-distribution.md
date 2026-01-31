# 📦 CLI 配布・パッケージングガイド

> **目的**: CLI ツールを npm、Homebrew、バイナリなど様々な方法で配布するための実践的な手法を習得する

## 📚 目次

1. [npm パッケージ配布](#npm-パッケージ配布)
2. [Homebrew 配布](#homebrew-配布)
3. [バイナリ配布](#バイナリ配布)
4. [GitHub Releases](#github-releases)
5. [自動更新](#自動更新)
6. [バージョン管理](#バージョン管理)

---

## npm パッケージ配布

### package.json 設定

```json
{
  "name": "my-cli-tool",
  "version": "1.0.0",
  "description": "A sample CLI tool",
  "main": "dist/index.js",
  "bin": {
    "my-cli": "./dist/index.js"
  },
  "files": [
    "dist",
    "README.md",
    "LICENSE"
  ],
  "scripts": {
    "build": "tsc",
    "prepublishOnly": "npm run build",
    "prepack": "npm run build"
  },
  "keywords": [
    "cli",
    "tool",
    "generator"
  ],
  "author": "Your Name <your.email@example.com>",
  "license": "MIT",
  "repository": {
    "type": "git",
    "url": "https://github.com/yourusername/my-cli-tool.git"
  },
  "bugs": {
    "url": "https://github.com/yourusername/my-cli-tool/issues"
  },
  "homepage": "https://github.com/yourusername/my-cli-tool#readme",
  "engines": {
    "node": ">=18.0.0"
  }
}
```

**重要フィールド**:
- **bin**: CLI コマンド名とエントリーポイント
- **files**: npm に含めるファイル（デフォルトは全て）
- **prepublishOnly**: 公開前に自動実行（ビルドなど）
- **engines**: 必要な Node.js バージョン

### .npmignore

```
# ソースファイル（dist のみ配布）
src/
*.ts
tsconfig.json

# テスト
__tests__/
*.test.js
*.spec.js

# 設定ファイル
.eslintrc
.prettierrc
.editorconfig

# CI/CD
.github/
.gitlab-ci.yml

# その他
node_modules/
.DS_Store
*.log
```

### npm 公開

```bash
# 1. npm アカウント作成（https://www.npmjs.com/）

# 2. ログイン
npm login

# 3. パッケージ名の確認（重複チェック）
npm search my-cli-tool

# 4. ビルド
npm run build

# 5. 公開前テスト
npm pack
# my-cli-tool-1.0.0.tgz が生成される

tar -xzf my-cli-tool-1.0.0.tgz
cd package
npm install -g

# 6. 公開
npm publish

# スコープ付きパッケージ（@username/package）の場合
npm publish --access public
```

### 更新

```bash
# バージョン更新
npm version patch  # 1.0.0 -> 1.0.1
npm version minor  # 1.0.0 -> 1.1.0
npm version major  # 1.0.0 -> 2.0.0

# 公開
npm publish
```

### npx 対応

**package.json**:
```json
{
  "name": "create-my-app",
  "bin": {
    "create-my-app": "./dist/index.js"
  }
}
```

**使用例**:
```bash
# インストールせずに実行
npx create-my-app my-project

# 特定バージョン
npx create-my-app@latest my-project
```

---

## Homebrew 配布

### Formula 作成

**1. tap リポジトリ作成**:
```bash
# GitHub に homebrew-<name> リポジトリを作成
# 例: homebrew-my-cli
```

**2. Formula ファイル作成**:

**Formula/my-cli.rb**:
```ruby
class MyCli < Formula
  desc "A sample CLI tool"
  homepage "https://github.com/yourusername/my-cli-tool"
  url "https://github.com/yourusername/my-cli-tool/archive/v1.0.0.tar.gz"
  sha256 "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
  license "MIT"

  depends_on "node"

  def install
    system "npm", "install", *Language::Node.std_npm_install_args(libexec)
    bin.install_symlink Dir["#{libexec}/bin/*"]
  end

  test do
    system "#{bin}/my-cli", "--version"
  end
end
```

**3. SHA256 ハッシュ取得**:
```bash
curl -L https://github.com/yourusername/my-cli-tool/archive/v1.0.0.tar.gz | shasum -a 256
```

**4. インストール**:
```bash
# tap 追加
brew tap yourusername/my-cli

# インストール
brew install my-cli

# 更新
brew upgrade my-cli
```

### Homebrew Core への提出

**要件**:
- 30日以上の歴史
- 75以上のスター
- 適切なテスト
- 安定版リリース

**手順**:
```bash
# 1. Homebrew Core をフォーク
# https://github.com/Homebrew/homebrew-core

# 2. Formula 作成
cd $(brew --repository homebrew/core)
brew create https://github.com/yourusername/my-cli-tool/archive/v1.0.0.tar.gz

# 3. テスト
brew install --build-from-source my-cli
brew test my-cli
brew audit --new-formula my-cli

# 4. PR 作成
```

---

## バイナリ配布

### pkg でバイナリ化

```bash
npm install -g pkg
```

**package.json**:
```json
{
  "name": "my-cli-tool",
  "bin": "dist/index.js",
  "pkg": {
    "scripts": "dist/**/*.js",
    "assets": [
      "templates/**/*"
    ],
    "targets": [
      "node18-linux-x64",
      "node18-macos-x64",
      "node18-macos-arm64",
      "node18-win-x64"
    ],
    "outputPath": "binaries"
  }
}
```

**ビルド**:
```bash
# すべてのプラットフォーム
pkg .

# 特定プラットフォーム
pkg -t node18-macos-arm64 .

# 出力
# binaries/my-cli-macos-arm64
# binaries/my-cli-linux-x64
# binaries/my-cli-win-x64.exe
```

### Bun でバイナリ化

```bash
# インストール
curl -fsSL https://bun.sh/install | bash

# ビルド
bun build ./src/index.ts --compile --outfile my-cli

# 実行
./my-cli
```

### esbuild + Node.js SEA

**Node.js Single Executable Application (SEA)**:

```bash
# 1. esbuild でバンドル
npx esbuild src/index.ts --bundle --platform=node --outfile=dist/bundle.js

# 2. SEA 設定
cat > sea-config.json << EOF
{
  "main": "dist/bundle.js",
  "output": "sea-prep.blob"
}
EOF

# 3. blob 生成
node --experimental-sea-config sea-config.json

# 4. 実行ファイル作成（macOS）
cp $(command -v node) my-cli
npx postject my-cli NODE_SEA_BLOB sea-prep.blob \
    --sentinel-fuse NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2 \
    --macho-segment-name NODE_SEA

# 5. 署名（macOS）
codesign --sign - my-cli

# 実行
./my-cli
```

---

## GitHub Releases

### リリース自動化（GitHub Actions）

**.github/workflows/release.yml**:
```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          registry-url: 'https://registry.npmjs.org'

      - name: Install dependencies
        run: npm ci

      - name: Build
        run: npm run build

      - name: Publish to npm
        run: npm publish
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          files: |
            dist/**/*
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### バイナリビルド + リリース

**.github/workflows/release-binaries.yml**:
```yaml
name: Release Binaries

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        include:
          - os: ubuntu-latest
            target: linux-x64
          - os: macos-latest
            target: macos-x64
          - os: windows-latest
            target: win-x64

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm ci

      - name: Build
        run: npm run build

      - name: Build binary
        run: npx pkg . -t node18-${{ matrix.target }} -o my-cli-${{ matrix.target }}

      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: my-cli-${{ matrix.target }}
          path: my-cli-${{ matrix.target }}*

  release:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Download all artifacts
        uses: actions/download-artifact@v3

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          files: |
            my-cli-*/*
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### リリース実行

```bash
# 1. バージョン更新
npm version minor  # 1.0.0 -> 1.1.0

# 2. タグをプッシュ（自動的に GitHub Actions が実行される）
git push --tags

# 3. リリースノート編集（GitHub UI で）
```

---

## 自動更新

### update-notifier

```bash
npm install update-notifier
```

**src/index.ts**:
```typescript
import updateNotifier from 'update-notifier'
import { readFileSync } from 'fs'
import { join } from 'path'

// package.json 読み込み
const pkg = JSON.parse(
  readFileSync(join(__dirname, '../package.json'), 'utf-8')
)

// 更新チェック
const notifier = updateNotifier({
  pkg,
  updateCheckInterval: 1000 * 60 * 60 * 24 // 24時間
})

// 更新通知
if (notifier.update) {
  notifier.notify({
    message: `Update available ${notifier.update.current} → ${notifier.update.latest}\nRun {updateCommand} to update`
  })
}

// CLI コマンド実行
program.parse()
```

**出力例**:
```bash
$ my-cli create myapp

   ╭───────────────────────────────────────────────────╮
   │                                                   │
   │   Update available 1.0.0 → 1.1.0                 │
   │   Run npm install -g my-cli-tool to update       │
   │                                                   │
   ╰───────────────────────────────────────────────────╯

Creating project...
```

### セルフアップデート機能

```typescript
import { Command } from 'commander'
import { execa } from 'execa'
import ora from 'ora'
import chalk from 'chalk'

export function updateCommand() {
  return new Command('update')
    .description('Update CLI to the latest version')
    .action(async () => {
      const spinner = ora('Checking for updates...').start()

      try {
        // 最新バージョン確認
        const { stdout } = await execa('npm', ['view', 'my-cli-tool', 'version'])
        const latestVersion = stdout.trim()

        const pkg = JSON.parse(
          readFileSync(join(__dirname, '../package.json'), 'utf-8')
        )
        const currentVersion = pkg.version

        if (latestVersion === currentVersion) {
          spinner.succeed(chalk.green('Already up to date!'))
          return
        }

        spinner.text = `Updating ${currentVersion} → ${latestVersion}...`

        // 更新
        await execa('npm', ['install', '-g', 'my-cli-tool@latest'], {
          stdio: 'inherit'
        })

        spinner.succeed(chalk.green('Updated successfully!'))

      } catch (error) {
        spinner.fail(chalk.red('Update failed'))
        console.error(error)
        process.exit(1)
      }
    })
}
```

**使用例**:
```bash
$ my-cli update
Checking for updates...
Updating 1.0.0 → 1.1.0...
✓ Updated successfully!
```

---

## バージョン管理

### Semantic Versioning

**フォーマット**: `MAJOR.MINOR.PATCH`

- **MAJOR**: 破壊的変更
- **MINOR**: 新機能追加（後方互換性あり）
- **PATCH**: バグ修正

**例**:
```
1.0.0 → 1.0.1  (バグ修正)
1.0.1 → 1.1.0  (新機能追加)
1.1.0 → 2.0.0  (破壊的変更)
```

### standard-version

```bash
npm install -D standard-version
```

**package.json**:
```json
{
  "scripts": {
    "release": "standard-version"
  }
}
```

**実行**:
```bash
# 自動的にバージョン決定 + CHANGELOG 更新 + タグ作成
npm run release

# プレリリース
npm run release -- --prerelease alpha
# 1.0.0 -> 1.0.1-alpha.0

# 特定バージョン
npm run release -- --release-as minor
# 1.0.0 -> 1.1.0
```

**CHANGELOG.md** が自動生成される:
```markdown
# Changelog

## [1.1.0](https://github.com/user/repo/compare/v1.0.0...v1.1.0) (2025-01-01)

### Features

* add new command ([abc1234](https://github.com/user/repo/commit/abc1234))
* improve error messages ([def5678](https://github.com/user/repo/commit/def5678))

### Bug Fixes

* fix crash on invalid input ([ghi9012](https://github.com/user/repo/commit/ghi9012))
```

### Conventional Commits

**コミットメッセージフォーマット**:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**タイプ**:
- `feat`: 新機能
- `fix`: バグ修正
- `docs`: ドキュメント
- `style`: コードスタイル
- `refactor`: リファクタリング
- `test`: テスト
- `chore`: その他

**例**:
```bash
git commit -m "feat(create): add template selection"
git commit -m "fix(install): resolve dependency conflict"
git commit -m "docs: update README with examples"
```

### Commitizen

```bash
npm install -D commitizen cz-conventional-changelog
```

**package.json**:
```json
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

**使用例**:
```bash
$ npm run commit

? Select the type of change that you're committing: (Use arrow keys)
❯ feat:     A new feature
  fix:      A bug fix
  docs:     Documentation only changes
  style:    Changes that do not affect the meaning of the code
  refactor: A code change that neither fixes a bug nor adds a feature
  perf:     A code change that improves performance
  test:     Adding missing tests

? What is the scope of this change (e.g. component or file name): (press enter to skip)
create

? Write a short, imperative tense description of the change (max 94 chars):
add template selection

? Provide a longer description of the change: (press enter to skip)

? Are there any breaking changes? No

? Does this change affect any open issues? No
```

---

## まとめ

### 配布方法の選択

| 配布方法 | メリット | デメリット | 対象 |
|---------|---------|-----------|------|
| **npm** | 簡単、自動更新、依存関係管理 | Node.js 必須 | Node.js ユーザー |
| **Homebrew** | macOS/Linux で人気、バージョン管理 | macOS/Linux のみ | Mac ユーザー |
| **バイナリ** | Node.js 不要、高速起動 | ファイルサイズ大、更新が手動 | 一般ユーザー |
| **npx** | インストール不要 | 毎回ダウンロード | 一時的な使用 |

### 配布チェックリスト

**npm 公開**:
- [ ] package.json の bin フィールド設定
- [ ] files フィールドで配布ファイル指定
- [ ] prepublishOnly スクリプトでビルド
- [ ] README.md 作成
- [ ] LICENSE ファイル追加

**Homebrew**:
- [ ] tap リポジトリ作成
- [ ] Formula ファイル作成
- [ ] テスト追加

**バイナリ配布**:
- [ ] pkg でマルチプラットフォームビルド
- [ ] GitHub Actions で自動ビルド
- [ ] GitHub Releases で配布

**バージョン管理**:
- [ ] Semantic Versioning 遵守
- [ ] Conventional Commits 使用
- [ ] CHANGELOG 自動生成
- [ ] 自動更新通知

---

## 実践例: 完全な配布フロー

### 1. 開発

```bash
# 機能実装
git add .
git commit -m "feat: add new command"
```

### 2. リリース

```bash
# バージョン更新 + CHANGELOG 生成
npm run release

# タグをプッシュ（GitHub Actions で自動公開）
git push --follow-tags origin main
```

### 3. GitHub Actions が自動実行

- npm に公開
- バイナリビルド
- GitHub Releases 作成
- Homebrew Formula 更新

### 4. ユーザーが更新

```bash
# npm
npm install -g my-cli-tool@latest

# Homebrew
brew upgrade my-cli

# バイナリ
# GitHub Releases からダウンロード

# セルフアップデート
my-cli update
```

---

*適切な配布方法で、より多くのユーザーにツールを届けましょう。*

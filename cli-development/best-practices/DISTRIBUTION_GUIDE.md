# CLI Distribution Guide

## 配布方法の選択

| 配布方法 | メリット | デメリット | 対象ユーザー |
|---------|---------|-----------|------------|
| **npm** | 簡単、自動更新、依存管理 | Node.js 必須 | Node.js 開発者 |
| **PyPI** | 簡単、依存管理 | Python 必須 | Python 開発者 |
| **Homebrew** | macOS で人気、バージョン管理 | macOS/Linux のみ | Mac ユーザー |
| **バイナリ** | 依存なし、高速起動 | サイズ大、更新手動 | 一般ユーザー |
| **Docker** | 環境統一、依存隔離 | Docker 必須 | DevOps |

## npm 公開

### 準備

**package.json**:
```json
{
  "name": "mycli",
  "version": "1.0.0",
  "description": "A powerful CLI tool",
  "bin": {
    "mycli": "./dist/index.js"
  },
  "files": [
    "dist",
    "README.md",
    "LICENSE"
  ],
  "scripts": {
    "build": "tsc",
    "prepublishOnly": "npm run build"
  },
  "keywords": ["cli", "tool", "generator"],
  "author": "Your Name <your.email@example.com>",
  "license": "MIT",
  "repository": {
    "type": "git",
    "url": "https://github.com/username/mycli.git"
  },
  "engines": {
    "node": ">=18.0.0"
  }
}
```

**.npmignore**:
```
src/
tests/
*.test.ts
*.spec.ts
tsconfig.json
.github/
.vscode/
```

### 公開スクリプト

**scripts/publish.sh**:
```bash
#!/bin/bash
set -e

echo "🚀 Publishing to npm..."

# バージョン確認
VERSION=$(node -p "require('./package.json').version")
echo "Version: $VERSION"

# ビルド
echo "Building..."
npm run build

# テスト
echo "Running tests..."
npm test

# リント
echo "Linting..."
npm run lint

# 公開確認
read -p "Publish version $VERSION to npm? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    npm publish
    echo "✅ Published version $VERSION"

    # Git タグ
    git tag "v$VERSION"
    git push --tags

    echo "✅ Tagged version v$VERSION"
else
    echo "❌ Cancelled"
fi
```

## PyPI 公開

### 準備

**pyproject.toml**:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mycli"
version = "1.0.0"
description = "A powerful CLI tool"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = ["cli", "tool", "generator"]
dependencies = [
    "typer[all]>=0.9.0",
    "rich>=13.0.0",
]

[project.scripts]
mycli = "cli.main:app"

[project.urls]
Homepage = "https://github.com/username/mycli"
Repository = "https://github.com/username/mycli.git"
```

**MANIFEST.in**:
```
include README.md
include LICENSE
recursive-include src *.py
```

### 公開スクリプト

**scripts/publish.sh**:
```bash
#!/bin/bash
set -e

echo "🚀 Publishing to PyPI..."

# バージョン確認
VERSION=$(python -c "import tomli; print(tomli.load(open('pyproject.toml', 'rb'))['project']['version'])")
echo "Version: $VERSION"

# テスト
echo "Running tests..."
pytest

# リント
echo "Linting..."
ruff check src/ tests/

# ビルド
echo "Building..."
python -m build

# 公開確認
read -p "Publish version $VERSION to PyPI? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    twine upload dist/*
    echo "✅ Published version $VERSION"

    # Git タグ
    git tag "v$VERSION"
    git push --tags

    echo "✅ Tagged version v$VERSION"
else
    echo "❌ Cancelled"
fi
```

## Homebrew Formula

### Formula 作成

**homebrew-mycli/Formula/mycli.rb**:
```ruby
class Mycli < Formula
  desc "A powerful CLI tool"
  homepage "https://github.com/username/mycli"
  url "https://github.com/username/mycli/archive/v1.0.0.tar.gz"
  sha256 "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
  license "MIT"

  depends_on "node"

  def install
    system "npm", "install", *Language::Node.std_npm_install_args(libexec)
    bin.install_symlink Dir["#{libexec}/bin/*"]
  end

  test do
    system "#{bin}/mycli", "--version"
  end
end
```

### Formula 更新スクリプト

**scripts/update-homebrew.sh**:
```bash
#!/bin/bash
set -e

VERSION=$1
if [ -z "$VERSION" ]; then
    echo "Usage: ./update-homebrew.sh <version>"
    exit 1
fi

# アーカイブダウンロード
URL="https://github.com/username/mycli/archive/v$VERSION.tar.gz"
wget -O mycli-$VERSION.tar.gz $URL

# SHA256 計算
SHA256=$(sha256sum mycli-$VERSION.tar.gz | awk '{print $1}')

echo "Version: $VERSION"
echo "SHA256: $SHA256"

# Formula 更新
cat > Formula/mycli.rb << EOF
class Mycli < Formula
  desc "A powerful CLI tool"
  homepage "https://github.com/username/mycli"
  url "$URL"
  sha256 "$SHA256"
  license "MIT"

  depends_on "node"

  def install
    system "npm", "install", *Language::Node.std_npm_install_args(libexec)
    bin.install_symlink Dir["#{libexec}/bin/*"]
  end

  test do
    system "#{bin}/mycli", "--version"
  end
end
EOF

echo "✅ Formula updated"

# クリーンアップ
rm mycli-$VERSION.tar.gz
```

## バイナリ配布

### pkg でバイナリ化

**package.json**:
```json
{
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

**scripts/build-binaries.sh**:
```bash
#!/bin/bash
set -e

echo "🔨 Building binaries..."

# ビルド
npm run build

# pkg でバイナリ化
npx pkg . -o binaries/mycli

echo "✅ Binaries created:"
ls -lh binaries/

# アーカイブ作成
cd binaries
for file in mycli-*; do
    if [[ $file == *.exe ]]; then
        zip "${file%.exe}.zip" "$file"
    else
        tar -czf "$file.tar.gz" "$file"
    fi
done

echo "✅ Archives created"
ls -lh *.{zip,tar.gz}
```

## GitHub Actions 自動化

### リリースワークフロー

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

      - name: Test
        run: npm test

      - name: Publish to npm
        run: npm publish
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          files: dist/**/*
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### バイナリビルドワークフロー

**.github/workflows/binaries.yml**:
```yaml
name: Build Binaries

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
        run: npx pkg . -t node18-${{ matrix.target }} -o mycli-${{ matrix.target }}

      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: mycli-${{ matrix.target }}
          path: mycli-${{ matrix.target }}*

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
            mycli-*/*
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## Docker 配布

### Dockerfile

**Dockerfile**:
```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY dist ./dist

ENTRYPOINT ["node", "dist/index.js"]
CMD ["--help"]
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  mycli:
    build: .
    image: mycli:latest
    volumes:
      - ./projects:/projects
    working_dir: /projects
```

### Docker Hub へ公開

**scripts/publish-docker.sh**:
```bash
#!/bin/bash
set -e

VERSION=$1
if [ -z "$VERSION" ]; then
    echo "Usage: ./publish-docker.sh <version>"
    exit 1
fi

echo "🐳 Building Docker image..."

# ビルド
docker build -t username/mycli:$VERSION .
docker tag username/mycli:$VERSION username/mycli:latest

# 公開
docker push username/mycli:$VERSION
docker push username/mycli:latest

echo "✅ Published Docker image"
```

## 自動更新機構

### update-notifier (Node.js)

```typescript
import updateNotifier from 'update-notifier'
import { readFileSync } from 'fs'
import { join } from 'path'

const pkg = JSON.parse(
  readFileSync(join(__dirname, '../package.json'), 'utf-8')
)

const notifier = updateNotifier({
  pkg,
  updateCheckInterval: 1000 * 60 * 60 * 24 // 24時間
})

if (notifier.update) {
  notifier.notify({
    message: `Update available ${notifier.update.current} → ${notifier.update.latest}\nRun {updateCommand} to update`
  })
}
```

### セルフアップデートコマンド

```typescript
import { Command } from 'commander'
import { execa } from 'execa'
import ora from 'ora'
import chalk from 'chalk'

export function updateCommand(): Command {
  return new Command('update')
    .description('Update CLI to the latest version')
    .action(async () => {
      const spinner = ora('Checking for updates...').start()

      try {
        const { stdout } = await execa('npm', ['view', 'mycli', 'version'])
        const latestVersion = stdout.trim()

        const currentVersion = pkg.version

        if (latestVersion === currentVersion) {
          spinner.succeed(chalk.green('Already up to date!'))
          return
        }

        spinner.text = `Updating ${currentVersion} → ${latestVersion}...`

        await execa('npm', ['install', '-g', 'mycli@latest'], {
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

## リリースプロセス

### 完全な リリースフロー

```bash
# 1. バージョン更新
npm version minor  # 1.0.0 -> 1.1.0

# 2. CHANGELOG 更新
npm run changelog

# 3. コミット
git add .
git commit -m "chore: release v1.1.0"

# 4. タグ作成
git tag v1.1.0

# 5. プッシュ（GitHub Actions が自動実行）
git push --follow-tags origin main

# 6. Homebrew Formula 更新
./scripts/update-homebrew.sh 1.1.0
cd ../homebrew-mycli
git add Formula/mycli.rb
git commit -m "Update mycli to 1.1.0"
git push
```

---

*適切な配布方法で、より多くのユーザーにツールを届けましょう。*

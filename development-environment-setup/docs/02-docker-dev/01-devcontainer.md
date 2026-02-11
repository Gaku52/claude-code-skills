# Dev Container

> .devcontainer 設定で開発環境をコンテナ化し、VS Code 統合と GitHub Codespaces によってチーム全員が同一環境で即座に開発を開始できる仕組みを構築する。

## この章で学ぶこと

1. **devcontainer.json の構造と設定パターン** -- コンテナベースの開発環境を宣言的に定義し、再現性を確保する手法を理解する
2. **VS Code Remote - Containers との統合** -- ローカルの VS Code からコンテナ内で透過的に開発するワークフローを構築する
3. **GitHub Codespaces の活用と最適化** -- クラウドベースの Dev Container で、ローカルマシンに依存しない開発環境を実現する

---

## 1. Dev Container の概要

### 1.1 Dev Container とは

```
+------------------------------------------------------------------+
|                 従来の開発環境 vs Dev Container                     |
+------------------------------------------------------------------+
|                                                                  |
|  [従来]                                                          |
|  開発者A: macOS + Node 18 + Python 3.9 + MySQL 8.0              |
|  開発者B: Windows + Node 20 + Python 3.11 + MySQL 5.7           |
|  開発者C: Ubuntu + Node 16 + Python 3.10 + MariaDB              |
|  → 環境差異によるバグ「自分の環境では動くんですが...」              |
|                                                                  |
|  [Dev Container]                                                 |
|  開発者A: macOS + Docker → コンテナ(Node 20 + Python 3.11 + ...)  |
|  開発者B: Windows + Docker → コンテナ(Node 20 + Python 3.11 + ...) |
|  開発者C: Ubuntu + Docker → コンテナ(Node 20 + Python 3.11 + ...) |
|  → 全員が同一環境。設定は .devcontainer/ にコード管理              |
|                                                                  |
+------------------------------------------------------------------+
```

### 1.2 Dev Container のアーキテクチャ

```
+------------------------------------------------------------------+
|              Dev Container アーキテクチャ                          |
+------------------------------------------------------------------+
|                                                                  |
|  [ホストマシン]                                                   |
|    +-- VS Code / Cursor                                          |
|    |     +-- Remote - Containers 拡張                             |
|    |     |     (JSON-RPC over stdio)                             |
|    |     v                                                       |
|    +-- Docker Engine                                             |
|          |                                                       |
|          v                                                       |
|    +-- Dev Container ──────────────────────+                     |
|    |   |  ベースイメージ (Ubuntu/Debian)     |                    |
|    |   |  + Node.js / Python / Go etc.     |                    |
|    |   |  + VS Code Server                 |                    |
|    |   |  + 拡張機能 (コンテナ内)            |                    |
|    |   |                                   |                    |
|    |   |  /workspace ← プロジェクトマウント  |                    |
|    |   +-----------------------------------+                    |
|    |                                                            |
|    +-- Volume: node_modules, .cache 等                           |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 2. devcontainer.json の設定

### 2.1 基本構成

```jsonc
// .devcontainer/devcontainer.json
{
  "name": "My Project Dev",

  // ベースイメージ（シンプルな場合）
  "image": "mcr.microsoft.com/devcontainers/typescript-node:20-bookworm",

  // または Dockerfile を使う場合
  // "build": {
  //   "dockerfile": "Dockerfile",
  //   "context": "..",
  //   "args": { "NODE_VERSION": "20" }
  // },

  // または Docker Compose を使う場合
  // "dockerComposeFile": "docker-compose.yml",
  // "service": "app",
  // "workspaceFolder": "/workspace",

  // Features (追加ツールのモジュラーインストール)
  "features": {
    "ghcr.io/devcontainers/features/git:1": {},
    "ghcr.io/devcontainers/features/github-cli:1": {},
    "ghcr.io/devcontainers/features/docker-in-docker:2": {},
    "ghcr.io/devcontainers/features/node:1": {
      "version": "20"
    }
  },

  // VS Code 設定
  "customizations": {
    "vscode": {
      "extensions": [
        "dbaeumer.vscode-eslint",
        "esbenp.prettier-vscode",
        "bradlc.vscode-tailwindcss",
        "ms-vscode.vscode-typescript-next"
      ],
      "settings": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "esbenp.prettier-vscode",
        "typescript.tsdk": "node_modules/typescript/lib"
      }
    }
  },

  // ポートフォワーディング
  "forwardPorts": [3000, 5432, 6379],
  "portsAttributes": {
    "3000": { "label": "App", "onAutoForward": "openBrowser" },
    "5432": { "label": "PostgreSQL", "onAutoForward": "silent" },
    "6379": { "label": "Redis", "onAutoForward": "silent" }
  },

  // ライフサイクルコマンド
  "postCreateCommand": "npm ci",
  "postStartCommand": "npm run db:migrate",
  "postAttachCommand": "echo 'Dev Container ready!'",

  // コンテナユーザー
  "remoteUser": "node",

  // マウント設定
  "mounts": [
    "source=${localWorkspaceFolder}/.env,target=/workspace/.env,type=bind,consistency=cached",
    "source=node_modules,target=/workspace/node_modules,type=volume"
  ],

  // 環境変数
  "remoteEnv": {
    "NODE_ENV": "development",
    "DATABASE_URL": "postgresql://postgres:postgres@db:5432/myapp_dev"
  }
}
```

### 2.2 Dockerfile を使った高度な設定

```dockerfile
# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/typescript-node:20-bookworm

# システムパッケージ
RUN apt-get update && apt-get install -y \
    postgresql-client \
    redis-tools \
    jq \
    && rm -rf /var/lib/apt/lists/*

# グローバル npm パッケージ
RUN su node -c "npm install -g \
    tsx \
    prisma \
    @biomejs/biome \
    turbo"

# AWS CLI (オプション)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
    -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

# カスタムシェル設定
COPY .devcontainer/.zshrc /home/node/.zshrc
RUN chown node:node /home/node/.zshrc
```

### 2.3 Docker Compose との連携

```yaml
# .devcontainer/docker-compose.yml
version: '3.8'

services:
  app:
    build:
      context: ..
      dockerfile: .devcontainer/Dockerfile
    volumes:
      - ..:/workspace:cached
      - node_modules:/workspace/node_modules
    command: sleep infinity
    networks:
      - dev
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started

  db:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: myapp_dev
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ../scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
    networks:
      - dev

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    ports:
      - "6379:6379"
    networks:
      - dev

volumes:
  node_modules:
  pgdata:

networks:
  dev:
```

---

## 3. Dev Container Features

### 3.1 Features の仕組み

```
+------------------------------------------------------------------+
|              Dev Container Features                               |
+------------------------------------------------------------------+
|                                                                  |
|  devcontainer.json                                               |
|    "features": {                                                 |
|      "ghcr.io/devcontainers/features/node:1": {}                |
|      "ghcr.io/devcontainers/features/python:1": {}              |
|      "ghcr.io/devcontainers/features/go:1": {}                  |
|    }                                                             |
|         |                                                        |
|         v  各 Feature は OCI イメージとして配布                    |
|    +-----------------------------------+                         |
|    | install.sh (インストールスクリプト) |                        |
|    | devcontainer-feature.json (定義)   |                        |
|    +-----------------------------------+                         |
|         |                                                        |
|         v  ベースイメージに順番に適用                              |
|    最終的な Dev Container イメージ                                |
|                                                                  |
+------------------------------------------------------------------+
```

### 3.2 よく使う Features 一覧

| Feature | 用途 | 設定例 |
|---------|------|--------|
| `node` | Node.js ランタイム | `"version": "20"` |
| `python` | Python ランタイム | `"version": "3.11"` |
| `go` | Go ランタイム | `"version": "1.21"` |
| `docker-in-docker` | コンテナ内 Docker | デフォルトでOK |
| `github-cli` | gh コマンド | デフォルトでOK |
| `aws-cli` | AWS CLI v2 | デフォルトでOK |
| `terraform` | Terraform | `"version": "latest"` |
| `kubectl-helm-minikube` | K8s ツール群 | デフォルトでOK |
| `git` | 最新 Git | デフォルトでOK |
| `common-utils` | zsh, Oh My Zsh 等 | デフォルトでOK |

---

## 4. GitHub Codespaces

### 4.1 Codespaces の設定

```jsonc
// .devcontainer/devcontainer.json (Codespaces 用追加設定)
{
  "name": "My Project (Codespaces)",
  "image": "mcr.microsoft.com/devcontainers/typescript-node:20",

  // Codespaces 固有設定
  "hostRequirements": {
    "cpus": 4,
    "memory": "8gb",
    "storage": "32gb"
  },

  // Prebuild 設定（初回起動を高速化）
  "updateContentCommand": "npm ci",
  "postCreateCommand": "npm run setup",

  // GitHub Codespaces 固有のカスタマイズ
  "customizations": {
    "codespaces": {
      "openFiles": ["README.md", "src/index.ts"]
    },
    "vscode": {
      "extensions": [
        "GitHub.copilot",
        "dbaeumer.vscode-eslint"
      ]
    }
  },

  // シークレット (Codespaces Settings で設定)
  // GITHUB_TOKEN は自動注入
  // その他は Settings > Codespaces > Secrets で管理
  "secrets": {
    "DATABASE_URL": {
      "description": "PostgreSQL connection string"
    },
    "API_KEY": {
      "description": "External API key for development"
    }
  }
}
```

### 4.2 Prebuild の設定

```yaml
# .github/codespaces/prebuild-configuration.yml
# リポジトリ Settings > Codespaces > Prebuild configuration で設定

# または devcontainer.json の lifecycle commands で対応:
# "updateContentCommand" は Prebuild 時に実行される
# "postCreateCommand" は初回作成時に実行される
# "postStartCommand" は起動ごとに実行される
```

### 4.3 ローカル Dev Container vs Codespaces 比較

| 項目 | ローカル Dev Container | GitHub Codespaces |
|------|----------------------|-------------------|
| 実行場所 | ローカルマシン | GitHub クラウド |
| Docker 必要 | 必要 | 不要 |
| スペック | ローカルに依存 | 2〜32 コア選択可 |
| コスト | Docker のみ | 従量課金(Free枠あり) |
| ネットワーク | ローカルネットワーク | GitHub ネットワーク |
| 起動速度 | イメージDL後は速い | Prebuild で高速化可 |
| オフライン | 可能 | 不可 |
| ポートフォワード | localhost直接 | ポート転送(公開URL可) |
| Git 認証 | ローカル設定 | GitHub 自動認証 |
| シークレット | .env ファイル | Codespaces Secrets |

---

## 5. 実践的なプロジェクトテンプレート

### 5.1 フルスタック Web アプリ

```jsonc
// .devcontainer/devcontainer.json (フルスタック)
{
  "name": "Full Stack App",
  "dockerComposeFile": "docker-compose.yml",
  "service": "app",
  "workspaceFolder": "/workspace",

  "features": {
    "ghcr.io/devcontainers/features/github-cli:1": {},
    "ghcr.io/devcontainers/features/docker-in-docker:2": {}
  },

  "customizations": {
    "vscode": {
      "extensions": [
        "dbaeumer.vscode-eslint",
        "esbenp.prettier-vscode",
        "prisma.prisma",
        "bradlc.vscode-tailwindcss",
        "ms-azuretools.vscode-docker",
        "humao.rest-client"
      ],
      "settings": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
          "source.fixAll.eslint": "explicit"
        },
        "[prisma]": {
          "editor.defaultFormatter": "Prisma.prisma"
        }
      }
    }
  },

  "forwardPorts": [3000, 5432, 6379, 8025],
  "portsAttributes": {
    "3000": { "label": "Frontend", "onAutoForward": "openBrowser" },
    "5432": { "label": "PostgreSQL", "onAutoForward": "silent" },
    "6379": { "label": "Redis", "onAutoForward": "silent" },
    "8025": { "label": "MailHog UI", "onAutoForward": "notify" }
  },

  "postCreateCommand": "bash .devcontainer/post-create.sh",
  "postStartCommand": "npm run dev &",
  "remoteUser": "node"
}
```

### 5.2 セットアップスクリプト

```bash
#!/bin/bash
# .devcontainer/post-create.sh

set -euo pipefail

echo "=== Dev Container Setup ==="

# 依存関係インストール
echo ">>> Installing dependencies..."
npm ci

# データベースマイグレーション
echo ">>> Running database migrations..."
npx prisma migrate dev --name init 2>/dev/null || npx prisma migrate deploy

# シードデータ
echo ">>> Seeding database..."
npx prisma db seed 2>/dev/null || echo "No seed script found, skipping"

# Git フック
echo ">>> Setting up Git hooks..."
npx husky install 2>/dev/null || echo "Husky not configured, skipping"

# 完了メッセージ
echo ""
echo "========================================="
echo "  Dev Container is ready!"
echo "  Run 'npm run dev' to start developing"
echo "========================================="
```

---

## アンチパターン

### アンチパターン 1: ベースイメージの肥大化

```jsonc
// NG: 不要なツールを大量にインストール
{
  "image": "ubuntu:22.04",
  "postCreateCommand": "apt-get update && apt-get install -y nodejs npm python3 python3-pip golang-go rustc ruby default-jdk php dotnet-sdk-7.0 && npm install -g yarn pnpm tsx typescript..."
}

// OK: 必要最小限の公式 Dev Container イメージ + Features
{
  "image": "mcr.microsoft.com/devcontainers/typescript-node:20-bookworm",
  "features": {
    "ghcr.io/devcontainers/features/github-cli:1": {}
  }
}
```

**問題点**: ベースイメージの肥大化はビルド時間の増大、ディスク容量の浪費、セキュリティリスクの増加を招く。Dev Container Features を使えば、必要なツールをモジュラーに追加できる。

### アンチパターン 2: node_modules のバインドマウント

```jsonc
// NG: node_modules をホストと共有 (パフォーマンス最悪)
{
  "mounts": []
  // デフォルトでプロジェクト全体がバインドマウントされ、
  // node_modules もホスト-コンテナ間で同期される
}

// OK: node_modules を名前付きボリュームに分離
{
  "mounts": [
    "source=myproject-node_modules,target=/workspace/node_modules,type=volume"
  ]
}
```

**問題点**: macOS / Windows ではバインドマウントの I/O 性能が Linux ネイティブに比べ大幅に低い。`node_modules` のような大量の小ファイルをバインドマウントすると `npm install` やビルドが10倍以上遅くなることがある。

---

## FAQ

### Q1: Dev Container はチーム全員が VS Code を使っていないと利用できませんか？

**A**: いいえ。Dev Container は Open Specification (devcontainers.github.io) として公開されており、VS Code 以外にも JetBrains の IntelliJ / WebStorm (Gateway 経由)、Neovim (devcontainer CLI 経由)、GitHub Codespaces (ブラウザ)、DevPod などがサポートしている。`devcontainer` CLI を直接使えば、任意のエディタと組み合わせることも可能。

### Q2: Dev Container 内から Docker コマンドを使いたい場合はどうすればよいですか？

**A**: `docker-in-docker` Feature を使うのが最も簡単。これはコンテナ内に独立した Docker デーモンを起動する。もう一つの方法は `docker-outside-of-docker` Feature で、ホストの Docker ソケットをマウントする方式。前者はクリーンだがリソース消費が大きく、後者は軽量だがホストのコンテナが見える。CI/CD テストでコンテナをビルドする場合は `docker-in-docker` が推奨。

### Q3: Codespaces の Prebuild はどのくらい起動時間を短縮できますか？

**A**: プロジェクトの規模によるが、一般的に `npm ci` + DB マイグレーションに 3-5 分かかるプロジェクトの場合、Prebuild を設定すると起動が 30 秒以内に短縮される。Prebuild は指定したブランチへのプッシュ時に自動実行され、事前にコンテナイメージと依存関係をキャッシュする。大規模 monorepo では特に効果が大きい。

---

## まとめ

| 項目 | 要点 |
|------|------|
| devcontainer.json | コンテナベース開発環境の宣言的定義ファイル |
| ベースイメージ | mcr.microsoft.com/devcontainers/ の公式イメージを推奨 |
| Features | ツールのモジュラーインストール機構。Dockerfile の代替 |
| Docker Compose 連携 | DB・Redis 等の依存サービスをまとめて管理 |
| VS Code 統合 | 拡張機能・設定をコンテナ内で一括管理 |
| GitHub Codespaces | クラウドベースの Dev Container。Prebuild で高速起動 |
| パフォーマンス | node_modules は Volume マウントで I/O 性能を確保 |
| マルチエディタ | devcontainer CLI で VS Code 以外からも利用可能 |

## 次に読むべきガイド

- [ローカルサービスの Docker 化](./02-local-services.md) -- PostgreSQL / Redis / MailHog の Docker 構成
- [プロジェクト標準](../03-team-setup/00-project-standards.md) -- EditorConfig / .npmrc / .nvmrc のチーム標準化
- [オンボーディング自動化](../03-team-setup/01-onboarding-automation.md) -- 新メンバーのセットアップ自動化

## 参考文献

1. **Dev Container Specification** -- https://containers.dev/ -- Dev Container の公式仕様と Features レジストリ
2. **VS Code Dev Containers ドキュメント** -- https://code.visualstudio.com/docs/devcontainers/containers -- VS Code での Dev Container の使い方
3. **GitHub Codespaces ドキュメント** -- https://docs.github.com/en/codespaces -- Codespaces の設定・Prebuild・課金の詳細
4. **Dev Container Features 一覧** -- https://github.com/devcontainers/features -- 公式 Features リポジトリとカスタム Feature の作成方法

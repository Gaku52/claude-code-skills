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

Dev Container は、開発環境全体をコンテナとして定義する仕組みである。従来の「README にインストール手順を書いて各自実行する」アプローチでは、OS の違い、ツールのバージョン差、設定のドリフトによって環境差異が避けられなかった。Dev Container はこの問題を根本的に解決する。

Dev Container の主な利点は以下の通りである。

1. **環境の再現性**: devcontainer.json にすべての環境定義が含まれるため、誰がいつビルドしても同じ環境が得られる
2. **オンボーディングの高速化**: 新メンバーは「リポジトリをクローンして Dev Container を開く」だけで開発を開始できる
3. **環境のバージョン管理**: .devcontainer/ ディレクトリがリポジトリに含まれるため、環境変更の履歴を Git で追跡できる
4. **ホストマシンの汚染防止**: すべてのツール・依存関係がコンテナ内に閉じるため、ホストマシンをクリーンに保てる
5. **マルチプロジェクト対応**: プロジェクトごとに異なるランタイムバージョンを利用しても競合しない

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

### 1.3 Dev Container の動作フロー

Dev Container のライフサイクルを理解することで、適切なタイミングで設定やコマンドを配置できる。

```
+------------------------------------------------------------------+
|          Dev Container ライフサイクル                                |
+------------------------------------------------------------------+
|                                                                  |
|  1. "Reopen in Container" (VS Code) or "devcontainer up" (CLI)   |
|     |                                                            |
|     v                                                            |
|  2. Docker イメージのビルド (Dockerfile / image / compose)         |
|     |  - ベースイメージの取得                                      |
|     |  - Features のインストール (install.sh 実行)                 |
|     |  - カスタム Dockerfile のビルド                               |
|     v                                                            |
|  3. コンテナの作成と起動                                           |
|     |  - ボリュームのマウント                                      |
|     |  - ポートフォワーディング設定                                 |
|     |  - 環境変数の注入                                            |
|     v                                                            |
|  4. initializeCommand (ホスト側で実行)                             |
|     |                                                            |
|     v                                                            |
|  5. onCreateCommand (初回作成時のみ)                               |
|     |                                                            |
|     v                                                            |
|  6. updateContentCommand (作成時 + Rebuild 時)                     |
|     |                                                            |
|     v                                                            |
|  7. postCreateCommand (作成完了後)                                 |
|     |                                                            |
|     v                                                            |
|  8. postStartCommand (起動ごとに実行)                              |
|     |                                                            |
|     v                                                            |
|  9. postAttachCommand (エディタ接続ごとに実行)                      |
|     |                                                            |
|     v                                                            |
|  10. VS Code Server 起動 + 拡張機能インストール                    |
|     → 開発準備完了                                                |
|                                                                  |
+------------------------------------------------------------------+
```

各ライフサイクルコマンドの使い分けは以下の通りである。

| コマンド | 実行タイミング | 用途の例 |
|---------|-------------|---------|
| `initializeCommand` | コンテナ作成前（ホスト側） | Git サブモジュールの初期化、.env ファイルの生成 |
| `onCreateCommand` | コンテナ初回作成時のみ | 大規模な依存関係のインストール、DB 初期化 |
| `updateContentCommand` | 作成時 + コンテンツ更新時 | `npm ci`、Prebuild で実行される |
| `postCreateCommand` | コンテナ作成完了後 | DB マイグレーション、Git フック設定 |
| `postStartCommand` | コンテナ起動ごと | バックグラウンドサービス起動、キャッシュウォーム |
| `postAttachCommand` | エディタ接続ごと | ウェルカムメッセージ、環境状態の表示 |

### 1.4 Dev Container 導入判断フローチャート

```
+------------------------------------------------------------------+
|        Dev Container 導入判断フロー                                 |
+------------------------------------------------------------------+
|                                                                  |
|  チーム開発か？                                                    |
|    |                                                             |
|    +--[Yes]--> 環境差異の問題が発生しているか？                     |
|    |             |                                                |
|    |             +--[Yes]--> Dev Container を導入                  |
|    |             |                                                |
|    |             +--[No]---> 新メンバーの参加頻度は？               |
|    |                          |                                   |
|    |                          +--[高い]--> Dev Container 推奨     |
|    |                          |                                   |
|    |                          +--[低い]--> Docker Compose のみ    |
|    |                                                              |
|    +--[No]----> 複数プロジェクトを同時開発か？                      |
|                   |                                               |
|                   +--[Yes]--> Dev Container 推奨                  |
|                   |                                               |
|                   +--[No]---> ローカル開発 or Docker Compose       |
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

### 2.3 マルチステージ Dockerfile パターン

大規模プロジェクトでは、Dev Container 用の Dockerfile をマルチステージビルドで構成すると効率的である。

```dockerfile
# .devcontainer/Dockerfile (マルチステージ)

# ステージ1: システム依存関係
FROM mcr.microsoft.com/devcontainers/typescript-node:20-bookworm AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client-16 \
    redis-tools \
    jq \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# ステージ2: CLI ツール群
FROM base AS tools

# Terraform
RUN curl -fsSL https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /usr/share/keyrings/hashicorp.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/hashicorp.gpg] https://apt.releases.hashicorp.com bookworm main" \
    > /etc/apt/sources.list.d/hashicorp.list \
    && apt-get update && apt-get install -y terraform \
    && rm -rf /var/lib/apt/lists/*

# AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip && ./aws/install && rm -rf aws awscliv2.zip

# gcloud CLI (オプション)
# RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/gcloud.tar.gz \
#     && mkdir -p /usr/local/gcloud && tar -C /usr/local/gcloud -xf /tmp/gcloud.tar.gz \
#     && /usr/local/gcloud/google-cloud-sdk/install.sh --quiet

# ステージ3: 最終イメージ
FROM tools AS devcontainer

# Node.js グローバルパッケージ (node ユーザーで)
RUN su node -c "npm install -g \
    tsx \
    prisma \
    @biomejs/biome \
    turbo \
    wrangler"

# シェル設定
COPY .devcontainer/.zshrc /home/node/.zshrc
COPY .devcontainer/.p10k.zsh /home/node/.p10k.zsh
RUN chown -R node:node /home/node

# ワークスペース
WORKDIR /workspace
```

### 2.4 Docker Compose との連携

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

### 2.5 devcontainer.json の高度な設定オプション

#### runArgs によるコンテナ実行オプション

```jsonc
{
  // Docker run に渡す追加引数
  "runArgs": [
    "--cap-add=SYS_PTRACE",   // デバッガのアタッチに必要
    "--security-opt", "seccomp=unconfined",  // パフォーマンスプロファイリング
    "--name", "my-devcontainer",
    "--network", "host",       // ホストネットワークモード (Linux のみ)
    "--gpus", "all"            // GPU アクセス (ML 開発)
  ]
}
```

#### containerEnv vs remoteEnv の違い

```jsonc
{
  // containerEnv: コンテナ全体で有効（ライフサイクルコマンドにも適用）
  "containerEnv": {
    "TZ": "Asia/Tokyo",
    "LANG": "ja_JP.UTF-8",
    "DOCKER_BUILDKIT": "1"
  },

  // remoteEnv: VS Code のターミナルプロセスでのみ有効
  "remoteEnv": {
    "NODE_ENV": "development",
    "DATABASE_URL": "postgresql://postgres:postgres@db:5432/myapp_dev",
    // ホストの環境変数を参照
    "LOCAL_USER": "${localEnv:USER}",
    // コンテナ内の環境変数を参照
    "PATH": "${containerEnv:PATH}:/workspace/scripts"
  }
}
```

#### 複数設定ファイルの管理

```
.devcontainer/
├── devcontainer.json          # デフォルト設定
├── Dockerfile
├── docker-compose.yml
├── post-create.sh
├── .zshrc
└── variants/
    ├── gpu/
    │   └── devcontainer.json  # GPU 開発用
    ├── minimal/
    │   └── devcontainer.json  # 軽量版
    └── full/
        └── devcontainer.json  # フル構成
```

VS Code のコマンドパレットで「Dev Containers: Open Folder in Container...」を実行すると、複数の devcontainer.json がある場合に選択ダイアログが表示される。

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

Features は OCI (Open Container Initiative) 仕様に準拠したパッケージとして配布される。各 Feature は以下のファイルで構成される。

```
my-feature/
├── devcontainer-feature.json   # Feature のメタデータとオプション定義
├── install.sh                  # インストールスクリプト
└── README.md                   # ドキュメント
```

### 3.2 よく使う Features 一覧

| Feature | 用途 | 設定例 |
|---------|------|--------|
| `node` | Node.js ランタイム | `"version": "20"` |
| `python` | Python ランタイム | `"version": "3.11"` |
| `go` | Go ランタイム | `"version": "1.21"` |
| `rust` | Rust ツールチェーン | `"profile": "default"` |
| `java` | Java JDK | `"version": "21"`, `"installGradle": true` |
| `docker-in-docker` | コンテナ内 Docker | デフォルトでOK |
| `docker-outside-of-docker` | ホスト Docker 共有 | 軽量だがホストのコンテナが見える |
| `github-cli` | gh コマンド | デフォルトでOK |
| `aws-cli` | AWS CLI v2 | デフォルトでOK |
| `terraform` | Terraform | `"version": "latest"` |
| `kubectl-helm-minikube` | K8s ツール群 | デフォルトでOK |
| `git` | 最新 Git | デフォルトでOK |
| `git-lfs` | Git Large File Storage | デフォルトでOK |
| `common-utils` | zsh, Oh My Zsh 等 | `"installZsh": true` |
| `sshd` | SSH サーバー | リモートデバッグ用 |

### 3.3 カスタム Feature の作成

チーム固有のツールや設定を Feature として配布できる。

```jsonc
// my-org-feature/devcontainer-feature.json
{
  "id": "my-org-tools",
  "version": "1.0.0",
  "name": "My Organization Development Tools",
  "description": "社内標準の開発ツールセット",
  "options": {
    "installLinter": {
      "type": "boolean",
      "default": true,
      "description": "社内カスタムリンターをインストールする"
    },
    "environment": {
      "type": "string",
      "enum": ["staging", "production"],
      "default": "staging",
      "description": "接続先環境"
    }
  }
}
```

```bash
#!/bin/bash
# my-org-feature/install.sh
set -e

# オプションの取得
INSTALL_LINTER="${INSTALLLINTER:-true}"
ENVIRONMENT="${ENVIRONMENT:-staging}"

echo "Installing My Org Tools..."

# 社内 CLI ツール
curl -fsSL https://internal.example.com/cli/install.sh | bash

# カスタムリンター
if [ "$INSTALL_LINTER" = "true" ]; then
    npm install -g @my-org/linter@latest
fi

# 環境設定ファイル
cat > /etc/my-org/config.json << EOF
{
  "environment": "$ENVIRONMENT",
  "apiEndpoint": "https://${ENVIRONMENT}.api.example.com"
}
EOF

echo "My Org Tools installed successfully!"
```

Feature を GitHub Container Registry に公開する場合の GitHub Actions は以下の通りである。

```yaml
# .github/workflows/publish-feature.yml
name: Publish Dev Container Feature

on:
  push:
    tags:
      - 'v*'

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4

      - name: Publish Feature
        uses: devcontainers/action@v1
        with:
          publish-features: true
          base-path-to-features: ./features
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

公開後、他のプロジェクトから以下のように参照できる。

```jsonc
{
  "features": {
    "ghcr.io/my-org/devcontainer-features/my-org-tools:1": {
      "installLinter": true,
      "environment": "staging"
    }
  }
}
```

### 3.4 Features のインストール順序と競合

Features は `devcontainer.json` に記載された順序でインストールされる。順序が重要になるケースがある。

```jsonc
{
  "features": {
    // 1. まず common-utils で zsh をインストール
    "ghcr.io/devcontainers/features/common-utils:2": {
      "installZsh": true,
      "configureZshAsDefaultShell": true
    },
    // 2. 次に Node.js をインストール（zsh の PATH に追加される）
    "ghcr.io/devcontainers/features/node:1": {
      "version": "20"
    },
    // 3. 最後に Docker-in-Docker（Docker デーモン起動）
    "ghcr.io/devcontainers/features/docker-in-docker:2": {
      "dockerDashComposeVersion": "v2"
    }
  }
}
```

Features 間で競合が発生する場合は `overrideFeatureInstallOrder` で制御できる。

```jsonc
{
  "overrideFeatureInstallOrder": [
    "ghcr.io/devcontainers/features/common-utils",
    "ghcr.io/devcontainers/features/node",
    "ghcr.io/devcontainers/features/docker-in-docker"
  ]
}
```

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

Prebuild を最大限活用するための構成は以下の通りである。

```jsonc
// .devcontainer/devcontainer.json (Prebuild 最適化)
{
  "name": "My Project (Prebuild Optimized)",
  "build": {
    "dockerfile": "Dockerfile",
    "context": "..",
    "cacheFrom": "ghcr.io/my-org/my-project-devcontainer:latest"
  },

  // updateContentCommand: Prebuild 時に実行される
  // → 依存関係のインストールをここで行う
  "updateContentCommand": {
    "install": "npm ci",
    "prisma": "npx prisma generate",
    "build-libs": "npm run build:libs"
  },

  // postCreateCommand: 初回作成時のみ
  // → DB マイグレーションなど環境固有の処理
  "postCreateCommand": {
    "migrate": "npm run db:migrate",
    "seed": "npm run db:seed",
    "hooks": "npx husky install"
  },

  // postStartCommand: 起動ごと
  // → 短時間で完了するタスクのみ
  "postStartCommand": "npm run dev:prepare"
}
```

### 4.3 Codespaces のコスト管理

Codespaces の課金は「コンピュートコスト（稼働時間）」と「ストレージコスト（保持時間）」の2軸で発生する。

```
+------------------------------------------------------------------+
|           Codespaces コスト管理                                    |
+------------------------------------------------------------------+
|                                                                  |
|  [コンピュートコスト]                                              |
|  マシンタイプ    | コア | RAM   | 時間単価 (目安)                  |
|  ───────────────┼──────┼───────┼──────────────────                |
|  2-core         | 2    | 8 GB  | $0.18/hr                       |
|  4-core         | 4    | 16 GB | $0.36/hr                       |
|  8-core         | 8    | 32 GB | $0.72/hr                       |
|  16-core        | 16   | 64 GB | $1.44/hr                       |
|  32-core        | 32   | 128GB | $2.88/hr                       |
|                                                                  |
|  [ストレージコスト]                                                |
|  $0.07/GB/月 (Prebuild スナップショット含む)                       |
|                                                                  |
|  [Free 枠 (Personal)]                                            |
|  - 120 コア時間/月 (2-core で 60 時間)                            |
|  - 15 GB ストレージ/月                                            |
|                                                                  |
+------------------------------------------------------------------+
```

コスト最適化のベストプラクティスは以下の通りである。

```jsonc
// リポジトリの .devcontainer/devcontainer.json
{
  // 最小限のマシンスペックを指定
  "hostRequirements": {
    "cpus": 2,        // フロントエンド開発なら 2 コアで十分
    "memory": "8gb"
  }
}
```

Organization レベルのポリシー設定（GitHub リポジトリの Settings > Codespaces）で以下を推奨する。

| 設定 | 推奨値 | 説明 |
|------|-------|------|
| Default idle timeout | 30 分 | 無操作時の自動停止 |
| Maximum idle timeout | 60 分 | ユーザーが設定できる最大値 |
| Retention period | 14 日 | 未使用 Codespace の自動削除 |
| Machine type policy | 2-core, 4-core | 利用可能なマシンタイプの制限 |
| Prebuild regions | 1 リージョン | 不要なリージョンの Prebuild を削減 |

### 4.4 ローカル Dev Container vs Codespaces 比較

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
| GPU 利用 | ホスト GPU パススルー | GPU マシン選択可 |
| コラボレーション | 不可 | Live Share 統合可 |
| セキュリティ | ローカルポリシー | Organization ポリシー |

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

### 5.2 Python + FastAPI テンプレート

```jsonc
// .devcontainer/devcontainer.json (Python FastAPI)
{
  "name": "Python FastAPI Dev",
  "build": {
    "dockerfile": "Dockerfile",
    "context": ".."
  },

  "features": {
    "ghcr.io/devcontainers/features/python:1": {
      "version": "3.12"
    },
    "ghcr.io/devcontainers/features/github-cli:1": {},
    "ghcr.io/devcontainers/features/docker-in-docker:2": {}
  },

  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "charliermarsh.ruff",
        "ms-python.mypy-type-checker",
        "mtxr.sqltools",
        "mtxr.sqltools-driver-pg",
        "humao.rest-client"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/workspace/.venv/bin/python",
        "python.terminal.activateEnvironment": true,
        "[python]": {
          "editor.defaultFormatter": "charliermarsh.ruff",
          "editor.formatOnSave": true,
          "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
          }
        }
      }
    }
  },

  "forwardPorts": [8000, 5432],
  "portsAttributes": {
    "8000": { "label": "FastAPI", "onAutoForward": "openBrowser" },
    "5432": { "label": "PostgreSQL", "onAutoForward": "silent" }
  },

  "postCreateCommand": "pip install -e '.[dev]' && alembic upgrade head",
  "postStartCommand": "uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &",
  "remoteUser": "vscode"
}
```

対応する Dockerfile は以下の通りである。

```dockerfile
# .devcontainer/Dockerfile (Python)
FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

# システムパッケージ
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client-16 \
    && rm -rf /var/lib/apt/lists/*

# uv (高速 pip 代替)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Poetry (オプション)
# RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /workspace
```

### 5.3 Go + gRPC テンプレート

```jsonc
// .devcontainer/devcontainer.json (Go gRPC)
{
  "name": "Go gRPC Dev",
  "image": "mcr.microsoft.com/devcontainers/go:1.22-bookworm",

  "features": {
    "ghcr.io/devcontainers/features/docker-in-docker:2": {},
    "ghcr.io/devcontainers/features/github-cli:1": {},
    "ghcr.io/devcontainers-contrib/features/protoc:1": {}
  },

  "customizations": {
    "vscode": {
      "extensions": [
        "golang.go",
        "zxh404.vscode-proto3",
        "ms-azuretools.vscode-docker"
      ],
      "settings": {
        "go.toolsManagement.autoUpdate": true,
        "go.lintTool": "golangci-lint",
        "go.lintFlags": ["--fast"],
        "[go]": {
          "editor.formatOnSave": true,
          "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
          }
        },
        "gopls": {
          "ui.semanticTokens": true,
          "formatting.gofumpt": true
        }
      }
    }
  },

  "postCreateCommand": "go mod download && go install google.golang.org/protobuf/cmd/protoc-gen-go@latest && go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest",
  "forwardPorts": [8080, 50051],
  "portsAttributes": {
    "8080": { "label": "HTTP API", "onAutoForward": "openBrowser" },
    "50051": { "label": "gRPC", "onAutoForward": "silent" }
  },
  "remoteUser": "vscode"
}
```

### 5.4 Monorepo テンプレート

Turborepo や Nx を使う monorepo プロジェクトでの Dev Container 設定は以下の通りである。

```jsonc
// .devcontainer/devcontainer.json (Monorepo)
{
  "name": "Monorepo Dev",
  "dockerComposeFile": "docker-compose.yml",
  "service": "app",
  "workspaceFolder": "/workspace",

  "features": {
    "ghcr.io/devcontainers/features/docker-in-docker:2": {},
    "ghcr.io/devcontainers/features/github-cli:1": {}
  },

  "customizations": {
    "vscode": {
      "extensions": [
        "dbaeumer.vscode-eslint",
        "esbenp.prettier-vscode",
        "prisma.prisma",
        "bradlc.vscode-tailwindcss",
        "ms-vscode.vscode-typescript-next",
        "biomejs.biome"
      ],
      "settings": {
        "editor.formatOnSave": true,
        "eslint.workingDirectories": [
          { "pattern": "apps/*" },
          { "pattern": "packages/*" }
        ],
        "typescript.tsdk": "node_modules/typescript/lib",
        "search.exclude": {
          "**/node_modules": true,
          "**/.turbo": true,
          "**/dist": true
        }
      }
    }
  },

  "forwardPorts": [3000, 3001, 5432, 6379],
  "portsAttributes": {
    "3000": { "label": "Web App", "onAutoForward": "openBrowser" },
    "3001": { "label": "Admin", "onAutoForward": "notify" },
    "5432": { "label": "PostgreSQL", "onAutoForward": "silent" },
    "6379": { "label": "Redis", "onAutoForward": "silent" }
  },

  // monorepo 用: pnpm + turbo のインストール
  "postCreateCommand": "corepack enable && pnpm install && pnpm turbo run build --filter='./packages/*'",
  "postStartCommand": "pnpm turbo run dev --filter='./apps/*' &",
  "remoteUser": "node",

  // monorepo で重要: node_modules を Volume にする
  "mounts": [
    "source=monorepo-node_modules,target=/workspace/node_modules,type=volume",
    "source=monorepo-turbo-cache,target=/workspace/.turbo,type=volume"
  ]
}
```

### 5.5 セットアップスクリプト

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

### 5.6 高度なセットアップスクリプト（Monorepo 対応）

```bash
#!/bin/bash
# .devcontainer/post-create.sh (Monorepo 対応版)

set -euo pipefail

# 色付き出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "============================================"
echo "   Dev Container Post-Create Setup"
echo "============================================"
echo ""

# 1. パッケージマネージャの設定
log_info "Configuring package manager..."
if command -v corepack &> /dev/null; then
    corepack enable
    log_ok "Corepack enabled"
fi

# 2. 依存関係のインストール
log_info "Installing dependencies..."
if [ -f "pnpm-lock.yaml" ]; then
    pnpm install --frozen-lockfile
    log_ok "pnpm install completed"
elif [ -f "yarn.lock" ]; then
    yarn install --frozen-lockfile
    log_ok "yarn install completed"
elif [ -f "package-lock.json" ]; then
    npm ci
    log_ok "npm ci completed"
else
    log_warn "No lockfile found, running npm install"
    npm install
fi

# 3. データベース準備
log_info "Setting up database..."
MAX_RETRIES=30
RETRY_COUNT=0
until pg_isready -h db -U postgres -q 2>/dev/null; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        log_error "Database connection timeout after ${MAX_RETRIES} attempts"
        exit 1
    fi
    log_info "Waiting for database... (${RETRY_COUNT}/${MAX_RETRIES})"
    sleep 1
done
log_ok "Database is ready"

# マイグレーション
if [ -f "prisma/schema.prisma" ]; then
    npx prisma migrate deploy
    npx prisma generate
    log_ok "Prisma migrations applied"
elif [ -d "alembic" ]; then
    alembic upgrade head
    log_ok "Alembic migrations applied"
fi

# シードデータ
if npm run --silent seed 2>/dev/null; then
    log_ok "Database seeded"
else
    log_warn "No seed script found, skipping"
fi

# 4. Git 設定
log_info "Configuring Git..."
git config --global --add safe.directory /workspace
if [ -f ".husky/_/husky.sh" ]; then
    npx husky install
    log_ok "Git hooks configured"
fi

# 5. 環境変数チェック
log_info "Checking environment..."
REQUIRED_VARS=("DATABASE_URL" "NODE_ENV")
MISSING_VARS=()
for VAR in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!VAR:-}" ]; then
        MISSING_VARS+=("$VAR")
    fi
done
if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    log_warn "Missing environment variables: ${MISSING_VARS[*]}"
else
    log_ok "All required environment variables set"
fi

# 6. 完了
echo ""
echo "============================================"
echo -e "  ${GREEN}Dev Container is ready!${NC}"
echo ""
echo "  Available commands:"
echo "    npm run dev      - Start development server"
echo "    npm run test     - Run tests"
echo "    npm run lint     - Run linter"
echo "    npm run build    - Build for production"
echo "============================================"
```

---

## 6. devcontainer CLI

### 6.1 CLI のインストールと基本操作

VS Code を使わずに Dev Container を操作したい場合は `devcontainer` CLI を使用する。

```bash
# CLI のインストール
npm install -g @devcontainers/cli

# Dev Container の起動
devcontainer up --workspace-folder .

# コンテナ内でコマンドを実行
devcontainer exec --workspace-folder . npm run test

# Dev Container のビルドのみ
devcontainer build --workspace-folder .

# Features のテスト
devcontainer features test --features ./my-feature
```

### 6.2 CI/CD での活用

Dev Container を CI/CD パイプラインで活用することで、開発環境と CI 環境を完全に一致させることができる。

```yaml
# .github/workflows/ci.yml
name: CI (Dev Container)

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Dev Container CLI
        run: npm install -g @devcontainers/cli

      - name: Build Dev Container
        run: devcontainer build --workspace-folder .

      - name: Run Tests in Dev Container
        run: devcontainer exec --workspace-folder . npm run test

      - name: Run Lint in Dev Container
        run: devcontainer exec --workspace-folder . npm run lint

      - name: Run Type Check
        run: devcontainer exec --workspace-folder . npm run type-check
```

より効率的な方法として、Dev Container をキャッシュ付きでビルドする。

```yaml
# .github/workflows/ci-cached.yml
name: CI (Dev Container with Cache)

on:
  push:
    branches: [main]
  pull_request:

jobs:
  build-devcontainer:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Pre-build Dev Container
        uses: devcontainers/ci@v0.3
        with:
          imageName: ghcr.io/${{ github.repository }}/devcontainer
          cacheFrom: ghcr.io/${{ github.repository }}/devcontainer:latest
          push: always

  test:
    needs: build-devcontainer
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Tests
        uses: devcontainers/ci@v0.3
        with:
          imageName: ghcr.io/${{ github.repository }}/devcontainer
          cacheFrom: ghcr.io/${{ github.repository }}/devcontainer:latest
          push: never
          runCmd: |
            npm run test
            npm run lint
            npm run type-check
```

---

## 7. マルチエディタ対応

### 7.1 JetBrains IDEs (Gateway)

JetBrains Gateway を使うと、IntelliJ IDEA / WebStorm / PyCharm などから Dev Container に接続できる。

```jsonc
// .devcontainer/devcontainer.json (JetBrains 対応)
{
  "name": "My Project",
  "image": "mcr.microsoft.com/devcontainers/typescript-node:20",

  "customizations": {
    // VS Code 設定
    "vscode": {
      "extensions": ["dbaeumer.vscode-eslint"]
    },
    // JetBrains 設定
    "jetbrains": {
      "backend": "WebStorm",
      "plugins": [
        "com.intellij.plugins.tailwindcss"
      ]
    }
  }
}
```

### 7.2 Neovim / ターミナルエディタ

```bash
# devcontainer CLI でコンテナを起動
devcontainer up --workspace-folder .

# コンテナ内で Neovim を起動
devcontainer exec --workspace-folder . nvim

# または docker exec で直接接続
docker exec -it $(docker ps -q --filter "label=devcontainer.local_folder=$(pwd)") bash
```

Neovim ユーザー向けの Features 設定は以下の通りである。

```jsonc
{
  "features": {
    "ghcr.io/devcontainers/features/common-utils:2": {
      "installZsh": true
    },
    "ghcr.io/rio/features/neovim:1": {
      "version": "stable"
    }
  },
  "postCreateCommand": "mkdir -p /home/node/.config && ln -sf /workspace/.devcontainer/nvim /home/node/.config/nvim"
}
```

### 7.3 DevPod

DevPod はオープンソースの Dev Container クライアントであり、任意のバックエンド（ローカル Docker、SSH、Kubernetes、クラウド）で Dev Container を実行できる。

```bash
# DevPod のインストール (macOS)
brew install loft-sh/tap/devpod

# ローカルの Docker で Dev Container を起動
devpod up . --provider docker

# SSH 経由のリモートマシンで実行
devpod up . --provider ssh --option HOST=dev.example.com

# Kubernetes クラスタで実行
devpod up . --provider kubernetes
```

---

## 8. パフォーマンス最適化

### 8.1 ビルド時間の短縮

```jsonc
// .devcontainer/devcontainer.json
{
  "build": {
    "dockerfile": "Dockerfile",
    "context": "..",
    // ビルドキャッシュの活用
    "cacheFrom": "ghcr.io/my-org/my-project-devcontainer:latest",
    "args": {
      "BUILDKIT_INLINE_CACHE": "1"
    }
  }
}
```

```dockerfile
# .devcontainer/Dockerfile (キャッシュ最適化)
FROM mcr.microsoft.com/devcontainers/typescript-node:20-bookworm

# 変更頻度の低いレイヤーを先に
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client-16 \
    redis-tools \
    jq \
    && rm -rf /var/lib/apt/lists/*

# グローバル npm パッケージも別レイヤーに
RUN su node -c "npm install -g tsx prisma @biomejs/biome turbo"

# プロジェクト依存関係は postCreateCommand で処理
# （Dockerfile に含めるとソース変更のたびに再ビルドになる）
```

### 8.2 I/O パフォーマンスの改善

macOS / Windows では Docker のファイルシステムブリッジがボトルネックになる。以下の方法で改善できる。

```jsonc
{
  // 1. node_modules を Volume にする（最も効果的）
  "mounts": [
    "source=myproject-node_modules,target=/workspace/node_modules,type=volume",
    "source=myproject-turbo-cache,target=/workspace/.turbo,type=volume",
    "source=myproject-next-cache,target=/workspace/.next,type=volume"
  ],

  // 2. ワークスペースの consistency を cached に
  // (docker-compose.yml で設定)
  // volumes:
  //   - ..:/workspace:cached

  // 3. 不要なファイルの同期を除外
  // (VS Code 設定)
  "customizations": {
    "vscode": {
      "settings": {
        "files.watcherExclude": {
          "**/node_modules/**": true,
          "**/.git/objects/**": true,
          "**/.turbo/**": true,
          "**/dist/**": true
        }
      }
    }
  }
}
```

### 8.3 OrbStack の活用（macOS）

macOS では Docker Desktop の代わりに OrbStack を使うことで、Dev Container のパフォーマンスが大幅に向上する。

```bash
# OrbStack のインストール
brew install --cask orbstack

# OrbStack は Docker CLI 互換
# Dev Container は設定変更なしで動作する
docker version  # OrbStack の Docker エンジンが表示される
```

OrbStack の利点は以下の通りである。

| 比較項目 | Docker Desktop | OrbStack |
|---------|---------------|----------|
| メモリ消費 | 2-4 GB | 0.5-1 GB |
| CPU 使用率 (idle) | 5-15% | ~0% |
| ファイル I/O (macOS) | 遅い (VirtioFS) | 高速 (独自実装) |
| 起動時間 | 30-60 秒 | 2-5 秒 |
| ネットワーク | NAT | ネイティブに近い |
| `npm install` 速度 | 基準値 | 2-3x 高速 |

---

## 9. セキュリティ

### 9.1 シークレット管理

```jsonc
// .devcontainer/devcontainer.json
{
  // NG: ハードコードされた秘密情報
  // "remoteEnv": {
  //   "API_KEY": "sk-1234567890abcdef"
  // }

  // OK: ホストの環境変数を参照
  "remoteEnv": {
    "API_KEY": "${localEnv:API_KEY}",
    "DATABASE_URL": "${localEnv:DATABASE_URL}"
  },

  // OK: .env ファイルをバインドマウント (.gitignore に追加)
  "mounts": [
    "source=${localWorkspaceFolder}/.env.local,target=/workspace/.env.local,type=bind,consistency=cached"
  ]
}
```

### 9.2 コンテナの権限設定

```jsonc
{
  // 非 root ユーザーで実行
  "remoteUser": "node",

  // 必要最小限の capabilities
  "runArgs": [
    "--cap-drop=ALL",
    "--cap-add=SYS_PTRACE",   // デバッガに必要
    "--cap-add=NET_RAW"       // ネットワークツールに必要な場合
  ],

  // read-only ファイルシステム（高セキュリティ環境）
  // "runArgs": ["--read-only", "--tmpfs=/tmp"]
}
```

### 9.3 ネットワーク分離

```yaml
# .devcontainer/docker-compose.yml
services:
  app:
    build:
      context: ..
      dockerfile: .devcontainer/Dockerfile
    networks:
      - dev-internal  # 内部ネットワークのみ

  db:
    image: postgres:16-alpine
    networks:
      - dev-internal
    # ports は公開しない（コンテナ間通信のみ）

networks:
  dev-internal:
    internal: true  # 外部アクセスを遮断
```

---

## 10. トラブルシューティング

### 10.1 よくある問題と対処法

| 問題 | 原因 | 対処法 |
|------|------|--------|
| コンテナが起動しない | Dockerfile のビルドエラー | `devcontainer build` でエラーログ確認 |
| 権限エラー (EACCES) | UID/GID の不一致 | `remoteUser` の設定確認、Volume の権限修正 |
| npm install が遅い | node_modules のバインドマウント | Volume マウントに切り替え |
| ポートフォワードが効かない | ポート競合 | `lsof -i :PORT` で確認、別ポートに変更 |
| Git が認証失敗 | SSH キーの未マウント | `ssh-agent` Feature を使用 |
| 拡張機能が動かない | コンテナ内に未インストール | `extensions` に追加して Rebuild |
| Rebuild で設定が反映されない | キャッシュ | `Dev Containers: Rebuild Without Cache` |
| Volume データが残る | 古い Volume | `docker volume prune` で削除 |
| WSL2 で遅い | Windows のファイルシステム | WSL2 内にリポジトリを配置 |
| Apple Silicon で動かない | amd64 イメージ | `--platform linux/arm64` を指定 |

### 10.2 デバッグ手順

```bash
# 1. Dev Container CLI でビルドエラーを確認
devcontainer build --workspace-folder . --log-level trace 2>&1 | tee build.log

# 2. コンテナの状態を確認
docker ps -a --filter "label=devcontainer.local_folder"

# 3. コンテナのログを確認
docker logs <container-id>

# 4. コンテナ内に直接接続
docker exec -it <container-id> bash

# 5. Volume の状態を確認
docker volume ls --filter "label=devcontainer"

# 6. ネットワークの状態を確認
docker network ls
docker network inspect <network-name>

# 7. リソース使用状況
docker stats --no-stream

# 8. Dev Container の設定を確認
devcontainer read-configuration --workspace-folder .
```

### 10.3 Apple Silicon (ARM64) 対応

Apple Silicon Mac では一部のイメージが amd64 のみ提供されている場合がある。

```jsonc
{
  "build": {
    "dockerfile": "Dockerfile",
    "args": {
      // プラットフォームを明示
      "TARGETPLATFORM": "linux/arm64"
    }
  },

  // Rosetta 2 エミュレーションで amd64 イメージを使用（遅い）
  // "runArgs": ["--platform", "linux/amd64"]
}
```

```dockerfile
# .devcontainer/Dockerfile (マルチアーキテクチャ)
FROM --platform=$BUILDPLATFORM mcr.microsoft.com/devcontainers/typescript-node:20-bookworm

ARG TARGETPLATFORM
ARG BUILDPLATFORM

# アーキテクチャに応じたバイナリのインストール
RUN case "$TARGETPLATFORM" in \
    "linux/arm64") ARCH="aarch64" ;; \
    "linux/amd64") ARCH="x86_64" ;; \
    *) echo "Unsupported platform: $TARGETPLATFORM" && exit 1 ;; \
    esac \
    && curl -fsSL "https://example.com/tool-${ARCH}.tar.gz" | tar xzf - -C /usr/local/bin/
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

### アンチパターン 3: postCreateCommand にすべてを詰め込む

```jsonc
// NG: 長い一行コマンド
{
  "postCreateCommand": "npm ci && npx prisma migrate deploy && npx prisma db seed && npx husky install && npm run build:libs && echo 'done'"
}

// OK: 外部スクリプトに分離
{
  "postCreateCommand": "bash .devcontainer/post-create.sh"
}

// さらに良い: ライフサイクルを分離
{
  "updateContentCommand": "npm ci",
  "postCreateCommand": {
    "migrate": "npx prisma migrate deploy",
    "seed": "npx prisma db seed",
    "hooks": "npx husky install"
  },
  "postStartCommand": "npm run dev:prepare"
}
```

**問題点**: 長い一行コマンドはデバッグが困難であり、一部のコマンドが失敗した場合にどこで止まったのかを特定しづらい。外部スクリプトに分離するか、オブジェクト形式で各タスクを名前付きで定義すると可視性が向上する。

### アンチパターン 4: Codespaces で大きなマシンタイプを常用

```jsonc
// NG: 必要以上に大きなマシンタイプ
{
  "hostRequirements": {
    "cpus": 16,
    "memory": "64gb"
  }
}

// OK: タスクに応じた最小限のスペック
{
  "hostRequirements": {
    "cpus": 4,
    "memory": "16gb"
  }
}
```

**問題点**: 大きなマシンタイプは時間単価が高い。フロントエンド開発は 2-core で十分であり、ビルドが遅い場合は Prebuild で対応する方がコスト効率が良い。

### アンチパターン 5: Feature の順序を考慮しない

```jsonc
// NG: Node.js Feature の前に common-utils をインストールしない
{
  "features": {
    "ghcr.io/devcontainers/features/node:1": { "version": "20" },
    "ghcr.io/devcontainers/features/common-utils:2": {
      "installZsh": true
    }
  }
}

// OK: common-utils を先にインストール
{
  "features": {
    "ghcr.io/devcontainers/features/common-utils:2": {
      "installZsh": true,
      "configureZshAsDefaultShell": true
    },
    "ghcr.io/devcontainers/features/node:1": { "version": "20" }
  }
}
```

**問題点**: Features のインストール順序によっては、PATH の設定やシェルの初期化スクリプトに問題が発生する。`common-utils` はシェル環境を構成するため、他の Features よりも先にインストールする。

---

## FAQ

### Q1: Dev Container はチーム全員が VS Code を使っていないと利用できませんか？

**A**: いいえ。Dev Container は Open Specification (devcontainers.github.io) として公開されており、VS Code 以外にも JetBrains の IntelliJ / WebStorm (Gateway 経由)、Neovim (devcontainer CLI 経由)、GitHub Codespaces (ブラウザ)、DevPod などがサポートしている。`devcontainer` CLI を直接使えば、任意のエディタと組み合わせることも可能。

### Q2: Dev Container 内から Docker コマンドを使いたい場合はどうすればよいですか？

**A**: `docker-in-docker` Feature を使うのが最も簡単。これはコンテナ内に独立した Docker デーモンを起動する。もう一つの方法は `docker-outside-of-docker` Feature で、ホストの Docker ソケットをマウントする方式。前者はクリーンだがリソース消費が大きく、後者は軽量だがホストのコンテナが見える。CI/CD テストでコンテナをビルドする場合は `docker-in-docker` が推奨。

### Q3: Codespaces の Prebuild はどのくらい起動時間を短縮できますか？

**A**: プロジェクトの規模によるが、一般的に `npm ci` + DB マイグレーションに 3-5 分かかるプロジェクトの場合、Prebuild を設定すると起動が 30 秒以内に短縮される。Prebuild は指定したブランチへのプッシュ時に自動実行され、事前にコンテナイメージと依存関係をキャッシュする。大規模 monorepo では特に効果が大きい。

### Q4: 既存プロジェクトに Dev Container を導入する際の手順は？

**A**: 以下の手順が推奨される。

1. VS Code のコマンドパレットで「Dev Containers: Add Dev Container Configuration Files...」を実行
2. プロジェクトのスタックに合ったテンプレートを選択（例: Node.js + TypeScript）
3. 必要な Features を追加（GitHub CLI、Docker-in-Docker 等）
4. 生成された `.devcontainer/devcontainer.json` をカスタマイズ
5. `postCreateCommand` にプロジェクト固有のセットアップを追加
6. チームメンバーにテスト依頼し、フィードバックを反映
7. `.devcontainer/` をリポジトリにコミット

### Q5: Dev Container で SSH 鍵を使うにはどうすればよいですか？

**A**: VS Code の Remote - Containers 拡張は、ホストの SSH Agent を自動的にコンテナに転送する。macOS / Linux では `ssh-add` でキーが Agent に登録されていれば、コンテナ内で透過的に使える。Windows では Pageant または OpenSSH Agent サービスを使用する。明示的に設定する場合は以下の通り。

```jsonc
{
  // SSH Agent の転送を明示
  "mounts": [
    "source=${localEnv:SSH_AUTH_SOCK},target=/ssh-agent,type=bind"
  ],
  "remoteEnv": {
    "SSH_AUTH_SOCK": "/ssh-agent"
  }
}
```

### Q6: Dev Container を使いつつ、ローカルの GPU にアクセスするには？

**A**: NVIDIA GPU の場合、`nvidia-container-toolkit` をホストにインストールした上で、以下の設定を行う。

```jsonc
{
  "runArgs": [
    "--gpus", "all"
  ],
  "features": {
    "ghcr.io/devcontainers/features/nvidia-cuda:1": {
      "installCudnn": true,
      "cudaVersion": "12.3"
    }
  }
}
```

### Q7: Codespaces の dotfiles リポジトリとは何ですか？

**A**: GitHub の Settings > Codespaces > Dotfiles repository で、自分の dotfiles リポジトリ（例: `username/dotfiles`）を指定すると、すべての Codespace 起動時にそのリポジトリが自動的にクローンされ、`install.sh` / `setup.sh` / `bootstrap.sh` のいずれかが実行される。`.zshrc`, `.gitconfig`, `.vimrc` などの個人設定をすべての Codespace に適用できる仕組みである。

---

## まとめ

| 項目 | 要点 |
|------|------|
| devcontainer.json | コンテナベース開発環境の宣言的定義ファイル |
| ベースイメージ | mcr.microsoft.com/devcontainers/ の公式イメージを推奨 |
| Features | ツールのモジュラーインストール機構。Dockerfile の代替 |
| カスタム Feature | OCI 準拠のパッケージとして社内配布可能 |
| Docker Compose 連携 | DB・Redis 等の依存サービスをまとめて管理 |
| VS Code 統合 | 拡張機能・設定をコンテナ内で一括管理 |
| JetBrains 対応 | Gateway 経由で IntelliJ / WebStorm から接続可能 |
| devcontainer CLI | VS Code なしで CLI からコンテナ操作・CI 統合 |
| GitHub Codespaces | クラウドベースの Dev Container。Prebuild で高速起動 |
| コスト管理 | マシンタイプの制限、idle timeout、Prebuild 最適化 |
| パフォーマンス | node_modules は Volume マウントで I/O 性能を確保 |
| OrbStack | macOS で Docker Desktop より高速軽量な代替 |
| マルチエディタ | devcontainer CLI / DevPod で VS Code 以外からも利用可能 |
| セキュリティ | 非 root 実行、capability 制限、ネットワーク分離 |
| Apple Silicon | ARM64 ネイティブイメージの使用を推奨 |

## 次に読むべきガイド

- [ローカルサービスの Docker 化](./02-local-services.md) -- PostgreSQL / Redis / MailHog の Docker 構成
- [プロジェクト標準](../03-team-setup/00-project-standards.md) -- EditorConfig / .npmrc / .nvmrc のチーム標準化
- [オンボーディング自動化](../03-team-setup/01-onboarding-automation.md) -- 新メンバーのセットアップ自動化

## 参考文献

1. **Dev Container Specification** -- https://containers.dev/ -- Dev Container の公式仕様と Features レジストリ
2. **VS Code Dev Containers ドキュメント** -- https://code.visualstudio.com/docs/devcontainers/containers -- VS Code での Dev Container の使い方
3. **GitHub Codespaces ドキュメント** -- https://docs.github.com/en/codespaces -- Codespaces の設定・Prebuild・課金の詳細
4. **Dev Container Features 一覧** -- https://github.com/devcontainers/features -- 公式 Features リポジトリとカスタム Feature の作成方法
5. **devcontainer CLI** -- https://github.com/devcontainers/cli -- CLI ツールのリポジトリとドキュメント
6. **DevPod** -- https://devpod.sh/ -- オープンソースの Dev Container クライアント
7. **OrbStack** -- https://orbstack.dev/ -- macOS 向け高速 Docker 実行環境

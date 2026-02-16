# オンボーディング自動化 (Onboarding Automation)

> 新メンバーが 1 コマンドで開発環境を構築できるセットアップスクリプトと Makefile を設計し、オンボーディングの時間を数日から数分に短縮する手法を学ぶ。

## この章で学ぶこと

1. **セットアップスクリプトの設計と実装** -- プラットフォーム差異を吸収し、依存ツールのインストールから初回ビルドまでを自動化するスクリプトを構築する
2. **Makefile によるタスクランナーの構築** -- よく使う開発タスクを `make` コマンドで標準化し、手順書の代わりにする
3. **環境検証と troubleshooting の自動化** -- セットアップの成否を自動検証し、問題発生時の診断情報を収集する仕組みを整備する
4. **マルチプラットフォーム対応のベストプラクティス** -- macOS / Linux / Windows (WSL2) の差異を吸収するクロスプラットフォームスクリプトの設計手法を習得する
5. **CI/CD でのセットアップスクリプト検証** -- セットアップスクリプトの陳腐化を防ぐ自動テストの仕組みを構築する

---

## 1. オンボーディングの課題

```
+------------------------------------------------------------------+
|              従来のオンボーディング vs 自動化                        |
+------------------------------------------------------------------+
|                                                                  |
|  [従来 - 手順書ベース]                                            |
|  1. Confluence の手順書を読む (30分)                               |
|  2. Homebrew をインストール (10分)                                 |
|  3. Node.js をインストール → バージョンが違う (30分)               |
|  4. npm install → エラー (1時間)                                  |
|  5. PostgreSQL をインストール → 設定がわからない (1時間)            |
|  6. 環境変数を設定 → 何を設定するかわからない (30分)               |
|  7. 先輩に質問 → 先輩の時間も消費 (2時間)                          |
|  合計: 1-2日                                                      |
|                                                                  |
|  [自動化]                                                        |
|  1. git clone && make setup                                      |
|  合計: 5-15分                                                     |
|                                                                  |
+------------------------------------------------------------------+
```

### 1.1 オンボーディング自動化の投資対効果

| 指標 | 手動 (年間) | 自動化 (年間) | 削減効果 |
|------|-----------|-------------|---------|
| 新メンバー1人あたりのセットアップ時間 | 8-16時間 | 0.5-1時間 | 90%削減 |
| 先輩エンジニアのサポート時間 | 4-8時間 | 0-0.5時間 | 95%削減 |
| 年間入社5人の場合の合計工数 | 60-120時間 | 2.5-7.5時間 | 95%削減 |
| セットアップ関連の問い合わせ件数 | 月10-20件 | 月0-2件 | 90%削減 |
| 環境不一致によるバグ | 月5-10件 | 月0-1件 | 90%削減 |
| 自動化スクリプトの初期構築コスト | -- | 8-16時間 | -- |
| 月次メンテナンスコスト | -- | 1-2時間 | -- |

### 1.2 オンボーディング自動化のアーキテクチャ

```
+------------------------------------------------------------------+
|              オンボーディング自動化の全体像                          |
+------------------------------------------------------------------+
|                                                                  |
|  [エントリーポイント]                                              |
|  make setup                                                      |
|    |                                                             |
|    v                                                             |
|  scripts/setup.sh                                                |
|    |                                                             |
|    +-- OS 検出 (macOS / Linux / WSL2)                            |
|    +-- 前提ツールチェック & インストール                            |
|    |   +-- Homebrew (macOS)                                      |
|    |   +-- git, curl, jq                                         |
|    |   +-- Docker Desktop / Docker Engine                        |
|    |   +-- Node.js バージョンマネージャ (fnm)                     |
|    +-- Node.js セットアップ (.nvmrc 準拠)                         |
|    +-- 依存関係インストール (pnpm install)                        |
|    +-- 環境変数セットアップ (.env.example → .env)                 |
|    +-- Docker サービス起動 (DB, Redis, etc.)                     |
|    +-- データベースマイグレーション & シード                       |
|    +-- ヘルスチェック & 検証                                      |
|    |                                                             |
|    v                                                             |
|  "Setup Complete! Run: make dev"                                 |
|                                                                  |
|  [日常運用]                                                       |
|  make dev      -- 開発サーバー起動                                |
|  make test     -- テスト実行                                      |
|  make lint     -- Lint チェック                                   |
|  make doctor   -- 環境診断                                        |
|  make help     -- コマンド一覧                                    |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 2. セットアップスクリプト

### 2.1 メインスクリプト

```bash
#!/bin/bash
# scripts/setup.sh
# プロジェクトの初回セットアップスクリプト

set -euo pipefail

# カラー出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# OS 検出
detect_os() {
  case "$(uname -s)" in
    Darwin*) echo "macos" ;;
    Linux*)  echo "linux" ;;
    MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
    *) echo "unknown" ;;
  esac
}

OS=$(detect_os)
log_info "OS detected: $OS"

# ======================================
# 1. 前提ツールのチェックとインストール
# ======================================
check_prerequisites() {
  log_info "前提条件をチェック中..."

  # Git
  if command -v git &>/dev/null; then
    log_ok "Git $(git --version | cut -d' ' -f3)"
  else
    log_error "Git が見つかりません"
    exit 1
  fi

  # Docker
  if command -v docker &>/dev/null; then
    log_ok "Docker $(docker --version | cut -d' ' -f3 | tr -d ',')"
  else
    log_warn "Docker が未インストール。DB 等のローカルサービスに必要です"
    log_info "  macOS: brew install --cask docker"
    log_info "  Linux: https://docs.docker.com/engine/install/"
  fi

  # Node.js バージョンマネージャ
  if command -v fnm &>/dev/null; then
    log_ok "fnm $(fnm --version)"
  elif command -v nvm &>/dev/null; then
    log_ok "nvm installed"
  elif command -v volta &>/dev/null; then
    log_ok "volta $(volta --version)"
  else
    log_warn "Node.js バージョンマネージャが未インストール"
    install_node_manager
  fi
}

install_node_manager() {
  log_info "fnm をインストール中..."
  case "$OS" in
    macos)
      if command -v brew &>/dev/null; then
        brew install fnm
      else
        curl -fsSL https://fnm.vercel.app/install | bash
      fi
      ;;
    linux)
      curl -fsSL https://fnm.vercel.app/install | bash
      ;;
    *)
      log_error "手動で fnm をインストールしてください: https://github.com/Schniz/fnm"
      ;;
  esac
}

# ======================================
# 2. Node.js のセットアップ
# ======================================
setup_node() {
  log_info "Node.js をセットアップ中..."

  # .nvmrc からバージョンを読み取り
  if [ -f .nvmrc ]; then
    NODE_VERSION=$(cat .nvmrc | tr -d '[:space:]')
  elif [ -f .node-version ]; then
    NODE_VERSION=$(cat .node-version | tr -d '[:space:]')
  else
    NODE_VERSION="20"
  fi

  if command -v fnm &>/dev/null; then
    fnm install "$NODE_VERSION"
    fnm use "$NODE_VERSION"
  elif command -v nvm &>/dev/null; then
    nvm install "$NODE_VERSION"
    nvm use "$NODE_VERSION"
  fi

  log_ok "Node.js $(node --version)"
}

# ======================================
# 3. 依存関係のインストール
# ======================================
install_dependencies() {
  log_info "依存関係をインストール中..."

  if [ -f pnpm-lock.yaml ]; then
    if ! command -v pnpm &>/dev/null; then
      npm install -g pnpm
    fi
    pnpm install --frozen-lockfile
    log_ok "pnpm install 完了"
  elif [ -f yarn.lock ]; then
    if ! command -v yarn &>/dev/null; then
      npm install -g yarn
    fi
    yarn install --frozen-lockfile
    log_ok "yarn install 完了"
  else
    npm ci
    log_ok "npm ci 完了"
  fi
}

# ======================================
# 4. 環境変数のセットアップ
# ======================================
setup_env() {
  log_info "環境変数をセットアップ中..."

  if [ -f .env.example ] && [ ! -f .env ]; then
    cp .env.example .env
    log_ok ".env.example から .env を作成"
    log_warn ".env の値を確認し、必要に応じて更新してください"
  elif [ -f .env ]; then
    log_ok ".env は既に存在"
  else
    log_warn ".env.example が見つかりません"
  fi
}

# ======================================
# 5. Docker サービスの起動
# ======================================
setup_services() {
  log_info "Docker サービスを起動中..."

  if command -v docker &>/dev/null && [ -f docker-compose.yml ]; then
    docker compose up -d
    log_info "サービスの起動を待機中..."
    sleep 5
    log_ok "Docker サービス起動完了"
  else
    log_warn "Docker Compose をスキップ"
  fi
}

# ======================================
# 6. データベースのセットアップ
# ======================================
setup_database() {
  log_info "データベースをセットアップ中..."

  if [ -f prisma/schema.prisma ]; then
    npx prisma migrate dev 2>/dev/null || npx prisma migrate deploy
    log_ok "Prisma マイグレーション完了"

    if npx prisma db seed 2>/dev/null; then
      log_ok "シードデータ投入完了"
    fi
  elif [ -f knexfile.js ] || [ -f knexfile.ts ]; then
    npx knex migrate:latest
    npx knex seed:run 2>/dev/null || true
    log_ok "Knex マイグレーション完了"
  fi
}

# ======================================
# 7. 検証
# ======================================
verify_setup() {
  log_info "セットアップを検証中..."
  local errors=0

  # Node.js バージョン
  if node --version &>/dev/null; then
    log_ok "Node.js: $(node --version)"
  else
    log_error "Node.js が動作しません"
    ((errors++))
  fi

  # TypeScript コンパイル
  if npx tsc --noEmit 2>/dev/null; then
    log_ok "TypeScript コンパイル成功"
  else
    log_warn "TypeScript にエラーがあります (後で修正可)"
  fi

  # DB 接続
  if command -v docker &>/dev/null; then
    if docker compose exec -T postgres pg_isready 2>/dev/null; then
      log_ok "PostgreSQL 接続可能"
    else
      log_warn "PostgreSQL に接続できません"
    fi
  fi

  if [ "$errors" -gt 0 ]; then
    log_error "$errors 個のエラーが見つかりました"
    return 1
  fi

  return 0
}

# ======================================
# メイン実行
# ======================================
main() {
  echo ""
  echo "=================================="
  echo "  Project Setup Script"
  echo "=================================="
  echo ""

  check_prerequisites
  setup_node
  install_dependencies
  setup_env
  setup_services
  setup_database
  verify_setup

  echo ""
  echo "=================================="
  echo -e "  ${GREEN}Setup Complete!${NC}"
  echo "=================================="
  echo ""
  echo "次のコマンドで開発を開始できます:"
  echo ""
  echo "  make dev    # 開発サーバー起動"
  echo "  make test   # テスト実行"
  echo "  make help   # 全コマンド一覧"
  echo ""
}

main "$@"
```

### 2.2 Homebrew による macOS セットアップ

macOS では Homebrew の Brewfile を使って、必要なツールを宣言的にインストールできる。

```ruby
# Brewfile
# macOS の開発ツールを宣言的にインストール
# 実行: brew bundle

# === CLI ツール ===
brew "git"
brew "gh"           # GitHub CLI
brew "jq"           # JSON パーサー
brew "yq"           # YAML パーサー
brew "fnm"          # Node.js バージョン管理
brew "mise"         # 多言語バージョン管理
brew "shellcheck"   # シェルスクリプト Linter
brew "act"          # GitHub Actions ローカル実行
brew "direnv"       # ディレクトリ別環境変数

# === コンテナ ===
cask "docker"       # Docker Desktop
brew "lazydocker"   # Docker TUI

# === データベースツール ===
brew "postgresql@16"  # PostgreSQL クライアント (psql)
brew "redis"          # Redis クライアント

# === エディタ ===
cask "visual-studio-code"
# cask "cursor"       # AI エディタ

# === フォント ===
cask "font-jetbrains-mono"
cask "font-jetbrains-mono-nerd-font"

# === その他ツール ===
cask "iterm2"       # ターミナル
cask "raycast"      # ランチャー
cask "figma"        # デザインツール
```

```bash
# Brewfile を使ったセットアップスクリプト (macOS 用)
#!/bin/bash
# scripts/setup-macos.sh

set -euo pipefail

# Homebrew がインストールされていない場合
if ! command -v brew &>/dev/null; then
  echo "Homebrew をインストール中..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Brewfile のインストール
echo "Brewfile のパッケージをインストール中..."
brew bundle --file=Brewfile

# fnm のシェル設定を追加
if ! grep -q "fnm env" ~/.zshrc 2>/dev/null; then
  echo 'eval "$(fnm env --use-on-cd)"' >> ~/.zshrc
  echo "fnm のシェル設定を .zshrc に追加しました"
fi

echo "macOS セットアップ完了"
```

### 2.3 Linux (Ubuntu/Debian) セットアップ

```bash
#!/bin/bash
# scripts/setup-linux.sh
# Ubuntu/Debian 向けのセットアップスクリプト

set -euo pipefail

log_info() { echo -e "\033[0;34m[INFO]\033[0m  $1"; }
log_ok()   { echo -e "\033[0;32m[OK]\033[0m    $1"; }

# 基本パッケージのインストール
log_info "基本パッケージをインストール中..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
  build-essential \
  curl \
  wget \
  git \
  jq \
  unzip \
  ca-certificates \
  gnupg \
  lsb-release

# Docker のインストール
if ! command -v docker &>/dev/null; then
  log_info "Docker をインストール中..."
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
  log_ok "Docker インストール完了 (再ログインが必要です)"
fi

# Docker Compose V2 の確認
if docker compose version &>/dev/null; then
  log_ok "Docker Compose $(docker compose version --short)"
fi

# fnm のインストール
if ! command -v fnm &>/dev/null; then
  log_info "fnm をインストール中..."
  curl -fsSL https://fnm.vercel.app/install | bash
  export PATH="$HOME/.local/share/fnm:$PATH"
  eval "$(fnm env)"
fi

# GitHub CLI のインストール
if ! command -v gh &>/dev/null; then
  log_info "GitHub CLI をインストール中..."
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \
    sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | \
    sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq gh
fi

log_ok "Linux セットアップ完了"
```

### 2.4 WSL2 環境のセットアップ

```bash
#!/bin/bash
# scripts/setup-wsl.sh
# WSL2 (Windows Subsystem for Linux) 向けの追加セットアップ

set -euo pipefail

log_info() { echo -e "\033[0;34m[INFO]\033[0m  $1"; }
log_ok()   { echo -e "\033[0;32m[OK]\033[0m    $1"; }
log_warn() { echo -e "\033[0;33m[WARN]\033[0m  $1"; }

# WSL2 の検出
if [ -z "${WSL_DISTRO_NAME:-}" ]; then
  echo "このスクリプトは WSL2 環境で実行してください"
  exit 1
fi

log_info "WSL2 環境: ${WSL_DISTRO_NAME}"

# Windows 側の Docker Desktop を使用する設定の確認
if command -v docker &>/dev/null; then
  log_ok "Docker は利用可能"
else
  log_warn "Docker Desktop for Windows をインストールし、WSL2 統合を有効にしてください"
  log_info "  Settings > Resources > WSL Integration > ${WSL_DISTRO_NAME} を有効化"
fi

# /mnt/c 以下ではなく WSL ファイルシステムを使用するよう案内
CURRENT_DIR=$(pwd)
if [[ "$CURRENT_DIR" == /mnt/* ]]; then
  log_warn "Windows ファイルシステム上で作業しています"
  log_warn "パフォーマンス向上のため、~/projects/ など WSL ファイルシステム上にリポジトリを配置してください"
  log_info "  例: cd ~ && mkdir -p projects && cd projects"
fi

# Git の改行コード設定
git config --global core.autocrlf input
log_ok "Git autocrlf を input に設定"

# Linux セットアップを実行
bash scripts/setup-linux.sh

log_ok "WSL2 セットアップ完了"
```

---

## 3. Makefile

### 3.1 プロジェクト用 Makefile

```makefile
# Makefile
.PHONY: help setup dev test lint format build clean docker-up docker-down db-migrate db-seed

# デフォルトターゲット
.DEFAULT_GOAL := help

# 変数
NODE_BIN := ./node_modules/.bin
DOCKER_COMPOSE := docker compose

# ======================================
# ヘルプ (make help)
# ======================================
help: ## コマンド一覧を表示
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ======================================
# 初期セットアップ
# ======================================
setup: ## 初回セットアップ (新メンバーはこれを実行)
	@bash scripts/setup.sh

setup-quick: node_modules .env ## 高速セットアップ (依存関係のみ)
	@echo "Quick setup complete"

node_modules: package.json pnpm-lock.yaml
	pnpm install --frozen-lockfile
	@touch node_modules

.env:
	cp .env.example .env
	@echo ".env created from .env.example"

# ======================================
# 開発
# ======================================
dev: ## 開発サーバー起動
	$(NODE_BIN)/next dev

dev-all: docker-up ## 全サービス + 開発サーバー起動
	$(NODE_BIN)/next dev

dev-turbo: ## Turbo モードで開発サーバー起動
	$(NODE_BIN)/next dev --turbo

# ======================================
# テスト
# ======================================
test: ## テスト実行
	$(NODE_BIN)/vitest run

test-watch: ## テスト (watch モード)
	$(NODE_BIN)/vitest

test-coverage: ## テスト + カバレッジ
	$(NODE_BIN)/vitest run --coverage

test-e2e: ## E2E テスト
	$(NODE_BIN)/playwright test

test-e2e-ui: ## E2E テスト (UI モード)
	$(NODE_BIN)/playwright test --ui

# ======================================
# コード品質
# ======================================
lint: ## Lint チェック
	$(NODE_BIN)/eslint src/
	$(NODE_BIN)/tsc --noEmit

lint-fix: ## Lint 自動修正
	$(NODE_BIN)/eslint src/ --fix

format: ## コードフォーマット
	$(NODE_BIN)/prettier --write "src/**/*.{ts,tsx,json,css}"

format-check: ## フォーマットチェック (CI 用)
	$(NODE_BIN)/prettier --check "src/**/*.{ts,tsx,json,css}"

typecheck: ## TypeScript 型チェック
	$(NODE_BIN)/tsc --noEmit

check: lint format-check typecheck test ## 全品質チェック実行 (CI 相当)

# ======================================
# ビルド
# ======================================
build: ## プロダクションビルド
	$(NODE_BIN)/next build

build-analyze: ## バンドル分析付きビルド
	ANALYZE=true $(NODE_BIN)/next build

# ======================================
# Docker
# ======================================
docker-up: ## Docker サービス起動
	$(DOCKER_COMPOSE) up -d
	@echo "Waiting for services..."
	@sleep 3
	@$(DOCKER_COMPOSE) ps

docker-down: ## Docker サービス停止
	$(DOCKER_COMPOSE) down

docker-logs: ## Docker ログ表示
	$(DOCKER_COMPOSE) logs -f

docker-clean: ## Docker ボリューム含めて完全削除
	$(DOCKER_COMPOSE) down -v --remove-orphans

docker-rebuild: ## Docker イメージ再ビルド
	$(DOCKER_COMPOSE) build --no-cache
	$(DOCKER_COMPOSE) up -d

# ======================================
# データベース
# ======================================
db-migrate: ## マイグレーション実行
	$(NODE_BIN)/prisma migrate dev

db-seed: ## シードデータ投入
	$(NODE_BIN)/prisma db seed

db-reset: ## DB リセット (データ全削除)
	$(NODE_BIN)/prisma migrate reset --force

db-studio: ## Prisma Studio 起動
	$(NODE_BIN)/prisma studio

db-generate: ## Prisma Client 再生成
	$(NODE_BIN)/prisma generate

# ======================================
# コード生成
# ======================================
generate: ## 全コード生成実行
	$(NODE_BIN)/prisma generate
	$(NODE_BIN)/graphql-codegen

generate-api: ## OpenAPI からクライアント生成
	$(NODE_BIN)/openapi-typescript api/openapi.yaml -o src/types/api.d.ts

# ======================================
# ユーティリティ
# ======================================
clean: ## ビルド成果物を削除
	rm -rf .next dist node_modules/.cache

clean-all: clean ## 全キャッシュ削除 (node_modules 含む)
	rm -rf node_modules

doctor: ## 環境診断
	@bash scripts/doctor.sh

update-deps: ## 依存関係の更新チェック
	$(NODE_BIN)/npm-check-updates -u
	pnpm install

storybook: ## Storybook 起動
	$(NODE_BIN)/storybook dev -p 6006

storybook-build: ## Storybook ビルド
	$(NODE_BIN)/storybook build
```

### 3.2 フローの可視化

```
+------------------------------------------------------------------+
|              Makefile タスク依存関係                                |
+------------------------------------------------------------------+
|                                                                  |
|  [初回セットアップ]                                                |
|  make setup                                                      |
|    +-- scripts/setup.sh                                          |
|        +-- check_prerequisites                                   |
|        +-- setup_node                                            |
|        +-- install_dependencies (= node_modules)                 |
|        +-- setup_env (= .env)                                    |
|        +-- setup_services (= docker-up)                          |
|        +-- setup_database (= db-migrate + db-seed)               |
|        +-- verify_setup                                          |
|                                                                  |
|  [日常開発]                                                       |
|  make dev-all                                                    |
|    +-- docker-up (DB, Redis 等)                                  |
|    +-- next dev                                                  |
|                                                                  |
|  [CI]                                                            |
|  make check                                                      |
|    +-- lint (ESLint + TypeScript)                                |
|    +-- format-check (Prettier)                                   |
|    +-- typecheck (tsc --noEmit)                                  |
|    +-- test (Vitest)                                             |
|                                                                  |
|  [リリース]                                                       |
|  make build                                                      |
|    +-- next build                                                |
|                                                                  |
+------------------------------------------------------------------+
```

### 3.3 Makefile の高度なテクニック

```makefile
# Makefile (高度な設定例)

# シェルの指定 (bash の機能を使うため)
SHELL := /bin/bash

# 変数
-include .env  # .env ファイルを読み込み (存在しない場合はスキップ)

PROJECT_NAME := myapp
GIT_HASH := $(shell git rev-parse --short HEAD)
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
TIMESTAMP := $(shell date +%Y%m%d-%H%M%S)

# ファイル依存による条件実行
# node_modules が package.json より古い場合のみ再インストール
node_modules: package.json pnpm-lock.yaml
	pnpm install --frozen-lockfile
	@touch $@

# 環境変数の検証
.PHONY: check-env
check-env: ## 環境変数の検証
	@test -n "$(DATABASE_URL)" || (echo "DATABASE_URL is not set" && exit 1)
	@test -n "$(REDIS_URL)" || (echo "REDIS_URL is not set" && exit 1)
	@echo "環境変数 OK"

# 並列実行の例
.PHONY: ci
ci: ## CI パイプラインを並列実行
	@$(MAKE) -j4 lint typecheck format-check test

# コンテナイメージのビルド
.PHONY: docker-build
docker-build: ## Docker イメージをビルド
	docker build \
		--build-arg GIT_HASH=$(GIT_HASH) \
		--build-arg BUILD_TIME=$(TIMESTAMP) \
		-t $(PROJECT_NAME):$(GIT_HASH) \
		-t $(PROJECT_NAME):latest \
		.

# 秘密情報の漏洩チェック
.PHONY: secrets-scan
secrets-scan: ## 秘密情報の漏洩チェック
	@if command -v gitleaks &>/dev/null; then \
		gitleaks detect --source . --verbose; \
	else \
		echo "gitleaks がインストールされていません: brew install gitleaks"; \
	fi

# ライセンスチェック
.PHONY: license-check
license-check: ## 依存パッケージのライセンスチェック
	$(NODE_BIN)/license-checker --production --onlyAllow \
		'MIT;Apache-2.0;BSD-2-Clause;BSD-3-Clause;ISC;0BSD;CC0-1.0'
```

---

## 4. 環境診断スクリプト (doctor)

```bash
#!/bin/bash
# scripts/doctor.sh
# 環境の問題を診断するスクリプト

set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

pass=0
warn=0
fail=0

check() {
  local name="$1"
  local cmd="$2"
  local expected="$3"

  if eval "$cmd" &>/dev/null; then
    echo -e "${GREEN}PASS${NC}  $name"
    ((pass++))
  elif [ -n "$expected" ]; then
    echo -e "${RED}FAIL${NC}  $name -- $expected"
    ((fail++))
  else
    echo -e "${YELLOW}WARN${NC}  $name"
    ((warn++))
  fi
}

check_version() {
  local name="$1"
  local cmd="$2"
  local expected="$3"
  local actual

  actual=$(eval "$cmd" 2>/dev/null | tr -d '[:space:]' || echo "")
  expected=$(echo "$expected" | tr -d '[:space:]')

  if [ "$actual" = "$expected" ]; then
    echo -e "${GREEN}PASS${NC}  $name: $actual"
    ((pass++))
  elif [ -n "$actual" ]; then
    echo -e "${YELLOW}WARN${NC}  $name: $actual (expected: $expected)"
    ((warn++))
  else
    echo -e "${RED}FAIL${NC}  $name: not found"
    ((fail++))
  fi
}

echo ""
echo "=== Environment Doctor ==="
echo ""

echo "--- System ---"
echo -e "${BLUE}INFO${NC}  OS: $(uname -s) $(uname -r)"
echo -e "${BLUE}INFO${NC}  Shell: $SHELL"
echo -e "${BLUE}INFO${NC}  User: $(whoami)"
echo ""

echo "--- Tools ---"
check "Git" "command -v git" "git をインストールしてください"
check "Node.js" "command -v node" "Node.js をインストールしてください"
check "Docker" "command -v docker" "Docker をインストールしてください"
check "Docker Compose" "docker compose version" "Docker Compose V2 が必要です"
check "pnpm" "command -v pnpm" "npm install -g pnpm を実行してください"
check "GitHub CLI (gh)" "command -v gh" "任意: brew install gh"

echo ""
echo "--- Node.js ---"
if [ -f .nvmrc ]; then
  check_version "Node version matches .nvmrc" \
    "node -v | tr -d 'v'" \
    "$(cat .nvmrc 2>/dev/null)"
elif [ -f .node-version ]; then
  check_version "Node version matches .node-version" \
    "node -v | tr -d 'v'" \
    "$(cat .node-version 2>/dev/null)"
fi
check "node_modules exists" "[ -d node_modules ]" "make setup を実行してください"
check "TypeScript compiles" "npx tsc --noEmit 2>/dev/null" ""

echo ""
echo "--- Services ---"
check "Docker daemon running" "docker info" "Docker Desktop を起動してください"
check "PostgreSQL reachable" "pg_isready -h localhost -p 5432 2>/dev/null" ""
check "Redis reachable" "redis-cli -h localhost ping 2>/dev/null" ""

echo ""
echo "--- Files ---"
check ".env exists" "[ -f .env ]" "cp .env.example .env を実行してください"
check ".env has DATABASE_URL" "grep -q DATABASE_URL .env 2>/dev/null" ""
check ".env has REDIS_URL" "grep -q REDIS_URL .env 2>/dev/null" ""

echo ""
echo "--- Ports ---"
check "Port 3000 available" "! lsof -i :3000 -sTCP:LISTEN" "ポート 3000 が使用中です"
check "Port 5432 (PostgreSQL)" "lsof -i :5432 -sTCP:LISTEN" ""
check "Port 6379 (Redis)" "lsof -i :6379 -sTCP:LISTEN" ""

echo ""
echo "--- Git ---"
check "Git user.name configured" "git config user.name" "git config --global user.name 'Your Name'"
check "Git user.email configured" "git config user.email" "git config --global user.email 'your@email.com'"
check "On main/develop branch" "git rev-parse --abbrev-ref HEAD | grep -E '^(main|develop)$'" ""

echo ""
echo "==========================="
echo -e "Results: ${GREEN}${pass} passed${NC}, ${YELLOW}${warn} warnings${NC}, ${RED}${fail} failed${NC}"
echo ""

if [ "$fail" -gt 0 ]; then
  echo -e "${RED}環境に問題があります。上記の FAIL 項目を修正してください。${NC}"
  echo ""
fi

exit $fail
```

### 4.1 自動修復スクリプト (doctor --fix)

```bash
#!/bin/bash
# scripts/doctor-fix.sh
# 自動修復可能な問題を修正する

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

log_fix()  { echo -e "${GREEN}[FIX]${NC}  $1"; }
log_skip() { echo -e "${YELLOW}[SKIP]${NC} $1"; }

echo ""
echo "=== Auto Fix ==="
echo ""

# .env ファイルの作成
if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  log_fix ".env ファイルを作成しました"
else
  log_skip ".env は既に存在します"
fi

# node_modules のインストール
if [ ! -d node_modules ]; then
  echo "node_modules をインストール中..."
  if [ -f pnpm-lock.yaml ]; then
    pnpm install --frozen-lockfile
  elif [ -f yarn.lock ]; then
    yarn install --frozen-lockfile
  else
    npm ci
  fi
  log_fix "node_modules をインストールしました"
else
  log_skip "node_modules は既に存在します"
fi

# Docker サービスの起動
if command -v docker &>/dev/null && [ -f docker-compose.yml ]; then
  RUNNING=$(docker compose ps --services --filter "status=running" 2>/dev/null | wc -l | tr -d ' ')
  DEFINED=$(docker compose config --services 2>/dev/null | wc -l | tr -d ' ')

  if [ "$RUNNING" -lt "$DEFINED" ]; then
    docker compose up -d
    log_fix "Docker サービスを起動しました"
  else
    log_skip "Docker サービスは全て起動済みです"
  fi
fi

# Node.js バージョンの切り替え
if [ -f .nvmrc ]; then
  EXPECTED=$(cat .nvmrc | tr -d '[:space:]')
  ACTUAL=$(node -v 2>/dev/null | tr -d 'v[:space:]')

  if [ "$ACTUAL" != "$EXPECTED" ]; then
    if command -v fnm &>/dev/null; then
      fnm install "$EXPECTED" && fnm use "$EXPECTED"
      log_fix "Node.js を $EXPECTED に切り替えました"
    elif command -v nvm &>/dev/null; then
      nvm install "$EXPECTED" && nvm use "$EXPECTED"
      log_fix "Node.js を $EXPECTED に切り替えました"
    fi
  else
    log_skip "Node.js バージョンは正しいです ($ACTUAL)"
  fi
fi

# Prisma Client 再生成
if [ -f prisma/schema.prisma ]; then
  npx prisma generate 2>/dev/null
  log_fix "Prisma Client を再生成しました"
fi

echo ""
echo "=== 修復完了 ==="
echo "make doctor で状態を確認してください"
echo ""
```

---

## 5. .env.example テンプレート

```bash
# .env.example
# このファイルをコピーして .env を作成してください
# cp .env.example .env

# ===== アプリケーション =====
NODE_ENV=development
PORT=3000
APP_URL=http://localhost:3000
APP_NAME=MyApp

# ===== データベース =====
# docker-compose.yml の設定と一致させること
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/myapp_development

# ===== Redis =====
REDIS_URL=redis://localhost:6379

# ===== メール (MailHog) =====
SMTP_HOST=localhost
SMTP_PORT=1025
SMTP_FROM=noreply@example.com

# ===== ストレージ (MinIO) =====
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=uploads
S3_REGION=us-east-1

# ===== 認証 =====
JWT_SECRET=dev-secret-change-in-production
SESSION_SECRET=dev-session-secret
# OAuth (Google)
# GOOGLE_CLIENT_ID=xxx
# GOOGLE_CLIENT_SECRET=xxx
# OAuth (GitHub)
# GITHUB_CLIENT_ID=xxx
# GITHUB_CLIENT_SECRET=xxx

# ===== 外部 API (開発用ダミー値) =====
# STRIPE_SECRET_KEY=sk_test_xxx
# SENDGRID_API_KEY=SG.xxx
# OPENAI_API_KEY=sk-xxx

# ===== モニタリング =====
# SENTRY_DSN=https://xxx@sentry.io/xxx
# DATADOG_API_KEY=xxx

# ===== ログ =====
LOG_LEVEL=debug
LOG_FORMAT=pretty
```

### 5.1 .env の管理戦略

```
+------------------------------------------------------------------+
|              .env ファイルの管理戦略                                 |
+------------------------------------------------------------------+
|                                                                  |
|  .env.example          -- Git にコミット。テンプレート              |
|  .env                  -- Git に含めない。ローカル設定              |
|  .env.test             -- Git にコミット可。テスト用設定             |
|  .env.development      -- Git にコミット可。開発共通設定             |
|  .env.local            -- Git に含めない。個人設定                   |
|  .env.production.local -- Git に含めない。本番設定                   |
|                                                                  |
|  読み込み優先順位 (Next.js の場合):                                 |
|  .env.local > .env.development > .env                            |
|                                                                  |
|  Secret Manager (本番環境):                                       |
|  - AWS Secrets Manager                                           |
|  - Google Secret Manager                                         |
|  - HashiCorp Vault                                               |
|  - Doppler                                                       |
|  - Infisical                                                     |
|                                                                  |
+------------------------------------------------------------------+
```

### 5.2 .env バリデーションスクリプト

```bash
#!/bin/bash
# scripts/validate-env.sh
# .env ファイルの必須項目を検証する

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

ENV_FILE="${1:-.env}"
ERRORS=0

# 必須環境変数の一覧
REQUIRED_VARS=(
  "DATABASE_URL"
  "REDIS_URL"
  "JWT_SECRET"
  "SESSION_SECRET"
)

# 警告レベルの環境変数
OPTIONAL_VARS=(
  "SMTP_HOST"
  "S3_ENDPOINT"
)

echo "=== .env バリデーション ==="
echo "File: $ENV_FILE"
echo ""

if [ ! -f "$ENV_FILE" ]; then
  echo -e "${RED}FAIL${NC} $ENV_FILE が見つかりません"
  echo "  cp .env.example .env を実行してください"
  exit 1
fi

# 必須チェック
echo "--- Required ---"
for var in "${REQUIRED_VARS[@]}"; do
  if grep -q "^${var}=" "$ENV_FILE" 2>/dev/null; then
    VALUE=$(grep "^${var}=" "$ENV_FILE" | cut -d'=' -f2-)
    if [ -z "$VALUE" ] || [ "$VALUE" = "xxx" ]; then
      echo -e "${RED}FAIL${NC} $var: 値が未設定です"
      ((ERRORS++))
    else
      echo -e "${GREEN}PASS${NC} $var"
    fi
  else
    echo -e "${RED}FAIL${NC} $var: キーが見つかりません"
    ((ERRORS++))
  fi
done

echo ""

# オプションチェック
echo "--- Optional ---"
for var in "${OPTIONAL_VARS[@]}"; do
  if grep -q "^${var}=" "$ENV_FILE" 2>/dev/null; then
    echo -e "${GREEN}SET ${NC} $var"
  else
    echo -e "SKIP $var (未設定)"
  fi
done

echo ""

# セキュリティチェック
echo "--- Security ---"
# デフォルト値が残っていないか
if grep -q "dev-secret-change-in-production" "$ENV_FILE" 2>/dev/null; then
  echo -e "${RED}WARN${NC} JWT_SECRET がデフォルト値です (開発環境以外では変更してください)"
fi

# 本番用シークレットが含まれていないか
if grep -qE "^(STRIPE_SECRET_KEY|SENDGRID_API_KEY)=.+[^x]" "$ENV_FILE" 2>/dev/null; then
  echo -e "${RED}WARN${NC} 本番用の API キーが含まれている可能性があります"
fi

echo ""
if [ "$ERRORS" -gt 0 ]; then
  echo -e "${RED}$ERRORS 個のエラーがあります${NC}"
  exit 1
else
  echo -e "${GREEN}バリデーション OK${NC}"
fi
```

---

## 6. Docker Compose 開発環境

### 6.1 docker-compose.yml の標準テンプレート

```yaml
# docker-compose.yml
# ローカル開発用のサービス構成

services:
  # === PostgreSQL ===
  postgres:
    image: postgres:16-alpine
    container_name: myapp-postgres
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: myapp_development
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  # === Redis ===
  redis:
    image: redis:7-alpine
    container_name: myapp-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

  # === MailHog (メール確認) ===
  mailhog:
    image: mailhog/mailhog:latest
    container_name: myapp-mailhog
    ports:
      - "1025:1025"   # SMTP
      - "8025:8025"   # Web UI

  # === MinIO (S3 互換ストレージ) ===
  minio:
    image: minio/minio:latest
    container_name: myapp-minio
    ports:
      - "9000:9000"   # API
      - "9001:9001"   # Console
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  redis_data:
  minio_data:
```

### 6.2 サービスの待機スクリプト

```bash
#!/bin/bash
# scripts/wait-for-services.sh
# Docker サービスが利用可能になるまで待機する

set -euo pipefail

MAX_RETRIES=30
RETRY_INTERVAL=2

wait_for_service() {
  local name="$1"
  local check_cmd="$2"
  local retries=0

  echo -n "Waiting for $name..."
  while ! eval "$check_cmd" &>/dev/null; do
    retries=$((retries + 1))
    if [ "$retries" -ge "$MAX_RETRIES" ]; then
      echo " TIMEOUT"
      return 1
    fi
    echo -n "."
    sleep "$RETRY_INTERVAL"
  done
  echo " OK"
}

wait_for_service "PostgreSQL" "pg_isready -h localhost -p 5432"
wait_for_service "Redis" "redis-cli -h localhost ping"

echo ""
echo "All services are ready!"
```

---

## 7. Task Runner 比較

| ツール | 設定ファイル | 言語 | 並列実行 | 依存管理 | 学習コスト |
|-------|-----------|------|---------|---------|-----------|
| Make | Makefile | シェル | 限定的 | ファイル依存 | 低 |
| npm scripts | package.json | シェル | npm-run-all | なし | 最低 |
| just | justfile | シェル | なし | なし | 低 |
| task (Taskfile) | Taskfile.yml | YAML+シェル | あり | タスク依存 | 低 |
| turbo | turbo.json | JSON | あり(高速) | パッケージ依存 | 中 |
| nx | nx.json | JSON | あり(高速) | プロジェクトグラフ | 高 |

### 7.1 just の設定例

just は Make の代替として設計された Rust 製のコマンドランナーで、Makefile の煩雑さを解消する。

```just
# justfile

# デフォルトレシピ
default:
  @just --list

# 初回セットアップ
setup:
  bash scripts/setup.sh

# 開発サーバー起動
dev:
  pnpm next dev

# 全サービス起動 + 開発サーバー
dev-all: docker-up
  pnpm next dev

# テスト
test *ARGS:
  pnpm vitest run {{ARGS}}

# テスト (watch モード)
test-watch:
  pnpm vitest

# Lint
lint:
  pnpm eslint src/
  pnpm tsc --noEmit

# Lint 自動修正
lint-fix:
  pnpm eslint src/ --fix

# フォーマット
format:
  pnpm prettier --write "src/**/*.{ts,tsx,json,css}"

# ビルド
build:
  pnpm next build

# Docker 起動
docker-up:
  docker compose up -d

# Docker 停止
docker-down:
  docker compose down

# DB マイグレーション
db-migrate:
  pnpm prisma migrate dev

# DB リセット
db-reset:
  pnpm prisma migrate reset --force

# 環境診断
doctor:
  bash scripts/doctor.sh

# 全品質チェック (CI 相当)
check: lint
  pnpm prettier --check "src/**/*.{ts,tsx,json,css}"
  pnpm vitest run
```

### 7.2 Taskfile (task) の設定例

```yaml
# Taskfile.yml
version: '3'

vars:
  NODE_BIN: ./node_modules/.bin

tasks:
  default:
    desc: コマンド一覧を表示
    cmds:
      - task --list

  setup:
    desc: 初回セットアップ
    cmds:
      - bash scripts/setup.sh

  dev:
    desc: 開発サーバー起動
    cmds:
      - "{{.NODE_BIN}}/next dev"

  test:
    desc: テスト実行
    cmds:
      - "{{.NODE_BIN}}/vitest run"

  lint:
    desc: Lint チェック
    cmds:
      - "{{.NODE_BIN}}/eslint src/"
      - "{{.NODE_BIN}}/tsc --noEmit"

  build:
    desc: プロダクションビルド
    cmds:
      - "{{.NODE_BIN}}/next build"

  docker:up:
    desc: Docker サービス起動
    cmds:
      - docker compose up -d
      - sleep 3
      - docker compose ps

  docker:down:
    desc: Docker サービス停止
    cmds:
      - docker compose down

  db:migrate:
    desc: DB マイグレーション
    cmds:
      - "{{.NODE_BIN}}/prisma migrate dev"

  check:
    desc: 全品質チェック
    deps: [lint]
    cmds:
      - "{{.NODE_BIN}}/prettier --check 'src/**/*.{ts,tsx,json,css}'"
      - "{{.NODE_BIN}}/vitest run"
```

---

## 8. CI/CD でのセットアップ検証

### 8.1 フレッシュインストールテスト

セットアップスクリプトの陳腐化を防ぐため、CI で定期的にフレッシュインストールを実行する。

```yaml
# .github/workflows/fresh-install-test.yml
name: Fresh Install Test

on:
  schedule:
    - cron: '0 9 * * 1'  # 毎週月曜 9:00 UTC
  workflow_dispatch:

jobs:
  fresh-install:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [20, 22]
    steps:
      - uses: actions/checkout@v4

      - name: Clean environment (simulate new developer)
        run: |
          rm -rf node_modules .env dist .next

      - uses: pnpm/action-setup@v4

      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}

      - name: Run setup script
        run: |
          cp .env.example .env
          pnpm install --frozen-lockfile

      - name: Verify build
        run: pnpm run build

      - name: Verify tests pass
        run: pnpm run test -- --run

      - name: Verify lint passes
        run: pnpm run lint

  fresh-install-docker:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: myapp_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - uses: pnpm/action-setup@v4

      - uses: actions/setup-node@v4
        with:
          node-version-file: '.node-version'

      - name: Setup with DB
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/myapp_test
          REDIS_URL: redis://localhost:6379
        run: |
          cp .env.example .env
          pnpm install --frozen-lockfile
          pnpm prisma migrate deploy
          pnpm prisma db seed
          pnpm run test -- --run
```

### 8.2 セットアップドキュメントの自動生成

```bash
#!/bin/bash
# scripts/generate-setup-docs.sh
# セットアップに必要な情報を自動で抽出してドキュメント化する

set -euo pipefail

echo "# 開発環境セットアップガイド"
echo ""
echo "自動生成: $(date +%Y-%m-%d)"
echo ""

# Node.js バージョン
echo "## 必要な Node.js バージョン"
if [ -f .nvmrc ]; then
  echo "- Node.js: $(cat .nvmrc)"
elif [ -f .node-version ]; then
  echo "- Node.js: $(cat .node-version)"
fi
echo ""

# パッケージマネージャ
echo "## パッケージマネージャ"
if [ -f pnpm-lock.yaml ]; then
  echo "- pnpm"
  PNPM_VERSION=$(grep -m1 'packageManager' package.json 2>/dev/null | grep -oP 'pnpm@\K[^"]+' || echo "latest")
  echo "- バージョン: $PNPM_VERSION"
elif [ -f yarn.lock ]; then
  echo "- yarn"
else
  echo "- npm"
fi
echo ""

# Docker サービス
if [ -f docker-compose.yml ]; then
  echo "## Docker サービス"
  docker compose config --services 2>/dev/null | while read -r service; do
    IMAGE=$(docker compose config 2>/dev/null | grep -A5 "^  $service:" | grep "image:" | awk '{print $2}' || echo "custom")
    echo "- $service ($IMAGE)"
  done
  echo ""
fi

# 環境変数
if [ -f .env.example ]; then
  echo "## 環境変数"
  echo ""
  echo '```bash'
  cat .env.example
  echo '```'
  echo ""
fi

# Make コマンド
if [ -f Makefile ]; then
  echo "## 利用可能なコマンド"
  echo ""
  echo '```'
  grep -E '^[a-zA-Z_-]+:.*?## .*$' Makefile | sort | \
    awk 'BEGIN {FS = ":.*?## "}; {printf "make %-20s %s\n", $1, $2}'
  echo '```'
fi
```

---

## 9. オンボーディングチェックリスト

### 9.1 新メンバー向けチェックリスト

```markdown
# 新メンバーオンボーディングチェックリスト

## Day 1: 環境構築
- [ ] GitHub アカウントで組織に招待されたことを確認
- [ ] リポジトリのアクセス権限を確認
- [ ] `git clone` でリポジトリを取得
- [ ] `make setup` を実行 (5-15分)
- [ ] `make dev` で開発サーバーが起動することを確認
- [ ] `make test` でテストが通ることを確認
- [ ] `make doctor` で環境に問題がないことを確認

## Day 1-2: 開発フロー理解
- [ ] ブランチ戦略を理解 (main / develop / feature/*)
- [ ] PR テンプレートを確認
- [ ] コミットメッセージ規約 (Conventional Commits) を理解
- [ ] CI/CD パイプラインの流れを確認
- [ ] コードレビューのガイドラインを読む

## Day 2-3: コードベース理解
- [ ] ディレクトリ構造を確認
- [ ] 主要なモジュールの概要を把握
- [ ] ADR (Architecture Decision Records) を読む
- [ ] API ドキュメントを確認
- [ ] テスト戦略を理解

## Week 1: 初めてのコントリビューション
- [ ] Good First Issue を1つ完了
- [ ] PR を作成しレビューを受ける
- [ ] レビューのフィードバックを反映してマージ

## 参考リンク
- ドキュメントサイト: http://docs.example.com
- Figma デザイン: https://figma.com/xxx
- Slack チャンネル: #team-dev
```

### 9.2 チェックリストの自動化

```bash
#!/bin/bash
# scripts/onboarding-check.sh
# 新メンバーのオンボーディング進捗を自動チェック

set -uo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

total=0
passed=0

check_item() {
  local description="$1"
  local check_cmd="$2"
  ((total++))

  if eval "$check_cmd" &>/dev/null; then
    echo -e "${GREEN}[x]${NC} $description"
    ((passed++))
  else
    echo -e "${RED}[ ]${NC} $description"
  fi
}

echo ""
echo "=== オンボーディング進捗チェック ==="
echo ""

echo "--- 環境構築 ---"
check_item "Git が設定されている" "git config user.name && git config user.email"
check_item "Node.js が正しいバージョン" "[ \"\$(node -v | tr -d 'v')\" = \"\$(cat .nvmrc 2>/dev/null | tr -d '[:space:]')\" ]"
check_item "node_modules がインストール済み" "[ -d node_modules ]"
check_item ".env が設定されている" "[ -f .env ]"
check_item "Docker サービスが起動している" "docker compose ps --services --filter 'status=running' | grep -q ."
check_item "テストが通る" "npx vitest run --reporter=silent 2>/dev/null"

echo ""
echo "--- Git 設定 ---"
check_item "SSH キーが設定されている" "ssh -T git@github.com 2>&1 | grep -qi 'success\\|authenticated'"
check_item "GitHub CLI が認証済み" "gh auth status 2>/dev/null"
check_item "husky フックがインストール済み" "[ -d .husky ]"

echo ""
echo "--- ツール ---"
check_item "VS Code がインストール済み" "command -v code"
check_item "Docker がインストール済み" "command -v docker"
check_item "pnpm がインストール済み" "command -v pnpm"

echo ""
echo "==========================="
echo -e "Progress: ${GREEN}${passed}${NC}/${total} ($(( passed * 100 / total ))%)"
echo ""
```

---

## アンチパターン

### アンチパターン 1: 手順書だけで自動化しない

```
# NG: Confluence に書かれた手動手順
1. https://nodejs.org から Node.js v20 をダウンロード
2. インストーラーを実行
3. ターミナルを開いて npm install を実行
4. .env.example を .env にコピー
5. DATABASE_URL を以下のように設定...
(20ステップ続く)

# OK: 1 コマンドで完了
$ make setup
```

**問題点**: 手順書は書いた時点で陳腐化し始める。ステップの抜け、順序の間違い、OS 差異への未対応が蓄積し、新メンバーが設定に丸一日費やすことになる。スクリプト化すれば、手順書自体がコードとして検証・メンテナンスされる。

### アンチパターン 2: セットアップスクリプトにエラーハンドリングがない

```bash
# NG: エラーを無視して続行
npm install
cp .env.example .env
npx prisma migrate dev
echo "Setup complete!"  # 途中でエラーがあっても表示される

# OK: エラー時に即座に停止し、明確なメッセージを出す
set -euo pipefail

npm ci || { echo "npm ci failed. Check Node.js version."; exit 1; }

if [ ! -f .env ]; then
  cp .env.example .env
  echo ".env created. Review and update values."
fi

npx prisma migrate dev || { echo "DB migration failed. Is PostgreSQL running?"; exit 1; }

echo "Setup complete!"
```

**問題点**: `set -e` なしだとエラーが握り潰され、後続ステップが不正な状態で実行される。新メンバーは「スクリプトは成功したのに動かない」という状況に陥り、デバッグに余計な時間がかかる。

### アンチパターン 3: root 権限を要求するスクリプト

```bash
# NG: sudo を多用するスクリプト
sudo apt-get install nodejs
sudo npm install -g pnpm
sudo chmod -R 777 /var/data

# OK: ユーザー権限で動作するように設計
# バージョンマネージャを使ってユーザー空間にインストール
fnm install 20
fnm use 20
# npm のグローバルインストールはホームディレクトリに
npm config set prefix ~/.npm-global
```

**問題点**: `sudo` を要求するスクリプトはセキュリティリスクが高く、企業環境では管理者権限が制限されていることも多い。可能な限りユーザー空間で完結するよう設計し、root 権限が必要な操作は明示的に分離する。

### アンチパターン 4: OS 差異を考慮しないスクリプト

```bash
# NG: macOS でしか動かない
brew install postgresql
open -a "Docker"

# OK: OS を検出して分岐
case "$(uname -s)" in
  Darwin*)
    brew install postgresql
    open -a "Docker"
    ;;
  Linux*)
    sudo apt-get install -y postgresql-client
    sudo systemctl start docker
    ;;
  *)
    echo "Unsupported OS. Please install manually."
    exit 1
    ;;
esac
```

**問題点**: macOS だけを想定したスクリプトは、Linux 環境の CI/CD や WSL2 ユーザーが使えない。OS を検出して処理を分岐させることで、クロスプラットフォーム対応が実現できる。

---

## FAQ

### Q1: Make と npm scripts のどちらを使うべきですか？

**A**: 両方使うのが現実的。npm scripts はパッケージのライフサイクル（`prepare`, `pretest` 等）に適しており、Make は OS レベルのタスク（Docker 操作、DB マイグレーション、複数ステップのワークフロー）に適している。Makefile は `make help` で全コマンドを一覧表示でき、新メンバーのディスカバラビリティが高い。npm scripts と Make が重複する部分は、Make から npm scripts を呼ぶ形で統一すると管理しやすい。

### Q2: Windows 環境で Make を使えますか？

**A**: WSL2 (Windows Subsystem for Linux) 経由であれば問題なく使える。Git for Windows に含まれる MinGW の make も利用可能。ただし、Windows ネイティブ環境では make が入っていないことが多い。代替として `just`（Rust 製、Windows ネイティブ対応）や `Taskfile.yml`（Go 製、クロスプラットフォーム）を検討してもよい。Dev Container を使う場合は Linux 環境なので make は問題なく動作する。

### Q3: セットアップスクリプトはどのくらいの頻度でメンテナンスすべきですか？

**A**: 最低限、以下のタイミングで更新すべき: (1) Node.js バージョンの変更時、(2) 新しいサービス（DB、Redis 等）の追加時、(3) 依存ツールの変更時（npm → pnpm 移行等）。CI/CD で定期的にセットアップスクリプトを実行する「フレッシュインストールテスト」を組み込むことで、スクリプトの陳腐化を自動検知できる。月に一度、新メンバーの視点でスクリプトを実行してみるのも効果的。

### Q4: Dev Container を使えばセットアップスクリプトは不要ですか？

**A**: Dev Container は開発環境の完全な再現性を提供するが、セットアップスクリプトも依然として必要。Dev Container の `postCreateCommand` としてセットアップスクリプトを呼び出す設計が最も効果的。Dev Container を使わないメンバー（パフォーマンス要件やハードウェア制約がある場合）のためにも、ネイティブ環境用のセットアップスクリプトは維持すべき。

### Q5: モノレポの場合、セットアップはどう設計しますか？

**A**: モノレポでは、ルートの Makefile から各パッケージのセットアップを呼び出す構成が基本。

```makefile
# ルート Makefile
setup: ## 全パッケージのセットアップ
	pnpm install --frozen-lockfile
	$(MAKE) -C apps/web setup
	$(MAKE) -C apps/api setup
	$(MAKE) -C packages/shared setup

# apps/web/Makefile
setup:
	cp .env.example .env
```

Turborepo や Nx を使う場合は、`turbo run setup` でパッケージ間の依存関係を考慮した並列セットアップが可能。

### Q6: 企業のプロキシ環境ではどう対応しますか？

**A**: プロキシ環境ではセットアップスクリプトでプロキシ設定を検出・適用する処理を追加する。

```bash
# プロキシ設定の検出と適用
if [ -n "${HTTP_PROXY:-}" ]; then
  npm config set proxy "$HTTP_PROXY"
  npm config set https-proxy "${HTTPS_PROXY:-$HTTP_PROXY}"
  git config --global http.proxy "$HTTP_PROXY"
  echo "Proxy configured: $HTTP_PROXY"
fi
```

---

## まとめ

| 項目 | 要点 |
|------|------|
| setup.sh | OS 検出・ツールインストール・依存関係・DB セットアップを 1 コマンドで |
| Makefile | `make help` で全タスク一覧。新メンバーのエントリーポイント |
| doctor.sh | 環境の問題を自動診断。トラブルシューティングの第一歩 |
| doctor --fix | 自動修復可能な問題を自動的に修正 |
| .env.example | 環境変数のテンプレート。コメント付きで必要な値を明記 |
| .env バリデーション | 必須項目の存在チェックとセキュリティ検証 |
| エラーハンドリング | `set -euo pipefail` + 明確なエラーメッセージ |
| クロスプラットフォーム | macOS / Linux / WSL2 の差異を OS 検出で吸収 |
| CI 統合 | セットアップスクリプトを CI で定期実行して陳腐化を防止 |
| Docker Compose | ローカルサービスを宣言的に管理。ヘルスチェック付き |
| タスクランナー | Make (汎用) + npm scripts (パッケージ) の組み合わせが実用的 |
| オンボーディングチェック | 新メンバーの進捗を自動的にチェック |

## 次に読むべきガイド

- [ドキュメント環境](./02-documentation-setup.md) -- VitePress / Docusaurus / ADR によるドキュメント基盤
- [プロジェクト標準](./00-project-standards.md) -- EditorConfig / .npmrc の共通設定
- [Dev Container](../02-docker-dev/01-devcontainer.md) -- コンテナベース環境でオンボーディングをさらに簡素化

## 参考文献

1. **GNU Make マニュアル** -- https://www.gnu.org/software/make/manual/ -- Make の完全なリファレンス
2. **just (コマンドランナー)** -- https://github.com/casey/just -- Make の代替となる Rust 製タスクランナー
3. **The Twelve-Factor App - Dev/prod parity** -- https://12factor.net/ja/dev-prod-parity -- 開発環境と本番環境の一致原則
4. **Taskfile** -- https://taskfile.dev/ -- YAML ベースのタスクランナー
5. **Homebrew Bundle** -- https://github.com/Homebrew/homebrew-bundle -- macOS パッケージの宣言的管理
6. **direnv** -- https://direnv.net/ -- ディレクトリ別の環境変数管理
7. **Doppler** -- https://www.doppler.com/ -- チーム向けシークレット管理サービス

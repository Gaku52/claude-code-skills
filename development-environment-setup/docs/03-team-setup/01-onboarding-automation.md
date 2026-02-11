# オンボーディング自動化 (Onboarding Automation)

> 新メンバーが 1 コマンドで開発環境を構築できるセットアップスクリプトと Makefile を設計し、オンボーディングの時間を数日から数分に短縮する手法を学ぶ。

## この章で学ぶこと

1. **セットアップスクリプトの設計と実装** -- プラットフォーム差異を吸収し、依存ツールのインストールから初回ビルドまでを自動化するスクリプトを構築する
2. **Makefile によるタスクランナーの構築** -- よく使う開発タスクを `make` コマンドで標準化し、手順書の代わりにする
3. **環境検証と troubleshooting の自動化** -- セットアップの成否を自動検証し、問題発生時の診断情報を収集する仕組みを整備する

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
|  make lint && make test && make build                            |
|                                                                  |
+------------------------------------------------------------------+
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

echo ""
echo "=== Environment Doctor ==="
echo ""

echo "--- Tools ---"
check "Git" "command -v git" "git をインストールしてください"
check "Node.js" "command -v node" "Node.js をインストールしてください"
check "Docker" "command -v docker" "Docker をインストールしてください"
check "Docker Compose" "docker compose version" "Docker Compose V2 が必要です"

echo ""
echo "--- Node.js ---"
check "Node version matches .nvmrc" \
  "[ \"$(node -v | tr -d 'v')\" = \"$(cat .nvmrc 2>/dev/null | tr -d '[:space:]')\" ]" \
  "fnm use or nvm use を実行してください"
check "node_modules exists" "[ -d node_modules ]" "make setup を実行してください"
check "TypeScript compiles" "npx tsc --noEmit 2>/dev/null" ""

echo ""
echo "--- Services ---"
check "PostgreSQL reachable" "pg_isready -h localhost -p 5432 2>/dev/null" ""
check "Redis reachable" "redis-cli -h localhost ping 2>/dev/null" ""

echo ""
echo "--- Files ---"
check ".env exists" "[ -f .env ]" "cp .env.example .env を実行してください"
check ".env has DATABASE_URL" "grep -q DATABASE_URL .env 2>/dev/null" ""

echo ""
echo "==========================="
echo -e "Results: ${GREEN}${pass} passed${NC}, ${YELLOW}${warn} warnings${NC}, ${RED}${fail} failed${NC}"
echo ""

exit $fail
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

# ===== 外部 API (開発用ダミー値) =====
# STRIPE_SECRET_KEY=sk_test_xxx
# SENDGRID_API_KEY=SG.xxx
```

---

## 6. Task Runner 比較

| ツール | 設定ファイル | 言語 | 並列実行 | 依存管理 | 学習コスト |
|-------|-----------|------|---------|---------|-----------|
| Make | Makefile | シェル | 限定的 | ファイル依存 | 低 |
| npm scripts | package.json | シェル | npm-run-all | なし | 最低 |
| just | justfile | シェル | なし | なし | 低 |
| task (Taskfile) | Taskfile.yml | YAML+シェル | あり | タスク依存 | 低 |
| turbo | turbo.json | JSON | あり(高速) | パッケージ依存 | 中 |
| nx | nx.json | JSON | あり(高速) | プロジェクトグラフ | 高 |

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

---

## FAQ

### Q1: Make と npm scripts のどちらを使うべきですか？

**A**: 両方使うのが現実的。npm scripts はパッケージのライフサイクル（`prepare`, `pretest` 等）に適しており、Make は OS レベルのタスク（Docker 操作、DB マイグレーション、複数ステップのワークフロー）に適している。Makefile は `make help` で全コマンドを一覧表示でき、新メンバーのディスカバラビリティが高い。npm scripts と Make が重複する部分は、Make から npm scripts を呼ぶ形で統一すると管理しやすい。

### Q2: Windows 環境で Make を使えますか？

**A**: WSL2 (Windows Subsystem for Linux) 経由であれば問題なく使える。Git for Windows に含まれる MinGW の make も利用可能。ただし、Windows ネイティブ環境では make が入っていないことが多い。代替として `just`（Rust 製、Windows ネイティブ対応）や `Taskfile.yml`（Go 製、クロスプラットフォーム）を検討してもよい。Dev Container を使う場合は Linux 環境なので make は問題なく動作する。

### Q3: セットアップスクリプトはどのくらいの頻度でメンテナンスすべきですか？

**A**: 最低限、以下のタイミングで更新すべき: (1) Node.js バージョンの変更時、(2) 新しいサービス（DB、Redis 等）の追加時、(3) 依存ツールの変更時（npm → pnpm 移行等）。CI/CD で定期的にセットアップスクリプトを実行する「フレッシュインストールテスト」を組み込むことで、スクリプトの陳腐化を自動検知できる。月に一度、新メンバーの視点でスクリプトを実行してみるのも効果的。

---

## まとめ

| 項目 | 要点 |
|------|------|
| setup.sh | OS 検出・ツールインストール・依存関係・DB セットアップを 1 コマンドで |
| Makefile | `make help` で全タスク一覧。新メンバーのエントリーポイント |
| doctor.sh | 環境の問題を自動診断。トラブルシューティングの第一歩 |
| .env.example | 環境変数のテンプレート。コメント付きで必要な値を明記 |
| エラーハンドリング | `set -euo pipefail` + 明確なエラーメッセージ |
| CI 統合 | セットアップスクリプトを CI で定期実行して陳腐化を防止 |
| タスクランナー | Make (汎用) + npm scripts (パッケージ) の組み合わせが実用的 |

## 次に読むべきガイド

- [ドキュメント環境](./02-documentation-setup.md) -- VitePress / Docusaurus / ADR によるドキュメント基盤
- [プロジェクト標準](./00-project-standards.md) -- EditorConfig / .npmrc の共通設定
- [Dev Container](../02-docker-dev/01-devcontainer.md) -- コンテナベース環境でオンボーディングをさらに簡素化

## 参考文献

1. **GNU Make マニュアル** -- https://www.gnu.org/software/make/manual/ -- Make の完全なリファレンス
2. **just (コマンドランナー)** -- https://github.com/casey/just -- Make の代替となる Rust 製タスクランナー
3. **The Twelve-Factor App - Dev/prod parity** -- https://12factor.net/ja/dev-prod-parity -- 開発環境と本番環境の一致原則

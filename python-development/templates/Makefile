# Makefile Template for Python Projects
# タスク自動化のための Makefile

.PHONY: help install dev test lint format type clean build docs run

# デフォルトターゲット
.DEFAULT_GOAL := help

# 変数
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy
BLACK := $(PYTHON) -m black

# ヘルプ
help: ## このヘルプメッセージを表示
	@echo "利用可能なコマンド:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# インストール
install: ## 依存関係をインストール
	$(PIP) install -e .

dev: ## 開発環境をセットアップ
	$(PIP) install -e ".[dev]"
	pre-commit install

# テスト
test: ## テストを実行
	$(PYTEST) tests/ -v

test-cov: ## カバレッジ付きでテストを実行
	$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term

test-watch: ## ファイル変更を監視してテストを実行
	$(PYTEST) tests/ -v --looponfail

# コード品質
lint: ## Linter を実行
	$(RUFF) check src tests

lint-fix: ## Linter でエラーを自動修正
	$(RUFF) check --fix src tests

format: ## コードをフォーマット
	$(RUFF) format src tests

format-check: ## フォーマットをチェック（変更なし）
	$(RUFF) format --check src tests

type: ## 型チェックを実行
	$(MYPY) src

# すべてのチェック
check: lint type test ## すべてのチェックを実行

# クリーンアップ
clean: ## 生成ファイルを削除
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__
	rm -rf htmlcov coverage.xml .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# ビルド
build: clean ## パッケージをビルド
	$(PYTHON) -m build

publish: build ## PyPI に公開
	$(PYTHON) -m twine upload dist/*

publish-test: build ## Test PyPI に公開
	$(PYTHON) -m twine upload --repository testpypi dist/*

# ドキュメント
docs: ## ドキュメントをビルド
	cd docs && make html

docs-serve: docs ## ドキュメントをローカルサーバーで表示
	$(PYTHON) -m http.server --directory docs/_build/html 8080

# 実行
run: ## アプリケーションを実行
	$(PYTHON) -m uvicorn src.main:app --reload

run-prod: ## プロダクションモードで実行
	$(PYTHON) -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

# データベース
db-migrate: ## マイグレーションを作成
	alembic revision --autogenerate -m "$(message)"

db-upgrade: ## マイグレーションを適用
	alembic upgrade head

db-downgrade: ## マイグレーションをロールバック
	alembic downgrade -1

db-reset: ## データベースをリセット
	alembic downgrade base
	alembic upgrade head

# Docker
docker-build: ## Docker イメージをビルド
	docker build -t my-app:latest .

docker-run: ## Docker コンテナを実行
	docker run -p 8000:8000 --env-file .env my-app:latest

docker-compose-up: ## Docker Compose で起動
	docker-compose up -d

docker-compose-down: ## Docker Compose を停止
	docker-compose down

# Pre-commit
pre-commit: ## Pre-commit フックを実行
	pre-commit run --all-files

# セキュリティ
security: ## セキュリティチェックを実行
	$(PYTHON) -m bandit -r src -c pyproject.toml

# 依存関係の更新
update: ## 依存関係を更新
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -e ".[dev]"

# 仮想環境
venv: ## 仮想環境を作成
	$(PYTHON) -m venv venv
	@echo "仮想環境を有効化してください: source venv/bin/activate"

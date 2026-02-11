# Dockerfile Template for Python Applications
# Multi-stage ビルドによる最適化

# ===== Builder Stage =====
FROM python:3.11-slim as builder

# 環境変数
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 作業ディレクトリ
WORKDIR /build

# システムパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 依存関係をインストール
COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --user -e .

# ===== Runtime Stage =====
FROM python:3.11-slim

# 環境変数
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/appuser/.local/bin:$PATH

# 非 root ユーザーを作成
RUN useradd --create-home --shell /bin/bash appuser

# 作業ディレクトリ
WORKDIR /app

# ランタイム依存関係をインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Builder からファイルをコピー
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
COPY --chown=appuser:appuser src/ src/

# ユーザーを切り替え
USER appuser

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# ポートを公開
EXPOSE 8000

# アプリケーションを起動
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

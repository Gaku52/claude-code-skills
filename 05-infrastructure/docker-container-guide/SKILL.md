# Docker コンテナガイド

> Docker はモダン開発の基盤インフラ。コンテナの基礎概念、Dockerfile のベストプラクティス、Docker Compose によるマルチサービス構成、ネットワーキング、本番運用、オーケストレーション、セキュリティまで、Docker の全てを体系的に解説する。

## このSkillの対象者

- Docker を使った開発・運用を学びたいエンジニア
- コンテナ化されたアプリケーションを本番デプロイする方
- Docker Compose でマルチサービス環境を構築する方

## 前提知識

- Linux の基本コマンド
- Web アプリケーションの基本構造
- ネットワークの基礎知識

## 学習ガイド

### 00-fundamentals — コンテナの基礎

| # | ファイル | 内容 |
|---|---------|------|

### 01-dockerfile — Dockerfile ベストプラクティス

| # | ファイル | 内容 |
|---|---------|------|

### 02-compose — Docker Compose

| # | ファイル | 内容 |
|---|---------|------|

### 03-networking — ネットワーキング

| # | ファイル | 内容 |
|---|---------|------|

### 04-production — 本番運用

| # | ファイル | 内容 |
|---|---------|------|

### 05-orchestration — オーケストレーション

| # | ファイル | 内容 |
|---|---------|------|

### 06-security — セキュリティ

| # | ファイル | 内容 |
|---|---------|------|

## クイックリファレンス

```
Docker コマンド早見表:
  docker build -t app:latest .        — イメージビルド
  docker run -d -p 3000:3000 app      — コンテナ起動
  docker compose up -d                — Compose 起動
  docker compose down -v              — Compose 停止+ボリューム削除
  docker logs -f <container>          — ログ追跡
  docker exec -it <container> sh      — コンテナ内シェル
  docker system prune -a              — 不要リソース削除

Dockerfile ベストプラクティス:
  ✓ マルチステージビルドでサイズ削減
  ✓ 非 root ユーザーで実行
  ✓ .dockerignore でビルドコンテキスト最適化
  ✓ COPY --chown でファイル所有者設定
  ✓ ヘルスチェック（HEALTHCHECK）設定
  ✓ 固定バージョンのベースイメージ
```

## 参考文献

1. Docker. "Documentation." docs.docker.com, 2024.
2. Docker. "Dockerfile Best Practices." docs.docker.com, 2024.
3. Kubernetes. "Documentation." kubernetes.io/docs, 2024.

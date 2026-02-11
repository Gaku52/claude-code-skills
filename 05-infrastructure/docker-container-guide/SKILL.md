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
| 00 | [[docs/00-fundamentals/00-container-basics.md]] | コンテナ vs VM、Docker アーキテクチャ、イメージとレイヤー |
| 01 | [[docs/00-fundamentals/01-docker-commands.md]] | 基本コマンド、ライフサイクル、ログ、exec、ボリューム |
| 02 | [[docs/00-fundamentals/02-image-management.md]] | イメージのビルド・プッシュ・プル、レジストリ、タグ戦略 |
| 03 | [[docs/00-fundamentals/03-storage-and-volumes.md]] | ボリューム、バインドマウント、tmpfs、データ永続化 |

### 01-dockerfile — Dockerfile ベストプラクティス

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-dockerfile/00-dockerfile-basics.md]] | 命令一覧、FROM/RUN/COPY/CMD/ENTRYPOINT の使い分け |
| 01 | [[docs/01-dockerfile/01-multi-stage-build.md]] | マルチステージビルド、ビルダーパターン、サイズ最適化 |
| 02 | [[docs/01-dockerfile/02-optimization.md]] | レイヤーキャッシュ、.dockerignore、BuildKit、distroless |
| 03 | [[docs/01-dockerfile/03-language-specific.md]] | Node.js / Python / Go / Rust 向け Dockerfile テンプレート |

### 02-compose — Docker Compose

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-compose/00-compose-basics.md]] | compose.yaml 構文、サービス定義、環境変数、依存関係 |
| 01 | [[docs/02-compose/01-development-setup.md]] | 開発環境 Compose、ホットリロード、デバッグ、プロファイル |
| 02 | [[docs/02-compose/02-production-compose.md]] | 本番 Compose、ヘルスチェック、リソース制限、ログ管理 |

### 03-networking — ネットワーキング

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-networking/00-docker-networking.md]] | ネットワークドライバー、ブリッジ、ホスト、DNS 解決 |
| 01 | [[docs/03-networking/01-reverse-proxy.md]] | Nginx/Traefik リバースプロキシ、SSL 終端、ロードバランシング |

### 04-production — 本番運用

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/04-production/00-production-best-practices.md]] | 非 root 実行、ヘルスチェック、graceful shutdown、ロギング |
| 01 | [[docs/04-production/01-ci-cd-integration.md]] | GitHub Actions でのビルド・プッシュ、マルチプラットフォーム |
| 02 | [[docs/04-production/02-monitoring.md]] | コンテナ監視、Prometheus/Grafana、リソース管理 |

### 05-orchestration — オーケストレーション

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/05-orchestration/00-kubernetes-basics.md]] | Kubernetes 入門、Pod/Service/Deployment の基礎 |
| 01 | [[docs/05-orchestration/01-docker-swarm.md]] | Docker Swarm、サービス定義、スケーリング |

### 06-security — セキュリティ

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/06-security/00-container-security.md]] | イメージスキャン、脆弱性対策、シークレット管理、rootless |
| 01 | [[docs/06-security/01-runtime-security.md]] | seccomp、AppArmor、capabilities、read-only ファイルシステム |

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

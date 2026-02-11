# 本番ベストプラクティス

> Docker本番環境で必須となる非rootユーザー実行、ヘルスチェック、リソース制限、ログ戦略の4本柱を体系的に習得する。

---

## この章で学ぶこと

1. **非rootユーザーでのコンテナ実行**とセキュリティ強化の手法を理解する
2. **ヘルスチェックとリソース制限**による堅牢な運用設計を習得する
3. **構造化ログとログドライバー**を活用した効率的なログ戦略を構築できるようになる

---

## 1. 非rootユーザーでの実行

コンテナのデフォルトはrootで実行される。これはコンテナエスケープ脆弱性が悪用された場合にホストのroot権限が奪取されるリスクを意味する。

### コード例1: 非rootユーザーの設定

```dockerfile
# Dockerfile - Node.jsアプリケーション
FROM node:20-alpine

# アプリケーションディレクトリを作成
WORKDIR /app

# 依存関係をインストール（rootで実行）
COPY package*.json ./
RUN npm ci --only=production

# アプリケーションコードをコピー
COPY --chown=node:node . .

# 非rootユーザーに切り替え
USER node

EXPOSE 3000
CMD ["node", "server.js"]
```

```dockerfile
# Dockerfile - Pythonアプリケーション（ユーザー作成パターン）
FROM python:3.12-slim

RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/false --create-home appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appgroup . .

USER appuser

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:create_app()"]
```

### rootユーザーの危険性

```
┌─────────────────────────────────────────────────┐
│  Container (root)                               │
│  UID=0                                          │
│                                                 │
│  コンテナエスケープ脆弱性                        │
│       │                                         │
│       ▼                                         │
│  ┌─────────────────────────────────────┐       │
│  │  Host (root)                        │       │
│  │  UID=0 → ホスト全体を掌握           │       │
│  └─────────────────────────────────────┘       │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  Container (non-root)                           │
│  UID=1001                                       │
│                                                 │
│  コンテナエスケープ脆弱性                        │
│       │                                         │
│       ▼                                         │
│  ┌─────────────────────────────────────┐       │
│  │  Host (UID=1001)                    │       │
│  │  権限なし → 被害を最小限に抑制       │       │
│  └─────────────────────────────────────┘       │
└─────────────────────────────────────────────────┘
```

---

## 2. ヘルスチェック

### コード例2: 各種ヘルスチェック設定

```dockerfile
# Dockerfile内でのヘルスチェック定義
FROM nginx:alpine

# HTTPエンドポイントによるヘルスチェック
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:80/health || exit 1
```

```dockerfile
# PostgreSQL用のヘルスチェック
FROM postgres:16-alpine

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
  CMD pg_isready -U postgres -d mydb || exit 1
```

```yaml
# docker-compose.yml でのヘルスチェック定義
version: "3.9"

services:
  api:
    image: my-api:latest
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 15s
      timeout: 5s
      retries: 3
      start_period: 30s
      start_interval: 5s  # 起動期間中のチェック間隔（Compose v2.3+）

  postgres:
    image: postgres:16-alpine
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  # 依存サービスのヘルスチェックを待ってから起動
  app:
    image: my-app:latest
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
```

### ヘルスチェックパラメータ比較表

| パラメータ | 説明 | 推奨値 | 注意点 |
|-----------|------|--------|--------|
| interval | チェック間隔 | 10-30s | 短すぎると負荷増大 |
| timeout | タイムアウト | 3-10s | intervalより短く設定 |
| retries | 失敗許容回数 | 3-5 | 一時的な障害を許容 |
| start_period | 起動猶予期間 | 10-60s | アプリの起動時間に合わせる |

---

## 3. リソース制限

### コード例3: メモリとCPUの制限

```yaml
# docker-compose.yml
version: "3.9"

services:
  api:
    image: my-api:latest
    deploy:
      resources:
        limits:
          memory: 512M       # ハード上限（超過でOOM Kill）
          cpus: "1.0"        # CPU 1コア分
        reservations:
          memory: 256M       # 最低保証メモリ
          cpus: "0.25"       # 最低保証CPU

  worker:
    image: my-worker:latest
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: "2.0"
        reservations:
          memory: 512M
          cpus: "0.5"
      # OOM優先度（OOMスコア調整）
    oom_score_adj: 100  # 正の値 → OOM Kill されやすい

  database:
    image: postgres:16-alpine
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "2.0"
        reservations:
          memory: 1G
          cpus: "1.0"
    oom_score_adj: -500  # 負の値 → OOM Kill されにくい
```

```bash
# docker run でのリソース制限
docker run -d \
  --name api \
  --memory=512m \
  --memory-swap=512m \       # スワップ無効化（メモリと同値）
  --memory-reservation=256m \
  --cpus=1.0 \
  --cpu-shares=512 \         # 相対的なCPU配分（デフォルト1024）
  --pids-limit=100 \         # プロセス数上限（fork爆弾対策）
  --ulimit nofile=65535:65535 \  # ファイルディスクリプタ上限
  my-api:latest

# リソース使用状況のリアルタイム監視
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.PIDs}}"
```

### リソース制限の動作

```
┌──────────────────────────────────────────┐
│            Docker Host (8GB RAM)         │
│                                          │
│  ┌──────────────┐  ┌──────────────┐    │
│  │   API        │  │   Worker     │    │
│  │  limit: 512M │  │  limit: 1G   │    │
│  │  ┌────────┐  │  │  ┌────────┐  │    │
│  │  │使用:   │  │  │  │使用:   │  │    │
│  │  │ 300M   │  │  │  │ 800M   │  │    │
│  │  └────────┘  │  │  └────────┘  │    │
│  │              │  │              │    │
│  │  512M到達 → │  │  1G到達 →   │    │
│  │  OOM Kill!  │  │  OOM Kill!  │    │
│  └──────────────┘  └──────────────┘    │
│                                          │
│  reservations: 最低保証                  │
│  limits: ハード上限（超過でOOM Kill）    │
└──────────────────────────────────────────┘
```

---

## 4. ログ戦略

### コード例4: 構造化ログの設計

```dockerfile
# Dockerfile - ログ設計のベストプラクティス
FROM node:20-alpine

WORKDIR /app
COPY . .

# アプリケーションは stdout/stderr に出力する
# ファイルへの書き込みは行わない
CMD ["node", "server.js"]

# server.js 内のログ出力例:
# console.log(JSON.stringify({
#   timestamp: new Date().toISOString(),
#   level: "info",
#   message: "Request handled",
#   method: "GET",
#   path: "/api/users",
#   status: 200,
#   duration_ms: 45,
#   request_id: "abc-123"
# }));
```

```yaml
# docker-compose.yml - ログドライバー設定
version: "3.9"

services:
  api:
    image: my-api:latest
    logging:
      driver: json-file    # デフォルトドライバー
      options:
        max-size: "10m"    # ログファイルの最大サイズ
        max-file: "5"      # ローテーションファイル数
        compress: "true"   # 圧縮有効化
        tag: "{{.Name}}/{{.ID}}"  # タグ付け

  # Fluentdへの転送
  worker:
    image: my-worker:latest
    logging:
      driver: fluentd
      options:
        fluentd-address: "localhost:24224"
        tag: "docker.{{.Name}}"
        fluentd-async: "true"     # 非同期送信（ログ損失のリスクあり）
        fluentd-retry-wait: "1s"
        fluentd-max-retries: "10"
```

### ログ出力のベストプラクティス比較表

| 方針 | 推奨 | 非推奨 | 理由 |
|------|------|--------|------|
| 出力先 | stdout / stderr | ファイル | Docker ログドライバーが処理 |
| フォーマット | JSON構造化 | プレーンテキスト | パース・フィルタリングが容易 |
| レベル管理 | 環境変数で制御 | ハードコード | 本番ではINFO以上のみ出力 |
| ローテーション | Dockerドライバーに委任 | アプリ内logrotate | 統一管理が可能 |

---

## 5. Graceful Shutdown

### コード例5: シグナルハンドリング

```javascript
// server.js - Node.js のGraceful Shutdown
const http = require("http");

const server = http.createServer((req, res) => {
  res.writeHead(200);
  res.end("OK");
});

server.listen(3000, () => {
  console.log("Server started on port 3000");
});

// SIGTERM: docker stop が送信するシグナル
process.on("SIGTERM", () => {
  console.log("SIGTERM received. Shutting down gracefully...");

  server.close(() => {
    console.log("HTTP server closed");
    // DB接続のクリーンアップ
    // メッセージキューの切断
    process.exit(0);
  });

  // 強制終了のタイムアウト（SIGKILLの前に自主終了）
  setTimeout(() => {
    console.error("Forced shutdown after timeout");
    process.exit(1);
  }, 10000);
});
```

```dockerfile
# Dockerfile - 正しいエントリポイント設定
FROM node:20-alpine

WORKDIR /app
COPY . .

# NG: shell形式（シグナルが /bin/sh に届き、nodeプロセスに伝わらない）
# CMD node server.js

# OK: exec形式（nodeプロセスがPID 1として起動し、シグナルを直接受信）
CMD ["node", "server.js"]

# または、tiniを使用してPID 1問題を解決
# RUN apk add --no-cache tini
# ENTRYPOINT ["tini", "--"]
# CMD ["node", "server.js"]
```

```yaml
# docker-compose.yml
services:
  api:
    image: my-api:latest
    stop_grace_period: 30s  # SIGTERMからSIGKILLまでの猶予時間
    stop_signal: SIGTERM    # デフォルト
```

### シグナルハンドリングのフロー

```
docker stop コンテナ
    │
    ▼
SIGTERM をPID 1に送信
    │
    ▼
┌────────────────────────────────────────┐
│  アプリケーション                       │
│  1. 新規リクエストの受付を停止          │
│  2. 処理中のリクエストを完了            │
│  3. DB接続をクローズ                    │
│  4. ファイルハンドルをクローズ          │
│  5. exit(0) で正常終了                  │
└────────────────────────────────────────┘
    │
    │ stop_grace_period 経過（デフォルト10秒）
    ▼
SIGKILL を送信（強制終了）
```

---

## 6. 本番用Dockerfileのテンプレート

### コード例6: 本番グレードのマルチステージDockerfile

```dockerfile
# === ビルドステージ ===
FROM node:20-alpine AS builder

WORKDIR /app

# 依存関係のインストール（キャッシュ活用）
COPY package.json package-lock.json ./
RUN npm ci

# ソースコードのコピーとビルド
COPY . .
RUN npm run build

# 不要な開発依存関係を除去
RUN npm prune --production

# === 本番ステージ ===
FROM node:20-alpine AS production

# セキュリティアップデート
RUN apk update && apk upgrade --no-cache && \
    apk add --no-cache tini dumb-init

# 非rootユーザー
RUN addgroup -g 1001 -S appgroup && \
    adduser -u 1001 -S appuser -G appgroup

WORKDIR /app

# ビルド成果物のみコピー
COPY --from=builder --chown=appuser:appgroup /app/dist ./dist
COPY --from=builder --chown=appuser:appgroup /app/node_modules ./node_modules
COPY --from=builder --chown=appuser:appgroup /app/package.json ./

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

# メタデータ
LABEL maintainer="team@example.com" \
      version="1.0.0" \
      description="Production API server"

# 非rootユーザーで実行
USER appuser

EXPOSE 3000

# tiniでPID 1問題を解決
ENTRYPOINT ["tini", "--"]
CMD ["node", "dist/server.js"]
```

---

## 7. 本番チェックリスト

### コード例7: docker-compose本番設定

```yaml
# docker-compose.prod.yml
version: "3.9"

services:
  api:
    image: registry.example.com/api:${VERSION:-latest}
    restart: unless-stopped        # 自動再起動
    read_only: true                # ルートFS読み取り専用
    tmpfs:
      - /tmp:size=100m,noexec     # tmpのみ書き込み可
    security_opt:
      - no-new-privileges:true     # 権限昇格を禁止
    cap_drop:
      - ALL                        # 全Capabilityを削除
    cap_add:
      - NET_BIND_SERVICE           # 必要なもののみ追加
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: "1.0"
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:3000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "5"
    environment:
      NODE_ENV: production
      LOG_LEVEL: info
    networks:
      - app-net
```

---

## アンチパターン

### アンチパターン1: rootでのコンテナ実行

```dockerfile
# NG: USERを指定しない（rootで実行される）
FROM node:20-alpine
WORKDIR /app
COPY . .
CMD ["node", "server.js"]

# OK: 専用ユーザーで実行
FROM node:20-alpine
WORKDIR /app
COPY . .
USER node
CMD ["node", "server.js"]
```

**なぜ問題か**: rootで実行されたコンテナが侵害されると、ホストのroot権限が奪取される可能性がある。最小権限の原則に従い、専用ユーザーで実行する。

### アンチパターン2: リソース制限なしでの本番運用

```yaml
# NG: リソース制限なし
services:
  api:
    image: my-api:latest
    # → メモリリークで他のコンテナを巻き込んでホストがクラッシュ

# OK: 適切なリソース制限を設定
services:
  api:
    image: my-api:latest
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: "1.0"
```

**なぜ問題か**: リソース制限のないコンテナがメモリリークを起こすと、ホスト全体のメモリを消費し、他の全コンテナとホストOSに影響する。

### アンチパターン3: ログファイルのコンテナ内蓄積

```bash
# NG: ログをコンテナ内のファイルに書き込み
# アプリが /var/log/app.log に書き込む → コンテナサイズ肥大化

# OK: stdout/stderrに出力し、Dockerログドライバーに委任
# console.log(), print(), fmt.Println() を使用
```

**なぜ問題か**: コンテナ内のファイルシステムは一時的で、コンテナ再起動でログが消失する。またコンテナのディスク使用量が増大し続ける。

---

## FAQ

### Q1: `restart: always` と `restart: unless-stopped` の違いは？

`always` はDocker デーモン起動時に常にコンテナを再起動する。`unless-stopped` はユーザーが明示的に `docker stop` したコンテナはデーモン再起動時に再起動しない。本番では `unless-stopped` を推奨。メンテナンスのために停止したコンテナが意図せず再起動されることを防ぐ。

### Q2: ヘルスチェックのテストコマンドに curl と wget のどちらを使うべき？

alpineベースイメージには `wget` が含まれているが `curl` は含まれていない。追加パッケージのインストールはイメージサイズ増加につながるため、alpineベースでは `wget` を使う。Debianベースでは `curl` が利用可能。最も軽量な方法はアプリケーション内にヘルスチェック用CLIを組み込むこと。

### Q3: `read_only: true` でアプリケーションが動作しない場合の対処法は？

`tmpfs` マウントで一時書き込み領域を提供する。多くのアプリケーションは `/tmp` や `/var/run` への書き込みが必要。

```yaml
services:
  api:
    read_only: true
    tmpfs:
      - /tmp:size=100m
      - /var/run:size=10m
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 非rootユーザー | `USER` 命令で専用ユーザーに切り替え。最小権限の原則 |
| ヘルスチェック | interval/timeout/retries/start_period の4パラメータを適切に設定 |
| リソース制限 | memory limits必須。cpus/pids-limitも設定。OOM Kill対策 |
| ログ戦略 | stdout/stderrへのJSON構造化出力。ログドライバーで転送 |
| Graceful Shutdown | SIGTERMハンドリング。exec形式CMD。tini活用 |
| 読取専用FS | `read_only: true` + tmpfsで書き込みを最小化 |
| Capability | `cap_drop: ALL` + 必要なもののみ `cap_add` |

---

## 次に読むべきガイド

- [モニタリング](./01-monitoring.md) -- Prometheus/Grafanaによる監視体制の構築
- [Docker CI/CD](./02-ci-cd-docker.md) -- ビルド・デプロイ自動化パイプライン
- [コンテナセキュリティ](../06-security/00-container-security.md) -- セキュリティの包括的な実践

---

## 参考文献

1. Docker公式ドキュメント "Docker security" -- https://docs.docker.com/engine/security/
2. CIS Docker Benchmark -- https://www.cisecurity.org/benchmark/docker
3. NIST SP 800-190 "Application Container Security Guide" -- https://csrc.nist.gov/publications/detail/sp/800-190/final
4. Liz Rice (2020) *Container Security: Fundamental Technology Concepts that Protect Containerized Applications*, O'Reilly
5. Docker公式ドキュメント "Configure logging drivers" -- https://docs.docker.com/config/containers/logging/

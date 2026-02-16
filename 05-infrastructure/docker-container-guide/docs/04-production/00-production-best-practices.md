# 本番ベストプラクティス

> Docker本番環境で必須となる非rootユーザー実行、ヘルスチェック、リソース制限、ログ戦略の4本柱を体系的に習得する。

---

## この章で学ぶこと

1. **非rootユーザーでのコンテナ実行**とセキュリティ強化の手法を理解する
2. **ヘルスチェックとリソース制限**による堅牢な運用設計を習得する
3. **構造化ログとログドライバー**を活用した効率的なログ戦略を構築できるようになる
4. **Graceful Shutdown**とシグナルハンドリングの正しい実装パターンを身につける
5. **本番用Dockerfile**と**docker-compose設定**のセキュリティ・パフォーマンス最適化を実践できるようになる

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

```dockerfile
# Dockerfile - Goアプリケーション（スクラッチベース）
FROM golang:1.22-alpine AS builder

WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /app/server ./cmd/server

# 本番ステージ: scratchベースで最小構成
FROM scratch

# 非rootユーザーを設定（/etc/passwdをコピー）
COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /etc/group /etc/group

# TLS証明書をコピー（外部HTTPSアクセス用）
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# バイナリをコピー
COPY --from=builder --chown=1001:1001 /app/server /server

USER 1001

EXPOSE 8080
ENTRYPOINT ["/server"]
```

```dockerfile
# Dockerfile - Javaアプリケーション（Spring Boot）
FROM eclipse-temurin:21-jre-alpine

RUN addgroup -g 1001 -S spring && \
    adduser -u 1001 -S spring -G spring -s /bin/false

WORKDIR /app

COPY --chown=spring:spring target/*.jar app.jar

# JVMのセキュリティ設定
ENV JAVA_OPTS="-XX:+UseContainerSupport \
    -XX:MaxRAMPercentage=75.0 \
    -Djava.security.egd=file:/dev/./urandom"

USER spring

EXPOSE 8080
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]
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

### User Namespace Remapping

Docker ホストレベルでの追加防御として、User Namespace Remapping を設定できる。これにより、コンテナ内の root (UID=0) がホスト上では非特権UID にマッピングされる。

```json
// /etc/docker/daemon.json
{
  "userns-remap": "default"
}
```

```bash
# User Namespace Remapping の確認
# コンテナ内で root として実行されていても
# ホスト上では別のUIDにマッピングされる
docker run --rm alpine id
# uid=0(root) gid=0(root) ← コンテナ内では root

# ホスト上での実際のUID確認
ps aux | grep "コンテナプロセス"
# 165536 (非特権UID) で実行されている
```

### Rootless Docker

Docker デーモン自体を非rootで実行する Rootless モードも本番環境で検討すべきオプションである。

```bash
# Rootless Docker のインストール
curl -fsSL https://get.docker.com/rootless | sh

# 環境変数の設定
export PATH=$HOME/bin:$PATH
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock

# Rootless Docker の動作確認
docker info | grep -i "root"
# rootless: true
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

```dockerfile
# Redis用のヘルスチェック
FROM redis:7-alpine

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD redis-cli ping | grep -q PONG || exit 1
```

```dockerfile
# MongoDB用のヘルスチェック
FROM mongo:7

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD mongosh --eval "db.adminCommand('ping').ok" --quiet || exit 1
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
| start_interval | 起動中チェック間隔 | 3-5s | 起動完了を素早く検知 |

### ヘルスチェックのベストプラクティス

```
ヘルスチェック設計の判断フロー:

1. エンドポイントの選択
   ├── Webアプリ → HTTP GET /health
   ├── データベース → 専用コマンド (pg_isready, redis-cli ping)
   ├── メッセージキュー → 接続確認
   └── バッチ処理 → プロセス存在確認 or ファイルタイムスタンプ

2. チェック内容の深さ
   ├── Shallow (浅い): プロセスが応答するか
   │   └── 高速、低負荷、基本的な死活監視
   ├── Medium (中程度): 依存サービスとの接続確認
   │   └── DB接続プール、キャッシュ接続
   └── Deep (深い): 完全な機能テスト
       └── 高コスト、本番では注意して使用

3. 推奨: /health は Shallow、/ready は Medium
```

### アプリケーション側のヘルスチェックエンドポイント実装

```javascript
// Node.js/Express - ヘルスチェックエンドポイント
const express = require("express");
const app = express();

// Shallow Health Check（Liveness用）
app.get("/health", (req, res) => {
  res.status(200).json({
    status: "ok",
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
  });
});

// Deep Health Check（Readiness用）
app.get("/ready", async (req, res) => {
  const checks = {};
  let isReady = true;

  // データベース接続チェック
  try {
    await db.query("SELECT 1");
    checks.database = "ok";
  } catch (err) {
    checks.database = "error";
    isReady = false;
  }

  // Redis接続チェック
  try {
    await redis.ping();
    checks.redis = "ok";
  } catch (err) {
    checks.redis = "error";
    isReady = false;
  }

  // 外部API接続チェック
  try {
    await fetch("https://api.external.com/status", { timeout: 3000 });
    checks.externalApi = "ok";
  } catch (err) {
    checks.externalApi = "error";
    isReady = false;
  }

  const statusCode = isReady ? 200 : 503;
  res.status(statusCode).json({
    status: isReady ? "ready" : "not_ready",
    checks,
    timestamp: new Date().toISOString(),
  });
});
```

```python
# Python/FastAPI - ヘルスチェックエンドポイント
from fastapi import FastAPI, Response
from datetime import datetime
import asyncpg
import aioredis

app = FastAPI()

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get("/ready")
async def readiness_check(response: Response):
    checks = {}
    is_ready = True

    # データベースチェック
    try:
        conn = await asyncpg.connect(dsn=DATABASE_URL)
        await conn.fetchval("SELECT 1")
        await conn.close()
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "error"
        is_ready = False

    # Redisチェック
    try:
        redis = await aioredis.from_url(REDIS_URL)
        await redis.ping()
        await redis.close()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "error"
        is_ready = False

    if not is_ready:
        response.status_code = 503

    return {
        "status": "ready" if is_ready else "not_ready",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }
```

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

### 言語ランタイム別のメモリ設定

各言語ランタイムには、コンテナのメモリ制限を認識するための設定が必要な場合がある。

```bash
# Java - コンテナのメモリ制限を自動認識
# JDK 8u191+ / JDK 11+ では UseContainerSupport がデフォルト有効
docker run -d \
  --memory=512m \
  -e JAVA_OPTS="-XX:+UseContainerSupport -XX:MaxRAMPercentage=75.0" \
  my-java-app:latest

# Node.js - ヒープサイズ制限
docker run -d \
  --memory=512m \
  -e NODE_OPTIONS="--max-old-space-size=384" \
  my-node-app:latest

# Python - メモリ制限はOS依存（特別な設定は不要だが監視は必要）
docker run -d \
  --memory=512m \
  -e PYTHONDONTWRITEBYTECODE=1 \
  my-python-app:latest

# Go - GOMEMLIMIT で GC を最適化（Go 1.19+）
docker run -d \
  --memory=512m \
  -e GOMEMLIMIT=400MiB \
  my-go-app:latest
```

### リソース使用量のサイジング指針

| サービスタイプ | メモリ目安 | CPU目安 | 備考 |
|---------------|-----------|---------|------|
| Webフロントエンド (nginx) | 64-128M | 0.1-0.5 | 静的配信は軽量 |
| APIサーバー (Node.js) | 256-512M | 0.25-1.0 | ヒープサイズに注意 |
| APIサーバー (Java) | 512M-2G | 0.5-2.0 | JVMヒープサイズ設定必須 |
| ワーカー/バッチ | 512M-4G | 1.0-4.0 | 処理内容に大きく依存 |
| PostgreSQL | 1-4G | 1.0-4.0 | shared_buffers = メモリの25% |
| Redis | 256M-2G | 0.5-1.0 | maxmemory設定必須 |
| Elasticsearch | 2-8G | 2.0-4.0 | ヒープ = メモリの50% |

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
| 相関ID | request_id / trace_id を含める | ID なし | 分散トレーシングに不可欠 |
| 機密情報 | マスクまたは除外 | そのまま出力 | パスワード・トークンの漏洩防止 |

### ログドライバー比較

| ドライバー | 特徴 | ユースケース | `docker logs` 対応 |
|-----------|------|-------------|-------------------|
| json-file | デフォルト、JSONで保存 | 小規模、開発 | 対応 |
| local | 最適化されたファイル形式 | 単一ホスト本番 | 対応 |
| fluentd | Fluentdに転送 | 中〜大規模 | 非対応 |
| syslog | syslogに転送 | Linuxネイティブ | 非対応 |
| awslogs | CloudWatch Logsに転送 | AWS環境 | 非対応 |
| gcplogs | Cloud Loggingに転送 | GCP環境 | 非対応 |
| gelf | Graylogに転送 | Graylog利用時 | 非対応 |

### Docker デーモンレベルのログ設定

```json
// /etc/docker/daemon.json - 全コンテナ共通のログ設定
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3",
    "compress": "true",
    "labels": "environment,service",
    "tag": "{{.ImageName}}/{{.Name}}/{{.ID}}"
  }
}
```

### 構造化ログの実装パターン（各言語）

```python
# Python - structlog を使った構造化ログ
import structlog
import logging

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()

# 使用例
logger.info("request_handled",
    method="GET",
    path="/api/users",
    status=200,
    duration_ms=45,
    request_id="abc-123",
)
# 出力: {"event":"request_handled","method":"GET","path":"/api/users","status":200,"duration_ms":45,"request_id":"abc-123","timestamp":"2024-01-15T10:30:00Z","level":"info"}
```

```go
// Go - slog を使った構造化ログ（Go 1.21+）
package main

import (
    "log/slog"
    "os"
)

func main() {
    // JSON形式でstdoutに出力
    logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
        Level: slog.LevelInfo,
    }))
    slog.SetDefault(logger)

    // 使用例
    slog.Info("request_handled",
        "method", "GET",
        "path", "/api/users",
        "status", 200,
        "duration_ms", 45,
        "request_id", "abc-123",
    )
}
```

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

```python
# Python/FastAPI - Graceful Shutdown
import signal
import asyncio
import uvicorn
from fastapi import FastAPI

app = FastAPI()
shutdown_event = asyncio.Event()

@app.on_event("shutdown")
async def shutdown():
    print("Shutting down gracefully...")
    # DB接続プールのクローズ
    await database.disconnect()
    # バックグラウンドタスクの完了待ち
    await task_queue.close()
    print("Cleanup completed")

# uvicornはSIGTERMを自動的にハンドリング
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_graceful_shutdown=30,
    )
```

```go
// Go - Graceful Shutdown
package main

import (
    "context"
    "log"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"
)

func main() {
    srv := &http.Server{Addr: ":8080"}

    // シグナルハンドリング
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGTERM, syscall.SIGINT)

    go func() {
        if err := srv.ListenAndServe(); err != http.ErrServerClosed {
            log.Fatalf("Server error: %v", err)
        }
    }()

    log.Println("Server started on :8080")

    // シグナル待ち
    sig := <-sigChan
    log.Printf("Received signal: %s. Shutting down...", sig)

    // Graceful Shutdown（30秒タイムアウト）
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    if err := srv.Shutdown(ctx); err != nil {
        log.Printf("Forced shutdown: %v", err)
    }

    log.Println("Server stopped")
}
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

### PID 1 問題と tini/dumb-init

コンテナ内のPID 1プロセスには、通常のLinuxプロセスと異なる特殊な挙動がある。

```
PID 1 の特殊性:
- SIGTERMのデフォルト動作（終了）が適用されない
- 子プロセスの終了（ゾンビプロセス）を回収する責任がある
- シェル形式の CMD では /bin/sh が PID 1 になり、
  アプリケーションプロセスにシグナルが伝播しない

解決策:
┌─────────────────────────────────────────────┐
│ 1. exec形式のCMD（推奨）                     │
│    CMD ["node", "server.js"]                 │
│    → node が PID 1 として直接シグナルを受信   │
│                                              │
│ 2. tini / dumb-init の使用（より堅牢）        │
│    ENTRYPOINT ["tini", "--"]                 │
│    CMD ["node", "server.js"]                 │
│    → tini が PID 1、node は PID 2            │
│    → ゾンビプロセス回収 + シグナル転送        │
│                                              │
│ 3. Docker の --init フラグ                    │
│    docker run --init my-app:latest           │
│    → Docker が自動的に tini を注入            │
└─────────────────────────────────────────────┘
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

### Python 本番Dockerfile

```dockerfile
# === ビルドステージ ===
FROM python:3.12-slim AS builder

WORKDIR /app

# 仮想環境を使用してシステムPythonを汚染しない
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# === 本番ステージ ===
FROM python:3.12-slim AS production

# セキュリティアップデート
RUN apt-get update && apt-get upgrade -y --no-install-recommends && \
    apt-get install -y --no-install-recommends tini wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 非rootユーザー
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -s /bin/false -m appuser

WORKDIR /app

# 仮想環境をコピー
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# アプリケーションコードをコピー
COPY --chown=appuser:appgroup . .

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

# 環境変数
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

ENTRYPOINT ["tini", "--"]
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120", "app:create_app()"]
```

### Go 本番Dockerfile

```dockerfile
# === ビルドステージ ===
FROM golang:1.22-alpine AS builder

RUN apk add --no-cache ca-certificates tzdata

WORKDIR /build

COPY go.mod go.sum ./
RUN go mod download && go mod verify

COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w -X main.version=$(cat VERSION)" \
    -o /app/server ./cmd/server

# === 本番ステージ（distroless） ===
FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /app/server /server

EXPOSE 8080

USER nonroot:nonroot

ENTRYPOINT ["/server"]
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

### 本番前デプロイチェックリスト

以下のチェックリストに全て合格してから本番デプロイを実施する。

```
## セキュリティチェック
□ 非rootユーザーで実行 (USER命令)
□ read_only: true 設定
□ cap_drop: ALL + 必要な cap_add のみ
□ no-new-privileges: true
□ 機密情報は環境変数 or シークレット管理
□ ベースイメージにセキュリティアップデート適用
□ Trivyでイメージスキャン済み（CRITICAL/HIGH なし）
□ .dockerignore で .env, .git, node_modules を除外

## 信頼性チェック
□ HEALTHCHECK 定義済み
□ restart: unless-stopped 設定
□ メモリ制限 (deploy.resources.limits.memory)
□ CPU制限 (deploy.resources.limits.cpus)
□ Graceful Shutdown 実装 (SIGTERM ハンドリング)
□ stop_grace_period 設定
□ 依存サービスの healthcheck + depends_on condition

## ログ・監視チェック
□ ログは stdout/stderr に出力
□ JSON構造化ログ
□ ログローテーション設定 (max-size, max-file)
□ /health エンドポイント実装
□ /metrics エンドポイント実装 (Prometheus)
□ request_id / trace_id をログに含める

## イメージチェック
□ マルチステージビルド（本番ステージにビルドツール不要）
□ 明示的なバージョンタグ（latestタグ不使用）
□ .dockerignore で不要ファイル除外
□ LABEL でメタデータ付与
□ exec形式の CMD（shell形式でない）
□ tini or dumb-init で PID 1 問題を解決

## ネットワークチェック
□ 不要なポートを EXPOSE していない
□ 内部通信用ネットワークは internal: true
□ TLS/SSL 設定（直接 or リバースプロキシ経由）
```

### 環境変数とシークレット管理

```yaml
# docker-compose.prod.yml - シークレット管理
version: "3.9"

services:
  api:
    image: my-api:latest
    environment:
      # 非機密設定は環境変数で直接指定
      NODE_ENV: production
      LOG_LEVEL: info
      PORT: "3000"
    env_file:
      - .env.production  # 環境固有の設定
    secrets:
      - db_password
      - api_key
      - jwt_secret

secrets:
  db_password:
    file: ./secrets/db_password.txt    # ファイルベース
  api_key:
    external: true                      # Docker Swarm シークレット
  jwt_secret:
    environment: JWT_SECRET             # 環境変数から（Compose v2.17+）
```

```javascript
// Node.js - Docker シークレットの読み取り
const fs = require("fs");
const path = require("path");

function readSecret(secretName) {
  const secretPath = path.join("/run/secrets", secretName);
  try {
    return fs.readFileSync(secretPath, "utf8").trim();
  } catch (err) {
    // シークレットファイルがない場合は環境変数にフォールバック
    return process.env[secretName.toUpperCase()];
  }
}

const dbPassword = readSecret("db_password");
const jwtSecret = readSecret("jwt_secret");
```

---

## 8. ネットワークセキュリティ

### 本番ネットワーク設計

```yaml
# docker-compose.prod.yml - ネットワーク分離
version: "3.9"

services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    networks:
      - frontend
    # nginx のみが外部に公開される

  api:
    image: my-api:latest
    networks:
      - frontend    # nginx からのリクエストを受信
      - backend     # DB/Redis への接続
    # ポートは公開しない（nginx経由のみ）

  postgres:
    image: postgres:16-alpine
    networks:
      - backend     # API からのみアクセス可能
    # ポートは公開しない

  redis:
    image: redis:7-alpine
    networks:
      - backend
    # ポートは公開しない

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # 外部アクセス不可（インターネット接続なし）
```

```
ネットワーク分離の図:

Internet
    │
    │ :443
    ▼
┌─────────────────────────────────────────┐
│  frontend network                       │
│  ┌──────────┐       ┌──────────┐       │
│  │  nginx   │──────►│   api    │       │
│  │ :443     │       │          │       │
│  └──────────┘       └────┬─────┘       │
│                          │              │
├──────────────────────────┼──────────────┤
│  backend network         │ (internal)   │
│                     ┌────▼─────┐        │
│  ┌──────────┐      │   api    │        │
│  │ postgres │◄─────│          │        │
│  │          │      └────┬─────┘        │
│  └──────────┘           │               │
│  ┌──────────┐      ┌───▼──────┐        │
│  │  redis   │◄─────│   api    │        │
│  └──────────┘      └──────────┘        │
│                                         │
│  ※ internal: true により                │
│    外部インターネットへの通信を遮断       │
└─────────────────────────────────────────┘
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

### アンチパターン4: 環境変数にシークレットを直接記述

```yaml
# NG: docker-compose.yml にパスワードを直書き
services:
  api:
    environment:
      DB_PASSWORD: "MySecretPassword123!"
      API_KEY: "sk-1234567890abcdef"

# OK: Docker Secrets または .env ファイル（.gitignore対象）を使用
services:
  api:
    env_file:
      - .env.production  # .gitignore に含める
    secrets:
      - db_password
```

**なぜ問題か**: docker-compose.yml をGitリポジトリにコミットすると、シークレットが履歴に残り、漏洩の原因になる。

### アンチパターン5: shell形式のCMD

```dockerfile
# NG: shell形式（/bin/sh -c でラップされる）
CMD node server.js
# PID 1 = /bin/sh, PID 2 = node
# → SIGTERMが/bin/shに届き、nodeに伝わらない

# OK: exec形式
CMD ["node", "server.js"]
# PID 1 = node
# → SIGTERMがnodeに直接届く
```

**なぜ問題か**: shell形式では、SIGTERMシグナルがシェルプロセスに届き、アプリケーションプロセスにはデフォルトで転送されない。Graceful Shutdownが機能しなくなる。

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

### Q4: Docker Composeの本番利用は推奨されるか？

Docker Compose は単一ホストでの本番運用に十分対応できる。ただし以下の制約を理解した上で使用する:

- **単一障害点**: ホスト障害で全サービス停止
- **スケーリング**: 同一ホスト内でのスケーリングのみ
- **ゼロダウンタイムデプロイ**: `--scale` と healthcheck で疑似的に実現可能だが完全ではない

中〜大規模や高可用性が必須の場合は、Docker Swarm や Kubernetes への移行を検討する。

### Q5: コンテナのセキュリティスキャンはどの頻度で行うべきか？

- **CIパイプライン**: 全ビルドでスキャン（必須）
- **定期スキャン**: 週1回以上、デプロイ済みイメージをスキャン
- **ベースイメージ更新時**: 即座にリビルド+スキャン

```bash
# Trivyでのイメージスキャン
trivy image --severity CRITICAL,HIGH my-app:latest

# 既知の脆弱性のみを検知（修正可能なもの）
trivy image --ignore-unfixed --severity CRITICAL my-app:latest

# SBOM（ソフトウェア部品表）の生成
trivy image --format spdx-json --output sbom.json my-app:latest
```

### Q6: distroless イメージとは何か？使うべきか？

Googleが提供する最小構成のコンテナイメージ。シェル、パッケージマネージャー、その他のOSユーティリティを含まない。攻撃対象面が極小で、CVE数も最少。Go や Java のような単一バイナリ/JARのアプリケーションに最適。

```dockerfile
# distroless を使った Go アプリケーション
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=builder /app/server /server
USER nonroot:nonroot
ENTRYPOINT ["/server"]
```

デバッグ時は `:debug` タグを使用するとシェルが含まれる。

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
| ネットワーク分離 | frontend/backend分離。internal: true で外部遮断 |
| シークレット管理 | Docker Secrets / env_file。直書き厳禁 |
| イメージセキュリティ | Trivyスキャン。distroless / alpine で最小構成 |

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
6. Google "Distroless" Container Images -- https://github.com/GoogleContainerTools/distroless
7. Docker公式ドキュメント "Rootless mode" -- https://docs.docker.com/engine/security/rootless/
8. Adrian Mouat (2023) *Docker: Up & Running*, 3rd Edition, O'Reilly

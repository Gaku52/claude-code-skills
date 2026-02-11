# リバースプロキシ

> Nginx / Traefikを使ったリバースプロキシ構成で、SSL終端・ロードバランシング・自動サービスディスカバリをDocker環境に実装する。

---

## この章で学ぶこと

1. **NginxとTraefikの特徴と使い分け**を理解する
2. **SSL/TLS終端とLet's Encrypt自動証明書**の設定を習得する
3. **Docker連携による動的ルーティングとロードバランシング**を構築できるようになる

---

## 1. リバースプロキシとは

リバースプロキシは、クライアントとバックエンドサーバーの間に位置し、リクエストを適切なサービスに振り分けるゲートウェイである。

### リバースプロキシの役割

```
クライアント (ブラウザ)
    │
    │ HTTPS (443)
    ▼
┌──────────────────────────────────────────┐
│         リバースプロキシ                   │
│  ┌────────────────────────────────────┐  │
│  │  - SSL/TLS終端                     │  │
│  │  - ルーティング (Host/Pathベース)   │  │
│  │  - ロードバランシング               │  │
│  │  - レート制限                       │  │
│  │  - レスポンスキャッシュ             │  │
│  │  - gzip圧縮                        │  │
│  │  - セキュリティヘッダー付与         │  │
│  └───────────┬────────────────────────┘  │
│              │ HTTP (80) 内部通信         │
│    ┌─────────┼─────────────┐             │
│    ▼         ▼             ▼             │
│ ┌──────┐ ┌──────┐    ┌──────┐          │
│ │ App1 │ │ App2 │    │ App3 │          │
│ │:3000 │ │:8080 │    │:5000 │          │
│ └──────┘ └──────┘    └──────┘          │
└──────────────────────────────────────────┘
```

### Nginx vs Traefik 比較表

| 特性 | Nginx | Traefik |
|------|-------|---------|
| 設定方式 | 静的設定ファイル | 動的（Docker API連携） |
| Docker連携 | 手動設定 or テンプレート | ラベルで自動検出 |
| SSL証明書 | 手動 or certbot | 内蔵ACME（自動取得・更新） |
| ダッシュボード | 有料（Nginx Plus） | 無料で組み込み |
| パフォーマンス | 非常に高い | 高い |
| 学習コスト | 低い（広く知られている） | 中程度 |
| ユースケース | 安定・高性能が必要 | Docker/K8s動的環境 |
| ミドルウェア | モジュールで拡張 | 組み込みミドルウェア |

---

## 2. Nginx リバースプロキシ

### コード例1: 基本的なNginxリバースプロキシ

```yaml
# docker-compose.yml
version: "3.9"

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    networks:
      - proxy-net
    depends_on:
      - app
      - api

  app:
    image: my-frontend:latest
    networks:
      - proxy-net
    # ポートは公開しない（Nginx経由のみ）

  api:
    image: my-api:latest
    networks:
      - proxy-net

networks:
  proxy-net:
    driver: bridge
```

```nginx
# nginx/conf.d/default.conf
upstream frontend {
    server app:3000;
}

upstream backend {
    server api:8080;
}

server {
    listen 80;
    server_name example.com;

    # HTTPSへリダイレクト
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name example.com;

    ssl_certificate     /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    # セキュリティヘッダー
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # フロントエンド
    location / {
        proxy_pass http://frontend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # API
    location /api/ {
        proxy_pass http://backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket対応
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### コード例2: Nginxロードバランシング

```nginx
# nginx/conf.d/load-balance.conf

# ラウンドロビン（デフォルト）
upstream app_round_robin {
    server app-1:8080;
    server app-2:8080;
    server app-3:8080;
}

# 重み付きラウンドロビン
upstream app_weighted {
    server app-1:8080 weight=5;  # 50%のリクエスト
    server app-2:8080 weight=3;  # 30%のリクエスト
    server app-3:8080 weight=2;  # 20%のリクエスト
}

# 最小接続数
upstream app_least_conn {
    least_conn;
    server app-1:8080;
    server app-2:8080;
    server app-3:8080;
}

# IPハッシュ（セッション維持）
upstream app_ip_hash {
    ip_hash;
    server app-1:8080;
    server app-2:8080;
    server app-3:8080;
}

# ヘルスチェック付き
upstream app_health {
    server app-1:8080 max_fails=3 fail_timeout=30s;
    server app-2:8080 max_fails=3 fail_timeout=30s;
    server app-backup:8080 backup;  # 全サーバー障害時のフォールバック
}

server {
    listen 80;

    location / {
        proxy_pass http://app_least_conn;
        proxy_next_upstream error timeout http_502 http_503;
        proxy_next_upstream_tries 3;
    }
}
```

### ロードバランシングアルゴリズム比較表

| アルゴリズム | 方式 | 特徴 | 適用場面 |
|------------|------|------|---------|
| round_robin | 順番に分配 | シンプル、均等分配 | ステートレスAPI |
| weighted | 重み付き順番 | サーバー性能差を考慮 | 異なるスペックのサーバー |
| least_conn | 最少接続 | 負荷を動的に平準化 | 処理時間にばらつきがある場合 |
| ip_hash | IPベース | 同一クライアントは同一サーバー | セッション維持が必要 |

---

## 3. Traefik リバースプロキシ

TraefikはDockerとネイティブに統合され、コンテナの起動・停止を自動検知してルーティングを動的に更新する。

### コード例3: Traefik基本構成

```yaml
# docker-compose.yml
version: "3.9"

services:
  traefik:
    image: traefik:v3.1
    command:
      - "--api.dashboard=true"
      - "--providers.docker=true"
      - "--providers.docker.exposedbydefault=false"
      - "--entryPoints.web.address=:80"
      - "--entryPoints.websecure.address=:443"
      # Let's Encrypt自動証明書
      - "--certificatesresolvers.letsencrypt.acme.httpchallenge=true"
      - "--certificatesresolvers.letsencrypt.acme.httpchallenge.entrypoint=web"
      - "--certificatesresolvers.letsencrypt.acme.email=admin@example.com"
      - "--certificatesresolvers.letsencrypt.acme.storage=/letsencrypt/acme.json"
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - letsencrypt:/letsencrypt
    networks:
      - proxy
    labels:
      # Traefikダッシュボードの設定
      - "traefik.enable=true"
      - "traefik.http.routers.dashboard.rule=Host(`traefik.example.com`)"
      - "traefik.http.routers.dashboard.service=api@internal"
      - "traefik.http.routers.dashboard.middlewares=auth"
      - "traefik.http.middlewares.auth.basicauth.users=admin:$$apr1$$xyz..."

  # --- アプリケーションサービス ---
  frontend:
    image: my-frontend:latest
    networks:
      - proxy
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.frontend.rule=Host(`app.example.com`)"
      - "traefik.http.routers.frontend.entrypoints=websecure"
      - "traefik.http.routers.frontend.tls.certresolver=letsencrypt"
      - "traefik.http.services.frontend.loadbalancer.server.port=3000"

  api:
    image: my-api:latest
    networks:
      - proxy
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.api.rule=Host(`app.example.com`) && PathPrefix(`/api`)"
      - "traefik.http.routers.api.entrypoints=websecure"
      - "traefik.http.routers.api.tls.certresolver=letsencrypt"
      - "traefik.http.services.api.loadbalancer.server.port=8080"
      # ミドルウェア: APIパスのストリップ
      - "traefik.http.routers.api.middlewares=api-strip"
      - "traefik.http.middlewares.api-strip.stripprefix.prefixes=/api"

networks:
  proxy:
    driver: bridge

volumes:
  letsencrypt:
```

### Traefikのサービスディスカバリフロー

```
┌─────────────────────────────────────────────────────┐
│                   Docker Host                       │
│                                                     │
│  ┌──────────┐                                      │
│  │ Traefik  │◄───── Docker Socket (/var/run/...)   │
│  │          │       コンテナのラベルを監視           │
│  └────┬─────┘                                      │
│       │                                             │
│       │  新コンテナ起動を検知                        │
│       │  ラベルからルーティングルールを生成           │
│       │                                             │
│       │  例: Host(`app.example.com`)                │
│       │      → frontend:3000 へルーティング          │
│       │                                             │
│  ┌────▼────────────────────────────────────────┐   │
│  │              ルーティングテーブル              │   │
│  │                                              │   │
│  │  app.example.com      → frontend:3000       │   │
│  │  app.example.com/api  → api:8080            │   │
│  │  admin.example.com    → admin:4000          │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### コード例4: Traefikミドルウェアチェーン

```yaml
# docker-compose.yml (サービス部分)
services:
  api:
    image: my-api:latest
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.api.rule=Host(`api.example.com`)"
      - "traefik.http.routers.api.entrypoints=websecure"
      - "traefik.http.routers.api.tls.certresolver=letsencrypt"

      # ミドルウェアチェーン
      - "traefik.http.routers.api.middlewares=rate-limit,compress,headers"

      # レート制限
      - "traefik.http.middlewares.rate-limit.ratelimit.average=100"
      - "traefik.http.middlewares.rate-limit.ratelimit.burst=50"
      - "traefik.http.middlewares.rate-limit.ratelimit.period=1m"

      # gzip圧縮
      - "traefik.http.middlewares.compress.compress=true"

      # セキュリティヘッダー
      - "traefik.http.middlewares.headers.headers.stsSeconds=31536000"
      - "traefik.http.middlewares.headers.headers.stsIncludeSubdomains=true"
      - "traefik.http.middlewares.headers.headers.frameDeny=true"
      - "traefik.http.middlewares.headers.headers.contentTypeNosniff=true"
      - "traefik.http.middlewares.headers.headers.browserXssFilter=true"

      # CORS設定
      - "traefik.http.middlewares.headers.headers.accessControlAllowOriginList=https://app.example.com"
      - "traefik.http.middlewares.headers.headers.accessControlAllowMethods=GET,POST,PUT,DELETE"
      - "traefik.http.middlewares.headers.headers.accessControlAllowHeaders=Content-Type,Authorization"
```

---

## 4. SSL/TLS設定

### コード例5: Let's Encrypt + Nginx（certbot）

```yaml
# docker-compose.yml
version: "3.9"

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - certbot-webroot:/var/www/certbot:ro
      - certbot-certs:/etc/letsencrypt:ro
    networks:
      - proxy-net

  certbot:
    image: certbot/certbot
    volumes:
      - certbot-webroot:/var/www/certbot
      - certbot-certs:/etc/letsencrypt
    # 初回証明書取得
    command: >
      certonly --webroot
      --webroot-path=/var/www/certbot
      --email admin@example.com
      --agree-tos --no-eff-email
      -d example.com -d www.example.com

volumes:
  certbot-webroot:
  certbot-certs:

networks:
  proxy-net:
```

```nginx
# nginx/conf.d/default.conf
server {
    listen 80;
    server_name example.com www.example.com;

    # ACME チャレンジ用
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name example.com www.example.com;

    ssl_certificate     /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;

    # Mozilla推奨のSSL設定 (Intermediate)
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:10m;
    ssl_session_tickets off;

    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;

    location / {
        proxy_pass http://app:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# 証明書の自動更新（cron or docker-compose）
# 更新スクリプト
#!/bin/bash
docker compose run --rm certbot renew --quiet
docker compose exec nginx nginx -s reload
```

---

## 5. 実践構成: マルチサービス本番環境

### コード例6: 完全な本番構成例

```yaml
# docker-compose.prod.yml
version: "3.9"

services:
  traefik:
    image: traefik:v3.1
    restart: always
    command:
      - "--log.level=WARN"
      - "--accesslog=true"
      - "--accesslog.filepath=/logs/access.log"
      - "--providers.docker=true"
      - "--providers.docker.exposedbydefault=false"
      - "--entryPoints.web.address=:80"
      - "--entryPoints.web.http.redirections.entryPoint.to=websecure"
      - "--entryPoints.websecure.address=:443"
      - "--certificatesresolvers.le.acme.httpchallenge.entrypoint=web"
      - "--certificatesresolvers.le.acme.email=ops@example.com"
      - "--certificatesresolvers.le.acme.storage=/letsencrypt/acme.json"
      - "--metrics.prometheus=true"
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - letsencrypt:/letsencrypt
      - traefik-logs:/logs
    networks:
      - proxy
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: "0.5"

  web:
    image: registry.example.com/web:${VERSION}
    restart: always
    networks:
      - proxy
      - app-internal
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.web.rule=Host(`www.example.com`)"
      - "traefik.http.routers.web.entrypoints=websecure"
      - "traefik.http.routers.web.tls.certresolver=le"
      - "traefik.http.services.web.loadbalancer.server.port=3000"
      - "traefik.http.services.web.loadbalancer.healthCheck.path=/health"
      - "traefik.http.services.web.loadbalancer.healthCheck.interval=10s"
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 512M
          cpus: "1.0"

  api:
    image: registry.example.com/api:${VERSION}
    restart: always
    networks:
      - proxy
      - app-internal
      - db-net
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.api.rule=Host(`api.example.com`)"
      - "traefik.http.routers.api.entrypoints=websecure"
      - "traefik.http.routers.api.tls.certresolver=le"
      - "traefik.http.services.api.loadbalancer.server.port=8080"

  db:
    image: postgres:16-alpine
    restart: always
    volumes:
      - pgdata:/var/lib/postgresql/data
    networks:
      - db-net
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password

networks:
  proxy:
    driver: bridge
  app-internal:
    driver: bridge
    internal: true
  db-net:
    driver: bridge
    internal: true

volumes:
  letsencrypt:
  traefik-logs:
  pgdata:

secrets:
  db_password:
    file: ./secrets/db_password.txt
```

---

## アンチパターン

### アンチパターン1: バックエンドポートの直接公開

```yaml
# NG: バックエンドサービスのポートを外部に公開
services:
  nginx:
    ports:
      - "80:80"
  app:
    ports:
      - "3000:3000"  # 直接アクセス可能 → プロキシをバイパスされる
  db:
    ports:
      - "5432:5432"  # 最悪のパターン

# OK: プロキシのみ公開、バックエンドはネットワーク分離
services:
  nginx:
    ports:
      - "80:80"
      - "443:443"
  app:
    # ポートは公開しない
    networks:
      - internal
  db:
    networks:
      - db-only
```

**なぜ問題か**: リバースプロキシのセキュリティ機能（レート制限、認証、ヘッダー付与）がバイパスされ、バックエンドに直接攻撃を受ける。

### アンチパターン2: Dockerソケットの無制限マウント

```yaml
# NG: Dockerソケットに書き込み権限を付与
volumes:
  - /var/run/docker.sock:/var/run/docker.sock

# OK: 読み取り専用でマウント
volumes:
  - /var/run/docker.sock:/var/run/docker.sock:ro

# より安全: Docker Socket Proxyを使用
services:
  socket-proxy:
    image: tecnativa/docker-socket-proxy
    environment:
      CONTAINERS: 1
      SERVICES: 0
      TASKS: 0
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
    networks:
      - socket-proxy

  traefik:
    image: traefik:v3.1
    command:
      - "--providers.docker.endpoint=tcp://socket-proxy:2375"
    networks:
      - socket-proxy
      - proxy
    # Dockerソケットを直接マウントしない
```

**なぜ問題か**: Dockerソケットへの書き込みアクセスはホストのroot権限と等価。Traefikが侵害された場合、ホスト全体が危険にさらされる。

---

## FAQ

### Q1: NginxとTraefikのどちらを選ぶべき？

**Nginx推奨**: 静的な構成で十分な場合、高トラフィック環境、既存のNginx運用ノウハウがある場合。
**Traefik推奨**: コンテナの動的な追加・削除が頻繁な場合、Let's Encrypt自動化が必要な場合、Kubernetes/Swarm環境。

小規模ならNginxのシンプルさが利点。マイクロサービスの動的環境ではTraefikのサービスディスカバリが圧倒的に便利。

### Q2: Let's Encryptの証明書更新が失敗した場合はどうする？

1. 80番ポートが外部から到達可能か確認（HTTP-01チャレンジ）
2. DNS設定が正しいか確認
3. レート制限に達していないか確認（1週間に5回まで）
4. `--staging` フラグでテスト環境の証明書を使ってデバッグ

```bash
# テスト環境での証明書取得テスト
certbot certonly --staging --webroot -w /var/www/certbot -d example.com
```

### Q3: リバースプロキシ配下でクライアントの実IPアドレスを取得するには？

```nginx
# Nginx側でヘッダーを設定
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
```

アプリケーション側で `X-Real-IP` または `X-Forwarded-For` ヘッダーを読む。ただし、信頼できるプロキシからのヘッダーのみを信頼するよう設定すること（IPスプーフィング防止）。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| Nginx | 静的設定、高性能、広い知見。`proxy_pass` で転送 |
| Traefik | Docker連携自動検出、ラベルで設定、ACME内蔵 |
| SSL/TLS | Let's Encryptで無料証明書。TLS 1.2以上を強制 |
| ロードバランシング | round_robin / least_conn / ip_hash を用途別に選択 |
| セキュリティ | バックエンドポート非公開、Dockerソケット読取専用 |
| ヘッダー | X-Real-IP / X-Forwarded-For / HSTS を必ず設定 |

---

## 次に読むべきガイド

- [本番ベストプラクティス](../04-production/00-production-best-practices.md) -- 非rootユーザー、ヘルスチェック、リソース制限
- [モニタリング](../04-production/01-monitoring.md) -- Traefikメトリクスの収集とダッシュボード
- [コンテナセキュリティ](../06-security/00-container-security.md) -- プロキシ層を含むセキュリティ強化

---

## 参考文献

1. Nginx公式ドキュメント "Reverse Proxy" -- https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy/
2. Traefik公式ドキュメント "Docker Provider" -- https://doc.traefik.io/traefik/providers/docker/
3. Mozilla "SSL Configuration Generator" -- https://ssl-config.mozilla.org/
4. Let's Encrypt Documentation -- https://letsencrypt.org/docs/
5. Traefik Labs (2024) "Traefik Proxy Documentation" -- https://doc.traefik.io/traefik/

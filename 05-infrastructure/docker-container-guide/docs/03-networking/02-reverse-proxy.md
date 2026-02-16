# リバースプロキシ

> Nginx / Traefikを使ったリバースプロキシ構成で、SSL終端・ロードバランシング・自動サービスディスカバリをDocker環境に実装する。

---

## この章で学ぶこと

1. **NginxとTraefikの特徴と使い分け**を理解する
2. **SSL/TLS終端とLet's Encrypt自動証明書**の設定を習得する
3. **Docker連携による動的ルーティングとロードバランシング**を構築できるようになる
4. **WebSocket・gRPC・TCP/UDPプロキシ**の設定パターンを習得する
5. **Caddy**を含む代替リバースプロキシの選択肢を理解する
6. **mTLS・CORS・WAF**を含む高度なセキュリティ構成を実装できるようになる

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

### リバースプロキシの主要機能一覧

| 機能 | 説明 | メリット |
|------|------|---------|
| SSL/TLS終端 | クライアントとの暗号化通信を処理 | バックエンドはHTTPのみで簡潔 |
| ルーティング | Host/Pathベースでリクエストを振り分け | 単一IPで複数サービス運用 |
| ロードバランシング | 複数インスタンスへ負荷分散 | スケーラビリティ・可用性向上 |
| レート制限 | リクエスト数を制御 | DDoS防御、API保護 |
| キャッシュ | レスポンスをキャッシュ | バックエンド負荷軽減、応答速度向上 |
| 圧縮 | gzip/Brotliでレスポンス圧縮 | 帯域幅削減、ページ速度向上 |
| セキュリティヘッダー | HSTS, CSP, X-Frame-Options等 | XSS, クリックジャッキング防止 |
| アクセスログ | リクエストの記録・分析 | 監査・トラブルシューティング |
| 認証 | Basic Auth, OAuth2プロキシ | バックエンドの認証機能をオフロード |

### Nginx vs Traefik vs Caddy 比較表

| 特性 | Nginx | Traefik | Caddy |
|------|-------|---------|-------|
| 設定方式 | 静的設定ファイル | 動的（Docker API連携） | Caddyfile / JSON API |
| Docker連携 | 手動設定 or テンプレート | ラベルで自動検出 | Docker modules（プラグイン） |
| SSL証明書 | 手動 or certbot | 内蔵ACME（自動取得・更新） | 内蔵ACME（自動、デフォルトHTTPS） |
| ダッシュボード | 有料（Nginx Plus） | 無料で組み込み | なし（API経由） |
| パフォーマンス | 非常に高い | 高い | 高い |
| 学習コスト | 低い（広く知られている） | 中程度 | 低い（シンプルな構文） |
| ユースケース | 安定・高性能が必要 | Docker/K8s動的環境 | シンプルなHTTPS環境 |
| ミドルウェア | モジュールで拡張 | 組み込みミドルウェア | ハンドラーチェーン |
| TCP/UDPプロキシ | stream モジュール | TCPルーター | Layer 4対応 |
| HTTP/3対応 | 実験的サポート | v3.xで対応 | デフォルト対応 |
| 設定リロード | `nginx -s reload` | 自動（ホットリロード） | 自動（API経由） |
| メモリ使用量 | 非常に少ない | 中程度 | 少ない |

### 選択フローチャート

```
リバースプロキシの選択
    │
    ├─ Docker/K8sで動的にサービスが増減する？
    │   ├─ Yes → Traefik推奨
    │   └─ No ─┐
    │          │
    ├─ とにかくシンプルにHTTPS化したい？
    │   ├─ Yes → Caddy推奨
    │   └─ No ─┐
    │          │
    ├─ 高性能・細かいチューニングが必要？
    │   ├─ Yes → Nginx推奨
    │   └─ No ─┐
    │          │
    └─ Nginx（実績と情報量で安心）
```

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
| hash | 任意キーのハッシュ | URIベースなど柔軟 | キャッシュ最適化 |
| random two | ランダム2台から最少接続 | 分散環境向き | 大規模クラスタ |

### コード例: Nginx レスポンスキャッシュ

```nginx
# nginx/conf.d/cache.conf

# キャッシュゾーン定義
proxy_cache_path /var/cache/nginx/api
    levels=1:2
    keys_zone=api_cache:10m
    max_size=1g
    inactive=60m
    use_temp_path=off;

proxy_cache_path /var/cache/nginx/static
    levels=1:2
    keys_zone=static_cache:10m
    max_size=5g
    inactive=7d
    use_temp_path=off;

server {
    listen 443 ssl http2;
    server_name example.com;

    # 静的ファイルのキャッシュ
    location /static/ {
        proxy_pass http://frontend;
        proxy_cache static_cache;
        proxy_cache_valid 200 7d;
        proxy_cache_valid 404 1m;
        proxy_cache_use_stale error timeout updating http_502 http_503;

        # キャッシュ状態をレスポンスヘッダーに追加（デバッグ用）
        add_header X-Cache-Status $upstream_cache_status;
    }

    # APIレスポンスのキャッシュ（GET のみ）
    location /api/ {
        proxy_pass http://backend;
        proxy_cache api_cache;
        proxy_cache_methods GET HEAD;
        proxy_cache_valid 200 5m;
        proxy_cache_valid 404 1m;
        proxy_cache_bypass $http_cache_control;
        proxy_no_cache $http_pragma;

        # POSTリクエストはキャッシュしない
        proxy_cache_bypass $request_method;

        add_header X-Cache-Status $upstream_cache_status;
    }

    # キャッシュパージ用エンドポイント（Nginx Plus相当の機能をproxy_cache_purgeで）
    location /purge/ {
        allow 10.0.0.0/8;
        deny all;
        proxy_cache_purge api_cache $scheme$proxy_host$request_uri;
    }
}
```

```yaml
# docker-compose.yml（キャッシュ用ボリューム付き）
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - nginx-cache:/var/cache/nginx
    tmpfs:
      - /var/cache/nginx/tmp:size=256m
    networks:
      - proxy-net

volumes:
  nginx-cache:
```

### コード例: Nginxレート制限

```nginx
# nginx/conf.d/rate-limit.conf

# レート制限ゾーン定義
# $binary_remote_addr: クライアントIPごとに制限
# zone: 共有メモリゾーン名とサイズ (1MB ≈ 16,000 IP)
# rate: 1秒あたりのリクエスト数
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=login_limit:10m rate=1r/s;
limit_req_zone $server_name zone=global_limit:10m rate=1000r/s;

# 接続数制限
limit_conn_zone $binary_remote_addr zone=conn_limit:10m;

server {
    listen 443 ssl http2;
    server_name api.example.com;

    # グローバル接続数制限（1 IPあたり100接続まで）
    limit_conn conn_limit 100;

    # APIエンドポイントのレート制限
    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;
        # burst: バーストで許容するリクエスト数
        # nodelay: バースト内のリクエストを即座に処理（待機させない）

        limit_req_status 429;
        limit_req_log_level warn;

        proxy_pass http://backend;
    }

    # ログインエンドポイント（厳しい制限）
    location /api/auth/login {
        limit_req zone=login_limit burst=5;

        # レート制限超過時のカスタムエラーページ
        error_page 429 = @rate_limited;

        proxy_pass http://backend;
    }

    # レート制限時のJSON応答
    location @rate_limited {
        default_type application/json;
        return 429 '{"error": "Too Many Requests", "retry_after": 60}';
    }
}
```

### コード例: Nginx WebSocket プロキシ

```nginx
# nginx/conf.d/websocket.conf

# WebSocketの接続マップ
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

upstream websocket_backend {
    # sticky session（WebSocketは同一サーバーに接続する必要がある）
    ip_hash;
    server ws-app-1:8080;
    server ws-app-2:8080;
}

server {
    listen 443 ssl http2;
    server_name ws.example.com;

    ssl_certificate     /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    # WebSocketエンドポイント
    location /ws {
        proxy_pass http://websocket_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket接続のタイムアウト設定
        proxy_read_timeout 86400s;   # 24時間
        proxy_send_timeout 86400s;
        proxy_connect_timeout 60s;

        # バッファリングを無効化（リアルタイム通信のため）
        proxy_buffering off;
    }

    # Socket.IOの場合（ポーリングフォールバック含む）
    location /socket.io/ {
        proxy_pass http://websocket_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        proxy_read_timeout 86400s;
        proxy_buffering off;
    }
}
```

### コード例: Nginx gRPC プロキシ

```nginx
# nginx/conf.d/grpc.conf

upstream grpc_backend {
    server grpc-app:50051;
}

server {
    listen 443 ssl http2;
    server_name grpc.example.com;

    ssl_certificate     /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;

    # gRPCプロキシ
    location / {
        grpc_pass grpc://grpc_backend;

        # gRPCヘルスチェック
        grpc_set_header Host $host;

        # タイムアウト設定（長時間ストリーミング対応）
        grpc_read_timeout 300s;
        grpc_send_timeout 300s;

        # エラーハンドリング
        error_page 502 = /error502grpc;
    }

    # gRPCエラーレスポンス
    location = /error502grpc {
        internal;
        default_type application/grpc;
        add_header grpc-status 14;
        add_header grpc-message "Upstream unavailable";
        return 204;
    }
}
```

```yaml
# docker-compose.yml（gRPCサービス構成）
services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    networks:
      - proxy-net

  grpc-app:
    image: my-grpc-service:latest
    networks:
      - proxy-net
    # ポートは公開しない（Nginx経由のみ）

networks:
  proxy-net:
    driver: bridge
```

### コード例: Nginx Stream モジュール（TCP/UDPプロキシ）

```nginx
# nginx/nginx.conf（メインコンテキスト）

# TCP/UDPロードバランシング
stream {
    # MySQL (TCP)
    upstream mysql_backend {
        server db-primary:3306;
        server db-replica-1:3306 backup;
    }

    server {
        listen 3306;
        proxy_pass mysql_backend;
        proxy_timeout 300s;
        proxy_connect_timeout 10s;
    }

    # Redis (TCP)
    upstream redis_backend {
        hash $remote_addr consistent;
        server redis-1:6379;
        server redis-2:6379;
        server redis-3:6379;
    }

    server {
        listen 6379;
        proxy_pass redis_backend;
        proxy_timeout 300s;
    }

    # DNS (UDP)
    upstream dns_backend {
        server dns-1:53;
        server dns-2:53;
    }

    server {
        listen 53 udp;
        proxy_pass dns_backend;
        proxy_timeout 5s;
        proxy_responses 1;
    }

    # SSL Passthrough（SSL終端をバックエンドに委任）
    upstream ssl_passthrough {
        server app:443;
    }

    server {
        listen 8443;
        proxy_pass ssl_passthrough;
        proxy_protocol on;
    }
}
```

```yaml
# docker-compose.yml（Nginx Stream構成）
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
      - "3306:3306"
      - "6379:6379"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
    networks:
      - proxy-net
      - db-net

  db-primary:
    image: mysql:8
    networks:
      - db-net

networks:
  proxy-net:
  db-net:
    internal: true
```

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

### コード例: Traefik ファイルプロバイダー

Docker ラベル以外に、YAML/TOML ファイルでルーティングルールを定義できる。外部サービスやDockerコンテナ以外のバックエンドに対して有用。

```yaml
# docker-compose.yml
services:
  traefik:
    image: traefik:v3.1
    command:
      - "--providers.docker=true"
      - "--providers.docker.exposedbydefault=false"
      - "--providers.file.directory=/etc/traefik/dynamic"
      - "--providers.file.watch=true"
      - "--entryPoints.web.address=:80"
      - "--entryPoints.websecure.address=:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./traefik/dynamic:/etc/traefik/dynamic:ro
    ports:
      - "80:80"
      - "443:443"
```

```yaml
# traefik/dynamic/external-services.yml
http:
  routers:
    # 外部サービスへのルーティング
    legacy-api:
      rule: "Host(`api.example.com`) && PathPrefix(`/v1`)"
      service: legacy-api-service
      entryPoints:
        - websecure
      tls:
        certResolver: letsencrypt
      middlewares:
        - api-headers

    # 静的ファイルサーバー
    cdn:
      rule: "Host(`cdn.example.com`)"
      service: cdn-service
      entryPoints:
        - websecure
      tls:
        certResolver: letsencrypt

  services:
    legacy-api-service:
      loadBalancer:
        servers:
          - url: "http://192.168.1.100:8080"
          - url: "http://192.168.1.101:8080"
        healthCheck:
          path: /health
          interval: "10s"
          timeout: "3s"

    cdn-service:
      loadBalancer:
        servers:
          - url: "http://192.168.1.200:80"

  middlewares:
    api-headers:
      headers:
        customRequestHeaders:
          X-Forwarded-Source: "traefik"
        customResponseHeaders:
          X-Custom-Header: "my-value"
```

### コード例: Traefik TCP/UDPルーティング

```yaml
# docker-compose.yml
services:
  traefik:
    image: traefik:v3.1
    command:
      - "--providers.docker=true"
      - "--providers.docker.exposedbydefault=false"
      - "--entryPoints.web.address=:80"
      - "--entryPoints.websecure.address=:443"
      - "--entryPoints.mysql.address=:3306"
      - "--entryPoints.postgres.address=:5432"
      - "--entryPoints.redis.address=:6379"
    ports:
      - "80:80"
      - "443:443"
      - "3306:3306"
      - "5432:5432"
      - "6379:6379"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro

  # MySQL TCP ルーティング
  mysql:
    image: mysql:8
    labels:
      - "traefik.enable=true"
      - "traefik.tcp.routers.mysql.rule=HostSNI(`*`)"
      - "traefik.tcp.routers.mysql.entrypoints=mysql"
      - "traefik.tcp.services.mysql.loadbalancer.server.port=3306"

  # PostgreSQL TCP ルーティング（TLS付き）
  postgres:
    image: postgres:16-alpine
    labels:
      - "traefik.enable=true"
      - "traefik.tcp.routers.postgres.rule=HostSNI(`db.example.com`)"
      - "traefik.tcp.routers.postgres.entrypoints=postgres"
      - "traefik.tcp.routers.postgres.tls=true"
      - "traefik.tcp.routers.postgres.tls.certresolver=letsencrypt"
      - "traefik.tcp.services.postgres.loadbalancer.server.port=5432"

  # Redis TCP ルーティング
  redis:
    image: redis:7-alpine
    labels:
      - "traefik.enable=true"
      - "traefik.tcp.routers.redis.rule=HostSNI(`*`)"
      - "traefik.tcp.routers.redis.entrypoints=redis"
      - "traefik.tcp.services.redis.loadbalancer.server.port=6379"
```

### コード例: Traefik WebSocket & gRPC

```yaml
# docker-compose.yml
services:
  traefik:
    image: traefik:v3.1
    command:
      - "--providers.docker=true"
      - "--providers.docker.exposedbydefault=false"
      - "--entryPoints.web.address=:80"
      - "--entryPoints.websecure.address=:443"
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro

  # WebSocket サービス
  ws-app:
    image: my-websocket-app:latest
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.ws.rule=Host(`ws.example.com`)"
      - "traefik.http.routers.ws.entrypoints=websecure"
      - "traefik.http.routers.ws.tls.certresolver=letsencrypt"
      - "traefik.http.services.ws.loadbalancer.server.port=8080"
      # WebSocketはTraefikがデフォルトで対応（特別な設定不要）
      # sticky session（WebSocket用）
      - "traefik.http.services.ws.loadbalancer.sticky.cookie=true"
      - "traefik.http.services.ws.loadbalancer.sticky.cookie.name=ws_session"

  # gRPC サービス
  grpc-app:
    image: my-grpc-service:latest
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.grpc.rule=Host(`grpc.example.com`)"
      - "traefik.http.routers.grpc.entrypoints=websecure"
      - "traefik.http.routers.grpc.tls.certresolver=letsencrypt"
      - "traefik.http.services.grpc.loadbalancer.server.port=50051"
      # gRPCにはh2cスキームを使用
      - "traefik.http.services.grpc.loadbalancer.server.scheme=h2c"
```

### Traefik ミドルウェア一覧

| ミドルウェア | 用途 | ラベル例 |
|-------------|------|---------|
| AddPrefix | パスにプレフィックスを追加 | `addprefix.prefix=/api` |
| StripPrefix | パスからプレフィックスを除去 | `stripprefix.prefixes=/api` |
| RateLimit | レート制限 | `ratelimit.average=100` |
| BasicAuth | Basic認証 | `basicauth.users=user:hash` |
| DigestAuth | Digest認証 | `digestauth.users=user:realm:hash` |
| ForwardAuth | 外部認証サーバーに委任 | `forwardauth.address=http://auth:9000` |
| Headers | カスタムヘッダー追加 | `headers.frameDeny=true` |
| Compress | gzip圧縮 | `compress=true` |
| IPWhiteList | IP制限 | `ipallowlist.sourcerange=10.0.0.0/8` |
| Retry | リトライ | `retry.attempts=3` |
| CircuitBreaker | サーキットブレーカー | `circuitbreaker.expression=...` |
| Buffering | リクエスト/レスポンスバッファリング | `buffering.maxRequestBodyBytes=2000000` |
| Chain | 複数ミドルウェアの組み合わせ | `chain.middlewares=auth,ratelimit` |
| RedirectScheme | HTTPSリダイレクト | `redirectscheme.scheme=https` |
| InFlightReq | 同時リクエスト数制限 | `inflightreq.amount=10` |
| PassTLSClientCert | mTLSクライアント証明書転送 | `passtlsclientcert.pem=true` |

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

### SSL/TLS設定のベストプラクティス

```nginx
# nginx/conf.d/ssl-params.conf
# Mozilla SSL Configuration Generator (Intermediate) に基づく推奨設定

# プロトコル: TLS 1.2 / 1.3 のみ許可（TLS 1.0/1.1は脆弱性あり）
ssl_protocols TLSv1.2 TLSv1.3;

# 暗号スイート（TLS 1.2向け。TLS 1.3はプロトコル側で暗号スイートが決まる）
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;

# サーバー側の暗号スイート優先順位を使用しない（TLS 1.3では不要）
ssl_prefer_server_ciphers off;

# DH パラメータ（鍵交換の安全性向上）
ssl_dhparam /etc/nginx/ssl/dhparam.pem;
# 生成コマンド: openssl dhparam -out dhparam.pem 4096

# セッション設定
ssl_session_timeout 1d;
ssl_session_cache shared:SSL:10m;  # 約40,000セッション
ssl_session_tickets off;

# OCSP Stapling（証明書の有効性をプロキシが確認）
ssl_stapling on;
ssl_stapling_verify on;
resolver 1.1.1.1 8.8.8.8 valid=300s;
resolver_timeout 5s;

# セキュリティヘッダー
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
add_header X-Frame-Options DENY always;
add_header X-Content-Type-Options nosniff always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self';" always;
add_header Permissions-Policy "camera=(), microphone=(), geolocation=(), payment=()" always;
```

### SSL設定のテスト

```bash
#!/bin/bash
# ssl-test.sh - SSL設定の確認スクリプト

DOMAIN="example.com"

echo "=== SSL証明書情報 ==="
echo | openssl s_client -connect ${DOMAIN}:443 -servername ${DOMAIN} 2>/dev/null | openssl x509 -noout -dates -subject -issuer

echo ""
echo "=== TLSプロトコルテスト ==="
for proto in tls1 tls1_1 tls1_2 tls1_3; do
    result=$(echo | openssl s_client -connect ${DOMAIN}:443 -${proto} 2>&1)
    if echo "$result" | grep -q "Cipher is"; then
        echo "${proto}: ENABLED"
    else
        echo "${proto}: DISABLED"
    fi
done

echo ""
echo "=== 暗号スイート ==="
nmap --script ssl-enum-ciphers -p 443 ${DOMAIN}

echo ""
echo "=== セキュリティヘッダー ==="
curl -sI https://${DOMAIN} | grep -iE "strict-transport|x-frame|x-content-type|x-xss|referrer-policy|content-security|permissions-policy"

echo ""
echo "=== SSL Labs テスト ==="
echo "https://www.ssllabs.com/ssltest/analyze.html?d=${DOMAIN}"
```

### mTLS（相互TLS認証）

```nginx
# nginx/conf.d/mtls.conf

server {
    listen 443 ssl http2;
    server_name api-internal.example.com;

    # サーバー証明書
    ssl_certificate     /etc/nginx/ssl/server.pem;
    ssl_certificate_key /etc/nginx/ssl/server-key.pem;

    # クライアント証明書の検証（mTLS）
    ssl_client_certificate /etc/nginx/ssl/ca.pem;
    ssl_verify_client on;
    ssl_verify_depth 2;

    # クライアント証明書情報をバックエンドに転送
    location / {
        proxy_pass http://internal-api:8080;
        proxy_set_header X-Client-CN $ssl_client_s_dn_cn;
        proxy_set_header X-Client-Serial $ssl_client_serial;
        proxy_set_header X-Client-Verify $ssl_client_verify;
    }
}
```

```yaml
# docker-compose.yml（mTLS構成）
services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - ./ssl/server.pem:/etc/nginx/ssl/server.pem:ro
      - ./ssl/server-key.pem:/etc/nginx/ssl/server-key.pem:ro
      - ./ssl/ca.pem:/etc/nginx/ssl/ca.pem:ro
    networks:
      - proxy-net

  internal-api:
    image: my-internal-api:latest
    networks:
      - proxy-net
```

```bash
# クライアント証明書を使った接続テスト
curl -v \
  --cert client.pem \
  --key client-key.pem \
  --cacert ca.pem \
  https://api-internal.example.com/health
```

---

## 5. Caddy リバースプロキシ

Caddyは自動HTTPSをデフォルトで提供するモダンなリバースプロキシである。設定が非常にシンプルで、小〜中規模環境に適している。

### コード例: Caddy 基本構成

```
# Caddyfile
{
    email admin@example.com
    # 自動HTTPS（デフォルト有効、Let's Encryptを使用）
}

# フロントエンド
app.example.com {
    reverse_proxy frontend:3000
}

# APIサーバー（パスベース）
app.example.com {
    handle /api/* {
        reverse_proxy api:8080
    }
    handle {
        reverse_proxy frontend:3000
    }
}

# ロードバランシング
app.example.com {
    reverse_proxy app-1:8080 app-2:8080 app-3:8080 {
        lb_policy least_conn
        health_uri /health
        health_interval 10s
        health_timeout 5s

        # ヘッダー設定
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-For {remote_host}
    }
}

# WebSocket
ws.example.com {
    reverse_proxy ws-app:8080
    # CaddyはWebSocketを自動的にサポート
}

# セキュリティヘッダー
*.example.com {
    header {
        Strict-Transport-Security "max-age=31536000; includeSubDomains; preload"
        X-Frame-Options DENY
        X-Content-Type-Options nosniff
        X-XSS-Protection "1; mode=block"
        Referrer-Policy strict-origin-when-cross-origin
        -Server
    }
}
```

```yaml
# docker-compose.yml（Caddy構成）
version: "3.9"

services:
  caddy:
    image: caddy:2-alpine
    restart: always
    ports:
      - "80:80"
      - "443:443"
      - "443:443/udp"  # HTTP/3 (QUIC)
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy-data:/data        # 証明書の保存先
      - caddy-config:/config    # Caddy設定
    networks:
      - proxy

  frontend:
    image: my-frontend:latest
    networks:
      - proxy

  api:
    image: my-api:latest
    networks:
      - proxy

networks:
  proxy:

volumes:
  caddy-data:
  caddy-config:
```

### Caddy vs Nginx vs Traefik 用途別推奨

| 用途 | 推奨 | 理由 |
|------|------|------|
| 個人ブログ/小規模サイト | Caddy | 最小設定で自動HTTPS |
| マイクロサービス（Docker） | Traefik | 動的サービスディスカバリ |
| 高トラフィックAPI | Nginx | 高性能・低メモリ |
| Kubernetes Ingress | Traefik / Nginx Ingress | エコシステム統合 |
| 社内ツール | Caddy | シンプル・高速セットアップ |
| レガシーシステム統合 | Nginx | 豊富なモジュール・設定事例 |

---

## 6. 実践構成: マルチサービス本番環境

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

### コード例: 本番 Nginx 構成（フルスタック）

```yaml
# docker-compose.prod.yml（Nginx版）
version: "3.9"

services:
  nginx:
    image: nginx:1.27-alpine
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - nginx-logs:/var/log/nginx
      - nginx-cache:/var/cache/nginx
    networks:
      - proxy
    healthcheck:
      test: ["CMD", "nginx", "-t"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 128M
          cpus: "0.5"

  frontend:
    image: registry.example.com/frontend:${VERSION}
    restart: always
    networks:
      - proxy
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 256M

  api:
    image: registry.example.com/api:${VERSION}
    restart: always
    networks:
      - proxy
      - backend
    environment:
      - DATABASE_URL=postgresql://app:${DB_PASSWORD}@db:5432/appdb
      - REDIS_URL=redis://redis:6379
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 512M

  worker:
    image: registry.example.com/worker:${VERSION}
    restart: always
    networks:
      - backend
    environment:
      - DATABASE_URL=postgresql://app:${DB_PASSWORD}@db:5432/appdb
      - REDIS_URL=redis://redis:6379
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 1G

  db:
    image: postgres:16-alpine
    restart: always
    volumes:
      - pgdata:/var/lib/postgresql/data
    networks:
      - backend
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
      POSTGRES_DB: appdb
      POSTGRES_USER: app
    secrets:
      - db_password
    deploy:
      resources:
        limits:
          memory: 1G

  redis:
    image: redis:7-alpine
    restart: always
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
    networks:
      - backend
    deploy:
      resources:
        limits:
          memory: 512M

networks:
  proxy:
    driver: bridge
  backend:
    driver: bridge
    internal: true

volumes:
  nginx-logs:
  nginx-cache:
  pgdata:
  redis-data:

secrets:
  db_password:
    file: ./secrets/db_password.txt
```

```nginx
# nginx/nginx.conf（本番用メイン設定）
user nginx;
worker_processes auto;
worker_rlimit_nofile 65535;

error_log /var/log/nginx/error.log warn;
pid       /var/run/nginx.pid;

events {
    worker_connections 4096;
    multi_accept on;
    use epoll;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    # ログフォーマット（JSON）
    log_format json_combined escape=json
        '{'
        '"time":"$time_iso8601",'
        '"remote_addr":"$remote_addr",'
        '"request":"$request",'
        '"status":$status,'
        '"body_bytes_sent":$body_bytes_sent,'
        '"request_time":$request_time,'
        '"upstream_response_time":"$upstream_response_time",'
        '"http_user_agent":"$http_user_agent",'
        '"http_referer":"$http_referer",'
        '"upstream_addr":"$upstream_addr"'
        '}';

    access_log /var/log/nginx/access.log json_combined;

    # パフォーマンス最適化
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    keepalive_requests 1000;
    types_hash_max_size 2048;
    client_max_body_size 50m;

    # gzip圧縮
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript image/svg+xml;

    # Brotli圧縮（モジュールが利用可能な場合）
    # brotli on;
    # brotli_comp_level 6;
    # brotli_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript image/svg+xml;

    # セキュリティ
    server_tokens off;

    # SSL共通設定
    include /etc/nginx/conf.d/ssl-params.conf;

    # サイト設定
    include /etc/nginx/conf.d/*.conf;
}
```

### コード例: ゼロダウンタイムデプロイ

```bash
#!/bin/bash
# deploy.sh - ゼロダウンタイムデプロイスクリプト

set -euo pipefail

VERSION=$1
SERVICE=$2
COMPOSE_FILE="docker-compose.prod.yml"
REPLICAS=${3:-3}

echo "=== デプロイ開始: ${SERVICE} v${VERSION} ==="

# 新イメージをプル
echo "[1/5] 新イメージをプル..."
docker compose -f ${COMPOSE_FILE} pull ${SERVICE}

# 新しいインスタンスを起動（旧インスタンスは残す）
echo "[2/5] ローリングアップデート開始..."
for i in $(seq 1 ${REPLICAS}); do
    echo "  インスタンス ${i}/${REPLICAS} を更新中..."

    # 1インスタンスずつ更新
    VERSION=${VERSION} docker compose -f ${COMPOSE_FILE} up -d \
        --no-deps --scale ${SERVICE}=${REPLICAS} ${SERVICE}

    # ヘルスチェック通過を待機
    echo "  ヘルスチェック待機中..."
    sleep 10

    # ヘルスチェック確認
    HEALTH_URL="http://localhost/health"
    for attempt in $(seq 1 30); do
        if curl -sf ${HEALTH_URL} > /dev/null 2>&1; then
            echo "  ヘルスチェック OK (試行 ${attempt})"
            break
        fi
        if [ ${attempt} -eq 30 ]; then
            echo "  ヘルスチェック失敗！ロールバック実行..."
            docker compose -f ${COMPOSE_FILE} rollback ${SERVICE} 2>/dev/null || true
            exit 1
        fi
        sleep 2
    done
done

# 古いコンテナをクリーンアップ
echo "[3/5] 古いコンテナを削除..."
docker image prune -f

# Nginxの設定をリロード（Nginx使用時）
echo "[4/5] プロキシ設定リロード..."
docker compose -f ${COMPOSE_FILE} exec -T nginx nginx -s reload 2>/dev/null || true

echo "[5/5] デプロイ完了: ${SERVICE} v${VERSION}"
echo "=== 成功 ==="
```

### コード例: ForwardAuth（OAuth2 Proxy）

外部認証サービスを使ったアクセス制御。Google, GitHub, Azure AD等のOAuth2プロバイダーと連携できる。

```yaml
# docker-compose.yml
services:
  traefik:
    image: traefik:v3.1
    command:
      - "--providers.docker=true"
      - "--providers.docker.exposedbydefault=false"
      - "--entryPoints.websecure.address=:443"
    ports:
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro

  # OAuth2 Proxy
  oauth2-proxy:
    image: quay.io/oauth2-proxy/oauth2-proxy:latest
    environment:
      OAUTH2_PROXY_PROVIDER: google
      OAUTH2_PROXY_CLIENT_ID: ${GOOGLE_CLIENT_ID}
      OAUTH2_PROXY_CLIENT_SECRET: ${GOOGLE_CLIENT_SECRET}
      OAUTH2_PROXY_COOKIE_SECRET: ${COOKIE_SECRET}
      OAUTH2_PROXY_EMAIL_DOMAINS: "example.com"
      OAUTH2_PROXY_UPSTREAM: "static://200"
      OAUTH2_PROXY_HTTP_ADDRESS: "0.0.0.0:4180"
      OAUTH2_PROXY_REVERSE_PROXY: "true"
      OAUTH2_PROXY_SET_XAUTHREQUEST: "true"
      OAUTH2_PROXY_COOKIE_SECURE: "true"
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.oauth2.rule=Host(`auth.example.com`)"
      - "traefik.http.routers.oauth2.entrypoints=websecure"
      - "traefik.http.routers.oauth2.tls.certresolver=le"
      - "traefik.http.services.oauth2.loadbalancer.server.port=4180"
      # ForwardAuthミドルウェア定義
      - "traefik.http.middlewares.oauth-auth.forwardauth.address=http://oauth2-proxy:4180/oauth2/auth"
      - "traefik.http.middlewares.oauth-auth.forwardauth.trustForwardHeader=true"
      - "traefik.http.middlewares.oauth-auth.forwardauth.authResponseHeaders=X-Auth-Request-User,X-Auth-Request-Email"

  # 保護対象アプリ
  admin-dashboard:
    image: my-admin-app:latest
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.admin.rule=Host(`admin.example.com`)"
      - "traefik.http.routers.admin.entrypoints=websecure"
      - "traefik.http.routers.admin.tls.certresolver=le"
      # OAuth2認証を適用
      - "traefik.http.routers.admin.middlewares=oauth-auth"
      - "traefik.http.services.admin.loadbalancer.server.port=3000"
```

---

## 7. 監視・ログ・トラブルシューティング

### Traefikメトリクス + Prometheus + Grafana

```yaml
# docker-compose.monitoring.yml
services:
  traefik:
    image: traefik:v3.1
    command:
      - "--metrics.prometheus=true"
      - "--metrics.prometheus.entryPoint=metrics"
      - "--entryPoints.metrics.address=:8082"
      - "--accesslog=true"
      - "--accesslog.format=json"
      - "--accesslog.filepath=/logs/access.log"
      - "--accesslog.fields.headers.defaultMode=keep"

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    networks:
      - monitoring
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.prometheus.rule=Host(`prometheus.example.com`)"
      - "traefik.http.routers.prometheus.middlewares=oauth-auth"

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
    networks:
      - monitoring
      - proxy
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.grafana.rule=Host(`grafana.example.com`)"
      - "traefik.http.services.grafana.loadbalancer.server.port=3000"

volumes:
  prometheus-data:
  grafana-data:

networks:
  monitoring:
    internal: true
```

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: traefik
    static_configs:
      - targets: ["traefik:8082"]

  - job_name: nginx
    static_configs:
      - targets: ["nginx-exporter:9113"]
```

### Nginx ログ分析とアラート

```yaml
# docker-compose.logging.yml
services:
  # Nginxメトリクスエクスポーター
  nginx-exporter:
    image: nginx/nginx-prometheus-exporter:latest
    command:
      - -nginx.scrape-uri=http://nginx:8080/stub_status
    networks:
      - monitoring

  # ログ転送
  fluentd:
    image: fluent/fluentd:v1.17
    volumes:
      - ./fluentd/fluent.conf:/fluentd/etc/fluent.conf:ro
      - nginx-logs:/var/log/nginx:ro
    networks:
      - monitoring
```

```nginx
# nginx/conf.d/status.conf
# メトリクス用スタブステータス（外部非公開）
server {
    listen 8080;
    allow 10.0.0.0/8;
    allow 172.16.0.0/12;
    deny all;

    location /stub_status {
        stub_status;
    }

    location /health {
        access_log off;
        return 200 "OK\n";
        add_header Content-Type text/plain;
    }
}
```

### トラブルシューティングチェックリスト

| 症状 | 原因 | 対処法 |
|------|------|--------|
| 502 Bad Gateway | バックエンド未起動 or ネットワーク不通 | `docker compose logs backend` でログ確認 |
| 503 Service Unavailable | upstream サーバーが全てダウン | ヘルスチェック確認、`docker compose ps` |
| 504 Gateway Timeout | バックエンドの応答が遅い | `proxy_read_timeout` の値を増やす |
| SSL証明書エラー | 証明書の期限切れ or パス誤り | `openssl x509 -dates -in cert.pem` |
| WebSocket接続切れ | タイムアウト設定不足 | `proxy_read_timeout 86400s` を設定 |
| Cookie紛失 | Secure/SameSite設定不整合 | HTTPS強制、Cookie設定を確認 |
| IPアドレスが127.0.0.1 | X-Forwarded-For 未設定 | `proxy_set_header X-Forwarded-For` |
| CORSエラー | ヘッダー未設定 | add_header / middleware で CORS設定 |
| Let's Encrypt失敗 | 80番ポート到達不可 | ファイアウォール、DNS設定確認 |
| 413 Entity Too Large | ファイルサイズ制限 | `client_max_body_size 50m` を設定 |

```bash
#!/bin/bash
# proxy-debug.sh - リバースプロキシのデバッグスクリプト

DOMAIN=${1:-"example.com"}

echo "=== DNS解決 ==="
dig +short ${DOMAIN}

echo ""
echo "=== ポート到達性 ==="
nc -zv ${DOMAIN} 80 2>&1
nc -zv ${DOMAIN} 443 2>&1

echo ""
echo "=== HTTP応答 ==="
curl -sI http://${DOMAIN} | head -5

echo ""
echo "=== HTTPS応答 ==="
curl -sI https://${DOMAIN} | head -10

echo ""
echo "=== SSL証明書 ==="
echo | openssl s_client -connect ${DOMAIN}:443 -servername ${DOMAIN} 2>/dev/null | openssl x509 -noout -dates

echo ""
echo "=== レスポンスヘッダー ==="
curl -sI https://${DOMAIN} | grep -iE "server|x-forwarded|strict-transport|x-frame|x-content-type"

echo ""
echo "=== Dockerコンテナ状態 ==="
docker compose ps 2>/dev/null || docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "=== Nginx設定テスト ==="
docker compose exec nginx nginx -t 2>&1 || echo "Nginxコンテナなし"

echo ""
echo "=== 直近のエラーログ ==="
docker compose logs --tail=20 nginx 2>/dev/null || docker compose logs --tail=20 traefik 2>/dev/null
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

### アンチパターン3: SSL設定の不備

```nginx
# NG: 古いプロトコル・弱い暗号スイートを許可
ssl_protocols SSLv3 TLSv1 TLSv1.1 TLSv1.2 TLSv1.3;
ssl_ciphers ALL:!aNULL;

# NG: HSTSヘッダーなし
# → ユーザーがHTTPでアクセスした場合のダウングレード攻撃に脆弱

# OK: TLS 1.2/1.3のみ、強い暗号スイート、HSTS付き
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
```

**なぜ問題か**: SSL Labs のスコアが低下し、中間者攻撃やプロトコルダウングレード攻撃のリスクがある。PCI DSS準拠にはTLS 1.2以上が必須。

### アンチパターン4: タイムアウト設定の不足

```nginx
# NG: デフォルトタイムアウト（60秒）のまま大きなファイルアップロードを受け付ける
location /api/upload {
    proxy_pass http://backend;
    # タイムアウト設定なし → 大きなファイルで504エラー
}

# OK: エンドポイントに応じた適切なタイムアウト
location /api/upload {
    proxy_pass http://backend;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
    proxy_connect_timeout 60s;
    client_max_body_size 100m;
    client_body_timeout 300s;
}

# WebSocketエンドポイントの場合
location /ws {
    proxy_pass http://ws-backend;
    proxy_read_timeout 86400s;  # 24時間
    proxy_send_timeout 86400s;
}
```

**なぜ問題か**: ファイルアップロードやWebSocket接続が不定期に切断され、ユーザー体験を損なう。逆にタイムアウトが長すぎるとリソースを浪費するため、エンドポイント単位で適切に設定する。

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

```nginx
# 信頼するプロキシの設定
set_real_ip_from 10.0.0.0/8;
set_real_ip_from 172.16.0.0/12;
real_ip_header X-Forwarded-For;
real_ip_recursive on;
```

### Q4: ワイルドカード証明書をDocker環境で使うには？

ワイルドカード証明書（`*.example.com`）にはDNS-01チャレンジが必要。TraefikではDNSプロバイダーを設定して自動取得できる。

```yaml
# Traefik + Cloudflare DNS-01チャレンジ
services:
  traefik:
    image: traefik:v3.1
    command:
      - "--certificatesresolvers.le.acme.dnschallenge=true"
      - "--certificatesresolvers.le.acme.dnschallenge.provider=cloudflare"
      - "--certificatesresolvers.le.acme.email=admin@example.com"
      - "--certificatesresolvers.le.acme.storage=/letsencrypt/acme.json"
    environment:
      CF_API_EMAIL: ${CF_API_EMAIL}
      CF_DNS_API_TOKEN: ${CF_DNS_API_TOKEN}
```

### Q5: Nginxでマイクロサービスの動的ルーティングを実現するには？

Nginx単体では動的ルーティングは困難だが、以下の方法で対応できる。

1. **nginx-proxy（jwilder/nginx-proxy）**: Docker環境変数ベースの自動設定
2. **consul-template**: Consul + テンプレートによる自動設定生成
3. **confd**: etcd/Consul/環境変数からのテンプレート生成

```yaml
# jwilder/nginx-proxy を使った動的ルーティング
services:
  nginx-proxy:
    image: nginxproxy/nginx-proxy
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/tmp/docker.sock:ro
      - certs:/etc/nginx/certs:ro

  # VIRTUAL_HOSTを設定するだけで自動ルーティング
  app1:
    image: my-app1:latest
    environment:
      VIRTUAL_HOST: app1.example.com
      VIRTUAL_PORT: 3000

  app2:
    image: my-app2:latest
    environment:
      VIRTUAL_HOST: app2.example.com
      VIRTUAL_PORT: 8080
```

### Q6: Docker Composeのscaleとロードバランシングを組み合わせるには？

```yaml
# docker-compose.yml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
    depends_on:
      - app

  app:
    image: my-app:latest
    # ポートは公開しない
    deploy:
      replicas: 3
```

```nginx
# nginx/conf.d/default.conf
# Docker Composeの内部DNSラウンドロビンを利用
upstream app_cluster {
    # DockerのDNS解決で自動的にスケールされた全インスタンスに分配
    server app:8080;
}

# ただし、DNS解決はキャッシュされるため、resolver設定を追加
resolver 127.0.0.11 valid=10s;

server {
    listen 80;
    location / {
        set $upstream app:8080;
        proxy_pass http://$upstream;
    }
}
```

```bash
# スケールアップ
docker compose up -d --scale app=5

# スケールダウン
docker compose up -d --scale app=2
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| Nginx | 静的設定、高性能、広い知見。`proxy_pass` で転送 |
| Traefik | Docker連携自動検出、ラベルで設定、ACME内蔵 |
| Caddy | デフォルトHTTPS、シンプルな設定、HTTP/3対応 |
| SSL/TLS | Let's Encryptで無料証明書。TLS 1.2以上を強制 |
| mTLS | クライアント証明書による相互認証。内部API保護に有用 |
| ロードバランシング | round_robin / least_conn / ip_hash を用途別に選択 |
| WebSocket | Upgrade/Connection ヘッダーとタイムアウト延長が必須 |
| gRPC | HTTP/2必須。Nginxは `grpc_pass`、Traefikは `h2c` スキーム |
| TCP/UDP | Nginx stream モジュール、Traefik TCPルーター |
| セキュリティ | バックエンドポート非公開、Dockerソケット読取専用 |
| ヘッダー | X-Real-IP / X-Forwarded-For / HSTS を必ず設定 |
| 監視 | Prometheus + Grafana でメトリクス可視化 |
| デプロイ | ローリングアップデートでゼロダウンタイム |

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
6. Caddy公式ドキュメント "Reverse Proxy" -- https://caddyserver.com/docs/caddyfile/directives/reverse_proxy
7. Nginx公式ドキュメント "Stream Module" -- https://nginx.org/en/docs/stream/ngx_stream_core_module.html
8. OWASP "Secure Headers Project" -- https://owasp.org/www-project-secure-headers/
9. Nginx公式ドキュメント "gRPC Proxy" -- https://nginx.org/en/docs/http/ngx_http_grpc_module.html
10. OAuth2 Proxy Documentation -- https://oauth2-proxy.github.io/oauth2-proxy/

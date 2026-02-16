# Dockerネットワーク

> コンテナ間通信とホスト・外部ネットワークとの接続を制御するDockerネットワーキングの全体像を理解する。

---

## この章で学ぶこと

1. **bridge / host / overlay の3大ネットワークドライバーの違いと使い分け**を理解する
2. **ポートマッピングと内蔵DNSによるサービスディスカバリ**の仕組みを習得する
3. **マルチホスト環境でのオーバーレイネットワーク**の構築手順を把握する
4. **ネットワーク分離によるセキュリティ設計**の実践パターンを学ぶ
5. **トラブルシューティング手法**を身につけ、ネットワーク問題を迅速に解決できるようになる

---

## 1. Dockerネットワークの基本概念

Dockerはコンテナごとに独立したネットワーク名前空間（Network Namespace）を割り当てる。これにより各コンテナは固有のIPアドレス、ルーティングテーブル、iptablesルールを持つ。

### ネットワークドライバーの全体像

```
┌─────────────────────────────────────────────────────┐
│                   Docker Host                       │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Container│  │ Container│  │ Container│         │
│  │   App A  │  │   App B  │  │   App C  │         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘         │
│       │              │              │               │
│  ┌────▼──────────────▼──────────────▼─────┐        │
│  │         docker0 (bridge)               │        │
│  │         172.17.0.0/16                  │        │
│  └────────────────┬───────────────────────┘        │
│                   │                                 │
│              ┌────▼────┐                           │
│              │  eth0   │  ← ホストNIC              │
│              └────┬────┘                           │
└───────────────────┼─────────────────────────────────┘
                    │
               外部ネットワーク
```

### ネットワーク名前空間の仕組み

Linux のネットワーク名前空間は、各コンテナに独立したネットワークスタックを提供する。これは以下の要素を含む。

```
┌───────────── Container Network Namespace ──────────────┐
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │  eth0 (veth pair)     172.17.0.2/16              │  │
│  │  ルーティングテーブル                              │  │
│  │    default via 172.17.0.1 dev eth0               │  │
│  │  DNS resolver                                     │  │
│  │    nameserver 127.0.0.11                          │  │
│  │  iptables ルール                                  │  │
│  │  ループバック (lo: 127.0.0.1)                     │  │
│  └─────────────────────────────────────────────────┘  │
│                                                         │
│      ↕ veth pair (仮想イーサネットペア)                   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │  Host側: vethXXXXXX → docker0 bridge に接続      │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### ネットワークドライバー比較表

| ドライバー | スコープ | 用途 | IPアドレス | パフォーマンス |
|-----------|---------|------|-----------|--------------|
| bridge | 単一ホスト | デフォルト、開発環境 | 自動割当(172.17.x.x) | 良好 |
| host | 単一ホスト | 最大パフォーマンス | ホストと共有 | 最高 |
| overlay | マルチホスト | Swarm/本番クラスタ | 自動割当(10.0.x.x) | VXLANオーバーヘッド |
| macvlan | 単一ホスト | 物理NIC直結が必要 | 物理ネットワーク | 良好 |
| ipvlan | 単一ホスト | macvlanの代替 | 物理ネットワーク | 良好 |
| none | - | ネットワーク無効化 | なし | - |

### ドライバー選択のフローチャート

```
コンテナにネットワークが必要か？
    │
    ├── No ──► none ドライバー
    │
    └── Yes ──► マルチホスト通信が必要か？
                   │
                   ├── Yes ──► overlay ドライバー (Swarm/K8s)
                   │
                   └── No ──► 最大パフォーマンスが必要か？
                                │
                                ├── Yes ──► host ドライバー (Linux のみ)
                                │
                                └── No ──► 物理ネットワークに直接参加が必要か？
                                             │
                                             ├── Yes ──► macvlan / ipvlan
                                             │
                                             └── No ──► bridge ドライバー (推奨)
```

---

## 2. Bridgeネットワーク

### デフォルトbridge vs ユーザー定義bridge

Dockerインストール時に作成される `docker0` ブリッジ（デフォルトbridge）と、ユーザーが明示的に作成するカスタムbridgeには重要な差がある。

| 特性 | デフォルトbridge | ユーザー定義bridge |
|-----|-----------------|------------------|
| DNS解決 | 不可（IPのみ） | コンテナ名で解決可能 |
| 自動接続 | 全コンテナが接続 | 明示的に指定 |
| ネットワーク分離 | 分離なし | ネットワーク単位で分離 |
| ライブ接続/切断 | 不可 | 可能 |
| 環境変数共有 | `--link`（非推奨） | 不要 |
| カスタムサブネット | 不可 | 可能 |
| MTU設定 | 不可 | 可能 |

### コード例1: ユーザー定義bridgeネットワークの作成

```bash
# ユーザー定義bridgeネットワークを作成
docker network create \
  --driver bridge \
  --subnet 192.168.100.0/24 \
  --gateway 192.168.100.1 \
  --ip-range 192.168.100.128/25 \
  my-app-network

# ネットワークの詳細を確認
docker network inspect my-app-network

# コンテナをネットワークに接続して起動
docker run -d \
  --name web-server \
  --network my-app-network \
  nginx:alpine

docker run -d \
  --name api-server \
  --network my-app-network \
  node:20-alpine sleep infinity

# web-server から api-server へ名前解決で通信できる
docker exec web-server ping -c 3 api-server

# 既存のコンテナをネットワークに動的に接続
docker network connect my-app-network existing-container

# ネットワークから切断
docker network disconnect my-app-network existing-container
```

### コード例2: Docker Composeでのネットワーク定義

```yaml
# docker-compose.yml
services:
  frontend:
    image: nginx:alpine
    networks:
      - frontend-net
    ports:
      - "80:80"

  api:
    build: ./api
    networks:
      - frontend-net
      - backend-net
    environment:
      DB_HOST: database  # DNS名で参照

  database:
    image: postgres:16-alpine
    networks:
      - backend-net
    environment:
      POSTGRES_PASSWORD: secret

networks:
  frontend-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/24
  backend-net:
    driver: bridge
    internal: true  # 外部アクセス不可
    ipam:
      config:
        - subnet: 172.21.0.0/24
```

ネットワーク分離の構造:

```
      外部 (インターネット)
          │
          │ :80
    ┌─────▼─────┐
    │  frontend  │──── frontend-net (172.20.0.0/24)
    └───────────┘          │
                     ┌─────▼─────┐
                     │    api    │──── backend-net (172.21.0.0/24)
                     └───────────┘      │  ※ internal: true
                                  ┌─────▼─────┐
                                  │  database  │
                                  └───────────┘
                                  (外部アクセス不可)
```

### コード例2b: 高度なネットワーク設定オプション

```yaml
# docker-compose.yml - 高度なネットワーク設定
networks:
  # カスタムサブネットとゲートウェイ
  custom-net:
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: 172.30.0.0/24
          gateway: 172.30.0.1
          ip_range: 172.30.0.128/25    # IPアドレスの割り当て範囲

  # 外部ネットワーク（既に存在するネットワークを参照）
  existing-net:
    external: true
    name: my-existing-network

  # MTUとドライバーオプション
  optimized-net:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.enable_icc: "true"           # コンテナ間通信
      com.docker.network.bridge.enable_ip_masquerade: "true"  # IPマスカレード
      com.docker.network.bridge.host_binding_ipv4: "0.0.0.0"  # バインドIP
      com.docker.network.driver.mtu: "1500"                   # MTU
    labels:
      environment: production
      team: platform
```

### コード例2c: 複数コンテナをネットワークに参加させる場合のIP固定

```yaml
services:
  app:
    image: my-app:latest
    networks:
      app-net:
        ipv4_address: 172.25.0.10
        aliases:
          - application
          - webapp

  db:
    image: postgres:16-alpine
    networks:
      app-net:
        ipv4_address: 172.25.0.20
        aliases:
          - database
          - postgres

  cache:
    image: redis:7-alpine
    networks:
      app-net:
        ipv4_address: 172.25.0.30
        aliases:
          - redis
          - cache

networks:
  app-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.25.0.0/24
          gateway: 172.25.0.1
```

---

## 3. ポートマッピング

### コード例3: 各種ポートマッピング

```bash
# 基本: ホストの8080番をコンテナの80番にマッピング
docker run -d -p 8080:80 nginx:alpine

# 特定IPにバインド（localhostのみ公開）
docker run -d -p 127.0.0.1:8080:80 nginx:alpine

# UDPポートの公開
docker run -d -p 5353:53/udp dns-server

# ランダムポート割当（ホスト側ポートを自動決定）
docker run -d -p 80 nginx:alpine
# → docker port <container> で確認

# 複数ポートの公開
docker run -d \
  -p 80:80 \
  -p 443:443 \
  -p 8443:8443 \
  nginx:alpine

# ポート範囲の公開
docker run -d -p 7000-7010:7000-7010 my-app

# TCP と UDP の両方を公開
docker run -d -p 53:53/tcp -p 53:53/udp dns-server

# 全ポートを公開 (-P: Dockerfile の EXPOSE で定義されたポート)
docker run -d -P nginx:alpine
```

### ポートマッピングの仕組み（iptables）

```
外部クライアント
    │
    │ :8080
    ▼
┌──────────────────────────────────────┐
│  iptables NAT テーブル               │
│  DNAT: 0.0.0.0:8080 → 172.17.0.2:80│
│                                      │
│  ┌────────────────────────────┐     │
│  │  docker-proxy (userland)   │     │
│  │  localhost:8080 → ctr:80   │     │
│  └────────────────────────────┘     │
│                                      │
│  ┌────────────────┐                 │
│  │  Container      │                 │
│  │  172.17.0.2:80  │                 │
│  └────────────────┘                 │
└──────────────────────────────────────┘
```

### Docker Compose でのポートマッピング

```yaml
services:
  web:
    image: nginx:alpine
    ports:
      # 短縮構文
      - "80:80"                    # ホスト:コンテナ
      - "443:443"
      - "127.0.0.1:8080:80"       # 特定IPにバインド

      # 長文構文（推奨: 意図が明確）
      - target: 80                 # コンテナ側ポート
        published: 8080            # ホスト側ポート
        protocol: tcp
        mode: host                 # host or ingress

  db:
    image: postgres:16-alpine
    ports:
      # 開発時のみ外部公開（本番では expose のみ）
      - "127.0.0.1:5432:5432"

    # expose: コンテナ間通信のみ（ホストには非公開）
    expose:
      - "5432"
```

### ポートの競合を防ぐパターン

```yaml
# .env ファイルでポートを変数化
APP_PORT=3000
DB_PORT=5432
REDIS_PORT=6379

# docker-compose.yml
services:
  app:
    ports:
      - "${APP_PORT:-3000}:3000"
  db:
    ports:
      - "127.0.0.1:${DB_PORT:-5432}:5432"
  redis:
    ports:
      - "127.0.0.1:${REDIS_PORT:-6379}:6379"
```

---

## 4. Hostネットワーク

hostドライバーではコンテナがホストのネットワーク名前空間を直接共有する。ネットワーク変換が不要なため、最もパフォーマンスが高い。

### コード例4: hostネットワークの使用

```bash
# hostネットワークでNginxを起動
# コンテナのポート80がそのままホストのポート80になる
docker run -d \
  --network host \
  --name web \
  nginx:alpine

# ポートマッピング不要（-p フラグは無視される）
curl http://localhost:80

# host ネットワークの確認
docker inspect web --format '{{.NetworkSettings.Networks}}'
```

> **注意**: hostネットワークはLinuxでのみフルサポートされる。macOS/Windowsでは Docker Desktop の仮想マシン内でのhost共有となるため、ホストOSから直接アクセスできない。

### hostネットワークの使い分け

| ユースケース | 理由 |
|-------------|------|
| 高頻度の小パケット通信 | NATオーバーヘッドを排除 |
| ネットワーク監視ツール | ホストのネットワークスタックに直接アクセス |
| パフォーマンスベンチマーク | ネットワーク変換の影響を排除 |
| レガシーアプリの移行 | ポートマッピングの変更が困難な場合 |

### Docker Compose でのhostネットワーク

```yaml
services:
  # パフォーマンスが重要なサービス
  high-perf-app:
    image: my-app:latest
    network_mode: host
    # ports: は使用不可（hostネットワークでは無視される）
    environment:
      PORT: 8080

  # 監視ツール（ホストのネットワークを監視）
  prometheus:
    image: prom/prometheus:latest
    network_mode: host
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

---

## 5. Overlayネットワーク

### マルチホスト通信の仕組み

```
┌─────────── Host A ───────────┐   ┌─────────── Host B ───────────┐
│ ┌─────────┐  ┌─────────┐    │   │    ┌─────────┐  ┌─────────┐ │
│ │  Web-1  │  │  Web-2  │    │   │    │  API-1  │  │  DB-1   │ │
│ └────┬────┘  └────┬────┘    │   │    └────┬────┘  └────┬────┘ │
│      │            │          │   │         │            │      │
│ ┌────▼────────────▼────┐    │   │    ┌────▼────────────▼────┐ │
│ │  Overlay: my-overlay │    │   │    │  Overlay: my-overlay │ │
│ │  (VXLAN トンネル)     │◄──┼───┼───►│  (VXLAN トンネル)     │ │
│ └──────────────────────┘    │   │    └──────────────────────┘ │
│           │                  │   │              │              │
│      ┌────▼────┐            │   │         ┌────▼────┐        │
│      │  eth0   │            │   │         │  eth0   │        │
└──────┴─────────┴────────────┘   └─────────┴─────────┴────────┘
            │    UDP 4789 (VXLAN)        │
            └────────────────────────────┘
```

### VXLANの詳細な仕組み

VXLAN (Virtual Extensible LAN) は、レイヤー2フレームをレイヤー3パケットにカプセル化する技術である。これにより、物理ネットワークを超えて仮想的なレイヤー2ネットワークを構築できる。

```
┌──────────────────────────────────────────────┐
│          VXLANカプセル化パケット                │
│                                               │
│  ┌──────────────────────────────────────┐    │
│  │ Outer Ethernet Header                │    │
│  │ Outer IP Header (Host A → Host B)   │    │
│  │ Outer UDP Header (src → dst:4789)   │    │
│  │ VXLAN Header (VNI: Network ID)      │    │
│  │  ┌────────────────────────────────┐ │    │
│  │  │ Inner Ethernet Header          │ │    │
│  │  │ Inner IP Header (Ctr → Ctr)   │ │    │
│  │  │ Payload (Application Data)    │ │    │
│  │  └────────────────────────────────┘ │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

### コード例5: Overlay ネットワークの構築（Docker Swarm）

```bash
# Swarmを初期化（マネージャーノード）
docker swarm init --advertise-addr 192.168.1.10

# ワーカーノードをSwarmに参加
docker swarm join --token SWMTKN-xxx 192.168.1.10:2377

# 暗号化付きオーバーレイネットワークを作成
docker network create \
  --driver overlay \
  --subnet 10.10.0.0/16 \
  --opt encrypted \
  --attachable \
  my-overlay

# サービスをオーバーレイネットワーク上にデプロイ
docker service create \
  --name web \
  --network my-overlay \
  --replicas 3 \
  --publish published=80,target=80 \
  nginx:alpine

# ネットワーク状態の確認
docker network inspect my-overlay

# オーバーレイネットワークの暗号化状態を確認
docker network inspect my-overlay --format '{{.Options}}'
```

### Overlay ネットワークの Docker Compose (Swarm mode)

```yaml
# docker-compose.yml (Swarm mode)
services:
  web:
    image: nginx:alpine
    deploy:
      replicas: 3
      placement:
        constraints:
          - node.role == worker
    networks:
      - frontend

  api:
    image: my-api:latest
    deploy:
      replicas: 2
    networks:
      - frontend
      - backend

  db:
    image: postgres:16-alpine
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.type == db
    networks:
      - backend
    volumes:
      - pgdata:/var/lib/postgresql/data

networks:
  frontend:
    driver: overlay
  backend:
    driver: overlay
    internal: true     # 外部アクセス不可
    driver_opts:
      encrypted: "true"  # IPsec暗号化

volumes:
  pgdata:
```

---

## 6. Macvlanネットワーク

macvlanドライバーは、コンテナに物理ネットワーク上のMACアドレスを割り当て、物理ネットワークに直接接続されているかのように見せる。レガシーアプリケーションやネットワーク機器との直接通信が必要な場合に使用する。

### コード例5b: Macvlanネットワークの構築

```bash
# macvlan ネットワークの作成
docker network create \
  --driver macvlan \
  --subnet 192.168.1.0/24 \
  --gateway 192.168.1.1 \
  --opt parent=eth0 \
  macvlan-net

# コンテナを macvlan ネットワークに接続
docker run -d \
  --name legacy-app \
  --network macvlan-net \
  --ip 192.168.1.100 \
  legacy-app:latest

# 802.1Q VLAN タグ付き macvlan
docker network create \
  --driver macvlan \
  --subnet 192.168.10.0/24 \
  --gateway 192.168.10.1 \
  --opt parent=eth0.10 \
  macvlan-vlan10
```

> **注意**: macvlanを使用する場合、ホストマシンからコンテナへの直接通信はデフォルトではできない。これはmacvlanの仕様による制限で、ホストとコンテナ間の通信が必要な場合はIPvlanを検討する。

---

## 7. DNS とサービスディスカバリ

Docker内蔵DNSサーバー（127.0.0.11）がユーザー定義ネットワーク内で自動的にコンテナ名を解決する。

### DNS解決フロー

```
Container A が "api-server" を名前解決
    │
    ▼
┌──────────────────────┐
│ コンテナ内 resolver   │
│ /etc/resolv.conf     │
│ nameserver 127.0.0.11│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐     見つかった
│ Docker 内蔵 DNS       │────────────►  172.20.0.3 (api-server)
│ (127.0.0.11)         │
└──────────┬───────────┘
           │ 見つからない
           ▼
┌──────────────────────┐
│ ホストの DNS resolver │────────────►  外部DNS応答
│ (8.8.8.8 等)         │
└──────────────────────┘
```

### DNS解決の対象と優先順位

Docker内蔵DNSは以下の順序で名前を解決する。

```
1. コンテナ名 (--name で指定した名前)
   例: "api-server" → 172.20.0.3

2. ネットワークエイリアス (--network-alias / networks.aliases)
   例: "api" → 172.20.0.3 (複数コンテナの場合はラウンドロビン)

3. Compose のサービス名
   例: "api" → サービスに属するコンテナのIP

4. 外部DNS (ホストのresolv.confを使用)
   例: "google.com" → 142.250.xxx.xxx
```

### コード例6: エイリアスとサービスディスカバリ

```yaml
# docker-compose.yml
services:
  app:
    image: my-app:latest
    networks:
      app-net:
        aliases:
          - application
          - webapp

  cache-primary:
    image: redis:7-alpine
    networks:
      app-net:
        aliases:
          - redis
          - cache

  cache-replica:
    image: redis:7-alpine
    command: redis-server --replicaof redis 6379
    networks:
      app-net:
        aliases:
          - redis    # 同一エイリアスでラウンドロビンDNS
          - cache

networks:
  app-net:
    driver: bridge
```

```bash
# DNS解決の確認
docker run --rm --network app-net busybox nslookup redis
# → cache-primary と cache-replica の両方のIPが返る

# 特定コンテナのDNS設定を確認
docker exec app cat /etc/resolv.conf

# DNS解決のテスト（dig コマンド）
docker run --rm --network app-net tutum/dnsutils dig redis

# 逆引きDNS
docker run --rm --network app-net busybox nslookup 172.20.0.3
```

### カスタムDNS設定

```yaml
services:
  app:
    image: my-app:latest
    dns:
      - 8.8.8.8
      - 8.8.4.4
    dns_search:
      - example.com
      - internal.example.com
    dns_opt:
      - "ndots:2"
      - "timeout:3"
    extra_hosts:
      - "host.docker.internal:host-gateway"   # ホストマシンへのアクセス
      - "legacy-server:192.168.1.100"         # 手動DNS設定
```

---

## 8. 実践的なネットワーク構成例

### コード例7: マイクロサービス構成

```yaml
# docker-compose.yml - マイクロサービス構成
services:
  # --- フロントエンド層 ---
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    networks:
      - public
      - app-tier
    depends_on:
      - frontend

  frontend:
    build: ./frontend
    networks:
      - app-tier

  # --- アプリケーション層 ---
  user-service:
    build: ./services/user
    networks:
      - app-tier
      - data-tier

  order-service:
    build: ./services/order
    networks:
      - app-tier
      - data-tier
      - message-tier

  notification-service:
    build: ./services/notification
    networks:
      - app-tier
      - message-tier

  # --- メッセージング層 ---
  rabbitmq:
    image: rabbitmq:3-management-alpine
    networks:
      - message-tier

  # --- データ層 ---
  postgres:
    image: postgres:16-alpine
    networks:
      - data-tier

  redis:
    image: redis:7-alpine
    networks:
      - data-tier

networks:
  public:
    driver: bridge
  app-tier:
    driver: bridge
    internal: false
  data-tier:
    driver: bridge
    internal: true   # 外部アクセスを完全遮断
  message-tier:
    driver: bridge
    internal: true
```

### ネットワークアクセスマトリクス

上記構成における各サービスのネットワークアクセスを整理する。

| サービス | public | app-tier | data-tier | message-tier |
|---------|--------|----------|-----------|-------------|
| nginx | ✅ | ✅ | - | - |
| frontend | - | ✅ | - | - |
| user-service | - | ✅ | ✅ | - |
| order-service | - | ✅ | ✅ | ✅ |
| notification-service | - | ✅ | - | ✅ |
| rabbitmq | - | - | - | ✅ |
| postgres | - | - | ✅ | - |
| redis | - | - | ✅ | - |

### コード例7b: セキュアなマルチテナント構成

```yaml
# docker-compose.yml - マルチテナント構成
services:
  # 共有リバースプロキシ
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    networks:
      - proxy-net

  # テナントA
  tenant-a-app:
    image: my-app:latest
    environment:
      TENANT_ID: tenant-a
    networks:
      - proxy-net
      - tenant-a-net

  tenant-a-db:
    image: postgres:16-alpine
    networks:
      - tenant-a-net    # テナントA専用ネットワークのみ

  # テナントB
  tenant-b-app:
    image: my-app:latest
    environment:
      TENANT_ID: tenant-b
    networks:
      - proxy-net
      - tenant-b-net

  tenant-b-db:
    image: postgres:16-alpine
    networks:
      - tenant-b-net    # テナントB専用ネットワークのみ

networks:
  proxy-net:
    driver: bridge
  tenant-a-net:
    driver: bridge
    internal: true     # テナントAのDBは外部アクセス不可
  tenant-b-net:
    driver: bridge
    internal: true     # テナントBのDBも外部アクセス不可
```

---

## 9. ネットワークのトラブルシューティング

### コード例8: デバッグコマンド集

```bash
# ネットワーク一覧を確認
docker network ls

# 特定ネットワークの詳細（接続コンテナ、IPAM設定）
docker network inspect my-app-network

# コンテナのネットワーク設定を確認
docker inspect --format='{{json .NetworkSettings.Networks}}' my-container | jq .

# コンテナのIPアドレスを取得
docker inspect --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' my-container

# コンテナのポートマッピングを確認
docker port my-container

# コンテナ内からネットワーク診断
docker run --rm --network my-app-network nicolaka/netshoot \
  bash -c "
    # DNS解決テスト
    nslookup api-server
    # TCP接続テスト
    nc -zv api-server 8080
    # ルーティング確認
    ip route
    # ネットワークインターフェース確認
    ip addr
    # TCPダンプ（パケットキャプチャ）
    tcpdump -i eth0 -c 20 port 80
  "

# ホスト側からiptablesルールを確認
sudo iptables -t nat -L -n | grep DOCKER

# Docker ネットワーク関連のイベントを監視
docker events --filter type=network

# 使用されていないネットワークの一括削除
docker network prune
```

### よくあるネットワーク問題と解決策

| 問題 | 原因 | 解決策 |
|------|------|--------|
| コンテナ間で名前解決できない | デフォルトbridgeを使用 | ユーザー定義ネットワークに変更 |
| ポートが既に使用中 | ホスト側でポート競合 | `docker ps` で確認し、停止 or ポート変更 |
| コンテナから外部に通信できない | DNS設定の問題 | `dns:` オプションで外部DNSを指定 |
| 通信が遅い | macOS のBind Mount I/O | VirtioFS に切り替え |
| ポートフォワードが動かない | iptables ルールの問題 | `sudo iptables -t nat -L -n` で確認 |
| コンテナ起動時にIP競合 | サブネットの重複 | カスタムサブネットを指定 |

### netshoot を使った詳細診断

```bash
# netshoot コンテナで対象ネットワークに接続して診断
docker run -it --rm --network my-app-network nicolaka/netshoot

# 内部で使えるコマンド:
# ping, traceroute, nslookup, dig
# curl, wget, nc (netcat)
# tcpdump, iperf, mtr
# ip, ss, netstat
# nmap, iftop

# 例: コンテナ間の帯域幅テスト
# サーバー側
docker run -d --name iperf-server --network my-app-network nicolaka/netshoot iperf3 -s
# クライアント側
docker run --rm --network my-app-network nicolaka/netshoot iperf3 -c iperf-server
```

---

## 10. IPv6 対応

```yaml
# docker-compose.yml - IPv6 有効化
services:
  app:
    image: my-app:latest
    networks:
      - dual-stack

networks:
  dual-stack:
    driver: bridge
    enable_ipv6: true
    ipam:
      config:
        - subnet: 172.28.0.0/24    # IPv4
        - subnet: fd00::/64         # IPv6 ULA
```

```bash
# Docker デーモンでIPv6を有効化
# /etc/docker/daemon.json
{
  "ipv6": true,
  "fixed-cidr-v6": "fd00::/80"
}
```

---

## アンチパターン

### アンチパターン1: デフォルトbridgeネットワークへの依存

```bash
# NG: デフォルトbridgeではDNS解決が機能しない
docker run -d --name app1 my-app
docker run -d --name app2 my-app
docker exec app1 ping app2  # 失敗: 名前解決できない

# OK: ユーザー定義ネットワークを使う
docker network create my-net
docker run -d --name app1 --network my-net my-app
docker run -d --name app2 --network my-net my-app
docker exec app1 ping app2  # 成功
```

**なぜ問題か**: デフォルトbridgeではコンテナ名によるDNS解決が無効。非推奨の `--link` を使わないと名前で通信できず、IPアドレスのハードコーディングが必要になる。

### アンチパターン2: 全コンテナを同一ネットワークに配置

```yaml
# NG: 全サービスがフラットに通信可能
services:
  nginx:
    networks: [shared]
  app:
    networks: [shared]
  database:
    networks: [shared]  # フロントエンドからDBに直接到達可能

# OK: 層ごとにネットワークを分離
services:
  nginx:
    networks: [frontend]
  app:
    networks: [frontend, backend]
  database:
    networks: [backend]  # backendからのみアクセス可能
```

**なぜ問題か**: 攻撃者がフロントエンドコンテナに侵入した場合、同一ネットワーク上のDBに直接アクセスできてしまう。ネットワーク分離はセキュリティの基本防御策。

### アンチパターン3: ポートの過剰公開

```bash
# NG: 0.0.0.0 にバインド（全インターフェースに公開）
docker run -d -p 5432:5432 postgres:16

# OK: localhostのみにバインド
docker run -d -p 127.0.0.1:5432:5432 postgres:16
```

**なぜ問題か**: 外部から直接データベースポートにアクセス可能になり、セキュリティリスクが極めて高い。

### アンチパターン4: ハードコードされたIPアドレスへの依存

```yaml
# NG: IPアドレスをハードコード
services:
  app:
    environment:
      DB_HOST: 172.18.0.5  # IPアドレスは変わる可能性がある

# OK: DNS名（サービス名）を使用
services:
  app:
    environment:
      DB_HOST: database    # Composeのサービス名はDNSで解決される
```

**なぜ問題か**: コンテナのIPアドレスは起動順序やネットワーク状態によって変わる。DNS名を使用することで、IPアドレスの変更に影響されない安定した通信が実現できる。

---

## FAQ

### Q1: コンテナ間通信でlocalhostを使えないのはなぜ？

各コンテナは独立したネットワーク名前空間を持つため、`localhost` は常に自分自身を指す。他のコンテナと通信するにはコンテナ名（DNS名）またはIPアドレスを使用する。Docker Composeではサービス名がそのままDNS名になる。

### Q2: overlayネットワークのパフォーマンスオーバーヘッドはどの程度？

VXLANカプセル化のオーバーヘッドは通常5-10%程度。`--opt encrypted` を有効にするとIPsec暗号化が加わり、CPU負荷が増加する。高スループットが必要な場合はホストネットワークまたはmacvlanを検討する。

### Q3: docker-compose up でネットワークが自動作成されるが、名前はどうなる？

`<プロジェクトディレクトリ名>_<networks名>` の形式で作成される。例えばディレクトリが `myapp` でネットワーク名が `backend` なら `myapp_backend` となる。`name:` フィールドで明示的に指定することも可能。

```yaml
networks:
  backend:
    name: my-custom-backend  # 明示的な名前指定
```

### Q4: コンテナからホストマシンにアクセスするには？

Docker Desktop (macOS/Windows) では `host.docker.internal` という特別なDNS名を使う。Linux では `--add-host=host.docker.internal:host-gateway` を指定する。

```yaml
services:
  app:
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      HOST_API: http://host.docker.internal:4000
```

### Q5: ネットワーク間の通信を制限するにはどうすればよい？

`internal: true` を設定すると、そのネットワークから外部（インターネット）への通信が遮断される。ネットワーク間の通信制限は、サービスが参加するネットワークの設計で制御する。異なるネットワークに属するコンテナは直接通信できない。

### Q6: Docker Composeで外部ネットワーク（既存ネットワーク）に接続するには？

`external: true` を指定することで、Composeプロジェクト外で作成済みのネットワークに接続できる。複数のComposeプロジェクト間で通信が必要な場合に活用する。

```yaml
# プロジェクトA (共有ネットワークを作成)
networks:
  shared:
    name: shared-network
    driver: bridge

# プロジェクトB (既存ネットワークに参加)
networks:
  shared:
    external: true
    name: shared-network
```

```bash
# 手動で共有ネットワークを作成
docker network create shared-network

# プロジェクトBからも利用可能
docker compose up -d
```

### Q7: Docker Composeでネットワークの優先度やデフォルトネットワークを設定するには？

Composeはサービスの最初に定義されたネットワークをデフォルトのルーティング先として使用する。`priority` オプションで明示的に優先度を設定できる。

```yaml
services:
  app:
    networks:
      frontend:
        priority: 1000   # 高い値が優先
      backend:
        priority: 500
      monitoring:
        priority: 100

networks:
  frontend:
  backend:
    internal: true
  monitoring:
    internal: true
```

### Q8: ネットワーク帯域幅の制限はどう設定する？

Docker単体ではネットワーク帯域幅制限のネイティブサポートは限定的だが、`tc`（Traffic Control）コマンドやDockerの `--network-bandwidth` オプション（Docker Swarmモード）で制御可能。

```bash
# tc（Traffic Control）によるコンテナの帯域幅制限
# コンテナのvethインターフェースを特定
CONTAINER_PID=$(docker inspect -f '{{.State.Pid}}' mycontainer)
VETH=$(ip link | grep -A1 "if${CONTAINER_PID}" | head -1 | awk '{print $2}' | tr -d ':')

# 帯域幅を100Mbpsに制限
tc qdisc add dev ${VETH} root tbf rate 100mbit burst 32kbit latency 100ms
```

```yaml
# Docker Swarmモードでのリソース制限
services:
  app:
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 512M
        reservations:
          cpus: "0.5"
          memory: 256M
```

### Q9: IPv6ネットワークの設定方法は？

Docker daemon でIPv6を有効化し、固定サブネットを割り当てる。

```json
{
  "ipv6": true,
  "fixed-cidr-v6": "2001:db8:1::/64"
}
```

```yaml
# docker-compose.yml
networks:
  dual-stack:
    enable_ipv6: true
    ipam:
      config:
        - subnet: 172.28.0.0/16
        - subnet: "2001:db8:2::/64"

services:
  app:
    networks:
      dual-stack:
        ipv4_address: 172.28.0.10
        ipv6_address: "2001:db8:2::10"
```

### Q10: Docker Desktop for Mac/Windowsでのネットワーク制限事項は？

Docker Desktop はLinux VMの中でDockerを実行するため、いくつかの制限がある。

| 機能 | Linux | macOS | Windows |
|------|-------|-------|---------|
| host ネットワーク | 完全サポート | 部分的（VM内のhost） | 部分的 |
| macvlan | 完全サポート | 非サポート | 非サポート |
| ホストからコンテナIP直接アクセス | 可能 | 不可（ポートマッピング必須） | 不可 |
| `host.docker.internal` | 要設定 | デフォルト有効 | デフォルト有効 |
| ネットワークパフォーマンス | ネイティブ | VMオーバーヘッドあり | VMオーバーヘッドあり |

macOS/Windowsでは `host.docker.internal` を使ってホストマシンにアクセスし、ポートマッピングでコンテナに外部からアクセスする。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| bridge | 単一ホストのデフォルトドライバー。ユーザー定義bridgeを推奨 |
| host | ポートマッピング不要で最高性能。Linuxのみフルサポート |
| overlay | マルチホスト通信。Swarm/Kubernetesで使用 |
| macvlan | 物理ネットワーク直結。レガシーアプリ統合に使用 |
| ポートマッピング | `-p host:container` でホストに公開。`127.0.0.1` バインド推奨 |
| DNS | ユーザー定義ネットワーク内で自動名前解決。127.0.0.11 |
| サービスディスカバリ | エイリアスでラウンドロビンDNS。Compose連携で簡便 |
| ネットワーク分離 | `internal: true` で外部遮断。層ごとに分離が鉄則 |
| トラブルシューティング | netshoot コンテナで診断。`docker network inspect` で確認 |

---

## 次に読むべきガイド

- [ボリュームとストレージ](./01-volume-and-storage.md) -- データ永続化とストレージドライバー
- [リバースプロキシ](./02-reverse-proxy.md) -- Nginx/TraefikによるHTTPルーティング
- [コンテナセキュリティ](../06-security/00-container-security.md) -- ネットワーク分離を含む包括的セキュリティ

---

## 参考文献

1. Docker公式ドキュメント "Networking overview" -- https://docs.docker.com/network/
2. Nigel Poulton (2023) *Docker Deep Dive*, Chapter 11: Docker Networking
3. Adrian Mouat (2023) *Using Docker*, Chapter 10: Networking and Service Discovery
4. Docker公式ドキュメント "Use bridge networks" -- https://docs.docker.com/network/bridge/
5. Docker公式ドキュメント "Use overlay networks" -- https://docs.docker.com/network/overlay/
6. Docker公式ドキュメント "Use macvlan networks" -- https://docs.docker.com/network/macvlan/
7. Linux Foundation "Container Networking From Scratch" -- https://www.youtube.com/watch?v=6v_BDHIgOY8
8. Docker公式ドキュメント "Networking with standalone containers" -- https://docs.docker.com/network/network-tutorial-standalone/

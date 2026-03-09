# Dockerネットワーク

> コンテナ間通信とホスト・外部ネットワークとの接続を制御するDockerネットワーキングの全体像を理解する。

---

## この章で学ぶこと

1. **bridge / host / overlay の3大ネットワークドライバーの違いと使い分け**を理解する
2. **ポートマッピングと内蔵DNSによるサービスディスカバリ**の仕組みを習得する
3. **マルチホスト環境でのオーバーレイネットワーク**の構築手順を把握する
4. **ネットワーク分離によるセキュリティ設計**の実践パターンを学ぶ
5. **トラブルシューティング手法**を身につけ、ネットワーク問題を迅速に解決できるようになる


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解

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

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## 実務での適用シナリオ

### シナリオ1: スタートアップでのMVP開発

**状況:** 限られたリソースで素早くプロダクトをリリースする必要がある

**アプローチ:**
- シンプルなアーキテクチャを選択
- 必要最小限の機能に集中
- 自動テストはクリティカルパスのみ
- モニタリングは早期から導入

**学んだ教訓:**
- 完璧を求めすぎない（YAGNI原則）
- ユーザーフィードバックを早期に取得
- 技術的負債は意識的に管理する

### シナリオ2: レガシーシステムのモダナイゼーション

**状況:** 10年以上運用されているシステムを段階的に刷新する

**アプローチ:**
- Strangler Fig パターンで段階的に移行
- 既存のテストがない場合はCharacterization Testを先に作成
- APIゲートウェイで新旧システムを共存
- データ移行は段階的に実施

| フェーズ | 作業内容 | 期間目安 | リスク |
|---------|---------|---------|--------|
| 1. 調査 | 現状分析、依存関係の把握 | 2-4週間 | 低 |
| 2. 基盤 | CI/CD構築、テスト環境 | 4-6週間 | 低 |
| 3. 移行開始 | 周辺機能から順次移行 | 3-6ヶ月 | 中 |
| 4. コア移行 | 中核機能の移行 | 6-12ヶ月 | 高 |
| 5. 完了 | 旧システム廃止 | 2-4週間 | 中 |

### シナリオ3: 大規模チームでの開発

**状況:** 50人以上のエンジニアが同一プロダクトを開発する

**アプローチ:**
- ドメイン駆動設計で境界を明確化
- チームごとにオーナーシップを設定
- 共通ライブラリはInner Source方式で管理
- APIファーストで設計し、チーム間の依存を最小化

```python
# チーム間のAPI契約定義
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """チーム間のAPI契約"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # レスポンスタイムSLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """SLA準拠の確認"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """OpenAPI形式で出力"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# 使用例
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### シナリオ4: パフォーマンスクリティカルなシステム

**状況:** ミリ秒単位のレスポンスが求められるシステム

**最適化ポイント:**
1. キャッシュ戦略（L1: インメモリ、L2: Redis、L3: CDN）
2. 非同期処理の活用
3. コネクションプーリング
4. クエリ最適化とインデックス設計

| 最適化手法 | 効果 | 実装コスト | 適用場面 |
|-----------|------|-----------|---------|
| インメモリキャッシュ | 高 | 低 | 頻繁にアクセスされるデータ |
| CDN | 高 | 低 | 静的コンテンツ |
| 非同期処理 | 中 | 中 | I/O待ちが多い処理 |
| DB最適化 | 高 | 高 | クエリが遅い場合 |
| コード最適化 | 低-中 | 高 | CPU律速の場合 |

---

## チーム開発での活用

### コードレビューのチェックリスト

このトピックに関連するコードレビューで確認すべきポイント:

- [ ] 命名規則が一貫しているか
- [ ] エラーハンドリングが適切か
- [ ] テストカバレッジは十分か
- [ ] パフォーマンスへの影響はないか
- [ ] セキュリティ上の問題はないか
- [ ] ドキュメントは更新されているか

### ナレッジ共有のベストプラクティス

| 方法 | 頻度 | 対象 | 効果 |
|------|------|------|------|
| ペアプログラミング | 随時 | 複雑なタスク | 即時のフィードバック |
| テックトーク | 週1回 | チーム全体 | 知識の水平展開 |
| ADR (設計記録) | 都度 | 将来のメンバー | 意思決定の透明性 |
| 振り返り | 2週間ごと | チーム全体 | 継続的改善 |
| モブプログラミング | 月1回 | 重要な設計 | 合意形成 |

### 技術的負債の管理

```
優先度マトリクス:

        影響度 高
          │
    ┌─────┼─────┐
    │ 計画 │ 即座 │
    │ 的に │ に   │
    │ 対応 │ 対応 │
    ├─────┼─────┤
    │ 記録 │ 次の │
    │ のみ │ Sprint│
    │     │ で   │
    └─────┼─────┘
          │
        影響度 低
    発生頻度 低  発生頻度 高
```
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

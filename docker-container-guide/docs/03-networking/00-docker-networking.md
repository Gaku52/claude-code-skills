# Dockerネットワーク

> コンテナ間通信とホスト・外部ネットワークとの接続を制御するDockerネットワーキングの全体像を理解する。

---

## この章で学ぶこと

1. **bridge / host / overlay の3大ネットワークドライバーの違いと使い分け**を理解する
2. **ポートマッピングと内蔵DNSによるサービスディスカバリ**の仕組みを習得する
3. **マルチホスト環境でのオーバーレイネットワーク**の構築手順を把握する

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

### ネットワークドライバー比較表

| ドライバー | スコープ | 用途 | IPアドレス | パフォーマンス |
|-----------|---------|------|-----------|--------------|
| bridge | 単一ホスト | デフォルト、開発環境 | 自動割当(172.17.x.x) | 良好 |
| host | 単一ホスト | 最大パフォーマンス | ホストと共有 | 最高 |
| overlay | マルチホスト | Swarm/本番クラスタ | 自動割当(10.0.x.x) | VXLANオーバーヘッド |
| macvlan | 単一ホスト | 物理NIC直結が必要 | 物理ネットワーク | 良好 |
| none | - | ネットワーク無効化 | なし | - |

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
```

### コード例2: Docker Composeでのネットワーク定義

```yaml
# docker-compose.yml
version: "3.9"

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
```

---

## 6. DNS とサービスディスカバリ

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

### コード例6: エイリアスとサービスディスカバリ

```yaml
# docker-compose.yml
version: "3.9"

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
```

---

## 7. 実践的なネットワーク構成例

### コード例7: マイクロサービス構成

```yaml
# docker-compose.yml - マイクロサービス構成
version: "3.9"

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

---

## 8. ネットワークのトラブルシューティング

### コード例8: デバッグコマンド集

```bash
# ネットワーク一覧を確認
docker network ls

# 特定ネットワークの詳細（接続コンテナ、IPAM設定）
docker network inspect my-app-network

# コンテナのネットワーク設定を確認
docker inspect --format='{{json .NetworkSettings.Networks}}' my-container | jq .

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

---

## まとめ

| 項目 | ポイント |
|------|---------|
| bridge | 単一ホストのデフォルトドライバー。ユーザー定義bridgeを推奨 |
| host | ポートマッピング不要で最高性能。Linuxのみフルサポート |
| overlay | マルチホスト通信。Swarm/Kubernetesで使用 |
| ポートマッピング | `-p host:container` でホストに公開。`127.0.0.1` バインド推奨 |
| DNS | ユーザー定義ネットワーク内で自動名前解決。127.0.0.11 |
| サービスディスカバリ | エイリアスでラウンドロビンDNS。Compose連携で簡便 |
| ネットワーク分離 | `internal: true` で外部遮断。層ごとに分離が鉄則 |

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
5. Linux Foundation "Container Networking From Scratch" -- https://www.youtube.com/watch?v=6v_BDHIgOY8

# ロードバランサー

> トラフィックを複数サーバーに分散させるロードバランサーの仕組みを理解し、L4/L7の違い、分散アルゴリズム、ヘルスチェック戦略を習得する。

## この章で学ぶこと

1. L4（トランスポート層）とL7（アプリケーション層）ロードバランサーの動作原理と使い分け
2. 主要な負荷分散アルゴリズム（ラウンドロビン、重み付き、最小接続、コンシステントハッシング）の実装と特性
3. ヘルスチェック（アクティブ/パッシブ）とフェイルオーバーの設計パターン
4. グローバル負荷分散とマルチリージョン構成の設計
5. gRPC/WebSocket などモダンプロトコルにおける負荷分散の注意点

---

## 前提知識

| トピック | 内容 | 参照ガイド |
|---------|------|-----------|
| TCP/IP基礎 | OSI参照モデル、TCP/UDPの違い | ネットワーク基礎 |
| HTTP/HTTPS | HTTPメソッド、ステータスコード、TLS | Web基礎 |
| スケーラビリティ | 水平・垂直スケーリングの概念 | [スケーラビリティ](../00-fundamentals/01-scalability.md) |
| 可用性 | 冗長化、SPOF排除の基本概念 | [信頼性](../00-fundamentals/02-reliability.md) |
| DNS | 名前解決の仕組み、レコードタイプ | ネットワーク基礎 |

---

## なぜロードバランサーを学ぶのか

ロードバランサーは現代の分散システムにおける**最も基本的なインフラコンポーネント**の一つである。あらゆる大規模Webサービスは、単一サーバーでは処理しきれないトラフィックに対応するためにロードバランサーを使用する。

**ビジネスインパクト:**
- **可用性**: サーバー障害時に自動的にトラフィックを正常なサーバーに迂回（ゼロダウンタイム）
- **スケーラビリティ**: バックエンドサーバーを追加するだけで処理能力を線形にスケール
- **レイテンシ**: 最も負荷の低いサーバーにルーティングすることで応答時間を均一化
- **セキュリティ**: バックエンドサーバーのIPアドレスを隠蔽し、DDoS攻撃の緩和に寄与

**具体例:**
- Googleは毎秒数十万リクエストをグローバルLBで処理
- Netflixは数千のマイクロサービスインスタンス間でトラフィックを分散
- AWSのELB (Elastic Load Balancing) は全AWSサービスの基盤インフラ

---

## 1. ロードバランサーとは

ロードバランサー（LB）は、クライアントからのリクエストを**複数のバックエンドサーバーに分散**させるコンポーネントである。目的は (1) スループットの向上、(2) 可用性の確保（1台故障しても継続）、(3) レイテンシの均一化、である。

### ASCII図解1: ロードバランサーの基本配置

```
  Clients
  ┌───┐ ┌───┐ ┌───┐ ┌───┐
  │ C1│ │ C2│ │ C3│ │ C4│
  └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘
    │      │      │      │
    └──────┴──┬───┴──────┘
              │
     ┌────────▼────────┐
     │  Load Balancer  │  ← 単一エントリーポイント
     │  (VIP: 1.2.3.4) │     Virtual IP で公開
     └──┬─────┬─────┬──┘
        │     │     │
   ┌────▼┐ ┌─▼───┐ ┌▼────┐
   │ Srv │ │ Srv │ │ Srv │  ← バックエンドプール
   │  1  │ │  2  │ │  3  │
   │:8080│ │:8080│ │:8080│
   └─────┘ └─────┘ └─────┘
```

### ロードバランサーの主要機能

```
┌──────────────────────────────────────────────┐
│           ロードバランサーの機能一覧            │
├──────────────────────────────────────────────┤
│ 1. トラフィック分散   - リクエストを複数サーバーに振り分け │
│ 2. ヘルスチェック     - 異常サーバーの自動検出・除外      │
│ 3. SSL/TLS終端       - 暗号化/復号の一元管理           │
│ 4. セッション永続化   - 同一クライアントの継続接続       │
│ 5. レート制限        - 過剰リクエストの制御             │
│ 6. コンテンツルーティング - URLパス/ヘッダーに基づく振り分け│
│ 7. DDoS緩和         - 不正トラフィックのフィルタリング   │
│ 8. 圧縮・キャッシュ   - レスポンスの最適化              │
└──────────────────────────────────────────────┘
```

---

## 2. L4 vs L7 ロードバランサー

### ASCII図解2: OSI参照モデルとLBの動作レイヤー

```
  OSI Layer    L4 LB         L7 LB
  ─────────────────────────────────────
  7 Application              ✓ HTTP/HTTPS
  6 Presentation             ✓ SSL終端
  5 Session                  ✓ Cookie
  4 Transport   ✓ TCP/UDP
  3 Network     ✓ IP
  2 Data Link
  1 Physical

  L4: IP + ポート番号のみで振り分け
      → 高速、低レイテンシ、プロトコル非依存

  L7: HTTPヘッダー、URL、Cookieを解析して振り分け
      → 柔軟、コンテンツベースルーティング、SSL終端
```

### L4 ロードバランサーの動作フロー

```
  Client (10.0.0.5:54321)
     │
     │ SYN パケット (DST: VIP:443)
     ▼
  ┌────────────────────┐
  │  L4 Load Balancer  │
  │  - IP/Port のみ参照 │
  │  - ペイロード未解析  │
  │  - NAT/DSR で転送   │
  └─────────┬──────────┘
            │
     ┌──────┼──────┐
     │      │      │
     ▼      ▼      ▼
   Srv1   Srv2   Srv3
   (TCP接続がそのまま確立)

  転送方式:
  ┌─────────────────────────────────────────┐
  │ NAT (Network Address Translation)       │
  │  - DST IP/Port を書き換えて転送         │
  │  - レスポンスもLB経由（ボトルネック注意）  │
  ├─────────────────────────────────────────┤
  │ DSR (Direct Server Return)              │
  │  - レスポンスはLBを経由せず直接クライアントへ│
  │  - 高スループット（動画配信等で有効）      │
  ├─────────────────────────────────────────┤
  │ IP Tunneling (IPIP)                     │
  │  - IPパケットをカプセル化してバックエンドに転送│
  │  - DSRの亜種、リモートDCへの転送に有効    │
  └─────────────────────────────────────────┘
```

### コード例1: L7ルーティングルール（Nginx風設定）

```nginx
# L7ロードバランサーの設定例

# バックエンドプールの定義
upstream api_servers {
    least_conn;                    # 最小接続数アルゴリズム
    server api1.internal:8080 weight=3;  # 高スペック
    server api2.internal:8080 weight=1;  # 低スペック
    server api3.internal:8080 weight=1;
    server api4.internal:8080 backup;    # バックアップ
}

upstream static_servers {
    server static1.internal:80;
    server static2.internal:80;
}

# gRPCバックエンドプール
upstream grpc_servers {
    least_conn;
    server grpc1.internal:50051;
    server grpc2.internal:50051;
}

server {
    listen 443 ssl http2;   # HTTP/2 有効化
    server_name example.com;

    # SSL終端（L7 LBの重要な役割）
    ssl_certificate /etc/ssl/cert.pem;
    ssl_certificate_key /etc/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # パスベースルーティング
    location /api/ {
        proxy_pass http://api_servers;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Host $host;

        # タイムアウト設定
        proxy_connect_timeout 5s;
        proxy_read_timeout 30s;
        proxy_send_timeout 10s;
    }

    location /static/ {
        proxy_pass http://static_servers;
        expires 30d;  # キャッシュヘッダー付与
        add_header Cache-Control "public, immutable";
    }

    # gRPCルーティング
    location /grpc/ {
        grpc_pass grpc://grpc_servers;
        grpc_set_header X-Real-IP $remote_addr;
    }

    # WebSocket対応
    location /ws/ {
        proxy_pass http://api_servers;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 3600s;  # WebSocket用の長いタイムアウト
    }

    # ヘッダーベースルーティング（カナリアリリース）
    location /api/v2/ {
        if ($http_x_canary = "true") {
            proxy_pass http://canary_servers;
        }
        proxy_pass http://api_servers;
    }
}
```

### コード例2: L4ロードバランサーの実装（簡易版）

```python
import socket
import threading
import itertools
import time
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Backend:
    """バックエンドサーバーの情報"""
    host: str
    port: int
    healthy: bool = True
    active_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    last_health_check: float = 0.0

    @property
    def address(self) -> tuple[str, int]:
        return (self.host, self.port)

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests


class L4LoadBalancer:
    """L4 (TCP) ロードバランサーの簡易実装

    機能:
    - ラウンドロビンによる負荷分散
    - パッシブヘルスチェック（エラー率監視）
    - コネクション数の追跡
    - グレースフルシャットダウン
    """

    def __init__(self, listen_port: int, backends: list[Backend],
                 max_connections: int = 1024):
        self.listen_port = listen_port
        self.backends = backends
        self.max_connections = max_connections
        self._backend_index = 0
        self._lock = threading.Lock()
        self._active = True
        self._connection_count = 0

    def get_next_backend(self) -> Optional[Backend]:
        """ラウンドロビンで次の正常なバックエンドを選択"""
        with self._lock:
            healthy_backends = [b for b in self.backends if b.healthy]
            if not healthy_backends:
                return None
            backend = healthy_backends[self._backend_index % len(healthy_backends)]
            self._backend_index += 1
            return backend

    def handle_connection(self, client_sock: socket.socket,
                         client_addr: tuple[str, int]):
        """クライアント接続をバックエンドに転送"""
        backend = self.get_next_backend()
        if backend is None:
            print(f"[ERROR] No healthy backends available for {client_addr}")
            client_sock.close()
            return

        backend.active_connections += 1
        backend.total_requests += 1
        print(f"[ROUTE] {client_addr} → {backend.host}:{backend.port} "
              f"(active: {backend.active_connections})")

        try:
            backend_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            backend_sock.settimeout(5.0)  # 接続タイムアウト
            backend_sock.connect(backend.address)

            # 双方向プロキシ（TCP透過転送）
            def forward(src, dst, label):
                try:
                    while self._active:
                        data = src.recv(4096)
                        if not data:
                            break
                        dst.sendall(data)
                except (socket.error, OSError):
                    pass
                finally:
                    try:
                        src.close()
                    except OSError:
                        pass
                    try:
                        dst.close()
                    except OSError:
                        pass

            t1 = threading.Thread(target=forward,
                                  args=(client_sock, backend_sock, "C→B"))
            t2 = threading.Thread(target=forward,
                                  args=(backend_sock, client_sock, "B→C"))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        except (socket.error, OSError) as e:
            backend.total_errors += 1
            print(f"[ERROR] Backend {backend.host}:{backend.port}: {e}")
            # パッシブヘルスチェック: エラー率が50%超で unhealthy
            if backend.error_rate > 0.5 and backend.total_requests >= 10:
                backend.healthy = False
                print(f"[HEALTH] {backend.host}:{backend.port} → UNHEALTHY "
                      f"(error_rate: {backend.error_rate:.1%})")
            client_sock.close()
        finally:
            backend.active_connections -= 1

    def start(self):
        """LBを起動してリクエストの待ち受けを開始"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", self.listen_port))
        server.listen(128)
        print(f"[START] L4 LB listening on :{self.listen_port}")
        print(f"[INFO] Backends: {[(b.host, b.port) for b in self.backends]}")

        try:
            while self._active:
                client_sock, addr = server.accept()
                if self._connection_count >= self.max_connections:
                    print(f"[REJECT] Max connections reached: {addr}")
                    client_sock.close()
                    continue
                self._connection_count += 1
                threading.Thread(
                    target=self._wrapped_handle,
                    args=(client_sock, addr),
                    daemon=True
                ).start()
        except KeyboardInterrupt:
            print("[SHUTDOWN] Graceful shutdown initiated...")
            self._active = False
        finally:
            server.close()

    def _wrapped_handle(self, client_sock, addr):
        try:
            self.handle_connection(client_sock, addr)
        finally:
            self._connection_count -= 1

    def get_stats(self) -> dict:
        """現在の統計情報を取得"""
        return {
            "active_connections": self._connection_count,
            "backends": [
                {
                    "address": f"{b.host}:{b.port}",
                    "healthy": b.healthy,
                    "active_connections": b.active_connections,
                    "total_requests": b.total_requests,
                    "error_rate": f"{b.error_rate:.1%}"
                }
                for b in self.backends
            ]
        }


# 実行例
if __name__ == "__main__":
    backends = [
        Backend("server1.internal", 8080),
        Backend("server2.internal", 8080),
        Backend("server3.internal", 8080),
    ]
    lb = L4LoadBalancer(listen_port=443, backends=backends)
    lb.start()
```

---

## 3. 負荷分散アルゴリズム

### ASCII図解3: 各アルゴリズムの動作イメージ

```
■ ラウンドロビン: 順番に1つずつ
  R1→S1, R2→S2, R3→S3, R4→S1, R5→S2, R6→S3 ...

■ 重み付きラウンドロビン: サーバー性能に応じた比率
  S1(weight=3): ●●●
  S2(weight=1): ●
  S3(weight=1): ●
  R1→S1, R2→S1, R3→S1, R4→S2, R5→S3, R6→S1 ...

■ 最小接続数: アクティブ接続が最も少ないサーバーへ
  S1: ■■■■ (4接続)
  S2: ■■ (2接続)    ← 次のリクエストはここへ
  S3: ■■■ (3接続)

■ IPハッシュ: 同一クライアントIPは常に同じサーバーへ
  hash(10.0.0.1) % 3 = 0 → S1
  hash(10.0.0.2) % 3 = 2 → S3
  hash(10.0.0.1) % 3 = 0 → S1 (同一)

■ コンシステントハッシュ: ハッシュリング上で最寄りノードへ
       S1
      / \
    /     \
  S3 ─── S2   key → リング上の時計回り最寄りサーバー
```

### コード例3: 主要アルゴリズムの実装

```python
import hashlib
import random
import time
from collections import defaultdict
from bisect import bisect_right
from dataclasses import dataclass
from typing import Optional

@dataclass
class Server:
    """バックエンドサーバー"""
    name: str
    weight: int = 1
    active_connections: int = 0
    response_time_ms: float = 0.0  # 平均応答時間

    def __repr__(self):
        return self.name


class LoadBalancerAlgorithms:
    """主要な負荷分散アルゴリズムの実装集"""

    def __init__(self, servers: list[Server]):
        self.servers = servers
        self.rr_index = 0

    # 1. ラウンドロビン (Round Robin)
    def round_robin(self) -> Server:
        """各サーバーに均等に振り分け。最もシンプル"""
        server = self.servers[self.rr_index % len(self.servers)]
        self.rr_index += 1
        return server

    # 2. 重み付きラウンドロビン (Weighted Round Robin)
    def weighted_round_robin(self) -> Server:
        """サーバーの性能差に応じた重み付き振り分け"""
        pool = []
        for server in self.servers:
            pool.extend([server] * server.weight)
        server = pool[self.rr_index % len(pool)]
        self.rr_index += 1
        return server

    # 3. 最小接続数 (Least Connections)
    def least_connections(self) -> Server:
        """アクティブ接続数が最も少ないサーバーを選択"""
        return min(self.servers, key=lambda s: s.active_connections)

    # 4. 重み付き最小接続数 (Weighted Least Connections)
    def weighted_least_connections(self) -> Server:
        """接続数 / weight が最小のサーバーを選択"""
        return min(self.servers,
                   key=lambda s: s.active_connections / max(s.weight, 1))

    # 5. IPハッシュ (IP Hash)
    def ip_hash(self, client_ip: str) -> Server:
        """同一IPは常に同一サーバーへ（セッション固定）"""
        hash_val = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return self.servers[hash_val % len(self.servers)]

    # 6. 最小応答時間 (Least Response Time)
    def least_response_time(self) -> Server:
        """平均応答時間が最短のサーバーを選択"""
        return min(self.servers, key=lambda s: s.response_time_ms)

    # 7. ランダム (Random)
    def random_select(self) -> Server:
        """ランダムに選択。大量サーバーでは統計的に均等に近づく"""
        return random.choice(self.servers)

    # 8. Power of Two Choices (P2C)
    def power_of_two_choices(self) -> Server:
        """ランダムに2台選び、接続数が少ない方を選択。
        純粋なランダムより均等で、全探索より低コスト。
        Nginx, Envoy で採用されている。"""
        if len(self.servers) < 2:
            return self.servers[0]
        s1, s2 = random.sample(self.servers, 2)
        return s1 if s1.active_connections <= s2.active_connections else s2


# === デモ実行 ===
servers = [
    Server("srv1", weight=3, active_connections=5, response_time_ms=10.0),
    Server("srv2", weight=1, active_connections=2, response_time_ms=25.0),
    Server("srv3", weight=1, active_connections=8, response_time_ms=15.0),
]
lb = LoadBalancerAlgorithms(servers)

print("=== ラウンドロビン ===")
for _ in range(6):
    print(f"  → {lb.round_robin()}")
# 出力:
#   → srv1
#   → srv2
#   → srv3
#   → srv1
#   → srv2
#   → srv3

print("\n=== 重み付きラウンドロビン (srv1:3, srv2:1, srv3:1) ===")
lb.rr_index = 0
for _ in range(5):
    print(f"  → {lb.weighted_round_robin()}")
# 出力:
#   → srv1
#   → srv1
#   → srv1
#   → srv2
#   → srv3

print("\n=== 最小接続数 ===")
print(f"  → {lb.least_connections()}")  # srv2 (2接続)

print("\n=== IPハッシュ ===")
for ip in ["10.0.0.1", "10.0.0.2", "10.0.0.1"]:
    print(f"  {ip} → {lb.ip_hash(ip)}")
# 10.0.0.1 → 同じサーバー（冪等）

print("\n=== Power of Two Choices ===")
random.seed(42)
for _ in range(5):
    print(f"  → {lb.power_of_two_choices()}")
```

### コード例4: コンシステントハッシング

```python
import hashlib
from bisect import bisect_right
from collections import Counter

class ConsistentHash:
    """
    コンシステントハッシング
    サーバー追加/削除時に再配置されるキーを最小化

    仮想ノード: 物理サーバー1台に対して複数のハッシュ値を割り当て、
    リング上の分布を均一化する。仮想ノード数が少ないと偏りが生じる。

    典型的な仮想ノード数:
    - 50: 標準偏差 ~10% → 開発/テスト環境
    - 150: 標準偏差 ~5% → 本番環境（推奨）
    - 500: 標準偏差 ~2% → 高精度が必要な場合
    """

    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring: list[int] = []           # ソート済みハッシュ値
        self.ring_map: dict[int, str] = {}  # ハッシュ値 → 実サーバー
        self.servers: set[str] = set()

    def _hash(self, key: str) -> int:
        """SHA-256ベースのハッシュ関数（MD5より均一な分布）"""
        return int(hashlib.sha256(key.encode()).hexdigest(), 16)

    def add_server(self, server: str):
        """サーバー追加: 仮想ノードをリングに配置"""
        self.servers.add(server)
        for i in range(self.virtual_nodes):
            vnode_key = f"{server}#vn{i}"
            h = self._hash(vnode_key)
            self.ring.append(h)
            self.ring_map[h] = server
        self.ring.sort()
        print(f"[ADD] {server} ({self.virtual_nodes} vnodes, "
              f"ring size: {len(self.ring)})")

    def remove_server(self, server: str):
        """サーバー削除: 仮想ノードをリングから除去"""
        self.servers.discard(server)
        self.ring = [h for h in self.ring
                     if self.ring_map.get(h) != server]
        self.ring_map = {h: s for h, s in self.ring_map.items()
                         if s != server}
        print(f"[REMOVE] {server} (ring size: {len(self.ring)})")

    def get_server(self, key: str) -> str:
        """キーに対応するサーバーを取得（時計回りの最寄り）"""
        if not self.ring:
            raise Exception("No servers available")
        h = self._hash(key)
        idx = bisect_right(self.ring, h)
        if idx == len(self.ring):
            idx = 0  # リングの末端を超えたら先頭に戻る
        return self.ring_map[self.ring[idx]]

    def get_replicas(self, key: str, count: int = 3) -> list[str]:
        """レプリカ配置: 時計回りに異なるサーバーをN台取得"""
        if not self.ring:
            raise Exception("No servers available")
        replicas = []
        h = self._hash(key)
        idx = bisect_right(self.ring, h)

        seen = set()
        for i in range(len(self.ring)):
            pos = (idx + i) % len(self.ring)
            server = self.ring_map[self.ring[pos]]
            if server not in seen:
                seen.add(server)
                replicas.append(server)
                if len(replicas) == count:
                    break
        return replicas


# === デモ: サーバー追加/削除時の影響を検証 ===
ch = ConsistentHash(virtual_nodes=100)
for s in ["server-A", "server-B", "server-C"]:
    ch.add_server(s)

# 1000キーの分布を確認
dist = Counter(ch.get_server(f"key-{i}") for i in range(1000))
print(f"\n分布: {dict(dist)}")
# 期待値: 各サーバー約333キー（±10%程度の偏差）

# サーバー追加時の影響（再配置率）
before = {f"key-{i}": ch.get_server(f"key-{i}") for i in range(1000)}
ch.add_server("server-D")
after = {f"key-{i}": ch.get_server(f"key-{i}") for i in range(1000)}
moved = sum(1 for k in before if before[k] != after[k])
print(f"再配置されたキー: {moved}/1000 ({moved/10:.1f}%)")
# 理論値: 1/4 = 25% (1000キー中約250キーが移動)

# レプリカ配置
print(f"\nレプリカ配置 (key-42): {ch.get_replicas('key-42', 3)}")
```

---

## 4. ヘルスチェック

### ASCII図解4: ヘルスチェックの種類

```
■ アクティブヘルスチェック（LB → Backend）

  LB ──GET /health──→ Backend 1  → 200 OK  ✓ healthy
  LB ──GET /health──→ Backend 2  → 200 OK  ✓ healthy
  LB ──GET /health──→ Backend 3  → timeout ✗ unhealthy
                                      │
                        3回連続失敗 → プールから除外
                        2回連続成功 → プールに復帰

■ パッシブヘルスチェック（実トラフィックで判定）

  Client ──req──→ LB ──→ Backend 3 → 502 Error
                   │      (エラー率 > 50%)
                   │           │
                   │     自動的にプールから除外
                   └──→ Backend 1 → 200 OK (リトライ)

■ ディープヘルスチェック（依存関係含む）

  LB ──GET /health/deep──→ Backend
                              │
                    ┌─────────┼──────────┐
                    ▼         ▼          ▼
                  [DB OK]  [Redis OK]  [Kafka OK]
                              │
                    全て OK → 200 {"status": "healthy"}
                    一部NG → 503 {"status": "degraded",
                                  "redis": "unhealthy"}
```

### コード例5: ヘルスチェック実装

```python
import asyncio
import aiohttp
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

class ServerState(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"    # 新規接続拒否、既存接続は完了待ち
    DEGRADED = "degraded"    # 一部機能制限で稼働

@dataclass
class BackendServer:
    host: str
    port: int
    state: ServerState = ServerState.HEALTHY
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    active_connections: int = 0
    last_check_time: float = 0.0
    last_response_time_ms: float = 0.0
    total_requests: int = 0
    total_errors: int = 0

    @property
    def url(self):
        return f"http://{self.host}:{self.port}"

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests


class HealthCheckManager:
    """アクティブ + パッシブヘルスチェックの統合管理

    設計ポイント:
    1. アクティブチェック: 定期的にヘルスエンドポイントを叩く
    2. パッシブチェック: 実トラフィックのレスポンスコードで判定
    3. ディープチェック: DB/Redis等の依存関係も含めた包括的チェック
    4. 状態遷移: hysteresis（ヒステリシス）で頻繁な状態変更を防止
    """

    def __init__(self, servers: list[BackendServer],
                 check_interval: float = 10.0,
                 unhealthy_threshold: int = 3,
                 healthy_threshold: int = 2,
                 timeout: float = 3.0,
                 on_state_change: Optional[Callable] = None):
        self.servers = servers
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.healthy_threshold = healthy_threshold
        self.timeout = timeout
        self.on_state_change = on_state_change

    async def active_check(self, server: BackendServer,
                           session: aiohttp.ClientSession):
        """アクティブヘルスチェック"""
        start_time = time.time()
        try:
            async with session.get(
                f"{server.url}/health",
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                elapsed = (time.time() - start_time) * 1000
                server.last_response_time_ms = elapsed
                server.last_check_time = time.time()

                if resp.status == 200:
                    body = await resp.json()
                    # ディープチェック: 依存関係のステータスも確認
                    if body.get("status") == "degraded":
                        self._transition(server, ServerState.DEGRADED,
                                        "Degraded dependencies")
                    else:
                        self._mark_success(server)
                elif resp.status == 503:
                    self._mark_failure(server, f"HTTP {resp.status}")
                else:
                    self._mark_failure(server, f"HTTP {resp.status}")
        except asyncio.TimeoutError:
            self._mark_failure(server, "Timeout")
        except Exception as e:
            self._mark_failure(server, str(e))

    def passive_check(self, server: BackendServer, status_code: int,
                     response_time_ms: float = 0.0):
        """パッシブヘルスチェック（実リクエスト結果から判定）"""
        server.total_requests += 1
        server.last_response_time_ms = response_time_ms

        if status_code >= 500:
            server.total_errors += 1
            self._mark_failure(server, f"HTTP {status_code}")
        else:
            self._mark_success(server)

    def _mark_success(self, server: BackendServer):
        server.consecutive_failures = 0
        server.consecutive_successes += 1
        if (server.state in (ServerState.UNHEALTHY, ServerState.DEGRADED)
                and server.consecutive_successes >= self.healthy_threshold):
            self._transition(server, ServerState.HEALTHY, "Recovered")

    def _mark_failure(self, server: BackendServer, reason: str):
        server.consecutive_successes = 0
        server.consecutive_failures += 1
        if (server.state == ServerState.HEALTHY
                and server.consecutive_failures >= self.unhealthy_threshold):
            self._transition(server, ServerState.UNHEALTHY, reason)

    def _transition(self, server: BackendServer, new_state: ServerState,
                    reason: str):
        """状態遷移とコールバック通知"""
        old_state = server.state
        if old_state == new_state:
            return
        server.state = new_state
        msg = (f"[HEALTH] {server.url}: {old_state.value} → "
               f"{new_state.value} ({reason})")
        print(msg)
        if self.on_state_change:
            self.on_state_change(server, old_state, new_state, reason)

    def get_healthy_servers(self) -> list[BackendServer]:
        """正常 or 縮退状態のサーバーを返す"""
        return [s for s in self.servers
                if s.state in (ServerState.HEALTHY, ServerState.DEGRADED)]

    def get_stats(self) -> list[dict]:
        """全サーバーの統計情報"""
        return [
            {
                "url": s.url,
                "state": s.state.value,
                "active_connections": s.active_connections,
                "error_rate": f"{s.error_rate:.1%}",
                "last_response_ms": f"{s.last_response_time_ms:.1f}",
                "consecutive_failures": s.consecutive_failures,
            }
            for s in self.servers
        ]

    async def run_periodic_checks(self):
        """定期的なアクティブヘルスチェックを実行"""
        async with aiohttp.ClientSession() as session:
            while True:
                tasks = [self.active_check(server, session)
                         for server in self.servers]
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.sleep(self.check_interval)
```

---

## 5. グローバル負荷分散

### ASCII図解5: マルチリージョン構成

```
  ユーザー (東京)              ユーザー (ロンドン)
      │                            │
      ▼                            ▼
  ┌──────────────────────────────────────┐
  │         GeoDNS / Route 53           │
  │   東京ユーザー → ap-northeast-1     │
  │   欧州ユーザー → eu-west-1          │
  └──────────┬───────────────┬──────────┘
             │               │
    ┌────────▼───────┐ ┌────▼──────────┐
    │ ap-northeast-1 │ │  eu-west-1    │
    │ ┌────────────┐ │ │ ┌──────────┐  │
    │ │ L7 LB (ALB)│ │ │ │L7 LB(ALB│  │
    │ └─────┬──────┘ │ │ └────┬─────┘  │
    │   ┌───┼───┐    │ │  ┌──┼───┐    │
    │   │   │   │    │ │  │  │   │    │
    │  AZ-a AZ-c AZ-d│ │ AZ-a AZ-b AZ-c│
    │  │   │   │     │ │  │  │   │    │
    │  └───┼───┘     │ │  └──┼───┘    │
    │  ┌───▼───┐     │ │ ┌──▼────┐   │
    │  │L4(NLB)│     │ │ │L4(NLB)│   │
    │  └───────┘     │ │ └───────┘   │
    └────────────────┘ └──────────────┘

  3層構成:
    Layer 1: GeoDNS (地理ベースルーティング)
    Layer 2: L7 LB (コンテンツベースルーティング)
    Layer 3: L4 LB (AZ間の高速転送)
```

### コード例6: グローバルLB設定（AWS CDK）

```python
# AWS CDKによるグローバルLB構成
from aws_cdk import (
    Stack, Duration,
    aws_elasticloadbalancingv2 as elbv2,
    aws_ec2 as ec2,
    aws_route53 as route53,
    aws_route53_targets as targets,
    aws_globalaccelerator as ga,
)
from constructs import Construct


class GlobalLoadBalancerStack(Stack):
    """グローバルLB構成のCDKスタック

    構成:
    1. AWS Global Accelerator (エニーキャスト)
    2. 各リージョンにALB (L7 LB)
    3. Auto Scaling Groupでバックエンド管理
    4. Route 53ヘルスチェックでリージョンフェイルオーバー
    """

    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # 1. VPC とサブネット
        vpc = ec2.Vpc(self, "VPC",
            max_azs=3,
            nat_gateways=1,
        )

        # 2. Application Load Balancer (L7)
        alb = elbv2.ApplicationLoadBalancer(self, "ALB",
            vpc=vpc,
            internet_facing=True,
            load_balancer_name="api-alb",
        )

        # 3. ターゲットグループ（ヘルスチェック付き）
        target_group = elbv2.ApplicationTargetGroup(self, "TG",
            vpc=vpc,
            port=8080,
            protocol=elbv2.ApplicationProtocol.HTTP,
            target_type=elbv2.TargetType.INSTANCE,
            health_check=elbv2.HealthCheck(
                path="/health",
                interval=Duration.seconds(10),
                timeout=Duration.seconds(5),
                healthy_threshold_count=2,
                unhealthy_threshold_count=3,
                healthy_http_codes="200",
            ),
            deregistration_delay=Duration.seconds(30),
        )

        # 4. リスナー設定（HTTPS + パスベースルーティング）
        https_listener = alb.add_listener("HTTPS",
            port=443,
            certificates=[certificate],
            default_action=elbv2.ListenerAction.forward([target_group]),
        )

        # パスベースルーティング
        https_listener.add_action("ApiV2",
            priority=10,
            conditions=[
                elbv2.ListenerCondition.path_patterns(["/api/v2/*"]),
            ],
            action=elbv2.ListenerAction.forward([v2_target_group]),
        )

        # カナリアルーティング (10%のトラフィックを新バージョンへ)
        https_listener.add_action("Canary",
            priority=20,
            conditions=[
                elbv2.ListenerCondition.path_patterns(["/api/*"]),
                elbv2.ListenerCondition.http_header("X-Canary", ["true"]),
            ],
            action=elbv2.ListenerAction.forward(
                target_groups=[target_group, canary_target_group],
                target_group_stickiness_duration=Duration.hours(1),
            ),
        )

        # 5. Global Accelerator
        accelerator = ga.Accelerator(self, "Accelerator",
            accelerator_name="global-api",
        )
        listener = accelerator.add_listener("Listener",
            port_ranges=[ga.PortRange(from_port=443, to_port=443)],
        )
        listener.add_endpoint_group("Region",
            endpoints=[ga.ApplicationLoadBalancerEndpoint(alb)],
        )
```

---

## 6. 比較表

### 比較表1: L4 vs L7 ロードバランサー

| 項目 | L4 (トランスポート層) | L7 (アプリケーション層) |
|------|---------------------|----------------------|
| 解析対象 | IP + ポート | HTTP/HTTPS/gRPC ヘッダー |
| SSL終端 | 不可（パススルー） | 可能 |
| コンテンツルーティング | 不可 | URL、ヘッダー、Cookie で振り分け |
| パフォーマンス | 高い（低オーバーヘッド） | 低い（ヘッダー解析コスト） |
| WebSocket | パススルーで対応 | ネイティブ対応 |
| gRPC | コネクション単位（偏りあり） | ストリーム単位（均等分散） |
| セキュリティ | 限定的（IPベース） | WAF、レート制限、認証 |
| 適するケース | TCP/UDP汎用、DB接続、高RPS | HTTP API、Webアプリ、マイクロサービス |
| 製品例 | AWS NLB, HAProxy (TCP) | AWS ALB, Nginx, Envoy, Traefik |

### 比較表2: 負荷分散アルゴリズムの比較

| アルゴリズム | 均一性 | セッション維持 | 実装複雑度 | CPU負荷 | 適するケース |
|-------------|--------|--------------|-----------|---------|-------------|
| ラウンドロビン | 高い | なし | 最低 | 最低 | 同性能サーバー群 |
| 重み付きRR | 高い | なし | 低 | 低 | 異性能サーバー混在 |
| 最小接続数 | 高い | なし | 中 | 低 | 処理時間が不均一 |
| 重み付き最小接続 | 最高 | なし | 中 | 低 | 異性能 + 不均一処理 |
| IPハッシュ | 中程度 | あり | 低 | 低 | セッション固定が必要 |
| コンシステントハッシュ | 高い | あり | 高 | 中 | キャッシュ、スケール頻出 |
| 最小応答時間 | 高い | なし | 中 | 中 | レイテンシ重視 |
| P2C | 高い | なし | 低 | 低 | 大規模クラスタ (Envoy) |
| ランダム | 中程度 | なし | 最低 | 最低 | 大量サーバー |

### 比較表3: 主要ロードバランサー製品の比較

| 製品 | タイプ | L4 | L7 | スループット | 運用モデル | 特徴 |
|------|--------|:---:|:---:|------------|-----------|------|
| AWS ALB | マネージド | - | ✓ | 自動スケール | フルマネージド | HTTPルーティング、Lambda統合 |
| AWS NLB | マネージド | ✓ | - | 数百万RPS | フルマネージド | 超低レイテンシ、静的IP |
| Nginx | ソフトウェア | ✓ | ✓ | ~10万RPS/台 | セルフホスト | 設定柔軟、実績豊富 |
| HAProxy | ソフトウェア | ✓ | ✓ | ~20万RPS/台 | セルフホスト | 高性能、詳細メトリクス |
| Envoy | サイドカー | ✓ | ✓ | ~5万RPS/台 | Kubernetes | サービスメッシュ、gRPC対応 |
| Traefik | ソフトウェア | ✓ | ✓ | ~5万RPS/台 | Kubernetes | 自動ディスカバリ、Let's Encrypt |
| Cloudflare | CDN/LB | ✓ | ✓ | 無制限 | フルマネージド | グローバル、DDoS防御 |

---

## 7. アンチパターン

### アンチパターン1: ロードバランサー自体がSPOF

```python
# NG: 単一LBで全トラフィックを処理

class SinglePointLB:
    """単一のLBインスタンス → SPOFになる"""

    def __init__(self, backends: list[str]):
        self.backends = backends
        # LB自体が1台 → LBがダウンすると全サービス停止

    def route(self, request):
        # このインスタンスが死んだら全リクエスト失敗
        backend = self.select_backend()
        return forward(request, backend)


# OK: Active-Standby構成でLB自体を冗長化

class RedundantLBCluster:
    """VRRP/keepalived による Active-Standby 冗長化"""

    def __init__(self, backends: list[str],
                 vip: str = "10.0.0.100",
                 priority: int = 100):
        self.backends = backends
        self.vip = vip          # 仮想IP（フローティングIP）
        self.priority = priority # 優先度（高い方がActive）
        self.is_active = False

    def configure_vrrp(self) -> dict:
        """VRRP (Virtual Router Redundancy Protocol) 設定"""
        return {
            "vrrp_instance": {
                "virtual_router_id": 51,
                "priority": self.priority,     # Active: 200, Standby: 100
                "virtual_ipaddress": [self.vip],
                "advert_int": 1,               # 1秒ごとのハートビート
                "authentication": {
                    "auth_type": "PASS",
                    "auth_pass": "secret123",
                },
                # Active障害検知 → Standbyが自動昇格
                "track_script": {
                    "check_lb": {
                        "script": "/usr/local/bin/check_lb_health.sh",
                        "interval": 2,
                        "weight": -50,  # 失敗時にpriority-50で降格
                    }
                }
            }
        }
        # Active LB がダウン → VIP が Standby に移動
        # フェイルオーバー時間: 通常1-3秒

    def failover_to_standby(self):
        """フェイルオーバーシミュレーション"""
        print(f"[FAILOVER] Active LB down → VIP {self.vip} moving to Standby")
        self.is_active = False
        # Standby側の priority が相対的に高くなり VIP を取得
```

### アンチパターン2: スティッキーセッションへの過度な依存

```python
# NG: 全リクエストをCookieで同一サーバーに固定

class StickySessionLB:
    """スティッキーセッション: 全てのリクエストを同一サーバーへ"""

    def __init__(self, backends: list[str]):
        self.backends = backends
        self.session_map: dict[str, str] = {}  # session_id → backend

    def route(self, request):
        session_id = request.cookies.get("SESSION_ID")
        if session_id in self.session_map:
            backend = self.session_map[session_id]
            # 問題1: バックエンドが落ちたらセッション消失
            # 問題2: 人気ユーザーのサーバーに負荷集中
            # 問題3: スケールアウト時に既存セッション移動不可
            return forward(request, backend)
        else:
            backend = self.select_backend()
            self.session_map[session_id] = backend
            return forward(request, backend)


# OK: セッションを外部ストアに保存してステートレス化

import redis
import json

class StatelessLB:
    """ステートレスLB + 外部セッションストア"""

    def __init__(self, backends: list[str],
                 session_store: redis.Redis):
        self.backends = backends
        self.session_store = session_store
        self.rr_index = 0

    def route(self, request):
        # セッションはRedisに保存 → どのサーバーでも読める
        session_id = request.cookies.get("SESSION_ID")
        if session_id:
            session_data = self.session_store.get(f"session:{session_id}")
            request.session = json.loads(session_data) if session_data else {}

        # 任意のバックエンドにルーティング可能
        backend = self.backends[self.rr_index % len(self.backends)]
        self.rr_index += 1
        return forward(request, backend)

    # メリット:
    # - サーバー障害時もセッション維持
    # - 完全に均等な負荷分散
    # - スケールアウトが自由自在
    # - JWTトークンによるステートレス認証も選択肢
```

### アンチパターン3: ヘルスチェックなしのバックエンド追加

```python
# NG: ヘルスチェックなしでバックエンドプールに追加

class UnsafeLB:
    """ヘルスチェックなし → 異常サーバーにもトラフィック送信"""

    def __init__(self, backends: list[str]):
        self.backends = backends

    def add_backend(self, backend: str):
        # 問題: サーバーが起動途中でもトラフィック送信
        # 問題: DB接続プール初期化前にリクエストが到着
        self.backends.append(backend)

    def route(self, request):
        backend = random.choice(self.backends)
        # 異常サーバーへのルーティングでエラー多発
        return forward(request, backend)


# OK: ウォームアップ期間とヘルスチェック付き追加

from enum import Enum

class BackendState(Enum):
    WARMING = "warming"    # 起動中（ヘルスチェック待ち）
    ACTIVE = "active"      # 正常（トラフィック受信可能）
    DRAINING = "draining"  # 排出中（新規接続拒否）
    REMOVED = "removed"    # 除外済み

class SafeLB:
    """ヘルスチェック付きの安全なバックエンド管理"""

    def __init__(self, backends: list[str],
                 warmup_health_checks: int = 3,
                 drain_timeout_sec: int = 30):
        self.backends = {b: BackendState.ACTIVE for b in backends}
        self.warmup_health_checks = warmup_health_checks
        self.drain_timeout_sec = drain_timeout_sec
        self._health_counts: dict[str, int] = {}

    def add_backend(self, backend: str):
        """ウォームアップ期間を経てからトラフィック送信"""
        self.backends[backend] = BackendState.WARMING
        self._health_counts[backend] = 0
        print(f"[ADD] {backend} → WARMING "
              f"(need {self.warmup_health_checks} health checks)")

    def on_health_check_pass(self, backend: str):
        """ヘルスチェック成功時のコールバック"""
        if self.backends.get(backend) == BackendState.WARMING:
            self._health_counts[backend] = \
                self._health_counts.get(backend, 0) + 1
            if self._health_counts[backend] >= self.warmup_health_checks:
                self.backends[backend] = BackendState.ACTIVE
                print(f"[ACTIVE] {backend} → トラフィック受信開始")

    def remove_backend(self, backend: str):
        """グレースフルに排出してから除外"""
        self.backends[backend] = BackendState.DRAINING
        print(f"[DRAIN] {backend} → 新規接続拒否, "
              f"{self.drain_timeout_sec}秒後に除外")
        # 既存接続の完了を待ってからREMOVEDに遷移

    def get_active_backends(self) -> list[str]:
        return [b for b, s in self.backends.items()
                if s == BackendState.ACTIVE]
```

---

## 8. 練習問題

### 演習1（基礎）: 重み付きラウンドロビンの分布検証

**課題**: 4台のサーバーにそれぞれ weight 5, 3, 1, 1 を設定し、10,000リクエストを分配した場合の理論値と実測値を比較せよ。

```python
# ヒント: LoadBalancerAlgorithms クラスを使用
from collections import Counter

servers = [
    Server("srv-A", weight=5),
    Server("srv-B", weight=3),
    Server("srv-C", weight=1),
    Server("srv-D", weight=1),
]
lb = LoadBalancerAlgorithms(servers)

results = Counter()
for _ in range(10000):
    server = lb.weighted_round_robin()
    results[server.name] += 1

print("=== 実測分布 ===")
for name, count in sorted(results.items()):
    print(f"  {name}: {count} ({count/100:.1f}%)")

# 理論値: srv-A=50%, srv-B=30%, srv-C=10%, srv-D=10%
```

**期待される出力**:
```
=== 実測分布 ===
  srv-A: 5000 (50.0%)
  srv-B: 3000 (30.0%)
  srv-C: 1000 (10.0%)
  srv-D: 1000 (10.0%)
```

### 演習2（応用）: コンシステントハッシングのノード追加/削除シミュレーション

**課題**: 5台のサーバーで運用中に1台追加・1台削除した場合のキー再配置率を計測し、仮想ノード数（50, 150, 500）ごとの分布偏差を比較せよ。

```python
# ヒント: ConsistentHash クラスを使用
import statistics

for vnode_count in [50, 150, 500]:
    ch = ConsistentHash(virtual_nodes=vnode_count)
    for i in range(5):
        ch.add_server(f"server-{i}")

    # 10000キーの分布を計測
    dist = Counter(ch.get_server(f"key-{i}") for i in range(10000))
    values = list(dist.values())
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)

    # サーバー追加時の再配置率
    before = {f"key-{i}": ch.get_server(f"key-{i}") for i in range(10000)}
    ch.add_server("server-5")
    after = {f"key-{i}": ch.get_server(f"key-{i}") for i in range(10000)}
    moved = sum(1 for k in before if before[k] != after[k])

    print(f"\n仮想ノード数: {vnode_count}")
    print(f"  分布 (平均: {mean:.0f}, 標準偏差: {stdev:.0f}, "
          f"CV: {stdev/mean*100:.1f}%)")
    print(f"  再配置率: {moved/100:.1f}% (理論値: {100/6:.1f}%)")
```

**期待される出力（概算）**:
```
仮想ノード数: 50
  分布 (平均: 2000, 標準偏差: 200, CV: 10.0%)
  再配置率: 17.5% (理論値: 16.7%)

仮想ノード数: 150
  分布 (平均: 2000, 標準偏差: 100, CV: 5.0%)
  再配置率: 16.9% (理論値: 16.7%)

仮想ノード数: 500
  分布 (平均: 2000, 標準偏差: 40, CV: 2.0%)
  再配置率: 16.7% (理論値: 16.7%)
```

### 演習3（発展）: L7ロードバランサーの完全実装

**課題**: 以下の機能を持つL7ロードバランサーを実装せよ。
1. パスベースルーティング（`/api/*` と `/static/*` で異なるバックエンドプール）
2. ヘルスチェック（10秒間隔、3回連続失敗で除外）
3. リクエストレート制限（クライアントIPごとに100req/秒）
4. リクエスト/レスポンスのログ出力

```python
# ヒント: aiohttp + asyncio で実装
import asyncio
import aiohttp
from aiohttp import web
from collections import defaultdict
import time

class L7LoadBalancer:
    """L7 ロードバランサーの発展的実装"""

    def __init__(self):
        self.route_table: dict[str, list[BackendServer]] = {}
        self.health_manager: Optional[HealthCheckManager] = None
        self.rate_limits: dict[str, list[float]] = defaultdict(list)
        self.rate_limit_rps = 100

    def add_route(self, prefix: str, backends: list[BackendServer]):
        """パスプレフィックスに対するバックエンドプールを登録"""
        self.route_table[prefix] = backends

    def check_rate_limit(self, client_ip: str) -> bool:
        """トークンバケット方式のレート制限"""
        now = time.time()
        # 過去1秒のリクエストタイムスタンプを保持
        self.rate_limits[client_ip] = [
            t for t in self.rate_limits[client_ip]
            if now - t < 1.0
        ]
        if len(self.rate_limits[client_ip]) >= self.rate_limit_rps:
            return False
        self.rate_limits[client_ip].append(now)
        return True

    def resolve_backend(self, path: str) -> Optional[BackendServer]:
        """パスに基づいてバックエンドを選択"""
        for prefix, backends in self.route_table.items():
            if path.startswith(prefix):
                healthy = [b for b in backends
                          if b.state == ServerState.HEALTHY]
                if healthy:
                    # 最小接続数で選択
                    return min(healthy,
                              key=lambda b: b.active_connections)
        return None

    async def handle_request(self, request: web.Request) -> web.Response:
        """リクエストハンドラ"""
        client_ip = request.remote
        start_time = time.time()

        # レート制限チェック
        if not self.check_rate_limit(client_ip):
            return web.json_response(
                {"error": "Rate limit exceeded"},
                status=429
            )

        # バックエンド解決
        backend = self.resolve_backend(request.path)
        if backend is None:
            return web.json_response(
                {"error": "No backend available"},
                status=503
            )

        # リクエスト転送
        backend.active_connections += 1
        try:
            async with aiohttp.ClientSession() as session:
                target_url = f"{backend.url}{request.path}"
                async with session.request(
                    method=request.method,
                    url=target_url,
                    headers=dict(request.headers),
                    data=await request.read(),
                ) as resp:
                    body = await resp.read()
                    elapsed = (time.time() - start_time) * 1000
                    print(f"[{request.method}] {request.path} → "
                          f"{backend.url} {resp.status} "
                          f"{elapsed:.1f}ms")
                    return web.Response(
                        body=body,
                        status=resp.status,
                        headers=dict(resp.headers),
                    )
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            print(f"[ERROR] {request.path} → {backend.url}: {e} "
                  f"{elapsed:.1f}ms")
            return web.json_response(
                {"error": "Backend error"},
                status=502
            )
        finally:
            backend.active_connections -= 1
```

---

## 9. FAQ

### Q1: グローバルサービスではどのようなLB構成にしますか？

3層構成が一般的: (1) DNS ベースの地理的分散（GeoDNS/Route 53）、(2) リージョンごとの L7 LB（ALB/Nginx）、(3) AZ ごとの L4 LB（NLB）。ユーザーに最も近いリージョンにルーティングし、リージョン内ではAZ間で分散する。CloudflareやAWS Global Acceleratorはエニーキャストを使い、BGPレベルで最寄りのPoP（Point of Presence）にルーティングする。AWS Global Acceleratorの場合、静的IPアドレスが2つ割り当てられ、世界中のエッジロケーションからユーザーに最寄りのリージョンへ低レイテンシで接続される。

### Q2: LBのスループット限界はどの程度ですか？

ソフトウェアLB（Nginx）は1台あたり数万〜数十万RPS、ハードウェアLB（F5 BIG-IP）は数百万RPSを処理できる。AWS ALBは自動スケーリングで理論上無制限だが、急激なトラフィック増にはウォームアップが必要（事前にAWSサポートに連絡してpre-warmingを依頼する）。NLBは数百万同時接続を処理可能で、レイテンシも数マイクロ秒レベル。ボトルネックになる場合はDNSラウンドロビンで複数LBに分散する。

### Q3: gRPCの負荷分散はHTTPと何が違いますか？

gRPCはHTTP/2を使うため、1つのTCPコネクション上に複数ストリームが多重化される。L4 LBだとコネクション単位の振り分けになり、1コネクションに全リクエストが流れるため偏る。gRPCにはL7 LB（Envoy、Linkerd）を使い、ストリーム単位で振り分ける必要がある。クライアントサイドLB（gRPC built-in の `round_robin` ポリシー）も選択肢となる。Kubernetes環境ではEnvoyベースのサービスメッシュ（Istio）がgRPC負荷分散のデファクトスタンダードとなっている。

### Q4: ALBとNLBの使い分けはどうすべきですか？

AWS ALB（L7）は HTTPリクエストの内容に基づくルーティング（パスベース、ホストベース、ヘッダーベース）が必要な場合に使う。REST API、WebSocket、gRPCに適する。AWS NLB（L4）は 超低レイテンシ（数マイクロ秒）と高スループット（数百万RPS）が求められる場合に使う。TCP/UDPの汎用負荷分散、静的IPアドレスが必要な場合、DBプロキシとして使う場合に適する。両方を組み合わせることも可能で、NLBの背後にALBを配置する構成はmTLS（相互TLS認証）が必要な場合に使われる。

### Q5: ロードバランサーのウォームアップとは何ですか？

AWS ALBなどのマネージドLBは、トラフィック量に応じて内部的にスケーリングする。急激なトラフィック増（例: セール開始時に0→100万RPS）では、LB自体のスケーリングが間に合わず503エラーが発生する可能性がある。これを防ぐために、(1) 事前にAWSサポートに連絡してpre-warmingを依頼する、(2) 段階的にトラフィックを増やす（ランプアップ）、(3) NLBを使う（NLBは静的にスケール済み）、のいずれかで対応する。

### Q6: サービスメッシュとLBの関係は？

サービスメッシュ（Istio/Envoy、Linkerd）は、マイクロサービス環境におけるサービス間通信の全てにサイドカープロキシとしてLB機能を提供する。各Podにサイドカーが注入され、サービスディスカバリ、負荷分散、リトライ、サーキットブレーカー、mTLS、メトリクス収集を透過的に行う。従来のLBが「入口」で一元管理するのに対し、サービスメッシュは「各ノード」で分散管理する。10サービス以下では従来のLBで十分だが、20+サービスでは可観測性とセキュリティの統一管理のためにサービスメッシュの導入を検討する。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| LBの役割 | トラフィック分散、可用性確保、レイテンシ均一化、セキュリティ |
| L4 vs L7 | L4は高速・汎用（TCP/UDP）、L7は柔軟・HTTP特化（ルーティング、SSL終端） |
| 主要アルゴリズム | RR、重み付き、最小接続、P2C、コンシステントハッシュ |
| ヘルスチェック | アクティブ（定期確認）+ パッシブ（実トラフィック判定）の併用が必須 |
| LB自体の冗長化 | Active-Standby (VRRP)、DNS分散で SPOF を排除 |
| グローバル構成 | GeoDNS → L7 LB → L4 LB の3層構成 |
| gRPC対応 | L7 LB（Envoy等）でストリーム単位の振り分けが必要 |
| サービスメッシュ | 20+サービスの大規模環境ではEnvoy/Istioベースの分散LBを検討 |

---

## 次に読むべきガイド

- [キャッシュ](./01-caching.md) -- LBの背後に配置するキャッシュレイヤー
- [CDN](./03-cdn.md) -- グローバルな静的コンテンツ配信とエッジLB
- [スケーラビリティ](../00-fundamentals/01-scalability.md) -- LBを活用した水平スケーリング
- [信頼性](../00-fundamentals/02-reliability.md) -- サーキットブレーカーやリトライとLBの連携
- [メッセージキュー](./02-message-queue.md) -- 非同期処理によるバックエンド負荷の平準化

---

## 参考文献

1. Karger, D. et al. (1997). "Consistent Hashing and Random Trees." *STOC '97*.
2. Mitzenmacher, M. (2001). "The Power of Two Choices in Randomized Load Balancing." *IEEE Transactions on Parallel and Distributed Systems*.
3. Nginx Documentation -- https://nginx.org/en/docs/http/load_balancing.html
4. Envoy Proxy Documentation -- https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/load_balancing/
5. AWS Elastic Load Balancing Documentation -- https://docs.aws.amazon.com/elasticloadbalancing/

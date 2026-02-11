# ロードバランサー

> トラフィックを複数サーバーに分散させるロードバランサーの仕組みを理解し、L4/L7の違い、分散アルゴリズム、ヘルスチェック戦略を習得する。

## この章で学ぶこと

1. L4（トランスポート層）とL7（アプリケーション層）ロードバランサーの動作原理と使い分け
2. 主要な負荷分散アルゴリズム（ラウンドロビン、重み付き、最小接続、コンシステントハッシング）
3. ヘルスチェックとフェイルオーバーの設計パターン

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

server {
    listen 443 ssl;
    server_name example.com;

    # SSL終端（L7 LBの重要な役割）
    ssl_certificate /etc/ssl/cert.pem;
    ssl_certificate_key /etc/ssl/key.pem;

    # パスベースルーティング
    location /api/ {
        proxy_pass http://api_servers;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Host $host;
    }

    location /static/ {
        proxy_pass http://static_servers;
        expires 30d;  # キャッシュヘッダー付与
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

class L4LoadBalancer:
    """L4 (TCP) ロードバランサーの簡易実装"""

    def __init__(self, listen_port: int, backends: list[tuple[str, int]]):
        self.listen_port = listen_port
        self.backends = backends
        self.backend_cycle = itertools.cycle(backends)
        self.lock = threading.Lock()

    def get_next_backend(self) -> tuple[str, int]:
        """ラウンドロビンで次のバックエンドを選択"""
        with self.lock:
            return next(self.backend_cycle)

    def handle_connection(self, client_sock: socket.socket):
        """クライアント接続をバックエンドに転送"""
        backend_host, backend_port = self.get_next_backend()
        print(f"Routing to {backend_host}:{backend_port}")

        backend_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        backend_sock.connect((backend_host, backend_port))

        # 双方向プロキシ（TCP透過転送）
        def forward(src, dst):
            while True:
                data = src.recv(4096)
                if not data:
                    break
                dst.sendall(data)
            src.close()
            dst.close()

        t1 = threading.Thread(target=forward, args=(client_sock, backend_sock))
        t2 = threading.Thread(target=forward, args=(backend_sock, client_sock))
        t1.start()
        t2.start()

    def start(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", self.listen_port))
        server.listen(128)
        print(f"L4 LB listening on :{self.listen_port}")

        while True:
            client_sock, addr = server.accept()
            threading.Thread(
                target=self.handle_connection,
                args=(client_sock,)
            ).start()
```

---

## 3. 負荷分散アルゴリズム

### コード例3: 主要アルゴリズムの実装

```python
import hashlib
import random
from collections import defaultdict
from bisect import bisect_right

class LoadBalancerAlgorithms:
    """主要な負荷分散アルゴリズム"""

    def __init__(self, servers: list[str]):
        self.servers = servers
        self.rr_index = 0
        self.connections = defaultdict(int)

    # 1. ラウンドロビン
    def round_robin(self) -> str:
        server = self.servers[self.rr_index % len(self.servers)]
        self.rr_index += 1
        return server

    # 2. 重み付きラウンドロビン
    def weighted_round_robin(self, weights: dict[str, int]) -> str:
        pool = []
        for server in self.servers:
            pool.extend([server] * weights.get(server, 1))
        server = pool[self.rr_index % len(pool)]
        self.rr_index += 1
        return server

    # 3. 最小接続数
    def least_connections(self) -> str:
        return min(self.servers, key=lambda s: self.connections[s])

    # 4. IPハッシュ
    def ip_hash(self, client_ip: str) -> str:
        hash_val = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return self.servers[hash_val % len(self.servers)]

    # 5. ランダム
    def random_select(self) -> str:
        return random.choice(self.servers)

# デモ
lb = LoadBalancerAlgorithms(["srv1", "srv2", "srv3"])

print("=== ラウンドロビン ===")
for _ in range(6):
    print(f"  → {lb.round_robin()}")

print("\n=== 重み付き (srv1:3, srv2:1, srv3:1) ===")
lb.rr_index = 0
weights = {"srv1": 3, "srv2": 1, "srv3": 1}
for _ in range(5):
    print(f"  → {lb.weighted_round_robin(weights)}")

print("\n=== IPハッシュ ===")
for ip in ["10.0.0.1", "10.0.0.2", "10.0.0.1"]:
    print(f"  {ip} → {lb.ip_hash(ip)}")
```

### コード例4: コンシステントハッシング

```python
import hashlib
from bisect import bisect_right

class ConsistentHash:
    """
    コンシステントハッシング
    サーバー追加/削除時に再配置されるキーを最小化
    """

    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring = []            # ソート済みハッシュ値
        self.ring_map = {}        # ハッシュ値 → 実サーバー
        self.servers = set()

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_server(self, server: str):
        """サーバー追加: 仮想ノードをリングに配置"""
        self.servers.add(server)
        for i in range(self.virtual_nodes):
            vnode_key = f"{server}#vn{i}"
            h = self._hash(vnode_key)
            self.ring.append(h)
            self.ring_map[h] = server
        self.ring.sort()
        print(f"Added {server} ({self.virtual_nodes} vnodes)")

    def remove_server(self, server: str):
        """サーバー削除: 仮想ノードをリングから除去"""
        self.servers.discard(server)
        self.ring = [h for h in self.ring if self.ring_map.get(h) != server]
        self.ring_map = {h: s for h, s in self.ring_map.items() if s != server}
        print(f"Removed {server}")

    def get_server(self, key: str) -> str:
        """キーに対応するサーバーを取得"""
        if not self.ring:
            raise Exception("No servers available")
        h = self._hash(key)
        idx = bisect_right(self.ring, h)
        if idx == len(self.ring):
            idx = 0
        return self.ring_map[self.ring[idx]]

# デモ: サーバー追加/削除時の影響
ch = ConsistentHash(virtual_nodes=100)
for s in ["server-A", "server-B", "server-C"]:
    ch.add_server(s)

# 1000キーの分布を確認
from collections import Counter
dist = Counter(ch.get_server(f"key-{i}") for i in range(1000))
print(f"\n分布: {dict(dist)}")

# サーバー追加時の影響（再配置率）
before = {f"key-{i}": ch.get_server(f"key-{i}") for i in range(1000)}
ch.add_server("server-D")
after = {f"key-{i}": ch.get_server(f"key-{i}") for i in range(1000)}
moved = sum(1 for k in before if before[k] != after[k])
print(f"再配置されたキー: {moved}/1000 ({moved/10:.1f}%)")
# 理論値: 1/4 = 25% (1000キー中約250キーが移動)
```

---

## 4. ヘルスチェック

### ASCII図解3: ヘルスチェックの種類

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
```

### コード例5: ヘルスチェック実装

```python
import asyncio
import aiohttp
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class ServerState(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"    # 新規接続拒否、既存接続は完了待ち

@dataclass
class BackendServer:
    host: str
    port: int
    state: ServerState = ServerState.HEALTHY
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    active_connections: int = 0

    @property
    def url(self):
        return f"http://{self.host}:{self.port}"

class HealthCheckManager:
    """アクティブ + パッシブヘルスチェックの統合管理"""

    def __init__(self, servers: list[BackendServer],
                 check_interval: float = 10.0,
                 unhealthy_threshold: int = 3,
                 healthy_threshold: int = 2,
                 timeout: float = 3.0):
        self.servers = servers
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.healthy_threshold = healthy_threshold
        self.timeout = timeout

    async def active_check(self, server: BackendServer,
                           session: aiohttp.ClientSession):
        """アクティブヘルスチェック"""
        try:
            async with session.get(
                f"{server.url}/health",
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                if resp.status == 200:
                    self._mark_success(server)
                else:
                    self._mark_failure(server, f"HTTP {resp.status}")
        except Exception as e:
            self._mark_failure(server, str(e))

    def passive_check(self, server: BackendServer, status_code: int):
        """パッシブヘルスチェック（実リクエスト結果から判定）"""
        if status_code >= 500:
            self._mark_failure(server, f"HTTP {status_code}")
        else:
            self._mark_success(server)

    def _mark_success(self, server: BackendServer):
        server.consecutive_failures = 0
        server.consecutive_successes += 1
        if (server.state == ServerState.UNHEALTHY and
                server.consecutive_successes >= self.healthy_threshold):
            server.state = ServerState.HEALTHY
            print(f"[HEALTH] {server.url} → HEALTHY (recovered)")

    def _mark_failure(self, server: BackendServer, reason: str):
        server.consecutive_successes = 0
        server.consecutive_failures += 1
        if (server.state == ServerState.HEALTHY and
                server.consecutive_failures >= self.unhealthy_threshold):
            server.state = ServerState.UNHEALTHY
            print(f"[HEALTH] {server.url} → UNHEALTHY ({reason})")

    def get_healthy_servers(self) -> list[BackendServer]:
        return [s for s in self.servers if s.state == ServerState.HEALTHY]
```

---

## 5. 比較表

### 比較表1: L4 vs L7 ロードバランサー

| 項目 | L4 (トランスポート層) | L7 (アプリケーション層) |
|------|---------------------|----------------------|
| 解析対象 | IP + ポート | HTTP/HTTPS/gRPC ヘッダー |
| SSL終端 | 不可（パススルー） | 可能 |
| コンテンツルーティング | 不可 | URL、ヘッダー、Cookie で振り分け |
| パフォーマンス | 高い（低オーバーヘッド） | 低い（ヘッダー解析コスト） |
| WebSocket | パススルーで対応 | ネイティブ対応 |
| 適するケース | TCP/UDP汎用、DB接続 | HTTP API、Webアプリ |
| 製品例 | AWS NLB, HAProxy (TCP) | AWS ALB, Nginx, Envoy |

### 比較表2: 負荷分散アルゴリズムの比較

| アルゴリズム | 均一性 | セッション維持 | 実装複雑度 | 適するケース |
|-------------|--------|--------------|-----------|-------------|
| ラウンドロビン | 高い | なし | 低 | 同性能サーバー群 |
| 重み付きRR | 高い | なし | 低 | 異性能サーバー混在 |
| 最小接続数 | 高い | なし | 中 | 処理時間が不均一 |
| IPハッシュ | 中程度 | あり | 低 | セッション固定が必要 |
| コンシステントハッシュ | 高い | あり | 高 | キャッシュ、スケール頻出 |
| ランダム | 中程度 | なし | 最低 | 大量サーバー |

---

## 6. アンチパターン

### アンチパターン1: ロードバランサー自体がSPOF

```
❌ ダメな例:
  Client → [単一LB] → Server群

  LBが落ちると全サービス停止

✅ 正しい構成:
  Client → [DNS] → [LB Active] → Server群
                    [LB Standby]  (VRRP/keepalived)

  または:
  Client → [DNS Round Robin]
           → LB-1 → Server群
           → LB-2 → Server群
```

### アンチパターン2: スティッキーセッションへの過度な依存

```
❌ ダメな例:
「全リクエストをCookieで同一サーバーに固定する」

問題:
- サーバー障害時にセッションが消失
- 負荷が偏る（人気ユーザーのサーバーに集中）
- スケールアウト時に既存セッションが移動不可

✅ 正しいアプローチ:
- セッションデータを Redis 等の共有ストアに保存
- JWTトークンでステートレス認証
- スティッキーセッションが必要な場合でも
  フォールバック先を用意する
```

---

## 7. FAQ

### Q1: グローバルサービスではどのようなLB構成にしますか？

3層構成が一般的: (1) DNS ベースの地理的分散（GeoDNS/Route 53）、(2) リージョンごとの L7 LB（ALB/Nginx）、(3) AZ ごとの L4 LB（NLB）。ユーザーに最も近いリージョンにルーティングし、リージョン内ではAZ間で分散する。CloudflareやAWS Global Acceleratorはエニーキャストを使い、BGPレベルで最寄りのPoP（Point of Presence）にルーティングする。

### Q2: LBのスループット限界はどの程度ですか？

ソフトウェアLB（Nginx）は1台あたり数万〜数十万RPS、ハードウェアLB（F5 BIG-IP）は数百万RPSを処理できる。AWS ALBは自動スケーリングで理論上無制限だが、急激なトラフィック増にはウォームアップが必要。NLBは数百万同時接続を処理可能。ボトルネックになる場合はDNSラウンドロビンで複数LBに分散する。

### Q3: gRPCの負荷分散はHTTPと何が違いますか？

gRPCはHTTP/2を使うため、1つのTCPコネクション上に複数ストリームが多重化される。L4 LBだとコネクション単位の振り分けになり、1コネクションに全リクエストが流れるため偏る。gRPCにはL7 LB（Envoy、Linkerd）を使い、ストリーム単位で振り分ける必要がある。クライアントサイドLB（gRPC built-in）も選択肢となる。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| LBの役割 | トラフィック分散、可用性確保、レイテンシ均一化 |
| L4 vs L7 | L4は高速・汎用、L7は柔軟・HTTP特化 |
| 主要アルゴリズム | RR、重み付き、最小接続、コンシステントハッシュ |
| ヘルスチェック | アクティブ（定期確認）+ パッシブ（実トラフィック判定） |
| LB自体の冗長化 | Active-Standby、DNS分散で SPOF を排除 |
| gRPC対応 | L7 LB（Envoy等）でストリーム単位の振り分けが必要 |

---

## 次に読むべきガイド

- [キャッシュ](./01-caching.md) — LBの背後に配置するキャッシュレイヤー
- [CDN](./03-cdn.md) — グローバルな静的コンテンツ配信
- [スケーラビリティ](../00-fundamentals/01-scalability.md) — LBを活用した水平スケーリング

---

## 参考文献

1. Karger, D. et al. (1997). "Consistent Hashing and Random Trees." *STOC '97*.
2. Nginx Documentation — https://nginx.org/en/docs/http/load_balancing.html
3. Envoy Proxy Documentation — https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/load_balancing/

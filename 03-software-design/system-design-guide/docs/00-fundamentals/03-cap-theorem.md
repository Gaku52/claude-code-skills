# CAP定理

> 分散システムにおける一貫性（Consistency）、可用性（Availability）、分断耐性（Partition Tolerance）のトレードオフを理解し、実システムの設計判断に活かす。

## この章で学ぶこと

1. CAP定理の正確な定義と「3つから2つ選ぶ」の正しい解釈
2. CP/APシステムの具体的な動作の違いとユースケース
3. PACELC定理による拡張的な理解と現実の設計判断

---

## 1. CAP定理とは

2000年にEric Brewerが提唱し、2002年にGilbert & Lynchが証明した定理。分散システムでは以下の3特性を**同時に全て満たすことは不可能**であることを示す。

```
C — Consistency（一貫性）
    全ノードが同一時刻に同じデータを返す
    （ここでの一貫性は「線形化可能性 / Linearizability」）

A — Availability（可用性）
    全てのリクエストが（成功/失敗の）レスポンスを受け取る
    ノード障害時にもレスポンスを返す

P — Partition Tolerance（分断耐性）
    ネットワーク分断が発生してもシステムが動作し続ける
```

### ASCII図解1: CAP定理の三角形

```
                    C (一貫性)
                    /\
                   /  \
                  /    \
                 / CP   \
                / 系統   \
               /          \
              /   CA       \
             /  (理論上     \
            /    のみ)      \
           /                 \
          /        AP         \
         /       系統          \
        /________________________\
      A (可用性)           P (分断耐性)

  ■ ネットワーク分断は避けられない
    → 実質的な選択は CP か AP

  CP: 分断時に一貫性を優先 → 一部リクエストを拒否
  AP: 分断時に可用性を優先 → 古いデータを返す可能性
  CA: 分断なし前提 → 単一ノード or 同一LAN内のみ
```

---

## 2. ネットワーク分断とは

### コード例1: ネットワーク分断のシミュレーション

```python
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class Node:
    name: str
    data: dict
    reachable: set  # 通信可能なノード名の集合

class DistributedStore:
    """ネットワーク分断を再現する分散ストア"""

    def __init__(self):
        self.nodes = {
            "node1": Node("node1", {"key": "value_v1"}, {"node2", "node3"}),
            "node2": Node("node2", {"key": "value_v1"}, {"node1", "node3"}),
            "node3": Node("node3", {"key": "value_v1"}, {"node1", "node2"}),
        }

    def partition(self, group_a: set, group_b: set):
        """ネットワーク分断を発生させる"""
        print(f"[PARTITION] {group_a} <-X-> {group_b}")
        for name in group_a:
            self.nodes[name].reachable -= group_b
        for name in group_b:
            self.nodes[name].reachable -= group_a

    def write_cp(self, node_name: str, key: str, value: str) -> bool:
        """CP方式: 過半数に書き込めなければ拒否"""
        node = self.nodes[node_name]
        reachable_count = 1 + len(node.reachable)
        quorum = len(self.nodes) // 2 + 1

        if reachable_count >= quorum:
            node.data[key] = value
            for peer_name in node.reachable:
                self.nodes[peer_name].data[key] = value
            print(f"[CP] 書き込み成功 ({reachable_count}/{len(self.nodes)} >= quorum {quorum})")
            return True
        else:
            print(f"[CP] 書き込み拒否 ({reachable_count}/{len(self.nodes)} < quorum {quorum})")
            return False

    def write_ap(self, node_name: str, key: str, value: str) -> bool:
        """AP方式: 到達可能なノードにのみ書き込み、常に成功"""
        node = self.nodes[node_name]
        node.data[key] = value
        for peer_name in node.reachable:
            self.nodes[peer_name].data[key] = value
        unreachable = set(self.nodes.keys()) - {node_name} - node.reachable
        if unreachable:
            print(f"[AP] 書き込み成功（{unreachable} は古いデータのまま）")
        else:
            print(f"[AP] 書き込み成功（全ノード同期済み）")
        return True

# デモ
store = DistributedStore()
store.partition({"node1", "node2"}, {"node3"})

print("\n--- CP方式 ---")
store.write_cp("node1", "key", "value_v2")  # 成功(2/3 >= 2)
store.write_cp("node3", "key", "value_v3")  # 拒否(1/3 < 2)

print("\n--- AP方式 ---")
store2 = DistributedStore()
store2.partition({"node1", "node2"}, {"node3"})
store2.write_ap("node1", "key", "value_v2")  # 成功(node3は古い)
store2.write_ap("node3", "key", "value_v3")  # 成功(node1,2は古い) → 矛盾！
```

---

## 3. CPシステムとAPシステム

### ASCII図解2: 分断時のCPとAPの動作

```
■ CPシステム（一貫性優先）

  Client A ──write "X=2"──→ Node1 ──sync──→ Node2
                                    ×  (分断)
                                   Node3

  Client B ──read "X"──→ Node3
  → エラー返却: "一貫性を保証できないため拒否"

■ APシステム（可用性優先）

  Client A ──write "X=2"──→ Node1 ──sync──→ Node2
                                    ×  (分断)
                                   Node3 (X=1のまま)

  Client B ──read "X"──→ Node3
  → "X=1" を返す（古いデータだがレスポンスは返る）
  → 分断解消後に Node3 を "X=2" に修復（結果整合性）
```

### コード例2: 結果整合性（Eventual Consistency）の実装

```python
import time
import threading
from collections import defaultdict

class EventuallyConsistentStore:
    """結果整合性を持つAPストア"""

    def __init__(self, node_id: str, peers: list):
        self.node_id = node_id
        self.data = {}
        self.vector_clock = defaultdict(int)
        self.peers = peers
        self.gossip_interval = 1.0  # 秒

    def write(self, key: str, value: str):
        """ローカルに書き込み、バックグラウンドで同期"""
        self.vector_clock[self.node_id] += 1
        self.data[key] = {
            "value": value,
            "clock": dict(self.vector_clock),
            "timestamp": time.time(),
        }
        print(f"[{self.node_id}] Write: {key}={value} "
              f"clock={dict(self.vector_clock)}")

    def read(self, key: str):
        """ローカルデータを即座に返す（古い可能性あり）"""
        entry = self.data.get(key)
        if entry:
            return entry["value"]
        return None

    def merge(self, key: str, remote_entry: dict):
        """リモートデータとマージ（Last-Writer-Wins）"""
        local_entry = self.data.get(key)
        if local_entry is None:
            self.data[key] = remote_entry
            return

        # タイムスタンプが新しい方を採用（LWW）
        if remote_entry["timestamp"] > local_entry["timestamp"]:
            self.data[key] = remote_entry
            print(f"[{self.node_id}] Merge: {key} updated to "
                  f"{remote_entry['value']}")

    def gossip(self):
        """ゴシッププロトコルで隣接ノードにデータを伝播"""
        for peer in self.peers:
            for key, entry in self.data.items():
                peer.merge(key, entry)
```

### コード例3: クォーラム読み書き

```python
class QuorumStore:
    """クォーラムベースの読み書き (Dynamo風)"""

    def __init__(self, n: int, w: int, r: int):
        """
        N: レプリカ数
        W: 書き込みに必要な応答数
        R: 読み込みに必要な応答数
        W + R > N → 強い一貫性を保証
        """
        self.n = n
        self.w = w
        self.r = r
        self.replicas = [{} for _ in range(n)]
        print(f"Quorum: N={n}, W={w}, R={r}")
        print(f"Strong consistency: W+R > N → {w}+{r} > {n} = {w+r > n}")

    def write(self, key: str, value: str) -> bool:
        """Wノードに書き込み成功で完了"""
        success = 0
        for i, replica in enumerate(self.replicas):
            # 実際にはネットワーク越しに書き込み
            replica[key] = {"value": value, "version": time.time()}
            success += 1
            if success >= self.w:
                print(f"Write OK: {key}={value} (ack: {success}/{self.n})")
                return True
        return False

    def read(self, key: str) -> str:
        """Rノードから読み込み、最新版を返す"""
        responses = []
        for i, replica in enumerate(self.replicas):
            entry = replica.get(key)
            if entry:
                responses.append(entry)
            if len(responses) >= self.r:
                break

        if not responses:
            return None

        # 最新のバージョンを返す
        latest = max(responses, key=lambda x: x["version"])
        return latest["value"]

# 強い一貫性: W + R > N
strong = QuorumStore(n=3, w=2, r=2)  # 2+2 > 3 → True

# 結果整合性: W + R <= N
eventual = QuorumStore(n=3, w=1, r=1)  # 1+1 > 3 → False
```

---

## 4. PACELC定理

CAPは分断時のみのトレードオフだが、PAcELCは**分断がない通常時のトレードオフ**も扱う。

### ASCII図解3: PACELC定理

```
  分断 (Partition) あり？
  ├── YES → A (可用性) vs C (一貫性) を選ぶ
  │         ├── PA: 可用性優先    (例: Cassandra, DynamoDB)
  │         └── PC: 一貫性優先    (例: MongoDB, HBase)
  │
  └── NO (Else) → L (レイテンシ) vs C (一貫性) を選ぶ
                   ├── EL: レイテンシ優先  (例: Cassandra, DynamoDB)
                   └── EC: 一貫性優先     (例: MongoDB, HBase)

  組み合わせ:
  ┌───────────────┬─────────────────────────────────┐
  │ PA/EL         │ Cassandra, DynamoDB, Riak       │
  │               │ → 常にレイテンシ・可用性重視     │
  ├───────────────┼─────────────────────────────────┤
  │ PC/EC         │ MongoDB, HBase, Spanner         │
  │               │ → 常に一貫性重視                 │
  ├───────────────┼─────────────────────────────────┤
  │ PA/EC         │ 分断時は可用性、通常時は一貫性   │
  │               │ → Yahoo PNUTS                   │
  └───────────────┴─────────────────────────────────┘
```

### コード例4: 一貫性レベルの設定（Cassandra風）

```python
from enum import Enum

class ConsistencyLevel(Enum):
    ONE = 1           # 1ノードの応答で完了（最速・最弱）
    QUORUM = "quorum" # 過半数の応答で完了（バランス）
    ALL = "all"       # 全ノードの応答で完了（最強・最遅）
    LOCAL_QUORUM = "local_quorum"  # ローカルDC内の過半数

class CassandraClient:
    def __init__(self, replication_factor: int = 3):
        self.rf = replication_factor

    def required_responses(self, level: ConsistencyLevel) -> int:
        if level == ConsistencyLevel.ONE:
            return 1
        elif level == ConsistencyLevel.QUORUM:
            return self.rf // 2 + 1
        elif level == ConsistencyLevel.ALL:
            return self.rf
        elif level == ConsistencyLevel.LOCAL_QUORUM:
            return self.rf // 2 + 1  # 簡略化

    def is_strongly_consistent(self, write_cl: ConsistencyLevel,
                                read_cl: ConsistencyLevel) -> bool:
        """W + R > N なら強い一貫性"""
        w = self.required_responses(write_cl)
        r = self.required_responses(read_cl)
        return (w + r) > self.rf

    def analyze(self, write_cl: ConsistencyLevel, read_cl: ConsistencyLevel):
        w = self.required_responses(write_cl)
        r = self.required_responses(read_cl)
        strong = self.is_strongly_consistent(write_cl, read_cl)
        print(f"W={write_cl.name}({w}) + R={read_cl.name}({r}) "
              f"{'>' if strong else '<='} N={self.rf} "
              f"→ {'強い一貫性' if strong else '結果整合性'}")

client = CassandraClient(replication_factor=3)
client.analyze(ConsistencyLevel.ONE, ConsistencyLevel.ONE)
# W=ONE(1) + R=ONE(1) <= N=3 → 結果整合性

client.analyze(ConsistencyLevel.QUORUM, ConsistencyLevel.QUORUM)
# W=QUORUM(2) + R=QUORUM(2) > N=3 → 強い一貫性

client.analyze(ConsistencyLevel.ALL, ConsistencyLevel.ONE)
# W=ALL(3) + R=ONE(1) > N=3 → 強い一貫性
```

### コード例5: コンフリクト解決戦略

```python
from dataclasses import dataclass
from typing import Any
import time

@dataclass
class VersionedValue:
    value: Any
    timestamp: float
    node_id: str
    vector_clock: dict

class ConflictResolver:
    """分断解消後のコンフリクト解決戦略"""

    @staticmethod
    def last_writer_wins(v1: VersionedValue, v2: VersionedValue) -> VersionedValue:
        """LWW: タイムスタンプが新しい方を採用"""
        winner = v1 if v1.timestamp > v2.timestamp else v2
        print(f"LWW: {winner.value} wins (timestamp: {winner.timestamp:.3f})")
        return winner

    @staticmethod
    def merge_values(v1: VersionedValue, v2: VersionedValue) -> VersionedValue:
        """マージ: 両方の値を保持（ショッピングカート等で有用）"""
        if isinstance(v1.value, set) and isinstance(v2.value, set):
            merged = v1.value | v2.value
            print(f"Merge: {v1.value} ∪ {v2.value} = {merged}")
            return VersionedValue(merged, time.time(), "merged", {})
        raise ValueError("Merge not supported for non-set values")

    @staticmethod
    def application_level(v1: VersionedValue, v2: VersionedValue,
                          resolver) -> VersionedValue:
        """アプリケーション固有のロジックで解決"""
        return resolver(v1, v2)

# 使用例
v1 = VersionedValue("price=100", 1000.001, "node1", {"node1": 1})
v2 = VersionedValue("price=120", 1000.005, "node2", {"node2": 1})

ConflictResolver.last_writer_wins(v1, v2)
# LWW: price=120 wins

cart1 = VersionedValue({"itemA", "itemB"}, 1000.0, "node1", {})
cart2 = VersionedValue({"itemB", "itemC"}, 1000.0, "node2", {})
ConflictResolver.merge_values(cart1, cart2)
# Merge: {'itemA', 'itemB'} ∪ {'itemB', 'itemC'} = {'itemA', 'itemB', 'itemC'}
```

---

## 5. 比較表

### 比較表1: 主要データベースのCAP分類

| データベース | CAP分類 | PACELC | 一貫性モデル | ユースケース |
|-------------|---------|--------|-------------|-------------|
| PostgreSQL | CA (単一ノード) | PC/EC | 強い一貫性 | 一般Web、金融 |
| MongoDB | CP | PC/EC | 強い一貫性 (Primary) | ドキュメントDB |
| Cassandra | AP | PA/EL | 結果整合性 (設定可) | IoT、時系列 |
| DynamoDB | AP | PA/EL | 結果整合性 (設定可) | EC、ゲーム |
| Google Spanner | CP | PC/EC | 強い一貫性 | グローバル金融 |
| Redis Cluster | AP | PA/EL | 結果整合性 | キャッシュ |
| etcd | CP | PC/EC | 強い一貫性 | 設定管理 |
| CockroachDB | CP | PC/EC | 強い一貫性 | 分散SQL |

### 比較表2: 一貫性モデルの比較

| 一貫性モデル | 強さ | レイテンシ | 説明 |
|-------------|------|-----------|------|
| 線形化可能性 | 最強 | 最高 | 全操作がリアルタイムで一貫 |
| 逐次一貫性 | 強 | 高 | 全ノードで同一順序 |
| 因果一貫性 | 中 | 中 | 因果関係がある操作のみ順序保証 |
| 結果整合性 | 弱 | 低 | いつかは一貫するが時間は不定 |
| Read-your-writes | 中 | 中 | 自分の書き込みは即座に読める |

---

## 6. アンチパターン

### アンチパターン1: CAP定理の誤解「3つから2つ選ぶ」

```
❌ よくある誤解:
「CAPから2つを選ぶ。うちはCA（一貫性+可用性）を選ぶ」

→ 分散システムではネットワーク分断(P)は避けられない
→ CAは「分散でないシステム」（単一ノード）にしか成立しない
→ 実際の選択は「分断時に C を優先するか A を優先するか」

✅ 正しい理解:
- P は選択肢ではなく前提条件
- 分断時: CP（一貫性優先）or AP（可用性優先）
- 分断がない通常時も L vs C のトレードオフが存在（PACELC）
```

### アンチパターン2: 全てに強い一貫性を要求する

```
❌ ダメな例:
「金融システムだから全データを強い一貫性で管理する」

→ 一貫性レベルはデータの性質ごとに使い分けるべき

✅ 正しいアプローチ:
  ┌──────────────┬───────────────────┐
  │ データ        │ 一貫性レベル       │
  ├──────────────┼───────────────────┤
  │ 口座残高      │ 強い一貫性（必須） │
  │ 取引履歴      │ 強い一貫性（必須） │
  │ ユーザー設定  │ 結果整合性でOK    │
  │ アクセスログ  │ 結果整合性でOK    │
  │ レコメンド    │ 結果整合性でOK    │
  └──────────────┴───────────────────┘
```

---

## 7. FAQ

### Q1: NoSQLは全てAPシステムですか？

全くそうではない。MongoDBはCP（Primaryでの書き込みを保証）、HBaseもCP（ZooKeeperによるリーダー選出）である。一方CassandraやDynamoDBはAPだが、クエリごとに一貫性レベルを調整できる。「NoSQL = AP」は誤解であり、データモデル（KV/Document/Column/Graph）とCAP特性は独立した概念である。

### Q2: Google Spannerは「CAPを破った」のですか？

Spannerは「事実上の CA」と言われることがあるが、厳密にはCPシステムである。Googleの専用ネットワーク（冗長な海底ケーブル等）により分断確率を極限まで下げ、TrueTime（原子時計+GPS）により低レイテンシで強い一貫性を実現している。CAP定理を破ったのではなく、Pの発生確率を工学的に極小化したと理解するのが正確である。

### Q3: 結果整合性で問題になるケースは？

典型的な問題は「書き込み直後の読み込み」で古いデータが返るケースである。例えば、ユーザーがプロフィール画像を更新した直後にページをリロードすると古い画像が表示される。対策として (1) Read-your-writes一貫性（自分の書き込みは即座に反映）、(2) Sticky Session（同じノードにルーティング）、(3) クライアント側のOptimistic UI（書き込み結果を即座に反映して後で同期）がある。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| CAP定理 | C(一貫性)・A(可用性)・P(分断耐性)の3つを同時に満たせない |
| Pは前提 | 分散システムで分断は避けられない。実質CP or AP の二択 |
| CPシステム | 分断時に一貫性優先。書き込み拒否の可能性 |
| APシステム | 分断時に可用性優先。古いデータを返す可能性 |
| PACELC | 通常時の L vs C のトレードオフも考慮する拡張定理 |
| 設計指針 | データの性質ごとに一貫性レベルを使い分ける |

---

## 次に読むべきガイド

- [DBスケーリング](../01-components/04-database-scaling.md) — レプリケーションとシャーディングの実践
- [メッセージキュー](../01-components/02-message-queue.md) — 非同期処理と結果整合性
- [イベント駆動アーキテクチャ](../02-architecture/03-event-driven.md) — 結果整合性を前提とした設計

---

## 参考文献

1. Brewer, E. (2012). "CAP Twelve Years Later: How the 'Rules' Have Changed." *IEEE Computer*, 45(2), 23-29.
2. Gilbert, S. & Lynch, N. (2002). "Brewer's conjecture and the feasibility of consistent, available, partition-tolerant web services." *ACM SIGACT News*, 33(2), 51-59.
3. Abadi, D. (2012). "Consistency Tradeoffs in Modern Distributed Database System Design." *IEEE Computer*, 45(2), 37-42.
4. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Chapter 9: Consistency and Consensus. O'Reilly Media.

# CAP定理

> 分散システムにおける一貫性（Consistency）、可用性（Availability）、分断耐性（Partition Tolerance）のトレードオフを理解し、PACELC定理を含む拡張的知識と、クォーラム・結果整合性・コンフリクト解決の実装を通じて、実システムの設計判断に活かす。

---

## この章で学ぶこと

1. CAP定理の正確な定義と「3つから2つ選ぶ」の正しい解釈（よくある誤解の是正）
2. CP/APシステムの具体的な動作の違い、クォーラムベースの一貫性制御の実装
3. PACELC定理による拡張的な理解と、データの性質ごとに一貫性レベルを使い分ける設計判断

---

## 前提知識

| トピック | 内容 | 参照先 |
|---------|------|--------|
| 信頼性 | 可用性、フォールトトレランスの基本概念 | [信頼性](./02-reliability.md) |
| スケーラビリティ | 水平スケーリング、レプリケーションの概念 | [スケーラビリティ](./01-scalability.md) |
| ネットワーク基礎 | TCP/IP、DNS、パケットロスの基本 | [Web/ネットワーク基礎](../../../04-web-and-network/) |
| Python 基礎 | dataclass、辞書操作、基本的なクラス設計 | [プログラミング基礎](../../../02-programming/) |

---

## 1. CAP定理とは

### 1.1 定義

2000年にEric Brewerが提唱し、2002年にGilbert & Lynchが証明した定理。分散システムでは以下の3特性を**同時に全て満たすことは不可能**であることを示す。

```
C — Consistency（一貫性）
    全ノードが同一時刻に同じデータを返す
    （ここでの一貫性は「線形化可能性 / Linearizability」を指す）
    ACIDのCとは異なる概念であることに注意

A — Availability（可用性）
    障害が発生していないノードへの全てのリクエストが
    （成功/失敗の）レスポンスを受け取る
    ノード障害時にもレスポンスを返す

P — Partition Tolerance（分断耐性）
    ネットワーク分断（ノード間の通信断絶）が発生しても
    システムが動作し続ける
```

### 1.2 WHY: なぜCAP定理を理解すべきか

分散システムの設計では、一貫性と可用性のトレードオフが常に存在する。CAP定理を正しく理解していないと、以下のような問題が発生する。

```
よくある失敗:
─────────────────────────────────────────────
1. 「全データに強い一貫性を適用」
   → レイテンシが悪化し、可用性が低下
   → ユーザー設定やログまで強い一貫性にする必要はない

2. 「全データに結果整合性を適用」
   → 決済データで残高不整合が発生
   → 二重引き落としなどの深刻な問題

3. 「CAは実現可能」と誤解
   → 分散システムでは分断は避けられない
   → 分断発生時にシステムが完全停止
─────────────────────────────────────────────
正しいアプローチ: データの性質ごとに一貫性レベルを使い分ける
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

### 1.3 一貫性モデルの階層

CAP定理の「C」は最も強い一貫性（線形化可能性）を指すが、現実には複数の一貫性レベルが存在する。

```
一貫性モデルの強さ（上ほど強い）:
─────────────────────────────────────────────
  線形化可能性 (Linearizability) ← CAPの「C」
    ↑ 全操作がリアルタイムの順序で一貫
    │ 実装: Raft, Paxos, 2PC
    │
  逐次一貫性 (Sequential Consistency)
    ↑ 全ノードで同一の操作順序を観測
    │
  因果一貫性 (Causal Consistency)
    ↑ 因果関係のある操作のみ順序保証
    │ 実装: ベクタークロック
    │
  Read-your-writes Consistency
    ↑ 自分の書き込みは即座に読める
    │ 実装: Sticky Session, Primary Read
    │
  結果整合性 (Eventual Consistency) ← APシステムの標準
    ↓ いつかは一貫するが時間は不定
    │ 実装: ゴシッププロトコル, CRDTs
─────────────────────────────────────────────
  強い一貫性ほどレイテンシが増大する
```

---

## 2. ネットワーク分断とは

### 2.1 分断の種類と発生原因

```
■ 完全分断（Full Partition）
  [Node A] ──×── [Node B]
  双方向の通信が完全に断絶

■ 部分分断（Partial Partition）
  [Node A] ──×── [Node B]
       \              /
        \            /
         [Node C]     ← CはAとBの両方と通信可能
  CがブリッジとなりA-B間の通信を中継できる可能性

■ 非対称分断（Asymmetric Partition）
  [Node A] ──→── [Node B]  ← AからBへは送信可能
  [Node A] ──×── [Node B]  ← BからAへは送信不可

発生原因:
  - ネットワーク機器の故障（スイッチ、ルーター）
  - ケーブル断線
  - ファイアウォールの誤設定
  - DNS障害
  - クラウドプロバイダのAZ間接続障害
  - GCによるSTW（Stop-The-World）で一時的にタイムアウト

現実の統計:
  Googleのデータ（2011年論文）:
  → 年間平均5.47回のネットワーク分断が発生
  → 分断の平均持続時間: 23分
  → データセンター間の分断は避けられないという結論
```

### コード例1: ネットワーク分断のシミュレーション

```python
from dataclasses import dataclass, field
from typing import Optional, Set, Dict
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

    def heal(self):
        """分断を解消する"""
        all_names = set(self.nodes.keys())
        for name, node in self.nodes.items():
            node.reachable = all_names - {name}
        print("[HEAL] ネットワーク分断を解消")

    def write_cp(self, node_name: str, key: str, value: str) -> bool:
        """CP方式: 過半数に書き込めなければ拒否（クォーラム）"""
        node = self.nodes[node_name]
        reachable_count = 1 + len(node.reachable)  # 自分自身 + 到達可能ノード
        quorum = len(self.nodes) // 2 + 1

        if reachable_count >= quorum:
            # 過半数に到達可能 → 書き込み成功
            node.data[key] = value
            for peer_name in node.reachable:
                self.nodes[peer_name].data[key] = value
            print(f"[CP] 書き込み成功 ({reachable_count}/{len(self.nodes)} >= quorum {quorum})")
            return True
        else:
            # 過半数に到達不可 → 書き込み拒否（一貫性を優先）
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

    def read_all(self, key: str) -> dict:
        """全ノードのデータを表示"""
        result = {}
        for name, node in self.nodes.items():
            result[name] = node.data.get(key, "NOT_FOUND")
        return result


# === デモ ===
print("=== ネットワーク分断のシミュレーション ===\n")

# 初期状態
store = DistributedStore()
print(f"初期状態: {store.read_all('key')}")

# 分断を発生させる
store.partition({"node1", "node2"}, {"node3"})

print("\n--- CP方式 ---")
store.write_cp("node1", "key", "value_v2")  # 成功 (2/3 >= 2)
store.write_cp("node3", "key", "value_v3")  # 拒否 (1/3 < 2)
print(f"データ状態: {store.read_all('key')}")
# → node1,node2 = "value_v2", node3 = "value_v1"（拒否されたので古いまま）

print("\n--- AP方式 ---")
store2 = DistributedStore()
store2.partition({"node1", "node2"}, {"node3"})
store2.write_ap("node1", "key", "value_v2")  # 成功 (node3は古い)
store2.write_ap("node3", "key", "value_v3")  # 成功 (node1,2とは異なる値)
print(f"データ状態: {store2.read_all('key')}")
# → node1,node2 = "value_v2", node3 = "value_v3" → 矛盾（コンフリクト）！
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

  メリット: 読み取りデータが常に正確
  デメリット: 分断時に一部ノードが応答不能
  代表: MongoDB, HBase, etcd, ZooKeeper

■ APシステム（可用性優先）

  Client A ──write "X=2"──→ Node1 ──sync──→ Node2
                                    ×  (分断)
                                   Node3 (X=1のまま)

  Client B ──read "X"──→ Node3
  → "X=1" を返す（古いデータだがレスポンスは返る）
  → 分断解消後に Node3 を "X=2" に修復（結果整合性）

  メリット: 常にレスポンスを返す（高可用性）
  デメリット: 古いデータが返る可能性がある
  代表: Cassandra, DynamoDB, CouchDB, Riak
```

### ASCII図解3: 分断時の判断フロー

```
分断が発生した場合の判断フロー:

  ネットワーク分断検知
        │
        ▼
  ┌─────────────────┐
  │ 一貫性が必要か？ │
  └────────┬────────┘
           │
     ┌─────┴─────┐
     │           │
    YES          NO
     │           │
     ▼           ▼
  ┌──────┐   ┌──────┐
  │  CP  │   │  AP  │
  │      │   │      │
  │少数側の│   │全ノード│
  │ノードは│   │が応答 │
  │読み書き│   │可能   │
  │を拒否  │   │      │
  └──────┘   └──────┘
     │           │
     ▼           ▼
  一貫した    古いデータの
  データを    可能性あり
  保証       （要:コンフリクト解決）
```

### コード例2: 結果整合性（Eventual Consistency）の実装

```python
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, List

@dataclass
class VersionedEntry:
    """バージョン付きデータエントリ"""
    value: str
    vector_clock: Dict[str, int]
    timestamp: float
    node_id: str

class EventuallyConsistentStore:
    """結果整合性を持つAPストア（ゴシッププロトコルベース）

    WHY ゴシッププロトコル:
      全ノードに一斉ブロードキャストすると、ネットワーク負荷が O(N^2) になる。
      ゴシッププロトコルでは各ノードがランダムに選んだ隣接ノードに情報を伝播
      するため、O(N log N) ラウンドで全ノードに伝わる（疫学モデルに由来）。
    """

    def __init__(self, node_id: str, peers: list = None):
        self.node_id = node_id
        self.data: Dict[str, VersionedEntry] = {}
        self.vector_clock: Dict[str, int] = defaultdict(int)
        self.peers: List['EventuallyConsistentStore'] = peers or []
        self.gossip_interval = 1.0

    def write(self, key: str, value: str):
        """ローカルに書き込み、バックグラウンドで同期"""
        self.vector_clock[self.node_id] += 1
        entry = VersionedEntry(
            value=value,
            vector_clock=dict(self.vector_clock),
            timestamp=time.time(),
            node_id=self.node_id,
        )
        self.data[key] = entry
        print(f"[{self.node_id}] Write: {key}={value} "
              f"clock={dict(self.vector_clock)}")

    def read(self, key: str) -> Optional[str]:
        """ローカルデータを即座に返す（古い可能性あり）"""
        entry = self.data.get(key)
        if entry:
            return entry.value
        return None

    def merge(self, key: str, remote_entry: VersionedEntry):
        """リモートデータとマージ（Last-Writer-Wins）

        WHY LWW:
          最も単純なコンフリクト解決戦略。タイムスタンプが新しい方を採用する。
          問題点: 時計のずれ（clock skew）により不正確になる可能性がある。
          代替: ベクタークロック比較、CRDTs、アプリケーション固有のマージロジック。
        """
        local_entry = self.data.get(key)
        if local_entry is None:
            self.data[key] = remote_entry
            return

        # ベクタークロックで因果関係を判定
        relation = self._compare_clocks(local_entry.vector_clock,
                                        remote_entry.vector_clock)
        if relation == "before":
            # ローカルが古い → リモートで上書き
            self.data[key] = remote_entry
            print(f"[{self.node_id}] Merge: {key} updated to {remote_entry.value}")
        elif relation == "concurrent":
            # 並行書き込み → LWW で解決
            if remote_entry.timestamp > local_entry.timestamp:
                self.data[key] = remote_entry
                print(f"[{self.node_id}] Merge (LWW): {key} = {remote_entry.value}")

    def _compare_clocks(self, clock_a: dict, clock_b: dict) -> str:
        """ベクタークロックの比較

        Returns: "before" | "after" | "concurrent"
        """
        all_keys = set(clock_a.keys()) | set(clock_b.keys())
        a_lte_b = all(clock_a.get(k, 0) <= clock_b.get(k, 0) for k in all_keys)
        b_lte_a = all(clock_b.get(k, 0) <= clock_a.get(k, 0) for k in all_keys)

        if a_lte_b and not b_lte_a:
            return "before"   # A は B より前
        elif b_lte_a and not a_lte_b:
            return "after"    # A は B より後
        else:
            return "concurrent"  # 並行（因果関係なし）

    def gossip(self):
        """ゴシッププロトコルで隣接ノードにデータを伝播"""
        for peer in self.peers:
            for key, entry in self.data.items():
                peer.merge(key, entry)


# === デモ ===
node_a = EventuallyConsistentStore("nodeA")
node_b = EventuallyConsistentStore("nodeB")
node_a.peers = [node_b]
node_b.peers = [node_a]

# 正常時: 書き込み → ゴシップで同期
node_a.write("user:1", "Alice")
node_a.gossip()
print(f"nodeB read: {node_b.read('user:1')}")  # → "Alice"

# 分断時: 両ノードが独立に書き込み
node_a.peers = []  # 分断シミュレーション
node_b.peers = []
node_a.write("user:1", "Alice_updated_by_A")
time.sleep(0.01)  # わずかな時間差
node_b.write("user:1", "Alice_updated_by_B")

# 分断解消 → ゴシップでマージ
node_a.peers = [node_b]
node_b.peers = [node_a]
node_a.gossip()
node_b.gossip()
# LWW により timestamp が新しい方が勝つ
```

### コード例3: クォーラム読み書き

```python
import time
from typing import Optional, List, Dict

class QuorumStore:
    """クォーラムベースの読み書き (Dynamo/Cassandra風)

    WHY クォーラム:
      N個のレプリカに対して、W個の書き込み確認 + R個の読み込み確認を要求する。
      W + R > N のとき、読み書きの「重なり」が保証され、
      最新のデータを必ず読める（強い一貫性）。

    トレードオフ:
      W=1, R=1 → 最速だが一貫性なし（結果整合性）
      W=N, R=1 → 書き込みが遅いが読み取りが速い
      W=1, R=N → 書き込みが速いが読み取りが遅い
      W=⌊N/2⌋+1, R=⌊N/2⌋+1 → バランス（最も一般的）
    """

    def __init__(self, n: int, w: int, r: int):
        self.n = n
        self.w = w
        self.r = r
        self.replicas: List[Dict] = [{} for _ in range(n)]
        self.is_strong = (w + r) > n
        print(f"Quorum: N={n}, W={w}, R={r}")
        print(f"Strong consistency: W+R > N → {w}+{r} > {n} = {self.is_strong}")

    def write(self, key: str, value: str) -> bool:
        """Wノードに書き込み成功で完了"""
        version = time.time()
        success = 0
        for i, replica in enumerate(self.replicas):
            replica[key] = {"value": value, "version": version}
            success += 1
            if success >= self.w:
                remaining = self.n - success
                print(f"Write OK: {key}={value} "
                      f"(ack: {success}/{self.n}, async: {remaining})")
                # 残りのレプリカは非同期で伝播
                return True
        return False

    def read(self, key: str) -> Optional[str]:
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

        # 最新のバージョンを返す（Read Repair の基礎）
        latest = max(responses, key=lambda x: x["version"])
        return latest["value"]

    def read_repair(self, key: str):
        """Read Repair: 読み取り時に古いレプリカを更新

        WHY Read Repair:
          クォーラム読み取りで最新値を取得した後、古い値を持つ
          レプリカを最新値で更新する。これにより、結果整合性が
          より速く収束する。Cassandraの重要な機能。
        """
        # 全レプリカから読み取り
        entries = [(i, r.get(key)) for i, r in enumerate(self.replicas)]
        valid = [(i, e) for i, e in entries if e is not None]

        if not valid:
            return

        # 最新バージョンを特定
        latest_idx, latest_entry = max(valid, key=lambda x: x[1]["version"])

        # 古いレプリカを更新
        for i, entry in valid:
            if entry["version"] < latest_entry["version"]:
                self.replicas[i][key] = latest_entry
                print(f"[Read Repair] Replica {i}: "
                      f"{entry['value']} → {latest_entry['value']}")


# === デモ ===
print("=== 強い一貫性: W + R > N ===")
strong = QuorumStore(n=3, w=2, r=2)  # 2+2 > 3 → True
strong.write("x", "100")
print(f"Read: {strong.read('x')}")

print("\n=== 結果整合性: W + R <= N ===")
eventual = QuorumStore(n=3, w=1, r=1)  # 1+1 > 3 → False
eventual.write("x", "200")
print(f"Read: {eventual.read('x')}")

print("\n=== 書き込み重視: W=1, R=N ===")
write_fast = QuorumStore(n=3, w=1, r=3)  # 1+3 > 3 → True
write_fast.write("x", "300")
print(f"Read: {write_fast.read('x')}")
```

---

## 4. PACELC定理

### 4.1 CAPの拡張

CAPは分断時のみのトレードオフだが、PAcELCは**分断がない通常時のトレードオフ**も扱う。

### ASCII図解4: PACELC定理

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

  WHY PACELC が重要:
    CAP は「分断が発生した特殊な状況」のみを扱う。
    しかし実際のシステムは分断がない時間の方が圧倒的に長い。
    通常時の L vs C のトレードオフが、日常的な性能に直結する。
```

### コード例4: 一貫性レベルの設定（Cassandra風）

```python
from enum import Enum

class ConsistencyLevel(Enum):
    ONE = 1           # 1ノードの応答で完了（最速・最弱）
    QUORUM = "quorum" # 過半数の応答で完了（バランス）
    ALL = "all"       # 全ノードの応答で完了（最強・最遅）
    LOCAL_QUORUM = "local_quorum"  # ローカルDC内の過半数
    EACH_QUORUM = "each_quorum"    # 各DC内の過半数

class CassandraClient:
    """Cassandra の一貫性レベル分析ツール

    WHY クエリごとに一貫性レベルを変えるのか:
      Cassandra はAPシステムだが、クエリ単位で一貫性レベルを指定できる。
      これにより、同じクラスタ内で「口座残高はQUORUM読み書き」
      「アクセスログはONE書き込み」と使い分けられる。
    """

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
            return self.rf // 2 + 1
        elif level == ConsistencyLevel.EACH_QUORUM:
            return self.rf // 2 + 1

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
        latency = "低" if w == 1 and r == 1 else "中" if w + r <= self.rf + 1 else "高"
        print(f"W={write_cl.name}({w}) + R={read_cl.name}({r}) "
              f"{'>' if strong else '<='} N={self.rf} "
              f"→ {'強い一貫性' if strong else '結果整合性'} "
              f"(レイテンシ: {latency})")

    def recommend(self, use_case: str):
        """ユースケースに応じた推奨設定"""
        recommendations = {
            "balance": (ConsistencyLevel.QUORUM, ConsistencyLevel.QUORUM,
                       "口座残高: 強い一貫性が必須"),
            "log": (ConsistencyLevel.ONE, ConsistencyLevel.ONE,
                   "アクセスログ: レイテンシ重視、多少の損失は許容"),
            "session": (ConsistencyLevel.LOCAL_QUORUM, ConsistencyLevel.LOCAL_QUORUM,
                       "セッション: DCローカルの一貫性で十分"),
            "config": (ConsistencyLevel.ALL, ConsistencyLevel.ONE,
                      "設定データ: 全レプリカに確実に書き込み、読み取りは速く"),
        }
        if use_case in recommendations:
            w_cl, r_cl, description = recommendations[use_case]
            print(f"\n{description}")
            self.analyze(w_cl, r_cl)


# === デモ ===
client = CassandraClient(replication_factor=3)

print("=== 一貫性レベルの分析 ===")
client.analyze(ConsistencyLevel.ONE, ConsistencyLevel.ONE)
# W=ONE(1) + R=ONE(1) <= N=3 → 結果整合性 (レイテンシ: 低)

client.analyze(ConsistencyLevel.QUORUM, ConsistencyLevel.QUORUM)
# W=QUORUM(2) + R=QUORUM(2) > N=3 → 強い一貫性 (レイテンシ: 中)

client.analyze(ConsistencyLevel.ALL, ConsistencyLevel.ONE)
# W=ALL(3) + R=ONE(1) > N=3 → 強い一貫性 (レイテンシ: 高)

print("\n=== ユースケース別推奨 ===")
client.recommend("balance")
client.recommend("log")
client.recommend("session")
```

### コード例5: コンフリクト解決戦略

```python
from dataclasses import dataclass
from typing import Any, Callable
import time

@dataclass
class VersionedValue:
    value: Any
    timestamp: float
    node_id: str
    vector_clock: dict

class ConflictResolver:
    """分断解消後のコンフリクト解決戦略

    WHY 複数の解決戦略が必要か:
      データの性質によって最適な解決方法が異なる。
      - 価格データ → LWW（最新値が正しい）
      - ショッピングカート → マージ（両方の追加を保持）
      - カウンター → CRDTs（数学的にマージ可能な構造）
      - 注文データ → アプリケーション固有のロジック
    """

    @staticmethod
    def last_writer_wins(v1: VersionedValue, v2: VersionedValue) -> VersionedValue:
        """LWW: タイムスタンプが新しい方を採用

        メリット: シンプル、実装が容易
        デメリット: 時計のずれに脆弱、データが静かに失われる
        用途: セッション、キャッシュ、最終更新状態
        """
        winner = v1 if v1.timestamp > v2.timestamp else v2
        print(f"LWW: {winner.value} wins (from {winner.node_id})")
        return winner

    @staticmethod
    def merge_values(v1: VersionedValue, v2: VersionedValue) -> VersionedValue:
        """マージ: 両方の値を保持（ショッピングカート等で有用）

        メリット: データの損失がない
        デメリット: 削除操作の扱いが難しい（Tombstone が必要）
        用途: ショッピングカート、タグ、お気に入りリスト
        """
        if isinstance(v1.value, set) and isinstance(v2.value, set):
            merged = v1.value | v2.value
            print(f"Merge: {v1.value} | {v2.value} = {merged}")
            return VersionedValue(merged, time.time(), "merged", {})
        raise ValueError("Merge not supported for non-set values")

    @staticmethod
    def higher_value_wins(v1: VersionedValue, v2: VersionedValue) -> VersionedValue:
        """数値が大きい方を採用（カウンター等で有用）

        用途: monotonically increasing カウンター（いいね数等）
        """
        if isinstance(v1.value, (int, float)) and isinstance(v2.value, (int, float)):
            winner = v1 if v1.value >= v2.value else v2
            print(f"Higher wins: {winner.value}")
            return winner
        raise ValueError("Numeric values required")

    @staticmethod
    def application_level(v1: VersionedValue, v2: VersionedValue,
                          resolver: Callable) -> VersionedValue:
        """アプリケーション固有のロジックで解決

        用途: 複雑なビジネスルール（注文状態の遷移等）
        """
        return resolver(v1, v2)


# === デモ ===
v1 = VersionedValue("price=100", 1000.001, "node1", {"node1": 1})
v2 = VersionedValue("price=120", 1000.005, "node2", {"node2": 1})
ConflictResolver.last_writer_wins(v1, v2)
# LWW: price=120 wins (from node2)

cart1 = VersionedValue({"itemA", "itemB"}, 1000.0, "node1", {})
cart2 = VersionedValue({"itemB", "itemC"}, 1000.0, "node2", {})
ConflictResolver.merge_values(cart1, cart2)
# Merge: {'itemA', 'itemB'} | {'itemB', 'itemC'} = {'itemA', 'itemB', 'itemC'}

counter1 = VersionedValue(42, 1000.0, "node1", {})
counter2 = VersionedValue(45, 1000.0, "node2", {})
ConflictResolver.higher_value_wins(counter1, counter2)
# Higher wins: 45
```

### コード例6: CRDTs（Conflict-free Replicated Data Types）

```python
from collections import defaultdict

class GCounter:
    """G-Counter (Grow-only Counter) - CRDT

    WHY CRDTs:
      コンフリクト解決を「データ構造レベル」で保証する手法。
      特別な調停ロジックなしに、任意の順序でマージしても
      全ノードが同じ値に収束する（数学的に証明済み）。

    G-Counter の仕組み:
      各ノードが自分のカウンターを持ち、インクリメントは
      自分のカウンターのみ行う。合計は全ノードのカウンターの和。
      マージは各ノードの max を取る → 順序に依存せず収束。
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.counters: dict[str, int] = defaultdict(int)

    def increment(self, amount: int = 1):
        """自分のカウンターのみインクリメント"""
        self.counters[self.node_id] += amount

    def value(self) -> int:
        """全ノードのカウンターの合計"""
        return sum(self.counters.values())

    def merge(self, other: 'GCounter'):
        """マージ: 各ノードの max を採用（順序に依存しない）"""
        all_keys = set(self.counters.keys()) | set(other.counters.keys())
        for key in all_keys:
            self.counters[key] = max(
                self.counters.get(key, 0),
                other.counters.get(key, 0)
            )

    def __repr__(self):
        return f"GCounter({dict(self.counters)}, total={self.value()})"


class PNCounter:
    """PN-Counter (Positive-Negative Counter) - CRDT

    G-Counter の拡張。増加用と減少用の2つのG-Counterを持つ。
    値 = P.value() - N.value()
    """

    def __init__(self, node_id: str):
        self.p = GCounter(node_id)  # 増加用
        self.n = GCounter(node_id)  # 減少用

    def increment(self, amount: int = 1):
        self.p.increment(amount)

    def decrement(self, amount: int = 1):
        self.n.increment(amount)

    def value(self) -> int:
        return self.p.value() - self.n.value()

    def merge(self, other: 'PNCounter'):
        self.p.merge(other.p)
        self.n.merge(other.n)


# === デモ ===
print("=== G-Counter (いいね数) ===")
node_a = GCounter("A")
node_b = GCounter("B")

node_a.increment(3)  # Aで3回いいね
node_b.increment(5)  # Bで5回いいね

# 分断中 → 各ノードは独立にカウント
print(f"A: {node_a}")  # GCounter({'A': 3}, total=3)
print(f"B: {node_b}")  # GCounter({'B': 5}, total=5)

# 分断解消 → マージ
node_a.merge(node_b)
node_b.merge(node_a)
print(f"After merge A: {node_a}")  # GCounter({'A': 3, 'B': 5}, total=8)
print(f"After merge B: {node_b}")  # GCounter({'A': 3, 'B': 5}, total=8)
# → 順序に関わらず、両方が 8 に収束する

print("\n=== PN-Counter (在庫数) ===")
stock_a = PNCounter("A")
stock_b = PNCounter("B")

stock_a.increment(100)  # 100個入荷
stock_a.decrement(3)    # 3個販売（Aで）
stock_b.decrement(5)    # 5個販売（Bで）

stock_a.merge(stock_b)
stock_b.merge(stock_a)
print(f"在庫: {stock_a.value()}")  # 92 (100 - 3 - 5)
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
| etcd | CP | PC/EC | 強い一貫性 (Raft) | 設定管理 |
| CockroachDB | CP | PC/EC | 強い一貫性 | 分散SQL |
| CouchDB | AP | PA/EL | 結果整合性 | オフライン同期 |
| TiDB | CP | PC/EC | 強い一貫性 | HTAP |

### 比較表2: 一貫性モデルの比較

| 一貫性モデル | 強さ | レイテンシ | 実装例 | 適するデータ |
|-------------|------|-----------|--------|-------------|
| 線形化可能性 | 最強 | 最高 | Raft, Paxos | 口座残高、ロック |
| 逐次一貫性 | 強 | 高 | Zab (ZooKeeper) | 設定データ |
| 因果一貫性 | 中 | 中 | ベクタークロック | メッセージング |
| Read-your-writes | 中 | 中 | Sticky Session | ユーザープロフィール |
| 結果整合性 | 弱 | 低 | ゴシッププロトコル | アクセスログ、メトリクス |

### 比較表3: コンフリクト解決戦略の比較

| 戦略 | メリット | デメリット | 用途 |
|------|---------|-----------|------|
| Last-Writer-Wins | シンプル | 時計のずれに脆弱、データ損失 | セッション、キャッシュ |
| ベクタークロック | 因果関係を追跡 | ストレージオーバーヘッド | 汎用 |
| CRDTs | 数学的に収束保証 | 実装できる操作に制限 | カウンター、セット |
| マージ (Union) | データ損失なし | 削除が困難 | カート、タグ |
| アプリケーション固有 | 最も柔軟 | 実装が複雑 | 注文状態、ゲーム |

---

## 6. アンチパターン

### アンチパターン1: CAP定理の誤解「3つから2つ選ぶ」

```python
# NG: 「CAを選ぶ」という誤った判断
class BadDistributedSystem:
    """
    誤解: 「うちはCAを選ぶ。一貫性と可用性を取って分断耐性は捨てる」

    問題:
    1. 分散システムでネットワーク分断は避けられない
    2. 分断耐性を「捨てる」ことはできない（物理的に不可避）
    3. CAは「分散でないシステム」（単一ノード）にしか成立しない
    """
    def __init__(self):
        # 単一ノードなら CA は成立するが、それは分散システムではない
        self.single_db = "postgresql://single-server:5432/db"

# OK: 正しい理解
class GoodDistributedSystem:
    """
    正しい理解:
    1. P は選択肢ではなく前提条件（分断は必ず発生する）
    2. 分断時: CP（一貫性優先）or AP（可用性優先）を選ぶ
    3. さらに、通常時も L vs C のトレードオフが存在する（PACELC）
    4. データの性質ごとに使い分けるのがベストプラクティス
    """
    def __init__(self):
        self.balance_db = "mongodb://..."       # CP: 口座残高
        self.session_cache = "cassandra://..."  # AP: セッション
        self.log_store = "cassandra://..."      # AP: ログ
```

### アンチパターン2: 全てに強い一貫性を要求する

```python
# NG: 金融システムだから全データを強い一貫性で管理
class BadFinancialSystem:
    def __init__(self):
        # 全データに QUORUM 読み書き → レイテンシが全体的に悪化
        self.consistency_level = "QUORUM"

    def get_balance(self, user_id):
        # OK: 残高は強い一貫性が必要
        return self.read(f"balance:{user_id}", cl="QUORUM")

    def get_user_preferences(self, user_id):
        # NG: ユーザー設定にまで QUORUM は不要
        return self.read(f"prefs:{user_id}", cl="QUORUM")

    def write_access_log(self, log_entry):
        # NG: アクセスログにまで QUORUM は不要（レイテンシ悪化）
        return self.write("access_log", log_entry, cl="QUORUM")


# OK: データの性質ごとに一貫性レベルを使い分け
class GoodFinancialSystem:
    """
    データ分類:
    ┌──────────────┬───────────────────┐
    │ データ        │ 一貫性レベル       │
    ├──────────────┼───────────────────┤
    │ 口座残高      │ 強い一貫性（必須） │
    │ 取引履歴      │ 強い一貫性（必須） │
    │ ユーザー設定  │ 結果整合性でOK    │
    │ アクセスログ  │ 結果整合性でOK    │
    │ レコメンド    │ 結果整合性でOK    │
    └──────────────┴───────────────────┘
    """
    def get_balance(self, user_id):
        return self.read(f"balance:{user_id}", cl="QUORUM")

    def get_user_preferences(self, user_id):
        return self.read(f"prefs:{user_id}", cl="ONE")

    def write_access_log(self, log_entry):
        return self.write("access_log", log_entry, cl="ONE")
```

### アンチパターン3: 結果整合性の「いつか」を放置する

```
NG:
「結果整合性だからいつか一貫する」と言い、収束時間を監視しない

問題:
1. ユーザーがプロフィールを更新 → リロードしても反映されない
2. 在庫を0にした → 他のリージョンでまだ注文できる
3. 権限を剥奪 → 数分間はまだアクセスできる

OK:
- 収束時間の SLO を設定（例: 95%のケースで 1秒以内に収束）
- Read-your-writes 一貫性を実装（自分の更新は即反映）
- 重要な操作には Sticky Session を使用
- 楽観的UI更新（書き込み結果をクライアントで即表示）
```

---

## 7. 実践演習

### 演習1（基礎）: クォーラムの計算

以下の構成でクォーラムを計算せよ。

```
問題:
レプリカ数 N=5 のクラスタで以下の設定を分析せよ。

1. W=3, R=3 → 強い一貫性か？ 最大何台のノード障害に耐えられるか？
2. W=1, R=5 → 強い一貫性か？ 書き込みレイテンシはどう変わるか？
3. W=4, R=2 → 強い一貫性か？ 書き込み可用性はどう変わるか？
4. W=1, R=1 → 強い一貫性か？ どのような用途に適するか？
5. 3台のノードが同時にダウンしても読み書き可能な W, R の組み合わせを求めよ。
```

**期待される出力:**

```
1. W=3, R=3: 3+3=6 > 5 → 強い一貫性
   書き込み: 5-3=2台のノード障害に耐えられる
   読み取り: 5-3=2台のノード障害に耐えられる

2. W=1, R=5: 1+5=6 > 5 → 強い一貫性
   書き込みは1台の応答で完了 → 超高速
   読み取りは全5台必要 → 1台でもダウンすると読み取り不可
   用途: Write-heavy ワークロード

3. W=4, R=2: 4+2=6 > 5 → 強い一貫性
   書き込みに4台必要 → 2台以上ダウンすると書き込み不可
   読み取りは2台で十分 → Read-heavy 向き

4. W=1, R=1: 1+1=2 ≤ 5 → 結果整合性
   最速だが、古いデータを読む可能性あり
   用途: ログ、メトリクス、キャッシュ

5. 3台ダウン → 残り2台
   W ≤ 2 かつ R ≤ 2 かつ W+R > 5
   → 不可能（最大 W+R=4 ≤ 5）
   → 3台同時ダウン時は強い一貫性を維持できない
   → W=2, R=2 なら読み書き可能だが結果整合性
```

### 演習2（応用）: コンフリクト検知と解決

以下のシナリオでコンフリクトを検知・解決するコードを実装せよ。

```python
"""
シナリオ: ECサイトのショッピングカート

分断中に2つのノードで同じユーザーのカートが更新された:
- Node A: ユーザーが「商品X」を追加、「商品Y」を削除
- Node B: ユーザーが「商品Z」を追加

分断解消後にどうマージするか？

要件:
1. 追加された商品は全て保持する（Union）
2. 削除された商品は正しく反映する（Tombstone）
3. 数量変更はLWWで解決する
"""

# ここに実装する
```

**期待される出力:**

```
分断前カート: {'itemA': 2, 'itemB': 1}
Node A のカート: {'itemA': 2, 'itemX': 1}  (itemB削除, itemX追加)
Node B のカート: {'itemA': 2, 'itemB': 1, 'itemZ': 3}  (itemZ追加)

マージ結果: {'itemA': 2, 'itemX': 1, 'itemZ': 3}
  - itemA: 両方に存在 → 保持
  - itemB: Node A で削除 → 削除を優先
  - itemX: Node A で追加 → 保持
  - itemZ: Node B で追加 → 保持
```

### 演習3（発展）: マルチリージョンデータベースの設計

以下の要件を満たすデータベース構成を設計せよ。

```
要件:
- リージョン: 東京、シンガポール、北米東部の3リージョン
- ユーザー数: 各リージョン100万人、計300万人
- データの種類:
  a) ユーザー認証情報 → 強い一貫性が必要
  b) ショッピングカート → 結果整合性で許容
  c) 商品カタログ → 読み取り重視、更新は低頻度
  d) 注文データ → 強い一貫性が必要
  e) 推薦データ → 結果整合性で許容

設計課題:
1. 各データの種類に対して CP/AP のどちらを選択するか、理由と共に示せ
2. 使用するデータベースの選定と理由
3. リージョン間のレプリケーション方式
4. 東京-シンガポール間の分断が発生した場合の動作を説明せよ
5. 全体の可用性を推定せよ
```

**期待される出力（概要）:**

```
1. データ種類別の CP/AP 選択:
   a) 認証情報: CP（不正アクセス防止のため一貫性必須）
   b) カート: AP（可用性重視、CRDTsでマージ）
   c) カタログ: AP（レイテンシ優先、短TTLのキャッシュ）
   d) 注文: CP（二重注文防止のため一貫性必須）
   e) 推薦: AP（古い推薦でも問題なし）

2. データベース選定:
   a,d) CockroachDB or Google Spanner（マルチリージョンCP）
   b,e) DynamoDB Global Tables（マルチリージョンAP）
   c) ElastiCache + CDN（読み取り最適化）

3. レプリケーション:
   CP データ: 同期レプリケーション（Raft/Paxos）
   AP データ: 非同期レプリケーション + CRDTs

4. 分断時の動作:
   CP: 少数側リージョンは書き込み拒否、多数側で継続
   AP: 全リージョンで読み書き可能、分断解消後にマージ

5. 可用性推定:
   CP部分: 99.99%（自動フェイルオーバー）
   AP部分: 99.999%（全ノードが応答可能）
   全体: min(99.99%, 99.999%) × 他コンポーネント
```

---

## 8. FAQ

### Q1: NoSQLは全てAPシステムですか？

全くそうではない。MongoDBはCP（Primaryでの書き込みを保証し、分断時にPrimaryを失った側は書き込み不可）、HBaseもCP（ZooKeeperによるリーダー選出で一貫性を保証）である。一方CassandraやDynamoDBはAPだが、クエリごとに一貫性レベルを調整できる。「NoSQL = AP」は誤解であり、データモデル（KV/Document/Column/Graph）とCAP特性は独立した概念である。

### Q2: Google Spannerは「CAPを破った」のですか？

Spannerは「事実上の CA」と言われることがあるが、厳密にはCPシステムである。Googleの専用ネットワーク（冗長な海底ケーブル等）により分断確率を極限まで下げ、TrueTime（原子時計+GPS）により低レイテンシで強い一貫性を実現している。CAP定理を破ったのではなく、Pの発生確率を工学的に極小化したと理解するのが正確である。一般企業が同じアプローチを取ることは現実的ではない。

### Q3: 結果整合性で問題になるケースは？

典型的な問題は「書き込み直後の読み込み」で古いデータが返るケースである。例えば、ユーザーがプロフィール画像を更新した直後にページをリロードすると古い画像が表示される。対策として (1) Read-your-writes一貫性（自分の書き込みは即座に反映）、(2) Sticky Session（同じノードにルーティング）、(3) クライアント側のOptimistic UI（書き込み結果を即座にUIに反映して後で同期）がある。

### Q4: CRDTsとは何ですか？どのような場面で使いますか？

CRDTs（Conflict-free Replicated Data Types）は、コンフリクトが数学的に発生しないデータ構造である。任意の順序でマージしても全レプリカが同じ値に収束することが保証される。G-Counter（加算のみのカウンター）、PN-Counter（加減算可能なカウンター）、G-Set（追加のみのセット）、OR-Set（追加/削除可能なセット）などがある。Riak、Redis（CRDTs対応）、Figmaのリアルタイム共同編集で使われている。

### Q5: 2PC（Two-Phase Commit）ではなく Saga を使うべきなのはなぜですか？

2PCは分散トランザクションの古典的な手法だが、マイクロサービスでは問題がある。(1) コーディネーターがSPOFになる、(2) 参加者がロックを保持し続けるためスループットが低下、(3) コーディネーター障害時に参加者がブロックされる。Saga パターンは各サービスがローカルトランザクションを実行し、失敗時に補償トランザクションで巻き戻す方式で、ロックフリーかつ高可用性を実現する。ただし、結果整合性を受け入れる必要がある。

### Q6: ベクタークロックとは何ですか？

ベクタークロックは分散システムでイベントの因果関係を追跡するための論理時計である。各ノードが「ノード名 → カウンター」のベクター（辞書）を保持し、イベント発生時に自分のカウンターをインクリメントする。2つのベクタークロックを比較することで、「AがBの前に発生した」「BがAの前に発生した」「AとBは並行（因果関係なし）」を判定できる。DynamoDBの内部実装で使われている。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| CAP定理 | C(一貫性)・A(可用性)・P(分断耐性)の3つを同時に満たせない |
| Pは前提 | 分散システムで分断は避けられない。実質 CP or AP の二択 |
| CPシステム | 分断時に一貫性優先。書き込み拒否の可能性。例: MongoDB, etcd |
| APシステム | 分断時に可用性優先。古いデータを返す可能性。例: Cassandra |
| PACELC | 通常時の L vs C のトレードオフも考慮する拡張定理 |
| クォーラム | W+R > N で強い一貫性を実現。バランスの調整が可能 |
| 結果整合性 | ゴシッププロトコル、Read Repair で収束を加速 |
| CRDTs | コンフリクトフリーなデータ構造。カウンター、セット等 |
| コンフリクト解決 | LWW、マージ、CRDTs、アプリケーション固有から選択 |
| 設計指針 | データの性質ごとに一貫性レベルを使い分ける |

---

## 次に読むべきガイド

- [DBスケーリング](../01-components/04-database-scaling.md) -- レプリケーションとシャーディングの実践
- [メッセージキュー](../01-components/02-message-queue.md) -- 非同期処理と結果整合性
- [モノリス vs マイクロサービス](../02-architecture/00-monolith-vs-microservices.md) -- 分散トランザクションと Saga パターン
- [信頼性](./02-reliability.md) -- サーキットブレーカーとフォールトトレランス
- [デザインパターン](../../../03-software-design/design-patterns-guide/docs/04-architectural/) -- アーキテクチャパターン

---

## 参考文献

1. Brewer, E. (2012). "CAP Twelve Years Later: How the 'Rules' Have Changed." *IEEE Computer*, 45(2), 23-29. -- CAP定理の提唱者による再解釈
2. Gilbert, S. & Lynch, N. (2002). "Brewer's conjecture and the feasibility of consistent, available, partition-tolerant web services." *ACM SIGACT News*, 33(2), 51-59. -- CAP定理の形式的証明
3. Abadi, D. (2012). "Consistency Tradeoffs in Modern Distributed Database System Design." *IEEE Computer*, 45(2), 37-42. -- PACELC定理の提唱
4. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Chapter 9: Consistency and Consensus. O'Reilly Media. -- 一貫性モデルとコンセンサスアルゴリズムの解説
5. Shapiro, M. et al. (2011). "Conflict-free Replicated Data Types." *SSS 2011*, Springer. -- CRDTsの原論文

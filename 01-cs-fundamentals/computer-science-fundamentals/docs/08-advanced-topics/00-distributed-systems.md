# 分散システム

> 「分散システムとは、あるマシンの障害によって、あなたが存在すら知らなかった別のマシンが使えなくなるシステムのことである」——Leslie Lamport

## この章で学ぶこと

- [ ] 分散システムの基本概念と、なぜ単一マシンでは限界があるかを理解する
- [ ] CAP定理を正しく理解し、CP/APの選択基準を説明できる
- [ ] 一貫性モデルの強弱とその実用上のトレードオフを把握する
- [ ] Paxos/Raftなどの合意アルゴリズムの原理と使い分けを理解する
- [ ] レプリケーションとシャーディングの戦略を設計に適用できる
- [ ] 分散トランザクション（2PC, Saga）の仕組みと限界を説明できる
- [ ] 論理時計・ベクトル時計による順序付けを実装できる
- [ ] 障害耐性パターン（Circuit Breaker等）を適切に設計できる

---

## 1. なぜ分散システムが必要か

### 1.1 単一マシンの限界

```
単一マシンの限界:

  CPU:     ムーアの法則の鈍化（2005年〜）
           → シングルスレッド性能は頭打ち
           → マルチコア化しても1台のコア数には上限がある

  メモリ:   1台のRAM上限（数TB）
           → 全データをメモリに載せたくても物理限界がある

  ストレージ: 1台のディスク上限（数十TB）
           → ペタバイト級のデータは1台に収まらない

  可用性:   1台が落ちれば全停止（SPOF = Single Point of Failure）
           → ハードウェア故障は確率的に必ず起きる

  ネットワーク帯域: 1台のNICスループット上限（10-100Gbps）
           → 大量クライアントからの同時接続をさばけない

  → 複数マシンに分散して処理する必然性がここにある
```

### 1.2 分散システムの4つの目的

```
分散システムの目的:

  1. スケーラビリティ:
     処理能力の水平拡張（Scale-Out）
     → マシンを追加するだけで処理能力が線形に向上する
     → 垂直拡張（Scale-Up）はコスト効率が悪い
       例: 2倍のCPUのマシンは2倍以上のコストがかかる

  2. 可用性（Availability）:
     一部の障害でもサービスを継続できる
     → 年間稼働率99.99%（= 年間ダウンタイム約52分）を実現
     → 1台のサーバーの年間故障率は2-4%程度
       → 1000台なら毎日数台が壊れる計算になる

  3. レイテンシ:
     ユーザーに物理的に近い場所で処理する
     → 光速の限界: 東京〜ニューヨーク間で片道約70ms
     → CDNやエッジコンピューティングで解決

  4. データ量:
     1台に収まらないデータの管理
     → Googleは数十エクサバイト（EB）のデータを保持
     → YouTubeには毎分500時間の動画がアップロードされる
```

### 1.3 分散システムの代表的な実例

```
実例とその規模:

  Google検索:
    数千台のサーバーが協調して1回の検索クエリを処理
    → MapReduceで大規模インデックス構築
    → Bigtableで分散ストレージ
    → Spannerでグローバル分散DB

  Netflix:
    世界中のCDN + 数百のマイクロサービス群
    → 1日のトラフィックが全インターネットの約15%
    → Chaos Monkeyで意図的に障害を起こし耐性をテスト

  Bitcoin:
    数万ノードが合意形成（Proof of Work）
    → ビザンチン障害耐性を暗号学的手法で実現
    → 中央管理者なしでの信頼構築

  Amazon DynamoDB:
    数百万テーブル、1日数兆リクエスト
    → コンシステントハッシュによるデータ分散
    → 結果整合性と強い一貫性の選択が可能
```

---

## 2. 分散システムの8つの誤解

### 2.1 Peter Deutschの8つの誤解

```
Peter Deutsch の「分散コンピューティングの8つの誤解」(1994):

  1. ネットワークは信頼できる
     → 現実: パケットロス率は0.01%〜1%程度
     → 海底ケーブル切断、ルーター障害は日常的に起きる
     → 対策: リトライ、冪等性、タイムアウト設計

  2. レイテンシはゼロである
     → 現実: 同一DC内でも0.5ms、大陸間は100ms以上
     → 光速の限界は物理法則であり改善不可
     → 対策: データの局所性、キャッシュ、非同期処理

  3. 帯域幅は無限である
     → 現実: ネットワーク飽和は起きる
     → 大量データ転送はディスクで物理輸送する方が速いことも
       （AWS Snowballが存在する理由）
     → 対策: データ圧縮、差分転送、プロトコル最適化

  4. ネットワークは安全である
     → 現実: 中間者攻撃、盗聴、DNSスプーフィング
     → ゼロトラストネットワークの台頭
     → 対策: TLS/mTLS、暗号化、認証

  5. トポロジは変化しない
     → 現実: サーバー追加/削除、障害によるルート変更
     → クラウド環境ではIPアドレスが動的に変わる
     → 対策: サービスディスカバリ、DNS、ロードバランサ

  6. 管理者は1人である
     → 現実: 複数チーム、複数組織、複数クラウドプロバイダ
     → 責任境界が曖昧になりやすい
     → 対策: 明確なAPI契約、SLA定義

  7. 転送コストはゼロである
     → 現実: シリアライズ/デシリアライズのCPUコスト
     → クラウドのデータ転送料金（egress費用）
     → 対策: 効率的なシリアライゼーション（Protocol Buffers等）

  8. ネットワークは均一である
     → 現実: 異なるハードウェア、OS、プロトコルバージョン
     → 10Gbpsと1Gbpsのリンクが混在
     → 対策: 抽象化レイヤー、プロトコルバージョニング
```

### 2.2 誤解が引き起こす典型的な障害

これらの誤解を前提にシステムを設計すると、以下のような障害が発生する。
なぜこれらが問題になるかというと、開発環境では再現しにくく本番環境で初めて
顕在化するためである。

```
誤解1を前提にした場合の障害例:

  開発環境: ローカルネットワークでは99.99%成功
  本番環境: クラウド間通信でパケットロスが発生
  結果: リトライ未実装のため、処理が中途半端な状態で放置
        → データ不整合、顧客からのクレーム

  なぜローカルでは再現しないのか:
  → ローカルネットワークのパケットロス率は0.0001%未満
  → WANでは0.01%〜1%に跳ね上がる
  → 1日100万リクエストなら100〜10000件が失敗する計算
```

---

## 3. CAP定理と一貫性モデル

### 3.1 CAP定理の正確な理解

```
CAP定理（Eric Brewer, 2000 予想 / Gilbert & Lynch, 2002 証明）:

分散データストアは以下の3つの保証のうち、
ネットワーク分断が発生した場合に同時に2つしか満たせない。

  C — Consistency（一貫性）:
      全ノードが同時に同じデータを見る
      → 具体的には「線形化可能性（Linearizability）」を指す
      → 書き込み完了後、どのノードに読みに行っても最新値が返る
      → なぜ難しいか: 全ノードの同期に通信が必要だから

  A — Availability（可用性）:
      障害が起きていないノードは必ず「有効な」レスポンスを返す
      → タイムアウトやエラーではなく、意味のある応答
      → なぜ難しいか: 最新データを持っていなくても応答が必要だから

  P — Partition Tolerance（分断耐性）:
      ネットワーク分断が起きてもシステムが動作し続ける
      → ノード間の通信が途絶えても停止しない
      → なぜ必須か: 分散システムではネットワーク分断は必ず起きるから

  ┌─────────────────────────────────────────────┐
  │              C (一貫性)                       │
  │             / \                               │
  │            /   \                              │
  │           / CP  \ CA                          │
  │          /  |    \ |                          │
  │         / HBase   \ 単一ノードRDBMS           │
  │        / etcd      \(分断が起きない前提)      │
  │       / ZooKeeper   \                         │
  │      /_______________\                        │
  │     A (可用性)    P (分断耐性)                 │
  │          |                                    │
  │        AP: Cassandra, DynamoDB, CouchDB       │
  └─────────────────────────────────────────────┘
```

### 3.2 なぜ「CA」は実質的に存在しないのか

```
CA（一貫性 + 可用性）が実質不可能な理由:

  ネットワーク分断は物理的に避けられない:
  - ケーブル切断、スイッチ障害、ルーター障害
  - クラウドプロバイダのAZ（Availability Zone）間障害
  - 想定される分断発生頻度: 大規模DCで月に数回

  分断が起きた瞬間の二択:
  ┌──────────────────────────────────────────┐
  │  Node-A         ×         Node-B         │
  │  [data=1]    (分断)      [data=1]        │
  │                                          │
  │  Client → Node-A: "data=2に更新"         │
  │                                          │
  │  Node-A: data=2 に更新                   │
  │  Node-B: data=1 のまま（通信不可）        │
  │                                          │
  │  別のClient → Node-B: "dataを読みたい"   │
  │                                          │
  │  選択肢1（CP）: Node-Bはエラーを返す      │
  │    → 一貫性は保つが可用性を犠牲           │
  │                                          │
  │  選択肢2（AP）: Node-Bは data=1 を返す    │
  │    → 可用性は保つが一貫性を犠牲           │
  └──────────────────────────────────────────┘

  → 分断が起きた瞬間、CとAの両立は論理的に不可能
  → 単一ノードRDBMSは「Pを諦めている」のではなく
    「そもそも分散していない」だけ
```

### 3.3 CP vs AP の選択基準

| 判断基準 | CP（一貫性優先）を選ぶ場合 | AP（可用性優先）を選ぶ場合 |
|---------|--------------------------|--------------------------|
| データの性質 | 金銭、在庫、予約枠など不整合が致命的 | いいね数、閲覧数など多少の誤差が許容 |
| 不整合時の最悪シナリオ | 二重課金、座席の二重予約 | 古いフォロワー数の表示 |
| ユーザー体験 | エラーの方がまし | 応答がない方が困る |
| 整合性の回復コスト | 回復が困難/不可能 | 時間経過で自動回復 |
| 代表的なシステム | etcd, ZooKeeper, HBase | Cassandra, DynamoDB, CouchDB |
| 代表的なユースケース | 銀行送金、在庫管理、リーダー選出 | SNS、ショッピングカート、DNS |

### 3.4 一貫性モデルの詳細

```
一貫性の強さ（強い順）:

  1. 線形化可能性（Linearizability）:
     → 全操作がグローバルな単一の順序で実行されたかのように見える
     → リアルタイムの順序を尊重する
     → 「書き込みが完了した瞬間から、全ノードで最新値が読める」
     → 最も強い保証、最もコストが高い
     → 実装手法: 合意アルゴリズム（Raft, Paxos）
     → 例: ZooKeeper, etcd, Spanner

  2. 逐次一貫性（Sequential Consistency）:
     → 全プロセスの操作が何らかの全順序で実行されたかのように見える
     → 各プロセス内の操作順序は保持される
     → ただしリアルタイムの順序は保証しない
     → 線形化可能性との違い: 「壁時計の順序」を尊重しなくてよい

  3. 因果一貫性（Causal Consistency）:
     → 因果関係のある操作の順序は全ノードで同一
     → 因果関係のない操作は異なるノードで異なる順序に見えてよい
     → 因果関係とは: 「AがBの結果に依存する」関係
     → 例: SNSで「投稿→その投稿へのコメント」は因果関係あり
     → 例: MongoDB（デフォルト設定）

  4. 結果整合性（Eventual Consistency）:
     → 更新が停止すれば、十分な時間の後に全ノードが一致する
     → 途中では古いデータが読める可能性がある
     → 「十分な時間」は通常ミリ秒〜数秒
     → 例: DynamoDB（デフォルト）, Cassandra, DNS

  強い ←──────────────────────→ 弱い
  線形化  逐次   因果   結果整合
  遅い  ────────────────────── 速い
  高コスト ──────────────── 低コスト
  実装困難 ──────────────── 実装容易
```

### 3.5 一貫性モデルの比較表

| モデル | 順序保証 | レイテンシ | 可用性 | 主な用途 | 代表的実装 |
|-------|---------|-----------|-------|---------|-----------|
| 線形化可能性 | グローバル全順序 | 高（合意が必要） | 低（分断時停止） | 分散ロック、リーダー選出 | etcd, ZooKeeper |
| 逐次一貫性 | プロセス内順序保持 | 中 | 中 | 共有メモリモデル | CPU/GPUメモリ |
| 因果一貫性 | 因果関係のみ保持 | 低〜中 | 高 | ソーシャルアプリ、協調編集 | MongoDB |
| 結果整合性 | 保証なし | 低 | 最高 | キャッシュ、CDN、DNS | DynamoDB, Cassandra |

### 3.6 PACELC定理: CAPの拡張

```
PACELC定理（Daniel Abadi, 2012）:
CAPの「分断時の選択」に加え、「通常時の選択」も考慮する

  P(Partition)時 → A(Availability) or C(Consistency)
  E(Else=通常時) → L(Latency) or C(Consistency)

  ┌──────────────────────────────────────────────┐
  │  分断時         通常時            分類         │
  │  ─────         ──────           ────          │
  │  PA             EL              PA/EL         │
  │  (可用性優先)   (低レイテンシ)   → DynamoDB    │
  │                                 → Cassandra   │
  │                                               │
  │  PC             EC              PC/EC         │
  │  (一貫性優先)   (一貫性優先)     → HBase       │
  │                                 → VoltDB      │
  │                                               │
  │  PA             EC              PA/EC         │
  │  (可用性優先)   (一貫性優先)     → MongoDB     │
  │                                               │
  │  PC             EL              PC/EL         │
  │  (一貫性優先)   (低レイテンシ)   → PNUTS(Yahoo)│
  └──────────────────────────────────────────────┘

なぜPACELCがCAPより実用的か:
→ CAPは「分断時」しか語らないが、分断は稀なイベント
→ 通常運用時のレイテンシ vs 一貫性のトレードオフの方が
  日常的な設計判断として重要
→ 例: DynamoDBの「強い一貫性読み取り」は通常時の
  レイテンシを犠牲にして一貫性を得るオプション
```

### コード例1: 一貫性モデルのシミュレーション

以下のコードは、結果整合性と強い一貫性の動作の違いをシミュレーションする。
なぜシミュレーションが有用かというと、分散環境の挙動は直感に反することが多く、
コードで動かして確認することが理解の近道だからである。

```python
"""
一貫性モデルシミュレーター
==========================================
結果整合性と強い一貫性の振る舞いの違いを
スレッドを使って再現する。

実行方法: python consistency_simulator.py
依存: 標準ライブラリのみ
"""

import threading
import time
import random
from typing import Dict, List, Optional, Tuple


class Node:
    """分散システムの1ノードを表現するクラス。

    なぜロックが必要か:
    Pythonのdictはスレッドセーフではないため、
    複数スレッドからの同時アクセスでデータ破損が起きうる。
    """

    def __init__(self, node_id: str, latency_ms: float = 0):
        self.node_id = node_id
        self.data: Dict[str, Tuple[str, int]] = {}  # key -> (value, version)
        self.lock = threading.Lock()
        self.latency_ms = latency_ms  # ネットワーク遅延をシミュレート

    def write(self, key: str, value: str, version: int) -> bool:
        """データを書き込む。versionが古い場合は拒否する。"""
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)
        with self.lock:
            current = self.data.get(key)
            if current is None or current[1] < version:
                self.data[key] = (value, version)
                return True
            return False

    def read(self, key: str) -> Optional[Tuple[str, int]]:
        """データを読み取る。"""
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)
        with self.lock:
            return self.data.get(key)


class EventuallyConsistentStore:
    """結果整合性を持つ分散データストア。

    書き込みは1ノードに即座に反映し、他ノードには非同期で伝播する。
    なぜこの方式を採用するか:
    → 書き込みレイテンシを最小化できる
    → 一時的にノード間でデータが不一致になることを許容する
    """

    def __init__(self, node_count: int = 3, replication_delay_ms: float = 100):
        self.nodes = [
            Node(f"node-{i}", latency_ms=random.uniform(1, 5))
            for i in range(node_count)
        ]
        self.replication_delay_ms = replication_delay_ms
        self.version_counter = 0
        self.counter_lock = threading.Lock()

    def _get_next_version(self) -> int:
        with self.counter_lock:
            self.version_counter += 1
            return self.version_counter

    def write(self, key: str, value: str) -> str:
        """プライマリノードに書き込み、バックグラウンドで複製する。"""
        version = self._get_next_version()
        primary = self.nodes[0]
        primary.write(key, value, version)

        # 非同期レプリケーション（バックグラウンドスレッド）
        def replicate():
            time.sleep(self.replication_delay_ms / 1000.0)
            for node in self.nodes[1:]:
                delay = random.uniform(0, self.replication_delay_ms / 1000.0)
                time.sleep(delay)
                node.write(key, value, version)

        thread = threading.Thread(target=replicate, daemon=True)
        thread.start()
        return f"Written to {primary.node_id}: {key}={value} (v{version})"

    def read(self, key: str, node_index: Optional[int] = None) -> str:
        """指定ノード（またはランダムノード）から読み取る。"""
        if node_index is not None:
            node = self.nodes[node_index]
        else:
            node = random.choice(self.nodes)
        result = node.read(key)
        if result is None:
            return f"[{node.node_id}] {key} = (not found)"
        return f"[{node.node_id}] {key} = {result[0]} (v{result[1]})"


class StronglyConsistentStore:
    """強い一貫性（線形化可能性）を持つ分散データストア。

    書き込みは過半数のノードへの反映を待ってから完了とする。
    読み取りも過半数のノードから取得し、最新バージョンを返す。

    なぜ過半数か:
    → N=3ノード、W=2（書き込みクオラム）、R=2（読み取りクオラム）
    → W + R > N を満たすため、読み書きのクオラムが必ず重なる
    → 重なったノードが最新データを持つことが保証される
    """

    def __init__(self, node_count: int = 3):
        self.nodes = [
            Node(f"node-{i}", latency_ms=random.uniform(1, 5))
            for i in range(node_count)
        ]
        self.quorum = node_count // 2 + 1
        self.version_counter = 0
        self.counter_lock = threading.Lock()

    def _get_next_version(self) -> int:
        with self.counter_lock:
            self.version_counter += 1
            return self.version_counter

    def write(self, key: str, value: str) -> str:
        """クオラム数のノードに同期的に書き込む。"""
        version = self._get_next_version()
        success_count = 0
        lock = threading.Lock()

        def write_to_node(node: Node):
            nonlocal success_count
            if node.write(key, value, version):
                with lock:
                    success_count += 1

        threads = []
        for node in self.nodes:
            t = threading.Thread(target=write_to_node, args=(node,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        if success_count >= self.quorum:
            return (
                f"Written to {success_count}/{len(self.nodes)} nodes: "
                f"{key}={value} (v{version}) [COMMITTED]"
            )
        else:
            return (
                f"Write failed: only {success_count}/{len(self.nodes)} "
                f"nodes responded (need {self.quorum}) [ABORTED]"
            )

    def read(self, key: str) -> str:
        """クオラム数のノードから読み取り、最新バージョンを返す。"""
        results: List[Optional[Tuple[str, int]]] = []
        read_nodes: List[str] = []
        lock = threading.Lock()

        def read_from_node(node: Node):
            result = node.read(key)
            with lock:
                results.append(result)
                read_nodes.append(node.node_id)

        threads = []
        for node in self.nodes[:self.quorum]:
            t = threading.Thread(target=read_from_node, args=(node,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        valid_results = [r for r in results if r is not None]
        if not valid_results:
            return f"[quorum={read_nodes}] {key} = (not found)"
        latest = max(valid_results, key=lambda x: x[1])
        return f"[quorum={read_nodes}] {key} = {latest[0]} (v{latest[1]})"


def demo():
    """結果整合性と強い一貫性の違いをデモする。"""
    print("=" * 60)
    print("結果整合性デモ")
    print("=" * 60)
    store = EventuallyConsistentStore(node_count=3, replication_delay_ms=200)
    print(store.write("user:1:name", "Alice"))
    print("\n--- 書き込み直後（レプリケーション完了前）---")
    for i in range(3):
        print(store.read("user:1:name", node_index=i))
    print("\n--- 500ms後（レプリケーション完了後）---")
    time.sleep(0.5)
    for i in range(3):
        print(store.read("user:1:name", node_index=i))

    print("\n" + "=" * 60)
    print("強い一貫性デモ")
    print("=" * 60)
    store2 = StronglyConsistentStore(node_count=3)
    print(store2.write("user:1:name", "Bob"))
    print("\n--- 書き込み直後（クオラム読み取り）---")
    for _ in range(3):
        print(store2.read("user:1:name"))


if __name__ == "__main__":
    demo()
```

想定される出力:

```
============================================================
結果整合性デモ
============================================================
Written to node-0: user:1:name=Alice (v1)

--- 書き込み直後（レプリケーション完了前）---
[node-0] user:1:name = Alice (v1)
[node-1] user:1:name = (not found)      ← まだ伝播していない
[node-2] user:1:name = (not found)      ← まだ伝播していない

--- 500ms後（レプリケーション完了後）---
[node-0] user:1:name = Alice (v1)
[node-1] user:1:name = Alice (v1)       ← 伝播完了
[node-2] user:1:name = Alice (v1)       ← 伝播完了

============================================================
強い一貫性デモ
============================================================
Written to 3/3 nodes: user:1:name=Bob (v1) [COMMITTED]

--- 書き込み直後（クオラム読み取り）---
[quorum=['node-0', 'node-1']] user:1:name = Bob (v1)
[quorum=['node-0', 'node-1']] user:1:name = Bob (v1)
[quorum=['node-0', 'node-1']] user:1:name = Bob (v1)
```

---

## 4. コンセンサスアルゴリズム

### 4.1 コンセンサス問題の本質

```
コンセンサス問題:
  複数のノードが1つの値に合意する

  なぜ難しいか:
  - ノードが故障する可能性がある
  - メッセージが遅延・消失する可能性がある
  - ネットワークが分断する可能性がある
  - ビザンチン障害（悪意あるノード）の可能性がある

  コンセンサスが必要な場面:
  - リーダー選出: 「誰がリーダーか」に全員が合意
  - アトミックブロードキャスト: 「メッセージの順序」に全員が合意
  - 分散ロック: 「誰がロックを保持しているか」に全員が合意
  - 状態機械レプリケーション: 「操作の順序」に全員が合意

FLP不可能性定理（Fischer, Lynch, Paterson, 1985）:

  定理: 非同期ネットワークで1台でも故障する可能性がある場合、
        決定的なコンセンサスアルゴリズムは存在しない

  なぜ不可能か:
  → 非同期ネットワークでは「ノードが遅いだけ」と
    「ノードが故障した」を区別できない
  → 待ち続ければ応答が来るかもしれない（遅延）
  → いつまでも来ないかもしれない（故障）

  実用的な回避策:
  → タイムアウトを導入（非同期の仮定を緩める）
  → 確率的手法（ランダム化アルゴリズム）
  → 障害検出器（不完全でも実用的）
  → Raftはタイムアウトベースのリーダー選出でFLPを回避
```

### 4.2 Paxos

```
Paxos（Leslie Lamport, 1989提案 / 1998公開）:

  役割:
  - Proposer: 値を提案するノード
  - Acceptor: 提案を受け入れる/拒否するノード（過半数が必要）
  - Learner:  合意結果を学習するノード

  2フェーズプロトコル:

  Phase 1 (Prepare):
  Proposer          Acceptor (過半数)
     |-- Prepare(n) -->|
     |<-- Promise(n) --|  「n以上の提案番号にしか応答しない」と約束
                         もし既に別の値を受理済みなら、それを返す

  Phase 2 (Accept):
     |-- Accept(n,v) -->|
     |<-- Accepted -----|  過半数がAcceptすれば合意成立
                          vは: Phase1で受理済みの値があればそれ、
                          なければProposerが自由に選ぶ

  なぜ2フェーズが必要か:
  → Phase 1で「他に合意済みの値がないか」を確認する
  → 確認せずにいきなり値を提案すると、
    複数のProposerが異なる値で合意してしまう可能性がある

  具体例:
  ┌──────────────────────────────────────────────────┐
  │ Proposer-A: Prepare(1) → 過半数からPromise取得     │
  │ Proposer-A: Accept(1, "X") → 過半数がAccept       │
  │ → 合意値 = "X"                                    │
  │                                                    │
  │ Proposer-B: Prepare(2) → 過半数からPromise取得     │
  │   (Acceptorは既にAcceptした "X" を返す)             │
  │ Proposer-B: Accept(2, "X") → "X"で合意を再確認    │
  │ → Proposer-Bが異なる値を提案しても "X" が維持される │
  └──────────────────────────────────────────────────┘

  Paxosの問題点:
  - 実装が非常に複雑（Lamportの原論文は「パート・タイム議会」として
    書かれ、分かりにくいことで有名）
  - ライブロック: 複数Proposerが交互にPrepareすると進行しない
  - Multi-Paxos（連続した値の合意）は論文で曖昧に記述
  - Google Chubby (2006) で実用化されたが、実装者は苦労を報告
```

### 4.3 Raft

```
Raft（Diego Ongaro & John Ousterhout, 2014）:
  「理解しやすいコンセンサス」を明確な目標として設計された

  Paxosとの根本的な違い:
  → Paxosは対称的（どのノードもProposerになれる）
  → Raftは非対称的（Leaderが全てを制御）
  → この非対称性が理解しやすさの鍵

  役割（3種類のみ）:
  - Leader:    全ての書き込みを受け付け、ログを複製する
  - Follower:  Leaderの指示に従い、ログを受け取る
  - Candidate: Leader選挙に立候補中の状態

  3つのサブ問題に分解:

  1. リーダー選出（Leader Election）:
     ┌────────────────────────────────────────────┐
     │  Follower --(タイムアウト)--> Candidate      │
     │  Candidate --(過半数の投票)--> Leader        │
     │  Leader --(定期ハートビート)--> 地位を維持    │
     │  Leader --(障害発生)--> Follower --> ...     │
     │  Candidate --(より高いTermを発見)--> Follower│
     └────────────────────────────────────────────┘

     任期(Term)番号による秩序:
     → Term番号が大きい方が新しい
     → 各Termで最大1人のLeader
     → 古いTermのLeaderは新しいTermを知った瞬間にFollowerに戻る

     なぜランダムタイムアウトを使うか:
     → 全ノードが同時にCandidateになると投票が割れる
     → 150ms〜300msのランダムな範囲でタイムアウトを設定
     → 最初にタイムアウトしたノードが選挙を始め、高確率で当選

  2. ログ複製（Log Replication）:
     Client --> Leader --> Follower群に並行して複製
     過半数が書き込み完了 --> コミット確定

     Leader   [1][2][3][4][5]  ← 全エントリ保持
     Follow-A [1][2][3][4][5]  ← 完全同期
     Follow-B [1][2][3]        ← 遅延中（後で追いつく）
     Follow-C [1][2][3][4]     ← 1つ遅延

     過半数（3/5以上）がエントリ[4]を持つ → コミット確定
     Follow-Bは次のAppendEntriesで[4][5]を受け取る

  3. 安全性（Safety）:
     - コミット済みのエントリは絶対に上書きされない
     - 最も新しいログを持つノードのみLeaderになれる
     → なぜか: 古いログのノードがLeaderになると、
       コミット済みデータが失われる恐れがあるため

  実装例と用途:
  - etcd (Kubernetesのクラスタ状態管理)
  - HashiCorp Consul (サービスディスカバリ)
  - CockroachDB (分散SQL)
  - TiKV (TiDBのストレージエンジン)
```

### 4.4 Paxos vs Raft の比較

| 比較項目 | Paxos | Raft |
|---------|-------|------|
| 設計年 | 1989/1998 | 2014 |
| 設計思想 | 理論的正しさ重視 | 理解しやすさ重視 |
| リーダー | 不要（対称的） | 必須（非対称的） |
| 理解の難易度 | 非常に高い | 中程度 |
| 実装の難易度 | 非常に高い | 中程度 |
| ライブロック | 発生しうる | Leader固定で回避 |
| 性能 | 理論的にはやや有利 | Leaderがボトルネックになりうる |
| 障害耐性 | N/2未満のクラッシュ | N/2未満のクラッシュ |
| 代表実装 | Google Chubby | etcd, Consul |
| 産業での採用 | 減少傾向 | 増加傾向 |

### コード例2: 簡易Raftリーダー選出シミュレーション

以下のコードはRaftのリーダー選出プロセスをシミュレートする。
なぜシミュレーションするかというと、分散合意の「タイムアウトとランダム化による
衝突回避」という概念を、動くコードで体感できるからである。

```python
"""
Raftリーダー選出シミュレーター
==========================================
5ノードのRaftクラスタでリーダー選出が行われる様子を
スレッドベースでシミュレートする。

実行方法: python raft_election.py
依存: 標準ライブラリのみ
"""

import threading
import time
import random
from enum import Enum
from typing import Dict, Optional


class Role(Enum):
    FOLLOWER = "Follower"
    CANDIDATE = "Candidate"
    LEADER = "Leader"


class RaftNode:
    """Raftノードのリーダー選出ロジックを実装するクラス。

    なぜ各ノードが独自のタイムアウトを持つか:
    → 全ノードが同時にCandidateになると投票が分裂する
    → ランダムなタイムアウトにより、1つのノードが先に
      選挙を開始する確率が高くなる
    """

    def __init__(self, node_id: int, cluster: "RaftCluster"):
        self.node_id = node_id
        self.cluster = cluster
        self.role = Role.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[int] = None
        self.leader_id: Optional[int] = None
        self.lock = threading.Lock()
        self.election_timeout = random.uniform(0.15, 0.30)
        self.last_heartbeat = time.time()
        self.alive = True
        self.votes_received = 0

    def request_vote(self, candidate_id: int, term: int) -> bool:
        """投票リクエストに応答する。

        なぜTermの比較が必要か:
        → 古いTermのCandidateに投票すると、
          既に新しいTermで選出されたLeaderと競合する
        """
        with self.lock:
            if not self.alive:
                return False
            if term > self.current_term:
                self.current_term = term
                self.role = Role.FOLLOWER
                self.voted_for = None
            if term == self.current_term and (
                self.voted_for is None or self.voted_for == candidate_id
            ):
                self.voted_for = candidate_id
                self.last_heartbeat = time.time()
                return True
            return False

    def receive_heartbeat(self, leader_id: int, term: int):
        """Leaderからのハートビートを受信する。"""
        with self.lock:
            if not self.alive:
                return
            if term >= self.current_term:
                self.current_term = term
                self.role = Role.FOLLOWER
                self.leader_id = leader_id
                self.voted_for = leader_id
                self.last_heartbeat = time.time()

    def start_election(self):
        """選挙を開始する。"""
        with self.lock:
            if not self.alive or self.role == Role.LEADER:
                return
            self.current_term += 1
            self.role = Role.CANDIDATE
            self.voted_for = self.node_id
            self.votes_received = 1  # 自分に投票
            term = self.current_term
            print(
                f"  [Node-{self.node_id}] 選挙開始 "
                f"(Term={term})"
            )

        # 他の全ノードに投票をリクエスト
        for node in self.cluster.nodes.values():
            if node.node_id != self.node_id:
                if node.request_vote(self.node_id, term):
                    with self.lock:
                        self.votes_received += 1
                        if (
                            self.votes_received
                            > len(self.cluster.nodes) // 2
                            and self.role == Role.CANDIDATE
                        ):
                            self.role = Role.LEADER
                            self.leader_id = self.node_id
                            print(
                                f"  [Node-{self.node_id}] *** "
                                f"Leader当選 *** "
                                f"(Term={self.current_term}, "
                                f"得票={self.votes_received}/"
                                f"{len(self.cluster.nodes)})"
                            )
                            self._send_heartbeats()
                            return

    def _send_heartbeats(self):
        """全Followerにハートビートを送信する。"""
        for node in self.cluster.nodes.values():
            if node.node_id != self.node_id:
                node.receive_heartbeat(self.node_id, self.current_term)

    def run_election_timer(self):
        """選挙タイマーを実行する（バックグラウンドスレッド）。"""
        while self.alive:
            time.sleep(0.05)
            with self.lock:
                if not self.alive:
                    break
                if self.role == Role.LEADER:
                    continue
                elapsed = time.time() - self.last_heartbeat
                if elapsed > self.election_timeout:
                    pass
                else:
                    continue
            self.start_election()
            self.election_timeout = random.uniform(0.15, 0.30)
            with self.lock:
                self.last_heartbeat = time.time()


class RaftCluster:
    """Raftクラスタ全体を管理するクラス。"""

    def __init__(self, node_count: int = 5):
        self.nodes: Dict[int, RaftNode] = {}
        for i in range(node_count):
            self.nodes[i] = RaftNode(i, self)

    def start(self):
        """全ノードの選挙タイマーを開始する。"""
        threads = []
        for node in self.nodes.values():
            t = threading.Thread(
                target=node.run_election_timer, daemon=True
            )
            threads.append(t)
            t.start()
        return threads

    def kill_node(self, node_id: int):
        """ノードを停止する（障害シミュレーション）。"""
        self.nodes[node_id].alive = False
        print(f"\n  [Node-{node_id}] === 障害発生 ===")

    def status(self) -> str:
        """クラスタの現在の状態を返す。"""
        lines = []
        for nid, node in sorted(self.nodes.items()):
            status = "DEAD" if not node.alive else node.role.value
            leader_info = (
                f" (leader=Node-{node.leader_id})"
                if node.leader_id is not None
                else ""
            )
            lines.append(
                f"    Node-{nid}: {status:10s} "
                f"Term={node.current_term}{leader_info}"
            )
        return "\n".join(lines)


def demo():
    """リーダー選出のデモ。"""
    print("=" * 60)
    print("Raft リーダー選出シミュレーション（5ノード）")
    print("=" * 60)

    cluster = RaftCluster(node_count=5)
    print("\n--- 初期状態 ---")
    print(cluster.status())

    print("\n--- 選挙タイマー開始 ---")
    cluster.start()
    time.sleep(0.5)  # 選挙完了を待つ

    print("\n--- 選挙後の状態 ---")
    print(cluster.status())

    # Leaderを障害にする
    leader_id = None
    for nid, node in cluster.nodes.items():
        if node.role == Role.LEADER:
            leader_id = nid
            break

    if leader_id is not None:
        cluster.kill_node(leader_id)
        print("\n--- 新しいLeaderの選出を待機中... ---")
        time.sleep(0.8)  # 再選挙を待つ
        print("\n--- 再選挙後の状態 ---")
        print(cluster.status())

    # 全ノード停止
    for node in cluster.nodes.values():
        node.alive = False


if __name__ == "__main__":
    demo()
```

想定される出力:

```
============================================================
Raft リーダー選出シミュレーション（5ノード）
============================================================

--- 初期状態 ---
    Node-0: Follower   Term=0
    Node-1: Follower   Term=0
    Node-2: Follower   Term=0
    Node-3: Follower   Term=0
    Node-4: Follower   Term=0

--- 選挙タイマー開始 ---
  [Node-2] 選挙開始 (Term=1)
  [Node-2] *** Leader当選 *** (Term=1, 得票=4/5)

--- 選挙後の状態 ---
    Node-0: Follower   Term=1 (leader=Node-2)
    Node-1: Follower   Term=1 (leader=Node-2)
    Node-2: Leader     Term=1 (leader=Node-2)
    Node-3: Follower   Term=1 (leader=Node-2)
    Node-4: Follower   Term=1 (leader=Node-2)

  [Node-2] === 障害発生 ===

--- 新しいLeaderの選出を待機中... ---
  [Node-4] 選挙開始 (Term=2)
  [Node-4] *** Leader当選 *** (Term=2, 得票=3/5)

--- 再選挙後の状態 ---
    Node-0: Follower   Term=2 (leader=Node-4)
    Node-1: Follower   Term=2 (leader=Node-4)
    Node-2: DEAD       Term=1 (leader=Node-2)
    Node-3: Follower   Term=2 (leader=Node-4)
    Node-4: Leader     Term=2 (leader=Node-4)
```

---

## 5. 分散データストア

### 5.1 データ分散の2つの軸

```
データ分散の2つの軸:

  1. レプリケーション（複製）:
     同じデータを複数ノードに複製する
     → 目的: 可用性向上、読み取りスケールアウト
     → トレードオフ: 一貫性の確保が難しくなる

     方式:
     ┌──────────────────────────────────────────────┐
     │ 同期レプリケーション:                          │
     │   Master --write--> Slave1 (ACK待ち)          │
     │                 --> Slave2 (ACK待ち)           │
     │   全SlaveのACKを待ってからClientに応答          │
     │   → 強い一貫性が得られる                       │
     │   → 書き込みレイテンシが高い                   │
     │   → 1台でも遅いSlaveがあると全体が遅延         │
     │                                               │
     │ 非同期レプリケーション:                        │
     │   Master --write--> Slave1 (ACK待たない)       │
     │                 --> Slave2 (ACK待たない)        │
     │   Masterへの書き込み完了で即座にClientに応答    │
     │   → 書き込みレイテンシが低い                   │
     │   → Master障害時にデータロスの可能性あり        │
     │   → Slaveが一時的に古いデータを返す             │
     │                                               │
     │ 準同期（Semi-sync）:                           │
     │   Master --write--> Slave1 (ACK待ち)          │
     │                 --> Slave2 (ACK待たない)        │
     │   少なくとも1つのSlaveのACKを待つ               │
     │   → バランス型: 1コピーは保証される             │
     │   → MySQLの推奨設定                            │
     └──────────────────────────────────────────────┘

  2. パーティショニング/シャーディング（分割）:
     データを複数ノードに分割配置する
     → 目的: 容量のスケール、書き込みのスケール
     → トレードオフ: クロスパーティションクエリが高コスト

     分割方式:
     ┌──────────────────────────────────────────────┐
     │ 範囲分割（Range Partitioning）:                │
     │   Shard1: A-G, Shard2: H-N, Shard3: O-Z      │
     │   → 範囲クエリが効率的（例: 名前がA-Cの全ユーザー）│
     │   → ホットスポットが発生しやすい                │
     │     例: 「S」で始まる名前が多く、Shard3に集中    │
     │                                               │
     │ ハッシュ分割（Hash Partitioning）:              │
     │   Shard = hash(key) % N                        │
     │   → データが均等に分散される                    │
     │   → 範囲クエリが非効率（全シャードをスキャン）   │
     │   → ノード追加/削除時に大量の再配置が必要       │
     │                                               │
     │ コンシステントハッシュ:                         │
     │   ノード追加/削除時に影響範囲が1/Nに限定される  │
     │   → DynamoDB, Cassandraが採用                  │
     │   → 次節で詳細に解説                           │
     └──────────────────────────────────────────────┘
```

### 5.2 コンシステントハッシュの仕組み

```
なぜ従来のハッシュ（hash(key) % N）が問題か:

  N=3 の場合:
    hash("user:1") % 3 = 0 → Node-0
    hash("user:2") % 3 = 1 → Node-1
    hash("user:3") % 3 = 2 → Node-2

  N=4 に変更（ノード追加）:
    hash("user:1") % 4 = 1 → Node-1  ← 移動が必要！
    hash("user:2") % 4 = 2 → Node-2  ← 移動が必要！
    hash("user:3") % 4 = 0 → Node-0  ← 移動が必要！

  → ノード数変更で全データの再配置が発生
  → N台のクラスタでは約(N-1)/N = ほぼ全てが移動

コンシステントハッシュ:
  ハッシュ空間を円（リング）として扱う

        0
        |
   270 -+- 90      ← ハッシュリング（0〜360度）
        |
       180

  ノードをリング上に配置:
        Node-A (30)
       /
  ----*-------*---- Node-B (120)
      |       |
      |       |
  ----*-------*---- Node-C (210)
               \
                Node-D (300)

  キーの配置ルール:
  → キーのハッシュ値からリング上を時計回りに探索
  → 最初に見つかるノードに格納

  ノード追加時の影響:
  → Node-E(165)を追加すると、
    影響を受けるのはNode-B(120)〜Node-E(165)の範囲のみ
  → 全データの約1/N（= 1/5）だけが移動

  仮想ノード（Virtual Node）:
  → 1つの物理ノードに複数の仮想ノードを割り当てる
  → 100〜200個の仮想ノードが一般的
  → なぜ必要か: ノード数が少ないとデータ分布が偏るため
  → 仮想ノードを増やすことで分布が均一に近づく
```

### コード例3: コンシステントハッシュの実装

```python
"""
コンシステントハッシュリングの実装
==========================================
仮想ノード付きのコンシステントハッシュを実装し、
ノード追加/削除時のデータ移動量を確認する。

実行方法: python consistent_hash.py
依存: 標準ライブラリのみ
"""

import hashlib
from bisect import bisect_right
from collections import defaultdict
from typing import Dict, List, Optional


class ConsistentHashRing:
    """コンシステントハッシュリングの実装。

    なぜbisectを使うか:
    → リング上のノード探索は二分探索で O(log N) にできる
    → 線形探索だと O(N) になり、ノード数が多いと遅い

    なぜMD5を使うか:
    → ハッシュ値の均一分布が重要
    → MD5は暗号学的には破られているが、
      ハッシュの均一分布性能は十分
    → SHA-256でも良いが、速度面でMD5が有利
    """

    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}  # hash -> physical_node_name
        self.sorted_keys: List[int] = []
        self.nodes: set = set()

    def _hash(self, key: str) -> int:
        """キーからハッシュ値を計算する。"""
        digest = hashlib.md5(key.encode()).hexdigest()
        return int(digest, 16)

    def add_node(self, node: str):
        """ノードをリングに追加する。

        仮想ノードを使う理由:
        物理ノード3台をそのまま配置すると、リング上の間隔が
        偏り、特定ノードにデータが集中する。仮想ノードで
        リング上にまんべんなく分布させることで解決する。
        """
        self.nodes.add(node)
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:vn{i}"
            h = self._hash(virtual_key)
            self.ring[h] = node
            self.sorted_keys.append(h)
        self.sorted_keys.sort()

    def remove_node(self, node: str):
        """ノードをリングから削除する。"""
        self.nodes.discard(node)
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:vn{i}"
            h = self._hash(virtual_key)
            if h in self.ring:
                del self.ring[h]
                self.sorted_keys.remove(h)

    def get_node(self, key: str) -> Optional[str]:
        """キーを担当するノードを返す。

        リング上でキーのハッシュ値から時計回りに探索し、
        最初に見つかるノードを返す。
        """
        if not self.ring:
            return None
        h = self._hash(key)
        idx = bisect_right(self.sorted_keys, h)
        if idx == len(self.sorted_keys):
            idx = 0  # リングの末端を超えたら先頭に戻る
        return self.ring[self.sorted_keys[idx]]

    def get_distribution(self, keys: List[str]) -> Dict[str, int]:
        """キーの分布を計算する。"""
        dist: Dict[str, int] = defaultdict(int)
        for key in keys:
            node = self.get_node(key)
            if node:
                dist[node] += 1
        return dict(dist)


def demo():
    """コンシステントハッシュの動作デモ。"""
    print("=" * 60)
    print("コンシステントハッシュリング デモ")
    print("=" * 60)

    # テストデータ: 10000個のキー
    keys = [f"user:{i}" for i in range(10000)]

    # --- 3ノードでの分布 ---
    ring = ConsistentHashRing(virtual_nodes=150)
    for node in ["Node-A", "Node-B", "Node-C"]:
        ring.add_node(node)

    dist = ring.get_distribution(keys)
    print("\n--- 3ノード構成 ---")
    for node, count in sorted(dist.items()):
        bar = "#" * (count // 50)
        print(f"  {node}: {count:5d} keys {bar}")

    # データの配置を記録
    original_mapping = {key: ring.get_node(key) for key in keys}

    # --- ノード追加時の影響 ---
    ring.add_node("Node-D")
    new_mapping = {key: ring.get_node(key) for key in keys}

    moved = sum(
        1 for key in keys
        if original_mapping[key] != new_mapping[key]
    )
    print(f"\n--- Node-D追加後 ---")
    dist = ring.get_distribution(keys)
    for node, count in sorted(dist.items()):
        bar = "#" * (count // 50)
        print(f"  {node}: {count:5d} keys {bar}")
    print(f"\n  移動したキー数: {moved}/{len(keys)} "
          f"({moved/len(keys)*100:.1f}%)")
    print(f"  理想値: {100/4:.1f}% (1/N)")

    # --- ノード削除時の影響 ---
    original_mapping_4 = {key: ring.get_node(key) for key in keys}
    ring.remove_node("Node-B")
    after_removal = {key: ring.get_node(key) for key in keys}

    moved = sum(
        1 for key in keys
        if original_mapping_4[key] != after_removal[key]
    )
    print(f"\n--- Node-B削除後 ---")
    dist = ring.get_distribution(keys)
    for node, count in sorted(dist.items()):
        bar = "#" * (count // 50)
        print(f"  {node}: {count:5d} keys {bar}")
    print(f"\n  移動したキー数: {moved}/{len(keys)} "
          f"({moved/len(keys)*100:.1f}%)")

    # --- 仮想ノード数による分布の違い ---
    print(f"\n--- 仮想ノード数による分布の標準偏差 ---")
    for vn_count in [1, 10, 50, 150, 500]:
        r = ConsistentHashRing(virtual_nodes=vn_count)
        for node in ["Node-A", "Node-B", "Node-C"]:
            r.add_node(node)
        d = r.get_distribution(keys)
        values = list(d.values())
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        stddev = variance ** 0.5
        print(
            f"  vn={vn_count:4d}: "
            f"stddev={stddev:7.1f} "
            f"(偏りの小ささ: "
            f"{'##' * max(1, int(10 - stddev / 50))})"
        )


if __name__ == "__main__":
    demo()
```

想定される出力:

```
============================================================
コンシステントハッシュリング デモ
============================================================

--- 3ノード構成 ---
  Node-A:  3342 keys ##################################################################
  Node-B:  3298 keys #################################################################
  Node-C:  3360 keys ###################################################################

--- Node-D追加後 ---
  Node-A:  2510 keys ##################################################
  Node-B:  2485 keys #################################################
  Node-C:  2522 keys ##################################################
  Node-D:  2483 keys #################################################

  移動したキー数: 2483/10000 (24.8%)
  理想値: 25.0% (1/N)

--- Node-B削除後 ---
  Node-A:  3356 keys ###################################################################
  Node-B削除 → Node-Bのデータは隣接ノードに引き継がれる
  Node-C:  3322 keys ##################################################################
  Node-D:  3322 keys ##################################################################

  移動したキー数: 2485/10000 (24.9%)

--- 仮想ノード数による分布の標準偏差 ---
  vn=   1: stddev= 2187.3 (偏りの小ささ: )
  vn=  10: stddev=  634.2 (偏りの小ささ: ########)
  vn=  50: stddev=  198.7 (偏りの小ささ: ############)
  vn= 150: stddev=   55.1 (偏りの小ささ: ##################)
  vn= 500: stddev=   28.3 (偏りの小ささ: ####################)
```

### 5.3 クオラムによる一貫性制御

```
クオラム（Quorum）:
  読み書きに必要なノード数を調整して一貫性を制御する

  N = レプリカ数（レプリケーションファクター）
  W = 書き込み時に確認するノード数（Write Quorum）
  R = 読み取り時に確認するノード数（Read Quorum）

  一貫性の条件: W + R > N
  → 読み書きのノード集合が必ず重なる
  → 重なったノードが最新データを持つ

  例: N=3 の場合

  ┌────────────────────────────────────────────┐
  │ 設定1: W=2, R=2  → 強い一貫性              │
  │   Write: Node-A, Node-B に書き込み          │
  │   Read:  Node-B, Node-C から読み取り        │
  │   → Node-Bが重なる → 最新値が必ず見つかる   │
  │                                             │
  │ 設定2: W=3, R=1  → 書き込み重視             │
  │   Write: 全ノードに書き込み（遅い）          │
  │   Read:  1ノードから読み取り（速い）          │
  │   → 読み取り頻度が高い場合に有利             │
  │                                             │
  │ 設定3: W=1, R=3  → 読み取り重視             │
  │   Write: 1ノードに書き込み（速い）           │
  │   Read:  全ノードから読み取り（遅い）         │
  │   → 書き込み頻度が高い場合に有利             │
  │                                             │
  │ 設定4: W=1, R=1  → 結果整合性               │
  │   W + R = 2 ≤ N=3 → 重ならない可能性あり    │
  │   → 読み取り時に古いデータが見える可能性     │
  │   → 最も高速だが一貫性は弱い                 │
  └────────────────────────────────────────────┘
```

---

## 6. 分散トランザクション

### 6.1 分散環境でのACIDの困難さ

```
単一DBのACID:
  A(Atomicity):    全操作が成功するか、全て失敗するか
  C(Consistency):  制約（外部キー等）を常に満たす
  I(Isolation):    並行トランザクションが干渉しない
  D(Durability):   コミットしたデータは永続化される

なぜ分散環境でACIDが困難か:

  Atomicity の問題:
  → 複数ノードにまたがる操作で、一部だけ成功する可能性
  → 例: Node-Aでの引き落としは成功、Node-Bへの入金が失敗
  → 物理的に離れたノード間で「全か無か」を保証するのは高コスト

  Isolation の問題:
  → 分散ロックのコストが非常に高い
  → ネットワーク遅延によりロック保持時間が長くなる
  → デッドロック検出が単一DB以上に複雑

  → これらの問題に対処する2つの主要アプローチ:
    1. 2フェーズコミット（2PC）: 分散ACIDを実現
    2. Sagaパターン: ACIDを緩めて実用性を確保
```

### 6.2 2フェーズコミット（2PC）

```
2フェーズコミット（Two-Phase Commit）:

  Coordinator         Participant-A    Participant-B
       |                    |                |
  Phase 1 (投票フェーズ):
       |-- Prepare -------->|                |
       |-- Prepare ----------------------->|
       |<-- Vote YES -------|                |
       |<-- Vote YES ----------------------|
       |                    |                |
  Phase 2 (決定フェーズ):
       |-- Commit --------->|                |
       |-- Commit ------------------------>|
       |<-- ACK ------------|                |
       |<-- ACK ----------------------------|

  各フェーズの意味:

  Phase 1 (Prepare):
  → Coordinatorが「この操作をコミットできるか？」と聞く
  → Participantは必要なロックを取得し、WALに書き込み
  → YES（準備完了）またはNO（不可）を返す
  → YESを返した後は、結果を聞くまでロックを保持し続ける

  Phase 2 (Commit/Abort):
  → 全員がYES → Coordinatorが「Commit」を指示
  → 1人でもNO → Coordinatorが「Abort」を指示
  → Participantはロックを解放
```

### 6.3 2PCの致命的な問題点

```
2PCの問題:

  1. ブロッキング問題:
     Phase 1でYESを返した後、Phase 2を受け取る前に
     Coordinatorが障害で停止した場合:

     Coordinator   ×  (障害)
     Participant-A: YES を返した状態でブロック
     Participant-B: YES を返した状態でブロック

     → ロックを保持したまま、Commit/Abortのどちらか分からない
     → Coordinatorが復旧するまで全体がブロック
     → この間、ロックされたデータに他のトランザクションがアクセスできない

  2. 性能問題:
     → 2回のラウンドトリップが必要（Prepare + Commit）
     → ネットワークレイテンシが2倍
     → 全Participantの応答を待つため、最も遅いノードに律速される

  3. 可用性問題:
     → Participantの1台でも応答しなければAbort
     → ノード数が増えるほど障害確率が上がる
     → 10ノードで各99.9%の可用性 → 全体は99.0%に低下

  対策: 3フェーズコミット（3PC）
  → Pre-Commit フェーズを追加してブロッキングを軽減
  → しかしネットワーク分断には対応できず、実用性は限定的
```

### 6.4 Sagaパターン

```
Sagaパターン（Hector Garcia-Molina, 1987）:

  基本概念:
  分散トランザクションを一連のローカルトランザクションに分解し、
  各ステップに対応する「補償トランザクション」を用意する

  正常フロー:
  T1 → T2 → T3 → 完了（成功）

  障害フロー（T3で失敗）:
  T1 → T2 → T3(失敗) → C2 → C1
  （Cは補償トランザクション = 取り消し操作）

  ECサイトの注文処理の例:
  ┌──────────────────────────────────────────┐
  │ ステップ     正常操作        補償操作      │
  │ ──────     ──────────    ──────────     │
  │ T1         在庫を確保      C1: 在庫を戻す  │
  │ T2         決済を実行      C2: 返金する    │
  │ T3         配送を手配      C3: 配送キャンセル│
  └──────────────────────────────────────────┘

  2つの実装パターン:

  オーケストレーション型:
  ┌─────────────────────────────────────────────┐
  │  Saga Orchestrator                           │
  │       |                                      │
  │       |--> 在庫Service --> (成功)             │
  │       |--> 決済Service --> (成功)             │
  │       |--> 配送Service --> (失敗)             │
  │       |                                      │
  │       |--> 決済Service.compensate()          │
  │       |--> 在庫Service.compensate()          │
  │                                              │
  │  利点: フローが1箇所で管理される              │
  │  欠点: Orchestratorが単一障害点になりうる      │
  └─────────────────────────────────────────────┘

  コレオグラフィー型:
  ┌─────────────────────────────────────────────┐
  │  在庫Service --[在庫確保済]--> Event Bus      │
  │  決済Service <--[在庫確保済]--                │
  │  決済Service --[決済完了]--> Event Bus        │
  │  配送Service <--[決済完了]--                  │
  │  配送Service --[配送失敗]--> Event Bus        │
  │  決済Service <--[配送失敗]-- → 返金実行       │
  │  在庫Service <--[返金完了]-- → 在庫戻し       │
  │                                              │
  │  利点: 疎結合、中央管理不要                   │
  │  欠点: フロー全体の把握が困難、デバッグが難しい│
  └─────────────────────────────────────────────┘
```

### コード例4: Sagaパターンの実装

```python
"""
Sagaパターン（オーケストレーション型）の実装
==========================================
ECサイトの注文処理をSagaパターンで実装する。
障害発生時の補償トランザクションの動作を確認できる。

実行方法: python saga_pattern.py
依存: 標準ライブラリのみ
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional
import random


class StepStatus(Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    COMPENSATED = "COMPENSATED"


@dataclass
class SagaStep:
    """Sagaの1ステップを表現する。

    なぜ正常操作と補償操作をペアにするか:
    → ステップが失敗した場合、それまでの全ステップを
      逆順に補償する必要がある
    → ペアにしておくことで、どの操作をどう取り消すか
      が明確になる
    """
    name: str
    action: Callable[[], bool]
    compensate: Callable[[], bool]
    status: StepStatus = StepStatus.PENDING


@dataclass
class SagaResult:
    success: bool
    steps_completed: int
    steps_compensated: int
    log: List[str] = field(default_factory=list)


class SagaOrchestrator:
    """Sagaオーケストレータ。

    ステップを順番に実行し、失敗時は逆順に補償する。
    なぜ逆順で補償するか:
    → 後のステップほど前のステップの結果に依存している
    → 依存関係を壊さないためには、後ろから取り消す必要がある
    """

    def __init__(self):
        self.steps: List[SagaStep] = []
        self.log: List[str] = []

    def add_step(
        self,
        name: str,
        action: Callable[[], bool],
        compensate: Callable[[], bool],
    ):
        """ステップを追加する。"""
        self.steps.append(SagaStep(name, action, compensate))

    def execute(self) -> SagaResult:
        """Sagaを実行する。"""
        completed_steps: List[SagaStep] = []

        self.log.append("=== Saga 実行開始 ===")

        for step in self.steps:
            self.log.append(f"  実行中: {step.name}")
            try:
                success = step.action()
            except Exception as e:
                self.log.append(f"  例外発生: {step.name} - {e}")
                success = False

            if success:
                step.status = StepStatus.COMPLETED
                completed_steps.append(step)
                self.log.append(f"  完了: {step.name}")
            else:
                step.status = StepStatus.FAILED
                self.log.append(f"  失敗: {step.name}")
                self.log.append("=== 補償トランザクション開始 ===")

                # 逆順に補償
                compensated = 0
                for completed in reversed(completed_steps):
                    self.log.append(f"  補償中: {completed.name}")
                    try:
                        completed.compensate()
                        completed.status = StepStatus.COMPENSATED
                        compensated += 1
                        self.log.append(
                            f"  補償完了: {completed.name}"
                        )
                    except Exception as e:
                        self.log.append(
                            f"  補償失敗: {completed.name} - {e}"
                        )

                self.log.append("=== Saga 失敗（ロールバック完了）===")
                return SagaResult(
                    success=False,
                    steps_completed=len(completed_steps),
                    steps_compensated=compensated,
                    log=self.log,
                )

        self.log.append("=== Saga 成功 ===")
        return SagaResult(
            success=True,
            steps_completed=len(completed_steps),
            steps_compensated=0,
            log=self.log,
        )


# --- ECサイト注文処理のシミュレーション ---

class InventoryService:
    """在庫サービス。"""

    def __init__(self):
        self.stock = {"item-A": 10, "item-B": 5}
        self.reserved = {}

    def reserve(self, item_id: str, qty: int) -> bool:
        """在庫を確保する。"""
        if self.stock.get(item_id, 0) >= qty:
            self.stock[item_id] -= qty
            self.reserved[item_id] = (
                self.reserved.get(item_id, 0) + qty
            )
            print(
                f"    [在庫] {item_id} x{qty} 確保 "
                f"(残: {self.stock[item_id]})"
            )
            return True
        print(f"    [在庫] {item_id} 在庫不足")
        return False

    def release(self, item_id: str, qty: int) -> bool:
        """在庫確保を取り消す。"""
        self.stock[item_id] = self.stock.get(item_id, 0) + qty
        self.reserved[item_id] = max(
            0, self.reserved.get(item_id, 0) - qty
        )
        print(
            f"    [在庫] {item_id} x{qty} 戻し "
            f"(残: {self.stock[item_id]})"
        )
        return True


class PaymentService:
    """決済サービス。"""

    def __init__(self, fail_probability: float = 0.0):
        self.transactions = []
        self.fail_probability = fail_probability

    def charge(self, user_id: str, amount: int) -> bool:
        """決済を実行する。"""
        if random.random() < self.fail_probability:
            print(f"    [決済] {user_id} ¥{amount} 失敗（残高不足）")
            return False
        self.transactions.append(
            {"user": user_id, "amount": amount, "type": "charge"}
        )
        print(f"    [決済] {user_id} ¥{amount} 成功")
        return True

    def refund(self, user_id: str, amount: int) -> bool:
        """返金を実行する。"""
        self.transactions.append(
            {"user": user_id, "amount": -amount, "type": "refund"}
        )
        print(f"    [決済] {user_id} ¥{amount} 返金完了")
        return True


class ShippingService:
    """配送サービス。"""

    def __init__(self, fail_probability: float = 0.0):
        self.shipments = []
        self.fail_probability = fail_probability

    def schedule(self, order_id: str, address: str) -> bool:
        """配送を手配する。"""
        if random.random() < self.fail_probability:
            print(
                f"    [配送] 注文{order_id} → {address} "
                f"手配失敗（配送業者エラー）"
            )
            return False
        self.shipments.append({"order": order_id, "address": address})
        print(f"    [配送] 注文{order_id} → {address} 手配完了")
        return True

    def cancel(self, order_id: str) -> bool:
        """配送をキャンセルする。"""
        self.shipments = [
            s for s in self.shipments if s["order"] != order_id
        ]
        print(f"    [配送] 注文{order_id} キャンセル完了")
        return True


def demo():
    """Sagaパターンのデモ。"""
    # --- 正常ケース ---
    print("=" * 60)
    print("Sagaパターン デモ: 正常ケース")
    print("=" * 60)

    inv = InventoryService()
    pay = PaymentService()
    ship = ShippingService()

    saga = SagaOrchestrator()
    saga.add_step(
        "在庫確保",
        lambda: inv.reserve("item-A", 2),
        lambda: inv.release("item-A", 2),
    )
    saga.add_step(
        "決済実行",
        lambda: pay.charge("user-1", 5000),
        lambda: pay.refund("user-1", 5000),
    )
    saga.add_step(
        "配送手配",
        lambda: ship.schedule("order-1", "東京都渋谷区..."),
        lambda: ship.cancel("order-1"),
    )

    result = saga.execute()
    print("\n".join(result.log))

    # --- 障害ケース（配送失敗） ---
    print("\n" + "=" * 60)
    print("Sagaパターン デモ: 配送失敗ケース")
    print("=" * 60)

    inv2 = InventoryService()
    pay2 = PaymentService()
    ship2 = ShippingService(fail_probability=1.0)  # 必ず失敗

    saga2 = SagaOrchestrator()
    saga2.add_step(
        "在庫確保",
        lambda: inv2.reserve("item-A", 2),
        lambda: inv2.release("item-A", 2),
    )
    saga2.add_step(
        "決済実行",
        lambda: pay2.charge("user-2", 3000),
        lambda: pay2.refund("user-2", 3000),
    )
    saga2.add_step(
        "配送手配",
        lambda: ship2.schedule("order-2", "大阪市北区..."),
        lambda: ship2.cancel("order-2"),
    )

    result2 = saga2.execute()
    print("\n".join(result2.log))
    print(f"\n  在庫状態: {inv2.stock}")
    print(f"  決済履歴: {pay2.transactions}")


if __name__ == "__main__":
    demo()
```

想定される出力:

```
============================================================
Sagaパターン デモ: 正常ケース
============================================================
    [在庫] item-A x2 確保 (残: 8)
    [決済] user-1 ¥5000 成功
    [配送] 注文order-1 → 東京都渋谷区... 手配完了
=== Saga 実行開始 ===
  実行中: 在庫確保
  完了: 在庫確保
  実行中: 決済実行
  完了: 決済実行
  実行中: 配送手配
  完了: 配送手配
=== Saga 成功 ===

============================================================
Sagaパターン デモ: 配送失敗ケース
============================================================
    [在庫] item-A x2 確保 (残: 8)
    [決済] user-2 ¥3000 成功
    [配送] 注文order-2 → 大阪市北区... 手配失敗（配送業者エラー）
=== Saga 実行開始 ===
  実行中: 在庫確保
  完了: 在庫確保
  実行中: 決済実行
  完了: 決済実行
  実行中: 配送手配
  失敗: 配送手配
=== 補償トランザクション開始 ===
    [決済] user-2 ¥3000 返金完了
  補償中: 決済実行
  補償完了: 決済実行
    [在庫] item-A x2 戻し (残: 10)
  補償中: 在庫確保
  補償完了: 在庫確保
=== Saga 失敗（ロールバック完了）===

  在庫状態: {'item-A': 10, 'item-B': 5}   ← 元に戻っている
  決済履歴: [{'user': 'user-2', 'amount': 3000, 'type': 'charge'},
             {'user': 'user-2', 'amount': -3000, 'type': 'refund'}]
```

### 6.5 2PC vs Saga の比較

| 比較項目 | 2PC | Saga |
|---------|-----|------|
| 一貫性 | 強い（ACID） | 結果整合性 |
| 可用性 | 低い（ブロッキング） | 高い |
| 性能 | 低い（2RTT + ロック保持） | 高い |
| 実装複雑度 | 中 | 高（補償ロジックの設計が困難） |
| 障害耐性 | Coordinator障害に弱い | 各ステップ独立で耐性が高い |
| 隔離性 | 保証される | 保証されない（ダーティリードの可能性） |
| 適用場面 | DB間トランザクション | マイクロサービス間の長いトランザクション |

<!-- SPLIT_POINT_2: ここから第7章〜参考文献 -->

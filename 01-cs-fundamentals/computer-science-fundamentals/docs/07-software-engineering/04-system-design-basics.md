# システム設計入門

> システム設計は「正解のない」問題であり、トレードオフの中で最善の選択をする技術である。

## この章で学ぶこと

- [ ] スケーラビリティの基本概念を理解する
- [ ] CAP定理を説明できる
- [ ] 主要なシステム設計パターンを知る
- [ ] ロードバランシングの仕組みと種類を理解する
- [ ] キャッシング戦略を適切に選択できる
- [ ] データベースの設計パターンを学ぶ
- [ ] メッセージキューの活用方法を理解する
- [ ] マイクロサービスアーキテクチャの長所と短所を把握する
- [ ] APIの設計原則を学ぶ
- [ ] 可用性と信頼性の設計手法を身につける
- [ ] 実践的なシステム設計の見積もり手法を習得する

---

## 1. スケーラビリティ

### 1.1 スケールの方向

```
スケールの2つの方向:

  垂直スケーリング（スケールアップ）:
  → マシンを強化（CPU, RAM追加）
  → 限界あり、ダウンタイムが発生
  → 単純だが高価

  水平スケーリング（スケールアウト）:
  → マシンを追加
  → 理論上無限にスケール
  → 複雑だがコスト効率が良い

  典型的なWebアーキテクチャ:
  ┌────────┐   ┌──────────────┐   ┌──────────┐
  │ Client │──→│ Load Balancer│──→│ Web x N  │
  └────────┘   └──────────────┘   └────┬─────┘
                                       │
                              ┌────────┼────────┐
                              ▼        ▼        ▼
                          ┌──────┐ ┌──────┐ ┌──────┐
                          │Cache │ │ DB   │ │Queue │
                          │Redis │ │Master│ │SQS   │
                          └──────┘ │Slave │ └──────┘
                                   └──────┘
```

### 1.2 垂直スケーリングの実際

```
垂直スケーリング（スケールアップ）の特徴:

  利点:
  - 実装が最も簡単（アプリケーション変更不要）
  - 分散システムの複雑さを回避できる
  - 単一ノードなのでデータ一貫性が自然に保たれる
  - 運用が容易（監視対象が少ない）

  限界:
  - ハードウェアの物理的限界がある
    - CPU: 最大 128〜256 コア
    - RAM: 最大 6〜12 TB
    - ストレージ: IOPS の上限
  - コストが指数関数的に増加
    - 2倍の性能 ≠ 2倍のコスト、通常は3〜5倍
  - 単一障害点（SPOF）になる
  - スケールアップ時にダウンタイムが発生する

  適用すべき場面:
  - トラフィックが比較的小さい場合（QPS < 1,000 程度）
  - プロジェクト初期でまだ規模が読めない場合
  - データベースの一時的なパフォーマンス改善
  - 短期的なトラフィック増加への対応

  具体例:
  - AWS EC2: t3.micro → m5.24xlarge（96 vCPU, 384 GB RAM）
  - RDS: db.t3.micro → db.r5.24xlarge
  - Azure: Standard_B1s → Standard_M128ms（128 vCPU, 3.8 TB RAM）
```

### 1.3 水平スケーリングの実際

```
水平スケーリング（スケールアウト）の特徴:

  利点:
  - 理論上無限にスケール可能
  - コスト効率が良い（コモディティハードウェアを使用）
  - 耐障害性が高い（1台が落ちても他が動作）
  - ダウンタイムなしでスケール変更可能

  課題:
  - 分散システムの複雑さ
    - データの一貫性の保証
    - 分散トランザクション
    - ネットワーク障害の対応
  - ステートの管理
    - セッション管理（Sticky Session vs 共有ストア）
    - キャッシュの一貫性
  - デプロイとオペレーションの複雑化

  スケールアウトで考慮すべきこと:
  ┌──────────────────────────────────────────────────┐
  │ レイヤー       │ 対策                             │
  ├──────────────────────────────────────────────────┤
  │ Web/API       │ ステートレス化 + LB               │
  │ セッション     │ Redis/Memcached に外出し          │
  │ データベース   │ Read Replica + シャーディング       │
  │ ファイル       │ S3/GCS 等のオブジェクトストレージ   │
  │ タスク処理     │ メッセージキュー + ワーカー         │
  │ 検索           │ Elasticsearch/Solr               │
  └──────────────────────────────────────────────────┘
```

### 1.4 ステートレスアーキテクチャ

```python
# --- ステートフル vs ステートレスサーバー ---

# ❌ ステートフル（スケールアウトが困難）
class StatefulServer:
    def __init__(self):
        self.sessions = {}  # セッションをサーバーメモリに保持

    def login(self, user_id, password):
        session_id = generate_session_id()
        self.sessions[session_id] = {
            "user_id": user_id,
            "login_time": datetime.now()
        }
        return session_id

    def get_user(self, session_id):
        # このサーバーにしかセッション情報がない!
        session = self.sessions.get(session_id)
        if not session:
            raise AuthenticationError("Invalid session")
        return session["user_id"]

# ✅ ステートレス（自由にスケールアウト可能）
class StatelessServer:
    def __init__(self, session_store):
        # セッションは外部ストア（Redis等）に保存
        self.session_store = session_store

    def login(self, user_id, password):
        session_id = generate_session_id()
        self.session_store.set(session_id, {
            "user_id": user_id,
            "login_time": datetime.now().isoformat()
        }, ttl=3600)  # 1時間で期限切れ
        return session_id

    def get_user(self, session_id):
        # どのサーバーからでもセッションを参照可能
        session = self.session_store.get(session_id)
        if not session:
            raise AuthenticationError("Invalid session")
        return session["user_id"]
```

```
ステートレス化のためのパターン:

  1. セッション外部化
     - Redis/Memcached にセッションを保存
     - JWT トークンでサーバー側にセッション不要に

  2. ファイルストレージの外部化
     - ユーザーアップロードは S3/GCS に保存
     - ローカルディスクに依存しない

  3. 設定の外部化
     - 環境変数
     - Consul, etcd 等の設定サービス
     - AWS Parameter Store / Secrets Manager

  4. キャッシュの外部化
     - Redis/Memcached をキャッシュレイヤーに
     - ローカルキャッシュは揮発性データのみに使用

  ┌─────────────────────────────────────────────┐
  │              Load Balancer                   │
  │   ┌─────────┬─────────┬─────────┐           │
  │   │ Server1 │ Server2 │ Server3 │ ← 任意に  │
  │   │ (無状態) │ (無状態) │ (無状態) │   追加可能 │
  │   └────┬────┴────┬────┴────┬────┘           │
  │        └─────────┼─────────┘                 │
  │                  ▼                           │
  │         ┌──────────────┐                     │
  │         │ Redis/共有DB  │ ← 状態はここに集約   │
  │         └──────────────┘                     │
  └─────────────────────────────────────────────┘
```

---

## 2. CAP定理

### 2.1 CAP定理の基本

```
CAP定理: 分散システムは3つのうち2つしか同時に保証できない

  C — Consistency（一貫性）: 全ノードが同時に同じデータを見る
  A — Availability（可用性）: 全リクエストがレスポンスを返す
  P — Partition Tolerance（分断耐性）: ネットワーク分断でも動作

  ネットワーク分断は避けられない → 実質 CP or AP の選択

  CP（一貫性優先）: 分断時にエラーを返す
  → 銀行送金、在庫管理
  → PostgreSQL, MongoDB(デフォルト), ZooKeeper

  AP（可用性優先）: 分断時に古いデータを返す可能性
  → SNSのタイムライン、ショッピングカート
  → Cassandra, DynamoDB, CouchDB

  PACELC定理（CAP拡張）:
  分断時(P): AかCを選択
  通常時(E): Latency(L)かConsistency(C)を選択
```

### 2.2 一貫性モデルの詳細

```
一貫性モデルの種類（強い順）:

  1. 線形化可能性（Linearizability）
     - 最も強い一貫性保証
     - 全操作が単一のグローバルな順序で実行されたかのように見える
     - 書き込み直後にどのノードからでも読める
     - 例: Zookeeper, etcd
     - コスト: レイテンシーが最も高い

  2. 逐次一貫性（Sequential Consistency）
     - 各プロセスの操作順序は維持される
     - プロセス間の順序は保証しない
     - 例: 分散キュー

  3. 因果一貫性（Causal Consistency）
     - 因果関係のある操作は順序が保証される
     - 因果関係のない操作は任意の順序で見える
     - 例: メッセージアプリ（返信は元メッセージの後に見える）

  4. 結果整合性（Eventual Consistency）
     - 最も弱い保証
     - 書き込みが停止すれば、いずれ全ノードが同じ値に収束
     - 読み取り時に古い値が返る可能性がある
     - 例: DNS, S3, DynamoDB（デフォルト設定）
     - コスト: レイテンシーが最も低い

  実務での選択指針:
  ┌──────────────────────────────────────────────────┐
  │ ユースケース           │ 推奨一貫性モデル          │
  ├──────────────────────────────────────────────────┤
  │ 銀行振込              │ 線形化可能性（強一貫性）   │
  │ 在庫管理              │ 線形化可能性 or 因果一貫性 │
  │ ユーザープロフィール    │ 結果整合性               │
  │ SNSのいいね数          │ 結果整合性               │
  │ メッセージ送信         │ 因果一貫性               │
  │ ECの注文処理           │ 線形化可能性             │
  │ 検索インデックス更新    │ 結果整合性               │
  │ リーダー選出           │ 線形化可能性             │
  └──────────────────────────────────────────────────┘
```

### 2.3 データレプリケーション

```
レプリケーション戦略:

  1. シングルリーダー（Single-Leader）
     ┌──────────┐    ┌───────────┐
     │  Leader   │───→│ Follower1 │  書き込みはLeaderのみ
     │ (Master)  │───→│ Follower2 │  読み取りはどこからでも
     └──────────┘    │ Follower3 │
                     └───────────┘
     - 利点: 一貫性が保ちやすい
     - 欠点: Leaderが単一障害点、書き込みスケール不可
     - 例: MySQL, PostgreSQL, MongoDB

  2. マルチリーダー（Multi-Leader）
     ┌──────────┐    ┌──────────┐
     │  Leader1  │←──→│  Leader2  │  どのリーダーにも書き込み可能
     │ (Tokyo)   │    │ (US-East) │
     └──────────┘    └──────────┘
     - 利点: 書き込みの可用性向上、低レイテンシー
     - 欠点: コンフリクト解決が必要
     - 例: CouchDB, Galera Cluster

  3. リーダーレス（Leaderless）
     ┌──────┐  ┌──────┐  ┌──────┐
     │Node1 │  │Node2 │  │Node3 │  全ノードが対等
     └──────┘  └──────┘  └──────┘
     - Quorum: W + R > N で一貫性を確保
       - W=書き込みノード数, R=読み取りノード数, N=総ノード数
       - 例: N=3, W=2, R=2 → 書き込み2台成功で完了、読み取り2台から取得
     - 利点: 高可用性、単一障害点なし
     - 欠点: 実装が複雑、コンフリクト解決が必要
     - 例: Cassandra, DynamoDB, Riak

  コンフリクト解決戦略:
  1. Last Write Wins (LWW)
     - タイムスタンプが最新の書き込みが勝つ
     - シンプルだがデータ損失の可能性
  2. マージ
     - 両方の変更を保持して結合
     - CRDTs（Conflict-free Replicated Data Types）が有効
  3. アプリケーションレベル解決
     - コンフリクトをユーザーに提示して選択させる
     - 例: Google Docs の共同編集
```

---

## 3. ロードバランシング

### 3.1 ロードバランシングの基本

```
ロードバランサーの設置場所:

  Client ─→ [LB1] ─→ Web Server ─→ [LB2] ─→ App Server ─→ [LB3] ─→ DB

  L4（トランスポート層）ロードバランサー:
  - TCP/UDPレベルで分散
  - パケットの中身を見ない
  - 高速、低オーバーヘッド
  - 例: AWS NLB, HAProxy(L4モード), Linux IPVS

  L7（アプリケーション層）ロードバランサー:
  - HTTP/HTTPSレベルで分散
  - URLパス、ヘッダー、Cookie 等で振り分け可能
  - SSL終端、圧縮、キャッシュ等の機能
  - 例: AWS ALB, Nginx, HAProxy(L7モード), Envoy

  分散アルゴリズム:
  ┌──────────────────────────────────────────────────┐
  │ アルゴリズム         │ 説明                        │
  ├──────────────────────────────────────────────────┤
  │ ラウンドロビン       │ 順番に振り分け               │
  │ 加重ラウンドロビン   │ 重みに応じて振り分け         │
  │ 最小接続数           │ 接続数が最も少ないサーバーへ  │
  │ 最短応答時間         │ レスポンスが最も速いサーバーへ │
  │ IPハッシュ           │ クライアントIPで固定         │
  │ コンシステントハッシュ│ ノード追加/削除時の影響を最小化│
  └──────────────────────────────────────────────────┘
```

### 3.2 コンシステントハッシュ

```python
import hashlib
from bisect import bisect_right

class ConsistentHash:
    """コンシステントハッシュの実装例"""

    def __init__(self, nodes=None, replicas=150):
        """
        Args:
            nodes: 初期ノードのリスト
            replicas: 各ノードの仮想ノード数（多いほど均一に分散）
        """
        self.replicas = replicas
        self.ring = {}       # ハッシュ値 → ノード名
        self.sorted_keys = []  # ソート済みハッシュ値リスト

        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash(self, key: str) -> int:
        """キーのハッシュ値を計算"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node: str):
        """ノードを追加"""
        for i in range(self.replicas):
            virtual_key = f"{node}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = node
            self.sorted_keys.append(hash_value)
        self.sorted_keys.sort()

    def remove_node(self, node: str):
        """ノードを削除"""
        for i in range(self.replicas):
            virtual_key = f"{node}:{i}"
            hash_value = self._hash(virtual_key)
            del self.ring[hash_value]
            self.sorted_keys.remove(hash_value)

    def get_node(self, key: str) -> str:
        """キーが属するノードを取得"""
        if not self.ring:
            raise ValueError("No nodes available")

        hash_value = self._hash(key)
        idx = bisect_right(self.sorted_keys, hash_value)
        if idx == len(self.sorted_keys):
            idx = 0  # リングを一周
        return self.ring[self.sorted_keys[idx]]

# 使用例
ch = ConsistentHash(["server-1", "server-2", "server-3"])

# データの割り当て
for key in ["user:1001", "user:1002", "user:1003", "order:5001"]:
    node = ch.get_node(key)
    print(f"{key} → {node}")

# ノード追加 → 影響を受けるキーは約 1/N のみ
ch.add_node("server-4")

# ノード削除 → 影響を受けるキーは約 1/N のみ
ch.remove_node("server-2")
```

### 3.3 ヘルスチェックとフェイルオーバー

```
ヘルスチェックの種類:

  1. パッシブヘルスチェック
     - 実際のリクエストの結果を監視
     - エラー率が閾値を超えたらノードを除外
     - 追加のトラフィックが不要

  2. アクティブヘルスチェック
     - 定期的に専用のエンドポイントにリクエスト
     - /health や /ready エンドポイント
     - より早く障害を検知可能
```

```python
# ヘルスチェックエンドポイントの実装例
from flask import Flask, jsonify
import psycopg2
import redis

app = Flask(__name__)

@app.route("/health")
def health_check():
    """簡易ヘルスチェック（サーバーが動いているか）"""
    return jsonify({"status": "ok"}), 200

@app.route("/health/detailed")
def detailed_health_check():
    """詳細ヘルスチェック（依存サービスの状態も確認）"""
    checks = {}

    # データベースの疎通確認
    try:
        conn = psycopg2.connect("postgresql://localhost/myapp")
        conn.execute("SELECT 1")
        conn.close()
        checks["database"] = {"status": "healthy"}
    except Exception as e:
        checks["database"] = {"status": "unhealthy", "error": str(e)}

    # Redisの疎通確認
    try:
        r = redis.Redis()
        r.ping()
        checks["cache"] = {"status": "healthy"}
    except Exception as e:
        checks["cache"] = {"status": "unhealthy", "error": str(e)}

    # ディスク容量の確認
    import shutil
    usage = shutil.disk_usage("/")
    free_percent = usage.free / usage.total * 100
    if free_percent > 10:
        checks["disk"] = {"status": "healthy", "free_percent": round(free_percent, 1)}
    else:
        checks["disk"] = {"status": "warning", "free_percent": round(free_percent, 1)}

    # 全体のステータス判定
    overall = "healthy"
    for check in checks.values():
        if check["status"] == "unhealthy":
            overall = "unhealthy"
            break
        if check["status"] == "warning":
            overall = "degraded"

    status_code = 200 if overall == "healthy" else 503
    return jsonify({"status": overall, "checks": checks}), status_code
```

```
フェイルオーバーパターン:

  1. アクティブ-パッシブ（Active-Passive）
     ┌──────────┐    ┌──────────┐
     │  Active   │    │ Passive  │  Heartbeat で監視
     │ (稼働中)  │───→│ (待機中)  │  Active が落ちたら
     └──────────┘    └──────────┘  Passive が昇格
     - メリット: シンプル、データ一貫性が高い
     - デメリット: パッシブ側のリソースが無駄

  2. アクティブ-アクティブ（Active-Active）
     ┌──────────┐    ┌──────────┐
     │ Active-1  │←──→│ Active-2  │  両方がリクエストを処理
     │ (稼働中)  │    │ (稼働中)  │  1台が落ちても継続
     └──────────┘    └──────────┘
     - メリット: リソース効率が良い、高可用性
     - デメリット: データ同期が複雑
```

---

## 4. キャッシング

### 4.1 キャッシュの階層

```
キャッシュの階層構造（上ほどクライアントに近い）:

  ┌─────────────────────────────────────────────────┐
  │  クライアントキャッシュ                            │
  │  - ブラウザキャッシュ（Cache-Control, ETag）      │
  │  - モバイルアプリのローカルストレージ               │
  │  - レイテンシー: 0ms（ネットワーク不要）            │
  ├─────────────────────────────────────────────────┤
  │  CDNキャッシュ                                    │
  │  - CloudFront, Cloudflare, Fastly                │
  │  - 地理的に分散したエッジサーバー                   │
  │  - レイテンシー: 1-10ms                           │
  ├─────────────────────────────────────────────────┤
  │  アプリケーションキャッシュ                         │
  │  - Redis, Memcached                              │
  │  - インメモリでサブミリ秒アクセス                   │
  │  - レイテンシー: 1-5ms                            │
  ├─────────────────────────────────────────────────┤
  │  データベースキャッシュ                             │
  │  - クエリキャッシュ                                │
  │  - バッファプール（InnoDB Buffer Pool等）          │
  │  - レイテンシー: 1-10ms                           │
  ├─────────────────────────────────────────────────┤
  │  ディスクキャッシュ                                │
  │  - OS ページキャッシュ                             │
  │  - SSDのキャッシュ                                │
  │  - レイテンシー: 0.1-1ms                          │
  └─────────────────────────────────────────────────┘
```

### 4.2 キャッシュ戦略

```python
import redis
import json
from datetime import timedelta

r = redis.Redis()

# --- 1. Cache-Aside（Lazy Loading） ---
# アプリケーションがキャッシュの読み書きを制御
# 最も一般的なパターン

def get_user_cache_aside(user_id: str) -> dict:
    """Cache-Aside: まずキャッシュを確認、なければDBから取得してキャッシュ"""
    cache_key = f"user:{user_id}"

    # 1. キャッシュを確認
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)  # キャッシュヒット

    # 2. キャッシュミス → DBから取得
    user = db.find_user(user_id)
    if user is None:
        return None

    # 3. キャッシュに保存（TTL付き）
    r.setex(cache_key, timedelta(hours=1), json.dumps(user.to_dict()))

    return user.to_dict()

def update_user_cache_aside(user_id: str, data: dict):
    """更新時はキャッシュを無効化"""
    db.update_user(user_id, data)
    r.delete(f"user:{user_id}")  # キャッシュを削除


# --- 2. Write-Through ---
# 書き込み時にキャッシュとDBを同時更新

def save_user_write_through(user_id: str, data: dict):
    """Write-Through: DBとキャッシュを同時に更新"""
    # DBに書き込み
    db.save_user(user_id, data)

    # キャッシュにも書き込み
    cache_key = f"user:{user_id}"
    r.setex(cache_key, timedelta(hours=1), json.dumps(data))


# --- 3. Write-Behind（Write-Back） ---
# まずキャッシュに書き込み、非同期でDBに反映

class WriteBehindCache:
    def __init__(self):
        self.write_queue = []

    def save(self, user_id: str, data: dict):
        """まずキャッシュに書き込み"""
        cache_key = f"user:{user_id}"
        r.setex(cache_key, timedelta(hours=1), json.dumps(data))

        # 書き込みキューに追加（非同期でDBに反映）
        self.write_queue.append(("user", user_id, data))

    def flush(self):
        """キューの内容をDBに一括書き込み"""
        while self.write_queue:
            entity_type, entity_id, data = self.write_queue.pop(0)
            db.save(entity_type, entity_id, data)


# --- 4. Read-Through ---
# キャッシュ自体がDBからの読み込みを管理

class ReadThroughCache:
    def __init__(self, loader):
        self.loader = loader  # データ取得関数

    def get(self, key: str) -> dict:
        cached = r.get(key)
        if cached:
            return json.loads(cached)

        # キャッシュが自動的にDBから取得
        data = self.loader(key)
        if data:
            r.setex(key, timedelta(hours=1), json.dumps(data))
        return data

# 使用例
user_cache = ReadThroughCache(loader=lambda key: db.find_user(key.split(":")[1]))
user = user_cache.get("user:1001")
```

```
キャッシュ戦略の比較:

  ┌────────────────────────────────────────────────────────┐
  │ 戦略          │ 利点              │ 欠点              │
  ├────────────────────────────────────────────────────────┤
  │ Cache-Aside   │ シンプル、汎用的   │ キャッシュミス時     │
  │               │                   │ のレイテンシー増大   │
  ├────────────────────────────────────────────────────────┤
  │ Write-Through │ 一貫性が高い       │ 書き込みが遅い      │
  │               │                   │ (2箇所に書く)       │
  ├────────────────────────────────────────────────────────┤
  │ Write-Behind  │ 書き込みが速い     │ データ損失リスク    │
  │               │ バッチ最適化可能   │ 実装が複雑          │
  ├────────────────────────────────────────────────────────┤
  │ Read-Through  │ アプリコードが     │ カスタマイズ性      │
  │               │ シンプルに         │ が低い場合がある    │
  └────────────────────────────────────────────────────────┘
```

### 4.3 キャッシュの課題と対策

```
キャッシュの一般的な課題:

  1. キャッシュスタンピード（Cache Stampede / Thundering Herd）
     - 人気のあるキーのキャッシュが期限切れ
     - 大量のリクエストが同時にDBへ → DB過負荷
     対策:
     - ロック: 1つのリクエストのみDBアクセス、他は待機
     - 確率的早期更新: TTL期限前にランダムに更新
     - 二重キャッシュ: 主キャッシュの裏にバックアップキャッシュ

  2. キャッシュペネトレーション（Cache Penetration）
     - 存在しないキーへの大量リクエスト
     - 常にキャッシュミス → 毎回DBアクセス
     対策:
     - ネガティブキャッシュ: 存在しないキーもキャッシュ（短いTTL）
     - ブルームフィルタ: 存在チェックを高速に行う

  3. キャッシュ雪崩（Cache Avalanche）
     - 多くのキーが同時に期限切れ
     - 一斉にDBアクセスが発生
     対策:
     - TTLにランダムなジッター（揺らぎ）を追加
     - キャッシュの段階的ウォームアップ

  4. データの不整合
     - DBを更新したのにキャッシュが古い
     対策:
     - Write-Through/Write-Behind で整合性を確保
     - キャッシュの無効化パターン
     - 短いTTLの設定
```

```python
# キャッシュスタンピード対策の実装例
import time
import threading

class StampedeProtectedCache:
    """キャッシュスタンピード対策付きキャッシュ"""

    def __init__(self):
        self.locks = {}  # キーごとのロック
        self.lock_manager = threading.Lock()

    def get_or_compute(self, key: str, compute_fn, ttl_seconds: int = 3600):
        """キャッシュ取得。ミス時はロックを取得して1リクエストのみDB問い合わせ"""
        # まずキャッシュを確認
        cached = r.get(key)
        if cached:
            return json.loads(cached)

        # ロックを取得（同一キーは1リクエストのみ）
        lock = self._get_lock(key)
        acquired = lock.acquire(timeout=5)

        if not acquired:
            # ロック取得失敗 → 少し待ってリトライ
            time.sleep(0.1)
            cached = r.get(key)
            return json.loads(cached) if cached else None

        try:
            # ダブルチェック（ロック取得中に別スレッドがキャッシュした可能性）
            cached = r.get(key)
            if cached:
                return json.loads(cached)

            # DBから取得してキャッシュ
            value = compute_fn()
            if value is not None:
                # TTLにジッターを追加（雪崩防止）
                import random
                jitter = random.randint(0, ttl_seconds // 10)
                r.setex(key, ttl_seconds + jitter, json.dumps(value))
            return value
        finally:
            lock.release()

    def _get_lock(self, key):
        with self.lock_manager:
            if key not in self.locks:
                self.locks[key] = threading.Lock()
            return self.locks[key]
```

---

## 5. データベース設計

### 5.1 RDB vs NoSQL の選択

```
データベースの選択基準:

  RDB（リレーショナルDB）を選ぶべき場合:
  - データ間の関係が重要（JOINが必要）
  - トランザクション（ACID）が必要
  - スキーマが安定している
  - 複雑なクエリが必要
  - データの一貫性が重要
  例: ユーザー管理、注文管理、会計システム

  NoSQL を選ぶべき場合:
  - 超大規模なデータ量
  - 高い書き込みスループットが必要
  - スキーマが頻繁に変わる
  - 地理的に分散したデータ
  - 柔軟なデータモデルが必要
  例: ログ保存、IoTデータ、コンテンツ管理

  NoSQL の種類:
  ┌────────────────────────────────────────────────────┐
  │ 種類              │ 特徴              │ 代表例       │
  ├────────────────────────────────────────────────────┤
  │ Key-Value         │ 高速、シンプル     │ Redis,       │
  │                   │                   │ DynamoDB     │
  ├────────────────────────────────────────────────────┤
  │ ドキュメント       │ 柔軟なスキーマ     │ MongoDB,     │
  │                   │ JSONライク        │ CouchDB      │
  ├────────────────────────────────────────────────────┤
  │ カラムファミリー    │ 大規模分析向け    │ Cassandra,   │
  │                   │ 書き込みが高速    │ HBase        │
  ├────────────────────────────────────────────────────┤
  │ グラフ            │ 関係性の探索      │ Neo4j,       │
  │                   │ 推薦、SNS向け     │ Neptune      │
  ├────────────────────────────────────────────────────┤
  │ 時系列            │ 時系列データ特化   │ InfluxDB,    │
  │                   │ IoT、メトリクス   │ TimescaleDB  │
  └────────────────────────────────────────────────────┘
```

### 5.2 シャーディング

```
シャーディング（水平分割）:
  - データを複数のデータベースに分散
  - 各シャードは独立したデータベースインスタンス

  シャーディング戦略:

  1. レンジシャーディング
     - キーの範囲で分割
     例: user_id 1-1000 → Shard1, 1001-2000 → Shard2
     - 利点: 範囲クエリが効率的
     - 欠点: ホットスポットが発生しやすい

  2. ハッシュシャーディング
     - キーのハッシュ値で分割
     例: hash(user_id) % 4 → Shard番号
     - 利点: 均一に分散される
     - 欠点: 範囲クエリが非効率

  3. ディレクトリベースシャーディング
     - ルックアップサービスがシャードを決定
     - 利点: 柔軟な割り当て
     - 欠点: ルックアップサービスが単一障害点

  シャーディングの課題:
  ┌──────────────────────────────────────────────────┐
  │ 課題              │ 対策                          │
  ├──────────────────────────────────────────────────┤
  │ JOINの困難        │ アプリケーション側でJOIN       │
  │                   │ データの非正規化               │
  ├──────────────────────────────────────────────────┤
  │ トランザクション   │ 2フェーズコミット              │
  │ の困難            │ サガパターン                   │
  ├──────────────────────────────────────────────────┤
  │ リシャーディング    │ コンシステントハッシュ          │
  │ （再分割）         │ Vitess 等のシャーディング      │
  │                   │ ミドルウェアの活用             │
  ├──────────────────────────────────────────────────┤
  │ ホットスポット     │ シャードキーの慎重な選択       │
  │                   │ ソルティング（saltの追加）      │
  └──────────────────────────────────────────────────┘
```

### 5.3 データベースのインデックス設計

```sql
-- インデックスの基本原則

-- 1. プライマリキーは自動的にインデックスが作成される
CREATE TABLE users (
    id BIGINT PRIMARY KEY,         -- 自動インデックス
    email VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP NOT NULL
);

-- 2. 検索条件に使うカラムにインデックスを作成
CREATE INDEX idx_users_email ON users (email);
CREATE INDEX idx_users_status ON users (status);

-- 3. 複合インデックス（カラムの順序が重要）
-- WHERE status = 'active' AND created_at > '2024-01-01' をサポート
CREATE INDEX idx_users_status_created ON users (status, created_at);
-- 左端のカラムから使われる（Leftmost Prefix Rule）
-- ✅ WHERE status = 'active' → 使われる
-- ✅ WHERE status = 'active' AND created_at > ... → 使われる
-- ❌ WHERE created_at > ... → 使われない（status が先頭）

-- 4. カバリングインデックス（必要な全カラムがインデックスに含まれる）
-- SELECT email, name FROM users WHERE status = 'active';
CREATE INDEX idx_users_covering ON users (status, email, name);
-- テーブルデータへのアクセス不要 → 非常に高速

-- 5. ユニークインデックス（一意制約）
CREATE UNIQUE INDEX idx_users_email_unique ON users (email);

-- インデックスを作りすぎない注意点:
-- - INSERT/UPDATE/DELETE が遅くなる（インデックスも更新が必要）
-- - ストレージを消費する
-- - 目安: テーブルあたり5-10個まで
```

---

## 6. メッセージキュー

### 6.1 メッセージキューの基本

```
メッセージキューの用途:

  1. 非同期処理
     - メール送信、画像処理、レポート生成
     - ユーザーの待ち時間を短縮

  2. ピーク負荷の平準化
     - 急激なリクエスト増加をキューで吸収
     - ワーカーが一定のペースで処理

  3. サービス間の疎結合
     - サービスAがキューにメッセージを送信
     - サービスBがキューからメッセージを受信
     - AとBは直接通信しない

  4. イベント通知
     - 注文完了 → 在庫更新、メール送信、ポイント付与
     - 各サービスが独立にイベントを処理

  アーキテクチャ:
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Producer │───→│  Queue   │───→│ Consumer │
  │ (送信者) │    │ (キュー)  │    │ (受信者) │
  └──────────┘    └──────────┘    └──────────┘

  主要なメッセージングパターン:
  ┌─────────────────────────────────────────────┐
  │ パターン         │ 説明                       │
  ├─────────────────────────────────────────────┤
  │ Point-to-Point  │ 1メッセージ → 1コンシューマ  │
  │ (キュー)         │ タスクの分散処理             │
  ├─────────────────────────────────────────────┤
  │ Pub/Sub          │ 1メッセージ → 複数の         │
  │ (トピック)       │ サブスクライバー             │
  ├─────────────────────────────────────────────┤
  │ Request-Reply    │ リクエストを送信し           │
  │                  │ 返信を待つ                  │
  └─────────────────────────────────────────────┘
```

### 6.2 メッセージキューの実装例

```python
# --- SQS + Pythonの例 ---
import boto3
import json

sqs = boto3.client("sqs", region_name="ap-northeast-1")
QUEUE_URL = "https://sqs.ap-northeast-1.amazonaws.com/123456789/my-queue"

# プロデューサー（メッセージ送信）
def send_email_task(to: str, subject: str, body: str):
    """メール送信タスクをキューに投入"""
    message = {
        "task": "send_email",
        "payload": {
            "to": to,
            "subject": subject,
            "body": body
        },
        "created_at": datetime.now().isoformat()
    }
    sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps(message),
        MessageGroupId="email-tasks"  # FIFOキューの場合
    )

# コンシューマー（メッセージ処理）
def process_messages():
    """キューからメッセージを取得して処理"""
    while True:
        response = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=20,  # ロングポーリング
            VisibilityTimeout=60  # 処理中は他のワーカーから見えない
        )

        messages = response.get("Messages", [])
        for msg in messages:
            try:
                task = json.loads(msg["Body"])
                handle_task(task)

                # 処理完了 → メッセージ削除
                sqs.delete_message(
                    QueueUrl=QUEUE_URL,
                    ReceiptHandle=msg["ReceiptHandle"]
                )
            except Exception as e:
                logger.error(f"Failed to process message: {e}")
                # 削除しない → VisibilityTimeout後に再処理される

def handle_task(task: dict):
    """タスクの種類に応じて処理"""
    if task["task"] == "send_email":
        payload = task["payload"]
        email_service.send(payload["to"], payload["subject"], payload["body"])
    elif task["task"] == "generate_report":
        report_service.generate(task["payload"]["report_id"])
    else:
        logger.warning(f"Unknown task type: {task['task']}")
```

### 6.3 メッセージングプラットフォームの比較

```
主要メッセージングプラットフォームの比較:

  ┌─────────────────────────────────────────────────────────┐
  │ 製品         │ 特徴                │ ユースケース        │
  ├─────────────────────────────────────────────────────────┤
  │ Amazon SQS   │ フルマネージド       │ シンプルな          │
  │              │ 無限スケール        │ タスクキュー        │
  │              │ FIFOキュー対応      │                    │
  ├─────────────────────────────────────────────────────────┤
  │ RabbitMQ     │ 豊富なルーティング   │ 複雑なルーティング   │
  │              │ プロトコル標準準拠   │ エンタープライズ     │
  │              │ (AMQP)             │ 統合                │
  ├─────────────────────────────────────────────────────────┤
  │ Apache Kafka │ 超高スループット     │ イベントストリーム   │
  │              │ ログベース          │ リアルタイム分析     │
  │              │ リプレイ可能        │ マイクロサービス     │
  ├─────────────────────────────────────────────────────────┤
  │ Redis Streams│ 低レイテンシー       │ リアルタイム        │
  │              │ シンプル            │ メッセージング       │
  │              │ Redis内蔵          │ チャット            │
  ├─────────────────────────────────────────────────────────┤
  │ Google       │ フルマネージド       │ GCPエコシステム     │
  │ Pub/Sub      │ グローバル分散      │ イベント駆動        │
  └─────────────────────────────────────────────────────────┘
```

---

## 7. マイクロサービスアーキテクチャ

### 7.1 モノリス vs マイクロサービス

```
モノリスアーキテクチャ:
  ┌─────────────────────────────────┐
  │           モノリス               │
  │  ┌─────┬──────┬──────┬─────┐   │
  │  │ UI  │ 認証 │ 注文 │ 決済 │   │
  │  └─────┴──────┴──────┴─────┘   │
  │          1つのデプロイ単位        │
  └─────────────────────────────────┘
  利点:
  - 開発・テストが容易
  - デプロイが単純
  - プロセス内通信で高速
  - トランザクションが容易

  欠点:
  - 部分的なスケーリングが困難
  - 技術スタックの固定
  - チーム間の調整が必要
  - コードベースが肥大化

マイクロサービスアーキテクチャ:
  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
  │ 認証  │  │ 注文 │  │ 決済 │  │ 通知 │
  │サービス│  │サービス│  │サービス│  │サービス│
  │      │  │      │  │      │  │      │
  │ DB   │  │ DB   │  │ DB   │  │ DB   │
  └──────┘  └──────┘  └──────┘  └──────┘
  利点:
  - 独立したデプロイ
  - 独立したスケーリング
  - 技術選択の自由
  - チームの自律性

  欠点:
  - 運用の複雑さ（分散システム）
  - ネットワーク通信のオーバーヘッド
  - データの一貫性確保が困難
  - デバッグが困難

判断基準:
  スタートアップ → モノリスから始める
  チーム10人以下 → モノリスで十分
  明確な境界がある → マイクロサービス検討
  独立スケールが必要 → マイクロサービス検討
  「モノリスファースト」アプローチが推奨
```

### 7.2 サービス間通信

```
サービス間通信パターン:

  1. 同期通信（REST/gRPC）
     ┌─────────┐   HTTP/gRPC   ┌─────────┐
     │ Service │──────────────→│ Service │
     │    A    │←──────────────│    B    │
     └─────────┘   Response    └─────────┘
     - 利点: 直感的、レスポンスが即座
     - 欠点: 結合度が高い、連鎖障害のリスク

  2. 非同期通信（メッセージキュー）
     ┌─────────┐    ┌───────┐    ┌─────────┐
     │ Service │──→│ Queue │──→│ Service │
     │    A    │   └───────┘   │    B    │
     └─────────┘               └─────────┘
     - 利点: 疎結合、耐障害性が高い
     - 欠点: 結果をすぐに得られない、デバッグが困難

  3. イベント駆動
     ┌─────────┐    ┌────────────┐    ┌─────────┐
     │ Service │──→│ Event Bus  │──→│ Service │
     │    A    │   │ (Kafka等)  │──→│ Service │
     └─────────┘   └────────────┘──→│ Service │
                                     └─────────┘
     - 利点: 完全な疎結合、新サービスの追加が容易
     - 欠点: イベント順序の保証、デバッグの困難さ
```

```python
# --- サービス間通信の実装例 ---

# 1. REST API 呼び出し（サーキットブレーカー付き）
import requests
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=30)
def call_payment_service(order_id: str, amount: float) -> dict:
    """決済サービスを呼び出す（サーキットブレーカー付き）"""
    try:
        response = requests.post(
            "http://payment-service/api/v1/charge",
            json={"order_id": order_id, "amount": amount},
            timeout=5  # タイムアウト設定は必須
        )
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        raise PaymentServiceTimeout("Payment service timed out")
    except requests.ConnectionError:
        raise PaymentServiceUnavailable("Payment service is unavailable")

# 2. サガパターン（分散トランザクション）
class OrderSaga:
    """注文処理のサガ（各ステップに補償トランザクションを用意）"""

    def execute(self, order):
        try:
            # Step 1: 在庫を予約
            reservation = inventory_service.reserve(order.items)

            try:
                # Step 2: 決済処理
                payment = payment_service.charge(order.amount)

                try:
                    # Step 3: 配送手配
                    shipping = shipping_service.schedule(order)
                except Exception:
                    # Step 3 失敗 → Step 2 の補償（返金）
                    payment_service.refund(payment.transaction_id)
                    raise
            except Exception:
                # Step 2 失敗 → Step 1 の補償（在庫解放）
                inventory_service.cancel_reservation(reservation.id)
                raise
        except Exception as e:
            order.status = "failed"
            order.failure_reason = str(e)
            raise OrderFailedError(str(e)) from e

        order.status = "completed"
        return order
```

### 7.3 API Gateway パターン

```
API Gatewayの役割:

  ┌────────┐
  │ Client │
  └───┬────┘
      │
  ┌───▼──────────────────────────┐
  │        API Gateway            │
  │  - ルーティング                │
  │  - 認証・認可                 │
  │  - レート制限                 │
  │  - リクエスト/レスポンス変換    │
  │  - ロギング・モニタリング       │
  │  - キャッシュ                  │
  │  - サーキットブレーカー         │
  └───┬──────┬──────┬────────────┘
      │      │      │
  ┌───▼─┐┌──▼──┐┌──▼──┐
  │認証  ││注文 ││商品 │
  │サービス││サービス││サービス│
  └─────┘└─────┘└─────┘

  代表的な実装:
  - AWS API Gateway
  - Kong
  - Envoy + Istio
  - Nginx
  - Netflix Zuul / Spring Cloud Gateway
```

---

## 8. API設計

### 8.1 RESTful API設計

```
REST API 設計原則:

  1. リソース指向
     - URL はリソース（名詞）を表す
     - HTTPメソッドで操作を表す

  ┌────────────────────────────────────────────────────┐
  │ 操作      │ メソッド │ URL例                        │
  ├────────────────────────────────────────────────────┤
  │ 一覧取得   │ GET     │ /api/v1/users               │
  │ 詳細取得   │ GET     │ /api/v1/users/123           │
  │ 作成       │ POST    │ /api/v1/users               │
  │ 全体更新   │ PUT     │ /api/v1/users/123           │
  │ 部分更新   │ PATCH   │ /api/v1/users/123           │
  │ 削除       │ DELETE  │ /api/v1/users/123           │
  │ サブリソース│ GET     │ /api/v1/users/123/orders    │
  └────────────────────────────────────────────────────┘

  2. ステータスコード
  ┌────────────────────────────────────────────────────┐
  │ コード │ 意味              │ 使用場面               │
  ├────────────────────────────────────────────────────┤
  │ 200    │ OK               │ 正常レスポンス          │
  │ 201    │ Created          │ リソース作成成功         │
  │ 204    │ No Content       │ 成功（レスポンスなし）   │
  │ 400    │ Bad Request      │ クライアントエラー       │
  │ 401    │ Unauthorized     │ 認証失敗               │
  │ 403    │ Forbidden        │ 認可失敗               │
  │ 404    │ Not Found        │ リソースが存在しない     │
  │ 409    │ Conflict         │ 競合（重複等）          │
  │ 422    │ Unprocessable    │ バリデーションエラー     │
  │ 429    │ Too Many Requests│ レート制限超過          │
  │ 500    │ Internal Error   │ サーバーエラー          │
  │ 503    │ Service Unavail. │ サービス一時停止         │
  └────────────────────────────────────────────────────┘

  3. ページネーション
  - カーソルベース（推奨）: /api/v1/users?cursor=abc123&limit=20
  - オフセットベース: /api/v1/users?page=3&per_page=20

  4. フィルタリング・ソート
  - /api/v1/users?status=active&sort=created_at&order=desc
  - /api/v1/products?min_price=1000&max_price=5000&category=electronics

  5. バージョニング
  - URLパス: /api/v1/users（最も一般的）
  - ヘッダー: Accept: application/vnd.myapi.v1+json
  - クエリパラメータ: /api/users?version=1
```

### 8.2 API レスポンス設計

```python
# --- 統一されたレスポンスフォーマット ---

# 成功レスポンス
{
    "status": "success",
    "data": {
        "id": "user-001",
        "name": "Alice",
        "email": "alice@example.com",
        "created_at": "2024-01-15T10:30:00Z"
    }
}

# 一覧レスポンス（ページネーション付き）
{
    "status": "success",
    "data": [
        {"id": "user-001", "name": "Alice"},
        {"id": "user-002", "name": "Bob"}
    ],
    "pagination": {
        "total": 150,
        "page": 1,
        "per_page": 20,
        "total_pages": 8,
        "next_cursor": "eyJpZCI6InVzZXItMDIwIn0="
    }
}

# エラーレスポンス
{
    "status": "error",
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "入力値に問題があります",
        "details": [
            {
                "field": "email",
                "message": "メールアドレスの形式が正しくありません"
            },
            {
                "field": "password",
                "message": "パスワードは8文字以上で入力してください"
            }
        ]
    }
}
```

### 8.3 レート制限

```python
# --- レート制限の実装パターン ---

# 1. 固定ウィンドウカウンター
import time

class FixedWindowRateLimiter:
    """固定ウィンドウ方式のレート制限"""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def is_allowed(self, client_id: str) -> bool:
        current_window = int(time.time() / self.window_seconds)
        key = f"rate_limit:{client_id}:{current_window}"

        count = r.incr(key)
        if count == 1:
            r.expire(key, self.window_seconds)

        return count <= self.max_requests

# 2. スライディングウィンドウログ
class SlidingWindowLogRateLimiter:
    """スライディングウィンドウログ方式"""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        key = f"rate_limit:{client_id}"

        # 古いエントリを削除
        r.zremrangebyscore(key, 0, window_start)

        # 現在のウィンドウ内のリクエスト数をカウント
        count = r.zcard(key)

        if count < self.max_requests:
            r.zadd(key, {str(now): now})
            r.expire(key, self.window_seconds)
            return True

        return False

# 3. トークンバケット
class TokenBucketRateLimiter:
    """トークンバケット方式"""

    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: バケットの最大トークン数
            refill_rate: 1秒あたりに補充されるトークン数
        """
        self.capacity = capacity
        self.refill_rate = refill_rate

    def is_allowed(self, client_id: str, tokens_needed: int = 1) -> bool:
        key = f"token_bucket:{client_id}"
        now = time.time()

        # 現在のトークン数と最終更新時刻を取得
        data = r.hgetall(key)
        if not data:
            # 初回: バケットを満タンにして1トークン消費
            r.hset(key, mapping={
                "tokens": str(self.capacity - tokens_needed),
                "last_refill": str(now)
            })
            r.expire(key, 3600)
            return True

        current_tokens = float(data[b"tokens"])
        last_refill = float(data[b"last_refill"])

        # 時間経過分のトークンを補充
        elapsed = now - last_refill
        new_tokens = min(
            self.capacity,
            current_tokens + elapsed * self.refill_rate
        )

        if new_tokens >= tokens_needed:
            r.hset(key, mapping={
                "tokens": str(new_tokens - tokens_needed),
                "last_refill": str(now)
            })
            return True

        return False

# 使い方
limiter = TokenBucketRateLimiter(capacity=100, refill_rate=10)  # 100リクエスト/バースト、10リクエスト/秒
if limiter.is_allowed("user:123"):
    process_request()
else:
    return {"error": "Rate limit exceeded"}, 429
```

---

## 9. 可用性と信頼性

### 9.1 可用性の計算

```
可用性（Availability）:
  SLA（Service Level Agreement）で目標を定める

  ┌──────────────────────────────────────────┐
  │ 可用性    │ ダウンタイム/年  │ よく言う呼び方 │
  ├──────────────────────────────────────────┤
  │ 99%       │ 3.65 日         │ ツーナイン     │
  │ 99.9%     │ 8.76 時間       │ スリーナイン   │
  │ 99.95%    │ 4.38 時間       │               │
  │ 99.99%    │ 52.6 分         │ フォーナイン   │
  │ 99.999%   │ 5.26 分         │ ファイブナイン │
  └──────────────────────────────────────────┘

  直列構成の可用性:
  A → B → C
  全体の可用性 = Aの可用性 × Bの可用性 × Cの可用性
  例: 99.9% × 99.9% × 99.9% = 99.7% (8.76時間→26.3時間のダウンタイム)

  並列構成の可用性:
  A ─→ ┐
       ├─→ 出力
  B ─→ ┘
  全体の可用性 = 1 - (1 - Aの可用性) × (1 - Bの可用性)
  例: 1 - (0.001 × 0.001) = 99.9999% (31.5秒のダウンタイム)
```

### 9.2 信頼性パターン

```
信頼性を向上させるパターン:

  1. リトライ（Retry）
     - 一時的な障害からの回復
     - 指数バックオフで再試行間隔を広げる
     - 最大リトライ回数を設定

  2. サーキットブレーカー（Circuit Breaker）
     ┌──────────┐  失敗増加  ┌──────────┐ タイムアウト ┌──────────┐
     │  Closed  │ ────────→ │   Open   │ ──────────→│Half-Open │
     │ (正常)   │           │ (遮断)   │            │ (試行)   │
     └──────────┘ ←──────── └──────────┘ ←────────── └──────────┘
                   成功増加                   失敗

  3. バルクヘッド（Bulkhead）
     - リソースを分離して障害の波及を防ぐ
     - 例: スレッドプールの分離、接続プールの分離

  4. タイムアウト（Timeout）
     - 全てのネットワーク呼び出しにタイムアウトを設定
     - デフォルトのタイムアウトに依存しない

  5. フォールバック（Fallback）
     - 主要な処理が失敗した場合の代替手段
     - キャッシュからのデータ提供
     - デフォルト値の返却
     - 縮退運転
```

```python
# --- リトライ with 指数バックオフ ---
import time
import random

def retry_with_exponential_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True
):
    """指数バックオフ付きリトライ"""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except (ConnectionError, TimeoutError) as e:
            if attempt == max_retries:
                raise  # 最後のリトライでも失敗 → 例外を上げる

            delay = min(base_delay * (2 ** attempt), max_delay)
            if jitter:
                delay = delay * (0.5 + random.random())  # 0.5〜1.5倍のジッター

            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)

# 使用例
result = retry_with_exponential_backoff(
    lambda: external_api.fetch_data("user-001"),
    max_retries=3,
    base_delay=1.0
)


# --- サーキットブレーカーの実装 ---
import time
import threading
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"        # 正常: リクエストを通す
    OPEN = "open"            # 遮断: リクエストを即座に拒否
    HALF_OPEN = "half_open"  # 試行: 1リクエストだけ通す

class CircuitBreaker:
    """サーキットブレーカーの実装"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()

    def call(self, func, *args, **kwargs):
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_try_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerOpenError(
                        "Circuit breaker is OPEN. Request rejected."
                    )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            else:
                self.failure_count = 0

    def _on_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

    def _should_try_reset(self):
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
```

---

## 10. 数字で考える（Back-of-the-Envelope Estimation）

### 10.1 レイテンシーの基礎知識

```
各種操作のレイテンシー概算:

  ┌──────────────────────────────────────────────────┐
  │ 操作                        │ レイテンシー        │
  ├──────────────────────────────────────────────────┤
  │ L1 キャッシュ参照             │ 0.5 ns            │
  │ L2 キャッシュ参照             │ 7 ns              │
  │ メインメモリ参照              │ 100 ns             │
  │ SSD ランダムリード            │ 150 μs            │
  │ HDD ランダムリード            │ 10 ms              │
  │ 同一DC内ラウンドトリップ       │ 0.5 ms            │
  │ Redis GET                   │ 0.1-1 ms           │
  │ MySQL クエリ（インデックス）   │ 1-10 ms            │
  │ 東京-大阪 ラウンドトリップ     │ 5-10 ms            │
  │ 東京-US ラウンドトリップ       │ 100-200 ms         │
  │ TCP接続確立                   │ 50-150 ms          │
  │ TLSハンドシェイク             │ 100-300 ms         │
  └──────────────────────────────────────────────────┘

  1秒 = 1,000 ms = 1,000,000 μs = 1,000,000,000 ns
```

### 10.2 概算計算の実践

```
システム設計の概算:

  DAU 100万人のSNS:
  - ピークQPS: 100万 / 86400 × 3 ≈ 35 QPS (書込)
  - 読み取りQPS: 35 × 100 = 3,500 QPS
  - 1台のWebサーバー: 1,000〜10,000 QPS → 1-4台で足りる
  - DBは読み取りレプリカ + キャッシュで対応可能

  DAU 1億人:
  - 書込QPS: 3,500
  - 読み取りQPS: 350,000
  - Webサーバー: 数十台
  - DBシャーディング必須
  - CDN + Redis 必須

ストレージ見積もりの例:
  DAU 100万人のチャットアプリ:
  - 1ユーザーあたり1日50メッセージ
  - 1メッセージ平均200バイト
  - 1日のメッセージ数: 100万 × 50 = 5,000万
  - 1日のデータ量: 5,000万 × 200B = 10 GB/日
  - 1年のデータ量: 10 GB × 365 = 3.65 TB/年
  - 5年保持: 約 18 TB（レプリケーション込みで 54 TB）

帯域幅の見積もり:
  動画ストリーミングサービス:
  - DAU 500万人
  - 平均視聴時間: 1時間/日
  - 動画ビットレート: 5 Mbps
  - ピーク時同時視聴者: DAU × 10% = 50万人
  - ピーク帯域幅: 50万 × 5 Mbps = 2.5 Tbps
  - CDN分散後の1拠点あたり: 2.5 Tbps / 10 = 250 Gbps

QPS計算のテンプレート:
  DAU × アクション数/日 ÷ 86400 = 平均QPS
  平均QPS × ピーク倍率(2〜5) = ピークQPS
  ピークQPS × 安全率(1.5〜2) = 必要キャパシティ
```

### 10.3 実践的な見積もり例

```
Twitter風サービスの設計見積もり:

  前提:
  - MAU: 3億
  - DAU: 1.5億（MAUの50%）
  - 1ユーザーあたりの投稿: 2ツイート/日
  - 1ユーザーあたりの閲覧: 200ツイート/日
  - 1ツイート: 平均300バイト（テキスト）
  - 画像添付率: 20%、画像平均サイズ: 500KB

  QPS:
  - 書き込み: 1.5億 × 2 / 86400 ≈ 3,500 QPS
  - 読み取り: 1.5億 × 200 / 86400 ≈ 350,000 QPS
  - ピーク読み取り: 350,000 × 3 ≈ 1,000,000 QPS（100万QPS!）

  ストレージ:
  - テキスト: 1.5億 × 2 × 300B = 90 GB/日 → 33 TB/年
  - 画像: 1.5億 × 2 × 0.2 × 500KB = 30 TB/日 → 10.8 PB/年
  - 5年保持: テキスト165TB + 画像54PB

  設計上の考慮:
  - 読み取りが圧倒的に多い → キャッシュが最重要
  - タイムラインは事前計算（Fan-out on write）が有効
  - 画像はCDN + オブジェクトストレージ
  - ホットユーザー（フォロワー数百万）は特別扱い
```

---

## 11. 実践的なシステム設計の例

### 11.1 URL短縮サービスの設計

```
要件:
  - 長いURLを短いURLに変換
  - 短いURLにアクセスすると元のURLにリダイレクト
  - DAU: 1,000万人
  - 短縮URL生成: 1日1億件
  - 読み取り/書き込み比: 100:1

設計:

  API:
  - POST /api/v1/shorten {"url": "https://very-long-url.com/..."}
    → {"short_url": "https://tny.io/a1b2c3"}
  - GET /a1b2c3 → 301 Redirect

  データモデル:
  ┌───────────────────────────────────┐
  │ short_urls テーブル                │
  ├───────────────────────────────────┤
  │ id (PK)          │ BIGINT         │
  │ short_key         │ VARCHAR(7)     │
  │ original_url      │ TEXT           │
  │ created_at        │ TIMESTAMP      │
  │ expires_at        │ TIMESTAMP      │
  │ click_count       │ BIGINT         │
  └───────────────────────────────────┘

  短縮キー生成方法:
  - Base62エンコード: [a-zA-Z0-9] = 62文字
  - 7文字: 62^7 = 3.5兆通り（十分なキー空間）
  - 方法1: カウンターベース（IDをBase62変換）
  - 方法2: ハッシュベース（MD5/SHA256の先頭7文字）
  - 方法3: 事前生成キーサービス（Key Generation Service）

  アーキテクチャ:
  ┌────────┐   ┌──────┐   ┌──────┐   ┌──────┐
  │ Client │──→│ LB   │──→│ API  │──→│ Cache│
  └────────┘   └──────┘   │Server│   │Redis │
                           └──┬───┘   └──┬───┘
                              │          │
                           ┌──▼──────────▼───┐
                           │    Database      │
                           │   (Sharded)      │
                           └─────────────────┘

  キャッシュ戦略:
  - 読み取りが100倍多い → Cache-Aside で Redis にキャッシュ
  - 人気のあるURL: TTL = 24時間
  - 全体のキャッシュヒット率: 80%以上を目標
```

### 11.2 通知システムの設計

```
要件:
  - プッシュ通知、SMS、メールの3チャネル
  - 1日1億件の通知送信
  - ユーザーごとの通知設定（オプトイン/アウト）
  - 配信保証（少なくとも1回配信）

アーキテクチャ:
  ┌──────────┐   ┌────────────┐
  │ Service A │──→│            │
  │ Service B │──→│ Notification│──→ ┌────────┐ ──→ APNs
  │ Service C │──→│  Service    │    │ Queue  │ ──→ FCM
  │ Scheduler │──→│            │──→ │ (Kafka)│ ──→ SMS Gateway
  └──────────┘   └────────────┘    └────────┘ ──→ Email(SES)

  処理フロー:
  1. サービスが通知リクエストを送信
  2. 通知サービスが受信
     - ユーザーの通知設定を確認
     - レート制限を適用
     - テンプレートを適用
  3. チャネルごとのキューに投入
  4. ワーカーが各プロバイダーに送信
  5. 配信結果をログに記録

  考慮事項:
  - 冪等性: 同じ通知を2回送らないための重複排除
  - 優先度: 緊急通知は優先キューで処理
  - バッチ処理: 大量送信はバッチで効率化
  - フォールバック: プッシュ通知失敗 → メールにフォールバック
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| スケーリング | 垂直(強化) vs 水平(追加)。水平が主流 |
| CAP定理 | CP(一貫性) vs AP(可用性)。用途で選択 |
| ロードバランシング | L4/L7、コンシステントハッシュ |
| キャッシュ | Cache-Aside が最も一般的。スタンピード対策必須 |
| データベース | RDB vs NoSQL の使い分け。シャーディング |
| メッセージキュー | 非同期処理。サービス間の疎結合 |
| マイクロサービス | モノリスファーストで始める |
| API設計 | RESTful原則、統一レスポンス、レート制限 |
| 可用性 | サーキットブレーカー、リトライ、フォールバック |
| 見積もり | DAU → QPS → サーバー台数の概算 |

---

## 次に読むべきガイド
→ [[05-version-control.md]] — バージョン管理

---

## 参考文献
1. Kleppmann, M. "Designing Data-Intensive Applications." O'Reilly, 2017.
2. Alex Xu. "System Design Interview." 2020.
3. Alex Xu. "System Design Interview Vol. 2." 2022.
4. Newman, S. "Building Microservices." 2nd Edition, O'Reilly, 2021.
5. Richardson, C. "Microservices Patterns." Manning, 2018.
6. Nygard, M. "Release It!" 2nd Edition, Pragmatic Bookshelf, 2018.
7. Burns, B. "Designing Distributed Systems." O'Reilly, 2018.
8. Fowler, M. "Patterns of Enterprise Application Architecture." Addison-Wesley, 2002.

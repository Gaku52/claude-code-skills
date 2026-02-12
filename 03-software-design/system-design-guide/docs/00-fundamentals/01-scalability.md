# スケーラビリティ

> システムが負荷増大に対して性能を維持しながら成長できる能力を理解し、水平・垂直スケーリングの戦略と実装パターンを習得する。

## この章で学ぶこと

1. **垂直/水平スケーリングの本質的違い**: スケールアップとスケールアウトのコスト曲線、限界点、適用場面を深く理解し、適切な判断ができるようになる
2. **ステートレス設計の原理と実装**: なぜステートレスが水平スケーリングの前提条件なのかをWHYレベルで理解し、具体的な実装パターン（セッション外部化、Twelve-Factor App）を習得する
3. **自動スケーリング（Auto Scaling）の設計**: 目標追跡型、ステップ型、予測型のスケーリングポリシーの違いを理解し、実際のクラウド設定ができるようになる
4. **データ層のスケーリング戦略**: アプリケーション層よりも困難なデータベースのスケーリング手法（Read Replica、シャーディング、CQRS）の概要を把握する
5. **スケーラビリティの定量的評価**: 負荷テスト・ベンチマークによってスケーラビリティを数値で評価する方法を学ぶ

---

## 前提知識

| 前提知識 | 内容 | 参照リンク |
|---------|------|-----------|
| システム設計の基礎 | 設計プロセス、要件分析、見積もり手法 | [システム設計概要](./00-system-design-overview.md) |
| ネットワーク基礎 | TCP/IP、HTTP、ロードバランサーの基本概念 | [04-web-and-network](../../../../04-web-and-network/) |
| データベース基礎 | RDBMS、インデックス、クエリ最適化 | [06-data-and-security](../../../../06-data-and-security/) |
| クラウド基礎 | AWS/GCPの基本的なサービス（EC2、RDS等） | [05-infrastructure](../../../../05-infrastructure/) |
| 設計原則 | SOLIDの原則、結合度と凝集度 | [clean-code-principles: 00-principles](../../clean-code-principles/docs/00-principles/) |

---

## 1. スケーラビリティとは

### 1.1 定義

スケーラビリティ（Scalability）とは、システムが**負荷の増大**（ユーザー数、データ量、トラフィック）に対して、**許容可能な性能レベルを維持しながら拡張できる能力**である。

```
負荷が10倍になったとき:
  スケーラブルなシステム:    レイテンシが微増（200ms → 250ms）
  非スケーラブルなシステム:  レイテンシが爆発（200ms → 5000ms）or ダウン
```

### 1.2 なぜスケーラビリティが重要なのか

スケーラビリティは「将来の問題への備え」ではなく、**ビジネスの成功に直結する技術的能力**である。

**歴史的事例:**

1. **Twitter (2007-2010)**: Ruby on Railsモノリスが急成長に耐えきれず「Fail Whale」が頻出。MySQLの書き込みがボトルネックとなり、スケールアウトに2年を要した
2. **Instagram (2012)**: 13人のエンジニアで3000万ユーザーを支えた。シンプルなアーキテクチャ（Django + PostgreSQL + Redis）を維持し、必要な箇所のみスケールすることで成功
3. **Pokemon GO (2016)**: 発売直後にDAUが想定の50倍に達し、複数国でサービス停止。GoogleのSREチームが介入して復旧

これらの事例から学べる教訓:

```
┌────────────────────────────────────────────────────────────┐
│              スケーラビリティの教訓                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. 「成功」がシステムを殺す                                │
│     → ユーザー急増はビジネスの成功だが、                     │
│       技術的には最大の危機になりうる                         │
│                                                            │
│  2. 全てを最初からスケールさせる必要はない                    │
│     → Instagram は最小構成で3000万ユーザーを支えた           │
│     → 重要なのは「スケールできる設計」をしておくこと          │
│                                                            │
│  3. ボトルネックは予測と異なる場所に現れる                    │
│     → 計測なしの最適化は時間の無駄                           │
│     → プロファイリングとモニタリングが不可欠                 │
│                                                            │
│  4. スケーリングは段階的に行う                               │
│     → DAU 1K → 10K → 100K → 1M → 10M で                   │
│       必要な技術スタックは全く異なる                         │
└────────────────────────────────────────────────────────────┘
```

### 1.3 スケーラビリティの3つの次元

スケーラビリティは単一の概念ではなく、複数の次元で考える必要がある。

```python
# スケーラビリティの3つの次元を分析するフレームワーク
class ScalabilityAnalyzer:
    """システムのスケーラビリティを3つの次元で分析"""

    def __init__(self, system_name: str):
        self.system_name = system_name
        self.dimensions = {}

    def assess_dimension(self, dimension: str, current: float,
                         target: float, bottleneck: str,
                         strategy: str):
        """各次元のスケーラビリティを評価"""
        self.dimensions[dimension] = {
            "current": current,
            "target": target,
            "ratio": target / current if current > 0 else float('inf'),
            "bottleneck": bottleneck,
            "strategy": strategy,
        }

    def report(self) -> str:
        lines = [f"=== {self.system_name} スケーラビリティ分析 ===\n"]
        for dim, data in self.dimensions.items():
            lines.append(f"【{dim}】")
            lines.append(f"  現在: {data['current']:,.0f}")
            lines.append(f"  目標: {data['target']:,.0f}")
            lines.append(f"  倍率: {data['ratio']:.1f}x")
            lines.append(f"  ボトルネック: {data['bottleneck']}")
            lines.append(f"  戦略: {data['strategy']}")
            lines.append("")
        return "\n".join(lines)


# 使用例: ECサイトのスケーラビリティ分析
analyzer = ScalabilityAnalyzer("ECサイト")

# 1. 負荷スケーラビリティ（Load Scalability）
analyzer.assess_dimension(
    "負荷スケーラビリティ",
    current=1000,          # 現在のQPS
    target=50000,          # 目標QPS
    bottleneck="DBの接続数上限（max_connections=500）",
    strategy="Read Replica導入 + コネクションプーリング（PgBouncer）"
)

# 2. データスケーラビリティ（Data Scalability）
analyzer.assess_dimension(
    "データスケーラビリティ",
    current=500,           # 現在のデータ量（GB）
    target=50000,          # 目標データ量（GB）
    bottleneck="単一DBのストレージ上限（16TB）とクエリ性能低下",
    strategy="シャーディング（user_id基準）+ コールドデータのS3退避"
)

# 3. 地理的スケーラビリティ（Geographic Scalability）
analyzer.assess_dimension(
    "地理的スケーラビリティ",
    current=1,             # 現在のリージョン数
    target=3,              # 目標リージョン数
    bottleneck="単一リージョンでの海外ユーザーのレイテンシ（300ms+）",
    strategy="CDN導入 + マルチリージョンRead Replica + Edge Computing"
)

print(analyzer.report())

# 出力:
# === ECサイト スケーラビリティ分析 ===
#
# 【負荷スケーラビリティ】
#   現在: 1,000
#   目標: 50,000
#   倍率: 50.0x
#   ボトルネック: DBの接続数上限（max_connections=500）
#   戦略: Read Replica導入 + コネクションプーリング（PgBouncer）
#
# 【データスケーラビリティ】
#   現在: 500
#   目標: 50,000
#   倍率: 100.0x
#   ボトルネック: 単一DBのストレージ上限（16TB）とクエリ性能低下
#   戦略: シャーディング（user_id基準）+ コールドデータのS3退避
#
# 【地理的スケーラビリティ】
#   現在: 1
#   目標: 3
#   倍率: 3.0x
#   ボトルネック: 単一リージョンでの海外ユーザーのレイテンシ（300ms+）
#   戦略: CDN導入 + マルチリージョンRead Replica + Edge Computing
```

### 1.4 AKF Scale Cube（スケーリングの3軸モデル）

AKF Scale Cubeは、スケーリングを3つの独立した軸で捉えるフレームワークである。Martin Abbott と Michael Fisher が提唱した。

```
                        Y軸: 機能分割
                        (マイクロサービス)
                            ^
                           /|
                          / |
                         /  |
                        /   |
                       /    |
                      /     |
                     /      |
                    /       |
                   /        |
                  /─────────/────────────→ X軸: 水平クローン
                 /         /               (ロードバランサー + 複数インスタンス)
                /         /
               /         /
              ▼         /
             Z軸: データ分割
             (シャーディング)

  ┌──────────────────────────────────────────────────────────┐
  │ X軸: 水平クローン                                        │
  │   同じアプリケーションを複数台にコピー                     │
  │   例: LB背後にApp Server を10台配置                       │
  │   効果: リクエスト処理能力の線形向上                       │
  │   制約: DB/キャッシュがボトルネックになりうる              │
  │                                                          │
  │ Y軸: 機能分割                                            │
  │   機能ごとに独立したサービスに分割                         │
  │   例: ユーザーサービス、商品サービス、決済サービス          │
  │   効果: 独立したスケーリングとデプロイが可能               │
  │   制約: サービス間通信のオーバーヘッド                    │
  │                                                          │
  │ Z軸: データ分割                                          │
  │   データの特定属性でリクエストを分割                       │
  │   例: user_id mod N でシャードを決定                      │
  │   効果: データ量に対するスケーリング                       │
  │   制約: クロスシャードクエリの複雑さ                      │
  └──────────────────────────────────────────────────────────┘
```

---

## 2. 垂直スケーリング vs 水平スケーリング

### 2.1 概念の比較

```
  ■ 垂直スケーリング（Scale Up）       ■ 水平スケーリング（Scale Out）
  ─────────────────────────           ──────────────────────────

  Before:    After:                   Before:      After:
  ┌─────┐   ┌─────────┐              ┌─────┐      ┌─────┐ ┌─────┐ ┌─────┐
  │ 4CPU│   │ 32 CPU  │              │ Srv │      │ Srv │ │ Srv │ │ Srv │
  │ 8GB │   │ 128 GB  │              │  1  │      │  1  │ │  2  │ │  3  │
  │ 1TB │   │ 10 TB   │              └─────┘      └─────┘ └─────┘ └─────┘
  └─────┘   └─────────┘
  (より強力な1台)                     (同程度の複数台)

  メリット:                          メリット:
  - 簡単（アプリ変更不要）            - 理論上無制限
  - 運用シンプル                     - 耐障害性向上
  - データ一貫性が保たれる            - コスト効率（線形）
                                    - 無停止でスケール可能
  デメリット:                        デメリット:
  - 物理的限界あり                   - 分散処理の複雑さ
  - コスト指数増加                   - データ一貫性の課題
  - SPOF（単一障害点）               - 運用の複雑さ増加
```

### 2.2 コスト曲線のシミュレーション

垂直スケーリングと水平スケーリングのコスト差は、規模が大きくなるほど劇的に広がる。

```python
import math

def vertical_scaling_cost(base_cost: float, scale_factor: int) -> float:
    """
    垂直スケーリングのコストモデル

    なぜ指数的に増加するか:
    1. CPUは周波数を上げるほど消費電力が指数的に増える（P ∝ f^3）
    2. 大容量メモリは製造コストが高い（ECCメモリ、多チャネル）
    3. 高性能マシンは市場が小さく、競争が少ない
    4. ベンダーロックインによるプレミアム価格

    実例: AWS EC2インスタンスの価格
    - t3.micro  (2vCPU,  1GB):   $0.0104/h  → 基準
    - t3.xlarge (4vCPU,  16GB):  $0.1664/h  → 16倍のスペック、16倍の価格
    - r5.4xlarge(16vCPU, 128GB): $1.008/h   → 128倍のスペック、97倍の価格
    - x1.32xlarge(128vCPU,1952GB):$13.338/h → 1952倍のスペック、1282倍の価格
    """
    # 実測データに基づく指数: 約1.6乗
    return base_cost * (scale_factor ** 1.6)


def horizontal_scaling_cost(base_cost: float, scale_factor: int,
                            overhead_pct: float = 10) -> float:
    """
    水平スケーリングのコストモデル

    なぜほぼ線形か:
    1. 同じスペックのマシンを追加するため、単価が変わらない
    2. クラウドでは大量購入による割引もある
    3. オーバーヘッドは固定的（LB、管理ノード等）

    overhead_pctの内訳:
    - ロードバランサー: 2-3%
    - 管理/モニタリング: 2-3%
    - サービスディスカバリ: 1-2%
    - 通信オーバーヘッド: 2-3%
    """
    overhead = 1 + overhead_pct / 100
    return base_cost * scale_factor * overhead


# コスト比較シミュレーション
base = 1000  # 基本コスト $1000/月
print("=== スケーリングコスト比較 ===\n")
print(f"{'倍率':>5s}  {'垂直コスト':>12s}  {'水平コスト':>12s}  "
      f"{'差額':>12s}  {'垂直/水平':>10s}")
print("-" * 65)

for factor in [1, 2, 4, 8, 16, 32, 64, 128]:
    v_cost = vertical_scaling_cost(base, factor)
    h_cost = horizontal_scaling_cost(base, factor)
    ratio = v_cost / h_cost if h_cost > 0 else 0
    print(f"  x{factor:<3d}  ${v_cost:>10,.0f}  ${h_cost:>10,.0f}  "
          f"${v_cost - h_cost:>10,.0f}  {ratio:>8.1f}x")

# 出力:
# === スケーリングコスト比較 ===
#
# 倍率    垂直コスト    水平コスト          差額    垂直/水平
# -----------------------------------------------------------------
#   x1       $1,000       $1,100         $-100       0.9x
#   x2       $3,031       $2,200          $831       1.4x
#   x4       $9,190       $4,400        $4,790       2.1x
#   x8      $27,858       $8,800       $19,058       3.2x
#   x16     $84,449      $17,600       $66,849       4.8x
#   x32    $256,000      $35,200      $220,800       7.3x
#   x64    $776,247      $70,400      $705,847      11.0x
#   x128  $2,353,487     $140,800    $2,212,687      16.7x
```

### 2.3 垂直スケーリングの実際の限界

```python
# クラウドプロバイダー別の最大インスタンスサイズ（2024年時点の概算）
cloud_max_instances = {
    "AWS": {
        "max_vcpu": 448,      # u-24tb1.metal
        "max_memory_gb": 24576,  # 24 TB
        "max_local_storage_tb": 60,
        "cost_per_hour": 218.40,
        "instance_type": "u-24tb1.metal",
        "use_case": "SAP HANA等のインメモリDB",
    },
    "GCP": {
        "max_vcpu": 416,      # m3-megamem-128
        "max_memory_gb": 8192,
        "max_local_storage_tb": 9,
        "cost_per_hour": 125.75,
        "instance_type": "m3-megamem-128",
        "use_case": "大規模OLAP",
    },
    "Azure": {
        "max_vcpu": 416,      # M416ms_v2
        "max_memory_gb": 11400,
        "max_local_storage_tb": 14,
        "cost_per_hour": 110.22,
        "instance_type": "M416ms_v2",
        "use_case": "SAP HANA",
    },
}

print("=== クラウド最大インスタンスサイズ ===\n")
for provider, spec in cloud_max_instances.items():
    print(f"  {provider}:")
    print(f"    インスタンス: {spec['instance_type']}")
    print(f"    vCPU: {spec['max_vcpu']}")
    print(f"    メモリ: {spec['max_memory_gb']:,} GB ({spec['max_memory_gb']/1024:.0f} TB)")
    print(f"    ストレージ: {spec['max_local_storage_tb']} TB")
    print(f"    コスト: ${spec['cost_per_hour']:.2f}/時間 "
          f"(${spec['cost_per_hour']*730:,.0f}/月)")
    print(f"    用途: {spec['use_case']}")
    print()

# 教訓:
# - 最大インスタンスでも vCPU 448、メモリ 24TB が上限
# - 月額 $160,000 以上のコストがかかる
# - これ以上のスケーリングには水平展開が必須
```

### 2.4 段階的スケーリング戦略

```
┌─────────────────────────────────────────────────────────────────────┐
│              段階的スケーリング戦略（DAU別）                          │
├────────────┬────────────────────────────────────────────────────────┤
│            │                                                        │
│  DAU < 1K  │  ■ 単一サーバー + マネージドDB                         │
│            │  ┌─────┐     ┌──────┐                                 │
│            │  │ App │────→│ DB   │                                 │
│            │  └─────┘     └──────┘                                 │
│            │  コスト: ~$50/月                                       │
│            │                                                        │
├────────────┼────────────────────────────────────────────────────────┤
│            │  ■ App複数台 + LB + DB垂直スケール                     │
│  DAU       │  ┌──┐  ┌─────┐ ┌─────┐     ┌──────┐                 │
│  1K-100K   │  │LB│─→│App 1│ │App 2│────→│ DB↑  │                 │
│            │  └──┘  └─────┘ └─────┘     └──────┘                 │
│            │  + Redis (セッション + キャッシュ)                     │
│            │  コスト: ~$500-2,000/月                                │
│            │                                                        │
├────────────┼────────────────────────────────────────────────────────┤
│            │  ■ App Auto Scaling + DB Read Replica + CDN            │
│  DAU       │  ┌──┐  ┌──────────┐     ┌────────┐ ┌──────────┐     │
│  100K-1M   │  │LB│─→│App x 3-10│────→│Primary │→│Replica x2│     │
│            │  └──┘  └──────────┘     └────────┘ └──────────┘     │
│            │  + CDN + Redis Cluster                                │
│            │  コスト: ~$2,000-10,000/月                             │
│            │                                                        │
├────────────┼────────────────────────────────────────────────────────┤
│            │  ■ マイクロサービス + DBシャーディング + 多層キャッシュ  │
│  DAU       │  ┌──┐  ┌──────────┐  ┌────────────┐                  │
│  1M-10M    │  │LB│─→│Service A │→│DB Shard 1-N│                  │
│            │  │  │─→│Service B │→│Cassandra   │                  │
│            │  │  │─→│Service C │→│Redis Cluster│                  │
│            │  └──┘  └──────────┘  └────────────┘                  │
│            │  + Kafka + Elasticsearch                              │
│            │  コスト: ~$10,000-100,000/月                           │
│            │                                                        │
├────────────┼────────────────────────────────────────────────────────┤
│            │  ■ マルチリージョン + カスタムインフラ                   │
│  DAU > 10M │  ┌──────────────────────────────────┐                 │
│            │  │ Region 1  │  Region 2  │ Region 3│                 │
│            │  │ Full Stack│ Full Stack │Full Stack│                 │
│            │  └──────────────────────────────────┘                 │
│            │  + Global LB + Cross-region replication               │
│            │  コスト: $100,000+/月                                  │
└────────────┴────────────────────────────────────────────────────────┘
```

---

## 3. ステートレス vs ステートフル

### 3.1 なぜステートレスが重要なのか

ステートレス設計は水平スケーリングの**前提条件**である。なぜなら、ステートフルなサーバーはリクエストの送り先が制限されるため、ロードバランサーが自由にリクエストを分配できないからである。

```
■ ステートフル（スケール困難）

  User A ──────────→ Server 1 (セッションA保持)
  User B ──────────→ Server 2 (セッションB保持)
  User C ──────────→ Server 3 (セッションC保持)

  問題:
  1. Server 2 が落ちると User B のセッションが消失
  2. User A は必ず Server 1 に行かなければならない（スティッキーセッション）
  3. Server 追加時に負荷が均等に分散されない
  4. キャパシティプランニングが困難

■ ステートレス（スケール容易）

  User A ─┐         ┌─ Server 1
  User B ─┼── LB ───┼─ Server 2    ← どのサーバーでもOK
  User C ─┘         └─ Server 3

              ↕
        ┌─────────────┐
        │ 共有ストア    │  (Redis / DB / S3)
        │ Session Data │
        │ User State   │
        │ File Upload  │
        └─────────────┘

  利点:
  1. 任意のサーバーが落ちても他が引き継ぐ
  2. LBが自由にラウンドロビンで分配可能
  3. Auto Scalingでサーバーの追加/削除が容易
  4. Blue/Green デプロイが容易
```

### 3.2 ステートレスAPIサーバーの実装

```python
from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel
import redis
import json
import uuid
import time
from typing import Optional

app = FastAPI()

# 共有ストア（Redis）への接続
# 全てのアプリサーバーが同じRedisクラスタを参照する
redis_client = redis.Redis(
    host="redis-cluster.internal",
    port=6379,
    decode_responses=True,
    # コネクションプールでパフォーマンス確保
    connection_pool=redis.ConnectionPool(max_connections=50)
)


# ========================================
# アンチパターン: ステートフルなAPI
# ========================================

# NG: メモリ内にセッションを保持
sessions_local = {}  # これはサーバーごとに異なる!
user_cache_local = {}  # Server 1 のキャッシュは Server 2 にはない

@app.get("/bad/profile")
def bad_get_profile(session_id: str):
    """
    アンチパターン: ローカルメモリにセッション保存

    問題点:
    1. Server 1 で作成したセッションは Server 2 では無効
    2. サーバー再起動でセッション全消失
    3. LBのスティッキーセッションが必要になり、負荷分散が不均等に
    4. Auto Scalingでサーバーが増減するたびにセッションが失われる
    """
    user = sessions_local.get(session_id)
    if not user:
        return {"error": "session not found"}
    return user


# ========================================
# ベストプラクティス: ステートレスなAPI
# ========================================

class SessionManager:
    """
    ステートレスなセッション管理

    設計原則:
    - サーバーは状態を持たない（Shared-Nothing Architecture）
    - 全ての状態は外部ストア（Redis）に保存
    - セッションIDのみをクライアントに返す
    - TTL（有効期限）で自動クリーンアップ
    """

    def __init__(self, redis_client: redis.Redis,
                 session_ttl: int = 3600):
        self.redis = redis_client
        self.session_ttl = session_ttl  # デフォルト1時間

    def create_session(self, user_id: str, user_data: dict) -> str:
        """新しいセッションを作成"""
        session_id = str(uuid.uuid4())
        session_data = {
            "user_id": user_id,
            "data": json.dumps(user_data),
            "created_at": time.time(),
        }
        # Redisに保存（TTL付き）
        self.redis.hset(f"session:{session_id}", mapping=session_data)
        self.redis.expire(f"session:{session_id}", self.session_ttl)
        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        """セッションを取得（どのサーバーからでも同じ結果）"""
        data = self.redis.hgetall(f"session:{session_id}")
        if not data:
            return None
        # アクセスのたびにTTLを延長（スライディングウィンドウ）
        self.redis.expire(f"session:{session_id}", self.session_ttl)
        return {
            "user_id": data["user_id"],
            "data": json.loads(data["data"]),
            "created_at": float(data["created_at"]),
        }

    def delete_session(self, session_id: str) -> bool:
        """セッションを削除（ログアウト時）"""
        return self.redis.delete(f"session:{session_id}") > 0


session_mgr = SessionManager(redis_client)


@app.get("/good/profile")
def good_get_profile(x_session_id: str = Header(...)):
    """
    ベストプラクティス: Redisにセッション保存

    どのサーバーにリクエストが来ても同じ結果を返す。
    LBのアルゴリズムに関係なく正常動作。
    """
    session = session_mgr.get_session(x_session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    return {"user_id": session["user_id"], "profile": session["data"]}


class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/auth/login")
def login(req: LoginRequest):
    """ログイン: セッションを作成してIDを返す"""
    # 実際にはDBでユーザー認証を行う
    user_data = {"name": req.username, "role": "user"}
    session_id = session_mgr.create_session(req.username, user_data)
    return {"session_id": session_id, "expires_in": 3600}


@app.post("/auth/logout")
def logout(x_session_id: str = Header(...)):
    """ログアウト: セッションを削除"""
    session_mgr.delete_session(x_session_id)
    return {"success": True}

# 出力例（curlで実行した場合）:
# $ curl -X POST http://localhost:8000/auth/login \
#   -H "Content-Type: application/json" \
#   -d '{"username": "gaku", "password": "secret"}'
# {"session_id": "a1b2c3d4-...", "expires_in": 3600}
#
# $ curl http://localhost:8000/good/profile \
#   -H "X-Session-Id: a1b2c3d4-..."
# {"user_id": "gaku", "profile": {"name": "gaku", "role": "user"}}
```

### 3.3 Twelve-Factor Appにおけるステートレス原則

```python
# Twelve-Factor App の12の原則とスケーラビリティとの関係

twelve_factors = [
    ("I.   コードベース", "1つのコードベースからN個のデプロイ",
     "全インスタンスが同じコードを実行"),
    ("II.  依存関係", "明示的に宣言し分離する",
     "requirements.txt / package.json で完結"),
    ("III. 設定", "環境変数に格納",
     "環境ごとの差異を環境変数で吸収。コードにDB接続先をハードコードしない"),
    ("IV.  バックエンド", "付帯サービスとしてアタッチ",
     "DB/Redis/MQは外部サービス。URLで接続先を切り替え可能"),
    ("V.   ビルド/リリース/実行", "3段階を厳密に分離",
     "ビルド成果物は不変。同じイメージを全サーバーにデプロイ"),
    ("VI.  プロセス", "ステートレスなプロセスとして実行 ★最重要★",
     "メモリやファイルシステムに状態を持たない。共有ストアを使う"),
    ("VII. ポートバインディング", "ポートバインディングでサービス公開",
     "自己完結型。リバースプロキシなしで直接ポートをリッスン"),
    ("VIII.並行性", "プロセスモデルによるスケールアウト",
     "水平スケーリング。プロセスを増やして負荷分散"),
    ("IX.  廃棄容易性", "高速な起動とグレースフルシャットダウン",
     "Auto Scalingでの迅速な追加/削除を可能にする"),
    ("X.   開発/本番一致", "開発、ステージング、本番をできるだけ一致させる",
     "Docker等で環境差異を最小化"),
    ("XI.  ログ", "イベントストリームとして扱う",
     "stdoutに出力。集約は外部ツール（Fluentd/CloudWatch）で"),
    ("XII. 管理プロセス", "ワンオフの管理タスクとして実行",
     "マイグレーション等は通常プロセスと同じ環境で実行"),
]

print("=== Twelve-Factor App とスケーラビリティ ===\n")
for factor, description, scalability_note in twelve_factors:
    print(f"  {factor}")
    print(f"    内容: {description}")
    print(f"    スケーラビリティ: {scalability_note}")
    print()

# 出力:
# === Twelve-Factor App とスケーラビリティ ===
#
#   I.   コードベース
#     内容: 1つのコードベースからN個のデプロイ
#     スケーラビリティ: 全インスタンスが同じコードを実行
#
#   II.  依存関係
#     内容: 明示的に宣言し分離する
#     スケーラビリティ: requirements.txt / package.json で完結
#   ...
```

### 3.4 水平スケーリング対応のワーカー設計

```python
import hashlib
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Task:
    id: str
    payload: dict
    priority: int = 0

class ConsistentHashRing:
    """
    コンシステントハッシングによるタスク分配

    なぜ単純なモジュロ演算（hash % N）ではダメなのか:
    - ワーカーをN=3→4に増やすと、ほぼ全てのタスクの割り当て先が変わる
    - キャッシュがあった場合、全てのキャッシュが無効になる

    コンシステントハッシングでは:
    - ワーカーをN=3→4に増やしても、約25%のタスクしか移動しない
    - K/N のタスクのみ再割り当て（K=タスク数、N=新ワーカー数）
    """

    def __init__(self, replicas: int = 150):
        self.replicas = replicas  # 各ノードの仮想ノード数
        self.ring = {}            # hash -> node_id
        self.sorted_keys = []     # ソート済みハッシュ値

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node_id: str):
        """ノードをリングに追加"""
        for i in range(self.replicas):
            virtual_key = f"{node_id}:{i}"
            h = self._hash(virtual_key)
            self.ring[h] = node_id
            self.sorted_keys.append(h)
        self.sorted_keys.sort()

    def remove_node(self, node_id: str):
        """ノードをリングから削除"""
        for i in range(self.replicas):
            virtual_key = f"{node_id}:{i}"
            h = self._hash(virtual_key)
            del self.ring[h]
            self.sorted_keys.remove(h)

    def get_node(self, key: str) -> str:
        """キーに対応するノードを取得"""
        if not self.ring:
            raise ValueError("No nodes in the ring")

        h = self._hash(key)
        # ハッシュ値以上の最小のキーを見つける（時計回り）
        for ring_key in self.sorted_keys:
            if ring_key >= h:
                return self.ring[ring_key]
        # 末尾に到達したら先頭に戻る
        return self.ring[self.sorted_keys[0]]


# デモ: ワーカー追加時のタスク再配置率を計測
def measure_redistribution():
    """ワーカー追加時のタスク再配置率を測定"""

    # 1000個のタスクを用意
    tasks = [f"task-{i}" for i in range(1000)]

    # 3ワーカーで初期配置
    ring_before = ConsistentHashRing(replicas=150)
    for worker in ["worker-1", "worker-2", "worker-3"]:
        ring_before.add_node(worker)

    assignments_before = {task: ring_before.get_node(task) for task in tasks}

    # 4ワーカーに増やす
    ring_after = ConsistentHashRing(replicas=150)
    for worker in ["worker-1", "worker-2", "worker-3", "worker-4"]:
        ring_after.add_node(worker)

    assignments_after = {task: ring_after.get_node(task) for task in tasks}

    # 再配置されたタスクをカウント
    moved = sum(1 for task in tasks
                if assignments_before[task] != assignments_after[task])

    print(f"=== コンシステントハッシング: ワーカー追加の影響 ===")
    print(f"  タスク数: {len(tasks)}")
    print(f"  ワーカー: 3 → 4")
    print(f"  再配置タスク: {moved} ({moved/len(tasks)*100:.1f}%)")
    print(f"  理論値: {1/4*100:.1f}% (1/N)")
    print()

    # 比較: 単純なモジュロ演算
    moved_modulo = sum(
        1 for task in tasks
        if (int(hashlib.md5(task.encode()).hexdigest(), 16) % 3 !=
            int(hashlib.md5(task.encode()).hexdigest(), 16) % 4)
    )
    print(f"  参考: モジュロ演算での再配置: {moved_modulo} "
          f"({moved_modulo/len(tasks)*100:.1f}%)")

measure_redistribution()

# 出力:
# === コンシステントハッシング: ワーカー追加の影響 ===
#   タスク数: 1000
#   ワーカー: 3 → 4
#   再配置タスク: 約250 (約25.0%)
#   理論値: 25.0% (1/N)
#
#   参考: モジュロ演算での再配置: 約750 (約75.0%)
```

---

## 4. 自動スケーリング（Auto Scaling）

### 4.1 Auto Scalingの動作原理

```
              ┌──────────────────────────────────────┐
              │         Auto Scaling Controller       │
              │                                      │
              │  ┌──────────┐    ┌───────────────┐   │
              │  │ Monitor  │───→│ Scaling Policy │   │
              │  │ (CPU,QPS,│    │ ・min: 2      │   │
              │  │  Memory, │    │ ・max: 20     │   │
              │  │  Custom) │    │ ・target: 70% │   │
              │  └──────────┘    └───────┬───────┘   │
              └──────────────────────────┼───────────┘
                                         │
            CPU < 30%:  Scale In         │        CPU > 70%:  Scale Out
         ┌──────────────────┐            │     ┌──────────────────────┐
         │ Remove instances │←───────────┼────→│   Add instances      │
         │ (cooldown: 5min) │            │     │   (cooldown: 3min)   │
         └──────────────────┘            │     └──────────────────────┘
                                         │
                          ┌──────────────┼──────────────┐
                          │              │              │
                     ┌────▼──┐     ┌────▼──┐     ┌────▼──┐
                     │Healthy│     │Healthy│     │  New  │
                     │  Srv  │     │  Srv  │     │  Srv  │ ← 新規追加
                     └───────┘     └───────┘     └───────┘
```

### 4.2 スケーリングポリシーの種類と実装

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
import time

class ScalingPolicyType(Enum):
    TARGET_TRACKING = "target_tracking"     # 目標追跡型
    STEP_SCALING = "step_scaling"           # ステップ型
    SCHEDULED = "scheduled"                 # スケジュール型
    PREDICTIVE = "predictive"              # 予測型


@dataclass
class ScalingConfig:
    """Auto Scaling の設定"""
    min_capacity: int = 2
    max_capacity: int = 20
    scale_out_cooldown: int = 180   # 秒（スケールアウト後の待機時間）
    scale_in_cooldown: int = 300    # 秒（スケールイン後の待機時間）


class TargetTrackingPolicy:
    """
    目標追跡型ポリシー（最も一般的）

    仕組み:
    - 特定のメトリクスを目標値に維持するように自動的にインスタンス数を調整
    - PID制御に似た仕組みで、オーバーシュートを防ぐ

    メリット:
    - 設定がシンプル（目標値を1つ指定するだけ）
    - AWSが自動的にCloudWatchアラームを作成・管理

    デメリット:
    - 単一メトリクスでしか判断できない
    - 突発的な負荷変動への反応が遅い
    """

    def __init__(self, target_value: float, metric_name: str = "CPUUtilization"):
        self.target_value = target_value
        self.metric_name = metric_name

    def calculate_desired(self, current_instances: int,
                          current_metric: float) -> int:
        """
        必要インスタンス数を計算

        公式: desired = ceil(current_instances * (current_metric / target_metric))
        """
        import math
        desired = math.ceil(
            current_instances * (current_metric / self.target_value)
        )
        return desired


class StepScalingPolicy:
    """
    ステップ型ポリシー

    仕組み:
    - メトリクスの値に応じて段階的にインスタンスを追加/削除
    - より細かい制御が可能

    メリット:
    - 負荷の大きさに応じた段階的な応答
    - 複数のステップを定義可能

    デメリット:
    - 設定が複雑
    - ステップ間の境界値の設定にチューニングが必要
    """

    def __init__(self):
        self.scale_out_steps = []  # (threshold, adjustment)
        self.scale_in_steps = []

    def add_scale_out_step(self, threshold: float, adjustment: int):
        """スケールアウトのステップを追加"""
        self.scale_out_steps.append((threshold, adjustment))
        self.scale_out_steps.sort(key=lambda x: x[0])

    def add_scale_in_step(self, threshold: float, adjustment: int):
        """スケールインのステップを追加"""
        self.scale_in_steps.append((threshold, adjustment))
        self.scale_in_steps.sort(key=lambda x: x[0], reverse=True)

    def calculate_adjustment(self, current_metric: float) -> int:
        """メトリクスに基づく調整数を計算"""
        # スケールアウト
        for threshold, adjustment in reversed(self.scale_out_steps):
            if current_metric >= threshold:
                return adjustment

        # スケールイン
        for threshold, adjustment in self.scale_in_steps:
            if current_metric <= threshold:
                return adjustment  # 負の値

        return 0  # 変更なし


class AutoScaler:
    """Auto Scalerのシミュレーター"""

    def __init__(self, config: ScalingConfig,
                 policy: TargetTrackingPolicy):
        self.config = config
        self.policy = policy
        self.current_instances = config.min_capacity
        self.last_scale_time = 0
        self.history = []

    def evaluate(self, current_metric: float, timestamp: float) -> dict:
        """現在のメトリクスに基づいてスケーリング判断"""
        desired = self.policy.calculate_desired(
            self.current_instances, current_metric)

        # min/maxの範囲に制限
        desired = max(self.config.min_capacity,
                      min(self.config.max_capacity, desired))

        # クールダウン期間のチェック
        elapsed = timestamp - self.last_scale_time
        if desired > self.current_instances:
            cooldown = self.config.scale_out_cooldown
            action = "SCALE_OUT"
        elif desired < self.current_instances:
            cooldown = self.config.scale_in_cooldown
            action = "SCALE_IN"
        else:
            action = "NO_CHANGE"
            cooldown = 0

        can_scale = elapsed >= cooldown or self.last_scale_time == 0

        result = {
            "timestamp": timestamp,
            "metric": current_metric,
            "current": self.current_instances,
            "desired": desired,
            "action": action,
            "can_scale": can_scale,
            "cooldown_remaining": max(0, cooldown - elapsed),
        }

        if can_scale and desired != self.current_instances:
            self.current_instances = desired
            self.last_scale_time = timestamp

        self.history.append(result)
        return result


# シミュレーション
config = ScalingConfig(min_capacity=2, max_capacity=20)
policy = TargetTrackingPolicy(target_value=70, metric_name="CPUUtilization")
scaler = AutoScaler(config, policy)

# 1日の負荷パターン（CPU使用率%）
# 深夜低負荷 → 朝方増加 → 昼ピーク → 夕方ピーク → 夜間低下
load_pattern = [
    (0, 20), (3, 15), (6, 30), (9, 65), (10, 80),
    (11, 90), (12, 95), (13, 85), (14, 75), (15, 70),
    (16, 60), (17, 80), (18, 90), (19, 85), (20, 70),
    (21, 50), (22, 35), (23, 25),
]

print("=== Auto Scaling シミュレーション (1日) ===\n")
print(f"{'時刻':>5s}  {'CPU%':>5s}  {'現在':>4s}  {'目標':>4s}  "
      f"{'アクション':>10s}  {'状態'}")
print("-" * 60)

for hour, cpu in load_pattern:
    timestamp = hour * 3600
    result = scaler.evaluate(cpu, timestamp)
    status = ""
    if not result["can_scale"] and result["action"] != "NO_CHANGE":
        status = f"(CD: {result['cooldown_remaining']:.0f}s)"
    print(f"  {hour:02d}:00  {cpu:>4d}%  {result['current']:>4d}  "
          f"{result['desired']:>4d}  {result['action']:>10s}  {status}")

# 出力:
# === Auto Scaling シミュレーション (1日) ===
#
#  時刻   CPU%   現在   目標   アクション   状態
# ------------------------------------------------------------
#  00:00    20%     2     2    NO_CHANGE
#  03:00    15%     2     2    NO_CHANGE
#  06:00    30%     2     2    NO_CHANGE
#  09:00    65%     2     2    NO_CHANGE
#  10:00    80%     2     3    SCALE_OUT
#  11:00    90%     3     4    SCALE_OUT
#  12:00    95%     4     6    SCALE_OUT
#  13:00    85%     6     8    SCALE_OUT
#  14:00    75%     8     9    SCALE_OUT
#  15:00    70%     9     9    NO_CHANGE
#  16:00    60%     9     8    SCALE_IN
#  17:00    80%     8    10    SCALE_OUT
#  18:00    90%    10    13    SCALE_OUT
#  19:00    85%    13    16    SCALE_OUT
#  20:00    70%    16    16    NO_CHANGE
#  21:00    50%    16    12    SCALE_IN
#  22:00    35%    12     6    SCALE_IN
#  23:00    25%     6     3    SCALE_IN
```

### 4.3 AWS CDK によるAuto Scaling設定（Infrastructure as Code）

```python
# AWS CDK (Python) でのAuto Scaling設定例
# 実際にAWSにデプロイ可能なコード

"""
from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    aws_autoscaling as autoscaling,
    aws_elasticloadbalancingv2 as elbv2,
    Duration,
)
from constructs import Construct

class ScalableWebServiceStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # VPC
        vpc = ec2.Vpc(self, "VPC", max_azs=3)

        # Auto Scaling Group
        asg = autoscaling.AutoScalingGroup(
            self, "ASG",
            vpc=vpc,
            instance_type=ec2.InstanceType("t3.medium"),
            machine_image=ec2.AmazonLinuxImage(
                generation=ec2.AmazonLinuxGeneration.AMAZON_LINUX_2
            ),
            min_capacity=2,        # 最小2台
            max_capacity=20,       # 最大20台
            desired_capacity=4,    # 初期4台
            health_check=autoscaling.HealthCheck.elb(
                grace=Duration.minutes(5)
            ),
        )

        # 目標追跡スケーリングポリシー (CPU)
        asg.scale_on_cpu_utilization(
            "CpuScaling",
            target_utilization_percent=70,
            cooldown=Duration.minutes(3),
            estimated_instance_warmup=Duration.minutes(5),
        )

        # 目標追跡スケーリングポリシー (リクエスト数)
        asg.scale_on_request_count(
            "RequestScaling",
            target_requests_per_minute=1000,
            # 各インスタンスあたり1000 req/min を目標
        )

        # ステップスケーリングポリシー (メモリ)
        asg.scale_on_metric(
            "MemoryScaling",
            metric=asg.metric("MemoryUtilization"),
            scaling_steps=[
                autoscaling.ScalingInterval(upper=30, change=-2),   # 30%以下: 2台削減
                autoscaling.ScalingInterval(lower=30, upper=70, change=0),  # 維持
                autoscaling.ScalingInterval(lower=70, upper=85, change=2),  # +2台
                autoscaling.ScalingInterval(lower=85, change=4),    # 85%以上: +4台
            ],
        )

        # スケジュールスケーリング（セール対応）
        asg.scale_on_schedule(
            "SaleStart",
            schedule=autoscaling.Schedule.cron(
                hour="8", minute="50"  # セール10分前
            ),
            min_capacity=10,
            max_capacity=50,
        )
        asg.scale_on_schedule(
            "SaleEnd",
            schedule=autoscaling.Schedule.cron(
                hour="23", minute="0"
            ),
            min_capacity=2,
            max_capacity=20,
        )

        # ALB
        lb = elbv2.ApplicationLoadBalancer(
            self, "ALB",
            vpc=vpc,
            internet_facing=True,
        )

        listener = lb.add_listener("Listener", port=443)
        listener.add_targets("AppTarget",
            port=8080,
            targets=[asg],
            health_check=elbv2.HealthCheck(
                path="/health",
                interval=Duration.seconds(30),
                healthy_threshold_count=2,
                unhealthy_threshold_count=3,
            ),
        )
"""

# 上記は実際のAWS CDKコードだが、ここではシミュレーションを示す
print("=== Auto Scaling 設定サマリー ===")
print()
print("  ポリシー1: CPU目標追跡 (70%)")
print("  ポリシー2: リクエスト目標追跡 (1000 req/min per instance)")
print("  ポリシー3: メモリステップスケーリング")
print("    - < 30%: -2台")
print("    - 30-70%: 維持")
print("    - 70-85%: +2台")
print("    - > 85%: +4台")
print("  ポリシー4: スケジュール（セール時 min=10, max=50）")
print()
print("  ※ 複数ポリシーが同時発火した場合、最も多いインスタンス数が採用される")
```

### 4.4 負荷テストによるスケーラビリティ検証

```python
import asyncio
import time
from dataclasses import dataclass, field
from typing import List
import statistics

@dataclass
class LoadTestResult:
    """負荷テストの結果"""
    total_requests: int = 0
    success: int = 0
    errors: int = 0
    latencies: List[float] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def throughput(self) -> float:
        return self.success / self.elapsed if self.elapsed > 0 else 0

    @property
    def p50(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        return sorted_lat[len(sorted_lat) // 2]

    @property
    def p95(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        return sorted_lat[int(len(sorted_lat) * 0.95)]

    @property
    def p99(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        return sorted_lat[int(len(sorted_lat) * 0.99)]

    @property
    def error_rate(self) -> float:
        return self.errors / self.total_requests * 100 if self.total_requests > 0 else 0


async def load_test(url: str, total_requests: int,
                    concurrency: int) -> LoadTestResult:
    """
    非同期HTTP負荷テスト

    使い方:
    asyncio.run(load_test("http://localhost:8080/api/health", 10000, 100))

    パラメータ:
    - url: テスト対象のエンドポイント
    - total_requests: 総リクエスト数
    - concurrency: 同時実行数（並行度）
    """
    import aiohttp

    semaphore = asyncio.Semaphore(concurrency)
    result = LoadTestResult(total_requests=total_requests)

    async def make_request(session):
        async with semaphore:
            start = time.monotonic()
            try:
                async with session.get(url) as resp:
                    await resp.text()
                    latency = (time.monotonic() - start) * 1000  # ms
                    result.latencies.append(latency)
                    if resp.status == 200:
                        result.success += 1
                    else:
                        result.errors += 1
            except Exception:
                result.errors += 1

    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session) for _ in range(total_requests)]
        start_time = time.monotonic()
        await asyncio.gather(*tasks)
        result.elapsed = time.monotonic() - start_time

    return result


def print_load_test_report(result: LoadTestResult, concurrency: int):
    """負荷テスト結果のレポートを出力"""
    print("=== 負荷テスト結果 ===\n")
    print(f"  リクエスト総数: {result.total_requests:,}")
    print(f"  同時実行数:     {concurrency}")
    print(f"  経過時間:       {result.elapsed:.2f}秒")
    print(f"  成功:           {result.success:,}")
    print(f"  エラー:         {result.errors:,} ({result.error_rate:.1f}%)")
    print(f"  スループット:   {result.throughput:,.0f} req/s")
    print(f"  レイテンシ P50: {result.p50:.1f}ms")
    print(f"  レイテンシ P95: {result.p95:.1f}ms")
    print(f"  レイテンシ P99: {result.p99:.1f}ms")

    # SLO判定
    print(f"\n  === SLO判定 ===")
    slo_latency = 200  # ms
    slo_error_rate = 1  # %
    slo_throughput = 1000  # req/s

    latency_ok = result.p99 < slo_latency
    error_ok = result.error_rate < slo_error_rate
    throughput_ok = result.throughput > slo_throughput

    print(f"  レイテンシ P99 < {slo_latency}ms: "
          f"{'PASS' if latency_ok else 'FAIL'} ({result.p99:.1f}ms)")
    print(f"  エラー率 < {slo_error_rate}%: "
          f"{'PASS' if error_ok else 'FAIL'} ({result.error_rate:.1f}%)")
    print(f"  スループット > {slo_throughput} req/s: "
          f"{'PASS' if throughput_ok else 'FAIL'} ({result.throughput:,.0f} req/s)")

# 使用方法:
# result = asyncio.run(load_test("http://localhost:8080/api/health", 10000, 100))
# print_load_test_report(result, 100)

# サンプル出力:
# === 負荷テスト結果 ===
#
#   リクエスト総数: 10,000
#   同時実行数:     100
#   経過時間:       8.53秒
#   成功:           9,950
#   エラー:         50 (0.5%)
#   スループット:   1,166 req/s
#   レイテンシ P50: 45.2ms
#   レイテンシ P95: 123.7ms
#   レイテンシ P99: 198.3ms
#
#   === SLO判定 ===
#   レイテンシ P99 < 200ms: PASS (198.3ms)
#   エラー率 < 1%: PASS (0.5%)
#   スループット > 1000 req/s: PASS (1,166 req/s)
```

---

## 5. データ層のスケーリング

アプリケーション層のスケーリング（ステートレスサーバーの水平展開）は比較的容易だが、データ層のスケーリングは本質的に困難である。なぜなら、データベースは**状態（state）を持つ**からである。

### 5.1 データ層スケーリングの階段

```
  データ層スケーリングの段階（コスト/複雑さ順）

  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  Level 1: 垂直スケール（最も簡単）                           │
  │  ┌──────────┐                                               │
  │  │ DB 8CPU  │  → │ DB 32CPU │  → │ DB 128CPU │             │
  │  │ 32GB RAM │    │ 128GB   │    │ 512GB     │             │
  │  └──────────┘                                               │
  │  限界: 物理的上限、コスト爆発                                │
  │                                                             │
  │  Level 2: Read Replica（読み込み分散）                       │
  │  ┌──────────┐    ┌───────────┐                              │
  │  │ Primary  │───→│ Replica 1 │  読み込みを分散              │
  │  │ (Write)  │───→│ Replica 2 │  書き込みは Primary のみ     │
  │  └──────────┘───→│ Replica 3 │                              │
  │                   └───────────┘                              │
  │  限界: 書き込みはスケールしない、レプリケーションラグ         │
  │                                                             │
  │  Level 3: コマンドクエリ分離（CQRS）                         │
  │  ┌──────────┐                  ┌───────────┐                │
  │  │ Write DB │  ──(Event)──→   │ Read DB   │                │
  │  │ (正規化) │                  │(非正規化) │                │
  │  └──────────┘                  └───────────┘                │
  │  Write側とRead側を完全に分離。読み書きの最適化が独立         │
  │                                                             │
  │  Level 4: シャーディング（データ分割）                        │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐                    │
  │  │ Shard 0  │ │ Shard 1  │ │ Shard 2  │                    │
  │  │ user 0-N │ │user N-2N │ │user 2N-3N│                    │
  │  └──────────┘ └──────────┘ └──────────┘                    │
  │  限界: クロスシャードクエリ、リシャーディングの複雑さ        │
  │                                                             │
  │  Level 5: 分散データベース（NewSQL/NoSQL）                   │
  │  ┌─────────────────────────────────────────┐                │
  │  │ CockroachDB / Spanner / TiDB / Cassandra│                │
  │  │ ・自動シャーディング                      │                │
  │  │ ・分散トランザクション                    │                │
  │  │ ・マルチリージョン対応                    │                │
  │  └─────────────────────────────────────────┘                │
  └─────────────────────────────────────────────────────────────┘
```

### 5.2 シャーディング戦略の比較

```python
# シャーディング戦略のシミュレーション
import hashlib

class ShardingStrategy:
    """シャーディング戦略の比較"""

    @staticmethod
    def range_based(user_id: int, shard_count: int) -> int:
        """
        レンジベースシャーディング

        仕組み: user_id の範囲でシャードを決定
        例: user 1-1M → Shard 0, user 1M-2M → Shard 1, ...

        長所: 範囲クエリが効率的
        短所: ホットスポット（新規ユーザーが集中するシャード）
        """
        users_per_shard = 1_000_000  # 100万ユーザー/シャード
        return min(user_id // users_per_shard, shard_count - 1)

    @staticmethod
    def hash_based(user_id: int, shard_count: int) -> int:
        """
        ハッシュベースシャーディング

        仕組み: user_id のハッシュ値でシャードを決定
        公式: shard = hash(user_id) % shard_count

        長所: データが均等に分散
        短所: 範囲クエリが全シャードに散る、リシャーディングが大変
        """
        h = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        return h % shard_count

    @staticmethod
    def directory_based(user_id: int, directory: dict) -> int:
        """
        ディレクトリベースシャーディング

        仕組み: ルックアップテーブル（ディレクトリ）でシャードを決定
        例: user_id → shard_id のマッピングをDBに保存

        長所: 柔軟なリバランシングが可能
        短所: ディレクトリがSPOFになる、追加のルックアップレイテンシ
        """
        return directory.get(user_id, 0)


# シャード分布のシミュレーション
def simulate_shard_distribution(strategy_name: str, strategy_func,
                                user_count: int, shard_count: int):
    """各シャーディング戦略のデータ分布を検証"""
    distribution = [0] * shard_count

    for user_id in range(user_count):
        shard = strategy_func(user_id, shard_count)
        distribution[shard] += 1

    avg = user_count / shard_count
    max_dev = max(abs(d - avg) / avg * 100 for d in distribution)
    min_count = min(distribution)
    max_count = max(distribution)

    print(f"\n=== {strategy_name} (ユーザー: {user_count:,}, シャード: {shard_count}) ===")
    for i, count in enumerate(distribution):
        bar = "#" * int(count / user_count * 100)
        deviation = (count - avg) / avg * 100
        print(f"  Shard {i}: {count:>8,} ({deviation:>+6.1f}%)  {bar}")
    print(f"  最大偏差: {max_dev:.1f}% (理想は0%)")

# レンジベース
simulate_shard_distribution(
    "レンジベース", ShardingStrategy.range_based, 3_500_000, 4)

# ハッシュベース
simulate_shard_distribution(
    "ハッシュベース", ShardingStrategy.hash_based, 3_500_000, 4)

# 出力:
# === レンジベース (ユーザー: 3,500,000, シャード: 4) ===
#   Shard 0: 1,000,000 (+14.3%)  #############################
#   Shard 1: 1,000,000 (+14.3%)  #############################
#   Shard 2: 1,000,000 (+14.3%)  #############################
#   Shard 3:   500,000 (-42.9%)  ##############
#   最大偏差: 42.9% (理想は0%)
#
# === ハッシュベース (ユーザー: 3,500,000, シャード: 4) ===
#   Shard 0:   875,123 (+0.0%)   #########################
#   Shard 1:   874,892 (-0.0%)   #########################
#   Shard 2:   875,034 (+0.0%)   #########################
#   Shard 3:   874,951 (-0.0%)   #########################
#   最大偏差: 0.0% (理想は0%)
```

---

## 6. 比較表

### 比較表1: 垂直 vs 水平スケーリング

| 項目 | 垂直スケーリング (Scale Up) | 水平スケーリング (Scale Out) |
|------|---------------------------|----------------------------|
| 方法 | より強力なマシンに交換 | マシンの台数を増やす |
| コスト曲線 | 指数的（高性能ほど割高） | 線形的（台数に比例） |
| 上限 | 物理的限界あり（最大448 vCPU） | 理論上は無制限 |
| ダウンタイム | 交換時に発生しうる | ローリングで無停止可能 |
| 複雑さ | 低い（アプリ変更不要） | 高い（分散処理の設計必要） |
| データ一貫性 | 容易（単一DB） | 困難（分散合意が必要） |
| 障害耐性 | SPOF（1台故障で全停止） | 1台故障でも他が継続 |
| 適するケース | DB、初期段階のスタートアップ | Webサーバー、APIサーバー |
| 適さないケース | 100万QPS以上 | DAU 1000以下 |
| 例 | AWS RDS のインスタンスサイズ変更 | ECS/K8sでのPod増加 |

### 比較表2: スケーリング戦略の比較

| 戦略 | 仕組み | レスポンス速度 | コスト効率 | 適するワークロード |
|------|--------|--------------|-----------|------------------|
| 予測スケーリング | 過去データからMLで予測 | 事前に準備済み | 高い | 周期的な負荷パターン |
| 目標追跡型 | メトリクスの目標値を維持 | 数分のラグ | 高い | 一般的なWebアプリ |
| ステップ型 | メトリクス閾値で段階調整 | 数分のラグ | 中程度 | 負荷変動が大きい |
| スケジュール | 時刻ベースで固定 | 事前に準備済み | 高い | イベント・セール |
| リアクティブ | メトリクス閾値で発火 | 数分のラグ | 中程度 | 突発的な負荷 |
| マニュアル | 人手で調整 | 遅い | 低い | 小規模・実験環境 |

### 比較表3: シャーディング戦略の比較

| 戦略 | データ分布 | 範囲クエリ | リシャーディング | 実装の複雑さ | 代表例 |
|------|----------|-----------|----------------|------------|--------|
| レンジベース | 偏りやすい | 効率的 | 比較的容易 | 低い | HBase, MongoDB |
| ハッシュベース | 均等 | 非効率 | 困難 | 中程度 | Cassandra, DynamoDB |
| ディレクトリベース | 制御可能 | 可能 | 柔軟 | 高い | カスタム実装 |
| コンシステントハッシュ | 均等 | 非効率 | 容易（K/Nのみ移動） | 高い | Amazon DynamoDB |

---

## 7. アンチパターン

### アンチパターン1: 早すぎる水平スケーリング

```
--- NG例 ---

DAU 1000 のサービスで最初から:
- Kubernetes 10ノードクラスタ（月額 $5,000+）
- 4つのマイクロサービス
- Kafka + Redis Cluster + Elasticsearch
- Istioサービスメッシュ

問題:
- 運用コストがトラフィックに対して50倍以上過大
- 2人のエンジニアでKubernetesの運用は回らない
- 分散システムのデバッグに時間を取られ、機能開発が止まる
- 障害時の原因特定が困難（「どのサービスが悪い？」）

--- OK例 ---

段階的アプローチ:
1. まず1台の垂直スケーリングで始める（Heroku/Railway/PaaS）
2. ボトルネックを計測してから分散化
3. 「スケールする設計」はしつつ、「スケールした実装」は必要になってから

目安:
- DAU < 10K:  単一サーバー + マネージドDB
- DAU < 100K: App 2-3台 + LB + DB垂直スケール
- DAU < 1M:   Auto Scaling + Read Replica + Redis
- DAU > 1M:   マイクロサービス検討開始

Jeff Bezosの言葉:
「スケーラビリティは後から追加できるが、
  スタートアップの速度は後からは取り戻せない」
```

### アンチパターン2: ステートフルサーバーの水平スケール

```python
# --- NG例 ---

class BadUserService:
    """アンチパターン: ステートフルなサービスの水平スケール"""

    def __init__(self):
        self.cache = {}  # インスタンスローカルのキャッシュ
        self.session_store = {}  # ローカルセッション

    def get_user(self, user_id: str) -> dict:
        """
        問題点:
        - Server 1 のキャッシュにあるデータは Server 2 では使えない
        - スティッキーセッションに依存するとLBが制限される
        - サーバー追加時にキャッシュヒット率が低下
        - サーバー障害時にキャッシュとセッションが全て消失
        """
        if user_id in self.cache:
            return self.cache[user_id]  # Server 1にしかない!
        user = {"id": user_id, "name": f"User {user_id}"}
        self.cache[user_id] = user
        return user


# --- OK例 ---

class GoodUserService:
    """ベストプラクティス: ステートレスなサービス"""

    def __init__(self, redis_client, db_client):
        self.redis = redis_client   # 共有キャッシュ
        self.db = db_client         # 共有DB

    def get_user(self, user_id: str) -> dict:
        """
        どのサーバーにリクエストが来ても同じ結果を返す

        キャッシュ戦略: Cache-Aside (Lazy Loading)
        1. Redisにキャッシュがあればそれを返す
        2. なければDBから取得してRedisにキャッシュ
        3. TTL付きで自動的に期限切れ
        """
        # 1. 共有キャッシュをチェック
        cached = self.redis.get(f"user:{user_id}")
        if cached:
            return json.loads(cached)

        # 2. DBから取得
        user = self.db.query(f"SELECT * FROM users WHERE id = %s", user_id)
        if not user:
            return None

        # 3. キャッシュに保存（TTL 5分）
        self.redis.setex(f"user:{user_id}", 300, json.dumps(user))
        return user
```

### アンチパターン3: スケーリングなしの無限リトライ

```
--- NG例 ---

サービスが過負荷状態の時に、クライアントが無限リトライ:

Client → (timeout) → Retry → (timeout) → Retry → (timeout) → ...

問題:
- タイムアウトしたリクエスト + リトライで負荷が2倍、3倍に増加
- 「リトライストーム」でサービスが完全にダウン
- 復旧しようとしても、待ちリクエストが膨大で再起動後すぐダウン

--- OK例 ---

Exponential Backoff + Jitter + Circuit Breaker:

1. 指数バックオフ: 1s → 2s → 4s → 8s ... で待ち時間を増加
2. ジッター: ランダムな遅延を追加して同時リトライを防ぐ
3. 最大リトライ回数: 3-5回で諦める
4. サーキットブレーカー: エラー率が高い場合はリクエスト自体を遮断

関連: [信頼性](./02-reliability.md) の回路遮断パターンで詳細解説
```

### アンチパターン4: キャッシュを考慮しないスケーリング

```
--- NG例 ---

「DBが遅いからサーバーを増やそう」

   LB → App x 10 → DB (1台)

結果: App が10台に増えてもDBの負荷は変わらない。
      むしろDB接続数が増えてさらに遅くなる。

--- OK例 ---

「DBが遅いから、まずキャッシュを入れよう」

   LB → App x 3 → Redis → DB (1台)

   キャッシュヒット率 80% の場合:
   - DBへのクエリが 1/5 に減少
   - レイテンシが 100ms → 5ms に改善（Redis応答時間）
   - App 3台で十分な性能に

教訓: スケーリングの前にキャッシュ。
      キャッシュの前にクエリ最適化。
      クエリ最適化の前にインデックス。

関連: [キャッシング](../01-components/01-caching.md) で詳細解説
```

---

## 8. 実践演習

### 演習1（基礎）: スケーリング戦略の選択

**課題:** 以下の3つのシナリオそれぞれに対して、最適なスケーリング戦略を選択し、その理由を説明してください。

```python
# 演習1: スケーリング戦略の選択

scenarios = [
    {
        "name": "社内チャットアプリ",
        "dau": 500,
        "peak_qps": 50,
        "data_growth": "月10GB",
        "budget": "$200/月",
        "team": "エンジニア1名（兼任）",
    },
    {
        "name": "ECサイト（セール時に10倍）",
        "dau": 100_000,
        "peak_qps": 5_000,
        "data_growth": "月100GB",
        "budget": "$5,000/月",
        "team": "エンジニア5名",
        "note": "年2回の大型セールで通常の10倍のトラフィック",
    },
    {
        "name": "動画配信サービス",
        "dau": 10_000_000,
        "peak_qps": 500_000,
        "data_growth": "月50TB",
        "budget": "$500,000/月",
        "team": "エンジニア50名",
        "note": "グローバル展開、4K動画対応",
    },
]

# TODO: 各シナリオに対して以下を回答
# 1. 推奨アーキテクチャ（図で表現）
# 2. スケーリング方針（垂直/水平、自動/手動）
# 3. データ層の構成
# 4. コスト配分の概算
```

**期待される出力の例:**

```
=== シナリオ1: 社内チャットアプリ ===

推奨: 単一サーバー + マネージドDB
- App: Heroku Hobby Dyno ($7/月)
- DB: Heroku Postgres Hobby ($0/月)
- WebSocket: Heroku上で直接

理由:
- DAU 500、QPS 50 なら単一プロセスで余裕
- 水平スケーリングの複雑さは不要
- 兼任エンジニア1名ではインフラ運用に時間を割けない
- PaaSで運用コストを最小化

将来的な移行ポイント:
- DAU 5,000 を超えたら、DBを垂直スケール
- DAU 50,000 を超えたら、App 2台 + LB 構成を検討
```

---

### 演習2（応用）: Auto Scaling ポリシーの設計

**課題:** 以下のワークロード特性を持つWebアプリケーションに対して、Auto Scalingポリシーを設計してください。

```python
# 演習2: Auto Scaling ポリシー設計

workload = {
    "service": "ニュースアプリAPI",
    "normal_qps": 2000,
    "peak_qps": 20000,       # 重大ニュース発生時
    "morning_spike": 8000,   # 朝の通勤時間帯
    "evening_spike": 6000,   # 夕方の通勤時間帯
    "latency_slo": "P99 < 200ms",
    "error_rate_slo": "< 0.1%",
    "instance_capacity": 500, # 1インスタンスあたりの処理能力 (req/s)
}

# TODO: 以下を設計
# 1. 通常時のスケーリングポリシー（目標追跡型）
# 2. スケジュールベースのスケーリング（朝夕の通勤時間帯）
# 3. 突発的な負荷（重大ニュース）への対策
# 4. クールダウン期間の設定と理由
# 5. 最小/最大インスタンス数の決定根拠
```

**期待される出力の例:**

```
=== ニュースアプリAPI Auto Scaling設計 ===

【基本設定】
  最小インスタンス数: 5  (2000 QPS / 500 cap * 余裕 1.25 = 5)
  最大インスタンス数: 50 (20000 QPS / 500 cap * 余裕 1.25 = 50)

【ポリシー1: 目標追跡（通常時）】
  メトリクス: ALB RequestCountPerTarget
  目標値: 400 req/s per instance (500 * 80% = 400)

【ポリシー2: スケジュール（通勤時間帯）】
  07:30 → min=17 (8000/500*1.05)
  09:30 → min=5 (通常に戻す)
  17:00 → min=13 (6000/500*1.05)
  19:00 → min=5

【ポリシー3: 突発負荷対策】
  ステップスケーリング:
  - QPS > 10000: +10台
  - QPS > 15000: +20台
  - クールダウン: 60秒（迅速な対応が必要）

【ポリシー4: クールダウン】
  スケールアウト: 60秒（ニュースの緊急性を考慮して短めに）
  スケールイン: 600秒（早すぎるスケールインを防ぐ）
```

---

### 演習3（発展）: データベースシャーディングの設計

**課題:** DAU 1000万のSNSサービスにおいて、メッセージテーブルのシャーディングを設計してください。

```python
# 演習3: シャーディング設計

system_spec = {
    "dau": 10_000_000,
    "messages_per_user_per_day": 50,
    "message_size_bytes": 500,
    "retention_years": 5,
    "read_pattern": "ユーザーの最新メッセージを時系列で表示",
    "write_pattern": "ユーザーが送信、グループメンバーに配信",
    "query_pattern": [
        "特定ユーザーの最新100件取得（最も頻繁）",
        "特定会話の全メッセージ取得",
        "全文検索（低頻度）",
    ],
}

# TODO: 以下を設計・回答
# 1. シャーディングキーの選定（なぜそのキーか、代替案との比較）
# 2. シャード数の決定（データ量ベースの計算）
# 3. シャーディング方式の選択（ハッシュ/レンジ/ディレクトリ）
# 4. クロスシャードクエリへの対応策
# 5. リシャーディング計画（将来のデータ増加への備え）
```

**期待される出力の例:**

```
=== メッセージテーブル シャーディング設計 ===

【データ量見積もり】
  1日: 10M * 50 * 500B = 250 GB/日
  1年: 250 * 365 = 91 TB/年
  5年: 455 TB

【シャーディングキー: user_id】
  理由:
  - 最も頻繁なクエリ（ユーザーの最新メッセージ）が単一シャードで完結
  - 書き込みもユーザー単位で1シャードに集中
  - ユーザー間のメッセージ偏りは比較的小さい

  代替案と不採用理由:
  - conversation_id: グループチャットでは良いが、1対1チャットでは非効率
  - message_id: 均等だが、ユーザーの全メッセージ取得が全シャード走査に
  - created_at: 時系列だが、最新データのシャードにホットスポット発生

【シャード数: 64】
  根拠: 5年分 455TB / 8TB per shard = 57 → 64 (2の冪乗)
  各シャードの容量: 約7.1 TB（余裕あり）
```

---

## 9. FAQ

### Q1: 垂直と水平、どちらから始めるべきですか？

初期段階では垂直スケーリングから始めるのが合理的である。理由は (1) アプリケーションコードの変更が不要、(2) 運用の複雑さが低い、(3) コストが予測しやすい、の3点。

垂直の限界に近づいたサイン:
- 単一サーバーのCPU使用率が常時70%を超える
- 月額コストが水平構成の2倍を超えた
- 次のスペックアップが物理的上限に近い
- SPOFのリスクが許容できなくなった

ただし、**設計は最初から水平スケーリングを見据えておくべき**である。具体的には、ステートレスな設計、設定の外部化、DBアクセスの抽象化を最初から行っておけば、移行がスムーズになる。

### Q2: データベースの水平スケーリングが難しいのはなぜですか？

データベースは**状態（state）を持つ**ため、水平スケーリング時にデータの一貫性と分散配置を両立させる必要がある。

具体的な課題:
1. **シャーディングキーの設計**: 不適切なキーはホットスポットやクロスシャードクエリを引き起こす
2. **クロスシャードクエリ**: JOINや集約クエリが複数シャードにまたがると性能が劇的に低下
3. **分散トランザクション**: 2Phase Commitは遅い、Sagaパターンは複雑
4. **リシャーディング**: データの再配置は長時間かかり、サービスに影響する
5. **レプリケーションラグ**: Primary→Replicaの同期遅延でデータ不整合が発生

詳細は [DBスケーリング](../01-components/04-database-scaling.md) で解説する。

### Q3: マイクロサービスにするとスケーラビリティは自動的に向上しますか？

自動的には向上しない。マイクロサービスは**独立したスケーリング**を可能にするが、以下の新たな複雑さが加わる:

- ネットワーク通信のオーバーヘッド（10-100倍の遅延）
- サービス間の依存関係管理（サービスメッシュの必要性）
- 分散トレーシングの必要性（Jaeger/Zipkin）
- データ一貫性の保証（分散トランザクション or 結果整合性）
- デプロイの複雑さ（サービスごとのCI/CD）

マイクロサービスへの移行は「チーム規模 > 50人」かつ「独立デプロイの必要性」が高い場合に検討すべきである。

詳細は [モノリス vs マイクロサービス](../02-architecture/00-monolith-vs-microservices.md) で解説する。

### Q4: スケーラビリティと性能（パフォーマンス）は同じですか？

異なる概念である。

- **性能（Performance）**: 現在の負荷でどれだけ速く処理できるか
  - 「100ユーザーでの応答時間は50ms」
- **スケーラビリティ（Scalability）**: 負荷が増えた時にどれだけ性能を維持できるか
  - 「100ユーザー→10万ユーザーでも応答時間は100ms以内」

高性能だがスケールしないシステム（例: 高度に最適化された単一サーバー）と、低性能だがスケールするシステム（例: 遅いが無限に横に並べられるサーバー）は、どちらも不適切である。両方を満たす設計が求められる。

### Q5: キャッシュを入れればスケーラビリティの問題は解決しますか？

キャッシュはスケーラビリティの重要な要素だが、「銀の弾丸」ではない。

キャッシュが有効なケース:
- 読み取りが書き込みよりはるかに多い（10:1以上）
- 同じデータへのアクセスが繰り返される（時間的局所性）
- データの鮮度に多少の遅延が許容される

キャッシュが効かないケース:
- 書き込みが多い（キャッシュ無効化が頻繁）
- アクセスパターンがランダム（キャッシュヒット率が低い）
- 強い一貫性が必要（キャッシュのデータが古い可能性がある）
- データ量が大きすぎてキャッシュに収まらない

詳細は [キャッシング](../01-components/01-caching.md) で解説する。

### Q6: Amdahlの法則とスケーラビリティの関係は？

Amdahlの法則は「システムの一部を改善しても、全体の改善は改善できない部分に制限される」という法則である。

```python
def amdahls_law(parallel_fraction: float, num_processors: int) -> float:
    """
    アムダールの法則でスケーリング効率を計算

    parallel_fraction: 並列化可能な部分の割合 (0.0 - 1.0)
    num_processors: プロセッサ数（スケーリング倍率）
    """
    serial_fraction = 1 - parallel_fraction
    speedup = 1 / (serial_fraction + parallel_fraction / num_processors)
    return speedup

# 並列化率による理論的スピードアップ上限
print("=== アムダールの法則: 並列化率による上限 ===\n")
print(f"{'並列化率':>8s}  {'2台':>6s}  {'4台':>6s}  {'8台':>6s}  "
      f"{'16台':>6s}  {'64台':>6s}  {'理論上限':>8s}")
print("-" * 60)

for pct in [0.5, 0.75, 0.9, 0.95, 0.99]:
    speedups = [amdahls_law(pct, n) for n in [2, 4, 8, 16, 64]]
    theoretical_max = 1 / (1 - pct)
    print(f"  {pct*100:>5.0f}%  " +
          "  ".join(f"{s:>5.1f}x" for s in speedups) +
          f"  {theoretical_max:>6.1f}x")

# 出力:
# === アムダールの法則: 並列化率による上限 ===
#
# 並列化率    2台    4台    8台   16台   64台  理論上限
# ------------------------------------------------------------
#    50%   1.3x   1.6x   1.8x   1.9x   2.0x    2.0x
#    75%   1.6x   2.3x   2.9x   3.4x   3.8x    4.0x
#    90%   1.8x   3.1x   4.7x   6.4x   8.8x   10.0x
#    95%   1.9x   3.5x   5.9x   9.1x  14.8x   20.0x
#    99%   2.0x   3.9x   7.5x  13.9x  39.3x  100.0x

# 教訓:
# - 並列化率が90%でも、64台に増やしてスピードアップは8.8倍止まり
# - 10%のシリアル部分が全体のボトルネックになる
# - スケーリングの前に、シリアル部分（DB、ロック、I/O）を最小化すべき
```

---

## 10. まとめ

| 項目 | ポイント |
|------|---------|
| スケーラビリティの定義 | 負荷増大に対して性能を維持しながら拡張できる能力 |
| 3つの次元 | 負荷、データ量、地理的分散の3軸で考える |
| AKF Scale Cube | X軸(クローン)、Y軸(機能分割)、Z軸(データ分割)の3つの拡張軸 |
| 垂直スケーリング | マシンスペック増強。簡単だが上限あり、コスト指数増加 |
| 水平スケーリング | マシン台数増加。上限なしだが設計の複雑さが増す |
| ステートレス設計 | 水平スケーリングの前提条件。状態は外部ストアに委譲 |
| Twelve-Factor App | スケーラブルなアプリ設計の12原則。特に第VI因子が重要 |
| Auto Scaling | 目標追跡/ステップ/スケジュール/予測の4種類のポリシー |
| データ層スケーリング | 垂直→Replica→CQRS→シャーディング→分散DBの段階 |
| 開始戦略 | 垂直から始めて、限界に近づいたら水平へ移行 |
| Amdahlの法則 | 並列化できない部分がスケーリングの理論的上限を決める |

---

## 次に読むべきガイド

- [信頼性](./02-reliability.md) -- スケーラビリティと信頼性の両立、障害時の自動回復
- [CAP定理](./03-cap-theorem.md) -- 分散システムの理論的制約と設計トレードオフ
- [ロードバランサー](../01-components/00-load-balancer.md) -- 水平スケーリングの要、L4/L7の違い
- [キャッシング](../01-components/01-caching.md) -- スケーリングの前にキャッシュ戦略を検討
- [メッセージキュー](../01-components/02-message-queue.md) -- 非同期処理による負荷平準化
- [DBスケーリング](../01-components/04-database-scaling.md) -- データ層のスケーリング戦略の詳細
- [モノリス vs マイクロサービス](../02-architecture/00-monolith-vs-microservices.md) -- アーキテクチャ選択の判断基準

### 関連する他のSkill

- [clean-code-principles](../../clean-code-principles/) -- コードレベルの設計原則。スケーラブルなコードの書き方
  - [結合度と凝集度](../../clean-code-principles/docs/00-principles/03-coupling-cohesion.md) -- サービス分割の基準
- [design-patterns-guide](../../design-patterns-guide/) -- スケーラビリティに関連するパターン
  - [Repository Pattern](../../design-patterns-guide/docs/04-architectural/01-repository-pattern.md) -- データアクセスの抽象化（DB移行を容易にする）
  - [Event Sourcing / CQRS](../../design-patterns-guide/docs/04-architectural/02-event-sourcing-cqrs.md) -- 読み書き分離によるスケーリング
  - [Observer Pattern](../../design-patterns-guide/docs/02-behavioral/00-observer.md) -- イベント駆動設計の基礎

---

## 参考文献

1. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Chapter 1: Reliable, Scalable, and Maintainable Applications. O'Reilly Media. -- スケーラビリティの理論的基礎
2. Abbott, M.L. & Fisher, M.T. (2015). *The Art of Scalability: Scalable Web Architecture, Processes, and Organizations for the Modern Enterprise*. 3rd Edition. Addison-Wesley. -- AKF Scale Cubeの原典
3. Hamilton, J. (2007). "On Designing and Deploying Internet-Scale Services." *LISA '07*. -- 大規模サービス設計の実践知
4. Amdahl, G.M. (1967). "Validity of the single processor approach to achieving large scale computing capabilities." *AFIPS Conference Proceedings*. -- アムダールの法則の原論文
5. Wiggins, A. (2012). "The Twelve-Factor App." -- https://12factor.net/ -- スケーラブルなアプリ設計の12原則
6. AWS Well-Architected Framework - Performance Efficiency Pillar -- https://docs.aws.amazon.com/wellarchitected/latest/performance-efficiency-pillar/ -- AWSのスケーリングベストプラクティス
7. Karger, D. et al. (1997). "Consistent Hashing and Random Trees." *STOC '97*. -- コンシステントハッシングの原論文
8. Dean, J. (2013). "The Tail at Scale." *Communications of the ACM*. -- テール遅延とスケーラビリティの関係

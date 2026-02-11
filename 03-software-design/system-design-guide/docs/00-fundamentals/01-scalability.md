# スケーラビリティ

> システムが負荷増大に対して性能を維持しながら成長できる能力を理解し、水平・垂直スケーリングの戦略と実装パターンを習得する。

## この章で学ぶこと

1. 垂直スケーリング（スケールアップ）と水平スケーリング（スケールアウト）の違いと使い分け
2. ステートレス設計によるスケーラビリティ確保の手法
3. 自動スケーリング（Auto Scaling）の仕組みと設計パターン

---

## 1. スケーラビリティとは

スケーラビリティとは、システムが**負荷の増大**（ユーザー数、データ量、トラフィック）に対して、**許容可能な性能レベルを維持しながら拡張できる能力**である。

```
負荷が10倍になったとき:
  ✅ スケーラブル: レイテンシが微増（200ms → 250ms）
  ❌ 非スケーラブル: レイテンシが爆発（200ms → 5000ms）
```

---

## 2. 垂直スケーリング vs 水平スケーリング

### ASCII図解1: 2つのスケーリング方向

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
```

### コード例1: 垂直スケーリングの限界シミュレーション

```python
import math

def vertical_scaling_cost(base_cost, scale_factor):
    """垂直スケーリングのコストは指数的に増加する"""
    # ムーアの法則の逆: 性能2倍にはコスト2倍以上
    return base_cost * (scale_factor ** 1.6)

def horizontal_scaling_cost(base_cost, scale_factor):
    """水平スケーリングのコストはほぼ線形"""
    overhead = 1.1  # 分散処理のオーバーヘッド10%
    return base_cost * scale_factor * overhead

base = 1000  # 基本コスト $1000/月
for factor in [1, 2, 4, 8, 16, 32]:
    v_cost = vertical_scaling_cost(base, factor)
    h_cost = horizontal_scaling_cost(base, factor)
    print(f"x{factor:2d}  垂直: ${v_cost:>10,.0f}  水平: ${h_cost:>10,.0f}  "
          f"差額: ${v_cost - h_cost:>10,.0f}")

# 出力:
# x 1  垂直:     $1,000  水平:     $1,100  差額:      $-100
# x 2  垂直:     $3,031  水平:     $2,200  差額:       $831
# x 4  垂直:     $9,190  水平:     $4,400  差額:     $4,790
# x 8  垂直:    $27,858  水平:     $8,800  差額:    $19,058
# x16  垂直:    $84,449  水平:    $17,600  差額:    $66,849
# x32  垂直:   $256,000  水平:    $35,200  差額:   $220,800
```

---

## 3. ステートレス vs ステートフル

### ASCII図解2: ステートフル vs ステートレスサーバー

```
■ ステートフル（スケール困難）

  User A ──────────→ Server 1 (セッションA保持)
  User B ──────────→ Server 2 (セッションB保持)
  User C ──────────→ Server 3 (セッションC保持)

  ※ Server 2 が落ちると User B のセッションが消失
  ※ User A は必ず Server 1 に行かなければならない

■ ステートレス（スケール容易）

  User A ─┐         ┌─ Server 1
  User B ─┼── LB ───┼─ Server 2    ← どのサーバーでもOK
  User C ─┘         └─ Server 3

              ↕
        ┌─────────────┐
        │ 共有ストア    │  (Redis / DB)
        │ Session Data │
        └─────────────┘
```

### コード例2: ステートレスAPIサーバー（FastAPI）

```python
from fastapi import FastAPI, Depends, Header
import redis
import json

app = FastAPI()
redis_client = redis.Redis(host="redis-cluster", port=6379, decode_responses=True)

# ❌ ステートフル: メモリ内にセッションを保持
sessions = {}  # これはサーバーごとに異なる！

@app.get("/bad/profile")
def bad_get_profile(session_id: str):
    """アンチパターン: ローカルメモリにセッション保存"""
    user = sessions.get(session_id)
    if not user:
        return {"error": "session not found"}
    return user

# ✅ ステートレス: 外部ストアにセッションを保持
@app.get("/good/profile")
def good_get_profile(session_id: str = Header(...)):
    """ベストプラクティス: Redisにセッション保存"""
    user_json = redis_client.get(f"session:{session_id}")
    if not user_json:
        return {"error": "session not found"}
    return json.loads(user_json)
```

### コード例3: 水平スケーリング対応のワーカー設計

```python
import hashlib
from dataclasses import dataclass

@dataclass
class Task:
    id: str
    payload: dict

class ScalableWorkerPool:
    """水平スケーリング対応のワーカープール"""

    def __init__(self, worker_count: int):
        self.worker_count = worker_count

    def assign_worker(self, task: Task) -> int:
        """コンシステントハッシングでタスクを分配"""
        hash_val = int(hashlib.md5(task.id.encode()).hexdigest(), 16)
        return hash_val % self.worker_count

    def scale_out(self, new_count: int):
        """ワーカー追加時のリバランシング"""
        old_count = self.worker_count
        self.worker_count = new_count
        print(f"Scaled: {old_count} → {new_count} workers")
        # 移動が必要なタスクの割合: 1 - (old/new) ≈ 追加分/全体
        migration_ratio = 1 - (old_count / new_count)
        print(f"推定タスク移動率: {migration_ratio:.1%}")

pool = ScalableWorkerPool(3)
pool.scale_out(6)
# Scaled: 3 → 6 workers
# 推定タスク移動率: 50.0%
```

---

## 4. 自動スケーリング

### ASCII図解3: Auto Scaling の動作フロー

```
              ┌──────────────────────────────────────┐
              │         Auto Scaling Controller       │
              │                                      │
              │  ┌──────────┐    ┌───────────────┐   │
              │  │ Monitor  │───→│ Scaling Policy │   │
              │  │ (CPU,QPS)│    │ ・min: 2      │   │
              │  └──────────┘    │ ・max: 20     │   │
              │                  │ ・target: 70% │   │
              │                  └───────┬───────┘   │
              └──────────────────────────┼───────────┘
                                         │
            CPU < 30%:  Scale In         │        CPU > 70%:  Scale Out
         ┌──────────────────┐            │     ┌──────────────────────┐
         │ Remove instances │←───────────┼────→│   Add instances      │
         │ (cooldown: 5min) │            │     │   (cooldown: 3min)   │
         └──────────────────┘            │     └──────────────────────┘
                                         │
   Instances: ■■□□□□□□□□□□□□□□□□□□      │  Instances: ■■■■■■■■□□□□□□□□□□□□
              (2/20 = 10%)               │             (8/20 = 40%)
```

### コード例4: Auto Scaling ポリシーの設定（AWS CDK風）

```python
# AWS CDKスタイルのAuto Scalingポリシー定義

class AutoScalingConfig:
    def __init__(self):
        self.min_capacity = 2
        self.max_capacity = 20
        self.target_cpu = 70           # CPU使用率 70% ターゲット
        self.scale_out_cooldown = 180  # 秒
        self.scale_in_cooldown = 300   # 秒

    def calculate_desired(self, current_instances: int, current_cpu: float) -> int:
        """目標追跡ポリシーで必要インスタンス数を計算"""
        # desired = current * (current_metric / target_metric)
        desired = current_instances * (current_cpu / self.target_cpu)
        desired = max(self.min_capacity, min(self.max_capacity, round(desired)))
        return desired

config = AutoScalingConfig()

# シナリオシミュレーション
scenarios = [
    (4, 30),   # 低負荷
    (4, 70),   # 適正負荷
    (4, 95),   # 高負荷
    (10, 90),  # 高負荷 + 多インスタンス
]

for instances, cpu in scenarios:
    desired = config.calculate_desired(instances, cpu)
    action = "維持" if desired == instances else ("増加" if desired > instances else "削減")
    print(f"現在: {instances}台, CPU: {cpu}% → 目標: {desired}台 ({action})")

# 出力:
# 現在: 4台, CPU: 30% → 目標: 2台 (削減)
# 現在: 4台, CPU: 70% → 目標: 4台 (維持)
# 現在: 4台, CPU: 95% → 目標: 5台 (増加)
# 現在: 10台, CPU: 90% → 目標: 13台 (増加)
```

### コード例5: 負荷テストスクリプト

```python
import asyncio
import aiohttp
import time

async def load_test(url: str, total_requests: int, concurrency: int):
    """簡易負荷テストでスケーラビリティを検証"""
    semaphore = asyncio.Semaphore(concurrency)
    results = {"success": 0, "error": 0, "latencies": []}

    async def make_request(session):
        async with semaphore:
            start = time.monotonic()
            try:
                async with session.get(url) as resp:
                    await resp.text()
                    latency = (time.monotonic() - start) * 1000
                    results["latencies"].append(latency)
                    results["success"] += 1
            except Exception:
                results["error"] += 1

    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session) for _ in range(total_requests)]
        start_time = time.monotonic()
        await asyncio.gather(*tasks)
        elapsed = time.monotonic() - start_time

    latencies = sorted(results["latencies"])
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0

    print(f"Total: {total_requests}, Concurrency: {concurrency}")
    print(f"Success: {results['success']}, Errors: {results['error']}")
    print(f"Throughput: {results['success'] / elapsed:.0f} req/s")
    print(f"P50: {p50:.1f}ms, P99: {p99:.1f}ms")

# asyncio.run(load_test("http://localhost:8080/api/health", 10000, 100))
```

---

## 5. 比較表

### 比較表1: 垂直 vs 水平スケーリング

| 項目 | 垂直スケーリング (Scale Up) | 水平スケーリング (Scale Out) |
|------|---------------------------|----------------------------|
| 方法 | より強力なマシンに交換 | マシンの台数を増やす |
| コスト曲線 | 指数的（高性能ほど割高） | 線形的（台数に比例） |
| 上限 | 物理的限界あり | 理論上は無制限 |
| ダウンタイム | 交換時に発生しうる | ローリングで無停止可能 |
| 複雑さ | 低い（アプリ変更不要） | 高い（分散処理の設計必要） |
| 適するケース | DB、初期段階のスタートアップ | Webサーバー、APIサーバー |
| SPOF | 1台故障で全停止 | 1台故障でも他が継続 |

### 比較表2: スケーリング戦略の比較

| 戦略 | 仕組み | レスポンス速度 | コスト効率 | 適するワークロード |
|------|--------|--------------|-----------|------------------|
| 予測スケーリング | 過去データから予測 | 事前に準備済み | 高い | 周期的な負荷パターン |
| リアクティブ | メトリクス閾値で発火 | 数分のラグ | 中程度 | 突発的な負荷 |
| スケジュール | 時刻ベースで固定 | 事前に準備済み | 高い | イベント・セール |
| マニュアル | 人手で調整 | 遅い | 低い | 小規模・実験環境 |

---

## 6. アンチパターン

### アンチパターン1: 早すぎる水平スケーリング

```
❌ ダメな例:
DAU 1000 のサービスで最初から Kubernetes + 10ノードクラスタ

問題:
- 運用コストがトラフィックに対して過大
- 分散システムのデバッグが困難
- チームの学習コストが高い

✅ 正しいアプローチ:
1. まず1台の垂直スケーリングで始める
2. ボトルネックを計測してから分散化
3. 「スケールする設計」はしつつ、「スケールした実装」は必要になってから

目安: DAU 10万未満ならモノリス+垂直スケールで十分なことが多い
```

### アンチパターン2: ステートフルサーバーの水平スケール

```
❌ ダメな例:
class UserService:
    def __init__(self):
        self.cache = {}  # インスタンスローカルのキャッシュ

    def get_user(self, user_id):
        if user_id in self.cache:
            return self.cache[user_id]  # Server 1のキャッシュにしかない！
        user = db.query(user_id)
        self.cache[user_id] = user
        return user

問題:
- Server 1 のキャッシュにあるデータは Server 2 では使えない
- スティッキーセッションに依存すると LB が制限される
- サーバー追加時にキャッシュヒット率が低下

✅ 正しいアプローチ:
- Redis等の共有キャッシュを使用
- サーバーはステートレスに保つ
- セッションは外部ストアに保存
```

---

## 7. FAQ

### Q1: 垂直と水平、どちらから始めるべきですか？

初期段階では垂直スケーリングから始めるのが合理的である。理由は (1) アプリケーションコードの変更が不要、(2) 運用の複雑さが低い、(3) コストが予測しやすい、の3点。垂直の限界（CPU/メモリの上限、コスト爆発、SPOF）に近づいたら水平への移行を検討する。目安として、単一サーバーのCPU使用率が常時70%を超えるか、月額コストが水平構成の2倍を超えたら移行時期である。

### Q2: データベースの水平スケーリングが難しいのはなぜですか？

データベースは**状態（state）を持つ**ため、水平スケーリング時にデータの一貫性と分散配置を両立させる必要がある。具体的には (1) シャーディングキーの設計、(2) クロスシャードクエリの性能低下、(3) リバランシング時のデータ移行、(4) 分散トランザクションのオーバーヘッド、が課題となる。詳細は「04-database-scaling.md」で解説する。

### Q3: マイクロサービスにするとスケーラビリティは自動的に向上しますか？

自動的には向上しない。マイクロサービスは**独立したスケーリング**を可能にするが、ネットワーク通信のオーバーヘッド、サービス間の依存関係管理、分散トレーシングの必要性など新たな複雑さが加わる。スケーラビリティ向上のためには、サービス境界の適切な設計、非同期通信の活用、回路遮断パターンの実装が不可欠である。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| スケーラビリティの定義 | 負荷増大に対して性能を維持しながら拡張できる能力 |
| 垂直スケーリング | マシンスペック増強。簡単だが上限あり、コスト指数増加 |
| 水平スケーリング | マシン台数増加。上限なしだが設計の複雑さが増す |
| ステートレス設計 | 水平スケーリングの前提条件。状態は外部ストアに |
| Auto Scaling | メトリクスベースで自動的にインスタンス数を調整 |
| 開始戦略 | 垂直から始めて、限界に近づいたら水平へ移行 |

---

## 次に読むべきガイド

- [信頼性](./02-reliability.md) — スケーラビリティと信頼性の両立
- [ロードバランサー](../01-components/00-load-balancer.md) — 水平スケールの要
- [DBスケーリング](../01-components/04-database-scaling.md) — データ層のスケーリング戦略

---

## 参考文献

1. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Chapter 1: Reliable, Scalable, and Maintainable Applications. O'Reilly Media.
2. Hamilton, J. (2007). "On Designing and Deploying Internet-Scale Services." *LISA '07*.
3. Abbott, M.L. & Fisher, M.T. (2015). *The Art of Scalability*. Addison-Wesley.

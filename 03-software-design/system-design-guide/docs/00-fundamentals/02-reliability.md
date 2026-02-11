# 信頼性

> システムが障害発生時にも正しく機能し続ける能力を理解し、フォールトトレランス・冗長化・障害復旧の設計パターンを習得する。

## この章で学ぶこと

1. 信頼性の定義と可用性（Availability）の定量的な計測方法
2. フォールトトレランスを実現する冗長化パターンとフェイルオーバー戦略
3. カオスエンジニアリングによる信頼性検証の手法

---

## 1. 信頼性とは

信頼性（Reliability）とは、システムが**障害（fault）が発生しても、期待される機能を正しく提供し続ける**能力を指す。障害をゼロにすることは不可能であるため、障害を**前提として設計**し、障害時の影響を最小化するアプローチが求められる。

```
障害 (Fault)   ≠  故障 (Failure)

Fault:   コンポーネントの一部が仕様から逸脱すること
Failure: システム全体がサービスを提供できなくなること

信頼性の目標: Fault が Failure に発展することを防ぐ
```

---

## 2. 可用性の計算

### コード例1: 可用性とダウンタイムの計算

```python
def availability_to_downtime(nines: int):
    """可用性のナイン数からダウンタイムを計算"""
    availability = 1 - (10 ** -nines)
    yearly_minutes = 365.25 * 24 * 60
    downtime_minutes = yearly_minutes * (1 - availability)

    if downtime_minutes >= 60:
        return f"{availability:.{nines}%} → {downtime_minutes / 60:.1f} 時間/年"
    elif downtime_minutes >= 1:
        return f"{availability:.{nines}%} → {downtime_minutes:.1f} 分/年"
    else:
        return f"{availability:.{nines}%} → {downtime_minutes * 60:.1f} 秒/年"

for nines in range(1, 6):
    print(f"{'9' * nines:>5s}: {availability_to_downtime(nines)}")

# 出力:
#     9: 90.0% → 876.6 時間/年
#    99: 99.00% → 87.7 時間/年
#   999: 99.900% → 8.8 時間/年
#  9999: 99.9900% → 52.6 分/年
# 99999: 99.99900% → 5.3 分/年
```

### ASCII図解1: 可用性レベルの目安

```
  可用性     年間ダウン    月間ダウン    用途の目安
  ─────────────────────────────────────────────────
  99%        3.65日        7.3時間      バッチ処理
  99.9%      8.76時間      43.8分       一般Webサービス
  99.95%     4.38時間      21.9分       ECサイト
  99.99%     52.6分        4.38分       決済システム
  99.999%    5.26分        26.3秒       航空管制、医療
  ─────────────────────────────────────────────────

  注意: 複合可用性 = 各コンポーネントの可用性の積
  例: Web(99.9%) × API(99.9%) × DB(99.9%) = 99.7%
```

### コード例2: 複合システムの可用性計算

```python
def series_availability(*components):
    """直列構成の可用性 = 各コンポーネントの可用性の積"""
    result = 1.0
    for a in components:
        result *= a
    return result

def parallel_availability(*components):
    """並列構成の可用性 = 1 - (各不可用性の積)"""
    result = 1.0
    for a in components:
        result *= (1 - a)
    return 1 - result

# 直列: LB → App → DB
serial = series_availability(0.999, 0.999, 0.999)
print(f"直列 (LB→App→DB): {serial:.6f} = {serial*100:.3f}%")
# 直列 (LB→App→DB): 0.997003 = 99.700%

# 並列: DB Primary + DB Replica
parallel_db = parallel_availability(0.999, 0.999)
print(f"並列 DB: {parallel_db:.6f} = {parallel_db*100:.4f}%")
# 並列 DB: 0.999999 = 99.9999%

# 組み合わせ: LB → App → (DB Primary || DB Replica)
combined = series_availability(0.999, 0.999, parallel_db)
print(f"組み合わせ: {combined:.6f} = {combined*100:.4f}%")
# 組み合わせ: 0.998001 = 99.8001%
```

---

## 3. 冗長化パターン

### ASCII図解2: Active-Passive vs Active-Active

```
■ Active-Passive (ホットスタンバイ)

  Client ──→ ┌──────────┐     ┌──────────┐
             │ Active   │────→│ Passive  │  (データ同期)
             │ (稼働中) │     │ (待機中) │
             └──────────┘     └──────────┘
                  │                │
                  ▼                │
             ┌──────────┐         │
             │ Service  │         │ Active障害時
             └──────────┘         │ 自動フェイルオーバー
                                  ▼
  Client ──→              ┌──────────┐
                          │ 旧Passive│ → 新Active に昇格
                          └──────────┘

■ Active-Active (負荷分散 + 冗長)

  Client ──→ ┌──────┐     ┌──────────┐     ┌──────┐
             │  LB  │────→│ Active 1 │←───→│Active│
             │      │────→│ Active 2 │     │  2   │
             └──────┘     └──────────┘     └──────┘
                              ↕  (双方向同期)
                          どちらが落ちても即座にもう一方が処理
```

### コード例3: サーキットブレーカーパターン

```python
import time
from enum import Enum
from dataclasses import dataclass, field

class CircuitState(Enum):
    CLOSED = "closed"        # 正常（リクエスト通過）
    OPEN = "open"            # 遮断（リクエスト拒否）
    HALF_OPEN = "half_open"  # 試行（一部リクエスト通過）

@dataclass
class CircuitBreaker:
    """サーキットブレーカー: 障害の連鎖を防ぐ"""
    failure_threshold: int = 5          # 障害回数の閾値
    recovery_timeout: float = 30.0      # 秒
    success_threshold: int = 3          # 半開→閉へ必要な成功回数

    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    failure_count: int = field(default=0, init=False)
    success_count: int = field(default=0, init=False)
    last_failure_time: float = field(default=0, init=False)

    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                print("[Circuit] OPEN → HALF_OPEN: 試行開始")
            else:
                raise Exception("Circuit OPEN: リクエスト拒否")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                print("[Circuit] HALF_OPEN → CLOSED: 正常復帰")

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            print(f"[Circuit] → OPEN: {self.failure_count}回連続障害")
```

### コード例4: リトライ with 指数バックオフ

```python
import random
import time
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=60.0, jitter=True):
    """指数バックオフ + ジッターによるリトライデコレータ"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        print(f"[Retry] 最大リトライ回数({max_retries})に到達")
                        raise

                    delay = min(base_delay * (2 ** attempt), max_delay)
                    if jitter:
                        delay *= random.uniform(0.5, 1.5)

                    print(f"[Retry] 試行{attempt+1}失敗, "
                          f"{delay:.1f}秒後にリトライ: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry_with_backoff(max_retries=4, base_delay=0.5)
def call_external_api(url: str):
    """外部APIの呼び出し（障害時に自動リトライ）"""
    import requests
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.json()
```

---

## 4. フェイルオーバー戦略

### ASCII図解3: フェイルオーバーの種類

```
■ コールドフェイルオーバー
  ┌─────────┐    障害     ┌─────────┐
  │ Primary │ ──×──→     │ Standby │  ← 起動に数分かかる
  │ (稼働)  │            │ (停止)  │     (コスト最小)
  └─────────┘            └─────────┘
  復旧時間: 数分〜数十分

■ ウォームフェイルオーバー
  ┌─────────┐    障害     ┌─────────┐
  │ Primary │ ──×──→     │ Standby │  ← 起動済み、データ同期に遅延
  │ (稼働)  │            │ (低速)  │     (中程度コスト)
  └─────────┘            └─────────┘
  復旧時間: 数十秒〜数分

■ ホットフェイルオーバー
  ┌─────────┐    障害     ┌─────────┐
  │ Primary │ ──×──→     │ Standby │  ← 同一状態で同期稼働
  │ (稼働)  │←──同期──→  │ (稼働)  │     (コスト最大)
  └─────────┘            └─────────┘
  復旧時間: 数秒以内
```

### コード例5: ヘルスチェックによる障害検知

```python
import asyncio
import aiohttp
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class HealthStatus:
    url: str
    healthy: bool
    latency_ms: float
    checked_at: datetime
    error: Optional[str] = None

class HealthChecker:
    """定期ヘルスチェックによる障害検知"""

    def __init__(self, targets: list[str], interval: float = 10.0,
                 timeout: float = 3.0, unhealthy_threshold: int = 3):
        self.targets = targets
        self.interval = interval
        self.timeout = timeout
        self.unhealthy_threshold = unhealthy_threshold
        self.failure_counts: dict[str, int] = {t: 0 for t in targets}
        self.statuses: dict[str, HealthStatus] = {}

    async def check_one(self, session: aiohttp.ClientSession, url: str):
        start = asyncio.get_event_loop().time()
        try:
            async with session.get(f"{url}/health",
                                   timeout=aiohttp.ClientTimeout(total=self.timeout)) as resp:
                latency = (asyncio.get_event_loop().time() - start) * 1000
                healthy = resp.status == 200
                self.statuses[url] = HealthStatus(
                    url=url, healthy=healthy,
                    latency_ms=latency, checked_at=datetime.now()
                )
                if healthy:
                    self.failure_counts[url] = 0
                else:
                    self._record_failure(url, f"HTTP {resp.status}")
        except Exception as e:
            self._record_failure(url, str(e))

    def _record_failure(self, url: str, error: str):
        self.failure_counts[url] += 1
        count = self.failure_counts[url]
        if count >= self.unhealthy_threshold:
            print(f"[ALERT] {url} が {count}回連続失敗 → フェイルオーバー発動")
            self._trigger_failover(url)

    def _trigger_failover(self, url: str):
        print(f"[FAILOVER] {url} をロードバランサーから除外")
        # LBのバックエンドプールから除外するAPI呼び出し
```

### コード例6: バルクヘッドパターン

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class BulkheadPattern:
    """バルクヘッド: リソースを分離して障害の影響範囲を制限"""

    def __init__(self):
        # サービスごとに独立したスレッドプールを割り当て
        self.pools = {
            "payment":       ThreadPoolExecutor(max_workers=10, thread_name_prefix="payment"),
            "notification":  ThreadPoolExecutor(max_workers=5,  thread_name_prefix="notif"),
            "analytics":     ThreadPoolExecutor(max_workers=3,  thread_name_prefix="analytics"),
        }
        # サービスごとに独立したセマフォ
        self.semaphores = {
            "payment":      asyncio.Semaphore(10),
            "notification": asyncio.Semaphore(5),
            "analytics":    asyncio.Semaphore(3),
        }

    async def call_service(self, service_name: str, func, *args):
        sem = self.semaphores.get(service_name)
        if sem is None:
            raise ValueError(f"Unknown service: {service_name}")

        async with sem:
            # notification が詰まっても payment には影響しない
            pool = self.pools[service_name]
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(pool, func, *args)
```

---

## 5. 比較表

### 比較表1: フェイルオーバー戦略の比較

| 項目 | コールド | ウォーム | ホット |
|------|---------|---------|--------|
| 復旧時間 (RTO) | 数分〜数十分 | 数十秒〜数分 | 数秒以内 |
| コスト | 低い | 中程度 | 高い（2倍） |
| データ損失 (RPO) | 大きい | 中程度 | ほぼゼロ |
| 運用複雑さ | 低い | 中程度 | 高い |
| 適するシステム | 開発環境、バッチ | 一般Web | 決済、医療 |

### 比較表2: 信頼性パターンの比較

| パターン | 目的 | 実装複雑度 | 効果 |
|---------|------|-----------|------|
| サーキットブレーカー | 障害の連鎖防止 | 中 | 高（カスケード障害防止） |
| リトライ+バックオフ | 一時的障害の回復 | 低 | 中（transient fault対応） |
| バルクヘッド | 障害の分離 | 中 | 高（影響範囲限定） |
| ヘルスチェック | 障害の検知 | 低 | 高（早期検知） |
| 冗長化 | 単一障害点の排除 | 高 | 非常に高い |
| グレースフルデグラデーション | 部分的サービス継続 | 中 | 高（UX維持） |

---

## 6. アンチパターン

### アンチパターン1: リトライストーム

```
❌ ダメな例:
全クライアントが同時に即座リトライ → 障害中のサーバーに負荷集中

  Client 1 ──retry──→
  Client 2 ──retry──→  ┌─────────────┐
  Client 3 ──retry──→  │ 障害中の     │ ← さらに負荷が増大
  Client 4 ──retry──→  │ サーバー     │    して復旧不能に
  Client 5 ──retry──→  └─────────────┘

✅ 正しいアプローチ:
- 指数バックオフ: 1s → 2s → 4s → 8s → ...
- ジッター追加: ランダムな遅延を加えてリトライを分散
- 最大リトライ回数を設定
- サーキットブレーカーと組み合わせる
```

### アンチパターン2: 単一障害点（SPOF）の見落とし

```
❌ 見落としがちなSPOF:
1. DNS: DNSサーバーが1台だけ
2. ロードバランサー: LB自体が冗長化されていない
3. 設定サーバー: 設定管理が1台のサーバーに依存
4. 認証サービス: 認証が落ちると全サービス停止
5. データベース: Primary 1台 + Replica 0台

✅ チェックリスト:
□ 全コンポーネントに冗長性があるか？
□ フェイルオーバーは自動化されているか？
□ フェイルオーバーのテストを定期的に実施しているか？
□ 依存サービスの障害時にグレースフルデグラデーション可能か？
```

---

## 7. FAQ

### Q1: SLA、SLO、SLI の違いは何ですか？

**SLI**（Service Level Indicator）は実測値（例: レイテンシP99 = 150ms）。**SLO**（Service Level Objective）はチーム内部の目標値（例: P99 < 200ms を99.9%の時間維持）。**SLA**（Service Level Agreement）は顧客との契約（例: 可用性99.95%を下回った場合クレジット返金）。SLA違反は金銭的ペナルティを伴うため、SLOはSLAより厳しく設定するのが通例である。

### Q2: カオスエンジニアリングとは何ですか？

カオスエンジニアリングは、本番環境で意図的に障害を注入し、システムの信頼性を検証する手法である。Netflixの「Chaos Monkey」が有名で、ランダムにインスタンスを停止させることでフォールトトレランスをテストする。実施の手順は (1) 定常状態の定義、(2) 仮説の設定（例: サーバー1台停止でもレイテンシP99は500ms以内）、(3) 障害の注入、(4) 結果の観測と改善、である。

### Q3: RPOとRTOの違いは何ですか？

**RPO**（Recovery Point Objective）は「どの時点のデータまで復旧できるか」を示す。例えば RPO = 1時間 なら、障害発生時に最大1時間分のデータが失われる可能性がある。**RTO**（Recovery Time Objective）は「障害発生からサービス復旧までの時間」を示す。例えば RTO = 15分 なら、15分以内にサービスを再開しなければならない。RPOはバックアップ頻度、RTOはフェイルオーバー方式で制御する。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 信頼性の定義 | 障害時にも正しく機能し続ける能力 |
| 可用性の計算 | 直列=積、並列=1-(不可用性の積) |
| 冗長化 | Active-Passive / Active-Active で SPOF を排除 |
| サーキットブレーカー | 障害の連鎖（カスケード障害）を防止 |
| リトライ戦略 | 指数バックオフ + ジッターで負荷を分散 |
| バルクヘッド | リソースを分離して障害の影響範囲を限定 |

---

## 次に読むべきガイド

- [CAP定理](./03-cap-theorem.md) — 分散システムにおける信頼性のトレードオフ
- [ロードバランサー](../01-components/00-load-balancer.md) — 冗長化された通信の振り分け
- [メッセージキュー](../01-components/02-message-queue.md) — 非同期処理による信頼性向上

---

## 参考文献

1. Nygard, M.T. (2018). *Release It!: Design and Deploy Production-Ready Software*, 2nd Edition. Pragmatic Bookshelf.
2. Rosenthal, C. & Jones, N. (2020). *Chaos Engineering: System Resiliency in Practice*. O'Reilly Media.
3. Beyer, B. et al. (2016). *Site Reliability Engineering*. O'Reilly Media. https://sre.google/sre-book/

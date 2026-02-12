# 信頼性（Reliability）

> システムが障害発生時にも正しく機能し続ける能力を理解し、フォールトトレランス・冗長化・障害復旧の設計パターンを、可用性計算・サーキットブレーカー・カオスエンジニアリングの実践を通じて習得する。

---

## この章で学ぶこと

1. 信頼性の定義と可用性（Availability）の定量的な計測方法、SLA/SLO/SLI の関係
2. フォールトトレランスを実現する冗長化パターン（Active-Passive、Active-Active）とフェイルオーバー戦略
3. サーキットブレーカー・リトライ・バルクヘッド等のレジリエンスパターンの実装と検証

---

## 前提知識

| トピック | 内容 | 参照先 |
|---------|------|--------|
| ネットワーク基礎 | TCP/IP、DNS、HTTP の基本概念 | [Web/ネットワーク基礎](../../../04-web-and-network/) |
| スケーラビリティ | 垂直/水平スケーリングの概念 | [スケーラビリティ](./01-scalability.md) |
| Python 基礎 | asyncio、dataclass、デコレータ | [プログラミング基礎](../../../02-programming/) |
| 分散システムの概念 | ノード、レプリケーションの基本理解 | [CAP定理](./03-cap-theorem.md) |

---

## 1. 信頼性とは

信頼性（Reliability）とは、システムが**障害（fault）が発生しても、期待される機能を正しく提供し続ける**能力を指す。障害をゼロにすることは不可能であるため、障害を**前提として設計**し、障害時の影響を最小化するアプローチが求められる。

### 1.1 Fault と Failure の区別

```
障害 (Fault)   ≠  故障 (Failure)

Fault:   コンポーネントの一部が仕様から逸脱すること
         例: ディスク1台の物理故障、ネットワークパケットの損失
Failure: システム全体がサービスを提供できなくなること
         例: Webサイトにアクセス不能、データの完全消失

信頼性の目標: Fault が Failure に発展することを防ぐ
         → フォールトトレラント（耐障害性）な設計
```

この区別は極めて重要である。個々のコンポーネントで Fault が発生しても、システム全体として Failure にならないように設計するのが信頼性エンジニアリングの本質である。

### 1.2 障害の3つの分類

信頼性設計では、障害を以下の3カテゴリに分類して対策を立てる。

```
┌─────────────────────────────────────────────────────────────┐
│                    障害の分類                                  │
├─────────────────┬───────────────────┬───────────────────────┤
│  ハードウェア障害  │  ソフトウェア障害    │   ヒューマンエラー     │
├─────────────────┼───────────────────┼───────────────────────┤
│ ・ディスク故障    │ ・バグ             │ ・設定ミス            │
│ ・メモリ破損      │ ・メモリリーク       │ ・誤ったデプロイ       │
│ ・電源障害       │ ・デッドロック       │ ・手順の誤り          │
│ ・ネットワーク断   │ ・カスケード障害     │ ・容量の見積もりミス    │
├─────────────────┼───────────────────┼───────────────────────┤
│ 対策:           │ 対策:              │ 対策:                │
│ 冗長化、RAID    │ テスト、監視        │ 自動化、レビュー       │
│ ホットスペア     │ カオスエンジニアリング │ ガードレール           │
└─────────────────┴───────────────────┴───────────────────────┘
```

### 1.3 WHY: なぜ信頼性が重要か

信頼性が不十分な場合の影響を定量的に示す。

```
ダウンタイムのビジネスインパクト:
─────────────────────────────────────────
Amazon: 1分のダウンで約 $220,000 の損失（2024年推計）
Google: 5分のダウンで約 $545,000 の損失
Facebook: 2021年の6時間障害で推定 $60M の損失

ECサイトの例:
  年間売上 10億円、可用性 99.9% の場合
  → 年間ダウンタイム 8.76時間
  → 損失額 ≒ 10億 × (8.76 / 8760) ≒ 100万円

  可用性を 99.99% に改善した場合
  → 年間ダウンタイム 52.6分
  → 損失額 ≒ 10億 × (52.6 / 525600) ≒ 10万円
  → 年間 90万円の損失削減
─────────────────────────────────────────
```

---

## 2. 可用性の計算

### 2.1 可用性とダウンタイム

可用性（Availability）は、システムが正常に稼働している時間の割合であり、「ナイン」の数で表現される。

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
  99%        3.65日        7.3時間      バッチ処理、内部ツール
  99.9%      8.76時間      43.8分       一般Webサービス
  99.95%     4.38時間      21.9分       ECサイト
  99.99%     52.6分        4.38分       決済システム、SaaS
  99.999%    5.26分        26.3秒       航空管制、医療、通信
  ─────────────────────────────────────────────────

  注意: 複合可用性 = 各コンポーネントの可用性の積
  例: Web(99.9%) × API(99.9%) × DB(99.9%) = 99.7%
```

### 2.2 複合システムの可用性

システム全体の可用性は、構成（直列・並列）によって大きく変わる。

### コード例2: 複合システムの可用性計算

```python
from typing import List

def series_availability(*components: float) -> float:
    """直列構成の可用性 = 各コンポーネントの可用性の積

    直列構成では全コンポーネントが動作しないとシステムが停止する。
    例: LB → App → DB（どれか1つ落ちるとサービス停止）
    """
    result = 1.0
    for a in components:
        result *= a
    return result

def parallel_availability(*components: float) -> float:
    """並列構成の可用性 = 1 - (各不可用性の積)

    並列構成では全コンポーネントが同時に故障しないとシステムは停止しない。
    例: DB Primary || DB Replica（両方同時に落ちなければOK）
    """
    result = 1.0
    for a in components:
        result *= (1 - a)
    return 1 - result

def format_availability(name: str, value: float) -> str:
    """可用性を分かりやすい形式で表示"""
    yearly_downtime_min = 525960 * (1 - value)
    return f"{name}: {value:.8f} ({value*100:.4f}%) → 年間ダウンタイム {yearly_downtime_min:.1f}分"

# === 直列構成 ===
serial = series_availability(0.999, 0.999, 0.999)
print(format_availability("直列 (LB→App→DB)", serial))
# 直列 (LB→App→DB): 0.99700300 (99.7003%) → 年間ダウンタイム 1577.9分

# === 並列構成 ===
parallel_db = parallel_availability(0.999, 0.999)
print(format_availability("並列 DB (Primary||Replica)", parallel_db))
# 並列 DB (Primary||Replica): 0.99999900 (99.9999%) → 年間ダウンタイム 0.5分

# === 組み合わせ ===
combined = series_availability(0.999, 0.999, parallel_db)
print(format_availability("LB→App→(DB||DB)", combined))
# LB→App→(DB||DB): 0.99800100 (99.8001%) → 年間ダウンタイム 1051.9分

# === 全レイヤーを冗長化 ===
parallel_lb = parallel_availability(0.999, 0.999)
parallel_app = parallel_availability(0.999, 0.999)
fully_redundant = series_availability(parallel_lb, parallel_app, parallel_db)
print(format_availability("(LB||LB)→(App||App)→(DB||DB)", fully_redundant))
# (LB||LB)→(App||App)→(DB||DB): 0.99999700 (99.9997%) → 年間ダウンタイム 1.6分
```

### ASCII図解2: 直列と並列の可用性

```
■ 直列構成（全てが動作する必要あり）

  Client → [LB] → [App] → [DB]
           99.9%   99.9%   99.9%

  合計 = 0.999 × 0.999 × 0.999 = 0.997 = 99.7%
  → コンポーネントが増えるほど可用性が下がる

■ 並列構成（冗長化）

  Client → [LB Active ] → [App 1] → [DB Primary]
           [LB Standby]   [App 2]   [DB Replica]
           99.9999%        99.9999%   99.9999%

  合計 = 0.999999 × 0.999999 × 0.999999 ≒ 99.9997%
  → 冗長化により各層の可用性を大幅に向上

■ 可用性改善の法則:
  直列に足すと下がる: 0.999 × 0.999 = 0.998
  並列に足すと上がる: 1 - (0.001 × 0.001) = 0.999999
```

---

## 3. SLA・SLO・SLI

信頼性の目標を定量的に管理するためのフレームワークとして、SLA・SLO・SLI がある。

### 3.1 三者の関係

```
  ┌─────────────────────────────────────────────────┐
  │  SLI (Service Level Indicator) — 実測値          │
  │  「今、実際にどうなっているか」                     │
  │  例: レイテンシP99 = 150ms、エラー率 = 0.02%      │
  │                                                  │
  │  ┌───────────────────────────────────────────┐   │
  │  │  SLO (Service Level Objective) — 内部目標  │   │
  │  │  「チームとして何を目指すか」                 │   │
  │  │  例: P99 < 200ms を99.9%の時間維持          │   │
  │  │                                            │   │
  │  │  ┌───────────────────────────────────┐     │   │
  │  │  │  SLA (Service Level Agreement)   │     │   │
  │  │  │  「顧客に何を約束するか」          │     │   │
  │  │  │  例: 可用性99.95%未達でクレジット  │     │   │
  │  │  └───────────────────────────────────┘     │   │
  │  └───────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────┘

  重要: SLO は SLA より厳しく設定する
  SLA: 99.95%（顧客との契約）
  SLO: 99.99%（内部目標、SLA違反の余裕を確保）

  エラーバジェット = SLO - 実績
  例: SLO 99.9% で今月のエラー率が 0.05% の場合
  → エラーバジェット残り = 0.1% - 0.05% = 0.05%
  → バジェット残りがあればリスクのある変更をデプロイ可能
```

### コード例3: SLI/SLO モニタリングの実装

```python
import time
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

@dataclass
class SLIMetric:
    """Service Level Indicator の計測"""
    name: str
    window_seconds: int = 3600  # 1時間のスライディングウィンドウ

    _measurements: deque = field(default_factory=deque, init=False)
    _good_count: int = field(default=0, init=False)
    _total_count: int = field(default=0, init=False)

    def record(self, is_good: bool, value: float = 0.0):
        """計測値を記録"""
        now = time.time()
        self._measurements.append((now, is_good, value))
        self._total_count += 1
        if is_good:
            self._good_count += 1
        self._evict_old(now)

    def _evict_old(self, now: float):
        """ウィンドウ外の古い計測値を削除"""
        while self._measurements and self._measurements[0][0] < now - self.window_seconds:
            _, was_good, _ = self._measurements.popleft()
            self._total_count -= 1
            if was_good:
                self._good_count -= 1

    @property
    def availability(self) -> float:
        """現在の可用性（成功率）"""
        if self._total_count == 0:
            return 1.0
        return self._good_count / self._total_count

    @property
    def percentile_latency(self) -> float:
        """P99 レイテンシ"""
        values = sorted(v for _, is_good, v in self._measurements if is_good)
        if not values:
            return 0.0
        idx = int(len(values) * 0.99)
        return values[min(idx, len(values) - 1)]


@dataclass
class SLOChecker:
    """SLO の達成状況を監視しエラーバジェットを管理"""
    sli: SLIMetric
    target_availability: float = 0.999   # 99.9%
    target_latency_p99: float = 200.0    # 200ms

    def check(self) -> dict:
        """SLO 達成状況を確認"""
        avail = self.sli.availability
        latency = self.sli.percentile_latency

        avail_ok = avail >= self.target_availability
        latency_ok = latency <= self.target_latency_p99

        # エラーバジェット計算
        error_budget_total = 1.0 - self.target_availability
        error_budget_used = 1.0 - avail
        error_budget_remaining = max(0, error_budget_total - error_budget_used)
        budget_percentage = (error_budget_remaining / error_budget_total * 100
                           if error_budget_total > 0 else 100)

        return {
            "availability": f"{avail*100:.4f}%",
            "availability_target": f"{self.target_availability*100:.2f}%",
            "availability_met": avail_ok,
            "latency_p99_ms": f"{latency:.1f}",
            "latency_target_ms": f"{self.target_latency_p99:.1f}",
            "latency_met": latency_ok,
            "error_budget_remaining": f"{budget_percentage:.1f}%",
            "slo_met": avail_ok and latency_ok,
        }


# 使用例
sli = SLIMetric("api-gateway", window_seconds=3600)

# リクエストを記録
for i in range(10000):
    is_success = i % 1000 != 0  # 0.1% のエラー率
    latency = 50 + (i % 100) * 2  # 50-248ms
    sli.record(is_success, latency)

checker = SLOChecker(sli, target_availability=0.999, target_latency_p99=200.0)
result = checker.check()
for k, v in result.items():
    print(f"  {k}: {v}")
# 出力例:
#   availability: 99.9000%
#   availability_target: 99.90%
#   availability_met: True
#   latency_p99_ms: 246.0
#   latency_target_ms: 200.0
#   latency_met: False
#   error_budget_remaining: 0.0%
#   slo_met: False
```

---

## 4. 冗長化パターン

### 4.1 Active-Passive と Active-Active

### ASCII図解3: Active-Passive vs Active-Active

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

  メリット: シンプル、データ整合性が保ちやすい
  デメリット: Passive のリソースが通常時は遊休
  用途: データベース、ステートフルサービス

■ Active-Active (負荷分散 + 冗長)

  Client ──→ ┌──────┐     ┌──────────┐
             │  LB  │────→│ Active 1 │
             │      │────→│ Active 2 │
             └──────┘     └──────────┘
                           ↕ (双方向同期)
                          どちらが落ちても即座にもう一方が処理

  メリット: リソース効率が良い、フェイルオーバーが高速
  デメリット: データ競合の管理が複雑
  用途: Webサーバー、ステートレスサービス

■ N+1 冗長化

  通常: Server 1, Server 2, Server 3 で負荷を分散
  +1:   Server 4 を追加（1台故障しても残り3台で処理可能）

  メリット: Active-Active より低コストでフォールトトレランス確保
  用途: Webサーバーファーム、ワーカープール
```

### 4.2 フェイルオーバー戦略

### ASCII図解4: フェイルオーバーの種類

```
■ コールドフェイルオーバー
  ┌─────────┐    障害     ┌─────────┐
  │ Primary │ ──×──→     │ Standby │  ← 起動に数分かかる
  │ (稼働)  │            │ (停止)  │     (コスト最小)
  └─────────┘            └─────────┘
  復旧時間: 数分〜数十分
  データ損失: バックアップ時点まで

■ ウォームフェイルオーバー
  ┌─────────┐    障害     ┌─────────┐
  │ Primary │ ──×──→     │ Standby │  ← 起動済み、データ同期に遅延
  │ (稼働)  │            │ (低速)  │     (中程度コスト)
  └─────────┘            └─────────┘
  復旧時間: 数十秒〜数分
  データ損失: レプリケーションラグ分

■ ホットフェイルオーバー
  ┌─────────┐    障害     ┌─────────┐
  │ Primary │ ──×──→     │ Standby │  ← 同一状態で同期稼働
  │ (稼働)  │←──同期──→  │ (稼働)  │     (コスト最大)
  └─────────┘            └─────────┘
  復旧時間: 数秒以内
  データ損失: ほぼゼロ
```

---

## 5. レジリエンスパターン

### 5.1 サーキットブレーカー

サーキットブレーカーは、障害が連鎖（カスケード障害）するのを防ぐパターンである。電気回路のブレーカーと同様に、異常時に回路を遮断してシステム全体を保護する。

```
サーキットブレーカーの状態遷移:

  ┌──────────┐   障害が閾値到達    ┌──────────┐
  │ CLOSED   │ ─────────────────→ │   OPEN   │
  │(正常通過) │                    │(即座に拒否)│
  └──────────┘                    └──────────┘
       ↑                               │
       │  成功が閾値到達          タイムアウト経過
       │                               │
  ┌──────────┐                         ▼
  │ HALF_OPEN│ ←───────────────────────
  │(一部試行) │   失敗 → OPEN に戻る
  └──────────┘
```

### コード例4: サーキットブレーカーパターン

```python
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"        # 正常（リクエスト通過）
    OPEN = "open"            # 遮断（リクエスト拒否）
    HALF_OPEN = "half_open"  # 試行（一部リクエスト通過）

class CircuitBreakerError(Exception):
    """サーキットブレーカーが開いている場合の例外"""
    pass

@dataclass
class CircuitBreaker:
    """
    サーキットブレーカー: 障害の連鎖を防ぐ

    内部実装の解説:
      CLOSED 状態で失敗が failure_threshold に達すると OPEN に遷移。
      OPEN 状態で recovery_timeout 秒が経過すると HALF_OPEN に遷移。
      HALF_OPEN で success_threshold 回成功すると CLOSED に復帰。
      HALF_OPEN で1回でも失敗すると再び OPEN に遷移。
    """
    failure_threshold: int = 5          # 障害回数の閾値
    recovery_timeout: float = 30.0      # OPEN 状態の維持時間（秒）
    success_threshold: int = 3          # HALF_OPEN→CLOSED に必要な成功回数
    half_open_max_calls: int = 3        # HALF_OPEN 中の最大同時試行数

    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    failure_count: int = field(default=0, init=False)
    success_count: int = field(default=0, init=False)
    last_failure_time: float = field(default=0, init=False)
    half_open_calls: int = field(default=0, init=False)

    # メトリクス
    total_calls: int = field(default=0, init=False)
    total_failures: int = field(default=0, init=False)
    total_rejections: int = field(default=0, init=False)

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """関数をサーキットブレーカー経由で実行"""
        self.total_calls += 1

        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self.half_open_calls = 0
                print(f"[Circuit] OPEN → HALF_OPEN: 試行開始")
            else:
                self.total_rejections += 1
                raise CircuitBreakerError(
                    f"Circuit OPEN: リクエスト拒否 "
                    f"(復旧まで {self.recovery_timeout - (time.time() - self.last_failure_time):.0f}秒)"
                )

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                self.total_rejections += 1
                raise CircuitBreakerError("Circuit HALF_OPEN: 試行上限に到達")
            self.half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except CircuitBreakerError:
            raise
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                print(f"[Circuit] HALF_OPEN → CLOSED: 正常復帰")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def _on_failure(self):
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            print(f"[Circuit] HALF_OPEN → OPEN: 試行失敗、再遮断")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            print(f"[Circuit] CLOSED → OPEN: {self.failure_count}回連続障害で遮断")

    def get_metrics(self) -> dict:
        return {
            "state": self.state.value,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_rejections": self.total_rejections,
            "failure_rate": (self.total_failures / max(1, self.total_calls) * 100),
        }


# 使用例
cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)

def unreliable_service():
    import random
    if random.random() < 0.7:
        raise ConnectionError("Service unavailable")
    return {"status": "ok"}

for i in range(10):
    try:
        result = cb.call(unreliable_service)
        print(f"  Call {i}: Success - {result}")
    except CircuitBreakerError as e:
        print(f"  Call {i}: Rejected - {e}")
    except ConnectionError as e:
        print(f"  Call {i}: Failed - {e}")

print(f"\nMetrics: {cb.get_metrics()}")
```

### 5.2 リトライ with 指数バックオフ

### コード例5: リトライ with 指数バックオフ + ジッター

```python
import random
import time
from functools import wraps
from typing import Tuple, Type

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    指数バックオフ + ジッターによるリトライデコレータ

    WHY 指数バックオフ:
      固定間隔でリトライすると、多数のクライアントが同時にリトライして
      障害中のサーバーにさらに負荷をかける（リトライストーム）。
      指数バックオフで間隔を広げ、ジッターでタイミングを分散させる。

    WHY ジッター:
      バックオフだけでは、同時に失敗したクライアント群が同じタイミングで
      リトライを繰り返す（同期現象）。ランダムなジッターで分散させる。

    ジッター戦略の比較:
      No Jitter:    delay = base * 2^attempt        (同期あり)
      Full Jitter:  delay = random(0, base * 2^attempt)  (最も分散)
      Equal Jitter: delay = base * 2^attempt / 2 + random(0, base * 2^attempt / 2)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    if attempt == max_retries:
                        print(f"[Retry] 最大リトライ回数({max_retries})に到達: {e}")
                        raise

                    delay = min(base_delay * (2 ** attempt), max_delay)
                    if jitter:
                        delay = random.uniform(0, delay)  # Full Jitter

                    print(f"[Retry] 試行{attempt+1}失敗, "
                          f"{delay:.2f}秒後にリトライ: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator


@retry_with_backoff(
    max_retries=4,
    base_delay=0.5,
    retryable_exceptions=(ConnectionError, TimeoutError),
)
def call_external_api(url: str) -> dict:
    """外部APIの呼び出し（障害時に自動リトライ）"""
    import requests
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.json()
```

### 5.3 バルクヘッドパターン

### コード例6: バルクヘッドパターン

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict

@dataclass
class BulkheadConfig:
    """バルクヘッドの設定"""
    max_concurrent: int       # 最大同時実行数
    max_queue_size: int = 0   # キュー待ちの最大数
    timeout: float = 30.0     # タイムアウト（秒）

class BulkheadPattern:
    """
    バルクヘッド: リソースを分離して障害の影響範囲を制限

    WHY:
      船舶の隔壁（Bulkhead）に由来。船が浸水しても隔壁で区切ることで
      全体の沈没を防ぐ。ソフトウェアでも同様に、あるサービスの障害が
      他のサービスのリソースを食い潰さないように分離する。

    例: payment サービスが遅延しても、notification のスレッドは影響を受けない
    """

    def __init__(self, configs: Dict[str, BulkheadConfig]):
        self.configs = configs
        self.pools = {
            name: ThreadPoolExecutor(
                max_workers=config.max_concurrent,
                thread_name_prefix=name
            )
            for name, config in configs.items()
        }
        self.semaphores = {
            name: asyncio.Semaphore(config.max_concurrent)
            for name, config in configs.items()
        }
        self.metrics = {name: {"calls": 0, "rejected": 0, "timeout": 0}
                       for name in configs}

    async def call_service(self, service_name: str, func, *args):
        """サービスをバルクヘッド経由で呼び出す"""
        if service_name not in self.configs:
            raise ValueError(f"Unknown service: {service_name}")

        config = self.configs[service_name]
        sem = self.semaphores[service_name]
        self.metrics[service_name]["calls"] += 1

        async with sem:
            try:
                pool = self.pools[service_name]
                loop = asyncio.get_event_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(pool, func, *args),
                    timeout=config.timeout,
                )
            except asyncio.TimeoutError:
                self.metrics[service_name]["timeout"] += 1
                raise TimeoutError(
                    f"Bulkhead: {service_name} がタイムアウト ({config.timeout}秒)"
                )


# 使用例
bulkhead = BulkheadPattern({
    "payment":       BulkheadConfig(max_concurrent=10, timeout=30.0),
    "notification":  BulkheadConfig(max_concurrent=5, timeout=10.0),
    "analytics":     BulkheadConfig(max_concurrent=3, timeout=5.0),
})
# notification が詰まっても payment には影響しない
```

### 5.4 ヘルスチェックによる障害検知

### コード例7: ヘルスチェック実装

```python
import asyncio
import aiohttp
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Callable, List
from enum import Enum

class HealthState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"      # 遅延が大きいが応答あり
    UNHEALTHY = "unhealthy"

@dataclass
class HealthStatus:
    url: str
    state: HealthState
    latency_ms: float
    checked_at: datetime
    error: Optional[str] = None

class HealthChecker:
    """
    多層ヘルスチェック: アクティブ + パッシブ + ディープ

    アクティブ: 定期的に /health エンドポイントをポーリング
    パッシブ:   実トラフィックのエラー率から判定
    ディープ:   DB接続、外部API接続等の依存関係まで確認
    """

    def __init__(
        self,
        targets: List[str],
        interval: float = 10.0,
        timeout: float = 3.0,
        unhealthy_threshold: int = 3,
        healthy_threshold: int = 2,
        degraded_latency_ms: float = 1000.0,
        on_state_change: Optional[Callable] = None,
    ):
        self.targets = targets
        self.interval = interval
        self.timeout = timeout
        self.unhealthy_threshold = unhealthy_threshold
        self.healthy_threshold = healthy_threshold
        self.degraded_latency_ms = degraded_latency_ms
        self.on_state_change = on_state_change

        self.failure_counts: dict[str, int] = {t: 0 for t in targets}
        self.success_counts: dict[str, int] = {t: 0 for t in targets}
        self.statuses: dict[str, HealthStatus] = {}

    async def check_one(self, session: aiohttp.ClientSession, url: str):
        """単一ターゲットのアクティブヘルスチェック"""
        start = asyncio.get_event_loop().time()
        try:
            async with session.get(
                f"{url}/health",
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                latency = (asyncio.get_event_loop().time() - start) * 1000
                if resp.status == 200:
                    if latency > self.degraded_latency_ms:
                        self._update_state(url, HealthState.DEGRADED, latency)
                    else:
                        self._mark_success(url, latency)
                else:
                    self._mark_failure(url, f"HTTP {resp.status}")
        except asyncio.TimeoutError:
            self._mark_failure(url, "Timeout")
        except Exception as e:
            self._mark_failure(url, str(e))

    def _mark_success(self, url: str, latency: float):
        self.failure_counts[url] = 0
        self.success_counts[url] = self.success_counts.get(url, 0) + 1
        old_state = self.statuses.get(url)
        if (old_state and old_state.state != HealthState.HEALTHY
            and self.success_counts[url] >= self.healthy_threshold):
            self._update_state(url, HealthState.HEALTHY, latency)
        elif not old_state or old_state.state == HealthState.HEALTHY:
            self._update_state(url, HealthState.HEALTHY, latency)

    def _mark_failure(self, url: str, error: str):
        self.success_counts[url] = 0
        self.failure_counts[url] += 1
        if self.failure_counts[url] >= self.unhealthy_threshold:
            self._update_state(url, HealthState.UNHEALTHY, 0, error)
            print(f"[ALERT] {url} が {self.failure_counts[url]}回連続失敗 → UNHEALTHY")

    def _update_state(self, url: str, state: HealthState,
                      latency: float, error: str = None):
        old = self.statuses.get(url)
        self.statuses[url] = HealthStatus(
            url=url, state=state, latency_ms=latency,
            checked_at=datetime.now(), error=error
        )
        if old and old.state != state and self.on_state_change:
            self.on_state_change(url, old.state, state)

    def get_healthy_targets(self) -> List[str]:
        return [url for url, s in self.statuses.items()
                if s.state == HealthState.HEALTHY]
```

### 5.5 グレースフルデグラデーション

### コード例8: グレースフルデグラデーションの実装

```python
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

class DegradationLevel(Enum):
    FULL = "full"               # 全機能稼働
    DEGRADED = "degraded"       # 一部機能を制限
    MINIMAL = "minimal"         # 最低限の機能のみ
    MAINTENANCE = "maintenance" # メンテナンスモード

@dataclass
class FeatureFlag:
    name: str
    enabled: bool
    min_level: DegradationLevel
    fallback_value: Any = None

class GracefulDegradation:
    """
    グレースフルデグラデーション: 障害時にサービスを段階的に縮退

    WHY:
      全機能を完璧に提供できない状況でも、コア機能は維持して
      ユーザーに最低限のサービスを提供する。
      「何もできない」よりも「一部でも使える」方がはるかに良い。

    例: ECサイトで推薦エンジンが障害 → 人気商品一覧を表示
        決済が障害 → カートに入れるまでは可能にする
    """

    def __init__(self):
        self.level = DegradationLevel.FULL
        self.features: Dict[str, FeatureFlag] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register("recommendation", DegradationLevel.FULL,
                      fallback={"items": [], "source": "fallback"})
        self.register("real_time_search", DegradationLevel.FULL, fallback=None)
        self.register("order_creation", DegradationLevel.DEGRADED, fallback=None)
        self.register("product_listing", DegradationLevel.MINIMAL, fallback=None)
        self.register("static_pages", DegradationLevel.MAINTENANCE, fallback=None)

    def register(self, name: str, min_level: DegradationLevel, fallback: Any = None):
        self.features[name] = FeatureFlag(
            name=name, enabled=True,
            min_level=min_level, fallback_value=fallback
        )

    def set_level(self, level: DegradationLevel):
        old = self.level
        self.level = level
        levels = list(DegradationLevel)
        current_idx = levels.index(level)

        for feature in self.features.values():
            feature_idx = levels.index(feature.min_level)
            feature.enabled = feature_idx >= current_idx

        print(f"[DEGRADATION] {old.value} → {level.value}")
        enabled = [f.name for f in self.features.values() if f.enabled]
        disabled = [f.name for f in self.features.values() if not f.enabled]
        print(f"  有効: {enabled}")
        print(f"  無効: {disabled}")

    def is_enabled(self, feature_name: str) -> bool:
        feature = self.features.get(feature_name)
        return feature.enabled if feature else False

    def get_fallback(self, feature_name: str) -> Any:
        feature = self.features.get(feature_name)
        return feature.fallback_value if feature else None


# 使用例
gd = GracefulDegradation()
gd.set_level(DegradationLevel.FULL)       # 全機能ON
gd.set_level(DegradationLevel.DEGRADED)   # 推薦・検索OFF、注文はOK
gd.set_level(DegradationLevel.MINIMAL)    # 商品一覧と静的ページのみ
```

---

## 6. カオスエンジニアリング

### 6.1 原則と手順

カオスエンジニアリングは、本番環境で意図的に障害を注入し、システムの信頼性を検証する手法である。

```
カオスエンジニアリングの4ステップ:

  ┌─────────────────────────────────────────────────────┐
  │ Step 1: 定常状態（Steady State）を定義              │
  │   例: P99レイテンシ < 200ms、エラー率 < 0.1%        │
  └──────────────┬──────────────────────────────────────┘
                 ▼
  ┌─────────────────────────────────────────────────────┐
  │ Step 2: 仮説を設定                                  │
  │   例: 「App Server 1台停止でも P99 < 300ms を維持」  │
  └──────────────┬──────────────────────────────────────┘
                 ▼
  ┌─────────────────────────────────────────────────────┐
  │ Step 3: 障害を注入（Blast Radius を制限）           │
  │   例: 1台のインスタンスを停止、CPU負荷を注入        │
  └──────────────┬──────────────────────────────────────┘
                 ▼
  ┌─────────────────────────────────────────────────────┐
  │ Step 4: 結果を観測し、改善策を実施                   │
  │   例: フェイルオーバーに45秒 → ヘルスチェック間隔短縮 │
  └─────────────────────────────────────────────────────┘
```

### コード例9: カオスエンジニアリングフレームワーク

```python
import asyncio
from dataclasses import dataclass, field
from typing import List, Callable, Optional
from enum import Enum

class FaultType(Enum):
    LATENCY = "latency"
    ERROR = "error"
    RESOURCE = "resource"
    NETWORK_PARTITION = "partition"

@dataclass
class ChaosExperiment:
    """カオス実験の定義"""
    name: str
    fault_type: FaultType
    target: str
    duration_seconds: float
    blast_radius: float = 0.1    # 影響範囲（0.0-1.0）
    latency_ms: float = 0
    error_rate: float = 0
    cpu_load: float = 0

@dataclass
class ExperimentResult:
    experiment: ChaosExperiment
    hypothesis_met: bool
    observations: List[str] = field(default_factory=list)
    metrics_before: dict = field(default_factory=dict)
    metrics_after: dict = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

class ChaosEngine:
    """カオスエンジニアリングの実行エンジン"""

    def __init__(self):
        self.experiments: List[ChaosExperiment] = []
        self.results: List[ExperimentResult] = []
        self.safety_checks: List[Callable[[], bool]] = []

    def add_safety_check(self, check: Callable[[], bool]):
        """安全チェックを追加（False を返したら即座に停止）"""
        self.safety_checks.append(check)

    def _check_safety(self) -> bool:
        return all(check() for check in self.safety_checks)

    async def run_experiment(
        self,
        experiment: ChaosExperiment,
        hypothesis: Callable[[], bool],
        collect_metrics: Callable[[], dict],
    ) -> ExperimentResult:
        print(f"\n{'='*60}")
        print(f"[CHAOS] 実験開始: {experiment.name}")
        print(f"  対象: {experiment.target}")
        print(f"  障害タイプ: {experiment.fault_type.value}")
        print(f"  影響範囲: {experiment.blast_radius*100:.0f}%")
        print(f"{'='*60}")

        metrics_before = collect_metrics()

        if not self._check_safety():
            print("[CHAOS] 安全チェック失敗: 実験を中止")
            return ExperimentResult(
                experiment=experiment, hypothesis_met=False,
                observations=["安全チェック失敗で中止"],
                metrics_before=metrics_before,
            )

        print(f"[CHAOS] 障害注入中... ({experiment.duration_seconds}秒間)")
        await asyncio.sleep(experiment.duration_seconds)

        metrics_after = collect_metrics()
        hypothesis_met = hypothesis()

        result = ExperimentResult(
            experiment=experiment, hypothesis_met=hypothesis_met,
            metrics_before=metrics_before, metrics_after=metrics_after,
        )

        if hypothesis_met:
            print(f"[CHAOS] 仮説達成: システムは耐障害性を維持")
        else:
            print(f"[CHAOS] 仮説未達成: 改善が必要")
            result.recommendations.append(
                f"{experiment.target} の {experiment.fault_type.value} 耐性を強化する必要がある"
            )

        self.results.append(result)
        return result


# 使用例
engine = ChaosEngine()
engine.add_safety_check(lambda: True)  # 本番ではメトリクスAPIを確認

experiment = ChaosExperiment(
    name="App Server レイテンシ耐性テスト",
    fault_type=FaultType.LATENCY,
    target="app-server-1",
    duration_seconds=300,
    blast_radius=0.1,
    latency_ms=500,
)
```

---

## 7. 比較表

### 比較表1: フェイルオーバー戦略の比較

| 項目 | コールド | ウォーム | ホット |
|------|---------|---------|--------|
| 復旧時間 (RTO) | 数分〜数十分 | 数十秒〜数分 | 数秒以内 |
| コスト | 低い（Standby停止） | 中程度（Standby低速稼働） | 高い（Standby全力稼働） |
| データ損失 (RPO) | 大きい（最終バックアップまで） | 中程度（レプリケーションラグ分） | ほぼゼロ（同期レプリケーション） |
| 運用複雑さ | 低い | 中程度 | 高い |
| 適するシステム | 開発環境、バッチ、社内ツール | 一般Web、ECサイト | 決済、医療、金融 |
| AWS サービス例 | S3 + EC2 AMI | RDS Multi-AZ (非同期) | Aurora Global DB (同期) |

### 比較表2: レジリエンスパターンの比較

| パターン | 目的 | 実装複雑度 | 効果 | 適用箇所 |
|---------|------|-----------|------|---------|
| サーキットブレーカー | 障害の連鎖防止 | 中 | 高（カスケード障害防止） | サービス間通信 |
| リトライ+バックオフ | 一時的障害の回復 | 低 | 中（transient fault対応） | API呼び出し |
| バルクヘッド | 障害の分離 | 中 | 高（影響範囲限定） | スレッドプール、接続プール |
| ヘルスチェック | 障害の検知 | 低 | 高（早期検知） | LB → Backend |
| 冗長化 | 単一障害点の排除 | 高 | 非常に高い | 全レイヤー |
| グレースフルデグラデーション | 部分的サービス継続 | 中 | 高（UX維持） | フロントエンド連携 |
| タイムアウト | 無限待ちの防止 | 低 | 中（リソース解放） | 全外部通信 |
| キャッシュフォールバック | 障害時のデータ提供 | 中 | 中（stale data許容時） | 読み取りパス |

### 比較表3: カオスエンジニアリングツールの比較

| ツール | 提供元 | 対象 | 特徴 |
|--------|--------|------|------|
| Chaos Monkey | Netflix | EC2インスタンス | ランダムにインスタンスを停止 |
| Litmus | CNCF | Kubernetes | K8s ネイティブ、CRDベース |
| Gremlin | Gremlin社 | マルチプラットフォーム | SaaS、GUI操作、安全機能充実 |
| AWS FIS | AWS | AWSリソース | AWSサービスとの統合 |
| Chaos Mesh | PingCAP | Kubernetes | K8s向け、ネットワーク障害が得意 |
| Toxiproxy | Shopify | TCP接続 | ネットワーク障害シミュレーション |

---

## 8. アンチパターン

### アンチパターン1: リトライストーム

```python
# NG: 全クライアントが即座にリトライ
def bad_retry(func, max_retries=5):
    for i in range(max_retries):
        try:
            return func()
        except Exception:
            time.sleep(1)  # 固定間隔1秒 → 全クライアントが同時にリトライ
    raise Exception("Max retries exceeded")

# 問題の図解:
# Client 1 ──retry(1s)──retry(1s)──retry(1s)──→
# Client 2 ──retry(1s)──retry(1s)──retry(1s)──→  ← 全員同時にサーバーに殺到
# Client 3 ──retry(1s)──retry(1s)──retry(1s)──→
# サーバー: 障害中にさらに負荷が3倍 → 復旧不能


# OK: 指数バックオフ + ジッター
import random

def good_retry(func, max_retries=5, base_delay=1.0):
    for i in range(max_retries):
        try:
            return func()
        except Exception:
            delay = base_delay * (2 ** i)          # 指数的に間隔を広げる
            delay = random.uniform(0, delay)        # ジッターで分散
            delay = min(delay, 60.0)                # 最大60秒にキャップ
            time.sleep(delay)
    raise Exception("Max retries exceeded")

# Client 1 ──retry(0.7s)────────retry(2.3s)──────────→
# Client 2 ──────retry(0.3s)──────────retry(3.1s)───→  ← リトライが分散
# Client 3 ────────retry(0.9s)────────retry(1.8s)───→
# サーバー: 負荷が時間分散 → 復旧可能
```

### アンチパターン2: 単一障害点（SPOF）の見落とし

```python
# NG: 隠れたSPOFがある構成
class BadArchitecture:
    def __init__(self):
        self.web_servers = ["web-1", "web-2", "web-3"]  # 冗長化 OK
        self.load_balancer = "lb-1"    # SPOF! LBが1台のみ
        self.config_server = "config-1" # SPOF! 設定サーバーが1台
        self.dns_server = "dns-1"       # SPOF! DNSが1台

# OK: 全レイヤーで冗長化
class GoodArchitecture:
    def __init__(self):
        self.load_balancers = ["lb-active", "lb-standby"]  # VRRP/keepalived
        self.web_servers = ["web-1", "web-2", "web-3", "web-4"]  # N+1構成
        self.db_primary = "db-primary"
        self.db_replicas = ["db-replica-1", "db-replica-2"]
        self.config_servers = ["etcd-1", "etcd-2", "etcd-3"]  # 分散型
        self.dns_providers = ["route53", "cloudflare"]  # 複数プロバイダ

# SPOF チェックリスト:
# □ ロードバランサーは冗長化されているか？
# □ DNS は複数プロバイダか？
# □ 認証サービスは冗長化されているか？
# □ 設定管理は分散型か？
# □ 外部API依存にフォールバックはあるか？
```

### アンチパターン3: 障害テストをしない

```
NG:
「冗長化したから大丈夫」と思い込み、一度もフェイルオーバーテストをしない

よくある失敗パターン:
1. フェイルオーバースクリプトにバグがあり、切り替わらない
2. Standby のデータが古く、切り替え後にデータ不整合
3. DNS の TTL が長すぎて、切り替えに30分かかる
4. 監視アラートの送信先が不正で、障害に気づかない

OK:
- 定期的なフェイルオーバー演習（月次/四半期）
- カオスエンジニアリングの継続的実施
- Game Day（本番環境での障害シミュレーション日）
- フェイルオーバー後の自動検証スクリプト
- アラート到達テスト（PagerDuty等の定期テスト）
```

---

## 9. 実践演習

### 演習1（基礎）: 可用性計算

以下のシステム構成の可用性を計算せよ。

```
構成:
  Internet → [DNS: 99.99%] → [CDN: 99.95%] → [LB: 99.99%]
           → [App Server x2 (各99.9%、並列)] → [DB Primary: 99.99%]

問題:
1. App Server が並列構成の場合、App層の可用性を求めよ
2. システム全体の可用性（直列部分の積）を求めよ
3. 年間ダウンタイムを分単位で求めよ
4. 可用性を 99.99% 以上にするには、どのコンポーネントを改善すべきか？
```

**期待される出力:**

```
1. App層（並列）: 1 - (1-0.999)^2 = 1 - 0.000001 = 0.999999 (99.9999%)
2. 全体: 0.9999 × 0.9995 × 0.9999 × 0.999999 × 0.9999
       = 0.9992 (99.92%)
3. 年間ダウンタイム: 525960 × (1 - 0.9992) ≒ 421分 ≒ 7時間
4. ボトルネックは CDN (99.95%)
   → CDNの冗長化またはマルチCDN構成で改善
   → 次にLBとDBの冗長化
```

### 演習2（応用）: サーキットブレーカーの状態遷移テスト

上記のコード例4の `CircuitBreaker` クラスを使い、以下のテストケースを実装せよ。

```python
"""
テストケース:
1. 正常時: 10回連続成功 → CLOSED のまま
2. 障害発生: 5回連続失敗 → OPEN に遷移
3. 復旧試行: 30秒後に HALF_OPEN に遷移、3回成功で CLOSED に復帰
4. 復旧失敗: HALF_OPEN 中に1回失敗 → 再び OPEN に遷移
5. メトリクス: total_calls, total_failures, total_rejections が正しいこと
"""

def test_circuit_breaker():
    import time

    # Test 1: 正常時
    cb = CircuitBreaker(failure_threshold=5, recovery_timeout=1.0, success_threshold=3)
    for i in range(10):
        cb.call(lambda: "ok")
    assert cb.state == CircuitState.CLOSED, "Test 1 failed"
    print("Test 1: 10 successful calls → state=CLOSED  OK")

    # Test 2: 障害発生
    cb2 = CircuitBreaker(failure_threshold=5, recovery_timeout=1.0)
    for i in range(5):
        try:
            cb2.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        except Exception:
            pass
    assert cb2.state == CircuitState.OPEN, "Test 2 failed"
    print("Test 2: 5 failed calls → state=OPEN  OK")

    # Test 3: 復旧試行（成功）
    time.sleep(1.1)  # recovery_timeout 待ち
    for i in range(3):
        cb2.call(lambda: "ok")
    assert cb2.state == CircuitState.CLOSED, "Test 3 failed"
    print("Test 3: After timeout, 3 successes → state=CLOSED  OK")

    print("\nAll tests passed!")

test_circuit_breaker()
```

**期待される出力:**

```
Test 1: 10 successful calls → state=CLOSED  OK
[Circuit] CLOSED → OPEN: 5回連続障害で遮断
Test 2: 5 failed calls → state=OPEN  OK
[Circuit] OPEN → HALF_OPEN: 試行開始
[Circuit] HALF_OPEN → CLOSED: 正常復帰
Test 3: After timeout, 3 successes → state=CLOSED  OK

All tests passed!
```

### 演習3（発展）: マイクロサービスの信頼性設計

以下のECサイトの構成に対して、信頼性を最大化する設計を行え。

```
要件:
- SLA: 99.95%（月間ダウンタイム 21.9分以内）
- RPO: 1分以内（最大1分のデータ損失を許容）
- RTO: 5分以内（5分以内にサービス復旧）
- ピークトラフィック: 10,000 RPS

サービス構成:
1. API Gateway
2. User Service
3. Order Service
4. Payment Service（外部決済APIに依存）
5. Notification Service（メール/SMS送信）
6. PostgreSQL（ユーザー・注文データ）
7. Redis（セッション・キャッシュ）

設計課題:
1. 各サービスにどのレジリエンスパターンを適用するか表にまとめよ
2. Payment Service の外部API障害時のフォールバック戦略を設計せよ
3. PostgreSQL の RPO 1分・RTO 5分を満たす構成を設計せよ
4. 全体の SLA 99.95% を達成するための各コンポーネントの必要可用性を計算せよ
5. カオスエンジニアリングの実験計画を3つ立案せよ
```

**期待される出力（概要）:**

```
1. レジリエンスパターン適用表:
   | サービス        | CB | リトライ | バルクヘッド | ヘルスチェック | デグラデーション |
   |----------------|:--:|:------:|:---------:|:----------:|:------------:|
   | API Gateway    | o  | -      | o        | o         | o           |
   | User Service   | o  | o     | -        | o         | -           |
   | Order Service  | o  | o     | o        | o         | -           |
   | Payment        | o  | o     | o        | o         | o           |
   | Notification   | -  | o     | -        | o         | o           |

2. Payment フォールバック:
   - サーキットブレーカー (threshold=3, timeout=30s)
   - 障害時: 注文を「決済保留」ステータスで保存
   - 復旧後: 保留注文をバッチ処理で決済
   - DLQ で未処理の決済を管理

3. PostgreSQL 構成:
   - Primary + Synchronous Replica + Async Replica
   - WALアーカイブ: 1分間隔でS3にアップロード (RPO <= 1分)
   - 自動フェイルオーバー: Patroni + etcd (RTO <= 30秒)

4. 可用性計算:
   7コンポーネント直列で 99.95% を達成するには
   各コンポーネント >= 99.993% が必要
   → 各サービスを Active-Active 冗長化で達成

5. カオス実験:
   (1) Payment外部API 500ms遅延注入 → CB動作検証
   (2) DB Primary強制停止 → フェイルオーバー時間計測 (<5分)
   (3) Redis全ノード停止 → キャッシュなし状態のDB負荷検証
```

---

## 10. FAQ

### Q1: SLA、SLO、SLI の違いは何ですか？

**SLI**（Service Level Indicator）は実測値（例: レイテンシP99 = 150ms、エラー率 = 0.02%）。**SLO**（Service Level Objective）はチーム内部の目標値（例: P99 < 200ms を99.9%の時間維持）。**SLA**（Service Level Agreement）は顧客との契約（例: 可用性99.95%を下回った場合クレジット返金）。SLA違反は金銭的ペナルティを伴うため、SLOはSLAより厳しく設定するのが通例である。また、SLI を継続的に計測し、SLO との差分を「エラーバジェット」として管理することで、信頼性と開発速度のバランスを取る。

### Q2: カオスエンジニアリングとは何ですか？実施する上での注意点は？

カオスエンジニアリングは、本番環境で意図的に障害を注入し、システムの信頼性を検証する手法である。Netflixの「Chaos Monkey」が有名で、ランダムにインスタンスを停止させることでフォールトトレランスをテストする。実施の手順は (1) 定常状態の定義、(2) 仮説の設定、(3) 障害の注入、(4) 結果の観測と改善、である。

注意点として、Blast Radius を制限すること（最初は1%のトラフィックから開始）、安全停止条件を事前に設けること（エラー率が閾値を超えたら自動停止）、営業時間内に実施すること（問題発生時に即座に対応）、全チームに事前通知すること（意図的障害と本物の障害を混同しない）が挙げられる。

### Q3: RPOとRTOの違いは何ですか？

**RPO**（Recovery Point Objective）は「どの時点のデータまで復旧できるか」を示す。例えば RPO = 1時間 なら、障害発生時に最大1時間分のデータが失われる可能性がある。**RTO**（Recovery Time Objective）は「障害発生からサービス復旧までの時間」を示す。例えば RTO = 15分 なら、15分以内にサービスを再開しなければならない。RPOはバックアップ頻度やレプリケーション方式で制御し、RTOはフェイルオーバー方式で制御する。

### Q4: エラーバジェットとは何ですか？

エラーバジェットは「SLO 目標の余裕分」を定量的に表したものである。例えば SLO が 99.9% なら、月に 0.1% の時間（約43分）はダウンしても許容される。この43分がエラーバジェットである。バジェットが余っている時はリスクのある新機能をリリースでき、不足している時はリリースを凍結して信頼性改善に集中する。このアプローチにより「信頼性 vs 開発速度」のトレードオフを定量的に管理できる。

### Q5: 分散システムで「正確に1回」の処理を実現できますか？

厳密な意味での「正確に1回（Exactly-once）」は分散システムでは実現が非常に難しい。ネットワーク障害やプロセスクラッシュにより、メッセージの重複配信は避けられない。実践的なアプローチは「少なくとも1回（At-least-once）」の配信 + 「べき等処理（Idempotent Processing）」の組み合わせである。具体的にはリクエストに一意な ID を付与し、処理済み ID を記録して重複を検知する。

### Q6: マルチリージョン構成での信頼性設計のポイントは？

マルチリージョン構成では、(1) データの同期方式（同期レプリケーションはレイテンシ増大、非同期はデータ損失リスク）、(2) コンフリクト解決（Active-Active の場合、同一データの同時書き込みをどう解決するか）、(3) フェイルオーバーのトリガー（リージョン全体の障害判定は難しい）、(4) DNS 切り替え時間（TTL の設定）、が主要な検討事項である。AWS では Route 53 の health check + failover routing、Azure では Traffic Manager を使うのが一般的である。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 信頼性の定義 | 障害時にも正しく機能し続ける能力。Fault → Failure を防ぐ |
| 可用性の計算 | 直列=積（下がる）、並列=1-(不可用性の積)（上がる） |
| SLA/SLO/SLI | SLI（実測）→ SLO（内部目標）→ SLA（顧客契約）の3層管理 |
| 冗長化 | Active-Passive / Active-Active / N+1 で SPOF を排除 |
| サーキットブレーカー | CLOSED→OPEN→HALF_OPEN でカスケード障害を防止 |
| リトライ戦略 | 指数バックオフ + Full Jitter で負荷を分散 |
| バルクヘッド | リソースを分離して障害の影響範囲を限定 |
| グレースフルデグラデーション | 全停止ではなく段階的にサービスを縮退 |
| カオスエンジニアリング | 意図的な障害注入で信頼性を継続的に検証 |
| エラーバジェット | 信頼性と開発速度のバランスを定量的に管理 |

---

## 次に読むべきガイド

- [CAP定理](./03-cap-theorem.md) -- 分散システムにおける一貫性・可用性・分断耐性のトレードオフ
- [ロードバランサー](../01-components/00-load-balancer.md) -- 冗長化されたトラフィック分散の設計
- [メッセージキュー](../01-components/02-message-queue.md) -- 非同期処理による信頼性向上とバックプレッシャー
- [モノリス vs マイクロサービス](../02-architecture/00-monolith-vs-microservices.md) -- サービス分割と障害分離の設計
- [デザインパターン](../../../03-software-design/design-patterns-guide/docs/02-behavioral/) -- オブザーバー等のイベント駆動パターン

---

## 参考文献

1. Nygard, M.T. (2018). *Release It!: Design and Deploy Production-Ready Software*, 2nd Edition. Pragmatic Bookshelf. -- サーキットブレーカー、バルクヘッド等のレジリエンスパターンの原典
2. Rosenthal, C. & Jones, N. (2020). *Chaos Engineering: System Resiliency in Practice*. O'Reilly Media. -- カオスエンジニアリングの体系的な解説
3. Beyer, B. et al. (2016). *Site Reliability Engineering*. O'Reilly Media. https://sre.google/sre-book/ -- Google の SRE プラクティス（SLI/SLO/SLA、エラーバジェット）
4. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Chapter 8: The Trouble with Distributed Systems. O'Reilly Media. -- 分散システムの障害モデルと信頼性設計
5. Burns, B. (2018). *Designing Distributed Systems*. O'Reilly Media. -- 分散システムのパターンカタログ

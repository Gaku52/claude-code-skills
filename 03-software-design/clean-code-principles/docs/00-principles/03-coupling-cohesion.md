# 結合度と凝集度 ── モジュール設計の基盤原則

> 優れたモジュール設計は「低結合・高凝集」に集約される。結合度はモジュール間の依存関係の強さ、凝集度はモジュール内の要素の関連性の強さを示す。この2つの指標を意識することで、変更に強く理解しやすいシステムが構築できる。

---

## この章で学ぶこと

1. **結合度の7段階** ── 内容結合からデータ結合まで、依存関係の種類と危険度を理解する
2. **凝集度の7段階** ── 偶発的凝集から機能的凝集まで、モジュール内のまとまり方を理解する
3. **低結合・高凝集を達成する設計技法** ── 具体的なリファクタリング手法を身につける
4. **結合度・凝集度の定量的測定方法** ── 静的解析ツールによるメトリクス計測を習得する
5. **アーキテクチャレベルでの適用** ── マイクロサービス、モジュラーモノリスにおける適用方法を理解する

---

## 前提知識

このガイドを最大限に活用するには、以下の知識が必要です。

| 前提知識 | 必要レベル | 参照先 |
|---------|----------|--------|
| SOLID原則（特にSRP, DIP） | 基本を理解 | [SOLID原則](./01-solid.md) |
| クリーンコードの概要 | 読了推奨 | [クリーンコード概論](./00-clean-code-overview.md) |
| DRY/KISS/YAGNI | 基本を理解 | [DRY/KISS/YAGNI](./02-dry-kiss-yagni.md) |
| オブジェクト指向の基本 | 実務経験あり | -- |
| デザインパターンの基礎 | 概要を把握 | [デザインパターン概論](../../design-patterns-guide/docs/00-creational/00-overview.md) |

---

## 1. 結合度（Coupling）── モジュール間の依存の強さ

### 1.1 なぜ結合度を理解すべきか

結合度（Coupling）の概念は1974年にLarry ConstantineとEdward Yourdonによって提唱された。彼らの研究は「ソフトウェアの保守コストの50-80%は、変更の波及効果の管理に費やされている」という実証データに基づいている。

```
  変更コストのモデル（Constantine-Yourdon, 1979）

  変更コスト = 直接コスト + 波及コスト + テストコスト

  ┌─────────────────────────────────────────────────────┐
  │  高結合のシステム                                      │
  │                                                       │
  │  直接コスト : ████ (20%)                              │
  │  波及コスト : ████████████████████ (55%)              │
  │  テストコスト: █████████ (25%)                         │
  │                                                       │
  │  → 実際の変更は全体の2割。残りは波及と検証            │
  └─────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │  低結合のシステム                                      │
  │                                                       │
  │  直接コスト : ████████████ (50%)                      │
  │  波及コスト : ████ (15%)                              │
  │  テストコスト: ████████ (35%)                          │
  │                                                       │
  │  → 変更の影響範囲が局所的で、テストも限定的           │
  └─────────────────────────────────────────────────────┘
```

結合度を理解する本質的な理由は以下の3点である。

1. **変更の局所化**: 低結合なシステムでは、1箇所の変更が他に波及しにくい
2. **テスト容易性**: モジュールを単独でテストできる（モック不要、または最小限）
3. **チーム並行開発**: 独立したモジュールなら、チームが並行して開発可能

実際のプロジェクトでの影響を定量化すると以下のようになる。

| 結合度レベル | 1変更あたりの影響ファイル数 | リグレッションバグ率 | ビルド時間（差分） |
|------------|------------------------|-------------------|-----------------|
| 高結合 | 10-50ファイル | 15-30% | 全ビルド必須 |
| 中結合 | 3-10ファイル | 5-15% | 部分ビルド可能 |
| 低結合 | 1-3ファイル | 1-5% | モジュール単位 |

### 1.2 結合度の7段階

Constantine-Yourdonの分類に基づく結合度の7段階を、危険度の高い順に解説する。

```
  危険度: 高 ←────────────────────────────────────────────→ 低

  ┌───────┬────────┬────────┬────────┬────────┬────────┬───────┐
  │ 内容   │ 共通    │ 外部    │ 制御    │ スタンプ│ データ  │ メッセージ│
  │ 結合   │ 結合    │ 結合    │ 結合    │ 結合    │ 結合    │ 結合    │
  │Content│Common  │External│Control │Stamp   │Data    │Message│
  │       │        │        │        │        │        │       │
  │他の内部│グローバル│外部の   │フラグで │データ   │必要な  │メッセージ│
  │を直接 │変数を  │フォーマ │動作を  │構造体を│プリミティ│のみで  │
  │参照   │共有    │ットを  │切替    │丸ごと  │ブ値を  │通信    │
  │       │        │共有    │        │渡す    │渡す    │        │
  └───────┴────────┴────────┴────────┴────────┴────────┴───────┘
   絶対避ける 避ける   最小化   最小限に  許容     目指す   理想
```

**各段階の詳細定義:**

| 段階 | 名称 | 定義 | 具体例 | 危険度 |
|------|------|------|--------|--------|
| 1 | 内容結合 (Content) | 他モジュールの内部実装（private変数、内部コード）に直接アクセス | `obj._private_field` にアクセス | 最高 |
| 2 | 共通結合 (Common) | 複数モジュールがグローバル変数/共有状態を読み書き | グローバル設定オブジェクトの共有 | 高 |
| 3 | 外部結合 (External) | 外部のデータフォーマット、通信プロトコル、デバイスインターフェースを共有 | 共通のCSVフォーマット、共有DBスキーマ | 中-高 |
| 4 | 制御結合 (Control) | フラグや制御値で相手の動作を切り替える | `process(data, is_pdf=True)` | 中 |
| 5 | スタンプ結合 (Stamp) | 必要以上のデータを含むデータ構造を丸ごと渡す | 関数が `User` オブジェクト全体を受け取るが `name` のみ使用 | 低-中 |
| 6 | データ結合 (Data) | 必要最小限のプリミティブ値のみ受け渡し | `calculate(subtotal, tax_rate)` | 低 |
| 7 | メッセージ結合 (Message) | メッセージ（イベント）のみで通信し、相手の存在を知らない | EventBus経由のイベント通知 | 最低 |

**コード例1: 結合度の7段階別コード**

```python
# === 1. 内容結合（最悪）: 他モジュールの内部実装に依存 ===
class OrderProcessor:
    def process(self, cart):
        # Cart の private 実装を直接操作
        cart._items[0]._price = cart._items[0]._price * 0.9
        cart._total_cache = None  # キャッシュを手動リセット
        # → Cart の内部実装が変わると即座に壊れる

# === 2. 共通結合（悪い）: グローバル変数を共有 ===
GLOBAL_CONFIG = {}

class ServiceA:
    def do_work(self):
        GLOBAL_CONFIG['last_run'] = datetime.now()
        GLOBAL_CONFIG['status'] = 'running'

class ServiceB:
    def do_work(self):
        # ServiceA の副作用に依存
        if GLOBAL_CONFIG.get('last_run'):
            pass
        # → ServiceA の実装変更が ServiceB に影響

# === 3. 外部結合（注意）: 外部フォーマットを共有 ===
class CsvExporter:
    def export(self, data):
        # 共通CSVフォーマット: "id,name,price\n" に依存
        return ",".join([str(data['id']), data['name'], str(data['price'])])

class CsvImporter:
    def import_data(self, csv_line):
        # 同じCSVフォーマットに依存
        parts = csv_line.split(",")
        return {'id': int(parts[0]), 'name': parts[1], 'price': float(parts[2])}
    # → フォーマット変更時に両方修正が必要

# === 4. 制御結合（要注意）: フラグで動作を切り替え ===
class ReportGenerator:
    def generate(self, data, format_type: str):
        if format_type == 'pdf':
            return self._generate_pdf(data)
        elif format_type == 'csv':
            return self._generate_csv(data)
        elif format_type == 'excel':
            return self._generate_excel(data)
    # → 呼び出し側が内部の分岐ロジックを知っている

# === 5. スタンプ結合（許容）: データ構造を丸ごと渡す ===
class EmailSender:
    def send_welcome(self, user: User):
        # User オブジェクト全体を受け取るが name と email のみ使用
        send_email(to=user.email, subject=f"Welcome {user.name}")
    # → User の構造変更が影響する可能性

# === 6. データ結合（理想）: 必要なプリミティブ値のみ受け渡し ===
class TaxCalculator:
    def calculate(self, subtotal: float, tax_rate: float) -> float:
        return subtotal * tax_rate
    # → 引数はプリミティブ値のみ。他の型に依存しない

# === 7. メッセージ結合（最理想）: メッセージのみで通信 ===
class OrderService:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

    def place_order(self, order):
        order.confirm()
        self.event_bus.publish('order_placed', {'order_id': order.id})
    # → 受信者の存在すら知らない
```

### 1.3 結合度を下げるテクニック

```
  直接依存                     間接依存（抽象を介する）

  ┌───────┐                   ┌───────┐
  │ ModuleA │                  │ ModuleA │
  └───┬───┘                   └───┬───┘
      │ import & new               │ 抽象に依存
      v                           v
  ┌───────┐               ┌─────────────┐
  │ ModuleB │               │ <<interface>>│
  └───────┘               │  IModuleB    │
                            └──────┬──────┘
                                   │ 実装
                                   v
                            ┌───────────┐
                            │ ModuleBImpl│
                            └───────────┘
```

**テクニック一覧:**

| テクニック | 効果 | 適用場面 | コスト |
|-----------|------|---------|--------|
| 依存性注入 (DI) | 具象クラスへの依存を排除 | サービス間の依存 | 低 |
| インターフェース抽出 | 実装の詳細を隠蔽 | モジュール境界 | 低-中 |
| イベント駆動 | 送信者と受信者を完全に分離 | 非同期処理、通知 | 中 |
| Facade パターン | 複雑なサブシステムへの依存を1点に集約 | レイヤー間の通信 | 低 |
| Adapter パターン | 外部ライブラリへの依存を隔離 | サードパーティ連携 | 低-中 |
| メッセージキュー | サービス間を物理的に分離 | マイクロサービス | 高 |

**コード例2: イベント駆動による疎結合化**

```python
from typing import Callable, Any
from dataclasses import dataclass, field
from datetime import datetime

# === Before: 強結合 ===
# OrderService が直接 InventoryService、NotificationService、AnalyticsService を呼ぶ

class OrderServiceTightlyCoupled:
    def __init__(self):
        self.inventory = InventoryService()     # 具象クラスに直接依存
        self.notification = NotificationService() # 具象クラスに直接依存
        self.analytics = AnalyticsService()     # 具象クラスに直接依存

    def place_order(self, order):
        self.inventory.reduce_stock(order.items)
        self.notification.send_confirmation(order)
        self.analytics.track_purchase(order)
        # → 新しい処理を追加するたびにこのクラスを修正する必要がある（OCP違反）
        # → 各サービスのテスト時にすべての依存を用意する必要がある


# === After: イベント駆動で疎結合 ===

@dataclass
class DomainEvent:
    """ドメインイベントの基底クラス"""
    occurred_at: datetime = field(default_factory=datetime.now)

@dataclass
class OrderPlacedEvent(DomainEvent):
    """注文確定イベント"""
    order_id: str = ""
    customer_id: str = ""
    items: list = field(default_factory=list)
    total_amount: float = 0.0

class EventBus:
    """シンプルなインメモリイベントバス"""
    def __init__(self):
        self._handlers: dict[type, list[Callable]] = {}

    def subscribe(self, event_type: type, handler: Callable) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: DomainEvent) -> None:
        for handler in self._handlers.get(type(event), []):
            handler(event)

    def unsubscribe(self, event_type: type, handler: Callable) -> None:
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

class OrderService:
    """注文サービス - イベントの発行のみを責任に持つ"""
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

    def place_order(self, order) -> None:
        order.confirm()
        # 他のサービスの存在を知らない
        self.event_bus.publish(OrderPlacedEvent(
            order_id=order.id,
            customer_id=order.customer_id,
            items=order.items,
            total_amount=order.total_amount
        ))

# 各ハンドラは独立して登録・テスト可能
class InventoryHandler:
    def handle_order_placed(self, event: OrderPlacedEvent) -> None:
        for item in event.items:
            self.reduce_stock(item.product_id, item.quantity)

class NotificationHandler:
    def handle_order_placed(self, event: OrderPlacedEvent) -> None:
        self.send_confirmation_email(event.customer_id, event.order_id)

class AnalyticsHandler:
    def handle_order_placed(self, event: OrderPlacedEvent) -> None:
        self.track_purchase(event.order_id, event.total_amount)

# 組み立て（Composition Root）
event_bus = EventBus()
event_bus.subscribe(OrderPlacedEvent, InventoryHandler().handle_order_placed)
event_bus.subscribe(OrderPlacedEvent, NotificationHandler().handle_order_placed)
event_bus.subscribe(OrderPlacedEvent, AnalyticsHandler().handle_order_placed)

# 新しいハンドラを追加しても OrderService は変更不要（OCP準拠）
# event_bus.subscribe(OrderPlacedEvent, LoyaltyPointHandler().handle_order_placed)
```

**コード例3: Dependency Injection による疎結合化**

```python
from abc import ABC, abstractmethod
from typing import Protocol

# === インターフェース（抽象）を定義 ===

class PaymentGateway(Protocol):
    """決済ゲートウェイのインターフェース"""
    def charge(self, amount: float, currency: str) -> PaymentResult: ...

class NotificationService(Protocol):
    """通知サービスのインターフェース"""
    def send(self, recipient: str, message: str) -> None: ...

class OrderRepository(Protocol):
    """注文リポジトリのインターフェース"""
    def save(self, order: Order) -> None: ...
    def find_by_id(self, order_id: str) -> Order | None: ...


# === 具象クラスの実装 ===

class StripePaymentGateway:
    """Stripe による決済実装"""
    def __init__(self, api_key: str):
        self.api_key = api_key

    def charge(self, amount: float, currency: str) -> PaymentResult:
        # Stripe API を呼び出す
        response = stripe.Charge.create(amount=int(amount * 100), currency=currency)
        return PaymentResult(success=True, transaction_id=response.id)

class EmailNotificationService:
    """メールによる通知実装"""
    def __init__(self, smtp_config: SmtpConfig):
        self.smtp = smtp_config

    def send(self, recipient: str, message: str) -> None:
        # SMTP経由でメール送信
        send_email(to=recipient, body=message, config=self.smtp)

class PostgresOrderRepository:
    """PostgreSQL による注文永続化"""
    def __init__(self, connection_pool):
        self.pool = connection_pool

    def save(self, order: Order) -> None:
        with self.pool.connection() as conn:
            conn.execute("INSERT INTO orders ...", order.to_dict())

    def find_by_id(self, order_id: str) -> Order | None:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT * FROM orders WHERE id = %s", [order_id])
            return Order.from_dict(row) if row else None


# === サービスは抽象にのみ依存 ===

class OrderService:
    """注文サービス - 具象クラスを一切知らない"""
    def __init__(
        self,
        repository: OrderRepository,
        payment: PaymentGateway,
        notification: NotificationService,
    ):
        self.repository = repository
        self.payment = payment
        self.notification = notification

    def place_order(self, order: Order) -> OrderResult:
        payment_result = self.payment.charge(order.total, order.currency)
        if not payment_result.success:
            return OrderResult.payment_failed(payment_result.error)

        self.repository.save(order)
        self.notification.send(
            order.customer_email,
            f"ご注文 {order.id} を承りました"
        )
        return OrderResult.success(order.id)


# === テスト時: モックを注入 ===

class MockPaymentGateway:
    def __init__(self, should_succeed: bool = True):
        self.should_succeed = should_succeed
        self.charges: list = []

    def charge(self, amount: float, currency: str) -> PaymentResult:
        self.charges.append((amount, currency))
        if self.should_succeed:
            return PaymentResult(success=True, transaction_id="mock-txn-001")
        return PaymentResult(success=False, error="Mock decline")

class MockNotificationService:
    def __init__(self):
        self.sent_messages: list = []

    def send(self, recipient: str, message: str) -> None:
        self.sent_messages.append((recipient, message))

# テスト
def test_place_order_success():
    mock_payment = MockPaymentGateway(should_succeed=True)
    mock_notification = MockNotificationService()
    mock_repository = InMemoryOrderRepository()

    service = OrderService(mock_repository, mock_payment, mock_notification)
    result = service.place_order(create_test_order())

    assert result.is_success
    assert len(mock_payment.charges) == 1
    assert len(mock_notification.sent_messages) == 1
```

### 1.4 結合度の定量的測定

結合度は主観的な判断だけでなく、静的解析ツールで定量的に測定できる。

| メトリクス | 定義 | 理想値 | ツール |
|-----------|------|--------|--------|
| CBO (Coupling Between Objects) | あるクラスが依存する他クラスの数 | 10以下 | SonarQube, JDepend |
| Ca (Afferent Coupling) | そのモジュールに依存している外部モジュール数 | -- | NDepend, Structure101 |
| Ce (Efferent Coupling) | そのモジュールが依存している外部モジュール数 | -- | NDepend, Structure101 |
| Instability = Ce / (Ca + Ce) | 不安定度。1に近いほど不安定 | 安定/不安定を設計的に決定 | NDepend |

```python
# CBO（Coupling Between Objects）の計測例
# 以下のクラスのCBOを数える

class OrderService:                  # CBO = 5
    def __init__(
        self,
        repository: OrderRepository,    # 1. OrderRepository
        payment: PaymentGateway,         # 2. PaymentGateway
        notification: NotificationService, # 3. NotificationService
        logger: Logger,                  # 4. Logger
    ):
        pass

    def place_order(self, order: Order) -> OrderResult:  # 5. Order, 6. OrderResult
        pass
    # → CBO = 6（依存先クラス数）
    # → 10以下なので許容範囲

# CBOが高すぎるクラスの例
class GodService:                    # CBO = 15+
    def __init__(
        self,
        user_repo, order_repo, product_repo,     # 3
        payment, shipping, tax_calculator,        # 3
        email_service, sms_service, push_service, # 3
        cache, logger, metrics,                   # 3
        config, event_bus, scheduler              # 3
    ):
        pass
    # → CBO = 15: リファクタリング対象
```

**Instability（不安定度）の設計活用:**

```
  安定依存の原則 (Stable Dependencies Principle)

  不安定なモジュールは安定したモジュールに依存すべきで、
  その逆はあってはならない。

  Instability = Ce / (Ca + Ce)

  安定（I=0）                     不安定（I=1）
  ┌───────────┐                  ┌───────────┐
  │ 抽象層     │ ←── 依存 ──── │ UI層       │
  │ Ca=10,Ce=0│                  │ Ca=0,Ce=5 │
  │ I = 0.0   │                  │ I = 1.0   │
  └───────────┘                  └───────────┘
  変更されにくい                   自由に変更可能
  （多くに依存されている）          （何にも依存されていない）

  ✗ 安定モジュールが不安定モジュールに依存 → 危険
  ✓ 不安定モジュールが安定モジュールに依存 → 安全
```

---

## 2. 凝集度（Cohesion）── モジュール内の要素の関連性

### 2.1 なぜ凝集度を理解すべきか

凝集度が低いモジュールは以下の問題を引き起こす。

1. **変更理由の多さ**: 無関係な要素が集まっているため、様々な理由で変更される（SRP違反）
2. **理解コストの増大**: モジュールの目的が不明確で、読み解くのに時間がかかる
3. **再利用性の低下**: 不要な依存まで持ち込むため、他のプロジェクトで再利用しにくい
4. **テスト困難**: 何をテストすべきかが不明確で、テストケースが膨大になる

```
  凝集度と保守性の関係（実証研究: Bieman & Kang, 1995）

  保守性
  (理解容易性)
    ^
    |                                    ★ 機能的凝集
    |                              ★ 逐次的凝集
    |                        ★ 通信的凝集
    |                  ★ 手続き的凝集
    |            ★ 時間的凝集
    |      ★ 論理的凝集
    |★ 偶発的凝集
    +──────────────────────────────────→ 凝集度
    低                                  高
```

### 2.2 凝集度の7段階

```
  品質: 低 ←──────────────────────────────────────────→ 高

  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │偶発的 │論理的 │時間的 │手続き │通信的 │逐次的 │機能的 │
  │Coinci│Logical│Tempor│Proced│Commun│Sequen│Functi│
  │dental │      │al    │ural  │icatio│tial  │onal  │
  │      │      │      │      │nal   │      │      │
  │無関係 │似た種 │同時に │特定の │同じデ │前の出 │1つの │
  │な要素 │類を集 │実行す │順序で │ータを │力が次 │明確な │
  │の寄せ │めた  │るだけ │実行  │操作  │の入力 │責任  │
  │集め  │だけ  │      │      │      │      │      │
  └──────┴──────┴──────┴──────┴──────┴──────┴──────┘
  避ける   避ける  注意   許容   良い   良い   最高
```

**各段階の詳細定義と具体例:**

| 段階 | 名称 | 定義 | 見分け方 |
|------|------|------|---------|
| 1 | 偶発的凝集 | 無関係な要素を1つにまとめただけ | クラス名が `Util`, `Manager`, `Helper` |
| 2 | 論理的凝集 | 論理的に似た種類を集めただけ | 引数やフラグで処理を切り替え |
| 3 | 時間的凝集 | 同じタイミングで実行する処理をまとめた | `initialize()`, `cleanup()` |
| 4 | 手続き的凝集 | 特定の実行順序で処理する | 順序を変えると壊れるが、データは共有しない |
| 5 | 通信的凝集 | 同じデータを操作する処理をまとめた | 全メソッドが同じフィールドを使う |
| 6 | 逐次的凝集 | 前の処理の出力が次の処理の入力になる | パイプライン処理 |
| 7 | 機能的凝集 | 1つの明確で単一の責任を持つ | 「このクラスは何をする？」に1文で答えられる |

**コード例4: 凝集度の段階別コード**

```java
// === 1. 偶発的凝集（最低）: 無関係な機能の寄せ集め ===
class Utilities {
    public static String formatDate(Date d) { /* 日付処理 */ }
    public static double calculateTax(double amount) { /* 税計算 */ }
    public static void sendEmail(String to, String body) { /* メール送信 */ }
    public static Image resizeImage(Image img, int w, int h) { /* 画像処理 */ }
    // → 日付、税、メール、画像に何の関係もない
}

// === 2. 論理的凝集（低い）: 似た種類を集めただけ ===
class InputHandler {
    public void handleMouseInput(MouseEvent e) { /* マウス処理 */ }
    public void handleKeyboardInput(KeyEvent e) { /* キーボード処理 */ }
    public void handleTouchInput(TouchEvent e) { /* タッチ処理 */ }
    public void handleGamepadInput(GamepadEvent e) { /* ゲームパッド処理 */ }
    // → 「入力」という論理的カテゴリで集めただけ。各処理は独立
}

// === 3. 時間的凝集（中程度）: 同じタイミングで実行するだけ ===
class AppInitializer {
    public void initialize() {
        loadConfig();       // 設定読み込み
        initDatabase();     // DB初期化
        startWebServer();   // Webサーバー起動
        registerShutdownHook(); // シャットダウンフック登録
    }
    // → 「アプリ起動時」というタイミングで集めただけ
}

// === 4. 手続き的凝集: 特定の順序で実行 ===
class FileProcessor {
    public void process(String path) {
        openFile(path);
        readHeader();
        parseBody();
        closeFile();
    }
    // → 順序は決まっているが、open/read/parse/closeは概念的に異なる
}

// === 5. 通信的凝集: 同じデータを操作 ===
class EmployeeReport {
    private List<Employee> employees;

    public double calculateAverageSalary() { /* employees を使う */ }
    public Employee findHighestPaid() { /* employees を使う */ }
    public List<Employee> filterByDepartment(String dept) { /* employees を使う */ }
    // → 全メソッドが employees フィールドを操作
}

// === 6. 逐次的凝集: パイプライン処理 ===
class DataPipeline {
    public Report generate(RawData raw) {
        CleanedData cleaned = clean(raw);       // 生データ → 洗浄データ
        AnalyzedData analyzed = analyze(cleaned); // 洗浄データ → 分析データ
        return format(analyzed);                  // 分析データ → レポート
    }
    // → 各ステップの出力が次のステップの入力
}

// === 7. 機能的凝集（最高）: 1つの明確な責任 ===
class PasswordHasher {
    private final int saltLength;
    private final int iterations;

    public PasswordHasher(int saltLength, int iterations) {
        this.saltLength = saltLength;
        this.iterations = iterations;
    }

    public String hash(String password) {
        byte[] salt = generateSalt();
        return pbkdf2(password, salt, iterations);
    }

    public boolean verify(String password, String hashedPassword) {
        byte[] salt = extractSalt(hashedPassword);
        String rehashed = pbkdf2(password, salt, iterations);
        return constantTimeEquals(rehashed, hashedPassword);
    }

    private byte[] generateSalt() { /* ソルト生成 */ }
    private byte[] extractSalt(String hash) { /* ソルト抽出 */ }
    private String pbkdf2(String password, byte[] salt, int iterations) { /* ハッシュ計算 */ }
    // → 「パスワードのハッシュ化」という1つの責任のみ
}
```

### 2.3 凝集度の定量的測定: LCOM

**LCOM (Lack of Cohesion in Methods)** はクラスの凝集度を定量的に測定するメトリクスである。

```
  LCOM の計算方法（Henderson-Sellers版 LCOM*）

  LCOM* = (m - (1/f) * Σsum(mf)) / (m - 1)

  m  = メソッド数
  f  = フィールド数
  mf = 各フィールドにアクセスするメソッド数の合計

  LCOM* の範囲: 0 ～ 1
  0 = 完全に凝集（全メソッドが全フィールドを使う）
  1 = 完全に非凝集（各メソッドが異なるフィールドを使う）
```

```python
# LCOM の計算例

class HighCohesion:
    """LCOM = 低い（凝集度が高い）"""
    def __init__(self, x, y):
        self.x = x  # フィールド1
        self.y = y  # フィールド2

    def distance_from_origin(self):
        return (self.x**2 + self.y**2) ** 0.5  # x, y 両方使用

    def move(self, dx, dy):
        self.x += dx  # x 使用
        self.y += dy  # y 使用

    def __str__(self):
        return f"({self.x}, {self.y})"  # x, y 両方使用

    # m=3, f=2
    # x: 3メソッドがアクセス, y: 3メソッドがアクセス
    # LCOM* = (3 - (1/2) * (3+3)) / (3-1) = (3 - 3) / 2 = 0
    # → LCOM = 0: 完全に凝集


class LowCohesion:
    """LCOM = 高い（凝集度が低い）"""
    def __init__(self):
        self.user_name = ""     # フィールド1
        self.order_total = 0.0  # フィールド2
        self.log_level = "INFO" # フィールド3

    def get_user_name(self):
        return self.user_name       # user_name のみ使用

    def calculate_total(self):
        return self.order_total * 1.1  # order_total のみ使用

    def set_log_level(self, level):
        self.log_level = level      # log_level のみ使用

    # m=3, f=3
    # user_name: 1, order_total: 1, log_level: 1
    # LCOM* = (3 - (1/3) * (1+1+1)) / (3-1) = (3 - 1) / 2 = 1.0
    # → LCOM = 1.0: 完全に非凝集 → 3つの独立したクラスに分割すべき
```

---

## 3. 低結合・高凝集の実現パターン

### 3.1 Facade パターンで結合度を管理

**コード例5: Facade パターン**

```typescript
// ============================================================
// Before: 高結合 - クライアントが複数のサブシステムに直接依存
// ============================================================
class OrderPage {
  placeOrder(cart: Cart) {
    const inventory = new InventorySystem();
    const payment = new PaymentGateway();
    const shipping = new ShippingCalculator();
    const notification = new EmailService();
    const loyalty = new LoyaltyPointService();

    // 5つのサブシステムに直接依存（CBO = 5）
    const available = inventory.check(cart.items);
    if (!available) throw new Error('在庫不足');

    const total = shipping.calculate(cart);
    const paymentResult = payment.charge(cart.customer, total);
    notification.sendConfirmation(cart.customer.email);
    loyalty.addPoints(cart.customer.id, Math.floor(total / 100));
  }
}

// ============================================================
// After: Facade で結合を集約
// ============================================================

// Facade は内部のサブシステムを隠蔽する
class OrderFacade {
  constructor(
    private inventory: InventorySystem,
    private payment: PaymentGateway,
    private shipping: ShippingCalculator,
    private notification: EmailService,
    private loyalty: LoyaltyPointService
  ) {}

  placeOrder(cart: Cart): OrderResult {
    // 内部の協調ロジックを Facade が管理
    if (!this.inventory.check(cart.items)) {
      return OrderResult.outOfStock();
    }

    const total = this.shipping.calculate(cart);

    const paymentResult = this.payment.charge(cart.customer, total);
    if (!paymentResult.success) {
      return OrderResult.paymentFailed(paymentResult.error);
    }

    // 非クリティカルな処理は失敗しても注文は成功とする
    this.trySendConfirmation(cart.customer.email);
    this.tryAddLoyaltyPoints(cart.customer.id, total);

    return OrderResult.success(paymentResult.transactionId);
  }

  private trySendConfirmation(email: string): void {
    try {
      this.notification.sendConfirmation(email);
    } catch (e) {
      console.warn('確認メール送信失敗', e);
    }
  }

  private tryAddLoyaltyPoints(customerId: string, total: number): void {
    try {
      this.loyalty.addPoints(customerId, Math.floor(total / 100));
    } catch (e) {
      console.warn('ポイント付与失敗', e);
    }
  }
}

// クライアントは Facade だけに依存（CBO = 1）
class OrderPage {
  constructor(private orderFacade: OrderFacade) {}

  placeOrder(cart: Cart) {
    return this.orderFacade.placeOrder(cart);
  }
}
```

### 3.2 パッケージ構造で凝集度を表現

**コード例6: ドメイン基準のパッケージ構成**

```
# ============================================================
# Before: 低凝集なパッケージ構成（技術レイヤー基準）
# ============================================================
# 1つの変更が複数ディレクトリに波及する
# 例: User の新フィールド追加 → 3ディレクトリを修正

src/
  controllers/           # 全ドメインのControllerが混在
    UserController.ts
    OrderController.ts
    ProductController.ts
  services/              # 全ドメインのServiceが混在
    UserService.ts
    OrderService.ts
    ProductService.ts
  repositories/          # 全ドメインのRepositoryが混在
    UserRepository.ts
    OrderRepository.ts
    ProductRepository.ts

# ============================================================
# After: 高凝集なパッケージ構成（ドメイン基準）
# ============================================================
# 1つの変更は1つのディレクトリ内で完結する
# 例: User の新フィールド追加 → user/ 内だけを修正

src/
  user/                  # User ドメインの全要素が集約
    UserController.ts
    UserService.ts
    UserRepository.ts
    User.ts
    UserValidator.ts
    user.test.ts
    index.ts             # 公開APIのみエクスポート
  order/                 # Order ドメインの全要素が集約
    OrderController.ts
    OrderService.ts
    OrderRepository.ts
    Order.ts
    OrderValidator.ts
    order.test.ts
    index.ts
  product/               # Product ドメインの全要素が集約
    ProductController.ts
    ProductService.ts
    ProductRepository.ts
    Product.ts
    product.test.ts
    index.ts
  shared/                # 共有ユーティリティ（最小限に）
    types.ts
    errors.ts
    logger.ts
```

### 3.3 Adapter パターンで外部結合を隔離

**コード例7: 外部ライブラリの変更影響を局所化**

```python
from abc import ABC, abstractmethod
from typing import Any

# === インターフェース: アプリケーションが期待する契約 ===

class FileStorage(ABC):
    """ファイルストレージのインターフェース"""
    @abstractmethod
    def upload(self, key: str, data: bytes) -> str:
        """ファイルをアップロードし、URLを返す"""
        pass

    @abstractmethod
    def download(self, key: str) -> bytes:
        """ファイルをダウンロードする"""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """ファイルを削除する"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """ファイルが存在するか確認する"""
        pass


# === Adapter: AWS S3 の実装 ===

class S3FileStorage(FileStorage):
    """AWS S3 による実装"""
    def __init__(self, bucket_name: str, region: str):
        import boto3
        self.s3 = boto3.client('s3', region_name=region)
        self.bucket = bucket_name

    def upload(self, key: str, data: bytes) -> str:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)
        return f"https://{self.bucket}.s3.amazonaws.com/{key}"

    def download(self, key: str) -> bytes:
        response = self.s3.get_object(Bucket=self.bucket, Key=key)
        return response['Body'].read()

    def delete(self, key: str) -> None:
        self.s3.delete_object(Bucket=self.bucket, Key=key)

    def exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.s3.exceptions.NoSuchKey:
            return False


# === Adapter: ローカルファイルシステムの実装（開発用） ===

class LocalFileStorage(FileStorage):
    """ローカルファイルシステムによる実装（開発・テスト用）"""
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def upload(self, key: str, data: bytes) -> str:
        file_path = self.base_dir / key
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(data)
        return f"file://{file_path}"

    def download(self, key: str) -> bytes:
        return (self.base_dir / key).read_bytes()

    def delete(self, key: str) -> None:
        (self.base_dir / key).unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return (self.base_dir / key).exists()


# === サービス: ストレージの実装を知らない ===

class DocumentService:
    """ドキュメントサービス - FileStorage インターフェースにのみ依存"""
    def __init__(self, storage: FileStorage):
        self.storage = storage

    def save_document(self, name: str, content: bytes) -> str:
        key = f"documents/{name}"
        return self.storage.upload(key, content)

    def get_document(self, name: str) -> bytes:
        key = f"documents/{name}"
        return self.storage.download(key)

# 使用例
# 本番: DocumentService(S3FileStorage("my-bucket", "ap-northeast-1"))
# 開発: DocumentService(LocalFileStorage("/tmp/dev-storage"))
# テスト: DocumentService(InMemoryFileStorage())
```

---

## 4. 結合度と凝集度の関係

### 4.1 2軸の組み合わせ

| 組み合わせ | 低結合 | 高結合 |
|-----------|--------|--------|
| **高凝集** | **理想的**。独立した明確なモジュール。変更が局所的で、テストが容易 | 責任は明確だが依存が多い。DIやイベント駆動で改善可能 |
| **低凝集** | 依存は少ないがモジュールの意味が不明。分割・統合で改善 | **最悪**。スパゲッティコード。全面的リファクタリングが必要 |

```
                 結合度
          低い ←────────→ 高い
         ┌──────┬──────┐
  凝  高 │  ★   │  △   │
  集  い │ 理想  │ DI等 │
  度     │      │で改善 │
         ├──────┼──────┤
     低  │  △   │  ✗   │
     い  │ 分割 │スパゲ │
         │で改善 │ッティ│
         └──────┴──────┘
```

### 4.2 改善アプローチの選択

| 改善アプローチ | 対象 | 具体的手法 | 優先度 |
|--------------|------|-----------|--------|
| 結合度を下げる | モジュール間 | DI、インターフェース、イベント駆動、Adapter | 高 |
| 凝集度を上げる | モジュール内 | Extract Class、Move Method、Inline Class | 高 |
| 両方同時に改善 | アーキテクチャ | ドメイン駆動設計（DDD）、モジュラーモノリス | 中 |
| パッケージ再構成 | ディレクトリ | 機能凝集のパッケージ構成に移行 | 中 |
| API境界の定義 | モジュール公開面 | Public API を最小限にし、内部を隠蔽 | 高 |

### 4.3 アーキテクチャパターン別の結合度・凝集度

```
  アーキテクチャパターンと結合度・凝集度の関係

  ┌─────────────────┬───────┬───────┬─────────────────┐
  │ パターン         │ 結合度 │ 凝集度 │ 特徴             │
  ├─────────────────┼───────┼───────┼─────────────────┤
  │ モノリス         │ 高    │ 低    │ 単一デプロイ      │
  │ (レイヤード)     │       │       │ 変更影響: 大      │
  ├─────────────────┼───────┼───────┼─────────────────┤
  │ モジュラーモノリス│ 低-中 │ 高    │ 単一デプロイ      │
  │                 │       │       │ モジュール境界明確 │
  ├─────────────────┼───────┼───────┼─────────────────┤
  │ マイクロサービス  │ 低    │ 高    │ 独立デプロイ      │
  │                 │       │       │ 運用コスト: 高    │
  ├─────────────────┼───────┼───────┼─────────────────┤
  │ イベント駆動     │ 最低  │ 高    │ 非同期通信        │
  │                 │       │       │ デバッグ困難      │
  └─────────────────┴───────┴───────┴─────────────────┘
```

**コード例8: モジュラーモノリスの境界設計**

```python
# === モジュラーモノリス: 明確な境界を持つモジュール設計 ===

# --- モジュールの公開API（index.py / __init__.py） ---

# user_module/__init__.py
"""User モジュールの公開API"""
from .service import UserService
from .models import User, UserProfile
from .events import UserCreatedEvent, UserDeletedEvent

# 内部クラスはエクスポートしない
# UserRepository, UserValidator, UserMapper は内部実装

__all__ = ['UserService', 'User', 'UserProfile', 'UserCreatedEvent', 'UserDeletedEvent']


# --- モジュール間の通信: 公開APIのみ使用 ---

# order_module/service.py
class OrderService:
    def __init__(self, user_service: UserService):  # 公開APIのみに依存
        self.user_service = user_service

    def create_order(self, user_id: str, items: list) -> Order:
        # UserModule の公開APIのみ使用（内部実装にはアクセスしない）
        user = self.user_service.get_user(user_id)
        if not user:
            raise UserNotFoundError(user_id)
        return Order(user_id=user.id, items=items)


# --- モジュール間のルール ---
# 1. 他モジュールの内部クラスを直接 import しない
# 2. 他モジュールのDBテーブルに直接アクセスしない
# 3. モジュール間通信はイベントまたは公開APIのみ
# 4. 共有データは共有カーネル（shared kernel）に配置
```

---

## 5. 実践的なリファクタリング手順

### 5.1 低凝集クラスの分割手順

```
  God Class の分割プロセス

  Step 1: 責任の識別
  ┌─────────────────────────────────────────────┐
  │  God Class (ApplicationManager)             │
  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐      │
  │  │ 認証  │ │ 決済  │ │ 通知  │ │ 在庫  │      │
  │  └──────┘ └──────┘ └──────┘ └──────┘      │
  └─────────────────────────────────────────────┘

  Step 2: Extract Class
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ AuthService│ │PayService │ │ NotifySvc │ │InventSvc │
  └──────────┘ └──────────┘ └──────────┘ └──────────┘

  Step 3: インターフェースで結合度を管理
  ┌──────────┐     ┌──────────────┐     ┌──────────┐
  │ AuthService│ ──→ │ <<interface>> │ ←── │PayService │
  └──────────┘     │  IPayment    │     └──────────┘
                    └──────────────┘
```

**コード例9: God Class のリファクタリング**

```python
# === Before: God Class（低凝集、高結合）===

class ApplicationManager:
    """すべての機能を持つ巨大クラス"""
    def __init__(self):
        self.db = Database()
        self.smtp = SmtpClient()
        self.cache = RedisCache()

    # 認証系
    def authenticate_user(self, username, password): ...
    def reset_password(self, email): ...
    def generate_token(self, user_id): ...
    def validate_token(self, token): ...

    # 注文系
    def create_order(self, user_id, items): ...
    def cancel_order(self, order_id): ...
    def calculate_shipping(self, order_id): ...

    # 通知系
    def send_email(self, to, subject, body): ...
    def send_sms(self, phone, message): ...
    def send_push_notification(self, device_id, message): ...

    # 在庫系
    def check_inventory(self, product_id): ...
    def update_stock(self, product_id, quantity): ...
    def reorder_if_low(self, product_id): ...

    # レポート系
    def generate_sales_report(self, period): ...
    def generate_inventory_report(self): ...

    # → LCOM は 1.0 に近い（各メソッドグループが異なるフィールドを使用）
    # → CBO は 20+ （大量の外部依存）
    # → 変更理由が 5+ （認証、注文、通知、在庫、レポート）


# === After: 責任ごとに分割（高凝集、低結合）===

class AuthService:
    """認証のみを担当（機能的凝集）"""
    def __init__(self, user_repo: UserRepository, token_provider: TokenProvider):
        self.user_repo = user_repo
        self.token_provider = token_provider

    def authenticate(self, username: str, password: str) -> AuthResult:
        user = self.user_repo.find_by_username(username)
        if user and user.verify_password(password):
            token = self.token_provider.generate(user.id)
            return AuthResult.success(token)
        return AuthResult.failure("認証失敗")

    def validate_token(self, token: str) -> TokenClaims | None:
        return self.token_provider.validate(token)


class OrderService:
    """注文処理のみを担当（機能的凝集）"""
    def __init__(
        self,
        order_repo: OrderRepository,
        inventory: InventoryService,
        event_bus: EventBus
    ):
        self.order_repo = order_repo
        self.inventory = inventory
        self.event_bus = event_bus

    def create_order(self, user_id: str, items: list[OrderItem]) -> Order:
        for item in items:
            if not self.inventory.is_available(item.product_id, item.quantity):
                raise InsufficientStockError(item.product_id)

        order = Order(user_id=user_id, items=items)
        self.order_repo.save(order)
        self.event_bus.publish(OrderCreatedEvent(order_id=order.id))
        return order


class NotificationService:
    """通知のみを担当（機能的凝集）"""
    def __init__(self, channels: list[NotificationChannel]):
        self.channels = channels

    def send(self, recipient: str, message: str, channel_type: str) -> None:
        channel = self._find_channel(channel_type)
        channel.send(recipient, message)


class InventoryService:
    """在庫管理のみを担当（機能的凝集）"""
    def __init__(self, inventory_repo: InventoryRepository, event_bus: EventBus):
        self.repo = inventory_repo
        self.event_bus = event_bus

    def is_available(self, product_id: str, quantity: int) -> bool:
        stock = self.repo.get_stock(product_id)
        return stock >= quantity

    def reduce_stock(self, product_id: str, quantity: int) -> None:
        self.repo.decrease(product_id, quantity)
        current = self.repo.get_stock(product_id)
        if current < self.repo.get_reorder_threshold(product_id):
            self.event_bus.publish(LowStockEvent(product_id=product_id))
```

### 5.2 高結合の解消手順

```
  高結合の解消フローチャート

  開始
    │
    ▼
  依存グラフを描く
    │
    ▼
  循環依存はあるか？ ─── Yes ──→ 共通インターフェース抽出
    │ No                          またはイベント駆動化
    ▼
  直接 new しているか？ ─── Yes ──→ DI コンテナ導入
    │ No
    ▼
  具象クラスに依存？ ─── Yes ──→ インターフェース抽出
    │ No
    ▼
  外部ライブラリに直接依存？ ─── Yes ──→ Adapter パターン
    │ No
    ▼
  グローバル状態を共有？ ─── Yes ──→ 引数渡しに変更
    │ No
    ▼
  現在の結合度は適切
```

---

## 6. マイクロサービスにおける結合度・凝集度

### 6.1 マイクロサービスの結合度問題

マイクロサービスにすれば自動的に低結合になるわけではない。以下の表は分散システム特有の結合度問題を示す。

| 結合の種類 | 説明 | 解決策 |
|-----------|------|--------|
| 共有データベース結合 | 複数サービスが同じDBテーブルを参照 | サービスごとにDBを分離 |
| 同期API結合 | サービスAがサービスBのAPIを同期的に呼ぶ | 非同期メッセージングに変更 |
| 共有ライブラリ結合 | 共通ライブラリのバージョンアップで全サービス再デプロイ | 契約テスト、独立バージョニング |
| 時間的結合 | サービスAが利用可能でないとサービスBが動けない | サーキットブレーカー、フォールバック |
| デプロイ結合 | サービスを一緒にデプロイしないと動かない | 独立デプロイ可能な設計に |

**コード例10: マイクロサービスの結合度管理**

```python
# === 悪い: 同期的なサービス間呼び出しチェーン ===

class OrderApiHandler:
    """注文API - 同期的に3つのサービスを呼ぶ"""
    async def create_order(self, request):
        # 1. ユーザーサービスに問い合わせ（同期）
        user = await self.http_client.get(f"http://user-service/users/{request.user_id}")
        # → user-service がダウンすると注文不可

        # 2. 在庫サービスに問い合わせ（同期）
        stock = await self.http_client.get(
            f"http://inventory-service/stock/{request.product_id}"
        )
        # → inventory-service がダウンすると注文不可

        # 3. 決済サービスに問い合わせ（同期）
        payment = await self.http_client.post(
            "http://payment-service/charge",
            json={"amount": request.total}
        )
        # → payment-service がダウンすると注文不可

        # 3つのサービスすべてが利用可能でないと動かない（時間的結合）


# === 良い: 非同期イベントとサーキットブレーカー ===

class OrderApiHandler:
    """注文API - 非同期イベント駆動"""
    def __init__(
        self,
        order_repo: OrderRepository,
        message_queue: MessageQueue,
        circuit_breaker: CircuitBreaker,
        user_cache: UserCache,
    ):
        self.order_repo = order_repo
        self.mq = message_queue
        self.cb = circuit_breaker
        self.user_cache = user_cache

    async def create_order(self, request) -> OrderResult:
        # ユーザー情報はキャッシュから取得（user-service がダウンしても動く）
        user = self.user_cache.get(request.user_id)
        if not user:
            user = await self.cb.call(
                lambda: self.http_client.get(f"http://user-service/users/{request.user_id}")
            )
            self.user_cache.set(request.user_id, user)

        # 注文を「保留」状態で保存（ローカルDBのみ）
        order = Order(user_id=user.id, items=request.items, status='PENDING')
        self.order_repo.save(order)

        # 非同期イベントを発行（他のサービスが独立して処理）
        await self.mq.publish('order.created', {
            'order_id': order.id,
            'items': [item.to_dict() for item in request.items],
            'total': request.total,
        })

        return OrderResult.pending(order.id)
        # → 他のサービスがダウンしても注文の受付自体は可能
```

---

## 7. アンチパターン

### アンチパターン1: God Module（低凝集の極致）

```python
# NG: 1つのモジュールがシステム全体の機能を持つ
class ApplicationManager:
    def authenticate_user(self, ...): ...
    def process_payment(self, ...): ...
    def generate_invoice(self, ...): ...
    def send_notification(self, ...): ...
    def update_inventory(self, ...): ...
    def calculate_shipping(self, ...): ...
    def manage_cache(self, ...): ...
    # 全ドメインの知識がここに集中
    # LCOM ≈ 1.0（各メソッドグループが異なるフィールドを使用）
    # CBO ≥ 20（大量の外部依存）

# OK: ドメインごとにサービスを分割
class AuthService:
    """認証のみ"""
    def authenticate(self, username, password): ...
    def validate_token(self, token): ...

class PaymentService:
    """決済のみ"""
    def charge(self, amount, payment_method): ...
    def refund(self, transaction_id): ...

class NotificationService:
    """通知のみ"""
    def send_email(self, to, template, data): ...
    def send_push(self, device_id, message): ...
```

### アンチパターン2: Shotgun Surgery（高結合の結果）

```python
# NG: 1つの変更が多数のファイルに波及する
# 「消費税率を8%→10%に変更」で修正が必要なファイル:
#
# - cart.py (税計算)
# - invoice.py (請求書の税額)
# - receipt.py (領収書の税額)
# - report.py (レポートの税表示)
# - api.py (APIレスポンスの税額)
# - frontend/cart.js (フロントの税表示)
# → 税率がDRY化されていない証拠
# → 1箇所の変更が6ファイルに波及 = Shotgun Surgery

# OK: 税計算を1箇所に集約
class TaxCalculator:
    """税計算の単一責任"""
    TAX_RATE = Decimal('0.10')  # 税率は1箇所で管理

    @classmethod
    def calculate(cls, subtotal: Decimal) -> TaxResult:
        tax = subtotal * cls.TAX_RATE
        return TaxResult(subtotal=subtotal, tax=tax, total=subtotal + tax)

    @classmethod
    def get_display_rate(cls) -> str:
        return f"{cls.TAX_RATE * 100}%"

# 全モジュールが TaxCalculator を使用
# 税率変更時は TaxCalculator.TAX_RATE の1箇所のみ修正
```

### アンチパターン3: 隠れた結合（Hidden Coupling）

```python
# NG: 暗黙の実行順序依存
class DataProcessor:
    def __init__(self):
        self.data = None
        self.processed = None

    def load(self, path):
        self.data = read_file(path)

    def process(self):
        # load() が先に呼ばれていることを暗黙に仮定
        self.processed = transform(self.data)  # data が None だとエラー

    def save(self, path):
        # process() が先に呼ばれていることを暗黙に仮定
        write_file(path, self.processed)  # processed が None だとエラー

# 呼び出し側が正しい順序を知っている必要がある（時間的結合）
processor = DataProcessor()
processor.load("input.csv")
processor.process()
processor.save("output.csv")

# OK: メソッドチェーンまたは単一メソッドで順序を保証
class DataProcessor:
    @staticmethod
    def process_file(input_path: str, output_path: str) -> None:
        """ファイル処理の全手順を1メソッドで実行"""
        data = read_file(input_path)
        processed = transform(data)
        write_file(output_path, processed)

# または不変オブジェクトのパイプライン
class DataProcessor:
    @staticmethod
    def load(path: str) -> RawData:
        return RawData(read_file(path))

    @staticmethod
    def process(data: RawData) -> ProcessedData:
        return ProcessedData(transform(data.content))

    @staticmethod
    def save(data: ProcessedData, path: str) -> None:
        write_file(path, data.content)

# 型システムが順序を強制
raw = DataProcessor.load("input.csv")
processed = DataProcessor.process(raw)
DataProcessor.save(processed, "output.csv")
```

---

## 8. 演習問題

### 演習1（基礎）: 結合度と凝集度の識別

以下のコードの結合度・凝集度を評価し、問題点を特定してください。

```python
# 評価対象
config = {"db_host": "localhost", "db_port": 5432, "api_key": "secret"}

class AppService:
    def get_user(self, user_id):
        host = config["db_host"]  # グローバル変数に依存
        conn = connect(host, config["db_port"])
        return conn.query(f"SELECT * FROM users WHERE id = {user_id}")

    def send_report(self, email):
        import smtplib
        server = smtplib.SMTP("smtp.example.com")
        server.sendmail("noreply@example.com", email, "Report attached")

    def resize_image(self, image_path, width, height):
        from PIL import Image
        img = Image.open(image_path)
        return img.resize((width, height))
```

**期待される回答:**

```
結合度の評価:
- 共通結合: グローバル変数 config に依存（Bad）
- 内容結合に近い: SQL文を直接構築（DB構造に依存）
- 外部結合: SMTP サーバーのアドレスをハードコード

凝集度の評価:
- 偶発的凝集: DB操作、メール送信、画像処理が1クラスに混在
- LCOM ≈ 1.0（各メソッドが異なるリソースを使用）

改善案:
1. UserRepository（DB操作専用）
2. EmailService（メール送信専用）
3. ImageProcessor（画像処理専用）
4. 各クラスにDIで依存を注入
5. グローバル変数 config は設定クラスとして注入
```

### 演習2（応用）: Facade パターンで結合度を改善

以下の高結合なコードをFacadeパターンでリファクタリングしてください。

```typescript
class CheckoutPage {
  async checkout(cartId: string) {
    const cart = await fetch(`/api/carts/${cartId}`).then(r => r.json());
    const user = await fetch(`/api/users/${cart.userId}`).then(r => r.json());
    const inventory = await fetch('/api/inventory/check', {
      method: 'POST', body: JSON.stringify(cart.items)
    }).then(r => r.json());

    if (!inventory.available) throw new Error('在庫不足');

    const tax = cart.subtotal * 0.1;
    const shipping = cart.items.length > 3 ? 0 : 500;
    const total = cart.subtotal + tax + shipping;

    const payment = await fetch('/api/payments/charge', {
      method: 'POST',
      body: JSON.stringify({ userId: user.id, amount: total, card: user.defaultCard })
    }).then(r => r.json());

    await fetch('/api/notifications/email', {
      method: 'POST',
      body: JSON.stringify({ to: user.email, template: 'order-confirm', data: { total } })
    });

    return { orderId: payment.orderId, total };
  }
}
```

**期待される回答:**

```typescript
// CheckoutFacade: 内部の複雑さを隠蔽
class CheckoutFacade {
  constructor(
    private cartService: CartService,
    private userService: UserService,
    private inventoryService: InventoryService,
    private pricingService: PricingService,
    private paymentService: PaymentService,
    private notificationService: NotificationService
  ) {}

  async checkout(cartId: string): Promise<CheckoutResult> {
    const cart = await this.cartService.getCart(cartId);
    const user = await this.userService.getUser(cart.userId);

    await this.inventoryService.ensureAvailable(cart.items);

    const pricing = this.pricingService.calculate(cart);
    const payment = await this.paymentService.charge(user, pricing.total);

    // 非クリティカル: 失敗しても注文は成功
    await this.notificationService
      .sendOrderConfirmation(user.email, pricing)
      .catch(err => console.warn('通知失敗', err));

    return { orderId: payment.orderId, total: pricing.total };
  }
}

// CheckoutPage は Facade だけに依存
class CheckoutPage {
  constructor(private checkout: CheckoutFacade) {}

  async checkout(cartId: string) {
    return this.checkout.checkout(cartId);
  }
}
```

### 演習3（発展）: LCOM の計算と改善

以下のクラスのLCOM（Henderson-Sellers版）を計算し、凝集度を改善するリファクタリングを行ってください。

```python
class ReportManager:
    def __init__(self):
        self.sales_data = []
        self.employee_data = []
        self.inventory_data = []

    def add_sale(self, sale):
        self.sales_data.append(sale)

    def get_total_sales(self):
        return sum(s.amount for s in self.sales_data)

    def get_top_seller(self):
        return max(self.sales_data, key=lambda s: s.amount)

    def add_employee(self, employee):
        self.employee_data.append(employee)

    def get_average_salary(self):
        return sum(e.salary for e in self.employee_data) / len(self.employee_data)

    def add_inventory_item(self, item):
        self.inventory_data.append(item)

    def get_low_stock_items(self):
        return [i for i in self.inventory_data if i.quantity < 10]

    def get_inventory_value(self):
        return sum(i.price * i.quantity for i in self.inventory_data)
```

**期待される回答:**

```
LCOM* 計算:
- メソッド数 (m) = 8
- フィールド数 (f) = 3
- sales_data: add_sale, get_total_sales, get_top_seller → 3メソッド
- employee_data: add_employee, get_average_salary → 2メソッド
- inventory_data: add_inventory_item, get_low_stock_items, get_inventory_value → 3メソッド

LCOM* = (8 - (1/3) * (3+2+3)) / (8-1)
       = (8 - 2.67) / 7
       = 5.33 / 7
       = 0.76

LCOM* = 0.76 → 凝集度が低い（1.0に近い）

改善: 3つのクラスに分割
```

```python
class SalesReport:
    """売上レポート（機能的凝集）"""
    def __init__(self):
        self.sales_data: list[Sale] = []

    def add_sale(self, sale: Sale) -> None:
        self.sales_data.append(sale)

    def get_total(self) -> float:
        return sum(s.amount for s in self.sales_data)

    def get_top_seller(self) -> Sale:
        return max(self.sales_data, key=lambda s: s.amount)
    # LCOM* = (3 - (1/1) * 3) / (3-1) = 0 / 2 = 0 ★完全凝集

class EmployeeReport:
    """従業員レポート（機能的凝集）"""
    def __init__(self):
        self.employee_data: list[Employee] = []

    def add_employee(self, employee: Employee) -> None:
        self.employee_data.append(employee)

    def get_average_salary(self) -> float:
        if not self.employee_data:
            return 0.0
        return sum(e.salary for e in self.employee_data) / len(self.employee_data)
    # LCOM* = (2 - (1/1) * 2) / (2-1) = 0 / 1 = 0 ★完全凝集

class InventoryReport:
    """在庫レポート（機能的凝集）"""
    def __init__(self):
        self.inventory_data: list[InventoryItem] = []

    def add_item(self, item: InventoryItem) -> None:
        self.inventory_data.append(item)

    def get_low_stock_items(self, threshold: int = 10) -> list[InventoryItem]:
        return [i for i in self.inventory_data if i.quantity < threshold]

    def get_total_value(self) -> float:
        return sum(i.price * i.quantity for i in self.inventory_data)
    # LCOM* = (3 - (1/1) * 3) / (3-1) = 0 / 2 = 0 ★完全凝集
```

---

## 9. FAQ

### Q1: 結合度ゼロは目指すべきか？

結合度ゼロは不可能であり、目指すべきでもない。モジュール間の通信がなければシステムは機能しない。目指すのは**必要最小限の、明示的な結合**。暗黙の依存（グローバル変数、隠れた副作用）を排除し、明示的なインターフェースを通じた依存に置き換えることが重要。

理想は「安定した抽象への依存」であり、具体的には以下を目指す。
- 具象クラスではなくインターフェースに依存する
- 広い公開APIではなく最小限のインターフェースに依存する
- 同期呼び出しではなく非同期イベントで通信する（適切な場面で）

### Q2: マイクロサービスにすれば自動的に低結合になるか？

ならない。分散システムでもサービス間の結合は存在する。共有データベース、同期的なAPI呼び出しチェーン、共有ライブラリによる結合は、モノリス以上に管理が困難になる場合がある（「分散モノリス」と呼ばれる）。

マイクロサービスの恩恵を受けるには、以下が必須。
- 各サービスが独自のデータストアを持つ
- サービス間の通信を非同期メッセージングにする
- 契約テスト（Contract Testing）でAPI互換性を保証する
- サーキットブレーカーで障害伝播を防ぐ

### Q3: ユーティリティクラスは凝集度が低いので作るべきではないか？

「何でも入りのUtils」は避けるべきだが、**明確なテーマを持ったユーティリティ**（例: `StringUtils`, `DateUtils`）は論理的凝集であり実用上許容される。ただし、ドメインロジックがユーティリティに漏れていないか定期的に検証する。

判断基準:
- `Utils` / `Helper` / `Manager` は名前がレッドフラグ
- `StringFormatter`, `DateParser`, `PathNormalizer` は許容範囲
- ユーティリティに入れたメソッドが特定のドメインの知識を必要とするなら、そのドメインクラスに移動すべき

### Q4: テストコードの結合度・凝集度も気にすべきか？

テストコードは本番コードとは異なる判断基準がある。テストの「3A (Arrange-Act-Assert)」パターンに従い、1テストが1つの振る舞いを検証していれば（機能的凝集）、テスト自体の凝集度は問題ない。

ただし、テストヘルパーやフィクスチャが神クラスになることは避ける。テストユーティリティも適切に分割する。

### Q5: LCOM のしきい値はいくつにすべきか？

一般的なガイドライン:
- LCOM* = 0.0 ~ 0.3: 凝集度が高い（問題なし）
- LCOM* = 0.3 ~ 0.6: 注意が必要（分割を検討）
- LCOM* = 0.6 ~ 1.0: 凝集度が低い（分割推奨）

ただし、LCOM はあくまで指標の1つ。クラスの責任が明確で変更理由が1つなら、数値が高くても問題ない場合がある。数値と設計意図を総合的に判断する。

---

## まとめ

| 指標 | 目標 | 達成手段 | 測定方法 | しきい値 |
|------|------|---------|---------|---------|
| 結合度 | 低く保つ | DI、IF、イベント、Adapter | CBO、Ca/Ce、Instability | CBO ≤ 10 |
| 凝集度 | 高く保つ | SRP、Extract Class | LCOM（凝集度メトリクス） | LCOM* ≤ 0.3 |
| バランス | 低結合+高凝集 | ドメイン駆動設計 | 変更影響範囲の分析 | -- |
| パッケージ | ドメイン基準 | 機能凝集のパッケージ構成 | 変更が1パッケージ内で完結 | -- |

| 改善シグナル | 疑うべき問題 | 確認方法 |
|------------|------------|---------|
| 1変更で多数ファイル修正 | Shotgun Surgery (高結合) | 変更履歴分析 |
| クラスが500行超 | God Class (低凝集) | 行数カウント |
| テストでモック10個以上 | 高結合 | テストコード確認 |
| `Utils` / `Helper` の肥大化 | 低凝集 | クラス名チェック |
| 循環依存の発生 | 設計レベルの問題 | 依存グラフ分析 |

---

## 次に読むべきガイド

- [デメテルの法則](./04-law-of-demeter.md) ── 結合度を下げるための具体的規則
- [SOLID原則](./01-solid.md) ── 特にSRPとDIPが結合度・凝集度に直結
- [合成 vs 継承](../03-practices-advanced/01-composition-over-inheritance.md) ── 結合度に影響する設計判断
- [コードスメル](../02-refactoring/00-code-smells.md) ── God Class、Shotgun Surgery の詳細
- [リファクタリング技法](../02-refactoring/01-refactoring-catalog.md) ── Extract Class、Move Method の手順

---

## 参考文献

1. **Larry Constantine, Edward Yourdon** 『Structured Design: Fundamentals of a Discipline of Computer Program and Systems Design』 Yourdon Press, 1979 ── 結合度・凝集度の原典
2. **Glenford J. Myers** 『Composite/Structured Design』 Van Nostrand Reinhold, 1978 ── 構造化設計の古典
3. **Robert C. Martin** 『Clean Architecture: A Craftsman's Guide to Software Structure and Design』 Prentice Hall, 2017 ── 安定依存の原則（SDP）、安定抽象の原則（SAP）
4. **Eric Evans** 『Domain-Driven Design: Tackling Complexity in the Heart of Software』 Addison-Wesley, 2003 ── Bounded Context とモジュール設計
5. **Sam Newman** 『Building Microservices: Designing Fine-Grained Systems』 O'Reilly Media, 2021 (2nd Edition) ── マイクロサービスの結合度管理
6. **Brian Henderson-Sellers** "Software Metrics" Prentice Hall, 1996 ── LCOM* メトリクスの定義
7. **J.M. Bieman, B.K. Kang** "Cohesion and Reuse in an Object-Oriented System" Proceedings of the 1995 Symposium on Software Reusability, 1995 ── 凝集度と再利用性の実証研究
8. **Martin Fowler** 『Refactoring: Improving the Design of Existing Code』 Addison-Wesley, 2018 ── Feature Envy、Shotgun Surgery、God Class のリファクタリング

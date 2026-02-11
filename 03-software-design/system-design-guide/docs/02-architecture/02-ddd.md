# ドメイン駆動設計 (DDD)

> 複雑なビジネスドメインをソフトウェアに正確に反映するための設計手法であり、集約・境界づけられたコンテキスト・ユビキタス言語を軸に、ドメインエキスパートと開発者が共通理解のもとで堅牢なモデルを構築する方法論を解説する

## この章で学ぶこと

1. **戦略的設計** — 境界づけられたコンテキスト、コンテキストマッピング、ユビキタス言語の確立
2. **戦術的設計** — エンティティ、値オブジェクト、集約、ドメインイベント、リポジトリの実装パターン
3. **集約設計の原則** — 集約ルート、トランザクション境界、整合性の保証

---

## 1. DDD の全体像

### 1.1 戦略的設計 vs 戦術的設計

```
DDD の2つの柱

【戦略的設計 (Strategic Design)】
  ─ 問題領域を分割し、チーム・システム境界を定義
  ─ 境界づけられたコンテキスト (Bounded Context)
  ─ コンテキストマップ
  ─ ユビキタス言語

【戦術的設計 (Tactical Design)】
  ─ コンテキスト内部のモデリングパターン
  ─ エンティティ / 値オブジェクト / 集約
  ─ ドメインサービス / ドメインイベント
  ─ リポジトリ / ファクトリ
```

### 1.2 境界づけられたコンテキスト

```
ECサイトのコンテキストマップ

  +------------------+     +------------------+     +------------------+
  |   注文コンテキスト  |     |  在庫コンテキスト  |     |  配送コンテキスト  |
  |                  |     |                  |     |                  |
  | Order            |     | StockItem        |     | Shipment         |
  | OrderItem        |     | Warehouse        |     | DeliveryRoute    |
  | Customer(注文者)  |     | Reservation      |     | Customer(届け先)  |
  +--------+---------+     +--------+---------+     +--------+---------+
           |                        |                        |
           | OrderPlaced            | StockReserved          |
           | (ドメインイベント)       | (ドメインイベント)       |
           +--------> [Event Bus] <-+----------------------->+

  ★ 同じ「Customer」でも各コンテキストで意味と属性が異なる
  ★ コンテキスト間はイベントで疎結合に連携
```

### 1.3 集約の構造

```
   集約 (Aggregate)
  +---------------------------------------------+
  |  [Order] ← 集約ルート (Aggregate Root)       |
  |     |                                        |
  |     +-- OrderItem (値オブジェクト or エンティティ)|
  |     +-- OrderItem                            |
  |     +-- ShippingAddress (値オブジェクト)       |
  |     +-- PaymentInfo (値オブジェクト)           |
  +---------------------------------------------+

  ルール:
  - 外部から集約内部への直接アクセス禁止
  - 全ての操作は集約ルート経由
  - 1トランザクション = 1集約の変更
  - 集約間の参照は ID のみ
```

---

## 2. 戦術パターンの実装

### 2.1 値オブジェクト (Value Object)

```python
# domain/value_objects/money.py
from dataclasses import dataclass

@dataclass(frozen=True)     # 不変 (Immutable)
class Money:
    """金額を表す値オブジェクト"""
    amount: int              # 最小単位（円）
    currency: str = "JPY"

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError(f"金額は0以上: {self.amount}")

    def add(self, other: 'Money') -> 'Money':
        if self.currency != other.currency:
            raise ValueError(f"通貨が異なります: {self.currency} vs {other.currency}")
        return Money(amount=self.amount + other.amount, currency=self.currency)

    def multiply(self, factor: int) -> 'Money':
        return Money(amount=self.amount * factor, currency=self.currency)

    # 値オブジェクトは値で等価判定（ID不要）
    # frozen=True で __eq__ と __hash__ は自動生成


# domain/value_objects/address.py
@dataclass(frozen=True)
class Address:
    """住所を表す値オブジェクト"""
    postal_code: str
    prefecture: str
    city: str
    street: str
    building: str = ""

    def __post_init__(self):
        if not self.postal_code or len(self.postal_code) != 7:
            raise ValueError(f"郵便番号は7桁: {self.postal_code}")

    @property
    def full_address(self) -> str:
        parts = [self.prefecture, self.city, self.street]
        if self.building:
            parts.append(self.building)
        return " ".join(parts)
```

### 2.2 エンティティ (Entity)

```python
# domain/entities/order.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from domain.value_objects.money import Money

@dataclass
class OrderItem:
    """注文明細エンティティ（集約内でのみ使用）"""
    id: str
    product_id: str
    product_name: str
    unit_price: Money
    quantity: int

    @property
    def subtotal(self) -> Money:
        return self.unit_price.multiply(self.quantity)

@dataclass
class Order:
    """注文集約ルート"""
    id: str
    customer_id: str          # 他の集約への参照は ID のみ
    items: List[OrderItem] = field(default_factory=list)
    status: str = "draft"
    _domain_events: List = field(default_factory=list, repr=False)

    @property
    def total_amount(self) -> Money:
        total = Money(0)
        for item in self.items:
            total = total.add(item.subtotal)
        return total

    def add_item(self, item: OrderItem) -> None:
        """注文明細追加（ビジネスルール付き）"""
        if self.status != "draft":
            raise ValueError("下書き状態でのみ明細追加可能")
        if len(self.items) >= 50:
            raise ValueError("1注文あたり最大50明細")
        # 同一商品がある場合は数量を加算
        existing = next((i for i in self.items if i.product_id == item.product_id), None)
        if existing:
            existing.quantity += item.quantity
        else:
            self.items.append(item)

    def place(self) -> None:
        """注文確定"""
        if self.status != "draft":
            raise ValueError(f"下書き状態でのみ確定可能 (現在: {self.status})")
        if not self.items:
            raise ValueError("明細が空の注文は確定できません")
        self.status = "placed"
        self._domain_events.append(OrderPlaced(
            order_id=self.id,
            customer_id=self.customer_id,
            total_amount=self.total_amount.amount,
            occurred_at=datetime.now(),
        ))

    def cancel(self) -> None:
        """注文キャンセル"""
        if self.status not in ("draft", "placed"):
            raise ValueError(f"キャンセル不可 (現在: {self.status})")
        self.status = "cancelled"
        self._domain_events.append(OrderCancelled(
            order_id=self.id, occurred_at=datetime.now()
        ))

    def collect_events(self) -> List:
        events = list(self._domain_events)
        self._domain_events.clear()
        return events
```

### 2.3 ドメインイベント

```python
# domain/events/order_events.py
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class OrderPlaced:
    """注文確定イベント"""
    order_id: str
    customer_id: str
    total_amount: int
    occurred_at: datetime

@dataclass(frozen=True)
class OrderCancelled:
    """注文キャンセルイベント"""
    order_id: str
    occurred_at: datetime
```

### 2.4 リポジトリ

```python
# domain/repositories/order_repository.py (インターフェース)
from typing import Protocol, Optional

class OrderRepository(Protocol):
    def save(self, order: Order) -> None: ...
    def find_by_id(self, order_id: str) -> Optional[Order]: ...
    def find_by_customer(self, customer_id: str) -> List[Order]: ...

# infrastructure/repositories/sqlalchemy_order_repository.py (実装)
class SQLAlchemyOrderRepository:
    def __init__(self, session):
        self._session = session

    def save(self, order: Order) -> None:
        model = self._to_model(order)
        self._session.merge(model)
        self._session.flush()

    def find_by_id(self, order_id: str) -> Optional[Order]:
        model = self._session.query(OrderModel).get(order_id)
        return self._to_entity(model) if model else None
```

### 2.5 アプリケーションサービス

```python
# application/services/order_service.py
class PlaceOrderService:
    """注文確定アプリケーションサービス"""

    def __init__(self, order_repo: OrderRepository,
                 event_publisher: EventPublisher,
                 unit_of_work: UnitOfWork):
        self._order_repo = order_repo
        self._events = event_publisher
        self._uow = unit_of_work

    def execute(self, order_id: str) -> None:
        with self._uow:
            order = self._order_repo.find_by_id(order_id)
            if not order:
                raise OrderNotFoundError(order_id)

            # ドメインロジック実行
            order.place()

            # 永続化
            self._order_repo.save(order)

            # ドメインイベント発行
            for event in order.collect_events():
                self._events.publish(event)

            self._uow.commit()
```

---

## 3. 比較表

| 概念 | エンティティ | 値オブジェクト |
|------|-----------|-------------|
| 同一性 | ID で識別 | 値で識別 |
| 可変性 | 可変 (Mutable) | 不変 (Immutable) |
| ライフサイクル | 作成・変更・削除 | 生成のみ（変更は新規生成） |
| 例 | Order, User, Product | Money, Address, Email |
| 等価判定 | id が同じなら同一 | 全属性が同じなら同一 |

| パターン | 責務 | 配置層 |
|---------|------|-------|
| エンティティ | ビジネスルール + ID による識別 | ドメイン層 |
| 値オブジェクト | 不変の値表現 | ドメイン層 |
| 集約 | トランザクション整合性の境界 | ドメイン層 |
| ドメインサービス | 複数集約にまたがるロジック | ドメイン層 |
| ドメインイベント | 集約間の非同期連携 | ドメイン層 |
| リポジトリ | 集約の永続化・取得 | インターフェース = ドメイン層、実装 = インフラ層 |
| アプリケーションサービス | ユースケースの調整 | アプリケーション層 |

---

## 4. アンチパターン

### アンチパターン 1: 貧血ドメインモデル (Anemic Domain Model)

```python
# BAD: エンティティにロジックがなく、サービスに全て集中
@dataclass
class Order:
    id: str
    status: str
    items: list
    # ← ビジネスルールが一切ない単なるデータ入れ物

class OrderService:
    def place_order(self, order):
        if order.status != "draft":
            raise ValueError("...")
        if not order.items:
            raise ValueError("...")
        order.status = "placed"
        # ← 本来 Order エンティティが持つべきロジック

# GOOD: エンティティ自身がビジネスルールを持つ
class Order:
    def place(self):
        if self.status != "draft":
            raise ValueError("下書き状態でのみ確定可能")
        if not self.items:
            raise ValueError("明細が空の注文は確定できません")
        self.status = "placed"
```

### アンチパターン 2: 集約が大きすぎる

```
BAD: 1つの集約に全てを含める
  Order (集約ルート)
    ├── Customer (全属性)
    ├── Product (全属性) x N
    ├── PaymentHistory x N
    └── ShippingLog x N
  → 更新のたびに巨大オブジェクトをロード、同時更新で競合頻発

GOOD: 集約を小さく保ち、ID で参照
  Order (集約ルート)
    ├── customer_id: str       ← ID のみ
    ├── OrderItem x N
    │     └── product_id: str  ← ID のみ
    └── shipping_address: Address (値オブジェクト)
```

---

## 5. FAQ

### Q1. DDD はいつ採用すべきか？

**A.** ドメインの複雑性が高いプロジェクトに適している。判断基準は「ビジネスルールが複雑で、CRUD では表現しきれない」こと。単純な管理画面やデータパイプラインでは過剰設計になる。ドメインエキスパートとの密なコミュニケーションが前提であり、技術者だけで進めても効果は限定的。

### Q2. 集約間のデータ整合性はどう保つ？

**A.** 結果整合性（Eventual Consistency）を基本とする。集約Aの変更でドメインイベントを発行し、集約Bがそのイベントを購読して非同期に自身を更新する。強い整合性が必要な場合は Saga パターンで補償トランザクションを実装する。「1トランザクション = 1集約」のルールを崩さないことが重要。

### Q3. ユビキタス言語とは何か？

**A.** 開発チームとドメインエキスパートが共通して使う用語体系。コード上の変数名・クラス名・メソッド名もこの言語に揃える。例えば「注文を確定する」を `placeOrder` とするか `confirmOrder` とするかは、ドメインエキスパートが自然に使う言葉に合わせる。ユビキタス言語は Bounded Context ごとに異なってよい。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 戦略的設計 | 境界づけられたコンテキストでドメインを分割。コンテキスト間はイベントで連携 |
| 戦術的設計 | エンティティ・値オブジェクト・集約・ドメインイベントで複雑さをモデリング |
| 集約設計 | 小さく保つ。1トランザクション = 1集約。集約間は ID 参照 |
| ユビキタス言語 | ドメインエキスパートとコードで同じ言葉を使う |
| 貧血モデル回避 | ビジネスロジックはエンティティに持たせ、サービスは調整役に徹する |
| 結果整合性 | 集約間はドメインイベントによる非同期連携を基本とする |

---

## 次に読むべきガイド

- [クリーンアーキテクチャ](./01-clean-architecture.md) — DDD と組み合わせるレイヤーアーキテクチャ
- [イベント駆動アーキテクチャ](./03-event-driven.md) — ドメインイベントを活用した疎結合設計
- [API設計](../../clean-code-principles/docs/03-practices-advanced/03-api-design.md) — 集約を公開する API の設計原則

---

## 参考文献

1. **Domain-Driven Design** — Eric Evans (Addison-Wesley, 2003) — DDD の原典
2. **Implementing Domain-Driven Design** — Vaughn Vernon (Addison-Wesley, 2013) — DDD の実装パターン詳細
3. **Domain-Driven Design Distilled** — Vaughn Vernon (Addison-Wesley, 2016) — DDD の簡潔な入門書
4. **Architecture Patterns with Python** — Harry Percival & Bob Gregory (O'Reilly, 2020) — Python での DDD 実践

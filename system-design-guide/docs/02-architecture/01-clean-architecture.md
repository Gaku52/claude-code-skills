# クリーンアーキテクチャ

> Robert C. Martin が提唱した依存性逆転の原則に基づくアーキテクチャパターンであり、ビジネスロジックをフレームワーク・DB・UI から独立させ、テスタブルで変更に強いシステムを構築する設計思想を解説する

## この章で学ぶこと

1. **同心円モデルと依存性ルール** — Entities、Use Cases、Interface Adapters、Frameworks の4層構造と内向きの依存方向
2. **依存性逆転の実装** — インターフェースを用いた外部依存の抽象化とDIコンテナの活用
3. **実プロジェクトへの適用** — ディレクトリ構成、レイヤー間のデータ変換、テスト戦略

---

## 1. 同心円モデル

### 1.1 4層構造

```
+---------------------------------------------------------------+
|                    Frameworks & Drivers                         |
|   (Web, DB, External API, UI, デバイス)                        |
|   +-------------------------------------------------------+   |
|   |              Interface Adapters                        |   |
|   |   (Controllers, Gateways, Presenters)                  |   |
|   |   +-----------------------------------------------+   |   |
|   |   |              Use Cases                         |   |   |
|   |   |   (アプリケーション固有のビジネスルール)         |   |   |
|   |   |   +---------------------------------------+   |   |   |
|   |   |   |           Entities                     |   |   |   |
|   |   |   |   (企業全体のビジネスルール)            |   |   |   |
|   |   |   +---------------------------------------+   |   |   |
|   |   +-----------------------------------------------+   |   |
|   +-------------------------------------------------------+   |
+---------------------------------------------------------------+

  依存性ルール: 外側 → 内側 のみ依存可能
  内側は外側の存在を知らない
```

### 1.2 依存性の方向

```
  Controller ──依存──> UseCase ──依存──> Entity
       |                   |
       |          UseCase は Repository の
       |          「インターフェース」に依存
       |                   |
       |            <<interface>>
       |           IOrderRepository
       |                   ^
       |                   |  実装
       |                   |
  PostgresOrderRepository ─+
  (Frameworks 層)

  ★ UseCase は具体的な DB 実装を知らない
  ★ DB を MongoDB に差し替えても UseCase は変更不要
```

### 1.3 データフロー

```
  HTTP Request
       |
  [Controller] --- DTO変換 ---> [UseCase] --- Entity操作 ---> [Entity]
       |                            |
       |                    [Repository Interface]
       |                            |
  [Presenter] <--- OutputDTO --- [UseCase]
       |
  HTTP Response (JSON)
```

---

## 2. 各レイヤーの実装

### 2.1 Entities (ドメインモデル)

```python
# domain/entities/order.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List

class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    CANCELLED = "cancelled"

@dataclass
class OrderItem:
    product_id: str
    name: str
    price: int          # 円単位
    quantity: int

    @property
    def subtotal(self) -> int:
        return self.price * self.quantity

@dataclass
class Order:
    """注文エンティティ: ビジネスルールを内包"""
    id: str
    user_id: str
    items: List[OrderItem]
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def total_amount(self) -> int:
        return sum(item.subtotal for item in self.items)

    def confirm(self) -> None:
        if self.status != OrderStatus.PENDING:
            raise ValueError(f"PENDING 状態でのみ確定可能 (現在: {self.status.value})")
        if not self.items:
            raise ValueError("注文アイテムが空です")
        self.status = OrderStatus.CONFIRMED

    def cancel(self) -> None:
        if self.status in (OrderStatus.SHIPPED, OrderStatus.CANCELLED):
            raise ValueError(f"出荷済み/キャンセル済みの注文は取消不可")
        self.status = OrderStatus.CANCELLED
```

### 2.2 Use Cases (アプリケーションロジック)

```python
# application/use_cases/create_order.py
from dataclasses import dataclass
from typing import List, Protocol

# --- Input/Output DTO ---
@dataclass
class CreateOrderInput:
    user_id: str
    items: List[dict]   # [{"product_id": "...", "quantity": 2}]

@dataclass
class CreateOrderOutput:
    order_id: str
    total_amount: int
    status: str

# --- Repository Interface (ポート) ---
class OrderRepository(Protocol):
    def save(self, order: Order) -> None: ...
    def find_by_id(self, order_id: str) -> Order | None: ...

class ProductRepository(Protocol):
    def find_by_id(self, product_id: str) -> Product | None: ...

class EventPublisher(Protocol):
    def publish(self, event_name: str, data: dict) -> None: ...

# --- Use Case ---
class CreateOrderUseCase:
    """注文作成ユースケース"""

    def __init__(self, order_repo: OrderRepository,
                 product_repo: ProductRepository,
                 event_publisher: EventPublisher):
        self._order_repo = order_repo
        self._product_repo = product_repo
        self._events = event_publisher

    def execute(self, input_dto: CreateOrderInput) -> CreateOrderOutput:
        # 1. 商品情報を取得して OrderItem を構築
        order_items = []
        for item_data in input_dto.items:
            product = self._product_repo.find_by_id(item_data['product_id'])
            if not product:
                raise ValueError(f"商品が見つかりません: {item_data['product_id']}")
            order_items.append(OrderItem(
                product_id=product.id,
                name=product.name,
                price=product.price,
                quantity=item_data['quantity'],
            ))

        # 2. Order エンティティ生成
        order = Order(id=generate_id(), user_id=input_dto.user_id, items=order_items)

        # 3. 永続化
        self._order_repo.save(order)

        # 4. イベント発行
        self._events.publish('order.created', {'order_id': order.id})

        # 5. Output DTO を返却
        return CreateOrderOutput(
            order_id=order.id,
            total_amount=order.total_amount,
            status=order.status.value,
        )
```

### 2.3 Interface Adapters (Controller / Repository 実装)

```python
# adapters/controllers/order_controller.py
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/orders', methods=['POST'])
def create_order():
    """HTTP リクエストを UseCase の入力に変換"""
    body = request.get_json()
    input_dto = CreateOrderInput(
        user_id=body['user_id'],
        items=body['items'],
    )

    # DI コンテナから UseCase を取得
    use_case = container.resolve(CreateOrderUseCase)
    output = use_case.execute(input_dto)

    return jsonify({
        'order_id': output.order_id,
        'total_amount': output.total_amount,
        'status': output.status,
    }), 201


# adapters/repositories/postgres_order_repository.py
from sqlalchemy.orm import Session

class PostgresOrderRepository:
    """OrderRepository インターフェースの PostgreSQL 実装"""

    def __init__(self, session: Session):
        self._session = session

    def save(self, order: Order) -> None:
        db_order = OrderModel(
            id=order.id,
            user_id=order.user_id,
            status=order.status.value,
            total_amount=order.total_amount,
            created_at=order.created_at,
        )
        for item in order.items:
            db_order.items.append(OrderItemModel(
                product_id=item.product_id,
                price=item.price,
                quantity=item.quantity,
            ))
        self._session.add(db_order)
        self._session.commit()

    def find_by_id(self, order_id: str) -> Order | None:
        db_order = self._session.query(OrderModel).get(order_id)
        if not db_order:
            return None
        return self._to_entity(db_order)

    def _to_entity(self, model: OrderModel) -> Order:
        """DB モデル → ドメインエンティティの変換"""
        return Order(
            id=model.id,
            user_id=model.user_id,
            items=[OrderItem(
                product_id=i.product_id,
                name=i.product.name,
                price=i.price,
                quantity=i.quantity,
            ) for i in model.items],
            status=OrderStatus(model.status),
            created_at=model.created_at,
        )
```

### 2.4 テスト (UseCase のユニットテスト)

```python
# tests/test_create_order.py
import pytest

class FakeOrderRepository:
    def __init__(self):
        self.saved = []
    def save(self, order):
        self.saved.append(order)
    def find_by_id(self, order_id):
        return next((o for o in self.saved if o.id == order_id), None)

class FakeProductRepository:
    def find_by_id(self, product_id):
        return Product(id=product_id, name="テスト商品", price=1000)

class FakeEventPublisher:
    def __init__(self):
        self.events = []
    def publish(self, event_name, data):
        self.events.append((event_name, data))

def test_create_order_success():
    order_repo = FakeOrderRepository()
    product_repo = FakeProductRepository()
    event_pub = FakeEventPublisher()

    use_case = CreateOrderUseCase(order_repo, product_repo, event_pub)

    result = use_case.execute(CreateOrderInput(
        user_id="user-1",
        items=[{"product_id": "prod-1", "quantity": 2}],
    ))

    assert result.total_amount == 2000
    assert result.status == "pending"
    assert len(order_repo.saved) == 1
    assert event_pub.events[0][0] == "order.created"
```

---

## 3. ディレクトリ構成

```
project/
├── domain/                      # Entities (最内層)
│   ├── entities/
│   │   ├── order.py
│   │   └── product.py
│   └── value_objects/
│       └── money.py
├── application/                 # Use Cases
│   ├── use_cases/
│   │   ├── create_order.py
│   │   └── cancel_order.py
│   └── interfaces/              # Repository Interface (Port)
│       ├── order_repository.py
│       └── event_publisher.py
├── adapters/                    # Interface Adapters
│   ├── controllers/
│   │   └── order_controller.py
│   ├── repositories/
│   │   └── postgres_order_repository.py
│   └── presenters/
│       └── order_presenter.py
├── infrastructure/              # Frameworks & Drivers
│   ├── db/
│   │   ├── models.py
│   │   └── connection.py
│   ├── web/
│   │   └── flask_app.py
│   └── messaging/
│       └── kafka_publisher.py
└── tests/
    ├── unit/                    # Entity + UseCase テスト
    ├── integration/             # Repository テスト
    └── e2e/                     # API テスト
```

---

## 4. レイヤー比較表

| レイヤー | 責務 | 依存先 | 変更頻度 | テスト方法 |
|---------|------|--------|---------|-----------|
| Entities | ビジネスルール | なし（自己完結） | 低 | ユニットテスト |
| Use Cases | アプリケーションロジック | Entities + Interface | 中 | ユニットテスト (Fake) |
| Adapters | 変換・接続 | Use Cases + Entities | 中〜高 | 統合テスト |
| Frameworks | 技術詳細 | 全レイヤー | 高 | E2Eテスト |

| アーキテクチャ | 中心となる考え | 特徴 |
|--------------|--------------|------|
| クリーンアーキテクチャ | 依存性の方向制御 | 4層同心円、フレームワーク非依存 |
| ヘキサゴナル (Ports & Adapters) | ポートとアダプター | 入出力ポートの明示的定義 |
| オニオンアーキテクチャ | ドメインモデル中心 | クリーンアーキテクチャとほぼ同じ |
| 従来型 3層 (MVC) | 画面・ロジック・データ分離 | フレームワーク依存になりやすい |

---

## 5. アンチパターン

### アンチパターン 1: Entity がフレームワークに依存

```python
# BAD: Entity が SQLAlchemy に依存
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

class Order(Base):           # ← フレームワーク依存
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True)
    # Entity とDBモデルが混在

# GOOD: Entity は純粋な Python クラス
@dataclass
class Order:
    id: str
    user_id: str
    items: List[OrderItem]
    # DB の知識は一切持たない
```

### アンチパターン 2: UseCase が HTTP リクエストに直接依存

```python
# BAD: UseCase が Flask の request に依存
from flask import request

class CreateOrderUseCase:
    def execute(self):
        user_id = request.json['user_id']  # ← Web フレームワーク依存

# GOOD: DTO を介して完全に分離
class CreateOrderUseCase:
    def execute(self, input_dto: CreateOrderInput):
        user_id = input_dto.user_id        # ← 純粋なデータクラス
```

---

## 6. FAQ

### Q1. クリーンアーキテクチャは小規模プロジェクトでも必要か？

**A.** 小規模・短期プロジェクトでは過剰設計になることが多い。レイヤー数を減らした「簡易版」を検討する。例えば Domain + Application + Infrastructure の3層に簡略化し、プロジェクトの成長に合わせて分離度を高める段階的アプローチが現実的。CRUD 中心の管理画面なら従来型 MVC で十分な場合もある。

### Q2. DTO とエンティティの変換が面倒だが省略できるか？

**A.** レイヤー間のデータ変換はクリーンアーキテクチャの本質的なコストである。省略するとレイヤー間の結合度が上がり、変更時の影響範囲が拡大する。ただし、dataclass の `asdict()` や AutoMapper 的なライブラリで変換コードの記述量は削減できる。変換の価値は「テスト容易性」と「変更容易性」で回収される。

### Q3. DI コンテナは必須か？

**A.** 必須ではないが推奨。小規模なら手動 DI（コンストラクタ注入）で十分。プロジェクト規模が大きくなったら `dependency-injector` (Python) や `tsyringe` (TypeScript) などの DI コンテナで依存関係を一元管理すると、設定変更や環境切り替え（テスト用 Fake / 本番用実装）が容易になる。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 依存性ルール | 外側から内側にのみ依存。内側は外側を知らない |
| Entities | ビジネスルールの核。フレームワーク非依存 |
| Use Cases | アプリケーションロジック。Repository Interface に依存 |
| Adapters | DTO変換、HTTP ↔ UseCase、DB ↔ Entity の橋渡し |
| テスト戦略 | 内側のレイヤーほどユニットテストが容易 |
| トレードオフ | 初期コスト vs 長期的な保守性・テスト容易性 |

---

## 次に読むべきガイド

- [DDD](./02-ddd.md) — 集約と境界づけられたコンテキストによるドメインモデリング
- [イベント駆動アーキテクチャ](./03-event-driven.md) — Pub/Sub による疎結合な連携
- [テスト原則](../../clean-code-principles/docs/01-practices/04-testing-principles.md) — AAA・FIRST によるテスト設計

---

## 参考文献

1. **Clean Architecture** — Robert C. Martin (Prentice Hall, 2017) — クリーンアーキテクチャの原典
2. **Hexagonal Architecture (Ports and Adapters)** — Alistair Cockburn — https://alistair.cockburn.us/hexagonal-architecture/
3. **Architecture Patterns with Python** — Harry Percival & Bob Gregory (O'Reilly, 2020) — Python でのクリーンアーキテクチャ実践

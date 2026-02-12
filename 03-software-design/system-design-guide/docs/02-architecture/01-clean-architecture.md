# クリーンアーキテクチャ

> Robert C. Martin が提唱した依存性逆転の原則に基づくアーキテクチャパターンであり、ビジネスロジックをフレームワーク・DB・UI から独立させ、テスタブルで変更に強いシステムを構築する設計思想を解説する

---

## この章で学ぶこと

1. **同心円モデルと依存性ルール** — Entities、Use Cases、Interface Adapters、Frameworks の4層構造と内向きの依存方向を理解する
2. **依存性逆転の実装** — インターフェースを用いた外部依存の抽象化と DI コンテナの活用方法を習得する
3. **実プロジェクトへの適用** — ディレクトリ構成、レイヤー間のデータ変換、テスト戦略を実装できるようになる
4. **類似アーキテクチャとの比較** — ヘキサゴナル、オニオン、従来型MVC との違いと使い分けを判断できるようになる
5. **段階的導入と現実的なトレードオフ** — プロジェクト規模に応じた適用戦略を策定できるようになる

---

## 前提知識

このガイドを読む前に、以下の知識を持っていることが望ましい。

| トピック | 内容 | 参照リンク |
|---------|------|-----------|
| SOLID 原則 | 特に依存性逆転の原則 (DIP)、単一責任の原則 (SRP) | [SOLID 原則](../../clean-code-principles/docs/00-principles/01-solid.md) |
| デザインパターン基礎 | Strategy、Observer、Repository パターン | [デザインパターン](../../design-patterns-guide/docs/00-creational/) |
| Python 基礎 | dataclass、Protocol、型ヒントの理解 | - |
| テストの基礎 | ユニットテスト、モック/スタブの概念 | [テスト原則](../../clean-code-principles/docs/01-practices/04-testing-principles.md) |

---

## 1. クリーンアーキテクチャの背景と哲学

### 1.1 なぜクリーンアーキテクチャが生まれたか

ソフトウェア開発の歴史を振り返ると、多くのプロジェクトが「フレームワーク依存の泥沼」に陥ってきた。Rails アプリケーションのモデルが ActiveRecord に密結合してテストできない、Spring Boot のサービスクラスが HTTP リクエストオブジェクトに依存してバッチ処理から呼べない、Django のビューに全てのビジネスロジックが書かれてリファクタリングが不可能 — これらは全て **ビジネスロジックが外部技術詳細に依存している** という根本原因に起因する。

Robert C. Martin（Uncle Bob）は2012年のブログ記事「The Clean Architecture」で、それまでに提案された複数のアーキテクチャパターンを統合する形で「クリーンアーキテクチャ」を提唱した。

```
クリーンアーキテクチャの系譜

  1979  MVC (Trygve Reenskaug)
         ↓ UI分離の概念
  1992  BCE (Ivar Jacobson)
         ↓ Boundary-Control-Entity
  2005  Hexagonal Architecture (Alistair Cockburn)
         ↓ Ports & Adapters
  2008  Onion Architecture (Jeffrey Palermo)
         ↓ ドメイン中心の同心円
  2012  Clean Architecture (Robert C. Martin)
         ↓ 上記を統合し体系化
  2017  書籍「Clean Architecture」出版
```

### 1.2 根底にある原則

クリーンアーキテクチャの核心は以下の3つの原則に集約される。

```
┌─────────────────────────────────────────────────────────┐
│                 3つの核心原則                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 依存性ルール (Dependency Rule)                       │
│     → 外側から内側にのみ依存可能                          │
│     → 内側のコードは外側の存在を一切知らない               │
│                                                         │
│  2. 抽象化の壁 (Abstraction Boundary)                    │
│     → レイヤー間はインターフェース（抽象）で通信            │
│     → 具体的な実装詳細は注入される                        │
│                                                         │
│  3. フレームワーク非依存 (Framework Independence)         │
│     → ビジネスロジックはフレームワークの存在を知らない      │
│     → フレームワークはプラグインとして扱う                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**WHY: なぜこの3原則が重要なのか？**

ソフトウェアの寿命は通常10年以上に及ぶが、Web フレームワーク、データベース、メッセージングシステムなどの外部技術は数年で世代交代する。ビジネスロジックが外部技術に依存していると、技術の入れ替えのたびにビジネスロジックまで書き直す必要が生じる。依存性を内向きに限定することで、**ビジネスロジックの安定性**と**外部技術の可換性**の両方を実現できる。

### 1.3 クリーンアーキテクチャが解決する問題

従来型アーキテクチャで発生する典型的な問題と、クリーンアーキテクチャによる解決策を対比する。

```
問題1: フレームワーク・ロックイン
  従来型: Django Model にビジネスロジックを書く
         → Django から FastAPI に移行できない（ロジックが Django に依存）
  Clean:  Entity は純粋な Python クラス
         → フレームワークを自由に切り替え可能

問題2: テスト困難
  従来型: Service クラスが直接 PostgreSQL に接続
         → テスト実行に DB が必要（遅い、セットアップが面倒）
  Clean:  UseCase は Repository Interface に依存
         → Fake 実装を注入して DB なしでテスト可能（高速）

問題3: 変更の影響範囲が広い
  従来型: DB スキーマ変更 → API レスポンス形式も変わる
         → フロントエンドも修正が必要
  Clean:  DB スキーマ変更 → Repository 実装のみ変更
         → UseCase / Entity / API は影響を受けない

問題4: ビジネスロジックの分散
  従来型: バリデーションが Controller / Service / Model に分散
         → 「このルールはどこに書かれている？」が不明確
  Clean:  ビジネスルールは Entity に集約
         → 単一の場所で管理
```

---

## 2. 同心円モデル

### 2.1 4層構造

```
+---------------------------------------------------------------+
|                    Frameworks & Drivers                         |
|   (Web Framework, DB Driver, External API, UI, Device)         |
|   +-------------------------------------------------------+   |
|   |              Interface Adapters                        |   |
|   |   (Controllers, Gateways, Presenters, ViewModels)      |   |
|   |   +-----------------------------------------------+   |   |
|   |   |              Use Cases                         |   |   |
|   |   |   (Application-Specific Business Rules)        |   |   |
|   |   |   +---------------------------------------+   |   |   |
|   |   |   |           Entities                     |   |   |   |
|   |   |   |   (Enterprise Business Rules)          |   |   |   |
|   |   |   +---------------------------------------+   |   |   |
|   |   +-----------------------------------------------+   |   |
|   +-------------------------------------------------------+   |
+---------------------------------------------------------------+

  依存性ルール: 外側 → 内側 のみ依存可能
  内側は外側の存在を知らない
```

### 2.2 各層の詳細解説

#### Entities（企業全体のビジネスルール）

最内層に位置し、**企業全体で共有されるビジネスルール**を表現する。特定のアプリケーションに依存せず、同じ企業の別システムでも再利用可能な概念をモデル化する。

```
Entities 層の特性:
  ┌────────────────────────────────────────────┐
  │  ・フレームワークへの依存: ゼロ               │
  │  ・外部ライブラリへの依存: 最小限             │
  │  ・変更頻度: 最も低い                        │
  │  ・テスト: 最も容易（純粋なユニットテスト）    │
  │  ・具体例: Order, User, Product, Invoice     │
  │  ・含まれるもの:                              │
  │    - エンティティ                             │
  │    - 値オブジェクト                           │
  │    - ドメインイベント                         │
  │    - ビジネスルール（バリデーション含む）       │
  └────────────────────────────────────────────┘
```

#### Use Cases（アプリケーション固有のビジネスルール）

**特定のアプリケーションに固有のビジネスルール**を実装する。「ユーザーが注文を作成する」「管理者が注文をキャンセルする」といった具体的なユースケースを表現する。

```
Use Cases 層の特性:
  ┌────────────────────────────────────────────┐
  │  ・依存先: Entities のみ + Port Interface    │
  │  ・変更頻度: 中程度                          │
  │  ・テスト: Fake/Stub で容易                  │
  │  ・具体例: CreateOrder, CancelOrder          │
  │  ・含まれるもの:                              │
  │    - ユースケースクラス                       │
  │    - Input/Output DTO                       │
  │    - Port（リポジトリインターフェース等）      │
  │    - アプリケーション例外                     │
  └────────────────────────────────────────────┘
```

#### Interface Adapters（変換層）

外部世界と内部世界の**データ形式を変換**する層。HTTP リクエストを UseCase の入力DTOに変換し、UseCase の出力DTOをHTTPレスポンスに変換する。データベースのレコードをエンティティに変換し、エンティティをデータベースのレコードに変換する。

```
Interface Adapters 層の特性:
  ┌────────────────────────────────────────────┐
  │  ・依存先: Use Cases + Entities             │
  │  ・変更頻度: 中〜高                          │
  │  ・テスト: 統合テスト                        │
  │  ・具体例:                                   │
  │    - Controller（HTTP → UseCase Input）      │
  │    - Presenter（UseCase Output → HTTP）      │
  │    - Repository 実装（Entity ↔ DB Record）   │
  │    - Gateway（Entity ↔ 外部API）             │
  └────────────────────────────────────────────┘
```

#### Frameworks & Drivers（外部技術詳細）

最外層。Web フレームワーク、データベースドライバ、外部API クライアントなどの**具体的な技術実装**がここに属する。

```
Frameworks & Drivers 層の特性:
  ┌────────────────────────────────────────────┐
  │  ・変更頻度: 最も高い                        │
  │  ・テスト: E2E テスト                        │
  │  ・具体例:                                   │
  │    - Flask / FastAPI / Django               │
  │    - PostgreSQL / MySQL / MongoDB           │
  │    - Redis / RabbitMQ / Kafka               │
  │    - AWS SDK / GCP Client                   │
  │  ・役割: 技術詳細のプラグイン                 │
  └────────────────────────────────────────────┘
```

### 2.3 依存性の方向と依存性逆転

```
  Controller ──依存──> UseCase ──依存──> Entity
       |                   |
       |          UseCase は Repository の
       |          「インターフェース」に依存
       |                   |
       |            <<interface>>
       |           IOrderRepository
       |                   ^
       |                   |  実装 (implements)
       |                   |
  PostgresOrderRepository ─+
  (Frameworks 層)

  ★ UseCase は具体的な DB 実装を知らない
  ★ DB を MongoDB に差し替えても UseCase は変更不要
  ★ 依存性逆転 (DIP) により、制御の流れと依存の方向が逆転する
```

**依存性逆転の内部メカニズム:**

通常のプログラムでは、「呼び出し側が呼び出される側に依存する」のが自然だ。UseCase が PostgreSQL にデータを保存したい場合、素朴に書けば UseCase が PostgresOrderRepository に直接依存する。しかしこれでは UseCase が外側のレイヤーに依存してしまい、依存性ルールに違反する。

依存性逆転は「間にインターフェースを挟む」ことでこの問題を解決する。UseCase は「何かを保存できるもの」という抽象（Protocol/Interface）に依存し、PostgresOrderRepository がそのインターフェースを実装する。制御の流れ（UseCase → PostgreSQL）と依存の方向（PostgresOrderRepository → Interface ← UseCase）が逆転する。

```
制御の流れ:
  UseCase ────呼び出し────> PostgresOrderRepository ──────> PostgreSQL
  （内側）                 （外側）

依存の方向（依存性逆転後）:
  UseCase ──依存──> IOrderRepository <──実装── PostgresOrderRepository
  （内側）          （内側で定義）               （外側）

  ★ 制御の流れと依存の方向が逆転している
  ★ UseCase は内側で定義されたインターフェースにのみ依存
```

### 2.4 データフロー

```
HTTP Request (JSON)
     │
     ▼
[Controller]
     │  (1) JSON → CreateOrderInput (DTO変換)
     ▼
[CreateOrderUseCase]
     │  (2) ビジネスロジック実行
     │  (3) Repository Interface 経由でデータ操作
     ▼
[Repository 実装]
     │  (4) Entity ↔ DB Model 変換
     ▼
[Database]

     ── 戻り ──

[CreateOrderUseCase]
     │  (5) CreateOrderOutput (Output DTO) 生成
     ▼
[Presenter / Controller]
     │  (6) Output DTO → JSON 変換
     ▼
HTTP Response (JSON)
```

### 2.5 境界を越えるデータ

**内側のレイヤーが外側のデータ構造を知ってはならない** という原則は、レイヤー間を通過するデータにも適用される。各境界ではデータ変換が必要になる。

```
外部世界           Controller       UseCase          Entity
                  (Adapter層)      (Application層)   (Domain層)

JSON Request  →  RequestDTO    →  InputDTO       →  Entity 操作
                  (HTTP固有)       (アプリ固有)       (ドメイン)

JSON Response ←  ResponseDTO   ←  OutputDTO      ←  Entity の状態
                  (HTTP固有)       (アプリ固有)       (ドメイン)

各レイヤーは自分に最適なデータ構造を持つ
変換コストはテスタビリティと変更容易性で回収される
```

---

## 3. 各レイヤーの実装

### 3.1 Entities（ドメインモデル）

```python
# domain/entities/order.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List

class OrderStatus(Enum):
    """注文ステータス: 状態遷移のルールはエンティティが管理"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

@dataclass(frozen=True)
class OrderItem:
    """注文明細: 値オブジェクトとして不変"""
    product_id: str
    name: str
    price: int          # 円単位（最小通貨単位で扱い浮動小数点を回避）
    quantity: int

    def __post_init__(self):
        if self.price < 0:
            raise ValueError(f"価格は0以上: {self.price}")
        if self.quantity < 1:
            raise ValueError(f"数量は1以上: {self.quantity}")

    @property
    def subtotal(self) -> int:
        return self.price * self.quantity

@dataclass
class Order:
    """
    注文エンティティ: ビジネスルールを内包する集約ルート

    設計方針:
    - 全てのビジネスルールはこのクラス内に閉じる
    - 外部フレームワーク（DB, Web）への依存ゼロ
    - 状態遷移のバリデーションを自身で管理
    """
    id: str
    user_id: str
    items: List[OrderItem]
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # --- ビジネスルール ---

    @property
    def total_amount(self) -> int:
        """合計金額を計算"""
        return sum(item.subtotal for item in self.items)

    @property
    def item_count(self) -> int:
        """明細数を返す"""
        return len(self.items)

    def confirm(self) -> None:
        """注文を確定する"""
        if self.status != OrderStatus.PENDING:
            raise ValueError(
                f"PENDING 状態でのみ確定可能（現在: {self.status.value}）"
            )
        if not self.items:
            raise ValueError("注文アイテムが空です")
        if self.total_amount <= 0:
            raise ValueError("合計金額が0以下です")
        self.status = OrderStatus.CONFIRMED
        self.updated_at = datetime.now()

    def ship(self) -> None:
        """注文を出荷する"""
        if self.status != OrderStatus.CONFIRMED:
            raise ValueError(
                f"CONFIRMED 状態でのみ出荷可能（現在: {self.status.value}）"
            )
        self.status = OrderStatus.SHIPPED
        self.updated_at = datetime.now()

    def deliver(self) -> None:
        """注文を配達完了にする"""
        if self.status != OrderStatus.SHIPPED:
            raise ValueError(
                f"SHIPPED 状態でのみ配達完了可能（現在: {self.status.value}）"
            )
        self.status = OrderStatus.DELIVERED
        self.updated_at = datetime.now()

    def cancel(self) -> None:
        """注文をキャンセルする"""
        if self.status in (OrderStatus.SHIPPED, OrderStatus.DELIVERED,
                           OrderStatus.CANCELLED):
            raise ValueError(
                f"出荷済み/配達済み/キャンセル済みの注文は取消不可"
            )
        self.status = OrderStatus.CANCELLED
        self.updated_at = datetime.now()

    # --- 状態遷移図 ---
    # PENDING → CONFIRMED → SHIPPED → DELIVERED
    #    ↓          ↓
    # CANCELLED  CANCELLED
```

### 3.2 値オブジェクトの実装

```python
# domain/value_objects/money.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Money:
    """
    金額を表す値オブジェクト

    WHY frozen=True?
    - 値オブジェクトは不変であるべき（同じ「1000円」は常に同じ）
    - 不変にすることで、意図しない状態変更を防ぐ
    - ハッシュ可能になるため、dict のキーや set の要素として使える
    """
    amount: int          # 最小通貨単位（日本円なら円）
    currency: str = "JPY"

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError(f"金額は0以上: {self.amount}")
        if not self.currency:
            raise ValueError("通貨コードは必須")

    def add(self, other: 'Money') -> 'Money':
        """加算: 同一通貨のみ許可"""
        self._assert_same_currency(other)
        return Money(amount=self.amount + other.amount, currency=self.currency)

    def subtract(self, other: 'Money') -> 'Money':
        """減算: 同一通貨のみ許可"""
        self._assert_same_currency(other)
        if self.amount < other.amount:
            raise ValueError("残高不足")
        return Money(amount=self.amount - other.amount, currency=self.currency)

    def multiply(self, factor: int) -> 'Money':
        """乗算"""
        return Money(amount=self.amount * factor, currency=self.currency)

    def _assert_same_currency(self, other: 'Money') -> None:
        if self.currency != other.currency:
            raise ValueError(
                f"通貨が異なります: {self.currency} vs {other.currency}"
            )

    def __str__(self) -> str:
        if self.currency == "JPY":
            return f"¥{self.amount:,}"
        return f"{self.amount} {self.currency}"


# domain/value_objects/email.py
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class Email:
    """メールアドレスを表す値オブジェクト"""
    value: str

    def __post_init__(self):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, self.value):
            raise ValueError(f"無効なメールアドレス: {self.value}")

    def domain(self) -> str:
        return self.value.split('@')[1]

    def __str__(self) -> str:
        return self.value
```

### 3.3 Use Cases（アプリケーションロジック）

```python
# application/use_cases/create_order.py
from dataclasses import dataclass
from typing import List, Protocol, Optional

# ========================================
# Input/Output DTO
# ========================================

@dataclass(frozen=True)
class OrderItemInput:
    """注文明細の入力DTO"""
    product_id: str
    quantity: int

@dataclass(frozen=True)
class CreateOrderInput:
    """注文作成の入力DTO"""
    user_id: str
    items: List[OrderItemInput]

@dataclass(frozen=True)
class CreateOrderOutput:
    """注文作成の出力DTO"""
    order_id: str
    total_amount: int
    item_count: int
    status: str

# ========================================
# Port (リポジトリインターフェース)
# ========================================

class OrderRepository(Protocol):
    """注文リポジトリのポート"""
    def save(self, order: 'Order') -> None: ...
    def find_by_id(self, order_id: str) -> Optional['Order']: ...
    def find_by_user_id(self, user_id: str) -> List['Order']: ...

class ProductRepository(Protocol):
    """商品リポジトリのポート"""
    def find_by_id(self, product_id: str) -> Optional['Product']: ...

class EventPublisher(Protocol):
    """イベント発行のポート"""
    def publish(self, event_name: str, data: dict) -> None: ...

class IdGenerator(Protocol):
    """ID生成のポート"""
    def generate(self) -> str: ...

# ========================================
# Use Case
# ========================================

class CreateOrderUseCase:
    """
    注文作成ユースケース

    責務:
    - 入力の検証（アプリケーションレベル）
    - エンティティの生成・操作の調整
    - リポジトリへの永続化指示
    - イベントの発行

    注意: ビジネスルール自体は Entity が持つ。
    UseCase はオーケストレーション（調整）に徹する。
    """

    def __init__(
        self,
        order_repo: OrderRepository,
        product_repo: ProductRepository,
        event_publisher: EventPublisher,
        id_generator: IdGenerator,
    ):
        self._order_repo = order_repo
        self._product_repo = product_repo
        self._events = event_publisher
        self._id_gen = id_generator

    def execute(self, input_dto: CreateOrderInput) -> CreateOrderOutput:
        # 1. 入力検証（アプリケーションレベル）
        if not input_dto.items:
            raise ValueError("注文アイテムが空です")

        # 2. 商品情報を取得して OrderItem を構築
        order_items = []
        for item_input in input_dto.items:
            product = self._product_repo.find_by_id(item_input.product_id)
            if not product:
                raise ProductNotFoundError(item_input.product_id)
            order_items.append(OrderItem(
                product_id=product.id,
                name=product.name,
                price=product.price,
                quantity=item_input.quantity,
            ))

        # 3. Order エンティティ生成
        order = Order(
            id=self._id_gen.generate(),
            user_id=input_dto.user_id,
            items=order_items,
        )

        # 4. 永続化
        self._order_repo.save(order)

        # 5. イベント発行
        self._events.publish('order.created', {
            'order_id': order.id,
            'user_id': order.user_id,
            'total_amount': order.total_amount,
        })

        # 6. Output DTO を返却
        return CreateOrderOutput(
            order_id=order.id,
            total_amount=order.total_amount,
            item_count=order.item_count,
            status=order.status.value,
        )


# application/use_cases/cancel_order.py

@dataclass(frozen=True)
class CancelOrderInput:
    order_id: str
    user_id: str       # 本人確認用
    reason: str = ""

@dataclass(frozen=True)
class CancelOrderOutput:
    order_id: str
    status: str
    cancelled_at: str

class CancelOrderUseCase:
    """注文キャンセルユースケース"""

    def __init__(
        self,
        order_repo: OrderRepository,
        event_publisher: EventPublisher,
    ):
        self._order_repo = order_repo
        self._events = event_publisher

    def execute(self, input_dto: CancelOrderInput) -> CancelOrderOutput:
        # 1. 注文を取得
        order = self._order_repo.find_by_id(input_dto.order_id)
        if not order:
            raise OrderNotFoundError(input_dto.order_id)

        # 2. 本人確認
        if order.user_id != input_dto.user_id:
            raise PermissionDeniedError("他者の注文はキャンセルできません")

        # 3. キャンセル実行（ビジネスルールは Entity が検証）
        order.cancel()

        # 4. 永続化
        self._order_repo.save(order)

        # 5. イベント発行
        self._events.publish('order.cancelled', {
            'order_id': order.id,
            'reason': input_dto.reason,
        })

        return CancelOrderOutput(
            order_id=order.id,
            status=order.status.value,
            cancelled_at=order.updated_at.isoformat(),
        )


# application/exceptions.py

class ApplicationError(Exception):
    """アプリケーション層の基底例外"""
    pass

class OrderNotFoundError(ApplicationError):
    def __init__(self, order_id: str):
        super().__init__(f"注文が見つかりません: {order_id}")
        self.order_id = order_id

class ProductNotFoundError(ApplicationError):
    def __init__(self, product_id: str):
        super().__init__(f"商品が見つかりません: {product_id}")
        self.product_id = product_id

class PermissionDeniedError(ApplicationError):
    pass
```

### 3.4 Interface Adapters（Controller / Repository 実装）

```python
# adapters/controllers/order_controller.py
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/orders', methods=['POST'])
def create_order():
    """
    HTTP リクエストを UseCase の入力に変換し、
    UseCase の出力を HTTP レスポンスに変換する。

    Controller の責務:
    - HTTPリクエストのパース
    - Input DTO への変換
    - UseCase の呼び出し
    - Output DTO から HTTP レスポンスへの変換
    - エラーハンドリング（HTTP ステータスコードへの変換）
    """
    try:
        body = request.get_json()

        # HTTP → Input DTO 変換
        input_dto = CreateOrderInput(
            user_id=body['user_id'],
            items=[
                OrderItemInput(
                    product_id=item['product_id'],
                    quantity=item['quantity'],
                )
                for item in body['items']
            ],
        )

        # DI コンテナから UseCase を取得して実行
        use_case = container.resolve(CreateOrderUseCase)
        output = use_case.execute(input_dto)

        # Output DTO → HTTP レスポンス変換
        return jsonify({
            'order_id': output.order_id,
            'total_amount': output.total_amount,
            'item_count': output.item_count,
            'status': output.status,
        }), 201

    except ProductNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except PermissionDeniedError as e:
        return jsonify({'error': str(e)}), 403


@app.route('/orders/<order_id>', methods=['DELETE'])
def cancel_order(order_id: str):
    """注文キャンセルエンドポイント"""
    try:
        body = request.get_json()
        input_dto = CancelOrderInput(
            order_id=order_id,
            user_id=body['user_id'],
            reason=body.get('reason', ''),
        )

        use_case = container.resolve(CancelOrderUseCase)
        output = use_case.execute(input_dto)

        return jsonify({
            'order_id': output.order_id,
            'status': output.status,
            'cancelled_at': output.cancelled_at,
        }), 200

    except OrderNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


# adapters/repositories/postgres_order_repository.py
from sqlalchemy.orm import Session

class PostgresOrderRepository:
    """
    OrderRepository インターフェースの PostgreSQL 実装

    責務:
    - ドメインエンティティ ↔ DB モデルの変換
    - SQLAlchemy を用いた永続化操作

    重要: このクラスは Infrastructure 層に属するが、
    OrderRepository インターフェース（Application 層）を実装する
    """

    def __init__(self, session: Session):
        self._session = session

    def save(self, order: Order) -> None:
        """エンティティをDBモデルに変換して保存"""
        db_order = self._to_model(order)
        self._session.merge(db_order)
        self._session.commit()

    def find_by_id(self, order_id: str) -> Order | None:
        """IDでDBモデルを検索し、エンティティに変換して返す"""
        db_order = self._session.query(OrderModel).get(order_id)
        if not db_order:
            return None
        return self._to_entity(db_order)

    def find_by_user_id(self, user_id: str) -> list[Order]:
        """ユーザーIDで注文一覧を取得"""
        db_orders = (
            self._session.query(OrderModel)
            .filter(OrderModel.user_id == user_id)
            .order_by(OrderModel.created_at.desc())
            .all()
        )
        return [self._to_entity(o) for o in db_orders]

    def _to_model(self, order: Order) -> 'OrderModel':
        """ドメインエンティティ → DB モデル"""
        db_order = OrderModel(
            id=order.id,
            user_id=order.user_id,
            status=order.status.value,
            total_amount=order.total_amount,
            created_at=order.created_at,
            updated_at=order.updated_at,
        )
        db_order.items = [
            OrderItemModel(
                product_id=item.product_id,
                name=item.name,
                price=item.price,
                quantity=item.quantity,
            )
            for item in order.items
        ]
        return db_order

    def _to_entity(self, model: 'OrderModel') -> Order:
        """DB モデル → ドメインエンティティ"""
        return Order(
            id=model.id,
            user_id=model.user_id,
            items=[
                OrderItem(
                    product_id=i.product_id,
                    name=i.name,
                    price=i.price,
                    quantity=i.quantity,
                )
                for i in model.items
            ],
            status=OrderStatus(model.status),
            created_at=model.created_at,
            updated_at=model.updated_at,
        )


# adapters/repositories/inmemory_order_repository.py

class InMemoryOrderRepository:
    """
    テスト用のインメモリリポジトリ実装

    同じ OrderRepository インターフェースを実装するため、
    UseCase のテストで PostgreSQL の代わりに使える。
    これが依存性逆転の真価。
    """

    def __init__(self):
        self._store: dict[str, Order] = {}

    def save(self, order: Order) -> None:
        self._store[order.id] = order

    def find_by_id(self, order_id: str) -> Order | None:
        return self._store.get(order_id)

    def find_by_user_id(self, user_id: str) -> list[Order]:
        return [
            o for o in self._store.values()
            if o.user_id == user_id
        ]
```

### 3.5 DI コンテナの構成

```python
# infrastructure/container.py
"""
DI コンテナ: 依存関係の組み立てを一箇所に集約する

WHY DI コンテナ?
- UseCase が「何を使うか」は UseCase 自身が定義（Protocol）
- 「何で実装するか」は DI コンテナが決定
- テスト時は Fake 実装を注入、本番は本物を注入
"""

from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    """アプリケーション全体の依存関係定義"""

    # --- 設定 ---
    config = providers.Configuration()

    # --- インフラストラクチャ ---
    db_session = providers.Singleton(
        create_session,
        database_url=config.database_url,
    )

    # --- リポジトリ ---
    order_repository = providers.Factory(
        PostgresOrderRepository,
        session=db_session,
    )

    product_repository = providers.Factory(
        PostgresProductRepository,
        session=db_session,
    )

    # --- イベント ---
    event_publisher = providers.Singleton(
        KafkaEventPublisher,
        bootstrap_servers=config.kafka_servers,
    )

    # --- ID生成 ---
    id_generator = providers.Singleton(UUIDGenerator)

    # --- ユースケース ---
    create_order_use_case = providers.Factory(
        CreateOrderUseCase,
        order_repo=order_repository,
        product_repo=product_repository,
        event_publisher=event_publisher,
        id_generator=id_generator,
    )

    cancel_order_use_case = providers.Factory(
        CancelOrderUseCase,
        order_repo=order_repository,
        event_publisher=event_publisher,
    )


# テスト用コンテナ
class TestContainer(containers.DeclarativeContainer):
    """テスト用: 全ての外部依存を Fake に差し替え"""

    order_repository = providers.Singleton(InMemoryOrderRepository)
    product_repository = providers.Singleton(FakeProductRepository)
    event_publisher = providers.Singleton(FakeEventPublisher)
    id_generator = providers.Singleton(SequentialIdGenerator)

    create_order_use_case = providers.Factory(
        CreateOrderUseCase,
        order_repo=order_repository,
        product_repo=product_repository,
        event_publisher=event_publisher,
        id_generator=id_generator,
    )
```

### 3.6 テスト（UseCase のユニットテスト）

```python
# tests/unit/test_create_order.py
import pytest

# === Fake 実装 ===

class FakeOrderRepository:
    """テスト用: メモリ上で注文を管理"""
    def __init__(self):
        self.saved: list[Order] = []

    def save(self, order: Order) -> None:
        self.saved.append(order)

    def find_by_id(self, order_id: str) -> Order | None:
        return next((o for o in self.saved if o.id == order_id), None)

    def find_by_user_id(self, user_id: str) -> list[Order]:
        return [o for o in self.saved if o.user_id == user_id]

class FakeProductRepository:
    """テスト用: 固定商品を返す"""
    def __init__(self, products: dict[str, 'Product'] | None = None):
        self._products = products or {
            "prod-1": Product(id="prod-1", name="テスト商品A", price=1000),
            "prod-2": Product(id="prod-2", name="テスト商品B", price=2500),
        }

    def find_by_id(self, product_id: str) -> 'Product | None':
        return self._products.get(product_id)

class FakeEventPublisher:
    """テスト用: イベントを記録"""
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def publish(self, event_name: str, data: dict) -> None:
        self.events.append((event_name, data))

class SequentialIdGenerator:
    """テスト用: 予測可能なID生成"""
    def __init__(self, prefix: str = "order"):
        self._counter = 0
        self._prefix = prefix

    def generate(self) -> str:
        self._counter += 1
        return f"{self._prefix}-{self._counter}"


# === テストケース ===

@pytest.fixture
def dependencies():
    """テスト用依存関係を構築"""
    return {
        'order_repo': FakeOrderRepository(),
        'product_repo': FakeProductRepository(),
        'event_pub': FakeEventPublisher(),
        'id_gen': SequentialIdGenerator(),
    }

@pytest.fixture
def use_case(dependencies):
    """テスト対象のUseCaseを構築"""
    return CreateOrderUseCase(
        order_repo=dependencies['order_repo'],
        product_repo=dependencies['product_repo'],
        event_publisher=dependencies['event_pub'],
        id_generator=dependencies['id_gen'],
    )


class TestCreateOrder:
    """注文作成ユースケースのテスト"""

    def test_正常系_注文が作成される(self, use_case, dependencies):
        """商品が存在する場合、注文が正常に作成される"""
        result = use_case.execute(CreateOrderInput(
            user_id="user-1",
            items=[
                OrderItemInput(product_id="prod-1", quantity=2),
                OrderItemInput(product_id="prod-2", quantity=1),
            ],
        ))

        # Output DTO の検証
        assert result.order_id == "order-1"
        assert result.total_amount == 4500   # 1000*2 + 2500*1
        assert result.item_count == 2
        assert result.status == "pending"

        # リポジトリに保存されたことを検証
        assert len(dependencies['order_repo'].saved) == 1
        saved_order = dependencies['order_repo'].saved[0]
        assert saved_order.user_id == "user-1"

        # イベントが発行されたことを検証
        assert len(dependencies['event_pub'].events) == 1
        event_name, event_data = dependencies['event_pub'].events[0]
        assert event_name == "order.created"
        assert event_data['order_id'] == "order-1"

    def test_異常系_存在しない商品(self, use_case):
        """存在しない商品IDが含まれる場合、エラーになる"""
        with pytest.raises(ProductNotFoundError):
            use_case.execute(CreateOrderInput(
                user_id="user-1",
                items=[
                    OrderItemInput(product_id="nonexistent", quantity=1),
                ],
            ))

    def test_異常系_空のアイテムリスト(self, use_case):
        """アイテムリストが空の場合、エラーになる"""
        with pytest.raises(ValueError, match="注文アイテムが空"):
            use_case.execute(CreateOrderInput(
                user_id="user-1",
                items=[],
            ))

    def test_正常系_複数の注文を作成(self, use_case, dependencies):
        """連続して注文を作成した場合、それぞれ別IDが割り当てられる"""
        result1 = use_case.execute(CreateOrderInput(
            user_id="user-1",
            items=[OrderItemInput(product_id="prod-1", quantity=1)],
        ))
        result2 = use_case.execute(CreateOrderInput(
            user_id="user-2",
            items=[OrderItemInput(product_id="prod-2", quantity=3)],
        ))

        assert result1.order_id == "order-1"
        assert result2.order_id == "order-2"
        assert len(dependencies['order_repo'].saved) == 2


# tests/unit/test_order_entity.py

class TestOrderEntity:
    """Order エンティティのビジネスルールテスト"""

    def _make_order(self, **kwargs) -> Order:
        defaults = {
            'id': 'order-1',
            'user_id': 'user-1',
            'items': [
                OrderItem(
                    product_id='prod-1',
                    name='テスト商品',
                    price=1000,
                    quantity=2,
                )
            ],
        }
        defaults.update(kwargs)
        return Order(**defaults)

    def test_合計金額の計算(self):
        order = self._make_order(items=[
            OrderItem(product_id='p1', name='A', price=1000, quantity=2),
            OrderItem(product_id='p2', name='B', price=500, quantity=3),
        ])
        assert order.total_amount == 3500   # 1000*2 + 500*3

    def test_PENDING_から_CONFIRMED_への遷移(self):
        order = self._make_order()
        order.confirm()
        assert order.status == OrderStatus.CONFIRMED

    def test_CONFIRMED_から_SHIPPED_への遷移(self):
        order = self._make_order()
        order.confirm()
        order.ship()
        assert order.status == OrderStatus.SHIPPED

    def test_SHIPPED_状態でキャンセル不可(self):
        order = self._make_order()
        order.confirm()
        order.ship()
        with pytest.raises(ValueError, match="取消不可"):
            order.cancel()

    def test_空の注文は確定不可(self):
        order = self._make_order(items=[])
        with pytest.raises(ValueError, match="注文アイテムが空"):
            order.confirm()
```

---

## 4. ディレクトリ構成

### 4.1 推奨ディレクトリ構成（Python）

```
project/
├── domain/                      # Entities（最内層）
│   ├── __init__.py
│   ├── entities/
│   │   ├── __init__.py
│   │   ├── order.py             # Order 集約ルート
│   │   ├── product.py           # Product エンティティ
│   │   └── user.py              # User エンティティ
│   ├── value_objects/
│   │   ├── __init__.py
│   │   ├── money.py             # Money 値オブジェクト
│   │   ├── email.py             # Email 値オブジェクト
│   │   └── address.py           # Address 値オブジェクト
│   ├── events/
│   │   ├── __init__.py
│   │   └── order_events.py      # OrderCreated, OrderCancelled 等
│   └── exceptions.py            # ドメイン例外
│
├── application/                 # Use Cases
│   ├── __init__.py
│   ├── use_cases/
│   │   ├── __init__.py
│   │   ├── create_order.py      # 注文作成ユースケース
│   │   ├── cancel_order.py      # 注文キャンセルユースケース
│   │   ├── get_order.py         # 注文取得ユースケース
│   │   └── list_orders.py       # 注文一覧ユースケース
│   ├── ports/                   # ポート（インターフェース定義）
│   │   ├── __init__.py
│   │   ├── order_repository.py  # OrderRepository Protocol
│   │   ├── product_repository.py
│   │   ├── event_publisher.py   # EventPublisher Protocol
│   │   └── id_generator.py      # IdGenerator Protocol
│   ├── dto/                     # データ転送オブジェクト
│   │   ├── __init__.py
│   │   ├── order_dto.py         # Input/Output DTO
│   │   └── product_dto.py
│   └── exceptions.py            # アプリケーション例外
│
├── adapters/                    # Interface Adapters
│   ├── __init__.py
│   ├── controllers/             # HTTP → UseCase
│   │   ├── __init__.py
│   │   ├── order_controller.py
│   │   └── product_controller.py
│   ├── repositories/            # DB → Entity
│   │   ├── __init__.py
│   │   ├── postgres_order_repository.py
│   │   ├── postgres_product_repository.py
│   │   └── inmemory_order_repository.py  # テスト用
│   ├── presenters/              # UseCase Output → 表示形式
│   │   ├── __init__.py
│   │   └── order_presenter.py
│   └── gateways/                # Entity → 外部 API
│       ├── __init__.py
│       └── payment_gateway.py
│
├── infrastructure/              # Frameworks & Drivers
│   ├── __init__.py
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py            # SQLAlchemy モデル
│   │   ├── connection.py        # DB接続設定
│   │   └── migrations/          # Alembic マイグレーション
│   ├── web/
│   │   ├── __init__.py
│   │   ├── flask_app.py         # Flask アプリケーション設定
│   │   └── middleware.py        # ミドルウェア
│   ├── messaging/
│   │   ├── __init__.py
│   │   └── kafka_publisher.py   # Kafka イベント発行
│   └── container.py             # DI コンテナ
│
├── tests/
│   ├── unit/                    # Entity + UseCase テスト
│   │   ├── test_order_entity.py
│   │   ├── test_create_order.py
│   │   └── test_cancel_order.py
│   ├── integration/             # Repository + DB テスト
│   │   ├── test_postgres_order_repository.py
│   │   └── conftest.py          # テスト用DB設定
│   └── e2e/                     # API テスト
│       ├── test_order_api.py
│       └── conftest.py          # テスト用サーバー設定
│
├── config/
│   ├── settings.py              # 環境別設定
│   └── logging.py               # ログ設定
│
└── main.py                      # エントリポイント
```

### 4.2 TypeScript プロジェクトの場合

```
src/
├── domain/
│   ├── entities/
│   │   ├── Order.ts
│   │   └── Product.ts
│   ├── valueObjects/
│   │   ├── Money.ts
│   │   └── Email.ts
│   └── events/
│       └── OrderEvents.ts
├── application/
│   ├── useCases/
│   │   ├── CreateOrder.ts
│   │   └── CancelOrder.ts
│   ├── ports/
│   │   ├── IOrderRepository.ts
│   │   └── IEventPublisher.ts
│   └── dto/
│       └── OrderDto.ts
├── adapters/
│   ├── controllers/
│   │   └── OrderController.ts
│   ├── repositories/
│   │   └── TypeOrmOrderRepository.ts
│   └── presenters/
│       └── OrderPresenter.ts
├── infrastructure/
│   ├── database/
│   │   └── typeOrmConfig.ts
│   ├── web/
│   │   └── expressApp.ts
│   └── container.ts             # tsyringe DI
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

### 4.3 Go プロジェクトの場合

```
internal/
├── domain/
│   ├── order.go                 # Order 構造体 + ビジネスロジック
│   ├── product.go
│   ├── money.go                 # 値オブジェクト
│   └── events.go                # ドメインイベント
├── application/
│   ├── create_order.go          # UseCase
│   ├── cancel_order.go
│   └── ports.go                 # interface 定義
├── adapters/
│   ├── http/
│   │   └── order_handler.go     # HTTP ハンドラ
│   └── postgres/
│       └── order_repository.go  # PostgreSQL 実装
├── infrastructure/
│   ├── server.go                # HTTP サーバー設定
│   └── database.go              # DB 接続設定
cmd/
└── api/
    └── main.go                  # エントリポイント
```

---

## 5. 高度なトピック

### 5.1 Presenter パターン

Controller が UseCase の出力を直接 JSON に変換する単純なケースだけでなく、**Presenter パターン**を使うことで出力形式の変換ロジックを分離できる。

```python
# adapters/presenters/order_presenter.py

class OrderPresenter:
    """
    UseCase の出力を様々な形式に変換する

    WHY Presenterを分離?
    - 同じ UseCase の出力を JSON / HTML / CSV / gRPC など
      異なる形式に変換する必要がある場合
    - Controller が変換ロジックで肥大化するのを防ぐ
    """

    @staticmethod
    def to_json(output: CreateOrderOutput) -> dict:
        """API レスポンス用 JSON"""
        return {
            'order_id': output.order_id,
            'total_amount': output.total_amount,
            'total_amount_display': f"¥{output.total_amount:,}",
            'item_count': output.item_count,
            'status': output.status,
        }

    @staticmethod
    def to_csv_row(output: CreateOrderOutput) -> str:
        """CSV エクスポート用"""
        return (f"{output.order_id},{output.total_amount},"
                f"{output.item_count},{output.status}")

    @staticmethod
    def to_notification(output: CreateOrderOutput) -> dict:
        """通知メッセージ用"""
        return {
            'title': '注文が作成されました',
            'body': (f'注文ID: {output.order_id} / '
                     f'合計: ¥{output.total_amount:,}'),
        }
```

### 5.2 CQRS との組み合わせ

CQRS (Command Query Responsibility Segregation) は「書き込み」と「読み取り」を分離するパターンで、クリーンアーキテクチャと非常に相性がよい。

```
クリーンアーキテクチャ + CQRS

  Command (書き込み)                    Query (読み取り)
  ┌──────────────────┐                ┌──────────────────┐
  │  CreateOrderInput │                │  GetOrderInput   │
  │        ↓          │                │        ↓          │
  │  CreateOrderUseCase│               │  GetOrderQuery   │
  │        ↓          │                │        ↓          │
  │  OrderRepository  │                │  OrderReadModel  │
  │  (書き込み専用)    │                │  (読み取り専用)   │
  │        ↓          │                │        ↓          │
  │  PostgreSQL       │──同期/非同期──>│  Elasticsearch   │
  └──────────────────┘                └──────────────────┘

  Command 側: ドメインモデル経由でビジネスルールを適用
  Query 側: パフォーマンス最適化されたRead Modelから直接取得
  → 書き込みと読み取りを独立してスケーリング可能
```

```python
# application/queries/get_order_query.py

@dataclass(frozen=True)
class GetOrderInput:
    order_id: str

@dataclass(frozen=True)
class OrderDetailOutput:
    order_id: str
    user_id: str
    items: list[dict]
    total_amount: int
    status: str
    created_at: str

class OrderReadRepository(Protocol):
    """読み取り専用リポジトリ（CQRS の Query 側）"""
    def find_detail_by_id(
        self, order_id: str
    ) -> OrderDetailOutput | None: ...
    def search(
        self, user_id: str, status: str | None = None
    ) -> list[OrderDetailOutput]: ...

class GetOrderQuery:
    """
    注文詳細取得クエリ

    注意: Query はドメインモデルを経由せず、
    Read Model から直接 DTO を返す。
    ビジネスルールの適用は不要なため。
    """

    def __init__(self, read_repo: OrderReadRepository):
        self._read_repo = read_repo

    def execute(self, input_dto: GetOrderInput) -> OrderDetailOutput:
        result = self._read_repo.find_detail_by_id(input_dto.order_id)
        if not result:
            raise OrderNotFoundError(input_dto.order_id)
        return result
```

### 5.3 マイクロサービスでのクリーンアーキテクチャ

```
マイクロサービス間の連携

  [注文サービス]                    [在庫サービス]
  ┌─────────────────┐              ┌─────────────────┐
  │ Domain           │              │ Domain           │
  │  Order Entity    │              │  Stock Entity    │
  ├─────────────────┤              ├─────────────────┤
  │ Application      │              │ Application      │
  │  CreateOrder     │──イベント──>│  ReserveStock    │
  ├─────────────────┤    (Kafka)   ├─────────────────┤
  │ Adapters         │              │ Adapters         │
  │  KafkaPublisher  │              │  KafkaConsumer   │
  ├─────────────────┤              ├─────────────────┤
  │ Infrastructure   │              │ Infrastructure   │
  │  PostgreSQL      │              │  MongoDB         │
  └─────────────────┘              └─────────────────┘

  各サービスが独立したクリーンアーキテクチャ構造を持つ
  サービス間はイベント（ドメインイベント）で疎結合に連携
  各サービスは独自のDB技術を選択可能（Polyglot Persistence）
```

```python
# adapters/gateways/inventory_gateway.py

class InventoryGateway(Protocol):
    """在庫サービスとの連携ポート"""
    def check_availability(
        self, product_id: str, quantity: int
    ) -> bool: ...
    def reserve(
        self, product_id: str, quantity: int
    ) -> str: ...

class HttpInventoryGateway:
    """在庫サービスの HTTP API クライアント実装"""

    def __init__(self, base_url: str, http_client: 'HttpClient'):
        self._base_url = base_url
        self._client = http_client

    def check_availability(
        self, product_id: str, quantity: int
    ) -> bool:
        response = self._client.get(
            f"{self._base_url}/products/{product_id}/availability",
            params={'quantity': quantity},
        )
        return response.json()['available']

    def reserve(self, product_id: str, quantity: int) -> str:
        response = self._client.post(
            f"{self._base_url}/reservations",
            json={'product_id': product_id, 'quantity': quantity},
        )
        return response.json()['reservation_id']
```

### 5.4 段階的導入戦略

プロジェクトの規模に応じた3段階のアプローチを示す。

```
Step 1: 最小構成（小規模プロジェクト / MVP）
────────────────────────────────────────────
  domain/
    entities/        ← ビジネスルールを持つエンティティ
  application/
    use_cases/       ← ユースケース + Port定義
  infrastructure/
    everything_else/ ← Controller + Repository + DB + 全部

  レイヤー数: 3（Domain / Application / Infrastructure）
  目的: 最低限の関心の分離を達成

Step 2: 標準構成（中規模プロジェクト）
────────────────────────────────────────────
  domain/
    entities/
    value_objects/
  application/
    use_cases/
    ports/
    dto/
  adapters/
    controllers/
    repositories/
  infrastructure/
    db/
    web/

  レイヤー数: 4（完全な同心円モデル）
  目的: テスタビリティと変更容易性の確保

Step 3: 完全構成（大規模 / マイクロサービス）
────────────────────────────────────────────
  上記 + CQRS + イベントソーシング + Saga
  モジュール単位で Bounded Context を分離
  各モジュールが独立したクリーンアーキテクチャ構造
```

### 5.5 TypeScript での実装例

Python 以外の言語での実装例として、TypeScript 版を示す。

```typescript
// domain/entities/Order.ts

export enum OrderStatus {
  PENDING = "pending",
  CONFIRMED = "confirmed",
  SHIPPED = "shipped",
  CANCELLED = "cancelled",
}

export class OrderItem {
  constructor(
    public readonly productId: string,
    public readonly name: string,
    public readonly price: number,
    public readonly quantity: number,
  ) {
    if (price < 0) throw new Error(`価格は0以上: ${price}`);
    if (quantity < 1) throw new Error(`数量は1以上: ${quantity}`);
  }

  get subtotal(): number {
    return this.price * this.quantity;
  }
}

export class Order {
  private _status: OrderStatus = OrderStatus.PENDING;

  constructor(
    public readonly id: string,
    public readonly userId: string,
    private _items: OrderItem[],
  ) {}

  get status(): OrderStatus { return this._status; }
  get items(): ReadonlyArray<OrderItem> { return this._items; }
  get totalAmount(): number {
    return this._items.reduce((sum, i) => sum + i.subtotal, 0);
  }

  confirm(): void {
    if (this._status !== OrderStatus.PENDING) {
      throw new Error(`PENDING でのみ確定可能 (現在: ${this._status})`);
    }
    if (this._items.length === 0) {
      throw new Error("注文アイテムが空です");
    }
    this._status = OrderStatus.CONFIRMED;
  }

  cancel(): void {
    if ([OrderStatus.SHIPPED, OrderStatus.CANCELLED].includes(this._status)) {
      throw new Error("出荷済み/キャンセル済みの注文は取消不可");
    }
    this._status = OrderStatus.CANCELLED;
  }
}

// application/ports/IOrderRepository.ts

export interface IOrderRepository {
  save(order: Order): Promise<void>;
  findById(orderId: string): Promise<Order | null>;
}

// application/useCases/CreateOrder.ts

export interface CreateOrderInput {
  userId: string;
  items: { productId: string; quantity: number }[];
}

export interface CreateOrderOutput {
  orderId: string;
  totalAmount: number;
  status: string;
}

export class CreateOrderUseCase {
  constructor(
    private orderRepo: IOrderRepository,
    private productRepo: IProductRepository,
    private eventPublisher: IEventPublisher,
  ) {}

  async execute(input: CreateOrderInput): Promise<CreateOrderOutput> {
    const orderItems: OrderItem[] = [];
    for (const item of input.items) {
      const product = await this.productRepo.findById(item.productId);
      if (!product) throw new Error(`商品が見つかりません: ${item.productId}`);
      orderItems.push(
        new OrderItem(product.id, product.name, product.price, item.quantity)
      );
    }

    const order = new Order(generateId(), input.userId, orderItems);
    await this.orderRepo.save(order);
    await this.eventPublisher.publish("order.created", {
      orderId: order.id,
    });

    return {
      orderId: order.id,
      totalAmount: order.totalAmount,
      status: order.status,
    };
  }
}
```

---

## 6. レイヤー比較表

### 6.1 各レイヤーの特性比較

| レイヤー | 責務 | 依存先 | 変更頻度 | テスト方法 | コード例 |
|---------|------|--------|---------|-----------|---------|
| Entities | ビジネスルール | なし（自己完結） | 最も低い | ユニットテスト | Order, Money |
| Use Cases | アプリケーションロジック | Entities + Port Interface | 中程度 | ユニットテスト (Fake) | CreateOrderUseCase |
| Adapters | データ形式の変換 | Use Cases + Entities | 中〜高 | 統合テスト | PostgresOrderRepository |
| Frameworks | 技術詳細の実装 | 全レイヤー（最外層） | 最も高い | E2E テスト | Flask, SQLAlchemy |

### 6.2 類似アーキテクチャとの比較

| 特性 | クリーンアーキテクチャ | ヘキサゴナル (Ports & Adapters) | オニオン | 従来型 3層 (MVC) |
|------|---------------------|-------------------------------|---------|-----------------|
| 提唱者 | Robert C. Martin (2012) | Alistair Cockburn (2005) | Jeffrey Palermo (2008) | Trygve Reenskaug (1979) |
| 中心概念 | 依存性の方向制御 | ポートとアダプター | ドメインモデル中心 | UI/Logic/Data の分離 |
| レイヤー数 | 4（明確に定義） | 2（内側/外側） | 4（Domain/Service/Infra/UI） | 3（View/Controller/Model） |
| 入出力の扱い | 同心円の外側に配置 | Primary/Secondary Port で明示 | 外層に配置 | Controller が直接処理 |
| フレームワーク依存 | プラグインとして扱う | アダプタ経由 | 外層で吸収 | 密結合になりやすい |
| テスタビリティ | 非常に高い | 高い | 高い | 中程度 |
| 学習コスト | 高い | 中程度 | 中程度 | 低い |
| 初期構築コスト | 高い | 中程度 | 中程度 | 低い |
| 適合プロジェクト規模 | 中〜大規模 | 中規模以上 | 中規模以上 | 小〜中規模 |

### 6.3 テスト戦略の比較

```
テストピラミッドとクリーンアーキテクチャの対応

          /\
         /  \         E2E テスト（少数）
        / E2E\        → Frameworks 層: 全レイヤー統合
       /------\
      /  統合   \      統合テスト（中程度）
     / テスト    \     → Adapters 層: DB接続テスト等
    /------------\
   /  ユニット    \    ユニットテスト（多数）
  /  テスト       \   → Entities + Use Cases: Fake で高速テスト
 /________________\

テスト実行時間:
  Entity テスト:   ~0.01秒/テスト（純粋な計算）
  UseCase テスト:  ~0.05秒/テスト（Fake リポジトリ）
  統合テスト:      ~0.5秒/テスト（DB接続あり）
  E2E テスト:      ~2秒/テスト（HTTP + DB）
```

### 6.4 各レイヤーで扱うデータ形式の比較

| レイヤー | 入力データ | 出力データ | 変換の責務者 |
|---------|-----------|-----------|------------|
| Frameworks | JSON / HTML Form / gRPC | JSON / HTML / Protobuf | フレームワーク自身 |
| Adapters (Controller) | Request DTO | Response DTO | Controller |
| Use Cases | Input DTO | Output DTO | UseCase 自身 |
| Entities | メソッド引数 | 戻り値 / プロパティ | Entity 自身 |

---

## 7. アンチパターン

### アンチパターン 1: Entity がフレームワークに依存

```python
# NG: Entity が SQLAlchemy に依存
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

class Order(Base):           # ← フレームワークへの依存
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50))
    status = Column(String(20))
    # Entity と DB モデルが混在
    # → DB 変更時にビジネスロジックに影響
    # → ユニットテストに DB 接続が必要

# OK: Entity は純粋な Python クラス
@dataclass
class Order:
    id: str
    user_id: str
    items: List[OrderItem]
    status: OrderStatus = OrderStatus.PENDING
    # DB の知識は一切持たない
    # → DB を変更しても Entity は不変
    # → ユニットテストが即座に実行可能

    def confirm(self) -> None:
        if self.status != OrderStatus.PENDING:
            raise ValueError("PENDING でのみ確定可能")
        self.status = OrderStatus.CONFIRMED
```

**WHY これが問題なのか？**

Entity が ORM に依存すると、(1) テスト時に DB 接続が必須になりテスト速度が低下する、(2) DB のスキーマ変更がビジネスロジックに影響する、(3) ORM の制約に合わせてドメインモデルを歪める必要が生じる。Django の Model で `full_clean()` に全てのバリデーションを押し込む設計は、まさにこのアンチパターンの典型例である。

### アンチパターン 2: UseCase が HTTP リクエストに直接依存

```python
# NG: UseCase が Flask の request に依存
from flask import request

class CreateOrderUseCase:
    def execute(self):
        user_id = request.json['user_id']  # ← Web フレームワーク依存
        items = request.json['items']
        # → CLI からの呼び出し不可
        # → バッチ処理からの呼び出し不可
        # → テスト時に Flask のリクエストコンテキストが必要

# OK: DTO を介して完全に分離
class CreateOrderUseCase:
    def execute(self, input_dto: CreateOrderInput):
        user_id = input_dto.user_id        # ← 純粋なデータクラス
        # → HTTP, CLI, バッチ、テスト、どこからでも呼び出し可能
```

**WHY これが問題なのか？**

UseCase が HTTP リクエストに依存すると、同じビジネスロジックを CLI やバッチ処理、メッセージキューのコンシューマーから呼び出せなくなる。DTO を介することで、UseCase は入力元を一切知らずに済む。

### アンチパターン 3: レイヤーを飛び越える依存

```python
# NG: Controller が直接 Repository 実装を使用
class OrderController:
    def __init__(self):
        self._repo = PostgresOrderRepository()  # ← 具象クラスに直接依存

    def get_order(self, order_id):
        # UseCase を経由せずに直接 DB にアクセス
        return self._repo.find_by_id(order_id)
        # → ビジネスルールがバイパスされる
        # → テスト時に PostgreSQL が必要

# OK: Controller → UseCase → Repository Interface の順序を守る
class OrderController:
    def __init__(self, get_order_use_case: GetOrderUseCase):
        self._use_case = get_order_use_case  # ← UseCase に依存

    def get_order(self, order_id):
        return self._use_case.execute(GetOrderInput(order_id=order_id))
        # → ビジネスルールが必ず適用される
        # → テスト時は UseCase を Fake に差し替え可能
```

### アンチパターン 4: UseCase の肥大化（God UseCase）

```python
# NG: 1つの UseCase に多すぎる責務
class OrderUseCase:
    def create_order(self, ...): ...
    def cancel_order(self, ...): ...
    def update_order(self, ...): ...
    def get_order(self, ...): ...
    def list_orders(self, ...): ...
    def export_orders(self, ...): ...
    # → 単一責任の原則に違反
    # → テストが複雑化
    # → 変更の影響範囲が広い

# OK: 1 UseCase = 1 操作
class CreateOrderUseCase:
    def execute(self, input_dto: CreateOrderInput) -> CreateOrderOutput:
        ...

class CancelOrderUseCase:
    def execute(self, input_dto: CancelOrderInput) -> CancelOrderOutput:
        ...

class GetOrderQuery:
    def execute(self, input_dto: GetOrderInput) -> OrderDetailOutput:
        ...
```

### アンチパターン 5: DTO を使わずにエンティティを直接公開

```python
# NG: UseCase がエンティティを直接返す
class GetOrderUseCase:
    def execute(self, order_id: str) -> Order:
        order = self._order_repo.find_by_id(order_id)
        return order    # ← Entity を直接返す
        # → Controller が Entity のメソッドを呼べてしまう
        # → Entity の内部構造変更が API レスポンスに影響
        # → シリアライズが Entity の責務になってしまう

# OK: Output DTO を介して情報を公開
class GetOrderUseCase:
    def execute(self, order_id: str) -> OrderDetailOutput:
        order = self._order_repo.find_by_id(order_id)
        return OrderDetailOutput(
            order_id=order.id,
            status=order.status.value,
            total_amount=order.total_amount,
        )
        # → Controller は DTO のフィールドしかアクセスできない
        # → Entity の内部構造変更は DTO 変換で吸収
```

---

## 8. 実践演習

### 演習1（基礎）: Entity のビジネスルール実装

**課題**: 以下の仕様を持つ `ShoppingCart` エンティティを実装せよ。

```
仕様:
- カートにアイテムを追加できる（add_item）
- 同じ商品は数量を加算する
- カート内のアイテム数上限は20
- 合計金額を計算できる（total_amount プロパティ）
- カートをクリアできる（clear）
- カート内のアイテム数を返せる（item_count プロパティ）
```

**期待される実装と出力**:

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass(frozen=True)
class CartItem:
    """カートアイテム（値オブジェクト）"""
    product_id: str
    name: str
    price: int
    quantity: int

    def __post_init__(self):
        if self.price < 0:
            raise ValueError(f"価格は0以上: {self.price}")
        if self.quantity < 1:
            raise ValueError(f"数量は1以上: {self.quantity}")

    @property
    def subtotal(self) -> int:
        return self.price * self.quantity

@dataclass
class ShoppingCart:
    """ショッピングカート集約ルート"""
    id: str
    user_id: str
    _items: List[CartItem] = field(default_factory=list)
    MAX_ITEMS = 20

    @property
    def items(self) -> List[CartItem]:
        return list(self._items)

    @property
    def item_count(self) -> int:
        return sum(item.quantity for item in self._items)

    @property
    def total_amount(self) -> int:
        return sum(item.subtotal for item in self._items)

    def add_item(self, item: CartItem) -> None:
        existing = self._find_item(item.product_id)
        if existing:
            new_quantity = existing.quantity + item.quantity
            if self._total_quantity() - existing.quantity + new_quantity > self.MAX_ITEMS:
                raise ValueError(f"カート上限({self.MAX_ITEMS})を超えます")
            idx = self._items.index(existing)
            self._items[idx] = CartItem(
                product_id=existing.product_id,
                name=existing.name,
                price=existing.price,
                quantity=new_quantity,
            )
        else:
            if self._total_quantity() + item.quantity > self.MAX_ITEMS:
                raise ValueError(f"カート上限({self.MAX_ITEMS})を超えます")
            self._items.append(item)

    def remove_item(self, product_id: str) -> None:
        existing = self._find_item(product_id)
        if not existing:
            raise ValueError(f"商品がカートにありません: {product_id}")
        self._items.remove(existing)

    def clear(self) -> None:
        self._items.clear()

    def _find_item(self, product_id: str) -> Optional[CartItem]:
        return next(
            (i for i in self._items if i.product_id == product_id), None
        )

    def _total_quantity(self) -> int:
        return sum(i.quantity for i in self._items)


# テスト実行
cart = ShoppingCart(id="cart-1", user_id="user-1")
cart.add_item(CartItem(product_id="p1", name="りんご", price=150, quantity=3))
cart.add_item(CartItem(product_id="p2", name="みかん", price=100, quantity=5))
print(f"アイテム数: {cart.item_count}")      # 出力: アイテム数: 8
print(f"合計金額: ¥{cart.total_amount:,}")    # 出力: 合計金額: ¥950

# 同じ商品を追加すると数量加算
cart.add_item(CartItem(product_id="p1", name="りんご", price=150, quantity=2))
print(f"アイテム数: {cart.item_count}")      # 出力: アイテム数: 10
print(f"合計金額: ¥{cart.total_amount:,}")    # 出力: 合計金額: ¥1,250

cart.clear()
print(f"クリア後: {cart.item_count}")        # 出力: クリア後: 0
```

### 演習2（応用）: UseCase + テスト実装

**課題**: 以下の `TransferMoneyUseCase`（送金ユースケース）を実装し、テストを書け。

```
仕様:
- 送金元アカウントの残高を減算
- 送金先アカウントの残高を加算
- 残高不足の場合はエラー
- 送金完了後にイベントを発行
- 同一アカウントへの送金はエラー
```

**期待される実装と出力**:

```python
# --- Entity ---
@dataclass
class Account:
    id: str
    owner_name: str
    balance: int   # 円単位

    def withdraw(self, amount: int) -> None:
        if amount <= 0:
            raise ValueError("出金額は正の値")
        if self.balance < amount:
            raise ValueError(
                f"残高不足: 残高{self.balance}円 < 出金{amount}円"
            )
        self.balance -= amount

    def deposit(self, amount: int) -> None:
        if amount <= 0:
            raise ValueError("入金額は正の値")
        self.balance += amount

# --- DTO ---
@dataclass(frozen=True)
class TransferInput:
    from_account_id: str
    to_account_id: str
    amount: int

@dataclass(frozen=True)
class TransferOutput:
    from_balance: int
    to_balance: int
    transferred_amount: int

# --- Port ---
class AccountRepository(Protocol):
    def find_by_id(self, account_id: str) -> Account | None: ...
    def save(self, account: Account) -> None: ...

# --- UseCase ---
class TransferMoneyUseCase:
    def __init__(
        self,
        account_repo: AccountRepository,
        event_publisher: EventPublisher,
    ):
        self._repo = account_repo
        self._events = event_publisher

    def execute(self, input_dto: TransferInput) -> TransferOutput:
        if input_dto.from_account_id == input_dto.to_account_id:
            raise ValueError("同一アカウントへの送金は不可")
        if input_dto.amount <= 0:
            raise ValueError("送金額は正の値")

        from_account = self._repo.find_by_id(input_dto.from_account_id)
        to_account = self._repo.find_by_id(input_dto.to_account_id)

        if not from_account:
            raise ValueError(
                f"送金元が見つかりません: {input_dto.from_account_id}"
            )
        if not to_account:
            raise ValueError(
                f"送金先が見つかりません: {input_dto.to_account_id}"
            )

        from_account.withdraw(input_dto.amount)
        to_account.deposit(input_dto.amount)

        self._repo.save(from_account)
        self._repo.save(to_account)

        self._events.publish('money.transferred', {
            'from': from_account.id,
            'to': to_account.id,
            'amount': input_dto.amount,
        })

        return TransferOutput(
            from_balance=from_account.balance,
            to_balance=to_account.balance,
            transferred_amount=input_dto.amount,
        )

# --- テスト ---
class FakeAccountRepository:
    def __init__(self):
        self._store: dict[str, Account] = {}

    def save(self, account: Account) -> None:
        self._store[account.id] = account

    def find_by_id(self, account_id: str) -> Account | None:
        return self._store.get(account_id)

    def add(self, account: Account) -> None:
        self._store[account.id] = account

def test_送金_正常系():
    repo = FakeAccountRepository()
    repo.add(Account(id="A", owner_name="田中", balance=10000))
    repo.add(Account(id="B", owner_name="鈴木", balance=5000))
    events = FakeEventPublisher()

    uc = TransferMoneyUseCase(repo, events)
    result = uc.execute(TransferInput(
        from_account_id="A", to_account_id="B", amount=3000
    ))

    assert result.from_balance == 7000    # 10000 - 3000
    assert result.to_balance == 8000      # 5000 + 3000
    assert result.transferred_amount == 3000
    assert events.events[0][0] == "money.transferred"
    print("OK: 送金正常系テスト通過")

def test_送金_残高不足():
    repo = FakeAccountRepository()
    repo.add(Account(id="A", owner_name="田中", balance=1000))
    repo.add(Account(id="B", owner_name="鈴木", balance=5000))
    events = FakeEventPublisher()

    uc = TransferMoneyUseCase(repo, events)
    try:
        uc.execute(TransferInput(
            from_account_id="A", to_account_id="B", amount=5000
        ))
        assert False, "例外が発生するべき"
    except ValueError as e:
        assert "残高不足" in str(e)
        print("OK: 残高不足テスト通過")

test_送金_正常系()     # 出力: OK: 送金正常系テスト通過
test_送金_残高不足()    # 出力: OK: 残高不足テスト通過
```

### 演習3（発展）: フレームワーク移行シミュレーション

**課題**: Flask で実装された既存の注文 API を FastAPI に移行せよ。クリーンアーキテクチャに従っていれば、**変更するのは Controller（Adapter 層）と Web Framework（Infrastructure 層）のみ**であることを確認せよ。

```
目標:
- domain/ と application/ は一切変更しない
- adapters/controllers/ を FastAPI 用に書き換える
- infrastructure/web/ を FastAPI 用に書き換える
- テストが全て通ることを確認する
```

**期待される実装**:

```python
# === Before: Flask (変更前) ===

# adapters/controllers/order_controller_flask.py
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/orders', methods=['POST'])
def create_order():
    body = request.get_json()
    input_dto = CreateOrderInput(
        user_id=body['user_id'],
        items=[OrderItemInput(**i) for i in body['items']],
    )
    use_case = container.resolve(CreateOrderUseCase)
    output = use_case.execute(input_dto)
    return jsonify(OrderPresenter.to_json(output)), 201


# === After: FastAPI (変更後) ===

# adapters/controllers/order_controller_fastapi.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class CreateOrderRequest(BaseModel):
    """FastAPI のリクエストバリデーション"""
    user_id: str
    items: list[dict]

class CreateOrderResponse(BaseModel):
    """FastAPI のレスポンスシリアライゼーション"""
    order_id: str
    total_amount: int
    item_count: int
    status: str

@app.post(
    '/orders',
    response_model=CreateOrderResponse,
    status_code=201,
)
async def create_order(req: CreateOrderRequest):
    input_dto = CreateOrderInput(
        user_id=req.user_id,
        items=[OrderItemInput(**i) for i in req.items],
    )
    try:
        # UseCase は Flask 版と全く同じものを使う
        use_case = container.resolve(CreateOrderUseCase)
        output = use_case.execute(input_dto)
        return CreateOrderResponse(
            order_id=output.order_id,
            total_amount=output.total_amount,
            item_count=output.item_count,
            status=output.status,
        )
    except ProductNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# 変更ファイルの確認:
# domain/              → 変更なし（0ファイル）
# application/         → 変更なし（0ファイル）
# adapters/controllers → order_controller.py のみ変更（1ファイル）
# infrastructure/web/  → flask_app.py → fastapi_app.py（1ファイル）
# tests/unit/          → 変更なし（Fake を使っているため）
# tests/e2e/           → Flask テストクライアント → httpx に変更

# フレームワーク移行で変更したのは Adapter + Infrastructure のみ
# ビジネスロジック（Entity + UseCase）は一切触れていない
# → これがクリーンアーキテクチャの真価
```

---

## 9. 実装上の注意点とベストプラクティス

### 9.1 エラーハンドリング戦略

各レイヤーでのエラーの扱い方を統一することが重要である。

```python
# ドメイン層: ドメイン例外
class DomainError(Exception):
    """ビジネスルール違反"""
    pass

class InsufficientBalanceError(DomainError):
    pass

# アプリケーション層: アプリケーション例外
class ApplicationError(Exception):
    pass

class OrderNotFoundError(ApplicationError):
    pass

# Adapter 層: 例外を HTTP ステータスに変換
ERROR_STATUS_MAP = {
    DomainError: 422,          # Unprocessable Entity
    OrderNotFoundError: 404,    # Not Found
    PermissionDeniedError: 403, # Forbidden
    ValueError: 400,            # Bad Request
}

@app.errorhandler(Exception)
def handle_error(error):
    status = ERROR_STATUS_MAP.get(type(error), 500)
    return jsonify({'error': str(error)}), status
```

### 9.2 ログ戦略

```python
# NG: Entity や UseCase 内でロガーを直接使う
import logging
logger = logging.getLogger(__name__)

class CreateOrderUseCase:
    def execute(self, input_dto):
        logger.info("注文作成開始")  # ← ロギングフレームワークへの依存
        ...

# OK: ログは Adapter 層またはデコレータで行う
class LoggingUseCaseDecorator:
    """UseCase にログ機能を追加するデコレータ"""

    def __init__(self, use_case, logger):
        self._use_case = use_case
        self._logger = logger

    def execute(self, input_dto):
        self._logger.info(
            f"UseCase開始: {type(self._use_case).__name__}"
        )
        try:
            result = self._use_case.execute(input_dto)
            self._logger.info(
                f"UseCase成功: {type(self._use_case).__name__}"
            )
            return result
        except Exception as e:
            self._logger.error(
                f"UseCase失敗: {type(self._use_case).__name__} - {e}"
            )
            raise
```

### 9.3 トランザクション管理

```python
# Unit of Work パターンでトランザクションを管理

class UnitOfWork(Protocol):
    def __enter__(self) -> 'UnitOfWork': ...
    def __exit__(self, *args) -> None: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...

class SQLAlchemyUnitOfWork:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def __enter__(self):
        self._session = self._session_factory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        self._session.close()

    def commit(self):
        self._session.commit()

    def rollback(self):
        self._session.rollback()


# UseCase でのトランザクション管理
class TransferMoneyUseCase:
    def __init__(self, account_repo, event_publisher, uow: UnitOfWork):
        self._repo = account_repo
        self._events = event_publisher
        self._uow = uow

    def execute(self, input_dto: TransferInput) -> TransferOutput:
        with self._uow:
            from_account = self._repo.find_by_id(input_dto.from_account_id)
            to_account = self._repo.find_by_id(input_dto.to_account_id)

            from_account.withdraw(input_dto.amount)
            to_account.deposit(input_dto.amount)

            self._repo.save(from_account)
            self._repo.save(to_account)
            self._uow.commit()

            self._events.publish('money.transferred', {...})

        return TransferOutput(...)
```

---

## 10. FAQ

### Q1. クリーンアーキテクチャは小規模プロジェクトでも必要か？

**A.** 小規模・短期プロジェクトでは過剰設計になることが多い。判断基準は以下の通り。

```
適用判断フローチャート:

  プロジェクトの寿命は1年以上？
    ├── No → 従来型 MVC で十分
    └── Yes
         ビジネスルールは複雑？
           ├── No → 簡易3層（Domain/Application/Infrastructure）
           └── Yes
                チームは3人以上？
                  ├── No → 簡易3層 + Port 定義
                  └── Yes → フルのクリーンアーキテクチャ
```

レイヤー数を減らした「簡易版」を検討する。例えば Domain + Application + Infrastructure の3層に簡略化し、プロジェクトの成長に合わせて分離度を高める段階的アプローチが現実的。CRUD 中心の管理画面なら従来型 MVC で十分な場合もある。

### Q2. DTO とエンティティの変換が面倒だが省略できるか？

**A.** レイヤー間のデータ変換はクリーンアーキテクチャの**本質的なコスト**である。省略するとレイヤー間の結合度が上がり、変更時の影響範囲が拡大する。

ただし、変換コードの記述量は以下の方法で削減できる。

```python
# 方法1: dataclass の asdict() を活用
from dataclasses import asdict

output = CreateOrderOutput(**{
    k: v for k, v in asdict(order).items()
    if k in CreateOrderOutput.__dataclass_fields__
})

# 方法2: マッピング関数を共通化
def map_to_output(order: Order) -> CreateOrderOutput:
    return CreateOrderOutput(
        order_id=order.id,
        total_amount=order.total_amount,
        item_count=order.item_count,
        status=order.status.value,
    )

# 方法3: pydantic の model_validate (v2) を活用
class CreateOrderOutput(BaseModel):
    @classmethod
    def from_entity(cls, order: Order) -> 'CreateOrderOutput':
        return cls(
            order_id=order.id,
            total_amount=order.total_amount,
            item_count=order.item_count,
            status=order.status.value,
        )
```

変換の価値は「テスト容易性」と「変更容易性」で回収される。プロジェクト初期は面倒に感じるが、6ヶ月後にフレームワーク移行やDB変更が発生した際に、その価値を実感する。

### Q3. DI コンテナは必須か？

**A.** 必須ではないが推奨。小規模なら手動 DI（コンストラクタ注入）で十分。

```python
# 手動 DI（小規模プロジェクト向け）
def create_order_use_case() -> CreateOrderUseCase:
    session = create_session()
    return CreateOrderUseCase(
        order_repo=PostgresOrderRepository(session),
        product_repo=PostgresProductRepository(session),
        event_publisher=KafkaEventPublisher(),
        id_generator=UUIDGenerator(),
    )

# DI コンテナ（中〜大規模プロジェクト向け）
# dependency-injector (Python), tsyringe (TypeScript)
# → 依存関係の一元管理、環境切り替えが容易
```

プロジェクト規模が大きくなったら `dependency-injector` (Python) や `tsyringe` (TypeScript) などの DI コンテナで依存関係を一元管理すると、設定変更や環境切り替え（テスト用 Fake / 本番用実装）が容易になる。

### Q4. Entity が複数の UseCase から呼ばれる場合の設計は？

**A.** Entity は複数の UseCase から呼ばれることが自然である。Entity はビジネスルールの Single Source of Truth であり、同じルール（例: 「出荷済み注文はキャンセル不可」）は Entity に一度だけ実装すればよい。UseCase は Entity のメソッドを呼び出してオーケストレーションに徹する。

```python
# Entity は1つ、UseCase は複数
class Order:
    def cancel(self):
        # キャンセルのビジネスルールは Entity に1箇所だけ
        ...

class UserCancelOrderUseCase:     # ユーザー起点のキャンセル
    def execute(self, ...):
        order.cancel()

class AdminCancelOrderUseCase:    # 管理者起点のキャンセル（追加処理あり）
    def execute(self, ...):
        order.cancel()
        self._notify_user(order)  # 管理者キャンセル固有の処理
```

### Q5. クリーンアーキテクチャでパフォーマンスは犠牲にならないか？

**A.** レイヤー間のデータ変換によるオーバーヘッドは存在するが、通常は無視できるレベル（DTO変換で数マイクロ秒）。パフォーマンスのボトルネックは DB クエリやネットワーク通信（数ミリ秒〜数百ミリ秒）であり、レイヤー間変換のコストは2〜3桁小さい。

ただし、大量データの一括処理（バッチインポート等）では、全データを Entity に変換するコストが累積する。そのような場合は CQRS の Query 側で直接 DB にアクセスする設計が妥当。

### Q6. クリーンアーキテクチャと DDD の関係は？

**A.** クリーンアーキテクチャは**レイヤー構造**を定義し、DDD は**ドメインモデリングの手法**を提供する。両者は補完関係にある。

```
クリーンアーキテクチャ: 「依存性をどの方向に向けるか」を定義
DDD:                  「Entities 層に何を実装するか」を定義

クリーン + DDD の組み合わせ:
  Entities 層 = DDD の集約、値オブジェクト、ドメインイベント
  Use Cases 層 = DDD のアプリケーションサービス
  Adapters 層 = DDD のリポジトリ実装
```

### Q7. 既存プロジェクトにクリーンアーキテクチャを導入するには？

**A.** ビッグバンリライトは避け、段階的に導入する。推奨手順は以下の通り。

```
Step 1: 新機能からクリーンアーキテクチャを適用
  → 既存コードは触らず、新しいユースケースだけ Clean で書く

Step 2: テストを追加しながら Entity を抽出
  → 既存の Model / Service からビジネスルールを Entity に移動

Step 3: Repository Interface を導入
  → 既存の DB アクセスコードを Repository 実装でラップ

Step 4: Controller を UseCase 経由に変更
  → API エンドポイントを1つずつ UseCase 経由に移行

目安期間: 中規模プロジェクトで3〜6ヶ月
```

---

## 11. まとめ

| 項目 | ポイント |
|------|---------|
| 依存性ルール | 外側から内側にのみ依存。内側は外側を知らない。これが最重要ルール |
| Entities | ビジネスルールの核。フレームワーク非依存。変更頻度が最も低い |
| Use Cases | アプリケーションロジック。Port（インターフェース）に依存。Entity の調整役 |
| Adapters | DTO 変換、HTTP ↔ UseCase、DB ↔ Entity の橋渡し。技術詳細の吸収層 |
| Frameworks | 技術詳細のプラグイン。最も変更頻度が高く、最も外側に位置する |
| テスト戦略 | 内側のレイヤーほどユニットテストが容易。Fake 実装で DB 不要 |
| DI | 依存性の組み立ては DI コンテナで一元管理。テスト/本番の切り替えが容易 |
| トレードオフ | 初期コスト（DTO 変換、レイヤー構築）vs 長期的な保守性・テスト容易性 |
| 段階的導入 | 小規模は3層、中規模は4層、大規模は CQRS 併用で段階的に適用 |
| 類似パターン | ヘキサゴナル、オニオンとほぼ同等。従来型 MVC とは根本的に異なる |

---

## 次に読むべきガイド

- [DDD（ドメイン駆動設計）](./02-ddd.md) — 集約と境界づけられたコンテキストによるドメインモデリング。クリーンアーキテクチャの Entities 層の設計手法
- [イベント駆動アーキテクチャ](./03-event-driven.md) — Pub/Sub による疎結合な連携。マイクロサービスでの適用
- [テスト原則](../../clean-code-principles/docs/01-practices/04-testing-principles.md) — AAA・FIRST によるテスト設計。クリーンアーキテクチャのテスト戦略の基礎
- [SOLID 原則](../../clean-code-principles/docs/00-principles/01-solid.md) — 依存性逆転の原則（DIP）の詳細。クリーンアーキテクチャの理論的基盤
- [Repository パターン](../../design-patterns-guide/docs/04-architectural/) — Port/Adapter の具体的な実装パターン
- [システム設計の基礎](../00-fundamentals/) — スケーラビリティ、可用性など非機能要件の基礎

---

## 参考文献

1. **Clean Architecture: A Craftsman's Guide to Software Structure and Design** — Robert C. Martin (Prentice Hall, 2017) — クリーンアーキテクチャの原典。同心円モデルと依存性ルールの詳細な解説
2. **Hexagonal Architecture (Ports and Adapters)** — Alistair Cockburn — https://alistair.cockburn.us/hexagonal-architecture/ — クリーンアーキテクチャの原型となった Ports & Adapters パターン
3. **Architecture Patterns with Python** — Harry Percival & Bob Gregory (O'Reilly, 2020) — Python でのクリーンアーキテクチャ実践。Repository パターン、Unit of Work の実装例
4. **The Clean Architecture (Blog Post)** — Robert C. Martin (2012) — https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html — クリーンアーキテクチャの初出ブログ記事
5. **Implementing Domain-Driven Design** — Vaughn Vernon (Addison-Wesley, 2013) — DDD とクリーンアーキテクチャの組み合わせ
6. **Clean Code: A Handbook of Agile Software Craftsmanship** — Robert C. Martin (Prentice Hall, 2008) — クリーンアーキテクチャの前提となるクリーンコードの原則

# ドメイン駆動設計 (DDD)

> 複雑なビジネスドメインをソフトウェアに正確に反映するための設計手法であり、集約・境界づけられたコンテキスト・ユビキタス言語を軸に、ドメインエキスパートと開発者が共通理解のもとで堅牢なモデルを構築する方法論を解説する

---

## この章で学ぶこと

1. **戦略的設計** — 境界づけられたコンテキスト、コンテキストマッピング、ユビキタス言語の確立方法を理解する
2. **戦術的設計** — エンティティ、値オブジェクト、集約、ドメインイベント、リポジトリの実装パターンを習得する
3. **集約設計の原則** — 集約ルート、トランザクション境界、整合性の保証を実装できるようになる
4. **ドメインサービスとアプリケーションサービスの使い分け** — ロジックの配置先を判断できるようになる
5. **結果整合性と Saga パターン** — 集約間の連携を設計できるようになる

---

## 前提知識

| トピック | 内容 | 参照リンク |
|---------|------|-----------|
| SOLID 原則 | 単一責任の原則、依存性逆転の原則 | [SOLID 原則](../../clean-code-principles/docs/00-principles/01-solid.md) |
| クリーンアーキテクチャ | レイヤー構造、依存性ルール | [クリーンアーキテクチャ](./01-clean-architecture.md) |
| Repository パターン | 永続化の抽象化 | [デザインパターン](../../design-patterns-guide/docs/04-architectural/) |
| Python 基礎 | dataclass、Protocol、型ヒント | - |

---

## 1. DDD の背景と哲学

### 1.1 なぜ DDD が必要なのか

ソフトウェア開発における最大の課題は、**技術的な複雑さ**ではなく**ドメイン（業務領域）の複雑さ**である。Eric Evans は2003年の著書「Domain-Driven Design」で、この本質的な複雑さに立ち向かうための体系的な手法を提唱した。

```
ソフトウェアプロジェクトの失敗原因

  技術的問題（パフォーマンス、スケーラビリティ等）
  ├── 解決策: 技術的知識、経験、ベストプラクティス
  └── 対処しやすい（明確な指標がある）

  ドメインの複雑さ（ビジネスルール、業務フロー等）
  ├── 解決策: ドメイン駆動設計
  └── 対処しにくい（暗黙知が多い、要件が変わる）
       → DDD はここに焦点を当てる
```

**WHY: 従来のアプローチの限界**

従来のソフトウェア開発では、ドメインの知識はドキュメント（要件定義書、ER図）に記述され、開発者はそれを「翻訳」してコードに落とし込んでいた。しかし、この「翻訳」の過程で情報が失われ、ドメインの複雑さがコードに正確に反映されないという問題が生じた。

DDD の核心は、**ドメインの知識をコードそのものに表現する**ことである。ドメインエキスパートが使う言葉をそのままクラス名・メソッド名にし、ビジネスルールをエンティティに直接実装する。これにより「ドキュメントとコードの乖離」問題を根本的に解消する。

### 1.2 DDD の全体像

```
DDD の2つの柱

【戦略的設計 (Strategic Design)】
  ─ 問題領域を分割し、チーム・システム境界を定義
  ─ 境界づけられたコンテキスト (Bounded Context)
  ─ コンテキストマップ（コンテキスト間の関係）
  ─ ユビキタス言語（共通言語の確立）
  ─ サブドメイン分類（コア/サポート/汎用）

【戦術的設計 (Tactical Design)】
  ─ コンテキスト内部のモデリングパターン
  ─ エンティティ / 値オブジェクト / 集約
  ─ ドメインサービス / ドメインイベント
  ─ リポジトリ / ファクトリ
  ─ 仕様パターン (Specification)
```

```
DDD の適用判断フロー

  ビジネスルールが複雑？
    ├── No → CRUD + 従来型アーキテクチャで十分
    └── Yes
         ドメインエキスパートにアクセスできる？
           ├── No → 戦術的パターンのみ部分採用
           └── Yes
                チーム規模は十分（3人以上）？
                  ├── No → 戦略的設計は簡易化、戦術的設計に注力
                  └── Yes → フル DDD を採用
```

---

## 2. 戦略的設計

### 2.1 境界づけられたコンテキスト (Bounded Context)

境界づけられたコンテキストは、DDD において**最も重要な概念**である。同じ用語でもコンテキストによって意味が異なることを明示的に認め、各コンテキスト内でユビキタス言語を統一する。

```
ECサイトのコンテキストマップ

  +--------------------+     +--------------------+     +--------------------+
  |  注文コンテキスト    |     |  在庫コンテキスト    |     |  配送コンテキスト    |
  |                    |     |                    |     |                    |
  | Order              |     | StockItem          |     | Shipment           |
  | OrderItem          |     | Warehouse          |     | DeliveryRoute      |
  | Customer(注文者)    |     | Reservation        |     | Customer(届け先)    |
  | 「確定する」= place |     | 「引当する」= reserve|     | 「発送する」= ship  |
  +--------+-----------+     +--------+-----------+     +--------+-----------+
           |                          |                          |
           | OrderPlaced              | StockReserved            |
           | (ドメインイベント)         | (ドメインイベント)         |
           +--------> [Event Bus] <---+------------------------->+

  ★ 同じ「Customer」でも各コンテキストで意味と属性が異なる
     注文: 名前、連絡先、注文履歴
     配送: 届け先住所、配達時間帯指定
  ★ コンテキスト間はイベントで疎結合に連携
  ★ 各コンテキストは独立してデプロイ・開発可能
```

**WHY 同じ概念を複数のコンテキストに分けるのか？**

1つの「Customer」クラスに全ての属性（注文情報、配送先情報、ポイント情報、問い合わせ履歴...）を持たせると、巨大で理解困難なクラスになる。さらに、注文チームの変更が配送チームに影響するなど、チーム間の依存関係が増大する。コンテキストを分割することで、各チームは自分のコンテキスト内の「Customer」だけに責任を持てばよい。

### 2.2 コンテキストマップのパターン

```
コンテキスト間の関係パターン

1. Shared Kernel（共有カーネル）
   ┌─────────┐   共有部分   ┌─────────┐
   │ Context A├────┤ Shared ├────┤ Context B│
   └─────────┘   └────────┘   └─────────┘
   → 2つのコンテキストが一部のモデルを共有
   → 変更時は両チームの合意が必要
   → 密結合になるため、使用は最小限に

2. Customer-Supplier（顧客-供給者）
   ┌──────────┐  API  ┌──────────┐
   │ Supplier  ├──────>│ Customer  │
   │(供給者)   │       │(顧客)     │
   └──────────┘       └──────────┘
   → 上流（Supplier）が下流（Customer）の要求を考慮
   → 下流チームが上流チームに要件を伝える

3. Conformist（準拠者）
   ┌──────────┐  API  ┌──────────┐
   │ Upstream  ├──────>│ Downstream│
   │(変更不可) │       │(準拠する) │
   └──────────┘       └──────────┘
   → 外部サービスのモデルにそのまま従う
   → 変換コストを許容できない場合

4. Anti-Corruption Layer（腐敗防止層）
   ┌──────────┐  ACL  ┌──────────┐
   │ External  ├──┤変換├──>│ Internal  │
   │ System    │  └──┘   │ Context  │
   └──────────┘          └──────────┘
   → 外部システムのモデルを自コンテキストのモデルに変換
   → レガシーシステムとの統合で特に重要

5. Published Language（公開言語）
   → 共通のスキーマ（JSON Schema, Protobuf等）で通信
   → イベント駆動アーキテクチャと相性が良い
```

### 2.3 サブドメインの分類

```
サブドメイン分類

  ┌─────────────────────────────────────────────────────┐
  │  コアドメイン (Core Domain)                          │
  │  ・ビジネスの競争優位性の源泉                        │
  │  ・最も複雑、最も重要                               │
  │  ・最高のチームを投入すべき領域                      │
  │  ・例: EC での「レコメンデーション」「価格最適化」    │
  ├─────────────────────────────────────────────────────┤
  │  サポートドメイン (Supporting Subdomain)              │
  │  ・コアを支える必要な機能                            │
  │  ・ビジネス固有だがコアほど重要ではない               │
  │  ・外部委託可能だが、カスタマイズは必要               │
  │  ・例: EC での「在庫管理」「配送管理」               │
  ├─────────────────────────────────────────────────────┤
  │  汎用ドメイン (Generic Subdomain)                    │
  │  ・どの企業でも共通の機能                            │
  │  ・既存ソリューション（SaaS、OSS）で代替可能         │
  │  ・自社開発する意味がない領域                        │
  │  ・例: 「認証」「メール送信」「ファイルストレージ」   │
  └─────────────────────────────────────────────────────┘

  投資配分の指針:
    コアドメイン:    70% のリソース → DDD フル適用
    サポートドメイン: 20% のリソース → DDD 戦術パターンのみ
    汎用ドメイン:    10% のリソース → 既存ソリューション採用
```

### 2.4 ユビキタス言語

```python
# ユビキタス言語の例: ECサイトの注文コンテキスト

# NG: 技術者の言葉でモデリング
class OrderData:
    def update_status(self, new_status: int):  # ステータスが数値
        self.status_code = new_status

# OK: ドメインエキスパートの言葉でモデリング
class Order:
    """注文（ちゅうもん）: 顧客が商品を購入する意思表示"""

    def place(self) -> None:
        """注文を確定する（かくていする）"""
        # ドメインエキスパートが「注文を確定する」と言う
        # → メソッド名は place_order ではなく place
        ...

    def cancel(self) -> None:
        """注文を取り消す（とりけす）"""
        ...

    def ship(self) -> None:
        """注文を出荷する（しゅっかする）"""
        ...

# ユビキタス言語辞書（コンテキストごとに定義）
"""
注文コンテキスト用語集:
  注文 (Order):       顧客が商品を購入する意思表示
  確定 (Place):       注文内容を確定し、処理を開始する行為
  取り消し (Cancel):  確定前または確定後の注文を無効にする行為
  明細 (OrderItem):   注文に含まれる個々の商品と数量の組
  顧客 (Customer):    注文を行う主体（名前、連絡先を持つ）
"""
```

---

## 3. 戦術パターンの実装

### 3.1 集約の構造

```
   集約 (Aggregate)
  +---------------------------------------------+
  |  [Order] ← 集約ルート (Aggregate Root)       |
  |     |                                        |
  |     +-- OrderItem (値オブジェクト/エンティティ)|
  |     +-- OrderItem                            |
  |     +-- ShippingAddress (値オブジェクト)       |
  |     +-- PaymentInfo (値オブジェクト)           |
  +---------------------------------------------+

  ルール:
  1. 外部から集約内部への直接アクセス禁止
  2. 全ての操作は集約ルート経由
  3. 1トランザクション = 1集約の変更
  4. 集約間の参照は ID のみ
  5. 集約は可能な限り小さく保つ
```

**集約設計の判断基準:**

```
集約の境界を決める質問:

  Q1: これらのオブジェクトは必ず一緒に変更されるか？
    → Yes なら同じ集約内に
    → No なら別の集約に

  Q2: これらのオブジェクト間で強い整合性（即時一貫性）が必要か？
    → Yes なら同じ集約内に
    → No（結果整合性で十分）なら別の集約に

  Q3: この集約は1つのトランザクションで更新できるサイズか？
    → Yes → OK
    → No → 集約が大きすぎる、分割を検討
```

### 3.2 値オブジェクト (Value Object)

```python
# domain/value_objects/money.py
from dataclasses import dataclass

@dataclass(frozen=True)     # 不変 (Immutable)
class Money:
    """
    金額を表す値オブジェクト

    値オブジェクトの特徴:
    1. 不変 (Immutable): 一度作成したら変更不可
    2. 値で等価判定: IDではなく全属性が同じなら同一
    3. 副作用なし: 操作は新しいオブジェクトを返す
    4. 自己検証: 生成時にバリデーション
    """
    amount: int              # 最小単位（円）
    currency: str = "JPY"

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError(f"金額は0以上: {self.amount}")
        if not self.currency:
            raise ValueError("通貨コードは必須")

    def add(self, other: 'Money') -> 'Money':
        """加算: 新しい Money を返す（元のオブジェクトは不変）"""
        if self.currency != other.currency:
            raise ValueError(
                f"通貨が異なります: {self.currency} vs {other.currency}"
            )
        return Money(
            amount=self.amount + other.amount,
            currency=self.currency,
        )

    def subtract(self, other: 'Money') -> 'Money':
        """減算"""
        if self.currency != other.currency:
            raise ValueError(
                f"通貨が異なります: {self.currency} vs {other.currency}"
            )
        if self.amount < other.amount:
            raise ValueError(
                f"負の金額は不正: {self.amount} - {other.amount}"
            )
        return Money(
            amount=self.amount - other.amount,
            currency=self.currency,
        )

    def multiply(self, factor: int) -> 'Money':
        """乗算"""
        return Money(amount=self.amount * factor, currency=self.currency)

    def is_greater_than(self, other: 'Money') -> bool:
        """比較"""
        if self.currency != other.currency:
            raise ValueError("通貨が異なります")
        return self.amount > other.amount

    def __str__(self) -> str:
        if self.currency == "JPY":
            return f"¥{self.amount:,}"
        return f"{self.amount / 100:.2f} {self.currency}"


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
        if not self.prefecture:
            raise ValueError("都道府県は必須")
        if not self.city:
            raise ValueError("市区町村は必須")

    @property
    def full_address(self) -> str:
        parts = [self.prefecture, self.city, self.street]
        if self.building:
            parts.append(self.building)
        return " ".join(parts)


# domain/value_objects/email.py
import re

@dataclass(frozen=True)
class EmailAddress:
    """メールアドレスを表す値オブジェクト"""
    value: str

    def __post_init__(self):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, self.value):
            raise ValueError(f"無効なメールアドレス: {self.value}")

    @property
    def domain(self) -> str:
        return self.value.split('@')[1]

    @property
    def local_part(self) -> str:
        return self.value.split('@')[0]

    def __str__(self) -> str:
        return self.value


# domain/value_objects/quantity.py
@dataclass(frozen=True)
class Quantity:
    """数量を表す値オブジェクト"""
    value: int

    def __post_init__(self):
        if self.value < 0:
            raise ValueError(f"数量は0以上: {self.value}")

    def add(self, other: 'Quantity') -> 'Quantity':
        return Quantity(value=self.value + other.value)

    def subtract(self, other: 'Quantity') -> 'Quantity':
        if self.value < other.value:
            raise ValueError("在庫不足")
        return Quantity(value=self.value - other.value)

    def is_zero(self) -> bool:
        return self.value == 0
```

### 3.3 エンティティと集約ルート

```python
# domain/entities/order.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional
from domain.value_objects.money import Money
from domain.value_objects.address import Address

class OrderStatus(Enum):
    DRAFT = "draft"
    PLACED = "placed"
    PAID = "paid"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

@dataclass
class OrderItem:
    """
    注文明細エンティティ（集約内でのみ使用）

    注意: OrderItem は集約外から直接アクセスされない。
    全ての操作は Order（集約ルート）経由で行う。
    """
    id: str
    product_id: str            # 他の集約への参照は ID のみ
    product_name: str
    unit_price: Money
    quantity: int

    def __post_init__(self):
        if self.quantity < 1:
            raise ValueError(f"数量は1以上: {self.quantity}")

    @property
    def subtotal(self) -> Money:
        return self.unit_price.multiply(self.quantity)

    def change_quantity(self, new_quantity: int) -> None:
        """数量変更（集約ルート経由でのみ呼ばれる）"""
        if new_quantity < 1:
            raise ValueError(f"数量は1以上: {new_quantity}")
        self.quantity = new_quantity


@dataclass
class Order:
    """
    注文集約ルート

    設計方針:
    - 全てのビジネスルールはこのクラス内に閉じる
    - 集約内部のオブジェクト（OrderItem）への操作は
      必ずこのクラスのメソッド経由で行う
    - ドメインイベントを生成して外部に変更を通知する
    """
    id: str
    customer_id: str              # 他の集約への参照は ID のみ
    items: List[OrderItem] = field(default_factory=list)
    shipping_address: Optional[Address] = None
    status: OrderStatus = OrderStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    _domain_events: List = field(default_factory=list, repr=False)

    # --- ビジネスルール定数 ---
    MAX_ITEMS = 50
    MIN_ORDER_AMOUNT = Money(100)   # 最低注文金額: 100円

    # --- 集約の不変条件 (Invariants) ---

    @property
    def total_amount(self) -> Money:
        """合計金額を計算"""
        total = Money(0)
        for item in self.items:
            total = total.add(item.subtotal)
        return total

    @property
    def item_count(self) -> int:
        return len(self.items)

    # --- コマンド（状態を変更する操作）---

    def add_item(self, item: OrderItem) -> None:
        """注文明細を追加する"""
        if self.status != OrderStatus.DRAFT:
            raise ValueError("下書き状態でのみ明細追加可能")
        if len(self.items) >= self.MAX_ITEMS:
            raise ValueError(f"1注文あたり最大{self.MAX_ITEMS}明細")

        # 同一商品がある場合は数量を加算
        existing = self._find_item_by_product(item.product_id)
        if existing:
            existing.change_quantity(existing.quantity + item.quantity)
        else:
            self.items.append(item)
        self.updated_at = datetime.now()

    def remove_item(self, product_id: str) -> None:
        """注文明細を削除する"""
        if self.status != OrderStatus.DRAFT:
            raise ValueError("下書き状態でのみ明細削除可能")
        existing = self._find_item_by_product(product_id)
        if not existing:
            raise ValueError(f"該当する明細がありません: {product_id}")
        self.items.remove(existing)
        self.updated_at = datetime.now()

    def set_shipping_address(self, address: Address) -> None:
        """配送先住所を設定する"""
        if self.status not in (OrderStatus.DRAFT, OrderStatus.PLACED):
            raise ValueError("確定前または確定後にのみ住所変更可能")
        self.shipping_address = address
        self.updated_at = datetime.now()

    def place(self) -> None:
        """注文を確定する"""
        if self.status != OrderStatus.DRAFT:
            raise ValueError(
                f"下書き状態でのみ確定可能（現在: {self.status.value}）"
            )
        if not self.items:
            raise ValueError("明細が空の注文は確定できません")
        if not self.shipping_address:
            raise ValueError("配送先住所が未設定です")
        if not self.total_amount.is_greater_than(self.MIN_ORDER_AMOUNT):
            raise ValueError(
                f"最低注文金額（{self.MIN_ORDER_AMOUNT}）に達していません"
            )

        self.status = OrderStatus.PLACED
        self.updated_at = datetime.now()

        # ドメインイベントを生成
        self._domain_events.append(OrderPlaced(
            order_id=self.id,
            customer_id=self.customer_id,
            total_amount=self.total_amount.amount,
            item_count=self.item_count,
            occurred_at=datetime.now(),
        ))

    def pay(self, payment_id: str) -> None:
        """注文を支払い済みにする"""
        if self.status != OrderStatus.PLACED:
            raise ValueError(
                f"確定済み状態でのみ支払い可能（現在: {self.status.value}）"
            )
        self.status = OrderStatus.PAID
        self.updated_at = datetime.now()
        self._domain_events.append(OrderPaid(
            order_id=self.id,
            payment_id=payment_id,
            amount=self.total_amount.amount,
            occurred_at=datetime.now(),
        ))

    def ship(self, tracking_number: str) -> None:
        """注文を出荷する"""
        if self.status != OrderStatus.PAID:
            raise ValueError(
                f"支払い済み状態でのみ出荷可能（現在: {self.status.value}）"
            )
        self.status = OrderStatus.SHIPPED
        self.updated_at = datetime.now()
        self._domain_events.append(OrderShipped(
            order_id=self.id,
            tracking_number=tracking_number,
            occurred_at=datetime.now(),
        ))

    def cancel(self) -> None:
        """注文をキャンセルする"""
        cancellable = (OrderStatus.DRAFT, OrderStatus.PLACED, OrderStatus.PAID)
        if self.status not in cancellable:
            raise ValueError(
                f"キャンセル不可（現在: {self.status.value}）"
            )
        self.status = OrderStatus.CANCELLED
        self.updated_at = datetime.now()
        self._domain_events.append(OrderCancelled(
            order_id=self.id,
            occurred_at=datetime.now(),
        ))

    # --- ドメインイベント管理 ---

    def collect_events(self) -> List:
        """
        ドメインイベントを回収する

        アプリケーションサービスが呼び出し、
        回収したイベントをイベントバスに発行する。
        """
        events = list(self._domain_events)
        self._domain_events.clear()
        return events

    # --- 内部ヘルパー ---

    def _find_item_by_product(self, product_id: str) -> Optional[OrderItem]:
        return next(
            (i for i in self.items if i.product_id == product_id), None
        )

    # --- 状態遷移図 ---
    # DRAFT → PLACED → PAID → SHIPPED → DELIVERED
    #   ↓       ↓       ↓
    # CANCELLED CANCELLED CANCELLED
```

### 3.4 ドメインイベント

```python
# domain/events/order_events.py
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class DomainEvent:
    """ドメインイベントの基底クラス"""
    occurred_at: datetime

@dataclass(frozen=True)
class OrderPlaced(DomainEvent):
    """注文確定イベント"""
    order_id: str
    customer_id: str
    total_amount: int
    item_count: int

@dataclass(frozen=True)
class OrderPaid(DomainEvent):
    """注文支払い完了イベント"""
    order_id: str
    payment_id: str
    amount: int

@dataclass(frozen=True)
class OrderShipped(DomainEvent):
    """注文出荷イベント"""
    order_id: str
    tracking_number: str

@dataclass(frozen=True)
class OrderCancelled(DomainEvent):
    """注文キャンセルイベント"""
    order_id: str
```

**WHY ドメインイベントを使うのか？**

```
ドメインイベントのメリット:

  1. 集約間の疎結合
     注文確定 → 在庫引当 を直接呼び出すと結合度が上がる
     注文確定 → OrderPlaced イベント発行 → 在庫サービスが購読
     → 注文サービスは在庫サービスの存在を知らない

  2. 監査ログの自動生成
     全てのイベントを記録すれば、何がいつ起きたかを追跡可能
     イベントソーシングへの拡張も容易

  3. 副作用の分離
     注文確定の「核心のロジック」と「通知メール送信」を分離
     → テストが容易になる

  4. 新機能追加の容易さ
     「注文確定時にポイントを付与する」を追加する場合
     → 既存コードを変更せず、新しいイベントハンドラを追加するだけ
```

### 3.5 ドメインサービス

```python
# domain/services/pricing_service.py

class PricingService:
    """
    価格計算ドメインサービス

    WHY ドメインサービス?
    - 割引計算は単一のエンティティに属さない
    - Order と Customer と CouponCode の情報を横断する
    - エンティティのメソッドにするとどれかが肥大化する
    → 複数の集約にまたがるロジックはドメインサービスに配置
    """

    def calculate_discount(
        self,
        order: 'Order',
        customer_tier: str,
        coupon_code: str | None = None,
    ) -> Money:
        """割引額を計算する"""
        base_amount = order.total_amount

        # 会員ランクに基づく割引
        tier_discount_rate = {
            "gold": 0.10,
            "silver": 0.05,
            "bronze": 0.02,
            "regular": 0.00,
        }
        rate = tier_discount_rate.get(customer_tier, 0.00)
        tier_discount = Money(int(base_amount.amount * rate))

        # クーポン割引
        coupon_discount = Money(0)
        if coupon_code:
            coupon_discount = self._apply_coupon(coupon_code, base_amount)

        # 割引の合計（上限: 注文金額の30%）
        total_discount = tier_discount.add(coupon_discount)
        max_discount = Money(int(base_amount.amount * 0.30))

        if total_discount.is_greater_than(max_discount):
            return max_discount
        return total_discount

    def _apply_coupon(self, code: str, amount: Money) -> Money:
        # 実際にはクーポンリポジトリから取得
        coupon_values = {"SAVE500": Money(500), "SAVE1000": Money(1000)}
        return coupon_values.get(code, Money(0))


# domain/services/transfer_service.py

class MoneyTransferService:
    """
    送金ドメインサービス

    WHY ドメインサービス?
    - 送金は2つの Account 集約にまたがる操作
    - Account.withdraw() と Account.deposit() をどの順序で
      呼ぶかの調整は、どちらの Account にも属さない
    """

    def transfer(
        self,
        source: 'Account',
        target: 'Account',
        amount: Money,
    ) -> None:
        if source.id == target.id:
            raise ValueError("同一アカウントへの送金は不可")
        source.withdraw(amount)
        target.deposit(amount)
```

### 3.6 リポジトリ

```python
# domain/repositories/order_repository.py (インターフェース)
from typing import Protocol, Optional, List

class OrderRepository(Protocol):
    """
    注文リポジトリのインターフェース

    注意:
    - リポジトリは集約単位で定義する
    - OrderItem 用のリポジトリは作らない（集約ルート経由）
    - インターフェースはドメイン層、実装はインフラ層に配置
    """
    def save(self, order: Order) -> None: ...
    def find_by_id(self, order_id: str) -> Optional[Order]: ...
    def find_by_customer(self, customer_id: str) -> List[Order]: ...
    def next_id(self) -> str: ...


# infrastructure/repositories/sqlalchemy_order_repository.py (実装)
from sqlalchemy.orm import Session

class SQLAlchemyOrderRepository:
    """OrderRepository の SQLAlchemy 実装"""

    def __init__(self, session: Session):
        self._session = session

    def save(self, order: Order) -> None:
        model = self._to_model(order)
        self._session.merge(model)
        self._session.flush()

    def find_by_id(self, order_id: str) -> Optional[Order]:
        model = self._session.query(OrderModel).get(order_id)
        return self._to_entity(model) if model else None

    def find_by_customer(self, customer_id: str) -> List[Order]:
        models = (
            self._session.query(OrderModel)
            .filter(OrderModel.customer_id == customer_id)
            .order_by(OrderModel.created_at.desc())
            .all()
        )
        return [self._to_entity(m) for m in models]

    def next_id(self) -> str:
        import uuid
        return str(uuid.uuid4())

    def _to_model(self, order: Order) -> 'OrderModel':
        """ドメインエンティティ → DBモデル"""
        return OrderModel(
            id=order.id,
            customer_id=order.customer_id,
            status=order.status.value,
            total_amount=order.total_amount.amount,
            shipping_postal_code=order.shipping_address.postal_code if order.shipping_address else None,
            shipping_prefecture=order.shipping_address.prefecture if order.shipping_address else None,
            shipping_city=order.shipping_address.city if order.shipping_address else None,
            shipping_street=order.shipping_address.street if order.shipping_address else None,
            created_at=order.created_at,
            updated_at=order.updated_at,
            items=[
                OrderItemModel(
                    id=item.id,
                    product_id=item.product_id,
                    product_name=item.product_name,
                    unit_price=item.unit_price.amount,
                    quantity=item.quantity,
                )
                for item in order.items
            ],
        )

    def _to_entity(self, model: 'OrderModel') -> Order:
        """DBモデル → ドメインエンティティ"""
        shipping_address = None
        if model.shipping_postal_code:
            shipping_address = Address(
                postal_code=model.shipping_postal_code,
                prefecture=model.shipping_prefecture,
                city=model.shipping_city,
                street=model.shipping_street,
            )

        return Order(
            id=model.id,
            customer_id=model.customer_id,
            items=[
                OrderItem(
                    id=item.id,
                    product_id=item.product_id,
                    product_name=item.product_name,
                    unit_price=Money(item.unit_price),
                    quantity=item.quantity,
                )
                for item in model.items
            ],
            shipping_address=shipping_address,
            status=OrderStatus(model.status),
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
```

### 3.7 アプリケーションサービス

```python
# application/services/order_service.py

class PlaceOrderService:
    """
    注文確定アプリケーションサービス

    アプリケーションサービスの責務:
    - トランザクション管理
    - リポジトリからの集約取得
    - ドメインロジックの呼び出し（調整役）
    - ドメインイベントの発行
    - 例外のハンドリング

    注意: ビジネスルールはここに書かない！
    ビジネスルールは Entity / ドメインサービスに配置する。
    """

    def __init__(
        self,
        order_repo: OrderRepository,
        event_publisher: EventPublisher,
        unit_of_work: UnitOfWork,
    ):
        self._order_repo = order_repo
        self._events = event_publisher
        self._uow = unit_of_work

    def execute(self, order_id: str) -> PlaceOrderOutput:
        with self._uow:
            # 1. 集約を取得
            order = self._order_repo.find_by_id(order_id)
            if not order:
                raise OrderNotFoundError(order_id)

            # 2. ドメインロジック実行（Entity に委譲）
            order.place()

            # 3. 永続化
            self._order_repo.save(order)

            # 4. ドメインイベント発行
            for event in order.collect_events():
                self._events.publish(event)

            # 5. コミット
            self._uow.commit()

        return PlaceOrderOutput(
            order_id=order.id,
            status=order.status.value,
            total_amount=order.total_amount.amount,
        )


class CreateOrderService:
    """注文作成アプリケーションサービス"""

    def __init__(
        self,
        order_repo: OrderRepository,
        product_repo: ProductRepository,
        unit_of_work: UnitOfWork,
    ):
        self._order_repo = order_repo
        self._product_repo = product_repo
        self._uow = unit_of_work

    def execute(self, input_dto: CreateOrderInput) -> CreateOrderOutput:
        with self._uow:
            order_id = self._order_repo.next_id()

            # 商品情報を取得して OrderItem を構築
            items = []
            for item_input in input_dto.items:
                product = self._product_repo.find_by_id(item_input.product_id)
                if not product:
                    raise ProductNotFoundError(item_input.product_id)
                items.append(OrderItem(
                    id=f"{order_id}-{len(items)+1}",
                    product_id=product.id,
                    product_name=product.name,
                    unit_price=Money(product.price),
                    quantity=item_input.quantity,
                ))

            # 集約生成
            order = Order(id=order_id, customer_id=input_dto.customer_id)
            for item in items:
                order.add_item(item)

            if input_dto.shipping_address:
                order.set_shipping_address(input_dto.shipping_address)

            # 永続化
            self._order_repo.save(order)
            self._uow.commit()

        return CreateOrderOutput(
            order_id=order.id,
            item_count=order.item_count,
            total_amount=order.total_amount.amount,
            status=order.status.value,
        )
```

### 3.8 ファクトリパターン

```python
# domain/factories/order_factory.py

class OrderFactory:
    """
    注文ファクトリ

    WHY ファクトリ?
    - 複雑な集約の生成ロジックをカプセル化
    - 生成時の不変条件（バリデーション）を一箇所に集約
    - テスト時にファクトリを差し替えることも可能
    """

    def __init__(self, id_generator: IdGenerator):
        self._id_gen = id_generator

    def create_from_cart(
        self,
        customer_id: str,
        cart_items: List[CartItemDTO],
        shipping_address: Address,
    ) -> Order:
        """ショッピングカートから注文を生成する"""
        order_id = self._id_gen.generate()
        order = Order(
            id=order_id,
            customer_id=customer_id,
            shipping_address=shipping_address,
        )

        for cart_item in cart_items:
            order.add_item(OrderItem(
                id=f"{order_id}-{cart_item.product_id}",
                product_id=cart_item.product_id,
                product_name=cart_item.product_name,
                unit_price=Money(cart_item.price),
                quantity=cart_item.quantity,
            ))

        return order

    def reconstitute(
        self,
        id: str,
        customer_id: str,
        items: List[dict],
        status: str,
        **kwargs,
    ) -> Order:
        """永続化されたデータから集約を再構築する"""
        # リポジトリの _to_entity で使用
        order = Order(
            id=id,
            customer_id=customer_id,
            status=OrderStatus(status),
            **kwargs,
        )
        # 再構築時はバリデーションをスキップ
        order.items = [
            OrderItem(**item_data) for item_data in items
        ]
        return order
```

---

## 4. 結果整合性と Saga パターン

### 4.1 結果整合性 (Eventual Consistency)

```
集約間の整合性: 結果整合性が基本

  [注文コンテキスト]              [在庫コンテキスト]
  ┌─────────────┐              ┌─────────────┐
  │  Order 確定   │              │  Stock 引当   │
  │  (即座に完了) │              │  (非同期)     │
  └──────┬──────┘              └──────┬──────┘
         │                            │
         │  OrderPlaced イベント        │
         └──────────────────────────>│
                                      │ StockReserved or
                                      │ StockReserveFailed
                                      └──────────────>...

  強い整合性（同一トランザクション）:
    → 集約内のオブジェクト間のみ
    → Order と OrderItem は常に整合

  結果整合性（非同期イベント）:
    → 集約間
    → Order 確定と在庫引当は別トランザクション
    → 一時的に不整合が生じるが、最終的に整合
```

### 4.2 Saga パターン

```python
# application/sagas/order_saga.py

class OrderSaga:
    """
    注文 Saga: 複数の集約にまたがるビジネスプロセスを管理

    フロー:
    1. 注文確定 → OrderPlaced
    2. 在庫引当 → StockReserved or StockReserveFailed
    3. 決済処理 → PaymentCompleted or PaymentFailed
    4. 出荷指示 → ShipmentCreated

    いずれかのステップが失敗した場合、
    それまでのステップを補償（ロールバック）する。
    """

    def __init__(
        self,
        order_repo: OrderRepository,
        inventory_service: InventoryService,
        payment_service: PaymentService,
        event_publisher: EventPublisher,
    ):
        self._order_repo = order_repo
        self._inventory = inventory_service
        self._payment = payment_service
        self._events = event_publisher

    def handle_order_placed(self, event: OrderPlaced) -> None:
        """注文確定イベントを処理"""
        try:
            # Step 1: 在庫引当
            reservation_id = self._inventory.reserve(
                order_id=event.order_id,
                items=self._get_order_items(event.order_id),
            )

            # Step 2: 決済処理
            payment_id = self._payment.charge(
                customer_id=event.customer_id,
                amount=event.total_amount,
            )

            # Step 3: 注文に支払い情報を記録
            order = self._order_repo.find_by_id(event.order_id)
            order.pay(payment_id)
            self._order_repo.save(order)

        except InventoryError:
            # 在庫引当失敗 → 注文キャンセル
            self._cancel_order(event.order_id, "在庫不足")

        except PaymentError:
            # 決済失敗 → 在庫引当を解放してから注文キャンセル
            self._inventory.release(event.order_id)
            self._cancel_order(event.order_id, "決済失敗")

    def _cancel_order(self, order_id: str, reason: str) -> None:
        """注文を補償キャンセルする"""
        order = self._order_repo.find_by_id(order_id)
        if order:
            order.cancel()
            self._order_repo.save(order)
            self._events.publish(OrderCancelled(
                order_id=order_id,
                occurred_at=datetime.now(),
            ))
```

```
Saga パターンの補償フロー:

  正常系:
    注文確定 → 在庫引当 → 決済完了 → 出荷指示

  在庫引当失敗:
    注文確定 → 在庫引当(失敗) → 注文キャンセル(補償)

  決済失敗:
    注文確定 → 在庫引当 → 決済(失敗)
      → 在庫解放(補償) → 注文キャンセル(補償)

  出荷失敗:
    注文確定 → 在庫引当 → 決済完了 → 出荷(失敗)
      → 返金処理(補償) → 在庫解放(補償) → 注文キャンセル(補償)
```

---

## 5. テスト

```python
# tests/unit/test_order.py
import pytest
from datetime import datetime

class TestOrder:
    """Order 集約のテスト"""

    def _make_item(self, **kwargs) -> OrderItem:
        defaults = {
            'id': 'item-1',
            'product_id': 'prod-1',
            'product_name': 'テスト商品',
            'unit_price': Money(1000),
            'quantity': 2,
        }
        defaults.update(kwargs)
        return OrderItem(**defaults)

    def _make_order(self, **kwargs) -> Order:
        defaults = {
            'id': 'order-1',
            'customer_id': 'cust-1',
        }
        defaults.update(kwargs)
        return Order(**defaults)

    def test_明細追加で合計金額が計算される(self):
        order = self._make_order()
        order.add_item(self._make_item(
            unit_price=Money(1000), quantity=2
        ))
        order.add_item(self._make_item(
            id='item-2', product_id='prod-2',
            unit_price=Money(500), quantity=3,
        ))
        assert order.total_amount == Money(3500)

    def test_同一商品の追加で数量が加算される(self):
        order = self._make_order()
        order.add_item(self._make_item(quantity=2))
        order.add_item(self._make_item(quantity=3))
        assert len(order.items) == 1
        assert order.items[0].quantity == 5

    def test_下書き以外の状態で明細追加不可(self):
        order = self._make_order()
        order.add_item(self._make_item())
        order.set_shipping_address(Address(
            postal_code='1000001', prefecture='東京都',
            city='千代田区', street='丸の内1-1-1',
        ))
        order.place()
        with pytest.raises(ValueError, match="下書き状態でのみ"):
            order.add_item(self._make_item(product_id='prod-2'))

    def test_注文確定でイベントが生成される(self):
        order = self._make_order()
        order.add_item(self._make_item())
        order.set_shipping_address(Address(
            postal_code='1000001', prefecture='東京都',
            city='千代田区', street='丸の内1-1-1',
        ))
        order.place()

        events = order.collect_events()
        assert len(events) == 1
        assert isinstance(events[0], OrderPlaced)
        assert events[0].order_id == 'order-1'

    def test_出荷済みの注文はキャンセル不可(self):
        order = self._make_order()
        order.add_item(self._make_item())
        order.set_shipping_address(Address(
            postal_code='1000001', prefecture='東京都',
            city='千代田区', street='丸の内1-1-1',
        ))
        order.place()
        order.pay("pay-1")
        order.ship("track-123")
        with pytest.raises(ValueError, match="キャンセル不可"):
            order.cancel()


class TestMoney:
    """Money 値オブジェクトのテスト"""

    def test_加算(self):
        a = Money(1000)
        b = Money(500)
        assert a.add(b) == Money(1500)

    def test_異なる通貨の加算はエラー(self):
        jpy = Money(1000, "JPY")
        usd = Money(500, "USD")
        with pytest.raises(ValueError, match="通貨が異なります"):
            jpy.add(usd)

    def test_不変性(self):
        a = Money(1000)
        b = a.add(Money(500))
        assert a.amount == 1000   # 元のオブジェクトは変わらない
        assert b.amount == 1500

    def test_等価性(self):
        a = Money(1000, "JPY")
        b = Money(1000, "JPY")
        assert a == b             # 値で比較

    def test_負の金額はエラー(self):
        with pytest.raises(ValueError, match="金額は0以上"):
            Money(-100)
```

---

## 6. 比較表

### 6.1 エンティティ vs 値オブジェクト

| 特性 | エンティティ | 値オブジェクト |
|------|-----------|-------------|
| 同一性 | ID で識別 | 値で識別 |
| 可変性 | 可変 (Mutable) | 不変 (Immutable) |
| ライフサイクル | 作成・変更・削除 | 生成のみ（変更は新規生成） |
| 例 | Order, User, Product | Money, Address, Email |
| 等価判定 | id が同じなら同一 | 全属性が同じなら同一 |
| テスト | 状態遷移を検証 | 値の計算を検証 |
| 永続化 | 独自テーブル/コレクション | 親エンティティに埋め込み |

### 6.2 戦術パターンの一覧

| パターン | 責務 | 配置層 | 使用例 |
|---------|------|-------|--------|
| エンティティ | ビジネスルール + ID | ドメイン層 | Order, User |
| 値オブジェクト | 不変の値表現 | ドメイン層 | Money, Address |
| 集約 | トランザクション境界 | ドメイン層 | Order + OrderItems |
| ドメインサービス | 複数集約にまたがるロジック | ドメイン層 | PricingService |
| ドメインイベント | 集約間の非同期連携 | ドメイン層 | OrderPlaced |
| リポジトリ | 集約の永続化・取得 | Interface=ドメイン、実装=インフラ | OrderRepository |
| ファクトリ | 複雑な集約の生成 | ドメイン層 | OrderFactory |
| アプリケーションサービス | ユースケースの調整 | アプリケーション層 | PlaceOrderService |

### 6.3 アプリケーションサービス vs ドメインサービス

| 観点 | アプリケーションサービス | ドメインサービス |
|------|----------------------|----------------|
| 配置層 | アプリケーション層 | ドメイン層 |
| 責務 | ユースケースの調整 | 複数集約にまたがるビジネスロジック |
| トランザクション管理 | 行う | 行わない |
| 外部依存 | リポジトリ、イベント等に依存 | ドメイン層のみに依存 |
| ビジネスルール | 含まない（Entity に委譲） | 含む（集約横断のルール） |
| テスト | 統合テスト寄り | ユニットテスト |
| 例 | PlaceOrderService | PricingService, TransferService |

---

## 7. アンチパターン

### アンチパターン 1: 貧血ドメインモデル (Anemic Domain Model)

```python
# NG: エンティティにロジックがなく、サービスに全て集中
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
    # → Order の状態遷移ルールが Service に分散
    # → 複数の Service が同じ Order を操作して矛盾が生じる

# OK: エンティティ自身がビジネスルールを持つ（リッチドメインモデル）
class Order:
    def place(self):
        if self.status != "draft":
            raise ValueError("下書き状態でのみ確定可能")
        if not self.items:
            raise ValueError("明細が空の注文は確定できません")
        self.status = "placed"
        # → ルールは Order に集約
        # → どこから呼んでも同じルールが適用される
```

**WHY これが問題なのか？**

貧血モデルでは、エンティティは単なる「データの入れ物」になり、ビジネスルールがサービスクラスに分散する。複数のサービスが同じエンティティを操作する場合、ルールの適用漏れや矛盾が生じやすい。Martin Fowler はこれを「ドメインモデルの最大のアンチパターン」と呼んでいる。

### アンチパターン 2: 集約が大きすぎる

```
NG: 1つの集約に全てを含める
  Order (集約ルート)
    ├── Customer (全属性)       ← 別の集約であるべき
    ├── Product (全属性) x N    ← 別の集約であるべき
    ├── PaymentHistory x N     ← 別の集約であるべき
    └── ShippingLog x N        ← 別の集約であるべき
  → 更新のたびに巨大オブジェクトをロード
  → 同時更新で楽観的ロック競合が頻発
  → テストが困難（大量のセットアップが必要）

OK: 集約を小さく保ち、ID で参照
  Order (集約ルート)
    ├── customer_id: str           ← ID のみ
    ├── OrderItem x N
    │     └── product_id: str      ← ID のみ
    └── shipping_address: Address  ← 値オブジェクト（埋め込み）
  → 軽量で高速にロード
  → 同時更新の競合が最小限
  → テストが容易
```

### アンチパターン 3: 全てをドメインイベントで処理

```python
# NG: 同一集約内の処理もイベント駆動にする
class Order:
    def place(self):
        self.status = "placed"
        # 合計金額の再計算をイベント経由で行う（過剰）
        self._events.append(RecalculateTotal(self.id))

# OK: 同一集約内の整合性は同期的に保つ
class Order:
    def place(self):
        if self.total_amount.amount < 100:
            raise ValueError("最低注文金額を満たしていません")
        self.status = "placed"
        # イベントは集約「外」への通知用
        self._domain_events.append(OrderPlaced(...))
```

### アンチパターン 4: リポジトリで集約以外を返す

```python
# NG: リポジトリが集約ルート以外を返す
class OrderRepository:
    def find_item_by_id(self, item_id: str) -> OrderItem:
        # OrderItem を直接返すと、集約の不変条件をバイパスできてしまう
        ...

# OK: リポジトリは常に集約ルートを返す
class OrderRepository:
    def find_by_id(self, order_id: str) -> Order:
        # Order（集約ルート）を返す
        # OrderItem へのアクセスは Order 経由で行う
        ...
```

---

## 8. 実践演習

### 演習1（基礎）: 値オブジェクトの設計

**課題**: 日本の電話番号を表す値オブジェクト `PhoneNumber` を実装せよ。

```
仕様:
- 形式: 数字のみ10〜11桁（ハイフン除去後）
- 表示用メソッド: "090-1234-5678" 形式に整形
- 値での等価判定
- 不変
```

**期待される実装と出力**:

```python
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class PhoneNumber:
    """日本の電話番号を表す値オブジェクト"""
    value: str   # ハイフンなしの数字文字列

    def __post_init__(self):
        # ハイフンを除去して正規化
        cleaned = self.value.replace('-', '').replace(' ', '')
        if not cleaned.isdigit():
            raise ValueError(f"数字以外が含まれています: {self.value}")
        if len(cleaned) < 10 or len(cleaned) > 11:
            raise ValueError(f"電話番号は10〜11桁: {cleaned} ({len(cleaned)}桁)")
        # frozen=True でも __post_init__ で設定可能
        object.__setattr__(self, 'value', cleaned)

    @property
    def formatted(self) -> str:
        """ハイフン付き表示形式"""
        v = self.value
        if len(v) == 11 and v.startswith('0'):
            # 携帯電話: 090-1234-5678
            return f"{v[:3]}-{v[3:7]}-{v[7:]}"
        elif len(v) == 10:
            # 固定電話: 03-1234-5678
            return f"{v[:2]}-{v[2:6]}-{v[6:]}"
        return v

    def __str__(self) -> str:
        return self.formatted


# テスト
p1 = PhoneNumber("090-1234-5678")
p2 = PhoneNumber("09012345678")
print(p1.formatted)     # 出力: 090-1234-5678
print(p1 == p2)          # 出力: True（値で比較）
print(p1.value)          # 出力: 09012345678

p3 = PhoneNumber("03-1234-5678")
print(p3.formatted)     # 出力: 03-1234-5678

try:
    PhoneNumber("123")
except ValueError as e:
    print(f"エラー: {e}")  # 出力: エラー: 電話番号は10〜11桁: 123 (3桁)
```

### 演習2（応用）: 集約の設計と実装

**課題**: 以下の仕様を持つ「在庫 (Stock)」集約を設計・実装せよ。

```
仕様:
- 商品ID、現在在庫数、引当済み数を管理
- 在庫の入荷（receive）: 在庫数を増加
- 在庫の引当（reserve）: 引当済み数を増加（利用可能数以下であること）
- 引当の解放（release）: 引当済み数を減少
- 出荷（ship）: 引当済み数と在庫数を両方減少
- 利用可能数 = 在庫数 - 引当済み数
```

**期待される実装と出力**:

```python
@dataclass
class Stock:
    """在庫集約ルート"""
    product_id: str
    quantity: int = 0           # 現在在庫数
    reserved: int = 0           # 引当済み数
    _domain_events: list = field(default_factory=list, repr=False)

    @property
    def available(self) -> int:
        """利用可能数"""
        return self.quantity - self.reserved

    def receive(self, amount: int) -> None:
        """入荷: 在庫を増やす"""
        if amount <= 0:
            raise ValueError("入荷数は正の値")
        self.quantity += amount
        self._domain_events.append(StockReceived(
            product_id=self.product_id, amount=amount,
        ))

    def reserve(self, amount: int) -> str:
        """引当: 利用可能数から確保する"""
        if amount <= 0:
            raise ValueError("引当数は正の値")
        if self.available < amount:
            raise ValueError(
                f"在庫不足: 利用可能{self.available} < 要求{amount}"
            )
        self.reserved += amount
        reservation_id = f"rsv-{self.product_id}-{self.reserved}"
        self._domain_events.append(StockReserved(
            product_id=self.product_id,
            reservation_id=reservation_id,
            amount=amount,
        ))
        return reservation_id

    def release(self, amount: int) -> None:
        """引当解放: 引当済みを戻す"""
        if amount <= 0:
            raise ValueError("解放数は正の値")
        if self.reserved < amount:
            raise ValueError("解放数が引当済み数を超えています")
        self.reserved -= amount

    def ship(self, amount: int) -> None:
        """出荷: 引当済みから出荷する"""
        if amount <= 0:
            raise ValueError("出荷数は正の値")
        if self.reserved < amount:
            raise ValueError("出荷数が引当済み数を超えています")
        self.reserved -= amount
        self.quantity -= amount

    def collect_events(self) -> list:
        events = list(self._domain_events)
        self._domain_events.clear()
        return events


# テスト
stock = Stock(product_id="prod-1")
stock.receive(100)
print(f"在庫: {stock.quantity}, 利用可能: {stock.available}")
# 出力: 在庫: 100, 利用可能: 100

rsv_id = stock.reserve(30)
print(f"引当後 - 在庫: {stock.quantity}, 引当: {stock.reserved}, 利用可能: {stock.available}")
# 出力: 引当後 - 在庫: 100, 引当: 30, 利用可能: 70

stock.ship(20)
print(f"出荷後 - 在庫: {stock.quantity}, 引当: {stock.reserved}, 利用可能: {stock.available}")
# 出力: 出荷後 - 在庫: 80, 引当: 10, 利用可能: 70

try:
    stock.reserve(80)
except ValueError as e:
    print(f"エラー: {e}")
# 出力: エラー: 在庫不足: 利用可能70 < 要求80
```

### 演習3（発展）: コンテキストマップと Anti-Corruption Layer

**課題**: 外部の決済サービス (Stripe) との連携に Anti-Corruption Layer を実装せよ。

```
仕様:
- 自ドメインの Payment エンティティと外部の Stripe API のモデルを変換
- Stripe の payment_intent を自ドメインの概念にマッピング
- 外部API の障害が自ドメインのモデルに影響しないこと
```

**期待される実装**:

```python
# 自ドメインのモデル
@dataclass
class Payment:
    """決済エンティティ（自ドメイン）"""
    id: str
    order_id: str
    amount: Money
    status: str = "pending"  # pending, completed, failed, refunded

    def complete(self) -> None:
        if self.status != "pending":
            raise ValueError(f"完了できません（現在: {self.status}）")
        self.status = "completed"

    def fail(self, reason: str) -> None:
        self.status = "failed"

    def refund(self) -> None:
        if self.status != "completed":
            raise ValueError("完了済みの決済のみ返金可能")
        self.status = "refunded"


# Anti-Corruption Layer（外部モデル → 自ドメインモデルの変換）
class StripePaymentGateway:
    """
    Stripe との連携を担う Anti-Corruption Layer

    責務:
    - 自ドメインの概念と Stripe API の概念を変換
    - Stripe 固有のエラーを自ドメインの例外に変換
    - Stripe の API 仕様変更の影響を吸収
    """

    def __init__(self, stripe_client):
        self._client = stripe_client

    def charge(self, payment: Payment) -> str:
        """決済を実行し、外部の payment_intent_id を返す"""
        try:
            # Stripe API の呼び出し（外部モデル）
            intent = self._client.PaymentIntent.create(
                amount=payment.amount.amount,
                currency=payment.amount.currency.lower(),
                metadata={
                    'order_id': payment.order_id,
                    'payment_id': payment.id,
                },
            )

            # Stripe のステータスを自ドメインのステータスに変換
            if intent.status == 'succeeded':
                payment.complete()
            elif intent.status in ('canceled', 'requires_payment_method'):
                payment.fail(f"Stripe status: {intent.status}")
            # 他のステータスは pending のまま

            return intent.id

        except self._client.error.CardError as e:
            payment.fail(f"カードエラー: {e.user_message}")
            raise PaymentDeclinedError(str(e))
        except self._client.error.StripeError as e:
            raise PaymentGatewayError(f"決済サービスエラー: {e}")

    def refund(self, payment_intent_id: str, amount: Money) -> str:
        """返金を実行"""
        try:
            refund = self._client.Refund.create(
                payment_intent=payment_intent_id,
                amount=amount.amount,
            )
            return refund.id
        except self._client.error.StripeError as e:
            raise PaymentGatewayError(f"返金エラー: {e}")


# 自ドメインの例外（Stripe の例外とは独立）
class PaymentDeclinedError(Exception):
    """決済が拒否された"""
    pass

class PaymentGatewayError(Exception):
    """決済ゲートウェイの技術的エラー"""
    pass

# テスト用 Fake
class FakeStripeClient:
    """テスト用の Stripe クライアント"""
    class PaymentIntent:
        @staticmethod
        def create(**kwargs):
            class Intent:
                id = "pi_test_123"
                status = "succeeded"
            return Intent()

# テスト
fake_stripe = FakeStripeClient()
gateway = StripePaymentGateway(fake_stripe)
payment = Payment(id="pay-1", order_id="order-1", amount=Money(5000))
intent_id = gateway.charge(payment)
print(f"決済完了: {intent_id}, ステータス: {payment.status}")
# 出力: 決済完了: pi_test_123, ステータス: completed
```

---

## 9. FAQ

### Q1. DDD はいつ採用すべきか？

**A.** ドメインの複雑性が高いプロジェクトに適している。判断基準は以下の通り。

```
DDD 採用判断チェックリスト:

  [x] ビジネスルールが複雑（単純な CRUD では表現しきれない）
  [x] ドメインエキスパートが存在しアクセス可能
  [x] プロジェクトの寿命が長い（1年以上）
  [x] チームに DDD の知識を持つメンバーがいる（または学ぶ意欲がある）
  [x] ビジネスの競争優位性がソフトウェアに依存している

  上記の3つ以上に該当 → DDD の採用を推奨
  1〜2つ → 戦術パターンのみ部分採用
  0つ → CRUD + 従来型アーキテクチャで十分
```

### Q2. 集約間のデータ整合性はどう保つ？

**A.** 結果整合性（Eventual Consistency）を基本とする。集約Aの変更でドメインイベントを発行し、集約Bがそのイベントを購読して非同期に自身を更新する。強い整合性が必要な場合は Saga パターンで補償トランザクションを実装する。「1トランザクション = 1集約」のルールを崩さないことが重要。

### Q3. ユビキタス言語はどう確立するか？

**A.** 以下の手順で段階的に確立する。

```
Step 1: ドメインエキスパートとの会話でキーワードを抽出
Step 2: 用語辞書を作成（コンテキストごとに）
Step 3: コード上のクラス名・メソッド名を用語辞書に合わせる
Step 4: レビュー時に「この名前はドメインエキスパートが使うか？」を確認
Step 5: 新しい概念が出たら辞書を更新し、コードも合わせる
```

### Q4. DDD と CQRS の関係は？

**A.** DDD は「ドメインをどうモデリングするか」、CQRS は「読み書きをどう分離するか」を扱う。DDD の集約は書き込みに最適化されたモデルだが、読み取り（一覧表示、検索等）には非効率な場合がある。CQRS を併用することで、書き込み側は DDD のリッチドメインモデルを使い、読み取り側はパフォーマンスに最適化された Read Model を使う、という棲み分けが可能になる。

### Q5. イベントソーシングは必須か？

**A.** 必須ではない。イベントソーシングは「全ての状態変更をイベントとして記録し、現在の状態はイベントの再生で導出する」パターン。DDD のドメインイベントとは別の概念。監査ログや時系列分析が重要な領域（金融、医療）では有用だが、複雑さも増すため慎重に判断すべき。まずはドメインイベントの発行から始め、必要に応じてイベントソーシングに発展させるのが現実的。

### Q6. マイクロサービスと DDD の関係は？

**A.** DDD の境界づけられたコンテキストは、マイクロサービスの自然な境界を提供する。1つの Bounded Context = 1つのマイクロサービスとする設計が理想的。ただし、モノリスでも DDD は有効であり、マイクロサービスは DDD の前提ではない。

---

## 10. まとめ

| 項目 | ポイント |
|------|---------|
| 戦略的設計 | 境界づけられたコンテキストでドメインを分割。コンテキスト間はイベントで連携 |
| 戦術的設計 | エンティティ・値オブジェクト・集約・ドメインイベントで複雑さをモデリング |
| 集約設計 | 小さく保つ。1トランザクション = 1集約。集約間は ID 参照 |
| ユビキタス言語 | ドメインエキスパートとコードで同じ言葉を使う |
| 貧血モデル回避 | ビジネスロジックはエンティティに持たせ、サービスは調整役に徹する |
| 結果整合性 | 集約間はドメインイベントによる非同期連携を基本とする |
| ドメインサービス | 複数集約にまたがるロジックの受け皿。エンティティに属さないルール |
| Saga パターン | 複数集約の長いトランザクションを補償トランザクションで管理 |
| ACL | 外部システムのモデルを自ドメインのモデルに変換する防御層 |
| サブドメイン | コア/サポート/汎用の分類でリソース配分を最適化 |

---

## 次に読むべきガイド

- [クリーンアーキテクチャ](./01-clean-architecture.md) — DDD と組み合わせるレイヤーアーキテクチャ。Entities 層の構造化
- [イベント駆動アーキテクチャ](./03-event-driven.md) — ドメインイベントを活用した疎結合設計。Saga パターンの実装基盤
- [API 設計](../../clean-code-principles/docs/03-practices-advanced/03-api-design.md) — 集約を公開する API の設計原則
- [Repository パターン](../../design-patterns-guide/docs/04-architectural/) — 集約の永続化パターン
- [システム設計の基礎](../00-fundamentals/) — スケーラビリティと可用性の基礎

---

## 参考文献

1. **Domain-Driven Design: Tackling Complexity in the Heart of Software** — Eric Evans (Addison-Wesley, 2003) — DDD の原典。戦略的設計と戦術的設計の体系的解説
2. **Implementing Domain-Driven Design** — Vaughn Vernon (Addison-Wesley, 2013) — DDD の実装パターン詳細。集約設計、リポジトリ実装の実践的ガイド
3. **Domain-Driven Design Distilled** — Vaughn Vernon (Addison-Wesley, 2016) — DDD の簡潔な入門書。戦略的設計に焦点
4. **Architecture Patterns with Python** — Harry Percival & Bob Gregory (O'Reilly, 2020) — Python での DDD 実践。Repository、Unit of Work の実装例
5. **Patterns, Principles, and Practices of Domain-Driven Design** — Scott Millett & Nick Tune (Wrox, 2015) — DDD パターンの包括的カタログ
6. **Event Storming** — Alberto Brandolini — https://www.eventstorming.com/ — ドメインイベントを発見するためのワークショップ手法

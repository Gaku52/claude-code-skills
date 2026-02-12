# イベント駆動アーキテクチャ

> コンポーネント間の通信をイベント（事実の記録）を中心に設計するアーキテクチャパターンであり、Pub/Sub モデル・イベントソーシング・CQRS を活用して疎結合でスケーラブルなシステムを構築する手法を解説する

## この章で学ぶこと

1. **イベント駆動の基本モデル** — イベント通知・イベントキャリー・イベントソーシングの3パターンとその使い分け
2. **Pub/Sub アーキテクチャ** — トピックベースのメッセージングによる疎結合な連携設計
3. **CQRS とイベントソーシング** — コマンドとクエリの分離、イベントログからの状態再構築
4. **Saga パターン** — 分散トランザクションの代替としての補償ベースの結果整合性
5. **運用と監視** — イベント駆動システムの可観測性・リトライ・Dead Letter Queue の実践

---

## 前提知識

| トピック | 必要レベル | 参考ガイド |
|---------|-----------|-----------|
| メッセージキュー基礎 | Kafka / RabbitMQ の概念を理解 | [メッセージキュー](../01-components/02-message-queue.md) |
| ドメインモデリング | エンティティ・集約・ドメインイベントの基礎 | [DDD](./02-ddd.md) |
| クリーンアーキテクチャ | レイヤー構成と依存方向の理解 | [クリーンアーキテクチャ](./01-clean-architecture.md) |
| データベース基礎 | RDB / NoSQL の読み書き特性 | [データベース](../01-components/01-database.md) |
| Python 中級 | dataclass, Protocol, async/await | [Python ガイド](../../../02-programming/python-guide/docs/00-basics/00-introduction.md) |

---

## 背景と哲学

### なぜイベント駆動が必要なのか

```
従来のモノリシック / 同期マイクロサービスの課題:

  1. 密結合: Order → Inventory → Payment → Notification
     → 1サービス障害で全体停止（カスケード障害）
     → 新サービス追加時に既存サービスの変更が必要

  2. スケーラビリティの壁:
     → 読み込み 10,000 req/s だが書き込みは 100 req/s
     → 同じモデルで両方を最適化できない

  3. ドメイン知識の漏洩:
     → Order Service が「在庫を引き当てろ」と命令
     → Order は Inventory の内部を知っている（密結合）

イベント駆動のパラダイムシフト:
  「命令」ではなく「事実の通知」

  Order Service: 「注文が確定した」(OrderPlaced) ← 事実を述べるだけ
  Inventory Service: 「じゃあ在庫を引き当てよう」← 自律的に判断
  Payment Service: 「じゃあ決済を開始しよう」← 自律的に判断
  Email Service: 「じゃあ確認メールを送ろう」← 自律的に判断

  → 各サービスが自律的に行動する。追加・削除が自由
```

### イベント駆動アーキテクチャの歴史

```
1987: Kent Beck - Smalltalk のイベントシステム
2003: Gregor Hohpe - Enterprise Integration Patterns
2005: Martin Fowler - Event Sourcing パターンの体系化
2006: Greg Young - CQRS の提唱
2011: Apache Kafka 登場 - 大規模イベントストリーミング基盤
2014: Reactive Manifesto - リアクティブシステムの4原則
2017: Martin Kleppmann - DDIA でストリーム処理の理論体系化
2020: Adam Bellemare - Event-Driven Microservices の実践ガイド

核となる考え方:
  「ソフトウェアの状態とは、これまでに起きたイベントの累積結果である」
  — Greg Young
```

### 4つの設計原則

```
1. 事実の記録 (Facts, not Commands)
   × "ReserveInventory" (命令)
   ○ "OrderPlaced"      (事実の記録)

2. 時間的分離 (Temporal Decoupling)
   発信者と受信者が同時に稼働している必要がない
   → メッセージキューがバッファリング

3. 自律性 (Autonomy)
   各サービスはイベントを受け取り、自律的に判断・行動する
   → 他サービスの内部実装に依存しない

4. 結果整合性 (Eventual Consistency)
   「全てのデータが常に整合する」から
   「いずれ整合する」へのパラダイムシフト
   → 強い整合性が必要な箇所を限定する設計
```

---

## 1. イベント駆動の3つのパターン

### 1.1 パターン概要

```
パターン1: Event Notification (イベント通知)
  Order Service --"OrderPlaced {id:123}"--> [Event Bus]
  → Inventory Service: 「注文123が来たので在庫を確認しに行こう」
  → 最小限の情報のみ。受信側がデータを取りに行く

パターン2: Event-Carried State Transfer (状態運搬イベント)
  Order Service --"OrderPlaced {id:123, items:[...], address:{...}}"--> [Event Bus]
  → Shipping Service: 「全情報が揃っているのでそのまま処理できる」
  → 必要なデータを全て含む。受信側からの問い合わせ不要

パターン3: Event Sourcing (イベントソーシング)
  全ての状態変更をイベントとして記録
  [OrderCreated] → [ItemAdded] → [ItemAdded] → [OrderPlaced] → [OrderShipped]
  → 現在の状態 = 全イベントの再生結果
```

### 1.2 3パターンの詳細比較

```python
# === パターン1: Event Notification ===
# 最小限の情報のみ送信。受信者は必要に応じてデータを取得する

@dataclass(frozen=True)
class OrderPlacedNotification(DomainEvent):
    """通知イベント: IDのみ"""
    event_type: str = "order.placed"
    order_id: str = ""
    # → 受信者は order_id で Order Service に問い合わせる

# メリット: イベントサイズが小さい、スキーマ変更の影響が少ない
# デメリット: 受信者から発信者への依存が残る（データ取得のため）


# === パターン2: Event-Carried State Transfer ===
# 必要な情報を全て含む。受信者は発信者に問い合わせ不要

@dataclass(frozen=True)
class OrderPlacedFull(DomainEvent):
    """状態運搬イベント: 全データ含む"""
    event_type: str = "order.placed"
    order_id: str = ""
    customer_id: str = ""
    customer_name: str = ""
    customer_email: str = ""
    items: tuple = ()
    shipping_address: str = ""
    total_amount: int = 0
    currency: str = "JPY"

# メリット: 完全な疎結合、高パフォーマンス
# デメリット: イベントサイズが大きい、スキーマ変更の影響が大きい


# === パターン3: Event Sourcing ===
# 状態変更を全てイベントとして記録し、再生で現在状態を復元

@dataclass(frozen=True)
class OrderCreated(DomainEvent):
    event_type: str = "order.created"
    order_id: str = ""
    customer_id: str = ""

@dataclass(frozen=True)
class ItemAddedToOrder(DomainEvent):
    event_type: str = "order.item_added"
    order_id: str = ""
    product_id: str = ""
    quantity: int = 0
    unit_price: int = 0

@dataclass(frozen=True)
class OrderConfirmed(DomainEvent):
    event_type: str = "order.confirmed"
    order_id: str = ""
    confirmed_at: str = ""

# メリット: 完全な監査証跡、任意時点の状態復元、デバッグ容易
# デメリット: 実装複雑、イベントストアの管理、スナップショット必要
```

### 1.3 パターン選択の判断基準

| 判断基準 | Event Notification | Event-Carried State | Event Sourcing |
|---------|:-----------------:|:------------------:|:--------------:|
| 疎結合度 | 中（データ取得で依存） | 高（完全独立） | 高（イベントログ自律） |
| イベントサイズ | 小 | 大 | 中（差分記録） |
| スキーマ変更の影響 | 小 | 大 | 中 |
| 監査証跡 | なし | なし | 完全 |
| 適用場面 | 社内マイクロサービス | 外部システム連携 | 金融・医療・法規制 |
| 実装複雑性 | 低 | 低 | 高 |

### 1.4 Pub/Sub の全体構成

```
                          Event Bus (Kafka / SNS+SQS)
                    +-----------------------------------+
                    |                                   |
  Order Service --->| Topic: order-events               |
                    |   "OrderPlaced"                   |
                    |   "OrderCancelled"                |
                    +---+----------+----------+---------+
                        |          |          |
                        v          v          v
                  +---------+ +--------+ +----------+
                  |Inventory| |Payment | |  Email   |
                  |Service  | |Service | | Service  |
                  +---------+ +--------+ +----------+

  ★ Order Service は下流サービスの存在を知らない
  ★ 新しい消費者を追加しても Order Service の変更不要
  ★ 各 Consumer は独自のペースで処理（バックプレッシャー制御）
```

### 1.5 同期 vs 非同期の詳細比較

```
【同期 (REST/gRPC)】
  Order --> Inventory --> Payment --> Notification
   |            |            |            |
   |<-----------+<-----------+<-----------+
   全体のレイテンシ = 各サービスの合計 (50ms + 200ms + 100ms = 350ms)
   1つ障害 = 全体障害 (カスケード障害)
   スケール = 最も遅いサービスがボトルネック

【非同期 (Event-Driven)】
  Order --event--> [Bus] ---> Inventory (独立処理)
                         ---> Payment   (独立処理)
                         ---> Notification (独立処理)
   Order は即座に応答可能 (レイテンシ = イベント発行の時間のみ ≈ 5ms)
   1つ障害 = そのサービスのみ影響（リトライで回復）
   スケール = 各サービスが独立してスケール

【ハイブリッド（実践的な推奨構成）】
  User → [API Gateway] → Order Service (同期レスポンス: "注文受付済み")
                              |
                         [Event Bus]
                          /   |    \
                    Inventory Payment Email
                    (非同期)  (非同期) (非同期)

  → ユーザーには即座にレスポンスを返し、
    後続処理は非同期で実行するのがベストプラクティス
```

---

## 2. Pub/Sub の実装

### 2.1 イベントの定義

```python
# domain/events/base.py
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, Any
import uuid
import json


@dataclass(frozen=True)
class DomainEvent:
    """ドメインイベントの基底クラス

    全てのイベントは不変（frozen=True）。
    一度発生したイベント（事実）は変更できないため。
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    event_type: str = ""
    # メタデータ: トレーシング用
    correlation_id: str = ""  # リクエスト全体を追跡する ID
    causation_id: str = ""    # このイベントの原因となったイベント/コマンドの ID

    def to_dict(self) -> dict:
        """シリアライズ用の辞書変換"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, (tuple, list)):
                result[key] = list(value)
            else:
                result[key] = value
        return result

    @property
    def aggregate_id(self) -> str:
        """集約IDを返す。サブクラスでオーバーライド"""
        raise NotImplementedError


@dataclass(frozen=True)
class OrderPlaced(DomainEvent):
    """注文確定イベント"""
    event_type: str = "order.placed"
    order_id: str = ""
    customer_id: str = ""
    items: tuple = ()          # frozen=True のため tuple を使用
    total_amount: int = 0
    currency: str = "JPY"

    @property
    def aggregate_id(self) -> str:
        return self.order_id


@dataclass(frozen=True)
class OrderCancelled(DomainEvent):
    """注文キャンセルイベント"""
    event_type: str = "order.cancelled"
    order_id: str = ""
    reason: str = ""
    cancelled_by: str = ""     # "customer" | "system" | "admin"

    @property
    def aggregate_id(self) -> str:
        return self.order_id


@dataclass(frozen=True)
class PaymentCompleted(DomainEvent):
    """決済完了イベント"""
    event_type: str = "payment.completed"
    payment_id: str = ""
    order_id: str = ""
    amount: int = 0
    currency: str = "JPY"
    payment_method: str = ""   # "credit_card" | "bank_transfer"

    @property
    def aggregate_id(self) -> str:
        return self.payment_id


@dataclass(frozen=True)
class InventoryReserved(DomainEvent):
    """在庫引当完了イベント"""
    event_type: str = "inventory.reserved"
    reservation_id: str = ""
    order_id: str = ""
    items: tuple = ()

    @property
    def aggregate_id(self) -> str:
        return self.reservation_id
```

### 2.2 イベントパブリッシャー（Protocol + Kafka 実装）

```python
# domain/ports/event_publisher.py
from typing import Protocol

class EventPublisher(Protocol):
    """イベント発行のポート（インターフェース）"""
    def publish(self, event: DomainEvent) -> None: ...
    def publish_batch(self, events: list[DomainEvent]) -> None: ...


# infrastructure/messaging/kafka_publisher.py
from confluent_kafka import Producer
import json
import logging

logger = logging.getLogger(__name__)


class KafkaEventPublisher:
    """Kafka ベースのイベントパブリッシャー

    設計判断:
    - acks='all': 全レプリカへの書き込みを保証（耐久性）
    - enable.idempotence=True: 重複メッセージ防止
    - パーティションキー = aggregate_id: 同一集約のイベント順序保証
    """

    def __init__(self, bootstrap_servers: str, schema_registry_url: str = ""):
        self._producer = Producer({
            'bootstrap.servers': bootstrap_servers,
            'acks': 'all',
            'enable.idempotence': True,
            'max.in.flight.requests.per.connection': 5,
            'retries': 3,
            'retry.backoff.ms': 100,
        })
        self._schema_registry_url = schema_registry_url

    def publish(self, event: DomainEvent) -> None:
        """単一イベントの発行"""
        topic = self._resolve_topic(event)
        key = event.aggregate_id.encode('utf-8')
        value = json.dumps(
            event.to_dict(), ensure_ascii=False, default=str
        ).encode('utf-8')

        # ヘッダーにメタデータを付与（トレーシング用）
        headers = [
            ('event_type', event.event_type.encode()),
            ('correlation_id', event.correlation_id.encode()),
            ('causation_id', event.causation_id.encode()),
        ]

        self._producer.produce(
            topic=topic,
            key=key,
            value=value,
            headers=headers,
            callback=self._delivery_report,
        )
        self._producer.flush()
        logger.info(
            "イベント発行完了",
            extra={
                'event_type': event.event_type,
                'event_id': event.event_id,
                'aggregate_id': event.aggregate_id,
                'topic': topic,
            }
        )

    def publish_batch(self, events: list[DomainEvent]) -> None:
        """複数イベントのバッチ発行"""
        for event in events:
            topic = self._resolve_topic(event)
            key = event.aggregate_id.encode('utf-8')
            value = json.dumps(
                event.to_dict(), ensure_ascii=False, default=str
            ).encode('utf-8')
            self._producer.produce(
                topic=topic,
                key=key,
                value=value,
                callback=self._delivery_report,
            )
        # バッチ全体を一度にフラッシュ
        self._producer.flush()
        logger.info(f"バッチイベント発行完了: {len(events)}件")

    def _resolve_topic(self, event: DomainEvent) -> str:
        """イベントタイプからトピック名を解決

        例: "order.placed" → "domain-events.order"
            "payment.completed" → "domain-events.payment"
        """
        domain = event.event_type.split('.')[0]
        return f"domain-events.{domain}"

    def _delivery_report(self, err, msg):
        if err:
            logger.error(f"イベント配信失敗: {err}, topic={msg.topic()}")
            raise RuntimeError(f"イベント配信失敗: {err}")
        logger.debug(
            f"配信成功: topic={msg.topic()}, "
            f"partition={msg.partition()}, offset={msg.offset()}"
        )


# infrastructure/messaging/in_memory_publisher.py
class InMemoryEventPublisher:
    """テスト用のインメモリイベントパブリッシャー"""

    def __init__(self):
        self.published_events: list[DomainEvent] = []

    def publish(self, event: DomainEvent) -> None:
        self.published_events.append(event)

    def publish_batch(self, events: list[DomainEvent]) -> None:
        self.published_events.extend(events)

    def get_events_of_type(self, event_type: str) -> list[DomainEvent]:
        return [e for e in self.published_events if e.event_type == event_type]

    def clear(self) -> None:
        self.published_events.clear()
```

### 2.3 イベントコンシューマー（Kafka Consumer）

```python
# infrastructure/messaging/kafka_consumer.py
from confluent_kafka import Consumer, KafkaError
import json
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class KafkaEventConsumer:
    """Kafka ベースのイベントコンシューマー

    設計判断:
    - enable.auto.commit=False: 処理完了後に手動コミット（at-least-once 保証）
    - Consumer Group: 同一グループ内でパーティションが分配される
    - 手動オフセット管理: 処理失敗時のリトライを制御
    """

    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str,
        topics: list[str],
    ):
        self._consumer = Consumer({
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False,       # 手動コミット
            'max.poll.interval.ms': 300000,    # 5分
            'session.timeout.ms': 45000,
        })
        self._consumer.subscribe(topics)
        self._handlers: dict[str, Callable] = {}
        self._running = False

    def register_handler(
        self, event_type: str, handler: Callable[[dict], None]
    ) -> None:
        """イベントタイプに対応するハンドラーを登録"""
        self._handlers[event_type] = handler
        logger.info(f"ハンドラー登録: {event_type}")

    def start(self) -> None:
        """コンシューマーループを開始"""
        self._running = True
        logger.info("コンシューマー開始")

        while self._running:
            msg = self._consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error(f"Consumer エラー: {msg.error()}")
                continue

            try:
                event_data = json.loads(msg.value().decode('utf-8'))
                event_type = event_data.get('event_type', '')

                handler = self._handlers.get(event_type)
                if handler:
                    logger.info(
                        f"イベント処理開始: {event_type}, "
                        f"event_id={event_data.get('event_id')}"
                    )
                    handler(event_data)

                    # 処理成功後にオフセットをコミット
                    self._consumer.commit(message=msg)
                    logger.info(f"イベント処理完了: {event_type}")
                else:
                    logger.warning(f"未登録のイベントタイプ: {event_type}")
                    self._consumer.commit(message=msg)

            except Exception as e:
                logger.error(
                    f"イベント処理失敗: {e}",
                    extra={'raw_message': msg.value()},
                    exc_info=True,
                )
                # 処理失敗 → コミットしない → 再配信される
                # Dead Letter Queue への転送を検討
                self._handle_processing_failure(msg, e)

    def stop(self) -> None:
        """コンシューマーを停止"""
        self._running = False
        self._consumer.close()
        logger.info("コンシューマー停止")

    def _handle_processing_failure(self, msg, error: Exception) -> None:
        """処理失敗時のハンドリング（DLQ 転送等）"""
        # リトライ回数をヘッダーから取得
        headers = dict(msg.headers() or [])
        retry_count = int(headers.get(b'retry_count', b'0'))

        if retry_count >= 3:
            logger.error(
                f"最大リトライ回数超過。DLQ に転送: "
                f"topic={msg.topic()}, offset={msg.offset()}"
            )
            # DLQ トピックに転送（実装は省略）
            self._consumer.commit(message=msg)
        else:
            logger.warning(
                f"リトライ予定 ({retry_count + 1}/3): {error}"
            )
```

### 2.4 イベントハンドラー

```python
# application/handlers/inventory_handler.py
import logging

logger = logging.getLogger(__name__)


class InventoryEventHandler:
    """在庫サービスのイベントハンドラー

    このハンドラーは Inventory Bounded Context 内に配置される。
    Order サービスのドメインイベントを受信し、
    在庫ドメインの操作（引当・解放）を自律的に行う。
    """

    def __init__(self, inventory_repo, stock_service, event_publisher):
        self._repo = inventory_repo
        self._stock = stock_service
        self._publisher = event_publisher

    def handle_order_placed(self, event_data: dict) -> None:
        """注文確定イベントの処理: 在庫引当

        冪等性の保証:
        - reservation_id = order_id で重複チェック
        - 既に引当済みの場合はスキップ
        """
        order_id = event_data['data']['order_id']
        items = event_data['data']['items']

        # 冪等性チェック: 既に処理済みか確認
        if self._repo.reservation_exists(order_id):
            logger.info(f"引当済みのためスキップ: order_id={order_id}")
            return

        reserved_items = []
        try:
            for item in items:
                success = self._stock.reserve(
                    product_id=item['product_id'],
                    quantity=item['quantity'],
                    reservation_id=order_id,
                )
                if not success:
                    # 在庫不足 → これまでの引当を全解放 + 補償イベント発行
                    self._rollback_reservations(reserved_items, order_id)
                    self._publisher.publish(InventoryInsufficientEvent(
                        order_id=order_id,
                        product_id=item['product_id'],
                        requested_quantity=item['quantity'],
                        correlation_id=event_data.get('correlation_id', ''),
                        causation_id=event_data.get('event_id', ''),
                    ))
                    return
                reserved_items.append(item)

            # 全アイテム引当成功 → 成功イベント発行
            self._publisher.publish(InventoryReserved(
                reservation_id=order_id,
                order_id=order_id,
                items=tuple(
                    (item['product_id'], item['quantity'])
                    for item in items
                ),
                correlation_id=event_data.get('correlation_id', ''),
                causation_id=event_data.get('event_id', ''),
            ))
            logger.info(f"在庫引当完了: order_id={order_id}")

        except Exception as e:
            logger.error(f"在庫引当エラー: {e}", exc_info=True)
            self._rollback_reservations(reserved_items, order_id)
            raise

    def handle_order_cancelled(self, event_data: dict) -> None:
        """注文キャンセルイベントの処理: 在庫解放

        冪等性の保証:
        - 引当が存在しない場合は何もしない
        """
        order_id = event_data['data']['order_id']

        if not self._repo.reservation_exists(order_id):
            logger.info(f"引当なしのためスキップ: order_id={order_id}")
            return

        self._stock.release_reservation(reservation_id=order_id)
        self._publisher.publish(InventoryReleasedEvent(
            order_id=order_id,
            correlation_id=event_data.get('correlation_id', ''),
            causation_id=event_data.get('event_id', ''),
        ))
        logger.info(f"在庫解放完了: order_id={order_id}")

    def _rollback_reservations(
        self, reserved_items: list[dict], order_id: str
    ) -> None:
        """引当済みアイテムのロールバック"""
        for item in reversed(reserved_items):
            try:
                self._stock.release(
                    product_id=item['product_id'],
                    reservation_id=order_id,
                )
            except Exception as e:
                logger.error(
                    f"ロールバック失敗: product_id={item['product_id']}, "
                    f"order_id={order_id}, error={e}"
                )
                # アラート送信（手動対応が必要）
```

### 2.5 Outbox パターン（トランザクション保証）

```python
# infrastructure/outbox/outbox_publisher.py
"""
Outbox パターン: イベント発行のトランザクション保証

問題:
  1. ドメインオブジェクトを DB に保存
  2. イベントを Kafka に発行
  → 1は成功したが2が失敗すると、データ不整合が発生

解決策: Outbox テーブル
  1. ドメインオブジェクトと Outbox レコードを同一トランザクションで保存
  2. 別プロセス（Outbox Relay）が Outbox テーブルからイベントを読み取り Kafka に発行
  → トランザクションの原子性を活用

  [DB Transaction]
  ┌─────────────────────────────────────┐
  │  1. orders テーブルに INSERT          │
  │  2. outbox テーブルに INSERT          │
  │  → 両方成功 or 両方ロールバック        │
  └─────────────────────────────────────┘

  [Outbox Relay (別プロセス)]
  outbox テーブル → Kafka に発行 → outbox レコードを published に更新
"""
from sqlalchemy import Column, String, DateTime, Text, Boolean
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import json
import uuid


class OutboxRecord(Base):
    """Outbox テーブルのモデル"""
    __tablename__ = 'outbox'

    id = Column(String(36), primary_key=True)
    aggregate_type = Column(String(100), nullable=False)
    aggregate_id = Column(String(100), nullable=False)
    event_type = Column(String(100), nullable=False)
    payload = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False)
    published = Column(Boolean, default=False, nullable=False)
    published_at = Column(DateTime, nullable=True)


class OutboxEventPublisher:
    """Outbox パターンによるイベント発行

    ドメインオブジェクトの保存と同一トランザクションで
    Outbox テーブルにイベントを書き込む
    """

    def __init__(self, session: Session):
        self._session = session

    def publish(self, event: DomainEvent) -> None:
        """Outbox テーブルにイベントを記録

        注意: このメソッドは呼び出し元のトランザクション内で実行される
        （session.commit() は呼び出し元が行う）
        """
        record = OutboxRecord(
            id=event.event_id,
            aggregate_type=event.event_type.split('.')[0],
            aggregate_id=event.aggregate_id,
            event_type=event.event_type,
            payload=json.dumps(event.to_dict(), ensure_ascii=False, default=str),
            created_at=datetime.now(timezone.utc),
            published=False,
        )
        self._session.add(record)
        # commit は呼び出し元のトランザクションに委ねる

    def publish_batch(self, events: list[DomainEvent]) -> None:
        for event in events:
            self.publish(event)


class OutboxRelay:
    """Outbox Relay: 未発行イベントを Kafka に転送する

    定期実行（例: 1秒間隔のポーリング）で
    Outbox テーブルから未発行レコードを取得し Kafka に発行する
    """

    def __init__(self, session: Session, kafka_publisher: KafkaEventPublisher):
        self._session = session
        self._kafka = kafka_publisher

    def relay_pending_events(self, batch_size: int = 100) -> int:
        """未発行イベントを Kafka に転送"""
        records = (
            self._session.query(OutboxRecord)
            .filter_by(published=False)
            .order_by(OutboxRecord.created_at)
            .limit(batch_size)
            .all()
        )

        published_count = 0
        for record in records:
            try:
                event_data = json.loads(record.payload)
                # Kafka に発行
                self._kafka._producer.produce(
                    topic=f"domain-events.{record.aggregate_type}",
                    key=record.aggregate_id.encode(),
                    value=record.payload.encode(),
                )
                # 発行済みに更新
                record.published = True
                record.published_at = datetime.now(timezone.utc)
                published_count += 1
            except Exception as e:
                logger.error(f"Outbox relay 失敗: {record.id}, {e}")

        self._kafka._producer.flush()
        self._session.commit()

        if published_count > 0:
            logger.info(f"Outbox relay 完了: {published_count}件")
        return published_count
```

### 2.6 Saga パターン（分散トランザクション）

```python
# application/sagas/order_saga.py
"""
Saga パターン: 分散環境でのトランザクション管理

従来の ACID トランザクション:
  BEGIN
    1. 在庫引当
    2. 決済処理
    3. 配送手配
  COMMIT  ← 全て成功 or 全てロールバック

分散環境の Saga:
  1. 在庫引当 → 成功
  2. 決済処理 → 失敗!
  3. 在庫引当の補償（解放）を実行

  各ステップは独立したトランザクション。
  失敗時は完了済みステップの補償処理を逆順で実行。

  ┌──────────────────────────────────────────────┐
  │ 正常フロー:                                    │
  │ reserve_inventory → process_payment →          │
  │ schedule_shipping → send_confirmation          │
  │                                                │
  │ 失敗時の補償フロー (payment で失敗):             │
  │ reserve_inventory → process_payment(FAIL!)     │
  │                   ← release_inventory           │
  └──────────────────────────────────────────────┘
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SagaStatus(Enum):
    STARTED = "started"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    FAILED = "failed"


@dataclass
class SagaStep:
    name: str
    execute_func: str
    compensate_func: str
    status: str = "pending"  # pending | completed | compensated | failed


class SagaFailedError(Exception):
    def __init__(self, step_name: str, reason: str):
        self.step_name = step_name
        self.reason = reason
        super().__init__(f"Saga failed at step '{step_name}': {reason}")


class OrderPlacementSaga:
    """注文確定の Saga: 複数サービスにまたがる処理を調整

    Orchestration パターン: Saga が全ステップを直接制御する
    （Choreography パターンとの比較は後述）
    """

    STEPS = [
        SagaStep('reserve_inventory', 'reserve_inventory', 'release_inventory'),
        SagaStep('process_payment',   'process_payment',   'refund_payment'),
        SagaStep('schedule_shipping', 'schedule_shipping', 'cancel_shipping'),
        SagaStep('send_confirmation', 'send_confirmation', 'noop'),
    ]

    def __init__(
        self,
        inventory_service,
        payment_service,
        shipping_service,
        notification_service,
        saga_log_repo,
    ):
        self._inventory = inventory_service
        self._payment = payment_service
        self._shipping = shipping_service
        self._notification = notification_service
        self._saga_log = saga_log_repo
        self._completed_steps: list[SagaStep] = []
        self._status = SagaStatus.STARTED

    def execute(self, order_id: str, order_data: dict) -> None:
        """Saga の実行"""
        saga_id = f"saga-{order_id}"
        self._saga_log.create(saga_id, order_id, "order_placement")
        logger.info(f"Saga 開始: {saga_id}")

        for step in self.STEPS:
            try:
                logger.info(f"Saga ステップ実行: {step.name}")
                execute_func = getattr(self, step.execute_func)
                execute_func(order_id, order_data)
                step.status = "completed"
                self._completed_steps.append(step)
                self._saga_log.update_step(saga_id, step.name, "completed")
            except Exception as e:
                logger.error(f"Saga 失敗 at {step.name}: {e}")
                step.status = "failed"
                self._saga_log.update_step(saga_id, step.name, "failed")
                self._status = SagaStatus.COMPENSATING
                self._compensate(saga_id, order_id)
                self._status = SagaStatus.FAILED
                raise SagaFailedError(step.name, str(e))

        self._status = SagaStatus.COMPLETED
        self._saga_log.complete(saga_id)
        logger.info(f"Saga 完了: {saga_id}")

    def _compensate(self, saga_id: str, order_id: str) -> None:
        """完了済みステップの補償処理を逆順で実行

        補償処理は「ベストエフォート」で実行する。
        補償処理自体が失敗した場合はアラートを送信し、手動対応を要請する。
        """
        logger.warning(f"補償処理開始: {saga_id}")
        for step in reversed(self._completed_steps):
            if step.compensate_func == 'noop':
                continue
            try:
                comp_func = getattr(self, step.compensate_func)
                comp_func(order_id)
                step.status = "compensated"
                self._saga_log.update_step(
                    saga_id, step.name, "compensated"
                )
                logger.info(f"補償処理完了: {step.name}")
            except Exception as e:
                logger.critical(
                    f"補償処理失敗（手動対応必要）: {step.name}, error={e}",
                    exc_info=True,
                )
                # アラート送信: PagerDuty, Slack, etc.

    # === 各ステップの実装 ===

    def reserve_inventory(self, order_id: str, order_data: dict) -> None:
        self._inventory.reserve(order_id, order_data['items'])

    def release_inventory(self, order_id: str) -> None:
        self._inventory.release_reservation(order_id)

    def process_payment(self, order_id: str, order_data: dict) -> None:
        self._payment.charge(
            order_id=order_id,
            amount=order_data['total_amount'],
            payment_method=order_data['payment_method'],
        )

    def refund_payment(self, order_id: str) -> None:
        self._payment.refund(order_id)

    def schedule_shipping(self, order_id: str, order_data: dict) -> None:
        self._shipping.schedule(
            order_id=order_id,
            address=order_data['shipping_address'],
        )

    def cancel_shipping(self, order_id: str) -> None:
        self._shipping.cancel(order_id)

    def send_confirmation(self, order_id: str, order_data: dict) -> None:
        self._notification.send_order_confirmation(
            order_id=order_id,
            customer_email=order_data['customer_email'],
        )

    def noop(self, order_id: str) -> None:
        pass
```

### 2.7 Choreography vs Orchestration

```
=== Orchestration パターン（上記の Saga）===

  [Saga Orchestrator]
       |
       |---> Inventory Service: "在庫を引き当てて"
       |<--- "完了"
       |
       |---> Payment Service: "決済して"
       |<--- "完了"
       |
       |---> Shipping Service: "配送手配して"
       |<--- "完了"

  メリット: フロー全体が1箇所で把握できる
  デメリット: Orchestrator がSPOF、ロジックが集中


=== Choreography パターン ===

  Order Service --"OrderPlaced"--> [Event Bus]
       |
       +---> Inventory Service: 在庫引当
                 |
                 +--"InventoryReserved"--> [Event Bus]
                       |
                       +---> Payment Service: 決済処理
                                 |
                                 +--"PaymentCompleted"--> [Event Bus]
                                       |
                                       +---> Shipping Service: 配送手配

  メリット: SPOF なし、各サービスが自律的
  デメリット: フロー全体の把握が困難、デバッグが難しい


=== 選択基準 ===

  Orchestration を選ぶ場合:
  - ステップが3つ以上で順序が重要
  - 補償処理が複雑
  - フロー全体の可視性が必要

  Choreography を選ぶ場合:
  - ステップが2-3個でシンプル
  - 各サービスの独立性を最大化したい
  - 新しい消費者の追加が頻繁
```

---

## 3. CQRS (Command Query Responsibility Segregation)

### 3.1 アーキテクチャ

```
                        CQRS アーキテクチャ

  Command Side                              Query Side
  (書き込み)                                (読み取り)

  [POST /orders]                           [GET /orders?user=123]
       |                                        |
  [Command Handler]                        [Query Handler]
       |                                        |
  [Domain Model]                           [Read Model]
  (正規化された                             (非正規化された
   ドメインエンティティ)                      ビュー/プロジェクション)
       |                                        ^
  [Write DB]                               [Read DB]
  (PostgreSQL)                              (Elasticsearch / Redis)
       |                                        |
       +--- Domain Events --->  [Projector] ----+
            (非同期で Read Model を更新)

  ★ 書き込み: ドメインモデルの整合性を重視（正規化）
  ★ 読み取り: クエリの効率を重視（非正規化、インデックス最適化）
  ★ Projector: イベントを受信して Read Model を更新する
```

### 3.2 CQRS の実装

```python
# === Command Side: コマンドハンドラー ===

# application/commands/place_order.py
from dataclasses import dataclass


@dataclass(frozen=True)
class PlaceOrderCommand:
    """注文確定コマンド"""
    customer_id: str
    items: list  # [{"product_id": "...", "quantity": 1}]
    shipping_address: str
    payment_method: str


class PlaceOrderCommandHandler:
    """注文確定コマンドのハンドラー

    Command Handler の責務:
    1. コマンドのバリデーション
    2. ドメインモデルの操作
    3. 永続化
    4. ドメインイベントの発行
    """

    def __init__(self, order_repo, event_publisher, pricing_service):
        self._order_repo = order_repo
        self._event_publisher = event_publisher
        self._pricing = pricing_service

    def handle(self, command: PlaceOrderCommand) -> str:
        # 1. ドメインモデルの生成
        order = Order.create(
            customer_id=command.customer_id,
            items=command.items,
            shipping_address=command.shipping_address,
        )

        # 2. ビジネスルールの適用
        total = self._pricing.calculate_total(order.items)
        order.set_total(total)
        order.confirm()

        # 3. 永続化
        self._order_repo.save(order)

        # 4. ドメインイベントの発行
        for event in order.collect_events():
            self._event_publisher.publish(event)

        return order.id


# === Query Side: クエリハンドラー ===

# application/queries/get_order_summary.py
@dataclass(frozen=True)
class GetOrderSummaryQuery:
    """注文サマリー取得クエリ"""
    customer_id: str
    page: int = 1
    page_size: int = 20


class GetOrderSummaryQueryHandler:
    """注文サマリーのクエリハンドラー

    Query Handler の責務:
    - Read Model（非正規化ビュー）から直接データを返す
    - ドメインロジックは含まない
    - パフォーマンス最適化に集中
    """

    def __init__(self, read_db):
        self._read_db = read_db

    def handle(self, query: GetOrderSummaryQuery) -> dict:
        # Read Model は既に非正規化されているため、
        # JOIN なしで高速にクエリ可能
        results = self._read_db.find(
            collection='order_summaries',
            filter={'customer_id': query.customer_id},
            sort=[('created_at', -1)],
            skip=(query.page - 1) * query.page_size,
            limit=query.page_size,
        )
        total_count = self._read_db.count(
            collection='order_summaries',
            filter={'customer_id': query.customer_id},
        )
        return {
            'orders': list(results),
            'page': query.page,
            'page_size': query.page_size,
            'total_count': total_count,
        }
```

### 3.3 Projector（Read Model の更新）

```python
# infrastructure/projectors/order_projector.py
"""
Projector: ドメインイベントを受信して Read Model を更新する

Write Side のイベント → Projector → Read Side の非正規化ビュー

  OrderPlaced イベント → OrderSummaryProjector
  → order_summaries コレクションに以下を作成:
    {
      "order_id": "ORD-123",
      "customer_id": "USR-456",
      "customer_name": "田中太郎",       ← 非正規化（JOIN 不要）
      "status": "confirmed",
      "item_count": 3,
      "total_amount": 15000,
      "created_at": "2026-01-15T10:30:00Z"
    }
"""
import logging

logger = logging.getLogger(__name__)


class OrderSummaryProjector:
    """注文サマリーの Read Model を管理する Projector"""

    def __init__(self, read_db, customer_repo):
        self._read_db = read_db
        self._customer_repo = customer_repo

    def handle_order_placed(self, event_data: dict) -> None:
        """OrderPlaced イベント → Read Model 作成"""
        data = event_data['data']

        # 顧客情報を取得（非正規化のため）
        customer = self._customer_repo.find_by_id(data['customer_id'])

        summary = {
            'order_id': data['order_id'],
            'customer_id': data['customer_id'],
            'customer_name': customer.name if customer else 'Unknown',
            'customer_email': customer.email if customer else '',
            'status': 'confirmed',
            'items': data['items'],
            'item_count': len(data['items']),
            'total_amount': data['total_amount'],
            'currency': data.get('currency', 'JPY'),
            'created_at': event_data['occurred_at'],
            'updated_at': event_data['occurred_at'],
        }

        self._read_db.upsert(
            collection='order_summaries',
            filter={'order_id': data['order_id']},
            document=summary,
        )
        logger.info(f"Read Model 更新: order_id={data['order_id']}")

    def handle_order_cancelled(self, event_data: dict) -> None:
        """OrderCancelled イベント → Read Model 更新"""
        data = event_data['data']
        self._read_db.update(
            collection='order_summaries',
            filter={'order_id': data['order_id']},
            update={
                '$set': {
                    'status': 'cancelled',
                    'cancel_reason': data.get('reason', ''),
                    'updated_at': event_data['occurred_at'],
                }
            },
        )
        logger.info(f"Read Model 更新 (cancelled): order_id={data['order_id']}")

    def handle_payment_completed(self, event_data: dict) -> None:
        """PaymentCompleted イベント → Read Model 更新"""
        data = event_data['data']
        self._read_db.update(
            collection='order_summaries',
            filter={'order_id': data['order_id']},
            update={
                '$set': {
                    'status': 'paid',
                    'payment_method': data.get('payment_method', ''),
                    'paid_at': event_data['occurred_at'],
                    'updated_at': event_data['occurred_at'],
                }
            },
        )

    def rebuild_all(self) -> None:
        """Read Model の全再構築

        Event Sourcing の強力な利点:
        イベントログから Read Model を任意に再構築できる
        → スキーマ変更、バグ修正後の再構築が容易
        """
        logger.info("Read Model 全再構築開始")
        self._read_db.drop_collection('order_summaries')
        # イベントストアから全イベントを再生
        # （実装は EventStore.load_all_events() を使用）
        logger.info("Read Model 全再構築完了")
```

### 3.4 イベントソーシングの実装

```python
# infrastructure/event_store.py
"""
イベントストア: 全てのドメインイベントを時系列で永続化する

従来の CRUD:
  UPDATE orders SET status = 'shipped' WHERE id = 123;
  → 以前の状態は失われる

イベントソーシング:
  INSERT INTO events (aggregate_id, version, type, data) VALUES
    ('ORD-123', 1, 'OrderCreated',   '{"customer_id": "USR-456"}'),
    ('ORD-123', 2, 'ItemAdded',      '{"product_id": "PRD-789", "qty": 2}'),
    ('ORD-123', 3, 'OrderConfirmed', '{"confirmed_at": "..."}'),
    ('ORD-123', 4, 'OrderShipped',   '{"tracking_id": "..."}');
  → 全ての履歴が保持される。任意時点の状態を復元可能
"""
from sqlalchemy import Column, String, Integer, DateTime, Text, func
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import json


class EventRecord(Base):
    """イベントストアのテーブルモデル"""
    __tablename__ = 'event_store'

    id = Column(Integer, primary_key=True, autoincrement=True)
    aggregate_id = Column(String(100), nullable=False, index=True)
    aggregate_type = Column(String(100), nullable=False)
    version = Column(Integer, nullable=False)
    event_type = Column(String(100), nullable=False)
    data = Column(Text, nullable=False)
    metadata = Column(Text, nullable=True)  # correlation_id, causation_id
    occurred_at = Column(DateTime, nullable=False)
    created_at = Column(
        DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        # 集約ID + バージョンでユニーク制約（楽観的ロック）
        {'unique_together': ('aggregate_id', 'version')},
    )


class ConcurrencyError(Exception):
    """楽観的ロックの競合エラー"""
    pass


class EventStore:
    """イベントストア: 全イベントを時系列で保存

    設計判断:
    - 追記のみ（UPDATE/DELETE 禁止）: イベントは不変の事実
    - 楽観的ロック: version による競合検出
    - 集約単位のストリーム: aggregate_id でグループ化
    """

    def __init__(self, session: Session):
        self._session = session

    def append(
        self,
        aggregate_id: str,
        aggregate_type: str,
        events: list[DomainEvent],
        expected_version: int,
    ) -> None:
        """イベントを追記（楽観的ロック付き）

        Args:
            aggregate_id: 集約の ID
            aggregate_type: 集約の型名
            events: 追記するイベントのリスト
            expected_version: 期待するバージョン（競合検出用）

        Raises:
            ConcurrencyError: バージョン競合時
        """
        current_version = self._get_current_version(aggregate_id)
        if current_version != expected_version:
            raise ConcurrencyError(
                f"Expected version {expected_version}, "
                f"but current version is {current_version}. "
                f"aggregate_id={aggregate_id}"
            )

        for i, event in enumerate(events):
            new_version = expected_version + i + 1
            self._session.add(EventRecord(
                aggregate_id=aggregate_id,
                aggregate_type=aggregate_type,
                version=new_version,
                event_type=event.event_type,
                data=json.dumps(event.to_dict(), ensure_ascii=False, default=str),
                metadata=json.dumps({
                    'correlation_id': event.correlation_id,
                    'causation_id': event.causation_id,
                }),
                occurred_at=event.occurred_at,
            ))
        self._session.commit()

    def load(self, aggregate_id: str) -> list[dict]:
        """集約の全イベントを取得"""
        records = (
            self._session.query(EventRecord)
            .filter_by(aggregate_id=aggregate_id)
            .order_by(EventRecord.version)
            .all()
        )
        return [
            {
                'version': r.version,
                'event_type': r.event_type,
                'data': json.loads(r.data),
                'metadata': json.loads(r.metadata) if r.metadata else {},
                'occurred_at': r.occurred_at,
            }
            for r in records
        ]

    def load_from_version(
        self, aggregate_id: str, from_version: int
    ) -> list[dict]:
        """指定バージョン以降のイベントを取得（スナップショットとの組み合わせ用）"""
        records = (
            self._session.query(EventRecord)
            .filter_by(aggregate_id=aggregate_id)
            .filter(EventRecord.version > from_version)
            .order_by(EventRecord.version)
            .all()
        )
        return [
            {
                'version': r.version,
                'event_type': r.event_type,
                'data': json.loads(r.data),
                'occurred_at': r.occurred_at,
            }
            for r in records
        ]

    def _get_current_version(self, aggregate_id: str) -> int:
        result = (
            self._session.query(func.max(EventRecord.version))
            .filter_by(aggregate_id=aggregate_id)
            .scalar()
        )
        return result or 0


class SnapshotStore:
    """スナップショットストア: 集約の状態を定期的にキャッシュ

    イベントソーシングの課題:
    - イベント数が増えると再生に時間がかかる
    - 例: 注文に1000個のイベントがある場合、毎回全再生は非効率

    解決策: スナップショット
    - N イベントごとに集約の状態をスナップショットとして保存
    - 復元時: スナップショット + それ以降のイベント再生
    """

    def __init__(self, session: Session):
        self._session = session

    def save_snapshot(
        self, aggregate_id: str, version: int, state: dict
    ) -> None:
        """スナップショットを保存"""
        self._session.execute(
            """
            INSERT INTO snapshots (aggregate_id, version, state, created_at)
            VALUES (:agg_id, :version, :state, :created_at)
            ON CONFLICT (aggregate_id) DO UPDATE SET
                version = :version,
                state = :state,
                created_at = :created_at
            """,
            {
                'agg_id': aggregate_id,
                'version': version,
                'state': json.dumps(state, ensure_ascii=False),
                'created_at': datetime.now(timezone.utc),
            }
        )
        self._session.commit()

    def load_snapshot(self, aggregate_id: str) -> dict | None:
        """最新のスナップショットを取得"""
        result = self._session.execute(
            "SELECT version, state FROM snapshots WHERE aggregate_id = :agg_id",
            {'agg_id': aggregate_id},
        ).fetchone()
        if result:
            return {
                'version': result[0],
                'state': json.loads(result[1]),
            }
        return None
```

### 3.5 Event Sourced Aggregate

```python
# domain/models/order.py
"""
Event Sourced Aggregate: イベントソーシングで管理される集約

従来の集約: 現在の状態を直接保持
Event Sourced 集約: イベントを適用して状態を構築

状態の変更は必ずイベント経由:
  order.add_item(product, qty)
    → ItemAddedToOrder イベントを生成
    → イベントを自身に適用して状態更新
    → 永続化時にイベントをイベントストアに追記
"""
from dataclasses import dataclass, field


class EventSourcedAggregate:
    """イベントソーシング対応の集約基底クラス"""

    def __init__(self):
        self._version: int = 0
        self._pending_events: list[DomainEvent] = []

    @property
    def version(self) -> int:
        return self._version

    def collect_events(self) -> list[DomainEvent]:
        """未永続化のイベントを取得してクリア"""
        events = self._pending_events.copy()
        self._pending_events.clear()
        return events

    def _apply_event(self, event: DomainEvent) -> None:
        """イベントを適用して状態を更新（サブクラスで実装）"""
        handler_name = f"_on_{self._to_snake_case(type(event).__name__)}"
        handler = getattr(self, handler_name, None)
        if handler:
            handler(event)
        self._version += 1

    def _raise_event(self, event: DomainEvent) -> None:
        """新しいイベントを発生させる"""
        self._apply_event(event)
        self._pending_events.append(event)

    def _load_from_history(self, events: list[DomainEvent]) -> None:
        """イベント履歴から状態を復元"""
        for event in events:
            self._apply_event(event)

    @staticmethod
    def _to_snake_case(name: str) -> str:
        import re
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


class Order(EventSourcedAggregate):
    """注文集約（Event Sourced）"""

    def __init__(self):
        super().__init__()
        self.id: str = ""
        self.customer_id: str = ""
        self.items: list = []
        self.status: str = ""
        self.total_amount: int = 0
        self.shipping_address: str = ""
        self.created_at: str = ""
        self.confirmed_at: str = ""
        self.shipped_at: str = ""
        self.cancelled_at: str = ""

    # === コマンドメソッド（イベントを発生させる）===

    @classmethod
    def create(cls, order_id: str, customer_id: str) -> 'Order':
        order = cls()
        order._raise_event(OrderCreated(
            order_id=order_id,
            customer_id=customer_id,
        ))
        return order

    def add_item(
        self, product_id: str, quantity: int, unit_price: int
    ) -> None:
        if self.status != "created":
            raise ValueError("確定済みの注文にはアイテムを追加できません")
        if quantity <= 0:
            raise ValueError("数量は1以上である必要があります")

        self._raise_event(ItemAddedToOrder(
            order_id=self.id,
            product_id=product_id,
            quantity=quantity,
            unit_price=unit_price,
        ))

    def confirm(self) -> None:
        if self.status != "created":
            raise ValueError(f"状態 '{self.status}' から確定はできません")
        if not self.items:
            raise ValueError("アイテムが空の注文は確定できません")

        self._raise_event(OrderConfirmed(
            order_id=self.id,
            confirmed_at=datetime.now(timezone.utc).isoformat(),
        ))

    def ship(self, tracking_id: str) -> None:
        if self.status != "confirmed":
            raise ValueError(f"状態 '{self.status}' から出荷はできません")

        self._raise_event(OrderShipped(
            order_id=self.id,
            tracking_id=tracking_id,
            shipped_at=datetime.now(timezone.utc).isoformat(),
        ))

    def cancel(self, reason: str, cancelled_by: str = "customer") -> None:
        if self.status in ("shipped", "cancelled"):
            raise ValueError(f"状態 '{self.status}' からキャンセルはできません")

        self._raise_event(OrderCancelled(
            order_id=self.id,
            reason=reason,
            cancelled_by=cancelled_by,
        ))

    # === イベントハンドラー（状態を更新する）===

    def _on_order_created(self, event: OrderCreated) -> None:
        self.id = event.order_id
        self.customer_id = event.customer_id
        self.status = "created"
        self.items = []
        self.total_amount = 0

    def _on_item_added_to_order(self, event: ItemAddedToOrder) -> None:
        self.items.append({
            'product_id': event.product_id,
            'quantity': event.quantity,
            'unit_price': event.unit_price,
        })
        self.total_amount += event.quantity * event.unit_price

    def _on_order_confirmed(self, event: OrderConfirmed) -> None:
        self.status = "confirmed"
        self.confirmed_at = event.confirmed_at

    def _on_order_shipped(self, event) -> None:
        self.status = "shipped"
        self.shipped_at = event.shipped_at

    def _on_order_cancelled(self, event: OrderCancelled) -> None:
        self.status = "cancelled"
        self.cancelled_at = datetime.now(timezone.utc).isoformat()

    # === リポジトリで使用 ===

    @classmethod
    def from_events(cls, events: list[DomainEvent]) -> 'Order':
        """イベント履歴から集約を復元"""
        order = cls()
        order._load_from_history(events)
        return order
```

---

## 4. テスト戦略

### 4.1 イベントハンドラーのテスト

```python
# tests/test_inventory_handler.py
import pytest


class FakeInventoryRepo:
    def __init__(self):
        self._reservations = set()

    def reservation_exists(self, order_id: str) -> bool:
        return order_id in self._reservations

    def add_reservation(self, order_id: str) -> None:
        self._reservations.add(order_id)


class FakeStockService:
    def __init__(self, available_products: set = None):
        self._available = available_products or set()
        self._reserved = {}
        self._released = []

    def reserve(
        self, product_id: str, quantity: int, reservation_id: str
    ) -> bool:
        if product_id not in self._available:
            return False
        self._reserved[reservation_id] = self._reserved.get(
            reservation_id, []
        )
        self._reserved[reservation_id].append(
            {'product_id': product_id, 'quantity': quantity}
        )
        return True

    def release_reservation(self, reservation_id: str) -> None:
        self._released.append(reservation_id)

    def release(self, product_id: str, reservation_id: str) -> None:
        self._released.append(
            {'product_id': product_id, 'reservation_id': reservation_id}
        )


class TestInventoryEventHandler:
    """在庫イベントハンドラーのテスト"""

    def setup_method(self):
        self.repo = FakeInventoryRepo()
        self.stock = FakeStockService(
            available_products={'PRD-001', 'PRD-002'}
        )
        self.publisher = InMemoryEventPublisher()
        self.handler = InventoryEventHandler(
            inventory_repo=self.repo,
            stock_service=self.stock,
            event_publisher=self.publisher,
        )

    def test_在庫引当_正常(self):
        """全アイテムの在庫が十分な場合、引当が成功する"""
        event_data = {
            'event_id': 'evt-001',
            'event_type': 'order.placed',
            'correlation_id': 'corr-001',
            'data': {
                'order_id': 'ORD-123',
                'items': [
                    {'product_id': 'PRD-001', 'quantity': 2},
                    {'product_id': 'PRD-002', 'quantity': 1},
                ],
            },
        }

        self.handler.handle_order_placed(event_data)

        # InventoryReserved イベントが発行される
        reserved_events = self.publisher.get_events_of_type(
            'inventory.reserved'
        )
        assert len(reserved_events) == 1
        assert reserved_events[0].order_id == 'ORD-123'

    def test_在庫不足_補償イベント発行(self):
        """在庫不足の場合、補償イベントが発行される"""
        event_data = {
            'event_id': 'evt-002',
            'event_type': 'order.placed',
            'correlation_id': 'corr-002',
            'data': {
                'order_id': 'ORD-456',
                'items': [
                    {'product_id': 'PRD-001', 'quantity': 1},
                    {'product_id': 'PRD-999', 'quantity': 1},  # 在庫なし
                ],
            },
        }

        self.handler.handle_order_placed(event_data)

        # 在庫不足イベントが発行される
        insufficient_events = self.publisher.get_events_of_type(
            'inventory.insufficient'
        )
        assert len(insufficient_events) == 1

    def test_冪等性_重複イベント無視(self):
        """同じ order_id のイベントが2回来ても、1回だけ処理される"""
        self.repo.add_reservation('ORD-789')

        event_data = {
            'event_id': 'evt-003',
            'event_type': 'order.placed',
            'data': {
                'order_id': 'ORD-789',
                'items': [{'product_id': 'PRD-001', 'quantity': 1}],
            },
        }

        self.handler.handle_order_placed(event_data)

        # 何もイベントが発行されない（スキップ）
        assert len(self.publisher.published_events) == 0
```

### 4.2 Event Sourced Aggregate のテスト

```python
# tests/test_order_aggregate.py
import pytest


class TestOrderAggregate:
    """Event Sourced Order 集約のテスト"""

    def test_注文作成(self):
        order = Order.create(order_id='ORD-001', customer_id='USR-001')

        assert order.id == 'ORD-001'
        assert order.customer_id == 'USR-001'
        assert order.status == 'created'
        assert order.version == 1

        events = order.collect_events()
        assert len(events) == 1
        assert events[0].event_type == 'order.created'

    def test_アイテム追加(self):
        order = Order.create(order_id='ORD-001', customer_id='USR-001')
        order.add_item('PRD-001', quantity=2, unit_price=1000)
        order.add_item('PRD-002', quantity=1, unit_price=2000)

        assert len(order.items) == 2
        assert order.total_amount == 4000  # 2*1000 + 1*2000
        assert order.version == 3

    def test_注文確定(self):
        order = Order.create(order_id='ORD-001', customer_id='USR-001')
        order.add_item('PRD-001', quantity=1, unit_price=1000)
        order.confirm()

        assert order.status == 'confirmed'

    def test_空の注文は確定できない(self):
        order = Order.create(order_id='ORD-001', customer_id='USR-001')

        with pytest.raises(ValueError, match="アイテムが空"):
            order.confirm()

    def test_出荷済み注文はキャンセルできない(self):
        order = Order.create(order_id='ORD-001', customer_id='USR-001')
        order.add_item('PRD-001', quantity=1, unit_price=1000)
        order.confirm()
        order.ship(tracking_id='TRK-001')

        with pytest.raises(ValueError, match="キャンセルはできません"):
            order.cancel(reason="気が変わった")

    def test_イベント履歴から復元(self):
        """Event Sourcing の核心: イベント再生で状態を復元"""
        # 1. 集約を操作してイベントを生成
        original = Order.create(order_id='ORD-001', customer_id='USR-001')
        original.add_item('PRD-001', quantity=2, unit_price=1000)
        original.add_item('PRD-002', quantity=1, unit_price=2000)
        original.confirm()
        events = original.collect_events()

        # 2. イベント履歴から集約を復元
        restored = Order.from_events(events)

        # 3. 復元した集約は元と同じ状態
        assert restored.id == original.id
        assert restored.customer_id == original.customer_id
        assert restored.status == original.status
        assert restored.total_amount == original.total_amount
        assert len(restored.items) == len(original.items)
        assert restored.version == original.version
```

### 4.3 Saga のテスト

```python
# tests/test_order_saga.py
import pytest


class FakeService:
    """テスト用のフェイクサービス"""
    def __init__(self, should_fail: bool = False):
        self._should_fail = should_fail
        self.calls = []

    def __getattr__(self, name):
        def method(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if self._should_fail and name == self._fail_method:
                raise RuntimeError(f"{name} failed")
        return method

    def set_fail_method(self, method_name: str):
        self._should_fail = True
        self._fail_method = method_name


class FakeSagaLogRepo:
    def __init__(self):
        self.logs = {}

    def create(self, saga_id, order_id, saga_type):
        self.logs[saga_id] = {'order_id': order_id, 'steps': {}}

    def update_step(self, saga_id, step_name, status):
        self.logs[saga_id]['steps'][step_name] = status

    def complete(self, saga_id):
        self.logs[saga_id]['completed'] = True


class TestOrderPlacementSaga:

    def test_正常完了(self):
        """全ステップが成功する場合"""
        inventory = FakeService()
        payment = FakeService()
        shipping = FakeService()
        notification = FakeService()
        saga_log = FakeSagaLogRepo()

        saga = OrderPlacementSaga(
            inventory_service=inventory,
            payment_service=payment,
            shipping_service=shipping,
            notification_service=notification,
            saga_log_repo=saga_log,
        )

        saga.execute('ORD-001', {
            'items': [{'product_id': 'PRD-001', 'quantity': 1}],
            'total_amount': 1000,
            'payment_method': 'credit_card',
            'shipping_address': '東京都渋谷区...',
            'customer_email': 'test@example.com',
        })

        assert saga._status == SagaStatus.COMPLETED

    def test_決済失敗時の補償(self):
        """決済が失敗した場合、在庫引当の補償が実行される"""
        inventory = FakeService()
        payment = FakeService()
        payment.set_fail_method('charge')
        shipping = FakeService()
        notification = FakeService()
        saga_log = FakeSagaLogRepo()

        saga = OrderPlacementSaga(
            inventory_service=inventory,
            payment_service=payment,
            shipping_service=shipping,
            notification_service=notification,
            saga_log_repo=saga_log,
        )

        with pytest.raises(SagaFailedError) as exc_info:
            saga.execute('ORD-002', {
                'items': [{'product_id': 'PRD-001', 'quantity': 1}],
                'total_amount': 1000,
                'payment_method': 'credit_card',
                'shipping_address': '東京都渋谷区...',
                'customer_email': 'test@example.com',
            })

        assert exc_info.value.step_name == 'process_payment'
        assert saga._status == SagaStatus.FAILED

        # 在庫の補償（解放）が呼ばれていることを確認
        inventory_calls = [c[0] for c in inventory.calls]
        assert 'release_reservation' in inventory_calls
```

---

## 5. 運用と監視

### 5.1 Dead Letter Queue (DLQ)

```
=== Dead Letter Queue の仕組み ===

  通常フロー:
  [Event Bus] → [Consumer] → 処理成功 → オフセットコミット

  DLQ フロー:
  [Event Bus] → [Consumer] → 処理失敗 (3回リトライ)
                                 ↓
                           [Dead Letter Queue]
                                 ↓
                           [DLQ Consumer / 手動確認]
                                 ↓
                           修正後に再処理 or 破棄

  DLQ に入るケース:
  - メッセージフォーマットの不正（デシリアライズ失敗）
  - ビジネスルール違反（存在しない注文の更新）
  - 一時的でない障害（外部 API の認証エラー等）

  DLQ の監視項目:
  - DLQ のメッセージ数（急増はシステム障害の兆候）
  - DLQ 内のメッセージの滞留時間
  - DLQ から再処理した成功率
```

### 5.2 監視とトレーシング

```
=== イベント駆動システムの可観測性 ===

  1. 分散トレーシング (OpenTelemetry)
     ┌──────────────────────────────────────┐
     │ Trace ID: abc-123                     │
     │                                       │
     │ [API Gateway] ─── 5ms ──┐             │
     │                          │             │
     │ [Order Service] ─── 20ms ──┐           │
     │                             │           │
     │ [Kafka Publish] ─── 3ms ──┐ │           │
     │                            │ │           │
     │ [Inventory Handler] ─── 50ms │           │
     │                               │           │
     │ [Payment Handler] ─── 200ms  │            │
     │                                           │
     │ Total: 278ms                              │
     └──────────────────────────────────────┘

  2. メトリクス (Prometheus + Grafana)
     - events_published_total (カウンター): 発行イベント数
     - events_consumed_total (カウンター): 消費イベント数
     - event_processing_duration_seconds (ヒストグラム): 処理時間
     - consumer_lag (ゲージ): Consumer の遅延
     - dlq_messages_total (カウンター): DLQ に入ったメッセージ数

  3. アラート条件
     - Consumer Lag > 10,000: Consumer が追いついていない
     - DLQ 増加率 > 100/min: 大規模障害の兆候
     - 処理時間 P99 > 5s: パフォーマンス劣化
     - エラー率 > 5%: ハンドラーのバグ or 依存サービス障害
```

### 5.3 冪等性の実装パターン

```python
# infrastructure/idempotency/idempotency_store.py
"""
冪等性: 同じイベントを複数回処理しても結果が変わらないことを保証

なぜ冪等性が必要か:
  - at-least-once 配信: Kafka は「少なくとも1回」配信を保証
  - Consumer 再起動時にオフセットが巻き戻る可能性
  - ネットワーク障害によるリトライ

冪等性の実装方法:
  1. event_id で処理済みチェック（推奨）
  2. 自然キーによる重複排除（例: order_id + event_type）
  3. 条件付き更新（例: WHERE version = expected_version）
"""


class IdempotencyStore:
    """冪等性ストア: 処理済みイベントを記録"""

    def __init__(self, session):
        self._session = session

    def is_processed(self, event_id: str) -> bool:
        """イベントが処理済みか確認"""
        result = self._session.execute(
            "SELECT 1 FROM processed_events WHERE event_id = :event_id",
            {'event_id': event_id},
        ).fetchone()
        return result is not None

    def mark_processed(self, event_id: str, handler_name: str) -> None:
        """イベントを処理済みとして記録"""
        self._session.execute(
            """
            INSERT INTO processed_events (event_id, handler_name, processed_at)
            VALUES (:event_id, :handler_name, :processed_at)
            ON CONFLICT (event_id, handler_name) DO NOTHING
            """,
            {
                'event_id': event_id,
                'handler_name': handler_name,
                'processed_at': datetime.now(timezone.utc),
            },
        )
        self._session.commit()


def idempotent_handler(idempotency_store: IdempotencyStore):
    """冪等性を保証するデコレーター"""
    def decorator(func):
        def wrapper(event_data: dict) -> None:
            event_id = event_data.get('event_id', '')
            handler_name = func.__qualname__

            if idempotency_store.is_processed(event_id):
                logger.info(
                    f"処理済みイベントをスキップ: "
                    f"event_id={event_id}, handler={handler_name}"
                )
                return

            func(event_data)
            idempotency_store.mark_processed(event_id, handler_name)
        return wrapper
    return decorator
```

---

## 6. 比較表

### 6.1 同期 vs 非同期

| 特性 | 同期 (REST/gRPC) | 非同期 (Event-Driven) |
|-----|:----------------:|:--------------------:|
| 結合度 | 高（直接呼び出し） | 低（イベントバス経由） |
| レイテンシ | 全サービスの合計 | 即座に応答可能 |
| 耐障害性 | カスケード障害リスク | サービスごとに独立 |
| データ整合性 | 強い整合性（可能） | 結果整合性 |
| デバッグ | 容易（同期フロー） | 困難（非同期フロー追跡） |
| スケーラビリティ | ボトルネック発生 | 独立スケール |
| 学習コスト | 低 | 高 |
| 運用コスト | 低 | 高（Kafka 等の運用） |

### 6.2 状態管理アプローチ

| アプローチ | 状態管理 | 監査ログ | 複雑性 | 適用場面 |
|-----------|---------|---------|--------|---------|
| CRUD | 最新状態のみ保持 | 別途実装が必要 | 低 | シンプルなアプリ |
| CQRS | 読み書き分離 | 別途実装が必要 | 中 | 読み書きの負荷特性が異なる |
| Event Sourcing | イベントログから再構築 | 自然に実現 | 高 | 完全な監査証跡が必要 |
| CQRS + ES | イベントログ + Read Model | 自然に実現 | 最高 | 金融・医療・法規制 |

### 6.3 メッセージブローカー比較

| 特性 | Apache Kafka | RabbitMQ | AWS SNS+SQS |
|-----|:-----------:|:--------:|:-----------:|
| スループット | 非常に高い（100万msg/s） | 高い（1万msg/s） | 高い（マネージド） |
| メッセージ保持 | 設定期間保持（再読可） | 消費後削除 | SQS: 最大14日 |
| 順序保証 | パーティション内保証 | キュー内保証 | FIFO SQS で保証 |
| Consumer Group | ネイティブサポート | 手動設定 | SQS がキュー分離 |
| 運用コスト | 高い（自前運用） | 中程度 | 低い（マネージド） |
| 適用場面 | 大規模ストリーミング | タスクキュー | AWS エコシステム |

### 6.4 Saga パターン比較

| 特性 | Orchestration | Choreography |
|-----|:------------:|:------------:|
| 制御の集中度 | 高い（Orchestrator） | 低い（分散） |
| フロー可視性 | 高い | 低い |
| SPOF リスク | あり（Orchestrator） | なし |
| デバッグ容易性 | 高い | 低い |
| サービス独立性 | 中程度 | 高い |
| 適用場面 | 3ステップ以上の複雑なフロー | 2-3ステップのシンプルなフロー |

---

## 7. アンチパターン

### アンチパターン 1: イベントを RPC の代替として使う

```
WHY: イベントは「起きた事実」を伝えるもの。
     「〜してくれ」という命令はイベントではなくコマンド。
     イベントで同期的な応答を期待すると、疎結合の利点が失われる。

BAD: イベントで同期的なリクエスト/レスポンスを模倣
  Order Service --> "PleaseReserveInventory" --> Inventory Service
  Order Service <-- "InventoryReserved" <-- Inventory Service
  → 命令形のイベント名。実質的に同期呼び出し
  → Order Service が Inventory Service の応答を待っている

GOOD: イベントは「起きた事実」を伝える
  Order Service --> "OrderPlaced" --> [Event Bus]
  → Inventory Service が独自に判断して在庫引当
  → Order Service は結果を待たない
  → 過去形のイベント名（OrderPlaced, PaymentCompleted）
```

### アンチパターン 2: 全てをイベント駆動にする

```
WHY: イベント駆動は万能ではない。
     同期処理が適切な場面（認証、データ参照）に
     イベントを使うと、不要な複雑性とレイテンシが発生する。

BAD: ユーザー認証やデータ取得もイベント駆動
  User --> "AuthenticateUser" --> [Event Bus] --> Auth Service
  → ログインに数秒のレイテンシ
  → 単純な GET リクエストにイベントバス経由

GOOD: 適材適所
  - 同期 (REST/gRPC):
    ユーザー認証、データ参照、リアルタイム応答が必要な処理
  - 非同期 (Event):
    注文処理、通知送信、データ同期、バッチ処理、
    複数サービスへのファンアウト
```

### アンチパターン 3: イベントにドメインロジックを含める

```
WHY: イベントは「何が起きたか」を記録するだけ。
     「どう処理するか」はConsumer のドメインロジック。
     イベントに処理ロジックを含めると、Consumer の自律性が失われる。

BAD: イベントに処理指示を含める
  {
    "event_type": "order.placed",
    "order_id": "ORD-123",
    "instructions": {
      "inventory": "reserve PRD-001 x 2",     ← 処理指示
      "payment": "charge 3000 JPY",            ← 処理指示
      "shipping": "express delivery to ..."     ← 処理指示
    }
  }
  → 発信者が全ての Consumer の処理方法を知っている（密結合）

GOOD: イベントは事実のみ
  {
    "event_type": "order.placed",
    "order_id": "ORD-123",
    "customer_id": "USR-456",
    "items": [{"product_id": "PRD-001", "quantity": 2}],
    "total_amount": 3000
  }
  → 各 Consumer が自律的に処理方法を決定
```

### アンチパターン 4: Outbox パターンなしでのイベント発行

```
WHY: DB 保存とイベント発行を別々のトランザクションで行うと、
     どちらかが失敗した場合にデータ不整合が発生する。

BAD: 2つの独立した操作
  def place_order(order):
      db.save(order)          # 1. DB に保存（成功）
      kafka.publish(event)    # 2. Kafka に発行（失敗する可能性）
      # → DB には保存されたが、イベントは発行されない
      # → 下流サービスは注文を認識しない

GOOD: Outbox パターン
  def place_order(order):
      with db.transaction():
          db.save(order)          # 1. 注文を保存
          outbox.save(event)      # 2. Outbox にイベントを保存
      # → 同一トランザクション内で原子的に保存
      # → Outbox Relay が別プロセスで Kafka に転送
```

### アンチパターン 5: 冪等性を考慮しないハンドラー

```
WHY: at-least-once 配信では同じイベントが複数回届く可能性がある。
     冪等性がないと、重複処理（二重課金、二重在庫引当）が発生する。

BAD: 冪等性なし
  def handle_payment(event):
      payment_service.charge(
          order_id=event['order_id'],
          amount=event['total_amount'],
      )
      # → 同じイベントが2回届くと、2回課金される

GOOD: 冪等性保証
  def handle_payment(event):
      if idempotency_store.is_processed(event['event_id']):
          return  # 処理済みならスキップ

      payment_service.charge(
          order_id=event['order_id'],
          amount=event['total_amount'],
      )
      idempotency_store.mark_processed(event['event_id'])
```

---

## 8. 演習問題

### 演習1: 基本 — イベント定義とパブリッシャー（30分）

**課題**: ECサイトの「商品レビュー投稿」のイベント駆動設計

以下のイベントを定義し、パブリッシャーを実装せよ:
1. `ReviewSubmitted`: レビュー投稿（review_id, product_id, user_id, rating, comment）
2. `ReviewApproved`: レビュー承認（review_id, approved_by）
3. `ReviewRejected`: レビュー却下（review_id, reason）

イベントハンドラーとして以下を実装:
- `ProductRatingUpdater`: ReviewApproved 受信時に商品の平均評価を更新
- `NotificationHandler`: ReviewApproved 受信時に投稿者に通知

**期待する出力**:
```python
# ReviewSubmitted イベントが発行される
publisher.publish(ReviewSubmitted(
    review_id="REV-001",
    product_id="PRD-001",
    user_id="USR-001",
    rating=5,
    comment="素晴らしい商品です"
))

# ProductRatingUpdater が平均評価を更新
# → product PRD-001 の平均評価が 4.5 に更新
# NotificationHandler が通知を送信
# → user USR-001 に「レビューが承認されました」通知
```

### 演習2: 応用 — Saga パターンの実装（60分）

**課題**: 旅行予約システムの Saga を実装せよ

ステップ:
1. フライト予約（reserve_flight / cancel_flight）
2. ホテル予約（reserve_hotel / cancel_hotel）
3. レンタカー予約（reserve_car / cancel_car）
4. 決済処理（charge_payment / refund_payment）

要件:
- ホテル予約で失敗した場合、フライト予約の補償が実行されること
- 各ステップの状態をログに記録すること
- テストを3ケース以上書くこと

**期待する出力**:
```python
# 正常系
saga.execute("TRIP-001", trip_data)
# → 全ステップ完了

# ホテル予約失敗時
saga.execute("TRIP-002", trip_data)
# → SagaFailedError: "Saga failed at step 'reserve_hotel': No rooms"
# → フライト予約の補償（キャンセル）が実行済み
```

### 演習3: 発展 — Event Sourced 集約の実装（90分）

**課題**: 銀行口座を Event Sourced Aggregate として実装せよ

イベント:
- `AccountOpened(account_id, owner_name, initial_balance)`
- `MoneyDeposited(account_id, amount, description)`
- `MoneyWithdrawn(account_id, amount, description)`
- `AccountFrozen(account_id, reason)`
- `AccountClosed(account_id)`

ビジネスルール:
- 残高がマイナスになる引き出しは拒否
- 凍結中の口座は入出金不可
- 閉鎖済みの口座は操作不可
- 1日の引き出し上限は100万円

テスト要件:
- イベント履歴からの状態復元テスト
- ビジネスルール違反の拒否テスト
- スナップショットからの復元テスト

**期待する出力**:
```python
# 口座を操作
account = BankAccount.open("ACC-001", "田中太郎", 100000)
account.deposit(50000, "給与振込")
account.withdraw(30000, "ATM引き出し")

assert account.balance == 120000
assert account.version == 3

# イベント履歴から復元
events = account.collect_events()
restored = BankAccount.from_events(events)
assert restored.balance == 120000

# ビジネスルール違反
with pytest.raises(ValueError, match="残高不足"):
    account.withdraw(200000, "大口引き出し")

# 凍結中は操作不可
account.freeze("不正利用の疑い")
with pytest.raises(ValueError, match="凍結中"):
    account.deposit(10000, "振込")
```

---

## 9. FAQ

### Q1. イベントの順序保証はどう実現する？

**A.** Kafka ではパーティションキーに集約IDを使うことで、同一集約のイベントは同一パーティション内で順序保証される。異なる集約間の順序は保証不要（各集約は独立しているため）。順序が必要な場合はイベントに `version` フィールドを含め、Consumer 側で順序検証を行う。

```python
# パーティションキーに aggregate_id を使用
producer.produce(
    topic="domain-events.order",
    key=order_id.encode(),  # ← パーティションキー
    value=event_payload,
)
# → 同じ order_id のイベントは必ず同じパーティションに入る
# → パーティション内では FIFO（先入れ先出し）保証

# Consumer 側の順序検証
def handle_event(event_data):
    expected_version = get_last_processed_version(event_data['aggregate_id'])
    actual_version = event_data['version']
    if actual_version <= expected_version:
        return  # 既に処理済み（冪等性）
    if actual_version > expected_version + 1:
        raise OutOfOrderError("イベントが飛んでいる。再処理が必要")
```

### Q2. 結果整合性で「不整合な期間」はどう扱う？

**A.** UIレベルとビジネスレベルの両方で対処する。

```
UI レベルの対処:
  1. 楽観的 UI: 「注文を受け付けました（処理中）」と即座に表示
  2. ポーリング / WebSocket: バックグラウンドで最新状態を通知
  3. 状態表示: "processing" → "confirmed" → "shipped" の遷移を表示

ビジネスレベルの対処:
  1. SLA の定義: 「注文確定から在庫引当まで最大30秒」
  2. タイムアウト: 指定時間内に処理完了しない場合はアラート
  3. 補償処理: 不整合を検出したら自動 or 手動で修正
  4. ドメインエキスパートとの合意: 「数秒〜数分の遅延は許容か？」

実例（ECサイト）:
  - 注文確定: 即座にレスポンス（「注文を受け付けました」）
  - 在庫引当: 5秒以内（非同期）
  - 決済処理: 10秒以内（非同期）
  - 発送通知: 数分〜数時間（ビジネス上許容）
```

### Q3. イベントスキーマの変更（バージョニング）はどう管理する？

**A.** 以下の4つの戦略を組み合わせる:

```python
# 戦略1: 後方互換を維持（推奨）
# フィールド追加は OK、削除・型変更は NG
@dataclass(frozen=True)
class OrderPlacedV1(DomainEvent):
    event_type: str = "order.placed"
    order_id: str = ""
    total_amount: int = 0

@dataclass(frozen=True)
class OrderPlacedV2(DomainEvent):  # V1 との後方互換あり
    event_type: str = "order.placed"
    order_id: str = ""
    total_amount: int = 0
    currency: str = "JPY"     # ← 新規追加（デフォルト値あり）
    discount_amount: int = 0  # ← 新規追加（デフォルト値あり）

# 戦略2: バージョン付きイベントタイプ（破壊的変更が必要な場合）
# "order.placed.v1" → "order.placed.v2"

# 戦略3: アップキャスター（古いバージョンを新しいバージョンに変換）
class EventUpCaster:
    def upcast(self, event_data: dict) -> dict:
        event_type = event_data.get('event_type', '')
        version = event_data.get('schema_version', 1)

        if event_type == 'order.placed' and version == 1:
            # V1 → V2 への変換
            event_data['currency'] = 'JPY'  # デフォルト値
            event_data['discount_amount'] = 0
            event_data['schema_version'] = 2
        return event_data

# 戦略4: Schema Registry（Confluent Schema Registry）
# → スキーマの一元管理と互換性の自動チェック
```

### Q4. Kafka のパーティション数はどう決める？

**A.** Consumer の並列度とスループット要件から決定する。

```
基本方針:
  パーティション数 >= Consumer インスタンス数

  例: Consumer を 10 インスタンスで運用したい場合
  → パーティション数 = 10 以上

  スループット計算:
  - 単一パーティションの書き込み: 約 10MB/s
  - 目標スループット: 100MB/s
  → パーティション数 = 100MB/s ÷ 10MB/s = 10

注意点:
  - パーティション数は増やせるが減らせない
  - 過剰なパーティション数はメタデータのオーバーヘッド
  - 推奨: 初期はトピックあたり 6-12 パーティション
  - 大規模: 数百パーティションも可能（Kafka の限界は数千）
```

### Q5. イベント駆動とマイクロサービスの関係は？

**A.** イベント駆動はマイクロサービス間の通信パターンの1つ。マイクロサービスが必須ではないが、相性が非常に良い。

```
モノリスでもイベント駆動は使える:
  [モジュールA] --event--> [EventBus(in-process)] --> [モジュールB]
  → モジュール間の疎結合を実現
  → 将来のマイクロサービス化の布石

マイクロサービス + イベント駆動:
  [Service A] --event--> [Kafka] --> [Service B]
  → サービス間の完全な疎結合
  → 独立デプロイ、独立スケール

マイクロサービスなしのイベント駆動:
  - 社内ツールの通知システム
  - バッチ処理のパイプライン
  - IoT デバイスのデータ収集
```

### Q6. イベントソーシングの「イベント数が膨大になる問題」はどう対処する？

**A.** スナップショットとアーカイブの組み合わせで対処する。

```
1. スナップショット（前述）:
   - N イベントごとに集約の状態をスナップショット保存
   - 復元時: スナップショット + それ以降のイベント再生
   - 推奨: 100 イベントごとにスナップショット

2. アーカイブ:
   - 古いイベントをコールドストレージ（S3 等）に移動
   - スナップショット以前のイベントはアーカイブ対象
   - 必要時にアーカイブから復元可能

3. Read Model の活用:
   - 通常のクエリは Read Model（Projector が更新）を使用
   - イベントストアは書き込みと監査用途に限定
   - Read Model は非正規化されており高速

実例:
  - 1集約あたり平均 50 イベント/年
  - 100万集約 → 5,000万イベント/年
  - スナップショット + 3年アーカイブで
    アクティブなイベント数を管理可能な範囲に維持
```

### Q7. テスト環境でのイベント駆動システムはどう構築する？

**A.** テスト用のインメモリ実装と Testcontainers を組み合わせる。

```python
# 単体テスト: インメモリ実装
class TestOrderWorkflow:
    def setup_method(self):
        self.publisher = InMemoryEventPublisher()
        self.handler = InventoryEventHandler(
            inventory_repo=FakeInventoryRepo(),
            stock_service=FakeStockService(),
            event_publisher=self.publisher,
        )

# 統合テスト: Testcontainers で Kafka を起動
import testcontainers.kafka

class TestKafkaIntegration:
    @classmethod
    def setup_class(cls):
        cls.kafka = testcontainers.kafka.KafkaContainer()
        cls.kafka.start()
        cls.bootstrap_servers = cls.kafka.get_bootstrap_server()

    @classmethod
    def teardown_class(cls):
        cls.kafka.stop()

    def test_publish_and_consume(self):
        publisher = KafkaEventPublisher(self.bootstrap_servers)
        # ... 統合テスト
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| イベント駆動の利点 | 疎結合、独立スケール、耐障害性 |
| 3つのパターン | Event Notification / Event-Carried State Transfer / Event Sourcing |
| Pub/Sub | プロデューサーとコンシューマーの完全な分離 |
| CQRS | 読み書きの最適化を独立して行う |
| Saga パターン | 分散トランザクションの代替（補償による結果整合性） |
| Outbox パターン | DB 保存とイベント発行のトランザクション保証 |
| 冪等性 | at-least-once 配信での重複処理防止 |
| Projector | イベントから Read Model を構築・更新する |
| スナップショット | イベントソーシングのパフォーマンス最適化 |
| DLQ | 処理失敗イベントの隔離と再処理 |
| トレードオフ | 結果整合性、デバッグの困難さ、運用の複雑性 |

---

## 次に読むべきガイド

- [メッセージキュー](../01-components/02-message-queue.md) — イベントバスの実装基盤（Kafka / RabbitMQ の詳細）
- [DDD](./02-ddd.md) — ドメインイベントの設計元となるドメインモデリング
- [クリーンアーキテクチャ](./01-clean-architecture.md) — イベントハンドラーの配置とレイヤー設計
- [URL短縮サービス](../03-case-studies/00-url-shortener.md) — イベント駆動を使わないシンプルなシステム設計の例
- [チャットシステム](../03-case-studies/01-chat-system.md) — WebSocket とイベント駆動の組み合わせ
- [通知システム](../03-case-studies/02-notification-system.md) — Pub/Sub の典型的な適用例

---

## 参考文献

1. **Building Event-Driven Microservices** — Adam Bellemare (O'Reilly, 2020) — イベント駆動マイクロサービスの包括的ガイド
2. **Designing Data-Intensive Applications** — Martin Kleppmann (O'Reilly, 2017) — ストリーム処理とイベントソーシングの理論的基盤
3. **Implementing Domain-Driven Design** — Vaughn Vernon (Addison-Wesley, 2013) — ドメインイベントと CQRS の実装パターン
4. **Enterprise Integration Patterns** — Gregor Hohpe & Bobby Woolf (Addison-Wesley, 2003) — メッセージングパターンの古典
5. **Reactive Messaging Patterns with the Actor Model** — Vaughn Vernon (Addison-Wesley, 2015) — リアクティブシステムとメッセージングの統合
6. **Kafka: The Definitive Guide** — Neha Narkhede et al. (O'Reilly, 2021) — Apache Kafka の包括的リファレンス
7. **Martin Fowler — Event Sourcing** — https://martinfowler.com/eaaDev/EventSourcing.html — Event Sourcing パターンの解説
8. **Greg Young — CQRS Documents** — https://cqrs.files.wordpress.com/2010/11/cqrs_documents.pdf — CQRS の原典

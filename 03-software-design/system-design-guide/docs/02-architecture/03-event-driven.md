# イベント駆動アーキテクチャ

> コンポーネント間の通信をイベント（事実の記録）を中心に設計するアーキテクチャパターンであり、Pub/Sub モデル・イベントソーシング・CQRS を活用して疎結合でスケーラブルなシステムを構築する手法を解説する

## この章で学ぶこと

1. **イベント駆動の基本モデル** — イベント通知・イベントキャリー・イベントソーシングの3パターンとその使い分け
2. **Pub/Sub アーキテクチャ** — トピックベースのメッセージングによる疎結合な連携設計
3. **CQRS とイベントソーシング** — コマンドとクエリの分離、イベントログからの状態再構築

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

### 1.2 Pub/Sub の全体構成

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
```

### 1.3 同期 vs 非同期の比較

```
【同期 (REST/gRPC)】
  Order --> Inventory --> Payment --> Notification
   |            |            |            |
   |<-----------+<-----------+<-----------+
   全体のレイテンシ = 各サービスの合計
   1つ障害 = 全体障害 (カスケード障害)

【非同期 (Event-Driven)】
  Order --event--> [Bus] ---> Inventory (独立処理)
                         ---> Payment   (独立処理)
                         ---> Notification (独立処理)
   Order は即座に応答可能
   1つ障害 = そのサービスのみ影響（リトライで回復）
```

---

## 2. Pub/Sub の実装

### 2.1 イベントの定義

```python
# domain/events/base.py
from dataclasses import dataclass, field
from datetime import datetime
import uuid

@dataclass(frozen=True)
class DomainEvent:
    """ドメインイベントの基底クラス"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    event_type: str = ""

@dataclass(frozen=True)
class OrderPlaced(DomainEvent):
    event_type: str = "order.placed"
    order_id: str = ""
    customer_id: str = ""
    items: tuple = ()      # frozen=True のため tuple
    total_amount: int = 0

@dataclass(frozen=True)
class OrderCancelled(DomainEvent):
    event_type: str = "order.cancelled"
    order_id: str = ""
    reason: str = ""
```

### 2.2 イベントパブリッシャー (Kafka)

```python
# infrastructure/messaging/kafka_publisher.py
from confluent_kafka import Producer
import json
from datetime import datetime

class KafkaEventPublisher:
    def __init__(self, bootstrap_servers: str):
        self._producer = Producer({
            'bootstrap.servers': bootstrap_servers,
            'acks': 'all',
            'enable.idempotence': True,
        })

    def publish(self, event: DomainEvent) -> None:
        topic = f"domain-events.{event.event_type.split('.')[0]}"
        payload = {
            'event_id': event.event_id,
            'event_type': event.event_type,
            'occurred_at': event.occurred_at.isoformat(),
            'data': self._extract_data(event),
        }
        self._producer.produce(
            topic=topic,
            key=getattr(event, 'order_id', event.event_id).encode(),
            value=json.dumps(payload, ensure_ascii=False).encode(),
            callback=self._delivery_report,
        )
        self._producer.flush()

    def _extract_data(self, event):
        d = event.__dict__.copy()
        d.pop('event_id', None)
        d.pop('occurred_at', None)
        d.pop('event_type', None)
        return d

    def _delivery_report(self, err, msg):
        if err:
            raise RuntimeError(f"イベント配信失敗: {err}")
```

### 2.3 イベントハンドラー

```python
# application/handlers/inventory_handler.py
class InventoryEventHandler:
    """在庫サービスのイベントハンドラー"""

    def __init__(self, inventory_repo, stock_service):
        self._repo = inventory_repo
        self._stock = stock_service

    def handle_order_placed(self, event_data: dict) -> None:
        """注文確定イベントの処理: 在庫引当"""
        order_id = event_data['order_id']
        items = event_data['items']

        for item in items:
            success = self._stock.reserve(
                product_id=item['product_id'],
                quantity=item['quantity'],
                reservation_id=order_id,
            )
            if not success:
                # 在庫不足 → 補償イベント発行
                self._publish_compensation(order_id, item['product_id'])
                return

    def handle_order_cancelled(self, event_data: dict) -> None:
        """注文キャンセルイベントの処理: 在庫解放"""
        order_id = event_data['order_id']
        self._stock.release_reservation(reservation_id=order_id)
```

### 2.4 Saga パターン（分散トランザクション）

```python
# application/sagas/order_saga.py
class OrderPlacementSaga:
    """注文確定のSaga: 複数サービスにまたがる処理を調整"""

    STEPS = [
        ('reserve_inventory', 'release_inventory'),    # (実行, 補償)
        ('process_payment',   'refund_payment'),
        ('schedule_shipping', 'cancel_shipping'),
    ]

    def __init__(self, event_publisher):
        self._publisher = event_publisher
        self._completed_steps = []

    def execute(self, order_id: str, order_data: dict):
        for step_name, compensation_name in self.STEPS:
            try:
                step_func = getattr(self, step_name)
                step_func(order_id, order_data)
                self._completed_steps.append(compensation_name)
            except Exception as e:
                print(f"Saga 失敗 at {step_name}: {e}")
                self._compensate(order_id)
                raise SagaFailedError(step_name, str(e))

    def _compensate(self, order_id: str):
        """完了済みステップの補償処理を逆順で実行"""
        for compensation_name in reversed(self._completed_steps):
            try:
                comp_func = getattr(self, compensation_name)
                comp_func(order_id)
            except Exception as e:
                print(f"補償処理失敗 {compensation_name}: {e}")
                # 補償失敗はアラート + 手動対応
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
```

### 3.2 イベントソーシングの実装

```python
# infrastructure/event_store.py
class EventStore:
    """イベントストア: 全イベントを時系列で保存"""

    def __init__(self, db_session):
        self._session = db_session

    def append(self, aggregate_id: str, events: list, expected_version: int):
        """イベントを追記（楽観的ロック付き）"""
        current_version = self._get_current_version(aggregate_id)
        if current_version != expected_version:
            raise ConcurrencyError(
                f"Expected version {expected_version}, got {current_version}")

        for i, event in enumerate(events):
            self._session.add(EventRecord(
                aggregate_id=aggregate_id,
                version=expected_version + i + 1,
                event_type=event.event_type,
                data=json.dumps(event.__dict__, default=str),
                occurred_at=event.occurred_at,
            ))
        self._session.commit()

    def load(self, aggregate_id: str) -> list:
        """集約の全イベントを取得"""
        records = (self._session.query(EventRecord)
                   .filter_by(aggregate_id=aggregate_id)
                   .order_by(EventRecord.version)
                   .all())
        return [self._deserialize(r) for r in records]

    def _get_current_version(self, aggregate_id: str) -> int:
        result = (self._session.query(func.max(EventRecord.version))
                  .filter_by(aggregate_id=aggregate_id)
                  .scalar())
        return result or 0
```

---

## 4. 比較表

| 特性 | 同期 (REST/gRPC) | 非同期 (Event-Driven) |
|-----|:----------------:|:--------------------:|
| 結合度 | 高（直接呼び出し） | 低（イベントバス経由） |
| レイテンシ | 全サービスの合計 | 即座に応答可能 |
| 耐障害性 | カスケード障害リスク | サービスごとに独立 |
| データ整合性 | 強い整合性（可能） | 結果整合性 |
| デバッグ | 容易（同期フロー） | 困難（非同期フロー追跡） |
| スケーラビリティ | ボトルネック発生 | 独立スケール |

| アプローチ | 状態管理 | 監査ログ | 複雑性 | 適用場面 |
|-----------|---------|---------|--------|---------|
| CRUD | 最新状態のみ保持 | 別途実装が必要 | 低 | シンプルなアプリ |
| CQRS | 読み書き分離 | 別途実装が必要 | 中 | 読み書きの負荷特性が異なる |
| Event Sourcing | イベントログから再構築 | 自然に実現 | 高 | 完全な監査証跡が必要 |
| CQRS + ES | イベントログ + Read Model | 自然に実現 | 最高 | 金融・医療・法規制 |

---

## 5. アンチパターン

### アンチパターン 1: イベントを RPC の代替として使う

```
BAD: イベントで同期的なリクエスト/レスポンスを模倣
  Order Service --> "PleaseReserveInventory" --> Inventory Service
  Order Service <-- "InventoryReserved" <-- Inventory Service
  → イベント駆動の意味がない。実質的に同期呼び出し

GOOD: イベントは「起きた事実」を伝える
  Order Service --> "OrderPlaced" --> [Event Bus]
  → Inventory Service が独自に判断して在庫引当
  → Order Service は結果を待たない
```

### アンチパターン 2: 全てをイベント駆動にする

```
BAD: ユーザー認証やデータ取得もイベント駆動
  → ログインに数秒のレイテンシ
  → 単純な GET リクエストにイベントバス経由

GOOD: 適材適所
  - 同期 (REST/gRPC): ユーザー認証、データ参照、リアルタイム応答が必要な処理
  - 非同期 (Event): 注文処理、通知送信、データ同期、バッチ処理
```

---

## 6. FAQ

### Q1. イベントの順序保証はどう実現する？

**A.** Kafka ではパーティションキーに集約ID を使うことで、同一集約のイベントは同一パーティション内で順序保証される。異なる集約間の順序は保証不要（各集約は独立しているため）。順序が必要な場合はイベントに `version` フィールドを含め、Consumer 側で順序検証を行う。

### Q2. 結果整合性で「不整合な期間」はどう扱う？

**A.** UIレベルで対処する。例えば注文確定後に「処理中」と表示し、バックグラウンドで在庫引当・決済が完了するまで待つ。ポーリングや WebSocket で最新状態を通知する。ビジネス上、数秒〜数分の遅延が許容されるかをドメインエキスパートと事前に合意しておくことが重要。

### Q3. イベントスキーマの変更（バージョニング）はどう管理する？

**A.** (1) 後方互換を維持する（フィールド追加はOK、削除・変更はNG）。(2) event_type にバージョンを含める（`order.placed.v2`）。(3) Consumer 側で複数バージョンに対応するアダプターを実装する。(4) Schema Registry（Confluent Schema Registry 等）でスキーマを一元管理し、互換性チェックを自動化する。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| イベント駆動の利点 | 疎結合、独立スケール、耐障害性 |
| 3つのパターン | Event Notification / Event-Carried State Transfer / Event Sourcing |
| Pub/Sub | プロデューサーとコンシューマーの完全な分離 |
| CQRS | 読み書きの最適化を独立して行う |
| Saga パターン | 分散トランザクションの代替（補償による結果整合性） |
| トレードオフ | 結果整合性、デバッグの困難さ、運用の複雑性 |

---

## 次に読むべきガイド

- [メッセージキュー](../01-components/02-message-queue.md) — イベントバスの実装基盤
- [DDD](./02-ddd.md) — ドメインイベントの設計元となるドメインモデリング
- [クリーンアーキテクチャ](./01-clean-architecture.md) — イベントハンドラーの配置

---

## 参考文献

1. **Building Event-Driven Microservices** — Adam Bellemare (O'Reilly, 2020) — イベント駆動マイクロサービスの包括的ガイド
2. **Designing Data-Intensive Applications** — Martin Kleppmann (O'Reilly, 2017) — ストリーム処理とイベントソーシングの理論
3. **Implementing Domain-Driven Design** — Vaughn Vernon (Addison-Wesley, 2013) — ドメインイベントと CQRS の実装パターン
4. **Enterprise Integration Patterns** — Gregor Hohpe & Bobby Woolf (Addison-Wesley, 2003) — メッセージングパターンの古典

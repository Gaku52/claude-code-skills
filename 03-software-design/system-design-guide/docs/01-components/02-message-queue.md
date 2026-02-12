# メッセージキュー

> 分散システムにおける非同期メッセージングの基盤技術であり、コンポーネント間の疎結合・スケーラビリティ・耐障害性を実現する中核パターンを、Kafka・RabbitMQ・SQS の実装比較を通じて解説する

## この章で学ぶこと

1. **メッセージキューの基本概念** -- Producer/Consumer モデル、キューとトピックの違い、配信保証（At-most-once / At-least-once / Exactly-once）
2. **主要プロダクトの比較と選定** -- Apache Kafka、RabbitMQ、Amazon SQS のアーキテクチャ・性能特性・適材適所
3. **実践的な設計パターン** -- Dead Letter Queue、バックプレッシャー、べき等処理、順序保証の実装手法
4. **運用とモニタリング** -- Consumer Lag 監視、キュー深度アラート、パフォーマンスチューニング
5. **障害対応パターン** -- メッセージ再処理、ポイズンメッセージ対策、グレースフルシャットダウン

---

## 前提知識

| トピック | 内容 | 参照ガイド |
|---------|------|-----------|
| 分散システム基礎 | CAP定理、結果整合性 | [CAP定理](../00-fundamentals/03-cap-theorem.md) |
| 信頼性パターン | リトライ、サーキットブレーカー | [信頼性](../00-fundamentals/02-reliability.md) |
| スケーラビリティ | 水平スケーリングの概念 | [スケーラビリティ](../00-fundamentals/01-scalability.md) |
| データベース基礎 | トランザクション、ACID特性 | DB基礎 |
| ネットワーク基礎 | TCP/IP、HTTP、非同期通信 | ネットワーク基礎 |

---

## なぜメッセージキューを学ぶのか

メッセージキューは**マイクロサービスアーキテクチャの接着剤**であり、現代の分散システムにおいて不可欠なインフラコンポーネントである。

**同期 vs 非同期の本質的な違い:**
```
同期呼び出し（HTTP直接通信）:
  OrderService --HTTP POST--> PaymentService --wait--> response
  問題: PaymentService がダウン → OrderService もエラー（カスケード障害）
  問題: PaymentService が遅い → OrderService も遅い（レイテンシ結合）

非同期メッセージング:
  OrderService --publish--> [Message Queue] ... PaymentService --consume-->
  利点: PaymentService がダウンしても OrderService は正常動作
  利点: キューがバッファとなり負荷を平準化
```

**ビジネスインパクト:**
- **耐障害性**: 下流サービスの障害がシステム全体に波及しない
- **スケーラビリティ**: Consumer を追加するだけで処理能力を線形にスケール
- **負荷平準化**: トラフィックスパイクをキューで吸収し、一定速度で処理
- **疎結合**: サービス間の依存関係を最小化し、独立した開発・デプロイを実現

**具体例:**
- LinkedIn: Kafka で毎秒数百万イベントを処理（アクティビティストリーム）
- Uber: Kafka でリアルタイムの位置情報・需要予測データをストリーミング
- Slack: RabbitMQ でメッセージ配信の信頼性を確保

---

## 1. メッセージキューの基本アーキテクチャ

### 1.1 全体構成図

```
+------------+     +------------------+     +-------------+
|  Producer  |---->|  Message Broker  |---->|  Consumer   |
|  (送信側)  |     |  (キュー/トピック) |     |  (受信側)   |
+------------+     +------------------+     +-------------+
      |                    |                       |
      |   Publish          |   Store & Forward     |   Subscribe/Poll
      +--------------------+-----------------------+

  同期呼び出し:  A --req--> B --res--> A  (A は B の応答を待つ)
  非同期キュー:  A --msg--> [Queue] ... B --poll--> [Queue]
                (A は B の処理完了を待たない)
```

### 1.2 Point-to-Point vs Pub/Sub

```
【Point-to-Point (Queue)】

  Producer A --+
               +--> [ Queue ] --> Consumer X
  Producer B --+     (1メッセージ = 1消費者のみ)

【Pub/Sub (Topic)】

  Producer ---> [ Topic ] --+--> Consumer Group A (Consumer A1, A2)
                            |
                            +--> Consumer Group B (Consumer B1, B2)
              (1メッセージ = 全購読グループに配信)
```

### 1.3 メッセージのライフサイクル

```
  Producer                Broker                Consumer
     |                      |                      |
     |--- 1. Publish ------>|                      |
     |<-- 1a. Ack(受領) ----|                      |
     |                      |-- 2. Persist(永続化) |
     |                      |                      |
     |                      |<-- 3. Poll/Push -----|
     |                      |--- 4. Deliver ------>|
     |                      |                      |-- 5. Process
     |                      |<-- 6. Ack(処理完了) -|
     |                      |-- 7. Mark Done ----->|
     |                      |                      |
```

### ASCII図解: メッセージキューのユースケースマップ

```
┌─────────────────────────────────────────────────┐
│          メッセージキューのユースケース             │
├─────────────────────────────────────────────────┤
│                                                 │
│  ■ タスクキュー (Work Queue)                     │
│    画像リサイズ、PDF生成、メール送信             │
│    → 重い処理をバックグラウンドに委譲            │
│                                                 │
│  ■ イベント駆動 (Event-Driven)                   │
│    注文作成→在庫更新→通知送信→分析記録         │
│    → サービス間の疎結合な連携                    │
│                                                 │
│  ■ ストリーミング (Stream Processing)            │
│    ログ集約、クリックストリーム、IoTデータ       │
│    → 大量データのリアルタイム処理                │
│                                                 │
│  ■ CQRS / イベントソーシング                     │
│    書き込みと読み取りの分離                      │
│    → スケーラブルなデータアーキテクチャ           │
│                                                 │
│  ■ 負荷平準化 (Load Leveling)                    │
│    セールのスパイクトラフィック吸収              │
│    → Consumer の処理能力に合わせた平準化         │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 2. 配信保証モデル

| 保証レベル | 説明 | メッセージ損失 | 重複配信 | 代表的ユースケース |
|-----------|------|:------------:|:-------:|-----------------|
| At-most-once | 最大1回配信。再送なし | あり得る | なし | ログ収集、メトリクス送信 |
| At-least-once | 最低1回配信。Ack失敗で再送 | なし | あり得る | 注文処理、メール送信 |
| Exactly-once | 正確に1回配信 | なし | なし | 決済処理、在庫更新 |

### 配信保証の実装コスト

```
  At-most-once         At-least-once        Exactly-once
  ┌──────────┐         ┌──────────┐         ┌──────────┐
  │ fire &   │         │ Ack +    │         │ トランザクション│
  │ forget   │         │ retry    │         │ + べき等  │
  └──────────┘         └──────────┘         └──────────┘
  コスト: 低            コスト: 中            コスト: 高
  レイテンシ: 最低      レイテンシ: 低        レイテンシ: 高

  Exactly-once の実現方法:
  1. Kafka Transactions (acks=all + enable.idempotence + transactional.id)
  2. Outbox パターン (DB トランザクション + CDC)
  3. Consumer 側べき等処理 (At-least-once + 重複排除)
```

---

## 3. Apache Kafka

### 3.1 アーキテクチャ

```
Kafka Cluster
+-------------------------------------------------------------+
|  Broker 0              Broker 1              Broker 2        |
|  +--------------+      +--------------+      +-----------+   |
|  | orders-topic |      | orders-topic |      |orders-topic|  |
|  | Partition 0  |      | Partition 1  |      | Partition 2|  |
|  | [Leader]     |      | [Leader]     |      | [Leader]   |  |
|  +--------------+      +--------------+      +-----------+   |
|  | orders-topic |      | orders-topic |      |orders-topic|  |
|  | Partition 1  |      | Partition 2  |      | Partition 0|  |
|  | [Follower]   |      | [Follower]   |      | [Follower] |  |
|  +--------------+      +--------------+      +-----------+   |
+-------------------------------------------------------------+
        ^                                           |
        |              Consumer Group               v
   Producers       +----------------------------+
                   | Consumer 0 <-- Partition 0 |
                   | Consumer 1 <-- Partition 1 |
                   | Consumer 2 <-- Partition 2 |
                   +----------------------------+
```

### 3.2 Producer の実装

```python
# Kafka Producer (Python - confluent-kafka)
from confluent_kafka import Producer
import json
import time

conf = {
    'bootstrap.servers': 'broker1:9092,broker2:9092,broker3:9092',
    'acks': 'all',                    # 全ISRレプリカの書き込み確認
    'retries': 5,
    'retry.backoff.ms': 100,
    'enable.idempotence': True,       # べき等プロデューサー（重複防止）
    'compression.type': 'snappy',     # 圧縮でスループット向上
    'linger.ms': 5,                   # バッチ化のための待機時間
    'batch.size': 32768,              # バッチサイズ (32KB)
    'max.in.flight.requests.per.connection': 5,  # べき等有効時は最大5
}

producer = Producer(conf)

def delivery_report(err, msg):
    if err:
        print(f"配信失敗: {err}")
    else:
        print(f"配信成功: topic={msg.topic()} "
              f"partition={msg.partition()} offset={msg.offset()}")

# メッセージ送信
for i in range(1000):
    order = {"order_id": i, "user_id": i % 100, "amount": 1000 + i,
             "timestamp": time.time()}
    producer.produce(
        topic='order-events',
        key=str(order['user_id']).encode('utf-8'),   # 同一ユーザーは同一パーティション
        value=json.dumps(order).encode('utf-8'),
        callback=delivery_report,
    )
    producer.poll(0)  # コールバック処理

producer.flush()  # 全メッセージの送信完了を待機
```

### 3.3 Consumer の実装

```python
# Kafka Consumer (Python - confluent-kafka)
from confluent_kafka import Consumer, KafkaError
import json

conf = {
    'bootstrap.servers': 'broker1:9092',
    'group.id': 'order-processing-group',
    'auto.offset.reset': 'earliest',       # 初回は最古から読む
    'enable.auto.commit': False,           # 手動コミットで確実な処理
    'max.poll.interval.ms': 300000,        # 処理タイムアウト 5分
    'session.timeout.ms': 45000,
    'fetch.min.bytes': 1024,               # 最低1KBでフェッチ（バッチ効率化）
    'fetch.max.wait.ms': 500,              # 最大500ms待機
}

consumer = Consumer(conf)
consumer.subscribe(['order-events'])

try:
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            raise Exception(f"Consumer error: {msg.error()}")

        order = json.loads(msg.value().decode('utf-8'))
        print(f"受信: partition={msg.partition()} "
              f"offset={msg.offset()} order_id={order['order_id']}")

        # ビジネスロジック実行
        process_order(order)

        # 処理完了後に手動コミット
        consumer.commit(asynchronous=False)
finally:
    consumer.close()
```

### 3.4 Kafka Streams による Stream Processing

```python
# Kafka を使ったリアルタイム集計の概念的実装
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

@dataclass
class WindowedCounter:
    """タンブリングウィンドウによるリアルタイム集計"""
    window_size_sec: int = 60
    windows: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    def add(self, key: str, value: int = 1):
        window_start = int(time.time() / self.window_size_sec) * self.window_size_sec
        self.windows[window_start][key] += value

    def get_current_window(self) -> dict:
        window_start = int(time.time() / self.window_size_sec) * self.window_size_sec
        return dict(self.windows[window_start])

    def get_window(self, window_start: int) -> dict:
        return dict(self.windows.get(window_start, {}))

    def cleanup_old_windows(self, retain_count: int = 10):
        """古いウィンドウを削除してメモリ節約"""
        sorted_windows = sorted(self.windows.keys())
        for w in sorted_windows[:-retain_count]:
            del self.windows[w]


class StreamProcessor:
    """Kafka Consumer ベースのストリーム処理フレームワーク

    機能:
    1. リアルタイムイベント集計（ウィンドウ関数）
    2. イベントフィルタリング / 変換
    3. 出力トピックへの書き込み
    """

    def __init__(self, consumer, producer,
                 input_topic: str, output_topic: str):
        self.consumer = consumer
        self.producer = producer
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.counter = WindowedCounter(window_size_sec=60)
        self.filters: list[Callable] = []
        self.transformers: list[Callable] = []

    def add_filter(self, predicate: Callable):
        """イベントフィルタを追加"""
        self.filters.append(predicate)

    def add_transformer(self, transform: Callable):
        """イベント変換を追加"""
        self.transformers.append(transform)

    def process(self):
        """メインの処理ループ"""
        self.consumer.subscribe([self.input_topic])

        while True:
            msg = self.consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                continue

            event = json.loads(msg.value().decode('utf-8'))

            # フィルタリング
            if not all(f(event) for f in self.filters):
                self.consumer.commit(asynchronous=False)
                continue

            # 変換
            for transform in self.transformers:
                event = transform(event)

            # 集計
            self.counter.add(event.get("category", "unknown"))

            # 出力
            self.producer.produce(
                topic=self.output_topic,
                key=event.get("key", "").encode('utf-8'),
                value=json.dumps(event).encode('utf-8'),
            )
            self.consumer.commit(asynchronous=False)


# 使用例: 注文イベントから高額注文のみをフィルタリングして集計
processor = StreamProcessor(consumer, producer,
                           "order-events", "high-value-orders")
processor.add_filter(lambda e: e.get("amount", 0) > 10000)
processor.add_transformer(lambda e: {**e, "flagged": True})
# processor.process()
```

---

## 4. RabbitMQ

### 4.1 Exchange/Queue モデル

```
                   Exchange (routing)
                  +------------------+
  Publisher -->   | Type: topic      |
                  |                  |
                  | order.created -->+--> [order_processing_queue] --> Consumer A
                  |                  |
                  | order.* -------->+--> [order_audit_queue]      --> Consumer B
                  |                  |
                  | payment.* ------>+--> [payment_queue]          --> Consumer C
                  +------------------+

  Exchange Types:
  ┌──────────┬─────────────────────────────────────────┐
  │ direct   │ routing_key が完全一致するキューに配信   │
  │ topic    │ ワイルドカードパターンマッチ (*.*, #)   │
  │ fanout   │ バインドされた全キューに配信（ブロードキャスト）│
  │ headers  │ メッセージヘッダーに基づくルーティング   │
  └──────────┴─────────────────────────────────────────┘
```

### 4.2 Producer / Consumer

```python
# RabbitMQ Producer (Python - pika)
import pika
import json
import uuid

connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host='localhost',
        credentials=pika.PlainCredentials('app_user', 'secret'),
        heartbeat=600,
        blocked_connection_timeout=300,
    )
)
channel = connection.channel()

# Exchange と Queue を宣言
channel.exchange_declare(exchange='order_exchange',
                        exchange_type='topic', durable=True)
channel.queue_declare(queue='order_processing', durable=True, arguments={
    'x-dead-letter-exchange': 'dlx_exchange',        # DLQ 設定
    'x-dead-letter-routing-key': 'order.failed',
    'x-message-ttl': 86400000,                        # TTL: 24時間
    'x-max-length': 100000,                           # キュー最大長
    'x-overflow': 'reject-publish',                   # 溢れ時に拒否
})
channel.queue_bind(exchange='order_exchange', queue='order_processing',
                   routing_key='order.created')

# メッセージ送信
message = json.dumps({"order_id": 123, "amount": 5000, "currency": "JPY"})
channel.basic_publish(
    exchange='order_exchange',
    routing_key='order.created',
    body=message,
    properties=pika.BasicProperties(
        delivery_mode=2,                  # メッセージ永続化
        content_type='application/json',
        message_id=str(uuid.uuid4()),
        timestamp=int(time.time()),
        headers={'retry_count': 0},
    ),
)
print(f"送信完了: {message}")
connection.close()
```

```python
# RabbitMQ Consumer (Python - pika) with retry logic
import pika
import json
import traceback

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='order_processing', durable=True)
channel.basic_qos(prefetch_count=10)    # 同時処理数を制御

MAX_RETRIES = 3

def callback(ch, method, properties, body):
    order = json.loads(body)
    retry_count = (properties.headers or {}).get('retry_count', 0)
    print(f"処理開始: order_id={order['order_id']} (retry={retry_count})")

    try:
        process_order(order)
        ch.basic_ack(delivery_tag=method.delivery_tag)        # 成功 → Ack
        print(f"処理完了: order_id={order['order_id']}")
    except Exception as e:
        print(f"処理失敗: {e}")
        traceback.print_exc()

        if retry_count < MAX_RETRIES:
            # リトライ: 同じキューに再投入（retry_count をインクリメント）
            ch.basic_publish(
                exchange='',
                routing_key='order_processing',
                body=body,
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    headers={'retry_count': retry_count + 1},
                ),
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)
            print(f"リトライ予約: order_id={order['order_id']} "
                  f"retry={retry_count + 1}")
        else:
            # リトライ上限超過 → DLQ へ
            ch.basic_nack(delivery_tag=method.delivery_tag,
                          requeue=False)
            print(f"DLQ送信: order_id={order['order_id']} (max retries exceeded)")

channel.basic_consume(queue='order_processing', on_message_callback=callback)
print("Consumer 起動完了。メッセージ待機中...")
channel.start_consuming()
```

---

## 5. Amazon SQS

```python
# Amazon SQS (Python - boto3)
import boto3
import json
import uuid
import time

sqs = boto3.client('sqs', region_name='ap-northeast-1')
QUEUE_URL = 'https://sqs.ap-northeast-1.amazonaws.com/123456789/order-queue.fifo'

# --- Producer ---
response = sqs.send_message(
    QueueUrl=QUEUE_URL,
    MessageBody=json.dumps({"order_id": 456, "amount": 8000}),
    MessageAttributes={
        'EventType': {'DataType': 'String', 'StringValue': 'OrderCreated'},
        'Priority': {'DataType': 'Number', 'StringValue': '1'},
    },
    MessageDeduplicationId=str(uuid.uuid4()),  # FIFO: 5分間の重複排除
    MessageGroupId='user-group-42',            # FIFO: 同グループ内で順序保証
)
print(f"送信完了 MessageId: {response['MessageId']}")

# --- Consumer (ロングポーリング) ---
while True:
    response = sqs.receive_message(
        QueueUrl=QUEUE_URL,
        MaxNumberOfMessages=10,         # 最大10件一括取得
        WaitTimeSeconds=20,             # ロングポーリング（20秒待機）
        VisibilityTimeout=120,          # 処理中の非表示期間
        MessageAttributeNames=['All'],
    )
    for message in response.get('Messages', []):
        body = json.loads(message['Body'])
        print(f"処理中: order_id={body['order_id']}")

        try:
            process_order(body)
            sqs.delete_message(
                QueueUrl=QUEUE_URL,
                ReceiptHandle=message['ReceiptHandle'],
            )
            print(f"処理完了・削除: order_id={body['order_id']}")
        except Exception as e:
            # VisibilityTimeout 後に自動的にキューに戻る
            print(f"処理失敗: {e} → VisibilityTimeout後に再配信")

    if not response.get('Messages'):
        time.sleep(1)  # メッセージなし → 短い待機
```

---

## 6. 主要プロダクト比較表

### 比較表1: 機能比較

| 特性 | Apache Kafka | RabbitMQ | Amazon SQS |
|-----|-------------|----------|------------|
| **モデル** | 分散コミットログ (Pull) | メッセージブローカー (Push/Pull) | マネージドキュー (Pull) |
| **スループット** | 数百万 msg/sec | 数万 msg/sec | 数千 msg/sec (標準) |
| **メッセージ保持** | 設定期間保持（再読可能） | 消費後削除 | 最大14日保持 |
| **順序保証** | パーティション内で保証 | キュー内で保証 | FIFO キューで保証 |
| **配信保証** | At-least-once / Exactly-once | At-least-once | At-least-once / FIFO で Exactly-once |
| **遅延メッセージ** | 非対応（外部実装が必要） | 対応（TTL + DLX） | 対応（最大15分） |
| **運用コスト** | 高（KRaft/ZooKeeper管理） | 中（Erlang VM管理） | 低（フルマネージド） |
| **最適用途** | ストリーミング、ログ集約、CQRS | タスクキュー、RPC、複雑なルーティング | サーバーレス連携、シンプルなキュー |

### 比較表2: 選定フローチャート

| 判断基準 | Kafka を選ぶ | RabbitMQ を選ぶ | SQS を選ぶ |
|---------|------------|----------------|-----------|
| メッセージの再読が必要 | YES | -- | -- |
| 秒間100万メッセージ超 | YES | -- | -- |
| 複雑なルーティングルール | -- | YES | -- |
| リクエスト/レスポンス型 RPC | -- | YES | -- |
| AWS ネイティブで運用最小化 | -- | -- | YES |
| Lambda との統合 | -- | -- | YES |
| イベントソーシング | YES | -- | -- |
| 優先度付きキュー | -- | YES | -- |
| ストリーム処理（集計、結合） | YES | -- | -- |
| 即座に始めたい（学習コスト低） | -- | -- | YES |

### 比較表3: 非機能要件の比較

| 項目 | Kafka | RabbitMQ | SQS |
|------|-------|----------|-----|
| レイテンシ (P99) | 5-15ms | 1-5ms | 20-50ms |
| 最大メッセージサイズ | 1MB (デフォルト) | 128MB | 256KB (S3で拡張可) |
| Consumer並列度上限 | パーティション数 | 無制限 | 無制限 |
| クラスタ最小構成 | 3ノード | 3ノード (Quorum) | N/A (マネージド) |
| 暗号化 | TLS + SASL | TLS + AMQP認証 | KMS + IAM |
| 監視 | JMX, Prometheus | Prometheus, Management UI | CloudWatch |

---

## 7. 設計パターン

### 7.1 Dead Letter Queue (DLQ)

```
                          処理成功
  [Main Queue] --------> Consumer ------> Done (Ack)
       |                     |
       |               処理失敗 (N回リトライ後)
       |                     |
       v                     v
  [Retry Queue]         [Dead Letter Queue]
  (遅延再配信)                |
       |                     +--> 監視アラート通知
       +---> [Main Queue]   +--> 管理画面で確認
                             +--> 手動再処理 or 補正
```

### 7.2 べき等処理パターン

```python
# べき等 Consumer の実装例
import redis
import hashlib
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

class IdempotentConsumer:
    """同じメッセージを何度受信しても結果が変わらないことを保証する

    実装方式:
    1. メッセージIDベース: message_id で重複チェック
    2. ビジネスキーベース: entity_id + version で重複チェック
    3. ハッシュベース: メッセージ内容のハッシュで重複チェック
    """

    def __init__(self, redis_client: redis.Redis,
                 dedup_ttl: int = 86400 * 7):
        self.redis = redis_client
        self.dedup_ttl = dedup_ttl  # 重複チェックの保持期間

    def process(self, message: dict) -> bool:
        """べき等にメッセージを処理する

        Returns:
            True: 処理成功（新規メッセージ）
            False: スキップ（処理済みメッセージ）
        """
        message_id = message.get('message_id')
        if not message_id:
            # message_id がない場合はコンテンツハッシュを使用
            content = json.dumps(message, sort_keys=True)
            message_id = hashlib.sha256(content.encode()).hexdigest()

        idempotency_key = f"processed:{message_id}"

        # SETNX (Set if Not eXists) で排他制御
        if not self.redis.set(idempotency_key, 'processing',
                              nx=True, ex=self.dedup_ttl):
            print(f"[SKIP] 処理済み or 処理中: {message_id}")
            return False

        try:
            # ビジネスロジック実行
            result = self._execute_business_logic(message)

            # 処理完了マーク
            self.redis.set(idempotency_key, json.dumps({
                'status': 'completed',
                'result': result,
                'processed_at': time.time(),
            }), ex=self.dedup_ttl)
            return True

        except Exception as e:
            # 失敗時はキーを削除してリトライ可能にする
            self.redis.delete(idempotency_key)
            raise

    def _execute_business_logic(self, message: dict):
        """ビジネスロジック（サブクラスでオーバーライド）"""
        raise NotImplementedError


class OrderConsumer(IdempotentConsumer):
    """注文処理の具体的な Consumer"""

    def _execute_business_logic(self, message: dict):
        order_id = message['order_id']
        amount = message['amount']
        print(f"[PROCESS] 注文処理: order_id={order_id}, amount={amount}")
        # DB更新、外部API呼び出し等
        return {"order_id": order_id, "status": "confirmed"}


# 使用例
consumer = OrderConsumer(redis_client)
messages = [
    {"message_id": "msg-001", "order_id": 123, "amount": 5000},
    {"message_id": "msg-001", "order_id": 123, "amount": 5000},  # 重複
    {"message_id": "msg-002", "order_id": 456, "amount": 8000},
]

for msg in messages:
    result = consumer.process(msg)
    print(f"  → processed={result}")
# [PROCESS] 注文処理: order_id=123, amount=5000
#   → processed=True
# [SKIP] 処理済み or 処理中: msg-001
#   → processed=False
# [PROCESS] 注文処理: order_id=456, amount=8000
#   → processed=True
```

### 7.3 Outbox パターン

```python
# Outbox パターン: DB トランザクションとメッセージ発行の原子性を保証

class OutboxPattern:
    """Outbox パターンの実装

    問題: DB更新とメッセージ発行の2つの操作を原子的に行えない
    → DB更新成功 + メッセージ発行失敗 = 不整合

    解決: DB トランザクション内で outbox テーブルにも書き込み、
    別プロセスが outbox を読んでメッセージブローカーに発行する

    DB Transaction:
      1. orders テーブルに INSERT
      2. outbox テーブルに INSERT (同一トランザクション)

    Outbox Relay (別プロセス):
      1. outbox テーブルをポーリング
      2. 未送信メッセージをブローカーに発行
      3. 送信済みマークを付与
    """

    def __init__(self, db_session, producer):
        self.db = db_session
        self.producer = producer

    def create_order(self, order_data: dict):
        """注文作成 + イベント発行（原子的）"""
        with self.db.begin():
            # 1. 注文を保存
            order = Order(**order_data)
            self.db.add(order)
            self.db.flush()  # IDを取得

            # 2. Outboxに書き込み（同一トランザクション）
            outbox_event = OutboxEvent(
                aggregate_type='Order',
                aggregate_id=str(order.id),
                event_type='OrderCreated',
                payload=json.dumps({
                    'order_id': order.id,
                    'user_id': order_data['user_id'],
                    'amount': order_data['amount'],
                    'created_at': time.time(),
                }),
                status='PENDING',
            )
            self.db.add(outbox_event)
            # トランザクション完了 → 両方が原子的にコミット

    def relay_outbox_events(self, batch_size: int = 100):
        """Outbox テーブルの未送信イベントをブローカーに発行"""
        pending_events = self.db.query(OutboxEvent)\
            .filter_by(status='PENDING')\
            .order_by(OutboxEvent.created_at)\
            .limit(batch_size)\
            .all()

        for event in pending_events:
            try:
                self.producer.produce(
                    topic=f"{event.aggregate_type.lower()}-events",
                    key=event.aggregate_id.encode('utf-8'),
                    value=event.payload.encode('utf-8'),
                )
                event.status = 'SENT'
                event.sent_at = time.time()
            except Exception as e:
                print(f"[OUTBOX ERROR] {event.id}: {e}")

        self.db.commit()
        self.producer.flush()
```

---

## 8. アンチパターン

### アンチパターン 1: キューを巨大データストアとして使う

```python
# NG: 大きなペイロードをキューに直接格納

class BadProducer:
    def send_invoice(self, order_id: int, pdf_data: bytes, images: list[bytes]):
        message = {
            "order_id": order_id,
            "pdf_invoice": base64.b64encode(pdf_data).decode(),  # 10MB
            "images": [base64.b64encode(img).decode() for img in images],  # 数MB
        }
        # 問題: ブローカーのメモリ/ディスクを圧迫
        # 問題: ネットワーク帯域を浪費
        # 問題: Consumer のデシリアライズが遅い
        self.producer.send("invoices", json.dumps(message).encode())


# OK: Claim-Check パターン（参照のみキューに格納）

import boto3

class GoodProducer:
    def __init__(self, producer, s3_client):
        self.producer = producer
        self.s3 = s3_client

    def send_invoice(self, order_id: int, pdf_data: bytes, images: list[bytes]):
        # Step 1: 大きなデータは S3 に保存
        pdf_key = f"invoices/{order_id}/invoice.pdf"
        self.s3.put_object(Bucket='my-bucket', Key=pdf_key, Body=pdf_data)

        image_keys = []
        for i, img in enumerate(images):
            key = f"invoices/{order_id}/image_{i}.jpg"
            self.s3.put_object(Bucket='my-bucket', Key=key, Body=img)
            image_keys.append(key)

        # Step 2: 軽量な参照のみキューに格納
        message = {
            "order_id": order_id,
            "invoice_s3_key": pdf_key,
            "image_s3_keys": image_keys,
        }
        self.producer.send("invoices", json.dumps(message).encode())
        # メッセージサイズ: 数百バイト（vs 数十MB）
```

### アンチパターン 2: 配信保証を考慮しない設計

```python
# NG: Auto-Ack で fire-and-forget

def bad_consumer():
    channel.basic_consume(
        queue='payment_queue',
        auto_ack=True,  # 問題: 受信した瞬間にAck
        on_message_callback=process_payment
    )
    # Consumer がクラッシュ → メッセージ消失 → 決済データ欠損


# OK: 手動 Ack + DLQ + べき等処理

def good_consumer():
    channel.basic_qos(prefetch_count=5)  # 同時処理数を制限
    channel.basic_consume(
        queue='payment_queue',
        auto_ack=False,  # 手動Ack
        on_message_callback=safe_process_payment
    )

def safe_process_payment(ch, method, properties, body):
    try:
        # べき等処理（重複を安全にスキップ）
        payment = json.loads(body)
        if is_already_processed(payment['payment_id']):
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        execute_payment(payment)
        mark_as_processed(payment['payment_id'])
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception:
        # 失敗 → DLQ へ（requeue=False で無限ループ防止）
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
```

### アンチパターン 3: Consumer の処理速度を考慮しないスケーリング

```python
# NG: Consumer 1台で大量メッセージを処理

class UnscalableConsumer:
    """Producer 1000 msg/sec vs Consumer 100 msg/sec
    → キューが無限に溜まり続け、遅延が際限なく増大"""
    pass


# OK: オートスケーリング + バックプレッシャー

class ScalableConsumerManager:
    """キュー深度に基づくオートスケーリング"""

    def __init__(self, queue_name: str,
                 target_lag: int = 1000,
                 max_consumers: int = 20,
                 min_consumers: int = 2):
        self.queue_name = queue_name
        self.target_lag = target_lag
        self.max_consumers = max_consumers
        self.min_consumers = min_consumers
        self.current_consumers = min_consumers

    def check_and_scale(self, current_lag: int):
        """キュー深度に基づいてConsumer数を調整"""
        if current_lag > self.target_lag * 2:
            # スケールアウト
            desired = min(
                self.current_consumers * 2,
                self.max_consumers
            )
            self._scale_to(desired)
            print(f"[SCALE OUT] {self.current_consumers} → {desired} "
                  f"(lag={current_lag})")
        elif current_lag < self.target_lag * 0.2:
            # スケールイン
            desired = max(
                self.current_consumers // 2,
                self.min_consumers
            )
            self._scale_to(desired)
            print(f"[SCALE IN] {self.current_consumers} → {desired} "
                  f"(lag={current_lag})")

    def _scale_to(self, count: int):
        self.current_consumers = count
        # 実際にはKubernetes HPA, ECS Service等で実装
```

---

## 9. 練習問題

### 演習1（基礎）: べき等 Consumer の重複排除テスト

**課題**: IdempotentConsumer を使い、10件のメッセージ（うち3件が重複）を処理し、実際に処理されたメッセージ数と重複スキップ数を計測せよ。

```python
# ヒント: IdempotentConsumer クラスを使用
consumer = OrderConsumer(redis.Redis())
messages = [
    {"message_id": f"msg-{i}", "order_id": i, "amount": 1000 * i}
    for i in range(7)
]
# 重複を追加
messages.extend([
    {"message_id": "msg-0", "order_id": 0, "amount": 0},
    {"message_id": "msg-3", "order_id": 3, "amount": 3000},
    {"message_id": "msg-5", "order_id": 5, "amount": 5000},
])

processed = sum(1 for msg in messages if consumer.process(msg))
skipped = len(messages) - processed
print(f"処理: {processed}, スキップ: {skipped}")
```

**期待される出力**:
```
処理: 7, スキップ: 3
```

### 演習2（応用）: DLQ 付きリトライフローの実装

**課題**: メッセージ処理が30%の確率で失敗する環境で、最大3回リトライ後にDLQに送るConsumerを実装し、100メッセージの処理結果（成功/DLQ送信）を集計せよ。

```python
import random

class RetryableConsumer:
    def __init__(self, max_retries: int = 3, failure_rate: float = 0.3):
        self.max_retries = max_retries
        self.failure_rate = failure_rate
        self.success_count = 0
        self.dlq_count = 0

    def process_with_retry(self, message: dict) -> str:
        for attempt in range(self.max_retries + 1):
            if random.random() > self.failure_rate:
                self.success_count += 1
                return "success"
        self.dlq_count += 1
        return "dlq"

consumer = RetryableConsumer(max_retries=3, failure_rate=0.3)
random.seed(42)
for i in range(100):
    consumer.process_with_retry({"id": i})
print(f"成功: {consumer.success_count}, DLQ: {consumer.dlq_count}")
```

**期待される出力（概算）**:
```
成功: ~99, DLQ: ~1
(30%失敗率で4回試行: 失敗確率 = 0.3^4 = 0.81% → 100件中約1件がDLQ)
```

### 演習3（発展）: Outbox パターンの完全実装

**課題**: SQLAlchemy + Kafka を使い、注文作成とイベント発行の原子性を保証する Outbox パターンを実装せよ。以下の要件を満たすこと。

1. 注文テーブルと Outbox テーブルを同一トランザクションで更新
2. Outbox Relay プロセスが未送信イベントをポーリングして Kafka に発行
3. 送信済みイベントの定期クリーンアップ

---

## 10. FAQ

### Q1. Kafka と RabbitMQ のどちらを選ぶべき？

ユースケースで判断する。大量データのストリーミング処理（ログ集約、イベントソーシング、リアルタイム分析パイプライン）なら **Kafka**。タスクキュー、RPC パターン、複雑なルーティング（優先度キュー、トピックベースフィルタリング）が必要なら **RabbitMQ**。迷ったら「メッセージの再読が必要か」で判断する。再読が必要なら Kafka 一択。

### Q2. Consumer がダウンした場合、メッセージはどうなる？

ブローカーがメッセージを保持し、Consumer 復旧後に再配信する。**Kafka** はオフセットベースで Consumer が自分のペースで読み進めるため、ダウンタイム中のメッセージは保持期間内なら失われない。**RabbitMQ** は Ack タイムアウト後にキューに戻す。**SQS** は Visibility Timeout 後に再度取得可能になる。いずれも Consumer 側のべき等処理が重複配信への対策として必須。

### Q3. メッセージの順序保証はどの粒度まで可能？

完全なグローバル順序保証はスケーラビリティとトレードオフになる。**Kafka** はパーティションキーで同一パーティションに送れば順序保証される（パーティション間は不保証）。**SQS FIFO** は MessageGroupId 単位で順序保証（グループ間は並列処理）。**RabbitMQ** は単一キュー・単一 Consumer で FIFO 保証。設計時に「どのエンティティ単位で順序が必要か」を明確にし、そのキー（ユーザー ID、注文 ID など）をパーティションキーに使うのが定石。

### Q4. Kafka の Consumer Group とは何か？

同じ group.id を持つ Consumer の集合体。トピックの各パーティションは、グループ内の1つの Consumer にのみ割り当てられる。これにより、Consumer を追加するだけで水平スケールでき、パーティション数 = 最大並列度となる。異なる Consumer Group は同じメッセージを独立して消費できるため、Pub/Sub モデルが実現される。Consumer がグループに参加/離脱すると**リバランス**が発生し、パーティションの再割り当てが行われる。

### Q5. Consumer Lag をどう監視・対処すべきか？

Consumer Lag（= 最新オフセット - Consumer の現在オフセット）はメッセージ処理の遅延を示す重要指標。**監視**: Kafka の `kafka-consumer-groups.sh --describe` コマンド、またはBurrow/Prometheus Exporterで継続監視。**アラート閾値**: Lag > 10,000 で警告、Lag > 100,000 で緊急。**対処**: (1) Consumer 数を増やす（パーティション数以下）、(2) Consumer のバッチサイズを最適化、(3) 処理ロジックの高速化（DBバッチ書き込み等）、(4) パーティション数の増加（ただし増加のみ可能、削減不可）。

### Q6. Exactly-once は本当に実現可能か？

理論的には分散システムで完全な Exactly-once は不可能だが、**実用的な Exactly-once** は以下の方法で実現できる。(1) **Kafka Transactions**: Producer と Consumer を同一トランザクション内で処理（Kafka Streams で利用）。(2) **Outbox パターン**: DB トランザクション + CDC で原子的にイベント発行。(3) **Consumer 側べき等処理**: At-least-once + 重複排除（最も一般的で推奨される方法）。特にマイクロサービスでは方式(3)が最も実用的で、メッセージIDをDBのユニーク制約で管理する方法がシンプルかつ確実。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| メッセージキューの役割 | コンポーネント間の疎結合、非同期処理、負荷平準化、耐障害性 |
| 配信保証 | At-most-once / At-least-once / Exactly-once を要件で選択 |
| Kafka | 高スループット・ストリーミング向け。パーティション分散。再読可能 |
| RabbitMQ | 柔軟なルーティング・タスクキュー向け。Exchange/Binding モデル |
| SQS | フルマネージド・サーバーレス連携。運用コスト最小 |
| DLQ | 処理失敗メッセージの隔離と後続対応に必須 |
| べき等処理 | At-least-once では重複配信を前提とした設計が必須 |
| Outbox パターン | DB更新とメッセージ発行の原子性を保証する標準手法 |
| ペイロード設計 | 大きなデータは外部ストレージに保存し参照のみキューに載せる |
| Consumer Lag | 最重要監視指標。オートスケーリングで自動対応 |

---

## 次に読むべきガイド

- [CDN](./03-cdn.md) -- コンテンツ配信ネットワークによるレイテンシ最適化
- [DBスケーリング](./04-database-scaling.md) -- データ層のシャーディングとレプリケーション
- [イベント駆動アーキテクチャ](../02-architecture/03-event-driven.md) -- メッセージキューを活用した Pub/Sub 設計
- [キャッシュ](./01-caching.md) -- メッセージキューと併用するキャッシュ更新戦略
- [信頼性](../00-fundamentals/02-reliability.md) -- リトライ、サーキットブレーカーとの連携

---

## 参考文献

1. **Designing Data-Intensive Applications** -- Martin Kleppmann (O'Reilly, 2017) -- 分散メッセージングの理論と実践の定番書
2. **Kafka: The Definitive Guide, 2nd Edition** -- Gwen Shapira et al. (O'Reilly, 2021) -- Kafka の包括的リファレンス
3. **RabbitMQ in Depth** -- Gavin M. Roy (Manning, 2017) -- RabbitMQ の内部アーキテクチャと運用パターン
4. **Amazon SQS Developer Guide** -- AWS Documentation -- https://docs.aws.amazon.com/sqs/
5. **Enterprise Integration Patterns** -- Gregor Hohpe & Bobby Woolf (Addison-Wesley, 2003) -- メッセージングパターンの古典

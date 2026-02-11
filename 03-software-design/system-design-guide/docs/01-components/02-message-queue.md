# メッセージキュー

> 分散システムにおける非同期メッセージングの基盤技術であり、コンポーネント間の疎結合・スケーラビリティ・耐障害性を実現する中核パターンを、Kafka・RabbitMQ・SQS の実装比較を通じて解説する

## この章で学ぶこと

1. **メッセージキューの基本概念** — Producer/Consumer モデル、キューとトピックの違い、配信保証（At-most-once / At-least-once / Exactly-once）
2. **主要プロダクトの比較と選定** — Apache Kafka、RabbitMQ、Amazon SQS のアーキテクチャ・性能特性・適材適所
3. **実践的な設計パターン** — Dead Letter Queue、バックプレッシャー、べき等処理、順序保証の実装手法

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

---

## 2. 配信保証モデル

| 保証レベル | 説明 | メッセージ損失 | 重複配信 | 代表的ユースケース |
|-----------|------|:------------:|:-------:|-----------------|
| At-most-once | 最大1回配信。再送なし | あり得る | なし | ログ収集、メトリクス送信 |
| At-least-once | 最低1回配信。Ack失敗で再送 | なし | あり得る | 注文処理、メール送信 |
| Exactly-once | 正確に1回配信 | なし | なし | 決済処理、在庫更新 |

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

conf = {
    'bootstrap.servers': 'broker1:9092,broker2:9092,broker3:9092',
    'acks': 'all',                    # 全ISRレプリカの書き込み確認
    'retries': 5,
    'retry.backoff.ms': 100,
    'enable.idempotence': True,       # べき等プロデューサー（重複防止）
    'compression.type': 'snappy',     # 圧縮でスループット向上
    'linger.ms': 5,                   # バッチ化のための待機時間
    'batch.size': 32768,              # バッチサイズ (32KB)
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
    order = {"order_id": i, "user_id": i % 100, "amount": 1000 + i}
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
```

### 4.2 Producer / Consumer

```python
# RabbitMQ Producer (Python - pika)
import pika
import json

connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host='localhost',
        credentials=pika.PlainCredentials('app_user', 'secret'),
        heartbeat=600,
    )
)
channel = connection.channel()

# Exchange と Queue を宣言
channel.exchange_declare(exchange='order_exchange', exchange_type='topic', durable=True)
channel.queue_declare(queue='order_processing', durable=True, arguments={
    'x-dead-letter-exchange': 'dlx_exchange',        # DLQ 設定
    'x-dead-letter-routing-key': 'order.failed',
    'x-message-ttl': 86400000,                        # TTL: 24時間
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
        message_id='msg-uuid-001',
    ),
)
print(f"送信完了: {message}")
connection.close()
```

```python
# RabbitMQ Consumer (Python - pika)
import pika
import json
import traceback

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='order_processing', durable=True)
channel.basic_qos(prefetch_count=10)    # 同時処理数を制御

def callback(ch, method, properties, body):
    order = json.loads(body)
    print(f"処理開始: order_id={order['order_id']}")
    try:
        process_order(order)
        ch.basic_ack(delivery_tag=method.delivery_tag)        # 成功 → Ack
        print(f"処理完了: order_id={order['order_id']}")
    except Exception as e:
        print(f"処理失敗: {e}")
        traceback.print_exc()
        ch.basic_nack(delivery_tag=method.delivery_tag,
                      requeue=False)                          # 失敗 → DLQ へ

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

sqs = boto3.client('sqs', region_name='ap-northeast-1')
QUEUE_URL = 'https://sqs.ap-northeast-1.amazonaws.com/123456789/order-queue.fifo'

# --- Producer ---
response = sqs.send_message(
    QueueUrl=QUEUE_URL,
    MessageBody=json.dumps({"order_id": 456, "amount": 8000}),
    MessageAttributes={
        'EventType': {'DataType': 'String', 'StringValue': 'OrderCreated'}
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

        process_order(body)

        sqs.delete_message(
            QueueUrl=QUEUE_URL,
            ReceiptHandle=message['ReceiptHandle'],
        )
        print(f"処理完了・削除: order_id={body['order_id']}")
```

---

## 6. 主要プロダクト比較表

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

### 選定フローチャート比較表

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

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def idempotent_consumer(message):
    """同じメッセージを何度受信しても結果が変わらないことを保証する"""
    # 一意な処理キーを生成
    idempotency_key = f"processed:{message['message_id']}"

    # SETNX (Set if Not eXists) で排他制御
    if not redis_client.set(idempotency_key, 'processing', nx=True, ex=3600):
        print(f"スキップ（処理済み or 処理中）: {message['message_id']}")
        return  # 既に処理済み

    try:
        # ビジネスロジック実行
        result = execute_business_logic(message)

        # 処理完了マーク
        redis_client.set(idempotency_key, 'completed', ex=86400 * 7)
        return result

    except Exception as e:
        # 失敗時はキーを削除してリトライ可能にする
        redis_client.delete(idempotency_key)
        raise
```

---

## 8. アンチパターン

### アンチパターン 1: キューを巨大データストアとして使う

```
BAD: 大きなペイロードをキューに直接格納
{
  "order_id": 123,
  "pdf_invoice": "<10MB の Base64 データ>",
  "images": ["<5MB>", "<3MB>"]
}
--> ブローカーのメモリ/ディスクを圧迫、スループット激減

GOOD: 参照（ポインタ）のみをキューに格納 (Claim-Check パターン)
{
  "order_id": 123,
  "invoice_s3_key": "invoices/2026/01/123.pdf",
  "image_s3_keys": ["images/a.jpg", "images/b.jpg"]
}
--> データは S3 に保存、キューは軽量な参照のみ
```

### アンチパターン 2: 配信保証を考慮しない設計

```python
# BAD: Auto-Ack で fire-and-forget
channel.basic_consume(queue='payment_queue', auto_ack=True,
                      on_message_callback=process_payment)
# Consumer がクラッシュ → メッセージ消失 → 決済データ欠損

# GOOD: 手動 Ack + DLQ + べき等処理
channel.basic_qos(prefetch_count=5)
channel.basic_consume(queue='payment_queue', auto_ack=False,
                      on_message_callback=safe_process_payment)
# 処理完了時のみ Ack、失敗時は DLQ へ、再処理はべき等
```

### アンチパターン 3: Consumer の処理速度を考慮しないスケーリング

```
BAD:
  Producer (1000 msg/sec) --> [Queue] --> Consumer 1台 (100 msg/sec)
  --> キューが無限に溜まり続け、遅延が際限なく増大

GOOD:
  Producer (1000 msg/sec) --> [Queue] --> Consumer 10台 (100 msg/sec x 10)
  + バックプレッシャー機構（キュー深度監視 → Producer 速度制御）
  + オートスケーリング（キュー深度 > 閾値 → Consumer 追加）
```

---

## 9. FAQ

### Q1. Kafka と RabbitMQ のどちらを選ぶべき？

**A.** ユースケースで判断する。大量データのストリーミング処理（ログ集約、イベントソーシング、リアルタイム分析パイプライン）なら **Kafka**。タスクキュー、RPC パターン、複雑なルーティング（優先度キュー、トピックベースフィルタリング）が必要なら **RabbitMQ**。迷ったら「メッセージの再読が必要か」で判断する。再読が必要なら Kafka 一択。

### Q2. Consumer がダウンした場合、メッセージはどうなる？

**A.** ブローカーがメッセージを保持し、Consumer 復旧後に再配信する。**Kafka** はオフセットベースで Consumer が自分のペースで読み進めるため、ダウンタイム中のメッセージは保持期間内なら失われない。**RabbitMQ** は Ack タイムアウト後にキューに戻す。**SQS** は Visibility Timeout 後に再度取得可能になる。いずれも Consumer 側のべき等処理が重複配信への対策として必須。

### Q3. メッセージの順序保証はどの粒度まで可能？

**A.** 完全なグローバル順序保証はスケーラビリティとトレードオフになる。**Kafka** はパーティションキーで同一パーティションに送れば順序保証される（パーティション間は不保証）。**SQS FIFO** は MessageGroupId 単位で順序保証（グループ間は並列処理）。**RabbitMQ** は単一キュー・単一 Consumer で FIFO 保証。設計時に「どのエンティティ単位で順序が必要か」を明確にし、そのキー（ユーザー ID、注文 ID など）をパーティションキーに使うのが定石。

### Q4. Kafka の Consumer Group とは何か？

**A.** 同じ group.id を持つ Consumer の集合体。トピックの各パーティションは、グループ内の1つの Consumer にのみ割り当てられる。これにより、Consumer を追加するだけで水平スケールでき、パーティション数 = 最大並列度となる。異なる Consumer Group は同じメッセージを独立して消費できるため、Pub/Sub モデルが実現される。

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
| ペイロード設計 | 大きなデータは外部ストレージに保存し参照のみキューに載せる |

---

## 次に読むべきガイド

- [CDN](./03-cdn.md) — コンテンツ配信ネットワークによるレイテンシ最適化
- [DBスケーリング](./04-database-scaling.md) — データ層のシャーディングとレプリケーション
- [イベント駆動アーキテクチャ](../02-architecture/03-event-driven.md) — メッセージキューを活用した Pub/Sub 設計

---

## 参考文献

1. **Designing Data-Intensive Applications** — Martin Kleppmann (O'Reilly, 2017) — 分散メッセージングの理論と実践の定番書
2. **Kafka: The Definitive Guide, 2nd Edition** — Gwen Shapira et al. (O'Reilly, 2021) — Kafka の包括的リファレンス
3. **RabbitMQ in Depth** — Gavin M. Roy (Manning, 2017) — RabbitMQ の内部アーキテクチャと運用パターン
4. **Amazon SQS Developer Guide** — AWS Documentation — https://docs.aws.amazon.com/sqs/

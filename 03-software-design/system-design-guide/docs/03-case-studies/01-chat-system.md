# チャットシステム設計

> LINE / Slack / WhatsApp のようなリアルタイムチャットシステムをゼロから設計する。WebSocket 管理、メッセージ配信、オフライン対応、グループチャットの設計パターンを体系的に学ぶ。

---

## この章で学ぶこと

1. **リアルタイム通信基盤** — WebSocket 接続管理、プレゼンス管理、ハートビートの設計
2. **メッセージ配信** — 1対1 チャット、グループチャット、メッセージ順序保証の仕組み
3. **スケーラビリティと信頼性** — メッセージ永続化、オフライン配信、既読管理、プッシュ通知連携
4. **セキュリティ** — エンドツーエンド暗号化、メッセージの完全性保証
5. **運用設計** — 監視、障害対応、容量計画

---

## 前提知識

| トピック | 必要レベル | 参考ガイド |
|---------|-----------|-----------|
| WebSocket プロトコル | HTTP アップグレード、双方向通信の基礎 | [ネットワーク基礎](../../../04-web-and-network/network-guide/docs/00-foundations/00-osi-model.md) |
| メッセージキュー | Kafka / RabbitMQ の概念 | [メッセージキュー](../01-components/02-message-queue.md) |
| NoSQL データベース | Cassandra / DynamoDB の基本 | [データベース](../01-components/01-database.md) |
| キャッシュ設計 | Redis の基本操作 | [キャッシュ](../01-components/00-cache.md) |
| イベント駆動設計 | Pub/Sub の概念 | [イベント駆動アーキテクチャ](../02-architecture/03-event-driven.md) |

---

## 背景

### なぜチャットシステムの設計は難しいのか

```
チャットシステムの技術的課題:

  1. リアルタイム性:
     → メッセージの遅延が 100ms を超えるとユーザーが不満を感じる
     → 数百万の同時接続を低レイテンシで処理する必要がある

  2. 信頼性:
     → メッセージの喪失は絶対に許容できない
     → 順序の逆転もUXを大きく損なう
     → ネットワーク断裂時もメッセージを保証する

  3. スケーラビリティ:
     → DAU 5,000万、同時接続 500万
     → グループチャットのファンアウト（1メッセージ → 500人に配信）
     → 書き込みと読み取りの両方が高スループット

  4. プレゼンス管理:
     → 「オンライン/オフライン」の状態をリアルタイムで追跡
     → 500万同時接続のプレゼンス情報を効率的に管理

面接での出題頻度が高い理由:
  - WebSocket（リアルタイム通信）
  - メッセージキュー（非同期処理）
  - データベース設計（高書き込みスループット）
  - キャッシュ設計（プレゼンス、既読）
  - プッシュ通知（外部サービス連携）
  → 1つの問題で多くの設計要素をカバーできる
```

---

## 1. 要件定義

### 1.1 機能要件

```
=== 必須機能 (Must Have) ===
- 1対1 チャット（テキストメッセージ）
- グループチャット（最大 500 人）
- オンライン/オフラインのプレゼンス表示
- メッセージの永続化と履歴表示
- プッシュ通知（オフラインユーザー向け）

=== 重要機能 (Should Have) ===
- メッセージの既読管理（既読/未読、既読マーク）
- 画像・ファイルの送受信
- メッセージの検索
- タイピングインジケーター

=== 追加機能 (Nice to Have) ===
- メッセージの編集・削除
- スレッド/リプライ
- リアクション（絵文字）
- 音声/ビデオ通話（WebRTC）
- エンドツーエンド暗号化（E2EE）
```

### 1.2 非機能要件

```
=== パフォーマンス ===
- メッセージ配信レイテンシ: P99 < 200ms
- メッセージ永続化: P99 < 100ms
- プレゼンス更新: P99 < 500ms

=== 可用性 ===
- メッセージ配信: 99.99%
- メッセージ喪失率: 0%（at-least-once 配信保証）

=== スケーラビリティ ===
- 同時 WebSocket 接続: 500万
- メッセージ QPS: 50,000（ピーク）
- 1日あたりメッセージ: 20億
```

### 1.3 スケール見積もり

```python
# === チャットシステムのスケール見積もり ===

dau = 50_000_000                    # DAU: 5,000万
messages_per_user_per_day = 40       # 1ユーザー平均 40メッセージ/日
daily_messages = dau * messages_per_user_per_day  # 20億メッセージ/日

# QPS
message_qps = daily_messages / 86400           # ≈ 23,148 QPS
message_qps_peak = message_qps * 2.5           # ≈ 57,870 QPS (ピーク)

# 同時接続
concurrent_connections = dau * 0.10             # 500万同時接続 (DAU の 10%)

# ストレージ
avg_message_size_bytes = 200                    # テキスト + メタデータ
daily_storage_gb = (daily_messages * avg_message_size_bytes) / (1024**3)  # ≈ 373 GB/日
yearly_storage_tb = daily_storage_gb * 365 / 1024  # ≈ 133 TB/年

# 帯域幅
# 1メッセージ送信 → 平均 2.5人に配信（1対1 + グループの平均）
fan_out_factor = 2.5
delivery_qps = message_qps_peak * fan_out_factor  # ≈ 144,675 配信/秒
bandwidth_mbps = (delivery_qps * avg_message_size_bytes * 8) / (1024**2)  # ≈ 22 Mbps

# WebSocket メモリ
# 1接続 ≈ 10KB (バッファ + メタデータ)
ws_memory_gb = (concurrent_connections * 10 * 1024) / (1024**3)  # ≈ 47 GB
ws_servers = ws_memory_gb / 16  # 1サーバー 16GB で ≈ 3台（最低）

print(f"""
=== チャットシステム スケール見積もり ===
メッセージ QPS:     {message_qps:,.0f} (ピーク: {message_qps_peak:,.0f})
同時 WebSocket:     {concurrent_connections:,.0f}
ストレージ:         {daily_storage_gb:.0f} GB/日, {yearly_storage_tb:.0f} TB/年
配信スループット:    {delivery_qps:,.0f} 配信/秒
WebSocket メモリ:    {ws_memory_gb:.0f} GB → 最低 {ws_servers:.0f} サーバー
""")
```

---

## 2. 高レベルアーキテクチャ

### 2.1 全体構成

```
                    チャットシステム全体構成

  ┌──────────┐                                    ┌──────────┐
  │ Client A │                                    │ Client B │
  │ (Mobile/ │                                    │ (Mobile/ │
  │  Web)    │                                    │  Web)    │
  └────┬─────┘                                    └────┬─────┘
       │ WebSocket                                     │ WebSocket
       v                                               v
  ┌──────────┐                                    ┌──────────┐
  │ WS       │                                    │ WS       │
  │ Gateway  │                                    │ Gateway  │
  │ (GW-1)   │                                    │ (GW-2)   │
  └────┬─────┘                                    └────┬─────┘
       │                                               │
       └──────────────────┬────────────────────────────┘
                          │
                    ┌─────┴─────┐
                    │ Message   │
                    │ Service   │
                    └─────┬─────┘
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            v             v             v
     ┌──────────┐  ┌──────────┐  ┌──────────┐
     │ Kafka    │  │ Cassandra│  │ Redis    │
     │ (メッセ  │  │ (メッセ  │  │ (接続    │
     │  ージQ)  │  │  ージDB) │  │  マップ,  │
     └──────────┘  └──────────┘  │  プレゼン │
                                 │  ス,既読) │
                                 └──────────┘

  補助サービス:
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Push     │  │ File     │  │ Search   │
  │ Notif.   │  │ Upload   │  │ (Elastic │
  │ (APNS/   │  │ (S3)     │  │  search) │
  │  FCM)    │  │          │  │          │
  └──────────┘  └──────────┘  └──────────┘
```

### 2.2 接続管理アーキテクチャ

```
=== WebSocket Gateway の役割 ===

  1. WebSocket 接続の終端
     Client ←── WebSocket ──→ Gateway ←── gRPC/HTTP ──→ Message Service

  2. 接続マップの管理 (Redis)
     user_id → gateway_id のマッピングを保持
     → どのユーザーがどの Gateway に接続しているかを追跡

  3. メッセージのルーティング
     受信メッセージ → Message Service → 宛先の Gateway → 受信者の WebSocket

  4. ハートビート管理
     30秒ごとに ping/pong で接続の生存確認
     → タイムアウト → 切断 → プレゼンス更新

  ┌────────────────────────────────────────────────┐
  │  Redis 接続マップ (Hash)                        │
  │                                                 │
  │  ws:connections                                  │
  │    user_A → gw-001                              │
  │    user_B → gw-001                              │
  │    user_C → gw-002                              │
  │    user_D → gw-002                              │
  │                                                 │
  │  メッセージ配信フロー (User A → User C):          │
  │  1. GW-1 がメッセージを受信                       │
  │  2. Message Service がメッセージを永続化           │
  │  3. Redis で User C の Gateway を検索 → GW-2     │
  │  4. Redis Pub/Sub で GW-2 にメッセージを転送      │
  │  5. GW-2 が User C の WebSocket に送信           │
  └────────────────────────────────────────────────┘
```

---

## 3. コア実装

### 3.1 WebSocket Gateway

```python
# infrastructure/gateway/websocket_gateway.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import redis.asyncio as aioredis
import json
import uuid
import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

app = FastAPI()
redis_client = aioredis.Redis(host="redis-host", port=6379)

# このゲートウェイインスタンスの ID
GATEWAY_ID = f"gw-{uuid.uuid4().hex[:8]}"

# ローカル接続マップ (user_id -> WebSocket)
local_connections: dict[str, WebSocket] = {}

# ハートビート設定
HEARTBEAT_INTERVAL = 30  # 秒
HEARTBEAT_TIMEOUT = 90   # 秒（3回ミス = 切断とみなす）
PRESENCE_TTL = 300       # 秒（プレゼンスの TTL）


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket, user_id: str, token: str):
    """WebSocket エンドポイント

    接続フロー:
    1. 認証（JWT トークン検証）
    2. 接続登録（Redis + ローカルマップ）
    3. プレゼンス更新（オンライン）
    4. オフラインメッセージの配信
    5. メッセージ送受信ループ
    6. 切断処理
    """
    # 1. 認証
    if not await authenticate_ws(token, user_id):
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()
    logger.info(f"WebSocket 接続: user_id={user_id}, gw={GATEWAY_ID}")

    # 2. 接続登録
    local_connections[user_id] = websocket
    await redis_client.hset("ws:connections", user_id, GATEWAY_ID)

    # 3. プレゼンス更新
    await redis_client.set(f"presence:{user_id}", "online", ex=PRESENCE_TTL)
    await broadcast_presence_change(user_id, "online")

    # 4. オフラインメッセージの配信
    await deliver_offline_messages(user_id, websocket)

    # 5. ハートビートタスクを開始
    heartbeat_task = asyncio.create_task(
        heartbeat_loop(user_id, websocket)
    )

    try:
        # 6. メッセージ送受信ループ
        while True:
            data = await websocket.receive_json()
            await handle_message(user_id, data)
    except WebSocketDisconnect:
        logger.info(f"WebSocket 切断: user_id={user_id}")
    except Exception as e:
        logger.error(f"WebSocket エラー: user_id={user_id}, error={e}")
    finally:
        # 7. 切断処理
        heartbeat_task.cancel()
        local_connections.pop(user_id, None)
        await redis_client.hdel("ws:connections", user_id)
        await redis_client.set(f"presence:{user_id}", "offline")
        await broadcast_presence_change(user_id, "offline")


async def heartbeat_loop(user_id: str, websocket: WebSocket):
    """ハートビートループ: 定期的に ping を送信

    WebSocket の接続状態を確認し、プレゼンスの TTL を延長する
    """
    try:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            try:
                await websocket.send_json({"type": "ping"})
                # プレゼンス TTL を延長
                await redis_client.set(
                    f"presence:{user_id}", "online", ex=PRESENCE_TTL
                )
            except Exception:
                logger.warning(f"ハートビート失敗: user_id={user_id}")
                break
    except asyncio.CancelledError:
        pass


async def handle_message(sender_id: str, data: dict):
    """受信メッセージをルーティングする"""
    msg_type = data.get("type")

    if msg_type == "chat":
        await handle_chat_message(sender_id, data)
    elif msg_type == "group_chat":
        await handle_group_message(sender_id, data)
    elif msg_type == "typing":
        await handle_typing_indicator(sender_id, data)
    elif msg_type == "read_receipt":
        await handle_read_receipt(sender_id, data)
    elif msg_type == "pong":
        pass  # ハートビート応答
    else:
        logger.warning(f"未知のメッセージタイプ: {msg_type}")


async def handle_chat_message(sender_id: str, data: dict):
    """1対1 チャットメッセージの処理"""
    recipient_id = data["to"]
    message = {
        "id": str(uuid.uuid4()),
        "from": sender_id,
        "to": recipient_id,
        "content": data["content"],
        "content_type": data.get("content_type", "text"),
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "type": "chat",
    }

    # 1. メッセージを永続化（Kafka 経由で非同期）
    await publish_to_kafka("messages", message)

    # 2. 送信者に ACK を返す
    sender_ws = local_connections.get(sender_id)
    if sender_ws:
        await sender_ws.send_json({
            "type": "ack",
            "message_id": message["id"],
            "status": "sent",
        })

    # 3. 受信者に配信
    await deliver_message(recipient_id, message)


async def deliver_message(recipient_id: str, message: dict):
    """メッセージを受信者に配信する

    配信フロー:
    1. Redis で受信者の Gateway を検索
    2. 同一 Gateway → 直接送信
    3. 異なる Gateway → Redis Pub/Sub で転送
    4. オフライン → プッシュ通知 + オフラインキュー
    """
    gateway_id = await redis_client.hget("ws:connections", recipient_id)

    if gateway_id is None:
        # オフライン → プッシュ通知 + オフラインキュー
        logger.info(f"オフライン配信: recipient={recipient_id}")
        await enqueue_offline_message(recipient_id, message)
        await send_push_notification(recipient_id, message)
        return

    gateway_id = gateway_id.decode()

    if gateway_id == GATEWAY_ID:
        # 同じ Gateway → 直接送信
        ws = local_connections.get(recipient_id)
        if ws:
            try:
                await ws.send_json(message)
                logger.debug(f"直接配信: recipient={recipient_id}")
            except Exception as e:
                logger.error(f"配信失敗: recipient={recipient_id}, {e}")
                await enqueue_offline_message(recipient_id, message)
        else:
            await enqueue_offline_message(recipient_id, message)
    else:
        # 異なる Gateway → Redis Pub/Sub で転送
        channel = f"gw:{gateway_id}"
        await redis_client.publish(channel, json.dumps(message))
        logger.debug(f"Pub/Sub 転送: recipient={recipient_id}, gw={gateway_id}")


async def deliver_offline_messages(user_id: str, websocket: WebSocket):
    """オフライン中に蓄積されたメッセージを配信"""
    queue_key = f"offline:{user_id}"
    messages = await redis_client.lrange(queue_key, 0, -1)

    if messages:
        logger.info(f"オフラインメッセージ配信: user={user_id}, count={len(messages)}")
        for msg_data in messages:
            try:
                message = json.loads(msg_data)
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"オフラインメッセージ配信失敗: {e}")

        # 配信完了後にキューをクリア
        await redis_client.delete(queue_key)


async def enqueue_offline_message(user_id: str, message: dict):
    """オフラインメッセージキューに追加"""
    queue_key = f"offline:{user_id}"
    await redis_client.rpush(queue_key, json.dumps(message))
    # 最大 1000 メッセージを保持（古いものは切り捨て）
    await redis_client.ltrim(queue_key, -1000, -1)
    # 7日間保持
    await redis_client.expire(queue_key, 7 * 86400)


async def broadcast_presence_change(user_id: str, status: str):
    """プレゼンス変更を関連ユーザーに通知"""
    # ユーザーの友達/チャット相手にプレゼンス変更を通知
    # 実装省略: 友達リストを取得して各友達に通知
    pass


async def send_push_notification(user_id: str, message: dict):
    """プッシュ通知を送信（APNS / FCM 経由）"""
    # 実装省略: Push Notification Service に委譲
    pass


async def publish_to_kafka(topic: str, message: dict):
    """Kafka にメッセージを発行"""
    # 実装省略: confluent_kafka の Producer を使用
    pass


async def authenticate_ws(token: str, user_id: str) -> bool:
    """WebSocket 接続の認証"""
    # 実装省略: JWT トークンの検証
    return True
```

### 3.2 グループチャットの配信

```python
# application/services/group_chat_service.py
import uuid
import time
import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class GroupChatService:
    """グループチャットの配信サービス

    設計判断:
    - 小グループ（<50人）: 書き込み時ファンアウト（即時配信）
    - 大グループ（>500人）: 読み取り時ファンアウト（遅延許容）
    - ハイブリッド: グループサイズに応じて自動切り替え
    """

    SMALL_GROUP_THRESHOLD = 50
    LARGE_GROUP_THRESHOLD = 500

    def __init__(self, redis_client, db, kafka_producer, push_service):
        self._redis = redis_client
        self._db = db
        self._kafka = kafka_producer
        self._push = push_service

    async def send_group_message(
        self,
        sender_id: str,
        group_id: str,
        content: str,
        content_type: str = "text",
    ):
        """グループメッセージを全メンバーに配信する"""
        message = {
            "id": str(uuid.uuid4()),
            "from": sender_id,
            "group_id": group_id,
            "content": content,
            "content_type": content_type,
            "timestamp": int(time.time() * 1000),
            "type": "group_chat",
        }

        # 1. メッセージを永続化（Kafka 経由で非同期）
        await self._kafka.send("group-messages", message)

        # 2. 送信者に ACK を返す
        await deliver_message(sender_id, {
            "type": "ack",
            "message_id": message["id"],
            "status": "sent",
        })

        # 3. グループメンバーを取得
        members = await self.get_group_members(group_id)
        group_size = len(members)

        # 4. グループサイズに応じたファンアウト戦略
        if group_size <= self.SMALL_GROUP_THRESHOLD:
            await self._fanout_push(sender_id, members, message)
        elif group_size <= self.LARGE_GROUP_THRESHOLD:
            await self._fanout_push_async(sender_id, members, message)
        else:
            await self._fanout_pull(sender_id, group_id, message)

    async def _fanout_push(
        self,
        sender_id: str,
        members: list[str],
        message: dict,
    ):
        """書き込み時ファンアウト: 各メンバーに即時配信"""
        online_members = []
        offline_members = []

        for member_id in members:
            if member_id == sender_id:
                continue
            presence = await self._redis.get(f"presence:{member_id}")
            if presence and presence.decode() == "online":
                online_members.append(member_id)
            else:
                offline_members.append(member_id)

        # オンラインメンバーに即時配信
        delivery_tasks = [
            deliver_message(member_id, message)
            for member_id in online_members
        ]
        await asyncio.gather(*delivery_tasks, return_exceptions=True)

        # オフラインメンバーにはプッシュ通知
        if offline_members:
            await self._push.batch_send(
                user_ids=offline_members,
                title=f"新着メッセージ ({message['group_id']})",
                body=message["content"][:100],
            )
            # オフラインキューにも保存
            for member_id in offline_members:
                await enqueue_offline_message(member_id, message)

    async def _fanout_push_async(
        self,
        sender_id: str,
        members: list[str],
        message: dict,
    ):
        """非同期ファンアウト: Kafka ワーカーで配信"""
        # Kafka にファンアウトジョブを発行
        await self._kafka.send("fanout-jobs", {
            "type": "group_fanout",
            "sender_id": sender_id,
            "members": members,
            "message": message,
        })

    async def _fanout_pull(
        self,
        sender_id: str,
        group_id: str,
        message: dict,
    ):
        """読み取り時ファンアウト: メンバーがアクセス時に取得"""
        # グループのタイムラインにメッセージを追加
        await self._redis.zadd(
            f"group_timeline:{group_id}",
            {json.dumps(message): message["timestamp"]},
        )
        # タイムラインのサイズ制限（最新 10,000 件）
        await self._redis.zremrangebyrank(
            f"group_timeline:{group_id}", 0, -10001
        )

    async def get_group_members(self, group_id: str) -> list[str]:
        """グループメンバーリストを取得する（キャッシュ付き）"""
        cached = await self._redis.smembers(f"group:{group_id}:members")
        if cached:
            return [m.decode() for m in cached]

        # DB からロードしてキャッシュ
        members = await self._db.fetch_all(
            "SELECT user_id FROM group_members WHERE group_id = :gid",
            {"gid": group_id}
        )
        member_ids = [m["user_id"] for m in members]
        if member_ids:
            await self._redis.sadd(
                f"group:{group_id}:members", *member_ids
            )
            await self._redis.expire(f"group:{group_id}:members", 3600)
        return member_ids
```

### 3.3 メッセージ順序保証

```python
# infrastructure/id_generator/snowflake.py
"""
メッセージ ID の生成: Snowflake 方式

要件:
  1. グローバルにユニーク
  2. 時系列順（ID の大小 = 時間の前後）
  3. 分散環境で衝突なし
  4. 高スループット（1マシンで 400万+ ID/秒）

構造: 64bit
  [1bit 符号][41bit タイムスタンプ][10bit マシンID][12bit シーケンス]

  タイムスタンプ: ミリ秒精度、約69年分
  マシンID: 1024台まで
  シーケンス: 1ミリ秒あたり 4096 ID
"""
import time
import threading


class MessageIdGenerator:
    EPOCH = 1704067200000  # 2024-01-01 00:00:00 UTC

    def __init__(self, machine_id: int):
        if machine_id < 0 or machine_id >= 1024:
            raise ValueError("machine_id は 0-1023 の範囲")
        self._machine_id = machine_id
        self._sequence = 0
        self._last_timestamp = -1
        self._lock = threading.Lock()

    def generate(self) -> int:
        with self._lock:
            timestamp = int(time.time() * 1000) - self.EPOCH

            if timestamp == self._last_timestamp:
                self._sequence = (self._sequence + 1) & 0xFFF  # 12bit
                if self._sequence == 0:
                    # 同一ミリ秒でシーケンス上限 → 次のミリ秒まで待機
                    while timestamp <= self._last_timestamp:
                        timestamp = int(time.time() * 1000) - self.EPOCH
            else:
                self._sequence = 0

            self._last_timestamp = timestamp

            return (
                (timestamp << 22)
                | (self._machine_id << 12)
                | self._sequence
            )

    def extract_timestamp(self, message_id: int) -> int:
        """メッセージ ID からタイムスタンプを抽出"""
        return (message_id >> 22) + self.EPOCH
```

### 3.4 既読管理

```python
# application/services/read_receipt_service.py
"""
既読管理の設計判断:

  方式1: 各メッセージに既読フラグ
    → メッセージ数 × ユーザー数の既読レコード
    → 20億メッセージ × 2人 = 40億レコード/日 → 非現実的

  方式2: 「最後に読んだメッセージ ID」のみ保持（採用）
    → チャットルーム × ユーザー数のレコードのみ
    → msg_id が Snowflake ID のため、大小比較で未読判定が可能
    → ストレージ効率が圧倒的に良い
"""
import logging

logger = logging.getLogger(__name__)


class ReadReceiptService:
    """既読状態を効率的に管理する

    各チャットルームで「最後に読んだメッセージ ID」のみを保持。
    Snowflake ID の単調増加性を利用して、
    last_read_msg_id 以降のメッセージ = 未読 と判定する。
    """

    def __init__(self, redis_client, db):
        self._redis = redis_client
        self._db = db

    async def mark_read(
        self,
        user_id: str,
        chat_id: str,
        last_read_msg_id: str,
    ) -> None:
        """指定メッセージまでを既読にする"""
        key = f"read:{chat_id}:{user_id}"
        current = await self._redis.get(key)

        # より新しいメッセージ ID の場合のみ更新（冪等性）
        if current is None or last_read_msg_id > current.decode():
            await self._redis.set(key, last_read_msg_id)

            # DB にも非同期で永続化（Redis 障害時の復旧用）
            await self._db.execute(
                """
                INSERT INTO read_receipts (user_id, chat_id, last_read_msg_id)
                VALUES (:uid, :cid, :mid)
                ON CONFLICT (user_id, chat_id) DO UPDATE
                SET last_read_msg_id = :mid
                WHERE read_receipts.last_read_msg_id < :mid
                """,
                {"uid": user_id, "cid": chat_id, "mid": last_read_msg_id}
            )

            # 相手に既読通知を送信
            await self._notify_read_receipt(
                chat_id, user_id, last_read_msg_id
            )

    async def get_unread_count(self, user_id: str, chat_id: str) -> int:
        """未読メッセージ数を取得する"""
        last_read = await self._redis.get(f"read:{chat_id}:{user_id}")

        if last_read is None:
            # 全メッセージが未読
            return await self._count_all_messages(chat_id)

        return await self._count_messages_after(
            chat_id, last_read.decode()
        )

    async def get_unread_counts_batch(
        self, user_id: str, chat_ids: list[str]
    ) -> dict[str, int]:
        """複数チャットの未読数を一括取得（チャット一覧画面用）"""
        result = {}
        pipe = self._redis.pipeline()
        for chat_id in chat_ids:
            pipe.get(f"read:{chat_id}:{user_id}")
        last_reads = await pipe.execute()

        for chat_id, last_read in zip(chat_ids, last_reads):
            if last_read is None:
                result[chat_id] = await self._count_all_messages(chat_id)
            else:
                result[chat_id] = await self._count_messages_after(
                    chat_id, last_read.decode()
                )
        return result

    async def _notify_read_receipt(
        self, chat_id: str, reader_id: str, last_read_msg_id: str
    ):
        """既読通知を相手に送信"""
        # チャットの相手ユーザーを取得
        participants = await self._redis.smembers(
            f"chat:{chat_id}:participants"
        )
        for participant in participants:
            pid = participant.decode()
            if pid != reader_id:
                await deliver_message(pid, {
                    "type": "read_receipt",
                    "chat_id": chat_id,
                    "reader_id": reader_id,
                    "last_read_msg_id": last_read_msg_id,
                })

    async def _count_all_messages(self, chat_id: str) -> int:
        """チャットの総メッセージ数"""
        result = await self._db.fetch_one(
            "SELECT COUNT(*) as cnt FROM messages WHERE chat_id = :cid",
            {"cid": chat_id}
        )
        return result["cnt"] if result else 0

    async def _count_messages_after(
        self, chat_id: str, msg_id: str
    ) -> int:
        """指定メッセージ以降のメッセージ数"""
        result = await self._db.fetch_one(
            "SELECT COUNT(*) as cnt FROM messages "
            "WHERE chat_id = :cid AND id > :mid",
            {"cid": chat_id, "mid": msg_id}
        )
        return result["cnt"] if result else 0
```

### 3.5 メッセージ永続化（Cassandra）

```python
# infrastructure/repositories/cassandra_message_repo.py
"""
Cassandra をメッセージストアに選択した理由:

  1. 書き込み最適化: LSM-tree ベースで高い書き込みスループット
  2. 水平スケール: ノード追加で線形にスケール
  3. 高可用性: レプリケーションファクター 3 で耐障害性
  4. 時系列データに最適: パーティションキー + クラスタリングキー

  テーブル設計:
  PRIMARY KEY ((chat_id), message_id)
  → chat_id がパーティションキー（同一チャットのメッセージを同一ノードに配置）
  → message_id がクラスタリングキー（Snowflake ID で時系列順にソート）
"""


class CassandraMessageRepository:
    """Cassandra ベースのメッセージリポジトリ"""

    CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS messages (
        chat_id     TEXT,
        message_id  BIGINT,
        sender_id   TEXT,
        content     TEXT,
        content_type TEXT,
        created_at  TIMESTAMP,
        PRIMARY KEY ((chat_id), message_id)
    ) WITH CLUSTERING ORDER BY (message_id DESC)
      AND compaction = {'class': 'TimeWindowCompactionStrategy',
                        'compaction_window_size': 1,
                        'compaction_window_unit': 'DAYS'};
    """

    def __init__(self, session):
        self._session = session

    async def save_message(self, message: dict) -> None:
        """メッセージを保存"""
        await self._session.execute_async(
            """
            INSERT INTO messages
                (chat_id, message_id, sender_id, content, content_type, created_at)
            VALUES (%s, %s, %s, %s, %s, toTimestamp(now()))
            """,
            (
                message.get("to") or message.get("group_id"),
                int(message["id"]),
                message["from"],
                message["content"],
                message.get("content_type", "text"),
            )
        )

    async def get_messages(
        self,
        chat_id: str,
        before_id: int | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """メッセージ履歴を取得（ページネーション対応）

        before_id: このメッセージ ID より前のメッセージを取得
        limit: 取得件数（デフォルト50）
        """
        if before_id:
            rows = await self._session.execute_async(
                """
                SELECT * FROM messages
                WHERE chat_id = %s AND message_id < %s
                ORDER BY message_id DESC
                LIMIT %s
                """,
                (chat_id, before_id, limit)
            )
        else:
            rows = await self._session.execute_async(
                """
                SELECT * FROM messages
                WHERE chat_id = %s
                ORDER BY message_id DESC
                LIMIT %s
                """,
                (chat_id, limit)
            )
        return [dict(row) for row in rows]

    async def get_message_count(self, chat_id: str) -> int:
        """メッセージ総数を取得（カウンターテーブル使用推奨）"""
        rows = await self._session.execute_async(
            "SELECT COUNT(*) as cnt FROM messages WHERE chat_id = %s",
            (chat_id,)
        )
        return rows[0]["cnt"] if rows else 0
```

---

## 4. テスト

### 4.1 WebSocket Gateway のテスト

```python
# tests/test_websocket_gateway.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch


class FakeRedis:
    def __init__(self):
        self._data = {}
        self._hash = {}
        self._pubsub_messages = []

    async def hset(self, name, key, value):
        self._hash.setdefault(name, {})[key] = value

    async def hget(self, name, key):
        return self._hash.get(name, {}).get(key)

    async def hdel(self, name, key):
        self._hash.get(name, {}).pop(key, None)

    async def set(self, key, value, ex=None):
        self._data[key] = value

    async def get(self, key):
        value = self._data.get(key)
        return value.encode() if isinstance(value, str) else value

    async def publish(self, channel, message):
        self._pubsub_messages.append((channel, message))

    async def rpush(self, key, *values):
        self._data.setdefault(key, []).extend(values)

    async def lrange(self, key, start, stop):
        return self._data.get(key, [])

    async def delete(self, key):
        self._data.pop(key, None)


class FakeWebSocket:
    def __init__(self):
        self.sent_messages = []
        self.closed = False

    async def send_json(self, data):
        self.sent_messages.append(data)

    async def accept(self):
        pass

    async def close(self, code=1000, reason=""):
        self.closed = True


class TestDeliverMessage:
    """メッセージ配信のテスト"""

    @pytest.fixture
    def redis(self):
        return FakeRedis()

    @pytest.mark.asyncio
    async def test_同一Gateway_直接配信(self, redis):
        """受信者が同じ Gateway に接続している場合、直接配信"""
        ws = FakeWebSocket()
        local_connections["user_b"] = ws
        await redis.hset("ws:connections", "user_b", GATEWAY_ID)

        message = {"type": "chat", "content": "Hello!"}
        await deliver_message("user_b", message)

        assert len(ws.sent_messages) == 1
        assert ws.sent_messages[0]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_別Gateway_PubSub転送(self, redis):
        """受信者が別の Gateway に接続している場合、Pub/Sub で転送"""
        await redis.hset("ws:connections", "user_c", "gw-other")

        message = {"type": "chat", "content": "Hello!"}
        await deliver_message("user_c", message)

        assert len(redis._pubsub_messages) == 1
        channel, _ = redis._pubsub_messages[0]
        assert channel == "gw:gw-other"

    @pytest.mark.asyncio
    async def test_オフライン_キュー保存(self, redis):
        """受信者がオフラインの場合、キューに保存"""
        # 接続マップにユーザーがいない = オフライン
        message = {"type": "chat", "content": "Hello!"}
        await deliver_message("user_offline", message)

        # オフラインキューにメッセージが保存される
        queue = redis._data.get("offline:user_offline", [])
        assert len(queue) == 1


class TestReadReceipt:
    """既読管理のテスト"""

    @pytest.mark.asyncio
    async def test_既読マーク(self):
        redis = FakeRedis()
        db = AsyncMock()
        service = ReadReceiptService(redis, db)

        await service.mark_read("user_a", "chat_001", "msg_100")

        last_read = await redis.get("read:chat_001:user_a")
        assert last_read.decode() == "msg_100"

    @pytest.mark.asyncio
    async def test_古いメッセージIDでは更新されない(self):
        redis = FakeRedis()
        db = AsyncMock()
        service = ReadReceiptService(redis, db)

        await service.mark_read("user_a", "chat_001", "msg_100")
        await service.mark_read("user_a", "chat_001", "msg_050")

        last_read = await redis.get("read:chat_001:user_a")
        assert last_read.decode() == "msg_100"  # 古い ID では更新されない


class TestMessageIdGenerator:
    """Snowflake ID ジェネレーターのテスト"""

    def test_ユニーク性(self):
        gen = MessageIdGenerator(machine_id=1)
        ids = [gen.generate() for _ in range(10000)]
        assert len(set(ids)) == 10000  # 全てユニーク

    def test_単調増加(self):
        gen = MessageIdGenerator(machine_id=1)
        ids = [gen.generate() for _ in range(1000)]
        assert ids == sorted(ids)  # 単調増加

    def test_異なるマシンIDでユニーク(self):
        gen1 = MessageIdGenerator(machine_id=1)
        gen2 = MessageIdGenerator(machine_id=2)
        ids1 = {gen1.generate() for _ in range(1000)}
        ids2 = {gen2.generate() for _ in range(1000)}
        assert len(ids1 & ids2) == 0  # 重複なし
```

---

## 5. ファンアウト戦略の比較

### 5.1 書き込み時 vs 読み取り時ファンアウト

| 観点 | 書き込み時ファンアウト (Push) | 読み取り時ファンアウト (Pull) |
|------|------------------------------|------------------------------|
| 方式 | 送信時に全受信者のインボックスに書き込む | 受信者がアクセス時にメッセージを取得 |
| 配信遅延 | 低い（即時配信） | 高い（取得時に集約が必要） |
| 書き込みコスト | 高い（N人に書き込み） | 低い（1箇所に書き込み） |
| 読み取りコスト | 低い（自分のインボックスを読むだけ） | 高い（複数ソースから集約） |
| ストレージ | 高い（N倍のコピー） | 低い（1コピー） |
| 適している場面 | 小〜中規模グループ（<50人） | 大規模チャンネル（>500人） |

### 5.2 ハイブリッドアプローチ

```
=== グループサイズ別のファンアウト戦略 ===

小グループ (< 50人): 書き込み時ファンアウト (Push)
  → 即時配信が重要、遅延が許容できない
  → 50人 × 40メッセージ/日 = 2000 書き込み/日（許容範囲）

中グループ (50-500人): 非同期ファンアウト
  → Kafka ワーカーで非同期配信
  → 送信者には即座に ACK を返す
  → 配信遅延は数百ms（許容範囲）

大チャンネル (> 500人): 読み取り時ファンアウト (Pull)
  → 書き込みコストが膨大（500人 × 配信 = 膨大な書き込み）
  → チャンネルのタイムラインに1回だけ書き込み
  → メンバーはアクセス時にタイムラインを読む

VIP ユーザー: 常に書き込み時ファンアウト
  → 体験を最優先
  → サービスレベルに応じた差別化
```

---

## 6. 比較表

### 6.1 通信プロトコル

| プロトコル | 方向 | レイテンシ | オーバーヘッド | 適用場面 |
|-----------|:----:|:---------:|:-----------:|---------|
| WebSocket | 双方向 | 最低 | 低 | リアルタイムチャット（推奨） |
| SSE (Server-Sent Events) | サーバー→クライアント | 低 | 中 | 通知・フィード更新 |
| Long Polling | 擬似双方向 | 中 | 高 | WebSocket 非対応環境 |
| HTTP Polling | 擬似双方向 | 高 | 最高 | レガシー環境のみ |

### 6.2 メッセージストア

| 特性 | Cassandra | DynamoDB | MongoDB |
|-----|:---------:|:--------:|:------:|
| 書き込みスループット | 非常に高い | 高い | 中程度 |
| 読み取りパターン | パーティションキー検索 | ハッシュキー検索 | 柔軟なクエリ |
| スケーラビリティ | 自動（ノード追加） | 自動（マネージド） | 手動シャーディング |
| 運用コスト | 高い（自前管理） | 低い（マネージド） | 中程度 |
| 適用場面 | 超大規模チャット | AWS エコシステム | 中小規模 |

---

## 7. アンチパターン

### アンチパターン 1: 全メッセージをポーリング

```
WHY: HTTP ポーリングでは大半のリクエストが空レスポンスとなり、
     サーバーリソースと帯域を浪費する。

NG:
  クライアント: GET /messages?since=xxx (毎秒ポーリング)
  → 50M DAU × 1 req/sec = 50M QPS (大半が無駄)
  → レイテンシ: 最大1秒の遅延
  → バッテリー消費が激しい

OK: WebSocket で常時接続し、サーバーからプッシュする
  → 接続数は多いが、無駄なリクエストが激減
  → リアルタイム配信（遅延 < 100ms）
  → ロングポーリングは WebSocket 非対応環境の代替
```

### アンチパターン 2: グループの全メンバーに同期配信

```
WHY: 同期配信ではメンバー数に比例して遅延が増加し、
     1人でもエラーがあると全体が遅延する。

NG:
  500人グループに同期的に500回の WebSocket 送信を待つ
  → 1配信 5ms × 500人 = 2.5秒のブロッキング
  → 1人の接続エラーが全体を遅延させる

OK: 非同期ファンアウト
  1. メッセージを Kafka に発行（1回の書き込み）
  2. ワーカーが非同期で各メンバーに配信
  3. 送信者には即座に ACK を返す
  4. 配信失敗はリトライキューで処理
```

### アンチパターン 3: 全メッセージの既読を個別記録

```
WHY: メッセージ数 × ユーザー数のレコードが必要となり、
     ストレージと書き込みが爆発する。

NG:
  20億メッセージ/日 × 2人 = 40億既読レコード/日
  → ストレージが膨大
  → 既読の書き込みがメッセージの書き込みより多い

OK: 「最後に読んだメッセージ ID」のみ保持
  → チャットルーム × ユーザー数 のレコードのみ
  → Snowflake ID の大小比較で未読判定
```

### アンチパターン 4: WebSocket 接続を単一サーバーで管理

```
WHY: 単一サーバーでは同時接続数に限界があり、
     そのサーバー障害で全ユーザーが切断される。

NG:
  500万接続 → 1台のサーバー → メモリ 47GB、CPU 過負荷

OK: WebSocket Gateway をクラスタ化
  → Redis で接続マップを管理（user_id → gateway_id）
  → Gateway 間は Redis Pub/Sub でメッセージ転送
  → Gateway 障害時はクライアントが別の Gateway に再接続
```

### アンチパターン 5: メッセージ順序をタイムスタンプで保証

```
WHY: 分散環境ではサーバー間の時計が微妙にずれるため、
     タイムスタンプだけでは正確な順序を保証できない。

NG:
  Server A のメッセージ: timestamp = 1000
  Server B のメッセージ: timestamp = 999
  → 実際には Server B が先だが、順序が逆転する

OK: Snowflake ID で順序を保証
  → タイムスタンプ + マシンID + シーケンス番号
  → 同一マシン内ではシーケンス番号で厳密に順序保証
  → 異なるマシン間でもミリ秒精度で概ね正しい順序
```

---

## 8. 演習問題

### 演習1: 基本 -- WebSocket チャットサーバー（30分）

**課題**: 簡易版の WebSocket チャットサーバーを実装

要件:
1. WebSocket 接続のハンドリング
2. 接続中のユーザーリスト管理
3. メッセージのブロードキャスト（全ユーザーに送信）
4. 切断時のクリーンアップ

**期待する出力**:
```python
# User A が接続
# User B が接続
# User A がメッセージ送信 → User B に配信
# User B が切断 → User A に通知
```

### 演習2: 応用 -- 既読管理の実装（60分）

**課題**: Snowflake ID ベースの既読管理システムを実装

要件:
1. ReadReceiptService の完全な実装
2. 未読数の取得（単一チャット + バッチ取得）
3. 既読通知の配信
4. テストを5ケース以上

**期待する出力**:
```python
service = ReadReceiptService(redis, db)

# 既読マーク
await service.mark_read("user_a", "chat_001", "msg_100")

# 未読数の取得
count = await service.get_unread_count("user_a", "chat_001")
assert count == 0  # msg_100 まで既読

# バッチ取得
counts = await service.get_unread_counts_batch(
    "user_a", ["chat_001", "chat_002", "chat_003"]
)
```

### 演習3: 発展 -- グループチャットのファンアウト（90分）

**課題**: ハイブリッドファンアウト戦略を持つグループチャットサービスを実装

要件:
1. 小グループ（<50人）: 即時配信
2. 中グループ（50-500人）: 非同期配信（Kafka 経由）
3. 大チャンネル（>500人）: 読み取り時ファンアウト
4. オフラインメンバーへのプッシュ通知
5. 各戦略のパフォーマンステスト

**期待する出力**:
```python
# 小グループ: 即時配信
await service.send_group_message("user_a", "small_group", "Hello!")
# → 全メンバーに即座に配信される

# 大チャンネル: タイムラインに書き込み
await service.send_group_message("user_a", "large_channel", "Hello!")
# → タイムラインに1回だけ書き込み
# → メンバーはアクセス時に取得
```

---

## 9. FAQ

### Q1: WebSocket 接続が切れた場合の再接続戦略は？

**A:** 指数バックオフ + ジッター付きの再接続を実装する。

```python
import random
import asyncio


async def reconnect_with_backoff(
    websocket_url: str,
    max_retries: int = 10,
    max_backoff: float = 30.0,
):
    """指数バックオフ + ジッターによる再接続"""
    for attempt in range(max_retries):
        try:
            ws = await connect_websocket(websocket_url)
            # 再接続成功 → 切断中のメッセージを同期
            await sync_missed_messages(ws, last_received_msg_id)
            return ws
        except ConnectionError:
            # 指数バックオフ: 1, 2, 4, 8, 16, ... 秒
            backoff = min(2 ** attempt, max_backoff)
            # ジッター: 0〜backoff の間でランダム（Thundering Herd 対策）
            jitter = random.uniform(0, backoff)
            await asyncio.sleep(jitter)

    raise ConnectionError("最大リトライ回数を超過")
```

### Q2: メッセージの暗号化はどう実装しますか？

**A:** エンドツーエンド暗号化（E2EE）には Signal Protocol が業界標準。

```
Signal Protocol の概要:

  鍵交換: X3DH (Extended Triple Diffie-Hellman)
  → 初回メッセージ送信時に共有鍵を確立
  → サーバーに秘密鍵を預けない

  メッセージ暗号化: Double Ratchet Algorithm
  → メッセージごとに新しい暗号鍵を生成
  → 1つの鍵が漏洩しても過去・未来のメッセージは安全

  サーバーの役割:
  → 暗号化されたメッセージを中継するだけ
  → サーバー運営者もメッセージを読めない

  グループ E2EE:
  → Sender Key プロトコル
  → グループメンバー全員に共有鍵を配布
  → メンバー変更時に鍵をローテーション

  採用企業: WhatsApp, Signal, LINE（一部）
```

### Q3: メッセージの保存期間とストレージの管理方法は？

**A:** 階層型ストレージを採用する。

```
ホットストレージ (30日): Cassandra / DynamoDB
  → 最新メッセージへの高速アクセス
  → 1日 373GB × 30日 = 11.2 TB

ウォームストレージ (1年): 圧縮して S3 + メタデータ DB
  → 古いメッセージの検索・表示
  → 圧縮率 5:1 → 133TB/年 → 26.6 TB/年

コールドストレージ (5年+): S3 Glacier / アーカイブ
  → 法的要件での保持
  → アクセス頻度: ほぼゼロ

メッセージ検索:
  → Elasticsearch にインデックスを作成
  → メッセージ本文の全文検索
  → ホットストレージのメッセージのみインデックス（30日分）
```

### Q4: 数百万の同時 WebSocket 接続はどう管理する？

**A:** Gateway のクラスタ化と接続マップの分散管理。

```
1サーバーの同時接続数:
  → C10K 問題の現代版: epoll/kqueue で 10万+ 接続/サーバー
  → 1接続あたりメモリ: 約 10KB
  → 16GB RAM のサーバーで約 100万接続

500万接続の場合:
  → WebSocket Gateway: 最低 5 サーバー（余裕を見て 10 サーバー）
  → Redis: 接続マップ（500万エントリ ≈ 数百 MB）
  → ロードバランサー: Sticky Session or IP Hash

スケーリング戦略:
  → Gateway はステートレス（接続状態は Redis に）
  → 新しい Gateway を追加するだけでスケール
  → Gateway 障害時は自動的に他の Gateway に再接続
```

### Q5: 面接での進め方は？

**A:** 以下のフレームワークで回答する。

```
Step 1: 要件確認 (3-5分)
  → 1対1? グループ? 両方?
  → 想定DAU、同時接続数
  → メッセージのタイプ（テキスト? 画像? 動画?）
  → E2EE は必要?

Step 2: 高レベル設計 (10-15分)
  → アーキテクチャ図: Client → WS Gateway → Message Service → DB
  → WebSocket の選択理由（vs ポーリング、SSE）
  → Redis 接続マップの説明

Step 3: 深掘り (15-20分)
  → グループチャットのファンアウト戦略
  → オフライン対応（キュー + プッシュ通知）
  → 既読管理の効率的な実装
  → メッセージ順序保証（Snowflake ID）

Step 4: スケーラビリティ (5-10分)
  → 500万同時接続の管理方法
  → Gateway のスケーリング
  → メッセージストア（Cassandra）の選択理由
```

### Q6: タイピングインジケーターの実装は？

**A:** 頻度を制限した軽量な Pub/Sub で実装する。

```python
# クライアント側: 入力中は3秒ごとに typing イベントを送信
# サーバー側: typing イベントを受信者に転送

async def handle_typing_indicator(sender_id: str, data: dict):
    """タイピングインジケーターの処理

    設計判断:
    - DB に保存しない（揮発性の高いデータ）
    - レート制限: 3秒に1回まで
    - TTL: 5秒後に自動的に消える
    """
    recipient_id = data["to"]
    key = f"typing:{sender_id}:{recipient_id}"

    # レート制限: 3秒に1回
    if await redis_client.exists(key):
        return
    await redis_client.set(key, "1", ex=3)

    # 受信者に通知（永続化しない）
    await deliver_message(recipient_id, {
        "type": "typing",
        "from": sender_id,
        "expires_in": 5000,  # 5秒後に消える
    })
```

### Q7: メッセージの編集・削除はどう実装する？

**A:** ソフトデリートと更新イベントで実装する。

```
メッセージ編集:
  1. DB のメッセージレコードを更新（edited_at を記録）
  2. 相手に "message_edited" イベントを送信
  3. クライアント側で表示を更新（「編集済み」マーク）

メッセージ削除:
  1. DB のメッセージを論理削除（is_deleted = true）
  2. content を "このメッセージは削除されました" に置換
  3. 相手に "message_deleted" イベントを送信

注意:
  - E2EE 環境ではサーバーが元の内容を知らない
  - 既に配信されたメッセージの「完全な削除」は不可能
  - 「相手の端末からも削除」は相手の協力が必要（LINE の Unsend）
```

---

## まとめ

| 設計要素 | 選択 | 理由 |
|----------|------|------|
| 通信プロトコル | WebSocket | リアルタイム双方向通信、低レイテンシ |
| 接続管理 | Redis (接続マップ) | 高速ルックアップ、Gateway 間の橋渡し |
| メッセージ ID | Snowflake ID | 時系列順序保証 + 分散生成 |
| メッセージ DB | Cassandra | 書き込みスケーラビリティ、時系列最適化 |
| メッセージキュー | Kafka | 非同期ファンアウト、at-least-once 保証 |
| キャッシュ | Redis | プレゼンス、既読、メンバーリスト |
| プッシュ通知 | APNS / FCM | オフラインユーザー対応 |
| ファンアウト | ハイブリッド | グループサイズに応じた最適化 |

---

## 次に読むべきガイド

- [通知システム設計](./02-notification-system.md) — プッシュ通知の大規模配信設計
- [URL 短縮サービス設計](./00-url-shortener.md) — シンプルなシステム設計の基本
- [レート制限設計](./03-rate-limiter.md) — API 保護のためのレート制限
- [イベント駆動アーキテクチャ](../02-architecture/03-event-driven.md) — Pub/Sub パターンの詳細
- [メッセージキュー](../01-components/02-message-queue.md) — Kafka の詳細設計

---

## 参考文献

1. **System Design Interview: An Insider's Guide** — Alex Xu (2020) — Chapter 12: Design a Chat System
2. **The X3DH Key Agreement Protocol** — Marlinspike, M. & Perrin, T. (Signal Foundation, 2016) — E2EE の鍵交換プロトコル
3. **Cassandra: A Decentralized Structured Storage System** — Lakshman, A. & Malik, P. (ACM SIGOPS, 2010) — 分散ストレージの設計
4. **The WebSocket Protocol** — RFC 6455 (IETF, 2011) — WebSocket の仕様
5. **Scaling WhatsApp** — https://highscalability.com/ — WhatsApp のスケーリング実践
6. **Discord Engineering Blog** — https://discord.com/blog/how-discord-stores-billions-of-messages — 数十億メッセージの保存戦略

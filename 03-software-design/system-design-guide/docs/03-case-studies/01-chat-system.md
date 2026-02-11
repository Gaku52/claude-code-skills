# チャットシステム設計

> LINE / Slack / WhatsApp のようなリアルタイムチャットシステムをゼロから設計する。WebSocket 管理、メッセージ配信、オフライン対応、グループチャットの設計パターンを体系的に学ぶ。

---

## この章で学ぶこと

1. **リアルタイム通信基盤** — WebSocket 接続管理、プレゼンス管理、ハートビートの設計
2. **メッセージ配信** — 1対1 チャット、グループチャット、メッセージ順序保証の仕組み
3. **スケーラビリティと信頼性** — メッセージ永続化、オフライン配信、既読管理、プッシュ通知連携

---

## 1. 要件定義

### 1.1 機能要件

- 1対1 チャット（テキスト、画像、ファイル）
- グループチャット（最大 500 人）
- オンライン/オフラインのプレゼンス表示
- メッセージの既読管理
- メッセージ履歴の永続化と同期
- プッシュ通知（オフラインユーザー向け）

### 1.2 スケール見積もり

```
前提:
  - DAU (Daily Active Users): 50M
  - 1ユーザーあたり平均 40 メッセージ/日
  - 1日あたりメッセージ数: 50M * 40 = 2B
  - メッセージ QPS: 2B / 86400 ≈ 23,000 QPS (ピーク ≈ 50,000)
  - 同時 WebSocket 接続: DAU の 10% = 5M 同時接続
  - 1メッセージ平均サイズ: 200 bytes
  - 1日のストレージ: 2B * 200 = 400 GB/日
```

---

## 2. 高レベルアーキテクチャ

### 2.1 全体構成

```
+----------+          +-------------+          +----------+
| クライ    |  WS接続  | WebSocket   |          | クライ    |
| アント A  |<-------->| Gateway     |          | アント B  |
+----------+          | Cluster     |          +----------+
                      +-------------+               ^
                           |    ^                    |
                      メッセージ |    | ルーティング     | WS接続
                           v    |                    |
                      +-------------+          +-------------+
                      | メッセージ   |          | WebSocket   |
                      | サービス    |<-------->| Gateway     |
                      +-------------+          | Cluster     |
                       |    |    |             +-------------+
                       v    v    v
                +------+ +------+ +------+
                |メッセ | |ユーザ | |プレゼ |
                |ージDB| |ー DB | |ンスDB|
                +------+ +------+ +------+
```

### 2.2 接続管理アーキテクチャ

```
クライアント           WS Gateway           サービス層
+--------+           +---------+          +----------+
| User A | -- WS --> | GW-1    |          |          |
| User B | -- WS --> | GW-1    |          | Message  |
+--------+           +---------+          | Service  |
                          |               |          |
+--------+           +---------+          |          |
| User C | -- WS --> | GW-2    |--------->|          |
| User D | -- WS --> | GW-2    |          |          |
+--------+           +---------+          +----------+
                          |                    |
                     +---------+          +----------+
                     | Redis   |          | Kafka    |
                     | (接続   |          | (メッセ  |
                     |  マップ) |          |  ージQ)  |
                     +---------+          +----------+

Redis 接続マップ:
  user_A -> GW-1
  user_B -> GW-1
  user_C -> GW-2
  user_D -> GW-2
```

---

## 3. コア実装

### 3.1 WebSocket Gateway

```python
# コード例 1: WebSocket Gateway (FastAPI + Redis)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import redis.asyncio as redis
import json
import uuid

app = FastAPI()
redis_client = redis.Redis(host="localhost", port=6379)

# このゲートウェイインスタンスの ID
GATEWAY_ID = f"gw-{uuid.uuid4().hex[:8]}"

# ローカル接続マップ (user_id -> WebSocket)
local_connections: dict[str, WebSocket] = {}

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()

    # 接続を登録
    local_connections[user_id] = websocket
    await redis_client.hset("ws:connections", user_id, GATEWAY_ID)
    await redis_client.set(f"presence:{user_id}", "online", ex=300)

    try:
        while True:
            data = await websocket.receive_json()
            await handle_message(user_id, data)
    except WebSocketDisconnect:
        # 接続解除
        local_connections.pop(user_id, None)
        await redis_client.hdel("ws:connections", user_id)
        await redis_client.set(f"presence:{user_id}", "offline")

async def handle_message(sender_id: str, data: dict):
    """受信メッセージをルーティングする"""
    msg_type = data.get("type")

    if msg_type == "chat":
        recipient_id = data["to"]
        message = {
            "id": str(uuid.uuid4()),
            "from": sender_id,
            "to": recipient_id,
            "content": data["content"],
            "timestamp": data.get("timestamp"),
            "type": "chat",
        }

        # メッセージを永続化
        await persist_message(message)

        # 受信者に配信
        await deliver_message(recipient_id, message)

async def deliver_message(recipient_id: str, message: dict):
    """メッセージを受信者に配信する"""
    # 受信者の接続先ゲートウェイを確認
    gateway_id = await redis_client.hget("ws:connections", recipient_id)

    if gateway_id is None:
        # オフライン → プッシュ通知 + オフラインキューに保存
        await enqueue_offline_message(recipient_id, message)
        await send_push_notification(recipient_id, message)
        return

    if gateway_id.decode() == GATEWAY_ID:
        # 同じゲートウェイ → 直接送信
        ws = local_connections.get(recipient_id)
        if ws:
            await ws.send_json(message)
    else:
        # 別のゲートウェイ → Redis Pub/Sub で転送
        channel = f"gw:{gateway_id.decode()}"
        await redis_client.publish(channel, json.dumps(message))
```

### 3.2 グループチャットの配信

```python
# コード例 2: グループチャットのファンアウト戦略
class GroupChatService:
    def __init__(self, redis_client, db):
        self.redis = redis_client
        self.db = db

    async def send_group_message(self, sender_id: str, group_id: str,
                                  content: str):
        """グループメッセージを全メンバーに配信する"""
        message = {
            "id": str(uuid.uuid4()),
            "from": sender_id,
            "group_id": group_id,
            "content": content,
            "timestamp": int(time.time() * 1000),
        }

        # メッセージを永続化
        await self.db.execute(
            "INSERT INTO messages (id, group_id, sender_id, content, ts) "
            "VALUES (:id, :gid, :sid, :content, :ts)",
            {"id": message["id"], "gid": group_id,
             "sid": sender_id, "content": content,
             "ts": message["timestamp"]}
        )

        # グループメンバーを取得
        members = await self.get_group_members(group_id)

        # 各メンバーに配信 (書き込みファンアウト)
        online_members = []
        offline_members = []

        for member_id in members:
            if member_id == sender_id:
                continue
            presence = await self.redis.get(f"presence:{member_id}")
            if presence and presence.decode() == "online":
                online_members.append(member_id)
            else:
                offline_members.append(member_id)

        # オンラインメンバーに即時配信
        for member_id in online_members:
            await deliver_message(member_id, message)

        # オフラインメンバーにはプッシュ通知
        if offline_members:
            await batch_push_notification(offline_members, message)

    async def get_group_members(self, group_id: str) -> list[str]:
        """グループメンバーリストを取得する（キャッシュ付き）"""
        cached = await self.redis.smembers(f"group:{group_id}:members")
        if cached:
            return [m.decode() for m in cached]
        # DB からロードしてキャッシュ
        members = await self.db.fetch_all(
            "SELECT user_id FROM group_members WHERE group_id = :gid",
            {"gid": group_id}
        )
        member_ids = [m["user_id"] for m in members]
        if member_ids:
            await self.redis.sadd(
                f"group:{group_id}:members", *member_ids
            )
            await self.redis.expire(f"group:{group_id}:members", 3600)
        return member_ids
```

### 3.3 メッセージ順序保証

```python
# コード例 3: Snowflake 風 ID 生成で時系列順序を保証する
import time
import threading

class MessageIdGenerator:
    """
    タイムスタンプベースのユニーク ID 生成器。
    構造: [41bit timestamp][10bit machine_id][12bit sequence]
    - 69年分のタイムスタンプ
    - 1024台のマシン
    - 1ms あたり 4096 メッセージ
    """
    EPOCH = 1704067200000  # 2024-01-01 00:00:00 UTC

    def __init__(self, machine_id: int):
        if machine_id < 0 or machine_id >= 1024:
            raise ValueError("machine_id must be 0-1023")
        self.machine_id = machine_id
        self.sequence = 0
        self.last_timestamp = -1
        self.lock = threading.Lock()

    def generate(self) -> int:
        with self.lock:
            timestamp = int(time.time() * 1000) - self.EPOCH

            if timestamp == self.last_timestamp:
                self.sequence = (self.sequence + 1) & 0xFFF  # 12bit
                if self.sequence == 0:
                    # 同一ミリ秒でシーケンス溢れ → 次のミリ秒まで待機
                    while timestamp <= self.last_timestamp:
                        timestamp = int(time.time() * 1000) - self.EPOCH
            else:
                self.sequence = 0

            self.last_timestamp = timestamp

            return (
                (timestamp << 22)
                | (self.machine_id << 12)
                | self.sequence
            )
```

### 3.4 既読管理

```python
# コード例 4: 効率的な既読管理
class ReadReceiptService:
    """
    既読状態を効率的に管理する。
    全メッセージの既読を個別に記録するのではなく、
    「最後に読んだメッセージID」のみを保持する。
    """

    def __init__(self, redis_client):
        self.redis = redis_client

    async def mark_read(self, user_id: str, chat_id: str,
                         last_read_msg_id: str):
        """指定メッセージまでを既読にする"""
        key = f"read:{chat_id}:{user_id}"
        current = await self.redis.get(key)

        # より新しいメッセージ ID の場合のみ更新
        if current is None or last_read_msg_id > current.decode():
            await self.redis.set(key, last_read_msg_id)

            # 相手に既読通知を送信
            await notify_read_receipt(chat_id, user_id, last_read_msg_id)

    async def get_unread_count(self, user_id: str, chat_id: str) -> int:
        """未読メッセージ数を取得する"""
        last_read = await self.redis.get(f"read:{chat_id}:{user_id}")
        if last_read is None:
            # 全メッセージが未読
            return await self.get_total_messages(chat_id)

        # last_read_msg_id 以降のメッセージ数をカウント
        return await self.count_messages_after(
            chat_id, last_read.decode()
        )
```

---

## 4. ファンアウト戦略の比較

### 4.1 書き込み時ファンアウト vs 読み取り時ファンアウト

| 観点 | 書き込み時ファンアウト (Push) | 読み取り時ファンアウト (Pull) |
|------|------------------------------|------------------------------|
| 方式 | 送信時に全受信者のインボックスに書き込む | 受信者がアクセス時にメッセージを取得 |
| 配信遅延 | 低い（即時配信） | 高い（取得時に集約が必要） |
| 書き込みコスト | 高い（N人に書き込み） | 低い（1箇所に書き込み） |
| 読み取りコスト | 低い（自分のインボックスを読むだけ） | 高い（複数ソースから集約） |
| 適している場面 | 小〜中規模グループ | 大規模チャンネル/フィード |

### 4.2 ハイブリッドアプローチ

```
小グループ (< 50人): 書き込み時ファンアウト
  → 即時配信、遅延が許容できないため

大チャンネル (> 500人): 読み取り時ファンアウト
  → 書き込みコストが膨大になるため

VIP ユーザー: 常に書き込み時ファンアウト
  → 体験を最優先
```

---

## 5. オフライン対応

### 5.1 オフラインメッセージキュー

```
ユーザーがオフラインの場合:

  メッセージ到着
       |
       v
  +----------------+     +----------------+
  | オフライン     | --> | プッシュ通知    |
  | メッセージキュー |     | サービス       |
  | (per user)     |     | (APNS/FCM)    |
  +----------------+     +----------------+
       |
       | ユーザーがオンラインに復帰
       v
  +----------------+
  | キュー内の     |
  | メッセージを   | --> WebSocket で順次配信
  | 一括配信       |
  +----------------+
```

---

## 6. アンチパターン

### アンチパターン 1: 「全メッセージをポーリング」

```
[誤り] HTTP ポーリングで定期的にメッセージを取得する

  クライアント: GET /messages?since=xxx  (毎秒ポーリング)
  サーバー: 空レスポンスが90%以上

問題点:
  - 50M DAU * 1 req/sec = 50M QPS （大半が無駄）
  - レイテンシ: 最大1秒の遅延
  - バッテリー消費が激しい

[正解] WebSocket で常時接続し、サーバーからプッシュする
  - 接続数は多いが、無駄なリクエストが激減
  - リアルタイム配信（遅延 < 100ms）
  - ロングポーリング (HTTP) は WebSocket が使えない場合の代替
```

### アンチパターン 2: 「グループの全メンバーに同期配信」

```
[誤り] 500人のグループにメッセージを送る際、
       同期的に500回のWebSocket送信を待つ

問題点:
  - 1配信に5ms × 500人 = 2.5秒のブロッキング
  - 1人でも接続エラーがあると全体が遅延
  - 送信者の体験が悪化

[正解] 非同期ファンアウト
  1. メッセージをKafkaに発行（1回の書き込み）
  2. ワーカーが非同期で各メンバーに配信
  3. 送信者には即座にACKを返す
  4. 配信失敗はリトライキューで処理
```

---

## 7. FAQ

### Q1: WebSocket 接続が切れた場合の再接続戦略は？

**A:** 指数バックオフ + ジッター付きの再接続を実装します。

1. 切断検知（ハートビート応答なし or onclose イベント）
2. 即座に再接続を試行
3. 失敗したら 1秒 → 2秒 → 4秒 → ... と間隔を広げる（最大 30 秒）
4. ランダムなジッター（0〜1秒）を加えて接続集中を防ぐ（Thundering Herd 対策）
5. 再接続成功時に、切断中のメッセージを同期（last_received_msg_id 以降を取得）

### Q2: メッセージの暗号化はどう実装しますか？

**A:** エンドツーエンド暗号化（E2EE）を実装する場合、Signal Protocol が業界標準です。

- **鍵交換**: X3DH (Extended Triple Diffie-Hellman)
- **メッセージ暗号化**: Double Ratchet Algorithm
- **サーバーの役割**: 暗号化されたメッセージを中継するだけ（復号不可）
- **グループ E2EE**: Sender Key プロトコル（全メンバーに共有鍵を配布）

### Q3: メッセージの保存期間とストレージの管理方法は？

**A:** 階層型ストレージを採用します。

- **ホットストレージ** (30日): Cassandra / DynamoDB（高速アクセス）
- **ウォームストレージ** (1年): 圧縮して S3 + メタデータ DB
- **コールドストレージ** (5年+): S3 Glacier / アーカイブストレージ

メッセージ検索には Elasticsearch を別途構築し、インデックスを作成します。

---

## 8. まとめ

| 設計要素 | 選択 | 理由 |
|----------|------|------|
| 通信プロトコル | WebSocket | リアルタイム双方向通信 |
| 接続管理 | Redis (接続マップ) | 高速ルックアップ |
| メッセージ ID | Snowflake ID | 時系列順序保証 + 分散生成 |
| メッセージ DB | Cassandra / DynamoDB | 書き込みスケーラビリティ |
| メッセージキュー | Kafka | 非同期ファンアウト |
| キャッシュ | Redis | プレゼンス、既読、メンバーリスト |
| プッシュ通知 | APNS / FCM | オフラインユーザー対応 |

---

## 次に読むべきガイド

- [通知システム設計](./02-notification-system.md) — プッシュ通知の大規模配信設計
- [URL 短縮サービス設計](./00-url-shortener.md) — シンプルなシステム設計の基本
- [リアルタイム音声](../../../ai-audio-generation/docs/03-development/02-real-time-audio.md) — WebRTC による音声通話の実装

---

## 参考文献

1. Xu, A. (2020). "System Design Interview: An Insider's Guide." *Chapter 12: Design a Chat System*. https://www.systemdesigninterview.com/
2. Marlinspike, M. & Perrin, T. (2016). "The X3DH Key Agreement Protocol." *Signal Foundation*. https://signal.org/docs/specifications/x3dh/
3. Lakshman, A. & Malik, P. (2010). "Cassandra: A Decentralized Structured Storage System." *ACM SIGOPS Operating Systems Review, 44*(2), 35-40. https://doi.org/10.1145/1773912.1773922

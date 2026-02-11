# 通知システム設計

> iOS プッシュ通知、Android FCM、SMS、メール、アプリ内通知を統合的に管理する大規模通知システムをゼロから設計する。配信保証、レート制限、ユーザー設定管理を含む。

---

## この章で学ぶこと

1. **通知チャネルの統合** — プッシュ通知、SMS、メール、アプリ内通知を統一 API で管理するアーキテクチャ
2. **信頼性と配信保証** — メッセージの重複排除、リトライ戦略、配信確認の仕組み
3. **ユーザー体験の最適化** — レート制限、優先度管理、ユーザー設定に基づくインテリジェント配信

---

## 1. 要件定義

### 1.1 機能要件

- 複数チャネル対応: iOS プッシュ (APNs)、Android プッシュ (FCM)、SMS、メール、アプリ内通知
- 通知テンプレート管理
- ユーザーごとの通知設定（チャネル別オン/オフ、時間帯指定）
- 通知履歴の保存と閲覧
- スケジュール配信

### 1.2 スケール見積もり

```
前提:
  - 登録ユーザー: 500M
  - DAU: 100M
  - 1日あたり通知数: 10B（プッシュ 5B + メール 3B + SMS 0.5B + アプリ内 1.5B）

プッシュ通知:
  5B / 86400 ≈ 57,870 QPS (ピーク ≈ 115,000 QPS)

メール:
  3B / 86400 ≈ 34,700 QPS (ピーク ≈ 70,000 QPS)

SMS:
  0.5B / 86400 ≈ 5,780 QPS

全チャネル合計ピーク: ≈ 200,000 QPS
```

---

## 2. 高レベルアーキテクチャ

### 2.1 全体構成

```
+----------+     +----------+     +-----------+     +----------+
| トリガー  | --> | 通知     | --> | メッセージ | --> | チャネル  |
| ソース    |     | サービス  |     | キュー     |     | アダプタ  |
+----------+     +----------+     +-----------+     +----------+
                      |                                  |
  - API呼び出し        |                            +----+----+
  - イベント駆動       v                            |    |    |
  - スケジュール  +----------+                +----+ +--+ +----+
  - ルールエンジン | 設定/    |                |APNs| |FCM| |SMTP|
                  | テンプレ |                +----+ +---+ +----+
                  | ート DB  |                  |      |      |
                  +----------+                  v      v      v
                                              iOS  Android  メール
```

### 2.2 詳細アーキテクチャ

```
+--------------------------------------------------------------+
|                     通知サービス (Notification Service)         |
+--------------------------------------------------------------+
|                                                                |
|  +----------+  +-----------+  +-----------+  +-----------+    |
|  | 受付API  |  | バリデー   |  | ユーザー   |  | テンプレ   |    |
|  |          |->| ション    |->| 設定確認   |->| ート展開   |    |
|  +----------+  +-----------+  +-----------+  +-----------+    |
|                                                    |           |
|                                              +-----------+    |
|                                              | 優先度/    |    |
|                                              | レート制限  |    |
|                                              +-----------+    |
|                                                    |           |
+--------------------------------------------------------------+
                                                     |
              +--------------------------------------+
              |              |              |
         +--------+    +--------+    +--------+
         | Push   |    | Email  |    | SMS    |
         | Queue  |    | Queue  |    | Queue  |
         +--------+    +--------+    +--------+
              |              |              |
         +--------+    +--------+    +--------+
         | Push   |    | Email  |    | SMS    |
         | Worker |    | Worker |    | Worker |
         +--------+    +--------+    +--------+
              |              |              |
         +--------+    +--------+    +--------+
         | APNs / |    | SMTP / |    | Twilio |
         | FCM    |    | SES    |    | etc.   |
         +--------+    +--------+    +--------+
```

---

## 3. コア実装

### 3.1 通知 API と配信パイプライン

```python
# コード例 1: 通知サービスのエントリポイント
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from enum import Enum
import uuid

app = FastAPI()

class Channel(str, Enum):
    PUSH = "push"
    EMAIL = "email"
    SMS = "sms"
    IN_APP = "in_app"

class NotificationRequest(BaseModel):
    user_ids: list[str]        # 送信先ユーザーID
    template_id: str           # テンプレートID
    channels: list[Channel]    # 配信チャネル
    data: dict                 # テンプレート変数
    priority: str = "normal"   # high / normal / low
    scheduled_at: str | None = None  # スケジュール配信

class NotificationService:
    def __init__(self, queue, user_service, template_service,
                 rate_limiter):
        self.queue = queue
        self.user_service = user_service
        self.template_service = template_service
        self.rate_limiter = rate_limiter

    async def send(self, request: NotificationRequest) -> str:
        batch_id = str(uuid.uuid4())

        for user_id in request.user_ids:
            # 1. ユーザー設定を確認
            prefs = await self.user_service.get_preferences(user_id)

            for channel in request.channels:
                # 2. ユーザーがこのチャネルを許可しているか
                if not prefs.is_channel_enabled(channel):
                    continue

                # 3. レート制限チェック
                if not await self.rate_limiter.allow(user_id, channel):
                    continue

                # 4. テンプレートを展開
                content = await self.template_service.render(
                    request.template_id,
                    channel,
                    request.data
                )

                # 5. メッセージキューに投入
                message = {
                    "id": str(uuid.uuid4()),
                    "batch_id": batch_id,
                    "user_id": user_id,
                    "channel": channel,
                    "content": content,
                    "priority": request.priority,
                }
                await self.queue.publish(
                    topic=f"notifications.{channel.value}",
                    message=message,
                    priority=request.priority
                )

        return batch_id

@app.post("/api/v1/notifications")
async def create_notification(request: NotificationRequest):
    service = NotificationService(...)
    batch_id = await service.send(request)
    return {"batch_id": batch_id, "status": "queued"}
```

### 3.2 チャネルアダプタ

```python
# コード例 2: プッシュ通知ワーカー (APNs / FCM)
import asyncio
from abc import ABC, abstractmethod

class PushProvider(ABC):
    @abstractmethod
    async def send(self, device_token: str, title: str,
                   body: str, data: dict) -> bool:
        pass

class APNsProvider(PushProvider):
    """Apple Push Notification service"""
    async def send(self, device_token, title, body, data):
        payload = {
            "aps": {
                "alert": {"title": title, "body": body},
                "badge": data.get("badge", 1),
                "sound": "default",
            },
            "custom_data": data,
        }
        # APNs HTTP/2 API に送信
        response = await self.client.post(
            f"https://api.push.apple.com/3/device/{device_token}",
            json=payload,
            headers={"authorization": f"bearer {self.jwt_token}"}
        )
        return response.status_code == 200

class FCMProvider(PushProvider):
    """Firebase Cloud Messaging"""
    async def send(self, device_token, title, body, data):
        message = {
            "message": {
                "token": device_token,
                "notification": {"title": title, "body": body},
                "data": {k: str(v) for k, v in data.items()},
                "android": {"priority": "high"},
            }
        }
        response = await self.client.post(
            "https://fcm.googleapis.com/v1/projects/my-project/messages:send",
            json=message,
            headers={"authorization": f"Bearer {self.access_token}"}
        )
        return response.status_code == 200

class PushNotificationWorker:
    """Kafka からメッセージを消費してプッシュ通知を送信する"""

    def __init__(self):
        self.providers = {
            "ios": APNsProvider(),
            "android": FCMProvider(),
        }

    async def process(self, message: dict):
        user_id = message["user_id"]
        content = message["content"]

        # ユーザーのデバイストークンを取得
        devices = await get_user_devices(user_id)

        for device in devices:
            provider = self.providers[device["platform"]]
            success = await provider.send(
                device_token=device["token"],
                title=content["title"],
                body=content["body"],
                data=content.get("data", {}),
            )

            if not success:
                # リトライキューに投入
                await enqueue_retry(message, device, retry_count=1)
```

### 3.3 レート制限

```python
# コード例 3: スライディングウィンドウによるレート制限
import time
import redis.asyncio as redis

class NotificationRateLimiter:
    """
    ユーザーあたり・チャネルあたりの通知レート制限。
    ユーザー体験を保護し、スパム的な通知を防止する。
    """

    # デフォルトのレート制限
    LIMITS = {
        "push":   {"per_hour": 10, "per_day": 50},
        "email":  {"per_hour": 3,  "per_day": 10},
        "sms":    {"per_hour": 2,  "per_day": 5},
        "in_app": {"per_hour": 20, "per_day": 100},
    }

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def allow(self, user_id: str, channel: str) -> bool:
        """通知を送信してよいか判定する"""
        limits = self.LIMITS.get(channel, {"per_hour": 10, "per_day": 50})
        now = time.time()

        # 1時間あたりのチェック
        hour_key = f"ratelimit:{user_id}:{channel}:hour"
        hour_count = await self._sliding_window_count(
            hour_key, now, window_seconds=3600
        )
        if hour_count >= limits["per_hour"]:
            return False

        # 1日あたりのチェック
        day_key = f"ratelimit:{user_id}:{channel}:day"
        day_count = await self._sliding_window_count(
            day_key, now, window_seconds=86400
        )
        if day_count >= limits["per_day"]:
            return False

        # カウントを記録
        pipe = self.redis.pipeline()
        pipe.zadd(hour_key, {str(now): now})
        pipe.zadd(day_key, {str(now): now})
        pipe.expire(hour_key, 3600)
        pipe.expire(day_key, 86400)
        await pipe.execute()

        return True

    async def _sliding_window_count(self, key: str, now: float,
                                     window_seconds: int) -> int:
        """スライディングウィンドウ内のカウントを取得する"""
        window_start = now - window_seconds
        # 古いエントリを削除
        await self.redis.zremrangebyscore(key, 0, window_start)
        # 現在のカウントを取得
        return await self.redis.zcard(key)
```

### 3.4 重複排除

```python
# コード例 4: べき等性キーによる重複通知の防止
class DeduplicationService:
    """
    同一通知の重複送信を防止する。
    イベント駆動アーキテクチャでは、同じイベントが
    複数回処理される可能性があるため必須。
    """

    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 3600 * 24  # 24時間

    async def is_duplicate(self, notification_id: str) -> bool:
        """この通知が既に処理済みかどうか確認する"""
        key = f"dedup:{notification_id}"
        # SETNX (SET if Not eXists) でアトミックにチェック
        result = await self.redis.set(key, "1", nx=True, ex=self.ttl)
        return result is None  # None = 既に存在 = 重複

    async def mark_sent(self, notification_id: str, channel: str):
        """通知の送信完了を記録する"""
        key = f"sent:{notification_id}:{channel}"
        await self.redis.set(key, "1", ex=self.ttl)
```

### 3.5 テンプレートエンジン

```python
# コード例 5: 多言語・多チャネル対応テンプレート
from jinja2 import Environment, DictLoader

class NotificationTemplateService:
    TEMPLATES = {
        "order_shipped": {
            "push": {
                "ja": {"title": "発送完了", "body": "ご注文 #{{order_id}} が発送されました。"},
                "en": {"title": "Order Shipped", "body": "Your order #{{order_id}} has been shipped."},
            },
            "email": {
                "ja": {
                    "subject": "【発送通知】ご注文 #{{order_id}}",
                    "body": "{{user_name}} 様\n\nご注文 #{{order_id}} を発送いたしました。\n配送番号: {{tracking_number}}"
                },
            },
        },
    }

    def __init__(self):
        self.env = Environment(loader=DictLoader({}))

    async def render(self, template_id: str, channel: str,
                      data: dict, locale: str = "ja") -> dict:
        templates = self.TEMPLATES.get(template_id, {})
        channel_templates = templates.get(channel, {})
        localized = channel_templates.get(locale, channel_templates.get("en", {}))

        rendered = {}
        for key, template_str in localized.items():
            template = self.env.from_string(template_str)
            rendered[key] = template.render(**data)

        return rendered
```

---

## 4. 信頼性設計

### 4.1 リトライ戦略

```
リトライポリシー:

  初回送信 → 失敗
       |
       v (1秒後)
  リトライ1 → 失敗
       |
       v (4秒後)
  リトライ2 → 失敗
       |
       v (16秒後)
  リトライ3 → 失敗
       |
       v
  Dead Letter Queue (DLQ) に移動
       |
       v
  アラート + 手動対応

指数バックオフ: delay = min(base * 2^attempt, max_delay)
チャネル別の最大リトライ:
  Push: 3回  (デバイスがオフラインの可能性)
  Email: 5回 (SMTPサーバーの一時障害)
  SMS: 2回   (コストが高いため)
```

---

## 5. 通知チャネルの比較

| チャネル | 到達率 | 遅延 | コスト | 文字数制限 | ユースケース |
|----------|--------|------|--------|-----------|-------------|
| iOS Push (APNs) | 高 | < 1秒 | 無料 | 4KB | リアルタイムアラート |
| Android Push (FCM) | 高 | < 1秒 | 無料 | 4KB | リアルタイムアラート |
| Email | 中 | 秒〜分 | 低 | 無制限 | 詳細な通知、レポート |
| SMS | 極高 | < 3秒 | 高 | 160文字 | 重要アラート、2FA |
| アプリ内 | 極高 (アクティブ時) | 即時 | 無料 | 無制限 | UX誘導、プロモーション |
| Web Push | 中 | < 1秒 | 無料 | 制限あり | ブラウザ利用者向け |

---

## 6. アンチパターン

### アンチパターン 1: 「通知の絨毯爆撃」

```
[誤り] ユーザーのアクションごとに即座に通知を送る

  10:00 「Aさんがいいねしました」
  10:01 「Bさんがいいねしました」
  10:02 「Cさんがいいねしました」
  ... (ユーザーは通知をオフにする)

[正解] インテリジェントなバッチングと集約を行う
  - 時間ウィンドウ内の同種通知を集約
    → 「Aさん、Bさん、他3人がいいねしました」
  - レート制限で1時間あたりの上限を設定
  - ユーザーの活動時間帯に合わせて配信
  - 重要度に応じた配信頻度の調整
```

### アンチパターン 2: 「単一キューで全チャネル処理」

```
[誤り] 1つのメッセージキューで Push / Email / SMS を全て処理する

問題点:
  - SMS の遅延（外部API待ち）がPush通知をブロック
  - チャネルごとにスケーリング要件が異なる
  - 障害の影響範囲が全チャネルに波及

[正解] チャネルごとに独立したキューとワーカーを用意する
  - Push Queue → Push Workers (高スループット)
  - Email Queue → Email Workers (バッチ最適化)
  - SMS Queue → SMS Workers (レート制限付き)
  - 各キューが独立してスケール・障害分離可能
```

---

## 7. FAQ

### Q1: APNs のデバイストークンが無効になった場合はどう対処しますか？

**A:** APNs は無効なトークンに対して HTTP 410 (Gone) を返します。以下の対応を行います。

1. **即座にトークンを無効化**: DB 上のトークンを削除または無効フラグを立てる
2. **フィードバックサービスの監視**: APNs のフィードバックサービスから無効トークン一覧を定期取得
3. **再登録の促進**: アプリ起動時に毎回トークンを再取得し、サーバーに送信する設計

### Q2: 通知の A/B テストはどう実装しますか？

**A:** 以下のアーキテクチャで実装します。

1. **テンプレートのバリアント管理**: 1つのテンプレート ID に複数のバリアント（A/B）を紐づけ
2. **ユーザーの振り分け**: ユーザー ID のハッシュで A/B グループに振り分け（一貫性を保つ）
3. **メトリクス収集**: 開封率、クリック率、コンバージョン率をバリアントごとに記録
4. **自動最適化**: 統計的に有意な差が出たら勝者バリアントに統一

### Q3: グローバルサービスでのタイムゾーン対応はどうしますか？

**A:** ユーザーのタイムゾーンに基づいたスケジューリングを行います。

- ユーザープロファイルにタイムゾーンを保存
- 「朝9時に送信」→ 各ユーザーのローカル時間の9時に送信
- 配信ウィンドウの設定: 「深夜0時〜朝7時は送信しない」をユーザーのタイムゾーンで適用
- スケジューラーは UTC で管理し、送信時にタイムゾーン変換

---

## 8. まとめ

| 設計要素 | 選択 | 理由 |
|----------|------|------|
| キューイング | Kafka (チャネル別トピック) | 高スループット + 障害分離 |
| 重複排除 | Redis SETNX | 高速なべき等性チェック |
| レート制限 | Redis Sorted Set (スライディングウィンドウ) | 正確なレート計算 |
| テンプレート | Jinja2 + DB管理 | 多言語・多チャネル対応 |
| リトライ | 指数バックオフ + DLQ | 信頼性確保 |
| 監視 | Prometheus + Grafana | 配信率・エラー率の可視化 |

---

## 次に読むべきガイド

- [チャットシステム設計](./01-chat-system.md) — リアルタイムメッセージングとの連携
- [URL 短縮サービス設計](./00-url-shortener.md) — 通知内リンクの短縮
- [認証・認可](../../../authentication-and-authorization/docs/01-fundamentals/00-overview.md) — 通知 API のセキュリティ

---

## 参考文献

1. Xu, A. (2020). "System Design Interview: An Insider's Guide." *Chapter 10: Design a Notification System*. https://www.systemdesigninterview.com/
2. Apple Inc. (2024). "Sending Notification Requests to APNs." *Apple Developer Documentation*. https://developer.apple.com/documentation/usernotifications/sending-notification-requests-to-apns
3. Google. (2024). "Firebase Cloud Messaging Architecture." *Firebase Documentation*. https://firebase.google.com/docs/cloud-messaging/fcm-architecture

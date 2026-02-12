# 通知システム設計

> iOS プッシュ通知、Android FCM、SMS、メール、アプリ内通知を統合的に管理する大規模通知システムをゼロから設計する。配信保証、レート制限、ユーザー設定管理、優先度制御、分析基盤を含む包括的なアーキテクチャを解説する。

---

## この章で学ぶこと

1. **通知チャネルの統合アーキテクチャ** — プッシュ通知、SMS、メール、アプリ内通知を統一 API で管理し、チャネル間の協調を実現する設計手法
2. **信頼性と配信保証の実現** — メッセージの重複排除、リトライ戦略、配信確認、Dead Letter Queue を用いた障害復旧の仕組み
3. **ユーザー体験の最適化** — レート制限、優先度管理、バッチ集約、タイムゾーン対応、ユーザー設定に基づくインテリジェント配信
4. **スケーラビリティと運用** — 数十億通/日を処理するための水平スケーリング、監視、A/B テスト、分析基盤の構築

---

## 前提知識

このガイドを読む前に、以下の知識があるとスムーズに理解できます。

| トピック | 参照先 |
|---------|--------|
| システム設計の基礎概念 | [システム設計概要](../00-fundamentals/00-system-design-overview.md) |
| スケーラビリティの原則 | [スケーラビリティ](../00-fundamentals/01-scalability.md) |
| メッセージキューの仕組み | [メッセージキュー](../01-components/02-message-queue.md) |
| キャッシュ戦略 | [キャッシング](../01-components/01-caching.md) |
| イベント駆動アーキテクチャ | [イベント駆動設計](../02-architecture/03-event-driven.md) |
| Observer パターン | [Observer パターン](../../design-patterns-guide/docs/02-behavioral/00-observer.md) |
| API 設計のベストプラクティス | [API 設計](../../clean-code-principles/docs/03-practices-advanced/03-api-design.md) |

---

## 1. 要件定義

### 1.1 機能要件

通知システムが提供すべき主要機能を明確化する。

```
機能要件一覧:

1. マルチチャネル配信
   - iOS プッシュ通知 (APNs)
   - Android プッシュ通知 (FCM)
   - SMS (Twilio, AWS SNS 等)
   - メール (SES, SendGrid 等)
   - アプリ内通知 (WebSocket / SSE)
   - Web Push (Service Worker)

2. テンプレート管理
   - 多言語テンプレート (i18n)
   - チャネル別テンプレート
   - 動的変数の埋め込み (Jinja2 / Mustache)
   - テンプレートのバージョン管理

3. ユーザー設定
   - チャネル別のオン/オフ
   - カテゴリ別の通知設定 (マーケティング/トランザクション等)
   - Quiet Hours (通知しない時間帯)
   - 頻度制限のカスタマイズ

4. 配信制御
   - スケジュール配信 (指定日時に送信)
   - 優先度制御 (critical / high / normal / low)
   - レート制限 (ユーザー単位 / チャネル単位)
   - バッチ集約 (同種通知のまとめ)

5. 運用機能
   - 通知履歴の保存と閲覧 API
   - 配信状況のリアルタイム監視
   - A/B テスト
   - 配信分析 (開封率、クリック率)
```

### 1.2 非機能要件

```
非機能要件:

1. パフォーマンス
   - 通知受付 API: p99 レイテンシ < 100ms
   - 配信遅延: 通常通知 < 30秒、優先通知 < 5秒
   - スループット: ピーク 200,000 QPS 以上

2. 可用性
   - SLA: 99.99% (年間ダウンタイム < 52分)
   - データセンター障害時の自動フェイルオーバー

3. 耐久性
   - 通知メッセージの損失ゼロ (at-least-once 配信)
   - 通知履歴の 90日間保持

4. スケーラビリティ
   - 水平スケーリングで 10x の負荷増に対応
   - チャネル追加が既存システムに影響しない

5. セキュリティ
   - API 認証・認可 (OAuth 2.0 / API Key)
   - PII データの暗号化
   - GDPR / 個人情報保護法への準拠
```

### 1.3 スケール見積もり

```
前提:
  - 登録ユーザー: 500M
  - DAU: 100M
  - 1日あたり通知数: 10B（プッシュ 5B + メール 3B + SMS 0.5B + アプリ内 1.5B）

スループット計算:
  プッシュ通知:
    5B / 86,400 ≈ 57,870 QPS (ピーク 2x ≈ 115,000 QPS)

  メール:
    3B / 86,400 ≈ 34,700 QPS (ピーク 2x ≈ 70,000 QPS)

  SMS:
    0.5B / 86,400 ≈ 5,780 QPS (ピーク 2x ≈ 11,560 QPS)

  アプリ内通知:
    1.5B / 86,400 ≈ 17,360 QPS (ピーク 2x ≈ 34,720 QPS)

  全チャネル合計ピーク: ≈ 231,280 QPS → 約 200K+ QPS

ストレージ計算:
  通知メタデータ (1通知 ≈ 500B):
    10B * 500B = 5TB/日
    90日保持: 450TB

  通知履歴 (ユーザーごとの閲覧用、直近100件):
    500M users * 100件 * 200B = 10TB

  テンプレート:
    10,000 テンプレート * 10KB = 100MB (無視可能)

帯域幅:
  メール本文 (平均 50KB):
    3B * 50KB = 150TB/日 → 約 14 Gbps

  プッシュペイロード (平均 1KB):
    5B * 1KB = 5TB/日 → 約 0.5 Gbps
```

---

## 2. 高レベルアーキテクチャ

### 2.1 全体構成図

```
通知システム 全体アーキテクチャ

  +-----------+    +-----------+    +-----------+
  | マイクロ   |    | 管理画面   |    | スケジュー |
  | サービス   |    | (CMS)     |    | ラー      |
  | (イベント) |    |           |    | (Cron)    |
  +-----------+    +-----------+    +-----------+
       |                |                |
       v                v                v
  +--------------------------------------------------+
  |              通知 API ゲートウェイ                   |
  |  (認証、レート制限、バリデーション、ルーティング)        |
  +--------------------------------------------------+
                        |
                        v
  +--------------------------------------------------+
  |              通知オーケストレーター                   |
  |                                                    |
  |  +----------+  +---------+  +---------+  +------+ |
  |  | ユーザー  |  | テンプレ |  | 優先度  |  | 重複 | |
  |  | 設定確認  |->| ート展開 |->| 判定    |->| 排除 | |
  |  +----------+  +---------+  +---------+  +------+ |
  +--------------------------------------------------+
                        |
          +-------------+-------------+
          |             |             |
     +---------+  +---------+  +---------+
     | Push    |  | Email   |  | SMS     |
     | Queue   |  | Queue   |  | Queue   |
     +---------+  +---------+  +---------+
          |             |             |
     +---------+  +---------+  +---------+
     | Push    |  | Email   |  | SMS     |
     | Workers |  | Workers |  | Workers |
     +---------+  +---------+  +---------+
          |             |             |
     +----+----+  +-----+-----+  +--------+
     |    |    |  |     |     |  |        |
    APNs  FCM WP  SES SendGrid  Twilio  Vonage

  +--------------------------------------------------+
  |         分析・監視基盤                              |
  |  配信ログ → Kafka → ClickHouse → Grafana          |
  +--------------------------------------------------+
```

### 2.2 詳細コンポーネント図

```
+-------------------------------------------------------------------+
|                    通知サービス (Notification Service)                |
+-------------------------------------------------------------------+
|                                                                     |
|  [受付 API]                                                         |
|      |                                                              |
|      v                                                              |
|  [バリデーション] --- 不正リクエスト → 400 Error                       |
|      |                                                              |
|      v                                                              |
|  [ユーザー設定確認] --- Redis Cache (TTL: 5min)                      |
|      |                   |                                          |
|      |              [User Preferences DB]                           |
|      |                                                              |
|      v                                                              |
|  [テンプレート展開] --- Template Cache (TTL: 1hour)                   |
|      |                   |                                          |
|      |              [Template DB (versioned)]                       |
|      |                                                              |
|      v                                                              |
|  [優先度判定・ルーティング]                                            |
|      |                                                              |
|      +-- critical → 即時配信 (キューバイパス)                          |
|      +-- high     → 優先キュー                                       |
|      +-- normal   → 通常キュー                                       |
|      +-- low      → バッチキュー (集約後配信)                          |
|      |                                                              |
|      v                                                              |
|  [重複排除] --- Redis SETNX (TTL: 24h)                               |
|      |                                                              |
|      v                                                              |
|  [レート制限] --- Redis Sorted Set (Sliding Window)                   |
|      |                                                              |
|      v                                                              |
|  [チャネル別キュー投入] --- Kafka Topics                               |
|                                                                     |
+-------------------------------------------------------------------+
```

### 2.3 データフロー図

```
通知の一生（ライフサイクル）

  [1. 生成]          [2. 処理]          [3. 配信]         [4. 追跡]

  イベント発生     →  設定確認       →  キュー投入     →  配信ログ記録
  API 呼び出し        テンプレ展開      ワーカー処理       開封トラッキング
  スケジュール実行    重複排除          プロバイダ送信     クリック追跡
  ルールエンジン      レート制限        リトライ処理       分析集計

  ステータス遷移:
  CREATED → PROCESSING → QUEUED → SENDING → DELIVERED → OPENED → CLICKED
                                     |
                                     +→ FAILED → RETRYING → DELIVERED
                                                     |
                                                     +→ DEAD_LETTERED
```

---

## 3. データモデル設計

### 3.1 主要テーブル

```sql
-- 通知リクエスト (バッチ単位)
CREATE TABLE notification_batches (
    batch_id        UUID PRIMARY KEY,
    template_id     VARCHAR(100) NOT NULL,
    channels        JSONB NOT NULL,          -- ["push", "email"]
    data            JSONB NOT NULL,          -- テンプレート変数
    priority        VARCHAR(10) DEFAULT 'normal',
    scheduled_at    TIMESTAMPTZ,
    created_by      VARCHAR(100) NOT NULL,   -- 送信元サービス
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    status          VARCHAR(20) DEFAULT 'processing'
);

-- 個別通知 (ユーザー × チャネル 単位)
CREATE TABLE notifications (
    notification_id UUID PRIMARY KEY,
    batch_id        UUID REFERENCES notification_batches(batch_id),
    user_id         VARCHAR(100) NOT NULL,
    channel         VARCHAR(20) NOT NULL,
    content         JSONB NOT NULL,          -- 展開済みコンテンツ
    priority        VARCHAR(10) NOT NULL,
    status          VARCHAR(20) DEFAULT 'queued',
    -- queued / sending / delivered / failed / dead_lettered
    retry_count     INT DEFAULT 0,
    sent_at         TIMESTAMPTZ,
    delivered_at    TIMESTAMPTZ,
    opened_at       TIMESTAMPTZ,
    clicked_at      TIMESTAMPTZ,
    error_message   TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    INDEX idx_notifications_user (user_id, created_at DESC),
    INDEX idx_notifications_status (status, channel),
    INDEX idx_notifications_batch (batch_id)
);

-- ユーザー通知設定
CREATE TABLE user_notification_preferences (
    user_id         VARCHAR(100) PRIMARY KEY,
    push_enabled    BOOLEAN DEFAULT TRUE,
    email_enabled   BOOLEAN DEFAULT TRUE,
    sms_enabled     BOOLEAN DEFAULT TRUE,
    in_app_enabled  BOOLEAN DEFAULT TRUE,
    quiet_hours     JSONB,  -- {"start": "23:00", "end": "07:00", "timezone": "Asia/Tokyo"}
    categories      JSONB,  -- {"marketing": false, "transaction": true, ...}
    frequency_limit JSONB,  -- カスタムレート制限
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- デバイストークン
CREATE TABLE device_tokens (
    token_id        UUID PRIMARY KEY,
    user_id         VARCHAR(100) NOT NULL,
    platform        VARCHAR(20) NOT NULL,    -- ios / android / web
    device_token    TEXT NOT NULL,
    app_version     VARCHAR(20),
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_used_at    TIMESTAMPTZ,

    UNIQUE(user_id, device_token),
    INDEX idx_device_tokens_user (user_id, is_active)
);

-- 通知テンプレート
CREATE TABLE notification_templates (
    template_id     VARCHAR(100) NOT NULL,
    version         INT NOT NULL,
    channel         VARCHAR(20) NOT NULL,
    locale          VARCHAR(10) NOT NULL,    -- ja, en, zh, ...
    subject         TEXT,                     -- メール件名等
    title           TEXT,                     -- プッシュタイトル
    body            TEXT NOT NULL,            -- 本文テンプレート
    metadata        JSONB,                   -- カスタムデータ
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (template_id, version, channel, locale)
);
```

### 3.2 Redis データ構造

```
Redis キー設計:

1. ユーザー設定キャッシュ
   Key:    user_prefs:{user_id}
   Type:   Hash
   TTL:    300s (5分)
   Fields: push_enabled, email_enabled, sms_enabled, quiet_hours, categories

2. レート制限カウンター
   Key:    ratelimit:{user_id}:{channel}:hour
   Type:   Sorted Set (score = timestamp)
   TTL:    3600s

   Key:    ratelimit:{user_id}:{channel}:day
   Type:   Sorted Set (score = timestamp)
   TTL:    86400s

3. 重複排除
   Key:    dedup:{notification_id}
   Type:   String ("1")
   TTL:    86400s (24時間)

4. デバイストークンキャッシュ
   Key:    devices:{user_id}
   Type:   Hash (platform → token)
   TTL:    600s (10分)

5. バッチ集約バッファ
   Key:    batch:{user_id}:{category}
   Type:   List (集約待ち通知)
   TTL:    300s (5分ウィンドウ)
```

---

## 4. コア実装

### 4.1 通知 API と配信パイプライン

```python
# コード例 1: 通知サービスのエントリポイント
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel, validator
from enum import Enum
from typing import Optional
import uuid
import logging

app = FastAPI(title="Notification Service", version="2.0")
logger = logging.getLogger(__name__)

class Channel(str, Enum):
    PUSH = "push"
    EMAIL = "email"
    SMS = "sms"
    IN_APP = "in_app"
    WEB_PUSH = "web_push"

class Priority(str, Enum):
    CRITICAL = "critical"  # システム障害、セキュリティ
    HIGH = "high"          # 決済完了、パスワード変更
    NORMAL = "normal"      # 一般通知
    LOW = "low"            # マーケティング、レコメンド

class NotificationRequest(BaseModel):
    user_ids: list[str]
    template_id: str
    channels: list[Channel]
    data: dict
    priority: Priority = Priority.NORMAL
    scheduled_at: Optional[str] = None
    idempotency_key: Optional[str] = None  # べき等性キー
    category: str = "general"               # 通知カテゴリ

    @validator('user_ids')
    def validate_user_ids(cls, v):
        if len(v) == 0:
            raise ValueError("user_ids must not be empty")
        if len(v) > 10000:
            raise ValueError("user_ids must not exceed 10000")
        return v

class NotificationResponse(BaseModel):
    batch_id: str
    status: str
    total_recipients: int
    accepted: int
    rejected: int

class NotificationService:
    """
    通知サービスのコアオーケストレーター。

    責務:
    1. リクエストのバリデーション
    2. ユーザー設定に基づくフィルタリング
    3. テンプレートの展開
    4. 重複排除
    5. レート制限の適用
    6. チャネル別キューへの投入
    """

    def __init__(self, queue, user_service, template_service,
                 rate_limiter, dedup_service, metrics):
        self.queue = queue
        self.user_service = user_service
        self.template_service = template_service
        self.rate_limiter = rate_limiter
        self.dedup_service = dedup_service
        self.metrics = metrics

    async def send(self, request: NotificationRequest) -> NotificationResponse:
        batch_id = str(uuid.uuid4())
        accepted = 0
        rejected = 0

        # べき等性チェック
        if request.idempotency_key:
            if await self.dedup_service.is_duplicate(request.idempotency_key):
                logger.info(f"Duplicate request: {request.idempotency_key}")
                raise HTTPException(status_code=409, detail="Duplicate request")

        for user_id in request.user_ids:
            try:
                result = await self._process_user(
                    batch_id, user_id, request
                )
                if result:
                    accepted += 1
                else:
                    rejected += 1
            except Exception as e:
                logger.error(f"Error processing user {user_id}: {e}")
                rejected += 1
                self.metrics.increment("notification.processing_error")

        # メトリクス記録
        self.metrics.increment("notification.batches_created")
        self.metrics.gauge("notification.batch_size", len(request.user_ids))

        return NotificationResponse(
            batch_id=batch_id,
            status="queued",
            total_recipients=len(request.user_ids),
            accepted=accepted,
            rejected=rejected,
        )

    async def _process_user(self, batch_id: str, user_id: str,
                             request: NotificationRequest) -> bool:
        """個別ユーザーへの通知処理"""

        # 1. ユーザー設定を確認（キャッシュ付き）
        prefs = await self.user_service.get_preferences(user_id)
        if prefs is None:
            logger.warning(f"User not found: {user_id}")
            return False

        # 2. カテゴリ設定の確認
        if not prefs.is_category_enabled(request.category):
            return False

        # 3. Quiet Hours の確認
        if prefs.is_quiet_hours():
            if request.priority not in (Priority.CRITICAL, Priority.HIGH):
                # 低優先度はスケジュール配信に回す
                await self._schedule_after_quiet_hours(
                    batch_id, user_id, request, prefs
                )
                return True

        channels_sent = 0
        for channel in request.channels:
            # 4. ユーザーがこのチャネルを許可しているか
            if not prefs.is_channel_enabled(channel):
                continue

            # 5. レート制限チェック
            if not await self.rate_limiter.allow(user_id, channel.value):
                self.metrics.increment("notification.rate_limited",
                                       tags={"channel": channel.value})
                continue

            # 6. テンプレートを展開
            locale = prefs.locale or "ja"
            content = await self.template_service.render(
                request.template_id, channel.value,
                request.data, locale=locale
            )

            # 7. 通知IDの生成と重複排除
            notification_id = f"{batch_id}:{user_id}:{channel.value}"
            if await self.dedup_service.is_duplicate(notification_id):
                continue

            # 8. メッセージキューに投入
            message = {
                "id": notification_id,
                "batch_id": batch_id,
                "user_id": user_id,
                "channel": channel.value,
                "content": content,
                "priority": request.priority.value,
                "category": request.category,
                "created_at": int(time.time()),
            }

            # 優先度に応じたキュー選択
            topic = self._select_topic(channel, request.priority)
            await self.queue.publish(topic=topic, message=message)
            channels_sent += 1

        return channels_sent > 0

    def _select_topic(self, channel: Channel, priority: Priority) -> str:
        """優先度に応じたKafkaトピックを選択"""
        if priority == Priority.CRITICAL:
            return f"notifications.{channel.value}.critical"
        elif priority == Priority.HIGH:
            return f"notifications.{channel.value}.high"
        else:
            return f"notifications.{channel.value}.normal"

    async def _schedule_after_quiet_hours(self, batch_id, user_id,
                                           request, prefs):
        """Quiet Hours 後にスケジュール配信"""
        send_at = prefs.get_quiet_hours_end_utc()
        await self.queue.publish(
            topic="notifications.scheduled",
            message={
                "batch_id": batch_id,
                "user_id": user_id,
                "request": request.dict(),
                "scheduled_at": send_at.isoformat(),
            }
        )

@app.post("/api/v1/notifications", response_model=NotificationResponse)
async def create_notification(request: NotificationRequest):
    """通知作成 API エンドポイント"""
    service = get_notification_service()  # DI コンテナから取得
    return await service.send(request)

@app.get("/api/v1/notifications/{batch_id}/status")
async def get_batch_status(batch_id: str):
    """バッチの配信状況を取得"""
    stats = await get_batch_statistics(batch_id)
    return {
        "batch_id": batch_id,
        "total": stats["total"],
        "delivered": stats["delivered"],
        "failed": stats["failed"],
        "pending": stats["pending"],
        "delivery_rate": stats["delivered"] / max(stats["total"], 1),
    }
```

### 4.2 チャネルアダプタ（Strategy パターン）

```python
# コード例 2: チャネルアダプタの統一インターフェース
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class DeliveryResult:
    """配信結果を統一的に表現"""
    success: bool
    provider_message_id: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    should_retry: bool = False

class ChannelAdapter(ABC):
    """
    全チャネルの共通インターフェース (Strategy パターン)。

    新しいチャネルを追加する場合は、このクラスを継承して
    send() メソッドを実装するだけでよい。
    Open/Closed Principle (OCP) に従った設計。
    """

    @abstractmethod
    async def send(self, user_id: str, content: dict,
                   metadata: dict) -> DeliveryResult:
        pass

    @abstractmethod
    async def validate_target(self, user_id: str) -> bool:
        """配信先が有効か確認 (トークン有効性等)"""
        pass

class APNsAdapter(ChannelAdapter):
    """Apple Push Notification service アダプタ"""

    def __init__(self, team_id: str, key_id: str, private_key: str,
                 bundle_id: str, use_sandbox: bool = False):
        self.team_id = team_id
        self.key_id = key_id
        self.private_key = private_key
        self.bundle_id = bundle_id
        self.base_url = (
            "https://api.sandbox.push.apple.com"
            if use_sandbox else
            "https://api.push.apple.com"
        )
        self._jwt_token = None
        self._jwt_expiry = 0

    async def send(self, user_id: str, content: dict,
                   metadata: dict) -> DeliveryResult:
        devices = await get_user_devices(user_id, platform="ios")
        if not devices:
            return DeliveryResult(success=False, error_code="NO_DEVICE")

        results = []
        for device in devices:
            result = await self._send_to_device(device, content, metadata)
            results.append(result)

            # 無効トークンの場合は即座に無効化
            if result.error_code == "INVALID_TOKEN":
                await invalidate_device_token(device["token_id"])

        # 1台でも成功すれば OK
        return DeliveryResult(
            success=any(r.success for r in results),
            should_retry=all(r.should_retry for r in results),
        )

    async def _send_to_device(self, device: dict, content: dict,
                                metadata: dict) -> DeliveryResult:
        token = await self._get_jwt_token()
        payload = {
            "aps": {
                "alert": {
                    "title": content.get("title", ""),
                    "body": content.get("body", ""),
                    "subtitle": content.get("subtitle", ""),
                },
                "badge": metadata.get("badge", 1),
                "sound": metadata.get("sound", "default"),
                "category": metadata.get("action_category", ""),
                "mutable-content": 1,  # Notification Service Extension
                "thread-id": metadata.get("thread_id", ""),
            },
            "custom_data": content.get("data", {}),
        }

        # HTTP/2 で APNs に送信
        headers = {
            "authorization": f"bearer {token}",
            "apns-topic": self.bundle_id,
            "apns-priority": "10" if metadata.get("priority") == "high" else "5",
            "apns-push-type": "alert",
            "apns-expiration": str(int(time.time()) + 86400),
        }

        try:
            response = await self.http_client.post(
                f"{self.base_url}/3/device/{device['device_token']}",
                json=payload,
                headers=headers,
                timeout=10,
            )

            if response.status_code == 200:
                return DeliveryResult(
                    success=True,
                    provider_message_id=response.headers.get("apns-id"),
                )
            elif response.status_code == 410:
                return DeliveryResult(
                    success=False,
                    error_code="INVALID_TOKEN",
                    error_message="Device token is no longer active",
                    should_retry=False,
                )
            elif response.status_code in (500, 503):
                return DeliveryResult(
                    success=False,
                    error_code="PROVIDER_ERROR",
                    error_message=f"APNs returned {response.status_code}",
                    should_retry=True,
                )
            else:
                body = response.json()
                return DeliveryResult(
                    success=False,
                    error_code=body.get("reason", "UNKNOWN"),
                    error_message=str(body),
                    should_retry=False,
                )
        except asyncio.TimeoutError:
            return DeliveryResult(
                success=False,
                error_code="TIMEOUT",
                should_retry=True,
            )

    async def validate_target(self, user_id: str) -> bool:
        devices = await get_user_devices(user_id, platform="ios")
        return len(devices) > 0

class FCMAdapter(ChannelAdapter):
    """Firebase Cloud Messaging アダプタ"""

    async def send(self, user_id: str, content: dict,
                   metadata: dict) -> DeliveryResult:
        devices = await get_user_devices(user_id, platform="android")
        if not devices:
            return DeliveryResult(success=False, error_code="NO_DEVICE")

        for device in devices:
            message = {
                "message": {
                    "token": device["device_token"],
                    "notification": {
                        "title": content.get("title", ""),
                        "body": content.get("body", ""),
                        "image": content.get("image_url"),
                    },
                    "data": {k: str(v) for k, v in content.get("data", {}).items()},
                    "android": {
                        "priority": "high" if metadata.get("priority") == "high" else "normal",
                        "notification": {
                            "channel_id": metadata.get("android_channel", "default"),
                            "click_action": metadata.get("click_action", ""),
                        },
                    },
                }
            }

            try:
                response = await self.http_client.post(
                    f"https://fcm.googleapis.com/v1/projects/{self.project_id}/messages:send",
                    json=message,
                    headers={"authorization": f"Bearer {await self._get_access_token()}"},
                    timeout=10,
                )

                if response.status_code == 200:
                    result = response.json()
                    return DeliveryResult(
                        success=True,
                        provider_message_id=result.get("name"),
                    )
                elif response.status_code == 404:
                    await invalidate_device_token(device["token_id"])
                    return DeliveryResult(
                        success=False,
                        error_code="INVALID_TOKEN",
                        should_retry=False,
                    )
                else:
                    return DeliveryResult(
                        success=False,
                        error_code=f"FCM_{response.status_code}",
                        should_retry=response.status_code >= 500,
                    )
            except asyncio.TimeoutError:
                return DeliveryResult(
                    success=False, error_code="TIMEOUT", should_retry=True
                )

        return DeliveryResult(success=False, error_code="ALL_FAILED")

    async def validate_target(self, user_id: str) -> bool:
        devices = await get_user_devices(user_id, platform="android")
        return len(devices) > 0

class EmailAdapter(ChannelAdapter):
    """メール配信アダプタ (SES / SendGrid 対応)"""

    async def send(self, user_id: str, content: dict,
                   metadata: dict) -> DeliveryResult:
        email = await get_user_email(user_id)
        if not email:
            return DeliveryResult(success=False, error_code="NO_EMAIL")

        ses_message = {
            "Source": metadata.get("from_email", "noreply@example.com"),
            "Destination": {"ToAddresses": [email]},
            "Message": {
                "Subject": {"Data": content.get("subject", "")},
                "Body": {
                    "Html": {"Data": content.get("html_body", "")},
                    "Text": {"Data": content.get("text_body", content.get("body", ""))},
                },
            },
            "Tags": [
                {"Name": "category", "Value": metadata.get("category", "general")},
                {"Name": "batch_id", "Value": metadata.get("batch_id", "")},
            ],
        }

        try:
            response = await self.ses_client.send_email(**ses_message)
            return DeliveryResult(
                success=True,
                provider_message_id=response["MessageId"],
            )
        except self.ses_client.exceptions.MessageRejected as e:
            return DeliveryResult(
                success=False,
                error_code="REJECTED",
                error_message=str(e),
                should_retry=False,
            )
        except Exception as e:
            return DeliveryResult(
                success=False,
                error_code="SES_ERROR",
                error_message=str(e),
                should_retry=True,
            )

    async def validate_target(self, user_id: str) -> bool:
        email = await get_user_email(user_id)
        return email is not None and await is_email_valid(email)

class SMSAdapter(ChannelAdapter):
    """SMS 配信アダプタ (Twilio 対応)"""

    async def send(self, user_id: str, content: dict,
                   metadata: dict) -> DeliveryResult:
        phone = await get_user_phone(user_id)
        if not phone:
            return DeliveryResult(success=False, error_code="NO_PHONE")

        try:
            message = await self.twilio_client.messages.create_async(
                body=content.get("body", ""),
                from_=self.from_number,
                to=phone,
                status_callback=f"{self.callback_base_url}/sms/status",
            )
            return DeliveryResult(
                success=True,
                provider_message_id=message.sid,
            )
        except Exception as e:
            return DeliveryResult(
                success=False,
                error_code="TWILIO_ERROR",
                error_message=str(e),
                should_retry="temporarily" in str(e).lower(),
            )

    async def validate_target(self, user_id: str) -> bool:
        phone = await get_user_phone(user_id)
        return phone is not None
```

### 4.3 ワーカーとリトライ処理

```python
# コード例 3: 通知ワーカーのリトライ戦略
import asyncio
import time
import json
from dataclasses import dataclass

@dataclass
class RetryConfig:
    """チャネル別リトライ設定"""
    max_retries: int
    base_delay: float       # 秒
    max_delay: float        # 秒
    backoff_factor: float   # 指数バックオフ係数

# チャネルごとの最適なリトライ設定
RETRY_CONFIGS = {
    "push": RetryConfig(max_retries=3, base_delay=1.0, max_delay=60.0, backoff_factor=2.0),
    "email": RetryConfig(max_retries=5, base_delay=5.0, max_delay=300.0, backoff_factor=3.0),
    "sms": RetryConfig(max_retries=2, base_delay=10.0, max_delay=60.0, backoff_factor=2.0),
    "in_app": RetryConfig(max_retries=3, base_delay=1.0, max_delay=30.0, backoff_factor=2.0),
}

class NotificationWorker:
    """
    Kafka からメッセージを消費して通知を配信するワーカー。

    リトライ戦略:
    - 指数バックオフ + ジッター
    - チャネルごとに最大リトライ回数を設定
    - 最大リトライ超過時は DLQ (Dead Letter Queue) に移動
    - DLQ のメッセージはアラート + 手動対応
    """

    def __init__(self, channel: str, adapter: ChannelAdapter,
                 consumer, dlq_producer, metrics):
        self.channel = channel
        self.adapter = adapter
        self.consumer = consumer
        self.dlq_producer = dlq_producer
        self.metrics = metrics
        self.retry_config = RETRY_CONFIGS.get(
            channel, RetryConfig(3, 1.0, 60.0, 2.0)
        )

    async def run(self):
        """メインループ: Kafka からメッセージを消費し続ける"""
        async for message in self.consumer:
            try:
                await self._process_message(json.loads(message.value))
                await self.consumer.commit()
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                self.metrics.increment("worker.unexpected_error")

    async def _process_message(self, message: dict):
        """メッセージの処理とリトライ"""
        notification_id = message["id"]
        retry_count = message.get("retry_count", 0)
        start_time = time.time()

        # 配信実行
        result = await self.adapter.send(
            user_id=message["user_id"],
            content=message["content"],
            metadata={
                "priority": message.get("priority"),
                "batch_id": message.get("batch_id"),
                "category": message.get("category"),
            },
        )

        # メトリクス記録
        latency = time.time() - start_time
        self.metrics.histogram(
            "notification.delivery_latency",
            latency,
            tags={"channel": self.channel, "success": str(result.success)},
        )

        if result.success:
            # 配信成功
            await self._record_delivery(notification_id, result)
            self.metrics.increment(
                "notification.delivered",
                tags={"channel": self.channel},
            )
        elif result.should_retry and retry_count < self.retry_config.max_retries:
            # リトライ
            delay = self._calculate_delay(retry_count)
            message["retry_count"] = retry_count + 1
            await self._schedule_retry(message, delay)
            self.metrics.increment(
                "notification.retry",
                tags={"channel": self.channel, "attempt": str(retry_count + 1)},
            )
        else:
            # DLQ に移動
            await self._send_to_dlq(message, result)
            self.metrics.increment(
                "notification.dead_lettered",
                tags={"channel": self.channel},
            )

    def _calculate_delay(self, retry_count: int) -> float:
        """指数バックオフ + ジッターでリトライ間隔を計算"""
        import random
        delay = min(
            self.retry_config.base_delay * (self.retry_config.backoff_factor ** retry_count),
            self.retry_config.max_delay,
        )
        # ジッター: 0.5x ~ 1.5x のランダム変動
        jitter = delay * (0.5 + random.random())
        return jitter

    async def _schedule_retry(self, message: dict, delay: float):
        """リトライキューにスケジュール投入"""
        await asyncio.sleep(delay)  # 簡易実装。本番では遅延キューを使用
        topic = f"notifications.{self.channel}.retry"
        await self.dlq_producer.send(topic, json.dumps(message).encode())

    async def _send_to_dlq(self, message: dict, result: DeliveryResult):
        """Dead Letter Queue に移動"""
        dlq_message = {
            **message,
            "dlq_reason": result.error_code,
            "dlq_error": result.error_message,
            "dlq_timestamp": int(time.time()),
        }
        topic = f"notifications.{self.channel}.dlq"
        await self.dlq_producer.send(topic, json.dumps(dlq_message).encode())

        # アラート発行
        if message.get("priority") in ("critical", "high"):
            await send_alert(
                f"High priority notification failed: {message['id']}",
                severity="warning",
            )

    async def _record_delivery(self, notification_id: str,
                                result: DeliveryResult):
        """配信結果をDBに記録"""
        await update_notification_status(
            notification_id=notification_id,
            status="delivered",
            provider_message_id=result.provider_message_id,
            delivered_at=int(time.time()),
        )
```

### 4.4 レート制限

```python
# コード例 4: スライディングウィンドウによるレート制限
import time
import redis.asyncio as redis

class NotificationRateLimiter:
    """
    ユーザーあたり・チャネルあたりの通知レート制限。

    WHY: レート制限が必要な理由
    1. ユーザー体験の保護: 過剰な通知はアプリのアンインストールを招く
    2. プロバイダの制限遵守: APNs/FCM にもレート制限がある
    3. コスト管理: SMS は1通あたり課金されるため
    4. 法的遵守: CAN-SPAM 法、特定電子メール法等

    アルゴリズム: スライディングウィンドウ (Redis Sorted Set)
    - Fixed Window の「境界問題」を回避
    - O(log N) の計算量で正確なカウント
    - Redis のアトミック操作でレースコンディション防止
    """

    # デフォルトのレート制限設定
    DEFAULT_LIMITS = {
        "push":     {"per_hour": 10, "per_day": 50},
        "email":    {"per_hour": 3,  "per_day": 10},
        "sms":      {"per_hour": 2,  "per_day": 5},
        "in_app":   {"per_hour": 20, "per_day": 100},
        "web_push": {"per_hour": 8,  "per_day": 40},
    }

    # 優先度別の制限倍率
    PRIORITY_MULTIPLIERS = {
        "critical": float('inf'),  # 制限なし
        "high": 5.0,
        "normal": 1.0,
        "low": 0.5,
    }

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def allow(self, user_id: str, channel: str,
                    priority: str = "normal") -> bool:
        """通知を送信してよいか判定する"""
        # Critical は常に許可
        if priority == "critical":
            return True

        # ユーザーカスタム制限があれば適用
        custom_limits = await self._get_custom_limits(user_id, channel)
        limits = custom_limits or self.DEFAULT_LIMITS.get(
            channel, {"per_hour": 10, "per_day": 50}
        )

        # 優先度による倍率適用
        multiplier = self.PRIORITY_MULTIPLIERS.get(priority, 1.0)
        effective_hour = int(limits["per_hour"] * multiplier)
        effective_day = int(limits["per_day"] * multiplier)

        now = time.time()

        # Lua スクリプトでアトミックにチェック＆記録
        result = await self._check_and_record(
            user_id, channel, now, effective_hour, effective_day
        )
        return result

    LUA_RATE_CHECK = """
    local hour_key = KEYS[1]
    local day_key = KEYS[2]
    local now = tonumber(ARGV[1])
    local hour_limit = tonumber(ARGV[2])
    local day_limit = tonumber(ARGV[3])

    -- 古いエントリを削除
    redis.call('ZREMRANGEBYSCORE', hour_key, 0, now - 3600)
    redis.call('ZREMRANGEBYSCORE', day_key, 0, now - 86400)

    -- 現在のカウントを確認
    local hour_count = redis.call('ZCARD', hour_key)
    local day_count = redis.call('ZCARD', day_key)

    if hour_count >= hour_limit or day_count >= day_limit then
        return 0  -- 拒否
    end

    -- カウントを記録
    local member = now .. ':' .. math.random(1000000)
    redis.call('ZADD', hour_key, now, member)
    redis.call('ZADD', day_key, now, member)
    redis.call('EXPIRE', hour_key, 3600)
    redis.call('EXPIRE', day_key, 86400)

    return 1  -- 許可
    """

    async def _check_and_record(self, user_id: str, channel: str,
                                 now: float, hour_limit: int,
                                 day_limit: int) -> bool:
        """Lua スクリプトでアトミックにレート制限チェック"""
        hour_key = f"ratelimit:{user_id}:{channel}:hour"
        day_key = f"ratelimit:{user_id}:{channel}:day"

        result = await self.redis.eval(
            self.LUA_RATE_CHECK,
            2,  # KEYS の数
            hour_key, day_key,
            now, hour_limit, day_limit,
        )
        return bool(result)

    async def _get_custom_limits(self, user_id: str,
                                  channel: str) -> dict | None:
        """ユーザーカスタムのレート制限設定を取得"""
        key = f"user_prefs:{user_id}"
        data = await self.redis.hget(key, "frequency_limit")
        if data:
            import json
            limits = json.loads(data)
            return limits.get(channel)
        return None

    async def get_remaining(self, user_id: str, channel: str) -> dict:
        """残りのクォータを取得（API レスポンス用）"""
        limits = self.DEFAULT_LIMITS.get(channel, {"per_hour": 10, "per_day": 50})
        now = time.time()

        hour_key = f"ratelimit:{user_id}:{channel}:hour"
        day_key = f"ratelimit:{user_id}:{channel}:day"

        pipe = self.redis.pipeline()
        pipe.zrangebyscore(hour_key, now - 3600, now)
        pipe.zrangebyscore(day_key, now - 86400, now)
        hour_entries, day_entries = await pipe.execute()

        return {
            "hour": {
                "limit": limits["per_hour"],
                "remaining": max(0, limits["per_hour"] - len(hour_entries)),
                "reset": int(now) + 3600,
            },
            "day": {
                "limit": limits["per_day"],
                "remaining": max(0, limits["per_day"] - len(day_entries)),
                "reset": int(now) + 86400,
            },
        }
```

### 4.5 重複排除サービス

```python
# コード例 5: べき等性キーによる重複通知の防止
import hashlib
import json

class DeduplicationService:
    """
    同一通知の重複送信を防止するサービス。

    WHY: なぜ重複排除が必要か

    1. イベント駆動アーキテクチャでの at-least-once 配信
       → 同じイベントが複数回処理される可能性がある
    2. Kafka コンシューマーのリバランス
       → オフセットコミット前にリバランスが発生すると再処理
    3. ユーザーの重複操作
       → 「送信」ボタンの二重クリック
    4. マイクロサービス間のリトライ
       → HTTP タイムアウトでリトライ → 実際は成功していた

    実装方式の比較:
    +------------------+----------+---------+--------+
    | 方式              | メモリ   | 精度    | 速度   |
    +------------------+----------+---------+--------+
    | Redis SETNX      | 中       | 高      | 高     | ← 採用
    | Bloom Filter      | 低       | 近似    | 最高   |
    | DB Unique Key    | 高       | 最高    | 低     |
    +------------------+----------+---------+--------+
    """

    def __init__(self, redis_client, ttl: int = 86400):
        self.redis = redis_client
        self.ttl = ttl  # デフォルト24時間

    async def is_duplicate(self, key: str) -> bool:
        """このキーが既に処理済みかどうかを確認する"""
        dedup_key = f"dedup:{self._hash_key(key)}"
        result = await self.redis.set(dedup_key, "1", nx=True, ex=self.ttl)
        return result is None  # None = 既に存在 = 重複

    async def is_duplicate_content(self, user_id: str, channel: str,
                                    content_hash: str,
                                    window_seconds: int = 300) -> bool:
        """
        同一内容の通知が短時間に送信されていないか確認。

        例: 同じ「注文完了」通知が5分以内に2回送られるのを防ぐ
        """
        key = f"dedup:content:{user_id}:{channel}:{content_hash}"
        result = await self.redis.set(key, "1", nx=True, ex=window_seconds)
        return result is None

    @staticmethod
    def compute_content_hash(content: dict) -> str:
        """通知内容のハッシュを計算"""
        serialized = json.dumps(content, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    async def mark_sent(self, notification_id: str, channel: str):
        """通知の送信完了を記録する（配信追跡用）"""
        key = f"sent:{notification_id}:{channel}"
        await self.redis.set(key, "1", ex=self.ttl)

    async def was_sent(self, notification_id: str, channel: str) -> bool:
        """通知が既に送信済みか確認（リトライ時の冪等性チェック用）"""
        key = f"sent:{notification_id}:{channel}"
        return await self.redis.exists(key) > 0

    @staticmethod
    def _hash_key(key: str) -> str:
        """長いキーをハッシュ化してRedisメモリを節約"""
        if len(key) > 64:
            return hashlib.sha256(key.encode()).hexdigest()
        return key
```

### 4.6 テンプレートエンジン

```python
# コード例 6: 多言語・多チャネル対応テンプレートサービス
from jinja2 import Environment, BaseLoader, TemplateSyntaxError
from typing import Optional
import json

class NotificationTemplateService:
    """
    通知テンプレートの管理と展開を行うサービス。

    設計ポイント:
    1. テンプレートはDBで管理し、キャッシュで高速化
    2. チャネルごとに異なるテンプレートを用意
    3. 多言語対応 (locale ベースのフォールバック)
    4. テンプレートのバージョン管理 (ロールバック可能)
    5. A/B テスト用のバリアント管理
    """

    def __init__(self, db, cache, metrics):
        self.db = db
        self.cache = cache
        self.metrics = metrics
        self.env = Environment(loader=BaseLoader())

    async def render(self, template_id: str, channel: str,
                      data: dict, locale: str = "ja",
                      variant: Optional[str] = None) -> dict:
        """テンプレートを展開して配信コンテンツを生成"""

        # 1. キャッシュからテンプレートを取得
        cache_key = f"template:{template_id}:{channel}:{locale}"
        if variant:
            cache_key += f":{variant}"

        template_data = await self.cache.get(cache_key)

        if not template_data:
            # 2. DB から取得（フォールバック付き）
            template_data = await self._load_template(
                template_id, channel, locale, variant
            )
            if template_data:
                await self.cache.set(cache_key, json.dumps(template_data), ex=3600)
            else:
                self.metrics.increment("template.not_found")
                raise TemplateNotFoundError(
                    f"Template not found: {template_id}/{channel}/{locale}"
                )
        else:
            template_data = json.loads(template_data)

        # 3. Jinja2 でレンダリング
        try:
            rendered = {}
            for key, template_str in template_data.items():
                template = self.env.from_string(template_str)
                rendered[key] = template.render(**data)
            return rendered
        except TemplateSyntaxError as e:
            self.metrics.increment("template.render_error")
            raise TemplateRenderError(f"Template render failed: {e}")

    async def _load_template(self, template_id: str, channel: str,
                               locale: str, variant: Optional[str]) -> dict:
        """
        DB からテンプレートを取得。フォールバックチェーン:
        1. 指定 locale + variant
        2. 指定 locale (variant なし)
        3. デフォルト locale (en)
        """
        for try_locale in [locale, "en"]:
            result = await self.db.fetch_one(
                """
                SELECT title, subject, body, metadata
                FROM notification_templates
                WHERE template_id = :template_id
                  AND channel = :channel
                  AND locale = :locale
                  AND is_active = TRUE
                ORDER BY version DESC
                LIMIT 1
                """,
                {
                    "template_id": template_id,
                    "channel": channel,
                    "locale": try_locale,
                },
            )
            if result:
                template = {}
                if result["title"]:
                    template["title"] = result["title"]
                if result["subject"]:
                    template["subject"] = result["subject"]
                if result["body"]:
                    template["body"] = result["body"]
                return template

        return None

    async def create_template(self, template_id: str, channel: str,
                                locale: str, content: dict,
                                created_by: str) -> int:
        """テンプレートの新バージョンを作成"""
        # 現在の最新バージョンを取得
        current = await self.db.fetch_one(
            "SELECT MAX(version) as ver FROM notification_templates "
            "WHERE template_id = :tid AND channel = :ch AND locale = :lo",
            {"tid": template_id, "ch": channel, "lo": locale},
        )
        new_version = (current["ver"] or 0) + 1

        await self.db.execute(
            """
            INSERT INTO notification_templates
            (template_id, version, channel, locale, title, subject, body, metadata)
            VALUES (:tid, :ver, :ch, :lo, :title, :subject, :body, :meta)
            """,
            {
                "tid": template_id,
                "ver": new_version,
                "ch": channel,
                "lo": locale,
                "title": content.get("title"),
                "subject": content.get("subject"),
                "body": content.get("body"),
                "meta": json.dumps({"created_by": created_by}),
            },
        )

        # キャッシュ無効化
        cache_key = f"template:{template_id}:{channel}:{locale}"
        await self.cache.delete(cache_key)

        return new_version

class TemplateNotFoundError(Exception):
    pass

class TemplateRenderError(Exception):
    pass
```

### 4.7 通知バッチ集約

```python
# コード例 7: 同種通知のインテリジェント集約
import asyncio
import json
from collections import defaultdict

class NotificationAggregator:
    """
    同種の通知をバッチ集約するサービス。

    例: 「Aがいいねしました」「Bがいいねしました」「Cがいいねしました」
    → 「A、B、他1人がいいねしました」

    設計:
    - Redis List をバッファとして使用
    - 一定時間(5分)または一定件数(10件)でフラッシュ
    - ユーザー × カテゴリ単位で集約
    """

    AGGREGATION_WINDOW = 300  # 5分
    MAX_BATCH_SIZE = 10       # 最大集約数

    def __init__(self, redis_client, notification_service):
        self.redis = redis_client
        self.notification_service = notification_service

    async def add(self, user_id: str, category: str,
                  notification: dict) -> bool:
        """
        通知をバッファに追加。閾値に達したらフラッシュ。

        Returns:
            True: バッファに追加（後でまとめて送信）
            False: 集約対象外（即時送信）
        """
        # 集約対象のカテゴリか確認
        if category not in self.AGGREGATABLE_CATEGORIES:
            return False

        key = f"batch:{user_id}:{category}"

        pipe = self.redis.pipeline()
        pipe.rpush(key, json.dumps(notification))
        pipe.expire(key, self.AGGREGATION_WINDOW)
        pipe.llen(key)
        results = await pipe.execute()

        current_count = results[2]

        # 最大集約数に達したら即フラッシュ
        if current_count >= self.MAX_BATCH_SIZE:
            await self.flush(user_id, category)

        return True

    async def flush(self, user_id: str, category: str):
        """バッファ内の通知をまとめて送信"""
        key = f"batch:{user_id}:{category}"

        # アトミックに取得＆削除
        pipe = self.redis.pipeline()
        pipe.lrange(key, 0, -1)
        pipe.delete(key)
        results = await pipe.execute()

        raw_notifications = results[0]
        if not raw_notifications:
            return

        notifications = [json.loads(n) for n in raw_notifications]

        # 集約メッセージの生成
        aggregated_content = self._aggregate_content(category, notifications)

        # 集約通知を送信
        await self.notification_service.send_aggregated(
            user_id=user_id,
            content=aggregated_content,
            original_count=len(notifications),
        )

    def _aggregate_content(self, category: str,
                            notifications: list) -> dict:
        """通知内容を集約してサマリーを生成"""
        if category == "like":
            actors = [n["data"]["actor_name"] for n in notifications]
            if len(actors) <= 3:
                names = "、".join(actors)
            else:
                names = f"{actors[0]}、{actors[1]}、他{len(actors)-2}人"
            return {
                "title": "いいね通知",
                "body": f"{names}があなたの投稿にいいねしました",
                "data": {
                    "type": "like_aggregated",
                    "count": len(notifications),
                    "actor_ids": [n["data"]["actor_id"] for n in notifications],
                },
            }
        elif category == "follow":
            actors = [n["data"]["actor_name"] for n in notifications]
            if len(actors) <= 3:
                names = "、".join(actors)
            else:
                names = f"{actors[0]}、{actors[1]}、他{len(actors)-2}人"
            return {
                "title": "フォロー通知",
                "body": f"{names}があなたをフォローしました",
                "data": {
                    "type": "follow_aggregated",
                    "count": len(notifications),
                },
            }
        else:
            return {
                "title": f"{len(notifications)}件の通知",
                "body": f"{category}に関する{len(notifications)}件の新着通知があります",
            }

    AGGREGATABLE_CATEGORIES = {"like", "follow", "comment", "mention"}

    async def flush_all_expired(self):
        """
        定期実行: 期限切れバッファの一括フラッシュ。
        Cron ジョブで1分ごとに実行する。
        """
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(
                cursor, match="batch:*", count=1000
            )
            for key in keys:
                ttl = await self.redis.ttl(key)
                if ttl < 0 or ttl < self.AGGREGATION_WINDOW * 0.1:
                    # TTL が残り少ないバッファをフラッシュ
                    parts = key.decode().split(":")
                    if len(parts) == 3:
                        await self.flush(parts[1], parts[2])
            if cursor == 0:
                break
```

---

## 5. 信頼性設計

### 5.1 リトライ戦略の全体像

```
通知配信のリトライフロー

  [通知ワーカー]
       |
       v
  配信試行 ──成功──> [配信完了記録] → Done
       |
     失敗
       |
       v
  リトライ可能？ ──No──> [DLQ] → アラート → 手動対応
       |
      Yes
       |
       v
  リトライ回数 < 最大？ ──No──> [DLQ]
       |
      Yes
       |
       v
  指数バックオフ + ジッター で待機
       |
       v
  リトライキューに投入
       |
       v
  [通知ワーカー] (ループ)


  チャネル別リトライ設定:
  +----------+----------+-----------+----------+-------------------+
  | チャネル  | 最大回数  | 初期遅延   | 最大遅延  | 理由               |
  +----------+----------+-----------+----------+-------------------+
  | Push     | 3        | 1秒       | 60秒     | デバイスオフライン  |
  | Email    | 5        | 5秒       | 300秒    | SMTP 一時障害      |
  | SMS      | 2        | 10秒      | 60秒     | コストが高い       |
  | In-App   | 3        | 1秒       | 30秒     | 接続切れ           |
  | Web Push | 3        | 2秒       | 120秒    | ブラウザオフライン  |
  +----------+----------+-----------+----------+-------------------+

  指数バックオフの計算式:
    delay = min(base_delay * backoff_factor^attempt, max_delay)
    jitter = delay * (0.5 + random())
    actual_delay = jitter

  例 (Push, base=1s, factor=2):
    Attempt 0: 1s  * (0.5~1.5) = 0.5~1.5s
    Attempt 1: 2s  * (0.5~1.5) = 1.0~3.0s
    Attempt 2: 4s  * (0.5~1.5) = 2.0~6.0s
    → 失敗 → DLQ
```

### 5.2 障害分離パターン

```
チャネル間の障害分離

  問題: SMS プロバイダ (Twilio) がダウンした場合

  [悪い設計: 単一キュー]
  +-----+     +--------+     +----------+
  | All | --> | Single | --> | Workers  | → SMS がスタック
  | Msg | --> | Queue  | --> | (mixed)  |   → Push/Email もブロック
  +-----+     +--------+     +----------+

  [良い設計: チャネル分離]
  +-----+     +--------+     +----------+
  | Msg | --> | Push Q | --> | Push W   | → 正常動作
  +-----+     +--------+     +----------+
              +--------+     +----------+
              | Email Q| --> | Email W  | → 正常動作
              +--------+     +----------+
              +--------+     +----------+
              | SMS Q  | --> | SMS W    | → 障害 (分離済み)
              +--------+     +----------+

  さらに優先度別の分離:
  +--------+     +---------+     +------------+
  | Push   | --> | High Q  | --> | High W (4) | ← 多めのワーカー
  +--------+     +---------+     +------------+
              +-> | Normal Q| --> | Normal W(2)|
              |   +---------+     +------------+
              +-> | Low Q   | --> | Low W (1)  | ← 少なめ
                  +---------+     +------------+

  サーキットブレーカー:
  [SMS Worker] ---> [Twilio API]
       |                 |
       |            障害検知 (5回連続失敗)
       |                 |
       v                 v
  [Circuit OPEN] ← サーキットブレーカー発動
       |           (60秒間 SMS 送信停止)
       |
       v
  [Half-Open] → テスト送信 → 成功 → [Circuit CLOSED]
                              → 失敗 → [Circuit OPEN] (再度待機)
```

### 5.3 配信保証レベル

```
通知の種類と配信保証の対応表:

+-------------------+----------+--------------------+------------------+
| 通知の種類         | 保証レベル | 配信戦略            | 例                |
+-------------------+----------+--------------------+------------------+
| セキュリティ       | 最高      | マルチチャネル同時   | 2FA コード        |
|                   |          | 確認応答必須         | 不正アクセス検知  |
+-------------------+----------+--------------------+------------------+
| トランザクション   | 高       | フォールバック付き   | 決済完了         |
|                   |          | Push→Email→SMS      | 発送通知         |
+-------------------+----------+--------------------+------------------+
| ソーシャル         | 中       | 最適チャネル1つ     | いいね           |
|                   |          | 集約配信可          | コメント          |
+-------------------+----------+--------------------+------------------+
| マーケティング     | 低       | ベストエフォート    | キャンペーン      |
|                   |          | リトライなし        | レコメンド        |
+-------------------+----------+--------------------+------------------+

マルチチャネルフォールバック戦略:

  [セキュリティ通知 "2FA コード"]
       |
       v
  Push送信 → 30秒以内に開封確認？ → Yes → 完了
       |                           → No
       v
  SMS送信 → 60秒以内に到達？ → Yes → 完了
       |                      → No
       v
  音声通話 → コード読み上げ → 完了
```

---

## 6. 通知チャネルの比較

| チャネル | 到達率 | 遅延 | コスト/通 | 文字数制限 | リッチコンテンツ | ユースケース |
|---------|--------|------|----------|-----------|---------------|-------------|
| iOS Push (APNs) | 高 (85-95%) | < 1秒 | 無料 | 4KB | 画像・アクション | リアルタイムアラート |
| Android Push (FCM) | 高 (85-95%) | < 1秒 | 無料 | 4KB | 画像・アクション | リアルタイムアラート |
| メール | 中 (60-80%) | 秒-分 | ~$0.001 | 無制限 | HTML・添付 | 詳細通知・レポート |
| SMS | 極高 (98%+) | < 3秒 | ~$0.05 | 160字 (70字日本語) | なし | 2FA・重要アラート |
| アプリ内通知 | 極高 (アクティブ時) | 即時 | 無料 | 無制限 | 自由 | UX誘導・プロモーション |
| Web Push | 中 (40-60%) | < 2秒 | 無料 | 制限あり | 画像 | ブラウザ利用者向け |

| 要素 | APNs | FCM | 比較ポイント |
|------|------|-----|-------------|
| プロトコル | HTTP/2 | HTTP/1.1 REST | APNs は HTTP/2 必須 |
| 認証 | JWT (P8キー) | OAuth 2.0 (サービスアカウント) | FCM は Google Cloud 統合 |
| ペイロード上限 | 4KB | 4KB | 同等 |
| トピック送信 | サポート | サポート | 両方対応 |
| サイレントプッシュ | content-available | data メッセージ | バックグラウンド更新 |
| 優先度制御 | 5 (低) / 10 (高) | normal / high | 省電力への配慮 |
| フィードバック | HTTP レスポンス + 410 | HTTP レスポンス + 404 | 無効トークン検知方法が異なる |
| レート制限 | 非公開 (推定 ~2000/秒) | 公式なし (推定 ~1000/秒) | バーストに注意 |

---

## 7. 監視とオブザーバビリティ

### 7.1 主要メトリクス

```
通知システムの監視ダッシュボード設計

[配信メトリクス]
  - notification_sent_total{channel, priority, status}
  - notification_delivery_latency{channel, percentile}
  - notification_retry_total{channel, attempt}
  - notification_dead_lettered_total{channel, error_code}

[ビジネスメトリクス]
  - notification_open_rate{channel, category}    -- 開封率
  - notification_click_rate{channel, category}   -- クリック率
  - notification_unsubscribe_rate{channel}       -- 配信停止率
  - notification_opt_out_rate{channel}           -- オプトアウト率

[インフラメトリクス]
  - kafka_consumer_lag{topic, consumer_group}    -- キューの滞留
  - worker_processing_rate{channel}              -- 処理レート
  - redis_memory_usage                           -- Redis メモリ
  - provider_api_latency{provider}               -- プロバイダ遅延
  - provider_error_rate{provider}                -- プロバイダエラー率

[アラート条件]
  CRITICAL:
    - 配信成功率 < 90% (5分間)
    - キュー滞留 > 100,000 件
    - DLQ メッセージ > 1,000 件/時
    - プロバイダ API エラー率 > 50%

  WARNING:
    - 配信遅延 p99 > 30秒
    - レート制限拒否率 > 20%
    - 配信停止率が前日比 200% 以上
```

### 7.2 配信追跡の実装

```python
# コード例 8: 配信イベント追跡パイプライン
from datetime import datetime

class DeliveryTracker:
    """
    通知の配信ライフサイクルを追跡する。

    イベントフロー:
    SENT → DELIVERED → OPENED → CLICKED
                    → BOUNCED (バウンス)
                    → COMPLAINED (苦情)
    """

    def __init__(self, event_store, analytics_producer):
        self.event_store = event_store
        self.analytics_producer = analytics_producer

    async def track_event(self, notification_id: str, event_type: str,
                           metadata: dict = None):
        """配信イベントを記録"""
        event = {
            "notification_id": notification_id,
            "event_type": event_type,  # sent, delivered, opened, clicked, ...
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        # 1. イベントストアに記録 (永続化)
        await self.event_store.append(event)

        # 2. 分析用 Kafka トピックに送信
        await self.analytics_producer.send(
            topic="notification-events",
            value=event,
        )

        # 3. リアルタイムメトリクス更新
        metrics.increment(
            f"notification.{event_type}",
            tags={"channel": metadata.get("channel", "unknown")},
        )

    async def track_email_webhook(self, webhook_data: dict):
        """
        メールプロバイダ (SES/SendGrid) からの Webhook を処理。

        SES の場合:
        - Delivery: 配信成功
        - Bounce: バウンス (無効アドレス)
        - Complaint: 苦情 (スパム報告)
        - Open: 開封 (トラッキングピクセル)
        - Click: クリック (リンクリダイレクト)
        """
        event_type_map = {
            "Delivery": "delivered",
            "Bounce": "bounced",
            "Complaint": "complained",
            "Open": "opened",
            "Click": "clicked",
        }

        ses_event = webhook_data.get("eventType", "")
        mapped_type = event_type_map.get(ses_event)

        if mapped_type:
            notification_id = webhook_data.get("mail", {}).get(
                "messageId", "unknown"
            )
            await self.track_event(
                notification_id=notification_id,
                event_type=mapped_type,
                metadata={
                    "channel": "email",
                    "provider": "ses",
                    "raw_event": ses_event,
                },
            )

            # バウンスの場合はメールアドレスを無効化
            if mapped_type == "bounced":
                bounce_type = webhook_data.get("bounce", {}).get("bounceType")
                if bounce_type == "Permanent":
                    email = webhook_data["mail"]["destination"][0]
                    await disable_email_address(email)

    async def get_delivery_stats(self, batch_id: str) -> dict:
        """バッチの配信統計を取得"""
        events = await self.event_store.get_by_batch(batch_id)

        stats = {
            "total": 0,
            "sent": 0,
            "delivered": 0,
            "opened": 0,
            "clicked": 0,
            "bounced": 0,
            "complained": 0,
            "failed": 0,
        }

        for event in events:
            event_type = event["event_type"]
            if event_type in stats:
                stats[event_type] += 1
            stats["total"] = max(stats["total"], stats["sent"])

        # レートの計算
        if stats["sent"] > 0:
            stats["delivery_rate"] = stats["delivered"] / stats["sent"]
            stats["open_rate"] = stats["opened"] / stats["delivered"] if stats["delivered"] > 0 else 0
            stats["click_rate"] = stats["clicked"] / stats["opened"] if stats["opened"] > 0 else 0

        return stats
```

---

## 8. アンチパターン

### アンチパターン 1: 「通知の絨毯爆撃」

```python
# NG: ユーザーのアクションごとに即座に通知を送る
# 結果: ユーザーが通知をオフにしてアプリをアンインストール

class BadNotificationHandler:
    async def on_like(self, post_id: str, liker_id: str):
        # 個別にプッシュ通知を送信
        await send_push(
            user_id=post.author_id,
            title="いいね",
            body=f"{liker.name}さんがいいねしました"
        )
        # 10:00 「Aさんがいいねしました」
        # 10:01 「Bさんがいいねしました」
        # 10:02 「Cさんがいいねしました」
        # ... ユーザーは通知をオフにする


# OK: インテリジェントなバッチングと集約
class GoodNotificationHandler:
    def __init__(self, aggregator: NotificationAggregator):
        self.aggregator = aggregator

    async def on_like(self, post_id: str, liker_id: str):
        # 集約バッファに追加
        await self.aggregator.add(
            user_id=post.author_id,
            category="like",
            notification={
                "data": {
                    "actor_id": liker_id,
                    "actor_name": liker.name,
                    "post_id": post_id,
                },
            },
        )
        # 5分後にまとめて送信:
        # 「A、B、他3人があなたの投稿にいいねしました」

    # さらに:
    # - レート制限で1時間あたりの上限を設定
    # - ユーザーの活動時間帯に合わせて配信
    # - 重要度に応じた配信頻度の調整
```

### アンチパターン 2: 「単一キューで全チャネル処理」

```python
# NG: 1つのキューとワーカーで全チャネルを処理
class BadWorker:
    async def process(self, message):
        channel = message["channel"]
        if channel == "push":
            await self.send_push(message)      # 高速 (< 100ms)
        elif channel == "email":
            await self.send_email(message)     # 中速 (< 1s)
        elif channel == "sms":
            await self.send_sms(message)       # 低速 (1-5s)
            # SMS の遅延が Push/Email をブロック!

        # 問題:
        # 1. SMS プロバイダ障害 → 全チャネルの配信が停止
        # 2. チャネルごとのスケーリングが不可能
        # 3. 優先度の高いPushが低優先度SMSの後に待機


# OK: チャネルごとに独立したキューとワーカー
class GoodArchitecture:
    """
    チャネル分離アーキテクチャ:

    Kafka Topics:
    - notifications.push.critical   → PushWorker (4 instances)
    - notifications.push.normal     → PushWorker (2 instances)
    - notifications.email.normal    → EmailWorker (3 instances)
    - notifications.sms.normal      → SMSWorker (1 instance)

    利点:
    1. 障害分離: SMS が落ちても Push/Email は継続
    2. 独立スケーリング: Push は高スループット、SMS は低スループット
    3. 優先度制御: critical トピックに多くのワーカーを割り当て
    4. 監視の粒度: チャネル別にメトリクスを取得
    """
    pass
```

### アンチパターン 3: 「レート制限なしの通知 API」

```python
# NG: レート制限なしで通知を受け付ける
@app.post("/api/v1/notifications")
async def bad_create_notification(request: NotificationRequest):
    # バリデーションのみで即座にキューに投入
    for user_id in request.user_ids:
        for channel in request.channels:
            await queue.publish({"user_id": user_id, "channel": channel, ...})
    return {"status": "queued"}

    # 問題:
    # 1. バグのあるサービスが無限ループで通知を送信
    # 2. 全ユーザーに大量通知 → UX 崩壊
    # 3. SMS コストが爆発 (1通 $0.05 × 数百万通 = ...)
    # 4. APNs/FCM のレート制限に抵触 → IP がブロックされる


# OK: 多層レート制限
@app.post("/api/v1/notifications")
async def good_create_notification(request: NotificationRequest):
    # 1. API レベル: 呼び出し元サービスのレート制限
    caller = get_caller_service(request)
    if not api_rate_limiter.allow(caller.id):
        raise HTTPException(429, "API rate limit exceeded")

    # 2. ユーザーレベル: ユーザーあたりのレート制限
    for user_id in request.user_ids:
        for channel in request.channels:
            if not user_rate_limiter.allow(user_id, channel):
                continue  # スキップ (ログに記録)

    # 3. グローバルレベル: システム全体の安全弁
    if not global_rate_limiter.allow():
        raise HTTPException(503, "System overloaded")

    return {"status": "queued"}
```

### アンチパターン 4: 「配信状況の確認なし (Fire and Forget)」

```python
# NG: 送信したら終わり、結果を確認しない
class BadSender:
    async def send_push(self, message):
        try:
            await apns_client.send(message)
            # 成功扱い。実際は APNs に到達しただけで
            # デバイスに届いたかは不明
        except Exception:
            pass  # エラーも無視


# OK: 配信結果を追跡し、フィードバックループを形成
class GoodSender:
    async def send_push(self, message):
        result = await apns_client.send(message)

        # 配信結果を記録
        await delivery_tracker.track_event(
            notification_id=message["id"],
            event_type="sent" if result.success else "failed",
            metadata={"provider_id": result.provider_message_id},
        )

        # 無効トークンの処理
        if result.error_code == "INVALID_TOKEN":
            await device_token_service.invalidate(message["device_token"])

        # 失敗時のリトライ
        if not result.success and result.should_retry:
            await retry_queue.publish(message)

        # メトリクス更新
        metrics.increment("push.sent", tags={"success": str(result.success)})
```

---

## 9. 実践演習

### 演習 1（基礎）: 通知設定 API の実装

**課題**: ユーザーが自分の通知設定を管理できる REST API を実装してください。

```python
# 要件:
# 1. GET /api/v1/users/{user_id}/notification-preferences
#    → ユーザーの通知設定を取得
# 2. PUT /api/v1/users/{user_id}/notification-preferences
#    → 通知設定を更新
# 3. チャネル別のオン/オフ設定
# 4. カテゴリ別の設定（marketing, transaction, social）
# 5. Quiet Hours の設定

# スケルトンコード:
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class QuietHours(BaseModel):
    enabled: bool = False
    start: str = "23:00"  # HH:MM
    end: str = "07:00"
    timezone: str = "Asia/Tokyo"

class ChannelSettings(BaseModel):
    push: bool = True
    email: bool = True
    sms: bool = True
    in_app: bool = True

class CategorySettings(BaseModel):
    marketing: bool = False
    transaction: bool = True
    social: bool = True
    security: bool = True  # セキュリティは常にTrue推奨

class NotificationPreferences(BaseModel):
    channels: ChannelSettings = ChannelSettings()
    categories: CategorySettings = CategorySettings()
    quiet_hours: QuietHours = QuietHours()

@app.get("/api/v1/users/{user_id}/notification-preferences")
async def get_preferences(user_id: str):
    # TODO: DBからユーザー設定を取得
    # TODO: キャッシュの活用
    # TODO: デフォルト値の設定
    pass

@app.put("/api/v1/users/{user_id}/notification-preferences")
async def update_preferences(user_id: str, prefs: NotificationPreferences):
    # TODO: バリデーション
    # TODO: DBに保存
    # TODO: キャッシュの無効化
    # TODO: 変更イベントの発行
    pass
```

**期待される出力**:
```json
GET /api/v1/users/user-123/notification-preferences

{
  "channels": {
    "push": true,
    "email": true,
    "sms": false,
    "in_app": true
  },
  "categories": {
    "marketing": false,
    "transaction": true,
    "social": true,
    "security": true
  },
  "quiet_hours": {
    "enabled": true,
    "start": "23:00",
    "end": "07:00",
    "timezone": "Asia/Tokyo"
  }
}
```

---

### 演習 2（応用）: マルチチャネルフォールバック配信の実装

**課題**: 重要な通知を確実に届けるため、複数チャネルを順番に試行するフォールバック機構を実装してください。

```python
# 要件:
# 1. Push → Email → SMS の順にフォールバック
# 2. 各チャネルで配信後、一定時間以内に開封確認がなければ次へ
# 3. タイムアウト設定はチャネルごとにカスタマイズ可能
# 4. 最終手段 (SMS) が失敗したらアラート

# スケルトンコード:
class FallbackDeliveryService:
    """
    マルチチャネルフォールバック配信サービス。

    フロー:
    1. Push 送信 → 30秒待ち → 開封確認あり → 完了
                                開封確認なし → 2へ
    2. Email 送信 → 5分待ち → 開封確認あり → 完了
                               開封確認なし → 3へ
    3. SMS 送信 → 完了 (SMS は開封確認なし)
    """

    FALLBACK_CHAIN = [
        {"channel": "push", "timeout_seconds": 30},
        {"channel": "email", "timeout_seconds": 300},
        {"channel": "sms", "timeout_seconds": 0},  # 最終手段
    ]

    async def deliver_with_fallback(self, user_id: str,
                                     notification: dict) -> str:
        # TODO: FALLBACK_CHAIN に沿って順番に配信
        # TODO: 各チャネルでタイムアウト後に次のチャネルへ
        # TODO: 全チャネル失敗時のアラート
        # TODO: 配信結果のログ記録
        pass
```

**期待される出力**:
```
配信ログ:
[2024-01-15 10:00:00] Push sent to user-456 (notification: order-confirm-789)
[2024-01-15 10:00:30] Push not opened within 30s, falling back to email
[2024-01-15 10:00:31] Email sent to user-456
[2024-01-15 10:05:31] Email opened by user-456 — delivery complete

結果: {"channel_used": "email", "attempts": 2, "total_time": "5m31s"}
```

---

### 演習 3（発展）: 通知分析ダッシュボードのバックエンド設計

**課題**: 通知の配信パフォーマンスを分析するダッシュボードのバックエンド API を設計・実装してください。

```python
# 要件:
# 1. チャネル別の配信成功率（日次/週次/月次）
# 2. カテゴリ別の開封率・クリック率
# 3. 時間帯別の配信パフォーマンス分析
# 4. A/B テスト結果の統計分析
# 5. ユーザーセグメント別のエンゲージメント分析
# 6. データは ClickHouse に格納されていると仮定

# スケルトンコード:
from datetime import date, timedelta

class NotificationAnalytics:
    """通知分析サービス"""

    async def get_channel_performance(self, start_date: date,
                                       end_date: date) -> dict:
        """
        チャネル別の配信パフォーマンスを取得。

        返り値の例:
        {
          "push": {"sent": 5000000, "delivered": 4750000,
                   "opened": 2375000, "delivery_rate": 0.95,
                   "open_rate": 0.50},
          "email": {"sent": 3000000, "delivered": 2400000,
                    "opened": 720000, "delivery_rate": 0.80,
                    "open_rate": 0.30},
          ...
        }
        """
        # TODO: ClickHouse からデータを集計
        # TODO: 日次/週次/月次の粒度で返す
        # TODO: 前期間との比較
        pass

    async def get_ab_test_results(self, test_id: str) -> dict:
        """
        A/B テスト結果の統計分析。

        返り値の例:
        {
          "test_id": "test-001",
          "variants": {
            "A": {"sent": 50000, "opened": 15000, "clicked": 3000,
                  "open_rate": 0.30, "click_rate": 0.20},
            "B": {"sent": 50000, "opened": 18000, "clicked": 4500,
                  "open_rate": 0.36, "click_rate": 0.25},
          },
          "winner": "B",
          "confidence": 0.97,
          "p_value": 0.003,
        }
        """
        # TODO: 統計検定 (カイ二乗検定 or Z検定) の実装
        # TODO: 統計的有意性の判定
        pass

    async def get_hourly_heatmap(self, channel: str,
                                  days: int = 7) -> list:
        """
        時間帯別の配信パフォーマンスヒートマップデータ。

        返り値: 24時間 × 7日 のマトリクス
        """
        # TODO: 時間帯別の開封率を計算
        # TODO: 最適送信時間帯の推定
        pass
```

**期待される出力**:
```json
GET /api/v1/analytics/channel-performance?start=2024-01-01&end=2024-01-31

{
  "period": {"start": "2024-01-01", "end": "2024-01-31"},
  "channels": {
    "push": {
      "sent": 155000000,
      "delivered": 147250000,
      "opened": 73625000,
      "clicked": 22087500,
      "delivery_rate": 0.95,
      "open_rate": 0.50,
      "click_rate": 0.15,
      "trend": "+2.3%"
    },
    "email": {
      "sent": 93000000,
      "delivered": 74400000,
      "opened": 22320000,
      "clicked": 6696000,
      "delivery_rate": 0.80,
      "open_rate": 0.30,
      "click_rate": 0.09,
      "trend": "-1.1%"
    }
  }
}
```

---

## 10. FAQ

### Q1: APNs のデバイストークンが無効になった場合はどう対処しますか？

**A:** APNs は無効なトークンに対して HTTP 410 (Gone) を返します。以下の対応を行います。

1. **即座にトークンを無効化**: DB 上のトークンに `is_active = false` を設定。次回の配信対象から除外する。
2. **フィードバックサービスの監視**: APNs の HTTP/2 レスポンスをリアルタイムで処理し、無効トークンのリストを収集する。
3. **再登録の促進**: アプリ起動時に毎回 `registerForRemoteNotifications()` を呼び出し、最新のトークンをサーバーに送信する設計にする。
4. **定期クリーンアップ**: 90日以上更新されていないトークンを定期的に無効化するバッチジョブを実行する。

```python
# トークン管理の実装例
async def handle_apns_response(device_token: str, status_code: int,
                                response_body: dict):
    if status_code == 410:  # Gone
        await db.execute(
            "UPDATE device_tokens SET is_active = FALSE "
            "WHERE device_token = :token",
            {"token": device_token},
        )
        # キャッシュも無効化
        user_id = await get_user_by_token(device_token)
        await cache.delete(f"devices:{user_id}")
    elif status_code == 400 and response_body.get("reason") == "BadDeviceToken":
        # 不正なトークン形式
        await db.execute(
            "DELETE FROM device_tokens WHERE device_token = :token",
            {"token": device_token},
        )
```

### Q2: 通知の A/B テストはどう実装しますか？

**A:** 以下のアーキテクチャで実装します。

1. **テンプレートのバリアント管理**: 1つのテンプレート ID に複数のバリアント (A/B/C...) を紐づけ、それぞれ異なる文言・レイアウトを設定する。
2. **ユーザーの振り分け**: ユーザー ID のハッシュ (consistent hashing) で A/B グループに振り分ける。同じユーザーが常に同じグループに入ることで一貫性を保つ。
3. **メトリクス収集**: 開封率、クリック率、コンバージョン率をバリアントごとに記録。イベントは Kafka 経由で ClickHouse に蓄積。
4. **統計的有意性の判定**: カイ二乗検定または Z 検定で p-value を計算し、95% 信頼区間で有意差が出たら勝者バリアントに統一する。
5. **自動最適化 (MAB)**: Multi-Armed Bandit アルゴリズム (Thompson Sampling) を使って、テスト中もパフォーマンスの良いバリアントに多くのトラフィックを配分する。

```python
# A/B テスト振り分けの実装例
import hashlib

def assign_variant(user_id: str, test_id: str,
                   variants: list[str] = ["A", "B"]) -> str:
    """ユーザーをバリアントに一貫して振り分ける"""
    hash_input = f"{user_id}:{test_id}"
    hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
    index = hash_value % len(variants)
    return variants[index]
```

### Q3: グローバルサービスでのタイムゾーン対応はどうしますか？

**A:** ユーザーのタイムゾーンに基づいたスケジューリングを行います。

- **ユーザープロファイルにタイムゾーンを保存**: IANA タイムゾーン名 (例: `Asia/Tokyo`) を使用。IP ジオロケーションから推定する場合は確認を求める。
- **ローカル時間でのスケジュール**: 「朝9時に送信」→ 各ユーザーのローカル時間の9時に送信。UTC に変換してスケジューラに登録。
- **Quiet Hours のタイムゾーン適用**: 「深夜0時から朝7時は送信しない」をユーザーのタイムゾーンで計算。
- **スケジューラの実装**: UTC ベースのスケジューラが1分ごとに「次の1分間に送信すべき通知」をクエリし、キューに投入する。
- **DST (夏時間) の考慮**: `pytz` や `zoneinfo` ライブラリで DST の切り替えを正しく処理する。

### Q4: 通知の開封率を正確に測定するにはどうしますか？

**A:** チャネルごとに異なるトラッキング手法を使います。

| チャネル | トラッキング手法 | 精度 |
|---------|----------------|------|
| メール | 1x1 トラッキングピクセル + リンクリダイレクト | 中 (画像ブロック時は検知不可) |
| Push (iOS) | `UNNotificationServiceExtension` でサーバーに通知 | 高 |
| Push (Android) | FCM Data Message + アプリ内ハンドラ | 高 |
| アプリ内 | 表示時にイベント送信 | 極高 |
| SMS | 短縮URLのクリック追跡のみ (開封は不可) | 低 |

注意点:
- Apple の Mail Privacy Protection (iOS 15+) はプロキシ経由で画像を読み込むため、メールの開封率が実態より高く出る
- プッシュ通知の「表示」と「タップ」を区別して計測する

### Q5: 大量配信 (ブロードキャスト) はどう処理しますか？

**A:** 全ユーザーへの一斉配信は特別な処理が必要です。

1. **セグメント分割**: 全ユーザーを N 個のセグメントに分割し、順次配信。一度に全送信するとプロバイダの制限に抵触する。
2. **Kafka パーティション活用**: ユーザー ID でパーティショニングし、複数ワーカーで並列処理。
3. **段階的ロールアウト**: まず 1% に配信して問題がないか確認、その後 10% → 50% → 100% と段階的に拡大。
4. **キャンセル機能**: 配信中にエラー率が閾値を超えたら自動停止。手動キャンセルも可能にする。
5. **プリウォーミング**: 大量配信の前にプロバイダに通知し、レート制限の一時緩和を依頼する (APNs/SES で可能)。

```python
# 段階的ロールアウトの実装例
class BroadcastService:
    ROLLOUT_STAGES = [0.01, 0.10, 0.50, 1.00]

    async def broadcast(self, notification: dict,
                         auto_rollout: bool = True):
        total_users = await get_total_user_count()

        for stage_pct in self.ROLLOUT_STAGES:
            target_count = int(total_users * stage_pct)
            users = await get_user_segment(
                offset=0, limit=target_count
            )

            batch_id = await send_to_users(users, notification)

            if auto_rollout:
                # 配信結果を監視
                await asyncio.sleep(300)  # 5分待機
                stats = await get_batch_stats(batch_id)

                if stats["error_rate"] > 0.05:  # 5% 超エラー
                    await cancel_broadcast(batch_id)
                    raise BroadcastError(
                        f"Error rate too high: {stats['error_rate']}"
                    )
            else:
                # 手動承認を待つ
                await wait_for_approval(batch_id)
```

---

## 11. まとめ

| 設計要素 | 選択 | 理由 |
|---------|------|------|
| メッセージキュー | Kafka (チャネル別 + 優先度別トピック) | 高スループット + 障害分離 + 優先度制御 |
| 重複排除 | Redis SETNX (TTL: 24h) | 高速なべき等性チェック、メモリ効率 |
| レート制限 | Redis Sorted Set + Lua スクリプト | 正確なスライディングウィンドウ、アトミック操作 |
| テンプレート | Jinja2 + DB管理 + Redis キャッシュ | 多言語・多チャネル対応、バージョン管理 |
| リトライ | 指数バックオフ + ジッター + DLQ | 信頼性確保、プロバイダ過負荷防止 |
| 配信追跡 | Kafka → ClickHouse → Grafana | リアルタイム分析 + 長期保存 |
| チャネル分離 | チャネル別キュー + ワーカー | 障害分離 + 独立スケーリング |
| ユーザー設定 | PostgreSQL + Redis キャッシュ | 永続性 + 高速読み取り |
| デバイストークン | PostgreSQL + Redis キャッシュ | トークン管理 + 高速ルックアップ |
| 監視 | Prometheus + Grafana + PagerDuty | メトリクス + ダッシュボード + アラート |

---

## 12. 設計面接での回答フレームワーク

```
通知システムの設計面接で聞かれるポイント:

1. 要件の明確化 (5分)
   - ユーザー規模は？ (100M DAU → 高スケール設計が必要)
   - 対応チャネルは？ (Push/Email/SMS/In-App)
   - 配信保証レベルは？ (at-least-once)
   - レイテンシ要件は？ (< 30秒)

2. 高レベル設計 (10分)
   - API → オーケストレーター → キュー → ワーカー → プロバイダ
   - チャネル別の分離
   - テンプレート + ユーザー設定

3. 詳細設計 (15分)
   - データモデル
   - レート制限アルゴリズム
   - リトライ戦略
   - 重複排除

4. スケーラビリティ (5分)
   - 水平スケーリング
   - Kafka パーティショニング
   - Redis クラスター

5. 運用 (5分)
   - 監視とアラート
   - 配信分析
   - A/B テスト
```

---

## 次に読むべきガイド

- [チャットシステム設計](./01-chat-system.md) — リアルタイムメッセージングとの連携パターン
- [URL 短縮サービス設計](./00-url-shortener.md) — 通知内リンクの短縮・トラッキング
- [レートリミッター設計](./03-rate-limiter.md) — 通知レート制限の詳細アルゴリズム
- [メッセージキュー](../01-components/02-message-queue.md) — Kafka / RabbitMQ の詳細
- [イベント駆動アーキテクチャ](../02-architecture/03-event-driven.md) — イベント駆動設計の原則
- [Observer パターン](../../design-patterns-guide/docs/02-behavioral/00-observer.md) — Pub/Sub パターンの基礎
- [Strategy パターン](../../design-patterns-guide/docs/02-behavioral/01-strategy.md) — チャネルアダプタの設計パターン
- [API 設計](../../clean-code-principles/docs/03-practices-advanced/03-api-design.md) — 通知 API のベストプラクティス

---

## 参考文献

1. Xu, A. (2020). *System Design Interview: An Insider's Guide*. Chapter 10: Design a Notification System. Byte Code LLC. https://www.systemdesigninterview.com/
2. Apple Inc. (2024). "Sending Notification Requests to APNs." *Apple Developer Documentation*. https://developer.apple.com/documentation/usernotifications/sending-notification-requests-to-apns
3. Google. (2024). "Firebase Cloud Messaging Architecture." *Firebase Documentation*. https://firebase.google.com/docs/cloud-messaging/fcm-architecture
4. Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly Media. Chapter 11: Stream Processing.
5. Amazon Web Services. (2024). "Amazon SES Developer Guide." https://docs.aws.amazon.com/ses/latest/dg/
6. Twilio. (2024). "Programmable Messaging Documentation." https://www.twilio.com/docs/messaging
7. Shopify Engineering. (2018). "Building a Notification System at Scale." https://shopify.engineering/

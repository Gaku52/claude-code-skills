# モノリス vs マイクロサービス

> モノリスとマイクロサービスそれぞれの特性を正確に理解し、プロジェクトの規模・チーム体制・ビジネス要件に応じた最適なアーキテクチャ選定ができる判断力を身につける。

## この章で学ぶこと

1. **構造的な違い** --- モノリスとマイクロサービスのアーキテクチャ、デプロイ、データ管理の本質的な違い
2. **トレードオフ分析** --- 各アーキテクチャの利点と欠点、適用条件の理解
3. **段階的移行** --- Strangler Fig パターンによるモノリスからマイクロサービスへの移行戦略
4. **設計パターン** --- サービス間通信、分散トランザクション (Saga)、API Gateway の実装
5. **判断フレームワーク** --- チーム規模・ドメイン成熟度・技術要件に基づく選定基準

## 前提知識

| トピック | 必要レベル | 参照先 |
|---------|-----------|--------|
| Web アプリケーション開発 | 中級 | [プログラミング](../../02-programming/) |
| REST API | 基礎 | [Web 基礎](../../04-web-and-network/) |
| データベース | 基礎 | [DBスケーリング](../01-components/04-database-scaling.md) |
| メッセージキュー | 基礎 | [メッセージキュー](../01-components/02-message-queue.md) |

---

## 0. WHY --- なぜアーキテクチャ選択が重要か

### 0.1 アーキテクチャ選択の影響範囲

```
アーキテクチャの選択は以下の全てに影響する:

  ┌──────────────────────────────────────────────┐
  │                                              │
  │  1. 開発速度                                  │
  │     └─ 初期: モノリス > マイクロサービス        │
  │        長期: マイクロサービス > モノリス        │
  │                                              │
  │  2. デプロイ頻度                               │
  │     └─ モノリス: 週1-2回                      │
  │        マイクロサービス: 日数回〜数十回         │
  │                                              │
  │  3. チーム自律性                               │
  │     └─ モノリス: 全員が同一コードベース         │
  │        マイクロサービス: チームがサービスを所有  │
  │                                              │
  │  4. スケーラビリティ                           │
  │     └─ モノリス: 全体を一括スケール            │
  │        マイクロサービス: ボトルネックのみ拡張    │
  │                                              │
  │  5. 障害の影響範囲                             │
  │     └─ モノリス: 1つのバグが全体に影響         │
  │        マイクロサービス: 障害が局所化される      │
  │                                              │
  │  6. 運用コスト                                 │
  │     └─ モノリス: 低 (単一プロセス)             │
  │        マイクロサービス: 高 (分散システム運用)   │
  │                                              │
  └──────────────────────────────────────────────┘
```

### 0.2 実際のトレードオフ

| 指標 | モノリス | マイクロサービス |
|------|---------|---------------|
| 初期開発コスト | $100K | $300K-500K |
| 月間運用コスト (小規模) | $500 | $2,000-5,000 |
| デプロイ頻度 | 週 1-2 回 | 日 5-20 回 |
| 障害復旧時間 | 数分 (再起動) | 数秒 (サービス単位) |
| 新機能リリース速度 (50人チーム) | 遅い (コンフリクト多) | 速い (独立デプロイ) |
| 必要なインフラ知識 | 基礎的 | 高度 (K8s, Service Mesh) |
| モニタリング複雑度 | 低 | 高 (分散トレーシング必須) |

---

## 1. モノリスとは

### 1.1 モノリスアーキテクチャ

```
  ┌────────────────────────────────────────┐
  │           モノリスアプリケーション        │
  │                                        │
  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌───────┐ │
  │  │ User │ │Order │ │ Pay- │ │Notifi-│ │
  │  │Module│ │Module│ │ ment │ │cation │ │
  │  └──┬───┘ └──┬───┘ └──┬───┘ └───┬───┘ │
  │     │        │        │         │     │
  │  ┌──▼────────▼────────▼─────────▼──┐  │
  │  │        共有データベース           │  │
  │  └─────────────────────────────────┘  │
  │                                        │
  │  1つのデプロイ単位、1つのプロセス       │
  │  1つのコードベース、1つのCI/CDパイプ    │
  └────────────────────────────────────────┘
```

### 1.2 モノリスの種類

```
1. シンプルモノリス (Simple Monolith)
   └─ 全コードが1つのパッケージ。小規模チームに最適。

2. モジュラーモノリス (Modular Monolith)  ← 推奨
   └─ 内部がモジュールに分割。モジュール間は定義された
      インターフェースで通信。DBスキーマもモジュール別。
      将来のマイクロサービス化の準備になる。
      例: Shopify, Basecamp

3. 分散モノリス (Distributed Monolith)  ← アンチパターン
   └─ サービスに分割したが密結合のまま。
      マイクロサービスのデメリットだけを享受。

  ┌──────────────────────────────────────────────┐
  │         モジュラーモノリスの内部構造           │
  │                                              │
  │  ┌──────────────┐    ┌──────────────┐        │
  │  │ User Module  │    │ Order Module │        │
  │  │              │    │              │        │
  │  │ UserService  │◄──►│ OrderService │        │
  │  │ UserRepo     │    │ OrderRepo    │        │
  │  │              │    │              │        │
  │  │ user_*       │    │ order_*      │        │
  │  │ テーブル群    │    │ テーブル群    │        │
  │  └──────────────┘    └──────────────┘        │
  │                                              │
  │  ルール:                                      │
  │  - 他モジュールのテーブルに直接アクセス禁止    │
  │  - 公開インターフェース経由のみ                │
  │  - モジュール内は自由に設計                    │
  └──────────────────────────────────────────────┘
```

---

## 2. マイクロサービスとは

### 2.1 マイクロサービスアーキテクチャ

```
  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐
  │  User   │    │  Order  │    │ Payment │    │ Notif.   │
  │ Service │    │ Service │    │ Service │    │ Service  │
  │         │←──→│         │←──→│         │←──→│          │
  └────┬────┘    └────┬────┘    └────┬────┘    └────┬─────┘
       │              │              │               │
  ┌────▼────┐    ┌────▼────┐    ┌────▼────┐    ┌────▼─────┐
  │ User DB │    │Order DB │    │ Pay DB  │    │Notif DB  │
  │(Postgres)│   │(Postgres)│   │(Postgres)│   │ (Redis)  │
  └─────────┘    └─────────┘    └─────────┘    └──────────┘

  各サービス: 独立デプロイ、独立DB、独立スケール
  通信: REST / gRPC / Message Queue
  所有: 1チーム = 1-3サービス (Two Pizza Team)
```

### 2.2 マイクロサービスの特性

```
マイクロサービスの 9つの特性 (Martin Fowler):

  1. サービスによるコンポーネント化
     └─ ライブラリではなく独立プロセスとして分割

  2. ビジネスケイパビリティに基づく組織化
     └─ 技術レイヤーではなくビジネス機能で分割
        NG: フロントチーム / バックエンドチーム / DBチーム
        OK: ユーザーチーム / 注文チーム / 決済チーム

  3. プロジェクトではなくプロダクト
     └─ チームがサービスのライフサイクル全体を所有

  4. スマートエンドポイントとダムパイプ
     └─ ロジックはサービス内、通信は軽量 (REST/gRPC/MQ)

  5. 分散ガバナンス
     └─ サービスごとに技術スタックを選択可能

  6. 分散データ管理
     └─ サービスごとに独立データストア (Database per Service)

  7. インフラ自動化
     └─ CI/CD, IaC, コンテナオーケストレーション

  8. 障害を前提とした設計
     └─ サーキットブレーカー、リトライ、フォールバック

  9. 進化的な設計
     └─ サービスの置き換え・廃止が容易
```

---

## 3. コード例で比較

### コード例 1: モノリスでの注文処理

```python
# monolith/order_service.py
# モノリス: 全てが1つのプロセス内

class OrderService:
    def __init__(self, db_session):
        self.db = db_session

    def create_order(self, user_id: str, items: list) -> dict:
        # 同一DB内でトランザクション → ACID 保証
        with self.db.begin():
            # 1. ユーザー確認（同一DB）
            user = self.db.query(User).get(user_id)
            if not user:
                raise ValueError("User not found")

            # 2. 在庫確認（同一DB）
            for item in items:
                product = self.db.query(Product).get(item["product_id"])
                if product.stock < item["quantity"]:
                    raise ValueError(f"Insufficient stock: {product.name}")
                product.stock -= item["quantity"]

            # 3. 注文作成（同一DB）
            order = Order(user_id=user_id, items=items, status="pending")
            self.db.add(order)

            # 4. 決済処理（同一プロセス内で呼び出し）
            payment = PaymentService(self.db)
            payment.charge(user_id, order.total_amount)

            # 5. 通知送信（同一プロセス内で呼び出し）
            notification = NotificationService(self.db)
            notification.send_order_confirmation(user_id, order.id)

            order.status = "confirmed"
            return {"order_id": order.id, "status": order.status}

        # メリット:
        # - 1つのトランザクションで ACID 保証
        # - デバッグが容易 (スタックトレースで追跡可能)
        # - テストがシンプル (モックが少ない)
        #
        # デメリット:
        # - 全モジュールが密結合
        # - 決済処理が遅いと注文全体がブロック
        # - 通知サービスの障害が注文処理を失敗させる
```

### コード例 2: マイクロサービスでの注文処理

```python
# microservices/order_service/app.py
# マイクロサービス: 各サービスが独立

import httpx
import json
from kafka import KafkaProducer

class OrderService:
    def __init__(self, db, producer: KafkaProducer):
        self.db = db
        self.producer = producer
        self.user_service_url = "http://user-service:8080"
        self.inventory_service_url = "http://inventory-service:8080"

    async def create_order(self, user_id: str, items: list) -> dict:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # 1. ユーザー確認（HTTPで別サービスに問い合わせ）
            resp = await client.get(
                f"{self.user_service_url}/users/{user_id}"
            )
            if resp.status_code != 200:
                raise ValueError("User not found")

            # 2. 在庫確認（HTTPで別サービスに問い合わせ）
            resp = await client.post(
                f"{self.inventory_service_url}/reserve",
                json={"items": items}
            )
            if resp.status_code != 200:
                raise ValueError("Insufficient stock")
            reservation_id = resp.json()["reservation_id"]

        # 3. 注文作成（自サービスのDB）
        order = Order(
            user_id=user_id,
            items=items,
            status="pending",
            reservation_id=reservation_id,
        )
        self.db.add(order)
        self.db.commit()

        # 4. イベント発行（非同期で決済・通知を処理）
        self.producer.send("order-events", json.dumps({
            "event_type": "order_created",
            "order_id": order.id,
            "user_id": user_id,
            "amount": order.total_amount,
            "reservation_id": reservation_id,
        }).encode())

        return {"order_id": order.id, "status": "pending"}

    # メリット:
    # - 独立デプロイ・スケール (注文サービスだけ 10台に拡張可能)
    # - 技術選択の自由 (決済は Java, 通知は Go でも OK)
    # - 障害の局所化 (通知障害が注文を止めない)
    #
    # デメリット:
    # - 分散トランザクション (在庫予約 + 注文 + 決済の整合性)
    # - ネットワークレイテンシ (HTTP 呼び出しのオーバーヘッド)
    # - テストの複雑化 (サービス間の統合テスト)
    # - 結果整合性 (注文ステータスの更新が遅延)
```

### コード例 3: Saga パターンによる分散トランザクション

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Any
import asyncio
import logging

logger = logging.getLogger(__name__)

class SagaStepStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"

@dataclass
class SagaStep:
    name: str
    action: Callable        # 正方向の処理
    compensation: Callable  # 補償処理（ロールバック）
    status: SagaStepStatus = SagaStepStatus.PENDING
    result: Any = None

class SagaOrchestrator:
    """
    Orchestration Saga: 中央のオーケストレーターが各ステップを制御

    フロー:
    1. 在庫予約 → 成功
    2. 決済処理 → 成功
    3. 注文確定 → 失敗！
    4. → 決済を返金 (補償)
    5. → 在庫予約を解除 (補償)
    """

    def __init__(self, saga_id: str):
        self.saga_id = saga_id
        self.steps: list[SagaStep] = []
        self.completed_steps: list[SagaStep] = []

    def add_step(
        self,
        name: str,
        action: Callable,
        compensation: Callable
    ):
        self.steps.append(SagaStep(name, action, compensation))

    async def execute(self, context: dict) -> dict:
        """Saga を実行"""
        logger.info(f"[SAGA {self.saga_id}] Starting saga")

        for step in self.steps:
            try:
                logger.info(f"[SAGA {self.saga_id}] Executing: {step.name}")
                result = await step.action(context)
                context.update(result or {})
                step.status = SagaStepStatus.COMPLETED
                step.result = result
                self.completed_steps.append(step)
            except Exception as e:
                logger.error(
                    f"[SAGA {self.saga_id}] Failed: {step.name} - {e}"
                )
                step.status = SagaStepStatus.FAILED
                await self._compensate(context)
                return {
                    "status": "failed",
                    "failed_step": step.name,
                    "error": str(e),
                }

        logger.info(f"[SAGA {self.saga_id}] Completed successfully")
        return {"status": "completed", "context": context}

    async def _compensate(self, context: dict):
        """完了済みステップを逆順に補償"""
        logger.info(f"[SAGA {self.saga_id}] Starting compensation")
        for step in reversed(self.completed_steps):
            try:
                logger.info(
                    f"[SAGA {self.saga_id}] Compensating: {step.name}"
                )
                await step.compensation(context)
                step.status = SagaStepStatus.COMPENSATED
            except Exception as e:
                logger.error(
                    f"[SAGA {self.saga_id}] Compensation failed: "
                    f"{step.name} - {e}"
                )
                # 補償失敗: 手動介入が必要 → DLQ に送信
                await self._send_to_dlq(step, context, e)

    async def _send_to_dlq(self, step, context, error):
        """補償失敗をデッドレターキューに送信"""
        logger.critical(
            f"[SAGA {self.saga_id}] Manual intervention required: "
            f"{step.name}"
        )

# 使用例: 注文処理 Saga
async def create_order_saga(
    user_id: str, items: list, amount: float
):
    saga = SagaOrchestrator(saga_id=f"order-{user_id}-{int(time.time())}")

    saga.add_step(
        "reserve_inventory",
        action=lambda ctx: inventory_service.reserve(ctx["items"]),
        compensation=lambda ctx: inventory_service.cancel_reservation(
            ctx["reservation_id"]
        ),
    )
    saga.add_step(
        "process_payment",
        action=lambda ctx: payment_service.charge(
            ctx["user_id"], ctx["amount"]
        ),
        compensation=lambda ctx: payment_service.refund(
            ctx["payment_id"]
        ),
    )
    saga.add_step(
        "confirm_order",
        action=lambda ctx: order_service.confirm(ctx["order_id"]),
        compensation=lambda ctx: order_service.cancel(ctx["order_id"]),
    )
    saga.add_step(
        "send_notification",
        action=lambda ctx: notification_service.send(
            ctx["user_id"], ctx["order_id"]
        ),
        compensation=lambda ctx: None,  # 通知は補償不要
    )

    context = {
        "user_id": user_id,
        "items": items,
        "amount": amount,
    }
    return await saga.execute(context)
```

### コード例 4: API Gateway パターン

```python
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import httpx
import time
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

app = FastAPI(title="API Gateway")

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SERVICE_REGISTRY = {
    "users":         "http://user-service:8080",
    "orders":        "http://order-service:8080",
    "payments":      "http://payment-service:8080",
    "notifications": "http://notification-service:8080",
}

# ── レートリミッター ──
class RateLimiter:
    """スライディングウィンドウレートリミッター"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        # 古いリクエストを削除
        self.requests[client_id] = [
            t for t in self.requests[client_id]
            if now - t < self.window
        ]
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

# ── サーキットブレーカー ──
class CircuitState(Enum):
    CLOSED = "closed"      # 正常
    OPEN = "open"          # 遮断中
    HALF_OPEN = "half_open"  # 試行中

@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0

    def record_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def is_available(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN: 1回だけ試行

circuit_breakers: dict[str, CircuitBreaker] = {
    name: CircuitBreaker() for name in SERVICE_REGISTRY
}

# ── ルーティング ──
@app.api_route(
    "/{service}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def gateway(service: str, path: str, request: Request):
    """全リクエストをルーティング"""
    # 1. サービス存在チェック
    base_url = SERVICE_REGISTRY.get(service)
    if not base_url:
        raise HTTPException(404, f"Service '{service}' not found")

    # 2. レートリミットチェック
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(429, "Rate limit exceeded")

    # 3. サーキットブレーカーチェック
    cb = circuit_breakers[service]
    if not cb.is_available():
        raise HTTPException(503, f"Service '{service}' is temporarily unavailable")

    # 4. リクエスト転送
    target_url = f"{base_url}/{path}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers={
                    k: v for k, v in request.headers.items()
                    if k.lower() not in ("host", "content-length")
                },
                content=await request.body(),
                params=dict(request.query_params),
            )
            cb.record_success()
            return response.json()

    except (httpx.TimeoutException, httpx.ConnectError) as e:
        cb.record_failure()
        raise HTTPException(502, f"Service '{service}' error: {e}")
```

### コード例 5: サービス間通信のリトライとフォールバック

```python
import httpx
import asyncio
from functools import wraps
from typing import Optional, Any
from dataclasses import dataclass
import time

@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 0.5      # 初回待機 500ms
    max_delay: float = 10.0      # 最大待機 10s
    exponential_base: float = 2  # 指数バックオフの基数
    retry_on_status: set = None  # リトライ対象のステータスコード

    def __post_init__(self):
        if self.retry_on_status is None:
            self.retry_on_status = {502, 503, 504, 429}

class ResilientServiceClient:
    """障害耐性のあるサービス間通信"""

    def __init__(
        self,
        base_url: str,
        timeout: float = 5.0,
        retry_config: RetryConfig = None,
        fallback: Optional[Any] = None,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        self.fallback = fallback
        self._metrics = {
            'total_requests': 0,
            'successful': 0,
            'retries': 0,
            'failures': 0,
            'fallbacks': 0,
        }

    async def get(self, path: str, **kwargs) -> dict:
        return await self._request("GET", path, **kwargs)

    async def post(self, path: str, data: dict = None, **kwargs) -> dict:
        return await self._request("POST", path, json=data, **kwargs)

    async def _request(self, method: str, path: str, **kwargs):
        self._metrics['total_requests'] += 1
        last_error = None

        for attempt in range(self.retry_config.max_retries):
            try:
                async with httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.timeout
                ) as client:
                    resp = await client.request(method, path, **kwargs)

                    # リトライ対象のステータスコードか確認
                    if resp.status_code in self.retry_config.retry_on_status:
                        raise httpx.HTTPStatusError(
                            f"Status {resp.status_code}",
                            request=resp.request,
                            response=resp,
                        )

                    resp.raise_for_status()
                    self._metrics['successful'] += 1
                    return resp.json()

            except (
                httpx.TimeoutException,
                httpx.HTTPStatusError,
                httpx.ConnectError,
            ) as e:
                last_error = e
                self._metrics['retries'] += 1

                if attempt < self.retry_config.max_retries - 1:
                    # 指数バックオフ + ジッター
                    delay = min(
                        self.retry_config.base_delay *
                        (self.retry_config.exponential_base ** attempt),
                        self.retry_config.max_delay
                    )
                    # ジッター: 0.5x〜1.5x のランダム化
                    import random
                    jitter = delay * (0.5 + random.random())
                    print(
                        f"[Retry {attempt+1}/{self.retry_config.max_retries}] "
                        f"{method} {path}: {e}, waiting {jitter:.1f}s"
                    )
                    await asyncio.sleep(jitter)

        # 全リトライ失敗
        self._metrics['failures'] += 1

        # フォールバック
        if self.fallback is not None:
            self._metrics['fallbacks'] += 1
            if callable(self.fallback):
                return self.fallback()
            return self.fallback

        raise last_error

    @property
    def metrics(self) -> dict:
        total = self._metrics['total_requests']
        return {
            **self._metrics,
            'success_rate': (
                self._metrics['successful'] / total if total > 0 else 0
            ),
            'fallback_rate': (
                self._metrics['fallbacks'] / total if total > 0 else 0
            ),
        }

# 使用例
user_client = ResilientServiceClient(
    "http://user-service:8080",
    timeout=3.0,
    retry_config=RetryConfig(max_retries=3, base_delay=0.5),
    fallback={"id": "unknown", "name": "Guest"},  # フォールバック値
)

order_client = ResilientServiceClient(
    "http://order-service:8080",
    timeout=10.0,
    retry_config=RetryConfig(max_retries=2, base_delay=1.0),
)
```

### コード例 6: モジュラーモノリスの実装

```python
"""モジュラーモノリス: モジュール境界を明確にした設計"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

# ── モジュール間のインターフェース定義 ──

class UserModuleInterface(ABC):
    """User モジュールの公開インターフェース"""
    @abstractmethod
    def get_user(self, user_id: str) -> Optional[dict]: ...
    @abstractmethod
    def validate_user(self, user_id: str) -> bool: ...

class OrderModuleInterface(ABC):
    """Order モジュールの公開インターフェース"""
    @abstractmethod
    def create_order(self, user_id: str, items: list) -> dict: ...
    @abstractmethod
    def get_order(self, order_id: str) -> Optional[dict]: ...

class PaymentModuleInterface(ABC):
    """Payment モジュールの公開インターフェース"""
    @abstractmethod
    def charge(self, user_id: str, amount: float) -> dict: ...
    @abstractmethod
    def refund(self, payment_id: str) -> dict: ...

# ── User モジュールの実装 ──

class UserModule(UserModuleInterface):
    """User モジュール: users_* テーブルのみアクセス"""

    def __init__(self, db_session):
        self.db = db_session

    def get_user(self, user_id: str) -> Optional[dict]:
        user = self.db.query(User).get(user_id)
        if not user:
            return None
        return {"id": user.id, "name": user.name, "email": user.email}

    def validate_user(self, user_id: str) -> bool:
        return self.db.query(User).get(user_id) is not None

# ── Order モジュールの実装 ──

class OrderModule(OrderModuleInterface):
    """Order モジュール: orders_* テーブルのみアクセス"""

    def __init__(
        self,
        db_session,
        user_module: UserModuleInterface,      # 依存はインターフェース経由
        payment_module: PaymentModuleInterface,
    ):
        self.db = db_session
        self.users = user_module      # 直接 DB アクセスしない
        self.payments = payment_module

    def create_order(self, user_id: str, items: list) -> dict:
        # User モジュールのインターフェースを通じてアクセス
        if not self.users.validate_user(user_id):
            raise ValueError("User not found")

        order = Order(user_id=user_id, items=items, status="pending")
        self.db.add(order)
        self.db.flush()

        # Payment モジュールのインターフェースを通じてアクセス
        total = sum(item["price"] * item["quantity"] for item in items)
        payment = self.payments.charge(user_id, total)

        order.payment_id = payment["id"]
        order.status = "confirmed"
        self.db.commit()

        return {"order_id": order.id, "status": order.status}

    def get_order(self, order_id: str) -> Optional[dict]:
        order = self.db.query(Order).get(order_id)
        return {"id": order.id, "status": order.status} if order else None

# ── モジュール組み立て (Composition Root) ──

class Application:
    """アプリケーションのエントリーポイント"""

    def __init__(self, db_session):
        # モジュールの初期化（依存性注入）
        self.user_module = UserModule(db_session)
        self.payment_module = PaymentModule(db_session)
        self.order_module = OrderModule(
            db_session,
            user_module=self.user_module,
            payment_module=self.payment_module,
        )

# メリット:
# - モノリスの簡便さ (1つのデプロイ単位、ACID トランザクション)
# - モジュール間の境界が明確 (将来のマイクロサービス化が容易)
# - 型安全なインターフェース
# - テストが容易 (モジュール単位でモック可能)
```

---

## 4. Strangler Fig パターン

### 4.1 段階的な移行

```
Phase 1: モノリスのまま
  ┌──────────────────────────────┐
  │         Monolith             │
  │  [User] [Order] [Pay] [Noti]│
  └──────────────────────────────┘

Phase 2: API Gateway + 最初のサービスを切り出し
  ┌────────────────────────┐
  │    Monolith (残り)      │     ┌───────────┐
  │  [User] [Order] [Pay]  │────→│ Notif.    │ (切り出し済み)
  └────────────────────────┘     │ Service   │
         ↑                       └───────────┘
         │
  ┌──────┴──────┐
  │ API Gateway │  ← ルーティングで振り分け
  └─────────────┘

Phase 3: 徐々にサービスを切り出し
  ┌────────────────┐
  │  Monolith      │     ┌───────────┐  ┌───────────┐
  │  [User] [Order]│────→│ Payment   │  │ Notif.    │
  └────────────────┘     │ Service   │  │ Service   │
         ↑               └───────────┘  └───────────┘
         │                     ↑              ↑
  ┌──────┴─────────────────────┴──────────────┴──┐
  │              API Gateway                      │
  └───────────────────────────────────────────────┘

Phase 4: 全てをマイクロサービスに
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │  User    │ │  Order   │ │ Payment  │ │  Notif.  │
  │ Service  │ │ Service  │ │ Service  │ │ Service  │
  └──────────┘ └──────────┘ └──────────┘ └──────────┘
         ↑          ↑           ↑            ↑
  ┌──────┴──────────┴───────────┴────────────┴──┐
  │              API Gateway                     │
  └──────────────────────────────────────────────┘
```

### 4.2 Strangler Fig の実装

```python
"""Strangler Fig パターン: API Gateway でルーティングを制御"""
from fastapi import FastAPI, Request, Response
import httpx
import re

app = FastAPI(title="Strangler Gateway")

# 移行ルーティング設定
# status: "monolith" → "dual" → "microservice"
MIGRATION_ROUTES = {
    # 完全移行済み
    "/api/notifications": {
        "status": "microservice",
        "service_url": "http://notification-service:8080",
    },
    # ダブルライト中 (両方に送信して結果を比較)
    "/api/payments": {
        "status": "dual",
        "monolith_url": "http://monolith:8080",
        "service_url": "http://payment-service:8080",
    },
    # まだモノリス
    "/api/users": {
        "status": "monolith",
        "monolith_url": "http://monolith:8080",
    },
    "/api/orders": {
        "status": "monolith",
        "monolith_url": "http://monolith:8080",
    },
}

@app.api_route("/api/{path:path}", methods=["GET","POST","PUT","DELETE"])
async def strangler_route(path: str, request: Request):
    """Strangler Fig ルーティング"""
    full_path = f"/api/{path}"

    # ルート設定の検索 (最長一致)
    route_config = None
    for prefix, config in sorted(
        MIGRATION_ROUTES.items(), key=lambda x: -len(x[0])
    ):
        if full_path.startswith(prefix):
            route_config = config
            break

    if not route_config:
        route_config = {
            "status": "monolith",
            "monolith_url": "http://monolith:8080",
        }

    # ステータスに応じたルーティング
    status = route_config["status"]

    if status == "microservice":
        return await forward_request(
            request, route_config["service_url"], full_path
        )

    elif status == "dual":
        # ダブルライト: 両方に送信してレスポンスを比較
        monolith_resp, service_resp = await asyncio.gather(
            forward_request(
                request, route_config["monolith_url"], full_path
            ),
            forward_request(
                request, route_config["service_url"], full_path
            ),
            return_exceptions=True,
        )

        # モノリスのレスポンスを返す (安全側)
        # バックグラウンドで差分をログに記録
        if monolith_resp != service_resp:
            log_diff(full_path, monolith_resp, service_resp)

        return monolith_resp

    else:  # monolith
        return await forward_request(
            request, route_config["monolith_url"], full_path
        )

async def forward_request(request: Request, base_url: str, path: str):
    """リクエストを転送"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.request(
            method=request.method,
            url=f"{base_url}{path}",
            headers={
                k: v for k, v in request.headers.items()
                if k.lower() not in ("host",)
            },
            content=await request.body(),
        )
        return resp.json()

def log_diff(path: str, monolith: dict, service: dict):
    """モノリスとサービスのレスポンス差分をログに記録"""
    import json
    print(f"[DIFF] {path}")
    print(f"  Monolith: {json.dumps(monolith, indent=2)}")
    print(f"  Service:  {json.dumps(service, indent=2)}")
```

### 4.3 移行の判断基準

```python
"""サービス切り出しの優先度判断"""
from dataclasses import dataclass

@dataclass
class ServiceExtractionCandidate:
    """切り出し候補の評価"""
    name: str
    change_frequency: str    # "高" / "中" / "低"
    coupling: str            # "低" / "中" / "高"
    team_ownership: str      # "明確" / "不明確"
    scalability_need: str    # "高" / "中" / "低"
    data_independence: str   # "独立" / "部分的" / "共有"
    priority: str            # "高" / "中" / "低"

EXTRACTION_CANDIDATES = [
    ServiceExtractionCandidate(
        name="通知サービス",
        change_frequency="高",
        coupling="低",
        team_ownership="明確",
        scalability_need="高",
        data_independence="独立",
        priority="高 (最初に切り出すべき)",
    ),
    ServiceExtractionCandidate(
        name="決済サービス",
        change_frequency="中",
        coupling="中",
        team_ownership="明確",
        scalability_need="中",
        data_independence="部分的",
        priority="中 (2番目に切り出す)",
    ),
    ServiceExtractionCandidate(
        name="ユーザーサービス",
        change_frequency="低",
        coupling="高",
        team_ownership="不明確",
        scalability_need="低",
        data_independence="共有",
        priority="低 (最後に切り出す、または切り出さない)",
    ),
]

# 切り出し判断基準:
# 1. 結合度が低い → 切り出しが容易
# 2. 変更頻度が高い → 独立デプロイのメリットが大きい
# 3. スケーラビリティ要件がある → 独立スケールが必要
# 4. チームの所有が明確 → 運用が容易
# 5. データが独立している → データ移行が容易
```

---

## 5. 可観測性 (Observability)

### 5.1 分散トレーシング

```python
"""マイクロサービスの分散トレーシング実装"""
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
import uuid

# トレーサーの初期化
provider = TracerProvider()
processor = BatchSpanProcessor(
    OTLPSpanExporter(endpoint="http://jaeger:4317")
)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# FastAPI と httpx の自動計装
FastAPIInstrumentor().instrument_app(app)
HTTPXClientInstrumentor().instrument()

# カスタムスパンの追加
@app.post("/api/orders")
async def create_order(request: OrderRequest):
    with tracer.start_as_current_span("create_order") as span:
        span.set_attribute("user_id", request.user_id)
        span.set_attribute("item_count", len(request.items))

        # 子スパン: ユーザー検証
        with tracer.start_as_current_span("validate_user"):
            user = await user_client.get(f"/users/{request.user_id}")

        # 子スパン: 在庫確認
        with tracer.start_as_current_span("check_inventory"):
            inventory = await inventory_client.post(
                "/check", data={"items": request.items}
            )

        # 子スパン: 注文保存
        with tracer.start_as_current_span("save_order"):
            order = await save_to_db(request)
            span.set_attribute("order_id", order.id)

        return {"order_id": order.id}

# トレース結果 (Jaeger UI で可視化):
#
# create_order [Order Service] ─── 150ms
#   ├── validate_user [→ User Service] ─── 30ms
#   ├── check_inventory [→ Inventory Service] ─── 45ms
#   └── save_order [DB] ─── 20ms
#
# ボトルネックの特定:
# - check_inventory が遅い → Inventory Service の最適化
# - DB への書き込みが遅い → インデックス追加
```

### 5.2 ヘルスチェックパターン

```python
"""マイクロサービスのヘルスチェック実装"""
from fastapi import FastAPI
from datetime import datetime
import psycopg2
import redis

app = FastAPI()

@app.get("/health/live")
async def liveness():
    """Liveness: プロセスが生きているか (K8s livenessProbe)"""
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness():
    """Readiness: リクエストを受け付けられるか (K8s readinessProbe)"""
    checks = {}

    # DB 接続チェック
    try:
        conn = psycopg2.connect(DB_DSN, connect_timeout=2)
        conn.close()
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"

    # Redis 接続チェック
    try:
        r = redis.Redis.from_url(REDIS_URL, socket_timeout=2)
        r.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    # 依存サービスチェック
    for service_name, url in SERVICE_REGISTRY.items():
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{url}/health/live")
                checks[service_name] = "ok" if resp.status_code == 200 else "degraded"
        except Exception:
            checks[service_name] = "unreachable"

    all_ok = all(v == "ok" for v in checks.values())

    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }

# Kubernetes マニフェスト例:
# livenessProbe:
#   httpGet:
#     path: /health/live
#     port: 8080
#   initialDelaySeconds: 10
#   periodSeconds: 10
#   failureThreshold: 3
#
# readinessProbe:
#   httpGet:
#     path: /health/ready
#     port: 8080
#   initialDelaySeconds: 5
#   periodSeconds: 5
#   failureThreshold: 2
```

---

## 6. 比較表

### 比較表 1: モノリス vs マイクロサービス

| 項目 | モノリス | マイクロサービス |
|------|---------|---------------|
| デプロイ | 全体を一括 | サービス単位で独立 |
| スケーリング | 全体を一括スケール | サービス単位で個別スケール |
| 技術スタック | 統一（1言語/1FW） | サービスごとに自由 |
| データベース | 共有1つ | サービスごとに独立 |
| トランザクション | ACID（単一DB） | 結果整合性（Saga等） |
| テスト | 統合テストが容易 | E2Eテストが複雑 |
| デバッグ | スタックトレースで追跡可能 | 分散トレーシングが必要 |
| チーム | 全員が同一コードベース | チームがサービスを所有 |
| 初期開発速度 | 速い | 遅い（インフラ構築コスト） |
| 長期保守性 | コードが肥大化しやすい | サービス境界で複雑さを分割 |
| 障害の影響 | 1バグが全体に影響 | 障害が局所化 |
| 運用コスト | 低 | 高 (K8s, Service Mesh, 監視) |

### 比較表 2: チーム規模とアーキテクチャの適合性

| チーム規模 | 推奨アーキテクチャ | 理由 |
|-----------|-----------------|------|
| 1-5人 | モノリス | オーバーヘッド最小、全員がコード全体を把握 |
| 5-15人 | モジュラーモノリス | モジュール分割で境界を明確に、デプロイは一括 |
| 15-50人 | モノリス→MS移行 | Strangler Fig で段階的に切り出し |
| 50人以上 | マイクロサービス | チームがサービスを所有、独立開発・デプロイ |

### 比較表 3: サービス間通信方式

| 方式 | レイテンシ | 信頼性 | 結合度 | ユースケース |
|------|----------|--------|--------|------------|
| REST (HTTP) | 中 | 低 (同期) | 中 | CRUD API、公開 API |
| gRPC | 低 | 低 (同期) | 高 (スキーマ) | サービス間高速通信 |
| メッセージキュー | 高 | 高 (非同期) | 低 | イベント駆動、非同期処理 |
| GraphQL Federation | 中 | 中 | 中 | BFF (Backend for Frontend) |

| 判断基準 | REST | gRPC | Message Queue |
|---------|------|------|---------------|
| リアルタイム応答が必要 | OK | 最適 | 不適 |
| 障害耐性が重要 | 不適 | 不適 | 最適 |
| レイテンシ要件が厳しい | 普通 | 最適 | 不適 |
| 大量データのストリーミング | 不適 | 最適 | OK |
| サービス間の疎結合 | 普通 | 密 | 疎 |

---

## 7. アンチパターン

### アンチパターン 1: 分散モノリス

```python
# NG: マイクロサービスに分割したが密結合
class DistributedMonolith:
    """分散モノリスの典型例"""

    def create_order(self, user_id: str, items: list):
        # 問題1: 全サービスが同期HTTPで密結合
        user = requests.get(f"http://user-service/users/{user_id}").json()
        inventory = requests.post(
            "http://inventory-service/check", json=items
        ).json()
        payment = requests.post(
            "http://payment-service/charge",
            json={"user_id": user_id, "amount": 100}
        ).json()
        notification = requests.post(
            "http://notification-service/send",
            json={"user_id": user_id, "type": "order"}
        ).json()

        # 問題2: 1サービスの障害が全体に波及
        # notification-service が 503 → 注文全体が失敗
        # → マイクロサービスのデメリットだけ享受した最悪の構成

        # 問題3: 共有データベースを使い続けている
        # → 全サービスが同じ DB スキーマに依存
        # → スキーマ変更で全サービスの再デプロイが必要

# OK: 正しいマイクロサービス設計
class ProperMicroservice:
    """疎結合のマイクロサービス"""

    def create_order(self, user_id: str, items: list):
        # 1. 必要な同期呼び出しは最小限 + タイムアウト + フォールバック
        try:
            user = await self.user_client.get(
                f"/users/{user_id}", timeout=3.0
            )
        except Exception:
            user = {"id": user_id}  # フォールバック: キャッシュから

        # 2. 注文作成 (自サービスの DB に書き込み)
        order = await self.save_order(user_id, items)

        # 3. 残りは非同期イベントで処理
        await self.publish_event("order_created", {
            "order_id": order.id,
            "user_id": user_id,
            "items": items,
        })
        # → 在庫サービスがイベントを消費して在庫を減らす
        # → 決済サービスがイベントを消費して課金する
        # → 通知サービスがイベントを消費して通知を送る
        # → 各サービスが独立して処理、障害は局所化

        return {"order_id": order.id, "status": "processing"}
```

### アンチパターン 2: 初期段階でのマイクロサービス

```python
# NG: スタートアップが Day 1 から20個のマイクロサービス
class PrematureMicroservices:
    """早すぎるマイクロサービス化"""
    services = [
        "user-service", "auth-service", "profile-service",
        "order-service", "cart-service", "inventory-service",
        "payment-service", "notification-service", "email-service",
        "search-service", "recommendation-service", "analytics-service",
        "logging-service", "config-service", "gateway-service",
        # ... 合計20サービス
    ]

    # 問題:
    # - サービス境界が不明確 (ドメイン理解が浅い段階)
    # - K8s + Kafka + Jaeger + CI/CD x 20 の構築コスト
    # - 3人のチームで20サービスの運用は不可能
    # - ピボット時にサービス境界の再設計が必要
    # - 初期開発速度が 3-5 倍遅い

# OK: 正しいアプローチ
class MonolithFirst:
    """まずモノリスで始めて、ドメインを理解してから分割"""

    # Phase 1: モノリスで MVP を作る (3-6ヶ月)
    # - ドメインを理解する
    # - ユーザーのニーズを検証する
    # - サービス境界の候補を特定する

    # Phase 2: モジュラーモノリスに移行 (6-12ヶ月)
    # - モジュール間のインターフェースを定義
    # - DB スキーマをモジュール別に分離
    # - 将来の切り出しに備える

    # Phase 3: 必要に応じてサービスを切り出す (12ヶ月+)
    # - 変更頻度が高いモジュール
    # - スケーラビリティ要件があるモジュール
    # - チームが所有するモジュール

    # Martin Fowler:
    # "Almost all the successful microservice stories have started
    #  with a monolith that got too big and was broken up."
    pass
```

### アンチパターン 3: サービスごとに異なる技術スタックの乱立

```python
# NG: 全サービスが異なる技術スタック
class TechStackChaos:
    """技術スタックの乱立"""
    services = {
        "user-service": "Python + Django + PostgreSQL",
        "order-service": "Go + Gin + MongoDB",
        "payment-service": "Java + Spring Boot + MySQL",
        "notification-service": "Node.js + Express + Redis",
        "search-service": "Rust + Actix + Elasticsearch",
        "analytics-service": "Scala + Play + ClickHouse",
    }
    # 問題:
    # - 6種類の言語を知る人材の採用が困難
    # - 共通ライブラリ (認証、ログ、メトリクス) を6言語で維持
    # - CI/CD パイプラインが6種類
    # - デバッグ・トラブルシューティングの難易度が極めて高い

# OK: 技術スタックを2-3種類に制限
class ControlledTechStack:
    """技術スタックの統制"""
    standards = {
        "primary": "Python + FastAPI + PostgreSQL",  # 80% のサービス
        "performance": "Go + PostgreSQL",             # 高性能が必要な 15%
        "data": "Python + Spark + ClickHouse",       # データパイプライン 5%
    }
    # メリット:
    # - 採用が容易 (2言語を知っていれば OK)
    # - 共通ライブラリを2言語で管理
    # - チーム間の異動が容易
```

---

## 8. 練習問題

### 演習 1 (基礎): アーキテクチャ選定

以下の各シナリオについて、モノリス / モジュラーモノリス / マイクロサービスのどれが適切か選定し、理由を述べよ。

```
シナリオ A:
- 創業2年のスタートアップ
- エンジニア4人
- ECサイト (月間アクセス10万PV)
- 2ヶ月で MVP をリリースしたい

シナリオ B:
- 大手金融機関
- エンジニア60人 (5チーム x 12人)
- 決済プラットフォーム (月間トランザクション1億件)
- 規制要件: 各サービスの独立監査が必要

シナリオ C:
- 中規模 SaaS 企業
- エンジニア20人
- 現在モノリスで運用中 (コードベース 50万行)
- 一部機能のスケーラビリティに課題
```

**期待される出力:**

```
シナリオ A: モノリス
理由: 小規模チーム、短期間リリース、低トラフィック。
      マイクロサービスのインフラコストが不要。
      ドメイン理解が浅い段階での分割は時期尚早。

シナリオ B: マイクロサービス
理由: 大規模チーム、高トラフィック、独立監査要件。
      チーム単位でサービスを所有し独立デプロイ。
      決済 / ユーザー管理 / レポートを分離。

シナリオ C: モジュラーモノリス → 段階的マイクロサービス移行
理由: 中規模チーム、既存モノリス、部分的スケーラビリティ課題。
      まずモジュール境界を明確にし、ボトルネックの機能から
      Strangler Fig パターンで切り出す。
```

### 演習 2 (応用): Saga パターンの設計

以下の注文処理フローを Saga パターンで設計せよ。

```
注文処理フロー:
1. 在庫を予約する (Inventory Service)
2. クレジットカード決済する (Payment Service)
3. 注文を確定する (Order Service)
4. 配送を手配する (Shipping Service)
5. 確認メールを送信する (Notification Service)

課題:
1. 各ステップの補償処理 (ロールバック) を定義せよ
2. ステップ3 (注文確定) で失敗した場合のフローを図示せよ
3. 補償処理自体が失敗した場合の対策を設計せよ
4. Choreography と Orchestration のどちらを採用するか、理由と共に述べよ
```

**期待される出力:**

```
補償処理:
1. 在庫予約 → 予約解除
2. 決済 → 返金
3. 注文確定 → 注文キャンセル
4. 配送手配 → 配送キャンセル
5. メール送信 → キャンセルメール送信 (or 何もしない)

ステップ3 失敗時のフロー:
  在庫予約 ✓ → 決済 ✓ → 注文確定 ✗ → 返金 → 在庫予約解除
```

### 演習 3 (上級): モノリスからの移行計画

以下のモノリスアプリケーションをマイクロサービスに移行する計画を作成せよ。

```
現状:
- Python/Django モノリス (30万行)
- PostgreSQL 1台 (テーブル数: 120)
- チーム: 15人
- デプロイ: 週1回 (金曜 深夜)
- 月間リクエスト: 5000万

モジュール構成:
- ユーザー管理 (認証、プロフィール、設定)
- 商品カタログ (商品、カテゴリ、検索)
- 注文管理 (カート、注文、注文履歴)
- 決済 (課金、返金、明細)
- 通知 (メール、プッシュ、SMS)
- レポート (売上集計、分析)

課題:
1. 切り出すサービスの優先順位を決定し、理由を述べよ
2. 各フェーズのマイルストーンとタイムラインを設計せよ
3. データ移行戦略を設計せよ
4. リスクと対策を列挙せよ
5. 成功指標 (KPI) を定義せよ
```

**期待される出力:** フェーズごとの詳細計画 (3-6ヶ月単位)、技術的な判断理由、リスク対策

---

## 9. FAQ

### Q1: モジュラーモノリスとは何ですか？

**A.** モジュラーモノリスは、デプロイは単一のモノリスだが、内部はモジュール（バウンデッドコンテキスト）に明確に分割された構成である。モジュール間は定義されたインターフェース（公開 API）のみで通信し、データベーステーブルもモジュールごとにスキーマを分離する。他モジュールのテーブルへの直接 SQL アクセスは禁止。モノリスの簡便さ（1つのデプロイ、ACID トランザクション、シンプルな運用）とマイクロサービスの疎結合（モジュール間の独立性、将来の分割容易性）を両立する。Shopify が大規模にこのアーキテクチャを採用していることで知られる。チーム規模 5-30 人の中規模プロジェクトに最適。

### Q2: マイクロサービスで分散トランザクションはどう扱いますか？

**A.** 2つの主要アプローチがある。(1) **Saga パターン** --- 各サービスがローカルトランザクションを実行し、失敗時は補償トランザクションで巻き戻す。Choreography（イベント駆動、各サービスが自律的に処理）と Orchestration（中央のオーケストレーターが制御）がある。5ステップ以下なら Choreography、それ以上なら Orchestration が推奨。(2) **結果整合性の受容** --- 厳密な即時一貫性を求めず、イベント駆動でデータを伝播させ、最終的に一貫する設計にする。2PC (Two-Phase Commit) は性能ペナルティが大きく、可用性も低下するため、マイクロサービスでは一般的に避ける。

### Q3: サービスメッシュは必要ですか？

**A.** 10サービス以下であれば不要。サービスメッシュ（Istio、Linkerd）はサービス間通信の暗号化（mTLS）、トラフィック管理（カナリアデプロイ、レートリミット）、可観測性（分散トレーシング）をインフラ層 (sidecar proxy) で提供する。学習コストと運用コスト（CPU/メモリオーバーヘッド約 10-15%）が高い。サービス数が 20 以上、マルチチームでサービス間の信頼性・セキュリティ要件が高い場合に検討する。まずはアプリケーション層でのリトライ・サーキットブレーカー・mTLS で十分なことが多い。

### Q4: マイクロサービスのテスト戦略は？

**A.** テストピラミッドをサービス単位で適用する: (1) **ユニットテスト** --- 各サービス内のビジネスロジック。外部依存はモックする。全体の 70%。(2) **統合テスト** --- サービス + DB、サービス + MQ の組み合わせ。Testcontainers で実際の DB/Redis を起動してテスト。20%。(3) **コントラクトテスト** --- サービス間の API 契約を検証。Pact 等を使用。Consumer が期待するリクエスト/レスポンスを Provider が満たすか自動検証。5%。(4) **E2E テスト** --- 最小限。ユーザーシナリオの主要パスのみ。5%。コントラクトテストが特に重要で、これにより全サービスを起動せずにAPI互換性を保証できる。

### Q5: gRPC と REST のどちらを使うべきですか？

**A.** 社内サービス間通信では gRPC、外部向け API では REST が一般的。gRPC の利点: (1) Protocol Buffers による強い型付け、(2) HTTP/2 ベースで高速 (REST 比 2-5 倍)、(3) 双方向ストリーミング対応、(4) コード自動生成。REST の利点: (1) ブラウザから直接呼び出し可能、(2) curl でテスト可能、(3) 学習コストが低い、(4) エコシステムが豊富。実務では「外部 API は REST、内部通信は gRPC、非同期処理はメッセージキュー」の組み合わせが多い。GraphQL は BFF (Backend for Frontend) パターンで、複数のマイクロサービスを集約してフロントエンドに最適なデータを提供する用途に適している。

### Q6: マイクロサービスのデータ管理で「Database per Service」は絶対か？

**A.** 原則として Yes だが、現実には段階的に移行する。「Database per Service」はサービスの独立性の根幹であり、これなしではスキーマ変更で全サービスの再デプロイが必要になる（分散モノリスのアンチパターン）。ただし、移行初期は「スキーマ per サービス」（同じ PostgreSQL インスタンス内でスキーマを分離）から始めるのが現実的。各サービスは自分のスキーマのテーブルのみアクセスし、他サービスのデータはAPI経由で取得する。運用が安定してから物理的に DB を分離する。CQRS パターンで読み取り専用のデータストアを各サービスに持たせることで、クロスサービスクエリの問題を解決できる。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| モノリス | 単一デプロイ、ACID保証、初期開発が高速。5人以下のチームに最適 |
| モジュラーモノリス | モノリスの簡便さ + モジュール分割。5-30人チームの推奨選択 |
| マイクロサービス | 独立デプロイ・スケール、チーム自律性。50人以上のチームで有効 |
| 選定基準 | チーム規模・ドメイン理解度・運用能力・スケーラビリティ要件で判断 |
| 移行戦略 | Strangler Fig で段階的に切り出し。最も結合度が低いモジュールから |
| 分散トランザクション | Saga パターンで補償処理。クロスサービスの強整合性は避ける |
| 通信方式 | 同期: REST/gRPC (最小限)、非同期: メッセージキュー (推奨) |
| 可観測性 | 分散トレーシング (OpenTelemetry + Jaeger) が必須 |
| 最重要原則 | 「まずモノリスで始め、ドメインを理解してから、必要に応じて分割」 |

---

## 次に読むべきガイド

- [クリーンアーキテクチャ](./01-clean-architecture.md) --- モジュール間の依存関係制御
- [DDD](./02-ddd.md) --- サービス境界の設計（バウンデッドコンテキスト）
- [イベント駆動アーキテクチャ](./03-event-driven.md) --- マイクロサービス間の非同期通信
- [メッセージキュー](../01-components/02-message-queue.md) --- 非同期メッセージング基盤
- [ロードバランサー](../01-components/00-load-balancer.md) --- サービスへのトラフィック分散

---

## 参考文献

1. **Building Microservices**, 2nd Edition --- Sam Newman (O'Reilly, 2021) --- マイクロサービスの包括的ガイド
2. **MonolithFirst** --- Martin Fowler (2015) --- https://martinfowler.com/bliki/MonolithFirst.html
3. **Microservices Patterns** --- Chris Richardson (Manning, 2018) --- Saga、CQRS、API Gateway 等のパターン集
4. **Domain-Driven Design** --- Eric Evans (Addison-Wesley, 2003) --- バウンデッドコンテキストによるサービス境界設計
5. **Production-Ready Microservices** --- Susan Fowler (O'Reilly, 2016) --- マイクロサービスの運用品質基準

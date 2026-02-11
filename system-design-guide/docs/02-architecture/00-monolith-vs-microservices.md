# モノリス vs マイクロサービス

> モノリスとマイクロサービスそれぞれの特性を正確に理解し、プロジェクトの規模・チーム体制・ビジネス要件に応じた最適なアーキテクチャ選定ができる判断力を身につける。

## この章で学ぶこと

1. モノリスとマイクロサービスの構造的な違いとトレードオフ
2. モノリスからマイクロサービスへの段階的な移行戦略（Strangler Fig パターン）
3. マイクロサービスにおけるサービス間通信とデータ管理の設計パターン

---

## 1. モノリスとは

### ASCII図解1: モノリスアーキテクチャ

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
  └────────────────────────────────────────┘
```

---

## 2. マイクロサービスとは

### ASCII図解2: マイクロサービスアーキテクチャ

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
```

---

## 3. コード例で比較

### コード例1: モノリスでの注文処理

```python
# monolith/order_service.py
# モノリス: 全てが1つのプロセス内

class OrderService:
    def __init__(self, db_session):
        self.db = db_session

    def create_order(self, user_id: str, items: list) -> dict:
        # 同一DB内でトランザクション
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

        # メリット: 1つのトランザクションで ACID 保証
        # デメリット: 全モジュールが密結合
```

### コード例2: マイクロサービスでの注文処理

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
        async with httpx.AsyncClient() as client:
            # 1. ユーザー確認（HTTPで別サービスに問い合わせ）
            resp = await client.get(f"{self.user_service_url}/users/{user_id}")
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
        order = Order(user_id=user_id, items=items, status="pending")
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

# メリット: 独立デプロイ・スケール、技術選択の自由
# デメリット: 分散トランザクション、結果整合性
```

### コード例3: Saga パターンによる分散トランザクション

```python
from enum import Enum
from dataclasses import dataclass

class SagaStepStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"

@dataclass
class SagaStep:
    name: str
    action: callable        # 正方向の処理
    compensation: callable  # 補償処理（ロールバック）
    status: SagaStepStatus = SagaStepStatus.PENDING

class OrderSaga:
    """Choreography Saga: 注文処理の分散トランザクション"""

    def __init__(self):
        self.steps: list[SagaStep] = []
        self.completed_steps: list[SagaStep] = []

    def add_step(self, name, action, compensation):
        self.steps.append(SagaStep(name, action, compensation))

    async def execute(self, context: dict):
        for step in self.steps:
            try:
                print(f"[SAGA] Executing: {step.name}")
                result = await step.action(context)
                context.update(result or {})
                step.status = SagaStepStatus.COMPLETED
                self.completed_steps.append(step)
            except Exception as e:
                print(f"[SAGA] Failed: {step.name} - {e}")
                step.status = SagaStepStatus.FAILED
                await self._compensate(context)
                raise

    async def _compensate(self, context: dict):
        """完了済みステップを逆順に補償"""
        for step in reversed(self.completed_steps):
            try:
                print(f"[SAGA] Compensating: {step.name}")
                await step.compensation(context)
                step.status = SagaStepStatus.COMPENSATED
            except Exception as e:
                print(f"[SAGA] Compensation failed: {step.name} - {e}")

# 使用例
saga = OrderSaga()
saga.add_step(
    "reserve_inventory",
    action=lambda ctx: inventory_service.reserve(ctx["items"]),
    compensation=lambda ctx: inventory_service.cancel_reservation(ctx["reservation_id"]),
)
saga.add_step(
    "process_payment",
    action=lambda ctx: payment_service.charge(ctx["user_id"], ctx["amount"]),
    compensation=lambda ctx: payment_service.refund(ctx["payment_id"]),
)
saga.add_step(
    "confirm_order",
    action=lambda ctx: order_service.confirm(ctx["order_id"]),
    compensation=lambda ctx: order_service.cancel(ctx["order_id"]),
)
```

### コード例4: API Gateway パターン

```python
from fastapi import FastAPI, Request, HTTPException
import httpx

app = FastAPI(title="API Gateway")

SERVICE_REGISTRY = {
    "users":         "http://user-service:8080",
    "orders":        "http://order-service:8080",
    "payments":      "http://payment-service:8080",
    "notifications": "http://notification-service:8080",
}

@app.api_route("/{service}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def gateway(service: str, path: str, request: Request):
    """全リクエストをルーティング"""
    base_url = SERVICE_REGISTRY.get(service)
    if not base_url:
        raise HTTPException(404, f"Service '{service}' not found")

    target_url = f"{base_url}/{path}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        # リクエスト転送
        response = await client.request(
            method=request.method,
            url=target_url,
            headers={k: v for k, v in request.headers.items()
                     if k.lower() != "host"},
            content=await request.body(),
        )
        return response.json()
```

### コード例5: サービス間通信のリトライとタイムアウト

```python
import httpx
import asyncio
from functools import wraps

class ResilientServiceClient:
    """障害耐性のあるサービス間通信"""

    def __init__(self, base_url: str, timeout: float = 5.0,
                 max_retries: int = 3):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    async def get(self, path: str) -> dict:
        return await self._request("GET", path)

    async def post(self, path: str, data: dict) -> dict:
        return await self._request("POST", path, json=data)

    async def _request(self, method: str, path: str, **kwargs):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.timeout
                ) as client:
                    resp = await client.request(method, path, **kwargs)
                    resp.raise_for_status()
                    return resp.json()
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_error = e
                delay = 0.5 * (2 ** attempt)
                print(f"[Retry {attempt+1}/{self.max_retries}] "
                      f"{method} {path}: {e}, waiting {delay}s")
                await asyncio.sleep(delay)

        raise last_error

# 使用例
user_client = ResilientServiceClient("http://user-service:8080")
order_client = ResilientServiceClient("http://order-service:8080", timeout=10.0)
```

---

## 4. Strangler Fig パターン

### ASCII図解3: 段階的な移行

```
Phase 1: モノリスのまま
  ┌──────────────────────────────┐
  │         Monolith             │
  │  [User] [Order] [Pay] [Noti]│
  └──────────────────────────────┘

Phase 2: 一部をマイクロサービスに切り出し
  ┌────────────────────────┐
  │    Monolith (残り)      │     ┌───────────┐
  │  [User] [Order] [Pay]  │────→│ Notif.    │ (切り出し済み)
  └────────────────────────┘     │ Service   │
         ↑                       └───────────┘
         │
  ┌──────┴──────┐
  │ API Gateway │  ← ルーティングで振り分け
  └─────────────┘

Phase 3: 全てをマイクロサービスに
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │  User    │ │  Order   │ │ Payment  │ │  Notif.  │
  │ Service  │ │ Service  │ │ Service  │ │ Service  │
  └──────────┘ └──────────┘ └──────────┘ └──────────┘
         ↑          ↑           ↑            ↑
  ┌──────┴──────────┴───────────┴────────────┴──┐
  │              API Gateway                     │
  └──────────────────────────────────────────────┘
```

---

## 5. 比較表

### 比較表1: モノリス vs マイクロサービス

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

### 比較表2: チーム規模とアーキテクチャの適合性

| チーム規模 | 推奨アーキテクチャ | 理由 |
|-----------|-----------------|------|
| 1-5人 | モノリス | オーバーヘッド最小、全員がコード全体を把握 |
| 5-15人 | モジュラーモノリス | モジュール分割で境界を明確に、デプロイは一括 |
| 15-50人 | モノリス→MS移行 | Strangler Fig で段階的に切り出し |
| 50人以上 | マイクロサービス | チームがサービスを所有、独立開発・デプロイ |

---

## 6. アンチパターン

### アンチパターン1: 分散モノリス

```
❌ ダメな例:
マイクロサービスに分割したが:
- 全サービスが同時にデプロイ必要
- サービス間が同期HTTPで密結合
- 1サービスの障害が全体に波及
- 共有データベースを使い続けている

→ マイクロサービスのデメリットだけ享受した最悪の構成

✅ 正しいマイクロサービス:
- 独立デプロイ可能
- 非同期通信（イベント駆動）で疎結合
- サービスごとに独立データストア
- 他サービスの障害時にグレースフルデグラデーション
```

### アンチパターン2: 初期段階でのマイクロサービス

```
❌ ダメな例:
スタートアップが Day 1 から20個のマイクロサービス

問題:
- サービス境界が不明確（ドメイン理解が浅い段階）
- Kubernetes + Kafka + 分散トレーシング + CI/CD の構築コスト
- 3人のチームで20サービスの運用は不可能
- ピボット時にサービス境界の再設計が必要

✅ 正しいアプローチ:
「まずモノリスで始めて、ドメインを理解してから分割」
— Martin Fowler, "MonolithFirst"
```

---

## 7. FAQ

### Q1: モジュラーモノリスとは何ですか？

モジュラーモノリスは、デプロイは単一のモノリスだが、内部はモジュール（バウンデッドコンテキスト）に明確に分割された構成である。モジュール間は定義されたインターフェースのみで通信し、データベーステーブルもモジュールごとにスキーマを分離する。モノリスの簡便さとマイクロサービスの疎結合を両立し、将来的なマイクロサービス化の準備にもなる。Shopifyが採用していることで知られる。

### Q2: マイクロサービスで分散トランザクションはどう扱いますか？

2つのアプローチがある: (1) **Saga パターン** — 各サービスがローカルトランザクションを実行し、失敗時は補償トランザクションで巻き戻す。Choreography（イベント駆動）と Orchestration（中央制御）がある。(2) **結果整合性の受容** — 厳密な即時一貫性を求めず、イベント駆動でデータを伝播させ、いつかは一貫する設計にする。2PC（Two-Phase Commit）は性能ペナルティが大きいため、マイクロサービスでは一般的に避ける。

### Q3: サービスメッシュは必要ですか？

10サービス以下であれば不要。サービスメッシュ（Istio、Linkerd）はサービス間通信の暗号化（mTLS）、トラフィック管理、可観測性をインフラ層で提供するが、学習コストと運用コストが高い。サービス数が20以上、マルチチームでサービス間の信頼性・セキュリティ要件が高い場合に検討する。まずはアプリケーション層でのリトライ・サーキットブレーカーで十分なことが多い。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| モノリス | 単一デプロイ、ACID保証、初期開発が高速 |
| マイクロサービス | 独立デプロイ・スケール、チーム自律性 |
| 選定基準 | チーム規模・ドメイン理解度・運用能力で判断 |
| 移行戦略 | Strangler Fig で段階的に切り出し |
| 分散トランザクション | Saga パターンで補償処理 |
| 最重要原則 | 「まずモノリスで始め、必要に応じて分割」 |

---

## 次に読むべきガイド

- [クリーンアーキテクチャ](./01-clean-architecture.md) — モジュール間の依存関係制御
- [DDD](./02-ddd.md) — サービス境界の設計（バウンデッドコンテキスト）
- [イベント駆動アーキテクチャ](./03-event-driven.md) — マイクロサービス間の非同期通信

---

## 参考文献

1. Newman, S. (2021). *Building Microservices*, 2nd Edition. O'Reilly Media.
2. Fowler, M. (2015). "MonolithFirst." https://martinfowler.com/bliki/MonolithFirst.html
3. Richardson, C. (2018). *Microservices Patterns*. Manning Publications.

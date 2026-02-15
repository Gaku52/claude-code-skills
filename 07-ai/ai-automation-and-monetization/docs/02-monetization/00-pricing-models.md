# 価格モデル — 従量制、サブスク、フリーミアム

> AI SaaSおよびAIサービスの価格設計を体系的に解説し、従量課金、サブスクリプション、フリーミアムの各モデルの設計原則、実装方法、最適化戦略を提供する。

---

## この章で学ぶこと

1. **AI特有の価格設計の原則** — コスト構造（API費用、GPU費用）を考慮した価格設定フレームワーク
2. **3大価格モデルの設計と実装** — 従量制、サブスクリプション、フリーミアムの詳細設計
3. **価格最適化と実験手法** — A/Bテスト、価格感度分析、LTV最大化の実践
4. **ハイブリッド価格モデル** — 複数モデルの組み合わせによる収益最大化
5. **心理的価格設計** — 行動経済学に基づくプライシングテクニック
6. **国際価格戦略** — 地域別価格設定とPPP（購買力平価）対応

---

## 1. AI価格設計の原則

### 1.1 AI SaaS のコスト構造

```
┌──────────────────────────────────────────────────────────┐
│              AI SaaS コスト構造分解                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  固定費（月額）           変動費（利用量比例）              │
│  ┌──────────────┐      ┌──────────────────┐            │
│  │ サーバー      │      │ AI API呼び出し    │ ← 最大変動要因│
│  │ ¥30,000      │      │ ¥0.5-50/リクエスト│            │
│  ├──────────────┤      ├──────────────────┤            │
│  │ DB/ストレージ │      │ GPU推論時間       │            │
│  │ ¥10,000      │      │ ¥0.1-5/秒        │            │
│  ├──────────────┤      ├──────────────────┤            │
│  │ 監視/ログ     │      │ ストレージ増分    │            │
│  │ ¥5,000       │      │ ¥0.01/MB         │            │
│  ├──────────────┤      ├──────────────────┤            │
│  │ ドメイン/SSL  │      │ 帯域/転送量       │            │
│  │ ¥2,000       │      │ ¥0.001/MB        │            │
│  └──────────────┘      └──────────────────┘            │
│  合計: ~¥47,000/月      合計: ¥1-100/ユーザー/日         │
│                                                          │
│  ★ AI APIコストが売上の20-40%を占めるのがAI SaaSの特徴    │
└──────────────────────────────────────────────────────────┘
```

### 1.2 価格設定フレームワーク

```python
# 価格設定の3要素
pricing_framework = {
    "cost_based": {
        "description": "原価+マージンで決定",
        "formula": "価格 = API原価 / (1 - 目標粗利率)",
        "example": "API原価¥200/回、粗利70%目標 → ¥667/回",
        "pros": "赤字回避",
        "cons": "価値を反映しない"
    },
    "value_based": {
        "description": "顧客が得る価値で決定",
        "formula": "価格 = 顧客の時間節約額 × 30-50%",
        "example": "3時間の作業を10分に → 時給¥5,000 × 3h × 30% = ¥4,500",
        "pros": "高単価可能",
        "cons": "価値の定量化が難しい"
    },
    "competition_based": {
        "description": "競合価格を参考に決定",
        "formula": "価格 = 競合平均 × 差別化係数",
        "example": "競合平均$49/月、品質1.5倍 → $69/月",
        "pros": "市場整合性",
        "cons": "価格競争に巻き込まれる"
    }
}
```

### 1.3 AI SaaS特有の価格設計原則

AI SaaSの価格設定は従来のSaaSとは根本的に異なる。最大の違いは**限界費用がゼロではない**ことだ。従来のSaaSでは新規ユーザー1人の追加コストはほぼゼロだが、AI SaaSではAPIコールやGPU推論のたびに直接コストが発生する。

```python
class AIPricingPrinciples:
    """AI SaaS価格設計の原則"""

    def __init__(self):
        self.principles = {
            "marginal_cost_awareness": {
                "description": "限界費用の可視化",
                "detail": "各APIコールのコストをリアルタイムで把握",
                "implementation": "コスト追跡ミドルウェアの導入"
            },
            "value_metric_alignment": {
                "description": "価値指標との整合",
                "detail": "顧客が感じる価値と課金単位を一致させる",
                "implementation": "アウトプット単位での課金設計"
            },
            "cost_floor_guarantee": {
                "description": "コストフロアの保証",
                "detail": "どのプランでも変動費を下回らない価格設定",
                "implementation": "動的価格下限の設定"
            },
            "usage_predictability": {
                "description": "利用量の予測可能性",
                "detail": "顧客が月額費用を予測できる仕組み",
                "implementation": "使用量ダッシュボードと予算アラート"
            }
        }

    def calculate_minimum_price(
        self,
        api_cost_per_request: float,
        avg_requests_per_user: int,
        target_gross_margin: float = 0.70,
        fixed_cost_per_user: float = 500
    ) -> dict:
        """最低価格の算出"""
        variable_cost = api_cost_per_request * avg_requests_per_user
        total_cost = variable_cost + fixed_cost_per_user
        minimum_price = total_cost / (1 - target_gross_margin)

        return {
            "variable_cost": variable_cost,
            "fixed_cost": fixed_cost_per_user,
            "total_cost": total_cost,
            "minimum_price": round(minimum_price),
            "target_gross_margin": f"{target_gross_margin * 100}%",
            "recommendation": f"¥{round(minimum_price / 100) * 100}以上に設定"
        }

    def model_cost_comparison(self) -> dict:
        """モデル別コスト比較"""
        return {
            "gpt-4o": {
                "input_per_1k_tokens": 2.5,  # ¥
                "output_per_1k_tokens": 10.0,
                "avg_cost_per_request": 15.0,
                "recommended_for": "高品質な分析・生成タスク"
            },
            "gpt-4o-mini": {
                "input_per_1k_tokens": 0.15,
                "output_per_1k_tokens": 0.6,
                "avg_cost_per_request": 1.0,
                "recommended_for": "大量処理・簡易タスク"
            },
            "claude-3.5-sonnet": {
                "input_per_1k_tokens": 3.0,
                "output_per_1k_tokens": 15.0,
                "avg_cost_per_request": 20.0,
                "recommended_for": "コード生成・複雑な推論"
            },
            "claude-3.5-haiku": {
                "input_per_1k_tokens": 0.25,
                "output_per_1k_tokens": 1.25,
                "avg_cost_per_request": 2.0,
                "recommended_for": "高速応答・分類タスク"
            }
        }


# 使用例
principles = AIPricingPrinciples()
min_price = principles.calculate_minimum_price(
    api_cost_per_request=15.0,  # GPT-4oの平均
    avg_requests_per_user=200,  # 月200リクエスト
    target_gross_margin=0.70
)
# → minimum_price: ¥10,500、recommendation: "¥10,500以上に設定"
```

### 1.4 価値指標（Value Metric）の選定

価値指標とは、顧客に課金する単位のことだ。正しい価値指標を選ぶことは、価格モデル全体の成功を左右する。

```python
class ValueMetricSelector:
    """価値指標の選定ツール"""

    VALUE_METRICS = {
        "api_calls": {
            "description": "API呼び出し回数",
            "pros": ["計測が簡単", "開発者に馴染み深い"],
            "cons": ["価値と乖離する可能性", "複雑なリクエストも同一課金"],
            "best_for": "開発者向けAPI（OpenAI、Anthropic）",
            "example": "$0.01/リクエスト"
        },
        "tokens": {
            "description": "入出力トークン数",
            "pros": ["リソース消費に正比例", "公平性が高い"],
            "cons": ["顧客に分かりにくい", "予算予測が困難"],
            "best_for": "LLM APIプロバイダー",
            "example": "$0.003/1Kトークン"
        },
        "outputs": {
            "description": "生成されたアウトプット数",
            "pros": ["価値と直結", "顧客に分かりやすい"],
            "cons": ["品質の差が反映されない"],
            "best_for": "コンテンツ生成ツール（Jasper、Copy.ai）",
            "example": "¥50/記事生成"
        },
        "seats": {
            "description": "利用ユーザー数",
            "pros": ["予測可能", "営業しやすい"],
            "cons": ["利用量と無関係", "シート共有の問題"],
            "best_for": "チーム向けSaaS（Notion AI）",
            "example": "¥1,500/ユーザー/月"
        },
        "outcomes": {
            "description": "成果ベース",
            "pros": ["最高の価値整合", "顧客満足度が高い"],
            "cons": ["計測が困難", "成果の定義が曖昧"],
            "best_for": "営業支援AI（受注確率向上）",
            "example": "成約金額の5%"
        },
        "compute_time": {
            "description": "計算時間",
            "pros": ["リソース消費に正確", "公平"],
            "cons": ["顧客体験が悪い", "最適化インセンティブ低"],
            "best_for": "ML学習プラットフォーム",
            "example": "$0.50/GPU時間"
        }
    }

    def recommend_metric(self, target_audience: str,
                         product_type: str) -> dict:
        """ターゲットに適した価値指標を推薦"""
        recommendations = {
            ("developer", "api"): ["tokens", "api_calls"],
            ("developer", "platform"): ["compute_time", "api_calls"],
            ("business", "tool"): ["outputs", "seats"],
            ("business", "platform"): ["seats", "outcomes"],
            ("consumer", "app"): ["outputs", "api_calls"],
            ("enterprise", "solution"): ["seats", "outcomes"]
        }

        key = (target_audience, product_type)
        if key in recommendations:
            metrics = recommendations[key]
            return {
                "primary": self.VALUE_METRICS[metrics[0]],
                "secondary": self.VALUE_METRICS[metrics[1]],
                "reasoning": f"{target_audience}向け{product_type}には"
                           f"{metrics[0]}が最適"
            }
        return {"error": "該当する組み合わせなし"}
```

---

## 2. 従量課金モデル

### 2.1 従量課金の設計

```python
class UsageBasedPricing:
    """従量課金エンジン"""

    def __init__(self):
        self.tiers = [
            {"name": "Tier 1", "up_to": 1000,   "price_per_unit": 5.0},
            {"name": "Tier 2", "up_to": 10000,  "price_per_unit": 3.0},
            {"name": "Tier 3", "up_to": 100000, "price_per_unit": 1.5},
            {"name": "Tier 4", "up_to": None,    "price_per_unit": 0.8}
        ]

    def calculate_cost(self, usage: int) -> dict:
        """階段型従量課金の計算"""
        total = 0
        remaining = usage
        breakdown = []

        for tier in self.tiers:
            limit = tier["up_to"] or float("inf")
            if remaining <= 0:
                break

            units_in_tier = min(remaining, limit - (
                self.tiers[self.tiers.index(tier) - 1]["up_to"]
                if self.tiers.index(tier) > 0 else 0
            ))
            cost = units_in_tier * tier["price_per_unit"]
            total += cost
            remaining -= units_in_tier

            breakdown.append({
                "tier": tier["name"],
                "units": units_in_tier,
                "rate": tier["price_per_unit"],
                "cost": cost
            })

        return {
            "total_usage": usage,
            "total_cost": total,
            "average_cost_per_unit": total / usage if usage > 0 else 0,
            "breakdown": breakdown
        }

# 使用例
pricing = UsageBasedPricing()
result = pricing.calculate_cost(5000)
# → ¥19,000 (1000×¥5 + 4000×¥3.5)
```

### 2.2 従量課金の変形パターン

| パターン | 課金単位 | 適用例 | メリット |
|---------|---------|--------|---------|
| リクエスト単位 | API呼び出し回数 | OpenAI API | 直感的 |
| トークン単位 | 入出力トークン数 | Claude API | 公平 |
| クレジット制 | プリペイドクレジット | Replicate | 先払い |
| 時間単位 | GPU利用秒数 | AWS SageMaker | 正確 |
| 成果単位 | 生成されたコンテンツ数 | Jasper | 価値連動 |

### 2.3 高度な従量課金実装

```python
from datetime import datetime, timedelta
from typing import Optional
import json
import redis


class AdvancedUsageTracker:
    """高度な従量課金トラッカー"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def record_usage(self, user_id: str, usage_type: str,
                     quantity: int, metadata: dict = None) -> dict:
        """利用量を記録"""
        now = datetime.utcnow()
        month_key = now.strftime("%Y-%m")
        day_key = now.strftime("%Y-%m-%d")

        # 月間累計
        monthly_key = f"usage:{user_id}:{usage_type}:{month_key}"
        self.redis.incrby(monthly_key, quantity)
        self.redis.expire(monthly_key, 90 * 86400)  # 90日保持

        # 日次累計
        daily_key = f"usage:{user_id}:{usage_type}:{day_key}"
        self.redis.incrby(daily_key, quantity)
        self.redis.expire(daily_key, 35 * 86400)  # 35日保持

        # 詳細ログ（課金エビデンス用）
        log_entry = {
            "timestamp": now.isoformat(),
            "user_id": user_id,
            "type": usage_type,
            "quantity": quantity,
            "metadata": metadata or {}
        }
        log_key = f"usage_log:{user_id}:{month_key}"
        self.redis.rpush(log_key, json.dumps(log_entry))

        monthly_total = int(self.redis.get(monthly_key) or 0)

        return {
            "recorded": quantity,
            "monthly_total": monthly_total,
            "daily_total": int(self.redis.get(daily_key) or 0)
        }

    def get_usage_summary(self, user_id: str,
                          period: str = "month") -> dict:
        """利用量サマリーを取得"""
        now = datetime.utcnow()

        if period == "month":
            key_pattern = f"usage:{user_id}:*:{now.strftime('%Y-%m')}"
        elif period == "day":
            key_pattern = f"usage:{user_id}:*:{now.strftime('%Y-%m-%d')}"
        else:
            key_pattern = f"usage:{user_id}:*"

        keys = self.redis.keys(key_pattern)
        summary = {}
        for key in keys:
            parts = key.decode().split(":")
            usage_type = parts[2]
            count = int(self.redis.get(key) or 0)
            summary[usage_type] = summary.get(usage_type, 0) + count

        return summary

    def check_quota(self, user_id: str, usage_type: str,
                    limit: int) -> dict:
        """クォータチェック"""
        now = datetime.utcnow()
        month_key = f"usage:{user_id}:{usage_type}:{now.strftime('%Y-%m')}"
        current = int(self.redis.get(month_key) or 0)

        remaining = max(0, limit - current)
        usage_ratio = current / limit if limit > 0 else 0

        alert_level = "normal"
        if usage_ratio >= 1.0:
            alert_level = "exceeded"
        elif usage_ratio >= 0.9:
            alert_level = "critical"
        elif usage_ratio >= 0.75:
            alert_level = "warning"

        return {
            "current": current,
            "limit": limit,
            "remaining": remaining,
            "usage_ratio": round(usage_ratio, 3),
            "alert_level": alert_level,
            "should_notify": alert_level in ("warning", "critical")
        }


class CreditSystem:
    """プリペイドクレジット制の実装"""

    # 操作ごとのクレジット消費量
    CREDIT_COSTS = {
        "text_generation_basic": 1,
        "text_generation_advanced": 5,
        "image_generation_sd": 3,
        "image_generation_dalle": 10,
        "code_generation": 3,
        "code_review": 2,
        "document_analysis": 4,
        "translation": 2,
        "summarization": 2,
        "voice_synthesis": 8
    }

    # クレジットパック定義
    CREDIT_PACKS = {
        "starter": {"credits": 100, "price": 980, "bonus": 0},
        "standard": {"credits": 500, "price": 3980, "bonus": 50},
        "pro": {"credits": 2000, "price": 12800, "bonus": 400},
        "enterprise": {"credits": 10000, "price": 49800, "bonus": 3000}
    }

    def __init__(self, db):
        self.db = db

    def purchase_credits(self, user_id: str,
                         pack_name: str) -> dict:
        """クレジット購入"""
        pack = self.CREDIT_PACKS[pack_name]
        total_credits = pack["credits"] + pack["bonus"]

        self.db.execute(
            "UPDATE users SET credits = credits + %s WHERE id = %s",
            (total_credits, user_id)
        )

        self.db.execute(
            """INSERT INTO credit_transactions
               (user_id, type, amount, pack, price, created_at)
               VALUES (%s, 'purchase', %s, %s, %s, NOW())""",
            (user_id, total_credits, pack_name, pack["price"])
        )

        return {
            "purchased": pack["credits"],
            "bonus": pack["bonus"],
            "total_added": total_credits,
            "price": pack["price"],
            "price_per_credit": round(pack["price"] / total_credits, 1)
        }

    def consume_credits(self, user_id: str,
                        operation: str) -> dict:
        """クレジット消費"""
        cost = self.CREDIT_COSTS.get(operation)
        if cost is None:
            return {"error": f"不明な操作: {operation}"}

        # 残高チェック
        balance = self.db.fetchone(
            "SELECT credits FROM users WHERE id = %s", (user_id,)
        )
        if balance["credits"] < cost:
            return {
                "error": "クレジット不足",
                "required": cost,
                "balance": balance["credits"],
                "suggested_pack": self._suggest_pack(cost)
            }

        # 消費実行
        self.db.execute(
            "UPDATE users SET credits = credits - %s WHERE id = %s",
            (cost, user_id)
        )

        new_balance = balance["credits"] - cost

        return {
            "consumed": cost,
            "operation": operation,
            "balance": new_balance,
            "low_balance_warning": new_balance < 10
        }

    def _suggest_pack(self, needed: int) -> dict:
        """推奨パックを提案"""
        for name, pack in self.CREDIT_PACKS.items():
            if pack["credits"] >= needed:
                return {"pack": name, "price": pack["price"]}
        return {"pack": "enterprise", "price": 49800}
```

### 2.4 従量課金のメータリングインフラ

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
from collections import defaultdict


class MeteringEventType(Enum):
    API_CALL = "api_call"
    TOKEN_USAGE = "token_usage"
    GPU_SECONDS = "gpu_seconds"
    STORAGE_MB = "storage_mb"
    BANDWIDTH_MB = "bandwidth_mb"


@dataclass
class MeteringEvent:
    """メータリングイベント"""
    user_id: str
    event_type: MeteringEventType
    quantity: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


class MeteringPipeline:
    """メータリングパイプライン"""

    def __init__(self, batch_size: int = 100,
                 flush_interval: float = 5.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.buffer: list[MeteringEvent] = []
        self.aggregated: dict = defaultdict(float)

    async def record(self, event: MeteringEvent):
        """イベントを記録（バッファリング）"""
        self.buffer.append(event)

        # バッチサイズに達したらフラッシュ
        if len(self.buffer) >= self.batch_size:
            await self.flush()

    async def flush(self):
        """バッファをデータベースにフラッシュ"""
        if not self.buffer:
            return

        events = self.buffer.copy()
        self.buffer.clear()

        # 集約
        aggregated = defaultdict(lambda: defaultdict(float))
        for event in events:
            key = (event.user_id, event.event_type.value)
            month = event.timestamp.strftime("%Y-%m")
            aggregated[key][month] += event.quantity

        # DB書き込み（バッチ）
        batch_inserts = []
        for (user_id, event_type), months in aggregated.items():
            for month, quantity in months.items():
                batch_inserts.append({
                    "user_id": user_id,
                    "event_type": event_type,
                    "month": month,
                    "quantity": quantity
                })

        await self._bulk_upsert(batch_inserts)
        return {"flushed": len(events), "aggregated": len(batch_inserts)}

    async def _bulk_upsert(self, records: list[dict]):
        """バルクUPSERT（実装はDB依存）"""
        # PostgreSQLの場合:
        # INSERT INTO metering (user_id, event_type, month, quantity)
        # VALUES ... ON CONFLICT (user_id, event_type, month)
        # DO UPDATE SET quantity = metering.quantity + EXCLUDED.quantity
        pass

    async def start_periodic_flush(self):
        """定期フラッシュタスク"""
        while True:
            await asyncio.sleep(self.flush_interval)
            await self.flush()


class InvoiceGenerator:
    """請求書生成"""

    def __init__(self, pricing_config: dict):
        self.pricing = pricing_config

    def generate_invoice(self, user_id: str,
                         usage: dict, month: str) -> dict:
        """月次請求書を生成"""
        line_items = []
        total = 0

        for event_type, quantity in usage.items():
            rate = self.pricing.get(event_type, {})
            if not rate:
                continue

            # 階段型課金の計算
            cost = self._calculate_tiered_cost(quantity, rate["tiers"])
            line_items.append({
                "description": rate["description"],
                "quantity": quantity,
                "unit": rate["unit"],
                "amount": cost
            })
            total += cost

        # 最低課金額チェック
        minimum = self.pricing.get("minimum_charge", 0)
        if total < minimum:
            line_items.append({
                "description": "最低利用料金調整",
                "quantity": 1,
                "unit": "式",
                "amount": minimum - total
            })
            total = minimum

        return {
            "user_id": user_id,
            "month": month,
            "line_items": line_items,
            "subtotal": total,
            "tax": round(total * 0.10),  # 消費税10%
            "total": round(total * 1.10),
            "due_date": f"{month}-28",
            "status": "draft"
        }

    def _calculate_tiered_cost(self, quantity: float,
                                tiers: list[dict]) -> float:
        """階段型コスト計算"""
        total = 0
        remaining = quantity

        for tier in tiers:
            if remaining <= 0:
                break
            tier_limit = tier.get("up_to", float("inf"))
            prev_limit = tier.get("from", 0)
            units = min(remaining, tier_limit - prev_limit)
            total += units * tier["rate"]
            remaining -= units

        return round(total)
```

---

## 3. サブスクリプションモデル

### 3.1 プラン設計

```
サブスクリプション 3プラン設計:

  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │  Starter │    │   Pro    │    │Enterprise│
  │  ¥2,980  │    │  ¥9,800  │    │ ¥49,800  │
  │   /月    │    │   /月    │    │   /月    │
  ├──────────┤    ├──────────┤    ├──────────┤
  │ 100回/月 │    │ 1000回/月│    │ 無制限   │
  │ 基本機能 │    │ 全機能   │    │ 全機能   │
  │ メール   │    │ チャット │    │ 専任担当 │
  │ サポート │    │ サポート │    │ SLA 99.9%│
  │          │    │ API連携  │    │ カスタム │
  │          │    │ チーム3人│    │ SSO/SAML │
  └──────────┘    └──────────┘    └──────────┘
       │               │               │
    フリーミアム     ★ 主力プラン     アカウント
    からの転換       （売上の60%）     マネージャー
```

### 3.2 Stripe実装

```python
import stripe

class SubscriptionManager:
    """Stripe サブスクリプション管理"""

    PLANS = {
        "starter": {
            "price_id": "price_starter_monthly",
            "amount": 2980,
            "credits": 100,
            "features": ["basic_generation", "email_support"]
        },
        "pro": {
            "price_id": "price_pro_monthly",
            "amount": 9800,
            "credits": 1000,
            "features": ["all_features", "api_access", "team_3"]
        },
        "enterprise": {
            "price_id": "price_enterprise_monthly",
            "amount": 49800,
            "credits": -1,  # 無制限
            "features": ["all_features", "api_access",
                        "unlimited_team", "sso", "sla"]
        }
    }

    def __init__(self, api_key: str):
        stripe.api_key = api_key

    def create_subscription(self, customer_id: str,
                            plan: str) -> dict:
        """サブスクリプション作成"""
        plan_config = self.PLANS[plan]
        subscription = stripe.Subscription.create(
            customer=customer_id,
            items=[{"price": plan_config["price_id"]}],
            payment_behavior="default_incomplete",
            expand=["latest_invoice.payment_intent"]
        )
        return {
            "subscription_id": subscription.id,
            "client_secret": (
                subscription.latest_invoice
                .payment_intent.client_secret
            ),
            "status": subscription.status
        }

    def handle_usage_overage(self, user_id: str,
                              current_usage: int) -> dict:
        """利用量超過時の処理"""
        user = get_user(user_id)
        plan = self.PLANS[user.plan]
        limit = plan["credits"]

        if limit == -1:  # 無制限プラン
            return {"status": "ok"}

        if current_usage >= limit:
            return {
                "status": "limit_reached",
                "options": [
                    {"action": "upgrade", "plan": "pro",
                     "message": "Proプランにアップグレードで10倍の利用量"},
                    {"action": "addon", "amount": 980,
                     "credits": 100,
                     "message": "追加100回 ¥980で購入"},
                    {"action": "wait",
                     "message": f"次月リセット: {next_reset_date()}"}
                ]
            }
        return {"status": "ok", "remaining": limit - current_usage}
```

### 3.3 年間プランと割引設計

```python
class AnnualPlanManager:
    """年間プラン管理"""

    def __init__(self):
        self.discount_rate = 0.20  # 年間プランで20%割引
        self.plans = {
            "starter": {
                "monthly": 2980,
                "annual_monthly": 2384,  # 2980 * 0.8
                "annual_total": 28608   # 2384 * 12
            },
            "pro": {
                "monthly": 9800,
                "annual_monthly": 7840,
                "annual_total": 94080
            },
            "enterprise": {
                "monthly": 49800,
                "annual_monthly": 39840,
                "annual_total": 478080
            }
        }

    def calculate_savings(self, plan: str) -> dict:
        """年間プランの節約額を計算"""
        p = self.plans[plan]
        monthly_total = p["monthly"] * 12
        annual_total = p["annual_total"]
        savings = monthly_total - annual_total

        return {
            "plan": plan,
            "monthly_price": f"¥{p['monthly']:,}/月",
            "annual_price": f"¥{p['annual_monthly']:,}/月（年払い）",
            "annual_total": f"¥{annual_total:,}/年",
            "savings": f"¥{savings:,}/年お得",
            "savings_months": f"約{savings / p['monthly']:.1f}ヶ月分お得",
            "message": f"年間プランで¥{savings:,}（"
                      f"{self.discount_rate*100:.0f}%）お得！"
        }

    def offer_annual_upgrade(self, user_id: str,
                              current_plan: str,
                              months_on_monthly: int) -> dict:
        """月払いユーザーへの年間プラン提案"""
        if months_on_monthly < 3:
            return {"offer": False, "reason": "利用期間が短い"}

        savings = self.calculate_savings(current_plan)
        p = self.plans[current_plan]

        # 過去の月払い合計
        past_spend = p["monthly"] * months_on_monthly

        return {
            "offer": True,
            "user_id": user_id,
            "current_monthly_spend": f"¥{p['monthly']:,}/月",
            "annual_offer": savings,
            "pitch": f"過去{months_on_monthly}ヶ月で"
                    f"¥{past_spend:,}お支払いいただきました。"
                    f"年間プランなら{savings['savings']}お得です！",
            "urgency": "今月中のお切り替えで初月無料"
        }


class TrialManager:
    """トライアル管理"""

    TRIAL_CONFIGS = {
        "standard": {
            "duration_days": 14,
            "plan": "pro",
            "requires_card": False,
            "conversion_target": 0.25  # 25%
        },
        "premium": {
            "duration_days": 7,
            "plan": "enterprise",
            "requires_card": True,
            "conversion_target": 0.40  # 40%
        },
        "extended": {
            "duration_days": 30,
            "plan": "pro",
            "requires_card": False,
            "conversion_target": 0.15
        }
    }

    def start_trial(self, user_id: str,
                    trial_type: str = "standard") -> dict:
        """トライアル開始"""
        config = self.TRIAL_CONFIGS[trial_type]
        end_date = datetime.utcnow() + timedelta(
            days=config["duration_days"]
        )

        return {
            "user_id": user_id,
            "trial_type": trial_type,
            "plan": config["plan"],
            "end_date": end_date.isoformat(),
            "requires_card": config["requires_card"],
            "features_unlocked": "all",
            "reminder_schedule": [
                {"day": config["duration_days"] - 3,
                 "type": "email", "subject": "トライアル残り3日"},
                {"day": config["duration_days"] - 1,
                 "type": "email", "subject": "トライアル明日終了"},
                {"day": config["duration_days"],
                 "type": "in_app", "subject": "トライアル終了"}
            ]
        }

    def check_trial_engagement(self, user_id: str,
                                usage_data: dict) -> dict:
        """トライアルユーザーのエンゲージメント分析"""
        score = 0
        signals = []

        if usage_data.get("days_active", 0) >= 5:
            score += 30
            signals.append("アクティブ日数5日以上")
        if usage_data.get("features_used", 0) >= 3:
            score += 25
            signals.append("3機能以上を利用")
        if usage_data.get("api_connected", False):
            score += 20
            signals.append("API連携済み")
        if usage_data.get("team_invited", False):
            score += 15
            signals.append("チームメンバー招待済み")
        if usage_data.get("export_used", False):
            score += 10
            signals.append("エクスポート機能利用")

        likelihood = "high" if score >= 60 else (
            "medium" if score >= 30 else "low"
        )

        return {
            "user_id": user_id,
            "engagement_score": score,
            "conversion_likelihood": likelihood,
            "positive_signals": signals,
            "recommended_action": self._get_action(likelihood)
        }

    def _get_action(self, likelihood: str) -> str:
        actions = {
            "high": "割引なしで転換促進メール送信",
            "medium": "10%割引オファーで転換促進",
            "low": "トライアル延長オファー（+7日）"
        }
        return actions.get(likelihood, "標準フォローアップ")
```

### 3.4 プランアップグレード/ダウングレードの処理

```python
class PlanChangeManager:
    """プラン変更管理"""

    def __init__(self, stripe_key: str):
        stripe.api_key = stripe_key

    def upgrade_plan(self, user_id: str,
                     from_plan: str, to_plan: str) -> dict:
        """アップグレード処理"""
        user = get_user(user_id)
        sub = stripe.Subscription.retrieve(user.subscription_id)

        # 日割り計算
        current_period_end = datetime.fromtimestamp(
            sub.current_period_end
        )
        days_remaining = (current_period_end - datetime.utcnow()).days
        total_days = 30  # 近似

        from_price = self.PLANS[from_plan]["amount"]
        to_price = self.PLANS[to_plan]["amount"]

        # プロレーション（日割り差額）
        proration = round(
            (to_price - from_price) * days_remaining / total_days
        )

        # Stripeで即時アップグレード
        stripe.Subscription.modify(
            sub.id,
            items=[{
                "id": sub["items"]["data"][0].id,
                "price": self.PLANS[to_plan]["price_id"]
            }],
            proration_behavior="create_prorations"
        )

        return {
            "action": "upgrade",
            "from_plan": from_plan,
            "to_plan": to_plan,
            "proration_charge": proration,
            "effective": "即時",
            "new_limits": self.PLANS[to_plan]["credits"],
            "message": f"{to_plan}プランへのアップグレード完了！"
                      f"差額¥{proration:,}を日割り請求します。"
        }

    def downgrade_plan(self, user_id: str,
                       from_plan: str, to_plan: str) -> dict:
        """ダウングレード処理"""
        user = get_user(user_id)
        sub = stripe.Subscription.retrieve(user.subscription_id)

        # ダウングレードは期末に適用
        stripe.Subscription.modify(
            sub.id,
            items=[{
                "id": sub["items"]["data"][0].id,
                "price": self.PLANS[to_plan]["price_id"]
            }],
            proration_behavior="none"  # 日割り返金なし
        )

        period_end = datetime.fromtimestamp(
            sub.current_period_end
        ).strftime("%Y-%m-%d")

        return {
            "action": "downgrade",
            "from_plan": from_plan,
            "to_plan": to_plan,
            "effective_date": period_end,
            "message": f"現在の請求期間終了({period_end})後に"
                      f"{to_plan}プランに変更されます。"
                      f"それまでは{from_plan}の機能をご利用いただけます。"
        }

    def handle_churn_prevention(self, user_id: str,
                                 cancel_reason: str) -> dict:
        """解約防止オファー"""
        offers = {
            "too_expensive": {
                "offer": "3ヶ月間50%割引",
                "discount_pct": 50,
                "duration_months": 3,
                "message": "特別価格でご継続いただけます"
            },
            "not_using_enough": {
                "offer": "下位プランへの変更",
                "action": "downgrade_suggestion",
                "message": "より適したプランをご提案します"
            },
            "missing_feature": {
                "offer": "機能リクエスト優先対応",
                "action": "feature_request",
                "message": "ご要望の機能を優先的に開発いたします"
            },
            "competitor": {
                "offer": "6ヶ月間40%割引 + 優先サポート",
                "discount_pct": 40,
                "duration_months": 6,
                "message": "乗り換え防止の特別オファーです"
            },
            "other": {
                "offer": "1ヶ月無料延長",
                "action": "free_month",
                "message": "もう1ヶ月無料でお試しください"
            }
        }

        reason_key = cancel_reason if cancel_reason in offers else "other"
        offer = offers[reason_key]

        return {
            "user_id": user_id,
            "cancel_reason": cancel_reason,
            "retention_offer": offer,
            "escalate_to_human": cancel_reason == "competitor"
        }
```

---

## 4. フリーミアムモデル

### 4.1 フリーミアム設計の黄金比率

```
フリーミアム コンバージョンファネル:

  100% ┤ ■■■■■■■■■■ 無料ユーザー
       │
   30% ┤ ■■■        アクティブユーザー（週1以上利用）
       │
   10% ┤ ■          パワーユーザー（制限にぶつかる）
       │
  3-5% ┤ ▪          有料転換ユーザー
       │
  0.5% ┤ ·          Enterprise転換
       └──────────────────────────────────────
                    目標転換率
```

### 4.2 無料/有料の境界設計

```python
# フリーミアム境界設計
freemium_design = {
    "free_tier": {
        "purpose": "価値体験 + バイラル獲得",
        "limits": {
            "generations_per_month": 10,
            "output_quality": "standard",  # GPT-3.5相当
            "export_format": ["txt"],
            "history_retention": "7日",
            "watermark": True,
            "api_access": False
        },
        "must_provide": [
            "コア機能の体験（制限付き）",
            "十分な回数で価値を実感",
            "シェア機能（バイラル）"
        ]
    },
    "paid_tier": {
        "purpose": "ヘビーユーザーの収益化",
        "unlocks": {
            "generations_per_month": 1000,
            "output_quality": "premium",  # GPT-4相当
            "export_format": ["txt", "docx", "pdf", "html"],
            "history_retention": "無制限",
            "watermark": False,
            "api_access": True,
            "team_features": True
        },
        "trigger_points": [
            "月10回の制限到達",
            "高品質モデルの利用",
            "エクスポート時",
            "チーム招待時"
        ]
    }
}
```

### 4.3 フリーミアム転換率最適化

```python
class FreemiumOptimizer:
    """フリーミアム転換率最適化エンジン"""

    def __init__(self):
        self.conversion_triggers = []
        self.paywall_events = []

    def analyze_conversion_funnel(self, users: list[dict]) -> dict:
        """コンバージョンファネル分析"""
        total = len(users)
        if total == 0:
            return {"error": "ユーザーデータなし"}

        stages = {
            "registered": total,
            "activated": sum(1 for u in users if u.get("activated")),
            "engaged": sum(1 for u in users
                          if u.get("sessions", 0) >= 3),
            "power_user": sum(1 for u in users
                             if u.get("hit_limit", False)),
            "converted": sum(1 for u in users
                            if u.get("plan") != "free"),
            "retained": sum(1 for u in users
                           if u.get("months_paid", 0) >= 3)
        }

        funnel = {}
        prev_count = total
        for stage, count in stages.items():
            funnel[stage] = {
                "count": count,
                "rate_from_total": f"{count/total*100:.1f}%",
                "rate_from_prev": f"{count/prev_count*100:.1f}%"
                                  if prev_count > 0 else "N/A"
            }
            prev_count = count

        return funnel

    def identify_conversion_opportunities(
        self, user_id: str, behavior: dict
    ) -> list[dict]:
        """転換機会の特定"""
        opportunities = []

        # 制限到達時
        if behavior.get("usage_ratio", 0) >= 0.8:
            opportunities.append({
                "trigger": "usage_limit_approaching",
                "urgency": "high",
                "message": "今月の無料枠の80%を使用しました。"
                          "Proプランで制限なしにアップグレード！",
                "cta": "Proプランを見る",
                "discount": None
            })

        # 高品質機能を試した時
        if behavior.get("tried_premium_feature", False):
            opportunities.append({
                "trigger": "premium_feature_tease",
                "urgency": "medium",
                "message": "GPT-4の高品質出力をお試しいただけます。"
                          "Proプランで常にご利用可能です。",
                "cta": "14日間無料トライアル",
                "discount": None
            })

        # チーム招待時
        if behavior.get("invite_attempted", False):
            opportunities.append({
                "trigger": "team_feature_gate",
                "urgency": "high",
                "message": "チーム機能はProプラン以上でご利用いただけます。",
                "cta": "チームプランを見る",
                "discount": "初月50%OFF"
            })

        # 長期無料ユーザー
        if behavior.get("days_on_free", 0) >= 30:
            if behavior.get("sessions", 0) >= 10:
                opportunities.append({
                    "trigger": "engaged_free_user",
                    "urgency": "low",
                    "message": "1ヶ月以上ご利用いただきありがとうございます！"
                              "特別割引でProプランをお試しください。",
                    "cta": "特別オファーを見る",
                    "discount": "初年度30%OFF"
                })

        return opportunities

    def design_paywall(self, context: str) -> dict:
        """コンテキスト別ペイウォール設計"""
        paywalls = {
            "soft": {
                "description": "ソフトペイウォール",
                "behavior": "機能は使えるが品質や速度を制限",
                "example": "無料版は低品質モデル、有料版は高品質モデル",
                "conversion_rate": "2-4%",
                "user_experience": "良い（ストレス低）"
            },
            "hard": {
                "description": "ハードペイウォール",
                "behavior": "制限到達で完全にブロック",
                "example": "月10回の上限に達したら使用不可",
                "conversion_rate": "4-8%",
                "user_experience": "やや悪い（フラストレーション）"
            },
            "metered": {
                "description": "メータードペイウォール",
                "behavior": "一定回数は無料、超過分は自動課金",
                "example": "月50回無料、51回目から¥10/回",
                "conversion_rate": "5-10%",
                "user_experience": "中程度（予測可能）"
            },
            "feature": {
                "description": "フィーチャーペイウォール",
                "behavior": "高度な機能のみ有料",
                "example": "基本生成は無料、API/エクスポートは有料",
                "conversion_rate": "3-6%",
                "user_experience": "良い（コア機能は無料）"
            }
        }

        return paywalls.get(context, paywalls["soft"])
```

### 4.4 フリーミアムの経済学

```python
class FreemiumEconomics:
    """フリーミアムの経済分析"""

    def calculate_unit_economics(
        self,
        total_users: int,
        free_users: int,
        paid_users: int,
        arpu: float,  # Average Revenue Per User (有料)
        cost_per_free_user: float,
        cost_per_paid_user: float,
        cac: float  # Customer Acquisition Cost
    ) -> dict:
        """ユニットエコノミクス計算"""
        conversion_rate = paid_users / total_users if total_users > 0 else 0

        # 収益
        monthly_revenue = paid_users * arpu
        annual_revenue = monthly_revenue * 12

        # コスト
        free_user_cost = free_users * cost_per_free_user
        paid_user_cost = paid_users * cost_per_paid_user
        total_cost = free_user_cost + paid_user_cost
        acquisition_cost = total_users * cac

        # 無料ユーザーのコストを有料ユーザーが負担
        effective_cost_per_paid_user = total_cost / paid_users if paid_users > 0 else 0

        # LTV計算（平均契約期間18ヶ月と仮定）
        avg_lifetime_months = 18
        ltv = arpu * avg_lifetime_months
        ltv_cac_ratio = ltv / cac if cac > 0 else 0

        return {
            "metrics": {
                "conversion_rate": f"{conversion_rate*100:.2f}%",
                "monthly_revenue": f"¥{monthly_revenue:,.0f}",
                "annual_revenue": f"¥{annual_revenue:,.0f}",
                "arpu": f"¥{arpu:,.0f}",
                "free_user_cost_monthly": f"¥{free_user_cost:,.0f}",
                "effective_cost_per_paid": f"¥{effective_cost_per_paid_user:,.0f}",
                "gross_margin": f"{(monthly_revenue - total_cost) / monthly_revenue * 100:.1f}%",
                "ltv": f"¥{ltv:,.0f}",
                "cac": f"¥{cac:,.0f}",
                "ltv_cac_ratio": f"{ltv_cac_ratio:.1f}x"
            },
            "health": {
                "conversion_rate_ok": conversion_rate >= 0.03,
                "ltv_cac_ok": ltv_cac_ratio >= 3.0,
                "margin_ok": (monthly_revenue - total_cost) / monthly_revenue >= 0.60
            },
            "recommendations": self._generate_recommendations(
                conversion_rate, ltv_cac_ratio, free_user_cost
            )
        }

    def _generate_recommendations(
        self, cvr: float, ltv_cac: float, free_cost: float
    ) -> list[str]:
        """改善推奨事項"""
        recs = []
        if cvr < 0.02:
            recs.append("転換率が低い: 無料プランの制限を強化するか、"
                       "有料プランの価値訴求を改善する")
        if cvr > 0.10:
            recs.append("転換率が高すぎる: 無料プランが制限しすぎの可能性。"
                       "バイラル成長が阻害されている恐れ")
        if ltv_cac < 3.0:
            recs.append("LTV/CAC比が低い: 解約率の改善またはARPUの向上が必要")
        if free_cost > 100:
            recs.append("無料ユーザーのコストが高い: キャッシュの導入や"
                       "軽量モデルへの切り替えを検討")
        return recs


# 使用例
economics = FreemiumEconomics()
result = economics.calculate_unit_economics(
    total_users=10000,
    free_users=9500,
    paid_users=500,
    arpu=9800,
    cost_per_free_user=50,      # 無料ユーザー1人月¥50
    cost_per_paid_user=2500,    # 有料ユーザー1人月¥2,500
    cac=3000                    # 獲得コスト¥3,000/人
)
```

---

## 5. ハイブリッド価格モデル

### 5.1 サブスク + 従量制ハイブリッド

最も成功しているAI SaaSの多くが採用するモデル。基本料金で一定量を含み、超過分を従量課金する。

```python
class HybridPricingEngine:
    """ハイブリッド価格エンジン"""

    PLANS = {
        "starter": {
            "base_price": 2980,
            "included_credits": 100,
            "overage_rate": 50,  # ¥50/回（超過分）
            "features": ["basic"],
            "max_overage": 30000  # 月間超過上限
        },
        "pro": {
            "base_price": 9800,
            "included_credits": 1000,
            "overage_rate": 30,
            "features": ["basic", "advanced", "api"],
            "max_overage": 100000
        },
        "enterprise": {
            "base_price": 49800,
            "included_credits": 10000,
            "overage_rate": 15,
            "features": ["basic", "advanced", "api",
                        "sso", "sla", "custom"],
            "max_overage": None  # 上限なし
        }
    }

    def calculate_monthly_bill(self, plan: str,
                                usage: int) -> dict:
        """月額請求額を計算"""
        config = self.PLANS[plan]
        base = config["base_price"]
        included = config["included_credits"]
        overage_rate = config["overage_rate"]
        max_overage = config["max_overage"]

        overage_units = max(0, usage - included)
        overage_charge = overage_units * overage_rate

        if max_overage is not None:
            overage_charge = min(overage_charge, max_overage)

        total = base + overage_charge

        return {
            "plan": plan,
            "base_charge": base,
            "included_usage": included,
            "actual_usage": usage,
            "overage_units": overage_units,
            "overage_charge": overage_charge,
            "total": total,
            "effective_rate_per_unit": round(total / usage, 1)
                                       if usage > 0 else 0,
            "within_included": usage <= included,
            "recommendation": self._recommend_plan(plan, usage)
        }

    def _recommend_plan(self, current_plan: str,
                        usage: int) -> str:
        """プラン推奨"""
        current = self.PLANS[current_plan]
        included = current["included_credits"]

        if usage > included * 1.5:
            # 次のプランの方が安い可能性
            plans = list(self.PLANS.keys())
            idx = plans.index(current_plan)
            if idx < len(plans) - 1:
                next_plan = plans[idx + 1]
                next_config = self.PLANS[next_plan]
                current_cost = (current["base_price"] +
                               max(0, usage - included) *
                               current["overage_rate"])
                next_cost = next_config["base_price"]
                if next_cost < current_cost:
                    return (f"{next_plan}プランへのアップグレード推奨"
                           f"（¥{current_cost - next_cost:,}お得）")
        elif usage < included * 0.3:
            plans = list(self.PLANS.keys())
            idx = plans.index(current_plan)
            if idx > 0:
                prev_plan = plans[idx - 1]
                return f"利用量が少ないため{prev_plan}プランも検討ください"

        return "現在のプランが最適です"

    def simulate_plans(self, expected_usage: int) -> list[dict]:
        """全プランのコストシミュレーション"""
        results = []
        for plan_name, config in self.PLANS.items():
            result = self.calculate_monthly_bill(plan_name,
                                                  expected_usage)
            results.append({
                "plan": plan_name,
                "monthly_cost": result["total"],
                "effective_rate": result["effective_rate_per_unit"],
                "overage": result["overage_charge"]
            })

        # コスト順ソート
        results.sort(key=lambda x: x["monthly_cost"])
        results[0]["best_value"] = True

        return results
```

### 5.2 成果報酬型ハイブリッド

```python
class OutcomeBasedPricing:
    """成果報酬型の価格モデル"""

    def __init__(self):
        self.outcome_definitions = {
            "lead_generated": {
                "description": "AI生成リードの獲得",
                "base_rate": 500,  # ¥500/リード
                "quality_multiplier": {
                    "hot": 3.0,    # ホットリード: ¥1,500
                    "warm": 1.5,   # ウォームリード: ¥750
                    "cold": 1.0    # コールドリード: ¥500
                }
            },
            "document_processed": {
                "description": "AI文書処理の完了",
                "base_rate": 100,
                "complexity_multiplier": {
                    "simple": 1.0,   # 1-5ページ
                    "medium": 2.0,   # 6-20ページ
                    "complex": 5.0   # 21+ページ
                }
            },
            "customer_resolved": {
                "description": "AI顧客対応の解決",
                "base_rate": 200,
                "channel_multiplier": {
                    "chat": 1.0,
                    "email": 1.5,
                    "phone": 3.0
                }
            }
        }

    def calculate_outcome_charge(
        self, outcome_type: str, quantity: int,
        quality_or_complexity: str
    ) -> dict:
        """成果報酬額の計算"""
        definition = self.outcome_definitions[outcome_type]
        multiplier_key = list(definition.keys())[-1]  # 最後のmultiplier
        multiplier_dict = definition[multiplier_key]
        multiplier = multiplier_dict.get(quality_or_complexity, 1.0)

        unit_price = definition["base_rate"] * multiplier
        total = unit_price * quantity

        return {
            "outcome_type": outcome_type,
            "quantity": quantity,
            "quality": quality_or_complexity,
            "unit_price": unit_price,
            "total": total,
            "breakdown": f"{quantity} × ¥{unit_price:,.0f} = ¥{total:,.0f}"
        }

    def design_hybrid_plan(self, base_monthly: float,
                           outcome_configs: list[dict]) -> dict:
        """ハイブリッドプラン設計"""
        return {
            "base_fee": {
                "amount": base_monthly,
                "includes": "プラットフォーム利用料 + 基本サポート",
                "billing": "月初固定請求"
            },
            "outcome_fees": [
                {
                    "type": config["type"],
                    "rate": config["rate"],
                    "cap": config.get("monthly_cap"),
                    "billing": "月末実績ベース請求"
                }
                for config in outcome_configs
            ],
            "minimum_commitment": base_monthly,
            "estimated_monthly": base_monthly + sum(
                c["rate"] * c.get("expected_volume", 0)
                for c in outcome_configs
            )
        }
```

---

## 6. 心理的価格設計

### 6.1 行動経済学に基づく価格テクニック

```python
class PsychologicalPricing:
    """心理的価格設計"""

    TECHNIQUES = {
        "charm_pricing": {
            "name": "端数価格（チャームプライシング）",
            "description": "¥9,800は¥10,000より大幅に安く感じる",
            "implementation": "価格を80/90で終わらせる",
            "effectiveness": "コンバージョン+8-15%",
            "examples": ["¥2,980", "¥9,800", "¥49,800"]
        },
        "anchoring": {
            "name": "アンカリング効果",
            "description": "高価格を先に見せることで中価格が安く感じる",
            "implementation": "Enterpriseプランを最初（左側）に表示",
            "effectiveness": "Pro選択率+20-30%",
            "examples": ["Enterprise ¥49,800 → Pro ¥9,800 が安く見える"]
        },
        "decoy_effect": {
            "name": "おとり効果（デコイ効果）",
            "description": "3プランで中間を魅力的に見せる",
            "implementation": "上位プランとわずかな差の中間プランを設計",
            "effectiveness": "中間プラン選択率+30-40%",
            "examples": [
                "Starter ¥2,980 (100回)",
                "Pro ¥9,800 (1000回) ★おすすめ",
                "Enterprise ¥49,800 (無制限)"
            ]
        },
        "loss_aversion": {
            "name": "損失回避",
            "description": "「今ならX%OFF」で機会損失を強調",
            "implementation": "期間限定割引 + カウントダウンタイマー",
            "effectiveness": "即時転換率+25-35%",
            "caution": "過度な使用はブランド毀損リスク"
        },
        "round_number_avoidance": {
            "name": "端数回避",
            "description": "B2Bでは逆にラウンドナンバーが信頼感を与える",
            "implementation": "Enterprise向けは¥50,000/月のようなきりの良い数字",
            "effectiveness": "B2B商談成約率+5-10%",
            "examples": ["¥50,000/月", "¥500,000/年"]
        }
    }

    def apply_charm_pricing(self, base_price: float) -> dict:
        """チャームプライシングの適用"""
        options = []
        # 80で終わる価格
        charm_80 = round(base_price / 100) * 100 - 20
        options.append({"price": charm_80, "ending": "80"})
        # 00で終わる価格（切り上げ）
        round_up = round(base_price / 1000) * 1000
        options.append({"price": round_up, "ending": "000"})
        # 980で終わる価格
        charm_980 = round(base_price / 1000) * 1000 - 20
        if charm_980 < 1000:
            charm_980 = 980
        options.append({"price": charm_980, "ending": "980"})

        return {
            "original": base_price,
            "options": options,
            "recommendation": options[2],  # 980が最も効果的
            "reasoning": "日本市場では980円台が最も効果が高い"
        }

    def design_pricing_page(self, plans: list[dict]) -> dict:
        """心理的に最適化された価格ページ設計"""
        if len(plans) != 3:
            return {"error": "3プラン構成を推奨"}

        return {
            "layout": {
                "order": "右から左: Enterprise → Pro → Starter",
                "highlight": "中間プラン（Pro）を視覚的に強調",
                "recommended_badge": "Pro に「最も人気」バッジ",
                "cta_color": "Proのみ主要カラー、他はグレー系"
            },
            "copy_techniques": {
                "starter": {
                    "label": "個人向け",
                    "cta": "まずはこちらから",
                    "emphasis": "リスクなし"
                },
                "pro": {
                    "label": "★ 最も人気",
                    "cta": "今すぐ始める",
                    "emphasis": "最もお得"
                },
                "enterprise": {
                    "label": "チーム・法人向け",
                    "cta": "お問い合わせ",
                    "emphasis": "フルサポート"
                }
            },
            "social_proof": {
                "position": "価格表の上部",
                "content": "10,000社以上が利用中",
                "logos": "認知度の高い企業ロゴ 5-8社"
            },
            "guarantee": {
                "position": "価格表の下部",
                "content": "30日間返金保証",
                "icon": "シールド/鍵アイコン"
            }
        }
```

### 6.2 価格表示の最適化

```python
class PriceDisplayOptimizer:
    """価格表示の最適化"""

    def format_price_display(self, monthly_price: float,
                              annual_price: float) -> dict:
        """最適な価格表示形式"""
        monthly_if_annual = annual_price / 12
        savings = monthly_price - monthly_if_annual
        savings_pct = savings / monthly_price * 100

        return {
            "primary_display": {
                "format": f"¥{monthly_if_annual:,.0f}/月",
                "subtext": f"年払い ¥{annual_price:,.0f}/年",
                "reasoning": "最小月額を強調（年払い時）"
            },
            "savings_display": {
                "format": f"月額¥{savings:,.0f}お得",
                "percentage": f"{savings_pct:.0f}%OFF",
                "annual_savings": f"年間¥{savings*12:,.0f}お得",
                "best_format": f"2ヶ月分無料"
                              if savings_pct >= 15 else
                              f"年間¥{savings*12:,.0f}お得"
            },
            "toggle_design": {
                "default": "annual",  # 年払いをデフォルト選択
                "monthly_label": "月払い",
                "annual_label": "年払い（お得）",
                "annual_badge": f"{savings_pct:.0f}%OFF"
            }
        }

    def price_localization(self, base_price_usd: float,
                           region: str) -> dict:
        """地域別価格最適化"""
        # PPP（購買力平価）係数
        ppp_multipliers = {
            "us": 1.0,
            "jp": 1.1,      # 日本: やや高め
            "eu": 0.95,     # EU: ほぼ同等
            "uk": 0.90,
            "in": 0.25,     # インド: 大幅割引
            "br": 0.35,     # ブラジル: 割引
            "sea": 0.30,    # 東南アジア: 割引
            "kr": 0.85,     # 韓国
            "cn": 0.40,     # 中国
            "au": 1.05      # オーストラリア
        }

        multiplier = ppp_multipliers.get(region, 1.0)
        local_price_usd = base_price_usd * multiplier

        # 通貨変換レート（概算）
        currency_rates = {
            "us": ("USD", 1.0), "jp": ("JPY", 150),
            "eu": ("EUR", 0.92), "uk": ("GBP", 0.79),
            "in": ("INR", 83), "br": ("BRL", 4.9),
            "sea": ("USD", 1.0), "kr": ("KRW", 1300),
            "cn": ("CNY", 7.1), "au": ("AUD", 1.5)
        }

        currency, rate = currency_rates.get(region, ("USD", 1.0))
        local_price = round(local_price_usd * rate)

        return {
            "region": region,
            "base_price_usd": base_price_usd,
            "ppp_adjusted_usd": round(local_price_usd, 2),
            "local_currency": currency,
            "local_price": local_price,
            "discount_from_base": f"{(1-multiplier)*100:.0f}%",
            "display": f"{currency} {local_price:,}"
        }
```

---

## 7. 価格実験と最適化

### 7.1 価格感度分析

```
Van Westendorp 価格感度メーター:

  回答率
  100%┤
     │  ＼安すぎ     高すぎ／
     │    ＼         ／
  50%┤     ＼ PMF  ／
     │      ＼ 価格／
     │       ＼  ／
     │        ＼／  ← 最適価格帯
   0%┤─────────╳────────────────
     └──┬──┬──┬──┬──┬──┬──
      ¥1k ¥3k ¥5k ¥8k ¥10k ¥15k
              価格

  「安すぎて不安」と「高すぎる」の交点 = 最適価格
```

### 7.2 A/Bテスト実装

```python
class PricingExperiment:
    """価格A/Bテスト"""

    def __init__(self):
        self.experiments = {}

    def create_experiment(self, name: str,
                          variants: list[dict]) -> str:
        """価格実験を作成"""
        experiment = {
            "name": name,
            "variants": variants,
            "results": {v["name"]: {"views": 0, "conversions": 0}
                       for v in variants},
            "status": "running"
        }
        self.experiments[name] = experiment
        return name

    def assign_variant(self, user_id: str,
                        experiment_name: str) -> dict:
        """ユーザーをバリアントに割り当て"""
        experiment = self.experiments[experiment_name]
        # 決定的ハッシュで一貫した割り当て
        variant_index = hash(f"{user_id}:{experiment_name}") % len(
            experiment["variants"]
        )
        variant = experiment["variants"][variant_index]
        experiment["results"][variant["name"]]["views"] += 1
        return variant

    def record_conversion(self, experiment_name: str,
                           variant_name: str):
        """コンバージョンを記録"""
        self.experiments[experiment_name]["results"][variant_name][
            "conversions"
        ] += 1

    def get_results(self, experiment_name: str) -> dict:
        """結果を取得"""
        results = self.experiments[experiment_name]["results"]
        for name, data in results.items():
            data["cvr"] = (
                data["conversions"] / data["views"] * 100
                if data["views"] > 0 else 0
            )
        return results

# 使用例
exp = PricingExperiment()
exp.create_experiment("pro_pricing", [
    {"name": "A", "price": 7800, "label": "¥7,800/月"},
    {"name": "B", "price": 9800, "label": "¥9,800/月"},
    {"name": "C", "price": 12800, "label": "¥12,800/月"}
])
```

### 7.3 統計的有意性の判定

```python
import math
from typing import Tuple


class StatisticalSignificance:
    """価格テストの統計的有意性判定"""

    @staticmethod
    def calculate_z_score(
        control_conversions: int, control_views: int,
        variant_conversions: int, variant_views: int
    ) -> Tuple[float, bool]:
        """Z検定によるA/Bテスト有意性判定"""
        p1 = control_conversions / control_views
        p2 = variant_conversions / variant_views

        # プールされた比率
        p_pool = (control_conversions + variant_conversions) / \
                 (control_views + variant_views)

        # 標準誤差
        se = math.sqrt(
            p_pool * (1 - p_pool) *
            (1 / control_views + 1 / variant_views)
        )

        if se == 0:
            return 0, False

        z_score = (p2 - p1) / se
        # 95%信頼水準（Z > 1.96で有意）
        is_significant = abs(z_score) > 1.96

        return round(z_score, 3), is_significant

    @staticmethod
    def calculate_sample_size(
        baseline_cvr: float,
        minimum_detectable_effect: float,
        significance_level: float = 0.05,
        power: float = 0.80
    ) -> int:
        """必要サンプルサイズの計算"""
        # Z値
        z_alpha = 1.96  # 95%信頼水準
        z_beta = 0.84   # 80%検出力

        p1 = baseline_cvr
        p2 = baseline_cvr * (1 + minimum_detectable_effect)

        n = (
            (z_alpha * math.sqrt(2 * p1 * (1 - p1)) +
             z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        ) / (p2 - p1) ** 2

        return math.ceil(n)

    def analyze_experiment(self, experiment_results: dict) -> dict:
        """実験結果の包括的分析"""
        variants = list(experiment_results.items())
        if len(variants) < 2:
            return {"error": "2つ以上のバリアントが必要"}

        # コントロール（最初のバリアント）
        control_name, control_data = variants[0]
        analyses = []

        for name, data in variants[1:]:
            z_score, significant = self.calculate_z_score(
                control_data["conversions"], control_data["views"],
                data["conversions"], data["views"]
            )

            control_cvr = control_data["conversions"] / control_data["views"] * 100
            variant_cvr = data["conversions"] / data["views"] * 100
            lift = (variant_cvr - control_cvr) / control_cvr * 100

            analyses.append({
                "variant": name,
                "control_cvr": f"{control_cvr:.2f}%",
                "variant_cvr": f"{variant_cvr:.2f}%",
                "lift": f"{lift:+.1f}%",
                "z_score": z_score,
                "significant": significant,
                "recommendation": (
                    f"{name}を採用（有意に改善）"
                    if significant and lift > 0
                    else f"{name}は有意差なし、テスト継続"
                    if not significant
                    else f"{name}はコントロールより悪い"
                )
            })

        return {
            "control": control_name,
            "analyses": analyses,
            "conclusion": self._derive_conclusion(analyses)
        }

    def _derive_conclusion(self, analyses: list[dict]) -> str:
        significant_winners = [
            a for a in analyses
            if a["significant"] and float(a["lift"].rstrip("%")) > 0
        ]
        if significant_winners:
            best = max(significant_winners,
                      key=lambda x: float(x["lift"].rstrip("%")))
            return f"推奨: {best['variant']}を採用（{best['lift']}改善）"
        return "有意な勝者なし。テストを継続するか、新バリアントを追加"
```

### 7.4 LTV最大化と価格最適化

```python
class LTVOptimizer:
    """LTV（顧客生涯価値）最大化"""

    def calculate_ltv(
        self,
        arpu: float,
        monthly_churn_rate: float,
        gross_margin: float = 0.70
    ) -> dict:
        """LTVの計算"""
        avg_lifetime_months = 1 / monthly_churn_rate if monthly_churn_rate > 0 else 0
        ltv_gross = arpu * avg_lifetime_months
        ltv_net = ltv_gross * gross_margin

        return {
            "arpu": f"¥{arpu:,.0f}",
            "monthly_churn": f"{monthly_churn_rate*100:.1f}%",
            "avg_lifetime_months": round(avg_lifetime_months, 1),
            "ltv_gross": f"¥{ltv_gross:,.0f}",
            "ltv_net": f"¥{ltv_net:,.0f}",
            "max_cac": f"¥{ltv_net / 3:,.0f}",  # LTV/CAC≥3
            "health": "健全" if avg_lifetime_months >= 12 else "要改善"
        }

    def price_sensitivity_matrix(self, prices: list[float],
                                  churn_rates: list[float]) -> list[dict]:
        """価格と解約率のマトリックス分析"""
        results = []
        for price in prices:
            for churn in churn_rates:
                ltv = price / churn if churn > 0 else 0
                revenue_index = ltv * (1 - churn)  # 粗収益指標
                results.append({
                    "price": price,
                    "churn": f"{churn*100:.0f}%",
                    "ltv": round(ltv),
                    "revenue_index": round(revenue_index),
                })

        # 最適な組み合わせを特定
        best = max(results, key=lambda x: x["revenue_index"])
        for r in results:
            r["optimal"] = (r["price"] == best["price"] and
                           r["churn"] == best["churn"])

        return results

    def cohort_analysis(self, cohorts: dict) -> dict:
        """コホート分析によるLTV予測"""
        predictions = {}
        for cohort_month, data in cohorts.items():
            initial_users = data["initial_users"]
            monthly_retention = data["monthly_active"]

            retention_rates = [
                active / initial_users
                for active in monthly_retention
            ]

            # 累積収益
            arpu = data.get("arpu", 9800)
            cumulative_revenue = [
                sum(monthly_retention[:i+1]) * arpu
                for i in range(len(monthly_retention))
            ]

            # LTV予測（指数減衰フィッティング）
            if len(retention_rates) >= 3:
                avg_decay = sum(
                    retention_rates[i+1] / retention_rates[i]
                    for i in range(len(retention_rates) - 1)
                    if retention_rates[i] > 0
                ) / (len(retention_rates) - 1)

                predicted_ltv = arpu * retention_rates[-1] / (1 - avg_decay)
            else:
                predicted_ltv = arpu * 12  # デフォルト推定

            predictions[cohort_month] = {
                "initial_users": initial_users,
                "current_retention": f"{retention_rates[-1]*100:.1f}%",
                "cumulative_revenue": cumulative_revenue[-1],
                "predicted_ltv": round(predicted_ltv),
                "months_observed": len(monthly_retention)
            }

        return predictions
```

---

## 8. 国際価格戦略

### 8.1 地域別価格設定

```python
class InternationalPricingStrategy:
    """国際価格戦略"""

    REGIONAL_CONFIGS = {
        "tier1": {
            "regions": ["us", "uk", "eu", "au", "jp"],
            "pricing_approach": "標準価格",
            "discount_pct": 0,
            "payment_methods": ["card", "paypal"]
        },
        "tier2": {
            "regions": ["kr", "tw", "sg", "hk"],
            "pricing_approach": "やや割引",
            "discount_pct": 15,
            "payment_methods": ["card", "paypal", "local"]
        },
        "tier3": {
            "regions": ["br", "mx", "th", "my", "ph"],
            "pricing_approach": "大幅割引",
            "discount_pct": 40,
            "payment_methods": ["card", "pix", "local"]
        },
        "tier4": {
            "regions": ["in", "id", "vn", "ng", "pk"],
            "pricing_approach": "PPP割引",
            "discount_pct": 65,
            "payment_methods": ["upi", "local", "card"]
        }
    }

    def get_regional_price(self, base_price_usd: float,
                           country_code: str) -> dict:
        """地域別価格の取得"""
        for tier_name, config in self.REGIONAL_CONFIGS.items():
            if country_code in config["regions"]:
                discount = config["discount_pct"] / 100
                adjusted_price = base_price_usd * (1 - discount)

                return {
                    "country": country_code,
                    "tier": tier_name,
                    "base_price_usd": base_price_usd,
                    "adjusted_price_usd": round(adjusted_price, 2),
                    "discount": f"{config['discount_pct']}%",
                    "payment_methods": config["payment_methods"],
                    "approach": config["pricing_approach"]
                }

        # デフォルト（Tier 1扱い）
        return {
            "country": country_code,
            "tier": "tier1",
            "base_price_usd": base_price_usd,
            "adjusted_price_usd": base_price_usd,
            "discount": "0%",
            "payment_methods": ["card", "paypal"],
            "approach": "標準価格"
        }

    def prevent_arbitrage(self, user_data: dict) -> dict:
        """地域別価格の裁定取引防止"""
        checks = {
            "ip_geolocation": user_data.get("ip_country"),
            "payment_country": user_data.get("card_country"),
            "billing_address": user_data.get("billing_country")
        }

        countries = set(v for v in checks.values() if v)

        if len(countries) > 1:
            return {
                "risk": "high",
                "action": "verify",
                "reason": "複数国の情報が検出されました",
                "detected": checks,
                "recommendation": "最も高い価格帯を適用"
            }

        return {
            "risk": "low",
            "action": "approve",
            "country": countries.pop() if countries else "unknown"
        }
```

---

## 9. アンチパターン

### アンチパターン1: コストを無視した価格設定

```python
# BAD: 競合に合わせて安価に設定、APIコストで赤字
pricing_bad = {
    "plan": "Pro",
    "price": 1980,   # ¥1,980/月
    "api_cost_per_user": 2500,  # API費用 ¥2,500/月
    "result": "使われるほど赤字が拡大"
}

# GOOD: コスト構造を把握した上で価格設定
pricing_good = {
    "plan": "Pro",
    "api_cost_per_user": 2500,
    "target_gross_margin": 0.70,
    "minimum_price": 2500 / (1 - 0.70),  # ¥8,333
    "set_price": 9800,  # マージン込みで¥9,800
    "actual_margin": (9800 - 2500) / 9800  # 74.5%
}
```

### アンチパターン2: 無料プランが豊富すぎる

```python
# BAD: 無料で十分使えてしまう
free_plan_bad = {
    "generations": 100,  # 月100回で十分
    "quality": "premium",  # 最高品質
    "result": "誰も有料にならない（転換率0.5%以下）"
}

# GOOD: 無料で価値は体験できるが、物足りなくなる設計
free_plan_good = {
    "generations": 10,  # 「もっと使いたい」が生まれる量
    "quality": "standard",  # 有料版の品質差を実感
    "result": "転換率3-5%、NPS高い"
}
```

### アンチパターン3: 複雑すぎる価格体系

```python
# BAD: 理解不能な価格体系
pricing_complex_bad = {
    "plans": 7,  # プランが多すぎる
    "add_ons": 12,  # アドオンが多すぎる
    "pricing_page": "スクロールが必要な長さ",
    "result": "顧客が混乱して離脱（直帰率60%+）"
}

# GOOD: シンプルで直感的な価格体系
pricing_simple_good = {
    "plans": 3,  # Starter / Pro / Enterprise
    "add_ons": 2,  # 追加クレジット / 追加メンバー
    "pricing_page": "1画面で全プラン比較可能",
    "result": "理解容易、意思決定が早い"
}
```

### アンチパターン4: 値上げの失敗パターン

```python
# BAD: 突然の大幅値上げ
price_increase_bad = {
    "old_price": 4980,
    "new_price": 9800,
    "increase_pct": "96.8%",
    "notice_period": "2週間",
    "grandfathering": False,
    "result": "大量解約（チャーン率35%増加）、SNSで炎上"
}

# GOOD: 段階的・透明な値上げ
price_increase_good = {
    "old_price": 4980,
    "new_price": 6980,  # 1回目: 40%値上げ
    "future_price": 9800,  # 2回目（6ヶ月後）: さらに値上げ
    "notice_period": "60日前",
    "grandfathering": True,  # 既存ユーザーは旧価格を12ヶ月維持
    "communication": [
        "新機能追加に伴う価格改定",
        "既存ユーザーは12ヶ月間旧価格",
        "年間プランなら旧価格をさらに12ヶ月延長"
    ],
    "result": "チャーン率5%増（許容範囲）、ARPU40%向上"
}
```

---

## 10. FAQ

### Q1: 従量制とサブスクどちらが良い？

**A:** ユースケースで判断する。(1) 利用量が予測しにくい → 従量制（開発者向けAPI等）、(2) 利用量が安定 → サブスク（ビジネスユーザー向け等）、(3) 最強は「サブスク + 従量制のハイブリッド」。基本料金で一定量含み、超過分は従量課金。Slack、Twilio、AWS等の成功例がこのモデル。

### Q2: 値上げのタイミングと方法は？

**A:** 3つの原則。(1) タイミング — PMF達成後、機能追加時、年1回の定期見直し、(2) 方法 — 既存ユーザーは旧価格据え置き（グランドファザリング）+ 新規は新価格、(3) 幅 — 一度に20%以上の値上げは避け、10-15%ずつ段階的に。通知は最低30日前、値上げの理由（新機能追加等）を明確に伝える。

### Q3: AI SaaSの適正粗利率は？

**A:** 業界目安は70-80%。ただしAI SaaSはAPIコストが高いため、立ち上げ期は60%でも許容範囲。改善方法: (1) キャッシュ導入で同一リクエストのAPI呼び出し削減（30-50%削減可能）、(2) 軽量モデルの活用（簡単なタスクはGPT-3.5で十分）、(3) バッチ処理でAPI効率化。年々APIコストが下がるトレンドもあり、粗利は自然に改善する傾向。

### Q4: 競合が大幅に安い場合どうする？

**A:** 価格で勝負しない。(1) 差別化要素を明確にする（精度、速度、サポート品質）、(2) ターゲットセグメントをずらす（SMBではなくEnterpriseにフォーカス）、(3) 無料トライアルで品質を体験させる。価格競争に巻き込まれると全員が負ける。価値ベースの価格設定に徹することが重要。

### Q5: フリーミアムの転換率が低い場合の改善策は？

**A:** 段階的に改善する。(1) 無料プランの制限を見直す（回数を減らすか、機能を制限）、(2) 有料プランの価値を強調するUIの改善（制限到達時のアップグレード導線）、(3) オンボーディングを改善して早期に価値を実感させる（Time to Valueの短縮）、(4) 期間限定割引のトリガーメールを設計。ただし無料プランを過度に制限するとバイラル効果が失われるので注意。

### Q6: B2BとB2Cで価格戦略はどう変わる？

**A:** 大きく異なる。B2Bは(1) 年間契約推奨、(2) シートベース課金、(3) カスタム価格交渉あり、(4) ROI訴求が重要、(5) 端数価格よりきりの良い数字。B2Cは(1) 月額課金が主流、(2) 利用量ベース課金、(3) 固定価格、(4) 感情・利便性訴求、(5) 端数価格（¥980）が効果的。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| コスト構造 | AI APIが売上の20-40%、粗利70%以上が目標 |
| 従量制 | 開発者向け・利用量変動大に最適 |
| サブスク | ビジネスユーザー向け・予測可能な収益 |
| フリーミアム | 転換率3-5%が目標、無料は「もう少し」の設計 |
| ハイブリッド | サブスク基本料 + 従量超過課金が最強 |
| 心理的価格 | アンカリング・おとり効果・端数価格を活用 |
| 国際展開 | PPPに基づく地域別価格設定 |
| 最適化 | A/Bテスト + Van Westendorp + LTV分析 |
| 値上げ | グランドファザリング + 段階的 + 透明性 |

---

## 次に読むべきガイド

- [01-cost-management.md](./01-cost-management.md) — API費用最適化
- [02-scaling-strategy.md](./02-scaling-strategy.md) — スケーリング戦略
- [../01-business/00-ai-saas.md](../01-business/00-ai-saas.md) — AI SaaSプロダクト設計

---

## 参考文献

1. **"Monetizing Innovation" — Madhavan Ramanujam** — 価格設計の体系的手法、SaaS価格の必読書
2. **OpenView Partners "SaaS Pricing" (2024)** — https://openviewpartners.com — SaaS価格ベンチマーク
3. **"The Psychology of Price" — Leigh Caldwell** — 行動経済学に基づく価格設計
4. **Stripe Billing Documentation** — https://stripe.com/docs/billing — サブスクリプション実装ガイド
5. **"Predictably Irrational" — Dan Ariely** — 価格の心理学と行動経済学
6. **ProfitWell "SaaS Pricing Strategy" (2024)** — https://www.profitwell.com — 価格最適化データ
7. **Kyle Poyar "Growth Unhinged" Newsletter** — PLG・従量課金の最新トレンド

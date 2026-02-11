# 価格モデル — 従量制、サブスク、フリーミアム

> AI SaaSおよびAIサービスの価格設計を体系的に解説し、従量課金、サブスクリプション、フリーミアムの各モデルの設計原則、実装方法、最適化戦略を提供する。

---

## この章で学ぶこと

1. **AI特有の価格設計の原則** — コスト構造（API費用、GPU費用）を考慮した価格設定フレームワーク
2. **3大価格モデルの設計と実装** — 従量制、サブスクリプション、フリーミアムの詳細設計
3. **価格最適化と実験手法** — A/Bテスト、価格感度分析、LTV最大化の実践

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

---

## 5. 価格実験と最適化

### 5.1 価格感度分析

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

### 5.2 A/Bテスト実装

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

---

## 6. アンチパターン

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

---

## 7. FAQ

### Q1: 従量制とサブスクどちらが良い？

**A:** ユースケースで判断する。(1) 利用量が予測しにくい → 従量制（開発者向けAPI等）、(2) 利用量が安定 → サブスク（ビジネスユーザー向け等）、(3) 最強は「サブスク + 従量制のハイブリッド」。基本料金で一定量含み、超過分は従量課金。Slack、Twilio、AWS等の成功例がこのモデル。

### Q2: 値上げのタイミングと方法は？

**A:** 3つの原則。(1) タイミング — PMF達成後、機能追加時、年1回の定期見直し、(2) 方法 — 既存ユーザーは旧価格据え置き（グランドファザリング）+ 新規は新価格、(3) 幅 — 一度に20%以上の値上げは避け、10-15%ずつ段階的に。通知は最低30日前、値上げの理由（新機能追加等）を明確に伝える。

### Q3: AI SaaSの適正粗利率は？

**A:** 業界目安は70-80%。ただしAI SaaSはAPIコストが高いため、立ち上げ期は60%でも許容範囲。改善方法: (1) キャッシュ導入で同一リクエストのAPI呼び出し削減（30-50%削減可能）、(2) 軽量モデルの活用（簡単なタスクはGPT-3.5で十分）、(3) バッチ処理でAPI効率化。年々APIコストが下がるトレンドもあり、粗利は自然に改善する傾向。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| コスト構造 | AI APIが売上の20-40%、粗利70%以上が目標 |
| 従量制 | 開発者向け・利用量変動大に最適 |
| サブスク | ビジネスユーザー向け・予測可能な収益 |
| フリーミアム | 転換率3-5%が目標、無料は「もう少し」の設計 |
| ハイブリッド | サブスク基本料 + 従量超過課金が最強 |
| 最適化 | A/Bテスト + Van Westendorp + LTV分析 |

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

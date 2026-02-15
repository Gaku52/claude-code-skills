# モデル比較 — ベンチマーク・価格・ユースケース別選定ガイド

> LLM の選定は単一指標では決まらない。ベンチマーク性能、コスト、レイテンシ、コンテキスト長、マルチモーダル対応、プライバシー要件を総合的に評価し、ユースケースに最適なモデルを選ぶ必要がある。

## この章で学ぶこと

1. **主要ベンチマークの読み方** — MMLU、HumanEval、MT-Bench 等の意味と限界
2. **コスト・性能のトレードオフ分析** — 価格対品質の最適点を見つける方法
3. **ユースケース別モデル選定フレームワーク** — 要件からモデルを逆引きする実践手法
4. **推論モデルの比較と選定** — o1/o3、DeepSeek-R1、Claude の Extended Thinking
5. **マルチモデルルーティング** — 複数モデルを組み合わせたコスト最適化
6. **評価パイプラインの構築** — 自社タスクでの定量的モデル評価

---

## 1. 主要ベンチマーク解説

### 1.1 ベンチマーク一覧

```
┌──────────────────────────────────────────────────────────┐
│              LLM 主要ベンチマーク体系                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  知識・推論                                               │
│  ├── MMLU (57科目の多肢選択)                              │
│  ├── MMLU-Pro (より難易度の高い改良版)                     │
│  ├── GPQA (大学院レベル科学問題)                           │
│  ├── ARC-Challenge (科学的推論)                           │
│  ├── BIG-Bench Hard (難問推論集)                          │
│  └── SimpleQA (事実性評価)                                │
│                                                          │
│  コード                                                   │
│  ├── HumanEval (Python関数生成, 164問)                    │
│  ├── HumanEval+ (拡張テストケース版)                      │
│  ├── MBPP (基本プログラミング, 974問)                      │
│  ├── SWE-bench (実リポジトリのバグ修正)                    │
│  ├── SWE-bench Verified (人手検証済みサブセット)            │
│  └── LiveCodeBench (競技プログラミング)                    │
│                                                          │
│  数学                                                     │
│  ├── GSM8K (小学校算数)                                   │
│  ├── MATH (高校〜大学数学)                                │
│  ├── AIME (数学オリンピック級)                             │
│  └── FrontierMath (研究レベル数学)                         │
│                                                          │
│  対話・指示追従                                            │
│  ├── MT-Bench (多ターン対話, GPT-4評価)                   │
│  ├── AlpacaEval 2.0 (指示追従, 長さ補正付き)              │
│  ├── LMSYS Chatbot Arena (人間ブラインド評価)             │
│  └── WildBench (実ユーザークエリ評価)                     │
│                                                          │
│  多言語                                                   │
│  ├── MGSM (多言語数学)                                    │
│  ├── JMMLU / JGLUE (日本語特化)                           │
│  └── MMMLU (大規模多言語MMLU)                             │
│                                                          │
│  安全性                                                   │
│  ├── TruthfulQA (誤情報耐性)                              │
│  ├── HarmBench (有害性評価)                               │
│  └── WMDP (危険知識評価)                                  │
└──────────────────────────────────────────────────────────┘
```

### 1.2 ベンチマークの詳細解説

#### MMLU（Massive Multitask Language Understanding）

```python
# MMLU の評価例: 57科目の多肢選択問題
mmlu_example = {
    "subject": "abstract_algebra",
    "question": "Find the degree of the extension Q(sqrt(2), sqrt(3)) over Q.",
    "choices": ["2", "4", "6", "8"],
    "answer": "B",  # 4
}

# MMLU の科目カテゴリ
mmlu_categories = {
    "STEM": [
        "abstract_algebra", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science",
        "college_mathematics", "college_physics",
        "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics",
        "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history",
        "high_school_us_history", "high_school_world_history",
        "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes",
        "moral_scenarios", "philosophy", "prehistory",
        "professional_law", "world_religions",
    ],
    "Social_Sciences": [
        "econometrics", "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology",
        "public_relations", "security_studies",
        "sociology", "us_foreign_policy",
    ],
    "Other": [
        "anatomy", "business_ethics", "clinical_knowledge",
        "college_medicine", "global_facts",
        "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous",
        "nutrition", "professional_accounting",
        "professional_medicine", "virology",
    ],
}

# MMLU-Pro の改良点
mmlu_pro_improvements = {
    "選択肢数": "4択 → 10択（推測困難に）",
    "難易度": "より専門的な問題を追加",
    "汚染対策": "新規作成問題を含む",
    "推論重視": "単純暗記では解けない問題設計",
}
```

#### SWE-bench（Software Engineering Benchmark）

```python
# SWE-bench の評価構造
swe_bench_structure = {
    "概要": "実際のGitHubリポジトリのIssueを解決するタスク",
    "評価方法": {
        "入力": "リポジトリ + Issue 記述",
        "出力": "パッチ (diff)",
        "判定": "テストスイートの通過率",
    },
    "対象リポジトリ": [
        "django/django",
        "scikit-learn/scikit-learn",
        "matplotlib/matplotlib",
        "sympy/sympy",
        "sphinx-doc/sphinx",
        "astropy/astropy",
        "pytest-dev/pytest",
        "pallets/flask",
        "psf/requests",
    ],
    "バリエーション": {
        "SWE-bench Full": "2,294問（全量）",
        "SWE-bench Lite": "300問（難易度均一サブセット）",
        "SWE-bench Verified": "500問（人手検証済み）",
    },
}

# SWE-bench のスコア推移（エージェント型アプローチ）
swe_bench_scores = {
    "モデル/システム": {
        "Claude 3.5 Sonnet (SWE-Agent)": 49.0,
        "GPT-4o (SWE-Agent)": 33.2,
        "DeepSeek-V3 (SWE-Agent)": 42.0,
        "Claude 3.5 Sonnet (Aider)": 45.3,
        "OpenHands (Claude)": 53.0,
        "Cognition Devin": 43.8,
    },
}
```

#### Chatbot Arena (LMSYS)

```
┌──────────────────────────────────────────────────────────┐
│        Chatbot Arena 評価メカニズム                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ユーザー → プロンプト入力                                 │
│     │                                                    │
│     ├─── モデルA（匿名）──→ 回答A                         │
│     │                                                    │
│     ├─── モデルB（匿名）──→ 回答B                         │
│     │                                                    │
│     └─── ユーザーが優劣を判定                              │
│           ├── A wins / B wins / Tie / Both bad            │
│           └── 判定後にモデル名を公開                       │
│                                                          │
│  Elo レーティング計算:                                     │
│  ├── チェスと同じ Elo 方式                                │
│  ├── 勝敗に基づきレーティングを更新                        │
│  ├── 強いモデルに勝つとより多くのポイント獲得               │
│  └── Bootstrap (ブートストラップ) で信頼区間算出            │
│                                                          │
│  カテゴリ別ランキング:                                     │
│  ├── Overall（総合）                                      │
│  ├── Hard Prompts（難問）                                 │
│  ├── Coding（コーディング）                               │
│  ├── Math（数学）                                         │
│  ├── Instruction Following（指示追従）                    │
│  ├── Longer Query（長文質問）                             │
│  ├── Multi-Turn（多ターン）                               │
│  └── Style Control（スタイル補正後）                      │
│                                                          │
│  信頼性:                                                  │
│  ├── 100万+投票数（最大規模の人間評価）                    │
│  ├── 匿名性によるバイアス排除                             │
│  └── カテゴリ別で用途に即した比較が可能                    │
└──────────────────────────────────────────────────────────┘
```

### 1.3 ベンチマークスコア比較 (2025年初頭時点)

| モデル | MMLU | HumanEval | MATH | MT-Bench | Arena Elo |
|--------|------|-----------|------|----------|-----------|
| GPT-4o | 88.7 | 90.2 | 76.6 | 9.3 | ~1280 |
| Claude 3.5 Sonnet | 88.7 | 92.0 | 78.3 | 9.2 | ~1270 |
| Gemini 1.5 Pro | 85.9 | 84.1 | 67.7 | 9.0 | ~1260 |
| Llama 3.1 405B | 87.3 | 89.0 | 73.8 | 8.8 | ~1200 |
| Qwen 2.5 72B | 86.1 | 86.4 | 71.9 | 8.7 | ~1190 |
| DeepSeek-V3 | 87.1 | 82.6 | 90.2 | 8.9 | ~1250 |
| Mixtral 8x22B | 77.8 | 75.0 | 49.8 | 8.1 | ~1140 |
| GPT-4o mini | 82.0 | 87.2 | 70.2 | 8.6 | ~1200 |
| Gemini 1.5 Flash | 78.9 | 74.3 | 54.9 | 8.2 | ~1170 |

*注: スコアは公開情報に基づく概算値。評価条件により変動する。*

### 1.4 推論モデルのベンチマーク比較

| モデル | AIME 2024 | GPQA Diamond | SWE-bench Verified | LiveCodeBench | FrontierMath |
|--------|-----------|-------------|-------------------|--------------|-------------|
| o1 | 83.3 | 78.0 | 48.9 | 63.4 | 25.2 |
| o3-mini (high) | 87.3 | 79.7 | 49.3 | 67.1 | 28.9 |
| DeepSeek-R1 | 79.8 | 71.5 | 49.2 | 65.9 | 23.5 |
| Claude 3.5 (ET) | 75.0 | 68.0 | 50.8 | 58.2 | 18.7 |
| Gemini 2.0 Flash Thinking | 73.3 | 65.4 | 42.1 | 55.3 | 15.8 |

*ET = Extended Thinking*

### 1.5 ベンチマークの限界と注意点

```python
# ベンチマーク評価の落とし穴を検出するチェッカー
class BenchmarkReliabilityChecker:
    """ベンチマークスコアの信頼性を評価する"""

    def __init__(self):
        self.known_issues = {
            "data_contamination": {
                "description": "訓練データにベンチマーク問題が混入",
                "affected": ["MMLU", "GSM8K", "HumanEval"],
                "severity": "高",
                "mitigation": "時系列でのスコア変動を確認、新規ベンチマーク優先",
            },
            "prompt_sensitivity": {
                "description": "プロンプト形式でスコアが大きく変動",
                "affected": ["MMLU", "ARC", "HellaSwag"],
                "severity": "中",
                "mitigation": "複数プロンプト形式での平均を取る",
            },
            "saturation": {
                "description": "上位モデルが天井に近づき差別化困難",
                "affected": ["GSM8K", "HellaSwag", "MMLU"],
                "severity": "中",
                "mitigation": "より難しいベンチマーク (GPQA, AIME) を参照",
            },
            "gaming": {
                "description": "ベンチマーク最適化による実力との乖離",
                "affected": ["全般"],
                "severity": "高",
                "mitigation": "Arena等の人間評価と照合",
            },
            "domain_mismatch": {
                "description": "ベンチマークが対象ユースケースと乖離",
                "affected": ["全般"],
                "severity": "高",
                "mitigation": "自社タスクでの独自評価を実施",
            },
        }

    def assess_reliability(self, benchmark: str) -> dict:
        """特定ベンチマークの信頼性を評価"""
        issues = []
        for issue_name, details in self.known_issues.items():
            if benchmark in details["affected"] or "全般" in details["affected"]:
                issues.append({
                    "issue": issue_name,
                    "description": details["description"],
                    "severity": details["severity"],
                    "mitigation": details["mitigation"],
                })

        reliability_score = max(0, 100 - len(issues) * 20)
        return {
            "benchmark": benchmark,
            "reliability_score": reliability_score,
            "issues": issues,
            "recommendation": self._get_recommendation(reliability_score),
        }

    def _get_recommendation(self, score: int) -> str:
        if score >= 80:
            return "参考指標として信頼度高い"
        elif score >= 60:
            return "他の指標と組み合わせて判断"
        elif score >= 40:
            return "足切りのみに使用、最終判断には不適"
        else:
            return "単独での使用は非推奨"

# 使用例
checker = BenchmarkReliabilityChecker()
for benchmark in ["MMLU", "SWE-bench Verified", "LMSYS Arena", "GPQA"]:
    result = checker.assess_reliability(benchmark)
    print(f"{benchmark}: 信頼度 {result['reliability_score']}% - {result['recommendation']}")
```

---

## 2. コスト比較

### 2.1 API 料金表 (2025年初頭)

```python
# モデル別コスト計算ツール（拡張版）
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelPricing:
    """モデルの価格情報"""
    name: str
    input_price: float    # $/1M tokens
    output_price: float   # $/1M tokens
    cached_input_price: Optional[float] = None  # キャッシュヒット時
    batch_discount: float = 1.0  # バッチAPI割引率
    context_window: int = 128_000
    max_output: int = 4_096

# 2025年初頭の価格情報
PRICING_TABLE = {
    "gpt-4o": ModelPricing(
        "GPT-4o", 2.50, 10.00,
        cached_input_price=1.25, batch_discount=0.5,
        context_window=128_000, max_output=16_384,
    ),
    "gpt-4o-mini": ModelPricing(
        "GPT-4o mini", 0.15, 0.60,
        cached_input_price=0.075, batch_discount=0.5,
        context_window=128_000, max_output=16_384,
    ),
    "o1": ModelPricing(
        "o1", 15.00, 60.00,
        cached_input_price=7.50, batch_discount=0.5,
        context_window=200_000, max_output=100_000,
    ),
    "o3-mini": ModelPricing(
        "o3-mini", 1.10, 4.40,
        cached_input_price=0.55, batch_discount=0.5,
        context_window=200_000, max_output=100_000,
    ),
    "claude-3.5-sonnet": ModelPricing(
        "Claude 3.5 Sonnet", 3.00, 15.00,
        cached_input_price=0.30, batch_discount=0.5,
        context_window=200_000, max_output=8_192,
    ),
    "claude-3.5-haiku": ModelPricing(
        "Claude 3.5 Haiku", 0.80, 4.00,
        cached_input_price=0.08, batch_discount=0.5,
        context_window=200_000, max_output=8_192,
    ),
    "gemini-1.5-pro": ModelPricing(
        "Gemini 1.5 Pro", 1.25, 5.00,
        context_window=2_000_000, max_output=8_192,
    ),
    "gemini-1.5-flash": ModelPricing(
        "Gemini 1.5 Flash", 0.075, 0.30,
        context_window=1_000_000, max_output=8_192,
    ),
    "gemini-2.0-flash": ModelPricing(
        "Gemini 2.0 Flash", 0.10, 0.40,
        context_window=1_000_000, max_output=8_192,
    ),
    "deepseek-v3": ModelPricing(
        "DeepSeek-V3", 0.27, 1.10,
        cached_input_price=0.07,
        context_window=128_000, max_output=8_192,
    ),
    "deepseek-r1": ModelPricing(
        "DeepSeek-R1", 0.55, 2.19,
        cached_input_price=0.14,
        context_window=128_000, max_output=8_192,
    ),
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_hit_rate: float = 0.0,
    use_batch: bool = False,
) -> dict:
    """詳細なコスト計算"""
    if model not in PRICING_TABLE:
        return {"error": f"Unknown model: {model}"}

    pricing = PRICING_TABLE[model]

    # キャッシュヒット分を計算
    cached_tokens = int(input_tokens * cache_hit_rate)
    uncached_tokens = input_tokens - cached_tokens

    if pricing.cached_input_price and cache_hit_rate > 0:
        input_cost = (
            (uncached_tokens / 1_000_000) * pricing.input_price +
            (cached_tokens / 1_000_000) * pricing.cached_input_price
        )
    else:
        input_cost = (input_tokens / 1_000_000) * pricing.input_price

    output_cost = (output_tokens / 1_000_000) * pricing.output_price

    # バッチAPI割引
    if use_batch:
        input_cost *= pricing.batch_discount
        output_cost *= pricing.batch_discount

    total = input_cost + output_cost

    return {
        "model": pricing.name,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total": total,
        "cache_savings": (
            (input_tokens / 1_000_000) * pricing.input_price - input_cost
            if cache_hit_rate > 0 else 0
        ),
    }


# 使用例: 月間コスト比較
print("=== 月間コスト比較 (100万req, 入力500tok, 出力200tok) ===")
for model_id in PRICING_TABLE:
    result = calculate_cost(model_id, 500_000_000, 200_000_000)
    if "error" not in result:
        print(f"{result['model']:25s}: ${result['total']:>10,.2f}/月")

print("\n=== キャッシュ利用時 (50%ヒット率) ===")
for model_id in ["claude-3.5-sonnet", "gpt-4o", "deepseek-v3"]:
    normal = calculate_cost(model_id, 500_000_000, 200_000_000)
    cached = calculate_cost(model_id, 500_000_000, 200_000_000, cache_hit_rate=0.5)
    savings = normal["total"] - cached["total"]
    print(f"{cached['model']:25s}: ${cached['total']:>10,.2f}/月 (節約: ${savings:,.2f})")

print("\n=== バッチAPI利用時 ===")
for model_id in ["gpt-4o", "claude-3.5-sonnet", "o1"]:
    normal = calculate_cost(model_id, 500_000_000, 200_000_000)
    batch = calculate_cost(model_id, 500_000_000, 200_000_000, use_batch=True)
    savings = normal["total"] - batch["total"]
    print(f"{batch['model']:25s}: ${batch['total']:>10,.2f}/月 (節約: ${savings:,.2f})")
```

### 2.2 コスト比較表

| モデル | 入力 ($/1M) | 出力 ($/1M) | 月100万req概算 | コスパ評価 |
|--------|-----------|-----------|-------------|----------|
| Gemini 1.5 Flash | $0.075 | $0.30 | $97 | 最安クラス |
| Gemini 2.0 Flash | $0.10 | $0.40 | $130 | 最安クラス |
| GPT-4o mini | $0.15 | $0.60 | $195 | 極めて高い |
| DeepSeek-V3 | $0.27 | $1.10 | $355 | 高い |
| DeepSeek-R1 | $0.55 | $2.19 | $713 | 高い（推論特化） |
| Claude 3.5 Haiku | $0.80 | $4.00 | $1,200 | 高い |
| o3-mini | $1.10 | $4.40 | $1,430 | 高い（推論特化） |
| Gemini 1.5 Pro | $1.25 | $5.00 | $1,625 | 中 |
| GPT-4o | $2.50 | $10.00 | $3,250 | 中 |
| Claude 3.5 Sonnet | $3.00 | $15.00 | $4,500 | 中 |
| o1 | $15.00 | $60.00 | $19,500 | 低（高精度用） |

*月100万req = 平均入力500tok + 出力200tok で概算*

### 2.3 コスト最適化テクニック

```python
class CostOptimizer:
    """LLM API コスト最適化エンジン"""

    def __init__(self):
        self.strategies = []

    def analyze_and_recommend(
        self,
        current_model: str,
        monthly_requests: int,
        avg_input_tokens: int,
        avg_output_tokens: int,
        current_monthly_cost: float,
        quality_threshold: float = 0.9,  # 現行品質の何%を維持するか
    ) -> list[dict]:
        """コスト最適化の推奨施策を分析"""
        recommendations = []

        # 戦略1: プロンプトキャッシュの活用
        if current_model in ["claude-3.5-sonnet", "gpt-4o", "deepseek-v3"]:
            pricing = PRICING_TABLE.get(current_model.replace(".", "-").replace(" ", "-").lower())
            if pricing and pricing.cached_input_price:
                cache_savings_rate = 1 - (pricing.cached_input_price / pricing.input_price)
                potential_savings = current_monthly_cost * 0.3 * cache_savings_rate
                recommendations.append({
                    "strategy": "プロンプトキャッシュ",
                    "description": "システムプロンプトやFew-shot例をキャッシュ",
                    "potential_savings": f"${potential_savings:,.0f}/月",
                    "effort": "低",
                    "risk": "なし",
                })

        # 戦略2: バッチAPIの活用
        if monthly_requests > 100_000:
            batch_savings = current_monthly_cost * 0.5
            recommendations.append({
                "strategy": "バッチAPI",
                "description": "非リアルタイム処理をバッチに移行",
                "potential_savings": f"${batch_savings:,.0f}/月（対象分の50%削減）",
                "effort": "中",
                "risk": "レイテンシ増（24h以内処理）",
            })

        # 戦略3: モデルダウングレード（品質確認付き）
        downgrade_map = {
            "gpt-4o": "gpt-4o-mini",
            "claude-3.5-sonnet": "claude-3.5-haiku",
            "gemini-1.5-pro": "gemini-2.0-flash",
            "o1": "o3-mini",
        }
        if current_model in downgrade_map:
            smaller = downgrade_map[current_model]
            small_pricing = PRICING_TABLE.get(smaller)
            current_pricing = PRICING_TABLE.get(current_model)
            if small_pricing and current_pricing:
                cost_ratio = (
                    (small_pricing.input_price + small_pricing.output_price) /
                    (current_pricing.input_price + current_pricing.output_price)
                )
                savings = current_monthly_cost * (1 - cost_ratio)
                recommendations.append({
                    "strategy": f"モデルダウングレード ({current_pricing.name} → {small_pricing.name})",
                    "description": "簡易タスクに小型モデルを使用",
                    "potential_savings": f"${savings:,.0f}/月",
                    "effort": "中（品質評価必要）",
                    "risk": "品質低下の可能性",
                })

        # 戦略4: スマートルーティング
        if monthly_requests > 50_000:
            recommendations.append({
                "strategy": "スマートルーティング",
                "description": "タスク難易度に応じてモデルを自動選択",
                "potential_savings": f"${current_monthly_cost * 0.4:,.0f}/月",
                "effort": "高",
                "risk": "ルーティング精度に依存",
            })

        # 戦略5: プロンプト最適化
        if avg_input_tokens > 500:
            token_reduction = 0.3  # 30% のトークン削減を見込む
            savings = current_monthly_cost * token_reduction * 0.6  # 入力コスト比率
            recommendations.append({
                "strategy": "プロンプト最適化",
                "description": "冗長なプロンプトを圧縮、不要な指示を削除",
                "potential_savings": f"${savings:,.0f}/月",
                "effort": "低",
                "risk": "品質低下の可能性",
            })

        return sorted(recommendations, key=lambda x: x["effort"])

# 使用例
optimizer = CostOptimizer()
recs = optimizer.analyze_and_recommend(
    current_model="gpt-4o",
    monthly_requests=1_000_000,
    avg_input_tokens=800,
    avg_output_tokens=300,
    current_monthly_cost=5000,
)
for i, rec in enumerate(recs, 1):
    print(f"\n{i}. {rec['strategy']}")
    print(f"   {rec['description']}")
    print(f"   節約見込み: {rec['potential_savings']}")
    print(f"   実装難易度: {rec['effort']}")
```

### 2.4 自前デプロイ vs API コスト

```
┌──────────────────────────────────────────────────────────┐
│        自前デプロイ vs API サービス コスト分析              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  コスト                                                   │
│  ^                                                       │
│  │                                                       │
│  │  API                                                  │
│  │  ╱                                                    │
│  │ ╱        自前デプロイ                                   │
│  │╱         ┌──────────────────────                      │
│  ├─────────┘                                             │
│  │  初期投資                                              │
│  │  (GPU購入/レンタル)                                    │
│  │                                                       │
│  └──────────────────────────────────▶ リクエスト数        │
│          ↑                                               │
│     損益分岐点                                            │
│     (月間約50-100万req)                                   │
│                                                          │
│  判断基準:                                                │
│  - 月間 <10万req → API 一択                               │
│  - 月間 10-100万req → 要計算                              │
│  - 月間 >100万req → 自前デプロイ検討                      │
│  - データ機密性要件 → 自前デプロイ推奨                     │
│  - GPUクラウド利用 → 月$2-4/GPU-hour (A100)               │
│  - GPU購入 → 初期$10-15K/GPU、1-2年で回収                 │
└──────────────────────────────────────────────────────────┘
```

```python
# 自前デプロイのコスト試算
class SelfHostCostCalculator:
    """自前デプロイのTCO（Total Cost of Ownership）計算"""

    # GPU別のスペックと価格
    GPU_SPECS = {
        "A100-80GB": {
            "cloud_hourly": 3.50,  # $/hour (AWS p4d)
            "purchase_price": 15000,
            "vram_gb": 80,
            "fp16_tflops": 312,
        },
        "H100-80GB": {
            "cloud_hourly": 5.50,  # $/hour (AWS p5)
            "purchase_price": 30000,
            "vram_gb": 80,
            "fp16_tflops": 989,
        },
        "L40S-48GB": {
            "cloud_hourly": 1.80,
            "purchase_price": 8000,
            "vram_gb": 48,
            "fp16_tflops": 362,
        },
        "RTX4090-24GB": {
            "cloud_hourly": 0.80,
            "purchase_price": 2000,
            "vram_gb": 24,
            "fp16_tflops": 165,
        },
    }

    # モデル別の必要GPU数
    MODEL_REQUIREMENTS = {
        "Llama-3.1-70B (FP16)": {"vram_needed": 140, "min_gpu": "A100-80GB", "gpu_count": 2},
        "Llama-3.1-70B (INT8)": {"vram_needed": 70, "min_gpu": "A100-80GB", "gpu_count": 1},
        "Llama-3.1-70B (INT4)": {"vram_needed": 35, "min_gpu": "L40S-48GB", "gpu_count": 1},
        "Llama-3.1-8B (FP16)": {"vram_needed": 16, "min_gpu": "RTX4090-24GB", "gpu_count": 1},
        "Qwen-2.5-72B (INT4)": {"vram_needed": 40, "min_gpu": "L40S-48GB", "gpu_count": 1},
        "DeepSeek-V3 (FP8)": {"vram_needed": 640, "min_gpu": "H100-80GB", "gpu_count": 8},
    }

    def calculate_tco(
        self,
        model: str,
        monthly_requests: int,
        deployment_type: str = "cloud",  # "cloud" or "on-premise"
        months: int = 12,
    ) -> dict:
        """TCO計算"""
        if model not in self.MODEL_REQUIREMENTS:
            return {"error": f"Unknown model: {model}"}

        req = self.MODEL_REQUIREMENTS[model]
        gpu_spec = self.GPU_SPECS[req["min_gpu"]]
        gpu_count = req["gpu_count"]

        if deployment_type == "cloud":
            monthly_gpu_cost = gpu_spec["cloud_hourly"] * 24 * 30 * gpu_count
            initial_cost = 0
            monthly_ops = monthly_gpu_cost * 0.1  # 運用コスト10%
        else:
            initial_cost = gpu_spec["purchase_price"] * gpu_count
            monthly_gpu_cost = 0
            # 電気代 + 冷却 + ネットワーク
            power_cost = 0.5 * gpu_count * 24 * 30 * 0.15  # kW * hours * $/kWh
            monthly_ops = power_cost + 500  # 固定運用コスト

        total_cost = initial_cost + (monthly_gpu_cost + monthly_ops) * months
        cost_per_request = total_cost / (monthly_requests * months) if monthly_requests > 0 else 0

        return {
            "model": model,
            "deployment": deployment_type,
            "gpu": f"{gpu_count}x {req['min_gpu']}",
            "initial_cost": f"${initial_cost:,.0f}",
            "monthly_cost": f"${monthly_gpu_cost + monthly_ops:,.0f}",
            "total_cost_12m": f"${total_cost:,.0f}",
            "cost_per_request": f"${cost_per_request:.6f}",
        }

# 使用例
calc = SelfHostCostCalculator()
for model in ["Llama-3.1-70B (INT4)", "Llama-3.1-8B (FP16)", "Qwen-2.5-72B (INT4)"]:
    result = calc.calculate_tco(model, monthly_requests=500_000, deployment_type="cloud")
    print(f"{model}: 月額 {result['monthly_cost']} / req単価 {result['cost_per_request']}")
```

---

## 3. 機能比較

### 3.1 機能マトリクス（詳細版）

| 機能 | GPT-4o | Claude 3.5 | Gemini 1.5 | Llama 3.1 | Qwen 2.5 | DeepSeek-V3 |
|------|--------|-----------|-----------|----------|----------|------------|
| テキスト生成 | S | S | S | A | A | S |
| コード生成 | S | S | A | A | A | S |
| 画像入力 | S | S | S | N/A | A (VL) | N/A |
| 音声入力 | S | N/A | S | N/A | A (Audio) | N/A |
| 動画入力 | N/A | N/A | S | N/A | N/A | N/A |
| 画像生成 | S | N/A | S | N/A | N/A | N/A |
| Function Calling | S | S | S | A | A | A |
| JSON Mode | S | S | S | A | A | S |
| Structured Output | S | S | A | A | A | A |
| System Prompt | S | S | S | S | S | S |
| ストリーミング | S | S | S | S | S | S |
| ファインチューニング | A | N/A | A | S | S | S |
| プロンプトキャッシュ | S | S | N/A | N/A | N/A | S |
| バッチAPI | S | S | N/A | N/A | N/A | N/A |
| 推論モード | S (o1) | S (ET) | A (Thinking) | N/A | A (QwQ) | S (R1) |

*S=優秀, A=対応, N/A=未対応*

### 3.2 コンテキスト長比較

| モデル | 最大コンテキスト | 実用的な精度維持範囲 | Needle-in-Haystack |
|--------|----------------|-------------------|--------------------|
| Gemini 1.5 Pro | 2,000K | ~1,000K | 99.7% (1M) |
| Gemini 2.0 Flash | 1,000K | ~500K | 99.2% (500K) |
| Claude 3.5 Sonnet | 200K | ~150K | 99.5% (200K) |
| GPT-4o | 128K | ~64K | 98.8% (128K) |
| o1 | 200K | ~128K | 99.0% (128K) |
| Llama 3.1 | 128K | ~64K | 97.5% (128K) |
| Qwen 2.5 | 128K | ~32K | 96.8% (64K) |
| DeepSeek-V3 | 128K | ~64K | 98.2% (128K) |
| Mixtral 8x22B | 64K | ~32K | 95.3% (32K) |

### 3.3 レイテンシ比較

```python
# レイテンシ測定・比較ツール
import asyncio
import time
import statistics
from dataclasses import dataclass

@dataclass
class LatencyResult:
    model: str
    ttft_ms: float      # Time to First Token
    tps: float           # Tokens per Second
    total_ms: float      # Total Response Time
    output_tokens: int

class LatencyBenchmark:
    """モデル間レイテンシ比較ベンチマーク"""

    # 公開ベンチマーク (Artificial Analysis) からの概算値
    LATENCY_DATA = {
        "gpt-4o": {
            "ttft_ms": 450,
            "tps": 95,
            "p50_total_ms": 2200,
            "p99_total_ms": 5500,
        },
        "gpt-4o-mini": {
            "ttft_ms": 280,
            "tps": 140,
            "p50_total_ms": 1200,
            "p99_total_ms": 3000,
        },
        "claude-3.5-sonnet": {
            "ttft_ms": 500,
            "tps": 80,
            "p50_total_ms": 2800,
            "p99_total_ms": 6000,
        },
        "claude-3.5-haiku": {
            "ttft_ms": 320,
            "tps": 120,
            "p50_total_ms": 1500,
            "p99_total_ms": 3500,
        },
        "gemini-1.5-flash": {
            "ttft_ms": 200,
            "tps": 160,
            "p50_total_ms": 900,
            "p99_total_ms": 2200,
        },
        "gemini-2.0-flash": {
            "ttft_ms": 180,
            "tps": 180,
            "p50_total_ms": 800,
            "p99_total_ms": 2000,
        },
        "deepseek-v3": {
            "ttft_ms": 600,
            "tps": 60,
            "p50_total_ms": 3500,
            "p99_total_ms": 8000,
        },
        "o1": {
            "ttft_ms": 5000,   # 推論時間を含む
            "tps": 70,
            "p50_total_ms": 15000,
            "p99_total_ms": 45000,
        },
    }

    def compare(self, use_case: str) -> list[dict]:
        """ユースケース別のレイテンシ適合性を評価"""
        latency_requirements = {
            "リアルタイムチャット": {"max_ttft": 500, "min_tps": 80},
            "ストリーミングUI": {"max_ttft": 1000, "min_tps": 50},
            "バックエンド処理": {"max_ttft": 5000, "min_tps": 30},
            "バッチ処理": {"max_ttft": 60000, "min_tps": 10},
        }

        if use_case not in latency_requirements:
            return [{"error": f"Unknown use case: {use_case}"}]

        req = latency_requirements[use_case]
        results = []

        for model, data in self.LATENCY_DATA.items():
            fits = (
                data["ttft_ms"] <= req["max_ttft"] and
                data["tps"] >= req["min_tps"]
            )
            results.append({
                "model": model,
                "ttft_ms": data["ttft_ms"],
                "tps": data["tps"],
                "fits_requirement": fits,
                "verdict": "適合" if fits else "不適合",
            })

        return sorted(results, key=lambda x: x["ttft_ms"])

# 使用例
bench = LatencyBenchmark()
for use_case in ["リアルタイムチャット", "ストリーミングUI", "バッチ処理"]:
    print(f"\n=== {use_case} ===")
    results = bench.compare(use_case)
    for r in results:
        mark = "OK" if r["fits_requirement"] else "NG"
        print(f"  [{mark}] {r['model']:25s} TTFT:{r['ttft_ms']:>6}ms  TPS:{r['tps']:>4}")
```

---

## 4. 推論モデルの比較

### 4.1 推論モデルの分類

```
┌──────────────────────────────────────────────────────────┐
│          推論モデル (Reasoning Models) 分類                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Test-Time Compute（推論時計算量増加）                     │
│  ├── OpenAI o-series                                     │
│  │   ├── o1: 高精度推論（高価格）                         │
│  │   ├── o1-mini: 軽量推論（STEM特化）                    │
│  │   ├── o3: 最高性能推論                                 │
│  │   └── o3-mini: コスパ最適推論                          │
│  │                                                       │
│  ├── DeepSeek R-series                                   │
│  │   ├── R1: OSS推論モデル（MIT license）                 │
│  │   ├── R1-Lite: 軽量版                                 │
│  │   └── R1蒸留モデル:                                   │
│  │       ├── R1-Distill-Qwen-32B                         │
│  │       ├── R1-Distill-Llama-70B                        │
│  │       └── R1-Distill-Qwen-7B                          │
│  │                                                       │
│  ├── Claude Extended Thinking                            │
│  │   └── Claude 3.5 Sonnet (Extended Thinking有効化)      │
│  │                                                       │
│  └── Gemini Thinking                                     │
│      └── Gemini 2.0 Flash Thinking                       │
│                                                          │
│  特徴:                                                    │
│  ├── Chain-of-Thought を内部で自動実行                    │
│  ├── 推論ステップ数に応じてコスト・レイテンシ増             │
│  ├── 数学・コード・論理推論で大幅な精度向上                 │
│  └── 簡単なタスクにはオーバースペック（コスト浪費）         │
└──────────────────────────────────────────────────────────┘
```

### 4.2 推論モデルの使い分け

```python
# 推論モデル選定ガイド
reasoning_model_guide = {
    "数学問題（大学レベル以上）": {
        "推奨": "o3-mini (high) または DeepSeek-R1",
        "理由": "AIME/MATH でトップ性能",
        "コスト注意": "o1は高価、R1はコスパ良好",
    },
    "複雑なコーディング（アーキテクチャ設計）": {
        "推奨": "Claude 3.5 Sonnet (Extended Thinking)",
        "理由": "SWE-bench最高性能 + コード理解力",
        "コスト注意": "通常のSonnetよりトークン消費増",
    },
    "科学的分析（論文解読等）": {
        "推奨": "o1 または DeepSeek-R1",
        "理由": "GPQA Diamond で高スコア",
        "コスト注意": "長い推論が必要、バッチAPIも検討",
    },
    "多段階の論理推論": {
        "推奨": "o3-mini (high)",
        "理由": "推論力とコストのバランス最良",
        "コスト注意": "reasoning_effort パラメータで制御可",
    },
    "簡単なQ&A・要約": {
        "推奨": "推論モデル不要 → GPT-4o mini / Gemini Flash",
        "理由": "推論モデルはオーバースペック",
        "コスト注意": "10-100倍のコスト浪費になる",
    },
}

# 推論モデルの使い分け判定
def should_use_reasoning_model(task_description: str, complexity: int) -> dict:
    """タスクの複雑さに応じて推論モデルの必要性を判定

    Args:
        task_description: タスクの説明
        complexity: 1-10 の複雑さスケール
    """
    if complexity <= 3:
        return {
            "use_reasoning": False,
            "recommended": "GPT-4o mini / Gemini Flash",
            "reason": "単純タスクに推論モデルは不要",
        }
    elif complexity <= 6:
        return {
            "use_reasoning": False,
            "recommended": "GPT-4o / Claude 3.5 Sonnet",
            "reason": "標準モデルで十分な複雑さ",
        }
    elif complexity <= 8:
        return {
            "use_reasoning": True,
            "recommended": "o3-mini (medium) / DeepSeek-R1",
            "reason": "推論が有益な複雑さ",
        }
    else:
        return {
            "use_reasoning": True,
            "recommended": "o1 / o3-mini (high)",
            "reason": "最高精度の推論が必要",
        }
```

---

## 5. ユースケース別選定フレームワーク

### 5.1 選定フローチャート

```
┌─────────────────────────────────────────────────────────┐
│          LLM 選定フローチャート（拡張版）                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  START: 要件定義                                         │
│    │                                                    │
│    ├─ データをクラウドに送れない?                          │
│    │   YES → OSS モデル自前デプロイ                       │
│    │          ├─ 推論重視 → DeepSeek-R1蒸留モデル          │
│    │          ├─ 日本語重視 → Qwen 2.5                    │
│    │          ├─ コード重視 → DeepSeek-Coder/Qwen-Coder  │
│    │          └─ 汎用    → Llama 3.1                     │
│    │                                                    │
│    NO ↓                                                 │
│    ├─ 予算制約が厳しい?                                  │
│    │   YES → 低コストモデル                              │
│    │          ├─ Gemini 2.0 Flash (最安+高速)             │
│    │          ├─ Gemini 1.5 Flash                        │
│    │          ├─ GPT-4o mini                             │
│    │          └─ DeepSeek-V3                             │
│    │                                                    │
│    NO ↓                                                 │
│    ├─ 複雑な推論が必要? (数学/科学/コード設計)             │
│    │   YES → 推論モデル                                  │
│    │          ├─ 最高精度 → o1 / o3                       │
│    │          ├─ コスパ  → o3-mini / DeepSeek-R1         │
│    │          └─ コード  → Claude 3.5 (Extended Thinking) │
│    │                                                    │
│    NO ↓                                                 │
│    ├─ 超長文書処理が必要? (>128K tokens)                  │
│    │   YES → Gemini 1.5 Pro (2M tokens)                 │
│    │                                                    │
│    NO ↓                                                 │
│    ├─ マルチモーダル (画像/音声/動画)?                     │
│    │   YES ├─ 動画含む → Gemini 1.5 Pro                  │
│    │       ├─ 画像+音声 → GPT-4o                         │
│    │       └─ 画像のみ → Claude 3.5 / GPT-4o            │
│    │                                                    │
│    NO ↓                                                 │
│    └─ 最高精度テキスト処理                               │
│         ├─ コード → Claude 3.5 Sonnet                   │
│         ├─ 日本語 → GPT-4o / Qwen 2.5                   │
│         └─ 汎用  → GPT-4o / Claude 3.5                  │
└─────────────────────────────────────────────────────────┘
```

### 5.2 ユースケース別推奨（詳細版）

```python
# ユースケース別モデル推奨辞書（拡張版）
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelRecommendation:
    model: str
    reason: str
    monthly_cost_estimate: str  # 月間コスト概算
    latency: str
    quality_score: float  # 0-1

@dataclass
class UseCaseRecommendation:
    use_case: str
    primary: ModelRecommendation
    secondary: ModelRecommendation
    budget_option: Optional[ModelRecommendation] = None
    key_requirements: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)

recommendations = {
    "カスタマーサポートBot": UseCaseRecommendation(
        use_case="カスタマーサポートBot",
        primary=ModelRecommendation(
            "GPT-4o mini", "コスパ最高、指示追従力高い",
            "$195/月 (100万req)", "TTFT: 280ms", 0.85,
        ),
        secondary=ModelRecommendation(
            "Gemini 2.0 Flash", "さらに安く高速",
            "$130/月 (100万req)", "TTFT: 180ms", 0.82,
        ),
        budget_option=ModelRecommendation(
            "Claude 3.5 Haiku", "高品質回答が必要な場合",
            "$1,200/月 (100万req)", "TTFT: 320ms", 0.88,
        ),
        key_requirements=["低レイテンシ", "大量リクエスト", "安定した応答品質"],
        anti_patterns=["o1を使う（コスト100倍、レイテンシ10倍）"],
    ),
    "コードレビュー・生成": UseCaseRecommendation(
        use_case="コードレビュー・生成",
        primary=ModelRecommendation(
            "Claude 3.5 Sonnet", "SWE-bench最高、コード理解力最高",
            "$4,500/月 (100万req)", "TTFT: 500ms", 0.95,
        ),
        secondary=ModelRecommendation(
            "GPT-4o", "幅広い言語対応、安定品質",
            "$3,250/月 (100万req)", "TTFT: 450ms", 0.90,
        ),
        budget_option=ModelRecommendation(
            "DeepSeek-V3", "コード品質高くコスパ良好",
            "$355/月 (100万req)", "TTFT: 600ms", 0.87,
        ),
        key_requirements=["コード理解力", "長いコンテキスト", "正確なdiff出力"],
        anti_patterns=["Gemini Flashでコードレビュー（精度不足）"],
    ),
    "法律文書分析": UseCaseRecommendation(
        use_case="法律文書分析",
        primary=ModelRecommendation(
            "Gemini 1.5 Pro", "200万token対応、文書一括処理",
            "$1,625/月 (100万req)", "TTFT: 800ms", 0.88,
        ),
        secondary=ModelRecommendation(
            "Claude 3.5 Sonnet", "200K token + 高精度分析",
            "$4,500/月 (100万req)", "TTFT: 500ms", 0.92,
        ),
        key_requirements=["長大文書処理", "正確な引用", "法律用語理解"],
        anti_patterns=["128K制限モデルで複数文書を一括処理しようとする"],
    ),
    "社内機密データ処理": UseCaseRecommendation(
        use_case="社内機密データ処理",
        primary=ModelRecommendation(
            "Qwen 2.5 72B (自前)", "データがクラウドに出ない",
            "$2,500/月 (GPU)", "環境依存", 0.85,
        ),
        secondary=ModelRecommendation(
            "Llama 3.1 70B (自前)", "Meta製、幅広い言語対応",
            "$2,500/月 (GPU)", "環境依存", 0.83,
        ),
        budget_option=ModelRecommendation(
            "DeepSeek-R1蒸留 Qwen-32B (自前)", "推論力あり、軽量",
            "$1,200/月 (GPU)", "環境依存", 0.80,
        ),
        key_requirements=["データプライバシー", "オンプレミス運用", "監査対応"],
        anti_patterns=["機密データをクラウドAPIに送信"],
    ),
    "数学・科学的推論": UseCaseRecommendation(
        use_case="数学・科学的推論",
        primary=ModelRecommendation(
            "o3-mini (high)", "推論力最高クラス、コスパ良好",
            "$1,430/月 (100万req)", "TTFT: 3000ms", 0.95,
        ),
        secondary=ModelRecommendation(
            "DeepSeek-R1", "OSS推論モデル、MIT license",
            "$713/月 (100万req)", "TTFT: 2000ms", 0.92,
        ),
        key_requirements=["段階的推論", "数式理解", "論理的一貫性"],
        anti_patterns=["Flash/miniモデルで大学レベル数学"],
    ),
    "リアルタイム翻訳": UseCaseRecommendation(
        use_case="リアルタイム翻訳",
        primary=ModelRecommendation(
            "Gemini 2.0 Flash", "最低レイテンシ、多言語対応",
            "$130/月 (100万req)", "TTFT: 180ms", 0.83,
        ),
        secondary=ModelRecommendation(
            "GPT-4o mini", "低レイテンシ、高品質翻訳",
            "$195/月 (100万req)", "TTFT: 280ms", 0.85,
        ),
        key_requirements=["低レイテンシ", "多言語対応", "ストリーミング"],
        anti_patterns=["推論モデルで翻訳（遅延大）"],
    ),
    "データ抽出・構造化": UseCaseRecommendation(
        use_case="データ抽出・構造化",
        primary=ModelRecommendation(
            "GPT-4o", "Structured Output対応、安定JSON出力",
            "$3,250/月 (100万req)", "TTFT: 450ms", 0.92,
        ),
        secondary=ModelRecommendation(
            "Claude 3.5 Sonnet", "高精度抽出、長文対応",
            "$4,500/月 (100万req)", "TTFT: 500ms", 0.90,
        ),
        budget_option=ModelRecommendation(
            "GPT-4o mini", "十分な抽出精度、低コスト",
            "$195/月 (100万req)", "TTFT: 280ms", 0.82,
        ),
        key_requirements=["JSON出力安定性", "スキーマ準拠", "エラーハンドリング"],
        anti_patterns=["非構造化出力モードでJSON生成を期待"],
    ),
    "クリエイティブライティング": UseCaseRecommendation(
        use_case="クリエイティブライティング",
        primary=ModelRecommendation(
            "Claude 3.5 Sonnet", "自然な文体、創造性高い",
            "$4,500/月 (100万req)", "TTFT: 500ms", 0.93,
        ),
        secondary=ModelRecommendation(
            "GPT-4o", "多様なスタイル対応",
            "$3,250/月 (100万req)", "TTFT: 450ms", 0.90,
        ),
        key_requirements=["文体の多様性", "一貫した物語構造", "感情表現"],
        anti_patterns=["推論モデルで創作（過度に論理的になる）"],
    ),
}

# 推奨結果の表示
def print_recommendation(use_case: str):
    if use_case not in recommendations:
        print(f"Unknown use case: {use_case}")
        return
    rec = recommendations[use_case]
    print(f"\n{'='*60}")
    print(f"ユースケース: {rec.use_case}")
    print(f"{'='*60}")
    print(f"\n【第1推奨】{rec.primary.model}")
    print(f"  理由: {rec.primary.reason}")
    print(f"  コスト: {rec.primary.monthly_cost_estimate}")
    print(f"  レイテンシ: {rec.primary.latency}")
    print(f"\n【第2推奨】{rec.secondary.model}")
    print(f"  理由: {rec.secondary.reason}")
    if rec.budget_option:
        print(f"\n【低予算】{rec.budget_option.model}")
        print(f"  理由: {rec.budget_option.reason}")
    print(f"\n必要要件: {', '.join(rec.key_requirements)}")
    print(f"アンチパターン: {', '.join(rec.anti_patterns)}")
```

---

## 6. モデル選定の実践コード

### 6.1 A/B テスト比較ツール

```python
import asyncio
import time
import json
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from dataclasses import dataclass

@dataclass
class ComparisonResult:
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    latency: float
    ttft: float
    cost: float

async def compare_models(
    prompt: str,
    models: list[str] = None,
    max_tokens: int = 1024,
) -> dict[str, ComparisonResult]:
    """複数モデルの出力を並行比較"""

    if models is None:
        models = ["gpt-4o", "claude-3.5-sonnet"]

    async def call_openai(model: str):
        client = AsyncOpenAI()
        start = time.time()
        ttft = None
        full_text = ""

        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        usage = None
        async for chunk in stream:
            if ttft is None and chunk.choices and chunk.choices[0].delta.content:
                ttft = time.time() - start
            if chunk.choices and chunk.choices[0].delta.content:
                full_text += chunk.choices[0].delta.content
            if chunk.usage:
                usage = chunk.usage

        latency = time.time() - start
        pricing = PRICING_TABLE.get(model, PRICING_TABLE["gpt-4o"])
        cost = (
            (usage.prompt_tokens / 1_000_000) * pricing.input_price +
            (usage.completion_tokens / 1_000_000) * pricing.output_price
        ) if usage else 0

        return ComparisonResult(
            model=model,
            text=full_text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency=latency,
            ttft=ttft or latency,
            cost=cost,
        )

    async def call_anthropic(model: str):
        client = AsyncAnthropic()
        start = time.time()
        ttft = None
        full_text = ""

        model_id = "claude-3-5-sonnet-20241022" if "sonnet" in model else "claude-3-5-haiku-20241022"

        async with client.messages.stream(
            model=model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                if ttft is None:
                    ttft = time.time() - start
                full_text += text

        message = await stream.get_final_message()
        latency = time.time() - start
        pricing = PRICING_TABLE.get(model, PRICING_TABLE["claude-3.5-sonnet"])
        cost = (
            (message.usage.input_tokens / 1_000_000) * pricing.input_price +
            (message.usage.output_tokens / 1_000_000) * pricing.output_price
        )

        return ComparisonResult(
            model=model,
            text=full_text,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency=latency,
            ttft=ttft or latency,
            cost=cost,
        )

    tasks = []
    for model in models:
        if "claude" in model:
            tasks.append(call_anthropic(model))
        else:
            tasks.append(call_openai(model))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    output = {}
    for model, result in zip(models, results):
        if isinstance(result, Exception):
            print(f"Error with {model}: {result}")
        else:
            output[model] = result

    return output


# A/Bテスト結果の表示
async def run_ab_test(prompt: str, models: list[str], num_trials: int = 5):
    """複数回の A/B テストを実行して統計を取る"""
    all_results = {model: [] for model in models}

    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}...")
        results = await compare_models(prompt, models)
        for model, result in results.items():
            all_results[model].append(result)

    # 統計出力
    print(f"\n{'='*70}")
    print(f"A/B テスト結果 ({num_trials} trials)")
    print(f"{'='*70}")
    print(f"{'モデル':25s} {'Avg Latency':>12s} {'Avg TTFT':>10s} {'Avg Cost':>10s}")
    print(f"{'-'*70}")

    for model in models:
        results = all_results[model]
        if not results:
            continue
        avg_latency = sum(r.latency for r in results) / len(results)
        avg_ttft = sum(r.ttft for r in results) / len(results)
        avg_cost = sum(r.cost for r in results) / len(results)
        print(f"{model:25s} {avg_latency:>10.2f}s {avg_ttft:>8.2f}s ${avg_cost:>8.6f}")

# 使用例
# asyncio.run(run_ab_test(
#     "Pythonのジェネレータを500字以内で解説してください",
#     ["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet"],
#     num_trials=5,
# ))
```

### 6.2 マルチモデルルーティング

```python
import re
from enum import Enum

class TaskDifficulty(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    REASONING = "reasoning"

class ModelRouter:
    """タスク難易度に応じて最適なモデルにルーティング"""

    def __init__(self, config: dict = None):
        self.config = config or {
            TaskDifficulty.SIMPLE: "gpt-4o-mini",
            TaskDifficulty.MODERATE: "gpt-4o",
            TaskDifficulty.COMPLEX: "claude-3.5-sonnet",
            TaskDifficulty.REASONING: "o3-mini",
        }
        self.routing_history = []

    def classify_task(self, prompt: str) -> TaskDifficulty:
        """プロンプトからタスク難易度を分類"""

        # キーワードベースの簡易分類（実運用ではLLM分類器を使用）
        reasoning_keywords = [
            "証明", "数学", "定理", "なぜ", "論理的に",
            "ステップバイステップ", "分析して", "比較検討",
            "最適化", "アルゴリズム",
        ]
        complex_keywords = [
            "コードレビュー", "リファクタリング", "アーキテクチャ",
            "設計", "実装して", "デバッグ", "テストケース",
            "長文", "詳細に", "包括的に",
        ]
        moderate_keywords = [
            "要約", "翻訳", "説明", "リスト", "分類",
            "書き換え", "修正して",
        ]

        prompt_lower = prompt.lower()

        # 推論タスクの判定
        reasoning_score = sum(1 for kw in reasoning_keywords if kw in prompt_lower)
        if reasoning_score >= 2:
            return TaskDifficulty.REASONING

        # 複雑タスクの判定
        complex_score = sum(1 for kw in complex_keywords if kw in prompt_lower)
        if complex_score >= 2 or len(prompt) > 2000:
            return TaskDifficulty.COMPLEX

        # 中程度タスクの判定
        moderate_score = sum(1 for kw in moderate_keywords if kw in prompt_lower)
        if moderate_score >= 1 or len(prompt) > 500:
            return TaskDifficulty.MODERATE

        return TaskDifficulty.SIMPLE

    def route(self, prompt: str) -> dict:
        """プロンプトを最適なモデルにルーティング"""
        difficulty = self.classify_task(prompt)
        model = self.config[difficulty]

        routing_info = {
            "prompt_length": len(prompt),
            "difficulty": difficulty.value,
            "selected_model": model,
            "estimated_cost_ratio": {
                TaskDifficulty.SIMPLE: 1.0,
                TaskDifficulty.MODERATE: 16.7,
                TaskDifficulty.COMPLEX: 20.0,
                TaskDifficulty.REASONING: 7.3,
            }[difficulty],
        }

        self.routing_history.append(routing_info)
        return routing_info

    def get_routing_stats(self) -> dict:
        """ルーティング統計を取得"""
        if not self.routing_history:
            return {"total": 0}

        total = len(self.routing_history)
        distribution = {}
        for entry in self.routing_history:
            d = entry["difficulty"]
            distribution[d] = distribution.get(d, 0) + 1

        # コスト削減率の推計
        # 全てcomplexモデルを使う場合と比較
        baseline_cost = total * 20.0
        actual_cost = sum(entry["estimated_cost_ratio"] for entry in self.routing_history)
        savings_rate = 1 - (actual_cost / baseline_cost)

        return {
            "total_requests": total,
            "distribution": distribution,
            "estimated_cost_savings": f"{savings_rate:.1%}",
        }


# 使用例
router = ModelRouter()

test_prompts = [
    "こんにちは",  # SIMPLE
    "この文章を英語に翻訳してください: ...",  # MODERATE
    "このPythonコードをリファクタリングして、テストケースも書いて",  # COMPLEX
    "この定理をステップバイステップで証明して、なぜ成立するか論理的に説明して",  # REASONING
    "天気を教えて",  # SIMPLE
    "この長文を要約してください",  # MODERATE
]

for prompt in test_prompts:
    result = router.route(prompt)
    print(f"[{result['difficulty']:10s}] → {result['selected_model']:25s} | {prompt[:40]}...")

stats = router.get_routing_stats()
print(f"\n=== ルーティング統計 ===")
print(f"総リクエスト: {stats['total_requests']}")
print(f"分布: {stats['distribution']}")
print(f"推定コスト削減率: {stats['estimated_cost_savings']}")
```

### 6.3 LLM-as-a-Judge による自動品質評価

```python
from anthropic import Anthropic

class LLMJudge:
    """LLM-as-a-Judge による回答品質の自動評価"""

    JUDGE_PROMPT = """あなたは公平なAI回答品質評価者です。
以下の質問に対する2つの回答を評価し、各基準でスコアを付けてください。

## 質問
{question}

## 回答A ({model_a})
{answer_a}

## 回答B ({model_b})
{answer_b}

## 評価基準（各10点満点）
1. 正確性: 事実の正確さ、誤情報の有無
2. 完全性: 質問への網羅的な回答
3. 明瞭性: 分かりやすさ、構成の論理性
4. 実用性: 実際に役立つ具体的な情報
5. 簡潔性: 冗長でなく必要十分な分量

## 出力形式（JSON）
{{
  "model_a_scores": {{"accuracy": X, "completeness": X, "clarity": X, "usefulness": X, "conciseness": X}},
  "model_b_scores": {{"accuracy": X, "completeness": X, "clarity": X, "usefulness": X, "conciseness": X}},
  "model_a_total": X,
  "model_b_total": X,
  "winner": "A" or "B" or "tie",
  "reasoning": "判定理由を1-2文で"
}}"""

    def __init__(self, judge_model: str = "claude-3-5-sonnet-20241022"):
        self.client = Anthropic()
        self.judge_model = judge_model
        self.results = []

    def evaluate(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        model_a: str,
        model_b: str,
    ) -> dict:
        """2つの回答を比較評価"""
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            model_a=model_a,
            answer_a=answer_a,
            model_b=model_b,
            answer_b=answer_b,
        )

        response = self.client.messages.create(
            model=self.judge_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        result = json.loads(response.content[0].text)
        self.results.append(result)
        return result

    def evaluate_batch(
        self,
        test_cases: list[dict],
    ) -> dict:
        """バッチ評価を実行"""
        wins = {"A": 0, "B": 0, "tie": 0}
        total_scores = {"A": 0, "B": 0}

        for case in test_cases:
            result = self.evaluate(
                question=case["question"],
                answer_a=case["answer_a"],
                answer_b=case["answer_b"],
                model_a=case["model_a"],
                model_b=case["model_b"],
            )
            wins[result["winner"]] += 1
            total_scores["A"] += result["model_a_total"]
            total_scores["B"] += result["model_b_total"]

        n = len(test_cases)
        return {
            "total_cases": n,
            "wins": wins,
            "win_rate_a": f"{wins['A']/n:.1%}",
            "win_rate_b": f"{wins['B']/n:.1%}",
            "avg_score_a": total_scores["A"] / n,
            "avg_score_b": total_scores["B"] / n,
        }

    def position_debiased_evaluate(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        model_a: str,
        model_b: str,
    ) -> dict:
        """位置バイアスを排除した評価（AとBの順序を入れ替えて2回評価）"""
        # 通常順序で評価
        result1 = self.evaluate(question, answer_a, answer_b, model_a, model_b)

        # 順序を入れ替えて評価
        result2 = self.evaluate(question, answer_b, answer_a, model_b, model_a)

        # 結果を統合（一致していれば信頼性高い）
        winner1 = result1["winner"]
        # result2では順序が逆なので、勝者も逆転して考える
        winner2_mapped = {"A": "B", "B": "A", "tie": "tie"}[result2["winner"]]

        if winner1 == winner2_mapped:
            consensus = winner1
            confidence = "高"
        else:
            consensus = "tie"
            confidence = "低（位置バイアスの影響あり）"

        return {
            "consensus_winner": consensus,
            "confidence": confidence,
            "round1": result1,
            "round2": result2,
        }
```

---

## 7. OSS モデルの比較と選定

### 7.1 主要 OSS モデル一覧

```
┌──────────────────────────────────────────────────────────┐
│          主要 OSS LLM モデル一覧                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Meta Llama 3.1 シリーズ                                  │
│  ├── 405B: 最高性能OSS、8xA100/H100必要                   │
│  ├── 70B: バランス型、2xA100で動作                        │
│  └── 8B: エッジ向け、1xRTX4090で動作                      │
│  License: Llama 3.1 Community License                    │
│                                                          │
│  Alibaba Qwen 2.5 シリーズ                                │
│  ├── 72B: CJK言語に強い、日本語性能良好                    │
│  ├── 32B: 中型モデル、コスパ良好                           │
│  ├── 14B / 7B / 3B / 1.5B / 0.5B                         │
│  └── 派生: Qwen-Coder, Qwen-VL, Qwen-Audio               │
│  License: Apache 2.0 (一部Qwen License)                   │
│                                                          │
│  DeepSeek シリーズ                                        │
│  ├── V3: MoE 671B (37B active)、高効率                    │
│  ├── R1: 推論特化、MIT License                            │
│  └── Coder V2: コード特化                                 │
│  License: MIT                                             │
│                                                          │
│  Mistral シリーズ                                         │
│  ├── Large 2 (123B): EU規制準拠                           │
│  ├── Mixtral 8x22B: MoE、高効率                           │
│  └── Mistral 7B / Mistral NeMo                           │
│  License: Apache 2.0                                      │
│                                                          │
│  Google Gemma シリーズ                                     │
│  ├── Gemma 2 27B: 軽量高性能                               │
│  └── Gemma 2 9B / 2B                                      │
│  License: Gemma Terms of Use                              │
│                                                          │
│  Microsoft Phi シリーズ                                    │
│  ├── Phi-3.5-MoE-instruct (42B, 6.6B active)              │
│  ├── Phi-3.5-mini (3.8B): 超軽量                          │
│  └── Phi-3.5-vision: マルチモーダル対応                    │
│  License: MIT                                             │
└──────────────────────────────────────────────────────────┘
```

### 7.2 OSS モデルの選定基準

```python
# OSS モデル選定マトリクス
oss_selection_matrix = {
    "日本語テキスト生成": {
        "推奨": ["Qwen-2.5-72B", "Llama-3.1-70B"],
        "理由": "Qwen は CJK 学習データが豊富",
        "GPU要件": "2x A100-80GB (FP16) or 1x A100 (INT8)",
    },
    "コード生成・補完": {
        "推奨": ["DeepSeek-Coder-V2", "Qwen-2.5-Coder-32B"],
        "理由": "コード特化学習、多言語対応",
        "GPU要件": "1-2x A100-80GB",
    },
    "推論タスク": {
        "推奨": ["DeepSeek-R1", "R1-Distill-Qwen-32B"],
        "理由": "推論CoTを内蔵、MIT License",
        "GPU要件": "R1: 8xH100 / 蒸留版: 1x A100",
    },
    "エッジデプロイ": {
        "推奨": ["Phi-3.5-mini", "Gemma-2-2B", "Qwen-2.5-3B"],
        "理由": "3B以下で実用的品質",
        "GPU要件": "RTX3060以上 / CPU推論可",
    },
    "マルチモーダル（画像）": {
        "推奨": ["Qwen-VL-72B", "Llava-1.6-34B", "Phi-3.5-vision"],
        "理由": "画像+テキスト統合理解",
        "GPU要件": "モデルサイズに応じて変動",
    },
    "RAG / 埋め込み": {
        "推奨": ["BGE-M3", "E5-Mistral-7B", "GTE-Qwen2"],
        "理由": "多言語埋め込み、高リコール",
        "GPU要件": "1x RTX4090以上",
    },
}
```

---

## 8. アンチパターン

### アンチパターン 1: ベンチマークスコアだけで選定

```
# NG: MMLU スコアが最も高いモデルを無条件に採用
"MMLUが88点だからこのモデルにしよう"
→ 実際のタスク (日本語要約) では MMLU との相関が低い
→ データ汚染でスコアが膨張している可能性

# OK: 実タスクでの評価を実施
1. 自社タスクの評価データセット (100問以上) を作成
2. 候補モデル 3-5 個で推論を実行
3. 人手評価 or LLM-as-a-Judge で品質比較
4. コスト・レイテンシも加味して総合判断
5. A/Bテストで本番環境での性能を検証
```

### アンチパターン 2: 単一モデルへのベンダーロック

```python
# NG: OpenAI 固有の API 仕様に依存しきったコード
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_schema", "json_schema": {...}},
    # OpenAI 固有機能に強く依存
)

# OK: 抽象レイヤーを挟んでモデル切り替え可能に
from litellm import completion  # マルチプロバイダー対応ライブラリ

response = completion(
    model="gpt-4o",  # 簡単に "claude-3.5-sonnet" 等に変更可能
    messages=[{"role": "user", "content": prompt}],
)

# さらに良い: 自前の抽象レイヤー
class LLMClient:
    """プロバイダー非依存のLLMクライアント"""

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self._init_client()

    def _init_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI()
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic()
        elif self.provider == "google":
            import google.generativeai as genai
            self.client = genai

    def generate(self, prompt: str, **kwargs) -> str:
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return resp.choices[0].message.content
        elif self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 1024),
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
```

### アンチパターン 3: レイテンシ無視の選定

```
# NG: リアルタイムチャットに高性能だが遅いモデル
ユーザー体験: 「...」 (10秒間の沈黙) → 離脱率増加

# OK: 用途に応じたレイテンシ要件の設定
- リアルタイムチャット → TTFT < 500ms → Flash/mini 系
- バッチ処理 → レイテンシ不問 → 最高精度モデル
- ストリーミング表示 → TTFT < 1s → 中堅モデルでもOK
- 推論タスク → レイテンシ許容 → o1/R1 系（ただしUX配慮）
```

### アンチパターン 4: 全タスクに同一モデルを使用

```python
# NG: 全てのタスクに GPT-4o を使用
# → 簡単なタスクにも高額モデルを使い、コスト10倍以上に

# OK: タスク複雑度に応じたモデル選択
class SmartLLMService:
    """タスク複雑度に応じてモデルを自動選択するサービス"""

    TIER_CONFIG = {
        "tier1_simple": {
            "model": "gpt-4o-mini",
            "max_tokens": 512,
            "examples": ["挨拶", "FAQ回答", "簡単な翻訳"],
        },
        "tier2_moderate": {
            "model": "gpt-4o",
            "max_tokens": 2048,
            "examples": ["要約", "文書分類", "データ抽出"],
        },
        "tier3_complex": {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 4096,
            "examples": ["コードレビュー", "長文分析", "レポート生成"],
        },
        "tier4_reasoning": {
            "model": "o3-mini",
            "max_tokens": 8192,
            "examples": ["数学証明", "複雑な分析", "アーキテクチャ設計"],
        },
    }

    # コスト比較: 全部tier3を使う vs スマートルーティング
    # 月100万req想定:
    #   全部tier3: $4,500/月
    #   スマートルーティング (60% tier1, 25% tier2, 12% tier3, 3% tier4):
    #     0.6*$195 + 0.25*$3,250 + 0.12*$4,500 + 0.03*$1,430
    #     = $117 + $812 + $540 + $43 = $1,512/月
    #   → 66% コスト削減
```

### アンチパターン 5: 推論モデルの乱用

```
# NG: 全タスクに o1 を使用
コスト: $19,500/月 (100万req)
レイテンシ: TTFT 5-30秒
→ 99% のタスクではオーバースペック

# OK: 推論モデルは本当に推論が必要な場合のみ
判定基準:
1. 数学的証明や複雑な論理推論が必要 → o3-mini
2. 段階的な分析が品質に直結する → DeepSeek-R1
3. コード設計で複数の選択肢を検討する必要がある → Claude ET
4. それ以外 → 通常モデルで十分
```

---

## 9. モデル評価パイプラインの構築

### 9.1 評価パイプライン全体像

```
┌──────────────────────────────────────────────────────────┐
│        モデル評価パイプライン                               │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. 評価データセット作成                                   │
│  ├── 自社タスクから代表的なケースを100-500問抽出            │
│  ├── カテゴリ分け（難易度、ドメイン、タスクタイプ）          │
│  ├── 正解ラベルの作成（人手 or 専門家）                     │
│  └── テスト/バリデーション分割                              │
│       │                                                   │
│  2. 候補モデルで推論実行                                   │
│  ├── 全候補モデル × 全テストケース                         │
│  ├── 同一プロンプトテンプレート使用                         │
│  ├── Temperature=0 で再現性確保                            │
│  └── レイテンシ・トークン数も記録                           │
│       │                                                   │
│  3. 品質評価                                               │
│  ├── 自動指標: BLEU, ROUGE, Exact Match                   │
│  ├── LLM-as-a-Judge（位置バイアス排除版）                  │
│  ├── 人手評価（重要ケースのサンプリング）                   │
│  └── ドメイン専門家レビュー                                │
│       │                                                   │
│  4. 総合スコアリング                                       │
│  ├── 品質スコア (40%)                                     │
│  ├── コスト (25%)                                         │
│  ├── レイテンシ (20%)                                     │
│  └── 運用容易性 (15%)                                     │
│       │                                                   │
│  5. 意思決定                                               │
│  ├── スコアカード作成                                     │
│  ├── ステークホルダーレビュー                              │
│  └── 段階的ロールアウト計画                               │
└──────────────────────────────────────────────────────────┘
```

### 9.2 評価パイプラインの実装

```python
import json
from dataclasses import dataclass, field
from typing import Callable, Optional

@dataclass
class EvalCase:
    """評価ケース"""
    id: str
    prompt: str
    expected: str
    category: str
    difficulty: str  # easy, medium, hard
    metadata: dict = field(default_factory=dict)

@dataclass
class EvalResult:
    """評価結果"""
    case_id: str
    model: str
    output: str
    latency: float
    input_tokens: int
    output_tokens: int
    cost: float
    scores: dict = field(default_factory=dict)

class ModelEvaluationPipeline:
    """モデル評価パイプライン"""

    def __init__(self, eval_cases: list[EvalCase]):
        self.eval_cases = eval_cases
        self.results: list[EvalResult] = []
        self.scorers: list[Callable] = []

    def add_scorer(self, scorer: Callable):
        """スコアラーを追加"""
        self.scorers.append(scorer)

    async def run_evaluation(self, models: list[str]) -> dict:
        """全モデルで全ケースの評価を実行"""
        for model in models:
            for case in self.eval_cases:
                result = await self._evaluate_single(model, case)
                # スコアリング
                for scorer in self.scorers:
                    score = scorer(case, result)
                    result.scores.update(score)
                self.results.append(result)

        return self._compile_report()

    async def _evaluate_single(self, model: str, case: EvalCase) -> EvalResult:
        """単一ケースの評価"""
        start = time.time()

        # モデル呼び出し（簡略化）
        output = await self._call_model(model, case.prompt)

        latency = time.time() - start

        return EvalResult(
            case_id=case.id,
            model=model,
            output=output["text"],
            latency=latency,
            input_tokens=output["input_tokens"],
            output_tokens=output["output_tokens"],
            cost=output["cost"],
        )

    async def _call_model(self, model: str, prompt: str) -> dict:
        """モデルAPIを呼び出す（抽象化）"""
        # 実装は省略 - LiteLLM等を使用
        pass

    def _compile_report(self) -> dict:
        """評価レポートを作成"""
        models = set(r.model for r in self.results)
        report = {}

        for model in models:
            model_results = [r for r in self.results if r.model == model]

            # 品質スコア
            quality_scores = [
                sum(r.scores.values()) / len(r.scores)
                for r in model_results if r.scores
            ]

            # コスト・レイテンシ
            total_cost = sum(r.cost for r in model_results)
            avg_latency = sum(r.latency for r in model_results) / len(model_results)

            # カテゴリ別スコア
            categories = set(
                case.category for case in self.eval_cases
            )
            category_scores = {}
            for cat in categories:
                cat_results = [
                    r for r in model_results
                    if any(c.id == r.case_id and c.category == cat for c in self.eval_cases)
                ]
                if cat_results and cat_results[0].scores:
                    category_scores[cat] = sum(
                        sum(r.scores.values()) / len(r.scores)
                        for r in cat_results
                    ) / len(cat_results)

            report[model] = {
                "avg_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                "total_cost": total_cost,
                "avg_latency": avg_latency,
                "category_scores": category_scores,
                "total_cases": len(model_results),
            }

        return report


# スコアラーの例
def exact_match_scorer(case: EvalCase, result: EvalResult) -> dict:
    """完全一致スコアラー"""
    return {"exact_match": 1.0 if result.output.strip() == case.expected.strip() else 0.0}

def contains_scorer(case: EvalCase, result: EvalResult) -> dict:
    """部分一致スコアラー"""
    return {"contains": 1.0 if case.expected.lower() in result.output.lower() else 0.0}

def length_penalty_scorer(case: EvalCase, result: EvalResult) -> dict:
    """長さペナルティスコアラー"""
    expected_len = len(case.expected)
    actual_len = len(result.output)
    ratio = actual_len / expected_len if expected_len > 0 else 0
    # 0.5-2.0 の範囲が理想
    if 0.5 <= ratio <= 2.0:
        score = 1.0
    elif ratio < 0.5:
        score = ratio * 2  # 短すぎるペナルティ
    else:
        score = max(0, 1.0 - (ratio - 2.0) * 0.2)  # 長すぎるペナルティ
    return {"length_score": score}
```

---

## 10. FAQ

### Q1: 新しいモデルが出たら毎回乗り換えるべき?

頻繁な乗り換えはコスト (検証工数、コード変更、プロンプト再調整) が大きい。
3-6 ヶ月に一度、主要モデルの比較評価を行い、有意な改善が確認できた場合のみ移行するのが現実的。
抽象レイヤー (LiteLLM 等) を導入しておくと切り替えコストが下がる。

**具体的な判断基準:**
- 自社タスクでの品質が10%以上向上
- コストが30%以上削減
- レイテンシが50%以上改善
- 新機能 (マルチモーダル等) が必須要件に追加
- 現行モデルの廃止アナウンス

### Q2: 複数モデルを組み合わせるメリットは?

Router パターンでタスク難易度に応じてモデルを振り分けると、コストを 60-80% 削減できることがある。
例: 簡単な質問 → Flash/mini、複雑な質問 → Pro/Sonnet、コード生成 → 特化モデル。
OpenRouter や LiteLLM を使えばルーティングを容易に実装できる。

**実装パターン:**
1. **静的ルーティング**: タスクタイプでモデルを固定割当
2. **動的ルーティング**: 入力の複雑さを分析して動的選択
3. **カスケード**: 小型モデルで試行 → 品質不足なら大型モデルにフォールバック
4. **アンサンブル**: 複数モデルの回答を統合（高コストだが高品質）

### Q3: ベンチマークと実運用の性能差はどの程度?

ベンチマーク汚染 (訓練データにベンチマーク問題が混入) の問題があり、特に MMLU では実力との乖離が指摘されている。
LMSYS Chatbot Arena のような人間評価が最も実態に近いが、自社タスクでの独自評価が最も信頼できる。
「ベンチマークは足切りに使い、最終判断は実タスク評価」が推奨される。

### Q4: 推論モデル (o1/R1) と通常モデルの使い分け基準は?

推論モデルは「考える時間」を増やすことで精度を上げるため、以下の場合に有効:
- 数学の証明や複雑な論理推論
- 複数ステップの分析が必要な問題
- 正確性が最優先で、レイテンシは許容できる場合

一方、以下の場合は通常モデルで十分:
- FAQ回答、翻訳、要約などの定型タスク
- レイテンシが重要なリアルタイムチャット
- 大量バッチ処理（コスト面で非現実的）

### Q5: 日本語タスクで最適なモデルは?

日本語性能は学習データの量と質に大きく依存する。2025年初頭時点での日本語性能ランキング（概算）:

1. **GPT-4o** — 日本語学習データが豊富、自然な表現
2. **Claude 3.5 Sonnet** — 高い日本語理解力、長文に強い
3. **Gemini 1.5 Pro** — 日本語性能向上、長文処理に圧倒的
4. **Qwen 2.5 72B** — CJK特化、OSS最高の日本語性能
5. **DeepSeek-V3** — 中国語ベースだが日本語も良好

自社での日本語特化評価データセット（JCommonsenseQA、JNLI、JSQuAD等）でのテストを推奨。

### Q6: ファインチューニングすべきか、プロンプトエンジニアリングで十分か?

```
判断フローチャート:

プロンプトエンジニアリングで目標品質達成?
├── YES → ファインチューニング不要
└── NO → Few-shot / RAG で改善?
    ├── YES → ファインチューニング不要
    └── NO → 学習データ1000件以上用意可能?
        ├── YES → ファインチューニング検討
        │   ├── API FT (GPT-4o mini, Gemini) → 手軽
        │   └── OSS FT (Llama, Qwen) → 自由度高い
        └── NO → データ収集から開始、または別モデルを検討
```

---

## まとめ

| 評価軸 | 最推奨モデル | 備考 |
|--------|------------|------|
| 総合性能 | GPT-4o / Claude 3.5 Sonnet | 僅差、タスク依存 |
| コストパフォーマンス | Gemini 2.0 Flash / GPT-4o mini | 10-50倍安い |
| 長文処理 | Gemini 1.5 Pro (2M) | 他を圧倒 |
| コード生成 | Claude 3.5 Sonnet | SWE-bench 最高 |
| 数学・推論 | o3-mini / DeepSeek-R1 | CoT 特化 |
| 日本語 | Qwen 2.5 / GPT-4o | CJK 強い |
| プライバシー | OSS 自前デプロイ | Qwen/Llama |
| マルチモーダル | GPT-4o / Gemini 1.5 | 動画は Gemini のみ |
| 超低コスト | Gemini 2.0 Flash | $0.10/$0.40 per 1M |
| 推論タスク | o3-mini (high) | reasoning_effort 調整可 |

---

## 次に読むべきガイド

- [../02-applications/00-prompt-engineering.md](../02-applications/00-prompt-engineering.md) — 選んだモデルの性能を最大化するプロンプト技法
- [../03-infrastructure/00-api-integration.md](../03-infrastructure/00-api-integration.md) — API 統合の実践
- [../03-infrastructure/03-evaluation.md](../03-infrastructure/03-evaluation.md) — 自社タスクでの評価手法

---

## 参考文献

1. LMSYS, "Chatbot Arena Leaderboard," https://chat.lmsys.org/
2. Hugging Face, "Open LLM Leaderboard," https://huggingface.co/spaces/open-llm-leaderboard
3. Artificial Analysis, "LLM Performance Leaderboard," https://artificialanalysis.ai/
4. Chiang et al., "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference," arXiv:2403.04132, 2024
5. DeepSeek, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," 2025
6. OpenAI, "Learning to Reason with LLMs," https://openai.com/index/learning-to-reason-with-llms/
7. Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" arXiv:2310.06770, 2023
8. Wang et al., "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark," 2024

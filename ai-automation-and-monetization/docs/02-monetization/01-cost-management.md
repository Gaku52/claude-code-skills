# コスト管理 — API費用最適化、キャッシュ戦略

> AI APIの費用を最適化し、キャッシュ戦略、モデル選択、バッチ処理、プロンプト最適化を通じてコストを50-80%削減する実践的な手法を体系的に解説する。

---

## この章で学ぶこと

1. **AI APIコストの構造と可視化** — トークン課金の仕組み、コスト配分の分析、予算管理ダッシュボード
2. **キャッシュ戦略の設計と実装** — セマンティックキャッシュ、階層キャッシュ、TTL最適化
3. **プロンプト/モデル最適化** — トークン削減、モデル使い分け、バッチ処理による費用削減

---

## 1. AI APIコスト構造

### 1.1 主要AI APIの料金比較

```
┌──────────────────────────────────────────────────────────┐
│           主要AI API 料金比較 (2025年時点)                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  モデル            入力 (/1M tokens)  出力 (/1M tokens)   │
│  ─────────────────────────────────────────────────       │
│  GPT-4o            $2.50              $10.00              │
│  GPT-4o-mini       $0.15              $0.60               │
│  GPT-4 Turbo       $10.00             $30.00              │
│  Claude Sonnet     $3.00              $15.00              │
│  Claude Haiku      $0.25              $1.25               │
│  Claude Opus       $15.00             $75.00              │
│  Gemini 1.5 Pro    $1.25              $5.00               │
│  Gemini 1.5 Flash  $0.075             $0.30               │
│  Llama 3 70B*      $0.00              $0.00 (セルフホスト) │
│                                                          │
│  * セルフホスト: GPU費用 $1-3/時間が別途必要               │
└──────────────────────────────────────────────────────────┘
```

### 1.2 コスト分析ダッシュボード

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

@dataclass
class APIUsageRecord:
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    endpoint: str
    user_id: str

class CostAnalyzer:
    """APIコスト分析エンジン"""

    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-sonnet": {"input": 3.00, "output": 15.00},
        "claude-haiku": {"input": 0.25, "output": 1.25},
    }

    def __init__(self):
        self.records: list[APIUsageRecord] = []

    def record(self, model: str, input_tokens: int,
               output_tokens: int, endpoint: str,
               user_id: str):
        """使用量記録"""
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        cost = (
            input_tokens / 1_000_000 * pricing["input"] +
            output_tokens / 1_000_000 * pricing["output"]
        )
        self.records.append(APIUsageRecord(
            timestamp=datetime.now(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            endpoint=endpoint,
            user_id=user_id
        ))

    def get_daily_report(self, date=None) -> dict:
        """日次コストレポート"""
        date = date or datetime.now().date()
        day_records = [
            r for r in self.records
            if r.timestamp.date() == date
        ]

        by_model = defaultdict(lambda: {"count": 0, "cost": 0})
        by_endpoint = defaultdict(lambda: {"count": 0, "cost": 0})

        for r in day_records:
            by_model[r.model]["count"] += 1
            by_model[r.model]["cost"] += r.cost
            by_endpoint[r.endpoint]["count"] += 1
            by_endpoint[r.endpoint]["cost"] += r.cost

        total_cost = sum(r.cost for r in day_records)

        return {
            "date": str(date),
            "total_cost": round(total_cost, 4),
            "total_requests": len(day_records),
            "avg_cost_per_request": round(
                total_cost / len(day_records), 4
            ) if day_records else 0,
            "by_model": dict(by_model),
            "by_endpoint": dict(by_endpoint)
        }
```

### 1.3 コスト構成の可視化

| 最適化項目 | 削減可能率 | 難易度 | 優先度 |
|-----------|----------|--------|--------|
| キャッシュ導入 | 30-50% | 中 | 最高 |
| モデル使い分け | 40-70% | 低 | 最高 |
| プロンプト最適化 | 20-40% | 低 | 高 |
| バッチ処理 | 10-30% | 中 | 高 |
| レスポンス制限 | 10-20% | 低 | 中 |
| セルフホスト移行 | 50-90% | 高 | 条件付き |

---

## 2. キャッシュ戦略

### 2.1 キャッシュアーキテクチャ

```
3層キャッシュアーキテクチャ:

  リクエスト
      │
      ▼
  ┌──────────┐  ヒット → 即応答（0ms, $0）
  │ L1: 完全  │
  │ 一致キャッシュ│  Redis / インメモリ
  └────┬─────┘
       │ ミス
       ▼
  ┌──────────┐  ヒット → 類似結果返却（10ms, $0）
  │ L2: セマン │
  │ ティック   │  ベクトルDB (Pinecone/pgvector)
  │ キャッシュ │  類似度 > 0.95 でヒット
  └────┬─────┘
       │ ミス
       ▼
  ┌──────────┐  API呼び出し（500ms, $0.01-$0.10）
  │ L3: AI API│
  │ 呼び出し  │  結果をL1/L2に保存
  └──────────┘
```

### 2.2 完全一致キャッシュ

```python
import hashlib
import json
import redis
from typing import Optional

class ExactMatchCache:
    """完全一致キャッシュ（Redis）"""

    def __init__(self, redis_url: str = "redis://localhost:6379",
                 default_ttl: int = 3600):
        self.redis = redis.from_url(redis_url)
        self.default_ttl = default_ttl
        self.stats = {"hits": 0, "misses": 0}

    def _make_key(self, model: str, messages: list,
                  params: dict) -> str:
        """キャッシュキー生成"""
        content = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": params.get("temperature", 1.0),
            "max_tokens": params.get("max_tokens")
        }, sort_keys=True)
        return f"ai_cache:{hashlib.sha256(content.encode()).hexdigest()}"

    def get(self, model: str, messages: list,
            params: dict) -> Optional[str]:
        """キャッシュ取得"""
        key = self._make_key(model, messages, params)
        result = self.redis.get(key)
        if result:
            self.stats["hits"] += 1
            return json.loads(result)
        self.stats["misses"] += 1
        return None

    def set(self, model: str, messages: list,
            params: dict, response: str,
            ttl: int = None):
        """キャッシュ保存"""
        key = self._make_key(model, messages, params)
        self.redis.setex(
            key,
            ttl or self.default_ttl,
            json.dumps(response)
        )

    @property
    def hit_rate(self) -> float:
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total if total > 0 else 0
```

### 2.3 セマンティックキャッシュ

```python
import numpy as np
from openai import OpenAI

class SemanticCache:
    """セマンティックキャッシュ（類似クエリで再利用）"""

    def __init__(self, similarity_threshold: float = 0.95):
        self.client = OpenAI()
        self.threshold = similarity_threshold
        self.cache: list[dict] = []

    def _get_embedding(self, text: str) -> list[float]:
        """テキストの埋め込みベクトル取得"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def _cosine_similarity(self, a: list[float],
                           b: list[float]) -> float:
        """コサイン類似度"""
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get(self, query: str) -> Optional[str]:
        """類似クエリのキャッシュを検索"""
        query_embedding = self._get_embedding(query)

        best_match = None
        best_score = 0

        for entry in self.cache:
            score = self._cosine_similarity(
                query_embedding, entry["embedding"]
            )
            if score > best_score:
                best_score = score
                best_match = entry

        if best_match and best_score >= self.threshold:
            return best_match["response"]
        return None

    def set(self, query: str, response: str):
        """キャッシュに追加"""
        embedding = self._get_embedding(query)
        self.cache.append({
            "query": query,
            "embedding": embedding,
            "response": response,
            "created_at": datetime.now()
        })
```

---

## 3. モデル使い分け戦略

### 3.1 タスク別最適モデル選択

```
タスク別モデル選択マトリクス:

  品質要求
  高 ┤ ● 契約書分析     ● コード生成
     │   → GPT-4/Opus     → GPT-4/Sonnet
     │
  中 ┤ ● 記事要約       ● 翻訳
     │   → Sonnet/Haiku    → Sonnet
     │
  低 ┤ ● メール分類     ● テキスト整形
     │   → Haiku/Mini      → Mini/Flash
     └──┬────────────┬────────────┬──
       低速OK       中速         高速必須
                 速度要求
```

### 3.2 インテリジェントルーティング

```python
class ModelRouter:
    """コスト最適化モデルルーティング"""

    MODELS = {
        "fast_cheap": {
            "name": "gpt-4o-mini",
            "cost_per_1k_tokens": 0.00015,
            "quality": 0.7,
            "speed": "fast"
        },
        "balanced": {
            "name": "claude-haiku",
            "cost_per_1k_tokens": 0.00025,
            "quality": 0.8,
            "speed": "fast"
        },
        "high_quality": {
            "name": "gpt-4o",
            "cost_per_1k_tokens": 0.0025,
            "quality": 0.95,
            "speed": "medium"
        },
        "best": {
            "name": "claude-sonnet",
            "cost_per_1k_tokens": 0.003,
            "quality": 0.98,
            "speed": "medium"
        }
    }

    def select_model(self, task_type: str,
                     quality_required: float = 0.8,
                     budget_sensitive: bool = False) -> str:
        """タスクに最適なモデルを選択"""
        task_mapping = {
            "classification": "fast_cheap",
            "summarization": "balanced",
            "translation": "balanced",
            "content_generation": "high_quality",
            "code_generation": "high_quality",
            "contract_analysis": "best",
            "creative_writing": "best"
        }

        # タスク種別でデフォルト選択
        default = task_mapping.get(task_type, "balanced")
        model = self.MODELS[default]

        # 品質要求で調整
        if quality_required > 0.95 and model["quality"] < 0.95:
            model = self.MODELS["best"]
        elif budget_sensitive and model["quality"] > quality_required:
            # 品質を満たす最安モデルを選択
            cheapest = min(
                (m for m in self.MODELS.values()
                 if m["quality"] >= quality_required),
                key=lambda m: m["cost_per_1k_tokens"]
            )
            model = cheapest

        return model["name"]

# 使用例
router = ModelRouter()
model = router.select_model("classification", budget_sensitive=True)
# → "gpt-4o-mini"（分類は軽量モデルで十分）
```

---

## 4. プロンプト最適化

### 4.1 トークン削減テクニック

```python
# トークン削減の実例

# BAD: 冗長なプロンプト（約200トークン）
prompt_verbose = """
あなたは非常に優秀なAIアシスタントです。
あなたの仕事は、与えられたテキストを読んで、
そのテキストの内容を短く要約することです。
要約は3行以内にしてください。
できるだけ重要な情報を含めてください。
以下のテキストを要約してください:

{text}

上記のテキストの要約を3行以内で書いてください。
"""

# GOOD: 簡潔なプロンプト（約50トークン、75%削減）
prompt_concise = """
3行で要約:
{text}
"""

# トークン数の差:
# verbose: ~200 tokens × 10,000回/月 = 2M tokens → $5.00
# concise: ~50 tokens × 10,000回/月 = 500K tokens → $1.25
# 月間節約: $3.75 (75%削減)
```

### 4.2 バッチ処理

```python
class BatchProcessor:
    """バッチ処理でAPI呼び出しを最適化"""

    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        self.queue: list[dict] = []

    def add_task(self, task: dict):
        """タスクをキューに追加"""
        self.queue.append(task)
        if len(self.queue) >= self.batch_size:
            return self.process_batch()
        return None

    def process_batch(self) -> list[dict]:
        """バッチ処理実行"""
        if not self.queue:
            return []

        # 複数タスクを1回のAPI呼び出しにまとめる
        combined_prompt = "以下の各項目を処理:\n\n"
        for i, task in enumerate(self.queue):
            combined_prompt += f"[{i+1}] {task['prompt']}\n"
        combined_prompt += "\nJSON配列で各項目の結果を返す。"

        response = call_ai(combined_prompt)
        results = parse_json_array(response)

        self.queue.clear()
        return results

# 効果:
# 個別処理: 10回 × (システムプロンプト100tokens + 入力) = 1000 tokens overhead
# バッチ処理: 1回 × (システムプロンプト100tokens + 全入力) = 100 tokens overhead
# → オーバーヘッド90%削減
```

---

## 5. アンチパターン

### アンチパターン1: キャッシュなしの全リクエストAPI呼び出し

```python
# BAD: 全リクエストを毎回APIに送信
def summarize(text):
    return call_api(text)  # 同じ入力でも毎回課金

# GOOD: 3層キャッシュで大幅削減
def summarize(text):
    # L1: 完全一致
    cached = exact_cache.get(text)
    if cached:
        return cached  # $0

    # L2: セマンティック
    similar = semantic_cache.get(text)
    if similar:
        return similar  # $0.0001 (embedding費用のみ)

    # L3: API呼び出し
    result = call_api(text)  # $0.01-$0.10
    exact_cache.set(text, result)
    semantic_cache.set(text, result)
    return result
```

### アンチパターン2: 最高性能モデルの一律使用

```python
# BAD: 全タスクにGPT-4を使用
def process_all(tasks):
    for task in tasks:
        result = call_ai(task, model="gpt-4")  # 全部GPT-4

# GOOD: タスク複雑度に応じたモデル選択
def process_all(tasks):
    for task in tasks:
        complexity = estimate_complexity(task)
        if complexity == "simple":
            result = call_ai(task, model="gpt-4o-mini")  # 1/17のコスト
        elif complexity == "medium":
            result = call_ai(task, model="claude-haiku")  # 1/12のコスト
        else:
            result = call_ai(task, model="gpt-4o")  # 高品質が必要な場合のみ
```

---

## 6. FAQ

### Q1: キャッシュのヒット率はどのくらいが目安？

**A:** 用途で大きく異なる。(1) カスタマーサポート（FAQ系）: 40-60%（同じ質問が繰り返される）、(2) コンテンツ生成: 10-20%（毎回異なる入力）、(3) 分類タスク: 30-50%。セマンティックキャッシュを導入すると完全一致の2-3倍のヒット率になる。目安として全体で30%以上を目指す。

### Q2: セルフホストLLMへの移行はいつすべき？

**A:** 3条件が揃ったとき。(1) 月間API費用が$5,000以上、(2) レイテンシ要件が厳しい（<100ms）、(3) データ主権の要件がある。GPU費用（A100: $2-3/時間）を考慮すると、月$3,000以下ならAPI利用の方が安い。Llama 3やMistralのファインチューニング版を使えば、GPT-4の80-90%の品質を1/10のコストで実現可能。

### Q3: 予算超過を防ぐ方法は？

**A:** 4層の防御を推奨。(1) ハード制限 — API使用量の上限をOpenAI/Anthropicの管理画面で設定、(2) アラート — 予算の50%/80%/100%でSlack通知、(3) ソフト制限 — ユーザー単位の日次/月次上限、(4) 回路遮断 — 異常な増加を検知したら自動停止。特にユーザー入力の長さ制限（max_tokens設定）が重要。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| コスト構造 | APIコストが全体の20-40%、入力/出力トークンで課金 |
| キャッシュ | 3層（完全一致→セマンティック→API）で30-50%削減 |
| モデル選択 | タスク複雑度でルーティング、40-70%削減可能 |
| プロンプト | 簡潔化で20-40%削減、バッチ処理で更に削減 |
| 監視 | 日次レポート + 予算アラート + 自動停止 |
| 目標 | 粗利70%以上を維持、改善は継続的に |

---

## 次に読むべきガイド

- [02-scaling-strategy.md](./02-scaling-strategy.md) — スケーリング戦略
- [00-pricing-models.md](./00-pricing-models.md) — 価格モデル設計
- [../00-automation/00-automation-overview.md](../00-automation/00-automation-overview.md) — AI自動化概要

---

## 参考文献

1. **OpenAI API Pricing** — https://openai.com/pricing — 最新のトークン料金表
2. **Anthropic API Pricing** — https://docs.anthropic.com — Claude APIの料金とベストプラクティス
3. **"Reducing LLM Costs" — Martian (2024)** — LLMコスト最適化の包括的ガイド
4. **Redis Documentation** — https://redis.io/docs — キャッシュ実装のベストプラクティス

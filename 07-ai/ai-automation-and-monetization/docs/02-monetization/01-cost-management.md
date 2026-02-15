# コスト管理 — API費用最適化、キャッシュ戦略

> AI APIの費用を最適化し、キャッシュ戦略、モデル選択、バッチ処理、プロンプト最適化を通じてコストを50-80%削減する実践的な手法を体系的に解説する。

---

## この章で学ぶこと

1. **AI APIコストの構造と可視化** — トークン課金の仕組み、コスト配分の分析、予算管理ダッシュボード
2. **キャッシュ戦略の設計と実装** — セマンティックキャッシュ、階層キャッシュ、TTL最適化
3. **プロンプト/モデル最適化** — トークン削減、モデル使い分け、バッチ処理による費用削減
4. **予算管理と監視** — リアルタイム監視、アラート設計、異常検知
5. **セルフホスト戦略** — オンプレミスLLM運用のコスト分析と実装

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

    def get_user_cost_report(self, user_id: str,
                              days: int = 30) -> dict:
        """ユーザー別コストレポート"""
        cutoff = datetime.now() - timedelta(days=days)
        user_records = [
            r for r in self.records
            if r.user_id == user_id and r.timestamp >= cutoff
        ]

        total_cost = sum(r.cost for r in user_records)
        total_tokens = sum(
            r.input_tokens + r.output_tokens for r in user_records
        )
        total_requests = len(user_records)

        # 日別集計
        daily_costs = defaultdict(float)
        for r in user_records:
            daily_costs[str(r.timestamp.date())] += r.cost

        # モデル別集計
        model_usage = defaultdict(lambda: {"count": 0, "cost": 0, "tokens": 0})
        for r in user_records:
            model_usage[r.model]["count"] += 1
            model_usage[r.model]["cost"] += r.cost
            model_usage[r.model]["tokens"] += r.input_tokens + r.output_tokens

        return {
            "user_id": user_id,
            "period_days": days,
            "total_cost": round(total_cost, 4),
            "total_tokens": total_tokens,
            "total_requests": total_requests,
            "avg_daily_cost": round(total_cost / days, 4),
            "avg_cost_per_request": round(
                total_cost / total_requests, 4
            ) if total_requests > 0 else 0,
            "daily_costs": dict(daily_costs),
            "model_usage": dict(model_usage)
        }

    def detect_anomalies(self, threshold_multiplier: float = 3.0) -> list[dict]:
        """コスト異常検知"""
        # 過去7日間の平均日次コストを計算
        today = datetime.now().date()
        daily_totals = defaultdict(float)
        for r in self.records:
            daily_totals[str(r.timestamp.date())] += r.cost

        costs = list(daily_totals.values())
        if len(costs) < 3:
            return []

        avg_cost = sum(costs[:-1]) / len(costs[:-1])  # 直近を除く
        std_cost = (
            sum((c - avg_cost) ** 2 for c in costs[:-1])
            / len(costs[:-1])
        ) ** 0.5

        threshold = avg_cost + threshold_multiplier * std_cost
        today_cost = daily_totals.get(str(today), 0)

        anomalies = []
        if today_cost > threshold:
            anomalies.append({
                "type": "daily_cost_spike",
                "date": str(today),
                "actual_cost": round(today_cost, 4),
                "expected_cost": round(avg_cost, 4),
                "threshold": round(threshold, 4),
                "severity": "critical" if today_cost > threshold * 2 else "warning",
                "message": f"本日のコスト${today_cost:.2f}が閾値${threshold:.2f}を超過"
            })

        return anomalies
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

### 1.4 コスト配分の分析フレームワーク

```python
class CostAllocationFramework:
    """コスト配分分析フレームワーク"""

    def analyze_cost_drivers(self, records: list[APIUsageRecord]) -> dict:
        """コストドライバー分析"""
        total_cost = sum(r.cost for r in records)
        if total_cost == 0:
            return {"error": "コストデータなし"}

        # エンドポイント別コスト比率
        endpoint_costs = defaultdict(float)
        for r in records:
            endpoint_costs[r.endpoint] += r.cost

        # トップ5コストドライバー
        sorted_endpoints = sorted(
            endpoint_costs.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # ユーザー別コスト（パレート分析）
        user_costs = defaultdict(float)
        for r in records:
            user_costs[r.user_id] += r.cost

        sorted_users = sorted(
            user_costs.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 上位20%のユーザーが何%のコストを占めるか
        top_20_pct = int(len(sorted_users) * 0.2) or 1
        top_20_cost = sum(c for _, c in sorted_users[:top_20_pct])
        pareto_ratio = top_20_cost / total_cost * 100

        # 時間帯別コスト
        hourly_costs = defaultdict(float)
        for r in records:
            hourly_costs[r.timestamp.hour] += r.cost

        peak_hour = max(hourly_costs, key=hourly_costs.get)

        return {
            "total_cost": f"${total_cost:.2f}",
            "top_endpoints": [
                {
                    "endpoint": ep,
                    "cost": f"${cost:.2f}",
                    "percentage": f"{cost/total_cost*100:.1f}%"
                }
                for ep, cost in sorted_endpoints
            ],
            "pareto_analysis": {
                "top_20_pct_users": top_20_pct,
                "cost_share": f"{pareto_ratio:.1f}%",
                "insight": f"上位{top_20_pct}ユーザーがコストの"
                          f"{pareto_ratio:.0f}%を消費"
            },
            "peak_hour": {
                "hour": peak_hour,
                "cost": f"${hourly_costs[peak_hour]:.2f}",
                "suggestion": "ピーク時間帯のリクエストをバッチ化検討"
            }
        }

    def calculate_unit_cost(self, records: list[APIUsageRecord],
                            revenue: float) -> dict:
        """ユニットコスト計算"""
        total_cost = sum(r.cost for r in records)
        total_requests = len(records)

        cost_per_request = total_cost / total_requests if total_requests > 0 else 0
        cost_revenue_ratio = total_cost / revenue * 100 if revenue > 0 else 0
        gross_margin = (revenue - total_cost) / revenue * 100 if revenue > 0 else 0

        return {
            "total_api_cost": f"${total_cost:.2f}",
            "total_requests": total_requests,
            "cost_per_request": f"${cost_per_request:.4f}",
            "revenue": f"${revenue:.2f}",
            "cost_revenue_ratio": f"{cost_revenue_ratio:.1f}%",
            "gross_margin": f"{gross_margin:.1f}%",
            "health": "healthy" if gross_margin >= 70 else (
                "acceptable" if gross_margin >= 50 else "unhealthy"
            ),
            "targets": {
                "cost_revenue_ratio": "20-30%が理想",
                "gross_margin": "70%以上が目標",
                "cost_per_request": f"${cost_per_request * 0.5:.4f}以下を目指す"
            }
        }
```

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

    def get_stats(self) -> dict:
        """キャッシュ統計"""
        total = self.stats["hits"] + self.stats["misses"]
        return {
            "total_requests": total,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": f"{self.hit_rate * 100:.1f}%",
            "estimated_savings": f"${self.stats['hits'] * 0.01:.2f}"
        }
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


class ProductionSemanticCache:
    """本番環境向けセマンティックキャッシュ（pgvector使用）"""

    def __init__(self, db_url: str,
                 similarity_threshold: float = 0.95):
        self.db_url = db_url
        self.threshold = similarity_threshold
        self.client = OpenAI()

    def setup_table(self):
        """テーブル作成（初回のみ）"""
        # PostgreSQL + pgvector
        sql = """
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS semantic_cache (
            id SERIAL PRIMARY KEY,
            query_text TEXT NOT NULL,
            query_hash VARCHAR(64) NOT NULL,
            embedding vector(1536),
            response JSONB NOT NULL,
            model VARCHAR(50) NOT NULL,
            hit_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW(),
            last_accessed TIMESTAMP DEFAULT NOW(),
            ttl_seconds INTEGER DEFAULT 86400
        );

        CREATE INDEX IF NOT EXISTS idx_cache_embedding
            ON semantic_cache
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);

        CREATE INDEX IF NOT EXISTS idx_cache_hash
            ON semantic_cache (query_hash);
        """
        return sql

    def search_similar(self, query: str,
                       model: str) -> Optional[dict]:
        """類似クエリ検索"""
        embedding = self._get_embedding(query)
        embedding_str = str(embedding)

        sql = """
        SELECT id, query_text, response, hit_count,
               1 - (embedding <=> %s::vector) AS similarity
        FROM semantic_cache
        WHERE model = %s
          AND created_at + (ttl_seconds * interval '1 second') > NOW()
        ORDER BY embedding <=> %s::vector
        LIMIT 1
        """

        # 結果が閾値以上なら返却
        # result = db.execute(sql, (embedding_str, model, embedding_str))
        # if result and result.similarity >= self.threshold:
        #     更新: hit_count += 1, last_accessed = NOW()
        #     return result.response

        return None

    def store(self, query: str, model: str,
              response: dict, ttl: int = 86400):
        """キャッシュ保存"""
        embedding = self._get_embedding(query)
        query_hash = hashlib.sha256(query.encode()).hexdigest()

        sql = """
        INSERT INTO semantic_cache
            (query_text, query_hash, embedding, response, model, ttl_seconds)
        VALUES (%s, %s, %s::vector, %s, %s, %s)
        ON CONFLICT (query_hash)
        DO UPDATE SET
            response = EXCLUDED.response,
            last_accessed = NOW()
        """
        # db.execute(sql, (query, query_hash, str(embedding),
        #                   json.dumps(response), model, ttl))

    def _get_embedding(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def cleanup_expired(self):
        """期限切れキャッシュの削除"""
        sql = """
        DELETE FROM semantic_cache
        WHERE created_at + (ttl_seconds * interval '1 second') < NOW()
        """
        # db.execute(sql)

    def get_cache_analytics(self) -> dict:
        """キャッシュ分析"""
        sql = """
        SELECT
            model,
            COUNT(*) as total_entries,
            SUM(hit_count) as total_hits,
            AVG(hit_count) as avg_hits,
            MIN(created_at) as oldest_entry,
            MAX(last_accessed) as latest_access,
            pg_size_pretty(pg_total_relation_size('semantic_cache'))
                as table_size
        FROM semantic_cache
        GROUP BY model
        """
        # return db.execute(sql)
        return {}
```

### 2.4 TTL最適化戦略

```python
class TTLOptimizer:
    """TTL（Time-to-Live）最適化"""

    # タスク種類別のTTL設定
    TTL_CONFIGS = {
        "static_knowledge": {
            "ttl_seconds": 86400 * 30,  # 30日
            "description": "変化しない知識（歴史、科学等）",
            "examples": ["Pythonの基本構文は？", "東京の人口は？"]
        },
        "semi_static": {
            "ttl_seconds": 86400 * 7,  # 7日
            "description": "頻繁には変化しない情報",
            "examples": ["AIモデルの比較", "プログラミングのベストプラクティス"]
        },
        "daily_update": {
            "ttl_seconds": 86400,  # 1日
            "description": "日次で更新される情報",
            "examples": ["天気予報の要約", "株価分析"]
        },
        "real_time": {
            "ttl_seconds": 300,  # 5分
            "description": "リアルタイム性が必要",
            "examples": ["ニュース要約", "リアルタイムチャット"]
        },
        "no_cache": {
            "ttl_seconds": 0,
            "description": "キャッシュ不可",
            "examples": ["個人データ分析", "セキュリティ関連"]
        }
    }

    def determine_ttl(self, task_type: str,
                      query: str) -> int:
        """クエリに適したTTLを決定"""
        # タスクタイプベース
        config = self.TTL_CONFIGS.get(task_type)
        if config:
            return config["ttl_seconds"]

        # キーワードベースのヒューリスティック
        real_time_keywords = ["今日", "現在", "最新", "リアルタイム"]
        if any(kw in query for kw in real_time_keywords):
            return 300  # 5分

        static_keywords = ["定義", "とは", "基本", "概要", "歴史"]
        if any(kw in query for kw in static_keywords):
            return 86400 * 30  # 30日

        return 86400  # デフォルト: 1日

    def adaptive_ttl(self, cache_key: str,
                      hit_frequency: float) -> int:
        """アクセス頻度に応じたTTL調整"""
        # 頻繁にヒットするキーは長めのTTL
        if hit_frequency >= 10:  # 1時間に10回以上
            return 86400 * 7  # 7日
        elif hit_frequency >= 1:
            return 86400  # 1日
        elif hit_frequency >= 0.1:
            return 3600  # 1時間
        else:
            return 300  # 5分
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

### 3.3 カスケードパターン

```python
class CascadeModelRouter:
    """カスケード型モデルルーティング

    軽量モデルで初回処理 → 品質不足なら上位モデルにエスカレーション
    """

    def __init__(self):
        self.cascade_order = [
            {"model": "gpt-4o-mini", "cost": 0.00015, "quality_threshold": 0.8},
            {"model": "claude-haiku", "cost": 0.00025, "quality_threshold": 0.9},
            {"model": "gpt-4o", "cost": 0.0025, "quality_threshold": 0.95},
            {"model": "claude-sonnet", "cost": 0.003, "quality_threshold": 1.0}
        ]

    def process(self, prompt: str, required_quality: float = 0.8,
                quality_evaluator=None) -> dict:
        """カスケード処理"""
        for level, config in enumerate(self.cascade_order):
            response = call_ai(prompt, model=config["model"])

            # 品質評価
            if quality_evaluator:
                quality_score = quality_evaluator(response)
            else:
                quality_score = config["quality_threshold"]

            if quality_score >= required_quality:
                return {
                    "response": response,
                    "model_used": config["model"],
                    "cascade_level": level,
                    "quality_score": quality_score,
                    "cost": config["cost"],
                    "message": f"レベル{level}で品質要件を達成"
                }

        # 最高品質モデルまで到達
        return {
            "response": response,
            "model_used": self.cascade_order[-1]["model"],
            "cascade_level": len(self.cascade_order) - 1,
            "quality_score": quality_score,
            "cost": self.cascade_order[-1]["cost"],
            "message": "最高品質モデルを使用"
        }

    def estimate_savings(self, task_distribution: dict) -> dict:
        """カスケードパターンによる節約額推定"""
        # task_distribution: {"simple": 0.6, "medium": 0.25, "complex": 0.15}
        baseline_cost = 0.003  # 全部sonnet使用
        cascade_cost = (
            task_distribution.get("simple", 0) * 0.00015 +
            task_distribution.get("medium", 0) * 0.00025 +
            task_distribution.get("complex", 0) * 0.003
        )

        savings_pct = (1 - cascade_cost / baseline_cost) * 100

        return {
            "baseline_cost_per_1k": f"${baseline_cost * 1000:.2f}",
            "cascade_cost_per_1k": f"${cascade_cost * 1000:.2f}",
            "savings_percentage": f"{savings_pct:.1f}%",
            "monthly_savings_at_100k_requests": f"${(baseline_cost - cascade_cost) * 100000:.0f}"
        }
```

### 3.4 フォールバック戦略

```python
import time
from typing import Callable


class ModelFallback:
    """モデルフォールバック（障害時の自動切り替え）"""

    def __init__(self):
        self.fallback_chains = {
            "primary": [
                {"model": "claude-sonnet", "provider": "anthropic"},
                {"model": "gpt-4o", "provider": "openai"},
                {"model": "gemini-1.5-pro", "provider": "google"}
            ],
            "fast": [
                {"model": "claude-haiku", "provider": "anthropic"},
                {"model": "gpt-4o-mini", "provider": "openai"},
                {"model": "gemini-1.5-flash", "provider": "google"}
            ]
        }
        self.provider_health = {
            "anthropic": {"healthy": True, "last_error": None},
            "openai": {"healthy": True, "last_error": None},
            "google": {"healthy": True, "last_error": None}
        }

    def call_with_fallback(self, prompt: str,
                           chain: str = "primary",
                           max_retries: int = 2) -> dict:
        """フォールバック付きAPI呼び出し"""
        models = self.fallback_chains[chain]

        for attempt, config in enumerate(models):
            provider = config["provider"]
            model = config["model"]

            # 不健全なプロバイダーはスキップ
            if not self.provider_health[provider]["healthy"]:
                continue

            for retry in range(max_retries):
                try:
                    response = self._call_api(
                        prompt, model, provider
                    )
                    return {
                        "response": response,
                        "model": model,
                        "provider": provider,
                        "attempt": attempt,
                        "retry": retry,
                        "fallback_used": attempt > 0
                    }
                except RateLimitError:
                    time.sleep(2 ** retry)  # エクスポネンシャルバックオフ
                except APIError as e:
                    self._mark_unhealthy(provider, str(e))
                    break  # 次のプロバイダーへ

        return {"error": "全プロバイダーで失敗"}

    def _call_api(self, prompt: str, model: str,
                  provider: str) -> str:
        """API呼び出し（プロバイダー別）"""
        # 実装はプロバイダーに依存
        pass

    def _mark_unhealthy(self, provider: str, error: str):
        """プロバイダーを不健全としてマーク"""
        self.provider_health[provider] = {
            "healthy": False,
            "last_error": error,
            "marked_at": datetime.now()
        }

    def health_check(self):
        """定期的なヘルスチェック"""
        for provider, status in self.provider_health.items():
            if not status["healthy"]:
                # 5分経過したら再試行
                if status.get("marked_at"):
                    elapsed = (datetime.now() - status["marked_at"]).seconds
                    if elapsed > 300:
                        self.provider_health[provider]["healthy"] = True
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

### 4.2 プロンプト圧縮ツール

```python
class PromptCompressor:
    """プロンプト圧縮ツール"""

    COMPRESSION_RULES = {
        "remove_filler": {
            "description": "フィラーワードの除去",
            "before": "あなたは非常に優秀なAIアシスタントです。",
            "after": "",
            "saving": "~15トークン"
        },
        "simplify_instruction": {
            "description": "指示の簡潔化",
            "before": "以下のテキストを読んで、その内容を短く要約してください。",
            "after": "要約:",
            "saving": "~20トークン"
        },
        "use_structured_format": {
            "description": "構造化フォーマット使用",
            "before": "名前は{name}で、年齢は{age}歳で、職業は{job}です。",
            "after": "name:{name}|age:{age}|job:{job}",
            "saving": "~10トークン"
        },
        "abbreviate_system_prompt": {
            "description": "システムプロンプトの短縮",
            "before": "You are a helpful assistant that specializes in...",
            "after": "Role: {role}. Task: {task}. Format: {format}.",
            "saving": "~30トークン"
        }
    }

    def compress(self, prompt: str) -> dict:
        """プロンプトを圧縮"""
        original_length = len(prompt.split())
        compressed = prompt

        # フィラーワード除去
        fillers = [
            "非常に優秀な", "とても素晴らしい",
            "できる限り", "可能な限り",
            "以下の", "上記の",
            "お願いします", "してください"
        ]
        for filler in fillers:
            compressed = compressed.replace(filler, "")

        # 冗長な表現の置換
        replacements = {
            "以下のテキストを要約してください": "要約:",
            "日本語で回答してください": "日本語で:",
            "箇条書きでリストアップしてください": "箇条書き:",
            "できるだけ詳しく説明してください": "詳細:",
        }
        for old, new in replacements.items():
            compressed = compressed.replace(old, new)

        compressed = compressed.strip()
        compressed_length = len(compressed.split())

        reduction = (1 - compressed_length / original_length) * 100 if original_length > 0 else 0

        return {
            "original": prompt,
            "compressed": compressed,
            "original_words": original_length,
            "compressed_words": compressed_length,
            "reduction": f"{reduction:.1f}%",
            "estimated_token_savings": int(
                (original_length - compressed_length) * 1.3
            )  # 日本語は1文字≒1.3トークン
        }

    def optimize_system_prompt(self, system_prompt: str) -> dict:
        """システムプロンプトの最適化"""
        # システムプロンプトは毎リクエストで送信されるため
        # 最適化効果が累積する
        compressed = self.compress(system_prompt)
        monthly_requests = 10000  # 仮定

        token_savings_per_request = compressed["estimated_token_savings"]
        monthly_token_savings = token_savings_per_request * monthly_requests

        # GPT-4oの入力コスト: $2.50/1M tokens
        monthly_cost_savings = monthly_token_savings / 1_000_000 * 2.50

        return {
            **compressed,
            "monthly_requests": monthly_requests,
            "monthly_token_savings": monthly_token_savings,
            "monthly_cost_savings": f"${monthly_cost_savings:.2f}",
            "annual_cost_savings": f"${monthly_cost_savings * 12:.2f}"
        }
```

### 4.3 バッチ処理

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

### 4.4 レスポンス長の最適化

```python
class ResponseOptimizer:
    """レスポンス長の最適化"""

    # タスク別の推奨max_tokens設定
    RECOMMENDED_MAX_TOKENS = {
        "classification": 10,       # "positive" or "negative"
        "sentiment": 5,            # 1-5のスコア
        "yes_no": 3,               # "yes" or "no"
        "short_answer": 50,        # 1-2文
        "summary": 200,            # 短い要約
        "long_summary": 500,       # 詳細な要約
        "article": 1000,           # 記事生成
        "code_snippet": 300,       # コードスニペット
        "full_code": 2000,         # 完全なコード
        "analysis": 800            # 分析レポート
    }

    def get_optimal_max_tokens(self, task_type: str) -> int:
        """タスクに応じた最適なmax_tokensを返す"""
        return self.RECOMMENDED_MAX_TOKENS.get(task_type, 500)

    def calculate_output_cost_savings(
        self,
        current_avg_output_tokens: int,
        optimized_output_tokens: int,
        monthly_requests: int,
        model: str = "gpt-4o"
    ) -> dict:
        """出力トークン最適化による節約額"""
        output_price = {"gpt-4o": 10.0, "claude-sonnet": 15.0,
                       "gpt-4o-mini": 0.6, "claude-haiku": 1.25}
        price = output_price.get(model, 10.0)

        current_monthly_cost = (
            current_avg_output_tokens * monthly_requests / 1_000_000 * price
        )
        optimized_monthly_cost = (
            optimized_output_tokens * monthly_requests / 1_000_000 * price
        )
        savings = current_monthly_cost - optimized_monthly_cost

        return {
            "current_avg_tokens": current_avg_output_tokens,
            "optimized_avg_tokens": optimized_output_tokens,
            "token_reduction": f"{(1-optimized_output_tokens/current_avg_output_tokens)*100:.0f}%",
            "current_monthly_cost": f"${current_monthly_cost:.2f}",
            "optimized_monthly_cost": f"${optimized_monthly_cost:.2f}",
            "monthly_savings": f"${savings:.2f}",
            "annual_savings": f"${savings*12:.2f}"
        }
```

---

## 5. 予算管理と監視

### 5.1 予算アラートシステム

```python
class BudgetManager:
    """AI API予算管理"""

    def __init__(self, monthly_budget: float):
        self.monthly_budget = monthly_budget
        self.alerts = []

    def check_budget(self, current_spend: float,
                      day_of_month: int) -> dict:
        """予算チェック"""
        # 線形消費ペースの計算
        expected_spend = self.monthly_budget * (day_of_month / 30)
        pace = current_spend / expected_spend if expected_spend > 0 else 0
        projected_monthly = current_spend * (30 / day_of_month)

        status = "on_track"
        if pace > 1.5:
            status = "critical"
        elif pace > 1.2:
            status = "warning"
        elif pace < 0.5:
            status = "under_utilized"

        alert = {
            "monthly_budget": f"${self.monthly_budget:.2f}",
            "current_spend": f"${current_spend:.2f}",
            "day_of_month": day_of_month,
            "expected_spend": f"${expected_spend:.2f}",
            "pace": f"{pace:.2f}x",
            "projected_monthly": f"${projected_monthly:.2f}",
            "remaining_budget": f"${self.monthly_budget - current_spend:.2f}",
            "remaining_days": 30 - day_of_month,
            "daily_budget_remaining": f"${(self.monthly_budget - current_spend) / (30 - day_of_month):.2f}" if day_of_month < 30 else "$0",
            "status": status,
            "actions": self._recommend_actions(status, pace)
        }

        return alert

    def _recommend_actions(self, status: str,
                            pace: float) -> list[str]:
        """推奨アクション"""
        if status == "critical":
            return [
                "即座にモデルを低コストモデルに切り替え",
                "キャッシュの閾値を緩和（類似度0.90に）",
                "非必須機能のAPI呼び出しを一時停止",
                "ユーザー当たりの日次制限を50%に削減"
            ]
        elif status == "warning":
            return [
                "コスト上位エンドポイントのモデル見直し",
                "バッチ処理の積極活用",
                "不要なリトライの削減"
            ]
        elif status == "under_utilized":
            return [
                "予算の再配分を検討",
                "キャッシュの有効期限を短縮して鮮度向上",
                "品質向上のために上位モデルの活用を検討"
            ]
        return ["現在のペースを維持"]

    def set_user_limits(self, plan: str) -> dict:
        """プラン別のユーザー制限"""
        limits = {
            "free": {
                "daily_requests": 10,
                "daily_tokens": 50000,
                "max_input_tokens": 2000,
                "max_output_tokens": 500,
                "models_allowed": ["gpt-4o-mini"],
                "rate_limit_rpm": 5  # リクエスト/分
            },
            "starter": {
                "daily_requests": 100,
                "daily_tokens": 500000,
                "max_input_tokens": 4000,
                "max_output_tokens": 2000,
                "models_allowed": ["gpt-4o-mini", "claude-haiku"],
                "rate_limit_rpm": 20
            },
            "pro": {
                "daily_requests": 1000,
                "daily_tokens": 5000000,
                "max_input_tokens": 8000,
                "max_output_tokens": 4000,
                "models_allowed": ["gpt-4o-mini", "claude-haiku",
                                  "gpt-4o", "claude-sonnet"],
                "rate_limit_rpm": 60
            },
            "enterprise": {
                "daily_requests": -1,  # 無制限
                "daily_tokens": -1,
                "max_input_tokens": 128000,
                "max_output_tokens": 8000,
                "models_allowed": ["all"],
                "rate_limit_rpm": 300
            }
        }
        return limits.get(plan, limits["free"])


class CostCircuitBreaker:
    """コスト回路遮断器"""

    def __init__(self, daily_limit: float,
                 hourly_limit: float):
        self.daily_limit = daily_limit
        self.hourly_limit = hourly_limit
        self.daily_spend = 0
        self.hourly_spend = 0
        self.is_open = False

    def check_and_record(self, cost: float) -> dict:
        """コストチェックと記録"""
        if self.is_open:
            return {
                "allowed": False,
                "reason": "回路遮断器がオープン",
                "message": "コスト上限に達したためリクエストを拒否"
            }

        self.daily_spend += cost
        self.hourly_spend += cost

        if self.hourly_spend >= self.hourly_limit:
            self.is_open = True
            return {
                "allowed": False,
                "reason": "時間当たりの上限超過",
                "hourly_spend": f"${self.hourly_spend:.2f}",
                "hourly_limit": f"${self.hourly_limit:.2f}"
            }

        if self.daily_spend >= self.daily_limit:
            self.is_open = True
            return {
                "allowed": False,
                "reason": "日次上限超過",
                "daily_spend": f"${self.daily_spend:.2f}",
                "daily_limit": f"${self.daily_limit:.2f}"
            }

        return {"allowed": True, "remaining_daily": self.daily_limit - self.daily_spend}
```

---

## 6. セルフホスト戦略

### 6.1 セルフホスト vs API のコスト比較

```python
class SelfHostAnalyzer:
    """セルフホスト vs API のコスト分析"""

    GPU_COSTS = {
        "a100_80gb": {
            "cloud_hourly": 3.00,  # AWS/GCP 1時間あたり
            "on_premise": 15000,   # 購入費用
            "power_monthly": 200,  # 電気代/月
            "throughput_tokens_per_second": 50000,
            "models_supported": ["llama-3-70b", "mixtral-8x7b"]
        },
        "a10g": {
            "cloud_hourly": 1.00,
            "on_premise": 3000,
            "power_monthly": 80,
            "throughput_tokens_per_second": 20000,
            "models_supported": ["llama-3-8b", "mistral-7b"]
        },
        "t4": {
            "cloud_hourly": 0.50,
            "on_premise": 2000,
            "power_monthly": 50,
            "throughput_tokens_per_second": 8000,
            "models_supported": ["mistral-7b"]
        }
    }

    def compare_costs(
        self,
        monthly_tokens: int,
        api_model: str = "gpt-4o",
        self_host_gpu: str = "a100_80gb",
        utilization: float = 0.70
    ) -> dict:
        """APIとセルフホストのコスト比較"""
        # API コスト
        api_pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00, "avg": 6.25},
            "claude-sonnet": {"input": 3.00, "output": 15.00, "avg": 9.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60, "avg": 0.375}
        }
        api_cost_per_1m = api_pricing.get(api_model, {}).get("avg", 5.0)
        monthly_api_cost = monthly_tokens / 1_000_000 * api_cost_per_1m

        # セルフホストコスト
        gpu = self.GPU_COSTS[self_host_gpu]
        tokens_per_month = gpu["throughput_tokens_per_second"] * 3600 * 24 * 30
        gpus_needed = max(1, int(
            monthly_tokens / (tokens_per_month * utilization)
        ))

        monthly_gpu_cost = gpu["cloud_hourly"] * 24 * 30 * gpus_needed
        monthly_infra_cost = 500  # 管理ツール、監視等
        monthly_self_host_cost = monthly_gpu_cost + monthly_infra_cost

        breakeven_tokens = monthly_self_host_cost / api_cost_per_1m * 1_000_000

        return {
            "monthly_tokens": f"{monthly_tokens:,}",
            "api_cost": {
                "model": api_model,
                "monthly": f"${monthly_api_cost:,.2f}",
                "per_1m_tokens": f"${api_cost_per_1m:.2f}"
            },
            "self_host_cost": {
                "gpu": self_host_gpu,
                "gpus_needed": gpus_needed,
                "monthly_gpu": f"${monthly_gpu_cost:,.2f}",
                "monthly_infra": f"${monthly_infra_cost:,.2f}",
                "monthly_total": f"${monthly_self_host_cost:,.2f}"
            },
            "comparison": {
                "cheaper_option": "self_host" if monthly_self_host_cost < monthly_api_cost else "api",
                "savings": f"${abs(monthly_api_cost - monthly_self_host_cost):,.2f}/月",
                "savings_pct": f"{abs(1 - monthly_self_host_cost/monthly_api_cost)*100:.0f}%"
                              if monthly_api_cost > 0 else "N/A",
                "breakeven_tokens": f"{breakeven_tokens:,.0f}トークン/月"
            },
            "recommendation": (
                f"月{breakeven_tokens/1_000_000:.0f}Mトークン以上なら"
                f"セルフホストが有利"
            )
        }
```

---

## 7. 総合コスト最適化戦略

### 7.1 最適化ロードマップ

```python
class CostOptimizationRoadmap:
    """コスト最適化ロードマップ"""

    PHASES = {
        "phase_1_quick_wins": {
            "timeline": "1-2週間",
            "expected_savings": "20-30%",
            "actions": [
                {
                    "action": "max_tokensの適切な設定",
                    "effort": "低",
                    "impact": "10-15%削減",
                    "detail": "タスク別にmax_tokensを制限"
                },
                {
                    "action": "プロンプトの簡潔化",
                    "effort": "低",
                    "impact": "10-20%削減",
                    "detail": "冗長な指示の除去"
                },
                {
                    "action": "基本的なモデル使い分け",
                    "effort": "低",
                    "impact": "20-40%削減",
                    "detail": "簡易タスクをmini/haikuに切り替え"
                }
            ]
        },
        "phase_2_caching": {
            "timeline": "2-4週間",
            "expected_savings": "30-50%（累積）",
            "actions": [
                {
                    "action": "完全一致キャッシュ（Redis）",
                    "effort": "中",
                    "impact": "15-25%削減",
                    "detail": "同一リクエストのキャッシュ"
                },
                {
                    "action": "セマンティックキャッシュ",
                    "effort": "中-高",
                    "impact": "10-20%追加削減",
                    "detail": "類似リクエストのキャッシュ"
                }
            ]
        },
        "phase_3_advanced": {
            "timeline": "1-3ヶ月",
            "expected_savings": "50-70%（累積）",
            "actions": [
                {
                    "action": "カスケードモデルルーティング",
                    "effort": "高",
                    "impact": "15-25%追加削減",
                    "detail": "品質評価 + 段階的モデル選択"
                },
                {
                    "action": "バッチ処理の全面導入",
                    "effort": "中",
                    "impact": "10-15%追加削減",
                    "detail": "非同期バッチ処理パイプライン"
                },
                {
                    "action": "予算管理自動化",
                    "effort": "中",
                    "impact": "予算超過防止",
                    "detail": "アラート + 回路遮断器"
                }
            ]
        },
        "phase_4_self_host": {
            "timeline": "3-6ヶ月",
            "expected_savings": "70-90%（累積）",
            "prerequisite": "月間API費用$5,000以上",
            "actions": [
                {
                    "action": "セルフホストLLMの導入",
                    "effort": "高",
                    "impact": "50-80%削減",
                    "detail": "Llama 3 / Mistralの運用"
                },
                {
                    "action": "ファインチューニング",
                    "effort": "高",
                    "impact": "品質維持+コスト削減",
                    "detail": "ドメイン特化モデルの構築"
                }
            ]
        }
    }

    def create_plan(self, current_monthly_cost: float,
                    target_reduction: float = 0.50) -> dict:
        """最適化計画を作成"""
        plan = []
        cumulative_savings = 0

        for phase_name, phase in self.PHASES.items():
            if cumulative_savings >= target_reduction:
                break

            # 位相の中の具体的節約見積もり
            phase_savings = float(
                phase["expected_savings"].split("-")[0].rstrip("%")
            ) / 100

            remaining_cost = current_monthly_cost * (1 - cumulative_savings)
            phase_dollar_savings = remaining_cost * phase_savings

            plan.append({
                "phase": phase_name,
                "timeline": phase["timeline"],
                "actions": [a["action"] for a in phase["actions"]],
                "estimated_savings": f"${phase_dollar_savings:,.0f}/月",
                "cumulative_savings": f"{(cumulative_savings + phase_savings)*100:.0f}%"
            })

            cumulative_savings += phase_savings

        return {
            "current_cost": f"${current_monthly_cost:,.0f}/月",
            "target_reduction": f"{target_reduction*100:.0f}%",
            "target_cost": f"${current_monthly_cost * (1-target_reduction):,.0f}/月",
            "plan": plan
        }
```

---

## 8. アンチパターン

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

### アンチパターン3: 入力テキストの無制限送信

```python
# BAD: ドキュメント全体を送信
def analyze_document(document):
    # 100ページのドキュメント全体を送信 → 100K tokens
    return call_ai(f"分析してください: {document}")

# GOOD: 前処理で必要部分のみ抽出
def analyze_document(document):
    # 1. チャンク分割
    chunks = split_into_chunks(document, max_tokens=2000)
    # 2. 関連チャンクのみ抽出
    relevant = find_relevant_chunks(chunks, query, top_k=3)
    # 3. 必要部分のみ送信 → 6K tokens（94%削減）
    return call_ai(f"分析: {' '.join(relevant)}")
```

### アンチパターン4: 予算監視なしの運用

```python
# BAD: 予算管理なし
def run_ai_service():
    while True:
        process_request()  # 上限なし、月末に請求書で驚く

# GOOD: 多層防御の予算管理
def run_ai_service():
    budget = BudgetManager(monthly_budget=5000)
    breaker = CostCircuitBreaker(daily_limit=200, hourly_limit=30)

    while True:
        # 回路遮断器チェック
        check = breaker.check_and_record(estimated_cost)
        if not check["allowed"]:
            return {"error": check["reason"]}

        # 予算チェック
        status = budget.check_budget(current_spend, day_of_month)
        if status["status"] == "critical":
            switch_to_cheap_model()

        process_request()
```

---

## 9. FAQ

### Q1: キャッシュのヒット率はどのくらいが目安？

**A:** 用途で大きく異なる。(1) カスタマーサポート（FAQ系）: 40-60%（同じ質問が繰り返される）、(2) コンテンツ生成: 10-20%（毎回異なる入力）、(3) 分類タスク: 30-50%。セマンティックキャッシュを導入すると完全一致の2-3倍のヒット率になる。目安として全体で30%以上を目指す。

### Q2: セルフホストLLMへの移行はいつすべき？

**A:** 3条件が揃ったとき。(1) 月間API費用が$5,000以上、(2) レイテンシ要件が厳しい（<100ms）、(3) データ主権の要件がある。GPU費用（A100: $2-3/時間）を考慮すると、月$3,000以下ならAPI利用の方が安い。Llama 3やMistralのファインチューニング版を使えば、GPT-4の80-90%の品質を1/10のコストで実現可能。

### Q3: 予算超過を防ぐ方法は？

**A:** 4層の防御を推奨。(1) ハード制限 — API使用量の上限をOpenAI/Anthropicの管理画面で設定、(2) アラート — 予算の50%/80%/100%でSlack通知、(3) ソフト制限 — ユーザー単位の日次/月次上限、(4) 回路遮断 — 異常な増加を検知したら自動停止。特にユーザー入力の長さ制限（max_tokens設定）が重要。

### Q4: プロンプト最適化だけでどのくらい節約できる？

**A:** 典型的には20-40%削減可能。(1) システムプロンプトの簡潔化: 15-25%（毎リクエストに影響）、(2) max_tokensの適切な設定: 10-15%（不要な出力を抑制）、(3) 出力フォーマットの指定: 5-10%（JSONなど構造化出力）。特にシステムプロンプトは全リクエストに含まれるため、最初に最適化すべき。

### Q5: 複数プロバイダー間の最適化はどう行う？

**A:** 3つのアプローチ。(1) コスト別ルーティング — タスク種別ごとに最安プロバイダーを選択、(2) フォールバック — 障害時に自動切り替え（レイテンシ低下を防止）、(3) A/Bテスト — 品質とコストのバランスを継続的に検証。LiteLLMのようなマルチプロバイダーライブラリを使うと実装が容易。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| コスト構造 | APIコストが全体の20-40%、入力/出力トークンで課金 |
| キャッシュ | 3層（完全一致→セマンティック→API）で30-50%削減 |
| モデル選択 | タスク複雑度でルーティング、40-70%削減可能 |
| カスケード | 軽量→高品質の段階的処理で追加15-25%削減 |
| プロンプト | 簡潔化で20-40%削減、バッチ処理で更に削減 |
| 予算管理 | アラート + 回路遮断器 + ユーザー制限の多層防御 |
| セルフホスト | 月$5,000以上なら検討、50-80%削減可能 |
| 監視 | 日次レポート + 異常検知 + 自動停止 |
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
5. **pgvector Documentation** — https://github.com/pgvector/pgvector — PostgreSQLベクトル検索
6. **LiteLLM** — https://github.com/BerriAI/litellm — マルチプロバイダーLLMライブラリ
7. **vLLM** — https://github.com/vllm-project/vllm — 高効率LLM推論エンジン

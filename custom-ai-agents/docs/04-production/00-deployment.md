# デプロイ

> スケーリング・モニタリング・可用性――AIエージェントを本番環境にデプロイし、安定的に運用するためのアーキテクチャ設計と運用プラクティス。

## この章で学ぶこと

1. エージェントの本番アーキテクチャとスケーリング戦略
2. モニタリング・ログ・アラートの設計と実装パターン
3. コスト管理・レート制限対策・障害回復の運用プラクティス

---

## 1. 本番アーキテクチャ

### 1.1 全体構成

```
エージェント本番アーキテクチャ

+-------------------------------------------------------------------+
|                         Client Layer                               |
|  [Web App] [Mobile App] [CLI] [API Client]                        |
+-------------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------------+
|                        API Gateway                                 |
|  [Rate Limiting] [Auth] [Load Balancer] [Request Routing]          |
+-------------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------------+
|                     Agent Service Layer                             |
|  +-------------------+  +-------------------+  +----------------+ |
|  | Agent Instance 1  |  | Agent Instance 2  |  | Agent Inst. N  | |
|  | [LLM Client]      |  | [LLM Client]      |  | [LLM Client]  | |
|  | [Tool Executor]   |  | [Tool Executor]   |  | [Tool Exec.]   | |
|  | [Memory Manager]  |  | [Memory Manager]  |  | [Memory Mgr.]  | |
|  +-------------------+  +-------------------+  +----------------+ |
+-------------------------------------------------------------------+
                    |              |              |
                    v              v              v
+-------------------------------------------------------------------+
|                     Infrastructure Layer                            |
|  +--------+  +----------+  +---------+  +----------+              |
|  | LLM    |  | Vector   |  | Cache   |  | Message  |              |
|  | APIs   |  | DB       |  | (Redis) |  | Queue    |              |
|  +--------+  +----------+  +---------+  +----------+              |
|  +--------+  +----------+  +---------+                             |
|  | SQL DB |  | Object   |  | MCP     |                             |
|  |        |  | Storage  |  | Servers |                             |
|  +--------+  +----------+  +---------+                             |
+-------------------------------------------------------------------+
```

### 1.2 デプロイパターン

```
デプロイパターンの選択肢

1. サーバーレス（Lambda / Cloud Functions）
   [リクエスト] → [API Gateway] → [Lambda] → [LLM API]
   + スケール自動、コスト従量
   - タイムアウト制限（通常15分）、コールドスタート

2. コンテナ（ECS / Cloud Run / Kubernetes）
   [リクエスト] → [ALB] → [ECS Fargate Container]
   + 柔軟、長時間実行可能
   - インフラ管理が必要

3. キューベース（非同期）
   [リクエスト] → [API] → [SQS/Redis] → [Worker] → [Callback]
   + 長時間タスクに適す、バックプレッシャー制御
   - リアルタイム応答が困難
```

---

## 2. スケーリング

### 2.1 スケーリング戦略

```python
# 非同期キューベースのスケーリング
import asyncio
from dataclasses import dataclass
import aiohttp

@dataclass
class AgentTask:
    task_id: str
    user_id: str
    input_message: str
    priority: int = 0

class AgentWorkerPool:
    def __init__(self, num_workers: int = 10, max_concurrent: int = 50):
        self.queue = asyncio.PriorityQueue()
        self.num_workers = num_workers
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks = {}

    async def submit(self, task: AgentTask) -> str:
        """タスクをキューに投入"""
        await self.queue.put((task.priority, task))
        return task.task_id

    async def worker(self, worker_id: int):
        """ワーカーループ"""
        while True:
            _, task = await self.queue.get()
            async with self.semaphore:
                try:
                    self.active_tasks[task.task_id] = "running"
                    result = await self._process_task(task)
                    self.active_tasks[task.task_id] = "completed"
                    await self._notify_completion(task, result)
                except Exception as e:
                    self.active_tasks[task.task_id] = "failed"
                    await self._handle_failure(task, e)
                finally:
                    self.queue.task_done()

    async def start(self):
        """ワーカープールを起動"""
        workers = [
            asyncio.create_task(self.worker(i))
            for i in range(self.num_workers)
        ]
        await asyncio.gather(*workers)
```

### 2.2 水平スケーリング設計

```
水平スケーリング

                    +---> [Worker 1] --+
                    |                  |
[Queue] --dispatch--+---> [Worker 2] --+---> [Result Store]
                    |                  |
                    +---> [Worker N] --+

スケーリングポリシー:
- Queue深さ > 100 → ワーカー追加
- Queue深さ < 10  → ワーカー削減
- CPU使用率 > 70% → ワーカー追加
- LLM APIレート制限 → リクエスト間隔調整
```

---

## 3. モニタリング

### 3.1 メトリクス設計

```python
# エージェントメトリクスの収集
import time
from prometheus_client import Counter, Histogram, Gauge

# メトリクス定義
AGENT_REQUESTS = Counter(
    "agent_requests_total",
    "Total agent requests",
    ["status", "intent"]
)
AGENT_LATENCY = Histogram(
    "agent_latency_seconds",
    "Agent response latency",
    buckets=[1, 5, 10, 30, 60, 120, 300]
)
AGENT_STEPS = Histogram(
    "agent_steps_count",
    "Number of agent steps per task",
    buckets=[1, 3, 5, 10, 15, 20, 30]
)
AGENT_COST = Counter(
    "agent_cost_usd",
    "Total API cost in USD"
)
ACTIVE_AGENTS = Gauge(
    "active_agents",
    "Currently running agent instances"
)
TOOL_CALLS = Counter(
    "agent_tool_calls_total",
    "Total tool calls",
    ["tool_name", "status"]
)

class MonitoredAgent:
    def run(self, task: str) -> str:
        ACTIVE_AGENTS.inc()
        start = time.time()

        try:
            result = self._agent_loop(task)
            AGENT_REQUESTS.labels(status="success", intent="general").inc()
            return result
        except Exception as e:
            AGENT_REQUESTS.labels(status="error", intent="general").inc()
            raise
        finally:
            AGENT_LATENCY.observe(time.time() - start)
            AGENT_STEPS.observe(self.step_count)
            AGENT_COST.inc(self.cost_tracker.total_cost)
            ACTIVE_AGENTS.dec()
```

### 3.2 ログ設計

```python
# 構造化ログの実装
import structlog
import json

logger = structlog.get_logger()

class LoggedAgent:
    def run(self, task: str, request_id: str) -> str:
        log = logger.bind(request_id=request_id, task=task[:100])

        log.info("agent_started")

        for step in range(self.max_steps):
            # LLM呼び出し
            log.info("llm_call", step=step, model=self.model)
            response = self._call_llm()

            # ツール実行
            for tool_call in response.tool_calls:
                log.info("tool_call",
                    step=step,
                    tool=tool_call.name,
                    input_preview=str(tool_call.input)[:200]
                )
                result = self._execute_tool(tool_call)
                log.info("tool_result",
                    step=step,
                    tool=tool_call.name,
                    result_preview=str(result)[:200],
                    success=not result.startswith("Error")
                )

        log.info("agent_completed",
            total_steps=step,
            total_tokens=self.token_count,
            cost_usd=self.cost
        )
```

### 3.3 ダッシュボード設計

```
エージェント運用ダッシュボード

+-------------------------------------------------------------------+
|  [成功率: 94.2%] [平均レイテンシ: 8.3s] [本日のコスト: $127.50]    |
+-------------------------------------------------------------------+
|                                                                     |
|  リクエスト数 (過去24h)        レイテンシ分布                        |
|  200|    *                     |   *                                |
|     |   * *     *              |  * *                               |
|  100|  *   *   * *    *        | *   *  *                           |
|     | *     * *   *  * *       |*     **  *                         |
|    0+--+--+--+--+--+--+--     +--+--+--+--+--                      |
|     0  4  8  12 16 20 24      0  5  10 30 60s                      |
|                                                                     |
+-------------------------------------------------------------------+
|  エラー率          ツール使用頻度       コスト推移                    |
|  5%|               [search: 40%]       $150|  *                     |
|  3%|   *           [read:   30%]       $100|*  *                    |
|  1%| *   *         [write:  20%]        $50|     *  *               |
|  0%+------         [exec:   10%]         $0+--------                |
+-------------------------------------------------------------------+
```

---

## 4. コスト管理

```python
# コスト制御の実装
class CostController:
    def __init__(self, daily_budget: float = 100.0,
                 per_task_limit: float = 5.0):
        self.daily_budget = daily_budget
        self.per_task_limit = per_task_limit
        self.daily_spend = 0.0
        self.task_spend = 0.0

    def check_budget(self) -> bool:
        """予算内かチェック"""
        if self.daily_spend >= self.daily_budget:
            raise BudgetExceededError("日次予算上限に達しました")
        if self.task_spend >= self.per_task_limit:
            raise BudgetExceededError("タスク予算上限に達しました")
        return True

    def track_usage(self, input_tokens: int, output_tokens: int,
                     model: str):
        """使用量を追跡"""
        cost = self._calculate_cost(input_tokens, output_tokens, model)
        self.daily_spend += cost
        self.task_spend += cost

    def _calculate_cost(self, input_tokens, output_tokens, model) -> float:
        rates = {
            "claude-sonnet-4-20250514": (3.0, 15.0),  # input, output per 1M
            "claude-haiku-4-20250514": (0.25, 1.25),
        }
        input_rate, output_rate = rates.get(model, (3.0, 15.0))
        return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
```

---

## 5. 比較表

### 5.1 デプロイメント方式比較

| 方式 | スケール | コスト | 長時間タスク | 運用負荷 |
|------|---------|--------|------------|---------|
| サーバーレス | 自動 | 従量制 | 制限あり | 低 |
| コンテナ(Fargate) | 半自動 | 中 | 対応 | 中 |
| Kubernetes | 手動/自動 | 中-高 | 対応 | 高 |
| VM | 手動 | 固定 | 対応 | 最高 |
| マネージドサービス | 自動 | 高 | 対応 | 最低 |

### 5.2 モニタリングツール比較

| ツール | メトリクス | ログ | トレース | コスト |
|--------|----------|------|---------|--------|
| LangSmith | エージェント特化 | あり | あり | 有料 |
| Datadog | 汎用 | あり | あり | 高額 |
| Grafana + Prometheus | 汎用 | Loki | Tempo | 無料/有料 |
| CloudWatch | AWS特化 | あり | X-Ray | 従量制 |
| Helicone | LLM特化 | あり | なし | フリーミアム |

---

## 6. 障害回復

```
障害回復戦略

1. リトライ (Retry)
   [失敗] → [待機 1s] → [再試行] → [待機 2s] → [再試行] → [成功 or 諦め]
   指数バックオフ + ジッター

2. フォールバック (Fallback)
   [Claude API 障害] → [OpenAI API にフォールバック]
   [Sonnet 障害] → [Haiku にダウングレード]

3. サーキットブレーカー
   [正常] → エラー率>50% → [オープン (全拒否)] → 30秒後 → [半開 (一部許可)] → 成功 → [正常]
```

```python
# サーキットブレーカーの実装
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"      # 正常
    OPEN = "open"          # 全拒否
    HALF_OPEN = "half_open" # 一部許可

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = 0

    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError("サーキットブレーカーがオープン状態")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

---

## 7. アンチパターン

### アンチパターン1: 同期的な長時間処理

```python
# NG: HTTPリクエストでエージェントを同期実行
@app.post("/agent/run")
def run_agent(request):
    result = agent.run(request.task)  # 5分かかる → タイムアウト
    return {"result": result}

# OK: 非同期キューベース
@app.post("/agent/run")
async def run_agent(request):
    task_id = await queue.submit(request.task)
    return {"task_id": task_id, "status_url": f"/agent/status/{task_id}"}

@app.get("/agent/status/{task_id}")
async def get_status(task_id: str):
    status = await queue.get_status(task_id)
    return status  # {"status": "running", "progress": 60}
```

### アンチパターン2: ログなしの本番運用

```python
# NG: print文でデバッグ
print(f"Processing: {task}")
print(f"Result: {result}")

# OK: 構造化ログ + メトリクス + トレース
logger.info("agent_step", extra={
    "request_id": request_id,
    "step": step_num,
    "tool": tool_name,
    "latency_ms": latency,
    "tokens": token_count
})
metrics.record_step(step_num, latency, token_count)
```

---

## 8. FAQ

### Q1: LLM APIの可用性にどう対処する？

- **マルチプロバイダ**: Claude + OpenAI のフォールバック構成
- **リトライ**: 指数バックオフ（1s, 2s, 4s, 8s）
- **サーキットブレーカー**: 連続失敗時にAPIへの呼び出しを一時停止
- **キャッシュ**: 同じ入力に対するレスポンスをキャッシュ

### Q2: エージェントのバージョン管理は？

- **プロンプトのバージョン管理**: Git管理 + A/Bテスト
- **ツールのバージョン管理**: セマンティックバージョニング
- **モデルバージョン**: モデルIDを固定（snapshot使用）
- **ロールバック**: 問題発生時に前バージョンに即時切り替え

### Q3: マルチリージョン展開は必要か？

レイテンシ要件による:
- **<1秒**: リージョン近接が重要（LLM APIのリージョン選択）
- **<10秒**: シングルリージョンでも十分
- **長時間タスク**: リージョンよりも可用性が重要

---

## まとめ

| 項目 | 内容 |
|------|------|
| アーキテクチャ | API Gateway + Agent Service + Infrastructure |
| スケーリング | キューベースの水平スケーリング |
| モニタリング | メトリクス + 構造化ログ + トレース |
| コスト管理 | 日次/タスク単位の予算制限 |
| 障害回復 | リトライ + フォールバック + サーキットブレーカー |
| 原則 | 非同期優先、ログ必須、段階的ロールアウト |

## 次に読むべきガイド

- [01-safety.md](./01-safety.md) — 本番環境での安全性確保
- [../02-implementation/04-evaluation.md](../02-implementation/04-evaluation.md) — 本番メトリクスの評価
- [../02-implementation/02-mcp-agents.md](../02-implementation/02-mcp-agents.md) — MCPサーバーのデプロイ

## 参考文献

1. Anthropic, "API rate limits" — https://docs.anthropic.com/en/api/rate-limits
2. LangSmith Documentation — https://docs.smith.langchain.com/
3. AWS, "Serverless patterns for AI/ML workloads" — https://aws.amazon.com/serverless/

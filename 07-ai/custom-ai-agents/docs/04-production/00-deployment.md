# デプロイ

> スケーリング・モニタリング・可用性――AIエージェントを本番環境にデプロイし、安定的に運用するためのアーキテクチャ設計と運用プラクティス。

## この章で学ぶこと

1. エージェントの本番アーキテクチャとスケーリング戦略
2. モニタリング・ログ・アラートの設計と実装パターン
3. コスト管理・レート制限対策・障害回復の運用プラクティス
4. CI/CD パイプラインによる安全なデプロイフロー
5. インフラストラクチャ・アズ・コードによる環境管理
6. カナリアデプロイ・ブルーグリーンデプロイの実践

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

### 1.3 コンテナベースデプロイの実装例

```python
# Dockerfile for Agent Service
"""
FROM python:3.12-slim

WORKDIR /app

# 依存関係のインストール（キャッシュ活用）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコード
COPY src/ ./src/
COPY configs/ ./configs/

# セキュリティ: 非rootユーザー
RUN useradd --create-home appuser
USER appuser

# ヘルスチェックエンドポイント
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
"""
```

```python
# docker-compose.yml（ローカル開発 + ステージング用）
"""
version: '3.9'
services:
  agent-api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@postgres:5432/agents
      - LOG_LEVEL=INFO
      - MAX_CONCURRENT_AGENTS=10
    depends_on:
      - redis
      - postgres
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
    restart: unless-stopped

  agent-worker:
    build: .
    command: ["python", "-m", "src.worker"]
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@postgres:5432/agents
      - WORKER_CONCURRENCY=5
    depends_on:
      - redis
      - postgres
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: agents
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - pg-data:/var/lib/postgresql/data

volumes:
  redis-data:
  pg-data:
"""
```

### 1.4 Kubernetes マニフェスト

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-api
  labels:
    app: agent-api
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-api
  template:
    metadata:
      labels:
        app: agent-api
        version: v1
    spec:
      containers:
        - name: agent-api
          image: your-registry/agent-api:latest
          ports:
            - containerPort: 8080
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2000m"
          env:
            - name: ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: agent-secrets
                  key: anthropic-api-key
            - name: REDIS_URL
              value: "redis://redis-service:6379"
            - name: MAX_CONCURRENT_AGENTS
              value: "10"
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
          startupProbe:
            httpGet:
              path: /health
              port: 8080
            failureThreshold: 30
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: agent-api-service
spec:
  selector:
    app: agent-api
  ports:
    - port: 80
      targetPort: 8080
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: agent_queue_depth
        target:
          type: AverageValue
          averageValue: "50"
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

### 2.3 レート制限対応のスケーリング

```python
import asyncio
import time
from collections import deque

class AdaptiveRateLimiter:
    """LLM APIのレート制限に適応するリミッター"""

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100_000,
        max_retries: int = 5
    ):
        self.rpm_limit = requests_per_minute
        self.tpm_limit = tokens_per_minute
        self.request_timestamps = deque()
        self.token_usage = deque()
        self.max_retries = max_retries
        self._lock = asyncio.Lock()
        self._backoff_until = 0.0

    async def acquire(self, estimated_tokens: int = 1000):
        """レート制限を考慮してリクエスト権を取得"""
        async with self._lock:
            now = time.time()

            # バックオフ中か確認
            if now < self._backoff_until:
                wait_time = self._backoff_until - now
                await asyncio.sleep(wait_time)
                now = time.time()

            # 1分以上前のエントリを削除
            cutoff = now - 60
            while self.request_timestamps and self.request_timestamps[0] < cutoff:
                self.request_timestamps.popleft()
            while self.token_usage and self.token_usage[0][0] < cutoff:
                self.token_usage.popleft()

            # RPMチェック
            if len(self.request_timestamps) >= self.rpm_limit:
                wait_time = 60 - (now - self.request_timestamps[0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            # TPMチェック
            current_tokens = sum(t[1] for t in self.token_usage)
            if current_tokens + estimated_tokens > self.tpm_limit:
                wait_time = 60 - (now - self.token_usage[0][0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            self.request_timestamps.append(time.time())
            self.token_usage.append((time.time(), estimated_tokens))

    def report_rate_limit(self, retry_after: float = 60.0):
        """429エラー時にバックオフ期間を設定"""
        self._backoff_until = time.time() + retry_after

    async def call_with_retry(self, func, *args, **kwargs):
        """リトライ付きのAPI呼び出し"""
        for attempt in range(self.max_retries):
            try:
                await self.acquire()
                result = await func(*args, **kwargs)
                return result
            except RateLimitError as e:
                retry_after = getattr(e, "retry_after", 2 ** attempt)
                self.report_rate_limit(retry_after)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(retry_after)
                else:
                    raise
            except Exception:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
```

### 2.4 マルチプロバイダロードバランシング

```python
import random
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LLMProvider:
    name: str
    client: object
    model: str
    weight: float = 1.0
    is_healthy: bool = True
    error_count: int = 0
    max_errors: int = 5
    cooldown_until: float = 0.0

class MultiProviderBalancer:
    """複数LLMプロバイダ間のロードバランシング"""

    def __init__(self):
        self.providers: list[LLMProvider] = []
        self.primary_index: int = 0

    def add_provider(self, provider: LLMProvider):
        self.providers.append(provider)

    def _get_available_providers(self) -> list[LLMProvider]:
        """利用可能なプロバイダを取得"""
        now = time.time()
        available = []
        for p in self.providers:
            if p.cooldown_until > 0 and now > p.cooldown_until:
                p.is_healthy = True
                p.error_count = 0
                p.cooldown_until = 0.0
            if p.is_healthy:
                available.append(p)
        return available

    def select_provider(self, strategy: str = "weighted") -> LLMProvider:
        """プロバイダを選択"""
        available = self._get_available_providers()
        if not available:
            raise AllProvidersUnavailableError(
                "全てのLLMプロバイダが利用不可です"
            )

        if strategy == "weighted":
            weights = [p.weight for p in available]
            return random.choices(available, weights=weights, k=1)[0]
        elif strategy == "round_robin":
            provider = available[self.primary_index % len(available)]
            self.primary_index += 1
            return provider
        elif strategy == "failover":
            return available[0]
        else:
            return random.choice(available)

    def report_error(self, provider: LLMProvider):
        """エラーを報告し、必要に応じてプロバイダを無効化"""
        provider.error_count += 1
        if provider.error_count >= provider.max_errors:
            provider.is_healthy = False
            provider.cooldown_until = time.time() + 300  # 5分間クールダウン

    def report_success(self, provider: LLMProvider):
        """成功を報告してエラーカウントをリセット"""
        provider.error_count = 0

    async def call(self, messages: list, **kwargs) -> str:
        """フォールバック付きのLLM呼び出し"""
        errors = []
        tried_providers = set()

        for _ in range(len(self.providers)):
            try:
                provider = self.select_provider()
                if provider.name in tried_providers:
                    continue
                tried_providers.add(provider.name)

                result = await provider.client.messages.create(
                    model=provider.model,
                    messages=messages,
                    **kwargs
                )
                self.report_success(provider)
                return result
            except Exception as e:
                self.report_error(provider)
                errors.append((provider.name, str(e)))

        raise AllProvidersFailedError(
            f"全プロバイダが失敗: {errors}"
        )
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

### 3.3 分散トレーシングの実装

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# トレーサーの設定
resource = Resource.create({"service.name": "agent-service"})
provider = TracerProvider(resource=resource)
exporter = OTLPSpanExporter(endpoint="http://jaeger:4317")
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("agent-service")

class TracedAgent:
    """OpenTelemetry対応のエージェント"""

    async def run(self, task: str, request_id: str) -> str:
        with tracer.start_as_current_span(
            "agent_run",
            attributes={
                "agent.task": task[:200],
                "agent.request_id": request_id,
                "agent.model": self.model,
            }
        ) as root_span:
            try:
                result = await self._agent_loop(task, root_span)
                root_span.set_attribute("agent.status", "success")
                root_span.set_attribute("agent.total_steps", self.step_count)
                root_span.set_attribute("agent.total_cost", self.cost)
                return result
            except Exception as e:
                root_span.set_attribute("agent.status", "error")
                root_span.record_exception(e)
                raise

    async def _agent_loop(self, task: str, parent_span) -> str:
        messages = [{"role": "user", "content": task}]

        for step in range(self.max_steps):
            # LLM呼び出しのスパン
            with tracer.start_as_current_span(
                "llm_call",
                attributes={
                    "llm.step": step,
                    "llm.model": self.model,
                    "llm.input_tokens": len(str(messages)),
                }
            ) as llm_span:
                response = await self._call_llm(messages)
                llm_span.set_attribute(
                    "llm.output_tokens",
                    response.usage.output_tokens
                )

            # ツール呼び出しのスパン
            for tool_call in response.tool_calls:
                with tracer.start_as_current_span(
                    f"tool_{tool_call.name}",
                    attributes={
                        "tool.name": tool_call.name,
                        "tool.step": step,
                    }
                ) as tool_span:
                    start = time.time()
                    result = await self._execute_tool(tool_call)
                    tool_span.set_attribute(
                        "tool.duration_ms",
                        (time.time() - start) * 1000
                    )
                    tool_span.set_attribute(
                        "tool.success",
                        not str(result).startswith("Error")
                    )

            if response.stop_reason == "end_turn":
                return response.content

        return "最大ステップ数に到達"
```

### 3.4 ダッシュボード設計

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

### 3.5 Grafana ダッシュボード定義

```json
{
  "dashboard": {
    "title": "AI Agent Operations",
    "panels": [
      {
        "title": "Request Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(agent_requests_total[5m])",
            "legendFormat": "{{status}}"
          }
        ]
      },
      {
        "title": "P50/P95/P99 Latency",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(agent_latency_seconds_bucket[5m]))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(agent_latency_seconds_bucket[5m]))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(agent_latency_seconds_bucket[5m]))",
            "legendFormat": "P99"
          }
        ]
      },
      {
        "title": "Active Agents",
        "type": "gauge",
        "targets": [
          {
            "expr": "active_agents",
            "legendFormat": "Active"
          }
        ]
      },
      {
        "title": "Hourly Cost",
        "type": "timeseries",
        "targets": [
          {
            "expr": "increase(agent_cost_usd[1h])",
            "legendFormat": "Cost USD"
          }
        ]
      },
      {
        "title": "Tool Call Distribution",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (tool_name) (agent_tool_calls_total)",
            "legendFormat": "{{tool_name}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(agent_requests_total{status='error'}[5m])) / sum(rate(agent_requests_total[5m])) * 100",
            "legendFormat": "Error %"
          }
        ]
      }
    ]
  }
}
```

### 3.6 アラートルール設計

```yaml
# prometheus-alerts.yaml
groups:
  - name: agent_alerts
    rules:
      # 高エラー率アラート
      - alert: AgentHighErrorRate
        expr: |
          sum(rate(agent_requests_total{status="error"}[5m]))
          / sum(rate(agent_requests_total[5m])) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "エージェントのエラー率が10%を超えています"
          description: "過去5分間のエラー率: {{ $value | humanizePercentage }}"

      # 高レイテンシアラート
      - alert: AgentHighLatency
        expr: |
          histogram_quantile(0.95, rate(agent_latency_seconds_bucket[5m])) > 60
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "P95レイテンシが60秒を超えています"

      # コスト超過アラート
      - alert: AgentCostExceeded
        expr: increase(agent_cost_usd[1h]) > 50
        for: 0m
        labels:
          severity: warning
        annotations:
          summary: "1時間あたりのコストが$50を超えました"
          description: "直近1時間のコスト: ${{ $value }}"

      # キュー深さアラート
      - alert: AgentQueueDepthHigh
        expr: agent_queue_depth > 500
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "エージェントタスクキューが深くなっています"
          description: "キュー深さ: {{ $value }}"

      # 全プロバイダ障害
      - alert: AllLLMProvidersDown
        expr: sum(llm_provider_healthy) == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "全てのLLMプロバイダが利用不可です"
```

---

## 4. コスト管理

### 4.1 コスト制御の実装

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

### 4.2 高度なコスト最適化戦略

```python
import hashlib
import json
from datetime import datetime, timedelta

class CostOptimizer:
    """コスト最適化のための包括的な戦略実装"""

    def __init__(self, redis_client, db_client):
        self.redis = redis_client
        self.db = db_client

    # --- キャッシュ戦略 ---

    async def cached_llm_call(
        self,
        messages: list,
        model: str,
        ttl: int = 3600
    ) -> dict:
        """同一入力に対するLLMレスポンスをキャッシュ"""
        cache_key = self._make_cache_key(messages, model)
        cached = await self.redis.get(cache_key)

        if cached:
            return json.loads(cached)

        result = await self._call_llm(messages, model)
        await self.redis.setex(
            cache_key,
            ttl,
            json.dumps(result)
        )
        return result

    def _make_cache_key(self, messages: list, model: str) -> str:
        content = json.dumps({"messages": messages, "model": model},
                            sort_keys=True)
        return f"llm_cache:{hashlib.sha256(content.encode()).hexdigest()}"

    # --- モデルルーティング ---

    def select_model(self, task_complexity: str, budget_remaining: float) -> str:
        """タスクの複雑さと残り予算に応じてモデルを選択"""
        if budget_remaining < 1.0:
            return "claude-haiku-4-20250514"  # 最安モデル

        model_selection = {
            "simple": "claude-haiku-4-20250514",      # 分類、要約
            "medium": "claude-sonnet-4-20250514",      # 一般的なタスク
            "complex": "claude-opus-4-20250514",       # 高度な推論
        }
        return model_selection.get(task_complexity, "claude-sonnet-4-20250514")

    # --- プロンプト最適化 ---

    def optimize_prompt(self, messages: list, max_input_tokens: int = 4000) -> list:
        """プロンプトを最適化してトークン使用量を削減"""
        optimized = []
        total_tokens_est = 0

        for msg in messages:
            token_est = len(msg["content"]) // 4  # 概算
            if total_tokens_est + token_est > max_input_tokens:
                # 古いメッセージを要約
                summary = self._summarize_message(msg["content"])
                optimized.append({
                    "role": msg["role"],
                    "content": summary
                })
                total_tokens_est += len(summary) // 4
            else:
                optimized.append(msg)
                total_tokens_est += token_est

        return optimized

    # --- コストレポート ---

    async def generate_cost_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> dict:
        """期間指定のコストレポートを生成"""
        records = await self.db.fetch_usage(start_date, end_date)

        report = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "total_cost": sum(r["cost"] for r in records),
            "total_requests": len(records),
            "by_model": {},
            "by_user": {},
            "by_hour": {},
            "top_expensive_tasks": [],
        }

        for record in records:
            # モデル別集計
            model = record["model"]
            if model not in report["by_model"]:
                report["by_model"][model] = {
                    "cost": 0, "requests": 0, "tokens": 0
                }
            report["by_model"][model]["cost"] += record["cost"]
            report["by_model"][model]["requests"] += 1
            report["by_model"][model]["tokens"] += record["total_tokens"]

            # ユーザー別集計
            user = record.get("user_id", "unknown")
            if user not in report["by_user"]:
                report["by_user"][user] = {"cost": 0, "requests": 0}
            report["by_user"][user]["cost"] += record["cost"]
            report["by_user"][user]["requests"] += 1

        # 高コストタスクの上位10件
        sorted_records = sorted(records, key=lambda r: r["cost"], reverse=True)
        report["top_expensive_tasks"] = [
            {
                "task_id": r["task_id"],
                "cost": r["cost"],
                "model": r["model"],
                "tokens": r["total_tokens"],
            }
            for r in sorted_records[:10]
        ]

        return report
```

---

## 5. CI/CD パイプライン

### 5.1 GitHub Actions によるデプロイ自動化

```yaml
# .github/workflows/deploy-agent.yaml
name: Deploy Agent Service

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  AWS_REGION: ap-northeast-1
  ECR_REPOSITORY: agent-service
  ECS_CLUSTER: agent-cluster
  ECS_SERVICE: agent-api

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install -r requirements-dev.txt

      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=src --cov-report=xml

      - name: Run integration tests
        run: pytest tests/integration/ -v
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY_TEST }}

      - name: Run agent evaluation suite
        run: python -m src.eval.run_evaluation --suite=smoke
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY_TEST }}

      - name: Check evaluation results
        run: |
          python -m src.eval.check_results \
            --min-accuracy=0.85 \
            --max-cost=2.0 \
            --max-latency=30

  build-and-push:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    outputs:
      image: ${{ steps.build.outputs.image }}
    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - uses: aws-actions/amazon-ecr-login@v2
        id: login-ecr

      - name: Build, tag, and push image
        id: build
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          echo "image=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to staging
        run: |
          aws ecs update-service \
            --cluster $ECS_CLUSTER-staging \
            --service $ECS_SERVICE \
            --force-new-deployment \
            --task-definition agent-api-staging

      - name: Wait for deployment
        run: |
          aws ecs wait services-stable \
            --cluster $ECS_CLUSTER-staging \
            --services $ECS_SERVICE

      - name: Run smoke tests against staging
        run: |
          python -m src.eval.smoke_test \
            --endpoint=https://staging-agent.example.com

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy canary (10%)
        run: |
          aws ecs update-service \
            --cluster $ECS_CLUSTER \
            --service $ECS_SERVICE-canary \
            --force-new-deployment

      - name: Monitor canary (5 minutes)
        run: |
          python -m src.deploy.monitor_canary \
            --duration=300 \
            --max-error-rate=0.05

      - name: Full rollout
        run: |
          aws ecs update-service \
            --cluster $ECS_CLUSTER \
            --service $ECS_SERVICE \
            --force-new-deployment
```

### 5.2 Terraform によるインフラ管理

```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    bucket = "agent-terraform-state"
    key    = "agent-service/terraform.tfstate"
    region = "ap-northeast-1"
  }
}

# ECS クラスター
resource "aws_ecs_cluster" "agent" {
  name = "agent-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# タスク定義
resource "aws_ecs_task_definition" "agent_api" {
  family                   = "agent-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 1024
  memory                   = 2048
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name  = "agent-api"
      image = "${aws_ecr_repository.agent.repository_url}:latest"
      portMappings = [{
        containerPort = 8080
        protocol      = "tcp"
      }]
      environment = [
        { name = "MAX_CONCURRENT_AGENTS", value = "10" },
        { name = "LOG_LEVEL", value = "INFO" },
      ]
      secrets = [
        {
          name      = "ANTHROPIC_API_KEY"
          valueFrom = aws_secretsmanager_secret.anthropic_key.arn
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/agent-api"
          "awslogs-region"        = "ap-northeast-1"
          "awslogs-stream-prefix" = "agent"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

# サービス
resource "aws_ecs_service" "agent_api" {
  name            = "agent-api"
  cluster         = aws_ecs_cluster.agent.id
  task_definition = aws_ecs_task_definition.agent_api.arn
  desired_count   = 3
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = var.private_subnet_ids
    security_groups = [aws_security_group.agent_api.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.agent_api.arn
    container_name   = "agent-api"
    container_port   = 8080
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
}

# Auto Scaling
resource "aws_appautoscaling_target" "agent_api" {
  max_capacity       = 20
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.agent.name}/${aws_ecs_service.agent_api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "agent_cpu" {
  name               = "agent-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.agent_api.resource_id
  scalable_dimension = aws_appautoscaling_target.agent_api.scalable_dimension
  service_namespace  = aws_appautoscaling_target.agent_api.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value       = 70.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}
```

---

## 6. 比較表

### 6.1 デプロイメント方式比較

| 方式 | スケール | コスト | 長時間タスク | 運用負荷 | 適用場面 |
|------|---------|--------|------------|---------|---------|
| サーバーレス | 自動 | 従量制 | 制限あり | 低 | 低頻度・短時間タスク |
| コンテナ(Fargate) | 半自動 | 中 | 対応 | 中 | 中規模プロダクション |
| Kubernetes | 手動/自動 | 中-高 | 対応 | 高 | 大規模・複雑なワークロード |
| VM | 手動 | 固定 | 対応 | 最高 | GPU利用・特殊要件 |
| マネージドサービス | 自動 | 高 | 対応 | 最低 | プロトタイプ・小規模 |

### 6.2 モニタリングツール比較

| ツール | メトリクス | ログ | トレース | コスト | エージェント対応 |
|--------|----------|------|---------|--------|---------------|
| LangSmith | エージェント特化 | あり | あり | 有料 | 最適 |
| Datadog | 汎用 | あり | あり | 高額 | カスタム構成 |
| Grafana + Prometheus | 汎用 | Loki | Tempo | 無料/有料 | カスタム構成 |
| CloudWatch | AWS特化 | あり | X-Ray | 従量制 | カスタム構成 |
| Helicone | LLM特化 | あり | なし | フリーミアム | 良好 |
| Langfuse | エージェント対応 | あり | あり | 無料/有料 | 良好 |

### 6.3 CI/CDツール比較

| ツール | 環境 | エージェントテスト | デプロイ自動化 | コスト |
|--------|------|-----------------|--------------|--------|
| GitHub Actions | GitHub | pytest + eval | ECS/K8s対応 | 無料枠あり |
| GitLab CI | GitLab | pytest + eval | K8s統合 | 無料枠あり |
| CircleCI | 汎用 | pytest + eval | 汎用 | 有料 |
| AWS CodePipeline | AWS | CodeBuild | ECS/Lambda | 従量制 |
| ArgoCD | K8s | なし | GitOps | 無料 |

---

## 7. 障害回復

### 7.1 障害回復戦略の概要

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

### 7.2 サーキットブレーカーの実装

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

### 7.3 グレースフルデグラデーション

```python
class GracefulDegradationAgent:
    """障害時に機能を段階的に縮退させるエージェント"""

    def __init__(self):
        self.degradation_level = 0  # 0=正常, 1=軽度, 2=中度, 3=重度
        self.circuit_breakers = {
            "llm_primary": CircuitBreaker(failure_threshold=3),
            "llm_secondary": CircuitBreaker(failure_threshold=5),
            "vector_db": CircuitBreaker(failure_threshold=3),
            "tools": CircuitBreaker(failure_threshold=5),
        }

    async def run(self, task: str) -> str:
        """縮退レベルに応じた処理"""
        if self.degradation_level == 0:
            return await self._full_capability(task)
        elif self.degradation_level == 1:
            return await self._reduced_tools(task)
        elif self.degradation_level == 2:
            return await self._llm_only(task)
        else:
            return self._static_response(task)

    async def _full_capability(self, task: str) -> str:
        """全機能利用可能"""
        try:
            context = await self._retrieve_context(task)
            return await self._call_llm_with_tools(task, context)
        except VectorDBError:
            self.degradation_level = 1
            return await self._reduced_tools(task)

    async def _reduced_tools(self, task: str) -> str:
        """RAGなし、基本ツールのみ"""
        try:
            return await self._call_llm_with_basic_tools(task)
        except LLMError:
            self.degradation_level = 2
            return await self._llm_only(task)

    async def _llm_only(self, task: str) -> str:
        """LLMのみ（ツールなし、フォールバックモデル）"""
        try:
            return await self._call_fallback_llm(task)
        except Exception:
            self.degradation_level = 3
            return self._static_response(task)

    def _static_response(self, task: str) -> str:
        """完全障害時の静的レスポンス"""
        return (
            "現在システムに問題が発生しています。"
            "しばらくしてから再度お試しください。"
            "緊急の場合はサポート窓口にお問い合わせください。"
        )
```

### 7.4 データ永続化とリカバリ

```python
import json
from datetime import datetime

class AgentCheckpointer:
    """エージェント実行状態のチェックポイント管理"""

    def __init__(self, storage_client):
        self.storage = storage_client

    async def save_checkpoint(
        self,
        task_id: str,
        step: int,
        messages: list,
        tool_results: list,
        metadata: dict
    ):
        """実行状態をチェックポイントとして保存"""
        checkpoint = {
            "task_id": task_id,
            "step": step,
            "messages": messages,
            "tool_results": tool_results,
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat(),
        }
        key = f"checkpoint:{task_id}:{step}"
        await self.storage.put(key, json.dumps(checkpoint))
        # 最新チェックポイントのポインタも更新
        await self.storage.put(
            f"checkpoint:{task_id}:latest",
            json.dumps({"step": step, "key": key})
        )

    async def restore_checkpoint(self, task_id: str) -> dict | None:
        """最新のチェックポイントからリストア"""
        latest_ref = await self.storage.get(
            f"checkpoint:{task_id}:latest"
        )
        if not latest_ref:
            return None

        ref = json.loads(latest_ref)
        checkpoint_data = await self.storage.get(ref["key"])
        if not checkpoint_data:
            return None

        return json.loads(checkpoint_data)

    async def resume_agent(self, task_id: str) -> str:
        """チェックポイントからエージェントを再開"""
        checkpoint = await self.restore_checkpoint(task_id)
        if not checkpoint:
            raise ValueError(f"チェックポイントが見つかりません: {task_id}")

        agent = Agent()
        agent.messages = checkpoint["messages"]
        agent.step_count = checkpoint["step"]
        agent.tool_results = checkpoint["tool_results"]

        # 中断したステップから再開
        return await agent.continue_from_step(checkpoint["step"])

    async def cleanup_checkpoints(
        self,
        task_id: str,
        keep_latest: int = 3
    ):
        """古いチェックポイントを削除"""
        latest_ref = await self.storage.get(
            f"checkpoint:{task_id}:latest"
        )
        if not latest_ref:
            return

        ref = json.loads(latest_ref)
        current_step = ref["step"]

        # 最新N件以外を削除
        for step in range(current_step - keep_latest):
            key = f"checkpoint:{task_id}:{step}"
            await self.storage.delete(key)
```

---

## 8. カナリアデプロイとブルーグリーンデプロイ

### 8.1 カナリアデプロイの実装

```python
class CanaryDeployer:
    """カナリアデプロイの自動化"""

    def __init__(self, ecs_client, cloudwatch_client):
        self.ecs = ecs_client
        self.cw = cloudwatch_client

    async def deploy_canary(
        self,
        cluster: str,
        service: str,
        new_task_def: str,
        canary_percentage: int = 10,
        monitoring_duration: int = 300,
        max_error_rate: float = 0.05
    ) -> bool:
        """カナリアデプロイの実行"""
        # 1. カナリアインスタンスをデプロイ
        await self._deploy_canary_instances(
            cluster, service, new_task_def, canary_percentage
        )

        # 2. モニタリング期間中にメトリクスを監視
        start_time = time.time()
        while time.time() - start_time < monitoring_duration:
            metrics = await self._get_canary_metrics(cluster, service)

            if metrics["error_rate"] > max_error_rate:
                # エラー率が閾値を超えた場合はロールバック
                await self._rollback(cluster, service)
                return False

            if metrics["p95_latency"] > 60:
                # レイテンシが高い場合もロールバック
                await self._rollback(cluster, service)
                return False

            await asyncio.sleep(30)  # 30秒ごとにチェック

        # 3. 成功した場合は全インスタンスを更新
        await self._promote_canary(cluster, service, new_task_def)
        return True

    async def _get_canary_metrics(
        self, cluster: str, service: str
    ) -> dict:
        """カナリアインスタンスのメトリクスを取得"""
        response = await self.cw.get_metric_data(
            MetricDataQueries=[
                {
                    "Id": "error_rate",
                    "MetricStat": {
                        "Metric": {
                            "Namespace": "AgentService",
                            "MetricName": "ErrorRate",
                            "Dimensions": [
                                {"Name": "Version", "Value": "canary"}
                            ]
                        },
                        "Period": 60,
                        "Stat": "Average"
                    }
                },
                {
                    "Id": "p95_latency",
                    "MetricStat": {
                        "Metric": {
                            "Namespace": "AgentService",
                            "MetricName": "Latency",
                            "Dimensions": [
                                {"Name": "Version", "Value": "canary"}
                            ]
                        },
                        "Period": 60,
                        "Stat": "p95"
                    }
                }
            ]
        )

        return {
            "error_rate": response["MetricDataResults"][0]["Values"][-1]
                if response["MetricDataResults"][0]["Values"] else 0,
            "p95_latency": response["MetricDataResults"][1]["Values"][-1]
                if response["MetricDataResults"][1]["Values"] else 0,
        }
```

### 8.2 ブルーグリーンデプロイ

```
ブルーグリーンデプロイの流れ

Phase 1: 準備
  [Blue (現行)] ← ALB ← トラフィック
  [Green (新版)] ← デプロイ中

Phase 2: テスト
  [Blue (現行)] ← ALB ← トラフィック
  [Green (新版)] ← スモークテスト実行

Phase 3: 切り替え
  [Blue (旧)]
  [Green (新版)] ← ALB ← トラフィック

Phase 4: クリーンアップ
  [Blue は待機 (ロールバック用、30分保持)]
  [Green (本番)] ← ALB ← トラフィック
```

---

## 9. セキュリティ考慮事項

### 9.1 シークレット管理

```python
import boto3
from functools import lru_cache

class SecretManager:
    """APIキー等のシークレット管理"""

    def __init__(self, region: str = "ap-northeast-1"):
        self.client = boto3.client(
            "secretsmanager", region_name=region
        )
        self._cache = {}
        self._cache_ttl = 300  # 5分

    @lru_cache(maxsize=32)
    def get_secret(self, secret_name: str) -> str:
        """シークレットを取得（キャッシュ付き）"""
        response = self.client.get_secret_value(SecretId=secret_name)
        return response["SecretString"]

    def rotate_api_key(self, secret_name: str, new_key: str):
        """APIキーをローテーション"""
        self.client.update_secret(
            SecretId=secret_name,
            SecretString=new_key
        )
        # キャッシュをクリア
        self.get_secret.cache_clear()
```

### 9.2 ネットワークセキュリティ

```yaml
# セキュリティグループ設定
# agent-api-sg: 8080ポートのみ、ALBからのみアクセス許可
# agent-worker-sg: アウトバウンドのみ（LLM API、Redis、DB）
# redis-sg: 6379ポート、エージェントサービスからのみ
# db-sg: 5432ポート、エージェントサービスからのみ

# WAFルール
# - レート制限: 1IPあたり100req/min
# - ボディサイズ制限: 1MB
# - SQLインジェクション保護
# - 地理的制限（必要に応じて）
```

---

## 10. アンチパターン

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

### アンチパターン3: シークレットのハードコード

```python
# NG: APIキーをコードに直書き
client = anthropic.Anthropic(api_key="sk-ant-xxxxx")

# OK: 環境変数 + シークレットマネージャー
import os
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
    or SecretManager().get_secret("anthropic-api-key")
)
```

### アンチパターン4: コスト制限なしの運用

```python
# NG: 無制限のエージェント実行
async def run_agent(task: str) -> str:
    while not done:
        response = await llm.call(messages)  # 無制限にトークン消費
        ...

# OK: 予算ガード付き
async def run_agent(task: str) -> str:
    cost_ctrl = CostController(per_task_limit=5.0)
    while not done:
        cost_ctrl.check_budget()  # 予算超過時に例外
        response = await llm.call(messages)
        cost_ctrl.track_usage(
            response.usage.input_tokens,
            response.usage.output_tokens,
            model
        )
```

### アンチパターン5: 単一障害点の放置

```python
# NG: 1つのLLMプロバイダに完全依存
client = anthropic.Anthropic()
response = client.messages.create(model="claude-sonnet-4-20250514", ...)

# OK: マルチプロバイダ + サーキットブレーカー
balancer = MultiProviderBalancer()
balancer.add_provider(LLMProvider(
    name="anthropic",
    client=anthropic.Anthropic(),
    model="claude-sonnet-4-20250514",
    weight=0.8
))
balancer.add_provider(LLMProvider(
    name="openai",
    client=openai.OpenAI(),
    model="gpt-4o",
    weight=0.2
))
response = await balancer.call(messages)
```

---

## 11. 運用チェックリスト

### 11.1 デプロイ前チェックリスト

```
本番デプロイ前チェックリスト

[ ] インフラ
  [ ] コンテナイメージがビルド・テスト済み
  [ ] 環境変数・シークレットが正しく設定されている
  [ ] ヘルスチェックエンドポイントが実装されている
  [ ] リソース制限（CPU/メモリ）が適切に設定されている
  [ ] オートスケーリングポリシーが設定されている

[ ] セキュリティ
  [ ] APIキーがシークレットマネージャーで管理されている
  [ ] ネットワークセキュリティグループが最小権限で設定されている
  [ ] WAFルールが適用されている
  [ ] 非rootユーザーでコンテナが実行される

[ ] 監視・ログ
  [ ] 構造化ログが実装されている
  [ ] メトリクス収集が設定されている
  [ ] ダッシュボードが作成されている
  [ ] アラートルールが設定されている
  [ ] 分散トレーシングが有効になっている

[ ] 障害対策
  [ ] サーキットブレーカーが実装されている
  [ ] フォールバック戦略が定義されている
  [ ] チェックポイント/リカバリが実装されている
  [ ] ロールバック手順が文書化されている

[ ] コスト
  [ ] 予算制限が設定されている
  [ ] コストアラートが設定されている
  [ ] モデル選択戦略が実装されている
  [ ] キャッシュ戦略が実装されている

[ ] テスト
  [ ] ユニットテストが通過
  [ ] 統合テストが通過
  [ ] エージェント評価スイートが基準を満たす
  [ ] 負荷テストが実施済み
```

### 11.2 インシデント対応フロー

```
インシデント対応フロー

1. 検知
   アラート発火 → PagerDuty通知 → オンコール担当に連絡

2. トリアージ（5分以内）
   - 影響範囲の特定（全ユーザー or 一部）
   - 重要度の判定（P1〜P4）
   - エスカレーション判断

3. 対応
   P1（全面障害）:
     → 即時ロールバック
     → 全トラフィック停止
     → 静的レスポンス返却

   P2（部分障害）:
     → 影響範囲のサービス縮退
     → フォールバック有効化
     → 原因調査開始

   P3（軽微な問題）:
     → 監視継続
     → 次営業日に修正

4. 復旧確認
   - メトリクス正常化の確認
   - ユーザー影響の解消確認
   - ポストモーテム作成
```

---

## 12. FAQ

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

### Q4: エージェントのA/Bテストはどう行う？

```python
class AgentABTester:
    """エージェントのA/Bテスト実装"""

    def __init__(self):
        self.variants = {}
        self.results = {}

    def add_variant(self, name: str, agent_config: dict, weight: float):
        """バリアントを追加"""
        self.variants[name] = {
            "config": agent_config,
            "weight": weight,
        }

    def select_variant(self, user_id: str) -> str:
        """ユーザーIDに基づいてバリアントを選択（一貫性のある割り当て）"""
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        normalized = hash_val / (2**128)

        cumulative = 0
        for name, variant in self.variants.items():
            cumulative += variant["weight"]
            if normalized < cumulative:
                return name

        return list(self.variants.keys())[-1]

    async def run_with_ab(self, user_id: str, task: str) -> dict:
        """A/Bテスト付きのエージェント実行"""
        variant = self.select_variant(user_id)
        config = self.variants[variant]["config"]

        agent = Agent(**config)
        start = time.time()
        result = await agent.run(task)
        latency = time.time() - start

        # 結果を記録
        await self._record_result(variant, {
            "user_id": user_id,
            "latency": latency,
            "cost": agent.total_cost,
            "steps": agent.step_count,
            "success": True,
        })

        return {"variant": variant, "result": result}
```

### Q5: 本番でのプロンプト更新はどう管理する？

- **GitOps**: プロンプトファイルをGit管理し、PR + レビューで更新
- **Feature Flag**: LaunchDarkly等で新プロンプトの段階的ロールアウト
- **評価ゲート**: 自動評価スイートを通過した場合のみデプロイ許可
- **即時ロールバック**: 問題検出時に前バージョンのプロンプトに戻す

### Q6: ステートフルなエージェントの永続化は？

- **Redis**: 短期的な会話状態（TTL付き）
- **PostgreSQL**: 長期的なタスク履歴・ユーザーコンテキスト
- **S3/GCS**: チェックポイントデータ・大容量出力
- **ベクトルDB**: エージェントのナレッジベース

---

## まとめ

| 項目 | 内容 |
|------|------|
| アーキテクチャ | API Gateway + Agent Service + Infrastructure |
| スケーリング | キューベースの水平スケーリング + レート制限適応 |
| モニタリング | メトリクス + 構造化ログ + 分散トレース |
| コスト管理 | 日次/タスク単位の予算制限 + モデルルーティング |
| 障害回復 | リトライ + フォールバック + サーキットブレーカー + チェックポイント |
| CI/CD | 自動テスト + カナリアデプロイ + 自動ロールバック |
| セキュリティ | シークレット管理 + ネットワーク分離 + WAF |
| 原則 | 非同期優先、ログ必須、段階的ロールアウト |

## 次に読むべきガイド

- [01-safety.md](./01-safety.md) -- 本番環境での安全性確保
- [../02-implementation/04-evaluation.md](../02-implementation/04-evaluation.md) -- 本番メトリクスの評価
- [../02-implementation/02-mcp-agents.md](../02-implementation/02-mcp-agents.md) -- MCPサーバーのデプロイ

## 参考文献

1. Anthropic, "API rate limits" -- https://docs.anthropic.com/en/api/rate-limits
2. LangSmith Documentation -- https://docs.smith.langchain.com/
3. AWS, "Serverless patterns for AI/ML workloads" -- https://aws.amazon.com/serverless/
4. OpenTelemetry Documentation -- https://opentelemetry.io/docs/
5. Prometheus Monitoring -- https://prometheus.io/docs/
6. Terraform AWS Provider -- https://registry.terraform.io/providers/hashicorp/aws/

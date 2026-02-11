# API 統合 — SDK・ストリーミング・リトライ戦略

> LLM API 統合はモデルの能力をアプリケーションに組み込むエンジニアリングであり、SDK 選定、ストリーミング実装、エラーハンドリング、レート制限対策、コスト管理を体系的に設計する必要がある。

## この章で学ぶこと

1. **主要プロバイダの SDK と共通抽象レイヤー** — OpenAI、Anthropic、Google、LiteLLM による統一アクセス
2. **ストリーミングの実装パターン** — SSE、WebSocket、バックプレッシャー制御
3. **プロダクション品質のエラーハンドリング** — リトライ、フォールバック、サーキットブレーカー

---

## 1. SDK 概要

```
┌──────────────────────────────────────────────────────────┐
│            LLM API 統合のレイヤー構造                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  アプリケーション                                         │
│       │                                                  │
│       ▼                                                  │
│  ┌──────────────────────────────────────┐               │
│  │  抽象レイヤー (LiteLLM / OpenRouter)  │  ← 推奨      │
│  │  - マルチプロバイダ対応               │               │
│  │  - 統一 API インターフェース           │               │
│  │  - 自動フォールバック                 │               │
│  └───────────┬──────────────────────────┘               │
│              │                                           │
│    ┌─────────┼────────────┬────────────┐                │
│    ▼         ▼            ▼            ▼                │
│  OpenAI   Anthropic    Google AI    Ollama              │
│  SDK      SDK          SDK          (Local)             │
│    │         │            │            │                │
│    ▼         ▼            ▼            ▼                │
│  GPT-4o   Claude 3.5   Gemini 1.5   Llama 3.1         │
└──────────────────────────────────────────────────────────┘
```

### 1.1 OpenAI SDK

```python
from openai import OpenAI, AsyncOpenAI

# 同期クライアント
client = OpenAI(
    api_key="sk-...",
    timeout=30.0,          # タイムアウト
    max_retries=3,         # 自動リトライ回数
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=1024,
)
print(response.choices[0].message.content)

# 非同期クライアント
async_client = AsyncOpenAI()
response = await async_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### 1.2 Anthropic SDK

```python
from anthropic import Anthropic, AsyncAnthropic

client = Anthropic()

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="あなたは有能なアシスタントです。",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.content[0].text)
print(f"入力: {response.usage.input_tokens}, 出力: {response.usage.output_tokens}")
```

### 1.3 LiteLLM (マルチプロバイダ統一)

```python
from litellm import completion, acompletion

# 同じインターフェースで異なるプロバイダを呼び出し
response = completion(
    model="gpt-4o",  # OpenAI
    messages=[{"role": "user", "content": "Hello"}],
)

response = completion(
    model="claude-3-5-sonnet-20241022",  # Anthropic
    messages=[{"role": "user", "content": "Hello"}],
)

response = completion(
    model="gemini/gemini-1.5-pro",  # Google
    messages=[{"role": "user", "content": "Hello"}],
)

response = completion(
    model="ollama/llama3.1",  # ローカル Ollama
    messages=[{"role": "user", "content": "Hello"}],
    api_base="http://localhost:11434",
)
```

---

## 2. ストリーミング実装

### 2.1 基本ストリーミング

```python
from openai import OpenAI

client = OpenAI()

# ストリーミング
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "日本の歴史を要約してください"}],
    stream=True,
)

full_response = ""
for chunk in stream:
    if chunk.choices[0].delta.content:
        token = chunk.choices[0].delta.content
        print(token, end="", flush=True)
        full_response += token
```

### 2.2 FastAPI + Server-Sent Events (SSE)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI
import json

app = FastAPI()
client = OpenAI()

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": request.message}],
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                data = json.dumps({
                    "token": chunk.choices[0].delta.content,
                    "done": False,
                })
                yield f"data: {data}\n\n"

        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

### 2.3 非同期ストリーミング

```python
from openai import AsyncOpenAI
import asyncio

async_client = AsyncOpenAI()

async def stream_response(prompt: str):
    """非同期ストリーミング"""
    stream = await async_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    tokens = []
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            tokens.append(token)
            yield token

    # 使用量はストリーム完了後に取得
    # (stream_options={"include_usage": True} で末尾チャンクに含まれる)
```

---

## 3. エラーハンドリングとリトライ

```
┌──────────────────────────────────────────────────────────┐
│           エラーハンドリング戦略                            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  HTTP Status    原因              対策                    │
│  ──────────    ─────             ─────                   │
│  400           不正リクエスト     入力検証・修正           │
│  401           認証エラー        APIキー確認              │
│  403           権限不足          プラン確認               │
│  404           モデル不存在      モデル名確認             │
│  429           レート制限        指数バックオフリトライ    │
│  500           サーバーエラー    リトライ + フォールバック │
│  503           過負荷           待機 + リトライ           │
│  タイムアウト  応答遅延         タイムアウト延長/リトライ  │
│                                                          │
│  リトライ対象: 429, 500, 503, タイムアウト                │
│  リトライ不可: 400, 401, 403, 404                        │
└──────────────────────────────────────────────────────────┘
```

### 3.1 指数バックオフリトライ

```python
import time
import random
from openai import OpenAI, RateLimitError, APIError, APITimeoutError

def call_with_retry(
    client: OpenAI,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    **kwargs,
):
    """指数バックオフ + ジッターによるリトライ"""
    retryable_errors = (RateLimitError, APIError, APITimeoutError)

    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)

        except retryable_errors as e:
            if attempt == max_retries:
                raise

            # 指数バックオフ + ジッター
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.5)
            wait_time = delay + jitter

            print(f"リトライ {attempt + 1}/{max_retries}: "
                  f"{type(e).__name__}, {wait_time:.1f}秒待機")
            time.sleep(wait_time)

        except Exception as e:
            # リトライ不可能なエラー
            raise

# 使用例
response = call_with_retry(
    client,
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### 3.2 フォールバック戦略

```python
from litellm import completion
from litellm.exceptions import (
    RateLimitError, ServiceUnavailableError, Timeout
)

# フォールバックチェーン定義
FALLBACK_CHAIN = [
    {"model": "gpt-4o", "provider": "openai"},
    {"model": "claude-3-5-sonnet-20241022", "provider": "anthropic"},
    {"model": "gemini/gemini-1.5-pro", "provider": "google"},
]

async def call_with_fallback(messages: list, **kwargs) -> str:
    """フォールバックチェーンで順に試行"""
    errors = []

    for config in FALLBACK_CHAIN:
        try:
            response = await acompletion(
                model=config["model"],
                messages=messages,
                timeout=30,
                **kwargs,
            )
            return response.choices[0].message.content

        except (RateLimitError, ServiceUnavailableError, Timeout) as e:
            errors.append(f"{config['model']}: {e}")
            continue

    raise Exception(f"全プロバイダが失敗: {errors}")
```

### 3.3 サーキットブレーカー

```python
import time
from dataclasses import dataclass, field

@dataclass
class CircuitBreaker:
    """サーキットブレーカーパターン"""
    failure_threshold: int = 5       # 失敗閾値
    recovery_timeout: float = 60.0   # 回復待機時間 (秒)
    failure_count: int = 0
    last_failure_time: float = 0
    state: str = "CLOSED"  # CLOSED / OPEN / HALF_OPEN

    def can_proceed(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        return True  # HALF_OPEN: 1回試行を許可

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

# プロバイダごとにサーキットブレーカーを持つ
breakers = {
    "openai": CircuitBreaker(),
    "anthropic": CircuitBreaker(),
    "google": CircuitBreaker(),
}
```

---

## 4. レート制限管理

### 4.1 トークンバケット

```python
import asyncio
import time

class TokenBucket:
    """トークンバケットによるレート制限"""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate          # 秒あたりのトークン補充速度
        self.capacity = capacity  # バケット容量
        self.tokens = capacity
        self.last_time = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_time = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return

            # トークン不足: 必要な待ち時間を計算
            wait = (tokens - self.tokens) / self.rate
            await asyncio.sleep(wait)
            self.tokens = 0

# RPM (Requests Per Minute) 制限
rpm_limiter = TokenBucket(rate=500/60, capacity=500)  # 500 RPM

# TPM (Tokens Per Minute) 制限
tpm_limiter = TokenBucket(rate=150_000/60, capacity=150_000)  # 150K TPM
```

---

## 5. コスト管理

### 5.1 使用量トラッキング

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class UsageTracker:
    """API使用量・コスト追跡"""

    daily_budget: float = 100.0  # 日次予算 (USD)
    records: list = field(default_factory=list)

    PRICING = {
        "gpt-4o":            {"input": 2.50, "output": 10.00},
        "gpt-4o-mini":       {"input": 0.15, "output": 0.60},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    }

    def record(self, model: str, input_tokens: int, output_tokens: int):
        prices = self.PRICING.get(model, {"input": 0, "output": 0})
        cost = (input_tokens / 1_000_000) * prices["input"] + \
               (output_tokens / 1_000_000) * prices["output"]

        self.records.append({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
        })

        # 予算超過チェック
        today_cost = self.get_today_cost()
        if today_cost > self.daily_budget:
            raise BudgetExceededError(
                f"日次予算超過: ${today_cost:.2f} / ${self.daily_budget:.2f}"
            )

        return cost

    def get_today_cost(self) -> float:
        today = datetime.now().date().isoformat()
        return sum(r["cost"] for r in self.records
                   if r["timestamp"].startswith(today))
```

---

## 6. 比較表

### 6.1 SDK 機能比較

| 機能 | OpenAI SDK | Anthropic SDK | Google SDK | LiteLLM |
|------|-----------|--------------|-----------|---------|
| 同期/非同期 | 両方 | 両方 | 同期のみ | 両方 |
| ストリーミング | 対応 | 対応 | 対応 | 対応 |
| 自動リトライ | 対応 (設定可) | 対応 | 限定的 | 対応 |
| 型安全性 | Pydantic | Pydantic | protobuf | 基本型 |
| マルチプロバイダ | N/A | N/A | N/A | 100+対応 |
| コスト追跡 | usage対応 | usage対応 | 限定的 | 統合対応 |

### 6.2 ストリーミング方式比較

| 方式 | レイテンシ | 実装複雑度 | ブラウザ対応 | 用途 |
|------|----------|----------|------------|------|
| SSE | 低 | 低 | ネイティブ | チャットUI |
| WebSocket | 最低 | 高 | ネイティブ | リアルタイム音声 |
| Long Polling | 中 | 低 | ネイティブ | レガシー対応 |
| gRPC Stream | 最低 | 高 | 間接的 | マイクロサービス |

---

## 7. アンチパターン

### アンチパターン 1: リトライなしの本番コード

```python
# NG: リトライなし — 一時的なエラーで即失敗
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
)
# → 429 (レート制限) や 500 (サーバーエラー) で即クラッシュ

# OK: SDK 組み込みリトライ + カスタムリトライ
client = OpenAI(
    max_retries=3,     # SDK レベルのリトライ
    timeout=30.0,
)
# + アプリレベルのフォールバック
response = call_with_fallback(messages)
```

### アンチパターン 2: API キーのハードコード

```python
# NG: ソースコードにキーを直書き
client = OpenAI(api_key="sk-abc123...")  # セキュリティリスク

# OK: 環境変数で管理
import os
client = OpenAI()  # 自動的に OPENAI_API_KEY 環境変数を使用

# さらに良い: シークレットマネージャー
from aws_secretsmanager import get_secret
api_key = get_secret("openai-api-key")
client = OpenAI(api_key=api_key)
```

---

## 8. FAQ

### Q1: 同期と非同期のどちらを使うべき?

Web アプリ (FastAPI等) では非同期が推奨。同時リクエスト処理が効率的になる。
バッチ処理やスクリプトでは同期で十分。
`asyncio.gather` で複数の LLM 呼び出しを並列化できるのが非同期の最大のメリット。

### Q2: ストリーミングの TTFT (Time to First Token) を改善するには?

プロンプトを短くする (入力トークン数の削減)。
Flash/mini 系の高速モデルを使用する。
CDN 経由ではなく直接 API エンドポイントに接続する。
System Prompt をキャッシュ可能にする (OpenAI, Anthropic の Prompt Caching)。

### Q3: 複数プロバイダを使い分けるベストプラクティスは?

LiteLLM や OpenRouter で抽象化し、環境変数でモデルを切り替え可能にする。
プロバイダごとのサーキットブレーカーを設置し、障害時は自動フォールバック。
コスト最適化のために、タスク難易度に応じてモデルをルーティングする (簡単→mini、難しい→Pro)。

---

## まとめ

| 項目 | 推奨 |
|------|------|
| SDK | LiteLLM (マルチプロバイダ) + 個別 SDK |
| ストリーミング | SSE (Web) / WebSocket (リアルタイム) |
| リトライ | 指数バックオフ + ジッター (最大5回) |
| フォールバック | 3プロバイダチェーン |
| レート制限 | トークンバケット + キュー |
| コスト管理 | 日次予算 + 使用量ダッシュボード |
| API キー管理 | 環境変数 + シークレットマネージャー |

---

## 次に読むべきガイド

- [01-vector-databases.md](./01-vector-databases.md) — ベクトル DB との統合
- [02-local-llm.md](./02-local-llm.md) — ローカル LLM のデプロイ
- [../02-applications/02-function-calling.md](../02-applications/02-function-calling.md) — Function Calling の統合

---

## 参考文献

1. OpenAI, "API Reference," https://platform.openai.com/docs/api-reference
2. Anthropic, "API Reference," https://docs.anthropic.com/claude/reference
3. LiteLLM, "Documentation," https://docs.litellm.ai/
4. Google, "Generative AI API," https://ai.google.dev/api

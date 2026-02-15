# API 統合 — SDK・ストリーミング・リトライ戦略

> LLM API 統合はモデルの能力をアプリケーションに組み込むエンジニアリングであり、SDK 選定、ストリーミング実装、エラーハンドリング、レート制限対策、コスト管理を体系的に設計する必要がある。

## この章で学ぶこと

1. **主要プロバイダの SDK と共通抽象レイヤー** — OpenAI、Anthropic、Google、LiteLLM による統一アクセス
2. **ストリーミングの実装パターン** — SSE、WebSocket、バックプレッシャー制御
3. **プロダクション品質のエラーハンドリング** — リトライ、フォールバック、サーキットブレーカー
4. **レート制限とコスト管理** — トークンバケット、予算管理、使用量監視
5. **プロンプトキャッシュとバッチ API** — コスト削減のための高度な API 活用
6. **セキュリティとオブザーバビリティ** — API キー管理、ログ、メトリクス

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
│  o1/o3    Haiku         Flash/Pro    Qwen 2.5          │
└──────────────────────────────────────────────────────────┘
```

### 1.1 OpenAI SDK

```python
from openai import OpenAI, AsyncOpenAI

# 同期クライアント
client = OpenAI(
    api_key="sk-...",          # 省略時は OPENAI_API_KEY 環境変数を使用
    timeout=30.0,              # タイムアウト
    max_retries=3,             # 自動リトライ回数
    base_url="https://api.openai.com/v1",  # カスタムエンドポイント対応
)

# 基本的なチャット補完
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "あなたは有能なアシスタントです。"},
        {"role": "user", "content": "Pythonのデコレータを解説してください"},
    ],
    temperature=0.7,
    max_tokens=1024,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)
print(response.choices[0].message.content)
print(f"トークン使用量: {response.usage.total_tokens}")
print(f"入力: {response.usage.prompt_tokens}, 出力: {response.usage.completion_tokens}")

# 非同期クライアント
async_client = AsyncOpenAI()
response = await async_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)

# Structured Output (JSON モード)
from pydantic import BaseModel

class ExtractedData(BaseModel):
    name: str
    age: int
    occupation: str

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "テキストから情報を抽出してJSON形式で返してください"},
        {"role": "user", "content": "田中太郎さんは35歳のエンジニアです"},
    ],
    response_format=ExtractedData,
)
data = response.choices[0].message.parsed
print(f"名前: {data.name}, 年齢: {data.age}, 職業: {data.occupation}")

# バッチ API
batch_input = client.files.create(
    file=open("batch_requests.jsonl", "rb"),
    purpose="batch",
)
batch_job = client.batches.create(
    input_file_id=batch_input.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)
print(f"バッチジョブID: {batch_job.id}, 状態: {batch_job.status}")
```

### 1.2 Anthropic SDK

```python
from anthropic import Anthropic, AsyncAnthropic

client = Anthropic()

# 基本的なメッセージ作成
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="あなたは有能なアシスタントです。",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.content[0].text)
print(f"入力: {response.usage.input_tokens}, 出力: {response.usage.output_tokens}")
print(f"停止理由: {response.stop_reason}")

# マルチモーダル入力（画像）
import base64
with open("image.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data,
                },
            },
            {
                "type": "text",
                "text": "この画像の内容を説明してください",
            },
        ],
    }],
)

# プロンプトキャッシュ
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "あなたは法律の専門家です。以下の法律文書を参照して回答してください...(長い文書)",
            "cache_control": {"type": "ephemeral"},  # キャッシュ有効化
        },
    ],
    messages=[{"role": "user", "content": "第3条について要約してください"}],
)
# cache_creation_input_tokens と cache_read_input_tokens で
# キャッシュの利用状況を確認できる
print(f"キャッシュ作成: {response.usage.cache_creation_input_tokens}")
print(f"キャッシュ読み: {response.usage.cache_read_input_tokens}")

# Extended Thinking
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=16384,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000,  # 推論に使えるトークン数の上限
    },
    messages=[{"role": "user", "content": "この数学問題を解いてください: ..."}],
)
# thinking ブロックと text ブロックが返される
for block in response.content:
    if block.type == "thinking":
        print(f"[思考過程] {block.thinking[:200]}...")
    elif block.type == "text":
        print(f"[回答] {block.text}")

# バッチ API
batch = client.messages.batches.create(
    requests=[
        {
            "custom_id": f"req-{i}",
            "params": {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": f"質問 {i}"}],
            },
        }
        for i in range(100)
    ],
)
print(f"バッチID: {batch.id}")
```

### 1.3 Google Generative AI SDK

```python
import google.generativeai as genai

genai.configure(api_key="AIza...")

# Gemini モデルの利用
model = genai.GenerativeModel("gemini-1.5-pro")

response = model.generate_content("日本の歴史を要約してください")
print(response.text)

# マルチモーダル（画像入力）
import PIL.Image
img = PIL.Image.open("chart.png")
response = model.generate_content(["このグラフを分析してください", img])

# マルチモーダル（動画入力）※ Gemini 独自機能
video_file = genai.upload_file("presentation.mp4")
response = model.generate_content(
    ["この動画の要点をまとめてください", video_file],
    request_options={"timeout": 600},
)

# 長文コンテキスト（200万トークン対応）
long_document = open("large_document.txt").read()
response = model.generate_content(
    f"以下の文書を分析して要点を抽出してください:\n\n{long_document}",
    generation_config=genai.types.GenerationConfig(
        temperature=0.3,
        max_output_tokens=8192,
    ),
)

# ストリーミング
response = model.generate_content("詳しく説明してください", stream=True)
for chunk in response:
    print(chunk.text, end="", flush=True)

# Safety 設定
from google.generativeai.types import HarmCategory, HarmBlockThreshold

response = model.generate_content(
    "...",
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    },
)
```

### 1.4 LiteLLM (マルチプロバイダ統一)

```python
from litellm import completion, acompletion
import litellm

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

# LiteLLM Router: 負荷分散 + フォールバック
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "primary",
            "litellm_params": {
                "model": "gpt-4o",
                "api_key": "sk-...",
            },
        },
        {
            "model_name": "primary",  # 同じ名前で複数モデルを設定
            "litellm_params": {
                "model": "claude-3-5-sonnet-20241022",
                "api_key": "sk-ant-...",
            },
        },
    ],
    routing_strategy="least-busy",  # latency-based-routing, simple-shuffle 等
    num_retries=3,
    fallbacks=[
        {"primary": ["gpt-4o-mini"]},  # primary が全て失敗したら mini にフォールバック
    ],
)

response = await router.acompletion(
    model="primary",
    messages=[{"role": "user", "content": "Hello"}],
)

# コスト追跡
litellm.success_callback = ["langfuse"]  # Langfuse でコスト・品質を追跡
litellm.set_verbose = True

# カスタムコールバック
def log_callback(kwargs, completion_response, start_time, end_time):
    print(f"モデル: {kwargs['model']}")
    print(f"レイテンシ: {end_time - start_time}")
    print(f"トークン: {completion_response.usage}")

litellm.success_callback = [log_callback]
```

---

## 2. ストリーミング実装

### 2.1 基本ストリーミング

```python
from openai import OpenAI

client = OpenAI()

# ストリーミング (同期)
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "日本の歴史を要約してください"}],
    stream=True,
    stream_options={"include_usage": True},  # 使用量情報を含める
)

full_response = ""
usage = None
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        token = chunk.choices[0].delta.content
        print(token, end="", flush=True)
        full_response += token
    if chunk.usage:
        usage = chunk.usage

print(f"\n\n入力: {usage.prompt_tokens}, 出力: {usage.completion_tokens}")
```

### 2.2 Anthropic ストリーミング

```python
from anthropic import Anthropic

client = Anthropic()

# ストリーミング（イベントベース）
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Pythonの非同期処理を解説してください"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# イベント詳細版
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
) as stream:
    for event in stream:
        if event.type == "message_start":
            print(f"[開始] モデル: {event.message.model}")
        elif event.type == "content_block_start":
            print(f"[ブロック開始] タイプ: {event.content_block.type}")
        elif event.type == "content_block_delta":
            if event.delta.type == "text_delta":
                print(event.delta.text, end="", flush=True)
        elif event.type == "message_delta":
            print(f"\n[完了] 停止理由: {event.delta.stop_reason}")
            print(f"出力トークン: {event.usage.output_tokens}")
```

### 2.3 FastAPI + Server-Sent Events (SSE)

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
import json
import asyncio
import time

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncOpenAI()

class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-4o"
    max_tokens: int = 1024
    temperature: float = 0.7
    stream: bool = True

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE でストリーミングレスポンスを返す"""

    async def generate():
        try:
            start_time = time.time()
            ttft = None

            stream = await client.chat.completions.create(
                model=request.model,
                messages=[{"role": "user", "content": request.message}],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    if ttft is None:
                        ttft = time.time() - start_time

                    data = json.dumps({
                        "token": token,
                        "done": False,
                        "ttft": ttft,
                    })
                    yield f"data: {data}\n\n"

                if chunk.usage:
                    usage_data = json.dumps({
                        "token": "",
                        "done": True,
                        "usage": {
                            "input_tokens": chunk.usage.prompt_tokens,
                            "output_tokens": chunk.usage.completion_tokens,
                        },
                        "latency": time.time() - start_time,
                        "ttft": ttft,
                    })
                    yield f"data: {usage_data}\n\n"

        except Exception as e:
            error_data = json.dumps({
                "error": str(e),
                "done": True,
            })
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# 非ストリーミング版（比較用）
@app.post("/chat")
async def chat(request: ChatRequest):
    """通常のJSON レスポンスを返す"""
    response = await client.chat.completions.create(
        model=request.model,
        messages=[{"role": "user", "content": request.message}],
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    return {
        "content": response.choices[0].message.content,
        "usage": {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    }
```

### 2.4 フロントエンド SSE クライアント

```typescript
// TypeScript: SSE クライアント
class LLMStreamClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async streamChat(
    message: string,
    onToken: (token: string) => void,
    onComplete: (usage: any) => void,
    onError: (error: string) => void,
  ): Promise<void> {
    const response = await fetch(`${this.baseUrl}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, stream: true }),
    });

    if (!response.ok) {
      onError(`HTTP error: ${response.status}`);
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      onError('No reader available');
      return;
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE イベントをパース
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || ''; // 最後の不完全な部分を保持

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));

          if (data.error) {
            onError(data.error);
            return;
          }

          if (data.done) {
            onComplete(data.usage);
          } else {
            onToken(data.token);
          }
        }
      }
    }
  }
}

// React Hook での使用例
function useLLMStream() {
  const [response, setResponse] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const client = new LLMStreamClient('/api');

  const sendMessage = async (message: string) => {
    setResponse('');
    setIsStreaming(true);

    await client.streamChat(
      message,
      (token) => setResponse((prev) => prev + token),
      (usage) => {
        setIsStreaming(false);
        console.log('使用量:', usage);
      },
      (error) => {
        setIsStreaming(false);
        console.error('エラー:', error);
      },
    );
  };

  return { response, isStreaming, sendMessage };
}
```

### 2.5 非同期ストリーミングとバックプレッシャー

```python
import asyncio
from openai import AsyncOpenAI
from collections import deque

class StreamBuffer:
    """バックプレッシャー対応ストリームバッファ"""

    def __init__(self, max_size: int = 100):
        self.buffer: deque = deque(maxlen=max_size)
        self.event = asyncio.Event()
        self.done = False

    async def put(self, item: str):
        """バッファにアイテムを追加"""
        while len(self.buffer) >= self.buffer.maxlen:
            # バッファが満杯: 消費されるまで待機
            await asyncio.sleep(0.01)
        self.buffer.append(item)
        self.event.set()

    async def get(self) -> str | None:
        """バッファからアイテムを取得"""
        while not self.buffer and not self.done:
            self.event.clear()
            await self.event.wait()
        if self.buffer:
            return self.buffer.popleft()
        return None

    def mark_done(self):
        self.done = True
        self.event.set()


async def producer(buffer: StreamBuffer, prompt: str):
    """LLM からのストリームをバッファに書き込む"""
    client = AsyncOpenAI()
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            await buffer.put(chunk.choices[0].delta.content)
    buffer.mark_done()


async def consumer(buffer: StreamBuffer):
    """バッファからトークンを消費して表示"""
    while True:
        token = await buffer.get()
        if token is None:
            break
        # ここで表示や加工処理を行う
        print(token, end="", flush=True)
        # 消費者が遅い場合のシミュレーション
        await asyncio.sleep(0.01)


async def main():
    buffer = StreamBuffer(max_size=50)
    await asyncio.gather(
        producer(buffer, "Pythonの非同期処理を解説してください"),
        consumer(buffer),
    )
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
│  413           入力が大きすぎる  トークン数削減            │
│  429           レート制限        指数バックオフリトライ    │
│  500           サーバーエラー    リトライ + フォールバック │
│  503           過負荷           待機 + リトライ           │
│  529           過負荷(Anthropic)  待機 + リトライ         │
│  タイムアウト  応答遅延         タイムアウト延長/リトライ  │
│                                                          │
│  リトライ対象: 429, 500, 503, 529, タイムアウト           │
│  リトライ不可: 400, 401, 403, 404, 413                   │
└──────────────────────────────────────────────────────────┘
```

### 3.1 指数バックオフリトライ（本格版）

```python
import time
import random
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable
from openai import OpenAI, RateLimitError, APIError, APITimeoutError, APIConnectionError

logger = logging.getLogger(__name__)

@dataclass
class RetryConfig:
    """リトライ設定"""
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter_factor: float = 0.5
    retryable_status_codes: set = field(default_factory=lambda: {429, 500, 503, 529})

class RetryableAPIClient:
    """リトライ対応 API クライアント"""

    def __init__(
        self,
        client: OpenAI,
        config: RetryConfig = None,
        on_retry: Optional[Callable] = None,
    ):
        self.client = client
        self.config = config or RetryConfig()
        self.on_retry = on_retry
        self.retry_count = 0
        self.total_wait_time = 0

    def call(self, **kwargs):
        """指数バックオフ + ジッターによるリトライ"""
        retryable_errors = (RateLimitError, APIError, APITimeoutError, APIConnectionError)

        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                if attempt > 0:
                    logger.info(f"リトライ成功 (試行 {attempt + 1})")
                return response

            except retryable_errors as e:
                if attempt == self.config.max_retries:
                    logger.error(f"最大リトライ回数到達: {e}")
                    raise

                # レート制限の場合、Retry-After ヘッダを尊重
                retry_after = None
                if hasattr(e, 'response') and e.response:
                    retry_after = e.response.headers.get('retry-after')

                if retry_after:
                    wait_time = float(retry_after)
                else:
                    # 指数バックオフ + ジッター
                    delay = min(
                        self.config.base_delay * (2 ** attempt),
                        self.config.max_delay,
                    )
                    jitter = random.uniform(0, delay * self.config.jitter_factor)
                    wait_time = delay + jitter

                self.retry_count += 1
                self.total_wait_time += wait_time

                logger.warning(
                    f"リトライ {attempt + 1}/{self.config.max_retries}: "
                    f"{type(e).__name__}, {wait_time:.1f}秒待機"
                )

                if self.on_retry:
                    self.on_retry(attempt, e, wait_time)

                time.sleep(wait_time)

            except Exception as e:
                # リトライ不可能なエラー（400, 401, 403等）
                logger.error(f"リトライ不可能なエラー: {type(e).__name__}: {e}")
                raise

    def get_stats(self) -> dict:
        return {
            "total_retries": self.retry_count,
            "total_wait_time": self.total_wait_time,
        }


# 使用例
client = OpenAI(max_retries=0)  # SDK のリトライは無効化
retryable = RetryableAPIClient(
    client,
    config=RetryConfig(max_retries=5, base_delay=1.0),
    on_retry=lambda attempt, err, wait: print(f"  → 待機中... ({wait:.1f}s)"),
)

response = retryable.call(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
print(f"リトライ統計: {retryable.get_stats()}")
```

### 3.2 フォールバック戦略

```python
import asyncio
import time
import logging
from dataclasses import dataclass, field
from litellm import acompletion
from litellm.exceptions import (
    RateLimitError, ServiceUnavailableError, Timeout, APIError
)

logger = logging.getLogger(__name__)

@dataclass
class FallbackConfig:
    model: str
    provider: str
    priority: int = 0  # 低い値が高優先
    max_retries: int = 2
    timeout: float = 30.0

class FallbackChain:
    """フォールバックチェーンで複数プロバイダに順に試行"""

    def __init__(self, configs: list[FallbackConfig]):
        self.configs = sorted(configs, key=lambda x: x.priority)
        self.call_history: list[dict] = []

    async def call(self, messages: list, **kwargs) -> dict:
        """フォールバックチェーンで API を呼び出す"""
        errors = []

        for config in self.configs:
            try:
                start = time.time()
                response = await acompletion(
                    model=config.model,
                    messages=messages,
                    timeout=config.timeout,
                    num_retries=config.max_retries,
                    **kwargs,
                )
                latency = time.time() - start

                result = {
                    "content": response.choices[0].message.content,
                    "model": config.model,
                    "provider": config.provider,
                    "latency": latency,
                    "usage": {
                        "input": response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens,
                    },
                    "fallback_depth": len(errors),
                }
                self.call_history.append(result)

                if errors:
                    logger.info(
                        f"フォールバック成功: {config.provider} "
                        f"(深度: {len(errors)})"
                    )
                return result

            except (RateLimitError, ServiceUnavailableError, Timeout, APIError) as e:
                error_info = {
                    "provider": config.provider,
                    "model": config.model,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                errors.append(error_info)
                logger.warning(f"フォールバック: {config.provider} 失敗 - {e}")
                continue

        raise Exception(
            f"全プロバイダが失敗:\n" +
            "\n".join(f"  - {e['provider']}: {e['error']}" for e in errors)
        )

    def get_stats(self) -> dict:
        """フォールバック統計"""
        if not self.call_history:
            return {"total_calls": 0}

        fallback_calls = sum(1 for c in self.call_history if c["fallback_depth"] > 0)
        return {
            "total_calls": len(self.call_history),
            "fallback_calls": fallback_calls,
            "fallback_rate": fallback_calls / len(self.call_history),
            "provider_distribution": {
                p: sum(1 for c in self.call_history if c["provider"] == p)
                for p in set(c["provider"] for c in self.call_history)
            },
        }


# 使用例
chain = FallbackChain([
    FallbackConfig("gpt-4o", "openai", priority=0),
    FallbackConfig("claude-3-5-sonnet-20241022", "anthropic", priority=1),
    FallbackConfig("gemini/gemini-1.5-pro", "google", priority=2),
])

result = await chain.call([{"role": "user", "content": "Hello"}])
print(f"回答: {result['content'][:100]}...")
print(f"プロバイダ: {result['provider']}, レイテンシ: {result['latency']:.2f}s")
```

### 3.3 サーキットブレーカー

```python
import time
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "CLOSED"      # 正常動作中
    OPEN = "OPEN"          # 遮断中（全リクエスト拒否）
    HALF_OPEN = "HALF_OPEN"  # 試行中（1リクエストのみ許可）

@dataclass
class CircuitBreaker:
    """サーキットブレーカーパターン"""
    name: str
    failure_threshold: int = 5       # 連続失敗閾値
    recovery_timeout: float = 60.0   # 回復待機時間 (秒)
    success_threshold: int = 3       # HALF_OPEN → CLOSED に必要な連続成功数

    def __post_init__(self):
        self.failure_count: int = 0
        self.success_count: int = 0
        self.last_failure_time: float = 0
        self.state: CircuitState = CircuitState.CLOSED
        self.total_trips: int = 0  # OPEN になった回数

    def can_proceed(self) -> bool:
        """リクエストを許可するかどうか"""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"[{self.name}] OPEN → HALF_OPEN (回復試行開始)")
                return True
            return False

        # HALF_OPEN: 1リクエストのみ許可
        return True

    def record_success(self):
        """成功を記録"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"[{self.name}] HALF_OPEN → CLOSED (回復完了)")
        else:
            self.failure_count = 0

    def record_failure(self):
        """失敗を記録"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.total_trips += 1
            logger.warning(f"[{self.name}] HALF_OPEN → OPEN (回復失敗)")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.total_trips += 1
            logger.warning(
                f"[{self.name}] CLOSED → OPEN "
                f"(連続失敗: {self.failure_count})"
            )

    def get_status(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "total_trips": self.total_trips,
            "time_since_last_failure": (
                time.time() - self.last_failure_time
                if self.last_failure_time > 0 else None
            ),
        }


# プロバイダごとにサーキットブレーカーを管理
class CircuitBreakerManager:
    """複数のサーキットブレーカーを統合管理"""

    def __init__(self):
        self.breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(self, name: str, **kwargs) -> CircuitBreaker:
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name=name, **kwargs)
        return self.breakers[name]

    def get_available_providers(self) -> list[str]:
        """利用可能なプロバイダのリスト"""
        return [
            name for name, breaker in self.breakers.items()
            if breaker.can_proceed()
        ]

    def get_all_status(self) -> list[dict]:
        return [b.get_status() for b in self.breakers.values()]


# 使用例
manager = CircuitBreakerManager()
for provider in ["openai", "anthropic", "google"]:
    manager.get_or_create(provider, failure_threshold=5, recovery_timeout=60)

# 利用可能なプロバイダを確認
available = manager.get_available_providers()
print(f"利用可能: {available}")
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

    async def acquire(self, tokens: int = 1) -> float:
        """トークンを取得（必要に応じて待機）。待機時間を返す。"""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_time = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            # トークン不足: 必要な待ち時間を計算
            wait = (tokens - self.tokens) / self.rate
            await asyncio.sleep(wait)
            self.tokens = 0
            return wait

    def available(self) -> float:
        """現在利用可能なトークン数"""
        now = time.monotonic()
        elapsed = now - self.last_time
        return min(self.capacity, self.tokens + elapsed * self.rate)


class RateLimitManager:
    """RPM と TPM の両方を管理するレートリミッター"""

    def __init__(self, rpm: int, tpm: int):
        self.rpm_limiter = TokenBucket(rate=rpm / 60, capacity=rpm)
        self.tpm_limiter = TokenBucket(rate=tpm / 60, capacity=tpm)
        self.total_wait_time = 0
        self.total_requests = 0

    async def acquire(self, estimated_tokens: int = 500):
        """リクエスト送信前にレート制限をチェック"""
        rpm_wait = await self.rpm_limiter.acquire(1)
        tpm_wait = await self.tpm_limiter.acquire(estimated_tokens)
        total_wait = rpm_wait + tpm_wait

        self.total_wait_time += total_wait
        self.total_requests += 1

        if total_wait > 0:
            logging.debug(f"レート制限待機: {total_wait:.2f}s")

    def get_stats(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_wait_time": f"{self.total_wait_time:.1f}s",
            "avg_wait_per_request": (
                f"{self.total_wait_time / self.total_requests:.3f}s"
                if self.total_requests > 0 else "N/A"
            ),
        }


# OpenAI の Tier 別レート制限
OPENAI_RATE_LIMITS = {
    "tier1": {"rpm": 500, "tpm": 200_000},
    "tier2": {"rpm": 5_000, "tpm": 2_000_000},
    "tier3": {"rpm": 5_000, "tpm": 5_000_000},
    "tier4": {"rpm": 10_000, "tpm": 10_000_000},
    "tier5": {"rpm": 10_000, "tpm": 30_000_000},
}

# 使用例
limiter = RateLimitManager(rpm=500, tpm=200_000)  # Tier 1

async def rate_limited_call(messages: list) -> str:
    await limiter.acquire(estimated_tokens=700)  # 入力+出力の概算
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    return response.choices[0].message.content
```

### 4.2 並行リクエスト制御

```python
import asyncio
from openai import AsyncOpenAI

class ConcurrencyController:
    """並行リクエスト数を制御するセマフォベースのコントローラー"""

    def __init__(
        self,
        max_concurrent: int = 10,
        rate_limiter: RateLimitManager = None,
    ):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limiter = rate_limiter
        self.active_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0

    async def execute(self, coro):
        """並行数制限付きで非同期タスクを実行"""
        async with self.semaphore:
            self.active_requests += 1
            try:
                if self.rate_limiter:
                    await self.rate_limiter.acquire()
                result = await coro
                self.completed_requests += 1
                return result
            except Exception as e:
                self.failed_requests += 1
                raise
            finally:
                self.active_requests -= 1

    async def execute_batch(
        self,
        tasks: list,
        on_progress: callable = None,
    ) -> list:
        """バッチタスクを並行数制限付きで実行"""
        results = []

        async def task_wrapper(i, task):
            result = await self.execute(task)
            if on_progress:
                on_progress(i, len(tasks))
            return result

        results = await asyncio.gather(
            *[task_wrapper(i, task) for i, task in enumerate(tasks)],
            return_exceptions=True,
        )
        return results


# 使用例: 100件のリクエストを並行10で処理
client = AsyncOpenAI()
controller = ConcurrencyController(
    max_concurrent=10,
    rate_limiter=RateLimitManager(rpm=500, tpm=200_000),
)

prompts = [f"質問 {i}: ..." for i in range(100)]

tasks = [
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": p}],
    )
    for p in prompts
]

results = await controller.execute_batch(
    tasks,
    on_progress=lambda i, total: print(f"\r進捗: {i+1}/{total}", end=""),
)
print(f"\n完了: {controller.completed_requests}, 失敗: {controller.failed_requests}")
```

---

## 5. コスト管理

### 5.1 使用量トラッキング

```python
from dataclasses import dataclass, field
from datetime import datetime, date
from collections import defaultdict
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class UsageRecord:
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    request_id: str = ""
    metadata: dict = field(default_factory=dict)

class BudgetExceededError(Exception):
    pass

class UsageTracker:
    """API使用量・コスト追跡"""

    PRICING = {
        "gpt-4o":              {"input": 2.50, "output": 10.00},
        "gpt-4o-mini":         {"input": 0.15, "output": 0.60},
        "o1":                  {"input": 15.00, "output": 60.00},
        "o3-mini":             {"input": 1.10, "output": 4.40},
        "claude-3-5-sonnet":   {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku":    {"input": 0.80, "output": 4.00},
        "gemini-1.5-pro":      {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash":    {"input": 0.075, "output": 0.30},
        "gemini-2.0-flash":    {"input": 0.10, "output": 0.40},
        "deepseek-v3":         {"input": 0.27, "output": 1.10},
    }

    def __init__(
        self,
        daily_budget: float = 100.0,
        monthly_budget: float = 3000.0,
        alert_threshold: float = 0.8,  # 予算の80%で警告
    ):
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self.alert_threshold = alert_threshold
        self.records: list[UsageRecord] = []

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        request_id: str = "",
        metadata: dict = None,
    ) -> float:
        """使用量を記録してコストを返す"""
        # モデル名の正規化
        model_key = self._normalize_model_name(model)
        prices = self.PRICING.get(model_key, {"input": 0, "output": 0})

        cost = (
            (input_tokens / 1_000_000) * prices["input"] +
            (output_tokens / 1_000_000) * prices["output"]
        )

        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            request_id=request_id,
            metadata=metadata or {},
        )
        self.records.append(record)

        # 予算チェック
        self._check_budget()

        return cost

    def _normalize_model_name(self, model: str) -> str:
        """モデル名を料金表のキーに正規化"""
        model = model.lower()
        for key in self.PRICING:
            if key in model.replace(".", "-"):
                return key
        return model

    def _check_budget(self):
        """予算超過チェック"""
        today_cost = self.get_today_cost()
        month_cost = self.get_month_cost()

        # 日次予算チェック
        if today_cost > self.daily_budget:
            raise BudgetExceededError(
                f"日次予算超過: ${today_cost:.2f} / ${self.daily_budget:.2f}"
            )

        # 月次予算チェック
        if month_cost > self.monthly_budget:
            raise BudgetExceededError(
                f"月次予算超過: ${month_cost:.2f} / ${self.monthly_budget:.2f}"
            )

        # 警告
        if today_cost > self.daily_budget * self.alert_threshold:
            logger.warning(
                f"日次予算警告: ${today_cost:.2f} / ${self.daily_budget:.2f} "
                f"({today_cost/self.daily_budget:.0%})"
            )

    def get_today_cost(self) -> float:
        today = date.today().isoformat()
        return sum(r.cost for r in self.records if r.timestamp.startswith(today))

    def get_month_cost(self) -> float:
        month = date.today().strftime("%Y-%m")
        return sum(r.cost for r in self.records if r.timestamp.startswith(month))

    def get_report(self) -> dict:
        """使用量レポートを生成"""
        today = date.today().isoformat()
        month = date.today().strftime("%Y-%m")

        # モデル別集計
        model_costs = defaultdict(float)
        model_tokens = defaultdict(lambda: {"input": 0, "output": 0})
        for r in self.records:
            if r.timestamp.startswith(month):
                model_costs[r.model] += r.cost
                model_tokens[r.model]["input"] += r.input_tokens
                model_tokens[r.model]["output"] += r.output_tokens

        return {
            "today": {
                "cost": self.get_today_cost(),
                "budget": self.daily_budget,
                "utilization": f"{self.get_today_cost()/self.daily_budget:.0%}",
            },
            "month": {
                "cost": self.get_month_cost(),
                "budget": self.monthly_budget,
                "utilization": f"{self.get_month_cost()/self.monthly_budget:.0%}",
            },
            "by_model": {
                model: {
                    "cost": f"${cost:.2f}",
                    "input_tokens": model_tokens[model]["input"],
                    "output_tokens": model_tokens[model]["output"],
                }
                for model, cost in sorted(
                    model_costs.items(), key=lambda x: -x[1]
                )
            },
            "total_requests": len(self.records),
        }


# 使用例
tracker = UsageTracker(daily_budget=100.0, monthly_budget=3000.0)

# API 呼び出し後にトラッキング
cost = tracker.record("gpt-4o", input_tokens=500, output_tokens=200)
print(f"今回のコスト: ${cost:.6f}")

report = tracker.get_report()
print(json.dumps(report, indent=2, ensure_ascii=False))
```

### 5.2 プロンプトキャッシュ戦略

```python
class PromptCacheStrategy:
    """プロンプトキャッシュ最適化戦略"""

    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def design_cacheable_prompt(
        system_prompt: str,
        few_shot_examples: list[dict],
        user_query: str,
    ) -> list[dict]:
        """キャッシュ効率の高いプロンプト構造を設計

        キャッシュのポイント:
        - system prompt と few-shot examples は先頭に配置（キャッシュ対象）
        - ユーザークエリは末尾に配置（変動部分）
        - Anthropic: cache_control で明示的にキャッシュ
        - OpenAI: 自動キャッシュ（先頭1024トークン以上が同一なら適用）
        """

        # Anthropic 向け: 明示的キャッシュ制御
        system_with_cache = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            },
        ]

        # Few-shot examples もキャッシュ対象に
        messages = []
        for i, example in enumerate(few_shot_examples):
            messages.append({"role": "user", "content": example["input"]})
            assistant_content = example["output"]
            # 最後の few-shot にキャッシュポイントを設定
            if i == len(few_shot_examples) - 1:
                messages.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": assistant_content,
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                })
            else:
                messages.append({"role": "assistant", "content": assistant_content})

        # ユーザークエリ（変動部分）
        messages.append({"role": "user", "content": user_query})

        return system_with_cache, messages

    @staticmethod
    def estimate_cache_savings(
        total_input_tokens: int,
        cacheable_tokens: int,
        requests_per_day: int,
        cache_hit_rate: float = 0.8,
        model: str = "claude-3-5-sonnet",
    ) -> dict:
        """キャッシュによるコスト削減を推定"""
        pricing = {
            "claude-3-5-sonnet": {"normal": 3.00, "cached": 0.30, "write": 3.75},
            "gpt-4o": {"normal": 2.50, "cached": 1.25, "write": 2.50},
        }
        p = pricing.get(model, pricing["claude-3-5-sonnet"])

        # キャッシュなしのコスト
        daily_tokens = total_input_tokens * requests_per_day
        cost_without_cache = (daily_tokens / 1_000_000) * p["normal"]

        # キャッシュありのコスト
        cached_tokens = cacheable_tokens * requests_per_day * cache_hit_rate
        uncached_tokens = daily_tokens - cached_tokens
        cache_write_tokens = cacheable_tokens * requests_per_day * (1 - cache_hit_rate)

        cost_with_cache = (
            (uncached_tokens / 1_000_000) * p["normal"] +
            (cached_tokens / 1_000_000) * p["cached"] +
            (cache_write_tokens / 1_000_000) * p["write"]
        )

        daily_savings = cost_without_cache - cost_with_cache
        monthly_savings = daily_savings * 30

        return {
            "daily_cost_without_cache": f"${cost_without_cache:.2f}",
            "daily_cost_with_cache": f"${cost_with_cache:.2f}",
            "daily_savings": f"${daily_savings:.2f}",
            "monthly_savings": f"${monthly_savings:.2f}",
            "savings_rate": f"{daily_savings/cost_without_cache:.0%}",
        }

# 使用例
savings = PromptCacheStrategy.estimate_cache_savings(
    total_input_tokens=2000,
    cacheable_tokens=1500,     # system + few-shot
    requests_per_day=10_000,
    cache_hit_rate=0.85,
    model="claude-3-5-sonnet",
)
print(f"月間節約額: {savings['monthly_savings']}")
print(f"削減率: {savings['savings_rate']}")
```

---

## 6. セキュリティとオブザーバビリティ

### 6.1 API キー管理

```python
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class SecureAPIKeyManager:
    """安全な API キー管理"""

    def __init__(self):
        self._keys: dict[str, str] = {}

    def get_key(self, provider: str) -> str:
        """API キーを安全に取得"""

        # 1. メモリキャッシュをチェック
        if provider in self._keys:
            return self._keys[provider]

        # 2. 環境変数から取得
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }

        env_var = env_map.get(provider)
        if env_var:
            key = os.environ.get(env_var)
            if key:
                self._keys[provider] = key
                return key

        # 3. AWS Secrets Manager から取得（本番環境向け）
        key = self._get_from_secrets_manager(provider)
        if key:
            self._keys[provider] = key
            return key

        raise ValueError(f"APIキーが見つかりません: {provider}")

    def _get_from_secrets_manager(self, provider: str) -> Optional[str]:
        """AWS Secrets Manager からキーを取得"""
        try:
            import boto3
            client = boto3.client("secretsmanager")
            response = client.get_secret_value(
                SecretId=f"llm-api-keys/{provider}",
            )
            return response["SecretString"]
        except Exception:
            return None

    @staticmethod
    def validate_key_format(provider: str, key: str) -> bool:
        """API キーの形式を検証"""
        patterns = {
            "openai": lambda k: k.startswith("sk-") and len(k) > 20,
            "anthropic": lambda k: k.startswith("sk-ant-") and len(k) > 20,
            "google": lambda k: k.startswith("AIza") and len(k) > 20,
        }
        validator = patterns.get(provider, lambda k: len(k) > 10)
        return validator(key)
```

### 6.2 リクエスト/レスポンスのログ

```python
import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime

@dataclass
class LLMRequestLog:
    """LLM リクエスト/レスポンスのログ"""
    request_id: str
    timestamp: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    ttft_ms: float = 0
    status: str = "success"
    error: str = ""
    cost: float = 0.0
    prompt_hash: str = ""  # プロンプト内容のハッシュ（PII保護）
    metadata: dict = field(default_factory=dict)

class LLMLogger:
    """LLM API 呼び出しのログ管理"""

    def __init__(self, log_prompts: bool = False):
        """
        Args:
            log_prompts: True の場合、プロンプト内容もログに記録
                        （PII が含まれる場合は False 推奨）
        """
        self.log_prompts = log_prompts
        self.logs: list[LLMRequestLog] = []
        self.logger = logging.getLogger("llm_logger")

    def log_request(
        self,
        request_id: str,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        ttft_ms: float = 0,
        status: str = "success",
        error: str = "",
        cost: float = 0.0,
        prompt: str = "",
        metadata: dict = None,
    ):
        """リクエストをログに記録"""
        # プロンプトのハッシュ化（内容は保存しない）
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16] if prompt else ""

        log = LLMRequestLog(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            status=status,
            error=error,
            cost=cost,
            prompt_hash=prompt_hash,
            metadata=metadata or {},
        )

        self.logs.append(log)
        self.logger.info(json.dumps(asdict(log)))

    def get_metrics(self) -> dict:
        """メトリクスサマリーを取得"""
        if not self.logs:
            return {"total_requests": 0}

        success_logs = [l for l in self.logs if l.status == "success"]
        error_logs = [l for l in self.logs if l.status != "success"]

        return {
            "total_requests": len(self.logs),
            "success_rate": len(success_logs) / len(self.logs),
            "error_rate": len(error_logs) / len(self.logs),
            "avg_latency_ms": (
                sum(l.latency_ms for l in success_logs) / len(success_logs)
                if success_logs else 0
            ),
            "p95_latency_ms": (
                sorted([l.latency_ms for l in success_logs])[int(len(success_logs) * 0.95)]
                if success_logs else 0
            ),
            "avg_ttft_ms": (
                sum(l.ttft_ms for l in success_logs) / len(success_logs)
                if success_logs else 0
            ),
            "total_cost": sum(l.cost for l in self.logs),
            "total_tokens": sum(l.input_tokens + l.output_tokens for l in self.logs),
            "errors_by_type": {
                error: sum(1 for l in error_logs if l.error == error)
                for error in set(l.error for l in error_logs)
            },
        }
```

---

## 7. 比較表

### 7.1 SDK 機能比較

| 機能 | OpenAI SDK | Anthropic SDK | Google SDK | LiteLLM |
|------|-----------|--------------|-----------|---------|
| 同期/非同期 | 両方 | 両方 | 同期中心 | 両方 |
| ストリーミング | 対応 | 対応 | 対応 | 対応 |
| 自動リトライ | 対応 (設定可) | 対応 | 限定的 | 対応 |
| 型安全性 | Pydantic | Pydantic | protobuf | 基本型 |
| マルチプロバイダ | N/A | N/A | N/A | 100+対応 |
| コスト追跡 | usage対応 | usage対応 | 限定的 | 統合対応 |
| Structured Output | 対応 | N/A | 対応 | プロバイダ依存 |
| プロンプトキャッシュ | 自動 | 明示的 | N/A | プロバイダ依存 |
| バッチAPI | 対応 | 対応 | N/A | プロバイダ依存 |
| Extended Thinking | N/A | 対応 | N/A | 対応 |

### 7.2 ストリーミング方式比較

| 方式 | レイテンシ | 実装複雑度 | ブラウザ対応 | 用途 |
|------|----------|----------|------------|------|
| SSE | 低 | 低 | ネイティブ | チャットUI |
| WebSocket | 最低 | 高 | ネイティブ | リアルタイム双方向 |
| Long Polling | 中 | 低 | ネイティブ | レガシー対応 |
| gRPC Stream | 最低 | 高 | 間接的 | マイクロサービス |

---

## 8. アンチパターン

### アンチパターン 1: リトライなしの本番コード

```python
# NG: リトライなし — 一時的なエラーで即失敗
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
)
# → 429 (レート制限) や 500 (サーバーエラー) で即クラッシュ

# OK: SDK 組み込みリトライ + カスタムリトライ + フォールバック
client = OpenAI(
    max_retries=3,     # SDK レベルのリトライ
    timeout=30.0,
)
# + アプリレベルのフォールバック
response = await chain.call(messages)
```

### アンチパターン 2: API キーのハードコード

```python
# NG: ソースコードにキーを直書き
client = OpenAI(api_key="sk-abc123...")  # セキュリティリスク大

# NG: .env ファイルを Git にコミット
# .gitignore に .env を追加していない

# OK: 環境変数で管理
import os
client = OpenAI()  # 自動的に OPENAI_API_KEY 環境変数を使用

# OK: シークレットマネージャー (本番環境)
key_manager = SecureAPIKeyManager()
client = OpenAI(api_key=key_manager.get_key("openai"))
```

### アンチパターン 3: レート制限を無視した並行処理

```python
# NG: 制限なしの並行リクエスト
tasks = [call_api(prompt) for prompt in prompts]  # 1000件同時
results = await asyncio.gather(*tasks)
# → 大量の 429 エラー → 全リクエスト失敗

# OK: セマフォ + レート制限で制御
controller = ConcurrencyController(
    max_concurrent=10,
    rate_limiter=RateLimitManager(rpm=500, tpm=200_000),
)
results = await controller.execute_batch(tasks)
```

### アンチパターン 4: ストリーミングなしの長時間応答

```python
# NG: 長い応答を非ストリーミングで待つ
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    max_tokens=4096,
)
# → ユーザーは10-30秒間何も表示されず待つ → 離脱

# OK: ストリーミングで即座にフィードバック
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    max_tokens=4096,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        yield chunk.choices[0].delta.content  # 即座に表示
```

### アンチパターン 5: コスト管理なしの本番運用

```python
# NG: 予算管理なし
# → 異常なトラフィックや無限ループで数万ドルの請求

# OK: 多層防御のコスト管理
tracker = UsageTracker(
    daily_budget=100.0,
    monthly_budget=3000.0,
    alert_threshold=0.8,
)

# API 呼び出しごとに追跡
cost = tracker.record(model, input_tokens, output_tokens)
# → 予算超過時は BudgetExceededError を送出
```

---

## 9. FAQ

### Q1: 同期と非同期のどちらを使うべき?

Web アプリ (FastAPI等) では非同期が推奨。同時リクエスト処理が効率的になる。
バッチ処理やスクリプトでは同期で十分。
`asyncio.gather` で複数の LLM 呼び出しを並列化できるのが非同期の最大のメリット。

**判断基準:**
- FastAPI/Starlette → 非同期必須
- Django (ASGI) → 非同期推奨
- Django (WSGI) → 同期
- CLI ツール → 同期で十分
- バッチ処理 (大量リクエスト) → 非同期 + セマフォ

### Q2: ストリーミングの TTFT を改善するには?

プロンプトを短くする (入力トークン数の削減)。
Flash/mini 系の高速モデルを使用する。
CDN 経由ではなく直接 API エンドポイントに接続する。
System Prompt をキャッシュ可能にする (OpenAI, Anthropic の Prompt Caching)。
リージョンを最寄りに設定する。

### Q3: 複数プロバイダを使い分けるベストプラクティスは?

LiteLLM や OpenRouter で抽象化し、環境変数でモデルを切り替え可能にする。
プロバイダごとのサーキットブレーカーを設置し、障害時は自動フォールバック。
コスト最適化のために、タスク難易度に応じてモデルをルーティングする。

### Q4: プロンプトキャッシュはどの程度コスト削減に効果的?

Anthropic の場合、キャッシュヒット時の入力トークン価格は通常の 1/10 (Claude 3.5 Sonnet: $3.00 → $0.30)。
システムプロンプトや Few-shot 例が固定の場合、入力コストの 50-80% を削減可能。
OpenAI は 1024 トークン以上の共通プレフィックスで自動キャッシュ（価格 50% 割引）。

### Q5: バッチ API を使うべき場面は?

リアルタイム性が不要な大量処理に最適:
- データの一括分類・タグ付け
- 大量のメール/文書の要約
- テスト・評価の実行
- コンテンツ生成のバッチ処理

メリット: 50% のコスト削減、高いスループット
デメリット: 結果取得まで最大24時間、リアルタイム処理には不向き

---

## まとめ

| 項目 | 推奨 |
|------|------|
| SDK | LiteLLM (マルチプロバイダ) + 個別 SDK |
| ストリーミング | SSE (Web) / WebSocket (リアルタイム) |
| リトライ | 指数バックオフ + ジッター (最大5回) |
| フォールバック | 3プロバイダチェーン |
| レート制限 | トークンバケット + セマフォ |
| コスト管理 | 日次/月次予算 + 使用量トラッキング |
| API キー管理 | 環境変数 + シークレットマネージャー |
| キャッシュ | プロンプトキャッシュ積極活用 |
| 監視 | リクエストログ + メトリクス収集 |

---

## 次に読むべきガイド

- [01-vector-databases.md](./01-vector-databases.md) — ベクトル DB との統合
- [02-local-llm.md](./02-local-llm.md) — ローカル LLM のデプロイ
- [../02-applications/02-function-calling.md](../02-applications/02-function-calling.md) — Function Calling の統合

---

## 参考文献

1. OpenAI, "API Reference," https://platform.openai.com/docs/api-reference
2. Anthropic, "API Reference," https://docs.anthropic.com/claude/reference
3. Anthropic, "Prompt Caching," https://docs.anthropic.com/claude/docs/prompt-caching
4. LiteLLM, "Documentation," https://docs.litellm.ai/
5. Google, "Generative AI API," https://ai.google.dev/api
6. OpenAI, "Batch API," https://platform.openai.com/docs/guides/batch
7. Anthropic, "Message Batches," https://docs.anthropic.com/claude/docs/message-batches

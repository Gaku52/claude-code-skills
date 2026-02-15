# GPT — OpenAI の大規模言語モデル

> GPT-4o、o1/o3 推論モデル、API の使い方と、OpenAI エコシステムの全体像を実践的に解説する。

## この章で学ぶこと

1. **GPT ファミリー**の進化と各モデル（GPT-4o、o1、o3）の特性
2. **OpenAI API** の実践的な使い方（Chat Completions、Assistants）
3. **推論モデル（o1/o3）**の仕組みと従来モデルとの使い分け

---

## 1. GPT ファミリーの概要

### ASCII 図解 1: GPT モデルの進化

```
GPT ファミリー進化史
──────────────────────────────────────────────────→ 時間

2020  GPT-3 (175B)
      │  初の大規模 API 提供
      ▼
2022  ChatGPT (GPT-3.5-turbo)
      │  対話型 AI の大衆化
      ▼
2023  GPT-4 (推定 MoE 1.8T)
      │  マルチモーダル対応
      ├── GPT-4 Turbo (128K context)
      │
      ▼
2024  GPT-4o ("omni")
      │  高速・低コスト・マルチモーダル統合
      ├── GPT-4o mini (小型・高速)
      │
      ├── o1-preview / o1-mini
      │   推論特化モデル (Chain-of-Thought 内蔵)
      ▼
2025  o3 / o3-mini
      │  高度な推論能力
      ├── GPT-4.5 (研究プレビュー)
      └── GPT-5 (予定)
```

### コード例 1: OpenAI API の基本

```python
from openai import OpenAI

client = OpenAI()  # OPENAI_API_KEY 環境変数を使用

# Chat Completions API
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "簡潔に日本語で回答してください。"},
        {"role": "user", "content": "量子コンピュータの原理を説明してください。"}
    ],
    temperature=0.7,
    max_tokens=500,
)

print(response.choices[0].message.content)
print(f"トークン使用量: {response.usage}")
```

### コード例 2: o1/o3 推論モデルの使用

```python
from openai import OpenAI

client = OpenAI()

# o1 / o3 は推論に時間をかける (Chain-of-Thought が内蔵)
response = client.chat.completions.create(
    model="o3-mini",
    messages=[{
        "role": "user",
        "content": """
        次の数学の問題を解いてください:

        100人が参加するトーナメント形式の試合で、
        1人の優勝者が決まるまでに必要な試合数は？
        その理由も説明してください。
        """
    }],
    # o1/o3 では temperature, top_p は設定不可
    # max_completion_tokens を使用 (max_tokens ではなく)
    max_completion_tokens=5000,
    reasoning_effort="medium",  # "low", "medium", "high"
)

print(response.choices[0].message.content)
# 推論トークン数も確認
print(f"推論トークン: {response.usage.completion_tokens_details.reasoning_tokens}")
```

### ASCII 図解 2: GPT-4o vs o1/o3 の処理フロー

```
GPT-4o (高速応答):
ユーザー → [プロンプト] → 即座に応答生成 → 応答
                         ~1-3秒
                         1パスで回答

o1/o3 (深い推論):
ユーザー → [プロンプト] → 内部思考チェーン → 応答
                         │                │
                         │ Step 1: 問題分析│
                         │ Step 2: 仮説    │
                         │ Step 3: 検証    │
                         │ Step 4: 再考    │
                         │ ...             │
                         └────────────────┘
                         ~10-60秒
                         多段階の推論
```

### コード例 3: Structured Outputs（構造化出力）

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class TechArticle(BaseModel):
    title: str
    summary: str
    tags: list[str]
    difficulty: str  # "beginner", "intermediate", "advanced"
    estimated_reading_minutes: int

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "技術記事を分析してJSON形式で出力。"},
        {"role": "user", "content": """
        以下の記事を分析してください:

        「Rustのライフタイムは、メモリ安全性を保証する仕組みです。
        コンパイラが参照の有効期間を追跡し、ダングリングポインタを
        防止します。'a のような記法で明示的に指定することもできます。」
        """}
    ],
    response_format=TechArticle,
)

article = response.choices[0].message.parsed
print(f"タイトル: {article.title}")
print(f"難易度: {article.difficulty}")
print(f"タグ: {article.tags}")
```

### コード例 4: Assistants API（永続的なスレッド）

```python
from openai import OpenAI

client = OpenAI()

# アシスタント作成
assistant = client.beta.assistants.create(
    name="データ分析アシスタント",
    instructions="あなたはデータ分析の専門家です。Pythonコードを使って分析を行います。",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o",
)

# スレッド作成
thread = client.beta.threads.create()

# メッセージ追加
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="1から100までの素数をリストアップし、その分布をグラフにしてください。"
)

# 実行
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

# 結果取得
if run.status == "completed":
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    for msg in messages.data:
        if msg.role == "assistant":
            for block in msg.content:
                if block.type == "text":
                    print(block.text.value)
```

### コード例 5: 画像生成（DALL-E 3）と画像理解の組み合わせ

```python
from openai import OpenAI
import base64

client = OpenAI()

# 1. 画像生成
image_response = client.images.generate(
    model="dall-e-3",
    prompt="ミニマリストなスタイルのAIロボットが本を読んでいるイラスト",
    size="1024x1024",
    quality="hd",
    n=1,
)
image_url = image_response.data[0].url
print(f"画像URL: {image_url}")

# 2. GPT-4o で画像理解
analysis = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "この画像を分析してください。"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }],
    max_tokens=500,
)
print(analysis.choices[0].message.content)
```

---

## 2. OpenAI エコシステムの全体像

### ASCII 図解 3: OpenAI エコシステム構成

```
┌──────────────────────────────────────────────────────────┐
│                  OpenAI エコシステム                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  基盤モデル                                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │ GPT-4o     │  │ o3/o3-mini │  │ GPT-4o     │         │
│  │ (汎用)     │  │ (推論)     │  │ mini       │         │
│  └────────────┘  └────────────┘  └────────────┘         │
│                                                          │
│  API サービス                                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │ Chat       │  │ Assistants │  │ Batch      │         │
│  │ Completions│  │ API        │  │ API        │         │
│  └────────────┘  └────────────┘  └────────────┘         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │ Embeddings │  │ Audio      │  │ Images     │         │
│  │ API        │  │ (Whisper)  │  │ (DALL-E 3) │         │
│  └────────────┘  └────────────┘  └────────────┘         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │ Fine-      │  │ Moderation │  │ Realtime   │         │
│  │ tuning     │  │ API        │  │ API        │         │
│  └────────────┘  └────────────┘  └────────────┘         │
│                                                          │
│  プロダクト                                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │ ChatGPT    │  │ Custom     │  │ ChatGPT    │         │
│  │ (消費者)   │  │ GPTs       │  │ Enterprise │         │
│  └────────────┘  └────────────┘  └────────────┘         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 3. 高度な API 機能

### 3.1 Function Calling の詳細実装

```python
from openai import OpenAI
import json
from typing import Any

client = OpenAI()

# 複数のツール定義
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "指定された銘柄の現在の株価を取得します",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "株式のティッカーシンボル（例: AAPL, GOOGL, MSFT）"
                    },
                    "currency": {
                        "type": "string",
                        "enum": ["USD", "JPY", "EUR"],
                        "description": "表示通貨"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_portfolio_return",
            "description": "ポートフォリオの期待リターンを計算します",
            "parameters": {
                "type": "object",
                "properties": {
                    "holdings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string"},
                                "weight": {"type": "number"},
                                "expected_return": {"type": "number"}
                            },
                            "required": ["symbol", "weight"]
                        },
                        "description": "保有銘柄のリスト"
                    },
                    "period": {
                        "type": "string",
                        "enum": ["1m", "3m", "6m", "1y", "5y"],
                        "description": "計算期間"
                    }
                },
                "required": ["holdings"]
            }
        }
    }
]

# 並列ツール呼び出し対応のエージェントループ
def agent_loop(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    max_iterations = 10
    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content

        # 並列ツール呼び出しを全て処理
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            # ツール実行（実際にはAPIコール等）
            result = dispatch_function(func_name, args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    return "処理が上限に達しました。"

def dispatch_function(name: str, args: dict) -> Any:
    """関数名に基づいてツールを実行"""
    functions = {
        "get_stock_price": lambda **a: {"price": 185.42, "currency": a.get("currency", "USD")},
        "calculate_portfolio_return": lambda **a: {"expected_return": 0.12, "risk": 0.08},
    }
    return functions.get(name, lambda **a: {"error": "Unknown function"})(**args)
```

### 3.2 Prompt Caching の活用

```python
from openai import OpenAI

client = OpenAI()

# 大きなシステムプロンプト（繰り返し使うもの）
LARGE_SYSTEM_PROMPT = """
あなたは金融アナリストです。以下のルールに従ってください：
1. 市場分析は客観的データに基づく
2. リスク要因を必ず明記する
3. 法的助言ではないことを明示する
... (大量の指示テキスト)
"""

# OpenAI のプロンプトキャッシュは自動的に適用される
# 同じプレフィックスが1024トークン以上ある場合にキャッシュが効く
# キャッシュヒット時は入力トークンのコストが50%割引

# キャッシュの効果を確認
for query in ["銘柄Aの分析", "銘柄Bの分析", "銘柄Cの分析"]:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": LARGE_SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
    )

    # usage にキャッシュ情報が含まれる
    usage = response.usage
    cached = getattr(usage, 'prompt_tokens_details', None)
    if cached:
        print(f"キャッシュヒット: {cached.cached_tokens} tokens")
    print(f"合計入力: {usage.prompt_tokens} tokens")
```

### 3.3 Batch API（非同期大量処理）

```python
from openai import OpenAI
import json

client = OpenAI()

# 1. バッチリクエストの準備（JSONL形式）
batch_requests = []
for i, query in enumerate(["分析1", "分析2", "分析3"]):
    batch_requests.append({
        "custom_id": f"request-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "簡潔に回答してください。"},
                {"role": "user", "content": query}
            ],
            "max_tokens": 200,
        }
    })

# JSONL ファイルに書き出し
with open("batch_input.jsonl", "w") as f:
    for req in batch_requests:
        f.write(json.dumps(req) + "\n")

# 2. ファイルアップロード
batch_file = client.files.create(
    file=open("batch_input.jsonl", "rb"),
    purpose="batch"
)

# 3. バッチジョブ作成
batch_job = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",  # 24時間以内に完了
    metadata={"description": "大量分析バッチ"}
)

print(f"バッチID: {batch_job.id}")
print(f"ステータス: {batch_job.status}")

# 4. ステータス確認
status = client.batches.retrieve(batch_job.id)
print(f"進捗: {status.request_counts}")

# 5. 結果取得（完了後）
if status.status == "completed":
    result_file = client.files.content(status.output_file_id)
    for line in result_file.text.strip().split("\n"):
        result = json.loads(line)
        print(f"{result['custom_id']}: {result['response']['body']['choices'][0]['message']['content'][:80]}...")

# コスト: 通常の50%割引で処理される
```

---

## 4. Realtime API（音声対話）

### 4.1 Realtime API の概要

```
┌──────────────────────────────────────────────────────────┐
│            GPT-4o Realtime API のアーキテクチャ            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  クライアント                 サーバー                    │
│  ┌──────────┐               ┌──────────┐               │
│  │ マイク入力│ ──WebSocket──▶│ GPT-4o   │               │
│  │ (PCM16)  │               │ Realtime │               │
│  └──────────┘               │          │               │
│  ┌──────────┐               │ ・音声認識│               │
│  │ スピーカー│ ◀──WebSocket──│ ・推論   │               │
│  │ (PCM16)  │               │ ・音声合成│               │
│  └──────────┘               └──────────┘               │
│                                                          │
│  特徴:                                                   │
│  - 音声→テキスト→推論→テキスト→音声 を統合処理           │
│  - 中間のテキスト変換をスキップ可能 (低レイテンシ)       │
│  - 会話の割り込み (バージイン) 対応                      │
│  - 6種類の音声 (alloy, echo, fable, onyx, nova, shimmer)│
│  - 入力: $5.00/1M tokens, 出力: $20.00/1M tokens        │
│                                                          │
│  ユースケース:                                           │
│  - 音声アシスタント                                      │
│  - カスタマーサポートの音声対応                           │
│  - 語学学習アプリ                                        │
│  - アクセシビリティツール                                 │
└──────────────────────────────────────────────────────────┘
```

### 4.2 Realtime API の実装例

```python
import asyncio
import websockets
import json
import base64

async def realtime_voice_assistant():
    """GPT-4o Realtime API を使った音声アシスタント"""

    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with websockets.connect(url, extra_headers=headers) as ws:
        # セッション設定
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": """あなたは日本語の会話アシスタントです。
                                  自然な口語で回答してください。
                                  専門用語は分かりやすく説明してください。""",
                "voice": "nova",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",  # サーバー側の音声検出
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "tools": [
                    {
                        "type": "function",
                        "name": "get_current_time",
                        "description": "現在の日時を取得します",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "timezone": {
                                    "type": "string",
                                    "description": "タイムゾーン（例: Asia/Tokyo）"
                                }
                            }
                        }
                    }
                ],
            },
        }))

        # メッセージ受信ループ
        async for message in ws:
            event = json.loads(message)
            event_type = event.get("type", "")

            if event_type == "session.created":
                print("セッション開始")

            elif event_type == "response.audio.delta":
                # 音声チャンクを受信 → スピーカーに送信
                audio_data = base64.b64decode(event["delta"])
                # play_audio(audio_data)  # 実際の再生処理

            elif event_type == "response.audio_transcript.delta":
                # テキスト変換結果
                print(event["delta"], end="", flush=True)

            elif event_type == "input_audio_buffer.speech_started":
                print("\n[音声検出開始]")

            elif event_type == "response.function_call_arguments.done":
                # ツール呼び出しの処理
                func_name = event.get("name")
                args = json.loads(event.get("arguments", "{}"))
                result = handle_function_call(func_name, args)

                await ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": event["call_id"],
                        "output": json.dumps(result),
                    }
                }))
                await ws.send(json.dumps({"type": "response.create"}))
```

---

## 5. Fine-tuning の実践

### 5.1 GPT-4o mini のファインチューニング

```python
from openai import OpenAI
import json

client = OpenAI()

# 1. 学習データの準備
training_data = [
    {
        "messages": [
            {"role": "system", "content": "あなたは技術ドキュメントの要約専門家です。"},
            {"role": "user", "content": "以下のドキュメントを200字以内で要約してください:\n\n[ドキュメントテキスト1]"},
            {"role": "assistant", "content": "[理想的な要約1]"}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "あなたは技術ドキュメントの要約専門家です。"},
            {"role": "user", "content": "以下のドキュメントを200字以内で要約してください:\n\n[ドキュメントテキスト2]"},
            {"role": "assistant", "content": "[理想的な要約2]"}
        ]
    },
    # 最低50例、推奨200-500例
]

# 2. バリデーションデータ（任意だが推奨）
validation_data = training_data[:10]  # 10%をバリデーションに

# JSONL ファイル作成
for filename, data in [("train.jsonl", training_data), ("val.jsonl", validation_data)]:
    with open(filename, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 3. ファイルアップロード
train_file = client.files.create(file=open("train.jsonl", "rb"), purpose="fine-tune")
val_file = client.files.create(file=open("val.jsonl", "rb"), purpose="fine-tune")

# 4. ファインチューニングジョブ作成
job = client.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=val_file.id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={
        "n_epochs": 3,
        "learning_rate_multiplier": "auto",
        "batch_size": "auto",
    },
    suffix="tech-summarizer",  # モデル名のサフィックス
)

print(f"ジョブID: {job.id}")
print(f"ステータス: {job.status}")

# 5. ステータス監視
import time
while True:
    job = client.fine_tuning.jobs.retrieve(job.id)
    print(f"ステータス: {job.status}")

    if job.status in ["succeeded", "failed", "cancelled"]:
        break
    time.sleep(60)

# 6. ファインチューニング済みモデルの使用
if job.status == "succeeded":
    ft_model = job.fine_tuned_model
    print(f"ファインチューニング済みモデル: {ft_model}")

    response = client.chat.completions.create(
        model=ft_model,
        messages=[
            {"role": "system", "content": "あなたは技術ドキュメントの要約専門家です。"},
            {"role": "user", "content": "以下のドキュメントを200字以内で要約してください:\n\n[新しいドキュメント]"}
        ]
    )
    print(response.choices[0].message.content)
```

### 5.2 ファインチューニングの評価と最適化

```python
from openai import OpenAI
import json

client = OpenAI()

def evaluate_fine_tuned_model(model_id: str, test_data: list) -> dict:
    """ファインチューニング済みモデルの品質を評価"""
    results = {
        "total": len(test_data),
        "correct": 0,
        "scores": [],
        "errors": [],
    }

    for item in test_data:
        # ファインチューニング済みモデルで推論
        response = client.chat.completions.create(
            model=model_id,
            messages=item["messages"][:-1],  # アシスタント応答を除く
            temperature=0,
        )
        prediction = response.choices[0].message.content
        expected = item["messages"][-1]["content"]

        # LLM-as-a-Judge で品質評価
        judge_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""2つのテキストを比較し、1-5のスコアで類似度を評価してください。

期待される出力: {expected}
実際の出力: {prediction}

JSON形式で回答: {{"score": <int>, "reason": "<string>"}}"""
            }],
            response_format={"type": "json_object"},
            temperature=0,
        )

        score_data = json.loads(judge_response.choices[0].message.content)
        results["scores"].append(score_data["score"])

        if score_data["score"] >= 4:
            results["correct"] += 1
        else:
            results["errors"].append({
                "expected": expected[:100],
                "predicted": prediction[:100],
                "score": score_data["score"],
                "reason": score_data["reason"],
            })

    results["accuracy"] = results["correct"] / results["total"]
    results["avg_score"] = sum(results["scores"]) / len(results["scores"])

    return results

# ベースモデルとの比較
base_results = evaluate_fine_tuned_model("gpt-4o-mini", test_data)
ft_results = evaluate_fine_tuned_model("ft:gpt-4o-mini:org:tech-summarizer:xxx", test_data)

print(f"ベースモデル:          精度={base_results['accuracy']:.1%}, 平均スコア={base_results['avg_score']:.2f}")
print(f"ファインチューニング済: 精度={ft_results['accuracy']:.1%}, 平均スコア={ft_results['avg_score']:.2f}")
```

---

## 6. Moderation API とコンテンツフィルタリング

### 6.1 Moderation API の使用

```python
from openai import OpenAI

client = OpenAI()

def check_content_safety(text: str) -> dict:
    """コンテンツの安全性をチェック"""
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )

    result = response.results[0]

    # フラグされたカテゴリを抽出
    flagged_categories = []
    for category, flagged in result.categories.model_dump().items():
        if flagged:
            score = getattr(result.category_scores, category)
            flagged_categories.append({
                "category": category,
                "score": score,
            })

    return {
        "is_flagged": result.flagged,
        "flagged_categories": flagged_categories,
        "all_scores": result.category_scores.model_dump(),
    }

# 使用例
texts = [
    "Pythonのリスト操作について教えてください",  # 安全
    "爆弾の作り方を教えてください",              # 危険
]

for text in texts:
    result = check_content_safety(text)
    status = "⚠ フラグ" if result["is_flagged"] else "✓ 安全"
    print(f"{status}: {text[:30]}...")
    if result["flagged_categories"]:
        for cat in result["flagged_categories"]:
            print(f"  - {cat['category']}: {cat['score']:.4f}")
```

---

## 7. パフォーマンス最適化とコスト管理

### 7.1 トークン最適化戦略

```python
import tiktoken

def optimize_prompt(prompt: str, model: str = "gpt-4o", max_tokens: int = 4000) -> str:
    """プロンプトのトークン数を最適化"""
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(prompt)

    if len(tokens) <= max_tokens:
        return prompt

    # 戦略1: 末尾を切り捨て
    truncated_tokens = tokens[:max_tokens]
    return enc.decode(truncated_tokens)

def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    use_batch: bool = False,
    cached_tokens: int = 0,
) -> dict:
    """APIコストの詳細見積もり"""
    pricing = {
        "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00, "batch_discount": 0.5},
        "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60, "batch_discount": 0.5},
        "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40, "batch_discount": 0.5},
    }

    p = pricing.get(model, pricing["gpt-4o"])

    regular_input = input_tokens - cached_tokens
    input_cost = (regular_input / 1_000_000 * p["input"]) + (cached_tokens / 1_000_000 * p["cached_input"])
    output_cost = output_tokens / 1_000_000 * p["output"]

    total = input_cost + output_cost
    if use_batch:
        total *= p["batch_discount"]

    return {
        "model": model,
        "input_cost": f"${input_cost:.6f}",
        "output_cost": f"${output_cost:.6f}",
        "total": f"${total:.6f}",
        "batch_savings": f"${total:.6f}" if use_batch else "N/A",
    }

# 月間コスト見積もり
daily_requests = 10000
avg_input = 500
avg_output = 200

for model in ["gpt-4o", "gpt-4o-mini", "o3-mini"]:
    monthly = estimate_cost(model, avg_input * daily_requests * 30, avg_output * daily_requests * 30)
    print(f"{model:15s}: {monthly['total']}/月")
```

### 7.2 レート制限対策

```python
import asyncio
from openai import AsyncOpenAI, RateLimitError
import time

class RateLimitedClient:
    """レート制限を考慮した OpenAI クライアント"""

    def __init__(self, rpm_limit: int = 500, tpm_limit: int = 150000):
        self.client = AsyncOpenAI()
        self.rpm_semaphore = asyncio.Semaphore(rpm_limit)
        self.request_times = []
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit

    async def create_completion(self, **kwargs):
        """レート制限を自動管理するリクエスト"""
        async with self.rpm_semaphore:
            # RPM制限チェック
            now = time.time()
            self.request_times = [t for t in self.request_times if now - t < 60]

            if len(self.request_times) >= self.rpm_limit:
                wait = 60 - (now - self.request_times[0])
                await asyncio.sleep(wait)

            self.request_times.append(time.time())

            # 指数バックオフリトライ
            for attempt in range(5):
                try:
                    return await self.client.chat.completions.create(**kwargs)
                except RateLimitError:
                    wait = 2 ** attempt
                    print(f"レート制限、{wait}秒待機...")
                    await asyncio.sleep(wait)

            raise Exception("リトライ上限超過")

# 大量リクエストの並列処理
async def process_batch(queries: list[str], model: str = "gpt-4o-mini"):
    rate_client = RateLimitedClient(rpm_limit=500)

    async def process_one(query: str):
        response = await rate_client.create_completion(
            model=model,
            messages=[{"role": "user", "content": query}],
            max_tokens=200,
        )
        return response.choices[0].message.content

    tasks = [process_one(q) for q in queries]
    return await asyncio.gather(*tasks)
```

---

## 8. GPT モデル選択ガイド

### 比較表 1: GPT モデルの詳細比較

| モデル | 入力料金 | 出力料金 | コンテキスト | 速度 | 最適用途 |
|--------|---------|---------|-------------|------|----------|
| GPT-4o | $2.50/1M | $10.00/1M | 128K | 速い | 汎用、マルチモーダル |
| GPT-4o mini | $0.15/1M | $0.60/1M | 128K | 非常に速い | 軽量タスク、分類 |
| o3-mini | $1.10/1M | $4.40/1M | 200K | 中程度 | 推論、数学、コード |
| o1 | $15.00/1M | $60.00/1M | 200K | 遅い | 高度な推論 |
| GPT-4 Turbo | $10.00/1M | $30.00/1M | 128K | 中程度 | レガシー |

### 比較表 2: GPT-4o vs o3 の使い分け

| 項目 | GPT-4o | o3 / o3-mini |
|------|--------|-------------|
| 応答速度 | 速い (1-3秒) | 遅い (10-60秒) |
| 単純な質問応答 | 最適 | 過剰 (非推奨) |
| 数学・論理パズル | 良好 | 優秀 |
| コード生成 | 優秀 | 優秀 (特に複雑なもの) |
| 創作・文章作成 | 優秀 | 不向き |
| temperature 制御 | 可能 | 不可 |
| ストリーミング | 可能 | 可能 |
| コスト効率 | 高い | 低い (推論トークンが大きい) |

### 比較表 3: API機能の詳細マトリクス

| 機能 | GPT-4o | GPT-4o mini | o3-mini | o1 |
|------|--------|-------------|---------|-----|
| Chat Completions | ✅ | ✅ | ✅ | ✅ |
| Structured Outputs | ✅ | ✅ | ✅ | ✅ |
| Function Calling | ✅ | ✅ | ✅ | ✅ |
| Vision (画像入力) | ✅ | ✅ | ❌ | ✅ |
| Audio (Realtime) | ✅ | ❌ | ❌ | ❌ |
| ストリーミング | ✅ | ✅ | ✅ | ✅ |
| Batch API | ✅ | ✅ | ✅ | ✅ |
| Fine-tuning | ✅ | ✅ | ❌ | ❌ |
| temperature 制御 | ✅ | ✅ | ❌ | ❌ |
| reasoning_effort | ❌ | ❌ | ✅ | ❌ |
| JSON Mode | ✅ | ✅ | ✅ | ✅ |
| Prompt Caching | ✅ | ✅ | ✅ | ✅ |

---

## 9. トラブルシューティング

### 9.1 よくある問題と対処法

```
┌──────────────────────────────────────────────────────────┐
│          GPT API トラブルシューティングガイド               │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  問題: 429 Too Many Requests                             │
│  原因: RPM/TPM のレート制限に到達                         │
│  対策:                                                   │
│    1. 指数バックオフリトライの実装                         │
│    2. Batch API への切り替え                              │
│    3. ティアの引き上げ申請                                │
│    4. 複数 API キーでのラウンドロビン                      │
│                                                          │
│  問題: 出力が途中で切れる                                 │
│  原因: max_tokens の設定不足                              │
│  対策:                                                   │
│    1. max_tokens を増やす                                 │
│    2. finish_reason を確認 ("stop" vs "length")           │
│    3. 長い出力が必要な場合は分割リクエスト                 │
│                                                          │
│  問題: JSON 出力が不正                                   │
│  原因: モデルがJSON形式を守らない                          │
│  対策:                                                   │
│    1. response_format={"type": "json_object"} を使用      │
│    2. Structured Outputs (Pydantic) を使用                │
│    3. temperature=0 で安定化                              │
│                                                          │
│  問題: o3 の応答が遅い                                   │
│  原因: 推論トークンの生成に時間がかかる                    │
│  対策:                                                   │
│    1. reasoning_effort="low" で速度優先                   │
│    2. 簡単なタスクは GPT-4o に振り分け                    │
│    3. ストリーミングで体感速度を改善                       │
│                                                          │
│  問題: コストが予想以上に高い                              │
│  原因: トークン消費の見積もりミス                          │
│  対策:                                                   │
│    1. Usage API でリアルタイム消費監視                     │
│    2. 日次予算アラートの設定                               │
│    3. GPT-4o mini への切り替え検討                        │
│    4. Prompt Caching の活用                               │
│    5. Batch API (50%割引) の活用                          │
└──────────────────────────────────────────────────────────┘
```

### 9.2 デバッグのベストプラクティス

```python
from openai import OpenAI
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openai_debug")

def debug_api_call(prompt: str, model: str = "gpt-4o") -> dict:
    """デバッグ情報付きの API 呼び出し"""
    client = OpenAI()

    import time
    start = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    elapsed = time.time() - start

    debug_info = {
        "model": response.model,
        "latency_seconds": round(elapsed, 3),
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "finish_reason": response.choices[0].finish_reason,
        "response_preview": response.choices[0].message.content[:200],
    }

    # 推論モデルの場合
    if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
        details = response.usage.completion_tokens_details
        if hasattr(details, 'reasoning_tokens') and details.reasoning_tokens:
            debug_info["reasoning_tokens"] = details.reasoning_tokens

    # キャッシュ情報
    if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
        details = response.usage.prompt_tokens_details
        if hasattr(details, 'cached_tokens') and details.cached_tokens:
            debug_info["cached_tokens"] = details.cached_tokens

    logger.info(f"API Call Debug: {debug_info}")
    return debug_info
```

---

## 10. 設計パターンとベストプラクティス

### 10.1 モデルルーティングパターン

```python
class GPTModelRouter:
    """タスクの複雑さに応じてモデルを自動選択"""

    def __init__(self):
        self.client = OpenAI()

    def route(self, task: str, complexity: str = "auto") -> str:
        """タスクに最適なモデルを選択"""
        if complexity == "auto":
            complexity = self._estimate_complexity(task)

        routing_table = {
            "simple": "gpt-4o-mini",      # 分類、抽出、簡単なQA
            "moderate": "gpt-4o",          # 文章作成、コード生成、分析
            "complex": "o3-mini",          # 数学、論理推論、複雑なコード
            "expert": "o1",               # 研究レベルの推論
        }

        return routing_table.get(complexity, "gpt-4o")

    def _estimate_complexity(self, task: str) -> str:
        """タスクの複雑さを推定（軽量モデルで判定）"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""以下のタスクの複雑さを判定してください。
タスク: {task}
回答は "simple", "moderate", "complex", "expert" のいずれか1つだけ:"""
            }],
            max_tokens=10,
            temperature=0,
        )
        return response.choices[0].message.content.strip().lower()

    async def execute(self, task: str) -> str:
        """ルーティング + 実行"""
        model = self.route(task)

        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": task}],
        }

        # 推論モデルの場合は max_completion_tokens を使用
        if model.startswith("o"):
            kwargs["max_completion_tokens"] = 5000
        else:
            kwargs["max_tokens"] = 2000
            kwargs["temperature"] = 0.7

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
```

### 10.2 フォールバックチェーン

```python
async def call_with_fallback(messages: list, **kwargs) -> str:
    """GPT モデルのフォールバックチェーン"""

    fallback_chain = [
        {"model": "gpt-4o", "timeout": 30},
        {"model": "gpt-4o-mini", "timeout": 15},
    ]

    errors = []
    for config in fallback_chain:
        try:
            response = await async_client.chat.completions.create(
                model=config["model"],
                messages=messages,
                timeout=config["timeout"],
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            errors.append(f"{config['model']}: {str(e)}")
            continue

    raise Exception(f"全モデルが失敗: {'; '.join(errors)}")
```

### チェックリスト: GPT API 本番導入前の確認事項

```
┌──────────────────────────────────────────────────────────┐
│          GPT API 本番導入チェックリスト                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  □ API キーは環境変数/シークレットマネージャで管理         │
│  □ リトライロジック（指数バックオフ + ジッター）を実装     │
│  □ フォールバックチェーン（複数モデル）を設定             │
│  □ レート制限対策（トークンバケット等）を実装             │
│  □ タイムアウトを適切に設定                               │
│  □ 入力バリデーション（最大長、危険文字列チェック）       │
│  □ 出力検証（JSON パース、サニタイズ）                   │
│  □ コスト上限（日次/月次予算）を設定                      │
│  □ 使用量のモニタリングダッシュボードを構築               │
│  □ Moderation API でコンテンツフィルタリングを実装        │
│  □ エラーログの収集と可視化                               │
│  □ ストリーミング（SSE）の実装とテスト                   │
│  □ プロンプトのバージョン管理                             │
│  □ A/B テスト基盤の準備                                  │
│  □ 負荷テスト（想定最大RPMでの動作確認）                 │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## アンチパターン

### アンチパターン 1: 推論モデルの不適切な使用

```
誤: 全てのタスクに o1/o3 を使用
  → 簡単な質問に対して不必要な推論コスト

正: タスクの複雑さで使い分け
  - 簡単な分類・抽出 → GPT-4o mini
  - 一般的なタスク → GPT-4o
  - 複雑な推論・数学 → o3-mini (reasoning_effort="medium")
  - 最高精度の推論 → o1 (reasoning_effort="high")
```

### アンチパターン 2: Assistants API の過度な使用

```
誤: 単発の質問応答に Assistants API を使用
  → Thread 管理のオーバーヘッド、コスト増

正: 用途に応じて API を選択
  - 単発の質問 → Chat Completions API
  - ファイル分析・コード実行 → Assistants API
  - 大量バッチ処理 → Batch API
```

### アンチパターン 3: プロンプトキャッシュの未活用

```
誤: 同じシステムプロンプトを毎回フル送信
  → 大量のトークン消費、コスト増

正: キャッシュを意識したプロンプト設計
  - システムプロンプトを先頭に配置（変更しない部分）
  - 1024トークン以上の共通プレフィックスを維持
  - ユーザー入力は末尾に配置
  → キャッシュヒットで入力コスト50%削減
```

### アンチパターン 4: Fine-tuning の早すぎる着手

```
誤: 最初からFine-tuningでタスクを解決しようとする
  → 高コスト、データ準備の工数、モデル管理の複雑化

正: 段階的アプローチ
  1. まずプロンプトエンジニアリングで解決を試みる
  2. Few-shot 例を追加する
  3. Structured Outputs で出力を制御する
  4. それでも品質不足の場合のみ Fine-tuning を検討
```

---

## FAQ

### Q1: GPT-4o と GPT-4 Turbo の違いは？

**A:** GPT-4o は GPT-4 Turbo の後継で、より高速・低コスト・マルチモーダル統合が進んでいます。特にテキスト、画像、音声を統一的に扱える点が特徴です。新規プロジェクトでは GPT-4o の使用が推奨されます。

### Q2: o1 と o3 の違いは？

**A:** o3 は o1 の改良版で、推論能力が向上しています。o3-mini は `reasoning_effort` パラメータで推論の深さを制御でき、コスト効率が良くなっています。o1 は非推奨になりつつあり、o3-mini への移行が推奨されます。

### Q3: OpenAI API の利用制限を上げるには？

**A:** 利用実績に応じて自動的にティアが上がります。Tier 1（$5 支払い）から始まり、Tier 5（$200+ 支払い）まであります。急ぎの場合は OpenAI のセールスチームに連絡して上限引き上げを依頼できます。

### Q4: GPT-4o mini と GPT-4o の品質差はどの程度ですか？

**A:** ベンチマーク上、GPT-4o mini は GPT-4o の約 90-95% の性能を発揮します。分類、要約、簡単なQAではほぼ同等ですが、複雑な推論、長文の論理構成、微妙なニュアンスの理解では差が出ます。コスト差が約17倍あるため、まず mini で試して品質不足の場合に 4o に切り替えるアプローチが推奨されます。

### Q5: Structured Outputs と JSON Mode の違いは？

**A:** JSON Mode (`response_format={"type": "json_object"}`) はJSONであることを保証しますが、スキーマの遵守は保証しません。Structured Outputs (`response_format=TechArticle` のようにPydanticモデルを指定) はスキーマに100%準拠したJSONを生成します。信頼性の高いデータ抽出にはStructured Outputsが推奨されます。

### Q6: Batch API はいつ使うべきですか？

**A:** リアルタイム応答が不要な大量処理に最適です。例えば、数千件のメール分類、大量ドキュメントの要約生成、データセットのラベリングなどです。コストが通常の50%割引になり、24時間以内に結果が返されます。日次のバッチ処理が発生するワークフローでは積極的に活用すべきです。

---

## まとめ

| 項目 | 要点 |
|------|------|
| GPT-4o | 高速・低コスト・マルチモーダルの汎用モデル |
| GPT-4o mini | 超低コストの軽量モデル、分類・抽出に最適 |
| o1/o3 | 内蔵 CoT による深い推論、数学・コードに強い |
| Structured Outputs | Pydantic スキーマで確実に構造化 JSON を取得 |
| Assistants API | ファイル分析、コード実行、永続スレッド |
| Batch API | 大量処理を 50% 割引で非同期実行 |
| Realtime API | WebSocket 経由の低レイテンシ音声対話 |
| Prompt Caching | 同一プレフィックスで入力コスト 50% 削減 |
| Fine-tuning | GPT-4o mini で効率的にドメイン特化 |
| Moderation | コンテンツの安全性を自動チェック |

---

## 次に読むべきガイド

- [02-gemini.md](./02-gemini.md) — Google Gemini の特徴
- [04-model-comparison.md](./04-model-comparison.md) — 全モデルの横断比較
- [../02-applications/02-function-calling.md](../02-applications/02-function-calling.md) — Function Calling の詳細

---

## 参考文献

1. OpenAI. (2024). "GPT-4o System Card." https://openai.com/index/gpt-4o-system-card/
2. OpenAI. (2024). "Learning to Reason with LLMs (o1)." https://openai.com/index/learning-to-reason-with-llms/
3. OpenAI. "API Reference." https://platform.openai.com/docs/api-reference
4. OpenAI. (2024). "Structured Outputs." https://platform.openai.com/docs/guides/structured-outputs
5. OpenAI. (2024). "Batch API." https://platform.openai.com/docs/guides/batch
6. OpenAI. (2024). "Realtime API." https://platform.openai.com/docs/guides/realtime

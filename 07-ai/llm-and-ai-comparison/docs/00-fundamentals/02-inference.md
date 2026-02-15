# 推論 — LLM の出力を制御するパラメータと技法

> 温度、Top-p、ストリーミング、バッチ処理など、推論時のパラメータ調整と最適化手法を実践的に学ぶ。

## この章で学ぶこと

1. **温度と Top-p/Top-k** による出力の多様性制御
2. **ストリーミング**の実装とユーザー体験の最適化
3. **バッチ処理と推論最適化**によるコスト・レイテンシの改善

---

## 1. 推論パラメータ

### ASCII 図解 1: 温度による確率分布の変化

```
確率
│
│  ██                          temperature = 0.0
│  ██                          (決定的: 最高確率のトークンのみ)
│  ██
│  ██ ░░
│  ██ ░░ ░░
│  ██ ░░ ░░ ░░
├──┬──┬──┬──┬──→ トークン
│  A  B  C  D

│  ██
│  ██ ██                       temperature = 0.7
│  ██ ██ ██                    (バランス: 多様性あり)
│  ██ ██ ██ ░░
│  ██ ██ ██ ░░
├──┬──┬──┬──┬──→ トークン
│  A  B  C  D

│  ██ ██ ██ ██                 temperature = 1.5
│  ██ ██ ██ ██                 (高多様性: ランダムに近い)
│  ██ ██ ██ ██
│  ██ ██ ██ ██
├──┬──┬──┬──┬──→ トークン
│  A  B  C  D
```

### コード例 1: 温度の効果を確認

```python
import anthropic

client = anthropic.Anthropic()

prompt = "AIの未来について一言で述べてください。"

for temp in [0.0, 0.5, 1.0]:
    responses = []
    for _ in range(3):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            temperature=temp,
            messages=[{"role": "user", "content": prompt}]
        )
        responses.append(response.content[0].text.strip())

    print(f"\n温度 {temp}:")
    for i, r in enumerate(responses, 1):
        print(f"  {i}. {r}")
```

### コード例 2: Top-p (Nucleus Sampling) の制御

```python
from openai import OpenAI

client = OpenAI()

prompt = "プログラミング言語のトップ3を挙げてください。"

# Top-p: 累積確率が p 以下のトークンのみ選択
for top_p in [0.1, 0.5, 0.9]:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        top_p=top_p,
        max_tokens=100,
    )
    print(f"top_p={top_p}: {response.choices[0].message.content[:80]}")

# 注意: temperature と top_p は同時に変更しないのがベストプラクティス
# 片方を固定し、もう片方だけ調整する
```

### ASCII 図解 2: Top-p のフィルタリング

```
確率 (ソート済み)
│
│ 0.40  ██ ─┐
│ 0.25  ██  │ 累積 0.65
│ 0.15  ██  │ 累積 0.80 ← top_p=0.8 ならここまで選択
│ 0.10  ░░ ─┘ 累積 0.90
│ 0.05  ░░   (除外)
│ 0.03  ░░   (除外)
│ 0.02  ░░   (除外)
├──┬──┬──┬──┬──┬──┬──→ トークン候補
│  A  B  C  D  E  F  G

██ = 選択対象   ░░ = 除外
top_p = 0.8 → A, B, C から確率的に選択
```

### 1.1 Top-k サンプリングの詳細

Top-k は確率上位 k 個のトークンのみを候補として残すフィルタリング手法である。Top-p が「確率の累積値」で切るのに対し、Top-k は「候補数」で切る。

```python
import numpy as np

def top_k_sampling(logits: np.ndarray, k: int = 50) -> int:
    """Top-k サンプリングの実装"""
    # 上位 k 個のインデックスを取得
    top_k_indices = np.argsort(logits)[-k:]

    # 上位 k 個以外のロジットを -inf に設定
    filtered_logits = np.full_like(logits, -np.inf)
    filtered_logits[top_k_indices] = logits[top_k_indices]

    # ソフトマックスで確率に変換
    exp_logits = np.exp(filtered_logits - np.max(filtered_logits))
    probs = exp_logits / np.sum(exp_logits)

    # 確率に基づいてサンプリング
    return np.random.choice(len(probs), p=probs)


def top_p_sampling(logits: np.ndarray, p: float = 0.9) -> int:
    """Top-p (Nucleus) サンプリングの実装"""
    # ソフトマックスで確率に変換
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    # 確率の降順にソート
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    # 累積確率が p を超えるまでのトークンを残す
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff_idx = np.searchsorted(cumulative_probs, p) + 1

    # 選択されたトークンのみ残す
    selected_indices = sorted_indices[:cutoff_idx]
    selected_probs = probs[selected_indices]
    selected_probs /= selected_probs.sum()  # 再正規化

    return np.random.choice(selected_indices, p=selected_probs)


def combined_sampling(
    logits: np.ndarray,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> int:
    """温度 + Top-k + Top-p の組み合わせサンプリング"""
    # 1. 温度スケーリング
    scaled_logits = logits / max(temperature, 1e-8)

    # 2. Top-k フィルタリング
    if top_k > 0:
        top_k_indices = np.argsort(scaled_logits)[-top_k:]
        mask = np.full_like(scaled_logits, -np.inf)
        mask[top_k_indices] = scaled_logits[top_k_indices]
        scaled_logits = mask

    # 3. ソフトマックス
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_logits / np.sum(exp_logits)

    # 4. Top-p フィルタリング
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative, top_p) + 1

    selected = sorted_indices[:cutoff]
    selected_probs = probs[selected]
    selected_probs /= selected_probs.sum()

    return np.random.choice(selected, p=selected_probs)


# 実験: 各サンプリング方式の出力分布を比較
np.random.seed(42)
vocab_size = 100
logits = np.random.randn(vocab_size)
logits[0] = 3.0   # トークン 0 を高確率に
logits[1] = 2.5
logits[2] = 2.0

n_samples = 10000
results = {"top_k": [], "top_p": [], "combined": []}

for _ in range(n_samples):
    results["top_k"].append(top_k_sampling(logits, k=10))
    results["top_p"].append(top_p_sampling(logits, p=0.9))
    results["combined"].append(combined_sampling(logits, temperature=0.7, top_k=50, top_p=0.9))

for method, samples in results.items():
    unique = len(set(samples))
    top3_ratio = sum(1 for s in samples if s in [0, 1, 2]) / n_samples
    print(f"{method:10s}: ユニークトークン数={unique:3d}, Top3占有率={top3_ratio:.2%}")
```

### ASCII 図解: サンプリング方式の比較

```
┌──────────────────────────────────────────────────────────┐
│         サンプリング方式の選択フローチャート                 │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  用途を判定                                               │
│    │                                                     │
│    ├─ 正確性重視 (コード生成、データ抽出)                    │
│    │   → temperature = 0.0 (Greedy Decoding)             │
│    │   → top_p = 1.0, top_k = 1                         │
│    │                                                     │
│    ├─ バランス型 (一般的なアシスタント)                      │
│    │   → temperature = 0.7                                │
│    │   → top_p = 0.9 (片方だけ調整)                       │
│    │                                                     │
│    ├─ 創造性重視 (ブレスト、物語生成)                       │
│    │   → temperature = 1.0 - 1.2                          │
│    │   → top_p = 0.95, top_k = 100                       │
│    │                                                     │
│    └─ 多様性探索 (複数候補生成)                             │
│        → temperature = 1.0                                │
│        → top_k = 50, n = 5 (5候補生成)                    │
│                                                          │
│  重要: temperature と top_p を同時に極端に設定しない        │
│  → 予測不能な挙動の原因となる                              │
└──────────────────────────────────────────────────────────┘
```

### 1.2 Repetition Penalty と Frequency Penalty

繰り返し抑制パラメータは、生成テキストの冗長性を制御する。

```python
from openai import OpenAI

client = OpenAI()

# Frequency Penalty: 既出トークンの確率を線形に低下
# Presence Penalty: 既出トークンの有無で一定量低下
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "AIの応用分野を列挙してください"}],
    temperature=0.7,
    frequency_penalty=0.5,   # 0.0 ~ 2.0: 同一トークンの繰り返しを抑制
    presence_penalty=0.3,    # 0.0 ~ 2.0: 新しいトピックへの誘導
    max_tokens=500,
)

# 比較実験: ペナルティなし vs あり
for fp, pp in [(0.0, 0.0), (0.5, 0.3), (1.5, 1.0)]:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "詩を書いてください"}],
        frequency_penalty=fp,
        presence_penalty=pp,
        max_tokens=200,
    )
    print(f"\nfreq={fp}, pres={pp}:")
    print(resp.choices[0].message.content[:150])
```

```
┌──────────────────────────────────────────────────────────┐
│       Penalty パラメータの効果                              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Frequency Penalty (頻度ペナルティ):                      │
│  ─────────────────────────────────                       │
│  トークン出現回数 × penalty 値 を logit から減算           │
│                                                          │
│  例: "猫" が3回出現、penalty=0.5                          │
│  → "猫" の logit から 3 × 0.5 = 1.5 を減算              │
│  → 出現回数が増えるほど確率が下がる                       │
│                                                          │
│  Presence Penalty (存在ペナルティ):                       │
│  ─────────────────────────────────                       │
│  トークンが1回でも出現していれば penalty 値を減算          │
│                                                          │
│  例: "猫" が出現済み、penalty=0.5                         │
│  → "猫" の logit から 0.5 を減算 (回数に関係なく一定)    │
│  → 新しい単語・トピックへの誘導に効果的                   │
│                                                          │
│  推奨設定:                                                │
│  ├── リスト生成 → freq=0.5, pres=0.3                     │
│  ├── 文章作成   → freq=0.3, pres=0.2                     │
│  ├── 対話     → freq=0.1, pres=0.1                       │
│  └── コード生成 → freq=0.0, pres=0.0 (変数名の繰り返しは正常)│
└──────────────────────────────────────────────────────────┘
```

### 1.3 Stop Sequences (停止シーケンス)

```python
from openai import OpenAI

client = OpenAI()

# Stop Sequences: 特定文字列が生成されたら停止
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "1から10まで数えて"}],
    stop=["5"],     # "5" が生成された時点で停止
    max_tokens=100,
)
print(response.choices[0].message.content)
# → "1, 2, 3, 4, "

# 実用例: JSON 抽出時に余分な出力を防ぐ
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": "以下のテキストからキーワードをJSON配列で抽出: 'Python機械学習入門'"
    }],
    stop=["```", "\n\n"],  # コードブロック終了や二重改行で停止
    max_tokens=200,
)

# Claude API での Stop Sequences
import anthropic

client_claude = anthropic.Anthropic()
response = client_claude.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=200,
    stop_sequences=["---", "END"],
    messages=[{"role": "user", "content": "レポートを書いてください"}],
)
print(f"停止理由: {response.stop_reason}")
# "stop_sequence" or "end_turn" or "max_tokens"
```

### 1.4 Seed パラメータによる再現性

```python
from openai import OpenAI

client = OpenAI()

# seed を指定して再現性を高める
responses = []
for _ in range(3):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "ランダムな6桁の数字を1つ生成して"}],
        seed=42,
        temperature=0.0,
    )
    responses.append(response.choices[0].message.content)
    # system_fingerprint で同一バックエンドの確認
    print(f"fingerprint: {response.system_fingerprint}")

# 同じ seed + temperature=0 でも 100% 再現は保証されない
# (GPU の非決定性、モデル更新による変化)
print(f"再現率: {len(set(responses))}/{len(responses)} ユニーク")

# ベストプラクティス: 再現性が重要な場合
# 1. seed を固定
# 2. temperature = 0
# 3. system_fingerprint をログに記録
# 4. モデルバージョンを固定 (例: gpt-4o-2024-08-06)
```

### 1.5 Logprobs (対数確率) の活用

```python
from openai import OpenAI
import math

client = OpenAI()

# logprobs を有効にして確率情報を取得
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "日本の首都は"}],
    max_tokens=10,
    logprobs=True,
    top_logprobs=5,  # 上位5トークンの確率を取得
)

# 各トークンの確率を表示
for token_info in response.choices[0].logprobs.content:
    prob = math.exp(token_info.logprob)
    print(f"\n選択: '{token_info.token}' (確率: {prob:.2%})")

    # 代替候補
    if token_info.top_logprobs:
        for alt in token_info.top_logprobs:
            alt_prob = math.exp(alt.logprob)
            print(f"  候補: '{alt.token}' (確率: {alt_prob:.2%})")


# 実用例: 信頼度スコアの計算
def get_confidence_score(response) -> float:
    """生成テキストの信頼度を logprobs から算出"""
    if not response.choices[0].logprobs:
        return 0.0

    log_probs = [
        t.logprob
        for t in response.choices[0].logprobs.content
        if t.logprob is not None
    ]

    if not log_probs:
        return 0.0

    # 平均対数確率 → 確率に変換
    avg_logprob = sum(log_probs) / len(log_probs)
    return math.exp(avg_logprob)


# 信頼度による分岐処理
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "この文を分類: '猫がかわいい'"}],
    max_tokens=20,
    logprobs=True,
)

confidence = get_confidence_score(response)
if confidence > 0.8:
    print("高信頼度: 自動処理可能")
elif confidence > 0.5:
    print("中信頼度: 追加確認推奨")
else:
    print("低信頼度: 人間レビュー必須")
```

---

## 2. ストリーミング

### コード例 3: Claude API でストリーミング

```python
import anthropic

client = anthropic.Anthropic()

print("ストリーミング応答:")
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=500,
    messages=[{
        "role": "user",
        "content": "Pythonの主要なデザインパターンを3つ説明してください。"
    }]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
print()  # 改行

# イベントベースの処理
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=500,
    messages=[{"role": "user", "content": "Hello"}]
) as stream:
    for event in stream:
        if event.type == "content_block_delta":
            print(f"[delta] {event.delta.text}", end="")
        elif event.type == "message_stop":
            print("\n[完了]")
```

### コード例 4: OpenAI API でストリーミング

```python
from openai import OpenAI

client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": "機械学習の基本ステップを説明してください。"
    }],
    stream=True,
    stream_options={"include_usage": True},  # 使用量も取得
)

full_response = ""
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        content = chunk.choices[0].delta.content
        full_response += content
        print(content, end="", flush=True)
    # ストリーム終了時にトークン使用量を取得
    if chunk.usage:
        print(f"\n\n使用トークン: {chunk.usage.total_tokens}")
```

### ASCII 図解 3: ストリーミング vs 非ストリーミング

```
非ストリーミング:
User ──リクエスト──→ API ──────────────────→ 全文応答
                     │   (生成中...待機)    │
                     │   TTFB: 3-10秒       │
                     └──────────────────────┘
                     ←──── 体感遅延 大 ────→

ストリーミング:
User ──リクエスト──→ API ─→ チャンク1
                          ─→ チャンク2
                          ─→ チャンク3
                          ─→ ...
                          ─→ [DONE]
                     ←──→
                     TTFB: 0.3-1秒
                     ←──── 体感遅延 小 ────→

TTFB = Time To First Byte（最初の応答までの時間）
```

### 2.1 Server-Sent Events (SSE) の詳細

ストリーミングは HTTP の Server-Sent Events (SSE) プロトコルを使用する。

```python
import httpx
import json

# SSE を直接パースする低レベル実装
async def stream_with_sse(prompt: str):
    """SSE プロトコルで直接ストリーミング"""
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
        ) as response:
            buffer = ""
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    chunk = json.loads(data)
                    if chunk["choices"][0]["delta"].get("content"):
                        token = chunk["choices"][0]["delta"]["content"]
                        buffer += token
                        yield token

# FastAPI でのストリーミングプロキシ実装
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(request: dict):
    async def generate():
        async for token in stream_with_sse(request["message"]):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx バッファリング無効化
        },
    )
```

### 2.2 フロントエンドでのストリーミング表示

```typescript
// TypeScript: フロントエンドでの SSE 受信
async function streamChat(message: string): Promise<void> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  if (!response.body) throw new Error("No response body");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE パーシング
    const lines = buffer.split("\n\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const data = line.slice(6);
        if (data === "[DONE]") return;

        const parsed = JSON.parse(data);
        appendToUI(parsed.token);  // UI にトークンを追加
      }
    }
  }
}

// React コンポーネント例
function ChatMessage({ streamUrl }: { streamUrl: string }) {
  const [text, setText] = useState("");
  const [isStreaming, setIsStreaming] = useState(true);

  useEffect(() => {
    const eventSource = new EventSource(streamUrl);

    eventSource.onmessage = (event) => {
      if (event.data === "[DONE]") {
        setIsStreaming(false);
        eventSource.close();
        return;
      }

      const data = JSON.parse(event.data);
      setText((prev) => prev + data.token);
    };

    eventSource.onerror = () => {
      setIsStreaming(false);
      eventSource.close();
    };

    return () => eventSource.close();
  }, [streamUrl]);

  return (
    <div className="message">
      {text}
      {isStreaming && <span className="cursor blink">|</span>}
    </div>
  );
}
```

### 2.3 ストリーミング中断とタイムアウト処理

```python
import asyncio
from openai import AsyncOpenAI

async def stream_with_timeout(prompt: str, timeout_seconds: float = 30.0):
    """タイムアウト付きストリーミング"""
    client = AsyncOpenAI()
    full_text = ""

    try:
        async with asyncio.timeout(timeout_seconds):
            stream = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=2000,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_text += token
                    print(token, end="", flush=True)

    except asyncio.TimeoutError:
        print(f"\n[タイムアウト] {timeout_seconds}秒経過")
        # 部分的な応答を返す
    except Exception as e:
        print(f"\n[エラー] {e}")

    return full_text


# ユーザーキャンセル対応
async def stream_with_cancel(prompt: str, cancel_event: asyncio.Event):
    """キャンセル可能なストリーミング"""
    client = AsyncOpenAI()
    full_text = ""

    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    async for chunk in stream:
        if cancel_event.is_set():
            print("\n[キャンセル]")
            break

        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_text += token
            yield token

    return full_text
```

---

## 3. バッチ処理と最適化

### コード例 5: バッチ API の活用

```python
import anthropic
import asyncio

client = anthropic.AsyncAnthropic()

async def process_batch(prompts: list[str]) -> list[str]:
    """複数プロンプトを並列処理"""
    async def single_request(prompt: str) -> str:
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    tasks = [single_request(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return results

# 使用例
prompts = [
    "Pythonの利点を3つ",
    "Rustの利点を3つ",
    "Goの利点を3つ",
    "TypeScriptの利点を3つ",
]

results = asyncio.run(process_batch(prompts))
for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result[:100]}...")
    print()

# OpenAI Batch API（非同期バッチ、50%割引）
from openai import OpenAI
client_oai = OpenAI()

batch_input = [
    {
        "custom_id": f"request-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": p}],
            "max_tokens": 200,
        }
    }
    for i, p in enumerate(prompts)
]
# JSONL ファイルに書き出してバッチ送信
```

### 3.1 OpenAI Batch API の完全ワークフロー

```python
import json
import time
from openai import OpenAI

client = OpenAI()

def run_batch_job(prompts: list[str], model: str = "gpt-4o-mini") -> list[dict]:
    """OpenAI Batch API の完全な利用フロー"""

    # 1. JSONL ファイルを作成
    batch_requests = []
    for i, prompt in enumerate(prompts):
        batch_requests.append({
            "custom_id": f"req-{i:06d}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": "簡潔に回答してください。"},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 500,
                "temperature": 0.3,
            },
        })

    input_file = "batch_input.jsonl"
    with open(input_file, "w") as f:
        for req in batch_requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    # 2. ファイルをアップロード
    uploaded = client.files.create(
        file=open(input_file, "rb"),
        purpose="batch",
    )
    print(f"アップロード完了: {uploaded.id}")

    # 3. バッチジョブを作成
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "batch processing experiment"},
    )
    print(f"バッチジョブ開始: {batch.id}")

    # 4. 完了を待機 (ポーリング)
    while True:
        status = client.batches.retrieve(batch.id)
        print(f"ステータス: {status.status} "
              f"(完了: {status.request_counts.completed}/"
              f"{status.request_counts.total})")

        if status.status == "completed":
            break
        elif status.status in ["failed", "cancelled", "expired"]:
            raise RuntimeError(f"バッチ失敗: {status.status}")

        time.sleep(30)  # 30秒ごとにチェック

    # 5. 結果をダウンロード
    output_file = client.files.content(status.output_file_id)
    results = []
    for line in output_file.text.strip().split("\n"):
        result = json.loads(line)
        results.append({
            "id": result["custom_id"],
            "status": result["response"]["status_code"],
            "content": result["response"]["body"]["choices"][0]["message"]["content"],
        })

    # 6. エラー結果の処理
    if status.error_file_id:
        error_file = client.files.content(status.error_file_id)
        for line in error_file.text.strip().split("\n"):
            error = json.loads(line)
            print(f"エラー: {error['custom_id']}: {error['response']['body']}")

    return results


# 使用例: 1000件のドキュメント分類
documents = [f"文書{i}の内容..." for i in range(1000)]
prompts = [f"以下の文書を「技術」「ビジネス」「その他」に分類: {doc}" for doc in documents]
results = run_batch_job(prompts)

# コスト比較: Batch API は通常 API の 50% OFF
# 1000件 × 500入力tok × 100出力tok = 50万入力 + 10万出力
# 通常: $0.075 + $0.060 = $0.135
# Batch: $0.0375 + $0.030 = $0.0675 (50% OFF)
```

### 3.2 Anthropic Message Batches API

```python
import anthropic
import time

client = anthropic.Anthropic()

def run_anthropic_batch(prompts: list[str]) -> list[dict]:
    """Anthropic Message Batches API の利用フロー"""

    # 1. バッチリクエストを作成
    requests = [
        {
            "custom_id": f"req-{i:06d}",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}],
            },
        }
        for i, prompt in enumerate(prompts)
    ]

    # 2. バッチ送信
    batch = client.messages.batches.create(requests=requests)
    print(f"バッチ ID: {batch.id}")

    # 3. 完了待機
    while True:
        status = client.messages.batches.retrieve(batch.id)
        counts = status.request_counts
        print(f"処理中: {counts.processing}, 完了: {counts.succeeded}, "
              f"エラー: {counts.errored}")

        if status.processing_status == "ended":
            break

        time.sleep(30)

    # 4. 結果取得
    results = []
    for result in client.messages.batches.results(batch.id):
        if result.result.type == "succeeded":
            results.append({
                "id": result.custom_id,
                "content": result.result.message.content[0].text,
            })
        else:
            print(f"エラー: {result.custom_id}: {result.result}")

    return results
```

### 3.3 レート制限の管理

```python
import asyncio
import time
from dataclasses import dataclass
from collections import deque

@dataclass
class RateLimiter:
    """トークンバケットベースのレート制限"""
    requests_per_minute: int
    tokens_per_minute: int
    _request_times: deque = None
    _token_counts: deque = None

    def __post_init__(self):
        self._request_times = deque()
        self._token_counts = deque()

    async def acquire(self, estimated_tokens: int = 1000):
        """レート制限内でリクエスト可能になるまで待機"""
        while True:
            now = time.time()
            window = now - 60  # 1分間のウィンドウ

            # 古いエントリを削除
            while self._request_times and self._request_times[0] < window:
                self._request_times.popleft()
            while self._token_counts and self._token_counts[0][0] < window:
                self._token_counts.popleft()

            # 現在のレートを確認
            current_requests = len(self._request_times)
            current_tokens = sum(tc[1] for tc in self._token_counts)

            if (current_requests < self.requests_per_minute and
                current_tokens + estimated_tokens < self.tokens_per_minute):
                self._request_times.append(now)
                self._token_counts.append((now, estimated_tokens))
                return

            # 最も古いエントリが期限切れになるまで待機
            if self._request_times:
                wait_time = self._request_times[0] - window + 0.1
                await asyncio.sleep(max(wait_time, 0.1))
            else:
                await asyncio.sleep(0.1)


# 使用例: レート制限付き並列処理
async def process_with_rate_limit(
    prompts: list[str],
    rpm: int = 60,
    tpm: int = 100_000,
):
    """レート制限を遵守しながら並列処理"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    limiter = RateLimiter(requests_per_minute=rpm, tokens_per_minute=tpm)

    async def process_one(prompt: str, idx: int) -> dict:
        estimated_tokens = len(prompt.split()) * 2 + 200  # 概算
        await limiter.acquire(estimated_tokens)

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )

        return {
            "index": idx,
            "content": response.choices[0].message.content,
            "tokens": response.usage.total_tokens,
        }

    tasks = [process_one(p, i) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successes = [r for r in results if isinstance(r, dict)]
    errors = [r for r in results if isinstance(r, Exception)]

    print(f"成功: {len(successes)}, エラー: {len(errors)}")
    return successes
```

### 3.4 リトライとエラーハンドリング

```python
import asyncio
import random
from openai import AsyncOpenAI, RateLimitError, APIError, APITimeoutError

async def resilient_request(
    client: AsyncOpenAI,
    prompt: str,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> str:
    """指数バックオフ付きリトライ"""

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                timeout=30.0,
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            # レート制限: より長く待つ
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            # Retry-After ヘッダがあればそれを使用
            retry_after = getattr(e, "retry_after", None)
            if retry_after:
                delay = max(delay, float(retry_after))
            print(f"レート制限 (attempt {attempt+1}): {delay:.1f}秒待機")
            await asyncio.sleep(delay)

        except APITimeoutError:
            delay = base_delay * (2 ** attempt)
            print(f"タイムアウト (attempt {attempt+1}): {delay:.1f}秒後にリトライ")
            await asyncio.sleep(delay)

        except APIError as e:
            if e.status_code and e.status_code >= 500:
                # サーバーエラー: リトライ
                delay = base_delay * (2 ** attempt)
                print(f"サーバーエラー {e.status_code} (attempt {attempt+1})")
                await asyncio.sleep(delay)
            else:
                # クライアントエラー: リトライしない
                raise

    raise RuntimeError(f"最大リトライ回数 ({max_retries}) に到達")


# Tenacity ライブラリを使った簡潔なリトライ
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type,
)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
)
async def reliable_request(client: AsyncOpenAI, prompt: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return response.choices[0].message.content
```

### 比較表 1: 推論パラメータの用途別推奨設定

| 用途 | temperature | top_p | max_tokens | 備考 |
|------|-----------|-------|-----------|------|
| コード生成 | 0.0-0.2 | 1.0 | 十分大きく | 決定的な出力が望ましい |
| 文章作成 | 0.7-0.9 | 0.95 | 用途に応じて | 多様性と品質のバランス |
| データ抽出 | 0.0 | 1.0 | 必要最小限 | 正確性重視 |
| ブレインストーミング | 1.0-1.2 | 0.95 | 大きめ | 創造性重視 |
| 翻訳 | 0.0-0.3 | 1.0 | 原文の1.5倍程度 | 正確性重視 |
| 要約 | 0.0-0.3 | 1.0 | 原文の1/3程度 | 正確性重視 |

### 比較表 2: 推論最適化手法の比較

| 手法 | レイテンシ改善 | スループット改善 | コスト削減 | 実装難易度 |
|------|-------------|----------------|-----------|-----------|
| ストリーミング | TTFB 大幅改善 | 変わらず | 変わらず | 低 |
| バッチ処理 | 変わらず | 大幅改善 | 50%削減 (OpenAI) | 中 |
| プロンプトキャッシュ | 改善 | 改善 | 最大90%削減 | 低 |
| KV キャッシュ | 改善 | 改善 | 間接的に削減 | 高（ローカルのみ） |
| 量子化 (ローカル) | 大幅改善 | 改善 | GPU削減 | 中〜高 |
| Speculative Decoding | 改善 | 改善 | 間接的に削減 | 高 |

---

## 4. プロンプトキャッシュ

### 4.1 Anthropic Prompt Caching

```python
import anthropic

client = anthropic.Anthropic()

# 長いシステムプロンプトをキャッシュ
long_system = """
あなたは金融分析の専門家です。以下のルールに従ってください:
1. 数値は必ず出典を明記する
2. 推測と事実を明確に区別する
3. リスクファクターを必ず言及する
... (数千トークンの詳細ルール)
"""

# cache_control で明示的にキャッシュ指定
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1000,
    system=[
        {
            "type": "text",
            "text": long_system,
            "cache_control": {"type": "ephemeral"},  # キャッシュ有効化
        }
    ],
    messages=[
        {"role": "user", "content": "今四半期の決算分析をお願いします"},
    ],
)

# キャッシュ利用状況を確認
print(f"入力トークン: {response.usage.input_tokens}")
print(f"キャッシュ作成: {response.usage.cache_creation_input_tokens}")
print(f"キャッシュ利用: {response.usage.cache_read_input_tokens}")

# 2回目以降のリクエストでは cache_read_input_tokens が増加
# キャッシュヒット時は入力トークン料金が 90% OFF
```

### 4.2 OpenAI Automatic Caching

```python
from openai import OpenAI

client = OpenAI()

# OpenAI は共通プレフィックスを自動キャッシュ (1024トークン以上)
long_context = "..." * 2000  # 長いコンテキスト

# 1回目: キャッシュ作成
response1 = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": long_context},
        {"role": "user", "content": "質問1"},
    ],
)
print(f"キャッシュトークン: {response1.usage.prompt_tokens_details.cached_tokens}")

# 2回目: 同じプレフィックスならキャッシュヒット (50% OFF)
response2 = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": long_context},  # 同じ
        {"role": "user", "content": "質問2"},           # 異なる
    ],
)
print(f"キャッシュトークン: {response2.usage.prompt_tokens_details.cached_tokens}")
```

---

## 5. 高度な推論最適化

### 5.1 Speculative Decoding

```
┌──────────────────────────────────────────────────────────┐
│         Speculative Decoding の仕組み                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  通常のデコーディング (Auto-Regressive):                  │
│  大モデル: [t1] → [t2] → [t3] → [t4] → [t5]           │
│  各ステップで大モデルの全計算が必要                        │
│  レイテンシ: 5 × T_large                                 │
│                                                          │
│  Speculative Decoding:                                   │
│  小モデル: [t1, t2, t3, t4, t5] ← 高速に推測生成        │
│  大モデル: [t1, t2, t3, ?, ?]   ← 一括検証              │
│            t1 ✓  t2 ✓  t3 ✓  t4 ✗ (却下)              │
│  大モデル: [t4'] → [t5']        ← 却下箇所から再生成    │
│                                                          │
│  メリット:                                                │
│  - 出力品質は大モデルと完全に同一 (保証あり)             │
│  - 小モデルの推測が当たれば 2-3 倍高速化                 │
│  - GPU メモリの追加消費は小モデル分のみ                  │
│                                                          │
│  デメリット:                                              │
│  - 小モデルの推測精度が低いと効果が薄い                  │
│  - 実装の複雑さ                                          │
│  - バッチ推論では効果が限定的                             │
└──────────────────────────────────────────────────────────┘
```

### 5.2 KV キャッシュの最適化

```python
# vLLM での KV キャッシュ設定
"""
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --enable-prefix-caching \        # プレフィックスキャッシュ有効化
    --max-num-batched-tokens 32768   # バッチトークン数上限
"""

# プレフィックスキャッシュの効果:
# 同じシステムプロンプトを持つリクエストが連続する場合、
# KV キャッシュを再利用して TTFT を大幅短縮

# llama.cpp での KV キャッシュ設定
"""
./llama-server \
    -m model.gguf \
    -c 8192 \          # コンテキスト長
    --cache-type-k q8_0 \  # K キャッシュを 8bit 量子化
    --cache-type-v q8_0 \  # V キャッシュを 8bit 量子化
    -ngl 99                 # 全レイヤー GPU
"""

# KV キャッシュのメモリ計算
def estimate_kv_cache_memory(
    num_layers: int = 32,
    num_heads: int = 32,
    head_dim: int = 128,
    seq_len: int = 8192,
    batch_size: int = 1,
    dtype_bytes: int = 2,  # FP16 = 2 bytes
) -> float:
    """KV キャッシュのメモリ使用量を推定 (GB)"""
    # K と V の両方
    kv_cache_bytes = (
        2 *  # K + V
        num_layers *
        num_heads *
        head_dim *
        seq_len *
        batch_size *
        dtype_bytes
    )
    return kv_cache_bytes / (1024 ** 3)

# Llama 3.1 8B の場合
memory_gb = estimate_kv_cache_memory(
    num_layers=32, num_heads=32, head_dim=128,
    seq_len=8192, batch_size=1, dtype_bytes=2,
)
print(f"KV キャッシュ: {memory_gb:.2f} GB")  # ~2GB
```

### 5.3 Structured Output (構造化出力)

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional

client = OpenAI()

# Pydantic モデルで出力スキーマを定義
class MovieReview(BaseModel):
    title: str
    rating: float
    sentiment: str  # positive / negative / neutral
    key_points: list[str]
    recommendation: bool

# Structured Outputs API (100% スキーマ準拠を保証)
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "映画レビューを分析してください"},
        {"role": "user", "content": "「この映画は最高！演技が素晴らしく、脚本も完璧。必見です。」"},
    ],
    response_format=MovieReview,
)

review = response.choices[0].message.parsed
print(f"タイトル: {review.title}")
print(f"評価: {review.rating}")
print(f"感情: {review.sentiment}")
print(f"推奨: {review.recommendation}")


# Claude での JSON 出力
import anthropic
import json

client_claude = anthropic.Anthropic()

response = client_claude.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1000,
    messages=[{
        "role": "user",
        "content": """以下のテキストから情報を抽出してJSON形式で出力してください。

テキスト: 「田中太郎（35歳）は東京都在住のエンジニアです。年収800万円。」

出力形式:
{"name": "", "age": 0, "location": "", "occupation": "", "income": 0}"""
    }],
)

# Claude は JSON Mode がないため、パース時にエラーハンドリング必要
try:
    data = json.loads(response.content[0].text)
except json.JSONDecodeError:
    # テキスト中から JSON 部分を抽出
    import re
    json_match = re.search(r'\{.*\}', response.content[0].text, re.DOTALL)
    if json_match:
        data = json.loads(json_match.group())
```

---

## 6. パフォーマンス計測とモニタリング

### 6.1 推論メトリクスの計測

```python
import time
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class InferenceMetrics:
    """推論パフォーマンスメトリクス"""
    ttfb: float = 0.0          # Time To First Byte (秒)
    total_time: float = 0.0     # 総応答時間 (秒)
    input_tokens: int = 0       # 入力トークン数
    output_tokens: int = 0      # 出力トークン数
    tokens_per_second: float = 0.0  # 出力速度 (tok/s)
    cost: float = 0.0           # コスト ($)
    model: str = ""
    cached_tokens: int = 0

    def __str__(self):
        return (
            f"Model: {self.model}\n"
            f"TTFB: {self.ttfb:.3f}s\n"
            f"Total: {self.total_time:.3f}s\n"
            f"Input: {self.input_tokens} tokens\n"
            f"Output: {self.output_tokens} tokens\n"
            f"Speed: {self.tokens_per_second:.1f} tok/s\n"
            f"Cache: {self.cached_tokens} tokens\n"
            f"Cost: ${self.cost:.6f}"
        )


async def measure_inference(
    client,
    model: str,
    prompt: str,
    pricing: tuple = (0.15, 0.60),  # (入力$/1M, 出力$/1M)
) -> tuple[str, InferenceMetrics]:
    """推論のパフォーマンスを計測"""
    metrics = InferenceMetrics(model=model)
    start = time.time()
    first_token_time = None
    full_text = ""

    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=1000,
    )

    async for chunk in stream:
        if first_token_time is None and chunk.choices and chunk.choices[0].delta.content:
            first_token_time = time.time()
            metrics.ttfb = first_token_time - start

        if chunk.choices and chunk.choices[0].delta.content:
            full_text += chunk.choices[0].delta.content

        if chunk.usage:
            metrics.input_tokens = chunk.usage.prompt_tokens
            metrics.output_tokens = chunk.usage.completion_tokens
            if hasattr(chunk.usage, "prompt_tokens_details") and chunk.usage.prompt_tokens_details:
                metrics.cached_tokens = chunk.usage.prompt_tokens_details.cached_tokens or 0

    metrics.total_time = time.time() - start
    if metrics.output_tokens > 0 and metrics.total_time > 0:
        metrics.tokens_per_second = metrics.output_tokens / metrics.total_time

    # コスト計算
    in_price, out_price = pricing
    metrics.cost = (
        (metrics.input_tokens / 1_000_000) * in_price +
        (metrics.output_tokens / 1_000_000) * out_price
    )

    return full_text, metrics
```

### 6.2 ダッシュボード用集計

```python
from collections import defaultdict
import statistics

class InferenceMonitor:
    """推論パフォーマンスの継続モニタリング"""

    def __init__(self):
        self.metrics_history: list[InferenceMetrics] = []

    def record(self, metrics: InferenceMetrics):
        self.metrics_history.append(metrics)

    def summary(self, last_n: int = 100) -> dict:
        """直近 N 件の集計サマリー"""
        recent = self.metrics_history[-last_n:]

        if not recent:
            return {}

        return {
            "total_requests": len(recent),
            "avg_ttfb": statistics.mean(m.ttfb for m in recent),
            "p50_ttfb": statistics.median(m.ttfb for m in recent),
            "p95_ttfb": sorted(m.ttfb for m in recent)[int(len(recent) * 0.95)],
            "avg_total_time": statistics.mean(m.total_time for m in recent),
            "avg_tokens_per_second": statistics.mean(m.tokens_per_second for m in recent),
            "total_input_tokens": sum(m.input_tokens for m in recent),
            "total_output_tokens": sum(m.output_tokens for m in recent),
            "total_cost": sum(m.cost for m in recent),
            "avg_cost_per_request": statistics.mean(m.cost for m in recent),
            "cache_hit_rate": (
                sum(m.cached_tokens for m in recent) /
                max(sum(m.input_tokens for m in recent), 1)
            ),
        }

    def print_report(self):
        """レポート出力"""
        s = self.summary()
        if not s:
            print("データなし")
            return

        print("=== 推論パフォーマンスレポート ===")
        print(f"リクエスト数: {s['total_requests']}")
        print(f"TTFB (avg/p50/p95): {s['avg_ttfb']:.3f}s / "
              f"{s['p50_ttfb']:.3f}s / {s['p95_ttfb']:.3f}s")
        print(f"総応答時間 (avg): {s['avg_total_time']:.3f}s")
        print(f"出力速度 (avg): {s['avg_tokens_per_second']:.1f} tok/s")
        print(f"トークン合計: IN={s['total_input_tokens']:,} / OUT={s['total_output_tokens']:,}")
        print(f"総コスト: ${s['total_cost']:.4f}")
        print(f"リクエスト単価: ${s['avg_cost_per_request']:.6f}")
        print(f"キャッシュヒット率: {s['cache_hit_rate']:.1%}")
```

---

## 7. トラブルシューティング

### 7.1 よくある問題と対処法

```
┌──────────────────────────────────────────────────────────┐
│         推論トラブルシューティングガイド                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  問題 1: 出力が途中で切れる                                │
│  原因: max_tokens が不足                                  │
│  対処:                                                    │
│  - finish_reason を確認 ("length" なら切れている)         │
│  - max_tokens を増やす                                   │
│  - 出力が長い場合は分割リクエストを検討                    │
│                                                          │
│  問題 2: レスポンスが遅い                                  │
│  原因: モデルサイズ、入力長、サーバー負荷                  │
│  対処:                                                    │
│  - Flash/mini 系モデルに切り替え                          │
│  - 入力プロンプトを短縮                                   │
│  - ストリーミングで TTFB を改善                           │
│  - プロンプトキャッシュを有効化                            │
│                                                          │
│  問題 3: 429 Too Many Requests                           │
│  原因: レート制限に到達                                    │
│  対処:                                                    │
│  - 指数バックオフ付きリトライ                              │
│  - Retry-After ヘッダに従う                               │
│  - 並列数を減らす                                        │
│  - バッチ API に切り替え                                  │
│  - 利用枠の増加を申請                                     │
│                                                          │
│  問題 4: 出力が毎回異なる (再現性の欠如)                   │
│  原因: サンプリングのランダム性                             │
│  対処:                                                    │
│  - temperature=0 に設定                                   │
│  - seed パラメータを指定                                  │
│  - system_fingerprint をログに記録                        │
│  - モデルバージョンを固定                                 │
│                                                          │
│  問題 5: JSON 出力が壊れる                                │
│  原因: 非構造化出力の不安定性                              │
│  対処:                                                    │
│  - response_format: json_object を指定 (OpenAI)          │
│  - Structured Outputs API を使用                         │
│  - 出力パース時にエラーハンドリング                        │
│  - リトライ + バリデーション                               │
└──────────────────────────────────────────────────────────┘
```

### 7.2 デバッグ用コード

```python
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")

def debug_response(response, model: str = "unknown"):
    """レスポンスの詳細デバッグ情報を出力"""

    choice = response.choices[0]

    logger.info(f"=== {model} レスポンスデバッグ ===")
    logger.info(f"finish_reason: {choice.finish_reason}")
    logger.info(f"input_tokens: {response.usage.prompt_tokens}")
    logger.info(f"output_tokens: {response.usage.completion_tokens}")
    logger.info(f"total_tokens: {response.usage.total_tokens}")

    if hasattr(response.usage, "prompt_tokens_details") and response.usage.prompt_tokens_details:
        details = response.usage.prompt_tokens_details
        logger.info(f"cached_tokens: {details.cached_tokens}")

    if hasattr(response, "system_fingerprint"):
        logger.info(f"system_fingerprint: {response.system_fingerprint}")

    # 出力の最初と最後を表示
    content = choice.message.content
    if content:
        logger.info(f"output_length: {len(content)} chars")
        logger.info(f"first_100: {content[:100]}")
        logger.info(f"last_100: {content[-100:]}")

    # finish_reason の診断
    if choice.finish_reason == "length":
        logger.warning("出力が max_tokens で切れています。値を増やしてください。")
    elif choice.finish_reason == "content_filter":
        logger.warning("コンテンツフィルタにより出力がブロックされました。")

    return {
        "finish_reason": choice.finish_reason,
        "tokens": response.usage.total_tokens,
        "content_length": len(content) if content else 0,
    }
```

---

## アンチパターン

### アンチパターン 1: temperature と top_p の同時変更

```
誤: 両方を極端に設定
  temperature=0.2, top_p=0.3
  → 予測困難な挙動、過度に制約された出力

正: 片方を固定し、もう片方だけ調整
  temperature=0.7, top_p=1.0  # temperature のみ調整
  temperature=1.0, top_p=0.8  # top_p のみ調整
```

### アンチパターン 2: max_tokens を常に最大値に設定

```
誤: max_tokens=4096 を全リクエストに設定
  → 不要な長文生成、コスト増大、レイテンシ増加

正: タスクに応じた適切な上限設定
  - 分類: max_tokens=10
  - 要約: max_tokens=500
  - コード生成: max_tokens=2000
  - 長文作成: max_tokens=4000
```

### アンチパターン 3: エラーハンドリングなしの API 呼び出し

```python
# NG: エラーハンドリングなし
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
)
result = response.choices[0].message.content

# OK: 包括的なエラーハンドリング
try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        timeout=30.0,
    )

    if response.choices[0].finish_reason == "length":
        logger.warning("出力が切れています")
    elif response.choices[0].finish_reason == "content_filter":
        logger.warning("コンテンツフィルタ発動")

    result = response.choices[0].message.content

except RateLimitError:
    # リトライロジック
    pass
except APITimeoutError:
    # タイムアウト処理
    pass
except APIError as e:
    logger.error(f"API エラー: {e.status_code} - {e.message}")
```

### アンチパターン 4: ストリーミング応答の未処理中断

```python
# NG: ストリーミング中のリソースリーク
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    stream=True,
)
for chunk in stream:
    if some_condition:
        break  # ストリームが適切にクローズされない可能性

# OK: コンテキストマネージャーで確実にクリーンアップ
with client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    stream=True,
) as stream:
    for chunk in stream:
        if some_condition:
            break  # __exit__ で確実にクリーンアップ
```

---

## FAQ

### Q1: temperature=0 でも同じプロンプトで異なる結果が出ることがありますか？

**A:** はい、あります。GPU の浮動小数点演算の非決定性や、バッチ処理の影響で、temperature=0 でも完全に同一の結果は保証されません。OpenAI では `seed` パラメータで再現性を高めることができますが、100%の保証はありません。

### Q2: ストリーミングを使うとコストは変わりますか？

**A:** いいえ、トークン消費量（コスト）は同じです。ストリーミングは応答の配信方法が異なるだけで、生成されるトークン数は変わりません。ただし、ストリーミングでは接続が長時間維持されるため、サーバーリソースの消費パターンが異なります。

### Q3: バッチ処理はいつ使うべきですか？

**A:** リアルタイム応答が不要な大量処理（数百〜数万リクエスト）に最適です。例えば、大量のドキュメント分類、データセットのラベリング、コンテンツ生成などです。OpenAI の Batch API は 50% 割引、Anthropic のバッチ API も同様の割引があり、24時間以内に結果が返されます。

### Q4: プロンプトキャッシュの効果はどのくらいですか？

**A:** Anthropic のプロンプトキャッシュでは、キャッシュヒット時に入力トークンの料金が 90% 削減されます。OpenAI では 50% 削減です。長いシステムプロンプトや RAG コンテキストを繰り返し使用する場合に特に効果的で、月間コストを 40-70% 削減できるケースがあります。キャッシュの有効期限は Anthropic が 5 分、OpenAI が自動管理です。

### Q5: Logprobs は何に使えますか？

**A:** 主に以下の用途があります。(1) 信頼度スコアの計算 -- 各トークンの確率から生成全体の信頼度を推定、(2) 自動フォールバック -- 低信頼度の場合に別モデルに切り替え、(3) 分類タスクの確率推定 -- Yes/No の確率を直接取得、(4) ハルシネーション検出 -- 低確率トークンが多い箇所を特定。ただし、Claude API では logprobs は提供されていません。

### Q6: Structured Outputs と JSON Mode の違いは？

**A:** JSON Mode (response_format: json_object) は「JSON 形式であること」のみを保証しますが、スキーマの準拠は保証しません。Structured Outputs は Pydantic モデル等で定義したスキーマに 100% 準拠した出力を保証します。重要なデータ抽出には Structured Outputs を使い、柔軟な JSON 出力には JSON Mode を使うのが推奨です。

---

## まとめ

| 項目 | 要点 |
|------|------|
| temperature | 0.0（決定的）〜 1.5（高多様性）で出力のランダム性を制御 |
| top_p | 累積確率で候補トークンをフィルタリング |
| ストリーミング | TTFB を大幅短縮、UX 向上に必須 |
| バッチ処理 | 大量リクエストの並列・非同期処理でコスト削減 |
| max_tokens | タスクに応じた適切な設定でコスト最適化 |
| プロンプトキャッシュ | 繰り返しプレフィックスで最大 90% コスト削減 |
| Logprobs | 信頼度推定、フォールバック判断に活用 |
| Structured Outputs | スキーマ準拠の構造化出力を保証 |
| 推論最適化 | キャッシュ・量子化・バッチの組み合わせが効果的 |
| エラーハンドリング | 指数バックオフ、タイムアウト、リトライが必須 |

---

## 次に読むべきガイド

- [03-fine-tuning.md](./03-fine-tuning.md) — ファインチューニングによるモデルのカスタマイズ
- [../02-applications/00-prompt-engineering.md](../02-applications/00-prompt-engineering.md) — プロンプト設計の技法
- [../03-infrastructure/00-api-integration.md](../03-infrastructure/00-api-integration.md) — API 統合の実践

---

## 参考文献

1. Holtzman, A. et al. (2020). "The Curious Case of Neural Text Degeneration." *ICLR 2020*. https://arxiv.org/abs/1904.09751
2. Anthropic. "Messages API Reference." https://docs.anthropic.com/en/api/messages
3. OpenAI. "Chat Completions API." https://platform.openai.com/docs/api-reference/chat
4. Leviathan, Y. et al. (2023). "Fast Inference from Transformers via Speculative Decoding." *ICML 2023*. https://arxiv.org/abs/2211.17192
5. OpenAI. "Structured Outputs Guide." https://platform.openai.com/docs/guides/structured-outputs
6. Anthropic. "Prompt Caching." https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching

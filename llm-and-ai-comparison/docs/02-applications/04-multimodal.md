# マルチモーダル — 画像・音声・動画入力と Vision API

> マルチモーダル AI はテキスト・画像・音声・動画など複数の情報様式 (モダリティ) を統合的に処理する技術であり、現実世界の多様な入力を理解・生成する LLM の最新進化形である。

## この章で学ぶこと

1. **マルチモーダル LLM の仕組みと対応状況** — Vision、Audio、Video の処理アーキテクチャ
2. **画像入力の実践的活用** — OCR、図表解析、UI 理解、画像分類
3. **音声・動画の処理と応用** — 音声文字起こし、動画要約、リアルタイム処理

---

## 1. マルチモーダル LLM の全体像

```
┌──────────────────────────────────────────────────────────┐
│          マルチモーダル LLM のアーキテクチャ                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  入力モダリティ      エンコーダ         言語モデル         │
│                                                          │
│  ┌──────┐          ┌──────────┐                         │
│  │ 画像 │ ──────▶ │ Vision   │                         │
│  └──────┘          │ Encoder  │──┐                      │
│                    │ (ViT)    │  │                      │
│  ┌──────┐          └──────────┘  │  ┌──────────────┐   │
│  │ 音声 │ ──────▶ ┌──────────┐  ├─▶│  Transformer  │   │
│  └──────┘          │ Audio    │  │  │  Decoder     │   │
│                    │ Encoder  │──┤  │  (LLM本体)   │   │
│  ┌──────┐          │(Whisper) │  │  └──────┬───────┘   │
│  │ 動画 │ ──────▶ └──────────┘  │         │           │
│  └──────┘          ┌──────────┐  │         ▼           │
│   (フレーム抽出)    │ Video    │──┘  テキスト出力       │
│                    │ Encoder  │      画像生成          │
│  ┌──────┐          └──────────┘      音声生成          │
│  │テキスト│ ──────────────────────┘                     │
│  └──────┘                                              │
│                                                          │
│  統合方式:                                                │
│  A) アーリーフュージョン: 入力段階で統合 (Gemini)         │
│  B) レイトフュージョン: 各エンコーダ後に統合 (GPT-4V)     │
│  C) クロスアテンション: LLM層で融合 (Flamingo)           │
└──────────────────────────────────────────────────────────┘
```

### 1.1 プロバイダ別マルチモーダル対応状況

| モダリティ | GPT-4o | Claude 3.5 | Gemini 1.5 | Qwen-VL | Llama 3.2 |
|-----------|--------|-----------|-----------|---------|----------|
| 画像入力 | S | S | S | S | S (11B+) |
| 画像生成 | S | N/A | S | N/A | N/A |
| 音声入力 | S | N/A | S | S (Audio) | N/A |
| 音声出力 | S | N/A | S | N/A | N/A |
| 動画入力 | N/A | N/A | S | N/A | N/A |
| PDF入力 | 画像化 | 直接対応 | 直接対応 | N/A | N/A |

*S=優秀, N/A=未対応*

---

## 2. 画像入力の実践

### 2.1 OpenAI Vision API

```python
from openai import OpenAI
import base64

client = OpenAI()

# 方法1: URL指定
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "この画像の内容を詳しく説明してください。"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                        "detail": "high",  # low / auto / high
                    },
                },
            ],
        }
    ],
    max_tokens=1024,
)
print(response.choices[0].message.content)

# 方法2: Base64エンコード (ローカル画像)
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

image_b64 = encode_image("screenshot.png")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "このスクリーンショットのUIの改善点を指摘してください。"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                },
            ],
        }
    ],
)
```

### 2.2 Claude Vision

```python
from anthropic import Anthropic
import base64

client = Anthropic()

with open("diagram.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {
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
                    "text": "このアーキテクチャ図を解説し、ボトルネックを特定してください。",
                },
            ],
        }
    ],
)
print(response.content[0].text)
```

### 2.3 Gemini マルチモーダル

```python
import google.generativeai as genai
from pathlib import Path

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-pro")

# 画像 + テキスト
image = genai.upload_file(Path("chart.png"))
response = model.generate_content([
    image,
    "このグラフのトレンドを分析し、来月の予測を行ってください。",
])
print(response.text)
```

---

## 3. 実践的な画像活用パターン

### 3.1 OCR + 構造化抽出

```python
async def extract_receipt(image_path: str) -> dict:
    """レシート画像から構造化データを抽出"""
    image_b64 = encode_image(image_path)

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": """
このレシート画像から以下の情報をJSON形式で抽出してください:
{
  "store_name": "店名",
  "date": "YYYY-MM-DD",
  "items": [{"name": "商品名", "quantity": 数量, "price": 価格}],
  "subtotal": 小計,
  "tax": 消費税,
  "total": 合計
}
""",
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)
```

### 3.2 複数画像の比較分析

```python
def compare_images(image_paths: list[str], comparison_prompt: str) -> str:
    """複数画像の比較分析"""
    content = []

    for i, path in enumerate(image_paths):
        content.append({
            "type": "text",
            "text": f"画像{i+1}:",
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image(path)}"},
        })

    content.append({"type": "text", "text": comparison_prompt})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content

# 使用例: UIのビフォーアフター比較
result = compare_images(
    ["design_v1.png", "design_v2.png"],
    "2つのUIデザインを比較し、v2の改善点と残課題を分析してください。"
)
```

---

## 4. 音声処理

### 4.1 Whisper (音声→テキスト)

```python
from openai import OpenAI

client = OpenAI()

# 音声文字起こし
with open("meeting.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="ja",
        response_format="verbose_json",  # タイムスタンプ付き
    )

for segment in transcript.segments:
    start = segment["start"]
    end = segment["end"]
    text = segment["text"]
    print(f"[{start:.1f}s - {end:.1f}s] {text}")
```

### 4.2 GPT-4o リアルタイム音声

```python
# GPT-4o Realtime API (WebSocket)
import asyncio
import websockets
import json

async def realtime_voice_chat():
    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with websockets.connect(url, extra_headers=headers) as ws:
        # セッション設定
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": "あなたは日本語の会話アシスタントです。",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
            },
        }))

        # 音声データを送信・受信するループ
        # (実際にはマイク入力とスピーカー出力の処理が必要)
```

---

## 5. 動画処理 (Gemini)

### 5.1 動画入力と分析

```python
import google.generativeai as genai
import time

genai.configure(api_key="YOUR_API_KEY")

# 動画アップロード
video_file = genai.upload_file("presentation.mp4")

# アップロード完了まで待機
while video_file.state.name == "PROCESSING":
    time.sleep(5)
    video_file = genai.get_file(video_file.name)

# 動画分析
model = genai.GenerativeModel("gemini-1.5-pro")
response = model.generate_content([
    video_file,
    """
この動画を分析して以下をレポートしてください:
1. 動画の概要 (30秒以内の要約)
2. 主要なトピック (タイムスタンプ付き)
3. 発表者の主張のまとめ
4. 改善提案
""",
])
print(response.text)
```

```
┌──────────────────────────────────────────────────────────┐
│          動画処理のアーキテクチャ                           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  動画ファイル                                             │
│    │                                                     │
│    ├──▶ フレーム抽出 (1fps等)                             │
│    │     └──▶ Vision Encoder ──┐                         │
│    │                           │                         │
│    ├──▶ 音声トラック分離        │     ┌──────────────┐   │
│    │     └──▶ Audio Encoder ──┼──▶ │  LLM 統合     │   │
│    │                           │     │  (Gemini)     │   │
│    └──▶ 字幕/メタデータ ───────┘     └──────┬───────┘   │
│                                              │           │
│                                        テキスト出力       │
│                                                          │
│  制約:                                                    │
│  - Gemini: 最大1時間の動画                                │
│  - フレーム数とトークン数は比例                            │
│  - 1分の動画 ≈ 数千トークン消費                           │
└──────────────────────────────────────────────────────────┘
```

---

## 6. マルチモーダル Embedding

```python
# CLIP ベースのマルチモーダル Embedding
from sentence_transformers import SentenceTransformer
from PIL import Image

model = SentenceTransformer("clip-ViT-B-32")

# テキストと画像を同一ベクトル空間に埋め込み
text_embeddings = model.encode(["猫の写真", "東京タワー", "プログラミング"])
image_embedding = model.encode(Image.open("cat.jpg"))

# テキスト↔画像のクロスモーダル検索が可能
from sentence_transformers.util import cos_sim
similarities = cos_sim(image_embedding, text_embeddings)
print(similarities)  # "猫の写真" が最も高スコア
```

---

## 7. 比較表

### 7.1 マルチモーダルモデル性能比較

| ベンチマーク | GPT-4o | Claude 3.5 | Gemini 1.5 Pro | Qwen-VL-Max |
|-------------|--------|-----------|---------------|-------------|
| MMMU (マルチモーダル理解) | 69.1 | 68.3 | 62.2 | 51.4 |
| MathVista (数学+視覚) | 63.8 | 61.6 | 58.0 | 51.0 |
| ChartQA (グラフ理解) | 85.7 | 90.8 | 81.3 | 79.8 |
| DocVQA (文書理解) | 92.8 | 95.2 | 93.1 | 93.8 |
| TextVQA (画像内テキスト) | 77.4 | - | 73.5 | 79.5 |

### 7.2 画像処理コスト比較

| モデル | 低解像度 | 高解像度 | 最大画像数 |
|--------|---------|---------|-----------|
| GPT-4o | 85 tokens | ~1,105 tokens | 制限なし |
| Claude 3.5 | ~1,600 tokens | ~1,600 tokens | 20枚 |
| Gemini 1.5 | 258 tokens | 258 tokens | 3,600枚 |

---

## 8. アンチパターン

### アンチパターン 1: 不必要に高解像度の画像を送信

```python
# NG: 4K画像をそのまま送信 → トークン消費大、コスト増
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{huge_4k_image}",
                "detail": "high",
            }},
            {"type": "text", "text": "何が写っていますか？"},
        ],
    }],
)

# OK: タスクに応じた解像度選択
from PIL import Image

def optimize_image(path: str, max_size: int = 1024) -> str:
    img = Image.open(path)
    img.thumbnail((max_size, max_size))
    # リサイズ後にBase64エンコード
    return encode_image_from_pil(img)

# 簡単な質問なら low detail で十分
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{optimized}",
                "detail": "low",  # 85 tokens固定
            }},
            {"type": "text", "text": "何が写っていますか？"},
        ],
    }],
)
```

### アンチパターン 2: 画像のハルシネーションを無検証で利用

```python
# NG: OCR結果を無検証で使用
receipt_data = extract_receipt(image)
process_payment(receipt_data["total"])  # 金額が間違っている可能性

# OK: 重要な数値は確認ステップを挟む
receipt_data = extract_receipt(image)
# 2回目の抽出で照合 (Self-Consistency)
receipt_data_2 = extract_receipt(image)

if receipt_data["total"] != receipt_data_2["total"]:
    # 不一致の場合は人間に確認を求める
    flag_for_human_review(receipt_data, receipt_data_2)
```

---

## 9. FAQ

### Q1: 画像入力のトークンコストはどう計算される?

GPT-4o: low detail = 85 tokens 固定、high detail = タイルサイズに依存 (512x512 タイルごとに 170 tokens + 基本 85 tokens)。
Claude: 画像サイズに応じて約 1,600 tokens。
Gemini: 約 258 tokens/画像 (固定に近い)。
コスト最適化するなら、まず low detail で試し、精度不足なら high detail に上げる。

### Q2: 動画分析ができるのは Gemini だけ?

2025年時点で、ネイティブ動画入力に対応しているのは Gemini 1.5 のみ。
GPT-4o / Claude ではフレーム抽出 (1-2fps) + 個別画像入力で疑似的に対応可能。
ただし、音声トラックの同時処理は Gemini でないとできない。

### Q3: マルチモーダル AI の精度はテキスト専用モデルに比べてどうか?

テキストタスクではテキスト専用モデルとほぼ同等。
画像+テキストの複合タスクではマルチモーダルモデルが圧倒的に有利。
ただし、画像内の細かいテキスト (小さな文字) や複雑な図表の読み取りは OCR 専用ツールに劣る場合がある。

---

## まとめ

| 項目 | 内容 |
|------|------|
| 画像入力推奨 | GPT-4o (汎用)、Claude 3.5 (文書)、Gemini (大量画像) |
| 音声処理 | Whisper (文字起こし)、GPT-4o Realtime (対話) |
| 動画処理 | Gemini 1.5 Pro (唯一のネイティブ対応) |
| コスト最適化 | 解像度を下げる、タスクに応じた detail 設定 |
| 精度最適化 | 画像前処理、Self-Consistency、専用OCRとの併用 |
| 今後の展望 | リアルタイム映像理解、空間認識、生成品質の向上 |

---

## 次に読むべきガイド

- [03-embeddings.md](./03-embeddings.md) — マルチモーダル Embedding の詳細
- [../01-models/02-gemini.md](../01-models/02-gemini.md) — Gemini のマルチモーダル機能
- [../03-infrastructure/00-api-integration.md](../03-infrastructure/00-api-integration.md) — マルチモーダル API の統合

---

## 参考文献

1. OpenAI, "Vision Guide," https://platform.openai.com/docs/guides/vision
2. Anthropic, "Vision Documentation," https://docs.anthropic.com/claude/docs/vision
3. Google, "Gemini Multimodal Guide," https://ai.google.dev/docs/multimodal_concepts
4. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision (CLIP)," ICML 2021

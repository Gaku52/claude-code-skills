# マルチモーダル — 画像・音声・動画入力と Vision API

> マルチモーダル AI はテキスト・画像・音声・動画など複数の情報様式 (モダリティ) を統合的に処理する技術であり、現実世界の多様な入力を理解・生成する LLM の最新進化形である。

## この章で学ぶこと

1. **マルチモーダル LLM の仕組みと対応状況** — Vision、Audio、Video の処理アーキテクチャ
2. **画像入力の実践的活用** — OCR、図表解析、UI 理解、画像分類
3. **音声・動画の処理と応用** — 音声文字起こし、動画要約、リアルタイム処理
4. **マルチモーダル RAG と Embedding** — CLIP、画像検索、クロスモーダル応用
5. **本番運用のパターンとコスト最適化** — 画像前処理、バッチ処理、品質保証

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

| モダリティ | GPT-4o | Claude 3.5/4 | Gemini 2.0 | Qwen-VL | Llama 3.2 |
|-----------|--------|-------------|-----------|---------|----------|
| 画像入力 | S | S | S | S | S (11B+) |
| 画像生成 | S | N/A | S | N/A | N/A |
| 音声入力 | S | N/A | S | S (Audio) | N/A |
| 音声出力 | S | N/A | S | N/A | N/A |
| 動画入力 | N/A | N/A | S | N/A | N/A |
| PDF入力 | 画像化 | 直接対応 | 直接対応 | N/A | N/A |

*S=対応, N/A=未対応*

### 1.2 モダリティ別のトークン消費

```
┌─────────────────────────────────────────────────────────────────┐
│              モダリティ別トークン消費の目安                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  テキスト                                                       │
│  └── 日本語: 約1文字 = 1-3トークン                               │
│                                                                 │
│  画像 (GPT-4o)                                                  │
│  ├── low detail: 85 tokens (固定)                               │
│  ├── high detail: 85 + 170 * タイル数                           │
│  │   └── 512x512 = 255 tokens                                  │
│  │   └── 1024x1024 = 765 tokens                                │
│  │   └── 2048x2048 = 1,105 tokens                              │
│  └── auto: モデルが自動選択                                     │
│                                                                 │
│  画像 (Claude)                                                  │
│  ├── ~1,600 tokens/画像 (サイズによらずほぼ固定)                  │
│  └── 最大解像度: 1,568 x 1,568 px                               │
│                                                                 │
│  画像 (Gemini)                                                  │
│  ├── 258 tokens/画像 (ほぼ固定)                                  │
│  └── 最大3,600画像/リクエスト                                    │
│                                                                 │
│  音声 (Whisper)                                                 │
│  └── 入力: 分あたり課金 ($0.006/min)                             │
│                                                                 │
│  動画 (Gemini)                                                  │
│  └── 1秒あたり約1,000トークン                                    │
│  └── 1分の動画 ≈ 60,000 トークン                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

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
    model="claude-sonnet-4-20250514",
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

### 2.3 Claude PDF 直接入力

```python
from anthropic import Anthropic
import base64

client = Anthropic()

# Claude は PDF を直接処理可能
with open("report.pdf", "rb") as f:
    pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_data,
                    },
                },
                {
                    "type": "text",
                    "text": """このPDFレポートを分析し、以下を抽出してください:
1. エグゼクティブサマリー
2. 主要な数値データ (表形式で)
3. 結論と推奨事項
4. リスク要因""",
                },
            ],
        }
    ],
)
```

### 2.4 Gemini マルチモーダル

```python
import google.generativeai as genai
from pathlib import Path

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-2.0-flash")

# 画像 + テキスト
image = genai.upload_file(Path("chart.png"))
response = model.generate_content([
    image,
    "このグラフのトレンドを分析し、来月の予測を行ってください。",
])
print(response.text)

# 複数画像の一括処理 (Gemini の強み)
images = [genai.upload_file(Path(f"slide_{i}.png")) for i in range(1, 21)]
response = model.generate_content([
    *images,
    "これら20枚のスライドの内容を要約してください。各スライドのポイントを箇条書きで。",
])
```

---

## 3. 実践的な画像活用パターン

### 3.1 OCR + 構造化抽出

```python
import json
from openai import OpenAI

client = OpenAI()

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


async def extract_business_card(image_path: str) -> dict:
    """名刺画像から構造化データを抽出"""
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
                        "text": """この名刺から以下の情報をJSON形式で抽出してください。
読み取れない項目はnullにしてください:
{
  "company_name": "会社名",
  "department": "部署",
  "title": "役職",
  "name": "氏名",
  "name_reading": "ふりがな",
  "email": "メールアドレス",
  "phone": "電話番号",
  "mobile": "携帯番号",
  "address": "住所",
  "website": "URL"
}""",
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

### 3.3 図表・グラフの解析

```python
async def analyze_chart(image_path: str) -> dict:
    """グラフ・チャートの詳細分析"""
    image_b64 = encode_image(image_path)

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high",  # グラフは高解像度推奨
                        },
                    },
                    {
                        "type": "text",
                        "text": """このグラフ/チャートを詳細に分析してください。
以下をJSON形式で返してください:
{
  "chart_type": "グラフの種類 (棒グラフ、折れ線グラフ等)",
  "title": "タイトル",
  "x_axis": "X軸ラベル",
  "y_axis": "Y軸ラベル",
  "data_points": [{"label": "ラベル", "value": 数値}],
  "trend": "全体的なトレンドの説明",
  "key_findings": ["主要な発見1", "主要な発見2"],
  "anomalies": ["異常値や注目すべき点"]
}""",
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)
```

### 3.4 UI/UX 分析

```python
async def analyze_ui(screenshot_path: str) -> dict:
    """UIスクリーンショットの分析"""
    image_b64 = encode_image(screenshot_path)

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "あなたはUI/UXの専門家です。ヒューリスティック評価の観点からUIを分析してください。",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"},
                    },
                    {
                        "type": "text",
                        "text": """このUIを以下の観点で評価してください (各5点満点):

1. 視認性 (Visibility of system status)
2. 一貫性 (Consistency and standards)
3. エラー防止 (Error prevention)
4. 効率性 (Flexibility and efficiency)
5. ミニマルデザイン (Aesthetic and minimalist design)
6. アクセシビリティ (Accessibility)

各観点でスコアと改善提案をJSON形式で返してください。""",
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)
```

### 3.5 画像分類パイプライン

```python
from enum import Enum

class ContentCategory(Enum):
    DOCUMENT = "document"       # 文書・テキスト
    CHART = "chart"             # グラフ・チャート
    PHOTO = "photo"             # 写真
    SCREENSHOT = "screenshot"   # スクリーンショット
    DIAGRAM = "diagram"         # 図表・フローチャート
    HANDWRITING = "handwriting" # 手書き
    OTHER = "other"

async def classify_and_process(image_path: str) -> dict:
    """画像を分類し、種類に応じた処理を実行"""

    # Step 1: 画像の種類を分類
    image_b64 = encode_image(image_path)
    classification = await client.chat.completions.create(
        model="gpt-4o-mini",  # 分類は軽量モデルで十分
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "low"}},
                {"type": "text", "text": f"この画像の種類を以下から1つ選んでください: {[e.value for e in ContentCategory]}"},
            ],
        }],
    )

    category = classification.choices[0].message.content.strip().lower()

    # Step 2: カテゴリに応じた処理
    processors = {
        "document": extract_document_text,
        "chart": analyze_chart,
        "photo": describe_photo,
        "screenshot": analyze_ui,
        "diagram": analyze_diagram,
        "handwriting": extract_handwriting,
    }

    processor = processors.get(category, describe_photo)
    result = await processor(image_path)

    return {
        "category": category,
        "analysis": result,
    }
```

---

## 4. 音声処理

### 4.1 Whisper (音声→テキスト)

```python
from openai import OpenAI

client = OpenAI()

# 基本的な音声文字起こし
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

### 4.2 長時間音声の分割処理

```python
from pydub import AudioSegment
import tempfile
import os

async def transcribe_long_audio(
    audio_path: str,
    chunk_duration_ms: int = 10 * 60 * 1000,  # 10分ごとに分割
    overlap_ms: int = 5000,  # 5秒のオーバーラップ
) -> list[dict]:
    """長時間音声の分割文字起こし"""

    audio = AudioSegment.from_file(audio_path)
    total_duration = len(audio)
    chunks = []

    start = 0
    while start < total_duration:
        end = min(start + chunk_duration_ms, total_duration)
        chunk = audio[start:end]

        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            chunk.export(tmp.name, format="mp3")
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="ja",
                    response_format="verbose_json",
                )

            # タイムスタンプをオフセット調整
            for segment in result.segments:
                segment["start"] += start / 1000
                segment["end"] += start / 1000

            chunks.append({
                "start_time": start / 1000,
                "end_time": end / 1000,
                "segments": result.segments,
                "text": result.text,
            })
        finally:
            os.unlink(tmp_path)

        start = end - overlap_ms  # オーバーラップ付きで次のチャンクへ

    return chunks


async def transcribe_with_speaker_diarization(audio_path: str) -> list[dict]:
    """話者分離付き文字起こし"""

    # Step 1: Whisper で文字起こし
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="ja",
            response_format="verbose_json",
        )

    # Step 2: LLM で話者分離 (簡易的な方法)
    full_text = transcript.text
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""以下の会話のテキストを話者分離してください。
話者が変わる箇所を推定し、以下の形式で返してください:

話者A: ...
話者B: ...
話者A: ...

テキスト:
{full_text}""",
        }],
    )

    return response.choices[0].message.content
```

### 4.3 GPT-4o リアルタイム音声

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

### 4.4 テキスト→音声 (TTS)

```python
from openai import OpenAI
from pathlib import Path

client = OpenAI()

# 基本的な音声生成
response = client.audio.speech.create(
    model="tts-1-hd",     # tts-1 (低品質/低コスト) or tts-1-hd (高品質)
    voice="nova",          # alloy, echo, fable, onyx, nova, shimmer
    input="こんにちは。本日のプレゼンテーションを始めます。",
    speed=1.0,             # 0.25 - 4.0
)

# ファイルに保存
speech_file = Path("output.mp3")
response.stream_to_file(speech_file)

# ストリーミング再生
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="ストリーミングで音声を生成します。",
)

# 音声データをチャンクごとに受信
with open("stream_output.mp3", "wb") as f:
    for chunk in response.iter_bytes(chunk_size=1024):
        f.write(chunk)
```

### 4.5 音声アプリケーション: 議事録自動生成

```python
async def generate_meeting_minutes(audio_path: str) -> dict:
    """会議音声から議事録を自動生成"""

    # Step 1: 文字起こし
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="ja",
            response_format="verbose_json",
        )

    # Step 2: LLM で議事録生成
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "あなたは議事録作成の専門家です。会議の文字起こしから構造化された議事録を作成してください。",
            },
            {
                "role": "user",
                "content": f"""以下の会議の文字起こしから議事録を作成してください。

文字起こし:
{transcript.text}

以下の形式でJSON出力してください:
{{
  "meeting_title": "会議タイトル (推定)",
  "date": "日付 (推定)",
  "participants": ["参加者 (推定)"],
  "agenda": ["議題1", "議題2"],
  "discussion_points": [
    {{
      "topic": "議題",
      "key_points": ["要点1", "要点2"],
      "decisions": ["決定事項"],
      "action_items": [
        {{
          "task": "タスク内容",
          "assignee": "担当者 (推定)",
          "deadline": "期限 (推定)"
        }}
      ]
    }}
  ],
  "next_steps": ["次のステップ"]
}}""",
            },
        ],
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)
```

---

## 5. 動画処理

### 5.1 Gemini 動画入力と分析

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
model = genai.GenerativeModel("gemini-2.0-flash")
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

### 5.2 フレーム抽出による疑似動画分析

```python
import cv2
import base64
from openai import OpenAI

client = OpenAI()

def extract_frames(video_path: str, fps: float = 1.0) -> list[str]:
    """動画からフレームを抽出してBase64エンコード"""
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # フレームをリサイズ
            frame = cv2.resize(frame, (512, 512))
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64 = base64.b64encode(buffer).decode("utf-8")
            frames.append(b64)

        frame_count += 1

    cap.release()
    return frames


async def analyze_video_with_frames(
    video_path: str,
    prompt: str,
    fps: float = 0.5,  # 0.5fps = 2秒に1フレーム
    max_frames: int = 20,
) -> str:
    """フレーム抽出 + Vision API で動画分析"""

    frames = extract_frames(video_path, fps=fps)[:max_frames]

    content = [{"type": "text", "text": f"以下は動画から {fps}fps で抽出したフレームです。\n\n{prompt}"}]

    for i, frame in enumerate(frames):
        timestamp = i / fps
        content.append({"type": "text", "text": f"[{timestamp:.1f}秒]:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame}", "detail": "low"},
        })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=2048,
    )

    return response.choices[0].message.content
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
│  - 1分の動画 ≈ 数千〜数万トークン消費                     │
└──────────────────────────────────────────────────────────┘
```

---

## 6. マルチモーダル Embedding

### 6.1 CLIP ベースの画像-テキスト Embedding

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

### 6.2 マルチモーダル RAG

```python
from dataclasses import dataclass

@dataclass
class MultimodalDocument:
    text: str
    image_path: str | None = None
    image_description: str | None = None
    source: str = ""

class MultimodalRAG:
    """画像とテキストを統合した RAG システム"""

    def __init__(self, text_embedder, image_embedder, qdrant_client):
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.qdrant = qdrant_client

    async def index_document(self, doc: MultimodalDocument):
        """マルチモーダルドキュメントのインデックス"""

        # テキスト Embedding
        text_vector = self.text_embedder.encode(doc.text)

        # 画像がある場合は画像の説明文も Embedding
        if doc.image_path:
            # VLM で画像の詳細な説明文を生成
            description = await self._describe_image(doc.image_path)
            doc.image_description = description

            # 画像説明をテキストに追加して再 Embedding
            combined_text = f"{doc.text}\n\n[画像の説明]: {description}"
            text_vector = self.text_embedder.encode(combined_text)

        # ベクトル DB に保存
        self.qdrant.upsert(
            collection_name="multimodal_docs",
            points=[PointStruct(
                id=doc.source,
                vector=text_vector.tolist(),
                payload={
                    "text": doc.text,
                    "image_path": doc.image_path,
                    "image_description": doc.image_description,
                    "source": doc.source,
                },
            )],
        )

    async def query(self, query: str, image_path: str | None = None) -> dict:
        """マルチモーダルクエリ"""

        # テキストクエリの Embedding
        query_vector = self.text_embedder.encode(query)

        # 画像クエリがある場合は統合
        if image_path:
            image_description = await self._describe_image(image_path)
            combined_query = f"{query}\n\n[画像の説明]: {image_description}"
            query_vector = self.text_embedder.encode(combined_query)

        # 検索
        results = self.qdrant.search(
            collection_name="multimodal_docs",
            query_vector=query_vector.tolist(),
            limit=5,
        )

        # 回答生成 (画像コンテキストを含む)
        context_parts = []
        for r in results:
            ctx = r.payload["text"]
            if r.payload.get("image_description"):
                ctx += f"\n[関連画像]: {r.payload['image_description']}"
            context_parts.append(ctx)

        context = "\n\n---\n\n".join(context_parts)

        answer = await self._generate_answer(query, context, image_path)
        return {"answer": answer, "sources": results}

    async def _describe_image(self, image_path: str) -> str:
        """VLM で画像の説明文を生成"""
        image_b64 = encode_image(image_path)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "low"}},
                    {"type": "text", "text": "この画像の内容を詳細に説明してください。テキスト、数値、グラフのデータなど全て含めてください。"},
                ],
            }],
        )
        return response.choices[0].message.content
```

### 6.3 画像検索システム

```python
import numpy as np
from PIL import Image

class ImageSearchEngine:
    """テキスト→画像、画像→画像の検索エンジン"""

    def __init__(self):
        self.clip_model = SentenceTransformer("clip-ViT-L-14")
        self.image_vectors = []
        self.image_metadata = []

    def index_images(self, image_paths: list[str]):
        """画像のバッチインデックス"""
        for path in image_paths:
            try:
                img = Image.open(path)
                vector = self.clip_model.encode(img)
                self.image_vectors.append(vector)
                self.image_metadata.append({
                    "path": path,
                    "size": img.size,
                    "format": img.format,
                })
            except Exception as e:
                print(f"Failed to index {path}: {e}")

    def search_by_text(self, query: str, top_k: int = 5) -> list[dict]:
        """テキストで画像を検索"""
        query_vector = self.clip_model.encode(query)

        vectors = np.array(self.image_vectors)
        similarities = cos_sim(query_vector, vectors)[0]

        top_indices = np.argsort(-similarities)[:top_k]

        return [
            {**self.image_metadata[i], "score": float(similarities[i])}
            for i in top_indices
        ]

    def search_by_image(self, image_path: str, top_k: int = 5) -> list[dict]:
        """画像で類似画像を検索"""
        query_img = Image.open(image_path)
        query_vector = self.clip_model.encode(query_img)

        vectors = np.array(self.image_vectors)
        similarities = cos_sim(query_vector, vectors)[0]

        top_indices = np.argsort(-similarities)[:top_k]

        return [
            {**self.image_metadata[i], "score": float(similarities[i])}
            for i in top_indices
        ]

# 使用例
engine = ImageSearchEngine()
engine.index_images(glob.glob("products/*.jpg"))

# テキストで画像検索
results = engine.search_by_text("赤いスニーカー")

# 類似画像検索
results = engine.search_by_image("reference_shoe.jpg")
```

---

## 7. 画像前処理と最適化

### 7.1 コスト最適化のための画像前処理

```python
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64

class ImageOptimizer:
    """マルチモーダル API 向け画像最適化"""

    @staticmethod
    def optimize_for_api(
        image_path: str,
        max_size: int = 1024,
        quality: int = 85,
        target_format: str = "JPEG",
    ) -> str:
        """API 送信用に画像を最適化してBase64返却"""

        img = Image.open(image_path)

        # 1. RGBA → RGB 変換 (JPEG は透過非対応)
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background

        # 2. リサイズ (アスペクト比維持)
        img.thumbnail((max_size, max_size), Image.LANCZOS)

        # 3. 圧縮して Base64 エンコード
        buffer = io.BytesIO()
        img.save(buffer, format=target_format, quality=quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def enhance_for_ocr(image_path: str) -> str:
        """OCR 精度向上のための画像強化"""

        img = Image.open(image_path)

        # 1. グレースケール変換
        img = img.convert("L")

        # 2. コントラスト強化
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)

        # 3. シャープネス強化
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)

        # 4. ノイズ除去
        img = img.filter(ImageFilter.MedianFilter(size=3))

        # 5. 二値化 (閾値処理)
        import numpy as np
        arr = np.array(img)
        threshold = 128
        arr = ((arr > threshold) * 255).astype(np.uint8)
        img = Image.fromarray(arr)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def estimate_cost(image_path: str, provider: str = "openai", detail: str = "auto") -> dict:
        """画像処理のトークンコストを推定"""
        img = Image.open(image_path)
        w, h = img.size

        if provider == "openai":
            if detail == "low":
                tokens = 85
            else:
                # high detail: タイル数を計算
                max_dim = max(w, h)
                if max_dim > 2048:
                    scale = 2048 / max_dim
                    w, h = int(w * scale), int(h * scale)

                min_dim = min(w, h)
                if min_dim > 768:
                    scale = 768 / min_dim
                    w, h = int(w * scale), int(h * scale)

                tiles_w = (w + 511) // 512
                tiles_h = (h + 511) // 512
                tokens = 85 + 170 * tiles_w * tiles_h

        elif provider == "anthropic":
            tokens = 1600

        elif provider == "gemini":
            tokens = 258

        return {
            "provider": provider,
            "original_size": img.size,
            "estimated_tokens": tokens,
            "estimated_cost_usd": tokens * 0.005 / 1000,  # GPT-4o input rate
        }
```

### 7.2 バッチ画像処理

```python
import asyncio
from typing import Any

async def batch_process_images(
    image_paths: list[str],
    prompt: str,
    model: str = "gpt-4o-mini",
    max_concurrent: int = 5,
    detail: str = "low",
) -> list[dict]:
    """大量画像のバッチ処理"""

    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * len(image_paths)

    async def process_one(index: int, path: str):
        async with semaphore:
            try:
                image_b64 = ImageOptimizer.optimize_for_api(path, max_size=512)

                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": detail,
                            }},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                    max_tokens=512,
                )

                results[index] = {
                    "path": path,
                    "result": response.choices[0].message.content,
                    "status": "success",
                }
            except Exception as e:
                results[index] = {
                    "path": path,
                    "error": str(e),
                    "status": "error",
                }

    tasks = [process_one(i, path) for i, path in enumerate(image_paths)]
    await asyncio.gather(*tasks)

    return results

# 使用例: 1000枚の商品画像を分類
import glob
image_files = glob.glob("products/**/*.jpg", recursive=True)
results = await batch_process_images(
    image_files,
    prompt="この商品のカテゴリを1つ選んでください: clothing, electronics, food, furniture, other",
    model="gpt-4o-mini",
    max_concurrent=10,
    detail="low",
)
```

---

## 8. 比較表

### 8.1 マルチモーダルモデル性能比較

| ベンチマーク | GPT-4o | Claude 3.5 | Gemini 1.5 Pro | Qwen-VL-Max |
|-------------|--------|-----------|---------------|-------------|
| MMMU (マルチモーダル理解) | 69.1 | 68.3 | 62.2 | 51.4 |
| MathVista (数学+視覚) | 63.8 | 61.6 | 58.0 | 51.0 |
| ChartQA (グラフ理解) | 85.7 | 90.8 | 81.3 | 79.8 |
| DocVQA (文書理解) | 92.8 | 95.2 | 93.1 | 93.8 |
| TextVQA (画像内テキスト) | 77.4 | - | 73.5 | 79.5 |

### 8.2 画像処理コスト比較

| モデル | 低解像度 | 高解像度 | 最大画像数 |
|--------|---------|---------|-----------|
| GPT-4o | 85 tokens | ~1,105 tokens | 制限なし |
| Claude 3.5 | ~1,600 tokens | ~1,600 tokens | 20枚 |
| Gemini 1.5 | 258 tokens | 258 tokens | 3,600枚 |

### 8.3 ユースケース別推奨モデル

| ユースケース | 推奨モデル | 理由 |
|-------------|-----------|------|
| OCR (文書) | Claude 3.5 | DocVQA 最高スコア |
| グラフ分析 | Claude 3.5 | ChartQA 最高スコア |
| 大量画像処理 | Gemini | 低コスト、大量画像対応 |
| UI分析 | GPT-4o | 汎用性が高い |
| 動画分析 | Gemini | ネイティブ動画対応 |
| リアルタイム対話 | GPT-4o | Realtime API 対応 |
| PDF 分析 | Claude 3.5 | 直接 PDF 入力対応 |
| 画像生成 | GPT-4o / Gemini | ネイティブ画像生成 |

---

## 9. 実務ユースケース

### 9.1 製造業: 品質検査自動化

```python
class QualityInspector:
    """製品画像の品質検査"""

    def __init__(self, reference_images: list[str]):
        self.reference_images = reference_images
        self.defect_categories = [
            "scratch",      # 傷
            "dent",         # へこみ
            "discoloration",# 変色
            "misalignment", # ズレ
            "contamination",# 汚れ
            "none",         # 正常
        ]

    async def inspect(self, product_image: str) -> dict:
        """製品画像の品質検査"""

        content = [
            {"type": "text", "text": "参考画像 (正常品):"},
        ]

        # 正常品の参考画像を添付
        for ref_path in self.reference_images[:3]:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(ref_path)}", "detail": "high"},
            })

        content.append({"type": "text", "text": "\n検査対象画像:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(product_image)}", "detail": "high"},
        })

        content.append({
            "type": "text",
            "text": f"""参考画像と検査対象画像を比較し、品質を評価してください。

JSON形式で返してください:
{{
  "judgment": "pass" or "fail",
  "confidence": 0.0-1.0,
  "defects": [
    {{
      "category": "{self.defect_categories}のいずれか",
      "severity": "minor" or "major" or "critical",
      "location": "欠陥の位置の説明",
      "description": "欠陥の詳細"
    }}
  ],
  "overall_quality_score": 0-100
}}""",
        })

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)
```

### 9.2 不動産: 物件画像分析

```python
async def analyze_property(images: list[str]) -> dict:
    """不動産物件の画像分析"""

    content = []
    for i, path in enumerate(images):
        content.append({"type": "text", "text": f"画像{i+1}:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(path)}", "detail": "high"},
        })

    content.append({
        "type": "text",
        "text": """これらの物件画像から以下を分析してください:

JSON形式で返してください:
{
  "property_type": "マンション/戸建て/事務所等",
  "rooms": [
    {
      "type": "部屋の種類 (リビング, キッチン, 寝室等)",
      "estimated_size": "推定サイズ (畳数)",
      "condition": "状態 (良好/普通/要修繕)",
      "features": ["特徴1", "特徴2"]
    }
  ],
  "overall_condition": "物件全体の状態",
  "strengths": ["長所1", "長所2"],
  "concerns": ["懸念点1", "懸念点2"],
  "estimated_age": "築年数の推定",
  "renovation_suggestions": ["リフォーム提案1", "リフォーム提案2"]
}""",
    })

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)
```

---

## 10. アンチパターンとベストプラクティス

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
optimized = ImageOptimizer.optimize_for_api("photo.jpg", max_size=1024)

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

### アンチパターン 3: 画像フォーマットの不適切な選択

```python
# NG: スクリーンショットをJPEGで送信 (テキスト部分がぼやける)
img.save("screenshot.jpg", quality=50)

# OK: テキストを含む画像はPNG、写真はJPEG
def choose_format(image_path: str, content_type: str) -> str:
    """コンテンツタイプに応じたフォーマット選択"""
    if content_type in ["screenshot", "document", "diagram", "chart"]:
        return "PNG"    # テキスト/線画はロスレス
    elif content_type in ["photo", "product"]:
        return "JPEG"   # 写真は圧縮可
    return "PNG"        # デフォルトはPNG
```

### アンチパターン 4: 1回のリクエストに大量画像

```python
# NG: 100枚の画像を1リクエストで送信
# → コンテキスト長超過、レイテンシ増大、コスト爆発

# OK: バッチ分割 + 並列処理
async def process_in_batches(images: list[str], batch_size: int = 5):
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_results = await asyncio.gather(
            *[process_single_image(img) for img in batch]
        )
        results.extend(batch_results)
    return results
```

---

## 11. FAQ

### Q1: 画像入力のトークンコストはどう計算される?

GPT-4o: low detail = 85 tokens 固定、high detail = タイルサイズに依存 (512x512 タイルごとに 170 tokens + 基本 85 tokens)。
Claude: 画像サイズに応じて約 1,600 tokens。
Gemini: 約 258 tokens/画像 (固定に近い)。
コスト最適化するなら、まず low detail で試し、精度不足なら high detail に上げる。

### Q2: 動画分析ができるのは Gemini だけ?

2025年時点で、ネイティブ動画入力に対応しているのは Gemini のみ。
GPT-4o / Claude ではフレーム抽出 (1-2fps) + 個別画像入力で疑似的に対応可能。
ただし、音声トラックの同時処理は Gemini でないとできない。

### Q3: マルチモーダル AI の精度はテキスト専用モデルに比べてどうか?

テキストタスクではテキスト専用モデルとほぼ同等。
画像+テキストの複合タスクではマルチモーダルモデルが圧倒的に有利。
ただし、画像内の細かいテキスト (小さな文字) や複雑な図表の読み取りは OCR 専用ツールに劣る場合がある。

### Q4: マルチモーダル RAG を構築する際の注意点は?

(1) 画像は直接ベクトル化するよりも VLM で説明文に変換してからテキスト Embedding する方が実用的。(2) CLIP で画像を直接ベクトル化する場合、テキストクエリとのギャップが生じやすいため、テキストベースの検索と併用すべき。(3) 図表やグラフは構造化データに変換してからインデックスすると検索精度が向上する。

### Q5: リアルタイム音声対話の遅延はどの程度?

GPT-4o Realtime API の場合、典型的なレイテンシは 300-500ms (テキスト入力) 〜 500-1000ms (音声入力)。WebSocket 接続のため、HTTP リクエストよりもオーバーヘッドが少ない。ただし、長い回答の場合はストリーミングで体感遅延を軽減する。

### Q6: 画像生成と画像入力を組み合わせるユースケースは?

(1) 画像編集: 元画像を入力し、「背景を変更して」「テキストを追加して」といった指示で編集。(2) 画像スタイル変換: 参考画像のスタイルで新しい画像を生成。(3) デザインイテレーション: 現在のデザインを入力し、改善版を生成。GPT-4o と Gemini がこのワークフローに対応している。

---

## まとめ

| 項目 | 内容 |
|------|------|
| 画像入力推奨 | GPT-4o (汎用)、Claude 3.5 (文書/PDF)、Gemini (大量画像) |
| 音声処理 | Whisper (文字起こし)、GPT-4o Realtime (対話)、TTS (音声生成) |
| 動画処理 | Gemini (ネイティブ対応)、GPT-4o (フレーム抽出で疑似対応) |
| コスト最適化 | 解像度を下げる、タスクに応じた detail 設定、バッチ処理 |
| 精度最適化 | 画像前処理、Self-Consistency、専用OCRとの併用 |
| マルチモーダル RAG | VLM で説明文変換 → テキスト Embedding が実用的 |
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
5. Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models," ICML 2023
6. Liu et al., "LLaVA: Large Language and Vision Assistant," NeurIPS 2023

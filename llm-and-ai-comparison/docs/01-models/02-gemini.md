# Gemini — Google DeepMind の統合マルチモーダル LLM

> Gemini は Google DeepMind が開発したマルチモーダルネイティブ LLM であり、テキスト・画像・音声・動画・コードを単一モデルで処理する次世代アーキテクチャを採用している。

## この章で学ぶこと

1. **Gemini のアーキテクチャと特徴** — Mixture of Experts、マルチモーダルネイティブ設計、長大コンテキストウィンドウの仕組み
2. **モデルラインナップと使い分け** — Ultra / Pro / Flash / Nano の性能差・コスト・適用領域
3. **Gemini API の実践的利用** — Google AI Studio・Vertex AI 経由での統合方法とベストプラクティス

---

## 1. Gemini のアーキテクチャ

### 1.1 マルチモーダルネイティブ設計

Gemini は従来の「テキストモデル + 視覚エンコーダの後付け」とは異なり、
訓練段階から複数モダリティを統合して学習するネイティブマルチモーダルアーキテクチャを採用している。

```
┌─────────────────────────────────────────────────┐
│              Gemini アーキテクチャ概要             │
├─────────────────────────────────────────────────┤
│                                                 │
│  入力モダリティ        統合エンコーダ              │
│  ┌──────────┐                                   │
│  │ テキスト  │──┐                                │
│  └──────────┘  │    ┌──────────────────┐        │
│  ┌──────────┐  ├──▶│  Unified Decoder  │──▶出力 │
│  │  画像    │──┤    │  (Transformer)    │        │
│  └──────────┘  │    └──────────────────┘        │
│  ┌──────────┐  │                                │
│  │  音声    │──┤    特徴:                        │
│  └──────────┘  │    - 各モダリティを同一          │
│  ┌──────────┐  │      潜在空間にマッピング        │
│  │  動画    │──┤    - クロスモーダル注意機構       │
│  └──────────┘  │    - 統一トークン化              │
│  ┌──────────┐  │                                │
│  │  コード  │──┘                                │
│  └──────────┘                                   │
└─────────────────────────────────────────────────┘
```

### 1.2 Mixture of Experts (MoE)

Gemini 1.5 以降は MoE アーキテクチャを採用し、推論時に一部の Expert のみを活性化することで、
パラメータ総数に対して計算コストを大幅に削減している。

```
┌───────────────────────────────────────────────┐
│           Mixture of Experts (MoE)            │
├───────────────────────────────────────────────┤
│                                               │
│  入力トークン                                  │
│      │                                        │
│      ▼                                        │
│  ┌────────┐                                   │
│  │ Router │  ← トークンごとに Expert を選択     │
│  └────────┘                                   │
│    │    │    │                                 │
│    ▼    ▼    ▼                                 │
│  ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐   │
│  │E1│ │E2│ │E3│ │E4│ │E5│ │E6│ │E7│ │E8│   │
│  └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘   │
│   ★         ★                   ★            │
│   活性化     活性化              活性化         │
│                                               │
│  ★ = 選択された Expert のみ計算               │
│  総パラメータ数 >> 活性パラメータ数             │
│  → 大容量の知識 + 低い推論コスト               │
└───────────────────────────────────────────────┘
```

### 1.3 長大コンテキストウィンドウ

Gemini 1.5 Pro は最大 200 万トークンのコンテキストウィンドウを提供する。

```python
# コンテキスト長の比較
context_windows = {
    "GPT-4o":          128_000,    # 128K
    "Claude 3.5":      200_000,    # 200K
    "Gemini 1.5 Pro":  2_000_000,  # 2M (!)
    "Gemini 1.5 Flash": 1_000_000, # 1M
}

for model, tokens in context_windows.items():
    pages = tokens // 500  # 1ページ ≈ 500 トークン
    print(f"{model}: {tokens:>10,} tokens ≈ {pages:>5,} pages")
```

---

## 2. モデルラインナップ

### 2.1 モデル比較表

| モデル | パラメータ規模 | コンテキスト | 主な用途 | 料金目安 (入力/1M tokens) |
|--------|-------------|------------|---------|------------------------|
| Gemini Ultra | 最大級 | 128K | 最高精度タスク | 高価格帯 |
| Gemini 1.5 Pro | 大規模 MoE | 2M | 汎用・長文書解析 | $1.25 - $5.00 |
| Gemini 1.5 Flash | 中規模 MoE | 1M | 高速・低コスト | $0.075 - $0.30 |
| Gemini 2.0 Flash | 次世代 MoE | 1M | 最新・高速 | $0.10 - $0.40 |
| Gemini Nano | 小規模 | 32K | オンデバイス | 無料 (端末内) |

### 2.2 ユースケース別選定ガイド

| ユースケース | 推奨モデル | 理由 |
|-------------|-----------|------|
| 大規模コードベース解析 | 1.5 Pro (2M) | 長大コンテキストが必須 |
| チャットボット | 2.0 Flash | 低レイテンシ・低コスト |
| 画像+テキスト理解 | 1.5 Pro / 2.0 Flash | マルチモーダル精度 |
| 端末内 AI アシスタント | Nano | オフライン動作 |
| 研究・最高精度 | Ultra | ベンチマークトップ |
| リアルタイム翻訳 | Flash | 速度重視 |

---

## 3. Gemini API の実践

### 3.1 Google AI Studio (Python SDK)

```python
import google.generativeai as genai

# API キー設定
genai.configure(api_key="YOUR_API_KEY")

# モデル初期化
model = genai.GenerativeModel("gemini-1.5-pro")

# テキスト生成
response = model.generate_content("量子コンピュータを小学生に説明して")
print(response.text)
```

### 3.2 マルチモーダル入力 (画像 + テキスト)

```python
import google.generativeai as genai
from pathlib import Path

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-pro")

# 画像読み込み
image = genai.upload_file(Path("architecture_diagram.png"))

# 画像 + テキストのプロンプト
response = model.generate_content([
    image,
    "この図のアーキテクチャを解説し、改善点を3つ提案してください。"
])
print(response.text)
```

### 3.3 ストリーミング応答

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")

# ストリーミングで生成
response = model.generate_content(
    "Pythonの非同期処理について詳しく解説してください。",
    stream=True
)

for chunk in response:
    print(chunk.text, end="", flush=True)
```

### 3.4 Vertex AI 経由 (エンタープライズ)

```python
import vertexai
from vertexai.generative_models import GenerativeModel

# Vertex AI 初期化
vertexai.init(project="my-project", location="us-central1")

model = GenerativeModel("gemini-1.5-pro")

response = model.generate_content(
    "当社の売上データを分析し、来期の予測を作成してください。",
    generation_config={
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
)
print(response.text)
```

### 3.5 システム指示と安全性設定

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    system_instruction="あなたは金融の専門家です。正確で慎重な回答を心がけてください。",
    safety_settings={
        "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
    }
)

# チャット形式
chat = model.start_chat()
response = chat.send_message("日経平均の今後の見通しについて教えてください")
print(response.text)

# 会話履歴の参照
for message in chat.history:
    print(f"{message.role}: {message.parts[0].text[:50]}...")
```

---

## 4. Gemini の技術的差別化要因

```
┌──────────────────────────────────────────────────────┐
│           Gemini の技術的差別化ポイント                 │
├──────────────────────────────────────────────────────┤
│                                                      │
│  1. 超長コンテキスト                                   │
│     └─ 200万トークン → 書籍1冊丸ごと処理可能           │
│                                                      │
│  2. ネイティブマルチモーダル                            │
│     └─ テキスト/画像/音声/動画を統一的に理解            │
│                                                      │
│  3. Google エコシステム統合                             │
│     └─ Search Grounding: Google 検索結果で回答補強     │
│     └─ Workspace 統合: Gmail, Docs, Sheets 連携       │
│                                                      │
│  4. Nano モデル (オンデバイス)                          │
│     └─ Pixel / Android 端末で直接実行                  │
│     └─ プライバシー保護 + オフライン動作                │
│                                                      │
│  5. コード生成特化                                     │
│     └─ AlphaCode 2 の知見を統合                        │
│     └─ 競技プログラミングレベルの推論能力               │
└──────────────────────────────────────────────────────┘
```

---

## 5. アンチパターン

### アンチパターン 1: コンテキストウィンドウの浪費

```python
# NG: 200万トークンあるからと全文書を無制限に投入
all_docs = load_all_company_documents()  # 300万トークン分
response = model.generate_content(all_docs + [query])
# → コンテキスト超過エラー or 精度低下 (中間部分の "Lost in the Middle" 問題)

# OK: 関連文書のみ抽出してから投入
relevant_docs = retrieve_relevant(query, all_docs, top_k=20)
response = model.generate_content(relevant_docs + [query])
# → 必要な情報に絞ることで精度向上 + コスト削減
```

### アンチパターン 2: モデル選定ミスによるコスト爆発

```python
# NG: 簡単な分類タスクに最高性能モデルを使用
model = genai.GenerativeModel("gemini-ultra")
for item in million_items:  # 100万件処理
    result = model.generate_content(f"カテゴリ分類: {item}")
# → 膨大なコスト発生

# OK: タスク難易度に応じたモデル選択
model = genai.GenerativeModel("gemini-1.5-flash")  # 分類タスクには Flash で十分
# さらに: バッチ API を使ってコスト削減
```

### アンチパターン 3: Safety フィルタの過剰緩和

```python
# NG: 安全性設定を全て無効化
safety = {cat: "BLOCK_NONE" for cat in all_categories}
# → 有害コンテンツ生成リスク、利用規約違反の可能性

# OK: ユースケースに応じた適切な設定
safety = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",  # 医療用途など必要に応じて緩和
}
```

---

## 6. FAQ

### Q1: Gemini と GPT-4o はどう使い分けるべきか?

長大な文書処理 (論文集、コードベース全体など) は Gemini の 200 万トークンコンテキストが圧倒的に有利。
一方、既存ツールチェーンとの統合性や Function Calling のエコシステム成熟度では GPT-4o に分がある。
マルチモーダルタスクでは両者とも高性能だが、動画入力は Gemini が先行している。

### Q2: Gemini Nano はどのデバイスで使えるか?

Google Pixel 8 以降、Samsung Galaxy S24 以降など、対応 Android 端末で利用可能。
AICore API を通じて呼び出す。オフラインで動作し、データが端末外に送信されない利点がある。
iOS では直接利用できないが、Chrome ブラウザ内蔵の Gemini Nano (on-device AI) が一部対応。

### Q3: Gemini API の料金体系はどうなっているか?

Google AI Studio の無料枠では 1 分あたり 15 リクエスト (1.5 Flash) / 2 リクエスト (1.5 Pro) が利用可能。
有料プランでは入力トークンと出力トークンに対してモデル別の従量課金。
128K 以下と以上でトークン単価が変わる二段階料金制を採用している点に注意。

### Q4: Search Grounding とは何か?

Gemini が Google 検索の結果をリアルタイムで参照し、回答の根拠とする機能。
ハルシネーション低減に効果的で、最新情報を含む質問に特に有効。Vertex AI で利用可能。

---

## まとめ

| 項目 | 内容 |
|------|------|
| 開発元 | Google DeepMind |
| アーキテクチャ | Transformer (MoE) + マルチモーダルネイティブ |
| 最大コンテキスト | 200 万トークン (1.5 Pro) |
| モダリティ | テキスト、画像、音声、動画、コード |
| モデルライン | Ultra / Pro / Flash / Nano |
| API アクセス | Google AI Studio (無料枠あり)、Vertex AI |
| 差別化要因 | 超長コンテキスト、Google 統合、オンデバイス |
| 主要競合 | GPT-4o (OpenAI)、Claude (Anthropic) |

---

## 次に読むべきガイド

- [03-open-source.md](./03-open-source.md) — オープンソース LLM との比較
- [04-model-comparison.md](./04-model-comparison.md) — 全モデル横断比較
- [../02-applications/04-multimodal.md](../02-applications/04-multimodal.md) — マルチモーダル活用の実践

---

## 参考文献

1. Google DeepMind, "Gemini: A Family of Highly Capable Multimodal Models," arXiv:2312.11805, 2023
2. Google, "Gemini API Documentation," https://ai.google.dev/docs
3. Google Cloud, "Vertex AI Gemini API," https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini
4. Reid et al., "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context," arXiv:2403.05530, 2024

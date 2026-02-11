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

---

## FAQ

### Q1: GPT-4o と GPT-4 Turbo の違いは？

**A:** GPT-4o は GPT-4 Turbo の後継で、より高速・低コスト・マルチモーダル統合が進んでいます。特にテキスト、画像、音声を統一的に扱える点が特徴です。新規プロジェクトでは GPT-4o の使用が推奨されます。

### Q2: o1 と o3 の違いは？

**A:** o3 は o1 の改良版で、推論能力が向上しています。o3-mini は `reasoning_effort` パラメータで推論の深さを制御でき、コスト効率が良くなっています。o1 は非推奨になりつつあり、o3-mini への移行が推奨されます。

### Q3: OpenAI API の利用制限を上げるには？

**A:** 利用実績に応じて自動的にティアが上がります。Tier 1（$5 支払い）から始まり、Tier 5（$200+ 支払い）まであります。急ぎの場合は OpenAI のセールスチームに連絡して上限引き上げを依頼できます。

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

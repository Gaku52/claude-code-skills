# Claude — Anthropic の AI アシスタント

> Constitutional AI を基盤とする Claude ファミリーの特徴、API 活用法、他モデルとの差別化ポイントを解説する。

## この章で学ぶこと

1. **Claude ファミリー**の各モデル（Haiku / Sonnet / Opus）の特性と使い分け
2. **Constitutional AI** の原理と Claude の安全性設計
3. **Claude API** の実践的な使い方とベストプラクティス

---

## 1. Claude ファミリーの概要

### ASCII 図解 1: Claude モデルファミリー

```
Claude モデルファミリー (2024-2025)
┌────────────────────────────────────────────────────┐
│                                                    │
│  Claude 3.5 / 4 ファミリー                          │
│                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │  Haiku   │  │  Sonnet  │  │  Opus    │         │
│  │          │  │          │  │          │         │
│  │ 高速     │  │ バランス  │  │ 最高性能  │         │
│  │ 低コスト  │  │ コスパ◎  │  │ 複雑推論  │         │
│  │ 軽量タスク│  │ 汎用     │  │ 研究/分析 │         │
│  └──────────┘  └──────────┘  └──────────┘         │
│                                                    │
│  性能:  Haiku < Sonnet < Opus                      │
│  速度:  Haiku > Sonnet > Opus                      │
│  コスト: Haiku < Sonnet < Opus                      │
│                                                    │
│  共通: 200K コンテキスト、マルチモーダル対応           │
└────────────────────────────────────────────────────┘
```

### コード例 1: Claude API の基本使用法

```python
import anthropic

client = anthropic.Anthropic()  # ANTHROPIC_API_KEY 環境変数を使用

# 基本的なメッセージ送信
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="あなたは日本語で回答する技術アシスタントです。",
    messages=[
        {"role": "user", "content": "Pythonのデコレータを説明してください"}
    ]
)

print(response.content[0].text)
print(f"入力トークン: {response.usage.input_tokens}")
print(f"出力トークン: {response.usage.output_tokens}")
```

### コード例 2: マルチターン会話

```python
import anthropic

client = anthropic.Anthropic()
conversation = []

def chat(user_message: str) -> str:
    """Claude とのマルチターン会話"""
    conversation.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system="あなたは親切なプログラミング講師です。段階的に教えてください。",
        messages=conversation,
    )

    assistant_message = response.content[0].text
    conversation.append({"role": "assistant", "content": assistant_message})
    return assistant_message

# 使用例
print(chat("Pythonのリスト内包表記を教えてください"))
print(chat("条件付きのリスト内包表記はどう書きますか？"))
print(chat("ネストされた場合はどうなりますか？"))
```

### コード例 3: Vision（画像理解）

```python
import anthropic
import base64

client = anthropic.Anthropic()

# 画像をBase64エンコード
with open("diagram.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-sonnet-4-20250514",
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
                "text": "この図を分析し、アーキテクチャの問題点を指摘してください。"
            }
        ],
    }]
)

print(response.content[0].text)
```

---

## 2. Constitutional AI

### ASCII 図解 2: Constitutional AI の仕組み

```
Constitutional AI (CAI) のプロセス:

Phase 1: 自己批評 (Red Teaming)
┌──────────┐    ┌──────────────┐    ┌──────────┐
│ 有害な    │ →  │ モデルが応答  │ →  │ 憲法原則  │
│ プロンプト│    │ を生成       │    │ に基づき  │
└──────────┘    └──────────────┘    │ 自己批評  │
                                    └────┬─────┘
                                         │
                                         ▼
                                    ┌──────────┐
                                    │ 改善版を  │
                                    │ 自己生成  │
                                    └──────────┘

Phase 2: RLAIF (RL from AI Feedback)
┌──────────┐    ┌──────────────┐    ┌──────────┐
│ 改善版    │ →  │ AI が好みを  │ →  │ 報酬モデル│
│ データ    │    │ 判定 (人間   │    │ で強化学習│
└──────────┘    │ の代わり)    │    └──────────┘
                └──────────────┘

「憲法」= 安全性・有用性・正直さの原則集
```

### コード例 4: システムプロンプトでの安全性活用

```python
import anthropic

client = anthropic.Anthropic()

# Claude の安全性特性を活用したシステム設計
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    system="""あなたは企業の法務アシスタントです。

以下の原則に従ってください:
1. 法的助言ではなく一般的な情報提供であることを明記する
2. 不確実な場合は「確認が必要」と述べる
3. 個人情報の取り扱いには特に注意する
4. 専門家への相談を適切に推奨する

Claude の Constitutional AI による安全性に加え、
上記のアプリケーション固有のガードレールを設定しています。""",
    messages=[{
        "role": "user",
        "content": "退職時の有給休暇の扱いについて教えてください"
    }]
)

print(response.content[0].text)
```

---

## 3. Claude API の高度な機能

### ASCII 図解 3: Claude のツール使用フロー

```
ユーザー → Claude API (ツール定義付き)
              │
              ▼
         ┌─────────────┐
         │ Claude が    │
         │ ツール呼び出し│
         │ を判断       │
         └──────┬──────┘
                │
           ┌────┴────┐
           │tool_use │
           │ブロック  │
           └────┬────┘
                │
                ▼
         アプリケーション側で
         ツールを実行
                │
                ▼
         ┌─────────────┐
         │ tool_result  │
         │ を返送       │
         └──────┬──────┘
                │
                ▼
         ┌─────────────┐
         │ Claude が    │
         │ 最終回答生成  │
         └─────────────┘
```

### コード例 5: ツール使用（Function Calling）

```python
import anthropic
import json

client = anthropic.Anthropic()

# ツール定義
tools = [
    {
        "name": "get_weather",
        "description": "指定された都市の現在の天気を取得",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "都市名（例: 東京、大阪）"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度の単位"
                }
            },
            "required": ["city"]
        }
    }
]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "東京の天気を教えてください"}]
)

# ツール呼び出しの処理
for block in response.content:
    if block.type == "tool_use":
        print(f"ツール: {block.name}")
        print(f"引数: {json.dumps(block.input, ensure_ascii=False)}")

        # ツール結果を返送
        result = {"temperature": 22, "condition": "晴れ", "humidity": 45}
        follow_up = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,
            messages=[
                {"role": "user", "content": "東京の天気を教えてください"},
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": block.id,
                     "content": json.dumps(result, ensure_ascii=False)}
                ]}
            ]
        )
        print(follow_up.content[0].text)
```

---

### 比較表 1: Claude モデルの詳細比較

| 項目 | Haiku | Sonnet | Opus |
|------|-------|--------|------|
| 最適用途 | 分類、抽出、軽量タスク | 汎用、コーディング、分析 | 複雑推論、研究、創作 |
| 入力料金 (/1M tokens) | $0.80 | $3.00 | $15.00 |
| 出力料金 (/1M tokens) | $4.00 | $15.00 | $75.00 |
| コンテキスト長 | 200K | 200K | 200K |
| 速度 (tokens/sec) | 非常に速い | 速い | 中程度 |
| コーディング能力 | 良好 | 優秀 | 最高 |
| 推論能力 | 良好 | 優秀 | 最高 |
| ビジョン対応 | あり | あり | あり |

### 比較表 2: Claude vs 他モデルの特徴比較

| 特徴 | Claude | GPT-4o | Gemini 1.5 Pro |
|------|--------|--------|----------------|
| 安全性アプローチ | Constitutional AI | RLHF | 非公開 |
| コンテキスト長 | 200K | 128K | 1M+ |
| 日本語能力 | 優秀 | 優秀 | 良好 |
| コード生成 | 優秀 | 優秀 | 良好 |
| 長文理解 | 非常に優秀 | 良好 | 最高 |
| ツール使用 | 優秀 | 優秀 | 良好 |
| 価格帯 | 中程度 | 中程度 | 中〜高 |

---

## アンチパターン

### アンチパターン 1: モデル選択の固定化

```
誤: 全タスクで Opus を使用
  → コストが不必要に高い、レイテンシも増大

正: タスクに応じてモデルを使い分ける
  Haiku: 分類、感情分析、簡単な質問応答
  Sonnet: コード生成、文書作成、一般的な分析
  Opus: 複雑な推論、研究レベルの分析、長文創作
```

### アンチパターン 2: プロンプトキャッシュの未活用

```
誤: 同じシステムプロンプトを毎回フル送信
  → 大量のトークン消費、レイテンシ増加

正: Claude のプロンプトキャッシュを活用
  # キャッシュ対象のブロックに cache_control を追加
  messages=[{
      "role": "user",
      "content": [
          {
              "type": "text",
              "text": "大量のコンテキスト...",
              "cache_control": {"type": "ephemeral"}
          },
          {"type": "text", "text": "質問"}
      ]
  }]
  # 2回目以降は 90% のコスト削減
```

---

## FAQ

### Q1: Claude の最大の強みは何ですか？

**A:** 長文コンテキストの理解力、安全性（Constitutional AI による）、そしてコーディング能力です。特に 200K トークンのコンテキストウィンドウを実用的に活用できる点は大きな強みで、大規模なコードベースや長文ドキュメントの分析に適しています。

### Q2: Claude API のレートリミットはどうなっていますか？

**A:** ティア制で管理されています。初期は RPM（分あたりリクエスト数）と TPM（分あたりトークン数）に制限があり、利用実績に応じてティアが上がります。Tier 1 で約 50 RPM、Tier 4 で約 4000 RPM が目安です。バッチ API を使えばリミットの影響を回避できます。

### Q3: Claude Code とは何ですか？

**A:** Anthropic 公式の CLI ツールで、ターミナルから Claude と対話しながらコーディングができます。ファイルの読み書き、Git 操作、テスト実行などをエージェント的に行え、MCP（Model Context Protocol）で外部ツールとも統合できます。

---

## まとめ

| 項目 | 要点 |
|------|------|
| Claude ファミリー | Haiku（速度）/ Sonnet（バランス）/ Opus（性能）の3段階 |
| Constitutional AI | 憲法原則に基づく自己批評でアラインメント |
| API 機能 | メッセージ、ストリーミング、ツール使用、ビジョン対応 |
| コンテキスト | 200K トークンの長文コンテキスト |
| プロンプトキャッシュ | 繰り返しコンテキストのコストを 90% 削減 |
| Claude Code | CLI ベースのAIコーディングアシスタント |

---

## 次に読むべきガイド

- [01-gpt.md](./01-gpt.md) — GPT ファミリーとの比較
- [04-model-comparison.md](./04-model-comparison.md) — 全モデルの横断比較
- [../02-applications/02-function-calling.md](../02-applications/02-function-calling.md) — Function Calling の詳細

---

## 参考文献

1. Anthropic. (2023). "Claude's Constitution." https://www.anthropic.com/index/claudes-constitution
2. Bai, Y. et al. (2022). "Constitutional AI: Harmlessness from AI Feedback." *arXiv:2212.08073*. https://arxiv.org/abs/2212.08073
3. Anthropic. "Claude API Documentation." https://docs.anthropic.com/

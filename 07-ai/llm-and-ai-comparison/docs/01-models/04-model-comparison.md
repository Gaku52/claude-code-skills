# モデル比較 — ベンチマーク・価格・ユースケース別選定ガイド

> LLM の選定は単一指標では決まらない。ベンチマーク性能、コスト、レイテンシ、コンテキスト長、マルチモーダル対応、プライバシー要件を総合的に評価し、ユースケースに最適なモデルを選ぶ必要がある。

## この章で学ぶこと

1. **主要ベンチマークの読み方** — MMLU、HumanEval、MT-Bench 等の意味と限界
2. **コスト・性能のトレードオフ分析** — 価格対品質の最適点を見つける方法
3. **ユースケース別モデル選定フレームワーク** — 要件からモデルを逆引きする実践手法

---

## 1. 主要ベンチマーク解説

### 1.1 ベンチマーク一覧

```
┌──────────────────────────────────────────────────────────┐
│              LLM 主要ベンチマーク体系                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  知識・推論                                               │
│  ├── MMLU (57科目の多肢選択)                              │
│  ├── GPQA (大学院レベル科学問題)                           │
│  ├── ARC-Challenge (科学的推論)                           │
│  └── BIG-Bench Hard (難問推論集)                          │
│                                                          │
│  コード                                                   │
│  ├── HumanEval (Python関数生成, 164問)                    │
│  ├── MBPP (基本プログラミング, 974問)                      │
│  ├── SWE-bench (実リポジトリのバグ修正)                    │
│  └── LiveCodeBench (競技プログラミング)                    │
│                                                          │
│  数学                                                     │
│  ├── GSM8K (小学校算数)                                   │
│  ├── MATH (高校〜大学数学)                                │
│  └── AIME (数学オリンピック級)                             │
│                                                          │
│  対話・指示追従                                            │
│  ├── MT-Bench (多ターン対話, GPT-4評価)                   │
│  ├── AlpacaEval (指示追従, GPT-4評価)                     │
│  └── LMSYS Chatbot Arena (人間ブラインド評価)             │
│                                                          │
│  多言語                                                   │
│  ├── MGSM (多言語数学)                                    │
│  └── JMMLU / JGLUE (日本語特化)                           │
└──────────────────────────────────────────────────────────┘
```

### 1.2 ベンチマークスコア比較 (2025年初頭時点)

| モデル | MMLU | HumanEval | MATH | MT-Bench | Arena Elo |
|--------|------|-----------|------|----------|-----------|
| GPT-4o | 88.7 | 90.2 | 76.6 | 9.3 | ~1280 |
| Claude 3.5 Sonnet | 88.7 | 92.0 | 78.3 | 9.2 | ~1270 |
| Gemini 1.5 Pro | 85.9 | 84.1 | 67.7 | 9.0 | ~1260 |
| Llama 3.1 405B | 87.3 | 89.0 | 73.8 | 8.8 | ~1200 |
| Qwen 2.5 72B | 86.1 | 86.4 | 71.9 | 8.7 | ~1190 |
| DeepSeek-V3 | 87.1 | 82.6 | 90.2 | 8.9 | ~1250 |
| Mixtral 8x22B | 77.8 | 75.0 | 49.8 | 8.1 | ~1140 |
| GPT-4o mini | 82.0 | 87.2 | 70.2 | 8.6 | ~1200 |
| Gemini 1.5 Flash | 78.9 | 74.3 | 54.9 | 8.2 | ~1170 |

*注: スコアは公開情報に基づく概算値。評価条件により変動する。*

---

## 2. コスト比較

### 2.1 API 料金表 (2025年初頭)

```python
# モデル別コスト計算ツール
pricing = {
    # (入力$/1M tokens, 出力$/1M tokens)
    "GPT-4o":              (2.50,   10.00),
    "GPT-4o mini":         (0.15,    0.60),
    "Claude 3.5 Sonnet":   (3.00,   15.00),
    "Claude 3.5 Haiku":    (0.80,    4.00),
    "Gemini 1.5 Pro":      (1.25,    5.00),
    "Gemini 1.5 Flash":    (0.075,   0.30),
    "DeepSeek-V3":         (0.27,    1.10),
}

def calculate_cost(model, input_tokens, output_tokens):
    """月間コストを計算"""
    in_price, out_price = pricing[model]
    cost = (input_tokens / 1_000_000) * in_price + \
           (output_tokens / 1_000_000) * out_price
    return cost

# 例: 月間100万リクエスト、平均入力500token、出力200token
for model, prices in pricing.items():
    monthly = calculate_cost(model, 500_000_000, 200_000_000)
    print(f"{model:25s}: ${monthly:>10,.2f}/月")
```

### 2.2 コスト比較表

| モデル | 入力 ($/1M) | 出力 ($/1M) | 月100万req概算 | コスパ評価 |
|--------|-----------|-----------|-------------|----------|
| GPT-4o mini | $0.15 | $0.60 | $195 | 極めて高い |
| Gemini 1.5 Flash | $0.075 | $0.30 | $97 | 最安クラス |
| DeepSeek-V3 | $0.27 | $1.10 | $355 | 高い |
| Claude 3.5 Haiku | $0.80 | $4.00 | $1,200 | 高い |
| GPT-4o | $2.50 | $10.00 | $3,250 | 中 |
| Claude 3.5 Sonnet | $3.00 | $15.00 | $4,500 | 中 |
| Gemini 1.5 Pro | $1.25 | $5.00 | $1,625 | 中 |

*月100万req = 平均入力500tok + 出力200tok で概算*

### 2.3 自前デプロイ vs API コスト

```
┌──────────────────────────────────────────────────────────┐
│        自前デプロイ vs API サービス コスト分析              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  コスト                                                   │
│  ^                                                       │
│  │                                                       │
│  │  API                                                  │
│  │  ╱                                                    │
│  │ ╱        自前デプロイ                                   │
│  │╱         ┌──────────────────────                      │
│  ├─────────┘                                             │
│  │  初期投資                                              │
│  │  (GPU購入/レンタル)                                    │
│  │                                                       │
│  └──────────────────────────────▶ リクエスト数            │
│          ↑                                               │
│     損益分岐点                                            │
│     (月間約50-100万req)                                   │
│                                                          │
│  判断基準:                                                │
│  - 月間 <10万req → API 一択                               │
│  - 月間 10-100万req → 要計算                              │
│  - 月間 >100万req → 自前デプロイ検討                      │
│  - データ機密性要件 → 自前デプロイ推奨                     │
└──────────────────────────────────────────────────────────┘
```

---

## 3. 機能比較

### 3.1 機能マトリクス

| 機能 | GPT-4o | Claude 3.5 | Gemini 1.5 | Llama 3.1 | Qwen 2.5 |
|------|--------|-----------|-----------|----------|----------|
| テキスト生成 | S | S | S | A | A |
| コード生成 | S | S | A | A | A |
| 画像入力 | S | S | S | N/A | A (VL) |
| 音声入力 | S | N/A | S | N/A | A (Audio) |
| 動画入力 | N/A | N/A | S | N/A | N/A |
| 画像生成 | S | N/A | S | N/A | N/A |
| Function Calling | S | S | S | A | A |
| JSON Mode | S | S | S | A | A |
| System Prompt | S | S | S | S | S |
| ストリーミング | S | S | S | S | S |
| ファインチューニング | A | N/A | A | S | S |

*S=優秀, A=対応, N/A=未対応*

### 3.2 コンテキスト長比較

| モデル | 最大コンテキスト | 実用的な精度維持範囲 |
|--------|----------------|-------------------|
| Gemini 1.5 Pro | 2,000K | ~1,000K |
| Claude 3.5 Sonnet | 200K | ~150K |
| GPT-4o | 128K | ~64K |
| Llama 3.1 | 128K | ~64K |
| Qwen 2.5 | 128K | ~32K |
| Mixtral 8x22B | 64K | ~32K |

---

## 4. ユースケース別選定フレームワーク

### 4.1 選定フローチャート

```
┌─────────────────────────────────────────────────────┐
│          LLM 選定フローチャート                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  START: 要件定義                                     │
│    │                                                │
│    ├─ データをクラウドに送れない?                      │
│    │   YES → OSS モデル自前デプロイ                   │
│    │          ├─ 日本語重視 → Qwen 2.5              │
│    │          ├─ コード重視 → DeepSeek-Coder        │
│    │          └─ 汎用    → Llama 3.1               │
│    │                                                │
│    NO ↓                                             │
│    ├─ 予算制約が厳しい?                              │
│    │   YES → 低コストモデル                          │
│    │          ├─ Gemini 1.5 Flash (最安)             │
│    │          ├─ GPT-4o mini                         │
│    │          └─ DeepSeek-V3                        │
│    │                                                │
│    NO ↓                                             │
│    ├─ 超長文書処理が必要?                            │
│    │   YES → Gemini 1.5 Pro (2M tokens)             │
│    │                                                │
│    NO ↓                                             │
│    ├─ マルチモーダル (画像/音声/動画)?                │
│    │   YES → GPT-4o / Gemini 1.5 Pro               │
│    │                                                │
│    NO ↓                                             │
│    └─ 最高精度テキスト処理                           │
│         ├─ コード → Claude 3.5 Sonnet               │
│         ├─ 推論  → DeepSeek-R1 / o1                 │
│         └─ 汎用  → GPT-4o / Claude 3.5             │
└─────────────────────────────────────────────────────┘
```

### 4.2 ユースケース別推奨

```python
# ユースケース別モデル推奨辞書
recommendations = {
    "カスタマーサポートBot": {
        "first":  "GPT-4o mini",      # コスパ最高
        "second": "Gemini 1.5 Flash", # さらに安い
        "reason": "大量リクエスト処理、コスト重視",
    },
    "コードレビュー・生成": {
        "first":  "Claude 3.5 Sonnet",  # コード品質最高
        "second": "GPT-4o",
        "reason": "コード理解力、長いコンテキスト",
    },
    "法律文書分析": {
        "first":  "Gemini 1.5 Pro",     # 200万token
        "second": "Claude 3.5 Sonnet",  # 200K + 高精度
        "reason": "長大文書の一括処理が必要",
    },
    "社内機密データ処理": {
        "first":  "Qwen 2.5 72B (自前)",
        "second": "Llama 3.1 70B (自前)",
        "reason": "データがクラウドに出ない",
    },
    "数学・科学的推論": {
        "first":  "DeepSeek-R1",
        "second": "o1-preview",
        "reason": "段階的推論に特化",
    },
    "マルチモーダル分析": {
        "first":  "GPT-4o",
        "second": "Gemini 1.5 Pro",
        "reason": "画像+テキスト統合理解",
    },
    "リアルタイム翻訳": {
        "first":  "Gemini 1.5 Flash",
        "second": "GPT-4o mini",
        "reason": "低レイテンシ + 低コスト",
    },
}
```

---

## 5. モデル選定の実践コード

### 5.1 A/B テスト比較ツール

```python
import asyncio
import time
from openai import AsyncOpenAI
import google.generativeai as genai
from anthropic import AsyncAnthropic

async def compare_models(prompt: str) -> dict:
    """複数モデルの出力を並行比較"""
    results = {}

    # GPT-4o
    async def call_openai():
        client = AsyncOpenAI()
        start = time.time()
        resp = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        latency = time.time() - start
        return {
            "text": resp.choices[0].message.content,
            "tokens": resp.usage.total_tokens,
            "latency": latency,
        }

    # Claude 3.5 Sonnet
    async def call_anthropic():
        client = AsyncAnthropic()
        start = time.time()
        resp = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.time() - start
        return {
            "text": resp.content[0].text,
            "tokens": resp.usage.input_tokens + resp.usage.output_tokens,
            "latency": latency,
        }

    gpt_result, claude_result = await asyncio.gather(
        call_openai(), call_anthropic()
    )

    return {"gpt4o": gpt_result, "claude": claude_result}

# 使用例
results = asyncio.run(compare_models("Pythonのジェネレータを解説してください"))
for model, data in results.items():
    print(f"\n=== {model} ===")
    print(f"レイテンシ: {data['latency']:.2f}s")
    print(f"トークン数: {data['tokens']}")
    print(f"回答: {data['text'][:200]}...")
```

### 5.2 コスト見積もりツール

```python
def estimate_monthly_cost(
    model: str,
    daily_requests: int,
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 200,
) -> dict:
    """月間コストを見積もる"""
    pricing = {
        "gpt-4o":          (2.50,  10.00),
        "gpt-4o-mini":     (0.15,   0.60),
        "claude-3.5-sonnet": (3.00, 15.00),
        "claude-3.5-haiku": (0.80,  4.00),
        "gemini-1.5-pro":  (1.25,   5.00),
        "gemini-1.5-flash": (0.075, 0.30),
        "deepseek-v3":     (0.27,   1.10),
    }

    if model not in pricing:
        return {"error": f"Unknown model: {model}"}

    in_price, out_price = pricing[model]
    monthly_requests = daily_requests * 30

    input_cost = (monthly_requests * avg_input_tokens / 1_000_000) * in_price
    output_cost = (monthly_requests * avg_output_tokens / 1_000_000) * out_price
    total = input_cost + output_cost

    return {
        "model": model,
        "monthly_requests": monthly_requests,
        "input_cost": f"${input_cost:,.2f}",
        "output_cost": f"${output_cost:,.2f}",
        "total_monthly": f"${total:,.2f}",
        "cost_per_request": f"${total/monthly_requests:.6f}",
    }

# 全モデルの月間コスト一括比較
for model in ["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet",
              "gemini-1.5-flash", "deepseek-v3"]:
    result = estimate_monthly_cost(model, daily_requests=10000)
    print(f"{result['model']:25s} → {result['total_monthly']:>12s}/月")
```

---

## 6. アンチパターン

### アンチパターン 1: ベンチマークスコアだけで選定

```
# NG: MMLU スコアが最も高いモデルを無条件に採用
"MMLUが88点だからこのモデルにしよう"
→ 実際のタスク (日本語要約) では MMLU との相関が低い

# OK: 実タスクでの評価を実施
1. 自社タスクの評価データセット (100問以上) を作成
2. 候補モデル 3-5 個で推論を実行
3. 人手評価 or LLM-as-a-Judge で品質比較
4. コスト・レイテンシも加味して総合判断
```

### アンチパターン 2: 単一モデルへのベンダーロック

```python
# NG: OpenAI 固有の API 仕様に依存しきったコード
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_schema", "json_schema": {...}},
    # OpenAI 固有機能に強く依存
)

# OK: 抽象レイヤーを挟んでモデル切り替え可能に
from litellm import completion  # マルチプロバイダー対応ライブラリ

response = completion(
    model="gpt-4o",  # 簡単に "claude-3.5-sonnet" 等に変更可能
    messages=[{"role": "user", "content": prompt}],
)
```

### アンチパターン 3: レイテンシ無視の選定

```
# NG: リアルタイムチャットに高性能だが遅いモデル
ユーザー体験: 「...」 (10秒間の沈黙) → 離脱率増加

# OK: 用途に応じたレイテンシ要件の設定
- リアルタイムチャット → TTFT < 500ms → Flash/mini 系
- バッチ処理 → レイテンシ不問 → 最高精度モデル
- ストリーミング表示 → TTFT < 1s → 中堅モデルでもOK
```

---

## 7. FAQ

### Q1: 新しいモデルが出たら毎回乗り換えるべき?

頻繁な乗り換えはコスト (検証工数、コード変更、プロンプト再調整) が大きい。
3-6 ヶ月に一度、主要モデルの比較評価を行い、有意な改善が確認できた場合のみ移行するのが現実的。
抽象レイヤー (LiteLLM 等) を導入しておくと切り替えコストが下がる。

### Q2: 複数モデルを組み合わせるメリットは?

Router パターンでタスク難易度に応じてモデルを振り分けると、コストを 60-80% 削減できることがある。
例: 簡単な質問 → Flash/mini、複雑な質問 → Pro/Sonnet、コード生成 → 特化モデル。
OpenRouter や LiteLLM を使えばルーティングを容易に実装できる。

### Q3: ベンチマークと実運用の性能差はどの程度?

ベンチマーク汚染 (訓練データにベンチマーク問題が混入) の問題があり、特に MMLU では実力との乖離が指摘されている。
LMSYS Chatbot Arena のような人間評価が最も実態に近いが、自社タスクでの独自評価が最も信頼できる。
「ベンチマークは足切りに使い、最終判断は実タスク評価」が推奨される。

---

## まとめ

| 評価軸 | 最推奨モデル | 備考 |
|--------|------------|------|
| 総合性能 | GPT-4o / Claude 3.5 Sonnet | 僅差、タスク依存 |
| コストパフォーマンス | Gemini 1.5 Flash / GPT-4o mini | 10-50倍安い |
| 長文処理 | Gemini 1.5 Pro (2M) | 他を圧倒 |
| コード生成 | Claude 3.5 Sonnet | SWE-bench 最高 |
| 数学・推論 | DeepSeek-R1 / o1 | CoT 特化 |
| 日本語 | Qwen 2.5 / GPT-4o | CJK 強い |
| プライバシー | OSS 自前デプロイ | Qwen/Llama |
| マルチモーダル | GPT-4o / Gemini 1.5 | 動画は Gemini のみ |

---

## 次に読むべきガイド

- [../02-applications/00-prompt-engineering.md](../02-applications/00-prompt-engineering.md) — 選んだモデルの性能を最大化するプロンプト技法
- [../03-infrastructure/00-api-integration.md](../03-infrastructure/00-api-integration.md) — API 統合の実践
- [../03-infrastructure/03-evaluation.md](../03-infrastructure/03-evaluation.md) — 自社タスクでの評価手法

---

## 参考文献

1. LMSYS, "Chatbot Arena Leaderboard," https://chat.lmsys.org/
2. Hugging Face, "Open LLM Leaderboard," https://huggingface.co/spaces/open-llm-leaderboard
3. Artificial Analysis, "LLM Performance Leaderboard," https://artificialanalysis.ai/
4. Chiang et al., "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference," arXiv:2403.04132, 2024

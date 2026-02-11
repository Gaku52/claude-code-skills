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

---

## FAQ

### Q1: temperature=0 でも同じプロンプトで異なる結果が出ることがありますか？

**A:** はい、あります。GPU の浮動小数点演算の非決定性や、バッチ処理の影響で、temperature=0 でも完全に同一の結果は保証されません。OpenAI では `seed` パラメータで再現性を高めることができますが、100%の保証はありません。

### Q2: ストリーミングを使うとコストは変わりますか？

**A:** いいえ、トークン消費量（コスト）は同じです。ストリーミングは応答の配信方法が異なるだけで、生成されるトークン数は変わりません。ただし、ストリーミングでは接続が長時間維持されるため、サーバーリソースの消費パターンが異なります。

### Q3: バッチ処理はいつ使うべきですか？

**A:** リアルタイム応答が不要な大量処理（数百〜数万リクエスト）に最適です。例えば、大量のドキュメント分類、データセットのラベリング、コンテンツ生成などです。OpenAI の Batch API は 50% 割引、Anthropic のバッチ API も同様の割引があり、24時間以内に結果が返されます。

---

## まとめ

| 項目 | 要点 |
|------|------|
| temperature | 0.0（決定的）〜 1.5（高多様性）で出力のランダム性を制御 |
| top_p | 累積確率で候補トークンをフィルタリング |
| ストリーミング | TTFB を大幅短縮、UX 向上に必須 |
| バッチ処理 | 大量リクエストの並列・非同期処理でコスト削減 |
| max_tokens | タスクに応じた適切な設定でコスト最適化 |
| 推論最適化 | キャッシュ・量子化・バッチの組み合わせが効果的 |

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

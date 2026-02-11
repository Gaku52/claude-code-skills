# トークナイゼーション — テキストをモデルが理解する単位に変換する

> BPE、SentencePiece、各モデルのトークナイザの違いと、トークン数管理の実践的テクニックを学ぶ。

## この章で学ぶこと

1. **BPE（Byte Pair Encoding）**の原理と主要な派生アルゴリズム
2. **SentencePiece** と各モデル固有トークナイザの特性と比較
3. **トークン数管理**の実践手法とコスト最適化

---

## 1. トークナイゼーションの基本

### ASCII 図解 1: トークナイゼーションの流れ

```
入力テキスト
"大規模言語モデルは素晴らしい"
        │
        ▼
┌─────────────────────┐
│  Pre-tokenization   │
│  (空白・記号で分割)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  サブワード分割       │
│  BPE / Unigram      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  トークンID変換       │
│  語彙テーブル参照     │
└─────────┬───────────┘
          │
          ▼
[15043, 30590, 29914, 234, ...]
```

### コード例 1: tiktoken（OpenAI）でトークン化

```python
import tiktoken

# GPT-4o 用エンコーダ
enc = tiktoken.encoding_for_model("gpt-4o")

text = "大規模言語モデルは素晴らしい技術です。"
tokens = enc.encode(text)

print(f"テキスト: {text}")
print(f"トークン数: {len(tokens)}")
print(f"トークンID: {tokens}")

# 各トークンの内容を確認
for token_id in tokens:
    token_bytes = enc.decode_single_token_bytes(token_id)
    print(f"  ID {token_id:>6} → {token_bytes}")
```

### コード例 2: Hugging Face Tokenizer の比較

```python
from transformers import AutoTokenizer

models = {
    "GPT-4o": "Xenova/gpt-4o",
    "Claude": "anthropic/claude-tokenizer",  # 仮想例
    "Llama-3": "meta-llama/Llama-3.1-8B-Instruct",
    "Gemma-2": "google/gemma-2-9b",
}

text = "東京タワーの高さは333メートルです。The height is 333 meters."

for name, model_id in models.items():
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer.encode(text)
        print(f"{name:>10}: {len(tokens):>3} トークン")
    except Exception as e:
        print(f"{name:>10}: (ロード不可)")
```

---

## 2. BPE と SentencePiece

### ASCII 図解 2: BPE のマージプロセス

```
初期状態（文字単位）:
["l", "o", "w"]  ["l", "o", "w", "e", "r"]  ["n", "e", "w"]

Step 1: 最頻ペア ("l", "o") → "lo" をマージ
["lo", "w"]  ["lo", "w", "e", "r"]  ["n", "e", "w"]

Step 2: 最頻ペア ("lo", "w") → "low" をマージ
["low"]  ["low", "e", "r"]  ["n", "e", "w"]

Step 3: 最頻ペア ("e", "r") → "er" をマージ
["low"]  ["low", "er"]  ["n", "e", "w"]

Step 4: 最頻ペア ("n", "e") → "ne" をマージ
["low"]  ["low", "er"]  ["ne", "w"]

Step 5: 最頻ペア ("ne", "w") → "new" をマージ
["low"]  ["low", "er"]  ["new"]

→ 語彙: {l, o, w, e, r, n, lo, low, er, ne, new, lower, ...}
```

### コード例 3: 簡易 BPE の実装

```python
from collections import Counter

def get_pairs(word_freqs):
    """全ペアの出現頻度を計算"""
    pairs = Counter()
    for word, freq in word_freqs.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_pair(pair, word_freqs):
    """最頻ペアをマージ"""
    new_word_freqs = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in word_freqs.items():
        new_word = word.replace(bigram, replacement)
        new_word_freqs[new_word] = freq
    return new_word_freqs

# 学習データの単語頻度
word_freqs = {
    "l o w": 5,
    "l o w e r": 2,
    "n e w e s t": 6,
    "w i d e s t": 3,
}

num_merges = 10
merges = []

for i in range(num_merges):
    pairs = get_pairs(word_freqs)
    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    word_freqs = merge_pair(best_pair, word_freqs)
    merges.append(best_pair)
    print(f"Merge {i+1}: {best_pair} → {''.join(best_pair)}")

print(f"\n最終語彙の一部: {list(word_freqs.keys())}")
```

### コード例 4: SentencePiece の学習と使用

```python
import sentencepiece as spm

# モデルの学習
spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=32000,
    model_type="bpe",           # "unigram" も選択可能
    character_coverage=0.9995,  # 日本語は高めに設定
    pad_id=3,
    unk_id=0,
    bos_id=1,
    eos_id=2,
)

# 学習済みモデルの使用
sp = spm.SentencePieceProcessor()
sp.load("my_tokenizer.model")

text = "大規模言語モデルの性能はトークナイザに大きく依存する"
tokens = sp.encode(text, out_type=str)
ids = sp.encode(text, out_type=int)

print(f"テキスト: {text}")
print(f"トークン: {tokens}")
print(f"ID: {ids}")
print(f"復元: {sp.decode(ids)}")
```

### 比較表 1: トークナイゼーション手法の比較

| 手法 | 特徴 | 採用モデル | 日本語対応 | 語彙サイズ |
|------|------|-----------|-----------|-----------|
| BPE (Byte-level) | バイト単位で未知語なし | GPT-4, Claude | 良好 | 100K-200K |
| SentencePiece (Unigram) | 確率的サブワード分割 | LLaMA, Gemma | 良好 | 32K-128K |
| SentencePiece (BPE) | SPP フレームワーク上のBPE | T5, mBART | 良好 | 32K-64K |
| WordPiece | BPE の亜種 | BERT | 要調整 | 30K-50K |
| tiktoken | OpenAI 独自の高速BPE | GPT-4o | 良好 | 100K-200K |

---

## 3. トークン数管理

### ASCII 図解 3: トークン数とコストの関係

```
API コスト構造:
┌──────────────────────────────────────────┐
│                                          │
│  入力トークン        出力トークン          │
│  ┌──────────┐      ┌──────────┐         │
│  │ システム  │      │ 生成     │         │
│  │ プロンプト│      │ テキスト │         │
│  │          │      │          │         │
│  │ ユーザー  │      │          │         │
│  │ メッセージ│      │          │         │
│  └──────────┘      └──────────┘         │
│   $X / 1M tokens    $Y / 1M tokens      │
│   (通常 Yの方が高い)                      │
│                                          │
│  合計コスト = 入力数×X + 出力数×Y          │
└──────────────────────────────────────────┘

例: Claude 3.5 Sonnet
  入力: $3.00 / 1M tokens
  出力: $15.00 / 1M tokens
```

### コード例 5: トークン数カウントとコスト見積もり

```python
import tiktoken

def estimate_cost(
    text: str,
    model: str = "gpt-4o",
    max_output_tokens: int = 1000
):
    """APIコストの見積もり"""
    pricing = {
        "gpt-4o":          {"input": 2.50, "output": 10.00},
        "gpt-4o-mini":     {"input": 0.15, "output": 0.60},
        "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3.5-haiku": {"input": 0.80, "output": 4.00},
    }

    enc = tiktoken.encoding_for_model("gpt-4o")
    input_tokens = len(enc.encode(text))

    if model in pricing:
        p = pricing[model]
        input_cost = input_tokens / 1_000_000 * p["input"]
        output_cost = max_output_tokens / 1_000_000 * p["output"]
        total = input_cost + output_cost

        print(f"モデル: {model}")
        print(f"入力トークン: {input_tokens:,}")
        print(f"出力トークン(最大): {max_output_tokens:,}")
        print(f"入力コスト: ${input_cost:.4f}")
        print(f"出力コスト: ${output_cost:.4f}")
        print(f"合計見積もり: ${total:.4f}")

text = "ここに長いプロンプトが入ります。" * 100
estimate_cost(text, model="claude-3.5-sonnet")
```

### 比較表 2: モデル別トークナイザの特性

| モデル | トークナイザ | 語彙サイズ | 日本語効率 | 特殊トークン |
|--------|------------|-----------|-----------|-------------|
| GPT-4o | cl100k_base+ | ~200K | 高 (改善済) | <\|endoftext\|> 等 |
| Claude 3.5 | 独自 BPE | ~150K | 高 | 非公開 |
| Llama 3.1 | tiktoken 派生 | 128K | 中〜高 | <\|begin_of_text\|> 等 |
| Gemini 1.5 | SentencePiece | ~256K | 高 | 非公開 |
| Gemma 2 | SentencePiece | 256K | 高 | <bos>, <eos> 等 |

---

## アンチパターン

### アンチパターン 1: トークン数を考慮しないプロンプト設計

```
誤: 冗長な指示を毎回フルに送信
  → トークン消費が膨大、コスト爆発

# 悪い例
prompt = """
あなたは非常に優秀なアシスタントです。あなたの役割は...
（1000トークンのシステムプロンプト）
""" + user_message  # 毎回 1000 トークンのオーバーヘッド

# 良い例: 簡潔なプロンプト + キャッシュ活用
prompt = "JSON形式で回答。" + user_message
# または API のシステムプロンプトキャッシュを活用
```

### アンチパターン 2: 言語によるトークン効率の無視

```
誤: 英語基準でトークン上限を設計
  → 日本語では同じ内容でも 1.5〜2 倍のトークンを消費

# 確認方法
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")

en = "The capital of Japan is Tokyo."
ja = "日本の首都は東京です。"

print(f"英語: {len(enc.encode(en))} tokens ({len(en)} chars)")
print(f"日本語: {len(enc.encode(ja))} tokens ({len(ja)} chars)")
# 日本語は文字数あたりのトークン数が多い傾向
```

---

## FAQ

### Q1: トークナイザが異なるモデル間でトークン数は比較できますか？

**A:** 正確な比較はできません。同じテキストでも、GPT-4o と Llama 3 ではトークン数が異なります。コスト比較する場合は、各モデルのトークナイザで個別にカウントする必要があります。ただし大まかな目安として、英語では 1 トークン ≒ 4 文字、日本語では 1 トークン ≒ 1〜2 文字が目安になります。

### Q2: コンテキストウィンドウを超えた場合どうなりますか？

**A:** API はエラーを返します。対策としては、(1) テキストを要約して短縮、(2) チャンク分割して複数回に分けて処理、(3) RAG で関連部分だけ取得、(4) より長いコンテキスト長を持つモデルに切り替え、があります。

### Q3: 日本語で最もトークン効率の良いモデルは？

**A:** 2024年時点では、GPT-4o と Gemini 1.5 が日本語のトークン効率に優れています。特に GPT-4o は前世代から大幅に改善されました。Claude 3.5 も高い日本語トークン効率を持ちます。ただし、トークン効率だけでなく、単価との掛け算で実際のコストを評価してください。

---

## まとめ

| 項目 | 要点 |
|------|------|
| BPE | 最頻ペアを統合してサブワード語彙を構築する手法 |
| SentencePiece | 言語非依存のトークナイゼーションフレームワーク |
| tiktoken | OpenAI の高速 BPE 実装、GPT モデルで使用 |
| 日本語効率 | 英語の 1.5〜2 倍のトークンが必要な場合が多い |
| コスト管理 | 入力/出力トークン数の把握とプロンプト最適化が重要 |
| 語彙サイズ | 32K〜256K の範囲で、大きいほど効率的だが学習コスト増 |

---

## 次に読むべきガイド

- [02-inference.md](./02-inference.md) — 推論パラメータ（温度、Top-p）の最適化
- [../02-applications/00-prompt-engineering.md](../02-applications/00-prompt-engineering.md) — プロンプトエンジニアリングの実践
- [../03-infrastructure/01-vector-databases.md](../03-infrastructure/01-vector-databases.md) — エンベディングとベクトルDB

---

## 参考文献

1. Sennrich, R. et al. (2016). "Neural Machine Translation of Rare Words with Subword Units (BPE)." *ACL 2016*. https://arxiv.org/abs/1508.07909
2. Kudo, T. & Richardson, J. (2018). "SentencePiece: A simple and language independent subword tokenizer." *EMNLP 2018*. https://arxiv.org/abs/1808.06226
3. OpenAI. "tiktoken: Fast BPE tokeniser for use with OpenAI's models." https://github.com/openai/tiktoken

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

### 1.1 トークナイゼーションの歴史と背景

自然言語処理（NLP）におけるテキストの分割手法は、長い進化の歴史を持つ。初期のNLPシステムでは単語単位の分割（Word-level tokenization）が主流だったが、語彙外単語（OOV: Out-of-Vocabulary）の問題が深刻だった。文字単位の分割（Character-level tokenization）はOOV問題を解消するが、シーケンス長が極端に長くなり、意味的な情報が失われる。

サブワード分割は、この2つのアプローチの中間に位置する画期的な手法である。頻出する単語はそのまま1つのトークンとして保持し、稀な単語はより小さな意味のある部分（サブワード）に分割する。これにより、語彙サイズを抑えながらも、あらゆるテキストを表現できるようになった。

```
分割手法の進化:

Word-level:        "unhappiness" → ["unhappiness"] (語彙に必要)
                   "unhappily"  → [UNK]           (語彙外!)

Character-level:   "unhappiness" → ["u","n","h","a","p","p","i","n","e","s","s"]
                   → 11トークン (長すぎる)

Subword (BPE):     "unhappiness" → ["un", "happiness"]
                   "unhappily"   → ["un", "happily"]
                   → 語彙サイズ小、OOVなし、意味を保持
```

### 1.2 Pre-tokenization の詳細

Pre-tokenization はサブワード分割の前段階で、テキストを大まかな単位に分割する処理である。この段階の設計がトークナイザ全体の性能に大きく影響する。

```python
# Pre-tokenization の各手法
import re

text = "Hello, World! 大規模言語モデル（LLM）は2024年に急速に発展した。"

# 方法1: 空白分割（最もシンプル）
whitespace_tokens = text.split()
print(f"空白分割: {whitespace_tokens}")
# → ['Hello,', 'World!', '大規模言語モデル（LLM）は2024年に急速に発展した。']

# 方法2: GPT-2/GPT-4o スタイル（正規表現ベース）
gpt2_pattern = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    re.UNICODE
)

# 方法3: バイトレベル（GPT-4o, Claude）
# 全てのテキストをUTF-8バイト列として処理
byte_sequence = text.encode("utf-8")
print(f"バイト数: {len(byte_sequence)}")
# 日本語は1文字あたり3バイト（UTF-8）

# 方法4: SentencePiece スタイル（空白を特殊文字に）
# 空白を "▁" (U+2581) に変換して扱う
sp_text = "▁" + text.replace(" ", "▁")
print(f"SentencePiece形式: {sp_text}")
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

### コード例: トークン化の詳細分析ツール

```python
import tiktoken
from collections import Counter

class TokenAnalyzer:
    """テキストのトークン化を詳細に分析するツール"""

    def __init__(self, model: str = "gpt-4o"):
        self.enc = tiktoken.encoding_for_model(model)
        self.model = model

    def analyze(self, text: str) -> dict:
        """テキストのトークン化を詳細に分析する"""
        tokens = self.enc.encode(text)
        token_strings = [
            self.enc.decode([t]) for t in tokens
        ]

        # トークンの種類を分類
        categories = {
            "ascii": 0,
            "japanese": 0,
            "number": 0,
            "punctuation": 0,
            "whitespace": 0,
            "other": 0,
        }

        for ts in token_strings:
            if ts.strip() == "":
                categories["whitespace"] += 1
            elif ts.isascii() and ts.isalpha():
                categories["ascii"] += 1
            elif ts.isdigit():
                categories["number"] += 1
            elif any(ord(c) > 0x3000 for c in ts):
                categories["japanese"] += 1
            elif not ts.isalnum():
                categories["punctuation"] += 1
            else:
                categories["other"] += 1

        return {
            "text_length_chars": len(text),
            "text_length_bytes": len(text.encode("utf-8")),
            "token_count": len(tokens),
            "chars_per_token": len(text) / len(tokens),
            "bytes_per_token": len(text.encode("utf-8")) / len(tokens),
            "token_categories": categories,
            "unique_tokens": len(set(tokens)),
            "token_ids": tokens,
            "token_strings": token_strings,
        }

    def compare_texts(self, texts: dict[str, str]) -> None:
        """複数テキストのトークン効率を比較する"""
        print(f"{'テキスト':<20} {'文字数':>6} {'バイト数':>8} "
              f"{'トークン数':>8} {'文字/トークン':>12}")
        print("-" * 70)
        for name, text in texts.items():
            result = self.analyze(text)
            print(f"{name:<20} {result['text_length_chars']:>6} "
                  f"{result['text_length_bytes']:>8} "
                  f"{result['token_count']:>8} "
                  f"{result['chars_per_token']:>12.2f}")

    def estimate_cost(self, text: str, model_pricing: dict) -> dict:
        """テキストのAPIコストを見積もる"""
        tokens = len(self.enc.encode(text))
        input_cost = (tokens / 1_000_000) * model_pricing["input"]
        return {
            "tokens": tokens,
            "input_cost_usd": input_cost,
        }

# 使用例
analyzer = TokenAnalyzer("gpt-4o")

# 日英比較
texts = {
    "英語（短文）": "The quick brown fox jumps over the lazy dog.",
    "日本語（短文）": "素早い茶色の狐が怠惰な犬を飛び越える。",
    "英語（技術文）": "Transformer architecture uses self-attention mechanism.",
    "日本語（技術文）": "Transformerアーキテクチャは自己注意機構を使用する。",
    "コード": "def hello(): return 'Hello, World!'",
    "混合": "GPT-4oは2024年にリリースされた最新のLLMです。",
}

analyzer.compare_texts(texts)
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

### 2.1 BPE のバリエーション

BPE にはいくつかの重要なバリエーションが存在し、それぞれ異なるモデルで採用されている。

```
BPE ファミリー:

1. 基本 BPE (Sennrich et al., 2016)
   - 文字単位から開始し、最頻ペアをマージ
   - 決定的: 同じコーパスからは同じ語彙が得られる
   - 採用: 初期のGPTシリーズ

2. Byte-Level BPE (GPT-2/GPT-4o/Claude)
   - バイト(0-255)を基本単位として使用
   - 未知語が原理的に発生しない
   - 任意の言語・記号を処理可能
   - UTF-8 バイト列に対して BPE を適用

3. WordPiece (BERT)
   - BPE の亜種: マージ基準が異なる
   - 頻度ではなく尤度の増加量でマージペアを選択
   - "##" プレフィックスで分割されたサブワードを表記
   - 例: "unhappiness" → ["un", "##happiness"]

4. Unigram LM (SentencePiece)
   - BPE とは逆: 大きな語彙から開始し、削除していく
   - 確率的: 同じ単語に複数の分割候補があり得る
   - 正則化効果あり（学習時のロバスト性向上）
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

### コード例: Byte-Level BPE の詳細実装

```python
from collections import Counter, defaultdict
from typing import Optional

class ByteLevelBPE:
    """Byte-Level BPE の教育用実装"""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges: list[tuple[bytes, bytes]] = []
        self.vocab: dict[int, bytes] = {}

    def _get_stats(self, ids_list: list[list[int]]) -> Counter:
        """全てのバイトペアの出現頻度を計算"""
        counts = Counter()
        for ids in ids_list:
            for i in range(len(ids) - 1):
                counts[(ids[i], ids[i + 1])] += 1
        return counts

    def _merge(self, ids: list[int], pair: tuple[int, int],
               new_id: int) -> list[int]:
        """指定ペアをマージして新しいIDに置換"""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, texts: list[str]) -> None:
        """テキストコーパスから BPE 語彙を学習する"""
        # 初期語彙: 0-255 のバイト値
        self.vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256

        # テキストをバイト列に変換
        ids_list = [list(text.encode("utf-8")) for text in texts]

        # 語彙サイズに達するまでマージを繰り返す
        while next_id < self.vocab_size:
            stats = self._get_stats(ids_list)
            if not stats:
                break

            # 最頻ペアを選択
            best_pair = max(stats, key=stats.get)

            # 全テキストでマージを実行
            ids_list = [
                self._merge(ids, best_pair, next_id)
                for ids in ids_list
            ]

            # 語彙に追加
            self.vocab[next_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges.append(best_pair)

            if next_id % 100 == 0:
                print(f"Merge {next_id - 256}: "
                      f"{self.vocab[best_pair[0]]!r} + "
                      f"{self.vocab[best_pair[1]]!r} → "
                      f"{self.vocab[next_id]!r}")

            next_id += 1

        print(f"\n学習完了: {len(self.vocab)} トークン, "
              f"{len(self.merges)} マージ")

    def encode(self, text: str) -> list[int]:
        """テキストをトークンIDのリストに変換"""
        ids = list(text.encode("utf-8"))
        for pair in self.merges:
            new_id = 256 + self.merges.index(pair)
            ids = self._merge(ids, pair, new_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """トークンIDのリストをテキストに復元"""
        byte_sequence = b"".join(self.vocab[i] for i in ids)
        return byte_sequence.decode("utf-8", errors="replace")

# 使用例
bpe = ByteLevelBPE(vocab_size=500)
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "大規模言語モデルは自然言語処理の革命です。",
    "Machine learning and deep learning are transforming AI.",
] * 100  # コーパスを繰り返して頻度を上げる

bpe.train(corpus)

test_text = "The quick fox"
encoded = bpe.encode(test_text)
decoded = bpe.decode(encoded)
print(f"原文: {test_text}")
print(f"トークン数: {len(encoded)}")
print(f"復元: {decoded}")
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

### コード例: SentencePiece の高度な設定

```python
import sentencepiece as spm

# 日本語に最適化した SentencePiece 学習
spm.SentencePieceTrainer.train(
    input="japanese_corpus.txt",
    model_prefix="jp_tokenizer",
    vocab_size=32000,
    model_type="unigram",        # 日本語には Unigram が適している場合が多い
    character_coverage=0.9995,   # 日本語の文字カバレッジ
    byte_fallback=True,          # 未知文字をバイト表現にフォールバック
    split_by_unicode_script=True,  # Unicode スクリプト境界で分割
    split_by_number=True,         # 数字の境界で分割
    split_by_whitespace=True,     # 空白で分割
    split_digits=True,            # 各桁を個別のトークンにする
    treat_whitespace_as_suffix=False,
    allow_whitespace_only_pieces=True,
    normalization_rule_name="nfkc",  # Unicode 正規化
    num_threads=8,
    # 特殊トークンの定義
    user_defined_symbols=["<code>", "</code>", "<math>", "</math>"],
    control_symbols=["<sep>", "<cls>", "<mask>"],
)

# 学習済みモデルの詳細な使用法
sp = spm.SentencePieceProcessor()
sp.load("jp_tokenizer.model")

text = "GPT-4oの性能は2024年に大幅に向上した。"

# 通常のエンコード
tokens_str = sp.encode(text, out_type=str)
tokens_id = sp.encode(text, out_type=int)
print(f"トークン(文字列): {tokens_str}")
print(f"トークン(ID): {tokens_id}")

# N-best エンコード（複数の分割候補を取得）
nbest = sp.nbest_encode(text, nbest_size=5, out_type=str)
print(f"\nN-best 分割候補:")
for i, candidate in enumerate(nbest):
    print(f"  候補{i+1}: {candidate}")

# サンプリングエンコード（正則化効果）
for i in range(3):
    sampled = sp.encode(text, out_type=str, enable_sampling=True,
                        alpha=0.1, nbest_size=-1)
    print(f"サンプル{i+1}: {sampled}")
```

### 2.2 Unigram Language Model の仕組み

```
Unigram LM トークナイゼーション:

BPE (ボトムアップ):
  文字 → マージ → マージ → ... → 最終語彙
  (小さい語彙から大きく)

Unigram (トップダウン):
  巨大語彙 → 削除 → 削除 → ... → 最終語彙
  (大きな語彙から小さく)

手順:
1. 十分に大きな初期語彙を用意
   (例: 全てのサブストリングのうち頻出するもの)

2. 各語彙要素の確率 P(x_i) を EM アルゴリズムで推定

3. 各語彙要素を削除した場合の損失増加を計算:
   loss_i = -sum(log P(sentence | vocab \ {x_i}))

4. 損失増加が最も小さい語彙要素を削除
   (= 削除しても影響が少ないものを除去)

5. 目標語彙サイズになるまで 2-4 を繰り返す

利点:
- 確率的分割: 同じ単語に複数の分割方法があり得る
  → 学習時の正則化効果（Subword Regularization）
- 分割の品質がより最適に近い
```

### 比較表 1: トークナイゼーション手法の比較

| 手法 | 特徴 | 採用モデル | 日本語対応 | 語彙サイズ |
|------|------|-----------|-----------|-----------|
| BPE (Byte-level) | バイト単位で未知語なし | GPT-4, Claude | 良好 | 100K-200K |
| SentencePiece (Unigram) | 確率的サブワード分割 | LLaMA, Gemma | 良好 | 32K-128K |
| SentencePiece (BPE) | SPP フレームワーク上のBPE | T5, mBART | 良好 | 32K-64K |
| WordPiece | BPE の亜種 | BERT | 要調整 | 30K-50K |
| tiktoken | OpenAI 独自の高速BPE | GPT-4o | 良好 | 100K-200K |

### 比較表: 各トークナイザの実装詳細

| 特性 | tiktoken | SentencePiece | HF Tokenizers |
|------|----------|--------------|---------------|
| 実装言語 | Rust + Python | C++ + Python | Rust + Python |
| 速度 (MB/s) | ~100 | ~50 | ~80 |
| マルチスレッド | 対応 | 対応 | 対応 |
| ストリーミング | 対応 | 限定的 | 対応 |
| カスタム学習 | 不可 | 可能 | 可能 |
| 語彙の拡張 | 不可 | 可能 | 可能 |
| メモリ効率 | 高 | 中 | 高 |
| ライセンス | MIT | Apache 2.0 | Apache 2.0 |

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

### 3.1 コンテキストウィンドウの管理

コンテキストウィンドウとは、モデルが一度に処理できるトークンの最大数である。入力トークンと出力トークンの合計がこの上限を超えることはできない。

```
コンテキストウィンドウの構成:

┌──────────────────────────────────────────────────┐
│              コンテキストウィンドウ (例: 200K)       │
│                                                    │
│  ┌────────────┐  ┌──────────┐  ┌───────────────┐ │
│  │ システム    │  │ 会話     │  │ 予約出力      │ │
│  │ プロンプト  │  │ 履歴     │  │ (max_tokens)  │ │
│  │ (固定)     │  │ (可変)   │  │ (固定)        │ │
│  │ ~2K tokens │  │ ~190K   │  │ ~8K tokens    │ │
│  └────────────┘  └──────────┘  └───────────────┘ │
│                                                    │
│  使用可能な会話履歴 = ウィンドウサイズ               │
│                    - システムプロンプト              │
│                    - max_tokens (出力予約)           │
└──────────────────────────────────────────────────┘
```

```python
class ContextWindowManager:
    """コンテキストウィンドウを管理するユーティリティ"""

    MODEL_LIMITS = {
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
        "claude-3-5-sonnet": 200_000,
        "claude-3-5-haiku": 200_000,
        "gemini-1.5-pro": 1_000_000,
        "llama-3.1-8b": 128_000,
    }

    def __init__(self, model: str, max_output_tokens: int = 4096,
                 system_prompt_tokens: int = 0):
        self.model = model
        self.context_limit = self.MODEL_LIMITS.get(model, 128_000)
        self.max_output_tokens = max_output_tokens
        self.system_prompt_tokens = system_prompt_tokens

    @property
    def available_input_tokens(self) -> int:
        """入力に使える残りトークン数"""
        return (self.context_limit
                - self.max_output_tokens
                - self.system_prompt_tokens)

    def can_fit(self, input_tokens: int) -> bool:
        """入力がコンテキストウィンドウに収まるか"""
        return input_tokens <= self.available_input_tokens

    def truncate_messages(self, messages: list[dict],
                          token_counter,
                          strategy: str = "sliding_window") -> list[dict]:
        """メッセージ履歴をコンテキストに収まるよう切り詰める"""
        if strategy == "sliding_window":
            return self._sliding_window(messages, token_counter)
        elif strategy == "summarize_old":
            return self._summarize_old(messages, token_counter)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _sliding_window(self, messages: list[dict],
                         token_counter) -> list[dict]:
        """古いメッセージから削除する（最新を優先）"""
        result = []
        total_tokens = 0
        limit = self.available_input_tokens

        # 最新のメッセージから逆順に追加
        for msg in reversed(messages):
            msg_tokens = token_counter(msg["content"])
            if total_tokens + msg_tokens > limit:
                break
            result.insert(0, msg)
            total_tokens += msg_tokens

        return result

    def _summarize_old(self, messages: list[dict],
                        token_counter) -> list[dict]:
        """古いメッセージを要約して圧縮する"""
        # 実装例: 古い部分を要約 + 新しい部分をそのまま保持
        limit = self.available_input_tokens
        half_limit = limit // 2

        # 新しいメッセージ（後半）
        recent = []
        recent_tokens = 0
        for msg in reversed(messages):
            msg_tokens = token_counter(msg["content"])
            if recent_tokens + msg_tokens > half_limit:
                break
            recent.insert(0, msg)
            recent_tokens += msg_tokens

        # 古いメッセージ（前半）を要約
        old_messages = messages[:len(messages) - len(recent)]
        if old_messages:
            summary_msg = {
                "role": "system",
                "content": f"[以前の会話の要約: {len(old_messages)}件のメッセージ]"
            }
            return [summary_msg] + recent

        return recent

# 使用例
manager = ContextWindowManager(
    model="claude-3-5-sonnet",
    max_output_tokens=4096,
    system_prompt_tokens=500,
)

print(f"モデル: {manager.model}")
print(f"コンテキスト上限: {manager.context_limit:,} tokens")
print(f"入力使用可能: {manager.available_input_tokens:,} tokens")
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

### 3.2 バッチ処理でのトークン最適化

```python
import tiktoken
from typing import Generator

class BatchTokenOptimizer:
    """大量テキストを処理する際のトークン最適化"""

    def __init__(self, model: str = "gpt-4o",
                 max_tokens_per_batch: int = 100_000):
        self.enc = tiktoken.encoding_for_model(model)
        self.max_tokens_per_batch = max_tokens_per_batch

    def create_batches(self, texts: list[str],
                        max_tokens: int = None
                        ) -> Generator[list[str], None, None]:
        """テキストをトークン数ベースでバッチに分割する"""
        max_tokens = max_tokens or self.max_tokens_per_batch
        current_batch = []
        current_tokens = 0

        for text in texts:
            text_tokens = len(self.enc.encode(text))

            if current_tokens + text_tokens > max_tokens and current_batch:
                yield current_batch
                current_batch = []
                current_tokens = 0

            current_batch.append(text)
            current_tokens += text_tokens

        if current_batch:
            yield current_batch

    def truncate_to_tokens(self, text: str,
                            max_tokens: int) -> str:
        """テキストを指定トークン数以内に切り詰める"""
        tokens = self.enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.enc.decode(truncated_tokens)

    def split_by_tokens(self, text: str,
                         chunk_size: int,
                         overlap: int = 0) -> list[str]:
        """テキストをトークン数ベースでチャンクに分割する"""
        tokens = self.enc.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self.enc.decode(chunk_tokens))
            start += chunk_size - overlap

        return chunks

# 使用例
optimizer = BatchTokenOptimizer("gpt-4o")

# 大量テキストをバッチ処理
documents = [f"Document {i}: " + "テスト文章。" * 100 for i in range(50)]

for batch_idx, batch in enumerate(optimizer.create_batches(documents)):
    print(f"バッチ {batch_idx + 1}: {len(batch)} ドキュメント")

# テキストのトークンベース分割
long_text = "これは非常に長いテキストです。" * 500
chunks = optimizer.split_by_tokens(long_text, chunk_size=512, overlap=64)
print(f"チャンク数: {len(chunks)}")
```

### 比較表 2: モデル別トークナイザの特性

| モデル | トークナイザ | 語彙サイズ | 日本語効率 | 特殊トークン |
|--------|------------|-----------|-----------|-------------|
| GPT-4o | cl100k_base+ | ~200K | 高 (改善済) | <\|endoftext\|> 等 |
| Claude 3.5 | 独自 BPE | ~150K | 高 | 非公開 |
| Llama 3.1 | tiktoken 派生 | 128K | 中〜高 | <\|begin_of_text\|> 等 |
| Gemini 1.5 | SentencePiece | ~256K | 高 | 非公開 |
| Gemma 2 | SentencePiece | 256K | 高 | <bos>, <eos> 等 |

### 比較表: 言語別トークン効率の詳細

| 言語 | GPT-4o | Llama 3.1 | Gemini 1.5 | 備考 |
|------|--------|-----------|-----------|------|
| 英語 | 1文字≒0.25トークン | 1文字≒0.25トークン | 1文字≒0.25トークン | ほぼ同等 |
| 日本語 | 1文字≒0.7トークン | 1文字≒1.2トークン | 1文字≒0.6トークン | 差が大きい |
| 中国語 | 1文字≒0.6トークン | 1文字≒1.0トークン | 1文字≒0.5トークン | 漢字の処理差 |
| 韓国語 | 1文字≒0.8トークン | 1文字≒1.3トークン | 1文字≒0.7トークン | ハングル処理 |
| コード | 1文字≒0.3トークン | 1文字≒0.3トークン | 1文字≒0.3トークン | ほぼ同等 |
| 数式 | 1文字≒0.5トークン | 1文字≒0.5トークン | 1文字≒0.4トークン | 特殊記号依存 |

---

## 4. トークナイゼーションの実践的課題

### 4.1 特殊文字と Unicode の扱い

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

# 特殊文字のトークン化を検証
test_cases = {
    "絵文字": "🎉🚀💡🔥",
    "数式記号": "∑∫∂∇∞",
    "CJK拡張": "𠮷𩸽𠀋",
    "制御文字": "タブ\tと\n改行",
    "ゼロ幅文字": "hello\u200bworld",  # ゼロ幅スペース
    "結合文字": "がぎぐげご",  # 半濁点・濁点
    "URL": "https://example.com/path?q=test&lang=ja",
    "JSON": '{"key": "value", "num": 42}',
    "コード": 'def hello():\n    print("Hello")',
}

for name, text in test_cases.items():
    tokens = enc.encode(text)
    print(f"{name:<12}: {len(text):>3}文字 → {len(tokens):>3}トークン "
          f"(効率: {len(text)/len(tokens):.2f}文字/トークン)")
```

### 4.2 トークン境界の問題

トークンの分割位置が意味的に不適切な場合、モデルの理解に影響を及ぼすことがある。

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

# トークン境界の問題を可視化
def visualize_tokenization(text: str):
    """トークン分割を視覚的に表示する"""
    tokens = enc.encode(text)
    result = []
    for token_id in tokens:
        token_str = enc.decode([token_id])
        result.append(f"[{token_str}]")
    print(f"原文: {text}")
    print(f"分割: {''.join(result)}")
    print(f"トークン数: {len(tokens)}")
    print()

# 問題のある例
visualize_tokenization("unhappiness")      # 正常: [un][happiness]
visualize_tokenization("123456789")         # 数字の分割
visualize_tokenization("user@example.com")  # メールアドレス
visualize_tokenization("2024-03-15T10:30:00Z")  # ISO日時
visualize_tokenization("192.168.1.1")       # IPアドレス
visualize_tokenization("東京都千代田区丸の内1-1-1")  # 日本語住所
```

### 4.3 プロンプトインジェクションとトークン化

```python
# トークナイゼーションを利用したプロンプトインジェクション攻撃の例と対策

class TokenSanitizer:
    """トークンレベルでの入力サニタイズ"""

    DANGEROUS_TOKEN_PATTERNS = [
        b"<|im_start|>",   # ChatML インジェクション
        b"<|im_end|>",
        b"<|endoftext|>",
        b"[INST]",          # Llama フォーマット
        b"[/INST]",
        b"<s>",
        b"</s>",
    ]

    def __init__(self, model: str = "gpt-4o"):
        import tiktoken
        self.enc = tiktoken.encoding_for_model(model)

    def sanitize(self, text: str) -> str:
        """危険なトークンパターンを除去する"""
        sanitized = text
        for pattern in self.DANGEROUS_TOKEN_PATTERNS:
            pattern_str = pattern.decode("utf-8", errors="ignore")
            if pattern_str in sanitized:
                sanitized = sanitized.replace(pattern_str, "")
        return sanitized

    def validate_token_count(self, text: str,
                              max_tokens: int) -> tuple[bool, int]:
        """トークン数が上限以内か検証する"""
        tokens = self.enc.encode(text)
        return len(tokens) <= max_tokens, len(tokens)
```

---

## 5. トラブルシューティング

### 5.1 よくある問題と対処法

| 問題 | 原因 | 対処法 |
|------|------|--------|
| トークン数が予想より多い | 日本語テキストの効率が悪い | トークン数ベースでチャンク分割 |
| 復元時に文字化け | マルチバイト文字がトークン境界で分断 | バイトレベルBPE使用モデルに変更 |
| API呼び出しが失敗する | コンテキスト長超過 | ContextWindowManager で管理 |
| コストが予算を超過 | 出力トークンの過小見積もり | max_tokens 設定 + コスト追跡 |
| トークン化速度が遅い | 大量テキストの逐次処理 | バッチ処理 + マルチスレッド |
| 異なるモデル間でトークン数不一致 | トークナイザの違い | モデル固有のカウンターを使用 |

### 5.2 デバッグテクニック

```python
def debug_tokenization(text: str, models: list[str] = None):
    """複数モデルのトークン化結果をデバッグ表示する"""
    import tiktoken

    if models is None:
        models = ["gpt-4o", "gpt-4o-mini"]

    print(f"テキスト: {text[:50]}{'...' if len(text) > 50 else ''}")
    print(f"文字数: {len(text)}, バイト数: {len(text.encode('utf-8'))}")
    print("-" * 60)

    for model in models:
        try:
            enc = tiktoken.encoding_for_model(model)
            tokens = enc.encode(text)
            decoded_tokens = [enc.decode([t]) for t in tokens]

            print(f"\n{model}:")
            print(f"  トークン数: {len(tokens)}")
            print(f"  文字/トークン: {len(text)/len(tokens):.2f}")
            print(f"  先頭5トークン: {decoded_tokens[:5]}")
            print(f"  末尾5トークン: {decoded_tokens[-5:]}")
        except Exception as e:
            print(f"  {model}: エラー - {e}")

# デバッグ実行
debug_tokenization("Transformerアーキテクチャは自然言語処理に革命をもたらした。")
```

---

## 6. パフォーマンス最適化

### 6.1 トークン化のベンチマーク

```python
import time
import tiktoken

def benchmark_tokenizer(text: str, iterations: int = 1000):
    """トークナイザの速度をベンチマークする"""
    enc = tiktoken.encoding_for_model("gpt-4o")

    # エンコード速度
    start = time.perf_counter()
    for _ in range(iterations):
        tokens = enc.encode(text)
    encode_time = (time.perf_counter() - start) / iterations

    # デコード速度
    tokens = enc.encode(text)
    start = time.perf_counter()
    for _ in range(iterations):
        enc.decode(tokens)
    decode_time = (time.perf_counter() - start) / iterations

    text_bytes = len(text.encode("utf-8"))
    print(f"テキストサイズ: {text_bytes:,} bytes")
    print(f"トークン数: {len(tokens):,}")
    print(f"エンコード: {encode_time*1000:.3f} ms "
          f"({text_bytes/encode_time/1024/1024:.1f} MB/s)")
    print(f"デコード: {decode_time*1000:.3f} ms "
          f"({text_bytes/decode_time/1024/1024:.1f} MB/s)")

# ベンチマーク実行
short_text = "Hello, World!" * 10
long_text = "大規模言語モデルの性能は素晴らしい。" * 1000

print("=== 短いテキスト ===")
benchmark_tokenizer(short_text)
print("\n=== 長いテキスト ===")
benchmark_tokenizer(long_text, iterations=100)
```

### 6.2 メモリ効率の最適化

```python
class StreamingTokenCounter:
    """ストリーミング方式でメモリ効率的にトークン数をカウントする"""

    def __init__(self, model: str = "gpt-4o"):
        import tiktoken
        self.enc = tiktoken.encoding_for_model(model)

    def count_file(self, filepath: str,
                    chunk_size: int = 1024 * 1024) -> int:
        """ファイルをチャンク読みしてトークン数をカウント"""
        total_tokens = 0

        with open(filepath, "r", encoding="utf-8") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                total_tokens += len(self.enc.encode(chunk))

        return total_tokens

    def count_streaming(self, text_generator) -> int:
        """ジェネレータからストリーミングでカウント"""
        total_tokens = 0
        for text in text_generator:
            total_tokens += len(self.enc.encode(text))
        return total_tokens
```

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

### アンチパターン 3: トークナイザの不一致

```
誤: GPT-4o のトークナイザで Claude のトークン数を見積もる
  → 実際のトークン数と乖離し、コスト見積もりが不正確

# 悪い例
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
claude_tokens = len(enc.encode(text))  # ← Claude のトークン数ではない!

# 良い例: 各プロバイダのトークンカウントAPIを使用
# Anthropic: response.usage.input_tokens で正確なカウントを取得
# または: anthropic.count_tokens() メソッド（利用可能な場合）
```

### アンチパターン 4: 特殊トークンの無視

```
誤: テキストのトークン数だけでコンテキスト使用量を計算する
  → 特殊トークン（BOS, EOS, 区切り記号等）が追加される

# 実際のトークン使用量:
# テキストトークン + 特殊トークン + メッセージフォーマットのオーバーヘッド
# OpenAI: 各メッセージに約4トークンのオーバーヘッド
# Claude: メッセージ構造に応じた追加トークン

# 正確なカウントにはAPIのusageフィールドを参照すべき
```

---

## FAQ

### Q1: トークナイザが異なるモデル間でトークン数は比較できますか？

**A:** 正確な比較はできません。同じテキストでも、GPT-4o と Llama 3 ではトークン数が異なります。コスト比較する場合は、各モデルのトークナイザで個別にカウントする必要があります。ただし大まかな目安として、英語では 1 トークン ≒ 4 文字、日本語では 1 トークン ≒ 1〜2 文字が目安になります。

### Q2: コンテキストウィンドウを超えた場合どうなりますか？

**A:** API はエラーを返します。対策としては、(1) テキストを要約して短縮、(2) チャンク分割して複数回に分けて処理、(3) RAG で関連部分だけ取得、(4) より長いコンテキスト長を持つモデルに切り替え、があります。

### Q3: 日本語で最もトークン効率の良いモデルは？

**A:** 2024年時点では、GPT-4o と Gemini 1.5 が日本語のトークン効率に優れています。特に GPT-4o は前世代から大幅に改善されました。Claude 3.5 も高い日本語トークン効率を持ちます。ただし、トークン効率だけでなく、単価との掛け算で実際のコストを評価してください。

### Q4: カスタムトークナイザを作成すべきケースはどのような場合ですか？

**A:** 以下のケースでカスタムトークナイザの検討が有効です。
- **ドメイン固有の専門用語**が多い場合（医療、法律、化学式等）
- **特殊な記号体系**を扱う場合（プログラミング言語、数式、楽譜等）
- **ローカルLLM**を自前で学習・ファインチューニングする場合
- **トークン効率**がコストに直結する大規模バッチ処理の場合

ただし、API経由で既存モデルを利用する場合は、そのモデルのトークナイザに合わせる必要があるため、カスタムトークナイザは使えません。

### Q5: Prompt Caching はトークンコストにどう影響しますか？

**A:** Prompt Caching を使うと、キャッシュされたプロンプト部分の入力コストが大幅に削減されます。Anthropic の場合、キャッシュヒット時の入力料金は通常の 10% になります。OpenAI も同様のキャッシュ機構を提供しています。長いシステムプロンプトや Few-shot 例を繰り返し使用する場合に特に効果的です。

### Q6: マルチモーダル入力（画像等）のトークン換算はどうなりますか？

**A:** 画像はピクセル数に基づいてトークン数に換算されます。
- **OpenAI GPT-4o**: 低解像度で約85トークン、高解像度で最大約1,700トークン（512x512タイルあたり170トークン）
- **Claude**: 画像サイズに応じて自動計算（おおよそ 1,000x1,000px で約 1,600トークン）
- **Gemini**: 画像1枚あたり約258トークン（固定）

---

## まとめ

| 項目 | 要点 |
|------|------|
| BPE | 最頻ペアを統合してサブワード語彙を構築する手法 |
| SentencePiece | 言語非依存のトークナイゼーションフレームワーク |
| tiktoken | OpenAI の高速 BPE 実装、GPT モデルで使用 |
| Byte-Level BPE | バイト単位で処理、未知語が原理的に発生しない |
| Unigram LM | トップダウン方式、確率的分割で正則化効果 |
| 日本語効率 | 英語の 1.5〜2 倍のトークンが必要な場合が多い |
| コスト管理 | 入力/出力トークン数の把握とプロンプト最適化が重要 |
| コンテキスト管理 | スライディングウィンドウや要約でウィンドウ内に収める |
| 語彙サイズ | 32K〜256K の範囲で、大きいほど効率的だが学習コスト増 |
| Prompt Caching | 繰り返しプロンプトのコストを最大 90% 削減可能 |

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
4. Kudo, T. (2018). "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates." *ACL 2018*. https://arxiv.org/abs/1804.10959
5. Radford, A. et al. (2019). "Language Models are Unsupervised Multitask Learners." *OpenAI*. (GPT-2 Byte-Level BPE)
6. Hugging Face. "Summary of the tokenizers." https://huggingface.co/docs/transformers/tokenizer_summary

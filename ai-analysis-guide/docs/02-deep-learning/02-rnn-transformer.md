# RNN/Transformer

> 系列データ処理の進化を RNN から Transformer まで辿り、LSTM、Attention、BERT の仕組みと応用を実践的に理解する

## この章で学ぶこと

1. **RNN の基礎と限界** — 時系列処理、勾配消失問題、LSTM/GRU による解決
2. **Attention メカニズム** — Self-Attention、Multi-Head Attention の計算原理
3. **Transformer アーキテクチャ** — Encoder-Decoder 構造、BERT、GPT の設計思想

---

## 1. RNN の基礎

```
RNN の展開図
=============

       h0    h1    h2    h3
        |     |     |     |
  x0 ->[RNN]->[RNN]->[RNN]->[RNN]-> y
        |     |     |     |
       "I"  "love" "deep" "learning"

各ステップ:
  h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
  y_t = W_hy * h_t

問題: 長い系列で勾配消失/勾配爆発
  h100 への勾配 = d(loss)/d(h0) --> W_hh を100回掛ける
  |W| < 1: 勾配消失 (情報が消える)
  |W| > 1: 勾配爆発 (値が発散)
```

### コード例 1: LSTM の構造

```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """LSTM による系列分類"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,  # 双方向 LSTM
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))  # [batch, seq_len, embed_dim]
        output, (hidden, cell) = self.lstm(embedded)
        # hidden: [num_layers*2, batch, hidden_dim]
        # 最後の層の forward と backward を結合
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        logits = self.classifier(self.dropout(hidden_cat))
        return logits

# 使用例
model = LSTMModel(vocab_size=30000, embed_dim=256, hidden_dim=512, num_classes=5)
```

```
LSTM セルの内部構造
=====================

         c_{t-1} ----[x]--------[+]----> c_t
                      |          |
                   [forget]   [input * candidate]
                      |          |        |
         h_{t-1} --> [f_t]    [i_t]    [~c_t]
                      |          |        |
                   sigmoid    sigmoid    tanh
                      |          |        |
                   +--+--+   +--+--+ +--+--+
                   | W_f |   | W_i | | W_c |
                   +-----+   +-----+ +-----+
                      ^          ^        ^
                   [h_{t-1}, x_t] を入力

  f_t: 忘却ゲート (何を忘れるか)
  i_t: 入力ゲート (何を記憶するか)
  o_t: 出力ゲート (何を出力するか)
```

---

## 2. Attention メカニズム

```
Self-Attention の計算
======================

入力: "The cat sat"

1. Q, K, V を計算
   Q = X * W_Q    (Query: 「何を探すか」)
   K = X * W_K    (Key: 「何を持っているか」)
   V = X * W_V    (Value: 「実際の値」)

2. Attention スコア
   Score = Q * K^T / sqrt(d_k)

3. Softmax で正規化
   Attention = softmax(Score)

4. 重み付き和
   Output = Attention * V

       The   cat   sat
  The [ 0.7  0.2  0.1 ]    <-- "The" は自身に最も注目
  cat [ 0.1  0.6  0.3 ]    <-- "cat" は自身と "sat" に注目
  sat [ 0.2  0.5  0.3 ]    <-- "sat" は "cat" に最も注目
```

### コード例 2: Self-Attention の実装

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [batch, heads, seq_len, d_k]
    K: [batch, heads, seq_len, d_k]
    V: [batch, heads, seq_len, d_v]
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 線形変換 + マルチヘッド分割
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention 計算
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # ヘッド結合 + 出力変換
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.W_o(attn_output)
        return output
```

---

## 3. Transformer アーキテクチャ

### コード例 3: Transformer Encoder

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention + Residual + LayerNorm
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-Forward + Residual + LayerNorm
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class TextClassifier(nn.Module):
    """Transformer ベースのテキスト分類器"""

    def __init__(self, vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, num_classes=5, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.dropout(self.embedding(x) + self.pos_encoding(positions))

        for layer in self.layers:
            x = layer(x)

        # [CLS] トークンの表現を分類に使用
        cls_output = x[:, 0]
        return self.classifier(cls_output)
```

### コード例 4: Hugging Face による BERT 活用

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

# 事前学習済みモデルのロード
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=3,
)

# テキストのトークナイズ
texts = ["この映画は素晴らしい", "つまらない作品だった"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 推論
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)

# ファインチューニング
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

---

## 4. モデル比較

### RNN vs Transformer 比較表

| 特性 | RNN (LSTM/GRU) | Transformer |
|---|---|---|
| **並列計算** | 不可（逐次処理） | 可能（全位置を同時処理） |
| **長距離依存** | 苦手（勾配消失） | 得意（直接参照） |
| **計算量** | O(n * d^2) | O(n^2 * d) |
| **メモリ** | O(d) | O(n^2) |
| **学習速度** | 遅い | 速い（GPU 並列化） |
| **短い系列** | 効率的 | オーバーヘッドあり |
| **長い系列** | 性能劣化 | 優秀（ただしメモリ制約） |

### 主要 Transformer モデル比較表

| モデル | 構造 | パラメータ | 学習方法 | 主な用途 |
|---|---|---|---|---|
| **BERT** | Encoder のみ | 110M/340M | マスク言語モデル | 分類、QA、NER |
| **GPT-4** | Decoder のみ | 非公開 | 次トークン予測 | テキスト生成 |
| **T5** | Encoder-Decoder | 220M-11B | Text-to-Text | 翻訳、要約、QA |
| **ViT** | Encoder のみ | 86M-632M | 画像パッチ | 画像分類 |
| **Whisper** | Encoder-Decoder | 39M-1.5B | 音声-テキスト | 音声認識 |

---

## コード例 5: 位置エンコーディング

```python
class SinusoidalPositionalEncoding(nn.Module):
    """Transformer 原論文の正弦波位置エンコーディング"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

---

## アンチパターン

### 1. 小規模データで大規模 Transformer を学習

**問題**: BERT-large を100件のデータでファインチューニングすると過学習する。パラメータ数がデータ数を大幅に上回る場合、汎化性能が極端に低下する。

**対策**: 小規模データには軽量モデル（DistilBERT）か、Few-shot learning（プロンプト設計）を使用する。データ拡張も検討する。

### 2. Attention の計算量を無視した設計

**問題**: Self-Attention は系列長の2乗のメモリを消費する。長文（10,000トークン以上）を処理しようとすると GPU メモリが不足する。

**対策**: Longformer（局所 + グローバル Attention）、Flash Attention（メモリ効率化）、または入力のチャンク分割を検討する。

---

## FAQ

### Q1: RNN はもう使わないのですか？

**A**: Transformer が主流ですが、RNN にも適用場面があります。リアルタイムのストリーミング処理、メモリ制約のあるエッジデバイス、短い固定長系列の処理では LSTM/GRU が効率的です。Mamba 等の State Space Model も RNN 的な逐次処理の利点を活かしつつ Transformer に匹敵する性能を達成しています。

### Q2: BERT と GPT の使い分けは？

**A**: BERT は双方向の文脈理解に優れ、分類・情報抽出・質問応答に最適です。GPT は自己回帰（左から右）で、テキスト生成・要約・翻訳に最適です。理解系タスクなら BERT 系、生成系タスクなら GPT 系を選択してください。

### Q3: Transformer の学習にはどれくらいの GPU が必要ですか？

**A**: ファインチューニングなら BERT-base で 1 GPU（16GB VRAM）、BERT-large で 1-2 GPU が目安です。事前学習は大規模計算資源が必要で、BERT-base でも数十 GPU-day かかります。

---

## まとめ

| 項目 | 要点 |
|---|---|
| RNN | 時系列の逐次処理。勾配消失で長距離依存が苦手 |
| LSTM/GRU | ゲート機構で勾配消失を緩和。中程度の系列に有効 |
| Self-Attention | 全位置間の関係を直接計算。長距離依存に強い |
| Multi-Head | 複数の観点から Attention を計算。表現力向上 |
| Transformer | Attention ベースの並列処理可能なアーキテクチャ |
| BERT | 双方向 Encoder。分類・理解タスクの標準 |
| GPT | 自己回帰 Decoder。テキスト生成の標準 |

## 次に読むべきガイド

- [コンピュータビジョン](../03-applied/01-computer-vision.md) — Vision Transformer の応用
- [MLOps](../03-applied/02-mlops.md) — Transformer モデルのデプロイと運用

## 参考文献

1. **Vaswani et al.**: [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — Transformer の原論文
2. **Devlin et al.**: [BERT: Pre-training of Deep Bidirectional Transformers (2018)](https://arxiv.org/abs/1810.04805) — BERT の原論文
3. **Jay Alammar**: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Transformer の視覚的解説

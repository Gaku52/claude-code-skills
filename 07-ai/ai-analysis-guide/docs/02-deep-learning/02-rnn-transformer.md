# RNN/Transformer

> 系列データ処理の進化を RNN から Transformer まで辿り、LSTM、Attention、BERT の仕組みと応用を実践的に理解する

## この章で学ぶこと

1. **RNN の基礎と限界** — 時系列処理、勾配消失問題、LSTM/GRU による解決
2. **Attention メカニズム** — Self-Attention、Multi-Head Attention の計算原理
3. **Transformer アーキテクチャ** — Encoder-Decoder 構造、BERT、GPT の設計思想
4. **実践的な学習・推論テクニック** — ファインチューニング、量子化、効率的推論
5. **最新動向** — State Space Models、Mixture of Experts、長文脈対応

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

### 1.1 RNN の数学的基礎

RNN は再帰的な構造を持つニューラルネットワークで、系列データの各ステップで隠れ状態を更新する。基本的な計算は以下の通り:

```
Vanilla RNN の順伝播:
=====================

入力: x = (x_1, x_2, ..., x_T)  系列長 T

各タイムステップ t:
  a_t = W_hh * h_{t-1} + W_xh * x_t + b_h    (活性化前の値)
  h_t = tanh(a_t)                               (隠れ状態)
  o_t = W_hy * h_t + b_y                       (出力)
  y_t = softmax(o_t)                            (予測)

パラメータ:
  W_xh ∈ R^{H×D}  (入力→隠れ)
  W_hh ∈ R^{H×H}  (隠れ→隠れ)
  W_hy ∈ R^{V×H}  (隠れ→出力)
  b_h ∈ R^H, b_y ∈ R^V

逆伝播 (BPTT: Backpropagation Through Time):
  ∂L/∂W_hh = Σ_t ∂L_t/∂W_hh

  ∂L_t/∂h_k = (∏_{i=k+1}^{t} diag(1-h_i²) * W_hh) * ∂L_t/∂h_t

  → T-k 回の行列積で勾配が指数的に変化
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

### コード例 1.5: GRU の実装と LSTM との比較

```python
import torch
import torch.nn as nn
import time

class GRUModel(nn.Module):
    """GRU による系列分類（LSTM より軽量）"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded)  # GRU: cell state なし
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        logits = self.classifier(self.dropout(hidden_cat))
        return logits


def compare_rnn_variants():
    """LSTM と GRU のパラメータ数・速度を比較"""
    vocab_size = 30000
    embed_dim = 256
    hidden_dim = 512
    num_classes = 5
    seq_len = 100
    batch_size = 32

    models = {
        "LSTM": LSTMModel(vocab_size, embed_dim, hidden_dim, num_classes),
        "GRU": GRUModel(vocab_size, embed_dim, hidden_dim, num_classes),
    }

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    for name, model in models.items():
        # パラメータ数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 推論速度
        model.eval()
        with torch.no_grad():
            start = time.time()
            for _ in range(100):
                _ = model(dummy_input)
            elapsed = time.time() - start

        print(f"{name}:")
        print(f"  パラメータ数: {total_params:,}")
        print(f"  学習可能: {trainable_params:,}")
        print(f"  推論時間 (100回): {elapsed:.3f}秒")
        print()

compare_rnn_variants()
```

```
GRU セルの内部構造
===================

  GRU は LSTM の簡略版（2ゲート）:

  z_t = σ(W_z * [h_{t-1}, x_t])       更新ゲート
  r_t = σ(W_r * [h_{t-1}, x_t])       リセットゲート
  ~h_t = tanh(W * [r_t ⊙ h_{t-1}, x_t])  候補状態
  h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ ~h_t

  LSTM vs GRU:
  ┌─────────────┬──────────────┬──────────────┐
  │             │ LSTM         │ GRU          │
  ├─────────────┼──────────────┼──────────────┤
  │ ゲート数    │ 3 (f, i, o) │ 2 (z, r)     │
  │ 状態数      │ 2 (h, c)    │ 1 (h)        │
  │ パラメータ  │ 4 × 行列    │ 3 × 行列     │
  │ メモリ      │ 多い        │ 少ない       │
  │ 長距離依存  │ やや良い    │ 同等〜やや劣 │
  │ 学習速度    │ 遅い        │ 速い         │
  │ 一般的用途  │ 長い系列    │ 短〜中の系列 │
  └─────────────┴──────────────┴──────────────┘
```

### コード例 1.7: 時系列予測 (LSTM)

```python
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    """スライディングウィンドウで時系列データセットを作成"""

    def __init__(self, data, window_size, forecast_horizon=1):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.data) - self.window_size - self.forecast_horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size].unsqueeze(-1)  # [window, 1]
        y = self.data[idx + self.window_size:idx + self.window_size + self.forecast_horizon]
        return x, y


class LSTMForecaster(nn.Module):
    """LSTM による時系列予測モデル"""

    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2,
                 forecast_horizon=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_horizon),
        )

    def forward(self, x):
        # x: [batch, window_size, input_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 最後のタイムステップの隠れ状態を使用
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_dim]
        prediction = self.fc(last_hidden)  # [batch, forecast_horizon]
        return prediction


def train_forecaster(data, window_size=30, forecast_horizon=7,
                     epochs=50, lr=0.001, batch_size=32):
    """時系列予測モデルの学習"""
    # データ分割
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    val_data = data[train_size:]

    # 正規化
    mean, std = train_data.mean(), train_data.std()
    train_normalized = (train_data - mean) / std
    val_normalized = (val_data - mean) / std

    # データセット
    train_ds = TimeSeriesDataset(train_normalized, window_size, forecast_horizon)
    val_ds = TimeSeriesDataset(val_normalized, window_size, forecast_horizon)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # モデル
    model = LSTMForecaster(
        input_dim=1, hidden_dim=64, num_layers=2,
        forecast_horizon=forecast_horizon
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # 学習
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # 検証
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                pred = model(x_batch)
                val_loss += criterion(pred, y_batch).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_forecaster.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 10:
            print(f"Early stopping at epoch {epoch}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    return model, mean, std

# 使用例
# data = np.sin(np.linspace(0, 100, 1000)) + np.random.randn(1000) * 0.1
# model, mean, std = train_forecaster(data)
```

### コード例 1.8: Seq2Seq モデル（Encoder-Decoder RNN）

```python
import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    """Seq2Seq エンコーダ"""

    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))  # [batch, src_len, embed_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    """Seq2Seq デコーダ"""

    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell):
        # input_token: [batch, 1]
        embedded = self.dropout(self.embedding(input_token))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))  # [batch, output_dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """Encoder-Decoder Seq2Seq モデル"""

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)

        # 最初の入力は <SOS> トークン
        input_token = trg[:, 0:1]  # [batch, 1]

        for t in range(1, trg_len):
            prediction, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t] = prediction

            # Teacher Forcing: 一定確率で正解を次の入力にする
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(dim=1, keepdim=True)
            input_token = trg[:, t:t+1] if teacher_force else top1

        return outputs

# 使用例
INPUT_DIM = 10000   # ソース語彙サイズ
OUTPUT_DIM = 8000   # ターゲット語彙サイズ
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
decoder = Decoder(OUTPUT_DIM, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)

print(f"Seq2Seq パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
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

### 2.1 Attention の種類

```
Attention メカニズムの分類
============================

1. Additive Attention (Bahdanau, 2014)
   score(s_i, h_j) = v^T * tanh(W_1 * s_i + W_2 * h_j)
   → Encoder-Decoder 間で使用
   → 計算量: O(d)

2. Dot-Product Attention (Luong, 2015)
   score(s_i, h_j) = s_i^T * h_j
   → 高速だがスケーリング問題
   → 計算量: O(1)

3. Scaled Dot-Product Attention (Vaswani, 2017)
   score(Q, K) = Q * K^T / sqrt(d_k)
   → Transformer で使用
   → sqrt(d_k) で勾配の安定化

4. Multi-Head Attention
   head_i = Attention(Q*W_Q_i, K*W_K_i, V*W_V_i)
   MultiHead = Concat(head_1, ..., head_h) * W_O
   → 異なる部分空間で異なるパターンを学習

5. Cross-Attention
   Q: Decoder の状態
   K, V: Encoder の出力
   → Encoder-Decoder 間の情報伝達

6. Causal (Masked) Attention
   未来のトークンへの Attention をマスク
   → GPT などの自己回帰モデルで使用
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

### コード例 2.5: Attention の可視化

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(tokens, attention_weights, layer=0, head=0, save_path=None):
    """
    Attention の重みをヒートマップで可視化

    Args:
        tokens: トークンのリスト
        attention_weights: [layers, heads, seq_len, seq_len] のテンソル
        layer: 可視化する層
        head: 可視化するヘッド
    """
    attn = attention_weights[layer][head].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attn, cmap="viridis", vmin=0, vmax=1)

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(tokens, fontsize=10)

    # 各セルに値を表示
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            ax.text(j, i, f"{attn[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if attn[i, j] > 0.5 else "black")

    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    ax.set_title(f"Attention Weights (Layer {layer}, Head {head})")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def extract_bert_attention(model, tokenizer, text):
    """BERT から Attention weights を抽出"""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # attentions: tuple of (batch, heads, seq_len, seq_len) per layer
    attentions = outputs.attentions
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # 各層のAttentionを可視化
    all_attentions = torch.stack([a.squeeze(0) for a in attentions])
    return tokens, all_attentions


def attention_rollout(attentions, head_fusion="mean"):
    """
    Attention Rollout: 複数層のAttentionを統合して
    入力トークンへの最終的な寄与度を計算

    Args:
        attentions: [num_layers, num_heads, seq_len, seq_len]
        head_fusion: ヘッドの統合方法 ("mean", "max", "min")
    """
    num_layers = attentions.shape[0]
    seq_len = attentions.shape[-1]

    # 残差接続を考慮してAttentionに単位行列を加算
    result = torch.eye(seq_len)

    for layer in range(num_layers):
        if head_fusion == "mean":
            attn = attentions[layer].mean(dim=0)
        elif head_fusion == "max":
            attn = attentions[layer].max(dim=0).values
        elif head_fusion == "min":
            attn = attentions[layer].min(dim=0).values

        # 残差接続の影響
        attn = 0.5 * attn + 0.5 * torch.eye(seq_len)
        # 行方向に正規化
        attn = attn / attn.sum(dim=-1, keepdim=True)
        result = torch.matmul(attn, result)

    return result
```

### コード例 2.7: Bahdanau Attention (Additive Attention)

```python
import torch
import torch.nn as nn

class BahdanauAttention(nn.Module):
    """Additive Attention (Bahdanau et al., 2014)"""

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.W_encoder = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.W_decoder = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        encoder_outputs: [batch, src_len, encoder_dim]
        decoder_hidden: [batch, decoder_dim]
        """
        # decoder_hidden を src_len 分繰り返す
        decoder_hidden = decoder_hidden.unsqueeze(1)  # [batch, 1, decoder_dim]

        # スコア計算
        energy = torch.tanh(
            self.W_encoder(encoder_outputs) + self.W_decoder(decoder_hidden)
        )  # [batch, src_len, attention_dim]

        scores = self.v(energy).squeeze(-1)  # [batch, src_len]
        attention_weights = torch.softmax(scores, dim=-1)  # [batch, src_len]

        # コンテキストベクトル
        context = torch.bmm(
            attention_weights.unsqueeze(1), encoder_outputs
        ).squeeze(1)  # [batch, encoder_dim]

        return context, attention_weights


class AttentionDecoder(nn.Module):
    """Attention 付き Decoder"""

    def __init__(self, output_dim, embed_dim, encoder_dim,
                 decoder_dim, attention_dim, dropout=0.3):
        super().__init__()
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.rnn = nn.GRU(embed_dim + encoder_dim, decoder_dim, batch_first=True)
        self.fc_out = nn.Linear(decoder_dim + encoder_dim + embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, decoder_hidden, encoder_outputs):
        """
        input_token: [batch]
        decoder_hidden: [1, batch, decoder_dim]
        encoder_outputs: [batch, src_len, encoder_dim]
        """
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))  # [batch, 1, embed_dim]

        context, attn_weights = self.attention(
            encoder_outputs, decoder_hidden.squeeze(0)
        )
        context = context.unsqueeze(1)  # [batch, 1, encoder_dim]

        rnn_input = torch.cat([embedded, context], dim=2)  # [batch, 1, embed_dim + encoder_dim]
        output, hidden = self.rnn(rnn_input, decoder_hidden)

        prediction = self.fc_out(
            torch.cat([output.squeeze(1), context.squeeze(1), embedded.squeeze(1)], dim=1)
        )  # [batch, output_dim]

        return prediction, hidden, attn_weights
```

---

## 3. Transformer アーキテクチャ

```
Transformer の全体構造
========================

       入力テキスト                    出力テキスト
           |                              |
    [Input Embedding]              [Output Embedding]
    [+ Positional Enc]             [+ Positional Enc]
           |                              |
    ┌──────┴──────┐                ┌──────┴──────┐
    │  Encoder x N │                │  Decoder x N │
    │             │                │             │
    │ ┌─────────┐ │                │ ┌─────────┐ │
    │ │Self-Attn│ │                │ │Masked    │ │
    │ │+ ResConn│ │                │ │Self-Attn │ │
    │ │+ LN     │ │                │ │+ ResConn │ │
    │ └────┬────┘ │                │ │+ LN      │ │
    │ ┌────┴────┐ │                │ └────┬────┘ │
    │ │FFN      │ │   K,V          │ ┌────┴────┐ │
    │ │+ ResConn│ │───────────────→│ │Cross-Attn│ │
    │ │+ LN     │ │                │ │+ ResConn │ │
    │ └─────────┘ │                │ │+ LN      │ │
    └─────────────┘                │ └────┬────┘ │
                                   │ ┌────┴────┐ │
                                   │ │FFN      │ │
                                   │ │+ ResConn│ │
                                   │ │+ LN     │ │
                                   │ └─────────┘ │
                                   └──────┬──────┘
                                          |
                                   [Linear + Softmax]
                                          |
                                      確率分布
```

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

### コード例 3.5: Transformer Decoder

```python
import torch
import torch.nn as nn

class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer（Causal Attention + Cross-Attention）"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # Cross-Attention (Encoder-Decoder)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        # Feed-Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked Self-Attention
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Cross-Attention
        cross_out = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_out))

        # Feed-Forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class TransformerModel(nn.Module):
    """完全な Transformer (Encoder + Decoder)"""

    def __init__(self, src_vocab, tgt_vocab, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 d_ff=2048, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.output_proj = nn.Linear(d_model, tgt_vocab)
        self.dropout = nn.Dropout(dropout)

    def generate_causal_mask(self, seq_len, device):
        """未来のトークンをマスクする Causal Mask を生成"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.bool()
        return ~mask  # True = attend, False = mask

    def encode(self, src):
        x = self.dropout(self.pos_encoding(
            self.src_embedding(src) * (self.d_model ** 0.5)
        ))
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode(self, tgt, encoder_output, tgt_mask=None):
        x = self.dropout(self.pos_encoding(
            self.tgt_embedding(tgt) * (self.d_model ** 0.5)
        ))
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask=tgt_mask)
        return self.output_proj(x)

    def forward(self, src, tgt):
        tgt_mask = self.generate_causal_mask(tgt.size(1), tgt.device)
        encoder_output = self.encode(src)
        output = self.decode(tgt, encoder_output, tgt_mask)
        return output
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

### コード例 4.5: BERT ファインチューニングの完全パイプライン

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

class TextDataset(Dataset):
    """テキスト分類用データセット"""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class BERTFineTuner:
    """BERT ファインチューニングの完全パイプライン"""

    def __init__(self, model_name="bert-base-multilingual-cased",
                 num_labels=3, max_length=256, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
        self.max_length = max_length

    def prepare_data(self, texts, labels, test_size=0.2, batch_size=16):
        """データの分割とDataLoader作成"""
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, stratify=labels, random_state=42
        )

        self.train_dataset = TextDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        self.val_dataset = TextDataset(
            val_texts, val_labels, self.tokenizer, self.max_length
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size
        )

    def train(self, epochs=3, lr=2e-5, warmup_ratio=0.1,
              weight_decay=0.01, max_grad_norm=1.0):
        """学習ループ"""
        # オプティマイザ（bias と LayerNorm に weight_decay を適用しない）
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

        # スケジューラ
        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        best_val_acc = 0.0

        for epoch in range(epochs):
            # 学習フェーズ
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch in self.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += len(batch["labels"])

            train_acc = correct / total
            avg_loss = total_loss / len(self.train_loader)

            # 検証フェーズ
            val_acc, val_report = self.evaluate()

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), "best_bert_model.pt")
                print(f"  -> Best model saved (acc={val_acc:.4f})")

        return best_val_acc

    def evaluate(self):
        """検証データで評価"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        report = classification_report(all_labels, all_preds)
        return acc, report

    def predict(self, texts):
        """新しいテキストの推論"""
        self.model.eval()
        inputs = self.tokenizer(
            texts, truncation=True, padding=True,
            max_length=self.max_length, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = probs.argmax(dim=-1)

        return preds.cpu().numpy(), probs.cpu().numpy()


# 使用例
# fine_tuner = BERTFineTuner(num_labels=3)
# fine_tuner.prepare_data(texts, labels, batch_size=16)
# best_acc = fine_tuner.train(epochs=3, lr=2e-5)
# preds, probs = fine_tuner.predict(["この映画は最高だった"])
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

### 詳細 Transformer バリアント比較表

| モデル | 年 | Attention 改良 | コンテキスト長 | 特徴 |
|---|---|---|---|---|
| **Transformer** | 2017 | Full Attention | 512 | 原論文 |
| **Transformer-XL** | 2019 | Segment Recurrence | 3,800 | メモリ機構で長文対応 |
| **Longformer** | 2020 | Local + Global | 4,096-16K | Sparse Attention |
| **BigBird** | 2020 | Random + Local + Global | 4,096 | 理論的保証付き |
| **Flash Attention** | 2022 | IO-Aware | 任意 | メモリ効率的な実装 |
| **Mamba** | 2023 | SSM (非Attention) | 非常に長い | 線形計算量 |
| **Ring Attention** | 2024 | Distributed | 数百万 | 分散環境対応 |

### 事前学習タスクの比較

| モデル | 事前学習タスク | マスキング戦略 | 方向性 |
|---|---|---|---|
| **BERT** | MLM + NSP | ランダム15%マスク | 双方向 |
| **RoBERTa** | MLM のみ | 動的マスキング | 双方向 |
| **ALBERT** | MLM + SOP | ランダム15%マスク | 双方向 |
| **ELECTRA** | RTD (置換検出) | Generator で置換 | 双方向 |
| **GPT** | CLM (次トークン予測) | Causal Mask | 左→右 |
| **T5** | Span Corruption | 連続トークンマスク | Encoder-Decoder |
| **XLNet** | PLM (順列言語モデル) | 順列組合せ | 双方向(順列) |

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

### コード例 5.5: Rotary Positional Embedding (RoPE)

```python
import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE)
    - LLaMA、GPT-NeoX 等で採用
    - 相対位置情報を回転行列で埋め込む
    - 長さの外挿（extrapolation）に強い
    """

    def __init__(self, dim, max_seq_len=4096, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # 事前計算
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, dim]
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x, seq_len=None):
        # x: [batch, heads, seq_len, dim]
        if seq_len is None:
            seq_len = x.shape[2]
        return (
            self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0),
            self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0),
        )


def rotate_half(x):
    """次元の前半と後半を入れ替えて符号反転"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Q, K に RoPE を適用"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPEMultiHeadAttention(nn.Module):
    """RoPE を使用した Multi-Head Attention"""

    def __init__(self, d_model, num_heads, max_seq_len=4096):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # RoPE 適用
        cos, sin = self.rope(Q, seq_len)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(output)
```

---

## 5. 効率的な推論と量子化

### コード例 6: モデルの量子化

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ModelOptimizer:
    """Transformer モデルの最適化ツール"""

    @staticmethod
    def quantize_dynamic(model):
        """動的量子化（CPU 推論向け）"""
        quantized = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return quantized

    @staticmethod
    def compare_model_sizes(original, quantized):
        """モデルサイズの比較"""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(original.state_dict(), f.name)
            original_size = os.path.getsize(f.name)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(quantized.state_dict(), f.name)
            quantized_size = os.path.getsize(f.name)

        print(f"元のモデル: {original_size / 1e6:.1f} MB")
        print(f"量子化後: {quantized_size / 1e6:.1f} MB")
        print(f"圧縮率: {quantized_size / original_size:.1%}")

    @staticmethod
    def benchmark_inference(model, tokenizer, texts, num_runs=100):
        """推論速度のベンチマーク"""
        import time

        model.eval()
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        # ウォームアップ
        with torch.no_grad():
            for _ in range(10):
                _ = model(**inputs)

        # 計測
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(**inputs)
        elapsed = time.time() - start

        print(f"平均推論時間: {elapsed / num_runs * 1000:.2f} ms")
        print(f"スループット: {num_runs / elapsed:.1f} 推論/秒")

    @staticmethod
    def export_onnx(model, tokenizer, output_path="model.onnx", max_length=128):
        """ONNX 形式でエクスポート"""
        model.eval()
        dummy_input = tokenizer(
            "サンプルテキスト", return_tensors="pt",
            padding="max_length", max_length=max_length, truncation=True
        )

        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=14,
        )
        print(f"ONNX モデル出力: {output_path}")


# KV Cache の実装
class KVCache:
    """
    Key-Value Cache: 自己回帰生成の高速化
    - 既に計算した K, V を再利用
    - 生成ステップごとの計算量を O(n) に削減
    """

    def __init__(self, max_batch_size=1, max_seq_len=2048,
                 num_heads=8, head_dim=64, num_layers=12):
        self.max_seq_len = max_seq_len
        self.cache = {}
        for layer in range(num_layers):
            self.cache[layer] = {
                "key": torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim),
                "value": torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim),
            }
        self.current_len = 0

    def update(self, layer, key, value):
        """新しい K, V をキャッシュに追加"""
        new_len = key.shape[2]
        self.cache[layer]["key"][:, :, self.current_len:self.current_len + new_len] = key
        self.cache[layer]["value"][:, :, self.current_len:self.current_len + new_len] = value

    def get(self, layer):
        """キャッシュから K, V を取得"""
        return (
            self.cache[layer]["key"][:, :, :self.current_len + 1],
            self.cache[layer]["value"][:, :, :self.current_len + 1],
        )

    def advance(self):
        """位置を1つ進める"""
        self.current_len += 1

    def reset(self):
        """キャッシュをリセット"""
        self.current_len = 0
        for layer in self.cache:
            self.cache[layer]["key"].zero_()
            self.cache[layer]["value"].zero_()
```

### コード例 6.5: LoRA (Low-Rank Adaptation) によるパラメータ効率的ファインチューニング

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """
    LoRA: Low-Rank Adaptation
    - 元の重み行列 W に低ランク行列 A*B を加算
    - W' = W + alpha/r * A * B
    - 学習パラメータ数を大幅に削減
    """

    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # 低ランク行列
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # 初期化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 元の重みをフリーズ
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_output + lora_output


def apply_lora_to_model(model, rank=8, alpha=16, target_modules=None):
    """
    モデルの特定の Linear 層に LoRA を適用

    Args:
        model: 対象モデル
        rank: LoRA のランク
        alpha: スケーリングファクター
        target_modules: LoRA を適用するモジュール名のリスト
    """
    if target_modules is None:
        target_modules = ["query", "key", "value", "dense"]

    total_params = 0
    lora_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_params += module.weight.numel()

            if any(target in name for target in target_modules):
                # LoRA レイヤーで置換
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                parent = model
                for part in parent_name.split("."):
                    if part:
                        parent = getattr(parent, part)

                lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
                setattr(parent, child_name, lora_layer)
                lora_params += rank * (module.in_features + module.out_features)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"元のパラメータ数: {total_params:,}")
    print(f"LoRA パラメータ数: {lora_params:,}")
    print(f"学習可能パラメータ数: {trainable:,} ({trainable/(trainable+frozen):.2%})")
    print(f"フリーズパラメータ数: {frozen:,}")

    return model

# 使用例
# from transformers import AutoModelForSequenceClassification
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
# model = apply_lora_to_model(model, rank=8, alpha=16)
```

---

## 6. 最新の系列モデル

### コード例 7: State Space Model (Mamba 風の実装)

```python
import torch
import torch.nn as nn

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (Mamba の簡略実装)
    - 入力依存のゲーティングで選択的に情報を保持
    - RNN のような逐次処理が可能（推論時は高速）
    - Transformer のような並列学習も可能
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        # 入力射影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D 畳み込み（局所依存性）
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner
        )

        # SSM パラメータ
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # A パラメータ（対角行列として初期化）
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(self.d_inner, -1))

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape

        # 入力射影 → x, z
        xz = self.in_proj(x)  # [batch, seq_len, d_inner * 2]
        x_branch, z = xz.chunk(2, dim=-1)

        # 1D 畳み込み
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = torch.silu(x_conv)

        # SSM パラメータの計算（入力依存）
        x_ssm = self.x_proj(x_conv)  # [batch, seq_len, d_state*2 + 1]
        B = x_ssm[:, :, :self.d_state]
        C = x_ssm[:, :, self.d_state:self.d_state*2]
        dt = torch.softplus(self.dt_proj(x_ssm[:, :, -1:]))

        # A の離散化
        A = -torch.exp(self.A_log)  # [d_inner, d_state]

        # SSM の逐次計算（簡略版）
        y = self._ssm_scan(x_conv, A, B, C, dt)

        # ゲーティング
        y = y * torch.silu(z)
        return self.out_proj(y)

    def _ssm_scan(self, x, A, B, C, dt):
        """SSM の逐次スキャン"""
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        h = torch.zeros(batch, d_inner, d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            # 状態更新: h = A_bar * h + B_bar * x
            dt_t = dt[:, t].unsqueeze(-1)  # [batch, d_inner, 1]
            A_bar = torch.exp(A.unsqueeze(0) * dt_t)  # [batch, d_inner, d_state]
            B_bar = dt_t * B[:, t].unsqueeze(1)  # [batch, d_inner, d_state]

            h = A_bar * h + B_bar * x[:, t].unsqueeze(-1)

            # 出力: y = C * h
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # [batch, d_inner]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # [batch, seq_len, d_inner]


class MambaBlock(nn.Module):
    """Mamba Block (SSM + Skip Connection + Norm)"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)

    def forward(self, x):
        return x + self.ssm(self.norm(x))
```

### コード例 7.5: Flash Attention の概念実装

```python
import torch
import torch.nn.functional as F
import math

def flash_attention_reference(Q, K, V, block_size=256):
    """
    Flash Attention の概念実装（参考用）
    - 実際の Flash Attention は CUDA カーネルで実装
    - ここではアルゴリズムの理解のための参考実装

    Key Idea:
    - Attention を小ブロックに分割して計算
    - SRAM（高速メモリ）に収まるサイズで処理
    - Softmax をオンラインで計算（log-sum-exp トリック）
    """
    batch, heads, seq_len, d_k = Q.shape

    # 出力とログサムエクスプの初期化
    O = torch.zeros_like(V)
    L = torch.zeros(batch, heads, seq_len, 1, device=Q.device)  # log-sum-exp
    M = torch.full((batch, heads, seq_len, 1), float('-inf'), device=Q.device)  # max

    num_blocks = math.ceil(seq_len / block_size)

    for j in range(num_blocks):
        # K, V のブロック
        j_start = j * block_size
        j_end = min((j + 1) * block_size, seq_len)
        K_block = K[:, :, j_start:j_end]
        V_block = V[:, :, j_start:j_end]

        for i in range(num_blocks):
            # Q のブロック
            i_start = i * block_size
            i_end = min((i + 1) * block_size, seq_len)
            Q_block = Q[:, :, i_start:i_end]

            # ブロック間の Attention スコア
            S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) / math.sqrt(d_k)

            # オンライン Softmax
            M_block = S_block.max(dim=-1, keepdim=True).values
            M_new = torch.max(M[:, :, i_start:i_end], M_block)

            # 指数関数の安定計算
            exp_old = torch.exp(M[:, :, i_start:i_end] - M_new)
            exp_new = torch.exp(S_block - M_new)

            L_new = exp_old * L[:, :, i_start:i_end] + exp_new.sum(dim=-1, keepdim=True)

            # 出力の更新
            O[:, :, i_start:i_end] = (
                exp_old * O[:, :, i_start:i_end] +
                torch.matmul(exp_new, V_block)
            )

            M[:, :, i_start:i_end] = M_new
            L[:, :, i_start:i_end] = L_new

    # 正規化
    O = O / L
    return O


# PyTorch 2.0+ では torch.nn.functional.scaled_dot_product_attention を使用
def efficient_attention_pytorch2(Q, K, V, is_causal=False):
    """PyTorch 2.0 の Flash Attention API"""
    # 自動的に Flash Attention または Memory-Efficient Attention を選択
    return F.scaled_dot_product_attention(
        Q, K, V,
        is_causal=is_causal,
        # dropout_p=0.0,  # 推論時は0
    )
```

---

## 7. 実践的なトラブルシューティング

### トラブルシューティングガイド

| 症状 | 原因 | 解決策 |
|---|---|---|
| Loss が NaN になる | 学習率が高すぎる / 勾配爆発 | 学習率を1/10に下げる、Gradient Clipping を追加 |
| Loss が減少しない | 学習率が低すぎる / データの問題 | 学習率を上げる、データを確認、ラベルの正しさを検証 |
| 過学習（train↓ val↑） | モデルが大きすぎる / データ不足 | Dropout増加、Weight Decay追加、データ拡張 |
| GPU メモリ不足 | バッチサイズ/系列長が大きい | バッチサイズ縮小、Gradient Accumulation、Mixed Precision |
| 学習が遅い | データローダのボトルネック | num_workers増加、pin_memory=True、前処理キャッシュ |
| Tokenizer エラー | 特殊文字/長すぎるテキスト | truncation=True、特殊文字の前処理 |
| ファインチューニングの不安定性 | 学習率が高い / Warmup 不足 | 2e-5以下の学習率、10%のWarmup Steps |

### コード例 8: Mixed Precision Training

```python
import torch
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    """Mixed Precision Training でメモリ使用量を削減し高速化"""

    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = GradScaler()

    def train_step(self, batch):
        self.optimizer.zero_grad()

        with autocast():
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else self.criterion(
                outputs, batch["labels"]
            )

        # Scaled backward
        self.scaler.scale(loss).backward()

        # Gradient clipping (unscale first)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def train_epoch(self, dataloader, accumulation_steps=4):
        """Gradient Accumulation 付きの学習エポック"""
        self.model.train()
        total_loss = 0

        for step, batch in enumerate(dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}

            with autocast():
                outputs = self.model(**batch)
                loss = outputs.loss / accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

        return total_loss / len(dataloader)
```

### コード例 8.5: Gradient Checkpointing でメモリ節約

```python
import torch
from torch.utils.checkpoint import checkpoint

class MemoryEfficientTransformer(nn.Module):
    """Gradient Checkpointing で VRAM を節約"""

    def __init__(self, d_model, num_heads, num_layers, d_ff,
                 vocab_size, max_len=512, use_checkpointing=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, 2)
        self.use_checkpointing = use_checkpointing

    def forward(self, x):
        x = self.pos_encoding(self.embedding(x))

        for layer in self.layers:
            if self.use_checkpointing and self.training:
                # Gradient Checkpointing: 中間活性値を保存せず、
                # 逆伝播時に再計算（メモリ削減、計算時間増加）
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        return self.classifier(x[:, 0])


def estimate_memory_savings(model, input_shape, dtype=torch.float32):
    """Gradient Checkpointing のメモリ削減効果を推定"""
    batch_size, seq_len = input_shape
    d_model = model.embedding.embedding_dim
    num_layers = len(model.layers)

    # 中間活性値のメモリ（概算）
    activation_per_layer = batch_size * seq_len * d_model * 4  # bytes (float32)
    total_activation = activation_per_layer * num_layers

    # Checkpointing 使用時: sqrt(num_layers) 分のメモリ
    import math
    checkpointed_activation = activation_per_layer * math.sqrt(num_layers)

    print(f"通常の中間活性値: {total_activation / 1e9:.2f} GB")
    print(f"Checkpointing 後: {checkpointed_activation / 1e9:.2f} GB")
    print(f"削減率: {1 - checkpointed_activation / total_activation:.1%}")
```

---

## 8. パフォーマンス最適化チェックリスト

### 学習の最適化

| カテゴリ | チェック項目 | 推奨設定 |
|---|---|---|
| **学習率** | Warmup + Linear Decay を使用 | BERT: 2e-5〜5e-5 |
| **バッチサイズ** | Gradient Accumulation で実効バッチを増加 | 実効32〜64 |
| **精度** | Mixed Precision (FP16/BF16) を使用 | AMP 有効化 |
| **メモリ** | Gradient Checkpointing を有効化 | 大規模モデルで必須 |
| **正則化** | Weight Decay（bias/LN除外） | 0.01〜0.1 |
| **Gradient Clipping** | max_norm を設定 | 1.0 |
| **Early Stopping** | Validation Loss で監視 | patience=3〜5 |
| **データ前処理** | Dynamic Padding / Bucketing | 系列長でグループ化 |

### 推論の最適化

| 手法 | メモリ削減 | 速度向上 | 精度影響 | 難易度 |
|---|---|---|---|---|
| **Dynamic Quantization (INT8)** | 2-4x | 1.5-3x | 微小 | 低 |
| **Static Quantization** | 2-4x | 2-4x | 小 | 中 |
| **KV Cache** | - | 2-10x | なし | 低 |
| **Flash Attention** | 2-4x | 2-4x | なし | 低 |
| **ONNX Runtime** | - | 1.5-3x | なし | 低 |
| **TensorRT** | - | 2-5x | 微小 | 中 |
| **LoRA / QLoRA** | 10-100x | - | 小〜中 | 中 |
| **Knowledge Distillation** | 3-10x | 3-10x | 中 | 高 |
| **Speculative Decoding** | - | 2-3x | なし | 高 |

---

## アンチパターン

### 1. 小規模データで大規模 Transformer を学習

**問題**: BERT-large を100件のデータでファインチューニングすると過学習する。パラメータ数がデータ数を大幅に上回る場合、汎化性能が極端に低下する。

**対策**: 小規模データには軽量モデル（DistilBERT）か、Few-shot learning（プロンプト設計）を使用する。データ拡張も検討する。

### 2. Attention の計算量を無視した設計

**問題**: Self-Attention は系列長の2乗のメモリを消費する。長文（10,000トークン以上）を処理しようとすると GPU メモリが不足する。

**対策**: Longformer（局所 + グローバル Attention）、Flash Attention（メモリ効率化）、または入力のチャンク分割を検討する。

### 3. Tokenizer の不適切な使用

**問題**: モデルに合わない Tokenizer を使用する、または前処理で特殊トークン（[CLS], [SEP]）を考慮しない。

```python
# BAD: padding/truncation なしでバッチ処理
inputs = tokenizer(texts)  # 長さが揃わない → エラー

# BAD: 異なるモデルの Tokenizer を使用
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
# → Token ID の対応がずれて無意味な結果に

# GOOD: モデルと一致する Tokenizer を使用し、適切に前処理
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
inputs = tokenizer(texts, padding=True, truncation=True,
                   max_length=512, return_tensors="pt")
```

### 4. 学習率スケジュールを使わない

**問題**: Transformer のファインチューニングで固定学習率を使用すると、学習が不安定になりやすい。

```python
# BAD: 固定学習率
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

# GOOD: Warmup + Linear Decay
from transformers import get_linear_schedule_with_warmup
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),  # 10% warmup
    num_training_steps=total_steps
)
```

### 5. Teacher Forcing を常に使用する

**問題**: Seq2Seq で学習時に常に正解を入力すると、推論時に誤りが蓄積する（exposure bias）。

```python
# BAD: 常に Teacher Forcing
teacher_forcing_ratio = 1.0  # 推論時とのギャップ大

# GOOD: Scheduled Sampling
# 学習の進行とともに Teacher Forcing 率を下げる
teacher_forcing_ratio = max(0.5, 1.0 - epoch * 0.1)
```

---

## FAQ

### Q1: RNN はもう使わないのですか？

**A**: Transformer が主流ですが、RNN にも適用場面があります。リアルタイムのストリーミング処理、メモリ制約のあるエッジデバイス、短い固定長系列の処理では LSTM/GRU が効率的です。Mamba 等の State Space Model も RNN 的な逐次処理の利点を活かしつつ Transformer に匹敵する性能を達成しています。

### Q2: BERT と GPT の使い分けは？

**A**: BERT は双方向の文脈理解に優れ、分類・情報抽出・質問応答に最適です。GPT は自己回帰（左から右）で、テキスト生成・要約・翻訳に最適です。理解系タスクなら BERT 系、生成系タスクなら GPT 系を選択してください。

### Q3: Transformer の学習にはどれくらいの GPU が必要ですか？

**A**: ファインチューニングなら BERT-base で 1 GPU（16GB VRAM）、BERT-large で 1-2 GPU が目安です。事前学習は大規模計算資源が必要で、BERT-base でも数十 GPU-day かかります。

### Q4: LoRA と Full Fine-tuning の使い分けは？

**A**: データ量とリソースで判断します。大量データ（10万件以上）+ 十分な GPU があれば Full Fine-tuning が精度最良です。少〜中量データ（数百〜数万件）では LoRA が効率的で、GPU メモリも大幅に節約できます。QLoRA（4bit 量子化 + LoRA）を使えば、7B パラメータのモデルも 1 GPU（16GB VRAM）でファインチューニング可能です。

### Q5: Flash Attention はいつ使うべきですか？

**A**: 系列長が 512 以上のタスクで、PyTorch 2.0 以上を使用している場合は常に有効化すべきです。`torch.nn.functional.scaled_dot_product_attention` で自動的に最適な実装が選択されます。特に長い系列（2048+ トークン）では、メモリ使用量が 2-4 倍削減され、速度も 2-4 倍向上します。

### Q6: Transformer の位置エンコーディングはどれを選ぶべきですか？

**A**: タスクと要件により異なります:
- **正弦波**: 固定長、シンプル、汎用性高い（原論文）
- **学習可能**: 特定タスクに最適化可能（BERT）
- **RoPE**: 長さの外挿に強い、相対位置に対応（LLaMA、GPT-NeoX）
- **ALiBi**: 位置バイアスを Attention に直接加算、外挿に強い（BLOOM）
最新の LLM では RoPE が主流です。

### Q7: Attention の重みを可視化する意味は？

**A**: モデルの解釈可能性の向上に役立ちます。具体的には: (1) モデルが入力のどの部分に注目しているかを確認、(2) デバッグ（期待通りのパターンを学習しているか）、(3) ドメイン専門家へのモデル説明。ただし、Attention の重みは必ずしもモデルの「理由」を反映しているわけではない（Jain & Wallace, 2019）点に注意が必要です。

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
| LoRA | パラメータ効率的ファインチューニング。少ないリソースで大規模モデルを適応 |
| Flash Attention | メモリ効率的な Attention 実装。長系列の処理に必須 |
| SSM (Mamba) | 線形計算量の系列モデル。RNN と Transformer の長所を統合 |

## 次に読むべきガイド

- [コンピュータビジョン](../03-applied/01-computer-vision.md) — Vision Transformer の応用
- [MLOps](../03-applied/02-mlops.md) — Transformer モデルのデプロイと運用

## 参考文献

1. **Vaswani et al.**: [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — Transformer の原論文
2. **Devlin et al.**: [BERT: Pre-training of Deep Bidirectional Transformers (2018)](https://arxiv.org/abs/1810.04805) — BERT の原論文
3. **Jay Alammar**: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Transformer の視覚的解説
4. **Hu et al.**: [LoRA: Low-Rank Adaptation of Large Language Models (2021)](https://arxiv.org/abs/2106.09685) — LoRA の原論文
5. **Dao et al.**: [FlashAttention: Fast and Memory-Efficient Exact Attention (2022)](https://arxiv.org/abs/2205.14135) — Flash Attention の原論文
6. **Gu & Dao**: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (2023)](https://arxiv.org/abs/2312.00752) — Mamba の原論文

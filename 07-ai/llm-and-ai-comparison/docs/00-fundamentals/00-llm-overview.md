# LLM概要 — 大規模言語モデルの基礎

> Transformer アーキテクチャからスケーリング則、学習手法、推論最適化まで、LLM の全体像を体系的に理解する。

## この章で学ぶこと

1. **Transformer アーキテクチャ**の仕組みと Self-Attention の原理
2. **スケーリング則**がモデル性能に与える影響とパラメータ数の意味
3. **事前学習・事後学習**の各段階と代表的な学習手法
4. **位置エンコーディング**の種類と長文脈対応の進化
5. **Mixture of Experts (MoE)** アーキテクチャの仕組みと利点
6. **推論最適化**の技術とデプロイ時の考慮事項

---

## 1. Transformer アーキテクチャ

### 1.1 Self-Attention の原理

Self-Attention は、入力シーケンス内の各トークンが他の全トークンとの関連度を計算し、文脈に応じた表現を生成するメカニズムです。従来の RNN/LSTM では系列を逐次処理する必要があったのに対し、Self-Attention は全トークン間の関係を一度に並列計算できるため、学習効率が飛躍的に向上しました。

### コード例 1: Self-Attention の計算（NumPy）

```python
import numpy as np

def self_attention(Q, K, V, mask=None):
    """スケーリングドット積アテンション

    Args:
        Q: クエリ行列 (seq_len, d_k)
        K: キー行列 (seq_len, d_k)
        V: バリュー行列 (seq_len, d_v)
        mask: オプションのマスク行列（因果マスクなど）

    Returns:
        出力行列とアテンション重み
    """
    d_k = Q.shape[-1]

    # スケーリングドット積
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)

    # マスクの適用（因果的言語モデルの場合）
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Softmax で正規化
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # 加重和
    output = np.matmul(weights, V)
    return output, weights

# 例: 4トークン、次元8
np.random.seed(42)
seq_len = 4
d_k = 8

Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

# 因果マスク（下三角行列）- デコーダ用
causal_mask = np.tril(np.ones((seq_len, seq_len)))
print("因果マスク:")
print(causal_mask)

output, weights = self_attention(Q, K, V, mask=causal_mask)
print(f"\n出力形状: {output.shape}")  # (4, 8)
print(f"アテンション重み形状: {weights.shape}")  # (4, 4)
print(f"\nアテンション重み（各行の合計=1）:")
print(weights.round(3))
```

### コード例 2: Multi-Head Attention の実装

```python
import numpy as np

class MultiHeadAttention:
    """Multi-Head Attention の NumPy 実装"""

    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 重み行列の初期化（Xavier初期化）
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

    def split_heads(self, x):
        """(seq_len, d_model) → (num_heads, seq_len, d_k)"""
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 0, 2)

    def forward(self, x, mask=None):
        """Multi-Head Attention の前方計算"""
        # 線形変換
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # ヘッド分割
        Q = self.split_heads(Q)  # (num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 各ヘッドでアテンション計算
        d_k = self.d_k
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        attended = np.matmul(weights, V)  # (num_heads, seq_len, d_k)

        # ヘッド結合
        seq_len = attended.shape[1]
        concat = attended.transpose(1, 0, 2).reshape(seq_len, self.d_model)

        # 出力射影
        output = concat @ self.W_o
        return output, weights

# 使用例
np.random.seed(42)
mha = MultiHeadAttention(d_model=64, num_heads=8)
x = np.random.randn(10, 64)  # 10トークン、次元64
output, weights = mha.forward(x)
print(f"入力形状: {x.shape}")       # (10, 64)
print(f"出力形状: {output.shape}")   # (10, 64)
print(f"重み形状: {weights.shape}")  # (8, 10, 10) - 8ヘッド
```

### ASCII 図解 1: Transformer ブロック構造

```
┌───────────────────────────────────┐
│         出力 (次の層へ)            │
├───────────────────────────────────┤
│      Layer Norm + 残差接続         │
│      output = LN(x + FFN(x))     │
├───────────────────────────────────┤
│     Feed-Forward Network (FFN)    │
│     ┌─────┐  ┌──────┐  ┌─────┐  │
│     │Lin. │→│SwiGLU│→│Lin. │  │
│     │d→4d │  │      │  │4d→d │  │
│     └─────┘  └──────┘  └─────┘  │
├───────────────────────────────────┤
│      Layer Norm + 残差接続         │
│      x = LN(input + MHA(input))  │
├───────────────────────────────────┤
│   Multi-Head Self-Attention       │
│  ┌──────┐ ┌──────┐ ┌──────┐     │
│  │Head 1│ │Head 2│ │Head N│     │
│  │Q K V │ │Q K V │ │Q K V │     │
│  └──────┘ └──────┘ └──────┘     │
│        ↓ Concat + Linear         │
├───────────────────────────────────┤
│   入力エンベディング + 位置符号化    │
│   x = Embed(token) + PosEnc(pos) │
└───────────────────────────────────┘
```

### 1.2 エンコーダ vs デコーダ

```
┌─────────────────────────────────────────────────────────────────┐
│                   Transformer アーキテクチャの分類                │
├─────────────┬──────────────────┬────────────────────────────────┤
│ タイプ       │ 代表モデル        │ 特徴                          │
├─────────────┼──────────────────┼────────────────────────────────┤
│ エンコーダ   │ BERT, RoBERTa   │ 双方向アテンション              │
│ のみ        │ DeBERTa          │ 分類・NER・類似度計算に最適     │
│             │                  │ 入力全体の文脈を同時に把握      │
├─────────────┼──────────────────┼────────────────────────────────┤
│ デコーダ     │ GPT, LLaMA      │ 因果的（左→右）アテンション     │
│ のみ        │ Claude, Gemini   │ テキスト生成・対話に最適         │
│             │                  │ 自己回帰的にトークンを生成      │
├─────────────┼──────────────────┼────────────────────────────────┤
│ エンコーダ   │ T5, BART        │ 入力をエンコード→出力をデコード  │
│ デコーダ     │ mBART            │ 翻訳・要約に最適               │
│             │                  │ Cross-Attention で入力を参照   │
└─────────────┴──────────────────┴────────────────────────────────┘

現代の LLM の主流:
  → デコーダのみ（Decoder-only）アーキテクチャ
  → 統一的に多様なタスクを処理可能
  → スケーリングの効率が最も良い
```

### コード例 3: PyTorch で Transformer レイヤー

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    """LLM スタイルの Transformer ブロック（Pre-LN 構成）"""

    def __init__(self, d_model=512, nhead=8, dim_ff=2048, dropout=0.1):
        super().__init__()

        # Pre-LayerNorm 構成（GPT-2 以降の標準）
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),              # ReLU → GELU が現代の標準
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        # Pre-LN + 残差接続（Self-Attention）
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask)
        x = x + attn_out

        # Pre-LN + 残差接続（FFN）
        normed = self.ln2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x

# 使用例
block = TransformerBlock(d_model=512, nhead=8, dim_ff=2048)
src = torch.randn(2, 10, 512)  # (バッチ=2, シーケンス長=10, 次元=512)
output = block(src)
print(f"出力形状: {output.shape}")  # torch.Size([2, 10, 512])

# パラメータ数の確認
total_params = sum(p.numel() for p in block.parameters())
print(f"パラメータ数: {total_params:,}")  # 約4.2M
```

### コード例 4: 因果マスクの生成

```python
import torch

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """因果的アテンションマスクの生成

    デコーダモデルでは、各位置は自分自身と
    それ以前の位置のみ参照可能。
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

# 8トークンのマスク
mask = create_causal_mask(8)
print("因果マスク（0=参照可能、-inf=マスク）:")
print(mask)

# Sliding Window Attention のマスク（Mistral 方式）
def create_sliding_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """Sliding Window Attention マスク

    各位置は window_size 以内のトークンのみ参照可能。
    長いシーケンスでもメモリ使用量を O(n*w) に抑制。
    """
    mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        # 因果マスク（未来は見えない）+ ウィンドウ制限
        start = max(0, i - window_size + 1)
        mask[i, :start] = float('-inf')
        mask[i, i+1:] = float('-inf')
    return mask

sw_mask = create_sliding_window_mask(8, window_size=3)
print("\nSliding Window マスク (window=3):")
print(sw_mask)
```

---

## 2. 位置エンコーディング

### 2.1 位置情報の必要性

Self-Attention は並列計算可能だが、トークンの順序情報を持たないため、位置エンコーディングが不可欠です。

### ASCII 図解 2: 位置エンコーディングの進化

```
┌──────────────────────────────────────────────────────────────────┐
│                位置エンコーディングの進化                          │
├────────────────┬─────────────────────────────────────────────────┤
│ 絶対位置       │ Sinusoidal (元論文) → 学習可能な埋め込み (GPT)  │
│                │ PE(pos, 2i) = sin(pos / 10000^(2i/d))          │
│                │ 固定長の制約あり                                │
├────────────────┼─────────────────────────────────────────────────┤
│ 相対位置       │ ALiBi (BLOOM) → 線形バイアスを直接加算           │
│                │ 学習不要、任意の長さに外挿可能                    │
├────────────────┼─────────────────────────────────────────────────┤
│ 回転位置符号化  │ RoPE (LLaMA, Qwen) → 回転行列で位置を符号化     │
│                │ 相対位置の内積を保存                             │
│                │ NTK-aware scaling で長文脈に対応                 │
├────────────────┼─────────────────────────────────────────────────┤
│ 最新手法       │ YaRN → RoPE の改良版、より効率的な外挿           │
│                │ LongRoPE → 数百万トークンまで対応                │
└────────────────┴─────────────────────────────────────────────────┘
```

### コード例 5: RoPE（Rotary Position Embedding）の実装

```python
import numpy as np

def compute_rope_embeddings(seq_len: int, d_model: int, base: float = 10000.0):
    """RoPE (Rotary Position Embedding) の計算

    各次元ペアに対して異なる周波数の回転を適用。
    位置 p のトークンの次元 (2i, 2i+1) に対して:
        cos(p * theta_i), sin(p * theta_i)
    を掛け合わせる。
    """
    # 各次元ペアの周波数 theta
    dim_pairs = d_model // 2
    theta = 1.0 / (base ** (np.arange(0, d_model, 2) / d_model))

    # 位置 × 周波数
    positions = np.arange(seq_len)
    angles = np.outer(positions, theta)  # (seq_len, dim_pairs)

    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)

    return cos_vals, sin_vals

def apply_rope(x, cos_vals, sin_vals):
    """RoPE をクエリ/キーベクトルに適用"""
    d = x.shape[-1]
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]

    # 回転の適用
    rotated = np.concatenate([
        x1 * cos_vals - x2 * sin_vals,
        x1 * sin_vals + x2 * cos_vals
    ], axis=-1)

    return rotated

# 使用例
seq_len = 16
d_model = 64
cos_vals, sin_vals = compute_rope_embeddings(seq_len, d_model)

# クエリとキーに RoPE を適用
Q = np.random.randn(seq_len, d_model)
K = np.random.randn(seq_len, d_model)

Q_rotated = apply_rope(Q, cos_vals, sin_vals)
K_rotated = apply_rope(K, cos_vals, sin_vals)

print(f"元のクエリ形状: {Q.shape}")
print(f"回転後のクエリ形状: {Q_rotated.shape}")

# 相対位置の内積が保持されることを確認
# 位置 i と j の内積は |i - j| のみに依存
dot_original = np.sum(Q[3] * K[5])
dot_rotated = np.sum(Q_rotated[3] * K_rotated[5])
print(f"\n元の内積 (pos 3, 5): {dot_original:.4f}")
print(f"RoPE後の内積 (pos 3, 5): {dot_rotated:.4f}")
```

---

## 3. スケーリング則

### ASCII 図解 3: スケーリング則の3要素

```
パフォーマンス (Loss)
│
│  ╲
│   ╲  ← パラメータ数 N を増加
│    ╲
│     ╲───────────
│
│  ╲
│   ╲  ← データ量 D を増加
│    ╲
│     ╲───────────
│
│  ╲
│   ╲  ← 計算量 C を増加
│    ╲
│     ╲───────────
└──────────────────→ 規模（対数スケール）

Chinchilla Scaling Law:
L(N, D) ≈ E + A/N^α + B/D^β
- N: パラメータ数
- D: 学習トークン数
- α ≈ 0.34, β ≈ 0.28
- E: 不可約損失（データのエントロピーに起因）

最適な計算配分:
  N_opt ∝ C^0.5   (パラメータ数は計算量の平方根に比例)
  D_opt ∝ C^0.5   (データ量も計算量の平方根に比例)
  → パラメータ数とデータ量を同じ比率で増やすのが最適
```

### コード例 6: スケーリング則の概算と可視化

```python
import numpy as np

def estimate_loss(params_billions, tokens_trillions):
    """Chinchilla 式での Loss 推定（簡易版）"""
    E = 1.69     # 不可約損失
    A = 406.4
    B = 410.7
    alpha = 0.34
    beta = 0.28

    N = params_billions * 1e9
    D = tokens_trillions * 1e12

    loss = E + A / (N ** alpha) + B / (D ** beta)
    return loss

def compute_optimal_allocation(compute_budget_flops):
    """与えられた計算予算での最適なパラメータ数・データ量を推定

    Chinchilla の知見: N ≈ D / 20 (トークン/パラメータ比 ≈ 20)
    FLOPs ≈ 6 * N * D
    """
    # C = 6 * N * D, D = 20 * N
    # C = 6 * N * 20 * N = 120 * N^2
    N_opt = np.sqrt(compute_budget_flops / 120)
    D_opt = 20 * N_opt
    return N_opt, D_opt

# 各モデルサイズでの推定 Loss
print("=" * 60)
print(f"{'Params':>8} {'Tokens':>8} {'Loss':>8} {'最適比':>10}")
print("=" * 60)
for params in [7, 13, 70, 405]:
    for tokens in [1, 2, 5, 15]:
        loss = estimate_loss(params, tokens)
        ratio = tokens * 1e12 / (params * 1e9)
        optimal = "***" if 15 <= ratio <= 25 else ""
        print(f"{params:>5}B  {tokens:>5}T  {loss:>8.3f}  {ratio:>6.0f}:1 {optimal}")

# 計算予算に対する最適配分
print("\n" + "=" * 60)
print("計算予算に対する最適配分（Chinchilla則）")
print("=" * 60)
budgets_names = [
    ("1e21 (小規模実験)", 1e21),
    ("1e23 (7B級)", 1e23),
    ("1e24 (70B級)", 1e24),
    ("1e25 (GPT-4級)", 1e25),
]
for name, budget in budgets_names:
    N_opt, D_opt = compute_optimal_allocation(budget)
    print(f"\n計算予算: {name}")
    print(f"  最適パラメータ数: {N_opt/1e9:.1f}B")
    print(f"  最適トークン数: {D_opt/1e12:.2f}T")
    print(f"  推定Loss: {estimate_loss(N_opt/1e9, D_opt/1e12):.3f}")
```

### 3.1 スケーリング則の実践的含意

```
┌──────────────────────────────────────────────────────────────────┐
│                 スケーリング則の実務的な意味                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│ 1. 予測可能性                                                    │
│    小規模実験 → 大規模モデルの性能を高精度に予測可能               │
│    → 事前に計算予算の配分を最適化できる                            │
│                                                                  │
│ 2. 投資判断                                                     │
│    10倍の計算量 → 約 X% の Loss 改善を定量的に見積もれる          │
│    → ROI を事前に評価可能                                        │
│                                                                  │
│ 3. Chinchilla vs GPT-3 の教訓                                   │
│    GPT-3: 175B params, 300B tokens (1.7:1)                      │
│    Chinchilla 最適: 175B params → 3.5T tokens (20:1)            │
│    → GPT-3 は「パラメータ過多・データ不足」だった                  │
│                                                                  │
│ 4. 推論効率の考慮（Llama 3 の戦略）                              │
│    Chinchilla 最適よりデータを多く使い、小さいモデルで高性能        │
│    8B + 15T tokens → 推論コスト削減 + 性能維持                   │
│    → 「推論時計算コスト」まで含めた全体最適化                      │
│                                                                  │
│ 5. Emergent Abilities（創発的能力）                              │
│    特定のスケール閾値を超えると突然新しい能力が出現                 │
│    → Chain-of-Thought 推論、コード生成、多言語能力など             │
│    → スケーリング則では予測できない質的変化                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 比較表 1: 代表モデルのパラメータ数とデータ量

| モデル | パラメータ数 | 学習トークン数 | コンテキスト長 | アーキテクチャ | 公開年 |
|--------|-------------|---------------|---------------|---------------|--------|
| GPT-3 | 175B | 300B | 2,048 | Dense Decoder | 2020 |
| Chinchilla | 70B | 1.4T | 2,048 | Dense Decoder | 2022 |
| LLaMA 2 | 7-70B | 2T | 4,096 | Dense Decoder | 2023 |
| GPT-4 | 非公開 (推定1.8T MoE) | 非公開 | 128K | MoE Decoder | 2023 |
| Mixtral 8x7B | 46.7B (活性12.9B) | 非公開 | 32K | MoE Decoder | 2023 |
| Claude 3.5 Sonnet | 非公開 | 非公開 | 200K | 非公開 | 2024 |
| LLaMA 3.1 | 8-405B | 15T+ | 128K | Dense Decoder | 2024 |
| Gemini 1.5 Pro | 非公開 | 非公開 | 1M+ | MoE Decoder | 2024 |
| DeepSeek-V3 | 671B (活性37B) | 14.8T | 128K | MoE Decoder | 2024 |
| Qwen 2.5 | 0.5-72B | 18T+ | 128K | Dense Decoder | 2024 |

---

## 4. Mixture of Experts (MoE) アーキテクチャ

### ASCII 図解 4: MoE の構造

```
┌─────────────────────────────────────────────────────────────┐
│                  Mixture of Experts (MoE)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  入力トークン x                                              │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────┐                                              │
│  │ ゲート    │ → softmax → Top-K 選択                       │
│  │ Network  │                                              │
│  └──────────┘                                              │
│       │                                                     │
│       ▼                                                     │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ... ┌─────┐            │
│  │ E_1 │ │ E_2 │ │ E_3 │ │ E_4 │     │ E_N │            │
│  │ FFN │ │ FFN │ │ FFN │ │ FFN │     │ FFN │            │
│  └─────┘ └─────┘ └─────┘ └─────┘     └─────┘            │
│    ✓       ✓                              (不活性)         │
│    │       │                                               │
│    ▼       ▼                                               │
│  加重合算: output = Σ g_i * Expert_i(x)                    │
│                                                             │
│  例: Mixtral 8x7B                                          │
│    - 8つのエキスパート、各トークンで上位2つを選択              │
│    - 総パラメータ: 46.7B、活性パラメータ: 12.9B              │
│    - Dense 13B モデルと同等の推論コストで 70B 級の性能         │
│                                                             │
│  例: DeepSeek-V3                                           │
│    - 256 エキスパート + 1 共有エキスパート                    │
│    - 各トークンで上位8つを選択                               │
│    - 総パラメータ: 671B、活性パラメータ: 37B                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### コード例 7: 簡易 MoE レイヤー

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    """Mixture of Experts レイヤー（簡易実装）"""

    def __init__(
        self,
        d_model: int = 512,
        num_experts: int = 8,
        top_k: int = 2,
        dim_ff: int = 2048,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # ゲートネットワーク
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # エキスパート（各エキスパートは FFN）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_ff),
                nn.GELU(),
                nn.Linear(dim_ff, d_model),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (B*S, d_model)

        # ゲート計算
        gate_logits = self.gate(x_flat)  # (B*S, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Top-K エキスパート選択
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # 各エキスパートの出力を加重合算
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]
            expert_weights = top_k_probs[:, k].unsqueeze(-1)

            for i in range(self.num_experts):
                mask = (expert_indices == i)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[i](expert_input)
                    output[mask] += expert_weights[mask] * expert_output

        return output.view(batch_size, seq_len, d_model)

# 使用例
moe = MoELayer(d_model=512, num_experts=8, top_k=2)
x = torch.randn(2, 10, 512)
output = moe(x)
print(f"入力: {x.shape} → 出力: {output.shape}")

total_params = sum(p.numel() for p in moe.parameters())
print(f"総パラメータ数: {total_params:,}")
# 8エキスパートの FFN + ゲート
```

### 比較表 2: Dense vs MoE の比較

| 項目 | Dense モデル | MoE モデル |
|------|-------------|-----------|
| パラメータ効率 | 全パラメータが常に活性 | 一部のみ活性（推論コスト削減） |
| 学習効率 | 安定 | ロードバランスが課題 |
| メモリ使用量 | パラメータ数に比例 | 全エキスパートをメモリに保持 |
| 推論速度 | パラメータ数に比例 | 活性パラメータ数に比例 |
| 性能/FLOP | 基準 | 同じFLOPでより高性能 |
| デプロイ難度 | 低い | 高い（エキスパート分散が必要） |
| 代表例 | LLaMA 3.1 405B | Mixtral 8x7B, DeepSeek-V3 |

---

## 5. 学習手法

### ASCII 図解 5: LLM の学習3段階

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: 事前学習 (Pre-training)                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 目的: 言語の統計的パターンを学習                              ││
│  │ データ: Web テキスト、書籍、コード（数兆トークン）              ││
│  │ 手法: 次トークン予測（自己教師あり学習）                       ││
│  │ 損失: L = -Σ log P(x_t | x_<t)                             ││
│  │ コスト: 数千〜数万 GPU × 数ヶ月 = 数百万〜数千万ドル          ││
│  │ 計算量: 6 * N * D FLOPs（N=パラメータ数、D=トークン数）       ││
│  └─────────────────────────────────────────────────────────────┘│
│                           ↓                                     │
│  Stage 2: 教師あり微調整 (SFT: Supervised Fine-Tuning)          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 目的: 指示に従って応答する能力を獲得                          ││
│  │ データ: 高品質な指示-応答ペア（数万〜数十万例）                ││
│  │ 手法: 指示部分はマスク、応答部分のみ Loss 計算                 ││
│  │ コスト: 数十 GPU × 数日 = 数千〜数万ドル                     ││
│  │ 重要: データ品質が量より重要（LIMA の知見）                    ││
│  └─────────────────────────────────────────────────────────────┘│
│                           ↓                                     │
│  Stage 3: アラインメント (RLHF / DPO / KTO)                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 目的: 人間の価値観・好みに合わせる                            ││
│  │                                                             ││
│  │ RLHF (Reinforcement Learning from Human Feedback):          ││
│  │   1. 報酬モデルを学習（人間の比較判断から）                    ││
│  │   2. PPO で方策を最適化（報酬最大化 + KL制約）                ││
│  │   → 複雑だが高品質な結果                                     ││
│  │                                                             ││
│  │ DPO (Direct Preference Optimization):                       ││
│  │   報酬モデル不要、直接比較データから学習                       ││
│  │   → シンプルで実装が容易                                     ││
│  │                                                             ││
│  │ KTO (Kahneman-Tversky Optimization):                       ││
│  │   ペア比較不要、各応答の good/bad ラベルのみで学習             ││
│  │   → データ収集コストが最も低い                                ││
│  │                                                             ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### コード例 8: Hugging Face で事前学習済みモデルのロードと推論

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# モデルのロード（量子化オプション付き）
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 量子化なし（フル精度）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # BF16 で半分のメモリ使用
    device_map="auto",            # 自動デバイス配置
    attn_implementation="sdpa",   # PyTorch の最適化アテンション
)

# チャットテンプレートを使用（推奨）
messages = [
    {"role": "system", "content": "あなたは親切なAIアシスタントです。"},
    {"role": "user", "content": "大規模言語モデルの仕組みを簡潔に説明してください。"}
]

# テンプレート適用
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# 生成パラメータの設定
outputs = model.generate(
    input_ids,
    max_new_tokens=200,
    temperature=0.7,        # 多様性の制御
    top_p=0.9,              # Nucleus sampling
    repetition_penalty=1.1, # 繰り返し抑制
    do_sample=True,
)

# 入力部分を除いて出力
response = tokenizer.decode(
    outputs[0][input_ids.shape[-1]:],
    skip_special_tokens=True
)
print(response)
```

### コード例 9: 4bit 量子化によるメモリ効率化

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 4bit 量子化設定
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4 量子化
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # 二重量子化でさらに圧縮
)

model_name = "meta-llama/Llama-3.1-70B-Instruct"

# 70B モデルを 4bit で読み込み（約35GB → 約17.5GB）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# メモリ使用量の確認
total_memory = sum(
    p.nelement() * p.element_size() for p in model.parameters()
)
print(f"モデルメモリ使用量: {total_memory / 1e9:.2f} GB")

# 量子化の品質比較
# FP16:  70B × 2 bytes = 140 GB
# INT8:  70B × 1 byte  = 70 GB
# INT4:  70B × 0.5 byte = 35 GB
# NF4 + Double Quant ≈ 17.5 GB
```

### コード例 10: 学習コストの詳細な概算

```python
def estimate_training_cost(
    params_billions: float,
    tokens_trillions: float,
    gpu_type: str = "H100",
    mfu: float = 0.4,  # Model FLOPs Utilization
):
    """学習コストの詳細な概算

    Args:
        params_billions: パラメータ数（十億単位）
        tokens_trillions: 学習トークン数（兆単位）
        gpu_type: GPU の種類
        mfu: Model FLOPs Utilization（通常 30-50%）
    """
    gpu_specs = {
        "A100_40GB": {"bf16_flops": 312e12, "cost_per_hour": 1.5, "memory_gb": 40},
        "A100_80GB": {"bf16_flops": 312e12, "cost_per_hour": 2.0, "memory_gb": 80},
        "H100_SXM": {"bf16_flops": 989e12, "cost_per_hour": 3.5, "memory_gb": 80},
        "H200":     {"bf16_flops": 989e12, "cost_per_hour": 4.5, "memory_gb": 141},
    }
    spec = gpu_specs.get(gpu_type, gpu_specs["H100_SXM"])

    # 理論的 FLOPs: 6 * N * D（前方 + 後方）
    total_flops = 6 * params_billions * 1e9 * tokens_trillions * 1e12

    # MFU を考慮した実効スループット
    effective_flops_per_gpu = spec["bf16_flops"] * mfu

    # 必要 GPU 時間
    gpu_seconds = total_flops / effective_flops_per_gpu
    gpu_hours = gpu_seconds / 3600

    # コスト計算
    cost = gpu_hours * spec["cost_per_hour"]

    # 電力消費概算（H100: ~700W）
    power_kwh = gpu_hours * 0.7  # kWh
    co2_tons = power_kwh * 0.4 / 1000  # CO2 トン（米国平均）

    print(f"{'='*60}")
    print(f"学習コスト概算: {params_billions}B params × {tokens_trillions}T tokens")
    print(f"{'='*60}")
    print(f"GPU: {gpu_type} (MFU: {mfu*100:.0f}%)")
    print(f"理論 FLOPs: {total_flops:.2e}")
    print(f"必要 GPU 時間: {gpu_hours:,.0f} 時間")
    print(f"推定コスト: ${cost:,.0f}")
    print(f"電力消費: {power_kwh:,.0f} kWh")
    print(f"CO2排出量: {co2_tons:,.0f} トン")
    print(f"\nGPU台数別の学習日数:")
    for num_gpus in [64, 256, 1024, 4096, 16384]:
        days = gpu_hours / num_gpus / 24
        if days >= 1:
            print(f"  {num_gpus:>6}台: {days:>8.1f} 日 (${cost/1e6:.1f}M)")
    print()

# 各モデル規模の概算
estimate_training_cost(8, 15, "H100_SXM", mfu=0.4)    # LLaMA 3.1 8B
estimate_training_cost(70, 15, "H100_SXM", mfu=0.35)   # LLaMA 3.1 70B
estimate_training_cost(405, 15, "H100_SXM", mfu=0.3)   # LLaMA 3.1 405B
```

### 比較表 3: 学習手法の詳細比較

| 手法 | 目的 | データ | 計算コスト | 必要な専門知識 | 主なフレームワーク |
|------|------|--------|-----------|--------------|------------------|
| 事前学習 (Pre-training) | 言語理解の獲得 | 数兆トークンのテキスト | 非常に高い (数百万ドル) | 非常に高い | Megatron-LM, DeepSpeed |
| SFT (教師あり微調整) | 指示追従能力 | 数万〜数十万の指示-応答ペア | 中程度 | 高い | Hugging Face TRL |
| RLHF | 人間の好みに合わせる | 比較ペア + 報酬モデル | 高い | 非常に高い | TRL + PPO |
| DPO | RLHF の簡略化 | 比較ペアのみ | 中程度 | 中程度 | TRL + DPOTrainer |
| KTO | ペア不要のアラインメント | 各応答のgood/badラベル | 中程度 | 中程度 | TRL |
| LoRA | 効率的な微調整 | タスク固有データ | 低い | 中程度 | PEFT |
| QLoRA | 量子化+LoRA | タスク固有データ | 非常に低い | 中程度 | PEFT + bitsandbytes |

---

## 6. 推論の最適化技術

### ASCII 図解 6: 推論最適化技術の全体像

```
┌────────────────────────────────────────────────────────────────┐
│                   推論最適化の階層                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ハードウェアレベル                                              │
│  ├── GPU/TPU/NPU の選択                                       │
│  ├── テンソルコア活用（BF16/FP8 演算）                          │
│  └── マルチGPU 推論（テンソル並列/パイプライン並列）              │
│                                                                │
│  モデルレベル                                                   │
│  ├── 量子化（INT8, INT4, GPTQ, AWQ, GGUF）                   │
│  ├── 蒸留（大モデル → 小モデルへの知識転移）                     │
│  ├── プルーニング（不要なパラメータの除去）                       │
│  └── アーキテクチャ改良（GQA, MQA, SWA）                       │
│                                                                │
│  ランタイムレベル                                               │
│  ├── KV Cache（再計算の回避）                                   │
│  ├── Continuous Batching（動的バッチ処理）                      │
│  ├── PagedAttention (vLLM)                                    │
│  ├── Flash Attention（メモリ効率的なアテンション）                │
│  ├── Speculative Decoding（投機的デコーディング）                │
│  └── Prefix Caching（共通プレフィックスの再利用）                │
│                                                                │
│  アプリケーションレベル                                          │
│  ├── プロンプト最適化（トークン数削減）                           │
│  ├── モデルルーティング（タスク難易度で振り分け）                  │
│  ├── キャッシュ戦略（類似リクエストの結果再利用）                  │
│  └── ストリーミング出力（体感レイテンシの改善）                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### コード例 11: KV Cache の概念

```python
import torch
import torch.nn as nn

class CausalSelfAttentionWithKVCache(nn.Module):
    """KV Cache 付き Self-Attention の概念実装"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        """
        初回（プリフィル）: 全トークンを処理、KV Cache を構築
        2回目以降（デコード）: 新トークンのみ処理、KV Cache を更新
        """
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)

        if kv_cache is not None:
            # デコードフェーズ: 過去の KV を結合
            past_K, past_V = kv_cache
            K = torch.cat([past_K, K], dim=1)
            V = torch.cat([past_V, V], dim=1)

        # 新しい KV Cache
        new_kv_cache = (K, V)

        # アテンション計算（Q は現在のトークンのみ）
        Q = Q.transpose(1, 2)  # (B, H, S_q, D)
        K = K.transpose(1, 2)  # (B, H, S_kv, D)
        V = V.transpose(1, 2)  # (B, H, S_kv, D)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)

        output = output.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        output = self.W_o(output)

        return output, new_kv_cache

# KV Cache の効果を概算
def kv_cache_memory_estimate(
    num_layers: int,
    d_model: int,
    max_seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,  # BF16
):
    """KV Cache のメモリ使用量を概算"""
    # 各層で K, V の2つのテンソル
    # 形状: (batch_size, seq_len, d_model)
    memory_per_layer = 2 * batch_size * max_seq_len * d_model * dtype_bytes
    total_memory = memory_per_layer * num_layers

    print(f"KV Cache メモリ推定:")
    print(f"  層数: {num_layers}, 次元: {d_model}, 最大長: {max_seq_len}")
    print(f"  バッチサイズ: {batch_size}")
    print(f"  合計: {total_memory / 1e9:.2f} GB")
    return total_memory

# 各モデルの KV Cache サイズ
kv_cache_memory_estimate(32, 4096, 8192, batch_size=1)    # 8B モデル
kv_cache_memory_estimate(80, 8192, 8192, batch_size=1)    # 70B モデル
kv_cache_memory_estimate(126, 16384, 8192, batch_size=1)  # 405B モデル
```

### コード例 12: vLLM を使った高速推論

```python
from vllm import LLM, SamplingParams

# vLLM でモデルロード（PagedAttention 自動適用）
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.9,  # GPU メモリの90%を使用
    tensor_parallel_size=1,       # GPU 数
    enable_prefix_caching=True,   # 共通プレフィックスのキャッシュ
)

# サンプリングパラメータ
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    repetition_penalty=1.1,
)

# バッチ推論（Continuous Batching が自動適用）
prompts = [
    "機械学習と深層学習の違いを説明してください。",
    "Pythonでクイックソートを実装してください。",
    "日本の経済政策の課題を3つ挙げてください。",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"プロンプト: {prompt[:50]}...")
    print(f"出力: {generated_text[:100]}...")
    print(f"トークン/秒: {len(output.outputs[0].token_ids) / output.metrics.finished_time:.1f}")
    print()
```

### 比較表 4: 推論エンジンの比較

| エンジン | 主な特徴 | 最適ユースケース | サポートモデル |
|----------|---------|----------------|---------------|
| vLLM | PagedAttention, Continuous Batching | 高スループットサーバー | 幅広い |
| TensorRT-LLM | NVIDIA最適化, FP8対応 | NVIDIA GPU最大活用 | 主要モデル |
| llama.cpp | CPU/Metal 推論, GGUF量子化 | ローカル推論 | GGUF対応モデル |
| Ollama | 簡単セットアップ, ローカル実行 | 開発・プロトタイプ | GGUF対応モデル |
| SGLang | RadixAttention, 構造化生成 | 複雑なパイプライン | 主要モデル |
| MLC-LLM | クロスプラットフォーム | モバイル/エッジ | 主要モデル |

---

## 7. GQA/MQA: アテンションの効率化

### ASCII 図解 7: MHA vs GQA vs MQA

```
Multi-Head Attention (MHA)        Grouped-Query Attention (GQA)
┌───┐ ┌───┐ ┌───┐ ┌───┐         ┌───┐ ┌───┐ ┌───┐ ┌───┐
│Q_1│ │Q_2│ │Q_3│ │Q_4│         │Q_1│ │Q_2│ │Q_3│ │Q_4│
└─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘         └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘
  │     │     │     │               ╲   ╱       ╲   ╱
┌─┴─┐ ┌─┴─┐ ┌─┴─┐ ┌─┴─┐         ┌──┴───┘     ┌──┴───┘
│K_1│ │K_2│ │K_3│ │K_4│         │K_1          │K_2
│V_1│ │V_2│ │V_3│ │V_4│         │V_1          │V_2
└───┘ └───┘ └───┘ └───┘         └─────┘       └─────┘
 KVヘッド数 = Qヘッド数            KVヘッド数 < Qヘッド数
 → KV Cache 最大                  → KV Cache 削減

Multi-Query Attention (MQA)
┌───┐ ┌───┐ ┌───┐ ┌───┐
│Q_1│ │Q_2│ │Q_3│ │Q_4│
└─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘
  ╲     │     │   ╱
   ╲    │     │  ╱
    ╲   │     │ ╱
     ┌──┴─────┴──┐
     │K_1        │
     │V_1        │
     └───────────┘
 KVヘッド数 = 1
 → KV Cache 最小、品質低下リスク

LLaMA 3.1 の採用:
  8B:  GQA (32 Q heads, 8 KV heads) → 4:1
  70B: GQA (64 Q heads, 8 KV heads) → 8:1
  405B: GQA (128 Q heads, 8 KV heads) → 16:1
```

---

## 8. 実務でのモデル選択フレームワーク

### ASCII 図解 8: モデル選択の判断フロー

```
タスク要件の分析
      │
      ▼
┌─────────────────┐
│ タスクの複雑さは？ │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
  単純        複雑
  │           │
  ▼           ▼
┌────────┐  ┌────────────┐
│分類     │  │推論・分析    │
│抽出     │  │コード生成   │
│フォーマット│  │クリエイティブ│
└────┬───┘  └─────┬──────┘
     │            │
     ▼            ▼
 小型モデル      大型モデル
 (7-8B)         (70B+)
 or API Haiku   or API Opus
     │            │
     ▼            ▼
┌──────────────────────────────────┐
│         非機能要件の確認           │
├────────────┬─────────────────────┤
│ レイテンシ  │ < 100ms → 小型     │
│            │ < 1s → 中型        │
│            │ 許容 → 大型        │
├────────────┼─────────────────────┤
│ コスト     │ 低予算 → OSS + LoRA │
│            │ 中予算 → API        │
│            │ 高予算 → 専用デプロイ │
├────────────┼─────────────────────┤
│ プライバシー│ 厳格 → オンプレミス  │
│            │ 普通 → API          │
├────────────┼─────────────────────┤
│ スループット│ 高 → vLLM + GPU複数 │
│            │ 低 → Ollama で十分  │
└────────────┴─────────────────────┘
```

### コード例 13: モデルルーティングの実装

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class ModelTier(Enum):
    SMALL = "small"    # 8B 級 / Haiku
    MEDIUM = "medium"  # 70B 級 / Sonnet
    LARGE = "large"    # 405B 級 / Opus

@dataclass
class RoutingDecision:
    model_tier: ModelTier
    reason: str
    estimated_cost_per_1k: float  # $/1K tokens

class ModelRouter:
    """タスクの複雑さに応じてモデルを振り分け"""

    # 複雑さの指標
    COMPLEXITY_SIGNALS = {
        "simple_keywords": [
            "分類", "抽出", "フォーマット", "変換",
            "翻訳", "要約", "ラベル付け"
        ],
        "complex_keywords": [
            "分析", "推論", "比較", "評価", "設計",
            "コード生成", "デバッグ", "創作", "戦略"
        ],
    }

    def route(self, task_description: str, max_latency_ms: Optional[int] = None) -> RoutingDecision:
        """タスク記述からモデルティアを決定"""
        desc_lower = task_description.lower()

        # レイテンシ制約がある場合
        if max_latency_ms and max_latency_ms < 200:
            return RoutingDecision(
                model_tier=ModelTier.SMALL,
                reason="レイテンシ制約により小型モデルを選択",
                estimated_cost_per_1k=0.0001,
            )

        # 複雑さの判定
        simple_score = sum(
            1 for kw in self.COMPLEXITY_SIGNALS["simple_keywords"]
            if kw in desc_lower
        )
        complex_score = sum(
            1 for kw in self.COMPLEXITY_SIGNALS["complex_keywords"]
            if kw in desc_lower
        )

        # テキスト長による追加判定
        if len(task_description) > 2000:
            complex_score += 1

        if complex_score > simple_score:
            if complex_score >= 3:
                return RoutingDecision(
                    model_tier=ModelTier.LARGE,
                    reason=f"高複雑度タスク (score: {complex_score})",
                    estimated_cost_per_1k=0.015,
                )
            return RoutingDecision(
                model_tier=ModelTier.MEDIUM,
                reason=f"中複雑度タスク (score: {complex_score})",
                estimated_cost_per_1k=0.003,
            )

        return RoutingDecision(
            model_tier=ModelTier.SMALL,
            reason=f"低複雑度タスク (score: {simple_score})",
            estimated_cost_per_1k=0.0001,
        )

# 使用例
router = ModelRouter()

tasks = [
    "このテキストをポジティブ/ネガティブに分類してください",
    "この設計書のセキュリティ脆弱性を分析し、改善策を推論してください",
    "複数のマイクロサービス間のAPI設計を比較・評価し、最適なアーキテクチャを提案してください",
]

for task in tasks:
    decision = router.route(task)
    print(f"タスク: {task[:50]}...")
    print(f"  → {decision.model_tier.value} ({decision.reason})")
    print(f"  → 推定コスト: ${decision.estimated_cost_per_1k}/1Kトークン")
    print()
```

---

## アンチパターン

### アンチパターン 1: 「大きいモデル = 常に良い」という誤解

```
誤: すべてのタスクに最大モデルを使う
  → コスト爆発、レイテンシ増大、環境負荷

正: タスクの複雑さに応じてモデルサイズを選択する
  - 分類・抽出 → 小型モデル (7-8B) で十分
  - 複雑な推論 → 大型モデル (70B+) が必要
  - ルーティングで振り分ける設計が最適

  実例:
  - Eメール分類: Haiku (約$0.25/1M tokens) で 95% 精度
  - 同じタスクに Opus ($15/1M tokens): 97% 精度
  - 2%の精度向上に60倍のコスト → ROI が合わない
```

### アンチパターン 2: スケーリング則の過信

```
誤: パラメータを増やせば全てのタスクで性能向上する
  → 特定タスクでは小型モデル+専門データが優秀

正: タスク固有の評価を必ず行う
  - ベンチマークスコアと実タスク性能は乖離する
  - ドメイン特化の微調整が汎用大型モデルに勝つことがある
  - 8B + LoRA が 70B のゼロショットに勝つケースは多い
```

### アンチパターン 3: 推論コストを無視した設計

```
誤: 学習コストだけを考慮する
  → 運用開始後、推論コストが学習コストを超える

正: TCO (Total Cost of Ownership) で評価
  - 学習コスト: 一度だけ
  - 推論コスト: 毎日・毎秒発生
  - 1日100万リクエストなら、レイテンシ50ms短縮 = $XXX/月の節約
  - 量子化やバッチ処理で推論コストを10分の1にできる可能性
```

### アンチパターン 4: コンテキスト長の濫用

```
誤: 128K トークンのコンテキストに大量のテキストを詰め込む
  → "Lost in the Middle" 問題で中間部分の情報を見落とす
  → コストが線形以上に増加
  → レイテンシが大幅に悪化

正: 必要な情報だけをコンテキストに含める
  - RAG で関連部分のみを取得
  - 要約してからコンテキストに入れる
  - チャンク分割して並列処理
  - 重要な情報はプロンプトの最初と最後に配置
```

---

## FAQ

### Q1: Transformer 以前のモデル（RNN/LSTM）との最大の違いは？

**A:** 並列処理能力です。RNN/LSTM は系列を逐次処理するため学習が遅く、長い文脈を扱うのが困難でした。Transformer の Self-Attention は全トークン間の関係を同時に計算でき、GPU の並列性を最大限に活用できます。これがスケーリングを可能にした最大の要因です。

具体的な比較:
- **RNN**: O(n) の逐次ステップ、勾配消失問題、並列化不可
- **LSTM**: 勾配消失を緩和、だが依然として逐次処理
- **Transformer**: O(1) の深さ（アテンションで全位置を参照）、完全並列化可能、O(n^2) のメモリだが GPU に最適

### Q2: パラメータ数が大きいほど常に高性能ですか？

**A:** いいえ。Chinchilla の研究が示したように、パラメータ数とデータ量のバランスが重要です。例えば 70B パラメータのモデルでも、十分なデータで学習すれば 175B の GPT-3 を上回ります。また Mixture of Experts (MoE) により、全パラメータを常に使わない効率的な設計も主流になっています。

さらに、推論時の効率も重要です:
- LLaMA 3.1 8B (15T tokens) は、Chinchilla 最適の3倍のデータで学習
- 推論コストは8B分だが、性能は「最適な13B」に匹敵
- 「学習時に余分に投資して、推論時にコスト削減」という戦略

### Q3: LLM の学習には何が最もコストがかかりますか？

**A:** 事前学習のGPU計算コストが圧倒的です。GPT-4 クラスのモデルでは推定数千万〜1億ドルの計算コストがかかります。一方、SFT や RLHF は比較的安価（数千〜数万ドル）で実行可能です。このため、多くの組織は事前学習済みモデルの上に微調整を行う戦略を取ります。

コスト内訳の典型例（70B モデル）:
1. **GPU計算**: 約80%（学習時間 × GPU台数 × 電力）
2. **データ準備**: 約10%（収集、クリーニング、フィルタリング）
3. **人件費**: 約5%（研究者、エンジニア）
4. **インフラ**: 約5%（ストレージ、ネットワーク、冷却）

### Q4: MoE と Dense モデル、どちらを選ぶべき？

**A:** 用途によります。
- **高スループットが必要** → MoE（同じ推論コストでより高性能）
- **メモリ制約が厳しい** → Dense（MoE は全エキスパートをメモリに保持）
- **安定した学習が必要** → Dense（MoE はロードバランスのチューニングが必要）
- **デプロイが簡単** → Dense（MoE は分散推論の設定が複雑）

### Q5: ローカルLLM と API、どちらを使うべき？

**A:** 以下の判断基準で選択します:

| 基準 | ローカル推奨 | API 推奨 |
|------|-------------|---------|
| データプライバシー | 機密データを扱う | 一般的なデータ |
| 初期投資 | GPU資産がある | GPU投資を避けたい |
| スケーラビリティ | 固定負荷 | 変動負荷 |
| 最新モデル | 不要 | 常に最新が必要 |
| カスタマイズ | 微調整が必要 | プロンプトで十分 |
| 運用チーム | MLOps チームがある | 運用は最小限にしたい |

### Q6: コンテキスト長はどこまで信頼できる？

**A:** 公称コンテキスト長と実効性能は異なります。

- **Needle in a Haystack テスト**: 多くのモデルは長いコンテキストの中間部分で情報を見落とす
- **実効的な推奨**: コンテキストの最初と最後に重要な情報を配置
- **RAG との組み合わせ**: 長いドキュメントは RAG で必要部分のみ取得する方が精度が高い
- **コスト**: 入力トークン数に比例してコストが増加するため、無駄なコンテキストは避ける

---

## まとめ

| 項目 | 要点 |
|------|------|
| Transformer | Self-Attention により並列処理と長文脈を実現 |
| 位置エンコーディング | RoPE が現代の標準、長文脈対応も進化中 |
| スケーリング則 | パラメータ・データ・計算の3要素でLoss が予測可能 |
| MoE | 活性パラメータを制限し推論効率を大幅改善 |
| 事前学習 | 兆トークン規模の次トークン予測で言語能力を獲得 |
| SFT | 指示追従能力を付与する教師あり微調整 |
| RLHF/DPO | 人間の好みに合わせるアラインメント手法 |
| 推論最適化 | KV Cache、量子化、vLLM 等でコスト・速度を改善 |
| GQA/MQA | KV Cache のメモリを削減しつつ品質を維持 |
| モデル選択 | タスクの複雑さ・コスト・レイテンシで最適解が変わる |

---

## 次に読むべきガイド

- [01-tokenization.md](./01-tokenization.md) -- トークナイゼーションの仕組みと管理手法
- [02-inference.md](./02-inference.md) -- 推論パラメータの最適化
- [03-fine-tuning.md](./03-fine-tuning.md) -- LoRA/QLoRA によるファインチューニング

---

## 参考文献

1. Vaswani, A. et al. (2017). "Attention Is All You Need." *NeurIPS 2017*. https://arxiv.org/abs/1706.03762
2. Hoffmann, J. et al. (2022). "Training Compute-Optimal Large Language Models (Chinchilla)." *arXiv:2203.15556*. https://arxiv.org/abs/2203.15556
3. Kaplan, J. et al. (2020). "Scaling Laws for Neural Language Models." *arXiv:2001.08361*. https://arxiv.org/abs/2001.08361
4. Ouyang, L. et al. (2022). "Training language models to follow instructions with human feedback (InstructGPT)." *NeurIPS 2022*. https://arxiv.org/abs/2203.02155
5. Su, J. et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv:2104.09864*. https://arxiv.org/abs/2104.09864
6. Shazeer, N. et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." *ICLR 2017*. https://arxiv.org/abs/1701.06538
7. Fedus, W. et al. (2021). "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." *arXiv:2101.03961*. https://arxiv.org/abs/2101.03961
8. Dao, T. et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS 2022*. https://arxiv.org/abs/2205.14135
9. Kwon, W. et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)." *SOSP 2023*. https://arxiv.org/abs/2309.06180
10. Ainslie, J. et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." *arXiv:2305.13245*. https://arxiv.org/abs/2305.13245

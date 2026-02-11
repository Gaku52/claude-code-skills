# LLM概要 — 大規模言語モデルの基礎

> Transformer アーキテクチャからスケーリング則、学習手法まで、LLM の全体像を体系的に理解する。

## この章で学ぶこと

1. **Transformer アーキテクチャ**の仕組みと Self-Attention の原理
2. **スケーリング則**がモデル性能に与える影響とパラメータ数の意味
3. **事前学習・事後学習**の各段階と代表的な学習手法

---

## 1. Transformer アーキテクチャ

### コード例 1: Self-Attention の計算（NumPy）

```python
import numpy as np

def self_attention(Q, K, V):
    """スケーリングドット積アテンション"""
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)  # スケーリング
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)  # Softmax
    return np.matmul(weights, V)

# 例: 4トークン、次元8
np.random.seed(42)
Q = np.random.randn(4, 8)
K = np.random.randn(4, 8)
V = np.random.randn(4, 8)

output = self_attention(Q, K, V)
print(f"出力形状: {output.shape}")  # (4, 8)
```

### ASCII 図解 1: Transformer ブロック構造

```
┌───────────────────────────────────┐
│         出力 (次の層へ)            │
├───────────────────────────────────┤
│      Layer Norm + 残差接続         │
├───────────────────────────────────┤
│     Feed-Forward Network (FFN)    │
│     ┌─────┐  ┌─────┐  ┌─────┐   │
│     │Lin. │→│ ReLU│→│Lin. │   │
│     └─────┘  └─────┘  └─────┘   │
├───────────────────────────────────┤
│      Layer Norm + 残差接続         │
├───────────────────────────────────┤
│   Multi-Head Self-Attention       │
│  ┌──────┐ ┌──────┐ ┌──────┐     │
│  │Head 1│ │Head 2│ │Head N│     │
│  │Q K V │ │Q K V │ │Q K V │     │
│  └──────┘ └──────┘ └──────┘     │
├───────────────────────────────────┤
│   入力エンベディング + 位置符号化    │
└───────────────────────────────────┘
```

### コード例 2: PyTorch で Transformer レイヤー

```python
import torch
import torch.nn as nn

# Transformer Encoder レイヤー
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=True
)

# 入力: (バッチ=2, シーケンス長=10, 次元=512)
src = torch.randn(2, 10, 512)
output = encoder_layer(src)
print(f"出力形状: {output.shape}")  # torch.Size([2, 10, 512])
```

---

## 2. スケーリング則

### ASCII 図解 2: スケーリング則の3要素

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
```

### コード例 3: スケーリング則の概算

```python
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

# 各モデルサイズでの推定 Loss
for params in [7, 13, 70, 405]:
    for tokens in [1, 2, 5, 15]:
        loss = estimate_loss(params, tokens)
        print(f"{params:>3}B params, {tokens:>2}T tokens → Loss: {loss:.3f}")
```

### 比較表 1: 代表モデルのパラメータ数とデータ量

| モデル | パラメータ数 | 学習トークン数 | コンテキスト長 | 公開年 |
|--------|-------------|---------------|---------------|--------|
| GPT-3 | 175B | 300B | 2,048 | 2020 |
| LLaMA 2 | 7-70B | 2T | 4,096 | 2023 |
| GPT-4 | 非公開 (推定1.8T MoE) | 非公開 | 128K | 2023 |
| Claude 3.5 Sonnet | 非公開 | 非公開 | 200K | 2024 |
| LLaMA 3.1 | 8-405B | 15T+ | 128K | 2024 |
| Gemini 1.5 Pro | 非公開 | 非公開 | 1M+ | 2024 |

---

## 3. 学習手法

### ASCII 図解 3: LLM の学習3段階

```
┌─────────────────────────────────────────────────┐
│  Stage 1: 事前学習 (Pre-training)                │
│  ┌─────────────────────────────────────────────┐ │
│  │ 大量テキスト → 次トークン予測 (自己教師あり)    │ │
│  │ 数兆トークン / 数千GPU / 数ヶ月              │ │
│  └─────────────────────────────────────────────┘ │
│                    ↓                             │
│  Stage 2: 教師あり微調整 (SFT)                    │
│  ┌─────────────────────────────────────────────┐ │
│  │ 指示-応答ペア → 対話能力を獲得                │ │
│  │ 数万〜数十万例 / 数日                        │ │
│  └─────────────────────────────────────────────┘ │
│                    ↓                             │
│  Stage 3: アラインメント (RLHF / DPO)            │
│  ┌─────────────────────────────────────────────┐ │
│  │ 人間のフィードバック → 安全性・有用性を向上    │ │
│  │ 比較データ / 報酬モデル                       │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

### コード例 4: Hugging Face で事前学習済みモデルのロード

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

prompt = "大規模言語モデルの仕組みを簡潔に説明してください。"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### コード例 5: 学習コストの概算

```python
def estimate_training_cost(
    params_billions: float,
    tokens_trillions: float,
    gpu_type: str = "H100"
):
    """学習コストの概算"""
    gpu_specs = {
        "A100": {"flops": 312e12, "cost_per_hour": 2.0},
        "H100": {"flops": 989e12, "cost_per_hour": 3.5},
    }
    spec = gpu_specs[gpu_type]

    # 6 * N * D の近似 (FLOPs)
    total_flops = 6 * params_billions * 1e9 * tokens_trillions * 1e12
    gpu_hours = total_flops / spec["flops"] / 3600
    cost = gpu_hours * spec["cost_per_hour"]

    print(f"モデル: {params_billions}B params, {tokens_trillions}T tokens")
    print(f"GPU: {gpu_type}")
    print(f"必要 GPU 時間: {gpu_hours:,.0f} 時間")
    print(f"推定コスト: ${cost:,.0f}")
    print(f"1000台で: {gpu_hours/1000/24:.0f} 日")

estimate_training_cost(70, 15, "H100")
# モデル: 70B params, 15T tokens
# 必要 GPU 時間: 1,821,031 時間
# 推定コスト: $6,373,610
# 1000台で: 76 日
```

### 比較表 2: 学習手法の比較

| 手法 | 目的 | データ | 計算コスト | 必要な専門知識 |
|------|------|--------|-----------|--------------|
| 事前学習 (Pre-training) | 言語理解の獲得 | 数兆トークンのテキスト | 非常に高い (数百万ドル) | 非常に高い |
| SFT (教師あり微調整) | 指示追従能力 | 数万〜数十万の指示-応答ペア | 中程度 | 高い |
| RLHF | 人間の好みに合わせる | 比較ペア + 報酬モデル | 高い | 非常に高い |
| DPO | RLHF の簡略化 | 比較ペアのみ | 中程度 | 中程度 |
| LoRA | 効率的な微調整 | タスク固有データ | 低い | 中程度 |

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
```

### アンチパターン 2: スケーリング則の過信

```
誤: パラメータを増やせば全てのタスクで性能向上する
  → 特定タスクでは小型モデル+専門データが優秀

正: タスク固有の評価を必ず行う
  - ベンチマークスコアと実タスク性能は乖離する
  - ドメイン特化の微調整が汎用大型モデルに勝つことがある
```

---

## FAQ

### Q1: Transformer 以前のモデル（RNN/LSTM）との最大の違いは？

**A:** 並列処理能力です。RNN/LSTM は系列を逐次処理するため学習が遅く、長い文脈を扱うのが困難でした。Transformer の Self-Attention は全トークン間の関係を同時に計算でき、GPU の並列性を最大限に活用できます。これがスケーリングを可能にした最大の要因です。

### Q2: パラメータ数が大きいほど常に高性能ですか？

**A:** いいえ。Chinchilla の研究が示したように、パラメータ数とデータ量のバランスが重要です。例えば 70B パラメータのモデルでも、十分なデータで学習すれば 175B の GPT-3 を上回ります。また Mixture of Experts (MoE) により、全パラメータを常に使わない効率的な設計も主流になっています。

### Q3: LLM の学習には何が最もコストがかかりますか？

**A:** 事前学習のGPU計算コストが圧倒的です。GPT-4 クラスのモデルでは推定数千万〜1億ドルの計算コストがかかります。一方、SFT や RLHF は比較的安価（数千〜数万ドル）で実行可能です。このため、多くの組織は事前学習済みモデルの上に微調整を行う戦略を取ります。

---

## まとめ

| 項目 | 要点 |
|------|------|
| Transformer | Self-Attention により並列処理と長文脈を実現 |
| スケーリング則 | パラメータ・データ・計算の3要素でLoss が予測可能 |
| 事前学習 | 兆トークン規模の次トークン予測で言語能力を獲得 |
| SFT | 指示追従能力を付与する教師あり微調整 |
| RLHF/DPO | 人間の好みに合わせるアラインメント手法 |
| モデル選択 | タスクの複雑さ・コスト・レイテンシで最適解が変わる |

---

## 次に読むべきガイド

- [01-tokenization.md](./01-tokenization.md) — トークナイゼーションの仕組みと管理手法
- [02-inference.md](./02-inference.md) — 推論パラメータの最適化
- [03-fine-tuning.md](./03-fine-tuning.md) — LoRA/QLoRA によるファインチューニング

---

## 参考文献

1. Vaswani, A. et al. (2017). "Attention Is All You Need." *NeurIPS 2017*. https://arxiv.org/abs/1706.03762
2. Hoffmann, J. et al. (2022). "Training Compute-Optimal Large Language Models (Chinchilla)." *arXiv:2203.15556*. https://arxiv.org/abs/2203.15556
3. Kaplan, J. et al. (2020). "Scaling Laws for Neural Language Models." *arXiv:2001.08361*. https://arxiv.org/abs/2001.08361
4. Ouyang, L. et al. (2022). "Training language models to follow instructions with human feedback (InstructGPT)." *NeurIPS 2022*. https://arxiv.org/abs/2203.02155

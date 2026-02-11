# 拡散モデル — DDPM、スコアマッチング、ノイズスケジュール

> 現代の画像生成を支える拡散モデルの数学的基礎から実装まで、前方拡散・逆拡散過程の仕組みを完全に解説する。

---

## この章で学ぶこと

1. **拡散モデルの数学的基礎** — 前方過程・逆過程、変分下限 (ELBO)、ノイズスケジュール
2. **DDPM とスコアマッチングの関係** — ノイズ除去とスコア関数の等価性
3. **高速サンプリング手法** — DDIM、DPM-Solver、Consistency Models による推論の高速化

---

## 1. 拡散モデルの基本概念

### コード例1: 前方拡散過程 (Forward Diffusion)

```python
import torch
import numpy as np

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """線形ノイズスケジュール"""
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """コサインノイズスケジュール (改良版)"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

class ForwardDiffusion:
    def __init__(self, timesteps=1000, schedule="linear"):
        if schedule == "linear":
            self.betas = linear_beta_schedule(timesteps)
        else:
            self.betas = cosine_beta_schedule(timesteps)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        """
        前方過程: q(x_t | x_0) = N(sqrt(ᾱ_t) * x_0, (1 - ᾱ_t) * I)
        任意の時刻 t のノイズ画像を一発で計算可能
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
```

### ASCII図解1: 前方拡散と逆拡散の全体像

```
前方拡散過程 q(x_t | x_{t-1})  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━>

  x₀           x₁           x₂          ...        x_{T-1}        x_T
[原画像] → [少しノイズ] → [更にノイズ] → ... → [ほぼノイズ] → [純粋ノイズ]
  ↑                                                                ↑
  清明な画像                                              N(0, I) ガウスノイズ

<━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  逆拡散過程 p_θ(x_{t-1} | x_t)

  x₀           x₁           x₂          ...        x_{T-1}        x_T
[生成画像] ← [ノイズ除去] ← [ノイズ除去] ← ... ← [ノイズ除去] ← [ランダム]

数学的表現:
  前方: q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
  逆方: p_θ(x_{t-1}| x_t) = N(x_{t-1}; μ_θ(x_t, t), σ²_t I)
```

---

## 2. DDPM の訓練と推論

### コード例2: DDPM のノイズ予測ネットワーク (簡易UNet)

```python
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """時刻 t の位置埋め込み"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class SimpleUNet(nn.Module):
    """簡易UNet: ノイズ ε_θ(x_t, t) を予測"""
    def __init__(self, in_ch=3, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
        )

        # エンコーダ (ダウンサンプリング)
        self.down1 = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        # ボトルネック
        self.bot = nn.Conv2d(256, 256, 3, padding=1)

        # デコーダ (アップサンプリング)
        self.up1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)  # skip接続
        self.out = nn.Conv2d(128, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # エンコーダ
        h1 = F.gelu(self.down1(x))         # [B, 64, H, W]
        h2 = F.gelu(self.down2(h1))        # [B, 128, H/2, W/2]
        h3 = F.gelu(self.down3(h2))        # [B, 256, H/4, W/4]

        # ボトルネック
        h = F.gelu(self.bot(h3))           # [B, 256, H/4, W/4]

        # デコーダ + スキップ接続
        h = F.gelu(self.up1(h))            # [B, 128, H/2, W/2]
        h = torch.cat([h, h2], dim=1)      # [B, 256, H/2, W/2]
        h = F.gelu(self.up2(h))            # [B, 64, H, W]
        h = torch.cat([h, h1], dim=1)      # [B, 128, H, W]
        return self.out(h)                 # [B, 3, H, W] ← ε予測
```

### コード例3: DDPM の訓練ループ

```python
def train_ddpm(model, dataloader, diffusion, optimizer, epochs=100):
    """
    DDPM訓練: ノイズ予測の単純な損失関数
    L = E_{t, x_0, ε} [ ||ε - ε_θ(x_t, t)||² ]
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x_0 = batch.to(device)
            batch_size = x_0.shape[0]

            # ランダムな時刻 t をサンプル
            t = torch.randint(0, diffusion.T, (batch_size,), device=device)

            # ノイズを生成
            noise = torch.randn_like(x_0)

            # ノイズを加えた画像 x_t を計算
            x_t = diffusion.q_sample(x_0, t, noise)

            # ノイズを予測
            noise_pred = model(x_t, t)

            # 損失計算 (単純なMSE)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")
```

### ASCII図解2: DDPM の訓練パイプライン

```
訓練時:
┌────────────────────────────────────────────────────┐
│                                                    │
│   x₀ (元画像)  t (ランダム時刻)  ε (ランダムノイズ)│
│      │              │                │             │
│      v              v                │             │
│   ┌──────────────────┐               │             │
│   │ q(x_t | x_0)    │               │             │
│   │ x_t = √ᾱ_t·x₀  │               │             │
│   │    + √(1-ᾱ_t)·ε │               │             │
│   └───────┬──────────┘               │             │
│           │                          │             │
│           v                          │             │
│   ┌──────────────┐                   │             │
│   │  UNet ε_θ    │──── ε_θ(x_t,t)   │             │
│   │  (x_t, t)    │        │          │             │
│   └──────────────┘        │          │             │
│                           v          v             │
│                     ┌────────────────────┐         │
│                     │  L = ||ε - ε_θ||²  │         │
│                     │  (MSE損失)         │         │
│                     └────────────────────┘         │
└────────────────────────────────────────────────────┘

推論時 (サンプリング):
  x_T ~ N(0,I) → UNet → x_{T-1} → UNet → ... → x_1 → UNet → x_0
```

---

## 3. スコアマッチングとの関係

### コード例4: スコア関数とノイズ予測の等価性

```python
"""
スコアマッチングの視点:
  スコア関数 s_θ(x, t) = ∇_x log p_t(x)

DDPMのノイズ予測との関係:
  s_θ(x_t, t) = -ε_θ(x_t, t) / √(1 - ᾱ_t)

つまり、ノイズ予測 ε_θ はスコア関数の
スケーリングされたバージョン。
"""

def score_from_noise_pred(noise_pred, t, alphas_cumprod):
    """ノイズ予測からスコア関数を計算"""
    sqrt_one_minus_alpha = torch.sqrt(1.0 - alphas_cumprod[t])
    score = -noise_pred / sqrt_one_minus_alpha.view(-1, 1, 1, 1)
    return score

def noise_from_score(score, t, alphas_cumprod):
    """スコア関数からノイズ予測を計算 (逆変換)"""
    sqrt_one_minus_alpha = torch.sqrt(1.0 - alphas_cumprod[t])
    noise = -score * sqrt_one_minus_alpha.view(-1, 1, 1, 1)
    return noise

# Denoising Score Matching (DSM) の損失関数
def dsm_loss(score_model, x_0, t, noise, alphas_cumprod):
    """
    DSM損失: E[ ||s_θ(x_t, t) - ∇_{x_t} log q(x_t|x_0)||² ]
    これはDDPMのノイズ予測損失と本質的に等価
    """
    x_t = forward_diffusion(x_0, t, noise, alphas_cumprod)
    score_pred = score_model(x_t, t)

    # 真のスコア: ∇_{x_t} log q(x_t|x_0) = -(x_t - √ᾱ_t·x_0) / (1-ᾱ_t)
    #           = -ε / √(1-ᾱ_t)
    sqrt_one_minus_alpha = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)
    true_score = -noise / sqrt_one_minus_alpha

    return F.mse_loss(score_pred, true_score)
```

---

## 4. 高速サンプリング手法

### コード例5: DDIM サンプリング

```python
class DDIMSampler:
    """
    DDIM: Denoising Diffusion Implicit Models
    決定論的サンプリングで大幅にステップ数を削減
    """
    def __init__(self, model, alphas_cumprod, timesteps=1000):
        self.model = model
        self.alphas_cumprod = alphas_cumprod
        self.timesteps = timesteps

    def sample(self, shape, num_steps=50, eta=0.0):
        """
        eta=0: 完全決定論的 (DDIM)
        eta=1: DDPMと等価
        """
        # サンプリングに使う時刻のサブセット
        step_size = self.timesteps // num_steps
        time_steps = list(range(0, self.timesteps, step_size))[::-1]

        x = torch.randn(shape, device=device)

        for i, t in enumerate(time_steps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # ノイズ予測
            eps_pred = self.model(x, t_batch)

            # x_0 の予測
            alpha_t = self.alphas_cumprod[t]
            x_0_pred = (x - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
            x_0_pred = torch.clamp(x_0_pred, -1, 1)  # クリッピング

            if i < len(time_steps) - 1:
                t_prev = time_steps[i + 1]
                alpha_prev = self.alphas_cumprod[t_prev]

                # 分散の計算
                sigma = eta * torch.sqrt(
                    (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
                )

                # 予測ノイズの方向
                dir_xt = torch.sqrt(1 - alpha_prev - sigma**2) * eps_pred

                # x_{t-1} の計算
                noise = torch.randn_like(x) if eta > 0 else 0
                x = torch.sqrt(alpha_prev) * x_0_pred + dir_xt + sigma * noise
            else:
                x = x_0_pred

        return x
```

### コード例6: DPM-Solver の概念

```python
"""
DPM-Solver: 微分方程式ソルバーによる高速サンプリング

拡散モデルの逆過程は確率微分方程式 (SDE) または
常微分方程式 (ODE) として定式化できる:

ODE形式 (Probability Flow ODE):
  dx = [f(x,t) - (1/2)g²(t)∇_x log p_t(x)] dt

DPM-SolverはこのODEを高次の方法で効率的に解く:
- 1次: DDIM相当
- 2次: 約20ステップで高品質
- 3次: 約10ステップで高品質
"""

# diffusers での利用例
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# わずか20ステップで高品質な生成
image = pipe(
    prompt="美しい日本庭園",
    num_inference_steps=20,  # 通常の50ステップから大幅削減
).images[0]
```

### ASCII図解3: サンプリング手法の比較

```
                    品質
                     ↑
                     │
            ●DDPM    │    ●DDIM
            (1000)   │    (50)
                     │         ●DPM-Solver
                     │          (20)
                     │               ●Consistency
                     │                (1-4)
                     │
                     │    ●LCM
                     │    (4-8)
                     │
           ──────────┼──────────────────────→ 速度
                     │           (ステップ数の少なさ)

手法別ステップ数と品質:
┌──────────────────┬───────────┬─────────┬─────────┐
│ 手法             │ ステップ  │ 品質    │ 確定的  │
├──────────────────┼───────────┼─────────┼─────────┤
│ DDPM             │ 1000      │ ★★★★★ │ No      │
│ DDIM             │ 50-100    │ ★★★★☆ │ Yes     │
│ DPM-Solver       │ 15-25     │ ★★★★★ │ Yes     │
│ LCM              │ 4-8       │ ★★★★☆ │ Yes     │
│ Consistency      │ 1-2       │ ★★★☆☆ │ Yes     │
│ Turbo/Lightning  │ 1-4       │ ★★★★☆ │ Yes     │
└──────────────────┴───────────┴─────────┴─────────┘
```

---

## 5. ノイズスケジュールの設計

### 比較表1: ノイズスケジュールの比較

| スケジュール | 数式 | 特徴 | 使用例 |
|-------------|------|------|--------|
| **線形 (Linear)** | β_t = β_min + (β_max - β_min) * t/T | 単純、最初に提案 | DDPM原論文 |
| **コサイン (Cosine)** | ᾱ_t = cos²((t/T + s)/(1+s) * π/2) | 終端付近のSNRが改善 | Improved DDPM |
| **スケーリング線形** | β_t を解像度でスケール | 高解像度に対応 | SD3, Flux |
| **シグモイド** | β_t = sigmoid(a + (b-a)*t/T) | 中間で急激に変化 | 一部の研究 |
| **二乗コサイン** | ᾱ_t にcos²を使用 | より滑らかな遷移 | 改良版モデル |

### 比較表2: 主要拡散モデルの比較

| モデル | 年 | パラメータ化 | スケジュール | 潜在空間 | 条件付け |
|--------|-----|-------------|-------------|---------|----------|
| **DDPM** | 2020 | ε予測 | 線形 | なし (ピクセル空間) | なし |
| **Improved DDPM** | 2021 | ε予測 | コサイン | なし | クラス条件 |
| **LDM/SD** | 2022 | ε予測 | 線形 | VAE (4x64x64) | Cross-Attention |
| **SDXL** | 2023 | ε予測 | 線形 | VAE | 複数テキストエンコーダ |
| **SD3** | 2024 | v予測/RF | コサイン変形 | VAE (16ch) | MM-DiT |
| **Flux** | 2024 | RF (Rectified Flow) | 線形 | VAE (16ch) | MM-DiT |

---

## 6. アンチパターン

### アンチパターン1: ステップ数を闇雲に増やす

```
[問題]
「品質が上がるはずだ」とサンプリングステップを500や1000に設定する。

[なぜ問題か]
- 最新のサンプラー (DPM-Solver等) では20-30ステップで十分
- ステップ数増加は計算コストに直結 (推論時間が線形に増加)
- 過剰なステップは品質向上に寄与しない (飽和する)
- GPUメモリとAPIコストの無駄遣い

[正しいアプローチ]
- DPM-Solver++: 20-25ステップ
- Euler Ancestral: 25-35ステップ
- LCM/Turbo: 4-8ステップ
- サンプラーに適したステップ数を選択する
```

### アンチパターン2: 数学を無視した実装

```
[問題]
拡散モデルを「ノイズを足して引くだけ」と単純化して
カスタム実装を行い、αやβの扱いを誤る。

[なぜ問題か]
- ᾱ_t の累積積の計算を誤ると生成品質が壊滅的に低下
- ノイズスケジュールの範囲 (β_min, β_max) が不適切だと
  訓練が収束しない
- 分散の扱いを間違えるとサンプルが発散する

[正しいアプローチ]
- 検証済みのライブラリ (diffusers) を使用する
- カスタム実装する場合は論文の数式を厳密に追う
- 中間ステップの可視化で正しさを確認する
```

---

## FAQ

### Q1: 拡散モデルはなぜGANより安定して訓練できる?

**A:** 主に以下の理由があります:

- **損失関数の単純さ:** DDPMの損失は単純なMSE。GANのように2つのネットワークを均衡させる必要がない
- **モード崩壊なし:** 確率分布全体をモデル化するため、多様性が保たれる
- **勾配の安定性:** ノイズ予測タスクは回帰問題であり、勾配消失/爆発が起きにくい
- **ただし:** 推論コストが高い (多ステップ) というトレードオフがある

### Q2: v-予測と ε-予測の違いは何?

**A:**

- **ε-予測:** 加えたノイズ ε を直接予測。DDPMの原始的な手法
- **v-予測:** v = √ᾱ_t · ε - √(1-ᾱ_t) · x₀ を予測。ノイズと信号の混合
- **v-予測の利点:** 高いSNR領域 (t≈0) と低いSNR領域 (t≈T) の両方で安定。数値的に安定
- **採用例:** SD 2.x はv-予測、SD3/Flux は Rectified Flow

### Q3: Rectified Flow (整流流) とは何?

**A:** Rectified Flow はノイズから画像への変換を**直線的な**パスでモデル化する手法です:

- **従来の拡散:** 曲がりくねったパスでノイズ→画像を変換
- **Rectified Flow:** x₀ と x₁ を直線で結ぶ。v = x₁ - x₀ を予測
- **利点:** より少ないステップで高品質な生成が可能
- **採用:** SD3、Flux で採用され、現在の主流に移行中

---

## まとめ表

| 項目 | 要点 |
|------|------|
| **前方過程** | ガウスノイズを段階的に加えてデータを破壊: q(x_t\|x_{t-1}) |
| **逆過程** | ニューラルネットがノイズを除去して画像を復元: p_θ(x_{t-1}\|x_t) |
| **訓練目的** | ε-予測 (MSE損失): L = E[\|\|ε - ε_θ(x_t, t)\|\|²] |
| **スコアマッチング** | ε-予測 ≡ スコア関数の近似 (数学的に等価) |
| **ノイズスケジュール** | 線形 → コサイン → Rectified Flow へと進化 |
| **高速サンプリング** | DDIM → DPM-Solver → LCM/Turbo (1000→4ステップ) |
| **最新動向** | Rectified Flow + DiT = SD3/Flux の基盤技術 |

---

## 次に読むべきガイド

- [02-prompt-engineering-visual.md](./02-prompt-engineering-visual.md) — プロンプト設計の実践
- [../01-image/00-image-generation.md](../01-image/00-image-generation.md) — Stable Diffusion の具体的な使い方
- [../01-image/02-upscaling.md](../01-image/02-upscaling.md) — 超解像技術との組み合わせ

---

## 参考文献

1. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*. https://arxiv.org/abs/2006.11239
2. Song, J., Meng, C., & Ermon, S. (2021). "Denoising Diffusion Implicit Models (DDIM)." *ICLR 2021*. https://arxiv.org/abs/2010.02502
3. Lu, C. et al. (2022). "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling." *NeurIPS 2022*. https://arxiv.org/abs/2206.00927
4. Song, Y. & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS 2019*. https://arxiv.org/abs/1907.05600
5. Liu, X. et al. (2023). "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." *ICLR 2023*. https://arxiv.org/abs/2209.03003

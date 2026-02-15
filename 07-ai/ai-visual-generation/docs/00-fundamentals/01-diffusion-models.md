# 拡散モデル — DDPM、スコアマッチング、ノイズスケジュール

> 現代の画像生成を支える拡散モデルの数学的基礎から実装まで、前方拡散・逆拡散過程の仕組みを完全に解説する。

---

## この章で学ぶこと

1. **拡散モデルの数学的基礎** — 前方過程・逆過程、変分下限 (ELBO)、ノイズスケジュール
2. **DDPM とスコアマッチングの関係** — ノイズ除去とスコア関数の等価性
3. **高速サンプリング手法** — DDIM、DPM-Solver、Consistency Models による推論の高速化
4. **Latent Diffusion Model (LDM)** — 潜在空間での拡散による計算効率化
5. **Classifier-Free Guidance** — 条件付き生成の品質向上メカニズム
6. **Rectified Flow と Flow Matching** — SD3/Flux の基盤技術

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
    """コサインノイズスケジュール (改良版)

    Nichol & Dhariwal (2021) が提案。
    線形スケジュールの問題点を解決:
    - 終端付近での情報損失が緩やか
    - SNR (Signal-to-Noise Ratio) の低下がより均一
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """シグモイドノイズスケジュール

    中間領域で急激にノイズが増加する特性を持つ。
    一部のモデルで使用される。
    """
    betas = torch.linspace(-6, 6, timesteps)
    betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    return betas

class ForwardDiffusion:
    """前方拡散過程の完全な実装

    数学的背景:
    - 前方過程: q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) * x_{t-1}, beta_t * I)
    - 任意のtへの直接遷移: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1-alpha_bar_t) * I)
    - ここで alpha_bar_t = prod_{s=1}^{t} (1 - beta_s)
    """
    def __init__(self, timesteps=1000, schedule="linear"):
        self.T = timesteps

        if schedule == "linear":
            self.betas = linear_beta_schedule(timesteps)
        elif schedule == "cosine":
            self.betas = cosine_beta_schedule(timesteps)
        elif schedule == "sigmoid":
            self.betas = sigmoid_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # 事前計算 (効率化のため全ての中間値を保存)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
        )

        # 前方過程用の定数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # 逆過程用の定数
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # 事後分布 q(x_{t-1} | x_t, x_0) の分散
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )

        # 事後分布の平均の係数
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

        # SNR (Signal-to-Noise Ratio) の計算
        self.snr = self.alphas_cumprod / (1.0 - self.alphas_cumprod)
        self.log_snr = torch.log(self.snr)

    def q_sample(self, x_0, t, noise=None):
        """
        前方過程: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        任意の時刻 t のノイズ画像を一発で計算可能
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        事後分布 q(x_{t-1} | x_t, x_0) の平均と分散を計算

        これは逆過程の「正解」であり、モデルはこれを近似する。
        mu = coef1 * x_0 + coef2 * x_t
        """
        mu = (
            self.posterior_mean_coef1[t].view(-1, 1, 1, 1) * x_0
            + self.posterior_mean_coef2[t].view(-1, 1, 1, 1) * x_t
        )
        var = self.posterior_variance[t].view(-1, 1, 1, 1)
        log_var = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1)
        return mu, var, log_var

    def get_snr(self, t):
        """指定時刻のSNRを取得 (損失の重み付けに使用)"""
        return self.snr[t]
```

### コード例1b: SNR (Signal-to-Noise Ratio) の分析

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze_schedules():
    """各ノイズスケジュールのSNR特性を比較分析"""
    timesteps = 1000
    schedules = {
        "Linear": linear_beta_schedule(timesteps),
        "Cosine": cosine_beta_schedule(timesteps),
        "Sigmoid": sigmoid_beta_schedule(timesteps),
    }

    analysis = {}
    for name, betas in schedules.items():
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        snr = alphas_cumprod / (1.0 - alphas_cumprod)
        log_snr = torch.log(snr)

        analysis[name] = {
            "betas_range": f"[{betas[0]:.6f}, {betas[-1]:.6f}]",
            "alpha_bar_final": f"{alphas_cumprod[-1]:.6f}",
            "snr_start": f"{snr[0]:.2f}",
            "snr_end": f"{snr[-1]:.6f}",
            "log_snr_range": f"[{log_snr[-1]:.2f}, {log_snr[0]:.2f}]",
            "half_snr_timestep": int(torch.argmin(torch.abs(log_snr)).item()),
        }

    for name, info in analysis.items():
        print(f"\n--- {name} Schedule ---")
        for key, value in info.items():
            print(f"  {key}: {value}")

    return analysis

# analyze_schedules()
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

直接サンプリング (任意のtへ):
  q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)
  where ᾱ_t = ∏_{s=1}^{t} (1 - β_s)
```

### ASCII図解1b: SNR (Signal-to-Noise Ratio) の時間変化

```
SNR (log scale)
  ↑
  │   ★★★★★ 高SNR (信号が支配的)
  │  ╲
  │   ╲── Linear
  │    ╲
  │     ╲      Cosine ──╲
  │      ╲                ╲
  │       ╲                 ╲
  │        ╲                  ╲
  │         ╲                   ╲
  │ ─ ─ ─ ─ ╲─ ─ ─ ─ ─ ─ ─ ─ ─╲─ ─ SNR = 1 (信号=ノイズ)
  │           ╲                    ╲
  │            ╲                     ╲
  │             ╲                      ╲
  │              ★★★★★ 低SNR (ノイズが支配的)
  └─────────────────────────────────────→ 時刻 t
  0                                    T

Linear: 終端で情報がほぼゼロに (最後のステップで急激に劣化)
Cosine: SNR低下がより滑らか (全ステップで均一な学習効果)
```

---

## 2. DDPM の訓練と推論

### コード例2: DDPM のノイズ予測ネットワーク (簡易UNet)

```python
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """時刻 t の位置埋め込み

    Transformer の位置エンコーディングと同様の手法で、
    整数の時刻 t を高次元の連続ベクトルに変換する。

    PE(t, 2i) = sin(t / 10000^(2i/d))
    PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """
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

class ResBlock(nn.Module):
    """残差ブロック + 時刻条件付け"""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(F.silu(self.conv1(x)))
        # 時刻情報を注入
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.norm2(F.silu(self.conv2(h)))
        return h + self.residual(x)

class SelfAttention(nn.Module):
    """Self-Attention ブロック (空間的な長距離依存関係を捉える)"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attention(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + h

class SimpleUNet(nn.Module):
    """簡易UNet: ノイズ epsilon_theta(x_t, t) を予測

    構造:
    - エンコーダ: ダウンサンプリング (畳み込み + ストライド)
    - ボトルネック: Self-Attention + ResBlock
    - デコーダ: アップサンプリング (転置畳み込み) + Skip Connection
    - 時刻条件付け: SinusoidalPositionEmbeddings → MLP → 各ResBlockに注入
    """
    def __init__(self, in_ch=3, time_dim=256, base_ch=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
        )

        # エンコーダ (ダウンサンプリング)
        self.down1 = ResBlock(in_ch, base_ch, time_dim)
        self.down2 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.pool1 = nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1)
        self.pool2 = nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1)
        self.down3 = ResBlock(base_ch * 2, base_ch * 4, time_dim)

        # ボトルネック
        self.bot = nn.Sequential(
            ResBlock(base_ch * 4, base_ch * 4, time_dim),
            SelfAttention(base_ch * 4),
        )

        # デコーダ (アップサンプリング)
        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.up_res1 = ResBlock(base_ch * 4, base_ch * 2, time_dim)  # skip接続で倍
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.up_res2 = ResBlock(base_ch * 2, base_ch, time_dim)      # skip接続で倍
        self.out = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # エンコーダ
        h1 = self.down1(x, t_emb)              # [B, 64, H, W]
        h1_pool = self.pool1(h1)                # [B, 64, H/2, W/2]
        h2 = self.down2(h1_pool, t_emb)         # [B, 128, H/2, W/2]
        h2_pool = self.pool2(h2)                # [B, 128, H/4, W/4]
        h3 = self.down3(h2_pool, t_emb)         # [B, 256, H/4, W/4]

        # ボトルネック (Self-Attention付き)
        h = self.bot[0](h3, t_emb)
        h = self.bot[1](h)

        # デコーダ + スキップ接続
        h = self.up1(h)                         # [B, 128, H/2, W/2]
        h = torch.cat([h, h2], dim=1)           # [B, 256, H/2, W/2]
        h = self.up_res1(h, t_emb)              # [B, 128, H/2, W/2]
        h = self.up2(h)                         # [B, 64, H, W]
        h = torch.cat([h, h1], dim=1)           # [B, 128, H, W]
        h = self.up_res2(h, t_emb)              # [B, 64, H, W]
        return self.out(h)                      # [B, 3, H, W] <- epsilon予測
```

### コード例3: DDPM の訓練ループ

```python
def train_ddpm(model, dataloader, diffusion, optimizer, epochs=100, device="cuda"):
    """
    DDPM訓練: ノイズ予測の単純な損失関数
    L = E_{t, x_0, epsilon} [ ||epsilon - epsilon_theta(x_t, t)||^2 ]

    重要なポイント:
    - t は [0, T) から一様にサンプル
    - epsilon は標準正規分布からサンプル
    - x_t は q_sample で計算 (前方過程の閉形式解)
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
            # 勾配クリッピング (訓練安定化)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
```

### コード例3b: SNR重み付き損失関数 (Min-SNR)

```python
def min_snr_loss(noise_pred, noise, t, snr, gamma=5.0):
    """
    Min-SNR-gamma 損失重み付け

    Hang et al. (2023) "Efficient Diffusion Training via Min-SNR Weighting Strategy"

    問題: 標準的なDDPM損失は全てのtに均一な重みを付けるが、
    低SNR (t≈T) のステップでは損失が大きく、訓練が不安定になりやすい。

    解決: SNR をクリッピングして重み付け
    weight = min(SNR(t), gamma) / SNR(t)
    """
    mse_loss = F.mse_loss(noise_pred, noise, reduction='none')
    mse_loss = mse_loss.mean(dim=[1, 2, 3])  # バッチ内の各サンプルのMSE

    # SNR-based weight
    snr_t = snr[t]
    weight = torch.clamp(snr_t, max=gamma) / snr_t

    weighted_loss = (weight * mse_loss).mean()
    return weighted_loss


def velocity_prediction_loss(model, x_0, t, noise, diffusion):
    """
    v-prediction 損失 (SD 2.x で使用)

    v = sqrt(alpha_bar_t) * epsilon - sqrt(1-alpha_bar_t) * x_0

    epsilon-prediction との違い:
    - 高SNR領域 (t≈0): epsilon-predictionは不安定になりがち → v-predictionは安定
    - 低SNR領域 (t≈T): 両者とも安定
    - 全体的に訓練が安定し、品質が向上
    """
    x_t = diffusion.q_sample(x_0, t, noise)

    # v の計算
    sqrt_alpha = diffusion.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    v_target = sqrt_alpha * noise - sqrt_one_minus_alpha * x_0

    # モデルで v を予測
    v_pred = model(x_t, t)

    return F.mse_loss(v_pred, v_target)
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

### コード例3c: DDPM のサンプリング (逆拡散過程)

```python
class DDPMSampler:
    """DDPM サンプリング: 逆拡散過程の完全な実装"""

    def __init__(self, model, diffusion, device="cuda"):
        self.model = model
        self.diffusion = diffusion
        self.device = device

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        1ステップの逆拡散: p_theta(x_{t-1} | x_t)

        アルゴリズム:
        1. UNet で epsilon を予測
        2. x_0 を推定
        3. 事後分布 q(x_{t-1} | x_t, x_0) の平均を計算
        4. ノイズを加えて x_{t-1} を生成 (t > 0 の場合)
        """
        betas = self.diffusion.betas
        alphas = self.diffusion.alphas
        alphas_cumprod = self.diffusion.alphas_cumprod

        t_batch = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long)

        # epsilon を予測
        eps_pred = self.model(x_t, t_batch)

        # x_0 を推定
        alpha_t = alphas_cumprod[t]
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)

        # 事後分布の平均と分散
        mu, var, log_var = self.diffusion.q_posterior_mean_variance(x_0_pred, x_t, t)

        # ノイズ付加 (t=0 の場合はノイズなし)
        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        x_prev = mu + torch.exp(0.5 * log_var) * noise

        return x_prev

    @torch.no_grad()
    def sample(self, shape, return_intermediates=False):
        """
        完全なサンプリングループ

        x_T ~ N(0, I) から始めて、T ステップかけて x_0 を生成
        """
        self.model.eval()

        # 純粋ノイズから開始
        x = torch.randn(shape, device=self.device)
        intermediates = [x.cpu()] if return_intermediates else None

        for t in reversed(range(self.diffusion.T)):
            x = self.p_sample(x, t)
            if return_intermediates and t % (self.diffusion.T // 10) == 0:
                intermediates.append(x.cpu())

        if return_intermediates:
            return x, intermediates
        return x
```

---

## 3. スコアマッチングとの関係

### コード例4: スコア関数とノイズ予測の等価性

```python
"""
スコアマッチングの視点:
  スコア関数 s_θ(x, t) = nabla_x log p_t(x)

DDPMのノイズ予測との関係:
  s_θ(x_t, t) = -ε_θ(x_t, t) / √(1 - ᾱ_t)

つまり、ノイズ予測 ε_θ はスコア関数の
スケーリングされたバージョン。

歴史的背景:
- Song & Ermon (2019): スコアマッチングによる生成モデル (NCSN)
- Ho et al. (2020): DDPM (ノイズ予測ベース)
- Song et al. (2021): SDE フレームワークで両者を統合
  → Score-based generative models through SDEs
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
    DSM損失: E[ ||s_θ(x_t, t) - nabla_{x_t} log q(x_t|x_0)||^2 ]
    これはDDPMのノイズ予測損失と本質的に等価
    """
    sqrt_alpha = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)

    x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    score_pred = score_model(x_t, t)

    # 真のスコア
    true_score = -noise / sqrt_one_minus_alpha

    return F.mse_loss(score_pred, true_score)
```

### ASCII図解2b: SDEフレームワークでの統一的理解

```
┌─────────────────── SDE フレームワーク ───────────────────┐
│                                                          │
│  前方SDE:  dx = f(x,t)dt + g(t)dw                       │
│  逆方SDE:  dx = [f(x,t) - g²(t)∇_x log p_t(x)]dt + g(t)dw̄  │
│                                                          │
│  Probability Flow ODE (決定論的):                         │
│    dx = [f(x,t) - ½g²(t)∇_x log p_t(x)]dt               │
│                                                          │
│  ┌──────────┐         ┌──────────┐        ┌──────────┐  │
│  │  DDPM    │  ≡      │  NCSN    │  ⊂     │  SDE     │  │
│  │(ε-予測) │  等価    │(スコア)  │ 特殊例  │(統一)   │  │
│  └──────────┘         └──────────┘        └──────────┘  │
│       │                     │                   │        │
│       v                     v                   v        │
│  ε_θ(x_t, t)      s_θ(x_t, t)         f, g, ∇log p    │
│                                                          │
│  関係: s_θ = -ε_θ / √(1-ᾱ_t)                            │
│                                                          │
│  ODE解法:                                                │
│    DDIM ← 1次Euler法                                    │
│    DPM-Solver ← 高次ソルバー (2次, 3次)                  │
│    Rectified Flow ← 直線的ODE                            │
└──────────────────────────────────────────────────────────┘
```

---

## 4. 高速サンプリング手法

### コード例5: DDIM サンプリング

```python
class DDIMSampler:
    """
    DDIM: Denoising Diffusion Implicit Models

    Song et al. (2021) が提案。

    特徴:
    - 決定論的サンプリング (eta=0)
    - 任意のステップ数でサンプリング可能
    - 同一の初期ノイズから一貫した結果
    - 画像間の補間 (latent space interpolation) が可能
    """
    def __init__(self, model, alphas_cumprod, timesteps=1000):
        self.model = model
        self.alphas_cumprod = alphas_cumprod
        self.timesteps = timesteps

    @torch.no_grad()
    def sample(self, shape, num_steps=50, eta=0.0, device="cuda"):
        """
        eta=0: 完全決定論的 (DDIM)
        eta=1: DDPMと等価
        0 < eta < 1: 中間的な確率性
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

    @torch.no_grad()
    def interpolate(self, x1, x2, num_steps=50, alpha_values=None, device="cuda"):
        """
        DDIM を使った画像間の補間

        1. x1, x2 を潜在空間にエンコード (DDIM inversion)
        2. 潜在空間で球面線形補間 (slerp)
        3. 逆拡散で画像に戻す
        """
        if alpha_values is None:
            alpha_values = torch.linspace(0, 1, 7)

        interpolated = []
        for alpha in alpha_values:
            # 球面線形補間 (slerp)
            theta = torch.acos(torch.clamp(
                torch.sum(x1 * x2) / (x1.norm() * x2.norm()), -1, 1
            ))
            if theta < 1e-5:
                z = (1 - alpha) * x1 + alpha * x2
            else:
                z = (torch.sin((1 - alpha) * theta) / torch.sin(theta)) * x1 + \
                    (torch.sin(alpha * theta) / torch.sin(theta)) * x2

            # 逆拡散でデコード
            img = self.sample_from_latent(z, num_steps, device)
            interpolated.append(img)

        return interpolated
```

### コード例6: DPM-Solver の概念と実装

```python
"""
DPM-Solver: 微分方程式ソルバーによる高速サンプリング

拡散モデルの逆過程は確率微分方程式 (SDE) または
常微分方程式 (ODE) として定式化できる:

ODE形式 (Probability Flow ODE):
  dx = [f(x,t) - (1/2)g^2(t) nabla_x log p_t(x)] dt

DPM-SolverはこのODEを高次の方法で効率的に解く:
- 1次: DDIM相当 (Euler法)
- 2次: 約20ステップで高品質 (Heun法相当)
- 3次: 約10ステップで高品質

核心技術:
- log-SNR空間での変数変換 (lambda = log(alpha/sigma))
- テイラー展開による予測の精度向上
- 適応的ステップサイズの選択
"""

class DPMSolverConcept:
    """DPM-Solver の概念的な実装 (教育目的)"""

    def __init__(self, model, alphas_cumprod):
        self.model = model
        self.alphas_cumprod = alphas_cumprod

    def get_lambda(self, t):
        """log-SNR: lambda(t) = log(alpha_t / sigma_t)"""
        alpha_t = torch.sqrt(self.alphas_cumprod[t])
        sigma_t = torch.sqrt(1 - self.alphas_cumprod[t])
        return torch.log(alpha_t / sigma_t)

    def first_order_update(self, x, t, t_prev):
        """
        1次更新 (DDIM相当)
        x_{t-1} = alpha_{t-1}/alpha_t * x_t
                  + sigma_{t-1} * (exp(-h) - 1) * epsilon_theta
        where h = lambda_{t-1} - lambda_t
        """
        alpha_t = torch.sqrt(self.alphas_cumprod[t])
        alpha_prev = torch.sqrt(self.alphas_cumprod[t_prev])
        sigma_t = torch.sqrt(1 - self.alphas_cumprod[t])
        sigma_prev = torch.sqrt(1 - self.alphas_cumprod[t_prev])

        eps = self.model(x, t)

        x_prev = (alpha_prev / alpha_t) * x + \
                 sigma_prev * (torch.exp(self.get_lambda(t_prev) - self.get_lambda(t)) - 1) * eps

        return x_prev

    def second_order_update(self, x, t, t_mid, t_prev):
        """
        2次更新 (精度向上)
        中間点での追加評価でテイラー展開の2次項を補正
        """
        # 1次予測で中間点を求める
        x_mid = self.first_order_update(x, t, t_mid)
        eps_mid = self.model(x_mid, t_mid)

        # 1次の予測
        eps_t = self.model(x, t)

        # 2次の補正
        alpha_prev = torch.sqrt(self.alphas_cumprod[t_prev])
        alpha_t = torch.sqrt(self.alphas_cumprod[t])
        sigma_prev = torch.sqrt(1 - self.alphas_cumprod[t_prev])

        h = self.get_lambda(t_prev) - self.get_lambda(t)
        r = self.get_lambda(t_mid) - self.get_lambda(t)

        # 2次DPMの更新式
        D0 = eps_t
        D1 = (eps_mid - eps_t) / r

        x_prev = (alpha_prev / alpha_t) * x + \
                 sigma_prev * (torch.exp(-h) - 1) * D0 + \
                 sigma_prev * ((torch.exp(-h) - 1) / h + 1) * D1

        return x_prev


# diffusers での利用例
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="dpmsolver++",
    solver_order=2,
    use_karras_sigmas=True,
)

# わずか20ステップで高品質な生成
image = pipe(
    prompt="美しい日本庭園",
    num_inference_steps=20,  # 通常の50ステップから大幅削減
).images[0]
```

### コード例6b: Consistency Model の概念

```python
"""
Consistency Models (Song et al., 2023)

核心アイデア:
- ODE軌道上の任意の点を直接 x_0 にマッピングする関数を学習
- f_theta(x_t, t) = x_0  (任意のtに対して)
- 1ステップで高品質な画像を生成可能

自己整合性条件:
- f_theta(x_t, t) = f_theta(x_{t'}, t')  (同じODE軌道上のx_t, x_{t'})
- つまり、ODE軌道上のどの点からでも同じ x_0 に到達する

訓練方法:
1. Consistency Distillation: 訓練済み拡散モデルからの蒸留
2. Consistency Training: スクラッチからの直接訓練
"""

class ConsistencyModelConcept:
    """Consistency Model の概念的な実装"""

    def __init__(self, backbone, sigma_min=0.002, sigma_max=80.0):
        self.backbone = backbone
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def consistency_function(self, x, t):
        """
        整合性関数: f_theta(x, t) → x_0

        境界条件: f_theta(x, sigma_min) = x
        (ノイズが最小のとき、入力をそのまま返す)

        実装: skip connection で境界条件を満たす
        f_theta(x, t) = c_skip(t) * x + c_out(t) * F_theta(x, t)
        """
        c_skip = self.sigma_min ** 2 / (t ** 2 + self.sigma_min ** 2)
        c_out = self.sigma_min * t / torch.sqrt(t ** 2 + self.sigma_min ** 2)

        F_out = self.backbone(x, t)
        return c_skip * x + c_out * F_out

    def sample(self, shape, device="cuda"):
        """1ステップサンプリング"""
        # 最大ノイズレベルのノイズから開始
        x = torch.randn(shape, device=device) * self.sigma_max
        t = torch.full((shape[0],), self.sigma_max, device=device)

        # 1ステップで x_0 に到達
        x_0 = self.consistency_function(x, t)
        return x_0

    def multistep_sample(self, shape, num_steps=4, device="cuda"):
        """マルチステップサンプリング (品質向上)"""
        sigmas = torch.linspace(
            self.sigma_max, self.sigma_min, num_steps + 1, device=device
        )

        x = torch.randn(shape, device=device) * self.sigma_max

        for i in range(num_steps):
            t = torch.full((shape[0],), sigmas[i], device=device)
            x_0 = self.consistency_function(x, t)

            if i < num_steps - 1:
                # ノイズを再付加して次のステップへ
                noise = torch.randn_like(x)
                x = x_0 + sigmas[i + 1] * noise
            else:
                x = x_0

        return x
```

### ASCII図解3: サンプリング手法の比較

```
                    品質
                     ↑
                     │
            *DDPM    │    *DDIM
            (1000)   │    (50)
                     │         *DPM-Solver
                     │          (20)
                     │               *Consistency
                     │                (1-4)
                     │
                     │    *LCM
                     │    (4-8)
                     │
           ----------+------------------------------> 速度
                     │           (ステップ数の少なさ)

手法別ステップ数と品質:
┌──────────────────┬───────────┬─────────┬─────────┐
│ 手法             │ ステップ  │ 品質    │ 確定的  │
├──────────────────┼───────────┼─────────┼─────────┤
│ DDPM             │ 1000      │ *****   │ No      │
│ DDIM             │ 50-100    │ ****    │ Yes     │
│ DPM-Solver       │ 15-25     │ *****   │ Yes     │
│ LCM              │ 4-8       │ ****    │ Yes     │
│ Consistency      │ 1-2       │ ***     │ Yes     │
│ Turbo/Lightning  │ 1-4       │ ****    │ Yes     │
└──────────────────┴───────────┴─────────┴─────────┘
```

---

## 5. Classifier-Free Guidance (CFG)

### コード例7: CFG の実装

```python
class ClassifierFreeGuidance:
    """
    Classifier-Free Guidance (Ho & Salimans, 2022)

    核心アイデア:
    - 訓練時: 条件 c を確率 p_uncond でランダムにドロップ (空の条件に置換)
    - 推論時: 条件付き予測と無条件予測を線形補間

    数式:
    eps_guided = eps_uncond + w * (eps_cond - eps_uncond)
              = (1 - w) * eps_uncond + w * eps_cond

    w > 1 で条件付き生成が強調される (テキストへの忠実度が上がる)

    トレードオフ:
    - w が大きい: テキストに忠実だが、多様性が低下し、品質が劣化する場合がある
    - w が小さい: 多様性が高いが、テキストとの一致度が下がる
    """

    def __init__(self, model, guidance_scale=7.5):
        self.model = model
        self.w = guidance_scale

    def predict_noise(self, x_t, t, cond, uncond):
        """
        CFG付きノイズ予測

        1. 条件付き予測: eps_cond = model(x_t, t, cond)
        2. 無条件予測: eps_uncond = model(x_t, t, uncond)
        3. ガイダンス: eps = eps_uncond + w * (eps_cond - eps_uncond)
        """
        # バッチで一括計算 (効率化)
        x_combined = torch.cat([x_t, x_t], dim=0)
        t_combined = torch.cat([t, t], dim=0)
        cond_combined = torch.cat([uncond, cond], dim=0)

        noise_pred = self.model(x_combined, t_combined, cond_combined)
        eps_uncond, eps_cond = noise_pred.chunk(2)

        # ガイダンス適用
        guided = eps_uncond + self.w * (eps_cond - eps_uncond)
        return guided

    def dynamic_cfg(self, x_t, t, cond, uncond, t_max):
        """
        動的CFG: 時刻によってガイダンス強度を変動

        初期ステップ (高ノイズ): 高いCFGで大まかな構造を決定
        後期ステップ (低ノイズ): 低いCFGでディテールを自然に
        """
        progress = t.float() / t_max
        dynamic_w = self.w * (0.5 + 0.5 * progress)  # 後半で減衰

        x_combined = torch.cat([x_t, x_t], dim=0)
        t_combined = torch.cat([t, t], dim=0)
        cond_combined = torch.cat([uncond, cond], dim=0)

        noise_pred = self.model(x_combined, t_combined, cond_combined)
        eps_uncond, eps_cond = noise_pred.chunk(2)

        guided = eps_uncond + dynamic_w * (eps_cond - eps_uncond)
        return guided


# 推奨CFG値 (モデル別)
CFG_RECOMMENDATIONS = {
    "SD 1.5":   {"range": (7, 12),  "default": 7.5,  "note": "高めのCFGが必要"},
    "SDXL":     {"range": (5, 8),   "default": 7.0,  "note": "SD1.5より低めが最適"},
    "SD3":      {"range": (4, 7),   "default": 5.0,  "note": "MMDiTはCFGへの感度が高い"},
    "Flux":     {"range": (2, 5),   "default": 3.5,  "note": "Rectified Flowは低CFGで最適"},
    "DALL-E 3": {"range": "N/A",    "default": "auto", "note": "内部で最適化済み"},
}
```

### ASCII図解4: CFG の効果

```
CFG Scale (w)
    1.0          3.5          7.5          15.0         30.0
     │            │            │             │            │
     v            v            v             v            v
  [多様だが     [バランス    [テキスト    [過剰な      [アーティ
   テキストと    良好]        に忠実]      彩度と       ファクト
   無関係な                               コントラスト] 発生]
   生成]

  ←── 多様性が高い ─────────────────── テキスト忠実度が高い ──→
  ←── 画質が安定 ──────────────────── 画質が劣化する傾向 ──→
```

---

## 6. Latent Diffusion Model (LDM)

### コード例8: LDM のアーキテクチャ

```python
"""
Latent Diffusion Model (Rombach et al., 2022)
= Stable Diffusion の基盤アーキテクチャ

核心アイデア:
- ピクセル空間ではなく潜在空間で拡散処理を行う
- 計算コストを大幅に削減 (512x512 → 64x64 の潜在表現)
- VAE (Variational Autoencoder) で画像 ↔ 潜在表現を変換

構成要素:
1. VAE Encoder: 画像 → 潜在表現 (3x512x512 → 4x64x64)
2. U-Net: 潜在空間でノイズ予測 (+ テキスト条件付け)
3. VAE Decoder: 潜在表現 → 画像 (4x64x64 → 3x512x512)
4. Text Encoder: テキスト → 埋め込みベクトル (CLIP)
"""

class LatentDiffusionConcept:
    """LDM の概念的な実装"""

    def __init__(self, vae, unet, text_encoder, scheduler):
        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.vae_scale_factor = 0.18215  # SD 1.x/2.x の VAE スケーリング

    def encode_image(self, image):
        """画像をVAEで潜在空間にエンコード"""
        latent = self.vae.encode(image).latent_dist.sample()
        latent = latent * self.vae_scale_factor
        return latent

    def decode_latent(self, latent):
        """潜在表現をVAEでデコード"""
        latent = latent / self.vae_scale_factor
        image = self.vae.decode(latent).sample
        return image

    def encode_text(self, text):
        """テキストをCLIPでエンコード"""
        tokens = self.text_encoder.tokenizer(
            text, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        )
        text_embeddings = self.text_encoder(tokens.input_ids)
        return text_embeddings

    @torch.no_grad()
    def generate(self, prompt, negative_prompt="", num_steps=50,
                 guidance_scale=7.5, height=512, width=512):
        """LDM の完全な生成パイプライン"""
        device = next(self.unet.parameters()).device

        # 1. テキストエンコード
        text_emb = self.encode_text(prompt)
        uncond_emb = self.encode_text(negative_prompt)
        text_emb_combined = torch.cat([uncond_emb, text_emb])

        # 2. 潜在空間のノイズを初期化
        latent_h = height // 8  # VAE のダウンサンプリング率
        latent_w = width // 8
        latents = torch.randn(1, 4, latent_h, latent_w, device=device)

        # 3. スケジューラの初期化
        self.scheduler.set_timesteps(num_steps)
        latents = latents * self.scheduler.init_noise_sigma

        # 4. 逆拡散ループ (潜在空間で実行)
        for t in self.scheduler.timesteps:
            # CFG: 条件付き/無条件をバッチで予測
            latent_input = torch.cat([latents] * 2)
            latent_input = self.scheduler.scale_model_input(latent_input, t)

            noise_pred = self.unet(latent_input, t, text_emb_combined)

            # CFG 適用
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            # スケジューラステップ
            latents = self.scheduler.step(noise_guided, t, latents).prev_sample

        # 5. VAEデコード (潜在表現 → 画像)
        image = self.decode_latent(latents)
        return image
```

### ASCII図解5: LDM のパイプライン

```
┌─────────────────── Latent Diffusion Model ───────────────────┐
│                                                               │
│  "A cat sitting on a sofa"                                    │
│         │                                                     │
│         v                                                     │
│  ┌──────────────┐                                             │
│  │ CLIP Text    │ テキスト → 77x768 埋め込み                  │
│  │ Encoder      │                                             │
│  └──────┬───────┘                                             │
│         │                                                     │
│         v                                                     │
│  ┌──────────────────────────────────────────┐                  │
│  │              U-Net (潜在空間)             │                  │
│  │                                          │                  │
│  │  入力: 4x64x64 (ノイズ付き潜在表現)     │                  │
│  │                                          │                  │
│  │  ┌──────────────┐  ┌────────────────┐   │  ← z_T (ノイズ) │
│  │  │ Self-        │  │ Cross-         │   │                  │
│  │  │ Attention    │  │ Attention      │   │                  │
│  │  │ (空間)       │  │ (テキスト条件) │   │                  │
│  │  └──────────────┘  └────────────────┘   │                  │
│  │                                          │                  │
│  │  出力: 4x64x64 (ε予測)                  │                  │
│  └──────────────┬───────────────────────────┘                  │
│                 │                                              │
│                 │ ×N ステップ (逆拡散)                         │
│                 │                                              │
│                 v                                              │
│  ┌──────────────┐                                             │
│  │ VAE Decoder  │  4x64x64 → 3x512x512                       │
│  └──────┬───────┘                                             │
│         │                                                     │
│         v                                                     │
│     生成画像 (3x512x512)                                      │
│                                                               │
│  計算量比較:                                                  │
│    ピクセル空間拡散: 3x512x512 = 786,432 次元                 │
│    潜在空間拡散:     4x64x64   =  16,384 次元                 │
│    → 約48倍の圧縮 → 推論・訓練が大幅に高速化                 │
└───────────────────────────────────────────────────────────────┘
```

---

## 7. Rectified Flow と Flow Matching

### コード例9: Rectified Flow の概念

```python
"""
Rectified Flow (Liu et al., 2023)
= SD3, Flux の基盤技術

核心アイデア:
- ノイズ x_0 ~ N(0, I) とデータ x_1 ~ p_data を直線で結ぶ
- 中間点: x_t = (1 - t) * x_0 + t * x_1  (t in [0, 1])
- 速度場 v(x_t, t) = x_1 - x_0 を学習

従来の拡散モデルとの違い:
- DDPM: 曲がりくねったパスでノイズ→画像 (多ステップ必要)
- Rectified Flow: 直線的なパス (少ステップで高品質)

損失関数:
- L = E_{t, x_0, x_1} [ ||v_theta(x_t, t) - (x_1 - x_0)||^2 ]
"""

class RectifiedFlowConcept:
    """Rectified Flow の概念的な実装"""

    def __init__(self, model):
        self.model = model

    def get_xt(self, x_0, x_1, t):
        """線形補間で中間点を計算

        x_t = (1 - t) * x_0 + t * x_1

        x_0: ノイズ (source distribution)
        x_1: データ (target distribution)
        t: 時刻 [0, 1]
        """
        t = t.view(-1, 1, 1, 1)
        return (1 - t) * x_0 + t * x_1

    def training_step(self, x_1, optimizer):
        """訓練の1ステップ

        目的: v_theta(x_t, t) が (x_1 - x_0) を予測するよう学習
        """
        batch_size = x_1.shape[0]
        device = x_1.device

        # ノイズをサンプル
        x_0 = torch.randn_like(x_1)

        # 時刻をサンプル
        t = torch.rand(batch_size, device=device)

        # 中間点を計算
        x_t = self.get_xt(x_0, x_1, t)

        # 速度場を予測
        v_pred = self.model(x_t, t)

        # 真の速度 (直線の方向)
        v_target = x_1 - x_0

        # 損失
        loss = F.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(self, shape, num_steps=28, device="cuda"):
        """Euler法によるサンプリング

        ODE: dx/dt = v_theta(x, t)
        x(0) = noise → x(1) = image
        """
        self.model.eval()

        # t=0 でノイズから開始
        x = torch.randn(shape, device=device)

        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((shape[0],), i * dt, device=device)
            v = self.model(x, t)
            x = x + v * dt

        return x


class FlowMatchingLoss:
    """
    Conditional Flow Matching (CFM) 損失

    Lipman et al. (2023) による一般化。
    Rectified Flow は CFM の特殊ケース。

    一般形: L = E_{t, x_0, x_1} [ ||v_theta(x_t, t) - u_t(x_t | x_0, x_1)||^2 ]

    直線パスの場合: u_t = x_1 - x_0 (Rectified Flow)
    """

    @staticmethod
    def compute(model, x_0, x_1, t):
        """CFM 損失の計算"""
        t_expanded = t.view(-1, 1, 1, 1)

        # 条件付きフローの中間点
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1

        # 条件付きフローの速度 (直線パス)
        u_t = x_1 - x_0

        # モデルの予測
        v_pred = model(x_t, t)

        # CFM 損失
        loss = F.mse_loss(v_pred, u_t)
        return loss
```

### ASCII図解6: Rectified Flow vs DDPM

```
DDPM (曲線パス):                    Rectified Flow (直線パス):

  データ x₁                           データ x₁
     *                                    *
    / ╲                                  /|
   /   ╲                                / |
  /     * x_{t3}                       /  |
 /       ╲                            /   |
*         * x_{t2}                   * x_t|  直線: x_t = (1-t)x₀ + t·x₁
 ╲         ╲                         |    |
  ╲         * x_{t1}                 |    |
   ╲       /                         |   /
    ╲     /                          |  /
     ╲   /                           | /
      * ──────────────                */
  ノイズ x₀                       ノイズ x₀

  T=1000 ステップ必要               28 ステップで十分
  曲がったパスを辿る                直線を辿るので効率的

速度場の学習:
  DDPM:  ε_θ(x_t, t) = ノイズ予測 (間接的)
  RF:    v_θ(x_t, t) = x₁ - x₀ (直接的な方向予測)
```

---

## 8. ノイズスケジュールの設計

### 比較表1: ノイズスケジュールの比較

| スケジュール | 数式 | 特徴 | 使用例 |
|-------------|------|------|--------|
| **線形 (Linear)** | beta_t = beta_min + (beta_max - beta_min) * t/T | 単純、最初に提案 | DDPM原論文 |
| **コサイン (Cosine)** | alpha_bar_t = cos^2((t/T + s)/(1+s) * pi/2) | 終端付近のSNRが改善 | Improved DDPM |
| **スケーリング線形** | beta_t を解像度でスケール | 高解像度に対応 | SD3, Flux |
| **シグモイド** | beta_t = sigmoid(a + (b-a)*t/T) | 中間で急激に変化 | 一部の研究 |
| **二乗コサイン** | alpha_bar_t にcos^2を使用 | より滑らかな遷移 | 改良版モデル |

### 比較表2: 主要拡散モデルの比較

| モデル | 年 | パラメータ化 | スケジュール | 潜在空間 | 条件付け |
|--------|-----|-------------|-------------|---------|----------|
| **DDPM** | 2020 | epsilon予測 | 線形 | なし (ピクセル空間) | なし |
| **Improved DDPM** | 2021 | epsilon予測 | コサイン | なし | クラス条件 |
| **Guided Diffusion** | 2021 | epsilon予測 | 線形 | なし | 分類器ガイダンス |
| **LDM/SD** | 2022 | epsilon予測 | 線形 | VAE (4x64x64) | Cross-Attention |
| **SDXL** | 2023 | epsilon予測 | 線形 | VAE | 複数テキストエンコーダ |
| **SD3** | 2024 | v予測/RF | コサイン変形 | VAE (16ch) | MM-DiT |
| **Flux** | 2024 | RF (Rectified Flow) | 線形 | VAE (16ch) | MM-DiT |

### 比較表3: パラメータ化手法の比較

| パラメータ化 | 予測対象 | 数式 | 利点 | 使用モデル |
|------------|---------|------|------|-----------|
| **epsilon-prediction** | ノイズ epsilon | L = \|\|epsilon - epsilon_theta\|\|^2 | 最も単純、理解しやすい | SD 1.x, SDXL |
| **x_0-prediction** | クリーン画像 x_0 | L = \|\|x_0 - x_0_theta\|\|^2 | 中間結果が可視化しやすい | 一部の研究 |
| **v-prediction** | 速度 v | L = \|\|v - v_theta\|\|^2 | 高/低SNR両方で安定 | SD 2.x |
| **Rectified Flow** | 方向 x_1 - x_0 | L = \|\|(x_1-x_0) - v_theta\|\|^2 | 少ステップで高品質 | SD3, Flux |

---

## 9. アンチパターン

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
カスタム実装を行い、alphaやbetaの扱いを誤る。

[なぜ問題か]
- alpha_bar_t の累積積の計算を誤ると生成品質が壊滅的に低下
- ノイズスケジュールの範囲 (beta_min, beta_max) が不適切だと
  訓練が収束しない
- 分散の扱いを間違えるとサンプルが発散する

[正しいアプローチ]
- 検証済みのライブラリ (diffusers) を使用する
- カスタム実装する場合は論文の数式を厳密に追う
- 中間ステップの可視化で正しさを確認する
```

### アンチパターン3: CFGスケールの固定

```
[問題]
全てのモデル・全てのプロンプトで同じCFGスケールを使用する。

[なぜ問題か]
- モデルによって最適なCFG範囲が大きく異なる
  (SD1.5: 7-12, SDXL: 5-8, Flux: 2-5)
- 過大なCFGは彩度異常やアーティファクトを引き起こす
- プロンプトの長さや複雑さによっても最適値が変わる

[正しいアプローチ]
- モデルの推奨CFG範囲を確認して使用
- 同じプロンプトで複数のCFG値を試して比較
- 動的CFG (時刻に応じてCFGを変動) を検討
```

---

## 10. FAQ

### Q1: 拡散モデルはなぜGANより安定して訓練できる?

**A:** 主に以下の理由があります:

- **損失関数の単純さ:** DDPMの損失は単純なMSE。GANのように2つのネットワークを均衡させる必要がない
- **モード崩壊なし:** 確率分布全体をモデル化するため、多様性が保たれる
- **勾配の安定性:** ノイズ予測タスクは回帰問題であり、勾配消失/爆発が起きにくい
- **ただし:** 推論コストが高い (多ステップ) というトレードオフがある

### Q2: v-予測と epsilon-予測の違いは何?

**A:**

- **epsilon-予測:** 加えたノイズ epsilon を直接予測。DDPMの原始的な手法
- **v-予測:** v = sqrt(alpha_bar_t) * epsilon - sqrt(1-alpha_bar_t) * x_0 を予測。ノイズと信号の混合
- **v-予測の利点:** 高いSNR領域 (t近似0) と低いSNR領域 (t近似T) の両方で安定。数値的に安定
- **採用例:** SD 2.x はv-予測、SD3/Flux は Rectified Flow

### Q3: Rectified Flow (整流流) とは何?

**A:** Rectified Flow はノイズから画像への変換を**直線的な**パスでモデル化する手法です:

- **従来の拡散:** 曲がりくねったパスでノイズ→画像を変換
- **Rectified Flow:** x_0 と x_1 を直線で結ぶ。v = x_1 - x_0 を予測
- **利点:** より少ないステップで高品質な生成が可能
- **採用:** SD3、Flux で採用され、現在の主流に移行中

### Q4: なぜ潜在空間で拡散するのか (LDM)?

**A:** 計算効率の問題です:

- **ピクセル空間:** 512x512x3 = 786,432 次元で拡散 → 非常に遅い
- **潜在空間:** 64x64x4 = 16,384 次元で拡散 → 約48倍の圧縮
- VAEで画質を維持しつつ、拡散処理の計算量を大幅に削減
- 副次効果: 潜在空間は意味的にまとまった表現を持ち、生成品質が向上

### Q5: Consistency Models と蒸留の関係は?

**A:**

- **Consistency Distillation:** 訓練済み拡散モデルの知識を1-4ステップモデルに蒸留
- **Consistency Training:** 蒸留なしで直接訓練 (teacher不要)
- **LCM (Latent Consistency Models):** LDM向けのConsistency蒸留。4-8ステップで商用品質
- **SDXL Turbo / Lightning:** 蒸留ベースの超高速サンプリング (1-4ステップ)

---

## まとめ表

| 項目 | 要点 |
|------|------|
| **前方過程** | ガウスノイズを段階的に加えてデータを破壊: q(x_t\|x_{t-1}) |
| **逆過程** | ニューラルネットがノイズを除去して画像を復元: p_theta(x_{t-1}\|x_t) |
| **訓練目的** | epsilon-予測 (MSE損失): L = E[\|\|epsilon - epsilon_theta(x_t, t)\|\|^2] |
| **スコアマッチング** | epsilon-予測 = スコア関数の近似 (数学的に等価) |
| **CFG** | 条件付き/無条件予測の線形補間で品質向上: w*(cond-uncond)+uncond |
| **LDM** | 潜在空間で拡散 → 計算量を約48倍圧縮 |
| **ノイズスケジュール** | 線形 → コサイン → Rectified Flow へと進化 |
| **高速サンプリング** | DDIM → DPM-Solver → LCM/Turbo (1000→4ステップ) |
| **Rectified Flow** | 直線パスで効率的な生成。SD3/Flux の基盤 |
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
6. Ho, J. & Salimans, T. (2022). "Classifier-Free Diffusion Guidance." *NeurIPS Workshop 2021*. https://arxiv.org/abs/2207.12598
7. Rombach, R. et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022*. https://arxiv.org/abs/2112.10752
8. Song, Y. et al. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR 2021*. https://arxiv.org/abs/2011.13456
9. Nichol, A. & Dhariwal, P. (2021). "Improved Denoising Diffusion Probabilistic Models." *ICML 2021*. https://arxiv.org/abs/2102.09672
10. Song, Y. et al. (2023). "Consistency Models." *ICML 2023*. https://arxiv.org/abs/2303.01469
11. Hang, T. et al. (2023). "Efficient Diffusion Training via Min-SNR Weighting Strategy." *CVPR 2024*. https://arxiv.org/abs/2303.09556
12. Lipman, Y. et al. (2023). "Flow Matching for Generative Modeling." *ICLR 2023*. https://arxiv.org/abs/2210.02747

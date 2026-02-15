# ビジュアルAI概要 — 画像生成の歴史と現在

> AI による視覚コンテンツ生成技術の全体像を、歴史的経緯から最新動向まで体系的に解説する。

---

## この章で学ぶこと

1. **画像生成AIの歴史的変遷** — GAN 登場以前から拡散モデル時代までの技術進化
2. **主要アーキテクチャの分類** — GAN、VAE、拡散モデル、Transformerベースの違い
3. **現在のエコシステムと応用領域** — 商用サービス、オープンソース、産業応用の全体マップ
4. **評価指標と品質測定** — FID、CLIP Score、IS などの客観的指標
5. **法的・倫理的課題** — 著作権、ディープフェイク、バイアスの問題
6. **産業別応用事例** — 広告、ゲーム、ファッション、建築、医療の具体的ケーススタディ

---

## 1. ビジュアルAIの歴史的タイムライン

### コード例1: 画像生成技術の年表データ構造

```python
timeline = [
    {"year": 2014, "event": "GAN (Generative Adversarial Network) 発表",
     "paper": "Goodfellow et al.", "impact": "生成モデルの革命的転換点"},
    {"year": 2015, "event": "DCGAN — 畳み込みGANの安定化",
     "paper": "Radford et al.", "impact": "高解像度画像生成の基盤"},
    {"year": 2016, "event": "Pix2Pix — 条件付き画像変換",
     "paper": "Isola et al.", "impact": "ペア画像による画像変換の確立"},
    {"year": 2017, "event": "Progressive GAN — 段階的成長",
     "paper": "Karras et al.", "impact": "1024x1024 の顔画像生成"},
    {"year": 2017, "event": "CycleGAN — 教師なし画像変換",
     "paper": "Zhu et al.", "impact": "ペアデータ不要のドメイン変換"},
    {"year": 2018, "event": "BigGAN — 大規模クラス条件付き生成",
     "paper": "Brock et al.", "impact": "ImageNetクラスの高品質生成"},
    {"year": 2019, "event": "StyleGAN — スタイル制御",
     "paper": "Karras et al.", "impact": "属性分離と高品質生成"},
    {"year": 2020, "event": "DDPM — 拡散モデルの実用化",
     "paper": "Ho et al.", "impact": "GAN を超える画質を達成"},
    {"year": 2020, "event": "StyleGAN2 — アーティファクト除去",
     "paper": "Karras et al.", "impact": "GANの品質限界を押し上げ"},
    {"year": 2021, "event": "DALL-E / CLIP — テキストから画像へ",
     "paper": "OpenAI", "impact": "自然言語による画像生成"},
    {"year": 2021, "event": "Guided Diffusion — 分類器ガイダンス",
     "paper": "Dhariwal & Nichol", "impact": "拡散モデルがGANを凌駕"},
    {"year": 2022, "event": "Stable Diffusion — オープンソース拡散モデル",
     "paper": "Stability AI", "impact": "民主化・ローカル実行"},
    {"year": 2022, "event": "DALL-E 2 — CLIPベース階層的生成",
     "paper": "OpenAI", "impact": "テキスト理解の飛躍的向上"},
    {"year": 2023, "event": "SDXL, Midjourney v5, DALL-E 3",
     "paper": "各社", "impact": "商用品質の確立"},
    {"year": 2023, "event": "ControlNet — 構造制御の革新",
     "paper": "Zhang et al.", "impact": "ポーズ・エッジ・深度による精密制御"},
    {"year": 2024, "event": "Sora, Flux, SD3",
     "paper": "OpenAI / BFL / Stability AI", "impact": "動画生成・アーキテクチャ刷新"},
    {"year": 2024, "event": "Rectified Flow Transformers",
     "paper": "Esser et al.", "impact": "DiTベースの高効率生成"},
    {"year": 2025, "event": "リアルタイム生成・3D統合",
     "paper": "各社", "impact": "インタラクティブ生成の時代"},
]

for entry in timeline:
    print(f"{entry['year']}: {entry['event']}")
    print(f"  論文: {entry['paper']}")
    print(f"  影響: {entry['impact']}")
    print()
```

### ASCII図解1: ビジュアルAI技術の進化系統図

```
2014        2017        2020        2022        2024
 |           |           |           |           |
 v           v           v           v           v
[GAN]--->[ProGAN]    [DDPM]--->[LDM]--->[SD3/Flux]
  |        |            |         |          |
  +-->[StyleGAN]    [DALL-E]  [SDXL]    [Sora]
  |     (2019)      (2021)    (2023)    (動画)
  |        |                    |
  |    [StyleGAN2]          [ControlNet]
  |     (2020)               (2023)
  |                    |
  +-->[Pix2Pix]   [CLIP]--->[DALL-E 2]--->[DALL-E 3]
  |   (2017)       (2021)     (2022)       (2023)
  |
  +-->[CycleGAN]--->[StarGAN]
      (2017)        (2018)

GAN時代 (2014-2020)     拡散モデル時代 (2020-現在)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 1.1 GAN以前の画像生成 (2012-2014)

画像生成AIの歴史はGAN以前にも遡る。初期の取り組みとして重要なものを整理する。

```python
pre_gan_history = {
    "Boltzmann Machine (2006-2012)": {
        "概要": "確率的生成モデル、制限付きボルツマンマシン(RBM)を積層",
        "限界": "訓練が極めて遅く、高解像度画像の生成は実用的でなかった",
        "貢献": "深層学習のプレトレーニング手法として活用"
    },
    "Deep Belief Networks (2009-2013)": {
        "概要": "RBMを積層した深層生成モデル",
        "限界": "画像の多様性と品質に限界",
        "貢献": "階層的特徴学習の概念を確立"
    },
    "Autoencoder / Denoising AE (2010-2014)": {
        "概要": "入力をエンコードして再構成する自己教師あり学習",
        "限界": "新規画像生成能力が限定的",
        "貢献": "潜在表現学習の基盤を構築"
    },
    "VAE (2013)": {
        "概要": "変分推論を組み合わせた確率的生成モデル",
        "限界": "生成画像がぼやける傾向",
        "貢献": "潜在空間の連続性・補間の概念を確立"
    }
}

for model, details in pre_gan_history.items():
    print(f"--- {model} ---")
    for key, value in details.items():
        print(f"  {key}: {value}")
    print()
```

### 1.2 GAN時代の詳細 (2014-2020)

```python
gan_evolution = {
    "GAN (2014)": {
        "アーキテクチャ": "Generator + Discriminator のミニマックスゲーム",
        "入力": "ランダムノイズ z ~ N(0, I)",
        "出力": "低解像度画像 (32x32, 64x64)",
        "課題": "モード崩壊、訓練不安定性",
        "革新性": "敵対的学習という新パラダイム"
    },
    "DCGAN (2015)": {
        "アーキテクチャ": "畳み込み層によるGAN安定化",
        "革新": "バッチ正規化、ストライド畳み込み、ReLU/LeakyReLU",
        "解像度": "64x64",
        "影響": "ほぼ全ての後続GANのベースラインに"
    },
    "Pix2Pix (2017)": {
        "アーキテクチャ": "条件付きGAN (cGAN) + U-Net Generator",
        "入力": "ペア画像 (入力画像 + 対応する出力画像)",
        "応用": "エッジ→写真、セマンティックマップ→写真",
        "損失": "L1損失 + Adversarial損失"
    },
    "CycleGAN (2017)": {
        "アーキテクチャ": "2組のGAN + Cycle Consistency Loss",
        "革新": "ペアデータ不要の教師なしドメイン変換",
        "応用": "馬→シマウマ、写真→絵画、季節変換"
    },
    "Progressive GAN (2017)": {
        "アーキテクチャ": "4x4 から 1024x1024 へ段階的に成長",
        "革新": "低解像度から訓練開始し段階的に層を追加",
        "影響": "高解像度生成の実現、StyleGANの基盤"
    },
    "StyleGAN (2019)": {
        "アーキテクチャ": "Mapping Network + Synthesis Network + AdaIN",
        "革新": "スタイル空間 W の導入、粗い/中間/細かいスタイル制御",
        "解像度": "1024x1024 の顔画像",
        "影響": "FFHQ データセットの公開、潜在空間操作の発展"
    },
    "StyleGAN2 (2020)": {
        "改善": "AdaIN → Weight Demodulation でアーティファクト除去",
        "追加": "Path Length Regularization で潜在空間の滑らかさ向上",
        "解像度": "1024x1024+",
        "到達点": "GAN品質の事実上の上限"
    }
}

for model, details in gan_evolution.items():
    print(f"\n{'='*50}")
    print(f"  {model}")
    print(f"{'='*50}")
    for key, value in details.items():
        print(f"  {key}: {value}")
```

### 1.3 拡散モデル時代の詳細 (2020-現在)

```python
diffusion_evolution = {
    "DDPM (2020)": {
        "正式名称": "Denoising Diffusion Probabilistic Models",
        "核心アイデア": "画像にノイズを段階的に付加→逆過程でノイズ除去して生成",
        "ステップ数": "1000ステップ (T=1000)",
        "画質": "GAN (FID) を初めて凌駕",
        "課題": "生成速度が非常に遅い (ピクセル空間での処理)"
    },
    "Guided Diffusion (2021)": {
        "革新": "分類器ガイダンスで条件付き生成を改善",
        "発見": "FID と IS の両方で GAN を上回ることを証明",
        "影響": "拡散モデルが生成AI研究の主流に"
    },
    "Classifier-Free Guidance (2022)": {
        "革新": "分類器不要のガイダンス手法",
        "仕組み": "条件付き/無条件の予測を線形補間",
        "パラメータ": "guidance_scale (CFG scale) の概念導入",
        "影響": "現在のほぼ全ての拡散モデルで使用"
    },
    "Latent Diffusion / Stable Diffusion (2022)": {
        "革新": "潜在空間で拡散処理 → 計算コスト大幅削減",
        "構成": "VAE Encoder + U-Net + CLIP Text Encoder + VAE Decoder",
        "影響": "一般消費者GPUでの実行を可能に",
        "オープンソース": "モデル重み公開、民主化の象徴"
    },
    "SDXL (2023)": {
        "改善": "U-Net 拡大、二段階 (Base + Refiner) パイプライン",
        "テキストエンコーダ": "CLIP + OpenCLIP のデュアルエンコーダ",
        "解像度": "1024x1024 ネイティブ",
        "品質": "Midjourney v5 に匹敵する商用品質"
    },
    "DALL-E 3 (2023)": {
        "革新": "キャプション改善による指示追従性の飛躍的向上",
        "テキスト描画": "画像内テキストの生成品質を大幅改善",
        "安全性": "C2PA メタデータ付与",
        "統合": "ChatGPT との統合によるプロンプト拡張"
    },
    "SD3 / Flux (2024)": {
        "アーキテクチャ": "Rectified Flow Transformer (DiT ベース)",
        "革新": "U-Net → Transformer への移行",
        "MMDiT": "テキストと画像の双方向 Attention",
        "品質": "テキスト描画能力の大幅改善"
    },
    "Sora (2024)": {
        "革新": "拡散 Transformer による長尺高品質動画生成",
        "入力": "テキストプロンプト / 画像",
        "出力": "最大60秒の1080p動画",
        "影響": "動画生成AIのブレークスルー"
    }
}

for model, details in diffusion_evolution.items():
    print(f"\n--- {model} ---")
    for key, value in details.items():
        print(f"  {key}: {value}")
```

---

## 2. 主要アーキテクチャの分類

### コード例2: GAN の基本構造

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    """生成器: ランダムノイズから画像を生成"""
    def __init__(self, latent_dim=100, img_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_channels * 64 * 64),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z).view(-1, 3, 64, 64)

class Discriminator(nn.Module):
    """識別器: 本物か生成画像かを判定"""
    def __init__(self, img_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_channels * 64 * 64, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.net(img.view(img.size(0), -1))

# GAN の訓練ループ概要
# min_G max_D [ E[log D(x)] + E[log(1 - D(G(z)))] ]
```

### コード例2b: DCGAN の畳み込み構造

```python
import torch
import torch.nn as nn

class DCGANGenerator(nn.Module):
    """DCGAN生成器: 転置畳み込みで画像を段階的にアップサンプリング"""
    def __init__(self, latent_dim=100, feature_maps=64, img_channels=3):
        super().__init__()
        self.main = nn.Sequential(
            # 入力: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # 状態: (feature_maps*8) x 4 x 4

            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # 状態: (feature_maps*4) x 8 x 8

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # 状態: (feature_maps*2) x 16 x 16

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # 状態: feature_maps x 32 x 32

            nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 出力: img_channels x 64 x 64
        )

    def forward(self, z):
        return self.main(z.view(z.size(0), -1, 1, 1))

class DCGANDiscriminator(nn.Module):
    """DCGAN識別器: ストライド畳み込みでダウンサンプリング"""
    def __init__(self, img_channels=3, feature_maps=64):
        super().__init__()
        self.main = nn.Sequential(
            # 入力: img_channels x 64 x 64
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.main(img).view(-1, 1)


# DCGAN 訓練の安定化テクニック
dcgan_best_practices = {
    "Generator": [
        "転置畳み込み (ConvTranspose2d) を使用",
        "バッチ正規化を全層に適用 (出力層を除く)",
        "ReLU 活性化 (出力層は Tanh)",
    ],
    "Discriminator": [
        "ストライド畳み込みでプーリングを置換",
        "バッチ正規化を全層に適用 (入力層を除く)",
        "LeakyReLU (slope=0.2) を使用",
    ],
    "Training": [
        "Adam optimizer: lr=0.0002, beta1=0.5",
        "重み初期化: N(0, 0.02)",
        "Label smoothing: real labels を 0.9 に",
    ]
}
```

### コード例3: VAE (変分オートエンコーダ) の概念

```python
class VAE(nn.Module):
    """VAE: 潜在空間を学習して画像を再構成・生成"""
    def __init__(self, latent_dim=128):
        super().__init__()
        # エンコーダ: 画像 → 潜在分布のパラメータ
        self.encoder = nn.Sequential(
            nn.Linear(784, 400), nn.ReLU()
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_var = nn.Linear(400, latent_dim)

        # デコーダ: 潜在ベクトル → 画像
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400), nn.ReLU(),
            nn.Linear(400, 784), nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        """再パラメータ化トリック: 勾配を通すための技法"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x.view(-1, 784))
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

# 損失関数: 再構成損失 + KLダイバージェンス
# L = -E[log p(x|z)] + KL(q(z|x) || p(z))
```

### コード例3b: VQ-VAE (ベクトル量子化VAE) の実装

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """VQ-VAE のコードブック: 連続的な潜在表現を離散コードに量子化"""
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )
        self.commitment_cost = commitment_cost

    def forward(self, z):
        # z: (B, D, H, W) → (B, H, W, D) に変換
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, z.shape[-1])

        # 最近傍のコードブックエントリを検索
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)

        # 損失計算
        loss = (
            F.mse_loss(z_q.detach(), z)  # commitment loss
            + self.commitment_cost * F.mse_loss(z_q, z.detach())  # embedding loss
        )

        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        return z_q.permute(0, 3, 1, 2), loss, encoding_indices


class VQVAE(nn.Module):
    """VQ-VAE: 離散潜在空間を持つVAE

    応用:
    - DALL-E (初代) の画像トークナイザとして使用
    - 音声合成 (WaveNet + VQ-VAE)
    - テクスチャ合成
    """
    def __init__(self, in_channels=3, hidden_dim=128,
                 num_embeddings=512, embedding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, embedding_dim, 3, 1, 1),
        )
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, indices
```

### ASCII図解2: 主要生成モデルの動作原理比較

```
┌─────── GAN ────────┐   ┌─────── VAE ────────┐
│                     │   │                     │
│  ノイズ z           │   │  入力 x             │
│     │               │   │     │               │
│     v               │   │     v               │
│  [Generator]        │   │  [Encoder]          │
│     │               │   │     │               │
│     v               │   │  μ, σ (潜在分布)    │
│  生成画像 G(z)      │   │     │               │
│     │               │   │  [再パラメータ化]   │
│     v               │   │     │               │
│  [Discriminator]    │   │     v               │
│     │               │   │  [Decoder]          │
│  本物/偽物          │   │     │               │
│                     │   │  再構成 x'          │
└─────────────────────┘   └─────────────────────┘

┌──── 拡散モデル ─────┐   ┌── Transformer系 ──┐
│                     │   │                     │
│  画像 x₀            │   │  テキスト           │
│     │ (ノイズ付加)   │   │     │               │
│     v               │   │  [Text Encoder]     │
│  x₁ → x₂ → ... xT │   │     │               │
│  (前方拡散過程)     │   │  [Cross-Attention]  │
│                     │   │     │               │
│  xT (純粋ノイズ)    │   │  [Image Decoder]    │
│     │ (ノイズ除去)   │   │     │               │
│     v               │   │  生成画像           │
│  ... → x₁ → x₀     │   │                     │
│  (逆拡散過程)       │   │  例: DALL-E,        │
│                     │   │      Parti           │
└─────────────────────┘   └─────────────────────┘

┌──── VQ-VAE ────────┐   ┌── Flow Matching ───┐
│                     │   │                     │
│  入力 x             │   │  ノイズ z ~ N(0,I)  │
│     │               │   │     │               │
│     v               │   │     v               │
│  [Encoder]          │   │  直線的な軌道       │
│     │               │   │  (Rectified Flow)   │
│  [量子化]           │   │     │               │
│  コードブック検索   │   │  [Velocity Field]   │
│     │               │   │  v(x_t, t)          │
│     v               │   │     │               │
│  [Decoder]          │   │     v               │
│     │               │   │  生成画像 x₁        │
│  再構成 x'          │   │                     │
│  (離散トークン)     │   │  例: Flux, SD3      │
└─────────────────────┘   └─────────────────────┘
```

### 比較表1: 主要生成アーキテクチャの比較

| 特徴 | GAN | VAE | 拡散モデル | Transformer系 | Flow Matching |
|------|-----|-----|-----------|---------------|---------------|
| **画質** | 高い (モード崩壊あり) | やや低い (ぼやけ) | 非常に高い | 非常に高い | 非常に高い |
| **多様性** | 低い傾向 | 高い | 高い | 高い | 高い |
| **訓練安定性** | 不安定 | 安定 | 安定 | 安定 | 安定 |
| **生成速度** | 高速 (1ステップ) | 高速 (1ステップ) | 遅い (多ステップ) | 中程度 | 少ステップ可 |
| **制御性** | 限定的 | 潜在空間操作 | テキスト条件付け | テキスト条件付け | テキスト条件付け |
| **代表モデル** | StyleGAN | VQ-VAE | Stable Diffusion | DALL-E | Flux / SD3 |
| **登場時期** | 2014 | 2013 | 2020 | 2021 | 2023 |
| **理論基盤** | ゲーム理論 | 変分推論 | 確率過程 | 自己回帰 | 常微分方程式 |
| **スケーラビリティ** | 中程度 | 中程度 | 高い | 非常に高い | 非常に高い |

### 比較表1b: 各アーキテクチャの損失関数

| アーキテクチャ | 損失関数 | 数式 | 特徴 |
|---------------|---------|------|------|
| **GAN** | Adversarial Loss | min_G max_D V(D,G) | ゲーム理論的最適化 |
| **VAE** | ELBO | -E[log p(x\|z)] + KL(q\|\|p) | 再構成 + 正則化 |
| **DDPM** | Simple Loss | E[\|\|epsilon - epsilon_theta(x_t, t)\|\|^2] | ノイズ予測 |
| **LDM** | Latent Loss | 潜在空間でのDDPM損失 | 計算効率化 |
| **Flow Matching** | CFM Loss | E[\|\|v_theta(x_t, t) - u_t\|\|^2] | 速度場の学習 |

---

## 3. 現在のエコシステム

### コード例4: 主要APIサービスの利用例

```python
# OpenAI DALL-E 3 API
from openai import OpenAI

client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="富士山の前に立つ赤い鳥居、浮世絵スタイル",
    size="1024x1024",
    quality="hd",
    n=1,
)
image_url = response.data[0].url
print(f"生成画像URL: {image_url}")

# リビジョンされたプロンプトの確認
revised_prompt = response.data[0].revised_prompt
print(f"修正されたプロンプト: {revised_prompt}")

# バリエーション生成 (DALL-E 2)
response_variation = client.images.create_variation(
    image=open("base_image.png", "rb"),
    n=3,
    size="1024x1024",
)
for i, img in enumerate(response_variation.data):
    print(f"バリエーション {i+1}: {img.url}")
```

```python
# Stability AI API (SD3 / SDXL)
import requests
import base64

API_KEY = "your-stability-api-key"

# SD3 による高品質生成
response = requests.post(
    "https://api.stability.ai/v2beta/stable-image/generate/sd3",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
    },
    files={"none": ""},
    data={
        "prompt": "A serene Japanese garden with cherry blossoms, "
                  "koi pond reflecting moonlight, traditional stone lantern",
        "negative_prompt": "low quality, blurry, distorted",
        "output_format": "png",
        "aspect_ratio": "16:9",
        "model": "sd3-large",
    },
)

if response.status_code == 200:
    result = response.json()
    image_data = base64.b64decode(result["image"])
    with open("japanese_garden.png", "wb") as f:
        f.write(image_data)
    print(f"生成完了: seed={result.get('seed')}")
else:
    print(f"エラー: {response.status_code} - {response.text}")
```

```python
# Google Imagen API (Vertex AI 経由)
from google.cloud import aiplatform
from vertexai.preview.vision_models import ImageGenerationModel

aiplatform.init(project="your-project-id", location="us-central1")

model = ImageGenerationModel.from_pretrained("imagegeneration@006")

response = model.generate_images(
    prompt="Professional product photo of a minimalist watch, "
           "studio lighting, white background, 4K quality",
    number_of_images=4,
    aspect_ratio="1:1",
    safety_filter_level="block_some",
    person_generation="allow_adult",
)

for i, image in enumerate(response.images):
    image.save(f"watch_product_{i}.png")
    print(f"画像 {i+1} を保存しました")
```

### コード例5: Hugging Face diffusers ライブラリでローカル実行

```python
from diffusers import StableDiffusionPipeline
import torch

# モデルのロード
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

# メモリ最適化オプション
pipe.enable_attention_slicing()  # Attention のメモリ使用量削減
pipe.enable_vae_slicing()        # VAE のバッチ処理メモリ削減

# 画像生成
image = pipe(
    prompt="東京の夜景、サイバーパンク風、ネオンライト",
    negative_prompt="低品質、ぼやけ、歪み",
    num_inference_steps=50,
    guidance_scale=7.5,
    width=768,
    height=512,
).images[0]

image.save("tokyo_cyberpunk.png")
print("画像を生成しました")
```

### コード例5b: SDXL パイプラインの完全実装

```python
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
)
import torch

# VAE の明示的ロード (品質改善)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)

# Base モデルのロード
base_pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
base_pipe.to("cuda")

# スケジューラの変更 (高速化)
base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    base_pipe.scheduler.config,
    use_karras_sigmas=True,
    algorithm_type="dpmsolver++",
)

# Refiner モデルのロード (オプション)
refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
refiner_pipe.to("cuda")

# 二段階生成 (Base → Refiner)
prompt = "A majestic dragon flying over ancient Japanese castle, "
prompt += "dramatic sunset, volumetric clouds, cinematic lighting, 8k"
negative_prompt = "low quality, blurry, distorted, watermark, text"

# Step 1: Base で粗い生成
base_image = base_pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=40,
    guidance_scale=7.5,
    width=1024,
    height=1024,
    output_type="latent",  # Refiner に渡すため latent で出力
).images

# Step 2: Refiner で細部を改善
refined_image = refiner_pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=base_image,
    num_inference_steps=25,
    strength=0.3,  # Refiner の強度
).images[0]

refined_image.save("dragon_castle_sdxl.png")
print("SDXL 二段階生成が完了しました")
```

### コード例5c: Flux パイプラインの実装

```python
from diffusers import FluxPipeline
import torch

# Flux.1-dev モデルのロード
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)

# メモリ最適化
pipe.enable_model_cpu_offload()  # GPU VRAM を節約

# Flux はテキスト描画が非常に得意
image = pipe(
    prompt='A wooden sign in a forest that reads "Welcome to the '
           'Enchanted Forest" in elegant calligraphy, '
           'moss-covered, sunlight filtering through leaves',
    num_inference_steps=28,
    guidance_scale=3.5,
    width=1024,
    height=768,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]

image.save("flux_text_rendering.png")
print("Flux による画像生成が完了しました")
```

### ASCII図解3: ビジュアルAIエコシステムマップ

```
┌─────────────── ビジュアルAIエコシステム ───────────────┐
│                                                       │
│  ┌─── 商用サービス ───┐  ┌── オープンソース ──┐       │
│  │ DALL-E 3 (OpenAI)  │  │ Stable Diffusion   │       │
│  │ Midjourney          │  │ Flux (BFL)         │       │
│  │ Adobe Firefly       │  │ PixArt             │       │
│  │ Google Imagen       │  │ SDXL               │       │
│  │ Canva AI            │  │ ComfyUI            │       │
│  │ Ideogram            │  │ Kolors (Kwai)      │       │
│  └─────────────────────┘  └────────────────────┘       │
│                                                       │
│  ┌─── フレームワーク ──┐  ┌── 応用領域 ───────┐       │
│  │ diffusers (HF)      │  │ 広告・マーケ       │       │
│  │ ComfyUI             │  │ ゲーム開発         │       │
│  │ Automatic1111       │  │ ファッション       │       │
│  │ InvokeAI            │  │ 建築・インテリア   │       │
│  │ Fooocus             │  │ 映画・動画制作     │       │
│  │ ForgeUI             │  │ 教育・研究         │       │
│  │ SwarmUI             │  │ 医療画像           │       │
│  └─────────────────────┘  └────────────────────┘       │
│                                                       │
│  ┌─── モデルハブ ──────┐  ┌── ハードウェア ───┐       │
│  │ Hugging Face        │  │ NVIDIA GPU         │       │
│  │ Civitai             │  │ Apple Silicon       │       │
│  │ Replicate           │  │ クラウドGPU        │       │
│  │ RunPod              │  │ (AWS/GCP/Azure)    │       │
│  └─────────────────────┘  └────────────────────┘       │
└───────────────────────────────────────────────────────┘
```

### 比較表2: 主要画像生成サービスの比較

| サービス | 提供形態 | 価格帯 | 強み | API提供 | ローカル実行 |
|---------|---------|--------|------|---------|------------|
| **DALL-E 3** | クラウドAPI | 従量課金 | テキスト理解力 | あり | 不可 |
| **Midjourney** | Discord/Web | サブスク $10~ | アート品質 | 限定的 | 不可 |
| **Stable Diffusion** | オープンソース | 無料 | カスタマイズ性 | あり | 可能 |
| **Adobe Firefly** | 統合ツール | Creative Cloud | 商用安全性 | あり | 不可 |
| **Flux** | オープンウェイト | 無料/有料 | テキスト描画 | あり | 可能 |
| **Google Imagen** | クラウドAPI | 従量課金 | フォトリアル | あり | 不可 |
| **Ideogram** | Web/API | フリーミアム | テキスト描画 | あり | 不可 |

---

## 4. 画像生成AIの評価指標

### コード例6: FID (Frechet Inception Distance) の計算

```python
import torch
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader

class FIDCalculator:
    """FID: 生成画像と実画像の分布間距離を測定する標準指標

    FID が低いほど生成画像の品質が高い。
    - FID < 10: 非常に高品質
    - FID 10-50: 良好
    - FID 50-100: 中程度
    - FID > 100: 低品質
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.model = inception_v3(pretrained=True, transform_input=False)
        self.model.fc = torch.nn.Identity()  # 最終層を除去
        self.model = self.model.to(device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225]),
        ])

    def extract_features(self, dataloader):
        """画像群からInceptionの特徴量を抽出"""
        features = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.device)
                feat = self.model(batch)
                features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0)

    def calculate_statistics(self, features):
        """平均ベクトルと共分散行列を計算"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def calculate_fid(self, real_features, gen_features):
        """FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2√(Σ_r·Σ_g))"""
        mu_r, sigma_r = self.calculate_statistics(real_features)
        mu_g, sigma_g = self.calculate_statistics(gen_features)

        # 平均の差のノルム
        diff = mu_r - mu_g
        diff_sq = np.sum(diff ** 2)

        # 共分散の平方根
        covmean = sqrtm(sigma_r @ sigma_g)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff_sq + np.trace(sigma_r + sigma_g - 2 * covmean)
        return float(fid)


# 使用例
calculator = FIDCalculator()
# real_loader, gen_loader は DataLoader オブジェクト
# real_features = calculator.extract_features(real_loader)
# gen_features = calculator.extract_features(gen_loader)
# fid_score = calculator.calculate_fid(real_features, gen_features)
# print(f"FID Score: {fid_score:.2f}")
```

### コード例7: CLIP Score の計算

```python
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

class CLIPScoreCalculator:
    """CLIP Score: テキストと生成画像の整合性を評価

    テキストプロンプトに対する生成画像の忠実度を測定する。
    スコアが高いほど、プロンプトとの一致度が高い。

    典型的なスコア範囲:
    - 0.30+: 非常に高い一致度
    - 0.25-0.30: 良好な一致度
    - 0.20-0.25: 中程度
    - 0.20未満: 低い一致度
    """
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def calculate_score(self, image: Image.Image, text: str) -> float:
        """単一画像とテキストのCLIPスコアを計算"""
        inputs = self.processor(
            text=[text], images=image,
            return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            # コサイン類似度を計算
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            score = (image_embeds @ text_embeds.T).item()
        return score

    def batch_evaluate(self, images, prompts):
        """バッチ評価: 複数の画像-テキストペアを一括評価"""
        results = []
        for img, prompt in zip(images, prompts):
            score = self.calculate_score(img, prompt)
            results.append({
                "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                "clip_score": round(score, 4),
            })

        avg_score = sum(r["clip_score"] for r in results) / len(results)
        return results, avg_score


# 使用例
# calc = CLIPScoreCalculator()
# image = Image.open("generated_image.png")
# score = calc.calculate_score(image, "a cat sitting on a sofa")
# print(f"CLIP Score: {score:.4f}")
```

### コード例8: Inception Score (IS) の計算

```python
import torch
import numpy as np
from torchvision.models import inception_v3
import torch.nn.functional as F

class InceptionScoreCalculator:
    """Inception Score: 生成画像の品質と多様性を同時に評価

    IS = exp(E[KL(p(y|x) || p(y))])

    - p(y|x): 個々の生成画像の分類確率分布 (明確であるほど良い)
    - p(y): 全体の周辺分布 (均一であるほど多様性が高い)

    典型的なスコア範囲 (ImageNet):
    - IS > 100: 非常に高品質 (実画像に近い)
    - IS 50-100: 良好
    - IS 10-50: 中程度
    - IS < 10: 低品質
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.model = inception_v3(pretrained=True).to(device).eval()

    def calculate_is(self, images, splits=10):
        """Inception Score を計算"""
        preds = []
        with torch.no_grad():
            for img in images:
                img = img.unsqueeze(0).to(self.device)
                pred = F.softmax(self.model(img), dim=1)
                preds.append(pred.cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        # スプリットごとにISを計算して平均
        scores = []
        chunk_size = len(preds) // splits
        for i in range(splits):
            chunk = preds[i * chunk_size:(i + 1) * chunk_size]
            p_y = np.mean(chunk, axis=0, keepdims=True)
            kl_div = chunk * (np.log(chunk + 1e-16) - np.log(p_y + 1e-16))
            kl_mean = np.mean(np.sum(kl_div, axis=1))
            scores.append(np.exp(kl_mean))

        return float(np.mean(scores)), float(np.std(scores))
```

### 比較表3: 画像生成の評価指標一覧

| 指標 | 測定対象 | 計算方法 | 利点 | 限界 |
|------|---------|---------|------|------|
| **FID** | 画質 + 多様性 | Inception特徴量の分布間距離 | 最も広く使用される | 最低数千枚必要 |
| **IS** | 画質 + 多様性 | 条件付き/周辺分布のKL | 計算が比較的軽い | データセットバイアス |
| **CLIP Score** | テキスト整合性 | CLIP埋め込みの類似度 | 意味的評価が可能 | CLIPの限界に依存 |
| **LPIPS** | 知覚的類似度 | 特徴量空間での距離 | 人間の知覚に近い | ペア画像が必要 |
| **SSIM** | 構造的類似度 | 輝度・コントラスト・構造 | 解釈しやすい | ピクセル単位の限界 |
| **Human Eval** | 総合品質 | 人間による主観評価 | 最も信頼性が高い | コストと時間がかかる |
| **Aesthetic Score** | 美的品質 | LAION Aesthetic Predictor | 美しさの定量評価 | 主観的基準 |

---

## 5. 産業別応用事例

### 5.1 広告・マーケティング

```python
# 広告バナー自動生成パイプラインの例
class AdBannerGenerator:
    """広告バナーの自動生成ワークフロー

    実務フロー:
    1. ブランドガイドラインの入力
    2. ターゲット層に合わせたプロンプト生成
    3. 複数バリエーションの一括生成
    4. 品質フィルタリング
    5. A/Bテスト候補の選定
    """
    def __init__(self, brand_config):
        self.brand_colors = brand_config["colors"]
        self.brand_style = brand_config["style"]
        self.target_sizes = brand_config["sizes"]

    def generate_prompt(self, product, campaign_theme, target_audience):
        """ブランドガイドラインに沿ったプロンプト生成"""
        base_prompt = f"{product}, {campaign_theme}, "
        style_prompt = f"{self.brand_style} style, "
        color_prompt = f"color palette: {', '.join(self.brand_colors)}, "
        audience_prompt = f"appealing to {target_audience}, "
        quality_prompt = "professional advertising photography, studio lighting, 4K"

        return base_prompt + style_prompt + color_prompt + audience_prompt + quality_prompt

    def generate_variations(self, prompt, num_variations=8):
        """複数バリエーションを生成"""
        variations = []
        # 構図バリエーション
        compositions = [
            "centered composition",
            "rule of thirds",
            "diagonal composition",
            "symmetrical layout",
        ]
        # 背景バリエーション
        backgrounds = [
            "clean white background",
            "gradient background",
        ]
        for comp in compositions:
            for bg in backgrounds:
                full_prompt = f"{prompt}, {comp}, {bg}"
                variations.append(full_prompt)
        return variations[:num_variations]

# 使用例
brand_config = {
    "colors": ["#2563EB", "#F59E0B", "#FFFFFF"],
    "style": "modern minimalist",
    "sizes": ["1200x628", "1080x1080", "160x600"],
}
generator = AdBannerGenerator(brand_config)
prompt = generator.generate_prompt(
    product="wireless earbuds",
    campaign_theme="summer freedom",
    target_audience="young professionals 25-35"
)
print(f"生成プロンプト: {prompt}")
```

### 5.2 ゲーム開発

```python
# ゲームアセット生成パイプラインの例
class GameAssetPipeline:
    """ゲーム開発における画像生成AIの活用パターン

    活用領域:
    - コンセプトアート: 初期デザイン案の高速プロトタイピング
    - テクスチャ生成: タイリング可能なテクスチャの自動生成
    - UIアイコン: アイテム・スキルアイコンのバリエーション
    - 背景: 2Dゲームの背景アートの自動生成
    """

    TEXTURE_PROMPT_TEMPLATE = (
        "{material} texture, seamless tileable, "
        "top-down view, flat lighting, {style}, "
        "game asset, PBR material, 512x512"
    )

    ICON_PROMPT_TEMPLATE = (
        "{item_type} icon, {rarity} rarity, "
        "game UI icon, {art_style}, "
        "centered, clean edges, transparent background"
    )

    CONCEPT_ART_TEMPLATE = (
        "{subject}, concept art, {genre} game style, "
        "{atmosphere}, detailed, professional illustration, "
        "artstation quality"
    )

    def generate_texture_prompts(self, materials):
        """テクスチャ生成用プロンプトの一括作成"""
        prompts = []
        for material in materials:
            prompt = self.TEXTURE_PROMPT_TEMPLATE.format(
                material=material["name"],
                style=material.get("style", "photorealistic"),
            )
            prompts.append({
                "prompt": prompt,
                "filename": f"texture_{material['name'].replace(' ', '_')}.png",
                "use_case": material.get("use_case", "environment"),
            })
        return prompts

    def generate_item_icons(self, items):
        """ゲームアイテムアイコンの生成"""
        rarity_styles = {
            "common": "simple design, grey border",
            "rare": "glowing blue outline, detailed",
            "epic": "purple aura, ornate design",
            "legendary": "golden glow, extremely detailed, particle effects",
        }
        prompts = []
        for item in items:
            rarity = item.get("rarity", "common")
            prompt = self.ICON_PROMPT_TEMPLATE.format(
                item_type=item["name"],
                rarity=rarity,
                art_style=rarity_styles.get(rarity, "simple design"),
            )
            prompts.append(prompt)
        return prompts

# 使用例
pipeline = GameAssetPipeline()
materials = [
    {"name": "cobblestone", "style": "medieval fantasy", "use_case": "floor"},
    {"name": "wooden planks", "style": "rustic", "use_case": "building"},
    {"name": "metal plate", "style": "sci-fi", "use_case": "spaceship"},
]
texture_prompts = pipeline.generate_texture_prompts(materials)
for tp in texture_prompts:
    print(f"[{tp['use_case']}] {tp['filename']}")
    print(f"  プロンプト: {tp['prompt'][:80]}...")
```

### 5.3 建築・インテリアデザイン

```python
# 建築ビジュアライゼーションの例
class ArchitecturalVisualizer:
    """建築・インテリア分野でのAI画像生成活用

    実務での活用パターン:
    1. 初期コンセプト: クライアントへの提案用イメージ高速作成
    2. スタイル探索: 複数のインテリアスタイルを迅速に比較
    3. リノベーション提案: 既存写真 → AI変換で完成予想図
    4. マテリアル試行: 同じ空間で異なる素材を試す
    """

    STYLES = {
        "modern": "modern minimalist interior, clean lines, "
                  "neutral colors, large windows, natural light",
        "japanese": "Japanese wabi-sabi interior, natural materials, "
                    "tatami, shoji screens, zen garden view",
        "scandinavian": "Scandinavian hygge interior, warm wood tones, "
                        "cozy textiles, simple furniture, white walls",
        "industrial": "industrial loft interior, exposed brick, "
                      "metal fixtures, concrete floor, high ceilings",
        "art_deco": "Art Deco interior, geometric patterns, "
                    "gold accents, velvet furniture, marble floors",
    }

    def generate_room_prompt(self, room_type, style,
                             specific_requirements=None):
        """部屋のビジュアライゼーション用プロンプト生成"""
        base = f"{room_type}, {self.STYLES.get(style, style)}"
        quality = (
            "architectural visualization, photorealistic rendering, "
            "interior design magazine quality, professional photography, "
            "8K resolution, ray tracing"
        )
        prompt = f"{base}, {quality}"
        if specific_requirements:
            prompt += f", {specific_requirements}"
        return prompt

# 使用例
viz = ArchitecturalVisualizer()
prompt = viz.generate_room_prompt(
    room_type="living room",
    style="japanese",
    specific_requirements="overlooking a mountain view, evening golden hour"
)
print(f"建築ビジュアライゼーション用プロンプト:\n{prompt}")
```

### 5.4 ファッション・Eコマース

```python
class FashionAIWorkflow:
    """ファッション業界でのAI画像生成活用

    主要ユースケース:
    - バーチャル試着 (Virtual Try-On)
    - 商品画像のバリエーション生成
    - ルックブック自動生成
    - テキスタイルパターンデザイン
    - モデル写真のポーズ・背景変更
    """

    def generate_product_variants(self, product_description, colors, backgrounds):
        """商品画像のカラーバリエーション生成"""
        prompts = []
        for color in colors:
            for bg in backgrounds:
                prompt = (
                    f"{color} {product_description}, "
                    f"{bg}, "
                    f"professional product photography, "
                    f"e-commerce style, clean, well-lit, "
                    f"high detail, studio lighting"
                )
                prompts.append({
                    "prompt": prompt,
                    "color": color,
                    "background": bg,
                })
        return prompts

    def generate_textile_pattern(self, pattern_type, color_scheme, season):
        """テキスタイルパターンの生成"""
        prompt = (
            f"{pattern_type} textile pattern, "
            f"color scheme: {color_scheme}, "
            f"{season} collection, "
            f"seamless tileable pattern, "
            f"fashion fabric design, high resolution, "
            f"surface pattern design"
        )
        return prompt

# 使用例
workflow = FashionAIWorkflow()
variants = workflow.generate_product_variants(
    product_description="leather handbag, tote style",
    colors=["black", "cognac brown", "navy blue"],
    backgrounds=["white studio background", "lifestyle outdoor setting"],
)
for v in variants:
    print(f"[{v['color']}] [{v['background']}]")
    print(f"  {v['prompt'][:80]}...")
```

### 5.5 医療画像への応用

```python
class MedicalImagingAI:
    """医療画像分野でのAI生成技術の応用

    注意: 医療分野では生成AIの使用に厳格な規制と倫理的配慮が必要。
    以下は研究・教育目的の例示。

    応用領域:
    - データ拡張: 少数サンプルの医療画像を増やして分類器を訓練
    - 匿名化: 患者プライバシーを保ちながら教育用画像を生成
    - シミュレーション: 稀な疾患の画像を合成して訓練データを補強
    - セグメンテーション支援: ラベル付き合成データで分割精度を向上
    """

    def create_augmentation_pipeline(self, modality, condition):
        """医療画像データ拡張パイプラインの設計"""
        pipeline_config = {
            "modality": modality,
            "condition": condition,
            "augmentation_steps": [
                {
                    "type": "geometric",
                    "methods": ["rotation", "flipping", "scaling"],
                    "note": "解剖学的に妥当な範囲に限定"
                },
                {
                    "type": "intensity",
                    "methods": ["brightness", "contrast", "noise"],
                    "note": "診断に影響しない範囲の変動"
                },
                {
                    "type": "generative",
                    "methods": ["GAN-based synthesis", "diffusion-based"],
                    "note": "専門医による品質検証が必須"
                }
            ],
            "validation": {
                "expert_review": "放射線科医による目視確認",
                "statistical_check": "分布の一致性検証 (FID等)",
                "downstream_eval": "分類/検出タスクでの性能検証",
            },
            "ethical_requirements": [
                "IRB (倫理審査委員会) の承認",
                "患者データの完全匿名化",
                "生成画像の明示的ラベリング (合成データであることの明記)",
                "臨床使用前の規制当局への申請",
            ]
        }
        return pipeline_config
```

---

## 6. コスト分析とインフラ選定

### コード例9: コスト比較計算ツール

```python
class VisualAICostCalculator:
    """画像生成AIのコスト比較ツール

    API課金 vs ローカルGPU vs クラウドGPU のコスト比較
    """

    # 2025年時点の概算価格
    API_PRICING = {
        "dall-e-3": {
            "standard_1024": 0.040,   # USD per image
            "hd_1024": 0.080,
            "standard_1792": 0.080,
            "hd_1792": 0.120,
        },
        "stability_sd3": {
            "per_image": 0.065,
        },
        "midjourney": {
            "basic_monthly": 10,      # 200 images/month
            "standard_monthly": 30,   # unlimited relax
            "pro_monthly": 60,
        },
    }

    LOCAL_GPU_COSTS = {
        "RTX 4090": {
            "purchase_price": 1600,
            "power_watts": 450,
            "vram_gb": 24,
            "images_per_hour_sdxl": 120,
            "images_per_hour_flux": 40,
        },
        "RTX 4070": {
            "purchase_price": 600,
            "power_watts": 200,
            "vram_gb": 12,
            "images_per_hour_sdxl": 60,
            "images_per_hour_flux": 15,
        },
    }

    CLOUD_GPU_COSTS = {
        "A100_40gb": {"hourly_rate": 3.50, "images_per_hour_sdxl": 200},
        "L4": {"hourly_rate": 0.80, "images_per_hour_sdxl": 80},
        "T4": {"hourly_rate": 0.35, "images_per_hour_sdxl": 30},
    }

    def calculate_monthly_cost(self, method, images_per_month,
                               electricity_rate_per_kwh=0.15):
        """月間コストの試算"""
        if method == "dall-e-3":
            return images_per_month * self.API_PRICING["dall-e-3"]["hd_1024"]

        elif method == "local_rtx4090":
            gpu = self.LOCAL_GPU_COSTS["RTX 4090"]
            hours_needed = images_per_month / gpu["images_per_hour_sdxl"]
            electricity = (gpu["power_watts"] / 1000) * hours_needed * electricity_rate_per_kwh
            # GPU減価償却 (3年)
            depreciation = gpu["purchase_price"] / 36
            return electricity + depreciation

        elif method == "cloud_a100":
            cloud = self.CLOUD_GPU_COSTS["A100_40gb"]
            hours_needed = images_per_month / cloud["images_per_hour_sdxl"]
            return hours_needed * cloud["hourly_rate"]

        return 0

    def compare_all(self, images_per_month):
        """全方式のコスト比較"""
        methods = ["dall-e-3", "local_rtx4090", "cloud_a100"]
        results = {}
        for method in methods:
            cost = self.calculate_monthly_cost(method, images_per_month)
            results[method] = {
                "monthly_cost_usd": round(cost, 2),
                "cost_per_image_usd": round(cost / images_per_month, 4)
                    if images_per_month > 0 else 0,
            }
        return results

# 使用例
calc = VisualAICostCalculator()
for volume in [100, 1000, 10000]:
    print(f"\n--- 月間 {volume:,} 枚生成の場合 ---")
    comparison = calc.compare_all(volume)
    for method, costs in comparison.items():
        print(f"  {method}: "
              f"${costs['monthly_cost_usd']:,.2f}/月 "
              f"(${costs['cost_per_image_usd']:.4f}/枚)")
```

### 比較表4: インフラ選定ガイド

| 条件 | 推奨方式 | 理由 |
|------|---------|------|
| 月100枚以下 | API (DALL-E 3等) | 初期投資不要、即座に利用開始 |
| 月100-1000枚 | クラウドGPU | 柔軟なスケーリング、中程度のコスト |
| 月1000枚以上 | ローカルGPU | 長期的にコスト最安、カスタマイズ自由 |
| 商用利用重視 | Adobe Firefly / DALL-E 3 | ライセンス明確、訴訟リスク低減 |
| カスタムモデル必要 | ローカルGPU + ファインチューニング | 完全制御、独自データでの訓練 |
| プロトタイプ段階 | Midjourney + API | 高速イテレーション、低コスト |

---

## 7. 法的・倫理的課題の詳細

### 7.1 著作権問題の現状

```python
copyright_landscape = {
    "日本": {
        "現行法": "著作権法30条の4 — AI学習目的の著作物利用は原則適法",
        "生成物の著作権": "「創作的寄与」の有無で判断 — プロンプトだけでは不十分な可能性",
        "議論状況": "文化審議会で検討中、ガイドライン策定が進行",
        "実務上の対応": [
            "生成物にAI生成であることを明記",
            "第三者の著作物との類似性チェック",
            "商用利用時は利用規約の確認",
        ]
    },
    "米国": {
        "現行法": "Copyright Office — 人間の創作的関与がない部分は著作権保護外",
        "判例": "Thaler v. Perlmutter (2023) — AI単独の著作物は登録不可",
        "混合事例": "人間とAIの共同制作は部分的に保護される可能性",
        "進行中": "複数の訴訟 (Getty Images vs Stability AI 等)",
    },
    "EU": {
        "AI Act": "2024年施行 — AI生成コンテンツの透明性義務",
        "ラベリング": "AI生成画像には明示的なラベル付けが必要",
        "学習データ": "オプトアウト権の保障が求められる",
    }
}

for region, details in copyright_landscape.items():
    print(f"\n=== {region} ===")
    for key, value in details.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    - {item}")
        else:
            print(f"  {key}: {value}")
```

### 7.2 ディープフェイクとバイアスの問題

```python
ethical_concerns = {
    "ディープフェイク": {
        "リスク": [
            "政治家・著名人の偽造映像による世論操作",
            "リベンジポルノ等の個人攻撃",
            "詐欺・なりすましへの悪用",
        ],
        "対策技術": [
            "C2PA (Coalition for Content Provenance and Authenticity) メタデータ",
            "SynthID (Google) — 不可視の電子透かし",
            "AIフォレンジクス — 生成画像の検出技術",
            "ブロックチェーンベースの来歴追跡",
        ],
        "規制動向": [
            "EU AI Act: ハイリスクAI分類、透明性義務",
            "日本: 不正競争防止法の改正議論",
            "米国: 州法レベルでの規制 (CA, TX 等)",
        ]
    },
    "バイアス": {
        "問題": [
            "学習データの偏りによるステレオタイプの再生産",
            "特定の人種・性別・文化の過小/過大表現",
            "NSFW コンテンツの生成リスク",
        ],
        "緩和策": [
            "多様なデータセットでの訓練",
            "安全フィルターの実装",
            "レッドチーミングによる脆弱性発見",
            "コミュニティガイドラインの策定",
        ]
    }
}

for topic, details in ethical_concerns.items():
    print(f"\n--- {topic} ---")
    for category, items in details.items():
        print(f"  [{category}]")
        for item in items:
            print(f"    - {item}")
```

---

## 8. アンチパターン

### アンチパターン1: 技術選定なき導入

```
[問題]
「とにかくAI画像生成を導入しよう」と要件定義なしに
最も話題のツールを採用する。

[なぜ問題か]
- 商用利用で著作権問題が発生するリスク
- コスト見積もりが甘く予算超過
- ユースケースに合わないモデルの選択

[正しいアプローチ]
1. 用途を明確化（広告素材? プロトタイプ? 最終成果物?）
2. 法的要件を確認（商用利用可否、学習データの透明性）
3. 品質要件を定義（解像度、スタイル、一貫性）
4. コスト試算（API従量課金 vs ローカルGPU投資）
5. 複数モデルでPoC実施後に選定
```

### アンチパターン2: 「AIが全て解決する」思考

```
[問題]
デザインワークフロー全体をAI生成に置き換えようとする。

[なぜ問題か]
- AI生成は「素材生成」であり「デザイン」ではない
- ブランドの一貫性維持が困難
- 細かい修正・調整に大量の試行錯誤が必要
- 人間のクリエイティブ判断は依然不可欠

[正しいアプローチ]
- AIを「アシスタント」として位置づけ
- 初期アイデア出し → AI生成 → 人間による選別・調整
- ブランドガイドラインとの整合性チェックは人間が担当
- AIの得意領域（バリエーション生成、背景生成）に集中
```

### アンチパターン3: プロンプトの使い回し

```
[問題]
一度うまくいったプロンプトを、異なるモデルやバージョンで
そのまま使い回す。

[なぜ問題か]
- モデルごとにプロンプトの解釈が異なる
- バージョンアップでプロンプト処理ロジックが変わる
- DALL-E 3 はプロンプトを内部で書き換える
- Midjourney と SD では最適なプロンプト構造が違う

[正しいアプローチ]
- モデルごとのプロンプトガイドを確認
- 同じ意図でもモデルに合わせてプロンプトを調整
- プロンプトテンプレートをモデル別に管理
- A/Bテストでプロンプト効果を検証
```

### アンチパターン4: 生成画像の無検証利用

```
[問題]
生成された画像を品質チェックなしで
そのまま本番環境に使用する。

[なぜ問題か]
- アーティファクト (手指の異常、テキストの破綻) が含まれる可能性
- 既存の著作物に酷似した画像が生成されるリスク
- 不適切なコンテンツが混入する可能性
- ブランドイメージとの不一致

[正しいアプローチ]
1. 自動品質チェック (CLIP Score, 人体検出, テキスト検出)
2. 類似画像検索 (TinEye, Google Reverse Image Search)
3. 人間レビュー (最終チェックは必ず人間が実施)
4. 承認フローの整備 (デザイナー/法務のサインオフ)
```

---

## 9. FAQ

### Q1: 画像生成AIを始めるのに必要なスペックは?

**A:** 用途によって異なります。

- **APIサービス利用のみ:** 普通のPC/スマホで十分
- **ローカル実行 (SD系):** VRAM 8GB以上のGPU推奨 (RTX 3060以上)
- **SDXL 実行:** VRAM 12GB以上推奨 (RTX 4070以上)
- **Flux 実行:** VRAM 16GB以上推奨 (RTX 4080/4090)
- **ファインチューニング:** VRAM 16-24GB (RTX 4090, A100)
- **Apple Silicon Mac:** M1以上で実行可能 (ただしGPUより遅い)
  - M1/M2: SD 1.5 は実用的、SDXL はやや遅い
  - M3 Max/Ultra: SDXL, Flux も実用的な速度

### Q2: 生成した画像の著作権はどうなる?

**A:** 国・サービスによって異なりますが、一般的に:

- **米国:** AI生成画像は著作権保護の対象外 (2023年著作権局ガイダンス)
- **日本:** AI生成物の著作権は議論中、創作的関与の度合いによる
- **サービス規約:** DALL-E、Midjourney等は商用利用を許可 (有料プラン)
- **学習データ問題:** 学習に使われた画像の著作権侵害リスクは別問題
- **実務対応:** 商用利用時はサービス利用規約を必ず確認し、社内法務と相談

### Q3: GANと拡散モデル、どちらを学ぶべき?

**A:** 2025年現在、**拡散モデルを優先的に学ぶことを推奨**します。

- 現在の主流モデル (SD, DALL-E, Flux) はすべて拡散モデルベース
- GANは理論的理解として重要だが、実用面では拡散モデルが優位
- ただし、リアルタイム生成等ではGAN的手法が復活する傾向もある
- 両方の基礎を理解し、拡散モデルに深く取り組むのが最適
- Flow Matching (Flux, SD3) が次のトレンドとして注目

### Q4: ComfyUI と Automatic1111 WebUI のどちらを使うべき?

**A:** 用途と経験レベルによります。

| 観点 | ComfyUI | Automatic1111 WebUI |
|------|---------|-------------------|
| **操作方式** | ノードベース (ビジュアルプログラミング) | フォーム入力型 |
| **学習コスト** | やや高い | 低い |
| **カスタマイズ性** | 非常に高い | 中程度 |
| **ワークフロー管理** | JSON で保存・共有可能 | 設定ファイルで管理 |
| **最新モデル対応** | 非常に速い | やや遅い |
| **推奨ユーザー** | 技術者、パワーユーザー | 初心者、デザイナー |
| **メモリ効率** | 良い | 普通 |

### Q5: 画像生成AIで稼ぐことはできる?

**A:** 以下のようなビジネスモデルが存在します。

1. **ストックフォト販売:** Adobe Stock, Shutterstock等が一部AI画像を受け入れ (要開示)
2. **受託デザイン:** AI生成をワークフローに組み込んだデザインサービス
3. **LoRAモデル販売:** Civitai等でカスタムモデルを販売/サブスク
4. **教育・コンサルティング:** 企業向けAI画像生成の導入支援
5. **アプリ開発:** AI画像生成機能を組み込んだSaaSの開発
6. **プリントオンデマンド:** AI生成アートのTシャツ・ポスター販売

ただし、競争が激化しており、差別化のためには以下が重要:
- 特定ドメインの専門知識 (医療、建築、ファッション等)
- 独自のワークフロー・パイプライン構築力
- ファインチューニング・LoRA訓練の技術力
- ブランディングとマーケティング能力

### Q6: 画像生成AIの限界は何か?

**A:** 2025年時点での主な限界:

- **手指・解剖学:** 改善されつつあるが完全ではない
- **テキスト描画:** Flux, DALL-E 3 で大幅改善、ただし長文は依然困難
- **一貫性:** 同一キャラクターの複数ポーズ生成は依然として難しい
- **精密な空間配置:** 「Aの左にBを、Bの上にCを」のような複雑な配置指示
- **カウンティング:** 正確な数の物体を生成すること (「3匹の猫」が4匹になる等)
- **物理法則:** 反射、影、透過などの物理現象の正確な再現
- **長尺動画:** 一貫性のある長時間動画の生成

---

## まとめ表

| 項目 | 要点 |
|------|------|
| **歴史** | GAN(2014) → VAE発展 → 拡散モデル(2020) → 商用化(2022~) → Flow Matching(2024~) |
| **現在の主流** | 拡散モデル (Latent Diffusion) + Transformerの融合、Flow Matchingへの移行 |
| **主要プレイヤー** | OpenAI, Stability AI, Midjourney, Adobe, Google, BFL, Ideogram |
| **オープンソース** | Stable Diffusion, Flux が中心。Civitai にコミュニティモデル |
| **応用領域** | 広告、ゲーム、ファッション、建築、映画、教育、医療 |
| **評価指標** | FID (品質), CLIP Score (整合性), IS (多様性), Human Eval (総合) |
| **法的課題** | 著作権、学習データの透明性、ディープフェイク規制、AI Act |
| **技術トレンド** | マルチモーダル化、リアルタイム生成、3D統合、Flow Matching |
| **コスト戦略** | 少量→API、中量→クラウドGPU、大量→ローカルGPU |

---

## 次に読むべきガイド

- [01-diffusion-models.md](./01-diffusion-models.md) — 拡散モデルの数学的基礎と実装
- [02-prompt-engineering-visual.md](./02-prompt-engineering-visual.md) — 効果的なプロンプト設計
- [../01-image/00-image-generation.md](../01-image/00-image-generation.md) — 具体的な画像生成ツールの使い方

---

## 参考文献

1. Goodfellow, I. et al. (2014). "Generative Adversarial Nets." *NeurIPS 2014*. https://arxiv.org/abs/1406.2661
2. Kingma, D.P. & Welling, M. (2013). "Auto-Encoding Variational Bayes." *ICLR 2014*. https://arxiv.org/abs/1312.6114
3. Radford, A. et al. (2015). "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." *ICLR 2016*. https://arxiv.org/abs/1511.06434
4. Isola, P. et al. (2017). "Image-to-Image Translation with Conditional Adversarial Networks." *CVPR 2017*. https://arxiv.org/abs/1611.07004
5. Zhu, J.-Y. et al. (2017). "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." *ICCV 2017*. https://arxiv.org/abs/1703.10593
6. Karras, T. et al. (2019). "A Style-Based Generator Architecture for Generative Adversarial Networks." *CVPR 2019*. https://arxiv.org/abs/1812.04948
7. Ho, J. et al. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*. https://arxiv.org/abs/2006.11239
8. Dhariwal, P. & Nichol, A. (2021). "Diffusion Models Beat GANs on Image Synthesis." *NeurIPS 2021*. https://arxiv.org/abs/2105.05233
9. Rombach, R. et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022*. https://arxiv.org/abs/2112.10752
10. Ramesh, A. et al. (2022). "Hierarchical Text-Conditional Image Generation with CLIP Latents." *arXiv*. https://arxiv.org/abs/2204.06125
11. Zhang, L. et al. (2023). "Adding Conditional Control to Text-to-Image Diffusion Models." *ICCV 2023*. https://arxiv.org/abs/2302.05543
12. Esser, P. et al. (2024). "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." *ICML 2024*. https://arxiv.org/abs/2403.03206
13. Heusel, M. et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." *NeurIPS 2017*. https://arxiv.org/abs/1706.08500
14. Radford, A. et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *ICML 2021*. https://arxiv.org/abs/2103.00020

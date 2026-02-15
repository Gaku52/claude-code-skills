# 動画生成 — Sora、Runway、Pika

> AI による動画生成技術の仕組みと主要プラットフォームの使い方を、テキスト-to-ビデオから画像-to-ビデオまで実践的に解説する。

---

## この章で学ぶこと

1. **動画生成モデルの原理** — 時間軸への拡散モデルの拡張、時空間アーキテクチャ
2. **主要プラットフォームの比較と使い分け** — Sora、Runway Gen-3、Pika、Kling の特徴
3. **プロダクション向けワークフロー** — 生成動画の品質管理と後処理パイプライン
4. **フレーム補間と超解像** — RIFE、Real-ESRGAN による動画品質向上テクニック
5. **実務統合** — API連携、バッチ処理、コスト管理の実践手法

---

## 1. 動画生成の技術基盤

### 1.1 動画拡散モデルの数学的基礎

動画拡散モデルは、画像拡散モデルの自然な拡張として構築される。画像拡散モデルがノイズ付きの2D画像からクリーンな画像を推定するのに対し、動画拡散モデルは3Dテンソル（フレーム数 x 高さ x 幅）を扱う。

```
画像拡散 vs 動画拡散:

画像: x ∈ R^{C×H×W}      →  空間的なノイズ除去
動画: x ∈ R^{F×C×H×W}    →  時空間的なノイズ除去

ここで:
  F = フレーム数 (16, 24, 48, etc.)
  C = チャネル数 (3: RGB)
  H, W = 高さ、幅

前方拡散過程:
  q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)

逆拡散過程 (学習対象):
  p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t, c), σ_t^2 I)

条件 c には:
  - テキストプロンプト (CLIP / T5 エンコーディング)
  - 参照画像 (Image-to-Video の場合)
  - カメラパラメータ (カメラ制御の場合)
  が含まれる
```

### コード例1: 時空間拡散モデルの概念

```python
"""
動画生成モデルの基本構造:
画像の拡散モデルに時間軸の処理を追加

Image Diffusion:  UNet(x_t, t, text) → ε(空間ノイズ)
Video Diffusion:  UNet(x_t, t, text) → ε(時空間ノイズ)
                  x_t の形状: [B, F, C, H, W]
                  (バッチ, フレーム数, チャネル, 高さ, 幅)
"""

import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    """フレーム間の時間的注意機構"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B*H*W, F, C] — 各空間位置でフレーム間の注意
        residual = x
        x = self.norm(x)
        x, _ = self.attention(x, x, x)
        return x + residual

class SpatioTemporalBlock(nn.Module):
    """
    時空間処理ブロック:
    1. 空間注意 (各フレーム内)
    2. 時間注意 (フレーム間)
    3. Cross-Attention (テキスト条件)
    """
    def __init__(self, dim):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(dim, 8)
        self.temporal_attn = TemporalAttention(dim)
        self.cross_attn = nn.MultiheadAttention(dim, 8)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, text_emb):
        B, F, C, H, W = x.shape

        # 1. 空間注意 (各フレーム独立)
        x_spatial = x.view(B * F, C, H * W).permute(0, 2, 1)
        x_spatial, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x = x_spatial.permute(0, 2, 1).view(B, F, C, H, W) + x

        # 2. 時間注意 (フレーム間)
        x_temporal = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, F, C)
        x_temporal = self.temporal_attn(x_temporal)
        x = x_temporal.reshape(B, H, W, F, C).permute(0, 3, 4, 1, 2) + x

        # 3. テキスト条件付けクロス注意
        # (簡略化)

        return x
```

### 1.2 DiT (Diffusion Transformer) ベースのアーキテクチャ

Sora に代表される最新の動画生成モデルは、従来の UNet ベースから DiT (Diffusion Transformer) ベースに移行している。DiT はスケーリング則に優れ、大規模データでの学習に適している。

```python
class VideoDiTBlock(nn.Module):
    """
    Video Diffusion Transformer Block

    Sora 等が採用するアーキテクチャの概念実装:
    - パッチ化: 動画を時空間パッチに分割
    - Transformer: パッチ間の全注意
    - スケーラビリティ: モデルサイズに比例して品質向上
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        # AdaLN (Adaptive Layer Norm) for timestep conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

    def forward(self, x, c):
        """
        x: [B, N, D] — N = 時空間パッチ数
        c: [B, D] — 条件ベクトル (timestep + text)
        """
        # AdaLN パラメータの計算
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # Self-Attention with modulation
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # Feed-Forward with modulation
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class VideoTokenizer:
    """動画を時空間パッチ (トークン) に変換"""

    def __init__(self, patch_size=(2, 16, 16), latent_dim=1024):
        """
        patch_size: (temporal, height, width)
        - temporal=2: 2フレームずつをグループ化
        - height=16, width=16: 16x16のパッチ
        """
        self.patch_size = patch_size
        self.latent_dim = latent_dim

    def patchify(self, video_latent):
        """
        video_latent: [B, F, C, H, W]
        → patches: [B, N, D]
        N = (F/pt) * (H/ph) * (W/pw)
        """
        B, F, C, H, W = video_latent.shape
        pt, ph, pw = self.patch_size

        # フレーム、高さ、幅をパッチに分割
        n_t = F // pt
        n_h = H // ph
        n_w = W // pw

        patches = video_latent.reshape(
            B, n_t, pt, C, n_h, ph, n_w, pw
        )
        patches = patches.permute(0, 1, 4, 6, 2, 3, 5, 7)
        patches = patches.reshape(B, n_t * n_h * n_w, -1)

        return patches  # [B, N, pt*C*ph*pw]

    def unpatchify(self, patches, original_shape):
        """パッチを動画に復元"""
        B, F, C, H, W = original_shape
        pt, ph, pw = self.patch_size
        n_t, n_h, n_w = F // pt, H // ph, W // pw

        patches = patches.reshape(B, n_t, n_h, n_w, pt, C, ph, pw)
        patches = patches.permute(0, 1, 4, 5, 2, 6, 3, 7)
        video = patches.reshape(B, F, C, H, W)

        return video
```

### ASCII図解1: 動画生成モデルのアーキテクチャ

```
┌──────────── 動画生成パイプライン ─────────────────┐
│                                                   │
│  テキスト/画像入力                                 │
│       │                                           │
│       v                                           │
│  ┌──────────────┐                                 │
│  │ Text Encoder │  (T5 / CLIP)                    │
│  └──────┬───────┘                                 │
│         │                                         │
│         v                                         │
│  ┌──────────────────────────────────────┐         │
│  │  Spatio-Temporal DiT / UNet          │         │
│  │                                      │         │
│  │  フレーム1  フレーム2  ...  フレームN │ ← ノイズ│
│  │  ┌──┐      ┌──┐           ┌──┐      │         │
│  │  │空│──時──│空│──時──...──│空│      │         │
│  │  │間│  間  │間│  間       │間│      │         │
│  │  │注│  注  │注│  注       │注│      │         │
│  │  │意│  意  │意│  意       │意│      │         │
│  │  └──┘      └──┘           └──┘      │         │
│  │       ↕         ↕              ↕     │         │
│  │  [Cross-Attention: テキスト条件]     │         │
│  └──────────────┬───────────────────────┘         │
│                 │                                  │
│                 v                                  │
│  ┌──────────────┐                                 │
│  │ VAE Decoder  │  各フレームをデコード             │
│  └──────┬───────┘                                 │
│         │                                         │
│         v                                         │
│  [フレーム1] [フレーム2] ... [フレームN] = 動画    │
└───────────────────────────────────────────────────┘
```

### ASCII図解: DiT vs UNet アーキテクチャの比較

```
UNet ベース (Stable Video Diffusion 等):

入力 ──→ [Down1]──→[Down2]──→[Down3]──→[Bottleneck]
                                              │
出力 ←── [Up1] ←──[Up2] ←──[Up3] ←──────────┘
          ↑          ↑          ↑
          └──Skip────┘──Skip────┘
※ 各ブロック内に空間注意 + 時間注意

DiT ベース (Sora 等):

入力動画 ──→ [パッチ化] ──→ [位置エンコーディング]
                                    │
                        ┌───────────┘
                        v
          ┌──── DiT Block 1 ────┐
          │  AdaLN → Self-Attn  │
          │  AdaLN → FFN        │
          └─────────┬───────────┘
                    v
          ┌──── DiT Block 2 ────┐
          │  (同様)             │
          └─────────┬───────────┘
                    v
                   ...
                    v
          ┌──── DiT Block N ────┐
          │  (同様)             │
          └─────────┬───────────┘
                    v
          [逆パッチ化] ──→ 出力動画

利点:
- スケーリング則に従いやすい
- Skip connection 不要でシンプル
- 可変解像度/可変長に対応可能
```

---

## 2. 主要プラットフォーム

### コード例2: Runway Gen-3 Alpha API

```python
"""
Runway ML Gen-3 Alpha API を使用した動画生成
"""
import requests
import time

class RunwayClient:
    BASE_URL = "https://api.runwayml.com/v1"

    def __init__(self, api_key):
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def generate_video(self, prompt, duration=4,
                       aspect_ratio="16:9", style=None):
        """
        テキストから動画を生成

        duration: 4 or 10 (秒)
        aspect_ratio: "16:9", "9:16", "1:1"
        """
        payload = {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
        }
        if style:
            payload["style"] = style

        resp = requests.post(
            f"{self.BASE_URL}/generations",
            headers=self.headers,
            json=payload,
        )
        task_id = resp.json()["id"]
        return self._poll_result(task_id)

    def image_to_video(self, image_url, prompt,
                       duration=4, motion_amount=5):
        """
        画像から動画を生成 (Image-to-Video)

        motion_amount: 1-10 (動きの量)
        """
        payload = {
            "image_url": image_url,
            "prompt": prompt,
            "duration": duration,
            "motion_amount": motion_amount,
        }

        resp = requests.post(
            f"{self.BASE_URL}/generations/image-to-video",
            headers=self.headers,
            json=payload,
        )
        task_id = resp.json()["id"]
        return self._poll_result(task_id)

    def _poll_result(self, task_id, timeout=300):
        """結果をポーリングで取得"""
        for _ in range(timeout // 5):
            resp = requests.get(
                f"{self.BASE_URL}/generations/{task_id}",
                headers=self.headers,
            )
            status = resp.json()["status"]
            if status == "completed":
                return resp.json()["output"]["video_url"]
            elif status == "failed":
                raise Exception(f"生成失敗: {resp.json()}")
            time.sleep(5)
        raise TimeoutError("生成タイムアウト")

# 使用例
client = RunwayClient("your_api_key")

# テキストから動画
video_url = client.generate_video(
    prompt="Aerial view of cherry blossom trees along a river "
           "in Tokyo, petals floating in the wind, golden hour",
    duration=4,
    aspect_ratio="16:9",
)

# 画像から動画
video_url = client.image_to_video(
    image_url="https://example.com/landscape.jpg",
    prompt="Gentle camera zoom out, clouds moving slowly",
    motion_amount=3,
)
```

### コード例3: Pika Labs による動画生成

```python
"""
Pika API (概念的なインターフェース)
公式APIは順次公開。Discordボット経由の利用が主流。
"""

PIKA_FEATURES = {
    "text_to_video": {
        "説明": "テキストプロンプトから3秒の動画を生成",
        "構文": "/create prompt: [テキスト]",
        "例": "/create prompt: a cat walking on a rainbow bridge, "
              "anime style -ar 16:9 -motion 2",
        "パラメータ": {
            "-ar": "アスペクト比 (16:9, 9:16, 1:1)",
            "-motion": "動きの量 (1-4)",
            "-gs": "ガイダンススケール (8-24)",
            "-seed": "再現性のためのシード値",
            "-fps": "フレームレート (8, 24)",
        },
    },
    "image_to_video": {
        "説明": "静止画にモーションを付与",
        "構文": "/animate [画像添付] prompt: [テキスト]",
        "強み": "入力画像の忠実度が高い",
    },
    "video_to_video": {
        "説明": "既存動画のスタイル変換",
        "構文": "/modify [動画添付] prompt: [テキスト]",
        "用途": "実写→アニメ変換、色調変更",
    },
    "lip_sync": {
        "説明": "キャラクターの口を音声に同期",
        "構文": "/lip-sync [動画] [音声]",
    },
}

# プロンプト設計のベストプラクティス
VIDEO_PROMPT_TIPS = """
動画プロンプトの書き方:

1. カメラワークを明示する:
   - "slow dolly forward" (ドリー前進)
   - "aerial flyover" (空撮)
   - "static shot" (固定ショット)
   - "tracking shot following..." (追跡ショット)

2. 動きを具体的に記述する:
   - "wind blowing through hair" (髪が風になびく)
   - "waves crashing on shore" (波が打ち寄せる)
   - "person slowly turning around" (人がゆっくり振り返る)

3. 時間的変化を指定する:
   - "transitioning from day to night" (昼→夜)
   - "flower blooming in timelapse" (花がタイムラプスで咲く)

4. 避けるべきこと:
   - 複雑な人物動作 (まだ苦手)
   - テキストの表示 (歪みやすい)
   - 物理的に不可能な動き
"""
```

### 2.1 Stable Video Diffusion (SVD) — オープンソース動画生成

```python
"""
Stable Video Diffusion: オープンソースの Image-to-Video モデル

特徴:
- Stability AI による公開モデル
- Image-to-Video に特化
- ローカル環境で実行可能
- ComfyUI / A1111 でも使用可能
"""
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import torch

def generate_video_svd(
    image_path,
    output_path="output.mp4",
    num_frames=25,
    fps=7,
    motion_bucket_id=127,
    noise_aug_strength=0.02,
    decode_chunk_size=8,
):
    """
    Stable Video Diffusion による動画生成

    Parameters:
        image_path: 入力画像パス
        num_frames: 生成フレーム数 (14 or 25)
        fps: 出力フレームレート
        motion_bucket_id: 動きの量 (0-255, 高い=動き大)
        noise_aug_strength: 入力画像へのノイズ (0-1)
        decode_chunk_size: VAEデコードのチャンクサイズ
    """
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    # メモリ最適化
    pipe.enable_model_cpu_offload()
    pipe.unet.enable_forward_chunking()

    # 入力画像の読み込みとリサイズ
    image = load_image(image_path)
    image = image.resize((1024, 576))  # SVD の推奨入力サイズ

    # 動画生成
    generator = torch.manual_seed(42)
    frames = pipe(
        image,
        num_frames=num_frames,
        decode_chunk_size=decode_chunk_size,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        generator=generator,
    ).frames[0]

    # 動画として保存
    export_to_video(frames, output_path, fps=fps)
    print(f"動画生成完了: {output_path}")
    print(f"  フレーム数: {num_frames}")
    print(f"  FPS: {fps}")
    print(f"  長さ: {num_frames / fps:.1f}秒")

    return output_path


def batch_image_to_video(
    image_dir,
    output_dir,
    motion_bucket_id=127,
    num_frames=25,
):
    """ディレクトリ内の全画像を動画化"""
    from pathlib import Path

    input_path = Path(image_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.enable_model_cpu_offload()

    for img_file in sorted(input_path.glob("*.{png,jpg,jpeg}")):
        image = load_image(str(img_file)).resize((1024, 576))
        frames = pipe(
            image,
            num_frames=num_frames,
            motion_bucket_id=motion_bucket_id,
        ).frames[0]

        output_file = out_path / f"{img_file.stem}.mp4"
        export_to_video(frames, str(output_file), fps=7)
        print(f"完了: {img_file.name} → {output_file.name}")
```

### 2.2 CogVideoX — オープンソース Text-to-Video

```python
"""
CogVideoX: テキストから動画を生成するオープンソースモデル

特徴:
- Text-to-Video 対応 (SVD は Image-to-Video のみ)
- 6秒の動画生成
- 720x480 解像度
- 5B / 2B パラメータのバリエーション
"""
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import torch

def generate_video_cogvideo(
    prompt,
    output_path="cogvideo_output.mp4",
    num_frames=49,
    guidance_scale=6.0,
    num_inference_steps=50,
):
    """
    CogVideoX によるテキストから動画生成

    Parameters:
        prompt: テキストプロンプト
        num_frames: フレーム数 (49 = 約6秒@8fps)
        guidance_scale: テキスト一致度
        num_inference_steps: 推論ステップ数
    """
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-5b",
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]

    export_to_video(video, output_path, fps=8)
    return output_path

# 使用例
generate_video_cogvideo(
    prompt="A golden retriever running on a beach at sunset, "
           "waves splashing, cinematic lighting, slow motion",
)
```

### ASCII図解2: 主要動画生成プラットフォームの機能マップ

```
┌──────── 動画生成プラットフォーム比較 ─────────────────┐
│                                                      │
│     Sora          Runway Gen-3     Pika 2.0         │
│  ┌─────────┐    ┌─────────────┐  ┌──────────┐      │
│  │最長60秒  │    │最長10秒     │  │最長10秒   │      │
│  │1080p     │    │4K対応       │  │1080p      │      │
│  │DiT base  │    │独自モデル   │  │独自モデル  │      │
│  │          │    │             │  │           │      │
│  │Text→Video│    │Text→Video   │  │Text→Video │      │
│  │Img→Video │    │Img→Video    │  │Img→Video  │      │
│  │          │    │Motion Brush │  │Vid→Video  │      │
│  │          │    │Camera Control│ │Lip Sync   │      │
│  └─────────┘    └─────────────┘  └──────────┘      │
│                                                      │
│     Kling         Luma Dream      Stable Video       │
│  ┌─────────┐    ┌─────────────┐  ┌──────────┐      │
│  │最長120秒 │    │最長5秒      │  │最長4秒    │      │
│  │1080p     │    │1080p        │  │1024x576   │      │
│  │中国発    │    │Dream Machine│  │SVD base   │      │
│  │Motion    │    │             │  │           │      │
│  │Transfer  │    │Text→Video   │  │Img→Video  │      │
│  └─────────┘    └─────────────┘  └──────────┘      │
│                                                      │
│  ┌─────────── オープンソース ─────────────────┐      │
│  │  CogVideoX    Open-Sora    AnimateDiff    │      │
│  │  Text→Video   Text→Video   Img→Video      │      │
│  │  5B/2B params  研究用       SD Extension   │      │
│  │  ローカル実行可  ローカル実行可  ComfyUI対応  │      │
│  └───────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────┘
```

---

## 3. プロダクション向けワークフロー

### コード例4: 動画生成の品質管理パイプライン

```python
import subprocess
from pathlib import Path
import json

class VideoProductionPipeline:
    """動画生成の品質管理ワークフロー"""

    def __init__(self, output_dir="./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_candidates(self, prompt, num_candidates=5):
        """複数の候補動画を生成"""
        candidates = []
        for i in range(num_candidates):
            # 各サービスで生成 (シードを変えて)
            video_path = self._generate_single(prompt, seed=i * 1000)
            candidates.append(video_path)
        return candidates

    def evaluate_quality(self, video_path):
        """動画品質のチェック"""
        checks = {
            "temporal_consistency": self._check_temporal_consistency(video_path),
            "artifact_score": self._check_artifacts(video_path),
            "motion_quality": self._check_motion_quality(video_path),
            "prompt_alignment": self._check_prompt_alignment(video_path),
        }
        overall = sum(checks.values()) / len(checks)
        return {"checks": checks, "overall": overall}

    def post_process(self, video_path, output_path):
        """FFmpegによる後処理"""
        # フレーム補間 (24fps → 60fps)
        subprocess.run([
            "ffmpeg", "-i", str(video_path),
            "-vf", "minterpolate=fps=60:mi_mode=mci",
            "-c:v", "libx264", "-crf", "18",
            str(output_path),
        ])

    def upscale_video(self, video_path, output_path, scale=2):
        """動画のAIアップスケーリング"""
        # フレーム抽出
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        subprocess.run([
            "ffmpeg", "-i", str(video_path),
            str(frames_dir / "frame_%04d.png"),
        ])

        # 各フレームをアップスケール (Real-ESRGAN等)
        for frame in sorted(frames_dir.glob("*.png")):
            self._upscale_frame(frame, scale)

        # フレームを動画に再結合
        subprocess.run([
            "ffmpeg", "-framerate", "24",
            "-i", str(frames_dir / "frame_%04d_upscaled.png"),
            "-c:v", "libx264", "-crf", "18",
            str(output_path),
        ])

    def _check_temporal_consistency(self, video_path):
        """時間的一貫性のチェック"""
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(video_path))
        prev_frame = None
        diffs = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if prev_frame is not None:
                diff = np.mean(np.abs(
                    frame.astype(float) - prev_frame.astype(float)
                ))
                diffs.append(diff)
            prev_frame = frame

        cap.release()

        if not diffs:
            return 0.0

        # 急激な変化がないかチェック
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        outliers = sum(1 for d in diffs if abs(d - mean_diff) > 2 * std_diff)
        consistency = 1.0 - (outliers / len(diffs))

        return max(0.0, min(1.0, consistency))

    def _check_motion_quality(self, video_path):
        """動きの品質チェック (オプティカルフロー解析)"""
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(video_path))
        prev_gray = None
        flow_magnitudes = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                flow_magnitudes.append(np.mean(mag))
            prev_gray = gray

        cap.release()

        if not flow_magnitudes:
            return 0.0

        # 動きの滑らかさを評価
        smoothness = 1.0 - np.std(flow_magnitudes) / (np.mean(flow_magnitudes) + 1e-8)
        return max(0.0, min(1.0, smoothness))
```

### コード例5: 動画生成プロンプトの体系的設計

```python
class VideoPromptBuilder:
    """動画生成用プロンプトの構築"""

    CAMERA_MOVES = {
        "固定": "static shot, locked camera",
        "パン左": "slow pan left, horizontal camera movement",
        "パン右": "slow pan right, horizontal camera movement",
        "ティルトアップ": "tilt up, camera looking upward",
        "ティルトダウン": "tilt down, camera looking downward",
        "ドリーイン": "slow dolly forward, approaching subject",
        "ドリーアウト": "slow dolly backward, pulling away",
        "空撮": "aerial drone shot, bird's eye view, flying over",
        "軌道": "orbiting shot, rotating around subject",
        "ステディカム": "steadicam tracking shot, smooth movement",
        "ズームイン": "slow zoom in, focusing on subject",
        "ズームアウト": "slow zoom out, revealing environment",
        "クレーン": "crane shot, moving upward and forward",
        "手持ち": "handheld camera, slight shake, documentary style",
    }

    MOTION_TYPES = {
        "自然": "natural movement, wind, water flow, clouds",
        "人物": "person walking, human motion, gesture",
        "動物": "animal movement, natural behavior",
        "抽象": "abstract motion, particle flow, morphing",
        "タイムラプス": "timelapse, accelerated time, day to night",
        "スローモーション": "slow motion, 120fps, dramatic timing",
        "静的": "minimal movement, subtle animation, cinemagraph",
    }

    LIGHTING_PRESETS = {
        "ゴールデンアワー": "golden hour, warm sunlight, long shadows",
        "ブルーアワー": "blue hour, twilight, cool tones",
        "スタジオ": "studio lighting, three-point, professional",
        "ネオン": "neon lights, cyberpunk, colorful reflections",
        "月明かり": "moonlight, nighttime, ethereal glow",
        "逆光": "backlit, silhouette, rim lighting",
        "曇天": "overcast, diffused light, no harsh shadows",
    }

    STYLE_PRESETS = {
        "シネマティック": "cinematic, film grain, anamorphic lens, 2.39:1",
        "ドキュメンタリー": "documentary style, natural, raw footage",
        "アニメ": "anime style, cel shading, vibrant colors",
        "ピクサー風": "Pixar style, 3D animation, colorful, cute",
        "ノワール": "film noir, black and white, high contrast",
        "ヴィンテージ": "vintage, 8mm film, desaturated, vignette",
        "ハイパーリアル": "hyperrealistic, photorealistic, 8K, HDR",
        "ミニチュア": "tilt-shift, miniature effect, shallow depth of field",
    }

    def __init__(self):
        self.parts = {}

    def set_scene(self, description):
        self.parts["scene"] = description
        return self

    def set_camera(self, move_type):
        self.parts["camera"] = self.CAMERA_MOVES.get(move_type, move_type)
        return self

    def set_motion(self, motion_type, details=""):
        base = self.MOTION_TYPES.get(motion_type, motion_type)
        self.parts["motion"] = f"{base}, {details}" if details else base
        return self

    def set_lighting(self, lighting_type):
        self.parts["lighting"] = self.LIGHTING_PRESETS.get(
            lighting_type, lighting_type
        )
        return self

    def set_style(self, style):
        self.parts["style"] = self.STYLE_PRESETS.get(style, style)
        return self

    def set_negative(self, negative):
        self.parts["negative"] = f"NOT: {negative}"
        return self

    def build(self):
        # negative は分離
        positive_parts = {
            k: v for k, v in self.parts.items() if k != "negative"
        }
        prompt = ", ".join(positive_parts.values())

        if "negative" in self.parts:
            return {
                "prompt": prompt,
                "negative_prompt": self.parts["negative"].replace("NOT: ", ""),
            }
        return {"prompt": prompt}

# 使用例
result = (
    VideoPromptBuilder()
    .set_scene("Ancient temple in Kyoto surrounded by maple trees")
    .set_camera("ドリーイン")
    .set_motion("自然", "leaves gently falling, mist rising")
    .set_lighting("ゴールデンアワー")
    .set_style("シネマティック")
    .set_negative("blurry, low quality, text, watermark")
    .build()
)
print(result["prompt"])
```

### ASCII図解3: プロダクション動画生成ワークフロー

```
┌─ Phase 1: 企画 ──┐  ┌─ Phase 2: 生成 ──┐  ┌─ Phase 3: 後処理 ┐
│                   │  │                   │  │                   │
│ ストーリーボード  │  │ プロンプト設計    │  │ フレーム補間      │
│       │           │  │       │           │  │ (24fps→60fps)     │
│       v           │  │       v           │  │       │           │
│ シーン分割       │→│ 複数候補生成      │→│ アップスケール    │
│ (3-5秒単位)      │  │ (5候補/シーン)    │  │ (1080p→4K)       │
│       │           │  │       │           │  │       │           │
│       v           │  │       v           │  │       v           │
│ カメラワーク設計  │  │ 品質評価・選択    │  │ 色調補正          │
│                   │  │                   │  │       │           │
└───────────────────┘  └───────────────────┘  │       v           │
                                               │ 音楽・SE追加      │
                                               │       │           │
                                               │       v           │
                                               │ 最終書き出し      │
                                               └───────────────────┘

品質チェックポイント:
  ✓ 時間的一貫性 (フリッカーなし)
  ✓ 物理的整合性 (不自然な動きなし)
  ✓ プロンプト忠実度
  ✓ アーティファクトなし
  ✓ シーン間の繋がり
```

---

## 4. フレーム補間と動画品質向上

### 4.1 RIFE によるフレーム補間

```python
"""
RIFE (Real-Time Intermediate Flow Estimation)
動画のフレームレートを向上させるAIベースのフレーム補間

用途:
- 24fps → 60fps (滑らかな再生)
- 低FPSの生成動画を高FPSに変換
- スローモーション効果
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess

class RIFEInterpolator:
    """RIFE によるフレーム補間パイプライン"""

    def __init__(self, model_path="weights/rife-v4.6.pth", device="cuda"):
        self.device = device
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        """RIFE モデルのロード"""
        # 実際にはRIFEのモデルクラスをインポート
        import torch
        from rife.RIFE import Model

        model = Model()
        model.load_model(model_path)
        model.eval()
        return model

    def interpolate_frames(self, frame1, frame2, num_mid_frames=1):
        """
        2フレーム間に中間フレームを生成

        Parameters:
            frame1, frame2: numpy array [H, W, C]
            num_mid_frames: 生成する中間フレームの数

        Returns:
            list of numpy arrays (中間フレーム)
        """
        import torch

        # numpy → tensor
        def to_tensor(img):
            t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
            return t.unsqueeze(0).to(self.device)

        t1 = to_tensor(frame1)
        t2 = to_tensor(frame2)

        mid_frames = []
        for i in range(1, num_mid_frames + 1):
            timestep = i / (num_mid_frames + 1)
            with torch.no_grad():
                mid = self.model.inference(t1, t2, timestep)
            # tensor → numpy
            mid_np = (mid.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            mid_frames.append(mid_np)

        return mid_frames

    def interpolate_video(self, input_path, output_path,
                          target_fps=60, codec="libx264", crf=18):
        """
        動画全体のフレーム補間

        Parameters:
            input_path: 入力動画
            output_path: 出力動画
            target_fps: 目標FPS
        """
        cap = cv2.VideoCapture(str(input_path))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        multiplier = int(target_fps / src_fps)
        num_mid = multiplier - 1

        print(f"入力: {src_fps}fps → 出力: {target_fps}fps "
              f"(x{multiplier}, 中間{num_mid}フレーム)")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path), fourcc, target_fps, (w, h)
        )

        prev_frame = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is not None:
                # 中間フレームを生成
                mid_frames = self.interpolate_frames(
                    prev_frame, frame, num_mid
                )
                # 前フレーム → 中間フレーム → 現フレーム の順で書き出し
                writer.write(prev_frame)
                for mf in mid_frames:
                    writer.write(mf)
            else:
                writer.write(frame)

            prev_frame = frame
            frame_idx += 1

            if frame_idx % 50 == 0:
                print(f"  [{frame_idx}/{total_frames}]")

        # 最後のフレーム
        if prev_frame is not None:
            writer.write(prev_frame)

        cap.release()
        writer.release()

        print(f"完了: {output_path}")
        return output_path
```

### 4.2 動画の色調統一（シーン間の一貫性）

```python
class VideoColorHarmonizer:
    """複数ショットの色調を統一するパイプライン"""

    def __init__(self):
        pass

    def extract_color_stats(self, video_path, num_samples=10):
        """動画の色統計情報を抽出"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = max(1, total_frames // num_samples)

        stats = {
            "mean_lab": [],
            "std_lab": [],
            "histogram": [],
        }

        for i in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
            stats["mean_lab"].append(lab.mean(axis=(0, 1)))
            stats["std_lab"].append(lab.std(axis=(0, 1)))

        cap.release()

        return {
            "mean_lab": np.mean(stats["mean_lab"], axis=0),
            "std_lab": np.mean(stats["std_lab"], axis=0),
        }

    def harmonize_videos(self, video_paths, reference_idx=0,
                         output_dir="harmonized"):
        """
        複数動画の色調をリファレンスに合わせる

        Parameters:
            video_paths: 動画パスのリスト
            reference_idx: リファレンスとする動画のインデックス
            output_dir: 出力ディレクトリ
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # リファレンスの色統計
        ref_stats = self.extract_color_stats(video_paths[reference_idx])

        for i, vp in enumerate(video_paths):
            if i == reference_idx:
                # リファレンスはそのままコピー
                import shutil
                shutil.copy(vp, out_dir / Path(vp).name)
                continue

            # 色補正を適用
            src_stats = self.extract_color_stats(vp)
            self._apply_color_transfer(
                vp,
                out_dir / Path(vp).name,
                src_stats,
                ref_stats,
            )

    def _apply_color_transfer(self, input_path, output_path,
                               src_stats, ref_stats):
        """フレームごとに色転写を適用"""
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

            for ch in range(3):
                lab[:, :, ch] = (
                    (lab[:, :, ch] - src_stats["mean_lab"][ch])
                    * (ref_stats["std_lab"][ch] / (src_stats["std_lab"][ch] + 1e-8))
                    + ref_stats["mean_lab"][ch]
                )

            lab = np.clip(lab, 0, 255).astype(np.uint8)
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            writer.write(result)

        cap.release()
        writer.release()
```

---

## 5. 動画生成の自動化と統合

### 5.1 マルチプラットフォーム動画生成オーケストレーター

```python
import asyncio
import aiohttp
from enum import Enum

class Platform(Enum):
    RUNWAY = "runway"
    PIKA = "pika"
    SORA = "sora"
    SVD_LOCAL = "svd_local"
    COGVIDEO_LOCAL = "cogvideo_local"

class VideoGenerationOrchestrator:
    """
    複数プラットフォームを統合する動画生成オーケストレーター

    機能:
    - プラットフォーム自動選択 (要件に基づく)
    - 並列生成
    - コスト管理
    - 品質評価と自動リトライ
    """

    def __init__(self, api_keys=None):
        self.api_keys = api_keys or {}
        self.cost_tracker = CostTracker()

    def recommend_platform(self, requirements):
        """要件に基づくプラットフォーム推薦"""
        duration = requirements.get("duration", 4)
        quality = requirements.get("quality", "high")
        budget = requirements.get("budget_per_clip", 1.0)
        input_type = requirements.get("input", "text")
        local_gpu = requirements.get("local_gpu", False)

        recommendations = []

        if input_type == "text":
            if quality == "highest" and budget >= 0.50:
                recommendations.append(("Sora", "最高品質、長尺対応"))
            if quality in ("high", "highest") and budget >= 0.10:
                recommendations.append(("Runway Gen-3", "高品質、カメラ制御"))
            if budget < 0.10:
                recommendations.append(("Pika", "コスパ最良"))
            if local_gpu:
                recommendations.append(("CogVideoX", "ローカル実行、コスト0"))

        elif input_type == "image":
            if quality == "highest":
                recommendations.append(("Runway Gen-3", "I2V最高品質"))
            if local_gpu:
                recommendations.append(("SVD", "ローカルI2V、コスト0"))

        if duration > 10:
            recommendations.insert(0, ("Kling", f"最長{duration}秒対応"))

        return recommendations

    async def generate_multi_platform(self, prompt, platforms=None,
                                       num_per_platform=2):
        """複数プラットフォームで並列生成し、最良の結果を選択"""
        if platforms is None:
            platforms = [Platform.RUNWAY, Platform.PIKA]

        tasks = []
        for platform in platforms:
            for i in range(num_per_platform):
                tasks.append(self._generate_one(
                    platform, prompt, seed=i * 42
                ))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # エラーを除外
        valid_results = [
            r for r in results if not isinstance(r, Exception)
        ]

        # 品質スコアで並べ替え
        scored_results = []
        for result in valid_results:
            quality = self._evaluate_quality(result["video_path"])
            scored_results.append({
                **result,
                "quality_score": quality,
            })

        scored_results.sort(key=lambda x: x["quality_score"], reverse=True)

        return scored_results

    async def _generate_one(self, platform, prompt, seed=0):
        """1つのプラットフォームで生成"""
        if platform == Platform.RUNWAY:
            client = RunwayClient(self.api_keys.get("runway"))
            url = client.generate_video(prompt)
            cost = 0.10
        elif platform == Platform.PIKA:
            # Pika API実装
            url = None
            cost = 0.05
        else:
            url = None
            cost = 0.0

        self.cost_tracker.add(platform.value, cost)

        return {
            "platform": platform.value,
            "video_url": url,
            "cost": cost,
            "seed": seed,
        }


class CostTracker:
    """動画生成コストの追跡"""

    def __init__(self):
        self.records = []

    def add(self, platform, cost, metadata=None):
        self.records.append({
            "platform": platform,
            "cost": cost,
            "metadata": metadata or {},
        })

    def get_total(self):
        return sum(r["cost"] for r in self.records)

    def get_by_platform(self):
        from collections import defaultdict
        result = defaultdict(float)
        for r in self.records:
            result[r["platform"]] += r["cost"]
        return dict(result)

    def report(self):
        total = self.get_total()
        by_platform = self.get_by_platform()
        return {
            "total_cost": round(total, 2),
            "by_platform": by_platform,
            "num_generations": len(self.records),
            "avg_cost_per_generation": round(
                total / max(len(self.records), 1), 3
            ),
        }
```

### 5.2 ストーリーボードからの自動動画生成

```python
class StoryboardToVideo:
    """ストーリーボードから動画を自動生成するパイプライン"""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.prompt_builder = VideoPromptBuilder()

    def parse_storyboard(self, storyboard_json):
        """
        ストーリーボードJSON を解析

        Expected format:
        {
            "title": "プロダクト紹介動画",
            "target_duration": 30,
            "shots": [
                {
                    "id": "shot_01",
                    "description": "高層ビルの屋上から都市を俯瞰",
                    "camera": "空撮",
                    "motion": "自然",
                    "duration": 5,
                    "style": "シネマティック",
                    "transition": "fade"
                },
                ...
            ]
        }
        """
        with open(storyboard_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        shots = []
        for shot_data in data["shots"]:
            prompt = (
                self.prompt_builder
                .set_scene(shot_data["description"])
                .set_camera(shot_data.get("camera", "固定"))
                .set_motion(shot_data.get("motion", "自然"))
                .set_style(shot_data.get("style", "シネマティック"))
                .build()
            )
            shots.append({
                "id": shot_data["id"],
                "prompt": prompt,
                "duration": shot_data.get("duration", 4),
                "transition": shot_data.get("transition", "cut"),
            })

        return {
            "title": data["title"],
            "total_duration": sum(s["duration"] for s in shots),
            "shots": shots,
        }

    def generate_all_shots(self, storyboard, num_candidates=3):
        """全ショットを生成"""
        results = {}
        for shot in storyboard["shots"]:
            print(f"Generating: {shot['id']}")
            candidates = []
            for i in range(num_candidates):
                video = self.orchestrator._generate_one(
                    Platform.RUNWAY,
                    shot["prompt"]["prompt"],
                    seed=i * 100,
                )
                candidates.append(video)
            results[shot["id"]] = candidates

        return results

    def assemble_final_video(self, shot_videos, transitions,
                              output_path, audio_path=None):
        """
        ショットを結合して最終動画を作成

        shot_videos: {"shot_01": "path/to/video.mp4", ...}
        transitions: {"shot_01→shot_02": "fade", ...}
        """
        # FFmpeg の concat フィルタで結合
        filter_complex = []
        inputs = []

        for i, (shot_id, video_path) in enumerate(shot_videos.items()):
            inputs.extend(["-i", str(video_path)])

        # トランジションの適用
        cmd = ["ffmpeg"]
        for inp in inputs:
            cmd.append(inp)

        # concat フィルタ
        n = len(shot_videos)
        filter_str = "".join(
            f"[{i}:v:0]" for i in range(n)
        )
        filter_str += f"concat=n={n}:v=1:a=0[outv]"

        cmd.extend([
            "-filter_complex", filter_str,
            "-map", "[outv]",
            "-c:v", "libx264",
            "-crf", "18",
        ])

        # 音声の追加
        if audio_path:
            cmd.extend([
                "-i", str(audio_path),
                "-c:a", "aac",
                "-shortest",
            ])

        cmd.append(str(output_path))
        subprocess.run(cmd, check=True)

        return output_path
```

---

## 6. 比較表

### 比較表1: 動画生成プラットフォーム詳細比較

| 項目 | Sora | Runway Gen-3 | Pika 2.0 | Kling | Luma |
|------|------|-------------|---------|-------|------|
| **最大長** | 60秒 | 10秒 | 10秒 | 120秒 | 5秒 |
| **解像度** | 1080p | 4K | 1080p | 1080p | 1080p |
| **Text→Video** | あり | あり | あり | あり | あり |
| **Img→Video** | あり | あり | あり | あり | あり |
| **Vid→Video** | あり | あり | あり | あり | なし |
| **カメラ制御** | 自動 | Motion Brush | パラメータ | あり | 限定的 |
| **品質** | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| **速度** | 数分 | 30秒~2分 | 30秒~1分 | 1~5分 | 30秒~1分 |
| **価格** | $20~/月 | $12~/月 | $8~/月 | 無料~有料 | $24~/月 |

### 比較表2: ユースケース別推奨プラットフォーム

| ユースケース | 推奨 | 理由 |
|-------------|------|------|
| **SNS短尺動画** | Pika / Runway | コスパと手軽さ |
| **広告動画プロトタイプ** | Sora / Runway | 高品質、カメラ制御 |
| **ミュージックビデオ** | Runway Gen-3 | Motion Brush、スタイル一貫性 |
| **プレゼン素材** | Pika | 低コスト、十分な品質 |
| **映画制作 (プリビズ)** | Sora | 最長尺、最高品質 |
| **ECサイト商品動画** | Kling | 長尺対応、コスパ |
| **個人/研究用** | CogVideoX / SVD | 無料、ローカル実行 |
| **アニメーション** | Pika + スタイル指定 | アニメスタイル変換 |

### 比較表3: オープンソース動画生成モデル

| モデル | タイプ | 解像度 | 長さ | VRAM | 特徴 |
|--------|-------|--------|------|------|------|
| **SVD** | Img→Video | 1024x576 | 4秒 | 12GB+ | 安定、高品質 |
| **SVD-XT** | Img→Video | 1024x576 | 4秒 | 16GB+ | 25フレーム対応 |
| **CogVideoX-2B** | Text→Video | 720x480 | 6秒 | 16GB+ | 軽量版 |
| **CogVideoX-5B** | Text→Video | 720x480 | 6秒 | 24GB+ | 高品質版 |
| **AnimateDiff** | Img→Video | 512x512 | 2秒 | 8GB | SD Extension |
| **Open-Sora** | Text→Video | 各種 | 可変 | 24GB+ | 研究目的 |

---

## 7. アンチパターン

### アンチパターン1: 長い動画を一度に生成しようとする

```
[問題]
30秒のCM動画を1つのプロンプトで生成しようとする。

[なぜ問題か]
- 現行モデルは4-10秒が実用的な上限
- 長い生成は品質が急激に低下
- 意図した展開をコントロールできない
- 生成時間とコストが非線形に増加

[正しいアプローチ]
- 3-5秒のショットに分割して個別生成
- ストーリーボードで各ショットを設計
- 後処理で繋ぎ合わせ、トランジションを追加
- 一貫性のためにスタイルリファレンスを共有
```

### アンチパターン2: テキストや複雑な人物動作を期待する

```
[問題]
「人がテニスをプレイし、スコアボードが表示される」
のような複雑なシーンを生成しようとする。

[なぜ問題か]
- テキスト/数字の表示は現行モデルの弱点
- 複雑な人体動作 (スポーツ等) は不自然になりやすい
- 複数人物のインタラクションは一貫性が低い

[正しいアプローチ]
- テキストは後処理で合成
- 複雑な動作は実写+AIスタイル変換を検討
- 自然現象、風景、抽象的動きを中心に活用
- 人物は固定or単純な動作に限定
```

### アンチパターン3: 生成動画をそのまま納品する

```
[問題]
AI生成動画の出力をそのまま最終成果物として使用する。

[なぜ問題か]
- フレームレートが低い (多くのモデルは8-24fps)
- 解像度が不十分 (720p-1080p)
- 色調が不統一 (特にマルチショット)
- 微細なアーティファクトが残る

[正しいアプローチ]
- フレーム補間 (RIFE) で滑らかに
- 超解像 (Real-ESRGAN) で高解像度化
- カラーグレーディングで統一感を出す
- 人の目で品質チェック後に書き出し
```

### アンチパターン4: プロンプトに文学的な表現を使う

```
[問題]
「時の流れが儚く、美しい桜が静寂の中で舞い散る」
のような抽象的・文学的なプロンプトを使用する。

[なぜ問題か]
- AIは具体的な視覚表現に反応する
- 抽象概念 (「儚い」「静寂」) は映像に直結しない
- 結果が意図と乖離しやすい

[正しいアプローチ]
- 視覚的に具体的な記述を使う
  × 「儚く美しい桜」
  ○ 「pink cherry blossom petals slowly falling,
      single tree, soft focus background,
      gentle breeze, golden hour lighting」
- カメラワークと動きを明示する
- スタイル修飾語を付加する
```

---

## FAQ

### Q1: Sora はいつから使える? 代替手段は?

**A:** Sora は2024年12月に一般公開されました:

- **ChatGPT Plus/Pro ユーザー:** Sora にアクセス可能
- **代替手段:** Runway Gen-3 Alpha が最も近い品質。Pika がコスパ最良
- **中国発:** Kling (快影) が120秒対応で注目
- **オープンソース:** CogVideo、Open-Sora が研究用に利用可能

### Q2: 生成動画の解像度や長さの制限を突破するには?

**A:**

- **長さ:** ショット分割 + 動画編集ソフトで結合。フレーム補間で滑らかに
- **解像度:** Real-ESRGAN のフレーム単位適用で 4K 化
- **FPS:** RIFE (Real-Time Intermediate Flow Estimation) でフレーム補間
- **注意:** 後処理の重ね掛けは品質低下のリスクあり。最小限に抑える

### Q3: 動画生成にかかるコストの目安は?

**A:**

- **Runway Gen-3:** 約$0.05~0.10/秒 (生成1回あたり)
- **Pika:** 月$8 で150クレジット (1動画=約10クレジット)
- **Sora:** ChatGPT Plus ($20/月) に含まれる
- **30秒CM制作例:** 6-10ショット x 5候補 x $0.10 = $3-5 (生成コスト)
- **総コスト:** 生成 + 後処理 + 人件費 を含めて見積もる

### Q4: ローカル環境で動画生成を行うには?

**A:** 以下のモデルがローカル実行に対応しています:

| モデル | 最低VRAM | インストール方法 |
|--------|---------|----------------|
| **SVD** | 12GB | `pip install diffusers` |
| **CogVideoX-2B** | 16GB | `pip install diffusers` |
| **AnimateDiff** | 8GB | ComfyUI Extension |
| **Open-Sora** | 24GB | GitHub clone + install |

推奨環境:
- GPU: NVIDIA RTX 4090 (24GB) 以上
- RAM: 32GB 以上
- ストレージ: SSD 100GB+ (モデルウェイト用)

### Q5: 複数ショットの一貫性を保つには?

**A:** シーン間の視覚的一貫性は動画生成の最大の課題の一つです。以下のアプローチが有効です:

1. **Image-to-Video を活用:** 統一したスタイルの画像を先に生成し、それを基に動画化
2. **スタイルリファレンス:** 共通のスタイルプロンプト接尾辞を使用
3. **カラーグレーディング:** 後処理で色調を統一 (前述の VideoColorHarmonizer)
4. **シード管理:** 同じシードを使用し、類似した雰囲気を維持
5. **LoRA 微調整:** 特定のスタイル/キャラクターに特化したモデルを作成

### Q6: 動画生成の著作権と商用利用は?

**A:**

| プラットフォーム | 商用利用 | 著作権 | 注意事項 |
|----------------|---------|--------|---------|
| **Sora** | 有料プランで可 | 生成者に帰属 | OpenAI利用規約に従う |
| **Runway** | 有料プランで可 | 生成者に帰属 | C2PA メタデータ付与 |
| **Pika** | 有料プランで可 | 生成者に帰属 | 利用規約確認 |
| **SVD** | 研究用ライセンス | 要確認 | 商用は追加許諾が必要な場合あり |
| **CogVideoX** | Apache 2.0 | 自由 | オープンソース |

---

## まとめ表

| 項目 | 要点 |
|------|------|
| **技術基盤** | 時空間拡散モデル (画像拡散 + 時間軸注意機構) |
| **最新アーキテクチャ** | DiT ベース (Sora 等)。スケーリング則に優れる |
| **プラットフォーム** | Sora (品質最高)、Runway (バランス)、Pika (コスパ) |
| **オープンソース** | SVD (I2V)、CogVideoX (T2V)、AnimateDiff |
| **実用長** | 4-10秒/ショットが現実的。長尺は分割+結合 |
| **プロンプト** | カメラワーク + 動きの記述 + スタイル が三本柱 |
| **後処理** | フレーム補間 (RIFE)、超解像、色調補正が必須 |
| **弱点** | テキスト表示、複雑な人物動作、物理シミュレーション |
| **コスト** | API: $0.05-0.50/クリップ、ローカル: GPU電気代のみ |

---

## 次に読むべきガイド

- [01-video-editing.md](./01-video-editing.md) — AI動画編集ツール
- [02-animation.md](./02-animation.md) — AIアニメーション技術
- [../01-image/00-image-generation.md](../01-image/00-image-generation.md) — 画像生成 (動画の入力素材)

---

## 参考文献

1. Brooks, T. et al. (2024). "Video generation models as world simulators (Sora)." *OpenAI Technical Report*. https://openai.com/research/video-generation-models-as-world-simulators
2. Blattmann, A. et al. (2023). "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets." *arXiv*. https://arxiv.org/abs/2311.15127
3. Hong, W. et al. (2023). "CogVideo: Large-scale Pretraining for Text-to-Video Generation." *ICLR 2023*. https://arxiv.org/abs/2205.15868
4. Singer, U. et al. (2023). "Make-A-Video: Text-to-Video Generation without Text-Video Data." *ICLR 2023*. https://arxiv.org/abs/2209.14792
5. Guo, Y. et al. (2023). "AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning." *ICLR 2024*. https://arxiv.org/abs/2307.04725
6. Peebles, W. & Xie, S. (2023). "Scalable Diffusion Models with Transformers (DiT)." *ICCV 2023*. https://arxiv.org/abs/2212.09748
7. Huang, Z. et al. (2023). "RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation." *ECCV 2022*. https://arxiv.org/abs/2011.06294

# 動画生成 — Sora、Runway、Pika

> AI による動画生成技術の仕組みと主要プラットフォームの使い方を、テキスト-to-ビデオから画像-to-ビデオまで実践的に解説する。

---

## この章で学ぶこと

1. **動画生成モデルの原理** — 時間軸への拡散モデルの拡張、時空間アーキテクチャ
2. **主要プラットフォームの比較と使い分け** — Sora、Runway Gen-3、Pika、Kling の特徴
3. **プロダクション向けワークフロー** — 生成動画の品質管理と後処理パイプライン

---

## 1. 動画生成の技術基盤

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
└──────────────────────────────────────────────────────┘
```

---

## 3. プロダクション向けワークフロー

### コード例4: 動画生成の品質管理パイプライン

```python
import subprocess
from pathlib import Path

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
        "ドリーイン": "slow dolly forward, approaching subject",
        "ドリーアウト": "slow dolly backward, pulling away",
        "空撮": "aerial drone shot, bird's eye view, flying over",
        "軌道": "orbiting shot, rotating around subject",
        "ステディカム": "steadicam tracking shot, smooth movement",
        "ズームイン": "slow zoom in, focusing on subject",
    }

    MOTION_TYPES = {
        "自然": "natural movement, wind, water flow, clouds",
        "人物": "person walking, human motion, gesture",
        "動物": "animal movement, natural behavior",
        "抽象": "abstract motion, particle flow, morphing",
        "タイムラプス": "timelapse, accelerated time, day to night",
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

    def set_style(self, style):
        self.parts["style"] = style
        return self

    def build(self):
        return ", ".join(self.parts.values())

# 使用例
prompt = (
    VideoPromptBuilder()
    .set_scene("Ancient temple in Kyoto surrounded by maple trees")
    .set_camera("ドリーイン")
    .set_motion("自然", "leaves gently falling, mist rising")
    .set_style("cinematic, film grain, anamorphic lens")
    .build()
)
print(prompt)
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

## 4. 比較表

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

---

## 5. アンチパターン

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
- **30秒CM制作例:** 6-10ショット × 5候補 × $0.10 = $3-5 (生成コスト)
- **総コスト:** 生成 + 後処理 + 人件費 を含めて見積もる

---

## まとめ表

| 項目 | 要点 |
|------|------|
| **技術基盤** | 時空間拡散モデル (画像拡散 + 時間軸注意機構) |
| **プラットフォーム** | Sora (品質最高)、Runway (バランス)、Pika (コスパ) |
| **実用長** | 4-10秒/ショットが現実的。長尺は分割+結合 |
| **プロンプト** | カメラワーク + 動きの記述 + スタイル が三本柱 |
| **後処理** | フレーム補間、アップスケール、色調補正が必須 |
| **弱点** | テキスト表示、複雑な人物動作、物理シミュレーション |

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

# アニメーション -- AIアニメーション

> AI技術を活用したアニメーション制作の自動化手法を、画像からの動画生成・キャラクターアニメーション・モーション生成の観点から実践的に解説し、従来数日かかっていた作業を数分に短縮する方法を示す

## この章で学ぶこと

1. **AI アニメーション生成の基本** -- Image-to-Video、Text-to-Video、モーション転写の仕組み
2. **主要ツールとモデル** -- Runway Gen-3、Pika、Stable Video Diffusion、AnimateDiff の比較
3. **実践ワークフロー** -- プロンプト設計、コンシステンシー維持、ループアニメーション制作

---

## 1. AI アニメーション技術の全体像

### 1.1 技術分類

```
AI アニメーション技術マップ

  Image-to-Video (静止画→動画)
  ├── Runway Gen-3 Alpha    --- 高品質な動き生成
  ├── Stable Video Diffusion --- オープンソース
  ├── Pika                   --- 簡単操作、3D変換
  └── Kling                  --- 長尺生成、高品質

  Text-to-Video (テキスト→動画)
  ├── Sora (OpenAI)          --- 超高品質（限定公開）
  ├── Runway Gen-3           --- テキスト→動画
  └── AnimateDiff            --- Stable Diffusion 拡張

  Motion Transfer (モーション転写)
  ├── ControlNet + Temporal  --- ポーズ制御
  ├── DWPose                 --- 人体ポーズ推定
  └── MagicAnimate           --- 参照ポーズからアニメーション

  Character Animation (キャラクターアニメーション)
  ├── Live2D + AI            --- 2Dキャラの動作生成
  ├── Mixamo                 --- 3Dキャラのリギング自動化
  └── Motion Diffusion Model --- テキスト→3Dモーション
```

### 1.2 生成パイプライン

```
  [テキストプロンプト]
       |
       v
  [AI 画像生成] ──→ [参照画像]
  (Stable Diffusion)     |
                         v
  [Image-to-Video] ──→ [動画クリップ 4秒]
  (Runway Gen-3)         |
                         v
  [フレーム補間] ───→ [滑らかな動画 4秒 60fps]
  (RIFE)                 |
                         v
  [動画結合・編集] ──→ [最終アニメーション]
  (DaVinci Resolve)
```

### 1.3 品質と時間のトレードオフ

```
  品質
  High |                          * Sora
       |                    * Gen-3 Alpha
       |              * Kling
       |         * SVD
       |    * Pika
       |* AnimateDiff
  Low  +--------------------------------
       Fast                        Slow
                  生成時間

  利用目的による選択:
  - SNS コンテンツ → Pika / AnimateDiff (速度重視)
  - CM / プレゼン → Gen-3 Alpha (品質重視)
  - 研究 / 実験  → SVD / AnimateDiff (カスタマイズ性)
```

---

## 2. Image-to-Video 生成

### 2.1 Runway Gen-3 Alpha

```python
# Runway API で Image-to-Video 生成 (擬似コード)
import runway

client = runway.Client(api_key="your-api-key")

# 静止画からアニメーション生成
task = client.image_to_video.create(
    model="gen3a_turbo",
    prompt_image="hero_image.png",
    prompt_text="camera slowly zooms in, cherry blossom petals falling gently, "
                "soft wind blowing through hair, cinematic lighting",
    duration=10,            # 最大10秒
    ratio="16:9",
    watermark=False,
)

# 生成結果の取得
result = task.wait()
result.download("output_animation.mp4")
```

### 2.2 Stable Video Diffusion

```python
# Stable Video Diffusion (ローカル実行)
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.to("cuda")

# 入力画像の読み込み
image = load_image("input_scene.png")
image = image.resize((1024, 576))

# 動画生成
frames = pipe(
    image,
    decode_chunk_size=8,
    motion_bucket_id=127,    # 動きの量 (0-255)
    noise_aug_strength=0.02,
    num_frames=25,
).frames[0]

# GIF / MP4 として保存
from diffusers.utils import export_to_video
export_to_video(frames, "svd_output.mp4", fps=7)
```

### 2.3 AnimateDiff

```python
# AnimateDiff: Stable Diffusion + モーション生成
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

# モーションアダプターの読み込み
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-3")

pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
    torch_dtype=torch.float16,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# テキストから動画生成
output = pipe(
    prompt="a cat walking on a sunny garden path, anime style, "
           "detailed fur, soft shadows, studio ghibli",
    negative_prompt="blurry, low quality, distorted",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
)

frames = output.frames[0]
export_to_video(frames, "animatediff_cat.mp4", fps=8)
```

---

## 3. キャラクターアニメーション

### 3.1 ポーズ制御

```
モーション転写ワークフロー

  参照動画 (人物のダンス)
  +------------------+
  | [ダンス映像]      |
  +--------+---------+
           |
  [DWPose でポーズ推定]
           |
  +--------v---------+
  | [ポーズシーケンス] |  ← 棒人間の連続フレーム
  +--------+---------+
           |
  [ControlNet + AnimateDiff]
           |
  +--------v---------+
  | [キャラが同じ     |  ← キャラクターが同じ動きを再現
  |  ダンスをする動画] |
  +------------------+
```

### 3.2 ループアニメーション

```python
# ループアニメーション生成のコツ
# 最初と最後のフレームが滑らかにつながるよう設定

def create_loop_animation(pipe, prompt, num_frames=16):
    """ループ可能なアニメーション生成"""
    # 通常生成
    output = pipe(prompt=prompt, num_frames=num_frames + 4)
    frames = output.frames[0]

    # 最後の4フレームを最初のフレームにクロスフェード
    loop_frames = []
    for i in range(num_frames):
        if i < num_frames - 4:
            loop_frames.append(frames[i])
        else:
            # クロスフェード
            alpha = (i - (num_frames - 4)) / 4.0
            blended = blend_frames(frames[i], frames[i - num_frames], alpha)
            loop_frames.append(blended)

    return loop_frames
```

---

## 4. ツール比較表

| 特性 | Runway Gen-3 | Pika | SVD | AnimateDiff | Sora |
|------|:-----------:|:----:|:---:|:-----------:|:----:|
| 入力 | 画像+テキスト | 画像+テキスト | 画像 | テキスト | テキスト |
| 最大長 | 10秒 | 4秒 | 4秒 | 2秒 | 60秒 |
| 解像度 | 1280x768 | 1024x576 | 1024x576 | 512x512 | 1920x1080 |
| カスタマイズ | 中 | 低 | 高 | 最高 | 低 |
| 料金 | $12-76/月 | $8-58/月 | 無料(OSS) | 無料(OSS) | 限定 |
| GPU 要件 | クラウド | クラウド | VRAM 16GB+ | VRAM 12GB+ | クラウド |

| ユースケース | 推奨ツール | 理由 |
|------------|-----------|------|
| SNS ショート動画 | Pika | 簡単操作、高速 |
| プロモーション映像 | Runway Gen-3 | 高品質、長尺対応 |
| アニメ風コンテンツ | AnimateDiff | スタイル制御が柔軟 |
| 研究・実験 | SVD | オープンソース、カスタマイズ可 |

---

## 5. アンチパターン

### アンチパターン 1: 一発で完成版を求める

```
BAD:
  1つのプロンプトで完璧な60秒アニメーションを期待
  → 現状の AI は4-10秒が限界、長尺は品質低下

GOOD: ショット単位で生成し、編集ソフトで結合
  1. ストーリーボードを作成（各ショット4-10秒）
  2. 各ショットを個別に AI 生成
  3. 編集ソフトでトランジション付きで結合
  4. BGM・効果音を追加
```

### アンチパターン 2: 解像度とフレームレートの不一致

```
BAD:
  512x512 で生成 → 4K ディスプレイで再生
  → ピクセルが目立ち低品質に見える

GOOD: 用途に合った解像度で生成
  - SNS (縦型): 720x1280 で十分
  - YouTube: 1280x720 以上
  - 大画面: 生成後に Real-ESRGAN でアップスケール
```

---

## 6. FAQ

### Q1. AI アニメーションの一貫性（キャラクターの見た目維持）を保つには？

**A.** (1) 参照画像を固定し、同じ画像から Image-to-Video で各カットを生成する。(2) ControlNet でポーズを制御しつつ、IP-Adapter で見た目を固定する。(3) LoRA を学習させて特定キャラクターのスタイルを維持する。完全な一貫性は現状の技術では難しいため、軽微な差異は編集ソフトで補正する。

### Q2. AI アニメーションの商用利用は可能か？

**A.** ツールの利用規約による。**Runway**: 有料プランで商用利用可。**Pika**: 有料プランで商用利用可。**Stable Video Diffusion**: Stability AI のライセンス（商用利用可、条件あり）。**AnimateDiff**: Apache 2.0 ライセンスで商用利用可。生成物に含まれる既存コンテンツの類似性には注意が必要。

### Q3. ローカル GPU がない場合の選択肢は？

**A.** (1) **クラウド API**: Runway、Pika はクラウド実行でGPU不要。(2) **Google Colab**: 無料で T4 GPU が使える（制限あり）。(3) **クラウド GPU**: Lambda Labs、Vast.ai で A100/H100 を時間レンタル。(4) **Apple Silicon**: M2/M3 Mac で一部モデルが動作（MPS バックエンド）。予算と頻度に応じて選択する。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| Image-to-Video | 静止画から動画を生成。Runway Gen-3、SVD が主要選択肢 |
| Text-to-Video | テキストから動画を直接生成。Sora が最高品質だが限定公開 |
| AnimateDiff | Stable Diffusion ベース。アニメ風に強く、カスタマイズ性最高 |
| ループアニメ | クロスフェード技法で最初と最後を繋げる |
| キャラクター一貫性 | 参照画像固定 + ControlNet + IP-Adapter で維持 |
| ワークフロー | ショット単位で生成 → 編集ソフトで結合が現実的 |

---

## 次に読むべきガイド

- [動画編集](./01-video-editing.md) -- AI 動画編集ツールとの連携
- [バーチャル試着](../03-3d/02-virtual-try-on.md) -- 3D + AI のアニメーション応用
- [倫理的考慮](../03-3d/03-ethical-considerations.md) -- AI 生成コンテンツの著作権

---

## 参考文献

1. **Runway Research** -- https://research.runwayml.com/ -- Gen-3 の技術論文
2. **Stable Video Diffusion** -- Stability AI (2023) -- https://stability.ai/stable-video
3. **AnimateDiff** -- Yuwei Guo et al. (2023) -- テキストから動画生成の研究論文

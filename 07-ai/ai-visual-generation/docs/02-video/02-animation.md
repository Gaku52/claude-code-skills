# アニメーション -- AIアニメーション

> AI技術を活用したアニメーション制作の自動化手法を、画像からの動画生成・キャラクターアニメーション・モーション生成の観点から実践的に解説し、従来数日かかっていた作業を数分に短縮する方法を示す

## この章で学ぶこと

1. **AI アニメーション生成の基本** -- Image-to-Video、Text-to-Video、モーション転写の仕組み
2. **主要ツールとモデル** -- Runway Gen-3、Pika、Stable Video Diffusion、AnimateDiff の比較
3. **実践ワークフロー** -- プロンプト設計、コンシステンシー維持、ループアニメーション制作
4. **高度なキャラクターアニメーション** -- モーション転写、表情制御、リアルタイム生成の最新技術
5. **プロダクションパイプライン** -- 商用レベルのAIアニメーション制作フロー

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

### 1.2 技術進化のタイムライン

```
AIアニメーション技術の進化

  2020  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ├── First Order Motion Model (顔モーション転写)
        └── VQ-VAE (動画トークン化の基盤)

  2021  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ├── CogVideo (テキスト→動画の初期モデル)
        └── FILM (フレーム補間)

  2022  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ├── Make-A-Video (Meta)
        ├── Imagen Video (Google)
        └── AnimateDiff v1 (SD拡張)

  2023  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ├── Stable Video Diffusion (Stability AI)
        ├── Runway Gen-2/Gen-3
        ├── Pika Labs 正式公開
        ├── Kling (中国・快影)
        └── MagicAnimate (人体アニメーション)

  2024  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ├── Sora (OpenAI, 一般公開)
        ├── AnimateDiff v3 + SparseCtrl
        ├── EMO (音声→顔アニメーション)
        └── LivePortrait (リアルタイム顔制御)

  2025  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ├── リアルタイム動画生成
        ├── 長尺一貫性の大幅改善
        └── 3D + アニメーションの統合
```

### 1.3 生成パイプライン

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

### 1.4 品質と時間のトレードオフ

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

### 2.4 AnimateDiff + ControlNet による高度な制御

```python
# AnimateDiff + SparseCtrl: キーフレーム指定による動画生成
import torch
from diffusers import (
    AnimateDiffSparseControlNetPipeline,
    MotionAdapter,
    SparseControlNetModel,
    AutoencoderKL,
)
from diffusers.utils import load_image, export_to_video

# SparseCtrl: 数枚のキーフレームで動画全体を制御
controlnet = SparseControlNetModel.from_pretrained(
    "guoyww/animatediff-sparsectrl-scribble",
    torch_dtype=torch.float16,
)

motion_adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-3",
    torch_dtype=torch.float16,
)

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float16,
)

pipe = AnimateDiffSparseControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=motion_adapter,
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
).to("cuda")

# キーフレーム画像を指定
# フレーム0とフレーム15にスケッチを指定し、間を補間
keyframe_images = {
    0: load_image("sketch_frame_0.png"),   # 開始ポーズ
    15: load_image("sketch_frame_15.png"), # 終了ポーズ
}

# 制御画像の条件マスク
conditioning_frames = [keyframe_images.get(i) for i in range(16)]

output = pipe(
    prompt="a warrior drawing a sword, dynamic action, anime style",
    negative_prompt="blurry, static, low quality",
    num_frames=16,
    conditioning_frames=conditioning_frames,
    controlnet_conditioning_scale=0.7,
    num_inference_steps=25,
    guidance_scale=7.5,
)

export_to_video(output.frames[0], "controlled_animation.mp4", fps=8)
```

### 2.5 RIFE によるフレーム補間

```python
# RIFE: Real-Time Intermediate Flow Estimation
# 生成動画のフレームレートを向上させる
import torch
from PIL import Image
import numpy as np

class RIFEInterpolator:
    """RIFE によるフレーム補間パイプライン"""

    def __init__(self, model_path: str = "pretrained/rife_v4.6"):
        from model.RIFE import Model
        self.model = Model()
        self.model.load_model(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def interpolate_pair(
        self, frame1: np.ndarray, frame2: np.ndarray, ratio: float = 0.5
    ) -> np.ndarray:
        """2フレーム間の中間フレームを生成"""
        # numpy → tensor
        img1 = torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img2 = torch.from_numpy(frame2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        with torch.no_grad():
            mid = self.model.inference(img1, img2, timestep=ratio)

        result = (mid[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return result

    def upscale_fps(
        self, frames: list[np.ndarray], target_multiplier: int = 4
    ) -> list[np.ndarray]:
        """
        フレームリストのFPSを倍増
        target_multiplier=2: 2倍 (8fps→16fps)
        target_multiplier=4: 4倍 (8fps→32fps)
        """
        result = []
        for i in range(len(frames) - 1):
            result.append(frames[i])
            # 中間フレームを再帰的に生成
            self._recursive_interpolate(
                frames[i], frames[i + 1], result,
                depth=0, max_depth=int(np.log2(target_multiplier))
            )
        result.append(frames[-1])
        return result

    def _recursive_interpolate(
        self, f1, f2, result_list, depth, max_depth
    ):
        if depth >= max_depth:
            return
        mid = self.interpolate_pair(f1, f2, 0.5)
        self._recursive_interpolate(f1, mid, result_list, depth + 1, max_depth)
        result_list.append(mid)
        self._recursive_interpolate(mid, f2, result_list, depth + 1, max_depth)


# 使用例: 8fps → 32fps に補間
interpolator = RIFEInterpolator()
original_frames = load_video_frames("animatediff_output.mp4")  # 16フレーム, 8fps
smooth_frames = interpolator.upscale_fps(original_frames, target_multiplier=4)
save_video(smooth_frames, "smooth_animation.mp4", fps=32)
print(f"元: {len(original_frames)}フレーム → 補間後: {len(smooth_frames)}フレーム")
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

### 3.2 ポーズ推定パイプライン実装

```python
# DWPose + ControlNet によるモーション転写の完全パイプライン
import cv2
import numpy as np
from controlnet_aux import DWposeDetector
from PIL import Image

class MotionTransferPipeline:
    """参照動画のモーションをキャラクターに転写する"""

    def __init__(self):
        self.pose_detector = DWposeDetector()
        self.video_frames = []
        self.pose_frames = []

    def extract_poses_from_video(
        self, video_path: str, target_fps: int = 8
    ) -> list[Image.Image]:
        """参照動画からポーズシーケンスを抽出"""
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(original_fps / target_fps))

        poses = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # BGR → RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)

                # ポーズ推定
                pose_image = self.pose_detector(pil_frame)
                poses.append(pose_image)
                self.video_frames.append(pil_frame)

            frame_idx += 1

        cap.release()
        self.pose_frames = poses
        return poses

    def apply_to_character(
        self,
        character_prompt: str,
        negative_prompt: str = "blurry, low quality",
        controlnet_scale: float = 0.8,
    ) -> list[Image.Image]:
        """抽出したポーズをキャラクターに適用"""
        import torch
        from diffusers import (
            AnimateDiffPipeline,
            MotionAdapter,
            ControlNetModel,
        )

        # ControlNet (OpenPose) + AnimateDiff
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float16,
        )

        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-3"
        )

        pipe = AnimateDiffPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            motion_adapter=adapter,
            torch_dtype=torch.float16,
        ).to("cuda")

        # ポーズ条件付きで動画生成
        output = pipe(
            prompt=character_prompt,
            negative_prompt=negative_prompt,
            num_frames=len(self.pose_frames),
            conditioning_frames=self.pose_frames,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=25,
            guidance_scale=7.5,
        )

        return output.frames[0]

    def create_comparison_video(
        self, character_frames: list, output_path: str, fps: int = 8
    ):
        """元動画とキャラクター動画を横並びで比較出力"""
        import ffmpeg

        comparison_frames = []
        for orig, char in zip(self.video_frames, character_frames):
            orig_resized = orig.resize(char.size)
            # 横に結合
            combined = Image.new(
                "RGB",
                (orig_resized.width + char.width, char.height)
            )
            combined.paste(orig_resized, (0, 0))
            combined.paste(char, (orig_resized.width, 0))
            comparison_frames.append(combined)

        # 動画として保存
        from diffusers.utils import export_to_video
        export_to_video(comparison_frames, output_path, fps=fps)


# 使用例
pipeline = MotionTransferPipeline()

# 1. 参照動画からポーズ抽出
poses = pipeline.extract_poses_from_video("dance_reference.mp4", target_fps=8)
print(f"抽出ポーズ数: {len(poses)}")

# 2. キャラクターに適用
character_frames = pipeline.apply_to_character(
    character_prompt="a magical girl in sailor uniform, "
                     "anime style, studio ghibli, detailed",
    controlnet_scale=0.75,
)

# 3. 比較動画を出力
pipeline.create_comparison_video(character_frames, "comparison.mp4")
```

### 3.3 表情アニメーション (LivePortrait)

```python
# LivePortrait: 静止画に表情アニメーションを付与
# リアルタイムで顔の表情を制御するパイプライン

class LivePortraitAnimator:
    """顔画像に対するリアルタイム表情制御"""

    def __init__(self, model_path: str = "pretrained/liveportrait"):
        self.model = self._load_model(model_path)
        self.expression_params = {
            "smile": {"mouth_open": 0.0, "lip_corner_raise": 0.6, "eye_blink": 0.0},
            "surprise": {"mouth_open": 0.7, "lip_corner_raise": 0.0, "eye_wide": 0.8},
            "wink_left": {"eye_blink_left": 0.9, "lip_corner_raise": 0.3},
            "talking": None,  # 音声から自動生成
        }

    def animate_with_expression(
        self,
        source_image: str,
        expression: str,
        duration: float = 2.0,
        fps: int = 30,
    ) -> list:
        """静止画に表情アニメーションを適用"""
        from PIL import Image
        import numpy as np

        source = Image.open(source_image)
        n_frames = int(duration * fps)
        params = self.expression_params.get(expression, {})

        frames = []
        for i in range(n_frames):
            t = i / n_frames
            # イージング関数で自然な動きに
            eased_t = self._ease_in_out(t)

            # 表情パラメータを補間
            current_params = {
                k: v * eased_t for k, v in params.items()
            } if params else {}

            frame = self.model.generate(
                source=source,
                expression_params=current_params,
            )
            frames.append(frame)

        return frames

    def animate_with_audio(
        self,
        source_image: str,
        audio_path: str,
        emotion: str = "neutral",
    ) -> list:
        """音声に同期したリップシンクアニメーション"""
        # Whisper で音声→テキスト→音素を抽出
        from audio_utils import extract_phonemes

        phonemes = extract_phonemes(audio_path)

        # 音素→口形状マッピング
        viseme_map = {
            "a": {"mouth_open": 0.7, "lip_width": 0.6},
            "i": {"mouth_open": 0.3, "lip_width": 0.8},
            "u": {"mouth_open": 0.4, "lip_width": 0.3},
            "e": {"mouth_open": 0.5, "lip_width": 0.7},
            "o": {"mouth_open": 0.6, "lip_width": 0.4},
            "m": {"mouth_open": 0.0, "lip_press": 0.8},
            "silence": {"mouth_open": 0.0, "lip_width": 0.5},
        }

        frames = []
        for phoneme_data in phonemes:
            viseme = viseme_map.get(phoneme_data["phoneme"], viseme_map["silence"])
            frame = self.model.generate(
                source=Image.open(source_image),
                expression_params={**viseme, "emotion": emotion},
            )
            frames.append(frame)

        return frames

    @staticmethod
    def _ease_in_out(t: float) -> float:
        """スムーズなイージング関数 (cubic)"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - (-2 * t + 2) ** 3 / 2
```

### 3.4 ループアニメーション

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


def create_ping_pong_loop(frames: list) -> list:
    """ピンポンループ: 往復で自然なループを作成"""
    # 元フレーム + 逆再生フレーム（最初と最後を除く）
    forward = frames
    backward = frames[-2:0:-1]  # 最後と最初を除いた逆順
    return forward + backward


class AdvancedLoopCreator:
    """高度なループアニメーション生成"""

    def __init__(self, interpolator=None):
        self.interpolator = interpolator  # RIFE等のフレーム補間器

    def create_seamless_loop(
        self,
        frames: list,
        blend_frames: int = 4,
        method: str = "crossfade",
    ) -> list:
        """シームレスなループを作成"""
        if method == "crossfade":
            return self._crossfade_loop(frames, blend_frames)
        elif method == "optical_flow":
            return self._optical_flow_loop(frames, blend_frames)
        elif method == "pingpong":
            return create_ping_pong_loop(frames)
        else:
            raise ValueError(f"未知のメソッド: {method}")

    def _crossfade_loop(self, frames, n_blend):
        """クロスフェードによるループ接続"""
        import numpy as np
        result = []
        total = len(frames)

        for i in range(total):
            if i < total - n_blend:
                result.append(frames[i])
            else:
                # ブレンド領域
                blend_idx = i - (total - n_blend)
                alpha = blend_idx / n_blend
                # フレームiとフレーム(blend_idx)をブレンド
                f1 = np.array(frames[i], dtype=np.float32)
                f2 = np.array(frames[blend_idx], dtype=np.float32)
                blended = (f1 * (1 - alpha) + f2 * alpha).astype(np.uint8)
                result.append(blended)

        return result

    def _optical_flow_loop(self, frames, n_blend):
        """オプティカルフローによる高品質ループ接続"""
        import cv2
        import numpy as np

        result = frames[:len(frames) - n_blend]

        for i in range(n_blend):
            alpha = i / n_blend
            idx_end = len(frames) - n_blend + i
            idx_start = i

            # オプティカルフローで動きを推定
            f_end = np.array(frames[idx_end])
            f_start = np.array(frames[idx_start])

            gray_end = cv2.cvtColor(f_end, cv2.COLOR_RGB2GRAY)
            gray_start = cv2.cvtColor(f_start, cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                gray_end, gray_start,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            # フローでワープしてブレンド
            h, w = f_end.shape[:2]
            flow_map = np.column_stack([
                np.tile(np.arange(w), h),
                np.repeat(np.arange(h), w),
            ]).reshape(h, w, 2).astype(np.float32)
            flow_map += flow * alpha

            warped = cv2.remap(
                f_end, flow_map[:, :, 0], flow_map[:, :, 1],
                cv2.INTER_LINEAR
            )

            blended = (warped * (1 - alpha) + f_start * alpha).astype(np.uint8)
            result.append(blended)

        return result
```

---

## 4. ツール比較表

### 4.1 主要ツール機能比較

| 特性 | Runway Gen-3 | Pika | SVD | AnimateDiff | Sora |
|------|:-----------:|:----:|:---:|:-----------:|:----:|
| 入力 | 画像+テキスト | 画像+テキスト | 画像 | テキスト | テキスト |
| 最大長 | 10秒 | 4秒 | 4秒 | 2秒 | 60秒 |
| 解像度 | 1280x768 | 1024x576 | 1024x576 | 512x512 | 1920x1080 |
| カスタマイズ | 中 | 低 | 高 | 最高 | 低 |
| 料金 | $12-76/月 | $8-58/月 | 無料(OSS) | 無料(OSS) | 限定 |
| GPU 要件 | クラウド | クラウド | VRAM 16GB+ | VRAM 12GB+ | クラウド |

### 4.2 ユースケース別推奨

| ユースケース | 推奨ツール | 理由 |
|------------|-----------|------|
| SNS ショート動画 | Pika | 簡単操作、高速 |
| プロモーション映像 | Runway Gen-3 | 高品質、長尺対応 |
| アニメ風コンテンツ | AnimateDiff | スタイル制御が柔軟 |
| 研究・実験 | SVD | オープンソース、カスタマイズ可 |
| ミュージックビデオ | Runway Gen-3 | Motion Brush対応 |
| キャラクターPV | AnimateDiff + ControlNet | ポーズ制御が可能 |
| 教育コンテンツ | Pika | コスパが良い |
| 映画プリビズ | Sora | 長尺・高品質 |

### 4.3 技術要素の比較

| 要素 | AnimateDiff | SVD | Runway Gen-3 | Sora |
|------|:----------:|:---:|:-----------:|:----:|
| テキスト理解力 | 中 (CLIP) | なし | 高 | 最高 |
| 一貫性 | 中 | 高 | 高 | 非常に高 |
| 物理法則準拠 | 低 | 中 | 中 | 高 |
| カメラ制御 | LoRA/ControlNet | 限定的 | Motion Brush | 自動推定 |
| LoRA対応 | あり | 限定的 | なし | なし |
| ControlNet対応 | あり | 限定的 | なし | なし |
| オフライン実行 | 可能 | 可能 | 不可 | 不可 |
| API提供 | なし(ローカル) | なし(ローカル) | あり | あり |

---

## 5. プロダクションパイプライン

### 5.1 商用アニメーション制作フロー

```
┌─ Phase 1: 企画 ────────────────────────────────────┐
│                                                     │
│  ストーリーボード作成 (Canva AI / Midjourney)        │
│       │                                             │
│       v                                             │
│  シーン分割 (各3-5秒, カメラワーク設計)              │
│       │                                             │
│       v                                             │
│  参照画像生成 (Stable Diffusion + LoRA)              │
│  ※キャラクター一貫性を IP-Adapter で維持             │
│                                                     │
└─────────────────────────────────────────────────────┘
       │
       v
┌─ Phase 2: 生成 ────────────────────────────────────┐
│                                                     │
│  各ショットをAIで生成                                │
│  ├─ Image-to-Video (Runway Gen-3)                   │
│  ├─ AnimateDiff (アニメスタイル)                     │
│  └─ モーション転写 (ControlNet + DWPose)             │
│       │                                             │
│       v                                             │
│  品質スクリーニング (5候補から最良を選択)              │
│  ├─ 時間的一貫性チェック                             │
│  ├─ キャラクター類似度チェック                        │
│  └─ 動きの自然さチェック                             │
│                                                     │
└─────────────────────────────────────────────────────┘
       │
       v
┌─ Phase 3: 後処理 ──────────────────────────────────┐
│                                                     │
│  フレーム補間 (RIFE: 8fps → 30fps)                  │
│       │                                             │
│       v                                             │
│  アップスケール (Real-ESRGAN: 512→1080p)             │
│       │                                             │
│       v                                             │
│  色調補正 (DaVinci Resolve)                          │
│       │                                             │
│       v                                             │
│  トランジション追加 (シーン間の接続)                  │
│       │                                             │
│       v                                             │
│  音楽・SE・ナレーション追加                           │
│       │                                             │
│       v                                             │
│  最終レンダリング (H.264/H.265, 1080p/4K)           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 5.2 バッチ処理による効率化

```python
# 複数ショットの自動バッチ生成パイプライン
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class ShotConfig:
    """ショット設定"""
    shot_id: str
    reference_image: str
    prompt: str
    negative_prompt: str = "blurry, low quality, distorted"
    duration: float = 4.0
    camera_move: str = "static"
    num_candidates: int = 5

class BatchAnimationPipeline:
    """複数ショットの一括生成・管理"""

    def __init__(self, output_dir: str = "./production"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quality_threshold = 0.7

    async def generate_all_shots(
        self, shots: list[ShotConfig]
    ) -> dict[str, str]:
        """全ショットを並列生成"""
        results = {}
        tasks = [self._generate_shot(shot) for shot in shots]
        completed = await asyncio.gather(*tasks)

        for shot, best_path in zip(shots, completed):
            results[shot.shot_id] = best_path

        return results

    async def _generate_shot(self, shot: ShotConfig) -> str:
        """1ショットの候補生成と品質選択"""
        candidates = []
        shot_dir = self.output_dir / shot.shot_id
        shot_dir.mkdir(exist_ok=True)

        for i in range(shot.num_candidates):
            output_path = shot_dir / f"candidate_{i}.mp4"
            await self._generate_single(shot, output_path, seed=i * 1000)
            quality = await self._evaluate_quality(output_path)
            candidates.append((output_path, quality))
            print(f"  {shot.shot_id} 候補{i}: 品質スコア {quality:.2f}")

        # 最高品質の候補を選択
        best = max(candidates, key=lambda x: x[1])
        if best[1] < self.quality_threshold:
            print(f"  警告: {shot.shot_id} の最高品質が閾値未満 ({best[1]:.2f})")

        return str(best[0])

    async def _generate_single(
        self, shot: ShotConfig, output_path: Path, seed: int
    ):
        """単一の候補動画を生成"""
        # Runway API / ローカルモデルを使って生成
        # (実装はAPIクライアントに依存)
        pass

    async def _evaluate_quality(self, video_path: Path) -> float:
        """動画品質の自動評価"""
        scores = {
            "temporal_consistency": self._check_temporal(video_path),
            "sharpness": self._check_sharpness(video_path),
            "motion_quality": self._check_motion(video_path),
        }
        return sum(scores.values()) / len(scores)

    def _check_temporal(self, path) -> float:
        """フレーム間一貫性チェック"""
        # SSIM等でフレーム間の類似度を計算
        return 0.8  # placeholder

    def _check_sharpness(self, path) -> float:
        """鮮明度チェック"""
        return 0.85  # placeholder

    def _check_motion(self, path) -> float:
        """動きの品質チェック"""
        return 0.75  # placeholder

    def concatenate_shots(
        self, shot_paths: dict[str, str], output_path: str,
        transition: str = "crossfade", transition_duration: float = 0.5,
    ):
        """複数ショットを結合して最終動画を作成"""
        import subprocess

        # FFmpeg のフィルタ複合で結合
        inputs = []
        filter_parts = []
        for i, (shot_id, path) in enumerate(sorted(shot_paths.items())):
            inputs.extend(["-i", path])
            filter_parts.append(f"[{i}:v]")

        # クロスフェードフィルタ
        if transition == "crossfade":
            concat_filter = "".join(filter_parts)
            concat_filter += f"concat=n={len(shot_paths)}:v=1:a=0[outv]"
        else:
            concat_filter = "".join(filter_parts)
            concat_filter += f"concat=n={len(shot_paths)}:v=1:a=0[outv]"

        cmd = [
            "ffmpeg", *inputs,
            "-filter_complex", concat_filter,
            "-map", "[outv]",
            "-c:v", "libx264", "-crf", "18",
            output_path,
        ]
        subprocess.run(cmd, check=True)


# 使用例
shots = [
    ShotConfig(
        shot_id="shot_01",
        reference_image="ref/castle.png",
        prompt="majestic castle, camera slowly zooms in, morning mist",
        camera_move="dolly_in",
    ),
    ShotConfig(
        shot_id="shot_02",
        reference_image="ref/garden.png",
        prompt="beautiful garden, cherry blossoms falling, gentle wind",
        camera_move="pan_right",
    ),
    ShotConfig(
        shot_id="shot_03",
        reference_image="ref/character.png",
        prompt="anime girl turns around and smiles, hair flowing",
        camera_move="static",
    ),
]

pipeline = BatchAnimationPipeline(output_dir="./my_animation")
# results = asyncio.run(pipeline.generate_all_shots(shots))
```

---

## 6. キャラクター一貫性の維持テクニック

### 6.1 IP-Adapter による一貫性制御

```python
# IP-Adapter: 参照画像でキャラクターの見た目を固定
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter
from ip_adapter import IPAdapter

class ConsistentCharacterAnimator:
    """キャラクター一貫性を維持したアニメーション生成"""

    def __init__(self):
        self.adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-3"
        )
        self.pipe = AnimateDiffPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            motion_adapter=self.adapter,
            torch_dtype=torch.float16,
        ).to("cuda")

        # IP-Adapter を適用
        self.ip_adapter = IPAdapter(
            self.pipe,
            image_encoder_path="models/image_encoder",
            ip_ckpt="models/ip-adapter_sd15.bin",
            device="cuda",
        )

    def generate_consistent_shots(
        self,
        character_reference: str,
        shot_prompts: list[str],
        ip_scale: float = 0.6,
    ) -> list:
        """同一キャラクターで複数ショットを生成"""
        from PIL import Image

        ref_image = Image.open(character_reference)
        results = []

        for prompt in shot_prompts:
            output = self.ip_adapter.generate(
                prompt=prompt,
                negative_prompt="blurry, inconsistent, different character",
                pil_image=ref_image,
                scale=ip_scale,  # 参照画像の影響度 (0.4-0.8推奨)
                num_frames=16,
                num_inference_steps=25,
            )
            results.append(output.frames[0])

        return results


# 使用例
animator = ConsistentCharacterAnimator()
shots = animator.generate_consistent_shots(
    character_reference="my_character.png",
    shot_prompts=[
        "character walking through a forest, sunlight filtering through trees",
        "character sitting by a lake, looking at reflection",
        "character standing on a cliff, wind blowing cape",
    ],
    ip_scale=0.65,
)
```

### 6.2 LoRA によるスタイル固定

```python
# LoRA学習でキャラクター/スタイルを固定
from diffusers import AnimateDiffPipeline, MotionAdapter

def setup_consistent_pipeline(
    base_model: str = "runwayml/stable-diffusion-v1-5",
    character_lora: str = "path/to/character_lora.safetensors",
    style_lora: str = "path/to/style_lora.safetensors",
    character_weight: float = 0.8,
    style_weight: float = 0.6,
):
    """キャラクター+スタイルLoRAを組み合わせたパイプライン"""
    import torch

    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3"
    )

    pipe = AnimateDiffPipeline.from_pretrained(
        base_model,
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    ).to("cuda")

    # キャラクターLoRA
    pipe.load_lora_weights(
        character_lora,
        adapter_name="character",
    )

    # スタイルLoRA
    pipe.load_lora_weights(
        style_lora,
        adapter_name="style",
    )

    # 重みを設定
    pipe.set_adapters(
        ["character", "style"],
        adapter_weights=[character_weight, style_weight],
    )

    return pipe


# キャラクターLoRAの学習コマンド例
LORA_TRAINING_CONFIG = """
# Kohya-ss sd-scripts による LoRA 学習
# キャラクター学習用の設定例

accelerate launch train_network.py \\
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\
    --train_data_dir="./training_images/my_character" \\
    --output_dir="./lora_output" \\
    --output_name="my_character_lora" \\
    --network_module=networks.lora \\
    --network_dim=128 \\
    --network_alpha=64 \\
    --resolution=512 \\
    --train_batch_size=1 \\
    --learning_rate=1e-4 \\
    --max_train_steps=1000 \\
    --caption_extension=".txt" \\
    --mixed_precision="fp16" \\
    --save_every_n_steps=200
"""
```

---

## 7. パフォーマンス最適化

### 7.1 GPU メモリ最適化

```python
# AnimateDiff のメモリ最適化テクニック集
import torch

def optimize_pipeline_memory(pipe):
    """メモリ使用量を最適化"""

    # 1. Attention Slicing (メモリ削減、速度低下あり)
    pipe.enable_attention_slicing(slice_size="auto")

    # 2. VAE Slicing (大きなフレーム数に有効)
    pipe.enable_vae_slicing()

    # 3. CPU Offload (VRAM不足時)
    pipe.enable_model_cpu_offload()

    # 4. xFormers (速度向上+メモリ削減)
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xFormers 有効化成功")
    except ImportError:
        print("xFormers未インストール、通常のAttentionを使用")

    return pipe


# メモリ使用量の目安
MEMORY_REQUIREMENTS = """
AnimateDiff メモリ要件（16フレーム生成時）

| 設定                    | VRAM使用量 | 生成速度 |
|------------------------|-----------|---------|
| FP32 (最適化なし)        | ~16GB    | 遅い    |
| FP16                    | ~10GB    | 標準    |
| FP16 + Attention Slicing| ~8GB     | やや遅い |
| FP16 + xFormers         | ~7GB     | 高速    |
| FP16 + CPU Offload      | ~5GB     | 遅い    |

SVD メモリ要件（25フレーム生成時）

| 設定                    | VRAM使用量 | 生成速度 |
|------------------------|-----------|---------|
| FP16                    | ~16GB    | 標準    |
| FP16 + decode_chunk=8   | ~12GB    | やや遅い |
| FP16 + CPU Offload      | ~8GB     | 遅い    |
"""
```

### 7.2 品質 vs 速度のトレードオフ設定

```python
# 用途に応じた品質/速度プリセット
QUALITY_PRESETS = {
    "draft": {
        "description": "高速プレビュー用",
        "num_inference_steps": 10,
        "guidance_scale": 5.0,
        "num_frames": 8,
        "width": 256,
        "height": 256,
        "fps": 4,
        "estimated_time": "~5秒",
    },
    "standard": {
        "description": "SNS投稿用",
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "num_frames": 16,
        "width": 512,
        "height": 512,
        "fps": 8,
        "estimated_time": "~30秒",
    },
    "high": {
        "description": "プロモーション用",
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "num_frames": 24,
        "width": 768,
        "height": 768,
        "fps": 12,
        "estimated_time": "~2分",
    },
    "production": {
        "description": "最終納品用",
        "num_inference_steps": 40,
        "guidance_scale": 8.0,
        "num_frames": 32,
        "width": 1024,
        "height": 576,
        "fps": 24,
        "estimated_time": "~5分",
        "post_processing": ["rife_interpolation", "realesrgan_upscale"],
    },
}
```

---

## 8. トラブルシューティング

### 8.1 よくある問題と解決策

| 問題 | 原因 | 解決策 |
|------|------|--------|
| フリッカー（ちらつき） | フレーム間の一貫性不足 | motion_bucket_idを下げる / guidance_scaleを調整 |
| キャラクターの顔が変化 | 一貫性制御不足 | IP-Adapter / LoRA を使用 |
| 動きが少なすぎる | motion パラメータが低い | motion_bucket_id を上げる (SVD) |
| 動きが激しすぎる | motion パラメータが高い | motion_bucket_id を下げる / noise_aug_strength を下げる |
| VRAM不足 | メモリ最適化不足 | attention_slicing / cpu_offload を有効化 |
| 生成が遅い | 設定が重すぎる | ステップ数削減 / 解像度を下げる / xFormers 使用 |
| 色の不一致 | VAE のバージョン差異 | 統一VAE (ft-mse) を使用 |
| ループ時の不連続 | 終端フレームの不一致 | クロスフェード / オプティカルフロー ブレンド |

### 8.2 デバッグ手法

```python
# アニメーション品質のデバッグツール
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

class AnimationDebugger:
    """生成アニメーションの品質診断"""

    def analyze_temporal_consistency(self, frames: list) -> dict:
        """フレーム間一貫性を分析"""
        ssim_scores = []
        for i in range(len(frames) - 1):
            f1 = np.array(frames[i])
            f2 = np.array(frames[i + 1])
            score = ssim(f1, f2, channel_axis=2)
            ssim_scores.append(score)

        return {
            "mean_ssim": np.mean(ssim_scores),
            "min_ssim": np.min(ssim_scores),
            "max_ssim": np.max(ssim_scores),
            "std_ssim": np.std(ssim_scores),
            "flicker_frames": [
                i for i, s in enumerate(ssim_scores) if s < 0.85
            ],
            "verdict": "良好" if np.mean(ssim_scores) > 0.9 else
                       "要改善" if np.mean(ssim_scores) > 0.8 else "不良",
        }

    def analyze_motion_magnitude(self, frames: list) -> dict:
        """動きの量を分析"""
        import cv2
        motion_scores = []
        for i in range(len(frames) - 1):
            f1 = cv2.cvtColor(np.array(frames[i]), cv2.COLOR_RGB2GRAY)
            f2 = cv2.cvtColor(np.array(frames[i + 1]), cv2.COLOR_RGB2GRAY)
            diff = np.mean(np.abs(f1.astype(float) - f2.astype(float)))
            motion_scores.append(diff)

        return {
            "mean_motion": np.mean(motion_scores),
            "max_motion": np.max(motion_scores),
            "motion_profile": motion_scores,
            "verdict": "動きが少なすぎ" if np.mean(motion_scores) < 2.0 else
                       "適切" if np.mean(motion_scores) < 15.0 else "動きが激しすぎ",
        }

    def generate_diagnostic_report(self, frames: list) -> str:
        """総合診断レポートを生成"""
        consistency = self.analyze_temporal_consistency(frames)
        motion = self.analyze_motion_magnitude(frames)

        report = f"""
=== アニメーション品質診断レポート ===

フレーム数: {len(frames)}

[時間的一貫性]
  平均SSIM: {consistency['mean_ssim']:.4f}
  最小SSIM: {consistency['min_ssim']:.4f}
  フリッカーフレーム: {consistency['flicker_frames']}
  判定: {consistency['verdict']}

[動きの量]
  平均動き量: {motion['mean_motion']:.2f}
  最大動き量: {motion['max_motion']:.2f}
  判定: {motion['verdict']}

[推奨アクション]
"""
        if consistency["verdict"] != "良好":
            report += "  - guidance_scale を 0.5-1.0 下げてみてください\n"
            report += "  - motion_bucket_id を下げてみてください\n"
        if motion["verdict"] == "動きが少なすぎ":
            report += "  - motion_bucket_id を上げてみてください\n"
            report += "  - プロンプトに具体的な動きの記述を追加してください\n"
        if motion["verdict"] == "動きが激しすぎ":
            report += "  - motion_bucket_id を下げてみてください\n"
            report += "  - noise_aug_strength を 0.01 に設定してください\n"

        return report
```

---

## 9. アンチパターン

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

### アンチパターン 3: フレーム補間の過剰適用

```
BAD:
  8fps → RIFE で 240fps に補間
  → 中間フレームにゴースト/アーティファクトが大量発生
  → 動きが不自然に「ぬるぬる」になる

GOOD: 適切な倍率で補間
  - 8fps → 24fps (3倍) が安全な上限
  - 8fps → 30fps (4倍) は許容範囲
  - それ以上の補間は品質低下リスクが高い
  - アニメ風は意図的に低FPS (12-15fps) の方が自然
```

### アンチパターン 4: キャラクター一貫性を無視した連続ショット

```
BAD:
  各ショットを独立に生成
  → ショットごとにキャラクターの顔・服装・体型が変わる
  → 視聴者に違和感を与える

GOOD: 一貫性維持の仕組みを組み込む
  1. 参照画像を全ショットで共有 (IP-Adapter)
  2. キャラクター LoRA を学習して使用
  3. シード値を関連付けて生成
  4. 後処理でFace Swapによる統一も検討
```

---

## 10. 演習

### 演習1: 基礎 -- AnimateDiff で初めてのアニメーション

```
目標: テキストプロンプトから16フレームのアニメーションGIFを生成する

手順:
1. AnimateDiff パイプラインをセットアップ
2. 風景プロンプト（例: "ocean waves at sunset"）で生成
3. guidance_scale を 5.0, 7.5, 10.0 で比較
4. motion_bucket_id の影響を確認（存在する場合）
5. GIF として保存し、ループ再生を確認

評価基準:
- 生成成功 (エラーなし)
- フリッカーが最小限
- プロンプトに沿った内容
```

### 演習2: 応用 -- モーション転写パイプライン構築

```
目標: 参照動画のポーズをアニメキャラクターに転写する

手順:
1. 5-10秒のダンス動画を用意
2. DWPose でポーズシーケンスを抽出
3. ControlNet + AnimateDiff でキャラクターに適用
4. IP-Adapter でキャラクターの一貫性を維持
5. RIFE でフレーム補間 (8fps → 24fps)

評価基準:
- ポーズの正確な転写
- キャラクターの一貫性維持
- 滑らかなフレーム補間
```

### 演習3: 発展 -- 30秒アニメーションPV制作

```
目標: ストーリーボードから30秒のアニメーションPVを制作する

手順:
1. 6-8ショットのストーリーボードを設計
2. Stable Diffusion でキャラクター参照画像を生成
3. 各ショットを AnimateDiff / Runway で生成 (各5候補)
4. 品質スクリーニングで最良候補を選択
5. RIFE + Real-ESRGAN で後処理
6. DaVinci Resolve でトランジション・BGM追加
7. 最終レンダリング (1080p, 30fps)

評価基準:
- キャラクター一貫性 (全ショット)
- ストーリーの伝達力
- 技術品質 (フリッカー、解像度、FPS)
- 全体の完成度
```

---

## 11. FAQ

### Q1. AI アニメーションの一貫性（キャラクターの見た目維持）を保つには？

**A.** (1) 参照画像を固定し、同じ画像から Image-to-Video で各カットを生成する。(2) ControlNet でポーズを制御しつつ、IP-Adapter で見た目を固定する。(3) LoRA を学習させて特定キャラクターのスタイルを維持する。完全な一貫性は現状の技術では難しいため、軽微な差異は編集ソフトで補正する。

### Q2. AI アニメーションの商用利用は可能か？

**A.** ツールの利用規約による。**Runway**: 有料プランで商用利用可。**Pika**: 有料プランで商用利用可。**Stable Video Diffusion**: Stability AI のライセンス（商用利用可、条件あり）。**AnimateDiff**: Apache 2.0 ライセンスで商用利用可。生成物に含まれる既存コンテンツの類似性には注意が必要。

### Q3. ローカル GPU がない場合の選択肢は？

**A.** (1) **クラウド API**: Runway、Pika はクラウド実行でGPU不要。(2) **Google Colab**: 無料で T4 GPU が使える（制限あり）。(3) **クラウド GPU**: Lambda Labs、Vast.ai で A100/H100 を時間レンタル。(4) **Apple Silicon**: M2/M3 Mac で一部モデルが動作（MPS バックエンド）。予算と頻度に応じて選択する。

### Q4. フレーム補間 (RIFE) と元の生成フレーム数を増やすのは、どちらが効果的か？

**A.** 状況による。(1) **生成フレーム数を増やす**: 動きの一貫性が高く、物理的に正しい中間フレームが得られる。ただしVRAM消費が増大し生成時間も長くなる。(2) **RIFE で補間**: 後処理なので生成時間に影響しない。VRAM消費も少ない。ただし激しい動きでゴーストが発生する場合がある。**推奨**: まず適切なフレーム数（16-24）で生成し、RIFE で2-3倍に補間するハイブリッドアプローチが最も効率的。

### Q5. アニメスタイルと実写スタイルで推奨設定は異なるか？

**A.** はい、大きく異なる。(1) **アニメスタイル**: セル画風のため低FPS（12-15fps）でも自然。AnimateDiff + アニメ特化LoRA が最適。guidance_scale は 7-9 が推奨。(2) **実写スタイル**: 24-30fps が必要。SVD / Runway Gen-3 が高品質。guidance_scale は 5-7 が推奨。RIFE 補間も実写の方が効果的に機能する。(3) **共通**: いずれもネガティブプロンプトで "blurry, low quality, distorted" を指定することで品質が向上する。

### Q6. 音声同期アニメーション（リップシンク）の現状は？

**A.** 2025年時点で急速に進化している分野。(1) **EMO (Alibaba)**: 音声から顔アニメーションを生成。品質は高いがリソース消費大。(2) **LivePortrait**: リアルタイムで表情を制御可能。軽量で実用的。(3) **Pika Lip Sync**: Pika の組み込み機能。手軽だが品質は限定的。(4) **SadTalker**: オープンソースで安定。品質は中程度。**推奨**: 高品質を求める場合は EMO や LivePortrait、手軽さを求める場合は Pika や SadTalker を使い分ける。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| Image-to-Video | 静止画から動画を生成。Runway Gen-3、SVD が主要選択肢 |
| Text-to-Video | テキストから動画を直接生成。Sora が最高品質だが限定公開 |
| AnimateDiff | Stable Diffusion ベース。アニメ風に強く、カスタマイズ性最高 |
| ループアニメ | クロスフェード/オプティカルフロー技法で最初と最後を繋げる |
| キャラクター一貫性 | 参照画像固定 + ControlNet + IP-Adapter + LoRA で維持 |
| ワークフロー | ショット単位で生成 → 編集ソフトで結合が現実的 |
| フレーム補間 | RIFE で 2-3倍補間が安全。過剰補間はアーティファクトの原因 |
| モーション転写 | DWPose → ControlNet で参照動画のポーズをキャラクターに適用 |
| 品質管理 | 自動診断ツール + 複数候補生成 + 人間による最終選択 |

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
4. **RIFE: Real-Time Intermediate Flow Estimation** -- Huang et al. (2022) -- https://arxiv.org/abs/2011.06294
5. **IP-Adapter** -- Ye et al. (2023) -- https://ip-adapter.github.io/
6. **DWPose** -- Yang et al. (2023) -- 効率的な全身ポーズ推定
7. **LivePortrait** -- https://github.com/KwaiVGI/LivePortrait -- リアルタイム顔制御
8. **SparseCtrl** -- Guo et al. (2024) -- AnimateDiff のキーフレーム制御拡張

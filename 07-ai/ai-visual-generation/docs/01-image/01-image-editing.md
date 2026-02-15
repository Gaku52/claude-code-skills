# 画像編集 — インペインティング、アウトペインティング

> AI による画像の部分修正・拡張技術を、マスク生成からシームレスな合成まで実践的に解説する。

---

## この章で学ぶこと

1. **インペインティングの原理と実装** — マスク領域の自然な補完、プロンプトによる部分書き換え
2. **アウトペインティングの技法** — 画像の境界を超えた拡張、パノラマ生成
3. **Img2Img と ControlNet による高度な編集** — 構図維持、スタイル変換、ポーズ制御
4. **自動マスク生成パイプライン** — SAM、Grounding DINO、テキスト指定による領域選択
5. **Instruct-Pix2Pix と自然言語編集** — テキスト指示による直感的な画像編集
6. **商用ワークフロー** — バッチ処理、品質管理、本番環境向けパイプライン

---

## 1. インペインティング

### 1.1 インペインティングの理論的背景

インペインティングは拡散モデルの逆拡散プロセスを応用し、マスク領域のみにノイズを付加して段階的にデノイズする。元画像の非マスク領域は各ステップで強制的に元のピクセル値に戻す（Repaint方式）か、専用モデルが元画像とマスクをコンディションとして受け取る。

```
数学的定式化:

逆拡散ステップ t:
  x_{t-1}^{masked} = denoise(x_t, t, prompt)   # マスク領域
  x_{t-1}^{unmasked} = original_image           # 非マスク領域
  x_{t-1} = mask * x_{t-1}^{masked} + (1 - mask) * x_{t-1}^{unmasked}

Repaint 方式の改善:
  - 各ステップで n 回のリサンプリング (jump_length)
  - 前方拡散で再ノイズ付加 → 逆拡散でデノイズ
  - 境界の一貫性が大幅に向上
  - n=10 が典型的な設定
```

### コード例1: diffusers によるインペインティング

```python
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image
import torch

# インペインティングパイプラインをロード
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# メモリ最適化
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

# 元画像とマスクを読み込み
image = Image.open("room.png").resize((1024, 1024))
mask = Image.open("mask.png").resize((1024, 1024))  # 白=編集領域

# インペインティング実行
result = pipe(
    prompt="a modern leather sofa, interior design, photorealistic",
    negative_prompt="low quality, blurry, distorted",
    image=image,
    mask_image=mask,
    num_inference_steps=30,
    guidance_scale=7.5,
    strength=0.85,    # 元画像からの逸脱度 (0=変更なし, 1=完全再生成)
).images[0]

result.save("room_edited.png")
```

### コード例2: プログラマティックなマスク生成（拡張版）

```python
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from typing import Optional, Tuple, List


class MaskGenerator:
    """
    多様なマスク生成手法を提供する統合クラス

    マスクの規約:
    - 白 (255) = 編集領域
    - 黒 (0) = 保護領域
    - グレー = ブレンド強度（フェザリング）
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def rectangular(self, bbox: Tuple[int, int, int, int],
                     feather: int = 0) -> Image.Image:
        """矩形マスクを生成"""
        mask = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(bbox, fill=255)
        if feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(feather))
        return mask

    def circular(self, center: Tuple[int, int],
                  radius: int, feather: int = 0) -> Image.Image:
        """円形マスクを生成"""
        mask = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(mask)
        x, y = center
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=255,
        )
        if feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(feather))
        return mask

    def polygon(self, points: List[Tuple[int, int]],
                 feather: int = 0) -> Image.Image:
        """多角形マスクを生成"""
        mask = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(points, fill=255)
        if feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(feather))
        return mask

    def freeform(self, points: List[Tuple[int, int]],
                  width: int = 30,
                  feather: int = 10) -> Image.Image:
        """フリーハンドのブラシストロークマスクを生成"""
        mask = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(mask)
        for i in range(len(points) - 1):
            draw.line(
                [points[i], points[i + 1]],
                fill=255,
                width=width,
            )
            # 丸い結合点
            x, y = points[i]
            r = width // 2
            draw.ellipse(
                [x - r, y - r, x + r, y + r],
                fill=255,
            )
        if feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(feather))
        return mask

    def gradient(self, direction: str = "left_to_right",
                  start: float = 0.0,
                  end: float = 1.0) -> Image.Image:
        """グラデーションマスクを生成"""
        arr = np.zeros((self.height, self.width), dtype=np.float32)

        if direction == "left_to_right":
            for x in range(self.width):
                arr[:, x] = start + (end - start) * x / self.width
        elif direction == "top_to_bottom":
            for y in range(self.height):
                arr[y, :] = start + (end - start) * y / self.height
        elif direction == "center_out":
            cx, cy = self.width // 2, self.height // 2
            max_dist = np.sqrt(cx ** 2 + cy ** 2)
            for y in range(self.height):
                for x in range(self.width):
                    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    arr[y, x] = start + (end - start) * dist / max_dist

        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def from_color_range(self, image: Image.Image,
                          target_color: Tuple[int, int, int],
                          tolerance: int = 30,
                          feather: int = 5) -> Image.Image:
        """色範囲によるマスク生成"""
        img_array = np.array(image)
        target = np.array(target_color)

        # 色差を計算
        diff = np.sqrt(
            np.sum((img_array.astype(float) - target) ** 2, axis=2)
        )

        # 閾値でマスク化
        mask_array = (diff < tolerance).astype(np.uint8) * 255
        mask = Image.fromarray(mask_array)

        if feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(feather))
        return mask

    def combine(self, masks: List[Image.Image],
                 mode: str = "union") -> Image.Image:
        """複数マスクを合成"""
        arrays = [np.array(m, dtype=float) for m in masks]

        if mode == "union":
            result = np.maximum.reduce(arrays)
        elif mode == "intersection":
            result = np.minimum.reduce(arrays)
        elif mode == "difference":
            result = np.clip(arrays[0] - arrays[1], 0, 255)
        elif mode == "xor":
            a, b = arrays[0] > 127, arrays[1] > 127
            result = np.logical_xor(a, b).astype(float) * 255
        else:
            result = arrays[0]

        return Image.fromarray(result.astype(np.uint8))

    def invert(self, mask: Image.Image) -> Image.Image:
        """マスクを反転"""
        return Image.fromarray(255 - np.array(mask))

    def dilate(self, mask: Image.Image,
                pixels: int = 10) -> Image.Image:
        """マスクを拡張（膨張）"""
        arr = np.array(mask)
        from scipy.ndimage import binary_dilation
        struct = np.ones((pixels * 2 + 1, pixels * 2 + 1))
        dilated = binary_dilation(arr > 127, structure=struct)
        return Image.fromarray((dilated * 255).astype(np.uint8))

    def erode(self, mask: Image.Image,
               pixels: int = 10) -> Image.Image:
        """マスクを縮小（浸食）"""
        arr = np.array(mask)
        from scipy.ndimage import binary_erosion
        struct = np.ones((pixels * 2 + 1, pixels * 2 + 1))
        eroded = binary_erosion(arr > 127, structure=struct)
        return Image.fromarray((eroded * 255).astype(np.uint8))


# 使用例
gen = MaskGenerator(1024, 1024)

# 矩形マスク（フェザリング付き）
rect_mask = gen.rectangular((200, 300, 800, 700), feather=20)

# 円形マスク
circle_mask = gen.circular((512, 512), 200, feather=15)

# 複数マスクを合成
combined = gen.combine([rect_mask, circle_mask], mode="union")

# マスクを拡張
expanded = gen.dilate(combined, pixels=15)
```

### コード例3: SAM (Segment Anything) による自動マスク生成

```python
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import numpy as np
import torch


class SAMAutoMasker:
    """
    Segment Anything Model による高精度自動マスク生成

    SAMは任意の物体を検出しセグメンテーションする
    汎用的なビジョンモデル。
    """

    def __init__(self, model_type: str = "vit_h",
                  checkpoint: str = "sam_vit_h_4b8939.pth"):
        self.sam = sam_model_registry[model_type](
            checkpoint=checkpoint
        )
        self.sam.to("cuda")
        self.predictor = SamPredictor(self.sam)

    def mask_from_point(self, image: Image.Image,
                         point: tuple[int, int],
                         label: int = 1) -> Image.Image:
        """
        ポイント指定でマスクを生成

        Args:
            image: 入力画像
            point: (x, y) 座標
            label: 1=前景, 0=背景
        """
        img_array = np.array(image)
        self.predictor.set_image(img_array)

        input_point = np.array([[point[0], point[1]]])
        input_label = np.array([label])

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # 最高スコアのマスクを選択
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]

        return Image.fromarray(
            (best_mask * 255).astype(np.uint8)
        )

    def mask_from_box(self, image: Image.Image,
                       box: tuple[int, int, int, int]) -> Image.Image:
        """
        バウンディングボックス指定でマスクを生成

        Args:
            image: 入力画像
            box: (x1, y1, x2, y2)
        """
        img_array = np.array(image)
        self.predictor.set_image(img_array)

        input_box = np.array(box)

        masks, scores, logits = self.predictor.predict(
            box=input_box,
            multimask_output=False,
        )

        return Image.fromarray(
            (masks[0] * 255).astype(np.uint8)
        )

    def mask_from_points_and_boxes(
        self,
        image: Image.Image,
        points: list[tuple[int, int]] = None,
        point_labels: list[int] = None,
        boxes: list[tuple[int, int, int, int]] = None,
    ) -> Image.Image:
        """
        複数のポイントとボックスを組み合わせてマスクを生成
        """
        img_array = np.array(image)
        self.predictor.set_image(img_array)

        kwargs = {"multimask_output": False}

        if points:
            kwargs["point_coords"] = np.array(points)
            kwargs["point_labels"] = np.array(
                point_labels or [1] * len(points)
            )

        if boxes:
            kwargs["box"] = np.array(boxes[0])

        masks, scores, logits = self.predictor.predict(**kwargs)

        return Image.fromarray(
            (masks[0] * 255).astype(np.uint8)
        )


# 使用例
masker = SAMAutoMasker()
image = Image.open("photo.jpg")

# ソファの中心をクリック → ソファのマスクを自動生成
sofa_mask = masker.mask_from_point(image, (400, 500))

# バウンディングボックスで指定
person_mask = masker.mask_from_box(image, (100, 50, 300, 600))
```

### コード例4: Grounding DINO + SAM によるテキスト指定マスク

```python
"""
テキストで「何をマスクするか」を指定する方法。
Grounding DINOで物体検出 → SAMでセグメンテーション。
"""

from groundingdino.util.inference import (
    load_model, load_image, predict
)
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
from PIL import Image
import torch


class TextGuidedMasker:
    """
    テキスト指示で対象物のマスクを自動生成

    処理フロー:
    1. Grounding DINO で対象物を検出（バウンディングボックス）
    2. SAM でバウンディングボックス内を精密セグメンテーション
    3. 結果マスクを返す
    """

    def __init__(self):
        # Grounding DINO
        self.dino_model = load_model(
            "groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "weights/groundingdino_swint_ogc.pth",
        )
        # SAM
        sam = sam_model_registry["vit_h"](
            checkpoint="weights/sam_vit_h_4b8939.pth"
        )
        sam.to("cuda")
        self.sam_predictor = SamPredictor(sam)

    def create_mask(self, image_path: str,
                     text_prompt: str,
                     box_threshold: float = 0.3,
                     text_threshold: float = 0.25,
                     feather: int = 5) -> dict:
        """
        テキスト指示でマスクを生成

        Args:
            image_path: 画像パス
            text_prompt: 対象物の説明（例: "sofa", "person"）
            box_threshold: 検出の信頼度閾値
            text_threshold: テキスト一致の閾値
            feather: マスク境界のぼかし

        Returns:
            dict: マスク画像、検出情報
        """
        # Grounding DINO で検出
        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model=self.dino_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        if len(boxes) == 0:
            return {"mask": None, "error": "対象物が検出されませんでした"}

        # SAM で精密セグメンテーション
        pil_image = Image.open(image_path).convert("RGB")
        self.sam_predictor.set_image(np.array(pil_image))

        # 全検出をマスク合成
        h, w = pil_image.size[1], pil_image.size[0]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for box in boxes:
            # 正規化座標をピクセル座標に変換
            x1, y1, x2, y2 = box.numpy()
            pixel_box = np.array([
                x1 * w, y1 * h, x2 * w, y2 * h
            ])

            masks, scores, _ = self.sam_predictor.predict(
                box=pixel_box,
                multimask_output=False,
            )
            combined_mask = np.maximum(
                combined_mask,
                (masks[0] * 255).astype(np.uint8),
            )

        mask_image = Image.fromarray(combined_mask)

        # フェザリング
        if feather > 0:
            from PIL import ImageFilter
            mask_image = mask_image.filter(
                ImageFilter.GaussianBlur(feather)
            )

        return {
            "mask": mask_image,
            "detections": len(boxes),
            "phrases": phrases,
            "confidence": logits.tolist(),
        }


# 使用例
masker = TextGuidedMasker()

# テキストで対象を指定
result = masker.create_mask(
    "living_room.jpg",
    text_prompt="old sofa",
    box_threshold=0.35,
)

if result["mask"]:
    print(f"検出数: {result['detections']}")
    print(f"検出物: {result['phrases']}")
    result["mask"].save("sofa_mask.png")
```

### ASCII図解1: インペインティングの処理フロー

```
入力:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 元画像       │    │ マスク       │    │ プロンプト   │
│ ┌───┐       │    │ ┌───┐       │    │ "革のソファ" │
│ │古い│       │    │ │███│       │    │             │
│ │椅子│       │    │ │███│ ←白   │    │             │
│ └───┘       │    │ └───┘       │    │             │
│    (保持)    │    │  (編集)     │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
      │                  │                  │
      v                  v                  v
┌────────────────────────────────────────────────┐
│            インペインティングモデル              │
│                                                │
│  1. マスク領域をノイズで置換                    │
│  2. 周囲のコンテキストを参照                    │
│  3. プロンプトに基づきノイズ除去                │
│  4. 境界をシームレスにブレンド                  │
└────────────────────┬───────────────────────────┘
                     │
                     v
┌─────────────┐
│ 結果画像     │
│ ┌───┐       │
│ │新しい│     │
│ │ソファ│     │
│ └───┘       │
│  (自然に合成)│
└─────────────┘

マスク処理の詳細:
┌─────────────────────────────────────────────┐
│ Step 1: マスク前処理                         │
│  元マスク    → 膨張(dilate)  → フェザリング  │
│  ┌──┐         ┌────┐          ┌────┐         │
│  │██│         │████│          │░▓█▓░│        │
│  │██│    →    │████│    →     │░▓█▓░│        │
│  └──┘         │████│          │░▓█▓░│        │
│               └────┘          └────┘         │
│  鮮明な境界   少し拡張        滑らかな境界    │
│                                              │
│ Step 2: ノイズスケジューリング                │
│  t=T:  全面ノイズ (マスク内のみ)             │
│  t=T/2: 構造が見え始める                     │
│  t=0:   最終結果                             │
│                                              │
│ Step 3: 境界ブレンド                         │
│  マスクのグレー値に応じて                     │
│  元画像と生成結果を線形補間                   │
│  result = mask * generated + (1-mask) * orig │
└─────────────────────────────────────────────┘
```

### 1.2 高度なインペインティングテクニック

```python
class AdvancedInpainter:
    """
    高度なインペインティング機能

    - 複数領域の同時編集
    - コンテキスト認識の強化
    - 反復的な品質改善
    """

    def __init__(self, model_id: str = None):
        from diffusers import StableDiffusionXLInpaintPipeline

        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id or
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipe.enable_model_cpu_offload()

    def inpaint_with_context(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        context_prompt: str = "",
        strength: float = 0.85,
        steps: int = 30,
        cfg_scale: float = 7.5,
    ) -> Image.Image:
        """
        コンテキスト認識を強化したインペインティング

        Args:
            image: 元画像
            mask: マスク画像
            prompt: 生成プロンプト
            context_prompt: 元画像のコンテキスト説明
            strength: 編集強度
        """
        # コンテキストをプロンプトに統合
        if context_prompt:
            full_prompt = (
                f"{prompt}, "
                f"consistent with {context_prompt}, "
                f"matching ambient lighting and perspective"
            )
        else:
            full_prompt = prompt

        result = self.pipe(
            prompt=full_prompt,
            negative_prompt=(
                negative_prompt or
                "low quality, blurry, inconsistent lighting, "
                "seam visible, color mismatch"
            ),
            image=image,
            mask_image=mask,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            strength=strength,
        ).images[0]

        return result

    def iterative_refinement(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        iterations: int = 3,
        strength_schedule: list[float] = None,
    ) -> list[Image.Image]:
        """
        反復的にインペインティング品質を改善

        1回目: 大まかな構造を生成 (high strength)
        2回目: ディテールを改善 (medium strength)
        3回目: 境界を滑らかに (low strength)
        """
        if strength_schedule is None:
            strength_schedule = [0.9, 0.6, 0.35]

        results = []
        current = image

        for i, strength in enumerate(strength_schedule[:iterations]):
            print(f"Iteration {i+1}/{iterations}, "
                  f"strength={strength}")

            result = self.pipe(
                prompt=prompt,
                negative_prompt="low quality, blurry, seam",
                image=current,
                mask_image=mask,
                num_inference_steps=30,
                guidance_scale=7.5 - i * 0.5,  # 段階的に下げる
                strength=strength,
            ).images[0]

            results.append(result)
            current = result

        return results

    def multi_region_edit(
        self,
        image: Image.Image,
        edits: list[dict],
    ) -> Image.Image:
        """
        複数領域を順番に編集

        Args:
            edits: [
                {"mask": mask_img, "prompt": "...", "strength": 0.8},
                {"mask": mask_img2, "prompt": "...", "strength": 0.7},
            ]
        """
        current = image

        for i, edit in enumerate(edits):
            print(f"Editing region {i+1}/{len(edits)}: "
                  f"{edit['prompt'][:50]}...")

            current = self.pipe(
                prompt=edit["prompt"],
                negative_prompt=edit.get(
                    "negative", "low quality, blurry"
                ),
                image=current,
                mask_image=edit["mask"],
                num_inference_steps=edit.get("steps", 30),
                guidance_scale=edit.get("cfg", 7.5),
                strength=edit.get("strength", 0.85),
            ).images[0]

        return current


# 使用例: 複数領域編集
inpainter = AdvancedInpainter()
image = Image.open("room.jpg").resize((1024, 1024))

gen = MaskGenerator(1024, 1024)

# ソファを交換 + 壁の色を変更
result = inpainter.multi_region_edit(
    image,
    edits=[
        {
            "mask": gen.rectangular((100, 400, 600, 800),
                                     feather=15),
            "prompt": "modern minimalist white sofa, "
                      "interior design, natural lighting",
            "strength": 0.85,
        },
        {
            "mask": gen.rectangular((0, 0, 1024, 300),
                                     feather=20),
            "prompt": "warm beige painted wall, "
                      "matching the interior style",
            "strength": 0.6,
        },
    ],
)
result.save("room_multi_edit.png")
```

---

## 2. アウトペインティング

### コード例5: アウトペインティング（拡張版）

```python
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image, ImageDraw, ImageFilter
import torch
import numpy as np
from typing import Literal


class OutpaintingEngine:
    """
    画像の境界を超えた拡張を行うエンジン

    特徴:
    - 4方向の拡張
    - 段階的な拡張で高品質を維持
    - 重複帯の自動処理
    - パノラマ生成
    """

    def __init__(self, model_id: str = None):
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id or
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipe.enable_model_cpu_offload()

    def extend(
        self,
        image: Image.Image,
        direction: Literal["left", "right", "up", "down"],
        extend_pixels: int = 256,
        overlap: int = 64,
        prompt: str = "seamless continuation of the scene",
        negative_prompt: str = "seam, border, inconsistent",
        steps: int = 30,
        strength: float = 0.85,
    ) -> Image.Image:
        """
        画像を指定方向に拡張する

        Args:
            image: 元画像
            direction: 拡張方向
            extend_pixels: 拡張するピクセル数
            overlap: 重複帯のピクセル数（境界のシームレス化）
            prompt: 生成プロンプト
        """
        w, h = image.size

        # 方向別の設定
        configs = {
            "right": {
                "canvas_size": (w + extend_pixels, h),
                "paste_pos": (0, 0),
                "mask_box": (w - overlap, 0,
                             w + extend_pixels, h),
            },
            "left": {
                "canvas_size": (w + extend_pixels, h),
                "paste_pos": (extend_pixels, 0),
                "mask_box": (0, 0, extend_pixels + overlap, h),
            },
            "down": {
                "canvas_size": (w, h + extend_pixels),
                "paste_pos": (0, 0),
                "mask_box": (0, h - overlap,
                             w, h + extend_pixels),
            },
            "up": {
                "canvas_size": (w, h + extend_pixels),
                "paste_pos": (0, extend_pixels),
                "mask_box": (0, 0, w, extend_pixels + overlap),
            },
        }

        cfg = configs[direction]

        # キャンバス作成
        canvas = Image.new("RGB", cfg["canvas_size"], (128, 128, 128))
        canvas.paste(image, cfg["paste_pos"])

        # マスク作成（グラデーション付き）
        mask = self._create_gradient_mask(
            cfg["canvas_size"], cfg["mask_box"],
            direction, overlap
        )

        # 生成サイズに調整（SDXL最適サイズ）
        target_size = (1024, 1024)
        canvas_resized = canvas.resize(target_size)
        mask_resized = mask.resize(target_size)

        # インペインティングで拡張
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=canvas_resized,
            mask_image=mask_resized,
            num_inference_steps=steps,
            guidance_scale=7.5,
            strength=strength,
        ).images[0]

        # 元のサイズに戻す
        return result.resize(cfg["canvas_size"])

    def _create_gradient_mask(
        self,
        size: tuple[int, int],
        mask_box: tuple[int, int, int, int],
        direction: str,
        overlap: int,
    ) -> Image.Image:
        """重複帯にグラデーションを付けたマスクを生成"""
        w, h = size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(mask_box, fill=255)

        # 重複帯にグラデーションを適用
        mask_arr = np.array(mask, dtype=float)

        if direction == "right":
            x_start = mask_box[0]
            for x in range(overlap):
                mask_arr[:, x_start + x] = (x / overlap) * 255
        elif direction == "left":
            x_end = mask_box[2]
            for x in range(overlap):
                mask_arr[:, x_end - x] = (x / overlap) * 255
        elif direction == "down":
            y_start = mask_box[1]
            for y in range(overlap):
                mask_arr[y_start + y, :] = (y / overlap) * 255
        elif direction == "up":
            y_end = mask_box[3]
            for y in range(overlap):
                mask_arr[y_end - y, :] = (y / overlap) * 255

        return Image.fromarray(mask_arr.astype(np.uint8))

    def create_panorama(
        self,
        image: Image.Image,
        prompt: str,
        extensions: int = 2,
        extend_pixels: int = 512,
        overlap: int = 96,
    ) -> Image.Image:
        """
        パノラマ画像を生成（左右に段階的に拡張）

        Args:
            image: 中央の基準画像
            prompt: シーンの説明
            extensions: 片方向の拡張回数
            extend_pixels: 各拡張のピクセル数
            overlap: 重複帯
        """
        current = image

        # 右に拡張
        for i in range(extensions):
            print(f"Extending right {i+1}/{extensions}...")
            current = self.extend(
                current, "right",
                extend_pixels=extend_pixels,
                overlap=overlap,
                prompt=f"{prompt}, seamless continuation",
            )

        # 左に拡張
        for i in range(extensions):
            print(f"Extending left {i+1}/{extensions}...")
            current = self.extend(
                current, "left",
                extend_pixels=extend_pixels,
                overlap=overlap,
                prompt=f"{prompt}, seamless continuation",
            )

        return current


# 使用例
engine = OutpaintingEngine()
image = Image.open("landscape.jpg").resize((1024, 1024))

# 右に拡張
extended = engine.extend(
    image, "right",
    extend_pixels=512,
    prompt="continuation of mountain landscape, "
           "same lighting and style, autumn foliage",
)

# パノラマ生成
panorama = engine.create_panorama(
    image,
    prompt="vast mountain landscape with autumn trees "
           "and clear blue sky",
    extensions=3,
    extend_pixels=512,
)
panorama.save("panorama.png")
```

### ASCII図解2: アウトペインティングの方向と重複領域

```
元画像からの拡張方向:

         ↑ (up)
    ┌──────────┐
    │ 新規生成  │
    │          │
    ├══════════┤ ← 重複帯 (ブレンド)
    │          │
    │ 元画像   │ → (right)  ┌──────┐
    │          │  ┌─┤ 新規  │
 ← │          │  │重│ 生成  │
    │          │  │複│       │
    └──────────┘  └─┴──────┘
         ↓ (down)

重複帯のグラデーション処理:
┌────────┬──────────────────┬──────────┐
│ 元画像  │   ブレンド帯     │ 新規生成  │
│ (保持)  │   (グラデーション)│ (AI生成)  │
│ 100%   │   100%  →  0%    │ 0% → 100%│
│ 元の    │   元画像の比率    │ 生成     │
│ ピクセル│   段階的に低下    │ ピクセル  │
└────────┴──────────────────┴──────────┘

グラデーションマスクの断面:
元画像側                    新規生成側
255 ████████▓▓▓▓▒▒▒▒░░░░          0
     保護    ←─グラデーション─→  編集

パノラマ生成 (連続アウトペインティング):
┌────┬────┬────┬────┬────┐
│ ←  │ ←  │元画│ →  │ →  │
│拡張3│拡張2│ 像 │拡張1│拡張2│
└────┴────┴────┴────┴────┘
= 元の5倍幅のパノラマ画像

各拡張ステップ:
  Step 1: [元画像][→拡張1]
  Step 2: [元画像][拡張1][→拡張2]
  Step 3: [←拡張1][元画像][拡張1][拡張2]
  Step 4: [←拡張2][拡張1][元画像][拡張1][拡張2]
```

---

## 3. Img2Img と ControlNet

### コード例6: Img2Img によるスタイル変換（拡張版）

```python
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import torch
from dataclasses import dataclass


@dataclass
class StylePreset:
    """スタイル変換のプリセット"""
    name: str
    prompt_template: str
    negative_prompt: str
    strength: float
    guidance_scale: float


STYLE_PRESETS = {
    "油絵_印象派": StylePreset(
        name="印象派油絵",
        prompt_template="{subject}, oil painting style, impressionist, "
                        "Claude Monet, visible brushstrokes, "
                        "vibrant colors, natural lighting",
        negative_prompt="photorealistic, sharp, digital, "
                        "smooth, flat colors",
        strength=0.65,
        guidance_scale=7.5,
    ),
    "水彩画": StylePreset(
        name="水彩画",
        prompt_template="{subject}, watercolor painting, soft edges, "
                        "color bleeding, delicate, paper texture, "
                        "transparent washes",
        negative_prompt="photorealistic, digital, sharp edges, "
                        "heavy colors, oil painting",
        strength=0.60,
        guidance_scale=7.0,
    ),
    "アニメ": StylePreset(
        name="アニメ化",
        prompt_template="{subject}, anime style, cel shading, "
                        "vibrant colors, detailed, "
                        "Studio Ghibli quality",
        negative_prompt="realistic, photographic, 3d render, "
                        "low quality, blurry",
        strength=0.70,
        guidance_scale=8.0,
    ),
    "サイバーパンク": StylePreset(
        name="サイバーパンク",
        prompt_template="{subject}, cyberpunk style, neon lights, "
                        "futuristic, dark atmosphere, "
                        "rain-soaked streets, holographic elements",
        negative_prompt="natural, daylight, vintage, "
                        "low quality, blurry",
        strength=0.75,
        guidance_scale=8.0,
    ),
    "ピクセルアート": StylePreset(
        name="ピクセルアート",
        prompt_template="{subject}, pixel art, 16-bit style, "
                        "retro game graphics, limited palette, "
                        "crisp pixels",
        negative_prompt="photorealistic, smooth, high resolution, "
                        "anti-aliased, blurry",
        strength=0.80,
        guidance_scale=9.0,
    ),
    "鉛筆スケッチ": StylePreset(
        name="鉛筆スケッチ",
        prompt_template="{subject}, pencil sketch, graphite drawing, "
                        "detailed hatching, paper texture, "
                        "monochrome, artistic",
        negative_prompt="colorful, photorealistic, digital, "
                        "painting, low quality",
        strength=0.55,
        guidance_scale=7.0,
    ),
}


class StyleTransformer:
    """スタイル変換エンジン"""

    def __init__(self):
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipe.enable_model_cpu_offload()

    def transform(
        self,
        image: Image.Image,
        style: str,
        subject_description: str = "the scene",
        custom_strength: float = None,
        seed: int = None,
    ) -> Image.Image:
        """
        プリセットスタイルで画像を変換

        Args:
            image: 入力画像
            style: プリセット名
            subject_description: 画像の内容説明
            custom_strength: カスタム強度（省略時はプリセット値）
            seed: 乱数シード
        """
        preset = STYLE_PRESETS.get(style)
        if not preset:
            raise ValueError(
                f"Unknown style: {style}. "
                f"Available: {list(STYLE_PRESETS.keys())}"
            )

        prompt = preset.prompt_template.replace(
            "{subject}", subject_description
        )

        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=preset.negative_prompt,
            image=image.resize((1024, 1024)),
            strength=custom_strength or preset.strength,
            num_inference_steps=30,
            guidance_scale=preset.guidance_scale,
            generator=generator,
        ).images[0]

        return result

    def compare_styles(
        self,
        image: Image.Image,
        styles: list[str],
        subject_description: str = "the scene",
        seed: int = 42,
    ) -> dict[str, Image.Image]:
        """複数スタイルを同一シードで比較"""
        results = {}
        for style in styles:
            results[style] = self.transform(
                image, style, subject_description, seed=seed,
            )
        return results

    def progressive_transform(
        self,
        image: Image.Image,
        style: str,
        subject_description: str = "the scene",
        strengths: list[float] = None,
    ) -> list[Image.Image]:
        """
        異なるstrengthで段階的に変換し、最適値を探す

        Returns:
            strength値ごとの変換結果リスト
        """
        if strengths is None:
            strengths = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        results = []
        for s in strengths:
            result = self.transform(
                image, style, subject_description,
                custom_strength=s, seed=42,
            )
            results.append(result)

        return results


# 使用例
transformer = StyleTransformer()
photo = Image.open("photo.jpg")

# 印象派スタイルに変換
monet = transformer.transform(
    photo, "油絵_印象派",
    subject_description="a garden with flowers and a pond",
)
monet.save("monet_style.png")

# 複数スタイルを比較
comparisons = transformer.compare_styles(
    photo,
    styles=["油絵_印象派", "水彩画", "アニメ", "鉛筆スケッチ"],
    subject_description="a garden with flowers",
)
for style_name, result in comparisons.items():
    result.save(f"comparison_{style_name}.png")
```

### コード例7: ControlNet による精密制御（拡張版）

```python
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    UniPCMultistepScheduler,
)
from controlnet_aux import (
    CannyDetector,
    OpenposeDetector,
    MidasDetector,
    HEDdetector,
    LineartDetector,
)
from PIL import Image
import torch
import numpy as np


class ControlNetEditor:
    """
    ControlNet を使った精密な画像編集

    対応する制御タイプ:
    - Canny Edge: 輪郭線による構図制御
    - OpenPose: 人体ポーズ制御
    - Depth: 奥行き情報による空間制御
    - Scribble/HED: ラフスケッチからの生成
    - Lineart: 線画からの着色
    """

    CONTROLNET_MODELS = {
        "canny": "diffusers/controlnet-canny-sdxl-1.0",
        "depth": "diffusers/controlnet-depth-sdxl-1.0",
        "openpose": "thibaud/controlnet-openpose-sdxl-1.0",
    }

    def __init__(self, control_type: str = "canny"):
        self.control_type = control_type

        # ControlNet モデルをロード
        controlnet = ControlNetModel.from_pretrained(
            self.CONTROLNET_MODELS[control_type],
            torch_dtype=torch.float16,
        )

        # パイプラインを構築
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        ).to("cuda")

        # スケジューラを高速なものに変更
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.enable_model_cpu_offload()

        # プリプロセッサの初期化
        self.preprocessors = {
            "canny": CannyDetector(),
            "depth": MidasDetector.from_pretrained(
                "lllyasviel/Annotators"
            ),
            "openpose": OpenposeDetector.from_pretrained(
                "lllyasviel/Annotators"
            ),
        }

    def preprocess(self, image: Image.Image,
                    **kwargs) -> Image.Image:
        """制御画像を前処理"""
        preprocessor = self.preprocessors.get(self.control_type)
        if preprocessor is None:
            raise ValueError(
                f"Preprocessor for {self.control_type} not found"
            )

        if self.control_type == "canny":
            return preprocessor(
                image,
                low_threshold=kwargs.get("low_threshold", 100),
                high_threshold=kwargs.get("high_threshold", 200),
            )
        elif self.control_type == "depth":
            return preprocessor(image)
        elif self.control_type == "openpose":
            return preprocessor(image)

        return preprocessor(image)

    def generate(
        self,
        control_image: Image.Image,
        prompt: str,
        negative_prompt: str = "low quality, blurry",
        conditioning_scale: float = 0.8,
        guidance_scale: float = 7.5,
        steps: int = 25,
        seed: int = None,
    ) -> Image.Image:
        """
        制御画像に基づいて画像を生成

        Args:
            control_image: 前処理済みの制御画像
            prompt: 生成プロンプト
            conditioning_scale: 制御の強さ (0.0-2.0)
                0.0: 制御なし（通常のtxt2img相当）
                0.5: 緩い制御
                0.8: 標準的な制御（推奨）
                1.0: 厳密な制御
                1.5+: 過剰制御（アーティファクト注意）
        """
        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            controlnet_conditioning_scale=conditioning_scale,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        return result

    def edit_with_original(
        self,
        original_image: Image.Image,
        prompt: str,
        conditioning_scale: float = 0.8,
        **kwargs,
    ) -> dict:
        """
        元画像から制御画像を抽出して新しい画像を生成

        Returns:
            dict: {
                "control": 制御画像,
                "result": 生成結果,
            }
        """
        # 前処理で制御画像を抽出
        control = self.preprocess(original_image, **kwargs)

        # 制御画像から新しい画像を生成
        result = self.generate(
            control, prompt,
            conditioning_scale=conditioning_scale,
        )

        return {
            "control": control,
            "result": result,
        }


# 使用例: 建物のスタイル変更
editor = ControlNetEditor(control_type="canny")
building = Image.open("building.jpg")

# エッジ抽出 → 新しいスタイルで生成
output = editor.edit_with_original(
    building,
    prompt="futuristic glass and steel building, "
           "sci-fi architecture, night city, "
           "neon reflections, cyberpunk",
    conditioning_scale=0.85,
    low_threshold=80,
    high_threshold=180,
)
output["control"].save("building_edges.png")
output["result"].save("building_futuristic.png")

# ポーズ制御
pose_editor = ControlNetEditor(control_type="openpose")
reference = Image.open("pose_reference.jpg")

output = pose_editor.edit_with_original(
    reference,
    prompt="professional ballet dancer in white tutu, "
           "elegant pose, stage lighting, "
           "photorealistic, high quality",
    conditioning_scale=0.9,
)
output["result"].save("ballet_dancer.png")
```

### ASCII図解3: ControlNet の制御タイプ一覧

```
┌─────── ControlNet 制御タイプ ───────────────────────┐
│                                                     │
│  ┌── エッジ系 ──┐  ┌── 深度系 ──┐  ┌── ポーズ ─┐  │
│  │ Canny Edge   │  │ Depth Map  │  │ OpenPose  │  │
│  │ ┌──┐         │  │ ┌──┐       │  │  O        │  │
│  │ │/\│         │  │ │濃│       │  │ /|\       │  │
│  │ │\/│ 輪郭線  │  │ │淡│ 遠近  │  │ / \  骨格 │  │
│  │ └──┘         │  │ └──┘       │  │           │  │
│  │ scale: 0.8   │  │ scale: 0.7 │  │ scale: 0.9│  │
│  └──────────────┘  └────────────┘  └───────────┘  │
│                                                     │
│  ┌── セグメント ─┐  ┌── 線画 ───┐  ┌── 法線 ──┐  │
│  │ Segmentation │  │ Scribble  │  │ Normal   │  │
│  │ ┌──┐         │  │  /~~\     │  │  →→→     │  │
│  │ │色│ 領域    │  │ |    | 手 │  │  →→→ 表面│  │
│  │ │分│ 分離    │  │  \__/ 描き│  │  →→→ 向き│  │
│  │ └──┘         │  │           │  │          │  │
│  │ scale: 0.7   │  │ scale: 0.6│  │ scale: 0.5│  │
│  └──────────────┘  └───────────┘  └──────────┘  │
│                                                     │
│  ┌── タイル ───┐   ┌── IP-Adapter ──┐             │
│  │ Tile        │   │ 画像ベース      │             │
│  │ ┌┬┬┐       │   │ 参照画像の      │             │
│  │ ├┼┼┤ 高解像│   │ スタイル/構図   │             │
│  │ ├┼┼┤ 度制御│   │ を転写          │             │
│  │ └┴┴┘       │   │                │             │
│  │ scale: 0.5  │   │ scale: 0.6-1.0 │             │
│  └─────────────┘   └────────────────┘             │
└─────────────────────────────────────────────────────┘

conditioning_scale の影響:
┌──────────────────────────────────────────────────┐
│ Scale:  0.0   0.3   0.5   0.8   1.0   1.5   2.0│
│ 影響:  なし   弱い  中程度  標準  厳密  過剰  破綻│
│ 自由度: 高い ←──────────────────────→ 低い        │
│ 忠実度: 低い ←──────────────────────→ 高い        │
│                                                  │
│ 推奨範囲: |........[===推奨===].........|         │
│                   0.5  0.7  0.9                  │
└──────────────────────────────────────────────────┘
```

---

## 4. Instruct-Pix2Pix: 自然言語による画像編集

### コード例8: InstructPix2Pix

```python
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
import torch


class TextInstructEditor:
    """
    テキスト指示で画像を編集する InstructPix2Pix ラッパー

    特徴:
    - マスク不要（テキスト指示だけで編集可能）
    - 直感的な自然言語による指示
    - image_guidance_scale で元画像の維持度を制御
    """

    def __init__(self):
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipe.enable_model_cpu_offload()

    def edit(
        self,
        image: Image.Image,
        instruction: str,
        image_guidance_scale: float = 1.5,
        guidance_scale: float = 7.5,
        steps: int = 30,
        seed: int = None,
    ) -> Image.Image:
        """
        テキスト指示で画像を編集

        Args:
            image: 入力画像
            instruction: 編集指示
                例: "Make it winter"
                    "Turn the car red"
                    "Add sunglasses"
                    "Make it look like a painting"
            image_guidance_scale: 元画像の維持度
                1.0: 元画像を弱く参照
                1.5: 標準（推奨）
                2.0: 元画像を強く維持
            guidance_scale: テキスト指示の従い度
        """
        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)

        result = self.pipe(
            instruction,
            image=image,
            num_inference_steps=steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        return result

    def batch_edit(
        self,
        image: Image.Image,
        instructions: list[str],
        seed: int = 42,
    ) -> list[Image.Image]:
        """複数の指示を同一画像に適用して比較"""
        return [
            self.edit(image, inst, seed=seed)
            for inst in instructions
        ]

    def chain_edits(
        self,
        image: Image.Image,
        instructions: list[str],
    ) -> list[Image.Image]:
        """
        複数の指示を順番に適用（チェーン編集）

        例: ["Make it sunset", "Add rain", "Make it oil painting"]
        → 夕焼け化 → 雨追加 → 油絵化 を順番に適用
        """
        results = []
        current = image

        for inst in instructions:
            current = self.edit(current, inst)
            results.append(current)

        return results


# 使用例
editor = TextInstructEditor()
image = Image.open("photo.jpg")

# 簡単な編集指示
winter = editor.edit(image, "Make it a snowy winter scene")
winter.save("winter_version.png")

# 段階的な変換
stages = editor.chain_edits(
    image,
    [
        "Make it sunset with golden light",
        "Add dramatic clouds in the sky",
        "Make it look like an oil painting",
    ],
)
for i, stage in enumerate(stages):
    stage.save(f"stage_{i+1}.png")
```

### ASCII図解4: InstructPix2Pix のパラメータバランス

```
image_guidance_scale vs guidance_scale のバランス:

                  guidance_scale (テキスト従い度)
                  低い(3)    中(7.5)     高い(15)
              ┌──────────┬──────────┬──────────┐
  image_      │ 変化なし  │ 部分変化  │ 大きく   │
  guidance    │ (両方弱い)│          │ 変化     │
  低い(1.0)   │          │          │          │
              ├──────────┼──────────┼──────────┤
  scale       │ 微細な   │ ★最適★  │ 過度な   │
  (元画像     │ 変化     │ バランス  │ 変化     │
  維持度)     │          │          │          │
  中(1.5)     ├──────────┼──────────┼──────────┤
              │ ほぼ     │ 控えめな │ 矛盾した │
  高い(2.0)   │ 変化なし │ 変化     │ 結果     │
              └──────────┴──────────┴──────────┘

推奨設定:
  通常の編集:      image_guidance=1.5, guidance=7.5
  微細な調整:      image_guidance=2.0, guidance=5.0
  大胆な変換:      image_guidance=1.0, guidance=10.0
  スタイル変換:    image_guidance=1.2, guidance=8.0
```

---

## 5. バッチ処理と本番ワークフロー

### コード例9: 商用バッチ編集パイプライン

```python
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image
import torch


@dataclass
class EditJob:
    """編集ジョブの定義"""
    job_id: str
    input_path: str
    output_path: str
    edit_type: str  # "inpaint", "outpaint", "style", "instruct"
    prompt: str
    negative_prompt: str = ""
    mask_path: Optional[str] = None
    strength: float = 0.85
    seed: int = 42
    status: str = "pending"
    error: Optional[str] = None
    processing_time: float = 0.0


class BatchEditPipeline:
    """
    本番環境向けバッチ画像編集パイプライン

    機能:
    - ジョブキュー管理
    - エラーハンドリングとリトライ
    - 進捗レポート
    - 結果の品質チェック
    """

    def __init__(self, output_dir: str = "./batch_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: list[EditJob] = []
        self.completed: list[EditJob] = []
        self.failed: list[EditJob] = []

    def add_job(self, **kwargs) -> EditJob:
        """ジョブを追加"""
        job = EditJob(**kwargs)
        self.jobs.append(job)
        return job

    def add_bulk_style_transfer(
        self,
        input_dir: str,
        style: str,
        prompt_template: str,
    ):
        """ディレクトリ内の全画像にスタイル変換を適用"""
        input_path = Path(input_dir)
        for i, img_path in enumerate(
            sorted(input_path.glob("*.{jpg,png,jpeg}"))
        ):
            self.add_job(
                job_id=f"style_{i:04d}",
                input_path=str(img_path),
                output_path=str(
                    self.output_dir / f"{style}_{img_path.name}"
                ),
                edit_type="style",
                prompt=prompt_template,
                strength=0.65,
            )

    def process_all(
        self,
        max_retries: int = 2,
        on_progress: callable = None,
    ) -> dict:
        """全ジョブを処理"""
        total = len(self.jobs)
        start_time = time.time()

        for i, job in enumerate(self.jobs):
            if on_progress:
                on_progress(i + 1, total, job.job_id)

            success = False
            for attempt in range(max_retries + 1):
                try:
                    self._process_single(job)
                    job.status = "completed"
                    self.completed.append(job)
                    success = True
                    break
                except Exception as e:
                    job.error = str(e)
                    if attempt < max_retries:
                        print(f"  Retry {attempt + 1} for {job.job_id}")
                        torch.cuda.empty_cache()
                    else:
                        job.status = "failed"
                        self.failed.append(job)

        total_time = time.time() - start_time

        report = {
            "total_jobs": total,
            "completed": len(self.completed),
            "failed": len(self.failed),
            "total_time_seconds": total_time,
            "avg_time_per_job": total_time / max(total, 1),
            "failed_jobs": [
                {"id": j.job_id, "error": j.error}
                for j in self.failed
            ],
        }

        # レポートを保存
        report_path = self.output_dir / "batch_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return report

    def _process_single(self, job: EditJob):
        """単一ジョブを処理"""
        start = time.time()
        image = Image.open(job.input_path).convert("RGB")

        if job.edit_type == "inpaint" and job.mask_path:
            mask = Image.open(job.mask_path).convert("L")
            result = self._inpaint(image, mask, job)
        elif job.edit_type == "style":
            result = self._style_transfer(image, job)
        elif job.edit_type == "instruct":
            result = self._instruct_edit(image, job)
        else:
            raise ValueError(f"Unknown edit type: {job.edit_type}")

        result.save(job.output_path)
        job.processing_time = time.time() - start

    def _inpaint(self, image, mask, job):
        """インペインティング処理"""
        from diffusers import StableDiffusionXLInpaintPipeline
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
        ).to("cuda")
        return pipe(
            prompt=job.prompt,
            negative_prompt=job.negative_prompt,
            image=image.resize((1024, 1024)),
            mask_image=mask.resize((1024, 1024)),
            strength=job.strength,
            num_inference_steps=30,
        ).images[0]

    def _style_transfer(self, image, job):
        """スタイル変換処理"""
        from diffusers import StableDiffusionXLImg2ImgPipeline
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
        ).to("cuda")
        return pipe(
            prompt=job.prompt,
            negative_prompt=job.negative_prompt,
            image=image.resize((1024, 1024)),
            strength=job.strength,
            num_inference_steps=30,
        ).images[0]

    def _instruct_edit(self, image, job):
        """テキスト指示編集処理"""
        from diffusers import StableDiffusionInstructPix2PixPipeline
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16,
        ).to("cuda")
        return pipe(
            job.prompt,
            image=image,
            num_inference_steps=30,
        ).images[0]


# 使用例
pipeline = BatchEditPipeline("./output/batch_results")

# 複数の編集ジョブを登録
pipeline.add_job(
    job_id="room_sofa",
    input_path="room.jpg",
    output_path="./output/room_new_sofa.png",
    edit_type="inpaint",
    mask_path="sofa_mask.png",
    prompt="modern minimalist white sofa, "
           "matching the room decor",
)

pipeline.add_job(
    job_id="photo_style",
    input_path="landscape.jpg",
    output_path="./output/landscape_watercolor.png",
    edit_type="style",
    prompt="watercolor painting, soft washes, "
           "paper texture, delicate colors",
    strength=0.6,
)

# 全ジョブを実行
report = pipeline.process_all(
    on_progress=lambda cur, total, jid:
        print(f"[{cur}/{total}] Processing {jid}...")
)
print(f"完了: {report['completed']}, 失敗: {report['failed']}")
```

---

## 6. 比較表

### 比較表1: 画像編集手法の比較

| 手法 | 入力 | 制御精度 | 用途 | 計算コスト | マスク要否 |
|------|------|---------|------|-----------|----------|
| **インペインティング** | 画像 + マスク + プロンプト | 高 | 部分置換・修正 | 中 | 必要 |
| **アウトペインティング** | 画像 + 方向指定 | 中 | 画像拡張 | 中 | 自動生成 |
| **Img2Img** | 画像 + プロンプト + strength | 中 | スタイル変換 | 中 | 不要 |
| **ControlNet** | 制御画像 + プロンプト | 非常に高 | 構図制御生成 | 高 | 不要 |
| **IP-Adapter** | 参照画像 + プロンプト | 中 | スタイル転写 | 中 | 不要 |
| **Instruct-Pix2Pix** | 画像 + 編集指示 | 中 | 自然言語編集 | 中 | 不要 |
| **Grounding DINO + SAM** | 画像 + テキスト | 高 | 自動マスク生成 | 中 | - |

### 比較表2: マスク生成手法の比較

| 手法 | 精度 | 自動化 | ツール | 適用シーン |
|------|------|--------|--------|-----------|
| **手動描画** | 最高 | 手動 | Photoshop, GIMP, WebUI | 複雑な形状 |
| **矩形/楕円** | 低 | 自動 | PIL/Pillow | 簡単な領域 |
| **SAM (セグメンテーション)** | 高 | 半自動 | segment-anything | 物体単位の選択 |
| **色範囲選択** | 中 | 自動 | OpenCV | 単色背景 |
| **テキスト指定** | 中~高 | 自動 | Grounding DINO + SAM | 意味的な選択 |
| **深度ベース** | 中 | 自動 | MiDaS + 閾値処理 | 前景/背景の分離 |
| **グラデーション** | - | 自動 | NumPy/PIL | アウトペインティング |

### 比較表3: strength パラメータの影響

| strength 範囲 | 変化の程度 | 用途 | 元画像の維持 |
|--------------|-----------|------|------------|
| 0.1 - 0.2 | 微細 | 色調補正、ノイズ除去 | 95%+ |
| 0.2 - 0.3 | 軽微 | 微細なスタイル調整 | 85-95% |
| 0.3 - 0.5 | 中程度 | スタイル変換（構図維持） | 60-85% |
| 0.5 - 0.7 | 大きい | スタイル変換（大幅変化） | 30-60% |
| 0.7 - 0.85 | 非常に大きい | ほぼ新規生成 | 10-30% |
| 0.85 - 1.0 | 完全再生成 | 構図参考のみ | 0-10% |

### 比較表4: ControlNet タイプ別の適用シーン

| ControlNet タイプ | 最適なシーン | conditioning_scale 推奨 | 前処理の計算量 |
|------------------|------------|----------------------|--------------|
| Canny Edge | 建築、プロダクト、構造物 | 0.7 - 0.9 | 低 |
| Depth | 風景、室内、空間構成 | 0.6 - 0.8 | 中 |
| OpenPose | 人物ポーズ、ダンス、スポーツ | 0.8 - 1.0 | 中 |
| Scribble | ラフスケッチからの生成 | 0.5 - 0.7 | 低 |
| Lineart | 線画の着色、漫画 | 0.6 - 0.8 | 中 |
| Tile | 高解像度化、ディテール追加 | 0.4 - 0.6 | 低 |
| Segmentation | 領域ごとのスタイル制御 | 0.6 - 0.8 | 高 |
| Normal Map | 3D的な表面制御 | 0.4 - 0.6 | 中 |

---

## 7. アンチパターン

### アンチパターン1: マスクの境界が鮮明すぎる

```
[問題]
マスクの境界をピクセル単位でくっきり作成し、
インペインティング結果に「貼り付けた」ような不自然さが出る。

[なぜ問題か]
- 鮮明な境界では元画像と生成部分の色調・テクスチャが不連続
- 光の当たり方や影が境界で急に変わる
- 人間の目は不連続性に非常に敏感

[正しいアプローチ]
- マスクにガウスぼかし (10-30px) を適用してフェザリング
- 編集領域を実際の対象より少し大きめに設定（10-20px膨張）
- strength パラメータで境界のブレンドを調整
- 後処理でさらに境界をブレンド
- MaskGenerator の feather パラメータを活用
```

### アンチパターン2: Img2Img の strength を理解せずに使う

```
[問題]
Img2Img で strength=1.0 を設定し「元画像が全く反映されない」
と困惑する。逆に strength=0.1 で「何も変わらない」と不満。

[なぜ問題か]
- strength はノイズ付加のレベルを制御するパラメータ
- 1.0 = 完全にランダムノイズから開始 (≒txt2img)
- 0.0 = ノイズなし = 元画像そのまま
- 用途に合った range を理解する必要がある

[正しいアプローチ]
- 微調整 (色補正等): 0.2-0.3
- スタイル変換: 0.4-0.6
- 大きな変更: 0.7-0.8
- 元画像は参考程度: 0.85-0.95
- progressive_transform() で最適値を探す
```

### アンチパターン3: ControlNet の conditioning_scale が不適切

```
[問題]
conditioning_scale を 1.5 や 2.0 に設定して
アーティファクトだらけの結果になる。

[なぜ問題か]
- 高すぎる scale は制御信号を過剰に増幅
- モデルが制御画像に過適合し、不自然なパターンが出現
- 特にエッジ検出ベースでは線が二重になったりする

[正しいアプローチ]
- 0.5-0.9 の範囲で調整（0.8 が標準的）
- 制御タイプごとに最適値が異なる
- 低めから始めて段階的に上げる
- 制御の「緩さ」が自然な結果につながる場合が多い
```

### アンチパターン4: アウトペインティングの一度に大きな拡張

```
[問題]
512x512 の画像を一度に 2048x512 に拡張しようとして
品質が著しく低下する。

[なぜ問題か]
- 大きな生成領域ではコンテキストが希薄になる
- モデルの学習解像度を大きく超える
- 境界の一貫性が保てない

[正しいアプローチ]
- 256-512px ずつ段階的に拡張
- 十分な重複帯 (64-96px) を確保
- 各拡張後に全体を Img2Img で軽く統一
- OutpaintingEngine.create_panorama() を使用
```

### アンチパターン5: InstructPix2Pix の曖昧な指示

```
[問題]
「もっと良くして」「きれいにして」のような
曖昧な編集指示を与える。

[なぜ問題か]
- モデルは具体的な変更を期待している
- 「良い」「きれい」は主観的で解釈が不定
- 結果が不安定になり、意図しない変更が入る

[正しいアプローチ]
- 具体的な変更を指示: "Make the sky more orange"
- 動詞 + 対象 + 変更内容: "Turn the car from blue to red"
- 1つの指示に1つの変更に絞る
- chain_edits() で段階的に適用
```

---

## 8. FAQ

### Q1: インペインティングで周囲と色味が合わない場合は?

**A:** 以下の順序で対処します:

1. **マスクを拡大する:** 境界を 20-50px 広げて周囲のコンテキストを含める
2. **フェザリング:** マスクに 15-25px のガウスぼかしを適用
3. **strength を下げる:** 0.7-0.8 程度にして元画像の色調を保持
4. **プロンプトに色調を明記:** "matching ambient lighting, same color temperature, consistent shadows"
5. **二段階処理:** 粗い生成 (strength=0.9) → 微調整 (strength=0.4)
6. **コンテキストプロンプト:** AdvancedInpainter の context_prompt を活用

### Q2: アウトペインティングで一貫性を保つコツは?

**A:**

- **重複領域を十分に取る:** 最低 64px、推奨 96px 以上
- **グラデーションマスクを使用:** OutpaintingEngine の gradient mask
- **同じプロンプトを使用:** 元画像の説明 + "seamless continuation"
- **一度に大きく拡張しない:** 256-512px ずつ段階的に拡張
- **シード固定:** 同じランダムシードで一貫性を保つ
- **後処理:** 拡張結果全体に Img2Img を軽く (strength=0.2) 適用して統一

### Q3: ControlNet はどのタイプを選ぶべき?

**A:** タスクに応じて選択します:

- **建築・インテリア:** Canny Edge (輪郭維持) + Depth (奥行き)
- **人物ポーズ指定:** OpenPose (骨格制御)
- **既存画像の高品質化:** Tile (ディテール維持)
- **手描きスケッチから:** Scribble (ラフな線画)
- **線画の着色:** Lineart (輪郭線のみ抽出)
- **スタイル転写:** IP-Adapter (参照画像ベース)
- **複数制御:** Multi-ControlNet で複数タイプを同時使用可能

### Q4: InstructPix2Pix と Img2Img + プロンプトの違いは?

**A:** 用途が異なります:

- **InstructPix2Pix:** 「変更の指示」を理解する。"Make it rainy" のように差分を記述。マスク不要で手軽。元画像の構造を保ちやすい。
- **Img2Img:** 「最終状態」を記述する。"A rainy landscape" のように結果を記述。strength で変化量を制御。より大きな変化に向いている。
- **使い分けの基準:** 部分的・微細な変更には InstructPix2Pix、スタイル全体の変換には Img2Img が適している。

### Q5: 商用利用での品質管理はどうすべき?

**A:** 以下のワークフローを推奨:

1. **テスト生成:** 複数シード(5+)で生成し、品質の安定性を確認
2. **人間レビュー:** 自動生成後に必ず人間がチェック
3. **品質指標:** CLIP Score や SSIM で定量的に評価
4. **バッチ処理:** BatchEditPipeline でジョブ管理し、エラーハンドリング
5. **バージョン管理:** 入力画像、マスク、プロンプト、シード、結果を全て記録
6. **A/Bテスト:** パラメータの異なる結果を比較して最適設定を見つける

### Q6: GPU メモリが足りない場合の対処法は?

**A:** 段階的に最適化します:

1. **enable_model_cpu_offload():** 使用中でないモデルをCPUに退避
2. **enable_vae_tiling():** VAEのタイリング処理
3. **torch.float16 (FP16):** 半精度浮動小数点の使用
4. **画像サイズの削減:** 1024x1024 → 768x768 → 512x512
5. **バッチサイズ=1:** 一度に1枚ずつ処理
6. **xformers:** メモリ効率の良い attention 実装
7. **torch.compile():** PyTorch 2.0+ のコンパイル最適化

---

## 9. まとめ表

| 項目 | 要点 |
|------|------|
| **インペインティング** | マスク+プロンプトで部分書き換え。フェザリング必須 |
| **マスク膨張** | 対象より10-20px大きくマスクを作る |
| **反復改善** | strength高→中→低 の3段階で品質向上 |
| **複数領域** | multi_region_edit で順番に編集 |
| **アウトペインティング** | 画像境界を拡張。グラデーションマスクで自然な接続 |
| **パノラマ** | 256-512pxずつ段階的に拡張 |
| **Img2Img** | strength で元画像の維持度を制御 (0.0-1.0) |
| **スタイルプリセット** | 用途別に最適なstrengthとcfgを事前定義 |
| **ControlNet** | エッジ/深度/ポーズ等で精密に構図を制御 |
| **conditioning_scale** | 0.5-0.9が推奨。高すぎるとアーティファクト |
| **SAMマスク** | ポイントクリックやバウンディングボックスで自動生成 |
| **テキスト指定マスク** | Grounding DINO + SAM でテキストから自動生成 |
| **InstructPix2Pix** | テキスト指示のみで画像編集（マスク不要） |
| **バッチ処理** | BatchEditPipeline でジョブ管理、エラー処理、レポート |
| **メモリ最適化** | cpu_offload、vae_tiling、fp16の活用 |

---

## 次に読むべきガイド

- [02-upscaling.md](./02-upscaling.md) — 編集後のアップスケーリング
- [03-design-tools.md](./03-design-tools.md) — Canva AI、Adobe Firefly の編集機能
- [../02-video/01-video-editing.md](../02-video/01-video-editing.md) — 動画に対する同様の編集技術

---

## 参考文献

1. Lugmayr, A. et al. (2022). "RePaint: Inpainting using Denoising Diffusion Probabilistic Models." *CVPR 2022*. https://arxiv.org/abs/2201.09865
2. Zhang, L. et al. (2023). "Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)." *ICCV 2023*. https://arxiv.org/abs/2302.05543
3. Kirillov, A. et al. (2023). "Segment Anything." *ICCV 2023*. https://arxiv.org/abs/2304.02643
4. Brooks, T. et al. (2023). "InstructPix2Pix: Learning to Follow Image Editing Instructions." *CVPR 2023*. https://arxiv.org/abs/2211.09800
5. Liu, S. et al. (2023). "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection." *arXiv*. https://arxiv.org/abs/2303.05499
6. Ye, H. et al. (2023). "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models." *arXiv*. https://arxiv.org/abs/2308.06721
7. Mou, C. et al. (2023). "T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models." *AAAI 2024*. https://arxiv.org/abs/2302.08453

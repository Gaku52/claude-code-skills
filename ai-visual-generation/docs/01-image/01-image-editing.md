# 画像編集 — インペインティング、アウトペインティング

> AI による画像の部分修正・拡張技術を、マスク生成からシームレスな合成まで実践的に解説する。

---

## この章で学ぶこと

1. **インペインティングの原理と実装** — マスク領域の自然な補完、プロンプトによる部分書き換え
2. **アウトペインティングの技法** — 画像の境界を超えた拡張、パノラマ生成
3. **Img2Img と ControlNet による高度な編集** — 構図維持、スタイル変換、ポーズ制御

---

## 1. インペインティング

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

### コード例2: プログラマティックなマスク生成

```python
from PIL import Image, ImageDraw
import numpy as np

def create_rectangular_mask(width, height, bbox):
    """矩形マスクを生成"""
    mask = Image.new("L", (width, height), 0)  # 黒=保護領域
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bbox, fill=255)  # 白=編集領域
    return mask

def create_circular_mask(width, height, center, radius):
    """円形マスクを生成"""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    x, y = center
    draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=255)
    return mask

def create_mask_from_segmentation(image_path, target_class="person"):
    """セグメンテーションモデルでマスクを自動生成"""
    from transformers import pipeline

    segmenter = pipeline(
        "image-segmentation",
        model="facebook/sam-vit-huge",
        device=0,
    )

    image = Image.open(image_path)
    results = segmenter(image, points_per_batch=64)

    # 対象クラスのマスクを合成
    combined_mask = np.zeros(
        (image.height, image.width), dtype=np.uint8
    )
    for result in results:
        if target_class.lower() in result["label"].lower():
            mask_array = np.array(result["mask"])
            combined_mask = np.maximum(combined_mask, mask_array)

    return Image.fromarray(combined_mask)

# マスクのフェザリング (境界をぼかす)
def feather_mask(mask, radius=10):
    """マスク境界をガウスぼかしで滑らかに"""
    from PIL import ImageFilter
    return mask.filter(ImageFilter.GaussianBlur(radius))
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
```

---

## 2. アウトペインティング

### コード例3: アウトペインティング実装

```python
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch

def outpaint(image_path, direction="right", extend_pixels=256):
    """
    画像を指定方向に拡張する

    direction: "left", "right", "up", "down"
    extend_pixels: 拡張するピクセル数
    """
    original = Image.open(image_path)
    w, h = original.size

    # 拡張方向に応じた新キャンバスサイズ
    if direction == "right":
        new_w, new_h = w + extend_pixels, h
        paste_pos = (0, 0)
        mask_box = (w - 32, 0, new_w, new_h)  # 32px重複でシームレス化
    elif direction == "left":
        new_w, new_h = w + extend_pixels, h
        paste_pos = (extend_pixels, 0)
        mask_box = (0, 0, extend_pixels + 32, new_h)
    elif direction == "down":
        new_w, new_h = w, h + extend_pixels
        paste_pos = (0, 0)
        mask_box = (0, h - 32, new_w, new_h)
    elif direction == "up":
        new_w, new_h = w, h + extend_pixels
        paste_pos = (0, extend_pixels)
        mask_box = (0, 0, new_w, extend_pixels + 32)

    # 新しいキャンバス (拡張部分はノイズで初期化)
    canvas = Image.new("RGB", (new_w, new_h), (128, 128, 128))
    canvas.paste(original, paste_pos)

    # マスク生成 (白=生成する領域)
    mask = Image.new("L", (new_w, new_h), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle(mask_box, fill=255)

    # インペインティングパイプラインで拡張
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    ).to("cuda")

    result = pipe(
        prompt="seamless continuation of the scene, same style",
        image=canvas.resize((512, 512)),
        mask_image=mask.resize((512, 512)),
        num_inference_steps=30,
    ).images[0]

    return result.resize((new_w, new_h))
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

重複帯の処理:
┌────────┬──────────┬──────────┐
│ 元画像  │ ブレンド  │ 新規生成  │
│ (保持)  │ (グラデ)  │ (AI生成)  │
│ 100%   │ 100%→0%  │ 0%→100%  │
│ 元の    │ 元の     │ 生成     │
│ ピクセル│ 比率低下  │ ピクセル  │
└────────┴──────────┴──────────┘

パノラマ生成 (連続アウトペインティング):
┌────┬────┬────┬────┬────┐
│ ←  │ ←  │元画│ →  │ →  │
│拡張3│拡張2│ 像 │拡張1│拡張2│
└────┴────┴────┴────┴────┘
= 元の5倍幅のパノラマ画像
```

---

## 3. Img2Img と ControlNet

### コード例4: Img2Img によるスタイル変換

```python
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import torch

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

# 元画像を読み込み
init_image = Image.open("photo.jpg").resize((1024, 1024))

# スタイル変換
result = pipe(
    prompt="oil painting style, impressionist, Claude Monet, "
           "visible brushstrokes, vibrant colors",
    negative_prompt="photorealistic, sharp, digital",
    image=init_image,
    strength=0.65,  # 0.3=微調整, 0.5=中間, 0.8=大きく変化
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]

result.save("monet_style.png")

# strength の影響:
# 0.2-0.3: 色調補正レベル、構図完全維持
# 0.4-0.6: スタイル変換、構図はほぼ維持
# 0.7-0.8: 大きな変化、構図の大枠は維持
# 0.9-1.0: ほぼ新規生成、元画像は参考程度
```

### コード例5: ControlNet による精密制御

```python
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from controlnet_aux import CannyDetector, OpenposeDetector
from PIL import Image
import torch

# Canny Edge ControlNet
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")

# エッジ検出
canny = CannyDetector()
image = Image.open("building.jpg")
canny_image = canny(image, low_threshold=100, high_threshold=200)

# エッジに基づいて新しいスタイルで生成
result = pipe(
    prompt="futuristic glass building, sci-fi architecture, night",
    negative_prompt="low quality",
    image=canny_image,
    controlnet_conditioning_scale=0.8,  # 制御の強さ
    num_inference_steps=25,
).images[0]

result.save("futuristic_building.png")
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
│  └──────────────┘  └────────────┘  └───────────┘  │
│                                                     │
│  ┌── セグメント ─┐  ┌── 線画 ───┐  ┌── 法線 ──┐  │
│  │ Segmentation │  │ Scribble  │  │ Normal   │  │
│  │ ┌──┐         │  │  /~~\     │  │  →→→     │  │
│  │ │色│ 領域    │  │ |    | 手 │  │  →→→ 表面│  │
│  │ │分│ 分離    │  │  \__/ 描き│  │  →→→ 向き│  │
│  │ └──┘         │  │           │  │          │  │
│  └──────────────┘  └───────────┘  └──────────┘  │
│                                                     │
│  ┌── タイル ───┐   ┌── IP-Adapter ──┐             │
│  │ Tile        │   │ 画像ベース      │             │
│  │ ┌┬┬┐       │   │ 参照画像の      │             │
│  │ ├┼┼┤ 高解像│   │ スタイル/構図   │             │
│  │ ├┼┼┤ 度制御│   │ を転写          │             │
│  │ └┴┴┘       │   │                │             │
│  └─────────────┘   └────────────────┘             │
└─────────────────────────────────────────────────────┘
```

---

## 4. 比較表

### 比較表1: 画像編集手法の比較

| 手法 | 入力 | 制御精度 | 用途 | 計算コスト |
|------|------|---------|------|-----------|
| **インペインティング** | 画像 + マスク + プロンプト | 高 | 部分置換・修正 | 中 |
| **アウトペインティング** | 画像 + 方向指定 | 中 | 画像拡張 | 中 |
| **Img2Img** | 画像 + プロンプト + strength | 中 | スタイル変換 | 中 |
| **ControlNet** | 制御画像 + プロンプト | 非常に高 | 構図制御生成 | 高 |
| **IP-Adapter** | 参照画像 + プロンプト | 中 | スタイル転写 | 中 |
| **Instruct-Pix2Pix** | 画像 + 編集指示 | 中 | 自然言語編集 | 中 |

### 比較表2: マスク生成手法の比較

| 手法 | 精度 | 自動化 | ツール |
|------|------|--------|--------|
| **手動描画** | 最高 | 手動 | Photoshop, GIMP, WebUI |
| **矩形/楕円** | 低 | 自動 | PIL/Pillow |
| **SAM (セグメンテーション)** | 高 | 半自動 | segment-anything |
| **色範囲選択** | 中 | 自動 | OpenCV |
| **テキスト指定** | 中~高 | 自動 | Grounding DINO + SAM |
| **深度ベース** | 中 | 自動 | MiDaS + 閾値処理 |

---

## 5. アンチパターン

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
- 編集領域を実際の対象より少し大きめに設定
- strength パラメータで境界のブレンドを調整
- 後処理でさらに境界をブレンド
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

[正しいアプローチ]
- 微調整 (色補正等): 0.2-0.3
- スタイル変換: 0.4-0.6
- 大きな変更: 0.7-0.8
- 元画像は参考程度: 0.85-0.95
- タスクに応じて適切な範囲を設定する
```

---

## FAQ

### Q1: インペインティングで周囲と色味が合わない場合は?

**A:** 以下の順序で対処します:

1. **マスクを拡大する:** 境界を 20-50px 広げて周囲のコンテキストを含める
2. **フェザリング:** マスクに 15-25px のガウスぼかしを適用
3. **strength を下げる:** 0.7-0.8 程度にして元画像の色調を保持
4. **プロンプトに色調を明記:** "matching ambient lighting, same color temperature"
5. **二段階処理:** 粗い生成 → Img2Img で微調整

### Q2: アウトペインティングで一貫性を保つコツは?

**A:**

- **重複領域を十分に取る:** 最低 32px、推奨 64px 以上
- **同じプロンプトを使用:** 元画像の説明 + "seamless continuation"
- **一度に大きく拡張しない:** 256px ずつ段階的に拡張
- **シード固定:** 同じランダムシードで一貫性を保つ
- **後処理:** 拡張結果全体に Img2Img を軽く適用して統一

### Q3: ControlNet はどのタイプを選ぶべき?

**A:** タスクに応じて選択します:

- **建築・インテリア:** Canny Edge (輪郭維持) + Depth (奥行き)
- **人物ポーズ指定:** OpenPose (骨格制御)
- **既存画像の高品質化:** Tile (ディテール維持)
- **手描きスケッチから:** Scribble (ラフな線画)
- **スタイル転写:** IP-Adapter (参照画像ベース)
- **複数制御:** Multi-ControlNet で複数タイプを同時使用可能

---

## まとめ表

| 項目 | 要点 |
|------|------|
| **インペインティング** | マスク+プロンプトで部分書き換え。フェザリング必須 |
| **アウトペインティング** | 画像境界を拡張。重複領域で自然な接続 |
| **Img2Img** | strength で元画像の維持度を制御 (0.0-1.0) |
| **ControlNet** | エッジ/深度/ポーズ等で精密に構図を制御 |
| **マスク生成** | 手動 / SAM自動セグメンテーション / テキスト指定 |
| **境界処理** | フェザリング、重複帯、後処理ブレンドの3段構え |

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

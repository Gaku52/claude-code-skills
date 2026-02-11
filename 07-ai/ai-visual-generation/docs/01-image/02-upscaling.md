# アップスケーリング — Real-ESRGAN、超解像

> AI超解像技術による画像の高解像度化を、古典手法から最新のディープラーニングモデルまで体系的に解説する。

---

## この章で学ぶこと

1. **超解像の原理と種類** — 単一画像超解像 (SISR) の数学的基礎と進化
2. **主要モデルの比較と使い分け** — Real-ESRGAN、SwinIR、SUPIR の特徴と適用場面
3. **拡散モデルベースの超解像** — Stable Diffusion を活用した高品質アップスケール

---

## 1. 超解像の基本概念

### コード例1: 古典手法と AI 超解像の比較

```python
from PIL import Image
import cv2
import numpy as np

def compare_upscaling_methods(image_path, scale=4):
    """古典手法とAI超解像の比較"""
    img = Image.open(image_path)
    w, h = img.size
    new_w, new_h = w * scale, h * scale

    results = {}

    # 1. 最近傍補間 (Nearest Neighbor)
    results["nearest"] = img.resize(
        (new_w, new_h), Image.NEAREST
    )

    # 2. バイリニア補間
    results["bilinear"] = img.resize(
        (new_w, new_h), Image.BILINEAR
    )

    # 3. バイキュービック補間
    results["bicubic"] = img.resize(
        (new_w, new_h), Image.BICUBIC
    )

    # 4. Lanczos補間
    results["lanczos"] = img.resize(
        (new_w, new_h), Image.LANCZOS
    )

    return results

# 品質メトリクスの計算
def calculate_psnr(original, upscaled):
    """PSNR (ピーク信号対雑音比) を計算"""
    mse = np.mean((np.array(original) - np.array(upscaled)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(original, upscaled):
    """SSIM (構造的類似性) を計算"""
    from skimage.metrics import structural_similarity
    return structural_similarity(
        np.array(original),
        np.array(upscaled),
        channel_axis=2,  # カラー画像
    )
```

### ASCII図解1: 超解像の種類と進化

```
超解像技術の分類:

┌─────────── 古典手法 ──────────────────────────────┐
│  最近傍 → バイリニア → バイキュービック → Lanczos   │
│  (1970s)   (1980s)     (1990s)         (2000s)    │
│  品質: ★   品質: ★★   品質: ★★★     品質: ★★★☆  │
└───────────────────────────────────────────────────┘
                      │
                      v  ディープラーニングの登場
┌─────────── CNN ベース ────────────────────────────┐
│  SRCNN → EDSR → RCAN → SwinIR                    │
│  (2014)  (2017)  (2018)  (2021)                   │
│  品質: ★★★★  品質: ★★★★☆ 品質: ★★★★★            │
└───────────────────────────────────────────────────┘
                      │
                      v  GAN / 拡散モデルの登場
┌─────────── 生成ベース ───────────────────────────┐
│  SRGAN → Real-ESRGAN → StableSR → SUPIR          │
│  (2017)   (2021)       (2023)     (2024)          │
│  品質: ★★★★☆  品質: ★★★★★  リアルさ: ★★★★★     │
│  ※ディテールを「生成」するため忠実度は低下する場合あり │
└───────────────────────────────────────────────────┘
```

---

## 2. Real-ESRGAN

### コード例2: Real-ESRGAN の使用

```python
# pip install realesrgan
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from PIL import Image
import numpy as np
import cv2

def upscale_with_realesrgan(image_path, scale=4, model_name="x4plus"):
    """Real-ESRGAN による超解像"""

    # モデル選択
    models = {
        "x4plus": {
            "model": RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4,
            ),
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/"
                   "download/v0.1.0/RealESRGAN_x4plus.pth",
            "scale": 4,
        },
        "x4plus_anime": {
            "model": RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=6, num_grow_ch=32, scale=4,
            ),
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/"
                   "download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "scale": 4,
        },
    }

    config = models[model_name]

    upsampler = RealESRGANer(
        scale=config["scale"],
        model_path=config["url"],
        model=config["model"],
        tile=0,          # タイル処理 (0=無効, 推奨: 400)
        tile_pad=10,     # タイル間のパディング
        pre_pad=0,
        half=True,       # FP16 で高速化
    )

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    output, _ = upsampler.enhance(img, outscale=scale)
    cv2.imwrite("upscaled.png", output)

    return output

# アニメ画像用
upscale_with_realesrgan("anime.png", model_name="x4plus_anime")

# 写真用
upscale_with_realesrgan("photo.jpg", model_name="x4plus")
```

### コード例3: タイル処理による大画像の超解像

```python
def tiled_upscale(image_path, tile_size=512, overlap=64, scale=4):
    """
    大画像をタイル分割して超解像

    メモリ制限のある環境で大画像を処理する手法:
    1. 画像をタイルに分割
    2. 各タイルを個別に超解像
    3. 重複領域をブレンドして結合
    """
    from PIL import Image
    import numpy as np

    img = np.array(Image.open(image_path))
    h, w, c = img.shape
    out_h, out_w = h * scale, w * scale

    output = np.zeros((out_h, out_w, c), dtype=np.float32)
    weight = np.zeros((out_h, out_w, c), dtype=np.float32)

    # タイルの位置を計算
    y_positions = list(range(0, h - tile_size + 1, tile_size - overlap))
    x_positions = list(range(0, w - tile_size + 1, tile_size - overlap))

    # 端が含まれない場合は追加
    if y_positions[-1] + tile_size < h:
        y_positions.append(h - tile_size)
    if x_positions[-1] + tile_size < w:
        x_positions.append(w - tile_size)

    for y in y_positions:
        for x in x_positions:
            # タイルを切り出し
            tile = img[y:y+tile_size, x:x+tile_size]

            # 超解像を適用 (ここでは仮のアップスケール)
            upscaled_tile = upscale_single_tile(tile, scale)

            # 重み付けマスク (端を滑らかにブレンド)
            blend_mask = create_blend_mask(
                tile_size * scale, tile_size * scale,
                overlap * scale
            )

            # 結果に加算
            oy, ox = y * scale, x * scale
            ts = tile_size * scale
            output[oy:oy+ts, ox:ox+ts] += upscaled_tile * blend_mask
            weight[oy:oy+ts, ox:ox+ts] += blend_mask

    # 正規化
    output = output / np.maximum(weight, 1e-8)
    return output.astype(np.uint8)

def create_blend_mask(h, w, margin):
    """タイルブレンド用のグラデーションマスク"""
    mask = np.ones((h, w, 1), dtype=np.float32)
    # 四辺にグラデーションを適用
    for i in range(margin):
        alpha = i / margin
        mask[i, :, :] *= alpha      # 上辺
        mask[h-1-i, :, :] *= alpha  # 下辺
        mask[:, i, :] *= alpha      # 左辺
        mask[:, w-1-i, :] *= alpha  # 右辺
    return mask
```

### ASCII図解2: タイル処理のブレンド概念

```
元画像のタイル分割:
┌────────┬──┬────────┐
│ Tile A │重│ Tile B │
│        │複│        │
├──── 重複 ──┼────────┤
│        │  │        │
│ Tile C │重│ Tile D │
│        │複│        │
└────────┴──┴────────┘

ブレンドマスク (1タイルの重み):
┌────────────────────┐
│ 0.0  0.5  1.0  0.5│  ← 左右のグラデーション
│ 0.5  1.0  1.0  0.5│
│ 1.0  1.0  1.0  1.0│  ← 中央は最大重み
│ 0.5  1.0  1.0  0.5│
│ 0.0  0.5  1.0  0.5│  ← 上下のグラデーション
└────────────────────┘

結合結果:
 タイルA×重みA + タイルB×重みB
───────────────────────────── = 最終ピクセル値
      重みA + 重みB
```

---

## 3. 拡散モデルベースの超解像

### コード例4: Stable Diffusion による超解像 (SD Upscaler)

```python
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
import torch

# SD x4 Upscaler パイプライン
pipe = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler",
    torch_dtype=torch.float16,
).to("cuda")

# 低解像度画像を読み込み
low_res = Image.open("small_image.png")  # 例: 256x256

# プロンプトガイド付きアップスケール
upscaled = pipe(
    prompt="high resolution, sharp details, photorealistic, 8K",
    negative_prompt="blurry, noisy, artifacts, low quality",
    image=low_res,
    num_inference_steps=25,
    guidance_scale=4.0,    # 低めが推奨 (高すぎるとアーティファクト)
    noise_level=20,        # ノイズレベル (0-350, 推奨: 20-50)
).images[0]

upscaled.save("high_res.png")  # 出力: 1024x1024
```

### コード例5: SUPIR による高品質アップスケール

```python
"""
SUPIR (Scaling Up to Excellence: Practicing Model Scaling
       for Photo-Realistic Image Restoration)

大規模言語モデル (SDXL) を活用した超解像。
テキストプロンプトで生成するディテールを制御可能。
"""

# SUPIR は通常CLIまたはGradioで使用
# pip install git+https://github.com/Fanghua-Yu/SUPIR.git

# CLI使用例:
# python inference.py \
#   --input_path input.png \
#   --output_path output.png \
#   --prompt "high quality photograph, sharp details" \
#   --upscale 4 \
#   --model_path SUPIR-v0Q.ckpt

# API的な使用 (概念コード):
class SUPIRUpscaler:
    """SUPIR超解像の概念的なラッパー"""

    def __init__(self, model_path, device="cuda"):
        self.device = device
        # モデルロード (実際にはSUPIRの設定に従う)
        self.model = self._load_model(model_path)

    def upscale(self, image, prompt="", scale=4,
                restoration_strength=0.7):
        """
        テキストガイド付き超解像

        restoration_strength:
          0.0-0.3: 忠実度重視 (元画像に近い)
          0.4-0.6: バランス
          0.7-1.0: 品質重視 (ディテール生成多め)
        """
        # 1. 低品質画像のエンコード
        # 2. テキストプロンプトのエンコード
        # 3. 拡散過程で高解像度画像を生成
        # 4. 元画像との整合性を保ちつつ復元
        pass

    def batch_upscale(self, image_dir, output_dir, **kwargs):
        """ディレクトリ内の全画像を一括処理"""
        from pathlib import Path
        for img_path in Path(image_dir).glob("*.{png,jpg,jpeg}"):
            img = Image.open(img_path)
            result = self.upscale(img, **kwargs)
            result.save(Path(output_dir) / img_path.name)
```

### ASCII図解3: 超解像パイプライン選択フローチャート

```
                    START
                      │
                      v
              ┌──────────────┐
              │ 用途は何か?  │
              └──────┬───────┘
                     │
        ┌────────────┼────────────┐
        v            v            v
   ┌────────┐  ┌──────────┐  ┌────────┐
   │写真/実写│  │アニメ/   │  │テキスト│
   │        │  │イラスト  │  │/文書   │
   └───┬────┘  └────┬─────┘  └───┬────┘
       │            │            │
       v            v            v
  ┌─────────┐ ┌──────────┐ ┌─────────┐
  │忠実度   │ │Real-ESRGAN│ │waifu2x  │
  │重視?    │ │anime      │ │/Lanczos │
  └──┬──────┘ └──────────┘ └─────────┘
     │
  ┌──┴──┐
  │Yes  │No
  v     v
┌──────┐ ┌──────────┐
│Real- │ │SUPIR /   │
│ESRGAN│ │StableSR  │
│x4plus│ │(生成型)  │
└──────┘ └──────────┘

判断基準:
  忠実度重視 = 医療画像、証拠写真、科学データ
  品質重視   = SNS投稿、印刷物、プレゼン資料
```

---

## 4. 比較表

### 比較表1: 超解像モデルの詳細比較

| モデル | 種類 | 最大倍率 | 速度 | 品質 | VRAM | 忠実度 |
|--------|------|---------|------|------|------|--------|
| **Lanczos** | 古典 | 無制限 | 極速 | ★★★ | 0 | 最高 |
| **Real-ESRGAN** | GAN | x4 | 速い | ★★★★☆ | 2GB | 高い |
| **ESRGAN anime** | GAN | x4 | 速い | ★★★★ | 2GB | 高い |
| **SwinIR** | Transformer | x4 | 中 | ★★★★☆ | 4GB | 高い |
| **SD x4 Upscaler** | 拡散 | x4 | 遅い | ★★★★★ | 6GB | 中程度 |
| **StableSR** | 拡散 | x4 | 遅い | ★★★★★ | 8GB | 中程度 |
| **SUPIR** | 拡散+LLM | x4 | 非常に遅い | ★★★★★ | 12GB+ | 中程度 |

### 比較表2: ユースケース別推奨モデル

| ユースケース | 推奨モデル | 理由 |
|-------------|-----------|------|
| **SNS投稿の高画質化** | Real-ESRGAN x4plus | 速度と品質のバランス |
| **アニメ/イラスト** | Real-ESRGAN anime | アニメ特化の学習 |
| **印刷用の高解像化** | SUPIR / StableSR | 最高品質のディテール生成 |
| **バッチ処理 (大量)** | Real-ESRGAN | 処理速度が速い |
| **医療/科学画像** | SwinIR / Lanczos | 忠実度が最優先 |
| **顔写真の復元** | GFPGAN + Real-ESRGAN | 顔特化モデルとの組み合わせ |
| **古い写真の復元** | SUPIR | テキストガイドでディテール追加 |

---

## 5. アンチパターン

### アンチパターン1: 超解像を何度も繰り返す

```
[問題]
4倍超解像を2回適用すれば16倍になると考え、
繰り返し処理を行う。

[なぜ問題か]
- 各処理でアーティファクトが蓄積される
- GAN系はパターンを過剰に強調してしまう
- 実在しないディテールが増幅される
- 2回目の処理は品質向上に寄与しないことが多い

[正しいアプローチ]
- 一度の処理で目標解像度に到達する (4x が限界の目安)
- 大きな拡大が必要な場合: Lanczos で中間サイズ → AI超解像
- 拡散ベース (SUPIR等) で一度に高品質化する
```

### アンチパターン2: 全画像に同じモデルを適用

```
[問題]
写真もアニメもイラストも全て Real-ESRGAN x4plus で処理する。

[なぜ問題か]
- 写真用モデルをアニメに使うとテクスチャが過剰
- アニメ用モデルを写真に使うとのっぺりした結果に
- テキスト画像には不向き (文字がぼやける)

[正しいアプローチ]
- 入力画像の種類を判別してモデルを切り替え
- 写真: x4plus / SUPIR
- アニメ: x4plus_anime / waifu2x
- テキスト: Lanczos / SwinIR
- 混合コンテンツ: 領域分割して個別処理
```

---

## FAQ

### Q1: 超解像とただの拡大の違いは?

**A:** 根本的に異なります:

- **拡大 (リサイズ):** 既存のピクセルを補間。新しい情報は追加されない。必ずぼやける
- **超解像 (Super Resolution):** AIが学習したパターンから**新しいディテールを推定/生成**する。エッジが鮮明になり、テクスチャが復元される
- **注意:** 超解像は「推定」なので、元画像に存在しないディテールを「想像」で追加する場合がある。忠実度が求められる場合は注意

### Q2: アップスケール倍率はどこまで実用的?

**A:**

- **2倍:** 最も安全。高品質で忠実度も高い
- **4倍:** 実用的な上限。写真なら十分な品質
- **8倍以上:** ディテールの多くがAI生成。アート/クリエイティブ用途なら可
- **目安:** 元画像が 512x512 なら 4倍 (2048x2048) が実用限界
- **大きな拡大が必要な場合:** 拡散ベース (SUPIR) で品質を担保

### Q3: 超解像処理でVRAMが足りない場合は?

**A:** 以下の対策があります:

1. **タイル処理:** 画像を分割して個別に処理 (推奨 tile_size=400-512)
2. **FP16 (半精度):** half=True でVRAM使用量を半減
3. **CPU処理:** 遅いがVRAM不要。Real-ESRGAN は CPU でも動作
4. **モデルサイズの削減:** RealESRGAN_x4plus_anime (6B) は軽量版
5. **クラウドAPI:** Replicate 等のAPIサービスを利用

---

## まとめ表

| 項目 | 要点 |
|------|------|
| **古典手法** | Lanczos が最良。忠実度100%だがディテール追加なし |
| **GAN系 (Real-ESRGAN)** | 高速・高品質。実写/アニメでモデルを使い分け |
| **Transformer系 (SwinIR)** | GAN系より忠実度が高い。処理はやや遅い |
| **拡散系 (SUPIR)** | 最高品質だが遅い。テキストガイドで制御可能 |
| **タイル処理** | 大画像やVRAM不足時の必須テクニック |
| **実用倍率** | 4倍が上限目安。2倍が最も安全 |

---

## 次に読むべきガイド

- [03-design-tools.md](./03-design-tools.md) — デザインツールに統合された超解像機能
- [../02-video/00-video-generation.md](../02-video/00-video-generation.md) — 動画の超解像
- [../03-3d/00-3d-generation.md](../03-3d/00-3d-generation.md) — 3Dテクスチャの高解像度化

---

## 参考文献

1. Wang, X. et al. (2021). "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data." *ICCV Workshop 2021*. https://arxiv.org/abs/2107.10833
2. Liang, J. et al. (2021). "SwinIR: Image Restoration Using Swin Transformer." *ICCV Workshop 2021*. https://arxiv.org/abs/2108.10257
3. Yu, F. et al. (2024). "Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild." *CVPR 2024*. https://arxiv.org/abs/2401.13627
4. Wang, J. et al. (2023). "Exploiting Diffusion Prior for Real-World Image Super-Resolution (StableSR)." *arXiv*. https://arxiv.org/abs/2305.07015

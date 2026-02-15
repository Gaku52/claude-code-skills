# アップスケーリング — Real-ESRGAN、超解像

> AI超解像技術による画像の高解像度化を、古典手法から最新のディープラーニングモデルまで体系的に解説する。

---

## この章で学ぶこと

1. **超解像の原理と種類** — 単一画像超解像 (SISR) の数学的基礎と進化
2. **主要モデルの比較と使い分け** — Real-ESRGAN、SwinIR、SUPIR の特徴と適用場面
3. **拡散モデルベースの超解像** — Stable Diffusion を活用した高品質アップスケール
4. **実務パイプライン構築** — バッチ処理、API統合、品質管理の実践手法
5. **顔復元と超解像の組み合わせ** — GFPGAN、CodeFormer との連携テクニック

---

## 1. 超解像の基本概念

### 1.1 数学的基礎

超解像は、低解像度画像 $I_{LR}$ から高解像度画像 $I_{HR}$ を推定する逆問題（Inverse Problem）として定式化される。劣化モデルは以下のように表現できる：

```
I_LR = (I_HR * k) ↓_s + n

ここで:
  k   = ぼかしカーネル（ガウシアンブラーなど）
  ↓_s = ダウンサンプリング（倍率 s）
  n   = ノイズ
  *   = 畳み込み演算
```

この劣化過程は不可逆であるため、超解像は本質的に不良設定問題（ill-posed problem）であり、1つの低解像度画像に対して複数の高解像度画像が候補となる。

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

def calculate_lpips(original, upscaled):
    """LPIPS (知覚的類似性) を計算 — 人間の知覚に近い指標"""
    import torch
    import lpips

    loss_fn = lpips.LPIPS(net='alex')

    # PIL → Tensor に変換 ([-1, 1] 範囲)
    def to_tensor(img):
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    t_orig = to_tensor(original)
    t_upsc = to_tensor(upscaled)

    with torch.no_grad():
        score = loss_fn(t_orig, t_upsc)

    return score.item()  # 低いほど類似度が高い


def comprehensive_quality_assessment(original_path, upscaled_path):
    """包括的な品質評価レポートを生成"""
    from PIL import Image
    import json

    original = Image.open(original_path)
    upscaled = Image.open(upscaled_path)

    # 解像度チェック
    orig_w, orig_h = original.size
    upsc_w, upsc_h = upscaled.size
    scale_w = upsc_w / orig_w
    scale_h = upsc_h / orig_h

    # メトリクス計算
    # ※PSNRとSSIMは同サイズの画像で比較するため、
    # 元画像を同じサイズにリサイズして比較
    original_resized = original.resize(
        (upsc_w, upsc_h), Image.LANCZOS
    )

    report = {
        "resolution": {
            "original": f"{orig_w}x{orig_h}",
            "upscaled": f"{upsc_w}x{upsc_h}",
            "scale_factor": f"{scale_w:.1f}x{scale_h:.1f}",
        },
        "metrics": {
            "psnr_db": round(
                calculate_psnr(original_resized, upscaled), 2
            ),
            "ssim": round(
                calculate_ssim(original_resized, upscaled), 4
            ),
        },
        "file_info": {
            "original_size_kb": round(
                os.path.getsize(original_path) / 1024, 1
            ),
            "upscaled_size_kb": round(
                os.path.getsize(upscaled_path) / 1024, 1
            ),
        },
    }

    return report
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

### 1.2 劣化モデルの詳細理解

実世界の画像劣化は、単純なダウンサンプリングとは異なる複合的な劣化である。Real-ESRGAN の成功の鍵は、この実世界の劣化をより正確にモデル化した点にある。

```python
def simulate_real_world_degradation(image, scale=4):
    """
    Real-ESRGAN の第2次劣化モデルを再現

    実世界の劣化 = ぼかし + ダウンサンプリング + ノイズ + JPEG圧縮
    これを2回繰り返す (second-order degradation)
    """
    import cv2
    import numpy as np

    img = np.array(image).astype(np.float32) / 255.0

    # === 第1次劣化 ===
    # 1. ぼかし (等方性/異方性ガウシアン)
    kernel_size = np.random.choice([7, 9, 11, 13, 15, 17, 19, 21])
    sigma = np.random.uniform(0.2, 3.0)
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    # 2. ダウンサンプリング (バイキュービック/バイリニア/エリア)
    h, w = img.shape[:2]
    method = np.random.choice([
        cv2.INTER_CUBIC,
        cv2.INTER_LINEAR,
        cv2.INTER_AREA,
    ])
    down_scale = np.random.uniform(1.0, scale)
    img = cv2.resize(
        img, (int(w / down_scale), int(h / down_scale)),
        interpolation=method,
    )

    # 3. ノイズ追加 (ガウシアン/ポアソン)
    noise_type = np.random.choice(["gaussian", "poisson"])
    if noise_type == "gaussian":
        sigma_n = np.random.uniform(1, 30) / 255.0
        noise = np.random.normal(0, sigma_n, img.shape)
        img = img + noise
    else:
        vals = 2 ** np.ceil(np.log2(len(np.unique(img))))
        img = np.random.poisson(img * vals) / vals

    # 4. JPEG圧縮アーティファクト
    quality = np.random.randint(30, 95)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', (img * 255).astype(np.uint8),
                          encode_param)
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

    # === 第2次劣化 (繰り返し) ===
    # 同様の処理をもう一度適用（パラメータは異なる範囲で）
    kernel_size2 = np.random.choice([7, 9, 11, 13, 15, 17, 19, 21])
    sigma2 = np.random.uniform(0.2, 1.5)
    img = cv2.GaussianBlur(img, (kernel_size2, kernel_size2), sigma2)

    # 最終リサイズ
    img = cv2.resize(
        img, (w // scale, h // scale),
        interpolation=cv2.INTER_LINEAR,
    )

    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)
```

### ASCII図解: Real-ESRGAN の第2次劣化モデル

```
Real-ESRGAN の学習用劣化パイプライン:

高解像度画像 (GT)
     │
     ▼ ─── 第1次劣化 ───
     │
     ├── ぼかし (等方性/異方性/焦点外)
     │     │ カーネル: ガウシアン, 一般化ガウシアン, プラトーガウシアン
     │     │ サイズ: 7-21, σ: 0.2-3.0
     │
     ├── リサイズ (ダウンサンプリング)
     │     │ 方式: バイキュービック / バイリニア / エリア
     │     │ 倍率: 0.15-1.5
     │
     ├── ノイズ
     │     │ ガウシアン: σ = 1-30
     │     │ ポアソン: scale = 0.05-3.0
     │
     └── JPEG圧縮
           │ 品質: 30-95
           ▼
     │
     ▼ ─── 第2次劣化 ─── (同じ4ステップを再度適用)
     │
     ├── ぼかし (σ: 0.2-1.5, より穏やか)
     ├── リサイズ
     ├── ノイズ (σ: 1-25)
     └── JPEG/WEBP圧縮
           │
           ▼
     低解像度画像 (LR) ← 学習ペアとして使用
```

---

## 2. Real-ESRGAN

### 2.1 アーキテクチャの詳細

Real-ESRGAN は RRDB (Residual in Residual Dense Block) ネットワークをバックボーンとして使用する。U-Net ディスクリミネータとスペクトル正規化を組み合わせることで、安定した学習を実現している。

```
RRDB ネットワーク構造:

入力 (3ch) ──→ Conv3x3 ──→ [RRDB x 23] ──→ Conv3x3 ──→ Upsample ──→ 出力 (3ch)
                              │                              │
                              └───── Skip Connection ─────────┘

1つの RRDB:
┌─────────────────────────────────┐
│  ┌──── Dense Block 1 ────┐      │
│  │ Conv → LeakyReLU       │      │
│  │ Conv → LeakyReLU       │      │
│  │ Conv → LeakyReLU       │      │
│  │ Conv → LeakyReLU       │      │
│  │ Conv                   │      │
│  └─── × β (0.2) ─────────┘      │
│                │                  │
│  ┌──── Dense Block 2 ────┐      │
│  │ (同様の構造)           │      │
│  └─── × β (0.2) ─────────┘      │
│                │                  │
│  ┌──── Dense Block 3 ────┐      │
│  │ (同様の構造)           │      │
│  └─── × β (0.2) ─────────┘      │
└──────── × β (0.2) ──── + 入力 ──┘
```

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
        "x2plus": {
            "model": RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=2,
            ),
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/"
                   "download/v0.2.1/RealESRGAN_x2plus.pth",
            "scale": 2,
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

### 2.2 Real-ESRGAN の高度な使い方

```python
class RealESRGANPipeline:
    """
    Real-ESRGAN を実務で使うための本格的なパイプライン

    機能:
    - 自動モデル選択 (画像種別判定)
    - バッチ処理
    - 品質評価
    - 進捗レポート
    """

    def __init__(self, device="cuda", half=True):
        self.device = device
        self.half = half
        self.models = {}
        self._load_models()

    def _load_models(self):
        """モデルの遅延ロード準備"""
        self.model_configs = {
            "photo": {
                "arch": RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4,
                ),
                "path": "weights/RealESRGAN_x4plus.pth",
                "scale": 4,
            },
            "anime": {
                "arch": RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=6, num_grow_ch=32, scale=4,
                ),
                "path": "weights/RealESRGAN_x4plus_anime_6B.pth",
                "scale": 4,
            },
        }

    def _get_upsampler(self, model_type, tile=0):
        """必要なモデルだけを遅延ロード"""
        if model_type not in self.models:
            config = self.model_configs[model_type]
            self.models[model_type] = RealESRGANer(
                scale=config["scale"],
                model_path=config["path"],
                model=config["arch"],
                tile=tile,
                tile_pad=10,
                pre_pad=0,
                half=self.half,
            )
        return self.models[model_type]

    def detect_image_type(self, image_path):
        """画像の種類を自動判別"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"画像読み込み失敗: {image_path}")

        # 色のユニーク数と分散で判断
        # アニメ/イラストは色数が少なく、グラデーションも少ない
        h, w = img.shape[:2]
        sample = img[::4, ::4]  # 4ピクセルおきにサンプリング

        unique_colors = len(np.unique(
            sample.reshape(-1, 3), axis=0
        ))
        color_ratio = unique_colors / (sample.shape[0] * sample.shape[1])

        # エッジの鮮明さ
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size

        # 判定ロジック
        if color_ratio < 0.3 and edge_ratio > 0.05:
            return "anime"
        else:
            return "photo"

    def upscale(self, image_path, output_path=None,
                model_type=None, scale=4, tile=0):
        """
        画像を超解像

        Parameters:
            image_path: 入力画像パス
            output_path: 出力画像パス (None=自動命名)
            model_type: "photo" / "anime" / None (自動判定)
            scale: 出力倍率 (2 or 4)
            tile: タイルサイズ (0=無効, 400-512推奨)

        Returns:
            dict: 処理結果 (出力パス、メタデータ)
        """
        import time
        start_time = time.time()

        # モデル選択
        if model_type is None:
            model_type = self.detect_image_type(image_path)

        # VRAM に応じたタイルサイズ自動決定
        if tile == 0:
            img = cv2.imread(image_path)
            h, w = img.shape[:2]
            total_pixels = h * w
            if total_pixels > 2_000_000:  # 2MP以上
                tile = 400
            elif total_pixels > 4_000_000:  # 4MP以上
                tile = 256

        upsampler = self._get_upsampler(model_type, tile=tile)

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        output, _ = upsampler.enhance(img, outscale=scale)

        if output_path is None:
            from pathlib import Path
            p = Path(image_path)
            output_path = str(p.parent / f"{p.stem}_upscaled{p.suffix}")

        cv2.imwrite(output_path, output)

        elapsed = time.time() - start_time

        return {
            "input": image_path,
            "output": output_path,
            "model": model_type,
            "scale": scale,
            "input_size": f"{img.shape[1]}x{img.shape[0]}",
            "output_size": f"{output.shape[1]}x{output.shape[0]}",
            "elapsed_seconds": round(elapsed, 2),
        }

    def batch_upscale(self, input_dir, output_dir,
                      model_type=None, scale=4, tile=400,
                      extensions=("png", "jpg", "jpeg", "webp")):
        """ディレクトリ内の全画像を一括超解像"""
        from pathlib import Path
        import json

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = []
        for ext in extensions:
            files.extend(input_path.glob(f"*.{ext}"))
            files.extend(input_path.glob(f"*.{ext.upper()}"))

        results = []
        total = len(files)

        for i, f in enumerate(sorted(files)):
            print(f"[{i+1}/{total}] Processing: {f.name}")
            try:
                result = self.upscale(
                    str(f),
                    str(output_path / f.name),
                    model_type=model_type,
                    scale=scale,
                    tile=tile,
                )
                result["status"] = "success"
            except Exception as e:
                result = {
                    "input": str(f),
                    "status": "error",
                    "error": str(e),
                }
            results.append(result)

        # レポート出力
        report_path = output_path / "upscale_report.json"
        with open(report_path, "w", encoding="utf-8") as fout:
            json.dump(results, fout, indent=2, ensure_ascii=False)

        success_count = sum(
            1 for r in results if r["status"] == "success"
        )
        print(f"\n完了: {success_count}/{total} 成功")
        return results
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

### 3.1 StableSR — 拡散事前学習を活用した超解像

```python
"""
StableSR: Stable Diffusion の事前知識を活用した超解像

特徴:
- Stable Diffusion の学習済み知識をそのまま利用
- Time-aware Encoder で忠実度と品質のバランスを制御
- CFW (Controllable Feature Wrapping) モジュール
"""

from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
)
import torch

class StableSRPipeline:
    """StableSR の概念的な実装"""

    def __init__(self, sd_model_path, sr_module_path, device="cuda"):
        self.device = device

        # Stable Diffusion ベースモデル
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            sd_model_path,
            torch_dtype=torch.float16,
        ).to(device)

        # 超解像用の追加モジュール
        self.encoder_module = self._load_sr_encoder(sr_module_path)

    def upscale(self, image, scale=4,
                positive_prompt="",
                negative_prompt="blurry, artifacts",
                num_steps=50,
                color_fix="wavelet"):
        """
        StableSR によるアップスケール

        Parameters:
            image: PIL Image (低解像度)
            scale: 拡大倍率
            positive_prompt: 品質向上プロンプト
            negative_prompt: 抑制プロンプト
            num_steps: 拡散ステップ数
            color_fix: 色補正方式 ("none", "adain", "wavelet")
        """
        # 1. 低解像度画像をエンコード
        lr_features = self.encoder_module.encode(image)

        # 2. 拡散過程で高解像度化
        #    Time-aware Encoder が各ステップの特徴を調整
        hr_latent = self.sd_pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            # 独自の条件付けメカニズム
            cross_attention_kwargs={
                "lr_features": lr_features
            },
        )

        # 3. 色補正
        result = hr_latent.images[0]
        if color_fix == "wavelet":
            result = self._wavelet_color_fix(image, result)
        elif color_fix == "adain":
            result = self._adain_color_fix(image, result)

        return result

    def _wavelet_color_fix(self, source, target):
        """ウェーブレット変換による色補正"""
        import pywt

        # 低周波成分（色情報）を元画像から取得
        # 高周波成分（ディテール）を超解像結果から取得
        source_resized = source.resize(target.size, Image.LANCZOS)

        src_arr = np.array(source_resized).astype(np.float32)
        tgt_arr = np.array(target).astype(np.float32)

        result = np.zeros_like(tgt_arr)

        for ch in range(3):
            # ウェーブレット分解
            src_coeffs = pywt.dwt2(src_arr[:, :, ch], 'haar')
            tgt_coeffs = pywt.dwt2(tgt_arr[:, :, ch], 'haar')

            # 低周波は元画像、高周波は超解像結果
            new_coeffs = (src_coeffs[0], tgt_coeffs[1])
            result[:, :, ch] = pywt.idwt2(new_coeffs, 'haar')

        return Image.fromarray(
            np.clip(result, 0, 255).astype(np.uint8)
        )

    def _adain_color_fix(self, source, target):
        """AdaIN による色補正"""
        source_resized = source.resize(target.size, Image.LANCZOS)
        src = np.array(source_resized).astype(np.float32)
        tgt = np.array(target).astype(np.float32)

        for ch in range(3):
            src_mean, src_std = src[:,:,ch].mean(), src[:,:,ch].std()
            tgt_mean, tgt_std = tgt[:,:,ch].mean(), tgt[:,:,ch].std()
            tgt[:,:,ch] = (
                (tgt[:,:,ch] - tgt_mean) / (tgt_std + 1e-8)
            ) * src_std + src_mean

        return Image.fromarray(
            np.clip(tgt, 0, 255).astype(np.uint8)
        )
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

## 4. 顔復元と超解像の組み合わせ

### 4.1 GFPGAN による顔特化復元

顔画像の超解像では、汎用モデルだけでは不十分な場合が多い。GFPGAN (Generative Facial Prior GAN) は、顔の幾何学的構造を事前知識として利用することで、高品質な顔復元を実現する。

```python
from gfpgan import GFPGANer
import cv2
import numpy as np

class FaceRestorationPipeline:
    """顔復元 + 超解像の統合パイプライン"""

    def __init__(self, device="cuda"):
        self.device = device

        # GFPGAN 顔復元モデル
        self.face_restorer = GFPGANer(
            model_path="weights/GFPGANv1.4.pth",
            upscale=4,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=self._create_bg_upsampler(),
        )

    def _create_bg_upsampler(self):
        """背景用の Real-ESRGAN アップサンプラー"""
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )
        return RealESRGANer(
            scale=4,
            model_path="weights/RealESRGAN_x4plus.pth",
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True,
        )

    def restore(self, image_path, output_path=None,
                fidelity_weight=0.5, only_center_face=False):
        """
        顔復元 + 背景超解像

        Parameters:
            image_path: 入力画像パス
            fidelity_weight: 忠実度の重み (0=品質重視, 1=忠実度重視)
            only_center_face: 最大の顔のみ処理するか

        Returns:
            dict: 復元結果と検出情報
        """
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # 顔復元実行
        _, _, output = self.face_restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=only_center_face,
            paste_back=True,
            weight=fidelity_weight,
        )

        if output_path:
            cv2.imwrite(output_path, output)

        return {
            "output": output,
            "input_size": f"{img.shape[1]}x{img.shape[0]}",
            "output_size": f"{output.shape[1]}x{output.shape[0]}",
        }


    def batch_restore(self, input_dir, output_dir,
                      fidelity_weight=0.5):
        """ディレクトリ内の全画像を顔復元"""
        from pathlib import Path

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []
        for img_file in sorted(input_path.glob("*")):
            if img_file.suffix.lower() in (
                ".png", ".jpg", ".jpeg", ".webp"
            ):
                try:
                    result = self.restore(
                        str(img_file),
                        str(output_path / img_file.name),
                        fidelity_weight=fidelity_weight,
                    )
                    result["status"] = "success"
                    result["file"] = img_file.name
                except Exception as e:
                    result = {
                        "file": img_file.name,
                        "status": "error",
                        "error": str(e),
                    }
                results.append(result)

        return results
```

### 4.2 CodeFormer — コードブックベースの顔復元

```python
"""
CodeFormer: 離散コードブックを用いた顔復元

GFPGAN との違い:
- コードブック: VQ-VAE で学習した離散表現を使用
- 忠実度制御: パラメータ w で連続的に制御可能
  w=0: 高品質だが忠実度低い
  w=1: 忠実度高いが品質は低い
  推奨: w=0.5-0.7
"""

# pip install codeformer-pip

def restore_face_codeformer(
    image_path,
    output_path,
    fidelity_weight=0.5,
    upscale=4,
    detection_model="retinaface_resnet50",
):
    """CodeFormer による顔復元"""
    import subprocess

    cmd = [
        "python", "inference_codeformer.py",
        "-i", image_path,
        "-o", output_path,
        "-w", str(fidelity_weight),
        "-s", str(upscale),
        "--detection_model", detection_model,
        "--bg_upsampler", "realesrgan",
        "--face_upsample",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"CodeFormer error: {result.stderr}")

    return output_path
```

### ASCII図解: 顔復元パイプラインの全体像

```
入力画像 (低解像度、劣化あり)
     │
     ▼
┌────────────────────────────────┐
│   顔検出 (RetinaFace/MTCNN)   │
│   ┌─────┐  ┌─────┐  ┌─────┐  │
│   │Face1│  │Face2│  │Face3│  │
│   └──┬──┘  └──┬──┘  └──┬──┘  │
└──────┼────────┼────────┼──────┘
       │        │        │
       ▼        ▼        ▼
┌─────────────────────────────────┐
│     顔アラインメント             │
│  (5点ランドマーク → affine変換)  │
│  → 512x512 に正規化             │
└──────────┬──────────────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌──────────┐ ┌───────────┐
│ GFPGAN   │ │CodeFormer │
│          │ │           │
│ GAN Prior│ │ Codebook  │
│ + Style  │ │ + VQ-VAE  │
│ Transfer │ │ + w制御   │
└────┬─────┘ └────┬──────┘
     │            │
     ▼            ▼
┌────────────────────────┐
│  逆アフィン変換         │
│  (元の位置に貼り戻し)   │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  背景超解像              │
│  (Real-ESRGAN x4plus)  │
│  顔以外の領域を処理      │
└────────┬───────────────┘
         │
         ▼
    高解像度出力画像
```

---

## 5. SwinIR — Transformer ベースの超解像

### 5.1 SwinIR の実装と使い方

```python
"""
SwinIR: Swin Transformer を超解像に適用

特徴:
- CNN より広い受容野 (ウィンドウアテンション)
- シフトウィンドウで隣接ウィンドウ間の情報交換
- 忠実度が高い (GAN系ほどディテールを「想像」しない)
"""

import torch
from PIL import Image
import numpy as np

def upscale_with_swinir(image_path, scale=4, task="real_sr"):
    """
    SwinIR による超解像

    task:
      "classical_sr": 古典的超解像 (bicubic劣化想定)
      "real_sr": 実世界超解像 (複合劣化想定)
      "lightweight_sr": 軽量版
      "jpeg_car": JPEG圧縮アーティファクト除去
      "color_dn": カラーノイズ除去
      "gray_dn": グレースケールノイズ除去
    """
    # モデル設定
    model_configs = {
        "classical_sr": {
            "model_path": "weights/001_classicalSR_DF2K_s64w8_"
                         f"SwinIR-M_x{scale}.pth",
            "window_size": 8,
            "img_size": 64,
        },
        "real_sr": {
            "model_path": "weights/003_realSR_BSRGAN_DFOWMFC_"
                         f"s64w8_SwinIR-L_x{scale}_GAN.pth",
            "window_size": 8,
            "img_size": 64,
        },
        "lightweight_sr": {
            "model_path": "weights/002_lightweightSR_DIV2K_"
                         f"s64w8_SwinIR-S_x{scale}.pth",
            "window_size": 8,
            "img_size": 64,
        },
    }

    config = model_configs[task]

    # 画像読み込みとパディング
    img = np.array(Image.open(image_path)).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    # ウィンドウサイズの倍数にパディング
    ws = config["window_size"]
    _, _, h, w = img.shape
    pad_h = (ws - h % ws) % ws
    pad_w = (ws - w % ws) % ws
    img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode="reflect")

    # モデルロードと推論
    model = torch.load(config["model_path"])
    model.eval()

    with torch.no_grad():
        output = model(img.to("cuda"))

    # パディング除去
    output = output[:, :, :h * scale, :w * scale]

    # Tensor → PIL Image
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(output)
```

### 5.2 SwinIR と Real-ESRGAN の使い分け

```
SwinIR vs Real-ESRGAN の判断マトリクス:

                 忠実度優先          品質優先
                  ┌──────┐          ┌──────┐
 処理速度重視 ──→ │SwinIR│          │Real- │ ←── 処理速度重視
                  │(L)   │          │ESRGAN│
                  └──────┘          └──────┘

                  ┌──────┐          ┌──────┐
 品質最優先 ───→  │SwinIR│          │SUPIR │ ←── 品質最優先
                  │+ 後処│          │      │
                  │理    │          │      │
                  └──────┘          └──────┘

具体的な選択基準:

┌─────────────────┬──────────┬─────────────┐
│ 要件            │ SwinIR   │ Real-ESRGAN │
├─────────────────┼──────────┼─────────────┤
│ 医療/科学画像   │ ◎        │ △           │
│ 証拠写真        │ ◎        │ △           │
│ EC商品写真      │ ○        │ ◎           │
│ SNS投稿         │ △        │ ◎           │
│ 印刷物          │ ○        │ ○           │
│ アニメ/イラスト  │ △        │ ◎           │
│ バッチ処理      │ ○        │ ◎           │
│ エッジ保持      │ ◎        │ ○           │
│ テクスチャ生成  │ △        │ ◎           │
└─────────────────┴──────────┴─────────────┘
```

---

## 6. 動画超解像

### 6.1 フレーム単位の超解像

```python
import cv2
from pathlib import Path
import time

class VideoUpscaler:
    """動画超解像パイプライン"""

    def __init__(self, model_type="realesrgan", scale=4, device="cuda"):
        self.scale = scale
        self.model_type = model_type
        self._init_model(device)

    def _init_model(self, device):
        """モデル初期化"""
        if self.model_type == "realesrgan":
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=self.scale,
            )
            self.upsampler = RealESRGANer(
                scale=self.scale,
                model_path=f"weights/RealESRGAN_x{self.scale}plus.pth",
                model=model,
                tile=400,
                tile_pad=10,
                half=True,
            )

    def upscale_video(self, input_path, output_path,
                      codec="libx264", crf=18,
                      audio_copy=True):
        """
        動画を超解像

        Parameters:
            input_path: 入力動画パス
            output_path: 出力動画パス
            codec: 出力コーデック
            crf: 品質 (0=ロスレス, 51=最低, 18推奨)
            audio_copy: 音声をコピーするか
        """
        cap = cv2.VideoCapture(input_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_w = w * self.scale
        out_h = h * self.scale

        # 一時ファイルに映像のみ出力
        tmp_video = output_path + ".tmp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            tmp_video, fourcc, fps, (out_w, out_h)
        )

        print(f"入力: {w}x{h} @ {fps}fps, {total_frames}フレーム")
        print(f"出力: {out_w}x{out_h}")

        start_time = time.time()
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 超解像
            output, _ = self.upsampler.enhance(
                frame, outscale=self.scale
            )
            writer.write(output)

            frame_idx += 1
            if frame_idx % 10 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_idx / elapsed
                eta = (total_frames - frame_idx) / fps_actual
                print(
                    f"  [{frame_idx}/{total_frames}] "
                    f"{fps_actual:.1f} fps, "
                    f"ETA: {eta:.0f}s"
                )

        cap.release()
        writer.release()

        # ffmpeg で音声を結合
        if audio_copy:
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_video,
                "-i", input_path,
                "-c:v", codec,
                "-crf", str(crf),
                "-c:a", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                output_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            Path(tmp_video).unlink()
        else:
            Path(tmp_video).rename(output_path)

        total_time = time.time() - start_time
        print(f"\n完了: {total_time:.1f}秒 "
              f"({total_frames / total_time:.1f} fps)")

        return {
            "input": input_path,
            "output": output_path,
            "frames": total_frames,
            "elapsed_seconds": round(total_time, 1),
            "avg_fps": round(total_frames / total_time, 2),
        }
```

### 6.2 時間的一貫性の確保

```python
def temporal_consistent_upscale(frames, upsampler, flow_model=None):
    """
    時間的一貫性を保った超解像

    単純にフレーム毎に超解像すると、フレーム間で
    ディテールが揺れる (temporal flickering) 問題が発生する。

    対策:
    1. オプティカルフロー整合化
    2. 前フレームの結果を参照
    3. テンポラルブレンディング
    """

    results = []
    prev_output = None

    for i, frame in enumerate(frames):
        # 現フレームを超解像
        current_output, _ = upsampler.enhance(
            frame, outscale=4
        )

        if prev_output is not None and flow_model is not None:
            # オプティカルフローで前フレームをワープ
            flow = flow_model.estimate(
                frames[i-1], frame
            )
            warped_prev = warp_image(prev_output, flow, scale=4)

            # 現フレームとワープ前フレームをブレンド
            # → テンポラルフリッカリングを抑制
            alpha = 0.3  # ブレンド比率
            occlusion_mask = compute_occlusion_mask(flow)

            current_output = np.where(
                occlusion_mask[..., None],
                current_output,  # オクルージョン領域は現フレームのみ
                (1 - alpha) * current_output
                + alpha * warped_prev  # その他はブレンド
            ).astype(np.uint8)

        results.append(current_output)
        prev_output = current_output

    return results
```

---

## 7. 実務パイプライン統合

### 7.1 REST API による超解像サービス

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import tempfile
import uuid
from pathlib import Path

app = FastAPI(title="Super Resolution API")

# グローバルパイプライン
pipeline = RealESRGANPipeline(device="cuda", half=True)

@app.post("/upscale")
async def upscale_image(
    file: UploadFile = File(...),
    scale: int = 4,
    model: str = "auto",
):
    """
    画像をアップスケールする API エンドポイント

    Parameters:
        file: アップロード画像
        scale: 拡大倍率 (2 or 4)
        model: モデル名 ("photo", "anime", "auto")
    """
    # バリデーション
    if scale not in (2, 4):
        raise HTTPException(400, "scale must be 2 or 4")

    allowed_types = {"image/png", "image/jpeg", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(400, f"Unsupported: {file.content_type}")

    # ファイルサイズ制限 (20MB)
    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 20MB)")

    # 一時ファイルに保存
    job_id = str(uuid.uuid4())
    tmp_dir = Path(tempfile.mkdtemp())
    input_path = tmp_dir / f"input_{job_id}.png"
    output_path = tmp_dir / f"output_{job_id}.png"

    with open(input_path, "wb") as f:
        f.write(contents)

    try:
        model_type = None if model == "auto" else model
        result = pipeline.upscale(
            str(input_path),
            str(output_path),
            model_type=model_type,
            scale=scale,
            tile=400,
        )
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")

    return FileResponse(
        str(output_path),
        media_type="image/png",
        filename=f"upscaled_{file.filename}",
        headers={
            "X-Input-Size": result["input_size"],
            "X-Output-Size": result["output_size"],
            "X-Model": result["model"],
            "X-Elapsed": str(result["elapsed_seconds"]),
        },
    )


@app.post("/upscale/batch")
async def batch_upscale(
    files: list[UploadFile] = File(...),
    scale: int = 4,
    model: str = "auto",
):
    """複数画像の一括アップスケール"""
    if len(files) > 50:
        raise HTTPException(400, "Max 50 files per batch")

    results = []
    for file in files:
        try:
            # 各ファイルを個別処理
            result = await upscale_image(file, scale, model)
            results.append({
                "filename": file.filename,
                "status": "success",
            })
        except HTTPException as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "detail": e.detail,
            })

    return {"results": results}
```

### 7.2 Gradio による GUI アプリケーション

```python
import gradio as gr
from PIL import Image
import numpy as np

def create_upscaling_ui():
    """Gradio ベースの超解像 GUI"""

    pipeline = RealESRGANPipeline(device="cuda")

    def upscale_handler(image, model_choice, scale, tile_size):
        """超解像処理ハンドラ"""
        if image is None:
            return None, "画像をアップロードしてください"

        # PIL → 一時ファイル → 処理 → PIL
        import tempfile
        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False
        ) as tmp:
            Image.fromarray(image).save(tmp.name)
            result = pipeline.upscale(
                tmp.name,
                model_type=(
                    None if model_choice == "自動判定"
                    else model_choice.lower()
                ),
                scale=scale,
                tile=tile_size,
            )

        output = Image.open(result["output"])
        info = (
            f"入力: {result['input_size']} → "
            f"出力: {result['output_size']}\n"
            f"モデル: {result['model']}\n"
            f"処理時間: {result['elapsed_seconds']}秒"
        )
        return np.array(output), info

    with gr.Blocks(title="AI 超解像") as demo:
        gr.Markdown("# AI 超解像ツール")

        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="入力画像")
                model_choice = gr.Radio(
                    ["自動判定", "Photo", "Anime"],
                    label="モデル",
                    value="自動判定",
                )
                scale = gr.Slider(
                    minimum=2, maximum=4, step=2,
                    value=4, label="拡大倍率",
                )
                tile_size = gr.Slider(
                    minimum=0, maximum=800, step=100,
                    value=400, label="タイルサイズ (0=無効)",
                )
                btn = gr.Button("超解像を実行", variant="primary")

            with gr.Column():
                output_img = gr.Image(label="出力画像")
                info_text = gr.Textbox(label="処理情報")

        btn.click(
            fn=upscale_handler,
            inputs=[input_img, model_choice, scale, tile_size],
            outputs=[output_img, info_text],
        )

    return demo

# demo = create_upscaling_ui()
# demo.launch(server_name="0.0.0.0", server_port=7860)
```

### 7.3 クラウド API 活用（Replicate / RunPod）

```python
import replicate
import requests
from pathlib import Path

class CloudUpscaler:
    """クラウド API を使った超解像 (ローカルGPU不要)"""

    def __init__(self, provider="replicate"):
        self.provider = provider

    def upscale_replicate(self, image_path, scale=4, model="real-esrgan"):
        """Replicate API で超解像"""
        model_versions = {
            "real-esrgan": "xinntao/realesrgan:latest",
            "supir": "cjwbw/supir:latest",
            "swinir": "jingyunliang/swinir:latest",
        }

        with open(image_path, "rb") as f:
            output = replicate.run(
                model_versions[model],
                input={
                    "image": f,
                    "scale": scale,
                    "face_enhance": True,
                },
            )

        # 結果のダウンロード
        output_path = Path(image_path).stem + "_upscaled.png"
        if isinstance(output, str):
            response = requests.get(output)
            with open(output_path, "wb") as f:
                f.write(response.content)
        elif isinstance(output, list):
            response = requests.get(output[0])
            with open(output_path, "wb") as f:
                f.write(response.content)

        return output_path

    def upscale_runpod(self, image_path, scale=4):
        """RunPod Serverless で超解像"""
        import base64

        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()

        response = requests.post(
            "https://api.runpod.ai/v2/<endpoint_id>/run",
            headers={
                "Authorization": f"Bearer {RUNPOD_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "input": {
                    "image": img_base64,
                    "scale": scale,
                    "model": "realesrgan",
                },
            },
        )

        job_id = response.json()["id"]

        # ポーリングで結果を取得
        import time
        while True:
            status = requests.get(
                f"https://api.runpod.ai/v2/<endpoint_id>/status/{job_id}",
                headers={
                    "Authorization": f"Bearer {RUNPOD_API_KEY}",
                },
            ).json()

            if status["status"] == "COMPLETED":
                output_b64 = status["output"]["image"]
                output_bytes = base64.b64decode(output_b64)
                output_path = Path(image_path).stem + "_upscaled.png"
                with open(output_path, "wb") as f:
                    f.write(output_bytes)
                return output_path

            elif status["status"] == "FAILED":
                raise RuntimeError(f"RunPod error: {status}")

            time.sleep(2)
```

---

## 8. 比較表

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
| **EC商品写真** | Real-ESRGAN + 色補正 | テクスチャ保持と色再現 |
| **監視カメラ映像** | SwinIR | 忠実度重視、証拠保全 |
| **衛星画像** | SwinIR + ドメイン特化微調整 | 地理情報の正確性 |

### 比較表3: 処理速度ベンチマーク

```
テスト条件: 512x512 → 2048x2048 (4x), NVIDIA RTX 4090

┌──────────────────┬───────────┬────────────┬────────────┐
│ モデル           │ 処理時間  │ VRAM使用量 │ スループット│
├──────────────────┼───────────┼────────────┼────────────┤
│ Lanczos (CPU)    │ 0.002s    │ 0          │ 500 img/s  │
│ Real-ESRGAN      │ 0.05s     │ 1.8GB      │ 20 img/s   │
│ Real-ESRGAN FP16 │ 0.03s     │ 0.9GB      │ 33 img/s   │
│ SwinIR-M         │ 0.15s     │ 3.2GB      │ 6.7 img/s  │
│ SwinIR-L         │ 0.35s     │ 5.1GB      │ 2.9 img/s  │
│ SD x4 Upscaler   │ 3.5s      │ 5.8GB      │ 0.29 img/s │
│ StableSR (50step)│ 8.2s      │ 7.5GB      │ 0.12 img/s │
│ SUPIR (50step)   │ 25s       │ 14GB       │ 0.04 img/s │
└──────────────────┴───────────┴────────────┴────────────┘

※ バッチ処理時はモデルロード時間を除く
※ タイル処理有効時は画像サイズに比例して増加
```

---

## 9. アンチパターン

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

### アンチパターン3: VRAM不足を無視してフルサイズ処理

```
[問題]
8000x6000 の画像を tile=0 (タイル処理無効) のまま
超解像しようとして OOM (Out of Memory) エラーが発生する。

[なぜ問題か]
- VRAM は画像サイズの二乗に比例して消費される
- 4x超解像では出力が 32000x24000 になり膨大なメモリが必要
- OOM発生時にGPUプロセスが残留し、他の処理にも影響

[正しいアプローチ]
- 常に tile パラメータを設定する (推奨: 400-512)
- 入力サイズに応じて自動的にタイルサイズを調整
- VRAM監視を組み込み、危険な場合は警告を出す
- try-except で OOM をキャッチし、タイルサイズを縮小して再試行
```

### アンチパターン4: 超解像結果を無検証で納品

```
[問題]
超解像の出力をそのまま最終成果物として使用する。

[なぜ問題か]
- GAN系は「ハルシネーション」を起こすことがある
  (存在しないテクスチャ/パターンの生成)
- 顔が微妙に変わる場合がある
- JPEG圧縮アーティファクトが増幅される場合がある
- 色味が変化する場合がある

[正しいアプローチ]
- 超解像後に人間によるレビューを必ず行う
- 品質メトリクス (PSNR, SSIM, LPIPS) を自動チェック
- 閾値を設けて異常値の場合はアラートを出す
- 特に顔や文字の部分は重点的に確認する
```

### アンチパターン5: 圧縮済み画像にさらにJPEG保存

```
[問題]
JPEG画像を超解像し、結果を再度JPEGで保存する。

[なぜ問題か]
- 超解像で改善したディテールがJPEG圧縮で失われる
- 二重のJPEG圧縮アーティファクト
- 特にエッジ部分でモスキートノイズが発生

[正しいアプローチ]
- 超解像の出力は必ず PNG (ロスレス) で保存
- 最終的にJPEGが必要な場合は quality=95 以上
- WebP (ロスレスモード) も良い選択肢
- ワークフロー全体でロスレスフォーマットを維持
```

---

## 10. トラブルシューティング

### 10.1 よくあるエラーと解決策

```python
class UpscaleErrorHandler:
    """超解像でよくあるエラーのハンドリング"""

    @staticmethod
    def handle_oom(func):
        """OOM エラー時にタイルサイズを縮小して再試行"""
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tile_sizes = [0, 512, 400, 256, 128]
            last_error = None

            for tile_size in tile_sizes:
                try:
                    kwargs["tile"] = tile_size
                    return func(*args, **kwargs)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        last_error = e
                        # GPU メモリをクリア
                        import torch
                        torch.cuda.empty_cache()
                        if tile_size == 0:
                            print(
                                "VRAM不足: タイル処理に切り替え "
                                f"(tile={tile_sizes[1]})"
                            )
                        else:
                            print(
                                f"VRAM不足: タイル縮小 "
                                f"(tile={tile_size}→次のサイズ)"
                            )
                    else:
                        raise

            raise RuntimeError(
                f"全タイルサイズで OOM: {last_error}"
            )

        return wrapper

    @staticmethod
    def validate_input(image_path):
        """入力画像のバリデーション"""
        import cv2
        from pathlib import Path

        path = Path(image_path)

        # ファイル存在チェック
        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {path}")

        # 拡張子チェック
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
        if path.suffix.lower() not in valid_exts:
            raise ValueError(
                f"非対応フォーマット: {path.suffix}\n"
                f"対応: {valid_exts}"
            )

        # 画像読み込みテスト
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"画像読み込み失敗 (破損?): {path}")

        h, w = img.shape[:2]

        # サイズチェック
        if h < 16 or w < 16:
            raise ValueError(
                f"画像が小さすぎます: {w}x{h} (最小: 16x16)"
            )

        if h > 10000 or w > 10000:
            print(
                f"警告: 大画像 ({w}x{h})。"
                "タイル処理を推奨します。"
            )

        # メモリ見積もり (4x超解像)
        estimated_vram_gb = (
            w * h * 3 * 4 * 16  # 概算
        ) / (1024 ** 3)

        return {
            "path": str(path),
            "size": f"{w}x{h}",
            "channels": img.shape[2] if len(img.shape) > 2 else 1,
            "estimated_vram_gb": round(estimated_vram_gb, 2),
            "recommendation": (
                "タイル処理推奨" if w * h > 2_000_000
                else "タイル処理不要"
            ),
        }
```

### 10.2 色味の変化への対処

```python
def post_process_color_correction(
    original_path, upscaled_path, output_path,
    method="histogram_matching"
):
    """
    超解像後の色補正

    超解像モデルは色味を変える場合がある。
    元画像の色分布を参照して補正する。
    """
    import cv2
    import numpy as np

    original = cv2.imread(original_path)
    upscaled = cv2.imread(upscaled_path)

    # 元画像を超解像サイズにリサイズ (参照用)
    h, w = upscaled.shape[:2]
    original_resized = cv2.resize(
        original, (w, h), interpolation=cv2.INTER_LANCZOS4
    )

    if method == "histogram_matching":
        # ヒストグラムマッチング
        result = np.zeros_like(upscaled)
        for ch in range(3):
            result[:, :, ch] = _match_histogram(
                upscaled[:, :, ch],
                original_resized[:, :, ch],
            )

    elif method == "color_transfer":
        # LAB色空間での統計的色転写
        result = _lab_color_transfer(original_resized, upscaled)

    elif method == "linear":
        # 線形回帰による色補正
        result = _linear_color_correction(
            original_resized, upscaled
        )

    cv2.imwrite(output_path, result)
    return output_path


def _match_histogram(source, reference):
    """ヒストグラムマッチング (単一チャネル)"""
    src_values, src_unique_indices, src_counts = np.unique(
        source.ravel(), return_inverse=True, return_counts=True
    )
    ref_values, ref_counts = np.unique(
        reference.ravel(), return_counts=True
    )

    src_cdf = np.cumsum(src_counts).astype(np.float64)
    src_cdf /= src_cdf[-1]

    ref_cdf = np.cumsum(ref_counts).astype(np.float64)
    ref_cdf /= ref_cdf[-1]

    interp_values = np.interp(src_cdf, ref_cdf, ref_values)
    return interp_values[src_unique_indices].reshape(source.shape).astype(
        np.uint8
    )


def _lab_color_transfer(source, target):
    """LAB色空間での色転写 (Reinhard et al.)"""
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    for ch in range(3):
        src_mean = source_lab[:, :, ch].mean()
        src_std = source_lab[:, :, ch].std()
        tgt_mean = target_lab[:, :, ch].mean()
        tgt_std = target_lab[:, :, ch].std()

        target_lab[:, :, ch] = (
            (target_lab[:, :, ch] - tgt_mean)
            * (src_std / (tgt_std + 1e-8))
            + src_mean
        )

    target_lab = np.clip(target_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(target_lab, cv2.COLOR_LAB2BGR)
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

### Q4: 超解像モデルをファインチューニングするには?

**A:** ドメイン特化の超解像が必要な場合、以下の手順でファインチューニングできます:

1. **データセット準備:** 高解像度画像を収集 (最低500枚、理想は5000枚以上)
2. **劣化ペア作成:** 高解像度画像から劣化モデルで低解像度画像を生成
3. **Real-ESRGAN のファインチューニング:**

```python
# BasicSR の設定ファイル例
# finetune_realesrgan_x4plus.yml

name: finetune_RealESRGANx4plus
model_type: RealESRGANModel
scale: 4
num_gpu: 1

datasets:
  train:
    name: custom_dataset
    type: RealESRGANPairedDataset
    dataroot_gt: /data/train/HR  # 高解像度画像
    dataroot_lq: /data/train/LR  # 低解像度画像
    io_backend:
      type: disk
    gt_size: 256
    use_hflip: true
    use_rot: true

# ネットワーク設定
network_g:
  type: RRDBNet
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 64
  num_block: 23
  num_grow_ch: 32
  scale: 4

# 学習設定
train:
  optim_g:
    type: Adam
    lr: !!float 1e-4
    weight_decay: 0
    betas: [0.9, 0.99]

  # 事前学習済みモデルから開始
  path:
    pretrain_network_g: weights/RealESRGAN_x4plus.pth
    strict_load_g: true

  total_iter: 50000
  warmup_iter: -1
```

### Q5: WebP や AVIF など新しいフォーマットでの超解像は?

**A:** 入力フォーマットに関しては、OpenCV / Pillow が対応していれば問題ありません。重要なのは出力フォーマットの選択です:

| フォーマット | ロスレス | 推奨用途 | 注意点 |
|-------------|---------|---------|--------|
| **PNG** | はい | 中間ファイル、品質最優先 | ファイルサイズ大 |
| **WebP** | 両方 | Web公開、ロスレス保存 | 一部ビューアで非対応 |
| **AVIF** | 両方 | 最新Web、高圧縮 | エンコード遅い |
| **JPEG XL** | 両方 | 将来標準、移行中 | ブラウザ対応限定 |
| **TIFF** | はい | 印刷、アーカイブ | ファイルサイズ大 |
| **JPEG** | いいえ | 最終配信のみ | Q95+推奨 |

### Q6: 複数の超解像モデルの結果をアンサンブルできる?

**A:** 可能です。ただし処理時間が倍増するため、品質が最重要の場合に限ります:

```python
def ensemble_upscale(image_path, models, weights=None):
    """複数モデルの結果を重み付き平均で統合"""
    results = []
    for model in models:
        result = model.upscale(image_path)
        results.append(np.array(result).astype(np.float32))

    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    ensemble = np.zeros_like(results[0])
    for result, weight in zip(results, weights):
        ensemble += result * weight

    return Image.fromarray(
        np.clip(ensemble, 0, 255).astype(np.uint8)
    )
```

---

## まとめ表

| 項目 | 要点 |
|------|------|
| **古典手法** | Lanczos が最良。忠実度100%だがディテール追加なし |
| **GAN系 (Real-ESRGAN)** | 高速・高品質。実写/アニメでモデルを使い分け |
| **Transformer系 (SwinIR)** | GAN系より忠実度が高い。処理はやや遅い |
| **拡散系 (SUPIR)** | 最高品質だが遅い。テキストガイドで制御可能 |
| **顔復元 (GFPGAN)** | 顔特化。背景は Real-ESRGAN と組み合わせ |
| **タイル処理** | 大画像やVRAM不足時の必須テクニック |
| **実用倍率** | 4倍が上限目安。2倍が最も安全 |
| **色補正** | 超解像後のヒストグラムマッチングが有効 |
| **動画超解像** | フレーム単位処理 + 時間的一貫性の確保が重要 |
| **品質評価** | PSNR/SSIM/LPIPS を組み合わせて総合判断 |

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
5. Wang, X. et al. (2021). "GFPGAN: Towards Real-World Blind Face Restoration with Generative Facial Prior." *CVPR 2021*. https://arxiv.org/abs/2101.04061
6. Zhou, S. et al. (2022). "Towards Robust Blind Face Restoration with Codebook Lookup Transformer (CodeFormer)." *NeurIPS 2022*. https://arxiv.org/abs/2206.11253
7. Ledig, C. et al. (2017). "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN)." *CVPR 2017*. https://arxiv.org/abs/1609.04802
8. Dong, C. et al. (2014). "Image Super-Resolution Using Deep Convolutional Networks (SRCNN)." *ECCV 2014*. https://arxiv.org/abs/1501.00092

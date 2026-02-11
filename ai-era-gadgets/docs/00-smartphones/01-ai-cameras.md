# AIカメラ — 計算フォトグラフィ、ナイトモード、AI編集

> スマートフォンカメラにおけるAI技術の全体像を解説する。計算フォトグラフィの原理、ナイトモード・HDR・ポートレートモードの仕組み、そしてAIを活用した写真編集機能まで包括的にカバーする。

---

## この章で学ぶこと

1. **計算フォトグラフィの原理** — 複数フレーム合成、HDR、セマンティック理解による画質向上
2. **ナイトモード / ポートレートの仕組み** — 長時間露光シミュレーション、深度推定、ボケ生成
3. **AI写真編集の実装** — Magic Eraser、背景生成、スタイル変換などの技術

---

## 1. 計算フォトグラフィのパイプライン

```
┌─────────────────────────────────────────────────────────────┐
│              計算フォトグラフィ パイプライン                      │
│                                                               │
│  シャッター押下                                                │
│      │                                                        │
│      ▼                                                        │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │ RAW取得  │──▶│ フレーム  │──▶│ セマンティ│──▶│ ISP +    │   │
│  │ (複数枚) │   │ アライン  │   │ ック分析  │   │ NPU処理  │   │
│  │ 9〜15枚 │   │ メント   │   │ (顔/空等)│   │          │   │
│  └─────────┘   └──────────┘   └──────────┘   └──────────┘   │
│                                                    │          │
│                                                    ▼          │
│                                  ┌──────────┐  ┌──────────┐  │
│                                  │ トーン    │◀─│ ノイズ   │  │
│                                  │ マッピング │  │ 除去(AI) │  │
│                                  └──────────┘  └──────────┘  │
│                                       │                       │
│                                       ▼                       │
│                                  最終JPEG/HEIF                │
└─────────────────────────────────────────────────────────────┘
```

### 1.1 マルチフレーム合成の仕組み

```
┌─────────────────────────────────────────────┐
│           HDR+ マルチフレーム合成              │
│                                               │
│  Frame 1 (暗)  ░░░░░░░░░░                    │
│  Frame 2       ░░░░████░░                    │
│  Frame 3       ░░████████                    │
│  Frame 4 (明)  ████████████                  │
│  ...                                          │
│  Frame 9       ░░░░████░░                    │
│                                               │
│         ↓ アライメント + マージ ↓              │
│                                               │
│  合成結果      ░░████████████ (ダイナミック   │
│                              レンジ拡大)      │
│                                               │
│  ✓ ノイズ低減（√N倍の改善）                   │
│  ✓ ダイナミックレンジ拡大                     │
│  ✓ 手ブレ補正（ロバスト推定）                 │
└─────────────────────────────────────────────┘
```

---

## 2. コード例

### コード例 1: OpenCV によるHDR合成

```python
import cv2
import numpy as np

# 異なる露出の画像を読み込み
images = [cv2.imread(f"exposure_{i}.jpg") for i in range(4)]
exposure_times = np.array([1/30, 1/15, 1/8, 1/4], dtype=np.float32)

# カメラレスポンス関数の推定
calibrate = cv2.createCalibrateDebevec()
response = calibrate.process(images, exposure_times)

# HDR画像の合成
merge = cv2.createMergeDebevec()
hdr_image = merge.process(images, exposure_times, response)

# トーンマッピング（HDR → 表示可能な画像へ）
tonemap = cv2.createTonemap(gamma=2.2)
ldr_image = tonemap.process(hdr_image)
ldr_image = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)

cv2.imwrite("hdr_result.jpg", ldr_image)
print("HDR合成完了: ダイナミックレンジを拡大しました")
```

### コード例 2: AI深度推定によるポートレートモード

```python
import torch
import cv2
import numpy as np

# MiDaS 深度推定モデル
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model.eval()

transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

def portrait_mode(image_path, blur_strength=25):
    """AI深度推定でポートレートモード（背景ぼかし）を実現"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 深度推定
    input_tensor = transform(img_rgb)
    with torch.no_grad():
        depth = model(input_tensor).squeeze().numpy()

    # 深度マップを正規化（0〜1）
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = cv2.resize(depth, (img.shape[1], img.shape[0]))

    # 前景マスク（深度が近いほど1に近い）
    foreground_mask = (depth > 0.5).astype(np.float32)
    foreground_mask = cv2.GaussianBlur(foreground_mask, (21, 21), 0)

    # 背景をぼかす
    blurred = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)

    # 合成: 前景は鮮明、背景はぼかし
    mask_3ch = np.stack([foreground_mask] * 3, axis=-1)
    result = (img * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)

    return result

result = portrait_mode("photo.jpg")
cv2.imwrite("portrait_result.jpg", result)
```

### コード例 3: ナイトモード シミュレーション

```python
import cv2
import numpy as np

def night_mode_simulation(frames, alignment_method="ecc"):
    """
    複数の暗い画像を合成してナイトモードを再現
    原理: N枚平均でノイズが1/√N に減少
    """
    # フレームのアライメント（手ブレ補正）
    reference = frames[0]
    aligned_frames = [reference.astype(np.float64)]

    for frame in frames[1:]:
        # ECC（Enhanced Correlation Coefficient）で位置合わせ
        gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            50, 0.001
        )
        _, warp_matrix = cv2.findTransformECC(
            gray_ref, gray_frame, warp_matrix,
            cv2.MOTION_EUCLIDEAN, criteria
        )

        aligned = cv2.warpAffine(
            frame, warp_matrix,
            (frame.shape[1], frame.shape[0])
        )
        aligned_frames.append(aligned.astype(np.float64))

    # 平均合成（ノイズ低減）
    merged = np.mean(aligned_frames, axis=0)

    # ガンマ補正で明るさ調整
    gamma = 2.0
    merged = np.power(merged / 255.0, 1.0 / gamma) * 255.0

    return merged.astype(np.uint8)

# 使用例: 15フレームを合成
frames = [cv2.imread(f"dark_frame_{i:02d}.jpg") for i in range(15)]
result = night_mode_simulation(frames)
cv2.imwrite("night_mode_result.jpg", result)
```

### コード例 4: ML Kit による顔検出と美顔処理

```kotlin
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions

fun detectAndEnhanceFaces(bitmap: Bitmap) {
    val options = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
        .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
        .build()

    val detector = FaceDetection.getClient(options)
    val image = InputImage.fromBitmap(bitmap, 0)

    detector.process(image)
        .addOnSuccessListener { faces ->
            for (face in faces) {
                val bounds = face.boundingBox
                val smile = face.smilingProbability ?: 0f
                val leftEyeOpen = face.leftEyeOpenProbability ?: 0f

                println("顔検出: ${bounds}")
                println("笑顔度: ${(smile * 100).toInt()}%")
                println("左目開き: ${(leftEyeOpen * 100).toInt()}%")

                // ベストショット選択: 笑顔 + 目が開いている
                if (smile > 0.8f && leftEyeOpen > 0.5f) {
                    println("→ ベストショット候補!")
                }
            }
        }
}
```

### コード例 5: AI消しゴム（Magic Eraser）簡易実装

```python
import cv2
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
import torch

def magic_eraser(image_path, mask_path, prompt="clean background"):
    """
    AI消しゴム: マスク領域を自然に補完する
    Stable Diffusion Inpainting を使用
    """
    # 画像とマスクの読み込み
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # PIL形式に変換
    from PIL import Image
    image_pil = Image.fromarray(image).resize((512, 512))
    mask_pil = Image.fromarray(mask).resize((512, 512))

    # Inpainting パイプライン
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    ).to("cuda")

    result = pipe(
        prompt=prompt,
        image=image_pil,
        mask_image=mask_pil,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    result.save("erased_result.png")
    print("AI消しゴム完了: 不要物を除去しました")

# 使用例
magic_eraser("photo.jpg", "mask.png", "seamless grass background")
```

---

## 3. 比較表

### 比較表 1: 主要スマートフォンのカメラAI機能

| 機能 | iPhone 16 Pro | Pixel 9 Pro | Galaxy S24 Ultra |
|------|-------------|------------|-----------------|
| HDR方式 | Smart HDR 5 | HDR+ with Bracketing | Adaptive HDR |
| ナイトモード | Deep Fusion | Night Sight | Nightography |
| ポートレート | LiDAR + Neural Engine | 機械学習深度推定 | ToF + NPU |
| AI消しゴム | Clean Up | Magic Eraser | Object Eraser |
| 動画HDR | Dolby Vision HDR | HDR+ Video | Super HDR |
| Raw処理 | ProRAW (48MP) | Pro-level RAW | Expert RAW |
| ズーム強化 | 5x光学 + AI超解像 | 30x Super Res Zoom | 100x Space Zoom |

### 比較表 2: 計算フォトグラフィ技術の比較

| 技術 | 原理 | 改善点 | 処理時間 | 必要フレーム数 |
|------|------|--------|---------|-------------|
| HDR+ | マルチフレーム合成 | ダイナミックレンジ | ~200ms | 9〜15枚 |
| ナイトモード | 長時間合成 + AI | 暗所ノイズ低減 | 1〜5秒 | 15〜30枚 |
| Deep Fusion | ピクセル単位最適化 | テクスチャ・ディテール | ~1秒 | 9枚 |
| Super Res Zoom | AIアップスケーリング | デジタルズーム画質 | ~300ms | 複数枚 |
| ポートレート | 深度推定 + ボケ合成 | 背景ぼかし | ~500ms | 1〜2枚 |
| セマンティックHDR | シーン認識 + 領域別処理 | 顔/空の個別最適化 | ~300ms | 9〜15枚 |

---

## 4. アンチパターン

### アンチパターン 1: RAW撮影で計算フォトグラフィを無効にする

```
❌ 誤解:
「RAWで撮ればすべてのAI処理より高画質になる」
→ RAWは計算フォトグラフィ（マルチフレーム合成）が適用されない
→ 特に暗所でノイズが多く、HDR+より劣る場合がある

✅ 正しい理解:
- RAW = 柔軟な後処理が必要な場合（プロフォトグラファー向け）
- JPEG/HEIF = AI最適化済みで多くの場合最良の結果
- ProRAW/Expert RAW = AI処理済み + RAWの柔軟性（推奨）
```

### アンチパターン 2: AI編集の過度な適用

```
❌ 悪い例:
美顔フィルタを最大強度で適用 → 不自然な「蝋人形」効果
AIアップスケーリングを繰り返し適用 → アーティファクト蓄積

✅ 正しいアプローチ:
- AI編集は「補助」として使用し、強度を控えめに設定
- 1回の高品質なAI処理 > 複数回の繰り返し処理
- 元画像を必ず保持し、非破壊編集を心がける
```

---

## 5. FAQ

### Q1: なぜスマートフォンカメラが一眼レフに近づけるのですか？

**A:** 計算フォトグラフィが物理的な制約を補うからです。小さなセンサーでも、9〜30枚のフレームを合成することで、ノイズ低減とダイナミックレンジ拡大を実現します。加えてAIによるシーン認識が、空・肌・テクスチャを個別に最適化します。ただし、大型センサーの物理的な光学ボケや浅い被写界深度は、AIシミュレーションでは完全には再現できません。

### Q2: ナイトモードの「3秒」「5秒」は何をしていますか？

**A:** シャッターを長く開けているのではなく、短い露出（数十ms）のフレームを多数撮影し、アライメント（位置合わせ）してから合成しています。3秒なら約15枚、5秒なら約30枚を使い、√N倍のノイズ改善を得ます。同時にAIがゴーストの除去や色補正も行います。

### Q3: AI消しゴム機能はどの程度信頼できますか？

**A:** 背景が単純（芝生、空、壁など）な場合は非常に自然に消去できます。一方、複雑なテクスチャ（群衆の中の一人、建物の一部など）では不自然なアーティファクトが生じることがあります。Google の Magic Eraser は複数回の適用とプロンプト入力で改善可能です。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 計算フォトグラフィ | マルチフレーム合成でセンサー限界を超える画質を実現 |
| HDR+ | 9〜15枚の合成でダイナミックレンジを拡大 |
| ナイトモード | √N倍のノイズ改善 + AI補正で暗所撮影を革新 |
| ポートレート | 深度推定AIで一眼レフ風ボケを再現 |
| AI編集 | 消しゴム、アップスケーリング、スタイル変換が端末上で可能 |
| セマンティック処理 | シーン/被写体認識で領域別に最適化処理 |

---

## 次に読むべきガイド

- [AIアシスタント — Siri/Google Assistant/Alexa](./02-ai-assistants.md)
- [ウェアラブル — Apple Watch/Galaxy Watch](./03-wearables.md)
- [コンピュータビジョン — 物体検出、セグメンテーション](../../ai-analysis-guide/docs/03-applied/01-computer-vision.md)

---

## 参考文献

1. **Google Research** — "HDR+ with Bracketing on Pixel Phones," Google AI Blog, 2023
2. **Apple** — "Deep Fusion and Photonic Engine," developer.apple.com, 2024
3. **Levoy, M.** — "Computational Photography: From Selfies to Black Holes," Google, 2019
4. **Ranftl, R. et al.** — "Towards Robust Monocular Depth Estimation," arXiv:1907.01341, 2021

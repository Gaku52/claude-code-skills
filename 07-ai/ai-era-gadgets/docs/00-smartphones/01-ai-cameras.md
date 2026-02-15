# AIカメラ — 計算フォトグラフィ、ナイトモード、AI編集

> スマートフォンカメラにおけるAI技術の全体像を解説する。計算フォトグラフィの原理、ナイトモード・HDR・ポートレートモードの仕組み、そしてAIを活用した写真編集機能まで包括的にカバーする。

---

## この章で学ぶこと

1. **計算フォトグラフィの原理** — 複数フレーム合成、HDR、セマンティック理解による画質向上
2. **ナイトモード / ポートレートの仕組み** — 長時間露光シミュレーション、深度推定、ボケ生成
3. **AI写真編集の実装** — Magic Eraser、背景生成、スタイル変換などの技術
4. **実践的な画像処理パイプライン** — ISP、NPU連携、リアルタイム処理の最適化手法
5. **カメラAI開発の実践** — Core ML / TFLite / MediaPipe を使ったカスタムカメラAI構築

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

### 1.2 ISP (Image Signal Processor) とAIの協調

従来のISPはハードウェアパイプラインで固定的な画像処理を行っていましたが、現在はNPU/GPU と連携してAIベースの処理を組み合わせるハイブリッド構成が主流です。

```
┌─────────────────────────────────────────────────────┐
│           ISP + NPU ハイブリッドパイプライン           │
│                                                       │
│  RAW Bayer データ                                     │
│      │                                                │
│      ▼                                                │
│  ┌──────────────────────────────────────────┐         │
│  │ ハードウェア ISP（固定パイプライン）        │         │
│  │  - デモザイキング（Bayer → RGB）           │         │
│  │  - ブラックレベル補正                      │         │
│  │  - レンズシェーディング補正                 │         │
│  │  - ホワイトバランス                        │         │
│  │  - ノイズ除去（基本フィルタ）               │         │
│  └──────────────┬───────────────────────────┘         │
│                  │                                     │
│      ┌───────────┼───────────┐                         │
│      ▼           ▼           ▼                         │
│  ┌────────┐ ┌────────┐ ┌────────┐                     │
│  │ NPU    │ │ GPU    │ │ CPU    │                     │
│  │ セマン │ │ トーン │ │ メタ   │                     │
│  │ ティック│ │ マッピ │ │ データ │                     │
│  │ 分析   │ │ ング   │ │ 処理   │                     │
│  └────────┘ └────────┘ └────────┘                     │
│      │           │           │                         │
│      └───────────┼───────────┘                         │
│                  ▼                                     │
│           最終出力画像                                  │
└─────────────────────────────────────────────────────┘

処理時間の内訳（iPhone 16 Pro の場合）:
  ISP ハードウェア処理:   ~15ms
  NPU セマンティック分析:  ~8ms
  GPU トーンマッピング:    ~5ms
  合計:                   ~28ms（リアルタイムプレビュー可能）
```

### 1.3 セマンティックセグメンテーションの役割

現代のスマートフォンカメラは、撮影時にシーン全体をセマンティックに理解してから画像処理を行います。

```
入力画像のセマンティック分解:

┌──────────────────────────────────┐
│ ┌──────────────────────────────┐ │
│ │ 空 (SKY)                     │ │  → 青を強調、ハイライト回復
│ │  ☁️ 雲 (CLOUDS)              │ │  → テクスチャ保持、白飛び防止
│ ├──────────────────────────────┤ │
│ │ 建物 (BUILDING)              │ │  → エッジ強調、歪み補正
│ ├─────────────┬────────────────┤ │
│ │ 人物 (PERSON) │ 背景 (BG)    │ │  → 人物: 肌色最適化、ボケ: 背景ぼかし
│ │  👤 顔 (FACE) │ 🌳 植物    │ │  → 顔: 露出優先、植物: 彩度強調
│ └─────────────┴────────────────┘ │
│ 地面 (GROUND)                    │  → ノイズ除去、ディテール保持
└──────────────────────────────────┘

各領域に対して異なるトーンカーブ・ノイズ除去・シャープネスを適用
= 「セマンティックHDR」（Apple Deep Fusion, Google HDR+ の本質）
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

### コード例 6: Core ML を使ったカスタム画像フィルタ（iOS）

```swift
import CoreML
import Vision
import CoreImage
import UIKit

class AIImageFilter {
    /// Core ML モデルを使ったリアルタイム画像フィルタ

    let model: VNCoreMLModel
    let context = CIContext()

    init() throws {
        // 事前に変換したセグメンテーションモデルをロード
        let config = MLModelConfiguration()
        config.computeUnits = .all  // CPU + GPU + Neural Engine
        let segModel = try DeepLabV3(configuration: config)
        self.model = try VNCoreMLModel(for: segModel.model)
    }

    func applyPortraitEffect(to image: UIImage) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }

        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNPixelBufferObservation],
                  let segMask = results.first?.pixelBuffer else { return }

            // セグメンテーションマスクを CIImage に変換
            let maskImage = CIImage(cvPixelBuffer: segMask)
            let originalImage = CIImage(cgImage: cgImage)

            // 背景にガウスぼかしを適用
            let blurred = originalImage.applyingGaussianBlur(sigma: 15)

            // マスクで合成（前景: 鮮明、背景: ぼかし）
            let composite = originalImage
                .applyingFilter("CIBlendWithMask", parameters: [
                    "inputBackgroundImage": blurred,
                    "inputMaskImage": maskImage
                ])
        }

        let handler = VNImageRequestHandler(cgImage: cgImage)
        try? handler.perform([request])

        return nil // 実際にはコールバックで結果を返す
    }

    /// カメラプレビューでのリアルタイムフィルタ適用
    func processLiveFrame(_ sampleBuffer: CMSampleBuffer) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let request = VNCoreMLRequest(model: model) { request, _ in
            // Neural Engine で ~8ms で推論完了
            // フレームレート 30fps を維持可能
        }

        // パフォーマンス最適化: 入力を縮小して推論
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        try? handler.perform([request])
    }
}
```

### コード例 7: MediaPipe を使った顔メッシュ検出と美顔フィルタ

```python
import mediapipe as mp
import cv2
import numpy as np

class AIBeautyFilter:
    """
    MediaPipe Face Mesh による468ポイント顔メッシュ検出と
    リアルタイム美顔フィルタの実装
    """
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,   # 虹彩検出も有効化
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame, smooth_skin=True,
                      brighten_eyes=True, slim_face=False):
        """リアルタイム美顔処理パイプライン"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return frame

        output = frame.copy()

        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = [(int(l.x * w), int(l.y * h))
                        for l in face_landmarks.landmark]

            if smooth_skin:
                output = self._smooth_skin(output, landmarks)

            if brighten_eyes:
                output = self._brighten_eyes(output, landmarks)

            if slim_face:
                output = self._slim_face(output, landmarks)

        return output

    def _smooth_skin(self, frame, landmarks):
        """
        肌のスムージング処理
        ビラテラルフィルタで肌の質感を保ちつつノイズ除去
        """
        # 顔領域のマスクを生成（顔の輪郭ランドマーク使用）
        face_outline = [10, 338, 297, 332, 284, 251, 389, 356,
                       454, 323, 361, 288, 397, 365, 379, 378,
                       400, 377, 152, 148, 176, 149, 150, 136,
                       172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

        points = np.array([landmarks[i] for i in face_outline])
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, points, 255)

        # 目・口の領域を除外（ぼかしたくない部分）
        left_eye = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153]
        right_eye = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380]
        mouth = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

        for region in [left_eye, right_eye, mouth]:
            pts = np.array([landmarks[i] for i in region])
            cv2.fillConvexPoly(mask, pts, 0)

        # ビラテラルフィルタ（エッジ保持スムージング）
        smoothed = cv2.bilateralFilter(frame, 9, 75, 75)

        # マスクで合成
        mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
        result = (smoothed * mask_3ch + frame * (1 - mask_3ch)).astype(np.uint8)

        return result

    def _brighten_eyes(self, frame, landmarks):
        """目元を明るくする処理"""
        eye_indices = [
            [33, 133, 160, 159, 158, 157, 173, 144, 145, 153],  # 左目
            [362, 263, 387, 386, 385, 384, 398, 373, 374, 380],  # 右目
        ]

        for indices in eye_indices:
            pts = np.array([landmarks[i] for i in indices])
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts, 255)

            # ガンマ補正で明るく
            eye_region = frame.copy()
            gamma = 1.3
            lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                           for i in range(256)]).astype(np.uint8)
            eye_region = cv2.LUT(eye_region, lut)

            mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
            frame = (eye_region * mask_3ch + frame * (1 - mask_3ch)).astype(np.uint8)

        return frame

    def _slim_face(self, frame, landmarks):
        """顔のスリム効果（ワープ処理）"""
        # 頬のランドマークを内側に移動
        # 液化（Liquify）変換で実装
        h, w = frame.shape[:2]

        # 左右の頬の基準点
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        chin = landmarks[152]

        # 基準点に向かってワープ（簡易版）
        # 実際の実装ではThin Plate Spline等を使用
        map_x = np.float32([[i for i in range(w)] for _ in range(h)])
        map_y = np.float32([[j for _ in range(w)] for j in range(h)])

        # 頬を中心に向かって2%収縮
        center_x = w // 2
        strength = 0.02
        for y in range(h):
            for x in range(w):
                dx = x - center_x
                if abs(dx) > w * 0.1:  # 頬の領域のみ
                    map_x[y][x] -= dx * strength

        result = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        return result

# 使用例: Webカメラでリアルタイム美顔
filter_engine = AIBeautyFilter()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    result = filter_engine.process_frame(frame, smooth_skin=True, brighten_eyes=True)
    cv2.imshow("AI Beauty Filter", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### コード例 8: AIアップスケーリング（超解像）

```python
import torch
import torch.nn as nn
import cv2
import numpy as np

class ESPCN(nn.Module):
    """
    Efficient Sub-Pixel CNN (ESPCN) による超解像
    モバイルデバイスでも動作する軽量モデル

    なぜESPCNか:
    - PixelShuffle で効率的なアップサンプリング
    - 低解像度空間で特徴抽出 → 計算量削減
    - リアルタイム処理が可能（~5ms on GPU）
    """
    def __init__(self, upscale_factor=2, num_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, num_channels * (upscale_factor ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

def super_resolve(image_path, scale=2):
    """画像を AI で高解像度化"""
    model = ESPCN(upscale_factor=scale)
    model.load_state_dict(torch.load("espcn_x2.pth", map_location="cpu"))
    model.eval()

    # 画像読み込みと前処理
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Y チャンネル（輝度）のみ超解像
    y_channel = img_ycrcb[:, :, 0].astype(np.float32) / 255.0
    y_tensor = torch.from_numpy(y_channel).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        y_sr = model(y_tensor).squeeze().numpy()

    y_sr = np.clip(y_sr * 255, 0, 255).astype(np.uint8)

    # CrCb チャンネルを双三次補間でアップスケール
    cr = cv2.resize(img_ycrcb[:, :, 1],
                    (y_sr.shape[1], y_sr.shape[0]),
                    interpolation=cv2.INTER_CUBIC)
    cb = cv2.resize(img_ycrcb[:, :, 2],
                    (y_sr.shape[1], y_sr.shape[0]),
                    interpolation=cv2.INTER_CUBIC)

    # 合成して出力
    result_ycrcb = cv2.merge([y_sr, cr, cb])
    result = cv2.cvtColor(result_ycrcb, cv2.COLOR_YCrCb2BGR)

    cv2.imwrite(f"sr_x{scale}_{image_path}", result)
    print(f"超解像完了: {img.shape[:2]} → {result.shape[:2]}")
    return result

# 使用例
super_resolve("low_res_photo.jpg", scale=2)
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

### 比較表 3: 深度推定方式の比較

| 方式 | 精度 | コスト | 消費電力 | 屋外/屋内 | 対応デバイス |
|------|------|--------|---------|----------|------------|
| LiDAR | 非常に高い | 高 | 中 | 両方（屋外はやや弱い） | iPhone Pro, iPad Pro |
| ToF (Time of Flight) | 高い | 中 | 中 | 屋内に強い | Galaxy S24 Ultra |
| ステレオカメラ | 中〜高 | 中 | 低 | 両方 | 一部Android |
| AI単眼深度推定 | 中 | 低（ソフトウェア） | 低 | 両方 | 全スマートフォン |
| 構造化光 | 高い | 中 | 中 | 屋内のみ | Face ID用 |

### 比較表 4: AI写真編集機能の詳細比較

| 機能 | Google Magic Editor | Apple Clean Up | Samsung Photo Assist |
|------|-------------------|---------------|---------------------|
| オブジェクト消去 | Magic Eraser | Clean Up | Object Eraser |
| 背景変更 | Reimagine（生成AI） | 非対応 | 限定的 |
| 被写体移動 | Magic Editor | 非対応 | 非対応 |
| 空の置換 | 自動提案 | 非対応 | Sky Guide |
| リサイズ/拡張 | Best Take | 非対応 | 非対応 |
| AI処理場所 | クラウド（Tensor Cloud） | オンデバイス | ハイブリッド |
| プライバシー | Google Photos必須 | 端末内完結 | Samsung Cloud可 |

---

## 4. 実践的ユースケース

### ユースケース 1: リアルタイムドキュメントスキャン

```python
import cv2
import numpy as np

class AIDocumentScanner:
    """
    AIを活用したドキュメントスキャナー
    - 自動エッジ検出で書類の四隅を特定
    - 透視変換で正面からの画像に補正
    - 二値化とノイズ除去で読みやすく変換
    """
    def __init__(self):
        self.edge_model = None  # 本番ではMLモデルを使用

    def detect_document(self, frame):
        """書類のエッジを検出"""
        # 前処理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # 輪郭検出
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:5]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:  # 四角形を検出
                return approx.reshape(4, 2)

        return None

    def perspective_transform(self, frame, corners):
        """透視変換で正面画像に補正"""
        # 四隅を並べ替え（左上、右上、右下、左下）
        rect = self._order_points(corners)
        (tl, tr, br, bl) = rect

        # 出力サイズの計算
        width = max(
            np.linalg.norm(br - bl),
            np.linalg.norm(tr - tl)
        )
        height = max(
            np.linalg.norm(tr - br),
            np.linalg.norm(tl - bl)
        )

        dst = np.array([
            [0, 0], [width - 1, 0],
            [width - 1, height - 1], [0, height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
        warped = cv2.warpPerspective(frame, M, (int(width), int(height)))

        return warped

    def enhance_document(self, image):
        """AIベースの文書画像強調"""
        # 適応的二値化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # CLAHE（コントラスト制限付き適応的ヒストグラム均等化）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 適応的閾値処理
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        return binary

    def _order_points(self, pts):
        """四隅を左上→右上→右下→左下の順に並べ替え"""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # 左上
        rect[2] = pts[np.argmax(s)]   # 右下
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # 右上
        rect[3] = pts[np.argmax(diff)] # 左下
        return rect

# 使用例
scanner = AIDocumentScanner()
frame = cv2.imread("document_photo.jpg")
corners = scanner.detect_document(frame)
if corners is not None:
    warped = scanner.perspective_transform(frame, corners)
    result = scanner.enhance_document(warped)
    cv2.imwrite("scanned_document.jpg", result)
```

### ユースケース 2: AI食事認識と栄養分析

```python
import torch
from torchvision import transforms, models
from PIL import Image
import json

class FoodAIAnalyzer:
    """
    AI食事認識: 写真から食材を識別し栄養情報を推定
    スマートフォンの NPU で動作可能な軽量モデルを使用
    """
    def __init__(self):
        # 食品分類モデル（MobileNetV3をFine-tuning済み）
        self.model = models.mobilenet_v3_small(pretrained=False)
        self.model.classifier[-1] = torch.nn.Linear(1024, 256)  # 256種の食品
        self.model.load_state_dict(torch.load("food_classifier.pth"))
        self.model.eval()

        # 栄養データベース（100gあたり）
        self.nutrition_db = {
            "white_rice": {"calories": 168, "protein": 2.5, "carbs": 37, "fat": 0.3},
            "grilled_salmon": {"calories": 208, "protein": 20, "carbs": 0, "fat": 13},
            "miso_soup": {"calories": 40, "protein": 3.3, "carbs": 4.3, "fat": 1.0},
            "salad": {"calories": 15, "protein": 1.2, "carbs": 2.5, "fat": 0.2},
        }

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def analyze_meal(self, image_path):
        """食事画像を分析"""
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)

            # 上位5件の予測
            top5_prob, top5_idx = torch.topk(probabilities, 5)

        results = []
        for prob, idx in zip(top5_prob[0], top5_idx[0]):
            food_name = self.idx_to_name[idx.item()]
            nutrition = self.nutrition_db.get(food_name, {})
            results.append({
                "food": food_name,
                "confidence": f"{prob.item():.1%}",
                "nutrition_per_100g": nutrition
            })

        return results
```

---

## 5. トラブルシューティング

### 問題 1: ナイトモードで手ブレが発生する

```
症状: ナイトモードの撮影結果がブレている

原因と対処法:
1. 長い露光時間（3-5秒）で手持ち撮影
   → 三脚またはスマートフォンスタンドを使用
   → 壁や机に端末を固定して撮影

2. 合成フレーム数が不足
   → ナイトモードの秒数を長く設定（可能な場合）
   → iPhone: シャッターボタン長押しで時間を延長

3. OIS（光学手ブレ補正）の限界
   → 歩きながらの撮影は避ける
   → シャッター後に端末を動かさない

4. 被写体ブレ（動く被写体）
   → ナイトモードでは動く被写体はゴーストが出る
   → 動く被写体にはフラッシュまたは通常モードを使用
```

### 問題 2: ポートレートモードのボケが不自然

```
症状: 髪の毛の境界がぼかしに巻き込まれる、メガネに偽ボケがかかる

原因と対処法:
1. 深度推定の精度不足（AI単眼推定の場合）
   → LiDAR対応デバイスを使用（iPhone Pro以上）
   → 被写体と背景の距離を1.5m以上確保
   → 単色背景を避ける（テクスチャが多い背景で精度向上）

2. 透明/反射物体の問題
   → ガラスコップ、メガネは深度推定が困難
   → 手動でフォーカスポイントを調整

3. エッジのハロー効果
   → 撮影後にポートレート編集でボケ量を下げる
   → Apple: 撮影後に f値を変更可能
   → Google: ボケの強度スライダーで調整

4. 複数人物の深度が正しくない
   → 全員が同じ深度平面にいるようにする
   → 横一列ではなく前後に並ぶと改善
```

### 問題 3: AI消しゴムで不自然な結果になる

```
症状: 消去部分にアーティファクトが残る、テクスチャが不自然

原因と対処法:
1. 消去対象が大きすぎる
   → 小さな領域ずつ段階的に消去
   → Google Magic Eraser: 複数回に分けてなぞる

2. 背景が複雑（パターン、テクスチャ）
   → 消去が困難な場合は別の角度から再撮影
   → 生成AIベースの編集（Google Reimagine）を試す

3. エッジが残る
   → 消去範囲を少し広めに指定
   → 消去後に追加で細部を修正

4. 反復適用によるアーティファクト蓄積
   → 1回の高品質な処理 > 複数回の繰り返し
   → 元画像を保持し、やり直しが可能なワークフローで
```

### 問題 4: HDR撮影でハロー（にじみ）が出る

```
症状: 明暗の境界にハロー（白いにじみ）が発生

原因:
- トーンマッピングが過度（ローカルトーンマッピングの副作用）
- HDR合成時のゴーストアーティファクト

対処法:
1. HDR強度を下げる（設定可能な場合）
   → Samsung: HDR自動ではなくマニュアルHDRに切り替え

2. RAWで撮影し、後からHDR処理
   → Adobe Lightroom で適切なトーンカーブを適用
   → ProRAW / Expert RAW を活用

3. ファームウェアアップデートを確認
   → HDRアルゴリズムはソフトウェア更新で改善される

4. コントラストが極端なシーンを避ける
   → 直射日光下の逆光は最も困難なケース
   → 反射板（レフ板）で影を軽減
```

---

## 6. パフォーマンス最適化Tips

### Tip 1: カメラAIモデルの最適化チェックリスト

```
カメラAI パフォーマンス最適化チェックリスト:

□ モデルサイズ
  ├── MobileNetV3 / EfficientNet-Lite を使用（~5MB以下）
  ├── 不要なレイヤーをプルーニング
  └── INT8量子化を適用（精度影響を検証済み）

□ 入力解像度
  ├── 推論用の入力は最小限に（224x224 or 320x320）
  ├── カメラプレビュー用とキャプチャ用で異なる解像度を使用
  └── ROI（関心領域）のみをクロップして推論

□ 推論エンジン
  ├── iOS: Core ML（Neural Engine自動活用）
  ├── Android: TFLite + NNAPI（NPU活用）
  ├── 汎用: ONNX Runtime（最適なEPを自動選択）
  └── NVIDIA: TensorRT（Jetson等のエッジデバイス）

□ フレーム処理
  ├── 全フレームではなく間引いて推論（30fps → 10fps推論）
  ├── 推論と描画を別スレッドで並列実行
  ├── 前フレームの結果をキャッシュして補間
  └── バッチ推論が可能なら複数フレームをまとめる

□ メモリ管理
  ├── モデルは1回だけロード、使い回す
  ├── 入出力バッファは事前確保（毎フレーム確保しない）
  ├── Core ML: MLModelConfiguration.computeUnits = .all
  └── TFLite: Interpreter.Options で GPU Delegate 有効化
```

### Tip 2: バッテリー効率の最適化

```
┌──────────────────────────────────────────────┐
│     カメラAI バッテリー効率の最適化            │
├──────────────────────────────────────────────┤
│                                                │
│  消費電力が高い処理:                           │
│  ├── リアルタイムセグメンテーション: ~2W       │
│  ├── 常時顔検出: ~1.5W                        │
│  ├── 動画HDR: ~3W                              │
│  └── AIフィルタプレビュー: ~2.5W              │
│                                                │
│  最適化戦略:                                   │
│  ├── 静止画撮影時のみAI処理を実行             │
│  │   → プレビュー時はダウンスケールした軽量推論│
│  ├── NPUに適した処理はNPUに委譲               │
│  │   → GPU より NPU の方が 5-10倍 省電力      │
│  ├── 顔検出の頻度を状況に応じて変更           │
│  │   → 顔が検出されたら高頻度、未検出なら低頻度│
│  └── サーマルスロットリングを検出して品質を下げる│
│      → API でチップ温度を監視                  │
└──────────────────────────────────────────────┘
```

### Tip 3: 画質評価指標の自動化

```python
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def evaluate_image_quality(original, processed):
    """
    画像処理AIの品質評価メトリクス
    モデル選定やハイパーパラメータ調整に使用
    """
    # PSNR（ピーク信号対雑音比）: 高いほど良い
    psnr_value = psnr(original, processed)

    # SSIM（構造的類似度）: 1に近いほど良い
    ssim_value = ssim(original, processed, channel_axis=2)

    # BRISQUE（無参照画質評価）: 低いほど良い
    # ブラインド/リファレンスレス画質評価
    # OpenCV の quality モジュールを使用

    # 色相の一貫性チェック
    hsv_orig = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    hsv_proc = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
    hue_diff = np.mean(np.abs(hsv_orig[:,:,0].astype(float) -
                               hsv_proc[:,:,0].astype(float)))

    results = {
        "PSNR (dB)": f"{psnr_value:.2f}",      # 目安: 30dB以上
        "SSIM": f"{ssim_value:.4f}",             # 目安: 0.90以上
        "色相差 (平均)": f"{hue_diff:.2f}",      # 目安: 5以下
        "判定": "PASS" if psnr_value > 30 and ssim_value > 0.90 else "FAIL"
    }

    return results

# 使用例: AI処理前後の品質を評価
original = cv2.imread("original.jpg")
processed = cv2.imread("ai_enhanced.jpg")
quality = evaluate_image_quality(original, processed)
for k, v in quality.items():
    print(f"  {k}: {v}")
```

---

## 7. 設計パターン

### パターン 1: プログレッシブ画像処理パイプライン

```
リアルタイムプレビューと最終出力で異なる品質レベルを使用:

┌───────────────────────────────────────────────┐
│  プレビュー時（低品質・高速）                   │
│  ├── 入力: 640x480 ダウンスケール              │
│  ├── 深度推定: MiDaS small (~5ms)             │
│  ├── ぼかし: Gaussian (kernel=15)             │
│  └── FPS: 30                                  │
├───────────────────────────────────────────────┤
│  シャッター時（高品質・低速）                   │
│  ├── 入力: フル解像度（4032x3024）             │
│  ├── 深度推定: MiDaS large (~200ms)           │
│  ├── ぼかし: Lens blur simulation (~100ms)    │
│  ├── HDR合成: 9フレーム合成 (~500ms)          │
│  └── 合計: ~800ms                             │
└───────────────────────────────────────────────┘

なぜこのパターンか:
- プレビュー時にフル推論を行うとバッテリーが急速に消耗
- ユーザーはプレビューでの品質差をほとんど認識しない
- シャッター後に高品質処理を行えば十分
```

### パターン 2: フォールバック付きAI処理

```python
class ResilientCameraAI:
    """
    AI処理が失敗した場合のフォールバック付きパイプライン

    なぜ必要か:
    - MLモデルの推論は入力によっては予期しない結果を返す
    - メモリ不足で推論が中断される場合がある
    - NPUが他のアプリに占有されている場合がある
    """
    def __init__(self):
        self.ai_depth_estimator = load_model("midas_small")
        self.fallback_depth = None  # 直前の成功結果をキャッシュ

    def estimate_depth(self, frame):
        try:
            # 優先: AI深度推定（NPU）
            depth = self.ai_depth_estimator.predict(frame)
            self.fallback_depth = depth  # キャッシュ
            return depth, "ai"
        except (RuntimeError, MemoryError):
            # フォールバック1: 直前の推定結果を再利用
            if self.fallback_depth is not None:
                return self.fallback_depth, "cache"

            # フォールバック2: 古典的手法（コントラストベース）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            depth_approx = np.abs(laplacian)
            return depth_approx, "classical"
```

---

## 8. アンチパターン

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

### アンチパターン 3: カメラアプリでのメモリリーク

```
❌ 悪い例:
フレームごとに MLモデルを再ロード、バッファを毎回確保
→ メモリが膨張し、アプリがクラッシュ

✅ 正しいアプローチ:
- モデルは init() で1回だけロード
- 入出力バッファは事前確保して再利用
- AutoreleasepoolまたはGC.collect()で不要なバッファを解放
- カメラセッション終了時にモデルを明示的に解放

// iOS の正しい実装パターン
class CameraProcessor {
    private lazy var model: VNCoreMLModel = {
        // lazy var で初回アクセス時のみロード
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let model = try! DeepLabV3(configuration: config)
        return try! VNCoreMLModel(for: model.model)
    }()

    deinit {
        // 明示的なクリーンアップ
        // Core ML モデルは deinit で自動解放
    }
}
```

### アンチパターン 4: 全フレームにフル推論を適用する

```
❌ 悪い例:
30fps の全フレームに対して重い深度推定 + セグメンテーションを実行
→ 処理が追いつかず、フレームドロップが発生
→ バッテリーが急速に消耗

✅ 正しいアプローチ:
- 推論は間引いて実行（30fps → 10fps推論）
- 中間フレームは前の推論結果を補間（optical flow活用）
- シーンが大きく変わったら即座に再推論
- フレームキューの深さを制限（古い結果は破棄）
```

---

## 9. エッジケース分析

### エッジケース 1: 逆光シーンでの顔検出失敗

逆光（バックライト）シーンでは、顔が暗く潰れてAI顔検出が失敗する。この場合、ハードウェアISPの露出制御とAI顔検出の優先度が競合する。

```
解決策:
1. 露出のプリスキャン: シャッター前の数フレームで顔の有無を高速検出
   → 顔が検出された場合、顔領域の露出に最適化
   → Smart HDR が顔用と背景用で異なる露出フレームを撮影

2. フェイスメータリング:
   → 測光ポイントを顔に自動設定
   → AE (Auto Exposure) が顔基準で露出を決定

3. フィルバック:
   → 顔検出に失敗した場合、中央重点測光にフォールバック
   → ユーザーにタップフォーカスを促すUIを表示
```

### エッジケース 2: 動画撮影中のAI処理と熱管理

長時間の4K動画撮影では、ISP + NPU + GPUの同時使用によりチップ温度が上昇し、サーマルスロットリングが発生する。

```
┌──────────────────────────────────────────┐
│     サーマルスロットリング対策             │
├──────────────────────────────────────────┤
│                                            │
│  チップ温度     AI処理レベル               │
│  ─────────     ──────────                  │
│  < 40℃        フル処理（HDR + スタビライザ │
│                + 顔検出 + ボケ）            │
│  40-50℃       AI処理を間引き              │
│                （顔検出を5fpsに低下）       │
│  50-60℃       スタビライザのみ             │
│                （AI処理を一時停止）         │
│  > 60℃        録画品質を1080pに低下       │
│                （最小限のISP処理のみ）      │
│                                            │
│  ベストプラクティス:                       │
│  - サーマルモニタリングAPIで温度を常時監視  │
│  - 段階的にAI処理の品質を落とす            │
│  - ユーザーに温度警告を表示                │
│  - ケースを外すよう促す（放熱改善）        │
└──────────────────────────────────────────┘
```

---

## 10. 開発者チェックリスト

```
カメラAIアプリ開発チェックリスト:

□ プラットフォーム選択
  □ iOS: AVFoundation + Core ML + Vision
  □ Android: CameraX + TFLite + ML Kit
  □ クロスプラットフォーム: MediaPipe + OpenCV

□ モデル選定
  □ タスクに適した軽量モデルを選択（MobileNet, EfficientNet-Lite）
  □ INT8量子化を適用済み
  □ ターゲットデバイスの NPU/GPU 互換性を確認

□ パフォーマンス
  □ プレビュー時は低解像度推論
  □ キャプチャ時にフル品質推論
  □ 推論スレッドとUIスレッドを分離
  □ FPS ≥ 25（プレビュー表示）

□ メモリ管理
  □ モデルのシングルトンロード
  □ バッファの事前確保と再利用
  □ カメラ停止時のリソース解放

□ バッテリー
  □ NPU を優先的に使用
  □ バックグラウンドでのカメラ使用を制限
  □ サーマルスロットリング対策

□ テスト
  □ 低照度環境でのテスト
  □ 逆光シーンでのテスト
  □ 動く被写体でのテスト
  □ 異なるデバイスでのパフォーマンステスト
```

---

## FAQ

### Q1: なぜスマートフォンカメラが一眼レフに近づけるのですか？

**A:** 計算フォトグラフィが物理的な制約を補うからです。小さなセンサーでも、9〜30枚のフレームを合成することで、ノイズ低減とダイナミックレンジ拡大を実現します。加えてAIによるシーン認識が、空・肌・テクスチャを個別に最適化します。ただし、大型センサーの物理的な光学ボケや浅い被写界深度は、AIシミュレーションでは完全には再現できません。

### Q2: ナイトモードの「3秒」「5秒」は何をしていますか？

**A:** シャッターを長く開けているのではなく、短い露出（数十ms）のフレームを多数撮影し、アライメント（位置合わせ）してから合成しています。3秒なら約15枚、5秒なら約30枚を使い、√N倍のノイズ改善を得ます。同時にAIがゴーストの除去や色補正も行います。

### Q3: AI消しゴム機能はどの程度信頼できますか？

**A:** 背景が単純（芝生、空、壁など）な場合は非常に自然に消去できます。一方、複雑なテクスチャ（群衆の中の一人、建物の一部など）では不自然なアーティファクトが生じることがあります。Google の Magic Eraser は複数回の適用とプロンプト入力で改善可能です。

### Q4: カメラAI開発の入門に最適なフレームワークは？

**A:** まず MediaPipe から始めることを推奨します。顔検出、ポーズ推定、セグメンテーションなどの事前学習済みモデルがすぐに使え、Python / iOS / Android すべてに対応しています。その後、カスタムモデルが必要になったら Core ML (iOS) または TFLite (Android) で独自モデルをデプロイする流れが効率的です。

### Q5: ProRAW と通常のRAWの違いは何ですか？

**A:** 通常のRAW（DNG）はセンサーの生データをそのまま保存するため、AI処理（マルチフレーム合成、Deep Fusion、ノイズ除去）が適用されません。一方、Apple ProRAW はAI処理済みのデータをRAW形式で保存するため、計算フォトグラフィの恩恵を受けながらも後処理の自由度を確保できます。ファイルサイズは通常のJPEGの10〜20倍（25-75MB程度）になります。

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
| ISP + NPU連携 | ハードウェアISPとAIの協調処理がリアルタイム性能の鍵 |
| パフォーマンス最適化 | プレビュー/キャプチャの品質分離、NPU優先活用 |

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
5. **MediaPipe** — "On-device Machine Learning Solutions," mediapipe.dev, 2024
6. **Apple** — "Core ML and Vision Framework," developer.apple.com, 2024

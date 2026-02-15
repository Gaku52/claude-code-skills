# バーチャル試着

> AIとコンピュータビジョン技術を活用したバーチャル試着システムの仕組みを、人体推定・衣服変形・リアルタイムレンダリングの観点から解説し、ECサイトやアパレル業界での実装手法を示す

## この章で学ぶこと

1. **バーチャル試着の技術基盤** -- 人体ポーズ推定、セマンティックセグメンテーション、衣服変形アルゴリズム
2. **主要アプローチの比較** -- 2D画像ベース vs 3Dモデルベース vs AR ベースの手法と精度
3. **実装パイプラインと課題** -- データ準備、モデル訓練、リアルタイム推論の実現

---

## 1. バーチャル試着の全体像

### 1.1 システムアーキテクチャ

```
バーチャル試着パイプライン

  入力                    処理                      出力
  +----------+           +-------------------+     +----------+
  | ユーザー  |           | 1. 人体推定        |     | 試着結果  |
  | 写真/動画 | -------> | 2. セグメンテーション| --> | 画像/動画 |
  +----------+           | 3. 衣服変形         |     +----------+
  +----------+           | 4. 合成・レンダリング|
  | 衣服画像  | -------> |                     |
  | (カタログ) |          +-------------------+
  +----------+
```

### 1.2 技術スタック

```
レイヤー構成

  フロントエンド
  ├── WebGL / Three.js    --- 3D レンダリング
  ├── MediaPipe           --- リアルタイム人体推定
  └── WebRTC              --- カメラ入力

  AI モデル
  ├── DWPose / OpenPose   --- 人体ポーズ推定
  ├── SAM (Meta)          --- セグメンテーション
  ├── HR-VITON            --- 画像ベース試着
  ├── CatVTON             --- カテゴリ対応試着
  └── StableVITON         --- Diffusion ベース試着

  バックエンド
  ├── ONNX Runtime        --- モデル推論
  ├── TensorRT            --- GPU 最適化推論
  └── Triton Server       --- 推論サーバー
```

### 1.3 3つのアプローチ

```
【2D 画像ベース（Image-based VTON）】
  ユーザー写真 + 衣服画像 → AI が合成
  精度: 中〜高
  速度: 中（1-3秒）
  用途: EC サイト、カタログ

【3D モデルベース】
  3Dボディスキャン + 3D衣服モデル → 物理シミュレーション
  精度: 最高
  速度: 低（数秒〜数十秒）
  用途: 高級アパレル、オーダーメイド

【AR リアルタイム】
  カメラ映像 + AR オーバーレイ → リアルタイム合成
  精度: 低〜中
  速度: リアルタイム（30fps）
  用途: 店舗ミラー、モバイルアプリ
```

### 1.4 技術進化のタイムライン

```
2018  VITON (Han et al.)
  │     └─ 画像ベース試着の原点、Thin Plate Spline 変形
  │
2019  CP-VTON (Wang et al.)
  │     └─ Geometric Matching Module の導入
  │
2020  ACGPN (Yang et al.)
  │     └─ セマンティックセグメンテーションの精緻化
  │
2021  PF-AFN (Ge et al.)
  │     └─ Parser-Free アプローチ（パーシング不要化）
  │
2022  HR-VITON (Lee et al.)
  │     └─ 高解像度対応、条件付き正規化フロー
  │
2023  StableVITON (Kim et al.)
  │     └─ Stable Diffusion ベース、高品質合成
  │   CatVTON (Zheng et al.)
  │     └─ カテゴリ認識型試着
  │
2024  OOTDiffusion (Xu et al.)
  │     └─ Outfitting Fusion、全身対応
  │   IDM-VTON
  │     └─ Identity-preserving 試着
  │
2025  マルチモーダル統合
        └─ テキスト指示による衣服変更、3D統合
```

---

## 2. 2D 画像ベース試着の実装

### 2.1 HR-VITON パイプライン

```python
# HR-VITON による試着画像生成 (擬似コード)
import torch
from hr_viton import HRVITONModel
from utils import load_image, preprocess

# モデルロード
model = HRVITONModel.from_pretrained("hr-viton-checkpoint")
model.to("cuda")

# 入力準備
person_image = load_image("person.jpg")           # ユーザー写真
garment_image = load_image("tshirt_catalog.jpg")   # 衣服カタログ画像

# 前処理: 人体パーシング + ポーズ推定
person_parse = segment_person(person_image)         # 体のパーツ分割
person_pose = estimate_pose(person_image)           # 関節位置推定
garment_mask = segment_garment(garment_image)       # 衣服領域抽出

# 試着画像生成
result = model.inference(
    person_image=person_image,
    garment_image=garment_image,
    person_parse=person_parse,
    person_pose=person_pose,
    garment_mask=garment_mask,
)

result.save("try_on_result.jpg")
```

### 2.2 DensePose による人体表面推定

```python
# DensePose: 人体表面の UV マッピング
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from densepose.config import add_densepose_config

cfg = get_cfg()
add_densepose_config(cfg)
cfg.merge_from_file(model_zoo.get_config_file(
    "densepose_rcnn_R_101_FPN_DL_s1x.yaml"
))
cfg.MODEL.WEIGHTS = "densepose_model.pkl"

predictor = DefaultPredictor(cfg)
outputs = predictor(person_image)

# DensePose は人体の表面を UV 座標にマッピング
# → 衣服テクスチャを人体表面に正確に貼り付け可能
```

### 2.3 セグメンテーション

```python
# 人体パーシング (LIP / ATR フォーマット)
# 各ピクセルを体のパーツに分類

# パーツラベル:
#  0: 背景, 1: 帽子, 2: 髪, 3: サングラス
#  4: 上着, 5: スカート, 6: パンツ, 7: ドレス
#  8: ベルト, 9: 左靴, 10: 右靴, 11: 顔
#  12: 左足, 13: 右足, 14: 左腕, 15: 右腕
#  16: バッグ, 17: スカーフ

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

processor = SegformerImageProcessor.from_pretrained(
    "mattmdjaga/segformer_b2_clothes"
)
model = SegformerForSemanticSegmentation.from_pretrained(
    "mattmdjaga/segformer_b2_clothes"
)

inputs = processor(images=person_image, return_tensors="pt")
outputs = model(**inputs)

# 各ピクセルのパーツラベルを取得
parse_map = outputs.logits.argmax(dim=1).squeeze()
```

### 2.4 OOTDiffusion による高品質試着

```python
# OOTDiffusion: Outfitting Fusion ベースの試着モデル
# Stable Diffusion をベースとした最新の VTON 手法

import torch
from ootd.inference import OOTDInference
from PIL import Image

class OOTDiffusionPipeline:
    """OOTDiffusion による高品質バーチャル試着"""

    def __init__(self, model_path: str = "levihsu/OOTDiffusion"):
        self.model = OOTDInference(
            model_path=model_path,
            model_type="hd",  # "hd" (半身) or "dc" (全身)
        )

    def try_on(
        self,
        person_image_path: str,
        garment_image_path: str,
        category: str = "upperbody",
        num_samples: int = 1,
        num_steps: int = 20,
        guidance_scale: float = 2.0,
        seed: int = 42,
    ) -> list:
        """
        バーチャル試着を実行

        category:
          - "upperbody": 上半身（Tシャツ、シャツ、ジャケット等）
          - "lowerbody": 下半身（パンツ、スカート等）
          - "dress": ワンピース、ドレス

        guidance_scale:
          - 1.0-2.0: 自然な仕上がり
          - 2.0-3.0: 衣服のディテール重視
          - 3.0+: 過度に強調（アーティファクト注意）
        """
        person_img = Image.open(person_image_path).resize((768, 1024))
        garment_img = Image.open(garment_image_path).resize((768, 1024))

        results = self.model(
            category=category,
            image_garm=garment_img,
            image_vton=person_img,
            n_samples=num_samples,
            n_steps=num_steps,
            image_scale=guidance_scale,
            seed=seed,
        )

        return results

    def batch_try_on(
        self,
        person_image_path: str,
        garment_dir: str,
        output_dir: str,
        category: str = "upperbody",
    ) -> dict:
        """複数の衣服を一括で試着"""
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        results = {}
        for garment_path in Path(garment_dir).glob("*.{jpg,png,jpeg}"):
            try:
                output = self.try_on(
                    person_image_path=person_image_path,
                    garment_image_path=str(garment_path),
                    category=category,
                )
                out_path = f"{output_dir}/{garment_path.stem}_tryon.png"
                output[0].save(out_path)
                results[garment_path.stem] = out_path
            except Exception as e:
                results[garment_path.stem] = f"Error: {e}"

        return results


# 使用例
pipeline = OOTDiffusionPipeline()

# 単一衣服の試着
result = pipeline.try_on(
    person_image_path="user_photo.jpg",
    garment_image_path="blue_tshirt.jpg",
    category="upperbody",
    num_samples=3,  # 3つのバリエーションを生成
    guidance_scale=2.0,
)

# 結果の保存
for i, img in enumerate(result):
    img.save(f"tryon_result_{i}.png")
```

### 2.5 衣服変形アルゴリズムの詳細

```python
# Thin Plate Spline (TPS) 変形
# 衣服画像をユーザーの体型に合わせて変形する手法

import numpy as np
import cv2

class ThinPlateSplineWarper:
    """TPS (薄板スプライン) による衣服変形"""

    def __init__(self, source_points: np.ndarray, target_points: np.ndarray):
        """
        source_points: 衣服画像上の制御点 (N, 2)
        target_points: ユーザー画像上の対応点 (N, 2)
        """
        self.source = source_points
        self.target = target_points
        self.n = len(source_points)

        # TPS パラメータの計算
        self._compute_parameters()

    def _compute_parameters(self):
        """TPS 変形パラメータを計算"""
        n = self.n
        K = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    r = np.linalg.norm(self.source[i] - self.source[j])
                    K[i, j] = r ** 2 * np.log(r + 1e-6)

        # 線形部分の行列
        P = np.hstack([np.ones((n, 1)), self.source])

        # 連立方程式の構築
        L = np.zeros((n + 3, n + 3))
        L[:n, :n] = K
        L[:n, n:] = P
        L[n:, :n] = P.T

        # 各軸について解く
        self.params_x = np.linalg.solve(
            L + np.eye(n + 3) * 1e-6,
            np.concatenate([self.target[:, 0], [0, 0, 0]])
        )
        self.params_y = np.linalg.solve(
            L + np.eye(n + 3) * 1e-6,
            np.concatenate([self.target[:, 1], [0, 0, 0]])
        )

    def warp_image(self, image: np.ndarray) -> np.ndarray:
        """画像を TPS 変形する"""
        h, w = image.shape[:2]
        output = np.zeros_like(image)

        for y in range(h):
            for x in range(w):
                point = np.array([x, y], dtype=float)

                # TPS による座標変換
                new_x = self._transform_point(point, self.params_x)
                new_y = self._transform_point(point, self.params_y)

                # バイリニア補間でピクセル値を取得
                if 0 <= new_x < w and 0 <= new_y < h:
                    output[y, x] = self._bilinear_interpolate(
                        image, new_x, new_y
                    )

        return output

    def _transform_point(self, point, params):
        """1点の座標変換"""
        result = params[self.n] + params[self.n + 1] * point[0] + params[self.n + 2] * point[1]
        for i in range(self.n):
            r = np.linalg.norm(point - self.source[i])
            if r > 0:
                result += params[i] * r ** 2 * np.log(r)
        return result

    def _bilinear_interpolate(self, image, x, y):
        """バイリニア補間"""
        x0 = int(np.floor(x))
        x1 = min(x0 + 1, image.shape[1] - 1)
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, image.shape[0] - 1)

        dx = x - x0
        dy = y - y0

        return (
            image[y0, x0] * (1 - dx) * (1 - dy)
            + image[y0, x1] * dx * (1 - dy)
            + image[y1, x0] * (1 - dx) * dy
            + image[y1, x1] * dx * dy
        ).astype(np.uint8)
```

---

## 3. AR リアルタイム試着

```python
# MediaPipe + Three.js でリアルタイム AR 試着 (概念コード)
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ポーズ推定
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # 肩、腰、腕の座標を取得
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # 衣服画像をポーズに合わせて変形・合成
        garment_overlay = warp_garment_to_pose(
            garment_image, landmarks, frame.shape
        )
        frame = overlay_transparent(frame, garment_overlay)

    cv2.imshow('Virtual Try-On', frame)
```

### 3.1 WebAR 実装パターン

```javascript
// WebAR を使ったブラウザベース試着
// Three.js + MediaPipe Holistic

class WebARTryOn {
  constructor(videoElement, canvasElement) {
    this.video = videoElement;
    this.canvas = canvasElement;

    // Three.js のセットアップ
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(75, 16 / 9, 0.1, 1000);
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      alpha: true,
    });

    // MediaPipe Pose の初期化
    this.pose = new Pose({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
      },
    });

    this.pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    this.pose.onResults(this.onPoseResults.bind(this));

    // 衣服メッシュ
    this.garmentMesh = null;
  }

  async loadGarment(glbPath) {
    const loader = new THREE.GLTFLoader();
    const gltf = await loader.loadAsync(glbPath);
    this.garmentMesh = gltf.scene;
    this.scene.add(this.garmentMesh);
  }

  onPoseResults(results) {
    if (!results.poseLandmarks || !this.garmentMesh) return;

    const landmarks = results.poseLandmarks;

    // 肩の座標からスケールと位置を計算
    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];
    const leftHip = landmarks[23];
    const rightHip = landmarks[24];

    // 肩幅から衣服のスケーリング
    const shoulderWidth = Math.sqrt(
      Math.pow(rightShoulder.x - leftShoulder.x, 2) +
      Math.pow(rightShoulder.y - leftShoulder.y, 2)
    );

    // 体の中心座標
    const centerX = (leftShoulder.x + rightShoulder.x) / 2;
    const centerY = (leftShoulder.y + leftHip.y) / 2;

    // 体の傾きを計算
    const angle = Math.atan2(
      rightShoulder.y - leftShoulder.y,
      rightShoulder.x - leftShoulder.x
    );

    // メッシュの位置・回転・スケールを更新
    this.garmentMesh.position.set(
      (centerX - 0.5) * 4,
      -(centerY - 0.5) * 4,
      0
    );
    this.garmentMesh.rotation.z = angle;
    this.garmentMesh.scale.setScalar(shoulderWidth * 5);

    this.renderer.render(this.scene, this.camera);
  }

  async start() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    this.video.srcObject = stream;

    const cameraLoop = new Camera(this.video, {
      onFrame: async () => {
        await this.pose.send({ image: this.video });
      },
      width: 1280,
      height: 720,
    });
    cameraLoop.start();
  }
}

// 使用例
const tryOn = new WebARTryOn(
  document.getElementById('video'),
  document.getElementById('canvas')
);
await tryOn.loadGarment('/models/tshirt.glb');
await tryOn.start();
```

### 3.2 サイズ推薦システム

```python
# AI ベースのサイズ推薦システム

import numpy as np
from dataclasses import dataclass

@dataclass
class BodyMeasurements:
    """体型計測データ"""
    height_cm: float
    weight_kg: float
    chest_cm: float
    waist_cm: float
    hip_cm: float
    shoulder_width_cm: float
    arm_length_cm: float
    inseam_cm: float

class SizeRecommender:
    """体型データに基づくサイズ推薦"""

    # ブランドごとのサイズチャート（例）
    SIZE_CHARTS = {
        "standard_jp": {
            "S": {"chest": (80, 88), "waist": (68, 76), "hip": (82, 90)},
            "M": {"chest": (88, 96), "waist": (76, 84), "hip": (90, 98)},
            "L": {"chest": (96, 104), "waist": (84, 92), "hip": (98, 106)},
            "XL": {"chest": (104, 112), "waist": (92, 100), "hip": (106, 114)},
        },
    }

    def __init__(self, brand: str = "standard_jp"):
        self.size_chart = self.SIZE_CHARTS.get(brand, self.SIZE_CHARTS["standard_jp"])

    def estimate_body_from_image(self, image_path: str, height_cm: float) -> BodyMeasurements:
        """
        写真と身長から体型を推定

        実際の実装では:
        1. DensePose で体表面の UV マッピング
        2. SMPL / SMPL-X で 3D 体型パラメータ推定
        3. パラメータから各部位の寸法を計算
        """
        # SMPL モデルによる体型推定（擬似コード）
        from smpl_estimation import estimate_smpl_params

        smpl_params = estimate_smpl_params(image_path, height_cm)

        measurements = BodyMeasurements(
            height_cm=height_cm,
            weight_kg=smpl_params.estimated_weight,
            chest_cm=smpl_params.chest_circumference,
            waist_cm=smpl_params.waist_circumference,
            hip_cm=smpl_params.hip_circumference,
            shoulder_width_cm=smpl_params.shoulder_width,
            arm_length_cm=smpl_params.arm_length,
            inseam_cm=smpl_params.inseam,
        )
        return measurements

    def recommend_size(
        self,
        measurements: BodyMeasurements,
        garment_type: str = "top",
        fit_preference: str = "regular",
    ) -> dict:
        """
        サイズを推薦

        fit_preference: "slim", "regular", "loose"
        """
        # フィット調整（cm）
        fit_adjustment = {
            "slim": -2,
            "regular": 0,
            "loose": 4,
        }
        adj = fit_adjustment.get(fit_preference, 0)

        # 各サイズとの適合度を計算
        scores = {}
        for size, ranges in self.size_chart.items():
            score = 0
            if garment_type in ("top", "outerwear"):
                chest_mid = (ranges["chest"][0] + ranges["chest"][1]) / 2
                score += 1.0 - abs(measurements.chest_cm + adj - chest_mid) / 20
            if garment_type in ("bottom",):
                waist_mid = (ranges["waist"][0] + ranges["waist"][1]) / 2
                score += 1.0 - abs(measurements.waist_cm + adj - waist_mid) / 20
                hip_mid = (ranges["hip"][0] + ranges["hip"][1]) / 2
                score += 1.0 - abs(measurements.hip_cm + adj - hip_mid) / 20

            scores[size] = max(0, score)

        # 最適サイズ
        best_size = max(scores, key=scores.get)
        confidence = scores[best_size] / max(sum(scores.values()), 1e-6)

        return {
            "recommended_size": best_size,
            "confidence": round(confidence, 2),
            "scores": scores,
            "fit_preference": fit_preference,
            "note": f"バスト{measurements.chest_cm}cm に基づく推薦"
                    if garment_type == "top"
                    else f"ウエスト{measurements.waist_cm}cm に基づく推薦",
        }


# 使用例
recommender = SizeRecommender(brand="standard_jp")

measurements = BodyMeasurements(
    height_cm=170,
    weight_kg=65,
    chest_cm=92,
    waist_cm=78,
    hip_cm=94,
    shoulder_width_cm=44,
    arm_length_cm=58,
    inseam_cm=76,
)

result = recommender.recommend_size(
    measurements=measurements,
    garment_type="top",
    fit_preference="regular",
)
print(f"推薦サイズ: {result['recommended_size']}")
print(f"信頼度: {result['confidence']}")
```

---

## 4. EC サイト向け実装ガイド

### 4.1 バックエンド API 設計

```python
# FastAPI を使ったバーチャル試着 API

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid

app = FastAPI(title="Virtual Try-On API")

class TryOnRequest(BaseModel):
    person_image_id: str
    garment_id: str
    category: str = "upperbody"  # upperbody, lowerbody, dress
    guidance_scale: float = 2.0
    num_samples: int = 1

class TryOnResponse(BaseModel):
    request_id: str
    status: str
    result_urls: list[str] = []
    processing_time_ms: float = 0
    size_recommendation: Optional[dict] = None

class VTONService:
    """バーチャル試着サービス"""

    def __init__(self):
        self.model = None  # 遅延ロード
        self._load_model()

    def _load_model(self):
        """モデルの遅延ロード"""
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # ここで VTON モデルをロード
        print(f"Model loaded on {self.device}")

    async def process_try_on(self, request: TryOnRequest) -> TryOnResponse:
        """試着リクエストを処理"""
        import time
        start = time.time()

        request_id = str(uuid.uuid4())

        try:
            # 画像取得
            person_img = await self._fetch_image(request.person_image_id)
            garment_img = await self._fetch_garment(request.garment_id)

            # 試着実行
            results = self._run_inference(
                person_img, garment_img,
                category=request.category,
                guidance_scale=request.guidance_scale,
                num_samples=request.num_samples,
            )

            # 結果保存
            result_urls = []
            for i, result in enumerate(results):
                url = await self._save_result(request_id, i, result)
                result_urls.append(url)

            elapsed = (time.time() - start) * 1000

            return TryOnResponse(
                request_id=request_id,
                status="completed",
                result_urls=result_urls,
                processing_time_ms=round(elapsed, 1),
            )

        except Exception as e:
            return TryOnResponse(
                request_id=request_id,
                status=f"error: {str(e)}",
            )

    def _run_inference(self, person_img, garment_img, **kwargs):
        """推論実行"""
        # 実際の推論ロジック
        pass

    async def _fetch_image(self, image_id):
        """S3 等から画像を取得"""
        pass

    async def _fetch_garment(self, garment_id):
        """カタログ DB から衣服画像を取得"""
        pass

    async def _save_result(self, request_id, index, image):
        """結果を S3 に保存"""
        pass

vton_service = VTONService()

@app.post("/api/v1/try-on", response_model=TryOnResponse)
async def try_on(request: TryOnRequest):
    """バーチャル試着 API エンドポイント"""
    return await vton_service.process_try_on(request)

@app.post("/api/v1/upload-photo")
async def upload_photo(file: UploadFile = File(...)):
    """ユーザー写真のアップロード"""
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, "JPEG または PNG のみ対応")

    image_id = str(uuid.uuid4())
    # S3 に保存
    return {"image_id": image_id, "status": "uploaded"}
```

### 4.2 フロントエンド統合パターン

```typescript
// React コンポーネントでの統合例

interface TryOnResult {
  requestId: string;
  status: string;
  resultUrls: string[];
  processingTimeMs: number;
}

async function virtualTryOn(
  personImageId: string,
  garmentId: string,
  category: 'upperbody' | 'lowerbody' | 'dress' = 'upperbody'
): Promise<TryOnResult> {
  const response = await fetch('/api/v1/try-on', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      person_image_id: personImageId,
      garment_id: garmentId,
      category: category,
      guidance_scale: 2.0,
      num_samples: 1,
    }),
  });

  if (!response.ok) {
    throw new Error(`Try-on failed: ${response.statusText}`);
  }

  return response.json();
}

// UX パターン: プログレッシブ表示
// 1. ローディングスケルトン表示
// 2. 低解像度のプレビューを先に表示（512px）
// 3. 高解像度の結果に差し替え（1024px+）
```

---

## 5. パフォーマンス最適化

### 5.1 推論高速化テクニック

```python
# TensorRT による推論高速化

import tensorrt as trt
import numpy as np

class TRTOptimizedVTON:
    """TensorRT で最適化されたバーチャル試着モデル"""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    @staticmethod
    def convert_to_trt(
        onnx_path: str,
        output_path: str,
        fp16: bool = True,
        max_batch: int = 4,
    ):
        """ONNX モデルを TensorRT エンジンに変換"""
        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))

        with open(onnx_path, "rb") as f:
            parser.parse(f.read())

        config = builder.create_builder_config()
        config.max_workspace_size = 4 << 30  # 4GB

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # ダイナミックバッチサイズ
        profile = builder.create_optimization_profile()
        input_shape = network.get_input(0).shape
        profile.set_shape(
            network.get_input(0).name,
            min=(1, *input_shape[1:]),
            opt=(2, *input_shape[1:]),
            max=(max_batch, *input_shape[1:]),
        )
        config.add_optimization_profile(profile)

        engine = builder.build_engine(network, config)
        with open(output_path, "wb") as f:
            f.write(engine.serialize())

        print(f"TRT engine saved: {output_path}")


# 高速化の効果比較
# ┌──────────────┬──────────┬──────────┬──────────┐
# │ 手法          │ 推論時間  │ VRAM     │ 品質     │
# ├──────────────┼──────────┼──────────┼──────────┤
# │ PyTorch FP32 │ 3.5秒    │ 8GB      │ 最高     │
# │ PyTorch FP16 │ 1.8秒    │ 4GB      │ ほぼ同等  │
# │ ONNX Runtime │ 1.2秒    │ 4GB      │ ほぼ同等  │
# │ TensorRT FP16│ 0.6秒    │ 3GB      │ ほぼ同等  │
# │ TensorRT INT8│ 0.3秒    │ 2GB      │ やや低下  │
# └──────────────┴──────────┴──────────┴──────────┘
```

### 5.2 バッチ処理とキャッシング

```python
# Redis を使った試着結果のキャッシング

import hashlib
import redis
import json

class TryOnCache:
    """試着結果のキャッシュ管理"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.ttl = 3600 * 24  # 24時間

    def _generate_key(self, person_hash: str, garment_id: str, params: dict) -> str:
        """キャッシュキーの生成"""
        param_str = json.dumps(params, sort_keys=True)
        raw = f"{person_hash}:{garment_id}:{param_str}"
        return f"vton:{hashlib.sha256(raw.encode()).hexdigest()}"

    def get_cached_result(self, person_hash: str, garment_id: str, params: dict):
        """キャッシュから結果を取得"""
        key = self._generate_key(person_hash, garment_id, params)
        result = self.redis.get(key)
        if result:
            return json.loads(result)
        return None

    def cache_result(self, person_hash: str, garment_id: str, params: dict, result_urls: list):
        """結果をキャッシュに保存"""
        key = self._generate_key(person_hash, garment_id, params)
        self.redis.setex(key, self.ttl, json.dumps(result_urls))

    def get_cache_stats(self) -> dict:
        """キャッシュ統計"""
        info = self.redis.info("stats")
        return {
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
            "hit_rate": info.get("keyspace_hits", 0) /
                        max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1),
        }
```

---

## 6. 比較表

| 手法 | 精度 | 速度 | コスト | 要件 |
|------|:----:|:----:|:-----:|------|
| 2D 画像ベース (HR-VITON) | 高 | 1-3秒 | 中 | GPU サーバー |
| Diffusion ベース (StableVITON) | 最高 | 5-15秒 | 高 | 高性能 GPU |
| OOTDiffusion | 最高 | 3-8秒 | 高 | 高性能 GPU |
| 3D モデルベース | 最高 | 10-30秒 | 最高 | 3D スキャンデータ |
| AR リアルタイム (MediaPipe) | 中 | リアルタイム | 低 | カメラのみ |
| 簡易オーバーレイ | 低 | リアルタイム | 最低 | なし |

| ユースケース | 推奨手法 | 理由 |
|------------|---------|------|
| EC 商品ページ | 2D 画像ベース | バッチ処理可能、高品質 |
| 店舗の AR ミラー | AR リアルタイム | 即時フィードバック |
| 高級ブランド | 3D モデルベース | 最高品質、サイズ感再現 |
| モバイルアプリ | AR リアルタイム | GPU 軽量、UX 良好 |
| ルックブック自動生成 | OOTDiffusion | 高品質、全身対応 |

### VTON モデル世代別比較

| 世代 | 代表モデル | 特徴 | 制限 |
|------|----------|------|------|
| 第1世代 (2018) | VITON, CP-VTON | TPS 変形 + 合成 | 変形の不自然さ |
| 第2世代 (2020) | ACGPN, PF-AFN | セグメンテーション改善 | 複雑な衣服に弱い |
| 第3世代 (2022) | HR-VITON | 高解像度、正規化フロー | 特定ポーズに限定 |
| 第4世代 (2023-) | StableVITON, OOTDiffusion | Diffusion ベース | 処理速度が遅い |

---

## 7. トラブルシューティング

### 7.1 よくある問題と対処法

```
問題1: 衣服のテクスチャが潰れる
──────────────────────────────
原因: 入力画像の解像度が低い、または衣服画像に影や折り目が多い
対処:
  1. 衣服画像は白背景・フラットな状態で撮影（推奨: 1024x1024以上）
  2. 影除去の前処理を適用
  3. guidance_scale を下げる（1.5-2.0）

問題2: 体型が不自然に変形する
──────────────────────────────
原因: 人体パーシングの精度が低い、ポーズ推定の失敗
対処:
  1. 正面・全身が写った写真を使用
  2. 背景がシンプルな写真を選択
  3. 複数の推定結果をアンサンブル

問題3: 衣服と肌の境界が不自然
──────────────────────────────
原因: セグメンテーションの精度不足、ブレンド処理の欠如
対処:
  1. マスクにフェザリング（ぼかし）を適用
  2. 境界部分の色調を統一する後処理
  3. DensePose の UV マップを活用した精密な合成

問題4: AR モードでちらつき（フリッカー）
──────────────────────────────
原因: ポーズ推定のフレーム間ジッター
対処:
  1. One Euro Filter でランドマークを平滑化
  2. min_tracking_confidence を上げる（0.7-0.8）
  3. 指数移動平均でポジションをスムージング

問題5: 処理速度が遅い（5秒以上）
──────────────────────────────
原因: モデルの最適化不足、大きすぎる解像度
対処:
  1. TensorRT / ONNX Runtime で推論を最適化
  2. FP16 精度に変更
  3. 入力解像度を 768x1024 に統一
  4. バッチ処理でスループットを向上
```

### 7.2 衣服画像の品質チェックリスト

```
撮影条件チェックリスト:
  □ 白または単色の背景
  □ 均一な照明（影なし）
  □ 正面から撮影（平置きまたはマネキン）
  □ 衣服全体が収まっている
  □ 最低 1024x1024 ピクセル
  □ JPEG 品質 90% 以上
  □ 色の正確性（ホワイトバランス調整済み）
  □ シワや折り目が最小限

前処理パイプライン:
  1. 背景除去 → 白背景に統一
  2. ホワイトバランス補正
  3. 影除去（Intrinsic Image Decomposition）
  4. 解像度統一（1024x1024 にリサイズ）
  5. 衣服マスクの生成（SAM or U2-Net）
```

---

## 8. アンチパターン

### アンチパターン 1: 体型の多様性を無視する

```
BAD:
  標準体型のモデルでのみ学習
  → 多様な体型のユーザーで試着結果が不自然
  → 衣服がはみ出す、伸びる、変形する

GOOD:
  - 多様な体型のデータで学習
  - 体型パラメータ（身長、体重、バスト/ウエスト/ヒップ）を入力として使用
  - ユーザーの体型に応じたサイズ推薦機能を併設
```

### アンチパターン 2: 照明・色味の不一致

```
BAD:
  スタジオ照明の衣服画像 + 屋外自然光のユーザー写真
  → 明らかな合成感、色味の不一致
  → ユーザーの購買判断に悪影響

GOOD:
  - 照明推定（Light Estimation）で環境光を分析
  - 衣服の色温度・明度をユーザー写真に合わせて調整
  - 影の方向を統一（シャドウハーモナイゼーション）
```

### アンチパターン 3: 単一アングルのみの対応

```
BAD:
  正面写真のみ対応し、横向き・後ろ姿は完全に非対応
  → ユーザーが衣服の全体像を把握できない
  → 「背中のデザインが見えない」というクレーム

GOOD:
  - マルチビュー対応: 正面、側面、背面の試着画像を生成
  - 3D ボディモデル推定 → 複数視点からのレンダリング
  - ユーザーに対して「正面写真」の撮影をガイドする UI
  - 将来的には動画入力から 3D 再構成
```

### アンチパターン 4: エラーハンドリングの欠如

```
BAD:
  ポーズ推定の失敗時に何も表示せず、変形した試着結果を返す
  → ユーザーが不自然な画像を見て離脱

GOOD:
  - ポーズ推定の信頼度スコアをチェック
  - 信頼度が低い場合は「撮り直し」を促すメッセージ
  - フォールバック: 標準モデルでの試着を代替表示
  - 各処理ステージの品質ゲートを設定
```

---

## 9. FAQ

### Q1. バーチャル試着の精度はどの程度実用的か？

**A.** 2D 画像ベースの最新手法（StableVITON 等）は、正面写真での上半身衣服の試着において実用レベルに達している。ただし、(1) 複雑な柄やテクスチャの再現、(2) 横向き・後ろ向きのポーズ、(3) レイヤード（重ね着）のシナリオではまだ課題がある。EC サイトでの「参考イメージ」としては十分な品質。

### Q2. バーチャル試着は返品率の削減に効果があるか？

**A.** 複数の調査で返品率の25-35%削減が報告されている。特にサイズ感の不一致による返品が大幅に減少する。AR 試着を導入したアパレルECでは、試着機能を使ったユーザーの購入率が2-3倍高いというデータもある。ただし、色の正確な再現がディスプレイ依存であるため、色味理由の返品には効果が限定的。

### Q3. 自社で実装する場合の最低要件は？

**A.** (1) **GPU サーバー**: NVIDIA A10G 以上（AWS: g5.xlarge 相当）。(2) **データ**: 衣服のカタログ画像（白背景、正面）。(3) **モデル**: HR-VITON や CatVTON のオープンソースモデルを出発点にする。(4) **推論時間**: バッチ処理なら2-5秒/枚が現実的。リアルタイム AR が必要なら MediaPipe + 簡易合成から始める。最小構成で PoC を作り、効果を検証してから本格投資する。

### Q4. バーチャル試着のプライバシー対策は？

**A.** ユーザーの体型写真は極めて機密性の高い個人情報である。(1) **オンデバイス処理**: 可能な限りユーザーのデバイス上で推論する（WebGPU / Core ML）。(2) **サーバー送信時**: TLS 暗号化必須、処理完了後に即時削除。(3) **保存ポリシー**: ユーザー写真は原則保存しない。必要な場合は明示的同意を取得。(4) **アクセス制御**: 試着結果へのアクセスはユーザー本人のみに限定。(5) **GDPR / 個人情報保護法**: データ処理の法的根拠を明確にし、プライバシーポリシーに明記。

### Q5. アクセサリー（帽子、メガネ、靴）の試着は可能か？

**A.** アクセサリーの試着はカテゴリごとに成熟度が異なる。**メガネ**: AR ベースが非常に成熟しており、Warby Parker 等で実用化済み。顔のランドマーク検出が正確なため高精度。**帽子**: 頭部のサイズ推定が課題だが、AR で一定の品質を実現可能。**靴**: 足のサイズ推定 + AR が進展中（Nike Fit 等）。**アクセサリー全般**: 3D モデルベースの AR が最も適している。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 技術基盤 | 人体ポーズ推定 + セグメンテーション + 衣服変形 + 合成 |
| 2D 画像ベース | EC サイト向け。HR-VITON、StableVITON が主要手法 |
| AR リアルタイム | 店舗・モバイル向け。MediaPipe で軽量実装可能 |
| 3D モデルベース | 最高精度だがコスト高。高級ブランド向け |
| ビジネス効果 | 返品率25-35%削減、購入率2-3倍向上の報告 |
| 課題 | 体型多様性、照明一致、複雑な衣服の再現 |
| 最適化 | TensorRT で 6 倍高速化、キャッシング必須 |
| プライバシー | ユーザー写真は即時削除、オンデバイス処理推奨 |

---

## 次に読むべきガイド

- [倫理的考慮](./03-ethical-considerations.md) -- AI 生成画像の倫理と著作権
- [アニメーション](../02-video/02-animation.md) -- AI アニメーション技術
- [デザインツール](../01-image/03-design-tools.md) -- 商品画像のAI編集

---

## 参考文献

1. **HR-VITON** -- Lee et al. (ECCV 2022) -- 高解像度バーチャル試着
2. **StableVITON** -- Kim et al. (2024) -- Diffusion ベースの試着モデル
3. **DensePose** -- Guler et al. (CVPR 2018) -- 人体表面の密なマッピング
4. **OOTDiffusion** -- Xu et al. (2024) -- Outfitting Fusion ベースの VTON
5. **SMPL** -- Loper et al. (SIGGRAPH Asia 2015) -- パラメトリック人体モデル
6. **MediaPipe Pose** -- Google (2020) -- リアルタイム人体ポーズ推定

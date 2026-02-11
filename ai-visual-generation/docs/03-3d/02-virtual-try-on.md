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

---

## 4. 比較表

| 手法 | 精度 | 速度 | コスト | 要件 |
|------|:----:|:----:|:-----:|------|
| 2D 画像ベース (HR-VITON) | 高 | 1-3秒 | 中 | GPU サーバー |
| Diffusion ベース (StableVITON) | 最高 | 5-15秒 | 高 | 高性能 GPU |
| 3D モデルベース | 最高 | 10-30秒 | 最高 | 3D スキャンデータ |
| AR リアルタイム (MediaPipe) | 中 | リアルタイム | 低 | カメラのみ |
| 簡易オーバーレイ | 低 | リアルタイム | 最低 | なし |

| ユースケース | 推奨手法 | 理由 |
|------------|---------|------|
| EC 商品ページ | 2D 画像ベース | バッチ処理可能、高品質 |
| 店舗の AR ミラー | AR リアルタイム | 即時フィードバック |
| 高級ブランド | 3D モデルベース | 最高品質、サイズ感再現 |
| モバイルアプリ | AR リアルタイム | GPU 軽量、UX 良好 |

---

## 5. アンチパターン

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

---

## 6. FAQ

### Q1. バーチャル試着の精度はどの程度実用的か？

**A.** 2D 画像ベースの最新手法（StableVITON 等）は、正面写真での上半身衣服の試着において実用レベルに達している。ただし、(1) 複雑な柄やテクスチャの再現、(2) 横向き・後ろ向きのポーズ、(3) レイヤード（重ね着）のシナリオではまだ課題がある。EC サイトでの「参考イメージ」としては十分な品質。

### Q2. バーチャル試着は返品率の削減に効果があるか？

**A.** 複数の調査で返品率の25-35%削減が報告されている。特にサイズ感の不一致による返品が大幅に減少する。AR 試着を導入したアパレルECでは、試着機能を使ったユーザーの購入率が2-3倍高いというデータもある。ただし、色の正確な再現がディスプレイ依存であるため、色味理由の返品には効果が限定的。

### Q3. 自社で実装する場合の最低要件は？

**A.** (1) **GPU サーバー**: NVIDIA A10G 以上（AWS: g5.xlarge 相当）。(2) **データ**: 衣服のカタログ画像（白背景、正面）。(3) **モデル**: HR-VITON や CatVTON のオープンソースモデルを出発点にする。(4) **推論時間**: バッチ処理なら2-5秒/枚が現実的。リアルタイム AR が必要なら MediaPipe + 簡易合成から始める。最小構成で PoC を作り、効果を検証してから本格投資する。

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

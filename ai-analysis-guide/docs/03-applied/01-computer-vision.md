# コンピュータビジョン

> 物体検出、セグメンテーション、画像分類の主要手法と実装パターンを実践的に理解する

## この章で学ぶこと

1. **画像分類と特徴抽出** — CNN、転移学習、Vision Transformer の活用
2. **物体検出** — YOLO、DETR の仕組みとリアルタイム検出
3. **セグメンテーション** — セマンティック/インスタンスセグメンテーション、SAM

---

## 1. 画像分類の基礎

```
CNN の基本構造
===============

入力画像 [224x224x3]
    |
    v
[Conv 3x3, 64] --> [ReLU] --> [MaxPool 2x2]  --> 特徴マップ [112x112x64]
    |
    v
[Conv 3x3, 128] --> [ReLU] --> [MaxPool 2x2] --> 特徴マップ [56x56x128]
    |
    v
[Conv 3x3, 256] --> [ReLU] --> [MaxPool 2x2] --> 特徴マップ [28x28x256]
    |
    v
[Global Average Pooling] --> [256]
    |
    v
[FC 256 -> num_classes] --> [Softmax] --> 分類結果

畳み込みの役割:
  浅い層: エッジ、テクスチャを検出
  中間層: パーツ（目、車輪等）を検出
  深い層: オブジェクト全体を認識
```

### コード例 1: 転移学習による画像分類

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# 事前学習済みモデルをベースに分類器を構築
class ImageClassifier(nn.Module):
    def __init__(self, num_classes, backbone="resnet50"):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # 最終層を置き換え
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

        # バックボーンの一部を凍結
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)

# データ拡張
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = ImageClassifier(num_classes=10)
```

---

## 2. 物体検出

```
物体検出の主要アプローチ
==========================

1-Stage (高速):
  YOLO: 画像をグリッドに分割、各セルで直接予測
  SSD:  マルチスケールの特徴マップで検出

  入力 --> [CNN Backbone] --> [Detection Head] --> Boxes + Classes
  速度: 30-150+ FPS

2-Stage (高精度):
  Faster R-CNN: Region Proposal + Classification
  Cascade R-CNN: 複数段階のリファインメント

  入力 --> [CNN] --> [RPN] --> [ROI Pooling] --> [Classifier]
  速度: 5-15 FPS

Transformer ベース:
  DETR: End-to-End の物体検出
  入力 --> [CNN] --> [Transformer Encoder] --> [Decoder + FFN] --> Boxes
```

### コード例 2: YOLOv8 による物体検出

```python
from ultralytics import YOLO

# 事前学習済みモデルのロード
model = YOLO("yolov8n.pt")  # nano (最速), s, m, l, x (最高精度)

# 画像で推論
results = model("image.jpg")

# 結果の処理
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        print(f"{class_name}: {confidence:.2f} at ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")

# カスタムデータでのファインチューニング
model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="cuda",
)

# リアルタイム動画検出
results = model("video.mp4", stream=True)
for result in results:
    annotated_frame = result.plot()
    # フレームの表示/保存
```

### コード例 3: DETR による物体検出

```python
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# 後処理: 閾値以上の検出結果を取得
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.7
)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(f"{model.config.id2label[label.item()]}: {score:.3f} {box}")
```

---

## 3. セグメンテーション

```
セグメンテーションの種類
==========================

セマンティックセグメンテーション:
  各ピクセルにクラスラベルを付与
  [空][空][木][木][車][車][道][道]
  個体の区別なし

インスタンスセグメンテーション:
  各オブジェクトインスタンスを区別
  [空][空][木1][木2][車1][車2][道][道]
  個体を区別

パノプティックセグメンテーション:
  セマンティック + インスタンス
  背景(stuff) + 前景(things) を統合
```

### コード例 4: SAM（Segment Anything Model）

```python
from segment_anything import SamPredictor, sam_model_registry

# モデルのロード
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
sam.to(device="cuda")
predictor = SamPredictor(sam)

# 画像の設定
image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# ポイントプロンプトでセグメンテーション
input_point = np.array([[500, 375]])  # クリック位置
input_label = np.array([1])  # 1=前景, 0=背景

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,  # 複数のマスク候補を返す
)

# 最も信頼度の高いマスク
best_mask = masks[scores.argmax()]

# バウンディングボックスプロンプト
input_box = np.array([100, 100, 400, 400])  # [x1, y1, x2, y2]
masks, _, _ = predictor.predict(
    box=input_box,
    multimask_output=False,
)
```

### コード例 5: セマンティックセグメンテーション

```python
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch
from PIL import Image

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")

image = Image.open("street.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# ピクセルごとのクラス予測
logits = outputs.logits  # [batch, num_classes, height, width]
upsampled = torch.nn.functional.interpolate(
    logits, size=image.size[::-1], mode="bilinear", align_corners=False
)
predicted = upsampled.argmax(dim=1).squeeze().numpy()

# クラスマッピング: 0=road, 1=sidewalk, 2=building, ...
```

---

## モデル選択比較表

| タスク | モデル | 速度 | 精度 | ユースケース |
|---|---|---|---|---|
| **画像分類** | EfficientNet | 速い | 高い | モバイル、エッジ |
| **画像分類** | ViT-Large | 遅い | 最高 | サーバーサイド |
| **物体検出（高速）** | YOLOv8n | 最速 | 中 | リアルタイム |
| **物体検出（高精度）** | YOLOv8x | 中 | 高い | 高精度要求 |
| **物体検出（E2E）** | DETR | 遅い | 高い | 研究、カスタム |
| **セグメンテーション** | SAM | 中 | 最高 | ゼロショット |
| **セグメンテーション** | SegFormer | 速い | 高い | 自動運転 |

### 画像サイズと精度の関係

| 入力サイズ | 推論速度 | 精度 | 用途 |
|---|---|---|---|
| 224x224 | 最速 | 低〜中 | モバイル分類 |
| 416x416 | 速い | 中 | リアルタイム検出 |
| 640x640 | 中 | 高い | 標準的な検出 |
| 1280x1280 | 遅い | 最高 | 高精度要求 |

---

## アンチパターン

### 1. データ拡張なしでの学習

**問題**: 小規模データセットでデータ拡張を行わないと、モデルが学習データに過学習し、本番で性能が出ない。

**対策**: 回転、反転、色調変換、Mixup、CutMix 等のデータ拡張を適用する。特に小規模データでは拡張が精度に大きく影響する。

### 2. 不適切な入力前処理

**問題**: 事前学習済みモデルの正規化パラメータ（ImageNet の mean/std）を使わずに推論すると、精度が大幅に低下する。

**対策**: 使用するモデルの前処理仕様を確認し、学習時と推論時で同一の前処理を適用する。

---

## FAQ

### Q1: CNN と Vision Transformer のどちらを使うべきですか？

**A**: データが少ない（数千枚以下）場合は CNN + 転移学習が安定します。大規模データ（数万枚以上）があれば ViT が高精度です。実用的には EfficientNet（CNN）か DINOv2（ViT ベースの自己教師学習）が汎用性が高いです。

### Q2: リアルタイム物体検出の最低要件は？

**A**: 30 FPS 以上を目指す場合、YOLOv8n + GPU（RTX 3060 以上）で 640x640 入力が基本です。エッジデバイスでは TensorRT や ONNX Runtime での最適化が必要です。

### Q3: SAM は何がすごいのですか？

**A**: SAM は「ゼロショット」でセグメンテーションを実行できます。特定のクラスの学習データなしに、クリック1つで任意のオブジェクトをセグメント化できるため、アノテーションツールや汎用的な画像編集に革命的です。

---

## まとめ

| 項目 | 要点 |
|---|---|
| 画像分類 | CNN + 転移学習が基本。大規模データでは ViT |
| 物体検出 | リアルタイムは YOLO、高精度は DETR |
| セグメンテーション | SAM でゼロショット、SegFormer で高速処理 |
| データ拡張 | 小規模データでは必須。精度に直結 |
| 前処理 | モデルごとの正規化パラメータを厳守 |
| エッジ推論 | TensorRT/ONNX で最適化、量子化で高速化 |

## 次に読むべきガイド

- [MLOps](./02-mlops.md) — CV モデルのデプロイと運用
- [RNN/Transformer](../02-deep-learning/02-rnn-transformer.md) — Vision Transformer の基礎

## 参考文献

1. **Ultralytics**: [YOLOv8 Documentation](https://docs.ultralytics.com/) — YOLO の最新ドキュメント
2. **Kirillov et al.**: [Segment Anything (2023)](https://arxiv.org/abs/2304.02643) — SAM の原論文
3. **Dosovitskiy et al.**: [An Image is Worth 16x16 Words (2020)](https://arxiv.org/abs/2010.11929) — Vision Transformer の原論文

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

### コード例 1b: 転移学習の完全な学習・評価パイプライン

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import json
from pathlib import Path

class TransferLearningTrainer:
    """転移学習の完全な学習・評価パイプライン"""

    def __init__(self, model, device="auto"):
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.history = {"train_loss": [], "train_acc": [],
                        "val_loss": [], "val_acc": []}

    def train(self, train_loader, val_loader, epochs=30, lr=1e-3,
              patience=5, save_dir="checkpoints"):
        """段階的ファインチューニング付き学習"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Phase 1: 分類ヘッドのみ学習
        print("Phase 1: 分類ヘッドの学習")
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        no_improve_count = 0

        for epoch in range(epochs):
            # 学習フェーズ
            self.model.train()
            total_loss, correct, total = 0, 0, 0
            start = time.time()

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            scheduler.step()
            train_loss = total_loss / total
            train_acc = correct / total

            # 検証フェーズ
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            elapsed = time.time() - start

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"Train Loss={train_loss:.4f} Acc={train_acc:.4f}  "
                  f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}  "
                  f"Time={elapsed:.1f}s")

            # Early Stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve_count = 0
                torch.save(self.model.state_dict(),
                           f"{save_dir}/best_model.pt")
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # 最良モデルの復元
        self.model.load_state_dict(
            torch.load(f"{save_dir}/best_model.pt")
        )
        return self.history

    def evaluate(self, loader, criterion=None):
        """検証/テスト評価"""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

    def predict(self, images):
        """バッチ推論"""
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
        return predicted.cpu(), probabilities.cpu()

    def save_training_report(self, path="training_report.json"):
        """学習レポートをJSON保存"""
        report = {
            "best_val_acc": max(self.history["val_acc"]),
            "final_train_acc": self.history["train_acc"][-1],
            "epochs_trained": len(self.history["train_loss"]),
            "history": self.history,
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

# 使用例
# trainer = TransferLearningTrainer(model)
# history = trainer.train(train_loader, val_loader, epochs=30)
# trainer.save_training_report()
```

### コード例 1c: Vision Transformer (ViT) による画像分類

```python
import torch
import torch.nn as nn
from torchvision import models

class ViTClassifier(nn.Module):
    """Vision Transformer ベースの画像分類器"""

    def __init__(self, num_classes, model_name="vit_b_16", pretrained=True):
        super().__init__()
        if model_name == "vit_b_16":
            self.backbone = models.vit_b_16(
                weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            )
        elif model_name == "vit_l_16":
            self.backbone = models.vit_l_16(
                weights=models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
            )

        # 分類ヘッドを置き換え
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        """バックボーンを凍結"""
        for name, param in self.backbone.named_parameters():
            if "heads" not in name:
                param.requires_grad = False

    def unfreeze_last_n_blocks(self, n=4):
        """最後のN個のTransformerブロックを解凍"""
        # まず全部凍結
        for param in self.backbone.parameters():
            param.requires_grad = False
        # ヘッドは常に学習可能
        for param in self.backbone.heads.parameters():
            param.requires_grad = True
        # 最後のNブロックを解凍
        total_blocks = len(self.backbone.encoder.layers)
        for i in range(total_blocks - n, total_blocks):
            for param in self.backbone.encoder.layers[i].parameters():
                param.requires_grad = True

    def get_attention_maps(self, x):
        """Attentionマップを取得（可視化用）"""
        self.backbone.eval()
        attention_maps = []
        hooks = []

        def hook_fn(module, input, output):
            # Multi-Head Attentionの出力を記録
            attention_maps.append(output)

        for layer in self.backbone.encoder.layers:
            hook = layer.self_attention.register_forward_hook(hook_fn)
            hooks.append(hook)

        with torch.no_grad():
            _ = self.backbone(x)

        for hook in hooks:
            hook.remove()

        return attention_maps

# 使用例
vit_model = ViTClassifier(num_classes=10, model_name="vit_b_16")
vit_model.freeze_backbone()
print(f"学習可能パラメータ: {sum(p.numel() for p in vit_model.parameters() if p.requires_grad):,}")
```

### コード例 1d: DINOv2 による自己教師あり特徴抽出

```python
import torch
import torch.nn as nn

class DINOv2Classifier(nn.Module):
    """DINOv2の事前学習済み特徴抽出器を使った分類器"""

    def __init__(self, num_classes, model_size="small"):
        super().__init__()
        # DINOv2モデルの読み込み
        model_name = f"dinov2_vit{model_size[0]}14"
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)

        # バックボーンの特徴次元を取得
        embed_dim = self.backbone.embed_dim

        # 分類ヘッド
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        # バックボーンを凍結（線形プロービング）
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)  # [batch, embed_dim]
        return self.classifier(features)

# 使用例
# dino_model = DINOv2Classifier(num_classes=10, model_size="small")
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

### コード例 2b: YOLOv8 カスタムデータセットの準備と学習

```python
import os
import yaml
import shutil
from pathlib import Path
import random

def prepare_yolo_dataset(image_dir, label_dir, output_dir,
                          train_ratio=0.8, val_ratio=0.1):
    """YOLO形式のデータセットを準備する"""
    output = Path(output_dir)
    for split in ["train", "val", "test"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 画像リストを取得してシャッフル
    images = sorted(Path(image_dir).glob("*.jpg"))
    random.shuffle(images)

    n = len(images)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split_name, split_images in splits.items():
        for img_path in split_images:
            # 画像をコピー
            shutil.copy2(img_path, output / "images" / split_name / img_path.name)
            # ラベルをコピー
            label_path = Path(label_dir) / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, output / "labels" / split_name / label_path.name)

    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, "
          f"Test: {len(splits['test'])}")

    # dataset.yaml を生成
    config = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "class_0", 1: "class_1", 2: "class_2"},  # クラス名を適宜変更
    }
    yaml_path = output / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return str(yaml_path)


def train_yolo_with_best_practices(dataset_yaml, model_size="n"):
    """ベストプラクティスを適用したYOLO学習"""
    from ultralytics import YOLO

    model = YOLO(f"yolov8{model_size}.pt")

    results = model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,          # Early stopping
        save=True,
        save_period=10,       # 10エポックごとにチェックポイント
        device="0",           # GPU 0
        workers=8,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,             # 最終学習率の比率
        warmup_epochs=3,
        warmup_momentum=0.8,
        cos_lr=True,          # Cosine学習率スケジューラ
        # データ拡張
        augment=True,
        hsv_h=0.015,          # 色相の変化
        hsv_s=0.7,            # 彩度の変化
        hsv_v=0.4,            # 明度の変化
        degrees=10.0,         # 回転角度
        translate=0.1,        # 平行移動
        scale=0.5,            # スケール変化
        fliplr=0.5,           # 左右反転確率
        mosaic=1.0,           # Mosaic拡張確率
        mixup=0.15,           # Mixup確率
    )
    return results


def evaluate_yolo_model(model_path, dataset_yaml):
    """学習済みモデルの詳細評価"""
    from ultralytics import YOLO

    model = YOLO(model_path)

    # 検証セットで評価
    metrics = model.val(data=dataset_yaml, split="val")

    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    # クラスごとの性能
    for i, name in enumerate(metrics.names.values()):
        print(f"  {name}: mAP50={metrics.box.ap50[i]:.4f}")

    return metrics
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

### コード例 3b: Grounding DINO によるオープンセット物体検出

```python
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import torch

class OpenVocabularyDetector:
    """テキストプロンプトで任意のオブジェクトを検出する"""

    def __init__(self, model_name="IDEA-Research/grounding-dino-base"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def detect(self, image_path, text_prompt, threshold=0.3):
        """テキストプロンプトに基づく物体検出"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=threshold,
            text_threshold=threshold,
            target_sizes=[image.size[::-1]]
        )[0]

        detections = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            detections.append({
                "label": label,
                "score": float(score),
                "box": [round(c, 2) for c in box.tolist()],
            })

        return detections

# 使用例
# detector = OpenVocabularyDetector()
# results = detector.detect("photo.jpg", "dog. cat. person.")
# for det in results:
#     print(f"  {det['label']}: {det['score']:.3f} {det['box']}")
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

### コード例 4b: SAM 2 による動画セグメンテーション

```python
import torch
import numpy as np

class VideoSegmentationPipeline:
    """SAM 2 を使った動画セグメンテーション"""

    def __init__(self, model_path="sam2_hiera_large.pt"):
        # SAM 2 の初期化（概念的なコード）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def segment_video(self, video_path, initial_prompts, output_path=None):
        """
        動画全体をセグメンテーション

        Parameters:
            video_path: 入力動画のパス
            initial_prompts: 初期フレームのプロンプト
                {"points": [[x, y]], "labels": [1], "frame_idx": 0}
            output_path: 出力動画のパス
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        masks_history = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_idx == initial_prompts.get("frame_idx", 0):
                # 初期フレーム: プロンプトベースのセグメンテーション
                mask = self._segment_with_prompt(frame_rgb, initial_prompts)
            else:
                # 後続フレーム: 前フレームのマスクを伝播
                mask = self._propagate_mask(frame_rgb, masks_history[-1])

            masks_history.append(mask)

            if output_path:
                # マスクをオーバーレイ
                overlay = self._apply_mask_overlay(frame, mask)
                writer.write(overlay)

            frame_idx += 1

        cap.release()
        if output_path:
            writer.release()

        return masks_history

    def _segment_with_prompt(self, frame, prompts):
        """プロンプトベースのセグメンテーション（初期フレーム）"""
        # 実装は SAM 2 の API に依存
        pass

    def _propagate_mask(self, frame, prev_mask):
        """前フレームのマスクを現フレームに伝播"""
        # SAM 2 のメモリメカニズムで追跡
        pass

    def _apply_mask_overlay(self, frame, mask, color=(0, 255, 0), alpha=0.4):
        """マスクを画像にオーバーレイ"""
        overlay = frame.copy()
        overlay[mask > 0] = (
            overlay[mask > 0] * (1 - alpha) +
            np.array(color) * alpha
        ).astype(np.uint8)
        return overlay
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

### コード例 5b: U-Net によるカスタムセグメンテーション

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """U-Netの基本ブロック: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """U-Net セグメンテーションモデル"""

    def __init__(self, in_channels=3, num_classes=2, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ダウンサンプリングパス（エンコーダ）
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # ボトルネック
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # アップサンプリングパス（デコーダ）
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # 最終出力
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # エンコーダ
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # デコーダ
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # TransposedConv
            skip = skip_connections[idx // 2]

            # サイズが一致しない場合のパディング
            if x.shape != skip.shape:
                x = nn.functional.pad(x, [
                    0, skip.shape[3] - x.shape[3],
                    0, skip.shape[2] - x.shape[2]
                ])

            x = torch.cat([skip, x], dim=1)  # スキップ接続
            x = self.ups[idx + 1](x)  # DoubleConv

        return self.final_conv(x)


def dice_loss(pred, target, smooth=1.0):
    """Dice Loss: セグメンテーション用の損失関数"""
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    return 1 - (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )


class CombinedLoss(nn.Module):
    """BCE + Dice の複合損失関数"""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        d_loss = dice_loss(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * d_loss

# 使用例
model = UNet(in_channels=3, num_classes=1)
criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 4. 画像処理の高度な応用

### コード例 6: 画像の品質評価と前処理パイプライン

```python
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2

class ImageQualityChecker:
    """画像品質の自動チェックと前処理"""

    def __init__(self, min_size=100, max_size=4096,
                 min_brightness=30, max_brightness=230):
        self.min_size = min_size
        self.max_size = max_size
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def check_image(self, image_path):
        """画像の品質チェック"""
        issues = []
        img = cv2.imread(str(image_path))
        if img is None:
            return {"path": str(image_path), "valid": False,
                    "issues": ["読み込み失敗"]}

        h, w = img.shape[:2]

        # サイズチェック
        if h < self.min_size or w < self.min_size:
            issues.append(f"小さすぎる: {w}x{h}")
        if h > self.max_size or w > self.max_size:
            issues.append(f"大きすぎる: {w}x{h}")

        # アスペクト比チェック
        aspect = max(h, w) / min(h, w)
        if aspect > 5:
            issues.append(f"極端なアスペクト比: {aspect:.1f}")

        # 明るさチェック
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()
        if mean_brightness < self.min_brightness:
            issues.append(f"暗すぎる: 平均輝度={mean_brightness:.0f}")
        if mean_brightness > self.max_brightness:
            issues.append(f"明るすぎる: 平均輝度={mean_brightness:.0f}")

        # ぼかし検出（Laplacianの分散）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:
            issues.append(f"ぼやけている: Laplacian分散={laplacian_var:.1f}")

        # 破損チェック
        if img.shape[2] != 3:
            issues.append(f"チャネル数が異常: {img.shape[2]}")

        return {
            "path": str(image_path),
            "valid": len(issues) == 0,
            "size": (w, h),
            "brightness": round(mean_brightness, 1),
            "sharpness": round(laplacian_var, 1),
            "issues": issues,
        }

    def check_dataset(self, image_dir, extensions=(".jpg", ".jpeg", ".png")):
        """データセット全体の品質チェック"""
        results = []
        image_dir = Path(image_dir)
        for ext in extensions:
            for img_path in image_dir.rglob(f"*{ext}"):
                results.append(self.check_image(img_path))

        total = len(results)
        valid = sum(1 for r in results if r["valid"])
        invalid = total - valid

        print(f"画像総数: {total}")
        print(f"正常: {valid} ({valid/total*100:.1f}%)")
        print(f"問題あり: {invalid} ({invalid/total*100:.1f}%)")

        # 問題のサマリー
        all_issues = [issue for r in results for issue in r["issues"]]
        from collections import Counter
        issue_counts = Counter(all_issues)
        if issue_counts:
            print("\n問題の種類:")
            for issue, count in issue_counts.most_common():
                print(f"  {issue}: {count}件")

        return results

# 使用例
# checker = ImageQualityChecker()
# results = checker.check_dataset("data/images/")
```

### コード例 7: TensorRT による推論最適化

```python
import torch
import time
import numpy as np

class InferenceOptimizer:
    """推論の最適化とベンチマーク"""

    @staticmethod
    def export_to_onnx(model, input_shape, output_path="model.onnx"):
        """PyTorchモデルをONNXにエクスポート"""
        model.eval()
        dummy_input = torch.randn(*input_shape)

        torch.onnx.export(
            model, dummy_input, output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
        print(f"ONNXモデル保存: {output_path}")

    @staticmethod
    def benchmark_model(model, input_shape, n_runs=100, device="cuda"):
        """モデルの推論速度をベンチマーク"""
        model.eval()
        model.to(device)
        dummy = torch.randn(*input_shape).to(device)

        # ウォームアップ
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy)

        if device == "cuda":
            torch.cuda.synchronize()

        # ベンチマーク
        latencies = []
        for _ in range(n_runs):
            start = time.time()
            with torch.no_grad():
                _ = model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.time() - start) * 1000)

        latencies = np.array(latencies)
        print(f"推論レイテンシ (ms):")
        print(f"  平均: {latencies.mean():.2f}")
        print(f"  中央値: {np.median(latencies):.2f}")
        print(f"  P95: {np.percentile(latencies, 95):.2f}")
        print(f"  P99: {np.percentile(latencies, 99):.2f}")
        print(f"  スループット: {1000 / latencies.mean():.1f} FPS")

        return {
            "mean_ms": latencies.mean(),
            "median_ms": np.median(latencies),
            "p95_ms": np.percentile(latencies, 95),
            "fps": 1000 / latencies.mean(),
        }

    @staticmethod
    def optimize_with_torch_compile(model):
        """torch.compile による最適化（PyTorch 2.0+）"""
        optimized = torch.compile(model, mode="reduce-overhead")
        return optimized

    @staticmethod
    def quantize_model(model, calibration_loader=None):
        """INT8 量子化"""
        model.eval()

        if calibration_loader:
            # 動的量子化
            quantized = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )
        else:
            quantized = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )

        # サイズ比較
        import os, tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(model.state_dict(), f.name)
            original_size = os.path.getsize(f.name) / 1024 / 1024
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(quantized.state_dict(), f.name)
            quantized_size = os.path.getsize(f.name) / 1024 / 1024

        print(f"元のサイズ: {original_size:.1f} MB")
        print(f"量子化後: {quantized_size:.1f} MB")
        print(f"圧縮率: {original_size / quantized_size:.1f}x")

        return quantized

# 使用例
# optimizer = InferenceOptimizer()
# optimizer.benchmark_model(model, (1, 3, 224, 224))
# optimizer.export_to_onnx(model, (1, 3, 224, 224))
```

---

## 5. 実践的なユースケース

### ユースケース1: 製品の外観検査（異常検出）

```python
import torch
import torch.nn as nn
from torchvision import models, transforms

class AnomalyDetector:
    """製造ラインでの外観異常検出"""

    def __init__(self, feature_extractor="resnet18"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 特徴抽出器として使用（最終層の前まで）
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]
        ).to(self.device)
        self.feature_extractor.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.normal_features = None
        self.threshold = None

    def fit(self, normal_images):
        """正常画像から特徴量分布を学習"""
        features_list = []
        for img in normal_images:
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.feature_extractor(img_tensor).squeeze()
            features_list.append(feat.cpu().numpy())

        import numpy as np
        self.normal_features = np.array(features_list)
        self.mean = self.normal_features.mean(axis=0)
        self.cov_inv = np.linalg.pinv(np.cov(self.normal_features.T))

        # マハラノビス距離の閾値を設定
        distances = [self._mahalanobis_distance(f) for f in self.normal_features]
        self.threshold = np.percentile(distances, 95)

    def _mahalanobis_distance(self, feature):
        """マハラノビス距離を計算"""
        import numpy as np
        diff = feature - self.mean
        return float(np.sqrt(diff @ self.cov_inv @ diff))

    def predict(self, image):
        """異常スコアを計算"""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.feature_extractor(img_tensor).squeeze().cpu().numpy()
        distance = self._mahalanobis_distance(feat)
        is_anomaly = distance > self.threshold
        return {
            "anomaly_score": distance,
            "is_anomaly": is_anomaly,
            "threshold": self.threshold,
        }
```

### ユースケース2: OCR（光学文字認識）パイプライン

```python
class OCRPipeline:
    """画像からテキストを抽出するOCRパイプライン"""

    def __init__(self):
        self.reader = None

    def initialize(self, languages=["ja", "en"]):
        """EasyOCR の初期化"""
        import easyocr
        self.reader = easyocr.Reader(languages, gpu=torch.cuda.is_available())

    def extract_text(self, image_path, detail=True):
        """画像からテキストを抽出"""
        results = self.reader.readtext(str(image_path))

        if detail:
            extracted = []
            for bbox, text, confidence in results:
                extracted.append({
                    "text": text,
                    "confidence": round(confidence, 4),
                    "bbox": bbox,
                })
            return extracted
        else:
            return " ".join([text for _, text, _ in results])

    def extract_from_document(self, image_path, layout_analysis=True):
        """文書画像からの構造化テキスト抽出"""
        results = self.extract_text(image_path, detail=True)

        if layout_analysis:
            # 上から下、左から右の順に並べ替え
            results.sort(key=lambda r: (
                min(p[1] for p in r["bbox"]),  # y座標でソート
                min(p[0] for p in r["bbox"]),  # x座標でソート
            ))

        lines = []
        current_line = []
        prev_y = None

        for result in results:
            y = min(p[1] for p in result["bbox"])
            if prev_y is not None and abs(y - prev_y) > 20:
                lines.append(" ".join([r["text"] for r in current_line]))
                current_line = []
            current_line.append(result)
            prev_y = y

        if current_line:
            lines.append(" ".join([r["text"] for r in current_line]))

        return "\n".join(lines)
```

---

## モデル選択比較表

| タスク | モデル | 速度 | 精度 | ユースケース |
|---|---|---|---|---|
| **画像分類** | EfficientNet | 速い | 高い | モバイル、エッジ |
| **画像分類** | ViT-Large | 遅い | 最高 | サーバーサイド |
| **画像分類** | DINOv2 | 中 | 最高 | ゼロショット・少数ショット |
| **物体検出（高速）** | YOLOv8n | 最速 | 中 | リアルタイム |
| **物体検出（高精度）** | YOLOv8x | 中 | 高い | 高精度要求 |
| **物体検出（E2E）** | DETR | 遅い | 高い | 研究、カスタム |
| **物体検出（オープン）** | Grounding DINO | 遅い | 高い | テキスト指定検出 |
| **セグメンテーション** | SAM | 中 | 最高 | ゼロショット |
| **セグメンテーション** | SAM 2 | 中 | 最高 | 動画セグメンテーション |
| **セグメンテーション** | SegFormer | 速い | 高い | 自動運転 |
| **セグメンテーション** | U-Net | 速い | 高い | 医療画像 |

### 画像サイズと精度の関係

| 入力サイズ | 推論速度 | 精度 | 用途 |
|---|---|---|---|
| 224x224 | 最速 | 低〜中 | モバイル分類 |
| 416x416 | 速い | 中 | リアルタイム検出 |
| 640x640 | 中 | 高い | 標準的な検出 |
| 1280x1280 | 遅い | 最高 | 高精度要求 |

### 推論最適化手法の比較

| 手法 | 速度向上 | 精度影響 | 導入難易度 | 対応フレームワーク |
|---|---|---|---|---|
| torch.compile | 1.5-3x | なし | 低 | PyTorch 2.0+ |
| ONNX Runtime | 1.5-2x | なし | 低 | フレームワーク非依存 |
| TensorRT | 2-5x | 微小 | 中 | NVIDIA GPU |
| INT8量子化 | 2-4x | 小 | 中 | 各フレームワーク |
| プルーニング | 1.5-3x | 小〜中 | 高 | PyTorch |
| 知識蒸留 | 2-10x | 小〜中 | 高 | 各フレームワーク |

---

## トラブルシューティング

### 問題1: GPUメモリ不足（CUDA Out of Memory）

```python
# 対処法1: バッチサイズを減らす
train_loader = DataLoader(dataset, batch_size=8)  # 16 → 8

# 対処法2: 混合精度学習（メモリ使用量を約半減）
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    output = model(images)
    loss = criterion(output, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 対処法3: 勾配蓄積（実効バッチサイズを維持しつつメモリ節約）
accumulation_steps = 4  # 4回分の勾配を蓄積
for i, (images, labels) in enumerate(train_loader):
    outputs = model(images.cuda())
    loss = criterion(outputs, labels.cuda()) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 対処法4: 画像サイズを縮小
transform = transforms.Resize(160)  # 224 → 160

# 対処法5: 不要なテンソルの解放
del outputs, loss
torch.cuda.empty_cache()
```

### 問題2: 学習が収束しない

```python
# チェックリスト:
# 1. 学習率が適切か確認
# 2. データの正規化が正しいか確認
# 3. 損失関数が適切か確認

# 学習率探索（LR Finder）
from torch.optim.lr_scheduler import OneCycleLR

# 小さい学習率から大きい学習率まで試す
lrs = []
losses = []
lr = 1e-7
model_copy = copy.deepcopy(model)
optimizer = optim.Adam(model_copy.parameters(), lr=lr)

for batch in train_loader:
    optimizer.param_groups[0]["lr"] = lr
    images, labels = batch
    outputs = model_copy(images.cuda())
    loss = criterion(outputs, labels.cuda())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    lrs.append(lr)
    losses.append(loss.item())
    lr *= 1.1  # 学習率を指数的に増加

    if lr > 1.0:
        break

# 損失が最も急激に減少する学習率を選択
```

### 問題3: クラス不均衡

```python
# 対処法1: 重み付き損失関数
from collections import Counter
class_counts = Counter(labels)
total = sum(class_counts.values())
weights = torch.tensor([total / class_counts[i] for i in range(num_classes)])
weights = weights / weights.sum() * num_classes
criterion = nn.CrossEntropyLoss(weight=weights.cuda())

# 対処法2: オーバーサンプリング
from torch.utils.data import WeightedRandomSampler
sample_weights = [1.0 / class_counts[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)

# 対処法3: Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

---

## アンチパターン

### 1. データ拡張なしでの学習

**問題**: 小規模データセットでデータ拡張を行わないと、モデルが学習データに過学習し、本番で性能が出ない。

**対策**: 回転、反転、色調変換、Mixup、CutMix 等のデータ拡張を適用する。特に小規模データでは拡張が精度に大きく影響する。

### 2. 不適切な入力前処理

**問題**: 事前学習済みモデルの正規化パラメータ（ImageNet の mean/std）を使わずに推論すると、精度が大幅に低下する。

**対策**: 使用するモデルの前処理仕様を確認し、学習時と推論時で同一の前処理を適用する。

### 3. 転移学習で全層を最初から学習

```python
# BAD: 全パラメータを同じ学習率で学習
optimizer = optim.Adam(model.parameters(), lr=0.001)

# GOOD: 段階的ファインチューニング
# Phase 1: ヘッドのみ
for param in model.backbone.parameters():
    param.requires_grad = False
optimizer = optim.Adam(model.backbone.fc.parameters(), lr=0.001)
# 数エポック学習

# Phase 2: 全体を低い学習率で
for param in model.parameters():
    param.requires_grad = True
optimizer = optim.Adam([
    {"params": model.backbone.layer4.parameters(), "lr": 1e-4},
    {"params": model.backbone.fc.parameters(), "lr": 1e-3},
], lr=1e-5)
```

### 4. テスト時にmodel.eval()を忘れる

```python
# BAD: 推論時にモードを切り替えない
output = model(test_images)  # Dropout/BNが学習モードのまま

# GOOD:
model.eval()
with torch.no_grad():
    output = model(test_images)
```

---

## パフォーマンス最適化チェックリスト

- [ ] **データローダー**: `num_workers > 0`, `pin_memory=True`（GPU使用時）
- [ ] **混合精度学習**: `torch.cuda.amp` でFP16を使用
- [ ] **バッチサイズ**: GPUメモリに収まる最大サイズを使用
- [ ] **データ拡張**: GPU上で実行（`torchvision.transforms.v2` or Albumentations）
- [ ] **モデル選択**: タスクに適したサイズのモデルを選択
- [ ] **学習率**: Warmup + Cosine Annealing が安定
- [ ] **Early Stopping**: 過学習防止のために必須
- [ ] **勾配クリッピング**: `torch.nn.utils.clip_grad_norm_` で安定化
- [ ] **推論最適化**: ONNX/TensorRT/torch.compile で高速化
- [ ] **キャッシュ**: 前処理済みデータのキャッシュで I/O削減

---

## FAQ

### Q1: CNN と Vision Transformer のどちらを使うべきですか？

**A**: データが少ない（数千枚以下）場合は CNN + 転移学習が安定します。大規模データ（数万枚以上）があれば ViT が高精度です。実用的には EfficientNet（CNN）か DINOv2（ViT ベースの自己教師学習）が汎用性が高いです。

### Q2: リアルタイム物体検出の最低要件は？

**A**: 30 FPS 以上を目指す場合、YOLOv8n + GPU（RTX 3060 以上）で 640x640 入力が基本です。エッジデバイスでは TensorRT や ONNX Runtime での最適化が必要です。

### Q3: SAM は何がすごいのですか？

**A**: SAM は「ゼロショット」でセグメンテーションを実行できます。特定のクラスの学習データなしに、クリック1つで任意のオブジェクトをセグメント化できるため、アノテーションツールや汎用的な画像編集に革命的です。

### Q4: 画像データのアノテーションを効率化するには？

**A**: (1) SAMをベースにした半自動アノテーション（クリック1つで輪郭生成）、(2) アクティブラーニング（モデルが不確実なサンプルを優先的にアノテーション）、(3) 弱教師あり学習（画像レベルのラベルからピクセルレベルの予測）、(4) データ拡張で少量のアノテーションから多くの学習データを生成。ツールとしてはLabel Studio、CVAT、Roboflowが代表的。

### Q5: エッジデバイスへのデプロイ方法は？

**A**: (1) モデルの軽量化（MobileNet、EfficientNet-Lite）、(2) 量子化（INT8、FP16）、(3) TensorRT/ONNX Runtimeでの最適化、(4) NVIDIA Jetson、Apple Neural Engine、Google Coral等のハードウェアアクセラレータの活用。Jetson Nanoでは YOLOv8n が TensorRT 最適化で 30+ FPS を達成可能。

### Q6: 3D コンピュータビジョンにはどんな手法がありますか？

**A**: (1) NeRF（Neural Radiance Fields）: 2D画像群から3Dシーンを再構成、(2) 3D Gaussian Splatting: リアルタイム3Dレンダリング、(3) Point Cloud処理: LiDARデータの分類・セグメンテーション（PointNet++、Point Transformer）、(4) Depth Estimation: 単眼深度推定（MiDaS、Depth Anything）。

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
| 品質管理 | 画像品質チェックをパイプラインに組み込む |
| 新技術 | DINOv2、SAM 2、Grounding DINO が注目 |

## 次に読むべきガイド

- [MLOps](./02-mlops.md) — CV モデルのデプロイと運用
- [RNN/Transformer](../02-deep-learning/02-rnn-transformer.md) — Vision Transformer の基礎

## 参考文献

1. **Ultralytics**: [YOLOv8 Documentation](https://docs.ultralytics.com/) — YOLO の最新ドキュメント
2. **Kirillov et al.**: [Segment Anything (2023)](https://arxiv.org/abs/2304.02643) — SAM の原論文
3. **Dosovitskiy et al.**: [An Image is Worth 16x16 Words (2020)](https://arxiv.org/abs/2010.11929) — Vision Transformer の原論文
4. **Oquab et al.**: [DINOv2: Learning Robust Visual Features (2023)](https://arxiv.org/abs/2304.07193) — DINOv2 の原論文
5. **Liu et al.**: [Grounding DINO (2023)](https://arxiv.org/abs/2303.05499) — オープンセット物体検出

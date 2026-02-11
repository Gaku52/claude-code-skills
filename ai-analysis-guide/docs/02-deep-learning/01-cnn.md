# CNN — 畳み込み、プーリング、画像認識

> 畳み込みニューラルネットワークの構造と原理を理解し、画像認識タスクに適用する

## この章で学ぶこと

1. **畳み込み演算** — カーネルによる局所的特徴抽出の仕組みとパラメータ共有
2. **プーリングとストライド** — 空間サイズの削減と平行移動不変性の獲得
3. **代表的アーキテクチャ** — LeNet、VGG、ResNet、EfficientNetの設計思想

---

## 1. 畳み込み演算の仕組み

### 2D畳み込みの動作

```
入力画像 (5x5)               カーネル (3x3)         出力 (3x3)

┌───┬───┬───┬───┬───┐       ┌───┬───┬───┐
│ 1 │ 0 │ 1 │ 0 │ 1 │       │ 1 │ 0 │ 1 │       ┌───┬───┬───┐
├───┼───┼───┼───┼───┤       ├───┼───┼───┤       │ 4 │ 3 │ 4 │
│ 0 │ 1 │ 0 │ 1 │ 0 │       │ 0 │ 1 │ 0 │       ├───┼───┼───┤
├───┼───┼───┼───┼───┤  *    ├───┼───┼───┤  =    │ 2 │ 4 │ 3 │
│ 1 │ 0 │ 1 │ 0 │ 1 │       │ 1 │ 0 │ 1 │       ├───┼───┼───┤
├───┼───┼───┼───┼───┤       └───┴───┴───┘       │ 4 │ 3 │ 4 │
│ 0 │ 1 │ 0 │ 1 │ 0 │                            └───┴───┴───┘
├───┼───┼───┼───┼───┤
│ 1 │ 0 │ 1 │ 0 │ 1 │
└───┴───┴───┴───┴───┘

出力サイズ = (入力サイズ - カーネルサイズ + 2×パディング) / ストライド + 1
           = (5 - 3 + 0) / 1 + 1 = 3

パディング (Padding):
  "valid" (パディングなし): 出力が縮小
  "same"  (ゼロパディング): 出力サイズ = 入力サイズ
```

### CNNの全体構造

```
入力画像          畳み込み層         プーリング層       全結合層        出力
(H×W×C)          (特徴抽出)         (ダウンサンプル)   (分類/回帰)

┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────┐    ┌──────┐
│ 224x224  │    │ Conv 3x3 │    │ MaxPool  │    │      │    │      │
│ x3 (RGB) │───>│ + ReLU   │───>│ 2x2      │───>│ FC   │───>│ 1000 │
│          │    │ 64フィルタ│    │          │    │ 層   │    │クラス│
│          │    │          │    │          │    │      │    │      │
└──────────┘    └──────────┘    └──────────┘    └──────┘    └──────┘
                     │               │
                     v               v
              特徴マップ         サイズ半減
              224x224x64        112x112x64

  → 畳み込み+プーリングを繰り返して特徴を階層的に抽出
  → 低レベル特徴（エッジ）→ 中レベル（テクスチャ）→ 高レベル（物体部品）
```

### コード例1: PyTorchでのCNN基本実装

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """基本的なCNNアーキテクチャ"""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 畳み込みブロック1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                                kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # 畳み込みブロック2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 畳み込みブロック3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # プーリング
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全結合層
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1: Conv → BN → ReLU → Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28x28 → 14x14
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14x14 → 7x7
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))              # 7x7
        x = self.adaptive_pool(x)                         # 1x1
        # Flatten → FC
        x = x.view(x.size(0), -1)                         # (B, 128)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# モデル概要
model = SimpleCNN(num_classes=10)
print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")

# ダミー入力で形状確認
dummy = torch.randn(1, 1, 28, 28)
output = model(dummy)
print(f"出力形状: {output.shape}")  # (1, 10)
```

### コード例2: 完全な学習ループ

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

def train_cnn():
    """MNIST分類の完全な学習パイプライン"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    # データ前処理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST("data", train=True, download=True,
                                    transform=transform)
    test_dataset = datasets.MNIST("data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = SimpleCNN(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss()

    # 学習ループ
    for epoch in range(10):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total

        # テスト評価
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()
                test_total += labels.size(0)

        test_acc = test_correct / test_total
        elapsed = time.time() - start
        print(f"Epoch {epoch+1:2d}  "
              f"Loss={total_loss/len(train_loader):.4f}  "
              f"Train Acc={train_acc:.4f}  "
              f"Test Acc={test_acc:.4f}  "
              f"Time={elapsed:.1f}s")

    return model

# model = train_cnn()
```

---

## 2. 代表的アーキテクチャ

### コード例3: ResNet残差ブロックの実装

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """ResNetの残差ブロック"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ショートカット接続（次元が異なる場合）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity     # 残差接続: F(x) + x
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    """簡易版ResNet"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

### コード例4: 転移学習の実装

```python
import torch
import torch.nn as nn
import torchvision.models as models

def create_transfer_model(num_classes: int, freeze_backbone: bool = True):
    """事前学習済みResNet50を使った転移学習モデル"""

    # ImageNetで事前学習済みのResNet50をロード
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # バックボーンの重みを凍結
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # 最終全結合層を置き換え
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    # 新しい層のパラメータは学習可能
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"全パラメータ: {total:,}")
    print(f"学習可能: {trainable:,} ({trainable/total:.1%})")

    return model

# 使用例
model = create_transfer_model(num_classes=5, freeze_backbone=True)
```

### コード例5: データ拡張パイプライン

```python
import torchvision.transforms as T

def get_transforms(image_size: int = 224, is_train: bool = True):
    """訓練/テスト用のデータ拡張を構築"""

    if is_train:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.1),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.2),
        ])
    else:
        return T.Compose([
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

# 使用例
train_transform = get_transforms(224, is_train=True)
test_transform = get_transforms(224, is_train=False)
```

---

## 比較表

### CNNアーキテクチャの進化

| モデル | 年 | 層数 | パラメータ | Top-1精度(ImageNet) | 主な革新 |
|---|---|---|---|---|---|
| LeNet-5 | 1998 | 5 | 60K | - | 最初のCNN |
| AlexNet | 2012 | 8 | 61M | 63.3% | ReLU, Dropout, GPU学習 |
| VGG-16 | 2014 | 16 | 138M | 71.5% | 3x3カーネルの深い積み重ね |
| GoogLeNet | 2014 | 22 | 6.8M | 74.8% | Inceptionモジュール |
| ResNet-50 | 2015 | 50 | 25M | 76.1% | 残差接続 (Skip Connection) |
| EfficientNet-B0 | 2019 | - | 5.3M | 77.1% | 複合スケーリング |
| ViT-B/16 | 2020 | 12 | 86M | 77.9% | 純粋Transformer |
| ConvNeXt-T | 2022 | - | 28M | 82.1% | モダンCNN設計 |

### 畳み込みの種類

| 畳み込み | パラメータ数 | 計算量 | 用途 | 特徴 |
|---|---|---|---|---|
| 標準 (3x3) | C_in × C_out × 9 | O(H×W×C_in×C_out×9) | 汎用 | 全チャネル間の結合 |
| 1x1 | C_in × C_out | O(H×W×C_in×C_out) | チャネル混合 | 次元調整 |
| Depthwise | C × 9 | O(H×W×C×9) | 軽量化 | チャネルごとに独立 |
| Separable | C×9 + C×C_out | 大幅削減 | モバイル | Depth+Pointwise |
| Dilated | C_in × C_out × 9 | 同等 | セグメンテーション | 受容野の拡大 |
| Transposed | C_in × C_out × 9 | 同等 | アップサンプリング | デコーダ |

---

## アンチパターン

### アンチパターン1: 転移学習で全層を最初からファインチューン

```python
# BAD: 少量データで全パラメータを学習 → 事前学習の知識が壊れる
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(2048, 5)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 全パラメータに同じlr

# GOOD: 段階的ファインチューニング
# Phase 1: バックボーン凍結、分類器のみ学習
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, 5)
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
# ... Phase 1 学習 ...

# Phase 2: バックボーンを解凍、低い学習率で全体を微調整
for param in model.parameters():
    param.requires_grad = True
optimizer = optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
], lr=1e-5)  # 浅い層はさらに低い学習率
```

### アンチパターン2: データ拡張なしの画像分類

```python
# BAD: 拡張なし → 少量データで過学習
transform = T.Compose([T.Resize(224), T.ToTensor()])

# GOOD: タスクに適した拡張を適用
# ただし不適切な拡張も害になる:
# - 数字認識で回転180度 → 6と9が区別できなくなる
# - 医療画像で過度な色変換 → 診断情報が失われる
```

---

## FAQ

### Q1: CNNとViT（Vision Transformer）のどちらを使うべき？

**A:** データが少ない場合（数千〜数万枚）はCNN（特にResNet + 転移学習）が安定。大規模データ（数百万枚以上）ではViTが優位。2022年以降のConvNeXtはCNNの設計をモダンにし、ViTと同等以上の性能を達成している。実務ではResNet/EfficientNet + 転移学習が最も汎用的。

### Q2: バッチ正規化はなぜ効くのか？

**A:** (1) 内部共変量シフトの緩和（各層の入力分布を安定化）、(2) 正則化効果（ミニバッチの統計量によるノイズ）、(3) 学習率を大きくできる（勾配の大きさが安定）。推論時はバッチ統計量ではなく学習時の移動平均を使うため、`model.eval()` の呼び出しが必須。

### Q3: GPUメモリが足りない場合の対処法は？

**A:** (1) バッチサイズを減らす、(2) 混合精度学習（FP16）を使う、(3) 勾配蓄積（Gradient Accumulation）で実効バッチサイズを維持、(4) 画像サイズを縮小、(5) モデルを軽量化（EfficientNet等）。PyTorchの `torch.cuda.amp` で混合精度を簡単に適用可能。

---

## まとめ

| 項目 | 要点 |
|---|---|
| 畳み込み | 局所的な特徴をカーネルで抽出。パラメータ共有で効率的 |
| プーリング | 空間解像度を削減し計算量と過学習を抑制 |
| 残差接続 | 深いネットワークの勾配消失を解決（ResNet） |
| 転移学習 | 事前学習済みモデルをファインチューン。少量データに有効 |
| データ拡張 | 訓練時の多様性を増やし汎化性能を向上 |

---

## 次に読むべきガイド

- [02-rnn-transformer.md](./02-rnn-transformer.md) — 系列データ処理のRNN/Transformer
- [../03-applied/01-computer-vision.md](../03-applied/01-computer-vision.md) — 物体検出、セグメンテーション

---

## 参考文献

1. **Kaiming He et al.** "Deep Residual Learning for Image Recognition" CVPR 2016
2. **Alexey Dosovitskiy et al.** "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" ICLR 2021
3. **CS231n: Convolutional Neural Networks for Visual Recognition** Stanford University — https://cs231n.stanford.edu/

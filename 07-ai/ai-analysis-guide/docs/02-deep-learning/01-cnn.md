# CNN — 畳み込み、プーリング、画像認識

> 畳み込みニューラルネットワークの構造と原理を理解し、画像認識タスクに適用する

## この章で学ぶこと

1. **畳み込み演算** — カーネルによる局所的特徴抽出の仕組みとパラメータ共有
2. **プーリングとストライド** — 空間サイズの削減と平行移動不変性の獲得
3. **代表的アーキテクチャ** — LeNet、VGG、ResNet、EfficientNetの設計思想
4. **実践的テクニック** — 転移学習、データ拡張、混合精度学習、モデル最適化
5. **物体検出・セグメンテーション** — CNNを応用した高度な視覚タスク
6. **モデルの可視化と解釈** — 特徴マップ、Grad-CAM、フィルタ可視化

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

### 畳み込み演算の数学的定義

```python
import numpy as np

def conv2d_manual(input_img: np.ndarray, kernel: np.ndarray,
                  stride: int = 1, padding: int = 0) -> np.ndarray:
    """畳み込み演算を手動で実装（理解のため）"""
    # パディング適用
    if padding > 0:
        input_img = np.pad(input_img,
                           ((padding, padding), (padding, padding)),
                           mode='constant', constant_values=0)

    h_in, w_in = input_img.shape
    k_h, k_w = kernel.shape

    # 出力サイズ計算
    h_out = (h_in - k_h) // stride + 1
    w_out = (w_in - k_w) // stride + 1
    output = np.zeros((h_out, w_out))

    # 畳み込み演算（相関演算）
    for i in range(h_out):
        for j in range(w_out):
            region = input_img[i*stride:i*stride+k_h,
                              j*stride:j*stride+k_w]
            output[i, j] = np.sum(region * kernel)

    return output

# エッジ検出カーネルの例
sobel_x = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float32)

sobel_y = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=np.float32)

# ガウシアンぼかし
gaussian_3x3 = np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]], dtype=np.float32) / 16.0

# シャープネスフィルタ
sharpen = np.array([[ 0, -1,  0],
                     [-1,  5, -1],
                     [ 0, -1,  0]], dtype=np.float32)

# テスト
test_img = np.random.rand(8, 8)
result = conv2d_manual(test_img, sobel_x, stride=1, padding=1)
print(f"入力: {test_img.shape} → 出力: {result.shape}")
```

### 受容野（Receptive Field）の計算

```python
def calculate_receptive_field(layers: list[dict]) -> dict:
    """各層の受容野を計算する

    Args:
        layers: [{"kernel": k, "stride": s, "padding": p}, ...]

    Returns:
        各層の受容野サイズとジャンプ
    """
    rf = 1      # 受容野サイズ
    jump = 1    # ジャンプ（ストライドの累積）
    start = 0.5 # 受容野の中心位置

    results = []
    for i, layer in enumerate(layers):
        k = layer["kernel"]
        s = layer["stride"]

        rf = rf + (k - 1) * jump
        jump = jump * s

        results.append({
            "layer": i + 1,
            "kernel": k,
            "stride": s,
            "receptive_field": rf,
            "jump": jump,
        })
        print(f"Layer {i+1}: kernel={k}, stride={s} → "
              f"RF={rf}, jump={jump}")

    return results

# VGG-16の受容野計算例
vgg_layers = [
    {"kernel": 3, "stride": 1, "padding": 1},  # conv1_1
    {"kernel": 3, "stride": 1, "padding": 1},  # conv1_2
    {"kernel": 2, "stride": 2, "padding": 0},  # pool1
    {"kernel": 3, "stride": 1, "padding": 1},  # conv2_1
    {"kernel": 3, "stride": 1, "padding": 1},  # conv2_2
    {"kernel": 2, "stride": 2, "padding": 0},  # pool2
    {"kernel": 3, "stride": 1, "padding": 1},  # conv3_1
    {"kernel": 3, "stride": 1, "padding": 1},  # conv3_2
    {"kernel": 3, "stride": 1, "padding": 1},  # conv3_3
    {"kernel": 2, "stride": 2, "padding": 0},  # pool3
]

print("=== VGG-16 受容野 ===")
results = calculate_receptive_field(vgg_layers)
# 最終層のRFが大きいほど、広い範囲のコンテキストを見ている
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

### コード例: 混合精度学習とGradient Accumulation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

class TrainerWithAMP:
    """混合精度学習と勾配蓄積を組み合わせたトレーナー"""

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,
                 criterion: nn.Module, device: torch.device,
                 accumulation_steps: int = 4):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.scaler = GradScaler()

    def train_epoch(self, train_loader, epoch: int):
        """1エポックの学習（AMP + Gradient Accumulation）"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        self.optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # 混合精度で前方計算
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                # 勾配蓄積のためにロスをスケーリング
                loss = loss / self.accumulation_steps

            # スケーラーを使って後方計算
            self.scaler.scale(loss).backward()

            # N回分のミニバッチ勾配を蓄積してからパラメータ更新
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # 勾配クリッピング（AMP使用時に重要）
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulation_steps
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
        return avg_loss, acc

    @torch.no_grad()
    def evaluate(self, test_loader):
        """評価（AMP対応）"""
        self.model.eval()
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast():
                outputs = self.model(images)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Test Accuracy: {acc:.4f}")
        return acc

# 使用例
# model = SimpleCNN(num_classes=10)
# optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# device = torch.device("cuda")
#
# trainer = TrainerWithAMP(model, optimizer, criterion, device,
#                          accumulation_steps=4)
# # 実効バッチサイズ = バッチサイズ × accumulation_steps
# # 例: batch_size=32 × 4 = 128
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

### Bottleneckブロック（ResNet-50以降）

```python
class BottleneckBlock(nn.Module):
    """ResNet-50/101/152で使用されるBottleneckブロック

    1x1 → 3x3 → 1x1 の3層構造で、
    チャネル数を圧縮してから畳み込みを行うことで計算量を削減する。
    expansion = 4 により出力チャネル数は入力の4倍になる。
    """
    expansion = 4

    def __init__(self, in_channels: int, mid_channels: int,
                 stride: int = 1, groups: int = 1, width_per_group: int = 64):
        super().__init__()

        # ResNeXtのグループ畳み込み対応
        width = int(mid_channels * (width_per_group / 64.0)) * groups

        # 1x1: チャネル圧縮
        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        # 3x3: 空間方向の特徴抽出（グループ畳み込み対応）
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride,
                                padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        # 1x1: チャネル拡張
        out_channels = mid_channels * self.expansion
        self.conv3 = nn.Conv2d(width, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # ショートカット
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        out = self.relu(out)

        return out

# パラメータ比較
basic = ResidualBlock(256, 256)
bottleneck = BottleneckBlock(256, 64)
print(f"BasicBlock パラメータ: "
      f"{sum(p.numel() for p in basic.parameters()):,}")
print(f"Bottleneck パラメータ: "
      f"{sum(p.numel() for p in bottleneck.parameters()):,}")
# Bottleneckの方がパラメータ効率が良い
```

### EfficientNet: 複合スケーリング

```python
import torch
import torch.nn as nn
import math

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution (MBConv)

    EfficientNetの基本ブロック。
    1x1拡張 → Depthwise 3x3/5x5 → SE → 1x1圧縮 + Skip
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, expand_ratio: int = 6,
                 se_ratio: float = 0.25, drop_rate: float = 0.2):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        mid_ch = in_ch * expand_ratio

        layers = []

        # 1x1 拡張（expand_ratio > 1 の場合のみ）
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.SiLU(inplace=True),  # Swish活性化
            ])

        # Depthwise Convolution
        layers.extend([
            nn.Conv2d(mid_ch, mid_ch, kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
        ])

        self.conv = nn.Sequential(*layers)

        # Squeeze-and-Excitation（チャネル注意機構）
        se_ch = max(1, int(in_ch * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_ch, se_ch, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_ch, mid_ch, 1),
            nn.Sigmoid(),
        )

        # 1x1 圧縮
        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        # Stochastic Depth (学習時のみ)
        self.drop_rate = drop_rate

    def forward(self, x):
        identity = x

        out = self.conv(x)
        out = out * self.se(out)   # SE: チャネル重み付け
        out = self.project(out)

        # Stochastic Depth
        if self.use_residual:
            if self.training and self.drop_rate > 0:
                keep_prob = 1 - self.drop_rate
                mask = torch.rand(out.shape[0], 1, 1, 1,
                                  device=out.device) < keep_prob
                out = out * mask / keep_prob
            out = out + identity

        return out

# EfficientNetの複合スケーリング
def efficientnet_scaling(base_width: float, base_depth: float,
                         base_resolution: int, phi: float):
    """EfficientNetの複合スケーリング係数を計算

    alpha^phi * beta^phi * gamma^phi ≈ 2
    (alpha=1.2, beta=1.1, gamma=1.15)
    """
    alpha, beta, gamma = 1.2, 1.1, 1.15

    width_mult = base_width * (alpha ** phi)
    depth_mult = base_depth * (beta ** phi)
    resolution = int(base_resolution * (gamma ** phi))

    print(f"phi={phi:.1f}: width={width_mult:.2f}x, "
          f"depth={depth_mult:.2f}x, resolution={resolution}")
    return width_mult, depth_mult, resolution

# B0〜B7のスケーリング
print("=== EfficientNet スケーリング ===")
for i in range(8):
    efficientnet_scaling(1.0, 1.0, 224, phi=i)
```

### ConvNeXt: モダンCNNの設計

```python
import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    """ConvNeXtブロック — ViTの設計原則をCNNに適用

    設計原則:
    1. Depthwise Conv (7x7) — 大きなカーネルでグローバルな受容野
    2. Layer Normalization — BNの代わり
    3. 逆ボトルネック — 拡張比4（MLP的構造）
    4. GELU活性化 — ReLUの代わり
    5. 少数の活性化/正規化 — 1ブロック1回
    """

    def __init__(self, dim: int, drop_path: float = 0.0,
                 layer_scale_init: float = 1e-6):
        super().__init__()

        # Depthwise Conv (7x7, 大きなカーネル)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7,
                                 padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # 逆ボトルネック MLP (拡張比4x)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # Layer Scale（学習可能なスケーリング係数）
        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(dim)
        ) if layer_scale_init > 0 else None

        # DropPath（Stochastic Depth）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        # (B, C, H, W) → Depthwise Conv
        x = self.dwconv(x)
        # (B, C, H, W) → (B, H, W, C) for LayerNorm
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # → (B, C, H, W)

        x = input + self.drop_path(x)
        return x


class DropPath(nn.Module):
    """Stochastic Depth の実装"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device) < keep_prob
        return x * mask / keep_prob
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

### 段階的ファインチューニング（Progressive Unfreezing）

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class ProgressiveFineTuner:
    """段階的にバックボーンを解凍しながらファインチューニング

    Phase 1: 分類器のみ学習（高い学習率）
    Phase 2: 後半の層を解凍（中程度の学習率）
    Phase 3: 全体を解凍（低い学習率）
    """

    def __init__(self, num_classes: int, device: torch.device):
        self.device = device
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        ).to(device)

        # 全層凍結
        for param in self.model.parameters():
            param.requires_grad = False

        # 分類器を交換
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        ).to(device)

    def get_layer_groups(self):
        """パラメータをグループに分割"""
        return {
            "early": list(self.model.conv1.parameters()) +
                     list(self.model.bn1.parameters()) +
                     list(self.model.layer1.parameters()) +
                     list(self.model.layer2.parameters()),
            "late": list(self.model.layer3.parameters()) +
                    list(self.model.layer4.parameters()),
            "head": list(self.model.fc.parameters()),
        }

    def phase1_head_only(self, lr: float = 1e-3):
        """Phase 1: 分類器のみ学習"""
        print("=== Phase 1: Head Only ===")
        optimizer = optim.Adam(self.model.fc.parameters(), lr=lr)
        return optimizer

    def phase2_unfreeze_late(self, lr_head: float = 1e-3,
                              lr_late: float = 1e-4):
        """Phase 2: 後半の層を解凍"""
        print("=== Phase 2: Unfreeze Late Layers ===")
        groups = self.get_layer_groups()

        for param in groups["late"]:
            param.requires_grad = True

        optimizer = optim.Adam([
            {"params": groups["late"], "lr": lr_late},
            {"params": groups["head"], "lr": lr_head},
        ])
        return optimizer

    def phase3_unfreeze_all(self, lr_early: float = 1e-5,
                             lr_late: float = 1e-4,
                             lr_head: float = 5e-4):
        """Phase 3: 全層解凍（discriminative learning rates）"""
        print("=== Phase 3: Unfreeze All ===")
        groups = self.get_layer_groups()

        for param in groups["early"]:
            param.requires_grad = True

        optimizer = optim.Adam([
            {"params": groups["early"], "lr": lr_early},
            {"params": groups["late"], "lr": lr_late},
            {"params": groups["head"], "lr": lr_head},
        ])
        return optimizer

    def count_trainable(self):
        """学習可能パラメータの確認"""
        trainable = sum(p.numel() for p in self.model.parameters()
                       if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"学習可能: {trainable:,} / {total:,} "
              f"({trainable/total:.1%})")

# 使用例
# device = torch.device("cuda")
# finetuner = ProgressiveFineTuner(num_classes=5, device=device)
#
# # Phase 1: 5エポック
# opt = finetuner.phase1_head_only(lr=1e-3)
# finetuner.count_trainable()
# # → 学習可能: 1,050,117 / 24,607,813 (4.3%)
#
# # Phase 2: 5エポック
# opt = finetuner.phase2_unfreeze_late(lr_head=5e-4, lr_late=1e-4)
# finetuner.count_trainable()
# # → 学習可能: 15,545,861 / 24,607,813 (63.2%)
#
# # Phase 3: 10エポック
# opt = finetuner.phase3_unfreeze_all()
# finetuner.count_trainable()
# # → 学習可能: 24,607,813 / 24,607,813 (100.0%)
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

### 高度なデータ拡張: Albumentations

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

def get_albumentations_transform(image_size: int = 224,
                                  is_train: bool = True):
    """Albumentationsによる高度なデータ拡張パイプライン

    torchvision.transformsとの違い:
    - より多くの拡張手法（Cutout, GridDistortion等）
    - 高速（OpenCV/NumPyベース）
    - バウンディングボックス・セグメンテーションマスクとの連動
    """
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(height=image_size, width=image_size,
                                scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),

            # 色変換
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.2, hue=0.1),
                A.HueSaturationValue(hue_shift_limit=20,
                                      sat_shift_limit=30,
                                      val_shift_limit=20),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                            contrast_limit=0.2),
            ], p=0.8),

            # 幾何変換
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15,
                                    rotate_limit=15),
                A.Affine(shear=(-10, 10)),
                A.Perspective(scale=(0.05, 0.1)),
            ], p=0.5),

            # ノイズ・ブラー
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50)),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MotionBlur(blur_limit=5),
            ], p=0.3),

            # カットアウト系
            A.OneOf([
                A.CoarseDropout(max_holes=8, max_height=image_size // 8,
                                max_width=image_size // 8,
                                fill_value=0),
                A.GridDropout(ratio=0.3, random_offset=True),
            ], p=0.3),

            # 正規化
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(int(image_size * 1.14), int(image_size * 1.14)),
            A.CenterCrop(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

# Mixup / CutMix の実装
class MixupCutmix:
    """Mixup と CutMix を動的に切り替えるデータ拡張"""

    def __init__(self, mixup_alpha: float = 0.2,
                 cutmix_alpha: float = 1.0,
                 mixup_prob: float = 0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob

    def __call__(self, images: 'torch.Tensor',
                 labels: 'torch.Tensor') -> tuple:
        """バッチに対してMixup/CutMixを適用"""
        import torch

        batch_size = images.size(0)
        indices = torch.randperm(batch_size, device=images.device)

        if np.random.rand() < self.mixup_prob:
            # Mixup: 画像とラベルを線形補間
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            mixed_images = lam * images + (1 - lam) * images[indices]
            return mixed_images, labels, labels[indices], lam
        else:
            # CutMix: 画像の一部を別の画像で置き換え
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            _, _, H, W = images.shape

            # カット領域の座標を計算
            cut_ratio = np.sqrt(1.0 - lam)
            cut_h = int(H * cut_ratio)
            cut_w = int(W * cut_ratio)
            cy = np.random.randint(H)
            cx = np.random.randint(W)

            y1 = np.clip(cy - cut_h // 2, 0, H)
            y2 = np.clip(cy + cut_h // 2, 0, H)
            x1 = np.clip(cx - cut_w // 2, 0, W)
            x2 = np.clip(cx + cut_w // 2, 0, W)

            mixed_images = images.clone()
            mixed_images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]

            # ラベルの比率を面積比で調整
            lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
            return mixed_images, labels, labels[indices], lam

# 使用例
# mixup_cutmix = MixupCutmix()
# mixed_images, labels_a, labels_b, lam = mixup_cutmix(images, labels)
# loss = lam * criterion(outputs, labels_a) + (1-lam) * criterion(outputs, labels_b)
```

---

## 3. モデルの可視化と解釈

### Grad-CAMの実装

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GradCAM:
    """Grad-CAM: 勾配重み付きクラス活性化マッピング

    CNNがどの領域に注目して判断を下したかを可視化する。
    最後の畳み込み層の特徴マップに、各チャネルの勾配の
    平均値で重み付けして合算する。
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # フックを登録
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor,
                 target_class: int = None) -> np.ndarray:
        """Grad-CAMヒートマップを生成"""
        self.model.eval()

        # 前方計算
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 対象クラスの勾配を計算
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # チャネルごとの勾配の平均（重み）
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # 重み付き和 + ReLU
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # 入力サイズにリサイズして正規化
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # 0-1に正規化
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def generate_batch(self, input_batch: torch.Tensor,
                       target_classes: list = None) -> list:
        """バッチ全体のGrad-CAMを生成"""
        results = []
        for i in range(input_batch.size(0)):
            target = target_classes[i] if target_classes else None
            cam = self.generate(input_batch[i:i+1], target)
            results.append(cam)
        return results


def visualize_gradcam(image: np.ndarray, cam: np.ndarray,
                       alpha: float = 0.5):
    """Grad-CAMヒートマップを画像に重ね合わせる"""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # ヒートマップ生成
    heatmap = cm.jet(cam)[:, :, :3]  # RGB部分のみ
    heatmap = (heatmap * 255).astype(np.uint8)

    # 元画像の非正規化（ImageNet）
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # オーバーレイ
    overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_result.png", dpi=150, bbox_inches="tight")
    plt.show()

# 使用例
# model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# gradcam = GradCAM(model, model.layer4[-1])
#
# input_tensor = preprocess(image).unsqueeze(0)
# cam = gradcam.generate(input_tensor, target_class=281)  # 281="tabby cat"
# visualize_gradcam(original_image, cam)
```

### フィルタ（カーネル）の可視化

```python
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

def visualize_filters(model: torch.nn.Module, layer_name: str = "conv1"):
    """畳み込み層のフィルタを可視化"""
    # 最初の畳み込み層のフィルタを取得
    layer = dict(model.named_modules())[layer_name]
    filters = layer.weight.data.cpu().numpy()

    n_filters = min(filters.shape[0], 64)
    n_cols = 8
    n_rows = (n_filters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        if i < n_filters:
            # 3チャネル（RGB）フィルタの場合
            f = filters[i]
            if f.shape[0] == 3:
                # 0-1に正規化して表示
                f = (f - f.min()) / (f.max() - f.min() + 1e-8)
                ax.imshow(f.transpose(1, 2, 0))
            else:
                ax.imshow(f[0], cmap="gray")
        ax.axis("off")

    plt.suptitle(f"Filters: {layer_name} ({filters.shape})", fontsize=14)
    plt.tight_layout()
    plt.savefig("filter_visualization.png", dpi=150)
    plt.show()


def visualize_feature_maps(model: torch.nn.Module,
                            input_tensor: torch.Tensor,
                            layer_names: list[str]):
    """中間層の特徴マップを可視化"""
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook

    # フックを登録
    handles = []
    for name, module in model.named_modules():
        if name in layer_names:
            h = module.register_forward_hook(hook_fn(name))
            handles.append(h)

    # 前方計算
    model.eval()
    with torch.no_grad():
        model(input_tensor)

    # フック解除
    for h in handles:
        h.remove()

    # 可視化
    for name, feat in activations.items():
        feat = feat[0]  # バッチの最初の要素
        n_channels = min(feat.shape[0], 16)

        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        fig.suptitle(f"Feature Maps: {name} "
                     f"(shape={tuple(feat.shape)})", fontsize=12)

        for i in range(16):
            ax = axes[i // 8, i % 8]
            if i < n_channels:
                ax.imshow(feat[i].numpy(), cmap="viridis")
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(f"feature_map_{name.replace('.', '_')}.png", dpi=150)
        plt.show()

# 使用例
# model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# visualize_filters(model, "conv1")
# visualize_feature_maps(model, input_tensor,
#                        ["layer1.0.conv1", "layer2.0.conv1",
#                         "layer3.0.conv1", "layer4.0.conv1"])
```

### t-SNEによる特徴空間の可視化

```python
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def extract_features(model: torch.nn.Module, dataloader,
                      device: torch.device,
                      max_samples: int = 2000) -> tuple:
    """CNNの中間特徴量を抽出（最終全結合層の直前）"""
    model.eval()
    features_list = []
    labels_list = []

    # 特徴抽出用のフック
    features_out = []
    def hook_fn(module, input, output):
        features_out.append(input[0].detach().cpu())

    # 最終FC層にフックを登録
    if hasattr(model, 'fc'):
        handle = model.fc.register_forward_hook(hook_fn)
    elif hasattr(model, 'classifier'):
        handle = model.classifier.register_forward_hook(hook_fn)

    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            if total >= max_samples:
                break
            images = images.to(device)
            model(images)
            labels_list.append(labels.numpy())
            total += images.size(0)

    handle.remove()

    features = torch.cat(features_out, dim=0)[:max_samples].numpy()
    labels = np.concatenate(labels_list)[:max_samples]

    return features, labels


def plot_tsne(features: np.ndarray, labels: np.ndarray,
               class_names: list = None, perplexity: int = 30):
    """t-SNEで特徴空間を2Dに圧縮して可視化"""
    print(f"t-SNE実行中... (samples={features.shape[0]}, "
          f"dims={features.shape[1]})")

    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=42, n_iter=1000)
    embeddings = tsne.fit_transform(features)

    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        name = class_names[label] if class_names else str(label)
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                    label=name, alpha=0.6, s=15)

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("t-SNE Visualization of CNN Features")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig("tsne_features.png", dpi=150, bbox_inches="tight")
    plt.show()

# 使用例
# features, labels = extract_features(model, test_loader, device)
# plot_tsne(features, labels, class_names=["cat", "dog", "bird", ...])
```

---

## 4. 物体検出とセグメンテーション

### 物体検出の基本: アンカーベース vs アンカーフリー

```
物体検出のアプローチ:

1. Two-Stage Detector (R-CNN系)
   入力 → Backbone → RPN (候補領域提案) → ROI Pooling → 分類+回帰
   精度が高いが遅い (Faster R-CNN, Mask R-CNN)

2. One-Stage Detector
   入力 → Backbone → 直接分類+回帰
   高速だが精度がやや劣る (YOLO, SSD, RetinaNet)

3. Anchor-Free
   入力 → Backbone → 中心点+サイズ予測
   シンプルで高速 (CenterNet, FCOS)

     ┌──────────┐    ┌──────────┐    ┌────────────────┐
     │          │    │ Feature  │    │ Detection Head │
     │  Input   │───>│ Pyramid  │───>│ (分類 + 回帰)  │
     │  Image   │    │ Network  │    │                │
     │          │    │ (FPN)    │    │ cls: [B,A,C]   │
     └──────────┘    └──────────┘    │ box: [B,A,4]   │
                                      └────────────────┘
```

### YOLOv8を使った物体検出

```python
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class ObjectDetector:
    """YOLOv8ベースの物体検出器"""

    def __init__(self, model_name: str = "yolov8n.pt"):
        """
        モデルサイズの選択:
        - yolov8n: Nano (3.2M params) - 最速、エッジ向け
        - yolov8s: Small (11.2M) - バランス型
        - yolov8m: Medium (25.9M) - 精度重視
        - yolov8l: Large (43.7M) - 高精度
        - yolov8x: XLarge (68.2M) - 最高精度
        """
        self.model = YOLO(model_name)

    def detect(self, image_path: str, conf_threshold: float = 0.5):
        """画像からの物体検出"""
        results = self.model(image_path, conf=conf_threshold)

        detections = []
        for result in results:
            for box in result.boxes:
                detection = {
                    "class": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                }
                detections.append(detection)

        return detections

    def train_custom(self, data_yaml: str, epochs: int = 100,
                      imgsz: int = 640, batch: int = 16):
        """カスタムデータセットでの学習"""
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,  # 最終学習率 = lr0 * lrf
            warmup_epochs=3,
            augment=True,
            mosaic=1.0,        # Mosaic拡張
            mixup=0.1,         # Mixup拡張
            copy_paste=0.1,    # Copy-Paste拡張
            patience=20,       # Early stopping
            save_period=10,    # チェックポイント保存間隔
        )
        return results

    def export_onnx(self, output_path: str = "model.onnx",
                     imgsz: int = 640):
        """ONNXフォーマットでエクスポート（推論最適化）"""
        self.model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=True,     # 動的バッチサイズ
            simplify=True,    # グラフ最適化
            opset=17,
        )
        print(f"Exported to {output_path}")

# データセット設定ファイル例 (data.yaml)
DATA_YAML_TEMPLATE = """
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: car
  2: bicycle
  3: dog
  4: cat

# アノテーション形式: YOLO (クラス x_center y_center width height)
# 全て0-1に正規化された相対座標
"""

# 使用例
# detector = ObjectDetector("yolov8m.pt")
# detections = detector.detect("photo.jpg", conf_threshold=0.5)
# for d in detections:
#     print(f"{d['class']}: {d['confidence']:.2f} at {d['bbox']}")
```

### セマンティックセグメンテーション

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """U-Net: セマンティックセグメンテーションの定番アーキテクチャ

    エンコーダ-デコーダ構造 + スキップ接続
    医療画像やリモートセンシングで広く使用される
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 21,
                 base_filters: int = 64):
        super().__init__()

        # エンコーダ（ダウンサンプリングパス）
        self.enc1 = self._double_conv(in_channels, base_filters)
        self.enc2 = self._double_conv(base_filters, base_filters * 2)
        self.enc3 = self._double_conv(base_filters * 2, base_filters * 4)
        self.enc4 = self._double_conv(base_filters * 4, base_filters * 8)

        # ボトルネック
        self.bottleneck = self._double_conv(base_filters * 8,
                                             base_filters * 16)

        # デコーダ（アップサンプリングパス）
        self.up4 = nn.ConvTranspose2d(base_filters * 16,
                                       base_filters * 8, 2, stride=2)
        self.dec4 = self._double_conv(base_filters * 16, base_filters * 8)

        self.up3 = nn.ConvTranspose2d(base_filters * 8,
                                       base_filters * 4, 2, stride=2)
        self.dec3 = self._double_conv(base_filters * 8, base_filters * 4)

        self.up2 = nn.ConvTranspose2d(base_filters * 4,
                                       base_filters * 2, 2, stride=2)
        self.dec2 = self._double_conv(base_filters * 4, base_filters * 2)

        self.up1 = nn.ConvTranspose2d(base_filters * 2,
                                       base_filters, 2, stride=2)
        self.dec1 = self._double_conv(base_filters * 2, base_filters)

        # 出力層
        self.out_conv = nn.Conv2d(base_filters, num_classes, 1)

        self.pool = nn.MaxPool2d(2, 2)

    def _double_conv(self, in_ch: int, out_ch: int):
        """Conv → BN → ReLU × 2"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # エンコーダ
        e1 = self.enc1(x)          # (B, 64, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool(e3))  # (B, 512, H/8, W/8)

        # ボトルネック
        b = self.bottleneck(self.pool(e4))  # (B, 1024, H/16, W/16)

        # デコーダ + スキップ接続
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)  # (B, num_classes, H, W)


class DiceLoss(nn.Module):
    """Dice Loss: セグメンテーション用損失関数

    IoU (Intersection over Union) に基づく損失。
    クラス不均衡に強い。
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred: (B, C, H, W) - logits
        target: (B, H, W) - class indices
        """
        num_classes = pred.shape[1]

        # ソフトマックスで確率に変換
        pred_soft = F.softmax(pred, dim=1)

        # ターゲットをone-hot化
        target_onehot = F.one_hot(target.long(),
                                   num_classes).permute(0, 3, 1, 2).float()

        # クラスごとのDice係数
        intersection = (pred_soft * target_onehot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice.mean()


class CombinedSegLoss(nn.Module):
    """CrossEntropy + Dice の組み合わせ損失"""

    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        return (self.ce_weight * self.ce(pred, target) +
                self.dice_weight * self.dice(pred, target))

# モデルの確認
# unet = UNet(in_channels=3, num_classes=21)
# dummy = torch.randn(2, 3, 256, 256)
# output = unet(dummy)
# print(f"出力形状: {output.shape}")  # (2, 21, 256, 256)
# print(f"パラメータ数: {sum(p.numel() for p in unet.parameters()):,}")
```

---

## 5. モデル軽量化と推論最適化

### Knowledge Distillation（知識蒸留）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationTrainer:
    """知識蒸留: 大きなモデル（Teacher）の知識を
    小さなモデル（Student）に転移する

    損失 = alpha * KL(soft_student, soft_teacher) +
           (1 - alpha) * CE(student, hard_labels)
    """

    def __init__(self, teacher: nn.Module, student: nn.Module,
                 temperature: float = 4.0, alpha: float = 0.7,
                 device: torch.device = None):
        self.teacher = teacher.eval()
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.device = device or torch.device("cpu")

        # Teacherの重みを凍結
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distillation_loss(self, student_logits: torch.Tensor,
                           teacher_logits: torch.Tensor,
                           labels: torch.Tensor) -> torch.Tensor:
        """蒸留損失の計算"""
        T = self.temperature

        # ソフトターゲット: Temperature付きソフトマックス
        soft_student = F.log_softmax(student_logits / T, dim=1)
        soft_teacher = F.softmax(teacher_logits / T, dim=1)

        # KL Divergence (T^2でスケーリング)
        kl_loss = F.kl_div(soft_student, soft_teacher,
                            reduction="batchmean") * (T * T)

        # ハードラベル損失
        hard_loss = F.cross_entropy(student_logits, labels)

        # 混合損失
        loss = self.alpha * kl_loss + (1 - self.alpha) * hard_loss
        return loss

    def train_epoch(self, train_loader, optimizer):
        """1エポックの蒸留学習"""
        self.student.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Teacher推論（勾配不要）
            with torch.no_grad():
                teacher_logits = self.teacher(images)

            # Student推論
            student_logits = self.student(images)

            # 蒸留損失
            loss = self.distillation_loss(
                student_logits, teacher_logits, labels
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

# 使用例
# teacher = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# student = SimpleCNN(num_classes=1000)  # 軽量モデル
#
# distiller = DistillationTrainer(
#     teacher=teacher, student=student,
#     temperature=4.0, alpha=0.7, device=device
# )
#
# for epoch in range(30):
#     loss = distiller.train_epoch(train_loader, optimizer)
#     print(f"Epoch {epoch+1}: Loss={loss:.4f}")
```

### モデルの量子化

```python
import torch
import torch.quantization as quant

def quantize_model_dynamic(model: torch.nn.Module):
    """動的量子化（推論時に重みをINT8に変換）

    特徴: 精度低下が少ない、設定が簡単
    対象: Linear, LSTM層
    """
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8,
    )

    # サイズ比較
    original_size = sum(
        p.nelement() * p.element_size()
        for p in model.parameters()
    ) / 1024 / 1024

    quantized_size = sum(
        p.nelement() * p.element_size()
        for p in quantized.parameters()
    ) / 1024 / 1024

    print(f"元モデル: {original_size:.1f} MB")
    print(f"量子化後: {quantized_size:.1f} MB")
    print(f"圧縮率: {original_size / quantized_size:.1f}x")

    return quantized


def quantize_model_static(model: torch.nn.Module,
                           calibration_loader,
                           device: torch.device):
    """静的量子化（キャリブレーションデータで最適な量子化パラメータを決定）

    特徴: 動的量子化より高速、CNN向き
    手順: 準備 → キャリブレーション → 変換
    """
    model.eval()
    model.cpu()

    # 量子化設定
    model.qconfig = quant.get_default_qconfig("x86")  # or "qnnpack" for ARM

    # 準備（オブザーバーを挿入）
    quant.prepare(model, inplace=True)

    # キャリブレーション（代表的なデータで推論して統計を収集）
    print("キャリブレーション中...")
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= 100:  # 100バッチで十分
                break
            model(images)

    # 量子化変換
    quantized = quant.convert(model, inplace=False)

    return quantized


class ModelBenchmark:
    """モデルの推論速度をベンチマーク"""

    @staticmethod
    def benchmark_latency(model: torch.nn.Module,
                           input_shape: tuple = (1, 3, 224, 224),
                           num_runs: int = 100,
                           device: str = "cpu"):
        """推論レイテンシーを計測"""
        import time

        model.eval()
        model = model.to(device)
        dummy = torch.randn(*input_shape).to(device)

        # ウォームアップ
        for _ in range(10):
            with torch.no_grad():
                model(dummy)

        # 計測
        if device == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        times = np.array(times) * 1000  # ms
        print(f"Latency: {times.mean():.2f} ± {times.std():.2f} ms")
        print(f"Throughput: {1000 / times.mean():.1f} FPS")
        return times.mean()

# 使用例
# quantized = quantize_model_dynamic(model)
# ModelBenchmark.benchmark_latency(model)
# ModelBenchmark.benchmark_latency(quantized)
```

### ONNXエクスポートとTensorRTによる最適化

```python
import torch
import torch.onnx

def export_to_onnx(model: torch.nn.Module, output_path: str,
                    input_shape: tuple = (1, 3, 224, 224),
                    dynamic_axes: dict = None):
    """PyTorchモデルをONNX形式でエクスポート"""
    model.eval()
    dummy_input = torch.randn(*input_shape)

    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,  # 定数畳み込み最適化
    )

    # ONNXモデルの検証
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported to {output_path}")

    # ファイルサイズ
    import os
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Model size: {size_mb:.1f} MB")


def run_onnx_inference(onnx_path: str, input_array):
    """ONNX Runtimeで推論"""
    import onnxruntime as ort
    import numpy as np

    # セッション作成（利用可能なプロバイダを自動選択）
    providers = ort.get_available_providers()
    print(f"利用可能なプロバイダ: {providers}")

    session = ort.InferenceSession(onnx_path, providers=providers)

    # 推論
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result = session.run([output_name],
                          {input_name: input_array.astype(np.float32)})

    return result[0]

# 使用例
# export_to_onnx(model, "resnet50.onnx")
# result = run_onnx_inference("resnet50.onnx", dummy_input.numpy())
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
| ResNeXt-50 | 2017 | 50 | 25M | 77.8% | グループ畳み込み |
| SENet-154 | 2017 | 154 | 115M | 81.3% | チャネル注意機構 (SE) |
| EfficientNet-B0 | 2019 | - | 5.3M | 77.1% | 複合スケーリング |
| EfficientNet-B7 | 2019 | - | 66M | 84.3% | 複合スケーリング（最大） |
| ViT-B/16 | 2020 | 12 | 86M | 77.9% | 純粋Transformer |
| ConvNeXt-T | 2022 | - | 28M | 82.1% | モダンCNN設計 |
| ConvNeXt-L | 2022 | - | 198M | 84.3% | モダンCNN設計（大） |

### 畳み込みの種類

| 畳み込み | パラメータ数 | 計算量 | 用途 | 特徴 |
|---|---|---|---|---|
| 標準 (3x3) | C_in × C_out × 9 | O(H×W×C_in×C_out×9) | 汎用 | 全チャネル間の結合 |
| 1x1 | C_in × C_out | O(H×W×C_in×C_out) | チャネル混合 | 次元調整 |
| Depthwise | C × 9 | O(H×W×C×9) | 軽量化 | チャネルごとに独立 |
| Separable | C×9 + C×C_out | 大幅削減 | モバイル | Depth+Pointwise |
| Dilated | C_in × C_out × 9 | 同等 | セグメンテーション | 受容野の拡大 |
| Transposed | C_in × C_out × 9 | 同等 | アップサンプリング | デコーダ |
| Deformable | C_in × C_out × 9 + オフセット | やや増加 | 物体検出 | 適応的受容野 |
| Group Conv | C_in × C_out × 9 / G | 削減 | 効率化 | チャネルグループ化 |

### 用途別モデル推奨ガイド

| ユースケース | 推奨モデル | 理由 |
|---|---|---|
| エッジ/モバイル | MobileNetV3, EfficientNet-B0 | 低計算量、小メモリ |
| 一般画像分類 | ResNet-50 + 転移学習 | 安定性と汎用性のバランス |
| 高精度画像分類 | EfficientNet-B4〜B7, ConvNeXt | スケーリングされた精度 |
| 物体検出 | YOLOv8, Faster R-CNN | リアルタイム / 高精度 |
| セマンティックセグメンテーション | U-Net, DeepLabV3+ | ピクセル単位の分類 |
| インスタンスセグメンテーション | Mask R-CNN, YOLACT | 個々のオブジェクト分離 |
| 医療画像 | U-Net + Attention | 少量データ対応 |
| 画像生成 | GAN, Diffusion Model | 高品質画像合成 |
| 超解像 | ESRGAN, SwinIR | 解像度向上 |

### 正規化手法の比較

| 手法 | 正規化次元 | バッチ依存 | 主な用途 | 備考 |
|---|---|---|---|---|
| Batch Norm | (N, H, W) | あり | CNN全般 | バッチサイズが大きい場合に有効 |
| Layer Norm | (C, H, W) | なし | Transformer, NLP | バッチサイズに非依存 |
| Instance Norm | (H, W) | なし | スタイル変換 | 各サンプル・チャネル独立 |
| Group Norm | (C/G, H, W) | なし | 小バッチCNN | BNの代替、バッチサイズ非依存 |

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

### アンチパターン3: 推論時にmodel.eval()を呼ばない

```python
# BAD: 推論時にtrainモードのまま
# → BatchNormがミニバッチ統計を使い、結果が不安定になる
# → Dropoutが有効のまま、出力がランダムに変動
predictions = model(test_images)

# GOOD: 推論時は必ずevalモードに切り替え
model.eval()
with torch.no_grad():  # 勾配計算も不要
    predictions = model(test_images)

# 学習に戻る時はtrainモードに
model.train()
```

### アンチパターン4: 入力画像の正規化ミスマッチ

```python
# BAD: 事前学習済みモデルに正規化なしの入力を渡す
# ImageNet事前学習モデルは mean=[0.485, 0.456, 0.406],
# std=[0.229, 0.224, 0.225] で正規化された入力を期待する
transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),  # 0-1に変換するが、正規化していない!
])

# GOOD: 事前学習時と同じ正規化を適用
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])
```

### アンチパターン5: GPUメモリリーク

```python
# BAD: テンソルがGPUメモリに蓄積される
all_predictions = []
for images, labels in test_loader:
    images = images.to(device)
    outputs = model(images)
    all_predictions.append(outputs)  # GPUテンソルが蓄積!

# GOOD: CPU/numpyに変換してからリストに追加
all_predictions = []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        # CPUに移動してnumpyに変換
        all_predictions.append(outputs.cpu().numpy())

# さらに良い: torch.catで結合
all_outputs = []
with torch.no_grad():
    for images, _ in test_loader:
        outputs = model(images.to(device))
        all_outputs.append(outputs.cpu())
all_outputs = torch.cat(all_outputs, dim=0)
```

---

## FAQ

### Q1: CNNとViT（Vision Transformer）のどちらを使うべき？

**A:** データが少ない場合（数千〜数万枚）はCNN（特にResNet + 転移学習）が安定。大規模データ（数百万枚以上）ではViTが優位。2022年以降のConvNeXtはCNNの設計をモダンにし、ViTと同等以上の性能を達成している。実務ではResNet/EfficientNet + 転移学習が最も汎用的。

判断基準のまとめ:

| 条件 | 推奨 | 理由 |
|---|---|---|
| データ < 1万枚 | ResNet + 転移学習 | CNNの帰納的バイアスが有利 |
| データ 1万〜100万枚 | EfficientNet or ConvNeXt | 効率的なスケーリング |
| データ > 100万枚 | ViT or DeiT | 大規模データでの性能 |
| リアルタイム推論 | MobileNetV3 or YOLOv8 | レイテンシー最適化 |
| 最高精度が必要 | ConvNeXt-L or ViT-L | 最新の大規模モデル |

### Q2: バッチ正規化はなぜ効くのか？

**A:** (1) 内部共変量シフトの緩和（各層の入力分布を安定化）、(2) 正則化効果（ミニバッチの統計量によるノイズ）、(3) 学習率を大きくできる（勾配の大きさが安定）。推論時はバッチ統計量ではなく学習時の移動平均を使うため、`model.eval()` の呼び出しが必須。

注意点:
- バッチサイズが小さい（< 16）場合はGroup NormやLayer Normの方が安定
- 推論時のバッチサイズが1の場合、BNのrunning_meanとrunning_varが使われる
- 分散学習ではSyncBatchNormを使って全GPUの統計量を同期する必要がある

### Q3: GPUメモリが足りない場合の対処法は？

**A:** 以下を優先度順に試す:

1. **バッチサイズを減らす** — 最も簡単。ただし小さすぎると学習が不安定に
2. **混合精度学習（AMP）** — `torch.cuda.amp` でFP16を使用。メモリ約40%削減
3. **勾配蓄積** — 小バッチで複数回前方計算し、勾配を蓄積してから更新
4. **画像サイズ縮小** — 224→160など。精度と要相談
5. **モデル軽量化** — EfficientNet-B0やMobileNetV3に変更
6. **Gradient Checkpointing** — メモリ削減（計算時間は増加）

```python
# Gradient Checkpointing の例
from torch.utils.checkpoint import checkpoint

class MemEfficientResNet(nn.Module):
    def forward(self, x):
        # layer3, layer4だけチェックポイントする
        x = self.layer1(x)
        x = self.layer2(x)
        x = checkpoint(self.layer3, x, use_reentrant=False)
        x = checkpoint(self.layer4, x, use_reentrant=False)
        return self.fc(self.avgpool(x).flatten(1))
```

### Q4: 学習が収束しない場合のデバッグ方法は？

**A:** 以下のチェックリストを順に確認する:

1. **データの確認**: 入力画像とラベルが正しく対応しているか可視化
2. **正規化の確認**: 入力の平均/標準偏差が適切か（0付近か）
3. **学習率**: 大きすぎるとlossが振動、小さすぎると収束が遅い。lr=1e-3から開始
4. **Loss関数**: 分類ならCrossEntropy、回帰ならMSE/MAEが適切か
5. **オーバーフィッティング確認**: 小さなサブセット（1バッチ）で完全にフィットするか
6. **勾配の確認**: gradient normが0やNaNになっていないか
7. **重みの初期化**: Kaiming初期化が使われているか

```python
# デバッグ用: 勾配と重みの統計をモニタリング
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            weight_norm = param.data.norm().item()
            ratio = grad_norm / (weight_norm + 1e-8)
            if grad_norm == 0:
                print(f"[WARNING] {name}: gradient is zero!")
            elif ratio > 100:
                print(f"[WARNING] {name}: gradient/weight ratio = {ratio:.1f}")
```

### Q5: CNNの推論速度を最適化するには？

**A:** 段階的に最適化を適用する:

| 手法 | 速度向上 | 精度低下 | 実装難易度 |
|---|---|---|---|
| torch.no_grad() | 1.2-1.5x | なし | 簡単 |
| model.eval() | 1.1x | なし | 簡単 |
| TorchScript (JIT) | 1.2-1.5x | なし | 中 |
| ONNX Runtime | 1.5-2x | なし | 中 |
| TensorRT (FP16) | 2-4x | 微小 | 難 |
| INT8量子化 | 2-4x | 小〜中 | 中 |
| Knowledge Distillation | モデル依存 | 小 | 中 |
| Pruning (枝刈り) | 1.5-3x | 小〜中 | 難 |

### Q6: 自分のデータセットが小さい（数百枚）場合の対策は？

**A:** 以下の方法を組み合わせる:

1. **転移学習**: ImageNet事前学習済みモデルを使い、最終層のみ学習
2. **強力なデータ拡張**: Albumentationsで多様な変換を適用
3. **Few-shot Learning**: Siamese Network やPrototypical Networkの活用
4. **合成データ生成**: 画像生成モデル（Stable Diffusion等）でデータを増やす
5. **Self-supervised Pre-training**: MAE, SimCLR等で事前学習してからファインチューン
6. **Cross-validation**: K-fold CVで全データを有効活用
7. **Label Smoothing**: 正則化効果で過学習を抑制

```python
# Label Smoothing の例
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# target [0, 0, 1, 0, 0] → [0.02, 0.02, 0.92, 0.02, 0.02]
# 過度な自信（確信度100%）を抑制し、汎化性能を向上
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| 畳み込み | 局所的な特徴をカーネルで抽出。パラメータ共有で効率的 |
| プーリング | 空間解像度を削減し計算量と過学習を抑制 |
| 残差接続 | 深いネットワークの勾配消失を解決（ResNet） |
| 転移学習 | 事前学習済みモデルをファインチューン。少量データに有効 |
| データ拡張 | 訓練時の多様性を増やし汎化性能を向上 |
| 複合スケーリング | 幅・深さ・解像度を同時にスケーリング（EfficientNet） |
| 可視化 | Grad-CAM、フィルタ可視化でモデルの判断根拠を理解 |
| 軽量化 | 量子化、蒸留、枝刈りで推論を高速化 |
| 物体検出 | YOLO等のOne-stage、Faster R-CNN等のTwo-stageがある |
| セグメンテーション | U-Netのエンコーダ-デコーダ構造が基本 |

---

## 次に読むべきガイド

- [02-rnn-transformer.md](./02-rnn-transformer.md) — 系列データ処理のRNN/Transformer
- [../03-applied/01-computer-vision.md](../03-applied/01-computer-vision.md) — 物体検出、セグメンテーション

---

## 参考文献

1. **Kaiming He et al.** "Deep Residual Learning for Image Recognition" CVPR 2016
2. **Alexey Dosovitskiy et al.** "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" ICLR 2021
3. **Zhuang Liu et al.** "A ConvNet for the 2020s" CVPR 2022
4. **Mingxing Tan, Quoc Le** "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" ICML 2019
5. **Ramprasaath R. Selvaraju et al.** "Grad-CAM: Visual Explanations from Deep Networks" ICCV 2017
6. **Olaf Ronneberger et al.** "U-Net: Convolutional Networks for Biomedical Image Segmentation" MICCAI 2015
7. **CS231n: Convolutional Neural Networks for Visual Recognition** Stanford University — https://cs231n.stanford.edu/

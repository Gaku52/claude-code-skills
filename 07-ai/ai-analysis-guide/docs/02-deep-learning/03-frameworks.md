# フレームワーク — PyTorch、TensorFlow、JAX

> 3大深層学習フレームワークの設計思想・コード比較・選択基準を実践的に解説する

## この章で学ぶこと

1. **PyTorch** — Define-by-Run、研究での標準、エコシステム
2. **TensorFlow/Keras** — プロダクション志向、TF Serving、TFLite
3. **JAX** — 関数型パラダイム、XLA、科学計算向け高速実行
4. **ONNX** — フレームワーク間の相互運用と統一的デプロイ
5. **実務的なフレームワーク選択** — プロジェクト要件に基づく判断基準
6. **高度なトピック** — 分散学習、混合精度、プロファイリング、カスタムオペレータ

---

## 1. フレームワークの歴史と設計思想

### 深層学習フレームワークの進化

```
年表:
2002  Torch (Lua) — NYU Yann LeCun研究室
2007  Theano — モントリオール大学、シンボリック微分の先駆者
2015  TensorFlow 1.0 — Google Brain、Define-and-Run
      Keras — François Chollet、高レベルAPI
      Caffe — Berkeley、画像認識特化
2016  PyTorch 0.1 — Facebook AI Research (FAIR)
      Define-by-Run (Chainer由来)
2017  JAX初期開発 — Google Research
      MXNet — Apache、AWS推奨
2018  PyTorch 1.0 — TorchScript導入
      ONNX 1.0 — フレームワーク間相互運用
2019  TensorFlow 2.0 — Eager Execution デフォルト化
      Keras統合
2020  PyTorch Lightning 1.0 — 構造化フレームワーク
      Hugging Face Transformers急成長
2021  JAX正式リリース — Flax/Haiku安定化
      PyTorch市場シェア50%超え
2022  PyTorch 2.0 — torch.compile (TorchDynamo)
2023  研究論文の90%以上がPyTorch
      JAXでGemini開発
2024  PyTorch 2.x — コンパイラ最適化成熟
      TensorFlow → Keras 3.0（マルチバックエンド）
```

### パラダイムの比較

```
PyTorch (Define-by-Run / Eager):
  ┌────────────────────────────────────────────┐
  │  Pythonコード = 計算グラフ                  │
  │  1行ずつ即座に実行                         │
  │  デバッグが容易（pdb使用可）               │
  │  動的な制御フロー（if/for）                │
  │  torch.compile で後からJIT最適化           │
  │  autograd による自動微分                    │
  └────────────────────────────────────────────┘

TensorFlow 2.x (Eager + @tf.function):
  ┌────────────────────────────────────────────┐
  │  デフォルトはEager Execution                │
  │  @tf.function で静的グラフ化               │
  │  AutoGraph: Python制御フローをグラフに変換  │
  │  SavedModelで本番デプロイ                  │
  │  TFLite, TF.js でマルチ環境               │
  │  tf.data による高性能データパイプライン     │
  └────────────────────────────────────────────┘

JAX (関数変換):
  ┌────────────────────────────────────────────┐
  │  NumPy互換API + 関数変換                    │
  │  jit: XLAコンパイルで高速化                │
  │  grad: 自動微分（任意次数）                │
  │  vmap: ベクトル化（自動バッチ化）          │
  │  pmap: マルチデバイス並列                  │
  │  関数型プログラミング（副作用なし）         │
  │  Pytreeで任意のデータ構造を微分可能に       │
  └────────────────────────────────────────────┘
```

### 設計哲学の根本的違い

```python
# === PyTorch: オブジェクト指向 + 命令型 ===
# 状態をオブジェクトに保持する
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 5)  # 状態（重み）をオブジェクトに保持

    def forward(self, x):
        return self.layer(x)  # selfを通じて状態にアクセス

model = Model()
# model.parameters() で全パラメータにアクセス可能
# model.state_dict() でシリアライズ

# === TensorFlow/Keras: 宣言型 + オブジェクト指向 ===
# レイヤーの宣言的な組み立て
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_shape=(10,))
])
# model.compile() で学習設定を宣言
# model.fit() で学習を一括実行

# === JAX: 関数型 ===
# 状態（パラメータ）と関数を分離
def model_fn(params, x):
    return jnp.dot(x, params['w']) + params['b']

# パラメータは外部で管理、関数は純粋（副作用なし）
params = {'w': jnp.ones((10, 5)), 'b': jnp.zeros(5)}
output = model_fn(params, x)  # 同じ入力 → 常に同じ出力
```

### コード例1: 同じモデルを3フレームワークで実装

```python
# ===== PyTorch =====
import torch
import torch.nn as nn

class PyTorchMLP(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x):
        return self.net(x)

# 学習
model = PyTorchMLP(784, 256, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
```

```python
# ===== TensorFlow / Keras =====
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu", input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, epochs=10, batch_size=128,
          validation_split=0.1, callbacks=[
              tf.keras.callbacks.EarlyStopping(patience=3),
              tf.keras.callbacks.ReduceLROnPlateau(),
          ])
```

```python
# ===== JAX + Flax =====
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

class JaxMLP(nn.Module):
    hidden: int
    out_features: int

    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.2, deterministic=not training)(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.2, deterministic=not training)(x)
        x = nn.Dense(self.out_features)(x)
        return x

model = JaxMLP(hidden=256, out_features=10)
key = jax.random.PRNGKey(42)
params = model.init(key, jnp.ones((1, 784)))

tx = optax.adam(learning_rate=0.001)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(params, batch["image"], training=True,
                                 rngs={"dropout": jax.random.PRNGKey(0)})
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, batch["label"]).mean()

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state
```

---

## 2. PyTorch エコシステム詳解

### PyTorch コアコンセプト

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# === テンソル操作の基本 ===

# テンソルの作成と演算
x = torch.randn(3, 4, requires_grad=True)  # 勾配計算を有効化
y = torch.randn(4, 5)
z = torch.matmul(x, y)  # 行列積

# GPU転送
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_gpu = x.to(device)

# テンソルのメモリレイアウト
print(f"ストライド: {x.stride()}")      # (4, 1) — 行優先
print(f"連続性: {x.is_contiguous()}")   # True
print(f"データ型: {x.dtype}")            # torch.float32
print(f"デバイス: {x.device}")           # cpu or cuda:0

# === autograd の仕組み ===

# 計算グラフの構築と逆伝播
a = torch.tensor([2.0, 3.0], requires_grad=True)
b = torch.tensor([4.0, 5.0], requires_grad=True)
c = a * b          # 要素積
d = c.sum()         # スカラーに集約
d.backward()        # 逆伝播

print(f"a.grad = {a.grad}")  # tensor([4., 5.]) = b
print(f"b.grad = {b.grad}")  # tensor([2., 3.]) = a

# 勾配計算の制御
with torch.no_grad():
    # この中では計算グラフを作成しない（推論時・パラメータ更新時）
    result = a * 2

# 勾配の累積とリセット
a.grad.zero_()  # 勾配をゼロに戻す（重要！）
```

### カスタム Dataset と DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import os

class CustomImageDataset(Dataset):
    """実務的なカスタムデータセット"""

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # ディレクトリ構造からラベルを構築
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            self.class_to_idx[class_name] = idx
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((
                        os.path.join(class_dir, fname), idx
                    ))

        print(f"[{split}] {len(self.samples)} samples, "
              f"{len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_weights(self):
        """クラス不均衡対策用の重み計算"""
        from collections import Counter
        label_counts = Counter(label for _, label in self.samples)
        total = len(self.samples)
        weights = {cls: total / count for cls, count in label_counts.items()}
        sample_weights = [weights[label] for _, label in self.samples]
        return sample_weights


# データ拡張パイプライン
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# DataLoader with WeightedRandomSampler（クラス不均衡対策）
dataset = CustomImageDataset("./data/train", split="train",
                             transform=train_transform)
sample_weights = dataset.get_class_weights()
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,         # shuffleの代わりにsampler
    num_workers=4,           # 並列データ読み込み
    pin_memory=True,         # GPU転送の高速化
    prefetch_factor=2,       # 先読みバッチ数
    persistent_workers=True, # ワーカープロセスを再利用
    drop_last=True,          # 最後の不完全バッチを除外
)
```

### コード例2: PyTorch Lightning による構造化

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics import Accuracy, F1Score, AUROC

class LitClassifier(pl.LightningModule):
    """PyTorch Lightningで構造化されたモデル"""

    def __init__(self, input_dim=784, hidden_dim=256,
                 num_classes=10, lr=0.001, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # メトリクス
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes,
                              average="macro")

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr * 10,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

# 使用例
model = LitClassifier()
trainer = pl.Trainer(
    max_epochs=10,
    accelerator="auto",
    precision="16-mixed",    # 混合精度学習
    gradient_clip_val=1.0,   # 勾配クリッピング
    accumulate_grad_batches=4,  # 勾配累積（実効バッチサイズ4倍）
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=3),
        pl.callbacks.ModelCheckpoint(
            monitor="val_acc", mode="max",
            filename="{epoch}-{val_acc:.3f}",
            save_top_k=3,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.RichProgressBar(),
    ],
    logger=[
        pl.loggers.TensorBoardLogger("logs/", name="experiment"),
        # pl.loggers.WandbLogger(project="my-project"),
    ],
)
# trainer.fit(model, train_dataloader, val_dataloader)
```

### PyTorch 2.x: torch.compile

```python
import torch
import torch.nn as nn
import time

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_ff=2048):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Pre-Norm Transformer
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = self.dropout(x) + residual

        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x) + residual
        return x

model = TransformerBlock().cuda()

# === torch.compile による最適化 ===
# PyTorch 2.0+ の新機能: TorchDynamo + TorchInductor
compiled_model = torch.compile(
    model,
    mode="reduce-overhead",  # "default", "reduce-overhead", "max-autotune"
    # fullgraph=True,        # グラフ全体をコンパイル（graph breakなし）
)

x = torch.randn(32, 128, 512).cuda()

# ベンチマーク
def benchmark(model, x, name, n_iter=100):
    # ウォームアップ
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_iter):
        _ = model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.3f}秒 ({elapsed/n_iter*1000:.1f}ms/iter)")

benchmark(model, x, "Eager")
benchmark(compiled_model, x, "Compiled")
# 典型的な結果: Compiled は Eager の 1.5-2x 高速
```

### PyTorch カスタムオペレータ（C++ Extension）

```python
# === custom_op.cpp ===
"""
#include <torch/extension.h>

torch::Tensor fused_gelu(torch::Tensor input) {
    // GELUの近似実装（tanh近似）
    auto x = input;
    auto cdf = 0.5 * (1.0 + torch::tanh(
        std::sqrt(2.0 / M_PI) * (x + 0.044715 * torch::pow(x, 3))
    ));
    return x * cdf;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gelu", &fused_gelu, "Fused GELU activation");
}
"""

# === Python側での使用 ===
# from torch.utils.cpp_extension import load
# custom_ops = load(name="custom_ops", sources=["custom_op.cpp"])
# output = custom_ops.fused_gelu(input_tensor)

# CUDA Extension の場合
"""
// custom_op_cuda.cu
__global__ void fused_gelu_kernel(
    const float* input, float* output, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf(
            sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)
        ));
        output[idx] = x * cdf;
    }
}
"""
```

---

## 3. TensorFlow / Keras エコシステム詳解

### コード例3: TensorFlow SavedModel と本番デプロイ

```python
import tensorflow as tf
import numpy as np

# モデル保存（TF SavedModel形式）
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(10),
])
model.build(input_shape=(None, 784))
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# SavedModel形式で保存（TF Servingに直接デプロイ可能）
tf.saved_model.save(model, "saved_model/my_model/1")

# TFLite変換（モバイル/エッジ向け）
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model/my_model/1")
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 量子化
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"TFLiteモデルサイズ: {len(tflite_model) / 1024:.1f} KB")

# TFLiteで推論
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# テスト推論
test_input = np.random.randn(1, 784).astype(np.float32)
interpreter.set_tensor(input_details[0]["index"], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]["index"])
print(f"TFLite推論結果形状: {output.shape}")
```

### Keras 3.0 マルチバックエンド

```python
# Keras 3.0 では PyTorch, JAX, TensorFlow をバックエンドとして切り替え可能
import os
os.environ["KERAS_BACKEND"] = "jax"  # "tensorflow", "torch", "jax"

import keras
from keras import layers, ops

class MultiBackendModel(keras.Model):
    """Keras 3.0 マルチバックエンド対応モデル"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = layers.Conv2D(32, 3, padding="same", activation="relu")
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(2)
        self.conv2 = layers.Conv2D(64, 3, padding="same", activation="relu")
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D(2)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation="relu")
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(num_classes)

    def call(self, x, training=False):
        x = self.pool1(self.bn1(self.conv1(x), training=training))
        x = self.pool2(self.bn2(self.conv2(x), training=training))
        x = self.flatten(x)
        x = self.dropout(self.dense1(x), training=training)
        return self.dense2(x)

model = MultiBackendModel()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
# 同じコードが TF / PyTorch / JAX のどのバックエンドでも動作
```

### tf.data による高性能データパイプライン

```python
import tensorflow as tf

def build_dataset(file_pattern, batch_size=32, is_training=True):
    """高性能なtf.dataパイプライン"""

    # TFRecordの読み込み
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    def parse_example(serialized):
        example = tf.io.parse_single_example(serialized, feature_description)
        image = tf.io.decode_jpeg(example["image"], channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, [224, 224])
        return image, example["label"]

    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        return image, label

    # パイプライン構築
    files = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)

    dataset = files.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.AUTOTUNE,  # 並列読み込み
        cycle_length=8,                        # 同時に読むファイル数
        deterministic=not is_training,
    )

    dataset = dataset.map(parse_example,
                          num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.map(augment,
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.repeat()

    dataset = (
        dataset
        .batch(batch_size, drop_remainder=is_training)
        .prefetch(tf.data.AUTOTUNE)  # GPU計算と並行してデータ準備
    )

    return dataset

# 使用例
train_ds = build_dataset("data/train-*.tfrecord", batch_size=64)
val_ds = build_dataset("data/val-*.tfrecord", batch_size=64, is_training=False)
```

### TensorFlow カスタム学習ループ

```python
import tensorflow as tf

class CustomTrainer:
    """tf.GradientTape を使ったカスタム学習ループ"""

    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy(
            name="val_accuracy")

    @tf.function  # グラフモードで高速化
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.loss_fn(y, logits)

        # 勾配計算と適用
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # 勾配クリッピング
        gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

        self.train_loss.update_state(loss)
        self.train_acc.update_state(y, logits)
        return loss

    @tf.function
    def val_step(self, x, y):
        logits = self.model(x, training=False)
        loss = self.loss_fn(y, logits)
        self.val_loss.update_state(loss)
        self.val_acc.update_state(y, logits)

    def fit(self, train_ds, val_ds, epochs):
        for epoch in range(epochs):
            # 学習
            self.train_loss.reset_state()
            self.train_acc.reset_state()
            for x_batch, y_batch in train_ds:
                self.train_step(x_batch, y_batch)

            # 検証
            self.val_loss.reset_state()
            self.val_acc.reset_state()
            for x_batch, y_batch in val_ds:
                self.val_step(x_batch, y_batch)

            print(
                f"Epoch {epoch+1}: "
                f"loss={self.train_loss.result():.4f}, "
                f"acc={self.train_acc.result():.4f}, "
                f"val_loss={self.val_loss.result():.4f}, "
                f"val_acc={self.val_acc.result():.4f}"
            )
```

### TF Serving によるモデルサービング

```python
# === モデルのバージョニングと署名 ===
import tensorflow as tf

class ServableModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.dense2 = tf.keras.layers.Dense(10)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 784], dtype=tf.float32, name="input")
    ])
    def serve(self, x):
        """サービング用のエンドポイント"""
        logits = self.dense2(self.dense1(x))
        probs = tf.nn.softmax(logits, axis=-1)
        return {
            "predictions": probs,
            "class_ids": tf.argmax(probs, axis=-1),
            "confidences": tf.reduce_max(probs, axis=-1),
        }

model = ServableModel()
model(tf.random.normal([1, 784]))  # ビルド

# 署名付きでSavedModel保存
tf.saved_model.save(
    model,
    "saved_model/classifier/1",
    signatures={"serving_default": model.serve},
)

# === Docker で TF Serving 起動 ===
"""
# docker-compose.yml
version: '3'
services:
  tf-serving:
    image: tensorflow/serving:latest
    ports:
      - "8501:8501"  # REST API
      - "8500:8500"  # gRPC
    volumes:
      - ./saved_model/classifier:/models/classifier
    environment:
      MODEL_NAME: classifier
    command: --enable_batching=true --batching_parameters_file=/models/batching.config
"""

# === Python クライアント ===
import requests
import numpy as np
import json

def predict_rest(input_data):
    """REST APIでの推論リクエスト"""
    url = "http://localhost:8501/v1/models/classifier:predict"
    payload = {
        "instances": input_data.tolist()
    }
    response = requests.post(url, json=payload)
    result = response.json()
    return result["predictions"]

# gRPCクライアント（高速）
"""
import grpc
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

channel = grpc.insecure_channel("localhost:8500")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = "classifier"
request.inputs["input"].CopyFrom(
    tf.make_tensor_proto(input_data, shape=input_data.shape)
)
response = stub.Predict(request)
"""
```

---

## 4. JAX エコシステム詳解

### コード例4: JAXによる高速科学計算

```python
import jax
import jax.numpy as jnp
from functools import partial
import time

# JAXの関数変換デモ

# 1. jit: XLAコンパイルによる高速化
@jax.jit
def matmul_jit(A, B):
    return jnp.dot(A, B)

A = jax.random.normal(jax.random.PRNGKey(0), (1000, 1000))
B = jax.random.normal(jax.random.PRNGKey(1), (1000, 1000))

# ウォームアップ（初回コンパイル）
_ = matmul_jit(A, B).block_until_ready()

start = time.time()
for _ in range(100):
    _ = matmul_jit(A, B).block_until_ready()
print(f"JIT行列積 (100回): {time.time()-start:.3f}秒")

# 2. grad: 自動微分
def loss_fn(params, x, y):
    pred = jnp.dot(x, params["w"]) + params["b"]
    return jnp.mean((pred - y) ** 2)

# 勾配関数を自動生成
grad_fn = jax.grad(loss_fn)

params = {"w": jnp.ones(10), "b": 0.0}
x = jax.random.normal(jax.random.PRNGKey(2), (100, 10))
y = jax.random.normal(jax.random.PRNGKey(3), (100,))

grads = grad_fn(params, x, y)
print(f"w勾配形状: {grads['w'].shape}, b勾配: {grads['b']:.4f}")

# 3. vmap: バッチ化の自動ベクトル化
def single_sample_loss(param, x, y):
    pred = jnp.dot(x, param)
    return (pred - y) ** 2

# 単一サンプルの関数をバッチに自動拡張
batched_loss = jax.vmap(single_sample_loss, in_axes=(None, 0, 0))
losses = batched_loss(jnp.ones(10), x, y)
print(f"バッチ損失形状: {losses.shape}")  # (100,)

# 4. 高次微分（ヘシアン行列）
def scalar_loss(params):
    return jnp.sum(params ** 2 + jnp.sin(params))

# 1階微分
first_grad = jax.grad(scalar_loss)
# 2階微分（ヘシアン）
hessian = jax.hessian(scalar_loss)

params = jnp.array([1.0, 2.0, 3.0])
print(f"勾配: {first_grad(params)}")
print(f"ヘシアン:\n{hessian(params)}")
```

### JAX 乱数管理（PRNG）

```python
import jax
import jax.numpy as jnp

# === JAXの乱数は「明示的」で「分割可能」===
# PyTorch/NumPyのグローバル状態とは根本的に異なる

key = jax.random.PRNGKey(42)

# BAD: 同じキーを使い回すと同じ値になる
x1 = jax.random.normal(key, (3,))
x2 = jax.random.normal(key, (3,))
print(f"同じキー: {jnp.allclose(x1, x2)}")  # True（同じ！）

# GOOD: キーを分割して使う
key, subkey1, subkey2 = jax.random.split(key, 3)
x1 = jax.random.normal(subkey1, (3,))
x2 = jax.random.normal(subkey2, (3,))
print(f"異なるキー: {jnp.allclose(x1, x2)}")  # False（異なる）

# 実用的パターン: ループ内でのキー分割
def training_loop(key, num_steps):
    params = jnp.zeros(10)
    for step in range(num_steps):
        key, step_key = jax.random.split(key)
        noise = jax.random.normal(step_key, params.shape)
        params = params + 0.01 * noise  # 例: ランダム探索
    return params

# 関数型スタイルではキーを引数として渡す
result = training_loop(jax.random.PRNGKey(0), 100)
```

### Flax による本格的なモデル実装

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
from typing import Sequence

class ResidualBlock(nn.Module):
    """Flaxによる残差ブロック"""
    features: int
    training: bool = True

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Dense(self.features)(x)
        x = nn.BatchNorm(use_running_average=not self.training)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features)(x)
        x = nn.BatchNorm(use_running_average=not self.training)(x)

        # 残差接続（次元が異なる場合は射影）
        if residual.shape[-1] != self.features:
            residual = nn.Dense(self.features)(residual)

        return nn.relu(x + residual)


class FlaxResNet(nn.Module):
    """Flaxによるカスタム ResNet風モデル"""
    num_classes: int
    hidden_dims: Sequence[int] = (128, 256, 512)

    @nn.compact
    def __call__(self, x, training: bool = True):
        for dim in self.hidden_dims:
            x = ResidualBlock(dim, training=training)(x)
            x = nn.Dropout(rate=0.1, deterministic=not training)(x)

        x = jnp.mean(x, axis=-1, keepdims=True)  # Global Average
        x = nn.Dense(self.num_classes)(x)
        return x


def create_train_state(rng, model, learning_rate, weight_decay):
    """TrainState の初期化"""
    variables = model.init(rng, jnp.ones((1, 784)), training=False)
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})

    # Optax: 学習率スケジューラ + AdamW
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=1000,
        decay_steps=50000,
        end_value=learning_rate * 0.01,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # 勾配クリッピング
        optax.adamw(schedule, weight_decay=weight_decay),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    ), batch_stats


@jax.jit
def train_step(state, batch_stats, batch, rng):
    """JITコンパイルされた学習ステップ"""
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': batch_stats}
        logits, new_model_state = state.apply_fn(
            variables,
            batch['image'],
            training=True,
            rngs={'dropout': rng},
            mutable=['batch_stats'],
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['label']
        ).mean()
        return loss, (logits, new_model_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_model_state)), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    new_batch_stats = new_model_state['batch_stats']

    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
    return state, new_batch_stats, {'loss': loss, 'accuracy': accuracy}


# チェックポイントの保存・復元
def save_checkpoint(state, batch_stats, step, ckpt_dir="checkpoints"):
    checkpoints.save_checkpoint(
        ckpt_dir,
        target={'state': state, 'batch_stats': batch_stats},
        step=step,
        keep=3,
    )

def load_checkpoint(state, batch_stats, ckpt_dir="checkpoints"):
    restored = checkpoints.restore_checkpoint(
        ckpt_dir,
        target={'state': state, 'batch_stats': batch_stats},
    )
    return restored['state'], restored['batch_stats']
```

### JAX pmap: マルチデバイス並列

```python
import jax
import jax.numpy as jnp

# マルチGPU/TPUでのデータ並列学習

@jax.pmap
def parallel_train_step(state, batch):
    """複数デバイスで並列実行される学習ステップ"""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['label']
        ).mean()

    grads = jax.grad(loss_fn)(state.params)

    # デバイス間で勾配を平均（All-Reduce）
    grads = jax.lax.pmean(grads, axis_name='batch')

    state = state.apply_gradients(grads=grads)
    return state

# 使用例
n_devices = jax.local_device_count()
print(f"利用可能デバイス数: {n_devices}")

# 状態をデバイス数分レプリケート
replicated_state = jax.device_put_replicated(state, jax.local_devices())

# バッチをデバイス数で分割
# [global_batch, ...] → [n_devices, per_device_batch, ...]
def shard_batch(batch, n_devices):
    return jax.tree.map(
        lambda x: x.reshape(n_devices, -1, *x.shape[1:]),
        batch,
    )

# sharded_batch = shard_batch(batch, n_devices)
# replicated_state = parallel_train_step(replicated_state, sharded_batch)
```

---

## 5. ONNX による相互運用

### コード例5: ONNX エクスポートと最適化

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)

model = SimpleModel()
model.eval()

# ONNX形式でエクスポート
dummy_input = torch.randn(1, 784)
torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    opset_version=17,
)
print("ONNXモデルを保存しました")

# ONNX Runtimeで推論（フレームワーク非依存）
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

test_input = np.random.randn(5, 784).astype(np.float32)
result = session.run(None, {input_name: test_input})
print(f"ONNX Runtime推論結果: {result[0].shape}")  # (5, 10)
```

### ONNX モデルの最適化と量子化

```python
import onnx
from onnxruntime.quantization import (
    quantize_dynamic, quantize_static,
    QuantType, CalibrationDataReader,
)
import onnxruntime as ort
import numpy as np

# === モデルの最適化 ===
from onnxruntime.transformers import optimizer
optimized_model = optimizer.optimize_model(
    "model.onnx",
    model_type="bert",  # "bert", "gpt2", "vit" など
    num_heads=12,
    hidden_size=768,
)
optimized_model.save_model_to_file("model_optimized.onnx")

# === 動的量子化（学習データ不要、簡単） ===
quantize_dynamic(
    model_input="model.onnx",
    model_output="model_dynamic_quant.onnx",
    weight_type=QuantType.QInt8,
)

# === 静的量子化（キャリブレーションデータ必要、高精度） ===
class CalibDataReader(CalibrationDataReader):
    def __init__(self, calibration_data):
        self.data = iter(calibration_data)

    def get_next(self):
        try:
            batch = next(self.data)
            return {"input": batch.astype(np.float32)}
        except StopIteration:
            return None

# キャリブレーションデータの準備
calib_data = [np.random.randn(1, 784) for _ in range(100)]
reader = CalibDataReader(calib_data)

quantize_static(
    model_input="model.onnx",
    model_output="model_static_quant.onnx",
    calibration_data_reader=reader,
    quant_format=ort.quantization.QuantFormat.QDQ,
)

# === パフォーマンス比較 ===
import time

def benchmark_onnx(model_path, input_data, n_iter=1000):
    """ONNXモデルのベンチマーク"""
    # 実行プロバイダの選択
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    # ウォームアップ
    for _ in range(10):
        session.run(None, {input_name: input_data})

    start = time.time()
    for _ in range(n_iter):
        session.run(None, {input_name: input_data})
    elapsed = time.time() - start

    # モデルサイズ
    import os
    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    print(f"{model_path}: {elapsed/n_iter*1000:.2f}ms/推論, "
          f"サイズ: {size_mb:.2f}MB")

input_data = np.random.randn(1, 784).astype(np.float32)
benchmark_onnx("model.onnx", input_data)
benchmark_onnx("model_dynamic_quant.onnx", input_data)
# 典型的な結果: 量子化で 2-4x 高速化、サイズ 2-4x 削減
```

### TensorFlow → ONNX 変換

```python
# tf2onnx を使った TensorFlow → ONNX 変換
import subprocess

# コマンドラインから変換
"""
python -m tf2onnx.convert \
    --saved-model saved_model/my_model/1 \
    --output model_from_tf.onnx \
    --opset 17
"""

# Python APIから変換
import tf2onnx
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu", input_shape=(784,)),
    tf.keras.layers.Dense(10),
])
model.build()

# Keras → ONNX
spec = (tf.TensorSpec((None, 784), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=17,
    output_path="model_from_keras.onnx",
)
print(f"変換完了: {len(model_proto.SerializeToString())} bytes")
```

---

## 6. 分散学習

### PyTorch DDP（DistributedDataParallel）

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os

def setup_distributed(rank, world_size):
    """分散学習の初期化"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(
        backend="nccl",  # GPU間通信に最適
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_distributed(rank, world_size, epochs=10):
    """分散学習のメイン関数"""
    setup_distributed(rank, world_size)

    # モデルをDDPでラップ
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # DistributedSamplerでデータを分割
    dataset = torch.utils.data.TensorDataset(
        torch.randn(10000, 784),
        torch.randint(0, 10, (10000,)),
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(dataset, batch_size=64, sampler=sampler,
                        pin_memory=True, num_workers=2)

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # エポックごとにシャッフル順を変更
        ddp_model.train()

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()  # 自動的にAll-Reduce
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    cleanup()

# 起動
# torchrun --nproc_per_node=4 train.py
# または
# mp.spawn(train_distributed, args=(world_size,), nprocs=world_size)
```

### PyTorch FSDP（Fully Sharded Data Parallel）

```python
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)

# FSDP: モデルパラメータをデバイス間でシャーディング
# DDP: 各デバイスがモデル全体のコピーを保持
# FSDP: 各デバイスがモデルの一部のみ保持 → メモリ効率が良い

def setup_fsdp(model, rank):
    """FSDPによる大規模モデル分散学習"""

    # 混合精度設定
    mixed_precision = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float32,
    )

    # 自動ラッピングポリシー
    auto_wrap_policy = size_based_auto_wrap_policy(
        min_num_params=1_000_000,  # 100万パラメータ以上のモジュールを分割
    )

    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        auto_wrap_policy=auto_wrap_policy,
        device_id=rank,
        use_orig_params=True,  # torch.compile互換
    )

    return fsdp_model

# DDPとFSDPの使い分け
"""
DDP を使うべきケース:
- モデルが1GPU のメモリに収まる
- 通信オーバーヘッドを最小化したい
- シンプルな実装を優先

FSDP を使うべきケース:
- モデルが1GPUのメモリに収まらない（数十億パラメータ）
- GPU数を増やしてもバッチサイズを変えたくない
- メモリ効率を最大化したい
"""
```

### TensorFlow 分散戦略

```python
import tensorflow as tf

# === MirroredStrategy: シングルノード・マルチGPU ===
strategy = tf.distribute.MirroredStrategy()
print(f"デバイス数: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu", input_shape=(784,)),
        tf.keras.layers.Dense(10),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

# model.fit() は自動的にデータ並列で実行される

# === MultiWorkerMirroredStrategy: マルチノード ===
"""
# TF_CONFIG 環境変数で設定
import json
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["worker0:2222", "worker1:2222"]
    },
    "task": {"type": "worker", "index": 0}
})

strategy = tf.distribute.MultiWorkerMirroredStrategy()
"""

# === TPUStrategy ===
"""
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)
"""
```

---

## 7. 混合精度学習（Mixed Precision Training）

### 混合精度の仕組み

```
混合精度学習の概念:

┌──────────────────────────────────────────────┐
│         Forward Pass (FP16)                   │
│  入力 ──→ [Layer1] ──→ [Layer2] ──→ 出力     │
│   FP16     FP16重み     FP16重み     FP16     │
└──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────┐
│         Loss Computation (FP32)               │
│  FP16出力 → FP32変換 → Loss計算 → Loss Scaling │
└──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────┐
│         Backward Pass (FP16)                  │
│  Scaled Loss → FP16勾配 → Unscale → FP32更新 │
└──────────────────────────────────────────────┘

メリット:
- メモリ使用量: 約50%削減（FP32→FP16）
- 計算速度: 2-3x高速化（Tensor Core活用）
- 学習精度: FP32とほぼ同等

データ型の範囲:
  FP32: ±3.4×10^38, 精度 ~7桁
  FP16: ±65,504,    精度 ~3桁
  BF16: ±3.4×10^38, 精度 ~3桁（指数部はFP32と同じ）
```

### PyTorch 混合精度

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# === 手動での混合精度 ===
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()  # Loss Scaling（勾配のアンダーフロー防止）

for epoch in range(10):
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        # autocast: 自動的にFP16/FP32を選択
        with autocast(dtype=torch.float16):
            output = model(data)
            loss = criterion(output, target)

        # GradScaler: FP16勾配をスケーリング
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # 勾配クリッピング前にunscale
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

# === BFloat16（推奨、Ampere GPU以降）===
with autocast(dtype=torch.bfloat16):
    # BF16は指数部がFP32と同じ → Loss Scaling不要
    output = model(data)
    loss = criterion(output, target)
loss.backward()
optimizer.step()

# === PyTorch Lightning の場合 ===
# trainer = pl.Trainer(precision="16-mixed")     # FP16
# trainer = pl.Trainer(precision="bf16-mixed")   # BF16
```

### TensorFlow 混合精度

```python
import tensorflow as tf

# グローバルポリシーで設定
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# モデル定義（最終層はFP32を明示）
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(10, dtype="float32"),  # 出力層はFP32
])

# Loss Scalingは自動的に適用される
optimizer = tf.keras.optimizers.Adam(0.001)
# TF2.x ではLossScaleOptimizerが自動的にラップされる

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True))
```

---

## 8. プロファイリングとデバッグ

### PyTorch Profiler

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = model.cuda()
input_data = torch.randn(64, 784).cuda()

# === 基本的なプロファイリング ===
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("model_inference"):
        output = model(input_data)

# テーブル表示
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=20
))

# Chrome Trace形式で出力（chrome://tracing で可視化）
prof.export_chrome_trace("trace.json")

# TensorBoard形式で出力
prof.export_stacks("profiler_stacks.txt", "self_cuda_time_total")

# === スケジューリングされたプロファイリング ===
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,       # 最初の1ステップは記録しない
        warmup=1,     # 次の1ステップはウォームアップ
        active=3,     # 3ステップ分を記録
        repeat=2,     # 上記を2回繰り返す
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/profiler"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step, (data, target) in enumerate(train_loader):
        if step >= (1 + 1 + 3) * 2:
            break
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()  # プロファイラに現在のステップを通知
```

### メモリプロファイリング

```python
import torch

# === GPU メモリの詳細な追跡 ===
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

print(f"初期割り当て: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

model = model.cuda()
print(f"モデル後: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

x = torch.randn(256, 784).cuda()
output = model(x)
print(f"Forward後: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

loss = output.sum()
loss.backward()
print(f"Backward後: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

print(f"ピークメモリ: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

# === メモリスナップショット（PyTorch 2.1+） ===
torch.cuda.memory._record_memory_history()
# ... 学習コード ...
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)

# memory_viz.py で可視化
# python torch/cuda/_memory_viz.py trace_plot memory_snapshot.pickle -o mem.html

# === 勾配チェックポイント（メモリ削減テクニック）===
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(512, 512), nn.ReLU())
            for _ in range(20)
        ])
        self.head = nn.Linear(512, 10)

    def forward(self, x):
        for layer in self.layers:
            # 勾配チェックポイント: Forward時の中間結果を保持しない
            # Backward時に再計算する → メモリ削減、計算コスト増
            x = checkpoint(layer, x, use_reentrant=False)
        return self.head(x)
```

### TensorBoard による可視化

```python
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

writer = SummaryWriter("runs/experiment_001")

# === スカラー値のログ ===
for epoch in range(100):
    train_loss = np.random.exponential(1.0 / (epoch + 1))
    val_loss = train_loss * 1.1
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/train", 1 - train_loss/10, epoch)

# === ヒストグラム（重みの分布） ===
for name, param in model.named_parameters():
    writer.add_histogram(f"params/{name}", param, epoch)
    if param.grad is not None:
        writer.add_histogram(f"grads/{name}", param.grad, epoch)

# === 画像のログ ===
from torchvision.utils import make_grid
images = torch.randn(16, 3, 32, 32)  # ダミー画像
grid = make_grid(images, nrow=4, normalize=True)
writer.add_image("samples", grid, epoch)

# === モデルグラフ ===
dummy_input = torch.randn(1, 784)
writer.add_graph(model.cpu(), dummy_input)

# === ハイパーパラメータ比較 ===
writer.add_hparams(
    {"lr": 0.001, "batch_size": 32, "hidden_dim": 256},
    {"hparam/accuracy": 0.95, "hparam/loss": 0.15},
)

# === Embedding可視化（t-SNE/PCA） ===
features = torch.randn(1000, 128)  # 特徴ベクトル
labels = torch.randint(0, 10, (1000,))
writer.add_embedding(features, metadata=labels.tolist(), tag="embeddings")

writer.close()
# tensorboard --logdir=runs で起動
```

---

## 比較表

### フレームワーク総合比較

| 項目 | PyTorch | TensorFlow 2.x | JAX |
|---|---|---|---|
| 設計思想 | Pythonic, 研究寄り | プロダクション寄り | 関数型, 科学計算 |
| 実行モード | Eager (デフォルト) | Eager + @tf.function | jit変換 |
| 自動微分 | autograd | GradientTape | jax.grad |
| 高レベルAPI | Lightning, HuggingFace | Keras | Flax, Haiku |
| デプロイ | TorchScript, ONNX | SavedModel, TFLite, TF.js | 要変換 |
| モバイル | PyTorch Mobile | TFLite (成熟) | 非対応 |
| TPUサポート | XLA経由 | ネイティブ | ネイティブ (最適) |
| コミュニティ | 研究者中心 | 企業+研究 | Google Research |
| 学習コスト | 低い | 中程度 | 高い |
| 2024年シェア | 約70% (研究) | 約25% | 約5% (急成長) |

### 用途別推奨フレームワーク

| ユースケース | 第一候補 | 理由 |
|---|---|---|
| 研究・論文実装 | PyTorch | 論文コードの大半がPyTorch |
| プロトタイピング | PyTorch | デバッグが容易 |
| 本番Webサービス | TensorFlow or PyTorch | TF Serving / TorchServe |
| モバイルデプロイ | TensorFlow (TFLite) | 最も成熟したモバイル推論 |
| ブラウザ実行 | TensorFlow (TF.js) | JavaScript対応 |
| 科学計算・HPC | JAX | vmap, pmap, XLA |
| 大規模言語モデル | PyTorch + HuggingFace | エコシステムの充実 |
| Kaggleコンペ | PyTorch | 柔軟なカスタマイズ |
| 教育目的 | PyTorch or Keras | 直感的なAPI |

### 高レベルライブラリ比較

| ライブラリ | 対応FW | 主な用途 | 特徴 |
|---|---|---|---|
| PyTorch Lightning | PyTorch | 学習構造化 | ボイラープレート削減、分散学習自動化 |
| Hugging Face Transformers | PyTorch, TF, JAX | NLP/Vision | 事前学習モデルのHub、統一API |
| Keras 3.0 | TF, PyTorch, JAX | 汎用 | マルチバックエンド、宣言的API |
| Optax | JAX | 最適化 | 関数型オプティマイザ合成 |
| Flax | JAX | モデル定義 | 関数型NN、Google公式 |
| timm | PyTorch | 画像モデル | 700+の事前学習済みモデル |
| TorchMetrics | PyTorch | 評価指標 | 分散学習対応メトリクス |
| Weights & Biases | 全FW | 実験管理 | ログ、可視化、ハイパラ最適化 |

### デプロイ手段比較

| 手段 | 対応FW | 対象環境 | レイテンシ | セットアップ難度 |
|---|---|---|---|---|
| TorchServe | PyTorch | サーバー | 低 | 中 |
| TF Serving | TF | サーバー | 低 | 低 |
| ONNX Runtime | 全FW | サーバー/エッジ | 最低 | 低 |
| TFLite | TF | モバイル/エッジ | 低 | 中 |
| PyTorch Mobile | PyTorch | モバイル | 中 | 中 |
| TF.js | TF | ブラウザ | 中 | 低 |
| TensorRT | PyTorch/TF | NVIDIA GPU | 最低 | 高 |
| Core ML | 全FW(変換) | iOS/macOS | 低 | 中 |
| Triton Server | 全FW | サーバー | 低 | 高 |

### スケーリング比較

| 項目 | PyTorch DDP | PyTorch FSDP | TF MirroredStrategy | JAX pmap |
|---|---|---|---|---|
| タイプ | データ並列 | シャーデッド並列 | データ並列 | データ並列 |
| メモリ効率 | 低（全パラメータ複製） | 高（パラメータ分散） | 低 | 中 |
| 通信量 | 勾配All-Reduce | パラメータ収集 | 勾配All-Reduce | 勾配All-Reduce |
| 最大モデルサイズ | 1GPUメモリ制限 | GPU数×メモリ | 1GPUメモリ制限 | 1GPUメモリ制限 |
| 設定の容易さ | 中 | やや難 | 簡単 | 簡単 |
| マルチノード | 対応 | 対応 | 対応 | 対応 |

---

## 9. 実務的なフレームワーク選択フロー

```
プロジェクト要件に基づく選択フローチャート:

[開始] → 研究/論文実装?
  │
  ├─ Yes → PyTorch（研究標準）
  │
  └─ No → モバイル/エッジデプロイ?
       │
       ├─ Yes → iOS? ──→ Core ML + coremltools
       │         │
       │         └─ Android/組み込み? ──→ TFLite（最成熟）
       │
       └─ No → ブラウザ実行?
            │
            ├─ Yes → TensorFlow.js
            │
            └─ No → 大規模並列計算（TPU/マルチGPU）?
                 │
                 ├─ Yes → モデル > 1GPU?
                 │         │
                 │         ├─ Yes → PyTorch FSDP or DeepSpeed
                 │         │
                 │         └─ No → JAX pmap（TPU最適）
                 │                 or PyTorch DDP
                 │
                 └─ No → サーバーサイドAPI?
                      │
                      ├─ Yes → ONNX Runtime（最高速）
                      │        or TorchServe / TF Serving
                      │
                      └─ No → PyTorch（汎用性最高）
```

### チーム・組織での選択基準

```python
# フレームワーク選択スコアリング関数
def score_framework(requirements):
    """プロジェクト要件からフレームワークスコアを計算"""

    weights = {
        "研究再現性": {"pytorch": 10, "tensorflow": 5, "jax": 7},
        "プロダクション": {"pytorch": 7, "tensorflow": 9, "jax": 4},
        "モバイル対応": {"pytorch": 5, "tensorflow": 10, "jax": 1},
        "学習コスト":   {"pytorch": 9, "tensorflow": 7, "jax": 3},
        "計算効率":     {"pytorch": 7, "tensorflow": 7, "jax": 10},
        "エコシステム": {"pytorch": 10, "tensorflow": 8, "jax": 5},
        "大規模学習":   {"pytorch": 8, "tensorflow": 7, "jax": 9},
        "デバッグ容易性": {"pytorch": 10, "tensorflow": 6, "jax": 4},
        "チーム既存知見": {"pytorch": 0, "tensorflow": 0, "jax": 0},
    }

    scores = {"pytorch": 0, "tensorflow": 0, "jax": 0}
    for req, importance in requirements.items():
        for fw in scores:
            scores[fw] += weights[req][fw] * importance

    return scores

# 使用例
requirements = {
    "研究再現性": 3,    # 重要度 1-5
    "プロダクション": 5,
    "モバイル対応": 2,
    "学習コスト": 4,
    "計算効率": 3,
    "エコシステム": 4,
    "大規模学習": 2,
    "デバッグ容易性": 4,
}
scores = score_framework(requirements)
for fw, score in sorted(scores.items(), key=lambda x: -x[1]):
    print(f"  {fw}: {score}")
```

---

## アンチパターン

### アンチパターン1: model.eval() を忘れる

```python
# BAD: 推論時に model.eval() を呼ばない
# → Dropout が有効のまま、BatchNorm がバッチ統計を使用
model.train()  # 学習モード
# ... 学習 ...
output = model(test_input)  # Dropout/BNが学習モードのまま!

# GOOD: 推論時は必ず eval() + no_grad()
model.eval()
with torch.no_grad():
    output = model(test_input)  # Dropout無効、BNは移動平均使用
```

### アンチパターン2: GPU/CPUデバイスの不一致

```python
# BAD: テンソルが異なるデバイスにある
model = model.cuda()
input_cpu = torch.randn(1, 784)  # CPU上
output = model(input_cpu)  # RuntimeError: expected CUDA tensor

# GOOD: デバイスを統一する
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_tensor = torch.randn(1, 784).to(device)
output = model(input_tensor)  # 同じデバイスで実行
```

### アンチパターン3: 不適切なデータ型変換

```python
# BAD: FP64テンソルをFP32モデルに渡す
x = torch.from_numpy(np.array([1.0, 2.0]))  # FP64 (NumPyデフォルト)
model = nn.Linear(2, 1)  # FP32
output = model(x)  # RuntimeError: expected Float but got Double

# GOOD: 明示的にデータ型を合わせる
x = torch.from_numpy(np.array([1.0, 2.0])).float()  # FP32に変換
output = model(x)

# さらに良い: テンソル作成時に型を指定
x = torch.tensor([1.0, 2.0], dtype=torch.float32)
```

### アンチパターン4: JAXの可変状態の誤用

```python
import jax.numpy as jnp

# BAD: JAXは副作用を許さない（NumPyスタイルの代入は不可）
x = jnp.array([1, 2, 3])
# x[0] = 10  # TypeError: JAX arrays are immutable

# GOOD: at[].set() を使う（新しい配列を返す）
x_new = x.at[0].set(10)  # x は変更されない、x_new が新しい配列
print(f"元: {x}, 新: {x_new}")  # 元: [1 2 3], 新: [10 2 3]

# BAD: グローバル変数やリストへの副作用をjit内で使う
results = []
@jax.jit
def bad_fn(x):
    results.append(x)  # 副作用! JIT内では予期しない動作
    return x * 2

# GOOD: 純粋関数として実装し、状態は外部で管理
@jax.jit
def good_fn(x, carry):
    new_carry = carry + x
    return x * 2, new_carry
```

### アンチパターン5: DataLoaderのnum_workersミスチューニング

```python
# BAD: num_workers が多すぎる → メモリ不足、プロセス生成オーバーヘッド
loader = DataLoader(dataset, batch_size=32, num_workers=64)

# BAD: num_workers=0 → データ読み込みがボトルネック
loader = DataLoader(dataset, batch_size=32, num_workers=0)

# GOOD: CPU数とGPU数に基づいて設定
import os
import torch

num_gpus = torch.cuda.device_count()
num_cpus = os.cpu_count()
# 目安: CPUコア数 / GPU数、最大でも8-16程度
optimal_workers = min(num_cpus // max(num_gpus, 1), 8)

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=optimal_workers,
    pin_memory=True,        # GPU転送高速化
    persistent_workers=True, # ワーカー再利用
    prefetch_factor=2,       # 先読みバッチ数
)
print(f"num_workers: {optimal_workers}")
```

### アンチパターン6: チェックポイント保存の不備

```python
# BAD: モデルの重みだけ保存 → 学習再開時にオプティマイザ状態が失われる
torch.save(model.state_dict(), "model.pth")

# GOOD: 学習再開に必要な全情報を保存
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict(),  # 混合精度
    "best_val_loss": best_val_loss,
    "train_loss_history": train_losses,
    "val_loss_history": val_losses,
    "config": config,  # ハイパーパラメータ
    "rng_state": torch.random.get_rng_state(),
    "cuda_rng_state": torch.cuda.get_rng_state_all(),
}
torch.save(checkpoint, f"checkpoint_epoch{epoch}.pth")

# 復元
checkpoint = torch.load("checkpoint_epoch5.pth", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
start_epoch = checkpoint["epoch"] + 1
```

---

## FAQ

### Q1: PyTorchとTensorFlowのどちらを学ぶべき？

**A:** 2024年時点では PyTorch を先に学ぶことを推奨。学術論文の90%以上がPyTorchで実装され、HuggingFace等の主要ライブラリもPyTorch中心。ただしプロダクションデプロイ（特にモバイル）ではTensorFlowが成熟している。両方の基本を知っておくのが理想。Keras 3.0の登場により、Kerasで書いてバックエンドを切り替えるアプローチも現実的になった。

### Q2: JAXはいつ使うべき？

**A:** (1) 大規模並列計算（TPU/マルチGPU）が必要な場合、(2) カスタムの微分可能アルゴリズムを実装する場合、(3) ベクトル化（vmap）が有用な科学計算。Google DeepMindの最新研究はJAXで行われることが多い。ただし学習コストは最も高い。関数型プログラミングに馴染みがない場合は、先にPyTorchを習得してからJAXに移行するのが効率的。

### Q3: モデルの本番デプロイにはどれが適している？

**A:** デプロイ先による。(1) Webサーバー → TorchServe or TF Serving、(2) モバイル → TFLite（最成熟）or PyTorch Mobile、(3) ブラウザ → TF.js、(4) フレームワーク非依存 → ONNX Runtime（推論専用、高速）。最近はONNXに変換して統一的にデプロイするパターンが増えている。レイテンシが最重要ならTensorRTやONNX Runtimeの最適化が効果的。

### Q4: torch.compile はいつ使うべき？

**A:** PyTorch 2.0以降で利用可能。(1) 学習・推論の速度を改善したい場合、(2) 既存のEagerコードを変更せずに高速化したい場合。`torch.compile(model)` を1行追加するだけで、典型的に1.5-2倍の速度向上が期待できる。ただし、動的な制御フロー（入力依存のif/for）が多いモデルではgraph breakが発生し、効果が限定的。`fullgraph=True` を試してエラーが出る箇所を確認するとよい。

### Q5: 混合精度学習は常に使うべき？

**A:** GPU（Volta世代以降、Tensor Core搭載）で学習する場合はほぼ常に使うべき。メモリ50%削減、速度2-3倍向上がほぼノーリスクで得られる。BFloat16が使えるGPU（Ampere以降）ではBF16が推奨（Loss Scaling不要で安定）。CPU学習や非常に精度が重要な科学計算（例: PDE求解）ではFP32を維持する価値がある。

### Q6: 分散学習のDDPとFSDPはどう使い分ける？

**A:** モデルが1GPUのメモリに収まる場合はDDPが簡単かつ高速。モデルが1GPUに収まらない（数十億パラメータ）場合、FSDPでパラメータをシャーディングすることでメモリ制約を回避できる。さらに大規模（数百億パラメータ以上）の場合はDeepSpeedやMegatron-LMのようなフレームワークを検討すべき。テンソル並列（モデル並列）とパイプライン並列の組み合わせが必要になることもある。

### Q7: ONNX変換で失敗するのはなぜ？

**A:** 主な原因: (1) 未サポートのオペレータ（カスタムオペレータ、特殊な関数）、(2) 動的な制御フロー（入力依存のif/for）、(3) opsetバージョンの不一致。対処法として、(a) opsetバージョンを上げる（17推奨）、(b) 問題のある演算を標準的な演算に置き換える、(c) `torch.onnx.export` の `verbose=True` でデバッグ情報を確認する。TensorFlowからの変換には `tf2onnx` を使い、`--fold_const` オプションで定数畳み込みを行うと成功率が上がる。

### Q8: 実験管理ツールは何を使うべき？

**A:** (1) **Weights & Biases (wandb)**: 最も人気、UIが優秀、チーム向け、有料プランあり。(2) **MLflow**: OSS、モデルレジストリ機能が充実、企業向け。(3) **TensorBoard**: Google製、無料、PyTorch/TF両対応、基本的な可視化。(4) **Neptune.ai**: チーム協業に強い。小規模プロジェクトならTensorBoard、チーム開発ならwandbまたはMLflowが推奨。PyTorch LightningやHugging Face Trainerは主要なロガーに標準対応している。

---

## まとめ

| 項目 | 要点 |
|---|---|
| PyTorch | Pythonic、研究標準、デバッグ容易、HuggingFaceエコシステム |
| TensorFlow | Keras高レベルAPI、本番デプロイ充実、モバイル/Web対応 |
| JAX | 関数変換（jit/grad/vmap/pmap）、科学計算、TPU最適化 |
| torch.compile | PyTorch 2.0+、1行でEagerコードを1.5-2x高速化 |
| Keras 3.0 | マルチバックエンド（TF/PyTorch/JAX）で同一コードが動作 |
| 選択基準 | 研究→PyTorch、モバイル→TF、HPC→JAX |
| 相互運用 | ONNX形式でフレームワーク間の移行が可能 |
| 分散学習 | DDP（データ並列）→ FSDP（シャーデッド並列）→ DeepSpeed |
| 混合精度 | BF16推奨（Ampere以降）、メモリ50%削減・速度2-3x向上 |
| 実験管理 | wandb/MLflow/TensorBoardで再現性とチーム協業を確保 |

---

## 次に読むべきガイド

- [../03-applied/00-nlp.md](../03-applied/00-nlp.md) — NLPの応用（Transformers活用）
- [../03-applied/02-mlops.md](../03-applied/02-mlops.md) — モデルのデプロイと運用

---

## 参考文献

1. **PyTorch Team** "PyTorch Documentation" — https://pytorch.org/docs/stable/
2. **TensorFlow Team** "TensorFlow Guide" — https://www.tensorflow.org/guide
3. **JAX Team** "JAX Reference Documentation" — https://jax.readthedocs.io/
4. **Flax Team** "Flax Documentation" — https://flax.readthedocs.io/
5. **ONNX Runtime** "ONNX Runtime Documentation" — https://onnxruntime.ai/docs/
6. **PyTorch Lightning** "Lightning Documentation" — https://lightning.ai/docs/
7. **Keras 3.0** "Keras Documentation" — https://keras.io/
8. **Weights & Biases** "Documentation" — https://docs.wandb.ai/
9. **Micikevicius et al.** "Mixed Precision Training" (2018) — https://arxiv.org/abs/1710.03740
10. **Zhao et al.** "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel" (2023) — https://arxiv.org/abs/2304.11277

# フレームワーク — PyTorch、TensorFlow、JAX

> 3大深層学習フレームワークの設計思想・コード比較・選択基準を実践的に解説する

## この章で学ぶこと

1. **PyTorch** — Define-by-Run、研究での標準、エコシステム
2. **TensorFlow/Keras** — プロダクション志向、TF Serving、TFLite
3. **JAX** — 関数型パラダイム、XLA、科学計算向け高速実行

---

## 1. フレームワークの設計思想

### パラダイムの比較

```
PyTorch (Define-by-Run / Eager):
  ┌────────────────────────────────┐
  │  Pythonコード = 計算グラフ      │
  │  1行ずつ即座に実行             │
  │  デバッグが容易（pdb使用可）   │
  │  動的な制御フロー（if/for）    │
  └────────────────────────────────┘

TensorFlow 2.x (Eager + @tf.function):
  ┌────────────────────────────────┐
  │  デフォルトはEager Execution    │
  │  @tf.function で静的グラフ化   │
  │  SavedModelで本番デプロイ      │
  │  TFLite, TF.js でマルチ環境   │
  └────────────────────────────────┘

JAX (関数変換):
  ┌────────────────────────────────┐
  │  NumPy互換API + 関数変換       │
  │  jit: XLAコンパイルで高速化    │
  │  grad: 自動微分                │
  │  vmap: ベクトル化              │
  │  pmap: マルチデバイス並列      │
  └────────────────────────────────┘
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

## 2. PyTorch エコシステム

### コード例2: PyTorch Lightning による構造化

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics import Accuracy

class LitClassifier(pl.LightningModule):
    """PyTorch Lightningで構造化されたモデル"""

    def __init__(self, input_dim=784, hidden_dim=256,
                 num_classes=10, lr=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

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
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]

# 使用例
model = LitClassifier()
trainer = pl.Trainer(
    max_epochs=10,
    accelerator="auto",
    precision="16-mixed",    # 混合精度学習
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=3),
        pl.callbacks.ModelCheckpoint(monitor="val_acc", mode="max"),
    ],
)
# trainer.fit(model, train_dataloader, val_dataloader)
```

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
```

### コード例5: ONNX による相互運用

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

---

## FAQ

### Q1: PyTorchとTensorFlowのどちらを学ぶべき？

**A:** 2024年時点では PyTorch を先に学ぶことを推奨。学術論文の90%以上がPyTorchで実装され、HuggingFace等の主要ライブラリもPyTorch中心。ただしプロダクションデプロイ（特にモバイル）ではTensorFlowが成熟している。両方の基本を知っておくのが理想。

### Q2: JAXはいつ使うべき？

**A:** (1) 大規模並列計算（TPU/マルチGPU）が必要な場合、(2) カスタムの微分可能アルゴリズムを実装する場合、(3) ベクトル化（vmap）が有用な科学計算。Google DeepMindの最新研究はJAXで行われることが多い。ただし学習コストは最も高い。

### Q3: モデルの本番デプロイにはどれが適している？

**A:** デプロイ先による。(1) Webサーバー → TorchServe or TF Serving、(2) モバイル → TFLite（最成熟）or PyTorch Mobile、(3) ブラウザ → TF.js、(4) フレームワーク非依存 → ONNX Runtime（推論専用、高速）。最近はONNXに変換して統一的にデプロイするパターンが増えている。

---

## まとめ

| 項目 | 要点 |
|---|---|
| PyTorch | Pythonic、研究標準、デバッグ容易、HuggingFaceエコシステム |
| TensorFlow | Keras高レベルAPI、本番デプロイ充実、モバイル/Web対応 |
| JAX | 関数変換（jit/grad/vmap/pmap）、科学計算、TPU最適化 |
| 選択基準 | 研究→PyTorch、モバイル→TF、HPC→JAX |
| 相互運用 | ONNX形式でフレームワーク間の移行が可能 |

---

## 次に読むべきガイド

- [../03-applied/00-nlp.md](../03-applied/00-nlp.md) — NLPの応用（Transformers活用）
- [../03-applied/02-mlops.md](../03-applied/02-mlops.md) — モデルのデプロイと運用

---

## 参考文献

1. **PyTorch Team** "PyTorch Documentation" — https://pytorch.org/docs/stable/
2. **TensorFlow Team** "TensorFlow Guide" — https://www.tensorflow.org/guide
3. **JAX Team** "JAX Reference Documentation" — https://jax.readthedocs.io/

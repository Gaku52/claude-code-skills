# クラウドAIハードウェアガイド

> TPU、Inferentia、GPU as a Serviceを活用し、スケーラブルなAIワークロードをクラウドで実行する

## この章で学ぶこと

1. **クラウドAIアクセラレータ** — Google TPU、AWS Inferentia/Trainium、NVIDIAクラウドGPUの特徴と使い分け
2. **GPU as a Service** — 主要クラウドプロバイダのAIインスタンス比較と最適な選択方法
3. **コスト最適化** — スポットインスタンス、リザーブド、サーバーレス推論の戦略

---

## 1. クラウドAIハードウェアの全体像

### クラウドAIアクセラレータの分類

```
+-----------------------------------------------------------+
|              クラウドAIアクセラレータ                        |
+-----------------------------------------------------------+
|                                                           |
|  +------------------+  +------------------+               |
|  | 汎用GPU          |  | 専用ASIC          |              |
|  | NVIDIA H100/A100 |  | Google TPU        |              |
|  | AMD MI300X       |  | AWS Inferentia    |              |
|  | 学習 + 推論      |  | AWS Trainium      |              |
|  +------------------+  +------------------+               |
|         |                      |                          |
|         v                      v                          |
|  柔軟性が高い            特定用途で高効率                   |
|  エコシステム充実        コスパが良い場合あり               |
|  移植性が高い            ベンダーロックイン                 |
+-----------------------------------------------------------+
```

### クラウドプロバイダ別AIアクセラレータ

```
+-----------------------------------------------------------+
| AWS                                                       |
|   GPU: P5 (H100), P4d (A100), G5 (A10G), G6 (L4)       |
|   ASIC: Inf2 (Inferentia2), Trn1 (Trainium)             |
+-----------------------------------------------------------+
| Google Cloud                                              |
|   GPU: A3 (H100), A2 (A100), G2 (L4)                    |
|   ASIC: TPU v5e, TPU v5p, TPU v4                        |
+-----------------------------------------------------------+
| Azure                                                     |
|   GPU: ND H100 v5, ND A100 v4, NC A100 v4               |
|   ASIC: AMD MI300X (ND MI300X v5)                        |
+-----------------------------------------------------------+
| 専業GPU クラウド                                           |
|   Lambda Labs, CoreWeave, RunPod, Vast.ai                |
|   → H100/A100 が主要クラウドより安い場合が多い            |
+-----------------------------------------------------------+
```

---

## 2. Google TPU

### TPU 世代比較表

| 世代 | 発表年 | チップ性能(BF16) | HBM | 接続 | 主な用途 |
|------|--------|-----------------|-----|------|---------|
| TPU v2 | 2017 | 45 TFLOPS | 8GB | 2D Torus | 学習入門 |
| TPU v3 | 2018 | 123 TFLOPS | 16GB | 2D Torus | 中規模学習 |
| TPU v4 | 2021 | 275 TFLOPS | 32GB | 3D Torus | 大規模学習 |
| TPU v5e | 2023 | 197 TFLOPS | 16GB | ICI | コスト効率重視 |
| TPU v5p | 2023 | 459 TFLOPS | 95GB | ICI | 最高性能学習 |

### TPU の構造

```
+-------------------------------------------------------+
|  TPU (Tensor Processing Unit)                         |
+-------------------------------------------------------+
|                                                       |
|  +------------------+  +------------------+           |
|  | MXU              |  | MXU              |           |
|  | (Matrix Multiply |  | (Matrix Multiply |           |
|  |  Unit)           |  |  Unit)           |           |
|  | 128x128 systolic |  | 128x128 systolic |           |
|  | array            |  | array            |           |
|  +------------------+  +------------------+           |
|                                                       |
|  +------------------+  +------------------+           |
|  | VPU (Vector      |  | SPU (Scalar      |           |
|  |  Processing)     |  |  Processing)     |           |
|  +------------------+  +------------------+           |
|                                                       |
|  +--------------------------------------------+      |
|  | HBM (High Bandwidth Memory) 16-95GB        |      |
|  +--------------------------------------------+      |
|                                                       |
|  TPU Pod: 最大数千チップを ICI で相互接続              |
+-------------------------------------------------------+
```

### コード例1: JAX + TPU での学習

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

# TPU デバイスの確認
print(f"デバイス: {jax.devices()}")
# [TpuDevice(id=0, ...), TpuDevice(id=1, ...), ...]

# シンプルなモデル定義
class MLP(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features)(x)
        return x

# TPU Pod 上での分散学習
@jax.pmap  # 自動的にTPUコア間でデータ並列化
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["input"])
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, batch["label"]
        ).mean()

    grads = jax.grad(loss_fn)(state.params)
    # 勾配は自動的にTPUコア間で同期
    grads = jax.lax.pmean(grads, axis_name="batch")
    state = state.apply_gradients(grads=grads)
    return state
```

### コード例2: TPU VM の起動と利用

```bash
# TPU VM の作成 (Google Cloud)
gcloud compute tpus tpu-vm create my-tpu \
    --zone=us-central1-a \
    --accelerator-type=v5litepod-8 \
    --version=tpu-ubuntu2204-base

# SSH 接続
gcloud compute tpus tpu-vm ssh my-tpu --zone=us-central1-a

# JAX の TPU 動作確認
python3 -c "import jax; print(jax.devices())"

# 学習実行
python3 train.py --tpu --batch_size=1024

# TPU VM の削除（課金停止）
gcloud compute tpus tpu-vm delete my-tpu --zone=us-central1-a
```

---

## 3. AWS Inferentia / Trainium

### コード例3: AWS Inferentia2 での推論

```python
# Neuron SDK を使った Inferentia2 での推論
import torch
import torch_neuronx

# PyTorch モデルを Neuron 形式にコンパイル
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model.eval()

# Neuron コンパイル（Inferentia2 向け最適化）
example_inputs = torch.zeros(1, 128, dtype=torch.long)
model_neuron = torch_neuronx.trace(model, example_inputs)

# コンパイル済みモデルの保存
model_neuron.save("bert_neuron.pt")

# 推論実行（Inferentia2 で高速処理）
model_neuron = torch.jit.load("bert_neuron.pt")
output = model_neuron(input_ids)
```

### コード例4: AWS Trainium での学習

```python
# Neuron SDK + PyTorch XLA での Trainium 学習
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

# Trainium デバイスの取得
device = xm.xla_device()

# モデルとオプティマイザをデバイスに配置
model = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 分散データローダー
train_loader = pl.MpDeviceLoader(train_dataloader, device)

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        xm.optimizer_step(optimizer)  # Trainium での勾配同期
```

---

## 4. GPU as a Service 比較

### 主要クラウドGPUインスタンス比較表

| プロバイダ | インスタンス | GPU | VRAM | 時間単価(USD) | 用途 |
|-----------|------------|-----|------|-------------|------|
| AWS | p5.48xlarge | 8x H100 | 640GB | $98.32 | 大規模学習 |
| AWS | p4d.24xlarge | 8x A100 | 320GB | $32.77 | 学習 |
| AWS | g5.xlarge | 1x A10G | 24GB | $1.01 | 推論 |
| AWS | inf2.xlarge | 1x Inferentia2 | 32GB | $0.76 | 推論（低コスト） |
| GCP | a3-highgpu-8g | 8x H100 | 640GB | $98.45 | 大規模学習 |
| GCP | a2-highgpu-1g | 1x A100 | 40GB | $3.67 | 学習 |
| GCP | TPU v5e-8 | 8x TPU v5e | 128GB | $12.88 | 学習（コスパ良） |
| Azure | ND H100 v5 | 8x H100 | 640GB | $98.32 | 大規模学習 |
| Lambda Labs | 1x H100 | 1x H100 | 80GB | $2.49 | 学習（安い） |
| RunPod | 1x A100 | 1x A100 | 80GB | $1.64 | 学習（安い） |

### GPU vs 専用ASIC の使い分けフロー

```
AIワークロードの種類は？
        |
        +-- 学習（Training）
        |       |
        |       +-- PyTorch/TF で柔軟に → NVIDIA GPU (H100/A100)
        |       +-- JAX + 大規模 → Google TPU
        |       +-- AWS固定 + 大規模 → Trainium
        |
        +-- 推論（Inference）
        |       |
        |       +-- 汎用・柔軟性 → NVIDIA GPU (L4/T4/A10G)
        |       +-- AWS + 低コスト → Inferentia2
        |       +-- 高スループット → TensorRT + GPU
        |
        +-- ファインチューニング
                |
                +-- 小〜中規模 → 1x A100/H100
                +-- 大規模 (70B+) → 8x H100 or TPU Pod
```

---

## 5. コスト最適化戦略

### コード例5: スポットインスタンスの活用

```python
# AWS Spot Instance での学習（70-90%割引）
import boto3

ec2 = boto3.client('ec2')

# スポットインスタンスリクエスト
response = ec2.request_spot_instances(
    SpotPrice='10.00',  # 最大入札価格
    InstanceCount=1,
    Type='one-time',
    LaunchSpecification={
        'ImageId': 'ami-xxxxx',  # Deep Learning AMI
        'InstanceType': 'p4d.24xlarge',
        'KeyName': 'my-key',
        'SecurityGroupIds': ['sg-xxxxx'],
    }
)

# チェックポイントベースの学習（中断に備える）
# train.py
class CheckpointCallback:
    def __init__(self, save_path, save_every=1000):
        self.save_path = save_path
        self.save_every = save_every

    def on_step_end(self, step, model, optimizer):
        if step % self.save_every == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{self.save_path}/checkpoint_{step}.pt")
            # S3にもバックアップ
            upload_to_s3(f"{self.save_path}/checkpoint_{step}.pt")
```

### コスト最適化手法の比較表

| 手法 | コスト削減 | リスク | 適した用途 |
|------|-----------|--------|-----------|
| Spot/Preemptible | 60-90% | 中断される可能性 | チェックポイント対応の学習 |
| Reserved (1年) | 30-40% | 契約期間の固定 | 継続利用が確実な推論 |
| Reserved (3年) | 50-60% | 長期契約リスク | 大規模推論インフラ |
| Savings Plan | 20-40% | 柔軟だが割引小 | 用途が変わりうる場合 |
| サーバーレス | 従量課金 | コールドスタート | 散発的な推論リクエスト |
| 専業クラウド | 40-60% | SLA/サポートが限定的 | 開発・研究用途 |

---

## 6. 推論サービングアーキテクチャ

### 推論サービングの構成

```
+-----------------------------------------------------------+
|  クライアント → API Gateway → Load Balancer               |
+-----------------------------------------------------------+
        |                    |                    |
        v                    v                    v
+---------------+  +---------------+  +---------------+
| 推論サーバー1  |  | 推論サーバー2  |  | 推論サーバー3  |
| GPU: T4       |  | GPU: T4       |  | GPU: T4       |
| Model: v2.1   |  | Model: v2.1   |  | Model: v2.1   |
| TensorRT      |  | TensorRT      |  | TensorRT      |
+---------------+  +---------------+  +---------------+
        |
        v
+-----------------------------------------------------------+
| NVIDIA Triton Inference Server                            |
| - モデルリポジトリ管理                                     |
| - 動的バッチング（リクエストをまとめて効率化）             |
| - マルチモデル同時サービング                               |
| - モデルバージョン管理（ブルーグリーンデプロイ）           |
+-----------------------------------------------------------+
```

---

## 7. アンチパターン

### アンチパターン1: 常時起動の高コストインスタンス

```
NG: p4d.24xlarge ($32.77/hr) を開発中も24時間起動
    → 月額 $23,594（約350万円）

OK:
    - 開発/テスト → g5.xlarge ($1.01/hr) で十分
    - 本番学習 → Spot Instance で60-90%割引
    - 推論 → オートスケーリングで需要に応じてスケール
    - 不使用時 → 自動停止スクリプトを設定
```

### アンチパターン2: ベンダーロックインへの無警戒

```
NG: TPU専用コード + JAX のみで開発
    → Google Cloud以外に移行不可能

OK:
    - PyTorch/TensorFlow ベースで開発（移植性確保）
    - ONNX 形式でモデルをエクスポート（プラットフォーム非依存）
    - Kubernetes (GKE/EKS/AKS) で統一的なデプロイ
    - コスト比較を定期的に実施し、最適なプロバイダに移行
```

---

## FAQ

### Q1. TPUとGPU、どちらを選ぶべきか？

PyTorchメインならGPU一択。JAXを使っていてGCPに固定されているならTPU v5eが非常にコスパが良い。特にTransformerベースのモデルの学習ではTPUの行列演算効率が活きる。ただしカスタムCUDAカーネルが必要な最先端研究ではGPUが必須。

### Q2. 推論のコストを最小化するには？

1) モデル量子化（INT8/INT4）でGPU利用効率を上げる、2) 動的バッチングでスループットを最大化、3) AWS InferentiaやGoogle TPU推論などASICで単価を下げる、4) SageMaker/Vertex AI のサーバーレス推論で従量課金にする。

### Q3. マルチクラウド戦略は現実的か？

学習と推論を別クラウドに分けるのは現実的。例えば学習はLambda Labs（安い）で行い、推論はAWS（SageMakerのエコシステム）で行う。ONNX形式でモデルを保存しておけば、プラットフォーム間の移動は容易。ただしデータ転送コストに注意。

---

## まとめ

| 概念 | 要点 |
|------|------|
| Google TPU | 行列演算特化ASIC、JAXとの親和性が高い |
| AWS Inferentia | 推論特化ASIC、低コスト推論 |
| AWS Trainium | 学習特化ASIC、Neuron SDK |
| H100/A100 | 汎用GPU、最大のエコシステム |
| Spot Instance | 60-90%割引、中断対策必須 |
| Triton Server | 推論サービングの業界標準 |
| 専業GPUクラウド | Lambda Labs、CoreWeave — 安価なGPU |
| ONNX | クラウド間の移植性を確保するモデル形式 |

---

## 次に読むべきガイド

- **01-computing/01-gpu-computing.md** — GPU：NVIDIA/AMD、CUDA
- **01-computing/02-edge-ai.md** — エッジAI：NPU、Coral、Jetson
- **02-emerging/03-future-hardware.md** — 未来のハードウェア：量子コンピュータ

---

## 参考文献

1. **Google Cloud — TPU ドキュメント** https://cloud.google.com/tpu/docs
2. **AWS — Neuron SDK ドキュメント** https://awsdocs-neuron.readthedocs-hosted.com/
3. **NVIDIA Triton Inference Server** https://developer.nvidia.com/triton-inference-server
4. **MLPerf Benchmark Results** https://mlcommons.org/benchmarks/

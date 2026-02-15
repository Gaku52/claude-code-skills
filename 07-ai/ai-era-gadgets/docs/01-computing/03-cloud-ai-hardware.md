# クラウドAIハードウェアガイド

> TPU、Inferentia、GPU as a Serviceを活用し、スケーラブルなAIワークロードをクラウドで実行する

## この章で学ぶこと

1. **クラウドAIアクセラレータ** — Google TPU、AWS Inferentia/Trainium、NVIDIAクラウドGPUの特徴と使い分け
2. **GPU as a Service** — 主要クラウドプロバイダのAIインスタンス比較と最適な選択方法
3. **コスト最適化** — スポットインスタンス、リザーブド、サーバーレス推論の戦略
4. **推論サービング** — Triton、vLLM、TensorRT-LLMによる本番推論基盤の構築
5. **マルチクラウド戦略** — ベンダーロックインの回避とポータブルなAIインフラの設計

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
|   Trn2 (Trainium2): 2025年〜提供開始                     |
+-----------------------------------------------------------+
| Google Cloud                                              |
|   GPU: A3 (H100), A2 (A100), G2 (L4)                    |
|   ASIC: TPU v5e, TPU v5p, TPU v4                        |
|   TPU v6e (Trillium): 2025年〜提供開始                    |
+-----------------------------------------------------------+
| Azure                                                     |
|   GPU: ND H100 v5, ND A100 v4, NC A100 v4               |
|   ASIC: AMD MI300X (ND MI300X v5)                        |
|   Maia 100: Microsoft自社AIチップ(2025年〜)               |
+-----------------------------------------------------------+
| 専業GPU クラウド                                           |
|   Lambda Labs, CoreWeave, RunPod, Vast.ai, Together AI   |
|   → H100/A100 が主要クラウドより安い場合が多い            |
+-----------------------------------------------------------+
```

### クラウドAI市場の構造

```
+-----------------------------------------------------------+
|  クラウドAIハードウェア市場の階層構造                       |
+-----------------------------------------------------------+
|                                                           |
|  Tier 1: ハイパースケーラー (AWS, GCP, Azure)              |
|  +-- 最大の品揃えとサービス統合                            |
|  +-- SLA: 99.9%+                                         |
|  +-- マネージドサービス充実 (SageMaker, Vertex AI)        |
|  +-- 高価格帯                                             |
|                                                           |
|  Tier 2: 専業GPUクラウド (CoreWeave, Lambda Labs)         |
|  +-- GPU特化で低価格                                      |
|  +-- H100/A100が豊富                                      |
|  +-- サービスはベアメタル/VM中心                           |
|  +-- SLA: 99.5-99.9%                                     |
|                                                           |
|  Tier 3: GPU マーケットプレイス (Vast.ai, RunPod)         |
|  +-- 最安値（個人GPU提供者含む）                           |
|  +-- 可用性は変動                                          |
|  +-- 開発・実験用途向け                                    |
|  +-- SLA: ベストエフォート                                |
|                                                           |
|  Tier 4: サーバーレス推論 (Replicate, Modal, Banana)      |
|  +-- 完全従量課金                                          |
|  +-- コールドスタート遅延あり                              |
|  +-- 小規模・散発的ワークロード向け                        |
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
| TPU v6e (Trillium) | 2024 | 918 TFLOPS | 32GB | ICI | 次世代学習・推論 |

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

### TPU Pod のスケーリング構造

```
+-----------------------------------------------------------+
|  TPU Pod のスケーリング                                     |
+-----------------------------------------------------------+
|                                                           |
|  TPU チップ (1個)                                          |
|    ↓                                                      |
|  TPU ボード (4チップ)                                      |
|    ↓                                                      |
|  TPU スライス (可変: 8, 16, 32, ... チップ)                |
|    ↓                                                      |
|  TPU Pod (最大数千チップ)                                   |
|                                                           |
|  例: TPU v5p Pod (8960 チップ)                             |
|  +----+  +----+  +----+  +----+                           |
|  |Chip|--|Chip|--|Chip|--|Chip|  ← ICI (Inter-Chip        |
|  +----+  +----+  +----+  +----+    Interconnect)          |
|    |       |       |       |       で直結                  |
|  +----+  +----+  +----+  +----+                           |
|  |Chip|--|Chip|--|Chip|--|Chip|  総演算: 459 TFLOPS ×     |
|  +----+  +----+  +----+  +----+  8960 = 4.1 EFLOPS       |
|    ...     ...     ...     ...                             |
|                                                           |
|  通信帯域: ICI 最大 4800 Gbps/chip                        |
|  → GPU NVLink (900 Gbps) の5倍以上                        |
+-----------------------------------------------------------+
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

### コード例3: TPU での大規模LLM学習（JAX + FSDP）

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils

# TPU Pod のデバイスメッシュを構築
# 8x4 = 32 TPU チップの場合
devices = mesh_utils.create_device_mesh((8, 4))
mesh = Mesh(devices, axis_names=('data', 'model'))

# FSDP (Fully Sharded Data Parallel) スタイルの分散
# モデルパラメータをTPUチップ間でシャーディング
def shard_params(params):
    """パラメータを 'model' 軸にシャード"""
    def shard_fn(x):
        return jax.device_put(
            x, NamedSharding(mesh, PartitionSpec('model'))
        )
    return jax.tree_map(shard_fn, params)

# データは 'data' 軸に分割
def shard_data(batch):
    """バッチデータを 'data' 軸に分割"""
    return jax.device_put(
        batch, NamedSharding(mesh, PartitionSpec('data'))
    )

# 学習ステップ（自動的にメッシュ上で分散実行）
@jax.jit
def train_step(state, batch):
    with mesh:
        def loss_fn(params):
            logits = model.apply(params, batch['input_ids'])
            labels = jax.nn.one_hot(batch['labels'], num_classes)
            return -jnp.sum(labels * jax.nn.log_softmax(logits)) / labels.shape[0]

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

# 学習ループ
for epoch in range(num_epochs):
    for batch in dataloader:
        batch = shard_data(batch)
        state, loss = train_step(state, batch)
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")
```

### コード例4: TPU Multislice 学習

```python
# TPU Multislice: 複数のTPU Podスライスを連結して超大規模学習
# Google Cloud で Multislice を構成する場合

# 1. マルチスライス TPU の作成
# gcloud compute tpus queued-resources create my-multislice \
#     --node-count=4 \
#     --accelerator-type=v5litepod-256 \
#     --runtime-version=v2-alpha-tpuv5-lite \
#     --zone=us-central1-a

import jax
from jax.experimental.multihost_utils import (
    sync_global_devices,
    process_allgather,
)

def setup_multislice():
    """Multislice TPU の初期化"""
    # 各スライスのプロセスが独立に起動
    jax.distributed.initialize()

    num_devices = jax.device_count()
    num_local = jax.local_device_count()
    num_processes = jax.process_count()
    process_id = jax.process_index()

    print(f"Process {process_id}/{num_processes}: "
          f"{num_local} local devices, {num_devices} total")

    # グローバルメッシュ（全スライスを統合）
    devices = jax.devices()
    # 4スライス × 256チップ = 1024チップのメッシュ
    mesh = Mesh(
        np.array(devices).reshape(num_processes, -1),
        axis_names=('slice', 'device')
    )
    return mesh

# Multislice の通信パターン
# スライス内: ICI (超高速、〜4800 Gbps)
# スライス間: DCN (データセンターネットワーク、〜200 Gbps)
# → 勾配同期は DCN がボトルネックになりうる
# → 非同期パイプライン並列で対策可能
```

---

## 3. AWS Inferentia / Trainium

### Inferentia と Trainium の世代比較

| 項目 | Inferentia1 | Inferentia2 | Trainium1 | Trainium2 |
|------|------------|-------------|-----------|-----------|
| 発表年 | 2019 | 2022 | 2022 | 2024 |
| NeuronCores | 4 | 2 | 2 | 2 |
| コア性能(BF16) | - | 190 TFLOPS | 190 TFLOPS | 380 TFLOPS |
| HBM | 8GB (DDR4) | 32GB (HBM2e) | 32GB (HBM2e) | 96GB (HBM3) |
| NeuronLink | なし | 対応 | 対応 | 対応(高速化) |
| 主な用途 | 推論 | 推論 | 学習 | 大規模学習 |
| EC2インスタンス | inf1 | inf2 | trn1 | trn2 |

### コード例5: AWS Inferentia2 での推論

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

### コード例6: AWS Trainium での学習

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

### コード例7: Neuron SDK での LLM 推論（transformers-neuronx）

```python
# transformers-neuronx で大規模LLMをInferentia2に展開
from transformers_neuronx import LlamaForSampling
from transformers import AutoTokenizer
import torch

# モデルのロードとコンパイル
# Llama-2-7B を Inferentia2 (inf2.48xlarge: 12チップ) に展開
model = LlamaForSampling.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    batch_size=4,             # 同時処理バッチ数
    tp_degree=12,             # テンソル並列度（12チップ使用）
    n_positions=2048,         # 最大シーケンス長
    amp='bf16',               # BFloat16精度
)

# コンパイル（初回は数分かかる）
model.to_neuron()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 推論実行
prompt = "AIハードウェアの未来について説明してください。"
input_ids = tokenizer(prompt, return_tensors='pt').input_ids

# トークン生成
with torch.no_grad():
    output_ids = model.sample(
        input_ids,
        sequence_length=512,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
    )

response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)

# パフォーマンス指標 (inf2.48xlarge)
# - Llama-2-7B: ~50 tokens/sec per request
# - Llama-2-13B: ~30 tokens/sec per request
# - コスト: inf2.48xlarge $12.98/hr vs p4d.24xlarge $32.77/hr
#   → 約60%のコスト削減（同等スループット時）
```

### コード例8: SageMaker + Inferentia2 での推論エンドポイント

```python
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

# SageMaker セッション設定
sess = sagemaker.Session()
role = sagemaker.get_execution_role()

# Inferentia2 上に HuggingFace モデルをデプロイ
hub_model = HuggingFaceModel(
    model_data=None,
    role=role,
    transformers_version="4.36.0",
    pytorch_version="2.1.0",
    py_version="py310",
    env={
        "HF_MODEL_ID": "meta-llama/Llama-2-7b-hf",
        "HF_TASK": "text-generation",
        "HF_NUM_CORES": "12",              # NeuronCores数
        "HF_AUTO_CAST_TYPE": "bf16",
        "HF_BATCH_SIZE": "4",
        "MAX_INPUT_LENGTH": "1024",
        "MAX_TOTAL_TOKENS": "2048",
    },
    image_uri=sagemaker.image_uris.retrieve(
        framework="huggingface-llm-neuron",
        region=sess.boto_region_name,
        version="0.0.23",
        instance_type="ml.inf2.48xlarge",
    ),
)

# エンドポイントのデプロイ
predictor = hub_model.deploy(
    initial_instance_count=1,
    instance_type="ml.inf2.48xlarge",
    endpoint_name="llama2-inf2-endpoint",
    model_data_download_timeout=1200,
    container_startup_health_check_timeout=1200,
)

# 推論リクエスト
response = predictor.predict({
    "inputs": "AIの未来について",
    "parameters": {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
    }
})
print(response)

# オートスケーリング設定
client = boto3.client("application-autoscaling")
client.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/{predictor.endpoint_name}/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=1,
    MaxCapacity=10,
)

# スケーリングポリシー（GPU使用率ベース）
client.put_scaling_policy(
    PolicyName="gpu-utilization-scaling",
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/{predictor.endpoint_name}/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 70.0,  # GPU使用率70%を維持
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
        },
        "ScaleInCooldown": 300,
        "ScaleOutCooldown": 60,
    },
)
```

---

## 4. GPU as a Service 比較

### 主要クラウドGPUインスタンス比較表

| プロバイダ | インスタンス | GPU | VRAM | 時間単価(USD) | 用途 |
|-----------|------------|-----|------|-------------|------|
| AWS | p5.48xlarge | 8x H100 | 640GB | $98.32 | 大規模学習 |
| AWS | p4d.24xlarge | 8x A100 | 320GB | $32.77 | 学習 |
| AWS | g5.xlarge | 1x A10G | 24GB | $1.01 | 推論 |
| AWS | g6.xlarge | 1x L4 | 24GB | $0.81 | 推論（新世代） |
| AWS | inf2.xlarge | 1x Inferentia2 | 32GB | $0.76 | 推論（低コスト） |
| GCP | a3-highgpu-8g | 8x H100 | 640GB | $98.45 | 大規模学習 |
| GCP | a2-highgpu-1g | 1x A100 | 40GB | $3.67 | 学習 |
| GCP | g2-standard-4 | 1x L4 | 24GB | $0.84 | 推論 |
| GCP | TPU v5e-8 | 8x TPU v5e | 128GB | $12.88 | 学習（コスパ良） |
| Azure | ND H100 v5 | 8x H100 | 640GB | $98.32 | 大規模学習 |
| Azure | NC A100 v4 | 1x A100 | 80GB | $3.67 | 学習 |
| Lambda Labs | 1x H100 | 1x H100 | 80GB | $2.49 | 学習（安い） |
| CoreWeave | 1x H100 | 1x H100 | 80GB | $2.06 | 学習（安い） |
| RunPod | 1x A100 | 1x A100 | 80GB | $1.64 | 学習（安い） |
| Vast.ai | 1x A100 | 1x A100 | 80GB | $0.80-1.50 | 実験（最安） |

### GPU vs 専用ASIC の使い分けフロー

```
AIワークロードの種類は？
        |
        +-- 学習（Training）
        |       |
        |       +-- PyTorch/TF で柔軟に → NVIDIA GPU (H100/A100)
        |       +-- JAX + 大規模 → Google TPU
        |       +-- AWS固定 + 大規模 → Trainium
        |       +-- 予算重視 + 中規模 → Lambda Labs / CoreWeave
        |
        +-- 推論（Inference）
        |       |
        |       +-- 汎用・柔軟性 → NVIDIA GPU (L4/T4/A10G)
        |       +-- AWS + 低コスト → Inferentia2
        |       +-- 高スループット → TensorRT + GPU
        |       +-- LLM 推論 → vLLM + H100 or Inferentia2
        |       +-- 散発的リクエスト → サーバーレス (Replicate, Modal)
        |
        +-- ファインチューニング
                |
                +-- 小〜中規模 (7B以下) → 1x A100/H100
                +-- 中規模 (7B-70B) → 4-8x A100/H100
                +-- 大規模 (70B+) → 8x H100 or TPU Pod
                +-- LoRA/QLoRA → 1x A100/RTX 4090 で十分
```

### 主要クラウド別マネージドAIサービス比較

```
+-----------------------------------------------------------+
| マネージド学習/推論サービスの比較                            |
+-----------------------------------------------------------+
|                                                           |
| AWS SageMaker                                             |
|   +-- SageMaker Training: マネージド学習ジョブ             |
|   +-- SageMaker Inference: リアルタイム/バッチ/非同期推論  |
|   +-- SageMaker JumpStart: 事前学習済みモデルの1クリック展開|
|   +-- SageMaker HyperPod: 大規模学習クラスタの自動管理     |
|                                                           |
| Google Cloud Vertex AI                                    |
|   +-- Vertex AI Training: カスタムジョブ/ハイパーチューン   |
|   +-- Vertex AI Prediction: オンライン/バッチ予測          |
|   +-- Model Garden: 事前学習済みモデルのギャラリー         |
|   +-- Vertex AI Pipelines: MLOps パイプライン              |
|                                                           |
| Azure Machine Learning                                    |
|   +-- Azure ML Training: コンピュートクラスタ管理          |
|   +-- Azure ML Endpoints: マネージドエンドポイント        |
|   +-- Azure AI Studio: 統合開発環境                       |
|   +-- Azure OpenAI Service: OpenAIモデルのホスティング     |
+-----------------------------------------------------------+
```

---

## 5. コスト最適化戦略

### コード例9: スポットインスタンスの活用

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

### コード例10: スポットインスタンス中断ハンドリング

```python
import requests
import signal
import time
import threading

class SpotInterruptionHandler:
    """
    AWSスポットインスタンスの中断通知をハンドリングし、
    安全にチェックポイントを保存する
    """

    METADATA_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"

    def __init__(self, checkpoint_fn, cleanup_fn=None):
        self.checkpoint_fn = checkpoint_fn
        self.cleanup_fn = cleanup_fn
        self.interrupted = False
        self._start_monitoring()

    def _start_monitoring(self):
        """バックグラウンドでスポット中断通知を監視"""
        def monitor():
            while not self.interrupted:
                try:
                    response = requests.get(self.METADATA_URL, timeout=2)
                    if response.status_code == 200:
                        action = response.json()
                        print(f"⚠ スポット中断通知受信: {action}")
                        print(f"  アクション: {action.get('action')}")
                        print(f"  中断時刻: {action.get('time')}")
                        self._handle_interruption()
                except requests.exceptions.RequestException:
                    pass  # 中断通知なし（正常）
                time.sleep(5)  # 5秒ごとにチェック

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

    def _handle_interruption(self):
        """中断時の処理"""
        self.interrupted = True
        print("チェックポイント保存中...")
        self.checkpoint_fn()
        if self.cleanup_fn:
            self.cleanup_fn()
        print("チェックポイント保存完了。安全に終了可能。")

# 使用例
def save_checkpoint():
    torch.save({
        'step': global_step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_loss': best_loss,
    }, '/tmp/checkpoint_latest.pt')
    # S3にアップロード
    boto3.client('s3').upload_file(
        '/tmp/checkpoint_latest.pt',
        'my-training-bucket',
        f'checkpoints/run_{run_id}/latest.pt'
    )

handler = SpotInterruptionHandler(checkpoint_fn=save_checkpoint)

# 学習ループ
for step in range(num_steps):
    if handler.interrupted:
        print("中断検出。学習を停止します。")
        break
    train_one_step()
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
| 混合精度学習 | GPU時間30-50%削減 | なし | 全学習ジョブ |
| モデル量子化 | 推論GPU 50-75%削減 | 精度低下（軽微） | 推論サービング |

### コード例11: GCP Preemptible TPU でのコスト削減

```bash
# Preemptible TPU（60-90%割引）の作成
gcloud compute tpus tpu-vm create my-preempt-tpu \
    --zone=us-central1-a \
    --accelerator-type=v5litepod-8 \
    --version=tpu-ubuntu2204-base \
    --preemptible  # プリエンプティブル（中断可能）

# コスト比較 (TPU v5e-8)
# オンデマンド: $12.88/hr → 月額 $9,274
# プリエンプティブル: $3.86/hr → 月額 $2,779（70%割引）
# Reserved (1年): $8.37/hr → 月額 $6,028（35%割引）

# 自動再起動スクリプト（プリエンプティブル中断対策）
#!/bin/bash
while true; do
    # TPUが存在するかチェック
    STATUS=$(gcloud compute tpus tpu-vm describe my-preempt-tpu \
        --zone=us-central1-a --format="get(state)" 2>/dev/null)

    if [ "$STATUS" != "READY" ]; then
        echo "TPU中断検出。再作成中..."
        gcloud compute tpus tpu-vm delete my-preempt-tpu \
            --zone=us-central1-a --quiet 2>/dev/null
        gcloud compute tpus tpu-vm create my-preempt-tpu \
            --zone=us-central1-a \
            --accelerator-type=v5litepod-8 \
            --version=tpu-ubuntu2204-base \
            --preemptible
        # チェックポイントからの学習再開
        gcloud compute tpus tpu-vm ssh my-preempt-tpu \
            --zone=us-central1-a \
            --command="cd /workspace && python train.py --resume"
    fi
    sleep 60
done
```

### コスト計算シミュレーション

```python
def calculate_training_cost(
    model_params_billion: float,
    tokens_billion: float,
    hardware: str = "h100",
    pricing: str = "on_demand",
    provider: str = "aws",
) -> dict:
    """LLM学習のクラウドコストを概算する"""

    # ハードウェア別 TFLOPS (BF16, 実効値)
    hw_specs = {
        "h100": {"tflops": 990, "gpus_per_node": 8, "price_od": 98.32, "price_spot": 29.50},
        "a100_80g": {"tflops": 312, "gpus_per_node": 8, "price_od": 32.77, "price_spot": 9.83},
        "tpu_v5e_8": {"tflops": 197*8, "gpus_per_node": 1, "price_od": 12.88, "price_spot": 3.86},
        "trainium": {"tflops": 190*16, "gpus_per_node": 1, "price_od": 21.50, "price_spot": 6.45},
    }

    spec = hw_specs[hardware]

    # Chinchilla スケーリング則: FLOPs ≈ 6 * N * D
    total_flops = 6 * model_params_billion * 1e9 * tokens_billion * 1e9

    # 実効GPU利用率 (MFU: Model FLOPs Utilization)
    mfu = 0.40  # 典型的な値: 30-50%
    effective_tflops = spec["tflops"] * spec["gpus_per_node"] * mfu * 1e12

    # 所要時間（秒）
    training_seconds = total_flops / effective_tflops
    training_hours = training_seconds / 3600

    # コスト計算
    price = spec["price_spot"] if pricing == "spot" else spec["price_od"]
    total_cost = training_hours * price

    return {
        "ハードウェア": hardware,
        "料金体系": pricing,
        "総FLOPs": f"{total_flops:.2e}",
        "所要時間": f"{training_hours:.0f} 時間 ({training_hours/24:.1f} 日)",
        "コスト": f"${total_cost:,.0f} (約{total_cost*150:,.0f}円)",
        "1ノード想定（並列化なし）": True,
    }

# 7B モデルの学習コスト比較
configs = [
    ("h100", "on_demand"), ("h100", "spot"),
    ("a100_80g", "on_demand"), ("a100_80g", "spot"),
    ("tpu_v5e_8", "on_demand"), ("tpu_v5e_8", "spot"),
]

for hw, pricing in configs:
    result = calculate_training_cost(7, 2000, hw, pricing)
    print(f"{hw} ({pricing}): {result['コスト']} / {result['所要時間']}")

# 出力例（概算）:
# h100 (on_demand): $26,373 (約395万円) / 268 時間 (11.2日)
# h100 (spot):      $7,912 (約119万円) / 268 時間
# a100_80g (on_demand): $87,651 (約1,315万円) / 2,676 時間 (111.5日)
# a100_80g (spot):   $26,295 (約394万円) / 2,676 時間
# tpu_v5e_8 (on_demand): $7,106 (約107万円) / 552 時間 (23.0日)
# tpu_v5e_8 (spot):  $2,132 (約32万円) / 552 時間
```

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

### LLM推論サービングスタック

```
+-----------------------------------------------------------+
|  LLM推論サービングの選択肢 (2025年)                        |
+-----------------------------------------------------------+
|                                                           |
|  vLLM                                                     |
|  +-- PagedAttention による効率的KVキャッシュ管理            |
|  +-- 連続バッチング (Continuous Batching)                  |
|  +-- OpenAI互換APIサーバー内蔵                             |
|  +-- Tensor Parallel対応                                   |
|  +-- AWQ/GPTQ/FP8 量子化対応                              |
|                                                           |
|  TensorRT-LLM                                             |
|  +-- NVIDIA GPU に最適化                                   |
|  +-- In-Flight Batching                                    |
|  +-- FP8/INT4 量子化                                       |
|  +-- 最高スループット（NVIDIA GPU使用時）                   |
|                                                           |
|  Text Generation Inference (TGI)                          |
|  +-- Hugging Face 公式                                     |
|  +-- Flash Attention 2 統合                                |
|  +-- Speculative Decoding                                  |
|  +-- 簡単セットアップ                                      |
|                                                           |
|  Ollama                                                    |
|  +-- ローカル/エッジ向け                                   |
|  +-- GGUF量子化モデル対応                                  |
|  +-- Docker不要、シングルバイナリ                          |
+-----------------------------------------------------------+
```

### コード例12: vLLM による LLM 推論サーバー

```python
# vLLM によるOpenAI互換推論サーバーの起動と利用

# --- サーバー起動 (CLI) ---
# pip install vllm
# python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-2-7b-hf \
#     --tensor-parallel-size 4 \
#     --gpu-memory-utilization 0.9 \
#     --max-model-len 4096 \
#     --port 8000

# --- クライアントコード ---
from openai import OpenAI

# vLLM サーバーに OpenAI 互換 API で接続
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# チャット形式の推論
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "system", "content": "あなたは親切なAIアシスタントです。"},
        {"role": "user", "content": "クラウドAIハードウェアの選び方を教えて。"},
    ],
    max_tokens=512,
    temperature=0.7,
    stream=True,  # ストリーミング応答
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### コード例13: Triton Inference Server の設定

```python
# Triton Inference Server のモデル設定と起動

# --- ディレクトリ構造 ---
# model_repository/
# ├── bert-classifier/
# │   ├── config.pbtxt
# │   ├── 1/              # バージョン1
# │   │   └── model.onnx
# │   └── 2/              # バージョン2（最新）
# │       └── model.onnx
# └── image-classifier/
#     ├── config.pbtxt
#     └── 1/
#         └── model.plan   # TensorRT エンジン

# --- config.pbtxt ---
# name: "bert-classifier"
# platform: "onnxruntime_onnx"
# max_batch_size: 64
# input [
#   {
#     name: "input_ids"
#     data_type: TYPE_INT64
#     dims: [128]
#   }
# ]
# output [
#   {
#     name: "logits"
#     data_type: TYPE_FP32
#     dims: [3]
#   }
# ]
# dynamic_batching {
#   max_queue_delay_microseconds: 100
#   preferred_batch_size: [8, 16, 32]
# }
# instance_group [
#   {
#     count: 2
#     kind: KIND_GPU
#     gpus: [0]
#   }
# ]

# --- Docker での起動 ---
# docker run --gpus=all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
#     -v $(pwd)/model_repository:/models \
#     nvcr.io/nvidia/tritonserver:24.01-py3 \
#     tritonserver --model-repository=/models

# --- Python クライアント ---
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")

# サーバーの状態確認
assert client.is_server_live()
assert client.is_model_ready("bert-classifier")

# 推論リクエスト
input_ids = np.random.randint(0, 30000, size=(1, 128)).astype(np.int64)
inputs = [httpclient.InferInput("input_ids", input_ids.shape, "INT64")]
inputs[0].set_data_from_numpy(input_ids)

outputs = [httpclient.InferRequestedOutput("logits")]

result = client.infer(
    model_name="bert-classifier",
    model_version="2",  # バージョン指定（省略で最新）
    inputs=inputs,
    outputs=outputs,
)

logits = result.as_numpy("logits")
print(f"予測結果: {np.argmax(logits)}")

# モデル統計の取得
stats = client.get_model_statistics("bert-classifier")
print(f"推論回数: {stats['model_stats'][0]['inference_count']}")
print(f"平均レイテンシ: {stats['model_stats'][0]['inference_stats']['success']['compute_infer']['avg']/1e6:.2f}ms")
```

---

## 7. Kubernetes でのAIワークロード管理

### コード例14: Kubernetes + GPU でのモデルサービング

```yaml
# Kubernetes で GPU ワークロードを管理する

# --- GPU ノードプール作成 (GKE) ---
# gcloud container node-pools create gpu-pool \
#     --cluster=ai-cluster \
#     --zone=us-central1-a \
#     --machine-type=g2-standard-8 \
#     --accelerator type=nvidia-l4,count=1 \
#     --num-nodes=3 \
#     --enable-autoscaling --min-nodes=1 --max-nodes=10

# --- vLLM Deployment ---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-inference
  labels:
    app: vllm
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - "--model"
        - "meta-llama/Llama-2-7b-hf"
        - "--tensor-parallel-size"
        - "1"
        - "--gpu-memory-utilization"
        - "0.9"
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1      # 1 GPU を要求
            memory: "32Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 180
          periodSeconds: 30
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-l4
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
# Horizontal Pod Autoscaler (GPU使用率ベース)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "70"
```

---

## 8. マルチクラウド・ハイブリッドクラウド戦略

### マルチクラウドAIアーキテクチャ

```
+-----------------------------------------------------------+
|  マルチクラウドAIアーキテクチャ                               |
+-----------------------------------------------------------+
|                                                           |
|  学習 (Training)                                          |
|  +-- Lambda Labs / CoreWeave (コスト最適)                 |
|  +-- GCP TPU (JAX大規模学習)                              |
|  +-- AWS Trainium (AWSエコシステム活用時)                  |
|      |                                                    |
|      v  モデルを ONNX / Safetensors でエクスポート         |
|      |                                                    |
|  推論 (Inference)                                         |
|  +-- AWS SageMaker (フルマネージド)                       |
|  +-- GCP Vertex AI (GCPエコシステム)                      |
|  +-- 自前 K8s + vLLM (柔軟性最大)                         |
|      |                                                    |
|      v  モデルレジストリで一元管理                          |
|      |                                                    |
|  モデルレジストリ                                          |
|  +-- MLflow Model Registry                                |
|  +-- Weights & Biases                                     |
|  +-- Hugging Face Hub (プライベートリポジトリ)             |
+-----------------------------------------------------------+
```

### コード例15: ONNX によるクラウド間モデルポータビリティ

```python
import torch
import onnx
import onnxruntime as ort

# --- PyTorch モデルを ONNX にエクスポート ---
model = MyModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)

# ONNX モデルの検証
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX モデルの検証: OK")

# --- 各クラウドでの ONNX 推論 ---

# 1. ローカル（CPU/GPU）
session = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
input_data = {"input": dummy_input.numpy()}
output = session.run(None, input_data)

# 2. AWS SageMaker にデプロイ
# sagemaker の ONNX コンテナを使用

# 3. GCP Vertex AI にデプロイ
# gcloud ai models upload --artifact-uri=gs://bucket/model.onnx

# 4. Azure ML にデプロイ
# az ml model deploy --model-path model.onnx --runtime onnx

# → 同一モデルが全クラウドで動作
```

---

## 9. セキュリティとコンプライアンス

### クラウドAI環境のセキュリティ考慮事項

```
+-----------------------------------------------------------+
|  クラウドAIワークロードのセキュリティ                        |
+-----------------------------------------------------------+
|                                                           |
|  データセキュリティ                                        |
|  +-- 学習データの暗号化（保存時 + 転送時）                 |
|  +-- VPC内のプライベートサブネットでGPU起動                |
|  +-- S3/GCS バケットのIAMポリシー厳格化                   |
|  +-- データの地理的制限（GDPR等）                          |
|                                                           |
|  モデルセキュリティ                                        |
|  +-- モデルの重みファイルの暗号化保存                       |
|  +-- モデルレジストリのアクセス制御                         |
|  +-- 推論APIの認証・認可（API Key、OAuth2）               |
|  +-- レートリミット・DDoS対策                              |
|                                                           |
|  インフラセキュリティ                                      |
|  +-- GPU VMへのSSH鍵管理                                   |
|  +-- セキュリティグループの最小権限原則                     |
|  +-- ネットワークACLによるIP制限                           |
|  +-- CloudTrail/Cloud Audit Log の有効化                   |
|                                                           |
|  コンプライアンス                                          |
|  +-- SOC 2, ISO 27001 対応クラウドの選択                   |
|  +-- HIPAA対応（医療データの場合）                         |
|  +-- 学習データの著作権・ライセンス確認                     |
+-----------------------------------------------------------+
```

---

## 10. 監視とオブザーバビリティ

### コード例16: GPU 監視ダッシュボードの構築

```python
# Prometheus + Grafana で GPU メトリクスを監視する

# --- DCGM Exporter の Kubernetes デプロイ ---
# NVIDIA DCGM (Data Center GPU Manager) でGPUメトリクスを収集

# dcgm-exporter DaemonSet
# apiVersion: apps/v1
# kind: DaemonSet
# metadata:
#   name: dcgm-exporter
# spec:
#   selector:
#     matchLabels:
#       app: dcgm-exporter
#   template:
#     spec:
#       containers:
#       - name: dcgm-exporter
#         image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.0-3.2.0-ubuntu22.04
#         ports:
#         - containerPort: 9400

# --- カスタム監視スクリプト ---
import time
import subprocess
import json
from prometheus_client import start_http_server, Gauge

# Prometheus メトリクス定義
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization', ['gpu_id'])
gpu_memory_used = Gauge('gpu_memory_used_bytes', 'GPU memory used', ['gpu_id'])
gpu_memory_total = Gauge('gpu_memory_total_bytes', 'GPU memory total', ['gpu_id'])
gpu_temperature = Gauge('gpu_temperature_celsius', 'GPU temperature', ['gpu_id'])
gpu_power_draw = Gauge('gpu_power_draw_watts', 'GPU power draw', ['gpu_id'])
inference_latency = Gauge('inference_latency_ms', 'Inference latency', ['model'])
inference_throughput = Gauge('inference_throughput_rps', 'Inference throughput', ['model'])

def collect_gpu_metrics():
    """nvidia-smi から GPU メトリクスを収集"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )

    for line in result.stdout.strip().split('\n'):
        parts = [p.strip() for p in line.split(',')]
        gpu_id = parts[0]
        gpu_utilization.labels(gpu_id=gpu_id).set(float(parts[1]))
        gpu_memory_used.labels(gpu_id=gpu_id).set(float(parts[2]) * 1e6)
        gpu_memory_total.labels(gpu_id=gpu_id).set(float(parts[3]) * 1e6)
        gpu_temperature.labels(gpu_id=gpu_id).set(float(parts[4]))
        gpu_power_draw.labels(gpu_id=gpu_id).set(float(parts[5]))

# Prometheus メトリクスサーバー起動
start_http_server(9090)
print("GPU メトリクスサーバー起動: http://localhost:9090")

while True:
    collect_gpu_metrics()
    time.sleep(10)

# --- アラートルール (Prometheus) ---
# groups:
# - name: gpu_alerts
#   rules:
#   - alert: GPUMemoryHigh
#     expr: gpu_memory_used_bytes / gpu_memory_total_bytes > 0.95
#     for: 5m
#     labels:
#       severity: warning
#     annotations:
#       summary: "GPU {{$labels.gpu_id}} のメモリ使用率が95%を超過"
#
#   - alert: GPUTemperatureHigh
#     expr: gpu_temperature_celsius > 85
#     for: 2m
#     labels:
#       severity: critical
#     annotations:
#       summary: "GPU {{$labels.gpu_id}} の温度が85°Cを超過"
```

---

## 11. アンチパターン

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

### アンチパターン3: 推論コストの無計画な増大

```
NG: 学習時と同じ大型インスタンスで推論を提供
    → 8x H100 ($98.32/hr) で推論 → 月額 $70,790

OK:
    1. モデル最適化
       - 量子化: INT8/INT4 で推論 → 必要GPU数を1/4に
       - 蒸留: 大モデル → 小モデルへ知識転移
       - プルーニング: 不要なパラメータを削除
    2. 適切なインスタンス選択
       - 推論には T4/L4/A10G で十分
       - Inferentia2 で更にコスト削減
    3. オートスケーリング
       - ゼロスケール可能な構成（Knative, KEDA）
       - 夜間/週末のスケールダウン
```

### アンチパターン4: データ転送コストの見落とし

```
NG: 学習はGCPのTPU、推論はAWSのSageMaker
    → 大規模モデル(数百GB)の転送に高額な料金

OK:
    - 転送コストを事前に計算
      AWS出力: $0.09/GB → 100GBモデル = $9
      GCP出力: $0.12/GB → 大量データは高額に
    - 学習と推論を同一クラウドに集約（可能な場合）
    - モデル圧縮後に転送
    - CDN/エッジキャッシュの活用（推論モデルの配布）
```

### アンチパターン5: GPUリソースの非効率利用

```
NG: 1リクエストずつ推論処理（GPU使用率5-10%）
    → $3.67/hr のA100を10%しか使っていない

OK:
    - 動的バッチング（Triton/vLLMの自動バッチ）
      → 複数リクエストをまとめてGPU処理
    - マルチモデルサービング
      → 1つのGPUで複数モデルを時分割実行
    - GPU共有（MIG: Multi-Instance GPU）
      → A100を最大7つの独立GPUに分割
    - リクエストキューの活用
      → 非同期処理でバッチサイズを最大化
```

---

## FAQ

### Q1. TPUとGPU、どちらを選ぶべきか？

PyTorchメインならGPU一択。JAXを使っていてGCPに固定されているならTPU v5eが非常にコスパが良い。特にTransformerベースのモデルの学習ではTPUの行列演算効率が活きる。ただしカスタムCUDAカーネルが必要な最先端研究ではGPUが必須。2025年時点ではTPU v6e（Trillium）が登場し、BF16性能が918 TFLOPSに達しており、H100の990 TFLOPSに匹敵する性能をより低コストで提供している。

### Q2. 推論のコストを最小化するには？

1) モデル量子化（INT8/INT4）でGPU利用効率を上げる、2) 動的バッチングでスループットを最大化、3) AWS InferentiaやGoogle TPU推論などASICで単価を下げる、4) SageMaker/Vertex AI のサーバーレス推論で従量課金にする。特にLLM推論ではvLLMのPagedAttentionが効果的で、KVキャッシュの効率化により同一GPUで2-4倍のスループットを実現できる。

### Q3. マルチクラウド戦略は現実的か？

学習と推論を別クラウドに分けるのは現実的。例えば学習はLambda Labs（安い）で行い、推論はAWS（SageMakerのエコシステム）で行う。ONNX形式でモデルを保存しておけば、プラットフォーム間の移動は容易。ただしデータ転送コストに注意。大規模モデル（数百GB）の転送には$10-50程度のコストがかかる。Kubernetes上でvLLMやTritonを動かす構成なら、GKE/EKS/AKSいずれでも同一のマニフェストで動作する。

### Q4. 専業GPUクラウド（Lambda Labs、CoreWeave）のリスクは？

主要クラウド（AWS、GCP、Azure）と比較して、SLAが緩い（99.5%程度）、マネージドサービスが少ない（自前でのインフラ管理が必要）、サポート体制が限定的という点がリスク。一方でH100の時間単価が$2-3/hrと主要クラウドの半額以下であり、研究・開発用途では非常にコスパが良い。本番推論には主要クラウドのマネージドサービスを使い、学習には専業クラウドを使う「ハイブリッド戦略」が現実的。

### Q5. GPUクラウドの品不足（GPU Drought）にどう対処するか？

H100/A100の需要が供給を上回る状況は2024-2025年に特に深刻だった。対策としては: 1) 複数プロバイダにアカウントを持ち在庫を確認、2) リザーブドインスタンスで確保、3) A100やL4など代替GPUの活用、4) Google TPUやAWS Trainiumなど非GPU代替の検討、5) モデル最適化（小型化、量子化）でGPU需要自体を削減する。

### Q6. オンプレミスGPUサーバーとクラウド、損益分岐点は？

A100 80GBを搭載したサーバー1台の購入費用は約$20,000-30,000。クラウドでA100を月720時間（常時稼働）使うと約$2,640/hr × 720 ≈ $2,000/月。したがって約10-15ヶ月で損益分岐する。ただしオンプレミスには電気代（$200-500/月）、冷却、保守、故障リスク、減価償却が加わるため、実質的な損益分岐は18-24ヶ月程度。GPU世代の更新サイクル（2-3年）を考えると、3年以上使う見込みがない限りクラウドが合理的。

---

## まとめ

| 概念 | 要点 |
|------|------|
| Google TPU | 行列演算特化ASIC、JAXとの親和性が高い、v6eで918 TFLOPS |
| AWS Inferentia | 推論特化ASIC、低コスト推論、SageMaker統合 |
| AWS Trainium | 学習特化ASIC、Neuron SDK、Trainium2で96GB HBM3 |
| H100/A100 | 汎用GPU、最大のエコシステム、柔軟性最高 |
| Spot Instance | 60-90%割引、中断対策（チェックポイント）必須 |
| Triton Server | 推論サービングの業界標準、動的バッチング |
| vLLM | LLM推論の事実上の標準、PagedAttention |
| 専業GPUクラウド | Lambda Labs、CoreWeave — 安価なGPU |
| ONNX | クラウド間の移植性を確保するモデル形式 |
| MIG | A100を最大7分割して効率的に利用 |
| Kubernetes | クラウド非依存のAIインフラ管理 |

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
5. **vLLM — PagedAttention** https://docs.vllm.ai/
6. **TensorRT-LLM** https://github.com/NVIDIA/TensorRT-LLM
7. **CoreWeave — GPU Cloud** https://www.coreweave.com/
8. **Lambda Labs — GPU Cloud** https://lambdalabs.com/

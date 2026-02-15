# GPU コンピューティングガイド

> NVIDIA/AMD GPU、CUDA、AI学習用GPU選びの実践知識を網羅する

## この章で学ぶこと

1. **GPU アーキテクチャ** の基本 — CPU との違い、NVIDIA/AMD の世代別特徴
2. **CUDA エコシステム** — プログラミングモデル、主要ライブラリ、ROCm との比較
3. **AI学習用GPU選び** — 用途別スペック比較、コストパフォーマンス分析
4. **VRAM管理と最適化** — 混合精度、勾配チェックポイント、DeepSpeed/FSDPの実践
5. **マルチGPU学習** — DDP、FSDP、パイプライン並列の設計と実装

---

## 1. GPU の基本アーキテクチャ

### CPU vs GPU の構造的違い

```
+----------------------------------+   +----------------------------------+
|          CPU (少数精鋭)           |   |          GPU (大量並列)           |
+----------------------------------+   +----------------------------------+
|                                  |   |                                  |
|  +------+  +------+              |   |  +--+ +--+ +--+ +--+ +--+ +--+ |
|  | Core |  | Core |  大きなコア  |   |  |SM| |SM| |SM| |SM| |SM| |SM| |
|  | (強) |  | (強) |  少数        |   |  +--+ +--+ +--+ +--+ +--+ +--+ |
|  +------+  +------+              |   |  +--+ +--+ +--+ +--+ +--+ +--+ |
|  +------+  +------+              |   |  |SM| |SM| |SM| |SM| |SM| |SM| |
|  | Core |  | Core |              |   |  +--+ +--+ +--+ +--+ +--+ +--+ |
|  | (強) |  | (強) |              |   |  +--+ +--+ +--+ +--+ +--+ +--+ |
|  +------+  +------+              |   |  |SM| |SM| |SM| |SM| |SM| |SM| |
|                                  |   |  +--+ +--+ +--+ +--+ +--+ +--+ |
|  +---------------------------+   |   |                                  |
|  |    大容量キャッシュ        |   |   |  SM = Streaming Multiprocessor  |
|  +---------------------------+   |   |  小さなコア × 数千〜数万         |
|                                  |   |                                  |
|  逐次処理に最適                  |   |  並列処理に最適                   |
|  分岐予測、投機的実行            |   |  同一命令を大量データに適用       |
+----------------------------------+   +----------------------------------+
```

### なぜGPUがAIに適しているか

```
+-----------------------------------------------------------+
|  AI計算（行列演算）とGPUの親和性                            |
+-----------------------------------------------------------+
|                                                           |
|  ニューラルネットワークの計算の本質:                        |
|                                                           |
|  Y = W × X + b                                           |
|  (出力 = 重み行列 × 入力ベクトル + バイアス)               |
|                                                           |
|  行列演算の特徴:                                           |
|  1. 各要素の計算が独立 → 並列化しやすい                    |
|  2. 同じ演算を大量データに適用 → SIMD向き                  |
|  3. メモリアクセスパターンが規則的 → バンド幅活用          |
|                                                           |
|  CPU: 8-64コア × 逐次処理                                  |
|       → 1024×1024 行列乗算: 数十ms                        |
|                                                           |
|  GPU: 数千〜数万コア × 並列処理                            |
|       → 1024×1024 行列乗算: 数百μs (100倍高速)            |
|                                                           |
|  Tensor Core: 行列演算専用ユニット                         |
|       → 1024×1024 行列乗算: 数十μs (さらに10倍高速)        |
+-----------------------------------------------------------+
```

### NVIDIA GPU 世代一覧

| 世代 | アーキテクチャ | 代表製品 | 発売年 | AI向け特徴 |
|------|--------------|---------|--------|-----------|
| Kepler | GK110 | Tesla K80 | 2014 | GPGPU黎明期 |
| Pascal | GP100 | Tesla P100, GTX 1080 | 2016 | FP16対応、NVLink |
| Volta | GV100 | Tesla V100 | 2017 | Tensor Core初搭載 |
| Turing | TU102 | RTX 2080 Ti | 2018 | RT Core、INT8推論 |
| Ampere | GA100/GA102 | A100, RTX 3090 | 2020 | TF32、構造化スパース性 |
| Hopper | GH100 | H100 | 2022 | FP8、Transformer Engine |
| Blackwell | GB100/GB202 | B200, RTX 5090 | 2024-25 | FP4、第2世代Transformer Engine |

### NVIDIA GPU 詳細スペック比較

| 項目 | RTX 4090 | A100 80GB | H100 SXM | B200 |
|------|---------|----------|----------|------|
| CUDAコア数 | 16,384 | 6,912 | 14,592 | 18,432 |
| Tensor Core | 512 (4th gen) | 432 (3rd gen) | 528 (4th gen) | 576 (5th gen) |
| VRAM | 24GB GDDR6X | 80GB HBM2e | 80GB HBM3 | 192GB HBM3e |
| メモリ帯域 | 1,008 GB/s | 2,039 GB/s | 3,350 GB/s | 8,000 GB/s |
| FP32 性能 | 82.6 TFLOPS | 19.5 TFLOPS | 66.9 TFLOPS | 90 TFLOPS |
| BF16 Tensor | 660 TFLOPS | 312 TFLOPS | 990 TFLOPS | 2,250 TFLOPS |
| FP8 Tensor | 1,321 TFLOPS | - | 1,979 TFLOPS | 4,500 TFLOPS |
| TDP | 450W | 300W | 700W | 1,000W |
| NVLink | なし | 600 GB/s | 900 GB/s | 1,800 GB/s |
| 価格帯 | ~$1,600 | ~$15,000 | ~$30,000 | ~$40,000 |

### SM (Streaming Multiprocessor) の内部構造

```
+-----------------------------------------------------------+
|  SM (Streaming Multiprocessor) — Hopper H100              |
+-----------------------------------------------------------+
|                                                           |
|  +-------+  +-------+  +-------+  +-------+              |
|  |FP32   |  |FP32   |  |FP32   |  |FP32   |  128 FP32   |
|  |Core x32| |Core x32| |Core x32| |Core x32|  CUDAコア  |
|  +-------+  +-------+  +-------+  +-------+              |
|                                                           |
|  +-------+  +-------+  +-------+  +-------+              |
|  |INT32  |  |INT32  |  |INT32  |  |INT32  |  128 INT32  |
|  |Core x32| |Core x32| |Core x32| |Core x32|  コア      |
|  +-------+  +-------+  +-------+  +-------+              |
|                                                           |
|  +--------------------------------------------------+    |
|  | Tensor Core × 4 (4th Generation)                  |    |
|  | FP64, TF32, BF16, FP16, FP8, INT8 対応            |    |
|  | 1 Tensor Core = 4×4×4 の行列演算を1クロックで実行 |    |
|  +--------------------------------------------------+    |
|                                                           |
|  +--------------------------------------------------+    |
|  | 共有メモリ / L1キャッシュ: 228 KB (設定可変)       |    |
|  +--------------------------------------------------+    |
|                                                           |
|  +--------------------------------------------------+    |
|  | Warp Scheduler × 4                                |    |
|  | 各32スレッドの Warp を管理                         |    |
|  +--------------------------------------------------+    |
|                                                           |
|  H100: 132 SM → 16,896 FP32 CUDA コア                    |
|  B200:  160 SM → 18,432 FP32 CUDA コア (推定)            |
+-----------------------------------------------------------+
```

---

## 2. CUDA エコシステム

### CUDA ソフトウェアスタック

```
+-------------------------------------------------------+
|                  アプリケーション層                      |
|  PyTorch / TensorFlow / JAX / DeepSpeed                |
+-------------------------------------------------------+
|                  ライブラリ層                            |
|  cuDNN / cuBLAS / NCCL / TensorRT / Triton             |
+-------------------------------------------------------+
|                  ランタイム層                            |
|  CUDA Runtime API / CUDA Driver API                    |
+-------------------------------------------------------+
|                  ドライバ層                              |
|  NVIDIA GPU Driver                                     |
+-------------------------------------------------------+
|                  ハードウェア層                          |
|  NVIDIA GPU (SM, Tensor Core, HBM)                     |
+-------------------------------------------------------+
```

### CUDA主要ライブラリの役割

```
+-----------------------------------------------------------+
|  CUDA ライブラリ体系                                        |
+-----------------------------------------------------------+
|                                                           |
|  cuBLAS — 線形代数（BLAS: Basic Linear Algebra Subprograms)|
|  +-- 行列乗算 (GEMM) の超高速実装                         |
|  +-- PyTorch の torch.mm() の裏で動作                     |
|  +-- FP64/FP32/FP16/BF16/FP8/INT8 対応                   |
|                                                           |
|  cuDNN — ディープラーニング基本演算                         |
|  +-- 畳み込み (Convolution) の最適化実装                   |
|  +-- BatchNorm, Pooling, Activation, RNN                  |
|  +-- 複数のアルゴリズムから最速を自動選択                   |
|                                                           |
|  NCCL — マルチGPU通信                                      |
|  +-- AllReduce, AllGather, ReduceScatter                  |
|  +-- NVLink / NVSwitch 対応                                |
|  +-- DDP/FSDP の通信バックエンド                           |
|                                                           |
|  TensorRT — 推論最適化                                     |
|  +-- グラフ最適化（レイヤー融合、不要演算削除）            |
|  +-- 量子化 (FP16/INT8/FP8/INT4)                          |
|  +-- 動的形状対応                                          |
|                                                           |
|  cuFFT — 高速フーリエ変換                                  |
|  cuSPARSE — スパース行列演算                               |
|  cuRAND — 乱数生成                                         |
+-----------------------------------------------------------+
```

### コード例1: CUDA カーネルの基本構造

```cuda
// vector_add.cu — ベクトル加算の CUDA カーネル
#include <stdio.h>

// GPU上で実行されるカーネル関数
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);

    // ホスト(CPU)メモリ確保
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // デバイス(GPU)メモリ確保
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // ホスト→デバイス転送
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // カーネル起動: 256スレッド/ブロック
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    // デバイス→ホスト転送
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // メモリ解放
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
```

### CUDA 実行モデルの階層構造

```
+-----------------------------------------------------------+
|  CUDA 実行モデル                                            |
+-----------------------------------------------------------+
|                                                           |
|  Grid (グリッド) ← 1回のカーネル起動                       |
|  +---+---+---+---+                                        |
|  | B | B | B | B |  Block (ブロック) × 多数                |
|  +---+---+---+---+                                        |
|  | B | B | B | B |  各ブロックは1つのSMに割り当て          |
|  +---+---+---+---+                                        |
|                                                           |
|  Block (ブロック) 内部:                                    |
|  +---+---+---+---+---+---+---+---+                        |
|  | T | T | T | T | T | T | T | T |  Thread × 最大1024    |
|  +---+---+---+---+---+---+---+---+                        |
|  | T | T | T | T | T | T | T | T |                        |
|  +---+---+---+---+---+---+---+---+                        |
|                                                           |
|  Warp (ワープ):                                            |
|  +---+---+---+---+  ...  +---+---+                        |
|  | T | T | T | T |  ...  | T | T |  32スレッドが同期実行  |
|  +---+---+---+---+  ...  +---+---+                        |
|                                                           |
|  threadIdx.x: ブロック内のスレッドID                        |
|  blockIdx.x:  グリッド内のブロックID                        |
|  blockDim.x:  ブロック内のスレッド数                        |
|  グローバルID = blockIdx.x * blockDim.x + threadIdx.x     |
+-----------------------------------------------------------+
```

### コード例2: PyTorch での GPU 活用

```python
import torch
import torch.nn as nn

# GPU 利用可能かチェック
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# モデルをGPUに転送
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).to(device)

# データをGPUに転送
x = torch.randn(64, 784).to(device)

# 混合精度学習（AMP）
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
optimizer = torch.optim.Adam(model.parameters())

with autocast():  # FP16で高速計算
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target.to(device))

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### コード例3: マルチGPU学習（DDP）

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train(rank, world_size):
    setup(rank, world_size)

    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(rank)
            targets = targets.to(rank)

            loss = model(inputs, targets)
            loss.backward()     # 勾配は自動的に全GPU間で同期
            optimizer.step()
            optimizer.zero_grad()

# 起動: torchrun --nproc_per_node=4 train.py
```

### コード例4: FSDP（Fully Sharded Data Parallel）

```python
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
import functools

# FSDP の混合精度ポリシー
mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,      # パラメータ: BF16
    reduce_dtype=torch.bfloat16,     # 勾配同期: BF16
    buffer_dtype=torch.bfloat16,     # バッファ: BF16
)

# Transformer レイヤー単位でシャーディング
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerDecoderLayer},
)

# FSDP でモデルをラップ
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3相当
    mixed_precision=mixed_precision_policy,
    auto_wrap_policy=auto_wrap_policy,
    device_id=torch.cuda.current_device(),
)

# DDP と同じ学習ループで使用可能
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# FSDP vs DDP のメモリ比較（7Bモデル、BF16）
# DDP:  各GPU に全パラメータのコピー → 14GB/GPU × 4 = 56GB
# FSDP: パラメータを分割 → 3.5GB/GPU × 4 = 14GB（計算時に再構築）
```

---

## 3. NVIDIA vs AMD 比較

### GPU エコシステム比較表

| 項目 | NVIDIA (CUDA) | AMD (ROCm) |
|------|--------------|------------|
| プログラミング言語 | CUDA C/C++ | HIP (CUDAとほぼ同一構文) |
| ディープラーニング | cuDNN, TensorRT | MIOpen |
| 通信ライブラリ | NCCL | RCCL |
| 線形代数 | cuBLAS | rocBLAS |
| フレームワーク対応 | PyTorch, TF, JAX 全対応 | PyTorch 対応、JAX 限定的 |
| クラウド対応 | AWS, GCP, Azure 全対応 | Azure (MI300X), 限定的 |
| エコシステム成熟度 | 非常に高い | 成長中 |
| コストパフォーマンス | 高価だが安定 | 競合価格帯、ドライバ改善中 |

### AMD MI300X のポジション

```
+-----------------------------------------------------------+
|  AMD MI300X — NVIDIAへの対抗馬                              |
+-----------------------------------------------------------+
|                                                           |
|  スペック:                                                  |
|  +-- HBM3: 192GB（H100の80GBの2.4倍）                     |
|  +-- メモリ帯域: 5.3 TB/s（H100の3.35 TB/sの1.6倍）       |
|  +-- FP16: 1,307 TFLOPS                                   |
|  +-- TDP: 750W                                             |
|                                                           |
|  利点:                                                      |
|  +-- 巨大なVRAM → 大規模LLM推論で有利                     |
|  +-- Llama 70B を1チップに載せられる                       |
|  +-- Azure ND MI300X v5 で利用可能                        |
|                                                           |
|  課題:                                                      |
|  +-- ROCm のソフトウェアエコシステムがまだ未成熟           |
|  +-- CUDAカーネル依存のライブラリが動かない場合あり        |
|  +-- Flash Attention の ROCm 対応が遅れがち                |
|  +-- デバッグツールがCUDAほど充実していない                 |
|                                                           |
|  推奨用途:                                                  |
|  +-- 大規模LLM推論（VRAMがボトルネックの場合）             |
|  +-- PyTorch ベースの標準的な学習                          |
|  +-- Azure環境でのH100代替                                 |
+-----------------------------------------------------------+
```

### Tensor Core の精度モード

```
+-------------------------------------------------------+
|  Tensor Core 精度モード（Hopper H100 以降）             |
+-------------------------------------------------------+
|                                                       |
|  FP64  ███████████████████████████████  最高精度       |
|        科学計算、シミュレーション                       |
|                                                       |
|  TF32  ██████████████████  19bit 仮数部               |
|        FP32の代替、学習のデフォルト                     |
|                                                       |
|  FP16  ████████████  半精度浮動小数点                  |
|        混合精度学習 (AMP)                              |
|                                                       |
|  BF16  ████████████  Brain Floating Point              |
|        大規模LLM学習の標準                             |
|                                                       |
|  FP8   ████████  8bit浮動小数点                        |
|        推論最適化、Hopper以降                          |
|                                                       |
|  INT8  ████████  8bit整数                              |
|        量子化推論                                      |
|                                                       |
|  FP4   ████  4bit浮動小数点                            |
|        Blackwell以降、超低精度推論                     |
+-------------------------------------------------------+
```

### 精度フォーマットの詳細比較

| フォーマット | ビット数 | 指数部 | 仮数部 | 範囲 | 主な用途 |
|-------------|---------|--------|--------|------|---------|
| FP32 | 32 | 8 | 23 | ±3.4e38 | 推論（高精度） |
| TF32 | 19 | 8 | 10 | ±3.4e38 | 学習（Ampere以降デフォルト） |
| BF16 | 16 | 8 | 7 | ±3.4e38 | LLM学習の標準 |
| FP16 | 16 | 5 | 10 | ±6.5e4 | 混合精度学習 |
| FP8 (E4M3) | 8 | 4 | 3 | ±448 | 推論（Hopper以降） |
| FP8 (E5M2) | 8 | 5 | 2 | ±57344 | 学習の勾配 |
| INT8 | 8 | - | - | -128〜127 | 量子化推論 |
| FP4 (E2M1) | 4 | 2 | 1 | ±6 | 超低精度推論（Blackwell） |
| INT4 | 4 | - | - | -8〜7 | GPTQ/AWQ量子化推論 |

---

## 4. AI 学習用 GPU 選び

### コード例5: GPU スペック確認スクリプト

```python
import torch
import subprocess

def gpu_info():
    if not torch.cuda.is_available():
        print("CUDA GPU が見つかりません")
        return

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"=== GPU {i}: {props.name} ===")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total VRAM: {props.total_mem / 1e9:.1f} GB")
        print(f"  SM Count: {props.multi_processor_count}")
        print(f"  Max Threads/SM: {props.max_threads_per_multi_processor}")

    # nvidia-smi で詳細情報
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,power.max_limit,temperature.gpu",
         "--format=csv,noheader"],
        capture_output=True, text=True
    )
    print(f"\nnvidia-smi:\n{result.stdout}")

gpu_info()
```

### コード例6: VRAM 使用量の見積もり

```python
def estimate_vram_for_training(
    param_count_billion: float,
    precision: str = "bf16",
    optimizer: str = "adam",
    batch_size: int = 1,
    seq_length: int = 2048,
) -> dict:
    """LLM学習に必要なVRAM量を概算する"""

    bytes_per_param = {
        "fp32": 4, "tf32": 4, "bf16": 2, "fp16": 2, "fp8": 1
    }[precision]

    params_bytes = param_count_billion * 1e9 * bytes_per_param

    # オプティマイザ状態（Adam: パラメータの2倍）
    optimizer_multiplier = {"adam": 2, "adamw": 2, "sgd": 0, "adafactor": 0.5}
    opt_bytes = param_count_billion * 1e9 * 4 * optimizer_multiplier[optimizer]

    # 勾配（パラメータと同じサイズ）
    grad_bytes = params_bytes

    # 活性化メモリ（概算）
    activation_bytes = batch_size * seq_length * param_count_billion * 1e9 * 2 / 1000

    total_gb = (params_bytes + opt_bytes + grad_bytes + activation_bytes) / 1e9

    return {
        "モデルパラメータ": f"{params_bytes / 1e9:.1f} GB",
        "オプティマイザ状態": f"{opt_bytes / 1e9:.1f} GB",
        "勾配": f"{grad_bytes / 1e9:.1f} GB",
        "活性化（概算）": f"{activation_bytes / 1e9:.1f} GB",
        "合計（概算）": f"{total_gb:.1f} GB",
    }

# 7Bモデルの学習に必要なVRAM
print(estimate_vram_for_training(7, "bf16", "adam"))
# モデルパラメータ: 14.0 GB
# オプティマイザ状態: 56.0 GB
# 勾配: 14.0 GB
# 合計: 約100 GB → H100 80GB x2 または A100 80GB x2 が必要
```

### コード例7: 推論時のVRAM見積もり

```python
def estimate_vram_for_inference(
    param_count_billion: float,
    precision: str = "fp16",
    batch_size: int = 1,
    seq_length: int = 2048,
    kv_cache: bool = True,
    num_layers: int = None,
    num_heads: int = None,
    head_dim: int = 128,
) -> dict:
    """LLM推論に必要なVRAM量を概算する"""

    bytes_per_param = {
        "fp32": 4, "fp16": 2, "bf16": 2, "int8": 1,
        "int4": 0.5, "fp8": 1, "gptq_4bit": 0.5, "awq_4bit": 0.5,
    }[precision]

    # モデルの重み
    model_bytes = param_count_billion * 1e9 * bytes_per_param

    # KVキャッシュ（LLM特有のメモリ消費）
    kv_bytes = 0
    if kv_cache and num_layers and num_heads:
        # KV cache = 2 (K+V) * batch * layers * heads * head_dim * seq * dtype_size
        kv_bytes = (2 * batch_size * num_layers * num_heads
                    * head_dim * seq_length * 2)  # FP16

    # 一時バッファ（約10%のオーバーヘッド）
    overhead = (model_bytes + kv_bytes) * 0.1

    total_gb = (model_bytes + kv_bytes + overhead) / 1e9

    return {
        "モデル重み": f"{model_bytes / 1e9:.1f} GB",
        "KVキャッシュ": f"{kv_bytes / 1e9:.1f} GB",
        "オーバーヘッド": f"{overhead / 1e9:.1f} GB",
        "合計": f"{total_gb:.1f} GB",
        "推奨GPU": recommend_gpu(total_gb),
    }

def recommend_gpu(vram_needed_gb):
    """必要VRAM量からGPUを推奨"""
    gpus = [
        (16, "RTX 4060 Ti 16GB"),
        (24, "RTX 4090"),
        (48, "RTX 6000 Ada / A6000"),
        (80, "A100 80GB / H100 80GB"),
        (192, "MI300X"),
    ]
    for vram, name in gpus:
        if vram_needed_gb <= vram * 0.9:  # 90%上限
            return name
    return "マルチGPU構成が必要"

# Llama 2 7B の推論VRAM（各精度）
for prec in ["fp16", "int8", "int4"]:
    result = estimate_vram_for_inference(
        7, prec, batch_size=1, seq_length=4096,
        num_layers=32, num_heads=32, head_dim=128
    )
    print(f"Llama-2-7B ({prec}): {result['合計']} → {result['推奨GPU']}")
# fp16: 15.4 GB → RTX 4090
# int8: 8.5 GB  → RTX 4060 Ti 16GB
# int4: 5.0 GB  → RTX 4060 Ti 16GB
```

### 用途別GPU推奨表

| 用途 | 推奨GPU | VRAM | 概算予算 | 備考 |
|------|---------|------|---------|------|
| 学習入門・推論 | RTX 4060 Ti 16GB | 16GB | 6万円 | ファインチューニング入門 |
| 中規模学習 | RTX 4090 | 24GB | 30万円 | 7Bモデルの推論、小規模学習 |
| 本格的学習 | A100 80GB (クラウド) | 80GB | 時間課金 | 7-13Bモデルの学習 |
| 大規模LLM | H100 80GB (クラウド) | 80GB | 時間課金 | 70B+モデル、マルチGPU |
| 推論最適化 | L4 / T4 (クラウド) | 24GB / 16GB | 低コスト | 量子化モデルの推論サービング |
| エッジ推論 | Jetson Orin | 8-64GB | 5-30万円 | ローカル推論 |
| 超大規模LLM推論 | MI300X (Azure) | 192GB | 時間課金 | 70B+ を1チップで推論 |

---

## 5. VRAM管理と最適化テクニック

### メモリ最適化手法の比較表

| 手法 | VRAM削減率 | 速度影響 | 精度影響 | 実装難度 |
|------|-----------|---------|---------|---------|
| 混合精度学習 (AMP) | 約40% | 高速化 | ほぼなし | 低 |
| 勾配チェックポイント | 約60% | 20-30%低下 | なし | 低 |
| DeepSpeed ZeRO-1 | 約40% | わずかに低下 | なし | 中 |
| DeepSpeed ZeRO-2 | 約60% | わずかに低下 | なし | 中 |
| DeepSpeed ZeRO-3 | 約80% | 通信オーバーヘッド | なし | 中 |
| LoRA / QLoRA | 約90% | 高速化 | わずかに低下 | 低 |
| 4bit量子化(推論) | 約75% | やや低下 | モデル依存 | 低 |
| Flash Attention 2 | 活性化50%削減 | 高速化 | なし | 低 |
| Paged Attention (vLLM) | KV 50%削減 | 同等 | なし | 自動 |

### コード例8: 勾配チェックポイントの実装

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class LargeTransformerBlock(nn.Module):
    """勾配チェックポイント対応のTransformerブロック"""

    def __init__(self, d_model=1024, nhead=16, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def _forward_impl(self, x):
        """実際の前方計算"""
        # Self-Attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = residual + x

        # Feed-Forward
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x

    def forward(self, x):
        if self.use_checkpoint and self.training:
            # 勾配チェックポイント: 前方計算の中間結果を保存せず、
            # 逆方向計算時に再計算する → メモリ削減
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)

# メモリ使用量の比較
# チェックポイントなし: 活性化メモリ ∝ レイヤー数
# チェックポイントあり: 活性化メモリ ∝ √レイヤー数
# 例: 32レイヤー → 活性化メモリが約1/5に削減
```

### コード例9: DeepSpeed ZeRO の設定

```python
# DeepSpeed ZeRO-3 の設定例
# deepspeed_config.json
deepspeed_config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,

    # FP16/BF16 設定
    "bf16": {
        "enabled": True
    },

    # ZeRO Stage 3: パラメータ + 勾配 + オプティマイザ状態を全GPU間で分割
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",           # オプティマイザ状態をCPUに退避
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",           # パラメータもCPUに退避（最大節約）
            "pin_memory": True
        },
        "overlap_comm": True,          # 通信と計算のオーバーラップ
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    },

    # オプティマイザ
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },

    # 学習率スケジューラ
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 1000,
            "total_num_steps": 100000
        }
    }
}

# ZeRO Stage比較（7Bモデル、4 GPU）
# Stage 0 (DDP):  全パラメータ複製     → 各GPU 100GB必要
# Stage 1:        オプティマイザ分割    → 各GPU 72GB
# Stage 2:        +勾配分割             → 各GPU 58GB
# Stage 3:        +パラメータ分割       → 各GPU 30GB
# Stage 3+Offload: +CPU退避            → 各GPU 5GB (CPUメモリ使用)
```

### コード例10: LoRA によるメモリ効率的なファインチューニング

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# ベースモデルのロード（量子化で更にメモリ削減）
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 4bit量子化
    bnb_4bit_quant_type="nf4",            # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16, # 計算はBF16
    bnb_4bit_use_double_quant=True,       # 二重量子化
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA 設定
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # LoRAランク（低いほどパラメータ少）
    lora_alpha=32,           # スケーリング係数
    lora_dropout=0.05,
    target_modules=[         # LoRAを適用するレイヤー
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
)

# LoRA適用
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 20,971,520 (0.31%)
# all params: 6,758,404,096
# → 全パラメータの0.31%のみ学習

# VRAM使用量比較（Llama-2-7B）
# フル学習 (BF16):    ~100 GB（4x A100必要）
# LoRA (BF16):        ~18 GB（1x A100で可能）
# QLoRA (4bit+LoRA):  ~6 GB（RTX 4060 Ti 16GBで可能）
```

---

## 6. NVLink とマルチGPUトポロジー

### NVLink の世代と帯域

```
+-----------------------------------------------------------+
|  NVLink の進化                                              |
+-----------------------------------------------------------+
|                                                           |
|  NVLink 1.0 (Pascal, 2016):   80 GB/s (双方向160)        |
|  NVLink 2.0 (Volta, 2017):   150 GB/s (双方向300)        |
|  NVLink 3.0 (Ampere, 2020):  300 GB/s (双方向600)        |
|  NVLink 4.0 (Hopper, 2022):  450 GB/s (双方向900)        |
|  NVLink 5.0 (Blackwell, 2024): 900 GB/s (双方向1800)     |
|                                                           |
|  比較: PCIe 5.0 x16 = 64 GB/s (双方向128)                |
|  → NVLink 4.0 は PCIe の約7倍の帯域                       |
|                                                           |
|  NVSwitch (Hopper):                                       |
|  +-- 8 GPU を全対全接続                                    |
|  +-- 任意の2GPU間で900 GB/s                                |
|  +-- DGX H100: 8x H100 + NVSwitch 構成                   |
|                                                           |
|  なぜ重要か:                                                |
|  +-- マルチGPU学習で勾配同期が高速                         |
|  +-- テンソル並列（モデルを複数GPUに分割）が実用的に       |
|  +-- Megatron-LM のパイプライン並列に不可欠                |
+-----------------------------------------------------------+
```

### マルチGPU学習の並列戦略

```
+-----------------------------------------------------------+
|  並列化戦略の比較                                           |
+-----------------------------------------------------------+
|                                                           |
|  データ並列 (Data Parallel / DDP)                          |
|  +-- 全GPUに同じモデルをコピー                             |
|  +-- データを分割して各GPUに配布                           |
|  +-- 勾配を AllReduce で同期                               |
|  +-- 簡単、スケーラブル                                    |
|  +-- 制約: 1GPUにモデルが収まる必要                        |
|                                                           |
|  テンソル並列 (Tensor Parallel)                             |
|  +-- 各レイヤーの重み行列を複数GPUに分割                   |
|  +-- NVLink 必須（高帯域通信が頻繁に発生）                 |
|  +-- 1ノード内の2-8 GPU で使用                             |
|  +-- Megatron-LM が代表的実装                              |
|                                                           |
|  パイプライン並列 (Pipeline Parallel)                       |
|  +-- レイヤーを複数GPUに分割（前半/後半など）              |
|  +-- マイクロバッチで各GPUを並列動作                       |
|  +-- ノード間でも使用可能                                  |
|  +-- パイプラインバブルによる効率低下あり                   |
|                                                           |
|  FSDP (Fully Sharded Data Parallel)                       |
|  +-- ZeRO-3: パラメータ+勾配+オプティマイザを分割          |
|  +-- 計算時にパラメータを収集、完了後に解放                |
|  +-- PyTorch ネイティブ実装                                |
+-----------------------------------------------------------+
```

---

## 7. GPU プロファイリングと最適化

### コード例11: PyTorch Profiler によるボトルネック分析

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# プロファイラの設定
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,        # 1バッチ待機
        warmup=1,      # 1バッチウォームアップ
        active=3,      # 3バッチプロファイリング
        repeat=1,
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 6:
            break

        with record_function("data_transfer"):
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)

        with record_function("forward"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        with record_function("backward"):
            loss.backward()

        with record_function("optimizer_step"):
            optimizer.step()
            optimizer.zero_grad()

        prof.step()

# プロファイル結果の表示
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20,
))

# よくあるボトルネック:
# 1. データローダー → num_workers増加、pin_memory=True
# 2. CPU-GPU転送 → 非同期転送、プリフェッチ
# 3. 小さなカーネルの連続 → カーネル融合、torch.compile()
# 4. 同期ポイント → 非同期AllReduceの活用
```

### コード例12: torch.compile() によるカーネル融合

```python
import torch

# PyTorch 2.0+ の torch.compile()
# JITコンパイルでカーネルを自動融合し高速化

model = MyModel().to(device)

# コンパイル（初回は遅い、以降は高速）
compiled_model = torch.compile(
    model,
    mode="max-autotune",  # 最大最適化（コンパイル時間は長い）
    # mode="reduce-overhead",  # オーバーヘッド削減重視
    # mode="default",          # バランス
    fullgraph=True,       # グラフ全体を最適化
    dynamic=False,        # 静的形状（動的形状ならTrue）
)

# 使い方は変わらない
output = compiled_model(input_tensor)

# 高速化の仕組み:
# 1. Triton カーネル生成: 複数の演算を1つのGPUカーネルに融合
# 2. メモリ最適化: 中間テンソルの割り当てを削減
# 3. 自動チューニング: 最適なブロックサイズを探索
#
# 典型的な高速化率:
# - 推論: 1.3-2.0倍
# - 学習: 1.1-1.5倍
# - Transformer モデルで特に効果的
```

### コード例13: Flash Attention 2 の活用

```python
import torch
from flash_attn import flash_attn_func

# Flash Attention 2: メモリ効率的な Attention 実装
# 標準 Attention: O(N^2) メモリ → Flash: O(N) メモリ

# 標準的な Attention（メモリ非効率）
def standard_attention(q, k, v):
    # q, k, v: (batch, seq_len, num_heads, head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
    # scores: (batch, num_heads, seq_len, seq_len) ← O(N^2) メモリ!
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output

# Flash Attention 2（メモリ効率的）
def flash_attention(q, k, v):
    # 入力形状: (batch, seq_len, num_heads, head_dim)
    output = flash_attn_func(
        q, k, v,
        dropout_p=0.0,
        softmax_scale=None,  # 自動計算
        causal=True,         # 因果的マスク（LLM用）
    )
    return output

# メモリ使用量比較 (batch=4, seq=8192, heads=32, dim=128)
# 標準 Attention: 8192^2 * 32 * 4 * 2 bytes = 16 GB （Attention行列）
# Flash Attention: O(N) ≈ 数MB （タイル処理で逐次計算）
#
# 速度比較:
# 標準: ~150ms
# Flash Attention 2: ~30ms (5倍高速)
#
# PyTorch 2.0+ では F.scaled_dot_product_attention() で自動的に使用
```

---

## 8. アンチパターン

### アンチパターン1: VRAM不足を無視した学習

```python
# NG: OOM (Out of Memory) に対処しない
model = LargeModel().to("cuda")
# RuntimeError: CUDA out of memory

# OK: 段階的にメモリ対策を適用
# 1. バッチサイズを下げる
# 2. 勾配累積を使う
# 3. AMP (混合精度) を有効化
# 4. 勾配チェックポイントを有効化
# 5. DeepSpeed / FSDP を導入
# 6. LoRA / QLoRA でパラメータを削減
```

### アンチパターン2: GPU使用率を確認しない

```bash
# NG: 学習を開始して放置
python train.py &

# OK: GPU使用率を常時モニタリング
watch -n 1 nvidia-smi
# GPU-Util が 90%+ であることを確認
# Memory-Usage が適切であることを確認

# より詳細なモニタリング
nvidia-smi dmon -s pucvmet -d 1
```

### アンチパターン3: データローダーのボトルネック

```python
# NG: GPU が データ待ちでアイドル状態
dataloader = DataLoader(dataset, batch_size=32)
# GPU-Util: 30-50%（データ転送がボトルネック）

# OK: データローダーを最適化
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,           # 並列データロード
    pin_memory=True,         # ページロックメモリ（転送高速化）
    persistent_workers=True, # ワーカー再起動を防止
    prefetch_factor=4,       # 先読み数
)
# GPU-Util: 95%+
```

### アンチパターン4: 混合精度を使わない学習

```python
# NG: FP32 のみで学習（遅い、メモリ消費大）
output = model(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()

# OK: AMP (Automatic Mixed Precision) を使用
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast(dtype=torch.bfloat16):  # BF16で計算
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 効果:
# - 速度: 1.5-3倍高速
# - メモリ: 30-50%削減
# - 精度: ほぼ同等（BF16の場合）
```

### アンチパターン5: torch.compile() を使わない

```python
# NG: PyTorch 2.0+ で compile を使わない
model = MyModel().to(device)
# → 個別のCUDAカーネルが逐次起動（オーバーヘッド大）

# OK: torch.compile() でカーネル融合
model = torch.compile(model, mode="reduce-overhead")
# → 複数演算が1つのカーネルに融合（20-50%高速化）

# 注意点:
# - 初回コンパイルに数分かかる
# - 動的形状（可変バッチサイズ等）では効果が減る
# - 一部の演算（custom op等）はコンパイル非対応の場合がある
```

---

## 9. NVIDIAエンタープライズ製品の理解

### DGX / HGX / SuperPOD の構成

```
+-----------------------------------------------------------+
|  NVIDIA AI インフラストラクチャの階層                        |
+-----------------------------------------------------------+
|                                                           |
|  DGX H100 (1ノード)                                       |
|  +-- 8x H100 SXM (640GB VRAM)                            |
|  +-- NVSwitch で全GPU間を900GB/s接続                      |
|  +-- 2x Intel Xeon                                        |
|  +-- 2TB システムメモリ                                    |
|  +-- 8x ConnectX-7 (400Gbps InfiniBand)                   |
|  +-- 価格: 約$300,000                                      |
|                                                           |
|  HGX H100 (ベースボード)                                   |
|  +-- DGX の GPU ボード部分のみ（OEM向け）                  |
|  +-- Dell, HPE, Lenovo 等がサーバーに組み込み              |
|                                                           |
|  DGX SuperPOD (クラスタ)                                   |
|  +-- 32-256 DGX H100 ノード                                |
|  +-- InfiniBand で全ノード接続                             |
|  +-- 256 ノード = 2,048 H100 GPU                           |
|  +-- 最大 ~40 EFLOPS (FP8)                                 |
|  +-- GPT-4 級モデルの学習が可能                            |
|                                                           |
|  DGX Cloud                                                 |
|  +-- DGX SuperPOD をクラウドで利用                         |
|  +-- AWS, GCP, Azure, Oracle 経由                          |
|  +-- 月額サブスクリプション                                |
+-----------------------------------------------------------+
```

---

## FAQ

### Q1. 消費者向けGPU（GeForce）とデータセンター向けGPU（A100/H100）の違いは？

主な違いは VRAM容量（24GB vs 80GB）、メモリ帯域（ECC HBM3 vs GDDR6X）、NVLink接続（マルチGPU通信速度）、FP64性能、24時間連続運用の耐久性。学習ではVRAMとNVLinkが特に重要。個人利用ではRTX 4090が最高コスパ。なおGeForce RTX 4090はNVLink非対応のため、マルチGPU学習ではPCIe経由の遅い通信になる点に注意。

### Q2. CUDAバージョンとドライバの互換性は？

CUDAは前方互換性を持つ。つまり新しいドライバは古いCUDA Toolkitで作られたアプリを実行できる。PyTorchはCUDA Toolkitをバンドルしているため、ドライバさえ対応バージョン以上であれば動作する。`nvidia-smi` の右上に表示されるCUDAバージョンはドライバが対応する最大バージョン。PyTorch 2.x はCUDA 11.8 と 12.x をサポートしている。

### Q3. クラウドGPUと自前GPU、どちらが得か？

短期間の実験・プロトタイピングはクラウド（AWS, GCP, Lambda Labs）が圧倒的に安い。継続的に月100時間以上使うなら自前GPUの方がTCOが低くなる場合がある。ただしH100クラスはクラウドでしか現実的に利用できない。Spot/Preemptible インスタンスを活用すると70-90%割引になる。

### Q4. Blackwell世代（B200/RTX 5090）の何が変わるか？

FP4（4bit浮動小数点）対応で推論性能が飛躍的に向上。B200はH100比で約2.5倍のAI学習性能、5倍の推論性能を実現する。NVLink 5.0（双方向1.8TB/s）でマルチGPU通信も2倍高速化。RTX 5090は消費者向けでもBF16 Tensor Core性能が大幅向上し、AI推論のローカル実行が更に実用的になる。

### Q5. AMD ROCm は実用的か？

2025年時点ではPyTorchの標準的な学習・推論ワークロードであれば実用レベルに達している。MI300Xの192GB VRAMは大規模LLM推論で大きな利点。ただしFlash Attention、カスタムCUDAカーネルを多用するライブラリ（bitsandbytes等）はROCm対応が遅れがちで、最先端の最適化手法がすぐには使えないことがある。Azure上でMI300Xを試すのが最も手軽。

### Q6. GPU以外のAIアクセラレータ（TPU、Inferentia）との比較は？

NVIDIA GPUの最大の利点はCUDAエコシステムの成熟度。PyTorch、TensorFlow、JAXのいずれもが最も安定して動作する。TPUはJAXとの組み合わせで行列演算のコスパが最高だがベンダーロックイン。InferentiaはAWSでの推論コスト最適化に特化。選択基準は「柔軟性ならGPU、コスパならTPU/Inferentia、エコシステム統合ならそのクラウドのASIC」。

---

## まとめ

| 概念 | 要点 |
|------|------|
| GPU vs CPU | GPUは並列処理、CPUは逐次処理に最適 |
| CUDA | NVIDIAのGPUプログラミングプラットフォーム |
| Tensor Core | 行列演算特化ユニット、AI学習を大幅高速化 |
| 混合精度（AMP） | FP16/BF16で高速化しつつFP32相当の精度を維持 |
| VRAM | GPU選びで最重要、モデルサイズに直結 |
| NVLink | マルチGPU間の高速通信、テンソル並列に必須 |
| ROCm | AMD GPUのCUDA対抗プラットフォーム |
| DeepSpeed / FSDP | 大規模モデル学習のメモリ最適化 |
| Flash Attention | O(N)メモリのAttention実装、5倍高速 |
| torch.compile | PyTorch 2.0+のJITコンパイラ、自動最適化 |
| LoRA / QLoRA | パラメータ効率的なファインチューニング |

---

## 次に読むべきガイド

- **01-computing/02-edge-ai.md** — エッジAI：NPU、Coral、Jetson
- **01-computing/03-cloud-ai-hardware.md** — クラウドAIハードウェア：TPU、Inferentia
- **02-emerging/03-future-hardware.md** — 未来のハードウェア：量子コンピュータ

---

## 参考文献

1. **NVIDIA CUDA Programming Guide** https://docs.nvidia.com/cuda/cuda-c-programming-guide/
2. **PyTorch — CUDA Semantics** https://pytorch.org/docs/stable/notes/cuda.html
3. **NVIDIA — GPU Architecture Whitepapers** https://www.nvidia.com/en-us/technologies/
4. **Hugging Face — Model Memory Calculator** https://huggingface.co/spaces/hf-accelerate/model-memory-usage
5. **Flash Attention 2** https://github.com/Dao-AILab/flash-attention
6. **DeepSpeed** https://www.deepspeed.ai/
7. **AMD ROCm** https://www.amd.com/en/products/software/rocm.html

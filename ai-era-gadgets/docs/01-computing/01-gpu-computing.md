# GPU コンピューティングガイド

> NVIDIA/AMD GPU、CUDA、AI学習用GPU選びの実践知識を網羅する

## この章で学ぶこと

1. **GPU アーキテクチャ** の基本 — CPU との違い、NVIDIA/AMD の世代別特徴
2. **CUDA エコシステム** — プログラミングモデル、主要ライブラリ、ROCm との比較
3. **AI学習用GPU選び** — 用途別スペック比較、コストパフォーマンス分析

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

---

## 4. AI 学習用 GPU 選び

### コード例4: GPU スペック確認スクリプト

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

### コード例5: VRAM 使用量の見積もり

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

### 用途別GPU推奨表

| 用途 | 推奨GPU | VRAM | 概算予算 | 備考 |
|------|---------|------|---------|------|
| 学習入門・推論 | RTX 4060 Ti 16GB | 16GB | 6万円 | ファインチューニング入門 |
| 中規模学習 | RTX 4090 | 24GB | 30万円 | 7Bモデルの推論、小規模学習 |
| 本格的学習 | A100 80GB (クラウド) | 80GB | 時間課金 | 7-13Bモデルの学習 |
| 大規模LLM | H100 80GB (クラウド) | 80GB | 時間課金 | 70B+モデル、マルチGPU |
| 推論最適化 | L4 / T4 (クラウド) | 24GB / 16GB | 低コスト | 量子化モデルの推論サービング |
| エッジ推論 | Jetson Orin | 8-64GB | 5-30万円 | ローカル推論 |

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

---

## 6. アンチパターン

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

---

## FAQ

### Q1. 消費者向けGPU（GeForce）とデータセンター向けGPU（A100/H100）の違いは？

主な違いは VRAM容量（24GB vs 80GB）、メモリ帯域（ECC HBM3 vs GDDR6X）、NVLink接続（マルチGPU通信速度）、FP64性能、24時間連続運用の耐久性。学習ではVRAMとNVLinkが特に重要。個人利用ではRTX 4090が最高コスパ。

### Q2. CUDAバージョンとドライバの互換性は？

CUDAは前方互換性を持つ。つまり新しいドライバは古いCUDA Toolkitで作られたアプリを実行できる。PyTorchはCUDA Toolkitをバンドルしているため、ドライバさえ対応バージョン以上であれば動作する。`nvidia-smi` の右上に表示されるCUDAバージョンはドライバが対応する最大バージョン。

### Q3. クラウドGPUと自前GPU、どちらが得か？

短期間の実験・プロトタイピングはクラウド（AWS, GCP, Lambda Labs）が圧倒的に安い。継続的に月100時間以上使うなら自前GPUの方がTCOが低くなる場合がある。ただしH100クラスはクラウドでしか現実的に利用できない。Spot/Preemptible インスタンスを活用すると70-90%割引になる。

---

## まとめ

| 概念 | 要点 |
|------|------|
| GPU vs CPU | GPUは並列処理、CPUは逐次処理に最適 |
| CUDA | NVIDIAのGPUプログラミングプラットフォーム |
| Tensor Core | 行列演算特化ユニット、AI学習を大幅高速化 |
| 混合精度（AMP） | FP16/BF16で高速化しつつFP32相当の精度を維持 |
| VRAM | GPU選びで最重要、モデルサイズに直結 |
| NVLink | マルチGPU間の高速通信 |
| ROCm | AMD GPUのCUDA対抗プラットフォーム |
| DeepSpeed / FSDP | 大規模モデル学習のメモリ最適化 |

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

# GPU and Parallel Computing

> The GPU "executes a single instruction across thousands of cores simultaneously" -- a fundamentally different approach from CPU-style sequential processing.

## Learning Objectives

- [ ] Explain the internal architecture of GPUs and how they differ from CPUs
- [ ] Understand the mechanisms of GPGPU (General-Purpose GPU Computing)
- [ ] Explain the role of GPUs in AI/ML
- [ ] Understand the basic patterns of CUDA programming
- [ ] Explain GPU interconnect technologies (NVLink, NVSwitch)
- [ ] Master the fundamentals of multi-GPU configurations and distributed training

## Prerequisites


---

## 1. GPU vs CPU

### 1.1 Design Philosophy Differences

```
CPU: Latency-Optimized (execute a single task as fast as possible)
  ┌──────────────────────────────────────┐
  │ ┌────────────────────┐ ┌──────────┐ │
  │ │ Large Cache         │ │ Branch   │ │  ← Complex control logic
  │ │ (L1: 64KB x2)      │ │ Predictor│ │  ← Few powerful cores
  │ │ (L2: 1MB)           │ └──────────┘ │
  │ │ (L3: 32MB)          │              │
  │ └────────────────────┘              │
  │ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐│
  │ │Core 0│ │Core 1│ │Core 2│ │Core 3││  ← 4-16 cores
  │ │(very  │ │(very  │ │(very  │ │(very  │
  │ │powerful)│ │powerful)│ │powerful)│ │powerful)│
  │ └──────┘ └──────┘ └──────┘ └──────┘│
  └──────────────────────────────────────┘

GPU: Throughput-Optimized (process massive tasks in parallel)
  ┌──────────────────────────────────────┐
  │ ┌────┐                  Small Cache  │
  │ │Ctrl│                               │  ← Simple control logic
  │ └────┘                               │  ← Thousands of small cores
  │ ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐  │
  │ │c││c││c││c││c││c││c││c││c││c│  │  ← 32-128 cores per SM
  │ └─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘  │    (Streaming Multiprocessor)
  │ ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐  │
  │ │c││c││c││c││c││c││c││c││c││c│  │
  │ └─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘  │  ← Tens to hundreds of SMs
  │ ... (thousands of cores)            │
  └──────────────────────────────────────┘
```

### 1.2 Numerical Comparison

| Metric | CPU (Ryzen 9 7950X) | GPU (RTX 4090) |
|------|---------------------|----------------|
| Core Count | 16 | 16,384 (CUDA) |
| Clock Speed | 5.7 GHz | 2.5 GHz |
| FP32 Performance | ~1.5 TFLOPS | 82.6 TFLOPS |
| Memory Bandwidth | 83 GB/s (DDR5) | 1,008 GB/s (GDDR6X) |
| TDP | 170W | 450W |
| Strengths | Sequential processing, branch-heavy code | Massive data parallel processing |
| Weaknesses | Large-scale parallelism | Branch-heavy code, sequential processing |

### 1.3 Differences in Transistor Allocation

```
CPU Die Area Allocation:
  ┌──────────────────────────────────────┐
  │ [       Cache: 40%       ]          │
  │ [Control Logic: 30%]                │
  │ [Branch Prediction: 10%]            │
  │ [ALUs: 15%]                         │
  │ [Other: 5%]                         │
  └──────────────────────────────────────┘
  → The majority of die area is devoted to data fetching and control;
    only about 15% is used for actual computation

GPU Die Area Allocation:
  ┌──────────────────────────────────────┐
  │ [       ALUs: 60%       ]           │
  │ [Cache: 15%]                        │
  │ [Control Logic: 10%]                │
  │ [Memory Controller: 10%]            │
  │ [Other: 5%]                         │
  └──────────────────────────────────────┘
  → The majority of die area is devoted to computation
  → Control is minimal; instead, massive threads hide latency
```

---

## 2. GPU Architecture

### 2.1 NVIDIA GPU Structure

```
NVIDIA GPU (Ada Lovelace Architecture):

  ┌─────────────────────────────────────────────┐
  │                  GPU Chip                     │
  │                                               │
  │  ┌─────────────────────────────────────────┐ │
  │  │  GPC (Graphics Processing Cluster) x 12 │ │
  │  │  ┌──────────────────────────────────┐   │ │
  │  │  │  TPC (Texture Processing Cluster) │   │ │
  │  │  │  ┌────────────────────────────┐  │   │ │
  │  │  │  │  SM (Streaming Multiprocessor)│ │   │ │
  │  │  │  │  ┌─────────────────────────┐│ │   │ │
  │  │  │  │  │ CUDA Core x 128        ││ │   │ │
  │  │  │  │  │ Tensor Core x 4        ││ │   │ │
  │  │  │  │  │ RT Core x 1            ││ │   │ │
  │  │  │  │  │ Shared Memory: 128KB   ││ │   │ │
  │  │  │  │  │ Register File: 256KB   ││ │   │ │
  │  │  │  │  │ L1 Cache: 128KB        ││ │   │ │
  │  │  │  │  └─────────────────────────┘│ │   │ │
  │  │  │  └────────────────────────────┘  │   │ │
  │  │  └──────────────────────────────────┘   │ │
  │  └─────────────────────────────────────────┘ │
  │                                               │
  │  ┌─────────────────┐  ┌────────────────────┐│
  │  │  L2 Cache: 72MB  │  │ Memory Controller  ││
  │  └─────────────────┘  │ GDDR6X 24GB        ││
  │                        │ 384-bit bus         ││
  │                        └────────────────────┘│
  └─────────────────────────────────────────────┘

  RTX 4090: 128 SM x 128 CUDA Cores = 16,384 CUDA Cores
```

### 2.2 Warps and Thread Hierarchy

```
NVIDIA GPU Thread Hierarchy:

  Grid (entire workload)
  ├── Block 0
  │   ├── Warp 0: Thread 0-31   ← 32 threads execute in perfect lockstep
  │   ├── Warp 1: Thread 32-63
  │   └── ...
  ├── Block 1
  │   ├── Warp 0: Thread 0-31
  │   └── ...
  └── ...

  All threads within a warp execute the same instruction simultaneously
  (SIMT: Single Instruction, Multiple Threads)

  → When branches occur, "warp divergence" happens:
  if (threadIdx.x < 16) {
      // First 16 threads execute here
      // Last 16 threads wait (wasted)
  } else {
      // Last 16 threads execute here
      // First 16 threads wait (wasted)
  }
  → This is why branch-heavy code is slow on GPUs
```

### 2.3 GPU Memory Hierarchy

```
GPU Memory Hierarchy Details:

  ┌──────────────────────────────────────────────┐
  │ Register File (per-thread private)            │
  │ Capacity: 256KB/SM                            │
  │ Latency: 1 cycle                              │
  │ Bandwidth: Maximum                            │
  ├──────────────────────────────────────────────┤
  │ Shared Memory (shared within a block)         │
  │ Capacity: Up to 100KB/SM (configurable)       │
  │ Latency: ~5 cycles                            │
  │ Bandwidth: ~128 bytes/cycle                   │
  │ → Fast memory explicitly managed by the programmer │
  ├──────────────────────────────────────────────┤
  │ L1 Cache (per-SM, automatically managed)      │
  │ Capacity: 128KB/SM (shared with shared mem)   │
  │ Latency: ~30 cycles                           │
  ├──────────────────────────────────────────────┤
  │ L2 Cache (shared across all SMs)              │
  │ Capacity: 72MB (RTX 4090)                     │
  │ Latency: ~200 cycles                          │
  ├──────────────────────────────────────────────┤
  │ Global Memory (GDDR6X / HBM)                 │
  │ Capacity: 24GB (RTX 4090) / 80GB (H100)      │
  │ Latency: ~400-600 cycles                      │
  │ Bandwidth: 1,008 GB/s (RTX 4090) / 3.35 TB/s (H100) │
  └──────────────────────────────────────────────┘

  Coalesced Memory Access:
  ┌──────────────────────────────────────────┐
  │ Good Example (coalesced access):          │
  │ Thread 0  → addr[0]                      │
  │ Thread 1  → addr[1]   → 1 memory transaction │
  │ Thread 2  → addr[2]                      │
  │ ...                                       │
  │ Thread 31 → addr[31]                     │
  │                                           │
  │ Bad Example (non-coalesced access):       │
  │ Thread 0  → addr[0]    → separate transaction │
  │ Thread 1  → addr[128]  → separate transaction │
  │ Thread 2  → addr[256]  → separate transaction │
  │ ...                                       │
  │ → Wastes bandwidth, performance drops 10-100x │
  └──────────────────────────────────────────┘
```

---

## 3. CUDA Programming

### 3.1 Basic Structure

```cuda
// CUDA: Vector Addition Example
// C = A + B (compute each element in parallel)

// Kernel function executed on the GPU
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    // Each thread handles one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];  // Simple: 1 thread = 1 addition
    }
}

int main() {
    int N = 1000000;  // 1 million elements
    float *d_A, *d_B, *d_C;

    // Allocate GPU memory
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Transfer data from CPU to GPU
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel: 3907 blocks x 256 threads = ~1 million threads
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    // Transfer data from GPU to CPU
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}

// CPU: 1 million iterations in a loop → ~1ms
// GPU: 1 million threads in parallel → ~0.01ms + memory transfer cost
```

### 3.2 Leveraging Shared Memory

```cuda
// Matrix Multiplication: Optimization using shared memory
// C = A x B (N x N matrices)

#define TILE_SIZE 32

__global__ void matMul(float *A, float *B, float *C, int N) {
    // Load tiles into shared memory
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Process tile by tile
    for (int t = 0; t < N / TILE_SIZE; t++) {
        // Load from global memory to shared memory
        tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];

        // Wait for all threads to finish loading
        __syncthreads();

        // Compute from shared memory (fast)
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}

// Why shared memory matters:
// Global memory: 400-600 cycle latency
// Shared memory: 5 cycle latency
// → Tiling allows repeated reads of the same data from shared memory
// → Dramatically reduces global memory access
// → Measured speedup of 5-10x
```

### 3.3 CUDA Streams and Asynchronous Execution

```cuda
// CUDA Streams: Overlapping kernel execution and data transfers

// Without streams (sequential execution):
// [H→D Transfer] → [Kernel Execution] → [D→H Transfer]
// Total time = transfer1 + compute + transfer2

// With streams (pipelined):
// Stream 0: [H→D Transfer(0)] → [Kernel(0)] → [D→H Transfer(0)]
// Stream 1:    [H→D Transfer(1)] → [Kernel(1)] → [D→H Transfer(1)]
// Stream 2:       [H→D Transfer(2)] → [Kernel(2)] → [D→H Transfer(2)]
// → Transfers and computation overlap, reducing total time

cudaStream_t stream[3];
for (int i = 0; i < 3; i++) {
    cudaStreamCreate(&stream[i]);
}

for (int i = 0; i < 3; i++) {
    int offset = i * chunkSize;
    // Asynchronous H→D transfer on each stream
    cudaMemcpyAsync(d_A + offset, h_A + offset,
                    chunkSize * sizeof(float),
                    cudaMemcpyHostToDevice, stream[i]);
    // Kernel execution
    kernel<<<grid, block, 0, stream[i]>>>(d_A + offset, d_C + offset);
    // D→H transfer
    cudaMemcpyAsync(h_C + offset, d_C + offset,
                    chunkSize * sizeof(float),
                    cudaMemcpyDeviceToHost, stream[i]);
}

// Wait for all streams to complete
cudaDeviceSynchronize();
```

### 3.4 When GPU Computing is Advantageous

```
GPU Advantages:
  ■ High data parallelism (same operation applied to massive data)
  ■ High arithmetic intensity (more computation than memory access)
  ■ Few branches (threads do not diverge on conditions)

  Examples: Matrix multiplication, image processing, physics simulation, AI training

GPU Disadvantages:
  □ Sequential processing (chained computations dependent on prior results)
  □ Many branches (complex if/else)
  □ Small data (parallelization overhead exceeds benefit)
  □ Random memory access (coalesced memory access is ineffective)

  Examples: File processing, text analysis, tree traversal

Arithmetic Intensity:
  Arithmetic Intensity = FLOPs / Memory Access Bytes

  For RTX 4090:
  Compute performance: 82.6 TFLOPS
  Memory bandwidth: 1,008 GB/s
  Balance point: 82.6 / 1.008 ≈ 82 FLOP/byte

  Intensity > 82: Compute-bound (GPU compute is the limit)
  Intensity < 82: Memory-bound (memory bandwidth is the limit)

  Arithmetic intensity of representative workloads:
  │ Workload              │ Intensity (FLOP/byte) │ Bottleneck     │
  │───────────────────────│───────────────────────│────────────────│
  │ Vector addition        │ 0.25                  │ Memory-bound   │
  │ Convolution (3x3)      │ 4.5                   │ Memory-bound   │
  │ Matrix multiply (large)│ N/3 (proportional to N)│ Compute-bound │
  │ Transformer Attention  │ ~10-50                │ Mixed          │
  │ GEMM (FP16)            │ ~100+                 │ Compute-bound  │
```

---

## 4. GPU Generational Evolution

| Generation | Year | Architecture | CUDA Cores | Key Feature |
|------|-----|-------------|-----------|------|
| Tesla | 2006 | G80 | 128 | CUDA introduced |
| Fermi | 2010 | GF100 | 512 | ECC, L1/L2 cache |
| Kepler | 2012 | GK110 | 2,880 | Dynamic Parallelism |
| Maxwell | 2014 | GM200 | 3,072 | 2x power efficiency |
| Pascal | 2016 | GP100 | 3,840 | FP16, NVLink |
| Volta | 2017 | GV100 | 5,120 | **Tensor Core introduced** |
| Turing | 2018 | TU102 | 4,608 | RT Core (ray tracing) |
| Ampere | 2020 | GA102 | 10,752 | TF32, structured sparsity |
| Hopper | 2022 | GH100 | 16,896 | Transformer Engine |
| Ada | 2022 | AD102 | 16,384 | DLSS 3 |
| Blackwell | 2024 | GB202 | ~21,760 | FP4, 5th gen Tensor Core |

### 4.1 Data Center GPU Evolution

```
NVIDIA Data Center GPU Key Specifications:

  │ GPU      │ Year │ FP16 TFLOPS │ VRAM      │ Mem BW    │ TDP  │ Interconnect │
  │──────────│──────│─────────────│───────────│──────────│──────│──────────────│
  │ V100     │ 2017 │ 125         │ 32GB HBM2 │ 900 GB/s │ 300W │ NVLink 2    │
  │ A100     │ 2020 │ 312         │ 80GB HBM2e│ 2.0 TB/s │ 400W │ NVLink 3    │
  │ H100     │ 2022 │ 989         │ 80GB HBM3 │ 3.35 TB/s│ 700W │ NVLink 4    │
  │ H200     │ 2024 │ 989         │ 141GB HBM3e│4.8 TB/s │ 700W │ NVLink 4    │
  │ B100     │ 2024 │ 1,800       │ 192GB HBM3e│8.0 TB/s │ 700W │ NVLink 5    │
  │ B200     │ 2024 │ 2,250       │ 192GB HBM3e│8.0 TB/s │1000W │ NVLink 5    │
  │ GB200    │ 2025 │ 4,500+      │ 384GB HBM3e│16 TB/s  │1200W │ NVLink 5    │

  H100 vs A100 Improvements:
  - FP16: 3.2x faster
  - FP8: 6.4x faster (newly added in Hopper)
  - Transformer Engine: Automatic FP8/FP16 switching
  - HBM3: 1.7x bandwidth
  - NVLink: 1.5x bandwidth (900 GB/s)

  Blackwell Innovations:
  - Dual-die design (chiplet)
  - FP4 support (2x inference efficiency)
  - 5th generation Tensor Core
  - NVLink 5: 1.8 TB/s
```

### 4.2 HBM (High Bandwidth Memory)

```
GDDR vs HBM Comparison:

  GDDR6X (Consumer GPUs):
  ┌──────────┐
  │ GPU Chip  │
  │          │── PCB traces ──→ [GDDR6X] [GDDR6X] ...
  │          │                 Chips placed outside the package
  └──────────┘
  Bandwidth: 1,008 GB/s (384-bit, RTX 4090)
  Power: High (long traces, high voltage operation)
  Cost: Low

  HBM3 (Data Center GPUs):
  ┌──────────────────────────┐
  │ Silicon Interposer         │
  │ ┌──────┐ ┌────┐ ┌────┐  │
  │ │ GPU  │ │HBM │ │HBM │  │  ← Chips stacked adjacent to GPU
  │ │      │ │    │ │    │  │
  │ └──────┘ └────┘ └────┘  │
  └──────────────────────────┘
  Bandwidth: 3.35 TB/s (H100) → 8.0 TB/s (B200)
  Power: Low (short traces, low voltage)
  Cost: High (interposer manufacturing cost)

  HBM Structure:
  ┌────┐
  │DRAM│  ← 8-16 DRAM dies stacked
  │DRAM│
  │DRAM│
  │DRAM│
  │base│  ← Logic die (TSV connected)
  └────┘
  TSV (Through-Silicon Via): Microscopic holes through silicon for inter-layer connections
  → Thousands of TSVs enable ultra-high bandwidth (1024-bit width per stack)
```

---

## 5. The GPU as an AI Training Engine

### 5.1 Tensor Cores

```
Regular CUDA Core: Scalar operations (1 operation per clock)
  a = b x c + d  → 1 clock

Tensor Core: Matrix operations (matrix multiply in 1 clock)
  ┌─────┐   ┌─────┐   ┌─────┐
  │ 4x4 │ x │ 4x4 │ = │ 4x4 │  → 1 clock
  │Mat A│   │Mat B│   │Mat C│
  └─────┘   └─────┘   └─────┘
  = 64 FMA (fused multiply-add) operations in 1 clock

  → The essence of AI training is massive matrix multiplication
  → Tensor Cores accelerate AI training by 10-100x

Tensor Core Matrix Sizes by Generation:
  │ Generation │ FP16 Matrix Size │ INT8 Matrix Size │ FP8 Support │
  │────────────│──────────────────│──────────────────│─────────────│
  │ Volta      │ 4x4x4            │ N/A              │ No          │
  │ Turing     │ 4x4x4            │ 8x8x16           │ No          │
  │ Ampere     │ 4x4x4            │ 8x8x16           │ No          │
  │ Hopper     │ 4x4x4            │ 8x8x16           │ Yes         │
  │ Blackwell  │ 4x4x4            │ 8x8x32           │ FP4 too     │
```

### 5.2 Precision Selection

| Precision | Bits | Use Case | Tensor Core Perf |
|------|---------|------|----------------|
| FP64 | 64 | Scientific computing | Baseline |
| FP32 | 32 | General computation | 2x |
| TF32 | 19 | AI training | 8x |
| FP16 | 16 | AI training/inference | 16x |
| BF16 | 16 | AI training (range-focused) | 16x |
| INT8 | 8 | AI inference | 32x |
| FP8 | 8 | AI training (Hopper+) | 64x |
| FP4 | 4 | AI inference (Blackwell+) | 128x |

> Lower precision yields higher speed. AI training uses Mixed Precision to maintain quality while accelerating computation.

### 5.3 Mixed Precision Training

```
How Mixed Precision Training Works:

  Traditional (FP32 only):
  Weights(FP32) → Compute(FP32) → Gradients(FP32) → Update(FP32)
  → Slow, high memory consumption

  Mixed Precision:
  ┌─────────────────────────────────────────────┐
  │ Master Weights: FP32 (precision maintained)  │
  │     ↓ Copy                                   │
  │ FP16 Weights: FP16 (for computation)         │
  │     ↓                                        │
  │ Forward Pass: FP16 (fast via Tensor Cores)   │
  │     ↓                                        │
  │ Loss Computation: FP32 (for precision)       │
  │     ↓                                        │
  │ Backward Pass: FP16 (fast via Tensor Cores)  │
  │     ↓                                        │
  │ Gradients: FP16 → Convert to FP32           │
  │     ↓                                        │
  │ Master Weight Update: FP32                   │
  └─────────────────────────────────────────────┘

  Loss Scaling:
  FP16 minimum normalized value = 6.1e-5
  → Small gradients underflow to 0
  → Solution: Scale the loss by 1024x (Loss Scaling)
  → Gradients are also 1024x, avoiding underflow
  → Divide by 1024 during the update to restore original values
```

```python
# Mixed Precision Training with PyTorch
import torch
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()  # Automatically manages Loss Scaling

for data, target in dataloader:
    data, target = data.cuda(), target.cuda()
    optimizer.zero_grad()

    # autocast: Execute forward/backward pass in FP16
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # GradScaler: Automatically applies Loss Scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Results:
# - Training speed: 1.5-3x faster
# - Memory usage: ~50% reduction
# - Training quality: Nearly equivalent to FP32
```

### 5.4 Transformer Engine (Hopper and Later)

```
How the Transformer Engine Works:

  Traditional mixed precision: Fixed FP16
  Transformer Engine: Automatically switches between FP8/FP16

  Monitors tensor statistics (max value, distribution) per layer:
  ┌────────────────────────────────────────────┐
  │ Layer N output tensor:                      │
  │ - Value range representable in FP8 → Compute in FP8 │
  │ - Value range exceeds FP8 → Fall back to FP16       │
  │ - Retry FP8 on the next iteration           │
  └────────────────────────────────────────────┘

  FP8 Formats:
  E4M3: 4-bit exponent + 3-bit mantissa (training-oriented, precision-focused)
  E5M2: 5-bit exponent + 2-bit mantissa (gradient-oriented, range-focused)

  Results:
  - H100 + Transformer Engine: Up to 9x training speedup vs A100
  - GPT-3 class model training becomes practical in terms of time
```

---

## 6. GPU Interconnect Technology

### 6.1 NVLink and NVSwitch

```
GPU Interconnect Comparison:

  PCIe 5.0 x16:
    Bandwidth: 63 GB/s (126 GB/s bidirectional)
    → Insufficient bandwidth relative to GPU compute speed

  NVLink 4 (Hopper):
    Bandwidth: 900 GB/s (18 links x 50 GB/s)
    → Approximately 7x PCIe
    → Enables direct memory read/write between GPUs

  NVLink 5 (Blackwell):
    Bandwidth: 1.8 TB/s
    → 2x vs Hopper

NVSwitch All-to-All Connectivity:

  PCIe Connection (Star Topology):
  ┌──────┐
  │ CPU  │── GPU0
  │      │── GPU1
  │      │── GPU2
  │      │── GPU3
  └──────┘
  → GPU-to-GPU communication goes through CPU (slow)

  NVSwitch Connection (Full Mesh):
  ┌──────┐   NVLink   ┌──────┐
  │ GPU0 │←──────────→│ GPU1 │
  └──┬───┘            └──┬───┘
     │ NVLink             │ NVLink
     │    ┌──────────┐    │
     │    │ NVSwitch │    │
     │    └──────────┘    │
     │ NVLink             │ NVLink
  ┌──┴───┐            ┌──┴───┐
  │ GPU2 │←──────────→│ GPU3 │
  └──────┘   NVLink   └──────┘
  → All GPUs directly connected via NVLink
  → 8-GPU configuration: 14.4 TB/s total bidirectional bandwidth

DGX H100 (8x H100):
  ┌───────────────────────────────────────────┐
  │ 4x NVSwitch                                │
  │                                             │
  │ ┌────┐ ┌────┐ ┌────┐ ┌────┐              │
  │ │H100│ │H100│ │H100│ │H100│              │
  │ └────┘ └────┘ └────┘ └────┘              │
  │ ┌────┐ ┌────┐ ┌────┐ ┌────┐              │
  │ │H100│ │H100│ │H100│ │H100│              │
  │ └────┘ └────┘ └────┘ └────┘              │
  │                                             │
  │ GPU-GPU bandwidth: 900 GB/s (each direction)│
  │ Total GPU memory: 640 GB (8 x 80GB)        │
  │ Total FP8 performance: 32 PFLOPS            │
  │ Power consumption: ~10kW                    │
  │ Price: ~$350,000                            │
  └───────────────────────────────────────────┘
```

### 6.2 Multi-Node Communication

```
Inter-Cluster GPU Communication:

  Intra-node: NVLink/NVSwitch (900 GB/s)
  Inter-node: InfiniBand/RoCE

  InfiniBand Generations:
  │ Generation │ BW (per port)   │ Latency    │
  │────────────│─────────────────│────────────│
  │ FDR        │ 56 Gbps         │ 700 ns     │
  │ EDR        │ 100 Gbps        │ 600 ns     │
  │ HDR        │ 200 Gbps        │ 500 ns     │
  │ NDR        │ 400 Gbps        │ 500 ns     │
  │ XDR        │ 800 Gbps        │ TBD        │

  NCCL (NVIDIA Collective Communications Library):
  A library that optimizes collective communication between GPUs
  - AllReduce: Average gradients across all GPUs
  - AllGather: Distribute data to all GPUs
  - ReduceScatter: Distribute and aggregate

  Communication Patterns:
  Ring AllReduce (bandwidth-optimal):
  GPU0 → GPU1 → GPU2 → GPU3 → GPU0
  → Each GPU passes and aggregates 1/(N-1) of the data in sequence
  → Bandwidth efficiency with N GPUs: (N-1)/N (near-linear scaling)
```

---

## 7. Non-GPU Accelerators

| Device | Developer | Features | Use Case |
|---------|--------|------|------|
| **TPU** | Google | Matrix operation dedicated, large-scale training | Google internal AI |
| **NPU** | Apple/Qualcomm | Low-power AI inference | Smartphones/PCs |
| **FPGA** | Intel/AMD | Programmable circuits | Custom processing |
| **IPU** | Graphcore | Optimized for graph structures | GNN, AI training |
| **Trainium** | AWS | Training-dedicated chip | AWS SageMaker |
| **Inferentia** | AWS | Inference-dedicated chip | AWS inference services |

### 7.1 Detailed Accelerator Comparison

```
GPU vs TPU vs Dedicated ASIC Comparison:

  NVIDIA H100 (GPU):
  ┌──────────────────────────────────────┐
  │ Versatility: High (CUDA ecosystem)   │
  │ FP16: 989 TFLOPS                     │
  │ INT8: 1,979 TOPS                     │
  │ VRAM: 80GB HBM3                      │
  │ Power: 700W                          │
  │ Pros: Supports a wide range of models│
  │ Cons: Expensive, high power draw     │
  └──────────────────────────────────────┘

  Google TPU v5e (TPU):
  ┌──────────────────────────────────────┐
  │ Versatility: Medium (JAX/TF focused) │
  │ BF16: 197 TFLOPS                     │
  │ INT8: 393 TOPS                       │
  │ HBM: 16GB HBM2e                      │
  │ Power: ~200W                         │
  │ Pros: Cost-efficient, Google Cloud   │
  │ Cons: No CUDA, limited PyTorch support│
  └──────────────────────────────────────┘

  Apple Neural Engine (NPU):
  ┌──────────────────────────────────────┐
  │ Versatility: Low (Apple exclusive)   │
  │ Performance: 38 TOPS (M4 Pro)        │
  │ Memory: Shared from unified memory   │
  │ Power: ~a few watts                  │
  │ Pros: Ultra-low power, always-on     │
  │ Cons: No training, Apple ecosystem only │
  └──────────────────────────────────────┘
```

---

## 8. Theory of Parallel Computing

### 8.1 Amdahl's Law

```
Amdahl's Law:

  Speedup = 1 / ((1-P) + P/N)

  P = Fraction that can be parallelized
  N = Number of processors

  Example: P=0.95 (95% parallelizable), N=1024 cores
  Speedup = 1 / (0.05 + 0.95/1024) = 1 / 0.0509 ≈ 19.6x

  ★ Even with 95% parallelizable code, the maximum speedup is only 19.6x
  ★ The 5% sequential portion is the bottleneck

  Parallelization ratio and maximum speedup:
  P=50%   → max 2x    (adding more cores is pointless)
  P=90%   → max 10x
  P=95%   → max 20x
  P=99%   → max 100x
  P=99.9% → max 1000x ← AI training falls in this range
```

### 8.2 Gustafson's Law

```
Gustafson's Law:

  Corrects Amdahl's Law assumption of "fixed problem size."
  "With more cores, you can solve a larger problem in the same time."

  Speedup = N - (1-P) x (N-1)

  → If you scale the problem size, parallelization benefits are nearly linear
  → This is the true value of GPU computing:
     "You can train a larger model in the same amount of time"
```

### 8.3 Roofline Model

```
Roofline Model:

  Achievable Performance = min(Peak Compute, Peak Memory Bandwidth x Arithmetic Intensity)

  For RTX 4090:
  Peak compute: 82.6 TFLOPS (FP32)
  Peak memory bandwidth: 1,008 GB/s

                Performance (TFLOPS)
                    │
  82.6 TFLOPS ──────┼──────────────── Compute-bound
                    │              /
                    │            /
                    │          /   ← Memory-bound
                    │        /       (bandwidth x intensity)
                    │      /
                    │    /
                    │  /
                    │/
                    └───────────────── Arithmetic Intensity (FLOP/byte)
                         82

  Optimization Directions:
  - Memory-bound → Improve data reuse (tiling, cache utilization)
  - Compute-bound → Improve algorithm efficiency, lower precision (FP16/FP8)
  - Both walls → Upgrade to more powerful hardware

  Practical Application:
  1. First, calculate the arithmetic intensity
  2. Determine the theoretical upper bound via the Roofline Model
  3. The gap between measured and theoretical values is the optimization opportunity
  → NVIDIA Nsight Compute can calculate this automatically
```

---

## 9. Distributed Training Methods

### 9.1 Data Parallelism and Model Parallelism

```
Data Parallelism:
  ┌──────────────────────────────────────────────┐
  │ Copy the same model to all GPUs               │
  │                                                │
  │ GPU 0: Model + Data[0:N/4]    → Gradients 0   │
  │ GPU 1: Model + Data[N/4:N/2]  → Gradients 1   │
  │ GPU 2: Model + Data[N/2:3N/4] → Gradients 2   │
  │ GPU 3: Model + Data[3N/4:N]   → Gradients 3   │
  │                                                │
  │ AllReduce: Average all gradients               │
  │ → Synchronously update models on all GPUs      │
  │                                                │
  │ Applicable: When the model fits in 1 GPU's VRAM│
  │ Scalability: Good (communication is bottleneck)│
  └──────────────────────────────────────────────┘

Model Parallelism:
  ┌──────────────────────────────────────────────┐
  │ Split the model across multiple GPUs           │
  │                                                │
  │ Tensor Parallelism:                            │
  │ GPU 0: First half of Layer N weights           │
  │ GPU 1: Second half of Layer N weights          │
  │ → One layer computed across multiple GPUs      │
  │ → High communication frequency (sync at each   │
  │   layer's input/output)                        │
  │                                                │
  │ Pipeline Parallelism:                          │
  │ GPU 0: Layers 0-9                              │
  │ GPU 1: Layers 10-19                            │
  │ GPU 2: Layers 20-29                            │
  │ GPU 3: Layers 30-39                            │
  │ → Micro-batches processed in pipeline fashion  │
  │ → Bubbles (idle time) occur                    │
  │                                                │
  │ Applicable: Very large models (GPT-4, Llama 3, etc.) │
  └──────────────────────────────────────────────┘

3D Parallelism (Data x Tensor x Pipeline):
  → Large language model training combines all three
  → Implemented by frameworks such as Megatron-LM, DeepSpeed, etc.
```

### 9.2 ZeRO (Zero Redundancy Optimizer)

```
ZeRO's 3 Stages:

  Standard data parallelism:
  Each GPU holds: Model weights + gradients + optimizer state
  Memory usage: 16 x phi x N (phi = parameter count, N = GPU count)

  ZeRO Stage 1: Partition optimizer state
  Each GPU holds: Model weights + gradients + 1/N of optimizer state
  Memory reduction: Up to 4x

  ZeRO Stage 2: Also partition gradients
  Each GPU holds: Model weights + 1/N of gradients + 1/N of optimizer state
  Memory reduction: Up to 8x

  ZeRO Stage 3: Also partition weights
  Each GPU holds: 1/N of model weights + 1/N of gradients + 1/N of optimizer state
  Memory reduction: Up to Nx (proportional to GPU count)

  → Enables training a 10B parameter model on 4x A100 GPUs
```

---

## 10. Hands-On Exercises

### Exercise 1: CPU vs GPU Determination (Fundamentals)

For each of the following tasks, determine whether CPU or GPU is better suited and explain your reasoning:
1. Applying the same filter to 1 million images
2. Parsing and transforming JSON files
3. Neural network training
4. Git diff computation
5. Video encoding

### Exercise 2: Amdahl's Law (Intermediate)

Given the following breakdown of a program's execution time:
- Input processing: 5% (sequential)
- Main computation: 80% (parallelizable)
- Result aggregation: 10% (partially parallelizable, 50%)
- Output processing: 5% (sequential)

Calculate the maximum speedup using a 16-core CPU and a 1024-core GPU.

### Exercise 3: GPU Utilization (Advanced)

Create a benchmark using PyTorch that compares CPU vs GPU matrix multiplication speed:
```python
import torch
import time

# Measure with varying matrix sizes: 100, 500, 1000, 5000, 10000
for N in [100, 500, 1000, 5000, 10000]:
    # CPU
    a_cpu = torch.randn(N, N)
    b_cpu = torch.randn(N, N)
    start = time.time()
    c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.time() - start

    # GPU (CUDA or MPS)
    # ... implement and compare
```

### Exercise 4: Memory Bandwidth Calculations (Intermediate)

For the H100 GPU (80GB HBM3, 3.35 TB/s bandwidth), calculate the following:
1. Minimum time to read all VRAM
2. Minimum bandwidth required for a forward pass of a 70B parameter model (FP16)
3. Estimated memory usage for training with batch size 32

### Exercise 5: Distributed Training Design (Advanced)

Design a large language model training system under the following conditions:
- Model: 13B parameters, Transformer
- GPUs: A100 80GB x 16 (2 nodes x 8 GPUs)
- Goal: Complete training on 1T tokens within 30 days

Design Items:
1. Parallelization strategy (data/tensor/pipeline)
2. Batch size and accumulation
3. Precision (FP16/BF16/FP8)
4. Identify and mitigate communication bottlenecks
5. Estimate required compute (FLOPS)


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|--------|------|--------|
| Initialization error | Configuration file issues | Verify config file path and format |
| Timeout | Network latency / insufficient resources | Adjust timeout values, add retry logic |
| Out of memory | Growing data volume | Implement batch processing, add pagination |
| Permission error | Insufficient access rights | Verify executing user permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Implement locking mechanisms, manage transactions |

### Debugging Procedure

1. **Check error messages**: Read the stack trace to identify the point of failure
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify step by step**: Use logging output or a debugger to verify hypotheses
5. **Fix and regression test**: After fixing, also run tests on related areas

```python
# Debugging utility
import logging
import traceback
from functools import wraps

# Logger configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """Decorator that logs function input/output"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Call: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """Data processing (debug target)"""
    if not items:
        raise ValueError("Empty data")
    return [item * 2 for item in items]
```

### Diagnosing Performance Issues

Steps for diagnosing performance issues:

1. **Identify bottlenecks**: Measure with profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O waits**: Verify disk and network I/O status
4. **Check connection count**: Verify connection pool status

| Problem Type | Diagnostic Tool | Solution |
|-----------|-----------|------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference release |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

A summary of decision criteria for technology selection:

| Criterion | When to Prioritize | When Compromise is Acceptable |
|---------|------------|-------------|
| Performance | Real-time processing, large-scale data | Admin dashboards, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Services expected to grow | Internal tools, fixed user base |
| Security | Personal data, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality-focused, mission-critical |

### Architecture Pattern Selection

```
┌─────────────────────────────────────────────────┐
│          Architecture Selection Flow             │
├─────────────────────────────────────────────────┤
│                                                 │
│  (1) Team size?                                 │
│    ├─ Small (1-5 people) → Monolith             │
│    └─ Large (10+ people) → Go to (2)            │
│                                                 │
│  (2) Deployment frequency?                       │
│    ├─ Once a week or less → Monolith + modular  │
│    └─ Daily / multiple times → Go to (3)        │
│                                                 │
│  (3) Inter-team independence?                    │
│    ├─ High → Microservices                       │
│    └─ Moderate → Modular monolith                │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs Long-term Costs**
- A short-term fast approach may become technical debt in the long run
- Conversely, over-engineering incurs high short-term costs and delays the project

**2. Consistency vs Flexibility**
- A unified technology stack has lower learning costs
- Adopting diverse technologies enables best-fit solutions but increases operational costs

**3. Level of Abstraction**
- High abstraction improves reusability but can make debugging harder
- Low abstraction is intuitive but leads to code duplication

```python
# Design decision record template
class ArchitectureDecisionRecord:
    """Create an ADR (Architecture Decision Record)"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """Describe background and challenges"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """Describe the decision"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """Add a consequence"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """Add a rejected alternative"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Output in Markdown format"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## Context\n{self.context}\n\n"
        md += f"## Decision\n{self.decision}\n\n"
        md += "## Consequences\n"
        for c in self.consequences:
            icon = "+" if c['type'] == 'positive' else "!"
            md += f"- [{icon}] {c['description']}\n"
        md += "\n## Rejected Alternatives\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## Practical Application Scenarios

### Scenario 1: MVP Development at a Startup

**Situation:** Need to release a product quickly with limited resources

**Approach:**
- Choose a simple architecture
- Focus on the minimum viable feature set
- Automated tests for critical paths only
- Introduce monitoring early

**Lessons Learned:**
- Do not pursue perfection (YAGNI principle)
- Obtain user feedback early
- Manage technical debt deliberately

### Scenario 2: Legacy System Modernization

**Situation:** Incrementally modernizing a system that has been in operation for 10+ years

**Approach:**
- Gradual migration using the Strangler Fig pattern
- Create Characterization Tests first if existing tests are absent
- Coexist old and new systems via an API gateway
- Execute data migration incrementally

| Phase | Work Content | Estimated Duration | Risk |
|---------|---------|---------|--------|
| 1. Investigation | Current state analysis, dependency mapping | 2-4 weeks | Low |
| 2. Foundation | CI/CD setup, test environment | 4-6 weeks | Low |
| 3. Migration Start | Sequential migration from peripheral features | 3-6 months | Medium |
| 4. Core Migration | Migration of core features | 6-12 months | High |
| 5. Completion | Legacy system decommission | 2-4 weeks | Medium |

### Scenario 3: Large-Team Development

**Situation:** 50+ engineers developing the same product

**Approach:**
- Define clear boundaries using Domain-Driven Design
- Assign ownership per team
- Manage shared libraries via Inner Source
- Design API-first to minimize inter-team dependencies

```python
# Inter-team API contract definition
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """Inter-team API contract"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # Response time SLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """Verify SLA compliance"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """Output in OpenAPI format"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# Usage example
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### Scenario 4: Performance-Critical System

**Situation:** A system requiring millisecond-level response times

**Optimization Points:**
1. Caching strategy (L1: in-memory, L2: Redis, L3: CDN)
2. Leverage asynchronous processing
3. Connection pooling
4. Query optimization and index design

| Optimization Technique | Impact | Implementation Cost | Applicable Scenario |
|-----------|------|-----------|---------|
| In-memory cache | High | Low | Frequently accessed data |
| CDN | High | Low | Static content |
| Async processing | Medium | Medium | I/O-heavy operations |
| DB optimization | High | High | Slow queries |
| Code optimization | Low-Medium | High | CPU-bound cases |

---

## Team Development Practices

### Code Review Checklist

Key points to verify during code reviews related to this topic:

- [ ] Naming conventions are consistent
- [ ] Error handling is appropriate
- [ ] Test coverage is sufficient
- [ ] There is no negative performance impact
- [ ] There are no security issues
- [ ] Documentation is updated

### Knowledge Sharing Best Practices

| Method | Frequency | Audience | Impact |
|------|------|------|------|
| Pair programming | As needed | Complex tasks | Immediate feedback |
| Tech talks | Weekly | Entire team | Horizontal knowledge spread |
| ADR (Decision Records) | As needed | Future members | Decision transparency |
| Retrospectives | Biweekly | Entire team | Continuous improvement |
| Mob programming | Monthly | Critical design | Consensus building |

### Technical Debt Management

```
Priority Matrix:

        Impact High
          │
    ┌─────┼─────┐
    │ Plan │ Act  │
    │ for  │ on   │
    │ later│ now  │
    ├─────┼─────┤
    │ Log  │ Next │
    │ only │Sprint│
    │      │      │
    └─────┼─────┘
          │
        Impact Low
    Frequency Low  Frequency High
```

---

## Security Considerations

### Common Vulnerabilities and Countermeasures

| Vulnerability | Risk Level | Countermeasure | Detection Method |
|--------|------------|------|---------|
| Injection attacks | High | Input validation, parameterized queries | SAST/DAST |
| Authentication flaws | High | Multi-factor auth, session management hardening | Penetration testing |
| Sensitive data exposure | High | Encryption, access control | Security audit |
| Misconfiguration | Medium | Security headers, principle of least privilege | Configuration scanning |
| Insufficient logging | Medium | Structured logs, audit trails | Log analysis |

### Secure Coding Best Practices

```python
# Secure coding example
import hashlib
import secrets
import hmac
from typing import Optional

class SecurityUtils:
    """Security utilities"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate a cryptographically secure token"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """Hash a password"""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        )
        return hashed.hex(), salt

    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """Verify a password"""
        new_hash, _ = SecurityUtils.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)

    @staticmethod
    def sanitize_input(value: str) -> str:
        """Sanitize input values"""
        dangerous_chars = ['<', '>', '"', "'", '&', '\\']
        result = value
        for char in dangerous_chars:
            result = result.replace(char, '')
        return result.strip()

# Usage example
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
```

### Security Checklist

- [ ] All input values are validated
- [ ] Sensitive information is not written to logs
- [ ] HTTPS is enforced
- [ ] CORS policy is properly configured
- [ ] Dependency vulnerability scanning is performed
- [ ] Error messages do not contain internal information

---

## Migration Guide

### Notes on Version Upgrades

| Version | Key Changes | Migration Work | Impact Scope |
|-----------|-----------|---------|---------|
| v1.x → v2.x | API design overhaul | Endpoint changes | All clients |
| v2.x → v3.x | Authentication method change | Token format update | Auth-related |
| v3.x → v4.x | Data model change | Run migration scripts | DB-related |

### Step-by-Step Migration Procedure

```python
# Migration script template
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)

class MigrationRunner:
    """Step-by-step migration execution engine"""

    def __init__(self, migration_dir: str):
        self.migration_dir = Path(migration_dir)
        self.migrations: List[Dict] = []
        self.completed: List[str] = []

    def register(self, version: str, description: str,
                 up: Callable, down: Callable):
        """Register a migration"""
        self.migrations.append({
            'version': version,
            'description': description,
            'up': up,
            'down': down,
            'registered_at': datetime.now().isoformat()
        })

    def run_up(self, target_version: str = None):
        """Execute migrations (upgrade)"""
        for migration in self.migrations:
            if migration['version'] in self.completed:
                continue
            logger.info(f"Running: {migration['version']} - "
                       f"{migration['description']}")
            try:
                migration['up']()
                self.completed.append(migration['version'])
                logger.info(f"Completed: {migration['version']}")
            except Exception as e:
                logger.error(f"Failed: {migration['version']}: {e}")
                raise
            if target_version and migration['version'] == target_version:
                break

    def run_down(self, target_version: str):
        """Rollback migrations"""
        for migration in reversed(self.migrations):
            if migration['version'] not in self.completed:
                continue
            if migration['version'] == target_version:
                break
            logger.info(f"Rolling back: {migration['version']}")
            migration['down']()
            self.completed.remove(migration['version'])

    def status(self) -> Dict:
        """Check migration status"""
        return {
            'total': len(self.migrations),
            'completed': len(self.completed),
            'pending': len(self.migrations) - len(self.completed),
            'versions': {
                m['version']: 'completed'
                if m['version'] in self.completed else 'pending'
                for m in self.migrations
            }
        }
```

### Rollback Plan

Always prepare a rollback plan for migration work:

1. **Data backup**: Take a complete backup before migration
2. **Test environment verification**: Validate in a production-equivalent environment beforehand
3. **Gradual rollout**: Deploy incrementally using canary releases
4. **Enhanced monitoring**: Shorten metrics monitoring intervals during migration
5. **Clear criteria**: Define rollback decision criteria in advance
---

## FAQ

### Q1: What is the difference between gaming GPUs and AI training GPUs?

**A**: Even with the same GPU chip, memory and Tensor Core emphasis differ by use case:
- **Gaming (RTX 4090)**: 24GB VRAM, RT Core emphasis, consumer-oriented
- **AI Training (A100/H100)**: 80GB VRAM, Tensor Core emphasis, ECC-enabled
- **AI Inference (L4/L40S)**: Low power consumption, INT8/FP8 optimized

### Q2: Can Apple Silicon GPUs compete with NVIDIA?

**A**: It depends on the use case:
- **Memory Bandwidth**: M4 Max (546GB/s) vs RTX 4090 (1,008GB/s) → NVIDIA advantage
- **Unified Memory**: M4 Max's 128GB directly accessible by GPU → Advantage for large LLM inference
- **Power Efficiency**: Apple Silicon is overwhelmingly superior
- **CUDA Ecosystem**: Majority of AI/ML libraries assume CUDA → NVIDIA advantage

### Q3: Should I learn GPU programming?

**A**: Essential if you work in AI/ML, game development, or HPC (High Performance Computing). Web developers do not need it directly, but understanding "why AI training requires GPUs" is important. PyTorch/TensorFlow abstract GPU details away, so there is rarely a need to write CUDA directly.

### Q4: What can I do when VRAM is insufficient?

**A**: Combine the following techniques:
- **Gradient Checkpointing**: Recompute intermediate results instead of storing them
- **Mixed Precision**: Halve usage with FP16/BF16
- **ZeRO Stage 3**: Distribute weights across multiple GPUs
- **Offloading**: Offload portions to CPU RAM/NVMe SSD
- **Quantization**: INT4/INT8 for inference
- **Model Sharding**: Place different layers on different GPUs

### Q5: What should I do about high GPU temperatures?

**A**: GPU temperature management guidelines:
- Below 80C: Normal
- 80-90C: Caution (thermal throttling may occur)
- Above 90C: Action needed
Countermeasures:
- Improve case airflow
- Adjust GPU fan curve (MSI Afterburner, etc.)
- Replace thermal pads
- Undervolt (reduce voltage to suppress heat, minor performance reduction)

---

## Summary

| Concept | Key Point |
|------|---------|
| CPU vs GPU | Latency-optimized vs throughput-optimized |
| CUDA Core | General-purpose parallel computing. Thousands to tens of thousands of cores |
| Tensor Core | Matrix operation dedicated. Accelerates AI training 10-100x |
| Warp | 32-thread synchronized execution unit. Divergence on branches |
| Amdahl's Law | The sequential portion determines the limit of parallelization |
| Precision | Lower precision (FP32→FP16→FP8) yields higher speed (quality trade-off) |
| NVLink | High-bandwidth GPU interconnect. Essential for multi-GPU training |
| Distributed Training | Combination of data/tensor/pipeline parallelism |

---

## Recommended Next Guides


---

## References

1. Kirk, D. B. & Hwu, W. W. "Programming Massively Parallel Processors." 4th Edition, Morgan Kaufmann, 2022.
2. NVIDIA. "CUDA C++ Programming Guide." https://docs.nvidia.com/cuda/
3. Hennessy, J. L. & Patterson, D. A. "Computer Architecture: A Quantitative Approach." 6th Edition.
4. NVIDIA. "NVIDIA H100 Tensor Core GPU Architecture." Whitepaper, 2022.
5. Amdahl, G. M. "Validity of the single processor approach to achieving large scale computing capabilities." AFIPS, 1967.
6. Rajbhandari, S. et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC, 2020.
7. Narayanan, D. et al. "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM." SC, 2021.
8. Williams, S. et al. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." CACM, 2009.
9. NVIDIA. "NVIDIA Blackwell Architecture Technical Brief." 2024.
10. Micikevicius, P. et al. "Mixed Precision Training." ICLR, 2018.

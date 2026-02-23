# GPUと並列計算

> GPUは「一つの命令を数千のコアで同時実行する」— CPU的な逐次処理とは根本的に異なるアプローチである。

## この章で学ぶこと

- [ ] GPUの内部アーキテクチャとCPUとの違いを説明できる
- [ ] GPGPU（汎用GPU計算）の仕組みを理解する
- [ ] AI/MLにおけるGPUの役割を説明できる
- [ ] CUDAプログラミングの基本パターンを理解する
- [ ] GPU間通信技術（NVLink、NVSwitch）を説明できる
- [ ] マルチGPU構成と分散学習の基礎を習得する

## 前提知識

- CPUアーキテクチャの基礎 → 参照: [[00-cpu-architecture.md]]

---

## 1. GPU vs CPU

### 1.1 設計思想の違い

```
CPU: レイテンシ最適化（1タスクを最速で）
  ┌──────────────────────────────────────┐
  │ ┌────────────────────┐ ┌──────────┐ │
  │ │ 大きなキャッシュ    │ │ 分岐予測 │ │  ← 複雑な制御ロジック
  │ │ (L1: 64KB×2)       │ │ 器       │ │  ← 少数の強力なコア
  │ │ (L2: 1MB)          │ └──────────┘ │
  │ │ (L3: 32MB)         │              │
  │ └────────────────────┘              │
  │ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐│
  │ │Core 0│ │Core 1│ │Core 2│ │Core 3││  ← 4-16コア
  │ │(超強力)│ │(超強力)│ │(超強力)│ │(超強力)│
  │ └──────┘ └──────┘ └──────┘ └──────┘│
  └──────────────────────────────────────┘

GPU: スループット最適化（大量タスクを並列で）
  ┌──────────────────────────────────────┐
  │ ┌────┐                   小キャッシュ │
  │ │制御│                               │  ← 単純な制御ロジック
  │ └────┘                               │  ← 数千の小さなコア
  │ ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐  │
  │ │c││c││c││c││c││c││c││c││c││c│  │  ← SM(ストリーミング
  │ └─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘  │    マルチプロセッサ)内
  │ ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐  │    に32-128コア
  │ │c││c││c││c││c││c││c││c││c││c│  │
  │ └─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘  │  ← SMが数十〜百個
  │ ... (数千コア)                       │
  └──────────────────────────────────────┘
```

### 1.2 数値比較

| 指標 | CPU (Ryzen 9 7950X) | GPU (RTX 4090) |
|------|---------------------|----------------|
| コア数 | 16 | 16,384 (CUDA) |
| クロック | 5.7 GHz | 2.5 GHz |
| FP32性能 | 〜1.5 TFLOPS | 82.6 TFLOPS |
| メモリ帯域 | 83 GB/s (DDR5) | 1,008 GB/s (GDDR6X) |
| TDP | 170W | 450W |
| 得意分野 | 逐次処理、分岐多い処理 | 大量データの並列処理 |
| 苦手分野 | 大規模並列 | 分岐多い処理、逐次処理 |

### 1.3 トランジスタの使われ方の違い

```
CPUのダイ面積配分:
  ┌──────────────────────────────────────┐
  │ [     キャッシュ: 40%     ]          │
  │ [制御ロジック: 30%]                  │
  │ [分岐予測: 10%]                      │
  │ [演算ユニット: 15%]                  │
  │ [その他: 5%]                         │
  └──────────────────────────────────────┘
  → 面積の大部分がデータの取得と制御に使われ、
    実際の演算に使われるのは15%程度

GPUのダイ面積配分:
  ┌──────────────────────────────────────┐
  │ [     演算ユニット: 60%     ]        │
  │ [キャッシュ: 15%]                    │
  │ [制御ロジック: 10%]                  │
  │ [メモリコントローラ: 10%]            │
  │ [その他: 5%]                         │
  └──────────────────────────────────────┘
  → 面積の大部分が演算に使われる
  → 制御は最小限、その分スレッドを大量に走らせてレイテンシを隠蔽
```

---

## 2. GPUアーキテクチャ

### 2.1 NVIDIA GPU の構造

```
NVIDIA GPU（Ada Lovelace アーキテクチャ）:

  ┌─────────────────────────────────────────────┐
  │                  GPU チップ                   │
  │                                               │
  │  ┌─────────────────────────────────────────┐ │
  │  │  GPC (Graphics Processing Cluster) × 12 │ │
  │  │  ┌──────────────────────────────────┐   │ │
  │  │  │  TPC (Texture Processing Cluster) │   │ │
  │  │  │  ┌────────────────────────────┐  │   │ │
  │  │  │  │  SM (Streaming Multiprocessor)│ │   │ │
  │  │  │  │  ┌─────────────────────────┐│ │   │ │
  │  │  │  │  │ CUDA Core × 128        ││ │   │ │
  │  │  │  │  │ Tensor Core × 4        ││ │   │ │
  │  │  │  │  │ RT Core × 1            ││ │   │ │
  │  │  │  │  │ 共有メモリ: 128KB      ││ │   │ │
  │  │  │  │  │ レジスタファイル: 256KB ││ │   │ │
  │  │  │  │  │ L1 Cache: 128KB        ││ │   │ │
  │  │  │  │  └─────────────────────────┘│ │   │ │
  │  │  │  └────────────────────────────┘  │   │ │
  │  │  └──────────────────────────────────┘   │ │
  │  └─────────────────────────────────────────┘ │
  │                                               │
  │  ┌─────────────────┐  ┌────────────────────┐│
  │  │  L2 Cache: 72MB  │  │ メモリコントローラ  ││
  │  └─────────────────┘  │ GDDR6X 24GB       ││
  │                        │ 384-bit バス      ││
  │                        └────────────────────┘│
  └─────────────────────────────────────────────┘

  RTX 4090: 128SM × 128 CUDA Core = 16,384 CUDA Core
```

### 2.2 ワープ（Warp）とスレッド階層

```
NVIDIA GPU のスレッド階層:

  Grid（全体）
  ├── Block 0
  │   ├── Warp 0: Thread 0-31   ← 32スレッドが完全同期実行
  │   ├── Warp 1: Thread 32-63
  │   └── ...
  ├── Block 1
  │   ├── Warp 0: Thread 0-31
  │   └── ...
  └── ...

  ワープ内の全スレッドは同じ命令を同時実行（SIMT: Single Instruction, Multiple Threads）

  → 分岐があると「ワープダイバージェンス」が発生:
  if (threadIdx.x < 16) {
      // 前半16スレッドがここを実行
      // 後半16スレッドは待機（無駄）
  } else {
      // 後半16スレッドがここを実行
      // 前半16スレッドは待機（無駄）
  }
  → GPU で分岐の多いコードが遅い理由
```

### 2.3 GPUメモリ階層

```
GPUメモリ階層の詳細:

  ┌──────────────────────────────────────────────┐
  │ レジスタファイル（各スレッド専用）              │
  │ 容量: 256KB/SM                                │
  │ レイテンシ: 1サイクル                          │
  │ 帯域: 最大                                    │
  ├──────────────────────────────────────────────┤
  │ 共有メモリ（ブロック内で共有）                  │
  │ 容量: 最大100KB/SM（設定可能）                 │
  │ レイテンシ: 〜5サイクル                        │
  │ 帯域: 〜128 bytes/cycle                       │
  │ → 明示的にプログラマが管理する高速メモリ       │
  ├──────────────────────────────────────────────┤
  │ L1 キャッシュ（SM内、自動管理）                │
  │ 容量: 128KB/SM（共有メモリと配分）             │
  │ レイテンシ: 〜30サイクル                       │
  ├──────────────────────────────────────────────┤
  │ L2 キャッシュ（全SM共有）                      │
  │ 容量: 72MB (RTX 4090)                         │
  │ レイテンシ: 〜200サイクル                      │
  ├──────────────────────────────────────────────┤
  │ グローバルメモリ（GDDR6X / HBM）              │
  │ 容量: 24GB (RTX 4090) / 80GB (H100)          │
  │ レイテンシ: 〜400-600サイクル                  │
  │ 帯域: 1,008 GB/s (RTX 4090) / 3.35 TB/s (H100)│
  └──────────────────────────────────────────────┘

  メモリ合体アクセス（Coalesced Memory Access）:
  ┌──────────────────────────────────────────┐
  │ 良い例（合体アクセス）:                     │
  │ Thread 0 → addr[0]                        │
  │ Thread 1 → addr[1]   → 1回のメモリトランザクション │
  │ Thread 2 → addr[2]                        │
  │ ...                                        │
  │ Thread 31 → addr[31]                      │
  │                                            │
  │ 悪い例（非合体アクセス）:                   │
  │ Thread 0 → addr[0]    → 個別のトランザクション │
  │ Thread 1 → addr[128]  → 個別のトランザクション │
  │ Thread 2 → addr[256]  → 個別のトランザクション │
  │ ...                                        │
  │ → 帯域を無駄遣い、性能が10-100倍低下       │
  └──────────────────────────────────────────┘
```

---

## 3. CUDAプログラミング

### 3.1 基本構造

```cuda
// CUDA: ベクトル加算の例
// C = A + B (各要素を並列に計算)

// GPU上で実行されるカーネル関数
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    // 各スレッドが1要素を担当
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];  // 超シンプル: 1スレッド1加算
    }
}

int main() {
    int N = 1000000;  // 100万要素
    float *d_A, *d_B, *d_C;

    // GPUメモリ確保
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // CPU→GPUにデータ転送
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // カーネル起動: 3907ブロック × 256スレッド = 〜100万スレッド
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    // GPU→CPUにデータ転送
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // GPUメモリ解放
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}

// CPU: 100万回のループ → 〜1ms
// GPU: 100万スレッド並列 → 〜0.01ms + メモリ転送コスト
```

### 3.2 共有メモリの活用

```cuda
// 行列乗算: 共有メモリを使った最適化
// C = A × B (N × N 行列)

#define TILE_SIZE 32

__global__ void matMul(float *A, float *B, float *C, int N) {
    // 共有メモリにタイルをロード
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // タイルごとに処理
    for (int t = 0; t < N / TILE_SIZE; t++) {
        // グローバルメモリ → 共有メモリにロード
        tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];

        // 全スレッドのロード完了を待つ
        __syncthreads();

        // 共有メモリから計算（高速）
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}

// なぜ共有メモリが重要か:
// グローバルメモリ: 400-600サイクルのレイテンシ
// 共有メモリ: 5サイクルのレイテンシ
// → タイリングにより同じデータを何度も共有メモリから読む
// → グローバルメモリアクセスを大幅削減
// → 実測で5-10倍の高速化
```

### 3.3 CUDAストリームと非同期実行

```cuda
// CUDAストリーム: カーネル実行とデータ転送の重ね合わせ

// ストリームなし（逐次実行）:
// [H→D転送] → [カーネル実行] → [D→H転送]
// 合計時間 = 転送1 + 計算 + 転送2

// ストリームあり（パイプライン）:
// Stream 0: [H→D転送(0)] → [カーネル(0)] → [D→H転送(0)]
// Stream 1:    [H→D転送(1)] → [カーネル(1)] → [D→H転送(1)]
// Stream 2:       [H→D転送(2)] → [カーネル(2)] → [D→H転送(2)]
// → 転送と計算が重なり、合計時間が短縮

cudaStream_t stream[3];
for (int i = 0; i < 3; i++) {
    cudaStreamCreate(&stream[i]);
}

for (int i = 0; i < 3; i++) {
    int offset = i * chunkSize;
    // 各ストリームで非同期にH→D転送
    cudaMemcpyAsync(d_A + offset, h_A + offset,
                    chunkSize * sizeof(float),
                    cudaMemcpyHostToDevice, stream[i]);
    // カーネル実行
    kernel<<<grid, block, 0, stream[i]>>>(d_A + offset, d_C + offset);
    // D→H転送
    cudaMemcpyAsync(h_C + offset, d_C + offset,
                    chunkSize * sizeof(float),
                    cudaMemcpyDeviceToHost, stream[i]);
}

// 全ストリームの完了を待つ
cudaDeviceSynchronize();
```

### 3.4 GPU計算が有利な条件

```
GPU が有利:
  ■ データ並列性が高い（同じ処理を大量データに適用）
  ■ 演算密度が高い（メモリアクセスより計算が多い）
  ■ 分岐が少ない（条件分岐でスレッドが分散しない）

  例: 行列乗算、画像処理、物理シミュレーション、AI学習

GPU が不利:
  □ 逐次処理（前の結果に依存する連続計算）
  □ 分岐が多い（if/else が複雑）
  □ データが小さい（並列化のオーバーヘッドの方が大きい）
  □ メモリアクセスがランダム（合体メモリアクセスが効かない）

  例: ファイル処理、テキスト解析、ツリー探索

演算強度（Arithmetic Intensity）:
  演算強度 = 浮動小数点演算数 / メモリアクセスバイト数

  RTX 4090 の場合:
  計算性能: 82.6 TFLOPS
  メモリ帯域: 1,008 GB/s
  バランスポイント: 82.6 / 1.008 ≈ 82 FLOP/byte

  演算強度 > 82: 計算律速（GPUの計算能力が限界）
  演算強度 < 82: メモリ律速（メモリ帯域が限界）

  代表的なワークロードの演算強度:
  │ ワークロード        │ 演算強度(FLOP/byte) │ 律速要因   │
  │────────────────────│─────────────────────│────────────│
  │ ベクトル加算         │ 0.25                │ メモリ律速 │
  │ 畳み込み(3×3)       │ 4.5                 │ メモリ律速 │
  │ 行列乗算(大規模)    │ N/3 (Nに比例)       │ 計算律速   │
  │ Transformer Attention│ 〜10-50             │ 混合       │
  │ GEMM (FP16)         │ 〜100+              │ 計算律速   │
```

---

## 4. GPU世代の進化

| 世代 | 年 | アーキテクチャ | CUDA Core | 特徴 |
|------|-----|-------------|-----------|------|
| Tesla | 2006 | G80 | 128 | CUDA登場 |
| Fermi | 2010 | GF100 | 512 | ECC、L1/L2キャッシュ |
| Kepler | 2012 | GK110 | 2,880 | Dynamic Parallelism |
| Maxwell | 2014 | GM200 | 3,072 | 電力効率2倍 |
| Pascal | 2016 | GP100 | 3,840 | FP16、NVLink |
| Volta | 2017 | GV100 | 5,120 | **Tensor Core登場** |
| Turing | 2018 | TU102 | 4,608 | RT Core（レイトレ） |
| Ampere | 2020 | GA102 | 10,752 | TF32、構造化スパース |
| Hopper | 2022 | GH100 | 16,896 | Transformer Engine |
| Ada | 2022 | AD102 | 16,384 | DLSS 3 |
| Blackwell | 2024 | GB202 | 〜21,760 | FP4、第5世代Tensor |

### 4.1 データセンターGPUの進化

```
NVIDIA データセンターGPU の主要スペック:

  │ GPU      │ 年   │ FP16 TFLOPS │ VRAM    │ メモリ帯域│ TDP  │ 相互接続     │
  │──────────│──────│─────────────│─────────│──────────│──────│──────────────│
  │ V100     │ 2017 │ 125         │ 32GB HBM2│ 900 GB/s │ 300W │ NVLink 2    │
  │ A100     │ 2020 │ 312         │ 80GB HBM2e│ 2.0 TB/s│ 400W │ NVLink 3    │
  │ H100     │ 2022 │ 989         │ 80GB HBM3│ 3.35 TB/s│ 700W │ NVLink 4    │
  │ H200     │ 2024 │ 989         │ 141GB HBM3e│4.8 TB/s│ 700W │ NVLink 4    │
  │ B100     │ 2024 │ 1,800       │ 192GB HBM3e│8.0 TB/s│ 700W │ NVLink 5    │
  │ B200     │ 2024 │ 2,250       │ 192GB HBM3e│8.0 TB/s│ 1000W│ NVLink 5    │
  │ GB200    │ 2025 │ 4,500+      │ 384GB HBM3e│16 TB/s │ 1200W│ NVLink 5    │

  H100 vs A100 の改善点:
  - FP16: 3.2倍高速
  - FP8: 6.4倍高速（Hopper で新規追加）
  - Transformer Engine: FP8/FP16の自動切り替え
  - HBM3: 帯域1.7倍
  - NVLink: 帯域1.5倍（900 GB/s）

  Blackwell の革新:
  - 2ダイ構成（チップレット）
  - FP4対応（推論効率2倍）
  - 第5世代Tensor Core
  - NVLink 5: 1.8 TB/s
```

### 4.2 HBM（High Bandwidth Memory）

```
GDDR vs HBM の比較:

  GDDR6X（コンシューマGPU）:
  ┌──────────┐
  │ GPU チップ│
  │          │── PCB配線 ──→ [GDDR6X] [GDDR6X] ...
  │          │                チップがパッケージ外に配置
  └──────────┘
  帯域: 1,008 GB/s (384-bit, RTX 4090)
  消費電力: 高い（長い配線、高電圧駆動）
  コスト: 安い

  HBM3（データセンターGPU）:
  ┌──────────────────────────┐
  │ シリコンインターポーザ      │
  │ ┌──────┐ ┌────┐ ┌────┐  │
  │ │ GPU  │ │HBM │ │HBM │  │  ← チップがGPUの隣に積層配置
  │ │      │ │    │ │    │  │
  │ └──────┘ └────┘ └────┘  │
  └──────────────────────────┘
  帯域: 3.35 TB/s (H100) → 8.0 TB/s (B200)
  消費電力: 低い（短い配線、低電圧）
  コスト: 高い（インターポーザ製造コスト）

  HBMの構造:
  ┌────┐
  │DRAM│  ← 8-16層のDRAMダイを積層
  │DRAM│
  │DRAM│
  │DRAM│
  │base│  ← ロジックダイ（TSV接続）
  └────┘
  TSV（Through-Silicon Via）: シリコンを貫通する微細な穴で層間接続
  → 数千のTSVで超広帯域を実現（1024ビット幅/スタック）
```

---

## 5. AI学習エンジンとしてのGPU

### 5.1 Tensor Core

```
通常のCUDA Core: スカラ演算（1クロックで1演算）
  a = b × c + d  → 1クロック

Tensor Core: 行列演算（1クロックで行列乗算）
  ┌─────┐   ┌─────┐   ┌─────┐
  │ 4×4 │ × │ 4×4 │ = │ 4×4 │  → 1クロック
  │行列A│   │行列B│   │行列C│
  └─────┘   └─────┘   └─────┘
  = 64回のFMA（積和演算）を1クロックで実行

  → AI学習の本質は巨大な行列乗算
  → Tensor Core がAI学習を10-100倍高速化

Tensor Coreの世代別対応行列サイズ:
  │ 世代     │ FP16行列サイズ │ INT8行列サイズ │ FP8対応 │
  │──────────│────────────────│────────────────│─────────│
  │ Volta    │ 4×4×4          │ なし           │ なし    │
  │ Turing   │ 4×4×4          │ 8×8×16         │ なし    │
  │ Ampere   │ 4×4×4          │ 8×8×16         │ なし    │
  │ Hopper   │ 4×4×4          │ 8×8×16         │ 対応    │
  │ Blackwell│ 4×4×4          │ 8×8×32         │ FP4も   │
```

### 5.2 精度の使い分け

| 精度 | ビット数 | 用途 | Tensor Core性能 |
|------|---------|------|----------------|
| FP64 | 64 | 科学計算 | 基準 |
| FP32 | 32 | 一般的な計算 | 2倍 |
| TF32 | 19 | AI学習 | 8倍 |
| FP16 | 16 | AI学習/推論 | 16倍 |
| BF16 | 16 | AI学習（範囲重視） | 16倍 |
| INT8 | 8 | AI推論 | 32倍 |
| FP8 | 8 | AI学習（Hopper〜） | 64倍 |
| FP4 | 4 | AI推論（Blackwell〜） | 128倍 |

> 精度を下げるほど高速。AI学習では混合精度（Mixed Precision）で品質を維持しつつ高速化。

### 5.3 混合精度学習（Mixed Precision Training）

```
混合精度学習の仕組み:

  従来（FP32のみ）:
  重み(FP32) → 計算(FP32) → 勾配(FP32) → 更新(FP32)
  → 遅い、メモリ消費大

  混合精度:
  ┌─────────────────────────────────────────────┐
  │ マスターウェイト: FP32（精度維持）            │
  │     ↓ コピー                                 │
  │ FP16ウェイト: FP16（計算用）                  │
  │     ↓                                        │
  │ 順伝播: FP16（Tensor Core で高速）            │
  │     ↓                                        │
  │ 損失計算: FP32（精度のため）                  │
  │     ↓                                        │
  │ 逆伝播: FP16（Tensor Core で高速）            │
  │     ↓                                        │
  │ 勾配: FP16 → FP32に変換                      │
  │     ↓                                        │
  │ マスターウェイト更新: FP32                    │
  └─────────────────────────────────────────────┘

  Loss Scaling:
  FP16の最小正規化数 = 6.1e-5
  → 小さい勾配がアンダーフローで0になる
  → 対策: 損失を1024倍に拡大（Loss Scaling）
  → 勾配も1024倍になり、アンダーフロー回避
  → 更新時に1024で割って元に戻す
```

```python
# PyTorch での混合精度学習
import torch
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()  # Loss Scaling を自動管理

for data, target in dataloader:
    data, target = data.cuda(), target.cuda()
    optimizer.zero_grad()

    # autocast: FP16で順伝播・逆伝播を実行
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # GradScaler: Loss Scalingを自動適用
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 効果:
# - 学習速度: 1.5-3倍高速化
# - メモリ使用量: 約50%削減
# - 学習品質: FP32とほぼ同等
```

### 5.4 Transformer Engine（Hopper以降）

```
Transformer Engine の仕組み:

  従来の混合精度: FP16 固定
  Transformer Engine: FP8/FP16 を自動切り替え

  各層ごとにテンソルの統計（最大値、分布）を監視:
  ┌────────────────────────────────────────────┐
  │ Layer N の出力テンソル:                      │
  │ - 値の範囲が FP8 で表現可能 → FP8 で計算   │
  │ - 値の範囲が FP8 を超える → FP16 にフォールバック│
  │ - 次の反復で再度 FP8 を試行                 │
  └────────────────────────────────────────────┘

  FP8 フォーマット:
  E4M3: 指数部4bit + 仮数部3bit（学習向き、精度重視）
  E5M2: 指数部5bit + 仮数部2bit（勾配向き、範囲重視）

  効果:
  - H100 + Transformer Engine: A100比で最大9倍の学習高速化
  - GPT-3級モデルの学習が実用的な時間に
```

---

## 6. GPU間通信技術

### 6.1 NVLink と NVSwitch

```
GPU間接続の比較:

  PCIe 5.0 x16:
    帯域: 63 GB/s（双方向126 GB/s）
    → GPUの計算速度に比べて帯域不足

  NVLink 4 (Hopper):
    帯域: 900 GB/s（18リンク × 50 GB/s）
    → PCIeの約7倍
    → GPU間でメモリを直接読み書き可能

  NVLink 5 (Blackwell):
    帯域: 1.8 TB/s
    → Hopper比2倍

NVSwitch による All-to-All 接続:

  PCIe接続（スター型）:
  ┌──────┐
  │ CPU  │── GPU0
  │      │── GPU1
  │      │── GPU2
  │      │── GPU3
  └──────┘
  → GPU間通信はCPU経由（遅い）

  NVSwitch接続（フルメッシュ）:
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
  → 全GPU間が直接NVLinkで接続
  → 8GPU構成で合計14.4 TB/sの双方向帯域

DGX H100 (8×H100):
  ┌───────────────────────────────────────────┐
  │ 4× NVSwitch                                │
  │                                             │
  │ ┌────┐ ┌────┐ ┌────┐ ┌────┐              │
  │ │H100│ │H100│ │H100│ │H100│              │
  │ └────┘ └────┘ └────┘ └────┘              │
  │ ┌────┐ ┌────┐ ┌────┐ ┌────┐              │
  │ │H100│ │H100│ │H100│ │H100│              │
  │ └────┘ └────┘ └────┘ └────┘              │
  │                                             │
  │ GPU-GPU帯域: 900 GB/s (各方向)             │
  │ 合計GPU メモリ: 640 GB (8 × 80GB)          │
  │ 合計FP8性能: 32 PFLOPS                     │
  │ 消費電力: 約10kW                           │
  │ 価格: 約5,000万円                          │
  └───────────────────────────────────────────┘
```

### 6.2 マルチノード通信

```
クラスター間GPU通信:

  ノード内: NVLink/NVSwitch（900 GB/s）
  ノード間: InfiniBand/RoCE

  InfiniBand の世代:
  │ 世代      │ 帯域(1ポート) │ レイテンシ │
  │───────────│───────────────│────────────│
  │ FDR       │ 56 Gbps      │ 700 ns     │
  │ EDR       │ 100 Gbps     │ 600 ns     │
  │ HDR       │ 200 Gbps     │ 500 ns     │
  │ NDR       │ 400 Gbps     │ 500 ns     │
  │ XDR       │ 800 Gbps     │ TBD        │

  NCCL (NVIDIA Collective Communications Library):
  GPU間の集団通信を最適化するライブラリ
  - AllReduce: 全GPUの勾配を平均
  - AllGather: 全GPUにデータを配布
  - ReduceScatter: 分散して集約

  通信パターン:
  Ring AllReduce（帯域最適）:
  GPU0 → GPU1 → GPU2 → GPU3 → GPU0
  → 各GPUが1/(N-1)のデータを順に渡し集約
  → N GPUで帯域効率 (N-1)/N（ほぼ線形スケール）
```

---

## 7. GPU以外のアクセラレータ

| デバイス | 開発元 | 特徴 | 用途 |
|---------|--------|------|------|
| **TPU** | Google | 行列演算専用、大規模学習 | Google内部AI |
| **NPU** | Apple/Qualcomm | 低消費電力AI推論 | スマホ/PC |
| **FPGA** | Intel/AMD | プログラマブル回路 | カスタム処理 |
| **IPU** | Graphcore | グラフ構造に最適化 | GNN、AI学習 |
| **Trainium** | AWS | 学習専用チップ | AWS SageMaker |
| **Inferentia** | AWS | 推論専用チップ | AWS推論サービス |

### 7.1 各アクセラレータの詳細比較

```
GPU vs TPU vs 専用ASIC の比較:

  NVIDIA H100 (GPU):
  ┌──────────────────────────────────────┐
  │ 汎用性: 高い（CUDA エコシステム）     │
  │ FP16: 989 TFLOPS                     │
  │ INT8: 1,979 TOPS                     │
  │ VRAM: 80GB HBM3                      │
  │ 消費電力: 700W                       │
  │ 利点: 幅広いモデルに対応             │
  │ 欠点: 高価、消費電力大               │
  └──────────────────────────────────────┘

  Google TPU v5e (TPU):
  ┌──────────────────────────────────────┐
  │ 汎用性: 中（JAX/TF向け）              │
  │ BF16: 197 TFLOPS                     │
  │ INT8: 393 TOPS                       │
  │ HBM: 16GB HBM2e                      │
  │ 消費電力: 〜200W                     │
  │ 利点: コスト効率、Google Cloudで利用 │
  │ 欠点: CUDA非対応、PyTorch対応限定    │
  └──────────────────────────────────────┘

  Apple Neural Engine (NPU):
  ┌──────────────────────────────────────┐
  │ 汎用性: 低い（Apple専用）             │
  │ 性能: 38 TOPS (M4 Pro)              │
  │ メモリ: 統合メモリから共有            │
  │ 消費電力: 〜数W                      │
  │ 利点: 超低消費電力、常時稼働可能     │
  │ 欠点: 学習不可、Apple エコシステム限定│
  └──────────────────────────────────────┘
```

---

## 8. 並列計算の理論

### 8.1 アムダールの法則

```
アムダールの法則:

  高速化 = 1 / ((1-P) + P/N)

  P = 並列化可能な割合
  N = プロセッサ数

  例: P=0.95（95%が並列化可能）、N=1024コア
  高速化 = 1 / (0.05 + 0.95/1024) = 1 / 0.0509 ≈ 19.6倍

  ★ 95%が並列化可能でも、最大19.6倍にしかならない
  ★ 5%の逐次部分がボトルネック

  並列化率と最大高速化:
  P=50%  → 最大 2倍   （コアを増やしても意味なし）
  P=90%  → 最大 10倍
  P=95%  → 最大 20倍
  P=99%  → 最大 100倍
  P=99.9%→ 最大 1000倍 ← AI学習はこの領域
```

### 8.2 グスタフソンの法則

```
グスタフソンの法則:

  アムダールの法則の「問題サイズ固定」を修正。
  「コアが増えれば、より大きな問題を同じ時間で解ける」

  高速化 = N - (1-P) × (N-1)

  → 問題サイズをスケールすれば、並列化の効果はほぼ線形
  → これがGPU計算の真の価値:
     「同じ時間で、より大きなモデルを学習できる」
```

### 8.3 ルーフラインモデル

```
ルーフラインモデル（Roofline Model）:

  達成可能性能 = min(ピーク計算性能, ピークメモリ帯域 × 演算強度)

  RTX 4090 の場合:
  ピーク計算性能: 82.6 TFLOPS (FP32)
  ピークメモリ帯域: 1,008 GB/s

                性能 (TFLOPS)
                    │
  82.6 TFLOPS ──────┼──────────────── 計算律速
                    │              /
                    │            /
                    │          /   ← メモリ律速
                    │        /       （帯域 × 演算強度）
                    │      /
                    │    /
                    │  /
                    │/
                    └───────────────── 演算強度 (FLOP/byte)
                         82

  最適化の方向:
  - メモリ律速 → データの再利用率向上（タイリング、キャッシュ活用）
  - 計算律速 → アルゴリズムの効率化、精度を下げる（FP16/FP8）
  - 両方の壁 → より高性能なハードウェアへ移行

  実務での活用:
  1. まず演算強度を計算
  2. ルーフラインモデルで理論上限を求める
  3. 実測値との差分が最適化の余地
  → NVIDIA Nsight Compute が自動計算してくれる
```

---

## 9. 分散学習の手法

### 9.1 データ並列とモデル並列

```
データ並列（Data Parallelism）:
  ┌──────────────────────────────────────────────┐
  │ 同じモデルを全GPUにコピー                      │
  │                                                │
  │ GPU 0: Model + Data[0:N/4]  → 勾配0           │
  │ GPU 1: Model + Data[N/4:N/2] → 勾配1          │
  │ GPU 2: Model + Data[N/2:3N/4] → 勾配2         │
  │ GPU 3: Model + Data[3N/4:N] → 勾配3           │
  │                                                │
  │ AllReduce: 全勾配を平均                        │
  │ → 全GPUのモデルを同期更新                      │
  │                                                │
  │ 適用: モデルが1GPU のVRAMに収まる場合          │
  │ スケーラビリティ: 良好（通信がボトルネック）     │
  └──────────────────────────────────────────────┘

モデル並列（Model Parallelism）:
  ┌──────────────────────────────────────────────┐
  │ モデルを分割して複数GPUに配置                   │
  │                                                │
  │ テンソル並列（Tensor Parallelism）:             │
  │ GPU 0: Layer N の前半の重み                     │
  │ GPU 1: Layer N の後半の重み                     │
  │ → 1つの層を複数GPUで計算                       │
  │ → 通信頻度が高い（各層の入出力で同期）          │
  │                                                │
  │ パイプライン並列（Pipeline Parallelism）:       │
  │ GPU 0: Layer 0-9                               │
  │ GPU 1: Layer 10-19                             │
  │ GPU 2: Layer 20-29                             │
  │ GPU 3: Layer 30-39                             │
  │ → マイクロバッチをパイプライン処理              │
  │ → バブル（空き時間）が発生                     │
  │                                                │
  │ 適用: 巨大モデル（GPT-4、Llama 3等）           │
  └──────────────────────────────────────────────┘

3D並列（データ × テンソル × パイプライン）:
  → 大規模言語モデルの学習ではこの3つを組み合わせ
  → Megatron-LM、DeepSpeed 等のフレームワークが実装
```

### 9.2 ZeRO（Zero Redundancy Optimizer）

```
ZeRO の3段階:

  通常のデータ並列:
  各GPUが保持: モデル重み + 勾配 + オプティマイザ状態
  メモリ使用: 16φ × N（φ=パラメータ数、N=GPU数）

  ZeRO Stage 1: オプティマイザ状態を分割
  各GPUが保持: モデル重み + 勾配 + 1/N のオプティマイザ状態
  メモリ削減: 最大4倍

  ZeRO Stage 2: 勾配も分割
  各GPUが保持: モデル重み + 1/N の勾配 + 1/N のオプティマイザ状態
  メモリ削減: 最大8倍

  ZeRO Stage 3: 重みも分割
  各GPUが保持: 1/N のモデル重み + 1/N の勾配 + 1/N のオプティマイザ状態
  メモリ削減: 最大N倍（GPU数に比例）

  → 10Bパラメータモデルでも4×A100で学習可能に
```

---

## 10. 実践演習

### 演習1: CPU vs GPU の判定（基礎）

以下のタスクについて、CPUとGPUのどちらが適しているか判定し理由を述べよ:
1. 100万枚の画像に同じフィルタを適用
2. JSONファイルの解析と変換
3. ニューラルネットワークの学習
4. Gitの差分計算
5. 動画のエンコード

### 演習2: アムダールの法則（応用）

あるプログラムの実行時間の内訳が以下の場合:
- 入力処理: 5% （逐次）
- メイン計算: 80% （並列化可能）
- 結果集約: 10% （部分的に並列化可能、50%）
- 出力処理: 5% （逐次）

16コアCPUと1024コアGPUを使った場合の最大高速化率を計算せよ。

### 演習3: GPU活用（発展）

PyTorchを使って、CPU vs GPUの行列乗算速度を比較するベンチマークを作成せよ:
```python
import torch
import time

# 行列サイズを変えて測定: 100, 500, 1000, 5000, 10000
for N in [100, 500, 1000, 5000, 10000]:
    # CPU
    a_cpu = torch.randn(N, N)
    b_cpu = torch.randn(N, N)
    start = time.time()
    c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.time() - start

    # GPU (CUDA or MPS)
    # ... 実装して比較
```

### 演習4: メモリ帯域の計算（応用）

H100 GPU（80GB HBM3、3.35 TB/s帯域）で以下を計算せよ:
1. 全VRAMを読み出すのにかかる最短時間
2. 70Bパラメータモデル（FP16）のフォワードパスに必要な最小帯域
3. バッチサイズ32で学習する場合のメモリ使用量見積もり

### 演習5: 分散学習の設計（発展）

以下の条件で大規模言語モデルの学習システムを設計せよ:
- モデル: 13Bパラメータ、Transformer
- GPU: A100 80GB × 16台（2ノード × 8GPU）
- 目標: 1Tトークンの学習を30日以内に完了

設計項目:
1. 並列化戦略（データ/テンソル/パイプライン）
2. バッチサイズとアキュムレーション
3. 精度（FP16/BF16/FP8）
4. 通信ボトルネックの特定と対策
5. 必要な計算量（FLOPS）の見積もり

---

## FAQ

### Q1: ゲーム用GPUとAI学習用GPUの違いは？

**A**: 同じGPUチップでも、用途によってメモリとTensor Coreが異なる:
- **ゲーム用（RTX 4090）**: VRAM 24GB、RT Core重視、消費者向け
- **AI学習用（A100/H100）**: VRAM 80GB、Tensor Core重視、ECC付き
- **AI推論用（L4/L40S）**: 低消費電力、INT8/FP8最適化

### Q2: Apple Silicon のGPUはNVIDIAに勝てますか？

**A**: 用途による:
- **メモリ帯域**: M4 Max (546GB/s) vs RTX 4090 (1,008GB/s) → NVIDIA優位
- **統合メモリ**: M4 Maxの128GBをGPUが直接使える → 大規模LLM推論で有利
- **電力効率**: Apple Siliconが圧倒的に優れる
- **CUDA エコシステム**: AI/MLライブラリの大半がCUDA前提 → NVIDIA優位

### Q3: GPUプログラミングを学ぶべきですか？

**A**: AI/ML、ゲーム開発、HPC（高性能計算）に関わるなら必須。Web開発者は直接は不要だが、「なぜAI学習にGPUが必要か」を理解することは重要。PyTorch/TensorFlowはGPUの詳細を抽象化してくれるため、CUDAを直接書く必要は少ない。

### Q4: VRAMが足りない場合の対処法は？

**A**: 以下の手法を組み合わせる:
- **勾配チェックポインティング**: 中間結果を保存せず再計算
- **混合精度**: FP16/BF16で使用量を半減
- **ZeRO Stage 3**: 重みを複数GPUに分散
- **オフロード**: CPU RAM/NVMe SSDに一部を退避
- **量子化**: INT4/INT8で推論
- **モデル分割**: 層ごとに異なるGPUに配置

### Q5: GPU温度が高い場合の対策は？

**A**: GPU温度管理の基準:
- 80度以下: 正常
- 80-90度: 注意（サーマルスロットリングの可能性）
- 90度以上: 対策必要
対策:
- ケース内エアフローの改善
- GPUファンカーブの調整（MSI Afterburner等）
- サーマルパッドの交換
- アンダーボルト（電圧を下げて発熱抑制、性能微減）

---

## まとめ

| 概念 | ポイント |
|------|---------|
| CPU vs GPU | レイテンシ最適化 vs スループット最適化 |
| CUDA Core | 汎用並列計算。数千〜数万コア |
| Tensor Core | 行列演算専用。AI学習を10-100倍高速化 |
| ワープ | 32スレッドの同期実行単位。分岐でダイバージェンス |
| アムダールの法則 | 逐次部分が並列化の限界を決める |
| 精度 | FP32→FP16→FP8と下げるほど高速（品質トレードオフ） |
| NVLink | GPU間高帯域接続。マルチGPU学習に必須 |
| 分散学習 | データ/テンソル/パイプライン並列の組み合わせ |

---

## 次に読むべきガイド

→ [[05-io-systems.md]] — I/Oシステムと割り込み

---

## 参考文献

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

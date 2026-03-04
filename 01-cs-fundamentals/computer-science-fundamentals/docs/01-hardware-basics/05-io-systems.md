# I/Oシステム

> I/O（入出力）はコンピュータと外部世界をつなぐ窓であり、多くのアプリケーションのボトルネックはCPUではなくI/Oにある。バス、割り込み、DMA、デバイスドライバ、I/Oスケジューリングを体系的に理解することが高性能システム設計の第一歩となる。

## この章で学ぶこと

- [ ] I/Oの基本概念（ポーリング、割り込み、DMA）を説明できる
- [ ] バスアーキテクチャの階層構造と帯域幅計算を理解する
- [ ] 割り込み処理の全体フロー（ハードウェア割り込み、ソフトウェア割り込み、MSI/MSI-X）を説明できる
- [ ] DMAの動作原理、バウンスバッファ、スキャッタ・ギャザーDMAを理解する
- [ ] デバイスドライバの設計パターンとLinuxカーネルモジュールの仕組みを説明できる
- [ ] I/Oスケジューリングアルゴリズム（CFQ、Deadline、mq-deadline、BFQ、none）を比較できる
- [ ] 非同期I/O（select、poll、epoll、kqueue、io_uring）の進化と使い分けを判断できる
- [ ] メモリマップドI/OとポートマップドI/Oの違いを理解する
- [ ] I/O性能のボトルネック分析と最適化手法を習得する

## 前提知識

- CPUとメモリの基礎 → 参照: [[00-cpu-architecture.md]], [[01-memory-hierarchy.md]]
- バスとマザーボードの基礎 → 参照: [[03-motherboard-and-bus.md]]
- ストレージシステムの基礎 → 参照: [[02-storage-systems.md]]

---

## 1. I/Oシステムの全体像

### 1.1 I/Oの位置づけ

コンピュータシステムにおいてI/O（Input/Output）は、CPU・メモリと外部デバイスの間でデータをやり取りする仕組みの総称である。キーボードやマウスからの入力、ディスプレイへの出力、ストレージへの読み書き、ネットワーク通信のすべてがI/Oに該当する。

現代のシステムでは、CPUの演算速度に比べてI/Oデバイスの速度が圧倒的に遅いことが根本的な課題となっている。この速度差を効率的に吸収するために、バス、割り込み、DMA、スケジューリングといった多層のメカニズムが発達してきた。

```
I/Oシステムの全体構造:

  ┌────────────────────────────────────────────────────────┐
  │  アプリケーション層                                      │
  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                 │
  │  │ Web  │ │ DB   │ │ ファイル│ │ゲーム │                 │
  │  │サーバ │ │エンジン│ │ 操作  │ │エンジン│                 │
  │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘                 │
  │     └────────┴────────┴────────┘                       │
  │                   │ システムコール (read/write/ioctl)    │
  ├───────────────────┼────────────────────────────────────┤
  │  カーネル層        │                                     │
  │     ┌─────────────▼──────────────┐                     │
  │     │ VFS (Virtual File System) │                      │
  │     └─────────────┬──────────────┘                     │
  │     ┌─────────────▼──────────────┐                     │
  │     │ I/Oスケジューラ             │ ← 要求の並べ替え     │
  │     └─────────────┬──────────────┘                     │
  │     ┌─────────────▼──────────────┐                     │
  │     │ デバイスドライバ            │ ← HW固有の操作       │
  │     └─────────────┬──────────────┘                     │
  │     ┌─────────────▼──────────────┐                     │
  │     │ 割り込みハンドラ / DMA制御  │ ← データ転送制御      │
  │     └─────────────┬──────────────┘                     │
  ├───────────────────┼────────────────────────────────────┤
  │  ハードウェア層    │                                     │
  │     ┌─────────────▼──────────────┐                     │
  │     │ バス (PCIe / USB / SATA)  │ ← 物理的データ経路   │
  │     └─────────────┬──────────────┘                     │
  │     ┌─────────────▼──────────────┐                     │
  │     │ デバイスコントローラ         │                     │
  │     └─────────────┬──────────────┘                     │
  │     ┌─────────────▼──────────────┐                     │
  │     │ I/Oデバイス (SSD/NIC/GPU) │                      │
  │     └────────────────────────────┘                     │
  └────────────────────────────────────────────────────────┘
```

### 1.2 デバイスの速度階層

I/Oデバイスの速度はデバイスの種類によって桁違いに異なる。設計時にはこの速度差を常に意識する必要がある。

| デバイス | 帯域幅 | レイテンシ | IOPS（目安） |
|---------|--------|----------|-------------|
| CPUレジスタ | 数TB/s | < 1ns | - |
| L1キャッシュ | 〜1TB/s | 〜1ns | - |
| DDR5メモリ | 〜50GB/s | 〜100ns | - |
| NVMe SSD (PCIe 5.0) | 〜14GB/s | 〜10μs | 〜2,000,000 |
| SATA SSD | 〜550MB/s | 〜50μs | 〜100,000 |
| HDD (7200RPM) | 〜200MB/s | 〜5ms | 〜150 |
| 10GbE NIC | 〜1.25GB/s | 〜10μs | - |
| USB 3.2 Gen 2 | 〜1.25GB/s | 〜1ms | - |
| キーボード | 数十B/s | 〜50ms | - |

この表から明らかなように、CPUレジスタとキーボードの間には10桁以上の速度差がある。この差異を隠蔽し、システム全体として効率的に動作させることがI/Oサブシステムの役割である。

### 1.3 I/Oアドレッシング方式

CPUがI/Oデバイスと通信するための基本的なアドレッシング方式には2種類がある。

```
(A) ポートマップドI/O (Port-Mapped I/O: PMIO)

  CPU                    メモリ空間        I/O空間
  ┌──────┐              ┌──────┐        ┌──────┐
  │      │──メモリ命令→ │ 0x00 │        │      │
  │      │  (MOV等)     │ 0x01 │        │      │
  │      │              │ ...  │        │      │
  │      │              │ 0xFF │        │      │
  │      │              └──────┘        │      │
  │      │                              │      │
  │      │──I/O命令──→               │ 0x00 │
  │      │  (IN/OUT)                    │ 0x01 │
  │      │                              │ ...  │
  └──────┘                              └──────┘

  特徴:
  - メモリ空間とI/O空間が完全に分離
  - x86固有の IN/OUT 命令を使用
  - I/Oアドレス空間は 0x0000〜0xFFFF（64KB）
  - レガシーデバイスで使用（シリアルポート: 0x3F8, キーボード: 0x60）

(B) メモリマップドI/O (Memory-Mapped I/O: MMIO)

  CPU                    統合アドレス空間
  ┌──────┐              ┌──────────────┐
  │      │              │ 0x00000000   │ ← メモリ領域
  │      │──メモリ命令→ │ ...          │
  │      │  (MOV等)     │ 0x7FFFFFFF   │
  │      │              ├──────────────┤
  │      │              │ 0x80000000   │ ← デバイスレジスタ領域
  │      │              │ (GPU VRAM)   │
  │      │              │ (NIC レジスタ)│
  │      │              │ 0xFFFFFFFF   │
  │      │              └──────────────┘
  └──────┘

  特徴:
  - メモリとI/Oが同じアドレス空間を共有
  - 通常のメモリ命令（MOV, LDR/STR等）でデバイスにアクセス
  - 現代のほとんどのデバイスが採用（PCIe BAR経由）
  - CPUキャッシュの影響を避けるため、非キャッシュ属性を設定する必要あり
```

**PMIO vs MMIO 比較表:**

| 項目 | PMIO | MMIO |
|------|------|------|
| アドレス空間 | 専用I/O空間（64KB） | メモリ空間の一部 |
| アクセス命令 | IN/OUT（x86専用） | MOV等の汎用命令 |
| アドレス幅 | 16ビット固定 | アーキテクチャ依存（64ビット可） |
| キャッシュ | 自動的に非キャッシュ | 明示的に非キャッシュ設定が必要 |
| アーキテクチャ | x86のみ | 全アーキテクチャ対応 |
| 主な用途 | レガシーデバイス | PCIeデバイス、現代のI/O |
| 性能 | 低速（専用命令のオーバーヘッド） | 高速（汎用命令で最適化可能） |

---

## 2. バスアーキテクチャ

### 2.1 バスの基本概念

バス（Bus）はコンピュータ内部でデータを転送する共有通信路である。「バス」という名称は、ラテン語の「omnibus（すべての人のために）」に由来し、複数のコンポーネントが共有する通信経路であることを意味する。

バスは以下の3種類の信号線で構成される。

```
バスの3つの信号線:

  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │  データバス (Data Bus)                                │
  │  ════════════════════════════════════════════         │
  │  データそのものを転送する信号線群                       │
  │  幅: 8/16/32/64ビット（一度に転送できるビット数）      │
  │                                                      │
  │  アドレスバス (Address Bus)                            │
  │  ════════════════════════════════════════════         │
  │  転送先/転送元のアドレスを指定する信号線群              │
  │  幅: 32ビット → 4GB, 64ビット → 16EB のアドレス空間    │
  │                                                      │
  │  制御バス (Control Bus)                                │
  │  ════════════════════════════════════════════         │
  │  読み書きの方向、割り込み要求、クロックなどの制御信号    │
  │  信号例: R/W, IRQ, CLK, RESET, READY                  │
  │                                                      │
  └──────────────────────────────────────────────────────┘

  帯域幅の計算:
    帯域幅 = バス幅(ビット) × クロック周波数 × 転送レート係数

    例: PCIe 5.0 x16
    帯域幅 = 16レーン × 32GT/s × 128b/130b符号化
           ≈ 63 GB/s（片方向）
           ≈ 126 GB/s（双方向合計）
```

### 2.2 バスの階層構造

現代のPCでは、すべてのデバイスが単一のバスを共有するのではなく、速度帯域に応じた階層構造を持つ。

```
現代のPC バス階層（2024年以降の典型的構成）:

  ┌──────────────────────────────────────────────────────────┐
  │                        CPU                                │
  │  ┌──────┐  ┌──────┐  ┌──────────────────────────────┐   │
  │  │ コア  │  │ コア  │  │ 統合メモリコントローラ        │   │
  │  │ 0-7  │  │ 8-15 │  │ DDR5: 〜89.6 GB/s           │   │
  │  └──┬───┘  └──┬───┘  └──────────┬───────────────────┘   │
  │     └────┬─────┘                 │                        │
  │          │ 内部インターコネクト    │                        │
  │  ┌───────▼────────────────────────▼───────────────────┐   │
  │  │            PCIe Root Complex                       │   │
  │  └───┬──────────────┬──────────────┬─────────────────┘   │
  └──────┼──────────────┼──────────────┼─────────────────────┘
         │              │              │
    PCIe 5.0 x16    PCIe 5.0 x4    PCIe 4.0 x4
    (63 GB/s)       (16 GB/s)      (8 GB/s)
         │              │              │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │  GPU    │    │  NVMe   │    │  NVMe   │
    │(RTX5090)│    │  SSD    │    │  SSD    │
    └─────────┘    └─────────┘    └─────────┘

  ┌─────────────────────────────────────────────────────────┐
  │                    チップセット (PCH)                     │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │  PCIe 4.0/3.0 スイッチ                          │    │
  │  └──┬────────┬────────┬────────┬────────┬─────────┘    │
  │     │        │        │        │        │              │
  │  SATA III  USB 3.2  2.5GbE  オーディオ  追加PCIe       │
  │  (600MB/s) (20Gbps) (312MB/s)          スロット         │
  │     │        │        │        │        │              │
  │  ┌──▼──┐ ┌──▼──┐ ┌──▼──┐ ┌──▼──┐ ┌──▼──┐           │
  │  │HDD  │ │USB  │ │NIC  │ │サウンド│ │拡張  │           │
  │  │/SSD │ │デバイス│ │     │ │カード│ │カード│            │
  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘            │
  └─────────────────────────────────────────────────────────┘
```

### 2.3 PCIeの詳細

PCIe（Peripheral Component Interconnect Express）は現代のI/Oバスの標準規格であり、従来のパラレルバス（PCI）をシリアルポイントツーポイント接続に置き換えたものである。

**PCIe世代別帯域幅一覧:**

| 世代 | 策定年 | 転送レート | x1帯域幅 | x16帯域幅 | 符号化 |
|------|--------|----------|---------|----------|--------|
| PCIe 1.0 | 2003 | 2.5 GT/s | 250 MB/s | 4 GB/s | 8b/10b |
| PCIe 2.0 | 2007 | 5 GT/s | 500 MB/s | 8 GB/s | 8b/10b |
| PCIe 3.0 | 2010 | 8 GT/s | 984 MB/s | 15.75 GB/s | 128b/130b |
| PCIe 4.0 | 2017 | 16 GT/s | 1.97 GB/s | 31.5 GB/s | 128b/130b |
| PCIe 5.0 | 2019 | 32 GT/s | 3.94 GB/s | 63 GB/s | 128b/130b |
| PCIe 6.0 | 2022 | 64 GT/s | 7.56 GB/s | 121 GB/s | PAM4+FEC |

PCIeの重要な概念としてBAR（Base Address Register）がある。BARはPCIeデバイスが使用するメモリ空間をシステムに通知するための仕組みで、OSはこのBARを読み取ってMMIOの領域を設定する。

```c
/* PCIe BARの読み取り例（Linuxカーネルドライバ） */
#include <linux/pci.h>

static int my_pci_probe(struct pci_dev *pdev,
                        const struct pci_device_id *id)
{
    int ret;
    void __iomem *bar0;
    resource_size_t bar0_start, bar0_len;

    /* PCIデバイスを有効化 */
    ret = pci_enable_device(pdev);
    if (ret)
        return ret;

    /* BAR0のリソースを取得 */
    bar0_start = pci_resource_start(pdev, 0);
    bar0_len   = pci_resource_len(pdev, 0);

    /* BARをメモリ空間にマッピング（MMIO） */
    bar0 = ioremap(bar0_start, bar0_len);
    if (!bar0) {
        pci_disable_device(pdev);
        return -ENOMEM;
    }

    /* デバイスレジスタへの読み書き */
    u32 status = ioread32(bar0 + DEVICE_STATUS_REG);
    iowrite32(0x1, bar0 + DEVICE_CONTROL_REG);

    /* 注意: 通常のポインタデリファレンスではなく
       ioread/iowrite を使う（メモリバリア、エンディアン考慮） */

    return 0;
}
```

### 2.4 USB、SATA、その他のバス規格

PCIe以外にも用途に応じた多様なバス規格が存在する。

| バス規格 | トポロジ | 最大帯域幅 | 主な用途 |
|---------|---------|----------|---------|
| USB 2.0 | ツリー（ハブ） | 480 Mbps | マウス、キーボード |
| USB 3.2 Gen 2x2 | ツリー | 20 Gbps | 外付けSSD |
| USB4 / Thunderbolt 4 | トンネリング | 40 Gbps | 外付けGPU、ドック |
| SATA III | ポイントツーポイント | 6 Gbps | 内蔵SSD/HDD |
| NVMe (PCIe 5.0 x4) | PCIe | 〜14 GB/s | 高速内蔵SSD |
| CXL 3.0 | PCIe物理層 | PCIe 6.0準拠 | メモリプーリング |
| InfiniBand HDR | スイッチ | 200 Gbps | HPC、データセンター |

---

## 3. I/Oの3つの方式

### 3.1 プログラムI/O（ポーリング）

プログラムI/O（Programmed I/O）は、CPUが能動的にデバイスの状態を繰り返し確認する方式である。「ポーリング（Polling）」とも呼ばれる。

```
ポーリング方式の動作:

  CPU                         デバイスコントローラ
  │                           │
  │──「コマンド書き込み」────→│ (1) CPUがデバイスに命令を発行
  │                           │
  │──「状態レジスタ読み取り」→│ (2) CPUがビジーフラグを確認
  │←─「BUSY」────────────────│     → まだ完了していない
  │                           │
  │──「状態レジスタ読み取り」→│ (3) 再び確認（ビジーウェイト）
  │←─「BUSY」────────────────│     → まだ完了していない
  │                           │
  │  ... (繰り返し) ...        │     ← CPUサイクルを浪費
  │                           │
  │──「状態レジスタ読み取り」→│ (n) 完了を検知
  │←─「DONE」────────────────│
  │                           │
  │──「データレジスタ読み取り」→│ (n+1) データを取得
  │←─ データ ────────────────│
  │                           │

  ビジーウェイトループの疑似コード:
    while (read_status_register() & BUSY_FLAG) {
        /* CPUは何もせず空回り */
    }
    data = read_data_register();
```

**ポーリングの利点:**
- 実装が極めて単純で、割り込みコントローラが不要
- レイテンシが最小（割り込み処理のオーバーヘッドがない）
- 予測可能なタイミング（リアルタイムシステム向き）

**ポーリングの欠点:**
- CPUサイクルを大量に消費（ビジーウェイト）
- デバイスが遅い場合、CPU利用率がほぼ100%になる
- マルチタスク環境では他のプロセスが実行できない

**ポーリングが適切な場面:**
現代でもポーリングは特定の用途で積極的に使われている。NVMe SSDの高性能モードでは、割り込みのオーバーヘッド（数μs）がI/O完了時間（〜10μs）に対して無視できないため、ポーリングモード（io_poll）でレイテンシを最小化する手法が用いられる。DPDK（Data Plane Development Kit）でもネットワークパケット処理にポーリングを採用している。

### 3.2 割り込み駆動I/O

割り込み（Interrupt）は、デバイスからCPUに対して「処理が完了した」ことを非同期に通知する仕組みである。CPUはデバイスの完了を待つ間、他のタスクを実行できる。

```
割り込み駆動I/Oの動作:

  CPU                     割り込みコントローラ     デバイス
  │                       (APIC)                │
  │──「コマンド発行」──────────────────────────→│ (1)
  │                                              │
  │ [他のプロセスを実行]                          │ (2) CPUは別の仕事
  │ [タスクA → タスクB → ...]                   │
  │                                              │
  │                       │←── IRQ信号 ─────────│ (3) デバイスが完了通知
  │                       │                      │
  │←── 割り込み通知 ──────│                      │ (4) APICがCPUに転送
  │                       │                      │
  │ [現在の状態を保存]     │                      │ (5) コンテキスト保存
  │ [割り込みベクタ参照]   │                      │ (6) IDTから
  │ [ISR実行]             │                      │     ハンドラアドレス取得
  │   ├─ データ読み取り    │                      │ (7) ISR内でデバイス操作
  │   ├─ バッファにコピー  │                      │
  │   └─ EOI送信 ────────→│                      │ (8) 割り込み完了通知
  │                       │                      │
  │ [状態を復元]           │                      │ (9) 元のタスクに復帰
  │ [元のタスク再開]       │                      │
  │                       │                      │
```

### 3.3 割り込みの種類と詳細

割り込みは発生源と用途によって複数の種類に分類される。

```
割り込みの分類体系:

  割り込み (Interrupt)
  ├── ハードウェア割り込み (外部割り込み)
  │   ├── マスク可能割り込み (Maskable: INTR)
  │   │   ├── レベルトリガ: 信号レベルがHighの間、割り込み有効
  │   │   └── エッジトリガ: 信号の立ち上がり時に割り込み発生
  │   └── マスク不可割り込み (Non-Maskable: NMI)
  │       └── メモリパリティエラー、ハードウェア障害等
  │
  ├── ソフトウェア割り込み (内部割り込み)
  │   ├── 例外 (Exception)
  │   │   ├── フォルト (Fault): 復帰可能（例: ページフォルト）
  │   │   ├── トラップ (Trap): 意図的（例: INT 0x80, syscall）
  │   │   └── アボート (Abort): 復帰不可（例: ダブルフォルト）
  │   └── システムコール (INT 0x80 / SYSCALL命令)
  │
  └── メッセージ信号割り込み (MSI/MSI-X)
      ├── MSI: PCIデバイスからメモリ書き込みで割り込み通知
      │   → 最大32個の割り込みベクタ
      └── MSI-X: MSIの拡張版
          → 最大2048個の割り込みベクタ
          → NVMe/NICの各キューに個別割り込みを割り当て可能
```

### 3.4 割り込み処理の詳細フロー（x86_64）

```
x86_64 での割り込み処理フロー:

  (1) デバイスが割り込み信号を発生
         │
         ▼
  (2) Local APIC が割り込みを受理
      - 優先度を確認 (TPR: Task Priority Register)
      - 現在実行中の割り込みより優先度が低ければ保留
         │
         ▼
  (3) CPUが現在の命令を完了後、割り込みを受け付け
      - RFLAGS, CS, RIP をスタックに自動保存
      - 特権レベルが変わる場合は RSP も切り替え
         │
         ▼
  (4) IDT (Interrupt Descriptor Table) を参照
      - 割り込みベクタ番号 → IDTエントリ
      - エントリからISRのアドレスを取得

      IDTの構造:
      ┌───────────┬──────────────────────────┐
      │ ベクタ番号 │ 用途                      │
      ├───────────┼──────────────────────────┤
      │ 0         │ #DE: ゼロ除算例外          │
      │ 1         │ #DB: デバッグ例外          │
      │ 2         │ NMI: マスク不可割り込み    │
      │ 6         │ #UD: 無効オペコード        │
      │ 13        │ #GP: 一般保護例外          │
      │ 14        │ #PF: ページフォルト        │
      │ 32-255    │ 外部割り込み / ユーザー定義 │
      └───────────┴──────────────────────────┘
         │
         ▼
  (5) ISR (Interrupt Service Routine) 実行
      - Top Half: 最小限の処理（割り込み禁止状態）
        ・デバイスレジスタ読み取り
        ・割り込みフラグクリア
        ・Bottom Half のスケジュール
      - Bottom Half: 遅延可能な処理
        ・softirq / tasklet / workqueue で実行
        ・割り込み許可状態で動作
         │
         ▼
  (6) EOI (End of Interrupt) を APIC に送信
         │
         ▼
  (7) IRET 命令で元のコンテキストに復帰
      - RIP, CS, RFLAGS をスタックから復元
```

### 3.5 Linux カーネルでの割り込みハンドラ登録

```c
/* Linux カーネルでの割り込みハンドラ実装例 */
#include <linux/interrupt.h>
#include <linux/module.h>

#define MY_IRQ 17  /* 割り込み番号 */

/* Top Half: 割り込みコンテキストで実行（高速に完了する必要あり） */
static irqreturn_t my_isr_top(int irq, void *dev_id)
{
    struct my_device *dev = dev_id;
    u32 status;

    /* デバイスの割り込み状態を確認 */
    status = ioread32(dev->regs + IRQ_STATUS_REG);
    if (!(status & MY_DEVICE_IRQ_MASK))
        return IRQ_NONE;  /* この割り込みは自デバイスのものではない */

    /* 割り込みフラグをクリア（デバイスに再割り込みを許可） */
    iowrite32(status, dev->regs + IRQ_ACK_REG);

    /* 受信データをデバイスローカルバッファに退避 */
    dev->pending_data = ioread32(dev->regs + DATA_REG);
    dev->irq_count++;

    /* Bottom Half をスケジュール */
    tasklet_schedule(&dev->my_tasklet);

    return IRQ_HANDLED;
}

/* Bottom Half: プロセスコンテキストに近い環境で実行 */
static void my_tasklet_handler(unsigned long data)
{
    struct my_device *dev = (struct my_device *)data;

    /* 時間のかかる処理をここで実行 */
    process_received_data(dev->pending_data);
    wake_up_interruptible(&dev->wait_queue);
}

/* ドライバ初期化時に割り込みを登録 */
static int my_driver_init(struct my_device *dev)
{
    int ret;

    tasklet_init(&dev->my_tasklet, my_tasklet_handler,
                 (unsigned long)dev);

    /* IRQF_SHARED: 他デバイスとIRQラインを共有可能
       第4引数: デバイス識別用のポインタ */
    ret = request_irq(MY_IRQ, my_isr_top,
                      IRQF_SHARED, "my_device", dev);
    if (ret) {
        pr_err("Failed to request IRQ %d\n", MY_IRQ);
        return ret;
    }

    return 0;
}

/* ドライバ終了時に割り込みを解除 */
static void my_driver_exit(struct my_device *dev)
{
    free_irq(MY_IRQ, dev);
    tasklet_kill(&dev->my_tasklet);
}
```

### 3.6 DMA（Direct Memory Access）

DMA（Direct Memory Access）は、CPUを介さずにメモリとI/Oデバイス間で直接データを転送する仕組みである。大容量データ転送において、CPUの負荷を劇的に削減する。

```
DMA転送の動作フロー:

  CPU              DMAコントローラ (DMAC)        デバイス       メモリ
  │                │                            │             │
  │ (1) DMA設定    │                            │             │
  │ ─転送元アドレス→│                            │             │
  │ ─転送先アドレス→│                            │             │
  │ ─転送バイト数 →│                            │             │
  │ ─転送開始命令 →│                            │             │
  │                │                            │             │
  │ (2) CPUは      │ (3) DMAがバスを使用         │             │
  │   別タスク実行  │ ──「データ要求」─────────→│             │
  │                │ ←── データブロック ─────────│             │
  │ [タスクA]      │ ──「メモリ書き込み」─────────────────────→│
  │ [タスクB]      │                            │             │
  │ [タスクC]      │ ──「データ要求」─────────→│             │
  │                │ ←── データブロック ─────────│             │
  │                │ ──「メモリ書き込み」─────────────────────→│
  │                │                            │             │
  │                │ (4) 転送完了               │             │
  │ ←── 完了割り込み─│                           │             │
  │                │                            │             │
  │ (5) ISRで後処理│                            │             │
  │                │                            │             │

  DMAの転送モード:
  ┌──────────────────────────────────────────────────────┐
  │ (A) ブロック転送モード                                 │
  │   バスを独占して一括転送。大容量データ向き。             │
  │   CPU は転送中バスにアクセスできない。                  │
  │                                                      │
  │ (B) サイクルスチールモード                              │
  │   CPUが使っていないバスサイクルを「盗んで」転送。        │
  │   CPUとバスを時分割で共有。少量データ向き。             │
  │                                                      │
  │ (C) バーストモード                                     │
  │   連続するアドレスを高速に転送。DDRのバースト転送と連携。 │
  │   現代のDMAの主流方式。                                │
  └──────────────────────────────────────────────────────┘
```

### 3.7 スキャッタ・ギャザーDMA

スキャッタ・ギャザーDMA（Scatter-Gather DMA, SG-DMA）は、物理的に不連続なメモリ領域に対して一回のDMA操作でデータを分散書き込み（スキャッタ）または集約読み出し（ギャザー）する技術である。

```
通常のDMA vs スキャッタ・ギャザーDMA:

(A) 通常のDMA: 連続した物理メモリが必要

  物理メモリ:
  ┌────┬────┬────┬────┬────┬────┬────┬────┐
  │使用│使用│空き│空き│空き│空き│使用│使用│
  └────┴────┴────┴────┴────┴────┴────┴────┘
                ↑                ↑
                └── 連続4ページ ──┘
                DMAバッファとして使用可能

  問題: メモリ断片化が進むと連続領域が確保困難

(B) スキャッタ・ギャザーDMA: 不連続でもOK

  物理メモリ:
  ┌────┬────┬────┬────┬────┬────┬────┬────┐
  │使用│ SG │使用│ SG │使用│使用│ SG │ SG │
  └────┴──┬─┴────┴──┬─┴────┴────┴──┬─┴──┬─┘
          │         │              │    │
          ▼         ▼              ▼    ▼
  SG リスト (Scatter-Gather List):
  ┌─────────────────────────────────────┐
  │ Entry 0: addr=0x1000, len=4096     │→ ページ1
  │ Entry 1: addr=0x3000, len=4096     │→ ページ3
  │ Entry 2: addr=0x6000, len=4096     │→ ページ6
  │ Entry 3: addr=0x7000, len=4096     │→ ページ7
  └─────────────────────────────────────┘

  DMAコントローラはSGリストを順番に処理し、
  不連続な物理ページへデータを分散転送する
```

### 3.8 DMAとキャッシュの一貫性問題

DMAはCPUキャッシュをバイパスしてメモリに直接アクセスするため、キャッシュ一貫性（Cache Coherency）の問題が発生する。

```
キャッシュ一貫性問題:

  (問題1) DMA書き込み後にCPUが古いキャッシュを読む

    CPU Cache: [データA (古い)]     ← CPUはこれを読んでしまう
                                   ↑ キャッシュにヒット
    メモリ:    [データB (DMAが更新)] ← DMAが新データを書いた

  (問題2) CPU書き込みがキャッシュに残りDMAが古いメモリを読む

    CPU Cache: [データC (最新)]     ← まだメモリに書き戻されていない
    メモリ:    [データD (古い)]     ← DMAはこれを読んでしまう

  解決策:
  ┌────────────────────────────────────────────────────────┐
  │ (A) キャッシュ無効化（Invalidate）                       │
  │   DMA読み取り前にキャッシュラインを無効化                  │
  │   → CPUが次にアクセスするとメモリから再読み込み            │
  │                                                        │
  │ (B) キャッシュフラッシュ（Flush/Clean）                   │
  │   DMA書き込み前にキャッシュ内容をメモリに書き戻す           │
  │   → DMAが最新データを読める                               │
  │                                                        │
  │ (C) 非キャッシュメモリ（Uncacheable）                     │
  │   DMAバッファをキャッシュ不可属性で確保                     │
  │   → 一貫性問題が起きないが性能低下                        │
  │                                                        │
  │ (D) ハードウェアキャッシュコヒーレントDMA                   │
  │   PCIeデバイスがCPUキャッシュを参照（スヌープ）             │
  │   → ARMの Cache Coherent Interconnect (CCI)             │
  │   → x86では基本的にPCIeがキャッシュコヒーレント            │
  └────────────────────────────────────────────────────────┘
```

### 3.9 3方式の比較

| 項目 | ポーリング | 割り込み | DMA |
|------|----------|---------|-----|
| CPU負荷 | 極めて高い（ビジーウェイト） | 中程度（ISR実行時のみ） | 低い（設定と完了処理のみ） |
| レイテンシ | 最小（即座に検知） | 中程度（1〜10μs） | 中〜大（設定オーバーヘッド） |
| スループット | 低い | 中程度 | 高い |
| 実装複雑度 | 低 | 中 | 高 |
| 適用場面 | 超低レイテンシ、DPDK、NVMe io_poll | 一般的なI/O、キーボード | 大容量転送、ディスク、NIC |
| ハードウェア要件 | 最小 | 割り込みコントローラ | DMAコントローラ |

---

## 4. デバイスドライバ

### 4.1 デバイスドライバの役割

デバイスドライバは、OSカーネルとハードウェアデバイスの間を仲介するソフトウェアモジュールである。ハードウェアの詳細を隠蔽し、統一的なインタフェースをカーネルに提供する。Linuxカーネルのソースコードの約70%がデバイスドライバで占められているという事実は、ドライバの重要性と多様性を物語っている。

```
デバイスドライバの位置づけ:

  ┌──────────────────────────────────────────────┐
  │  ユーザー空間 (User Space)                    │
  │  ┌──────────────────────────────────────┐    │
  │  │ アプリケーション                       │    │
  │  │ open(), read(), write(), ioctl(),    │    │
  │  │ mmap(), close()                      │    │
  │  └──────────────┬───────────────────────┘    │
  │                  │ システムコール              │
  ├──────────────────┼───────────────────────────┤
  │  カーネル空間 (Kernel Space)                   │
  │                  ▼                            │
  │  ┌──────────────────────────────────────┐    │
  │  │ VFS (Virtual File System)            │    │
  │  │ 統一インタフェース層                   │    │
  │  │ → "Everything is a file" の実装       │    │
  │  └─────────┬───────────┬────────────────┘    │
  │             │           │                     │
  │    ┌────────▼──┐  ┌────▼──────────┐          │
  │    │ ブロック   │  │ キャラクタ    │           │
  │    │ デバイス層 │  │ デバイス層    │           │
  │    └────────┬──┘  └────┬──────────┘          │
  │             │           │                     │
  │    ┌────────▼──┐  ┌────▼──────────┐          │
  │    │I/Oスケジュ│  │ TTY/入力     │           │
  │    │ ーラ      │  │ サブシステム  │           │
  │    └────────┬──┘  └────┬──────────┘          │
  │             │           │                     │
  │    ┌────────▼───────────▼──────────┐          │
  │    │ デバイスドライバ               │          │
  │    │ ┌────────┐ ┌────────┐ ┌─────┐│          │
  │    │ │SATAドライバ│ │NVMeドライバ│ │USBドライバ│          │
  │    │ └────────┘ └────────┘ └─────┘│          │
  │    └─────────────┬─────────────────┘          │
  │                  │                            │
  └──────────────────┼────────────────────────────┘
                     ▼
  ┌──────────────────────────────────────────────┐
  │  ハードウェア（SSD、NIC、GPU等）               │
  └──────────────────────────────────────────────┘
```

### 4.2 UNIXのデバイス分類

UNIXでは、デバイスを3つのカテゴリに分類する。

| 分類 | 特徴 | アクセス単位 | 例 |
|------|------|------------|-----|
| キャラクタデバイス | シーケンシャルアクセス、バッファリングなし | バイト単位 | 端末、シリアルポート、マウス |
| ブロックデバイス | ランダムアクセス、カーネルバッファリング | ブロック単位（512B/4KB） | HDD、SSD、USBメモリ |
| ネットワークデバイス | パケット単位の送受信、ソケットAPI | パケット単位 | Ethernet NIC、Wi-Fi |

```
/dev ディレクトリの構造例:

  $ ls -la /dev/sd* /dev/tty* /dev/null /dev/zero 2>/dev/null | head -20

  brw-rw---- 1 root disk    8,  0  /dev/sda      ← ブロックデバイス (b)
  brw-rw---- 1 root disk    8,  1  /dev/sda1     ← パーティション
  crw-rw-rw- 1 root root    1,  3  /dev/null     ← キャラクタデバイス (c)
  crw-rw-rw- 1 root root    1,  5  /dev/zero     ← キャラクタデバイス
  crw--w---- 1 root tty     4,  0  /dev/tty0     ← 端末デバイス

  デバイス番号 (Major, Minor):
  - Major番号: ドライバの識別（例: 8 = SCSIディスク）
  - Minor番号: デバイス内の個体識別（例: 0 = 最初のディスク）
```

### 4.3 Linuxカーネルモジュールの実装例

Linuxでは、デバイスドライバをカーネルモジュール（Loadable Kernel Module: LKM）として動的にロード・アンロードできる。

```c
/* シンプルなキャラクタデバイスドライバの例 */
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>

#define DEVICE_NAME "mychardev"
#define BUF_SIZE 1024

static dev_t dev_num;
static struct cdev my_cdev;
static struct class *my_class;
static char kernel_buf[BUF_SIZE];
static int buf_len = 0;

/* open: デバイスファイルを開いたときに呼ばれる */
static int my_open(struct inode *inode, struct file *filp)
{
    pr_info("mychardev: device opened\n");
    return 0;
}

/* read: ユーザーがデバイスからデータを読むとき */
static ssize_t my_read(struct file *filp, char __user *buf,
                       size_t count, loff_t *offset)
{
    int bytes_to_read;

    if (*offset >= buf_len)
        return 0;  /* EOF */

    bytes_to_read = min((int)count, buf_len - (int)*offset);

    /* カーネル空間 → ユーザー空間 へのコピー
       直接ポインタ経由でコピーしてはならない（セキュリティ） */
    if (copy_to_user(buf, kernel_buf + *offset, bytes_to_read))
        return -EFAULT;

    *offset += bytes_to_read;
    return bytes_to_read;
}

/* write: ユーザーがデバイスにデータを書くとき */
static ssize_t my_write(struct file *filp, const char __user *buf,
                        size_t count, loff_t *offset)
{
    int bytes_to_write = min((int)count, BUF_SIZE - 1);

    /* ユーザー空間 → カーネル空間 へのコピー */
    if (copy_from_user(kernel_buf, buf, bytes_to_write))
        return -EFAULT;

    kernel_buf[bytes_to_write] = '\0';
    buf_len = bytes_to_write;

    pr_info("mychardev: received %d bytes\n", bytes_to_write);
    return bytes_to_write;
}

/* release: デバイスファイルを閉じたときに呼ばれる */
static int my_release(struct inode *inode, struct file *filp)
{
    pr_info("mychardev: device closed\n");
    return 0;
}

/* file_operations 構造体: VFSとドライバを接続 */
static const struct file_operations my_fops = {
    .owner   = THIS_MODULE,
    .open    = my_open,
    .read    = my_read,
    .write   = my_write,
    .release = my_release,
};

/* モジュール初期化 */
static int __init my_init(void)
{
    int ret;

    /* デバイス番号を動的に確保 */
    ret = alloc_chrdev_region(&dev_num, 0, 1, DEVICE_NAME);
    if (ret < 0)
        return ret;

    /* cdev構造体を初期化し、file_operationsを登録 */
    cdev_init(&my_cdev, &my_fops);
    ret = cdev_add(&my_cdev, dev_num, 1);
    if (ret < 0) {
        unregister_chrdev_region(dev_num, 1);
        return ret;
    }

    /* /dev/mychardev を自動作成（udev連携） */
    my_class = class_create(THIS_MODULE, DEVICE_NAME);
    device_create(my_class, NULL, dev_num, NULL, DEVICE_NAME);

    pr_info("mychardev: registered with major=%d minor=%d\n",
            MAJOR(dev_num), MINOR(dev_num));
    return 0;
}

/* モジュール終了処理 */
static void __exit my_exit(void)
{
    device_destroy(my_class, dev_num);
    class_destroy(my_class);
    cdev_del(&my_cdev);
    unregister_chrdev_region(dev_num, 1);
    pr_info("mychardev: unregistered\n");
}

module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Example character device driver");
```

使用方法:
```bash
# カーネルモジュールのビルドとロード
$ make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
$ sudo insmod mychardev.ko

# デバイスの確認
$ ls -la /dev/mychardev
crw------- 1 root root 237, 0 ... /dev/mychardev

# デバイスへの書き込みと読み取り
$ echo "Hello, kernel!" | sudo tee /dev/mychardev
$ sudo cat /dev/mychardev
Hello, kernel!

# カーネルログの確認
$ dmesg | tail -5
mychardev: registered with major=237 minor=0
mychardev: device opened
mychardev: received 15 bytes
mychardev: device closed

# モジュールのアンロード
$ sudo rmmod mychardev
```

### 4.4 ユーザー空間ドライバ（UIO / VFIO）

従来のドライバはカーネル空間で動作するが、近年はユーザー空間でドライバを実装する手法が注目されている。

```
カーネルドライバ vs ユーザー空間ドライバ:

  (A) 従来のカーネルドライバ:
  ┌──────────────┐
  │ ユーザー空間  │  アプリケーション
  │              │      ↓ syscall
  ├──────────────┤───── カーネル境界 ─────
  │ カーネル空間  │  ドライバコード
  │              │      ↓ MMIO/DMA
  ├──────────────┤
  │ ハードウェア  │  デバイス
  └──────────────┘

  メリット: 全HW機能にアクセス可能
  デメリット: バグでカーネルパニック、開発が困難

  (B) ユーザー空間ドライバ (UIO/VFIO):
  ┌──────────────┐
  │ ユーザー空間  │  アプリケーション + ドライバ
  │              │      ↓ mmap() でデバイスレジスタに直接アクセス
  ├──────────────┤───── カーネル境界 ─────
  │ カーネル空間  │  薄いUIO/VFIOスタブ（割り込み通知のみ）
  ├──────────────┤
  │ ハードウェア  │  デバイス
  └──────────────┘

  メリット: バグがプロセスクラッシュで済む、デバッグ容易
  デメリット: コンテキストスイッチのオーバーヘッド

  代表的なユーザー空間ドライバフレームワーク:
  - DPDK: 高速パケット処理（Intelが主導）
  - SPDK: 高速ストレージI/O
  - VFIO: VMへのデバイスパススルー
```

---

## 5. I/Oスケジューリング

### 5.1 I/Oスケジューリングの必要性

I/Oスケジューラは、アプリケーションからの複数のI/O要求を効率的に並べ替え・統合する役割を持つ。特にHDDのようなシーク時間が支配的なデバイスでは、要求の順序を最適化することでスループットが劇的に向上する。

```
I/Oスケジューリングがない場合 vs ある場合:

  ディスク上の要求位置: [100] [500] [120] [480] [130] [510]

  (A) スケジューリングなし (FIFO):
    ヘッド移動: 100→500→120→480→130→510
    総移動量: 400 + 380 + 360 + 350 + 380 = 1870 トラック
    → ヘッドが行ったり来たり（非効率）

  (B) スケジューリングあり (SCAN):
    ヘッド移動: 100→120→130→480→500→510
    総移動量: 20 + 10 + 350 + 20 + 10 = 410 トラック
    → 一方向に順番に処理（効率的）

    改善率: (1870 - 410) / 1870 ≈ 78% 削減
```

### 5.2 古典的なディスクスケジューリングアルゴリズム

```
主要なディスクスケジューリングアルゴリズム:

(1) FCFS (First Come First Served)
    到着順に処理。公平だが非効率。

    要求キュー: 98, 183, 37, 122, 14, 124, 65, 67
    ヘッド初期位置: 53

    処理順: 53→98→183→37→122→14→124→65→67
    総移動: 45+85+146+85+108+110+59+2 = 640

(2) SSTF (Shortest Seek Time First)
    現在のヘッド位置に最も近い要求を次に処理。

    処理順: 53→65→67→37→14→98→122→124→183
    総移動: 12+2+30+23+84+24+2+59 = 236

    問題: 飢餓（starvation）が発生しうる
    → 端のトラックの要求がいつまでも処理されない

(3) SCAN (エレベータアルゴリズム)
    ヘッドが一方向に移動しながら要求を処理。
    端に到達したら反転。エレベータの動きに類似。

    処理順: 53→37→14→0→65→67→98→122→124→183
    総移動: 16+23+14+65+2+31+24+2+59 = 236

    利点: SSTFの飢餓問題を解消

(4) C-SCAN (Circular SCAN)
    一方向のみサービス。端到達後は先頭に戻る。
    応答時間の均一性が高い。

(5) LOOK / C-LOOK
    SCAN/C-SCANの改良版。端まで行かず、
    最後の要求位置で反転/リセット。
```

### 5.3 Linuxの現代的I/Oスケジューラ

Linuxカーネルは、デバイス特性に応じた複数のI/Oスケジューラを提供する。カーネル5.0以降はマルチキュー（blk-mq）ベースのスケジューラが標準となっている。

| スケジューラ | 対象デバイス | アルゴリズム | 特徴 |
|------------|------------|-----------|------|
| **none** | NVMe SSD | なし（FIFO） | オーバーヘッド最小。デバイス側にFTLあり |
| **mq-deadline** | SATA SSD / HDD | Deadline | 読み取り優先、期限保証、飢餓防止 |
| **bfq** | デスクトップ | Budget Fair Queueing | 低レイテンシ、公平性重視 |
| **kyber** | 高速SSD | トークンベース | 軽量、読み/書き/破棄の3キュー |

```bash
# 現在のI/Oスケジューラを確認
$ cat /sys/block/sda/queue/scheduler
[mq-deadline] kyber bfq none

# NVMe SSDの場合（通常はnone）
$ cat /sys/block/nvme0n1/queue/scheduler
[none] mq-deadline kyber bfq

# I/Oスケジューラを変更
$ echo "bfq" | sudo tee /sys/block/sda/queue/scheduler

# I/Oキューの深さを確認
$ cat /sys/block/nvme0n1/queue/nr_requests
1023

# I/Oスケジューラの統計情報
$ cat /sys/block/sda/queue/stat
# 読み取り: 完了数  マージ数  セクタ数  時間(ms)
# 書き込み: 完了数  マージ数  セクタ数  時間(ms)
```

### 5.4 mq-deadline スケジューラの詳細

mq-deadline（マルチキュー版Deadline）は、各I/O要求に期限（deadline）を設定し、飢餓を防止しつつスループットを最大化するスケジューラである。

```
mq-deadline の内部構造:

  アプリケーションからの要求
          │
          ▼
  ┌───────────────────────────────────────────┐
  │ ソフトウェアキュー (per-CPU)               │
  │ ┌────────────┐ ┌────────────┐             │
  │ │ CPU 0 キュー│ │ CPU 1 キュー│ ...        │
  │ └─────┬──────┘ └─────┬──────┘             │
  │       └──────┬───────┘                    │
  │              ▼                            │
  │ ┌────────────────────────────────────┐    │
  │ │ mq-deadline スケジューラ            │    │
  │ │                                    │    │
  │ │ ┌─ ソート済みキュー (セクタ順) ─┐  │    │
  │ │ │ [LBA:100] [LBA:200] [LBA:300] │  │    │
  │ │ └────────────────────────────────┘  │    │
  │ │                                    │    │
  │ │ ┌─ FIFOキュー (到着順) ─────────┐  │    │
  │ │ │ [期限:T1] [期限:T2] [期限:T3] │  │    │
  │ │ └────────────────────────────────┘  │    │
  │ │                                    │    │
  │ │ ディスパッチ判定:                    │    │
  │ │ 1. 期限切れ要求があれば最優先        │    │
  │ │ 2. なければソート済みキューから選択   │    │
  │ │ 3. 読み取りの期限 = 500ms (デフォルト)│    │
  │ │ 4. 書き込みの期限 = 5000ms           │    │
  │ │ → 読み取りを優先（対話性向上）       │    │
  │ └────────────────────────────────────┘    │
  │              │                            │
  └──────────────┼────────────────────────────┘
                 ▼
  ┌───────────────────────────────────────────┐
  │ ハードウェアディスパッチキュー              │
  │ → デバイスドライバへ                       │
  └───────────────────────────────────────────┘
```

### 5.5 BFQ（Budget Fair Queueing）の仕組み

BFQはCFQ（Completely Fair Queueing）の後継として開発されたスケジューラで、各プロセスに「バジェット」（処理可能なセクタ数）を割り当て、公平なI/O分配を実現する。デスクトップ環境でのインタラクティブ性能に優れている。

BFQの主な特徴:
- プロセスごとにI/Oバジェットを動的に調整
- アイドルタイム（待機時間）を設けてシーケンシャルI/Oを優遇
- 軽いI/O負荷のプロセス（GUIアプリ等）を自動的に優先
- 重いバックグラウンドI/O（cp, rsync等）の影響を低減

---

## 6. 非同期I/Oの進化

### 6.1 同期I/O vs 非同期I/O

```
(A) 同期I/O（ブロッキング）:

  スレッド1:  ──[read()]──────────待機──────────[データ取得]──→
  スレッド2:  ──[read()]──────────待機──────────[データ取得]──→
  スレッド3:  ──[read()]──────────待機──────────[データ取得]──→

  問題: 1接続 = 1スレッド → 10,000接続 = 10,000スレッド
  → メモリ消費: 10,000 × 8MB(スタック) = 80GB
  → コンテキストスイッチコストが膨大

(B) 非同期I/O（ノンブロッキング + イベント多重化）:

  スレッド1:  ──[要求登録]──[他の処理]──[イベント受信]──[処理]──→
              10,000接続を1スレッドで管理

  利点: メモリ効率が高い、コンテキストスイッチが少ない
  実装: select → poll → epoll → io_uring
```

### 6.2 select / poll（レガシー方式）

```c
/* select の基本的な使用例 */
#include <sys/select.h>

int main(void)
{
    fd_set read_fds;
    struct timeval timeout;
    int max_fd, nready;

    /* 1024個のFDのビットマップを毎回初期化 */
    FD_ZERO(&read_fds);
    FD_SET(sock_fd, &read_fds);
    max_fd = sock_fd;

    timeout.tv_sec = 5;
    timeout.tv_usec = 0;

    /* カーネルに全FDの状態を問い合わせ
       → FD数に比例する O(n) のスキャン */
    nready = select(max_fd + 1, &read_fds, NULL, NULL, &timeout);

    if (nready > 0 && FD_ISSET(sock_fd, &read_fds)) {
        /* データ読み取り可能 */
        read(sock_fd, buf, sizeof(buf));
    }

    return 0;
}

/*
  select の制限:
  - FD_SETSIZE = 1024（コンパイル時固定）
  - 毎回 fd_set をカーネルにコピー（O(n)）
  - カーネル内で全FDをスキャン（O(n)）
  - 結果の fd_set をユーザー空間にコピー（O(n)）
  → 同時接続数が増えると性能が線形に劣化
*/
```

### 6.3 epollの仕組みと実装

epollはLinux 2.6で導入された高性能I/Oイベント通知メカニズムである。selectの根本的な問題を解決し、O(1)でのイベント通知を実現する。

```
epoll の内部動作:

  ユーザー空間                        カーネル空間
  ┌─────────────────┐                ┌──────────────────────┐
  │ アプリケーション  │                │ epoll インスタンス     │
  │                  │ epoll_create() │                      │
  │ (1) epollインスタ│──────────────→│ ┌──────────────────┐ │
  │     ンス作成     │                │ │ Red-Black Tree   │ │
  │                  │                │ │ (監視FD管理)      │ │
  │                  │ epoll_ctl()    │ └──────────────────┘ │
  │ (2) FDを登録    │──────────────→│                      │
  │  (ADD/MOD/DEL)  │                │ ┌──────────────────┐ │
  │                  │                │ │ Ready List       │ │
  │                  │ epoll_wait()   │ │ (準備完了FDリスト) │ │
  │ (3) イベント待機 │──────────────→│ └──────────────────┘ │
  │     (ブロック)   │                │                      │
  │                  │                │ デバイスからの割り込み │
  │                  │                │ → コールバックで      │
  │                  │                │   Ready Listに追加   │
  │                  │                │                      │
  │ (4) Ready FDの  │←── Ready FDs ─│ Ready Listから返却    │
  │     みを返却    │                │                      │
  └─────────────────┘                └──────────────────────┘

  selectとの本質的な違い:
  ┌────────────────────────────────────────────────────────┐
  │ select: 毎回「全FDを調べてくれ」と依頼                    │
  │   → カーネルが10,000個のFDを毎回スキャン                  │
  │   → O(n) × 呼び出し回数                                 │
  │                                                        │
  │ epoll: 「変化があったFDだけ教えてくれ」                    │
  │   → カーネルがコールバックでReady Listに追加              │
  │   → epoll_wait は Ready List を返すだけ                  │
  │   → O(1) のイベント通知                                  │
  └────────────────────────────────────────────────────────┘
```

**epollを使った高性能TCPサーバの例:**

```c
/* epoll を使った echo サーバー */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <errno.h>

#define MAX_EVENTS 1024
#define BUF_SIZE   4096
#define PORT       8080

/* ソケットをノンブロッキングに設定 */
static void set_nonblocking(int fd)
{
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

int main(void)
{
    int listen_fd, epoll_fd, nfds, i;
    struct epoll_event ev, events[MAX_EVENTS];
    struct sockaddr_in addr;

    /* リスニングソケット作成 */
    listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(PORT);

    bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr));
    listen(listen_fd, SOMAXCONN);
    set_nonblocking(listen_fd);

    /* epoll インスタンス作成 */
    epoll_fd = epoll_create1(0);

    /* リスニングソケットを epoll に登録 */
    ev.events = EPOLLIN;
    ev.data.fd = listen_fd;
    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, listen_fd, &ev);

    printf("Echo server listening on port %d\n", PORT);

    /* イベントループ */
    for (;;) {
        /* 準備完了のFDを待つ（タイムアウト: -1 = 無限待ち） */
        nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);

        for (i = 0; i < nfds; i++) {
            if (events[i].data.fd == listen_fd) {
                /* 新しい接続を受け付け */
                int client_fd = accept(listen_fd, NULL, NULL);
                if (client_fd < 0) continue;

                set_nonblocking(client_fd);
                ev.events = EPOLLIN | EPOLLET;  /* エッジトリガ */
                ev.data.fd = client_fd;
                epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &ev);

            } else {
                /* クライアントからのデータを処理 */
                char buf[BUF_SIZE];
                ssize_t n = read(events[i].data.fd, buf, sizeof(buf));

                if (n <= 0) {
                    /* 接続終了 or エラー */
                    epoll_ctl(epoll_fd, EPOLL_CTL_DEL,
                              events[i].data.fd, NULL);
                    close(events[i].data.fd);
                } else {
                    /* エコーバック */
                    write(events[i].data.fd, buf, n);
                }
            }
        }
    }

    close(listen_fd);
    close(epoll_fd);
    return 0;
}
```

### 6.4 epollのトリガモード

| モード | 動作 | 特徴 | 用途 |
|--------|------|------|------|
| レベルトリガ (LT) | データが残っている限り通知し続ける | selectと互換。取りこぼしが起きにくい | デフォルト。一般用途 |
| エッジトリガ (ET) | 状態が変化した時だけ1回通知 | 高効率だがデータ取り残しに注意 | 高性能サーバー（Nginx） |

エッジトリガ使用時は、通知を受けたらEAGAINが返るまでループで全データを読み切る必要がある。これを怠ると、データが残っているにもかかわらず次の通知が来ず、接続がハングする。

### 6.5 io_uring（Linux 5.1+）

io_uringは2019年にLinuxカーネル5.1で導入された革新的な非同期I/Oインタフェースである。従来のepollやAIO（Linux AIO）の制限を克服し、ゼロコピーに近い性能でファイルI/OとネットワークI/Oを統一的に扱える。

```
io_uring のアーキテクチャ:

  ┌─────────────────────────────────────────────────────────┐
  │                  共有メモリ領域                           │
  │                                                         │
  │  Submission Queue (SQ)           Completion Queue (CQ)  │
  │  ┌───┬───┬───┬───┬───┐         ┌───┬───┬───┬───┬───┐  │
  │  │SQE│SQE│SQE│   │   │         │CQE│CQE│   │   │   │  │
  │  │ 0 │ 1 │ 2 │   │   │         │ 0 │ 1 │   │   │   │  │
  │  └─▲─┴───┴───┴───┴───┘         └───┴───┴─│─┴───┴───┘  │
  │    │                                      │             │
  │    │ ユーザーが投入                 カーネルが│結果記入     │
  │    │ (syscall不要!)                        │             │
  │    │                                      ▼             │
  └────┼──────────────────────────────────────┼─────────────┘
       │                                      │
  ┌────┴──────────┐                    ┌──────┴────────┐
  │ ユーザー空間   │                    │ ユーザー空間   │
  │ SQEを作成して │                    │ CQEを読んで   │
  │ リングに投入   │                    │ 結果を処理     │
  └───────────────┘                    └───────────────┘

  SQE (Submission Queue Entry) の構造:
  ┌────────────────────────────────────┐
  │ opcode:   IORING_OP_READ          │ ← 操作種別
  │ fd:       ファイルディスクリプタ     │
  │ addr:     バッファアドレス          │
  │ len:      転送バイト数             │
  │ offset:   ファイルオフセット        │
  │ user_data: ユーザー識別子          │ ← CQEと対応付け
  └────────────────────────────────────┘

  CQE (Completion Queue Entry) の構造:
  ┌────────────────────────────────────┐
  │ user_data: ユーザー識別子          │ ← SQEと同じ値
  │ res:       結果（バイト数 or エラー）│
  └────────────────────────────────────┘
```

**io_uringの革新的な点:**

1. **システムコールの削減:** SQへの投入はユーザー空間からの共有メモリ書き込みだけで完了する。`io_uring_enter()` も `SQPOLL` モードではカーネルスレッドが自動的にSQを監視するため不要になる。

2. **バッチ処理:** 複数のI/O要求を一度にまとめてSQに投入できる。従来は1要求 = 1システムコールだったのに対し、io_uringでは数百の要求を0〜1回のシステムコールで処理できる。

3. **統一インタフェース:** ファイル読み書き、ネットワーク送受信、タイマー、ファイル同期（fsync）などを同じリングバッファで扱える。

### 6.6 I/O多重化方式の総合比較

| 方式 | 導入年 | 計算量 | FD上限 | 機能 | 主な利用先 |
|------|--------|--------|--------|------|----------|
| **select** | 1983 | O(n) | 1024 | 基本的な多重化 | レガシーシステム |
| **poll** | 1986 | O(n) | なし | selectの拡張 | 小規模サーバー |
| **epoll** | 2002 | O(1) | なし | イベント駆動 | Nginx, Redis, Node.js |
| **kqueue** | 2000 | O(1) | なし | epoll相当（BSD） | macOS, FreeBSD |
| **IOCP** | 2000 | O(1) | なし | プロアクターモデル | Windows (.NET) |
| **io_uring** | 2019 | O(1) | なし | ゼロコピー、統一API | 高性能DB, ストレージ |

---

## 7. 実務でのI/O最適化

### 7.1 I/O性能測定ツール

I/Oのボトルネックを特定するためのLinuxツール群を理解する。

```bash
# (1) iostat: デバイスレベルのI/O統計
$ iostat -xz 1
Device  r/s    w/s   rkB/s   wkB/s  rrqm/s  wrqm/s  %util  await
sda     150    50    6000    2000   10      30       65%    4.2
nvme0n1 5000   3000  200000  120000 0       0        40%    0.1

# 注目すべき指標:
# - %util: デバイス使用率。100%に近いと飽和
# - await: 平均I/O待ち時間(ms)。高いとボトルネック
# - r/s, w/s: 1秒あたりの読み書きI/O数

# (2) blktrace + blkparse: ブロックI/Oの詳細トレース
$ sudo blktrace -d /dev/sda -o - | blkparse -i -
  8,0  1  1  0.000000000  1234  Q  R  100 + 8  [myapp]
  8,0  1  2  0.000001000  1234  G  R  100 + 8  [myapp]
  8,0  1  3  0.000005000  1234  D  R  100 + 8  [myapp]
  8,0  1  4  0.004200000  1234  C  R  100 + 8  [0]

# Q=キューイング, G=取得, D=ディスパッチ, C=完了

# (3) strace: システムコールのトレース
$ strace -e trace=read,write,open,close -c ./myapp
% time     seconds  usecs/call     calls    errors  syscall
------ ----------- ----------- --------- --------- --------
 85.30    1.234567          12    102400           read
 10.20    0.147654           8     18000           write
  4.50    0.065123          65      1000           open

# (4) perf: I/O関連イベントのプロファイリング
$ sudo perf record -e block:block_rq_insert,block:block_rq_complete -a
$ sudo perf report
```

### 7.2 ゼロコピー技術

従来のファイル送信では、データがカーネルバッファとユーザーバッファの間で何度もコピーされる。ゼロコピー技術はこの不要なコピーを排除する。

```
従来の read() + write() (4回コピー):

  ディスク → [DMA] → カーネルバッファ → [CPU] → ユーザーバッファ
                                                      │
  ソケット ← [DMA] ← カーネルバッファ ← [CPU] ← ユーザーバッファ

  コピー回数: 4回
  コンテキストスイッチ: 4回（read×2 + write×2）

sendfile() によるゼロコピー (2回コピー):

  ディスク → [DMA] → カーネルバッファ ──[CPU]──→ ソケットバッファ
                                                     │
  ソケット ← [DMA] ←─────────────────────────────────┘

  コピー回数: 2回（ユーザー空間を経由しない）
  コンテキストスイッチ: 2回

splice() / sendfile() + DMA Scatter-Gather:

  ディスク → [DMA] → カーネルバッファ ──参照情報──→ ソケットバッファ
                         │                              │
                         └──────── [DMA] ───────────→ NIC

  コピー回数: 0回（CPU コピーなし、DMA のみ）
  コンテキストスイッチ: 2回
```

### 7.3 Node.js のイベントループとI/O

```javascript
/*
 * Node.js のイベントループの内部構造
 * libuv → epoll (Linux) / kqueue (macOS) / IOCP (Windows)
 */

const fs = require('fs');
const http = require('http');

/*
 * ファイルI/O: libuvのスレッドプールで実行
 * （epollはファイルI/Oに対応していないため）
 */
fs.readFile('/path/to/large-file', (err, data) => {
    /* スレッドプール内のワーカーがread()を実行
       完了後、コールバックをイベントキューに投入 */
    if (err) throw err;
    console.log(`Read ${data.length} bytes`);
});

/*
 * ネットワークI/O: epoll/kqueue で直接多重化
 * スレッドプール不使用（ノンブロッキングソケット）
 */
const server = http.createServer((req, res) => {
    /* 数万の同時接続を1スレッドで処理可能 */
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('Hello, World!\n');
});
server.listen(3000);

/*
 * イベントループの各フェーズ:
 *
 *   ┌───────────────────────────┐
 *   │      timers               │  ← setTimeout, setInterval
 *   ├───────────────────────────┤
 *   │      pending callbacks    │  ← I/Oコールバック（一部）
 *   ├───────────────────────────┤
 *   │      idle, prepare        │  ← 内部処理
 *   ├───────────────────────────┤
 *   │      poll                 │  ← epoll_wait() でI/O待機
 *   │      (I/Oイベント取得)     │     新しいI/Oコールバックを実行
 *   ├───────────────────────────┤
 *   │      check                │  ← setImmediate()
 *   ├───────────────────────────┤
 *   │      close callbacks      │  ← socket.on('close', ...)
 *   └───────────────────────────┘
 *          ↑                │
 *          └────────────────┘  (ループ)
 *
 * process.nextTick() は各フェーズの間に割り込んで実行
 */
```

---

## 8. アンチパターンと対策

### 8.1 アンチパターン1: 同期I/Oの安易な使用

```
問題: Webサーバーでリクエストごとに同期ファイル読み取り

  ┌──────────────────────────────────────────────────────┐
  │ BAD: 同期I/Oでブロッキング                             │
  │                                                      │
  │  リクエスト1: ──[read(file)]───── 50ms待ち ────→処理   │
  │  リクエスト2: ────────────────── 待機 ──────────→      │
  │  リクエスト3: ────────────────── 待機 ──────────→      │
  │                                                      │
  │  100リクエスト/秒なら平均応答時間: 2.5秒               │
  │  → スレッドが1つだとI/O待ちで詰まる                    │
  └──────────────────────────────────────────────────────┘

  根本原因:
  - read() / write() はデフォルトでブロッキング
  - スレッド数が少ないとI/O待ちで処理が詰まる
  - スレッドを増やすとメモリ消費とコンテキストスイッチが増大

  ┌──────────────────────────────────────────────────────┐
  │ GOOD: 非同期I/O + イベント駆動                         │
  │                                                      │
  │  リクエスト1: ──[async read]──→ [他の処理] ──→ 完了   │
  │  リクエスト2: ──[async read]──→ [他の処理] ──→ 完了   │
  │  リクエスト3: ──[async read]──→ [他の処理] ──→ 完了   │
  │                                                      │
  │  1スレッドで数万リクエストを並行処理                     │
  │  → Nginx, Node.js, Go が採用するモデル                 │
  └──────────────────────────────────────────────────────┘

  対策:
  1. ノンブロッキングI/O + epoll/kqueue を使用
  2. async/await パターンを活用（Python asyncio, Rust tokio）
  3. スレッドプールでI/O操作をオフロード（Java NIO, libuv）
```

### 8.2 アンチパターン2: DMAバッファのキャッシュ管理忘れ

```
問題: DMAバッファのキャッシュ一貫性を考慮しないドライバ実装

  ┌──────────────────────────────────────────────────────┐
  │ BAD: kmalloc + virt_to_phys で直接DMAアドレス取得     │
  │                                                      │
  │   buf = kmalloc(4096, GFP_KERNEL);                   │
  │   dma_addr = virt_to_phys(buf);  /* 危険! */         │
  │   /* キャッシュ一貫性が保証されない */                  │
  │   /* IOMMU非対応、バウンスバッファ未考慮 */             │
  │   /* 32ビットDMAデバイスで4GB以上にアクセス不可 */      │
  │                                                      │
  │ 症状:                                                │
  │ - データ化けが「たまに」発生（再現困難）                │
  │ - 特定のハードウェア構成でのみクラッシュ                │
  │ - 負荷が高い時だけ問題が顕在化                         │
  └──────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────┐
  │ GOOD: DMA API を正しく使用                             │
  │                                                      │
  │   /* コヒーレントDMAバッファ確保 */                     │
  │   buf = dma_alloc_coherent(dev, 4096,                │
  │                            &dma_handle, GFP_KERNEL); │
  │                                                      │
  │   /* または、ストリーミングDMAマッピング */              │
  │   dma_handle = dma_map_single(dev, buf, 4096,        │
  │                                DMA_FROM_DEVICE);     │
  │   /* I/O完了後 */                                     │
  │   dma_unmap_single(dev, dma_handle, 4096,            │
  │                     DMA_FROM_DEVICE);                 │
  │                                                      │
  │ 利点:                                                │
  │ - キャッシュ一貫性を自動的に保証                       │
  │ - IOMMUとの連携（仮想化環境で必須）                    │
  │ - バウンスバッファの自動処理                            │
  └──────────────────────────────────────────────────────┘
```

### 8.3 アンチパターン3: I/Oスケジューラの不適切な選択

```
問題: NVMe SSDに対してBFQスケジューラを使用

  ┌──────────────────────────────────────────────────────┐
  │ BAD: NVMe SSD + BFQ                                  │
  │                                                      │
  │ NVMe SSD の特性:                                      │
  │ - ランダムアクセスとシーケンシャルアクセスの差が小さい   │
  │ - ハードウェアに複数キュー（最大65535）を持つ           │
  │ - 内部FTLがI/O最適化を行う                            │
  │                                                      │
  │ BFQ のオーバーヘッド:                                  │
  │ - プロセスごとのバジェット計算                          │
  │ - 要求のソートとマージ                                 │
  │ - → SSDには不要な処理で性能が低下                      │
  │ - → 高IOPS環境でCPUボトルネックに                     │
  └──────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────┐
  │ GOOD: デバイス特性に合ったスケジューラを選択            │
  │                                                      │
  │ NVMe SSD     → none（スケジューラなし）                │
  │ SATA SSD     → mq-deadline                           │
  │ HDD          → mq-deadline or bfq                    │
  │ デスクトップ用 → bfq（対話性重視）                      │
  │ サーバー用    → mq-deadline（スループット重視）          │
  └──────────────────────────────────────────────────────┘
```

---

## 9. 実践演習

### 演習1: I/O方式の選択（基礎）

以下のデバイスと使用シナリオに対して、最適なI/O方式（ポーリング、割り込み、DMA）を選び、その理由を述べよ。

1. キーボード入力（ユーザーが文字を入力）
2. NVMe SSD のランダム4KB読み取り（100万IOPS環境）
3. 10Gbpsネットワーク受信（大容量ファイル転送）
4. 温度センサーの定期読み取り（1秒間隔、組み込みシステム）
5. GPU からの大容量フレームバッファ転送（4K 60fps）

**模範解答:**

1. **キーボード → 割り込み方式**
   理由: 入力頻度が低く（数十〜数百回/秒）、ポーリングではCPU浪費が大きい。割り込みなら入力があった瞬間だけCPUが反応する。DMAは転送データ量が極めて少ない（1〜数バイト）ため不要。

2. **NVMe SSD 高IOPS → ポーリング方式（io_poll）**
   理由: I/O完了時間が〜10μsであるのに対し、割り込みのオーバーヘッドが1〜5μs。割り込みコストが相対的に大きいため、ポーリングでレイテンシを最小化する。Linux の `io_poll` フラグがこの用途に相当する。

3. **10Gbps大容量転送 → DMA + 割り込み**
   理由: 転送データ量が膨大（〜1.25GB/s）であり、CPUを介してコピーすると帯域を消費しきれない。DMAでNIC→メモリへ直接転送し、完了を割り込みで通知する。さらにNAPI（Linux）ではパケット到着時のみ割り込み、以降はポーリングに切り替えるハイブリッド方式を採用する。

4. **温度センサー → ポーリング方式**
   理由: 1秒間隔の定期読み取りであり、タイマー割り込みでポーリングを駆動すれば十分。割り込みを使うほどの即応性は不要。組み込みシステムでは割り込みコントローラのリソースが限られるため、ポーリングが合理的。

5. **GPU フレームバッファ → DMA**
   理由: 4K 60fpsのフレームデータは 3840×2160×4B×60 ≈ 1.99GB/sに達する。CPUを介した転送では帯域不足になるため、PCIe経由のDMA（Bus Master DMA）で転送する。GPUが自律的にDMAを制御し、VSync割り込みで完了を通知する。

### 演習2: epollサーバーの実装（応用）

以下の要件を満たすチャットサーバーをepollを使って実装せよ。

- 最大10,000同時接続をサポート
- エッジトリガモードを使用
- あるクライアントからのメッセージを全クライアントにブロードキャスト
- 接続/切断をログに出力

ヒント: エッジトリガではEAGAINまでループで読み切ること。接続リストの管理にはハッシュテーブルまたは配列を使用する。

### 演習3: I/O性能分析（発展）

以下の手順でI/Oボトルネック分析を実施せよ。

1. `fio` を使って対象ストレージのベースライン性能を測定する
```bash
# シーケンシャル読み取り
fio --name=seq-read --rw=read --bs=1M --size=1G \
    --numjobs=1 --ioengine=libaio --direct=1 --runtime=30

# ランダム読み取り（4KB）
fio --name=rand-read --rw=randread --bs=4k --size=1G \
    --numjobs=4 --iodepth=32 --ioengine=libaio --direct=1 --runtime=30

# io_uringエンジンでの比較
fio --name=io-uring --rw=randread --bs=4k --size=1G \
    --numjobs=4 --iodepth=32 --ioengine=io_uring --direct=1 --runtime=30
```

2. `strace -c` で対象アプリケーションのシステムコール分布を確認する
3. `iostat -xz 1` でデバイス使用率とI/O待ち時間を確認する
4. ボトルネックの原因を特定し、改善策を提案する

---

## 10. FAQ

### Q1: 「全てはファイル」とはどういう意味ですか？

**A**: UNIXの設計思想の一つで、デバイス（`/dev/sda`）、プロセス情報（`/proc/`）、カーネルパラメータ（`/sys/`）、ネットワークソケットなど、ほぼ全てのリソースを「ファイル」として抽象化し、`open()`、`read()`、`write()`、`close()` という統一的なAPIでアクセスできるようにするという考え方である。

この思想により、ファイル操作のノウハウがそのままデバイス操作に転用でき、パイプやリダイレクションによるプログラム間連携も容易になる。Plan 9（UNIXの後継研究OS）では、ネットワーク通信さえファイルシステム経由で行う設計が採用された。

### Q2: async/awaitとepollの関係は何ですか？

**A**: `async/await` はプログラミング言語の構文糖（シンタックスシュガー）であり、その裏ではepoll（Linux）やkqueue（macOS）ベースのイベントループが動作している。`await` を呼ぶと、現在の関数の実行が中断（サスペンド）され、対象のI/O操作がepollに登録される。epoll_wait() がI/O完了を検知すると、中断された関数が再開（レジューム）される。

具体的な対応関係:
- Python asyncio → epoll_wait()（Linux）/ kqueue（macOS）をラップ
- Rust tokio → epoll(Linux) / kqueue(macOS) / IOCP(Windows)をラップ
- Go goroutine → netpoller（epoll/kqueueのGo独自ラッパー）を内部使用
- Node.js → libuv（epoll/kqueue/IOCPの抽象化ライブラリ）を使用

### Q3: io_uringはいつ使うべきですか？

**A**: io_uringが特に有効なのは以下のケースである。

1. **高スループットストレージ:** NVMe SSD の性能を最大限に引き出したい場合。従来のlibaioでは1要求ごとにシステムコールが発生するが、io_uringではバッチ投入でシステムコールを削減できる。
2. **データベースエンジン:** RocksDB、ScyllaDB、TiKVなどがio_uringを採用し、書き込みレイテンシの削減を実現している。
3. **ファイルサーバー:** 大量のファイルI/Oを並行処理する場合。

一方、一般的なWebアプリケーションではepoll（Node.js、Nginx）で十分な場合が多い。io_uringの恩恵が顕著になるのは、I/O操作が毎秒数十万回以上の規模になるケースである。また、io_uringにはセキュリティ上の懸念があり、一部のLinuxディストリビューション（Ubuntu等）ではデフォルトで非特権ユーザーからの使用が制限されている点にも留意が必要である。

### Q4: NVMe SSD にI/Oスケジューラは不要ですか？

**A**: 多くの場合、NVMe SSD には `none`（スケジューラなし）が最適である。その理由は3つある。

第一に、NVMe SSD はHDDと異なりシーク時間がないため、要求の並べ替えによる性能向上が見込めない。第二に、NVMe SSD は内部にFTL（Flash Translation Layer）を持ち、デバイス内部で独自のI/O最適化を行っている。第三に、NVMe はハードウェアレベルで最大65535個のキューをサポートしており、ソフトウェアスケジューラを介さず直接ハードウェアキューにディスパッチした方が効率的である。

ただし、マルチテナント環境（クラウドVM等）でI/O公平性が求められる場合は、`mq-deadline` や `bfq` を設定することもある。

### Q5: Windowsの I/O Completion Port (IOCP) と epoll の違いは何ですか？

**A**: 根本的な設計思想が異なる。

epollは「リアクターモデル」に基づく。「I/Oが可能になったら通知してくれ」とカーネルに依頼し、通知を受けたアプリケーションが自分でI/Oを実行する（`epoll_wait()` → `read()`）。

IOCPは「プロアクターモデル」に基づく。「このI/Oを実行しておいてくれ」とカーネルに依頼し、カーネルがI/Oを完了した後に結果を通知する（`GetQueuedCompletionStatus()` で完了結果を受け取る）。

IOCPの方がアプリケーション側のコードは簡潔になるが、OSの内部実装は複雑になる。性能面では大きな差はなく、プラットフォームの選択に依存する。

---

## 11. まとめ

| 概念 | ポイント |
|------|---------|
| I/Oアドレッシング | PMIOはレガシー、MMIOが現代の主流。PCIe BARでマッピング |
| バスアーキテクチャ | PCIeが標準。帯域幅 = レーン数 × 転送レート × 符号化効率 |
| ポーリング | 単純で低レイテンシだがCPU浪費。DPDK、NVMe io_pollで現役 |
| 割り込み | CPUを効率的に使える一般的な方式。Top Half / Bottom Half で分離 |
| DMA | CPU介さず大容量転送。SG-DMA、キャッシュ一貫性に注意 |
| デバイスドライバ | VFS経由の統一インタフェース。LKMで動的ロード可能 |
| I/Oスケジューラ | NVMe→none、SATA SSD/HDD→mq-deadline、デスクトップ→bfq |
| epoll | O(1)イベント通知、C10K問題の解決。Nginx/Redis/Node.jsの基盤 |
| io_uring | ゼロコピー非同期I/O。共有リングバッファでsyscall削減 |
| ゼロコピー | sendfile/splice でCPUコピーを排除。高帯域転送に必須 |

---

## 次に読むべきガイド

→ [[06-pcb-and-circuits.md]] — 電子回路と半導体の基礎

---

## 参考文献

1. Love, R. *Linux Kernel Development.* 3rd Edition, Addison-Wesley, 2010. — Linuxカーネルの割り込み処理、デバイスドライバ、メモリ管理を包括的に解説した定番書。
2. Arpaci-Dusseau, R. H. and Arpaci-Dusseau, A. C. *Operating Systems: Three Easy Pieces (OSTEP).* Chapter 36: I/O Devices, Chapter 37: Hard Disk Drives. https://pages.cs.wisc.edu/~remzi/OSTEP/ — OSの教科書として世界中の大学で採用。I/Oデバイスの章はポーリング・割り込み・DMAの理解に最適。
3. Axboe, J. "Efficient I/O with io_uring." Linux kernel documentation, 2019. https://kernel.dk/io_uring.pdf — io_uringの設計者Jens Axboe自身による技術解説文書。リングバッファの設計思想とベンチマーク結果を詳述。
4. Stevens, W. R. and Rago, S. A. *Advanced Programming in the UNIX Environment.* 3rd Edition, Addison-Wesley, 2013. — UNIXにおけるI/Oプログラミングの決定版。select/poll/epollの歴史的文脈と実装詳細。
5. Corbet, J., Rubini, A., and Kroah-Hartman, G. *Linux Device Drivers.* 3rd Edition, O'Reilly, 2005. https://lwn.net/Kernel/LDD3/ — Linuxデバイスドライバ開発のバイブル。キャラクタデバイス、ブロックデバイス、DMA、割り込みの実装を網羅。オンラインで無料公開。
6. Tanenbaum, A. S. and Bos, H. *Modern Operating Systems.* 4th Edition, Pearson, 2014. — I/Oソフトウェアの階層構造（割り込みハンドラ→デバイスドライバ→デバイス独立ソフトウェア→ユーザー空間）の解説が秀逸。
7. Patterson, D. A. and Hennessy, J. L. *Computer Organization and Design: The Hardware/Software Interface.* 6th Edition, Morgan Kaufmann, 2020. — バスプロトコル、I/Oインタフェース、DMAの仕組みをハードウェアの視点から解説。

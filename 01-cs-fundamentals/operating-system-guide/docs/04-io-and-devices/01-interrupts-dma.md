# 割り込みとDMA — CPU・デバイス間通信の全体像

> **割り込み (Interrupt)** はCPUと外部デバイスが非同期に通信するための根幹メカニズムであり、**DMA (Direct Memory Access)** はCPUの介在なしに高速なデータ転送を実現する技術である。この2つを正しく理解することは、OSカーネル開発、デバイスドライバ設計、組み込みシステム構築、さらにはパフォーマンスチューニングの全てにおいて不可欠である。

---

## この章で学ぶこと

- [ ] 割り込みの分類（ハードウェア割り込み・ソフトウェア割り込み・例外）を体系的に理解する
- [ ] 割り込みベクタテーブル (IDT) の構造とルックアップ手順を説明できる
- [ ] Linux カーネルにおけるトップハーフ/ボトムハーフの分離設計を理解する
- [ ] DMA転送の初期化・実行・完了通知の一連の流れを追跡できる
- [ ] スキャッタ/ギャザーDMAとIOMMUの役割を説明できる
- [ ] MSI/MSI-X、RDMA、NVMe など現代のI/O技術の位置づけを把握する
- [ ] 割り込みアフィニティやIRQバランシングによる性能最適化を実践できる
- [ ] 代表的なアンチパターンを認識し、回避策を設計できる

---

## 全体アーキテクチャ俯瞰図

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Space                                  │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│   │ App (A)  │  │ App (B)  │  │ App (C)  │  │ App (D)  │          │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│        │ syscall      │ syscall     │ read()      │ write()        │
├────────┼──────────────┼─────────────┼─────────────┼────────────────┤
│        ▼              ▼             ▼             ▼                 │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │              VFS (Virtual File System)                    │     │
│   └──────────────────────┬───────────────────────────────────┘     │
│                          ▼                                          │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │           Block / Character Device Layer                  │     │
│   │  ┌────────────┐  ┌─────────────┐  ┌──────────────┐      │     │
│   │  │ I/O        │  │ Interrupt   │  │ DMA          │      │     │
│   │  │ Scheduler  │  │ Handler     │  │ Engine       │      │     │
│   │  └─────┬──────┘  └──────┬──────┘  └──────┬───────┘      │     │
│   └────────┼────────────────┼────────────────┼───────────────┘     │
│            ▼                ▼                ▼                       │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │              Hardware Abstraction Layer                    │     │
│   │   ┌──────┐  ┌───────┐  ┌────────┐  ┌──────────┐        │     │
│   │   │ PIC/ │  │ APIC  │  │ IOMMU  │  │ DMA      │        │     │
│   │   │ 8259 │  │       │  │        │  │ Controller│        │     │
│   │   └──┬───┘  └───┬───┘  └───┬────┘  └────┬─────┘        │     │
│   └──────┼──────────┼──────────┼────────────┼───────────────┘     │
│          ▼          ▼          ▼            ▼                       │
│                      Kernel Space                                   │
├─────────────────────────────────────────────────────────────────────┤
│                     Hardware Bus (PCIe / AHB / AXI)                 │
│   ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐     │
│   │Keyboard│  │  NIC   │  │  NVMe  │  │  GPU   │  │ Timer  │     │
│   │        │  │        │  │  SSD   │  │        │  │        │     │
│   └────────┘  └────────┘  └────────┘  └────────┘  └────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

この図が示す通り、ユーザ空間のアプリケーションがI/Oリクエストを発行すると、VFSを経由してデバイス層に到達し、最終的にハードウェアとの通信は割り込みとDMAによって制御される。本章ではこの一連の仕組みを深く掘り下げる。

---

## 1. 割り込みの基礎概念

### 1.1 なぜ割り込みが必要なのか

CPUとI/Oデバイスの間には、桁違いの速度差が存在する。

| 動作 | 所要サイクル（概算） | 所要時間（概算） |
|------|---------------------|-----------------|
| CPUレジスタアクセス | 1 サイクル | 0.3 ns |
| L1キャッシュヒット | 4 サイクル | 1.2 ns |
| L3キャッシュヒット | 40 サイクル | 12 ns |
| メインメモリアクセス | 200 サイクル | 60 ns |
| SSD (NVMe) 読み取り | — | 10-100 us |
| HDD シーク | — | 3-10 ms |
| ネットワーク往復 (LAN) | — | 0.1-1 ms |
| ネットワーク往復 (WAN) | — | 10-100 ms |

もし割り込みがなければ、CPUはデバイスの準備完了を繰り返しチェックする **ポーリング (Polling)** を行うしかない。ポーリングでは、デバイスが応答するまでCPUサイクルが無駄に消費される。

```
ポーリング vs 割り込み:

  [ポーリング方式]
  CPU: チェック → 未完了 → チェック → 未完了 → ... → 完了 → 処理
       ^^^^^^^^   ^^^^^^^^   ^^^^^^^^
       CPUサイクル浪費（ビジーウェイト）

  [割り込み方式]
  CPU: I/O要求発行 → 他タスク実行 → ... → 割り込み受信 → 処理
                     ^^^^^^^^^^^^^^^^^^^^^
                     CPUを有効活用
```

ただし、割り込みが万能というわけではない。高頻度I/O（10Gbps NIC でのパケット受信など）では、割り込みのオーバーヘッド自体がボトルネックになる場合がある。この問題に対しては、後述するポーリングモード（NAPI）やハイブリッド方式で対処する。

### 1.2 割り込みの3大分類

割り込みは発生源と性質に基づいて3種類に分類される。

```
┌──────────────────────────────────────────────────────────────┐
│                    割り込みの分類体系                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────┐            │
│  │ 1. ハードウェア割り込み（外部割り込み）       │            │
│  │    発生源: 外部デバイス                       │            │
│  │    ┌────────────────┬───────────────────┐    │            │
│  │    │ マスカブル      │ ノンマスカブル     │    │            │
│  │    │ (INTR)         │ (NMI)             │    │            │
│  │    │ CLI命令で      │ 無視不可           │    │            │
│  │    │ 禁止可能       │ メモリパリティ     │    │            │
│  │    │ キーボード     │ エラー、ウォッチ   │    │            │
│  │    │ ディスク完了   │ ドッグタイマー     │    │            │
│  │    │ NIC受信        │ ハードウェア障害   │    │            │
│  │    └────────────────┴───────────────────┘    │            │
│  └─────────────────────────────────────────────┘            │
│                                                              │
│  ┌─────────────────────────────────────────────┐            │
│  │ 2. ソフトウェア割り込み（トラップ）           │            │
│  │    発生源: プログラム命令                     │            │
│  │    - INT n 命令 (x86: int 0x80)              │            │
│  │    - SYSCALL / SYSENTER 命令                  │            │
│  │    - SVC 命令 (ARM)                           │            │
│  │    - デバッグブレークポイント (INT 3)          │            │
│  └─────────────────────────────────────────────┘            │
│                                                              │
│  ┌─────────────────────────────────────────────┐            │
│  │ 3. 例外 (Exception)                          │            │
│  │    発生源: CPU内部                            │            │
│  │    ┌──────────┬──────────┬──────────┐        │            │
│  │    │ Fault    │ Trap     │ Abort    │        │            │
│  │    │ 再実行可 │ 次命令   │ 復帰不可 │        │            │
│  │    │ ページ   │ オーバー │ ダブル   │        │            │
│  │    │ フォルト │ フロー   │ フォルト │        │            │
│  │    │ GPF      │ デバッグ │ マシン   │        │            │
│  │    │          │ トラップ │ チェック │        │            │
│  │    └──────────┴──────────┴──────────┘        │            │
│  └─────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

#### 例外のサブ分類の詳細

| 分類 | 戻り先 | 復帰可能性 | 代表例 | x86ベクタ番号 |
|------|--------|-----------|--------|--------------|
| Fault | 例外発生命令 | 再実行可能 | ページフォルト (#PF) | 14 |
| Fault | 例外発生命令 | 再実行可能 | 一般保護例外 (#GP) | 13 |
| Fault | 例外発生命令 | 再実行可能 | ゼロ除算 (#DE) | 0 |
| Trap | 次の命令 | 続行可能 | ブレークポイント (#BP) | 3 |
| Trap | 次の命令 | 続行可能 | オーバーフロー (#OF) | 4 |
| Abort | なし | 復帰不可 | ダブルフォルト (#DF) | 8 |
| Abort | なし | 復帰不可 | マシンチェック (#MC) | 18 |

### 1.3 x86/x86-64 割り込みベクタテーブル

x86アーキテクチャでは、割り込みは **IDT (Interrupt Descriptor Table)** を通じてハンドラにディスパッチされる。IDTは最大256エントリを持ち、各エントリがハンドラのアドレスとその属性を定義する。

```
x86-64 IDTエントリ構造 (16バイト / Gate Descriptor):

  ビット位置    フィールド
  ┌───────────────────────────────────────┐
  │ 127:96  Reserved (上位DWORD)          │
  │  95:64  Offset[63:32]                 │
  │  63:48  Offset[31:16]                 │
  │  47     P (Present bit)               │
  │  46:45  DPL (特権レベル 0-3)          │
  │  44     0 (固定)                      │
  │  43:40  Gate Type                     │
  │         0xE = Interrupt Gate           │
  │         0xF = Trap Gate                │
  │  39:35  Reserved                      │
  │  34:32  IST (Interrupt Stack Table)   │
  │  31:16  Segment Selector              │
  │  15:0   Offset[15:0]                  │
  └───────────────────────────────────────┘

  IDTRレジスタ:
  ┌──────────────────────────────┐
  │ Base Address (64bit) │ Limit │  ← LIDT命令でロード
  └──────────────────────────────┘
```

**Interrupt Gate と Trap Gate の違い:**
- Interrupt Gate: ハンドラ実行時にIFフラグ（割り込み許可フラグ）を自動的にクリア。以降の割り込みが禁止される
- Trap Gate: IFフラグを変更しない。ハンドラ実行中も割り込みが受け付けられる

この違いにより、ハードウェア割り込みハンドラには通常 Interrupt Gate を使い、ソフトウェア割り込み（システムコール）には Trap Gate を使うのが一般的である。

---

## 2. 割り込み処理の詳細フロー

### 2.1 ハードウェア割り込みの発生から復帰まで

```
 Device          PIC/APIC           CPU                   Memory
   │                │                │                      │
   │─IRQ信号──────→│                │                      │
   │                │─INTR信号────→│                      │
   │                │                │                      │
   │                │                │◆ 現在の命令を完了    │
   │                │                │                      │
   │                │                │◆ RFLAGS, CS, RIP    │
   │                │                │  をスタックにPUSH ──→│
   │                │                │                      │
   │                │←INTA(確認応答)─│                      │
   │                │                │                      │
   │                │─ベクタ番号───→│                      │
   │                │                │                      │
   │                │                │◆ IDT[ベクタ番号]     │
   │                │                │  からハンドラ取得 ←──│
   │                │                │                      │
   │                │                │◆ 特権レベル確認      │
   │                │                │  Ring3→Ring0ならTSS  │
   │                │                │  からRSP0をロード    │
   │                │                │                      │
   │                │                │◆ ハンドラ実行開始    │
   │                │                │  (トップハーフ)       │
   │                │                │                      │
   │                │                │◆ EOI (End of        │
   │                │←EOI送信───────│  Interrupt) 送信     │
   │                │                │                      │
   │                │                │◆ IRET命令で復帰     │
   │                │                │  RIP, CS, RFLAGS    │
   │                │                │  をPOPして元の処理へ │
   │                │                │                      │
```

### 2.2 コード例: x86-64 割り込みハンドラの骨格（C + インラインアセンブリ）

以下は、Linux カーネル風の割り込みハンドラ登録と実装の模式コードである。

```c
/* コード例1: x86-64 割り込みハンドラの基本構造 */

#include <linux/interrupt.h>
#include <linux/module.h>

#define MY_DEVICE_IRQ  11

/* 割り込みコンテキスト情報 */
struct my_device_data {
    unsigned long irq_count;
    spinlock_t    lock;
    void __iomem *base_addr;
    /* デバイス固有のリングバッファ等 */
};

/*
 * トップハーフ: 割り込みハンドラ本体
 * - 割り込み禁止状態で実行される
 * - 最小限の処理のみ行う
 * - スリープ禁止（GFP_ATOMIC のみ使用可）
 */
static irqreturn_t my_device_isr(int irq, void *dev_id)
{
    struct my_device_data *data = dev_id;
    u32 status;

    /* デバイスの割り込みステータスレジスタを読み取り */
    status = ioread32(data->base_addr + STATUS_REG);

    /* 自デバイスの割り込みか確認（共有IRQ対応） */
    if (!(status & MY_DEVICE_IRQ_PENDING))
        return IRQ_NONE;  /* 他デバイスの割り込み */

    /* デバイス側の割り込みをクリア（ACK） */
    iowrite32(status, data->base_addr + STATUS_REG);

    spin_lock(&data->lock);
    data->irq_count++;
    spin_unlock(&data->lock);

    /* ボトムハーフをスケジュール */
    tasklet_schedule(&my_device_tasklet);

    return IRQ_HANDLED;
}

/*
 * ボトムハーフ: 遅延処理
 * - 割り込み有効状態で実行される
 * - 比較的長い処理が可能
 */
static void my_device_tasklet_fn(unsigned long arg)
{
    struct my_device_data *data = (struct my_device_data *)arg;

    /* 受信データの処理、バッファコピーなど */
    process_received_data(data);

    /* ユーザ空間への通知 */
    wake_up_interruptible(&data->wait_queue);
}

DECLARE_TASKLET(my_device_tasklet, my_device_tasklet_fn, 0);

/*
 * デバイス初期化時にIRQを登録
 */
static int my_device_probe(struct pci_dev *pdev,
                           const struct pci_device_id *id)
{
    struct my_device_data *data;
    int ret;

    data = devm_kzalloc(&pdev->dev, sizeof(*data), GFP_KERNEL);
    if (!data)
        return -ENOMEM;

    spin_lock_init(&data->lock);

    /*
     * request_irq() のフラグ:
     *   IRQF_SHARED   - 他デバイスとIRQ共有可能
     *   IRQF_ONESHOT  - threaded IRQ でワンショット
     */
    ret = request_irq(pdev->irq, my_device_isr,
                      IRQF_SHARED, "my_device", data);
    if (ret) {
        dev_err(&pdev->dev, "IRQ %d の登録に失敗: %d\n",
                pdev->irq, ret);
        return ret;
    }

    dev_info(&pdev->dev, "IRQ %d を登録\n", pdev->irq);
    return 0;
}

/*
 * デバイス除去時にIRQを解放
 */
static void my_device_remove(struct pci_dev *pdev)
{
    struct my_device_data *data = pci_get_drvdata(pdev);
    free_irq(pdev->irq, data);
}
```

### 2.3 割り込みコントローラの進化

```
┌─────────────────────────────────────────────────────────────────┐
│               割り込みコントローラの世代変遷                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  第1世代: 8259A PIC (1980s)                                     │
│  ┌─────────────────────────────────────────┐                   │
│  │  Master 8259A ──── Slave 8259A          │                   │
│  │  IRQ0: Timer       IRQ8:  RTC           │                   │
│  │  IRQ1: Keyboard    IRQ9:  Redirect      │                   │
│  │  IRQ2: → Cascade   IRQ10: (空き)        │                   │
│  │  IRQ3: COM2        IRQ11: (空き)        │                   │
│  │  IRQ4: COM1        IRQ12: PS/2 Mouse    │                   │
│  │  IRQ5: LPT2/Sound  IRQ13: FPU          │                   │
│  │  IRQ6: Floppy      IRQ14: Primary IDE   │                   │
│  │  IRQ7: LPT1        IRQ15: Secondary IDE │                   │
│  │                                          │                   │
│  │  制約: 最大15本のIRQ、優先順位固定        │                   │
│  └─────────────────────────────────────────┘                   │
│                          ↓                                      │
│  第2世代: APIC (1990s - 現在)                                   │
│  ┌─────────────────────────────────────────┐                   │
│  │  I/O APIC ←→ System Bus ←→ Local APIC   │                   │
│  │                             (CPU毎に1個) │                   │
│  │                                          │                   │
│  │  改善点:                                  │                   │
│  │  - 224本のIRQベクタ (32-255)             │                   │
│  │  - マルチプロセッサ対応                    │                   │
│  │  - プログラマブル優先度                    │                   │
│  │  - 特定CPUへの割り込み配送               │                   │
│  └─────────────────────────────────────────┘                   │
│                          ↓                                      │
│  第3世代: MSI / MSI-X (2000s - 現在)                            │
│  ┌─────────────────────────────────────────┐                   │
│  │  デバイスがメモリ書き込みで割り込みを通知  │                   │
│  │  - 専用のIRQピン不要                      │                   │
│  │  - MSI: 最大32ベクタ                      │                   │
│  │  - MSI-X: 最大2048ベクタ                  │                   │
│  │  - PCIeデバイスの標準方式                  │                   │
│  │  - NVMe, 高速NIC の必須機能               │                   │
│  └─────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Linux カーネルのトップハーフ/ボトムハーフ設計

### 3.1 設計原則

割り込みハンドラ（トップハーフ）の実行中は、同じIRQラインの割り込みが禁止される。これにより、ハンドラが長時間実行されると、後続の割り込みが失われたり、システム全体の応答性が低下したりする。

この問題を解決するために、Linux カーネルは割り込み処理を2段階に分離する。

| 特性 | トップハーフ | ボトムハーフ |
|------|------------|------------|
| 実行コンテキスト | 割り込みコンテキスト | softirq/tasklet: 割り込みコンテキスト, workqueue: プロセスコンテキスト |
| 割り込み状態 | 同一IRQ禁止 | 割り込み有効 |
| スリープ可否 | 不可 | workqueue のみ可 |
| メモリ確保 | GFP_ATOMIC のみ | workqueue なら GFP_KERNEL 可 |
| 実行タイミング | 即座 | 遅延（ただし softirq は高速） |
| 典型的な処理 | ACK送信、フラグ設定、ボトムハーフのスケジュール | データコピー、プロトコル処理、ユーザ通知 |

### 3.2 ボトムハーフの3方式比較

```
 ┌──────────────────────────────────────────────────────────────┐
 │         ボトムハーフ実行メカニズムの比較                       │
 │                                                              │
 │  ┌──────────┐    ┌──────────┐    ┌──────────────┐           │
 │  │ softirq  │    │ tasklet  │    │  workqueue   │           │
 │  │          │    │          │    │              │           │
 │  │ 静的定義 │    │ 動的生成 │    │ カーネル     │           │
 │  │ (10種類) │    │ 可能     │    │ スレッド上   │           │
 │  │          │    │          │    │ で実行       │           │
 │  │ 複数CPU  │    │ 同一     │    │              │           │
 │  │ 同時実行 │    │ tasklet  │    │ スリープ可   │           │
 │  │ 可能     │    │ は直列化 │    │ mutex使用可  │           │
 │  │          │    │          │    │              │           │
 │  │ 高性能   │    │ 中間     │    │ 柔軟性高     │           │
 │  │ NET_RX等 │    │          │    │ USB等        │           │
 │  └──────────┘    └──────────┘    └──────────────┘           │
 │                                                              │
 │  性能:     softirq > tasklet >> workqueue                    │
 │  柔軟性:   workqueue > tasklet > softirq                     │
 │  実装難度:  softirq > tasklet > workqueue                    │
 └──────────────────────────────────────────────────────────────┘
```

### 3.3 コード例: threaded IRQ の活用

Linux 2.6.30 以降では、`request_threaded_irq()` によってボトムハーフをカーネルスレッドとして実行できる。

```c
/* コード例2: threaded IRQ によるハンドラ分離 */

#include <linux/interrupt.h>

/*
 * ハードIRQハンドラ（トップハーフ）
 * 最小限の処理のみ — デバイスのACKと判定
 */
static irqreturn_t my_hard_irq(int irq, void *dev_id)
{
    struct my_device *dev = dev_id;
    u32 status = ioread32(dev->regs + IRQ_STATUS);

    if (!(status & DEVICE_IRQ_FLAG))
        return IRQ_NONE;

    /* デバイスの割り込みを確認応答 */
    iowrite32(status, dev->regs + IRQ_ACK);

    /* threaded handler の実行を要求 */
    return IRQ_WAKE_THREAD;
}

/*
 * スレッドハンドラ（ボトムハーフ）
 * プロセスコンテキストで実行 — スリープ可能
 */
static irqreturn_t my_thread_fn(int irq, void *dev_id)
{
    struct my_device *dev = dev_id;

    /* 重い処理をここで実行可能 */
    mutex_lock(&dev->data_mutex);

    /* DMA完了データの処理 */
    process_dma_buffer(dev);

    /* I2C/SPI 通信（スリープを伴う可能性あり） */
    update_device_config(dev);

    mutex_unlock(&dev->data_mutex);

    return IRQ_HANDLED;
}

static int my_device_init(struct platform_device *pdev)
{
    struct my_device *dev = platform_get_drvdata(pdev);
    int irq = platform_get_irq(pdev, 0);

    /*
     * request_threaded_irq():
     *   第2引数: ハードIRQハンドラ (NULL可 → デフォルトでIRQ_WAKE_THREAD)
     *   第3引数: スレッドハンドラ
     *   IRQF_ONESHOT: スレッドハンドラ完了までIRQを再有効化しない
     */
    return request_threaded_irq(irq,
                                my_hard_irq,
                                my_thread_fn,
                                IRQF_ONESHOT | IRQF_SHARED,
                                "my_device",
                                dev);
}
```

### 3.4 Linuxの softirq 一覧

Linuxカーネルでは以下の10種類の softirq が静的に定義されている（優先度順）。

| 番号 | 名称 | 用途 |
|------|------|------|
| 0 | HI_SOFTIRQ | 高優先度 tasklet |
| 1 | TIMER_SOFTIRQ | タイマーコールバック |
| 2 | NET_TX_SOFTIRQ | ネットワーク送信 |
| 3 | NET_RX_SOFTIRQ | ネットワーク受信 |
| 4 | BLOCK_SOFTIRQ | ブロックI/O完了 |
| 5 | IRQ_POLL_SOFTIRQ | IRQポーリング |
| 6 | TASKLET_SOFTIRQ | 通常優先度 tasklet |
| 7 | SCHED_SOFTIRQ | スケジューラ負荷分散 |
| 8 | HRTIMER_SOFTIRQ | 高精度タイマー |
| 9 | RCU_SOFTIRQ | RCU処理 |

---

## 4. DMA (Direct Memory Access) の仕組み

### 4.1 DMAの基本原理

DMAは、CPUの介在なしにデバイスとメインメモリ間でデータを直接転送する技術である。CPUはDMAコントローラに転送パラメータ（ソースアドレス、宛先アドレス、転送サイズ）を設定するだけで、実際のデータ転送はDMAコントローラが実行する。

```
DMA転送の完全なシーケンス:

  CPU                 DMA Controller        Device           Memory
   │                      │                   │                │
   │ (1) DMAバッファ確保   │                   │                │
   │────────────────────────────────────────────────────────→│
   │                      │                   │                │
   │ (2) 転送パラメータ設定│                   │                │
   │  src_addr = 0xFE000  │                   │                │
   │  dst_addr = 0x80000  │                   │                │
   │  length   = 4096     │                   │                │
   │  direction = DEV→MEM │                   │                │
   │─────────────────────→│                   │                │
   │                      │                   │                │
   │ (3) DMA開始          │                   │                │
   │─────────────────────→│                   │                │
   │                      │                   │                │
   │ (4) CPU は別タスクへ │                   │                │
   │  ...                 │                   │                │
   │                      │ (5) バスマスタ    │                │
   │                      │  としてバスを     │                │
   │                      │  獲得             │                │
   │                      │                   │                │
   │                      │←─ データ読出 ────│                │
   │                      │                   │                │
   │                      │── メモリ書込 ──────────────────→│
   │                      │                   │                │
   │                      │  (6) 転送サイズ分 │                │
   │                      │  繰り返し         │                │
   │                      │                   │                │
   │                      │ (7) 転送完了      │                │
   │←── 割り込み ─────────│                   │                │
   │                      │                   │                │
   │ (8) 完了処理          │                   │                │
   │  (バッファ解放等)     │                   │                │
   │                      │                   │                │
```

### 4.2 DMAマッピングの種類

Linux カーネルにおけるDMAメモリマッピングには、用途に応じた複数の方式がある。

| 方式 | API | 用途 | 特徴 |
|------|-----|------|------|
| コヒーレントマッピング | `dma_alloc_coherent()` | デバイスとCPUが頻繁にアクセスするバッファ | キャッシュ一貫性が自動維持される。リングバッファ、DMAディスクリプタに適する |
| ストリーミングマッピング | `dma_map_single()` | 一方向の一時的な転送 | CPU側でキャッシュ管理が必要。高性能だがAPI使用に注意が必要 |
| スキャッタ/ギャザー | `dma_map_sg()` | 不連続メモリ領域の転送 | 物理的に不連続なページを一度のDMA操作で転送。ネットワークバッファに最適 |

### 4.3 コード例: Linux DMA APIの使用

```c
/* コード例3: DMAコヒーレントバッファの確保と使用 */

#include <linux/dma-mapping.h>
#include <linux/pci.h>

struct my_dma_device {
    struct pci_dev    *pdev;
    void              *dma_buf_virt;   /* CPU側仮想アドレス */
    dma_addr_t         dma_buf_phys;   /* デバイス側DMAアドレス */
    size_t             buf_size;
};

static int setup_dma_buffer(struct my_dma_device *dev)
{
    /* DMAマスクの設定 — 32bit DMA対応デバイスの場合 */
    if (dma_set_mask_and_coherent(&dev->pdev->dev, DMA_BIT_MASK(32))) {
        dev_err(&dev->pdev->dev, "32bit DMA未サポート\n");
        return -EIO;
    }

    dev->buf_size = PAGE_SIZE * 4;  /* 16KB */

    /*
     * dma_alloc_coherent():
     *   - CPUとデバイスの双方からアクセス可能なバッファを確保
     *   - キャッシュコヒーレンシが自動的に維持される
     *   - 戻り値: CPU側の仮想アドレス
     *   - dma_buf_phys: デバイス側が使用するDMAアドレス
     */
    dev->dma_buf_virt = dma_alloc_coherent(&dev->pdev->dev,
                                            dev->buf_size,
                                            &dev->dma_buf_phys,
                                            GFP_KERNEL);
    if (!dev->dma_buf_virt) {
        dev_err(&dev->pdev->dev, "DMAバッファ確保失敗\n");
        return -ENOMEM;
    }

    dev_info(&dev->pdev->dev,
             "DMAバッファ確保: virt=%p phys=%pad size=%zu\n",
             dev->dma_buf_virt, &dev->dma_buf_phys, dev->buf_size);

    return 0;
}

static void cleanup_dma_buffer(struct my_dma_device *dev)
{
    if (dev->dma_buf_virt) {
        dma_free_coherent(&dev->pdev->dev,
                          dev->buf_size,
                          dev->dma_buf_virt,
                          dev->dma_buf_phys);
        dev->dma_buf_virt = NULL;
    }
}

/*
 * ストリーミングDMAの使用例:
 * 一方向転送（デバイス → メモリ）
 */
static int start_dma_read(struct my_dma_device *dev,
                           void *buffer, size_t len)
{
    dma_addr_t dma_handle;

    /*
     * dma_map_single():
     *   - 既存のカーネルバッファをDMAマッピング
     *   - DMA_FROM_DEVICE: デバイスがメモリに書き込む方向
     */
    dma_handle = dma_map_single(&dev->pdev->dev,
                                 buffer, len,
                                 DMA_FROM_DEVICE);

    if (dma_mapping_error(&dev->pdev->dev, dma_handle)) {
        dev_err(&dev->pdev->dev, "DMAマッピング失敗\n");
        return -EIO;
    }

    /* デバイスにDMAアドレスと長さを設定 */
    iowrite32(lower_32_bits(dma_handle),
              dev->regs + DMA_SRC_ADDR_LO);
    iowrite32(upper_32_bits(dma_handle),
              dev->regs + DMA_SRC_ADDR_HI);
    iowrite32(len, dev->regs + DMA_LENGTH);

    /* DMA転送開始 */
    iowrite32(DMA_START, dev->regs + DMA_CONTROL);

    return 0;
}

/*
 * DMA完了後のクリーンアップ
 * (割り込みハンドラから呼ばれる)
 */
static void finish_dma_read(struct my_dma_device *dev,
                             void *buffer, size_t len,
                             dma_addr_t dma_handle)
{
    /*
     * dma_unmap_single():
     *   - DMAマッピングを解除
     *   - CPUキャッシュの無効化を実行
     *   - この後 buffer の内容をCPUから読めるようになる
     */
    dma_unmap_single(&dev->pdev->dev,
                      dma_handle, len,
                      DMA_FROM_DEVICE);

    /* ここで buffer の内容を処理 */
}
```

### 4.4 スキャッタ/ギャザーDMA

ネットワークパケットやファイルI/Oでは、転送すべきデータが物理的に不連続なメモリページに分散していることが多い。スキャッタ/ギャザーDMAは、この不連続なメモリ領域を1回のDMA操作で転送する技術である。

```
スキャッタ/ギャザーDMAの概念:

  [通常のDMA — 連続メモリ必要]

  物理メモリ:
  ┌──────┬──────┬──────┬──────┬──────┐
  │ Used │ FREE │ Used │ FREE │ Used │
  └──────┴──────┴──────┴──────┴──────┘
                 ↓
  コピーして連続領域を作る必要あり（CPU負荷）

  [スキャッタ/ギャザーDMA — 不連続OK]

  SG List (Scatter/Gather List):
  ┌──────────────────────┐
  │ Entry 0:             │     物理メモリ
  │  addr=0x1000 len=512 │──→ ┌──────┐
  │ Entry 1:             │     │ Data │ page A
  │  addr=0x5000 len=256 │──→ ├──────┤
  │ Entry 2:             │     │      │
  │  addr=0x9000 len=768 │──→ │ Data │ page C
  └──────────────────────┘     ├──────┤
                               │      │
    DMAコントローラがSG Listを  │ Data │ page E
    順に処理、CPU介在不要       └──────┘
```

この技術は以下の場面で特に有効である。

- **ネットワーク**: パケットヘッダとペイロードが別バッファにある場合（ゼロコピー送信）
- **ストレージ**: ファイルシステムの複数ブロックを一度に読み書きする場合
- **仮想化**: ゲストOSの物理メモリがホスト上で不連続な場合

---

## 5. IOMMU — DMAのアドレス仮想化と保護

### 5.1 IOMMUの必要性

DMAはCPUを介さずにメモリにアクセスできるが、これはセキュリティ上の大きなリスクを伴う。悪意のあるデバイス（または不具合のあるドライバ）が任意の物理メモリアドレスにDMA転送を行えば、カーネルメモリの破壊やデータ漏洩が発生しうる。

**IOMMU (Input/Output Memory Management Unit)** は、デバイスのDMAアドレスを仮想化し、許可されたメモリ領域のみにアクセスを制限する。CPUにとってのMMUと同様の役割を、I/Oデバイスに対して果たす。

```
IOMMUの位置づけ:

  CPU側:                              デバイス側:
  ┌─────┐    ┌─────┐                 ┌────────┐    ┌───────┐
  │ CPU │───→│ MMU │───→物理メモリ←──│ IOMMU  │←───│Device │
  └─────┘    └─────┘                 └────────┘    └───────┘
              │                        │
              ▼                        ▼
         ページテーブル            I/Oページテーブル
         (仮想→物理)              (DMAアドレス→物理)

  MMU : CPUの仮想アドレス → 物理アドレスの変換
  IOMMU: デバイスのDMAアドレス → 物理アドレスの変換

  ┌──────────────────────────────────────────────┐
  │  デバイスが DMA addr 0x2000 にアクセス        │
  │         ↓                                    │
  │  IOMMU がI/Oページテーブルを参照              │
  │         ↓                                    │
  │  [許可]  → 物理addr 0xA8000 に変換して転送    │
  │  [拒否]  → DMA Fault 発生 → カーネルに通知    │
  └──────────────────────────────────────────────┘
```

### 5.2 IOMMUの主要な用途

| 用途 | 説明 |
|------|------|
| DMA保護 | デバイスがアクセス可能な物理メモリ領域を制限する |
| デバイスの仮想化パススルー | VT-d/AMD-Vi により、仮想マシンにデバイスを直接割り当て（PCIパススルー）する際に、ゲストの物理アドレスをホストの物理アドレスに変換する |
| DMAリマッピング | 32bit DMA制限のあるデバイスでも、4GB超のメモリにアクセス可能にする（バウンスバッファの回避） |
| 割り込みリマッピング | デバイスからの割り込みを検証し、不正な割り込みインジェクションを防止する |

### 5.3 Linuxにおける IOMMU の設定

```bash
# コード例4: IOMMU関連のカーネルパラメータと確認コマンド

# カーネルブートパラメータ（GRUB設定）
# Intel VT-d の有効化
GRUB_CMDLINE_LINUX="intel_iommu=on"

# AMD-Vi の有効化
GRUB_CMDLINE_LINUX="amd_iommu=on"

# IOMMUグループの確認
# 各デバイスがどのIOMMUグループに属するかを表示
for d in /sys/kernel/iommu_groups/*/devices/*; do
    n=$(echo "$d" | rev | cut -d/ -f1 | rev)
    g=$(echo "$d" | rev | cut -d/ -f3 | rev)
    echo "IOMMU Group $g: $(lspci -nns "$n")"
done

# dmesg での IOMMU 初期化確認
dmesg | grep -i iommu
# 出力例:
# [    0.123456] DMAR: IOMMU enabled
# [    0.234567] DMAR: Intel(R) Virtualization Technology for Directed I/O

# /proc/interrupts で割り込み分布を確認
cat /proc/interrupts | head -20

# 特定IRQのアフィニティ確認
cat /proc/irq/24/smp_affinity
# 出力例: f  (CPU 0-3 に配送)

# IRQアフィニティの設定（CPU 2 のみに配送）
echo 4 > /proc/irq/24/smp_affinity
# ビットマスク: 4 = 0100 → CPU 2
```

---

## 6. 現代のI/O技術

### 6.1 MSI / MSI-X (Message Signaled Interrupts)

従来の割り込み方式では、デバイスは専用のIRQピン（物理配線）を使ってCPUに割り込みを通知していた。MSI/MSI-Xでは、デバイスが特定のメモリアドレスにデータを書き込むことで割り込みを通知する。

```
従来方式 vs MSI/MSI-X:

  [従来 (ピンベース割り込み)]
  ┌────────┐  IRQピン  ┌────────┐  INTR  ┌─────┐
  │ Device ├──────────→│ I/O    ├───────→│ CPU │
  │   A    │           │ APIC   │        │     │
  └────────┘           │        │        └─────┘
  ┌────────┐  IRQピン  │        │
  │ Device ├──────────→│        │  問題:
  │   B    │           └────────┘  - ピン数に限り (24本)
  └────────┘                       - 共有IRQで性能低下
                                   - 動的な配送先変更が困難

  [MSI-X]
  ┌────────┐                          ┌─────────┐
  │ Device │── メモリ書き込み ────────→│ Local   │
  │   A    │  (addr=0xFEE00xxx,       │ APIC    │
  │        │   data=vector_num)       │ (CPU 0) │
  │  2048  │                          └─────────┘
  │  ベクタ│── メモリ書き込み ────────→┌─────────┐
  │  まで  │  (別アドレス/データ)      │ Local   │
  └────────┘                          │ APIC    │
                                      │ (CPU 3) │
  利点:                               └─────────┘
  - デバイス毎に最大2048ベクタ
  - 各ベクタを異なるCPUに配送可能
  - IRQ共有不要 → ハンドラが高速
  - NVMe: キュー毎にMSI-Xベクタを割当
```

### 6.2 MSI-X の性能上の利点

| 特性 | ピンベース割り込み | MSI | MSI-X |
|------|-------------------|-----|-------|
| ベクタ数 | 1 (共有) | 最大32 | 最大2048 |
| IRQ共有 | 必要 | 不要 | 不要 |
| CPUターゲティング | 制限あり | 限定的 | ベクタ毎に自由 |
| レイテンシ | 高い | 低い | 低い |
| マルチキュー対応 | 不可 | 制限あり | 完全対応 |
| PCIe互換性 | レガシー | 標準 | 推奨 |

### 6.3 NVMe (Non-Volatile Memory Express)

NVMeは、SSD (NAND Flash / 3D XPoint) の性能を最大限に引き出すために設計された、PCIeネイティブのストレージプロトコルである。旧来のAHCI (Advanced Host Controller Interface) がHDDの回転待ちを前提に設計されていたのに対し、NVMeは大量の並列I/Oを効率的に処理する。

```
AHCI vs NVMe アーキテクチャ比較:

  [AHCI (SATAベース)]
  ┌──────┐     1本のキュー       ┌──────────┐
  │ CPU  │────(最大32コマンド)──→│ SATA SSD │
  │      │     深度: 32          │          │
  └──────┘                       └──────────┘
  ↑ 全コマンドが1キューに直列化 = ボトルネック

  [NVMe (PCIeベース)]
  ┌──────┐  Submission Q 0 ────→┌──────────┐
  │ CPU  │  Submission Q 1 ────→│ NVMe SSD │
  │ Core │  Submission Q 2 ────→│          │
  │  0   │                      │ 内部     │
  │      │  Completion Q 0 ←───│ コントロ │
  └──────┘  Completion Q 1 ←───│ ーラ     │
  ┌──────┐  Submission Q 3 ────→│          │
  │ CPU  │  Submission Q 4 ────→│ Flash    │
  │ Core │                      │ チャネル │
  │  1   │  Completion Q 2 ←───│ ×8-16   │
  └──────┘                      └──────────┘

  最大 65,535 キュー × 65,536 エントリ/キュー
  各CPUコアに専用のSubmission/Completionキューペアを割当
  MSI-Xベクタもキュー毎に割当 → ロックフリー設計
```

| 比較項目 | AHCI (SATA) | NVMe |
|----------|-------------|------|
| キュー数 | 1 | 最大 65,535 |
| キュー深度 | 32 | 最大 65,536 |
| ホストインタフェース | SATA (6 Gbps) | PCIe Gen4 x4 (64 Gbps) |
| 割り込み方式 | ピンベース / MSI | MSI-X (キュー毎) |
| コマンド発行 | レジスタ書込×4回 | Doorbell レジスタ×1回 |
| CPU使用率 | 高い | 低い |
| IOPS (4K Random Read) | 約100,000 | 約1,000,000+ |

### 6.4 RDMA (Remote Direct Memory Access)

RDMAは、ネットワーク越しに相手マシンのメモリに直接アクセスする技術である。通常のネットワーク通信では、データはアプリケーション→カーネル→NICドライバ→NIC→ネットワーク→NIC→NICドライバ→カーネル→アプリケーションという多段階のコピーとコンテキストスイッチを経る。RDMAはこれを「ゼロコピー」「OSバイパス」で実現する。

```
通常のネットワーク通信 vs RDMA:

  [通常のTCP/IP通信]
  App → [copy] → Kernel TCP/IP Stack → [copy] → NIC Driver → NIC
                     ↓ 割り込み / コンテキストスイッチ多数
  NIC → NIC Driver → [copy] → Kernel TCP/IP Stack → [copy] → App

  合計4回のメモリコピー + 複数回のコンテキストスイッチ

  [RDMA]
  App ──────→ RNIC (RDMA NIC) ──────→ Network
                ↓ Hardware offload         ↓
  Network ──→ RNIC ──────→ App のメモリに直接書き込み

  ゼロコピー、OSバイパス、CPU関与なし

  RDMA動詞 (Verbs):
  ┌─────────────────────────────────────────────┐
  │ RDMA Read  : リモートメモリの読み取り        │
  │ RDMA Write : リモートメモリへの書き込み      │
  │ Send/Recv  : メッセージパッシング            │
  │ Atomic     : リモートでのCAS/Fetch-Add       │
  └─────────────────────────────────────────────┘
```

RDMAの主要なトランスポート技術は以下の3つである。

| 技術 | 物理層 | 帯域幅 | レイテンシ | 用途 |
|------|--------|--------|-----------|------|
| InfiniBand | 専用ファブリック | 200-400 Gbps (HDR/NDR) | < 1 us | HPC、AIクラスタ |
| RoCEv2 | Ethernet | 25-400 Gbps | 1-2 us | データセンター |
| iWARP | Ethernet + TCP | 10-100 Gbps | 5-10 us | 汎用 |

---

## 7. 割り込みアフィニティとパフォーマンスチューニング

### 7.1 IRQバランシングの課題

マルチコアシステムでは、割り込みがどのCPUで処理されるかが性能に大きく影響する。デフォルトでは、Linux の `irqbalance` デーモンが割り込みをCPU間に分散するが、高性能を要求するワークロードでは手動チューニングが必要になる場合がある。

```
割り込みアフィニティ設定の考え方:

  ┌─────────────────────────────────────────────────────┐
  │           NUMA Node 0          NUMA Node 1          │
  │  ┌──────┐ ┌──────┐     ┌──────┐ ┌──────┐          │
  │  │CPU 0 │ │CPU 1 │     │CPU 2 │ │CPU 3 │          │
  │  └──┬───┘ └──┬───┘     └──┬───┘ └──┬───┘          │
  │     │        │            │        │               │
  │  ┌──┴────────┴──┐     ┌──┴────────┴──┐            │
  │  │  L3 Cache    │     │  L3 Cache    │            │
  │  │  Memory      │     │  Memory      │            │
  │  │  Controller  │     │  Controller  │            │
  │  └──────┬───────┘     └──────┬───────┘            │
  │         │                    │                     │
  │  ┌──────┴───────┐     ┌─────┴────────┐            │
  │  │ PCIe Root    │     │ PCIe Root    │            │
  │  │ Complex      │     │ Complex      │            │
  │  └──────┬───────┘     └──────┬───────┘            │
  │         │                    │                     │
  │      ┌──┴──┐             ┌───┴──┐                 │
  │      │ NIC │             │ NVMe │                 │
  │      │eth0 │             │nvme0 │                 │
  │      └─────┘             └──────┘                 │
  └─────────────────────────────────────────────────────┘

  ベストプラクティス:
  - NICの割り込み → NICと同じNUMAノードのCPUに配送
  - NVMeの割り込み → NVMeと同じNUMAノードのCPUに配送
  - NUMAをまたぐメモリアクセスはレイテンシが増大する
```

### 7.2 コード例: 割り込みアフィニティの設定スクリプト

```bash
# コード例5: NICの割り込みアフィニティを手動設定するスクリプト

#!/bin/bash
# nic_irq_affinity.sh — NIC割り込みをNUMAローカルCPUに固定

DEVICE="eth0"
NUMA_NODE=$(cat /sys/class/net/${DEVICE}/device/numa_node)

echo "=== ${DEVICE} の割り込みアフィニティ設定 ==="
echo "NUMA Node: ${NUMA_NODE}"

# NUMAノードに属するCPUリストを取得
CPULIST=$(cat /sys/devices/system/node/node${NUMA_NODE}/cpulist)
echo "Local CPUs: ${CPULIST}"

# irqbalance を停止（手動設定と競合するため）
systemctl stop irqbalance 2>/dev/null

# デバイスのIRQ番号一覧を取得
IRQS=$(grep "${DEVICE}" /proc/interrupts | awk '{print $1}' | tr -d ':')

CPU_IDX=0
CPUS=($(echo "${CPULIST}" | tr ',' ' ' | tr '-' ' '))

for IRQ in ${IRQS}; do
    # CPUをラウンドロビンで割当
    TARGET_CPU=${CPUS[$((CPU_IDX % ${#CPUS[@]}))]}

    # アフィニティマスクを計算
    MASK=$(printf "%x" $((1 << TARGET_CPU)))

    echo "  IRQ ${IRQ} → CPU ${TARGET_CPU} (mask: ${MASK})"
    echo "${MASK}" > /proc/irq/${IRQ}/smp_affinity

    CPU_IDX=$((CPU_IDX + 1))
done

echo "=== 設定完了 ==="

# 確認
echo ""
echo "現在の割り込み分布:"
grep "${DEVICE}" /proc/interrupts
```

### 7.3 NAPI — ネットワーク割り込みの最適化

高速ネットワーク（10GbE以上）では、パケット毎に割り込みが発生すると、割り込みのオーバーヘッド自体がCPUを圧迫する（**割り込みストーム**）。Linux NAPI (New API) はこの問題を、割り込みとポーリングのハイブリッド方式で解決する。

```
NAPI の動作フロー:

  パケット到着レート: 低い                  高い
  ←─────────────────────────────────────────→

  [割り込みモード]          [ポーリングモード]
  パケット毎に             割り込みを無効化し
  割り込み発生              CPUがNICを定期的にポーリング

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  (1) パケット到着 → NICが割り込み発生             │
  │         ↓                                        │
  │  (2) 割り込みハンドラ: napi_schedule()            │
  │      → 以降の割り込みを無効化                     │
  │         ↓                                        │
  │  (3) softirq (NET_RX_SOFTIRQ) が起動             │
  │         ↓                                        │
  │  (4) NAPI poll関数がNICのリングバッファを         │
  │      繰り返しチェック (ポーリング)                 │
  │         ↓                                        │
  │  (5) バジェット(通常64パケット)分を処理           │
  │         ↓                                        │
  │  (6a) まだパケットあり → (4)に戻る               │
  │  (6b) パケット枯渇 → napi_complete_done()        │
  │       → 割り込みを再有効化 → (1)に戻る           │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

この設計により、低負荷時は割り込みの低レイテンシを活かし、高負荷時はポーリングでスループットを最大化する自動的な切り替えが実現される。

---

## 8. アンチパターンと対策

### 8.1 アンチパターン1: 割り込みハンドラ内での長時間処理

```
[問題のあるコード]

static irqreturn_t bad_isr(int irq, void *dev_id)
{
    /* 危険: 割り込みハンドラ内でmutexを取得 */
    mutex_lock(&data->big_lock);         /* スリープ可能 → BUG */

    /* 危険: 大量データのコピー処理 */
    memcpy(user_buf, dma_buf, 1048576);  /* 1MB コピー → 長時間 */

    /* 危険: カーネルメモリの通常確保 */
    buf = kmalloc(65536, GFP_KERNEL);    /* スリープ可能 → BUG */

    mutex_unlock(&data->big_lock);
    return IRQ_HANDLED;
}
```

**問題点:**
- 割り込みコンテキストではスリープ不可。`mutex_lock()` と `GFP_KERNEL` はスリープする可能性がある
- 長時間の処理は他の割り込みをブロックし、システム全体の応答性を劣化させる
- 最悪の場合、ウォッチドッグタイマーが発火してシステムがリセットされる

**対策:**
- トップハーフでは最小限の処理（ACK、フラグ設定）のみ行う
- 重い処理はボトムハーフ（tasklet、workqueue、threaded IRQ）に委譲する
- 割り込みコンテキストでは `spin_lock()` と `GFP_ATOMIC` のみ使用する

```
[修正後のコード]

static irqreturn_t good_isr(int irq, void *dev_id)
{
    struct my_device *dev = dev_id;
    u32 status;

    status = ioread32(dev->regs + IRQ_STATUS);
    if (!(status & MY_IRQ_FLAG))
        return IRQ_NONE;

    /* 最小限の処理: ACKとフラグ設定のみ */
    iowrite32(status, dev->regs + IRQ_ACK);

    spin_lock(&dev->lock);
    dev->pending_status |= status;
    spin_unlock(&dev->lock);

    /* 重い処理はボトムハーフへ */
    return IRQ_WAKE_THREAD;
}

static irqreturn_t good_thread_fn(int irq, void *dev_id)
{
    struct my_device *dev = dev_id;

    /* プロセスコンテキスト: mutex、GFP_KERNEL、スリープ全て可能 */
    mutex_lock(&dev->big_lock);
    memcpy(user_buf, dma_buf, 1048576);
    buf = kmalloc(65536, GFP_KERNEL);
    process_data(dev);
    mutex_unlock(&dev->big_lock);

    return IRQ_HANDLED;
}
```

### 8.2 アンチパターン2: DMAバッファのキャッシュ一貫性違反

```
[問題のあるコード]

/* ストリーミングDMAで受信したデータを読むケース */
dma_handle = dma_map_single(dev, buffer, len, DMA_FROM_DEVICE);

/* デバイスにDMA転送を開始させる */
start_device_dma(dev, dma_handle, len);

/* 転送完了を待つ */
wait_for_completion(&dev->dma_done);

/* 危険: unmap前にCPUがバッファにアクセス */
process_data(buffer);    /* キャッシュに古いデータが残っている可能性 */

/* 後から unmap — 手遅れ */
dma_unmap_single(dev, dma_handle, len, DMA_FROM_DEVICE);
```

**問題点:**
- `dma_unmap_single()` はCPUキャッシュの無効化（invalidate）を行う
- unmap前にCPUがバッファを読むと、キャッシュに残っている古いデータを読む可能性がある
- この種のバグは、特定のアーキテクチャ（ARMなど、キャッシュコヒーレンシが弱い環境）でのみ再現し、x86では発見しにくい

**対策:**
- 必ず `dma_unmap_single()` の後にバッファにアクセスする
- 繰り返しDMAを行う場合は `dma_sync_single_for_cpu()` / `dma_sync_single_for_device()` を使用する

```
[修正後のコード]

dma_handle = dma_map_single(dev, buffer, len, DMA_FROM_DEVICE);
start_device_dma(dev, dma_handle, len);
wait_for_completion(&dev->dma_done);

/* 正しい順序: まず unmap してからアクセス */
dma_unmap_single(dev, dma_handle, len, DMA_FROM_DEVICE);
process_data(buffer);    /* キャッシュは無効化済み → 正しいデータ */

/* もしくは、マッピングを維持したまま同期する場合 */
dma_sync_single_for_cpu(dev, dma_handle, len, DMA_FROM_DEVICE);
process_data(buffer);    /* 同期済み → 正しいデータ */

/* 再びデバイスにDMAさせる前に */
dma_sync_single_for_device(dev, dma_handle, len, DMA_FROM_DEVICE);
start_device_dma(dev, dma_handle, len);
```

### 8.3 アンチパターン3: NUMA非対応のIRQ配置

```
問題のある構成:

  NUMA Node 0                   NUMA Node 1
  ┌──────────┐                 ┌──────────┐
  │ CPU 0-7  │                 │ CPU 8-15 │
  │          │                 │          │
  │ Memory   │   QPI/UPI      │ Memory   │
  │ (ローカル)│←─────────────→│ (ローカル)│
  └────┬─────┘                 └────┬─────┘
       │                            │
  ┌────┴─────┐                      │
  │ 10GbE   │                      │
  │ NIC     │                      │
  └──────────┘                      │

  問題: NICの割り込みをCPU 8-15（Node 1）で処理
  → パケットデータはNode 0のメモリに到着
  → CPU 8-15がNode 0のメモリにアクセス
  → NUMAリモートアクセスでレイテンシ50-100%増加
```

**対策:**
- デバイスと同じNUMAノードのCPUにIRQアフィニティを設定する
- アプリケーションも同じNUMAノードで実行する（`numactl --cpunodebind=0 --membind=0`）
- `irqbalance` の `--banirq` オプションで特定IRQの自動移動を禁止する

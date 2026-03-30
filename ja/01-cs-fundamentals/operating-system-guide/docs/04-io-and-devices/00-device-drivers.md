# デバイスドライバ

> デバイスドライバは「ハードウェアの方言をOSの共通語に翻訳する」通訳者である。
> カーネルとハードウェアの間に立ち、統一されたインターフェースを提供することで、
> アプリケーション開発者がデバイス固有の複雑さから解放される仕組みを実現する。

---

## この章で学ぶこと

- [ ] デバイスドライバの役割と設計思想を理解する
- [ ] キャラクタデバイス・ブロックデバイス・ネットワークデバイスの違いを知る
- [ ] Linuxカーネルモジュールの仕組みとライフサイクルを理解する
- [ ] I/O制御方式（ポーリング、割り込み、DMA）を比較できる
- [ ] デバイスドライバの内部構造（file_operations、probe/remove）を読み解ける
- [ ] ユーザ空間ドライバ（UIO / VFIO）のメリット・デメリットを説明できる
- [ ] ドライバ開発における典型的なアンチパターンを回避できる
- [ ] デバイスツリーとACPIによるハードウェア記述を理解する


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解

---

## 1. デバイスドライバの基本概念

### 1.1 デバイスドライバとは何か

デバイスドライバとは、オペレーティングシステムのカーネル内（またはカーネルと連携する空間）で動作し、特定のハードウェアデバイスを制御するソフトウェアモジュールである。OSが提供する抽象的なインターフェース（ファイル操作、ネットワークスタック等）と、物理デバイスの具体的なレジスタ操作・プロトコル処理との間を仲介する。

デバイスドライバが必要とされる根本的理由は以下の3点に集約される。

1. **ハードウェア多様性の吸収**: 同じ機能（例: ストレージ）でも、ベンダーやモデルごとにレジスタレイアウト、コマンドセット、タイミング仕様が異なる。ドライバがこの差異を吸収し、統一APIを上位層に提供する。

2. **特権操作の管理**: ハードウェアレジスタへの直接アクセスやDMA設定にはカーネル特権（Ring 0）が必要である。ドライバがこれらの特権操作を安全にカプセル化する。

3. **リソース共有の調整**: 複数のプロセスが同一デバイスへアクセスする場合の排他制御、バッファ管理、優先度制御をドライバが担う。

```
┌─────────────────────────────────────────────────────────┐
│                   ユーザ空間                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ アプリA   │  │ アプリB   │  │ アプリC   │               │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
│       │ open/read    │ write       │ ioctl               │
├───────┼──────────────┼─────────────┼─────────────────────┤
│       ▼              ▼             ▼     カーネル空間      │
│  ┌─────────────────────────────────────────────┐         │
│  │        システムコールインターフェース           │         │
│  └──────────────────┬──────────────────────────┘         │
│                     ▼                                     │
│  ┌─────────────────────────────────────────────┐         │
│  │     VFS（仮想ファイルシステム）                 │         │
│  └──────┬───────────┬──────────────┬───────────┘         │
│         ▼           ▼              ▼                      │
│  ┌──────────┐ ┌──────────┐  ┌──────────────┐            │
│  │charドライバ│ │blkドライバ│  │ネットドライバ │            │
│  │(tty,input)│ │(SCSI,NVMe)│ │(e1000,iwlwifi)│           │
│  └────┬─────┘ └────┬─────┘  └──────┬───────┘            │
├───────┼─────────────┼───────────────┼────────────────────┤
│       ▼             ▼               ▼   ハードウェア       │
│  ┌──────────┐ ┌──────────┐  ┌──────────────┐            │
│  │キーボード │ │SSD/HDD   │  │NIC           │            │
│  │マウス     │ │NVMe      │  │Wi-Fiアダプタ  │            │
│  └──────────┘ └──────────┘  └──────────────┘            │
└─────────────────────────────────────────────────────────┘
```

### 1.2 デバイスの分類体系

Linuxカーネルは、デバイスを大きく3つのカテゴリに分類する。この分類はUNIXの伝統を継承しつつ、現代のハードウェア事情に合わせて拡張されている。

#### キャラクタデバイス（Character Device）

バイトストリームとして逐次アクセスされるデバイスである。データの単位はバイトであり、基本的にシーケンシャルな読み書きを行う。`/dev` ディレクトリ下にデバイスファイルとして現れ、メジャー番号・マイナー番号の組で識別される。

代表例:
- `/dev/tty*` — 端末デバイス
- `/dev/input/event*` — 入力デバイス（キーボード、マウス）
- `/dev/random`, `/dev/urandom` — 乱数生成器
- `/dev/null`, `/dev/zero` — 特殊デバイス
- `/dev/video0` — ビデオキャプチャ（V4L2）
- `/dev/snd/*` — サウンドデバイス（ALSA）

#### ブロックデバイス（Block Device）

固定サイズのブロック単位（通常512バイトまたは4096バイト）でランダムアクセスが可能なデバイスである。カーネルのブロック層（Block Layer）を介してアクセスされ、I/Oスケジューラによるリクエストの最適化やバッファキャッシュの恩恵を受ける。

代表例:
- `/dev/sda`, `/dev/sdb` — SCSI/SATAディスク
- `/dev/nvme0n1` — NVMeストレージ
- `/dev/mmcblk0` — eMMC/SDカード
- `/dev/loop0` — ループバックデバイス
- `/dev/dm-0` — Device Mapper（LVM、暗号化）

#### ネットワークデバイス（Network Device）

パケット単位でデータを送受信するデバイスである。他の2つと異なり、`/dev`配下のファイルとしては表現されず、ソケットインターフェースを通じてアクセスする。`ip`コマンドや`ifconfig`コマンドで管理される。

代表例:
- `eth0`, `ens33` — 有線イーサネット
- `wlan0`, `wlp2s0` — 無線LAN
- `lo` — ループバックインターフェース
- `docker0`, `br0` — ブリッジインターフェース

### 1.3 デバイス分類の比較表

| 特性 | キャラクタデバイス | ブロックデバイス | ネットワークデバイス |
|:-----|:-------------------|:-----------------|:--------------------|
| アクセス単位 | バイト（ストリーム） | ブロック（512B/4KB） | パケット（可変長） |
| アクセスパターン | 主にシーケンシャル | ランダムアクセス可能 | パケット送受信 |
| `/dev`エントリ | あり（`c`で表示） | あり（`b`で表示） | なし |
| バッファキャッシュ | なし（通常） | あり（ページキャッシュ） | SKB（Socket Buffer） |
| I/Oスケジューラ | なし | あり（mq-deadline等） | なし（Qdisc） |
| 登録API | `register_chrdev_region()` | `register_blkdev()` | `register_netdev()` |
| 主要ops構造体 | `file_operations` | `block_device_operations` | `net_device_ops` |
| 代表的デバイス | tty, input, GPU | HDD, SSD, NVMe | Ethernet, Wi-Fi |
| seek操作 | 不可（多くの場合） | 可能 | 概念なし |

---

## 2. Linuxカーネルモジュールの仕組み

### 2.1 カーネルモジュールとは

Linuxカーネルモジュール（Loadable Kernel Module: LKM）は、カーネルの再コンパイルや再起動なしに動的にロード・アンロードできるコードの単位である。多くのデバイスドライバはカーネルモジュールとして実装されており、必要な時にのみメモリにロードされる。

この仕組みにより以下のメリットが得られる。

- カーネルイメージのサイズを最小限に保てる
- デバイス接続時に自動的に適切なドライバをロードできる（udev連携）
- ドライバの開発・テストサイクルが短縮される（再起動不要）
- 使用していないドライバをアンロードしてメモリを解放できる

### 2.2 最小限のカーネルモジュール実装

以下は最も基本的なLinuxカーネルモジュールの完全なソースコードである。カーネルモジュール開発の出発点として、初期化関数と終了関数のペアが必須であることを示している。

```c
/* hello_driver.c — 最小限のカーネルモジュール */
#include <linux/module.h>    /* MODULE_LICENSE, module_init/exit */
#include <linux/kernel.h>    /* printk, KERN_INFO */
#include <linux/init.h>      /* __init, __exit マクロ */

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Example Author");
MODULE_DESCRIPTION("Minimal kernel module example");
MODULE_VERSION("1.0");

/*
 * モジュールロード時に呼ばれる初期化関数
 * __init マクロ: 初期化完了後にこの関数のメモリを解放してよいことを示す
 * 戻り値: 0 = 成功、負の値 = エラー（errno値の符号反転）
 */
static int __init hello_init(void)
{
    pr_info("hello_driver: module loaded\n");
    pr_info("hello_driver: kernel version = %s\n", UTS_RELEASE);

    /*
     * ここでデバイスの初期化処理を行う:
     *   - デバイス番号の割り当て (alloc_chrdev_region)
     *   - cdev構造体の初期化と登録 (cdev_init, cdev_add)
     *   - デバイスクラスの作成 (class_create)
     *   - デバイスノードの作成 (device_create)
     *   - ハードウェアの初期化 (レジスタ設定等)
     */

    return 0;  /* 成功 */
}

/*
 * モジュールアンロード時に呼ばれる終了関数
 * __exit マクロ: モジュールが組み込みの場合、この関数を省略できることを示す
 */
static void __exit hello_exit(void)
{
    pr_info("hello_driver: module unloaded\n");

    /*
     * ここで後始末処理を行う（init の逆順）:
     *   - device_destroy
     *   - class_destroy
     *   - cdev_del
     *   - unregister_chrdev_region
     */
}

/* カーネルにエントリポイントを登録 */
module_init(hello_init);
module_exit(hello_exit);
```

ビルド用の`Makefile`は以下のようになる。

```makefile
# Makefile for hello_driver kernel module
obj-m += hello_driver.o

# KDIR: カーネルソースツリーのパス（通常はヘッダのみで十分）
KDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

# モジュールのロード/アンロード（テスト用）
load:
	sudo insmod hello_driver.ko

unload:
	sudo rmmod hello_driver

# カーネルログの確認
log:
	dmesg | tail -20
```

### 2.3 モジュール管理コマンド体系

```
┌──────────────────────────────────────────────────────┐
│             モジュール管理コマンドの関係図              │
│                                                      │
│   insmod hello.ko          modprobe hello            │
│      │ (単体ロード)            │ (依存解決+ロード)     │
│      ▼                        ▼                      │
│   ┌────────────────────────────────┐                 │
│   │     カーネルモジュールローダー    │                 │
│   │  ┌───────────────────────┐     │                 │
│   │  │ モジュール検証          │     │                 │
│   │  │ (署名チェック,          │     │                 │
│   │  │  バージョン確認)        │     │                 │
│   │  └───────┬───────────────┘     │                 │
│   │          ▼                      │                 │
│   │  ┌───────────────────────┐     │                 │
│   │  │ シンボル解決            │     │                 │
│   │  │ (依存モジュールの       │     │                 │
│   │  │  エクスポートシンボル)   │     │                 │
│   │  └───────┬───────────────┘     │                 │
│   │          ▼                      │                 │
│   │  ┌───────────────────────┐     │                 │
│   │  │ init関数呼び出し        │     │                 │
│   │  └───────────────────────┘     │                 │
│   └────────────────────────────────┘                 │
│                                                      │
│   rmmod hello              modprobe -r hello         │
│      │ (単体アンロード)        │ (依存込みアンロード)   │
│      ▼                        ▼                      │
│   exit関数呼び出し → メモリ解放                         │
│                                                      │
│   lsmod           → /proc/modules の整形表示          │
│   modinfo hello   → モジュールのメタ情報表示           │
│   depmod -a       → modules.dep 依存関係DB再構築      │
└──────────────────────────────────────────────────────┘
```

主要コマンドの使用例:

```bash
# ロード済みモジュールの一覧と参照カウント
$ lsmod | head -10
Module                  Size  Used by
nvidia_drm             77824  10
nvidia_modeset       1236992  18 nvidia_drm
nvidia              56467456  656 nvidia_modeset
snd_hda_intel          57344  4

# モジュールの詳細情報を確認
$ modinfo e1000e
filename:       /lib/modules/6.1.0/kernel/drivers/net/ethernet/intel/e1000e/e1000e.ko
version:        3.2.6-k
license:        GPL v2
description:    Intel(R) PRO/1000 Network Driver
author:         Intel Corporation
depends:
retpoline:      Y

# 依存関係を自動解決してロード
$ sudo modprobe snd_hda_intel

# パラメータを指定してロード
$ sudo modprobe usbcore autosuspend=-1

# モジュールパラメータの確認
$ cat /sys/module/usbcore/parameters/autosuspend

# 依存モジュールごとアンロード
$ sudo modprobe -r snd_hda_intel
```

### 2.4 udevとホットプラグ

デバイスが接続されると、カーネルがueventを発行し、ユーザ空間のデーモン`udevd`がルールに従ってデバイスノードの作成やドライバのロードを自動的に行う。

```
デバイス接続からドライバロードまでの流れ:

  USBデバイス挿入
       │
       ▼
  ┌──────────────────┐
  │ USBコアドライバ    │  カーネル内でデバイス検出
  │ (usb-core)       │
  └────────┬─────────┘
           │ kobject_uevent() でカーネルイベント発行
           ▼
  ┌──────────────────┐
  │ netlink socket   │  カーネル → ユーザ空間通知
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ udevd            │  ueventを受信
  │ (systemd-udevd)  │
  └────────┬─────────┘
           │ /etc/udev/rules.d/ のルールを評価
           ▼
  ┌──────────────────────────────────────┐
  │ ルール評価結果に基づく処理:             │
  │  1. /dev/xxx デバイスノード作成         │
  │  2. パーミッション・所有者設定          │
  │  3. シンボリックリンク作成              │
  │  4. modprobe でドライバロード           │
  │  5. RUN= で外部スクリプト実行           │
  └──────────────────────────────────────┘
```

udevルールの実例:

```bash
# /etc/udev/rules.d/99-usb-serial.rules
# USB-シリアル変換器に固定デバイス名を割り当てるルール

# ベンダーID=0403（FTDI）, プロダクトID=6001 のデバイスに
# /dev/ttyFTDI というシンボリックリンクを作成し、
# グループ dialout に所属させ、パーミッション 0666 を設定
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", \
    SYMLINK+="ttyFTDI", GROUP="dialout", MODE="0666"

# 特定のシリアル番号を持つデバイスに一意の名前を割り当て
SUBSYSTEM=="tty", ATTRS{serial}=="A50285BI", SYMLINK+="sensor_gps"
SUBSYSTEM=="tty", ATTRS{serial}=="FTXYZ123", SYMLINK+="sensor_imu"
```

---

## 3. I/O制御方式の詳細

### 3.1 三つのI/O制御方式

オペレーティングシステムがハードウェアデバイスとデータをやり取りする方式は、大きく3つに分類される。それぞれの方式は、CPU負荷、レイテンシ、スループットのトレードオフが異なり、デバイスの特性や用途に応じて使い分けられる。

#### (a) ポーリング（Programmed I/O / Polling）

CPUがデバイスのステータスレジスタを繰り返し読み取り、操作の完了を検出する方式である。最も単純な実装であるが、待機中もCPUサイクルを消費し続けるため、ビジーウェイトとも呼ばれる。

```c
/*
 * ポーリング方式の擬似コード
 * シリアルポートからの1バイト受信を例にする
 */
#define UART_LSR    0x3FD   /* Line Status Register */
#define UART_RBR    0x3F8   /* Receiver Buffer Register */
#define LSR_DR      0x01    /* Data Ready ビット */

uint8_t uart_read_polling(void)
{
    /* ステータスレジスタを繰り返しチェック（ビジーウェイト） */
    while (!(inb(UART_LSR) & LSR_DR)) {
        /* CPUはここでスピンし続ける
         * 他のタスクを実行できない
         * デバイスが応答しない場合、永久にループする危険がある
         */
        cpu_relax();  /* スピンループ用ヒント命令 */
    }

    /* データレディになったらデータレジスタを読み取る */
    return inb(UART_RBR);
}
```

#### (b) 割り込み駆動I/O（Interrupt-Driven I/O）

デバイスが操作の完了や注目すべきイベントの発生時に、割り込み信号（IRQ）をCPUに送信する方式である。CPUは割り込みを待つ間、他のタスクを実行できるため、ポーリングと比較してCPU効率が大幅に向上する。

```c
/*
 * 割り込み駆動I/Oの擬似コード
 * シリアルポートの割り込みハンドラを例にする
 */
#include <linux/interrupt.h>

/* 受信バッファ（循環バッファ） */
static DECLARE_KFIFO(rx_fifo, unsigned char, 1024);
static DECLARE_WAIT_QUEUE_HEAD(rx_waitq);

/*
 * 割り込みハンドラ（トップハーフ）
 * IRQ発生時にカーネルから呼ばれる
 * 注意: 割り込みコンテキストではスリープ不可
 */
static irqreturn_t uart_irq_handler(int irq, void *dev_id)
{
    uint8_t status = inb(UART_LSR);

    if (!(status & LSR_DR))
        return IRQ_NONE;  /* この割り込みは自分のデバイスではない */

    /* データを読み取り、FIFOに格納 */
    while (status & LSR_DR) {
        uint8_t data = inb(UART_RBR);
        kfifo_put(&rx_fifo, data);
        status = inb(UART_LSR);
    }

    /* 待機中のプロセスを起床 */
    wake_up_interruptible(&rx_waitq);

    return IRQ_HANDLED;  /* 割り込みを処理した */
}

/* ドライバ初期化時にIRQハンドラを登録 */
static int uart_probe(struct platform_device *pdev)
{
    int ret;

    ret = request_irq(IRQ_UART, uart_irq_handler,
                      IRQF_SHARED,       /* 共有IRQ対応 */
                      "my_uart",          /* /proc/interrupts での表示名 */
                      pdev);              /* dev_id: ハンドラに渡される */
    if (ret) {
        dev_err(&pdev->dev, "Failed to request IRQ %d\n", IRQ_UART);
        return ret;
    }

    return 0;
}
```

#### (c) DMA（Direct Memory Access）

DMAコントローラ（またはバスマスタリング対応デバイス自身）がCPUを介さずにデバイスとメインメモリ間で直接データ転送を行う方式である。大量データの転送においてCPU負荷を最小化できるため、ディスクI/O、ネットワーク通信、オーディオ・ビデオストリーミングなど、高スループットが求められる場面で不可欠な技術である。

```c
/*
 * DMA転送の擬似コード
 * ブロックデバイスからのデータ読み取りを例にする
 */
#include <linux/dma-mapping.h>

static int setup_dma_transfer(struct device *dev, void *buffer, size_t len)
{
    dma_addr_t dma_handle;

    /*
     * DMAマッピングの作成
     * CPUの仮想アドレスをデバイスがアクセス可能なバスアドレスに変換
     * IOMMUが存在する場合、IOMMU経由のアドレス変換が行われる
     */
    dma_handle = dma_map_single(dev, buffer, len, DMA_FROM_DEVICE);
    if (dma_mapping_error(dev, dma_handle)) {
        dev_err(dev, "DMA mapping failed\n");
        return -ENOMEM;
    }

    /*
     * デバイスにDMA転送を指示
     * - 転送元/先のバスアドレス
     * - 転送サイズ
     * - 転送方向
     * デバイス固有のレジスタに書き込む
     */
    writel(dma_handle, dev_regs + DMA_ADDR_REG);
    writel(len, dev_regs + DMA_LEN_REG);
    writel(DMA_START | DMA_DIR_READ, dev_regs + DMA_CTRL_REG);

    /* CPUは転送完了を待たず他の処理を行える
     * 転送完了は割り込みで通知される */
    return 0;
}

/* DMA完了割り込みハンドラ */
static irqreturn_t dma_complete_handler(int irq, void *dev_id)
{
    struct device *dev = dev_id;

    /* DMAマッピングを解除（キャッシュ同期も行われる） */
    dma_unmap_single(dev, dma_handle, len, DMA_FROM_DEVICE);

    /* 転送完了を上位層に通知 */
    complete(&dma_completion);

    return IRQ_HANDLED;
}
```

### 3.2 I/O制御方式の比較表

| 特性 | ポーリング | 割り込み駆動 | DMA |
|:-----|:----------|:------------|:----|
| CPU負荷 | 非常に高い（ビジーウェイト） | 低い（イベント駆動） | 最小（転送中はCPU不関与） |
| レイテンシ | 最小（即座に検出） | 中程度（割り込み遅延） | 中程度（セットアップ＋割り込み） |
| スループット | 低い（CPU律速） | 中程度 | 高い（バス帯域幅上限） |
| 実装複雑度 | 最も単純 | 中程度（ハンドラ設計） | 高い（アドレス変換、キャッシュ整合性） |
| 適用場面 | 組み込み、BIOS、短時間待機 | 一般的なI/O操作 | 大量データ転送（ディスク、NIC） |
| マルチタスク適性 | 不適 | 適している | 最も適している |
| デバイス例 | GPIO、簡易センサ | キーボード、マウス | NVMe SSD、10GbE NIC |
| ハードウェア要件 | なし | IRQライン | DMAコントローラ/バスマスタ |
| キャッシュ整合性 | 問題なし | 問題なし | 要考慮（dma_sync_*） |

---

## 4. デバイスドライバの内部構造

### 4.1 キャラクタデバイスドライバの構造

Linuxのキャラクタデバイスドライバは、`file_operations`構造体を通じてユーザ空間にインターフェースを公開する。この構造体は、`open`、`read`、`write`、`ioctl`、`release`（close）などの関数ポインタを含み、VFS（仮想ファイルシステム）層からのファイル操作をデバイス固有の処理にディスパッチする。

```c
/*
 * simplechar.c — 教育用キャラクタデバイスドライバの完全実装
 * メモリバッファをデバイスとして公開する
 */
#include <linux/module.h>
#include <linux/fs.h>         /* file_operations, register_chrdev_region */
#include <linux/cdev.h>       /* cdev 構造体 */
#include <linux/device.h>     /* device_create, class_create */
#include <linux/uaccess.h>    /* copy_to_user, copy_from_user */
#include <linux/mutex.h>

#define DEVICE_NAME   "simplechar"
#define CLASS_NAME    "simple"
#define BUFFER_SIZE   4096

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Educational character device driver");

/* ドライバのプライベートデータ */
struct simplechar_dev {
    struct cdev    cdev;          /* カーネル内cdev構造体 */
    struct class   *class;        /* デバイスクラス */
    struct device  *device;       /* デバイス */
    dev_t          devno;         /* デバイス番号(major:minor) */
    struct mutex   lock;          /* 排他制御用ミューテックス */
    char           buffer[BUFFER_SIZE];  /* データバッファ */
    size_t         data_len;      /* バッファ内の有効データ長 */
    int            open_count;    /* オープン参照カウント */
};

static struct simplechar_dev *sc_dev;

/* --- file_operations コールバック実装 --- */

static int sc_open(struct inode *inode, struct file *filp)
{
    struct simplechar_dev *dev;

    /* container_of でinode内のcdevからデバイス構造体を取得 */
    dev = container_of(inode->i_cdev, struct simplechar_dev, cdev);
    filp->private_data = dev;  /* 以降のread/write/releaseで使用 */

    mutex_lock(&dev->lock);
    dev->open_count++;
    pr_info("%s: opened (count=%d)\n", DEVICE_NAME, dev->open_count);
    mutex_unlock(&dev->lock);

    return 0;
}

static ssize_t sc_read(struct file *filp, char __user *ubuf,
                        size_t count, loff_t *f_pos)
{
    struct simplechar_dev *dev = filp->private_data;
    ssize_t retval;

    mutex_lock(&dev->lock);

    /* ファイルオフセットがデータ長を超えている場合はEOF */
    if (*f_pos >= dev->data_len) {
        retval = 0;  /* EOF */
        goto out;
    }

    /* 読み取り量をデータ残量に制限 */
    if (*f_pos + count > dev->data_len)
        count = dev->data_len - *f_pos;

    /*
     * copy_to_user: カーネル空間 → ユーザ空間へのコピー
     * ユーザ空間ポインタの検証も内部で行われる
     * 戻り値: コピーできなかったバイト数（0 = 成功）
     */
    if (copy_to_user(ubuf, dev->buffer + *f_pos, count)) {
        retval = -EFAULT;  /* 不正なユーザ空間アドレス */
        goto out;
    }

    *f_pos += count;  /* ファイルオフセットを更新 */
    retval = count;
    pr_info("%s: read %zu bytes from offset %lld\n",
            DEVICE_NAME, count, *f_pos - count);

out:
    mutex_unlock(&dev->lock);
    return retval;
}

static ssize_t sc_write(struct file *filp, const char __user *ubuf,
                         size_t count, loff_t *f_pos)
{
    struct simplechar_dev *dev = filp->private_data;
    ssize_t retval;

    mutex_lock(&dev->lock);

    /* バッファオーバーフロー防止 */
    if (*f_pos + count > BUFFER_SIZE)
        count = BUFFER_SIZE - *f_pos;

    if (count == 0) {
        retval = -ENOSPC;  /* No space left on device */
        goto out;
    }

    if (copy_from_user(dev->buffer + *f_pos, ubuf, count)) {
        retval = -EFAULT;
        goto out;
    }

    *f_pos += count;
    if (*f_pos > dev->data_len)
        dev->data_len = *f_pos;

    retval = count;
    pr_info("%s: wrote %zu bytes at offset %lld\n",
            DEVICE_NAME, count, *f_pos - count);

out:
    mutex_unlock(&dev->lock);
    return retval;
}

static int sc_release(struct inode *inode, struct file *filp)
{
    struct simplechar_dev *dev = filp->private_data;

    mutex_lock(&dev->lock);
    dev->open_count--;
    pr_info("%s: released (count=%d)\n", DEVICE_NAME, dev->open_count);
    mutex_unlock(&dev->lock);

    return 0;
}

/*
 * file_operations 構造体 — VFSとドライバの接続点
 * ここで定義されていない操作は、VFSのデフォルト動作になる
 */
static const struct file_operations sc_fops = {
    .owner   = THIS_MODULE,    /* モジュール参照カウント管理 */
    .open    = sc_open,
    .read    = sc_read,
    .write   = sc_write,
    .release = sc_release,
    /* .llseek, .unlocked_ioctl, .poll, .mmap 等も定義可能 */
};

/* --- モジュール初期化・終了 --- */

static int __init sc_init(void)
{
    int ret;

    /* デバイス構造体の割り当て */
    sc_dev = kzalloc(sizeof(*sc_dev), GFP_KERNEL);
    if (!sc_dev)
        return -ENOMEM;

    mutex_init(&sc_dev->lock);

    /* Step 1: デバイス番号の動的割り当て */
    ret = alloc_chrdev_region(&sc_dev->devno, 0, 1, DEVICE_NAME);
    if (ret < 0) {
        pr_err("%s: alloc_chrdev_region failed\n", DEVICE_NAME);
        goto err_alloc_region;
    }
    pr_info("%s: registered with major=%d, minor=%d\n",
            DEVICE_NAME, MAJOR(sc_dev->devno), MINOR(sc_dev->devno));

    /* Step 2: cdev構造体の初期化と登録 */
    cdev_init(&sc_dev->cdev, &sc_fops);
    sc_dev->cdev.owner = THIS_MODULE;
    ret = cdev_add(&sc_dev->cdev, sc_dev->devno, 1);
    if (ret < 0) {
        pr_err("%s: cdev_add failed\n", DEVICE_NAME);
        goto err_cdev_add;
    }

    /* Step 3: デバイスクラスの作成（/sys/class/simple/） */
    sc_dev->class = class_create(CLASS_NAME);
    if (IS_ERR(sc_dev->class)) {
        ret = PTR_ERR(sc_dev->class);
        goto err_class;
    }

    /* Step 4: デバイスノードの作成（/dev/simplechar） */
    sc_dev->device = device_create(sc_dev->class, NULL,
                                    sc_dev->devno, NULL, DEVICE_NAME);
    if (IS_ERR(sc_dev->device)) {
        ret = PTR_ERR(sc_dev->device);
        goto err_device;
    }

    pr_info("%s: driver initialized successfully\n", DEVICE_NAME);
    return 0;

/* エラー時の後始末（初期化の逆順） */
err_device:
    class_destroy(sc_dev->class);
err_class:
    cdev_del(&sc_dev->cdev);
err_cdev_add:
    unregister_chrdev_region(sc_dev->devno, 1);
err_alloc_region:
    kfree(sc_dev);
    return ret;
}

static void __exit sc_exit(void)
{
    /* 初期化の逆順で後始末 */
    device_destroy(sc_dev->class, sc_dev->devno);
    class_destroy(sc_dev->class);
    cdev_del(&sc_dev->cdev);
    unregister_chrdev_region(sc_dev->devno, 1);
    kfree(sc_dev);
    pr_info("%s: driver removed\n", DEVICE_NAME);
}

module_init(sc_init);
module_exit(sc_exit);
```

### 4.2 ドライバ初期化のシーケンス

上記のコードにおけるドライバ初期化の各ステップを図示する。

```
ドライバ初期化シーケンス（成功パス）:

  module_init(sc_init) 呼び出し
       │
       ▼
  ┌─────────────────────────────┐
  │ 1. kzalloc()                │  ドライバ構造体のメモリ確保
  │    sc_dev を確保             │
  └──────────┬──────────────────┘
             ▼
  ┌─────────────────────────────┐
  │ 2. alloc_chrdev_region()    │  メジャー/マイナー番号の割り当て
  │    devno = (major, minor)   │  /proc/devices に登録
  └──────────┬──────────────────┘
             ▼
  ┌─────────────────────────────┐
  │ 3. cdev_init() + cdev_add() │  fopsとcdevの紐付け
  │    VFSからの操作をルーティング │ cdev_mapに登録
  └──────────┬──────────────────┘
             ▼
  ┌─────────────────────────────┐
  │ 4. class_create()           │  /sys/class/simple/ 作成
  │    sysfsエントリを作成        │  udevが監視
  └──────────┬──────────────────┘
             ▼
  ┌─────────────────────────────┐
  │ 5. device_create()          │  uevent発行
  │    /dev/simplechar 作成      │  udevがデバイスノード作成
  └──────────┬──────────────────┘
             ▼
        初期化完了 (return 0)

  ※ 各ステップが失敗した場合、それまでに確保した
     リソースを逆順で解放する（gotoエラーハンドリング）
```

### 4.3 file_operations構造体の主要メンバ

`file_operations`はLinuxドライバの中核となる構造体であり、VFSが呼び出すコールバック関数のテーブルである。以下に主要なメンバとその用途を整理する。

| メンバ | システムコール | 用途 |
|:-------|:-------------|:-----|
| `.owner` | — | モジュール参照カウント管理（`THIS_MODULE`） |
| `.open` | `open(2)` | デバイスオープン時の初期化処理 |
| `.release` | `close(2)` | 最後のfdがクローズされたときの後処理 |
| `.read` | `read(2)` | デバイスからユーザ空間へのデータ読み取り |
| `.write` | `write(2)` | ユーザ空間からデバイスへのデータ書き込み |
| `.unlocked_ioctl` | `ioctl(2)` | デバイス固有の制御コマンド |
| `.compat_ioctl` | `ioctl(2)` | 32bitプロセスからの64bitカーネル互換ioctl |
| `.poll` | `poll(2)`/`select(2)` | I/O可能状態の通知（非同期I/O） |
| `.mmap` | `mmap(2)` | デバイスメモリのユーザ空間マッピング |
| `.llseek` | `lseek(2)` | ファイルオフセットの変更 |
| `.fasync` | `fcntl(2)` | 非同期通知の設定（SIGIO） |
| `.flush` | `close(2)` | fd毎のクローズ時処理（全fdクローズ前） |

---

## 5. プラットフォームバスとデバイスツリー

### 5.1 バスモデルとデバイス-ドライバのマッチング

Linuxカーネルは、デバイスとドライバを結び付けるための抽象的なバスモデルを持つ。PCIやUSBのように自己記述的な（ディスカバリ可能な）バスでは、デバイスが自身のベンダーID・プロダクトIDを報告し、カーネルがそれに合致するドライバを自動的に選択する。

一方、組み込みシステムで使用されるSoC（System on Chip）上のデバイスは自己記述能力を持たないことが多い。このようなデバイスを扱うための仮想的なバスが「プラットフォームバス（platform bus）」である。

```
バスモデルの全体像:

  ┌──────────────────────────────────────────────┐
  │            Linuxデバイスモデル                   │
  │                                              │
  │  bus_type ─── device_driver ─── device       │
  │   (バス)        (ドライバ)        (デバイス)    │
  │                                              │
  │  マッチング:                                   │
  │    bus->match(dev, drv) が真のとき             │
  │    drv->probe(dev) が呼ばれる                  │
  │                                              │
  ├─────────────┬───────────────┬────────────────┤
  │  PCI Bus    │   USB Bus     │ Platform Bus   │
  │             │               │                │
  │ vendor_id + │ vendor_id +   │ compatible文字列│
  │ device_id   │ product_id    │ または name    │
  │ でマッチ     │ でマッチ       │ でマッチ        │
  │             │               │                │
  │ 自動検出可能 │ ホットプラグ対応│ 静的記述が必要  │
  │ (エニュメレ  │ (デバイス      │ (デバイスツリー  │
  │  ーション)   │  ディスクリプタ)│  or ACPI)      │
  └─────────────┴───────────────┴────────────────┘
```

### 5.2 デバイスツリー（Device Tree）

デバイスツリーは、ハードウェアの構成をツリー構造で記述するデータフォーマットである。ARM、RISC-V、PowerPCなどのアーキテクチャで広く使用されている。カーネルのソースコードにハードウェア情報をハードコードする代わりに、DTB（Device Tree Blob）として外部から供給する。

```
デバイスツリーソース (.dts) の例:

/dts-v1/;

/ {
    compatible = "vendor,board-name";
    model = "Example Development Board";

    /* SoC 内部のデバイス */
    soc {
        compatible = "simple-bus";
        #address-cells = <1>;
        #size-cells = <1>;

        /* UART コントローラ */
        uart0: serial@10010000 {
            compatible = "ns16550a";          /* ドライバマッチに使用 */
            reg = <0x10010000 0x100>;         /* レジスタのベースアドレスとサイズ */
            interrupts = <10>;                /* IRQ番号 */
            clock-frequency = <48000000>;     /* クロック周波数 */
            status = "okay";                  /* デバイスを有効化 */
        };

        /* GPIO コントローラ */
        gpio0: gpio@10020000 {
            compatible = "vendor,gpio-controller";
            reg = <0x10020000 0x40>;
            #gpio-cells = <2>;
            gpio-controller;
            interrupt-controller;
            #interrupt-cells = <2>;
        };

        /* I2C バス上の温度センサ */
        i2c0: i2c@10030000 {
            compatible = "vendor,i2c-controller";
            reg = <0x10030000 0x100>;
            #address-cells = <1>;
            #size-cells = <0>;

            temperature-sensor@48 {
                compatible = "ti,tmp102";
                reg = <0x48>;                 /* I2Cスレーブアドレス */
            };
        };
    };

    /* ボード上のLED */
    leds {
        compatible = "gpio-leds";
        led-heartbeat {
            gpios = <&gpio0 5 0>;             /* GPIO0のpin5を使用 */
            linux,default-trigger = "heartbeat";
        };
    };
};
```

デバイスツリーのコンパイルとデプロイ:

```bash
# デバイスツリーソース (.dts) をバイナリ (.dtb) にコンパイル
$ dtc -I dts -O dtb -o board.dtb board.dts

# 既存の .dtb を逆コンパイルして内容を確認
$ dtc -I dtb -O dts -o decompiled.dts /boot/dtbs/board.dtb

# 実行中のシステムのデバイスツリーを確認
$ ls /proc/device-tree/
$ cat /proc/device-tree/model

# デバイスツリーオーバーレイ（既存DTBに差分を適用）
$ dtc -I dts -O dtb -o overlay.dtbo overlay.dts
$ sudo dtoverlay overlay.dtbo
```

### 5.3 ACPI（Advanced Configuration and Power Interface）

x86/x64プラットフォームでは、デバイスツリーの代わりにACPIがハードウェア記述の標準規格として使用される。ACPIテーブルはファームウェア（BIOS/UEFI）が提供し、カーネルがブート時に解析する。AML（ACPI Machine Language）と呼ばれるバイトコードが含まれ、カーネル内のACPIインタプリタが実行する。

```bash
# ACPIテーブルの確認
$ sudo acpidump | head -30
$ sudo acpidump -b     # バイナリ形式でダンプ
$ iasl -d DSDT.aml     # AMLを逆アセンブル

# ACPIデバイスの一覧
$ ls /sys/bus/acpi/devices/

# 電源管理状態の確認
$ cat /sys/bus/acpi/devices/*/power_state
```

---

## 6. ユーザ空間ドライバ

### 6.1 ユーザ空間ドライバの動機

従来のカーネルドライバには以下の課題がある。

- カーネルクラッシュのリスク: ドライバのバグがシステム全体をクラッシュさせる（カーネルパニック）
- デバッグの困難さ: カーネル空間ではgdb等の一般的なデバッガが直接使えない
- 開発サイクルの長さ: 変更のたびにモジュールのビルド・ロード・テストが必要
- ライセンスの制約: GPLカーネルモジュールはGPL互換ライセンスが求められることがある

これらを解決するために、デバイスドライバの一部または全部をユーザ空間で動作させるフレームワークが存在する。

### 6.2 UIO（Userspace I/O）

UIOは、カーネル側に最小限のスタブドライバを置き、実際のデバイス制御ロジックをユーザ空間のプロセスとして実装するフレームワークである。

```c
/*
 * UIOユーザ空間ドライバの例
 * /dev/uio0 を通じてデバイスレジスタにアクセスする
 */
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdint.h>
#include <unistd.h>

int main(void)
{
    int fd;
    volatile uint32_t *regs;
    uint32_t irq_count;

    /* UIOデバイスをオープン */
    fd = open("/dev/uio0", O_RDWR);
    if (fd < 0) {
        perror("open /dev/uio0");
        return 1;
    }

    /*
     * デバイスレジスタをユーザ空間にマッピング
     * offset 0 = BAR0（PCI）またはレジスタ領域
     * mmapにより物理レジスタに直接アクセス可能になる
     */
    regs = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                MAP_SHARED, fd, 0);
    if (regs == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    /* レジスタへの読み書き（デバイス固有の操作） */
    printf("Device ID: 0x%08x\n", regs[0]);  /* レジスタ0を読み取り */
    regs[1] = 0x00000001;                     /* レジスタ1に書き込み */

    /*
     * 割り込み待ち
     * read() でブロックし、割り込み発生時に返る
     * 戻り値は割り込み発生回数
     */
    while (1) {
        ssize_t n = read(fd, &irq_count, sizeof(irq_count));
        if (n != sizeof(irq_count)) {
            perror("read (IRQ wait)");
            break;
        }
        printf("Interrupt #%u received\n", irq_count);

        /* 割り込み処理（ユーザ空間で自由にロジックを記述） */
        uint32_t status = regs[2];  /* ステータスレジスタ読み取り */
        regs[3] = status;           /* 割り込みクリア */
    }

    munmap((void *)regs, 4096);
    close(fd);
    return 0;
}
```

### 6.3 VFIO（Virtual Function I/O）

VFIOは、IOMMUを活用してデバイスをユーザ空間に安全にパススルーするフレームワークである。仮想化環境でゲストOSにデバイスを直接割り当てるPCIパススルーや、DPDK（Data Plane Development Kit）などの高性能ネットワーキングスタックで使用される。

```
UIO と VFIO の比較:

  UIO:
  ┌──────────┐     ┌──────────────┐     ┌──────────┐
  │ ユーザ空間 │────→│ /dev/uio0    │────→│ デバイス   │
  │ ドライバ   │     │ (mmap+read)  │     │ レジスタ   │
  └──────────┘     └──────────────┘     └──────────┘
       │                                       ▲
       │  DMAバッファは                          │
       │  hugepagesで確保      直接メモリアクセス   │
       └───────────────────────────────────────┘
       ※ IOMMUによる保護なし → アドレス指定を誤ると危険

  VFIO:
  ┌──────────┐     ┌──────────────┐     ┌──────────┐
  │ ユーザ空間 │────→│ /dev/vfio/*  │────→│ IOMMU    │
  │ ドライバ   │     │ (ioctl+mmap) │     │          │
  └──────────┘     └──────────────┘     └────┬─────┘
                                              │
                        IOMMUがDMAアドレスを     │ アドレス変換
                        検証・変換               │ + 保護
                                              ▼
                                        ┌──────────┐
                                        │ デバイス   │
                                        └──────────┘
       ※ IOMMUにより不正なDMAアドレスはブロックされる
```

| 特性 | UIO | VFIO |
|:-----|:----|:-----|
| IOMMU要件 | 不要 | 必須 |
| DMAアクセス保護 | なし（任意アドレスにDMA可能） | あり（IOMMUで制限） |
| 割り込み処理 | read()でブロック | eventfd経由 |
| 仮想化対応 | 限定的 | PCIパススルーに最適 |
| 複数デバイスの分離 | 不可 | IOMMUグループ単位で分離 |
| パフォーマンス | 高い | 高い（IOMMU変換のオーバーヘッドは小さい） |
| セキュリティ | 低い（root権限必須） | 高い（IOMMUで保護） |
| 代表的ユーザ | 産業用/組込デバイス | DPDK, SPDK, 仮想化 |

### 6.4 FUSE（Filesystem in Userspace）

FUSEはファイルシステムをユーザ空間で実装するためのフレームワークであり、一種のユーザ空間ドライバと見なすことができる。SSHFS、NTFS-3G、GlusterFSなど多数のプロジェクトがFUSEを活用している。

---

## 7. 割り込み処理の高度なトピック

### 7.1 トップハーフとボトムハーフ

割り込みハンドラ（トップハーフ）はハードウェア割り込みを処理するコンテキストで実行されるため、以下の厳しい制約がある。

- **スリープ不可**: mutex_lock()やkmalloc(GFP_KERNEL)などスリープする可能性のある関数を呼べない
- **実行時間の最小化**: 割り込みが禁止された状態で実行されるため、長時間の処理は他の割り込みの遅延を引き起こす
- **ユーザ空間アクセス不可**: copy_to_user()/copy_from_user()は使えない

これらの制約を緩和するため、割り込み処理を「トップハーフ（即座に実行すべき最小限の処理）」と「ボトムハーフ（後で実行してよい重い処理）」に分割する設計が一般的である。

```
割り込み処理の分割モデル:

  ハードウェア割り込み発生
       │
       ▼
  ┌─────────────────────────────────┐
  │ トップハーフ（ハードIRQコンテキスト） │
  │                                 │
  │  ・割り込み原因の確認             │  ← 割り込み禁止状態
  │  ・デバイスの割り込みフラグクリア   │  ← スリープ不可
  │  ・最低限のデータ退避             │  ← 最速で完了すべき
  │  ・ボトムハーフのスケジュール       │
  │                                 │
  │  return IRQ_HANDLED;            │
  └──────────┬──────────────────────┘
             │ スケジュール
             ▼
  ┌─────────────────────────────────┐
  │ ボトムハーフ（遅延実行）           │
  │                                 │
  │ ┌───────────┐ ┌───────────┐     │
  │ │ softirq   │ │ tasklet   │     │  ← 割り込み有効状態
  │ │ (高優先)   │ │ (中優先)   │     │  ← ただしスリープ不可
  │ └───────────┘ └───────────┘     │
  │                                 │
  │ ┌───────────┐ ┌──────────────┐  │
  │ │ workqueue │ │ threaded IRQ │  │  ← プロセスコンテキスト
  │ │ (汎用)    │ │ (推奨)       │  │  ← スリープ可能
  │ └───────────┘ └──────────────┘  │
  └─────────────────────────────────┘
```

### 7.2 ボトムハーフメカニズムの比較

| メカニズム | コンテキスト | スリープ | 優先度 | 用途 |
|:----------|:-----------|:--------|:------|:-----|
| softirq | ソフト割り込み | 不可 | 最高 | ネットワーク(NET_RX), ブロックI/O |
| tasklet | ソフト割り込み | 不可 | 高 | 汎用的な短い遅延処理 |
| workqueue | プロセス | 可能 | 通常 | スリープを要する処理全般 |
| threaded IRQ | プロセス（カーネルスレッド） | 可能 | 高 | 現代的なドライバの推奨方式 |

### 7.3 Threaded IRQ（スレッド化割り込み）

現代のLinuxカーネルでは、`request_threaded_irq()`によるスレッド化割り込みが推奨されている。ハードIRQハンドラで最小限の処理を行った後、カーネルスレッドとして残りの処理を実行する。プロセスコンテキストで動作するためスリープが可能であり、mutexの使用やDMA完了待ちなどが自然に記述できる。

```c
/*
 * スレッド化割り込みの実装例
 */
static irqreturn_t sensor_hard_irq(int irq, void *dev_id)
{
    struct sensor_dev *sdev = dev_id;

    /*
     * ハードIRQハンドラ（トップハーフ）
     * 最小限の処理のみ行う
     */
    sdev->irq_status = readl(sdev->regs + IRQ_STATUS_REG);

    if (!(sdev->irq_status & IRQ_PENDING))
        return IRQ_NONE;  /* このデバイスの割り込みではない */

    /* 割り込みをマスク（スレッドハンドラ完了まで抑制） */
    writel(0, sdev->regs + IRQ_ENABLE_REG);

    return IRQ_WAKE_THREAD;  /* スレッドハンドラを起床 */
}

static irqreturn_t sensor_thread_fn(int irq, void *dev_id)
{
    struct sensor_dev *sdev = dev_id;
    int ret;

    /*
     * スレッドハンドラ（ボトムハーフ）
     * プロセスコンテキストで動作 → スリープ可能
     */

    /* I2C経由でセンサデータを読み取り（スリープを伴う） */
    mutex_lock(&sdev->lock);
    ret = i2c_smbus_read_word_data(sdev->i2c_client, DATA_REG);
    if (ret >= 0) {
        sdev->last_value = ret;
        sysfs_notify(&sdev->dev->kobj, NULL, "value");
    }
    mutex_unlock(&sdev->lock);

    /* 割り込みを再度有効化 */
    writel(IRQ_ENABLE, sdev->regs + IRQ_ENABLE_REG);

    return IRQ_HANDLED;
}

/* ドライバ初期化時 */
static int sensor_probe(struct platform_device *pdev)
{
    int ret;

    ret = request_threaded_irq(
        sdev->irq,
        sensor_hard_irq,     /* ハードIRQハンドラ（トップハーフ） */
        sensor_thread_fn,     /* スレッドハンドラ（ボトムハーフ） */
        IRQF_ONESHOT,         /* スレッド完了まで割り込みをマスク */
        "sensor_drv",
        sdev
    );

    return ret;
}
```

### 7.4 MSI/MSI-X（Message Signaled Interrupts）

PCIe世代のデバイスでは、従来のピンベース割り込み（INTx）に代わり、MSI（Message Signaled Interrupts）またはMSI-X（Extended MSI）が使用される。MSIはメモリ書き込みにより割り込みを通知する方式であり、以下の利点がある。

- 物理的なIRQラインが不要（共有による問題がない）
- 1デバイスに複数の割り込みベクタを割り当て可能（MSI-X: 最大2048本）
- レイテンシが低い（メモリ書き込みのみ）
- 割り込みの順序保証がある

---

## 8. 電源管理とサスペンド/レジューム

### 8.1 ランタイム電源管理

デバイスドライバは、システムの電源管理に積極的に参加する必要がある。Linuxカーネルは、デバイスレベルの電源管理として「ランタイムPM」と「システムスリープ」の2つのメカニズムを提供する。

ランタイムPMは、個々のデバイスが使用されていない時に自動的に低電力状態に遷移させる仕組みである。

```c
/*
 * ランタイム電源管理の実装例
 */
#include <linux/pm_runtime.h>

static int mydev_probe(struct platform_device *pdev)
{
    struct mydev *dev = platform_get_drvdata(pdev);

    /* ランタイムPMの有効化 */
    pm_runtime_set_active(&pdev->dev);
    pm_runtime_enable(&pdev->dev);

    /*
     * 自動サスペンドの設定
     * 最後の操作から2秒後に自動的にサスペンドする
     */
    pm_runtime_set_autosuspend_delay(&pdev->dev, 2000);
    pm_runtime_use_autosuspend(&pdev->dev);

    return 0;
}

/* デバイスを使用する前に呼ぶ */
static int mydev_do_io(struct mydev *dev)
{
    int ret;

    /* デバイスをアクティブ状態に遷移（必要なら電源ON） */
    ret = pm_runtime_get_sync(dev->dev);
    if (ret < 0) {
        pm_runtime_put_noidle(dev->dev);
        return ret;
    }

    /* デバイスI/O操作 */
    writel(cmd, dev->regs + CMD_REG);

    /* 使用完了を通知（自動サスペンドタイマー開始） */
    pm_runtime_mark_last_busy(dev->dev);
    pm_runtime_put_autosuspend(dev->dev);

    return 0;
}

/* ランタイムサスペンド・レジュームコールバック */
static int mydev_runtime_suspend(struct device *dev)
{
    struct mydev *mdev = dev_get_drvdata(dev);

    /* デバイスのクロックを停止 */
    clk_disable_unprepare(mdev->clk);
    /* レギュレータをOFF */
    regulator_disable(mdev->supply);

    return 0;
}

static int mydev_runtime_resume(struct device *dev)
{
    struct mydev *mdev = dev_get_drvdata(dev);
    int ret;

    /* レギュレータをON */
    ret = regulator_enable(mdev->supply);
    if (ret)
        return ret;
    /* クロックを再開 */
    ret = clk_prepare_enable(mdev->clk);
    if (ret) {
        regulator_disable(mdev->supply);
        return ret;
    }

    /* デバイスの再初期化（レジスタの復元等） */
    mydev_hw_init(mdev);

    return 0;
}

static const struct dev_pm_ops mydev_pm_ops = {
    SET_RUNTIME_PM_OPS(mydev_runtime_suspend,
                       mydev_runtime_resume, NULL)
    SET_SYSTEM_SLEEP_PM_OPS(mydev_system_suspend,
                            mydev_system_resume)
};
```

### 8.2 システムスリープ（S3/S4）

システム全体がサスペンド状態（S3: Sleep / S4: Hibernate）に遷移する際、すべてのデバイスドライバのsuspend/resumeコールバックが呼ばれる。ドライバは以下を行う必要がある。

**サスペンド時（suspend）:**
1. 進行中のI/O操作を完了またはキャンセルする
2. 新しいI/O要求の受付を停止する
3. デバイスのハードウェア状態を保存する
4. 割り込みを無効化する
5. デバイスを低電力状態に遷移させる

**レジューム時（resume）:**
1. デバイスの電源を復帰する
2. ハードウェア状態を復元する（レジスタの再設定）
3. 割り込みを再有効化する
4. I/O要求の受付を再開する

### 8.3 電源状態の階層

```
システム全体の電源状態（ACPIベース）:

  S0 (Working)     ── 通常動作
  │
  ├── S0ix (Modern Standby) ── 低電力アイドル（ネットワーク維持可能）
  │
  S1 (Standby)     ── CPU停止、メモリ保持
  │
  S2 (—)           ── ほぼ未使用
  │
  S3 (Sleep)       ── メモリのみ通電（Suspend to RAM）
  │
  S4 (Hibernate)   ── メモリ内容をディスクに保存、完全電源断
  │
  S5 (Soft Off)    ── ソフトウェア電源OFF

デバイスレベルの電源状態:

  D0 (Full Power)  ── 完全動作状態
  │
  D1 (Light Sleep) ── 低電力（一部機能停止）
  │
  D2 (Deep Sleep)  ── さらに低電力（コンテキスト部分喪失）
  │
  D3hot            ── 最低電力（バス接続は維持）
  │
  D3cold           ── 完全電源断（バス接続も切断）
```

---

## 9. sysfsとデバイス属性

### 9.1 sysfsの役割

sysfsは、カーネル内部のオブジェクト（デバイス、ドライバ、バス等）をファイルシステムツリーとして `/sys` にエクスポートする仮想ファイルシステムである。ドライバ開発者は、sysfs属性を通じてデバイスの設定値や状態をユーザ空間に公開できる。

```bash
# sysfsの構造を確認する例
$ ls /sys/class/net/eth0/
address  carrier  device  duplex  mtu  operstate  speed  statistics  ...

$ cat /sys/class/net/eth0/mtu
1500

$ cat /sys/class/net/eth0/address
00:1a:2b:3c:4d:5e

# ブロックデバイスのI/Oスケジューラ確認・変更
$ cat /sys/block/sda/queue/scheduler
[mq-deadline] kyber bfq none

$ echo "bfq" | sudo tee /sys/block/sda/queue/scheduler

# デバイスの電源状態を確認
$ cat /sys/devices/pci0000:00/0000:00:1f.0/power/runtime_status
active
```

### 9.2 カスタムsysfs属性の実装

ドライバ固有の設定や状態をsysfs経由で公開する方法を以下に示す。

```c
/*
 * sysfs属性の実装例
 * /sys/class/simple/simplechar/debug_level として公開
 */

static int debug_level = 0;

/* read: cat /sys/.../debug_level */
static ssize_t debug_level_show(struct device *dev,
                                 struct device_attribute *attr,
                                 char *buf)
{
    return sysfs_emit(buf, "%d\n", debug_level);
}

/* write: echo 3 > /sys/.../debug_level */
static ssize_t debug_level_store(struct device *dev,
                                  struct device_attribute *attr,
                                  const char *buf, size_t count)
{
    int val;
    int ret;

    ret = kstrtoint(buf, 10, &val);
    if (ret)
        return ret;

    if (val < 0 || val > 5)
        return -EINVAL;

    debug_level = val;
    pr_info("debug_level set to %d\n", debug_level);

    return count;
}

/* DEVICE_ATTR_RW マクロで show/store を登録 */
static DEVICE_ATTR_RW(debug_level);

/* 複数属性をグループ化 */
static struct attribute *mydev_attrs[] = {
    &dev_attr_debug_level.attr,
    NULL,
};
ATTRIBUTE_GROUPS(mydev);

/* probe() でデバイス作成時に属性グループを指定 */
/* class->dev_groups = mydev_groups; */
```

---

## 10. デバッグ技法

### 10.1 カーネルデバッグの基本ツール

| ツール | 用途 | 使用場面 |
|:-------|:-----|:---------|
| `printk` / `pr_info` | カーネルログ出力 | 基本的なトレース |
| `dmesg` | カーネルリングバッファの表示 | ドライバメッセージ確認 |
| `ftrace` | 関数トレース | 呼び出し経路の追跡 |
| `perf` | パフォーマンスプロファイリング | ボトルネック特定 |
| `crash` / `kdump` | カーネルクラッシュダンプ解析 | 事後解析 |
| `/proc/interrupts` | 割り込み統計 | IRQ配分の確認 |
| `/proc/iomem` | I/Oメモリマップ | アドレス空間の確認 |
| `/proc/ioports` | I/Oポートマップ | ポートアドレスの確認 |
| `strace` | システムコールトレース | ユーザ空間からの呼び出し追跡 |

### 10.2 動的デバッグ（Dynamic Debug）

Linuxカーネルのpr_debug()やdev_dbg()は、動的デバッグ機構を通じて実行時にON/OFF切り替えが可能である。

```bash
# 動的デバッグの有効化
# 特定のファイル内の全デバッグメッセージを有効化
$ echo "file mydriver.c +p" | sudo tee /sys/kernel/debug/dynamic_debug/control

# 特定の関数のデバッグメッセージを有効化
$ echo "func mydev_probe +p" | sudo tee /sys/kernel/debug/dynamic_debug/control

# 特定モジュールの全デバッグメッセージを有効化
$ echo "module mydriver +p" | sudo tee /sys/kernel/debug/dynamic_debug/control

# 現在有効なデバッグポイントの確認
$ cat /sys/kernel/debug/dynamic_debug/control | grep mydriver

# ftraceによる関数トレース
$ echo function > /sys/kernel/debug/tracing/current_tracer
$ echo mydev_* > /sys/kernel/debug/tracing/set_ftrace_filter
$ echo 1 > /sys/kernel/debug/tracing/tracing_on
$ cat /sys/kernel/debug/tracing/trace
```

---

## 11. アンチパターン集

### 11.1 アンチパターン1: 割り込みハンドラ内でのスリープ

**問題**: 割り込みコンテキスト（トップハーフ）でスリープ可能な関数を呼び出すと、カーネルが「BUG: scheduling while atomic」というエラーを発生させ、最悪の場合カーネルパニックに至る。

```c
/* NG: 割り込みハンドラ内でのスリープ（絶対にやってはいけない） */
static irqreturn_t bad_irq_handler(int irq, void *dev_id)
{
    struct my_dev *dev = dev_id;
    void *buf;

    /*
     * NG: GFP_KERNEL はスリープ可能な割り当て
     * 割り込みコンテキストではスリープ不可のため、
     * カーネルがBUGメッセージを出力しクラッシュする可能性がある
     */
    buf = kmalloc(4096, GFP_KERNEL);  /* NG! */

    /*
     * NG: mutex_lock はスリープ可能
     * 別のコンテキストがロックを保持している場合、
     * 割り込みハンドラがスリープしてデッドロックに至る
     */
    mutex_lock(&dev->lock);           /* NG! */

    /* NG: copy_to_user はページフォルトを起こす可能性がある */
    copy_to_user(ubuf, data, len);    /* NG! */

    mutex_unlock(&dev->lock);
    kfree(buf);
    return IRQ_HANDLED;
}

/* OK: 正しい実装 — スレッド化割り込みまたはworkqueueを使用 */
static irqreturn_t good_hard_irq(int irq, void *dev_id)
{
    struct my_dev *dev = dev_id;

    /* 最小限の処理: ステータス読み取りと割り込みクリア */
    dev->irq_status = readl(dev->regs + STATUS_REG);
    writel(dev->irq_status, dev->regs + IRQ_ACK_REG);

    return IRQ_WAKE_THREAD;  /* スレッドハンドラに委譲 */
}

static irqreturn_t good_thread_fn(int irq, void *dev_id)
{
    struct my_dev *dev = dev_id;
    void *buf;

    /* OK: プロセスコンテキストではGFP_KERNELが使える */
    buf = kmalloc(4096, GFP_KERNEL);

    /* OK: mutex_lockも使用可能 */
    mutex_lock(&dev->lock);
    /* データ処理 */
    mutex_unlock(&dev->lock);

    kfree(buf);
    return IRQ_HANDLED;
}
```

**教訓**: 割り込みコンテキストで安全に使える関数は限られている。メモリ割り当てには`GFP_ATOMIC`を使い、排他制御には`spin_lock_irqsave()`を使い、重い処理はボトムハーフに委譲する。最も推奨される方法は`request_threaded_irq()`によるスレッド化割り込みである。

### 11.2 アンチパターン2: リソースリークのあるエラーハンドリング

**問題**: ドライバの初期化関数で複数のリソースを獲得する際、途中のステップでエラーが発生した場合に、それまでに獲得したリソースを適切に解放しないとリソースリークが発生する。

```c
/* NG: リソースリークのあるエラーハンドリング */
static int bad_probe(struct platform_device *pdev)
{
    struct my_dev *dev;
    int ret;

    dev = kzalloc(sizeof(*dev), GFP_KERNEL);
    if (!dev)
        return -ENOMEM;

    dev->clk = clk_get(&pdev->dev, "main_clk");
    if (IS_ERR(dev->clk))
        return PTR_ERR(dev->clk);    /* NG: dev のメモリが解放されていない */

    ret = clk_prepare_enable(dev->clk);
    if (ret)
        return ret;                   /* NG: clk と dev が解放されていない */

    dev->regs = devm_ioremap_resource(&pdev->dev, res);
    if (IS_ERR(dev->regs))
        return PTR_ERR(dev->regs);    /* NG: clk が有効なまま、dev が未解放 */

    /* ... */
    return 0;
}

/* OK: 正しいエラーハンドリング — gotoチェーンパターン */
static int good_probe(struct platform_device *pdev)
{
    struct my_dev *dev;
    int ret;

    dev = kzalloc(sizeof(*dev), GFP_KERNEL);
    if (!dev)
        return -ENOMEM;

    dev->clk = clk_get(&pdev->dev, "main_clk");
    if (IS_ERR(dev->clk)) {
        ret = PTR_ERR(dev->clk);
        goto err_free_dev;            /* dev を解放 */
    }

    ret = clk_prepare_enable(dev->clk);
    if (ret)
        goto err_put_clk;             /* clk を put し、dev を解放 */

    dev->regs = devm_ioremap_resource(&pdev->dev, res);
    if (IS_ERR(dev->regs)) {
        ret = PTR_ERR(dev->regs);
        goto err_disable_clk;         /* clk を disable/put し、dev を解放 */
    }

    return 0;

err_disable_clk:
    clk_disable_unprepare(dev->clk);
err_put_clk:
    clk_put(dev->clk);
err_free_dev:
    kfree(dev);
    return ret;
}

/*
 * さらに良い方法: devm_* (device-managed) APIを使用
 * devm_ 系APIで確保したリソースはドライバのremove時に自動解放される
 */
static int best_probe(struct platform_device *pdev)
{
    struct my_dev *dev;

    /* devm_kzalloc: デバイスのライフサイクルに紐付けたメモリ確保 */
    dev = devm_kzalloc(&pdev->dev, sizeof(*dev), GFP_KERNEL);
    if (!dev)
        return -ENOMEM;

    /* devm_clk_get: 自動解放されるクロック取得 */
    dev->clk = devm_clk_get(&pdev->dev, "main_clk");
    if (IS_ERR(dev->clk))
        return PTR_ERR(dev->clk);  /* devm_kzalloc分は自動解放 */

    /* devm_ioremap_resource: 自動解放されるI/Oメモリマッピング */
    dev->regs = devm_ioremap_resource(&pdev->dev, res);
    if (IS_ERR(dev->regs))
        return PTR_ERR(dev->regs);  /* 上記リソースすべて自動解放 */

    return 0;
    /* remove時: devm_* で確保した全リソースが逆順に自動解放 */
}
```

**教訓**: Linuxカーネルの `devm_*`（device-managed）APIを積極的に活用する。devm_ 系APIで確保したリソースは、ドライバのアンバインド時に自動的に解放されるため、エラーハンドリングのgotoチェーンを大幅に簡素化でき、リソースリークのリスクを根本的に排除できる。

### 11.3 アンチパターン3: 不適切なロック粒度

**問題**: 大域ロック（BKL: Big Kernel Lock）のようにドライバ全体を単一のロックで保護すると、並行性が著しく損なわれる。逆にロック粒度が細かすぎると、デッドロックやrace conditionのリスクが増大する。

- 粒度が粗すぎる: 1つのmutexでドライバ全体を保護 → 全操作がシリアライズされ、マルチコアの性能を活かせない
- 粒度が細かすぎる: データ構造のフィールドごとにロック → ロック順序の管理が困難になりデッドロックのリスク増大
- 適切な粒度: デバイスインスタンスごとにロック、または論理的に独立したデータ構造ごとにロック

---

## 12. 実践演習

### 演習1: [基礎] デバイスの観察と情報収集

**目標**: Linuxシステム上のデバイスとドライバの関係を理解する。

```bash
# === Step 1: デバイスファイルの種類を確認 ===
# 'b' = ブロック、'c' = キャラクタ
ls -la /dev/sda /dev/null /dev/tty0 /dev/random 2>/dev/null

# メジャー番号・マイナー番号の読み方
# crw-rw-rw- 1 root root 1, 3 ... /dev/null
#                         ^  ^
#                    major=1 minor=3

# === Step 2: ブロックデバイスの階層構造 ===
lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,MODEL

# === Step 3: PCIデバイスとドライバのマッピング ===
lspci -v | head -40
# "Kernel driver in use:" でどのドライバが使われているか確認

# === Step 4: ロード済みカーネルモジュールの調査 ===
lsmod | sort -k3 -rn | head -20
# 3列目(Used by)でソート → 依存関係が多いモジュールが上位に

# === Step 5: 割り込みの分布を確認 ===
cat /proc/interrupts | head -20
# 各CPU (CPU0, CPU1, ...) ごとの割り込み回数が表示される

# === Step 6: カーネルメッセージからドライバの動作を追跡 ===
dmesg | grep -i "driver\|probe\|loaded" | tail -20
```

**課題**: 上記コマンドの出力結果を基に、以下の表を埋めよ。

| デバイス名 | デバイスタイプ | メジャー番号 | 使用ドライバ |
|:-----------|:-------------|:------------|:------------|
| /dev/sda | ブロック | ? | ? |
| /dev/null | キャラクタ | ? | ? |
| (NIC名) | ネットワーク | — | ? |

### 演習2: [中級] カーネルモジュールのビルドとロード

**目標**: 最小限のカーネルモジュールをビルドし、ロード・アンロードのライフサイクルを体験する。

**前提条件**: Linux環境（仮想マシン推奨）、`build-essential`と`linux-headers-$(uname -r)`がインストール済みであること。

```bash
# === Step 1: ヘッダパッケージの確認 ===
ls /lib/modules/$(uname -r)/build/
# Makefile, include/ 等が存在すれば OK

# === Step 2: モジュールソースの作成 ===
mkdir -p ~/driver_lab && cd ~/driver_lab

# hello_driver.c を作成（本章 2.2 のコードを使用）
# Makefile を作成（本章 2.2 のMakefileを使用）

# === Step 3: ビルド ===
make
# 成功すると hello_driver.ko が生成される

# === Step 4: モジュール情報の確認 ===
modinfo hello_driver.ko

# === Step 5: ロードとログ確認 ===
sudo insmod hello_driver.ko
dmesg | tail -5
lsmod | grep hello

# === Step 6: アンロードとログ確認 ===
sudo rmmod hello_driver
dmesg | tail -5

# === Step 7: パラメータ付きモジュールに拡張 ===
# hello_driver.c に以下を追加してみよう:
#   static int repeat = 1;
#   module_param(repeat, int, 0644);
#   MODULE_PARM_DESC(repeat, "Number of greeting repetitions");
# init関数内で repeat 回ループして pr_info を出力する
```

**発展課題**: パラメータ`repeat`の値を`/sys/module/hello_driver/parameters/repeat`から読み取れることを確認し、実行時に値を変更して動作が変わることを検証せよ。

### 演習3: [上級] キャラクタデバイスドライバの実装と検証

**目標**: 本章 4.1 のsimplecharドライバをビルドし、ユーザ空間からの読み書きを検証する。

```bash
# === Step 1: ドライバのビルドとロード ===
cd ~/driver_lab
# simplechar.c を作成（本章 4.1 のコードを使用）
# Makefile の obj-m 行を修正: obj-m += simplechar.o
make
sudo insmod simplechar.ko

# === Step 2: デバイスノードの確認 ===
ls -la /dev/simplechar
# crw------- 1 root root 240, 0 ... /dev/simplechar
# (メジャー番号は動的に割り当てられるため異なる場合がある)

# パーミッション変更（テスト用）
sudo chmod 666 /dev/simplechar

# === Step 3: 書き込みテスト ===
echo "Hello, kernel driver!" > /dev/simplechar
dmesg | tail -3

# === Step 4: 読み取りテスト ===
cat /dev/simplechar
# "Hello, kernel driver!" が表示されるはず

# === Step 5: ddコマンドによるオフセット付き読み取り ===
dd if=/dev/simplechar bs=1 skip=7 count=6 2>/dev/null
# "kernel" が表示される

# === Step 6: 複数プロセスからの同時アクセステスト ===
# ターミナル1:
while true; do echo "Writer1: $(date)" > /dev/simplechar; done &
# ターミナル2:
while true; do cat /dev/simplechar; done &
# mutexによる排他制御が正しく機能していることを確認

# === Step 7: クリーンアップ ===
sudo rmmod simplechar
dmesg | tail -5
```

**発展課題**:
1. `unlocked_ioctl`を追加し、バッファのクリア機能（`ioctl(fd, SIMPLECHAR_CLEAR, 0)`）を実装せよ
2. `poll`を追加し、データが書き込まれたときに`select()`/`poll()`で検出できるようにせよ
3. `/sys/class/simple/simplechar/buffer_usage` 属性を追加し、現在のバッファ使用率をパーセントで表示せよ

---

## 13. OS別ドライバモデルの比較

| 特性 | Linux | Windows (WDM/WDF) | macOS (IOKit/DriverKit) | FreeBSD |
|:-----|:------|:-------------------|:------------------------|:--------|
| ドライバ言語 | C（Rustも段階的に導入中） | C/C++ | C++/Swift（DriverKit） | C |
| ロード単位 | カーネルモジュール (.ko) | ドライバパッケージ (.sys) | kext / dext | カーネルモジュール (.ko) |
| デバイス記述 | デバイスツリー / ACPI | INFファイル | IOKitマッチング | hints / FDT |
| ユーザ空間ドライバ | UIO / VFIO | UMDF | DriverKit (dext) | なし（標準） |
| ドライバ署名 | 任意（Secure Boot時は必須） | 必須（WHQL推奨） | 必須（公証） | 任意 |
| ホットプラグ | udev + uevent | PnPマネージャ | IOKit matching | devd |
| 電源管理 | ランタイムPM + ACPI | WDF電源ポリシー | IOPMPowerState | ACPI |
| デバッグ | printk, ftrace, kgdb | WinDbg, Driver Verifier | lldb, IOKitDebug | kgdb, DTrace |

---

## 14. よくある質問（FAQ）

### Q1: カーネルモジュールとカーネル組み込みドライバの違いは何か？

**A1**: 機能的には同等である。違いはロードのタイミングと方法にある。

- **カーネル組み込み（built-in）**: カーネルイメージ（vmlinuz）自体にコンパイルされ、ブート時に自動的に利用可能になる。カーネルの `.config` で `CONFIG_XXX=y` と設定する。ルートファイルシステムのマウントに必要なドライバ（ストレージコントローラ、ファイルシステム）は通常組み込みにする必要がある（initramfsを使わない場合）。
- **カーネルモジュール（loadable）**: `.ko` ファイルとして `/lib/modules/` 以下に配置され、`modprobe` や udev により動的にロードされる。`.config` で `CONFIG_XXX=m` と設定する。使わないデバイスのドライバはメモリを消費しないという利点がある。

多くのディストリビューションでは、可能な限り多くのドライバをモジュールとしてビルドし、initramfs内にブートに必要な最小限のモジュールを含める方式を採用している。

### Q2: デバイスドライバのバグでシステム全体がクラッシュするのはなぜか？

**A2**: 従来のカーネルドライバは、カーネルと同じアドレス空間・同じ特権レベルで動作するため、メモリ保護の恩恵を受けられない。具体的には以下の理由による。

- **NULLポインタ参照**: カーネル空間でのNULLポインタ参照はページフォルトを起こし、回復不能なoopsまたはカーネルパニックとなる
- **バッファオーバーフロー**: カーネルの重要なデータ構造を破壊する可能性がある
- **デッドロック**: カーネルスレッドや割り込みハンドラが永久にブロックされると、システム全体が応答不能になる
- **不正なメモリ解放**: use-after-freeやdouble-freeはカーネルのメモリアロケータを破壊する

この問題を軽減するため、以下のアプローチが採られている。
1. ユーザ空間ドライバ（UIO, VFIO, DriverKit）による隔離
2. eBPFによる安全なカーネル拡張
3. Rustによるメモリ安全なドライバ実装（Linux 6.1以降）
4. マイクロカーネルアーキテクチャ（MINIX 3, seL4）

### Q3: 新しいハードウェアを接続してもドライバが見つからない場合の対処法は？

**A3**: 以下の手順で調査と対処を行う。

1. **デバイスの認識確認**: `lspci -nn`（PCI）や `lsusb -v`（USB）でベンダーID・プロダクトIDを確認する
2. **カーネルログの確認**: `dmesg | tail -30` でエラーメッセージや未対応デバイスの警告を確認する
3. **対応ドライバの検索**: ベンダーID:プロダクトID（例: `8086:1533`）でカーネルソースを検索し、対応するドライバモジュール名を特定する
4. **モジュールの手動ロード**: `sudo modprobe <module_name>` を試みる
5. **カーネルバージョンの確認**: 新しいデバイスは最新のカーネルでのみサポートされている場合がある。`uname -r` で確認し、必要なら新しいカーネルにアップデートする
6. **OOT（Out-of-Tree）ドライバの導入**: ベンダー提供のドライバやDKMS経由のサードパーティドライバを検討する
7. **ファームウェアの確認**: 一部のデバイスは `linux-firmware` パッケージに含まれるファームウェアバイナリが必要である。`/lib/firmware/` 以下にファームウェアが存在するか確認する

### Q4: Linuxカーネルへのドライバ取り込み（メインライン化）のメリットは何か？

**A4**: カーネルのメインラインツリーにドライバが取り込まれると、以下のメリットがある。

- **継続的なメンテナンス**: カーネルのAPI変更に伴う修正がコミュニティによって行われる
- **広範なテスト**: CI/CDシステムと多数のテスターによる品質保証
- **ディストリビューション同梱**: 主要ディストリビューションのカーネルパッケージに含まれ、ユーザが手動でドライバをインストールする必要がなくなる
- **セキュリティ修正**: 脆弱性が発見された場合、カーネルセキュリティチームが迅速に対応する

### Q5: GPUドライバはなぜ特殊なのか？

**A5**: GPUドライバは、他のデバイスドライバと比較して以下の点で特殊かつ複雑である。

- **DRM/KMSサブシステム**: ディスプレイ出力の制御（モード設定、CRTC、エンコーダ、コネクタの管理）を担当するカーネル側フレームワーク
- **ユーザ空間コンポーネント**: Mesa（OpenGL/Vulkan実装）やlibdrm等の巨大なユーザ空間ライブラリと密接に連携する
- **メモリ管理の複雑さ**: VRAM（ビデオメモリ）の管理、GEM/TTMによるバッファオブジェクト管理、CPUとGPU間のメモリコヒーレンシ制御
- **コマンド投入**: GPUのコマンドキューへのジョブ投入と完了待ちのスケジューリング
- **ベンダー固有の複雑さ**: NVIDIA（プロプライエタリ + Nouveau）、AMD（amdgpu）、Intel（i915/xe）でアーキテクチャが大きく異なる

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## まとめ

| 概念 | ポイント |
|:-----|:--------|
| デバイスドライバの役割 | ハードウェア固有の操作をOS統一APIに変換する通訳者 |
| キャラクタデバイス | バイトストリーム。file_operations経由で逐次アクセス |
| ブロックデバイス | ブロック単位。I/Oスケジューラ、ページキャッシュ経由 |
| ネットワークデバイス | パケット単位。ソケットAPI経由。/devに現れない |
| カーネルモジュール | 動的ロード可能。module_init/module_exit |
| ポーリング | CPU常時監視。単純だが非効率。組込み向き |
| 割り込み駆動 | イベント通知。CPU効率が良い。一般的な方式 |
| DMA | CPU不関与の直接メモリ転送。大量データに最適 |
| トップハーフ/ボトムハーフ | 割り込み処理の分割。重い処理を遅延実行 |
| threaded IRQ | 現代的推奨方式。プロセスコンテキストでスリープ可能 |
| UIO/VFIO | ユーザ空間ドライバ。安全性・開発容易性の向上 |
| devm_* API | デバイス管理リソース。自動解放でリーク防止 |
| デバイスツリー | ハードウェア記述。ARM/RISC-Vで標準 |
| ACPI | x86のハードウェア記述・電源管理標準規格 |
| ランタイムPM | デバイス単位の動的電源管理 |

---

## 次に読むべきガイド


---

## 参考文献

1. Corbet, J., Rubini, A., Kroah-Hartman, G. "Linux Device Drivers." 3rd Edition, O'Reilly Media, 2005. — Linuxドライバ開発の定番書。カーネルAPIの変更により一部古くなっているが、設計思想と基本概念は今なお有効。オンライン版: https://lwn.net/Kernel/LDD3/

2. Love, R. "Linux Kernel Development." 3rd Edition, Addison-Wesley, 2010. — カーネル内部の仕組みを包括的に解説。プロセス管理、メモリ管理、割り込み処理、同期機構などドライバ開発の前提知識が網羅されている。

3. The Linux Kernel Documentation — "Driver Model." https://www.kernel.org/doc/html/latest/driver-api/index.html — カーネル公式のドライバAPI文書。最新のAPIリファレンスとして最も信頼性が高い。デバイスモデル、DMAマッピング、割り込み処理、電源管理等の公式ガイドラインを含む。

4. Kroah-Hartman, G. "Linux Kernel in a Nutshell." O'Reilly Media, 2006. — カーネルの設定、ビルド、モジュール管理に焦点を当てた実践的ガイド。

5. Venkateswaran, S. "Essential Linux Device Drivers." Prentice Hall, 2008. — キャラクタドライバ、ブロックドライバ、ネットワークドライバ、USBドライバ等の実装を体系的に解説。

6. Mauerer, W. "Professional Linux Kernel Architecture." Wiley, 2008. — カーネルアーキテクチャの詳細な内部解説。仮想メモリ、プロセススケジューラ、VFS、ネットワークスタック等の実装を深く掘り下げている。


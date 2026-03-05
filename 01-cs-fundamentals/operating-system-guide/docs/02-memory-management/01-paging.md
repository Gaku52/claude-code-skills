# ページング ── 仮想メモリ・ページテーブル・TLB・ページ置換・スワッピング

> **ページング (Paging)** は、物理メモリを固定サイズの「フレーム」に、仮想アドレス空間を同じサイズの「ページ」に分割し、ページテーブルを介して両者を動的に対応付けるメモリ管理方式である。現代のほぼ全ての汎用 OS はページングを基盤としており、プロセス分離・共有メモリ・デマンドページング・スワッピングといった機構を支えている。

## この章で学ぶこと

- [ ] 仮想アドレスから物理アドレスへの変換過程を図示できる
- [ ] 単一レベル／多階層ページテーブルの構造とトレードオフを説明できる
- [ ] TLB（Translation Lookaside Buffer）の役割とミス時のペナルティを定量的に議論できる
- [ ] LRU・Clock・LFU 等のページ置換アルゴリズムを比較し、実装できる
- [ ] スワッピングとデマンドページングの関係を理解し、Linux のメモリ管理と結び付けられる
- [ ] ページサイズの選択がシステム性能に与える影響を分析できる

---

## 1. なぜページングが必要なのか

### 1.1 セグメンテーションの限界

プログラムを論理的な単位（コード・データ・スタック・ヒープ）に分割するセグメンテーションは、直感的なメモリ管理を提供するが、**可変長**であるが故に **外部断片化 (External Fragmentation)** を引き起こす。

```
外部断片化の例:

物理メモリ (100KB):
┌────────┐
│ A: 20KB│  ← プロセスA
├────────┤
│空: 10KB│  ← Aが解放された残り？ いいえ、別の隙間
├────────┤
│ B: 30KB│  ← プロセスB
├────────┤
│空: 15KB│  ← Cが解放された跡
├────────┤
│ D: 25KB│  ← プロセスD
└────────┘

合計空き = 10KB + 15KB = 25KB
しかし、連続 25KB の領域は存在しない。
→ 25KB のプロセスE を配置できない！

解決策1: コンパクション（メモリの再配置）
  → 全プロセスを停止してメモリを移動する必要があり、非常にコストが高い

解決策2: ページング（固定サイズ分割）
  → 外部断片化を原理的に排除できる
```

### 1.2 ページングの基本思想

ページングでは、仮想アドレス空間を **ページ (Page)**、物理メモリを **フレーム (Frame)** という固定サイズの単位に分割する。一般的なページサイズは **4KB (4096 バイト = 2^12 バイト)** である。

```
ページングの基本構造:

仮想アドレス空間              物理メモリ
┌──────────┐               ┌──────────┐
│ Page 0   │──────────────→│ Frame 5  │
├──────────┤               ├──────────┤
│ Page 1   │──────┐        │ Frame 1  │
├──────────┤      │        ├──────────┤
│ Page 2   │──┐   │        │ Frame 2  │←─┐
├──────────┤  │   │        ├──────────┤  │
│ Page 3   │  │   └───────→│ Frame 3  │  │
├──────────┤  │            ├──────────┤  │
│ Page 4   │  │            │ Frame 4  │  │
├──────────┤  └───────────→│ Frame 6  │  │
│ Page 5   │───────────────┼──────────┘  │
├──────────┤               │ Frame 7  │  │
│  ...     │               │  ...     │  │
│ Page N   │───────────────┼─────────────┘
└──────────┘               └──────────┘

  仮想ページ → 物理フレーム の対応は
  「ページテーブル」が保持する。

  連続した仮想ページが、物理的に連続している必要はない
  → 外部断片化が発生しない
```

**なぜ 4KB なのか:**

| 判断基準 | 小さいページ (例: 512B) | 大きいページ (例: 64KB) |
|---------|----------------------|----------------------|
| 内部断片化 | 平均 256B（小さい） | 平均 32KB（大きい） |
| ページテーブルサイズ | 巨大になる | 小さくて済む |
| ディスク I/O 効率 | 非効率（小さな転送が多い） | 効率的（まとめて転送） |
| メモリ利用効率 | 高い（無駄が少ない） | 低い（未使用部分が多い） |
| TLB カバレッジ | 狭い | 広い |

4KB は内部断片化の小ささとページテーブルサイズ・I/O 効率のバランスが取れた値として、1990 年代から標準的に採用されている。ただし、現代の大容量メモリ環境では **Huge Pages (2MB / 1GB)** も広く利用される。

---

## 2. 仮想アドレスから物理アドレスへの変換

### 2.1 アドレス変換の仕組み

仮想アドレスは **ページ番号 (VPN: Virtual Page Number)** と **オフセット (Offset)** に分解される。

```
32ビット仮想アドレス（ページサイズ = 4KB = 2^12）:

  31                    12 11                0
  ┌───────────────────────┬─────────────────┐
  │  VPN (20ビット)        │ Offset (12ビット)│
  └───────────────────────┴─────────────────┘
       2^20 = 1,048,576 ページ     4096 バイト

変換手順:
  1. 仮想アドレスから VPN を抽出: VPN = VA >> 12
  2. ページテーブルで VPN → PFN (Physical Frame Number) に変換
  3. 物理アドレスを構成: PA = (PFN << 12) | Offset

具体例: 仮想アドレス 0x00403A7C
  ┌───────────────────────┬─────────────────┐
  │  VPN = 0x00403        │ Offset = 0xA7C  │
  └───────────────────────┴─────────────────┘
  ページテーブルで VPN 0x00403 → PFN 0x0007B と対応
  物理アドレス = 0x0007BA7C
  ┌───────────────────────┬─────────────────┐
  │  PFN = 0x0007B        │ Offset = 0xA7C  │
  └───────────────────────┴─────────────────┘
```

### 2.2 ページテーブルエントリ (PTE)

各ページテーブルエントリは、フレーム番号だけでなく、保護・状態に関する制御ビットを保持する。

```
x86 ページテーブルエントリ (32ビット):

  31              12 11  9  8   7   6   5   4   3   2   1   0
  ┌─────────────────┬─────┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
  │ PFN (20ビット)   │Avail│ G │PAT│ D │ A │PCD│PWT│U/S│R/W│ P │
  └─────────────────┴─────┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

  P   (Present)     : 1 = ページが物理メモリに存在
                      0 = スワップアウトされているか未割当
  R/W (Read/Write)  : 1 = 書き込み可, 0 = 読み取り専用
  U/S (User/Super)  : 1 = ユーザーモードアクセス可, 0 = カーネルのみ
  A   (Accessed)    : MMU がアクセス時に自動セット（Clock アルゴリズムで使用）
  D   (Dirty)       : 書き込み発生時に自動セット（スワップアウト時の判断に使用）
  G   (Global)      : コンテキストスイッチで TLB フラッシュ対象外
  PCD (Page Cache Disable) : キャッシュ無効化（MMIO 領域で使用）
  PWT (Page Write Through) : ライトスルーキャッシュ制御

なぜ Dirty ビットが重要なのか:
  → ページをスワップアウトする際、Dirty = 0 ならディスクに書き戻す必要がない
  → I/O を削減でき、ページ置換の効率が大幅に向上する
```

---

## 3. 多階層ページテーブル

### 3.1 単一レベルページテーブルの問題

32 ビットアドレス空間で 4KB ページの場合、ページテーブルは 2^20 = 約 100 万エントリを持つ。各エントリが 4 バイトなら、**1 プロセスあたり 4MB** のページテーブルが必要になる。しかし実際のプロセスが使用する仮想アドレス空間はごく一部であるため、大半のエントリは無効 (P=0) であり、メモリの無駄が極めて大きい。

### 3.2 2 階層ページテーブル (x86 32ビット)

```
2階層ページテーブル (x86 32ビット):

仮想アドレス (32ビット):
  31          22 21          12 11           0
  ┌─────────────┬──────────────┬─────────────┐
  │ Dir (10ビット)│ Table(10ビット)│Offset(12ビット)│
  └──────┬──────┴──────┬───────┴─────────────┘
         │             │
         ▼             │
  ┌────────────┐       │
  │ ページ      │       │
  │ ディレクトリ │       │
  │ (1024個)    │       │
  │ ┌────────┐ │       │
  │ │ Entry0 │ │       │
  │ ├────────┤ │       │
  │ │ Entry1 │─┼───┐   │
  │ ├────────┤ │   │   │
  │ │  ...   │ │   │   │
  │ └────────┘ │   │   │
  └────────────┘   │   │
         CR3       ▼   │
                ┌────────────┐
                │ ページ      │
                │ テーブル    │
                │ (1024個)    │
                │ ┌────────┐ │    ┌──────────┐
                │ │ Entry0 │ │    │ 物理     │
                │ ├────────┤ │    │ フレーム │
                │ │ Entry1 │─┼───→│ (4KB)   │
                │ ├────────┤ │    │          │
                │ │  ...   │ │    └──────────┘
                │ └────────┘ │
                └────────────┘

メモリ節約の仕組み:
  使われていない仮想アドレス領域に対応するページテーブルは
  割り当てない（ページディレクトリのエントリを無効にする）

  例: プロセスが 8MB しか使わない場合
    ページディレクトリ: 4KB (常に必要)
    ページテーブル: 4KB × 2 = 8KB (8MB / 4MB per table)
    合計: 12KB  ← 単一レベルの 4MB と比べて劇的に節約
```

### 3.3 4 階層ページテーブル (x86-64)

64 ビット環境では仮想アドレス空間が広大になるため、4 階層のページテーブルが使用される。ただし、実際に使用されるのは 48 ビット（256TB）のみである。

```
4階層ページテーブル (x86-64, 48ビット仮想アドレス):

仮想アドレス (64ビット):
  63    48 47    39 38    30 29    21 20    12 11     0
  ┌──────┬────────┬────────┬────────┬────────┬────────┐
  │符号拡張│ PML4   │  PDPT  │  PD    │  PT    │Offset  │
  │(16bit)│ (9bit) │ (9bit) │ (9bit) │ (9bit) │(12bit) │
  └──────┴───┬────┴───┬────┴───┬────┴───┬────┴────────┘
             │        │        │        │
             ▼        ▼        ▼        ▼
  CR3 → [PML4] → [PDPT] → [PD] → [PT] → 物理フレーム
         512個    512個    512個   512個

  各テーブルは 512 エントリ × 8 バイト = 4KB（1ページに収まる）

  アドレスカバレッジ:
    1 PTE       = 4KB
    1 PT        = 512 × 4KB    = 2MB
    1 PD        = 512 × 2MB    = 1GB
    1 PDPT      = 512 × 1GB    = 512GB
    1 PML4      = 512 × 512GB  = 256TB

  なぜ 48ビット で十分なのか:
    256TB の仮想アドレス空間は現在の実用上十分であり、
    ページテーブルの階層を深くしすぎるとアドレス変換のオーバーヘッドが増大する。
    Intel は 5階層 (57ビット, LA57) も定義済みで、128PB まで拡張可能。
```

### 3.4 多階層ページテーブルの比較

| 特性 | 1階層 | 2階層 (x86-32) | 4階層 (x86-64) |
|------|-------|----------------|----------------|
| 仮想アドレス幅 | 32bit | 32bit | 48bit |
| テーブルエントリ数 | 2^20 | 2^10 × 2^10 | 4段 × 2^9 |
| 最小メモリ消費 | 4MB 固定 | ~12KB | ~16KB |
| 最大メモリ消費 | 4MB 固定 | 4MB + 4KB | 理論上巨大 |
| アドレス変換のメモリ参照回数 | 1回 | 2回 | 4回 |
| 空きページの扱い | 全エントリ保持 | テーブル省略可 | テーブル省略可 |
| 採用 OS 例 | 教育用 OS | Windows XP (32bit) | Linux, Windows 10/11 |

---

## 4. TLB (Translation Lookaside Buffer)

### 4.1 なぜ TLB が必要なのか

4 階層ページテーブルでは、1 回のメモリアクセスに対して **4 回のページテーブル参照 + 1 回のデータアクセス = 計 5 回**のメモリアクセスが必要になる。これでは性能が 1/5 に低下してしまう。TLB は、最近のアドレス変換結果をキャッシュする高速連想メモリ（CAM: Content-Addressable Memory）であり、この問題を解決する。

```
TLB によるアドレス変換の高速化:

                    ┌─────────┐
仮想アドレス ───────→│  TLB    │
     │              │ (高速)  │
     │              │VPN→PFN  │
     │              └────┬────┘
     │                   │
     │          ┌────────┴────────┐
     │          │                 │
     │       TLB Hit          TLB Miss
     │     (1サイクル)       (数十〜数百サイクル)
     │          │                 │
     │          ▼                 ▼
     │    物理アドレス      ┌──────────┐
     │    を即座に取得      │ページ     │
     │                     │テーブル   │
     │                     │ウォーク   │
     │                     │(4回参照)  │
     │                     └────┬─────┘
     │                          │
     │                          ▼
     │                    TLBに結果を登録
     │                    + 物理アドレス取得
     ▼
  メモリアクセス

TLB のヒット率が 99% の場合の実効アクセス時間:
  TLB hit  = 1ns (TLB参照) + 100ns (メモリアクセス) = 101ns
  TLB miss = 1ns + 4×100ns (ページウォーク) + 100ns = 501ns
  実効時間 = 0.99 × 101 + 0.01 × 501 = 99.99 + 5.01 = 105ns
  オーバーヘッド = (105 - 100) / 100 = 5%

  → 99% のヒット率があれば、オーバーヘッドはわずか 5%
  → ヒット率が 90% に下がると: 0.9×101 + 0.1×501 = 141ns → 41% 増
```

### 4.2 TLB の構造

```
TLB エントリ:
┌───────┬──────┬───┬───┬───┬───┬─────┐
│  VPN  │  PFN │ V │ D │ G │ASID│Prot │
└───────┴──────┴───┴───┴───┴───┴─────┘

  VPN  : 仮想ページ番号（検索キー）
  PFN  : 物理フレーム番号（検索結果）
  V    : Valid ビット（このエントリが有効か）
  D    : Dirty ビット（書き込みがあったか）
  G    : Global ビット（全プロセス共有、コンテキストスイッチで保持）
  ASID : Address Space Identifier（プロセス識別子）
         → ASID により、コンテキストスイッチ時の TLB フラッシュを回避
  Prot : 保護ビット（読み/書き/実行）

典型的な TLB サイズ:
  L1 ITLB (命令用) : 64〜128 エントリ, 4-way セットアソシアティブ
  L1 DTLB (データ用): 64〜72 エントリ, 4-way セットアソシアティブ
  L2 STLB (統合)   : 1024〜2048 エントリ, 8-12-way

  なぜ TLB はこんなに小さいのか:
    TLB は全エントリを並列検索する CAM で構成されており、
    エントリ数を増やすと消費電力と面積が急増し、
    検索速度も低下する。小さくても高いヒット率を維持できるのは、
    プログラムの「局所性 (locality)」のおかげである。
```

### 4.3 ASID (Address Space Identifier)

コンテキストスイッチが発生すると、新しいプロセスのページテーブルは異なる VPN → PFN 対応を持つ。ASID がない場合、全 TLB エントリをフラッシュ（無効化）する必要があるが、ASID を使えばプロセスごとのエントリを区別でき、フラッシュを回避できる。

```
ASID によるコンテキストスイッチの最適化:

ASID なし:
  プロセスA実行 → TLB: [VPN=0x100→PFN=0x5, VPN=0x200→PFN=0x8, ...]
  コンテキストスイッチ → TLB 全フラッシュ（全エントリ無効化）
  プロセスB実行 → TLB コールドスタート（全て miss）

ASID あり:
  プロセスA (ASID=1) 実行 → TLB: [(ASID=1,VPN=0x100)→PFN=0x5, ...]
  コンテキストスイッチ → TLB フラッシュ不要
  プロセスB (ASID=2) 実行 → TLB: [(ASID=2,VPN=0x100)→PFN=0x3, ...]
  プロセスA に戻る → ASID=1 のエントリがまだ残っている可能性
  → TLB hit が期待でき、性能低下を抑制
```

---

## 5. デマンドページングと仮想メモリ

### 5.1 デマンドページングの仕組み

デマンドページングでは、プロセス起動時にページを物理メモリにロードせず、実際にアクセスされた時点で初めてロードする。これにより、起動時間の短縮とメモリ使用量の削減が実現される。

```
デマンドページングの流れ:

  1. プロセス起動: 全ページが「無効 (P=0)」のページテーブルを作成
  2. CPU が仮想アドレスにアクセス
  3. MMU がページテーブルを参照 → P=0 → ページフォルト例外を発生
  4. OS のページフォルトハンドラが起動:
     a. アクセスが正当か確認（セグメンテーション違反なら SIGSEGV）
     b. 空き物理フレームを確保（なければページ置換を実行）
     c. ディスクからページの内容を読み込み
     d. ページテーブルを更新: PFN を設定し P=1
     e. TLB を更新
  5. 中断された命令を再実行 → 今度はページが存在するので正常にアクセス

  ┌─────────┐     ページフォルト      ┌──────────┐
  │   CPU   │ ───────────────────→  │    OS    │
  │         │                       │ハンドラ   │
  │ 命令再開│ ←──────────────────── │          │
  └─────────┘   ページテーブル更新    └────┬─────┘
                                         │
                                         ▼
                                    ┌──────────┐
                                    │ ディスク  │
                                    │ (スワップ │
                                    │  領域)    │
                                    └──────────┘
```

### 5.2 ページフォルトの種類

| 種類 | 原因 | OS の対応 | コスト |
|------|------|----------|--------|
| Minor (Soft) | ページは物理メモリにあるがテーブルが未設定 | PTE を設定するだけ | 数マイクロ秒 |
| Major (Hard) | ページがディスク上にある | ディスクから読み込み | 数ミリ秒 (SSD) 〜数十ミリ秒 (HDD) |
| Invalid | 不正なアドレスへのアクセス | プロセスに SIGSEGV を送信 | プロセス終了 |

**ページフォルトのコスト分析:**

```
Major ページフォルトのコスト想定値:
  SSD ランダムリード: ~100μs = 100,000ns
  HDD ランダムリード: ~10ms  = 10,000,000ns
  メモリアクセス     : ~100ns

  HDD 上のスワップでの実効メモリアクセス時間:
    ページフォルト率 p とすると:
    EAT = (1-p) × 100ns + p × 10,000,000ns

    p = 1/1000 (0.1%) の場合:
    EAT = 0.999 × 100 + 0.001 × 10,000,000 = 99.9 + 10,000 = 10,099.9ns
    → メモリアクセスが 100 倍遅くなる

    性能低下を 10% 以内に抑えるには:
    110 > (1-p)×100 + p×10,000,000
    10 > p × 9,999,900
    p < 0.000001 = 0.0001%
    → 100万回に 1回以下のページフォルトが必要

  これがページ置換アルゴリズムの性能が極めて重要である理由。
```

### 5.3 Copy-on-Write (COW)

`fork()` システムコールはプロセスの完全なコピーを作成するが、全ページをコピーすると非常にコストが高い。COW は、`fork()` 直後は親子で同じ物理ページを共有し、どちらかが書き込みを行った時点で初めてコピーを作成する。

```
Copy-on-Write の仕組み:

fork() 直後:
  親プロセス          子プロセス
  ページテーブル       ページテーブル
  ┌─────────┐        ┌─────────┐
  │VPN 0→F3 │        │VPN 0→F3 │  ← 同じフレームを共有
  │VPN 1→F7 │        │VPN 1→F7 │  ← R/W → Read-Only に変更
  │VPN 2→F1 │        │VPN 2→F1 │
  └─────────┘        └─────────┘
                物理メモリ
              ┌──────────┐
         F1   │ 共有データ │
         F3   │ 共有データ │
         F7   │ 共有データ │
              └──────────┘

子プロセスが VPN 1 に書き込み:
  1. ページフォルト発生（Read-Only ページへの書き込み）
  2. OS が COW と判断
  3. 新しいフレーム F9 を割り当て
  4. F7 の内容を F9 にコピー
  5. 子プロセスの PTE を VPN 1→F9 (R/W) に更新
  6. 親プロセスの PTE を VPN 1→F7 (R/W) に戻す（参照カウント=1）

  親プロセス          子プロセス
  ┌─────────┐        ┌─────────┐
  │VPN 0→F3 │        │VPN 0→F3 │  ← まだ共有
  │VPN 1→F7 │        │VPN 1→F9 │  ← 分離完了
  │VPN 2→F1 │        │VPN 2→F1 │  ← まだ共有
  └─────────┘        └─────────┘
```

---

## 6. ページ置換アルゴリズム

物理メモリが満杯の状態で新しいページを読み込む必要がある場合、既存のページを追い出す（evict する）必要がある。どのページを追い出すかを決定するのがページ置換アルゴリズムである。

### 6.1 最適アルゴリズム (OPT / Belady's Algorithm)

将来最も長い間参照されないページを置換する。**理論上最適**だが、将来のアクセスパターンは予測不能なため実装不可能。他のアルゴリズムの性能評価のベースラインとして使用される。

### 6.2 FIFO (First-In, First-Out)

最も古くロードされたページを置換する。実装が単純だが、頻繁に使用されるページも追い出してしまう可能性がある。また、**Belady の異常 (Belady's Anomaly)** が発生し得る：フレーム数を増やしたにもかかわらずページフォルトが増加する現象。

### 6.3 LRU (Least Recently Used)

最も長い間参照されていないページを置換する。過去のアクセスパターンが将来を予測するという局所性の原理に基づく。OPT に近い性能を示すが、厳密な実装にはアクセス順序の正確な記録が必要でコストが高い。

### 6.4 Clock アルゴリズム (Second-Chance)

LRU を近似する実用的なアルゴリズム。フレームを円形リストに配置し、各フレームに参照ビット（Accessed ビット）を持たせる。

```
Clock アルゴリズムの動作:

  フレームを円形に配置（時計の針が巡回）:

          針
          ↓
    ┌───┐   ┌───┐
    │F0 │   │F1 │
    │A=1│   │A=0│  ← A=0 なので置換候補
    └───┘   └───┘
   /               \
  ┌───┐           ┌───┐
  │F5 │           │F2 │
  │A=1│           │A=1│
  └───┘           └───┘
   \               /
    ┌───┐   ┌───┐
    │F4 │   │F3 │
    │A=0│   │A=1│
    └───┘   └───┘

  置換が必要になった場合:
  1. 針の位置のフレームを確認
  2. A=1 なら A=0 にセットし、針を次に進める（Second Chance を与える）
  3. A=0 なら、そのフレームを置換対象とする
  4. 針を次の位置に進める

  上の例では:
    針→F0 (A=1): A=0にして進む
    針→F1 (A=0): ★ F1 を置換！

  Enhanced Clock (NRU: Not Recently Used):
    (A, D) の組み合わせで 4 クラスに分類:
    (0,0): 最近参照されず、変更なし → 最優先で置換
    (0,1): 最近参照されないが、変更あり → 書き戻しが必要
    (1,0): 最近参照されたが、変更なし
    (1,1): 最近参照され、変更あり → 最後に置換
```

### 6.5 アルゴリズム比較表

| アルゴリズム | ページフォルト率 | 実装コスト | Belady の異常 | 実用性 |
|------------|----------------|-----------|--------------|--------|
| OPT | 最小（理論最適） | 実装不可能 | なし | ベンチマーク用 |
| FIFO | 高い | 非常に低い | あり | 単純なシステム |
| LRU | OPT に近い | 高い（完全実装） | なし | 概念的に重要 |
| Clock | LRU に近い | 低い | なし | Linux/BSD で採用 |
| LFU | 場合による | 中程度 | なし | 特定ワークロード |
| LRU-K | 非常に低い | 中〜高 | なし | データベース (PostgreSQL) |

---

## 7. コード例

### コード例 1: 仮想アドレスの分解と変換シミュレーション (C)

```c
/*
 * virtual_address_translation.c
 *
 * 仮想アドレスを VPN とオフセットに分解し、
 * 簡易ページテーブルを用いて物理アドレスに変換するシミュレーション。
 *
 * コンパイル: gcc -Wall -o vat virtual_address_translation.c
 * 実行: ./vat
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PAGE_SIZE       4096        /* 4KB = 2^12 */
#define PAGE_SHIFT      12          /* log2(PAGE_SIZE) */
#define NUM_PAGES       (1 << 20)   /* 2^20 = 1M ページ (32bit) */
#define PT_SIZE         1024        /* シミュレーション用の小さなテーブル */

/* ページテーブルエントリの制御ビット */
#define PTE_PRESENT     (1 << 0)
#define PTE_WRITABLE    (1 << 1)
#define PTE_USER        (1 << 2)
#define PTE_ACCESSED    (1 << 3)
#define PTE_DIRTY       (1 << 4)

typedef struct {
    uint32_t entry;  /* PFN (上位20ビット) + フラグ (下位12ビット) */
} PageTableEntry;

typedef struct {
    PageTableEntry entries[PT_SIZE];
    int num_entries;
} PageTable;

/* PTE からフレーム番号を取得 */
static inline uint32_t pte_get_pfn(PageTableEntry pte) {
    return pte.entry >> PAGE_SHIFT;
}

/* PTE にフレーム番号とフラグを設定 */
static inline PageTableEntry pte_make(uint32_t pfn, uint32_t flags) {
    PageTableEntry pte;
    pte.entry = (pfn << PAGE_SHIFT) | (flags & 0xFFF);
    return pte;
}

/* PTE のフラグを確認 */
static inline int pte_is_present(PageTableEntry pte) {
    return pte.entry & PTE_PRESENT;
}

/* ページテーブルの初期化 */
void page_table_init(PageTable *pt) {
    memset(pt->entries, 0, sizeof(pt->entries));
    pt->num_entries = PT_SIZE;
}

/* マッピングの追加 */
void page_table_map(PageTable *pt, uint32_t vpn, uint32_t pfn, uint32_t flags) {
    if (vpn >= (uint32_t)pt->num_entries) {
        fprintf(stderr, "Error: VPN %u exceeds page table size\n", vpn);
        return;
    }
    pt->entries[vpn] = pte_make(pfn, flags | PTE_PRESENT);
    printf("  Mapped: VPN 0x%05X -> PFN 0x%05X (flags: ", vpn, pfn);
    if (flags & PTE_WRITABLE) printf("W ");
    if (flags & PTE_USER)     printf("U ");
    printf(")\n");
}

/* 仮想アドレスの変換 */
int translate_address(PageTable *pt, uint32_t virtual_addr,
                      uint32_t *physical_addr) {
    uint32_t vpn    = virtual_addr >> PAGE_SHIFT;
    uint32_t offset = virtual_addr & (PAGE_SIZE - 1);

    printf("\n--- Address Translation ---\n");
    printf("Virtual Address : 0x%08X\n", virtual_addr);
    printf("  VPN           : 0x%05X (page %u)\n", vpn, vpn);
    printf("  Offset        : 0x%03X (%u bytes)\n", offset, offset);

    if (vpn >= (uint32_t)pt->num_entries) {
        printf("  Result        : FAULT (VPN out of range)\n");
        return -1;
    }

    PageTableEntry pte = pt->entries[vpn];
    if (!pte_is_present(pte)) {
        printf("  Result        : PAGE FAULT (page not present)\n");
        return -1;
    }

    uint32_t pfn = pte_get_pfn(pte);
    *physical_addr = (pfn << PAGE_SHIFT) | offset;

    /* Accessed ビットをセット（ハードウェアが行う処理を模倣） */
    pt->entries[vpn].entry |= PTE_ACCESSED;

    printf("  PFN           : 0x%05X (frame %u)\n", pfn, pfn);
    printf("  Physical Addr : 0x%08X\n", *physical_addr);
    printf("  Result        : SUCCESS\n");
    return 0;
}

int main(void) {
    PageTable pt;
    page_table_init(&pt);

    printf("=== Page Table Setup ===\n");
    page_table_map(&pt, 0x00000, 0x00005, PTE_WRITABLE | PTE_USER);
    page_table_map(&pt, 0x00001, 0x00003, PTE_WRITABLE | PTE_USER);
    page_table_map(&pt, 0x00002, 0x0000B, PTE_USER);  /* 読み取り専用 */
    page_table_map(&pt, 0x00010, 0x0007B, PTE_WRITABLE | PTE_USER);

    uint32_t pa;

    /* 正常な変換 */
    translate_address(&pt, 0x00000A7C, &pa);  /* VPN=0, offset=0xA7C */
    translate_address(&pt, 0x00001500, &pa);  /* VPN=1, offset=0x500 */
    translate_address(&pt, 0x00010FF0, &pa);  /* VPN=0x10, offset=0xFF0 */

    /* ページフォルト: マッピングされていないページ */
    translate_address(&pt, 0x00003000, &pa);  /* VPN=3, マッピングなし */

    /* ページフォルト: 範囲外 */
    translate_address(&pt, 0xFFFFF000, &pa);

    return 0;
}
```

### コード例 2: LRU ページ置換シミュレーション (Python)

```python
"""
lru_page_replacement.py

LRU (Least Recently Used) ページ置換アルゴリズムのシミュレーション。
ページ参照列に対するページフォルト回数を計算し、
各ステップのフレーム状態を可視化する。

実行: python3 lru_page_replacement.py
"""

from collections import OrderedDict
from typing import List, Tuple


class LRUPageReplacer:
    """LRU ページ置換アルゴリズムの実装。

    OrderedDict を使用して、アクセス順序を効率的に管理する。
    最も最近アクセスされたページが末尾、最も古いページが先頭に位置する。
    """

    def __init__(self, num_frames: int):
        """
        Args:
            num_frames: 利用可能な物理フレーム数
        """
        if num_frames <= 0:
            raise ValueError("フレーム数は正の整数でなければならない")
        self.num_frames = num_frames
        self.frames: OrderedDict[int, bool] = OrderedDict()
        self.page_faults = 0
        self.history: List[Tuple[int, list, bool]] = []

    def access_page(self, page: int) -> bool:
        """ページにアクセスする。

        Args:
            page: アクセスするページ番号

        Returns:
            True: ページフォルトが発生した場合
            False: ページが既にフレーム内にあった場合 (ヒット)
        """
        fault = False

        if page in self.frames:
            # ヒット: ページを末尾（最新）に移動
            self.frames.move_to_end(page)
        else:
            # ミス: ページフォルト
            fault = True
            self.page_faults += 1

            if len(self.frames) >= self.num_frames:
                # フレームが満杯 → LRU ページ（先頭）を追い出す
                evicted_page, _ = self.frames.popitem(last=False)

            # 新しいページを末尾に追加
            self.frames[page] = True

        # 状態を履歴に記録
        self.history.append((page, list(self.frames.keys()), fault))
        return fault

    def simulate(self, reference_string: List[int]) -> int:
        """ページ参照列全体をシミュレーションする。

        Args:
            reference_string: ページ参照列

        Returns:
            総ページフォルト回数
        """
        for page in reference_string:
            self.access_page(page)
        return self.page_faults

    def print_trace(self) -> None:
        """シミュレーションのトレースを表示する。"""
        print(f"\n{'='*60}")
        print(f"LRU Page Replacement Simulation (Frames: {self.num_frames})")
        print(f"{'='*60}")
        print(f"{'Step':>4} | {'Page':>4} | {'Frames':<25} | {'Result'}")
        print(f"{'-'*4:>4}-+-{'-'*4:>4}-+-{'-'*25:<25}-+-{'-'*10}")

        for i, (page, frames, fault) in enumerate(self.history):
            frames_str = str(frames)
            result = "FAULT" if fault else "HIT"
            print(f"{i+1:>4} | {page:>4} | {frames_str:<25} | {result}")

        print(f"\nTotal page faults: {self.page_faults}")
        hit_count = len(self.history) - self.page_faults
        hit_rate = hit_count / len(self.history) * 100 if self.history else 0
        print(f"Hit rate: {hit_rate:.1f}%")


def compare_algorithms(reference_string: List[int],
                       num_frames: int) -> None:
    """FIFO と LRU を比較する。"""

    # --- FIFO ---
    from collections import deque
    fifo_frames: deque = deque()
    fifo_set: set = set()
    fifo_faults = 0
    for page in reference_string:
        if page not in fifo_set:
            fifo_faults += 1
            if len(fifo_frames) >= num_frames:
                old = fifo_frames.popleft()
                fifo_set.discard(old)
            fifo_frames.append(page)
            fifo_set.add(page)

    # --- LRU ---
    lru = LRUPageReplacer(num_frames)
    lru_faults = lru.simulate(reference_string)

    # --- OPT ---
    opt_faults = 0
    opt_frames: list = []
    for i, page in enumerate(reference_string):
        if page not in opt_frames:
            opt_faults += 1
            if len(opt_frames) >= num_frames:
                # 将来最も遅く参照されるページを探す
                farthest = -1
                victim = -1
                for f in opt_frames:
                    try:
                        next_use = reference_string[i+1:].index(f)
                    except ValueError:
                        next_use = float('inf')
                    if next_use > farthest:
                        farthest = next_use
                        victim = f
                opt_frames.remove(victim)
            opt_frames.append(page)

    print(f"\n{'='*50}")
    print(f"Algorithm Comparison (Frames: {num_frames})")
    print(f"Reference String: {reference_string}")
    print(f"{'='*50}")
    print(f"{'Algorithm':<12} | {'Page Faults':>12} | {'Hit Rate':>10}")
    print(f"{'-'*12}-+-{'-'*12:>12}-+-{'-'*10}")
    total = len(reference_string)
    print(f"{'OPT':<12} | {opt_faults:>12} | "
          f"{(total-opt_faults)/total*100:>9.1f}%")
    print(f"{'LRU':<12} | {lru_faults:>12} | "
          f"{(total-lru_faults)/total*100:>9.1f}%")
    print(f"{'FIFO':<12} | {fifo_faults:>12} | "
          f"{(total-fifo_faults)/total*100:>9.1f}%")


if __name__ == "__main__":
    # 教科書的な参照列
    ref_string = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]

    lru = LRUPageReplacer(num_frames=3)
    lru.simulate(ref_string)
    lru.print_trace()

    compare_algorithms(ref_string, num_frames=3)
    compare_algorithms(ref_string, num_frames=4)
```

### コード例 3: Clock アルゴリズムの実装 (C)

```c
/*
 * clock_algorithm.c
 *
 * Clock (Second-Chance) ページ置換アルゴリズムの実装。
 * 円形バッファとリファレンスビットを使用して LRU を近似する。
 *
 * コンパイル: gcc -Wall -o clock clock_algorithm.c
 * 実行: ./clock
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FRAMES 64

typedef struct {
    int page_number;    /* ページ番号 (-1 = 空) */
    int reference_bit;  /* 参照ビット (Accessed ビットに相当) */
    int dirty_bit;      /* 変更ビット */
} Frame;

typedef struct {
    Frame frames[MAX_FRAMES];
    int num_frames;
    int hand;           /* 時計の針 (次に検査するフレームのインデックス) */
    int used_frames;    /* 現在使用中のフレーム数 */
    int page_faults;
    int writes_back;    /* ディスクへの書き戻し回数 */
} ClockReplacer;

void clock_init(ClockReplacer *cr, int num_frames) {
    if (num_frames > MAX_FRAMES) {
        fprintf(stderr, "Error: num_frames exceeds MAX_FRAMES\n");
        exit(1);
    }
    cr->num_frames = num_frames;
    cr->hand = 0;
    cr->used_frames = 0;
    cr->page_faults = 0;
    cr->writes_back = 0;
    for (int i = 0; i < num_frames; i++) {
        cr->frames[i].page_number = -1;
        cr->frames[i].reference_bit = 0;
        cr->frames[i].dirty_bit = 0;
    }
}

/* フレーム内にページが存在するか検索 */
int clock_find_page(ClockReplacer *cr, int page) {
    for (int i = 0; i < cr->num_frames; i++) {
        if (cr->frames[i].page_number == page) {
            return i;
        }
    }
    return -1;
}

/* ページアクセス処理 */
int clock_access(ClockReplacer *cr, int page, int is_write) {
    int idx = clock_find_page(cr, page);

    if (idx >= 0) {
        /* ヒット: 参照ビットを 1 にセット */
        cr->frames[idx].reference_bit = 1;
        if (is_write) {
            cr->frames[idx].dirty_bit = 1;
        }
        return 0; /* ページフォルトなし */
    }

    /* ページフォルト */
    cr->page_faults++;

    if (cr->used_frames < cr->num_frames) {
        /* 空きフレームがある場合 */
        for (int i = 0; i < cr->num_frames; i++) {
            if (cr->frames[i].page_number == -1) {
                cr->frames[i].page_number = page;
                cr->frames[i].reference_bit = 1;
                cr->frames[i].dirty_bit = is_write ? 1 : 0;
                cr->used_frames++;
                return 1;
            }
        }
    }

    /* Clock アルゴリズムで置換対象を選択 */
    while (1) {
        if (cr->frames[cr->hand].reference_bit == 0) {
            /* 置換対象が見つかった */
            int evicted = cr->frames[cr->hand].page_number;
            if (cr->frames[cr->hand].dirty_bit) {
                cr->writes_back++;
                printf("    [Write-back] Page %d written to disk\n", evicted);
            }

            cr->frames[cr->hand].page_number = page;
            cr->frames[cr->hand].reference_bit = 1;
            cr->frames[cr->hand].dirty_bit = is_write ? 1 : 0;

            printf("    [Evict] Page %d replaced by Page %d at Frame %d\n",
                   evicted, page, cr->hand);

            cr->hand = (cr->hand + 1) % cr->num_frames;
            return 1;
        }
        /* Second Chance: 参照ビットをクリアして次へ */
        cr->frames[cr->hand].reference_bit = 0;
        cr->hand = (cr->hand + 1) % cr->num_frames;
    }
}

/* フレームの状態を表示 */
void clock_print_state(ClockReplacer *cr) {
    printf("  Frames: [");
    for (int i = 0; i < cr->num_frames; i++) {
        if (i > 0) printf(", ");
        if (cr->frames[i].page_number == -1) {
            printf("  -  ");
        } else {
            printf("P%d(%c%c)", cr->frames[i].page_number,
                   cr->frames[i].reference_bit ? 'R' : '-',
                   cr->frames[i].dirty_bit ? 'D' : '-');
        }
    }
    printf("]  Hand→%d\n", cr->hand);
}

int main(void) {
    ClockReplacer cr;
    clock_init(&cr, 4);

    /* ページ参照列: (ページ番号, 書き込みかどうか) */
    int refs[] =    {1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5};
    int writes[] =  {0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    int n = sizeof(refs) / sizeof(refs[0]);

    printf("=== Clock (Second-Chance) Algorithm ===\n");
    printf("Frames: 4, Reference String Length: %d\n\n", n);

    for (int i = 0; i < n; i++) {
        int fault = clock_access(&cr, refs[i], writes[i]);
        printf("Step %2d: Access Page %d (%s) -> %s\n",
               i + 1, refs[i],
               writes[i] ? "WRITE" : "READ",
               fault ? "FAULT" : "HIT");
        clock_print_state(&cr);
        printf("\n");
    }

    printf("=== Summary ===\n");
    printf("Total page faults : %d\n", cr.page_faults);
    printf("Total write-backs : %d\n", cr.writes_back);
    printf("Hit rate          : %.1f%%\n",
           (double)(n - cr.page_faults) / n * 100.0);

    return 0;
}
```

### コード例 4: Linux の mmap を使ったメモリマップトファイル (C)

```c
/*
 * mmap_demo.c
 *
 * mmap() を使ってファイルをメモリにマッピングし、
 * 通常のメモリアクセスでファイルを読み書きするデモ。
 * デマンドページングの恩恵を直接観察できる。
 *
 * コンパイル: gcc -Wall -o mmap_demo mmap_demo.c
 * 実行: ./mmap_demo
 *
 * 動作環境: Linux / macOS (POSIX 準拠)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>

#define FILE_PATH "/tmp/mmap_demo.dat"
#define FILE_SIZE (4096 * 4)  /* 16KB = 4 ページ分 */

/* デモ用のデータファイルを作成する */
int create_demo_file(const char *path, size_t size) {
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("open");
        return -1;
    }

    /* ファイルサイズを設定。ftruncate がファイルを指定サイズに拡張する。
       なぜ必要か: mmap はファイルサイズを超える領域にはマッピングできない */
    if (ftruncate(fd, size) < 0) {
        perror("ftruncate");
        close(fd);
        return -1;
    }

    return fd;
}

int main(void) {
    printf("=== mmap Demo: Memory-Mapped File I/O ===\n\n");

    /* Step 1: ファイルを作成 */
    printf("[1] Creating demo file: %s (%d bytes = %d pages)\n",
           FILE_PATH, FILE_SIZE, FILE_SIZE / 4096);
    int fd = create_demo_file(FILE_PATH, FILE_SIZE);
    if (fd < 0) return 1;

    /* Step 2: ファイルをメモリにマッピング
       MAP_SHARED: 変更がファイルに反映される（他プロセスからも見える）
       MAP_PRIVATE にすると COW でプライベートコピーが作成される */
    printf("[2] Mapping file to memory with mmap()\n");
    char *mapped = mmap(NULL, FILE_SIZE, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    /* mmap 後は fd を閉じても、マッピングは有効
       なぜなら: カーネルがファイルの参照カウントを管理しているため */
    close(fd);
    printf("    File descriptor closed (mapping still valid)\n");

    /* Step 3: メモリ書き込み = ファイル書き込み */
    printf("[3] Writing data through memory mapping\n");
    const char *messages[] = {
        "Page 0: Hello from mmap!",
        "Page 1: This is page-aligned data.",
        "Page 2: Memory-mapped I/O is efficient.",
        "Page 3: No read()/write() syscalls needed."
    };

    for (int i = 0; i < 4; i++) {
        /* 各ページの先頭に書き込み。
           初回アクセス時にデマンドページングでページフォルトが発生し、
           物理フレームが割り当てられる */
        char *page_start = mapped + (i * 4096);
        snprintf(page_start, 4096, "%s", messages[i]);
        printf("    Wrote to page %d (offset %d): \"%s\"\n",
               i, i * 4096, messages[i]);
    }

    /* Step 4: メモリ読み出し = ファイル読み出し */
    printf("[4] Reading data through memory mapping\n");
    for (int i = 0; i < 4; i++) {
        char *page_start = mapped + (i * 4096);
        printf("    Page %d: \"%s\"\n", i, page_start);
    }

    /* Step 5: msync でディスクへの同期を強制
       なぜ必要か: カーネルはパフォーマンスのために書き込みをバッファリングする。
       msync を呼ぶことで、変更が確実にディスクに書き出される */
    printf("[5] Syncing changes to disk with msync()\n");
    if (msync(mapped, FILE_SIZE, MS_SYNC) < 0) {
        perror("msync");
    }

    /* Step 6: アンマップ */
    printf("[6] Unmapping memory\n");
    if (munmap(mapped, FILE_SIZE) < 0) {
        perror("munmap");
    }

    /* Step 7: 通常の read() で確認 */
    printf("[7] Verifying with normal read()\n");
    fd = open(FILE_PATH, O_RDONLY);
    if (fd >= 0) {
        char buf[64];
        ssize_t n = read(fd, buf, sizeof(buf) - 1);
        if (n > 0) {
            buf[n] = '\0';
            printf("    Read from file: \"%s\"\n", buf);
        }
        close(fd);
    }

    /* クリーンアップ */
    unlink(FILE_PATH);
    printf("\n=== Demo Complete ===\n");

    return 0;
}
```

### コード例 5: ページング統計情報の取得 (Python / Linux)

```python
"""
paging_stats.py

Linux の /proc ファイルシステムからページング関連の統計情報を
読み取り、わかりやすく表示するツール。

仮想メモリの使用状況、ページフォルト回数、スワップ使用量、
TLB フラッシュ回数などを確認できる。

実行: python3 paging_stats.py
動作環境: Linux のみ
"""

import os
import sys
from typing import Dict, Optional


def read_proc_file(path: str) -> Optional[str]:
    """procfs からファイルを読み取る。"""
    try:
        with open(path, 'r') as f:
            return f.read()
    except (FileNotFoundError, PermissionError) as e:
        print(f"  Warning: Cannot read {path}: {e}")
        return None


def parse_meminfo() -> Dict[str, int]:
    """
    /proc/meminfo を解析し、メモリ情報を辞書として返す。
    値は全て KB 単位。
    """
    content = read_proc_file('/proc/meminfo')
    if content is None:
        return {}

    info = {}
    for line in content.strip().split('\n'):
        parts = line.split(':')
        if len(parts) == 2:
            key = parts[0].strip()
            # 数値部分を取り出す。"1234 kB" → 1234
            value_str = parts[1].strip().split()[0]
            try:
                info[key] = int(value_str)
            except ValueError:
                pass
    return info


def parse_vmstat() -> Dict[str, int]:
    """
    /proc/vmstat を解析し、仮想メモリ統計を辞書として返す。
    """
    content = read_proc_file('/proc/vmstat')
    if content is None:
        return {}

    stats = {}
    for line in content.strip().split('\n'):
        parts = line.split()
        if len(parts) == 2:
            try:
                stats[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return stats


def parse_process_status(pid: int) -> Dict[str, str]:
    """
    /proc/[pid]/status を解析し、プロセスのメモリ情報を返す。
    """
    content = read_proc_file(f'/proc/{pid}/status')
    if content is None:
        return {}

    status = {}
    for line in content.strip().split('\n'):
        parts = line.split(':', 1)
        if len(parts) == 2:
            status[parts[0].strip()] = parts[1].strip()
    return status


def format_kb(kb: int) -> str:
    """KB を人間が読みやすい形式に変換する。"""
    if kb >= 1048576:
        return f"{kb / 1048576:.1f} GB"
    elif kb >= 1024:
        return f"{kb / 1024:.1f} MB"
    else:
        return f"{kb} KB"


def print_memory_overview(meminfo: Dict[str, int]) -> None:
    """メモリの概要を表示する。"""
    print("=" * 60)
    print("MEMORY OVERVIEW")
    print("=" * 60)

    total = meminfo.get('MemTotal', 0)
    free = meminfo.get('MemFree', 0)
    available = meminfo.get('MemAvailable', 0)
    buffers = meminfo.get('Buffers', 0)
    cached = meminfo.get('Cached', 0)

    used = total - free - buffers - cached
    usage_pct = (used / total * 100) if total > 0 else 0

    print(f"  Total     : {format_kb(total)}")
    print(f"  Used      : {format_kb(used)} ({usage_pct:.1f}%)")
    print(f"  Free      : {format_kb(free)}")
    print(f"  Available : {format_kb(available)}")
    print(f"  Buffers   : {format_kb(buffers)}")
    print(f"  Cached    : {format_kb(cached)}")

    # ページサイズ情報
    page_size = os.sysconf('SC_PAGE_SIZE')
    print(f"\n  Page Size : {page_size} bytes ({page_size // 1024} KB)")
    print(f"  Total Pages: {total * 1024 // page_size:,}")


def print_swap_info(meminfo: Dict[str, int]) -> None:
    """スワップ情報を表示する。"""
    print(f"\n{'='*60}")
    print("SWAP INFORMATION")
    print("=" * 60)

    swap_total = meminfo.get('SwapTotal', 0)
    swap_free = meminfo.get('SwapFree', 0)
    swap_used = swap_total - swap_free
    swap_cached = meminfo.get('SwapCached', 0)

    if swap_total == 0:
        print("  Swap is not configured")
        return

    usage_pct = (swap_used / swap_total * 100) if swap_total > 0 else 0
    print(f"  Total  : {format_kb(swap_total)}")
    print(f"  Used   : {format_kb(swap_used)} ({usage_pct:.1f}%)")
    print(f"  Free   : {format_kb(swap_free)}")
    print(f"  Cached : {format_kb(swap_cached)}")


def print_paging_stats(vmstat: Dict[str, int]) -> None:
    """ページング関連の統計情報を表示する。"""
    print(f"\n{'='*60}")
    print("PAGING STATISTICS (since boot)")
    print("=" * 60)

    # ページフォルト
    pgfault = vmstat.get('pgfault', 0)
    pgmajfault = vmstat.get('pgmajfault', 0)
    pgminfault = pgfault - pgmajfault

    print(f"  Page Faults (total) : {pgfault:>15,}")
    print(f"    Minor faults      : {pgminfault:>15,}")
    print(f"    Major faults      : {pgmajfault:>15,}")
    if pgfault > 0:
        major_pct = pgmajfault / pgfault * 100
        print(f"    Major fault ratio : {major_pct:>14.4f}%")

    # ページイン/ページアウト
    pgpgin = vmstat.get('pgpgin', 0)
    pgpgout = vmstat.get('pgpgout', 0)
    print(f"\n  Pages In  (from disk) : {pgpgin:>12,} KB")
    print(f"  Pages Out (to disk)   : {pgpgout:>12,} KB")

    # スワップイン/スワップアウト
    pswpin = vmstat.get('pswpin', 0)
    pswpout = vmstat.get('pswpout', 0)
    print(f"\n  Swap In  : {pswpin:>12,} pages")
    print(f"  Swap Out : {pswpout:>12,} pages")


def print_process_memory(pid: int) -> None:
    """特定プロセスのメモリ情報を表示する。"""
    print(f"\n{'='*60}")
    print(f"PROCESS MEMORY (PID: {pid})")
    print("=" * 60)

    status = parse_process_status(pid)
    if not status:
        print(f"  Cannot read process {pid} info")
        return

    print(f"  Name     : {status.get('Name', 'Unknown')}")
    print(f"  VmSize   : {status.get('VmSize', 'N/A'):>12} (仮想メモリサイズ)")
    print(f"  VmRSS    : {status.get('VmRSS', 'N/A'):>12} (物理メモリ使用量)")
    print(f"  VmSwap   : {status.get('VmSwap', 'N/A'):>12} (スワップ使用量)")
    print(f"  VmPeak   : {status.get('VmPeak', 'N/A'):>12} (仮想メモリピーク)")
    print(f"  VmData   : {status.get('VmData', 'N/A'):>12} (データ領域)")
    print(f"  VmStk    : {status.get('VmStk', 'N/A'):>12} (スタック領域)")
    print(f"  VmLib    : {status.get('VmLib', 'N/A'):>12} (共有ライブラリ)")

    # ページフォルト情報は /proc/[pid]/stat から取得
    stat_content = read_proc_file(f'/proc/{pid}/stat')
    if stat_content:
        fields = stat_content.split()
        if len(fields) > 11:
            minflt = int(fields[9])
            majflt = int(fields[11])
            print(f"\n  Minor faults: {minflt:>12,}")
            print(f"  Major faults: {majflt:>12,}")


def main():
    if sys.platform != 'linux':
        print("This tool is designed for Linux systems.")
        print("Demonstrating with simulated data...\n")

        # Linux 以外でもデモ表示
        print("=" * 60)
        print("SIMULATED PAGING STATISTICS")
        print("=" * 60)
        print("  On a Linux system, this tool reads from:")
        print("    /proc/meminfo    - Memory usage overview")
        print("    /proc/vmstat     - Virtual memory statistics")
        print("    /proc/[pid]/stat - Per-process page fault counters")
        print("\n  Key metrics to monitor:")
        print("    - Major page faults: High values indicate thrashing")
        print("    - Swap usage: Non-zero means physical memory is insufficient")
        print("    - Minor/Major ratio: Should be >99% minor faults")
        return

    meminfo = parse_meminfo()
    vmstat = parse_vmstat()

    print_memory_overview(meminfo)
    print_swap_info(meminfo)
    print_paging_stats(vmstat)
    print_process_memory(os.getpid())

    print(f"\n{'='*60}")
    print("HUGE PAGES")
    print("=" * 60)
    hp_total = meminfo.get('HugePages_Total', 0)
    hp_free = meminfo.get('HugePages_Free', 0)
    hp_size = meminfo.get('Hugepagesize', 0)
    print(f"  Total     : {hp_total}")
    print(f"  Free      : {hp_free}")
    print(f"  Page Size : {format_kb(hp_size)}")

    thp = meminfo.get('AnonHugePages', 0)
    print(f"  Transparent Huge Pages (anon): {format_kb(thp)}")


if __name__ == '__main__':
    main()
```

---

## 8. スワッピングとスラッシング

### 8.1 スワッピングの仕組み

物理メモリが不足した場合、OS は使用頻度の低いページをディスク上の**スワップ領域 (Swap Space)** に退避させ、物理フレームを解放する。この処理を **スワップアウト (Swap Out)** と呼び、退避されたページが再びアクセスされた際にディスクから読み戻す処理を **スワップイン (Swap In)** と呼ぶ。

```
スワッピングの流れ:

  物理メモリ                 スワップ領域（ディスク）
  ┌──────────┐              ┌──────────────────┐
  │ Frame 0  │ ←(使用中)    │                  │
  ├──────────┤              │                  │
  │ Frame 1  │ ─swap out──→ │ Page X の内容     │
  ├──────────┤              │                  │
  │ Frame 2  │ ←(使用中)    │ Page Y の内容     │
  ├──────────┤              │                  │
  │ Frame 3  │ ←(新ページ)  │ Page Z の内容     │
  ├──────────┤              │                  │
  │   ...    │              └──────────────────┘
  └──────────┘

  スワップアウトの判断基準:
    1. Dirty ビット = 0 のページを優先（書き戻し不要）
    2. 参照ビット = 0 のページを優先（最近使われていない）
    3. カーネルページはスワップしない
    4. ロックされたページ (mlock) はスワップしない
```

### 8.2 Linux のスワップ管理

```
Linux のスワップ構成:

  /proc/swaps で確認:
    Filename    Type        Size       Used    Priority
    /dev/sda2   partition   8388604    102400  -2
    /swapfile   file        4194300    0       -3

  swappiness パラメータ (/proc/sys/vm/swappiness):
    値の範囲: 0〜200 (デフォルト: 60)

    0   : 可能な限りスワップしない（ファイルキャッシュを優先的に解放）
    60  : バランスの取れたデフォルト値
    100 : ページキャッシュとスワップを同等に扱う
    200 : 積極的にスワップする

  なぜ swappiness を調整するのか:
    データベースサーバー → swappiness=10〜20
      理由: DB は自前のキャッシュを持ち、ページキャッシュは不要。
            スワップが発生するとレイテンシが跳ね上がる。

    デスクトップ → swappiness=60 (デフォルト)
      理由: アプリの応答性とファイルキャッシュのバランスが重要。

    メモリ潤沢なサーバー → swappiness=1
      理由: OOM Killer よりはスワップの方がまし、という安全弁のみ必要。
```

### 8.3 スラッシング (Thrashing)

スラッシングとは、物理メモリが極度に不足し、ページフォルトが頻発して CPU がページの入れ替えばかりに時間を費やし、実際の処理がほとんど進まなくなる状態を指す。

```
スラッシングの発生メカニズム:

  CPU使用率
  100% ┤
       │        ┌──────┐
       │       ╱│      │╲
       │      ╱ │      │ ╲
   50% ┤     ╱  │      │  ╲
       │    ╱   │      │   ╲___________
       │   ╱    │      │    スラッシング
       │  ╱     │      │   (CPU が I/O 待ち
       │ ╱      │      │    ばかりになる)
    0% ┤╱───────┴──────┴───────────────
       └─────────────────────────────→
        少ない                    多い
             同時実行プロセス数

  対策:
    1. ワーキングセットモデル:
       各プロセスの「ワーキングセット」（最近参照したページの集合）を追跡し、
       ワーキングセットを物理メモリに収容できないプロセスはスワップアウトする

    2. PFF (Page Fault Frequency):
       プロセスのページフォルト頻度を監視し、
       閾値を超えたらフレームを追加割り当て、
       閾値を下回ったらフレームを回収する

    3. OOM Killer (Linux):
       メモリが完全に枯渇した場合、最もメモリを消費する
       プロセスを強制終了して物理メモリを確保する
```

---

## 9. Huge Pages（ラージページ）

### 9.1 なぜ Huge Pages が必要なのか

通常の 4KB ページでは、大量のメモリを使用するアプリケーション（データベース、仮想マシンモニタ、科学計算）で TLB ミスが多発する。Huge Pages を使うことで、1 つの TLB エントリでカバーできるメモリ領域を拡大し、TLB ミスを大幅に削減できる。

| ページサイズ | TLB 1エントリのカバー範囲 | 1024 エントリ TLB のカバー範囲 |
|------------|------------------------|------------------------------|
| 4KB | 4KB | 4MB |
| 2MB | 2MB | 2GB |
| 1GB | 1GB | 1TB |

```
Huge Pages の設定 (Linux):

  ■ 静的 Huge Pages (hugetlbfs):
    # 1024 個の 2MB ページを予約（合計 2GB）
    echo 1024 > /proc/sys/vm/nr_hugepages

    # 確認
    cat /proc/meminfo | grep -i huge
    HugePages_Total:    1024
    HugePages_Free:     1024
    Hugepagesize:       2048 kB

    # アプリケーションからの使用
    # shmget + SHM_HUGETLB、または mmap + MAP_HUGETLB

  ■ Transparent Huge Pages (THP):
    # カーネルが自動的に 4KB ページを 2MB ページに統合
    cat /sys/kernel/mm/transparent_hugepage/enabled
    [always] madvise never

    # madvise モード: アプリが MADV_HUGEPAGE で明示的に要求した場合のみ
    echo madvise > /sys/kernel/mm/transparent_hugepage/enabled

  なぜ THP を無効にするケースがあるのか:
    THP のページ統合（khugepaged）がバックグラウンドで CPU を消費し、
    レイテンシに敏感なアプリケーション（Redis 等）では性能低下を引き起こす
    ことがある。このため、Redis の公式ドキュメントでは THP の無効化を推奨。
```

---

## 10. Linux のメモリ管理アーキテクチャ

### 10.1 全体像

```
Linux のメモリ管理スタック:

  ┌────────────────────────────────────────────────────┐
  │                  ユーザー空間                       │
  │  malloc() / free() / new / delete / mmap()         │
  │                    ↓                               │
  │  ┌──────────────────────────────────────────┐      │
  │  │ ユーザー空間アロケータ                     │      │
  │  │   glibc ptmalloc2 / jemalloc / tcmalloc  │      │
  │  │   → フリーリスト管理、スレッドキャッシュ    │      │
  │  └──────────────────────────────────────────┘      │
  │                    ↓                               │
  │  brk() / sbrk() : ヒープ領域の拡張                  │
  │  mmap()         : 新しい仮想メモリ領域の確保         │
  ├────────────────────────────────────────────────────┤
  │                  カーネル空間                       │
  │  ┌──────────────────────────────────────────┐      │
  │  │ VMA (Virtual Memory Area) 管理            │      │
  │  │   vm_area_struct の赤黒木 / リスト         │      │
  │  │   → プロセスの仮想アドレス空間を記述       │      │
  │  └──────────────────────────────────────────┘      │
  │                    ↓                               │
  │  ┌──────────────────────────────────────────┐      │
  │  │ ページフォルトハンドラ                      │      │
  │  │   do_page_fault() → handle_mm_fault()     │      │
  │  │   → デマンドページング、COW の処理          │      │
  │  └──────────────────────────────────────────┘      │
  │                    ↓                               │
  │  ┌──────────────────────────────────────────┐      │
  │  │ ページアロケータ (Buddy System)            │      │
  │  │   物理ページを 2^n 個のブロック単位で管理   │      │
  │  │   /proc/buddyinfo で確認可能               │      │
  │  └──────────────────────────────────────────┘      │
  │                    ↓                               │
  │  ┌──────────────────────────────────────────┐      │
  │  │ SLUB アロケータ                            │      │
  │  │   カーネルオブジェクト用の小さなメモリブロック│      │
  │  │   /proc/slabinfo で確認可能                │      │
  │  └──────────────────────────────────────────┘      │
  │                    ↓                               │
  │  ┌──────────────────────────────────────────┐      │
  │  │ ページ回収 (Page Reclamation)              │      │
  │  │   kswapd デーモン / direct reclaim         │      │
  │  │   LRU リスト: active / inactive            │      │
  │  │   → Clock 系アルゴリズムで管理              │      │
  │  └──────────────────────────────────────────┘      │
  └────────────────────────────────────────────────────┘
```

### 10.2 Buddy System

Buddy System は物理ページの割り当て・解放を管理するアルゴリズムで、2 のべき乗サイズのブロック単位で動作する。

```
Buddy System の動作例:

  初期状態: 64ページの連続領域

  order=6 (64ページ)
  ┌─────────────────────────────────────────────────────────┐
  │                         64                               │
  └─────────────────────────────────────────────────────────┘

  8ページを要求 → order=3 のブロックが必要:
  1. order=6 を分割 → 2つの order=5 (32ページ)
  2. order=5 を分割 → 2つの order=4 (16ページ)
  3. order=4 を分割 → 2つの order=3 (8ページ)

  order=3   order=3   order=4      order=5
  ┌────────┬────────┬────────────┬──────────────────────────┐
  │ 使用中  │ 空き   │   空き      │        空き               │
  │  (8)   │  (8)   │   (16)     │        (32)              │
  └────────┴────────┴────────────┴──────────────────────────┘

  8ページを解放 → buddy（隣接する同サイズの空きブロック）と結合:
  order=3 + order=3 → order=4
  order=4 + order=4 → order=5
  order=5 + order=5 → order=6 (元に戻る)

  なぜ Buddy System なのか:
    - 結合判定が O(1): buddy のアドレスはビット演算で計算できる
    - 外部断片化を抑制: 結合により大きなブロックを維持
    - /proc/buddyinfo で各 order の空きブロック数を確認可能
```

### 10.3 OOM Killer

```
OOM Killer の動作:

  メモリ完全枯渇時の最後の手段。
  各プロセスに oom_score を付与し、最もスコアの高いプロセスを終了する。

  oom_score の計算要素:
    - プロセスの物理メモリ使用量（大きいほどスコア高）
    - スワップ使用量
    - プロセスの実行時間（長いほどスコア低）
    - nice 値

  確認と制御:
    # 特定プロセスの OOM スコアを確認
    cat /proc/<pid>/oom_score

    # OOM 対象から除外 (-1000 = 完全除外)
    echo -1000 > /proc/<pid>/oom_score_adj

    # OOM 優先度を上げる (1000 = 最優先で kill)
    echo 1000 > /proc/<pid>/oom_score_adj

  なぜ OOM Killer が必要なのか:
    Linux のオーバーコミット（malloc 成功 ≠ 物理メモリ確保）により、
    全プロセスが同時にメモリを使い始めると物理メモリが足りなくなる。
    OOM Killer はシステム全体のハングを防ぐための安全弁である。
```

---

## 11. 逆引きページテーブルとハッシュページテーブル

### 11.1 逆引きページテーブル (Inverted Page Table)

通常のページテーブルはプロセスごとの仮想アドレス空間全体をカバーするため、64 ビット環境では極めて大きくなり得る。逆引きページテーブルは発想を逆転させ、**物理フレームごとに 1 エントリ**を持つ。

```
逆引きページテーブル:

  通常のページテーブル (forward):
    プロセスごとに VPN → PFN のテーブルを保持
    エントリ数 = 仮想ページ数 (巨大になり得る)

  逆引きページテーブル (inverted):
    システム全体で 1 つ、PFN → (PID, VPN) のテーブルを保持
    エントリ数 = 物理フレーム数 (物理メモリサイズに比例)

  ┌────────────┐
  │ Frame 0    │ → (PID=5, VPN=0x100)
  ├────────────┤
  │ Frame 1    │ → (PID=3, VPN=0x200)
  ├────────────┤
  │ Frame 2    │ → (PID=5, VPN=0x300)
  ├────────────┤
  │ Frame 3    │ → (PID=7, VPN=0x050)
  ├────────────┤
  │    ...     │
  └────────────┘

  変換: (PID, VPN) → PFN
    全エントリを線形検索 → O(N) で遅い
    → ハッシュテーブルを併用して O(1) に

  採用例: PowerPC (IBM POWER), IA-64 (Itanium)

  メリット:
    - 物理メモリサイズに比例するため、メモリ消費が予測可能
    - 64ビット環境でもテーブルサイズが爆発しない

  デメリット:
    - ハッシュ衝突の処理が必要
    - 共有メモリの実装が複雑（1フレームに複数の VPN が対応）
```

---

## 12. メモリマップトファイル (mmap)

### 12.1 mmap の仕組み

`mmap()` システムコールは、ファイルや匿名メモリ領域をプロセスの仮想アドレス空間にマッピングする。マッピングされた領域に対するメモリアクセスは、カーネルが自動的にファイル I/O に変換する。

```
mmap の種類:

  ┌─────────────────┬──────────────────────────────────────────┐
  │ 種類             │ 説明                                     │
  ├─────────────────┼──────────────────────────────────────────┤
  │ MAP_SHARED      │ 変更がファイルに反映される。               │
  │ (ファイル)       │ 複数プロセスで共有可能。                   │
  │                 │ データベースで多用 (SQLite, LMDB)。        │
  ├─────────────────┼──────────────────────────────────────────┤
  │ MAP_PRIVATE     │ 変更はプロセスローカル (COW)。             │
  │ (ファイル)       │ 共有ライブラリの .text セクションに使用。  │
  ├─────────────────┼──────────────────────────────────────────┤
  │ MAP_ANONYMOUS   │ ファイルに紐付かない匿名メモリ。           │
  │ + MAP_PRIVATE   │ malloc の大きな割り当てに使用。             │
  ├─────────────────┼──────────────────────────────────────────┤
  │ MAP_ANONYMOUS   │ プロセス間共有メモリ。                     │
  │ + MAP_SHARED    │ fork() 後に親子で共有される。              │
  └─────────────────┴──────────────────────────────────────────┘

  mmap vs read()/write():

  ┌──────────────┬──────────────────┬──────────────────┐
  │ 項目          │ read()/write()   │ mmap()           │
  ├──────────────┼──────────────────┼──────────────────┤
  │ データコピー   │ カーネル→ユーザー │ ゼロコピー        │
  │              │ (2回コピー)      │ (ページテーブル   │
  │              │                  │  のみ設定)       │
  ├──────────────┼──────────────────┼──────────────────┤
  │ ランダム      │ lseek + read     │ ポインタ演算     │
  │ アクセス      │ (システムコール)  │ (ユーザー空間)   │
  ├──────────────┼──────────────────┼──────────────────┤
  │ 小さなファイル │ 効率的           │ mmap のオーバー   │
  │              │                  │ ヘッドが相対的に大│
  ├──────────────┼──────────────────┼──────────────────┤
  │ 大きなファイル │ バッファ管理が   │ 非常に効率的     │
  │              │ 必要             │                  │
  └──────────────┴──────────────────┴──────────────────┘
```

---

## 13. コード例（続き）

### コード例 6: ワーキングセットの推定 (Python)

```python
"""
working_set_estimator.py

ページ参照列からワーキングセットサイズ (WSS) を推定する。
ワーキングセットとは、ある時間窓 Δ 内でアクセスされた
ページの集合であり、スラッシング防止の基礎概念である。

実行: python3 working_set_estimator.py
"""

from typing import List, Set, Tuple
import random


def compute_working_set(reference_string: List[int],
                        window_size: int) -> List[Tuple[int, Set[int], int]]:
    """各時刻のワーキングセットを計算する。

    Args:
        reference_string: ページ参照列
        window_size: ワーキングセットの時間窓 Δ

    Returns:
        各時刻の (時刻, ワーキングセット, WSSサイズ) のリスト
    """
    results = []

    for t in range(len(reference_string)):
        # 時刻 t から過去 window_size 回分のアクセスを見る
        start = max(0, t - window_size + 1)
        window = reference_string[start:t + 1]
        ws = set(window)
        results.append((t, ws, len(ws)))

    return results


def analyze_working_set(reference_string: List[int],
                        window_sizes: List[int]) -> None:
    """異なるウィンドウサイズでワーキングセットを分析する。"""
    print("=" * 70)
    print("Working Set Analysis")
    print(f"Reference String ({len(reference_string)} accesses):")
    print(f"  {reference_string}")
    print("=" * 70)

    for delta in window_sizes:
        results = compute_working_set(reference_string, delta)

        # 平均 WSS を計算
        avg_wss = sum(wss for _, _, wss in results) / len(results)
        max_wss = max(wss for _, _, wss in results)
        min_wss = min(wss for _, _, wss in results)

        print(f"\nWindow size (Δ) = {delta}")
        print(f"  Average WSS: {avg_wss:.2f} pages")
        print(f"  Max WSS    : {max_wss} pages")
        print(f"  Min WSS    : {min_wss} pages")

        # 各時刻の詳細（短い参照列の場合のみ表示）
        if len(reference_string) <= 20:
            print(f"\n  {'Time':>4} | {'Page':>4} | {'Working Set':<25} | {'WSS':>3}")
            print(f"  {'-'*4}-+-{'-'*4}-+-{'-'*25}-+-{'-'*3}")
            for t, ws, wss in results:
                ws_str = str(sorted(ws))
                print(f"  {t:>4} | {reference_string[t]:>4} | "
                      f"{ws_str:<25} | {wss:>3}")


def simulate_thrashing(total_frames: int, num_processes: int,
                       wss_per_process: int) -> None:
    """スラッシングのシミュレーション。

    全プロセスのワーキングセット合計が物理フレーム数を超えると
    スラッシングが発生する様子を示す。
    """
    total_wss = num_processes * wss_per_process

    print(f"\n{'='*60}")
    print("Thrashing Simulation")
    print(f"{'='*60}")
    print(f"  Physical frames     : {total_frames}")
    print(f"  Processes           : {num_processes}")
    print(f"  WSS per process     : {wss_per_process} pages")
    print(f"  Total WSS demand    : {total_wss} pages")
    print(f"  Overcommit ratio    : {total_wss / total_frames:.2f}x")

    if total_wss <= total_frames:
        print(f"\n  Status: STABLE")
        print(f"  All working sets fit in physical memory.")
        print(f"  Expected page fault rate: LOW (compulsory faults only)")
    elif total_wss <= total_frames * 1.5:
        print(f"\n  Status: WARNING - Moderate swapping expected")
        print(f"  Some processes may experience elevated page faults.")
        print(f"  Expected performance degradation: 20-50%")
    else:
        print(f"\n  Status: THRASHING - Severe performance degradation")
        print(f"  Working sets cannot fit in memory.")
        print(f"  CPU will spend most time on page fault handling.")
        print(f"  Recommendation: Reduce processes or add memory.")

        # どのプロセスを退避すべきか
        max_active = total_frames // wss_per_process
        print(f"\n  Maximum concurrent processes: {max_active}")
        print(f"  Processes to suspend: {num_processes - max_active}")


if __name__ == "__main__":
    # 局所性のあるページ参照列
    ref_string = [1, 2, 3, 2, 1, 3, 4, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 1, 2]

    analyze_working_set(ref_string, window_sizes=[3, 5, 8])

    # スラッシングシミュレーション
    simulate_thrashing(total_frames=1000, num_processes=5, wss_per_process=150)
    simulate_thrashing(total_frames=1000, num_processes=10, wss_per_process=150)
    simulate_thrashing(total_frames=1000, num_processes=20, wss_per_process=150)
```

### コード例 7: ページテーブルウォークのシミュレーション (Python)

```python
"""
page_table_walk.py

x86-64 の 4 階層ページテーブルウォークをシミュレーションする。
仮想アドレスを各レベルのインデックスに分解し、
ページテーブルを辿って物理アドレスに変換する過程を表示する。

実行: python3 page_table_walk.py
"""

from typing import Optional, Dict, Tuple


class PageTableEntry:
    """ページテーブルエントリを表現するクラス。"""

    def __init__(self, pfn: int = 0, present: bool = False,
                 writable: bool = True, user: bool = True,
                 accessed: bool = False, dirty: bool = False,
                 huge: bool = False):
        self.pfn = pfn
        self.present = present
        self.writable = writable
        self.user = user
        self.accessed = accessed
        self.dirty = dirty
        self.huge = huge  # Huge Page (2MB / 1GB)

    def __repr__(self) -> str:
        flags = []
        if self.present:  flags.append("P")
        if self.writable: flags.append("W")
        if self.user:     flags.append("U")
        if self.accessed: flags.append("A")
        if self.dirty:    flags.append("D")
        if self.huge:     flags.append("H")
        return f"PTE(PFN=0x{self.pfn:05X}, flags={'|'.join(flags)})"


class FourLevelPageTable:
    """x86-64 の 4 階層ページテーブルをシミュレーションする。

    階層構造:
      PML4 (Page Map Level 4)       → 9ビット (bits 47-39)
      PDPT (Page Directory Pointer)  → 9ビット (bits 38-30)
      PD   (Page Directory)          → 9ビット (bits 29-21)
      PT   (Page Table)              → 9ビット (bits 20-12)
      Offset                         → 12ビット (bits 11-0)
    """

    PAGE_SHIFT = 12
    ENTRIES_PER_TABLE = 512  # 2^9
    INDEX_BITS = 9

    def __init__(self):
        # 各テーブルを辞書の辞書として表現
        # tables[level][table_pfn][index] = PageTableEntry
        self.tables: Dict[int, Dict[int, Dict[int, PageTableEntry]]] = {
            4: {},  # PML4
            3: {},  # PDPT
            2: {},  # PD
            1: {},  # PT
        }
        self.cr3 = 0x1000  # PML4 の物理アドレス (フレーム番号)
        self._init_pml4()
        self.walk_count = 0

    def _init_pml4(self):
        """PML4 テーブルを初期化する。"""
        self.tables[4][self.cr3] = {}

    def _ensure_table(self, level: int, table_pfn: int):
        """テーブルが存在しなければ作成する。"""
        if table_pfn not in self.tables[level]:
            self.tables[level][table_pfn] = {}

    def map_page(self, virtual_addr: int, physical_frame: int,
                 writable: bool = True, user: bool = True) -> None:
        """仮想ページを物理フレームにマッピングする。"""
        indices = self._extract_indices(virtual_addr)

        # Level 4 (PML4) → Level 3 (PDPT) のエントリ
        current_table_pfn = self.cr3
        next_pfn = physical_frame + 0x1000  # 中間テーブル用のフレーム

        for level in range(4, 1, -1):
            idx = indices[level]
            self._ensure_table(level, current_table_pfn)

            if idx not in self.tables[level][current_table_pfn]:
                # 新しい中間テーブルを割り当て
                new_table_pfn = next_pfn
                next_pfn += 1
                self.tables[level][current_table_pfn][idx] = PageTableEntry(
                    pfn=new_table_pfn, present=True,
                    writable=True, user=True
                )
                self._ensure_table(level - 1, new_table_pfn)

            entry = self.tables[level][current_table_pfn][idx]
            current_table_pfn = entry.pfn

        # Level 1 (PT) の最終エントリ
        idx = indices[1]
        self._ensure_table(1, current_table_pfn)
        self.tables[1][current_table_pfn][idx] = PageTableEntry(
            pfn=physical_frame, present=True,
            writable=writable, user=user
        )

    def _extract_indices(self, virtual_addr: int) -> Dict[int, int]:
        """仮想アドレスから各階層のインデックスを抽出する。"""
        offset = virtual_addr & 0xFFF
        pt_idx  = (virtual_addr >> 12) & 0x1FF
        pd_idx  = (virtual_addr >> 21) & 0x1FF
        pdpt_idx = (virtual_addr >> 30) & 0x1FF
        pml4_idx = (virtual_addr >> 39) & 0x1FF

        return {
            4: pml4_idx,
            3: pdpt_idx,
            2: pd_idx,
            1: pt_idx,
            0: offset
        }

    def walk(self, virtual_addr: int, verbose: bool = True
             ) -> Optional[int]:
        """4 階層ページテーブルウォークを実行する。"""
        self.walk_count += 1
        indices = self._extract_indices(virtual_addr)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Page Table Walk #{self.walk_count}")
            print(f"Virtual Address: 0x{virtual_addr:016X}")
            print(f"{'='*60}")
            print(f"  PML4 index : {indices[4]:>3} (0x{indices[4]:03X})")
            print(f"  PDPT index : {indices[3]:>3} (0x{indices[3]:03X})")
            print(f"  PD   index : {indices[2]:>3} (0x{indices[2]:03X})")
            print(f"  PT   index : {indices[1]:>3} (0x{indices[1]:03X})")
            print(f"  Offset     : {indices[0]:>3} (0x{indices[0]:03X})")
            print()

        level_names = {4: "PML4", 3: "PDPT", 2: "PD  ", 1: "PT  "}
        current_table_pfn = self.cr3
        memory_accesses = 0

        for level in range(4, 0, -1):
            idx = indices[level]
            memory_accesses += 1

            if (current_table_pfn not in self.tables[level] or
                idx not in self.tables[level][current_table_pfn]):
                if verbose:
                    print(f"  [{level_names[level]}] Table@0x{current_table_pfn:05X}"
                          f"[{idx}] -> PAGE FAULT (not present)")
                return None

            entry = self.tables[level][current_table_pfn][idx]

            if not entry.present:
                if verbose:
                    print(f"  [{level_names[level]}] Table@0x{current_table_pfn:05X}"
                          f"[{idx}] -> PAGE FAULT (P=0)")
                return None

            if verbose:
                print(f"  [{level_names[level]}] Table@0x{current_table_pfn:05X}"
                      f"[{idx}] -> {entry}")

            # Accessed ビットをセット
            entry.accessed = True

            if level > 1:
                current_table_pfn = entry.pfn
            else:
                # 最終レベル: 物理アドレスを計算
                physical_addr = (entry.pfn << self.PAGE_SHIFT) | indices[0]
                memory_accesses += 1  # データアクセス

                if verbose:
                    print(f"\n  Physical Address: 0x{physical_addr:016X}")
                    print(f"  Memory accesses for translation: {memory_accesses}")
                return physical_addr

        return None


if __name__ == "__main__":
    pt = FourLevelPageTable()

    # いくつかのページをマッピング
    print("Setting up page mappings...")
    pt.map_page(0x0000_0040_0000, 0x00100)  # 仮想 0x400000 → 物理フレーム 0x100
    pt.map_page(0x0000_0040_1000, 0x00200)  # 仮想 0x401000 → 物理フレーム 0x200
    pt.map_page(0x0000_7FFF_F000, 0x00300)  # スタック領域の仮想アドレス
    pt.map_page(0x0000_0000_1000, 0x00050)  # 低アドレス

    # ページテーブルウォークを実行
    pt.walk(0x0000_0040_0A7C)  # VPN=0x400, offset=0xA7C
    pt.walk(0x0000_0040_1500)  # VPN=0x401, offset=0x500
    pt.walk(0x0000_7FFF_F100)  # スタック近辺
    pt.walk(0x0000_0000_1234)  # 低アドレス

    # マッピングされていないアドレス → ページフォルト
    pt.walk(0x0000_DEAD_BEEF)
```

---

## 14. アンチパターン

### アンチパターン 1: mlock の過剰使用

```
問題:
  「ページフォルトが怖いので、全メモリを mlock() でロックする」

  mlock() は指定したメモリ領域をスワップ不可にし、常に物理メモリに保持する。
  リアルタイムシステムや暗号鍵の保護には適切だが、過剰使用は危険。

なぜ問題なのか:
  1. 他のプロセスが使える物理メモリが減少する
  2. OOM Killer が発動しやすくなる
  3. ロックしたメモリは実際に使っていなくても解放されない
  4. コンテナ環境では cgroup のメモリ制限と衝突する

正しいアプローチ:
  - 暗号鍵や認証トークンなど、スワップに書き出されると
    セキュリティリスクになるデータのみを mlock する
  - RLIMIT_MEMLOCK でプロセスがロックできるメモリ量を制限する
  - ページフォルトの削減が目的なら、Huge Pages や
    madvise(MADV_WILLNEED) を検討する
```

```c
/* アンチパターンの例 */
void *buf = malloc(HUGE_SIZE);
mlock(buf, HUGE_SIZE);  /* 全メモリをロック → 危険 */

/* 正しいアプローチ */
void *secret_key = malloc(KEY_SIZE);
mlock(secret_key, KEY_SIZE);  /* 必要最小限のみロック */
/* 使用後 */
memset(secret_key, 0, KEY_SIZE);  /* ゼロクリア */
munlock(secret_key, KEY_SIZE);
free(secret_key);
```

### アンチパターン 2: 32 ビットプロセスでの仮想アドレス空間枯渇

```
問題:
  「メモリは 8GB あるのに、malloc が失敗する」

原因:
  32 ビットプロセスの仮想アドレス空間は 4GB まで。
  そのうちカーネル空間（Linux デフォルト: 1GB）を除くと
  ユーザー空間は 3GB しかない。

  さらに、以下が仮想アドレスを消費する:
  - スタック (デフォルト 8MB)
  - 共有ライブラリ (.so / .dll)
  - mmap で確保した領域
  - ヒープのフラグメンテーション

  結果、物理メモリに空きがあっても仮想アドレス空間が足りず
  malloc が NULL を返す。

なぜ起きるのか:
  mmap は仮想アドレスを消費するが、munmap しないと仮想アドレスが
  断片化する。特に大量の中程度（128KB〜1MB）の割り当て・解放を
  繰り返すと、仮想アドレス空間のフラグメンテーションが進行する。

正しいアプローチ:
  1. 64 ビットビルドに移行する（仮想アドレス空間: 128TB〜）
  2. 32 ビットが必要な場合、Linux では 3G/1G 分割を 3.5G/0.5G に変更
  3. メモリプールを使い、mmap/munmap の回数を減らす
  4. jemalloc や tcmalloc でフラグメンテーションを抑制
```

---

## 15. エッジケース分析

### エッジケース 1: NUMA 環境でのページ配置

```
NUMA (Non-Uniform Memory Access) 環境:

  ┌─────────────┐         ┌─────────────┐
  │   CPU 0     │         │   CPU 1     │
  │  ┌───────┐  │         │  ┌───────┐  │
  │  │ Core0 │  │         │  │ Core2 │  │
  │  │ Core1 │  │         │  │ Core3 │  │
  │  └───────┘  │         │  └───────┘  │
  │      │      │  QPI/   │      │      │
  │  ┌───────┐  │  UPI    │  ┌───────┐  │
  │  │Local  │  │←-------→│  │Local  │  │
  │  │Memory │  │  リンク  │  │Memory │  │
  │  │(Node0)│  │         │  │(Node1)│  │
  │  └───────┘  │         │  └───────┘  │
  └─────────────┘         └─────────────┘

  ローカルメモリアクセス : ~100ns
  リモートメモリアクセス : ~150-300ns (1.5〜3倍遅い)

問題:
  ページ置換アルゴリズムが「空きフレーム」を選ぶとき、
  リモートノードのフレームを割り当てると性能が大幅に低下する。

  例: CPU 0 上で動作するプロセスに Node 1 のフレームが割り当てられると、
  毎回のメモリアクセスが QPI/UPI リンクを経由し 1.5〜3 倍遅くなる。

Linux の対策:
  - デフォルトポリシー (local): ページフォルトが発生した CPU の
    ローカルノードからフレームを割り当てる
  - numactl / set_mempolicy() で制御可能:
    numactl --membind=0 ./app   # Node 0 のメモリのみ使用
    numactl --interleave=all ./app  # 全ノードに均等分散

  監視:
    numastat -p <pid>  # プロセスの NUMA メモリ使用状況
    cat /proc/buddyinfo  # ノードごとの空きフレーム
```

### エッジケース 2: fork() 後のメモリ圧力と COW ストーム

```
問題:
  大量のメモリを使用するプロセス（例: Redis 10GB）が fork() した場合、
  COW により親子は同じ物理ページを共有する。
  しかし、親プロセスが書き込みを続けると、書き込まれたページごとに
  コピーが発生し、一時的にメモリ使用量がほぼ 2 倍になる。

  Redis 10GB + 書き込み率が高い場合:
    fork() 直後: 10GB (共有)
    全ページに書き込み発生: 最大 20GB 必要
    → 物理メモリが 16GB しかなければ、OOM Killer が発動

  これが「COW ストーム」と呼ばれる現象で、
  Redis の RDB 永続化や BGSAVE で頻繁に発生する。

対策:
  1. overcommit_memory の設定:
     echo 1 > /proc/sys/vm/overcommit_memory
     → fork() を常に許可（OOM のリスクはあるが BGSAVE が失敗しない）

  2. 十分なスワップ領域の確保:
     一時的なメモリ増加をスワップで吸収する

  3. Huge Pages の使用を避ける:
     COW のコピー単位が 2MB になり、コストが増大する

  4. Redis 7.0 以降では fork-less な永続化方式も検討
```

---

## 16. 実践演習

### 演習 1 [基礎]: ページテーブルサイズの計算

```
問題:
  32ビット仮想アドレス空間、4KB ページサイズ、PTE = 4 バイトの場合:

  (1) 1 階層ページテーブルのサイズを計算せよ。
  (2) 2 階層ページテーブルで、プロセスが 4MB のメモリを使用する場合、
      必要なページテーブルメモリの最小量を計算せよ。
  (3) なぜ多階層が効率的か、具体的な数値を用いて説明せよ。

解答例:
  (1) ページ数 = 2^32 / 2^12 = 2^20 = 1,048,576 ページ
      テーブルサイズ = 1,048,576 × 4 バイト = 4MB

  (2) 4MB = 1024 ページ = 1 つのページテーブル (1024 エントリ)
      ページディレクトリ: 4KB (1024 エントリ × 4 バイト)
      ページテーブル: 4KB × 1 (4MB は 1 つのページディレクトリエントリの
                     カバー範囲 4MB にちょうど収まる)
      合計: 4KB + 4KB = 8KB

  (3) 単一レベル: 全プロセスに 4MB 必要（たとえ 1 バイトしか使わなくても）
      2 階層: 使用領域に応じて 8KB〜4MB+4KB
      100 プロセスの場合:
        単一レベル: 100 × 4MB = 400MB がページテーブルだけで消費
        2 階層 (各 4MB 使用): 100 × 8KB = 800KB → 500 倍の節約
```

### 演習 2 [応用]: TLB ヒット率と実効アクセス時間

```
問題:
  以下の条件でシステムの実効メモリアクセス時間を計算せよ:

  条件:
    TLB アクセス時間: 1ns
    メモリアクセス時間: 100ns
    ページテーブル階層数: 4
    TLB ヒット率: 95%

  (1) TLB ヒット時のアクセス時間
  (2) TLB ミス時のアクセス時間
  (3) 実効アクセス時間 (EAT)
  (4) TLB がない場合と比較して、何倍高速か
  (5) ヒット率を 99% に改善すると EAT はいくつになるか

解答例:
  (1) TLB ヒット: 1ns (TLB) + 100ns (メモリ) = 101ns

  (2) TLB ミス: 1ns (TLB) + 4 × 100ns (ページウォーク) + 100ns (メモリ)
              = 1 + 400 + 100 = 501ns

  (3) EAT = 0.95 × 101 + 0.05 × 501
          = 95.95 + 25.05 = 121ns

  (4) TLB なし: 4 × 100 + 100 = 500ns
      高速化率: 500 / 121 = 4.13 倍

  (5) EAT(99%) = 0.99 × 101 + 0.01 × 501
               = 99.99 + 5.01 = 105ns
      → 4% のヒット率改善で 13% の性能向上
```

### 演習 3 [発展]: ページ置換アルゴリズムの比較シミュレーション

```
問題:
  以下のページ参照列に対して、フレーム数 3 で各アルゴリズムの
  ページフォルト回数を計算し、比較せよ:

  参照列: 1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5

  (1) FIFO のページフォルト回数を求め、各ステップのフレーム状態を示せ
  (2) LRU のページフォルト回数を求め、各ステップのフレーム状態を示せ
  (3) OPT のページフォルト回数を求め、各ステップのフレーム状態を示せ
  (4) フレーム数を 4 に増やした場合の FIFO の結果を求め、
      Belady の異常が発生するか確認せよ

解答例 (FIFO, フレーム数=3):
  Step 1: Page 1 → [1, -, -] FAULT
  Step 2: Page 2 → [1, 2, -] FAULT
  Step 3: Page 3 → [1, 2, 3] FAULT
  Step 4: Page 4 → [4, 2, 3] FAULT (1を置換)
  Step 5: Page 1 → [4, 1, 3] FAULT (2を置換)
  Step 6: Page 2 → [4, 1, 2] FAULT (3を置換)
  Step 7: Page 5 → [5, 1, 2] FAULT (4を置換)
  Step 8: Page 1 → [5, 1, 2] HIT
  Step 9: Page 2 → [5, 1, 2] HIT
  Step10: Page 3 → [3, 1, 2] FAULT (5を置換)  ※FIFO順: 5が最古
  Step11: Page 4 → [3, 4, 2] FAULT (1を置換)
  Step12: Page 5 → [3, 4, 5] FAULT (2を置換)
  FIFO フォルト数: 10

  ※ LRU と OPT は本章のコード例 2 で確認可能
```

---

## 17. FAQ

### Q1: malloc() は実際に何をしているのか？ページングとの関係は？

`malloc()` は C ライブラリ（glibc の ptmalloc2 など）の関数であり、直接的なシステムコールではない。内部動作は以下の通り:

1. **小さな割り当て (< 128KB 程度)**: `brk()` / `sbrk()` でヒープを拡張。glibc は内部でフリーリストを管理し、解放されたメモリを再利用する。
2. **大きな割り当て (>= 128KB 程度)**: `mmap(MAP_ANONYMOUS | MAP_PRIVATE)` で新しい仮想メモリ領域を確保。解放時は `munmap()` で OS に返却。
3. **物理メモリの割り当てはアクセス時**: `malloc()` が返すのは仮想アドレスのみ。実際の物理フレームは、そのアドレスに初めてアクセスした時にデマンドページングで割り当てられる。

これが「malloc 成功 ≠ 物理メモリ確保」と言われる理由である。Linux のオーバーコミット機構により、物理メモリ + スワップの合計を超える仮想メモリを `malloc()` で確保できてしまう。

### Q2: カーネル空間のメモリはページングされるのか？

カーネル空間のメモリ管理はユーザー空間とは異なる:

- **カーネルのコードとデータ**: 起動時に物理メモリにロードされ、通常はスワップされない。カーネルのページテーブルは恒久的にマッピングされる。
- **SLUB アロケータのオブジェクト**: カーネル内の小さなオブジェクト（`struct task_struct` 等）は Buddy System + SLUB で管理され、スワップ対象外。
- **ページキャッシュ**: ファイルの読み書きに使われるキャッシュページは、メモリ圧力に応じて回収（解放）される。ただし「スワップアウト」ではなく、ファイルに書き戻した後にフレームを解放する形式。
- **vmalloc 領域**: カーネル内で仮想的に連続なメモリを確保する。物理的には不連続だが、ページテーブルで連続にマッピング。

まとめると、カーネル自体のコードやデータ構造はスワップされないが、カーネルが管理するページキャッシュは回収の対象になる。

### Q3: 仮想マシン (VM) のメモリ管理はどうなっているのか？

仮想化環境では、ページングが **2 段階** になる:

```
2段階アドレス変換 (Nested Paging / EPT):

  ゲストOS:
    ゲスト仮想アドレス (GVA)
      → ゲストページテーブル
    ゲスト物理アドレス (GPA)

  ハイパーバイザー:
    ゲスト物理アドレス (GPA)
      → ネステッドページテーブル (EPT / NPT)
    ホスト物理アドレス (HPA)

  GVA → GPA → HPA の 2 段階変換

  TLB ミス時のコスト:
    ゲスト 4 階層 × ホスト 4 階層 = 最大 24 回のメモリアクセス
    → ハードウェア支援 (Intel EPT / AMD NPT) で TLB に GVA→HPA を
      直接キャッシュし、性能低下を最小限に抑える
```

さらに、ハイパーバイザーは **バルーニング (ballooning)** 技術で VM のメモリ量を動的に調整できる。バルーンドライバがゲスト OS 内で「メモリを消費」することで、ゲスト OS にメモリ圧力をかけ、不要なページをスワップアウトさせる。回収されたフレームは他の VM に再配分される。

### Q4: ページサイズを変更することはできるのか？

x86-64 ではページサイズはハードウェアで 4KB / 2MB / 1GB に固定されており、OS が任意のサイズを選択することはできない。ただし、ARM アーキテクチャでは 4KB / 16KB / 64KB のベースページサイズを起動時に選択可能であり、Apple Silicon (macOS / iOS) は 16KB ページを採用している。

16KB ページの影響:
- TLB カバレッジが 4 倍に拡大（同じ TLB エントリ数でより広い範囲をカバー）
- 内部断片化は最大 16KB-1（平均 8KB）に増大
- I/O 効率が向上（1 回のページフォルトで 16KB を転送）
- 4KB 前提のソフトウェアとの互換性問題が発生し得る

---

## 18. まとめ

| 概念 | ポイント |
|------|---------|
| ページング | 固定サイズ (4KB) 分割。外部断片化なし。現代のメモリ管理の基盤 |
| 多階層ページテーブル | 使用されない領域のテーブルを省略し、メモリを節約 |
| TLB | アドレス変換キャッシュ。ヒット率 99% で実用的な性能を達成 |
| デマンドページング | アクセス時に初めて物理ページを割り当て。起動時間とメモリ使用量を削減 |
| COW (Copy-on-Write) | fork() の効率化。書き込みが発生するまでページを共有 |
| ページ置換 | LRU が理論上最良に近いが、実装コストから Clock が実用される |
| スワッピング | ディスクを仮想メモリの延長として使用。Major ページフォルトのコストは巨大 |
| スラッシング | ワーキングセットが物理メモリに収まらず、性能が激しく低下する状態 |
| Huge Pages | TLB カバレッジ拡大。大規模メモリアプリケーションで有効 |
| NUMA | メモリの物理的配置が性能に影響。ローカルノード優先のページ配置が重要 |

---

## 19. 用語集

| 用語 | 英語 | 説明 |
|------|------|------|
| ページ | Page | 仮想アドレス空間の固定サイズ単位 (通常 4KB) |
| フレーム | Frame | 物理メモリの固定サイズ単位 |
| VPN | Virtual Page Number | 仮想アドレスのページ番号部分 |
| PFN | Physical Frame Number | 物理アドレスのフレーム番号部分 |
| PTE | Page Table Entry | ページテーブルの 1 エントリ |
| TLB | Translation Lookaside Buffer | アドレス変換キャッシュ |
| ASID | Address Space Identifier | TLB エントリのプロセス識別子 |
| COW | Copy-on-Write | 書き込み時コピー |
| OOM | Out of Memory | メモリ枯渇状態 |
| WSS | Working Set Size | ワーキングセットのサイズ |
| NUMA | Non-Uniform Memory Access | 不均一メモリアクセス |
| THP | Transparent Huge Pages | 透過的ラージページ |

---

## 次に読むべきガイド

- [[02-memory-allocation.md]] -- メモリ割り当て戦略（Buddy System、SLUB アロケータの詳細）
- [[03-virtual-memory-advanced.md]] -- 仮想メモリの応用（NUMA、メモリ圧縮、KSM）

---

## 参考文献

1. Silberschatz, A., Galvin, P. B., & Gagne, G. "Operating System Concepts." 10th Edition, Chapter 9-10 (Virtual Memory), Wiley, 2018.
   - ページング、デマンドページング、ページ置換アルゴリズムの理論的基礎を網羅的に解説した教科書の定番。

2. Bovet, D. P. & Cesati, M. "Understanding the Linux Kernel." 3rd Edition, Chapter 2, 8-9, O'Reilly, 2005.
   - Linux カーネルのメモリ管理実装（Buddy System、SLUB、ページフォルトハンドラ）を詳述。

3. Gorman, M. "Understanding the Linux Virtual Memory Manager." Prentice Hall, 2004. (https://www.kernel.org/doc/gorman/)
   - Linux の仮想メモリサブシステムを包括的に解説。ページ回収、スワップ、NUMA に関する実装レベルの情報を提供。

4. Intel Corporation. "Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 3A: System Programming Guide." Chapter 4 (Paging), 2024.
   - x86-64 のページング機構（4 階層ページテーブル、TLB、EPT）のハードウェア仕様書。

5. Love, R. "Linux Kernel Development." 3rd Edition, Chapter 15 (The Process Address Space), Addison-Wesley, 2010.
   - Linux カーネルのメモリ管理を開発者の視点で平易に解説。VMA、デマンドページング、COW の実装を含む。

6. Arpaci-Dusseau, R. H. & Arpaci-Dusseau, A. C. "Operating Systems: Three Easy Pieces." Chapter 18-23, Arpaci-Dusseau Books, 2018. (https://pages.cs.wisc.edu/~remzi/OSTEP/)
   - ページング、TLB、ページ置換、スワッピングを段階的に解説した無料のオンライン教科書。初学者に強く推奨。

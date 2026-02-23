# 仮想マシンの基礎

> 仮想化は「1台の物理マシン上で複数の独立した仮想マシンを実行する」技術であり、クラウドコンピューティングの基盤。

## この章で学ぶこと

- [ ] 仮想化の種類を区別できる
- [ ] ハイパーバイザの仕組みを理解する
- [ ] 主要な仮想化技術を知る
- [ ] CPU仮想化・メモリ仮想化・I/O仮想化の仕組みを理解する
- [ ] KVM/QEMU の構成と運用を実践できる
- [ ] ライブマイグレーションの仕組みを理解する
- [ ] 仮想化のパフォーマンスチューニングができる
- [ ] ネステッド仮想化とその活用を知る

---

## 1. 仮想化の歴史と概要

### 1.1 仮想化の歴史

```
仮想化の歴史:

  1960年代:
  ┌──────────────────────────────────────────────────┐
  │ IBM System/360 Model 67 (1966):                   │
  │ → 世界初の仮想マシンモニター                     │
  │ → CP-40 / CP-67: VMM の原型                     │
  │ → 1台のメインフレームで複数のOSを同時実行       │
  │                                                    │
  │ IBM VM/370 (1972):                                │
  │ → 商用の仮想化プラットフォーム                   │
  │ → 各ユーザーに独立した仮想マシンを提供           │
  │ → CMS（Conversational Monitor System）を実行     │
  └──────────────────────────────────────────────────┘

  1990年代〜2000年代:
  ┌──────────────────────────────────────────────────┐
  │ VMware（1999）:                                   │
  │ → x86アーキテクチャでの仮想化を商用化            │
  │ → バイナリ変換で特権命令を処理                   │
  │ → VMware Workstation, ESX Server                  │
  │                                                    │
  │ Xen（2003）:                                      │
  │ → ケンブリッジ大学で開発                         │
  │ → 準仮想化による高性能                           │
  │ → Amazon EC2 の初期基盤                          │
  │                                                    │
  │ Intel VT-x / AMD-V（2005-2006）:                  │
  │ → ハードウェア支援仮想化                         │
  │ → CPU に仮想化専用命令を追加                     │
  │ → 完全仮想化でも高性能を実現                     │
  └──────────────────────────────────────────────────┘

  2007年〜現在:
  ┌──────────────────────────────────────────────────┐
  │ KVM（2007, Linux 2.6.20）:                        │
  │ → Linux カーネルモジュールとして実装              │
  │ → Linux 自体をハイパーバイザに変える             │
  │ → クラウドの標準的な仮想化基盤に成長             │
  │                                                    │
  │ クラウド時代（2006〜）:                           │
  │ → AWS EC2（2006）: Xen → Nitro (KVM)            │
  │ → Google Compute Engine: KVM                      │
  │ → Azure: Hyper-V                                  │
  │ → 仮想化がクラウドコンピューティングの基盤に     │
  │                                                    │
  │ 軽量仮想化（2018〜）:                             │
  │ → Firecracker: マイクロVM                        │
  │ → Cloud Hypervisor: Rust製軽量VMM               │
  │ → QEMU の代替として高セキュリティ・低オーバーヘッド │
  └──────────────────────────────────────────────────┘
```

### 1.2 Popek & Goldberg の仮想化要件

```
仮想化の理論的基盤（1974年）:
  Popek & Goldberg が定義した仮想マシンモニター（VMM）の3要件:

  1. 等価性（Equivalence / Fidelity）:
     VMM上で実行されるプログラムは、実機と同じ動作をする
     → 一部の例外: タイミング、リソースの可用性

  2. 効率性（Efficiency）:
     ゲストの命令の大部分はハードウェア上で直接実行される
     → エミュレーションではなく、直接実行

  3. リソース制御（Resource Control / Safety）:
     VMMはすべてのハードウェアリソースを完全に制御する
     → ゲストがVMMをバイパスしてリソースにアクセスできない

  x86の仮想化困難性:
  ┌──────────────────────────────────────────────────┐
  │ 問題: x86 には「センシティブだが特権的でない命令」│
  │ が存在した                                        │
  │                                                    │
  │ 特権命令: Ring 0 以外で実行すると例外が発生      │
  │ → VMM がトラップして処理可能                     │
  │                                                    │
  │ センシティブ命令: システム状態に影響するが        │
  │ 特権レベルに関係なく実行可能                      │
  │ → VMM がトラップできない!                        │
  │                                                    │
  │ 問題のある命令の例:                                │
  │ - SGDT/SIDT: ディスクリプタテーブルの読み取り    │
  │ - SMSW: マシンステータスワードの読み取り         │
  │ - PUSHF/POPF: フラグレジスタの操作               │
  │                                                    │
  │ 解決策:                                            │
  │ 1. バイナリ変換（VMware）                         │
  │ 2. 準仮想化（Xen）                                │
  │ 3. ハードウェア支援（VT-x/AMD-V）← 根本解決    │
  └──────────────────────────────────────────────────┘
```

---

## 2. 仮想化の種類

### 2.1 完全仮想化

```
完全仮想化（Full Virtualization）:
  ゲストOSを無修正で実行

  方式1: バイナリ変換（Binary Translation）
  ┌──────────────────────────────────────────────────┐
  │ ゲスト OS のコード                                │
  │       ↓                                           │
  │ バイナリ変換エンジン                              │
  │   → ユーザーモード命令: 直接実行                 │
  │   → センシティブ命令: 安全な命令列に変換         │
  │       ↓                                           │
  │ 変換済みコード                                    │
  │   → キャッシュに保存して再利用                   │
  │       ↓                                           │
  │ 物理ハードウェア上で実行                          │
  │                                                    │
  │ VMware が開発・実用化                             │
  │ → 性能オーバーヘッド: 10-30%                     │
  │ → ゲストOSの修正不要が最大のメリット             │
  └──────────────────────────────────────────────────┘

  方式2: ハードウェア支援仮想化（HW-Assisted）
  ┌──────────────────────────────────────────────────┐
  │ CPU に仮想化支援命令を追加:                       │
  │ → Intel VT-x（VMX: Virtual Machine Extensions）  │
  │ → AMD-V（SVM: Secure Virtual Machine）           │
  │                                                    │
  │ CPU 動作モード:                                   │
  │ ┌─────────────────────────┐                       │
  │ │ VMX Root Mode（ホスト）│                       │
  │ │   Ring 0: ハイパーバイザ│                       │
  │ │   Ring 3: ホストアプリ  │                       │
  │ └─────────┬───────────────┘                       │
  │           │ VM Entry                              │
  │           ↓                                       │
  │ ┌─────────────────────────┐                       │
  │ │ VMX Non-Root（ゲスト） │                       │
  │ │   Ring 0: ゲストカーネル│                       │
  │ │   Ring 3: ゲストアプリ  │                       │
  │ └─────────┬───────────────┘                       │
  │           │ VM Exit（特権操作時）                 │
  │           ↓                                       │
  │ VMX Root Mode で処理後、VM Entry で復帰          │
  │                                                    │
  │ VMCS（Virtual Machine Control Structure）:        │
  │ → ゲスト/ホストの状態を保存する構造体            │
  │ → VM Entry/Exit 時に自動的にロード/セーブ       │
  │ → VM Exit の条件を設定可能                       │
  └──────────────────────────────────────────────────┘
```

### 2.2 準仮想化

```
準仮想化（Paravirtualization）:
  ゲストOSを仮想環境向けに修正

  ┌──────────────────────────────────────────────────┐
  │ ゲスト OS（修正版）                               │
  │       ↓ ハイパーコール                           │
  │ ハイパーバイザ                                    │
  │       ↓ 実際のハードウェア操作                   │
  │ 物理ハードウェア                                  │
  │                                                    │
  │ ハイパーコール（Hypercall）:                      │
  │ → ゲストOSからハイパーバイザへの直接要求         │
  │ → システムコールのようなインターフェース         │
  │ → 特権操作をバイナリ変換なしで実行可能           │
  │                                                    │
  │ 利点:                                              │
  │ - バイナリ変換より高性能                          │
  │ - 割り込み処理が効率的                            │
  │ - I/O が高速                                      │
  │                                                    │
  │ 欠点:                                              │
  │ - ゲストOSのカーネル修正が必要                   │
  │ - Windows等の修正不可能なOSでは使用不可          │
  │ - ハードウェア支援仮想化の普及で利点が減少       │
  └──────────────────────────────────────────────────┘

  準仮想化ドライバ（Virtio）:
  → ゲストOSのカーネル全体を修正せず、
    ドライバ部分のみ準仮想化する折衷案
  → 現代の仮想化で広く使用
  → ネットワーク、ストレージ、メモリバルーンなど
```

### 2.3 エミュレーション

```
エミュレーション（Emulation）:
  ハードウェア全体をソフトウェアで模倣

  ┌──────────────────────────────────────────────────┐
  │ エミュレーション vs 仮想化:                       │
  │                                                    │
  │ エミュレーション:                                 │
  │ → 異なるアーキテクチャをソフトウェアで再現       │
  │ → 例: x86上でARM、MIPS上でx86                    │
  │ → 非常に遅い（10-100倍のオーバーヘッド）        │
  │ → QEMU（JIT変換で高速化）                        │
  │                                                    │
  │ 仮想化:                                           │
  │ → 同じアーキテクチャで複数インスタンスを実行     │
  │ → ゲスト命令を直接CPUで実行                      │
  │ → ほぼネイティブ性能                             │
  │ → KVM + QEMU（デバイスエミュレーション部分のみ）│
  │                                                    │
  │ QEMU の2つのモード:                               │
  │ 1. Full System Emulation:                         │
  │    完全なシステムをエミュレート                   │
  │    → qemu-system-aarch64（ARM システムを模倣）   │
  │                                                    │
  │ 2. User Mode Emulation:                           │
  │    ユーザープログラムのみエミュレート             │
  │    → qemu-aarch64 ./arm_binary                   │
  │    → Docker の multi-platform ビルドで使用       │
  └──────────────────────────────────────────────────┘
```

---

## 3. ハイパーバイザのアーキテクチャ

### 3.1 Type 1 ハイパーバイザ（ベアメタル）

```
Type 1（ベアメタル）ハイパーバイザ:
  ハードウェア上に直接動作

  ┌──────┐ ┌──────┐ ┌──────┐
  │ VM 1 │ │ VM 2 │ │ VM 3 │
  │ OS A │ │ OS B │ │ OS C │
  └──┬───┘ └──┬───┘ └──┬───┘
  ┌──┴────────┴────────┴───┐
  │ Type 1 Hypervisor      │  ← ハードウェア上に直接
  └────────────────────────┘
  ┌────────────────────────┐
  │ Hardware               │
  └────────────────────────┘

  主要な Type 1 ハイパーバイザ:
  ┌──────────────────────────────────────────────────┐
  │ VMware ESXi:                                      │
  │ → 商用のエンタープライズ標準                     │
  │ → vSphere / vCenter で統合管理                   │
  │ → vMotion によるライブマイグレーション            │
  │ → DRS（Distributed Resource Scheduler）           │
  │ → HA（High Availability）                        │
  │ → 大企業の仮想化基盤として圧倒的シェア          │
  │                                                    │
  │ Microsoft Hyper-V:                                │
  │ → Windows Server に統合                          │
  │ → Azure の基盤技術                               │
  │ → System Center で管理                           │
  │ → Windows環境との親和性が高い                    │
  │ → Generation 2 VM で UEFI, Secure Boot 対応     │
  │                                                    │
  │ Xen:                                              │
  │ → オープンソース                                 │
  │ → Dom0（特権ドメイン）+ DomU（ゲスト）          │
  │ → AWS EC2 の初期基盤（現在は Nitro/KVM）        │
  │ → Citrix Hypervisor（旧 XenServer）              │
  │ → Qubes OS のセキュリティ基盤                    │
  │                                                    │
  │ KVM（Kernel-based Virtual Machine）:              │
  │ → Linux カーネルモジュール                       │
  │ → Linux をType 1ハイパーバイザに変える           │
  │ → QEMU と組み合わせてデバイスエミュレーション    │
  │ → AWS EC2 (Nitro), GCE, OpenStack の基盤        │
  │ → 最も広く使われるオープンソースハイパーバイザ   │
  └──────────────────────────────────────────────────┘
```

### 3.2 Type 2 ハイパーバイザ（ホスト型）

```
Type 2（ホスト型）ハイパーバイザ:
  ホストOS上のアプリケーションとして動作

  ┌──────┐ ┌──────┐
  │ VM 1 │ │ VM 2 │
  │ OS A │ │ OS B │
  └──┬───┘ └──┬───┘
  ┌──┴────────┴───┐
  │ Hypervisor    │  ← ホストOS上のアプリケーション
  ├───────────────┤
  │ Host OS       │
  └───────────────┘
  ┌───────────────┐
  │ Hardware      │
  └───────────────┘

  主要な Type 2 ハイパーバイザ:
  ┌──────────────────────────────────────────────────┐
  │ VirtualBox（Oracle）:                             │
  │ → オープンソース（GPLv2）                        │
  │ → クロスプラットフォーム（Windows/Mac/Linux）    │
  │ → 開発・テスト環境で広く使用                     │
  │ → 簡単なGUI管理                                  │
  │ → VBoxManage CLI で自動化可能                    │
  │ → スナップショット、共有フォルダ対応             │
  │                                                    │
  │ VMware Workstation / Fusion:                      │
  │ → 商用（個人利用は無料化）                       │
  │ → Workstation: Windows/Linux ホスト              │
  │ → Fusion: macOS ホスト                           │
  │ → 高性能、エンタープライズ機能                   │
  │ → Unity モード（ゲストアプリをホストデスクトップに）│
  │                                                    │
  │ Parallels Desktop:                                │
  │ → macOS 専用                                     │
  │ → Apple Silicon (M1/M2/M3) ネイティブ対応       │
  │ → Coherence モード（シームレスな統合）           │
  │ → 最も高性能なMac仮想化ソリューション           │
  │                                                    │
  │ QEMU:                                             │
  │ → オープンソースのエミュレータ/仮想化ツール      │
  │ → 単体ではエミュレーション                       │
  │ → KVM と組み合わせてハードウェア支援仮想化       │
  │ → 多数のアーキテクチャをサポート                 │
  │ → libvirt / virsh で管理                          │
  └──────────────────────────────────────────────────┘

  KVM の分類について:
  ┌──────────────────────────────────────────────────┐
  │ KVM は厳密には Type 1 と Type 2 の中間:          │
  │                                                    │
  │ → Linux カーネルの一部として動作                 │
  │   → Linux がハイパーバイザとして機能 (Type 1的) │
  │ → しかし Linux というフルOSの上で動作            │
  │   → ホストOS上のモジュール (Type 2的)           │
  │                                                    │
  │ 一般的にはType 1として分類される                  │
  │ （カーネルモジュールとして直接HWにアクセス）     │
  └──────────────────────────────────────────────────┘
```

---

## 4. CPU仮想化

### 4.1 Intel VT-x の仕組み

```
Intel VT-x（Virtual Machine Extensions）:

  VMX 動作:
  ┌──────────────────────────────────────────────────┐
  │ VMXON: VMX モードを有効化                        │
  │       ↓                                           │
  │ VMCS の作成・設定                                 │
  │       ↓                                           │
  │ VMLAUNCH: ゲストの初回起動                       │
  │       ↓                                           │
  │ ┌─── ゲスト実行 ───┐                             │
  │ │ 通常命令: 直接実行│                             │
  │ │ 特権操作: VM Exit │                             │
  │ └────────┬─────────┘                             │
  │          ↓                                        │
  │ VMM による処理（VM Exit ハンドラ）               │
  │          ↓                                        │
  │ VMRESUME: ゲスト実行を再開                       │
  │          ↓                                        │
  │ （VM Exit → VMM処理 → VMRESUME のループ）       │
  │          ↓                                        │
  │ VMXOFF: VMX モードを無効化                       │
  └──────────────────────────────────────────────────┘

  VMCS（Virtual Machine Control Structure）:
  ┌──────────────────────────────────────────────────┐
  │ Guest State Area:                                  │
  │ → ゲストのCPU状態（レジスタ、CR、MSR等）        │
  │ → VM Exit 時に自動保存                           │
  │ → VM Entry 時に自動復元                          │
  │                                                    │
  │ Host State Area:                                   │
  │ → ホスト（VMM）のCPU状態                        │
  │ → VM Exit 時に自動復元                           │
  │                                                    │
  │ VM-Execution Control Fields:                       │
  │ → VM Exit の条件を設定                           │
  │ → 例: 外部割り込み時にExit, I/Oポートアクセス時にExit │
  │ → ビットマップで細かく制御可能                   │
  │                                                    │
  │ VM-Exit Control Fields:                            │
  │ → VM Exit 時の動作を設定                         │
  │                                                    │
  │ VM-Entry Control Fields:                           │
  │ → VM Entry 時の動作を設定                        │
  │ → 例: 割り込みの注入                             │
  │                                                    │
  │ VM-Exit Information Fields:                        │
  │ → VM Exit の理由（Exit Reason）                  │
  │ → 例: External interrupt, I/O instruction        │
  │ → 例: EPT violation, CPUID                       │
  └──────────────────────────────────────────────────┘

  主な VM Exit の原因:
  ┌────────────────────────┬────────────────────────────┐
  │ 原因                   │ 説明                       │
  ├────────────────────────┼────────────────────────────┤
  │ External interrupt     │ 外部ハードウェア割り込み   │
  │ HLT                    │ CPU停止命令                │
  │ I/O instruction        │ IN/OUT命令                 │
  │ CR access              │ 制御レジスタのアクセス     │
  │ MSR read/write         │ モデル固有レジスタ操作     │
  │ CPUID                  │ CPU情報の取得              │
  │ EPT violation          │ メモリアクセス違反         │
  │ VMCALL                 │ ハイパーコール              │
  │ Task switch            │ タスクスイッチ              │
  │ INVLPG                 │ TLBエントリの無効化        │
  └────────────────────────┴────────────────────────────┘

  VM Exit のオーバーヘッド:
  → 1回のVM Exit: 数百〜数千CPU サイクル
  → VM Exit の削減が仮想化性能チューニングの鍵
  → Posted Interrupts: 割り込みのExit を削減
  → APIC Virtualization: APIC操作のExitを削減
```

### 4.2 AMD-V の特徴

```
AMD-V（AMD Virtualization / SVM）:

  Intel VT-x との比較:
  ┌─────────────────┬────────────────┬────────────────┐
  │ 機能            │ Intel VT-x     │ AMD-V (SVM)    │
  ├─────────────────┼────────────────┼────────────────┤
  │ 制御構造体      │ VMCS           │ VMCB           │
  │ VM Entry        │ VMLAUNCH/      │ VMRUN          │
  │                 │ VMRESUME       │                │
  │ VM Exit         │ VM Exit        │ #VMEXIT        │
  │ 有効化          │ VMXON          │ EFER.SVME      │
  │ ネステッド      │ VMCS shadowing │ ネイティブ対応│
  │ EPT/NPT         │ EPT            │ NPT (RVI)      │
  │ I/O制御         │ I/O bitmap     │ I/O permission │
  │ 割り込み        │ Posted Int.    │ AVIC           │
  │ メモリ暗号化    │ TDX (別機能)   │ SEV-SNP        │
  └─────────────────┴────────────────┴────────────────┘

  AMD SEV-SNP（Secure Encrypted Virtualization - SNP）:
  → VM のメモリを AES-128 で暗号化
  → 各VMに固有の暗号鍵
  → ハイパーバイザがゲストメモリを読めない
  → SNP: ページレベルの整合性保護を追加
  → クラウドでの Confidential VM の基盤
```

---

## 5. メモリ仮想化

### 5.1 アドレス変換

```
メモリ仮想化のアドレス空間:

  3段階のアドレス変換:
  ┌──────────────────────────────────────────────────┐
  │ ゲスト仮想アドレス (GVA)                          │
  │     ↓ ゲストのページテーブル                     │
  │ ゲスト物理アドレス (GPA)                          │
  │     ↓ ハイパーバイザの変換                       │
  │ ホスト物理アドレス (HPA)                          │
  │     ↓                                             │
  │ 物理メモリ (DRAM)                                 │
  └──────────────────────────────────────────────────┘

  方式1: シャドウページテーブル
  ┌──────────────────────────────────────────────────┐
  │ ハイパーバイザが GVA → HPA の直接マッピングを    │
  │ シャドウページテーブルとして管理                  │
  │                                                    │
  │ ゲスト PT (GVA→GPA) + VMM 変換 (GPA→HPA)       │
  │     → シャドウ PT (GVA→HPA)                     │
  │                                                    │
  │ 問題:                                              │
  │ - ゲストがページテーブルを変更するたびに          │
  │   シャドウテーブルの同期が必要                    │
  │ - VM Exit が頻発して性能低下                     │
  │ - メモリ消費が増大                                │
  │ - 実装が複雑                                      │
  └──────────────────────────────────────────────────┘

  方式2: EPT / NPT（ハードウェア支援）
  ┌──────────────────────────────────────────────────┐
  │ Intel EPT (Extended Page Tables):                 │
  │ AMD NPT (Nested Page Tables) / RVI:              │
  │                                                    │
  │ GVA → ゲストPT → GPA → EPT/NPT → HPA          │
  │                                                    │
  │ → ハードウェアが2段階のアドレス変換を自動実行   │
  │ → シャドウページテーブル不要                     │
  │ → VM Exit の大幅削減                             │
  │ → TLBミス時のペナルティ: 4レベル×4レベル=最大24回│
  │   のメモリアクセス（ページウォーク）              │
  │ → VPID（Virtual Processor ID）: TLBフラッシュを  │
  │   回避してVM切り替え時の性能を改善               │
  │                                                    │
  │ EPT の構造（4レベル）:                            │
  │ EPT PML4 → EPT PDPT → EPT PD → EPT PT → HPA  │
  │ → ゲストPTとEPTの両方を辿る必要があるが        │
  │   ハードウェアが自動で処理                        │
  └──────────────────────────────────────────────────┘
```

### 5.2 メモリ効率化技術

```
メモリ効率化:

  KSM（Kernel Same-page Merging）:
  ┌──────────────────────────────────────────────────┐
  │ 同一内容のメモリページを共有                      │
  │                                                    │
  │ 動作:                                              │
  │ 1. ksmd デーモンがページをスキャン               │
  │ 2. 同一内容のページを発見                         │
  │ 3. 1つのページを共有（Copy-on-Write）            │
  │ 4. 書き込み時にコピーを作成                       │
  │                                                    │
  │ 効果:                                              │
  │ - 同じOSのVM が多い場合: 30-50% のメモリ節約    │
  │ - デスクトップVDI環境で特に効果的                 │
  │ - OS のカーネル、共有ライブラリが共有対象         │
  │                                                    │
  │ 注意:                                              │
  │ - CPU オーバーヘッドがある（スキャン処理）       │
  │ - サイドチャネル攻撃のリスク（タイミング攻撃）   │
  │ - セキュリティ重視の環境では無効化を推奨         │
  │                                                    │
  │ 設定:                                              │
  │ echo 1 > /sys/kernel/mm/ksm/run                   │
  │ echo 1000 > /sys/kernel/mm/ksm/sleep_millisecs    │
  │ cat /sys/kernel/mm/ksm/pages_sharing              │
  └──────────────────────────────────────────────────┘

  メモリバルーニング（Ballooning）:
  ┌──────────────────────────────────────────────────┐
  │ ゲストOS内のバルーンドライバがメモリを「膨らます」│
  │ → ゲストが使用可能なメモリを一時的に減らす       │
  │ → ホストが回収して他のVMに割り当て               │
  │                                                    │
  │ 膨張（Inflate）:                                   │
  │ ゲストのメモリ: [■■■■■□□□□□]                    │
  │ バルーン膨張:   [■■■■■████□]                    │
  │ → ■=ゲスト使用, █=バルーン, □=空き              │
  │ → バルーン分のページをホストが回収               │
  │                                                    │
  │ 収縮（Deflate）:                                   │
  │ バルーン収縮:   [■■■■■■■□□□]                    │
  │ → ゲストが再びメモリを使用可能                   │
  │                                                    │
  │ 利点:                                              │
  │ - メモリのオーバーコミットを実現                  │
  │ - ゲストOSのメモリ管理を尊重                     │
  │ - ゲストが不要なページを swap out する            │
  │                                                    │
  │ virtio-balloon ドライバで実装                     │
  └──────────────────────────────────────────────────┘

  Huge Pages:
  ┌──────────────────────────────────────────────────┐
  │ 通常: 4KB ページ → TLBエントリを大量に消費      │
  │ Huge Pages: 2MB or 1GB ページ                    │
  │ → TLB ミスを大幅に削減                           │
  │ → 特にメモリ集約型のVMで効果的                   │
  │                                                    │
  │ Transparent Huge Pages (THP):                     │
  │ → カーネルが自動的に Huge Pages を使用           │
  │ → VM にも自動適用                                │
  │                                                    │
  │ Static Huge Pages:                                │
  │ → 事前に予約                                     │
  │ → QEMU: -mem-path /dev/hugepages -mem-prealloc   │
  │ → より確実だがメモリの柔軟性が低下               │
  │                                                    │
  │ 設定:                                              │
  │ # Huge Pages の予約                               │
  │ echo 1024 > /sys/kernel/mm/hugepages/             │
  │   hugepages-2048kB/nr_hugepages                   │
  │ # → 1024 × 2MB = 2GB を予約                     │
  │                                                    │
  │ # 確認                                            │
  │ cat /proc/meminfo | grep Huge                     │
  │ # HugePages_Total:    1024                        │
  │ # HugePages_Free:     1024                        │
  │ # Hugepagesize:       2048 kB                     │
  └──────────────────────────────────────────────────┘
```

---

## 6. I/O仮想化

### 6.1 I/O仮想化の方式

```
I/O仮想化の3つの方式:

  1. エミュレーション（全仮想化）:
  ┌──────────────────────────────────────────────────┐
  │ ゲスト → I/O命令 → VM Exit → VMM がエミュレート│
  │                                                    │
  │ 例: QEMU が仮想的な NE2000 NIC をエミュレート    │
  │ → ゲストは標準のドライバを使用可能               │
  │ → 非常に遅い（VM Exit が大量発生）               │
  │ → 互換性は最高                                   │
  └──────────────────────────────────────────────────┘

  2. 準仮想化 I/O（Virtio）:
  ┌──────────────────────────────────────────────────┐
  │ ゲスト → Virtio ドライバ → 共有メモリリング     │
  │                → VM Exit 最小 → ホスト処理      │
  │                                                    │
  │ Virtio の仕組み:                                  │
  │ ┌─────────────────────────────────────┐           │
  │ │ ゲスト                               │           │
  │ │ ┌─────────────┐                     │           │
  │ │ │ Virtio Driver│                     │           │
  │ │ └──────┬──────┘                     │           │
  │ │        ↓                             │           │
  │ │ ┌──────────────┐                    │           │
  │ │ │ Virtqueue    │ ← リングバッファ   │           │
  │ │ │ (desc/avail/ │                    │           │
  │ │ │  used ring)  │                    │           │
  │ │ └──────┬──────┘                     │           │
  │ └────────┼────────────────────────────┘           │
  │          ↓ 共有メモリ                             │
  │ ┌────────┼────────────────────────────┐           │
  │ │ ホスト ↓                             │           │
  │ │ ┌──────────────┐                    │           │
  │ │ │ Virtio Backend│                    │           │
  │ │ │ (vhost-net等) │                    │           │
  │ │ └──────────────┘                    │           │
  │ └─────────────────────────────────────┘           │
  │                                                    │
  │ Virtio デバイスの種類:                            │
  │ - virtio-net: ネットワーク                       │
  │ - virtio-blk: ブロックストレージ                 │
  │ - virtio-scsi: SCSI ストレージ                   │
  │ - virtio-serial: シリアル通信                    │
  │ - virtio-balloon: メモリバルーニング             │
  │ - virtio-gpu: グラフィックス                     │
  │ - virtio-fs: ファイルシステム共有                │
  │ - virtio-vsock: ホスト-ゲスト通信               │
  │                                                    │
  │ vhost: Virtio バックエンドをカーネル空間に移動   │
  │ → QEMU のユーザー空間オーバーヘッドを削減       │
  │ → vhost-net: カーネル内のネットワークバックエンド│
  │ → vhost-user: DPDKなどのユーザー空間バックエンド│
  └──────────────────────────────────────────────────┘

  3. デバイスパススルー（SR-IOV）:
  ┌──────────────────────────────────────────────────┐
  │ SR-IOV（Single Root I/O Virtualization）:         │
  │                                                    │
  │ 物理デバイス（NIC）:                              │
  │ ┌─────────────────────────────────┐               │
  │ │ PF (Physical Function)          │               │
  │ │ ┌─────┐ ┌─────┐ ┌─────┐       │               │
  │ │ │ VF1 │ │ VF2 │ │ VF3 │ ...   │               │
  │ │ └──┬──┘ └──┬──┘ └──┬──┘       │               │
  │ └────┼───────┼───────┼───────────┘               │
  │      │       │       │                            │
  │   ┌──┴──┐ ┌──┴──┐ ┌──┴──┐                       │
  │   │ VM1 │ │ VM2 │ │ VM3 │                       │
  │   └─────┘ └─────┘ └─────┘                       │
  │                                                    │
  │ PF (Physical Function): 物理デバイスの完全な機能 │
  │ VF (Virtual Function): 軽量な仮想デバイス        │
  │                                                    │
  │ → ハイパーバイザをバイパスして直接アクセス       │
  │ → ほぼネイティブのI/O性能                        │
  │ → IOMMU（Intel VT-d / AMD-Vi）でDMAを保護       │
  │ → ライブマイグレーションが困難（デバイス依存）   │
  │                                                    │
  │ 性能比較:                                         │
  │ ┌────────────┬──────────┬──────────┐              │
  │ │ 方式       │ レイテンシ│ スループット│           │
  │ ├────────────┼──────────┼──────────┤              │
  │ │ エミュレート│ 高        │ 低         │           │
  │ │ Virtio     │ 中        │ 高         │           │
  │ │ SR-IOV     │ 低        │ 非常に高   │           │
  │ │ ネイティブ │ 最低      │ 最高       │           │
  │ └────────────┴──────────┴──────────┘              │
  └──────────────────────────────────────────────────┘
```

---

## 7. KVM/QEMU の実践

### 7.1 KVM の基本構成

```
KVM + QEMU のアーキテクチャ:

  ┌──────────────────────────────────────────────────┐
  │ ゲスト OS                                         │
  │ ┌──────────────────┐                              │
  │ │ アプリケーション  │                              │
  │ │ ゲストカーネル    │                              │
  │ │ virtio ドライバ   │                              │
  │ └────────┬─────────┘                              │
  │          │                                         │
  │ QEMU プロセス（ユーザー空間）                     │
  │ ┌────────┴─────────┐                              │
  │ │ デバイスエミュレーション│                       │
  │ │ (NIC, ディスク, VGA等) │                        │
  │ │ ioctl(KVM_RUN)         │                        │
  │ └────────┬───────────────┘                        │
  │          │                                         │
  │ KVM カーネルモジュール                             │
  │ ┌────────┴─────────┐                              │
  │ │ /dev/kvm          │                              │
  │ │ VMCS管理          │                              │
  │ │ VM Entry/Exit処理 │                              │
  │ │ EPT管理           │                              │
  │ └────────┬─────────┘                              │
  │          │                                         │
  │ ハードウェア（VT-x/AMD-V + VT-d/AMD-Vi）         │
  └──────────────────────────────────────────────────┘

  各コンポーネントの役割:
  KVM: CPU仮想化、メモリ仮想化（EPT/NPT）
  QEMU: デバイスエミュレーション、VM管理
  libvirt: VM管理API（virsh, virt-manager のバックエンド）
```

### 7.2 VM の作成と管理

```bash
# KVM の確認
lsmod | grep kvm
# kvm_intel     xxx  0
# kvm           xxx  1 kvm_intel

# CPU が仮想化支援をサポートしているか確認
grep -E 'vmx|svm' /proc/cpuinfo

# QEMU で直接 VM を起動
qemu-system-x86_64 \
  -enable-kvm \
  -cpu host \
  -smp 4 \
  -m 4096 \
  -drive file=disk.qcow2,if=virtio,format=qcow2 \
  -netdev user,id=net0,hostfwd=tcp::2222-:22 \
  -device virtio-net-pci,netdev=net0 \
  -display none \
  -daemonize

# ディスクイメージの作成
qemu-img create -f qcow2 disk.qcow2 50G
# qcow2: Copy-on-Write、スナップショット対応、シンプロビジョニング

# ディスクイメージの情報
qemu-img info disk.qcow2
# image: disk.qcow2
# file format: qcow2
# virtual size: 50 GiB
# disk size: 196 MiB  ← 実際のディスク使用量

# スナップショット
qemu-img snapshot -c snap1 disk.qcow2    # 作成
qemu-img snapshot -l disk.qcow2          # 一覧
qemu-img snapshot -a snap1 disk.qcow2    # 復元
qemu-img snapshot -d snap1 disk.qcow2    # 削除
```

### 7.3 libvirt / virsh による管理

```bash
# libvirt でのVM管理（推奨）

# VM の一覧
virsh list --all

# VM の起動/停止
virsh start myvm
virsh shutdown myvm      # ゲストOSに正常終了を要求
virsh destroy myvm       # 強制停止（電源断に相当）
virsh reboot myvm

# VM の作成（XML定義）
virsh define myvm.xml
virsh create myvm.xml    # 定義して即起動

# VM の情報
virsh dominfo myvm
virsh vcpuinfo myvm
virsh domblklist myvm
virsh domiflist myvm

# コンソール接続
virsh console myvm

# スナップショット
virsh snapshot-create-as myvm snap1 "Initial snapshot"
virsh snapshot-list myvm
virsh snapshot-revert myvm snap1
virsh snapshot-delete myvm snap1

# リソースの動的変更
virsh setmem myvm 8G --live           # メモリ変更（ライブ）
virsh setvcpus myvm 8 --live          # CPU数変更（ライブ）

# VM のモニタリング
virsh domstats myvm                    # 統計情報
virt-top                               # リアルタイムモニタ
```

```xml
<!-- VM 定義ファイルの例 (myvm.xml) -->
<domain type='kvm'>
  <name>myvm</name>
  <memory unit='GiB'>4</memory>
  <vcpu placement='static'>4</vcpu>

  <cpu mode='host-passthrough' check='none' migratable='on'/>

  <os>
    <type arch='x86_64' machine='q35'>hvm</type>
    <boot dev='hd'/>
  </os>

  <features>
    <acpi/>
    <apic/>
  </features>

  <devices>
    <!-- Virtio ディスク -->
    <disk type='file' device='disk'>
      <driver name='qemu' type='qcow2' cache='writeback' discard='unmap'/>
      <source file='/var/lib/libvirt/images/myvm.qcow2'/>
      <target dev='vda' bus='virtio'/>
    </disk>

    <!-- Virtio ネットワーク -->
    <interface type='network'>
      <source network='default'/>
      <model type='virtio'/>
    </interface>

    <!-- VNC コンソール -->
    <graphics type='vnc' port='-1' autoport='yes'/>

    <!-- Virtio メモリバルーン -->
    <memballoon model='virtio'>
      <stats period='10'/>
    </memballoon>

    <!-- virtio-serial（ゲストエージェント通信用） -->
    <channel type='unix'>
      <target type='virtio' name='org.qemu.guest_agent.0'/>
    </channel>
  </devices>
</domain>
```

---

## 8. ライブマイグレーション

### 8.1 ライブマイグレーションの仕組み

```
ライブマイグレーション:
  VMを停止せずに別の物理サーバーに移動する技術

  Pre-copy マイグレーション（最も一般的）:
  ┌──────────────────────────────────────────────────┐
  │ Phase 1: メモリの一括コピー                       │
  │   → 全メモリページをコピー先に転送               │
  │   → VM は稼働中のまま                            │
  │                                                    │
  │ Phase 2: 差分コピー（反復）                       │
  │   → Phase 1 中に変更されたページ（dirty pages）  │
  │     を転送                                        │
  │   → 再び変更されたページを転送（繰り返し）       │
  │   → 差分が十分小さくなるまで繰り返す             │
  │                                                    │
  │ Phase 3: 最終同期（Stop-and-Copy）                │
  │   → VM を一時停止                                │
  │   → 残りの差分ページを転送                       │
  │   → CPU 状態、デバイス状態を転送                 │
  │   → 移行先で VM を再開                           │
  │   → ダウンタイム: 数十ms〜数百ms                 │
  │                                                    │
  │ Phase 4: ネットワーク切り替え                     │
  │   → ARP の更新（RARP パケット送信）              │
  │   → 外部からは同じIPで到達可能                   │
  └──────────────────────────────────────────────────┘

  Post-copy マイグレーション:
  ┌──────────────────────────────────────────────────┐
  │ 1. CPU状態とデバイス状態を先に転送               │
  │ 2. VM を移行先で即座に起動                       │
  │ 3. メモリページはアクセス時にオンデマンドで転送  │
  │    → userfaultfd でページフォールトを処理        │
  │                                                    │
  │ 利点: ダウンタイムが非常に短い                    │
  │ 欠点: ページフォールトによる性能低下              │
  │       ネットワーク断で VM がクラッシュする        │
  │ → Pre-copy と Post-copy のハイブリッドも存在     │
  └──────────────────────────────────────────────────┘

  ライブマイグレーションの要件:
  ┌──────────────────────────────────────────────────┐
  │ 1. 共有ストレージ（NFS, Ceph, SAN）              │
  │    → ディスクイメージが両方のホストからアクセス可能│
  │    → ストレージマイグレーションの場合は不要      │
  │                                                    │
  │ 2. 同一のCPU機能（または互換性）                  │
  │    → cpu mode='host-model' で互換性確保          │
  │    → QEMU の CPU機能マスクで調整                 │
  │                                                    │
  │ 3. 十分なネットワーク帯域幅                      │
  │    → 10Gbps以上推奨                              │
  │    → dirty page rate < 転送速度 が必要           │
  │                                                    │
  │ 4. 同一のlibvirt / QEMU バージョン               │
  │    → プロトコル互換性の確保                      │
  └──────────────────────────────────────────────────┘
```

```bash
# ライブマイグレーションの実行

# virsh によるマイグレーション
virsh migrate --live --verbose myvm \
  qemu+ssh://dest-host/system \
  --migrateuri tcp://dest-host:49152

# 帯域幅制限付きマイグレーション
virsh migrate --live myvm \
  qemu+ssh://dest-host/system \
  --bandwidth 500  # 500 MiB/s

# 圧縮付きマイグレーション（帯域幅の節約）
virsh migrate --live myvm \
  qemu+ssh://dest-host/system \
  --comp-methods xbzrle

# ストレージマイグレーション（共有ストレージ不要）
virsh migrate --live --copy-storage-all myvm \
  qemu+ssh://dest-host/system

# マイグレーションの進捗確認
virsh domjobinfo myvm

# マイグレーションのキャンセル
virsh domjobabort myvm
```

---

## 9. パフォーマンスチューニング

### 9.1 CPU のチューニング

```bash
# CPU ピンニング（vCPU を物理CPUコアに固定）
virsh vcpupin myvm 0 2    # vCPU 0 → 物理コア 2
virsh vcpupin myvm 1 3    # vCPU 1 → 物理コア 3

# NUMA ノードへの配置
virsh numatune myvm --nodeset 0 --mode strict
# → NUMA ノード 0 のメモリのみ使用

# CPU アフィニティの確認
virsh vcpuinfo myvm

# エミュレータスレッドのピンニング
virsh emulatorpin myvm 0-1
# → QEMU のエミュレータスレッドをコア0-1に固定

# IOスレッドのピンニング
virsh iothreadpin myvm 1 4
# → IOスレッド1をコア4に固定
```

### 9.2 メモリのチューニング

```bash
# Huge Pages の使用
# VM 定義ファイルに追加:
# <memoryBacking>
#   <hugepages>
#     <page size='2048' unit='KiB'/>
#   </hugepages>
# </memoryBacking>

# NUMA-aware メモリ配置
# <numatune>
#   <memory mode='strict' nodeset='0'/>
# </numatune>
# <cpu>
#   <numa>
#     <cell id='0' cpus='0-3' memory='4' unit='GiB'/>
#   </numa>
# </cpu>

# メモリロック（スワップアウト防止）
# <memoryBacking>
#   <locked/>
# </memoryBacking>
```

### 9.3 ストレージのチューニング

```
ストレージチューニング:

  ディスクキャッシュモード:
  ┌──────────────┬──────────────────────────────────┐
  │ none         │ ホストキャッシュなし。直接I/O     │
  │              │ → データの一貫性が最も高い        │
  │              │ → 推奨（ゲストにキャッシュ任せ）  │
  │ writethrough │ 読取キャッシュあり、書込は即座    │
  │              │ → 安全だが書き込みが遅い          │
  │ writeback    │ 読取/書込キャッシュあり           │
  │              │ → 高速だがデータ損失リスクあり    │
  │ unsafe       │ すべてのfsync を無視              │
  │              │ → テスト環境のみ（データ損失大）  │
  │ directsync   │ 直接I/O + 同期書き込み           │
  │              │ → 最も安全だが最も遅い            │
  └──────────────┴──────────────────────────────────┘

  I/Oスケジューラの設定:
  ┌──────────────────────────────────────────────────┐
  │ ホスト側:                                         │
  │   SSD: none (noop) が最適                        │
  │   HDD: mq-deadline が最適                        │
  │                                                    │
  │ ゲスト側:                                         │
  │   virtio-blk: none (noop) が最適                 │
  │   → ホスト側でスケジューリングするため          │
  └──────────────────────────────────────────────────┘

  ディスクフォーマット:
  ┌──────────────┬──────────────────────────────────┐
  │ qcow2        │ スナップショット、圧縮、暗号化対応│
  │              │ → 柔軟性が高い、やや遅い          │
  │ raw          │ シンプル。最高性能                  │
  │              │ → 動的サイズ変更不可               │
  │ qcow2 +     │ qcow2 の事前割当                   │
  │ preallocation│ → raw に近い性能                   │
  └──────────────┴──────────────────────────────────┘
```

---

## 10. ネステッド仮想化

```
ネステッド仮想化（Nested Virtualization）:
  VM の中で VM を実行する技術

  ┌──────────────────────────────────────────────────┐
  │ L0: 物理ハードウェア + ホスト KVM                │
  │   └── L1: ゲスト VM（この中でKVMを実行）        │
  │         └── L2: ネストされた VM                  │
  │                                                    │
  │ 用途:                                              │
  │ - クラウド上での仮想化テスト・開発               │
  │ - CI/CD パイプラインでの VM テスト               │
  │ - KVM/QEMU の開発・デバッグ                      │
  │ - ハイパーバイザのセキュリティテスト              │
  │ - 教育・トレーニング環境                          │
  │                                                    │
  │ 性能:                                              │
  │ → L1 の約 60-80% の性能                          │
  │ → VM Exit のネストにより追加オーバーヘッド       │
  │ → VMCS shadowing (Intel) でオーバーヘッド削減    │
  └──────────────────────────────────────────────────┘
```

```bash
# ネステッド仮想化の有効化

# Intel の場合
cat /sys/module/kvm_intel/parameters/nested
# N → 無効

# 有効化（一時的）
sudo modprobe -r kvm_intel
sudo modprobe kvm_intel nested=1

# 永続的に有効化
echo "options kvm_intel nested=1" | \
  sudo tee /etc/modprobe.d/kvm-nested.conf

# AMD の場合
echo "options kvm_amd nested=1" | \
  sudo tee /etc/modprobe.d/kvm-nested.conf

# VM の CPU設定でVMX/SVMを公開
# <cpu mode='host-passthrough'>
#   <feature policy='require' name='vmx'/>
# </cpu>
```

---

## 11. クラウドの仮想化

### 11.1 AWS Nitro System

```
AWS Nitro System:
  専用ハードウェアでネットワーク/ストレージ/セキュリティを
  メインCPUからオフロード

  Nitro の構成:
  ┌──────────────────────────────────────────────────┐
  │ EC2 インスタンス                                  │
  │ ┌──────────────────────────────────┐              │
  │ │ ゲスト VM (Customer Workload)    │              │
  │ │ → CPU のほぼ100%を使用可能      │              │
  │ └──────────────────────────────────┘              │
  │                                                    │
  │ Nitro Cards（専用ASIC）:                          │
  │ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
  │ │ Nitro    │ │ Nitro    │ │ Nitro    │          │
  │ │ Network  │ │ EBS      │ │ Security │          │
  │ │ Card     │ │ Card     │ │ Chip     │          │
  │ └──────────┘ └──────────┘ └──────────┘          │
  │ → VPC、EBS、暗号化をハードウェアで処理          │
  │ → ホストCPUリソースの消費なし                    │
  │                                                    │
  │ Nitro Hypervisor:                                 │
  │ → KVM ベースの軽量ハイパーバイザ                 │
  │ → 従来の Xen ハイパーバイザを置き換え            │
  │ → CPU とメモリの仮想化のみ担当                   │
  │ → I/O は Nitro Cards にオフロード               │
  │                                                    │
  │ Nitro Enclaves:                                   │
  │ → 高度に隔離された計算環境                       │
  │ → ネットワーク接続なし、永続ストレージなし       │
  │ → vsock のみで親VMと通信                        │
  │ → 暗号鍵の管理、機密データの処理に使用          │
  └──────────────────────────────────────────────────┘
```

### 11.2 クラウドインスタンスの仕組み

```
クラウドインスタンスのライフサイクル:

  ユーザー → API → コントロールプレーン
                         │
                    ┌─────┴─────┐
                    │ スケジューラ│
                    └─────┬─────┘
                          │ VMの配置先を決定
                    ┌─────┴─────┐
                    │ 物理サーバー│
                    │ KVM + QEMU │
                    │ ┌───┐┌───┐│
                    │ │VM1││VM2││
                    │ └───┘└───┘│
                    └───────────┘

  スケジューラの配置アルゴリズム:
  ┌──────────────────────────────────────────────────┐
  │ 1. リソースフィルタリング:                        │
  │    → CPU, メモリ, ストレージの要件を満たすホスト│
  │                                                    │
  │ 2. アフィニティ/アンチアフィニティ:              │
  │    → 特定のVMを同じ/異なるホストに配置           │
  │                                                    │
  │ 3. NUMA最適化:                                    │
  │    → NUMAトポロジに基づく最適配置                │
  │                                                    │
  │ 4. 可用性ゾーン:                                  │
  │    → 障害ドメインの分散                           │
  │                                                    │
  │ 5. コスト最適化:                                  │
  │    → ビンパッキング（効率的なリソース充填）      │
  │    → or スプレッド（分散配置）                   │
  └──────────────────────────────────────────────────┘
```

---

## 実践演習

### 演習1: [基礎] -- 仮想化支援の確認

```bash
# CPU の仮想化支援を確認
grep -E 'vmx|svm' /proc/cpuinfo | head -1
# flags : ... vmx ...  → Intel VT-x 対応

# KVM モジュールの確認
lsmod | grep kvm
# kvm_intel     xxxxx  0
# kvm           xxxxx  1 kvm_intel

# /dev/kvm の確認
ls -la /dev/kvm
# crw-rw---- 1 root kvm 10, 232 Jan  1 00:00 /dev/kvm
```

### 演習2: [基礎] -- QEMU で VM を起動

```bash
# ディスクイメージの作成
qemu-img create -f qcow2 test.qcow2 10G

# Ubuntu Server の ISO でインストール
qemu-system-x86_64 \
  -enable-kvm \
  -cpu host \
  -smp 2 \
  -m 2048 \
  -drive file=test.qcow2,if=virtio \
  -cdrom ubuntu-server.iso \
  -boot d \
  -vnc :0

# インストール後の起動
qemu-system-x86_64 \
  -enable-kvm \
  -cpu host \
  -smp 2 \
  -m 2048 \
  -drive file=test.qcow2,if=virtio \
  -netdev user,id=net0,hostfwd=tcp::2222-:22 \
  -device virtio-net-pci,netdev=net0 \
  -nographic

# SSH 接続
ssh -p 2222 user@localhost
```

### 演習3: [応用] -- virsh による VM 管理

```bash
# VM の作成
virt-install \
  --name testvm \
  --ram 2048 \
  --vcpus 2 \
  --disk size=20,format=qcow2 \
  --os-variant ubuntu22.04 \
  --cdrom /path/to/ubuntu-22.04.iso \
  --network network=default \
  --graphics vnc

# スナップショットの管理
virsh snapshot-create-as testvm clean-install "Fresh install"
virsh snapshot-list testvm
virsh snapshot-revert testvm clean-install

# リソースの動的変更
virsh setmem testvm 4G --live
virsh setvcpus testvm 4 --live

# VM の統計情報
virsh domstats testvm
virt-top
```

### 演習4: [応用] -- パフォーマンスチューニング

```bash
# CPU ピンニング
virsh vcpupin testvm 0 0
virsh vcpupin testvm 1 1

# Huge Pages の設定
echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
# VM 定義に hugepages を追加

# I/O チューニング
# ディスクのキャッシュモードを none に変更
virsh attach-disk testvm /path/to/disk.qcow2 vdb \
  --driver qemu --subdriver qcow2 --cache none

# パフォーマンス測定
# ゲスト内で
fio --name=seqwrite --rw=write --bs=4k --size=1G --numjobs=4
sysbench cpu --threads=4 run
iperf3 -c host-ip
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 完全仮想化 | ゲストOS無修正。バイナリ変換 or HW支援 |
| 準仮想化 | ゲストOS修正。ハイパーコールで高性能 |
| Type 1 | ベアメタル。KVM, ESXi, Xen。クラウドの基盤 |
| Type 2 | ホスト型。VirtualBox, Parallels。開発環境 |
| VT-x/AMD-V | ハードウェア支援。VMX Root/Non-Root モード |
| EPT/NPT | ハードウェア支援メモリ仮想化。シャドウPT不要 |
| Virtio | 準仮想化I/O。ネットワーク、ストレージの高速化 |
| SR-IOV | デバイスの直接パススルー。ほぼネイティブI/O性能 |
| ライブマイグレーション | Pre-copy方式。ダウンタイム数十ms |
| KSM | 同一ページ共有。メモリ効率化 |
| Nitro | AWS独自。HWオフロード。CPU100%をゲストに |
| ネステッド仮想化 | VM内でVM。開発・テスト用途 |

---

## 次に読むべきガイド
→ [[01-containers.md]] -- コンテナ技術

---

## 参考文献
1. Portnoy, M. "Virtualization Essentials." 2nd Ed, Sybex, 2016.
2. Popek, G. J. & Goldberg, R. P. "Formal Requirements for Virtualizable Third Generation Architectures." Communications of the ACM, 1974.
3. Agesen, O. et al. "Software and Hardware Techniques for x86 Virtualization." VMware Technical Report, 2012.
4. Kivity, A. et al. "kvm: the Linux Virtual Machine Monitor." Proceedings of the Linux Symposium, 2007.
5. Adams, K. & Agesen, O. "A Comparison of Software and Hardware Techniques for x86 Virtualization." ASPLOS, 2006.
6. Amazon. "AWS Nitro System." AWS Documentation, 2024.
7. Red Hat. "Virtualization Deployment and Administration Guide." RHEL Documentation, 2024.
8. QEMU Project. "QEMU Documentation." qemu.org, 2024.
9. Intel. "Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 3C: System Programming Guide, Part 3." Chapter 23-34 (VMX), 2024.
10. Habib, I. "Virtualization with KVM." Linux Journal, 2008.

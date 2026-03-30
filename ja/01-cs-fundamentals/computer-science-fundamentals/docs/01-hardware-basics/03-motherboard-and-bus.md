# マザーボードとバス

> マザーボードはコンピュータの「神経系」であり、全てのコンポーネント間の通信を司る。

## この章で学ぶこと

- [ ] マザーボードの主要コンポーネントと役割を説明できる
- [ ] バスアーキテクチャの進化を理解する
- [ ] ブートプロセスの各段階を説明できる
- [ ] PCIeの詳細仕様とレーン配分を理解する
- [ ] USB規格の進化と実務での選定基準を習得する
- [ ] サーバーアーキテクチャとの違いを説明できる

## 前提知識


---

## 1. マザーボードの構成要素

```
マザーボードのレイアウト（概念図）:

  ┌──────────────────────────────────────────────────┐
  │  ┌──────────┐          ┌──────────────────────┐ │
  │  │ CPU      │←────────→│ メモリスロット       │ │
  │  │ ソケット │ メモリバス │ DIMM1 DIMM2 DIMM3  │ │
  │  └────┬─────┘          └──────────────────────┘ │
  │       │                                          │
  │       │ PCIe x16                                 │
  │       ▼                                          │
  │  ┌──────────────────────────────────┐            │
  │  │       PCH (Platform Controller Hub)│           │
  │  │  ┌─────────────────────────────┐  │           │
  │  │  │ PCIe x4 → NVMe SSD スロット │  │           │
  │  │  │ PCIe x16 → GPU スロット     │  │           │
  │  │  │ SATA → HDD/SSD              │  │           │
  │  │  │ USB 3.x/4.0 コントローラ    │  │           │
  │  │  │ Ethernet コントローラ        │  │           │
  │  │  │ Audio コントローラ           │  │           │
  │  │  │ Wi-Fi/Bluetooth              │  │           │
  │  │  └─────────────────────────────┘  │           │
  │  └──────────────────────────────────┘            │
  │                                                   │
  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
  │  │ BIOS/UEFI│  │ 電源      │  │ I/Oポート    │  │
  │  │ (SPI     │  │ コネクタ  │  │ USB,HDMI,    │  │
  │  │  Flash)  │  │ (ATX)     │  │ Ethernet...  │  │
  │  └──────────┘  └──────────┘  └──────────────┘  │
  └──────────────────────────────────────────────────┘
```

### 1.1 マザーボードの主要コンポーネント詳解

```
各コンポーネントの詳細:

  ■ CPUソケット
    - Intel LGA (Land Grid Array): ピンがソケット側
      LGA1700 (12th-14th Gen), LGA1851 (Arrow Lake)
    - AMD PGA (Pin Grid Array): ピンがCPU側（AM4まで）
    - AMD LGA: AM5からLGAに移行
    - サーバー: LGA4094 (AMD SP5), LGA4677 (Intel)

    ソケットの互換性:
    ┌───────────────────────────────────────────┐
    │ プラットフォーム │ ソケット │ 世代            │
    │──────────────────│──────────│─────────────────│
    │ Intel Desktop    │ LGA1700  │ 12th-14th Gen   │
    │ Intel Desktop    │ LGA1851  │ Arrow Lake+     │
    │ AMD Desktop      │ AM4      │ Ryzen 1000-5000 │
    │ AMD Desktop      │ AM5      │ Ryzen 7000+     │
    │ Intel Server     │ LGA4677  │ Sapphire Rapids+ │
    │ AMD Server       │ SP5      │ EPYC 9004+      │
    └───────────────────────────────────────────┘

  ■ メモリスロット（DIMMスロット）
    - 通常2本または4本（デスクトップ）
    - サーバーでは8-12本/CPU
    - DDR5: 288ピン、デュアルチャネル（各チャネル32ビット）
    - DIMM種類:
      UDIMM: アンバッファード（デスクトップ）
      RDIMM: レジスタード（サーバー、ECC対応）
      LRDIMM: ロードリデュースト（大容量サーバー）
      SO-DIMM: ノートPC用（小型）

  ■ VRM (Voltage Regulator Module)
    - CPUに安定した電圧を供給
    - フェーズ数が多いほど安定（高性能マザーボードは16-20フェーズ）
    - オーバークロック時に特に重要
    - MOSFETの品質がVRMの品質を決定

    VRMの構成:
    ┌──────────────────────────────────────────┐
    │ 12V (ATX電源) → VRM → 1.1V (CPU VCore)  │
    │                                           │
    │ ┌─────┐ ┌─────┐ ┌─────┐    ┌──────┐  │
    │ │Phase│ │Phase│ │Phase│... │ CPU   │  │
    │ │ 1   │ │ 2   │ │ 3   │    │       │  │
    │ └─────┘ └─────┘ └─────┘    └──────┘  │
    │ PWMコントローラが各フェーズを交互に動作   │
    │ → 電流の安定化、発熱の分散               │
    └──────────────────────────────────────────┘

  ■ SPI フラッシュ（BIOS/UEFI ROM）
    - 容量: 16-32MB（UEFI + マイクロコード）
    - SPI (Serial Peripheral Interface) バスで接続
    - 電源投入時に最初に読まれるチップ
    - Dual BIOSの場合は2チップ搭載（障害対策）
```

### 1.2 フォームファクタ

```
マザーボードのフォームファクタ:

  ┌────────────────────────────────────────────────────┐
  │ ATX (305 × 244 mm)                                 │
  │ ┌──────────────────────────────────────────────┐  │
  │ │                                              │  │
  │ │  PCIe x16 × 2-3                             │  │
  │ │  M.2 スロット × 2-4                          │  │
  │ │  DIMM × 4                                    │  │
  │ │  SATA × 4-8                                  │  │
  │ │  USB ヘッダ × 多数                           │  │
  │ │                                              │  │
  │ └──────────────────────────────────────────────┘  │
  │ → 最も一般的、拡張性最高                           │
  └────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────┐
  │ Micro-ATX (244 × 244 mm)                │
  │ ┌──────────────────────────────────┐    │
  │ │  PCIe x16 × 1-2                  │    │
  │ │  M.2 × 1-2                       │    │
  │ │  DIMM × 2-4                      │    │
  │ │  SATA × 4-6                      │    │
  │ └──────────────────────────────────┘    │
  │ → コスパ重視、程よいサイズ              │
  └──────────────────────────────────────────┘

  ┌────────────────────────────────┐
  │ Mini-ITX (170 × 170 mm)       │
  │ ┌──────────────────────┐      │
  │ │  PCIe x16 × 1        │      │
  │ │  M.2 × 1-2           │      │
  │ │  DIMM × 2            │      │
  │ │  SATA × 2-4          │      │
  │ └──────────────────────┘      │
  │ → 小型PC、HTPC向け            │
  └────────────────────────────────┘

  サーバー向け:
  ┌────────────────────────────────────────────────────────┐
  │ E-ATX (305 × 330 mm)                                   │
  │ → デュアルソケット対応、DIMM × 8-16                     │
  │                                                         │
  │ EEB (305 × 330 mm)                                     │
  │ → サーバー標準、多数のPCIeスロット                       │
  │                                                         │
  │ OCP (Open Compute Project)                              │
  │ → データセンター向けオープン規格                          │
  └────────────────────────────────────────────────────────┘
```

---

## 2. バスの種類と進化

### 2.1 バスの歴史

| 規格 | 年代 | 帯域幅 | 特徴 |
|------|------|--------|------|
| ISA | 1981 | 8 MB/s | IBM PC初期のバス |
| PCI | 1992 | 133 MB/s | 共有バス、プラグ&プレイ |
| AGP | 1997 | 2.1 GB/s | GPU専用バス |
| PCI Express 1.0 | 2003 | 250 MB/s/lane | ポイントtoポイント、レーン制 |
| PCIe 2.0 | 2007 | 500 MB/s/lane | 帯域2倍 |
| PCIe 3.0 | 2010 | 985 MB/s/lane | 128b/130bエンコーディング |
| PCIe 4.0 | 2017 | 1,969 MB/s/lane | NVMe SSDの標準 |
| PCIe 5.0 | 2019 | 3,938 MB/s/lane | サーバー、ハイエンド |
| PCIe 6.0 | 2022 | 7,877 MB/s/lane | PAM4、FEC必須 |
| PCIe 7.0 | 2025 | 15,754 MB/s/lane | 策定中 |

### 2.2 PCIe の構造

```
PCIe レーン構成:

  PCIe x1:  ──→  1レーン  =  3.9 GB/s (PCIe 5.0)
  PCIe x4:  ────→ 4レーン  = 15.8 GB/s (NVMe SSD)
  PCIe x8:  ────────→ 8レーン  = 31.5 GB/s
  PCIe x16: ────────────────→ 16レーン = 63.0 GB/s (GPU)

  各レーンは独立した送受信ペア（差動信号）:
  ┌──────┐         ┌──────┐
  │ CPU  │ ──TX──→ │ GPU  │  送信
  │      │ ←──RX── │      │  受信
  └──────┘         └──────┘
  → 全二重通信（同時送受信）
```

### 2.3 PCIe の詳細技術

```
PCIeのプロトコル層:

  ┌─────────────────────────────────────┐
  │ トランザクション層 (TLP)             │
  │ - メモリ読み/書き、I/O、設定         │
  │ - パケットベースの通信               │
  │ - フロー制御（クレジットベース）      │
  ├─────────────────────────────────────┤
  │ データリンク層 (DLLP)                │
  │ - CRCによるエラー検出                │
  │ - ACK/NAKによる再送制御              │
  │ - フロー制御情報の送受信             │
  ├─────────────────────────────────────┤
  │ 物理層                               │
  │ - 差動信号ペア                       │
  │ - エンコーディング                   │
  │   PCIe 1.0-2.0: 8b/10b (20%オーバーヘッド) │
  │   PCIe 3.0-5.0: 128b/130b (1.5%オーバーヘッド) │
  │   PCIe 6.0-7.0: PAM4 + FEC          │
  │ - レーン幅: x1, x2, x4, x8, x16    │
  └─────────────────────────────────────┘

PCIe世代ごとの帯域幅計算:

  PCIe 3.0 x4 (NVMe SSD):
    転送レート: 8 GT/s × 4レーン = 32 GT/s
    エンコーディング: 128b/130b
    実効帯域幅: 32 × (128/130) / 8 = 3.938 GB/s
    → 約3.9 GB/s（片方向）

  PCIe 5.0 x16 (GPU):
    転送レート: 32 GT/s × 16レーン = 512 GT/s
    エンコーディング: 128b/130b
    実効帯域幅: 512 × (128/130) / 8 = 63.0 GB/s
    → 約63 GB/s（片方向）、双方向で126 GB/s

  PCIe 6.0 x16:
    転送レート: 64 GT/s × 16レーン = 1024 GT/s
    変調方式: PAM4（2ビット/シンボル）
    FECオーバーヘッド: 約3%
    実効帯域幅: 約121 GB/s（片方向）
```

### 2.4 PCIeレーン配分の実例

```
Intel 14th Gen (Raptor Lake) のPCIeレーン配分:

  CPU直結レーン（合計20レーン + 4 DMI）:
  ┌─────────────────────────────────────────┐
  │ CPU                                      │
  │ ├── PCIe 5.0 x16 → GPU                  │
  │ ├── PCIe 4.0 x4  → M.2 SSD (1番目)     │
  │ └── DMI 4.0 x4   → PCH                  │
  └─────────────────────────────────────────┘

  PCH (Z790) レーン（合計28レーン）:
  ┌─────────────────────────────────────────┐
  │ PCH (Z790)                               │
  │ ├── PCIe 4.0 x4 → M.2 SSD (2番目)      │
  │ ├── PCIe 3.0 x4 → M.2 SSD (3番目)      │
  │ ├── PCIe 3.0 x16 → 拡張スロット         │
  │ ├── SATA × 8                             │
  │ ├── USB 3.2 × 5                          │
  │ ├── USB 2.0 × 14                        │
  │ ├── Ethernet                             │
  │ └── Audio, Wi-Fi 等                      │
  └─────────────────────────────────────────┘

  注意: PCHのレーンは共有リソース
  → M.2 SSDを使うとSATAポートが無効化されることがある
  → マザーボードのマニュアルで帯域共有を確認する必要あり

AMD Ryzen 7000 (AM5) のPCIeレーン配分:

  CPU直結レーン（合計28レーン + 4 GMI）:
  ┌─────────────────────────────────────────┐
  │ CPU                                      │
  │ ├── PCIe 5.0 x16 → GPU                  │
  │ ├── PCIe 5.0 x4  → M.2 SSD (1番目)     │
  │ ├── PCIe 4.0 x4  → M.2 SSD (2番目)     │
  │ ├── USB4 × 2                             │
  │ └── GMI → チップセット                   │
  └─────────────────────────────────────────┘
  → AMD はCPU直結レーンが多く、GPU分岐（x8+x8）も可能
```

### 2.5 PCIeの電力供給

```
PCIeスロットの電力供給能力:

  │ スロット │ PCIe 3.0 │ PCIe 4.0 │ PCIe 5.0 │ PCIe 6.0 │
  │──────────│──────────│──────────│──────────│──────────│
  │ x1       │ 10W      │ 10W      │ 10W      │ 10W      │
  │ x4       │ 25W      │ 25W      │ 25W      │ 25W      │
  │ x8       │ 25W      │ 25W      │ 25W      │ 25W      │
  │ x16      │ 75W      │ 75W      │ 75W      │ 75W      │

  GPU の追加電力供給:
  ┌─────────────────────────────────────────────┐
  │ コネクタ          │ 電力    │ 使用例         │
  │───────────────────│─────────│────────────────│
  │ PCIeスロットのみ   │ 75W    │ ローエンドGPU  │
  │ + 6ピン ×1        │ 150W   │ ミドルレンジ    │
  │ + 8ピン ×1        │ 225W   │ ハイエンド      │
  │ + 8ピン ×2        │ 375W   │ RTX 3090等     │
  │ 12VHPWR (600W)    │ 675W   │ RTX 4090       │
  │ 12V-2×6 (600W)    │ 675W   │ RTX 50系列     │
  └─────────────────────────────────────────────┘

  12VHPWR コネクタ（PCIe 5.0電源コネクタ）:
  - 16ピン（12ピン電力 + 4ピンセンス）
  - 最大600W供給可能
  - ケーブル接続不良による溶融問題が報告あり（Gen5時代の課題）
```

---

## 3. USB規格

### 3.1 USB規格比較

| 規格 | 年 | 速度 | 電力供給 | コネクタ |
|------|-----|------|---------|---------|
| USB 1.1 | 1998 | 12 Mbps | 2.5W | Type-A/B |
| USB 2.0 | 2000 | 480 Mbps | 2.5W | Type-A/B |
| USB 3.0 | 2008 | 5 Gbps | 4.5W | Type-A(青)/B |
| USB 3.1 Gen2 | 2013 | 10 Gbps | 100W (PD) | Type-C |
| USB 3.2 Gen2x2 | 2017 | 20 Gbps | 100W (PD) | Type-C |
| USB4 v1 | 2019 | 40 Gbps | 100W (PD) | Type-C |
| USB4 v2 | 2022 | 80 Gbps | 240W (EPR) | Type-C |
| Thunderbolt 5 | 2024 | 120 Gbps | 240W | Type-C |

### 3.2 USB Type-C の統一

```
USB Type-Cコネクタ（24ピン）:

  ┌─────────────────────────────────┐
  │ ● ● ● ● ● ● ● ● ● ● ● ● │ 上段12ピン
  │ ● ● ● ● ● ● ● ● ● ● ● ● │ 下段12ピン
  └─────────────────────────────────┘

  リバーシブル: どちら向きでも挿せる
  統合: USB、Thunderbolt、DisplayPort、電力供給を1本で

  ただし注意: Type-Cコネクタ ≠ USB4
  → 見た目は同じType-Cでも、USB 2.0の速度しか出ないケーブルもある
  → ケーブル/デバイスの仕様確認が重要
```

### 3.3 USB Power Delivery (PD) の詳細

```
USB PD の電圧・電流の組み合わせ:

  USB PD 3.1 (SPR: Standard Power Range):
  │ 電圧    │ 最大電流 │ 最大電力 │ 用途              │
  │─────────│──────────│──────────│───────────────────│
  │ 5V      │ 3A       │ 15W      │ スマホ充電         │
  │ 9V      │ 3A       │ 27W      │ タブレット充電     │
  │ 15V     │ 3A       │ 45W      │ 薄型ノートPC       │
  │ 20V     │ 3A       │ 60W      │ 標準ノートPC       │
  │ 20V     │ 5A       │ 100W     │ 高性能ノートPC     │

  USB PD 3.1 (EPR: Extended Power Range):
  │ 電圧    │ 最大電流 │ 最大電力 │ 用途              │
  │─────────│──────────│──────────│───────────────────│
  │ 28V     │ 5A       │ 140W     │ ゲーミングノート   │
  │ 36V     │ 5A       │ 180W     │ モバイルワークステーション│
  │ 48V     │ 5A       │ 240W     │ 高性能デバイス     │

  PD ネゴシエーション:
  ┌──────────┐                    ┌──────────┐
  │ 充電器   │ ── CC Line ──→   │ デバイス  │
  │ (Source) │ ← USB PD Message │ (Sink)   │
  └──────────┘                    └──────────┘

  1. デバイスが充電器に接続
  2. CCライン（Configuration Channel）で通信開始
  3. 充電器が対応電圧・電流を通知（Source Capabilities）
  4. デバイスが希望する電圧・電流を要求（Request）
  5. 充電器が承認 → 電圧切り替え
  → 全て自動、ユーザー操作不要
```

### 3.4 USB の内部プロトコル

```
USB のデータ転送タイプ:

  ■ コントロール転送
    - デバイス設定、ステータス取得
    - 双方向、小サイズ
    - 全USBデバイスが使用

  ■ バルク転送
    - 大量データ転送（ストレージ、プリンタ）
    - 帯域保証なし、エラー訂正あり
    - 空き帯域を最大限活用

  ■ アイソクロナス転送
    - リアルタイムデータ（音声、動画）
    - 帯域保証あり、エラー訂正なし
    - 遅延より連続性を重視

  ■ インタラプト転送
    - 少量の定期データ（キーボード、マウス）
    - ポーリング間隔保証
    - 低レイテンシ

USB 3.0以降の物理層:
  ┌─────────────────────────────────────────┐
  │ USB 2.0 ペア (D+/D-): 480 Mbps          │ ← 後方互換
  │ USB 3.0 TX ペア:       5 Gbps            │ ← 追加
  │ USB 3.0 RX ペア:       5 Gbps            │ ← 追加
  └─────────────────────────────────────────┘
  → USB 3.0以降はUSB 2.0信号線も同時搭載（互換性維持）
  → Type-Cケーブルでは更にCC、SBU、VBUS等のピンが追加
```

---

## 4. ブートプロセス

### 4.1 電源投入からOS起動まで

```
ブートプロセスの全段階:

  ┌──────────────┐
  │ 1. 電源投入   │ ← 電源ユニットがPower Good信号を送出
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 2. リセット   │ ← CPUのリセットベクタ（0xFFFFFFF0）にジャンプ
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 3. BIOS/UEFI │ ← SPIフラッシュからファームウェアをロード
  │    初期化    │    CPUキャッシュをRAMとして一時使用（CAR）
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 4. POST      │ ← Power-On Self Test
  │    (自己診断)│    メモリ検出、デバイス初期化、エラーチェック
  └──────┬───────┘    ビープ音でエラー通知（メモリなし=連続ビープ等）
         ▼
  ┌──────────────┐
  │ 5. ブート     │ ← ブートデバイスを検索（NVMe→USB→Network）
  │    デバイス   │    UEFI: ESP (EFI System Partition) を探す
  │    選択      │    BIOS: MBRの先頭512バイトを読む
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 6. ブート     │ ← GRUB, systemd-boot, Windows Boot Manager等
  │    ローダー  │    カーネルイメージとinitramfsをメモリにロード
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 7. カーネル   │ ← ハードウェア初期化、ドライバーロード
  │    初期化    │    ルートファイルシステムのマウント
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 8. init/     │ ← systemd (PID 1) がサービスを起動
  │    systemd   │    ネットワーク、ログ、GUI等
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 9. ログイン   │ ← ユーザーの操作可能状態
  └──────────────┘

  所要時間: 数秒（NVMe + UEFI + SSD）〜数分（HDD + BIOS）
```

### 4.2 BIOS vs UEFI

| 項目 | BIOS (Legacy) | UEFI |
|------|-------------|------|
| 策定 | 1975年 (IBM PC) | 2007年 (Intel主導) |
| インターフェース | テキストベース | GUIサポート |
| ブートドライバ | 16ビット | 64ビット |
| パーティション | MBR (最大2TB) | GPT (最大8ZB) |
| セキュリティ | なし | Secure Boot |
| 起動速度 | 遅い | 高速 |
| ネットワーク | なし | PXEブート標準 |

### 4.3 UEFI の詳細

```
UEFI ブートの詳細フロー:

  1. SEC (Security Phase)
     - CPUの初期化（マイクロコード適用）
     - 一時RAM（CAR: Cache As RAM）の設定
     - セキュリティ検証の開始

  2. PEI (Pre-EFI Initialization)
     - メモリコントローラの初期化
     - DRAM トレーニング（タイミング最適化）
     - 実RAMが使用可能に

  3. DXE (Driver Execution Environment)
     - デバイスドライバのロード
     - プロトコルの初期化
     - PCI/PCIeデバイスの列挙

  4. BDS (Boot Device Selection)
     - ブートオプションの列挙
     - ESP（EFI System Partition）の探索
     - ユーザー選択またはデフォルトでブート

  5. TSL (Transient System Load)
     - ブートローダーの実行
     - OS カーネルのロード

  6. RT (Runtime)
     - OS実行中もUEFIランタイムサービスが利用可能
     - 時計、変数ストア、電源管理等

ESP（EFI System Partition）の構造:
  /boot/efi/ (FAT32, 通常 100-500MB)
  ├── EFI/
  │   ├── BOOT/
  │   │   └── BOOTX64.EFI    ← デフォルトブートエントリ
  │   ├── ubuntu/
  │   │   └── grubx64.efi    ← Ubuntu のGRUB
  │   ├── Microsoft/
  │   │   └── Boot/
  │   │       └── bootmgfw.efi ← Windows Boot Manager
  │   └── fedora/
  │       └── shimx64.efi    ← Fedora（Secure Boot対応）
  └── ...

Secure Boot:
  ┌──────────────────────────────────────────────┐
  │ 信頼チェーン:                                  │
  │                                                │
  │ Platform Key (PK)                              │
  │   └── Key Exchange Key (KEK)                   │
  │       └── db (許可された署名のデータベース)     │
  │           └── ブートローダーの署名を検証        │
  │               └── カーネルの署名を検証          │
  │                                                │
  │ → 未署名のコードはブート時に実行不可            │
  │ → マルウェアの早期検出                         │
  └──────────────────────────────────────────────┘
```

```bash
# UEFI関連の確認コマンド（Linux）

# ブートエントリの一覧
efibootmgr -v

# ブート順序の変更
sudo efibootmgr -o 0001,0002,0003

# 新しいブートエントリの追加
sudo efibootmgr -c -d /dev/nvme0n1 -p 1 \
    -l /EFI/ubuntu/grubx64.efi -L "Ubuntu"

# Secure Bootの状態確認
mokutil --sb-state

# UEFI変数の表示
ls /sys/firmware/efi/efivars/

# systemdのブート時間分析
systemd-analyze
systemd-analyze blame | head -20
systemd-analyze plot > boot.svg

# dmesg でブートログ確認
dmesg | head -100
```

---

## 5. チップセットアーキテクチャ

### 5.1 進化の歴史

```
旧式（2000年代）:
  ┌─────┐   FSB    ┌───────────┐   ┌──────┐
  │ CPU │←────────→│ノースブリッジ│──→│ GPU  │
  └─────┘          │(MCH)       │   └──────┘
                   └──────┬────┘
                          │
                   ┌──────┴────┐
                   │サウスブリッジ│──→ USB, SATA, Audio
                   │(ICH)       │
                   └───────────┘

現代（2020年代）:
  ┌──────────────────────┐
  │        CPU            │
  │  ┌──────────────────┐│
  │  │ メモリコントローラ ││──→ DDR5 RAM
  │  │ PCIe コントローラ  ││──→ GPU (PCIe x16)
  │  │                    ││──→ NVMe (PCIe x4)
  │  └──────────────────┘│
  └──────────┬───────────┘
             │ DMI 4.0 (〜8GB/s)
             ▼
  ┌──────────────────────┐
  │   PCH (Platform      │
  │   Controller Hub)    │──→ USB, SATA, Audio
  │                      │──→ Wi-Fi, Ethernet
  │                      │──→ 追加PCIeレーン
  └──────────────────────┘

  進化: ノースブリッジ機能がCPUに統合
  → メモリアクセスの高速化（バスのボトルネック除去）
  → GPU接続の低レイテンシ化
```

### 5.2 Intel vs AMD チップセット比較

```
Intel Z790 vs AMD X670E チップセット比較:

  │ 機能              │ Z790         │ X670E         │
  │───────────────────│──────────────│───────────────│
  │ CPU-PCH接続       │ DMI 4.0 x4   │ GMI（独自）   │
  │ CPU直結PCIe 5.0   │ x16 + x4     │ x16 + x4 + x4│
  │ CPU直結PCIe 4.0   │ x4           │ なし          │
  │ PCH PCIe 4.0      │ x12          │ x12           │
  │ PCH PCIe 3.0      │ x16          │ x8            │
  │ USB 3.2 Gen2x2    │ 5            │ 6             │
  │ USB4               │ なし         │ 2（CPU直結）  │
  │ SATA               │ 8            │ 8             │
  │ DDRサポート        │ DDR4/DDR5    │ DDR5のみ      │
  │ OC対応             │ 対応         │ 対応          │

  AMD X670E はチップセット自体が2チップ構成:
  ┌──────────┐     ┌──────────┐
  │ Promontory│────│ Promontory│
  │ チップ1   │    │ チップ2   │
  └──────────┘     └──────────┘
  → より多くのI/Oを提供するが、消費電力増
```

### 5.3 DMI（Direct Media Interface）のボトルネック

```
DMIボトルネックの理解:

  CPU直結のPCIe: 高帯域・低レイテンシ
  PCH経由のデバイス: DMIが帯域の上限

  DMI 4.0 x4 の帯域幅:
  = PCIe 4.0 x4 = 約 8 GB/s

  PCH配下のデバイスが全て同時にアクセスすると:
  NVMe SSD (PCH経由): 最大3.5 GB/s
  + USB 3.2 Gen2 ×2: 2.5 GB/s
  + 2.5G Ethernet: 0.3 GB/s
  + SATA SSD ×2: 1.0 GB/s
  合計: 7.3 GB/s → DMI帯域に収まるが余裕は少ない

  対策:
  - 高帯域が必要なデバイス（GPU、プライマリNVMe）はCPU直結を選ぶ
  - PCH経由のNVMe SSDは2番目以降のストレージに
  - ネットワークカードをPCIeスロットに増設する場合も注意
```

---

## 6. メモリバスとDDR

### 6.1 DDR世代の比較

```
DDRメモリの世代比較:

  │ 規格     │ 年    │ 転送レート  │ 帯域幅(1ch) │ 電圧  │
  │──────────│───────│─────────────│─────────────│───────│
  │ DDR3     │ 2007  │ 800-2133    │ 17 GB/s     │ 1.5V  │
  │ DDR4     │ 2014  │ 2133-5333   │ 42.7 GB/s   │ 1.2V  │
  │ DDR5     │ 2020  │ 4800-8800   │ 70.4 GB/s   │ 1.1V  │
  │ DDR5 OC  │ 2024  │ 9200+       │ 73.6 GB/s   │ 1.35V │

  DDR5 の主な改良点:
  ┌───────────────────────────────────────────────┐
  │ DDR4:                                          │
  │ ┌──────────────────────────────────────────┐  │
  │ │ 1チャネル × 64ビット幅                    │  │
  │ │ バースト長: 8                              │  │
  │ │ バンクグループ: 4                          │  │
  │ │ PMIC: マザーボード上                       │  │
  │ └──────────────────────────────────────────┘  │
  │                                                │
  │ DDR5:                                          │
  │ ┌──────────────────────────────────────────┐  │
  │ │ 2サブチャネル × 32ビット幅               │  │
  │ │ バースト長: 16                             │  │
  │ │ バンクグループ: 8                          │  │
  │ │ PMIC: DIMM上（電圧調整がモジュール側に）  │  │
  │ │ On-die ECC: 内蔵エラー訂正                │  │
  │ └──────────────────────────────────────────┘  │
  │                                                │
  │ → 2サブチャネルで帯域効率向上                   │
  │ → PMICがモジュール上でクリーンな電力供給         │
  └───────────────────────────────────────────────┘

デュアルチャネル vs シングルチャネル:
  シングルチャネル: 1本のDIMM = 帯域幅 × 1
  デュアルチャネル: 2本のDIMM = 帯域幅 × 2
  クアッドチャネル: 4本のDIMM = 帯域幅 × 4（サーバー/HEDT）
  オクタチャネル:   8本のDIMM = 帯域幅 × 8（サーバー）

  → iGPUを使う場合、デュアルチャネルでフレームレートが2倍近くになる場合も
  → 常にデュアルチャネル構成（2枚または4枚）で組むべき
```

---

## 7. 最新トレンド

### 7.1 CXL（Compute Express Link）

```
CXL の3つのプロトコル:

  CXL.io   — PCIeベースのデバイスI/O（レガシー互換）
  CXL.cache — デバイスがホストメモリをキャッシュ
  CXL.mem   — ホストがデバイスメモリにアクセス

  応用例:
  ┌──────┐        CXL        ┌──────────────────┐
  │ CPU  │←──────────────────→│ CXL メモリ拡張    │
  └──────┘                    │ (DRAM増設、       │
                              │  不揮発メモリ)    │
                              └──────────────────┘
  → RAMをサーバー間で共有（メモリプーリング）
  → 従来不可能だったTB級メモリ空間を実現

CXL のバージョン:
  │ バージョン │ 年   │ 帯域幅     │ 主な機能                │
  │────────────│──────│────────────│─────────────────────────│
  │ CXL 1.1    │ 2020 │ PCIe 5.0   │ メモリ拡張の基本        │
  │ CXL 2.0    │ 2022 │ PCIe 5.0   │ メモリプーリング        │
  │ CXL 3.0    │ 2023 │ PCIe 6.0   │ マルチレベルスイッチング│
  │ CXL 3.1    │ 2024 │ PCIe 6.0   │ TSP（セキュリティ強化） │

CXLのデータセンター活用:
  従来: 各サーバーに固定量のDRAM
  ┌──────┐ ┌──────┐ ┌──────┐
  │128GB │ │256GB │ │64GB  │  ← メモリの無駄が発生
  │(50%  │ │(30%  │ │(90%  │
  │使用) │ │使用) │ │使用) │
  └──────┘ └──────┘ └──────┘

  CXL: メモリプール共有
  ┌──────┐ ┌──────┐ ┌──────┐
  │CPU 1 │ │CPU 2 │ │CPU 3 │
  └──┬───┘ └──┬───┘ └──┬───┘
     │        │        │
  ┌──┴────────┴────────┴──┐
  │   CXL メモリプール     │
  │   合計: 448GB          │
  │   各CPUが必要な分だけ使用│
  └────────────────────────┘
  → メモリ利用率の大幅向上（TCO削減）
```

### 7.2 チップレットアーキテクチャ

| アプローチ | 説明 | 例 |
|-----------|------|-----|
| モノリシック | 1枚の大きなダイ | Intel Core (旧世代) |
| チップレット | 複数の小ダイを接続 | AMD EPYC, Apple M2 Ultra |
| UCIe | チップレット間の標準規格 | Intel, AMD, ARM共同策定 |

> 歩留まり向上、異なるプロセスノードの混在、柔軟なスケーリング。

```
チップレット接続技術:

  ■ AMD Infinity Fabric
    ┌────────┐  IF  ┌────────┐
    │ CCD 0  │────→│ CCD 1  │   CCD = Core Complex Die
    │ 8コア  │←────│ 8コア  │
    └───┬────┘     └───┬────┘
        │              │
    ┌───┴──────────────┴───┐
    │      IOD (I/O Die)    │
    │  メモリコントローラ    │
    │  PCIeコントローラ      │
    └───────────────────────┘
    → CCD: 5nm (最先端)、IOD: 6nm (安価) で混在可能

  ■ Apple UltraFusion
    ┌────────────────┐  UF  ┌────────────────┐
    │ M2 Max Die 1   │────→│ M2 Max Die 2   │
    │                │←────│                │
    │ GPU 38コア     │      │ GPU 38コア     │
    │ CPU 12コア     │      │ CPU 12コア     │
    └────────────────┘      └────────────────┘
    帯域幅: 2.5 TB/s（シリコンインターポーザ）
    → 2つのM2 Maxを1つのM2 Ultraとして使用

  ■ UCIe (Universal Chiplet Interconnect Express)
    - 業界標準のチップレット接続規格
    - 異なるベンダーのチップレットを組み合わせ可能
    - 帯域幅: 最大256 GB/s/mm
    - Intel, AMD, ARM, Samsung, TSMC等が参画
```

---

## 8. サーバーアーキテクチャ

### 8.1 サーバーとデスクトップの違い

```
サーバーマザーボードの特徴:

  │ 機能              │ デスクトップ    │ サーバー            │
  │───────────────────│─────────────────│─────────────────────│
  │ CPUソケット       │ 1              │ 1-2（デュアルソケット）│
  │ メモリスロット    │ 2-4            │ 8-24                │
  │ メモリ種類        │ UDIMM          │ RDIMM/LRDIMM (ECC) │
  │ 最大メモリ        │ 128GB          │ 4-6TB               │
  │ PCIeスロット      │ 1-3            │ 6-10                │
  │ ネットワーク      │ 1GbE-2.5GbE   │ 10/25/100GbE        │
  │ ストレージ        │ M.2 × 2-3     │ U.2/E1.S × 24+     │
  │ リモート管理      │ なし           │ IPMI/BMC/iLO/iDRAC │
  │ 冗長電源          │ なし           │ あり（ホットスワップ）│
  │ ECC メモリ        │ 基本なし       │ 必須                │

  IPMI/BMC（Baseboard Management Controller）:
  ┌────────────────────────────────────────┐
  │ BMC チップ                              │
  │ - 独立したCPU（ARM Cortex等）          │
  │ - 独立したネットワーク接続             │
  │ - OS停止中もサーバー管理可能           │
  │ - 電源ON/OFF                           │
  │ - KVM（キーボード・ビデオ・マウス）    │
  │ - ファームウェアアップデート           │
  │ - ハードウェア監視（温度、電圧、ファン）│
  │ - シリアルコンソール                   │
  └────────────────────────────────────────┘
  → データセンターでの物理アクセスなしでサーバー管理
```

### 8.2 NUMAアーキテクチャ

```
NUMA（Non-Uniform Memory Access）:

  デュアルソケットサーバーの場合:
  ┌─────────────────────────────────────────────┐
  │ NUMA Node 0               NUMA Node 1       │
  │ ┌─────────┐               ┌─────────┐     │
  │ │ CPU 0   │               │ CPU 1   │     │
  │ │ 32コア  │──── QPI/UPI ──│ 32コア  │     │
  │ └────┬────┘               └────┬────┘     │
  │      │                         │           │
  │ ┌────┴────┐               ┌────┴────┐     │
  │ │ DDR5    │               │ DDR5    │     │
  │ │ 256GB   │               │ 256GB   │     │
  │ │ ローカル │               │ ローカル │     │
  │ └─────────┘               └─────────┘     │
  └─────────────────────────────────────────────┘

  メモリアクセスレイテンシ:
  - ローカルメモリ: 80ns
  - リモートメモリ: 130ns（約1.6倍遅い）

  → OSとアプリケーションはNUMAを意識したメモリ配置が重要
  → numactl コマンドでNUMA制御可能
```

```bash
# NUMA情報の確認
numactl --hardware

# NUMA Node 0 でプロセスを実行
numactl --cpunodebind=0 --membind=0 ./my_application

# NUMAバランスの確認
cat /proc/sys/kernel/numa_balancing

# メモリのNUMA配置状況
numastat -m
```

---

## 9. 実践演習

### 演習1: スペックの読み方（基礎）

自分のPC/Macのスペックを調べ、以下を特定せよ:
1. CPUソケット/チップの種類
2. メモリの規格（DDR4/DDR5）、チャネル数、帯域幅
3. ストレージの接続方式（NVMe/SATA）
4. USBポートの種類と速度

### 演習2: ボトルネック分析（応用）

以下のシステムで、ボトルネックになりうる箇所を特定せよ:
- CPU: AMD Ryzen 9 7950X (16コア)
- RAM: DDR5-5200 64GB（デュアルチャネル）
- GPU: NVIDIA RTX 4090 (PCIe 4.0 x16)
- Storage: Samsung 990 Pro 2TB (PCIe 4.0 x4)
- ワークロード: 4K動画編集 + AI学習

### 演習3: ブートプロセスの観察（発展）

LinuxマシンでUEFIブートプロセスを観察せよ:
```bash
# ブートログの確認
journalctl -b | head -100

# UEFIブート変数の確認
efibootmgr -v

# PCIeデバイスの列挙
lspci -vvv | head -50

# USBデバイスの列挙
lsusb -t
```

### 演習4: PCIeレーン配分の設計（応用）

以下の要件を満たすシステムのPCIeレーン配分を設計せよ:
- GPU: RTX 4090（x16必須）
- NVMe SSD: 2TB × 2台（各x4）
- 10GbE NIC: x4
- Thunderbolt 4 カード: x4
- プラットフォーム: Intel Z790

レーンの不足をどう解決するか、トレードオフを論じよ。

### 演習5: サーバー構成の設計（発展）

以下の要件のサーバーを設計せよ:
- 用途: PostgreSQL データベースサーバー
- CPU: デュアルソケット希望
- メモリ: 512GB以上（ECC必須）
- ストレージ: NVMe SSD RAID 10
- ネットワーク: 25GbE × 2（冗長）
- 予算: 300万円以内

設計項目:
1. CPU/プラットフォームの選定
2. メモリ構成（DIMM数、チャネル配分）
3. ストレージ構成（台数、RAID）
4. ネットワーク構成
5. 冗長性の確保方法


---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## 実務での適用シナリオ

### シナリオ1: スタートアップでのMVP開発

**状況:** 限られたリソースで素早くプロダクトをリリースする必要がある

**アプローチ:**
- シンプルなアーキテクチャを選択
- 必要最小限の機能に集中
- 自動テストはクリティカルパスのみ
- モニタリングは早期から導入

**学んだ教訓:**
- 完璧を求めすぎない（YAGNI原則）
- ユーザーフィードバックを早期に取得
- 技術的負債は意識的に管理する

### シナリオ2: レガシーシステムのモダナイゼーション

**状況:** 10年以上運用されているシステムを段階的に刷新する

**アプローチ:**
- Strangler Fig パターンで段階的に移行
- 既存のテストがない場合はCharacterization Testを先に作成
- APIゲートウェイで新旧システムを共存
- データ移行は段階的に実施

| フェーズ | 作業内容 | 期間目安 | リスク |
|---------|---------|---------|--------|
| 1. 調査 | 現状分析、依存関係の把握 | 2-4週間 | 低 |
| 2. 基盤 | CI/CD構築、テスト環境 | 4-6週間 | 低 |
| 3. 移行開始 | 周辺機能から順次移行 | 3-6ヶ月 | 中 |
| 4. コア移行 | 中核機能の移行 | 6-12ヶ月 | 高 |
| 5. 完了 | 旧システム廃止 | 2-4週間 | 中 |

### シナリオ3: 大規模チームでの開発

**状況:** 50人以上のエンジニアが同一プロダクトを開発する

**アプローチ:**
- ドメイン駆動設計で境界を明確化
- チームごとにオーナーシップを設定
- 共通ライブラリはInner Source方式で管理
- APIファーストで設計し、チーム間の依存を最小化

```python
# チーム間のAPI契約定義
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
    """チーム間のAPI契約"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # レスポンスタイムSLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """SLA準拠の確認"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """OpenAPI形式で出力"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# 使用例
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

### シナリオ4: パフォーマンスクリティカルなシステム

**状況:** ミリ秒単位のレスポンスが求められるシステム

**最適化ポイント:**
1. キャッシュ戦略（L1: インメモリ、L2: Redis、L3: CDN）
2. 非同期処理の活用
3. コネクションプーリング
4. クエリ最適化とインデックス設計

| 最適化手法 | 効果 | 実装コスト | 適用場面 |
|-----------|------|-----------|---------|
| インメモリキャッシュ | 高 | 低 | 頻繁にアクセスされるデータ |
| CDN | 高 | 低 | 静的コンテンツ |
| 非同期処理 | 中 | 中 | I/O待ちが多い処理 |
| DB最適化 | 高 | 高 | クエリが遅い場合 |
| コード最適化 | 低-中 | 高 | CPU律速の場合 |

---

## チーム開発での活用

### コードレビューのチェックリスト

このトピックに関連するコードレビューで確認すべきポイント:

- [ ] 命名規則が一貫しているか
- [ ] エラーハンドリングが適切か
- [ ] テストカバレッジは十分か
- [ ] パフォーマンスへの影響はないか
- [ ] セキュリティ上の問題はないか
- [ ] ドキュメントは更新されているか

### ナレッジ共有のベストプラクティス

| 方法 | 頻度 | 対象 | 効果 |
|------|------|------|------|
| ペアプログラミング | 随時 | 複雑なタスク | 即時のフィードバック |
| テックトーク | 週1回 | チーム全体 | 知識の水平展開 |
| ADR (設計記録) | 都度 | 将来のメンバー | 意思決定の透明性 |
| 振り返り | 2週間ごと | チーム全体 | 継続的改善 |
| モブプログラミング | 月1回 | 重要な設計 | 合意形成 |

### 技術的負債の管理

```
優先度マトリクス:

        影響度 高
          │
    ┌─────┼─────┐
    │ 計画 │ 即座 │
    │ 的に │ に   │
    │ 対応 │ 対応 │
    ├─────┼─────┤
    │ 記録 │ 次の │
    │ のみ │ Sprint│
    │     │ で   │
    └─────┼─────┘
          │
        影響度 低
    発生頻度 低  発生頻度 高
```

---

## セキュリティの考慮事項

### 一般的な脆弱性と対策

| 脆弱性 | リスクレベル | 対策 | 検出方法 |
|--------|------------|------|---------|
| インジェクション攻撃 | 高 | 入力値のバリデーション・パラメータ化クエリ | SAST/DAST |
| 認証の不備 | 高 | 多要素認証・セッション管理の強化 | ペネトレーションテスト |
| 機密データの露出 | 高 | 暗号化・アクセス制御 | セキュリティ監査 |
| 設定の不備 | 中 | セキュリティヘッダー・最小権限の原則 | 構成スキャン |
| ログの不足 | 中 | 構造化ログ・監査証跡 | ログ分析 |

### セキュアコーディングのベストプラクティス

```python
# セキュアコーディング例
import hashlib
import secrets
import hmac
from typing import Optional

class SecurityUtils:
    """セキュリティユーティリティ"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """暗号学的に安全なトークン生成"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """パスワードのハッシュ化"""
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
        """パスワードの検証"""
        new_hash, _ = SecurityUtils.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)

    @staticmethod
    def sanitize_input(value: str) -> str:
        """入力値のサニタイズ"""
        dangerous_chars = ['<', '>', '"', "'", '&', '\\']
        result = value
        for char in dangerous_chars:
            result = result.replace(char, '')
        return result.strip()

# 使用例
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
```

### セキュリティチェックリスト

- [ ] 全ての入力値がバリデーションされている
- [ ] 機密情報がログに出力されていない
- [ ] HTTPS が強制されている
- [ ] CORS ポリシーが適切に設定されている
- [ ] 依存パッケージの脆弱性スキャンが実施されている
- [ ] エラーメッセージに内部情報が含まれていない

---

## マイグレーションガイド

### バージョンアップ時の注意点

| バージョン | 主な変更点 | 移行作業 | 影響範囲 |
|-----------|-----------|---------|---------|
| v1.x → v2.x | API設計の刷新 | エンドポイント変更 | 全クライアント |
| v2.x → v3.x | 認証方式の変更 | トークン形式更新 | 認証関連 |
| v3.x → v4.x | データモデル変更 | マイグレーションスクリプト実行 | DB関連 |

### 段階的移行の手順

```python
# マイグレーションスクリプトのテンプレート
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)

class MigrationRunner:
    """段階的マイグレーション実行エンジン"""

    def __init__(self, migration_dir: str):
        self.migration_dir = Path(migration_dir)
        self.migrations: List[Dict] = []
        self.completed: List[str] = []

    def register(self, version: str, description: str,
                 up: Callable, down: Callable):
        """マイグレーションの登録"""
        self.migrations.append({
            'version': version,
            'description': description,
            'up': up,
            'down': down,
            'registered_at': datetime.now().isoformat()
        })

    def run_up(self, target_version: str = None):
        """マイグレーションの実行（アップグレード）"""
        for migration in self.migrations:
            if migration['version'] in self.completed:
                continue
            logger.info(f"実行中: {migration['version']} - "
                       f"{migration['description']}")
            try:
                migration['up']()
                self.completed.append(migration['version'])
                logger.info(f"完了: {migration['version']}")
            except Exception as e:
                logger.error(f"失敗: {migration['version']}: {e}")
                raise
            if target_version and migration['version'] == target_version:
                break

    def run_down(self, target_version: str):
        """マイグレーションのロールバック"""
        for migration in reversed(self.migrations):
            if migration['version'] not in self.completed:
                continue
            if migration['version'] == target_version:
                break
            logger.info(f"ロールバック: {migration['version']}")
            migration['down']()
            self.completed.remove(migration['version'])

    def status(self) -> Dict:
        """マイグレーション状態の確認"""
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

### ロールバック計画

移行作業には必ずロールバック計画を準備してください:

1. **データのバックアップ**: 移行前に完全バックアップを取得
2. **テスト環境での検証**: 本番と同等の環境で事前検証
3. **段階的なロールアウト**: カナリアリリースで段階的に展開
4. **監視の強化**: 移行中はメトリクスの監視間隔を短縮
5. **判断基準の明確化**: ロールバックを判断する基準を事前に定義
---

## FAQ

### Q1: マザーボードの選び方は？

**A**: 以下の順で決める:
1. CPUソケット（Intel LGA1700、AMD AM5等）
2. チップセット（機能差: PCIeレーン数、USB数、オーバークロック対応）
3. フォームファクタ（ATX/mATX/Mini-ITX）
4. メモリスロット数とDDR世代
5. M.2/NVMeスロット数
6. 拡張性（PCIeスロット、USB、ネットワーク）

### Q2: Thunderbolt と USB4 の違いは？

**A**: Thunderbolt 4/5 は USB4 のスーパーセット:
- USB4: 最低20Gbps保証
- Thunderbolt 4: 40Gbps保証 + DP 2.0 + PCIeトンネリング
- Thunderbolt 5: 80-120Gbps + 240W給電

全てType-Cコネクタを使用するが、性能はケーブルとデバイスの対応次第。

### Q3: なぜPCIeは「レーン」単位なのですか？

**A**: 柔軟なスケーリングのため。デバイスの帯域要求に応じてレーン数を変えられる:
- NVMe SSD: x4で十分（〜8GB/s）
- GPU: x16で最大帯域（〜32GB/s）
- Wi-Fiカード: x1で十分（〜1GB/s）
マザーボード上のPCIeレーン数はCPU+チップセットで決まり、配分はBIOSで設定可能。

### Q4: DDR5はDDR4よりどのくらい速いですか？

**A**: 帯域幅は約1.5-2倍だが、レイテンシは同等かやや悪い:
- DDR4-3200: 帯域25.6GB/s、CL16 = 10ns
- DDR5-5600: 帯域44.8GB/s、CL36 = 12.86ns
- 帯域重視のワークロード（動画編集、AI）はDDR5が有利
- レイテンシ重視のワークロード（ゲーム）はDDR4との差が小さい

### Q5: VRMの品質はどう見分けますか？

**A**: 以下のポイントを確認:
- フェーズ数: 12フェーズ以上が望ましい（OC用途なら16+）
- MOSFET: DrMOS（高効率統合型）が理想
- ヒートシンク: VRM上に大型ヒートシンクがあるか
- PWMコントローラ: Renesas/Infineonの高品質チップ
- レビューサイトのサーモグラフィーテスト結果

### Q6: Secure Bootを無効にしても大丈夫ですか？

**A**: 一般的なLinux使用なら多くのディストリビューションがSecure Boot対応済み。無効化が必要なケース:
- カスタムカーネルの使用
- 署名されていないドライバ（一部のNVIDIAドライバ等）
- デュアルブートの特殊構成
セキュリティの観点からは有効のままが推奨。企業環境では必須であることが多い。

---

## まとめ

| 概念 | ポイント |
|------|---------|
| マザーボード | CPU、メモリ、PCH（チップセット）を接続する基盤 |
| PCIe | ポイントtoポイント、レーン制、世代ごとに帯域2倍 |
| USB | Type-Cに統一傾向、USB4/Thunderboltで高速化 |
| ブート | 電源→UEFI→POST→ブートローダー→カーネル→init |
| 進化 | ノースブリッジ→CPU統合、CXL、チップレット |
| DDR5 | 2サブチャネル、帯域向上、On-die ECC |
| NUMA | デュアルソケットではメモリ配置が性能に直結 |

---

## 次に読むべきガイド


---

## 参考文献

1. PCI-SIG. "PCI Express Base Specification." Various Revisions.
2. USB Implementers Forum. "Universal Serial Bus Specification." Various Revisions.
3. UEFI Forum. "Unified Extensible Firmware Interface Specification."
4. Intel. "Platform Controller Hub (PCH) Datasheets."
5. CXL Consortium. "Compute Express Link Specification." https://www.computeexpresslink.org/
6. JEDEC. "DDR5 SDRAM Standard (JESD79-5)." 2020.
7. AMD. "AMD EPYC 9004 Series Architecture." Whitepaper.
8. UCIe Consortium. "Universal Chiplet Interconnect Express Specification." 2022.
9. Intel. "12th Gen Intel Core Processor Datasheet." Volume 1.
10. AMD. "AMD Ryzen 7000 Series Platform Technology." 2022.

# クラウドとリアルタイムOS

> クラウドOSは大規模な分散リソースを抽象化し、リアルタイムOSは時間制約のある制御を保証する — 対極にある2つのOS領域。

## この章で学ぶこと

- [ ] クラウドにおけるOSの役割を理解する
- [ ] リアルタイムOSの概念を知る
- [ ] 将来のOS技術を把握する

---

## 1. クラウドとOS

```
クラウドでのOS階層:

  ユーザーのアプリ
     ↓
  コンテナ / FaaS ← ユーザーが意識するのはここ
     ↓
  コンテナランタイム (containerd)
     ↓
  ホストOS (Amazon Linux, Ubuntu)
     ↓
  ハイパーバイザ (KVM, Nitro)
     ↓
  物理ハードウェア

AWS Nitro System:
  専用ハードウェアでネットワーク/ストレージ/セキュリティを
  メインCPUからオフロード
  → CPU のほぼ100%をゲストVMに提供
  → Nitro Enclaves: 高度に隔離された計算環境

サーバーレスのOS:
  AWS Lambda:
  → Firecracker マイクロVM（起動125ms以下）
  → 関数単位で実行、OS管理不要
  → ユーザーはコードのみに集中

  将来: OSが「見えない」時代へ
  → 開発者がOSを意識するのは異常時のみ
  → しかし、OSの知識はトラブルシューティングに不可欠
```

---

## 2. リアルタイムOS（RTOS）

```
リアルタイムOS:
  処理をデッドラインまでに「必ず」完了する保証

  ハードリアルタイム:
  → デッドライン違反 = 致命的エラー
  → 航空機制御、自動車ABS、医療機器、原子力
  → 例: VxWorks, QNX, INTEGRITY, SafeRTOS

  ソフトリアルタイム:
  → デッドライン違反 = 品質低下
  → 動画再生、音声通話、ゲーム
  → 例: Linux + PREEMPT_RT

  RTOSの特徴:
  ┌──────────────────────────────────────┐
  │ 決定論的スケジューリング:            │
  │ → 最悪実行時間（WCET）が保証される   │
  │ → 優先度ベース + 優先度逆転の防止    │
  │                                      │
  │ 低レイテンシ:                        │
  │ → 割り込みレイテンシ: マイクロ秒以下 │
  │ → コンテキストスイッチ: マイクロ秒以下│
  │                                      │
  │ 小さなフットプリント:                │
  │ → カーネルサイズ: 数KB〜数百KB       │
  │ → RAM: 数KB〜数MB で動作             │
  └──────────────────────────────────────┘

  FreeRTOS（Amazon所有）:
  → マイクロコントローラ向け
  → ESP32, STM32 等で使用
  → IoTデバイスのデファクト
  → AWS IoT Core と統合
```

---

## 3. 将来のOS技術

```
1. Unikernel:
   アプリ+OS機能を1バイナリに
   → 超高速起動、最小攻撃面
   → FaaS/エッジで有望

2. Library OS:
   OSサービスをアプリにリンク
   → Demikernel: ネットワークスタックをユーザー空間に

3. Fuchsia (Google):
   Zircon マイクロカーネル
   → Linux依存からの脱却
   → Nest Hub で採用開始

4. Rust in the Kernel:
   Linux 6.1 から Rust がカーネルに参入
   → メモリ安全なドライバ開発
   → Android, Linux ドライバの一部で採用開始

5. CXL (Compute Express Link):
   メモリの共有・プーリング
   → OSのメモリ管理が根本的に変化する可能性
   → 遠いメモリ（Far Memory）の概念

6. Confidential Computing:
   実行中のデータも暗号化（TEE）
   → AMD SEV, Intel TDX, ARM CCA
   → クラウドでもデータが保護される
```

---

## まとめ

| 分野 | 特徴 | 例 |
|------|------|-----|
| クラウドOS | 仮想化+コンテナ、管理不要化 | Amazon Linux, Firecracker |
| RTOS | 時間保証、小フットプリント | VxWorks, FreeRTOS, QNX |
| 将来 | Unikernel, Rust, CXL, TEE | Fuchsia, MirageOS |

---

## シリーズ完結

このガイドで **オペレーティングシステムガイド** シリーズは完結です。

```
学習パスの振り返り:
  00 OS入門 → 01 プロセス管理 → 02 メモリ管理
  → 03 ファイルシステム → 04 I/O → 05 セキュリティ
  → 06 仮想化 → 07 現代のOS（本章）
```

→ 次のステップ: [[../../linux-cli-mastery/SKILL.md]] — Linux CLI実践

---

## 参考文献
1. Tanenbaum, A. "Modern Operating Systems." 4th Ed, Pearson, 2014.
2. Barry, R. "Mastering the FreeRTOS Real Time Kernel." 2016.

# モバイルOS

> モバイルOSはバッテリー、タッチ操作、センサーという制約の中で最高のUXを実現するために進化してきた。

## この章で学ぶこと

- [ ] iOS と Android のアーキテクチャの違いを説明できる
- [ ] モバイルOS特有のセキュリティモデルを理解する

---

## 1. iOS vs Android アーキテクチャ

```
iOS:
  ┌──────────────────────┐
  │ アプリケーション      │  UIKit / SwiftUI
  ├──────────────────────┤
  │ フレームワーク        │  Cocoa Touch
  ├──────────────────────┤
  │ Core Services        │  Foundation, CoreData
  ├──────────────────────┤
  │ Core OS              │  Darwin (XNU カーネル)
  └──────────────────────┘
  → クローズドソース（Darwin除く）
  → ハードウェアとの垂直統合
  → App Store のみ（サイドローディング制限）

Android:
  ┌──────────────────────┐
  │ アプリケーション      │  Kotlin / Java
  ├──────────────────────┤
  │ Android Framework    │  Activity, Service
  ├──────────────────────┤
  │ Android Runtime (ART)│  AOTコンパイル
  ├──────────────────────┤
  │ HAL                  │  ハードウェア抽象層
  ├──────────────────────┤
  │ Linux Kernel         │  修正版
  └──────────────────────┘
  → オープンソース（AOSP）
  → 多様なハードウェアメーカー
  → 複数のアプリストア+サイドローディング

比較:
┌──────────────┬──────────────┬──────────────┐
│ 項目         │ iOS          │ Android      │
├──────────────┼──────────────┼──────────────┤
│ カーネル     │ XNU (Mach+BSD)│ Linux        │
│ 言語         │ Swift/ObjC   │ Kotlin/Java  │
│ アプリ隔離   │ サンドボックス│ SELinux+UID  │
│ 更新         │ 全端末一斉   │ メーカー依存 │
│ カスタム     │ 制限的       │ 自由         │
│ セキュリティ │ ◎           │ ○           │
└──────────────┴──────────────┴──────────────┘
```

---

## 2. モバイルOS特有の技術

```
電力管理:
  モバイル最大の制約 = バッテリー
  → CPUのDVFS（動的電圧・周波数スケーリング）
  → big.LITTLE / 高効率コア+高性能コア
  → バックグラウンド実行の制限
  → iOS: Background App Refresh
  → Android: Doze モード、App Standby

プッシュ通知:
  iOS: APNs (Apple Push Notification service)
  Android: FCM (Firebase Cloud Messaging)
  → 常時接続を避けてバッテリー節約

センサー統合:
  → 加速度、ジャイロ、GPS、気圧、近接、光
  → CoreMotion (iOS), SensorManager (Android)
  → 低消費電力コプロセッサ（M1 Motion, Sensor Hub）

セキュリティ:
  → Secure Enclave (iOS) / Titan M (Pixel)
  → 生体認証（Face ID, Touch ID / 指紋, 顔）
  → デバイス暗号化（FBE: File-Based Encryption）
  → リモートワイプ（紛失時）
```

---

## まとめ

| 項目 | iOS | Android |
|------|-----|---------|
| 基盤 | XNU (Unix) | Linux |
| 強み | セキュリティ、統合性 | 自由度、多様性 |
| 開発 | Swift + Xcode | Kotlin + Android Studio |
| 配布 | App Store | Play Store + サイドロード |

---

## 次に読むべきガイド
→ [[01-cloud-os.md]] — クラウドとリアルタイムOS

---

## 参考文献
1. Levin, J. "Mac OS X and iOS Internals." Wiley, 2012.
2. Yaghmour, K. "Embedded Android." O'Reilly, 2013.

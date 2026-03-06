# モバイルOS ── iOS・Android アーキテクチャと設計原理の徹底解説

> **前提知識**: プロセス管理、メモリ管理、ファイルシステムの基礎知識
> **学習時間目安**: 約 6 時間
> **難易度**: ★★★☆☆（中級）

モバイルOSは、バッテリー駆動・タッチ操作・多種多様なセンサーという厳しい制約の中で、デスクトップOSに匹敵する処理能力と最高のユーザ体験を両立させるために独自の進化を遂げてきた。本ガイドでは iOS と Android を中心に、モバイルOS のカーネル構造・プロセスモデル・メモリ管理・電力制御・セキュリティモデル・アプリケーションライフサイクルを体系的に解説する。

---

## この章で学ぶこと

- [ ] iOS (XNU) と Android (Linux) のカーネルアーキテクチャの違いを図解付きで説明できる
- [ ] モバイルOS特有のプロセスライフサイクルと優先度モデルを理解する
- [ ] サンドボックスと権限モデルによるセキュリティ設計を説明できる
- [ ] 電力管理 (DVFS, Doze, Background App Refresh) の仕組みを理解する
- [ ] プッシュ通知・センサー統合・IPC の設計原理を把握する
- [ ] Android の HAL / HIDL と iOS の IOKit ドライバモデルを比較できる
- [ ] モバイルアプリのビルド・テスト・配布パイプラインを構築できる

---

## 目次

1. [モバイルOSの歴史と進化](#1-モバイルosの歴史と進化)
2. [カーネルアーキテクチャの比較](#2-カーネルアーキテクチャの比較)
3. [プロセスモデルとアプリケーションライフサイクル](#3-プロセスモデルとアプリケーションライフサイクル)
4. [メモリ管理と仮想メモリ](#4-メモリ管理と仮想メモリ)
5. [電力管理とスケジューリング](#5-電力管理とスケジューリング)
6. [セキュリティモデルとサンドボックス](#6-セキュリティモデルとサンドボックス)
7. [プロセス間通信とプッシュ通知](#7-プロセス間通信とプッシュ通知)
8. [センサー・ハードウェア抽象化レイヤ](#8-センサーハードウェア抽象化レイヤ)
9. [アプリ開発とビルドパイプライン](#9-アプリ開発とビルドパイプライン)
10. [アンチパターンと設計上の落とし穴](#10-アンチパターンと設計上の落とし穴)
11. [段階別演習](#11-段階別演習)
12. [FAQ ── よくある質問](#12-faq--よくある質問)
13. [まとめと次のステップ](#13-まとめと次のステップ)
14. [参考文献](#14-参考文献)

---

## 1. モバイルOSの歴史と進化

### 1.1 モバイルOSの系譜

モバイルOSの歴史は、1990年代のPDA用OSにまで遡る。Palm OS や Windows CE がその先駆けであり、制約の厳しい組込み環境でGUIを提供するという挑戦であった。

```
モバイルOS の進化タイムライン
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1996 ──── Palm OS 1.0
           │  タッチペン操作、シングルタスク
           │  16MHz Motorola 68000、128KB RAM
           ▼
2000 ──── Symbian OS (Nokia)
           │  マルチタスク、C++ ベース
           │  携帯電話向け初の本格OS
           ▼
2002 ──── BlackBerry OS
           │  プッシュメール、企業向けセキュリティ
           │  QWERTYキーボード特化
           ▼
2005 ──── Windows Mobile 5.0
           │  .NET Compact Framework
           │  PocketPC と Smartphone の統合
           ▼
2007 ──── iPhone OS 1.0 (後の iOS)  ← 革命的転換点
           │  マルチタッチ、Safari、Visual Voicemail
           │  サードパーティアプリなし（Web Apps のみ）
           ▼
2008 ──── Android 1.0 (HTC Dream)
           │  オープンソース (AOSP)
           │  Google サービス統合、Android Market
           ▼
2008 ──── iPhone OS 2.0 + App Store
           │  ネイティブ SDK 公開
           │  サードパーティアプリのエコシステム開始
           ▼
2010 ──── Android 2.2 (Froyo)   / iOS 4
           │  JIT コンパイル導入      マルチタスク対応
           ▼
2014 ──── Android 5.0 (Lollipop) / iOS 8
           │  ART (AOT コンパイル)    Extensions, Metal API
           │  Material Design          HealthKit, HomeKit
           ▼
2017 ──── Android 8.0 (Oreo)     / iOS 11
           │  Project Treble (HAL分離) ARKit, Core ML
           │  バックグラウンド制限強化  HEIF/HEVC 標準化
           ▼
2020 ──── Android 11              / iOS 14
           │  スコープドストレージ強制  App Clips, ウィジェット
           │  ワンタイム権限            App Library
           ▼
2023 ──── Android 14              / iOS 17
           │  大画面対応強化           Interactive Widgets
           │  Privacy Sandbox          StandBy モード
           ▼
2025 ──── Android 16              / iOS 19
           オンデバイスAI統合、プライバシー強化の最前線
```

### 1.2 なぜモバイルOSは特別なのか

デスクトップOSとモバイルOSの根本的な違いは「制約の厳しさ」にある。

```
デスクトップ OS vs モバイル OS ── 設計制約の比較
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                Desktop OS              Mobile OS
              ┌──────────────┐       ┌──────────────┐
  電源        │ AC電源(無限) │       │ バッテリー   │
              │              │       │ (3000-5000mAh)│
              ├──────────────┤       ├──────────────┤
  冷却        │ ファン/水冷  │       │ パッシブ冷却 │
              │ 数百W放熱可  │       │ 数W上限      │
              ├──────────────┤       ├──────────────┤
  メモリ      │ 16-128 GB   │       │ 4-16 GB      │
              │ スワップ自由 │       │ スワップ制限 │
              ├──────────────┤       ├──────────────┤
  入力        │ キーボード   │       │ タッチ       │
              │ マウス       │       │ ジェスチャー │
              ├──────────────┤       ├──────────────┤
  通信        │ 有線LAN/WiFi │       │ セルラー/WiFi│
              │ 常時接続     │       │ 断続的接続   │
              ├──────────────┤       ├──────────────┤
  セキュリティ│ ユーザ信頼型 │       │ ゼロトラスト │
              │              │       │ アプリ隔離   │
              └──────────────┘       └──────────────┘

  モバイルOSの設計原則:
  1. 電力効率を最優先 ─ 全ての機能は消費電力の観点で評価される
  2. 応答性の保証 ─ UIスレッドのブロックは絶対に許容しない
  3. プライバシー・バイ・デザイン ─ アプリは最小権限で動作する
  4. 断続的接続の前提 ─ オフラインでも基本機能を維持する
  5. セキュリティの多層防御 ─ ハードウェアからアプリまで一貫した保護
```

### 1.3 市場シェアと影響力

2025年時点でモバイルOSの市場は実質的に iOS と Android の二極体制である。

| 指標 | iOS | Android | その他 |
|------|-----|---------|--------|
| 世界シェア | 約 27% | 約 72% | 約 1% |
| 北米シェア | 約 55% | 約 44% | 約 1% |
| 日本シェア | 約 65% | 約 34% | 約 1% |
| アクティブ端末数 | 約 20 億台 | 約 35 億台 | - |
| 年間アプリ売上 | 約 850 億ドル | 約 480 億ドル | - |
| 開発者数 | 約 3,400 万人 | 約 4,000 万人 | - |

> **注目点**: Android は端末数で圧倒するが、アプリ売上では iOS が上回る。これはユーザ層の購買力の違いと、iOS の統一されたエコシステムによるものである。

---

## 2. カーネルアーキテクチャの比較

### 2.1 iOS: XNU カーネル

iOS は Apple の Darwin OS をベースとし、その中核となる XNU (X is Not Unix) カーネルは、Mach マイクロカーネルと FreeBSD のモノリシックカーネルを融合したハイブリッド設計である。

```
XNU カーネル アーキテクチャ詳細図
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────────────────────────────────┐
  │              ユーザ空間 (User Space)             │
  │                                                 │
  │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐│
  │  │ UIKit /  │ │ Cocoa    │ │ System Daemons   ││
  │  │ SwiftUI  │ │ Touch    │ │ (launchd,        ││
  │  │ アプリ   │ │Framework │ │  configd, etc.)  ││
  │  └────┬─────┘ └────┬─────┘ └────────┬─────────┘│
  │       │            │                │           │
  │  ─────┴────────────┴────────────────┴───────────│
  │       │  libSystem (libc, libdispatch, etc.)    │
  ├───────┼─────────────────────────────────────────┤
  │       │        カーネル空間 (Kernel Space)       │
  │       ▼                                         │
  │  ┌──────────────────────────────────────────┐   │
  │  │            BSD Layer                      │   │
  │  │  ┌────────────┬──────────┬─────────────┐ │   │
  │  │  │ POSIX API  │ VFS     │ Networking  │ │   │
  │  │  │ (syscall)  │(仮想FS) │ (TCP/IP)    │ │   │
  │  │  │ プロセス   │ HFS+    │ BSD Socket  │ │   │
  │  │  │ シグナル   │ APFS    │             │ │   │
  │  │  └────────────┴──────────┴─────────────┘ │   │
  │  ├──────────────────────────────────────────┤   │
  │  │            Mach Layer                     │   │
  │  │  ┌────────────┬──────────┬─────────────┐ │   │
  │  │  │ タスク     │ IPC     │ スケジューラ│ │   │
  │  │  │ (プロセス) │ (Mach   │ (優先度     │ │   │
  │  │  │ スレッド   │  Port)  │  ベース)    │ │   │
  │  │  │ 仮想メモリ │ MIG     │ リアルタイム│ │   │
  │  │  └────────────┴──────────┴─────────────┘ │   │
  │  ├──────────────────────────────────────────┤   │
  │  │            IOKit (ドライバフレームワーク)  │   │
  │  │  C++ ベース、オブジェクト指向ドライバモデル│   │
  │  │  電力管理、ホットプラグ、デバイスツリー     │   │
  │  └──────────────────────────────────────────┘   │
  │                                                 │
  │  ┌──────────────────────────────────────────┐   │
  │  │   Secure Enclave Processor (SEP)          │   │
  │  │   独立プロセッサ / 独自OS (sepOS)         │   │
  │  │   暗号鍵管理 / 生体認証データ保護          │   │
  │  └──────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────┘
```

**XNU の設計上の特徴:**

- **Mach マイクロカーネル**: タスク管理、スレッド管理、仮想メモリ管理、IPC を担当。Mach ポートによるメッセージパッシングが IPC の基盤
- **BSD レイヤ**: POSIX 互換 API、ファイルシステム (APFS)、ネットワーキングスタック (TCP/IP)、ユーザ/グループ管理を提供
- **IOKit**: C++ で記述されたオブジェクト指向ドライバフレームワーク。電力管理やデバイスのホットプラグをサポート
- **Secure Enclave**: メインプロセッサとは物理的に分離された専用チップで、暗号鍵・生体認証データを管理

### 2.2 Android: 修正版 Linux カーネル

Android は Linux カーネルをベースとしているが、標準 Linux とは大きく異なる修正が加えられている。

```
Android システムアーキテクチャ詳細図
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────────────────────────────────┐
  │          アプリケーション層 (Applications)       │
  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐ │
  │  │ 電話   │ │ Chrome │ │カメラ  │ │サードパーティ││
  │  │        │ │        │ │        │ │ アプリ    ││
  │  └───┬────┘ └───┬────┘ └───┬────┘ └────┬─────┘ │
  ├──────┴──────────┴──────────┴────────────┴───────┤
  │        Android Framework (Java/Kotlin API)      │
  │  ┌────────────┬──────────────┬────────────────┐ │
  │  │Activity    │Content       │PackageManager  │ │
  │  │Manager     │Provider      │(アプリ管理)    │ │
  │  ├────────────┼──────────────┼────────────────┤ │
  │  │Window      │Notification  │Telephony       │ │
  │  │Manager     │Manager       │Manager         │ │
  │  ├────────────┼──────────────┼────────────────┤ │
  │  │Resource    │Location      │Sensor          │ │
  │  │Manager     │Manager       │Manager         │ │
  │  └────────────┴──────────────┴────────────────┘ │
  ├─────────────────────────────────────────────────┤
  │     Android Runtime (ART) / ネイティブライブラリ │
  │  ┌─────────────────┐  ┌──────────────────────┐ │
  │  │  ART             │  │ ネイティブライブラリ │ │
  │  │  ・AOT コンパイル│  │ ・libc (Bionic)     │ │
  │  │  ・GC (CC)       │  │ ・OpenGL ES / Vulkan│ │
  │  │  ・JNI           │  │ ・Media Framework   │ │
  │  │  ・プロファイル   │  │ ・SQLite            │ │
  │  │    ガイドコンパイル│  │ ・SSL (BoringSSL)  │ │
  │  └─────────────────┘  └──────────────────────┘ │
  ├─────────────────────────────────────────────────┤
  │        Hardware Abstraction Layer (HAL)          │
  │  ┌────────┬────────┬────────┬────────┬────────┐ │
  │  │Audio   │Camera  │Sensor  │Graphics│Power   │ │
  │  │HAL     │HAL     │HAL     │HAL     │HAL     │ │
  │  └────────┴────────┴────────┴────────┴────────┘ │
  │        HIDL/AIDL (HAL インターフェース定義言語)  │
  ├─────────────────────────────────────────────────┤
  │              Linux Kernel (修正版)               │
  │  ┌──────────┬──────────┬──────────┬───────────┐ │
  │  │ Binder   │ Ashmem / │ wakelocks│ Low Memory│ │
  │  │ IPC      │ ION      │ (電力)   │ Killer    │ │
  │  │ ドライバ │ メモリ   │          │ (OOM強化) │ │
  │  ├──────────┼──────────┼──────────┼───────────┤ │
  │  │ SELinux  │ cgroups  │ネットワーク│ ファイル │ │
  │  │ (強制   │ (リソース│ ドライバ  │ システム  │ │
  │  │  アクセス│  制御)   │           │ (ext4/   │ │
  │  │  制御)  │          │           │  f2fs)   │ │
  │  └──────────┴──────────┴──────────┴───────────┘ │
  └─────────────────────────────────────────────────┘
```

**Android 固有の Linux カーネル修正:**

| 機能 | 標準 Linux | Android 修正版 |
|------|-----------|---------------|
| IPC | SysV IPC, Unix Socket, D-Bus | Binder (高速 IPC) |
| メモリ共有 | POSIX shm, tmpfs | Ashmem → memfd (Android 10+) |
| GPU メモリ | DRM/GEM | ION → DMA-BUF heaps (Android 12+) |
| OOM 処理 | oom_killer (単純) | LowMemoryKiller (多段階) |
| 電力管理 | runtime PM | wakelocks (ユーザ空間制御可) |
| セキュリティ | DAC + AppArmor/SELinux | SELinux 強制 + seccomp |
| ロギング | syslog / journald | logd (リングバッファ) |

### 2.3 カーネル比較の詳細

```
XNU vs Linux (Android) ── 設計思想の違い
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌────────────────┬───────────────────────┬───────────────────────┐
│     観点       │   XNU (iOS)           │   Linux (Android)     │
├────────────────┼───────────────────────┼───────────────────────┤
│ カーネル型     │ ハイブリッド           │ モノリシック(モジュール)│
│ 設計起源       │ Mach + BSD 融合        │ Linus Torvalds 設計   │
│ IPC 基盤       │ Mach Port (msg pass)  │ Binder (Android固有)  │
│ ドライバ       │ IOKit (C++ OOP)       │ カーネルモジュール (C) │
│ スケジューラ   │ Mach 優先度ベース     │ CFS (Completely Fair) │
│ ファイルシステム│ APFS                  │ ext4 / f2fs           │
│ メモリ管理     │ Mach VM + jetsam      │ Linux VM + LMK        │
│ ライセンス     │ APSL + 非公開部分     │ GPL v2                │
│ コード行数     │ 約 1,200 万行         │ 約 3,000 万行以上     │
│ リリースサイクル│ 年1回 (iOS メジャー)  │ 年1回 + セキュリティ月次│
│ カスタマイズ性 │ Apple のみ            │ 任意のベンダーが可能   │
└────────────────┴───────────────────────┴───────────────────────┘
```

### 2.4 コード例: カーネル情報の取得

**コード例1: iOS (Swift) ── システム情報の取得**

```swift
import UIKit
import Darwin

/// iOS デバイスのシステム情報を取得するユーティリティ
struct SystemInfo {

    /// カーネルバージョンを取得
    static func kernelVersion() -> String {
        var size = 0
        // sysctl で XNU カーネルバージョンを取得
        sysctlbyname("kern.osrelease", nil, &size, nil, 0)
        var version = [CChar](repeating: 0, count: size)
        sysctlbyname("kern.osrelease", &version, &size, nil, 0)
        return String(cString: version)
    }

    /// 物理メモリ量を取得
    static func physicalMemory() -> UInt64 {
        return ProcessInfo.processInfo.physicalMemory
    }

    /// アクティブプロセッサ数を取得
    static func processorCount() -> Int {
        return ProcessInfo.processInfo.activeProcessorCount
    }

    /// サーマル状態を監視
    static func thermalState() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:   return "通常"
        case .fair:      return "やや高温"
        case .serious:   return "高温 - パフォーマンス低下中"
        case .critical:  return "危険 - 緊急スロットリング"
        @unknown default: return "不明"
        }
    }

    /// デバイス情報のサマリー
    static func summary() -> String {
        """
        ===== iOS デバイス情報 =====
        カーネル: Darwin \(kernelVersion())
        メモリ: \(physicalMemory() / 1_073_741_824) GB
        プロセッサ数: \(processorCount())
        サーマル状態: \(thermalState())
        OS: \(UIDevice.current.systemName) \(UIDevice.current.systemVersion)
        モデル: \(UIDevice.current.model)
        ================================
        """
    }
}

// 使用例
// print(SystemInfo.summary())
// 出力例:
// ===== iOS デバイス情報 =====
// カーネル: Darwin 24.1.0
// メモリ: 6 GB
// プロセッサ数: 6
// サーマル状態: 通常
// OS: iOS 18.2
// モデル: iPhone
// ================================
```

**コード例2: Android (Kotlin) ── システム情報の取得**

```kotlin
import android.os.Build
import android.app.ActivityManager
import android.content.Context
import java.io.File

/**
 * Android デバイスのシステム情報を取得するユーティリティ
 *
 * Android では /proc ファイルシステムを通じて
 * Linux カーネルの情報に直接アクセスできる。
 */
object SystemInfoUtil {

    /** Linux カーネルバージョンを取得 */
    fun getKernelVersion(): String {
        return try {
            File("/proc/version").readText().trim()
        } catch (e: Exception) {
            "取得失敗: ${e.message}"
        }
    }

    /** CPU 情報を取得 */
    fun getCpuInfo(): Map<String, String> {
        val info = mutableMapOf<String, String>()
        try {
            File("/proc/cpuinfo").readLines().forEach { line ->
                val parts = line.split(":")
                if (parts.size == 2) {
                    info[parts[0].trim()] = parts[1].trim()
                }
            }
        } catch (e: Exception) {
            info["error"] = e.message ?: "不明なエラー"
        }
        return info
    }

    /** メモリ情報を取得 */
    fun getMemoryInfo(context: Context): String {
        val activityManager = context.getSystemService(
            Context.ACTIVITY_SERVICE
        ) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)

        val totalMB = memInfo.totalMem / (1024 * 1024)
        val availMB = memInfo.availMem / (1024 * 1024)
        val usedMB = totalMB - availMB
        val usagePercent = (usedMB.toDouble() / totalMB * 100).toInt()

        return """
            |===== Android メモリ情報 =====
            |総メモリ: ${totalMB} MB
            |使用中:   ${usedMB} MB (${usagePercent}%)
            |空き:     ${availMB} MB
            |低メモリ状態: ${memInfo.lowMemory}
            |閾値:     ${memInfo.threshold / (1024 * 1024)} MB
            |==============================
        """.trimMargin()
    }

    /** ビルド情報を取得 */
    fun getBuildInfo(): String {
        return """
            |===== Android ビルド情報 =====
            |デバイス: ${Build.DEVICE}
            |モデル: ${Build.MODEL}
            |メーカー: ${Build.MANUFACTURER}
            |Android バージョン: ${Build.VERSION.RELEASE}
            |SDK レベル: ${Build.VERSION.SDK_INT}
            |セキュリティパッチ: ${Build.VERSION.SECURITY_PATCH}
            |ビルド番号: ${Build.DISPLAY}
            |ABI: ${Build.SUPPORTED_ABIS.joinToString(", ")}
            |===============================
        """.trimMargin()
    }

    /** SELinux の状態を取得 */
    fun getSeLinuxStatus(): String {
        return try {
            val process = Runtime.getRuntime().exec("getenforce")
            val result = process.inputStream.bufferedReader().readText().trim()
            "SELinux: $result"  // Enforcing, Permissive, or Disabled
        } catch (e: Exception) {
            "SELinux: 取得失敗"
        }
    }
}
```

---

## 3. プロセスモデルとアプリケーションライフサイクル

### 3.1 iOS のプロセスモデル

iOS ではすべてのアプリが独立したプロセスとして動作し、厳格なサンドボックス内で実行される。アプリのライフサイクルは OS によって厳密に管理され、バックグラウンドでの実行は極めて制限される。

```
iOS アプリケーションライフサイクル
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    ┌───────────────┐
                    │   未起動      │
                    │ (Not Running) │
                    └───────┬───────┘
                            │ ユーザがタップ / システムが起動
                            ▼
                    ┌───────────────┐
                    │   非アクティブ │
                    │  (Inactive)   │◄────── 電話着信、
                    └───────┬───────┘        通知表示 等
                            │
                            ▼
         ┌──────────────────────────────────────┐
         │            アクティブ                 │
         │           (Active)                   │
         │  ・フォアグラウンドで実行中            │
         │  ・UIイベントを受信可能               │
         │  ・全リソースにアクセス可能            │
         └──────────┬───────────────────────────┘
                    │ ホームに戻る / アプリ切替
                    ▼
         ┌──────────────────────────────────────┐
         │          バックグラウンド              │
         │        (Background)                  │
         │  ・約 30 秒間のタスク完了猶予          │
         │  ・特定API: 音楽再生, 位置情報,       │
         │    VoIP, Bluetooth, ダウンロード       │
         │  ・Background App Refresh (任意)      │
         └──────────┬───────────────────────────┘
                    │ リソース不足 / 一定時間経過
                    ▼
         ┌──────────────────────────────────────┐
         │           サスペンド                   │
         │         (Suspended)                  │
         │  ・メモリ上に残存するが実行されない     │
         │  ・CPU 時間ゼロ                       │
         │  ・メモリ不足時に OS が自動終了        │
         │    (jetsam メカニズム)                 │
         └──────────────────────────────────────┘

  ※ jetsam: iOS 独自のメモリ回収機構
     メモリ逼迫時にサスペンド中のアプリから優先的にプロセスを kill
     アプリごとのメモリ上限 (footprint limit) を超えると即時 kill
```

### 3.2 Android のプロセスモデル

Android はアプリを重要度に応じて 5 段階に分類し、メモリ不足時には低重要度のプロセスから順に終了させる。

```
Android プロセス優先度階層
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  重要度: 高
  ┌─────────────────────────────────────────────┐
  │  1. フォアグラウンドプロセス                  │
  │     ・ユーザが操作中の Activity               │
  │     ・onReceive() 実行中の BroadcastReceiver  │
  │     ・実行中の Service の onCreate/onStart    │
  │     → 最後まで kill しない                    │
  ├─────────────────────────────────────────────┤
  │  2. 可視プロセス                              │
  │     ・見えているが操作対象でない Activity      │
  │     ・foregroundService 実行中の Service      │
  │     → フォアグラウンド維持に必要なら kill      │
  ├─────────────────────────────────────────────┤
  │  3. サービスプロセス                          │
  │     ・startService() で起動されたサービス     │
  │     ・音楽再生、データ同期など                 │
  │     → 30分以上でキャッシュに降格              │
  ├─────────────────────────────────────────────┤
  │  4. キャッシュプロセス                        │
  │     ・非表示の Activity (onStop 済み)         │
  │     ・LRU リストで管理                        │
  │     → メモリ不足時に kill 対象               │
  ├─────────────────────────────────────────────┤
  │  5. 空プロセス                                │
  │     ・アクティブなコンポーネントなし           │
  │     ・キャッシュ目的でのみ保持                │
  │     → 最初に kill される                     │
  └─────────────────────────────────────────────┘
  重要度: 低

  LowMemoryKiller (LMK) の動作:
    adj 値    プロセス種別        kill 閾値 (例)
    ─────────────────────────────────────────
    0         フォアグラウンド    kill しない
    100       可視                72 MB 以下
    200       サービス            64 MB 以下
    700       キャッシュ (前回)   56 MB 以下
    900       キャッシュ (古い)   48 MB 以下
    906       キャッシュ (空)     40 MB 以下
```

### 3.3 Android の 4 大コンポーネント

Android アプリは 4 つの基本コンポーネントから構成される。各コンポーネントは独立したエントリーポイントとして機能する。

| コンポーネント | 役割 | ライフサイクル | 使用例 |
|--------------|------|-------------|--------|
| **Activity** | UIを持つ画面 | Created→Started→Resumed→Paused→Stopped→Destroyed | メイン画面、設定画面 |
| **Service** | バックグラウンド処理 | Created→Started→Destroyed / Created→Bound→Destroyed | 音楽再生、同期処理 |
| **BroadcastReceiver** | システムイベント受信 | onReceive() のみ | バッテリー低下通知、ネットワーク変化 |
| **ContentProvider** | データ共有 | onCreate() → CRUD操作 | 連絡先、メディアストア |

### 3.4 コード例: ライフサイクル管理

**コード例3: Android (Kotlin) ── Activity ライフサイクルの実装**

```kotlin
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ProcessLifecycleOwner

/**
 * Android Activity ライフサイクルの包括的な実装例
 *
 * 各コールバックがいつ呼ばれるかを理解することは、
 * メモリリークやクラッシュを防ぐ上で不可欠である。
 */
class MainLifecycleActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "LifecycleDemo"
        private const val KEY_COUNTER = "counter"
    }

    private var counter = 0

    // =========================================================
    // Activity ライフサイクルコールバック
    // =========================================================

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 状態の復元（画面回転等で再生成された場合）
        savedInstanceState?.let {
            counter = it.getInt(KEY_COUNTER, 0)
            Log.d(TAG, "状態復元: counter = $counter")
        }

        // プロセスレベルのライフサイクル監視
        ProcessLifecycleOwner.get().lifecycle.addObserver(
            AppLifecycleObserver()
        )

        Log.d(TAG, "onCreate: Activity が生成された")
    }

    override fun onStart() {
        super.onStart()
        // Activity が画面に表示される直前
        // ここで UI の更新やセンサーの登録を行う
        Log.d(TAG, "onStart: Activity が可視状態になった")
    }

    override fun onResume() {
        super.onResume()
        // Activity がフォアグラウンドに来た
        // ユーザ操作を受け付ける状態
        Log.d(TAG, "onResume: Activity がアクティブになった")
    }

    override fun onPause() {
        super.onPause()
        // 別の Activity が前面に来た (部分的に見える場合あり)
        // 重い処理をここで行ってはならない (ANR の原因)
        Log.d(TAG, "onPause: Activity が一時停止した")
    }

    override fun onStop() {
        super.onStop()
        // Activity が完全に非表示になった
        // データの保存やリソースの解放を行う
        Log.d(TAG, "onStop: Activity が停止した")
    }

    override fun onDestroy() {
        super.onDestroy()
        // Activity が破棄される
        // 最終的なクリーンアップ
        Log.d(TAG, "onDestroy: Activity が破棄された")
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        // 構成変更 (画面回転等) 前に状態を保存
        outState.putInt(KEY_COUNTER, counter)
        Log.d(TAG, "状態保存: counter = $counter")
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        // OS からのメモリ解放要求
        when (level) {
            TRIM_MEMORY_UI_HIDDEN ->
                Log.d(TAG, "UIが非表示 - UIキャッシュを解放可能")
            TRIM_MEMORY_RUNNING_LOW ->
                Log.w(TAG, "メモリ低下 - 不要なリソースを解放すべき")
            TRIM_MEMORY_RUNNING_CRITICAL ->
                Log.e(TAG, "メモリ危険 - 即座にリソースを解放")
            TRIM_MEMORY_COMPLETE ->
                Log.e(TAG, "バックグラウンドで最初にkillされる状態")
        }
    }

    // =========================================================
    // アプリケーション全体のライフサイクル監視
    // =========================================================

    /**
     * ProcessLifecycleOwner を使ったアプリ全体のライフサイクル監視
     * アプリがフォアグラウンド/バックグラウンドに遷移したことを検知する
     */
    inner class AppLifecycleObserver : LifecycleEventObserver {
        override fun onStateChanged(
            source: LifecycleOwner,
            event: Lifecycle.Event
        ) {
            when (event) {
                Lifecycle.Event.ON_START ->
                    Log.d(TAG, "アプリがフォアグラウンドに遷移")
                Lifecycle.Event.ON_STOP ->
                    Log.d(TAG, "アプリがバックグラウンドに遷移")
                else -> { /* その他のイベント */ }
            }
        }
    }
}
```

---

## 4. メモリ管理と仮想メモリ

### 4.1 iOS のメモリ管理: Jetsam

iOS には従来の意味でのスワップ領域が存在しない。NAND フラッシュへの過度な書き込みはストレージ寿命を短縮するためである。代わりに **jetsam** と呼ばれるメカニズムでメモリを管理する。

```
iOS Jetsam メモリ管理の概要
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  物理メモリ (例: 6 GB)
  ┌─────────────────────────────────────────────────┐
  │████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░│
  │◄─── カーネル+デーモン ──►◄── アプリ領域 ──────►│
  │     (約 1-2 GB 固定)      (約 4-5 GB 共有)     │
  └─────────────────────────────────────────────────┘

  アプリメモリ制限 (footprint limit):
  ┌──────────────┬──────────────┬──────────────────┐
  │ デバイス     │ 物理RAM      │ アプリ上限 (目安) │
  ├──────────────┼──────────────┼──────────────────┤
  │ iPhone SE(3) │ 4 GB         │ 約 1.4 GB        │
  │ iPhone 15    │ 6 GB         │ 約 2.8 GB        │
  │ iPhone 15 Pro│ 8 GB         │ 約 3.5 GB        │
  │ iPad Pro M4  │ 16 GB        │ 約 5.0 GB        │
  └──────────────┴──────────────┴──────────────────┘

  Jetsam の動作フロー:

  メモリ使用量増加
       │
       ▼
  ┌─────────────┐    No     ┌──────────────────┐
  │ 圧縮可能?   │─────────►│ footprint 超過?  │
  │ (WK compress)│          └───────┬──────────┘
  └──────┬──────┘                  │ Yes
         │ Yes                     ▼
         ▼                  ┌──────────────────┐
  ┌─────────────┐          │ 該当アプリを即時  │
  │ メモリ圧縮  │          │ kill (SIGKILL)    │
  │ (WKdm/LZ4) │          │ → クラッシュログ  │
  └──────┬──────┘          └──────────────────┘
         │
         ▼
  ┌─────────────┐
  │ まだ不足?   │─── No ──► 通常動作に復帰
  └──────┬──────┘
         │ Yes
         ▼
  ┌───────────────────────────────────────────┐
  │ 優先度の低いサスペンドアプリから順に kill  │
  │ (jetsam priority band に基づく)            │
  │                                           │
  │ Band 0-10:  バックグラウンドアプリ (最初)  │
  │ Band 10-20: メール、カレンダーなど          │
  │ Band 20+:   システムデーモン (最後)        │
  └───────────────────────────────────────────┘
```

### 4.2 Android のメモリ管理: LowMemoryKiller と zRAM

Android は Linux の OOM Killer を拡張した LowMemoryKiller を使用し、さらに zRAM によるメモリ圧縮を活用する。

```
Android メモリ管理アーキテクチャ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  物理メモリ (例: 8 GB)
  ┌────────────────────────────────────────────────────────┐
  │███████│██████████│░░░░░░░░░░░░░░░░░░│▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
  │Kernel │  zRAM    │   アプリ領域      │   ファイル    │
  │       │(圧縮RAM) │                   │   キャッシュ  │
  │~1.5GB │ ~2 GB    │    ~3 GB          │   ~1.5 GB     │
  └────────────────────────────────────────────────────────┘

  zRAM: メモリ内圧縮
  ┌──────────┐ 圧縮    ┌──────────┐
  │ ページ A │───────►│ 圧縮済A  │  通常 50-70% に圧縮
  │ (4 KB)   │        │ (~2 KB)  │  ディスクI/Oなし
  └──────────┘        └──────────┘  ← NAND寿命を保護

  LowMemoryKiller (lmkd) デーモン:
  ┌─────────────────────────────────────────────┐
  │ 1. /proc/meminfo を定期的に監視             │
  │ 2. 空きメモリが閾値を下回ったら:            │
  │    a. oom_adj_score の高い(重要度の低い)     │
  │       プロセスを特定                         │
  │    b. PSI (Pressure Stall Information) で   │
  │       メモリ圧力を測定 (Android 10+)        │
  │    c. 対象プロセスに SIGKILL を送信          │
  │ 3. 空きメモリが回復するまで繰り返す          │
  └─────────────────────────────────────────────┘
```

### 4.3 iOS と Android のメモリ管理比較

| 観点 | iOS | Android |
|------|-----|---------|
| スワップ | なし (圧縮のみ) | zRAM (メモリ内圧縮) |
| OOM 対策 | jetsam (proactive kill) | LowMemoryKiller (reactive kill) |
| 圧縮方式 | WKdm + LZ4 | LZ4 / LZO |
| GC | ARC (コンパイル時) | Tracing GC (ART) |
| メモリ警告 | didReceiveMemoryWarning | onTrimMemory / onLowMemory |
| 共有メモリ | Mach VM (copy-on-write) | Ashmem / mmap |
| アプリ上限 | footprint limit (厳格) | 設定可能 (largeHeap) |
| 圧縮比率 | 約 40-60% | 約 30-50% |

### 4.4 コード例: メモリ監視

**コード例4: iOS (Swift) ── メモリ使用量のモニタリング**

```swift
import Foundation
import os.log

/// iOS アプリのメモリ使用量を監視するユーティリティ
///
/// jetsam による強制終了を回避するため、
/// メモリ使用量を定期的に監視し、閾値を超えた場合に
/// キャッシュの解放等を行う。
class MemoryMonitor {

    private let logger = Logger(
        subsystem: "com.example.app",
        category: "memory"
    )

    private var timer: Timer?
    private let warningThreshold: UInt64  // バイト単位

    /// 警告閾値をMB単位で指定して初期化
    init(warningThresholdMB: UInt64 = 200) {
        self.warningThreshold = warningThresholdMB * 1024 * 1024
    }

    /// 現在のアプリメモリ使用量を取得 (バイト)
    func currentMemoryUsage() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size
        ) / 4

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(
                to: integer_t.self,
                capacity: Int(count)
            ) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0,
                    &count
                )
            }
        }

        guard result == KERN_SUCCESS else {
            return 0
        }
        return info.resident_size
    }

    /// メモリ使用量を MB 単位で返す
    func currentMemoryMB() -> Double {
        Double(currentMemoryUsage()) / (1024.0 * 1024.0)
    }

    /// 定期監視を開始 (指定間隔ごとにチェック)
    func startMonitoring(intervalSeconds: TimeInterval = 5.0) {
        timer = Timer.scheduledTimer(
            withTimeInterval: intervalSeconds,
            repeats: true
        ) { [weak self] _ in
            guard let self = self else { return }
            let usage = self.currentMemoryUsage()
            let usageMB = Double(usage) / (1024.0 * 1024.0)

            self.logger.info("メモリ使用量: \(usageMB, format: .fixed(precision: 1)) MB")

            if usage > self.warningThreshold {
                self.logger.warning(
                    "メモリ警告: \(usageMB, format: .fixed(precision: 1)) MB " +
                    "(閾値: \(Double(self.warningThreshold) / (1024.0 * 1024.0)) MB)"
                )
                // キャッシュの解放やリソースの縮小を実行
                self.handleMemoryPressure()
            }
        }
    }

    /// 監視を停止
    func stopMonitoring() {
        timer?.invalidate()
        timer = nil
    }

    /// メモリ圧力への対処
    private func handleMemoryPressure() {
        // 画像キャッシュのクリア
        URLCache.shared.removeAllCachedResponses()

        // カスタムキャッシュの解放
        NotificationCenter.default.post(
            name: .init("MemoryPressureWarning"),
            object: nil
        )

        logger.info("メモリ圧力対応: キャッシュをクリアしました")
    }
}

// 使用例:
// let monitor = MemoryMonitor(warningThresholdMB: 500)
// monitor.startMonitoring(intervalSeconds: 3.0)
// print("現在のメモリ: \(monitor.currentMemoryMB()) MB")
```


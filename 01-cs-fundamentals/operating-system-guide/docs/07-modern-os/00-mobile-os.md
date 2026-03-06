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

---

## 5. 電力管理とスケジューリング

### 5.1 モバイルCPUの電力最適化

モバイル端末における最大の制約はバッテリーである。SoC (System on a Chip) は CPU・GPU・NPU・モデム・ISP (Image Signal Processor) を単一チップに統合し、チップ間通信の電力を削減する。

```
モバイル SoC アーキテクチャ (例: Apple A17 Pro / Snapdragon 8 Gen 3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────────────────────────────────────┐
  │                    SoC パッケージ                    │
  │                                                     │
  │  ┌───────────────────────────────────────────────┐  │
  │  │              CPU クラスタ                      │  │
  │  │  ┌─────────────┐  ┌────────────────────────┐  │  │
  │  │  │ 高性能コア  │  │ 高効率コア             │  │  │
  │  │  │ (P-core)    │  │ (E-core)               │  │  │
  │  │  │ 2-4 コア    │  │ 4-6 コア               │  │  │
  │  │  │ 3-4 GHz     │  │ 1-2 GHz                │  │  │
  │  │  │ 高IPC       │  │ 低消費電力             │  │  │
  │  │  └─────────────┘  └────────────────────────┘  │  │
  │  └───────────────────────────────────────────────┘  │
  │                                                     │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
  │  │   GPU    │ │  NPU/    │ │  ISP     │            │
  │  │  (描画)  │ │  Neural  │ │ (カメラ  │            │
  │  │          │ │  Engine  │ │  処理)   │            │
  │  └──────────┘ └──────────┘ └──────────┘            │
  │                                                     │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
  │  │ モデム   │ │ Secure   │ │ メモリ   │            │
  │  │ (5G/LTE) │ │ Enclave  │ │コントローラ│            │
  │  └──────────┘ └──────────┘ └──────────┘            │
  └─────────────────────────────────────────────────────┘

  DVFS (Dynamic Voltage and Frequency Scaling):
  ─────────────────────────────────────────────
  負荷に応じて CPU の電圧と周波数を動的に変更する

  消費電力 ∝ C × V² × f
    C = キャパシタンス (回路の複雑さ)
    V = 動作電圧
    f = クロック周波数

  → 電圧を半分にすれば消費電力は 1/4 に
  → 軽いタスクは E-core + 低周波数で処理
  → 重いタスクのみ P-core + 高周波数を使用
```

### 5.2 iOS の電力管理

iOS は以下の多層的な電力管理メカニズムを実装している。

| 機能 | 説明 | 効果 |
|------|------|------|
| **Background App Refresh** | 使用パターンに基づいてバックグラウンド更新タイミングを最適化 | 不要な起動を排除 |
| **App Nap** | 非表示のアプリのタイマー・ネットワーク活動を抑制 | CPU 使用を最小化 |
| **Coalesced Timer** | 複数アプリのタイマーをまとめて処理 | CPU 起床回数を削減 |
| **Discretionary Background Tasks** | OS が最適な実行タイミングを決定 | 充電中・WiFi 接続時に実行 |
| **Low Power Mode** | ユーザ操作で省電力モードを有効化 | バックグラウンド停止、描画レート低下、5G無効化 |
| **Thermal Throttling** | サーマルセンサーに基づく段階的パフォーマンス制限 | 過熱を防止 |

### 5.3 Android の電力管理: Doze と App Standby

Android 6.0 で導入された Doze モードは、端末が静止・画面オフ・未充電の状態が続くと段階的に制限を強化する。

```
Android Doze モードの段階的制限
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  画面ON                    画面OFF + 静止 + 未充電
  ──────►                  ──────────────────────►
                                    時間経過
  ┌──────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │通常  │  │ Light    │  │ Deep     │  │ Deep     │
  │動作  │→│ Doze    │→│ Doze    │→│ Doze    │
  │      │  │ (軽度)   │  │ (初期)   │  │ (完全)   │
  └──────┘  └──────────┘  └──────────┘  └──────────┘

  Light Doze (数分後):
  ├── ネットワークアクセス: 制限
  ├── ジョブ/同期: 延期
  ├── アラーム: 延期 (setExactAndAllowWhileIdle は許可)
  └── GPS: 停止

  Deep Doze (約30分後):
  ├── ネットワークアクセス: 停止
  ├── WiFi スキャン: 停止
  ├── wakelocks: 無視
  ├── アラーム: 延期
  └── ジョブ/同期: すべて延期

  メンテナンスウィンドウ:
  ┌───┐     ┌───┐          ┌───┐               ┌───┐
  │   │     │   │          │   │               │   │
  │処理│     │処理│          │処理│               │処理│
  │   │     │   │          │   │               │   │
  └─┬─┘     └─┬─┘          └─┬─┘               └─┬─┘
    │  Doze    │    Doze      │      Doze          │
    ├─────────┤              │                    │
    │ 短い間隔 │              │                    │
    ├─────────┴──────────────┤                    │
    │       間隔が徐々に延びる│                    │
    ├────────────────────────┴────────────────────┤
    │              最大間隔は数時間                │

  App Standby Buckets (Android 9+):
  ┌───────────┬──────────────┬──────────────────────┐
  │ バケット   │ ジョブ上限   │ 条件                 │
  ├───────────┼──────────────┼──────────────────────┤
  │ Active    │ 制限なし     │ 現在使用中            │
  │ Working   │ 2時間に1回   │ 頻繁に使用            │
  │ Frequent  │ 8時間に1回   │ 定期的に使用          │
  │ Rare      │ 24時間に1回  │ めったに使わない      │
  │ Restricted│ 1日1回       │ ほぼ使わない(Android12)│
  └───────────┴──────────────┴──────────────────────┘
```

### 5.4 CPU スケジューリングの比較

| 観点 | iOS (XNU) | Android (Linux) |
|------|-----------|----------------|
| スケジューラ | Mach 優先度スケジューラ | CFS (Completely Fair Scheduler) |
| 優先度レベル | 128 段階 (0-127) | nice 値 (-20 to 19) + RT 優先度 |
| リアルタイム | FIFO / RR (Mach RT threads) | SCHED_FIFO / SCHED_RR |
| QoS 分類 | UserInteractive, UserInitiated, Utility, Background | THREAD_PRIORITY_* |
| big.LITTLE 制御 | OS が完全制御 (非公開アルゴリズム) | EAS (Energy Aware Scheduling) |
| UI 優先 | Main Thread 最高優先度 | RenderThread 優先 (Android 11+) |

---

## 6. セキュリティモデルとサンドボックス

### 6.1 多層防御アーキテクチャ

モバイルOSのセキュリティは、ハードウェアからアプリケーション層まで一貫した多層防御 (Defense in Depth) で設計されている。

```
モバイルOS セキュリティ多層防御モデル
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  攻撃者 ─────────────────────────────────────────────►

  Layer 7: アプリケーション層
  ┌─────────────────────────────────────────────────┐
  │ ・App Store / Play Store 審査                   │
  │ ・コード署名 (必須)                              │
  │ ・Runtime 権限チェック                           │
  └─────────────────────────────┬───────────────────┘
                                │ 突破した場合
  Layer 6: サンドボックス層      ▼
  ┌─────────────────────────────────────────────────┐
  │ ・iOS: App Sandbox (Seatbelt)                   │
  │ ・Android: SELinux + UID 分離 + seccomp-bpf     │
  │ ・各アプリは隔離された環境で実行                  │
  └─────────────────────────────┬───────────────────┘
                                │ 突破した場合
  Layer 5: ランタイム保護層      ▼
  ┌─────────────────────────────────────────────────┐
  │ ・ASLR (Address Space Layout Randomization)     │
  │ ・Stack Canary / Stack Protector                │
  │ ・PAC (Pointer Authentication, ARM v8.3+)       │
  │ ・MTE (Memory Tagging Extension, Android 14+)   │
  │ ・CFI (Control Flow Integrity)                  │
  └─────────────────────────────┬───────────────────┘
                                │ 突破した場合
  Layer 4: カーネル保護層        ▼
  ┌─────────────────────────────────────────────────┐
  │ ・KTRR / CTRR (iOS: カーネルテキスト読取専用)    │
  │ ・KASLR (カーネル ASLR)                         │
  │ ・W^X (Write XOR Execute) ポリシー              │
  │ ・PPL (Page Protection Layer, iOS)              │
  └─────────────────────────────┬───────────────────┘
                                │ 突破した場合
  Layer 3: ファームウェア層      ▼
  ┌─────────────────────────────────────────────────┐
  │ ・Secure Boot Chain (ブート検証の連鎖)           │
  │ ・iOS: iBoot → カーネル → kext 全段階で署名検証 │
  │ ・Android: Verified Boot (AVB 2.0)              │
  │ ・ブートローダロック                             │
  └─────────────────────────────┬───────────────────┘
                                │ 突破した場合
  Layer 2: ハードウェアセキュリティ ▼
  ┌─────────────────────────────────────────────────┐
  │ ・Secure Enclave (iOS) / Titan M (Pixel)        │
  │ ・TrustZone (ARM)                               │
  │ ・eFuse (ワンタイムプログラマブル)               │
  │ ・物理攻撃対策 (tamper detection)                │
  └─────────────────────────────┬───────────────────┘
                                │ 突破した場合
  Layer 1: 暗号化基盤            ▼
  ┌─────────────────────────────────────────────────┐
  │ ・FBE (File-Based Encryption)                   │
  │ ・AES-256-XTS (ストレージ暗号化)                │
  │ ・ハードウェアバウンド暗号鍵 (SEPから取得不可)   │
  │ ・Secure Element (eSIM, NFC決済)                │
  └─────────────────────────────────────────────────┘
```

### 6.2 サンドボックスの実装詳細

**iOS サンドボックス (Seatbelt / sandbox_init)**

iOS では各アプリに専用のコンテナディレクトリが割り当てられ、そこからの脱出は原則不可能である。

```
iOS アプリ サンドボックス構造
━━━━━━━━━━━━━━━━━━━━━━━━━━

  /var/mobile/Containers/
  ├── Bundle/
  │   └── Application/
  │       └── <UUID>/
  │           └── MyApp.app/     ← 読み取り専用
  │               ├── MyApp      (実行バイナリ)
  │               ├── Info.plist
  │               └── Assets/
  │
  └── Data/
      └── Application/
          └── <UUID>/            ← アプリ固有のデータ領域
              ├── Documents/     ← ユーザデータ (iCloud同期可)
              ├── Library/
              │   ├── Caches/    ← キャッシュ (OS削除可)
              │   └── Preferences/ ← UserDefaults
              ├── tmp/           ← 一時ファイル (OS削除可)
              └── SystemData/    ← システム管理データ

  アクセス制御:
  ┌────────────────────────┬──────────────────────┐
  │ リソース               │ アクセス              │
  ├────────────────────────┼──────────────────────┤
  │ 自アプリのコンテナ     │ 読み書き可            │
  │ 他アプリのコンテナ     │ アクセス不可          │
  │ システムファイル       │ アクセス不可          │
  │ カメラ/マイク          │ ユーザ許可が必要      │
  │ 位置情報              │ ユーザ許可が必要      │
  │ 連絡先/カレンダー     │ ユーザ許可が必要      │
  │ ネットワーク          │ 許可制 (iOS 14+)     │
  │ Bluetooth             │ ユーザ許可が必要      │
  │ Keychain              │ 同一チームIDのみ共有  │
  └────────────────────────┴──────────────────────┘
```

**Android のセキュリティ境界**

Android は Linux の UID ベースの分離と SELinux の強制アクセス制御を組み合わせる。

```
Android アプリ分離モデル
━━━━━━━━━━━━━━━━━━━━━━━━

  インストール時:
  ┌──────────────────────────────────────────┐
  │ PackageManager がアプリに固有の           │
  │ Linux UID/GID を割り当て                  │
  │                                          │
  │ com.example.app1 → UID 10045             │
  │ com.example.app2 → UID 10046             │
  │ com.example.app3 → UID 10047             │
  └──────────────────────────────────────────┘

  データディレクトリ:
  /data/data/com.example.app1/   (UID 10045 のみアクセス可)
  ├── databases/                 (SQLite データベース)
  ├── shared_prefs/              (SharedPreferences XML)
  ├── files/                     (内部ストレージ)
  └── cache/                     (キャッシュ)

  SELinux ポリシー (Android 5.0+):
  ┌────────────────────────────────────────────────┐
  │ 全アプリに SELinux コンテキストが割り当てられる │
  │                                                │
  │ untrusted_app : 通常のサードパーティアプリ      │
  │ platform_app  : system パーティションのアプリ   │
  │ priv_app      : 特権アプリ                     │
  │ isolated_app  : WebView レンダラなど (最小権限) │
  │                                                │
  │ ポリシー例:                                    │
  │ allow untrusted_app app_data_file:file          │
  │   { read write create };                       │
  │ neverallow untrusted_app system_data_file:file  │
  │   { read write };                              │
  └────────────────────────────────────────────────┘

  Scoped Storage (Android 10+):
  ┌────────────────────────────────────────────────┐
  │ 外部ストレージへのアクセスを MediaStore 経由に  │
  │ 制限し、他アプリのファイルに直接アクセス不可     │
  │                                                │
  │ /storage/emulated/0/                           │
  │ ├── DCIM/     ← MediaStore.Images 経由         │
  │ ├── Music/    ← MediaStore.Audio 経由          │
  │ ├── Download/ ← SAF (Storage Access Framework)  │
  │ └── Android/data/com.example.app/              │
  │               ← アプリ専用 (他からアクセス不可) │
  └────────────────────────────────────────────────┘
```

### 6.3 権限モデルの進化

| バージョン | iOS | Android |
|-----------|-----|---------|
| 初期 | インストール時に全権限付与 | インストール時に全権限付与 |
| 過渡期 | iOS 6: 一部の権限で実行時許可 | Android 6.0: 危険な権限は実行時許可 |
| 現在 | iOS 14+: 概算位置、限定写真アクセス、Appトラッキング透明性 | Android 12+: 概算位置、近くのデバイス権限、写真ピッカー |
| 最新 | iOS 17+: 権限のリセット、通信安全性 | Android 14+: 写真/動画の部分アクセス、健康データ専用権限 |

---

## 7. プロセス間通信とプッシュ通知

### 7.1 IPC メカニズムの比較

```
モバイルOS における IPC メカニズム
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  iOS の IPC:
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ┌─────────┐  Mach Port   ┌─────────┐             │
  │  │ App A   │─────────────►│ App B   │  限定的     │
  │  └─────────┘  (カーネル経由)└─────────┘             │
  │                                                    │
  │  主な IPC 手段:                                    │
  │  1. XPC Services    ─ 特権分離されたヘルパー        │
  │  2. URL Scheme      ─ アプリ間のディープリンク       │
  │  3. Universal Links ─ HTTP URL からのアプリ起動     │
  │  4. App Groups      ─ 同一開発者のアプリ間データ共有 │
  │  5. UIPasteboard    ─ クリップボード経由            │
  │  6. Extensions      ─ Share, Today, Action         │
  │  7. App Intents     ─ Siri / ショートカット統合     │
  │                                                    │
  │  制約: 任意のアプリ間で直接通信は不可               │
  │       すべて OS が仲介する                         │
  └────────────────────────────────────────────────────┘

  Android の IPC (Binder):
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ┌─────────┐   Binder    ┌─────────┐              │
  │  │ App A   │────────────►│ App B   │              │
  │  │ (Client)│  /dev/binder│(Server) │              │
  │  └─────────┘     │       └─────────┘              │
  │                  │                                 │
  │          ┌───────┴───────┐                         │
  │          │ Binder Driver │  カーネル空間            │
  │          │ (1回コピー)   │  copy_from_user →       │
  │          │               │  target の mmap 領域へ   │
  │          └───────────────┘                         │
  │                                                    │
  │  主な IPC 手段:                                    │
  │  1. Intent         ─ コンポーネント間メッセージング │
  │  2. AIDL           ─ プロセス間メソッド呼び出し     │
  │  3. ContentProvider─ 構造化データの共有             │
  │  4. Messenger      ─ Handler ベースのメッセージ     │
  │  5. BroadcastReceiver ─ システムイベント通知        │
  │  6. FileProvider   ─ ファイル共有 (URI 経由)       │
  │                                                    │
  │  Binder の特長:                                    │
  │  ・1回のメモリコピーで転送 (送信→カーネルmmap)     │
  │  ・呼び出し元の UID/PID を自動検証 (なりすまし防止) │
  │  ・参照カウント管理 (death notification)            │
  │  ・16MB のトランザクションバッファ上限              │
  └────────────────────────────────────────────────────┘
```

### 7.2 プッシュ通知アーキテクチャ

プッシュ通知は、各アプリが独自にサーバとの常時接続を維持する代わりに、OS が一元管理する単一の接続を共有する仕組みである。

```
プッシュ通知のアーキテクチャ比較
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  iOS (APNs: Apple Push Notification service):
  ┌──────────┐    HTTPS     ┌──────────┐   TLS 常時接続  ┌──────────┐
  │ アプリ   │─────────────►│  APNs    │───────────────►│  iOS     │
  │ サーバ   │  証明書認証   │  サーバ  │  Port 5223     │  デバイス│
  └──────────┘              └──────────┘                └──────────┘
       │                         │                          │
       │ 1. デバイストークン取得  │                          │
       │◄────────────────────────┤◄─────────────────────────┤
       │                         │ 2. 通知ペイロード送信     │
       │────────────────────────►│                          │
       │                         │ 3. デバイスへ配信         │
       │                         │─────────────────────────►│
       │                         │                     4. アプリに通知
       │                         │                     UNUserNotification
       │                         │                     Center で処理

  Android (FCM: Firebase Cloud Messaging):
  ┌──────────┐    HTTPS     ┌──────────┐   XMPP/HTTPS   ┌──────────┐
  │ アプリ   │─────────────►│  FCM     │───────────────►│ Android  │
  │ サーバ   │  APIキー認証  │  サーバ  │  GMS 経由      │  デバイス│
  └──────────┘              └──────────┘                └──────────┘
       │                         │                          │
       │ 1. 登録トークン取得      │                          │
       │◄────────────────────────┤◄─────────────────────────┤
       │                         │ 2. メッセージ送信         │
       │────────────────────────►│                          │
       │                         │ 3. GCM Connection        │
       │                         │    Service 経由で配信     │
       │                         │─────────────────────────►│
       │                         │                     4. onMessageReceived
       │                         │                     FirebaseMessaging
       │                         │                     Service で処理

  比較:
  ┌─────────────────┬──────────────────┬──────────────────┐
  │ 項目            │ APNs (iOS)       │ FCM (Android)    │
  ├─────────────────┼──────────────────┼──────────────────┤
  │ 接続プロトコル  │ 独自 (TLS)       │ XMPP / HTTP/2    │
  │ ペイロード上限  │ 4 KB             │ 4 KB (データ)    │
  │ 優先度          │ 5 (即時) / 1-4   │ High / Normal    │
  │ トピック購読    │ あり             │ あり             │
  │ サイレント通知  │ content-available│ data message     │
  │ Doze 中の配信   │ N/A              │ High のみ配信    │
  │ GMS 依存        │ なし (OS 内蔵)   │ あり (GMS必須)   │
  │ 信頼性          │ Best Effort      │ Best Effort      │
  └─────────────────┴──────────────────┴──────────────────┘
```

### 7.3 コード例: プッシュ通知の実装

**コード例5: Android (Kotlin) ── FCM 通知の受信と処理**

```kotlin
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage

/**
 * Firebase Cloud Messaging (FCM) の通知受信サービス
 *
 * このサービスは以下の2種類のメッセージを処理する:
 * 1. 通知メッセージ (notification) - OS が自動表示
 * 2. データメッセージ (data) - アプリが処理を制御
 *
 * Doze モード中でも High Priority メッセージは配信される。
 */
class AppFirebaseMessagingService : FirebaseMessagingService() {

    companion object {
        private const val TAG = "FCMService"
        private const val CHANNEL_ID = "default_channel"
        private const val CHANNEL_NAME = "一般通知"
    }

    /**
     * FCM トークンが更新された時に呼ばれる
     * 新しいトークンをアプリサーバに送信する必要がある
     */
    override fun onNewToken(token: String) {
        super.onNewToken(token)
        Log.d(TAG, "FCM トークン更新: $token")
        sendTokenToServer(token)
    }

    /**
     * メッセージを受信した時に呼ばれる
     *
     * 注意:
     * - フォアグラウンド: notification + data 両方ここで処理
     * - バックグラウンド: notification は OS が表示、data のみここ
     */
    override fun onMessageReceived(remoteMessage: RemoteMessage) {
        super.onMessageReceived(remoteMessage)

        Log.d(TAG, "送信元: ${remoteMessage.from}")
        Log.d(TAG, "メッセージID: ${remoteMessage.messageId}")

        // データペイロードの処理
        if (remoteMessage.data.isNotEmpty()) {
            Log.d(TAG, "データ: ${remoteMessage.data}")
            handleDataMessage(remoteMessage.data)
        }

        // 通知ペイロードの処理 (フォアグラウンド時)
        remoteMessage.notification?.let { notification ->
            Log.d(TAG, "通知タイトル: ${notification.title}")
            Log.d(TAG, "通知本文: ${notification.body}")
            showNotification(
                title = notification.title ?: "通知",
                body = notification.body ?: "",
                data = remoteMessage.data
            )
        }
    }

    /**
     * データメッセージの処理
     * バックグラウンド同期やサイレント更新に使用
     */
    private fun handleDataMessage(data: Map<String, String>) {
        val type = data["type"] ?: return
        when (type) {
            "sync" -> {
                // バックグラウンドデータ同期をスケジュール
                Log.d(TAG, "同期リクエスト受信")
            }
            "update" -> {
                // アプリ内データの更新
                val payload = data["payload"]
                Log.d(TAG, "更新データ: $payload")
            }
            "silent" -> {
                // サイレント通知 (UI表示なし)
                Log.d(TAG, "サイレント処理実行")
            }
        }
    }

    /**
     * 通知の表示
     * Android 8.0+ では NotificationChannel が必須
     */
    private fun showNotification(
        title: String,
        body: String,
        data: Map<String, String>
    ) {
        val notificationManager = getSystemService(
            Context.NOTIFICATION_SERVICE
        ) as NotificationManager

        // Android 8.0+ 用の通知チャンネル作成
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                CHANNEL_NAME,
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "アプリの一般通知"
                enableVibration(true)
            }
            notificationManager.createNotificationChannel(channel)
        }

        // タップ時に開くActivityの設定
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_CLEAR_TOP
            data.forEach { (key, value) -> putExtra(key, value) }
        }
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_ONE_SHOT or PendingIntent.FLAG_IMMUTABLE
        )

        // 通知の構築と表示
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(title)
            .setContentText(body)
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setContentIntent(pendingIntent)
            .build()

        notificationManager.notify(
            System.currentTimeMillis().toInt(),
            notification
        )
    }

    /** FCM トークンをアプリサーバに送信 */
    private fun sendTokenToServer(token: String) {
        // 実装: APIサーバにHTTPS POSTで送信
        // RetrofitやOkHttpを使用するのが一般的
        Log.d(TAG, "トークンをサーバに送信: $token")
    }
}
```

---

## 8. センサー・ハードウェア抽象化レイヤ

### 8.1 モバイル端末のセンサー群

現代のスマートフォンには 10 種類以上のセンサーが搭載されている。これらのセンサーは OS のハードウェア抽象化レイヤ (HAL) を通じてアプリケーションに統一的な API を提供する。

```
モバイル端末搭載センサー一覧
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌──────────────────────────────────────────────────┐
  │                スマートフォン断面図               │
  │                                                  │
  │  [前面カメラ] [近接] [環境光] [Face ID/IRドット]  │
  │  ┌────────────────────────────────────────────┐  │
  │  │                                            │  │
  │  │           ディスプレイ                      │  │
  │  │         (タッチセンサー内蔵)                │  │
  │  │         (画面内指紋センサー)                │  │
  │  │                                            │  │
  │  ├────────────────────────────────────────────┤  │
  │  │  [加速度]  [ジャイロ]  [磁気]  [気圧]     │  │
  │  │  [GPS/GNSS]  [UWB]  [WiFi RTT]            │  │
  │  │  [NFC]  [温度]  [マイク x2-4]             │  │
  │  │  [振動モータ (Haptic Engine)]              │  │
  │  ├────────────────────────────────────────────┤  │
  │  │  [背面カメラ x2-4] [LiDAR] [ToF]          │  │
  │  └────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────┘

  センサー分類:
  ┌──────────────┬────────────────────┬───────────────────┐
  │ カテゴリ     │ センサー           │ 用途               │
  ├──────────────┼────────────────────┼───────────────────┤
  │ 動き         │ 加速度計           │ 歩数、傾き検知     │
  │              │ ジャイロスコープ   │ 回転検知、AR       │
  │              │ 磁力計             │ コンパス           │
  ├──────────────┼────────────────────┼───────────────────┤
  │ 環境         │ 気圧計             │ 高度測定、天気     │
  │              │ 環境光センサー     │ 画面輝度調整       │
  │              │ 温度センサー       │ バッテリー/SoC温度 │
  │              │ 湿度センサー       │ 環境モニタリング   │
  ├──────────────┼────────────────────┼───────────────────┤
  │ 位置         │ GPS/GNSS           │ 位置測位           │
  │              │ WiFi RTT           │ 屋内測位           │
  │              │ UWB                │ 近距離高精度測位   │
  ├──────────────┼────────────────────┼───────────────────┤
  │ 近接/深度    │ 近接センサー       │ 通話時画面消灯     │
  │              │ LiDAR/ToF          │ 3D空間スキャン     │
  │              │ 構造化光 (Face ID) │ 顔認証             │
  ├──────────────┼────────────────────┼───────────────────┤
  │ 生体         │ 指紋センサー       │ 認証               │
  │              │ TrueDepth カメラ   │ Face ID            │
  ├──────────────┼────────────────────┼───────────────────┤
  │ 通信         │ NFC                │ 非接触決済         │
  │              │ Bluetooth 5.x     │ 周辺機器接続       │
  │              │ WiFi 6E/7          │ 高速通信           │
  └──────────────┴────────────────────┴───────────────────┘
```

### 8.2 HAL (Hardware Abstraction Layer) の設計

```
iOS IOKit vs Android HAL ── ドライバ抽象化の比較
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  iOS (IOKit):
  ┌─────────────────────────────────────────┐
  │ アプリケーション (Swift/ObjC)           │
  ├─────────────────────────────────────────┤
  │ Framework (CoreMotion, CoreLocation)   │
  ├─────────────────────────────────────────┤
  │ IOKit User Client (ユーザ空間ドライバ) │
  ├─────────────────────────────────────────┤
  │ IOKit Kernel (C++ オブジェクト階層)     │
  │  IOService → IOHIDDevice               │
  │           → IOAccelerator              │
  │           → AppleSensorKit             │
  ├─────────────────────────────────────────┤
  │ ハードウェア                            │
  └─────────────────────────────────────────┘

  Android (Project Treble 以降 HIDL/AIDL HAL):
  ┌─────────────────────────────────────────┐
  │ アプリケーション (Kotlin/Java)          │
  ├─────────────────────────────────────────┤
  │ Framework API (SensorManager 等)       │
  ├─────────────────────────────────────────┤
  │ System Server (system_server プロセス)  │
  ├─────────────────────────────────────────┤
  │ HIDL/AIDL HAL Interface                 │
  │  (android.hardware.sensors@2.1)         │
  │  ← ベンダーとフレームワークの境界       │
  ├─────────────────────────────────────────┤
  │ Vendor HAL 実装 (ベンダー提供)          │
  │  (Qualcomm, Samsung, MediaTek 等)       │
  ├─────────────────────────────────────────┤
  │ カーネルドライバ                        │
  ├─────────────────────────────────────────┤
  │ ハードウェア                            │
  └─────────────────────────────────────────┘

  Project Treble の意義:
  ┌──────────────────────────────────────────────┐
  │ Before Treble:                                │
  │  OS更新 = Google → SoC → OEM → キャリア     │
  │  各段階でHAL修正が必要 → 更新に数ヶ月〜年    │
  │                                               │
  │ After Treble:                                 │
  │  HALインターフェースが安定                     │
  │  OS更新時にベンダーHALの再コンパイル不要       │
  │  Generic System Image (GSI) でテスト可能       │
  │  → 更新速度が大幅に改善                       │
  └──────────────────────────────────────────────┘
```

### 8.3 低消費電力コプロセッサ

メインCPUを起動せずにセンサーデータを処理する専用チップが、電力効率を大幅に向上させる。

| コプロセッサ | プラットフォーム | 機能 | 消費電力 |
|-------------|----------------|------|---------|
| Apple Motion Coprocessor (M-series) | iOS | 加速度・ジャイロ・気圧の常時監視、歩数カウント | メインCPUの約 1/100 |
| Sensor Hub (Qualcomm SSC) | Android (Snapdragon) | Always-on センサー処理、Activity Recognition | 数 mW |
| Samsung CHUB | Android (Exynos) | 環境センサー処理、ジェスチャー検知 | 数 mW |
| Google Tensor Context Hub | Android (Pixel) | 顔検知、音楽認識、環境コンテキスト推定 | 数 mW |

### 8.4 コード例: センサーの利用

**コード例6: iOS (Swift) ── CoreMotion を用いた動作検知**

```swift
import CoreMotion
import Foundation

/**
 * CoreMotion を使ったデバイスの動き検知
 *
 * Motion Coprocessor により、アプリがバックグラウンドでも
 * 低消費電力でセンサーデータを収集できる。
 *
 * 歩数計測、階段昇降検知、車両乗車検知などに利用する。
 */
class MotionDetector {

    private let motionManager = CMMotionManager()
    private let pedometer = CMPedometer()
    private let activityManager = CMMotionActivityManager()
    private let altimeter = CMAltimeter()

    // ===================================================
    // 加速度・ジャイロの生データ取得
    // ===================================================

    /// 加速度センサーを開始 (ゲーム、AR 用)
    func startAccelerometer(
        interval: TimeInterval = 0.1,
        handler: @escaping (CMAccelerometerData) -> Void
    ) {
        guard motionManager.isAccelerometerAvailable else {
            print("加速度センサーが利用不可")
            return
        }

        motionManager.accelerometerUpdateInterval = interval
        motionManager.startAccelerometerUpdates(
            to: .main
        ) { data, error in
            guard let data = data, error == nil else { return }
            // data.acceleration.x/y/z は重力加速度を含む (-1.0 ~ +1.0 G)
            handler(data)
        }
    }

    /// デバイスモーション (センサーフュージョン結果)
    func startDeviceMotion(
        interval: TimeInterval = 0.05,
        handler: @escaping (CMDeviceMotion) -> Void
    ) {
        guard motionManager.isDeviceMotionAvailable else { return }

        motionManager.deviceMotionUpdateInterval = interval
        motionManager.startDeviceMotionUpdates(
            using: .xArbitraryZVertical,
            to: .main
        ) { motion, error in
            guard let motion = motion, error == nil else { return }
            // motion.attitude    : 姿勢 (roll, pitch, yaw)
            // motion.rotationRate: 角速度
            // motion.gravity     : 重力成分
            // motion.userAcceleration: ユーザの加速度 (重力除去済み)
            handler(motion)
        }
    }

    // ===================================================
    // 歩数計測 (Motion Coprocessor 利用)
    // ===================================================

    /// 歩数のリアルタイム計測を開始
    func startPedometer(
        handler: @escaping (Int, Double) -> Void
    ) {
        guard CMPedometer.isStepCountingAvailable() else {
            print("歩数計測が利用不可")
            return
        }

        pedometer.startUpdates(from: Date()) { data, error in
            guard let data = data, error == nil else { return }
            let steps = data.numberOfSteps.intValue
            let distance = data.distance?.doubleValue ?? 0
            handler(steps, distance)
        }
    }

    // ===================================================
    // アクティビティ認識 (歩行/走行/自動車/自転車)
    // ===================================================

    /// ユーザのアクティビティタイプを検知
    func startActivityRecognition(
        handler: @escaping (String) -> Void
    ) {
        guard CMMotionActivityManager.isActivityAvailable() else {
            return
        }

        activityManager.startActivityUpdates(
            to: .main
        ) { activity in
            guard let activity = activity else { return }
            var type = "不明"
            if activity.walking   { type = "歩行" }
            if activity.running   { type = "走行" }
            if activity.cycling   { type = "自転車" }
            if activity.automotive{ type = "自動車" }
            if activity.stationary{ type = "静止" }
            handler(type)
        }
    }

    // ===================================================
    // クリーンアップ
    // ===================================================

    func stopAll() {
        motionManager.stopAccelerometerUpdates()
        motionManager.stopDeviceMotionUpdates()
        pedometer.stopUpdates()
        activityManager.stopActivityUpdates()
    }
}

// 使用例:
// let detector = MotionDetector()
// detector.startPedometer { steps, distance in
//     print("歩数: \(steps), 距離: \(distance)m")
// }
// detector.startActivityRecognition { type in
//     print("現在の活動: \(type)")
// }
```

### 8.5 コード例: Android センサーの利用

**コード例7: Android (Kotlin) ── SensorManager による加速度検知**

```kotlin
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import kotlin.math.sqrt

/**
 * Android SensorManager を使った加速度とシェイク検知
 *
 * HAL を通じてベンダー固有のセンサーハードウェアに
 * 統一的にアクセスする例を示す。
 *
 * 電力効率のため、不要になったら必ず unregister すること。
 */
class ShakeDetector(
    private val context: Context,
    private val onShake: () -> Unit
) : SensorEventListener {

    companion object {
        private const val TAG = "ShakeDetector"
        private const val SHAKE_THRESHOLD = 12.0f  // m/s^2
        private const val MIN_TIME_BETWEEN_SHAKES = 1000L  // ms
    }

    private val sensorManager: SensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private var lastShakeTime = 0L

    /** センサーの登録と開始 */
    fun start() {
        val accelerometer = sensorManager.getDefaultSensor(
            Sensor.TYPE_LINEAR_ACCELERATION  // 重力成分を除去済み
        )
        if (accelerometer != null) {
            sensorManager.registerListener(
                this,
                accelerometer,
                SensorManager.SENSOR_DELAY_UI  // 約60ms間隔
            )
            Log.d(TAG, "加速度センサー開始")
        } else {
            Log.w(TAG, "LINEAR_ACCELERATION センサーが利用不可")
        }
    }

    /** センサーの登録解除 */
    fun stop() {
        sensorManager.unregisterListener(this)
        Log.d(TAG, "加速度センサー停止")
    }

    override fun onSensorChanged(event: SensorEvent) {
        val x = event.values[0]
        val y = event.values[1]
        val z = event.values[2]

        // 3軸の合成加速度を計算
        val magnitude = sqrt(x * x + y * y + z * z)

        if (magnitude > SHAKE_THRESHOLD) {
            val now = System.currentTimeMillis()
            if (now - lastShakeTime > MIN_TIME_BETWEEN_SHAKES) {
                lastShakeTime = now
                Log.d(TAG, "シェイク検知: 加速度 = $magnitude m/s^2")
                onShake()
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
        Log.d(TAG, "精度変更: ${sensor.name} → $accuracy")
    }

    /** 利用可能な全センサーを列挙 */
    fun listAllSensors(): List<String> {
        return sensorManager.getSensorList(Sensor.TYPE_ALL).map {
            "${it.name} (type=${it.type}, vendor=${it.vendor}, " +
            "power=${it.power}mA, resolution=${it.resolution})"
        }
    }
}

// 使用例 (Activity 内):
// val shakeDetector = ShakeDetector(this) {
//     Toast.makeText(this, "シェイクを検知しました", Toast.LENGTH_SHORT).show()
// }
// override fun onResume() { super.onResume(); shakeDetector.start() }
// override fun onPause() { super.onPause(); shakeDetector.stop() }
```

---

## 9. アプリ開発とビルドパイプライン

### 9.1 開発環境の比較

| 項目 | iOS | Android |
|------|-----|---------|
| 公式 IDE | Xcode | Android Studio |
| ビルドシステム | xcodebuild / Swift Package Manager | Gradle (Kotlin DSL) |
| 言語 | Swift (主流), Objective-C (レガシー) | Kotlin (主流), Java (レガシー) |
| UI フレームワーク | SwiftUI (宣言的), UIKit (命令的) | Jetpack Compose (宣言的), View (命令的) |
| テスト | XCTest, XCUITest | JUnit, Espresso, Compose Testing |
| プロファイラ | Instruments | Android Profiler (CPU, Memory, Network) |
| パッケージ管理 | SPM, CocoaPods | Gradle dependencies, Maven Central |
| CI/CD | Xcode Cloud, Fastlane | GitHub Actions, Fastlane |
| 配布 | App Store Connect, TestFlight | Google Play Console, Firebase App Distribution |
| コード署名 | 必須 (Provisioning Profile + Certificate) | 必須 (APK/AAB 署名) |
| 最小ターゲット | 通常 iOS N-2 (現在は iOS 16+) | minSdk 24+ (Android 7.0) が一般的 |

### 9.2 ビルドパイプラインの全体像

```
モバイルアプリ CI/CD パイプライン
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  開発者
    │
    ├── git push
    │
    ▼
  ┌──────────────────────────────────────────────────────┐
  │                  CI サーバ                           │
  │                                                      │
  │  1. ソースコード取得 (git clone)                     │
  │     ▼                                                │
  │  2. 依存関係解決                                     │
  │     iOS:  swift package resolve / pod install        │
  │     Android: ./gradlew dependencies                  │
  │     ▼                                                │
  │  3. 静的解析                                         │
  │     iOS:  SwiftLint, SwiftFormat                     │
  │     Android: ktlint, detekt, Android Lint            │
  │     ▼                                                │
  │  4. 単体テスト                                       │
  │     iOS:  xcodebuild test -scheme MyApp              │
  │     Android: ./gradlew testDebugUnitTest             │
  │     ▼                                                │
  │  5. ビルド                                           │
  │     iOS:  xcodebuild archive → .xcarchive           │
  │     Android: ./gradlew assembleRelease → .apk/.aab  │
  │     ▼                                                │
  │  6. コード署名                                       │
  │     iOS:  Provisioning Profile + Distribution Cert   │
  │     Android: Keystore による APK/AAB 署名            │
  │     ▼                                                │
  │  7. UIテスト / E2Eテスト                             │
  │     iOS:  XCUITest (Simulator / Device Farm)        │
  │     Android: Espresso (Emulator / Firebase Test Lab)│
  │     ▼                                                │
  │  8. 配布                                             │
  │     ┌──────────────────┬──────────────────────┐      │
  │     │ テスト配布        │ 本番配布             │      │
  │     │ TestFlight       │ App Store            │      │
  │     │ Firebase App Dist│ Google Play          │      │
  │     │ DeployGate       │ (段階的公開 10→100%) │      │
  │     └──────────────────┴──────────────────────┘      │
  └──────────────────────────────────────────────────────┘
```

### 9.3 クロスプラットフォーム開発

iOS と Android の両方を単一のコードベースで開発するアプローチも広く採用されている。

| フレームワーク | 言語 | レンダリング方式 | パフォーマンス | 採用企業例 |
|--------------|------|---------------|-------------|-----------|
| **Flutter** | Dart | 独自エンジン (Skia/Impeller) | 高 (ネイティブ同等) | Google, BMW, Alibaba |
| **React Native** | JavaScript/TypeScript | ネイティブUI (Bridge/JSI) | 中〜高 | Meta, Microsoft, Shopify |
| **Kotlin Multiplatform** | Kotlin | ネイティブUI (各プラットフォーム) | 高 (ロジック共有) | Netflix, VMware, Philips |
| **MAUI (.NET)** | C# | ネイティブUI (Handlers) | 中 | Microsoft, UPS |
| **Capacitor/Ionic** | Web (HTML/CSS/JS) | WebView | 低〜中 | Burger King, Sanvello |

### 9.4 コード例: Gradle ビルド設定

**コード例8: Android (Kotlin DSL) ── build.gradle.kts の設定例**

```kotlin
// app/build.gradle.kts
// Android アプリのビルド設定

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.hilt)
    alias(libs.plugins.ksp)
}

android {
    namespace = "com.example.mobileapp"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.mobileapp"
        minSdk = 26          // Android 8.0 以上
        targetSdk = 35       // 最新 API レベル
        versionCode = 1
        versionName = "1.0.0"

        testInstrumentationRunner =
            "androidx.test.runner.AndroidJUnitRunner"

        // BuildConfig にビルド情報を埋め込む
        buildConfigField(
            "String", "BUILD_TIME",
            "\"${java.time.Instant.now()}\""
        )
    }

    // ビルドバリアント: debug / staging / release
    buildTypes {
        debug {
            isDebuggable = true
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-debug"
        }
        create("staging") {
            isDebuggable = false
            applicationIdSuffix = ".staging"
            signingConfig = signingConfigs.getByName("debug")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        release {
            isMinifyEnabled = true    // R8 によるコード縮小
            isShrinkResources = true  // 未使用リソース削除
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    // Jetpack Compose 設定
    buildFeatures {
        compose = true
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    // Jetpack Compose
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.material3)
    implementation(libs.compose.navigation)

    // ライフサイクル
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.lifecycle.runtime.compose)

    // DI (Hilt)
    implementation(libs.hilt.android)
    ksp(libs.hilt.compiler)

    // ネットワーク
    implementation(libs.retrofit)
    implementation(libs.okhttp)

    // テスト
    testImplementation(libs.junit)
    testImplementation(libs.mockk)
    androidTestImplementation(libs.compose.ui.test)
}
```

---

## 10. アンチパターンと設計上の落とし穴

### 10.1 アンチパターン1: メインスレッドでの重い処理

**問題**: UIスレッド (メインスレッド) でネットワーク通信やディスクI/O、重い計算を行うと、UIがフリーズ (Application Not Responding = ANR) する。

```
アンチパターン: メインスレッドブロック
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  時間軸 ────────────────────────────────────►

  メインスレッド:
  ┌─────┐  ┌───────────────────────────┐  ┌─────┐
  │ UI  │  │  ネットワーク通信 (3秒)   │  │ UI  │
  │描画 │  │  ← この間UIがフリーズ →  │  │描画 │
  └─────┘  └───────────────────────────┘  └─────┘
  16ms      ← ANR発生 (Android: 5秒 / iOS: watchdog) →

  正しいパターン: バックグラウンドスレッドに委譲
  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
  │ UI  │  │ UI  │  │ UI  │  │ UI  │  │ UI  │
  │描画 │  │描画 │  │描画 │  │描画 │  │更新 │
  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘
  メインスレッドは常に応答可能

  バックグラウンド:
  ┌───────────────────────────┐
  │  ネットワーク通信 (3秒)   │ → 完了後にメインスレッドに通知
  └───────────────────────────┘
```

**悪い例 (Android / Kotlin)**:
```kotlin
// NG: メインスレッドでネットワーク通信
// Android では NetworkOnMainThreadException が発生する
fun loadDataBad() {
    val url = URL("https://api.example.com/data")
    val data = url.readText()  // メインスレッドをブロック
    textView.text = data
}
```

**良い例 (Android / Kotlin)**:
```kotlin
// OK: コルーチンでバックグラウンド処理
fun loadDataGood() {
    viewModelScope.launch {
        val data = withContext(Dispatchers.IO) {
            // IO ディスパッチャでネットワーク通信
            repository.fetchData()
        }
        // メインスレッドで UI 更新 (自動的に Dispatchers.Main)
        _uiState.value = UiState.Success(data)
    }
}
```

### 10.2 アンチパターン2: メモリリークの放置

**問題**: Activity やContext の参照を長寿命オブジェクトが保持し続けると、GC で回収できずメモリリークが発生する。画面回転のたびに Activity が再生成されるため、短時間でメモリが枯渇する。

```
メモリリークのパターンと対策
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  典型的なメモリリーク:

  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │  static / companion object                      │
  │  ┌───────────────┐                              │
  │  │ リスナー/     │──── 強参照 ───► Activity A   │
  │  │ コールバック  │              (画面回転で破棄  │
  │  │ (長寿命)      │               されるべき)     │
  │  └───────────────┘                              │
  │                                                 │
  │  Activity A は GC 対象にならない → メモリリーク │
  │  画面回転のたびに新しい Activity が生成され      │
  │  古い Activity が蓄積していく                    │
  └─────────────────────────────────────────────────┘

  対策:

  1. WeakReference を使用
     val activityRef = WeakReference(activity)

  2. ライフサイクルに連動してリスナーを解除
     override fun onDestroy() {
         listener.unregister()
         super.onDestroy()
     }

  3. ViewModel + LiveData / StateFlow を使用
     ViewModel は構成変更 (画面回転) で生存するため、
     Activity への直接参照が不要

  4. LeakCanary で検出
     debugImplementation("com.squareup.leakcanary:leakcanary-android:2.x")
     → ビルド時に自動でメモリリークを検出・通知
```

### 10.3 アンチパターン3: 過剰なバックグラウンド処理

**問題**: 不要なバックグラウンドサービスやアラームを多用すると、バッテリーを急速に消耗させる。Android 8.0 以降ではバックグラウンドサービスの起動が制限されており、違反するとシステムが強制停止する。

**対策**: WorkManager を使用して OS が最適なタイミングで実行することを許可する。即時性が不要な処理は `Constraints` を指定して充電中・WiFi 接続時に限定する。

---

## 11. 段階別演習

### 演習1: 初級 ── システム情報の収集と表示

**目標**: iOS または Android で端末のシステム情報を収集し、画面に表示するアプリを作成する。

**要件**:
1. OS バージョン、デバイスモデル、メモリ量、CPU コア数を取得する
2. バッテリー残量と充電状態をリアルタイムで表示する
3. ストレージの使用量と空き容量を円グラフで表示する

**確認項目**:
- [ ] sysctl (iOS) または /proc (Android) から情報を正しく取得できた
- [ ] バッテリー状態の変化をリアルタイムで反映できた
- [ ] メモリ使用量の変化を監視できた

**ヒント**:
- iOS: `ProcessInfo`, `UIDevice`, `FileManager` を使用
- Android: `Build`, `ActivityManager`, `BatteryManager`, `StatFs` を使用

---

### 演習2: 中級 ── バックグラウンドタスクと通知

**目標**: バックグラウンドで定期的にデータを取得し、条件に応じてローカル通知を送信するアプリを作成する。

**要件**:
1. 15分間隔でバックグラウンドタスクを実行する
2. 天気APIからデータを取得し、ローカルDBに保存する
3. 雨の予報があればローカル通知でユーザに知らせる
4. Doze モード / Background App Refresh を考慮した設計にする

**確認項目**:
- [ ] バックグラウンドタスクが正しくスケジュールされた
- [ ] Doze モード復帰後にタスクが実行された (Android)
- [ ] Background App Refresh 無効時の挙動を確認した (iOS)
- [ ] 通知の権限リクエストを適切に実装した
- [ ] バッテリー消費が過大でないことをプロファイラで確認した

**ヒント**:
- iOS: `BGTaskScheduler`, `UNUserNotificationCenter` を使用
- Android: `WorkManager` + `NotificationCompat` を使用

---

### 演習3: 上級 ── セキュアなデータストレージとIPC

**目標**: 生体認証で保護されたセキュアストレージを実装し、別のアプリ/Extension とデータを共有する。

**要件**:
1. 生体認証 (Face ID / 指紋認証) によるロック解除を実装する
2. 機密データを Keychain (iOS) / EncryptedSharedPreferences (Android) に保存する
3. Widget / App Extension とデータを共有する (App Groups / ContentProvider)
4. アプリ改竄検知 (Jailbreak/Root 検知) を実装する
5. SSL Pinning でネットワーク通信を保護する

**確認項目**:
- [ ] 生体認証失敗時のフォールバック (パスコード入力) が動作する
- [ ] Keychain / EncryptedSharedPreferences に正しく保存された
- [ ] Widget / Extension からデータを読み取れた
- [ ] Root / Jailbreak 環境で検知メッセージが表示された
- [ ] Charles Proxy 等の中間者でSSLを傍受できないことを確認した

**ヒント**:
- iOS: `LAContext` (LocalAuthentication), `Keychain Services`, `App Groups`
- Android: `BiometricPrompt`, `EncryptedSharedPreferences`, `ContentProvider`

---

## 12. FAQ ── よくある質問

### Q1: iOS はなぜスワップを使わないのか?

**A**: iOS がスワップ (ディスクへのページアウト) を使用しない理由は主に3つある。

1. **NAND フラッシュの寿命**: スワップによる大量の書き込みは NAND フラッシュの書き換え回数を急速に消費する。モバイル端末のストレージは交換不可能であり、寿命の低下は端末の使用期間を直接短縮する。

2. **レイテンシ**: NAND フラッシュのランダムリード/ライトは DRAM と比較して 100-1000 倍遅い。スワップ発生時のパフォーマンス低下はモバイルの応答性要求と相容れない。

3. **電力消費**: ストレージI/Oは CPU の idle 状態より消費電力が大きく、バッテリー駆動のモバイル端末には不適切である。

代替手段として、iOS はメモリ圧縮 (WKdm/LZ4 アルゴリズム) と jetsam によるプロセス終了を組み合わせている。圧縮により物理メモリの実効容量を約 1.5-2 倍に拡大し、それでも不足する場合は優先度の低いプロセスを終了する。

### Q2: Android のフラグメンテーション問題はどの程度深刻か?

**A**: Android のフラグメンテーション (断片化) とは、市場に存在する端末の OS バージョン、画面サイズ、ハードウェア構成が極めて多様であるという問題を指す。

**主な課題と対策:**

- **OS バージョンの断片化**: 2025年時点で Android 10-15 がそれぞれ 5-20% のシェアを持つ。Google は Jetpack ライブラリで後方互換性を提供し、新機能を古い OS バージョンでも利用可能にしている。

- **画面サイズの多様性**: 4インチのスマートフォンから 13インチのタブレット、折りたたみ端末まで対応が必要。Jetpack Compose の `WindowSizeClass` や Material Design の Adaptive Layout で対応する。

- **セキュリティパッチの遅延**: Project Treble (Android 8.0) と Project Mainline (Android 10) により、Google がセキュリティモジュールを Play Store 経由で直接更新できるようになり、OEMのパッチ適用遅延を部分的に解消した。

- **ハードウェアの多様性**: 数千種類の端末が存在するが、CTS (Compatibility Test Suite) により最低限の互換性は保証される。Firebase Test Lab などの端末ファームでの自動テストが推奨される。

### Q3: モバイルアプリの実行時パーミッション設計のベストプラクティスは?

**A**: 権限リクエストのタイミングと方法はユーザ体験に大きく影響する。以下のベストプラクティスに従うべきである。

1. **必要な時に必要な権限だけ要求する**: アプリ起動直後にすべての権限をまとめて要求してはならない。カメラを使う機能を開いた時にカメラ権限を、地図を表示する時に位置情報権限を要求する。

2. **事前説明を表示する**: OS の権限ダイアログを表示する前に、なぜその権限が必要かをアプリ内で説明する画面を表示する。特に iOS では一度拒否すると再リクエストできないため、事前説明が重要である。

3. **権限なしでも動作する設計にする**: 位置情報が拒否されても手動で住所を入力できる、カメラが拒否されてもギャラリーから選択できる、といったフォールバックを用意する。

4. **最小限の精度で運用する**: 正確な位置情報が不要な場合は概算位置 (approximate location) を使用する。継続的なバックグラウンド位置情報の取得は、ユーザの信頼を損なうためごく限られたユースケースに留める。

### Q4: iOS と Android でアプリのサイズを削減するには?

**A**: アプリサイズの削減は、ダウンロード率と初回起動率に直接影響するため重要である。

| 手法 | iOS | Android |
|------|-----|---------|
| コード縮小 | Swift Compiler 最適化 (-Osize) | R8 / ProGuard (isMinifyEnabled) |
| リソース削減 | Asset Catalog (1x/2x/3x 自動選択) | isShrinkResources + WebP 変換 |
| アプリ分割 | App Thinning (Slicing + ODR) | App Bundle (.aab) + Dynamic Delivery |
| 動的配信 | On-Demand Resources | Dynamic Feature Modules |
| 画像形式 | HEIF, WebP | WebP, AVIF |
| ネイティブライブラリ | arm64 のみ (Universal Binary 不要) | ABI Split (arm64-v8a のみ) |

### Q5: Kotlin Multiplatform (KMP) と Flutter のどちらを選ぶべきか?

**A**: 以下の基準で判断する。

- **KMP が適するケース**: ビジネスロジック (データ処理、API通信、ローカルDB) を共有しつつ、UIは各プラットフォームのネイティブで実装したい場合。既にKotlinの知見があるチームに適する。

- **Flutter が適するケース**: UI も含めて完全に統一したい場合。高いパフォーマンスのカスタムUIが必要で、Web版やデスクトップ版にも展開したい場合。Dart の学習コストを受容できるチームに適する。

---

## 13. まとめと次のステップ

### 13.1 総合比較表

| 項目 | iOS | Android |
|------|-----|---------|
| カーネル | XNU (Mach + BSD ハイブリッド) | Linux (モノリシック + モジュール) |
| IPC | Mach Port, XPC | Binder |
| メモリ管理 | Jetsam + メモリ圧縮 | LMK + zRAM |
| スワップ | なし | zRAM (メモリ内圧縮) |
| GC / メモリ解放 | ARC (コンパイル時参照カウント) | Tracing GC (ART の Concurrent Copying) |
| ファイルシステム | APFS | ext4 / f2fs |
| セキュリティ | Sandbox + KTRR + SEP | SELinux + seccomp + TrustZone |
| 電力管理 | Background App Refresh, App Nap | Doze, App Standby Buckets |
| プッシュ通知 | APNs | FCM |
| ドライバ | IOKit (C++) | HAL/HIDL/AIDL + Linux Driver |
| 更新方式 | 全端末一斉 (Apple 管理) | Project Mainline + OEM 依存 |
| 開発言語 | Swift / Objective-C | Kotlin / Java |
| UI 宣言的 | SwiftUI | Jetpack Compose |
| アプリ配布 | App Store (審査あり) | Play Store + サイドロード |
| ソースコード | 非公開 (Darwin 除く) | AOSP (オープンソース) |

### 13.2 今後のトレンド

1. **オンデバイス AI の統合**: Apple Intelligence, Google Gemini Nano がOS レベルで統合され、テキスト生成、画像認識、音声理解がローカルで完結する方向に進んでいる。NPU (Neural Processing Unit) の性能が急速に向上しており、クラウドに依存しない AI 体験が標準となりつつある。

2. **プライバシー強化の深化**: App Tracking Transparency (iOS), Privacy Sandbox (Android) に続き、広告 ID の廃止、IP アドレスの秘匿化、オンデバイス機械学習による個人化など、プライバシーを保ちながらパーソナライゼーションを実現する技術が進化している。

3. **空間コンピューティング**: Apple Vision Pro (visionOS), Android XR により、モバイル OS の概念が 2D 画面から 3D 空間に拡張されつつある。ARKit / ARCore の技術がヘッドマウントディスプレイ向けに発展し、新しい入力パラダイム (視線追跡、ハンドジェスチャー) が導入されている。

4. **衛星通信の統合**: iOS 14 以降の衛星SOS、Android 15 の衛星メッセージング対応により、セルラー圏外でも基本的な通信機能が利用可能になる。これに伴い、OS レベルでの通信スタック設計が拡張されている。

5. **セキュリティの進化**: MTE (Memory Tagging Extension) のハードウェアサポート拡大、PAC (Pointer Authentication Code) の普及により、メモリ安全性が OS レベルで強化されている。ゼロデイ攻撃の難易度が年々上昇している。

### 13.3 学習ロードマップ

```
モバイルOS 学習ロードマップ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  入門 ─────────────────────────────────
  │ OS の基本概念 (プロセス、メモリ、FS)
  │ Swift / Kotlin の文法
  │ 公式チュートリアルでアプリ作成
  ▼
  中級 ─────────────────────────────────
  │ ライフサイクル管理
  │ バックグラウンド処理と電力管理
  │ セキュリティモデルの理解
  │ パフォーマンスプロファイリング
  ▼
  上級 ─────────────────────────────────
  │ カーネル内部構造の理解
  │ IPC の設計と実装
  │ ドライバ / HAL の仕組み
  │ セキュリティ研究 (脆弱性分析)
  ▼
  専門家 ────────────────────────────────
    OS カスタマイズ (AOSP ビルド)
    カーネルモジュール開発
    リバースエンジニアリング
    セキュリティ監査
```

---

## 次に読むべきガイド

- [[01-cloud-os.md]] -- クラウドとリアルタイムOS

---

## 14. 参考文献

### 書籍

1. Levin, J. *"\*OS Internals, Volume I: User Mode"*. Technologeeks Press, 2017. -- iOS/macOS のユーザ空間アーキテクチャを体系的に解説した決定版。XNU カーネルの上に構築されたフレームワーク群の内部構造を詳述する。

2. Levin, J. *"\*OS Internals, Volume II: Kernel Mode"*. Technologeeks Press, 2019. -- XNU カーネルの内部構造 (Mach, BSD, IOKit) を深く掘り下げた上級者向けの参考書。jetsam, sandbox, コード署名の実装詳細を含む。

3. Levin, J. *"\*OS Internals, Volume III: Security & Insecurity"*. Technologeeks Press, 2020. -- iOS/macOS のセキュリティアーキテクチャを網羅的に解説。Secure Enclave, コード署名チェーン、AMFI, サンドボックスプロファイルの詳細を含む。

4. Yaghmour, K. *"Embedded Android: Porting, Extending, and Customizing"*. O'Reilly Media, 2013. -- AOSP のビルドシステム、HAL の実装、カーネルカスタマイズの実践ガイド。

5. Gargenta, M. and Nakamura, M. *"Learning Android"*. O'Reilly Media, 2014. -- Android アプリ開発の基礎からシステムサービスの利用まで体系的に学べる入門書。

### 公式ドキュメント

6. Apple Inc. *"Apple Platform Security Guide"*. 2024. https://support.apple.com/guide/security/ -- iOS, iPadOS, macOS のセキュリティ設計を Apple が公式に解説した文書。ハードウェアセキュリティ、暗号化、認証の仕組みを網羅する。

7. Android Open Source Project. *"Android Architecture"*. https://source.android.com/docs/core/architecture -- AOSP の公式アーキテクチャドキュメント。HAL, Treble, AIDL の設計と実装ガイドラインを含む。

8. Google. *"Android Developer Documentation"*. https://developer.android.com/docs -- Android アプリ開発の公式リファレンス。API ガイド、ベストプラクティス、コードラボを含む。

9. Apple Inc. *"iOS App Dev Tutorials"*. https://developer.apple.com/tutorials/app-dev-training -- SwiftUI を使った iOS アプリ開発の公式チュートリアル。

### 学術論文・技術文書

10. Felt, A. P., et al. "Android Permissions Demystified". *ACM CCS*, 2011. -- Android の権限モデルの設計と実際の使われ方を分析した先駆的な論文。

11. Enck, W., et al. "TaintDroid: An Information-Flow Tracking System for Realtime Privacy Monitoring on Smartphones". *OSDI*, 2010. -- モバイルアプリにおけるプライバシー情報の流出を追跡するシステムの提案。

12. Singh, A. *"Mac OS X Internals: A Systems Approach"*. Addison-Wesley, 2006. -- XNU カーネルの歴史的背景と設計原理を理解するための古典的参考書。Darwin の BSD/Mach 統合の経緯を詳述する。


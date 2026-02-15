# デスクトップアプリの全体像

> デスクトップアプリ開発は Web 技術の進化により大きく変化した。ネイティブ、ハイブリッド、Web ベースの各アプローチの特徴を理解し、プロジェクトに最適な技術を選定する基準を解説する。

## この章で学ぶこと

- [ ] デスクトップアプリの種類と技術スタックを理解する
- [ ] Web 技術ベースとネイティブの違いを把握する
- [ ] プロジェクト要件に基づく技術選定ができるようになる
- [ ] 各技術のアーキテクチャとプロセスモデルを理解する
- [ ] デスクトップアプリのセキュリティモデルを把握する
- [ ] 実務レベルでの技術比較評価ができるようになる

---

## 1. デスクトップアプリの種類

### 1.1 分類の全体像

```
デスクトップアプリの分類:

  +------------------------------------------+
  |           デスクトップアプリ                 |
  +------------------------------------------+
  |                                          |
  |  ① ネイティブ          ② クロスプラットフォーム |
  |  ├─ WPF / WinUI 3     ├─ .NET MAUI      |
  |  ├─ Win32 / MFC       ├─ Qt             |
  |  ├─ SwiftUI / AppKit  ├─ Flutter Desktop |
  |  └─ GTK               └─ Avalonia UI    |
  |                                          |
  |  ③ Web 技術ベース       ④ PWA            |
  |  ├─ Electron          └─ ブラウザ内       |
  |  ├─ Tauri                インストール     |
  |  ├─ Neutralinojs                         |
  |  └─ Wails                                |
  |                                          |
  +------------------------------------------+
```

### 1.2 ネイティブアプリケーション

OS のネイティブ API を直接使用するアプリケーション。最高のパフォーマンスと OS 統合を実現するが、OS ごとに別実装が必要になる。

```
ネイティブアプリの特徴:

  Windows ネイティブ:
    ┌─────────────────────────────────────────────────┐
    │ WPF (Windows Presentation Foundation)            │
    │ ├─ .NET Framework / .NET 8+ 上で動作             │
    │ ├─ XAML ベースの宣言的 UI                         │
    │ ├─ データバインディング / MVVM パターン             │
    │ ├─ DirectX ベースのレンダリング                    │
    │ └─ 業務アプリのデファクトスタンダード               │
    ├─────────────────────────────────────────────────┤
    │ WinUI 3 (Windows App SDK)                        │
    │ ├─ WPF の後継となるモダン UI フレームワーク         │
    │ ├─ Fluent Design System 完全対応                  │
    │ ├─ Win32 / UWP 両方の API にアクセス可能           │
    │ ├─ MSIX パッケージングによる安全な配布              │
    │ └─ Windows 10 1809+ / Windows 11 対応            │
    ├─────────────────────────────────────────────────┤
    │ Win32 / MFC / ATL                                │
    │ ├─ C/C++ で記述する最もローレベルな API            │
    │ ├─ 最小のメモリフットプリント                      │
    │ ├─ 完全な OS API アクセス                         │
    │ ├─ レガシーシステムとの互換性                      │
    │ └─ 開発生産性は低い                               │
    └─────────────────────────────────────────────────┘

  macOS ネイティブ:
    ┌─────────────────────────────────────────────────┐
    │ SwiftUI                                          │
    │ ├─ Apple の最新 UI フレームワーク                  │
    │ ├─ 宣言的構文、プレビュー機能                     │
    │ └─ macOS 11+ / iOS 15+ 対応                     │
    ├─────────────────────────────────────────────────┤
    │ AppKit (Cocoa)                                   │
    │ ├─ macOS 向けの成熟した UI フレームワーク           │
    │ ├─ Objective-C / Swift で記述                     │
    │ └─ 最も細かい macOS UI カスタマイズが可能           │
    └─────────────────────────────────────────────────┘

  Linux ネイティブ:
    ┌─────────────────────────────────────────────────┐
    │ GTK (GIMP Toolkit)                               │
    │ ├─ GNOME デスクトップ環境の標準                    │
    │ ├─ C / Python / Vala / Rust で記述可能            │
    │ └─ GTK 4 で大幅なパフォーマンス改善               │
    ├─────────────────────────────────────────────────┤
    │ Qt                                               │
    │ ├─ C++ / QML でクロスプラットフォーム対応           │
    │ ├─ KDE デスクトップ環境の標準                      │
    │ └─ 商用ライセンスと LGPL デュアルライセンス         │
    └─────────────────────────────────────────────────┘
```

### 1.3 クロスプラットフォームネイティブ

1つのコードベースで複数の OS に対応するアプローチ。OS 固有の UI レンダリングエンジンを使うため、ネイティブに近いパフォーマンスを実現する。

```
クロスプラットフォームネイティブの比較:

  .NET MAUI:
    言語: C# / XAML
    対応OS: Windows, macOS, iOS, Android
    レンダリング: 各OS のネイティブコントロール
    特徴:
      → Xamarin.Forms の後継
      → .NET エコシステムとの統合
      → Hot Reload 対応
      → Visual Studio での統合開発環境
    適用例:
      → 社内業務アプリのマルチプラットフォーム対応
      → .NET 資産を持つ企業のモバイル展開

  Qt:
    言語: C++ / QML
    対応OS: Windows, macOS, Linux, iOS, Android, 組み込み
    レンダリング: 独自レンダリングエンジン
    特徴:
      → 20年以上の実績
      → 最も幅広い OS サポート
      → 高パフォーマンス
      → シグナル/スロットによるイベントシステム
    適用例:
      → CAD / 3D モデリングソフト
      → 産業用制御システム
      → メディアプレイヤー

  Flutter Desktop:
    言語: Dart
    対応OS: Windows, macOS, Linux, iOS, Android, Web
    レンダリング: Impeller（独自レンダリングエンジン）
    特徴:
      → Google 開発
      → 完全にカスタム描画（OS ネイティブ UI を使わない）
      → Hot Reload で高速開発
      → 統一された UI / UX
    適用例:
      → 統一 UI が重要なマルチプラットフォームアプリ
      → モバイルファーストのデスクトップ拡張

  Avalonia UI:
    言語: C# / AXAML
    対応OS: Windows, macOS, Linux, iOS, Android, Web
    レンダリング: Skia ベースの独自レンダリング
    特徴:
      → WPF ライクな API
      → MVVM パターン対応
      → .NET エコシステム活用
      → Linux でも安定した UI
    適用例:
      → WPF アプリの Linux / macOS 移植
      → .NET ベースのクロスプラットフォームアプリ
```

### 1.4 Web 技術ベースアプリケーション

HTML/CSS/JavaScript で UI を構築するアプリケーション。Web 開発者のスキルをそのまま活用でき、豊富なエコシステムが利用可能。

```
Web 技術ベースの比較:

  Electron:
    アーキテクチャ:
      ┌──────────────────────────────┐
      │  メインプロセス (Node.js)      │
      │  ├─ アプリのライフサイクル管理   │
      │  ├─ ネイティブ API アクセス     │
      │  ├─ ファイルシステム操作        │
      │  └─ IPC 通信                  │
      ├──────────────────────────────┤
      │  レンダラープロセス (Chromium)  │
      │  ├─ HTML/CSS/JS でUI描画      │
      │  ├─ React/Vue/Svelte 等      │
      │  └─ 各ウィンドウに1プロセス    │
      └──────────────────────────────┘
    特徴:
      → Chromium 同梱で一貫したレンダリング
      → Node.js のフルパワー
      → 最大のコミュニティとエコシステム
    課題:
      → バンドルサイズが ~150MB
      → メモリ使用量が多い（~200MB+）
      → Chromium のバージョン管理

  Tauri v2:
    アーキテクチャ:
      ┌──────────────────────────────┐
      │  バックエンド (Rust)           │
      │  ├─ コマンドハンドラ           │
      │  ├─ プラグインシステム         │
      │  ├─ ネイティブ API アクセス     │
      │  └─ セキュリティ制御           │
      ├──────────────────────────────┤
      │  フロントエンド (OS WebView)    │
      │  ├─ HTML/CSS/JS でUI描画      │
      │  ├─ React/Vue/Svelte 等      │
      │  └─ OS の WebView を使用      │
      └──────────────────────────────┘
    特徴:
      → OS の WebView を使用（Chromium 同梱不要）
      → Rust の安全性とパフォーマンス
      → バンドルサイズ ~5MB
      → セキュリティファーストの設計
    課題:
      → OS の WebView バージョン差異
      → Rust の学習コスト
      → Electron より小さいエコシステム

  Neutralinojs:
    アーキテクチャ:
      → 軽量な C++ バックエンド + OS WebView
      → Electron の 1/10 以下のサイズ
      → Node.js 不要
    特徴:
      → シンプルな API
      → 低学習コスト
      → 小規模アプリに最適

  Wails:
    アーキテクチャ:
      → Go バックエンド + OS WebView
      → Go のエコシステムを活用
    特徴:
      → Go 開発者向け
      → 高いパフォーマンス
      → シンプルなビルドプロセス
```

### 1.5 PWA (Progressive Web App)

```
PWA の位置づけ:

  ブラウザベースのインストール可能アプリ:
    ┌──────────────────────────────────────────────┐
    │  Service Worker                               │
    │  ├─ オフラインキャッシュ                        │
    │  ├─ バックグラウンド同期                        │
    │  └─ プッシュ通知                               │
    ├──────────────────────────────────────────────┤
    │  Web App Manifest                             │
    │  ├─ アプリ名、アイコン、テーマカラー              │
    │  ├─ スタンドアロン表示モード                     │
    │  └─ OS 統合（ショートカット等）                  │
    ├──────────────────────────────────────────────┤
    │  利用可能な API                                │
    │  ├─ File System Access API                    │
    │  ├─ Web Bluetooth / Web USB                   │
    │  ├─ Web Share API                             │
    │  ├─ Notifications API                         │
    │  └─ ※ ネイティブ機能はブラウザ対応に依存         │
    └──────────────────────────────────────────────┘

  PWA の利点:
    → 配布が最も容易（URL だけで利用開始）
    → 自動更新（Service Worker の仕組み）
    → 最小のストレージ消費
    → インストール不要でも利用可能

  PWA の制限:
    → ネイティブ API アクセスが限定的
    → ファイルシステムアクセスに制限
    → ブラウザエンジンに依存
    → OS 統合が不完全（特に macOS / Linux）
```

---

## 2. 技術スタックの詳細比較

### 2.1 基本スペック比較

```
主要技術の比較:

  技術       │ 言語      │ サイズ  │ メモリ │ OS対応         │ 用途
  ──────────┼──────────┼───────┼──────┼──────────────┼──────
  Electron  │ JS/TS    │ ~150MB│ 200MB│ Win/Mac/Linux│ 汎用
  Tauri v2  │ Rust+JS  │ ~5MB  │ 50MB │ Win/Mac/Linux│ 軽量
  WPF       │ C#/XAML  │ ~20MB │ 100MB│ Windows のみ  │ 業務
  WinUI 3   │ C#/XAML  │ ~20MB │ 100MB│ Windows のみ  │ モダンUI
  MAUI      │ C#/XAML  │ ~30MB │ 120MB│ Win/Mac/and  │ クロス
  Flutter   │ Dart     │ ~20MB │ 80MB │ Win/Mac/Linux│ クロス
  Qt        │ C++      │ ~30MB │ 60MB │ Win/Mac/Linux│ 高性能
  Avalonia  │ C#/AXAML │ ~25MB │ 90MB │ Win/Mac/Linux│ クロス
  Wails     │ Go+JS    │ ~8MB  │ 50MB │ Win/Mac/Linux│ 軽量

有名アプリの技術スタック:

  Electron:
    → VS Code, Slack, Discord, Notion, Figma Desktop
    → Spotify Desktop, GitHub Desktop, Postman
    → 1Password (v8), Microsoft Teams, Obsidian
    → Signal Desktop, Bitwarden Desktop

  Tauri:
    → Cody (Sourcegraph), Crabnebula
    → 新規プロジェクトでの採用が増加中
    → DevTools 系ツールでの採用が増加

  WPF/WinUI:
    → Visual Studio, Windows Terminal
    → Windows 標準アプリ群
    → Paint.NET, LINQPad

  Qt:
    → VirtualBox, OBS Studio
    → Adobe Substance, Autodesk Maya
    → VLC Media Player, KDE Plasma

  Flutter Desktop:
    → Google Earth, Superlist
    → Ente (写真管理), AppFlowy
```

### 2.2 パフォーマンス詳細比較

```
起動時間の比較（典型的な中規模アプリ）:

  技術          │ コールドスタート │ ウォームスタート │ 初回描画
  ─────────────┼──────────────┼──────────────┼─────────
  Win32/MFC    │    100ms     │     30ms     │   50ms
  WPF          │    500ms     │    200ms     │  300ms
  WinUI 3      │    600ms     │    250ms     │  350ms
  Qt           │    300ms     │    100ms     │  150ms
  Flutter      │    400ms     │    150ms     │  200ms
  Tauri v2     │    800ms     │    300ms     │  500ms
  Electron     │   1500ms     │    500ms     │  800ms

  ※ コールドスタート = OS 起動後の最初の起動
  ※ ウォームスタート = 2回目以降の起動（キャッシュあり）

メモリ使用量の比較（空のウィンドウ表示時）:

  技術          │ 初期メモリ │ 安定時メモリ │ プロセス数
  ─────────────┼─────────┼──────────┼─────────
  Win32/MFC    │   5MB   │   8MB    │    1
  WPF          │  30MB   │  50MB    │    1
  WinUI 3      │  35MB   │  55MB    │    1
  Qt           │  20MB   │  35MB    │    1
  Flutter      │  25MB   │  45MB    │    1
  Tauri v2     │  15MB   │  30MB    │    2
  Electron     │  80MB   │ 120MB    │    3+

CPU 使用率の比較（アイドル時）:

  技術          │ アイドル CPU │ アニメーション時
  ─────────────┼────────────┼──────────────
  Win32/MFC    │    0.0%    │     2-5%
  WPF          │    0.1%    │     3-8%
  WinUI 3      │    0.1%    │     2-6%
  Qt           │    0.0%    │     2-5%
  Flutter      │    0.1%    │     3-7%
  Tauri v2     │    0.2%    │     5-10%
  Electron     │    0.5%    │     8-15%
```

### 2.3 開発生産性の比較

```
開発生産性の評価:

  技術      │ 学習曲線 │ Hot Reload │ デバッグ │ テスト │ CI/CD
  ─────────┼────────┼──────────┼───────┼──────┼──────
  Electron │  低    │    ○     │  優秀  │  豊富 │ 容易
  Tauri v2 │  中    │    ○     │  良好  │  豊富 │ 容易
  WPF      │  中    │    △     │  優秀  │  良好 │ 中程度
  WinUI 3  │  中    │    △     │  良好  │  良好 │ 中程度
  MAUI     │  中    │    ○     │  良好  │  良好 │ 複雑
  Flutter  │  中    │    ◎     │  優秀  │  豊富 │ 容易
  Qt       │  高    │    △     │  良好  │  良好 │ 複雑

  ※ ◎=非常に優れている ○=良い △=制限あり

開発者エコシステムの規模:

  技術      │ npm/パッケージ │ Stack Overflow │ GitHub Stars
  ─────────┼─────────────┼──────────────┼────────────
  Electron │  npm 全体    │    50,000+   │   115,000+
  Tauri v2 │  npm + crates│    10,000+   │    85,000+
  WPF      │  NuGet      │    80,000+   │    N/A
  WinUI 3  │  NuGet      │     5,000+   │     4,000+
  Flutter  │  pub.dev    │    60,000+   │   165,000+
  Qt       │  独自        │    70,000+   │    N/A
```

---

## 3. 技術選定ガイド

### 3.1 選定フローチャート

```
選定フローチャート:

  プロジェクト要件の確認
  │
  ├─ Q1: ターゲット OS は？
  │   ├─ Windows のみ → Q2a
  │   ├─ Windows + macOS → Q3
  │   └─ 全 OS 対応 → Q3
  │
  ├─ Q2a: Windows ネイティブの外観が必要？
  │   ├─ はい → Q2b
  │   └─ いいえ → Q3
  │
  ├─ Q2b: .NET / C# の既存資産がある？
  │   ├─ はい → WinUI 3 または WPF
  │   └─ いいえ → Q3
  │
  ├─ Q3: Web 技術チームのスキル？
  │   ├─ Web 技術メイン → Q4
  │   ├─ .NET / C# メイン → MAUI / Avalonia
  │   ├─ C++ メイン → Qt
  │   └─ Dart / Flutter 経験 → Flutter Desktop
  │
  ├─ Q4: バンドルサイズとメモリの制約は？
  │   ├─ 厳しい（10MB以下） → Tauri v2
  │   ├─ 中程度（50MB以下） → Tauri v2
  │   └─ 制約なし → Q5
  │
  └─ Q5: Node.js エコシステムへの依存度は？
      ├─ 高い（既存 npm パッケージ多数） → Electron
      ├─ 中程度 → Tauri v2 または Electron
      └─ 低い → Tauri v2
```

### 3.2 ユースケース別の推奨技術

```
ユースケース別推奨:

  ① 社内業務アプリ（Windows 限定）:
     推奨: WinUI 3 / WPF
     理由:
       → Active Directory / LDAP 統合が容易
       → .NET の業務ライブラリが豊富
       → MSIX パッケージで社内配布管理
       → グループポリシーでの制御が可能
     代替: Electron + electron-edge-js

  ② 既存 Web アプリのデスクトップ化:
     推奨: Electron
     理由:
       → 既存コードの最大限の再利用
       → Node.js バックエンドとの統合
       → 実績のあるフレームワーク
     代替: Tauri v2（新規部分が多い場合）

  ③ 開発者向けツール:
     推奨: Electron または Tauri v2
     理由:
       → VS Code 拡張のようなエコシステム
       → カスタマイズ性の高い UI
       → コマンドパレット / ターミナル統合
     実例: VS Code, Postman, Insomnia

  ④ メディア / クリエイティブツール:
     推奨: Qt または WinUI 3
     理由:
       → GPU レンダリングが容易
       → カスタム描画パフォーマンス
       → リアルタイム処理
     実例: OBS Studio, VLC, Blender

  ⑤ IoT / エッジデバイス向け:
     推奨: Qt または Tauri v2
     理由:
       → 低リソース消費
       → クロスプラットフォーム
       → 組み込みLinux 対応
     代替: Flutter（タッチ UI が中心の場合）

  ⑥ エンタープライズ マルチプラットフォーム:
     推奨: Electron または Flutter
     理由:
       → 統一された UI / UX
       → 大規模チーム開発に適した構造
       → CI/CD パイプラインの豊富なサポート
     代替: .NET MAUI（.NET エコシステムの場合）
```

### 3.3 非機能要件による選定

```
非機能要件マトリクス:

  要件           │ 最適な技術          │ 理由
  ──────────────┼──────────────────┼────────────────────
  起動速度       │ Win32, Qt          │ ネイティブ描画
  メモリ効率     │ Tauri, Win32       │ 軽量ランタイム
  バンドルサイズ  │ Tauri, PWA         │ WebView 再利用 / Web ベース
  セキュリティ    │ Tauri, WinUI 3     │ サンドボックス / MSIX
  アクセシビリティ│ WPF, WinUI 3       │ UI Automation 完全対応
  オフライン動作  │ 全ネイティブ技術     │ ローカル実行
  自動更新       │ Electron, Tauri    │ 組み込みアップデーター
  マルチモニター  │ WPF, Electron      │ 高度なウィンドウ管理
  タッチ対応     │ WinUI 3, Flutter   │ タッチ最適化済み
  GPU 活用      │ Qt, Flutter, WPF   │ ハードウェアアクセラレーション

パフォーマンス要件による選定:
  高パフォーマンス必須: Qt > Win32 > WPF > Tauri > Electron
  開発速度重視:        Electron > Flutter > Tauri > WinUI 3 > Qt
  メモリ効率重視:      Win32 > Tauri > Qt > Flutter > Electron
  セキュリティ重視:    Tauri > WinUI 3 > Electron > Qt > Win32
```

---

## 4. Web 技術ベースの利点と課題

### 4.1 Web 技術ベースの利点

```
Web 技術ベースの利点（詳細）:

  ① 開発者プールの広さ:
     → Web 開発者は世界で最も多い技術者層
     → JavaScript/TypeScript の熟練者が多い
     → 採用が容易でチーム構築が速い
     → フロントエンドとデスクトップの両方に対応できる人材

  ② エコシステムの豊富さ:
     → npm レジストリの 200万以上のパッケージ
     → React/Vue/Svelte/Angular 等の成熟した UI フレームワーク
     → UI コンポーネントライブラリ（shadcn/ui, Radix, MUI 等）
     → テストツール（Vitest, Playwright, Cypress）
     → ビルドツール（Vite, webpack, esbuild）

  ③ 高速な開発サイクル:
     → ホットリロード / HMR で即座にプレビュー
     → ブラウザの DevTools でデバッグ
     → CSS でピクセルパーフェクトな UI 制御
     → 宣言的 UI の生産性

  ④ コード共有:
     → Web 版とデスクトップ版でコード共有
     → 共通のビジネスロジック
     → 共通の UI コンポーネント
     → モノレポで管理可能
```

### 4.2 Web 技術ベースの課題と対策

```
課題と対策:

  ① メモリ消費:
     課題:
       → Electron: Chromium 同梱で最低 ~80MB
       → 複数ウィンドウで各プロセスにメモリ割当
       → メモリリークが発生しやすい

     対策:
       → Tauri に移行して OS WebView を使用
       → Electron の場合: 不要なウィンドウの適切な破棄
       → メモリプロファイリングの定期実施
       → Web Worker でメモリ集約処理を分離

     実装例（Electron のメモリ最適化）:
```

```javascript
// Electron: BrowserWindow のメモリ最適化
const { BrowserWindow } = require('electron');

function createOptimizedWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      // 不要な機能を無効化してメモリ節約
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      // 背景スロットリングを有効化
      backgroundThrottling: true,
      // スペルチェックを無効化（不要な場合）
      spellcheck: false,
    },
    // GPU アクセラレーションの制御
    // 必要に応じて無効化
    show: false, // ready-to-show で表示してフリッカー防止
  });

  // ウィンドウが非表示の間はレンダリングを抑制
  win.on('hide', () => {
    win.webContents.setBackgroundThrottling(true);
  });

  // 準備完了後に表示
  win.once('ready-to-show', () => {
    win.show();
  });

  // ウィンドウ破棄時のクリーンアップ
  win.on('closed', () => {
    // 参照をクリアしてGC対象にする
    // win = null; は呼び出し元で実行
  });

  return win;
}
```

```
  ② 起動時間:
     課題:
       → Chromium の初期化に時間がかかる
       → JavaScript のパース・コンパイル時間
       → 大量の npm 依存関係の読み込み

     対策:
       → スプラッシュスクリーンで体感速度を改善
       → コードスプリッティングで初回読み込みを最小化
       → V8 スナップショットでスタートアップを高速化
       → 遅延インポートで必要時にモジュールをロード

  ③ OS API アクセスの制限:
     課題:
       → ネイティブ機能へのアクセスに制限
       → OS 固有の UI コンポーネントが使えない
       → システムトレイ等の統合が不完全な場合がある

     対策:
       → Electron: Native Node Modules で C++ アドオン
       → Tauri: Rust プラグインで任意のネイティブ API
       → FFI (Foreign Function Interface) の活用
       → WebView + ネイティブのハイブリッド構成
```

### 4.3 セキュリティモデルの比較

```
セキュリティモデル:

  Electron のセキュリティ:
    ┌──────────────────────────────────────┐
    │  メインプロセス                        │
    │  ├─ フル Node.js アクセス              │
    │  ├─ ファイルシステム操作               │
    │  └─ OS API アクセス                   │
    ├──────────────────────────────────────┤
    │  preload スクリプト（ブリッジ）         │
    │  ├─ contextBridge で安全な API 公開    │
    │  └─ IPC 通信の仲介                    │
    ├──────────────────────────────────────┤
    │  レンダラープロセス（サンドボックス）    │
    │  ├─ contextIsolation: true            │
    │  ├─ nodeIntegration: false            │
    │  └─ sandbox: true                     │
    └──────────────────────────────────────┘

    注意点:
      → nodeIntegration: true は危険（XSS でフル OS アクセス）
      → remote モジュールは非推奨（セキュリティリスク）
      → CSP (Content Security Policy) の設定が重要

  Tauri のセキュリティ:
    ┌──────────────────────────────────────┐
    │  Rust バックエンド                     │
    │  ├─ コマンドの明示的な許可リスト        │
    │  ├─ ファイルシステムのスコープ制限       │
    │  └─ API のきめ細かい権限制御            │
    ├──────────────────────────────────────┤
    │  OS WebView（サンドボックス）           │
    │  ├─ OS レベルのセキュリティ             │
    │  ├─ プロセス分離                       │
    │  └─ Node.js なし（攻撃面が小さい）      │
    └──────────────────────────────────────┘

    利点:
      → デフォルトで安全（許可リスト方式）
      → Rust のメモリ安全性
      → Node.js を含まないため攻撃面が小さい
      → CSP がデフォルトで厳格
```

セキュリティ設定の実装例:

```javascript
// Electron: セキュアな BrowserWindow 設定
const { BrowserWindow, session } = require('electron');

function createSecureWindow() {
  const win = new BrowserWindow({
    webPreferences: {
      // セキュリティ必須設定
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      // リモートコンテンツの制限
      webSecurity: true,
      allowRunningInsecureContent: false,
      // preload スクリプトのみを許可
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  // CSP の設定
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [
          "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';"
        ],
      },
    });
  });

  // 外部リンクのナビゲーション制限
  win.webContents.on('will-navigate', (event, url) => {
    const parsedUrl = new URL(url);
    if (parsedUrl.origin !== 'http://localhost:3000') {
      event.preventDefault();
    }
  });

  // 新しいウィンドウの作成を制限
  win.webContents.setWindowOpenHandler(({ url }) => {
    // 外部URLはデフォルトブラウザで開く
    shell.openExternal(url);
    return { action: 'deny' };
  });

  return win;
}
```

```json
// Tauri: tauri.conf.json のセキュリティ設定
{
  "app": {
    "security": {
      "csp": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';",
      "dangerousDisableAssetCspModification": false,
      "freezePrototype": true
    }
  },
  "plugins": {
    "fs": {
      "scope": {
        "allow": ["$APPDATA/**", "$DOWNLOAD/**"],
        "deny": ["$HOME/.ssh/**", "$HOME/.gnupg/**"]
      }
    },
    "shell": {
      "scope": {
        "allow": [
          { "name": "open-url", "cmd": "open", "args": true }
        ]
      }
    }
  }
}
```

---

## 5. プロセスモデルとアーキテクチャ

### 5.1 プロセスモデルの詳細

```
各技術のプロセスモデル:

  Electron:
    ┌─────────────┐   IPC    ┌─────────────────┐
    │ Main Process│◄────────►│ Renderer Process │ (Window 1)
    │ (Node.js)   │          └─────────────────┘
    │             │   IPC    ┌─────────────────┐
    │ ・ app      │◄────────►│ Renderer Process │ (Window 2)
    │ ・ ipcMain  │          └─────────────────┘
    │ ・ dialog   │   IPC    ┌─────────────────┐
    │ ・ Menu     │◄────────►│ Utility Process  │ (重い処理)
    └─────────────┘          └─────────────────┘

    特徴:
      → マルチプロセスで安定性確保
      → 1ウィンドウ = 1レンダラープロセス
      → Utility Process で重い処理を分離可能
      → SharedArrayBuffer でプロセス間メモリ共有

  Tauri:
    ┌─────────────┐  invoke  ┌─────────────────┐
    │ Rust Core   │◄────────►│ WebView Process  │
    │             │          │ (OS WebView)      │
    │ ・ commands │  events  │                   │
    │ ・ plugins  │◄────────►│ ・ HTML/CSS/JS    │
    │ ・ state    │          │ ・ React/Vue/etc  │
    └─────────────┘          └─────────────────┘

    特徴:
      → 2プロセスモデル（Core + WebView）
      → invoke/events で双方向通信
      → Rust 側で状態管理
      → プラグインシステムで機能拡張

  WPF / WinUI 3:
    ┌──────────────────────────────────┐
    │ 単一プロセス                       │
    │ ├─ UI スレッド（メインスレッド）     │
    │ │   ├─ XAML レンダリング           │
    │ │   ├─ イベントハンドリング         │
    │ │   └─ データバインディング更新      │
    │ ├─ コンポジションスレッド           │
    │ │   └─ DirectX レンダリング        │
    │ └─ バックグラウンドスレッド          │
    │     └─ Task.Run / ThreadPool      │
    └──────────────────────────────────┘

    特徴:
      → 単一プロセスで効率的
      → UI スレッドとバックグラウンドの分離
      → Dispatcher でスレッド間通信
      → async/await で非同期処理
```

### 5.2 IPC (Inter-Process Communication) パターン

```typescript
// Electron IPC パターン

// --- preload.js ---
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // レンダラー → メインへのリクエスト
  openFile: () => ipcRenderer.invoke('dialog:openFile'),
  saveFile: (data: string) => ipcRenderer.invoke('file:save', data),

  // メイン → レンダラーへの通知を購読
  onUpdateAvailable: (callback: Function) => {
    ipcRenderer.on('update-available', (_event, info) => callback(info));
  },

  // 双方向ストリーミング
  onProgressUpdate: (callback: Function) => {
    ipcRenderer.on('progress', (_event, value) => callback(value));
  },
});

// --- main.js ---
const { ipcMain, dialog } = require('electron');

// invoke ハンドラ（Promise ベース）
ipcMain.handle('dialog:openFile', async (event) => {
  const result = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [
      { name: 'テキスト', extensions: ['txt', 'md'] },
      { name: '全ファイル', extensions: ['*'] },
    ],
  });

  if (!result.canceled) {
    const content = await fs.readFile(result.filePaths[0], 'utf-8');
    return { path: result.filePaths[0], content };
  }
  return null;
});

ipcMain.handle('file:save', async (event, data) => {
  const result = await dialog.showSaveDialog({
    defaultPath: 'untitled.txt',
  });

  if (!result.canceled) {
    await fs.writeFile(result.filePath, data, 'utf-8');
    return { success: true, path: result.filePath };
  }
  return { success: false };
});

// --- renderer.js ---
// preload で公開された API を使用
async function handleOpenFile() {
  const result = await window.electronAPI.openFile();
  if (result) {
    editor.setValue(result.content);
    setCurrentPath(result.path);
  }
}
```

```rust
// Tauri IPC パターン

// --- src-tauri/src/main.rs ---
use tauri::Manager;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct FileResult {
    path: String,
    content: String,
}

// コマンドハンドラ
#[tauri::command]
async fn open_file(app: tauri::AppHandle) -> Result<Option<FileResult>, String> {
    use tauri_plugin_dialog::DialogExt;

    let file_path = app.dialog()
        .file()
        .add_filter("テキスト", &["txt", "md"])
        .blocking_pick_file();

    match file_path {
        Some(path) => {
            let content = std::fs::read_to_string(&path.path)
                .map_err(|e| e.to_string())?;
            Ok(Some(FileResult {
                path: path.path.to_string_lossy().to_string(),
                content,
            }))
        }
        None => Ok(None),
    }
}

#[tauri::command]
async fn save_file(path: String, content: String) -> Result<bool, String> {
    std::fs::write(&path, &content).map_err(|e| e.to_string())?;
    Ok(true)
}

// イベント発行（バックエンド → フロントエンド）
fn emit_progress(app: &tauri::AppHandle, progress: f64) {
    app.emit("progress", progress).unwrap();
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![open_file, save_file])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

```typescript
// Tauri フロントエンド側
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

// コマンド呼び出し
async function handleOpenFile() {
  const result = await invoke<{ path: string; content: string } | null>('open_file');
  if (result) {
    editor.setValue(result.content);
    setCurrentPath(result.path);
  }
}

// イベントリスナー
const unlisten = await listen<number>('progress', (event) => {
  updateProgressBar(event.payload);
});

// クリーンアップ
onDestroy(() => {
  unlisten();
});
```

---

## 6. 開発環境のセットアップ

### 6.1 共通の開発ツール

```
共通の開発環境:

  エディタ / IDE:
    ┌─────────────────────────────────────────────┐
    │ VS Code（推奨）                               │
    │ ├─ 拡張: ESLint, Prettier, TypeScript         │
    │ ├─ 拡張: Tauri (tauri-apps.tauri-vscode)      │
    │ ├─ 拡張: rust-analyzer（Tauri 開発時）          │
    │ ├─ 拡張: C# Dev Kit（WPF/WinUI 開発時）        │
    │ └─ 拡張: XAML Styler                          │
    ├─────────────────────────────────────────────┤
    │ Visual Studio 2022（WPF/WinUI 開発時）         │
    │ ├─ .NET Desktop 開発ワークロード               │
    │ ├─ Windows App SDK                            │
    │ ├─ XAML デザイナー                             │
    │ └─ Hot Reload 対応                            │
    ├─────────────────────────────────────────────┤
    │ JetBrains Rider（.NET 全般）                   │
    │ └─ WPF / MAUI / Avalonia 対応                 │
    └─────────────────────────────────────────────┘

  パッケージマネージャー:
    → pnpm（推奨: 高速・ディスク効率）
    → npm / yarn（代替）
    → NuGet（.NET プロジェクト）
    → Cargo（Rust / Tauri）

  バージョン管理:
    → Git + GitHub / GitLab / Azure DevOps
    → 大きなバイナリ: Git LFS

  CI/CD:
    → GitHub Actions（最も一般的）
    → Azure Pipelines（Windows 特化）
    → GitLab CI/CD
```

### 6.2 Electron 開発環境

```bash
# Electron 開発環境のセットアップ

# 前提条件
node --version  # v20.0.0 以上

# プロジェクト作成（electron-forge 推奨）
npm init electron-app@latest my-electron-app -- \
  --template=vite-typescript

cd my-electron-app

# 基本的な依存関係
npm install electron-store     # データ永続化
npm install electron-updater   # 自動更新
npm install electron-log       # ログ管理

# 開発用依存関係
npm install -D @electron/rebuild  # ネイティブモジュール再ビルド
npm install -D electron-devtools-installer  # DevTools

# プロジェクト構造
# my-electron-app/
# ├── src/
# │   ├── main/           # メインプロセス
# │   │   ├── index.ts    # エントリポイント
# │   │   └── preload.ts  # preload スクリプト
# │   ├── renderer/       # レンダラープロセス
# │   │   ├── index.html
# │   │   ├── App.tsx     # React コンポーネント
# │   │   └── main.tsx    # レンダラーエントリ
# │   └── shared/         # 共有型定義
# │       └── types.ts
# ├── forge.config.ts     # Electron Forge 設定
# ├── vite.main.config.ts
# ├── vite.renderer.config.ts
# ├── tsconfig.json
# └── package.json

# 開発サーバー起動
npm start

# ビルド
npm run make
```

### 6.3 Tauri 開発環境

```bash
# Tauri v2 開発環境のセットアップ

# 前提条件
# Rust のインストール
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustc --version  # 1.77.0 以上

# Node.js
node --version  # v20.0.0 以上

# OS 別の追加依存関係
# Windows:
#   → Visual Studio Build Tools 2022
#   → WebView2 Runtime（Windows 10+ にはプリインストール）
#
# macOS:
#   → Xcode Command Line Tools: xcode-select --install
#
# Linux (Ubuntu/Debian):
#   sudo apt install libwebkit2gtk-4.1-dev build-essential \
#     curl wget file libxdo-dev libssl-dev \
#     libayatana-appindicator3-dev librsvg2-dev

# プロジェクト作成
npm create tauri-app@latest my-tauri-app
cd my-tauri-app

# フロントエンドフレームワーク選択時の推奨:
#   → React + TypeScript + Vite
#   → SvelteKit
#   → Vue + Vite

# プロジェクト構造
# my-tauri-app/
# ├── src/                    # フロントエンド
# │   ├── App.tsx
# │   ├── main.tsx
# │   └── styles.css
# ├── src-tauri/              # Rust バックエンド
# │   ├── src/
# │   │   ├── main.rs         # エントリポイント
# │   │   ├── commands.rs     # コマンドハンドラ
# │   │   └── lib.rs
# │   ├── Cargo.toml          # Rust 依存関係
# │   ├── tauri.conf.json     # Tauri 設定
# │   ├── capabilities/       # 権限設定
# │   └── icons/              # アプリアイコン
# ├── package.json
# ├── tsconfig.json
# └── vite.config.ts

# 開発サーバー起動
npm run tauri dev

# ビルド
npm run tauri build
```

### 6.4 WinUI 3 開発環境

```powershell
# WinUI 3 開発環境のセットアップ（Windows のみ）

# 前提条件
# Visual Studio 2022 のインストール
# ワークロード: .NET デスクトップ開発 + Windows アプリ開発
# 個別コンポーネント: Windows App SDK

# .NET SDK
dotnet --version  # 8.0 以上

# プロジェクト作成
dotnet new winui3 -n MyWinUIApp
cd MyWinUIApp

# NuGet パッケージの追加
dotnet add package CommunityToolkit.Mvvm        # MVVM ツールキット
dotnet add package CommunityToolkit.WinUI       # UI コンポーネント
dotnet add package Microsoft.Extensions.DependencyInjection  # DI

# プロジェクト構造
# MyWinUIApp/
# ├── App.xaml             # アプリケーション定義
# ├── App.xaml.cs          # アプリケーション起動ロジック
# ├── MainWindow.xaml      # メインウィンドウ
# ├── MainWindow.xaml.cs   # コードビハインド
# ├── ViewModels/          # ViewModel 層
# │   └── MainViewModel.cs
# ├── Views/               # View 層
# │   └── SettingsPage.xaml
# ├── Models/              # Model 層
# │   └── AppSettings.cs
# ├── Services/            # サービス層
# │   └── NavigationService.cs
# ├── Helpers/             # ユーティリティ
# ├── Assets/              # 画像・アイコン
# ├── Strings/             # ローカライゼーション
# ├── Package.appxmanifest # パッケージマニフェスト
# └── MyWinUIApp.csproj    # プロジェクトファイル

# ビルド・実行
dotnet build
dotnet run
```

---

## 7. デスクトップアプリのライフサイクル

### 7.1 アプリケーションライフサイクル

```
デスクトップアプリのライフサイクル:

  ┌──────────────────────────────────────────────────────┐
  │                    起動フェーズ                        │
  ├──────────────────────────────────────────────────────┤
  │ 1. プロセス起動                                       │
  │ 2. ランタイム初期化（.NET CLR / V8 / Rust Runtime）    │
  │ 3. アプリケーション設定の読み込み                       │
  │ 4. DI コンテナのセットアップ                            │
  │ 5. メインウィンドウの作成                               │
  │ 6. UI の初期レンダリング                               │
  │ 7. バックグラウンドサービスの起動                       │
  ├──────────────────────────────────────────────────────┤
  │                    実行フェーズ                        │
  ├──────────────────────────────────────────────────────┤
  │ ・ イベントループ（メッセージポンプ）                    │
  │ ・ ユーザー入力の処理                                  │
  │ ・ UI の更新                                          │
  │ ・ バックグラウンド処理                                │
  │ ・ ファイル I/O / ネットワーク通信                      │
  │ ・ 状態の永続化                                       │
  ├──────────────────────────────────────────────────────┤
  │                   終了フェーズ                         │
  ├──────────────────────────────────────────────────────┤
  │ 1. 終了リクエスト（ユーザー操作 / OS シャットダウン）    │
  │ 2. 未保存データの確認ダイアログ                         │
  │ 3. バックグラウンドサービスの停止                       │
  │ 4. リソースの解放                                     │
  │ 5. 設定の保存                                        │
  │ 6. プロセス終了                                      │
  └──────────────────────────────────────────────────────┘
```

### 7.2 各技術でのライフサイクル管理

```typescript
// Electron のライフサイクル管理
import { app, BrowserWindow } from 'electron';

// 単一インスタンスの保証
const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  app.quit();
} else {
  app.on('second-instance', (event, commandLine, workingDirectory) => {
    // 2つ目のインスタンスが起動された場合、最初のウィンドウにフォーカス
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });
}

// アプリの準備完了
app.whenReady().then(async () => {
  // 設定の読み込み
  await loadSettings();

  // メインウィンドウ作成
  createMainWindow();

  // 自動更新チェック
  checkForUpdates();

  // macOS: ドックアイコンクリックでウィンドウ再作成
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createMainWindow();
    }
  });
});

// すべてのウィンドウが閉じられた時
app.on('window-all-closed', () => {
  // macOS 以外はアプリを終了
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// 終了前の処理
app.on('before-quit', async (event) => {
  event.preventDefault();

  // 未保存データの確認
  const hasUnsaved = await checkUnsavedChanges();
  if (hasUnsaved) {
    const result = await dialog.showMessageBox({
      type: 'question',
      buttons: ['保存して終了', '保存せず終了', 'キャンセル'],
      message: '未保存の変更があります。',
    });

    if (result.response === 2) return; // キャンセル
    if (result.response === 0) await saveAll(); // 保存
  }

  // クリーンアップ
  await cleanup();
  app.exit(0);
});

// クラッシュレポート
app.on('render-process-gone', (event, webContents, details) => {
  console.error('Renderer process gone:', details.reason);
  // クラッシュレポートの送信
  reportCrash(details);
});
```

```csharp
// WinUI 3 のライフサイクル管理
// App.xaml.cs
using Microsoft.UI.Xaml;
using Microsoft.Windows.AppLifecycle;
using Windows.ApplicationModel.Activation;

public partial class App : Application
{
    private Window? _mainWindow;

    public App()
    {
        InitializeComponent();

        // 未処理の例外ハンドラ
        UnhandledException += OnUnhandledException;
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        // 単一インスタンスの保証
        var mainInstance = AppInstance.FindOrRegisterForKey("main");
        if (!mainInstance.IsCurrent)
        {
            // 既存インスタンスにリダイレクト
            var activatedArgs = AppInstance.GetCurrent().GetActivatedEventArgs();
            mainInstance.RedirectActivationToAsync(activatedArgs).AsTask().Wait();
            System.Diagnostics.Process.GetCurrentProcess().Kill();
            return;
        }

        // アクティベーション処理の登録
        mainInstance.Activated += OnActivated;

        // DI コンテナの構築
        ConfigureServices();

        // メインウィンドウの作成
        _mainWindow = new MainWindow();
        _mainWindow.Activate();

        // ウィンドウ閉じる時の処理
        _mainWindow.Closed += async (sender, e) =>
        {
            // 未保存データの確認
            if (await HasUnsavedChangesAsync())
            {
                e.Handled = true; // 閉じるのをキャンセル
                await PromptSaveAsync();
            }
        };
    }

    private void OnActivated(object? sender, AppActivationArguments args)
    {
        // アプリがアクティベートされた時（ファイルの関連付け等）
        _mainWindow?.DispatcherQueue.TryEnqueue(() =>
        {
            (_mainWindow as MainWindow)?.BringToFront();
        });
    }

    private void OnUnhandledException(object sender,
        Microsoft.UI.Xaml.UnhandledExceptionEventArgs e)
    {
        // クラッシュレポート
        LogError(e.Exception);
        e.Handled = true;
    }
}
```

---

## 8. データ永続化パターン

### 8.1 デスクトップアプリのデータ保存

```
データ永続化の選択肢:

  ① ファイルベース:
     ├─ JSON / YAML: 設定ファイル、小規模データ
     ├─ SQLite: 構造化データ、クエリが必要な場合
     ├─ LevelDB / RocksDB: キーバリューストア
     └─ Protocol Buffers: バイナリシリアライゼーション

  ② OS 提供のストレージ:
     ├─ Windows: レジストリ、Credential Manager
     ├─ macOS: UserDefaults, Keychain
     └─ Linux: GSettings, Secret Service API

  ③ 暗号化ストレージ:
     ├─ electron-store + safeStorage
     ├─ tauri-plugin-store
     └─ DPAPI (Windows) / Keychain (macOS)

  保存先のベストプラクティス:
    ┌─────────────────────────────────────────────────┐
    │ データの種類       │ 保存先                       │
    │ ─────────────────┼───────────────────────────── │
    │ アプリ設定         │ %APPDATA% / ~/Library/...    │
    │ ユーザーデータ     │ ドキュメントフォルダ            │
    │ キャッシュ         │ %TEMP% / ~/Library/Caches    │
    │ ログ             │ %APPDATA%/logs               │
    │ 認証情報          │ OS の資格情報マネージャー       │
    └─────────────────────────────────────────────────┘
```

```typescript
// Electron でのデータ永続化

// electron-store を使用した設定管理
import Store from 'electron-store';

interface AppSettings {
  theme: 'light' | 'dark' | 'system';
  language: string;
  windowBounds: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  recentFiles: string[];
  autoSave: boolean;
  autoSaveInterval: number; // 秒
}

const store = new Store<AppSettings>({
  defaults: {
    theme: 'system',
    language: 'ja',
    windowBounds: { x: 100, y: 100, width: 1200, height: 800 },
    recentFiles: [],
    autoSave: true,
    autoSaveInterval: 300,
  },
  // 暗号化が必要な場合
  encryptionKey: 'your-encryption-key',
  // スキーマバリデーション
  schema: {
    theme: { type: 'string', enum: ['light', 'dark', 'system'] },
    language: { type: 'string' },
    autoSaveInterval: { type: 'number', minimum: 30, maximum: 3600 },
  },
});

// 使用例
store.set('theme', 'dark');
const theme = store.get('theme');

// ウィンドウ位置の保存と復元
function saveWindowBounds(win: BrowserWindow) {
  const bounds = win.getBounds();
  store.set('windowBounds', bounds);
}

function restoreWindowBounds(): Electron.Rectangle {
  return store.get('windowBounds');
}

// 最近使ったファイルの管理
function addRecentFile(filePath: string) {
  const recent = store.get('recentFiles');
  const updated = [filePath, ...recent.filter(f => f !== filePath)].slice(0, 10);
  store.set('recentFiles', updated);

  // OS のジャンプリスト / 最近使った項目にも追加
  app.addRecentDocument(filePath);
}

// 認証情報の安全な保存（safeStorage）
import { safeStorage } from 'electron';

function saveCredential(key: string, value: string) {
  if (safeStorage.isEncryptionAvailable()) {
    const encrypted = safeStorage.encryptString(value);
    store.set(`credentials.${key}`, encrypted.toString('base64'));
  }
}

function loadCredential(key: string): string | null {
  const encrypted = store.get(`credentials.${key}`) as string | undefined;
  if (encrypted && safeStorage.isEncryptionAvailable()) {
    const buffer = Buffer.from(encrypted, 'base64');
    return safeStorage.decryptString(buffer);
  }
  return null;
}
```

```rust
// Tauri でのデータ永続化

// Cargo.toml
// [dependencies]
// tauri-plugin-store = "2"
// serde = { version = "1", features = ["derive"] }
// serde_json = "1"

use serde::{Deserialize, Serialize};
use tauri_plugin_store::StoreExt;

#[derive(Serialize, Deserialize, Clone)]
struct AppSettings {
    theme: String,
    language: String,
    auto_save: bool,
    auto_save_interval: u32,
    recent_files: Vec<String>,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            theme: "system".to_string(),
            language: "ja".to_string(),
            auto_save: true,
            auto_save_interval: 300,
            recent_files: Vec::new(),
        }
    }
}

#[tauri::command]
fn save_settings(app: tauri::AppHandle, settings: AppSettings) -> Result<(), String> {
    let store = app.store("settings.json")
        .map_err(|e| e.to_string())?;

    store.set("theme", serde_json::to_value(&settings.theme).unwrap());
    store.set("language", serde_json::to_value(&settings.language).unwrap());
    store.set("auto_save", serde_json::to_value(&settings.auto_save).unwrap());
    store.set("recent_files", serde_json::to_value(&settings.recent_files).unwrap());

    store.save().map_err(|e| e.to_string())
}

#[tauri::command]
fn load_settings(app: tauri::AppHandle) -> Result<AppSettings, String> {
    let store = app.store("settings.json")
        .map_err(|e| e.to_string())?;

    let theme = store.get("theme")
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_else(|| "system".to_string());

    Ok(AppSettings {
        theme,
        ..Default::default()
    })
}
```

---

## 9. テスト戦略

### 9.1 デスクトップアプリのテストピラミッド

```
テストピラミッド:

                    ┌─────────────┐
                    │   E2E テスト  │  ← 少数・低速・高コスト
                    │  (Playwright) │
                  ┌─┴─────────────┴─┐
                  │  統合テスト       │  ← 中程度
                  │  (コンポーネント)  │
                ┌─┴─────────────────┴─┐
                │  ユニットテスト       │  ← 多数・高速・低コスト
                │  (Vitest / xUnit)   │
                └─────────────────────┘

  Electron のテスト戦略:
    ユニットテスト:
      → Vitest でビジネスロジックをテスト
      → メインプロセスのモック
      → IPC ハンドラの個別テスト

    統合テスト:
      → React Testing Library で UI コンポーネント
      → Electron API のモック

    E2E テスト:
      → Playwright + @playwright/test
      → electron アプリの起動・操作・検証

  Tauri のテスト戦略:
    ユニットテスト:
      → Vitest でフロントエンドのテスト
      → cargo test で Rust バックエンドのテスト
      → Tauri コマンドの個別テスト

    統合テスト:
      → コンポーネントテスト
      → Tauri mock IPC

    E2E テスト:
      → WebDriver (tauri-driver)
      → Playwright（実験的サポート）
```

```typescript
// Electron E2E テスト（Playwright）
import { test, expect, _electron as electron } from '@playwright/test';

test.describe('メインウィンドウ', () => {
  let electronApp: any;
  let page: any;

  test.beforeAll(async () => {
    electronApp = await electron.launch({
      args: ['.'],
      env: {
        ...process.env,
        NODE_ENV: 'test',
      },
    });
    page = await electronApp.firstWindow();
    await page.waitForLoadState('domcontentloaded');
  });

  test.afterAll(async () => {
    await electronApp.close();
  });

  test('アプリが正常に起動する', async () => {
    const title = await page.title();
    expect(title).toBe('My Electron App');
  });

  test('ファイルを開くダイアログが表示される', async () => {
    await page.click('[data-testid="open-file-btn"]');
    // ダイアログの処理をモック
    const isDialogShown = await electronApp.evaluate(({ dialog }: any) => {
      return dialog.showOpenDialog !== undefined;
    });
    expect(isDialogShown).toBe(true);
  });

  test('テーマの切り替えが動作する', async () => {
    await page.click('[data-testid="theme-toggle"]');
    const isDarkMode = await page.evaluate(() => {
      return document.documentElement.classList.contains('dark');
    });
    expect(isDarkMode).toBe(true);
  });
});
```

---

## 10. 実務的な技術選定ケーススタディ

### 10.1 ケーススタディ: 社内文書管理システム

```
要件:
  ・Windows 10/11 の社内 PC で動作
  ・Active Directory 認証
  ・ローカルファイルの暗号化保存
  ・PDF プレビュー
  ・オフライン動作必須
  ・社内ネットワーク経由で配布

技術選定の検討:

  候補 ①: WinUI 3
    利点:
      → Windows ネイティブの外観
      → AD 認証が .NET ライブラリで容易
      → DPAPI でファイル暗号化
      → MSIX パッケージでグループポリシー配布
    欠点:
      → Windows のみ（将来の macOS 対応が不可）
      → XAML の学習コスト

  候補 ②: Electron
    利点:
      → Web チームのスキル活用
      → PDF.js でプレビュー実装が容易
      → 将来の macOS 対応が可能
    欠点:
      → バンドルサイズが大きい
      → AD 認証の実装が複雑
      → メモリ消費が多い

  候補 ③: Tauri v2
    利点:
      → 軽量で社内 PC の負荷が小さい
      → Rust でセキュアな暗号化処理
      → 将来のクロスプラットフォーム対応
    欠点:
      → AD 認証の Rust ライブラリが限定的
      → 社内の Rust 人材が少ない

  最終選定: WinUI 3
  理由:
    → Windows 限定の要件に最適
    → .NET の業務ライブラリ資産
    → AD 統合の容易さが決定要因
    → IT 部門の MSIX 配布要件との親和性
```

### 10.2 ケーススタディ: クロスプラットフォーム チャットアプリ

```
要件:
  ・Windows / macOS / Linux 対応
  ・リアルタイムメッセージング（WebSocket）
  ・ファイル共有 / 画像プレビュー
  ・通知（システムトレイ常駐）
  ・自動更新
  ・スクリーンショット撮影

技術選定の検討:

  候補 ①: Electron
    利点:
      → Slack / Discord の実績
      → WebSocket は Web 標準で容易
      → 通知 API が成熟
      → electron-updater で自動更新
      → desktopCapturer でスクリーンショット
    欠点:
      → メモリ消費が大きい（常駐アプリとして負担）
      → バンドルサイズ

  候補 ②: Tauri v2
    利点:
      → 常駐アプリとして低メモリ消費
      → tray icon プラグインで簡単実装
      → tauri-plugin-updater で自動更新
    欠点:
      → スクリーンショット機能の実装が複雑
      → WebSocket はフロントエンド側で処理

  最終選定: Electron
  理由:
    → リッチなメディア機能（スクリーンショット、ビデオ通話拡張）
    → WebRTC 統合の容易さ
    → チャットアプリの実績（Slack, Discord, Signal）
    → メモリ消費はトレードオフとして許容
```

---

## まとめ

| 技術 | 最適な用途 | バンドルサイズ | メモリ | 学習コスト |
|------|----------|-------------|--------|----------|
| Electron | 既存 Web アプリのデスクトップ化 | ~150MB | ~200MB | 低 |
| Tauri v2 | 新規の軽量デスクトップアプリ | ~5MB | ~50MB | 中 |
| WinUI 3 | Windows ネイティブ業務アプリ | ~20MB | ~100MB | 中 |
| WPF | Windows 業務アプリ（レガシー含む） | ~20MB | ~100MB | 中 |
| MAUI | .NET マルチプラットフォーム | ~30MB | ~120MB | 中 |
| Flutter | 統一UI のマルチプラットフォーム | ~20MB | ~80MB | 中 |
| Qt | 高性能・産業用途 | ~30MB | ~60MB | 高 |
| PWA | 軽量なWebベースアプリ | 0MB | ブラウザ依存 | 低 |

---

## 次に読むべきガイド
-> [[01-architecture-patterns.md]] -- アーキテクチャパターン

---

## 参考文献
1. Electron. "Quick Start." electronjs.org/docs, 2024.
2. Tauri. "Prerequisites." tauri.app/start, 2024.
3. Microsoft. "Windows App SDK." learn.microsoft.com, 2024.
4. Microsoft. "WPF Overview." learn.microsoft.com, 2024.
5. Flutter. "Desktop support for Flutter." flutter.dev, 2024.
6. Qt. "Qt for Application Development." qt.io, 2024.
7. Electron. "Security Best Practices." electronjs.org/docs/latest/tutorial/security, 2024.
8. Tauri. "Security." tauri.app/security, 2024.

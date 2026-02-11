# デスクトップアプリの全体像

> デスクトップアプリ開発は Web 技術の進化により大きく変化した。ネイティブ、ハイブリッド、Web ベースの各アプローチの特徴を理解し、プロジェクトに最適な技術を選定する基準を解説する。

## この章で学ぶこと

- [ ] デスクトップアプリの種類と技術スタックを理解する
- [ ] Web 技術ベースとネイティブの違いを把握する
- [ ] プロジェクト要件に基づく技術選定ができるようになる

---

## 1. デスクトップアプリの種類

```
デスクトップアプリの分類:

  ① ネイティブ:
     → OS のネイティブ API を直接使用
     → Windows: WPF / WinUI 3 / Win32
     → macOS: SwiftUI / AppKit
     → Linux: GTK / Qt
     → 最高のパフォーマンスと OS 統合
     → OS ごとに別実装が必要

  ② クロスプラットフォーム ネイティブ:
     → 1つのコードベースで複数 OS 対応
     → .NET MAUI（C#）
     → Qt（C++）
     → Flutter Desktop（Dart）
     → ネイティブに近いパフォーマンス

  ③ Web 技術ベース:
     → HTML/CSS/JavaScript でUI構築
     → Electron（Node.js + Chromium）
     → Tauri（Rust + OS WebView）
     → Web 開発者が参入しやすい
     → エコシステムが豊富

  ④ PWA（Progressive Web App）:
     → ブラウザベースのインストール可能アプリ
     → ネイティブ機能は限定的
     → 最も軽量、配布が容易
```

---

## 2. 技術スタックの比較

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

有名アプリの技術スタック:
  Electron:
    → VS Code, Slack, Discord, Notion, Figma Desktop
    → Spotify Desktop, GitHub Desktop, Postman

  Tauri:
    → Cody (Sourcegraph), 1Password (一部)
    → 新規プロジェクトでの採用が増加中

  WPF/WinUI:
    → Visual Studio, Windows Terminal
    → Windows 標準アプリ群

  Qt:
    → VirtualBox, OBS Studio
    → Adobe Substance, Autodesk Maya
```

---

## 3. 技術選定ガイド

```
選定フローチャート:

  Windows のみ？
    │yes              │no
    │                 │
    ネイティブ感重視？    クロスプラットフォーム
    │yes    │no       │
    │       │         │
    WinUI3  │         軽量・セキュリティ重視？
            │         │yes        │no
            │         │           │
            │         Tauri       Electron
            │
            Web 技術で開発したい？
            │yes        │no
            │           │
            Tauri       MAUI / Flutter

  詳細な選定基準:

  Electron を選ぶべき場合:
    → 既存の Web アプリをデスクトップ化
    → Node.js のエコシステムを活用
    → 実績・コミュニティの大きさが重要
    → 複雑な Web 機能（WebRTC、Canvas 等）

  Tauri を選ぶべき場合:
    → バンドルサイズを最小化したい
    → メモリ使用量を抑えたい
    → セキュリティを最優先
    → Rust の恩恵（安全性・パフォーマンス）

  WinUI 3 を選ぶべき場合:
    → Windows ネイティブの外観と操作感
    → .NET / C# の既存資産
    → Windows API の完全なアクセス
    → Microsoft Store での配布

  パフォーマンス要件:
    高: Qt > Tauri > WinUI 3 > Electron
    開発速度:
    高: Electron > Tauri > WinUI 3 > Qt
```

---

## 4. Web 技術ベースの利点と課題

```
利点:
  ✓ Web 開発者のスキルがそのまま活用可能
  ✓ React/Vue/Svelte 等のフレームワーク使用可能
  ✓ npm エコシステムの全資産
  ✓ ホットリロードで高速開発
  ✓ UI の柔軟性（CSS のフルパワー）
  ✓ 1つのコードベースで Web + デスクトップ

課題:
  ✗ ネイティブと比較してメモリ消費が多い
  ✗ 起動時間がやや遅い
  ✗ OS の最新 API への対応に遅れる場合
  ✗ Electron: Chromium 同梱でサイズ大
  ✗ セキュリティ: Web の脆弱性がデスクトップにも影響

対策:
  → Tauri: OS の WebView 使用でサイズ削減
  → Electron: v28+ で ESM 対応、パフォーマンス改善
  → 両方: preload スクリプトでセキュリティ強化
```

---

## 5. 開発環境のセットアップ

```
共通の開発ツール:

  エディタ: VS Code（+ 各フレームワーク拡張）
  パッケージマネージャー: pnpm（推奨）
  バージョン管理: Git
  CI/CD: GitHub Actions

  Electron 追加要件:
    → Node.js 20+
    → electron-forge（ビルドツール）

  Tauri 追加要件:
    → Rust（rustup でインストール）
    → OS 別の依存:
      Windows: Visual Studio Build Tools + WebView2
      macOS: Xcode Command Line Tools
      Linux: webkit2gtk, libappindicator

  WinUI 3 追加要件:
    → Visual Studio 2022
    → Windows App SDK
    → .NET 8+
```

---

## まとめ

| 技術 | 最適な用途 | バンドルサイズ |
|------|----------|-------------|
| Electron | 既存 Web アプリのデスクトップ化 | ~150MB |
| Tauri | 新規の軽量デスクトップアプリ | ~5MB |
| WinUI 3 | Windows ネイティブ業務アプリ | ~20MB |
| MAUI | .NET エコシステム活用 | ~30MB |
| PWA | 軽量なWebベースアプリ | 0MB |

---

## 次に読むべきガイド
→ [[01-architecture-patterns.md]] — アーキテクチャパターン

---

## 参考文献
1. Electron. "Quick Start." electronjs.org/docs, 2024.
2. Tauri. "Prerequisites." tauri.app/start, 2024.
3. Microsoft. "Windows App SDK." learn.microsoft.com, 2024.

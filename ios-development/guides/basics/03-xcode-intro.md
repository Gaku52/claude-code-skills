# Xcode入門 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [Xcodeとは](#xcodeとは)
3. [Xcodeのインストール](#xcodeのインストール)
4. [Xcodeの画面構成](#xcodeの画面構成)
5. [新規プロジェクト作成](#新規プロジェクト作成)
6. [シミュレータの使い方](#シミュレータの使い方)
7. [便利なショートカット](#便利なショートカット)
8. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

- Xcodeの基本操作
- プロジェクトの作成方法
- シミュレータの使い方
- 開発効率を上げるショートカット

### 学習時間：40〜50分

---

## Xcodeとは

### 定義

**Xcode**は、Appleが提供する統合開発環境（IDE）です。

**機能**：
- コードエディタ
- Interface Builder（UI設計ツール）
- デバッガー
- シミュレータ
- パフォーマンス測定ツール
- Git統合

### Xcodeで開発できるもの

```
iOS アプリ       → iPhone/iPad
macOS アプリ     → Mac
watchOS アプリ   → Apple Watch
tvOS アプリ      → Apple TV
```

---

## Xcodeのインストール

### 方法1：Mac App Store（推奨）

```
1. Mac App Storeを開く
2. 「Xcode」で検索
3. 「入手」をクリック
4. ダウンロード完了まで待つ（約13GB）
```

### 方法2：コマンドライン

```bash
# Command Line Toolsのインストール
xcode-select --install

# Xcodeのバージョン確認
xcodebuild -version
```

### システム要件

**最小要件**：
- macOS 13.5 (Ventura) 以降
- 空き容量 40GB以上
- メモリ 8GB以上（16GB推奨）

---

## Xcodeの画面構成

### メイン画面

```
┌─────────────────────────────────────────┐
│  Toolbar（ツールバー）                   │
├────────┬──────────────────┬─────────────┤
│        │                  │             │
│ Nav.   │   Editor Area    │  Inspector  │
│ Area   │   (エディタ)     │  (右側)     │
│(左側)  │                  │             │
│        │                  │             │
├────────┴──────────────────┴─────────────┤
│  Debug Area（デバッグエリア）            │
└─────────────────────────────────────────┘
```

### 1. Toolbar（ツールバー）

```
[Run] [Stop] [デバイス選択] [ステータス]
```

### 2. Navigator Area（左側）

**ナビゲータ**：プロジェクト内のファイルやフォルダを表示

- **Project Navigator**：ファイル一覧
- **Search Navigator**：検索結果
- **Issue Navigator**：エラー・警告一覧
- **Debug Navigator**：実行中の情報

### 3. Editor Area（中央）

**エディタ**：コードやUIを編集

- **Source Editor**：コード編集
- **Interface Builder**：UI設計
- **Preview**：SwiftUIのプレビュー

### 4. Inspector（右側）

**インスペクタ**：選択した要素の詳細設定

- **File Inspector**：ファイル設定
- **Attributes Inspector**：属性設定
- **Size Inspector**：サイズ・位置設定

### 5. Debug Area（下部）

**デバッグエリア**：実行中のログや変数の値を表示

---

## 新規プロジェクト作成

### ステップ1：プロジェクトテンプレート選択

```
1. Xcode起動
2. 「Create New Project」をクリック
3. 「iOS」→「App」を選択
4. 「Next」をクリック
```

### ステップ2：プロジェクト設定

```
Product Name:        MyFirstApp
Team:                (なし、または個人アカウント)
Organization ID:     com.example
Interface:           SwiftUI
Language:            Swift
```

### ステップ3：保存場所選択

```
プロジェクトを保存する場所を選択
例: ~/Documents/Projects/MyFirstApp
```

### プロジェクト構成

```
MyFirstApp/
├── MyFirstApp/
│   ├── MyFirstAppApp.swift    # アプリのエントリーポイント
│   ├── ContentView.swift       # メイン画面
│   └── Assets.xcassets/        # 画像・色などのアセット
├── MyFirstApp.xcodeproj        # プロジェクトファイル
└── README.md
```

---

## シミュレータの使い方

### シミュレータとは

**シミュレータ**は、Mac上でiPhoneやiPadの動作を再現するツールです。

### デバイス選択

```
Toolbar → デバイス選択ボタン
例: iPhone 15 Pro, iPad Pro (12.9-inch)
```

### 実行

```bash
# 方法1：ツールバーの再生ボタン
⌘ + R

# 方法2：メニュー
Product → Run
```

### シミュレータの操作

#### ホームボタン

```
Shift + ⌘ + H  # ホーム画面に戻る
```

#### 画面回転

```
⌘ + ← / ⌘ + →  # 左右回転
```

#### スクリーンショット

```
⌘ + S  # スクリーンショットを保存
```

#### デバイスを変更

```
Window → Destination → Choose Destination
```

---

## 便利なショートカット

### 基本操作

| ショートカット | 動作 |
|--------------|------|
| **⌘ + R** | ビルド＆実行 |
| **⌘ + B** | ビルドのみ |
| **⌘ + .** | 実行停止 |
| **⌘ + Shift + K** | クリーンビルド |

### エディタ

| ショートカット | 動作 |
|--------------|------|
| **⌘ + /** | コメントアウト |
| **⌘ + [** | インデント削除 |
| **⌘ + ]** | インデント追加 |
| **⌘ + ⌥ + [** | 行を上に移動 |
| **⌘ + ⌥ + ]** | 行を下に移動 |

### ナビゲーション

| ショートカット | 動作 |
|--------------|------|
| **⌘ + Shift + O** | ファイル検索 |
| **⌘ + Shift + F** | プロジェクト内検索 |
| **⌘ + Shift + J** | Navigatorで現在のファイルを表示 |
| **⌘ + クリック** | 定義へジャンプ |

### その他

| ショートカット | 動作 |
|--------------|------|
| **⌘ + 0** | Navigator表示/非表示 |
| **⌘ + ⌥ + 0** | Inspector表示/非表示 |
| **⌘ + Shift + Y** | Debug Area表示/非表示 |
| **⌘ + ⌥ + Enter** | Preview表示 |

---

## SwiftUIのプレビュー

### Previewとは

**Preview**は、SwiftUIのUIをリアルタイムで確認できる機能です。

```swift
struct ContentView: View {
    var body: some View {
        Text("Hello, World!")
            .font(.title)
            .foregroundColor(.blue)
    }
}

// プレビュー
#Preview {
    ContentView()
}
```

### プレビューの操作

```
Resume:   ⌘ + ⌥ + P （プレビュー更新）
Pin:      現在のプレビューを固定
Live:     インタラクティブモードに切り替え
```

---

## 実機でのテスト

### 必要なもの

- iPhone/iPad（実機）
- USBケーブル
- Apple ID（無料アカウントでOK）

### 手順

#### 1. デバイスを接続

```
iPhoneをMacにUSBケーブルで接続
```

#### 2. チームを設定

```
1. Project Settings → Signing & Capabilities
2. Team: 自分のApple IDを選択
3. 「Automatically manage signing」にチェック
```

#### 3. デバイスを選択

```
Toolbar → デバイス選択 → 接続した実機を選択
```

#### 4. 実行

```
⌘ + R で実行
初回は「信頼する」ダイアログが表示される
```

---

## よくあるエラーと解決方法

### エラー1：「Command CodeSign failed」

**原因**：コード署名の問題

**解決**：
```
1. Project Settings → Signing & Capabilities
2. Team を正しく設定
3. Bundle Identifier がユニークか確認
```

### エラー2：「Unable to boot simulator」

**原因**：シミュレータの不具合

**解決**：
```
1. シミュレータを閉じる
2. Xcode → Product → Clean Build Folder
3. Mac を再起動
```

### エラー3：「No such module 'SwiftUI'」

**原因**：ターゲットOSが古い

**解決**：
```
Project Settings → Deployment Target → iOS 15.0以上に設定
```

---

## 演習問題

### 問題：Hello Worldアプリを作る

以下の手順で、簡単なアプリを作ってください：

1. 新規プロジェクト作成
2. `ContentView.swift`を開く
3. テキストを「Hello, World!」に変更
4. フォントサイズを大きくする
5. 色を変更する
6. シミュレータで実行

**解答例**：

```swift
import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack {
            Text("Hello, World!")
                .font(.largeTitle)
                .fontWeight(.bold)
                .foregroundColor(.blue)

            Text("Welcome to iOS Development")
                .font(.title2)
                .foregroundColor(.gray)
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
```

---

## 次のステップ

### このガイドで学んだこと

- ✅ Xcodeの基本操作
- ✅ プロジェクトの作成方法
- ✅ シミュレータの使い方
- ✅ 便利なショートカット

### 次に学ぶべきガイド

**次のガイド**：[04-swiftui-basics.md](./04-swiftui-basics.md) - SwiftUI基礎

---

**前のガイド**：[02-swift-basics.md](./02-swift-basics.md)

**親ガイド**：[iOS Development - SKILL.md](../../SKILL.md)

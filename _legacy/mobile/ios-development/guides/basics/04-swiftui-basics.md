# SwiftUI基礎 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [SwiftUIとは](#swiftuiとは)
3. [基本的なView](#基本的なview)
4. [レイアウト](#レイアウト)
5. [修飾子(Modifier)](#修飾子modifier)
6. [画像とアイコン](#画像とアイコン)
7. [リスト表示](#リスト表示)
8. [演習問題](#演習問題)
9. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

- SwiftUIの基本概念
- 基本的なViewコンポーネント
- レイアウトの作成方法
- 修飾子(Modifier)の使い方

### 学習時間：1〜1.5時間

---

## SwiftUIとは

### 定義

**SwiftUI**は、Appleが2019年に発表した宣言的UIフレームワークです。

### 従来のUIKitとの違い

#### UIKit（命令的）

```swift
// UIKit - 命令的
let label = UILabel()
label.text = "Hello"
label.textColor = .blue
label.font = .systemFont(ofSize: 24)
view.addSubview(label)
```

#### SwiftUI（宣言的）

```swift
// SwiftUI - 宣言的
Text("Hello")
    .foregroundColor(.blue)
    .font(.system(size: 24))
```

### SwiftUIの特徴

- **宣言的構文**：「何を表示するか」を記述
- **ライブプレビュー**：コードを書くと即座に確認
- **クロスプラットフォーム**：iOS/macOS/watchOS/tvOSで動作
- **自動レイアウト**：デバイスサイズに自動対応

---

## 基本的なView

### Text（テキスト）

```swift
// 基本
Text("Hello, World!")

// スタイル付き
Text("大きな文字")
    .font(.largeTitle)
    .foregroundColor(.blue)
    .fontWeight(.bold)

// 複数行
Text("これは\n複数行の\nテキストです")
    .multilineTextAlignment(.center)
```

### Image（画像）

```swift
// システムアイコン（SF Symbols）
Image(systemName: "star.fill")
    .foregroundColor(.yellow)
    .font(.system(size: 50))

// カスタム画像
Image("myImage")
    .resizable()
    .scaledToFit()
    .frame(width: 100, height: 100)
```

### Button（ボタン）

```swift
// 基本
Button("タップ") {
    print("タップされました")
}

// スタイル付き
Button(action: {
    print("タップ")
}) {
    Text("送信")
        .font(.title)
        .padding()
        .background(Color.blue)
        .foregroundColor(.white)
        .cornerRadius(10)
}
```

---

## レイアウト

### VStack（縦並び）

```swift
VStack {
    Text("1行目")
    Text("2行目")
    Text("3行目")
}

// 間隔とアライメント
VStack(alignment: .leading, spacing: 20) {
    Text("左寄せ")
    Text("20pxの間隔")
}
```

### HStack（横並び）

```swift
HStack {
    Image(systemName: "star.fill")
    Text("評価")
    Text("4.5")
}

// 均等配置
HStack {
    Text("左")
    Spacer()  // 余白
    Text("右")
}
```

### ZStack（重ね合わせ）

```swift
ZStack {
    // 背景
    Color.blue
        .ignoresSafeArea()

    // 前景
    Text("重なっています")
        .foregroundColor(.white)
        .font(.largeTitle)
}
```

### 複合レイアウト

```swift
VStack {
    HStack {
        Image(systemName: "person.circle")
            .font(.largeTitle)

        VStack(alignment: .leading) {
            Text("太郎")
                .font(.title)
            Text("東京都")
                .font(.subheadline)
                .foregroundColor(.gray)
        }
    }
    .padding()

    Divider()  // 区切り線

    Text("プロフィール内容...")
        .padding()
}
```

---

## 修飾子(Modifier)

### サイズ・位置

```swift
Text("固定サイズ")
    .frame(width: 200, height: 100)

Text("最小サイズ")
    .frame(minWidth: 100, minHeight: 50)

Text("最大サイズ")
    .frame(maxWidth: .infinity, maxHeight: 200)
```

### 余白・背景

```swift
Text("Padding")
    .padding()  // デフォルト余白

Text("Custom Padding")
    .padding(.horizontal, 20)  // 左右20px
    .padding(.vertical, 10)    // 上下10px

Text("背景色")
    .padding()
    .background(Color.blue)
    .cornerRadius(10)
```

### 境界線・影

```swift
// 境界線
Text("Border")
    .padding()
    .border(Color.blue, width: 2)

// 角丸
Text("Rounded")
    .padding()
    .background(Color.blue)
    .cornerRadius(10)

// 影
Text("Shadow")
    .padding()
    .shadow(color: .gray, radius: 5, x: 0, y: 5)
```

---

## 画像とアイコン

### SF Symbols

**SF Symbols**は、Appleが提供する5,000以上のアイコンセットです。

```swift
// 基本
Image(systemName: "heart.fill")

// サイズ変更
Image(systemName: "star.fill")
    .font(.system(size: 50))

// 色変更
Image(systemName: "house.fill")
    .foregroundColor(.blue)

// よく使うアイコン
VStack(spacing: 20) {
    Image(systemName: "envelope.fill")  // メール
    Image(systemName: "phone.fill")     // 電話
    Image(systemName: "gear")           // 設定
    Image(systemName: "person.circle")  // ユーザー
    Image(systemName: "magnifyingglass") // 検索
}
.font(.largeTitle)
```

### カスタム画像

```swift
// Assets.xcassetsに追加した画像
Image("myPhoto")
    .resizable()
    .scaledToFit()
    .frame(width: 200, height: 200)
    .clipShape(Circle())  // 円形にクリップ
    .shadow(radius: 10)
```

---

## リスト表示

### List

```swift
// 基本
List {
    Text("項目1")
    Text("項目2")
    Text("項目3")
}

// 配列から生成
let fruits = ["りんご", "バナナ", "ぶどう"]

List(fruits, id: \.self) { fruit in
    Text(fruit)
}

// セクション付き
List {
    Section(header: Text("果物")) {
        Text("りんご")
        Text("バナナ")
    }

    Section(header: Text("野菜")) {
        Text("にんじん")
        Text("トマト")
    }
}
```

### カスタムリスト

```swift
struct Item: Identifiable {
    let id = UUID()
    let name: String
    let icon: String
}

let items = [
    Item(name: "ホーム", icon: "house.fill"),
    Item(name: "設定", icon: "gear"),
    Item(name: "プロフィール", icon: "person.circle")
]

List(items) { item in
    HStack {
        Image(systemName: item.icon)
            .foregroundColor(.blue)
        Text(item.name)
            .font(.headline)
    }
}
```

---

## 実践例

### Example 1: プロフィールカード

```swift
struct ProfileCard: View {
    var body: some View {
        VStack(spacing: 20) {
            // プロフィール画像
            Image(systemName: "person.circle.fill")
                .font(.system(size: 100))
                .foregroundColor(.blue)

            // 名前
            Text("山田太郎")
                .font(.title)
                .fontWeight(.bold)

            // 役職
            Text("iOS Developer")
                .font(.subheadline)
                .foregroundColor(.gray)

            // SNSボタン
            HStack(spacing: 40) {
                Button(action: {}) {
                    Image(systemName: "envelope.fill")
                        .font(.title2)
                }

                Button(action: {}) {
                    Image(systemName: "phone.fill")
                        .font(.title2)
                }

                Button(action: {}) {
                    Image(systemName: "link")
                        .font(.title2)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(20)
        .padding()
    }
}

#Preview {
    ProfileCard()
}
```

### Example 2: タスクリスト

```swift
struct TaskListView: View {
    let tasks = [
        "メールを送る",
        "資料を作成",
        "会議に参加",
        "コードレビュー"
    ]

    var body: some View {
        NavigationView {
            List(tasks, id: \.self) { task in
                HStack {
                    Image(systemName: "circle")
                        .foregroundColor(.gray)
                    Text(task)
                        .font(.body)
                }
            }
            .navigationTitle("タスク")
        }
    }
}

#Preview {
    TaskListView()
}
```

---

## よくある間違い

### ❌ 間違い1：Viewを返さない

```swift
struct MyView: View {
    var body: some View {
        print("Hello")  // エラー：Viewを返していない
    }
}
```

**✅ 正しい方法**：

```swift
struct MyView: View {
    var body: some View {
        Text("Hello")
    }
}
```

### ❌ 間違い2：複数のルートView

```swift
struct MyView: View {
    var body: some View {
        Text("1")
        Text("2")  // エラー：ルートViewは1つ
    }
}
```

**✅ 正しい方法**：

```swift
struct MyView: View {
    var body: some View {
        VStack {
            Text("1")
            Text("2")
        }
    }
}
```

---

## 演習問題

### 問題：天気カードを作る

以下の要件でカードを作成してください：
- 天気アイコン（sun.max.fill）
- 気温（25°C）
- 場所（東京）
- 青い背景、白文字

**解答例**：

```swift
struct WeatherCard: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "sun.max.fill")
                .font(.system(size: 80))
                .foregroundColor(.yellow)

            Text("25°C")
                .font(.system(size: 60, weight: .bold))
                .foregroundColor(.white)

            Text("東京")
                .font(.title2)
                .foregroundColor(.white)
        }
        .frame(width: 300, height: 400)
        .background(
            LinearGradient(
                colors: [.blue, .cyan],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .cornerRadius(30)
        .shadow(radius: 10)
    }
}

#Preview {
    WeatherCard()
}
```

---

## 次のステップ

### このガイドで学んだこと

- ✅ SwiftUIの基本概念
- ✅ 基本的なViewコンポーネント
- ✅ レイアウトの作成方法
- ✅ 修飾子(Modifier)の使い方

### 次に学ぶべきガイド

**次のガイド**：[05-view-layout.md](./05-view-layout.md) - レイアウトとUI

---

**前のガイド**：[03-xcode-intro.md](./03-xcode-intro.md)

**親ガイド**：[iOS Development - SKILL.md](../../SKILL.md)

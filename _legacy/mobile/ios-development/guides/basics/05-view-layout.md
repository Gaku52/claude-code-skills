# レイアウトとUI - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [レイアウトの基礎](#レイアウトの基礎)
3. [Spacer](#spacer)
4. [Divider](#divider)
5. [ScrollView](#scrollview)
6. [GeometryReader](#geometryreader)
7. [カスタムコンポーネント](#カスタムコンポーネント)
8. [演習問題](#演習問題)
9. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

- 高度なレイアウト技術
- Spacer、Divider の使い方
- ScrollView によるスクロール実装
- カスタムViewコンポーネントの作成

### 学習時間：1〜1.5時間

---

## レイアウトの基礎

### アライメント（配置）

```swift
// 左寄せ
VStack(alignment: .leading) {
    Text("左寄せ1")
    Text("左寄せ2")
}

// 中央寄せ（デフォルト）
VStack(alignment: .center) {
    Text("中央1")
    Text("中央2")
}

// 右寄せ
VStack(alignment: .trailing) {
    Text("右寄せ1")
    Text("右寄せ2")
}
```

### スペーシング（間隔）

```swift
// 間隔なし
VStack(spacing: 0) {
    Text("1")
    Text("2")
}

// 間隔20px
VStack(spacing: 20) {
    Text("1")
    Text("2")
}

// 個別に間隔指定
VStack {
    Text("1")
    Text("2")
        .padding(.top, 30)
}
```

---

## Spacer

### Spacerとは

**Spacer**は、余白を自動的に埋める要素です。

```swift
// 上下に均等配置
VStack {
    Text("上")
    Spacer()
    Text("下")
}

// 左右に均等配置
HStack {
    Text("左")
    Spacer()
    Text("右")
}
```

### 実践例

#### ツールバー風レイアウト

```swift
HStack {
    Button(action: {}) {
        Image(systemName: "chevron.left")
    }

    Spacer()

    Text("タイトル")
        .font(.headline)

    Spacer()

    Button(action: {}) {
        Image(systemName: "ellipsis")
    }
}
.padding()
```

#### フッターボタン

```swift
VStack {
    Text("コンテンツ")
        .padding()

    Spacer()

    Button("続ける") {}
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color.blue)
        .foregroundColor(.white)
        .cornerRadius(10)
        .padding()
}
```

---

## Divider

### Dividerとは

**Divider**は、区切り線を表示します。

```swift
// 水平線
VStack {
    Text("セクション1")
    Divider()
    Text("セクション2")
}

// カスタマイズ
VStack {
    Text("セクション1")

    Rectangle()
        .fill(Color.blue)
        .frame(height: 2)

    Text("セクション2")
}
```

---

## ScrollView

### ScrollViewとは

**ScrollView**は、スクロール可能なコンテンツを表示します。

```swift
// 縦スクロール
ScrollView {
    VStack(spacing: 20) {
        ForEach(1...20, id: \.self) { i in
            Text("アイテム \(i)")
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue.opacity(0.1))
                .cornerRadius(10)
        }
    }
    .padding()
}

// 横スクロール
ScrollView(.horizontal, showsIndicators: false) {
    HStack(spacing: 20) {
        ForEach(1...10, id: \.self) { i in
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.blue)
                .frame(width: 150, height: 200)
                .overlay(Text("カード \(i)").foregroundColor(.white))
        }
    }
    .padding()
}
```

### 実践例：ニュースフィード

```swift
struct NewsItem {
    let title: String
    let description: String
    let time: String
}

struct NewsFeedView: View {
    let news = [
        NewsItem(title: "新機能リリース", description: "SwiftUI 5.0がリリースされました", time: "1時間前"),
        NewsItem(title: "アップデート情報", description: "iOS 18の新機能", time: "2時間前"),
        NewsItem(title: "開発者向けイベント", description: "WWDC 2024開催", time: "3時間前")
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                ForEach(news.indices, id: \.self) { index in
                    VStack(alignment: .leading, spacing: 10) {
                        Text(news[index].title)
                            .font(.headline)

                        Text(news[index].description)
                            .font(.subheadline)
                            .foregroundColor(.gray)

                        Text(news[index].time)
                            .font(.caption)
                            .foregroundColor(.blue)
                    }
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
                }
            }
            .padding()
        }
    }
}

#Preview {
    NewsFeedView()
}
```

---

## GeometryReader

### GeometryReaderとは

**GeometryReader**は、親Viewのサイズを取得します。

```swift
GeometryReader { geometry in
    Text("幅: \(geometry.size.width)")
        .frame(width: geometry.size.width, height: geometry.size.height)
        .background(Color.blue)
}
```

### 実践例：レスポンシブカード

```swift
struct ResponsiveCard: View {
    var body: some View {
        GeometryReader { geometry in
            VStack {
                Rectangle()
                    .fill(Color.blue)
                    .frame(width: geometry.size.width * 0.9, height: 200)
                    .cornerRadius(10)

                Text("幅の90%")
                    .padding()
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
        }
    }
}
```

---

## カスタムコンポーネント

### 再利用可能なView

```swift
// カスタムボタン
struct PrimaryButton: View {
    let title: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.headline)
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue)
                .cornerRadius(10)
        }
    }
}

// 使用例
VStack {
    PrimaryButton(title: "保存") {
        print("保存")
    }

    PrimaryButton(title: "キャンセル") {
        print("キャンセル")
    }
}
.padding()
```

### カスタムカード

```swift
struct InfoCard: View {
    let icon: String
    let title: String
    let value: String

    var body: some View {
        HStack(spacing: 15) {
            Image(systemName: icon)
                .font(.largeTitle)
                .foregroundColor(.blue)
                .frame(width: 50)

            VStack(alignment: .leading, spacing: 5) {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.gray)

                Text(value)
                    .font(.title2)
                    .fontWeight(.bold)
            }

            Spacer()
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }
}

// 使用例
VStack(spacing: 15) {
    InfoCard(icon: "person.fill", title: "ユーザー数", value: "1,234")
    InfoCard(icon: "heart.fill", title: "いいね数", value: "5,678")
    InfoCard(icon: "eye.fill", title: "閲覧数", value: "9,012")
}
.padding()
```

---

## 実践例

### Example 1: プロフィール画面

```swift
struct ProfileView: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // ヘッダー
                VStack(spacing: 15) {
                    Image(systemName: "person.circle.fill")
                        .font(.system(size: 100))
                        .foregroundColor(.blue)

                    Text("山田太郎")
                        .font(.title)
                        .fontWeight(.bold)

                    Text("iOS Developer")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                }
                .padding()

                Divider()

                // 統計情報
                HStack(spacing: 20) {
                    StatView(value: "123", label: "投稿")
                    Divider()
                    StatView(value: "456", label: "フォロワー")
                    Divider()
                    StatView(value: "789", label: "フォロー中")
                }
                .frame(height: 60)
                .padding(.horizontal)

                Divider()

                // プロフィール情報
                VStack(alignment: .leading, spacing: 15) {
                    ProfileRow(icon: "envelope.fill", text: "taro@example.com")
                    ProfileRow(icon: "mappin.circle.fill", text: "東京都")
                    ProfileRow(icon: "link", text: "https://example.com")
                }
                .padding()
            }
        }
    }
}

struct StatView: View {
    let value: String
    let label: String

    var body: some View {
        VStack {
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
            Text(label)
                .font(.caption)
                .foregroundColor(.gray)
        }
        .frame(maxWidth: .infinity)
    }
}

struct ProfileRow: View {
    let icon: String
    let text: String

    var body: some View {
        HStack(spacing: 15) {
            Image(systemName: icon)
                .foregroundColor(.blue)
                .frame(width: 30)

            Text(text)
                .font(.body)

            Spacer()
        }
    }
}

#Preview {
    ProfileView()
}
```

### Example 2: ダッシュボード

```swift
struct DashboardView: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // ヘッダー
                HStack {
                    VStack(alignment: .leading) {
                        Text("おはようございます")
                            .font(.subheadline)
                            .foregroundColor(.gray)
                        Text("太郎さん")
                            .font(.title)
                            .fontWeight(.bold)
                    }

                    Spacer()

                    Image(systemName: "bell.fill")
                        .font(.title2)
                        .foregroundColor(.blue)
                }
                .padding()

                // 統計カード
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 15) {
                        DashboardCard(icon: "chart.line.uptrend.xyaxis", title: "売上", value: "¥123,456", color: .blue)
                        DashboardCard(icon: "person.2.fill", title: "ユーザー", value: "1,234", color: .green)
                        DashboardCard(icon: "bag.fill", title: "注文", value: "567", color: .orange)
                    }
                    .padding(.horizontal)
                }

                // セクション
                VStack(alignment: .leading, spacing: 15) {
                    Text("最近のアクティビティ")
                        .font(.headline)
                        .padding(.horizontal)

                    VStack(spacing: 10) {
                        ActivityRow(title: "新規注文", time: "5分前", icon: "cart.fill")
                        ActivityRow(title: "ユーザー登録", time: "10分前", icon: "person.badge.plus")
                        ActivityRow(title: "レビュー投稿", time: "15分前", icon: "star.fill")
                    }
                }
            }
            .padding(.vertical)
        }
    }
}

struct DashboardCard: View {
    let icon: String
    let title: String
    let value: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Image(systemName: icon)
                .font(.largeTitle)
                .foregroundColor(color)

            VStack(alignment: .leading, spacing: 5) {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.gray)

                Text(value)
                    .font(.title2)
                    .fontWeight(.bold)
            }
        }
        .frame(width: 150, height: 120)
        .padding()
        .background(color.opacity(0.1))
        .cornerRadius(15)
    }
}

struct ActivityRow: View {
    let title: String
    let time: String
    let icon: String

    var body: some View {
        HStack(spacing: 15) {
            Image(systemName: icon)
                .foregroundColor(.blue)
                .frame(width: 30)

            VStack(alignment: .leading) {
                Text(title)
                    .font(.headline)
                Text(time)
                    .font(.caption)
                    .foregroundColor(.gray)
            }

            Spacer()

            Image(systemName: "chevron.right")
                .foregroundColor(.gray)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
        .padding(.horizontal)
    }
}

#Preview {
    DashboardView()
}
```

---

## 演習問題

### 問題：設定画面を作る

以下の要件で設定画面を作成してください：
- スクロール可能
- セクション分け（アカウント、通知、その他）
- 各項目にアイコンとタイトル

**解答例**：

```swift
struct SettingsView: View {
    var body: some View {
        NavigationView {
            List {
                Section(header: Text("アカウント")) {
                    SettingRow(icon: "person.fill", title: "プロフィール")
                    SettingRow(icon: "lock.fill", title: "プライバシー")
                }

                Section(header: Text("通知")) {
                    SettingRow(icon: "bell.fill", title: "プッシュ通知")
                    SettingRow(icon: "envelope.fill", title: "メール通知")
                }

                Section(header: Text("その他")) {
                    SettingRow(icon: "info.circle.fill", title: "アプリ情報")
                    SettingRow(icon: "arrow.right.square.fill", title: "ログアウト")
                }
            }
            .navigationTitle("設定")
        }
    }
}

struct SettingRow: View {
    let icon: String
    let title: String

    var body: some View {
        HStack(spacing: 15) {
            Image(systemName: icon)
                .foregroundColor(.blue)
                .frame(width: 25)

            Text(title)

            Spacer()

            Image(systemName: "chevron.right")
                .foregroundColor(.gray)
                .font(.caption)
        }
    }
}

#Preview {
    SettingsView()
}
```

---

## 次のステップ

### このガイドで学んだこと

- ✅ 高度なレイアウト技術
- ✅ Spacer、Divider の使い方
- ✅ ScrollView によるスクロール実装
- ✅ カスタムViewコンポーネントの作成

### 次に学ぶべきガイド

**次のガイド**：[06-state-management.md](./06-state-management.md) - 状態管理

---

**前のガイド**：[04-swiftui-basics.md](./04-swiftui-basics.md)

**親ガイド**：[iOS Development - SKILL.md](../../SKILL.md)

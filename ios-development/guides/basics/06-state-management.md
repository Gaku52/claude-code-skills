# 状態管理 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [状態管理とは](#状態管理とは)
3. [@State](#state)
4. [@Binding](#binding)
5. [@StateObject と @ObservedObject](#stateobject-と-observedobject)
6. [フォーム入力](#フォーム入力)
7. [演習問題](#演習問題)
8. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

- SwiftUIの状態管理の基礎
- @State による状態管理
- @Binding による親子間のデータ共有
- @StateObject / @ObservedObject の使い方

### 学習時間：1〜1.5時間

---

## 状態管理とは

### 定義

**状態（State）**とは、アプリが保持する「変化する値」のことです。

**例**：
- ボタンが押されたか
- テキストフィールドの入力内容
- いいねボタンの ON/OFF
- リストの項目数

### SwiftUIの状態管理

SwiftUIは**宣言的UI**です。状態が変わると、自動的にUIが更新されます。

```
状態変更 → UIが自動更新
```

---

## @State

### @Stateとは

**@State**は、View内で管理する状態を定義します。

```swift
struct CounterView: View {
    @State private var count = 0  // 状態

    var body: some View {
        VStack {
            Text("カウント: \(count)")
                .font(.largeTitle)

            Button("増やす") {
                count += 1  // 状態を変更 → 自動的にUIが更新
            }
        }
    }
}
```

### @Stateの使い方

```swift
// Bool
@State private var isOn = false

Toggle("通知", isOn: $isOn)

// String
@State private var text = ""

TextField("名前", text: $text)

// Int
@State private var score = 0

Stepper("スコア: \(score)", value: $score, in: 0...100)
```

### 実践例：いいねボタン

```swift
struct LikeButton: View {
    @State private var isLiked = false
    @State private var likeCount = 123

    var body: some View {
        HStack {
            Button(action: {
                isLiked.toggle()
                likeCount += isLiked ? 1 : -1
            }) {
                Image(systemName: isLiked ? "heart.fill" : "heart")
                    .foregroundColor(isLiked ? .red : .gray)
                    .font(.title)
            }

            Text("\(likeCount)")
                .font(.title2)
        }
    }
}

#Preview {
    LikeButton()
}
```

---

## @Binding

### @Bindingとは

**@Binding**は、親Viewから子Viewへ状態を共有します。

```swift
// 親View
struct ParentView: View {
    @State private var isOn = false

    var body: some View {
        VStack {
            Text(isOn ? "ON" : "OFF")

            // 子Viewに状態を渡す
            ChildView(isOn: $isOn)
        }
    }
}

// 子View
struct ChildView: View {
    @Binding var isOn: Bool  // 親から受け取る

    var body: some View {
        Toggle("スイッチ", isOn: $isOn)
    }
}
```

### 実践例：モーダル表示

```swift
struct ContentView: View {
    @State private var showModal = false

    var body: some View {
        VStack {
            Button("モーダルを開く") {
                showModal = true
            }
        }
        .sheet(isPresented: $showModal) {
            ModalView(isPresented: $showModal)
        }
    }
}

struct ModalView: View {
    @Binding var isPresented: Bool

    var body: some View {
        VStack(spacing: 20) {
            Text("モーダル")
                .font(.title)

            Button("閉じる") {
                isPresented = false
            }
        }
        .padding()
    }
}
```

---

## @StateObject と @ObservedObject

### ObservableObject

**ObservableObject**は、複雑な状態を管理するクラスです。

```swift
import Combine

class UserViewModel: ObservableObject {
    @Published var name = ""
    @Published var age = 0
    @Published var isLoggedIn = false

    func login() {
        isLoggedIn = true
    }

    func logout() {
        isLoggedIn = false
    }
}
```

### @StateObject

**@StateObject**は、ViewがViewModelを「所有」します。

```swift
struct ContentView: View {
    @StateObject private var viewModel = UserViewModel()

    var body: some View {
        VStack {
            Text("名前: \(viewModel.name)")
            Text("年齢: \(viewModel.age)")

            if viewModel.isLoggedIn {
                Button("ログアウト") {
                    viewModel.logout()
                }
            } else {
                Button("ログイン") {
                    viewModel.login()
                }
            }
        }
    }
}
```

### @ObservedObject

**@ObservedObject**は、ViewModelを親から受け取ります。

```swift
struct ChildView: View {
    @ObservedObject var viewModel: UserViewModel

    var body: some View {
        Text("名前: \(viewModel.name)")
    }
}
```

### @StateObject vs @ObservedObject

| 項目 | @StateObject | @ObservedObject |
|------|-------------|-----------------|
| **所有** | Viewが所有 | 親から受け取る |
| **ライフサイクル** | Viewと同じ | 親に依存 |
| **使用例** | ルートView | 子View |

---

## フォーム入力

### TextField

```swift
struct LoginForm: View {
    @State private var email = ""
    @State private var password = ""

    var body: some View {
        VStack(spacing: 20) {
            TextField("メールアドレス", text: $email)
                .textFieldStyle(.roundedBorder)
                .textInputAutocapitalization(.never)
                .keyboardType(.emailAddress)

            SecureField("パスワード", text: $password)
                .textFieldStyle(.roundedBorder)

            Button("ログイン") {
                print("Email: \(email)")
                print("Password: \(password)")
            }
            .disabled(email.isEmpty || password.isEmpty)
        }
        .padding()
    }
}
```

### Picker

```swift
struct SettingsView: View {
    @State private var selectedTheme = "Light"
    let themes = ["Light", "Dark", "Auto"]

    var body: some View {
        Form {
            Picker("テーマ", selection: $selectedTheme) {
                ForEach(themes, id: \.self) { theme in
                    Text(theme)
                }
            }
        }
    }
}
```

### Slider

```swift
struct VolumeControl: View {
    @State private var volume: Double = 50

    var body: some View {
        VStack {
            Text("音量: \(Int(volume))")
                .font(.title)

            Slider(value: $volume, in: 0...100, step: 1)
                .padding()
        }
    }
}
```

---

## 実践例

### Example 1: TODOリスト

```swift
struct TodoItem: Identifiable {
    let id = UUID()
    var title: String
    var isDone = false
}

class TodoViewModel: ObservableObject {
    @Published var todos: [TodoItem] = [
        TodoItem(title: "買い物"),
        TodoItem(title: "掃除"),
        TodoItem(title: "勉強")
    ]

    func toggleDone(item: TodoItem) {
        if let index = todos.firstIndex(where: { $0.id == item.id }) {
            todos[index].isDone.toggle()
        }
    }

    func addTodo(_ title: String) {
        todos.append(TodoItem(title: title))
    }

    func deleteTodo(at offsets: IndexSet) {
        todos.remove(atOffsets: offsets)
    }
}

struct TodoListView: View {
    @StateObject private var viewModel = TodoViewModel()
    @State private var newTodoTitle = ""

    var body: some View {
        NavigationView {
            VStack {
                // 入力欄
                HStack {
                    TextField("新しいTODO", text: $newTodoTitle)
                        .textFieldStyle(.roundedBorder)

                    Button(action: {
                        if !newTodoTitle.isEmpty {
                            viewModel.addTodo(newTodoTitle)
                            newTodoTitle = ""
                        }
                    }) {
                        Image(systemName: "plus.circle.fill")
                            .font(.title2)
                    }
                }
                .padding()

                // TODOリスト
                List {
                    ForEach(viewModel.todos) { todo in
                        HStack {
                            Image(systemName: todo.isDone ? "checkmark.circle.fill" : "circle")
                                .foregroundColor(todo.isDone ? .green : .gray)
                                .onTapGesture {
                                    viewModel.toggleDone(item: todo)
                                }

                            Text(todo.title)
                                .strikethrough(todo.isDone)
                                .foregroundColor(todo.isDone ? .gray : .primary)
                        }
                    }
                    .onDelete(perform: viewModel.deleteTodo)
                }
            }
            .navigationTitle("TODO")
        }
    }
}

#Preview {
    TodoListView()
}
```

### Example 2: カウンターアプリ

```swift
class CounterViewModel: ObservableObject {
    @Published var count = 0
    @Published var step = 1

    func increment() {
        count += step
    }

    func decrement() {
        count -= step
    }

    func reset() {
        count = 0
    }
}

struct CounterView: View {
    @StateObject private var viewModel = CounterViewModel()

    var body: some View {
        VStack(spacing: 30) {
            // カウント表示
            Text("\(viewModel.count)")
                .font(.system(size: 80, weight: .bold))
                .foregroundColor(.blue)

            // ステップ設定
            Stepper("ステップ: \(viewModel.step)", value: $viewModel.step, in: 1...10)
                .padding(.horizontal)

            // ボタン
            HStack(spacing: 20) {
                Button(action: viewModel.decrement) {
                    Image(systemName: "minus.circle.fill")
                        .font(.system(size: 60))
                        .foregroundColor(.red)
                }

                Button(action: viewModel.reset) {
                    Image(systemName: "arrow.counterclockwise.circle.fill")
                        .font(.system(size: 60))
                        .foregroundColor(.gray)
                }

                Button(action: viewModel.increment) {
                    Image(systemName: "plus.circle.fill")
                        .font(.system(size: 60))
                        .foregroundColor(.green)
                }
            }
        }
        .padding()
    }
}

#Preview {
    CounterView()
}
```

---

## よくある間違い

### ❌ 間違い1：@Stateを使わない

```swift
struct CounterView: View {
    var count = 0  // エラー：@Stateがない

    var body: some View {
        Button("増やす") {
            count += 1  // UIが更新されない
        }
    }
}
```

**✅ 正しい方法**：

```swift
struct CounterView: View {
    @State private var count = 0

    var body: some View {
        Button("増やす") {
            count += 1  // UIが自動更新
        }
    }
}
```

### ❌ 間違い2：@Bindingで$を忘れる

```swift
struct ParentView: View {
    @State private var text = ""

    var body: some View {
        ChildView(text: text)  // エラー：$がない
    }
}
```

**✅ 正しい方法**：

```swift
struct ParentView: View {
    @State private var text = ""

    var body: some View {
        ChildView(text: $text)  // $でBindingとして渡す
    }
}
```

---

## 演習問題

### 問題：ログインフォームを作る

以下の要件でログインフォームを作成してください：
- メールアドレス入力
- パスワード入力
- ログインボタン（入力が空の場合は無効）
- 入力内容をコンソールに出力

**解答例**：

```swift
struct LoginView: View {
    @State private var email = ""
    @State private var password = ""
    @State private var showAlert = false

    var isFormValid: Bool {
        !email.isEmpty && !password.isEmpty && email.contains("@")
    }

    var body: some View {
        VStack(spacing: 20) {
            Text("ログイン")
                .font(.largeTitle)
                .fontWeight(.bold)

            TextField("メールアドレス", text: $email)
                .textFieldStyle(.roundedBorder)
                .textInputAutocapitalization(.never)
                .keyboardType(.emailAddress)

            SecureField("パスワード", text: $password)
                .textFieldStyle(.roundedBorder)

            Button(action: {
                print("Email: \(email)")
                print("Password: \(password)")
                showAlert = true
            }) {
                Text("ログイン")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(isFormValid ? Color.blue : Color.gray)
                    .cornerRadius(10)
            }
            .disabled(!isFormValid)
        }
        .padding()
        .alert("ログイン成功", isPresented: $showAlert) {
            Button("OK") {}
        }
    }
}

#Preview {
    LoginView()
}
```

---

## 次のステップ

### このガイドで学んだこと

- ✅ SwiftUIの状態管理の基礎
- ✅ @State による状態管理
- ✅ @Binding による親子間のデータ共有
- ✅ @StateObject / @ObservedObject の使い方

### 次に学ぶべきガイド

**次のガイド**：[07-first-app-tutorial.md](./07-first-app-tutorial.md) - 初めてのアプリ作成

---

**前のガイド**：[05-view-layout.md](./05-view-layout.md)

**親ガイド**：[iOS Development - SKILL.md](../../SKILL.md)

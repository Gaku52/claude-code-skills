# State Management - SwiftUI状態管理パターン完全ガイド

## 📋 目次

1. [概要](#概要)
2. [SwiftUIの状態管理の基本原則](#swiftuiの状態管理の基本原則)
3. [@State - ローカル状態管理](#state---ローカル状態管理)
4. [@Binding - 状態の共有](#binding---状態の共有)
5. [@StateObject - 参照型の状態管理](#stateobject---参照型の状態管理)
6. [@ObservedObject - 外部所有の状態](#observedobject---外部所有の状態)
7. [@EnvironmentObject - グローバル状態](#environmentobject---グローバル状態)
8. [@Environment - システム環境値](#environment---システム環境値)
9. [ObservableObject詳細](#observableobject詳細)
10. [状態管理アーキテクチャパターン](#状態管理アーキテクチャパターン)
11. [パフォーマンス最適化](#パフォーマンス最適化)
12. [テスト戦略](#テスト戦略)
13. [よくある問題と解決策](#よくある問題と解決策)

## 概要

SwiftUIの状態管理は、UIとデータの同期を自動的に行う宣言的UIフレームワークの核心部分です。適切な状態管理パターンを選択することで、保守性が高く、パフォーマンスの良いアプリケーションを構築できます。

### このガイドの対象者

- SwiftUI初学者〜中級者
- iOSアプリ開発者
- 状態管理の設計パターンを学びたい方

### 学べること

- 各プロパティラッパーの使い分け
- 効率的な状態管理アーキテクチャ
- パフォーマンス最適化テクニック
- 実践的な設計パターン

## SwiftUIの状態管理の基本原則

### 単一データソース (Single Source of Truth)

**原則:**
状態は常に1箇所で管理し、複数の場所で重複させない。

```swift
// ✅ 良い例: 状態は親で管理
struct ParentView: View {
    @State private var username: String = ""

    var body: some View {
        VStack {
            DisplayNameView(name: username)
            EditNameView(name: $username)
        }
    }
}

struct DisplayNameView: View {
    let name: String  // 読み取り専用
    var body: some View { Text(name) }
}

struct EditNameView: View {
    @Binding var name: String  // 双方向バインディング
    var body: some View { TextField("Name", text: $name) }
}

// ❌ 悪い例: 状態が複数箇所に存在
struct BadParentView: View {
    @State private var username: String = ""
    var body: some View {
        VStack {
            BadChildView() // 独自の@Stateを持つ
        }
    }
}

struct BadChildView: View {
    @State private var username: String = "" // 重複!
    var body: some View { TextField("Name", text: $username) }
}
```

### データフローの方向性

**原則:**
データは親から子へ一方向に流れる。子からの変更は@Bindingまたはコールバックで通知。

```swift
// ✅ 正しいデータフロー
struct TodoListView: View {
    @State private var todos: [Todo] = []

    var body: some View {
        List {
            ForEach(todos) { todo in
                TodoRow(todo: todo, onToggle: { id in
                    toggleTodo(id: id)
                })
            }
        }
    }

    private func toggleTodo(id: UUID) {
        if let index = todos.firstIndex(where: { $0.id == id }) {
            todos[index].isCompleted.toggle()
        }
    }
}

struct TodoRow: View {
    let todo: Todo
    let onToggle: (UUID) -> Void

    var body: some View {
        HStack {
            Text(todo.title)
            Spacer()
            Button(action: { onToggle(todo.id) }) {
                Image(systemName: todo.isCompleted ? "checkmark.circle.fill" : "circle")
            }
        }
    }
}
```

### 状態の所有権

**原則:**
状態を所有するViewが状態の変更に責任を持つ。

```swift
// ✅ 明確な所有権
struct SettingsView: View {
    @StateObject private var viewModel = SettingsViewModel() // この View が所有

    var body: some View {
        Form {
            SettingsRow(setting: viewModel.notificationSetting)
        }
    }
}

// ❌ 曖昧な所有権
struct BadSettingsView: View {
    @ObservedObject var viewModel: SettingsViewModel // 誰が所有?

    var body: some View {
        Form {
            SettingsRow(setting: viewModel.notificationSetting)
        }
    }
}
```

## @State - ローカル状態管理

### 基本概念

`@State`は、View内に閉じた値型(struct, enum, Int, String等)の状態管理に使用します。

**特徴:**
- View所有の状態
- 値型専用
- privateであるべき
- 軽量で高速

### 基本的な使用例

```swift
struct CounterView: View {
    @State private var count: Int = 0

    var body: some View {
        VStack(spacing: 20) {
            Text("Count: \(count)")
                .font(.largeTitle)

            HStack(spacing: 15) {
                Button("Decrement") {
                    count -= 1
                }

                Button("Reset") {
                    count = 0
                }

                Button("Increment") {
                    count += 1
                }
            }
        }
        .padding()
    }
}
```

### 複雑な値型の管理

```swift
struct FormData {
    var username: String = ""
    var email: String = ""
    var age: Int = 0
    var agreedToTerms: Bool = false
}

struct RegistrationView: View {
    @State private var formData = FormData()
    @State private var isSubmitting = false

    var body: some View {
        Form {
            Section("Personal Information") {
                TextField("Username", text: $formData.username)
                TextField("Email", text: $formData.email)
                    .keyboardType(.emailAddress)
                    .textContentType(.emailAddress)

                Stepper("Age: \(formData.age)", value: $formData.age, in: 0...120)
            }

            Section {
                Toggle("I agree to the terms", isOn: $formData.agreedToTerms)
            }

            Section {
                Button("Submit") {
                    submitForm()
                }
                .disabled(!formData.agreedToTerms || isSubmitting)
            }
        }
    }

    private func submitForm() {
        isSubmitting = true
        // API呼び出しなど
        Task {
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            isSubmitting = false
        }
    }
}
```

### @Stateのベストプラクティス

```swift
struct BestPracticesView: View {
    // ✅ private修飾子を付ける
    @State private var isOn = false

    // ✅ デフォルト値を提供
    @State private var text = ""

    // ✅ 値型を使用
    @State private var count: Int = 0
    @State private var settings = UserSettings()

    // ❌ 参照型は@StateObjectを使う
    // @State private var viewModel = ViewModel() // NG!

    // ❌ publicにしない
    // @State var publicState = 0 // NG!

    var body: some View {
        VStack {
            Toggle("Switch", isOn: $isOn)
            TextField("Text", text: $text)
        }
    }
}

struct UserSettings {
    var darkMode = false
    var notifications = true
}
```

### 配列とコレクションの管理

```swift
struct TodoListView: View {
    @State private var todos: [Todo] = [
        Todo(title: "Buy groceries"),
        Todo(title: "Walk the dog"),
        Todo(title: "Read a book")
    ]
    @State private var newTodoTitle = ""

    var body: some View {
        NavigationStack {
            VStack {
                // 新規追加フォーム
                HStack {
                    TextField("New todo", text: $newTodoTitle)
                        .textFieldStyle(.roundedBorder)

                    Button(action: addTodo) {
                        Image(systemName: "plus.circle.fill")
                            .font(.title2)
                    }
                    .disabled(newTodoTitle.isEmpty)
                }
                .padding()

                // Todo一覧
                List {
                    ForEach(todos) { todo in
                        TodoRowView(todo: todo, onToggle: {
                            toggleTodo(id: todo.id)
                        })
                    }
                    .onDelete(perform: deleteTodos)
                }
            }
            .navigationTitle("Todos")
            .toolbar {
                EditButton()
            }
        }
    }

    private func addTodo() {
        let newTodo = Todo(title: newTodoTitle)
        todos.append(newTodo)
        newTodoTitle = ""
    }

    private func toggleTodo(id: UUID) {
        if let index = todos.firstIndex(where: { $0.id == id }) {
            todos[index].isCompleted.toggle()
        }
    }

    private func deleteTodos(at offsets: IndexSet) {
        todos.remove(atOffsets: offsets)
    }
}

struct Todo: Identifiable {
    let id = UUID()
    var title: String
    var isCompleted = false
}

struct TodoRowView: View {
    let todo: Todo
    let onToggle: () -> Void

    var body: some View {
        HStack {
            Button(action: onToggle) {
                Image(systemName: todo.isCompleted ? "checkmark.circle.fill" : "circle")
                    .foregroundColor(todo.isCompleted ? .green : .gray)
            }

            Text(todo.title)
                .strikethrough(todo.isCompleted)
                .foregroundColor(todo.isCompleted ? .secondary : .primary)
        }
    }
}
```

## @Binding - 状態の共有

### 基本概念

`@Binding`は、親Viewが所有する状態への参照を子Viewに渡すために使用します。

**特徴:**
- 双方向データバインディング
- 状態の所有権は持たない
- 親の状態を直接変更可能
- $プレフィックスで取得

### 基本的な使用例

```swift
struct ParentView: View {
    @State private var isOn = false

    var body: some View {
        VStack(spacing: 20) {
            Text("Status: \(isOn ? "ON" : "OFF")")
                .font(.headline)

            // $isOnで@Bindingを渡す
            ToggleControlView(isOn: $isOn)

            // 別のコンポーネントでも同じ状態を共有
            StatusIndicator(isActive: $isOn)
        }
    }
}

struct ToggleControlView: View {
    @Binding var isOn: Bool

    var body: some View {
        Toggle("Control Switch", isOn: $isOn)
            .padding()
    }
}

struct StatusIndicator: View {
    @Binding var isActive: Bool

    var body: some View {
        Circle()
            .fill(isActive ? Color.green : Color.red)
            .frame(width: 50, height: 50)
            .onTapGesture {
                isActive.toggle()
            }
    }
}
```

### フォームでの活用

```swift
struct ProfileEditView: View {
    @State private var profile = UserProfile(
        name: "John Doe",
        email: "john@example.com",
        bio: "iOS Developer"
    )

    var body: some View {
        Form {
            Section("Basic Information") {
                NameField(name: $profile.name)
                EmailField(email: $profile.email)
            }

            Section("About") {
                BioEditor(bio: $profile.bio)
            }

            Section {
                Button("Save") {
                    saveProfile()
                }
            }
        }
    }

    private func saveProfile() {
        // プロファイル保存処理
        print("Saving: \(profile)")
    }
}

struct NameField: View {
    @Binding var name: String

    var body: some View {
        HStack {
            Text("Name")
            TextField("Enter your name", text: $name)
                .multilineTextAlignment(.trailing)
        }
    }
}

struct EmailField: View {
    @Binding var email: String

    var body: some View {
        HStack {
            Text("Email")
            TextField("Enter your email", text: $email)
                .keyboardType(.emailAddress)
                .textContentType(.emailAddress)
                .multilineTextAlignment(.trailing)
        }
    }
}

struct BioEditor: View {
    @Binding var bio: String

    var body: some View {
        VStack(alignment: .leading) {
            Text("Bio")
                .font(.headline)
            TextEditor(text: $bio)
                .frame(height: 100)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
                )
        }
    }
}

struct UserProfile {
    var name: String
    var email: String
    var bio: String
}
```

### カスタムバインディング

```swift
struct CustomBindingView: View {
    @State private var temperatureCelsius: Double = 20.0

    // 計算されたBinding
    private var temperatureFahrenheit: Binding<Double> {
        Binding(
            get: { self.temperatureCelsius * 9/5 + 32 },
            set: { self.temperatureCelsius = ($0 - 32) * 5/9 }
        )
    }

    var body: some View {
        VStack(spacing: 20) {
            Text("Temperature Converter")
                .font(.headline)

            VStack {
                Text("Celsius: \(temperatureCelsius, specifier: "%.1f")°C")
                Slider(value: $temperatureCelsius, in: -40...50)
            }

            VStack {
                Text("Fahrenheit: \(temperatureFahrenheit.wrappedValue, specifier: "%.1f")°F")
                Slider(value: temperatureFahrenheit, in: -40...122)
            }
        }
        .padding()
    }
}
```

### Bindingのバリデーション

```swift
struct ValidatedInputView: View {
    @State private var username: String = ""
    @State private var isValid: Bool = true

    // バリデーション付きBinding
    private var validatedUsername: Binding<String> {
        Binding(
            get: { username },
            set: { newValue in
                username = newValue
                isValid = validateUsername(newValue)
            }
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("Username", text: validatedUsername)
                .textFieldStyle(.roundedBorder)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(isValid ? Color.clear : Color.red, lineWidth: 2)
                )

            if !isValid {
                Text("Username must be 3-20 characters and alphanumeric only")
                    .font(.caption)
                    .foregroundColor(.red)
            }
        }
        .padding()
    }

    private func validateUsername(_ username: String) -> Bool {
        let regex = "^[a-zA-Z0-9]{3,20}$"
        return username.range(of: regex, options: .regularExpression) != nil
    }
}
```

### プレビューでの.constant()使用

```swift
struct ToggleComponentView: View {
    @Binding var isEnabled: Bool

    var body: some View {
        Toggle("Feature Enabled", isEnabled: $isEnabled)
            .padding()
    }
}

#Preview("Enabled State") {
    ToggleComponentView(isEnabled: .constant(true))
}

#Preview("Disabled State") {
    ToggleComponentView(isEnabled: .constant(false))
}

#Preview("Interactive") {
    struct PreviewWrapper: View {
        @State private var isEnabled = false

        var body: some View {
            ToggleComponentView(isEnabled: $isEnabled)
        }
    }

    return PreviewWrapper()
}
```

## @StateObject - 参照型の状態管理

### 基本概念

`@StateObject`は、ObservableObjectに準拠した参照型のライフサイクルを管理します。

**特徴:**
- View所有のObservableObject
- Viewが破棄されるまで保持される
- 初期化時に一度だけ作成
- @Publishedプロパティの変更でView更新

### 基本的な使用例

```swift
class CounterViewModel: ObservableObject {
    @Published var count: Int = 0

    func increment() {
        count += 1
    }

    func decrement() {
        count -= 1
    }

    func reset() {
        count = 0
    }
}

struct CounterViewWithViewModel: View {
    @StateObject private var viewModel = CounterViewModel()

    var body: some View {
        VStack(spacing: 20) {
            Text("Count: \(viewModel.count)")
                .font(.largeTitle)

            HStack(spacing: 15) {
                Button("−") { viewModel.decrement() }
                Button("Reset") { viewModel.reset() }
                Button("+") { viewModel.increment() }
            }
            .buttonStyle(.bordered)
        }
        .padding()
    }
}
```

### ネットワークリクエストを含むViewModel

```swift
struct User: Identifiable, Codable {
    let id: Int
    let name: String
    let email: String
}

class UserListViewModel: ObservableObject {
    @Published var users: [User] = []
    @Published var isLoading = false
    @Published var errorMessage: String?

    func loadUsers() async {
        await MainActor.run {
            isLoading = true
            errorMessage = nil
        }

        do {
            // API呼び出しのシミュレーション
            try await Task.sleep(nanoseconds: 1_000_000_000)

            // 実際のコード例:
            // let url = URL(string: "https://api.example.com/users")!
            // let (data, _) = try await URLSession.shared.data(from: url)
            // let users = try JSONDecoder().decode([User].self, from: data)

            // デモデータ
            let demoUsers = [
                User(id: 1, name: "Alice", email: "alice@example.com"),
                User(id: 2, name: "Bob", email: "bob@example.com"),
                User(id: 3, name: "Charlie", email: "charlie@example.com")
            ]

            await MainActor.run {
                self.users = demoUsers
                self.isLoading = false
            }
        } catch {
            await MainActor.run {
                self.errorMessage = error.localizedDescription
                self.isLoading = false
            }
        }
    }
}

struct UserListView: View {
    @StateObject private var viewModel = UserListViewModel()

    var body: some View {
        NavigationStack {
            Group {
                if viewModel.isLoading {
                    ProgressView("Loading...")
                } else if let errorMessage = viewModel.errorMessage {
                    ContentUnavailableView(
                        "Error",
                        systemImage: "exclamationmark.triangle",
                        description: Text(errorMessage)
                    )
                } else {
                    List(viewModel.users) { user in
                        VStack(alignment: .leading) {
                            Text(user.name)
                                .font(.headline)
                            Text(user.email)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("Users")
            .toolbar {
                Button("Refresh") {
                    Task {
                        await viewModel.loadUsers()
                    }
                }
            }
            .task {
                await viewModel.loadUsers()
            }
        }
    }
}
```

### ViewModelの依存性注入

```swift
protocol UserRepositoryProtocol {
    func fetchUsers() async throws -> [User]
}

class UserRepository: UserRepositoryProtocol {
    func fetchUsers() async throws -> [User] {
        // 実装
        []
    }
}

class MockUserRepository: UserRepositoryProtocol {
    func fetchUsers() async throws -> [User] {
        [
            User(id: 1, name: "Test User", email: "test@example.com")
        ]
    }
}

class DIUserListViewModel: ObservableObject {
    @Published var users: [User] = []
    private let repository: UserRepositoryProtocol

    init(repository: UserRepositoryProtocol = UserRepository()) {
        self.repository = repository
    }

    func loadUsers() async {
        do {
            let fetchedUsers = try await repository.fetchUsers()
            await MainActor.run {
                self.users = fetchedUsers
            }
        } catch {
            print("Error: \(error)")
        }
    }
}

struct DIUserListView: View {
    @StateObject private var viewModel: DIUserListViewModel

    init(repository: UserRepositoryProtocol = UserRepository()) {
        _viewModel = StateObject(wrappedValue: DIUserListViewModel(repository: repository))
    }

    var body: some View {
        List(viewModel.users) { user in
            Text(user.name)
        }
    }
}

#Preview("Production") {
    DIUserListView(repository: UserRepository())
}

#Preview("Mock") {
    DIUserListView(repository: MockUserRepository())
}
```

## @ObservedObject - 外部所有の状態

### 基本概念

`@ObservedObject`は、親から受け取ったObservableObjectを監視します。

**@StateObjectとの違い:**
- 所有権を持たない
- Viewの再作成時に維持されない可能性がある
- 親が管理するオブジェクトに使用

### 適切な使用例

```swift
class AppSettings: ObservableObject {
    @Published var isDarkMode = false
    @Published var notificationsEnabled = true
    @Published var fontSize: Double = 16.0
}

struct SettingsContainerView: View {
    @StateObject private var settings = AppSettings() // 親が所有

    var body: some View {
        NavigationStack {
            List {
                NavigationLink("Appearance") {
                    AppearanceSettingsView(settings: settings)
                }
                NavigationLink("Notifications") {
                    NotificationSettingsView(settings: settings)
                }
            }
            .navigationTitle("Settings")
        }
    }
}

struct AppearanceSettingsView: View {
    @ObservedObject var settings: AppSettings // 親から受け取る

    var body: some View {
        Form {
            Toggle("Dark Mode", isOn: $settings.isDarkMode)

            VStack(alignment: .leading) {
                Text("Font Size: \(Int(settings.fontSize))")
                Slider(value: $settings.fontSize, in: 12...24, step: 1)
            }
        }
        .navigationTitle("Appearance")
    }
}

struct NotificationSettingsView: View {
    @ObservedObject var settings: AppSettings // 親から受け取る

    var body: some View {
        Form {
            Toggle("Enable Notifications", isOn: $settings.notificationsEnabled)
        }
        .navigationTitle("Notifications")
    }
}
```

### 誤った使用例と修正

```swift
// ❌ 誤った使用: 子Viewで@ObservedObjectを初期化
struct BadChildView: View {
    @ObservedObject var viewModel = ViewModel() // Viewが再作成されるたびに新しいインスタンスが作られる

    var body: some View {
        Text("Count: \(viewModel.count)")
    }
}

// ✅ 修正1: @StateObjectを使用
struct GoodChildView1: View {
    @StateObject private var viewModel = ViewModel() // View所有

    var body: some View {
        Text("Count: \(viewModel.count)")
    }
}

// ✅ 修正2: 親から受け取る
struct GoodParentView: View {
    @StateObject private var viewModel = ViewModel() // 親が所有

    var body: some View {
        GoodChildView2(viewModel: viewModel) // 子に渡す
    }
}

struct GoodChildView2: View {
    @ObservedObject var viewModel: ViewModel // 親から受け取る

    var body: some View {
        Text("Count: \(viewModel.count)")
    }
}

class ViewModel: ObservableObject {
    @Published var count = 0
}
```

## @EnvironmentObject - グローバル状態

### 基本概念

`@EnvironmentObject`は、View階層全体で共有される状態を管理します。

**使用場面:**
- アプリ全体で共有される認証状態
- テーマ設定
- ユーザー設定
- 深い階層への状態伝播

### 基本的な使用例

```swift
class AuthenticationManager: ObservableObject {
    @Published var isAuthenticated = false
    @Published var currentUser: User?

    func login(username: String, password: String) async {
        // ログイン処理のシミュレーション
        try? await Task.sleep(nanoseconds: 1_000_000_000)

        await MainActor.run {
            self.isAuthenticated = true
            self.currentUser = User(id: 1, name: username, email: "\(username)@example.com")
        }
    }

    func logout() {
        isAuthenticated = false
        currentUser = nil
    }
}

@main
struct MyApp: App {
    @StateObject private var authManager = AuthenticationManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(authManager) // ここで注入
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var authManager: AuthenticationManager

    var body: some View {
        if authManager.isAuthenticated {
            MainTabView()
        } else {
            LoginView()
        }
    }
}

struct LoginView: View {
    @EnvironmentObject var authManager: AuthenticationManager
    @State private var username = ""
    @State private var password = ""

    var body: some View {
        VStack(spacing: 20) {
            TextField("Username", text: $username)
                .textFieldStyle(.roundedBorder)

            SecureField("Password", text: $password)
                .textFieldStyle(.roundedBorder)

            Button("Login") {
                Task {
                    await authManager.login(username: username, password: password)
                }
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }
}

struct MainTabView: View {
    var body: some View {
        TabView {
            HomeView()
                .tabItem {
                    Label("Home", systemImage: "house")
                }

            ProfileView()
                .tabItem {
                    Label("Profile", systemImage: "person")
                }
        }
    }
}

struct HomeView: View {
    @EnvironmentObject var authManager: AuthenticationManager

    var body: some View {
        VStack {
            Text("Welcome, \(authManager.currentUser?.name ?? "Guest")!")
            Button("Logout") {
                authManager.logout()
            }
        }
    }
}

struct ProfileView: View {
    @EnvironmentObject var authManager: AuthenticationManager

    var body: some View {
        VStack {
            Text("Profile")
            Text("Email: \(authManager.currentUser?.email ?? "")")
            Button("Logout") {
                authManager.logout()
            }
        }
    }
}
```

### プレビューでの注入

```swift
struct SomeView: View {
    @EnvironmentObject var authManager: AuthenticationManager

    var body: some View {
        Text("User: \(authManager.currentUser?.name ?? "None")")
    }
}

#Preview("Not Logged In") {
    SomeView()
        .environmentObject(AuthenticationManager())
}

#Preview("Logged In") {
    let manager = AuthenticationManager()
    manager.isAuthenticated = true
    manager.currentUser = User(id: 1, name: "John Doe", email: "john@example.com")

    return SomeView()
        .environmentObject(manager)
}
```

### 複数のEnvironmentObjectの管理

```swift
class ThemeManager: ObservableObject {
    @Published var colorScheme: ColorScheme = .light
    @Published var accentColor: Color = .blue
}

class SettingsManager: ObservableObject {
    @Published var language: String = "en"
    @Published var region: String = "US"
}

@main
struct MultiEnvironmentApp: App {
    @StateObject private var authManager = AuthenticationManager()
    @StateObject private var themeManager = ThemeManager()
    @StateObject private var settingsManager = SettingsManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(authManager)
                .environmentObject(themeManager)
                .environmentObject(settingsManager)
        }
    }
}

struct DetailView: View {
    @EnvironmentObject var authManager: AuthenticationManager
    @EnvironmentObject var themeManager: ThemeManager
    @EnvironmentObject var settingsManager: SettingsManager

    var body: some View {
        VStack {
            Text("User: \(authManager.currentUser?.name ?? "Guest")")
            Text("Theme: \(themeManager.colorScheme == .dark ? "Dark" : "Light")")
            Text("Language: \(settingsManager.language)")
        }
        .foregroundColor(themeManager.accentColor)
    }
}
```

## @Environment - システム環境値

### 基本概念

`@Environment`は、SwiftUIが提供するシステム環境値にアクセスします。

**主要な環境値:**
- colorScheme: ライト/ダークモード
- dismiss: View の dismiss
- scenePhase: アプリのライフサイクル
- openURL: URLを開く

### 基本的な使用例

```swift
struct EnvironmentExamplesView: View {
    @Environment(\.colorScheme) var colorScheme
    @Environment(\.dismiss) var dismiss
    @Environment(\.openURL) var openURL

    var body: some View {
        VStack(spacing: 20) {
            Text("Current theme: \(colorScheme == .dark ? "Dark" : "Light")")

            Button("Open Website") {
                if let url = URL(string: "https://www.apple.com") {
                    openURL(url)
                }
            }

            Button("Close") {
                dismiss()
            }
        }
    }
}
```

### カスタム環境値の定義

```swift
// カスタム環境キーの定義
private struct UserPreferencesKey: EnvironmentKey {
    static let defaultValue = UserPreferences()
}

extension EnvironmentValues {
    var userPreferences: UserPreferences {
        get { self[UserPreferencesKey.self] }
        set { self[UserPreferencesKey.self] = newValue }
    }
}

struct UserPreferences {
    var showTips: Bool = true
    var animationsEnabled: Bool = true
}

// Viewでの使用
struct PreferencesView: View {
    @Environment(\.userPreferences) var preferences

    var body: some View {
        VStack {
            if preferences.showTips {
                Text("Tip: Swipe to navigate")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Text("Content")
                .animation(preferences.animationsEnabled ? .default : .none, value: UUID())
        }
    }
}

// 親Viewでの設定
struct RootView: View {
    @State private var preferences = UserPreferences()

    var body: some View {
        PreferencesView()
            .environment(\.userPreferences, preferences)
    }
}
```

(続く... 文字数制限のため、残りの章は次のセクションで提供します)

## ObservableObject詳細

### @Publishedの動作

```swift
class DataManager: ObservableObject {
    // ✅ @Publishedは変更を自動的に通知
    @Published var items: [String] = []
    @Published var isLoading: Bool = false

    // ✅ 計算プロパティは@Publishedにできない（必要もない）
    var itemCount: Int {
        items.count
    }

    // ✅ willSetで変更前にUI更新をトリガー
    @Published var selectedItem: String? {
        willSet {
            print("Changing from \(selectedItem ?? "none") to \(newValue ?? "none")")
        }
    }
}
```

### objectWillChangeの手動制御

```swift
class ManualUpdateManager: ObservableObject {
    // 手動で通知を制御
    var internalValue: Int = 0 {
        didSet {
            // 条件付きで更新を通知
            if internalValue % 10 == 0 {
                objectWillChange.send()
            }
        }
    }

    func updateValue(_ value: Int) {
        internalValue = value
        // 必要な時だけUI更新
        objectWillChange.send()
    }
}
```

## 状態管理アーキテクチャパターン

### MVVM (Model-View-ViewModel)

```swift
// Model
struct Article: Identifiable, Codable {
    let id: UUID
    var title: String
    var content: String
    var publishedAt: Date
}

// ViewModel
class ArticleListViewModel: ObservableObject {
    @Published var articles: [Article] = []
    @Published var isLoading = false
    @Published var error: Error?

    private let repository: ArticleRepository

    init(repository: ArticleRepository = ArticleRepository()) {
        self.repository = repository
    }

    @MainActor
    func loadArticles() async {
        isLoading = true
        defer { isLoading = false }

        do {
            articles = try await repository.fetchArticles()
        } catch {
            self.error = error
        }
    }

    @MainActor
    func deleteArticle(at indexSet: IndexSet) {
        articles.remove(atOffsets: indexSet)
        // API呼び出しなど
    }
}

// View
struct ArticleListView: View {
    @StateObject private var viewModel = ArticleListViewModel()

    var body: some View {
        NavigationStack {
            List {
                ForEach(viewModel.articles) { article in
                    ArticleRow(article: article)
                }
                .onDelete { indexSet in
                    Task {
                        await viewModel.deleteArticle(at: indexSet)
                    }
                }
            }
            .navigationTitle("Articles")
            .overlay {
                if viewModel.isLoading {
                    ProgressView()
                }
            }
            .task {
                await viewModel.loadArticles()
            }
        }
    }
}

struct ArticleRow: View {
    let article: Article

    var body: some View {
        VStack(alignment: .leading) {
            Text(article.title)
                .font(.headline)
            Text(article.publishedAt, style: .date)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}

// Repository (データ層)
class ArticleRepository {
    func fetchArticles() async throws -> [Article] {
        // API呼び出しなど
        []
    }
}
```

### Unidirectional Data Flow (Redux-like)

```swift
// State
struct AppState {
    var counter: Int = 0
    var todos: [Todo] = []
}

// Action
enum AppAction {
    case increment
    case decrement
    case addTodo(String)
    case toggleTodo(UUID)
    case deleteTodo(UUID)
}

// Reducer
func appReducer(state: inout AppState, action: AppAction) {
    switch action {
    case .increment:
        state.counter += 1
    case .decrement:
        state.counter -= 1
    case .addTodo(let title):
        let todo = Todo(title: title)
        state.todos.append(todo)
    case .toggleTodo(let id):
        if let index = state.todos.firstIndex(where: { $0.id == id }) {
            state.todos[index].isCompleted.toggle()
        }
    case .deleteTodo(let id):
        state.todos.removeAll { $0.id == id }
    }
}

// Store
class Store: ObservableObject {
    @Published private(set) var state = AppState()

    func dispatch(_ action: AppAction) {
        appReducer(state: &state, action: action)
    }
}

// View
struct ReduxStyleView: View {
    @StateObject private var store = Store()

    var body: some View {
        VStack {
            Text("Counter: \(store.state.counter)")

            HStack {
                Button("−") { store.dispatch(.decrement) }
                Button("+") { store.dispatch(.increment) }
            }

            List(store.state.todos) { todo in
                HStack {
                    Text(todo.title)
                    Spacer()
                    Button(action: { store.dispatch(.toggleTodo(todo.id)) }) {
                        Image(systemName: todo.isCompleted ? "checkmark.circle.fill" : "circle")
                    }
                }
            }
        }
    }
}
```

## パフォーマンス最適化

### 不要な再描画を避ける

```swift
// ✅ Equatable準拠で再描画を制御
struct OptimizedView: View, Equatable {
    let data: ExpensiveData

    var body: some View {
        // 重い描画処理
        ComplexVisualization(data: data)
    }

    static func == (lhs: OptimizedView, rhs: OptimizedView) -> Bool {
        lhs.data.id == rhs.data.id
    }
}

struct ParentView: View {
    @State private var counter = 0
    let data: ExpensiveData

    var body: some View {
        VStack {
            Button("Increment: \(counter)") {
                counter += 1
            }

            // dataが変わらない限り再描画されない
            OptimizedView(data: data)
                .equatable()
        }
    }
}

struct ExpensiveData: Identifiable {
    let id: UUID
    let values: [Double]
}

struct ComplexVisualization: View {
    let data: ExpensiveData
    var body: some View { Text("Chart") }
}
```

### @Publishedの最適化

```swift
class OptimizedViewModel: ObservableObject {
    // ✅ UIに影響するプロパティのみ@Published
    @Published var displayText: String = ""
    @Published var isError: Bool = false

    // ❌ 頻繁に変わる内部状態は@Publishedにしない
    private var internalCounter = 0

    func updateData() {
        internalCounter += 1

        // 10回に1回だけUI更新
        if internalCounter % 10 == 0 {
            displayText = "Count: \(internalCounter)"
        }
    }
}
```

## テスト戦略

### ViewModelのユニットテスト

```swift
@testable import MyApp
import XCTest

final class CounterViewModelTests: XCTestCase {
    var viewModel: CounterViewModel!

    override func setUp() {
        super.setUp()
        viewModel = CounterViewModel()
    }

    func testInitialState() {
        XCTAssertEqual(viewModel.count, 0)
    }

    func testIncrement() {
        viewModel.increment()
        XCTAssertEqual(viewModel.count, 1)
    }

    func testDecrement() {
        viewModel.increment()
        viewModel.increment()
        viewModel.decrement()
        XCTAssertEqual(viewModel.count, 1)
    }

    func testReset() {
        viewModel.increment()
        viewModel.reset()
        XCTAssertEqual(viewModel.count, 0)
    }
}
```

## よくある問題と解決策

### 問題1: @StateObjectと@ObservedObjectの混同

```swift
// ❌ 誤った使用
struct BadView: View {
    @ObservedObject var viewModel = ViewModel() // Viewが再作成されるたびに新規作成
    var body: some View { Text("\(viewModel.count)") }
}

// ✅ 正しい使用
struct GoodView: View {
    @StateObject private var viewModel = ViewModel() // View所有
    var body: some View { Text("\(viewModel.count)") }
}
```

### 問題2: @EnvironmentObjectの注入忘れ

```swift
// ❌ 実行時エラーが発生
struct ProblemView: View {
    @EnvironmentObject var authManager: AuthenticationManager
    var body: some View { Text("Hello") }
}

#Preview {
    ProblemView() // Fatal error: No ObservableObject of type AuthenticationManager found.
}

// ✅ 正しいプレビュー
#Preview {
    ProblemView()
        .environmentObject(AuthenticationManager())
}
```

### 問題3: @Publishedの過剰使用

```swift
// ❌ 全てのプロパティを@Publishedにしている
class BadViewModel: ObservableObject {
    @Published var tempValue1 = 0
    @Published var tempValue2 = 0
    @Published var displayText = ""
}

// ✅ 必要最小限のプロパティのみ@Published
class GoodViewModel: ObservableObject {
    private var tempValue1 = 0 // 内部状態
    private var tempValue2 = 0 // 内部状態
    @Published var displayText = "" // UI表示用のみ
}
```

---

**関連ガイド:**
- [02-layout-navigation.md](./02-layout-navigation.md) - レイアウトとナビゲーション
- [03-performance-best-practices.md](./03-performance-best-practices.md) - パフォーマンス最適化

**関連Skills:**
- [ios-development](../../ios-development/SKILL.md) - iOS開発全般
- [testing-strategy](../../testing-strategy/SKILL.md) - テスト戦略

**参考資料:**
- [SwiftUI Documentation - State and Data Flow](https://developer.apple.com/documentation/swiftui/state-and-data-flow)
- [WWDC - Data Essentials in SwiftUI](https://developer.apple.com/videos/play/wwdc2020/10040/)

**更新履歴:**
- 2025-12-30: 初版作成

# iOS トラブルシューティング

## 目次

1. [概要](#概要)
2. [ビルド・Xcodeエラー](#ビルドxcodeエラー)
3. [SwiftUI エラー](#swiftuiエラー)
4. [状態管理エラー](#状態管理エラー)
5. [ナビゲーションエラー](#ナビゲーションエラー)
6. [データ永続化エラー](#データ永続化エラー)
7. [ネットワークエラー](#ネットワークエラー)
8. [パフォーマンス問題](#パフォーマンス問題)

---

## 概要

このガイドは、iOS開発で頻繁に遭遇するエラーと解決策をまとめたトラブルシューティングデータベースです。

**収録エラー数:** 25個

**対象バージョン:** iOS 15.0+, Xcode 14.0+

---

## ビルド・Xcodeエラー

### ❌ エラー1: Build Failed - No such module

```
No such module 'SwiftUI'
```

**原因:**
- iOSデプロイメントターゲットが低すぎる
- モジュールがインポートされていない

**解決策:**

```swift
// ✅ プロジェクト設定を確認
// Xcode > Project > Deployment Info
// iOS Deployment Target: 15.0 以上

// ✅ 正しいインポート
import SwiftUI
import Combine
```

**Package.swift（SPM使用時）:**

```swift
// ✅ Swift Package Manager設定
let package = Package(
    name: "MyApp",
    platforms: [
        .iOS(.v15)
    ],
    products: [
        .library(
            name: "MyApp",
            targets: ["MyApp"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/Alamofire/Alamofire.git", from: "5.8.0")
    ],
    targets: [
        .target(
            name: "MyApp",
            dependencies: ["Alamofire"]
        )
    ]
)
```

---

### ❌ エラー2: Provisioning Profile Error

```
No profiles for 'com.example.MyApp' were found
```

**原因:**
- プロビジョニングプロファイルが期限切れ
- Bundle Identifierが一致していない

**解決策:**

```bash
# ✅ 証明書とプロファイルを更新
# Xcode > Preferences > Accounts > Download Manual Profiles

# または自動管理に切り替え
# Xcode > Project > Signing & Capabilities
# Automatically manage signing: ON
```

**手動管理の場合:**

1. Apple Developer にログイン
2. Certificates, Identifiers & Profiles
3. Profilesから新しいプロファイル作成
4. Xcodeでダウンロード

---

### ❌ エラー3: Simulator Not Found

```
Unable to boot the Simulator
```

**原因:**
- Simulatorが壊れている
- ディスク容量不足

**解決策:**

```bash
# ✅ Simulatorをリセット
xcrun simctl erase all

# 特定のデバイスのみリセット
xcrun simctl list devices
xcrun simctl erase <DEVICE_ID>

# Simulatorキャッシュを削除
rm -rf ~/Library/Developer/Xcode/DerivedData
rm -rf ~/Library/Caches/com.apple.dt.Xcode

# Xcodeを再起動
killall Xcode
```

**新しいSimulatorを作成:**

```bash
# iPhone 15 Pro Simulatorを作成
xcrun simctl create "iPhone 15 Pro" "iPhone 15 Pro"
```

---

### ❌ エラー4: Command PhaseScriptExecution failed

```
Command PhaseScriptExecution failed with a nonzero exit code
```

**原因:**
- Build Phaseのスクリプトエラー
- SwiftLintエラー

**解決策:**

```bash
# ✅ SwiftLintエラーを確認
if which swiftlint >/dev/null; then
  swiftlint
else
  echo "warning: SwiftLint not installed"
fi

# SwiftLintを無効化（一時的）
# Xcode > Project > Build Phases > Run Script
# スクリプトをコメントアウト
```

**.swiftlint.yml:**

```yaml
# ✅ SwiftLint設定
disabled_rules:
  - line_length
  - trailing_whitespace

opt_in_rules:
  - empty_count
  - closure_spacing

included:
  - Sources

excluded:
  - Pods
  - DerivedData
```

---

### ❌ エラー5: The app bundle could not be found

```
The application bundle could not be found at the path provided
```

**原因:**
- DerivedDataが破損している

**解決策:**

```bash
# ✅ DerivedDataを削除
rm -rf ~/Library/Developer/Xcode/DerivedData

# Xcodeでクリーン
# Product > Clean Build Folder (Shift + Cmd + K)

# 再ビルド
# Product > Build (Cmd + B)
```

---

## SwiftUI エラー

### ❌ エラー6: SwiftUI Preview Crashed

```
SwiftUI Preview crashed
```

**原因:**
- プレビュー用のダミーデータがない
- 環境オブジェクトが設定されていない

**解決策:**

```swift
// ❌ 環境オブジェクトがない
struct ContentView: View {
    @EnvironmentObject var viewModel: AppViewModel

    var body: some View {
        Text("Hello")
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()  // エラー
    }
}

// ✅ プレビューで環境オブジェクトを提供
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .environmentObject(AppViewModel())
    }
}

// ✅ プレビュー用のモックデータ
struct UserView_Previews: PreviewProvider {
    static var previews: some View {
        UserView(user: User.preview)
    }
}

extension User {
    static var preview: User {
        User(id: 1, name: "John Doe", email: "john@example.com")
    }
}
```

**プレビューをリフレッシュ:**

```bash
# Option + Cmd + P でプレビューを再開
# または
# Editor > Canvas > Resume Preview
```

---

### ❌ エラー7: Cannot convert value of type 'String' to expected argument type 'Binding<String>'

```
Cannot convert value of type 'String' to expected argument type 'Binding<String>'
```

**原因:**
- `@State`または`@Binding`が不足している

**解決策:**

```swift
// ❌ Bindingが必要なのにStringを渡している
struct SearchBar: View {
    var searchText: String  // ❌

    var body: some View {
        TextField("Search", text: searchText)  // エラー
    }
}

// ✅ Bindingを使用
struct SearchBar: View {
    @Binding var searchText: String

    var body: some View {
        TextField("Search", text: $searchText)
    }
}

// ✅ 親ビュー
struct ParentView: View {
    @State private var searchText = ""

    var body: some View {
        SearchBar(searchText: $searchText)
    }
}
```

---

### ❌ エラー8: Type 'SomeView' does not conform to protocol 'View'

```
Type 'ContentView' does not conform to protocol 'View'
```

**原因:**
- `body`プロパティが実装されていない
- `body`の戻り値の型が間違っている

**解決策:**

```swift
// ❌ bodyがない
struct ContentView: View {
    // エラー
}

// ✅ bodyを実装
struct ContentView: View {
    var body: some View {
        Text("Hello, World!")
    }
}

// ❌ bodyの戻り値が間違っている
struct ContentView: View {
    var body: String {  // ❌
        "Hello"
    }
}

// ✅ some Viewを返す
struct ContentView: View {
    var body: some View {
        Text("Hello")
    }
}
```

---

## 状態管理エラー

### ❌ エラー9: @State not updating UI

**原因:**
- `@State`を構造体のプロパティとして使っている
- ビュー更新のタイミングが間違っている

**解決策:**

```swift
// ❌ @Stateが機能しない
struct ContentView: View {
    let count = 0  // ❌ let

    var body: some View {
        Button("Count: \(count)") {
            count += 1  // エラー
        }
    }
}

// ✅ @Stateを使用
struct ContentView: View {
    @State private var count = 0

    var body: some View {
        Button("Count: \(count)") {
            count += 1
        }
    }
}

// ❌ クラスのプロパティが更新されない
class User {
    var name: String = "John"
}

struct ProfileView: View {
    @State private var user = User()

    var body: some View {
        Button("Change Name") {
            user.name = "Jane"  // UIが更新されない
        }
    }
}

// ✅ @StateObjectまたは構造体を使用
@MainActor
class UserViewModel: ObservableObject {
    @Published var name: String = "John"
}

struct ProfileView: View {
    @StateObject private var viewModel = UserViewModel()

    var body: some View {
        Button("Change Name") {
            viewModel.name = "Jane"  // UIが更新される
        }
    }
}
```

---

### ❌ エラー10: Published property wrapper can only be applied to classes

```
@Published is only available on properties of classes
```

**原因:**
- 構造体で`@Published`を使用している

**解決策:**

```swift
// ❌ 構造体で@Published
struct User {
    @Published var name: String  // エラー
}

// ✅ クラスで@Published
@MainActor
class UserViewModel: ObservableObject {
    @Published var name: String = ""
    @Published var email: String = ""
    @Published var isLoading: Bool = false
}

// ✅ 使用例
struct UserProfileView: View {
    @StateObject private var viewModel = UserViewModel()

    var body: some View {
        Form {
            TextField("Name", text: $viewModel.name)
            TextField("Email", text: $viewModel.email)

            if viewModel.isLoading {
                ProgressView()
            }
        }
    }
}
```

---

### ❌ エラー11: EnvironmentObject not found

```
Fatal error: No ObservableObject of type AppSettings found.
```

**原因:**
- `@EnvironmentObject`が親ビューで提供されていない

**解決策:**

```swift
// ❌ EnvironmentObjectが提供されていない
@main
struct MyApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()  // AppSettingsが提供されていない
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var settings: AppSettings  // エラー

    var body: some View {
        Text("Theme: \(settings.theme)")
    }
}

// ✅ EnvironmentObjectを提供
@main
struct MyApp: App {
    @StateObject private var settings = AppSettings()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(settings)
        }
    }
}

@MainActor
class AppSettings: ObservableObject {
    @Published var theme: String = "light"
}
```

---

### ❌ エラー12: Modifying state during view update

```
Modifying state during view update, this will cause undefined behavior.
```

**原因:**
- ビューの`body`内で状態を更新している

**解決策:**

```swift
// ❌ body内で状態更新
struct ContentView: View {
    @State private var count = 0

    var body: some View {
        count += 1  // ❌ エラー
        return Text("Count: \(count)")
    }
}

// ✅ onAppear や ボタンアクションで更新
struct ContentView: View {
    @State private var count = 0

    var body: some View {
        VStack {
            Text("Count: \(count)")

            Button("Increment") {
                count += 1
            }
        }
        .onAppear {
            // 初回のみ実行
            count = 0
        }
    }
}

// ✅ Taskで非同期更新
struct UserListView: View {
    @State private var users: [User] = []

    var body: some View {
        List(users) { user in
            Text(user.name)
        }
        .task {
            users = await fetchUsers()
        }
    }
}
```

---

## ナビゲーションエラー

### ❌ エラー13: NavigationLink not working

**原因:**
- NavigationViewで囲まれていない
- iOS 16+ で古いNavigationView APIを使用

**解決策:**

```swift
// ❌ NavigationViewがない
struct ContentView: View {
    var body: some View {
        NavigationLink("Go to Detail", destination: DetailView())  // 動作しない
    }
}

// ✅ NavigationViewで囲む（iOS 15以下）
struct ContentView: View {
    var body: some View {
        NavigationView {
            NavigationLink("Go to Detail", destination: DetailView())
        }
    }
}

// ✅ NavigationStack（iOS 16+、推奨）
struct ContentView: View {
    var body: some View {
        NavigationStack {
            NavigationLink("Go to Detail", destination: DetailView())
        }
    }
}

// ✅ NavigationPath で型安全なナビゲーション
struct ContentView: View {
    @State private var path = NavigationPath()

    var body: some View {
        NavigationStack(path: $path) {
            List {
                Button("Go to User") {
                    path.append(User(id: 1, name: "John"))
                }
            }
            .navigationDestination(for: User.self) { user in
                UserDetailView(user: user)
            }
        }
    }
}
```

---

### ❌ エラー14: Sheet not dismissing

**原因:**
- `@Environment(\.dismiss)`が使えない（iOS 14以下）
- Bindingが正しく渡されていない

**解決策:**

```swift
// ❌ iOS 14以下でdismissが使えない
struct DetailView: View {
    @Environment(\.dismiss) var dismiss  // iOS 15+

    var body: some View {
        Button("Close") {
            dismiss()
        }
    }
}

// ✅ iOS 14互換性のある方法
struct DetailView: View {
    @Environment(\.presentationMode) var presentationMode

    var body: some View {
        Button("Close") {
            presentationMode.wrappedValue.dismiss()
        }
    }
}

// ✅ Bindingで制御
struct ContentView: View {
    @State private var isPresented = false

    var body: some View {
        Button("Show Sheet") {
            isPresented = true
        }
        .sheet(isPresented: $isPresented) {
            DetailView(isPresented: $isPresented)
        }
    }
}

struct DetailView: View {
    @Binding var isPresented: Bool

    var body: some View {
        Button("Close") {
            isPresented = false
        }
    }
}
```

---

### ❌ エラー15: TabView selection not working

**原因:**
- `@State`のタグがTabのタグと一致していない

**解決策:**

```swift
// ❌ タグが一致していない
struct ContentView: View {
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            HomeView()
                .tabItem {
                    Label("Home", systemImage: "house")
                }
                .tag("home")  // ❌ Intではなくstring

            ProfileView()
                .tabItem {
                    Label("Profile", systemImage: "person")
                }
                .tag("profile")
        }
    }
}

// ✅ タグの型を統一
struct ContentView: View {
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            HomeView()
                .tabItem {
                    Label("Home", systemImage: "house")
                }
                .tag(0)

            ProfileView()
                .tabItem {
                    Label("Profile", systemImage: "person")
                }
                .tag(1)
        }
    }
}

// ✅ Enumを使用（推奨）
enum Tab {
    case home, profile, settings
}

struct ContentView: View {
    @State private var selectedTab = Tab.home

    var body: some View {
        TabView(selection: $selectedTab) {
            HomeView()
                .tabItem {
                    Label("Home", systemImage: "house")
                }
                .tag(Tab.home)

            ProfileView()
                .tabItem {
                    Label("Profile", systemImage: "person")
                }
                .tag(Tab.profile)
        }
    }
}
```

---

## データ永続化エラー

### ❌ エラー16: UserDefaults not persisting

**原因:**
- 値が保存されていない
- キーが間違っている

**解決策:**

```swift
// ❌ 保存されない
class SettingsManager {
    func saveTheme(_ theme: String) {
        UserDefaults.standard.setValue(theme, forKey: "theme")
        // synchronize()は不要（自動で同期される）
    }
}

// ✅ @AppStorageを使用（SwiftUI）
struct SettingsView: View {
    @AppStorage("theme") private var theme: String = "light"

    var body: some View {
        Picker("Theme", selection: $theme) {
            Text("Light").tag("light")
            Text("Dark").tag("dark")
        }
    }
}

// ✅ カスタムプロパティラッパー
@propertyWrapper
struct UserDefault<T> {
    let key: String
    let defaultValue: T

    var wrappedValue: T {
        get {
            UserDefaults.standard.object(forKey: key) as? T ?? defaultValue
        }
        set {
            UserDefaults.standard.set(newValue, forKey: key)
        }
    }
}

class Settings {
    @UserDefault(key: "username", defaultValue: "")
    static var username: String

    @UserDefault(key: "isLoggedIn", defaultValue: false)
    static var isLoggedIn: Bool
}
```

---

### ❌ エラー17: Keychain access error

```
OSStatus -25300: errSecItemNotFound
```

**原因:**
- Keychainに保存されていない項目にアクセスしている

**解決策:**

```swift
// ✅ Keychainラッパー
import Security

class KeychainManager {
    static let shared = KeychainManager()

    func save(key: String, data: Data) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: data
        ]

        // 既存のアイテムを削除
        SecItemDelete(query as CFDictionary)

        // 新しいアイテムを追加
        let status = SecItemAdd(query as CFDictionary, nil)

        guard status == errSecSuccess else {
            throw KeychainError.unableToSave
        }
    }

    func load(key: String) throws -> Data {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess else {
            throw KeychainError.itemNotFound
        }

        guard let data = result as? Data else {
            throw KeychainError.invalidData
        }

        return data
    }

    func delete(key: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]

        let status = SecItemDelete(query as CFDictionary)

        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.unableToDelete
        }
    }
}

enum KeychainError: Error {
    case unableToSave
    case itemNotFound
    case invalidData
    case unableToDelete
}

// ✅ 使用例
struct LoginViewModel {
    func saveToken(_ token: String) {
        guard let data = token.data(using: .utf8) else { return }

        do {
            try KeychainManager.shared.save(key: "authToken", data: data)
        } catch {
            print("Failed to save token: \(error)")
        }
    }

    func loadToken() -> String? {
        do {
            let data = try KeychainManager.shared.load(key: "authToken")
            return String(data: data, encoding: .utf8)
        } catch {
            print("Failed to load token: \(error)")
            return nil
        }
    }
}
```

---

### ❌ エラー18: Core Data crash on save

```
NSInvalidArgumentException: Illegal attempt to establish a relationship
```

**原因:**
- リレーションシップの設定が間違っている

**解決策:**

```swift
// ✅ Core Dataスタック
import CoreData

class PersistenceController {
    static let shared = PersistenceController()

    let container: NSPersistentContainer

    init() {
        container = NSPersistentContainer(name: "Model")

        container.loadPersistentStores { description, error in
            if let error = error {
                fatalError("Core Data failed to load: \(error.localizedDescription)")
            }
        }

        container.viewContext.automaticallyMergesChangesFromParent = true
    }

    func save() {
        let context = container.viewContext

        if context.hasChanges {
            do {
                try context.save()
            } catch {
                print("Failed to save context: \(error)")
            }
        }
    }
}

// ✅ SwiftUIで使用
@main
struct MyApp: App {
    let persistenceController = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistenceController.container.viewContext)
        }
    }
}

struct ContentView: View {
    @Environment(\.managedObjectContext) private var viewContext

    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \Item.timestamp, ascending: false)],
        animation: .default
    )
    private var items: FetchedResults<Item>

    var body: some View {
        List {
            ForEach(items) { item in
                Text(item.name ?? "")
            }
        }
    }

    func addItem() {
        let newItem = Item(context: viewContext)
        newItem.timestamp = Date()
        newItem.name = "New Item"

        PersistenceController.shared.save()
    }
}
```

---

## ネットワークエラー

### ❌ エラー19: URLSession data task not executing

**原因:**
- タスクが`resume()`されていない

**解決策:**

```swift
// ❌ resume()を呼んでいない
func fetchData() {
    let url = URL(string: "https://api.example.com/users")!

    URLSession.shared.dataTask(with: url) { data, response, error in
        // タスクが実行されない
    }
    // ❌ resume()を忘れている
}

// ✅ resume()を呼ぶ
func fetchData() {
    let url = URL(string: "https://api.example.com/users")!

    let task = URLSession.shared.dataTask(with: url) { data, response, error in
        if let error = error {
            print("Error: \(error)")
            return
        }

        guard let data = data else { return }
        // データ処理
    }

    task.resume()  // ✅ タスクを開始
}

// ✅ async/await（iOS 15+、推奨）
func fetchUsers() async throws -> [User] {
    let url = URL(string: "https://api.example.com/users")!

    let (data, response) = try await URLSession.shared.data(from: url)

    guard let httpResponse = response as? HTTPURLResponse,
          httpResponse.statusCode == 200 else {
        throw NetworkError.invalidResponse
    }

    let users = try JSONDecoder().decode([User].self, from: data)
    return users
}

// ✅ SwiftUIで使用
struct UserListView: View {
    @State private var users: [User] = []
    @State private var isLoading = false
    @State private var errorMessage: String?

    var body: some View {
        List(users) { user in
            Text(user.name)
        }
        .task {
            isLoading = true
            defer { isLoading = false }

            do {
                users = try await fetchUsers()
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }
}
```

---

### ❌ エラー20: App Transport Security blocked

```
App Transport Security has blocked a cleartext HTTP (http://) resource load
```

**原因:**
- HTTPSではなくHTTPを使用している

**解決策:**

```xml
<!-- ✅ Info.plist - 開発環境のみHTTP許可 -->
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsArbitraryLoads</key>
    <false/>
    <key>NSExceptionDomains</key>
    <dict>
        <key>localhost</key>
        <dict>
            <key>NSExceptionAllowsInsecureHTTPLoads</key>
            <true/>
        </dict>
    </dict>
</dict>
```

**注意:** 本番環境では必ずHTTPSを使用してください。

```swift
// ✅ 本番環境ではHTTPS
let url = URL(string: "https://api.example.com/users")!
```

---

### ❌ エラー21: JSON decoding failed

```
DecodingError: keyNotFound
```

**原因:**
- JSONのキーとSwiftのプロパティが一致していない

**解決策:**

```swift
// ❌ キーが一致していない
struct User: Codable {
    let id: Int
    let name: String
    let email: String
}

// JSON: {"id": 1, "user_name": "John", "email": "john@example.com"}
// エラー: keyNotFound(name)

// ✅ CodingKeysでマッピング
struct User: Codable {
    let id: Int
    let name: String
    let email: String

    enum CodingKeys: String, CodingKey {
        case id
        case name = "user_name"  // JSONのキーをマッピング
        case email
    }
}

// ✅ Optionalで安全にデコード
struct User: Codable {
    let id: Int
    let name: String
    let email: String?  // emailがない場合でもエラーにならない
    let createdAt: Date?
}

// ✅ カスタムデコーダー
extension User {
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        id = try container.decode(Int.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        email = try container.decodeIfPresent(String.self, forKey: .email)

        // 日付フォーマット変換
        if let dateString = try container.decodeIfPresent(String.self, forKey: .createdAt) {
            let formatter = ISO8601DateFormatter()
            createdAt = formatter.date(from: dateString)
        } else {
            createdAt = nil
        }
    }
}
```

---

## パフォーマンス問題

### ❌ エラー22: List scrolling is laggy

**原因:**
- 複雑なビューが再レンダリングされている

**解決策:**

```swift
// ❌ 重い処理がbody内にある
struct UserRow: View {
    let user: User

    var body: some View {
        HStack {
            // 毎回画像を処理（重い）
            Image(uiImage: processImage(user.avatarURL))
                .resizable()
                .frame(width: 50, height: 50)

            Text(user.name)
        }
    }

    func processImage(_ url: String) -> UIImage {
        // 重い画像処理
        return UIImage()
    }
}

// ✅ LazyVStackとキャッシュを使用
struct UserListView: View {
    let users: [User]

    var body: some View {
        ScrollView {
            LazyVStack {
                ForEach(users) { user in
                    UserRow(user: user)
                }
            }
        }
    }
}

struct UserRow: View {
    let user: User

    var body: some View {
        HStack {
            AsyncImage(url: URL(string: user.avatarURL)) { image in
                image
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } placeholder: {
                ProgressView()
            }
            .frame(width: 50, height: 50)
            .clipShape(Circle())

            Text(user.name)
        }
    }
}
```

---

### ❌ エラー23: Memory warning

```
Received memory warning
```

**原因:**
- 大量の画像をメモリに保持している

**解決策:**

```swift
// ✅ 画像キャッシュマネージャー
class ImageCache {
    static let shared = ImageCache()
    private let cache = NSCache<NSString, UIImage>()

    private init() {
        cache.countLimit = 100
        cache.totalCostLimit = 50 * 1024 * 1024  // 50MB
    }

    func image(for url: String) -> UIImage? {
        return cache.object(forKey: url as NSString)
    }

    func setImage(_ image: UIImage, for url: String) {
        cache.setObject(image, forKey: url as NSString)
    }

    func clear() {
        cache.removeAllObjects()
    }
}

// メモリ警告時にキャッシュをクリア
class AppDelegate: NSObject, UIApplicationDelegate {
    func applicationDidReceiveMemoryWarning(_ application: UIApplication) {
        ImageCache.shared.clear()
    }
}

// ✅ 画像の遅延読み込み
struct ImageGalleryView: View {
    let imageURLs: [String]

    var body: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 100))]) {
                ForEach(imageURLs, id: \.self) { url in
                    AsyncImage(url: URL(string: url)) { phase in
                        switch phase {
                        case .empty:
                            ProgressView()
                        case .success(let image):
                            image
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                        case .failure:
                            Image(systemName: "photo")
                        @unknown default:
                            EmptyView()
                        }
                    }
                    .frame(width: 100, height: 100)
                    .clipped()
                }
            }
        }
    }
}
```

---

### ❌ エラー24: Animation stuttering

**原因:**
- アニメーション中に重い処理が実行されている

**解決策:**

```swift
// ❌ アニメーション中に同期処理
struct ContentView: View {
    @State private var isExpanded = false

    var body: some View {
        VStack {
            Button("Toggle") {
                withAnimation {
                    isExpanded.toggle()
                }

                // ❌ 重い処理（アニメーションがカクつく）
                processData()
            }

            if isExpanded {
                DetailView()
                    .transition(.slide)
            }
        }
    }

    func processData() {
        // 重い処理
    }
}

// ✅ 非同期処理に分離
struct ContentView: View {
    @State private var isExpanded = false

    var body: some View {
        VStack {
            Button("Toggle") {
                withAnimation {
                    isExpanded.toggle()
                }

                // ✅ 非同期で実行
                Task {
                    await processData()
                }
            }

            if isExpanded {
                DetailView()
                    .transition(.slide)
            }
        }
    }

    func processData() async {
        // 重い処理
    }
}

// ✅ スムーズなアニメーション
struct AnimatedView: View {
    @State private var scale: CGFloat = 1.0

    var body: some View {
        Circle()
            .fill(.blue)
            .frame(width: 100, height: 100)
            .scaleEffect(scale)
            .animation(.spring(response: 0.5, dampingFraction: 0.6), value: scale)
            .onTapGesture {
                scale = scale == 1.0 ? 1.5 : 1.0
            }
    }
}
```

---

### ❌ エラー25: Background task timeout

```
Background task was terminated because it exceeded its background time limit
```

**原因:**
- バックグラウンドタスクが時間内に完了していない

**解決策:**

```swift
// ✅ バックグラウンドタスク管理
import UIKit

class BackgroundTaskManager {
    static let shared = BackgroundTaskManager()

    private var backgroundTask: UIBackgroundTaskIdentifier = .invalid

    func beginBackgroundTask() {
        backgroundTask = UIApplication.shared.beginBackgroundTask {
            // タイムアウト前に呼ばれる
            self.endBackgroundTask()
        }
    }

    func endBackgroundTask() {
        if backgroundTask != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTask)
            backgroundTask = .invalid
        }
    }

    func performBackgroundTask() async {
        beginBackgroundTask()

        defer {
            endBackgroundTask()
        }

        // バックグラウンド処理
        await uploadData()
    }

    private func uploadData() async {
        // データアップロード処理
    }
}

// ✅ App Delegateで使用
class AppDelegate: NSObject, UIApplicationDelegate {
    func applicationDidEnterBackground(_ application: UIApplication) {
        Task {
            await BackgroundTaskManager.shared.performBackgroundTask()
        }
    }
}

// ✅ Background Modes（Info.plist）
/*
<key>UIBackgroundModes</key>
<array>
    <string>fetch</string>
    <string>remote-notification</string>
</array>
*/
```

---

## まとめ

### このガイドで学んだこと

- iOS開発における25の頻出エラー
- 各エラーの原因と解決策
- SwiftUI、Core Data、ネットワーキングのベストプラクティス

### エラー解決の基本手順

1. **エラーメッセージを読む** - Xcodeのエラーメッセージを確認
2. **ブレークポイント** - エラー発生箇所を特定
3. **プレビューを確認** - SwiftUIプレビューでデバッグ
4. **Console.app** - デバイスログを確認
5. **このガイドで検索** - よくあるエラーはここに記載

### デバッグツール

```swift
// print デバッグ
print("Debug: \(value)")

// Dump（詳細情報）
dump(object)

// Instruments
// Xcode > Product > Profile (Cmd + I)
// - Time Profiler
// - Allocations
// - Leaks

// SwiftUI Preview
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .previewDevice("iPhone 15 Pro")
            .previewDisplayName("iPhone 15 Pro")
    }
}
```

### さらに学ぶ

- **[Apple Developer Documentation](https://developer.apple.com/documentation/)**
- **[SwiftUI Tutorials](https://developer.apple.com/tutorials/swiftui)**
- **[WWDC Videos](https://developer.apple.com/videos/)**

---

**関連ガイド:**
- [iOS Development - iOS開発基礎](../ios-development/SKILL.md)
- [SwiftUI Patterns - SwiftUIパターン](../swiftui-patterns/SKILL.md)

**親ガイド:** [トラブルシューティングDB](./README.md)

# iOS Design Patterns 完全ガイド

## 目次
1. [デザインパターンの基礎](#デザインパターンの基礎)
2. [Creational Patterns (生成パターン)](#creational-patterns-生成パターン)
3. [Structural Patterns (構造パターン)](#structural-patterns-構造パターン)
4. [Behavioral Patterns (振る舞いパターン)](#behavioral-patterns-振る舞いパターン)
5. [iOS特有のパターン](#ios特有のパターン)
6. [Coordinatorパターン](#coordinatorパターン)
7. [Reactive Patterns](#reactive-patterns)
8. [アンチパターンと回避策](#アンチパターンと回避策)

---

## デザインパターンの基礎

### パターンの分類と適用場面

```swift
/*
デザインパターン分類:

1. Creational (生成) - オブジェクトの生成に関するパターン
   - Singleton: グローバルインスタンスの管理
   - Factory: オブジェクト生成の抽象化
   - Builder: 複雑なオブジェクトの段階的構築

2. Structural (構造) - オブジェクトの構成に関するパターン
   - Adapter: インターフェースの変換
   - Decorator: 機能の動的追加
   - Facade: 複雑なサブシステムの簡略化

3. Behavioral (振る舞い) - オブジェクト間の相互作用に関するパターン
   - Observer: 変更の通知
   - Strategy: アルゴリズムの切り替え
   - Command: 操作のカプセル化
*/

// ❌ 悪い例: パターンを使わない密結合
class PaymentProcessor {
    func processPayment(amount: Double, method: String) {
        if method == "credit_card" {
            // クレジットカード処理
            print("Processing credit card payment: \(amount)")
        } else if method == "paypal" {
            // PayPal処理
            print("Processing PayPal payment: \(amount)")
        } else if method == "apple_pay" {
            // Apple Pay処理
            print("Processing Apple Pay payment: \(amount)")
        }
    }
}

// ✅ 良い例: Strategyパターンを使用
protocol PaymentStrategy {
    func process(amount: Double) async throws
}

class CreditCardPayment: PaymentStrategy {
    func process(amount: Double) async throws {
        print("Processing credit card payment: \(amount)")
    }
}

class PayPalPayment: PaymentStrategy {
    func process(amount: Double) async throws {
        print("Processing PayPal payment: \(amount)")
    }
}

class ApplePayPayment: PaymentStrategy {
    func process(amount: Double) async throws {
        print("Processing Apple Pay payment: \(amount)")
    }
}

class PaymentContext {
    private let strategy: PaymentStrategy

    init(strategy: PaymentStrategy) {
        self.strategy = strategy
    }

    func executePayment(amount: Double) async throws {
        try await strategy.process(amount: amount)
    }
}
```

---

## Creational Patterns (生成パターン)

### Singleton Pattern

```swift
// ❌ 悪い例: テストが困難なSingleton
class NetworkManager {
    static let shared = NetworkManager()

    private init() {}

    func fetchData() async throws -> Data {
        // ネットワーク処理
        return Data()
    }
}

// 使用時にモックが困難
class ViewModel {
    func loadData() async {
        _ = try? await NetworkManager.shared.fetchData()
    }
}

// ✅ 良い例: DIを使った柔軟なSingleton
protocol NetworkService {
    func fetchData() async throws -> Data
}

class NetworkManager: NetworkService {
    static let shared = NetworkManager()

    private init() {}

    func fetchData() async throws -> Data {
        // 実装
        return Data()
    }
}

class ViewModel {
    private let networkService: NetworkService

    init(networkService: NetworkService = NetworkManager.shared) {
        self.networkService = networkService
    }

    func loadData() async {
        _ = try? await networkService.fetchData()
    }
}

// テスト時
class MockNetworkService: NetworkService {
    func fetchData() async throws -> Data {
        return Data()
    }
}

// Thread-safe Singleton
final class ThreadSafeManager {
    static let shared = ThreadSafeManager()

    private let queue = DispatchQueue(label: "com.app.manager")
    private var _value: Int = 0

    private init() {}

    var value: Int {
        get {
            queue.sync { _value }
        }
        set {
            queue.async(flags: .barrier) {
                self._value = newValue
            }
        }
    }
}
```

### Factory Pattern

```swift
// Abstract Factory
protocol ViewControllerFactory {
    func makeUserListViewController() -> UIViewController
    func makeUserDetailViewController(user: User) -> UIViewController
}

class DefaultViewControllerFactory: ViewControllerFactory {
    private let container: DIContainer

    init(container: DIContainer) {
        self.container = container
    }

    func makeUserListViewController() -> UIViewController {
        let viewModel = UserListViewModel(
            fetchUsersUseCase: container.resolve(FetchUsersUseCase.self)
        )
        return UserListViewController(viewModel: viewModel)
    }

    func makeUserDetailViewController(user: User) -> UIViewController {
        let viewModel = UserDetailViewModel(user: user)
        return UserDetailViewController(viewModel: viewModel)
    }
}

// Factory Method
protocol ViewFactory {
    associatedtype ViewType: UIView
    func createView() -> ViewType
}

class ButtonFactory: ViewFactory {
    enum Style {
        case primary
        case secondary
        case destructive
    }

    private let style: Style
    private let title: String

    init(style: Style, title: String) {
        self.style = style
        self.title = title
    }

    func createView() -> UIButton {
        let button = UIButton(type: .system)
        button.setTitle(title, for: .normal)

        switch style {
        case .primary:
            button.backgroundColor = .systemBlue
            button.setTitleColor(.white, for: .normal)
        case .secondary:
            button.backgroundColor = .systemGray
            button.setTitleColor(.white, for: .normal)
        case .destructive:
            button.backgroundColor = .systemRed
            button.setTitleColor(.white, for: .normal)
        }

        button.layer.cornerRadius = 8
        button.contentEdgeInsets = UIEdgeInsets(top: 12, left: 24, bottom: 12, right: 24)

        return button
    }
}

// 使用例
let primaryButton = ButtonFactory(style: .primary, title: "Submit").createView()
let secondaryButton = ButtonFactory(style: .secondary, title: "Cancel").createView()
```

### Builder Pattern

```swift
// 複雑なオブジェクトの構築
class URLRequestBuilder {
    private var url: URL?
    private var method: String = "GET"
    private var headers: [String: String] = [:]
    private var body: Data?
    private var timeout: TimeInterval = 30

    func setURL(_ url: URL) -> Self {
        self.url = url
        return self
    }

    func setMethod(_ method: String) -> Self {
        self.method = method
        return self
    }

    func addHeader(key: String, value: String) -> Self {
        headers[key] = value
        return self
    }

    func setBody(_ body: Data) -> Self {
        self.body = body
        return self
    }

    func setTimeout(_ timeout: TimeInterval) -> Self {
        self.timeout = timeout
        return self
    }

    func build() throws -> URLRequest {
        guard let url = url else {
            throw BuilderError.missingURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = method
        request.allHTTPHeaderFields = headers
        request.httpBody = body
        request.timeoutInterval = timeout

        return request
    }

    enum BuilderError: Error {
        case missingURL
    }
}

// 使用例
let request = try URLRequestBuilder()
    .setURL(URL(string: "https://api.example.com/users")!)
    .setMethod("POST")
    .addHeader(key: "Content-Type", value: "application/json")
    .addHeader(key: "Authorization", value: "Bearer token")
    .setBody(jsonData)
    .setTimeout(60)
    .build()

// Result Builder (SwiftUI風)
@resultBuilder
struct ViewBuilder {
    static func buildBlock(_ components: UIView...) -> [UIView] {
        components
    }

    static func buildOptional(_ component: [UIView]?) -> [UIView] {
        component ?? []
    }

    static func buildEither(first component: [UIView]) -> [UIView] {
        component
    }

    static func buildEither(second component: [UIView]) -> [UIView] {
        component
    }
}

class StackViewBuilder {
    private let stackView = UIStackView()

    init(axis: NSLayoutConstraint.Axis = .vertical, spacing: CGFloat = 8) {
        stackView.axis = axis
        stackView.spacing = spacing
    }

    @ViewBuilder
    func build(@ViewBuilder _ content: () -> [UIView]) -> UIStackView {
        let views = content()
        views.forEach { stackView.addArrangedSubview($0) }
        return stackView
    }
}

// 使用例
let stack = StackViewBuilder(axis: .vertical, spacing: 16).build {
    titleLabel
    subtitleLabel
    if showDescription {
        descriptionLabel
    }
    actionButton
}
```

---

## Structural Patterns (構造パターン)

### Adapter Pattern

```swift
// 既存のサードパーティライブラリ
class ThirdPartyImageLoader {
    func downloadImage(from urlString: String, completion: @escaping (UIImage?) -> Void) {
        // 実装
    }
}

// 自アプリのプロトコル
protocol ImageLoader {
    func loadImage(from url: URL) async throws -> UIImage
}

// Adapter
class ThirdPartyImageLoaderAdapter: ImageLoader {
    private let loader = ThirdPartyImageLoader()

    func loadImage(from url: URL) async throws -> UIImage {
        try await withCheckedThrowingContinuation { continuation in
            loader.downloadImage(from: url.absoluteString) { image in
                if let image = image {
                    continuation.resume(returning: image)
                } else {
                    continuation.resume(throwing: ImageError.loadFailed)
                }
            }
        }
    }

    enum ImageError: Error {
        case loadFailed
    }
}

// Core Dataアダプター
protocol UserRepository {
    func fetchUsers() async throws -> [User]
    func save(user: User) async throws
}

class CoreDataUserRepository: UserRepository {
    private let context: NSManagedObjectContext

    init(context: NSManagedObjectContext) {
        self.context = context
    }

    func fetchUsers() async throws -> [User] {
        let request = UserEntity.fetchRequest()
        let entities = try context.fetch(request)
        return entities.map { entity in
            User(
                id: entity.id!,
                name: entity.name!,
                email: entity.email!
            )
        }
    }

    func save(user: User) async throws {
        let entity = UserEntity(context: context)
        entity.id = user.id
        entity.name = user.name
        entity.email = user.email
        try context.save()
    }
}
```

### Decorator Pattern

```swift
// Base Protocol
protocol Coffee {
    var cost: Double { get }
    var description: String { get }
}

// Concrete Component
class SimpleCoffee: Coffee {
    var cost: Double { 2.0 }
    var description: String { "Simple Coffee" }
}

// Decorator Base
class CoffeeDecorator: Coffee {
    private let coffee: Coffee

    init(coffee: Coffee) {
        self.coffee = coffee
    }

    var cost: Double {
        coffee.cost
    }

    var description: String {
        coffee.description
    }
}

// Concrete Decorators
class MilkDecorator: CoffeeDecorator {
    override var cost: Double {
        super.cost + 0.5
    }

    override var description: String {
        super.description + ", Milk"
    }
}

class SugarDecorator: CoffeeDecorator {
    override var cost: Double {
        super.cost + 0.2
    }

    override var description: String {
        super.description + ", Sugar"
    }
}

class WhipDecorator: CoffeeDecorator {
    override var cost: Double {
        super.cost + 0.7
    }

    override var description: String {
        super.description + ", Whip"
    }
}

// 使用例
var coffee: Coffee = SimpleCoffee()
coffee = MilkDecorator(coffee: coffee)
coffee = SugarDecorator(coffee: coffee)
coffee = WhipDecorator(coffee: coffee)

print(coffee.description) // "Simple Coffee, Milk, Sugar, Whip"
print(coffee.cost) // 3.4

// ログデコレーター
protocol DataService {
    func fetch() async throws -> Data
}

class APIDataService: DataService {
    func fetch() async throws -> Data {
        // API call
        return Data()
    }
}

class LoggingDataServiceDecorator: DataService {
    private let service: DataService
    private let logger: Logger

    init(service: DataService, logger: Logger) {
        self.service = service
        self.logger = logger
    }

    func fetch() async throws -> Data {
        logger.info("Fetching data...")
        let start = Date()

        do {
            let data = try await service.fetch()
            let duration = Date().timeIntervalSince(start)
            logger.info("Data fetched successfully in \(duration)s")
            return data
        } catch {
            logger.error("Failed to fetch data: \(error)")
            throw error
        }
    }
}
```

### Facade Pattern

```swift
// 複雑なサブシステム
class AuthenticationService {
    func login(email: String, password: String) async throws -> String {
        // 認証処理
        return "auth_token"
    }

    func logout() async throws {
        // ログアウト処理
    }
}

class UserProfileService {
    func fetchProfile(token: String) async throws -> UserProfile {
        // プロフィール取得
        return UserProfile(name: "User", email: "user@example.com")
    }
}

class AnalyticsService {
    func trackLogin(userId: String) {
        // アナリティクス
    }
}

class NotificationService {
    func requestPermissions() async throws {
        // 通知許可リクエスト
    }

    func registerToken(_ token: String) async throws {
        // トークン登録
    }
}

// Facade
class UserSessionFacade {
    private let auth = AuthenticationService()
    private let profile = UserProfileService()
    private let analytics = AnalyticsService()
    private let notifications = NotificationService()

    func login(email: String, password: String) async throws -> UserProfile {
        // 1. 認証
        let token = try await auth.login(email: email, password: password)

        // 2. プロフィール取得
        let userProfile = try await profile.fetchProfile(token: token)

        // 3. アナリティクス
        analytics.trackLogin(userId: userProfile.id)

        // 4. 通知設定
        try? await notifications.requestPermissions()

        return userProfile
    }

    func logout() async throws {
        try await auth.logout()
        // その他のクリーンアップ処理
    }
}

// 使用例
let facade = UserSessionFacade()
let profile = try await facade.login(email: "user@example.com", password: "password")
```

---

## Behavioral Patterns (振る舞いパターン)

### Observer Pattern

```swift
// NotificationCenter を使ったObserver
extension Notification.Name {
    static let userDidLogin = Notification.Name("userDidLogin")
    static let userDidLogout = Notification.Name("userDidLogout")
}

class UserSessionManager {
    func login(user: User) {
        // ログイン処理
        NotificationCenter.default.post(
            name: .userDidLogin,
            object: nil,
            userInfo: ["user": user]
        )
    }

    func logout() {
        // ログアウト処理
        NotificationCenter.default.post(name: .userDidLogout, object: nil)
    }
}

class ProfileViewController: UIViewController {
    private var loginObserver: NSObjectProtocol?

    override func viewDidLoad() {
        super.viewDidLoad()

        loginObserver = NotificationCenter.default.addObserver(
            forName: .userDidLogin,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            if let user = notification.userInfo?["user"] as? User {
                self?.updateUI(for: user)
            }
        }
    }

    deinit {
        if let observer = loginObserver {
            NotificationCenter.default.removeObserver(observer)
        }
    }

    private func updateUI(for user: User) {
        // UI更新
    }
}

// Custom Observer Pattern
protocol Observer: AnyObject {
    func update<T>(with value: T)
}

class Observable<T> {
    private var observers: [WeakObserver<T>] = []
    private var value: T

    init(_ value: T) {
        self.value = value
    }

    func subscribe(_ observer: Observer) {
        let weakObserver = WeakObserver(observer)
        observers.append(weakObserver)
    }

    func unsubscribe(_ observer: Observer) {
        observers.removeAll { $0.observer === observer }
    }

    func notify(_ value: T) {
        self.value = value
        observers.forEach { $0.observer?.update(with: value) }

        // 弱参照がnilになっているものを削除
        observers.removeAll { $0.observer == nil }
    }
}

private class WeakObserver<T> {
    weak var observer: Observer?

    init(_ observer: Observer) {
        self.observer = observer
    }
}

// 使用例
class UserManager {
    let userObservable = Observable<User?>(nil)

    func updateUser(_ user: User) {
        userObservable.notify(user)
    }
}

class ProfileView: Observer {
    func update<T>(with value: T) {
        if let user = value as? User {
            print("User updated: \(user.name)")
        }
    }
}
```

### Strategy Pattern

```swift
// ソート戦略
protocol SortStrategy {
    func sort<T: Comparable>(_ array: [T]) -> [T]
}

class BubbleSortStrategy: SortStrategy {
    func sort<T: Comparable>(_ array: [T]) -> [T] {
        var arr = array
        for i in 0..<arr.count {
            for j in 0..<arr.count - i - 1 {
                if arr[j] > arr[j + 1] {
                    arr.swapAt(j, j + 1)
                }
            }
        }
        return arr
    }
}

class QuickSortStrategy: SortStrategy {
    func sort<T: Comparable>(_ array: [T]) -> [T] {
        guard array.count > 1 else { return array }

        let pivot = array[array.count / 2]
        let less = array.filter { $0 < pivot }
        let equal = array.filter { $0 == pivot }
        let greater = array.filter { $0 > pivot }

        return sort(less) + equal + sort(greater)
    }
}

class SortContext<T: Comparable> {
    private var strategy: SortStrategy

    init(strategy: SortStrategy) {
        self.strategy = strategy
    }

    func setStrategy(_ strategy: SortStrategy) {
        self.strategy = strategy
    }

    func executeStrategy(_ array: [T]) -> [T] {
        strategy.sort(array)
    }
}

// 使用例
let numbers = [5, 2, 8, 1, 9]
let context = SortContext<Int>(strategy: QuickSortStrategy())
let sorted = context.executeStrategy(numbers)

// バリデーション戦略
protocol ValidationStrategy {
    func validate(_ text: String) -> Bool
}

class EmailValidation: ValidationStrategy {
    func validate(_ text: String) -> Bool {
        let emailRegex = "[A-Z0-9a-z._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,64}"
        return NSPredicate(format: "SELF MATCHES %@", emailRegex).evaluate(with: text)
    }
}

class PasswordValidation: ValidationStrategy {
    func validate(_ text: String) -> Bool {
        text.count >= 8 && text.contains(where: { $0.isNumber })
    }
}

class Validator {
    private let strategy: ValidationStrategy

    init(strategy: ValidationStrategy) {
        self.strategy = strategy
    }

    func isValid(_ text: String) -> Bool {
        strategy.validate(text)
    }
}
```

### Command Pattern

```swift
// Command Protocol
protocol Command {
    func execute()
    func undo()
}

// Concrete Commands
class AddTextCommand: Command {
    private let textView: UITextView
    private let text: String
    private var previousText: String = ""

    init(textView: UITextView, text: String) {
        self.textView = textView
        self.text = text
    }

    func execute() {
        previousText = textView.text
        textView.text += text
    }

    func undo() {
        textView.text = previousText
    }
}

class DeleteTextCommand: Command {
    private let textView: UITextView
    private let range: NSRange
    private var deletedText: String = ""

    init(textView: UITextView, range: NSRange) {
        self.textView = textView
        self.range = range
    }

    func execute() {
        if let textRange = Range(range, in: textView.text) {
            deletedText = String(textView.text[textRange])
            textView.text.removeSubrange(textRange)
        }
    }

    func undo() {
        if let textRange = Range(range, in: textView.text) {
            textView.text.insert(contentsOf: deletedText, at: textRange.lowerBound)
        }
    }
}

// Command Manager (Invoker)
class CommandManager {
    private var history: [Command] = []
    private var currentIndex = -1

    func execute(_ command: Command) {
        // 現在位置より後のコマンドを削除
        history.removeSubrange((currentIndex + 1)...)

        command.execute()
        history.append(command)
        currentIndex += 1
    }

    func undo() {
        guard canUndo else { return }

        history[currentIndex].undo()
        currentIndex -= 1
    }

    func redo() {
        guard canRedo else { return }

        currentIndex += 1
        history[currentIndex].execute()
    }

    var canUndo: Bool {
        currentIndex >= 0
    }

    var canRedo: Bool {
        currentIndex < history.count - 1
    }
}

// 使用例
let textView = UITextView()
let manager = CommandManager()

let addCommand = AddTextCommand(textView: textView, text: "Hello")
manager.execute(addCommand)

manager.undo() // テキストが削除される
manager.redo() // テキストが再度追加される
```

---

## iOS特有のパターン

### Delegate Pattern

```swift
// ❌ 悪い例: 循環参照
class DataFetcher {
    var delegate: DataFetcherDelegate? // strongリファレンス
}

protocol DataFetcherDelegate {
    func didFetchData(_ data: Data)
}

// ✅ 良い例: weakリファレンスを使用
protocol DataFetcherDelegate: AnyObject {
    func dataFetcher(_ fetcher: DataFetcher, didFetchData data: Data)
    func dataFetcher(_ fetcher: DataFetcher, didFailWithError error: Error)
}

class DataFetcher {
    weak var delegate: DataFetcherDelegate?

    func fetchData() async {
        do {
            let data = try await performFetch()
            delegate?.dataFetcher(self, didFetchData: data)
        } catch {
            delegate?.dataFetcher(self, didFailWithError: error)
        }
    }

    private func performFetch() async throws -> Data {
        // 実装
        return Data()
    }
}

class ViewController: UIViewController, DataFetcherDelegate {
    private let fetcher = DataFetcher()

    override func viewDidLoad() {
        super.viewDidLoad()
        fetcher.delegate = self
    }

    func dataFetcher(_ fetcher: DataFetcher, didFetchData data: Data) {
        // データ受信処理
    }

    func dataFetcher(_ fetcher: DataFetcher, didFailWithError error: Error) {
        // エラー処理
    }
}

// Multicast Delegate
class MulticastDelegate<T> {
    private var delegates: [WeakDelegate<T>] = []

    func add(_ delegate: T) {
        let weakDelegate = WeakDelegate(delegate as AnyObject)
        delegates.append(weakDelegate)
    }

    func remove(_ delegate: T) {
        delegates.removeAll { $0.value === (delegate as AnyObject) }
    }

    func invoke(_ invocation: (T) -> Void) {
        delegates.forEach { weakDelegate in
            if let delegate = weakDelegate.value as? T {
                invocation(delegate)
            }
        }

        // nilになったdelegateを削除
        delegates.removeAll { $0.value == nil }
    }

    private class WeakDelegate<T> {
        weak var value: AnyObject?

        init(_ value: AnyObject) {
            self.value = value
        }
    }
}

// 使用例
protocol NetworkDelegate: AnyObject {
    func networkDidChange(isConnected: Bool)
}

class NetworkMonitor {
    private let delegates = MulticastDelegate<NetworkDelegate>()

    func addDelegate(_ delegate: NetworkDelegate) {
        delegates.add(delegate)
    }

    func removeDelegate(_ delegate: NetworkDelegate) {
        delegates.remove(delegate)
    }

    private func notifyDelegates(isConnected: Bool) {
        delegates.invoke { delegate in
            delegate.networkDidChange(isConnected: isConnected)
        }
    }
}
```

このガイドでは、iOSアプリ開発における主要なデザインパターンを網羅しました。各パターンの適用場面と実装例を理解し、プロジェクトの要件に応じて適切なパターンを選択することが重要です。

# iOS Architecture 完全ガイド

## 目次
1. [アーキテクチャの基礎](#アーキテクチャの基礎)
2. [MVVM (Model-View-ViewModel)](#mvvm-model-view-viewmodel)
3. [Clean Architecture](#clean-architecture)
4. [VIPER](#viper)
5. [依存性注入 (DI)](#依存性注入-di)
6. [Combineとアーキテクチャ](#combineとアーキテクチャ)
7. [レイヤー分離と責務](#レイヤー分離と責務)
8. [実践的なアーキテクチャ設計](#実践的なアーキテクチャ設計)

---

## アーキテクチャの基礎

### アーキテクチャパターンの選択基準

```swift
// アーキテクチャの選択マトリクス
/*
┌─────────────┬──────────┬────────────┬──────────┬────────────┐
│ Pattern     │ 複雑度   │ テスト容易性│ 学習曲線 │ 適用規模   │
├─────────────┼──────────┼────────────┼──────────┼────────────┤
│ MVC         │ 低       │ 低         │ 低       │ 小         │
│ MVVM        │ 中       │ 高         │ 中       │ 中〜大     │
│ Clean Arch  │ 高       │ 最高       │ 高       │ 大         │
│ VIPER       │ 最高     │ 最高       │ 最高     │ 大規模     │
└─────────────┴──────────┴────────────┴──────────┴────────────┘
*/

// ❌ 悪い例: MVC (Massive View Controller)
class UserViewController: UIViewController {
    private var users: [User] = []

    override func viewDidLoad() {
        super.viewDidLoad()

        // ネットワーク処理、ビジネスロジック、UI更新が混在
        URLSession.shared.dataTask(with: URL(string: "https://api.example.com/users")!) { data, _, _ in
            guard let data = data,
                  let users = try? JSONDecoder().decode([User].self, from: data) else { return }

            self.users = users.filter { $0.isActive } // ビジネスロジック

            DispatchQueue.main.async {
                self.tableView.reloadData() // UI更新
            }
        }.resume()
    }
}

// ✅ 良い例: 責務の分離
protocol UserListViewModelProtocol {
    var users: Published<[User]>.Publisher { get }
    func fetchUsers()
}

class UserListViewModel: UserListViewModelProtocol {
    @Published private(set) var users: [User] = []

    private let userRepository: UserRepositoryProtocol

    init(userRepository: UserRepositoryProtocol) {
        self.userRepository = userRepository
    }

    func fetchUsers() {
        userRepository.fetchUsers { [weak self] result in
            switch result {
            case .success(let users):
                self?.users = users.filter { $0.isActive }
            case .failure(let error):
                // エラーハンドリング
                break
            }
        }
    }
}
```

### アーキテクチャの原則 (SOLID)

```swift
// Single Responsibility Principle (単一責任の原則)
// ❌ 悪い例: 複数の責務
class UserManager {
    func fetchUsers() -> [User] { /* ... */ }
    func saveToDatabase(_ users: [User]) { /* ... */ }
    func formatUserNames(_ users: [User]) -> [String] { /* ... */ }
    func sendAnalytics(_ event: String) { /* ... */ }
}

// ✅ 良い例: 責務の分離
protocol UserFetching {
    func fetchUsers() async throws -> [User]
}

protocol UserPersisting {
    func save(_ users: [User]) async throws
}

protocol UserFormatting {
    func formatNames(_ users: [User]) -> [String]
}

// Open/Closed Principle (開放/閉鎖の原則)
protocol UserValidator {
    func validate(_ user: User) -> Bool
}

class AgeValidator: UserValidator {
    func validate(_ user: User) -> Bool {
        user.age >= 18
    }
}

class EmailValidator: UserValidator {
    func validate(_ user: User) -> Bool {
        user.email.contains("@")
    }
}

class CompositeValidator: UserValidator {
    private let validators: [UserValidator]

    init(validators: [UserValidator]) {
        self.validators = validators
    }

    func validate(_ user: User) -> Bool {
        validators.allSatisfy { $0.validate(user) }
    }
}

// Liskov Substitution Principle (リスコフの置換原則)
protocol Shape {
    func area() -> Double
}

class Rectangle: Shape {
    let width: Double
    let height: Double

    init(width: Double, height: Double) {
        self.width = width
        self.height = height
    }

    func area() -> Double {
        width * height
    }
}

class Square: Shape {
    let side: Double

    init(side: Double) {
        self.side = side
    }

    func area() -> Double {
        side * side
    }
}

// Interface Segregation Principle (インターフェース分離の原則)
// ❌ 悪い例: 大きすぎるインターフェース
protocol Worker {
    func work()
    func eat()
    func sleep()
}

// ✅ 良い例: 小さな専用インターフェース
protocol Workable {
    func work()
}

protocol Eatable {
    func eat()
}

protocol Sleepable {
    func sleep()
}

// Dependency Inversion Principle (依存性逆転の原則)
// ❌ 悪い例: 具象クラスへの依存
class UserService {
    private let apiClient = APIClient() // 具象クラスに依存

    func fetchUsers() {
        apiClient.get("/users")
    }
}

// ✅ 良い例: 抽象への依存
protocol HTTPClient {
    func get(_ path: String) async throws -> Data
}

class UserService {
    private let httpClient: HTTPClient

    init(httpClient: HTTPClient) {
        self.httpClient = httpClient
    }

    func fetchUsers() async throws -> [User] {
        let data = try await httpClient.get("/users")
        return try JSONDecoder().decode([User].self, from: data)
    }
}
```

---

## MVVM (Model-View-ViewModel)

### MVVMの基本実装

```swift
// Model
struct User: Codable, Identifiable {
    let id: UUID
    let name: String
    let email: String
    let avatarURL: URL?
    let isActive: Bool
}

// Repository (Data Layer)
protocol UserRepositoryProtocol {
    func fetchUsers() async throws -> [User]
    func updateUser(_ user: User) async throws
}

class UserRepository: UserRepositoryProtocol {
    private let apiClient: HTTPClient
    private let cache: CacheService

    init(apiClient: HTTPClient, cache: CacheService) {
        self.apiClient = apiClient
        self.cache = cache
    }

    func fetchUsers() async throws -> [User] {
        // キャッシュチェック
        if let cachedUsers = cache.get([User].self, forKey: "users") {
            return cachedUsers
        }

        // APIから取得
        let data = try await apiClient.get("/users")
        let users = try JSONDecoder().decode([User].self, from: data)

        // キャッシュに保存
        cache.set(users, forKey: "users")

        return users
    }

    func updateUser(_ user: User) async throws {
        let data = try JSONEncoder().encode(user)
        _ = try await apiClient.put("/users/\(user.id)", body: data)

        // キャッシュの無効化
        cache.remove(forKey: "users")
    }
}

// ViewModel
@MainActor
class UserListViewModel: ObservableObject {
    @Published private(set) var users: [User] = []
    @Published private(set) var isLoading = false
    @Published private(set) var error: Error?

    private let userRepository: UserRepositoryProtocol
    private let analytics: AnalyticsService

    init(
        userRepository: UserRepositoryProtocol,
        analytics: AnalyticsService
    ) {
        self.userRepository = userRepository
        self.analytics = analytics
    }

    func fetchUsers() async {
        isLoading = true
        error = nil

        do {
            users = try await userRepository.fetchUsers()
            analytics.track(.userListViewed(count: users.count))
        } catch {
            self.error = error
            analytics.track(.userListError(error))
        }

        isLoading = false
    }

    func toggleUserStatus(_ user: User) async {
        var updatedUser = user
        updatedUser.isActive.toggle()

        do {
            try await userRepository.updateUser(updatedUser)

            if let index = users.firstIndex(where: { $0.id == user.id }) {
                users[index] = updatedUser
            }
        } catch {
            self.error = error
        }
    }

    // Computed Properties for View
    var activeUsersCount: Int {
        users.filter(\.isActive).count
    }

    var hasUsers: Bool {
        !users.isEmpty
    }
}

// View (SwiftUI)
struct UserListView: View {
    @StateObject private var viewModel: UserListViewModel

    init(viewModel: UserListViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        NavigationView {
            Group {
                if viewModel.isLoading {
                    ProgressView()
                } else if let error = viewModel.error {
                    ErrorView(error: error, retry: {
                        Task { await viewModel.fetchUsers() }
                    })
                } else if viewModel.hasUsers {
                    userList
                } else {
                    EmptyStateView()
                }
            }
            .navigationTitle("Users (\(viewModel.activeUsersCount))")
            .task {
                await viewModel.fetchUsers()
            }
        }
    }

    private var userList: some View {
        List(viewModel.users) { user in
            UserRow(user: user) {
                Task {
                    await viewModel.toggleUserStatus(user)
                }
            }
        }
    }
}
```

### MVVM with Combine

```swift
import Combine

class CombineUserListViewModel: ObservableObject {
    @Published private(set) var users: [User] = []
    @Published private(set) var isLoading = false
    @Published private(set) var error: Error?

    private let userRepository: UserRepositoryProtocol
    private var cancellables = Set<AnyCancellable>()

    init(userRepository: UserRepositoryProtocol) {
        self.userRepository = userRepository
    }

    func fetchUsers() {
        isLoading = true
        error = nil

        Future { [weak self] promise in
            Task {
                do {
                    let users = try await self?.userRepository.fetchUsers() ?? []
                    promise(.success(users))
                } catch {
                    promise(.failure(error))
                }
            }
        }
        .receive(on: DispatchQueue.main)
        .sink(
            receiveCompletion: { [weak self] completion in
                self?.isLoading = false
                if case .failure(let error) = completion {
                    self?.error = error
                }
            },
            receiveValue: { [weak self] users in
                self?.users = users
            }
        )
        .store(in: &cancellables)
    }
}

// Input/Output パターン
class InputOutputViewModel {
    struct Input {
        let fetchTrigger: AnyPublisher<Void, Never>
        let refreshTrigger: AnyPublisher<Void, Never>
        let searchText: AnyPublisher<String, Never>
    }

    struct Output {
        let users: AnyPublisher<[User], Never>
        let isLoading: AnyPublisher<Bool, Never>
        let error: AnyPublisher<Error?, Never>
    }

    private let userRepository: UserRepositoryProtocol
    private var cancellables = Set<AnyCancellable>()

    init(userRepository: UserRepositoryProtocol) {
        self.userRepository = userRepository
    }

    func transform(input: Input) -> Output {
        let errorSubject = PassthroughSubject<Error?, Never>()
        let loadingSubject = CurrentValueSubject<Bool, Never>(false)

        let fetchedUsers = Publishers.Merge(
            input.fetchTrigger,
            input.refreshTrigger
        )
        .handleEvents(
            receiveOutput: { _ in loadingSubject.send(true) }
        )
        .flatMap { [userRepository] _ in
            Future<[User], Error> { promise in
                Task {
                    do {
                        let users = try await userRepository.fetchUsers()
                        promise(.success(users))
                    } catch {
                        promise(.failure(error))
                    }
                }
            }
        }
        .handleEvents(
            receiveCompletion: { completion in
                loadingSubject.send(false)
                if case .failure(let error) = completion {
                    errorSubject.send(error)
                }
            }
        )
        .replaceError(with: [])
        .share()

        let filteredUsers = Publishers.CombineLatest(
            fetchedUsers,
            input.searchText
        )
        .map { users, searchText in
            guard !searchText.isEmpty else { return users }
            return users.filter { $0.name.localizedCaseInsensitiveContains(searchText) }
        }
        .eraseToAnyPublisher()

        return Output(
            users: filteredUsers,
            isLoading: loadingSubject.eraseToAnyPublisher(),
            error: errorSubject.eraseToAnyPublisher()
        )
    }
}
```

---

## Clean Architecture

### Clean Architectureのレイヤー構造

```swift
// Domain Layer - Entities
struct User {
    let id: UUID
    let name: String
    let email: String

    func validate() throws {
        guard !name.isEmpty else {
            throw ValidationError.emptyName
        }

        guard email.contains("@") else {
            throw ValidationError.invalidEmail
        }
    }
}

enum ValidationError: Error {
    case emptyName
    case invalidEmail
}

// Domain Layer - Use Cases
protocol FetchUsersUseCase {
    func execute() async throws -> [User]
}

class FetchUsersUseCaseImpl: FetchUsersUseCase {
    private let userRepository: UserRepositoryProtocol
    private let logger: Logger

    init(
        userRepository: UserRepositoryProtocol,
        logger: Logger
    ) {
        self.userRepository = userRepository
        self.logger = logger
    }

    func execute() async throws -> [User] {
        logger.info("Fetching users")

        do {
            let users = try await userRepository.fetchUsers()
            logger.info("Fetched \(users.count) users")
            return users
        } catch {
            logger.error("Failed to fetch users: \(error)")
            throw error
        }
    }
}

protocol UpdateUserUseCase {
    func execute(user: User) async throws
}

class UpdateUserUseCaseImpl: UpdateUserUseCase {
    private let userRepository: UserRepositoryProtocol
    private let validator: UserValidator

    init(
        userRepository: UserRepositoryProtocol,
        validator: UserValidator
    ) {
        self.userRepository = userRepository
        self.validator = validator
    }

    func execute(user: User) async throws {
        // ビジネスルールの検証
        guard validator.validate(user) else {
            throw ValidationError.invalidUser
        }

        try user.validate()
        try await userRepository.updateUser(user)
    }
}

// Data Layer - Repository Implementation
class UserRepositoryImpl: UserRepositoryProtocol {
    private let remoteDataSource: UserRemoteDataSource
    private let localDataSource: UserLocalDataSource

    init(
        remoteDataSource: UserRemoteDataSource,
        localDataSource: UserLocalDataSource
    ) {
        self.remoteDataSource = remoteDataSource
        self.localDataSource = localDataSource
    }

    func fetchUsers() async throws -> [User] {
        // ローカルデータを先に返す
        let localUsers = try await localDataSource.fetchUsers()

        // バックグラウンドでリモートデータを取得
        Task {
            do {
                let remoteUsers = try await remoteDataSource.fetchUsers()
                try await localDataSource.save(remoteUsers)
            } catch {
                // エラーは無視（オフライン対応）
            }
        }

        return localUsers
    }

    func updateUser(_ user: User) async throws {
        // まずローカルに保存
        try await localDataSource.update(user)

        // リモートに同期
        try await remoteDataSource.update(user)
    }
}

// Data Layer - Data Sources
protocol UserRemoteDataSource {
    func fetchUsers() async throws -> [User]
    func update(_ user: User) async throws
}

class UserAPIDataSource: UserRemoteDataSource {
    private let httpClient: HTTPClient

    init(httpClient: HTTPClient) {
        self.httpClient = httpClient
    }

    func fetchUsers() async throws -> [User] {
        let data = try await httpClient.get("/api/users")
        let dto = try JSONDecoder().decode([UserDTO].self, from: data)
        return dto.map { $0.toDomain() }
    }

    func update(_ user: User) async throws {
        let dto = UserDTO.from(user)
        let data = try JSONEncoder().encode(dto)
        _ = try await httpClient.put("/api/users/\(user.id)", body: data)
    }
}

protocol UserLocalDataSource {
    func fetchUsers() async throws -> [User]
    func save(_ users: [User]) async throws
    func update(_ user: User) async throws
}

class UserCoreDataSource: UserLocalDataSource {
    private let context: NSManagedObjectContext

    init(context: NSManagedObjectContext) {
        self.context = context
    }

    func fetchUsers() async throws -> [User] {
        let request = UserEntity.fetchRequest()
        let entities = try context.fetch(request)
        return entities.map { $0.toDomain() }
    }

    func save(_ users: [User]) async throws {
        // 既存データを削除
        let deleteRequest = NSBatchDeleteRequest(
            fetchRequest: UserEntity.fetchRequest()
        )
        try context.execute(deleteRequest)

        // 新しいデータを保存
        for user in users {
            let entity = UserEntity(context: context)
            entity.update(from: user)
        }

        try context.save()
    }

    func update(_ user: User) async throws {
        let request = UserEntity.fetchRequest()
        request.predicate = NSPredicate(format: "id == %@", user.id as CVarArg)

        if let entity = try context.fetch(request).first {
            entity.update(from: user)
            try context.save()
        }
    }
}

// Presentation Layer
@MainActor
class CleanUserListViewModel: ObservableObject {
    @Published private(set) var users: [User] = []
    @Published private(set) var isLoading = false
    @Published private(set) var error: Error?

    private let fetchUsersUseCase: FetchUsersUseCase
    private let updateUserUseCase: UpdateUserUseCase

    init(
        fetchUsersUseCase: FetchUsersUseCase,
        updateUserUseCase: UpdateUserUseCase
    ) {
        self.fetchUsersUseCase = fetchUsersUseCase
        self.updateUserUseCase = updateUserUseCase
    }

    func loadUsers() async {
        isLoading = true
        error = nil

        do {
            users = try await fetchUsersUseCase.execute()
        } catch {
            self.error = error
        }

        isLoading = false
    }

    func updateUser(_ user: User) async {
        do {
            try await updateUserUseCase.execute(user: user)
            await loadUsers()
        } catch {
            self.error = error
        }
    }
}
```

---

## VIPER

### VIPERアーキテクチャの完全実装

```swift
// Entity
struct UserEntity {
    let id: UUID
    let name: String
    let email: String
    let createdAt: Date
}

// Interactor
protocol UserListInteractorProtocol {
    func fetchUsers()
}

protocol UserListInteractorOutputProtocol: AnyObject {
    func didFetchUsers(_ users: [UserEntity])
    func didFailToFetchUsers(_ error: Error)
}

class UserListInteractor: UserListInteractorProtocol {
    weak var presenter: UserListInteractorOutputProtocol?
    private let userRepository: UserRepositoryProtocol

    init(userRepository: UserRepositoryProtocol) {
        self.userRepository = userRepository
    }

    func fetchUsers() {
        Task {
            do {
                let users = try await userRepository.fetchUsers()
                await MainActor.run {
                    presenter?.didFetchUsers(users)
                }
            } catch {
                await MainActor.run {
                    presenter?.didFailToFetchUsers(error)
                }
            }
        }
    }
}

// Presenter
protocol UserListPresenterProtocol {
    func viewDidLoad()
    func didSelectUser(at index: Int)
    func didTapRefresh()
}

class UserListPresenter: UserListPresenterProtocol {
    weak var view: UserListViewProtocol?
    var interactor: UserListInteractorProtocol?
    var router: UserListRouterProtocol?

    private var users: [UserEntity] = []

    func viewDidLoad() {
        view?.showLoading()
        interactor?.fetchUsers()
    }

    func didSelectUser(at index: Int) {
        let user = users[index]
        router?.navigateToUserDetail(user: user)
    }

    func didTapRefresh() {
        view?.showLoading()
        interactor?.fetchUsers()
    }
}

extension UserListPresenter: UserListInteractorOutputProtocol {
    func didFetchUsers(_ users: [UserEntity]) {
        self.users = users
        let viewModels = users.map { user in
            UserViewModel(
                name: user.name,
                email: user.email,
                subtitle: formatDate(user.createdAt)
            )
        }
        view?.hideLoading()
        view?.displayUsers(viewModels)
    }

    func didFailToFetchUsers(_ error: Error) {
        view?.hideLoading()
        view?.displayError(error.localizedDescription)
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return formatter.string(from: date)
    }
}

// View
protocol UserListViewProtocol: AnyObject {
    func showLoading()
    func hideLoading()
    func displayUsers(_ users: [UserViewModel])
    func displayError(_ message: String)
}

struct UserViewModel {
    let name: String
    let email: String
    let subtitle: String
}

class UserListViewController: UIViewController, UserListViewProtocol {
    var presenter: UserListPresenterProtocol?

    private let tableView = UITableView()
    private let loadingView = UIActivityIndicatorView(style: .large)

    private var users: [UserViewModel] = []

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        presenter?.viewDidLoad()
    }

    private func setupUI() {
        title = "Users"

        navigationItem.rightBarButtonItem = UIBarButtonItem(
            barButtonSystemItem: .refresh,
            target: self,
            action: #selector(refreshTapped)
        )

        // TableView setup
        view.addSubview(tableView)
        tableView.delegate = self
        tableView.dataSource = self
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "Cell")

        // Loading view setup
        view.addSubview(loadingView)
        loadingView.center = view.center
    }

    @objc private func refreshTapped() {
        presenter?.didTapRefresh()
    }

    // MARK: - UserListViewProtocol

    func showLoading() {
        loadingView.startAnimating()
        tableView.isHidden = true
    }

    func hideLoading() {
        loadingView.stopAnimating()
        tableView.isHidden = false
    }

    func displayUsers(_ users: [UserViewModel]) {
        self.users = users
        tableView.reloadData()
    }

    func displayError(_ message: String) {
        let alert = UIAlertController(
            title: "Error",
            message: message,
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
}

extension UserListViewController: UITableViewDelegate, UITableViewDataSource {
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        users.count
    }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "Cell", for: indexPath)
        let user = users[indexPath.row]

        var config = cell.defaultContentConfiguration()
        config.text = user.name
        config.secondaryText = user.email
        cell.contentConfiguration = config

        return cell
    }

    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        presenter?.didSelectUser(at: indexPath.row)
        tableView.deselectRow(at: indexPath, animated: true)
    }
}

// Router
protocol UserListRouterProtocol {
    func navigateToUserDetail(user: UserEntity)
}

class UserListRouter: UserListRouterProtocol {
    weak var viewController: UIViewController?

    func navigateToUserDetail(user: UserEntity) {
        let detailModule = UserDetailModule.build(user: user)
        viewController?.navigationController?.pushViewController(
            detailModule,
            animated: true
        )
    }
}

// Module Builder
enum UserListModule {
    static func build(userRepository: UserRepositoryProtocol) -> UIViewController {
        let view = UserListViewController()
        let presenter = UserListPresenter()
        let interactor = UserListInteractor(userRepository: userRepository)
        let router = UserListRouter()

        view.presenter = presenter
        presenter.view = view
        presenter.interactor = interactor
        presenter.router = router
        interactor.presenter = presenter
        router.viewController = view

        return view
    }
}
```

---

## 依存性注入 (DI)

### DIコンテナの実装

```swift
// DIコンテナ
class DIContainer {
    static let shared = DIContainer()

    private var services: [String: Any] = [:]
    private var factories: [String: () -> Any] = [:]

    private init() {}

    // シングルトン登録
    func register<T>(_ type: T.Type, instance: T) {
        let key = String(describing: type)
        services[key] = instance
    }

    // ファクトリー登録
    func register<T>(_ type: T.Type, factory: @escaping () -> T) {
        let key = String(describing: type)
        factories[key] = factory
    }

    // 解決
    func resolve<T>(_ type: T.Type) -> T {
        let key = String(describing: type)

        // シングルトンをチェック
        if let service = services[key] as? T {
            return service
        }

        // ファクトリーをチェック
        if let factory = factories[key] {
            if let service = factory() as? T {
                return service
            }
        }

        fatalError("No registration for type \(type)")
    }
}

// DI setup
extension DIContainer {
    func registerDependencies() {
        // Infrastructure
        register(HTTPClient.self, instance: URLSessionHTTPClient())
        register(CacheService.self, instance: UserDefaultsCache())

        // Data Sources
        register(
            UserRemoteDataSource.self,
            factory: {
                UserAPIDataSource(httpClient: self.resolve(HTTPClient.self))
            }
        )

        // Repositories
        register(
            UserRepositoryProtocol.self,
            factory: {
                UserRepositoryImpl(
                    remoteDataSource: self.resolve(UserRemoteDataSource.self),
                    localDataSource: self.resolve(UserLocalDataSource.self)
                )
            }
        )

        // Use Cases
        register(
            FetchUsersUseCase.self,
            factory: {
                FetchUsersUseCaseImpl(
                    userRepository: self.resolve(UserRepositoryProtocol.self),
                    logger: self.resolve(Logger.self)
                )
            }
        )
    }
}

// Property Wrapper for DI
@propertyWrapper
struct Injected<T> {
    private let container: DIContainer

    init(container: DIContainer = .shared) {
        self.container = container
    }

    var wrappedValue: T {
        container.resolve(T.self)
    }
}

// 使用例
class UserViewModel {
    @Injected private var fetchUsersUseCase: FetchUsersUseCase
    @Injected private var updateUserUseCase: UpdateUserUseCase

    func loadUsers() async {
        _ = try? await fetchUsersUseCase.execute()
    }
}
```

このガイドでは、iOSアーキテクチャの基礎から、MVVM、Clean Architecture、VIPERまでの実践的な実装パターンを網羅しました。各アーキテクチャの特性を理解し、プロジェクトの規模や要件に応じて適切なパターンを選択することが重要です。

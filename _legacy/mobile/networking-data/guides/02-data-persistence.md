# Data Persistence - iOSデータ永続化完全ガイド

## 📋 目次

1. [概要](#概要)
2. [UserDefaults](#userdefaults)
3. [FileManager](#filemanager)
4. [Keychain](#keychain)
5. [Core Data](#core-data)
6. [Realm](#realm)
7. [SQLite](#sqlite)
8. [データ永続化戦略](#データ永続化戦略)
9. [マイグレーション](#マイグレーション)
10. [パフォーマンス最適化](#パフォーマンス最適化)
11. [よくある問題と解決策](#よくある問題と解決策)

## 概要

iOSアプリにおけるデータ永続化は、ユーザー体験の質を大きく左右します。このガイドでは、各ストレージ手段の特性と適切な使い分けを解説します。

### このガイドの対象者

- iOSエンジニア
- データ永続化を実装する開発者
- パフォーマンス最適化を目指す方

### 学べること

- 各ストレージの使い分け
- Core Dataの効率的な使用
- セキュアなデータ保存
- マイグレーション戦略

## UserDefaults

### 基本的な使用

```swift
class AppSettings {
    static let shared = AppSettings()
    private let defaults = UserDefaults.standard

    // Bool値
    var notificationsEnabled: Bool {
        get { defaults.bool(forKey: Keys.notificationsEnabled) }
        set { defaults.set(newValue, forKey: Keys.notificationsEnabled) }
    }

    // String値
    var username: String? {
        get { defaults.string(forKey: Keys.username) }
        set { defaults.set(newValue, forKey: Keys.username) }
    }

    // Int値
    var loginCount: Int {
        get { defaults.integer(forKey: Keys.loginCount) }
        set { defaults.set(newValue, forKey: Keys.loginCount) }
    }

    // Date値
    var lastLoginDate: Date? {
        get { defaults.object(forKey: Keys.lastLoginDate) as? Date }
        set { defaults.set(newValue, forKey: Keys.lastLoginDate) }
    }

    private enum Keys {
        static let notificationsEnabled = "notificationsEnabled"
        static let username = "username"
        static let loginCount = "loginCount"
        static let lastLoginDate = "lastLoginDate"
    }

    private init() {
        registerDefaults()
    }

    private func registerDefaults() {
        defaults.register(defaults: [
            Keys.notificationsEnabled: true,
            Keys.loginCount: 0
        ])
    }
}
```

### Codableでの使用

```swift
extension UserDefaults {
    func setCodable<T: Codable>(_ value: T, forKey key: String) throws {
        let encoder = JSONEncoder()
        let data = try encoder.encode(value)
        set(data, forKey: key)
    }

    func codable<T: Codable>(forKey key: String) throws -> T? {
        guard let data = data(forKey: key) else { return nil }
        let decoder = JSONDecoder()
        return try decoder.decode(T.self, from: data)
    }

    func removeCodable(forKey key: String) {
        removeObject(forKey: key)
    }
}

// 使用例
struct UserPreferences: Codable {
    var theme: Theme
    var language: String
    var fontSize: FontSize

    enum Theme: String, Codable {
        case light, dark, auto
    }

    enum FontSize: Int, Codable {
        case small = 12
        case medium = 14
        case large = 16
    }
}

class PreferencesManager {
    private let defaults = UserDefaults.standard
    private let preferencesKey = "userPreferences"

    var preferences: UserPreferences {
        get {
            (try? defaults.codable(forKey: preferencesKey)) ?? UserPreferences.default
        }
        set {
            try? defaults.setCodable(newValue, forKey: preferencesKey)
        }
    }
}

extension UserPreferences {
    static let `default` = UserPreferences(
        theme: .auto,
        language: "en",
        fontSize: .medium
    )
}
```

### PropertyWrapperでの型安全な実装

```swift
@propertyWrapper
struct UserDefault<T> {
    let key: String
    let defaultValue: T
    var container: UserDefaults = .standard

    var wrappedValue: T {
        get {
            container.object(forKey: key) as? T ?? defaultValue
        }
        set {
            container.set(newValue, forKey: key)
        }
    }
}

@propertyWrapper
struct CodableUserDefault<T: Codable> {
    let key: String
    let defaultValue: T
    var container: UserDefaults = .standard

    var wrappedValue: T {
        get {
            guard let data = container.data(forKey: key),
                  let value = try? JSONDecoder().decode(T.self, from: data) else {
                return defaultValue
            }
            return value
        }
        set {
            let data = try? JSONEncoder().encode(newValue)
            container.set(data, forKey: key)
        }
    }
}

// 使用例
class Settings {
    @UserDefault(key: "isDarkMode", defaultValue: false)
    var isDarkMode: Bool

    @UserDefault(key: "volume", defaultValue: 0.5)
    var volume: Double

    @CodableUserDefault(key: "preferences", defaultValue: .default)
    var preferences: UserPreferences
}
```

## FileManager

### ディレクトリの取得

```swift
enum Directory {
    case documents
    case caches
    case temporary
    case applicationSupport

    var url: URL {
        let fileManager = FileManager.default
        switch self {
        case .documents:
            return fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        case .caches:
            return fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        case .temporary:
            return fileManager.temporaryDirectory
        case .applicationSupport:
            return fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        }
    }
}
```

### ファイル操作マネージャー

```swift
class FileStorageManager {
    static let shared = FileStorageManager()
    private let fileManager = FileManager.default

    private init() {}

    // MARK: - Save
    func save<T: Codable>(_ object: T, to directory: Directory, filename: String) throws {
        let url = directory.url.appendingPathComponent(filename)
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(object)
        try data.write(to: url, options: .atomic)
    }

    func saveData(_ data: Data, to directory: Directory, filename: String) throws {
        let url = directory.url.appendingPathComponent(filename)
        try data.write(to: url, options: .atomic)
    }

    // MARK: - Load
    func load<T: Codable>(from directory: Directory, filename: String) throws -> T {
        let url = directory.url.appendingPathComponent(filename)
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(T.self, from: data)
    }

    func loadData(from directory: Directory, filename: String) throws -> Data {
        let url = directory.url.appendingPathComponent(filename)
        return try Data(contentsOf: url)
    }

    // MARK: - Delete
    func delete(from directory: Directory, filename: String) throws {
        let url = directory.url.appendingPathComponent(filename)
        try fileManager.removeItem(at: url)
    }

    // MARK: - Exists
    func fileExists(in directory: Directory, filename: String) -> Bool {
        let url = directory.url.appendingPathComponent(filename)
        return fileManager.fileExists(atPath: url.path)
    }

    // MARK: - List Files
    func listFiles(in directory: Directory) throws -> [String] {
        let contents = try fileManager.contentsOfDirectory(
            at: directory.url,
            includingPropertiesForKeys: nil
        )
        return contents.map { $0.lastPathComponent }
    }

    // MARK: - File Attributes
    func fileSize(in directory: Directory, filename: String) throws -> Int {
        let url = directory.url.appendingPathComponent(filename)
        let attributes = try fileManager.attributesOfItem(atPath: url.path)
        return attributes[.size] as? Int ?? 0
    }

    func modificationDate(in directory: Directory, filename: String) throws -> Date? {
        let url = directory.url.appendingPathComponent(filename)
        let attributes = try fileManager.attributesOfItem(atPath: url.path)
        return attributes[.modificationDate] as? Date
    }
}

// 使用例
struct Note: Codable {
    let id: UUID
    let title: String
    let content: String
    let createdAt: Date
}

let note = Note(id: UUID(), title: "My Note", content: "Content", createdAt: Date())
try FileStorageManager.shared.save(note, to: .documents, filename: "note.json")

let loadedNote: Note = try FileStorageManager.shared.load(from: .documents, filename: "note.json")
```

### 画像の保存と読み込み

```swift
class ImageStorageManager {
    static let shared = ImageStorageManager()

    func saveImage(_ image: UIImage, filename: String, compressionQuality: CGFloat = 0.8) throws {
        guard let data = image.jpegData(compressionQuality: compressionQuality) else {
            throw StorageError.imageConversionFailed
        }

        try FileStorageManager.shared.saveData(data, to: .documents, filename: filename)
    }

    func loadImage(filename: String) throws -> UIImage {
        let data = try FileStorageManager.shared.loadData(from: .documents, filename: filename)
        guard let image = UIImage(data: data) else {
            throw StorageError.imageLoadFailed
        }
        return image
    }

    func deleteImage(filename: String) throws {
        try FileStorageManager.shared.delete(from: .documents, filename: filename)
    }
}

enum StorageError: Error {
    case imageConversionFailed
    case imageLoadFailed
}
```

## Keychain

### Keychainマネージャー

```swift
import Security

class KeychainManager {
    static let shared = KeychainManager()

    private init() {}

    // MARK: - Save
    func save(_ data: Data, service: String, account: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecValueData as String: data
        ]

        // 既存のアイテムを削除
        SecItemDelete(query as CFDictionary)

        // 新しいアイテムを追加
        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw KeychainError.saveFailed(status)
        }
    }

    func save(_ string: String, service: String, account: String) throws {
        guard let data = string.data(using: .utf8) else {
            throw KeychainError.stringConversionFailed
        }
        try save(data, service: service, account: account)
    }

    // MARK: - Load
    func load(service: String, account: String) throws -> Data {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess else {
            throw KeychainError.loadFailed(status)
        }

        guard let data = result as? Data else {
            throw KeychainError.dataConversionFailed
        }

        return data
    }

    func loadString(service: String, account: String) throws -> String {
        let data = try load(service: service, account: account)
        guard let string = String(data: data, encoding: .utf8) else {
            throw KeychainError.stringConversionFailed
        }
        return string
    }

    // MARK: - Delete
    func delete(service: String, account: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]

        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.deleteFailed(status)
        }
    }

    // MARK: - Update
    func update(_ data: Data, service: String, account: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]

        let attributes: [String: Any] = [
            kSecValueData as String: data
        ]

        let status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        guard status == errSecSuccess else {
            // アイテムが存在しない場合は新規作成
            if status == errSecItemNotFound {
                try save(data, service: service, account: account)
            } else {
                throw KeychainError.updateFailed(status)
            }
        }
    }
}

enum KeychainError: Error, LocalizedError {
    case saveFailed(OSStatus)
    case loadFailed(OSStatus)
    case deleteFailed(OSStatus)
    case updateFailed(OSStatus)
    case stringConversionFailed
    case dataConversionFailed

    var errorDescription: String? {
        switch self {
        case .saveFailed(let status):
            return "Failed to save to keychain: \(status)"
        case .loadFailed(let status):
            return "Failed to load from keychain: \(status)"
        case .deleteFailed(let status):
            return "Failed to delete from keychain: \(status)"
        case .updateFailed(let status):
            return "Failed to update keychain: \(status)"
        case .stringConversionFailed:
            return "String conversion failed"
        case .dataConversionFailed:
            return "Data conversion failed"
        }
    }
}
```

### トークンマネージャー

```swift
class TokenManager {
    private let keychain = KeychainManager.shared
    private let service = "com.example.app"

    func saveAccessToken(_ token: String) throws {
        try keychain.save(token, service: service, account: "accessToken")
    }

    func loadAccessToken() throws -> String {
        try keychain.loadString(service: service, account: "accessToken")
    }

    func deleteAccessToken() throws {
        try keychain.delete(service: service, account: "accessToken")
    }

    func saveRefreshToken(_ token: String) throws {
        try keychain.save(token, service: service, account: "refreshToken")
    }

    func loadRefreshToken() throws -> String {
        try keychain.loadString(service: service, account: "refreshToken")
    }

    func deleteAllTokens() throws {
        try? deleteAccessToken()
        try? keychain.delete(service: service, account: "refreshToken")
    }
}
```

## Core Data

### Core Data Stack

```swift
class CoreDataManager {
    static let shared = CoreDataManager()

    lazy var persistentContainer: NSPersistentContainer = {
        let container = NSPersistentContainer(name: "AppModel")

        container.loadPersistentStores { description, error in
            if let error = error {
                fatalError("Unable to load persistent stores: \(error)")
            }
        }

        // 自動マージ設定
        container.viewContext.automaticallyMergesChangesFromParent = true
        container.viewContext.mergePolicy = NSMergeByPropertyObjectTrumpMergePolicy

        return container
    }()

    var viewContext: NSManagedObjectContext {
        persistentContainer.viewContext
    }

    func newBackgroundContext() -> NSManagedObjectContext {
        persistentContainer.newBackgroundContext()
    }

    func save(context: NSManagedObjectContext? = nil) {
        let context = context ?? viewContext

        guard context.hasChanges else { return }

        do {
            try context.save()
        } catch {
            let nsError = error as NSError
            fatalError("Unresolved error \(nsError), \(nsError.userInfo)")
        }
    }

    func performBackgroundTask<T>(_ block: @escaping (NSManagedObjectContext) throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            persistentContainer.performBackgroundTask { context in
                do {
                    let result = try block(context)
                    try context.save()
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}
```

### エンティティ拡張

```swift
// User+CoreDataClass.swift
@objc(User)
public class User: NSManagedObject {
    @NSManaged public var id: UUID
    @NSManaged public var name: String
    @NSManaged public var email: String
    @NSManaged public var createdAt: Date
    @NSManaged public var posts: NSSet?
}

// User+CoreDataProperties.swift
extension User {
    @nonobjc public class func fetchRequest() -> NSFetchRequest<User> {
        return NSFetchRequest<User>(entityName: "User")
    }

    static func create(in context: NSManagedObjectContext, name: String, email: String) -> User {
        let user = User(context: context)
        user.id = UUID()
        user.name = name
        user.email = email
        user.createdAt = Date()
        return user
    }

    static func fetchAll(in context: NSManagedObjectContext) throws -> [User] {
        let request = fetchRequest()
        request.sortDescriptors = [NSSortDescriptor(key: "createdAt", ascending: false)]
        return try context.fetch(request)
    }

    static func fetch(byId id: UUID, in context: NSManagedObjectContext) throws -> User? {
        let request = fetchRequest()
        request.predicate = NSPredicate(format: "id == %@", id as CVarArg)
        request.fetchLimit = 1
        return try context.fetch(request).first
    }
}

// 関連オブジェクトアクセス
extension User {
    var postsArray: [Post] {
        let set = posts as? Set<Post> ?? []
        return set.sorted { $0.createdAt > $1.createdAt }
    }

    func addPost(_ post: Post) {
        addToPosts(post)
    }

    func removePost(_ post: Post) {
        removeFromPosts(post)
    }
}
```

### Repository実装

```swift
protocol UserDataStore {
    func create(name: String, email: String) async throws -> User
    func fetchAll() async throws -> [User]
    func fetch(byId id: UUID) async throws -> User?
    func update(_ user: User, name: String?, email: String?) async throws
    func delete(_ user: User) async throws
}

class CoreDataUserStore: UserDataStore {
    private let coreData = CoreDataManager.shared

    func create(name: String, email: String) async throws -> User {
        try await coreData.performBackgroundTask { context in
            let user = User.create(in: context, name: name, email: email)
            return user
        }
    }

    func fetchAll() async throws -> [User] {
        try await coreData.performBackgroundTask { context in
            try User.fetchAll(in: context)
        }
    }

    func fetch(byId id: UUID) async throws -> User? {
        try await coreData.performBackgroundTask { context in
            try User.fetch(byId: id, in: context)
        }
    }

    func update(_ user: User, name: String?, email: String?) async throws {
        try await coreData.performBackgroundTask { context in
            guard let objectID = user.objectID as NSManagedObjectID?,
                  let user = try? context.existingObject(with: objectID) as? User else {
                throw CoreDataError.objectNotFound
            }

            if let name = name {
                user.name = name
            }
            if let email = email {
                user.email = email
            }
        }
    }

    func delete(_ user: User) async throws {
        try await coreData.performBackgroundTask { context in
            guard let objectID = user.objectID as NSManagedObjectID?,
                  let user = try? context.existingObject(with: objectID) else {
                throw CoreDataError.objectNotFound
            }

            context.delete(user)
        }
    }
}

enum CoreDataError: Error {
    case objectNotFound
    case saveFailed
}
```

### NSFetchedResultsController

```swift
class UserListViewModel: NSObject, ObservableObject {
    @Published var users: [User] = []

    private lazy var fetchedResultsController: NSFetchedResultsController<User> = {
        let request = User.fetchRequest()
        request.sortDescriptors = [NSSortDescriptor(key: "createdAt", ascending: false)]

        let controller = NSFetchedResultsController(
            fetchRequest: request,
            managedObjectContext: CoreDataManager.shared.viewContext,
            sectionNameKeyPath: nil,
            cacheName: nil
        )

        controller.delegate = self
        return controller
    }()

    func fetchUsers() {
        do {
            try fetchedResultsController.performFetch()
            users = fetchedResultsController.fetchedObjects ?? []
        } catch {
            print("Failed to fetch users: \(error)")
        }
    }
}

extension UserListViewModel: NSFetchedResultsControllerDelegate {
    func controllerDidChangeContent(_ controller: NSFetchedResultsController<NSFetchRequestResult>) {
        users = controller.fetchedObjects as? [User] ?? []
    }
}
```

## Realm

### Realmモデル定義

```swift
import RealmSwift

class RealmUser: Object {
    @Persisted(primaryKey: true) var id: UUID
    @Persisted var name: String
    @Persisted var email: String
    @Persisted var createdAt: Date
    @Persisted var posts: List<RealmPost>

    convenience init(name: String, email: String) {
        self.init()
        self.id = UUID()
        self.name = name
        self.email = email
        self.createdAt = Date()
    }
}

class RealmPost: Object {
    @Persisted(primaryKey: true) var id: UUID
    @Persisted var title: String
    @Persisted var content: String
    @Persisted var createdAt: Date
    @Persisted(originProperty: "posts") var owner: LinkingObjects<RealmUser>

    convenience init(title: String, content: String) {
        self.init()
        self.id = UUID()
        self.title = title
        self.content = content
        self.createdAt = Date()
    }
}
```

### Realm Repository

```swift
class RealmUserStore {
    private var realm: Realm {
        try! Realm()
    }

    // Create
    func create(name: String, email: String) throws -> RealmUser {
        let user = RealmUser(name: name, email: email)

        try realm.write {
            realm.add(user)
        }

        return user
    }

    // Read
    func fetchAll() -> [RealmUser] {
        Array(realm.objects(RealmUser.self).sorted(byKeyPath: "createdAt", ascending: false))
    }

    func fetch(byId id: UUID) -> RealmUser? {
        realm.object(ofType: RealmUser.self, forPrimaryKey: id)
    }

    // Update
    func update(_ user: RealmUser, name: String? = nil, email: String? = nil) throws {
        try realm.write {
            if let name = name {
                user.name = name
            }
            if let email = email {
                user.email = email
            }
        }
    }

    // Delete
    func delete(_ user: RealmUser) throws {
        try realm.write {
            realm.delete(user)
        }
    }

    // Query
    func search(nameContains query: String) -> [RealmUser] {
        Array(realm.objects(RealmUser.self).filter("name CONTAINS[cd] %@", query))
    }
}
```

## SQLite

### SQLiteラッパー

```swift
import SQLite3

class SQLiteManager {
    private var db: OpaquePointer?
    private let dbPath: String

    init(dbName: String = "app.db") {
        let fileURL = try! FileManager.default
            .url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
            .appendingPathComponent(dbName)

        dbPath = fileURL.path

        if sqlite3_open(dbPath, &db) != SQLITE_OK {
            print("Error opening database")
        }
    }

    deinit {
        sqlite3_close(db)
    }

    func createTable() {
        let createTableQuery = """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        """

        if sqlite3_exec(db, createTableQuery, nil, nil, nil) != SQLITE_OK {
            print("Error creating table")
        }
    }

    func insert(user: User) -> Bool {
        let insertQuery = "INSERT INTO users (id, name, email, created_at) VALUES (?, ?, ?, ?);"
        var statement: OpaquePointer?

        if sqlite3_prepare_v2(db, insertQuery, -1, &statement, nil) == SQLITE_OK {
            sqlite3_bind_text(statement, 1, user.id.uuidString, -1, nil)
            sqlite3_bind_text(statement, 2, user.name, -1, nil)
            sqlite3_bind_text(statement, 3, user.email, -1, nil)
            sqlite3_bind_double(statement, 4, user.createdAt.timeIntervalSince1970)

            if sqlite3_step(statement) == SQLITE_DONE {
                sqlite3_finalize(statement)
                return true
            }
        }

        sqlite3_finalize(statement)
        return false
    }

    func fetchAll() -> [User] {
        let query = "SELECT id, name, email, created_at FROM users ORDER BY created_at DESC;"
        var statement: OpaquePointer?
        var users: [User] = []

        if sqlite3_prepare_v2(db, query, -1, &statement, nil) == SQLITE_OK {
            while sqlite3_step(statement) == SQLITE_ROW {
                let idString = String(cString: sqlite3_column_text(statement, 0))
                let name = String(cString: sqlite3_column_text(statement, 1))
                let email = String(cString: sqlite3_column_text(statement, 2))
                let timestamp = sqlite3_column_double(statement, 3)

                if let id = UUID(uuidString: idString) {
                    let user = User(
                        id: id,
                        name: name,
                        email: email,
                        createdAt: Date(timeIntervalSince1970: timestamp)
                    )
                    users.append(user)
                }
            }
        }

        sqlite3_finalize(statement)
        return users
    }
}
```

## データ永続化戦略

### ストレージ選択ガイド

```swift
enum StorageType {
    case userDefaults    // 小さな設定値
    case fileManager     // ドキュメント、画像
    case keychain        // パスワード、トークン
    case coreData        // 構造化データ、リレーション
    case realm           // 高速な読み書き
    case sqlite          // カスタムクエリ

    static func recommended(for dataType: DataType) -> StorageType {
        switch dataType {
        case .settings:
            return .userDefaults
        case .credentials:
            return .keychain
        case .documents:
            return .fileManager
        case .structuredData:
            return .coreData
        case .cache:
            return .fileManager
        }
    }

    enum DataType {
        case settings       // アプリ設定
        case credentials    // 認証情報
        case documents      // ドキュメント
        case structuredData // 構造化データ
        case cache          // キャッシュ
    }
}
```

### データ同期戦略

```swift
class DataSyncManager {
    private let local: UserDataStore
    private let remote: UserRepository

    init(local: UserDataStore, remote: UserRepository) {
        self.local = local
        self.remote = remote
    }

    func sync() async throws {
        // リモートから取得
        let remoteUsers = try await remote.fetchAllUsers()

        // ローカルに保存
        for remoteUser in remoteUsers {
            if let localUser = try await local.fetch(byId: remoteUser.id) {
                // 更新
                try await local.update(localUser, name: remoteUser.name, email: remoteUser.email)
            } else {
                // 新規作成
                _ = try await local.create(name: remoteUser.name, email: remoteUser.email)
            }
        }
    }
}
```

## マイグレーション

### Core Dataマイグレーション

```swift
class CoreDataMigrationManager {
    static let shared = CoreDataMigrationManager()

    func migrateIfNeeded() {
        guard needsMigration() else { return }

        performMigration()
    }

    private func needsMigration() -> Bool {
        let storeURL = CoreDataManager.shared.persistentContainer.persistentStoreDescriptions.first?.url

        guard let url = storeURL,
              let metadata = try? NSPersistentStoreCoordinator.metadataForPersistentStore(
                ofType: NSSQLiteStoreType,
                at: url
              ) else {
            return false
        }

        let model = CoreDataManager.shared.persistentContainer.managedObjectModel
        return !model.isConfiguration(withName: nil, compatibleWithStoreMetadata: metadata)
    }

    private func performMigration() {
        // 軽量マイグレーション
        let description = NSPersistentStoreDescription()
        description.shouldMigrateStoreAutomatically = true
        description.shouldInferMappingModelAutomatically = true

        // 重量マイグレーションの例
        // let sourceModel = NSManagedObjectModel.mergedModel(from: [Bundle.main], forStoreMetadata: sourceMetadata)
        // let mappingModel = NSMappingModel(from: [Bundle.main], forSourceModel: sourceModel, destinationModel: destinationModel)
    }
}
```

## パフォーマンス最適化

### バッチ処理

```swift
extension CoreDataUserStore {
    func batchInsert(users: [CreateUserRequest]) async throws {
        try await CoreDataManager.shared.performBackgroundTask { context in
            for request in users {
                _ = User.create(in: context, name: request.name, email: request.email)
            }
            // コンテキストを一度だけ保存
        }
    }

    func batchDelete(predicate: NSPredicate) async throws {
        let request = NSBatchDeleteRequest(fetchRequest: User.fetchRequest())
        request.predicate = predicate

        try await CoreDataManager.shared.performBackgroundTask { context in
            try context.execute(request)
        }
    }
}
```

## よくある問題と解決策

### 問題1: Core Dataのスレッド安全性

```swift
// ❌ 誤った使用
func badExample() {
    let user = User.create(in: viewContext, name: "Test", email: "test@example.com")

    DispatchQueue.global().async {
        // クラッシュ!異なるスレッドからアクセス
        user.name = "Updated"
    }
}

// ✅ 正しい使用
func goodExample() async {
    try await CoreDataManager.shared.performBackgroundTask { context in
        let user = User.create(in: context, name: "Test", email: "test@example.com")
        user.name = "Updated"
    }
}
```

### 問題2: メモリリーク

```swift
// ✅ NSFetchedResultsControllerのリセット
func resetFetchedResultsController() {
    fetchedResultsController.delegate = nil
    NSFetchedResultsController<User>.deleteCache(withName: nil)
}
```

---

**関連ガイド:**
- [01-networking.md](./01-networking.md) - ネットワーク通信
- [03-caching-strategy.md](./03-caching-strategy.md) - キャッシュ戦略

**関連Skills:**
- [ios-development](../../ios-development/SKILL.md) - iOS開発全般
- [ios-security](../../ios-security/SKILL.md) - セキュリティ実装

**参考資料:**
- [Core Data - Apple Developer](https://developer.apple.com/documentation/coredata)
- [Keychain Services - Apple Developer](https://developer.apple.com/documentation/security/keychain_services)

**更新履歴:**
- 2025-12-31: 初版作成

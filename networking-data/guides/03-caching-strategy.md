# Caching Strategy - iOSキャッシュ戦略完全ガイド

## 📋 目次

1. [概要](#概要)
2. [キャッシュの基本原則](#キャッシュの基本原則)
3. [メモリキャッシュ](#メモリキャッシュ)
4. [ディスクキャッシュ](#ディスクキャッシュ)
5. [HTTPキャッシュ](#httpキャッシュ)
6. [画像キャッシュ](#画像キャッシュ)
7. [データキャッシュ](#データキャッシュ)
8. [キャッシュ無効化戦略](#キャッシュ無効化戦略)
9. [オフライン対応](#オフライン対応)
10. [パフォーマンス最適化](#パフォーマンス最適化)
11. [よくある問題と解決策](#よくある問題と解決策)

## 概要

適切なキャッシュ戦略は、アプリのパフォーマンスとユーザー体験を大きく向上させます。このガイドでは、iOSアプリにおける効果的なキャッシュ実装を解説します。

### このガイドの対象者

- iOSエンジニア
- パフォーマンス改善を目指す開発者
- オフライン対応を実装する方

### 学べること

- 効率的なキャッシュ戦略
- メモリとディスクの使い分け
- 画像キャッシュの最適化
- オフラインファースト設計

## キャッシュの基本原則

### キャッシュ戦略の選択

```swift
enum CacheStrategy {
    case networkOnly           // ネットワークのみ
    case cacheOnly            // キャッシュのみ
    case cacheFirst           // キャッシュ優先、なければネットワーク
    case networkFirst         // ネットワーク優先、失敗したらキャッシュ
    case cacheAndNetwork      // キャッシュを表示後、ネットワークで更新
    case cacheWithTimeout     // 期限付きキャッシュ

    static func recommended(for contentType: ContentType) -> CacheStrategy {
        switch contentType {
        case .staticAssets:
            return .cacheFirst
        case .userContent:
            return .cacheAndNetwork
        case .realtime:
            return .networkFirst
        case .criticalData:
            return .networkOnly
        }
    }

    enum ContentType {
        case staticAssets    // 静的アセット
        case userContent     // ユーザーコンテンツ
        case realtime        // リアルタイムデータ
        case criticalData    // 重要なデータ
    }
}
```

### キャッシュポリシー

```swift
struct CachePolicy {
    let maxAge: TimeInterval          // 最大キャッシュ期間
    let maxSize: Int                   // 最大キャッシュサイズ（バイト）
    let maxCount: Int                  // 最大アイテム数
    let evictionPolicy: EvictionPolicy // 削除ポリシー

    enum EvictionPolicy {
        case lru   // Least Recently Used
        case lfu   // Least Frequently Used
        case fifo  // First In First Out
        case ttl   // Time To Live
    }

    static let `default` = CachePolicy(
        maxAge: 3600,              // 1時間
        maxSize: 50 * 1024 * 1024, // 50MB
        maxCount: 100,
        evictionPolicy: .lru
    )

    static let aggressive = CachePolicy(
        maxAge: 86400,              // 24時間
        maxSize: 200 * 1024 * 1024, // 200MB
        maxCount: 500,
        evictionPolicy: .lru
    )

    static let conservative = CachePolicy(
        maxAge: 300,               // 5分
        maxSize: 10 * 1024 * 1024, // 10MB
        maxCount: 50,
        evictionPolicy: .ttl
    )
}
```

## メモリキャッシュ

### NSCacheベースの実装

```swift
class MemoryCache<Key: Hashable, Value> {
    private let cache = NSCache<WrappedKey, Entry>()
    private let dateProvider: () -> Date
    private let entryLifetime: TimeInterval

    init(
        maximumEntryCount: Int = 50,
        maximumTotalCost: Int = 50 * 1024 * 1024,
        entryLifetime: TimeInterval = 3600,
        dateProvider: @escaping () -> Date = Date.init
    ) {
        self.dateProvider = dateProvider
        self.entryLifetime = entryLifetime

        cache.countLimit = maximumEntryCount
        cache.totalCostLimit = maximumTotalCost
    }

    func insert(_ value: Value, forKey key: Key) {
        let date = dateProvider().addingTimeInterval(entryLifetime)
        let entry = Entry(value: value, expirationDate: date)
        cache.setObject(entry, forKey: WrappedKey(key))
    }

    func value(forKey key: Key) -> Value? {
        guard let entry = cache.object(forKey: WrappedKey(key)) else {
            return nil
        }

        guard dateProvider() < entry.expirationDate else {
            removeValue(forKey: key)
            return nil
        }

        return entry.value
    }

    func removeValue(forKey key: Key) {
        cache.removeObject(forKey: WrappedKey(key))
    }

    func removeAllValues() {
        cache.removeAllObjects()
    }

    // MARK: - Supporting Types
    private final class WrappedKey: NSObject {
        let key: Key

        init(_ key: Key) {
            self.key = key
        }

        override var hash: Int {
            key.hashValue
        }

        override func isEqual(_ object: Any?) -> Bool {
            guard let other = object as? WrappedKey else {
                return false
            }
            return key == other.key
        }
    }

    private final class Entry {
        let value: Value
        let expirationDate: Date

        init(value: Value, expirationDate: Date) {
            self.value = value
            self.expirationDate = expirationDate
        }
    }
}

// 使用例
class ImageMemoryCache {
    private let cache = MemoryCache<URL, UIImage>(
        maximumEntryCount: 100,
        maximumTotalCost: 50 * 1024 * 1024,
        entryLifetime: 3600
    )

    func insert(_ image: UIImage, for url: URL) {
        cache.insert(image, forKey: url)
    }

    func image(for url: URL) -> UIImage? {
        cache.value(forKey: url)
    }

    func clear() {
        cache.removeAllValues()
    }
}
```

### Actorベースのスレッドセーフキャッシュ

```swift
actor ThreadSafeCache<Key: Hashable, Value> {
    private var storage: [Key: CacheEntry] = [:]
    private let policy: CachePolicy

    struct CacheEntry {
        let value: Value
        let timestamp: Date
        var accessCount: Int
    }

    init(policy: CachePolicy = .default) {
        self.policy = policy
    }

    func insert(_ value: Value, forKey key: Key) {
        let entry = CacheEntry(value: value, timestamp: Date(), accessCount: 0)
        storage[key] = entry

        Task {
            await enforcePolicy()
        }
    }

    func value(forKey key: Key) -> Value? {
        guard var entry = storage[key] else {
            return nil
        }

        // TTLチェック
        if Date().timeIntervalSince(entry.timestamp) > policy.maxAge {
            storage.removeValue(forKey: key)
            return nil
        }

        // アクセスカウント更新
        entry.accessCount += 1
        storage[key] = entry

        return entry.value
    }

    func removeValue(forKey key: Key) {
        storage.removeValue(forKey: key)
    }

    func removeAll() {
        storage.removeAll()
    }

    private func enforcePolicy() {
        // サイズ制限を超えた場合、削除ポリシーに従って削除
        while storage.count > policy.maxCount {
            switch policy.evictionPolicy {
            case .lru:
                evictLRU()
            case .lfu:
                evictLFU()
            case .fifo:
                evictFIFO()
            case .ttl:
                evictExpired()
            }
        }
    }

    private func evictLRU() {
        guard let oldestKey = storage.min(by: { $0.value.timestamp < $1.value.timestamp })?.key else {
            return
        }
        storage.removeValue(forKey: oldestKey)
    }

    private func evictLFU() {
        guard let leastUsedKey = storage.min(by: { $0.value.accessCount < $1.value.accessCount })?.key else {
            return
        }
        storage.removeValue(forKey: leastUsedKey)
    }

    private func evictFIFO() {
        guard let firstKey = storage.keys.first else {
            return
        }
        storage.removeValue(forKey: firstKey)
    }

    private func evictExpired() {
        let now = Date()
        storage = storage.filter { now.timeIntervalSince($0.value.timestamp) <= policy.maxAge }
    }
}
```

## ディスクキャッシュ

### ファイルベースのディスクキャッシュ

```swift
class DiskCache {
    private let fileManager = FileManager.default
    private let cacheDirectory: URL
    private let policy: CachePolicy
    private let queue = DispatchQueue(label: "com.app.diskcache", qos: .utility)

    init(name: String = "DiskCache", policy: CachePolicy = .default) {
        let cacheURL = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        self.cacheDirectory = cacheURL.appendingPathComponent(name)
        self.policy = policy

        createCacheDirectoryIfNeeded()
        Task {
            await cleanupExpiredCache()
        }
    }

    // MARK: - Write
    func save(_ data: Data, forKey key: String) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            queue.async {
                do {
                    let fileURL = self.fileURL(forKey: key)
                    try data.write(to: fileURL, options: .atomic)

                    // メタデータを保存
                    try self.saveMetadata(forKey: key, size: data.count)

                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - Read
    func data(forKey key: String) async -> Data? {
        await withCheckedContinuation { continuation in
            queue.async {
                do {
                    // メタデータチェック
                    guard let metadata = try? self.loadMetadata(forKey: key) else {
                        continuation.resume(returning: nil)
                        return
                    }

                    // 期限チェック
                    if Date().timeIntervalSince(metadata.timestamp) > self.policy.maxAge {
                        try? self.remove(forKey: key)
                        continuation.resume(returning: nil)
                        return
                    }

                    // データ読み込み
                    let fileURL = self.fileURL(forKey: key)
                    let data = try Data(contentsOf: fileURL)

                    // アクセス時刻を更新
                    try? self.updateAccessTime(forKey: key)

                    continuation.resume(returning: data)
                } catch {
                    continuation.resume(returning: nil)
                }
            }
        }
    }

    // MARK: - Delete
    func remove(forKey key: String) throws {
        let fileURL = fileURL(forKey: key)
        try fileManager.removeItem(at: fileURL)
        try removeMetadata(forKey: key)
    }

    func removeAll() throws {
        try fileManager.removeItem(at: cacheDirectory)
        createCacheDirectoryIfNeeded()
    }

    // MARK: - Metadata
    private struct CacheMetadata: Codable {
        let key: String
        let size: Int
        let timestamp: Date
        var lastAccessTime: Date
    }

    private func saveMetadata(forKey key: String, size: Int) throws {
        let metadata = CacheMetadata(
            key: key,
            size: size,
            timestamp: Date(),
            lastAccessTime: Date()
        )

        let data = try JSONEncoder().encode(metadata)
        let metadataURL = metadataURL(forKey: key)
        try data.write(to: metadataURL)
    }

    private func loadMetadata(forKey key: String) throws -> CacheMetadata {
        let metadataURL = metadataURL(forKey: key)
        let data = try Data(contentsOf: metadataURL)
        return try JSONDecoder().decode(CacheMetadata.self, from: data)
    }

    private func removeMetadata(forKey key: String) throws {
        let metadataURL = metadataURL(forKey: key)
        try fileManager.removeItem(at: metadataURL)
    }

    private func updateAccessTime(forKey key: String) throws {
        var metadata = try loadMetadata(forKey: key)
        metadata.lastAccessTime = Date()
        let data = try JSONEncoder().encode(metadata)
        try data.write(to: metadataURL(forKey: key))
    }

    // MARK: - Cleanup
    private func cleanupExpiredCache() async {
        do {
            let contents = try fileManager.contentsOfDirectory(
                at: cacheDirectory,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: .skipsHiddenFiles
            )

            for url in contents where url.pathExtension == "meta" {
                let key = url.deletingPathExtension().lastPathComponent

                if let metadata = try? loadMetadata(forKey: key),
                   Date().timeIntervalSince(metadata.timestamp) > policy.maxAge {
                    try? remove(forKey: key)
                }
            }
        } catch {
            print("Cleanup failed: \(error)")
        }
    }

    // MARK: - Helper Methods
    private func createCacheDirectoryIfNeeded() {
        if !fileManager.fileExists(atPath: cacheDirectory.path) {
            try? fileManager.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        }
    }

    private func fileURL(forKey key: String) -> URL {
        let filename = key.addingPercentEncoding(withAllowedCharacters: .alphanumerics) ?? key
        return cacheDirectory.appendingPathComponent(filename)
    }

    private func metadataURL(forKey key: String) -> URL {
        fileURL(forKey: key).appendingPathExtension("meta")
    }
}
```

## HTTPキャッシュ

### URLCache の設定

```swift
class HTTPCacheManager {
    static func configure() {
        let memoryCapacity = 50 * 1024 * 1024  // 50 MB
        let diskCapacity = 200 * 1024 * 1024   // 200 MB

        let cache = URLCache(
            memoryCapacity: memoryCapacity,
            diskCapacity: diskCapacity,
            diskPath: "http_cache"
        )

        URLCache.shared = cache
    }

    static func clearCache() {
        URLCache.shared.removeAllCachedResponses()
    }

    static func clearCache(for url: URL) {
        guard let request = try? URLRequest(url: url, method: .get) else {
            return
        }
        URLCache.shared.removeCachedResponse(for: request)
    }
}

// URLSessionConfigurationでのキャッシュ設定
extension URLSessionConfiguration {
    static var cachedConfiguration: URLSessionConfiguration {
        let config = URLSessionConfiguration.default
        config.requestCachePolicy = .returnCacheDataElseLoad
        config.urlCache = URLCache.shared
        return config
    }

    static var noCacheConfiguration: URLSessionConfiguration {
        let config = URLSessionConfiguration.default
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        config.urlCache = nil
        return config
    }
}
```

### カスタムキャッシュポリシー

```swift
class CachedAPIService {
    private let session: URLSession
    private let cache: URLCache

    init() {
        let config = URLSessionConfiguration.default
        config.requestCachePolicy = .useProtocolCachePolicy

        self.cache = URLCache(
            memoryCapacity: 50 * 1024 * 1024,
            diskCapacity: 200 * 1024 * 1024
        )
        config.urlCache = cache

        self.session = URLSession(configuration: config)
    }

    func request<T: Decodable>(
        _ endpoint: Endpoint,
        cachePolicy: CacheStrategy = .cacheFirst
    ) async throws -> T {
        let request = try endpoint.makeRequest()

        switch cachePolicy {
        case .networkOnly:
            return try await fetchFromNetwork(request)

        case .cacheOnly:
            return try await fetchFromCache(request)

        case .cacheFirst:
            if let cached = try? await fetchFromCache(request) {
                return cached
            }
            return try await fetchFromNetwork(request)

        case .networkFirst:
            do {
                return try await fetchFromNetwork(request)
            } catch {
                return try await fetchFromCache(request)
            }

        case .cacheAndNetwork:
            // キャッシュを即座に返し、バックグラウンドで更新
            Task {
                try? await fetchFromNetwork(request)
            }
            return try await fetchFromCache(request)

        case .cacheWithTimeout:
            return try await fetchWithTimeout(request)
        }
    }

    private func fetchFromNetwork<T: Decodable>(_ request: URLRequest) async throws -> T {
        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw NetworkError.invalidResponse
        }

        return try JSONDecoder().decode(T.self, from: data)
    }

    private func fetchFromCache<T: Decodable>(_ request: URLRequest) async throws -> T {
        guard let cachedResponse = cache.cachedResponse(for: request) else {
            throw CacheError.notFound
        }

        return try JSONDecoder().decode(T.self, from: cachedResponse.data)
    }

    private func fetchWithTimeout<T: Decodable>(_ request: URLRequest) async throws -> T {
        if let cachedResponse = cache.cachedResponse(for: request) {
            // キャッシュの鮮度をチェック
            if let httpResponse = cachedResponse.response as? HTTPURLResponse,
               let cacheControl = httpResponse.allHeaderFields["Cache-Control"] as? String {
                // Cache-Controlヘッダーを解析して期限をチェック
                if isCacheValid(cacheControl: cacheControl, response: cachedResponse) {
                    return try JSONDecoder().decode(T.self, from: cachedResponse.data)
                }
            }
        }

        return try await fetchFromNetwork(request)
    }

    private func isCacheValid(cacheControl: String, response: CachedURLResponse) -> Bool {
        // Cache-Control: max-age=3600 の解析
        let components = cacheControl.components(separatedBy: "=")
        guard components.count == 2,
              let maxAge = TimeInterval(components[1]) else {
            return false
        }

        let cacheDate = response.userInfo?["cacheDate"] as? Date ?? Date()
        return Date().timeIntervalSince(cacheDate) < maxAge
    }
}

enum CacheError: Error {
    case notFound
    case expired
}
```

## 画像キャッシュ

### 2層キャッシュシステム

```swift
actor ImageCacheManager {
    static let shared = ImageCacheManager()

    private let memoryCache: MemoryCache<URL, UIImage>
    private let diskCache: DiskCache

    private var inProgressTasks: [URL: Task<UIImage, Error>] = [:]

    init() {
        self.memoryCache = MemoryCache(
            maximumEntryCount: 100,
            maximumTotalCost: 50 * 1024 * 1024,
            entryLifetime: 3600
        )
        self.diskCache = DiskCache(name: "ImageCache")

        setupMemoryWarningObserver()
    }

    // MARK: - Fetch Image
    func image(for url: URL) async throws -> UIImage {
        // メモリキャッシュをチェック
        if let cached = memoryCache.value(forKey: url) {
            return cached
        }

        // ディスクキャッシュをチェック
        if let data = await diskCache.data(forKey: url.absoluteString),
           let image = UIImage(data: data) {
            memoryCache.insert(image, forKey: url)
            return image
        }

        // 進行中のタスクをチェック
        if let task = inProgressTasks[url] {
            return try await task.value
        }

        // ダウンロード
        let task = Task {
            try await downloadImage(from: url)
        }

        inProgressTasks[url] = task

        do {
            let image = try await task.value
            inProgressTasks.removeValue(forKey: url)
            return image
        } catch {
            inProgressTasks.removeValue(forKey: url)
            throw error
        }
    }

    private func downloadImage(from url: URL) async throws -> UIImage {
        let (data, response) = try await URLSession.shared.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw ImageError.downloadFailed
        }

        guard let image = UIImage(data: data) else {
            throw ImageError.invalidData
        }

        // キャッシュに保存
        memoryCache.insert(image, forKey: url)
        try? await diskCache.save(data, forKey: url.absoluteString)

        return image
    }

    // MARK: - Prefetch
    func prefetch(urls: [URL]) {
        Task {
            for url in urls {
                try? await image(for: url)
            }
        }
    }

    // MARK: - Clear
    func clearMemoryCache() {
        memoryCache.removeAllValues()
    }

    func clearDiskCache() {
        try? diskCache.removeAll()
    }

    func clearAll() {
        clearMemoryCache()
        clearDiskCache()
    }

    // MARK: - Memory Warning
    private func setupMemoryWarningObserver() {
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task {
                await self?.clearMemoryCache()
            }
        }
    }
}

enum ImageError: Error {
    case downloadFailed
    case invalidData
}

// SwiftUI Image View
struct CachedAsyncImage: View {
    let url: URL
    @State private var image: UIImage?
    @State private var isLoading = false

    var body: some View {
        Group {
            if let image = image {
                Image(uiImage: image)
                    .resizable()
            } else if isLoading {
                ProgressView()
            } else {
                Color.gray
            }
        }
        .task {
            await loadImage()
        }
    }

    private func loadImage() async {
        isLoading = true
        defer { isLoading = false }

        do {
            image = try await ImageCacheManager.shared.image(for: url)
        } catch {
            print("Failed to load image: \(error)")
        }
    }
}
```

## データキャッシュ

### APIレスポンスキャッシュ

```swift
class APIResponseCache {
    private let diskCache = DiskCache(name: "APICache")

    func cacheResponse<T: Codable>(_ response: T, for endpoint: Endpoint) async throws {
        let key = cacheKey(for: endpoint)
        let data = try JSONEncoder().encode(response)
        try await diskCache.save(data, forKey: key)
    }

    func cachedResponse<T: Codable>(for endpoint: Endpoint) async -> T? {
        let key = cacheKey(for: endpoint)
        guard let data = await diskCache.data(forKey: key) else {
            return nil
        }

        return try? JSONDecoder().decode(T.self, from: data)
    }

    func removeCachedResponse(for endpoint: Endpoint) throws {
        let key = cacheKey(for: endpoint)
        try diskCache.remove(forKey: key)
    }

    private func cacheKey(for endpoint: Endpoint) -> String {
        let request = try? endpoint.makeRequest()
        let url = request?.url?.absoluteString ?? ""
        let method = request?.httpMethod ?? ""
        return "\(method)_\(url)".sha256Hash()
    }
}

extension String {
    func sha256Hash() -> String {
        // 簡易ハッシュ実装（実際はCryptoKitを使用）
        return self.data(using: .utf8)?.base64EncodedString() ?? self
    }
}

// 使用例
class CachedUserRepository: UserRepository {
    private let apiService: APIService
    private let cache = APIResponseCache()

    init(apiService: APIService) {
        self.apiService = apiService
    }

    func fetchUser(id: Int) async throws -> User {
        let endpoint = UserEndpoint.getUser(id: id)

        // キャッシュをチェック
        if let cached: User = await cache.cachedResponse(for: endpoint) {
            return cached
        }

        // ネットワークから取得
        let user: User = try await apiService.request(endpoint)

        // キャッシュに保存
        try? await cache.cacheResponse(user, for: endpoint)

        return user
    }
}
```

## キャッシュ無効化戦略

### Time-Based Invalidation

```swift
class TimedCache<Key: Hashable, Value> {
    private var storage: [Key: CachedValue] = [:]
    private let defaultTTL: TimeInterval

    struct CachedValue {
        let value: Value
        let expirationDate: Date
    }

    init(defaultTTL: TimeInterval = 3600) {
        self.defaultTTL = defaultTTL
    }

    func insert(_ value: Value, forKey key: Key, ttl: TimeInterval? = nil) {
        let expiration = Date().addingTimeInterval(ttl ?? defaultTTL)
        storage[key] = CachedValue(value: value, expirationDate: expiration)
    }

    func value(forKey key: Key) -> Value? {
        guard let cached = storage[key] else {
            return nil
        }

        if Date() > cached.expirationDate {
            storage.removeValue(forKey: key)
            return nil
        }

        return cached.value
    }

    func invalidate(forKey key: Key) {
        storage.removeValue(forKey: key)
    }

    func invalidateAll() {
        storage.removeAll()
    }

    func cleanup() {
        let now = Date()
        storage = storage.filter { now < $0.value.expirationDate }
    }
}
```

### Event-Based Invalidation

```swift
class EventBasedCache {
    private let diskCache = DiskCache()
    private var cacheInvalidationRules: [String: [CacheInvalidationRule]] = [:]

    struct CacheInvalidationRule {
        let event: String
        let keys: [String]
    }

    func registerInvalidationRule(event: String, invalidatingKeys keys: [String]) {
        var rules = cacheInvalidationRules[event] ?? []
        rules.append(CacheInvalidationRule(event: event, keys: keys))
        cacheInvalidationRules[event] = rules
    }

    func triggerEvent(_ event: String) async {
        guard let rules = cacheInvalidationRules[event] else {
            return
        }

        for rule in rules {
            for key in rule.keys {
                try? diskCache.remove(forKey: key)
            }
        }
    }
}

// 使用例
class UserService {
    private let cache = EventBasedCache()

    init() {
        // ユーザー更新時にキャッシュを無効化
        cache.registerInvalidationRule(
            event: "user.updated",
            invalidatingKeys: ["users.list", "user.profile"]
        )
    }

    func updateUser() async {
        // ユーザー更新処理
        await cache.triggerEvent("user.updated")
    }
}
```

## オフライン対応

### ネットワーク監視

```swift
import Network

class NetworkMonitor: ObservableObject {
    static let shared = NetworkMonitor()

    @Published var isConnected = true
    @Published var connectionType: ConnectionType = .unknown

    private let monitor = NWPathMonitor()
    private let queue = DispatchQueue(label: "NetworkMonitor")

    enum ConnectionType {
        case wifi
        case cellular
        case ethernet
        case unknown
    }

    private init() {
        monitor.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async {
                self?.isConnected = path.status == .satisfied
                self?.updateConnectionType(path)
            }
        }
        monitor.start(queue: queue)
    }

    private func updateConnectionType(_ path: NWPath) {
        if path.usesInterfaceType(.wifi) {
            connectionType = .wifi
        } else if path.usesInterfaceType(.cellular) {
            connectionType = .cellular
        } else if path.usesInterfaceType(.wiredEthernet) {
            connectionType = .ethernet
        } else {
            connectionType = .unknown
        }
    }
}
```

### オフラインファーストリポジトリ

```swift
class OfflineFirstRepository: UserRepository {
    private let remoteAPI: APIService
    private let localCache: DiskCache
    private let networkMonitor = NetworkMonitor.shared

    init(remoteAPI: APIService) {
        self.remoteAPI = remoteAPI
        self.localCache = DiskCache(name: "OfflineCache")
    }

    func fetchUser(id: Int) async throws -> User {
        let cacheKey = "user_\(id)"

        // オフラインの場合は即座にキャッシュを返す
        if !networkMonitor.isConnected {
            if let data = await localCache.data(forKey: cacheKey),
               let user = try? JSONDecoder().decode(User.self, from: data) {
                return user
            }
            throw NetworkError.networkUnavailable
        }

        // キャッシュを先に返し、バックグラウンドで更新
        if let data = await localCache.data(forKey: cacheKey),
           let cachedUser = try? JSONDecoder().decode(User.self, from: data) {

            Task {
                try? await updateCache(id: id, cacheKey: cacheKey)
            }

            return cachedUser
        }

        // キャッシュがない場合はネットワークから取得
        return try await updateCache(id: id, cacheKey: cacheKey)
    }

    private func updateCache(id: Int, cacheKey: String) async throws -> User {
        let user: User = try await remoteAPI.request(UserEndpoint.getUser(id: id))

        // キャッシュに保存
        if let data = try? JSONEncoder().encode(user) {
            try? await localCache.save(data, forKey: cacheKey)
        }

        return user
    }
}
```

## パフォーマンス最適化

### プリフェッチ

```swift
class PrefetchManager {
    private let imageCache = ImageCacheManager.shared

    func prefetchImages(for items: [Item]) {
        let urls = items.compactMap { $0.imageURL }
        imageCache.prefetch(urls: urls)
    }

    func prefetchNextPage(currentIndex: Int, items: [Item], threshold: Int = 10) {
        guard currentIndex >= items.count - threshold else {
            return
        }

        // 次のページを取得してプリフェッチ
        Task {
            // 次のページのデータを取得
            // プリフェッチ実行
        }
    }
}

struct Item {
    let imageURL: URL?
}
```

### バックグラウンドキャッシュ更新

```swift
class BackgroundCacheUpdater {
    func scheduleBackgroundRefresh() {
        // Background App Refreshを使用してキャッシュを更新
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: "com.app.cache.refresh",
            using: nil
        ) { task in
            self.handleCacheRefresh(task: task as! BGAppRefreshTask)
        }
    }

    private func handleCacheRefresh(task: BGAppRefreshTask) {
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1

        task.expirationHandler = {
            queue.cancelAllOperations()
        }

        let operation = CacheRefreshOperation()
        operation.completionBlock = {
            task.setTaskCompleted(success: !operation.isCancelled)
        }

        queue.addOperation(operation)
    }
}

class CacheRefreshOperation: Operation {
    override func main() {
        guard !isCancelled else { return }

        // キャッシュ更新処理
    }
}
```

## よくある問題と解決策

### 問題1: メモリリーク

```swift
// ❌ 強参照サイクル
class BadImageCache {
    var images: [URL: UIImage] = [:]  // メモリリーク!

    func insert(_ image: UIImage, for url: URL) {
        images[url] = image
    }
}

// ✅ NSCacheを使用
class GoodImageCache {
    private let cache = NSCache<NSString, UIImage>()

    init() {
        cache.countLimit = 100
        cache.totalCostLimit = 50 * 1024 * 1024
    }

    func insert(_ image: UIImage, for url: URL) {
        cache.setObject(image, forKey: url.absoluteString as NSString)
    }
}
```

### 問題2: キャッシュの一貫性

```swift
// ✅ バージョン管理
class VersionedCache {
    private let cache = DiskCache()
    private let currentVersion = "1.0"

    func save<T: Codable>(_ value: T, forKey key: String) async throws {
        let versionedKey = "\(currentVersion)_\(key)"
        let data = try JSONEncoder().encode(value)
        try await cache.save(data, forKey: versionedKey)
    }

    func load<T: Codable>(forKey key: String) async -> T? {
        let versionedKey = "\(currentVersion)_\(key)"
        guard let data = await cache.data(forKey: versionedKey) else {
            return nil
        }
        return try? JSONDecoder().decode(T.self, from: data)
    }
}
```

---

**関連ガイド:**
- [01-networking.md](./01-networking.md) - ネットワーク通信
- [02-data-persistence.md](./02-data-persistence.md) - データ永続化

**関連Skills:**
- [ios-development](../../ios-development/SKILL.md) - iOS開発全般
- [frontend-performance](../../frontend-performance/SKILL.md) - パフォーマンス最適化

**参考資料:**
- [URLCache - Apple Developer](https://developer.apple.com/documentation/foundation/urlcache)
- [NSCache - Apple Developer](https://developer.apple.com/documentation/foundation/nscache)

**更新履歴:**
- 2025-12-31: 初版作成

# Networking - iOS API通信完全ガイド

## 📋 目次

1. [概要](#概要)
2. [URLSession基礎](#urlsession基礎)
3. [API通信アーキテクチャ](#api通信アーキテクチャ)
4. [Endpointパターン](#endpointパターン)
5. [エラーハンドリング](#エラーハンドリング)
6. [認証とトークン管理](#認証とトークン管理)
7. [リクエストのキャンセルと優先度制御](#リクエストのキャンセルと優先度制御)
8. [WebSocket通信](#websocket通信)
9. [アップロード・ダウンロード](#アップロード・ダウンロード)
10. [テストとモック](#テストとモック)
11. [パフォーマンス最適化](#パフォーマンス最適化)
12. [よくある問題と解決策](#よくある問題と解決策)

## 概要

iOSアプリにおけるネットワーク通信は、ユーザー体験の質を左右する重要な要素です。このガイドでは、URLSessionを使った型安全で保守性の高いAPI通信の実装方法を解説します。

### このガイドの対象者

- iOSエンジニア
- API通信を実装する開発者
- アーキテクチャ設計を学びたい方

### 学べること

- 型安全なAPI通信の実装
- エラーハンドリングのベストプラクティス
- WebSocketを使ったリアルタイム通信
- テスタブルなネットワーク層の設計

## URLSession基礎

### シンプルなGETリクエスト

```swift
struct User: Codable {
    let id: Int
    let name: String
    let email: String
}

// 最もシンプルな実装
func fetchUser(id: Int) async throws -> User {
    let url = URL(string: "https://api.example.com/users/\(id)")!
    let (data, _) = try await URLSession.shared.data(from: url)
    return try JSONDecoder().decode(User.self, from: data)
}

// エラーハンドリングを追加
func fetchUserWithValidation(id: Int) async throws -> User {
    guard let url = URL(string: "https://api.example.com/users/\(id)") else {
        throw NetworkError.invalidURL
    }

    let (data, response) = try await URLSession.shared.data(from: url)

    guard let httpResponse = response as? HTTPURLResponse else {
        throw NetworkError.invalidResponse
    }

    guard (200...299).contains(httpResponse.statusCode) else {
        throw NetworkError.httpError(httpResponse.statusCode)
    }

    do {
        return try JSONDecoder().decode(User.self, from: data)
    } catch {
        throw NetworkError.decodingError(error)
    }
}

enum NetworkError: Error, LocalizedError {
    case invalidURL
    case invalidResponse
    case httpError(Int)
    case decodingError(Error)

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .invalidResponse:
            return "Invalid response from server"
        case .httpError(let code):
            return "HTTP error: \(code)"
        case .decodingError(let error):
            return "Decoding failed: \(error.localizedDescription)"
        }
    }
}
```

### POSTリクエスト

```swift
struct CreateUserRequest: Codable {
    let name: String
    let email: String
    let password: String
}

func createUser(request: CreateUserRequest) async throws -> User {
    guard let url = URL(string: "https://api.example.com/users") else {
        throw NetworkError.invalidURL
    }

    var urlRequest = URLRequest(url: url)
    urlRequest.httpMethod = "POST"
    urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")

    do {
        urlRequest.httpBody = try JSONEncoder().encode(request)
    } catch {
        throw NetworkError.encodingError(error)
    }

    let (data, response) = try await URLSession.shared.data(for: urlRequest)

    guard let httpResponse = response as? HTTPURLResponse,
          (200...299).contains(httpResponse.statusCode) else {
        throw NetworkError.invalidResponse
    }

    return try JSONDecoder().decode(User.self, from: data)
}
```

### URLSessionConfigurationのカスタマイズ

```swift
class NetworkManager {
    static let shared = NetworkManager()

    private let session: URLSession

    private init() {
        let configuration = URLSessionConfiguration.default

        // タイムアウト設定
        configuration.timeoutIntervalForRequest = 30  // リクエストタイムアウト
        configuration.timeoutIntervalForResource = 300  // リソース全体のタイムアウト

        // 接続性の設定
        configuration.waitsForConnectivity = true  // 接続を待つ
        configuration.allowsCellularAccess = true  // セルラーアクセスを許可

        // キャッシュ設定
        configuration.requestCachePolicy = .returnCacheDataElseLoad

        // HTTPヘッダー設定
        configuration.httpAdditionalHeaders = [
            "Accept": "application/json",
            "User-Agent": "MyApp/1.0"
        ]

        // Cookie設定
        configuration.httpCookieAcceptPolicy = .always

        // HTTP/2設定
        configuration.httpMaximumConnectionsPerHost = 5

        self.session = URLSession(configuration: configuration)
    }

    func request<T: Decodable>(url: URL) async throws -> T {
        let (data, response) = try await session.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw NetworkError.invalidResponse
        }

        return try JSONDecoder().decode(T.self, from: data)
    }
}
```

## API通信アーキテクチャ

### レイヤー構造

```swift
// MARK: - Domain Layer (Models)
struct User: Codable, Identifiable {
    let id: Int
    let name: String
    let email: String
    let createdAt: Date
}

// MARK: - Data Layer (Repository Protocol)
protocol UserRepository {
    func fetchUser(id: Int) async throws -> User
    func fetchAllUsers() async throws -> [User]
    func createUser(_ request: CreateUserRequest) async throws -> User
    func updateUser(id: Int, _ request: UpdateUserRequest) async throws -> User
    func deleteUser(id: Int) async throws
}

// MARK: - Data Layer (API Service)
protocol APIService {
    func request<T: Decodable>(_ endpoint: Endpoint) async throws -> T
}

class APIServiceImpl: APIService {
    private let session: URLSession

    init(session: URLSession = .shared) {
        self.session = session
    }

    func request<T: Decodable>(_ endpoint: Endpoint) async throws -> T {
        let request = try endpoint.makeRequest()
        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw NetworkError.invalidResponse
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            throw NetworkError.httpError(httpResponse.statusCode)
        }

        do {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode(T.self, from: data)
        } catch {
            throw NetworkError.decodingError(error)
        }
    }
}

// MARK: - Data Layer (Repository Implementation)
class UserRepositoryImpl: UserRepository {
    private let apiService: APIService

    init(apiService: APIService) {
        self.apiService = apiService
    }

    func fetchUser(id: Int) async throws -> User {
        try await apiService.request(UserEndpoint.getUser(id: id))
    }

    func fetchAllUsers() async throws -> [User] {
        try await apiService.request(UserEndpoint.getAllUsers)
    }

    func createUser(_ request: CreateUserRequest) async throws -> User {
        try await apiService.request(UserEndpoint.createUser(request))
    }

    func updateUser(id: Int, _ request: UpdateUserRequest) async throws -> User {
        try await apiService.request(UserEndpoint.updateUser(id: id, request))
    }

    func deleteUser(id: Int) async throws {
        let _: EmptyResponse = try await apiService.request(UserEndpoint.deleteUser(id: id))
    }
}

struct EmptyResponse: Codable {}

// MARK: - Presentation Layer (ViewModel)
@MainActor
class UserListViewModel: ObservableObject {
    @Published var users: [User] = []
    @Published var isLoading = false
    @Published var error: Error?

    private let repository: UserRepository

    init(repository: UserRepository) {
        self.repository = repository
    }

    func loadUsers() async {
        isLoading = true
        defer { isLoading = false }

        do {
            users = try await repository.fetchAllUsers()
            error = nil
        } catch {
            self.error = error
        }
    }
}
```

## Endpointパターン

### Endpointプロトコルの定義

```swift
protocol Endpoint {
    var baseURL: String { get }
    var path: String { get }
    var method: HTTPMethod { get }
    var headers: [String: String]? { get }
    var queryItems: [URLQueryItem]? { get }
    var body: Data? { get }

    func makeRequest() throws -> URLRequest
}

enum HTTPMethod: String {
    case get = "GET"
    case post = "POST"
    case put = "PUT"
    case delete = "DELETE"
    case patch = "PATCH"
}

extension Endpoint {
    var baseURL: String {
        "https://api.example.com"
    }

    var headers: [String: String]? {
        ["Content-Type": "application/json"]
    }

    var queryItems: [URLQueryItem]? {
        nil
    }

    var body: Data? {
        nil
    }

    func makeRequest() throws -> URLRequest {
        var components = URLComponents(string: baseURL + path)
        components?.queryItems = queryItems

        guard let url = components?.url else {
            throw NetworkError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = method.rawValue
        request.httpBody = body

        headers?.forEach { key, value in
            request.setValue(value, forHTTPHeaderField: key)
        }

        return request
    }
}
```

### 具体的なEndpointの実装

```swift
enum UserEndpoint: Endpoint {
    case getAllUsers
    case getUser(id: Int)
    case createUser(CreateUserRequest)
    case updateUser(id: Int, UpdateUserRequest)
    case deleteUser(id: Int)
    case searchUsers(query: String, page: Int)

    var path: String {
        switch self {
        case .getAllUsers:
            return "/users"
        case .getUser(let id), .updateUser(let id, _), .deleteUser(let id):
            return "/users/\(id)"
        case .createUser:
            return "/users"
        case .searchUsers:
            return "/users/search"
        }
    }

    var method: HTTPMethod {
        switch self {
        case .getAllUsers, .getUser, .searchUsers:
            return .get
        case .createUser:
            return .post
        case .updateUser:
            return .put
        case .deleteUser:
            return .delete
        }
    }

    var queryItems: [URLQueryItem]? {
        switch self {
        case .searchUsers(let query, let page):
            return [
                URLQueryItem(name: "q", value: query),
                URLQueryItem(name: "page", value: "\(page)")
            ]
        default:
            return nil
        }
    }

    var body: Data? {
        switch self {
        case .createUser(let request):
            return try? JSONEncoder().encode(request)
        case .updateUser(_, let request):
            return try? JSONEncoder().encode(request)
        default:
            return nil
        }
    }
}

// 使用例
class UserService {
    private let apiService: APIService

    init(apiService: APIService = APIServiceImpl()) {
        self.apiService = apiService
    }

    func getUser(id: Int) async throws -> User {
        try await apiService.request(UserEndpoint.getUser(id: id))
    }

    func searchUsers(query: String, page: Int = 1) async throws -> [User] {
        try await apiService.request(UserEndpoint.searchUsers(query: query, page: page))
    }
}
```

## エラーハンドリング

### 包括的なエラー定義

```swift
enum NetworkError: Error, LocalizedError {
    case invalidURL
    case invalidResponse
    case httpError(Int, Data?)
    case decodingError(Error)
    case encodingError(Error)
    case noData
    case networkUnavailable
    case timeout
    case cancelled
    case unauthorized
    case forbidden
    case notFound
    case serverError(Int)
    case unknown(Error)

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "The URL is invalid"
        case .invalidResponse:
            return "Invalid response from server"
        case .httpError(let code, _):
            return "HTTP error: \(code)"
        case .decodingError(let error):
            return "Failed to decode response: \(error.localizedDescription)"
        case .encodingError(let error):
            return "Failed to encode request: \(error.localizedDescription)"
        case .noData:
            return "No data received from server"
        case .networkUnavailable:
            return "Network connection unavailable"
        case .timeout:
            return "Request timed out"
        case .cancelled:
            return "Request was cancelled"
        case .unauthorized:
            return "Unauthorized. Please login again."
        case .forbidden:
            return "Access forbidden"
        case .notFound:
            return "Resource not found"
        case .serverError(let code):
            return "Server error: \(code)"
        case .unknown(let error):
            return "Unknown error: \(error.localizedDescription)"
        }
    }

    var recoverySuggestion: String? {
        switch self {
        case .networkUnavailable:
            return "Please check your internet connection and try again."
        case .timeout:
            return "The request took too long. Please try again later."
        case .unauthorized:
            return "Please login again to continue."
        case .serverError:
            return "The server is experiencing issues. Please try again later."
        default:
            return nil
        }
    }
}

// HTTPステータスコードからエラーを生成
extension NetworkError {
    static func from(statusCode: Int, data: Data? = nil) -> NetworkError {
        switch statusCode {
        case 401:
            return .unauthorized
        case 403:
            return .forbidden
        case 404:
            return .notFound
        case 408:
            return .timeout
        case 500...599:
            return .serverError(statusCode)
        default:
            return .httpError(statusCode, data)
        }
    }
}
```

### エラーハンドリングの実装

```swift
class RobustAPIService: APIService {
    private let session: URLSession

    init(session: URLSession = .shared) {
        self.session = session
    }

    func request<T: Decodable>(_ endpoint: Endpoint) async throws -> T {
        let request = try endpoint.makeRequest()

        do {
            let (data, response) = try await session.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                throw NetworkError.invalidResponse
            }

            // ステータスコードチェック
            guard (200...299).contains(httpResponse.statusCode) else {
                throw NetworkError.from(statusCode: httpResponse.statusCode, data: data)
            }

            // データの存在チェック
            guard !data.isEmpty else {
                throw NetworkError.noData
            }

            // デコード
            do {
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                decoder.keyDecodingStrategy = .convertFromSnakeCase
                return try decoder.decode(T.self, from: data)
            } catch {
                throw NetworkError.decodingError(error)
            }

        } catch let error as NetworkError {
            throw error
        } catch let error as URLError {
            throw mapURLError(error)
        } catch {
            throw NetworkError.unknown(error)
        }
    }

    private func mapURLError(_ error: URLError) -> NetworkError {
        switch error.code {
        case .notConnectedToInternet, .networkConnectionLost:
            return .networkUnavailable
        case .timedOut:
            return .timeout
        case .cancelled:
            return .cancelled
        default:
            return .unknown(error)
        }
    }
}
```

### リトライロジック

```swift
func requestWithRetry<T: Decodable>(
    _ endpoint: Endpoint,
    maxRetries: Int = 3,
    retryDelay: TimeInterval = 2.0
) async throws -> T {
    var lastError: Error?

    for attempt in 0..<maxRetries {
        do {
            return try await apiService.request(endpoint)
        } catch let error as NetworkError {
            lastError = error

            // リトライすべきエラーかチェック
            guard shouldRetry(error: error, attempt: attempt, maxRetries: maxRetries) else {
                throw error
            }

            // 指数バックオフ
            let delay = retryDelay * pow(2.0, Double(attempt))
            try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))

        } catch {
            throw error
        }
    }

    throw lastError ?? NetworkError.unknown(NSError(domain: "Unknown", code: -1))
}

private func shouldRetry(error: NetworkError, attempt: Int, maxRetries: Int) -> Bool {
    guard attempt < maxRetries - 1 else { return false }

    switch error {
    case .timeout, .networkUnavailable:
        return true
    case .serverError(let code) where code >= 500:
        return true
    default:
        return false
    }
}
```

## 認証とトークン管理

### トークンベース認証

```swift
protocol AuthTokenProvider {
    func getAccessToken() async throws -> String
    func refreshToken() async throws -> String
}

class AuthManager: AuthTokenProvider {
    private var accessToken: String?
    private var refreshToken: String?
    private let keychainManager = KeychainManager.shared

    func getAccessToken() async throws -> String {
        if let token = accessToken {
            return token
        }

        // Keychainから取得
        if let token = try? keychainManager.loadToken() {
            accessToken = token
            return token
        }

        throw NetworkError.unauthorized
    }

    func refreshToken() async throws -> String {
        guard let refreshToken = self.refreshToken else {
            throw NetworkError.unauthorized
        }

        struct RefreshRequest: Codable {
            let refreshToken: String
        }

        struct RefreshResponse: Codable {
            let accessToken: String
            let refreshToken: String
        }

        let endpoint = AuthEndpoint.refreshToken(RefreshRequest(refreshToken: refreshToken))
        let response: RefreshResponse = try await APIServiceImpl().request(endpoint)

        // 新しいトークンを保存
        self.accessToken = response.accessToken
        self.refreshToken = response.refreshToken
        try? keychainManager.saveToken(response.accessToken)

        return response.accessToken
    }

    func logout() {
        accessToken = nil
        refreshToken = nil
        try? keychainManager.deleteToken()
    }
}

// 認証付きリクエスト
class AuthenticatedAPIService: APIService {
    private let session: URLSession
    private let authManager: AuthTokenProvider

    init(session: URLSession = .shared, authManager: AuthTokenProvider) {
        self.session = session
        self.authManager = authManager
    }

    func request<T: Decodable>(_ endpoint: Endpoint) async throws -> T {
        var request = try endpoint.makeRequest()

        // アクセストークンを追加
        let token = try await authManager.getAccessToken()
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw NetworkError.invalidResponse
        }

        // 401エラーの場合、トークンをリフレッシュして再試行
        if httpResponse.statusCode == 401 {
            let newToken = try await authManager.refreshToken()
            request.setValue("Bearer \(newToken)", forHTTPHeaderField: "Authorization")

            let (retryData, retryResponse) = try await session.data(for: request)
            guard let retryHttpResponse = retryResponse as? HTTPURLResponse,
                  (200...299).contains(retryHttpResponse.statusCode) else {
                throw NetworkError.unauthorized
            }

            return try JSONDecoder().decode(T.self, from: retryData)
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            throw NetworkError.from(statusCode: httpResponse.statusCode, data: data)
        }

        return try JSONDecoder().decode(T.self, from: data)
    }
}
```

## リクエストのキャンセルと優先度制御

### Taskのキャンセル

```swift
@MainActor
class SearchViewModel: ObservableObject {
    @Published var searchResults: [User] = []
    @Published var isSearching = false

    private var searchTask: Task<Void, Never>?
    private let repository: UserRepository

    init(repository: UserRepository) {
        self.repository = repository
    }

    func search(query: String) {
        // 既存の検索タスクをキャンセル
        searchTask?.cancel()

        guard !query.isEmpty else {
            searchResults = []
            return
        }

        searchTask = Task {
            isSearching = true
            defer { isSearching = false }

            // デバウンス
            try? await Task.sleep(nanoseconds: 300_000_000)

            guard !Task.isCancelled else { return }

            do {
                let results = try await repository.searchUsers(query: query)
                guard !Task.isCancelled else { return }

                searchResults = results
            } catch {
                if !Task.isCancelled {
                    print("Search failed: \(error)")
                }
            }
        }
    }

    deinit {
        searchTask?.cancel()
    }
}
```

### URLSessionTaskの優先度制御

```swift
class PriorityNetworkManager {
    enum Priority {
        case high
        case normal
        case low

        var urlSessionPriority: Float {
            switch self {
            case .high: return URLSessionTask.highPriority
            case .normal: return URLSessionTask.defaultPriority
            case .low: return URLSessionTask.lowPriority
            }
        }
    }

    func request<T: Decodable>(
        _ endpoint: Endpoint,
        priority: Priority = .normal
    ) async throws -> T {
        let request = try endpoint.makeRequest()
        let task = URLSession.shared.dataTask(with: request)
        task.priority = priority.urlSessionPriority

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                task.completionHandler = { data, response, error in
                    if let error = error {
                        continuation.resume(throwing: error)
                        return
                    }

                    guard let data = data,
                          let httpResponse = response as? HTTPURLResponse,
                          (200...299).contains(httpResponse.statusCode) else {
                        continuation.resume(throwing: NetworkError.invalidResponse)
                        return
                    }

                    do {
                        let decoded = try JSONDecoder().decode(T.self, from: data)
                        continuation.resume(returning: decoded)
                    } catch {
                        continuation.resume(throwing: NetworkError.decodingError(error))
                    }
                }
                task.resume()
            }
        } onCancel: {
            task.cancel()
        }
    }
}
```

## WebSocket通信

### WebSocketマネージャーの実装

```swift
actor WebSocketManager {
    enum State {
        case disconnected
        case connecting
        case connected
        case disconnecting
    }

    private(set) var state: State = .disconnected
    private var webSocketTask: URLSessionWebSocketTask?
    private let url: URL

    init(url: URL) {
        self.url = url
    }

    func connect() {
        guard state == .disconnected else { return }

        state = .connecting

        let session = URLSession(configuration: .default)
        webSocketTask = session.webSocketTask(with: url)
        webSocketTask?.resume()

        state = .connected

        Task {
            await startReceiving()
        }

        // ピング送信
        Task {
            await sendPing()
        }
    }

    func disconnect() {
        state = .disconnecting
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
        state = .disconnected
    }

    func send(_ message: String) async throws {
        guard state == .connected else {
            throw WebSocketError.notConnected
        }

        let message = URLSessionWebSocketTask.Message.string(message)
        try await webSocketTask?.send(message)
    }

    private func startReceiving() async {
        while state == .connected {
            do {
                guard let message = try await webSocketTask?.receive() else {
                    break
                }

                await handleMessage(message)
            } catch {
                print("WebSocket receive error: \(error)")
                break
            }
        }
    }

    private func handleMessage(_ message: URLSessionWebSocketTask.Message) async {
        switch message {
        case .string(let text):
            await onMessageReceived(text)
        case .data(let data):
            if let text = String(data: data, encoding: .utf8) {
                await onMessageReceived(text)
            }
        @unknown default:
            break
        }
    }

    private func sendPing() async {
        while state == .connected {
            try? await Task.sleep(nanoseconds: 30_000_000_000)  // 30秒ごと

            do {
                try await webSocketTask?.sendPing(pongReceiveHandler: { error in
                    if let error = error {
                        print("Ping failed: \(error)")
                    }
                })
            } catch {
                print("Ping error: \(error)")
            }
        }
    }

    private func onMessageReceived(_ message: String) async {
        // サブクラスでオーバーライド
    }
}

enum WebSocketError: Error {
    case notConnected
    case invalidMessage
}
```

### リアルタイムチャットの実装

```swift
@MainActor
class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var isConnected = false

    private let webSocket: WebSocketManager
    private let decoder = JSONDecoder()
    private let encoder = JSONEncoder()

    struct ChatMessage: Codable, Identifiable {
        let id: String
        let userId: String
        let username: String
        let text: String
        let timestamp: Date
    }

    init(url: URL) {
        self.webSocket = WebSocketManager(url: url)
    }

    func connect() async {
        await webSocket.connect()
        isConnected = true
    }

    func disconnect() async {
        await webSocket.disconnect()
        isConnected = false
    }

    func sendMessage(_ text: String) async {
        let message = ChatMessage(
            id: UUID().uuidString,
            userId: currentUserId,
            username: currentUsername,
            text: text,
            timestamp: Date()
        )

        guard let data = try? encoder.encode(message),
              let jsonString = String(data: data, encoding: .utf8) else {
            return
        }

        try? await webSocket.send(jsonString)
    }

    private func handleIncomingMessage(_ jsonString: String) {
        guard let data = jsonString.data(using: .utf8),
              let message = try? decoder.decode(ChatMessage.self, from: data) else {
            return
        }

        messages.append(message)
    }
}
```

(続く... 文字数制限のため)

## アップロード・ダウンロード

### ファイルアップロード

```swift
func uploadImage(_ image: UIImage) async throws -> UploadResponse {
    guard let imageData = image.jpegData(compressionQuality: 0.8) else {
        throw NetworkError.encodingError(NSError(domain: "Image", code: -1))
    }

    let url = URL(string: "https://api.example.com/upload")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"

    let boundary = UUID().uuidString
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

    var body = Data()

    // ファイルデータ
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"file\"; filename=\"image.jpg\"\r\n".data(using: .utf8)!)
    body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
    body.append(imageData)
    body.append("\r\n".data(using: .utf8)!)

    // メタデータ
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"description\"\r\n\r\n".data(using: .utf8)!)
    body.append("My uploaded image".data(using: .utf8)!)
    body.append("\r\n".data(using: .utf8)!)

    body.append("--\(boundary)--\r\n".data(using: .utf8)!)

    request.httpBody = body

    let (data, response) = try await URLSession.shared.data(for: request)

    guard let httpResponse = response as? HTTPURLResponse,
          (200...299).contains(httpResponse.statusCode) else {
        throw NetworkError.invalidResponse
    }

    return try JSONDecoder().decode(UploadResponse.self, from: data)
}

struct UploadResponse: Codable {
    let url: String
    let id: String
}
```

### 進捗表示付きダウンロード

```swift
class DownloadManager: NSObject, ObservableObject, URLSessionDownloadDelegate {
    @Published var progress: Double = 0
    @Published var isDownloading = false

    private var downloadTask: URLSessionDownloadTask?

    func download(from url: URL) async throws -> URL {
        isDownloading = true
        progress = 0

        let config = URLSessionConfiguration.default
        let session = URLSession(configuration: config, delegate: self, delegateQueue: nil)

        return try await withCheckedThrowingContinuation { continuation in
            downloadTask = session.downloadTask(with: url) { location, response, error in
                self.isDownloading = false

                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }

                guard let location = location else {
                    continuation.resume(throwing: NetworkError.noData)
                    return
                }

                continuation.resume(returning: location)
            }
            downloadTask?.resume()
        }
    }

    func cancelDownload() {
        downloadTask?.cancel()
        isDownloading = false
    }

    // URLSessionDownloadDelegate
    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        DispatchQueue.main.async {
            self.progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        // ファイルを永続的な場所に移動
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let destinationURL = documentsPath.appendingPathComponent(downloadTask.originalRequest?.url?.lastPathComponent ?? "download")

        try? FileManager.default.removeItem(at: destinationURL)
        try? FileManager.default.moveItem(at: location, to: destinationURL)
    }
}
```

## テストとモック

### Mockable Protocol

```swift
protocol HTTPClient {
    func data(for request: URLRequest) async throws -> (Data, URLResponse)
}

extension URLSession: HTTPClient {}

// モック実装
class MockHTTPClient: HTTPClient {
    var mockData: Data?
    var mockResponse: URLResponse?
    var mockError: Error?

    func data(for request: URLRequest) async throws -> (Data, URLResponse) {
        if let error = mockError {
            throw error
        }

        guard let data = mockData, let response = mockResponse else {
            throw NetworkError.noData
        }

        return (data, response)
    }
}

// テスタブルなAPIService
class TestableAPIService: APIService {
    private let httpClient: HTTPClient

    init(httpClient: HTTPClient = URLSession.shared) {
        self.httpClient = httpClient
    }

    func request<T: Decodable>(_ endpoint: Endpoint) async throws -> T {
        let request = try endpoint.makeRequest()
        let (data, response) = try await httpClient.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw NetworkError.invalidResponse
        }

        return try JSONDecoder().decode(T.self, from: data)
    }
}

// テスト
final class APIServiceTests: XCTestCase {
    func testSuccessfulRequest() async throws {
        // Arrange
        let mockClient = MockHTTPClient()
        let user = User(id: 1, name: "Test", email: "test@example.com")
        mockClient.mockData = try JSONEncoder().encode(user)
        mockClient.mockResponse = HTTPURLResponse(
            url: URL(string: "https://api.example.com")!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: nil
        )

        let service = TestableAPIService(httpClient: mockClient)

        // Act
        let result: User = try await service.request(UserEndpoint.getUser(id: 1))

        // Assert
        XCTAssertEqual(result.id, 1)
        XCTAssertEqual(result.name, "Test")
    }

    func testNetworkError() async {
        // Arrange
        let mockClient = MockHTTPClient()
        mockClient.mockError = NetworkError.networkUnavailable

        let service = TestableAPIService(httpClient: mockClient)

        // Act & Assert
        do {
            let _: User = try await service.request(UserEndpoint.getUser(id: 1))
            XCTFail("Should throw error")
        } catch {
            XCTAssertTrue(error is NetworkError)
        }
    }
}
```

## パフォーマンス最適化

### リクエストのバッチ処理

```swift
actor BatchRequestManager {
    private var pendingRequests: [String: Task<Data, Error>] = [:]

    func batchedRequest(url: URL) async throws -> Data {
        let key = url.absoluteString

        if let existingTask = pendingRequests[key] {
            return try await existingTask.value
        }

        let task = Task {
            let (data, _) = try await URLSession.shared.data(from: url)
            return data
        }

        pendingRequests[key] = task

        defer {
            pendingRequests.removeValue(forKey: key)
        }

        return try await task.value
    }
}
```

### HTTP/2 マルチプレクシング

```swift
let configuration = URLSessionConfiguration.default
configuration.httpMaximumConnectionsPerHost = 1
configuration.allowsExpensiveNetworkAccess = true
configuration.waitsForConnectivity = true

let session = URLSession(configuration: configuration)
```

## よくある問題と解決策

### 問題1: メモリリーク

```swift
// ❌ 強参照サイクル
class BadNetworkManager {
    var completion: ((Result<Data, Error>) -> Void)?

    func fetch() {
        URLSession.shared.dataTask(with: url) { data, response, error in
            self.completion?(.success(data!))  // selfへの強参照
        }.resume()
    }
}

// ✅ weak参照で解決
class GoodNetworkManager {
    var completion: ((Result<Data, Error>) -> Void)?

    func fetch() {
        URLSession.shared.dataTask(with: url) { [weak self] data, response, error in
            self?.completion?(.success(data!))
        }.resume()
    }
}
```

### 問題2: データ競合

```swift
// ✅ Actorで解決
actor NetworkCache {
    private var cache: [URL: Data] = [:]

    func data(for url: URL) -> Data? {
        cache[url]
    }

    func setData(_ data: Data, for url: URL) {
        cache[url] = data
    }
}
```

---

**関連ガイド:**
- [02-data-persistence.md](./02-data-persistence.md) - データ永続化
- [03-caching-strategy.md](./03-caching-strategy.md) - キャッシュ戦略

**関連Skills:**
- [ios-development](../../ios-development/SKILL.md) - iOS開発全般
- [ios-security](../../ios-security/SKILL.md) - セキュリティ実装
- [backend-development](../../backend-development/SKILL.md) - API設計

**参考資料:**
- [URLSession - Apple Developer](https://developer.apple.com/documentation/foundation/urlsession)
- [WWDC - Networking](https://developer.apple.com/videos/networking/)

**更新履歴:**
- 2025-12-31: 初版作成

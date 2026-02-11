# Data Security 完全ガイド

## 目次
1. [データセキュリティの基礎](#データセキュリティの基礎)
2. [安全なデータストレージ](#安全なデータストレージ)
3. [Core Data暗号化](#core-data暗号化)
4. [ファイル保護](#ファイル保護)
5. [ネットワーク通信の保護](#ネットワーク通信の保護)
6. [メモリセキュリティ](#メモリセキュリティ)
7. [データ漏洩防止](#データ漏洩防止)
8. [セキュアな削除](#セキュアな削除)

---

## データセキュリティの基礎

### データ分類とセキュリティレベル

```swift
/*
データ分類:

Level 1 - Public (公開)
- アプリ名、バージョン情報
- 保護不要

Level 2 - Internal (内部)
- ユーザー設定、キャッシュデータ
- UserDefaults可

Level 3 - Confidential (機密)
- 個人情報、トークン
- Keychain必須

Level 4 - Restricted (厳密な機密)
- パスワード、支払い情報
- Keychain + 暗号化必須
*/

enum DataClassification {
    case `public`
    case `internal`
    case confidential
    case restricted

    var storageStrategy: DataStorageStrategy {
        switch self {
        case .public:
            return .userDefaults
        case .internal:
            return .userDefaults
        case .confidential:
            return .keychain
        case .restricted:
            return .keychainEncrypted
        }
    }
}

enum DataStorageStrategy {
    case userDefaults
    case fileSystem
    case keychain
    case keychainEncrypted
    case coreData
    case coreDataEncrypted
}

// ❌ 悪い例: 機密情報をUserDefaultsに保存
class BadDataStorage {
    func saveUserData(email: String, password: String, creditCard: String) {
        UserDefaults.standard.set(email, forKey: "email")
        UserDefaults.standard.set(password, forKey: "password") // 絶対にNG!
        UserDefaults.standard.set(creditCard, forKey: "creditCard") // 絶対にNG!
    }
}

// ✅ 良い例: データ分類に応じた適切な保存
class SecureDataStorage {
    private let keychain: KeychainService
    private let encryption: EncryptionService

    init(keychain: KeychainService, encryption: EncryptionService) {
        self.keychain = keychain
        self.encryption = encryption
    }

    func saveUserData(email: String, password: String, creditCard: String) {
        // Level 2: Email (内部データ)
        UserDefaults.standard.set(email, forKey: "email")

        // Level 4: Password (厳密な機密)
        do {
            try keychain.save(password, for: .password)
        } catch {
            print("Failed to save password: \(error)")
        }

        // Level 4: Credit Card (厳密な機密)
        do {
            let encrypted = try encryption.encrypt(creditCard.data(using: .utf8)!)
            try keychain.save(encrypted.base64EncodedString(), for: .creditCard)
        } catch {
            print("Failed to save credit card: \(error)")
        }
    }
}

extension KeychainKey {
    static let creditCard = KeychainKey(rawValue: "com.app.creditCard")
}
```

---

## 安全なデータストレージ

### セキュアなUserDefaults

```swift
// UserDefaultsラッパー
class SecureUserDefaults {
    private let suite: UserDefaults
    private let encryption: EncryptionService

    init(suiteName: String? = nil, encryption: EncryptionService) {
        self.suite = UserDefaults(suiteName: suiteName) ?? .standard
        self.encryption = encryption
    }

    // 通常の値（暗号化不要）
    func set(_ value: Any?, forKey key: String) {
        suite.set(value, forKey: key)
    }

    func object(forKey key: String) -> Any? {
        suite.object(forKey: key)
    }

    // 暗号化された値
    func setSecure(_ string: String, forKey key: String) throws {
        guard let data = string.data(using: .utf8) else {
            throw SecureStorageError.encodingFailed
        }

        let (encrypted, nonce) = try encryption.encrypt(data, using: getEncryptionKey())

        let combined = nonce + encrypted
        suite.set(combined.base64EncodedString(), forKey: key)
    }

    func secureString(forKey key: String) throws -> String? {
        guard let base64String = suite.string(forKey: key),
              let combined = Data(base64Encoded: base64String) else {
            return nil
        }

        let nonceSize = 12
        let nonce = combined.prefix(nonceSize)
        let encrypted = combined.dropFirst(nonceSize)

        let decrypted = try encryption.decrypt(
            ciphertext: Data(encrypted),
            nonce: Data(nonce),
            using: getEncryptionKey()
        )

        return String(data: decrypted, encoding: .utf8)
    }

    private func getEncryptionKey() throws -> SymmetricKey {
        try encryption.loadKey(for: "userdefaults_encryption_key")
    }
}

enum SecureStorageError: Error {
    case encodingFailed
    case encryptionFailed
    case decryptionFailed
}

// @propertyWrapper for Secure Storage
@propertyWrapper
struct SecureStorage {
    private let key: String
    private let storage: SecureUserDefaults

    init(key: String, storage: SecureUserDefaults) {
        self.key = key
        self.storage = storage
    }

    var wrappedValue: String? {
        get {
            try? storage.secureString(forKey: key)
        }
        set {
            if let value = newValue {
                try? storage.setSecure(value, forKey: key)
            } else {
                storage.set(nil, forKey: key)
            }
        }
    }
}

// 使用例
class UserSettings {
    @SecureStorage(key: "api_token", storage: secureDefaults)
    var apiToken: String?

    @SecureStorage(key: "refresh_token", storage: secureDefaults)
    var refreshToken: String?
}
```

---

## Core Data暗号化

### 暗号化されたCore Dataスタック

```swift
import CoreData

class EncryptedCoreDataStack {
    static let shared = EncryptedCoreDataStack()

    lazy var persistentContainer: NSPersistentContainer = {
        let container = NSPersistentContainer(name: "Model")

        // ファイル保護の設定
        let storeURL = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("encrypted.sqlite")

        let description = NSPersistentStoreDescription(url: storeURL)

        // ファイル保護レベル
        description.setOption(
            FileProtectionType.complete as NSObject,
            forKey: NSPersistentStoreFileProtectionKey
        )

        // SQLite暗号化（オプション）
        // 注: iOSのデフォルトファイル暗号化で通常は十分
        if let encryptionKey = getEncryptionKey() {
            description.setOption(
                encryptionKey as NSObject,
                forKey: NSSQLitePragmasOption
            )
        }

        container.persistentStoreDescriptions = [description]

        container.loadPersistentStores { description, error in
            if let error = error {
                fatalError("Failed to load Core Data stack: \(error)")
            }
        }

        return container
    }()

    var context: NSManagedObjectContext {
        persistentContainer.viewContext
    }

    private func getEncryptionKey() -> String? {
        // Keychainから暗号化キーを取得
        let keychain = KeychainServiceImpl()
        return try? keychain.get(.coreDataEncryptionKey)
    }

    func save() {
        let context = persistentContainer.viewContext

        guard context.hasChanges else { return }

        do {
            try context.save()
        } catch {
            let nsError = error as NSError
            fatalError("Unresolved error \(nsError), \(nsError.userInfo)")
        }
    }
}

extension KeychainKey {
    static let coreDataEncryptionKey = KeychainKey(rawValue: "com.app.coreDataKey")
}

// 暗号化されたAttributeの実装
@objc(EncryptedString)
class EncryptedString: NSManagedObject {
    @NSManaged var encryptedValue: Data?

    private static let encryption = EncryptionService()

    var decryptedValue: String? {
        get {
            guard let encryptedValue = encryptedValue else { return nil }

            let nonceSize = 12
            let nonce = encryptedValue.prefix(nonceSize)
            let ciphertext = encryptedValue.dropFirst(nonceSize)

            do {
                let key = try EncryptedString.encryption.loadKey(for: "attribute_encryption_key")
                let decrypted = try EncryptedString.encryption.decrypt(
                    ciphertext: Data(ciphertext),
                    nonce: Data(nonce),
                    using: key
                )
                return String(data: decrypted, encoding: .utf8)
            } catch {
                return nil
            }
        }
        set {
            guard let value = newValue,
                  let data = value.data(using: .utf8) else {
                encryptedValue = nil
                return
            }

            do {
                let key = try EncryptedString.encryption.loadKey(for: "attribute_encryption_key")
                let (ciphertext, nonce) = try EncryptedString.encryption.encrypt(data, using: key)
                encryptedValue = nonce + ciphertext
            } catch {
                encryptedValue = nil
            }
        }
    }
}
```

---

## ファイル保護

### ファイル保護レベル

```swift
class SecureFileManager {
    private let fileManager = FileManager.default

    enum ProtectionLevel {
        case complete
        case completeUnlessOpen
        case completeUntilFirstUserAuthentication
        case none

        var attribute: FileProtectionType {
            switch self {
            case .complete:
                return .complete
            case .completeUnlessOpen:
                return .completeUnlessOpen
            case .completeUntilFirstUserAuthentication:
                return .completeUntilFirstUserAuthentication
            case .none:
                return .none
            }
        }
    }

    // ファイルの作成（保護レベル指定）
    func createFile(
        at url: URL,
        contents: Data,
        protection: ProtectionLevel = .complete
    ) throws {
        try contents.write(to: url, options: [.atomic, .completeFileProtection])

        // 保護レベルの設定
        try fileManager.setAttributes(
            [.protectionKey: protection.attribute],
            ofItemAtPath: url.path
        )
    }

    // 既存ファイルの保護レベル変更
    func setProtectionLevel(_ level: ProtectionLevel, for url: URL) throws {
        try fileManager.setAttributes(
            [.protectionKey: level.attribute],
            ofItemAtPath: url.path
        )
    }

    // 保護レベルの確認
    func getProtectionLevel(for url: URL) throws -> ProtectionLevel {
        let attributes = try fileManager.attributesOfItem(atPath: url.path)

        guard let protection = attributes[.protectionKey] as? FileProtectionType else {
            return .none
        }

        switch protection {
        case .complete:
            return .complete
        case .completeUnlessOpen:
            return .completeUnlessOpen
        case .completeUntilFirstUserAuthentication:
            return .completeUntilFirstUserAuthentication
        default:
            return .none
        }
    }

    // ディレクトリの保護レベル設定
    func setDirectoryProtection(_ level: ProtectionLevel, at url: URL) throws {
        // ディレクトリ自体
        try setProtectionLevel(level, for: url)

        // サブアイテムの取得
        let contents = try fileManager.contentsOfDirectory(
            at: url,
            includingPropertiesForKeys: nil
        )

        // 再帰的に設定
        for itemURL in contents {
            var isDirectory: ObjCBool = false
            fileManager.fileExists(atPath: itemURL.path, isDirectory: &isDirectory)

            if isDirectory.boolValue {
                try setDirectoryProtection(level, at: itemURL)
            } else {
                try setProtectionLevel(level, for: itemURL)
            }
        }
    }
}

// 暗号化されたファイルストレージ
class EncryptedFileStorage {
    private let fileManager: SecureFileManager
    private let encryption: EncryptionService

    init(fileManager: SecureFileManager, encryption: EncryptionService) {
        self.fileManager = fileManager
        self.encryption = encryption
    }

    func save(_ data: Data, to url: URL, protectionLevel: SecureFileManager.ProtectionLevel = .complete) throws {
        // データの暗号化
        let key = try encryption.loadKey(for: "file_encryption_key")
        let (ciphertext, nonce) = try encryption.encrypt(data, using: key)

        // 暗号化されたデータの保存
        let encryptedData = nonce + ciphertext
        try fileManager.createFile(at: url, contents: encryptedData, protection: protectionLevel)
    }

    func load(from url: URL) throws -> Data {
        // ファイルの読み込み
        let encryptedData = try Data(contentsOf: url)

        // 復号化
        let nonceSize = 12
        let nonce = encryptedData.prefix(nonceSize)
        let ciphertext = encryptedData.dropFirst(nonceSize)

        let key = try encryption.loadKey(for: "file_encryption_key")
        return try encryption.decrypt(
            ciphertext: Data(ciphertext),
            nonce: Data(nonce),
            using: key
        )
    }
}
```

---

## ネットワーク通信の保護

### 安全なAPI通信

```swift
class SecureAPIClient {
    private let session: URLSession
    private let tokenManager: TokenManager

    init(tokenManager: TokenManager) {
        self.tokenManager = tokenManager

        let configuration = URLSessionConfiguration.default
        configuration.tlsMinimumSupportedProtocolVersion = .TLSv13
        configuration.httpCookieAcceptPolicy = .never

        self.session = URLSession(configuration: configuration)
    }

    func request<T: Decodable>(
        _ endpoint: String,
        method: HTTPMethod = .get,
        body: Encodable? = nil
    ) async throws -> T {
        guard let url = URL(string: "https://api.example.com\(endpoint)") else {
            throw APIError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = method.rawValue

        // セキュリティヘッダー
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("no-cache", forHTTPHeaderField: "Cache-Control")

        // 認証トークン
        if let token = try? tokenManager.getAccessToken() {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        // リクエストボディの暗号化（オプション）
        if let body = body {
            let jsonData = try JSONEncoder().encode(body)
            // End-to-End暗号化が必要な場合
            // request.httpBody = try encryptRequestBody(jsonData)
            request.httpBody = jsonData
        }

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }

        // ステータスコードの検証
        guard (200...299).contains(httpResponse.statusCode) else {
            throw APIError.httpError(httpResponse.statusCode)
        }

        // レスポンスの復号化
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func encryptRequestBody(_ data: Data) throws -> Data {
        // E2E暗号化の実装
        // クライアントとサーバー間の公開鍵暗号化
        return data
    }
}

enum HTTPMethod: String {
    case get = "GET"
    case post = "POST"
    case put = "PUT"
    case delete = "DELETE"
}

enum APIError: Error {
    case invalidURL
    case invalidResponse
    case httpError(Int)
}
```

---

## メモリセキュリティ

### 機密データのメモリ管理

```swift
// セキュアメモリバッファ
class SecureMemoryBuffer {
    private var buffer: UnsafeMutableRawPointer
    private let size: Int

    init(size: Int) {
        self.size = size
        self.buffer = UnsafeMutableRawPointer.allocate(
            byteCount: size,
            alignment: MemoryLayout<UInt8>.alignment
        )
        // メモリをゼロで初期化
        memset(buffer, 0, size)
    }

    deinit {
        // メモリを安全にクリア
        memset_s(buffer, size, 0, size)
        buffer.deallocate()
    }

    func withUnsafeBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        let bufferPointer = UnsafeRawBufferPointer(start: buffer, count: size)
        return try body(bufferPointer)
    }

    func write(_ data: Data) {
        precondition(data.count <= size, "Data too large for buffer")
        data.copyBytes(to: UnsafeMutableRawBufferPointer(start: buffer, count: size))
    }

    func clear() {
        memset_s(buffer, size, 0, size)
    }
}

// セキュア文字列
class SecureString {
    private var buffer: SecureMemoryBuffer

    init(_ string: String) {
        let data = string.data(using: .utf8)!
        self.buffer = SecureMemoryBuffer(size: data.count)
        buffer.write(data)
    }

    deinit {
        buffer.clear()
    }

    var value: String {
        buffer.withUnsafeBytes { bytes in
            String(data: Data(bytes), encoding: .utf8) ?? ""
        }
    }

    func clear() {
        buffer.clear()
    }
}

// スクリーンショット保護
class ScreenshotProtection {
    func protect(_ view: UIView) {
        // Secure Textfieldの設定
        if let textField = view as? UITextField {
            textField.isSecureTextEntry = true
        }

        // スクリーンショット通知の監視
        NotificationCenter.default.addObserver(
            forName: UIApplication.userDidTakeScreenshotNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleScreenshot()
        }

        // 録画検知
        NotificationCenter.default.addObserver(
            forName: UIScreen.capturedDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleScreenRecording()
        }
    }

    private func handleScreenshot() {
        // スクリーンショット検出時の処理
        // - アラート表示
        // - ログ記録
        // - セキュリティイベント送信
        print("Screenshot detected!")
    }

    private func handleScreenRecording() {
        if UIScreen.main.isCaptured {
            // 録画開始時の処理
            // - 機密情報の非表示
            // - 警告表示
            print("Screen recording started!")
        } else {
            // 録画停止時の処理
            print("Screen recording stopped!")
        }
    }
}

// アプリバックグラウンド時の保護
extension UIViewController {
    func setupBackgroundProtection() {
        NotificationCenter.default.addObserver(
            forName: UIApplication.didEnterBackgroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.hideSecretInfo()
        }

        NotificationCenter.default.addObserver(
            forName: UIApplication.willEnterForegroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.showSecretInfo()
        }
    }

    private func hideSecretInfo() {
        // 機密情報を非表示に
        // - パスワードフィールドをクリア
        // - 残高などを隠す
    }

    private func showSecretInfo() {
        // 機密情報を再表示
    }
}
```

---

## データ漏洩防止

### クリップボード保護

```swift
class ClipboardProtection {
    func copySecurely(_ text: String, expirationSeconds: TimeInterval = 60) {
        UIPasteboard.general.string = text

        // 一定時間後にクリップボードをクリア
        DispatchQueue.main.asyncAfter(deadline: .now() + expirationSeconds) {
            if UIPasteboard.general.string == text {
                UIPasteboard.general.string = ""
            }
        }
    }

    func preventCopy(for textField: UITextField) {
        textField.isSecureTextEntry = true
    }

    func monitorClipboard() {
        NotificationCenter.default.addObserver(
            forName: UIPasteboard.changedNotification,
            object: nil,
            queue: .main
        ) { _ in
            self.handleClipboardChange()
        }
    }

    private func handleClipboardChange() {
        // クリップボード変更時の処理
        // - ログ記録
        // - セキュリティイベント送信
    }
}
```

---

## セキュアな削除

### データの完全削除

```swift
class SecureDataDeletion {
    private let fileManager = FileManager.default

    // ファイルのセキュアな削除
    func secureDelete(fileAt url: URL) throws {
        // 1. ファイルを複数回上書き
        let fileSize = try fileManager.attributesOfItem(atPath: url.path)[.size] as! UInt64

        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }

        // ランダムデータで3回上書き
        for _ in 0..<3 {
            var randomData = Data(count: Int(fileSize))
            _ = randomData.withUnsafeMutableBytes { bytes in
                SecRandomCopyBytes(kSecRandomDefault, bytes.count, bytes.baseAddress!)
            }

            try handle.seek(toOffset: 0)
            try handle.write(contentsOf: randomData)
            try handle.synchronize()
        }

        // 2. ファイル削除
        try fileManager.removeItem(at: url)
    }

    // Core Dataのセキュアな削除
    func secureDeleteCoreData(entity: NSManagedObject, context: NSManagedObjectContext) {
        // 1. すべての属性をゼロで上書き
        let entityDescription = entity.entity
        for property in entityDescription.properties {
            if let attributeDescription = property as? NSAttributeDescription {
                switch attributeDescription.attributeType {
                case .stringAttributeType:
                    entity.setValue("", forKey: attributeDescription.name)
                case .integer16AttributeType, .integer32AttributeType, .integer64AttributeType:
                    entity.setValue(0, forKey: attributeDescription.name)
                case .booleanAttributeType:
                    entity.setValue(false, forKey: attributeDescription.name)
                default:
                    entity.setValue(nil, forKey: attributeDescription.name)
                }
            }
        }

        // 2. 変更を保存
        try? context.save()

        // 3. エンティティを削除
        context.delete(entity)
        try? context.save()
    }

    // Keychainのセキュアな削除
    func secureDeleteKeychain(key: KeychainKey) throws {
        let keychain = KeychainServiceImpl()

        // 1. 値をゼロで上書き
        try keychain.save(String(repeating: "\0", count: 256), for: key)

        // 2. 削除
        try keychain.delete(key)
    }
}
```

このガイドでは、iOSデータセキュリティの包括的な実装方法を網羅しました。データ分類、暗号化、ファイル保護、メモリ管理、データ漏洩防止、セキュアな削除まで、機密データを適切に保護するための実践的な手法を提供しています。セキュリティは継続的な取り組みが必要であり、常に最新の脅威とベストプラクティスを把握することが重要です。

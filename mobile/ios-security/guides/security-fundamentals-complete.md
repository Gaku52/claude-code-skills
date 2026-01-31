# iOS Security Fundamentals 完全ガイド

## 目次
1. [セキュリティの基礎](#セキュリティの基礎)
2. [App Transport Security (ATS)](#app-transport-security-ats)
3. [Keychain Services](#keychain-services)
4. [データ暗号化](#データ暗号化)
5. [証明書ピンニング](#証明書ピンニング)
6. [Jailbreak検出](#jailbreak検出)
7. [コード難読化](#コード難読化)
8. [セキュリティベストプラクティス](#セキュリティベストプラクティス)

---

## セキュリティの基礎

### セキュリティの原則

```swift
/*
セキュリティの3原則:

1. 機密性 (Confidentiality)
   - データの暗号化
   - アクセス制御
   - 安全な通信

2. 完全性 (Integrity)
   - データの改ざん防止
   - デジタル署名
   - ハッシュ検証

3. 可用性 (Availability)
   - サービスの継続性
   - DoS攻撃対策
   - バックアップ

OWASP Mobile Top 10 (2024):
1. Improper Platform Usage
2. Insecure Data Storage
3. Insecure Communication
4. Insecure Authentication
5. Insufficient Cryptography
6. Insecure Authorization
7. Client Code Quality
8. Code Tampering
9. Reverse Engineering
10. Extraneous Functionality
*/

// ❌ 悪い例: 機密情報をハードコーディング
class BadAPIClient {
    private let apiKey = "sk_live_1234567890abcdef" // 絶対にNG!
    private let apiSecret = "secret_key_here" // 絶対にNG!

    func fetchData() {
        // API呼び出し
    }
}

// ✅ 良い例: 環境変数と安全な保存
class GoodAPIClient {
    private let apiKey: String
    private let keychainService: KeychainService

    init(keychainService: KeychainService) {
        self.keychainService = keychainService

        // Build Configurationから取得
        #if DEBUG
        self.apiKey = Bundle.main.infoDictionary?["API_KEY_DEV"] as? String ?? ""
        #else
        self.apiKey = Bundle.main.infoDictionary?["API_KEY_PROD"] as? String ?? ""
        #endif
    }

    func fetchData(with token: String) async throws -> Data {
        // トークンはKeychainから取得
        guard let token = try? keychainService.getToken() else {
            throw SecurityError.noToken
        }

        // 安全なAPI呼び出し
        return Data()
    }
}

enum SecurityError: Error {
    case noToken
    case invalidCredentials
    case encryptionFailed
}
```

### 安全なデータ処理

```swift
// ❌ 悪い例: 平文でUserDefaultsに保存
class BadUserSession {
    func saveCredentials(email: String, password: String) {
        UserDefaults.standard.set(email, forKey: "email")
        UserDefaults.standard.set(password, forKey: "password") // 絶対にNG!
    }
}

// ✅ 良い例: Keychainに暗号化して保存
class SecureUserSession {
    private let keychain: KeychainService

    init(keychain: KeychainService) {
        self.keychain = keychain
    }

    func saveCredentials(email: String, password: String) throws {
        // パスワードは絶対にKeychainへ
        try keychain.save(password, for: .password)

        // メールアドレスはUserDefaultsでも可（機密情報ではない）
        UserDefaults.standard.set(email, forKey: "email")
    }

    func getPassword() throws -> String {
        try keychain.get(.password)
    }
}

// メモリからの機密情報削除
class SecureString {
    private var data: Data

    init(_ string: String) {
        self.data = Data(string.utf8)
    }

    var value: String {
        String(data: data, encoding: .utf8) ?? ""
    }

    func clear() {
        // メモリをゼロで上書き
        data.withUnsafeMutableBytes { bytes in
            memset(bytes.baseAddress, 0, bytes.count)
        }
    }

    deinit {
        clear()
    }
}

// 使用例
func processPassword() {
    let password = SecureString("secret_password")

    // パスワード使用

    password.clear() // 使用後は即座にクリア
}
```

---

## App Transport Security (ATS)

### ATS設定

```xml
<!-- Info.plist -->
<key>NSAppTransportSecurity</key>
<dict>
    <!-- ❌ 悪い例: ATSを完全に無効化 -->
    <key>NSAllowsArbitraryLoads</key>
    <false/>

    <!-- ✅ 良い例: 特定ドメインのみ例外設定 -->
    <key>NSExceptionDomains</key>
    <dict>
        <key>example.com</key>
        <dict>
            <key>NSIncludesSubdomains</key>
            <true/>
            <key>NSExceptionRequiresForwardSecrecy</key>
            <false/>
            <key>NSExceptionMinimumTLSVersion</key>
            <string>TLSv1.2</string>
        </dict>
    </dict>
</dict>
```

### 安全なHTTP通信

```swift
// URLSessionの安全な設定
class SecureNetworkClient {
    private let session: URLSession

    init() {
        let configuration = URLSessionConfiguration.default

        // タイムアウト設定
        configuration.timeoutIntervalForRequest = 30
        configuration.timeoutIntervalForResource = 60

        // キャッシュポリシー
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData

        // Cookie設定
        configuration.httpCookieAcceptPolicy = .never
        configuration.httpShouldSetCookies = false

        // TLS設定
        configuration.tlsMinimumSupportedProtocolVersion = .TLSv12

        self.session = URLSession(
            configuration: configuration,
            delegate: self,
            delegateQueue: nil
        )
    }

    func request(_ url: URL) async throws -> Data {
        var request = URLRequest(url: url)

        // セキュリティヘッダー
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("no-cache", forHTTPHeaderField: "Cache-Control")

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw NetworkError.invalidResponse
        }

        // ステータスコード検証
        guard (200...299).contains(httpResponse.statusCode) else {
            throw NetworkError.httpError(httpResponse.statusCode)
        }

        return data
    }
}

extension SecureNetworkClient: URLSessionDelegate {
    func urlSession(
        _ session: URLSession,
        didReceive challenge: URLAuthenticationChallenge,
        completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
    ) {
        // 証明書ピンニング（後述）
        guard let serverTrust = challenge.protectionSpace.serverTrust else {
            completionHandler(.cancelAuthenticationChallenge, nil)
            return
        }

        // 証明書検証ロジック
        if validateCertificate(serverTrust) {
            let credential = URLCredential(trust: serverTrust)
            completionHandler(.useCredential, credential)
        } else {
            completionHandler(.cancelAuthenticationChallenge, nil)
        }
    }

    private func validateCertificate(_ serverTrust: SecTrust) -> Bool {
        // 証明書ピンニングの実装
        return true
    }
}

enum NetworkError: Error {
    case invalidResponse
    case httpError(Int)
}
```

---

## Keychain Services

### Keychainサービスの実装

```swift
protocol KeychainService {
    func save(_ value: String, for key: KeychainKey) throws
    func get(_ key: KeychainKey) throws -> String
    func delete(_ key: KeychainKey) throws
    func deleteAll() throws
}

enum KeychainKey: String {
    case accessToken = "com.app.accessToken"
    case refreshToken = "com.app.refreshToken"
    case password = "com.app.password"
    case encryptionKey = "com.app.encryptionKey"
}

class KeychainServiceImpl: KeychainService {
    private let service: String

    init(service: String = Bundle.main.bundleIdentifier ?? "com.app") {
        self.service = service
    }

    func save(_ value: String, for key: KeychainKey) throws {
        guard let data = value.data(using: .utf8) else {
            throw KeychainError.invalidData
        }

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key.rawValue,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        ]

        // 既存アイテムを削除
        SecItemDelete(query as CFDictionary)

        // 新しいアイテムを追加
        let status = SecItemAdd(query as CFDictionary, nil)

        guard status == errSecSuccess else {
            throw KeychainError.saveFailed(status)
        }
    }

    func get(_ key: KeychainKey) throws -> String {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key.rawValue,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess else {
            throw KeychainError.itemNotFound
        }

        guard let data = result as? Data,
              let value = String(data: data, encoding: .utf8) else {
            throw KeychainError.invalidData
        }

        return value
    }

    func delete(_ key: KeychainKey) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key.rawValue
        ]

        let status = SecItemDelete(query as CFDictionary)

        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.deleteFailed(status)
        }
    }

    func deleteAll() throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service
        ]

        let status = SecItemDelete(query as CFDictionary)

        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.deleteFailed(status)
        }
    }
}

enum KeychainError: Error {
    case invalidData
    case itemNotFound
    case saveFailed(OSStatus)
    case deleteFailed(OSStatus)
}

// Biometric認証付きKeychain
class BiometricKeychainService: KeychainService {
    private let service: String
    private let context = LAContext()

    init(service: String = Bundle.main.bundleIdentifier ?? "com.app") {
        self.service = service
    }

    func save(_ value: String, for key: KeychainKey) throws {
        guard let data = value.data(using: .utf8) else {
            throw KeychainError.invalidData
        }

        var error: NSError?
        guard context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error) else {
            throw KeychainError.biometricsNotAvailable
        }

        let accessControl = SecAccessControlCreateWithFlags(
            nil,
            kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            .biometryCurrentSet,
            nil
        )

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key.rawValue,
            kSecValueData as String: data,
            kSecAttrAccessControl as String: accessControl as Any,
            kSecUseAuthenticationContext as String: context
        ]

        SecItemDelete(query as CFDictionary)

        let status = SecItemAdd(query as CFDictionary, nil)

        guard status == errSecSuccess else {
            throw KeychainError.saveFailed(status)
        }
    }

    // その他のメソッドは同様に実装
}

extension KeychainError {
    case biometricsNotAvailable
}
```

---

## データ暗号化

### AES暗号化の実装

```swift
import CryptoKit

class EncryptionService {
    // AES-256 GCM暗号化
    func encrypt(_ data: Data, using key: SymmetricKey) throws -> (ciphertext: Data, nonce: Data) {
        let sealedBox = try AES.GCM.seal(data, using: key)

        guard let ciphertext = sealedBox.ciphertext,
              let nonce = sealedBox.nonce.withUnsafeBytes({ Data($0) }) else {
            throw EncryptionError.encryptionFailed
        }

        return (ciphertext, nonce)
    }

    func decrypt(ciphertext: Data, nonce: Data, using key: SymmetricKey) throws -> Data {
        guard let nonceValue = try? AES.GCM.Nonce(data: nonce) else {
            throw EncryptionError.invalidNonce
        }

        let sealedBox = try AES.GCM.SealedBox(nonce: nonceValue, ciphertext: ciphertext, tag: Data())
        return try AES.GCM.open(sealedBox, using: key)
    }

    // 鍵の生成
    func generateKey() -> SymmetricKey {
        SymmetricKey(size: .bits256)
    }

    // 鍵の保存（Keychainに保存）
    func saveKey(_ key: SymmetricKey, for identifier: String) throws {
        let keyData = key.withUnsafeBytes { Data($0) }

        let query: [String: Any] = [
            kSecClass as String: kSecClassKey,
            kSecAttrApplicationLabel as String: identifier,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly,
            kSecValueData as String: keyData
        ]

        SecItemDelete(query as CFDictionary)

        let status = SecItemAdd(query as CFDictionary, nil)

        guard status == errSecSuccess else {
            throw EncryptionError.keyStorageFailed
        }
    }

    // 鍵の取得
    func loadKey(for identifier: String) throws -> SymmetricKey {
        let query: [String: Any] = [
            kSecClass as String: kSecClassKey,
            kSecAttrApplicationLabel as String: identifier,
            kSecReturnData as String: true
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess,
              let keyData = result as? Data else {
            throw EncryptionError.keyNotFound
        }

        return SymmetricKey(data: keyData)
    }
}

enum EncryptionError: Error {
    case encryptionFailed
    case decryptionFailed
    case invalidNonce
    case keyStorageFailed
    case keyNotFound
}

// ファイル暗号化
class SecureFileStorage {
    private let encryption: EncryptionService
    private let fileManager = FileManager.default

    init(encryption: EncryptionService) {
        self.encryption = encryption
    }

    func saveSecurely(_ data: Data, to filename: String) throws {
        // 暗号化キーの取得または生成
        let key: SymmetricKey
        do {
            key = try encryption.loadKey(for: "file_encryption_key")
        } catch {
            key = encryption.generateKey()
            try encryption.saveKey(key, for: "file_encryption_key")
        }

        // データの暗号化
        let (ciphertext, nonce) = try encryption.encrypt(data, using: key)

        // 暗号化されたデータの保存
        let encryptedData = nonce + ciphertext
        let url = getDocumentsDirectory().appendingPathComponent(filename)

        try encryptedData.write(to: url, options: [.completeFileProtection])
    }

    func loadSecurely(from filename: String) throws -> Data {
        let url = getDocumentsDirectory().appendingPathComponent(filename)
        let encryptedData = try Data(contentsOf: url)

        // Nonceとciphertextの分離
        let nonceSize = 12 // AES.GCM.Nonce size
        let nonce = encryptedData.prefix(nonceSize)
        let ciphertext = encryptedData.dropFirst(nonceSize)

        // 復号化
        let key = try encryption.loadKey(for: "file_encryption_key")
        return try encryption.decrypt(
            ciphertext: Data(ciphertext),
            nonce: Data(nonce),
            using: key
        )
    }

    private func getDocumentsDirectory() -> URL {
        fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }
}
```

---

## 証明書ピンニング

### SSL Certificate Pinning

```swift
class CertificatePinner: NSObject, URLSessionDelegate {
    private let pinnedCertificates: Set<Data>

    init(pinnedCertificates: Set<Data>) {
        self.pinnedCertificates = pinnedCertificates
    }

    func urlSession(
        _ session: URLSession,
        didReceive challenge: URLAuthenticationChallenge,
        completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
    ) {
        guard challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
              let serverTrust = challenge.protectionSpace.serverTrust else {
            completionHandler(.cancelAuthenticationChallenge, nil)
            return
        }

        // 証明書の検証
        if validateCertificates(serverTrust) {
            let credential = URLCredential(trust: serverTrust)
            completionHandler(.useCredential, credential)
        } else {
            completionHandler(.cancelAuthenticationChallenge, nil)
        }
    }

    private func validateCertificates(_ serverTrust: SecTrust) -> Bool {
        // サーバー証明書の取得
        guard let serverCertificate = SecTrustGetCertificateAtIndex(serverTrust, 0) else {
            return false
        }

        let serverCertificateData = SecCertificateCopyData(serverCertificate) as Data

        // ピン留めされた証明書と比較
        return pinnedCertificates.contains(serverCertificateData)
    }

    // Public Key Pinning
    private func validatePublicKeys(_ serverTrust: SecTrust) -> Bool {
        var policy = SecPolicyCreateSSL(true, nil)
        var secTrustResultType = SecTrustResultType.invalid

        SecTrustEvaluate(serverTrust, &secTrustResultType)

        guard let serverCertificate = SecTrustGetCertificateAtIndex(serverTrust, 0),
              let serverPublicKey = SecCertificateCopyKey(serverCertificate) else {
            return false
        }

        let serverPublicKeyData = SecKeyCopyExternalRepresentation(serverPublicKey, nil) as Data?

        // ピン留めされた公開鍵と比較
        return pinnedCertificates.contains { pinnedCert in
            guard let pinnedCertRef = SecCertificateCreateWithData(nil, pinnedCert as CFData),
                  let pinnedPublicKey = SecCertificateCopyKey(pinnedCertRef),
                  let pinnedPublicKeyData = SecKeyCopyExternalRepresentation(pinnedPublicKey, nil) as Data? else {
                return false
            }

            return serverPublicKeyData == pinnedPublicKeyData
        }
    }
}

// 証明書の読み込み
extension CertificatePinner {
    static func loadCertificates(from bundle: Bundle = .main) -> Set<Data> {
        var certificates = Set<Data>()

        let certificateNames = ["api.example.com", "cdn.example.com"]

        for name in certificateNames {
            if let certificatePath = bundle.path(forResource: name, ofType: "cer"),
               let certificateData = try? Data(contentsOf: URL(fileURLWithPath: certificatePath)) {
                certificates.insert(certificateData)
            }
        }

        return certificates
    }
}

// 使用例
class SecureAPIClient {
    private let session: URLSession

    init() {
        let certificates = CertificatePinner.loadCertificates()
        let pinner = CertificatePinner(pinnedCertificates: certificates)

        let configuration = URLSessionConfiguration.default
        self.session = URLSession(
            configuration: configuration,
            delegate: pinner,
            delegateQueue: nil
        )
    }

    func request(_ url: URL) async throws -> Data {
        let (data, _) = try await session.data(from: url)
        return data
    }
}
```

---

## Jailbreak検出

### Jailbreak検出の実装

```swift
class JailbreakDetector {
    static func isJailbroken() -> Bool {
        #if targetEnvironment(simulator)
        return false
        #else
        return checkSuspiciousFiles()
            || checkSuspiciousApps()
            || checkSystemWrite()
            || checkFork()
            || checkSymlinks()
        #endif
    }

    // 疑わしいファイルの存在チェック
    private static func checkSuspiciousFiles() -> Bool {
        let suspiciousFiles = [
            "/Applications/Cydia.app",
            "/Library/MobileSubstrate/MobileSubstrate.dylib",
            "/bin/bash",
            "/usr/sbin/sshd",
            "/etc/apt",
            "/private/var/lib/apt/",
            "/private/var/lib/cydia",
            "/private/var/stash"
        ]

        return suspiciousFiles.contains { FileManager.default.fileExists(atPath: $0) }
    }

    // 疑わしいアプリの存在チェック
    private static func checkSuspiciousApps() -> Bool {
        let suspiciousSchemes = [
            "cydia://",
            "undecimus://",
            "sileo://",
            "zbra://"
        ]

        return suspiciousSchemes.contains { scheme in
            if let url = URL(string: scheme) {
                return UIApplication.shared.canOpenURL(url)
            }
            return false
        }
    }

    // システムへの書き込みチェック
    private static func checkSystemWrite() -> Bool {
        let testPath = "/private/jailbreak_test.txt"
        let testString = "test"

        do {
            try testString.write(toFile: testPath, atomically: true, encoding: .utf8)
            try FileManager.default.removeItem(atPath: testPath)
            return true // 書き込み成功 = Jailbreak
        } catch {
            return false
        }
    }

    // Fork関数のチェック
    private static func checkFork() -> Bool {
        let result = fork()

        if result >= 0 {
            if result > 0 {
                kill(result, SIGTERM)
            }
            return true
        }

        return false
    }

    // シンボリックリンクのチェック
    private static func checkSymlinks() -> Bool {
        let paths = [
            "/Applications",
            "/Library/Ringtones",
            "/Library/Wallpaper",
            "/usr/arm-apple-darwin9",
            "/usr/include",
            "/usr/libexec",
            "/usr/share"
        ]

        return paths.contains { path in
            if let attributes = try? FileManager.default.attributesOfItem(atPath: path),
               let fileType = attributes[.type] as? FileAttributeType {
                return fileType == .typeSymbolicLink
            }
            return false
        }
    }
}

// Jailbreak検出時の対処
class SecurityManager {
    func performSecurityCheck() {
        if JailbreakDetector.isJailbroken() {
            handleJailbrokenDevice()
        }
    }

    private func handleJailbrokenDevice() {
        // オプション1: アプリを終了
        // exit(0)

        // オプション2: 機能を制限
        restrictFeatures()

        // オプション3: 警告を表示
        showJailbreakWarning()
    }

    private func restrictFeatures() {
        // 重要な機能を無効化
        UserDefaults.standard.set(true, forKey: "isJailbroken")
    }

    private func showJailbreakWarning() {
        // 警告ダイアログを表示
    }
}
```

このガイドでは、iOSセキュリティの基礎から、App Transport Security、Keychain、暗号化、証明書ピンニング、Jailbreak検出まで、セキュアなiOSアプリ開発に必要な知識を網羅しました。セキュリティは継続的な取り組みが必要であり、常に最新の脅威とベストプラクティスを把握することが重要です。

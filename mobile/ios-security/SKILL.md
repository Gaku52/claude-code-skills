---
name: ios-security
description: iOSセキュリティ実装ガイド。認証・認可、データ暗号化、Keychain、証明書ピンニング、App Transport Security、脱獄検知、難読化など、セキュアなiOSアプリケーション開発のベストプラクティス。
---

# iOS Security Skill

## 📋 目次

1. [概要](#概要)
2. [認証・認可](#認証認可)
3. [データ暗号化](#データ暗号化)
4. [Keychain活用](#keychain活用)
5. [ネットワークセキュリティ](#ネットワークセキュリティ)
6. [App Transport Security](#app-transport-security)
7. [証明書ピンニング](#証明書ピンニング)
8. [脱獄検知](#脱獄検知)
9. [コード難読化](#コード難読化)
10. [セキュリティチェックリスト](#セキュリティチェックリスト)

## 概要

iOSアプリケーションのセキュリティ実装における実践的なパターンとベストプラクティスを提供します。

**対象:**
- iOSエンジニア
- セキュリティエンジニア
- アプリアーキテクト

**このSkillでできること:**
- セキュアな認証フローの実装
- データの安全な保存と暗号化
- ネットワーク通信のセキュリティ確保
- 脱獄・改ざん検知の実装

## 📚 公式ドキュメント・参考リソース

**このガイドで学べること**: iOS認証実装、データ暗号化、Keychain活用、ネットワークセキュリティ、脱獄検知
**公式で確認すべきこと**: 最新のセキュリティアップデート、新しい暗号化アルゴリズム、プラットフォームセキュリティ機能

### 主要な公式ドキュメント

- **[Apple Security Documentation](https://developer.apple.com/documentation/security)** - Apple公式セキュリティフレームワーク
  - [CryptoKit](https://developer.apple.com/documentation/cryptokit)
  - [Keychain Services](https://developer.apple.com/documentation/security/keychain_services)
  - [LocalAuthentication](https://developer.apple.com/documentation/localauthentication)

- **[App Transport Security](https://developer.apple.com/documentation/security/preventing_insecure_network_connections)** - ネットワークセキュリティ

- **[Secure Coding Guide](https://developer.apple.com/library/archive/documentation/Security/Conceptual/SecureCodingGuide/)** - セキュアコーディングガイド

- **[OAuth 2.0 RFC](https://oauth.net/2/)** - OAuth 2.0認証標準

### 関連リソース

- **[OWASP Mobile Security](https://owasp.org/www-project-mobile-security/)** - モバイルセキュリティベストプラクティス
- **[OWASP Mobile Top 10](https://owasp.org/www-project-mobile-top-10/)** - モバイルアプリの脆弱性トップ10
- **[NSHipster Security](https://nshipster.com/)** - iOSセキュリティ実践ガイド

---

## 認証・認可

### OAuth 2.0 / OpenID Connect

**認証フロー実装:**

```swift
import AuthenticationServices

class AuthManager: NSObject, ObservableObject {
    @Published var isAuthenticated = false
    @Published var user: User?

    private let authURL = URL(string: "https://auth.example.com/oauth/authorize")!
    private let clientID = "your-client-id"
    private let redirectURI = "myapp://callback"

    func signIn(presenting viewController: UIViewController) {
        guard var components = URLComponents(url: authURL, resolvingAgainstBaseURL: false) else {
            return
        }

        components.queryItems = [
            URLQueryItem(name: "client_id", value: clientID),
            URLQueryItem(name: "redirect_uri", value: redirectURI),
            URLQueryItem(name: "response_type", value: "code"),
            URLQueryItem(name: "scope", value: "openid profile email")
        ]

        guard let url = components.url else { return }

        let session = ASWebAuthenticationSession(
            url: url,
            callbackURLScheme: "myapp"
        ) { [weak self] callbackURL, error in
            if let error = error {
                print("Authentication error: \(error)")
                return
            }

            guard let callbackURL = callbackURL,
                  let code = URLComponents(url: callbackURL, resolvingAgainstBaseURL: false)?
                    .queryItems?
                    .first(where: { $0.name == "code" })?
                    .value else {
                return
            }

            Task {
                await self?.exchangeCodeForToken(code)
            }
        }

        session.presentationContextProvider = self
        session.prefersEphemeralWebBrowserSession = true
        session.start()
    }

    private func exchangeCodeForToken(_ code: String) async {
        // トークンエンドポイントにPOSTリクエスト
        let tokenURL = URL(string: "https://auth.example.com/oauth/token")!
        var request = URLRequest(url: tokenURL)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")

        let body = [
            "grant_type": "authorization_code",
            "code": code,
            "client_id": clientID,
            "redirect_uri": redirectURI
        ]
        request.httpBody = body.percentEncoded()

        do {
            let (data, _) = try await URLSession.shared.data(for: request)
            let response = try JSONDecoder().decode(TokenResponse.self, from: data)

            // トークンをKeychainに安全に保存
            try KeychainManager.shared.saveToken(response.accessToken)
            try KeychainManager.shared.save(
                response.refreshToken.data(using: .utf8)!,
                forKey: "refreshToken"
            )

            await MainActor.run {
                self.isAuthenticated = true
            }
        } catch {
            print("Token exchange error: \(error)")
        }
    }

    func signOut() {
        try? KeychainManager.shared.deleteToken()
        try? KeychainManager.shared.delete(forKey: "refreshToken")
        isAuthenticated = false
        user = nil
    }
}

struct TokenResponse: Codable {
    let accessToken: String
    let refreshToken: String
    let expiresIn: Int
    let tokenType: String

    enum CodingKeys: String, CodingKey {
        case accessToken = "access_token"
        case refreshToken = "refresh_token"
        case expiresIn = "expires_in"
        case tokenType = "token_type"
    }
}

extension AuthManager: ASWebAuthenticationPresentationContextProviding {
    func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
        return ASPresentationAnchor()
    }
}
```

### トークンリフレッシュ

```swift
class TokenRefreshManager {
    static let shared = TokenRefreshManager()

    private var refreshTask: Task<String, Error>?

    func getValidAccessToken() async throws -> String {
        // 既存のリフレッシュタスクがあれば待機
        if let task = refreshTask {
            return try await task.value
        }

        // 現在のトークンをチェック
        if let token = try? KeychainManager.shared.loadToken(),
           !isTokenExpired(token) {
            return token
        }

        // 新しいリフレッシュタスクを開始
        let task = Task { () -> String in
            defer { self.refreshTask = nil }
            return try await self.refreshAccessToken()
        }

        refreshTask = task
        return try await task.value
    }

    private func refreshAccessToken() async throws -> String {
        let refreshTokenData = try KeychainManager.shared.load(forKey: "refreshToken")
        guard let refreshToken = String(data: refreshTokenData, encoding: .utf8) else {
            throw AuthError.invalidRefreshToken
        }

        let url = URL(string: "https://auth.example.com/oauth/token")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"

        let body = [
            "grant_type": "refresh_token",
            "refresh_token": refreshToken,
            "client_id": clientID
        ]
        request.httpBody = body.percentEncoded()

        let (data, _) = try await URLSession.shared.data(for: request)
        let response = try JSONDecoder().decode(TokenResponse.self, from: data)

        // 新しいトークンを保存
        try KeychainManager.shared.saveToken(response.accessToken)

        return response.accessToken
    }

    private func isTokenExpired(_ token: String) -> Bool {
        // JWTのペイロードをデコードしてexpチェック
        let parts = token.components(separatedBy: ".")
        guard parts.count == 3,
              let payloadData = Data(base64Encoded: parts[1].base64Padded()),
              let payload = try? JSONDecoder().decode(JWTPayload.self, from: payloadData) else {
            return true
        }

        return Date().timeIntervalSince1970 >= payload.exp
    }
}

struct JWTPayload: Codable {
    let exp: TimeInterval
}

enum AuthError: Error {
    case invalidRefreshToken
}
```

### 生体認証（Face ID / Touch ID）

```swift
import LocalAuthentication

class BiometricAuthManager {
    static let shared = BiometricAuthManager()

    func authenticate() async throws {
        let context = LAContext()
        var error: NSError?

        guard context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error) else {
            throw BiometricError.notAvailable
        }

        let reason = "アプリにアクセスするために認証してください"

        return try await withCheckedThrowingContinuation { continuation in
            context.evaluatePolicy(
                .deviceOwnerAuthenticationWithBiometrics,
                localizedReason: reason
            ) { success, error in
                if success {
                    continuation.resume()
                } else if let error = error {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func biometricType() -> BiometricType {
        let context = LAContext()
        guard context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: nil) else {
            return .none
        }

        switch context.biometryType {
        case .faceID:
            return .faceID
        case .touchID:
            return .touchID
        case .none:
            return .none
        @unknown default:
            return .none
        }
    }
}

enum BiometricType {
    case faceID
    case touchID
    case none
}

enum BiometricError: Error {
    case notAvailable
}
```

## データ暗号化

### AES暗号化

```swift
import CryptoKit

class EncryptionManager {
    static let shared = EncryptionManager()

    func encrypt(_ data: Data, using key: SymmetricKey) throws -> Data {
        let sealedBox = try AES.GCM.seal(data, using: key)
        return sealedBox.combined!
    }

    func decrypt(_ data: Data, using key: SymmetricKey) throws -> Data {
        let sealedBox = try AES.GCM.SealedBox(combined: data)
        return try AES.GCM.open(sealedBox, using: key)
    }

    func generateKey() -> SymmetricKey {
        SymmetricKey(size: .bits256)
    }

    func deriveKey(from password: String, salt: Data) -> SymmetricKey {
        let passwordData = Data(password.utf8)
        return HKDF<SHA256>.deriveKey(
            inputKeyMaterial: SymmetricKey(data: passwordData),
            salt: salt,
            outputByteCount: 32
        )
    }
}

// 使用例
let manager = EncryptionManager.shared
let key = manager.generateKey()
let plainData = "Sensitive information".data(using: .utf8)!

let encrypted = try manager.encrypt(plainData, using: key)
let decrypted = try manager.decrypt(encrypted, using: key)
```

### ファイル暗号化

```swift
class SecureFileManager {
    private let encryptionManager = EncryptionManager.shared
    private let fileManager = FileManager.default

    func saveSecurely(_ data: Data, to filename: String, key: SymmetricKey) throws {
        let encrypted = try encryptionManager.encrypt(data, using: key)

        let url = try fileManager
            .url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent(filename)

        try encrypted.write(to: url, options: .completeFileProtection)
    }

    func loadSecurely(from filename: String, key: SymmetricKey) throws -> Data {
        let url = try fileManager
            .url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
            .appendingPathComponent(filename)

        let encrypted = try Data(contentsOf: url)
        return try encryptionManager.decrypt(encrypted, using: key)
    }
}
```

## Keychain活用

### 高度なKeychain操作

```swift
class SecureKeychainManager {
    static let shared = SecureKeychainManager()

    func save(
        _ data: Data,
        forKey key: String,
        accessGroup: String? = nil,
        accessibility: CFString = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
    ) throws {
        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
            kSecAttrAccessible as String: accessibility
        ]

        if let accessGroup = accessGroup {
            query[kSecAttrAccessGroup as String] = accessGroup
        }

        // 既存のアイテムを削除
        SecItemDelete(query as CFDictionary)

        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw KeychainError.unableToSave(status)
        }
    }

    func load(
        forKey key: String,
        accessGroup: String? = nil
    ) throws -> Data {
        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        if let accessGroup = accessGroup {
            query[kSecAttrAccessGroup as String] = accessGroup
        }

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess,
              let data = result as? Data else {
            throw KeychainError.itemNotFound(status)
        }

        return data
    }

    func update(
        _ data: Data,
        forKey key: String,
        accessGroup: String? = nil
    ) throws {
        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]

        if let accessGroup = accessGroup {
            query[kSecAttrAccessGroup as String] = accessGroup
        }

        let attributes: [String: Any] = [
            kSecValueData as String: data
        ]

        let status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        guard status == errSecSuccess else {
            throw KeychainError.unableToUpdate(status)
        }
    }
}

enum KeychainError: Error {
    case unableToSave(OSStatus)
    case itemNotFound(OSStatus)
    case unableToUpdate(OSStatus)
}
```

## ネットワークセキュリティ

### URLSessionDelegate with SSL Pinning

```swift
class SecureNetworkManager: NSObject {
    static let shared = SecureNetworkManager()

    private lazy var session: URLSession = {
        let configuration = URLSessionConfiguration.default
        return URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
    }()

    func request(_ url: URL) async throws -> Data {
        let (data, response) = try await session.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw NetworkError.invalidResponse
        }

        return data
    }
}

extension SecureNetworkManager: URLSessionDelegate {
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

        // 証明書ピンニング検証
        if evaluateServerTrust(serverTrust, forDomain: challenge.protectionSpace.host) {
            completionHandler(.useCredential, URLCredential(trust: serverTrust))
        } else {
            completionHandler(.cancelAuthenticationChallenge, nil)
        }
    }

    private func evaluateServerTrust(_ serverTrust: SecTrust, forDomain domain: String) -> Bool {
        // サーバー証明書を取得
        guard let serverCertificate = SecTrustGetCertificateAtIndex(serverTrust, 0) else {
            return false
        }

        // ピンニングする証明書のハッシュ（SHA256）
        let pinnedHashes = [
            "sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
            "sha256/BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB="
        ]

        let serverCertificateData = SecCertificateCopyData(serverCertificate) as Data
        let serverCertificateHash = sha256(serverCertificateData)

        return pinnedHashes.contains(serverCertificateHash)
    }

    private func sha256(_ data: Data) -> String {
        let hash = SHA256.hash(data: data)
        return "sha256/" + Data(hash).base64EncodedString()
    }
}
```

## App Transport Security

### Info.plist設定

```xml
<key>NSAppTransportSecurity</key>
<dict>
    <!-- 全体的にATSを有効にし、例外を最小限にする -->
    <key>NSAllowsArbitraryLoads</key>
    <false/>

    <!-- 特定のドメインのみHTTPを許可（開発環境など） -->
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

### プログラマティックチェック

```swift
class NetworkSecurityValidator {
    static func validateURL(_ url: URL) throws {
        guard url.scheme == "https" else {
            throw SecurityError.insecureConnection
        }

        // 開発環境のみlocalhostを許可
        #if DEBUG
        if url.host == "localhost" {
            return
        }
        #endif

        // 本番環境では必ずHTTPSを要求
        guard url.scheme == "https" else {
            throw SecurityError.httpsRequired
        }
    }
}

enum SecurityError: Error {
    case insecureConnection
    case httpsRequired
}
```

## 証明書ピンニング

### Public Key Pinning

```swift
class PublicKeyPinner {
    private let pinnedKeys: Set<SecKey>

    init(certificates: [String]) {
        var keys = Set<SecKey>()

        for certName in certificates {
            if let path = Bundle.main.path(forResource: certName, ofType: "cer"),
               let certData = try? Data(contentsOf: URL(fileURLWithPath: path)),
               let certificate = SecCertificateCreateWithData(nil, certData as CFData),
               let publicKey = SecCertificateCopyKey(certificate) {
                keys.insert(publicKey)
            }
        }

        self.pinnedKeys = keys
    }

    func validate(_ serverTrust: SecTrust) -> Bool {
        guard let serverKey = SecTrustCopyKey(serverTrust) else {
            return false
        }

        return pinnedKeys.contains(serverKey)
    }
}
```

## 脱獄検知

### 脱獄検知実装

```swift
class JailbreakDetector {
    static let shared = JailbreakDetector()

    func isJailbroken() -> Bool {
        #if targetEnvironment(simulator)
        return false
        #else
        return checkSuspiciousFiles() ||
               checkSuspiciousApps() ||
               checkWritableLocations() ||
               checkCydiaURLScheme()
        #endif
    }

    private func checkSuspiciousFiles() -> Bool {
        let suspiciousFiles = [
            "/Applications/Cydia.app",
            "/Library/MobileSubstrate/MobileSubstrate.dylib",
            "/bin/bash",
            "/usr/sbin/sshd",
            "/etc/apt",
            "/private/var/lib/apt/"
        ]

        return suspiciousFiles.contains { FileManager.default.fileExists(atPath: $0) }
    }

    private func checkSuspiciousApps() -> Bool {
        guard let cydiaURL = URL(string: "cydia://package/com.example.package") else {
            return false
        }
        return UIApplication.shared.canOpenURL(cydiaURL)
    }

    private func checkWritableLocations() -> Bool {
        let testPath = "/private/jailbreak-test.txt"
        let testString = "test"

        do {
            try testString.write(toFile: testPath, atomically: true, encoding: .utf8)
            try FileManager.default.removeItem(atPath: testPath)
            return true // 書き込みできたら脱獄済み
        } catch {
            return false
        }
    }

    private func checkCydiaURLScheme() -> Bool {
        return UIApplication.shared.canOpenURL(URL(string: "cydia://")!)
    }

    func handleJailbrokenDevice() {
        // 脱獄デバイス検知時の対応
        #if !DEBUG
        // 本番環境では機能制限やアラート表示
        showJailbreakAlert()
        disableSensitiveFeatures()
        #endif
    }

    private func showJailbreakAlert() {
        // ユーザーに警告を表示
    }

    private func disableSensitiveFeatures() {
        // 機密機能を無効化
    }
}
```

## コード難読化

### 文字列の難読化

```swift
// ビルド時スクリプトで難読化
class ObfuscatedStrings {
    static func apiKey() -> String {
        // XORで簡易的な難読化
        let obfuscated: [UInt8] = [0x41, 0x42, 0x43] // 実際はもっと長い
        let key: UInt8 = 0x42

        return String(bytes: obfuscated.map { $0 ^ key }, encoding: .utf8)!
    }

    static func baseURL() -> String {
        // Base64 + ROT13など複数の変換を組み合わせる
        let encoded = "aHR0cHM6Ly9hcGkuZXhhbXBsZS5jb20="
        guard let data = Data(base64Encoded: encoded),
              let decoded = String(data: data, encoding: .utf8) else {
            return ""
        }
        return decoded
    }
}
```

### クラス名・メソッド名の難読化

```bash
# SwiftShield等のツールを使用
swiftshield obfuscate -input MyApp.xcodeproj
```

## セキュリティチェックリスト

### 開発フェーズ

- [ ] 機密情報（APIキー、トークン）をコードに直接記述しない
- [ ] すべてのネットワーク通信でHTTPSを使用
- [ ] UserDefaultsに機密情報を保存しない
- [ ] ログ出力に機密情報を含めない
- [ ] デバッグコードを本番ビルドから除外

### 認証・認可

- [ ] OAuth 2.0 / OpenID Connectを使用
- [ ] トークンをKeychainに安全に保存
- [ ] トークンリフレッシュ機構を実装
- [ ] 生体認証（Face ID/Touch ID）を実装
- [ ] セッションタイムアウトを設定

### データ保護

- [ ] 機密データをAES暗号化
- [ ] ファイル保護属性を設定（.completeFileProtection）
- [ ] Keychainのアクセシビリティ属性を適切に設定
- [ ] データベースを暗号化（SQLCipher等）

### ネットワークセキュリティ

- [ ] 証明書ピンニングを実装
- [ ] App Transport Securityを有効化
- [ ] API通信を認証トークンで保護
- [ ] レスポンスを検証

### その他

- [ ] 脱獄検知を実装
- [ ] コード難読化を検討
- [ ] スクリーンショット防止（必要に応じて）
- [ ] リバースエンジニアリング対策

---

**関連Skills:**
- [ios-development](../ios-development/SKILL.md) - iOS開発全般
- [networking-data](../networking-data/SKILL.md) - ネットワーク・データ永続化
- [backend-development](../backend-development/SKILL.md) - API設計・セキュリティ

**更新履歴:**
- 2025-12-24: 初版作成

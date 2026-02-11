# Authentication Implementation 完全ガイド

## 目次
1. [認証の基礎](#認証の基礎)
2. [OAuth 2.0実装](#oauth-20実装)
3. [JWT (JSON Web Token)](#jwt-json-web-token)
4. [Biometric認証](#biometric認証)
5. [Multi-Factor Authentication (MFA)](#multi-factor-authentication-mfa)
6. [Sign in with Apple](#sign-in-with-apple)
7. [セッション管理](#セッション管理)
8. [認証フロー実装](#認証フロー実装)

---

## 認証の基礎

### 認証 vs 認可

```swift
/*
認証 (Authentication): "あなたは誰ですか？"
- ユーザーの身元確認
- ログイン処理
- パスワード、生体認証

認可 (Authorization): "何ができますか？"
- アクセス権限の確認
- ロールベースアクセス制御 (RBAC)
- スコープ、パーミッション

認証フロー:
1. ユーザー認証 (Username/Password, OAuth, Biometric)
2. トークン発行 (Access Token, Refresh Token)
3. トークン検証
4. トークン更新
5. ログアウト
*/

// ❌ 悪い例: パスワードを平文で送信
struct BadLoginRequest: Codable {
    let email: String
    let password: String // 平文で送信は絶対にNG!
}

// ✅ 良い例: セキュアな認証実装
protocol AuthenticationService {
    func login(email: String, password: String) async throws -> AuthToken
    func refreshToken(_ refreshToken: String) async throws -> AuthToken
    func logout() async throws
}

struct AuthToken: Codable {
    let accessToken: String
    let refreshToken: String
    let expiresIn: TimeInterval
    let tokenType: String

    var expirationDate: Date {
        Date().addingTimeInterval(expiresIn)
    }
}

class SecureAuthService: AuthenticationService {
    private let apiClient: APIClient
    private let keychain: KeychainService
    private let encryptionService: EncryptionService

    init(
        apiClient: APIClient,
        keychain: KeychainService,
        encryptionService: EncryptionService
    ) {
        self.apiClient = apiClient
        self.keychain = keychain
        self.encryptionService = encryptionService
    }

    func login(email: String, password: String) async throws -> AuthToken {
        // パスワードのハッシュ化（クライアント側）
        let hashedPassword = hashPassword(password)

        let request = LoginRequest(email: email, passwordHash: hashedPassword)
        let response: LoginResponse = try await apiClient.post("/auth/login", body: request)

        // トークンの保存
        try saveTokens(response.token)

        return response.token
    }

    func refreshToken(_ refreshToken: String) async throws -> AuthToken {
        let request = RefreshTokenRequest(refreshToken: refreshToken)
        let response: RefreshTokenResponse = try await apiClient.post("/auth/refresh", body: request)

        try saveTokens(response.token)

        return response.token
    }

    func logout() async throws {
        // サーバー側でトークンを無効化
        if let token = try? keychain.get(.accessToken) {
            try? await apiClient.post("/auth/logout", body: ["token": token])
        }

        // ローカルトークンの削除
        try keychain.delete(.accessToken)
        try keychain.delete(.refreshToken)
    }

    private func saveTokens(_ token: AuthToken) throws {
        try keychain.save(token.accessToken, for: .accessToken)
        try keychain.save(token.refreshToken, for: .refreshToken)

        // 有効期限の保存
        UserDefaults.standard.set(
            token.expirationDate,
            forKey: "token_expiration_date"
        )
    }

    private func hashPassword(_ password: String) -> String {
        // PBKDF2を使用したハッシュ化
        // 注: 通常はサーバー側でハッシュ化すべき
        // クライアント側でのハッシュ化は追加のセキュリティ層として
        return password.sha256() // 実装は簡略化
    }
}

struct LoginRequest: Codable {
    let email: String
    let passwordHash: String
}

struct LoginResponse: Codable {
    let token: AuthToken
    let user: User
}

struct RefreshTokenRequest: Codable {
    let refreshToken: String
}

struct RefreshTokenResponse: Codable {
    let token: AuthToken
}
```

---

## OAuth 2.0実装

### OAuth 2.0フロー

```swift
import AuthenticationServices

// OAuth Configuration
struct OAuthConfig {
    let clientId: String
    let redirectURI: String
    let scope: String
    let authorizationEndpoint: URL
    let tokenEndpoint: URL

    static let google = OAuthConfig(
        clientId: "YOUR_CLIENT_ID",
        redirectURI: "com.yourapp://oauth-callback",
        scope: "openid profile email",
        authorizationEndpoint: URL(string: "https://accounts.google.com/o/oauth2/v2/auth")!,
        tokenEndpoint: URL(string: "https://oauth2.googleapis.com/token")!
    )
}

// OAuth Service
class OAuthService: NSObject {
    private let config: OAuthConfig
    private var authSession: ASWebAuthenticationSession?
    private var continuation: CheckedContinuation<AuthToken, Error>?

    init(config: OAuthConfig) {
        self.config = config
    }

    func authenticate() async throws -> AuthToken {
        try await withCheckedThrowingContinuation { continuation in
            self.continuation = continuation

            // Authorization Code取得
            let authURL = buildAuthorizationURL()

            authSession = ASWebAuthenticationSession(
                url: authURL,
                callbackURLScheme: extractScheme(from: config.redirectURI)
            ) { [weak self] callbackURL, error in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }

                guard let callbackURL = callbackURL,
                      let code = self?.extractCode(from: callbackURL) else {
                    continuation.resume(throwing: OAuthError.invalidCallback)
                    return
                }

                // トークン交換
                Task {
                    do {
                        let token = try await self?.exchangeCodeForToken(code)
                        continuation.resume(returning: token!)
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }

            authSession?.presentationContextProvider = self
            authSession?.start()
        }
    }

    private func buildAuthorizationURL() -> URL {
        var components = URLComponents(
            url: config.authorizationEndpoint,
            resolvingAgainstBaseURL: false
        )!

        // PKCE (Proof Key for Code Exchange)
        let codeVerifier = generateCodeVerifier()
        let codeChallenge = generateCodeChallenge(from: codeVerifier)

        // Code Verifierを保存
        UserDefaults.standard.set(codeVerifier, forKey: "oauth_code_verifier")

        components.queryItems = [
            URLQueryItem(name: "client_id", value: config.clientId),
            URLQueryItem(name: "redirect_uri", value: config.redirectURI),
            URLQueryItem(name: "response_type", value: "code"),
            URLQueryItem(name: "scope", value: config.scope),
            URLQueryItem(name: "code_challenge", value: codeChallenge),
            URLQueryItem(name: "code_challenge_method", value: "S256"),
            URLQueryItem(name: "state", value: UUID().uuidString)
        ]

        return components.url!
    }

    private func exchangeCodeForToken(_ code: String) async throws -> AuthToken {
        guard let codeVerifier = UserDefaults.standard.string(forKey: "oauth_code_verifier") else {
            throw OAuthError.missingCodeVerifier
        }

        var request = URLRequest(url: config.tokenEndpoint)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")

        let parameters = [
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": config.redirectURI,
            "client_id": config.clientId,
            "code_verifier": codeVerifier
        ]

        request.httpBody = parameters
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: "&")
            .data(using: .utf8)

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw OAuthError.tokenExchangeFailed
        }

        let tokenResponse = try JSONDecoder().decode(OAuthTokenResponse.self, from: data)

        return AuthToken(
            accessToken: tokenResponse.accessToken,
            refreshToken: tokenResponse.refreshToken ?? "",
            expiresIn: TimeInterval(tokenResponse.expiresIn),
            tokenType: tokenResponse.tokenType
        )
    }

    // PKCE ヘルパーメソッド
    private func generateCodeVerifier() -> String {
        var buffer = [UInt8](repeating: 0, count: 32)
        _ = SecRandomCopyBytes(kSecRandomDefault, buffer.count, &buffer)
        return Data(buffer).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    private func generateCodeChallenge(from verifier: String) -> String {
        guard let data = verifier.data(using: .utf8) else { return "" }
        let hashed = SHA256.hash(data: data)
        return Data(hashed).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    private func extractCode(from url: URL) -> String? {
        URLComponents(url: url, resolvingAgainstBaseURL: false)?
            .queryItems?
            .first(where: { $0.name == "code" })?
            .value
    }

    private func extractScheme(from urlString: String) -> String {
        urlString.components(separatedBy: "://").first ?? ""
    }
}

extension OAuthService: ASWebAuthenticationPresentationContextProviding {
    func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
        UIApplication.shared.windows.first { $0.isKeyWindow }!
    }
}

struct OAuthTokenResponse: Codable {
    let accessToken: String
    let refreshToken: String?
    let expiresIn: Int
    let tokenType: String

    enum CodingKeys: String, CodingKey {
        case accessToken = "access_token"
        case refreshToken = "refresh_token"
        case expiresIn = "expires_in"
        case tokenType = "token_type"
    }
}

enum OAuthError: Error {
    case invalidCallback
    case missingCodeVerifier
    case tokenExchangeFailed
}
```

---

## JWT (JSON Web Token)

### JWT実装

```swift
import CryptoKit

struct JWT {
    let header: Header
    let payload: Payload
    let signature: String

    struct Header: Codable {
        let alg: String
        let typ: String
    }

    struct Payload: Codable {
        let sub: String // Subject (User ID)
        let iss: String // Issuer
        let aud: String // Audience
        let exp: Int    // Expiration Time
        let iat: Int    // Issued At
        let jti: String // JWT ID

        var isExpired: Bool {
            Date().timeIntervalSince1970 > Double(exp)
        }

        var expirationDate: Date {
            Date(timeIntervalSince1970: Double(exp))
        }
    }

    static func decode(_ token: String) throws -> JWT {
        let parts = token.split(separator: ".")

        guard parts.count == 3 else {
            throw JWTError.invalidFormat
        }

        // Header
        let headerData = try base64URLDecode(String(parts[0]))
        let header = try JSONDecoder().decode(Header.self, from: headerData)

        // Payload
        let payloadData = try base64URLDecode(String(parts[1]))
        let payload = try JSONDecoder().decode(Payload.self, from: payloadData)

        // Signature
        let signature = String(parts[2])

        return JWT(header: header, payload: payload, signature: signature)
    }

    func validate(with secret: String) throws {
        // 有効期限チェック
        guard !payload.isExpired else {
            throw JWTError.expired
        }

        // 署名検証
        let message = "\(encodeBase64URL(header)).\(encodeBase64URL(payload))"
        let expectedSignature = try sign(message, with: secret)

        guard signature == expectedSignature else {
            throw JWTError.invalidSignature
        }
    }

    private func sign(_ message: String, with secret: String) throws -> String {
        guard let messageData = message.data(using: .utf8),
              let secretData = secret.data(using: .utf8) else {
            throw JWTError.encodingFailed
        }

        let key = SymmetricKey(data: secretData)
        let signature = HMAC<SHA256>.authenticationCode(for: messageData, using: key)

        return Data(signature).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    private static func base64URLDecode(_ string: String) throws -> Data {
        var base64 = string
            .replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")

        // パディング追加
        let remainder = base64.count % 4
        if remainder > 0 {
            base64 += String(repeating: "=", count: 4 - remainder)
        }

        guard let data = Data(base64Encoded: base64) else {
            throw JWTError.decodingFailed
        }

        return data
    }

    private func encodeBase64URL<T: Encodable>(_ value: T) -> String {
        guard let data = try? JSONEncoder().encode(value) else { return "" }

        return data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }
}

enum JWTError: Error {
    case invalidFormat
    case expired
    case invalidSignature
    case encodingFailed
    case decodingFailed
}

// JWT Manager
class JWTManager {
    private let keychain: KeychainService

    init(keychain: KeychainService) {
        self.keychain = keychain
    }

    func saveToken(_ token: String) throws {
        // トークンを検証してから保存
        let jwt = try JWT.decode(token)

        guard !jwt.payload.isExpired else {
            throw JWTError.expired
        }

        try keychain.save(token, for: .accessToken)
    }

    func getToken() throws -> String? {
        guard let token = try? keychain.get(.accessToken) else {
            return nil
        }

        // 有効期限チェック
        let jwt = try JWT.decode(token)

        guard !jwt.payload.isExpired else {
            // 期限切れトークンを削除
            try? keychain.delete(.accessToken)
            throw JWTError.expired
        }

        return token
    }

    func isTokenValid() -> Bool {
        guard let token = try? getToken(),
              let jwt = try? JWT.decode(token) else {
            return false
        }

        return !jwt.payload.isExpired
    }
}
```

---

## Biometric認証

### Face ID / Touch ID実装

```swift
import LocalAuthentication

class BiometricAuthService {
    private let context = LAContext()

    enum BiometricType {
        case none
        case touchID
        case faceID

        var displayName: String {
            switch self {
            case .none: return "None"
            case .touchID: return "Touch ID"
            case .faceID: return "Face ID"
            }
        }
    }

    var biometricType: BiometricType {
        var error: NSError?

        guard context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error) else {
            return .none
        }

        switch context.biometryType {
        case .touchID:
            return .touchID
        case .faceID:
            return .faceID
        case .none:
            return .none
        @unknown default:
            return .none
        }
    }

    func authenticate(reason: String) async throws -> Bool {
        try await withCheckedThrowingContinuation { continuation in
            context.evaluatePolicy(
                .deviceOwnerAuthenticationWithBiometrics,
                localizedReason: reason
            ) { success, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: success)
                }
            }
        }
    }

    func authenticateWithFallback(reason: String) async throws -> Bool {
        try await withCheckedThrowingContinuation { continuation in
            context.evaluatePolicy(
                .deviceOwnerAuthentication, // パスコードフォールバック
                localizedReason: reason
            ) { success, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: success)
                }
            }
        }
    }
}

// Biometric付きログイン
class BiometricLoginService {
    private let biometric: BiometricAuthService
    private let keychain: KeychainService
    private let authService: AuthenticationService

    init(
        biometric: BiometricAuthService,
        keychain: KeychainService,
        authService: AuthenticationService
    ) {
        self.biometric = biometric
        self.keychain = keychain
        self.authService = authService
    }

    func enableBiometricLogin(email: String, password: String) async throws {
        // 通常のログイン
        let token = try await authService.login(email: email, password: password)

        // Biometric認証のテスト
        let success = try await biometric.authenticate(
            reason: "Enable \(biometric.biometricType.displayName) for quick login"
        )

        guard success else {
            throw BiometricError.authenticationFailed
        }

        // 認証情報をBiometric保護付きでKeychainに保存
        try saveBiometricCredentials(email: email, password: password)
    }

    func loginWithBiometric() async throws -> AuthToken {
        // Biometric認証
        let success = try await biometric.authenticate(
            reason: "Login with \(biometric.biometricType.displayName)"
        )

        guard success else {
            throw BiometricError.authenticationFailed
        }

        // 保存された認証情報を取得
        let (email, password) = try getBiometricCredentials()

        // ログイン
        return try await authService.login(email: email, password: password)
    }

    private func saveBiometricCredentials(email: String, password: String) throws {
        // Biometric保護付きKeychain保存
        let accessControl = SecAccessControlCreateWithFlags(
            nil,
            kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            .biometryCurrentSet,
            nil
        )!

        let emailQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: "biometric_email",
            kSecValueData as String: email.data(using: .utf8)!,
            kSecAttrAccessControl as String: accessControl
        ]

        let passwordQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: "biometric_password",
            kSecValueData as String: password.data(using: .utf8)!,
            kSecAttrAccessControl as String: accessControl
        ]

        SecItemDelete(emailQuery as CFDictionary)
        SecItemDelete(passwordQuery as CFDictionary)

        SecItemAdd(emailQuery as CFDictionary, nil)
        SecItemAdd(passwordQuery as CFDictionary, nil)
    }

    private func getBiometricCredentials() throws -> (email: String, password: String) {
        // Keychainから取得（Biometric認証が必要）
        let emailQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: "biometric_email",
            kSecReturnData as String: true
        ]

        let passwordQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: "biometric_password",
            kSecReturnData as String: true
        ]

        var emailResult: AnyObject?
        var passwordResult: AnyObject?

        guard SecItemCopyMatching(emailQuery as CFDictionary, &emailResult) == errSecSuccess,
              SecItemCopyMatching(passwordQuery as CFDictionary, &passwordResult) == errSecSuccess,
              let emailData = emailResult as? Data,
              let passwordData = passwordResult as? Data,
              let email = String(data: emailData, encoding: .utf8),
              let password = String(data: passwordData, encoding: .utf8) else {
            throw BiometricError.credentialsNotFound
        }

        return (email, password)
    }
}

enum BiometricError: Error {
    case authenticationFailed
    case credentialsNotFound
    case notAvailable
}
```

---

## Sign in with Apple

### Sign in with Apple実装

```swift
import AuthenticationServices

class AppleSignInService: NSObject {
    private var continuation: CheckedContinuation<AppleIDCredential, Error>?

    func signIn() async throws -> AppleIDCredential {
        try await withCheckedThrowingContinuation { continuation in
            self.continuation = continuation

            let request = ASAuthorizationAppleIDProvider().createRequest()
            request.requestedScopes = [.fullName, .email]

            let controller = ASAuthorizationController(authorizationRequests: [request])
            controller.delegate = self
            controller.presentationContextProvider = self
            controller.performRequests()
        }
    }

    func checkCredentialState(userID: String) async throws -> ASAuthorizationAppleIDProvider.CredentialState {
        try await withCheckedThrowingContinuation { continuation in
            ASAuthorizationAppleIDProvider().getCredentialState(forUserID: userID) { state, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: state)
                }
            }
        }
    }
}

extension AppleSignInService: ASAuthorizationControllerDelegate {
    func authorizationController(
        controller: ASAuthorizationController,
        didCompleteWithAuthorization authorization: ASAuthorization
    ) {
        guard let credential = authorization.credential as? ASAuthorizationAppleIDCredential else {
            continuation?.resume(throwing: AppleSignInError.invalidCredential)
            return
        }

        continuation?.resume(returning: AppleIDCredential(credential: credential))
    }

    func authorizationController(
        controller: ASAuthorizationController,
        didCompleteWithError error: Error
    ) {
        continuation?.resume(throwing: error)
    }
}

extension AppleSignInService: ASAuthorizationControllerPresentationContextProviding {
    func presentationAnchor(for controller: ASAuthorizationController) -> ASPresentationAnchor {
        UIApplication.shared.windows.first { $0.isKeyWindow }!
    }
}

struct AppleIDCredential {
    let userID: String
    let fullName: PersonNameComponents?
    let email: String?
    let identityToken: String?
    let authorizationCode: String?

    init(credential: ASAuthorizationAppleIDCredential) {
        self.userID = credential.user
        self.fullName = credential.fullName
        self.email = credential.email

        if let tokenData = credential.identityToken {
            self.identityToken = String(data: tokenData, encoding: .utf8)
        } else {
            self.identityToken = nil
        }

        if let codeData = credential.authorizationCode {
            self.authorizationCode = String(data: codeData, encoding: .utf8)
        } else {
            self.authorizationCode = nil
        }
    }
}

enum AppleSignInError: Error {
    case invalidCredential
    case cancelled
}

// SwiftUI View
struct SignInWithAppleButton: View {
    let onSignIn: (AppleIDCredential) -> Void

    @StateObject private var appleSignIn = AppleSignInService()

    var body: some View {
        Button(action: {
            Task {
                do {
                    let credential = try await appleSignIn.signIn()
                    onSignIn(credential)
                } catch {
                    print("Sign in failed: \(error)")
                }
            }
        }) {
            Label("Sign in with Apple", systemImage: "applelogo")
                .font(.headline)
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.black)
                .cornerRadius(10)
        }
    }
}
```

このガイドでは、iOS認証実装の基礎から、OAuth 2.0、JWT、Biometric認証、Sign in with Appleまで、包括的な認証機能の実装方法を網羅しました。セキュアな認証実装は、ユーザーデータ保護の基盤となる重要な要素です。

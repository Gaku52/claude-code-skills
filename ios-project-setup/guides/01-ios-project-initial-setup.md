# iOS Project Initial Setup - 完全ガイド

## 目次

1. [プロジェクト初期化の概要](#プロジェクト初期化の概要)
2. [Xcodeプロジェクト作成](#xcodeプロジェクト作成)
3. [プロジェクト構造設計](#プロジェクト構造設計)
4. [Build Settings最適化](#build-settings最適化)
5. [Scheme Configuration](#scheme-configuration)
6. [Asset Management](#asset-management)
7. [Info.plist設定](#infoplist設定)
8. [Code Signing設定](#code-signing設定)
9. [Git セットアップ](#gitセットアップ)
10. [依存関係管理](#依存関係管理)
11. [Xcconfig活用](#xcconfig活用)
12. [テスト環境構築](#テスト環境構築)
13. [CI/CD初期設定](#cicd初期設定)
14. [チーム開発準備](#チーム開発準備)
15. [トラブルシューティング](#トラブルシューティング)

---

## プロジェクト初期化の概要

### なぜ適切な初期設定が重要か

iOSプロジェクトの初期設定は、長期的な開発効率と保守性に大きな影響を与えます。適切な初期設定により以下のメリットが得られます：

- **開発効率の向上**: 統一された構造により、コードの配置場所が明確
- **保守性の向上**: 拡張しやすい構造で、機能追加がスムーズ
- **チーム協業の円滑化**: 統一されたルールで、レビューやマージが容易
- **CI/CDの導入**: 自動化しやすい構造で、デプロイが安全
- **技術的負債の削減**: 初期から適切な設計で、リファクタリングコストを削減

### プロジェクト作成前のチェックリスト

```bash
# プロジェクト作成前の確認事項

## ビジネス要件
- [ ] アプリ名の決定
- [ ] ターゲットユーザーの明確化
- [ ] 主要機能のリストアップ
- [ ] リリース時期の目標設定
- [ ] 予算とリソースの確認

## 技術要件
- [ ] 対応iOSバージョンの決定
- [ ] UI Framework選定（SwiftUI / UIKit / 混合）
- [ ] アーキテクチャの選定（MVVM / Clean Architecture / VIPER）
- [ ] 外部サービスの選定（Firebase / AWS / Azure）
- [ ] 必須ライブラリのリストアップ

## チーム体制
- [ ] 開発メンバーの確認
- [ ] ロールと責任の明確化
- [ ] コミュニケーションツールの選定
- [ ] コードレビュープロセスの決定
- [ ] ブランチ戦略の決定

## 環境準備
- [ ] Xcode最新版のインストール
- [ ] Apple Developer Programへの加入
- [ ] Git リポジトリの作成
- [ ] プロジェクト管理ツールのセットアップ（Jira / GitHub Projects）
- [ ] デザインツールへのアクセス（Figma / Sketch）
```

### プロジェクト命名規則

```swift
// アプリ命名のベストプラクティス

/*
プロジェクト名:
- PascalCase を使用: MyAwesomeApp
- スペース不可: ❌ My Awesome App → ✅ MyAwesomeApp
- 特殊文字不可: ❌ My-App, My_App
- 簡潔で覚えやすい名前
- 検索可能な名前（一般的すぎない）

Bundle Identifier:
- リバースドメイン形式: com.company.appname
- 小文字推奨: com.company.myawesomeapp
- ハイフン不可: ❌ com.my-company.app
- アンダースコア可: ✅ com.company.my_app
- サブドメイン活用: com.company.ios.appname

環境別 Bundle Identifier:
- Production: com.company.myapp
- Staging:    com.company.myapp.staging
- Development: com.company.myapp.dev
- Beta:       com.company.myapp.beta

Target名:
- Main Target: MyApp
- Unit Tests: MyAppTests
- UI Tests: MyAppUITests
- Shared Framework: MyAppKit
- Extensions: MyAppWidgetExtension
*/

// 例: 実際のアプリケーション
/*
Twitter:
- Project: Twitter
- Bundle ID: com.twitter.twitter
- Display Name: Twitter

Instagram:
- Project: Instagram
- Bundle ID: com.instagram.instagram
- Display Name: Instagram

Spotify:
- Project: Spotify
- Bundle ID: com.spotify.spotify
- Display Name: Spotify
*/
```

---

## Xcodeプロジェクト作成

### ステップバイステップガイド

#### Step 1: Xcodeプロジェクトの新規作成

```bash
# Xcode起動
1. Xcode.app を起動
2. "Create a new Xcode project" を選択
3. テンプレート選択画面へ

# テンプレート選択
Platform: iOS
Template: App

# 推奨テンプレート:
- シンプルなアプリ: App
- ゲーム: Game
- AR アプリ: Augmented Reality App
- Document Based: Document App
```

#### Step 2: プロジェクト設定

```swift
// Project Options

/*
Product Name: MyApp
- アプリのプロジェクト名
- PascalCase推奨
- スペースなし

Team: Your Apple Developer Team
- Apple Developer アカウント
- 個人開発者 or 組織

Organization Identifier: com.company
- リバースドメイン形式
- Bundle IDのプレフィックスになる

Bundle Identifier: com.company.MyApp
- 自動生成される
- 世界で一意である必要がある

Language: Swift
- Swift 推奨（Objective-Cとの混在も可能）

User Interface: SwiftUI
- SwiftUI: モダンな宣言的UI
- Storyboard: 従来型のUI（UIKit）

Include Tests: ✅
- Unit Test Target を自動生成
- 必須: チェック推奨

Include UI Tests: ✅
- UI Test Target を自動生成
- 推奨: 初期からテスト基盤を用意

Storage: None / Core Data / SwiftData
- None: 手動でデータベース選択
- Core Data: Apple純正ORM
- SwiftData: iOS 17+ の新しいデータ管理
*/
```

#### Step 3: 保存場所の選択

```bash
# プロジェクト保存場所の決定

推奨ディレクトリ構造:
~/Development/
├── Personal/
│   └── MyApp/
├── Company/
│   ├── ProductA/
│   └── ProductB/
└── OpenSource/
    └── MyLibrary/

# Git リポジトリの初期化
☑️ Create Git repository on my Mac
- チェック推奨: 自動的にGit初期化

# 保存
1. 適切なディレクトリを選択
2. "Create" をクリック
3. プロジェクトが開く
```

### プロジェクト作成直後の設定

#### General タブの設定

```swift
// TARGETS > MyApp > General

/*
Identity:
- Display Name: MyApp
  → ホーム画面に表示される名前
  → 環境ごとに変更可能: "MyApp Dev", "MyApp Staging"

- Bundle Identifier: com.company.myapp
  → App Store で一意
  → 変更不可（一度公開したら）

- Version: 1.0.0
  → セマンティックバージョニング
  → マーケティングバージョン（ユーザー向け）

- Build: 1
  → ビルド番号（内部管理用）
  → リリースごとにインクリメント
  → TestFlight では必須

Deployment Info:
- iOS Deployment Target: 15.0
  → 対応する最小iOSバージョン
  → 推奨: iOS 15.0+ （2024年時点）
  → トレードオフ: 新機能 vs ユーザーカバレッジ

- iPhone, iPad
  → 対応デバイス
  → iPhone のみ / iPad のみ / Universal

- Device Orientation
  → Portrait: 縦向き（推奨: ON）
  → Landscape Left/Right: 横向き
  → Upside Down: 上下逆（iPhone: OFF推奨）

- Status Bar Style
  → Default: システムデフォルト
  → Light/Dark: 明暗指定

App Icons and Launch Screen:
- App Icon Source: AppIcon
  → Assets.xcassets 内のアイコンセット
  → 必須サイズ: 1024x1024 (@1x)

- Launch Screen: LaunchScreen
  → 起動画面
  → SwiftUI or Storyboard
*/
```

#### Signing & Capabilities タブの設定

```swift
// TARGETS > MyApp > Signing & Capabilities

/*
Signing:

Automatically manage signing: ✅ (推奨: 開発初期)
- Xcode が自動的に証明書とプロビジョニングプロファイルを管理
- メリット: 設定が簡単、エラーが少ない
- デメリット: CI/CD での制御が難しい

Team: Your Team Name
- Apple Developer アカウント
- Personal Team（無料アカウント）も選択可能
  → ただし実機テストは制限あり

Bundle Identifier: com.company.myapp
- General タブと同期

Provisioning Profile: Xcode Managed Profile
- 自動生成される
- 開発用 / AdHoc / App Store 用が自動選択

Capabilities: （必要に応じて追加）

+ Capability から追加:
  - Push Notifications: プッシュ通知
  - Background Modes: バックグラウンド実行
  - Sign in with Apple: Apple ID ログイン
  - App Groups: アプリ間データ共有
  - Associated Domains: Universal Links
  - iCloud: クラウドストレージ
  - HealthKit: 健康データアクセス
  - HomeKit: スマートホーム連携
  - Wallet: Apple Pay
  - Game Center: ゲーム機能
*/
```

#### Build Settings タブの重要設定

```swift
// TARGETS > MyApp > Build Settings

/*
重要な設定項目:

Swift Compiler - Language:
- Swift Language Version: Swift 5
  → 最新の Swift バージョンを使用

Swift Compiler - Code Generation:
- Optimization Level:
  → Debug: No Optimization [-Onone]
  → Release: Optimize for Speed [-O]

- Compilation Mode:
  → Debug: Incremental（差分ビルド）
  → Release: Whole Module（最適化優先）

Deployment:
- iOS Deployment Target: 15.0
  → サポート最小バージョン

- Strip Debug Symbols During Copy:
  → Debug: No
  → Release: Yes（アプリサイズ削減）

- Make Strings Read-Only: Yes
  → セキュリティ向上

Architectures:
- Build Active Architecture Only:
  → Debug: Yes（ビルド時間短縮）
  → Release: No（全アーキテクチャ対応）

- Excluded Architectures: i386, x86_64
  → Simulatorのみの古いアーキテクチャを除外

Search Paths:
- Framework Search Paths: $(inherited)
  → SPM / CocoaPods が使用

- Header Search Paths: $(inherited)
  → C/Objective-C ヘッダー

Build Options:
- Enable Bitcode: No
  → iOS 15+ では非推奨
  → アプリサイズに影響なし

- Debug Information Format:
  → Debug: DWARF
  → Release: DWARF with dSYM File
  → クラッシュレポート解析に必須

Packaging:
- Product Name: $(TARGET_NAME)
  → アプリの内部名

- Product Bundle Identifier: com.company.myapp
  → Bundle ID

User-Defined:
（カスタム設定は xcconfig ファイルで管理推奨）
*/
```

---

## プロジェクト構造設計

### MVVM アーキテクチャによるフォルダ構成

```
MyApp/
├── App/
│   ├── MyApp.swift                    # @main App Entry Point (SwiftUI)
│   ├── AppDelegate.swift              # UIKit Lifecycle (Optional)
│   └── SceneDelegate.swift            # Scene Management (Optional)
│
├── Features/                          # Feature-Based Organization
│   ├── Authentication/
│   │   ├── Views/
│   │   │   ├── LoginView.swift
│   │   │   ├── SignUpView.swift
│   │   │   └── ForgotPasswordView.swift
│   │   ├── ViewModels/
│   │   │   ├── LoginViewModel.swift
│   │   │   └── SignUpViewModel.swift
│   │   ├── Models/
│   │   │   ├── User.swift
│   │   │   └── AuthCredentials.swift
│   │   └── Services/
│   │       └── AuthenticationService.swift
│   │
│   ├── Home/
│   │   ├── Views/
│   │   │   ├── HomeView.swift
│   │   │   ├── FeedView.swift
│   │   │   └── Components/
│   │   │       ├── FeedItemView.swift
│   │   │       └── EmptyStateView.swift
│   │   ├── ViewModels/
│   │   │   └── HomeViewModel.swift
│   │   └── Models/
│   │       └── FeedItem.swift
│   │
│   ├── Profile/
│   │   ├── Views/
│   │   │   ├── ProfileView.swift
│   │   │   └── EditProfileView.swift
│   │   ├── ViewModels/
│   │   │   └── ProfileViewModel.swift
│   │   └── Models/
│   │       └── UserProfile.swift
│   │
│   └── Settings/
│       ├── Views/
│       │   ├── SettingsView.swift
│       │   ├── AccountSettingsView.swift
│       │   └── PrivacySettingsView.swift
│       └── ViewModels/
│           └── SettingsViewModel.swift
│
├── Core/                              # Shared Core Components
│   ├── Networking/
│   │   ├── HTTPClient.swift           # Generic HTTP Client
│   │   ├── APIClient.swift            # App-specific API Client
│   │   ├── Endpoint.swift             # API Endpoints Definition
│   │   ├── NetworkError.swift         # Error Types
│   │   └── RequestBuilder.swift       # Request Construction
│   │
│   ├── Database/
│   │   ├── CoreDataStack.swift        # Core Data Setup
│   │   ├── DatabaseManager.swift      # Database Operations
│   │   ├── Entities/
│   │   │   └── MyAppModel.xcdatamodeld
│   │   └── Migrations/
│   │       └── MigrationPolicy.swift
│   │
│   ├── Services/
│   │   ├── LocationService.swift      # Location Manager
│   │   ├── NotificationService.swift  # Push Notifications
│   │   ├── AnalyticsService.swift     # Analytics Tracking
│   │   ├── CrashReportingService.swift # Crash Reports
│   │   └── RemoteConfigService.swift  # Feature Flags
│   │
│   ├── Repositories/
│   │   ├── UserRepository.swift       # User Data Repository
│   │   ├── ContentRepository.swift    # Content Repository
│   │   └── Protocols/
│   │       ├── Repository.swift
│   │       └── DataSource.swift
│   │
│   └── Storage/
│       ├── UserDefaults+Extension.swift
│       ├── KeychainManager.swift      # Secure Storage
│       └── FileManager+Extension.swift
│
├── Common/                            # Common/Shared Components
│   ├── Extensions/
│   │   ├── Foundation/
│   │   │   ├── String+Extensions.swift
│   │   │   ├── Date+Extensions.swift
│   │   │   ├── URL+Extensions.swift
│   │   │   └── Data+Extensions.swift
│   │   ├── UIKit/
│   │   │   ├── UIView+Extensions.swift
│   │   │   ├── UIViewController+Extensions.swift
│   │   │   ├── UIColor+Extensions.swift
│   │   │   └── UIImage+Extensions.swift
│   │   └── SwiftUI/
│   │       ├── View+Extensions.swift
│   │       ├── Color+Extensions.swift
│   │       └── Font+Extensions.swift
│   │
│   ├── Utilities/
│   │   ├── Logger.swift               # Logging Utility
│   │   ├── Validator.swift            # Input Validation
│   │   ├── DateFormatter.swift        # Date Formatting
│   │   ├── ImageLoader.swift          # Image Loading/Caching
│   │   └── Debouncer.swift            # Debounce Utility
│   │
│   ├── Constants/
│   │   ├── AppConstants.swift         # App-wide Constants
│   │   ├── APIConstants.swift         # API URLs/Keys
│   │   ├── ColorPalette.swift         # Color Definitions
│   │   └── FontFamily.swift           # Font Definitions
│   │
│   └── Helpers/
│       ├── KeyboardHelper.swift       # Keyboard Management
│       ├── HapticHelper.swift         # Haptic Feedback
│       └── BiometricHelper.swift      # Face ID / Touch ID
│
├── UI/                                # UI Components
│   ├── Components/                    # Reusable UI Components
│   │   ├── Buttons/
│   │   │   ├── PrimaryButton.swift
│   │   │   ├── SecondaryButton.swift
│   │   │   └── IconButton.swift
│   │   ├── TextFields/
│   │   │   ├── CustomTextField.swift
│   │   │   └── SearchTextField.swift
│   │   ├── Cards/
│   │   │   ├── ContentCard.swift
│   │   │   └── UserCard.swift
│   │   ├── LoadingViews/
│   │   │   ├── LoadingSpinner.swift
│   │   │   └── SkeletonView.swift
│   │   └── EmptyStates/
│   │       ├── EmptyListView.swift
│   │       └── ErrorView.swift
│   │
│   ├── Modifiers/                     # Custom View Modifiers
│   │   ├── CardModifier.swift
│   │   ├── ShimmerModifier.swift
│   │   └── PulseAnimationModifier.swift
│   │
│   └── Styles/                        # Custom Styles
│       ├── ButtonStyles.swift
│       ├── TextFieldStyles.swift
│       └── ToggleStyles.swift
│
├── Navigation/                        # Navigation Logic
│   ├── Coordinator.swift              # Coordinator Protocol
│   ├── AppCoordinator.swift           # Root Coordinator
│   ├── TabCoordinator.swift           # Tab Navigation
│   └── Routes.swift                   # Route Definitions
│
├── Resources/                         # Resource Files
│   ├── Assets.xcassets/               # Image Assets
│   │   ├── AppIcon.appiconset
│   │   ├── Colors/
│   │   │   ├── Primary.colorset
│   │   │   ├── Secondary.colorset
│   │   │   └── Background.colorset
│   │   ├── Images/
│   │   │   ├── Logo.imageset
│   │   │   └── Placeholder.imageset
│   │   └── Icons/
│   │       ├── Home.imageset
│   │       ├── Profile.imageset
│   │       └── Settings.imageset
│   │
│   ├── Fonts/                         # Custom Fonts
│   │   ├── CustomFont-Regular.ttf
│   │   ├── CustomFont-Bold.ttf
│   │   └── CustomFont-Light.ttf
│   │
│   ├── Localizable/                   # Localization
│   │   ├── en.lproj/
│   │   │   └── Localizable.strings
│   │   ├── ja.lproj/
│   │   │   └── Localizable.strings
│   │   └── es.lproj/
│   │       └── Localizable.strings
│   │
│   └── Configurations/                # Configuration Files
│       ├── GoogleService-Info-Dev.plist
│       ├── GoogleService-Info-Staging.plist
│       └── GoogleService-Info-Prod.plist
│
├── Config/                            # Build Configurations
│   ├── Base.xcconfig
│   ├── Debug.xcconfig
│   ├── Staging.xcconfig
│   └── Release.xcconfig
│
├── Supporting Files/
│   ├── Info.plist
│   └── MyApp.entitlements
│
├── MyAppTests/                        # Unit Tests
│   ├── Features/
│   │   ├── AuthenticationTests/
│   │   │   ├── LoginViewModelTests.swift
│   │   │   └── AuthServiceTests.swift
│   │   └── HomeTests/
│   │       └── HomeViewModelTests.swift
│   │
│   ├── Core/
│   │   ├── NetworkingTests/
│   │   │   ├── HTTPClientTests.swift
│   │   │   └── APIClientTests.swift
│   │   └── RepositoryTests/
│   │       └── UserRepositoryTests.swift
│   │
│   ├── Mocks/
│   │   ├── MockAPIClient.swift
│   │   ├── MockUserRepository.swift
│   │   └── MockData.swift
│   │
│   └── Helpers/
│       └── XCTestCase+Extensions.swift
│
└── MyAppUITests/                      # UI Tests
    ├── Flows/
    │   ├── LoginFlowUITests.swift
    │   └── OnboardingFlowUITests.swift
    ├── Screens/
    │   ├── HomeScreenUITests.swift
    │   └── ProfileScreenUITests.swift
    └── Helpers/
        ├── UITestHelper.swift
        └── ScreenObjects/
            ├── LoginScreen.swift
            └── HomeScreen.swift
```

### Clean Architecture によるフォルダ構成

```
MyApp/ (Clean Architecture Version)
├── Domain/                            # Business Logic Layer
│   ├── Entities/
│   │   ├── User.swift
│   │   ├── Post.swift
│   │   └── Comment.swift
│   │
│   ├── UseCases/
│   │   ├── Authentication/
│   │   │   ├── LoginUseCase.swift
│   │   │   ├── LogoutUseCase.swift
│   │   │   └── RefreshTokenUseCase.swift
│   │   ├── User/
│   │   │   ├── FetchUserUseCase.swift
│   │   │   ├── UpdateUserUseCase.swift
│   │   │   └── DeleteUserUseCase.swift
│   │   └── Content/
│   │       ├── FetchPostsUseCase.swift
│   │       └── CreatePostUseCase.swift
│   │
│   └── RepositoryInterfaces/
│       ├── UserRepositoryProtocol.swift
│       ├── PostRepositoryProtocol.swift
│       └── AuthRepositoryProtocol.swift
│
├── Data/                              # Data Layer
│   ├── Repositories/
│   │   ├── UserRepository.swift
│   │   ├── PostRepository.swift
│   │   └── AuthRepository.swift
│   │
│   ├── DataSources/
│   │   ├── Remote/
│   │   │   ├── UserRemoteDataSource.swift
│   │   │   ├── PostRemoteDataSource.swift
│   │   │   └── API/
│   │   │       ├── APIClient.swift
│   │   │       └── Endpoints.swift
│   │   └── Local/
│   │       ├── UserLocalDataSource.swift
│   │       ├── PostLocalDataSource.swift
│   │       └── CoreData/
│   │           ├── CoreDataStack.swift
│   │           └── MyApp.xcdatamodeld
│   │
│   └── DTOs/                          # Data Transfer Objects
│       ├── UserDTO.swift
│       ├── PostDTO.swift
│       └── Mappers/
│           ├── UserMapper.swift
│           └── PostMapper.swift
│
└── Presentation/                      # Presentation Layer
    ├── Features/
    │   ├── Authentication/
    │   │   ├── Views/
    │   │   ├── ViewModels/
    │   │   └── Coordinator/
    │   ├── Home/
    │   └── Profile/
    │
    └── Common/
        ├── Components/
        ├── Extensions/
        └── Utilities/
```

### ファイル配置の実装例

```swift
// Features/Authentication/Views/LoginView.swift

import SwiftUI

struct LoginView: View {
    @StateObject private var viewModel: LoginViewModel

    var body: some View {
        VStack(spacing: 20) {
            // UI Implementation
        }
        .onAppear {
            viewModel.onAppear()
        }
    }
}

// Features/Authentication/ViewModels/LoginViewModel.swift

import Foundation
import Combine

@MainActor
final class LoginViewModel: ObservableObject {
    @Published var email: String = ""
    @Published var password: String = ""
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?

    private let loginUseCase: LoginUseCase
    private var cancellables = Set<AnyCancellable>()

    init(loginUseCase: LoginUseCase) {
        self.loginUseCase = loginUseCase
    }

    func login() async {
        isLoading = true
        defer { isLoading = false }

        do {
            try await loginUseCase.execute(email: email, password: password)
            // Navigate to home
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

// Core/Networking/HTTPClient.swift

import Foundation

protocol HTTPClient {
    func request<T: Decodable>(
        endpoint: Endpoint,
        responseType: T.Type
    ) async throws -> T
}

final class URLSessionHTTPClient: HTTPClient {
    private let session: URLSession

    init(session: URLSession = .shared) {
        self.session = session
    }

    func request<T: Decodable>(
        endpoint: Endpoint,
        responseType: T.Type
    ) async throws -> T {
        let request = try endpoint.makeRequest()
        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw NetworkError.invalidResponse
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            throw NetworkError.statusCode(httpResponse.statusCode)
        }

        return try JSONDecoder().decode(T.self, from: data)
    }
}

enum NetworkError: Error {
    case invalidResponse
    case statusCode(Int)
    case decodingFailed
}

// Core/Networking/Endpoint.swift

import Foundation

struct Endpoint {
    let path: String
    let method: HTTPMethod
    let headers: [String: String]?
    let body: Data?
    let queryItems: [URLQueryItem]?

    func makeRequest() throws -> URLRequest {
        guard let url = URL(string: APIConstants.baseURL + path) else {
            throw NetworkError.invalidURL
        }

        var components = URLComponents(url: url, resolvingAgainstBaseURL: false)
        components?.queryItems = queryItems

        guard let finalURL = components?.url else {
            throw NetworkError.invalidURL
        }

        var request = URLRequest(url: finalURL)
        request.httpMethod = method.rawValue
        request.allHTTPHeaderFields = headers
        request.httpBody = body

        return request
    }
}

enum HTTPMethod: String {
    case get = "GET"
    case post = "POST"
    case put = "PUT"
    case delete = "DELETE"
    case patch = "PATCH"
}
```

---

## Build Settings最適化

### Debug vs Release 設定の違い

```swift
// Debug Configuration の最適化

/*
目的: 開発効率の最大化

Swift Optimization Level: -Onone
- コンパイル時間: 最速
- デバッグ: 容易（変数検査可能）
- パフォーマンス: 低（本番の 50-70%）
- 使用場面: 日常的な開発

Swift Compilation Mode: singlefile
- ファイル単位でコンパイル
- 変更ファイルのみ再コンパイル
- インクリメンタルビルドで高速

Enable Testability: YES
- テストコードからの internal アクセス可能
- @testable import MyApp

Debug Information Format: DWARF
- デバッグ情報を埋め込み
- dSYM ファイル不要（開発時）

Generate Debug Symbols: YES
- ブレークポイント、変数検査に必須

Active Compilation Conditions: DEBUG
- #if DEBUG で分岐可能
- デバッグ専用コード挿入

Preprocessor Macros: DEBUG=1
- Objective-C コードでの分岐用
*/

// Release Configuration の最適化

/*
目的: パフォーマンスとサイズの最適化

Swift Optimization Level: -O
- 最高速度での実行
- アグレッシブな最適化
- コンパイル時間: 遅い（5-10倍）

Swift Compilation Mode: wholemodule
- モジュール全体を最適化
- 関数のインライン展開
- デッドコード削除

Enable Testability: NO
- テストアクセス無効
- セキュリティ向上
- バイナリサイズ削減

Debug Information Format: DWARF with dSYM File
- クラッシュレポート解析用
- dSYM を Crashlytics / Firebase にアップロード

Strip Debug Symbols: YES
- デバッグシンボル削除
- アプリサイズ 20-30% 削減

Dead Code Stripping: YES
- 未使用コード削除
- リンク時最適化

Validate Product: YES
- App Store 提出前の検証
- 不正なコードを検出
*/

// 具体的な設定値

/*
Debug:
SWIFT_OPTIMIZATION_LEVEL = -Onone
SWIFT_COMPILATION_MODE = singlefile
ENABLE_TESTABILITY = YES
DEBUG_INFORMATION_FORMAT = dwarf
GCC_OPTIMIZATION_LEVEL = 0
COPY_PHASE_STRIP = NO
SWIFT_ACTIVE_COMPILATION_CONDITIONS = DEBUG
GCC_PREPROCESSOR_DEFINITIONS = DEBUG=1

Release:
SWIFT_OPTIMIZATION_LEVEL = -O
SWIFT_COMPILATION_MODE = wholemodule
ENABLE_TESTABILITY = NO
DEBUG_INFORMATION_FORMAT = dwarf-with-dsym
GCC_OPTIMIZATION_LEVEL = s
COPY_PHASE_STRIP = YES
STRIP_INSTALLED_PRODUCT = YES
DEAD_CODE_STRIPPING = YES
VALIDATE_PRODUCT = YES
LLVM_LTO = YES_THIN
*/
```

### パフォーマンス最適化設定

```swift
// Whole Module Optimization (WMO)

/*
設定: SWIFT_COMPILATION_MODE = wholemodule

効果:
1. コンパイラがモジュール全体を解析
2. ファイル間の関数インライン展開
3. デッドコード削除の精度向上
4. バイナリサイズ 5-15% 削減
5. 実行速度 10-20% 向上

トレードオフ:
- ビルド時間の大幅な増加（3-10倍）
- メモリ使用量の増加
- CI/CD での並列ビルドが困難

推奨:
- Debug: singlefile（開発速度優先）
- Release: wholemodule（パフォーマンス優先）
*/

// Link-Time Optimization (LTO)

/*
設定: LLVM_LTO = YES_THIN

種類:
1. YES_THIN (推奨)
   - 軽量版 LTO
   - ビルド時間 +20-50%
   - 効果は標準 LTO の 70-80%

2. YES (Full LTO)
   - 完全な LTO
   - ビルド時間 +100-300%
   - 最大限の最適化

効果:
- バイナリサイズ 5-10% 削減
- 実行速度 0-15% 向上
- デッドコード削除
- 関数のインライン展開

推奨:
- Debug: NO
- Release: YES_THIN
- 大規模プロジェクト: NO（ビルド時間考慮）
*/

// Size Optimization

/*
App サイズ削減のための設定:

1. Optimization Level for Size:
   GCC_OPTIMIZATION_LEVEL = s
   SWIFT_OPTIMIZATION_LEVEL = -Osize

2. Strip Symbols:
   STRIP_INSTALLED_PRODUCT = YES
   COPY_PHASE_STRIP = YES
   STRIP_SWIFT_SYMBOLS = YES

3. Dead Code Stripping:
   DEAD_CODE_STRIPPING = YES

4. Asset Catalog Optimization:
   ASSETCATALOG_COMPILER_OPTIMIZATION = space

5. Bitcode (非推奨):
   ENABLE_BITCODE = NO
   → iOS 14+ では無効化推奨

効果:
- 総削減: 20-40% のサイズ削減
- ダウンロードサイズの低減
- ユーザー体験の向上

注意:
- -Osize は -O より実行速度が 5-10% 遅い
- サイズと速度のトレードオフを考慮
*/
```

### ビルド時間の最適化

```swift
// ビルド時間計測と最適化

/*
1. ビルド時間の計測:

Build Settings に追加:
OTHER_SWIFT_FLAGS = -Xfrontend -debug-time-function-bodies

出力例:
123.4ms  @objc MyViewController.viewDidLoad()
45.6ms   MyViewModel.fetchData()
234.5ms  MyView.body.getter

2. 遅い箇所の最適化:

原因:
- 複雑な型推論
- 大きな SwiftUI body
- 複雑な式

対策:
- 型を明示的に指定
- body を分割
- 複雑な計算を分離

例:

// ❌ 遅い（型推論が複雑）
let result = data
    .filter { $0.isActive }
    .map { $0.name }
    .reduce("") { $0 + ", " + $1 }

// ✅ 速い（型を明示）
let filtered: [User] = data.filter { $0.isActive }
let names: [String] = filtered.map { $0.name }
let result: String = names.reduce("") { $0 + ", " + $1 }

// ❌ 遅い SwiftUI（body が大きい）
var body: some View {
    VStack {
        // 100行のコード
    }
}

// ✅ 速い SwiftUI（分割）
var body: some View {
    VStack {
        headerView
        contentView
        footerView
    }
}

private var headerView: some View {
    // ヘッダー部分
}

3. 並列ビルドの活用:

Xcode Preferences > Locations > Derived Data
ビルドシステム: New Build System

並列タスク数の設定:
defaults write com.apple.dt.Xcode IDEBuildOperationMaxNumberOfConcurrentCompileTasks 8

4. モジュール分割:

大規模プロジェクトの場合:
- Framework / Library に分割
- 変更頻度の低い部分をモジュール化
- 再ビルド範囲の削減

例:
MyApp
├── MyAppCore (Framework)
├── MyAppUI (Framework)
└── MyApp (App Target)
*/
```

---

## Scheme Configuration

### Scheme の役割と種類

```swift
// Scheme とは

/*
Scheme: ビルド・実行の設定セット

含まれる設定:
1. Build Configuration (Debug/Release/Custom)
2. Build Targets
3. Test Plans
4. Run/Test/Profile/Analyze/Archive の設定
5. Environment Variables
6. Arguments

使い分け:
- Development: 日常的な開発用
- Staging: QA・テスト環境用
- Production: App Store リリース用
- Testing: 自動テスト用
*/
```

### 推奨 Scheme 構成

```bash
# 3つの主要 Scheme

## 1. MyApp (Development)
Purpose: 日常的な開発
Configuration: Debug
Settings:
  - Run: Debug build
  - Faster builds (no optimization)
  - Debug logging enabled
  - Dev API endpoints
  - Test data enabled

## 2. MyApp (Staging)
Purpose: QA・内部テスト
Configuration: Staging (Custom)
Settings:
  - Run: Release-like build
  - Staging API endpoints
  - Analytics enabled
  - Crash reporting enabled
  - TestFlight distribution

## 3. MyApp (Production)
Purpose: App Store リリース
Configuration: Release
Settings:
  - Archive only
  - Production API endpoints
  - Full optimization
  - All analytics enabled
  - App Store distribution
```

### Scheme の作成手順

```bash
# Scheme 作成ステップ

1. Xcode > Product > Scheme > Manage Schemes...

2. Duplicate "MyApp" scheme
   - Name: "MyApp (Staging)"
   - Shared: ✅ (チーム共有)

3. Edit Scheme: "MyApp (Staging)"

4. Build タブ:
   - Targets: MyApp を選択
   - Find Implicit Dependencies: ✅

5. Run タブ:
   - Build Configuration: Staging
   - Debugger: LLDB
   - Debug executable: ✅ (必要に応じて)

   Info タブ:
   - Executable: MyApp.app
   - Launch: Automatically

   Arguments タブ:
   - Arguments Passed On Launch:
     -FIRDebugEnabled (Firebase debug)
     -com.apple.CoreData.SQLDebug 1 (Core Data debug)

   - Environment Variables:
     API_BASE_URL = https://staging.api.example.com
     ENVIRONMENT = staging
     LOG_LEVEL = debug

6. Test タブ:
   - Build Configuration: Debug
   - Test Plans: MyAppTests
   - Code Coverage: ✅

7. Profile タブ:
   - Build Configuration: Release
   - Use the Run action's arguments and environment

8. Analyze タブ:
   - Build Configuration: Debug

9. Archive タブ:
   - Build Configuration: Staging
   - Reveal Archive in Organizer: ✅
```

### Environment Variables の活用

```swift
// Environment Variables の設定

/*
Scheme > Edit Scheme > Run > Arguments > Environment Variables

設定例:

Name                    Value
------------------------|---------------------------------
API_BASE_URL            https://dev.api.example.com
ENVIRONMENT             development
LOG_LEVEL               verbose
ENABLE_MOCK_DATA        YES
DISABLE_ANIMATIONS      YES  # UI テスト用
SNAPSHOT_TESTING        YES  # スナップショットテスト
*/

// コードでの利用

struct Environment {
    static var apiBaseURL: String {
        ProcessInfo.processInfo.environment["API_BASE_URL"]
            ?? "https://api.example.com"
    }

    static var logLevel: LogLevel {
        let level = ProcessInfo.processInfo.environment["LOG_LEVEL"] ?? "error"
        return LogLevel(rawValue: level) ?? .error
    }

    static var isDebugMode: Bool {
        #if DEBUG
        return true
        #else
        return false
        #endif
    }

    static var isMockDataEnabled: Bool {
        ProcessInfo.processInfo.environment["ENABLE_MOCK_DATA"] == "YES"
    }
}

// 使用例
let apiClient = APIClient(baseURL: Environment.apiBaseURL)
Logger.shared.setLevel(Environment.logLevel)

if Environment.isMockDataEnabled {
    // モックデータを使用
    userRepository = MockUserRepository()
}
```

### Launch Arguments の活用

```swift
// Launch Arguments の設定と利用

/*
Scheme > Edit Scheme > Run > Arguments > Arguments Passed On Launch

設定例:
-FIRDebugEnabled                    # Firebase デバッグログ
-FIRAnalyticsDebugEnabled           # Firebase Analytics デバッグ
-com.apple.CoreData.SQLDebug 1      # Core Data SQL ログ
-com.apple.CoreData.MigrationDebug  # Core Data マイグレーション
*/

// コードでの確認
func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
) -> Bool {

    // UI テスト実行時の判定
    if CommandLine.arguments.contains("-uitesting") {
        // テストデータのセットアップ
        setupTestData()

        // アニメーション無効化
        UIView.setAnimationsEnabled(false)
    }

    // スナップショットテスト実行時
    if CommandLine.arguments.contains("-snapshot") {
        isSnapshotTesting = true
    }

    return true
}

// UI テストからの起動時に Launch Arguments を設定
final class MyAppUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false

        let app = XCUIApplication()
        app.launchArguments = [
            "-uitesting",
            "-FIRDebugEnabled"
        ]
        app.launch()
    }
}
```

---

## Asset Management

### Assets.xcassets の構成

```
Assets.xcassets/
├── AppIcon.appiconset/                # App Icon
│   ├── Contents.json
│   ├── Icon-1024.png                  # 1024x1024 (必須)
│   ├── Icon-20@2x.png                 # iPad Pro
│   ├── Icon-20@3x.png                 # iPhone
│   ├── Icon-29@2x.png
│   ├── Icon-29@3x.png
│   ├── Icon-40@2x.png
│   ├── Icon-40@3x.png
│   ├── Icon-60@2x.png
│   └── Icon-60@3x.png
│
├── AccentColor.colorset/              # Accent Color
│   └── Contents.json
│
├── Colors/                            # カラーパレット
│   ├── Primary.colorset/
│   │   └── Contents.json
│   ├── Secondary.colorset/
│   ├── Background.colorset/
│   ├── Surface.colorset/
│   ├── Error.colorset/
│   ├── Success.colorset/
│   └── Warning.colorset/
│
├── Images/                            # 画像アセット
│   ├── Logo.imageset/
│   │   ├── Contents.json
│   │   ├── Logo@2x.png
│   │   └── Logo@3x.png
│   ├── Placeholder.imageset/
│   └── EmptyState.imageset/
│
├── Icons/                             # アイコン
│   ├── TabBar/
│   │   ├── Home.imageset/
│   │   ├── Search.imageset/
│   │   ├── Profile.imageset/
│   │   └── Settings.imageset/
│   └── Navigation/
│       ├── Back.imageset/
│       ├── Close.imageset/
│       └── More.imageset/
│
└── Symbols/                           # SF Symbols (参照のみ)
    └── custom.symbol
```

### App Icon の設定

```swift
// App Icon 要件

/*
必須サイズ (iOS 15+):
- 1024x1024: App Store（@1x, 必須）
- 60x60: iPhone App（@2x, @3x）
- 20x20: iPhone Notification（@2x, @3x）
- 29x29: iPhone Settings（@2x, @3x）
- 40x40: iPhone Spotlight（@2x, @3x）

iPad サポート時:
- 76x76: iPad App（@2x）
- 20x20: iPad Notification（@1x, @2x）
- 29x29: iPad Settings（@1x, @2x）
- 40x40: iPad Spotlight（@1x, @2x）
- 83.5x83.5: iPad Pro（@2x）

デザインガイドライン:
1. 正方形（角丸は自動適用）
2. 透明度なし（アルファチャンネル不可）
3. フルカラー（RGB）
4. PNG 形式
5. 1024x1024 は必須（App Store 掲載用）

ツール:
- App Icon Generator: https://appicon.co
- SF Symbols: https://developer.apple.com/sf-symbols/
- Icon Set Creator: Xcode 内蔵ツール
*/

// AppIcon.appiconset/Contents.json の例

{
  "images" : [
    {
      "filename" : "Icon-20@2x.png",
      "idiom" : "iphone",
      "scale" : "2x",
      "size" : "20x20"
    },
    {
      "filename" : "Icon-20@3x.png",
      "idiom" : "iphone",
      "scale" : "3x",
      "size" : "20x20"
    },
    {
      "filename" : "Icon-60@2x.png",
      "idiom" : "iphone",
      "scale" : "2x",
      "size" : "60x60"
    },
    {
      "filename" : "Icon-60@3x.png",
      "idiom" : "iphone",
      "scale" : "3x",
      "size" : "60x60"
    },
    {
      "filename" : "Icon-1024.png",
      "idiom" : "ios-marketing",
      "scale" : "1x",
      "size" : "1024x1024"
    }
  ],
  "info" : {
    "author" : "xcode",
    "version" : 1
  }
}
```

### Color Assets の活用

```swift
// Color Assets の定義

// Colors/Primary.colorset/Contents.json
{
  "colors" : [
    {
      "color" : {
        "color-space" : "srgb",
        "components" : {
          "alpha" : "1.000",
          "blue" : "0.965",
          "green" : "0.361",
          "red" : "0.545"
        }
      },
      "idiom" : "universal"
    },
    {
      "appearances" : [
        {
          "appearance" : "luminosity",
          "value" : "dark"
        }
      ],
      "color" : {
        "color-space" : "srgb",
        "components" : {
          "alpha" : "1.000",
          "blue" : "0.965",
          "green" : "0.529",
          "red" : "0.678"
        }
      },
      "idiom" : "universal"
    }
  ],
  "info" : {
    "author" : "xcode",
    "version" : 1
  }
}

// SwiftUI での使用
struct ContentView: View {
    var body: some View {
        Text("Hello, World!")
            .foregroundColor(Color("Primary"))
            .background(Color("Background"))
    }
}

// UIKit での使用
let primaryColor = UIColor(named: "Primary")

// Color Extension で型安全に
extension Color {
    static let primary = Color("Primary")
    static let secondary = Color("Secondary")
    static let background = Color("Background")
    static let surface = Color("Surface")
}

// 使用
Text("Hello")
    .foregroundColor(.primary)
    .background(.surface)
```

### Image Assets の最適化

```swift
// Image Assets のベストプラクティス

/*
1. 解像度:
   - @1x: 基準サイズ（通常は不要）
   - @2x: Retina Display（必須）
   - @3x: Super Retina（iPhone 6+ 以降）

2. フォーマット:
   - PNG: アイコン、ロゴ（透過必要）
   - JPEG: 写真（透過不要、サイズ重視）
   - PDF/SVG: ベクター画像（Preserve Vector Data）
   - HEIC: iOS 11+ の高効率フォーマット

3. サイズ最適化:
   - TinyPNG: https://tinypng.com
   - ImageOptim: https://imageoptim.com
   - Xcode: Automatically で最適化

4. Render As:
   - Default: 通常の画像
   - Template: Tint Color 適用（アイコン）
   - Original: オリジナル（変更なし）

5. Asset Catalog Compiler Options:
   ASSETCATALOG_COMPILER_OPTIMIZATION = time  # ビルド時間優先
   ASSETCATALOG_COMPILER_OPTIMIZATION = space # サイズ優先
*/

// PDF Vector Images の活用

/*
メリット:
- 1ファイルで全解像度対応
- 拡大縮小しても品質劣化なし
- Asset Catalog での管理が簡単

設定:
1. PDF ファイルを Image Set にドラッグ
2. Attributes Inspector
   - Scales: Single Scale
   - Preserve Vector Data: ✅

Xcode が自動的に @2x, @3x を生成
*/

// SF Symbols の活用

/*
Apple 純正アイコンライブラリ

メリット:
- 4,000+ のアイコン
- Dynamic Type 対応
- カラー対応（iOS 15+）
- サイズ可変
- 軽量（バンドル不要）

使用例:
*/

// SwiftUI
Image(systemName: "heart.fill")
    .foregroundColor(.red)
    .font(.system(size: 32))

// UIKit
let config = UIImage.SymbolConfiguration(pointSize: 32, weight: .medium)
let image = UIImage(systemName: "heart.fill", withConfiguration: config)

// カスタム SF Symbol の作成
/*
1. SF Symbols App でテンプレートをダウンロード
2. ベクターグラフィックツールで編集
3. .svg をエクスポート
4. Assets.xcassets に追加
5. Render As: Template Image
*/
```

---

## Info.plist設定

### 必須設定項目

```xml
<!-- Info.plist の重要設定 -->

<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- アプリ基本情報 -->
    <key>CFBundleName</key>
    <string>$(PRODUCT_NAME)</string>

    <key>CFBundleDisplayName</key>
    <string>MyApp</string>
    <!-- ホーム画面に表示される名前 -->

    <key>CFBundleIdentifier</key>
    <string>$(PRODUCT_BUNDLE_IDENTIFIER)</string>

    <key>CFBundleVersion</key>
    <string>$(CURRENT_PROJECT_VERSION)</string>
    <!-- ビルド番号（自動インクリメント推奨） -->

    <key>CFBundleShortVersionString</key>
    <string>$(MARKETING_VERSION)</string>
    <!-- ユーザー向けバージョン: 1.0.0 -->

    <!-- 対応デバイス・OS -->
    <key>LSRequiresIPhoneOS</key>
    <true/>

    <key>UIRequiredDeviceCapabilities</key>
    <array>
        <string>armv7</string>
    </array>

    <key>MinimumOSVersion</key>
    <string>15.0</string>

    <!-- UI Configuration -->
    <key>UILaunchStoryboardName</key>
    <string>LaunchScreen</string>

    <key>UIApplicationSceneManifest</key>
    <dict>
        <key>UIApplicationSupportsMultipleScenes</key>
        <false/>
        <key>UISceneConfigurations</key>
        <dict>
            <key>UIWindowSceneSessionRoleApplication</key>
            <array>
                <dict>
                    <key>UISceneConfigurationName</key>
                    <string>Default Configuration</string>
                    <key>UISceneDelegateClassName</key>
                    <string>$(PRODUCT_MODULE_NAME).SceneDelegate</string>
                </dict>
            </array>
        </dict>
    </dict>

    <!-- Device Orientation -->
    <key>UISupportedInterfaceOrientations</key>
    <array>
        <string>UIInterfaceOrientationPortrait</string>
        <string>UIInterfaceOrientationLandscapeLeft</string>
        <string>UIInterfaceOrientationLandscapeRight</string>
    </array>

    <key>UISupportedInterfaceOrientations~ipad</key>
    <array>
        <string>UIInterfaceOrientationPortrait</string>
        <string>UIInterfaceOrientationPortraitUpsideDown</string>
        <string>UIInterfaceOrientationLandscapeLeft</string>
        <string>UIInterfaceOrientationLandscapeRight</string>
    </array>

    <!-- Status Bar -->
    <key>UIStatusBarStyle</key>
    <string>UIStatusBarStyleDefault</string>

    <key>UIViewControllerBasedStatusBarAppearance</key>
    <true/>
    <!-- ViewController ごとに Status Bar を制御 -->

    <!-- App Transport Security -->
    <key>NSAppTransportSecurity</key>
    <dict>
        <key>NSAllowsArbitraryLoads</key>
        <false/>
        <!-- 本番では false 推奨 -->

        <key>NSExceptionDomains</key>
        <dict>
            <key>example.com</key>
            <dict>
                <key>NSExceptionAllowsInsecureHTTPLoads</key>
                <true/>
                <key>NSIncludesSubdomains</key>
                <true/>
            </dict>
        </dict>
    </dict>
</dict>
</plist>
```

### Privacy 関連の Usage Description

```xml
<!-- Privacy Permissions -->

<!--
App Store 審査で必須:
使用する機能に応じて、必ず Usage Description を記載
記載がない場合、リジェクトされる可能性大
-->

<!-- カメラ -->
<key>NSCameraUsageDescription</key>
<string>プロフィール写真を撮影するためにカメラへのアクセスが必要です</string>

<!-- フォトライブラリ -->
<key>NSPhotoLibraryUsageDescription</key>
<string>写真を選択してアップロードするためにフォトライブラリへのアクセスが必要です</string>

<key>NSPhotoLibraryAddUsageDescription</key>
<string>画像を保存するためにフォトライブラリへのアクセスが必要です</string>

<!-- 位置情報 -->
<key>NSLocationWhenInUseUsageDescription</key>
<string>近くの店舗を検索するために位置情報へのアクセスが必要です</string>

<key>NSLocationAlwaysAndWhenInUseUsageDescription</key>
<string>バックグラウンドでも位置情報を追跡するためにアクセスが必要です</string>

<key>NSLocationAlwaysUsageDescription</key>
<string>常に位置情報にアクセスします</string>

<!-- マイク -->
<key>NSMicrophoneUsageDescription</key>
<string>音声メッセージを録音するためにマイクへのアクセスが必要です</string>

<!-- 連絡先 -->
<key>NSContactsUsageDescription</key>
<string>友達を招待するために連絡先へのアクセスが必要です</string>

<!-- カレンダー -->
<key>NSCalendarsUsageDescription</key>
<string>イベントをカレンダーに追加するためにアクセスが必要です</string>

<key>NSRemindersUsageDescription</key>
<string>リマインダーを作成するためにアクセスが必要です</string>

<!-- Bluetooth -->
<key>NSBluetoothAlwaysUsageDescription</key>
<string>周辺デバイスと接続するためにBluetoothへのアクセスが必要です</string>

<key>NSBluetoothPeripheralUsageDescription</key>
<string>Bluetoothデバイスと通信します</string>

<!-- Motion -->
<key>NSMotionUsageDescription</key>
<string>歩数を計測するためにモーションセンサーへのアクセスが必要です</string>

<!-- Health -->
<key>NSHealthShareUsageDescription</key>
<string>健康データを読み取るためにHealthKitへのアクセスが必要です</string>

<key>NSHealthUpdateUsageDescription</key>
<string>健康データを記録するためにHealthKitへのアクセスが必要です</string>

<!-- Face ID -->
<key>NSFaceIDUsageDescription</key>
<string>安全にログインするためにFace IDを使用します</string>

<!-- Speech Recognition -->
<key>NSSpeechRecognitionUsageDescription</key>
<string>音声をテキストに変換するために音声認識を使用します</string>

<!-- Media Library -->
<key>NSAppleMusicUsageDescription</key>
<string>Apple Musicライブラリにアクセスします</string>

<!-- Siri -->
<key>NSSiriUsageDescription</key>
<string>Siriショートカットを作成します</string>

<!-- Tracking (iOS 14.5+) -->
<key>NSUserTrackingUsageDescription</key>
<string>パーソナライズされた広告を表示するためにトラッキングを使用します</string>
```

### Background Modes

```xml
<!-- Background Modes -->

<key>UIBackgroundModes</key>
<array>
    <!-- オーディオ再生 -->
    <string>audio</string>

    <!-- 位置情報更新 -->
    <string>location</string>

    <!-- VoIP -->
    <string>voip</string>

    <!-- 外部アクセサリ通信 -->
    <string>external-accessory</string>

    <!-- Bluetooth アクセサリ -->
    <string>bluetooth-central</string>
    <string>bluetooth-peripheral</string>

    <!-- Background Fetch -->
    <string>fetch</string>

    <!-- Remote Notifications -->
    <string>remote-notification</string>

    <!-- Processing -->
    <string>processing</string>
</array>
```

### カスタム URL Scheme と Universal Links

```xml
<!-- URL Schemes -->

<key>CFBundleURLTypes</key>
<array>
    <dict>
        <key>CFBundleTypeRole</key>
        <string>Editor</string>
        <key>CFBundleURLName</key>
        <string>com.company.myapp</string>
        <key>CFBundleURLSchemes</key>
        <array>
            <string>myapp</string>
        </array>
    </dict>
</array>

<!-- 使用例: myapp://profile/123 -->

<!-- Universal Links -->

<key>com.apple.developer.associated-domains</key>
<array>
    <string>applinks:www.example.com</string>
    <string>applinks:staging.example.com</string>
</array>

<!--
apple-app-site-association ファイルをサーバーに配置:
https://www.example.com/.well-known/apple-app-site-association
-->
```

---

## Code Signing設定

### Code Signing の概要

```swift
// Code Signing とは

/*
目的:
1. アプリの発行元を証明
2. アプリの改ざんを防止
3. Apple のセキュリティポリシーに準拠

必要な要素:
1. Certificate (証明書)
   - Development: 開発用
   - Distribution: 配布用

2. App ID (アプリケーション識別子)
   - Bundle Identifier と紐付け
   - Explicit: com.company.myapp
   - Wildcard: com.company.*

3. Provisioning Profile (プロビジョニングプロファイル)
   - Certificate + App ID + Devices の組み合わせ
   - Development: 開発・テスト用
   - Ad Hoc: 限定配布用
   - App Store: App Store 配布用

4. Entitlements (資格)
   - アプリの機能権限
   - Push Notifications, iCloud, etc.
*/
```

### 自動 Code Signing（推奨: 開発初期）

```swift
// Xcode による自動管理

/*
設定手順:
1. TARGETS > Signing & Capabilities
2. Automatically manage signing: ✅
3. Team: Apple Developer アカウントを選択
4. Bundle Identifier: com.company.myapp

Xcode が自動的に:
- Development Certificate を生成
- Provisioning Profile を生成・更新
- Entitlements を設定

メリット:
- 設定が簡単
- 証明書の更新が自動
- 初心者に優しい

デメリット:
- CI/CD での制御が困難
- 複数人開発での同期が煩雑
- チーム全体での統一が難しい
*/
```

### 手動 Code Signing（推奨: 本格開発）

```bash
# 手動 Code Signing の手順

## Step 1: Certificate の作成

1. Keychain Access を起動
2. Certificate Assistant > Request a Certificate from a Certificate Authority
3. Email: あなたのメールアドレス
4. Common Name: Your Name
5. Save to disk

6. Apple Developer Portal にログイン
   https://developer.apple.com/account/

7. Certificates, IDs & Profiles > Certificates
8. "+" ボタン > iOS App Development
9. CSR ファイルをアップロード
10. Download > ダブルクリックでインストール

## Step 2: App ID の登録

1. Identifiers > "+"
2. App IDs > Continue
3. Description: MyApp
4. Bundle ID: com.company.myapp (Explicit)
5. Capabilities: Push Notifications, Sign in with Apple, etc.
6. Register

## Step 3: Devices の登録（Development のみ）

1. Devices > "+"
2. Device Name: iPhone 15 Pro
3. UDID: デバイスの UDID
   → Xcode > Window > Devices and Simulators
4. Register

## Step 4: Provisioning Profile の作成

1. Profiles > "+"
2. Development > iOS App Development
3. App ID: com.company.myapp を選択
4. Certificate: 先ほど作成した証明書を選択
5. Devices: テストデバイスを選択
6. Profile Name: MyApp Development
7. Download > ダブルクリックでインストール

## Step 5: Xcode での設定

1. TARGETS > Signing & Capabilities
2. Automatically manage signing: ❌
3. Provisioning Profile: MyApp Development を選択
4. Signing Certificate: iOS Developer を選択
```

### Fastlane Match による管理（推奨: チーム開発）

```ruby
# Fastlane Match とは

# 証明書とプロファイルを Git リポジトリで一元管理
# チーム全体で同じ証明書を共有
# CI/CD での自動化が容易

# Gemfile
source "https://rubygems.org"

gem "fastlane"

# Terminal
bundle install
fastlane init

# fastlane/Matchfile

git_url("https://github.com/company/certificates.git")
storage_mode("git")

type("development") # development, adhoc, appstore, enterprise

app_identifier(["com.company.myapp"])
username("your@email.com")
team_id("ABCD123456")

# 証明書の生成・同期
fastlane match development
fastlane match appstore

# Xcode での設定
# TARGETS > Signing & Capabilities
# Provisioning Profile: match Development com.company.myapp

# CI/CD での使用
# .github/workflows/ci.yml

- name: Setup Certificates
  env:
    MATCH_PASSWORD: ${{ secrets.MATCH_PASSWORD }}
    MATCH_GIT_BASIC_AUTHORIZATION: ${{ secrets.MATCH_GIT_BASIC_AUTHORIZATION }}
  run: |
    fastlane match development --readonly

# メリット:
# - チーム全体で同じ証明書を共有
# - 新メンバーの追加が簡単
# - CI/CD での自動化
# - 証明書の更新を一元管理

# デメリット:
# - 初期設定がやや複雑
# - Git リポジトリの管理が必要
# - パスワード管理が必要
```

---

(続きは文字数制限のため、次のセクションで)
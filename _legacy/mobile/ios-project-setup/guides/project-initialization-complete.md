# Project Initialization 完全ガイド

## 目次
1. [プロジェクト初期化の基礎](#プロジェクト初期化の基礎)
2. [Xcodeプロジェクト構成](#xcodeプロジェクト構成)
3. [フォルダ構造](#フォルダ構造)
4. [依存関係管理](#依存関係管理)
5. [Gitセットアップ](#gitセットアップ)
6. [CI/CD初期設定](#cicd初期設定)
7. [ドキュメンテーション](#ドキュメンテーション)
8. [チーム開発環境](#チーム開発環境)

---

## プロジェクト初期化の基礎

### プロジェクト作成チェックリスト

```bash
# 新規iOSプロジェクト作成チェックリスト

## 1. Xcodeプロジェクト作成
- [ ] プロジェクト名の決定
- [ ] Bundle Identifierの設定
- [ ] Deployment Targetの設定
- [ ] Language: Swift
- [ ] UI Framework: SwiftUI / UIKit
- [ ] Test Targetsの作成
- [ ] Core Dataの有無

## 2. フォルダ構造
- [ ] ソースコードの整理
- [ ] リソースフォルダの作成
- [ ] テストフォルダの構成

## 3. 依存関係管理
- [ ] Swift Package Manager設定
- [ ] 必要なライブラリの追加

## 4. Git設定
- [ ] .gitignoreの作成
- [ ] 初期コミット
- [ ] リモートリポジトリの設定

## 5. 開発環境
- [ ] SwiftLint設定
- [ ] SwiftFormat設定
- [ ] Pre-commit hookの設定

## 6. ドキュメント
- [ ] README.md作成
- [ ] CONTRIBUTING.md作成
- [ ] CODE_OF_CONDUCT.md作成
```

### プロジェクト構成の例

```
MyApp/
├── MyApp/                          # Main Target
│   ├── App/
│   │   ├── AppDelegate.swift
│   │   ├── SceneDelegate.swift
│   │   └── MyApp.swift            # SwiftUI App
│   ├── Features/
│   │   ├── Home/
│   │   │   ├── Views/
│   │   │   ├── ViewModels/
│   │   │   └── Models/
│   │   ├── Profile/
│   │   └── Settings/
│   ├── Core/
│   │   ├── Networking/
│   │   ├── Database/
│   │   ├── Services/
│   │   └── Extensions/
│   ├── Resources/
│   │   ├── Assets.xcassets
│   │   ├── Colors.xcassets
│   │   ├── Localizable.strings
│   │   └── Fonts/
│   └── Supporting Files/
│       ├── Info.plist
│       └── Configuration/
│           ├── Development.xcconfig
│           ├── Staging.xcconfig
│           └── Production.xcconfig
├── MyAppTests/                     # Unit Tests
│   ├── Features/
│   ├── Core/
│   └── Mocks/
├── MyAppUITests/                   # UI Tests
│   ├── Screenshots/
│   └── Tests/
├── MyAppKit/                       # Shared Framework (Optional)
│   └── Sources/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .swiftlint.yml
├── .swiftformat
├── .gitignore
├── Package.swift                   # SPM Dependencies
└── README.md
```

---

## Xcodeプロジェクト構成

### プロジェクト設定のベストプラクティス

```swift
// Build Settings Configuration

/*
Recommended Settings:

General:
- Display Name: MyApp
- Bundle Identifier: com.company.myapp
- Version: 1.0.0
- Build: 1
- Deployment Target: iOS 15.0+

Build Settings:
- Swift Language Version: Swift 5.9
- Optimization Level (Debug): None [-Onone]
- Optimization Level (Release): Optimize for Speed [-O]
- Compilation Mode (Debug): Incremental
- Compilation Mode (Release): Whole Module
- Enable Bitcode: No
- Strip Debug Symbols During Copy: Yes (Release)
- Dead Code Stripping: Yes
- Warnings:
  - Treat Warnings as Errors: Yes
  - Suspicious Implicit Conversions: Yes
*/

// .xcconfig ファイルの使用
// Development.xcconfig
/*
APP_NAME = MyApp Dev
BUNDLE_ID_SUFFIX = .dev
API_BASE_URL = https://dev.api.example.com
*/

// Production.xcconfig
/*
APP_NAME = MyApp
BUNDLE_ID_SUFFIX =
API_BASE_URL = https://api.example.com
*/

// Info.plist での環境変数使用
/*
<key>CFBundleDisplayName</key>
<string>$(APP_NAME)</string>

<key>CFBundleIdentifier</key>
<string>$(PRODUCT_BUNDLE_IDENTIFIER)$(BUNDLE_ID_SUFFIX)</string>

<key>API_BASE_URL</key>
<string>$(API_BASE_URL)</string>
*/

// Swiftコードで環境変数を取得
enum Environment {
    static var apiBaseURL: String {
        Bundle.main.infoDictionary?["API_BASE_URL"] as? String ?? ""
    }

    static var appName: String {
        Bundle.main.infoDictionary?["CFBundleDisplayName"] as? String ?? ""
    }

    static var bundleIdentifier: String {
        Bundle.main.bundleIdentifier ?? ""
    }
}
```

### Scheme設定

```bash
# Schemes Configuration

## Development Scheme
- Build Configuration: Debug
- Run: Development.xcconfig
- Test: Debug
- Profile: Debug
- Analyze: Debug
- Archive: Not applicable

## Staging Scheme
- Build Configuration: Release
- Run: Staging.xcconfig
- Test: Release
- Profile: Release
- Analyze: Release
- Archive: Staging.xcconfig

## Production Scheme
- Build Configuration: Release
- Run: Production.xcconfig
- Test: Release
- Profile: Release
- Analyze: Release
- Archive: Production.xcconfig
```

---

## フォルダ構造

### Clean Architectureに基づくフォルダ構成

```swift
// Features モジュール構成

/*
Features/
├── Home/
│   ├── Domain/              # ビジネスロジック層
│   │   ├── Entities/
│   │   │   └── User.swift
│   │   ├── UseCases/
│   │   │   ├── FetchUsersUseCase.swift
│   │   │   └── UpdateUserUseCase.swift
│   │   └── Repositories/
│   │       └── UserRepositoryProtocol.swift
│   ├── Data/                # データアクセス層
│   │   ├── Repositories/
│   │   │   └── UserRepository.swift
│   │   ├── DataSources/
│   │   │   ├── Remote/
│   │   │   │   └── UserRemoteDataSource.swift
│   │   │   └── Local/
│   │   │       └── UserLocalDataSource.swift
│   │   └── DTOs/
│   │       └── UserDTO.swift
│   └── Presentation/        # UI層
│       ├── Views/
│       │   ├── HomeView.swift
│       │   └── UserRow.swift
│       ├── ViewModels/
│       │   └── HomeViewModel.swift
│       └── Coordinators/
│           └── HomeCoordinator.swift
*/

// Core モジュール構成

/*
Core/
├── Networking/
│   ├── HTTPClient.swift
│   ├── APIClient.swift
│   ├── RequestBuilder.swift
│   └── ResponseHandler.swift
├── Database/
│   ├── CoreDataStack.swift
│   ├── DatabaseManager.swift
│   └── Migrations/
├── Services/
│   ├── AuthenticationService.swift
│   ├── AnalyticsService.swift
│   └── PushNotificationService.swift
├── Extensions/
│   ├── String+Extensions.swift
│   ├── Date+Extensions.swift
│   └── View+Extensions.swift
└── Utils/
    ├── Logger.swift
    ├── Validator.swift
    └── Constants.swift
*/
```

### ファイル命名規則

```swift
// ファイル命名規則

/*
Views:
- SwiftUI: ContentView.swift, UserDetailView.swift
- UIKit: HomeViewController.swift, UserDetailViewController.swift

ViewModels:
- HomeViewModel.swift, UserDetailViewModel.swift

Models/Entities:
- User.swift, Post.swift, Comment.swift

Services:
- AuthenticationService.swift, NetworkService.swift

Protocols:
- UserRepositoryProtocol.swift, DataSourceProtocol.swift

Extensions:
- String+Extensions.swift, UIView+Extensions.swift

Tests:
- UserViewModelTests.swift, AuthServiceTests.swift

UI Tests:
- LoginFlowUITests.swift, CheckoutFlowUITests.swift
*/
```

---

## 依存関係管理

### Swift Package Manager (SPM)

```swift
// Package.swift

import PackageDescription

let package = Package(
    name: "MyApp",
    platforms: [
        .iOS(.v15)
    ],
    products: [
        .library(
            name: "MyAppKit",
            targets: ["MyAppKit"]
        )
    ],
    dependencies: [
        // Networking
        .package(url: "https://github.com/Alamofire/Alamofire.git", from: "5.8.0"),

        // UI
        .package(url: "https://github.com/kean/Nuke.git", from: "12.0.0"),
        .package(url: "https://github.com/onevcat/Kingfisher.git", from: "7.10.0"),

        // Reactive
        .package(url: "https://github.com/CombineCommunity/CombineExt.git", from: "1.8.0"),

        // Analytics
        .package(url: "https://github.com/firebase/firebase-ios-sdk", from: "10.20.0"),

        // Development Tools
        .package(url: "https://github.com/realm/SwiftLint", from: "0.54.0"),
        .package(url: "https://github.com/nicklockwood/SwiftFormat", from: "0.52.0")
    ],
    targets: [
        .target(
            name: "MyAppKit",
            dependencies: [
                "Alamofire",
                .product(name: "Nuke", package: "Nuke"),
                .product(name: "FirebaseAnalytics", package: "firebase-ios-sdk")
            ]
        ),
        .testTarget(
            name: "MyAppKitTests",
            dependencies: ["MyAppKit"]
        )
    ]
)
```

### パッケージ管理のベストプラクティス

```swift
// DependencyContainer.swift

protocol HasNetworking {
    var httpClient: HTTPClient { get }
    var apiClient: APIClient { get }
}

protocol HasDatabase {
    var coreDataStack: CoreDataStack { get }
}

protocol HasServices {
    var authService: AuthenticationService { get }
    var analyticsService: AnalyticsService { get }
}

class DependencyContainer: HasNetworking, HasDatabase, HasServices {
    // Networking
    lazy var httpClient: HTTPClient = {
        URLSessionHTTPClient()
    }()

    lazy var apiClient: APIClient = {
        APIClient(httpClient: httpClient)
    }()

    // Database
    lazy var coreDataStack: CoreDataStack = {
        CoreDataStack.shared
    }()

    // Services
    lazy var authService: AuthenticationService = {
        AuthenticationService(
            apiClient: apiClient,
            keychain: KeychainService()
        )
    }()

    lazy var analyticsService: AnalyticsService = {
        FirebaseAnalyticsService()
    }()

    // Repositories
    func makeUserRepository() -> UserRepository {
        UserRepository(
            remoteDataSource: UserRemoteDataSource(apiClient: apiClient),
            localDataSource: UserLocalDataSource(context: coreDataStack.context)
        )
    }

    // ViewModels
    func makeHomeViewModel() -> HomeViewModel {
        HomeViewModel(
            fetchUsersUseCase: FetchUsersUseCase(
                userRepository: makeUserRepository()
            )
        )
    }
}

// App起動時の設定
@main
struct MyApp: App {
    private let container = DependencyContainer()

    var body: some Scene {
        WindowGroup {
            HomeView(viewModel: container.makeHomeViewModel())
        }
    }
}
```

---

## Gitセットアップ

### .gitignore設定

```bash
# .gitignore

# Xcode
*.xcodeproj/*
!*.xcodeproj/project.pbxproj
!*.xcodeproj/xcshareddata/
!*.xcworkspace/contents.xcworkspacedata
**/xcshareddata/WorkspaceSettings.xcsettings

# Build
build/
DerivedData/
*.ipa
*.dSYM.zip
*.dSYM

# Swift Package Manager
.swiftpm/
Packages/
Package.resolved
*.xcworkspace

# CocoaPods
Pods/
*.podspec

# Carthage
Carthage/Build/

# fastlane
fastlane/report.xml
fastlane/Preview.html
fastlane/screenshots/**/*.png
fastlane/test_output

# Code coverage
*.gcov
*.gcda
*.gcno
coverage/

# Environment
.env
.env.local
secrets.plist
GoogleService-Info.plist

# macOS
.DS_Store
.AppleDouble
.LSOverride

# IDEs
.vscode/
.idea/

# Temporary
*.swp
*.swo
*~.nib
*.moved-aside
```

### Git Hook設定

```bash
#!/bin/bash
# .git/hooks/pre-commit

# SwiftLint
if which swiftlint >/dev/null; then
    swiftlint --strict
    if [ $? -ne 0 ]; then
        echo "❌ SwiftLint failed. Please fix the issues before committing."
        exit 1
    fi
    echo "✅ SwiftLint passed"
else
    echo "⚠️  SwiftLint not installed. Run: brew install swiftlint"
fi

# SwiftFormat
if which swiftformat >/dev/null; then
    swiftformat --lint .
    if [ $? -ne 0 ]; then
        echo "❌ SwiftFormat found issues. Run: swiftformat ."
        exit 1
    fi
    echo "✅ SwiftFormat passed"
else
    echo "⚠️  SwiftFormat not installed. Run: brew install swiftformat"
fi

# Tests
echo "Running tests..."
xcodebuild test \
    -scheme MyApp \
    -destination 'platform=iOS Simulator,name=iPhone 15 Pro' \
    -quiet

if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Please fix the failing tests before committing."
    exit 1
fi

echo "✅ All tests passed"
echo "✅ Pre-commit checks completed successfully!"
```

### Gitフロー

```bash
# Git Branch Strategy

# Main branches
main          # Production-ready code
develop       # Integration branch

# Supporting branches
feature/*     # New features
bugfix/*      # Bug fixes
hotfix/*      # Production hotfixes
release/*     # Release preparation

# Example workflow

# 1. Feature開発
git checkout develop
git checkout -b feature/user-profile
# ... work ...
git add .
git commit -m "feat: add user profile screen"
git push origin feature/user-profile
# Create PR to develop

# 2. Release準備
git checkout develop
git checkout -b release/1.0.0
# ... final touches ...
git commit -m "chore: prepare release 1.0.0"
# Create PR to main

# 3. Hotfix
git checkout main
git checkout -b hotfix/critical-bug
# ... fix ...
git commit -m "fix: resolve critical authentication bug"
# Create PR to main AND develop
```

---

## CI/CD初期設定

### GitHub Actions設定

```yaml
# .github/workflows/ci.yml

name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  lint:
    name: SwiftLint
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4

      - name: SwiftLint
        run: |
          brew install swiftlint
          swiftlint --strict

  test:
    name: Unit Tests
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4

      - name: Select Xcode
        run: sudo xcode-select -s /Applications/Xcode_15.2.app

      - name: Cache SPM
        uses: actions/cache@v3
        with:
          path: .build
          key: ${{ runner.os }}-spm-${{ hashFiles('**/Package.resolved') }}

      - name: Build and Test
        run: |
          xcodebuild test \
            -scheme MyApp \
            -destination 'platform=iOS Simulator,name=iPhone 15 Pro' \
            -enableCodeCoverage YES \
            -resultBundlePath TestResults

      - name: Upload Coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./TestResults/Coverage.txt

  build:
    name: Build
    runs-on: macos-14
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4

      - name: Build
        run: |
          xcodebuild build \
            -scheme MyApp \
            -destination 'platform=iOS Simulator,name=iPhone 15 Pro'
```

このガイドでは、iOSプロジェクトの初期化から、フォルダ構成、依存関係管理、Git設定、CI/CD初期設定まで、プロジェクト開始時に必要なすべての要素を網羅しました。適切な初期設定により、開発効率とコード品質を大きく向上させることができます。

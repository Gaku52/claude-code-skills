# Build Configuration 完全ガイド

## 目次
1. [ビルド設定の基礎](#ビルド設定の基礎)
2. [xcconfig活用](#xcconfig活用)
3. [Build Phases最適化](#build-phases最適化)
4. [コード署名](#コード署名)
5. [最適化設定](#最適化設定)
6. [App Thinning](#app-thinning)
7. [デバッグとリリースビルド](#デバッグとリリースビルド)
8. [トラブルシューティング](#トラブルシューティング)

---

## ビルド設定の基礎

### Build Configurationの概要

```swift
/*
Build Configuration階層:

Project Settings
└── Target Settings
    └── Scheme Settings
        └── Build Configuration
            ├── Debug
            ├── Release
            ├── Staging (Custom)
            └── Production (Custom)

設定の優先順位:
1. Target Settings (最優先)
2. Project Settings
3. xcconfig Files
4. Xcode Defaults
*/

// Build Settings カテゴリ

/*
主要な設定カテゴリ:

1. Architectures
   - Build Active Architecture Only
   - Supported Platforms
   - Valid Architectures

2. Build Options
   - Compiler Optimization Level
   - Swift Optimization Level
   - Debug Information Format

3. Deployment
   - iOS Deployment Target
   - Strip Debug Symbols
   - Enable Bitcode

4. Signing
   - Code Signing Identity
   - Development Team
   - Provisioning Profile

5. Swift Compiler
   - Swift Language Version
   - Compilation Mode
   - Active Compilation Conditions
*/
```

### Configuration別設定例

```swift
// Debug Configuration

/*
CONFIGURATION: Debug

Optimization:
- Optimization Level: None [-Onone]
- Swift Optimization: No Optimization [-Onone]
- Compilation Mode: Incremental

Debug:
- Debug Information Format: DWARF with dSYM
- Generate Debug Symbols: Yes
- Strip Debug Symbols: No

Performance:
- Enable Testability: Yes
- Whole Module Optimization: No

Other:
- SWIFT_ACTIVE_COMPILATION_CONDITIONS: DEBUG
- GCC_PREPROCESSOR_DEFINITIONS: DEBUG=1
*/

// Release Configuration

/*
CONFIGURATION: Release

Optimization:
- Optimization Level: Fastest, Smallest [-Os]
- Swift Optimization: Optimize for Speed [-O]
- Compilation Mode: Whole Module

Debug:
- Debug Information Format: DWARF with dSYM
- Generate Debug Symbols: Yes
- Strip Debug Symbols: Yes (Copy Phase)

Performance:
- Enable Testability: No
- Whole Module Optimization: Yes

Other:
- SWIFT_ACTIVE_COMPILATION_CONDITIONS: RELEASE
- DEAD_CODE_STRIPPING: Yes
*/

// コンパイル条件の使用
class Config {
    static var apiBaseURL: String {
        #if DEBUG
        return "https://dev.api.example.com"
        #elseif STAGING
        return "https://staging.api.example.com"
        #else
        return "https://api.example.com"
        #endif
    }

    static var logLevel: LogLevel {
        #if DEBUG
        return .verbose
        #elseif STAGING
        return .info
        #else
        return .error
        #endif
    }

    static var isDebugMode: Bool {
        #if DEBUG
        return true
        #else
        return false
        #endif
    }
}

enum LogLevel {
    case verbose
    case debug
    case info
    case warning
    case error
}
```

---

## xcconfig活用

### xcconfigファイルの作成

```bash
# Config/Development.xcconfig

// App Configuration
APP_NAME = MyApp Dev
BUNDLE_ID_SUFFIX = .dev
APP_VERSION = 1.0.0
BUILD_NUMBER = 1

// Server Configuration
API_BASE_URL = https:/$()/dev.api.example.com
API_KEY = dev_api_key_here
WEB_SOCKET_URL = wss:/$()/dev.ws.example.com

// Feature Flags
ENABLE_ANALYTICS = NO
ENABLE_CRASH_REPORTING = NO
ENABLE_DEBUG_MENU = YES

// Build Settings
SWIFT_ACTIVE_COMPILATION_CONDITIONS = DEBUG DEV
GCC_PREPROCESSOR_DEFINITIONS = DEBUG=1 DEV=1

// Code Signing
CODE_SIGN_STYLE = Automatic
DEVELOPMENT_TEAM = ABCD123456
CODE_SIGN_IDENTITY = iPhone Developer

// Deployment
IPHONEOS_DEPLOYMENT_TARGET = 15.0
TARGETED_DEVICE_FAMILY = 1,2

// Optimization
SWIFT_OPTIMIZATION_LEVEL = -Onone
GCC_OPTIMIZATION_LEVEL = 0
SWIFT_COMPILATION_MODE = singlefile
```

```bash
# Config/Staging.xcconfig

// App Configuration
APP_NAME = MyApp Staging
BUNDLE_ID_SUFFIX = .staging
APP_VERSION = 1.0.0
BUILD_NUMBER = 1

// Server Configuration
API_BASE_URL = https:/$()/staging.api.example.com
API_KEY = staging_api_key_here
WEB_SOCKET_URL = wss:/$()/staging.ws.example.com

// Feature Flags
ENABLE_ANALYTICS = YES
ENABLE_CRASH_REPORTING = YES
ENABLE_DEBUG_MENU = YES

// Build Settings
SWIFT_ACTIVE_COMPILATION_CONDITIONS = STAGING
GCC_PREPROCESSOR_DEFINITIONS = STAGING=1

// Code Signing
CODE_SIGN_STYLE = Manual
DEVELOPMENT_TEAM = ABCD123456
CODE_SIGN_IDENTITY = iPhone Distribution
PROVISIONING_PROFILE_SPECIFIER = MyApp Staging

// Deployment
IPHONEOS_DEPLOYMENT_TARGET = 15.0

// Optimization
SWIFT_OPTIMIZATION_LEVEL = -O
GCC_OPTIMIZATION_LEVEL = s
SWIFT_COMPILATION_MODE = wholemodule
```

```bash
# Config/Production.xcconfig

// App Configuration
APP_NAME = MyApp
BUNDLE_ID_SUFFIX =
APP_VERSION = 1.0.0
BUILD_NUMBER = 1

// Server Configuration
API_BASE_URL = https:/$()/api.example.com
API_KEY = prod_api_key_here
WEB_SOCKET_URL = wss:/$()/ws.example.com

// Feature Flags
ENABLE_ANALYTICS = YES
ENABLE_CRASH_REPORTING = YES
ENABLE_DEBUG_MENU = NO

// Build Settings
SWIFT_ACTIVE_COMPILATION_CONDITIONS = RELEASE
GCC_PREPROCESSOR_DEFINITIONS =

// Code Signing
CODE_SIGN_STYLE = Manual
DEVELOPMENT_TEAM = ABCD123456
CODE_SIGN_IDENTITY = iPhone Distribution
PROVISIONING_PROFILE_SPECIFIER = MyApp Production

// Deployment
IPHONEOS_DEPLOYMENT_TARGET = 15.0

// Optimization
SWIFT_OPTIMIZATION_LEVEL = -O
GCC_OPTIMIZATION_LEVEL = s
SWIFT_COMPILATION_MODE = wholemodule
DEAD_CODE_STRIPPING = YES
STRIP_INSTALLED_PRODUCT = YES
```

### xcconfigの継承

```bash
# Config/Base.xcconfig
// 共通設定

IPHONEOS_DEPLOYMENT_TARGET = 15.0
TARGETED_DEVICE_FAMILY = 1,2
SWIFT_VERSION = 5.9

ENABLE_BITCODE = NO
ALWAYS_EMBED_SWIFT_STANDARD_LIBRARIES = YES

CLANG_WARN_QUOTED_INCLUDE_IN_FRAMEWORK_HEADER = YES
CLANG_WARN_DOCUMENTATION_COMMENTS = YES

# Config/Debug.xcconfig
#include "Base.xcconfig"

APP_NAME = MyApp Dev
SWIFT_OPTIMIZATION_LEVEL = -Onone
ENABLE_TESTABILITY = YES

# Config/Release.xcconfig
#include "Base.xcconfig"

APP_NAME = MyApp
SWIFT_OPTIMIZATION_LEVEL = -O
ENABLE_TESTABILITY = NO
```

---

## Build Phases最適化

### カスタムBuild Phaseの追加

```bash
# Run Script Phase: SwiftLint

# スクリプト名: SwiftLint
# Input Files: (空)
# Output Files: (空)

if which swiftlint >/dev/null; then
    swiftlint
else
    echo "warning: SwiftLint not installed, download from https://github.com/realm/SwiftLint"
fi

# Run Script Phase: SwiftFormat

# スクリプト名: SwiftFormat
if which swiftformat >/dev/null; then
    swiftformat --lint "$PROJECT_DIR"
else
    echo "warning: SwiftFormat not installed"
fi

# Run Script Phase: Increment Build Number

# スクリプト名: Increment Build Number (Archive Only)
# Based on configuration: Release

if [ "$CONFIGURATION" = "Release" ]; then
    buildNumber=$(/usr/libexec/PlistBuddy -c "Print CFBundleVersion" "${INFOPLIST_FILE}")
    buildNumber=$(($buildNumber + 1))
    /usr/libexec/PlistBuddy -c "Set :CFBundleVersion $buildNumber" "${INFOPLIST_FILE}"
    echo "Build number incremented to: $buildNumber"
fi

# Run Script Phase: Copy Google Service Info

# スクリプト名: Copy Google Service Info
# Based on configuration

case "${CONFIGURATION}" in
    "Debug" )
        cp -r "${PROJECT_DIR}/Config/GoogleService-Info-Dev.plist" "${BUILT_PRODUCTS_DIR}/${PRODUCT_NAME}.app/GoogleService-Info.plist"
        ;;
    "Staging" )
        cp -r "${PROJECT_DIR}/Config/GoogleService-Info-Staging.plist" "${BUILT_PRODUCTS_DIR}/${PRODUCT_NAME}.app/GoogleService-Info.plist"
        ;;
    "Release" )
        cp -r "${PROJECT_DIR}/Config/GoogleService-Info-Prod.plist" "${BUILT_PRODUCTS_DIR}/${PRODUCT_NAME}.app/GoogleService-Info.plist"
        ;;
esac

# Run Script Phase: Generate Build Info

# スクリプト名: Generate Build Info

BUILD_INFO_PLIST="${BUILT_PRODUCTS_DIR}/${PRODUCT_NAME}.app/BuildInfo.plist"

/usr/libexec/PlistBuddy -c "Add :BuildDate string '$(date)'" "$BUILD_INFO_PLIST"
/usr/libexec/PlistBuddy -c "Add :GitCommit string '$(git rev-parse HEAD)'" "$BUILD_INFO_PLIST"
/usr/libexec/PlistBuddy -c "Add :GitBranch string '$(git rev-parse --abbrev-ref HEAD)'" "$BUILD_INFO_PLIST"
/usr/libexec/PlistBuddy -c "Add :Configuration string '${CONFIGURATION}'" "$BUILD_INFO_PLIST"

echo "Build info generated"
```

### Copy Files Phaseの最適化

```bash
# Embed Frameworks
- Destination: Frameworks
- Code Sign On Copy: Yes
- Items:
  - MyFramework.framework
  - ThirdPartySDK.framework

# Copy Resources
- Destination: Resources
- Items:
  - Models.momd
  - Assets.car

# Strip Debug Symbols (Release Only)

# スクリプト名: Strip Debug Symbols
if [ "${CONFIGURATION}" = "Release" ]; then
    APP_PATH="${TARGET_BUILD_DIR}/${WRAPPER_NAME}"

    # Strip debug symbols from frameworks
    find "$APP_PATH" -name '*.framework' | while read framework; do
        find "$framework" -type f -name '*' | while read file; do
            if file "$file" | grep -q "Mach-O"; then
                echo "Stripping $file"
                strip -x "$file"
            fi
        done
    done
fi
```

---

## コード署名

### 自動コード署名

```swift
// Project.pbxproj の設定

/*
Automatically manage signing:

TargetAttributes = {
    ABC123DEF456 = {
        DevelopmentTeam = ABCD123456;
        ProvisioningStyle = Automatic;
    };
};

Build Settings:
CODE_SIGN_STYLE = Automatic;
DEVELOPMENT_TEAM = ABCD123456;
CODE_SIGN_IDENTITY = "Apple Development";
*/

// 手動コード署名

/*
TargetAttributes = {
    ABC123DEF456 = {
        DevelopmentTeam = ABCD123456;
        ProvisioningStyle = Manual;
    };
};

Build Settings:
CODE_SIGN_STYLE = Manual;
DEVELOPMENT_TEAM = ABCD123456;
CODE_SIGN_IDENTITY = "iPhone Distribution";
PROVISIONING_PROFILE_SPECIFIER = "MyApp Production";
*/
```

### Fastlane Matchの使用

```ruby
# Fastfile

default_platform(:ios)

platform :ios do
  before_all do
    setup_ci if ENV['CI']
  end

  desc "Sync code signing"
  lane :sync_signing do |options|
    type = options[:type] || "development"

    match(
      type: type,
      app_identifier: ["com.company.myapp", "com.company.myapp.dev"],
      readonly: is_ci,
      git_url: "https://github.com/company/certificates.git"
    )
  end

  desc "Build app"
  lane :build do |options|
    configuration = options[:configuration] || "Debug"

    sync_signing(type: configuration == "Release" ? "appstore" : "development")

    build_app(
      scheme: "MyApp",
      configuration: configuration,
      export_method: configuration == "Release" ? "app-store" : "development",
      output_directory: "./build",
      output_name: "MyApp.ipa"
    )
  end

  desc "Upload to TestFlight"
  lane :beta do
    build(configuration: "Release")

    upload_to_testflight(
      skip_waiting_for_build_processing: true,
      changelog: git_log
    )
  end
end

def git_log
  changelog_from_git_commits(
    between: [last_git_tag, "HEAD"],
    pretty: "- %s",
    merge_commit_filtering: "exclude_merges"
  )
end
```

---

## 最適化設定

### コンパイラ最適化

```swift
// Optimization Levels

/*
Debug Build:
- Swift Optimization: -Onone (最適化なし)
  - ビルド時間: 最速
  - デバッグ: 容易
  - パフォーマンス: 低

- GCC Optimization: -O0 (最適化なし)

Release Build:
- Swift Optimization: -O (速度優先)
  - ビルド時間: 遅い
  - デバッグ: 困難
  - パフォーマンス: 高

- GCC Optimization: -Os (サイズ優先)
  - アプリサイズを最小化
  - 適度なパフォーマンス

Alternative Release:
- Swift Optimization: -Osize (サイズ優先)
  - アプリサイズを最優先
  - パフォーマンスとのトレードオフ
*/

// Link-Time Optimization (LTO)

/*
LLVM_LTO = YES

効果:
- バイナリサイズの削減 (5-10%)
- 実行速度の向上 (0-15%)
- ビルド時間の増加

推奨:
- Release buildでのみ有効化
*/

// Whole Module Optimization

/*
SWIFT_COMPILATION_MODE = wholemodule

効果:
- コンパイラがモジュール全体を最適化
- 高速なコード生成
- インライン展開の改善

推奨:
- Release buildで必須
- Debug buildでは無効（ビルド時間削減）
*/
```

### ビルド時間の最適化

```swift
// ビルド時間計測

// Build Settingsに追加:
OTHER_SWIFT_FLAGS = -Xfrontend -debug-time-function-bodies -Xfrontend -debug-time-compilation

// ビルドログから遅い関数を特定:
// Build > Show > Report Navigator > Build Log

/*
計測結果の例:

0.5ms  MyClass.complexFunction()
12.3ms MyViewModel.expensiveComputation()
45.2ms MyView.body.getter (SwiftUI)

対策:
- 型推論を明示的に
- 複雑な式を分割
- @inline属性の使用
*/

// 型推論の最適化

// ❌ 遅い: 型推論が複雑
let result = data
    .filter { $0.isActive }
    .map { $0.name }
    .reduce("") { $0 + ", " + $1 }

// ✅ 速い: 明示的な型指定
let filteredData: [User] = data.filter { $0.isActive }
let names: [String] = filteredData.map { $0.name }
let result: String = names.reduce("") { $0 + ", " + $1 }

// Incremental Build の活用

/*
SWIFT_COMPILATION_MODE = singlefile (Debug)

効果:
- ファイル単位のコンパイル
- 変更ファイルのみ再コンパイル
- ビルド時間の大幅短縮
*/
```

---

## App Thinning

### App Thinningの設定

```swift
/*
App Thinning技術:

1. App Slicing
   - デバイスごとに最適化されたバイナリ
   - 自動的に適用
   - 設定不要

2. Bitcode
   - ENABLE_BITCODE = NO (iOS推奨)
   - Appleがサーバー側で最適化
   - 現在は非推奨

3. On-Demand Resources
   - ENABLE_ON_DEMAND_RESOURCES = YES
   - 初回起動時に不要なリソースを除外
*/

// On-Demand Resources の実装

// 1. Xcodeでリソースをタグ付け
// Assets.xcassets > Resource Tags

// 2. コードから要求
class ResourceManager {
    func loadLevel(_ level: Int) async throws {
        let tag = "level\(level)"

        let request = NSBundleResourceRequest(tags: [tag])

        // ダウンロード開始
        try await request.beginAccessingResources()

        // リソースの使用
        // ...

        // 使用終了を通知（システムが必要に応じて削除）
        request.endAccessingResources()
    }
}

// Asset Catalog の最適化

/*
Compression:
- Assets.xcassets > Inspector
- Compression: Lossy / Lossless

Image Sets:
- @1x, @2x, @3x の適切な提供
- 不要な解像度は除外

Vector Assets:
- PDFやSVGを使用
- Preserve Vector Data: Yes
- Single Scale適用
*/
```

---

## デバッグとリリースビルド

### デバッグビルドの設定

```swift
// Debug Configuration

/*
開発効率を最優先:

Optimization:
SWIFT_OPTIMIZATION_LEVEL = -Onone
GCC_OPTIMIZATION_LEVEL = 0
SWIFT_COMPILATION_MODE = singlefile

Debug:
DEBUG_INFORMATION_FORMAT = dwarf
ENABLE_TESTABILITY = YES
GCC_PREPROCESSOR_DEFINITIONS = DEBUG=1

Features:
ENABLE_BITCODE = NO
VALIDATE_PRODUCT = NO
*/

// デバッグ専用コード

#if DEBUG
class DebugManager {
    static func showDebugMenu(in window: UIWindow) {
        let debugVC = DebugViewController()
        debugVC.modalPresentationStyle = .fullScreen
        window.rootViewController?.present(debugVC, animated: true)
    }

    static func logNetworkRequest(_ request: URLRequest) {
        print("📡 Request: \(request.url?.absoluteString ?? "")")
        print("Headers: \(request.allHTTPHeaderFields ?? [:])")
    }

    static func injectMockData() {
        UserDefaults.standard.set(true, forKey: "useMockData")
    }
}
#endif

// アプリ起動時
#if DEBUG
if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
   let window = windowScene.windows.first {
    // デバッグメニューをシェイクで表示
    NotificationCenter.default.addObserver(
        forName: UIDevice.deviceDidShakeNotification,
        object: nil,
        queue: .main
    ) { _ in
        DebugManager.showDebugMenu(in: window)
    }
}
#endif
```

### リリースビルドの設定

```swift
// Release Configuration

/*
パフォーマンスとサイズを最優先:

Optimization:
SWIFT_OPTIMIZATION_LEVEL = -O
GCC_OPTIMIZATION_LEVEL = s
SWIFT_COMPILATION_MODE = wholemodule
LLVM_LTO = YES_THIN

Size Reduction:
DEAD_CODE_STRIPPING = YES
STRIP_INSTALLED_PRODUCT = YES
COPY_PHASE_STRIP = YES
STRIP_SWIFT_SYMBOLS = YES

Security:
ENABLE_TESTABILITY = NO
VALIDATE_PRODUCT = YES
*/

// リリースビルドチェックリスト

/*
Pre-Release Checklist:

□ バージョン番号の確認
□ ビルド番号のインクリメント
□ デバッグコードの除去
□ ログレベルの調整
□ APIキーの確認
□ 証明書の有効期限確認
□ アプリサイズの確認
□ クラッシュレポートの設定確認
□ アナリティクスの動作確認
□ プライバシーポリシーの更新
*/
```

このガイドでは、iOSビルド設定の基礎から、xcconfig活用、最適化設定、コード署名、App Thinningまで、プロダクションレベルのビルド構成に必要なすべての要素を網羅しました。適切なビルド設定により、開発効率とアプリ品質を大きく向上させることができます。

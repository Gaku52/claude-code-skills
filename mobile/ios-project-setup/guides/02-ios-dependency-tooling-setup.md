# iOS Dependency & Tooling Setup - 完全ガイド

## 目次

1. [依存関係管理の概要](#依存関係管理の概要)
2. [Swift Package Manager (SPM)](#swift-package-manager-spm)
3. [CocoaPods](#cocoapods)
4. [Carthage](#carthage)
5. [依存関係管理の比較と選択](#依存関係管理の比較と選択)
6. [Fastlane セットアップ](#fastlaneセットアップ)
7. [SwiftLint 設定](#swiftlint設定)
8. [SwiftFormat 設定](#swiftformat設定)
9. [Danger.swift](#dangerswift)
10. [Pre-commit Hooks](#pre-commit-hooks)
11. [CI/CD 初期設定](#cicd初期設定)
12. [開発ツール統合](#開発ツール統合)
13. [トラブルシューティング](#トラブルシューティング)

---

## 依存関係管理の概要

### なぜ依存関係管理が重要か

```swift
// 依存関係管理の目的

/*
1. コードの再利用
   - 車輪の再発明を避ける
   - 実証済みのライブラリを活用
   - 開発時間の短縮

2. バージョン管理
   - ライブラリのバージョンを固定
   - チーム全体で同じバージョンを使用
   - アップデート時の影響範囲を把握

3. 依存関係の解決
   - ライブラリ間の依存関係を自動解決
   - バージョンの競合を検出
   - 最適な組み合わせを選択

4. セキュリティ
   - 脆弱性のあるライブラリを検出
   - アップデートの追跡
   - セキュリティパッチの適用

5. チーム開発
   - 全員が同じ環境で開発
   - セットアップの簡素化
   - ビルドの再現性
*/

// ライブラリ選定の基準

/*
評価ポイント:

1. メンテナンス状況
   - 最終更新日（6ヶ月以内が理想）
   - GitHub のスター数（1,000+ が目安）
   - Issues の対応速度
   - コミット頻度

2. ライセンス
   - MIT, Apache 2.0: 商用利用可
   - GPL: 注意が必要（ソースコード公開義務）
   - 企業の利用ポリシーを確認

3. ドキュメント
   - README が充実している
   - API ドキュメントが整備されている
   - サンプルコードがある
   - チュートリアルが提供されている

4. コミュニティ
   - Stack Overflow での質問数
   - GitHub Discussions の活発さ
   - ユーザー事例

5. パフォーマンス
   - バイナリサイズへの影響
   - 実行速度
   - メモリ使用量

6. 依存関係
   - 他のライブラリへの依存が少ない
   - バージョン要件が緩い
   - 競合が起きにくい

7. Swift / iOS サポート
   - 最新の Swift バージョン対応
   - 最新の iOS 対応
   - SwiftUI サポート（必要に応じて）
*/
```

---

## Swift Package Manager (SPM)

### SPM の概要

```swift
// Swift Package Manager とは

/*
Apple 公式のパッケージマネージャー

メリット:
✅ Xcode に統合済み（追加インストール不要）
✅ シンプルな設定（Package.swift のみ）
✅ ビルドが高速
✅ Git ベースで管理が簡単
✅ Swift に特化した設計
✅ Apple のファーストパーティサポート

デメリット:
❌ 一部のライブラリが未対応
❌ バイナリフレームワークのサポートが限定的
❌ リソースファイルの扱いが煩雑（改善中）
❌ Xcode 11+ 必須

推奨度: ⭐⭐⭐⭐⭐
→ 新規プロジェクトでは SPM を第一選択
*/
```

### SPM によるパッケージ追加

```swift
// Xcode GUI からの追加

/*
1. File > Add Packages...
2. 検索バーに URL を入力:
   https://github.com/Alamofire/Alamofire.git
3. Dependency Rule を選択:
   - Up to Next Major Version: 5.0.0 < 6.0.0
   - Up to Next Minor Version: 5.8.0 < 5.9.0
   - Exact Version: 5.8.0
   - Branch: main
   - Commit: abc123def
4. Add Package をクリック
5. Target に追加するライブラリを選択
*/

// Package.swift による管理（推奨）

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
        // MARK: - Networking
        .package(
            url: "https://github.com/Alamofire/Alamofire.git",
            from: "5.8.0"
        ),
        .package(
            url: "https://github.com/Moya/Moya.git",
            from: "15.0.0"
        ),

        // MARK: - UI
        .package(
            url: "https://github.com/onevcat/Kingfisher.git",
            from: "7.10.0"
        ),
        .package(
            url: "https://github.com/kean/Nuke.git",
            from: "12.0.0"
        ),
        .package(
            url: "https://github.com/airbnb/lottie-ios.git",
            from: "4.4.0"
        ),

        // MARK: - Reactive Programming
        .package(
            url: "https://github.com/CombineCommunity/CombineExt.git",
            from: "1.8.0"
        ),
        .package(
            url: "https://github.com/ReactiveX/RxSwift.git",
            from: "6.6.0"
        ),

        // MARK: - Database
        .package(
            url: "https://github.com/realm/realm-swift.git",
            from: "10.45.0"
        ),
        .package(
            url: "https://github.com/stephencelis/SQLite.swift.git",
            from: "0.14.1"
        ),

        // MARK: - Firebase
        .package(
            url: "https://github.com/firebase/firebase-ios-sdk.git",
            from: "10.20.0"
        ),

        // MARK: - Utility
        .package(
            url: "https://github.com/SwiftyJSON/SwiftyJSON.git",
            from: "5.0.1"
        ),
        .package(
            url: "https://github.com/kishikawakatsumi/KeychainAccess.git",
            from: "4.2.2"
        ),

        // MARK: - Development Tools
        .package(
            url: "https://github.com/realm/SwiftLint.git",
            from: "0.54.0"
        ),
        .package(
            url: "https://github.com/nicklockwood/SwiftFormat.git",
            from: "0.52.0"
        )
    ],
    targets: [
        .target(
            name: "MyAppKit",
            dependencies: [
                "Alamofire",
                .product(name: "Moya", package: "Moya"),
                .product(name: "RxMoya", package: "Moya"),
                "Kingfisher",
                .product(name: "Nuke", package: "Nuke"),
                .product(name: "NukeUI", package: "Nuke"),
                .product(name: "Lottie", package: "lottie-ios"),
                "CombineExt",
                .product(name: "RxSwift", package: "RxSwift"),
                .product(name: "RxCocoa", package: "RxSwift"),
                .product(name: "RealmSwift", package: "realm-swift"),
                .product(name: "SQLite", package: "SQLite.swift"),
                .product(name: "FirebaseAnalytics", package: "firebase-ios-sdk"),
                .product(name: "FirebaseCrashlytics", package: "firebase-ios-sdk"),
                .product(name: "FirebaseAuth", package: "firebase-ios-sdk"),
                .product(name: "FirebaseFirestore", package: "firebase-ios-sdk"),
                "SwiftyJSON",
                "KeychainAccess"
            ]
        ),
        .testTarget(
            name: "MyAppKitTests",
            dependencies: ["MyAppKit"]
        )
    ]
)

// バージョン指定の方法

/*
1. from: "1.0.0"
   → 1.0.0 以上、2.0.0 未満

2. exact: "1.5.0"
   → 正確に 1.5.0

3. .upToNextMajor(from: "1.5.0")
   → 1.5.0 以上、2.0.0 未満

4. .upToNextMinor(from: "1.5.0")
   → 1.5.0 以上、1.6.0 未満

5. branch: "main"
   → main ブランチの最新

6. revision: "abc123def"
   → 特定のコミット

推奨:
- 本番: from: "x.y.z" （セマンティックバージョニング）
- 開発: branch: "develop" （最新機能を試す）
- 安定: exact: "x.y.z" （バージョン固定）
*/
```

### SPM のワークフロー

```bash
# Package の追加・更新

# 1. Package を追加
# File > Add Packages... (GUI)
# または Package.swift を編集

# 2. Package を更新
# File > Packages > Update to Latest Package Versions
# または
xcodebuild -resolvePackageDependencies

# 3. Package を削除
# Project Navigator > Package Dependencies > 右クリック > Remove

# 4. Package.resolved の管理

# Package.resolved: 現在インストールされているバージョンを記録
# Git で管理すべき: チーム全体で同じバージョンを使用

git add Package.resolved
git commit -m "chore: update package dependencies"

# 5. キャッシュのクリア

# 問題が起きた場合:
# File > Packages > Reset Package Caches

# コマンドライン:
rm -rf ~/Library/Caches/org.swift.swiftpm
rm -rf ~/Library/Developer/Xcode/DerivedData

# 6. Package の検索

# Swift Package Index:
# https://swiftpackageindex.com

# GitHub Topic: swift-package-manager
# https://github.com/topics/swift-package-manager
```

### ローカル Package の作成

```swift
// ローカル Package でモジュール化

/*
メリット:
- コードの分離とカプセル化
- ビルド時間の短縮（変更部分のみ再ビルド）
- テストの容易さ
- 再利用性の向上
*/

// MyAppCore という Package を作成

// File > New > Package...
// Package Name: MyAppCore
// Add to: MyApp
// Group: Frameworks

// MyAppCore/Package.swift
import PackageDescription

let package = Package(
    name: "MyAppCore",
    platforms: [
        .iOS(.v15)
    ],
    products: [
        .library(
            name: "MyAppCore",
            targets: ["MyAppCore"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/Alamofire/Alamofire.git", from: "5.8.0"),
    ],
    targets: [
        .target(
            name: "MyAppCore",
            dependencies: ["Alamofire"]
        ),
        .testTarget(
            name: "MyAppCoreTests",
            dependencies: ["MyAppCore"]
        ),
    ]
)

// MyAppCore/Sources/MyAppCore/MyAppCore.swift
public struct MyAppCore {
    public init() {}

    public func hello() -> String {
        "Hello from MyAppCore!"
    }
}

// メインアプリから使用
import MyAppCore

let core = MyAppCore()
print(core.hello())

// プロジェクト構成例

/*
MyApp/
├── MyApp/                  # メインアプリ
├── MyAppCore/              # ビジネスロジック
│   ├── Sources/
│   │   └── MyAppCore/
│   │       ├── Networking/
│   │       ├── Database/
│   │       └── Services/
│   └── Tests/
├── MyAppUI/                # UI コンポーネント
│   ├── Sources/
│   │   └── MyAppUI/
│   │       ├── Components/
│   │       ├── Modifiers/
│   │       └── Styles/
│   └── Tests/
└── MyAppTests/
*/
```

---

## CocoaPods

### CocoaPods の概要

```ruby
# CocoaPods とは

=begin
Ruby ベースのパッケージマネージャー

メリット:
✅ 最も多くのライブラリが対応
✅ バイナリフレームワークのサポート
✅ リソースファイルの管理が容易
✅ 細かい設定が可能
✅ 豊富なドキュメント

デメリット:
❌ Ruby のインストールが必要
❌ ビルドが遅い（特に大規模プロジェクト）
❌ Xcode プロジェクトファイルを変更
❌ .xcworkspace での管理が必要

推奨度: ⭐⭐⭐
→ SPM 未対応のライブラリがある場合に使用
=end
```

### CocoaPods のインストール

```bash
# Ruby のインストール確認
ruby -v

# CocoaPods のインストール
sudo gem install cocoapods

# または Homebrew 経由（推奨）
brew install cocoapods

# インストール確認
pod --version

# CocoaPods のセットアップ
pod setup

# バージョン管理（Bundler 使用を推奨）
# Gemfile を作成
cat > Gemfile << 'EOF'
source "https://rubygems.org"

gem "cocoapods", "~> 1.14"
gem "fastlane", "~> 2.219"
EOF

# 依存関係のインストール
bundle install

# 以降は bundle exec を使用
bundle exec pod install
```

### Podfile の作成

```ruby
# Podfile

# iOS 最小バージョン
platform :ios, '15.0'

# Swift バージョンを無視（警告抑制）
inhibit_all_warnings!

# use_frameworks! の設定
use_frameworks!
# または動的フレームワークを使用しない場合:
# use_modular_headers!

# ソースリポジトリ
source 'https://github.com/CocoaPods/Specs.git'

# メインターゲット
target 'MyApp' do
  # MARK: - Networking
  pod 'Alamofire', '~> 5.8'
  pod 'Moya', '~> 15.0'
  pod 'Moya/RxSwift', '~> 15.0'

  # MARK: - UI
  pod 'Kingfisher', '~> 7.10'
  pod 'SnapKit', '~> 5.6'
  pod 'lottie-ios', '~> 4.4'
  pod 'SkeletonView', '~> 1.30'

  # MARK: - Reactive
  pod 'RxSwift', '~> 6.6'
  pod 'RxCocoa', '~> 6.6'
  pod 'RxGesture', '~> 4.0'

  # MARK: - Database
  pod 'RealmSwift', '~> 10.45'

  # MARK: - Firebase
  pod 'Firebase/Analytics', '~> 10.20'
  pod 'Firebase/Crashlytics', '~> 10.20'
  pod 'Firebase/Auth', '~> 10.20'
  pod 'Firebase/Firestore', '~> 10.20'
  pod 'Firebase/RemoteConfig', '~> 10.20'
  pod 'Firebase/Messaging', '~> 10.20'

  # MARK: - Utility
  pod 'SwiftyJSON', '~> 5.0'
  pod 'KeychainAccess', '~> 4.2'
  pod 'IQKeyboardManagerSwift', '~> 6.5'

  # MARK: - Development Tools (Debug のみ)
  pod 'SwiftLint', '~> 0.54', :configurations => ['Debug']
  pod 'FLEX', '~> 5.22', :configurations => ['Debug']

  # Test ターゲット
  target 'MyAppTests' do
    inherit! :search_paths
    pod 'Quick', '~> 7.3'
    pod 'Nimble', '~> 12.3'
    pod 'RxBlocking', '~> 6.6'
    pod 'RxTest', '~> 6.6'
  end

  # UI Test ターゲット
  target 'MyAppUITests' do
    inherit! :search_paths
  end
end

# Post Install フック
post_install do |installer|
  installer.pods_project.targets.each do |target|
    target.build_configurations.each do |config|
      # iOS 最小バージョンを統一
      config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '15.0'

      # Bitcode を無効化
      config.build_settings['ENABLE_BITCODE'] = 'NO'

      # Swift バージョン警告を抑制
      config.build_settings['SWIFT_SUPPRESS_WARNINGS'] = 'YES'

      # Dead Code Stripping を有効化
      config.build_settings['DEAD_CODE_STRIPPING'] = 'YES'
    end
  end
end

# バージョン指定の方法

=begin
1. 'Pod名', '~> 1.0'
   → 1.0 以上、2.0 未満

2. 'Pod名', '~> 1.5.0'
   → 1.5.0 以上、1.6.0 未満

3. 'Pod名', '1.5.0'
   → 正確に 1.5.0

4. 'Pod名', '>= 1.0', '< 2.0'
   → 1.0 以上、2.0 未満

5. 'Pod名', :git => 'https://github.com/user/repo.git'
   → Git リポジトリから直接

6. 'Pod名', :git => '...', :branch => 'develop'
   → 特定のブランチ

7. 'Pod名', :git => '...', :tag => 'v1.0.0'
   → 特定のタグ

8. 'Pod名', :git => '...', :commit => 'abc123'
   → 特定のコミット

9. 'Pod名', :path => '../LocalPod'
   → ローカルの Pod
=end
```

### CocoaPods のワークフロー

```bash
# 1. Podfile の作成
pod init

# 2. Podfile を編集
# エディタで Podfile を開き、依存関係を追加

# 3. Pod のインストール
pod install

# 出力:
# Analyzing dependencies
# Downloading dependencies
# Installing Alamofire (5.8.0)
# Installing Moya (15.0.0)
# ...
# Generating Pods project
# Integrating client project

# 4. .xcworkspace を開く
open MyApp.xcworkspace

# 注意: 今後は .xcworkspace を使用
# .xcodeproj ではなく .xcworkspace

# 5. Pod の更新
pod update

# 特定の Pod のみ更新
pod update Alamofire

# 6. Pod の削除
# Podfile から該当行を削除して
pod install

# 7. Pod の検索
pod search Alamofire

# 8. Pod の詳細確認
pod spec cat Alamofire

# 9. Podfile.lock の管理
# Podfile.lock: インストールされたバージョンを記録
# Git で管理必須

git add Podfile.lock
git commit -m "chore: update pods"

# 10. キャッシュのクリア
pod cache clean --all
rm -rf ~/Library/Caches/CocoaPods
rm -rf Pods
pod install

# 11. デバッグ
pod install --verbose
pod install --no-repo-update  # Specs の更新をスキップ
```

### CocoaPods のベストプラクティス

```ruby
# 1. Podfile.lock を Git で管理

# .gitignore に追加しない
# Podfile.lock

# 理由:
# - チーム全員が同じバージョンを使用
# - ビルドの再現性
# - CI/CD での一貫性

# 2. Pods ディレクトリは Git で管理しない

# .gitignore
Pods/

# 理由:
# - リポジトリサイズの肥大化を防ぐ
# - pod install で復元可能
# - マージコンフリクトを避ける

# 3. バージョンを固定

# ❌ 避けるべき
pod 'Alamofire'  # 最新版（予期しない変更）

# ✅ 推奨
pod 'Alamofire', '~> 5.8'  # セマンティックバージョニング

# 4. Subspecs を活用

# Firebase の必要な機能のみインストール
pod 'Firebase/Analytics'
pod 'Firebase/Crashlytics'

# 不要な機能は除外（ビルド時間削減）
# pod 'Firebase'  # 全機能（非推奨）

# 5. Development Pods の分離

pod 'SwiftLint', :configurations => ['Debug']
pod 'FLEX', :configurations => ['Debug']

# Release ビルドには含まれない

# 6. Post Install で設定を統一

post_install do |installer|
  installer.pods_project.targets.each do |target|
    target.build_configurations.each do |config|
      config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '15.0'
    end
  end
end

# 7. Bundler で CocoaPods のバージョン管理

# Gemfile
gem "cocoapods", "~> 1.14"

# インストール
bundle install

# 使用
bundle exec pod install

# CI/CD でも同じバージョンを使用
```

---

## Carthage

### Carthage の概要

```bash
# Carthage とは

# 分散型のパッケージマネージャー
# バイナリフレームワークをビルド・配布

# メリット:
# ✅ Xcode プロジェクトを変更しない
# ✅ ビルド済みフレームワークで高速
# ✅ シンプルな設計

# デメリット:
# ❌ 対応ライブラリが少ない
# ❌ 設定が煩雑
# ❌ メンテナンスが停滞気味

# 推奨度: ⭐
# → 新規プロジェクトでは非推奨
# → SPM または CocoaPods を推奨
```

### Carthage のインストール

```bash
# Homebrew でインストール
brew install carthage

# バージョン確認
carthage version
```

### Cartfile の作成

```bash
# Cartfile

# GitHub リポジトリ
github "Alamofire/Alamofire" ~> 5.8
github "Moya/Moya" ~> 15.0

# Git リポジトリ
git "https://github.com/username/repo.git" ~> 1.0

# バイナリのみ
binary "https://example.com/framework.json" ~> 1.0

# バージョン指定
# ~> 5.8: 5.8 以上、6.0 未満
# == 5.8.0: 正確に 5.8.0
# >= 5.8: 5.8 以上
# branch: develop ブランチ
# revision: 特定のコミット
```

### Carthage のワークフロー

```bash
# 1. フレームワークのビルド
carthage update --use-xcframeworks --platform iOS

# 2. プロジェクトに追加
# TARGETS > General > Frameworks, Libraries, and Embedded Content
# Carthage/Build/*.xcframework をドラッグ&ドロップ

# 3. Run Script Phase を追加
# TARGETS > Build Phases > "+" > New Run Script Phase
# Script:
/usr/local/bin/carthage copy-frameworks

# Input Files:
$(SRCROOT)/Carthage/Build/Alamofire.xcframework

# 4. 更新
carthage update Alamofire --use-xcframeworks --platform iOS

# 5. キャッシュの利用
carthage update --cache-builds

# 6. .gitignore
Carthage/Build/
Carthage/Checkouts/

# Git で管理するのは Cartfile と Cartfile.resolved のみ
```

---

## 依存関係管理の比較と選択

### 機能比較表

```swift
// SPM vs CocoaPods vs Carthage

/*
┌─────────────────────┬─────────┬────────────┬──────────┐
│ 機能                │ SPM     │ CocoaPods  │ Carthage │
├─────────────────────┼─────────┼────────────┼──────────┤
│ Apple 公式          │ ✅      │ ❌         │ ❌       │
│ Xcode 統合          │ ✅      │ ⭕         │ ❌       │
│ 追加インストール    │ 不要    │ 必要(Ruby) │ 必要     │
│ 設定の簡単さ        │ ⭐⭐⭐ │ ⭐⭐       │ ⭐       │
│ ビルド速度          │ ⭐⭐⭐ │ ⭐         │ ⭐⭐⭐   │
│ ライブラリ対応数    │ ⭐⭐   │ ⭐⭐⭐     │ ⭐       │
│ バイナリサポート    │ ⭕     │ ✅         │ ✅       │
│ リソース管理        │ ⭕     │ ✅         │ ❌       │
│ プロジェクト変更    │ 最小限  │ 大きい     │ なし     │
│ 学習コスト          │ 低      │ 中         │ 高       │
│ CI/CD との親和性    │ ⭐⭐⭐ │ ⭐⭐       │ ⭐⭐     │
│ 将来性              │ ⭐⭐⭐ │ ⭐⭐       │ ⭐       │
└─────────────────────┴─────────┴────────────┴──────────┘

推奨度:
1. SPM: ⭐⭐⭐⭐⭐ (第一選択)
2. CocoaPods: ⭐⭐⭐ (SPM 未対応時)
3. Carthage: ⭐ (非推奨)
*/
```

### 選択ガイドライン

```swift
// プロジェクトタイプ別推奨

/*
1. 新規プロジェクト
   → SPM 単独
   理由: 最もシンプル、将来性が高い

2. 既存プロジェクト（CocoaPods）
   → CocoaPods 継続 or 段階的に SPM 移行
   理由: 移行コストを考慮

3. SPM 未対応ライブラリあり
   → SPM + CocoaPods 併用
   理由: SPM を優先し、必要時のみ CocoaPods

4. バイナリフレームワーク中心
   → SPM (XCFramework 対応)
   理由: SPM のバイナリサポートが向上

5. 大規模プロジェクト
   → SPM (モジュール分割)
   理由: ビルド時間の最適化
*/

// 併用時の注意点

/*
SPM + CocoaPods 併用:

1. 同じライブラリを重複させない
   ❌ SPM と CocoaPods で Alamofire をインストール
   ✅ SPM のみまたは CocoaPods のみ

2. 依存関係の競合に注意
   - バージョンの不一致
   - トランジティブ依存関係の衝突

3. ビルド設定の統一
   - iOS Deployment Target
   - Swift Version
   - Build Settings

4. .gitignore の管理
   .gitignore
   Pods/
   .swiftpm/
   *.xcworkspace
   Package.resolved (プロジェクトによる)
   Podfile.lock (Git で管理)
*/
```

---

## Fastlane セットアップ

### Fastlane の概要

```ruby
# Fastlane とは

=begin
iOS / Android アプリの開発・デプロイを自動化

主な機能:
- ビルド自動化
- スクリーンショット自動生成
- App Store / TestFlight へのアップロード
- 証明書・プロファイル管理（match）
- バージョン番号の管理
- プッシュ通知のテスト

メリット:
✅ 反復作業の自動化
✅ ヒューマンエラーの削減
✅ CI/CD との統合
✅ チーム全体で同じプロセスを共有
=end
```

### Fastlane のインストール

```bash
# Homebrew でインストール（推奨）
brew install fastlane

# または RubyGems
sudo gem install fastlane -NV

# Bundler 経由（推奨: バージョン固定）
# Gemfile
source "https://rubygems.org"

gem "fastlane", "~> 2.219"

# インストール
bundle install

# 以降は bundle exec を使用
bundle exec fastlane
```

### Fastlane の初期化

```bash
# プロジェクトルートで実行
fastlane init

# 対話形式で質問に答える

# 1. What would you like to use fastlane for?
#    4: Manual setup

# 2. Confirm Apple ID
#    your@email.com

# 3. 自動的に Fastfile と Appfile が生成される

# ディレクトリ構造:
# fastlane/
# ├── Appfile       # Apple ID, Bundle ID
# ├── Fastfile      # Lane の定義
# └── README.md
```

### Appfile の設定

```ruby
# fastlane/Appfile

# Apple ID
apple_id("your@email.com")

# Team ID (Apple Developer Portal で確認)
team_id("ABCD123456")

# iTunes Connect Team ID (複数チームに所属している場合)
itunes_connect_id("12345678")

# Bundle Identifier
app_identifier("com.company.myapp")

# 環境ごとに異なる Bundle ID
for_lane :development do
  app_identifier("com.company.myapp.dev")
end

for_lane :staging do
  app_identifier("com.company.myapp.staging")
end

for_lane :production do
  app_identifier("com.company.myapp")
end
```

### Fastfile の基本構成

```ruby
# fastlane/Fastfile

default_platform(:ios)

platform :ios do
  # 変数定義
  before_all do
    setup_ci if ENV['CI']
    ensure_git_status_clean unless ENV['CI']
  end

  # MARK: - Development

  desc "Development ビルド"
  lane :dev do
    build_app(
      scheme: "MyApp (Development)",
      configuration: "Debug",
      export_method: "development",
      output_directory: "./build",
      output_name: "MyApp-Dev.ipa",
      clean: true
    )
  end

  # MARK: - Testing

  desc "Unit Tests 実行"
  lane :test do
    scan(
      scheme: "MyApp",
      device: "iPhone 15 Pro",
      code_coverage: true,
      output_directory: "./test_output",
      clean: true
    )
  end

  desc "UI Tests 実行"
  lane :ui_test do
    scan(
      scheme: "MyApp",
      device: "iPhone 15 Pro",
      only_testing: ["MyAppUITests"],
      output_directory: "./test_output"
    )
  end

  # MARK: - Staging

  desc "Staging ビルドと TestFlight アップロード"
  lane :staging do
    # バージョン番号を確認
    ensure_git_branch(branch: "develop")

    # 証明書とプロファイルの同期
    match(
      type: "appstore",
      app_identifier: "com.company.myapp.staging",
      readonly: is_ci
    )

    # ビルド番号をインクリメント
    increment_build_number(
      build_number: latest_testflight_build_number + 1
    )

    # ビルド
    build_app(
      scheme: "MyApp (Staging)",
      configuration: "Staging",
      export_method: "app-store",
      output_directory: "./build",
      output_name: "MyApp-Staging.ipa"
    )

    # TestFlight へアップロード
    upload_to_testflight(
      skip_waiting_for_build_processing: true,
      changelog: git_changelog,
      distribute_external: false,
      groups: ["Internal Testers"]
    )

    # Slack 通知
    slack(
      message: "Staging build uploaded to TestFlight!",
      success: true
    )
  end

  # MARK: - Production

  desc "Production ビルドと App Store 申請"
  lane :release do
    ensure_git_branch(branch: "main")
    ensure_git_status_clean

    # バージョン番号の更新
    version = prompt(text: "Enter version number (e.g. 1.2.0): ")
    increment_version_number(version_number: version)

    # 証明書とプロファイルの同期
    match(
      type: "appstore",
      app_identifier: "com.company.myapp",
      readonly: is_ci
    )

    # ビルド番号をインクリメント
    increment_build_number(
      build_number: latest_testflight_build_number + 1
    )

    # ビルド
    build_app(
      scheme: "MyApp (Production)",
      configuration: "Release",
      export_method: "app-store"
    )

    # TestFlight へアップロード
    upload_to_testflight(
      skip_waiting_for_build_processing: false,
      changelog: git_changelog
    )

    # App Store へ申請（手動レビュー）
    # deliver(
    #   submit_for_review: false,
    #   automatic_release: false,
    #   force: true
    # )

    # Git タグを作成
    add_git_tag(
      tag: "v#{version}"
    )
    push_git_tags

    # Slack 通知
    slack(
      message: "Version #{version} uploaded to TestFlight!",
      success: true
    )
  end

  # MARK: - Utilities

  desc "スクリーンショット生成"
  lane :screenshots do
    capture_screenshots(
      scheme: "MyApp",
      devices: [
        "iPhone 15 Pro Max",
        "iPhone 15 Pro",
        "iPhone SE (3rd generation)",
        "iPad Pro (12.9-inch) (6th generation)"
      ],
      languages: ["en-US", "ja"],
      output_directory: "./screenshots",
      clear_previous_screenshots: true
    )

    frame_screenshots(
      white: false,
      path: "./screenshots"
    )
  end

  desc "バージョン番号を上げる"
  lane :bump_version do |options|
    type = options[:type] || "patch"

    case type
    when "major"
      increment_version_number(bump_type: "major")
    when "minor"
      increment_version_number(bump_type: "minor")
    when "patch"
      increment_version_number(bump_type: "patch")
    else
      UI.user_error!("Invalid bump type: #{type}")
    end

    version = get_version_number
    commit_version_bump(message: "chore: bump version to #{version}")
  end

  desc "証明書とプロファイルの同期"
  lane :sync_certificates do
    match(type: "development")
    match(type: "appstore")
  end

  # エラーハンドリング
  error do |lane, exception, options|
    slack(
      message: "Error in lane #{lane}: #{exception.message}",
      success: false
    )
  end

  # ヘルパーメソッド
  def git_changelog
    changelog_from_git_commits(
      between: [last_git_tag, "HEAD"],
      pretty: "- %s",
      merge_commit_filtering: "exclude_merges"
    )
  end
end
```

### Fastlane の使用

```bash
# Lane の実行
bundle exec fastlane dev
bundle exec fastlane test
bundle exec fastlane staging
bundle exec fastlane release

# オプション付き実行
bundle exec fastlane bump_version type:minor

# 利用可能な Lane の一覧
bundle exec fastlane lanes

# ドキュメント生成
bundle exec fastlane docs

# 環境変数の使用
SLACK_URL="https://hooks.slack.com/..." bundle exec fastlane staging

# CI/CD での実行（headless モード）
CI=true bundle exec fastlane test
```

---

(文字数制限のため続く...)
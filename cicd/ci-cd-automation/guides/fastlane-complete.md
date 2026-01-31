# Fastlane 完全ガイド

## 目次
1. [Fastlaneの基礎](#fastlaneの基礎)
2. [セットアップ](#セットアップ)
3. [Lane設計](#lane設計)
4. [証明書・プロビジョニング管理](#証明書プロビジョニング管理)
5. [ビルド自動化](#ビルド自動化)
6. [配布自動化](#配布自動化)
7. [スクリーンショット自動化](#スクリーンショット自動化)
8. [トラブルシューティング](#トラブルシューティング)

---

## Fastlaneの基礎

### Fastlaneとは

Fastlaneは、iOSとAndroidアプリの開発・ビルド・デプロイプロセスを自動化するツールです。

```ruby
# Fastlaneの主な機能

# 1. ビルド自動化
build_app(scheme: "MyApp")

# 2. テスト自動化
run_tests(scheme: "MyApp")

# 3. 証明書管理
match(type: "appstore")

# 4. スクリーンショット生成
snapshot

# 5. App Store Connect操作
upload_to_app_store

# 6. TestFlight配布
upload_to_testflight
```

### Fastlaneのアーキテクチャ

```
Project/
├── fastlane/
│   ├── Fastfile              # Lane定義
│   ├── Appfile               # アプリ情報
│   ├── Matchfile             # 証明書管理設定
│   ├── Snapfile              # スクリーンショット設定
│   ├── Deliverfile           # App Store設定
│   ├── Gymfile               # ビルド設定
│   ├── Scanfile              # テスト設定
│   └── metadata/             # App Store メタデータ
│       ├── en-US/
│       │   ├── name.txt
│       │   ├── subtitle.txt
│       │   ├── description.txt
│       │   ├── keywords.txt
│       │   └── screenshots/
│       └── ja/
└── MyApp.xcodeproj
```

---

## セットアップ

### 初期セットアップ

```bash
# 1. Fastlaneのインストール
sudo gem install fastlane -NV

# または Homebrew
brew install fastlane

# 2. プロジェクトでFastlaneを初期化
cd /path/to/your/project
fastlane init

# 対話形式で選択:
# 1. 📸  Automate screenshots
# 2. 👩‍✈️  Automate beta distribution to TestFlight
# 3. 🚀  Automate App Store distribution
# 4. 🛠  Manual setup
```

### Appfileの設定

```ruby
# fastlane/Appfile

app_identifier("com.company.myapp")           # Bundle Identifier
apple_id("developer@company.com")             # Apple ID
itc_team_id("123456789")                      # App Store Connect Team ID
team_id("ABCDE12345")                         # Developer Portal Team ID

# 環境変数から取得する場合
# app_identifier(ENV["APP_IDENTIFIER"])
# apple_id(ENV["APPLE_ID"])

# 複数のターゲットがある場合
for_platform :ios do
  for_lane :production do
    app_identifier("com.company.myapp")
  end

  for_lane :staging do
    app_identifier("com.company.myapp.staging")
  end
end
```

### Gemfileの作成

```ruby
# Gemfile

source "https://rubygems.org"

gem "fastlane"
gem "cocoapods"

# プラグイン
plugins_path = File.join(File.dirname(__FILE__), 'fastlane', 'Pluginfile')
eval_gemfile(plugins_path) if File.exist?(plugins_path)
```

```bash
# Gemfileから依存関係をインストール
bundle install

# 以降はbundleを通してfastlaneを実行
bundle exec fastlane [lane_name]
```

---

## Lane設計

### 基本的なLane

```ruby
# fastlane/Fastfile

default_platform(:ios)

platform :ios do
  # 開発ビルド
  lane :dev do
    build_app(
      scheme: "MyApp-Dev",
      configuration: "Debug",
      export_method: "development"
    )
  end

  # ステージングビルド → TestFlight
  lane :staging do
    # 1. テスト実行
    run_tests(scheme: "MyApp-Staging")

    # 2. ビルド番号インクリメント
    increment_build_number

    # 3. ビルド
    build_app(
      scheme: "MyApp-Staging",
      configuration: "Release",
      export_method: "app-store"
    )

    # 4. TestFlightにアップロード
    upload_to_testflight(
      skip_waiting_for_build_processing: true
    )

    # 5. Slackに通知
    slack(
      message: "Staging build uploaded to TestFlight! 🚀",
      success: true
    )
  end

  # プロダクションビルド → App Store
  lane :release do
    # 1. テスト実行
    run_tests(scheme: "MyApp")

    # 2. バージョン番号確認
    ensure_git_status_clean
    ensure_git_branch(branch: 'main')

    # 3. ビルド番号インクリメント
    increment_build_number

    # 4. ビルド
    build_app(
      scheme: "MyApp",
      configuration: "Release",
      export_method: "app-store"
    )

    # 5. App Store Connectにアップロード
    upload_to_app_store(
      skip_metadata: true,
      skip_screenshots: true,
      submit_for_review: false
    )

    # 6. Gitタグ作成
    add_git_tag(
      tag: "v#{get_version_number}-#{get_build_number}"
    )

    # 7. Slackに通知
    slack(
      message: "Production build uploaded to App Store! 🎉",
      success: true
    )
  end
end
```

### 共通処理の抽出

```ruby
# fastlane/Fastfile

platform :ios do
  # 共通の前処理
  before_all do
    # Cocoapodsの更新
    cocoapods(
      clean_install: true
    )

    # 証明書とプロビジョニングプロファイルの同期
    match(type: "appstore", readonly: true)
  end

  # エラー時の処理
  error do |lane, exception, options|
    slack(
      message: "Lane #{lane} failed: #{exception}",
      success: false
    )
  end

  # 成功時の処理
  after_all do |lane, options|
    notification(
      subtitle: "Fastlane",
      message: "Lane #{lane} completed successfully!"
    )
  end

  # プライベートLane（他のLaneから呼び出し専用）
  private_lane :prepare_build do
    clear_derived_data
    increment_build_number
  end

  # 使用例
  lane :beta do
    prepare_build
    build_app(scheme: "MyApp")
    upload_to_testflight
  end
end
```

### パラメータ付きLane

```ruby
platform :ios do
  desc "Build with custom scheme"
  lane :build do |options|
    # パラメータの取得
    scheme = options[:scheme] || "MyApp"
    configuration = options[:configuration] || "Release"
    clean = options[:clean] || false

    # Clean Build
    if clean
      clear_derived_data
    end

    # ビルド
    build_app(
      scheme: scheme,
      configuration: configuration,
      export_method: "app-store"
    )
  end
end

# 実行例
# fastlane build scheme:MyApp-Staging configuration:Debug clean:true
```

---

## 証明書・プロビジョニング管理

### Match による証明書管理

```ruby
# fastlane/Matchfile

git_url("git@github.com:company/certificates.git")
git_branch("main")

storage_mode("git")
type("appstore")

app_identifier(["com.company.myapp", "com.company.myapp.staging"])
username("developer@company.com")
team_id("ABCDE12345")

# 暗号化パスフレーズ（環境変数から取得推奨）
# ENV["MATCH_PASSWORD"]
```

### Match の使用

```ruby
# fastlane/Fastfile

platform :ios do
  # 初回セットアップ（証明書とプロファイルを作成してGitに保存）
  lane :setup_match do
    match(
      type: "development",
      app_identifier: "com.company.myapp"
    )

    match(
      type: "appstore",
      app_identifier: "com.company.myapp"
    )
  end

  # 証明書の同期（CI/CDや新しいマシンで実行）
  lane :sync_certificates do
    match(
      type: "appstore",
      readonly: true  # 読み取り専用（新規作成しない）
    )
  end

  # 証明書の更新
  lane :renew_certificates do
    match(
      type: "appstore",
      force_for_new_devices: true  # 新しいデバイスが追加された場合
    )
  end
end
```

### 環境変数の管理

```bash
# .env ファイル（Gitにはコミットしない）

# Match
MATCH_PASSWORD=your_encryption_password
MATCH_GIT_URL=git@github.com:company/certificates.git

# App Store Connect
FASTLANE_USER=developer@company.com
FASTLANE_PASSWORD=@keychain:fastlane_password
FASTLANE_APPLE_APPLICATION_SPECIFIC_PASSWORD=xxxx-xxxx-xxxx-xxxx

# Slack
SLACK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
```

```ruby
# Fastfileで環境変数を読み込む
before_all do
  # .envファイルを読み込む（fastlane-plugin-dotenvが必要）
  Dotenv.load('.env')
end
```

---

## ビルド自動化

### Gym（build_app）の詳細設定

```ruby
# fastlane/Gymfile

# スキームとConfiguration
scheme("MyApp")
configuration("Release")

# エクスポート設定
export_method("app-store")  # app-store, ad-hoc, development, enterprise
output_directory("./build")
output_name("MyApp.ipa")

# ビルドオプション
clean(true)
include_bitcode(false)
include_symbols(true)
export_xcargs("-allowProvisioningUpdates")

# コード署名
codesigning_identity("iPhone Distribution: Company Name (ABCDE12345)")
export_options({
  provisioningProfiles: {
    "com.company.myapp" => "match AppStore com.company.myapp"
  }
})
```

### Fastfileでのビルド設定

```ruby
platform :ios do
  lane :build_production do
    # 1. Derived Dataをクリア
    clear_derived_data

    # 2. ビルド番号を自動インクリメント
    increment_build_number(
      build_number: latest_testflight_build_number + 1
    )

    # 3. ビルド
    build_app(
      scheme: "MyApp",
      configuration: "Release",
      export_method: "app-store",
      output_directory: "./build/#{Time.now.strftime('%Y%m%d_%H%M%S')}",
      output_name: "MyApp-#{get_version_number}-#{get_build_number}.ipa",
      clean: true,
      include_bitcode: false,
      include_symbols: true,
      export_options: {
        method: "app-store",
        provisioningProfiles: {
          "com.company.myapp" => "match AppStore com.company.myapp"
        },
        signingStyle: "manual",
        stripSwiftSymbols: true,
        uploadSymbols: true,
        compileBitcode: false
      },
      xcargs: "-allowProvisioningUpdates"
    )

    # 4. dSYMをFirebase Crashlyticsにアップロード
    upload_symbols_to_crashlytics(
      gsp_path: "./MyApp/GoogleService-Info.plist",
      binary_path: "./Pods/FirebaseCrashlytics/upload-symbols"
    )
  end
end
```

### バージョン管理

```ruby
platform :ios do
  # バージョン番号取得
  lane :get_version do
    version = get_version_number(target: "MyApp")
    build = get_build_number

    puts "Current version: #{version} (#{build})"
  end

  # バージョン番号設定
  lane :set_version do |options|
    increment_version_number(
      version_number: options[:version],
      xcodeproj: "MyApp.xcodeproj"
    )
  end

  # ビルド番号インクリメント
  lane :bump_build do
    increment_build_number(
      build_number: latest_testflight_build_number + 1
    )
  end

  # セマンティックバージョニング
  lane :bump_major do
    increment_version_number(bump_type: "major")
  end

  lane :bump_minor do
    increment_version_number(bump_type: "minor")
  end

  lane :bump_patch do
    increment_version_number(bump_type: "patch")
  end
end
```

---

## 配布自動化

### TestFlight配布

```ruby
platform :ios do
  desc "Upload to TestFlight"
  lane :beta do
    # 1. テスト実行
    run_tests(
      scheme: "MyApp",
      devices: ["iPhone 15 Pro"]
    )

    # 2. 証明書同期
    match(type: "appstore", readonly: true)

    # 3. ビルド
    build_app(
      scheme: "MyApp",
      export_method: "app-store"
    )

    # 4. TestFlightにアップロード
    upload_to_testflight(
      # ベータ情報
      changelog: "Bug fixes and improvements",
      beta_app_description: "MyApp beta version for testing",
      beta_app_feedback_email: "feedback@company.com",

      # テストグループ
      groups: ["Internal Testers", "External Testers"],

      # オプション
      skip_submission: false,
      skip_waiting_for_build_processing: false,
      distribute_external: true,
      notify_external_testers: true,

      # App Store Connect API Key（2FAを避ける）
      api_key_path: "./fastlane/app_store_connect_api_key.json"
    )

    # 5. Slackに通知
    slack(
      message: "New beta build is live on TestFlight! 🎉",
      channel: "#ios-releases",
      payload: {
        "Version" => get_version_number,
        "Build" => get_build_number
      }
    )
  end
end
```

### App Store配布

```ruby
platform :ios do
  desc "Deploy to App Store"
  lane :deploy do
    # 1. Gitの状態確認
    ensure_git_status_clean
    ensure_git_branch(branch: 'main')

    # 2. テスト実行
    run_tests(scheme: "MyApp")

    # 3. ビルド
    build_app(scheme: "MyApp")

    # 4. App Store Connectにアップロード
    upload_to_app_store(
      # メタデータ
      submit_for_review: true,
      automatic_release: false,
      force: true,

      # 審査情報
      submission_information: {
        add_id_info_limits_tracking: true,
        add_id_info_serves_ads: false,
        add_id_info_tracks_action: false,
        add_id_info_tracks_install: true,
        add_id_info_uses_idfa: true,
        content_rights_has_rights: true,
        content_rights_contains_third_party_content: false,
        export_compliance_platform: 'ios',
        export_compliance_compliance_required: false,
        export_compliance_encryption_updated: false,
        export_compliance_app_type: nil,
        export_compliance_uses_encryption: false,
        export_compliance_is_exempt: false,
        export_compliance_contains_third_party_cryptography: false,
        export_compliance_contains_proprietary_cryptography: false
      },

      # リリースノート
      release_notes: {
        "en-US" => "Bug fixes and performance improvements",
        "ja" => "バグ修正とパフォーマンスの改善"
      },

      # フェーズドリリース
      phased_release: true,

      # App Store Connect API Key
      api_key_path: "./fastlane/app_store_connect_api_key.json"
    )

    # 5. Gitタグ作成
    version = get_version_number
    build = get_build_number
    add_git_tag(tag: "release/v#{version}-#{build}")
    push_git_tags

    # 6. GitHub Releaseを作成
    github_release = set_github_release(
      repository_name: "company/myapp",
      api_token: ENV["GITHUB_TOKEN"],
      name: "v#{version} (#{build})",
      tag_name: "release/v#{version}-#{build}",
      description: "Release notes here",
      is_draft: false,
      is_prerelease: false
    )

    # 7. Slackに通知
    slack(
      message: "New version submitted to App Store! 🚀",
      channel: "#ios-releases",
      payload: {
        "Version" => version,
        "Build" => build,
        "GitHub Release" => github_release["html_url"]
      }
    )
  end
end
```

### App Store Connect API Key

```bash
# App Store Connect API Keyの作成

# 1. App Store Connectにログイン
# 2. Users and Access → Keys → App Store Connect API
# 3. Generate API Key
#    - Name: Fastlane CI
#    - Access: Developer または App Manager
# 4. APIキーをダウンロード（AuthKey_XXXXXX.p8）
```

```json
// fastlane/app_store_connect_api_key.json

{
  "key_id": "ABCDE12345",
  "issuer_id": "12345678-1234-1234-1234-123456789012",
  "key": "-----BEGIN PRIVATE KEY-----\nMIGTA...\n-----END PRIVATE KEY-----",
  "duration": 1200,
  "in_house": false
}
```

```ruby
# Fastfileでの使用
api_key = app_store_connect_api_key(
  key_id: "ABCDE12345",
  issuer_id: "12345678-1234-1234-1234-123456789012",
  key_filepath: "./fastlane/AuthKey_ABCDE12345.p8",
  duration: 1200,
  in_house: false
)

upload_to_testflight(api_key: api_key)
```

---

## スクリーンショット自動化

### Snapshot設定

```ruby
# fastlane/Snapfile

devices([
  "iPhone 15 Pro Max",
  "iPhone 15 Pro",
  "iPhone SE (3rd generation)",
  "iPad Pro (12.9-inch) (6th generation)"
])

languages([
  "en-US",
  "ja"
])

scheme("MyAppUITests")

output_directory("./fastlane/screenshots")
clear_previous_screenshots(true)
override_status_bar(true)

# ステータスバーのオーバーライド設定
override_status_bar_arguments("--time 9:41 --dataNetwork wifi --wifiBars 3 --cellularMode active --batteryState charged --batteryLevel 100")
```

### UIテストでのスクリーンショット撮影

```swift
// MyAppUITests/ScreenshotTests.swift

import XCTest

class ScreenshotTests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false

        let app = XCUIApplication()
        setupSnapshot(app)
        app.launch()
    }

    func testTakeScreenshots() throws {
        let app = XCUIApplication()

        // 1. ホーム画面
        snapshot("01Home")

        // 2. 検索画面
        app.tabBars.buttons["Search"].tap()
        snapshot("02Search")

        // 3. プロフィール画面
        app.tabBars.buttons["Profile"].tap()
        snapshot("03Profile")

        // 4. 設定画面
        app.buttons["Settings"].tap()
        snapshot("04Settings")
    }
}
```

### Laneでのスクリーンショット生成

```ruby
platform :ios do
  desc "Generate screenshots"
  lane :screenshots do
    # 1. スクリーンショット撮影
    capture_screenshots(
      workspace: "MyApp.xcworkspace",
      scheme: "MyAppUITests"
    )

    # 2. フレーム付き画像生成
    frame_screenshots(
      white: true,
      path: "./fastlane/screenshots"
    )

    # 3. App Store Connectにアップロード
    upload_to_app_store(
      skip_binary_upload: true,
      skip_metadata: true,
      overwrite_screenshots: true
    )
  end
end
```

---

## トラブルシューティング

### よくあるエラーと解決方法

#### 1. 証明書エラー

```bash
# エラー
[!] Could not find a matching code signing identity for type 'AppStore'

# 解決方法
# 1. Matchで証明書を再同期
bundle exec fastlane match appstore --readonly

# 2. Keychainを確認
security find-identity -v -p codesigning

# 3. 証明書が期限切れの場合は再作成
bundle exec fastlane match appstore --force
```

#### 2. プロビジョニングプロファイルエラー

```bash
# エラー
Provisioning profile doesn't include the currently selected device

# 解決方法
# 1. デバイスをDeveloper Portalに登録
bundle exec fastlane run register_device udid:"xxxxx" name:"iPhone"

# 2. プロビジョニングプロファイルを再生成
bundle exec fastlane match development --force_for_new_devices
```

#### 3. TestFlight アップロードエラー

```bash
# エラー
The provided entity includes an attribute with a value that has already been used

# 解決方法
# ビルド番号が重複している
# 1. TestFlightの最新ビルド番号を取得してインクリメント
bundle exec fastlane run increment_build_number build_number:$(expr $(bundle exec fastlane run latest_testflight_build_number) + 1)
```

#### 4. 2FA（二要素認証）エラー

```bash
# エラー
Two-factor authentication is enabled

# 解決方法
# App Store Connect APIキーを使用
# 1. APIキーを作成してダウンロード
# 2. Fastfileで使用
api_key = app_store_connect_api_key(
  key_id: "KEY_ID",
  issuer_id: "ISSUER_ID",
  key_filepath: "./AuthKey_KEY_ID.p8"
)

upload_to_testflight(api_key: api_key)
```

### デバッグテクニック

```ruby
# Verbose モード
bundle exec fastlane beta --verbose

# 環境変数の表示
lane :debug do
  puts "App Identifier: #{CredentialsManager::AppfileConfig.try_fetch_value(:app_identifier)}"
  puts "Apple ID: #{CredentialsManager::AppfileConfig.try_fetch_value(:apple_id)}"
  puts "Team ID: #{CredentialsManager::AppfileConfig.try_fetch_value(:team_id)}"
end

# ドライラン（実際にアップロードしない）
upload_to_testflight(
  skip_submission: true,
  skip_waiting_for_build_processing: true
)
```

### パフォーマンス最適化

```ruby
platform :ios do
  lane :beta do
    # キャッシュを活用
    cocoapods(
      clean_install: false  # Podsディレクトリがあればスキップ
    )

    # Derived Dataは必要な時だけクリア
    # clear_derived_data

    # 並列テスト実行
    run_tests(
      devices: ["iPhone 15 Pro"],
      max_concurrent_simulators: 4
    )

    # ビルドのみ（アーカイブスキップ）
    build_app(
      skip_archive: true,
      skip_codesigning: true
    )
  end
end
```

---

このガイドでは、Fastlaneを使ったiOSアプリのCI/CD自動化について、セットアップから実際の運用まで詳細に解説しました。Match による証明書管理、TestFlightへの自動配布、スクリーンショット自動生成など、実践的なワークフローを構築できます。

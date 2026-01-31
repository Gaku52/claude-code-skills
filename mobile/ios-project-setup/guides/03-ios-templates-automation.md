# iOS Project Template & Automation - 完全ガイド

## 目次

1. [プロジェクトテンプレートの概要](#プロジェクトテンプレートの概要)
2. [Xcode Project Template作成](#xcode-project-template作成)
3. [セットアップ自動化スクリプト](#セットアップ自動化スクリプト)
4. [ボイラープレートコード生成](#ボイラープレートコード生成)
5. [環境設定の自動化](#環境設定の自動化)
6. [Feature Flags設定](#feature-flags設定)
7. [Analytics統合](#analytics統合)
8. [Crash Reporting統合](#crash-reporting統合)
9. [Localization Setup](#localization-setup)
10. [Accessibility Configuration](#accessibility-configuration)
11. [CI/CD Template](#cicd-template)
12. [ドキュメント自動生成](#ドキュメント自動生成)

---

## プロジェクトテンプレートの概要

### なぜテンプレートが必要か

```swift
// プロジェクトテンプレートの目的

/*
1. 開発時間の短縮
   - 繰り返し作業の削減
   - ボイラープレートの自動生成
   - 初期設定の統一

2. ベストプラクティスの適用
   - 推奨アーキテクチャ
   - コーディング規約
   - フォルダ構成

3. 品質の向上
   - テストの自動化
   - リント設定
   - CI/CD パイプライン

4. チーム統一
   - 全プロジェクトで同じ構成
   - オンボーディングの簡素化
   - 知識の共有

5. スケーラビリティ
   - 機能追加が容易
   - リファクタリングしやすい
   - 保守性の向上
*/

// テンプレートに含めるべき要素

/*
必須項目:
✅ プロジェクト構成（MVVM / Clean Architecture）
✅ Git設定（.gitignore, .gitattributes）
✅ 依存関係管理（SPM / CocoaPods）
✅ コード品質ツール（SwiftLint, SwiftFormat）
✅ CI/CD 設定（GitHub Actions）
✅ テスト環境
✅ ドキュメント（README, CONTRIBUTING）

推奨項目:
⭕ Fastlane 設定
⭕ Feature Flags
⭕ Analytics 統合
⭕ Crash Reporting
⭕ Localization
⭕ Accessibility
⭕ デザインシステム（色、フォント）
⭕ ネットワーク層
⭕ データベース設定
*/
```

---

## Xcode Project Template作成

### カスタムテンプレートの作成

```bash
# Xcode Project Template の場所

# システムテンプレート:
# /Applications/Xcode.app/Contents/Developer/Library/Xcode/Templates/Project Templates/

# カスタムテンプレート:
~/Library/Developer/Xcode/Templates/Project Templates/Custom/

# ディレクトリ作成
mkdir -p ~/Library/Developer/Xcode/Templates/Project\ Templates/Custom

cd ~/Library/Developer/Xcode/Templates/Project\ Templates/Custom
```

### テンプレート構造

```
MyApp Template.xctemplate/
├── TemplateInfo.plist              # テンプレート定義
├── TemplateIcon.png                # テンプレートアイコン
├── TemplateIcon@2x.png
├── MyApp/                          # プロジェクト構造
│   ├── App/
│   │   ├── ___PACKAGENAMEASIDENTIFIER___App.swift
│   │   └── AppDelegate.swift
│   ├── Features/
│   │   └── Home/
│   │       ├── Views/
│   │       │   └── HomeView.swift
│   │       ├── ViewModels/
│   │       │   └── HomeViewModel.swift
│   │       └── Models/
│   ├── Core/
│   │   ├── Networking/
│   │   │   ├── HTTPClient.swift
│   │   │   ├── APIClient.swift
│   │   │   └── Endpoint.swift
│   │   ├── Database/
│   │   └── Services/
│   ├── Common/
│   │   ├── Extensions/
│   │   ├── Utilities/
│   │   └── Constants/
│   ├── Resources/
│   │   └── Assets.xcassets/
│   └── Supporting Files/
│       └── Info.plist
├── MyAppTests/
│   └── MyAppTests.swift
├── MyAppUITests/
│   └── MyAppUITests.swift
├── .gitignore
├── .swiftlint.yml
├── .swiftformat
├── README.md
└── Package.swift
```

### TemplateInfo.plist

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Kind</key>
    <string>Xcode.Xcode3.ProjectTemplateUnitKind</string>

    <key>Identifier</key>
    <string>com.company.myAppTemplate</string>

    <key>Ancestors</key>
    <array>
        <string>com.apple.dt.unit.coreDataCocoaTouchApplication</string>
    </array>

    <key>Concrete</key>
    <true/>

    <key>Description</key>
    <string>Modern iOS App Template with MVVM Architecture</string>

    <key>SortOrder</key>
    <integer>1</integer>

    <key>Options</key>
    <array>
        <!-- プロジェクト名 -->
        <dict>
            <key>Identifier</key>
            <string>productName</string>
            <key>Required</key>
            <true/>
            <key>Name</key>
            <string>Product Name:</string>
            <key>Description</key>
            <string>The name of the product</string>
            <key>Type</key>
            <string>text</string>
            <key>Default</key>
            <string>MyApp</string>
        </dict>

        <!-- Bundle Identifier -->
        <dict>
            <key>Identifier</key>
            <string>bundleIdentifierPrefix</string>
            <key>Required</key>
            <true/>
            <key>Name</key>
            <string>Organization Identifier:</string>
            <key>Description</key>
            <string>Bundle identifier prefix (e.g., com.company)</string>
            <key>Type</key>
            <string>text</string>
            <key>Default</key>
            <string>com.company</string>
        </dict>

        <!-- UI Framework -->
        <dict>
            <key>Identifier</key>
            <string>uiFramework</string>
            <key>Required</key>
            <true/>
            <key>Name</key>
            <string>User Interface:</string>
            <key>Description</key>
            <string>The user interface framework</string>
            <key>Type</key>
            <string>popup</string>
            <key>Default</key>
            <string>SwiftUI</string>
            <key>Values</key>
            <array>
                <string>SwiftUI</string>
                <string>UIKit</string>
            </array>
        </dict>

        <!-- Architecture -->
        <dict>
            <key>Identifier</key>
            <string>architecture</string>
            <key>Required</key>
            <true/>
            <key>Name</key>
            <string>Architecture:</string>
            <key>Description</key>
            <string>Project architecture pattern</string>
            <key>Type</key>
            <string>popup</string>
            <key>Default</key>
            <string>MVVM</string>
            <key>Values</key>
            <array>
                <string>MVVM</string>
                <string>Clean Architecture</string>
                <string>VIPER</string>
            </array>
        </dict>

        <!-- Include Tests -->
        <dict>
            <key>Identifier</key>
            <string>includeUnitTests</string>
            <key>Required</key>
            <false/>
            <key>Name</key>
            <string>Include Unit Tests</string>
            <key>Description</key>
            <string>Whether to include unit test target</string>
            <key>Type</key>
            <string>checkbox</string>
            <key>Default</key>
            <string>true</string>
        </dict>

        <!-- Include UI Tests -->
        <dict>
            <key>Identifier</key>
            <string>includeUITests</string>
            <key>Required</key>
            <false/>
            <key>Name</key>
            <string>Include UI Tests</string>
            <key>Description</key>
            <string>Whether to include UI test target</string>
            <key>Type</key>
            <string>checkbox</string>
            <key>Default</key>
            <string>true</string>
        </dict>
    </array>

    <key>Definitions</key>
    <dict>
        <!-- App Entry Point -->
        <key>App/___PACKAGENAMEASIDENTIFIER___App.swift</key>
        <dict>
            <key>Path</key>
            <string>App/___PACKAGENAMEASIDENTIFIER___App.swift</string>
            <key>Group</key>
            <string>App</string>
            <key>TargetIndices</key>
            <array>
                <integer>0</integer>
            </array>
        </dict>

        <!-- SwiftLint Configuration -->
        <key>.swiftlint.yml</key>
        <dict>
            <key>Path</key>
            <string>.swiftlint.yml</string>
        </dict>
    </dict>
</dict>
</plist>
```

### テンプレートファイルの例

```swift
// ___PACKAGENAMEASIDENTIFIER___App.swift

import SwiftUI

@main
struct ___PACKAGENAMEASIDENTIFIER___App: App {
    // MARK: - Properties

    @StateObject private var appState = AppState()

    // MARK: - Body

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
        }
    }
}

// MARK: - App State

final class AppState: ObservableObject {
    @Published var isAuthenticated: Bool = false

    init() {
        // アプリ起動時の初期化処理
        setupServices()
    }

    private func setupServices() {
        // Firebase などのサービス初期化
    }
}
```

---

## セットアップ自動化スクリプト

### プロジェクト作成スクリプト

```bash
#!/bin/bash
# create-ios-project.sh - iOS プロジェクト自動作成スクリプト

set -e

# 色付き出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ヘルパー関数
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# パラメータ
PROJECT_NAME="${1:-MyApp}"
BUNDLE_ID="${2:-com.company.myapp}"
ORGANIZATION="${3:-Company}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  iOS Project Generator"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Project Name: $PROJECT_NAME"
echo "Bundle ID: $BUNDLE_ID"
echo "Organization: $ORGANIZATION"
echo ""

# 確認
read -p "続行しますか? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_error "中止しました"
    exit 1
fi

# ディレクトリ作成
print_info "プロジェクトディレクトリを作成中..."
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"
print_success "ディレクトリ作成完了"

# Git 初期化
print_info "Git リポジトリを初期化中..."
git init
git lfs install
print_success "Git 初期化完了"

# .gitignore 作成
print_info ".gitignore を作成中..."
cat > .gitignore << 'EOF'
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
.build/
Packages/
Package.resolved
*.xcworkspace

# CocoaPods
Pods/

# Carthage
Carthage/Build/
Carthage/Checkouts/

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
EOF
print_success ".gitignore 作成完了"

# SwiftLint 設定
print_info "SwiftLint 設定を作成中..."
cat > .swiftlint.yml << 'EOF'
excluded:
  - Pods
  - build
  - DerivedData
  - .build
  - Carthage

disabled_rules:
  - trailing_whitespace

opt_in_rules:
  - empty_count
  - closure_spacing
  - explicit_init
  - attributes
  - closure_end_indentation
  - contains_over_first_not_nil
  - empty_string
  - fatal_error_message
  - first_where
  - force_unwrapping
  - implicit_return
  - multiline_arguments
  - multiline_parameters
  - operator_usage_whitespace
  - redundant_nil_coalescing
  - sorted_imports
  - toggle_bool
  - trailing_closure
  - vertical_parameter_alignment_on_call

line_length:
  warning: 120
  error: 200
  ignores_comments: true

file_length:
  warning: 500
  error: 1000

function_body_length:
  warning: 50
  error: 100

type_body_length:
  warning: 300
  error: 500

cyclomatic_complexity:
  warning: 10
  error: 20

identifier_name:
  min_length:
    warning: 3
  max_length:
    warning: 40
    error: 50
  excluded:
    - id
    - x
    - y
    - z

custom_rules:
  no_print:
    name: "No Print"
    regex: "\\bprint\\("
    message: "Use Logger instead of print()"
    severity: warning

  no_force_try:
    name: "No Force Try"
    regex: "try!"
    message: "Avoid using try!"
    severity: error
EOF
print_success "SwiftLint 設定作成完了"

# SwiftFormat 設定
print_info "SwiftFormat 設定を作成中..."
cat > .swiftformat << 'EOF'
--swiftversion 5.9
--indent 4
--indentcase false
--indentstrings false
--maxwidth 120
--wraparguments before-first
--wrapcollections before-first
--wrapparameters before-first
--wrapternary before-operators
--trimwhitespace always
--commas inline
--closingparen same-line
--elseposition same-line
--guardelse same-line
--importgrouping testable-bottom
--organizetypes class,struct,enum,extension
--patternlet inline
--self remove
--stripunusedargs closure-only
--enable isEmpty
--enable sortedImports
--enable redundantReturn
--enable redundantSelf
--disable andOperator
EOF
print_success "SwiftFormat 設定作成完了"

# README.md 作成
print_info "README.md を作成中..."
cat > README.md << EOF
# $PROJECT_NAME

## 概要

[プロジェクトの説明をここに記載]

## 要件

- Xcode 15.0+
- iOS 15.0+
- Swift 5.9+

## セットアップ

\`\`\`bash
# 依存関係のインストール
./scripts/setup.sh

# プロジェクトを開く
open $PROJECT_NAME.xcodeproj
\`\`\`

## アーキテクチャ

- **UI Framework**: SwiftUI
- **Architecture**: MVVM
- **Dependency Injection**: Manual
- **Navigation**: Coordinator Pattern

## フォルダ構成

\`\`\`
$PROJECT_NAME/
├── App/                    # App Entry Point
├── Features/               # 機能ごとの実装
├── Core/                   # 共通コンポーネント
├── Common/                 # ユーティリティ
├── Resources/              # リソースファイル
└── Supporting Files/       # 設定ファイル
\`\`\`

## 依存関係

### Swift Package Manager

- [Alamofire](https://github.com/Alamofire/Alamofire) - ネットワーキング
- [Kingfisher](https://github.com/onevcat/Kingfisher) - 画像読み込み
- [Firebase](https://github.com/firebase/firebase-ios-sdk) - Analytics, Crashlytics

## ビルドと実行

\`\`\`bash
# Debug ビルド
xcodebuild -scheme $PROJECT_NAME -configuration Debug build

# Release ビルド
xcodebuild -scheme $PROJECT_NAME -configuration Release build

# テスト実行
xcodebuild test -scheme $PROJECT_NAME -destination 'platform=iOS Simulator,name=iPhone 15 Pro'
\`\`\`

## CI/CD

GitHub Actions を使用した CI/CD パイプライン

- **Lint**: SwiftLint による静的解析
- **Test**: Unit Tests + UI Tests
- **Build**: Debug/Release ビルド確認
- **Deploy**: TestFlight への自動デプロイ

## コントリビューション

[CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。

## ライセンス

[LICENSE](LICENSE) を参照してください。

## 作成者

$ORGANIZATION

## 変更履歴

### 1.0.0 (2024-XX-XX)

- 初回リリース
EOF
print_success "README.md 作成完了"

# CONTRIBUTING.md 作成
print_info "CONTRIBUTING.md を作成中..."
cat > CONTRIBUTING.md << 'EOF'
# Contributing Guide

## 開発環境のセットアップ

1. リポジトリをクローン
2. `./scripts/setup.sh` を実行
3. Xcode でプロジェクトを開く

## ブランチ戦略

- `main`: 本番環境
- `develop`: 開発環境
- `feature/*`: 新機能
- `bugfix/*`: バグ修正
- `hotfix/*`: 緊急修正

## コミットメッセージ規約

Conventional Commits に従う:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: 新機能
- `fix`: バグ修正
- `docs`: ドキュメント
- `style`: コードスタイル
- `refactor`: リファクタリング
- `test`: テスト
- `chore`: ビルド・設定

Example:
```
feat(auth): add login functionality

- Add login screen
- Implement authentication service
- Add unit tests

Closes #123
```

## コードレビュー

1. Pull Request を作成
2. CI が通ることを確認
3. レビュアーを指定
4. 承認後にマージ

## テスト

- Unit Tests: 必須
- UI Tests: 重要な機能のみ
- カバレッジ: 80% 以上を目標

## スタイルガイド

- SwiftLint に従う
- SwiftFormat で自動整形
- コメントは英語または日本語

## 質問・提案

Issue を作成してください。
EOF
print_success "CONTRIBUTING.md 作成完了"

# scripts ディレクトリ作成
print_info "スクリプトディレクトリを作成中..."
mkdir -p scripts

# setup.sh 作成
cat > scripts/setup.sh << 'SETUPSCRIPT'
#!/bin/bash
set -e

echo "Setting up development environment..."

# Homebrew チェック
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# 開発ツールのインストール
echo "Installing development tools..."
brew install swiftlint
brew install swiftformat
brew install fastlane

# Git hooks のセットアップ
echo "Setting up Git hooks..."
cat > .git/hooks/pre-commit << 'HOOK'
#!/bin/bash
if which swiftlint >/dev/null; then
    swiftlint --strict
else
    echo "warning: SwiftLint not installed"
fi
HOOK

chmod +x .git/hooks/pre-commit

echo "✓ Setup complete!"
SETUPSCRIPT

chmod +x scripts/setup.sh
print_success "セットアップスクリプト作成完了"

# フォルダ構成を作成
print_info "プロジェクトフォルダ構成を作成中..."
mkdir -p "$PROJECT_NAME"/{App,Features/Home/{Views,ViewModels,Models},Core/{Networking,Database,Services},Common/{Extensions,Utilities,Constants},Resources/Assets.xcassets,"Supporting Files"}
mkdir -p "${PROJECT_NAME}Tests"
mkdir -p "${PROJECT_NAME}UITests"
print_success "フォルダ構成作成完了"

# 初回コミット
print_info "初回コミットを作成中..."
git add .
git commit -m "feat(init): initial project setup

- Setup project structure
- Add SwiftLint and SwiftFormat configuration
- Add README and CONTRIBUTING guide
- Add setup scripts"
print_success "初回コミット完了"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_success "プロジェクト作成完了!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
print_info "次のステップ:"
echo "  1. Xcode でプロジェクトを作成"
echo "  2. ./scripts/setup.sh を実行"
echo "  3. Happy coding! 🎉"
echo ""
```

### 開発環境セットアップスクリプト

```bash
#!/bin/bash
# setup-dev-environment.sh - 開発環境一括セットアップ

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  iOS Development Environment Setup${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Homebrew のインストール確認
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo -e "${GREEN}✓${NC} Homebrew installed"
else
    echo -e "${GREEN}✓${NC} Homebrew already installed"
fi

# 必須ツールのインストール
echo ""
echo "Installing development tools..."

# Git
if ! command -v git &> /dev/null; then
    brew install git
    echo -e "${GREEN}✓${NC} Git installed"
else
    echo -e "${GREEN}✓${NC} Git already installed"
fi

brew install git-lfs
git lfs install
echo -e "${GREEN}✓${NC} Git LFS configured"

# iOS 開発ツール
tools=(
    "swiftlint"
    "swiftformat"
    "fastlane"
    "cocoapods"
    "carthage"
    "xcodegen"
)

for tool in "${tools[@]}"; do
    if ! command -v $tool &> /dev/null; then
        brew install $tool
        echo -e "${GREEN}✓${NC} $tool installed"
    else
        echo -e "${GREEN}✓${NC} $tool already installed"
    fi
done

# Ruby (Bundler 用)
if ! command -v ruby &> /dev/null; then
    brew install ruby
    echo -e "${GREEN}✓${NC} Ruby installed"
fi

# Bundler
if ! command -v bundle &> /dev/null; then
    gem install bundler
    echo -e "${GREEN}✓${NC} Bundler installed"
fi

# Node.js (オプション: Danger 用)
if ! command -v node &> /dev/null; then
    brew install node
    echo -e "${GREEN}✓${NC} Node.js installed"
fi

# Xcode Command Line Tools
if ! xcode-select -p &> /dev/null; then
    xcode-select --install
    echo -e "${GREEN}✓${NC} Xcode Command Line Tools installed"
fi

# CocoaPods のセットアップ
echo ""
echo "Setting up CocoaPods..."
pod setup
echo -e "${GREEN}✓${NC} CocoaPods configured"

# Fastlane のセットアップ
echo ""
echo "Setting up Fastlane..."
if [ ! -f "Gemfile" ]; then
    cat > Gemfile << 'EOF'
source "https://rubygems.org"

gem "fastlane", "~> 2.219"
gem "cocoapods", "~> 1.14"
EOF
    bundle install
    echo -e "${GREEN}✓${NC} Fastlane configured"
fi

# Git global config
echo ""
echo "Configuring Git..."
git config --global pull.rebase false
git config --global init.defaultBranch main
echo -e "${GREEN}✓${NC} Git configured"

# Xcode バージョン確認
echo ""
if command -v xcodebuild &> /dev/null; then
    echo "Xcode version:"
    xcodebuild -version
    echo -e "${GREEN}✓${NC} Xcode is installed"
else
    echo "⚠️  Xcode is not installed"
    echo "Please install Xcode from the App Store"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ Development environment setup complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
```

---

## ボイラープレートコード生成

### コード生成スクリプト

```bash
#!/bin/bash
# generate-feature.sh - Feature モジュール自動生成

FEATURE_NAME=$1

if [ -z "$FEATURE_NAME" ]; then
    echo "Usage: ./scripts/generate-feature.sh <FeatureName>"
    exit 1
fi

BASE_DIR="MyApp/Features/$FEATURE_NAME"

echo "Generating feature: $FEATURE_NAME"

# ディレクトリ作成
mkdir -p "$BASE_DIR"/{Views,ViewModels,Models,Services}

# View
cat > "$BASE_DIR/Views/${FEATURE_NAME}View.swift" << EOF
import SwiftUI

struct ${FEATURE_NAME}View: View {
    // MARK: - Properties

    @StateObject private var viewModel: ${FEATURE_NAME}ViewModel

    // MARK: - Initialization

    init(viewModel: ${FEATURE_NAME}ViewModel = ${FEATURE_NAME}ViewModel()) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    // MARK: - Body

    var body: some View {
        VStack {
            Text("${FEATURE_NAME}")
        }
        .navigationTitle("${FEATURE_NAME}")
        .onAppear {
            viewModel.onAppear()
        }
    }
}

// MARK: - Preview

#Preview {
    NavigationView {
        ${FEATURE_NAME}View()
    }
}
EOF

# ViewModel
cat > "$BASE_DIR/ViewModels/${FEATURE_NAME}ViewModel.swift" << EOF
import Foundation
import Combine

@MainActor
final class ${FEATURE_NAME}ViewModel: ObservableObject {
    // MARK: - Published Properties

    @Published var isLoading: Bool = false
    @Published var errorMessage: String?

    // MARK: - Private Properties

    private var cancellables = Set<AnyCancellable>()

    // MARK: - Initialization

    init() {
        setupBindings()
    }

    // MARK: - Public Methods

    func onAppear() {
        // 画面表示時の処理
    }

    func refresh() async {
        isLoading = true
        defer { isLoading = false }

        do {
            // データ取得処理
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    // MARK: - Private Methods

    private func setupBindings() {
        // Combine のバインディング設定
    }
}
EOF

# Model
cat > "$BASE_DIR/Models/${FEATURE_NAME}Model.swift" << EOF
import Foundation

struct ${FEATURE_NAME}Model: Codable, Identifiable {
    let id: String
    let name: String
    let createdAt: Date

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case createdAt = "created_at"
    }
}

extension ${FEATURE_NAME}Model {
    static var preview: ${FEATURE_NAME}Model {
        ${FEATURE_NAME}Model(
            id: "1",
            name: "Sample",
            createdAt: Date()
        )
    }
}
EOF

# Service (オプション)
cat > "$BASE_DIR/Services/${FEATURE_NAME}Service.swift" << EOF
import Foundation

protocol ${FEATURE_NAME}ServiceProtocol {
    func fetch() async throws -> [${FEATURE_NAME}Model]
}

final class ${FEATURE_NAME}Service: ${FEATURE_NAME}ServiceProtocol {
    private let apiClient: APIClient

    init(apiClient: APIClient) {
        self.apiClient = apiClient
    }

    func fetch() async throws -> [${FEATURE_NAME}Model] {
        // API リクエスト実装
        throw NSError(domain: "Not implemented", code: -1)
    }
}

// Mock for testing
final class Mock${FEATURE_NAME}Service: ${FEATURE_NAME}ServiceProtocol {
    func fetch() async throws -> [${FEATURE_NAME}Model] {
        [${FEATURE_NAME}Model.preview]
    }
}
EOF

echo "✓ Feature generated: $BASE_DIR"
echo ""
echo "Files created:"
echo "  - Views/${FEATURE_NAME}View.swift"
echo "  - ViewModels/${FEATURE_NAME}ViewModel.swift"
echo "  - Models/${FEATURE_NAME}Model.swift"
echo "  - Services/${FEATURE_NAME}Service.swift"
```

### ネットワーク層テンプレート

```swift
// Templates/HTTPClient.swift

import Foundation

// MARK: - HTTPClient Protocol

protocol HTTPClient {
    func request<T: Decodable>(
        endpoint: Endpoint,
        responseType: T.Type
    ) async throws -> T
}

// MARK: - URLSession Implementation

final class URLSessionHTTPClient: HTTPClient {
    private let session: URLSession
    private let decoder: JSONDecoder

    init(
        session: URLSession = .shared,
        decoder: JSONDecoder = .init()
    ) {
        self.session = session
        self.decoder = decoder

        // Decoder の設定
        decoder.dateDecodingStrategy = .iso8601
        decoder.keyDecodingStrategy = .convertFromSnakeCase
    }

    func request<T: Decodable>(
        endpoint: Endpoint,
        responseType: T.Type
    ) async throws -> T {
        let request = try endpoint.makeRequest()

        #if DEBUG
        Logger.network.debug("Request: \(request.url?.absoluteString ?? "")")
        #endif

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw NetworkError.invalidResponse
        }

        #if DEBUG
        Logger.network.debug("Response: \(httpResponse.statusCode)")
        #endif

        guard (200...299).contains(httpResponse.statusCode) else {
            throw NetworkError.statusCode(httpResponse.statusCode)
        }

        return try decoder.decode(T.self, from: data)
    }
}

// MARK: - Endpoint

struct Endpoint {
    let path: String
    let method: HTTPMethod
    let headers: [String: String]?
    let body: Data?
    let queryItems: [URLQueryItem]?

    init(
        path: String,
        method: HTTPMethod = .get,
        headers: [String: String]? = nil,
        body: Data? = nil,
        queryItems: [URLQueryItem]? = nil
    ) {
        self.path = path
        self.method = method
        self.headers = headers
        self.body = body
        self.queryItems = queryItems
    }

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
        request.httpBody = body

        // デフォルトヘッダー
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        // 認証トークン
        if let token = KeychainManager.shared.getToken() {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        // カスタムヘッダー
        headers?.forEach { key, value in
            request.setValue(value, forHTTPHeaderField: key)
        }

        return request
    }
}

// MARK: - HTTP Method

enum HTTPMethod: String {
    case get = "GET"
    case post = "POST"
    case put = "PUT"
    case delete = "DELETE"
    case patch = "PATCH"
}

// MARK: - Network Error

enum NetworkError: LocalizedError {
    case invalidURL
    case invalidResponse
    case statusCode(Int)
    case decodingFailed
    case unknown

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .invalidResponse:
            return "Invalid response from server"
        case .statusCode(let code):
            return "Server returned status code: \(code)"
        case .decodingFailed:
            return "Failed to decode response"
        case .unknown:
            return "Unknown error occurred"
        }
    }
}

// MARK: - API Constants

enum APIConstants {
    static var baseURL: String {
        #if DEBUG
        return "https://dev.api.example.com"
        #elseif STAGING
        return "https://staging.api.example.com"
        #else
        return "https://api.example.com"
        #endif
    }

    static let timeout: TimeInterval = 30
}

// MARK: - Logger

enum Logger {
    static let network = os.Logger(subsystem: "com.app", category: "Network")
}
```

---

## 環境設定の自動化

### .env ファイル管理

```bash
# .env.template - 環境変数テンプレート

# API Configuration
API_BASE_URL=https://api.example.com
API_KEY=your_api_key_here
API_SECRET=your_api_secret_here

# Firebase
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_API_KEY=your-firebase-api-key

# Analytics
MIXPANEL_TOKEN=your-mixpanel-token
AMPLITUDE_API_KEY=your-amplitude-key

# Feature Flags
ENABLE_ANALYTICS=true
ENABLE_CRASH_REPORTING=true
ENABLE_DEBUG_MENU=false

# Other
LOG_LEVEL=info
```

### 環境変数読み込みスクリプト

```bash
#!/bin/bash
# load-env.sh - .env ファイルを読み込み Xcode 用に変換

set -e

ENV_FILE="${1:-.env}"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found"
    exit 1
fi

# .env を読み込み Config.xcconfig に変換
cat > Config/Generated.xcconfig << EOF
// Auto-generated from $ENV_FILE
// Do not edit manually

EOF

while IFS='=' read -r key value; do
    # コメント行とスキップ
    [[ $key =~ ^#.*$ ]] && continue
    # 空行をスキップ
    [[ -z $key ]] && continue

    # xcconfig 形式で出力
    echo "$key = $value" >> Config/Generated.xcconfig
done < "$ENV_FILE"

echo "✓ Generated Config/Generated.xcconfig from $ENV_FILE"
```

### Swift での環境変数アクセス

```swift
// Environment.swift

import Foundation

enum Environment {
    // MARK: - API Configuration

    static var apiBaseURL: String {
        bundleValue(for: "API_BASE_URL") ?? "https://api.example.com"
    }

    static var apiKey: String {
        bundleValue(for: "API_KEY") ?? ""
    }

    static var apiSecret: String {
        bundleValue(for: "API_SECRET") ?? ""
    }

    // MARK: - Firebase

    static var firebaseProjectID: String {
        bundleValue(for: "FIREBASE_PROJECT_ID") ?? ""
    }

    static var firebaseAPIKey: String {
        bundleValue(for: "FIREBASE_API_KEY") ?? ""
    }

    // MARK: - Feature Flags

    static var isAnalyticsEnabled: Bool {
        bundleValue(for: "ENABLE_ANALYTICS") == "true"
    }

    static var isCrashReportingEnabled: Bool {
        bundleValue(for: "ENABLE_CRASH_REPORTING") == "true"
    }

    static var isDebugMenuEnabled: Bool {
        #if DEBUG
        return true
        #else
        return bundleValue(for: "ENABLE_DEBUG_MENU") == "true"
        #endif
    }

    // MARK: - Logging

    static var logLevel: LogLevel {
        let level = bundleValue(for: "LOG_LEVEL") ?? "info"
        return LogLevel(rawValue: level) ?? .info
    }

    // MARK: - Helpers

    private static func bundleValue(for key: String) -> String? {
        Bundle.main.infoDictionary?[key] as? String
    }
}

enum LogLevel: String {
    case verbose
    case debug
    case info
    case warning
    case error
}
```

---

## Feature Flags設定

### Feature Flag Manager

```swift
// FeatureFlagManager.swift

import Foundation

// MARK: - Feature Flag Protocol

protocol FeatureFlagProvider {
    func isEnabled(_ feature: FeatureFlag) -> Bool
    func getValue<T>(_ feature: FeatureFlag) -> T?
}

// MARK: - Feature Flags

enum FeatureFlag: String {
    // UI Features
    case newHomescreenDesign = "new_homescreen_design"
    case darkModeSupport = "dark_mode_support"
    case swiftUIRewrite = "swiftui_rewrite"

    // Business Features
    case premiumSubscription = "premium_subscription"
    case socialSharing = "social_sharing"
    case offlineMode = "offline_mode"

    // Experimental
    case betaFeature1 = "beta_feature_1"
    case betaFeature2 = "beta_feature_2"

    // Debug
    case debugMenu = "debug_menu"
    case verboseLogging = "verbose_logging"
}

// MARK: - Local Feature Flags

final class LocalFeatureFlagProvider: FeatureFlagProvider {
    private var flags: [String: Any] = [:]

    init() {
        loadDefaultFlags()
    }

    func isEnabled(_ feature: FeatureFlag) -> Bool {
        flags[feature.rawValue] as? Bool ?? false
    }

    func getValue<T>(_ feature: FeatureFlag) -> T? {
        flags[feature.rawValue] as? T
    }

    func setFlag(_ feature: FeatureFlag, value: Any) {
        flags[feature.rawValue] = value
        UserDefaults.standard.set(value, forKey: "feature_flag_\(feature.rawValue)")
    }

    private func loadDefaultFlags() {
        #if DEBUG
        flags[FeatureFlag.debugMenu.rawValue] = true
        flags[FeatureFlag.verboseLogging.rawValue] = true
        #endif

        // UserDefaults から読み込み
        FeatureFlag.allCases.forEach { feature in
            if let value = UserDefaults.standard.object(forKey: "feature_flag_\(feature.rawValue)") {
                flags[feature.rawValue] = value
            }
        }
    }
}

extension FeatureFlag: CaseIterable {}

// MARK: - Remote Feature Flags (Firebase Remote Config)

import FirebaseRemoteConfig

final class RemoteFeatureFlagProvider: FeatureFlagProvider {
    private let remoteConfig: RemoteConfig

    init() {
        remoteConfig = RemoteConfig.remoteConfig()

        let settings = RemoteConfigSettings()
        settings.minimumFetchInterval = 3600 // 1時間

        #if DEBUG
        settings.minimumFetchInterval = 0 // Debug では即座に取得
        #endif

        remoteConfig.configSettings = settings

        // デフォルト値を設定
        let defaults: [String: NSObject] = [
            FeatureFlag.newHomescreenDesign.rawValue: false as NSObject,
            FeatureFlag.premiumSubscription.rawValue: false as NSObject,
            FeatureFlag.darkModeSupport.rawValue: true as NSObject,
        ]
        remoteConfig.setDefaults(defaults)
    }

    func fetch() async throws {
        try await remoteConfig.fetch()
        try await remoteConfig.activate()
    }

    func isEnabled(_ feature: FeatureFlag) -> Bool {
        remoteConfig.configValue(forKey: feature.rawValue).boolValue
    }

    func getValue<T>(_ feature: FeatureFlag) -> T? {
        let value = remoteConfig.configValue(forKey: feature.rawValue)

        if T.self == String.self {
            return value.stringValue as? T
        } else if T.self == Int.self {
            return value.numberValue.intValue as? T
        } else if T.self == Double.self {
            return value.numberValue.doubleValue as? T
        } else if T.self == Bool.self {
            return value.boolValue as? T
        }

        return nil
    }
}

// MARK: - Feature Flag Manager

final class FeatureFlagManager {
    static let shared = FeatureFlagManager()

    private var provider: FeatureFlagProvider

    private init() {
        #if DEBUG
        provider = LocalFeatureFlagProvider()
        #else
        provider = RemoteFeatureFlagProvider()
        #endif
    }

    func isEnabled(_ feature: FeatureFlag) -> Bool {
        provider.isEnabled(feature)
    }

    func getValue<T>(_ feature: FeatureFlag) -> T? {
        provider.getValue(feature)
    }
}

// MARK: - Usage

extension View {
    func featureFlag(_ feature: FeatureFlag, @ViewBuilder content: () -> some View) -> some View {
        if FeatureFlagManager.shared.isEnabled(feature) {
            content()
        } else {
            EmptyView()
        }
    }
}

// 使用例
struct HomeView: View {
    var body: some View {
        VStack {
            if FeatureFlagManager.shared.isEnabled(.newHomescreenDesign) {
                NewHomescreen()
            } else {
                LegacyHomescreen()
            }
        }
    }
}
```

---

(文字数制限のため、以降は templates ディレクトリと scripts の実ファイルを作成します)
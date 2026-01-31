# Development Environment 完全ガイド

## 目次
1. [開発環境セットアップ](#開発環境セットアップ)
2. [Xcode設定](#xcode設定)
3. [開発ツール](#開発ツール)
4. [コード品質ツール](#コード品質ツール)
5. [デバッグツール](#デバッグツール)
6. [チーム開発環境](#チーム開発環境)
7. [パフォーマンス計測](#パフォーマンス計測)
8. [トラブルシューティング](#トラブルシューティング)

---

## 開発環境セットアップ

### 必須ツールのインストール

```bash
#!/bin/bash
# setup-dev-environment.sh - 開発環境セットアップスクリプト

set -e

echo "🚀 Setting up iOS development environment..."

# Homebrewのインストール確認
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# 必須ツールのインストール
echo "📦 Installing development tools..."

# Git
brew install git
brew install git-lfs

# iOS開発ツール
brew install --cask xcodes
brew install xcodegen
brew install swiftlint
brew install swiftformat

# パッケージマネージャー
brew install cocoapods
brew install carthage

# デバッグ・解析ツール
brew install --cask proxyman
brew install --cask charles

# デザインツール
brew install --cask figma
brew install --cask sketch

# CI/CD
brew install fastlane

# その他便利ツール
brew install jq
brew install tree
brew install gh  # GitHub CLI

echo "✅ All tools installed successfully!"

# Xcodeのインストール確認
echo "📱 Checking Xcode installation..."
if ! command -v xcodebuild &> /dev/null; then
    echo "⚠️  Xcode is not installed. Please install from App Store or xcodes."
    echo "Run: xcodes install --latest"
else
    xcodebuild -version
fi

# CocoaPodsのセットアップ
echo "🔧 Setting up CocoaPods..."
pod setup

# Fastlaneのセットアップ
echo "🚄 Setting up Fastlane..."
fastlane init

echo "🎉 Development environment setup complete!"
```

### プロジェクト初期設定スクリプト

```bash
#!/bin/bash
# setup-project.sh - プロジェクト初期設定スクリプト

set -e

PROJECT_NAME="${1:-MyApp}"

echo "🏗️  Setting up project: $PROJECT_NAME"

# ディレクトリ作成
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Git初期化
echo "📂 Initializing Git repository..."
git init
git lfs install

# .gitignoreの作成
curl -o .gitignore https://raw.githubusercontent.com/github/gitignore/main/Swift.gitignore

# SwiftLint設定
cat > .swiftlint.yml << 'EOF'
excluded:
  - Pods
  - build
  - DerivedData
  - .build

disabled_rules:
  - trailing_whitespace

opt_in_rules:
  - empty_count
  - closure_spacing
  - explicit_init

line_length: 120

identifier_name:
  min_length: 3
  excluded:
    - id
    - x
    - y
EOF

# SwiftFormat設定
cat > .swiftformat << 'EOF'
--swiftversion 5.9
--indent 4
--maxwidth 120
--wraparguments before-first
--wrapcollections before-first
--closingparen same-line
EOF

# README作成
cat > README.md << EOF
# $PROJECT_NAME

## Requirements
- Xcode 15.0+
- iOS 15.0+
- Swift 5.9+

## Setup
\`\`\`bash
# Install dependencies
./scripts/setup.sh

# Build and run
open $PROJECT_NAME.xcodeproj
\`\`\`

## Architecture
- MVVM + Clean Architecture
- SwiftUI

## Dependencies
- [List dependencies here]

## License
[Your License]
EOF

# スクリプトディレクトリ
mkdir -p scripts

# Git hook セットアップスクリプト
cat > scripts/setup-hooks.sh << 'EOF'
#!/bin/bash
# Pre-commit hook

cat > .git/hooks/pre-commit << 'HOOK'
#!/bin/bash
if which swiftlint >/dev/null; then
    swiftlint --strict
else
    echo "warning: SwiftLint not installed"
fi
HOOK

chmod +x .git/hooks/pre-commit
EOF

chmod +x scripts/setup-hooks.sh

echo "✅ Project setup complete!"
echo "Next steps:"
echo "1. Create Xcode project"
echo "2. Run: ./scripts/setup-hooks.sh"
echo "3. Happy coding! 🎉"
```

---

## Xcode設定

### おすすめXcode設定

```swift
/*
Xcode Preferences 推奨設定:

General:
- Issue Navigator Detail: Show all
- File Extensions: Show All
- Command-click on Code: Jumps to Definition

Navigation:
- Command-click on Code: Jumps to Definition
- Uses Focused Editor: Checked
- Navigation Style: Open in Place

Text Editing:
- Indentation:
  - Prefer indent using: Spaces
  - Tab width: 4 spaces
  - Indent width: 4 spaces
- Line wrapping: Wrap lines to editor width
- Show:
  - Line numbers: Yes
  - Code folding ribbon: Yes
  - Page guide at column: 120

Fonts & Colors:
- Theme: Xcode Default / Custom
- Console: SF Mono 12pt

Key Bindings:
- Command + R: Run
- Command + B: Build
- Command + U: Test
- Command + Shift + K: Clean Build Folder
- Command + Option + K: Clean
- Command + Control + Up/Down: Switch between .swift and Tests
*/
```

### Xcodeショートカット

```bash
# 必須ショートカット

## ナビゲーション
Cmd + Shift + O          # Open Quickly (ファイル検索)
Cmd + Control + J        # Jump to Definition
Cmd + Option + Left/Right # 前/次のファイル
Cmd + Shift + J          # Reveal in Navigator

## 編集
Cmd + /                  # コメントトグル
Cmd + ]                  # Indent Right
Cmd + [                  # Indent Left
Cmd + Option + [         # Move Line Up
Cmd + Option + ]         # Move Line Down
Control + I              # Re-Indent

## ビルド・実行
Cmd + B                  # Build
Cmd + R                  # Run
Cmd + .                  # Stop
Cmd + U                  # Test
Cmd + Shift + K          # Clean Build Folder

## デバッグ
Cmd + \                  # Toggle Breakpoint
Cmd + Y                  # Activate/Deactivate Breakpoints
F6                       # Step Over
F7                       # Step Into
F8                       # Continue

## その他
Cmd + Shift + Y          # Show/Hide Debug Area
Cmd + Option + Enter     # Show Assistant Editor
Cmd + Enter              # Hide Assistant Editor
Cmd + 0                  # Show/Hide Navigator
Cmd + Option + 0         # Show/Hide Inspector
```

---

## 開発ツール

### Xcode Extensions

```swift
// おすすめXcode Extensions

/*
1. SourceKitten
   - SwiftLintとSwiftFormatをXcode内で実行
   - インストール: App Store

2. SwiftFormat for Xcode
   - コードフォーマット
   - インストール: brew install --cask swiftformat-for-xcode

3. Injection for Xcode
   - Hot reload (SwiftUIのプレビュー的機能)
   - https://github.com/johnno1962/InjectionIII

4. QuickType
   - JSONからSwift structを自動生成
   - https://app.quicktype.io

5. Paste JSON as Code
   - JSONをSwift Codableに変換
   - Xcode Extension
*/
```

### コマンドラインツール

```bash
# xcodeproj - Xcodeプロジェクトの編集
gem install xcodeproj

# XcodeGen - project.ymlからXcodeprojを生成
brew install xcodegen

# project.yml の例
cat > project.yml << 'EOF'
name: MyApp

options:
  bundleIdPrefix: com.company
  deploymentTarget:
    iOS: "15.0"

targets:
  MyApp:
    type: application
    platform: iOS
    sources:
      - MyApp
    settings:
      PRODUCT_BUNDLE_IDENTIFIER: com.company.myapp
      INFOPLIST_FILE: MyApp/Info.plist
    dependencies:
      - package: Alamofire

packages:
  Alamofire:
    url: https://github.com/Alamofire/Alamofire
    from: 5.8.0
EOF

# プロジェクト生成
xcodegen generate

# xcode-install - 複数バージョンのXcode管理
gem install xcode-install

# xcpretty - xcodebuildの出力を整形
gem install xcpretty

# 使用例
xcodebuild test -scheme MyApp | xcpretty

# xclogparser - ビルドログの解析
brew install xclogparser

# ビルドログ解析
xclogparser parse --project MyApp.xcodeproj
```

---

## コード品質ツール

### SwiftLint詳細設定

```yaml
# .swiftlint.yml

# 除外パス
excluded:
  - Pods
  - Carthage
  - build
  - .build
  - DerivedData
  - */Generated/*

# 無効化するルール
disabled_rules:
  - trailing_whitespace
  - todo

# オプトインルール
opt_in_rules:
  - anyobject_protocol
  - array_init
  - attributes
  - closure_end_indentation
  - closure_spacing
  - collection_alignment
  - colon
  - comma
  - conditional_returns_on_newline
  - contains_over_first_not_nil
  - empty_count
  - empty_string
  - explicit_init
  - fallthrough
  - fatal_error_message
  - file_header
  - first_where
  - force_unwrapping
  - implicit_return
  - joined_default_parameter
  - let_var_whitespace
  - literal_expression_end_indentation
  - modifier_order
  - multiline_arguments
  - multiline_function_chains
  - multiline_parameters
  - operator_usage_whitespace
  - overridden_super_call
  - pattern_matching_keywords
  - prefer_self_type_over_type_of_self
  - redundant_nil_coalescing
  - redundant_type_annotation
  - single_test_class
  - sorted_first_last
  - sorted_imports
  - strict_fileprivate
  - toggle_bool
  - trailing_closure
  - unavailable_function
  - unneeded_parentheses_in_closure_argument
  - vertical_parameter_alignment_on_call
  - vertical_whitespace_closing_braces
  - vertical_whitespace_opening_braces
  - yoda_condition

# カスタム設定
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

# カスタムルール
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
```

### SwiftFormat詳細設定

```bash
# .swiftformat

# Version
--swiftversion 5.9

# Indentation
--indent 4
--indentcase false
--indentstrings false

# Wrapping
--maxwidth 120
--wraparguments before-first
--wrapcollections before-first
--wrapparameters before-first
--wrapternary before-operators

# Spacing
--trimwhitespace always
--commas inline
--decimalgrouping 3,4
--exponentgrouping disabled
--fractiongrouping disabled
--hexgrouping 4,8
--hexliteralcase uppercase
--octalgrouping 4,8
--semicolons inline

# Parentheses
--closingparen same-line
--elseposition same-line
--guardelse same-line

# Organization
--importgrouping testable-bottom
--organizetypes class,struct,enum,extension
--patternlet inline
--self remove
--stripunusedargs closure-only

# Enabled rules
--enable isEmpty
--enable sortedImports
--enable redundantReturn
--enable redundantSelf

# Disabled rules
--disable andOperator
--disable blankLinesAtStartOfScope
--disable blankLinesAtEndOfScope
```

---

## デバッグツール

### LLDB コマンド

```bash
# LLDB デバッグコマンド

## 基本コマンド
po variableName              # Print Object
p variableName               # Print (short)
v variableName               # Print all properties
expr variableName = newValue # 変数の書き換え

## ブレークポイント
b ViewController.swift:42    # 行指定
b viewDidLoad                # メソッド指定
br list                      # ブレークポイント一覧
br delete 1                  # 削除
br disable 1                 # 無効化

## 実行制御
c                            # Continue
n                            # Next (Step Over)
s                            # Step Into
finish                       # Step Out
thread return                # 現在の関数から即座にreturn

## オブジェクト検査
po self                      # 現在のインスタンス
po self.view                 # ビュー階層
po self.view.subviews        # サブビュー

## UIデバッグ
e (void)[CATransaction flush] # 画面を即座に更新
expr -l objc++ -O -- [[UIWindow keyWindow] recursiveDescription] # ビュー階層の表示

## メモリデバッグ
memory read --size 4 --format x --count 4 0x...  # メモリダンプ
```

### Custom LLDB Commands

```python
# ~/.lldbinit

# カスタムコマンド
command alias bd breakpoint disable
command alias be breakpoint enable
command alias bdel breakpoint delete

# SwiftUIのデバッグ
command script import ~/lldb_scripts/swiftui.py

# プリントの色付け
settings set use-color true

# フレームの自動表示
settings set frame-format "frame #${frame.index}: ${frame.pc}{ ${module.file.basename}{`${function.name-with-args}${function.pc-offset}}}\n"
```

### Instruments活用

```swift
// Instruments Template

/*
推奨テンプレート:

1. Time Profiler
   - CPU使用率の分析
   - ホットスポットの特定

2. Allocations
   - メモリ使用量
   - メモリリークの検出

3. Leaks
   - メモリリークの検出
   - 循環参照の特定

4. Network
   - ネットワーク通信の監視
   - データ転送量の分析

5. System Trace
   - スレッド使用状況
   - システムコールの分析

6. SwiftUI
   - ビュー更新の追跡
   - パフォーマンス問題の特定
*/

// Signpostを使用したカスタム計測

import os.signpost

class PerformanceMonitor {
    private static let log = OSLog(subsystem: "com.app.performance", category: "Network")

    static func measureNetworkRequest(_ name: String, _ block: () async throws -> Void) async rethrows {
        let signpostID = OSSignpostID(log: log)

        os_signpost(.begin, log: log, name: "Network Request", signpostID: signpostID, "%{public}s", name)

        try await block()

        os_signpost(.end, log: log, name: "Network Request", signpostID: signpostID)
    }
}

// 使用例
await PerformanceMonitor.measureNetworkRequest("Fetch Users") {
    try await apiClient.fetchUsers()
}
```

---

## チーム開発環境

### EditorConfig

```ini
# .editorconfig

root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true

[*.swift]
indent_style = space
indent_size = 4
max_line_length = 120

[*.{yml,yaml}]
indent_style = space
indent_size = 2

[*.md]
trim_trailing_whitespace = false

[Makefile]
indent_style = tab
```

### 統一された開発スクリプト

```makefile
# Makefile

.PHONY: setup build test clean

setup:
	@echo "🔧 Setting up project..."
	brew bundle
	pod install
	./scripts/setup-hooks.sh

build:
	@echo "🔨 Building project..."
	xcodebuild -workspace MyApp.xcworkspace \
		-scheme MyApp \
		-configuration Debug \
		build

test:
	@echo "🧪 Running tests..."
	xcodebuild test \
		-workspace MyApp.xcworkspace \
		-scheme MyApp \
		-destination 'platform=iOS Simulator,name=iPhone 15 Pro'

clean:
	@echo "🧹 Cleaning..."
	xcodebuild clean
	rm -rf build DerivedData

lint:
	@echo "🔍 Running SwiftLint..."
	swiftlint --strict

format:
	@echo "✨ Formatting code..."
	swiftformat .

help:
	@echo "Available commands:"
	@echo "  make setup  - Setup development environment"
	@echo "  make build  - Build project"
	@echo "  make test   - Run tests"
	@echo "  make clean  - Clean build artifacts"
	@echo "  make lint   - Run SwiftLint"
	@echo "  make format - Format code with SwiftFormat"
```

このガイドでは、iOS開発環境のセットアップから、Xcode設定、開発ツール、コード品質ツール、デバッグツール、チーム開発環境まで、効率的な開発に必要なすべての要素を網羅しました。適切な開発環境により、開発効率とコード品質を大きく向上させることができます。

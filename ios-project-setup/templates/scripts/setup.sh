#!/bin/bash
#
# setup.sh - プロジェクト環境セットアップスクリプト
# 新規メンバーのオンボーディングやクリーンセットアップに使用
#

set -e

# 色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# アイコン
CHECK="${GREEN}✓${NC}"
CROSS="${RED}✗${NC}"
INFO="${BLUE}ℹ${NC}"
WARN="${YELLOW}⚠${NC}"

# ヘルパー関数
print_header() {
    echo ""
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${PURPLE}  $1${NC}"
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_success() {
    echo -e "${CHECK} $1"
}

print_error() {
    echo -e "${CROSS} $1"
}

print_info() {
    echo -e "${INFO} $1"
}

print_warn() {
    echo -e "${WARN} $1"
}

# バージョンチェック
check_version() {
    local cmd=$1
    local required=$2
    local current=$3

    if [ -z "$current" ]; then
        print_error "$cmd not found"
        return 1
    fi

    print_success "$cmd version $current"
}

# Homebrew チェック
check_homebrew() {
    if command -v brew &> /dev/null; then
        local version=$(brew --version | head -n1)
        print_success "Homebrew installed: $version"
        return 0
    else
        print_error "Homebrew not found"
        return 1
    fi
}

# Xcode チェック
check_xcode() {
    if command -v xcodebuild &> /dev/null; then
        local version=$(xcodebuild -version | head -n1)
        print_success "Xcode installed: $version"

        # Command Line Tools チェック
        if xcode-select -p &> /dev/null; then
            print_success "Command Line Tools installed"
        else
            print_warn "Command Line Tools not found"
            print_info "Installing Command Line Tools..."
            xcode-select --install
        fi
        return 0
    else
        print_error "Xcode not found"
        return 1
    fi
}

# メイン処理
main() {
    print_header "iOS Project Setup"

    print_info "This script will set up your development environment."
    print_info "It will install necessary tools and dependencies."
    echo ""

    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Setup cancelled"
        exit 1
    fi

    # システム要件チェック
    print_header "Checking System Requirements"

    # macOS バージョン
    macos_version=$(sw_vers -productVersion)
    print_info "macOS version: $macos_version"

    # Homebrew
    if ! check_homebrew; then
        print_info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        # PATH 設定
        if [[ $(uname -m) == "arm64" ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi

        print_success "Homebrew installed"
    fi

    # Xcode
    check_xcode || {
        print_error "Please install Xcode from the App Store"
        exit 1
    }

    # 開発ツールのインストール
    print_header "Installing Development Tools"

    # Git
    if command -v git &> /dev/null; then
        check_version "Git" "" "$(git --version)"
    else
        brew install git
        print_success "Git installed"
    fi

    # Git LFS
    if command -v git-lfs &> /dev/null; then
        print_success "Git LFS installed"
    else
        brew install git-lfs
        git lfs install
        print_success "Git LFS installed and configured"
    fi

    # SwiftLint
    if command -v swiftlint &> /dev/null; then
        check_version "SwiftLint" "" "$(swiftlint version)"
    else
        brew install swiftlint
        print_success "SwiftLint installed"
    fi

    # SwiftFormat
    if command -v swiftformat &> /dev/null; then
        check_version "SwiftFormat" "" "$(swiftformat --version)"
    else
        brew install swiftformat
        print_success "SwiftFormat installed"
    fi

    # Fastlane
    print_header "Setting up Fastlane"

    if [ ! -f "Gemfile" ]; then
        cat > Gemfile << 'EOF'
source "https://rubygems.org"

gem "fastlane", "~> 2.219"
gem "cocoapods", "~> 1.14"
EOF
        print_success "Gemfile created"
    fi

    if command -v bundle &> /dev/null; then
        print_success "Bundler installed"
    else
        gem install bundler
        print_success "Bundler installed"
    fi

    bundle install
    print_success "Gems installed"

    # CocoaPods
    if [ -f "Podfile" ]; then
        print_header "Installing CocoaPods Dependencies"
        bundle exec pod install
        print_success "Pods installed"
    fi

    # Swift Package Manager
    if [ -f "Package.swift" ] || [ -f "*.xcodeproj/project.pbxproj" ]; then
        print_header "Resolving Swift Package Dependencies"
        xcodebuild -resolvePackageDependencies
        print_success "Swift packages resolved"
    fi

    # Git Hooks
    print_header "Setting up Git Hooks"

    if [ ! -d ".git/hooks" ]; then
        mkdir -p .git/hooks
    fi

    # Pre-commit hook
    cat > .git/hooks/pre-commit << 'HOOK'
#!/bin/bash
# Pre-commit hook

echo "Running pre-commit checks..."

# SwiftLint
if which swiftlint >/dev/null; then
    swiftlint --strict
    if [ $? -ne 0 ]; then
        echo "❌ SwiftLint failed"
        exit 1
    fi
    echo "✓ SwiftLint passed"
else
    echo "⚠️  SwiftLint not installed"
fi

# SwiftFormat (check only)
if which swiftformat >/dev/null; then
    swiftformat --lint .
    if [ $? -ne 0 ]; then
        echo "❌ SwiftFormat check failed"
        echo "Run: swiftformat . to fix formatting issues"
        exit 1
    fi
    echo "✓ SwiftFormat check passed"
else
    echo "⚠️  SwiftFormat not installed"
fi

echo "✓ All pre-commit checks passed"
HOOK

    chmod +x .git/hooks/pre-commit
    print_success "Git hooks configured"

    # 環境設定
    print_header "Configuring Environment"

    # .env.template から .env を作成
    if [ -f ".env.template" ] && [ ! -f ".env" ]; then
        cp .env.template .env
        print_success ".env file created from template"
        print_warn "Please update .env with your configuration"
    fi

    # 完了
    print_header "Setup Complete!"

    echo ""
    print_success "Your development environment is ready!"
    echo ""
    print_info "Next steps:"
    echo "  1. Update .env with your API keys and configuration"
    echo "  2. Open the .xcworkspace file (if using CocoaPods)"
    echo "  3. Build and run the project in Xcode"
    echo ""
    print_info "For more information, see README.md"
    echo ""
}

# スクリプト実行
main "$@"

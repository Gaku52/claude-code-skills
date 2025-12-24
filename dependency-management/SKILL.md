---
name: dependency-management
description: 依存関係管理ガイド。Swift Package Manager、CocoaPods、npm、pip等のパッケージマネージャー運用、バージョン管理、セキュリティアップデート、ライセンス管理など、依存関係の効率的な管理方法。
---

# Dependency Management Skill

## 📋 目次

1. [概要](#概要)
2. [Swift Package Manager](#swift-package-manager)
3. [CocoaPods](#cocoapods)
4. [npm / yarn / pnpm](#npm--yarn--pnpm)
5. [pip / Poetry](#pip--poetry)
6. [バージョン管理戦略](#バージョン管理戦略)
7. [セキュリティ管理](#セキュリティ管理)
8. [ライセンス管理](#ライセンス管理)
9. [依存関係の最適化](#依存関係の最適化)
10. [トラブルシューティング](#トラブルシューティング)

## 概要

プロジェクトの依存関係を効率的かつ安全に管理するためのベストプラクティスを提供します。

**対象:**
- iOS/Webエンジニア
- DevOpsエンジニア
- プロジェクトリーダー

**このSkillでできること:**
- パッケージマネージャーの適切な選択と運用
- 依存関係のバージョン管理
- セキュリティリスクの早期発見と対応
- ビルド時間の最適化

## Swift Package Manager

### 基本的な使い方

**Package.swiftの定義:**

```swift
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MyLibrary",
    platforms: [
        .iOS(.v15),
        .macOS(.v12)
    ],
    products: [
        .library(
            name: "MyLibrary",
            targets: ["MyLibrary"]
        ),
    ],
    dependencies: [
        // 依存パッケージの定義
        .package(url: "https://github.com/Alamofire/Alamofire.git", from: "5.8.0"),
        .package(url: "https://github.com/realm/realm-swift.git", exact: "10.45.0"),
        .package(url: "https://github.com/SDWebImage/SDWebImage.git", .upToNextMajor(from: "5.18.0")),
    ],
    targets: [
        .target(
            name: "MyLibrary",
            dependencies: [
                .product(name: "Alamofire", package: "Alamofire"),
                .product(name: "RealmSwift", package: "realm-swift"),
            ]
        ),
        .testTarget(
            name: "MyLibraryTests",
            dependencies: ["MyLibrary"]
        ),
    ]
)
```

### バージョン指定方法

```swift
// 特定バージョンを指定
.package(url: "...", exact: "1.0.0")

// 最小バージョン以上
.package(url: "...", from: "1.0.0")

// 範囲指定
.package(url: "...", "1.0.0"..<"2.0.0")

// 次のメジャーバージョンまで
.package(url: "...", .upToNextMajor(from: "1.0.0"))

// 次のマイナーバージョンまで
.package(url: "...", .upToNextMinor(from: "1.0.0"))

// ブランチ指定
.package(url: "...", branch: "develop")

// コミットハッシュ指定
.package(url: "...", revision: "abc123...")
```

### ローカルパッケージ開発

```swift
// Package.swift
dependencies: [
    // ローカルパスを指定（開発時）
    .package(path: "../MyLocalPackage"),
]
```

### XcodeでのSPM利用

```bash
# パッケージの追加
File → Add Package Dependencies → URL入力

# パッケージの更新
File → Packages → Update to Latest Package Versions

# パッケージの削除
プロジェクトナビゲーターから削除
```

## CocoaPods

### Podfileの基本

```ruby
# Podfile
platform :ios, '15.0'
use_frameworks!
inhibit_all_warnings! # 全ての警告を抑制（オプション）

target 'MyApp' do
  # 基本的なPod
  pod 'Alamofire', '~> 5.8'

  # 特定バージョン
  pod 'Realm', '10.45.0'

  # GitHubから直接
  pod 'MyPrivatePod', :git => 'https://github.com/user/MyPrivatePod.git', :tag => '1.0.0'

  # ローカルパス
  pod 'MyLocalPod', :path => '../MyLocalPod'

  # サブスペックの指定
  pod 'SDWebImage/WebP'

  # テストターゲット
  target 'MyAppTests' do
    inherit! :search_paths
    pod 'Quick'
    pod 'Nimble'
  end

  # UIテストターゲット
  target 'MyAppUITests' do
    inherit! :search_paths
  end
end

# ビルド設定のカスタマイズ
post_install do |installer|
  installer.pods_project.targets.each do |target|
    target.build_configurations.each do |config|
      # iOS Deployment Targetを統一
      config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '15.0'

      # Bitcodeを無効化
      config.build_settings['ENABLE_BITCODE'] = 'NO'

      # 警告を抑制
      config.build_settings['GCC_WARN_INHIBIT_ALL_WARNINGS'] = 'YES'
    end
  end
end
```

### 基本コマンド

```bash
# 初期化
pod init

# Podのインストール
pod install

# Podの更新
pod update

# 特定のPodのみ更新
pod update Alamofire

# キャッシュクリア
pod cache clean --all

# デバッグ情報表示
pod install --verbose

# Podfileのバリデーション
pod lib lint
```

### バージョン指定

```ruby
# 完全一致
pod 'Alamofire', '5.8.0'

# 以上
pod 'Alamofire', '>= 5.8.0'

# 未満
pod 'Alamofire', '< 6.0.0'

# ペシミスティックオペレーター（推奨）
pod 'Alamofire', '~> 5.8.0'  # >= 5.8.0 かつ < 5.9.0
pod 'Alamofire', '~> 5.8'    # >= 5.8 かつ < 6.0
```

### Podfile.lock の管理

```bash
# Podfile.lockは必ずGitにコミット
git add Podfile.lock
git commit -m "Update pod dependencies"

# チームメンバーは同じバージョンを使用
pod install  # updateではなくinstallを使用
```

## npm / yarn / pnpm

### package.jsonの基本

```json
{
  "name": "my-app",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "~1.6.0",
    "lodash": "4.17.21"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "typescript": "^5.3.0",
    "eslint": "^8.56.0",
    "prettier": "^3.1.0",
    "vite": "^5.0.0"
  },
  "peerDependencies": {
    "react": ">=18.0.0"
  },
  "optionalDependencies": {
    "fsevents": "^2.3.3"
  },
  "engines": {
    "node": ">=18.0.0",
    "npm": ">=9.0.0"
  },
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "lint": "eslint . --ext ts,tsx",
    "format": "prettier --write \"src/**/*.{ts,tsx}\""
  }
}
```

### バージョン指定

```json
{
  "dependencies": {
    "package1": "1.0.0",        // 完全一致
    "package2": "^1.0.0",       // >= 1.0.0 < 2.0.0（デフォルト）
    "package3": "~1.0.0",       // >= 1.0.0 < 1.1.0
    "package4": ">1.0.0",       // 1.0.0より大きい
    "package5": ">=1.0.0",      // 1.0.0以上
    "package6": "<2.0.0",       // 2.0.0未満
    "package7": "<=2.0.0",      // 2.0.0以下
    "package8": "1.0.0 - 2.0.0", // 範囲指定
    "package9": "*",            // 任意のバージョン（非推奨）
    "package10": "latest"       // 最新バージョン（非推奨）
  }
}
```

### npm コマンド

```bash
# 初期化
npm init

# パッケージのインストール
npm install              # package.jsonから全てインストール
npm install <package>    # 新しいパッケージを追加
npm install -D <package> # devDependenciesに追加
npm install -g <package> # グローバルインストール

# パッケージの更新
npm update               # 全てのパッケージを更新
npm update <package>     # 特定パッケージを更新
npm outdated             # 古いパッケージを確認

# パッケージの削除
npm uninstall <package>

# セキュリティ監査
npm audit
npm audit fix            # 自動修正
npm audit fix --force    # 破壊的変更も含めて修正

# キャッシュクリア
npm cache clean --force

# ロックファイルの再生成
rm package-lock.json
npm install
```

### yarn / pnpm

```bash
# yarn
yarn install
yarn add <package>
yarn add -D <package>
yarn upgrade
yarn remove <package>

# pnpm（高速・ディスク効率的）
pnpm install
pnpm add <package>
pnpm add -D <package>
pnpm update
pnpm remove <package>
```

### .npmrc / .yarnrc 設定

```bash
# .npmrc
registry=https://registry.npmjs.org/
save-exact=true                # 完全一致でバージョン保存
engine-strict=true             # engines指定を厳密にチェック
package-lock=true              # package-lock.jsonを生成
audit-level=high               # 高レベルの脆弱性のみ報告

# .yarnrc.yml (Yarn Berry)
nodeLinker: node-modules
yarnPath: .yarn/releases/yarn-3.6.4.cjs
```

## pip / Poetry

### requirements.txt

```txt
# requirements.txt

# 完全一致
Django==4.2.0

# 最小バージョン
requests>=2.31.0

# 範囲指定
numpy>=1.24.0,<2.0.0

# ペシミスティック
flask~=3.0.0

# Gitリポジトリから
git+https://github.com/user/repo.git@v1.0.0#egg=package

# ローカルパッケージ
-e ./local-package

# 他のrequirementsファイルを含める
-r requirements-dev.txt
```

### Poetryの使用

```toml
# pyproject.toml
[tool.poetry]
name = "my-app"
version = "1.0.0"
description = ""
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.11"
django = "^4.2.0"
requests = "^2.31.0"
numpy = "~1.24.0"

[tool.poetry.dev-dependencies]
pytest = "^7.4.0"
black = "^23.12.0"
mypy = "^1.7.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

```bash
# Poetry コマンド
poetry init                    # 初期化
poetry install                 # 依存関係インストール
poetry add <package>           # パッケージ追加
poetry add -D <package>        # 開発依存として追加
poetry update                  # 更新
poetry remove <package>        # 削除
poetry show                    # インストール済みパッケージ一覧
poetry show --outdated         # 古いパッケージ確認
poetry lock                    # poetry.lockを生成
poetry export -f requirements.txt --output requirements.txt  # requirements.txt生成
```

## バージョン管理戦略

### Semantic Versioning

```
MAJOR.MINOR.PATCH

例: 2.4.1

MAJOR (2): 破壊的変更
MINOR (4): 後方互換性のある機能追加
PATCH (1): 後方互換性のあるバグ修正
```

### バージョン固定戦略

**1. 完全固定（Exact Pinning）:**
```json
{
  "dependencies": {
    "react": "18.2.0"
  }
}
```
- メリット: 最も予測可能、再現性が高い
- デメリット: セキュリティアップデートが遅れる

**2. ペシミスティック固定（推奨）:**
```json
{
  "dependencies": {
    "react": "^18.2.0"  // npm/yarn
  }
}
```
```ruby
pod 'Alamofire', '~> 5.8.0'  # CocoaPods
```
- メリット: バグ修正とセキュリティアップデートを自動取得
- デメリット: 稀に非互換が発生

**3. 範囲指定:**
```json
{
  "dependencies": {
    "react": ">=18.0.0 <19.0.0"
  }
}
```

### 更新頻度の方針

```markdown
## 依存関係更新ポリシー

### セキュリティアップデート
- 頻度: 即時（Critical/High）、週次（Medium/Low）
- 対応: 自動化（Dependabot、Renovate）

### パッチバージョン
- 頻度: 週次
- 対応: 自動マージ（CI通過後）

### マイナーバージョン
- 頻度: 月次
- 対応: レビュー後マージ

### メジャーバージョン
- 頻度: 四半期ごと、または必要時
- 対応: 影響範囲調査、テスト、段階的移行
```

## セキュリティ管理

### 脆弱性スキャン

**npm audit:**
```bash
# 監査実行
npm audit

# 自動修正
npm audit fix

# レポート生成
npm audit --json > audit-report.json
```

**GitHub Dependabot:**
```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    reviewers:
      - "team-name"
    labels:
      - "dependencies"
    commit-message:
      prefix: "chore"

  - package-ecosystem: "swift"
    directory: "/"
    schedule:
      interval: "weekly"
```

**Snyk:**
```bash
# Snyk CLI
snyk test                      # 脆弱性テスト
snyk monitor                   # 継続的監視
snyk fix                       # 自動修正
```

### セキュリティアドバイザリー監視

```bash
# GitHub Security Advisories
# リポジトリの Settings → Security → Dependabot alerts を有効化

# npm
npm audit

# CocoaPods
pod outdated --verbose
```

## ライセンス管理

### ライセンススキャン

```bash
# npm-license
npm install -g npm-license
npm-license

# license-checker
npm install -g license-checker
license-checker --json > licenses.json
license-checker --onlyAllow="MIT;Apache-2.0;BSD-3-Clause"

# CocoaPods
pod install --verbose | grep "License:"
```

### ライセンス互換性マトリクス

```markdown
## 許可されるライセンス

### 商用利用可能
- MIT
- Apache 2.0
- BSD (2-Clause, 3-Clause)
- ISC

### 要検討
- LGPL (動的リンク可)
- MPL 2.0

### 禁止
- GPL (コピーレフト)
- AGPL
- 商用利用不可ライセンス
```

### ライセンス表示

```swift
// Settings.bundle/Acknowledgements.plist
// または
// AboutViewController でライセンス一覧を表示

class LicenseViewController: UIViewController {
    let licenses = [
        License(name: "Alamofire", license: "MIT", url: "https://..."),
        License(name: "Realm", license: "Apache 2.0", url: "https://..."),
    ]
}
```

## 依存関係の最適化

### 不要な依存の削除

```bash
# npm
npm prune                      # 未使用パッケージ削除
npx depcheck                   # 使われていない依存を検出

# CocoaPods
pod deintegrate                # Podsを完全削除
pod install                    # 再インストール
```

### バンドルサイズの削減

```bash
# webpack-bundle-analyzer
npm install --save-dev webpack-bundle-analyzer

# 使用
npx webpack-bundle-analyzer dist/stats.json

# Tree shaking（webpack）
# production modeで自動的に有効
npm run build

# 個別インポート（例: lodash）
# ❌ import _ from 'lodash'
# ✅ import debounce from 'lodash/debounce'
```

### ビルド時間の最適化

```swift
// XcodeでのSPM最適化
// Build Settings → Build Options
// Compilation Mode: Whole Module (Release)
// Optimization Level: Optimize for Speed (-O)

// パッケージのビルド結果をキャッシュ
// File → Workspace Settings → Derived Data → Default
```

```bash
# npm キャッシュ活用
npm ci  # package-lock.jsonから高速インストール

# pnpm（最も高速）
pnpm install
```

## トラブルシューティング

### 問題1: ビルドが失敗する

**SPM:**
```bash
# キャッシュクリア
rm -rf ~/Library/Caches/org.swift.swiftpm
rm -rf .build
xcodebuild clean

# パッケージ再取得
File → Packages → Reset Package Caches
File → Packages → Resolve Package Versions
```

**CocoaPods:**
```bash
# キャッシュクリア
pod cache clean --all
pod deintegrate
rm Podfile.lock
pod install
```

**npm:**
```bash
# node_modulesとロックファイル削除
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

### 問題2: バージョン競合

**SPM:**
```swift
// Package.resolvedを削除して再解決
rm Package.resolved
swift package resolve
```

**CocoaPods:**
```ruby
# Podfileで明示的に指定
pod 'ConflictingPod', '1.0.0'
```

**npm:**
```bash
# resolutionsで強制指定（yarn）
{
  "resolutions": {
    "package": "1.0.0"
  }
}

# overridesで強制指定（npm 8.3+）
{
  "overrides": {
    "package": "1.0.0"
  }
}
```

### 問題3: ローカル開発での依存関係

```bash
# npm link
cd ~/projects/my-package
npm link

cd ~/projects/my-app
npm link my-package

# yarn link
cd ~/projects/my-package
yarn link

cd ~/projects/my-app
yarn link my-package

# SPM local override
dependencies: [
    .package(path: "../MyLocalPackage")
]
```

### 問題4: Xcodeビルドエラー

```bash
# Derived Dataをクリア
rm -rf ~/Library/Developer/Xcode/DerivedData

# Xcodeキャッシュクリア
Product → Clean Build Folder (⌘⇧K)
```

---

**関連Skills:**
- [ios-project-setup](../ios-project-setup/SKILL.md) - プロジェクト初期設定
- [ci-cd-automation](../ci-cd-automation/SKILL.md) - CI/CD自動化
- [ios-security](../ios-security/SKILL.md) - セキュリティ管理
- [web-development](../web-development/SKILL.md) - Web開発での依存管理

**更新履歴:**
- 2025-12-24: 初版作成

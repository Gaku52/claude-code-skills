# Package Manager Best Practices

## 📋 目次

1. [概要](#概要)
2. [パッケージマネージャー選択ガイド](#パッケージマネージャー選択ガイド)
3. [npm/yarn/pnpm](#npmyarnpnpm)
4. [Swift Package Manager](#swift-package-manager)
5. [CocoaPods](#cocoapods)
6. [pip/Poetry](#pippoetry)
7. [共通ベストプラクティス](#共通ベストプラクティス)

## 概要

プロジェクトに適したパッケージマネージャーの選択と効率的な運用方法を解説します。

## パッケージマネージャー選択ガイド

### JavaScript/TypeScript

| マネージャー | 特徴 | 推奨用途 |
|------------|------|---------|
| **npm** | 標準、広く使われている | 一般的なプロジェクト |
| **yarn** | 高速、ワークスペース対応 | モノレポ、大規模プロジェクト |
| **pnpm** | 最速、ディスク効率的 | パフォーマンス重視 |

### iOS/Swift

| マネージャー | 特徴 | 推奨用途 |
|------------|------|---------|
| **Swift Package Manager** | 公式、Xcode統合 | 新規プロジェクト（推奨） |
| **CocoaPods** | 成熟、豊富なライブラリ | レガシープロジェクト |

### Python

| マネージャー | 特徴 | 推奨用途 |
|------------|------|---------|
| **pip** | 標準 | シンプルなプロジェクト |
| **Poetry** | モダン、依存関係解決 | 新規プロジェクト（推奨） |

## npm/yarn/pnpm

### 基本セットアップ

**package.json:**
```json
{
  "name": "my-app",
  "version": "1.0.0",
  "private": true,
  "engines": {
    "node": ">=18.0.0",
    "npm": ">=9.0.0"
  },
  "dependencies": {
    "react": "^18.2.0",
    "axios": "^1.6.0"
  },
  "devDependencies": {
    "typescript": "^5.3.0",
    "vite": "^5.0.0"
  }
}
```

### 基本コマンド比較

| 操作 | npm | yarn | pnpm |
|------|-----|------|------|
| インストール | `npm install` | `yarn` | `pnpm install` |
| 追加 | `npm install <pkg>` | `yarn add <pkg>` | `pnpm add <pkg>` |
| 削除 | `npm uninstall <pkg>` | `yarn remove <pkg>` | `pnpm remove <pkg>` |
| 更新 | `npm update` | `yarn upgrade` | `pnpm update` |
| 実行 | `npm run <script>` | `yarn <script>` | `pnpm <script>` |

### ベストプラクティス

**1. ロックファイルをコミット:**
```bash
git add package-lock.json  # npm
git add yarn.lock          # yarn
git add pnpm-lock.yaml     # pnpm
```

**2. CI/CDでは`npm ci`を使用:**
```bash
# package-lock.jsonから厳密にインストール
npm ci

# yarn
yarn install --frozen-lockfile

# pnpm
pnpm install --frozen-lockfile
```

**3. .npmrcで設定を統一:**
```bash
# .npmrc
save-exact=true           # バージョンを完全一致で保存
engine-strict=true        # engines指定を厳密にチェック
audit-level=high          # 高レベルの脆弱性のみ報告
```

## Swift Package Manager

### Package.swift定義

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
        .library(name: "MyLibrary", targets: ["MyLibrary"]),
    ],
    dependencies: [
        // 推奨: from（最小バージョン指定）
        .package(url: "https://github.com/Alamofire/Alamofire.git", from: "5.8.0"),

        // 本番環境: exact（完全固定）
        .package(url: "https://github.com/realm/realm-swift.git", exact: "10.45.0"),

        // 開発時: ローカルパス
        .package(path: "../MyLocalPackage"),
    ],
    targets: [
        .target(
            name: "MyLibrary",
            dependencies: [
                .product(name: "Alamofire", package: "Alamofire"),
            ]
        ),
    ]
)
```

### Xcodeでの操作

```bash
# パッケージ追加
File → Add Package Dependencies → URL入力

# 更新
File → Packages → Update to Latest Package Versions

# キャッシュクリア
File → Packages → Reset Package Caches
```

### トラブルシューティング

```bash
# キャッシュクリア
rm -rf ~/Library/Caches/org.swift.swiftpm
rm -rf .build

# パッケージ再解決
rm Package.resolved
swift package resolve
```

## CocoaPods

### Podfile基本設定

```ruby
platform :ios, '15.0'
use_frameworks!

target 'MyApp' do
  # 推奨: ペシミスティックオペレーター
  pod 'Alamofire', '~> 5.8'

  # 完全固定（本番推奨）
  pod 'Realm', '10.45.0'

  # GitHubから
  pod 'MyPod', :git => 'https://github.com/user/MyPod.git', :tag => '1.0.0'

  # ローカル開発
  pod 'MyLocalPod', :path => '../MyLocalPod'

  target 'MyAppTests' do
    inherit! :search_paths
    pod 'Quick'
    pod 'Nimble'
  end
end

# ビルド設定の統一
post_install do |installer|
  installer.pods_project.targets.each do |target|
    target.build_configurations.each do |config|
      config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '15.0'
      config.build_settings['ENABLE_BITCODE'] = 'NO'
    end
  end
end
```

### 基本コマンド

```bash
# 初期化
pod init

# インストール（初回・Podfile変更時）
pod install

# 更新（全体）
pod update

# 特定Podのみ更新
pod update Alamofire

# キャッシュクリア
pod cache clean --all
pod deintegrate
rm Podfile.lock
pod install
```

### ベストプラクティス

**1. Podfile.lockをコミット:**
```bash
git add Podfile.lock
git commit -m "Lock pod dependencies"
```

**2. .xcworkspaceを使用:**
```bash
# ❌ .xcodeprojを開かない
# ✅ .xcworkspaceを開く
open MyApp.xcworkspace
```

## pip/Poetry

### requirements.txt（pip）

```txt
# requirements.txt

# 完全固定（本番推奨）
Django==4.2.0
requests==2.31.0

# 最小バージョン（開発時）
numpy>=1.24.0

# 範囲指定
flask>=3.0.0,<4.0.0

# Git
git+https://github.com/user/repo.git@v1.0.0#egg=package
```

```bash
# インストール
pip install -r requirements.txt

# 依存関係の固定
pip freeze > requirements.txt

# アップグレード
pip install --upgrade <package>
```

### Poetry（推奨）

**pyproject.toml:**
```toml
[tool.poetry]
name = "my-app"
version = "1.0.0"

[tool.poetry.dependencies]
python = "^3.11"
django = "^4.2.0"
requests = "^2.31.0"

[tool.poetry.dev-dependencies]
pytest = "^7.4.0"
black = "^23.12.0"
```

**基本コマンド:**
```bash
# 初期化
poetry init

# インストール
poetry install

# パッケージ追加
poetry add django
poetry add -D pytest  # 開発依存

# 更新
poetry update

# requirements.txt生成
poetry export -f requirements.txt --output requirements.txt
```

## 共通ベストプラクティス

### 1. ロックファイルの管理

| マネージャー | ロックファイル | Git管理 |
|------------|--------------|---------|
| npm | package-lock.json | ✅ コミット |
| yarn | yarn.lock | ✅ コミット |
| pnpm | pnpm-lock.yaml | ✅ コミット |
| SPM | Package.resolved | ✅ コミット |
| CocoaPods | Podfile.lock | ✅ コミット |
| pip | - | - |
| Poetry | poetry.lock | ✅ コミット |

### 2. バージョン指定戦略

**開発環境:**
```bash
# 柔軟なバージョン（最新の改善を取得）
^1.0.0  # >= 1.0.0 < 2.0.0
~> 1.0  # >= 1.0 < 2.0
```

**本番環境:**
```bash
# 完全固定（予測可能性重視）
1.0.0
```

### 3. セキュリティチェック

```bash
# npm
npm audit
npm audit fix

# yarn
yarn audit

# pnpm
pnpm audit

# pip
pip-audit

# Poetry
poetry show --outdated
```

### 4. 依存関係の最適化

```bash
# 未使用パッケージの検出
npx depcheck  # npm/yarn/pnpm

# バンドルサイズ分析
npx webpack-bundle-analyzer dist/stats.json
```

### 5. CI/CD統合

```yaml
# GitHub Actions例
- name: Install dependencies
  run: |
    npm ci  # または yarn install --frozen-lockfile

- name: Audit
  run: npm audit --audit-level=high
```

## まとめ

### 推奨構成

**JavaScript/TypeScript:**
- 新規: **pnpm**（高速・効率的）
- 一般: **npm**（安定）
- モノレポ: **yarn workspaces**

**iOS/Swift:**
- 新規: **Swift Package Manager**
- レガシー: **CocoaPods**

**Python:**
- 新規: **Poetry**
- シンプル: **pip + requirements.txt**

### チェックリスト

- [ ] ロックファイルをGitにコミット
- [ ] CI/CDで厳密インストール（`npm ci`等）
- [ ] 定期的なセキュリティ監査
- [ ] バージョン固定戦略の明確化
- [ ] 依存関係の最適化

---

**関連ガイド:**
- [Version Management](./02-version-management.md)
- [Security & License](./03-security-license.md)

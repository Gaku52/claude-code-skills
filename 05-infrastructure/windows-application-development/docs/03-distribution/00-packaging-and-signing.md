# パッケージングと署名

> Electron と Tauri アプリケーションを各 OS 向けにパッケージングし、コード署名を適用してユーザーに安全に配布するためのインストーラー作成プロセスを体系的に学ぶ。

---

## この章で学ぶこと

1. **Electron（Forge / Builder）と Tauri bundler** のそれぞれのパッケージングツールを使いこなせるようになる
2. **コード署名**の仕組みを理解し、Windows（Authenticode）と macOS（Apple 署名）の署名を設定できるようになる
3. **各 OS 向けのインストーラー**（NSIS, MSI, DMG, AppImage, deb）を作成できるようになる

---

## 1. パッケージング概要

### 1.1 全体フロー

```
ソースコード
    │
    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  ビルド      │     │ パッケージング │     │  コード署名  │
│             │     │             │     │             │
│ - TypeScript│────→│ - バンドル   │────→│ - 証明書    │
│ - React     │     │ - asar 化   │     │ - タイムスタンプ│
│ - Rust      │     │ - リソース   │     │ - 公証      │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │ インストーラー作成 │
                                    │                 │
                                    │ Windows: NSIS/MSI│
                                    │ macOS: DMG      │
                                    │ Linux: AppImage │
                                    └─────────────────┘
```

### 1.2 OS 別インストーラー形式の比較

| 形式 | OS | 特徴 | ファイルサイズ |
|---|---|---|---|
| NSIS (.exe) | Windows | カスタムインストーラー。最も一般的 | 小 (圧縮効率高) |
| MSI | Windows | Windows Installer 標準。エンタープライズ向け | 中 |
| MSIX | Windows | モダン形式。ストア配布対応 | 中 |
| DMG | macOS | ディスクイメージ。ドラッグ&ドロップインストール | 中 |
| pkg | macOS | インストーラーパッケージ。ストア配布対応 | 中 |
| AppImage | Linux | 単一実行ファイル。インストール不要 | 大 |
| deb | Linux | Debian/Ubuntu パッケージ | 中 |
| rpm | Linux | Red Hat/Fedora パッケージ | 中 |
| snap | Linux | Snap パッケージ。自動更新対応 | 大 |

---

## 2. Electron パッケージング

### 2.1 Electron Forge vs Electron Builder

| 項目 | Electron Forge | Electron Builder |
|---|---|---|
| 運営 | Electron 公式 | コミュニティ |
| 設定方式 | `forge.config.ts` | `electron-builder.yml` |
| プラグイン | Maker / Publisher | 単一設定ファイル |
| 自動更新 | `@electron-forge/publisher-*` | `electron-updater` |
| NSIS カスタマイズ | 限定的 | 詳細な制御可能 |
| ユースケース | 公式推奨。シンプルな構成 | 複雑な要件。細かい制御 |

### コード例 1: Electron Forge の設定

```typescript
// forge.config.ts — Electron Forge 設定ファイル
import type { ForgeConfig } from '@electron-forge/shared-types'
import { MakerSquirrel } from '@electron-forge/maker-squirrel'
import { MakerZIP } from '@electron-forge/maker-zip'
import { MakerDMG } from '@electron-forge/maker-dmg'
import { MakerDeb } from '@electron-forge/maker-deb'
import { MakerRpm } from '@electron-forge/maker-rpm'
import { VitePlugin } from '@electron-forge/plugin-vite'
import { PublisherGithub } from '@electron-forge/publisher-github'

const config: ForgeConfig = {
  // パッケージング設定
  packagerConfig: {
    // アプリ名
    name: 'MyApp',
    // 実行ファイル名
    executableName: 'my-app',
    // アイコン（拡張子なしで指定。OS に合わせて自動選択）
    icon: './resources/icon',
    // asar アーカイブ化（ソースコードの保護）
    asar: true,
    // 不要ファイルの除外パターン
    ignore: [
      /\.git/,
      /\.vscode/,
      /src\//,       // ソースコードは含めない
      /node_modules\/.*\/test/,
    ],
    // macOS 用署名設定
    osxSign: {},
    osxNotarize: {
      appleId: process.env.APPLE_ID!,
      appleIdPassword: process.env.APPLE_PASSWORD!,
      teamId: process.env.APPLE_TEAM_ID!,
    },
    // Windows 用署名設定
    windowsSign: {
      certificateFile: process.env.WIN_CERT_FILE,
      certificatePassword: process.env.WIN_CERT_PASSWORD,
    },
  },

  // Maker 設定（インストーラー生成）
  makers: [
    // Windows: Squirrel インストーラー
    new MakerSquirrel({
      name: 'my-app',
      setupIcon: './resources/icon.ico',
    }),
    // macOS: DMG ディスクイメージ
    new MakerDMG({
      icon: './resources/icon.icns',
      format: 'ULFO',
    }),
    // macOS/Linux: ZIP（汎用）
    new MakerZIP({}, ['darwin', 'linux']),
    // Linux: deb パッケージ
    new MakerDeb({
      options: {
        maintainer: 'Dev Team',
        homepage: 'https://example.com',
        icon: './resources/icon.png',
        categories: ['Utility'],
      },
    }),
    // Linux: RPM パッケージ
    new MakerRpm({}),
  ],

  // Publisher 設定（配布先）
  publishers: [
    new PublisherGithub({
      repository: { owner: 'myorg', name: 'my-app' },
      prerelease: false,
    }),
  ],

  // Vite プラグイン
  plugins: [
    new VitePlugin({
      build: [
        { entry: 'src/main/index.ts', config: 'vite.main.config.ts' },
        { entry: 'src/preload/index.ts', config: 'vite.preload.config.ts' },
      ],
      renderer: [
        { name: 'main_window', config: 'vite.renderer.config.ts' },
      ],
    }),
  ],
}

export default config
```

### コード例 2: Electron Builder の設定

```yaml
# electron-builder.yml — Electron Builder 設定ファイル
appId: com.example.my-app
productName: My App
copyright: Copyright (c) 2025 My Company

# ビルド出力ディレクトリ
directories:
  output: release
  buildResources: resources

# 共通設定
asar: true
compression: maximum  # 圧縮レベル: store / normal / maximum

# ファイルフィルター（含めるファイルを明示）
files:
  - "out/**/*"
  - "package.json"
  - "!node_modules/**/*.{md,txt,map}"
  - "!node_modules/**/{test,tests,__tests__}/**"

# Windows 設定
win:
  target:
    - target: nsis
      arch: [x64, arm64]
    - target: msi
      arch: [x64]
  icon: resources/icon.ico
  # コード署名
  signingHashAlgorithms: [sha256]
  certificateFile: ${WIN_CERT_FILE}
  certificatePassword: ${WIN_CERT_PASSWORD}
  # タイムスタンプサーバー
  rfc3161TimeStampServer: http://timestamp.digicert.com

# NSIS インストーラー設定
nsis:
  oneClick: false           # ワンクリックインストールを無効化
  allowToChangeInstallationDirectory: true  # インストール先の変更を許可
  installerIcon: resources/icon.ico
  uninstallerIcon: resources/icon.ico
  installerHeaderIcon: resources/icon.ico
  createDesktopShortcut: true
  createStartMenuShortcut: true
  license: LICENSE.txt
  # 多言語対応
  language: 1041            # 日本語

# macOS 設定
mac:
  target:
    - target: dmg
      arch: [x64, arm64]
    - target: zip
      arch: [x64, arm64]
  icon: resources/icon.icns
  category: public.app-category.productivity
  hardenedRuntime: true
  gatekeeperAssess: false
  entitlements: build/entitlements.mac.plist
  entitlementsInherit: build/entitlements.mac.plist

# macOS 公証（Notarization）
afterSign: scripts/notarize.js

# DMG 設定
dmg:
  contents:
    - x: 410
      y: 150
      type: link
      path: /Applications
    - x: 130
      y: 150
      type: file

# Linux 設定
linux:
  target:
    - target: AppImage
      arch: [x64]
    - target: deb
      arch: [x64]
    - target: rpm
      arch: [x64]
  icon: resources/icons
  category: Utility
  maintainer: dev@example.com

# 自動更新設定（GitHub Releases）
publish:
  provider: github
  owner: myorg
  repo: my-app
  releaseType: release
```

---

## 3. Tauri パッケージング

### コード例 3: Tauri バンドラー設定

```json
// src-tauri/tauri.conf.json — バンドル設定
{
  "bundle": {
    "active": true,
    "targets": "all",
    "icon": [
      "icons/32x32.png",
      "icons/128x128.png",
      "icons/128x128@2x.png",
      "icons/icon.icns",
      "icons/icon.ico"
    ],
    "resources": [
      "assets/*"
    ],
    "copyright": "Copyright (c) 2025 My Company",
    "category": "Productivity",
    "shortDescription": "生産性向上ツール",
    "longDescription": "チームの生産性を向上させるためのオールインワンツール",

    "windows": {
      "certificateThumbprint": null,
      "digestAlgorithm": "sha256",
      "timestampUrl": "http://timestamp.digicert.com",
      "nsis": {
        "languages": ["Japanese", "English"],
        "displayLanguageSelector": true,
        "installerIcon": "icons/icon.ico",
        "headerImage": "icons/nsis-header.bmp",
        "sidebarImage": "icons/nsis-sidebar.bmp"
      },
      "wix": {
        "language": ["ja-JP", "en-US"]
      }
    },

    "macOS": {
      "entitlements": "Entitlements.plist",
      "signingIdentity": "-",
      "minimumSystemVersion": "10.15"
    },

    "linux": {
      "deb": {
        "depends": ["libwebkit2gtk-4.1-0", "libgtk-3-0"],
        "section": "utils"
      },
      "appimage": {
        "bundleMediaFramework": true
      }
    }
  }
}
```

```bash
# Tauri アプリのビルドとパッケージング
# 開発ビルド（デバッグモード）
cargo tauri build --debug

# リリースビルド（最適化あり）
cargo tauri build

# 特定ターゲットのみビルド
cargo tauri build --target x86_64-pc-windows-msvc
cargo tauri build --bundles nsis    # NSIS のみ
cargo tauri build --bundles msi     # MSI のみ
cargo tauri build --bundles dmg     # DMG のみ
cargo tauri build --bundles appimage # AppImage のみ
```

---

## 4. コード署名

### 4.1 署名の仕組み

```
+----------------------------------------------------------+
|                   コード署名のフロー                       |
+----------------------------------------------------------+
|                                                          |
|  開発者                                                   |
|  ┌──────────────────────────────────────────────────┐    |
|  │  1. 証明書の取得                                   │    |
|  │     CA (認証局) から Code Signing 証明書を購入      │    |
|  │                                                  │    |
|  │  2. バイナリへの署名                               │    |
|  │     秘密鍵でバイナリのハッシュに署名               │    |
|  │     ┌──────┐    ┌──────────┐    ┌─────────┐     │    |
|  │     │.exe  │ +  │秘密鍵    │ →  │署名済み │     │    |
|  │     │.dll  │    │(.pfx)   │    │.exe     │     │    |
|  │     └──────┘    └──────────┘    └─────────┘     │    |
|  │                                                  │    |
|  │  3. タイムスタンプの付与                            │    |
|  │     証明書の有効期限後も署名が有効になる            │    |
|  └──────────────────────────────────────────────────┘    |
|                                                          |
|  ユーザー                                                 |
|  ┌──────────────────────────────────────────────────┐    |
|  │  4. 署名の検証                                     │    |
|  │     OS が証明書チェーンを検証                      │    |
|  │     → SmartScreen / Gatekeeper の警告が消える     │    |
|  └──────────────────────────────────────────────────┘    |
+----------------------------------------------------------+
```

### 4.2 Windows (Authenticode) 署名

### コード例 4: Windows 署名の設定

```bash
# signtool.exe を使った手動署名（Windows SDK に含まれる）
# PFX ファイルの場合
signtool sign /f "certificate.pfx" \
  /p "パスワード" \
  /fd sha256 \
  /tr http://timestamp.digicert.com \
  /td sha256 \
  /d "My Application" \
  "path/to/app.exe"

# EV 証明書（ハードウェアトークン）の場合
signtool sign /n "Company Name" \
  /fd sha256 \
  /tr http://timestamp.digicert.com \
  /td sha256 \
  "path/to/app.exe"
```

```typescript
// Electron Builder での署名設定（環境変数で秘密情報を渡す）
// package.json の scripts
{
  "scripts": {
    "build:win": "cross-env CSC_LINK=./cert.pfx CSC_KEY_PASSWORD=$CERT_PASS electron-builder --win",
    "build:mac": "electron-builder --mac"
  }
}
```

### 4.3 macOS 署名と公証（Notarization）

```
macOS コード署名 + 公証のフロー:

  1. Apple Developer Program に登録（年間 $99）
  2. Developer ID Application 証明書を取得
  3. アプリにコード署名
  4. Apple に公証申請（Notarization）
  5. Staple（公証チケットをアプリに添付）

  ┌─────────────┐    ┌───────────────┐    ┌──────────┐
  │ コード署名   │───→│ Apple に送信   │───→│ Staple   │
  │ (codesign)  │    │ (notarytool)  │    │          │
  └─────────────┘    └───────────────┘    └──────────┘
                          ↓
                     Apple サーバーで
                     マルウェアスキャン
                     (数分～数十分)
```

### コード例 5: macOS 公証スクリプト

```javascript
// scripts/notarize.js — Electron Builder の afterSign フック
const { notarize } = require('@electron/notarize')

exports.default = async function notarizing(context) {
  const { electronPlatformName, appOutDir } = context

  // macOS 以外ではスキップ
  if (electronPlatformName !== 'darwin') {
    return
  }

  const appName = context.packager.appInfo.productFilename

  console.log(`公証を開始: ${appName}`)

  await notarize({
    // Apple ID（環境変数から取得）
    appleId: process.env.APPLE_ID,
    // App-specific password
    appleIdPassword: process.env.APPLE_APP_SPECIFIC_PASSWORD,
    // チーム ID
    teamId: process.env.APPLE_TEAM_ID,
    // アプリのパス
    appPath: `${appOutDir}/${appName}.app`,
  })

  console.log('公証が完了しました')
}
```

```xml
<!-- build/entitlements.mac.plist — macOS エンタイトルメント -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- ハードンドランタイム必須設定 -->
    <key>com.apple.security.cs.allow-jit</key>
    <true/>
    <!-- 署名なしの実行可能ファイルを許可（Electron 用） -->
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>
    <!-- dylib の読み込みを許可 -->
    <key>com.apple.security.cs.allow-dyld-environment-variables</key>
    <true/>
    <!-- ネットワークアクセス -->
    <key>com.apple.security.network.client</key>
    <true/>
    <!-- ファイルアクセス（ユーザー選択） -->
    <key>com.apple.security.files.user-selected.read-write</key>
    <true/>
</dict>
</plist>
```

---

## 5. 署名証明書の種類と費用

| 証明書種類 | 対象 OS | 年間費用目安 | SmartScreen 即時信頼 |
|---|---|---|---|
| OV (Organization Validation) | Windows | $200-400 | いいえ (実績蓄積が必要) |
| EV (Extended Validation) | Windows | $300-600 | はい |
| Apple Developer ID | macOS | $99 | はい (公証後) |
| 自己署名証明書 | 開発用 | 無料 | いいえ |

---

## 6. アンチパターン

### アンチパターン 1: 秘密鍵や証明書パスワードをリポジトリにコミットする

```yaml
# NG: ハードコードされた秘密情報
win:
  certificateFile: "./secrets/cert.pfx"         # リポジトリに含まれる
  certificatePassword: "my-secret-password-123"  # パスワードが平文
```

```yaml
# OK: 環境変数から秘密情報を参照
win:
  certificateFile: ${WIN_CERT_FILE}              # CI/CD の Secret から注入
  certificatePassword: ${WIN_CERT_PASSWORD}       # CI/CD の Secret から注入

# CI/CD (GitHub Actions) での設定例:
# secrets.WIN_CERT_FILE → Base64 エンコードされた PFX ファイル
# secrets.WIN_CERT_PASSWORD → 証明書のパスワード
```

### アンチパターン 2: 署名なしでアプリを配布する

```
署名なしの場合に表示される警告:

Windows:
┌──────────────────────────────────────────┐
│  Windows によって PC が保護されました       │
│                                          │
│  Windows SmartScreen は認識されないアプリの │
│  起動を停止しました。                      │
│                                          │
│  [実行しない]  [詳細情報 → 実行]           │
└──────────────────────────────────────────┘

macOS:
┌──────────────────────────────────────────┐
│  "MyApp" は開発元が未確認のため           │
│  開けません。                             │
│                                          │
│  [ゴミ箱に入れる]  [キャンセル]           │
└──────────────────────────────────────────┘

→ ユーザーの信頼を損ない、インストール率が大幅に低下する
→ 必ずコード署名を行うこと
```

---

## 7. FAQ

### Q1: EV 証明書と OV 証明書のどちらを選ぶべきか？

**A:** 初期段階では EV 証明書を推奨する。OV 証明書は SmartScreen での信頼を蓄積するのに数週間〜数ヶ月かかるが、EV 証明書は即座に SmartScreen の警告を回避できる。ただし EV 証明書はハードウェアトークン（USB キー）が必要であり、CI/CD での自動署名にはクラウド署名サービス（DigiCert KeyLocker, Azure Trusted Signing など）との組み合わせが必要になる。

### Q2: macOS の公証（Notarization）はどのくらい時間がかかるか？

**A:** 通常 1〜5 分で完了する。ただし Apple のサーバー負荷によっては 15 分以上かかる場合もある。CI/CD パイプラインでは公証完了まで待機するタイムアウトを十分に設定すること（最低 30 分推奨）。`xcrun notarytool submit --wait` コマンドで完了を待機できる。

### Q3: Linux アプリにはコード署名は必要か？

**A:** Linux にはWindows/macOS のような OS レベルのコード署名チェック機構がないため、技術的には不要である。ただし、GPG 署名でパッケージの完全性を証明したり、AppImage に署名を埋め込んだりすることはできる。配布チャネルに応じて（Snap Store は自動署名される等）検討すればよい。

---

## 8. まとめ

| トピック | キーポイント |
|---|---|
| Electron Forge | 公式推奨ツール。Maker/Publisher プラグインで拡張 |
| Electron Builder | 詳細な制御が可能。NSIS のカスタマイズが強力 |
| Tauri bundler | `cargo tauri build` で NSIS/MSI/DMG/AppImage を生成 |
| Windows 署名 | Authenticode。EV 証明書で SmartScreen を即時回避 |
| macOS 署名 | Developer ID + Notarization + Staple が必須 |
| 証明書管理 | 秘密鍵は CI/CD の Secret で管理。リポジトリにコミットしない |
| インストーラー | NSIS(Win) + DMG(Mac) + AppImage(Linux) が標準構成 |

---

## 次に読むべきガイド

- **[01-auto-update.md](./01-auto-update.md)** — 自動更新の実装（electron-updater / Tauri updater）
- **[02-store-distribution.md](./02-store-distribution.md)** — Microsoft Store / Mac App Store への配布

---

## 参考文献

1. Electron Forge, "Configuration", https://www.electronforge.io/configuration
2. Electron Builder, "Configuration", https://www.electron.build/configuration
3. Tauri, "Building", https://v2.tauri.app/distribute/
4. Microsoft, "Authenticode Code Signing", https://learn.microsoft.com/windows/win32/seccrypto/cryptography-tools
5. Apple, "Notarizing macOS Software", https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution

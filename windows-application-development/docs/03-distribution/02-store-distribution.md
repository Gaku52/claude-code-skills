# ストア配布 (Store Distribution)

> Microsoft Store (MSIX)、Mac App Store、GitHub Releases を活用し、CI/CD パイプラインで自動化されたアプリケーション配布を実現する技術を学ぶ。

## この章で学ぶこと

1. **MSIX パッケージングと Microsoft Store 配布** -- Windows アプリを MSIX 形式でパッケージし、Microsoft Store に公開するまでの全工程を理解する
2. **Mac App Store と GitHub Releases の配布戦略** -- macOS アプリのサンドボックス対応と、GitHub Releases を使った直接配布を使い分ける
3. **CI/CD による自動ビルド・署名・公開パイプライン** -- GitHub Actions を中心に、マルチプラットフォームのリリース自動化を構築する

---

## 1. 配布チャネルの全体像

```
+------------------------------------------------------------------+
|                    配布チャネルの選択マップ                          |
+------------------------------------------------------------------+
|                                                                  |
|  ソースコード (GitHub / GitLab)                                   |
|       |                                                          |
|       v                                                          |
|  CI/CD パイプライン                                               |
|       |                                                          |
|       +-----> Microsoft Store (MSIX)     [Windows ユーザー]       |
|       |          - 自動更新・サンドボックス                         |
|       |          - 企業配布(LOB)対応                               |
|       |                                                          |
|       +-----> Mac App Store (pkg/app)    [macOS ユーザー]         |
|       |          - サンドボックス必須                               |
|       |          - Notarization 必須                              |
|       |                                                          |
|       +-----> GitHub Releases            [開発者/パワーユーザー]   |
|       |          - 直接DL・自由な形式                              |
|       |          - electron-updater 連携                          |
|       |                                                          |
|       +-----> 自社サイト / CDN            [企業向け]              |
|                  - 完全制御                                       |
|                  - カスタムインストーラー                           |
|                                                                  |
+------------------------------------------------------------------+
```

### 配布チャネル比較表

| 項目 | Microsoft Store | Mac App Store | GitHub Releases | 自社サイト |
|------|----------------|---------------|-----------------|----------|
| 審査 | あり(1-3日) | あり(1-7日) | なし | なし |
| 手数料 | 15%(アプリ) / 12%(ゲーム) | 30% / 15%(小規模) | 無料 | 無料 |
| 自動更新 | OS 標準 | OS 標準 | 自前実装 | 自前実装 |
| サンドボックス | MSIX は制限あり | 必須 | なし | なし |
| 到達範囲 | Windows 10/11 ユーザー | macOS ユーザー | 技術者中心 | 任意 |
| パッケージ形式 | MSIX / MSIX Bundle | pkg / app (zip) | exe/msi/dmg/AppImage | 任意 |
| 企業配布 | LOB 対応 | MDM 対応 | 非対応 | 任意 |

---

## 2. Microsoft Store (MSIX) 配布

### 2.1 MSIX パッケージの構造

```
+-----------------------------------------------+
|  MSIX パッケージ (.msix)                        |
+-----------------------------------------------+
|                                               |
|  AppxManifest.xml    ← アプリ情報・権限宣言     |
|  Assets/                                      |
|    +-- Square150x150Logo.png                  |
|    +-- Square44x44Logo.png                    |
|    +-- StoreLogo.png                          |
|    +-- Wide310x150Logo.png                    |
|  app/                                         |
|    +-- myapp.exe                              |
|    +-- resources/                             |
|    +-- node_modules/ (Electron の場合)         |
|  AppxBlockMap.xml    ← ブロック単位ハッシュ      |
|  AppxSignature.p7x   ← デジタル署名            |
|  [Content_Types].xml                          |
|                                               |
+-----------------------------------------------+
```

### 2.2 electron-builder での MSIX 生成

```yaml
# electron-builder.yml
appId: com.example.myapp
productName: MyApp
copyright: Copyright (c) 2025 Example Inc.

win:
  target:
    - target: msix
      arch: [x64, arm64]
    - target: nsis
      arch: [x64]

msix:
  identityName: "12345ExampleInc.MyApp"
  publisher: "CN=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"
  publisherDisplayName: "Example Inc."
  applicationId: "MyApp"
  setBuildNumber: true
  languages:
    - "ja-JP"
    - "en-US"

publish:
  - provider: github
    owner: example-inc
    repo: myapp
```

### 2.3 AppxManifest.xml の設定

```xml
<?xml version="1.0" encoding="utf-8"?>
<Package
  xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
  xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
  xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities">

  <Identity
    Name="12345ExampleInc.MyApp"
    Version="1.2.0.0"
    Publisher="CN=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"
    ProcessorArchitecture="x64" />

  <Properties>
    <DisplayName>MyApp</DisplayName>
    <PublisherDisplayName>Example Inc.</PublisherDisplayName>
    <Logo>Assets\StoreLogo.png</Logo>
  </Properties>

  <Dependencies>
    <TargetDeviceFamily
      Name="Windows.Desktop"
      MinVersion="10.0.17763.0"
      MaxVersionTested="10.0.22621.0" />
  </Dependencies>

  <Resources>
    <Resource Language="ja-JP" />
    <Resource Language="en-US" />
  </Resources>

  <Applications>
    <Application
      Id="MyApp"
      Executable="app\myapp.exe"
      EntryPoint="Windows.FullTrustApplication">
      <uap:VisualElements
        DisplayName="MyApp"
        Description="My awesome desktop application"
        BackgroundColor="transparent"
        Square150x150Logo="Assets\Square150x150Logo.png"
        Square44x44Logo="Assets\Square44x44Logo.png">
        <uap:DefaultTile Wide310x150Logo="Assets\Wide310x150Logo.png" />
      </uap:VisualElements>

      <!-- ファイル関連付け -->
      <Extensions>
        <uap:Extension Category="windows.fileTypeAssociation">
          <uap:FileTypeAssociation Name="myapp-project">
            <uap:SupportedFileTypes>
              <uap:FileType>.myapp</uap:FileType>
            </uap:SupportedFileTypes>
          </uap:FileTypeAssociation>
        </uap:Extension>
      </Extensions>
    </Application>
  </Applications>

  <Capabilities>
    <Capability Name="internetClient" />
    <rescap:Capability Name="runFullTrust" />
  </Capabilities>
</Package>
```

### 2.4 Partner Center への提出フロー

```
+------------------------------------------------------------------+
|           Microsoft Store 提出フロー                               |
+------------------------------------------------------------------+
|                                                                  |
|  1. Partner Center アカウント作成 ($19 一回のみ)                   |
|       |                                                          |
|  2. アプリ名の予約                                                |
|       |                                                          |
|  3. Identity 情報の取得                                           |
|     (identityName, publisher, publisherDisplayName)               |
|       |                                                          |
|  4. MSIX パッケージのビルド & 署名                                 |
|       |                                                          |
|  5. Partner Center にアップロード                                  |
|     - パッケージ (.msix / .msixbundle)                            |
|     - スクリーンショット (最低1枚)                                 |
|     - 説明文 (日本語/英語)                                        |
|     - プライバシーポリシー URL                                     |
|       |                                                          |
|  6. 認定テスト (自動 + 手動)                                      |
|     - セキュリティスキャン                                        |
|     - API 準拠チェック                                            |
|     - コンテンツポリシー                                          |
|       |                                                          |
|  7. 公開 (審査通過後、即時 or スケジュール)                        |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 3. Mac App Store 配布

### 3.1 サンドボックス対応の entitlements

```xml
<!-- entitlements.mac.plist (App Store 用) -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- App Sandbox 必須 -->
    <key>com.apple.security.app-sandbox</key>
    <true/>

    <!-- ネットワークアクセス -->
    <key>com.apple.security.network.client</key>
    <true/>

    <!-- ファイルアクセス (ユーザー選択のみ) -->
    <key>com.apple.security.files.user-selected.read-write</key>
    <true/>

    <!-- ダウンロードフォルダ -->
    <key>com.apple.security.files.downloads.read-write</key>
    <true/>

    <!-- Hardened Runtime -->
    <key>com.apple.security.cs.allow-jit</key>
    <true/>
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>
</dict>
</plist>
```

### 3.2 electron-builder の Mac App Store 設定

```yaml
# electron-builder.yml (macOS 部分)
mac:
  target:
    - target: mas  # Mac App Store 用
      arch: [x64, arm64, universal]
    - target: dmg  # 直接配布用
      arch: [universal]
  category: public.app-category.developer-tools
  hardenedRuntime: true
  gatekeeperAssess: false
  entitlements: build/entitlements.mac.plist
  entitlementsInherit: build/entitlements.mac.plist

mas:
  entitlements: build/entitlements.mas.plist
  entitlementsInherit: build/entitlements.mas.inherit.plist
  provisioningProfile: build/embedded.provisionprofile

afterSign: scripts/notarize.js
```

### 3.3 Notarization スクリプト

```javascript
// scripts/notarize.js
const { notarize } = require('@electron/notarize');

exports.default = async function notarizing(context) {
  const { electronPlatformName, appOutDir } = context;

  if (electronPlatformName !== 'darwin') return;

  // Mac App Store ビルドでは不要 (Apple が署名する)
  if (context.packager.config.mac?.target?.[0]?.target === 'mas') return;

  const appName = context.packager.appInfo.productFilename;

  console.log(`Notarizing ${appName}...`);

  await notarize({
    appBundleId: 'com.example.myapp',
    appPath: `${appOutDir}/${appName}.app`,
    appleId: process.env.APPLE_ID,
    appleIdPassword: process.env.APPLE_APP_SPECIFIC_PASSWORD,
    teamId: process.env.APPLE_TEAM_ID,
  });

  console.log('Notarization 完了');
};
```

---

## 4. GitHub Releases 配布

### 4.1 GitHub Actions ワークフロー

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

permissions:
  contents: write

jobs:
  build:
    strategy:
      matrix:
        include:
          - os: windows-latest
            platform: win
          - os: macos-latest
            platform: mac
          - os: ubuntu-latest
            platform: linux

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - run: npm ci

      # Windows 署名
      - name: Import Windows Certificate
        if: matrix.platform == 'win'
        run: |
          echo "${{ secrets.WIN_CERT_BASE64 }}" | base64 -d > cert.pfx
        shell: bash

      # macOS 署名
      - name: Import macOS Certificate
        if: matrix.platform == 'mac'
        uses: apple-actions/import-codesign-certs@v2
        with:
          p12-file-base64: ${{ secrets.MAC_CERT_BASE64 }}
          p12-password: ${{ secrets.MAC_CERT_PASSWORD }}

      # ビルド & 公開
      - name: Build and Publish
        run: npx electron-builder --publish always
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          WIN_CSC_LINK: cert.pfx
          WIN_CSC_KEY_PASSWORD: ${{ secrets.WIN_CERT_PASSWORD }}
          APPLE_ID: ${{ secrets.APPLE_ID }}
          APPLE_APP_SPECIFIC_PASSWORD: ${{ secrets.APPLE_ASP }}
          APPLE_TEAM_ID: ${{ secrets.APPLE_TEAM_ID }}

      # リリースノート自動生成
      - name: Generate Release Notes
        if: matrix.platform == 'linux'  # 1回だけ実行
        uses: softprops/action-gh-release@v2
        with:
          generate_release_notes: true
          draft: true
```

### 4.2 Tauri の GitHub Actions ワークフロー

```yaml
# .github/workflows/tauri-release.yml
name: Tauri Release

on:
  push:
    tags:
      - 'v*'

jobs:
  create-release:
    runs-on: ubuntu-latest
    outputs:
      release_id: ${{ steps.create.outputs.result }}
    steps:
      - uses: actions/github-script@v7
        id: create
        with:
          script: |
            const { data } = await github.rest.repos.createRelease({
              owner: context.repo.owner,
              repo: context.repo.repo,
              tag_name: `${context.ref.replace('refs/tags/', '')}`,
              name: `Release ${context.ref.replace('refs/tags/', '')}`,
              draft: true,
              prerelease: false,
            });
            return data.id;

  build:
    needs: create-release
    strategy:
      matrix:
        include:
          - os: windows-latest
            target: x86_64-pc-windows-msvc
          - os: macos-latest
            target: aarch64-apple-darwin
          - os: macos-latest
            target: x86_64-apple-darwin
          - os: ubuntu-latest
            target: x86_64-unknown-linux-gnu

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.target }}

      - uses: actions/setup-node@v4
        with:
          node-version: 20

      - run: npm ci

      - uses: tauri-apps/tauri-action@v0
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          TAURI_SIGNING_PRIVATE_KEY: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY }}
          TAURI_SIGNING_PRIVATE_KEY_PASSWORD: ${{ secrets.TAURI_SIGNING_KEY_PASS }}
        with:
          releaseId: ${{ needs.create-release.outputs.release_id }}
          args: --target ${{ matrix.target }}
```

---

## 5. CI/CD パイプライン全体設計

### 5.1 リリースパイプラインの全体像

```
+------------------------------------------------------------------+
|                    リリースパイプライン全体像                        |
+------------------------------------------------------------------+
|                                                                  |
|  [開発者]                                                        |
|     |                                                            |
|     v  git tag v1.2.0 && git push --tags                        |
|                                                                  |
|  [GitHub Actions]                                                |
|     |                                                            |
|     +---> [Windows Runner]                                       |
|     |       - npm ci                                             |
|     |       - npm test                                           |
|     |       - electron-builder --win (NSIS + MSIX)               |
|     |       - signtool (Authenticode 署名)                       |
|     |       - Upload to GitHub Release                           |
|     |       - Upload to Partner Center (MSIX)                    |
|     |                                                            |
|     +---> [macOS Runner]                                         |
|     |       - npm ci                                             |
|     |       - npm test                                           |
|     |       - electron-builder --mac (DMG + MAS)                 |
|     |       - codesign + notarize                                |
|     |       - Upload to GitHub Release                           |
|     |       - Upload to App Store Connect (xcrun altool)         |
|     |                                                            |
|     +---> [Linux Runner]                                         |
|             - npm ci                                             |
|             - npm test                                           |
|             - electron-builder --linux (AppImage + deb + snap)   |
|             - Upload to GitHub Release                           |
|             - Upload to Snap Store                               |
|                                                                  |
+------------------------------------------------------------------+
```

### 5.2 セマンティックバージョニングとリリース自動化

```typescript
// scripts/bump-version.ts
import { execSync } from 'child_process';
import { readFileSync, writeFileSync } from 'fs';
import semver from 'semver';

type ReleaseType = 'patch' | 'minor' | 'major';

function bumpVersion(type: ReleaseType): void {
  const pkg = JSON.parse(readFileSync('package.json', 'utf-8'));
  const currentVersion = pkg.version;
  const newVersion = semver.inc(currentVersion, type)!;

  // package.json 更新
  pkg.version = newVersion;
  writeFileSync('package.json', JSON.stringify(pkg, null, 2) + '\n');

  // tauri.conf.json 更新 (Tauri の場合)
  try {
    const tauriConf = JSON.parse(readFileSync('src-tauri/tauri.conf.json', 'utf-8'));
    tauriConf.package.version = newVersion;
    writeFileSync('src-tauri/tauri.conf.json', JSON.stringify(tauriConf, null, 2) + '\n');
  } catch { /* Tauri 未使用なら無視 */ }

  // Git tag
  execSync(`git add -A`);
  execSync(`git commit -m "chore: bump version to v${newVersion}"`);
  execSync(`git tag v${newVersion}`);

  console.log(`Version bumped: ${currentVersion} -> ${newVersion}`);
  console.log(`Run "git push --follow-tags" to trigger release`);
}

const type = (process.argv[2] as ReleaseType) || 'patch';
bumpVersion(type);
```

---

## 6. パッケージ形式の比較

| 形式 | プラットフォーム | サンドボックス | 自動更新 | サイズ | 用途 |
|------|---------------|--------------|---------|-------|------|
| MSIX | Windows 10+ | 部分的 | Store 経由 | 中 | Store 配布 |
| NSIS | Windows | なし | electron-updater | 小 | 直接配布 |
| MSI | Windows | なし | なし(WiX) | 中 | 企業配布 |
| DMG | macOS | なし | 自前 | 大 | 直接配布 |
| pkg (MAS) | macOS | 必須 | Store 経由 | 中 | App Store |
| AppImage | Linux | なし | AppImageUpdate | 大 | 汎用 |
| deb | Debian/Ubuntu | なし | apt repo | 小 | Debian系 |
| snap | Linux | あり | snapd | 大 | Ubuntu中心 |
| flatpak | Linux | あり | Flathub | 大 | 汎用 |

---

## アンチパターン

### アンチパターン 1: シークレットのハードコード

```yaml
# NG: シークレットを直接ワークフローに記述
- name: Sign
  run: signtool sign /f cert.pfx /p "MyP@ssw0rd123" app.exe

# OK: GitHub Secrets を使用
- name: Sign
  run: signtool sign /f cert.pfx /p "${{ secrets.WIN_CERT_PASSWORD }}" app.exe
  env:
    WIN_CSC_LINK: ${{ secrets.WIN_CERT_BASE64 }}
```

**問題点**: CI/CD ログにパスワードが出力される可能性があり、リポジトリにアクセスできる全員に証明書の秘密鍵が漏洩する。必ず Secrets 管理機能を使い、ログマスキングを確認する。

### アンチパターン 2: 全プラットフォーム同時リリース

```yaml
# NG: 全プラットフォームを同時に公開
- name: Publish All
  run: npx electron-builder --publish always --win --mac --linux

# OK: Draft で作成し、テスト後に公開
- name: Build (Draft)
  run: npx electron-builder --publish always
  # GitHub Release は draft: true で作成
  # QA テスト完了後に手動で公開
```

**問題点**: あるプラットフォームで問題があった場合に、全プラットフォームのリリースを巻き戻す必要が生じる。Draft リリースで段階的に検証し、問題なければ公開する方が安全。

---

## FAQ

### Q1: Microsoft Store の審査でよく落ちるポイントは何ですか？

**A**: 最も多い理由は (1) プライバシーポリシーの不備または不適切な URL、(2) アプリの説明文とスクリーンショットの不一致、(3) クラッシュやハングアップなどの品質問題。特に Electron アプリでは、起動時間が遅いと審査に影響することがある。事前に Windows App Certification Kit (WACK) を実行してセルフチェックすることを強く推奨する。

### Q2: Mac App Store のサンドボックス制限で困る機能はありますか？

**A**: ファイルシステムへの広範なアクセス、グローバルキーボードショートカット、他アプリとのプロセス間通信、カーネル拡張などがサンドボックスで制限される。特に開発ツールでは Terminal.app の操作や `/usr/local` 配下のファイルアクセスが困難。これらの機能が必須の場合は、Mac App Store 外での直接配布（DMG + Notarization）を検討すべき。

### Q3: GitHub Releases と Store 配布を両方やるべきですか？

**A**: 理想的には両方提供すべき。Store 配布はエンドユーザーにとってインストール・更新が容易で、信頼性も高い。一方 GitHub Releases は開発者コミュニティへのリーチと、Store 審査を待たない即時配布に有利。CI/CD パイプラインで両方のチャネルに同時デプロイする設計にすることで、追加の運用コストを最小化できる。

---

## まとめ

| 項目 | 要点 |
|------|------|
| Microsoft Store | MSIX パッケージング + Partner Center 提出。審査 1-3 日 |
| Mac App Store | サンドボックス対応 + Notarization。審査 1-7 日 |
| GitHub Releases | 即時配布。electron-updater / Tauri updater と連携 |
| パッケージ形式 | MSIX(Win Store)、NSIS(Win直接)、DMG(Mac直接)、AppImage(Linux) |
| CI/CD | GitHub Actions でマルチプラットフォーム自動ビルド・署名・公開 |
| 署名 | Windows=Authenticode、macOS=codesign+notarize を CI で自動化 |
| リリース戦略 | Draft リリース → QA → 公開の段階的プロセスが安全 |
| バージョン管理 | semver + git tag でリリースを自動トリガー |

## 次に読むべきガイド

- [自動更新](./01-auto-update.md) -- electron-updater / Tauri updater による OTA 更新
- インストーラーのカスタマイズ -- NSIS スクリプトと WiX ツールセットの活用
- マルチアーキテクチャ対応 -- x64 / ARM64 / Universal Binary の戦略

## 参考文献

1. **Microsoft Store アプリの公開ガイド** -- https://learn.microsoft.com/ja-jp/windows/apps/publish/publish-your-app/overview -- Partner Center でのアプリ提出から公開までの公式ガイド
2. **Apple Developer - App Store 配布** -- https://developer.apple.com/distribute/ -- Mac App Store へのアプリ提出と Notarization の公式ドキュメント
3. **electron-builder 公式ドキュメント** -- https://www.electron.build/ -- マルチプラットフォームビルドと署名の包括的リファレンス
4. **GitHub Actions for Tauri** -- https://github.com/tauri-apps/tauri-action -- Tauri アプリのクロスプラットフォームビルドとリリース用 Action

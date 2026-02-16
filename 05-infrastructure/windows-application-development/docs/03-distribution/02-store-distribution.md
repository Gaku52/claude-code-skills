# ストア配布 (Store Distribution)

> Microsoft Store (MSIX)、Mac App Store、GitHub Releases を活用し、CI/CD パイプラインで自動化されたアプリケーション配布を実現する技術を学ぶ。

## この章で学ぶこと

1. **MSIX パッケージングと Microsoft Store 配布** -- Windows アプリを MSIX 形式でパッケージし、Microsoft Store に公開するまでの全工程を理解する
2. **Mac App Store と GitHub Releases の配布戦略** -- macOS アプリのサンドボックス対応と、GitHub Releases を使った直接配布を使い分ける
3. **CI/CD による自動ビルド・署名・公開パイプライン** -- GitHub Actions を中心に、マルチプラットフォームのリリース自動化を構築する
4. **Linux パッケージ配布** -- Snap Store、Flathub、自前 APT リポジトリへの配布手法を理解する
5. **企業内配布 (LOB/MDM)** -- Microsoft Intune や Jamf を使った企業向けアプリ配布を理解する

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
|       +-----> Snap Store / Flathub       [Linux ユーザー]         |
|       |          - サンドボックス配布                               |
|       |          - 自動更新対応                                    |
|       |                                                          |
|       +-----> 自社サイト / CDN            [企業向け]              |
|       |          - 完全制御                                       |
|       |          - カスタムインストーラー                           |
|       |                                                          |
|       +-----> 企業内配布 (LOB/MDM)        [社内ユーザー]          |
|                  - Intune / Jamf 連携                             |
|                  - サイレントインストール                           |
|                                                                  |
+------------------------------------------------------------------+
```

### 1.1 配布チャネル比較表

| 項目 | Microsoft Store | Mac App Store | GitHub Releases | Snap Store | 自社サイト |
|------|----------------|---------------|-----------------|------------|----------|
| 審査 | あり(1-3日) | あり(1-7日) | なし | あり(自動) | なし |
| 手数料 | 15%(アプリ) / 12%(ゲーム) | 30% / 15%(小規模) | 無料 | 無料 | 無料 |
| 自動更新 | OS 標準 | OS 標準 | 自前実装 | snapd | 自前実装 |
| サンドボックス | MSIX は制限あり | 必須 | なし | あり(strict) | なし |
| 到達範囲 | Windows 10/11 ユーザー | macOS ユーザー | 技術者中心 | Ubuntu中心 | 任意 |
| パッケージ形式 | MSIX / MSIX Bundle | pkg / app (zip) | exe/msi/dmg/AppImage | snap | 任意 |
| 企業配布 | LOB 対応 | MDM 対応 | 非対応 | 非対応 | 任意 |
| オフラインインストール | 可能(サイドロード) | 不可 | 可能 | 可能 | 可能 |

### 1.2 配布チャネル選定のフローチャート

アプリケーションの性質に応じて最適な配布チャネルを選定する指針を以下に示す。

```
[ターゲットユーザーは誰か？]
     |
     +--- 一般消費者（非技術者）
     |        |
     |        +--- Windows → Microsoft Store (MSIX) を第一候補
     |        +--- macOS → Mac App Store を第一候補
     |        +--- Linux → Snap Store / Flathub
     |
     +--- 開発者・技術者
     |        |
     |        +--- GitHub Releases + 自動更新 (electron-updater / Tauri updater)
     |        +--- 補助: Store にも掲載して発見性を高める
     |
     +--- 企業内ユーザー
     |        |
     |        +--- Windows → Intune + MSIX サイドロード
     |        +--- macOS → Jamf + pkg
     |        +--- 共通 → 自社 CDN + MDM 連携
     |
     +--- 全プラットフォーム最大到達
              |
              +--- Store + GitHub Releases + 自社サイトの三本柱
              +--- CI/CD で全チャネル同時配信
```

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
|    +-- LargeTile.png (310x310)               |
|    +-- SplashScreen.png (620x300)            |
|    +-- BadgeLogo.png (24x24)                 |
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
  xmlns:uap3="http://schemas.microsoft.com/appx/manifest/uap/windows10/3"
  xmlns:desktop="http://schemas.microsoft.com/appx/manifest/desktop/windows10"
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
    <Description>高機能なデスクトップアプリケーション</Description>
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
        <uap:DefaultTile
          Wide310x150Logo="Assets\Wide310x150Logo.png"
          Square310x310Logo="Assets\LargeTile.png"
          ShortName="MyApp" />
        <uap:SplashScreen Image="Assets\SplashScreen.png" />
      </uap:VisualElements>

      <!-- ファイル関連付け -->
      <Extensions>
        <uap:Extension Category="windows.fileTypeAssociation">
          <uap:FileTypeAssociation Name="myapp-project">
            <uap:SupportedFileTypes>
              <uap:FileType>.myapp</uap:FileType>
              <uap:FileType>.myproj</uap:FileType>
            </uap:SupportedFileTypes>
            <uap:DisplayName>MyApp Project</uap:DisplayName>
            <uap:Logo>Assets\FileIcon.png</uap:Logo>
          </uap:FileTypeAssociation>
        </uap:Extension>

        <!-- プロトコルハンドラ (myapp:// で起動) -->
        <uap:Extension Category="windows.protocol">
          <uap:Protocol Name="myapp">
            <uap:DisplayName>MyApp Protocol</uap:DisplayName>
          </uap:Protocol>
        </uap:Extension>

        <!-- スタートアップ起動 -->
        <desktop:Extension Category="windows.startupTask">
          <desktop:StartupTask
            TaskId="MyAppStartup"
            Enabled="false"
            DisplayName="MyApp を起動時に実行" />
        </desktop:Extension>

        <!-- トースト通知アクティベーション -->
        <desktop:Extension Category="windows.toastNotificationActivation">
          <desktop:ToastNotificationActivation
            ToastActivatorCLSID="XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX" />
        </desktop:Extension>
      </Extensions>
    </Application>
  </Applications>

  <Capabilities>
    <Capability Name="internetClient" />
    <Capability Name="internetClientServer" />
    <rescap:Capability Name="runFullTrust" />
    <uap3:Capability Name="backgroundMediaPlayback" />
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

### 2.5 Partner Center API による自動提出

CI/CD から Partner Center API を使って MSIX パッケージを自動提出するスクリプトを以下に示す。

```typescript
// scripts/submit-to-store.ts
// Partner Center Submission API を使った自動提出
import fetch from 'node-fetch';
import * as fs from 'fs';
import * as path from 'path';

interface SubmissionConfig {
  tenantId: string;
  clientId: string;
  clientSecret: string;
  appId: string;
}

interface AccessTokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

interface Submission {
  id: string;
  status: string;
  statusDetails: {
    errors: Array<{ code: string; details: string }>;
    warnings: Array<{ code: string; details: string }>;
  };
  fileUploadUrl: string;
}

class PartnerCenterClient {
  private config: SubmissionConfig;
  private accessToken: string = '';

  constructor(config: SubmissionConfig) {
    this.config = config;
  }

  // Azure AD でアクセストークンを取得
  async authenticate(): Promise<void> {
    const tokenUrl = `https://login.microsoftonline.com/${this.config.tenantId}/oauth2/v2.0/token`;

    const response = await fetch(tokenUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'client_credentials',
        client_id: this.config.clientId,
        client_secret: this.config.clientSecret,
        scope: 'https://manage.devcenter.microsoft.com/.default',
      }),
    });

    const data = (await response.json()) as AccessTokenResponse;
    this.accessToken = data.access_token;
    console.log('Partner Center 認証成功');
  }

  // 新しいサブミッションを作成
  async createSubmission(): Promise<Submission> {
    const url = `https://manage.devcenter.microsoft.com/v1.0/my/applications/${this.config.appId}/submissions`;

    // 既存の保留中サブミッションを削除
    const pendingResponse = await fetch(url, {
      headers: { Authorization: `Bearer ${this.accessToken}` },
    });
    const pending = await pendingResponse.json() as any;
    if (pending.pendingApplicationSubmission) {
      await fetch(`${url}/${pending.pendingApplicationSubmission.id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${this.accessToken}` },
      });
      console.log('既存の保留中サブミッションを削除しました');
    }

    // 新規サブミッション作成
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.accessToken}`,
        'Content-Type': 'application/json',
      },
    });

    const submission = (await response.json()) as Submission;
    console.log(`サブミッション作成: ${submission.id}`);
    return submission;
  }

  // MSIX パッケージをアップロード
  async uploadPackage(submission: Submission, msixPath: string): Promise<void> {
    const zipPath = msixPath.replace('.msix', '.zip');

    // MSIX を ZIP として準備（Azure Blob Storage への直接アップロード）
    const fileBuffer = fs.readFileSync(msixPath);

    const response = await fetch(submission.fileUploadUrl, {
      method: 'PUT',
      headers: {
        'x-ms-blob-type': 'BlockBlob',
        'Content-Type': 'application/octet-stream',
      },
      body: fileBuffer,
    });

    if (!response.ok) {
      throw new Error(`パッケージアップロード失敗: ${response.status}`);
    }
    console.log('パッケージアップロード完了');
  }

  // サブミッションをコミット (審査に提出)
  async commitSubmission(submissionId: string): Promise<void> {
    const url =
      `https://manage.devcenter.microsoft.com/v1.0/my/applications/${this.config.appId}/submissions/${submissionId}/commit`;

    const response = await fetch(url, {
      method: 'POST',
      headers: { Authorization: `Bearer ${this.accessToken}` },
    });

    if (!response.ok) {
      throw new Error(`サブミッションコミット失敗: ${response.status}`);
    }
    console.log('サブミッションをコミットしました（審査開始）');
  }

  // サブミッション状態のポーリング
  async waitForCompletion(submissionId: string): Promise<string> {
    const url =
      `https://manage.devcenter.microsoft.com/v1.0/my/applications/${this.config.appId}/submissions/${submissionId}/status`;

    const maxRetries = 60; // 最大60回 (= 1時間)
    for (let i = 0; i < maxRetries; i++) {
      const response = await fetch(url, {
        headers: { Authorization: `Bearer ${this.accessToken}` },
      });
      const status = (await response.json()) as { status: string; statusDetails: any };

      console.log(`ステータス: ${status.status} (${i + 1}/${maxRetries})`);

      if (status.status === 'CommitFailed') {
        console.error('エラー:', JSON.stringify(status.statusDetails, null, 2));
        throw new Error('サブミッションのコミットが失敗しました');
      }

      if (status.status === 'PreProcessing' || status.status === 'Certification') {
        // まだ処理中
      }

      if (status.status === 'Published' || status.status === 'Release') {
        return status.status;
      }

      // 1分待機してリトライ
      await new Promise((resolve) => setTimeout(resolve, 60_000));
    }

    throw new Error('タイムアウト: サブミッション完了を待機中');
  }
}

// メイン実行
async function main(): Promise<void> {
  const config: SubmissionConfig = {
    tenantId: process.env.AZURE_TENANT_ID!,
    clientId: process.env.AZURE_CLIENT_ID!,
    clientSecret: process.env.AZURE_CLIENT_SECRET!,
    appId: process.env.PARTNER_CENTER_APP_ID!,
  };

  const client = new PartnerCenterClient(config);

  await client.authenticate();
  const submission = await client.createSubmission();
  await client.uploadPackage(submission, path.resolve(process.argv[2]));
  await client.commitSubmission(submission.id);

  console.log('サブミッションが審査に提出されました');
  console.log('Partner Center ダッシュボードで進捗を確認してください');
}

main().catch((error) => {
  console.error('提出エラー:', error.message);
  process.exit(1);
});
```

### 2.6 WACK (Windows App Certification Kit) による事前検証

Microsoft Store に提出する前に、WACK を使ってローカルで事前検証を行う。

```powershell
# PowerShell: WACK による MSIX 検証
# WACK のパス (Windows 10 SDK に含まれる)
$wackPath = "C:\Program Files (x86)\Windows Kits\10\App Certification Kit\appcert.exe"

# テスト対象の MSIX パッケージ
$msixPath = ".\dist\MyApp-1.2.0-x64.msix"

# レポート出力先
$reportPath = ".\dist\wack-report.xml"

# MSIX パッケージのテスト実行
& $wackPath test -appxpackagepath $msixPath -reportoutputpath $reportPath

# レポートの解析
[xml]$report = Get-Content $reportPath
$overallResult = $report.REPORT.OVERALL_RESULT.InnerText

if ($overallResult -eq "PASS") {
    Write-Host "✓ WACK テスト合格" -ForegroundColor Green
} else {
    Write-Host "✗ WACK テスト不合格" -ForegroundColor Red

    # 失敗したテストの一覧を表示
    $failedTests = $report.REPORT.REQUIREMENTS.REQUIREMENT | Where-Object {
        $_.RESULT -eq "FAIL"
    }

    foreach ($test in $failedTests) {
        Write-Host "  不合格: $($test.TEST.NAME)" -ForegroundColor Yellow
        Write-Host "  理由: $($test.TEST.DESCRIPTION)" -ForegroundColor Yellow
    }
}
```

```powershell
# PowerShell: MSIX のサイドロードインストール（テスト用）
# 開発者モードの確認
$devMode = Get-WindowsDeveloperLicenseRegistration -ErrorAction SilentlyContinue
if (-not $devMode) {
    Write-Host "開発者モードを有効にしてください:" -ForegroundColor Yellow
    Write-Host "設定 → 更新とセキュリティ → 開発者向け → 開発者モード"
    exit 1
}

# MSIX パッケージのインストール
Add-AppxPackage -Path ".\dist\MyApp-1.2.0-x64.msix"

# インストール確認
$app = Get-AppxPackage | Where-Object { $_.Name -like "*MyApp*" }
if ($app) {
    Write-Host "インストール成功: $($app.Name) v$($app.Version)" -ForegroundColor Green
    Write-Host "PackageFamilyName: $($app.PackageFamilyName)"
} else {
    Write-Host "インストール失敗" -ForegroundColor Red
}
```

### 2.7 MSIX Bundle (マルチアーキテクチャ)

```powershell
# PowerShell: 複数アーキテクチャの MSIX を Bundle にまとめる
$makeAppxPath = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\makeappx.exe"

# 各アーキテクチャの MSIX を作成済みとする
$x64Msix = ".\dist\MyApp-1.2.0-x64.msix"
$arm64Msix = ".\dist\MyApp-1.2.0-arm64.msix"

# Bundle の作成
$bundlePath = ".\dist\MyApp-1.2.0.msixbundle"
& $makeAppxPath bundle /d ".\dist\msix-packages" /p $bundlePath

# Bundle の署名
$signtoolPath = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\signtool.exe"
& $signtoolPath sign /fd SHA256 /a /f ".\certs\store-cert.pfx" /p $env:CERT_PASSWORD $bundlePath

Write-Host "MSIX Bundle 作成完了: $bundlePath"
```

---

## 3. Mac App Store 配布

### 3.1 サンドボックス対応の entitlements

```xml
<!-- entitlements.mas.plist (Mac App Store 用) -->
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

```xml
<!-- entitlements.mas.inherit.plist (子プロセス用) -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.app-sandbox</key>
    <true/>
    <key>com.apple.security.inherit</key>
    <true/>
</dict>
</plist>
```

```xml
<!-- entitlements.mac.plist (直接配布用 - サンドボックスなし) -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- Hardened Runtime (Notarization に必要) -->
    <key>com.apple.security.cs.allow-jit</key>
    <true/>
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>

    <!-- 直接配布ではサンドボックスなし -->
    <!-- ファイルシステム全体にアクセス可能 -->
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
  # App Store 固有の設定
  binaries:
    # Electron Helper 等、子プロセスの署名設定
    - x86_64: false
      arm64: false

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

### 3.4 App Store Connect への提出自動化

```bash
#!/bin/bash
# scripts/submit-to-app-store.sh
# App Store Connect への自動提出スクリプト

set -euo pipefail

APP_PATH="$1"
APPLE_ID="${APPLE_ID:?APPLE_ID が未設定です}"
APPLE_ASP="${APPLE_APP_SPECIFIC_PASSWORD:?APPLE_APP_SPECIFIC_PASSWORD が未設定です}"
TEAM_ID="${APPLE_TEAM_ID:?APPLE_TEAM_ID が未設定です}"

echo "=== App Store Connect 提出スクリプト ==="

# 1. pkg ファイルの作成（MAS ビルドの場合）
echo "[1/4] pkg ファイルの作成..."
INSTALLER_CERT="3rd Party Mac Developer Installer: Example Inc. (${TEAM_ID})"
PKG_PATH="${APP_PATH%.app}.pkg"

productbuild \
  --component "$APP_PATH" /Applications \
  --sign "$INSTALLER_CERT" \
  "$PKG_PATH"

echo "pkg 作成完了: $PKG_PATH"

# 2. パッケージの検証
echo "[2/4] パッケージの検証..."
xcrun altool --validate-app \
  --file "$PKG_PATH" \
  --type macos \
  --username "$APPLE_ID" \
  --password "$APPLE_ASP" \
  --team-id "$TEAM_ID"

echo "検証合格"

# 3. App Store Connect へアップロード
echo "[3/4] App Store Connect へアップロード..."
xcrun altool --upload-app \
  --file "$PKG_PATH" \
  --type macos \
  --username "$APPLE_ID" \
  --password "$APPLE_ASP" \
  --team-id "$TEAM_ID"

echo "アップロード完了"

# 4. (代替) notarytool を使った方法 (macOS 13+)
# xcrun notarytool submit "$PKG_PATH" \
#   --apple-id "$APPLE_ID" \
#   --password "$APPLE_ASP" \
#   --team-id "$TEAM_ID" \
#   --wait

echo "[4/4] App Store Connect ダッシュボードで審査を開始してください"
echo "=== 提出完了 ==="
```

### 3.5 Provisioning Profile の管理

```bash
#!/bin/bash
# Mac App Store 用 Provisioning Profile のセットアップ
# CI 環境での自動化

set -euo pipefail

PROFILE_BASE64="${MAS_PROVISIONING_PROFILE_BASE64:?Profile が未設定です}"
PROFILE_DIR="$HOME/Library/MobileDevice/Provisioning Profiles"
PROFILE_PATH="${PROFILE_DIR}/embedded.provisionprofile"

# ディレクトリ作成
mkdir -p "$PROFILE_DIR"

# Base64 デコードして配置
echo "$PROFILE_BASE64" | base64 -d > "$PROFILE_PATH"

# プロファイル情報の確認
echo "=== Provisioning Profile 情報 ==="
security cms -D -i "$PROFILE_PATH" 2>/dev/null | \
  /usr/libexec/PlistBuddy -c "Print :Name" /dev/stdin
security cms -D -i "$PROFILE_PATH" 2>/dev/null | \
  /usr/libexec/PlistBuddy -c "Print :TeamIdentifier:0" /dev/stdin
security cms -D -i "$PROFILE_PATH" 2>/dev/null | \
  /usr/libexec/PlistBuddy -c "Print :ExpirationDate" /dev/stdin

echo "プロファイル配置完了: $PROFILE_PATH"

# ビルドディレクトリにもコピー（electron-builder が参照する）
cp "$PROFILE_PATH" "./build/embedded.provisionprofile"
```

### 3.6 Tauri アプリの Mac App Store 対応

```rust
// src-tauri/src/main.rs
// Tauri v2 で Mac App Store 対応する場合の設定

#[cfg(target_os = "macos")]
mod mac_app_store {
    use std::path::PathBuf;

    /// Mac App Store のサンドボックス環境で正しいパスを取得
    pub fn get_container_path() -> Option<PathBuf> {
        // サンドボックス環境では ~/Library/Containers/<bundle-id>/Data 配下に制限
        let home = std::env::var("HOME").ok()?;
        let container = PathBuf::from(home);

        if container.to_string_lossy().contains("Containers") {
            // サンドボックス内で実行中
            Some(container)
        } else {
            // サンドボックス外 (開発時)
            None
        }
    }

    /// サンドボックス環境かどうかを判定
    pub fn is_sandboxed() -> bool {
        std::env::var("APP_SANDBOX_CONTAINER_ID").is_ok()
    }

    /// ブックマークを使った永続的なファイルアクセス (Security-Scoped Bookmarks)
    /// サンドボックスでユーザーが選択したファイルへの継続的アクセスを実現
    pub fn save_security_scoped_bookmark(url: &str) -> Result<Vec<u8>, String> {
        // macOS の Security-Scoped Bookmark API を呼び出す
        // 実際の実装では objc クレートや cocoa クレートが必要
        // ここでは概念的な実装を示す
        Err("未実装: objc バインディングが必要".to_string())
    }
}

use tauri::Manager;

#[tauri::command]
fn check_sandbox_status() -> serde_json::Value {
    #[cfg(target_os = "macos")]
    {
        let sandboxed = mac_app_store::is_sandboxed();
        let container = mac_app_store::get_container_path()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();

        serde_json::json!({
            "sandboxed": sandboxed,
            "containerPath": container,
            "platform": "macos"
        })
    }

    #[cfg(not(target_os = "macos"))]
    {
        serde_json::json!({
            "sandboxed": false,
            "containerPath": "",
            "platform": std::env::consts::OS
        })
    }
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![check_sandbox_status])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

---

## 4. GitHub Releases 配布

### 4.1 GitHub Actions ワークフロー (Electron)

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

      # Linux 依存パッケージ
      - name: Install Linux dependencies
        if: matrix.os == 'ubuntu-latest'
        run: |
          sudo apt-get update
          sudo apt-get install -y \
            libwebkit2gtk-4.1-dev \
            libappindicator3-dev \
            librsvg2-dev \
            patchelf

      - uses: tauri-apps/tauri-action@v0
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          TAURI_SIGNING_PRIVATE_KEY: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY }}
          TAURI_SIGNING_PRIVATE_KEY_PASSWORD: ${{ secrets.TAURI_SIGNING_KEY_PASS }}
        with:
          releaseId: ${{ needs.create-release.outputs.release_id }}
          args: --target ${{ matrix.target }}

  # Draft リリースを公開
  publish-release:
    needs: [create-release, build]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/github-script@v7
        with:
          script: |
            await github.rest.repos.updateRelease({
              owner: context.repo.owner,
              repo: context.repo.repo,
              release_id: ${{ needs.create-release.outputs.release_id }},
              draft: false,
            });
```

### 4.3 リリースノート自動生成

```typescript
// scripts/generate-release-notes.ts
// Conventional Commits からリリースノートを自動生成
import { execSync } from 'child_process';

interface CommitInfo {
  hash: string;
  type: string;
  scope: string;
  subject: string;
  breaking: boolean;
}

function parseCommits(since: string): CommitInfo[] {
  const log = execSync(
    `git log ${since}..HEAD --pretty=format:"%H|%s" --no-merges`
  ).toString();

  return log
    .split('\n')
    .filter(Boolean)
    .map((line) => {
      const [hash, subject] = line.split('|');
      // Conventional Commit パターン: type(scope): subject
      const match = subject.match(/^(\w+)(?:\(([^)]*)\))?(!)?:\s*(.*)$/);

      if (!match) {
        return { hash, type: 'other', scope: '', subject, breaking: false };
      }

      return {
        hash: hash.substring(0, 8),
        type: match[1],
        scope: match[2] || '',
        subject: match[4],
        breaking: match[3] === '!',
      };
    });
}

function generateNotes(commits: CommitInfo[]): string {
  const sections: Record<string, { title: string; items: string[] }> = {
    feat: { title: '新機能', items: [] },
    fix: { title: 'バグ修正', items: [] },
    perf: { title: 'パフォーマンス改善', items: [] },
    refactor: { title: 'リファクタリング', items: [] },
    docs: { title: 'ドキュメント', items: [] },
    chore: { title: 'その他', items: [] },
  };

  const breakingChanges: string[] = [];

  for (const commit of commits) {
    if (commit.breaking) {
      breakingChanges.push(`- ${commit.subject} (${commit.hash})`);
    }

    const section = sections[commit.type] || sections.chore;
    const scope = commit.scope ? `**${commit.scope}**: ` : '';
    section.items.push(`- ${scope}${commit.subject} (${commit.hash})`);
  }

  let notes = '';

  if (breakingChanges.length > 0) {
    notes += '## 破壊的変更\n\n';
    notes += breakingChanges.join('\n') + '\n\n';
  }

  for (const [_, section] of Object.entries(sections)) {
    if (section.items.length > 0) {
      notes += `## ${section.title}\n\n`;
      notes += section.items.join('\n') + '\n\n';
    }
  }

  return notes;
}

// 直前のタグから HEAD までのコミットを解析
const lastTag = execSync('git describe --tags --abbrev=0 HEAD~1 2>/dev/null || echo ""')
  .toString()
  .trim();

const since = lastTag || execSync('git rev-list --max-parents=0 HEAD').toString().trim();
const commits = parseCommits(since);
const notes = generateNotes(commits);

console.log(notes);
```

---

## 5. Linux パッケージ配布

### 5.1 Snap パッケージ

```yaml
# snap/snapcraft.yaml
name: myapp
base: core22
version: '1.2.0'
summary: 高機能デスクトップアプリケーション
description: |
  MyApp は高機能なデスクトップアプリケーションです。
  クロスプラットフォーム対応で、直感的な UI を提供します。

grade: stable
confinement: strict

architectures:
  - build-on: [amd64]
  - build-on: [arm64]

apps:
  myapp:
    command: myapp
    desktop: usr/share/applications/myapp.desktop
    extensions: [gnome]
    plugs:
      - home
      - network
      - network-bind
      - desktop
      - desktop-legacy
      - wayland
      - x11
      - browser-support
      - removable-media

parts:
  myapp:
    plugin: dump
    source: dist/linux-unpacked/
    stage-packages:
      - libgtk-3-0
      - libnotify4
      - libnss3
      - libxss1
      - libxtst6
      - xdg-utils
      - libatspi2.0-0
      - libuuid1
      - libsecret-1-0
```

```bash
#!/bin/bash
# scripts/publish-to-snap-store.sh
# Snap Store への自動公開

set -euo pipefail

SNAP_FILE="$1"
CHANNEL="${2:-edge}"  # edge, beta, candidate, stable

echo "=== Snap Store 公開スクリプト ==="

# Snapcraft ログイン (CI 環境ではエクスポートした認証情報を使用)
if [ -n "${SNAPCRAFT_STORE_CREDENTIALS:-}" ]; then
    echo "$SNAPCRAFT_STORE_CREDENTIALS" | snapcraft login --with -
fi

# Snap パッケージの検証
echo "[1/3] パッケージ検証中..."
snap review "$SNAP_FILE" || true  # 警告は無視

# アップロード & リリース
echo "[2/3] $CHANNEL チャネルにアップロード中..."
snapcraft upload "$SNAP_FILE" --release="$CHANNEL"

# 公開状態の確認
echo "[3/3] 公開状態の確認..."
snapcraft status myapp

echo "=== Snap Store 公開完了 (チャネル: $CHANNEL) ==="
```

### 5.2 Flatpak パッケージ

```yaml
# com.example.MyApp.yaml (Flatpak マニフェスト)
app-id: com.example.MyApp
runtime: org.freedesktop.Platform
runtime-version: '23.08'
sdk: org.freedesktop.Sdk
command: myapp
finish-args:
  - --share=ipc
  - --socket=x11
  - --socket=wayland
  - --socket=pulseaudio
  - --share=network
  - --device=dri
  - --filesystem=home
  - --filesystem=/tmp
  - --talk-name=org.freedesktop.Notifications
  - --talk-name=org.kde.StatusNotifierWatcher

modules:
  - name: myapp
    buildsystem: simple
    build-commands:
      - install -Dm755 myapp /app/bin/myapp
      - install -Dm644 myapp.desktop /app/share/applications/com.example.MyApp.desktop
      - install -Dm644 myapp-icon-256.png /app/share/icons/hicolor/256x256/apps/com.example.MyApp.png
    sources:
      - type: archive
        url: https://github.com/example-inc/myapp/releases/download/v1.2.0/myapp-linux-x64.tar.gz
        sha256: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### 5.3 自前 APT リポジトリの構築

```bash
#!/bin/bash
# scripts/setup-apt-repo.sh
# S3 + CloudFront を使った APT リポジトリの構築

set -euo pipefail

REPO_DIR="./apt-repo"
GPG_KEY_ID="${GPG_SIGNING_KEY_ID:?GPG_SIGNING_KEY_ID が未設定です}"
S3_BUCKET="${APT_REPO_BUCKET:?APT_REPO_BUCKET が未設定です}"

# ディレクトリ構成
mkdir -p "$REPO_DIR/pool/main"
mkdir -p "$REPO_DIR/dists/stable/main/binary-amd64"
mkdir -p "$REPO_DIR/dists/stable/main/binary-arm64"

# deb パッケージをプールに配置
cp dist/*.deb "$REPO_DIR/pool/main/"

# Packages ファイルの生成
cd "$REPO_DIR"
dpkg-scanpackages pool/main /dev/null > dists/stable/main/binary-amd64/Packages
gzip -k dists/stable/main/binary-amd64/Packages

# Release ファイルの生成
cat > dists/stable/Release << EOF
Origin: MyApp Repository
Label: MyApp
Suite: stable
Codename: stable
Version: 1.0
Architectures: amd64 arm64
Components: main
Description: MyApp APT Repository
EOF

# チェックサムの追加
apt-ftparchive release dists/stable >> dists/stable/Release

# GPG 署名
gpg --default-key "$GPG_KEY_ID" -abs -o dists/stable/Release.gpg dists/stable/Release
gpg --default-key "$GPG_KEY_ID" --clearsign -o dists/stable/InRelease dists/stable/Release

# S3 にアップロード
aws s3 sync "$REPO_DIR" "s3://$S3_BUCKET/" \
  --delete \
  --cache-control "max-age=300"

echo "=== APT リポジトリ更新完了 ==="
echo "ユーザー側の設定:"
echo "  curl -fsSL https://apt.example.com/gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/myapp.gpg"
echo "  echo 'deb [signed-by=/etc/apt/keyrings/myapp.gpg] https://apt.example.com stable main' | sudo tee /etc/apt/sources.list.d/myapp.list"
echo "  sudo apt update && sudo apt install myapp"
```

---

## 6. CI/CD パイプライン全体設計

### 6.1 リリースパイプラインの全体像

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

### 6.2 マルチストア統合リリースワークフロー

```yaml
# .github/workflows/multi-store-release.yml
# 全ストアへの統合リリースワークフロー
name: Multi-Store Release

on:
  workflow_dispatch:
    inputs:
      version:
        description: 'リリースバージョン (例: 1.2.0)'
        required: true
      channels:
        description: 'リリースチャネル (comma separated: github,msstore,appstore,snap)'
        required: true
        default: 'github,msstore,appstore,snap'
      prerelease:
        description: 'プレリリースとして公開'
        type: boolean
        default: false

jobs:
  # ステップ 1: バージョンバンプとタグ作成
  prepare:
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.version.outputs.version }}
      tag: ${{ steps.version.outputs.tag }}
    steps:
      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.PAT_TOKEN }}

      - name: Set version
        id: version
        run: |
          VERSION="${{ inputs.version }}"
          echo "version=$VERSION" >> $GITHUB_OUTPUT
          echo "tag=v$VERSION" >> $GITHUB_OUTPUT

      - name: Bump version in package.json
        run: |
          npm version ${{ steps.version.outputs.version }} --no-git-tag-version
          git add package.json package-lock.json
          git commit -m "chore: bump version to v${{ steps.version.outputs.version }}"
          git tag v${{ steps.version.outputs.version }}
          git push --follow-tags

  # ステップ 2: マルチプラットフォームビルド
  build-windows:
    needs: prepare
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: v${{ needs.prepare.outputs.version }}

      - uses: actions/setup-node@v4
        with:
          node-version: 20

      - run: npm ci

      - name: Build Windows packages
        run: npx electron-builder --win --publish never
        env:
          WIN_CSC_LINK: ${{ secrets.WIN_CERT_BASE64 }}
          WIN_CSC_KEY_PASSWORD: ${{ secrets.WIN_CERT_PASSWORD }}

      - uses: actions/upload-artifact@v4
        with:
          name: windows-artifacts
          path: |
            dist/*.exe
            dist/*.msix
            dist/*.msixbundle
            dist/latest.yml

  build-macos:
    needs: prepare
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: v${{ needs.prepare.outputs.version }}

      - uses: actions/setup-node@v4
        with:
          node-version: 20

      - uses: apple-actions/import-codesign-certs@v2
        with:
          p12-file-base64: ${{ secrets.MAC_CERT_BASE64 }}
          p12-password: ${{ secrets.MAC_CERT_PASSWORD }}

      - run: npm ci

      - name: Build macOS packages
        run: npx electron-builder --mac --publish never
        env:
          APPLE_ID: ${{ secrets.APPLE_ID }}
          APPLE_APP_SPECIFIC_PASSWORD: ${{ secrets.APPLE_ASP }}
          APPLE_TEAM_ID: ${{ secrets.APPLE_TEAM_ID }}

      - uses: actions/upload-artifact@v4
        with:
          name: macos-artifacts
          path: |
            dist/*.dmg
            dist/*.pkg
            dist/*.zip
            dist/latest-mac.yml

  build-linux:
    needs: prepare
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: v${{ needs.prepare.outputs.version }}

      - uses: actions/setup-node@v4
        with:
          node-version: 20

      - run: npm ci

      - name: Build Linux packages
        run: npx electron-builder --linux --publish never

      - uses: actions/upload-artifact@v4
        with:
          name: linux-artifacts
          path: |
            dist/*.AppImage
            dist/*.deb
            dist/*.snap
            dist/latest-linux.yml

  # ステップ 3: GitHub Releases
  publish-github:
    needs: [prepare, build-windows, build-macos, build-linux]
    if: contains(inputs.channels, 'github')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          tag_name: v${{ needs.prepare.outputs.version }}
          name: Release v${{ needs.prepare.outputs.version }}
          draft: false
          prerelease: ${{ inputs.prerelease }}
          generate_release_notes: true
          files: |
            windows-artifacts/*.exe
            windows-artifacts/*.msix
            windows-artifacts/latest.yml
            macos-artifacts/*.dmg
            macos-artifacts/*.zip
            macos-artifacts/latest-mac.yml
            linux-artifacts/*.AppImage
            linux-artifacts/*.deb
            linux-artifacts/latest-linux.yml

  # ステップ 4: Microsoft Store
  publish-msstore:
    needs: [prepare, build-windows]
    if: contains(inputs.channels, 'msstore')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          name: windows-artifacts

      - name: Submit to Partner Center
        run: npx ts-node scripts/submit-to-store.ts "*.msixbundle"
        env:
          AZURE_TENANT_ID: ${{ secrets.AZURE_TENANT_ID }}
          AZURE_CLIENT_ID: ${{ secrets.AZURE_CLIENT_ID }}
          AZURE_CLIENT_SECRET: ${{ secrets.AZURE_CLIENT_SECRET }}
          PARTNER_CENTER_APP_ID: ${{ secrets.PARTNER_CENTER_APP_ID }}

  # ステップ 5: Snap Store
  publish-snap:
    needs: [prepare, build-linux]
    if: contains(inputs.channels, 'snap')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: linux-artifacts

      - uses: snapcore/action-publish@v1
        env:
          SNAPCRAFT_STORE_CREDENTIALS: ${{ secrets.SNAPCRAFT_STORE_CREDENTIALS }}
        with:
          snap: "*.snap"
          release: stable
```

### 6.3 セマンティックバージョニングとリリース自動化

```typescript
// scripts/bump-version.ts
import { execSync } from 'child_process';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import semver from 'semver';

type ReleaseType = 'patch' | 'minor' | 'major';

interface VersionFile {
  path: string;
  // JSON パス (dot 区切り) でバージョンフィールドを指定
  jsonPath: string;
}

// バージョンを更新すべきファイルの一覧
const VERSION_FILES: VersionFile[] = [
  { path: 'package.json', jsonPath: 'version' },
  { path: 'src-tauri/tauri.conf.json', jsonPath: 'version' },
  { path: 'src-tauri/Cargo.toml', jsonPath: '' }, // TOML は別処理
];

function updateJsonVersion(filePath: string, jsonPath: string, newVersion: string): boolean {
  if (!existsSync(filePath)) return false;

  const content = JSON.parse(readFileSync(filePath, 'utf-8'));
  const keys = jsonPath.split('.');
  let obj = content;

  for (let i = 0; i < keys.length - 1; i++) {
    obj = obj[keys[i]];
    if (!obj) return false;
  }

  obj[keys[keys.length - 1]] = newVersion;
  writeFileSync(filePath, JSON.stringify(content, null, 2) + '\n');
  console.log(`  更新: ${filePath} -> ${newVersion}`);
  return true;
}

function updateCargoToml(filePath: string, newVersion: string): boolean {
  if (!existsSync(filePath)) return false;

  let content = readFileSync(filePath, 'utf-8');
  content = content.replace(
    /^version\s*=\s*"[^"]*"/m,
    `version = "${newVersion}"`
  );
  writeFileSync(filePath, content);
  console.log(`  更新: ${filePath} -> ${newVersion}`);
  return true;
}

function bumpVersion(type: ReleaseType): void {
  const pkg = JSON.parse(readFileSync('package.json', 'utf-8'));
  const currentVersion = pkg.version;
  const newVersion = semver.inc(currentVersion, type)!;

  console.log(`バージョン更新: ${currentVersion} -> ${newVersion}`);

  // 各ファイルのバージョンを更新
  for (const vf of VERSION_FILES) {
    if (vf.jsonPath) {
      updateJsonVersion(vf.path, vf.jsonPath, newVersion);
    } else if (vf.path.endsWith('.toml')) {
      updateCargoToml(vf.path, newVersion);
    }
  }

  // CHANGELOG の更新 (もし存在すれば)
  if (existsSync('CHANGELOG.md')) {
    const changelog = readFileSync('CHANGELOG.md', 'utf-8');
    const date = new Date().toISOString().split('T')[0];
    const newEntry = `## [${newVersion}] - ${date}\n\n`;
    const updated = changelog.replace(
      /^## \[Unreleased\]/m,
      `## [Unreleased]\n\n${newEntry}`
    );
    writeFileSync('CHANGELOG.md', updated);
    console.log(`  更新: CHANGELOG.md`);
  }

  // Git 操作
  execSync(`git add -A`);
  execSync(`git commit -m "chore: bump version to v${newVersion}"`);
  execSync(`git tag v${newVersion}`);

  console.log(`\nVersion bumped: ${currentVersion} -> ${newVersion}`);
  console.log(`Run "git push --follow-tags" to trigger release`);
}

const type = (process.argv[2] as ReleaseType) || 'patch';
bumpVersion(type);
```

---

## 7. 企業内配布 (LOB / MDM)

### 7.1 Microsoft Intune による配布

```powershell
# PowerShell: Intune への LOB アプリ登録スクリプト
# Microsoft Graph API を使用

$TenantId = $env:AZURE_TENANT_ID
$ClientId = $env:AZURE_CLIENT_ID
$ClientSecret = $env:AZURE_CLIENT_SECRET
$MsixPath = $args[0]

# アクセストークン取得
$tokenBody = @{
    grant_type    = "client_credentials"
    client_id     = $ClientId
    client_secret = $ClientSecret
    scope         = "https://graph.microsoft.com/.default"
}

$tokenResponse = Invoke-RestMethod `
    -Uri "https://login.microsoftonline.com/$TenantId/oauth2/v2.0/token" `
    -Method POST `
    -ContentType "application/x-www-form-urlencoded" `
    -Body $tokenBody

$token = $tokenResponse.access_token
$headers = @{
    "Authorization" = "Bearer $token"
    "Content-Type"  = "application/json"
}

# LOB アプリの作成
$appBody = @{
    "@odata.type"       = "#microsoft.graph.windowsAppX"
    displayName         = "MyApp"
    description         = "社内向けデスクトップアプリケーション"
    publisher           = "Example Inc."
    applicableArchitectures = "x64"
    identityName        = "12345ExampleInc.MyApp"
    identityPublisherHash = "XXXXXXXX"
    identityVersion     = "1.2.0.0"
    minimumSupportedOperatingSystem = @{
        v10_1809 = $true
    }
} | ConvertTo-Json -Depth 10

$app = Invoke-RestMethod `
    -Uri "https://graph.microsoft.com/v1.0/deviceAppManagement/mobileApps" `
    -Method POST `
    -Headers $headers `
    -Body $appBody

Write-Host "LOB アプリ作成完了: $($app.id)"

# アプリの割り当て (全デバイスに配布)
$assignmentBody = @{
    mobileAppAssignments = @(
        @{
            "@odata.type" = "#microsoft.graph.mobileAppAssignment"
            intent        = "required"  # required = 強制インストール
            target        = @{
                "@odata.type" = "#microsoft.graph.allDevicesAssignmentTarget"
            }
            settings      = @{
                "@odata.type"    = "#microsoft.graph.windowsAppXAppAssignmentSettings"
                useDeviceContext = $true  # デバイスコンテキストでインストール
            }
        }
    )
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
    -Uri "https://graph.microsoft.com/v1.0/deviceAppManagement/mobileApps/$($app.id)/assign" `
    -Method POST `
    -Headers $headers `
    -Body $assignmentBody

Write-Host "アプリ割り当て完了（全デバイスに必須インストール）"
```

### 7.2 macOS (Jamf Pro) での配布

```bash
#!/bin/bash
# scripts/deploy-to-jamf.sh
# Jamf Pro API を使った macOS アプリの企業配布

set -euo pipefail

JAMF_URL="${JAMF_PRO_URL:?JAMF_PRO_URL が未設定です}"
JAMF_USER="${JAMF_API_USER:?JAMF_API_USER が未設定です}"
JAMF_PASS="${JAMF_API_PASSWORD:?JAMF_API_PASSWORD が未設定です}"
PKG_PATH="$1"
APP_NAME="MyApp"

echo "=== Jamf Pro デプロイスクリプト ==="

# 1. Bearer Token の取得
echo "[1/4] 認証中..."
TOKEN=$(curl -s -X POST "${JAMF_URL}/api/v1/auth/token" \
  -u "${JAMF_USER}:${JAMF_PASS}" | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")

HEADERS=(
  -H "Authorization: Bearer ${TOKEN}"
  -H "Accept: application/json"
)

# 2. パッケージのアップロード
echo "[2/4] パッケージアップロード中..."
UPLOAD_RESULT=$(curl -s -X POST "${JAMF_URL}/api/v1/packages" \
  "${HEADERS[@]}" \
  -F "file=@${PKG_PATH}" \
  -F "name=${APP_NAME}-$(date +%Y%m%d)")

PACKAGE_ID=$(echo "$UPLOAD_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "パッケージ ID: $PACKAGE_ID"

# 3. ポリシーの作成（配布ルールの設定）
echo "[3/4] 配布ポリシー作成中..."
POLICY_XML="<policy>
  <general>
    <name>Deploy ${APP_NAME}</name>
    <enabled>true</enabled>
    <trigger>recurring check-in</trigger>
    <frequency>Once per computer</frequency>
  </general>
  <scope>
    <all_computers>true</all_computers>
  </scope>
  <package_configuration>
    <packages>
      <size>1</size>
      <package>
        <id>${PACKAGE_ID}</id>
        <action>Install</action>
      </package>
    </packages>
  </package_configuration>
</policy>"

curl -s -X POST "${JAMF_URL}/JSSResource/policies" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/xml" \
  -d "$POLICY_XML"

echo "[4/4] 配布ポリシー作成完了"
echo "=== Jamf Pro デプロイ完了 ==="
```

---

## 8. パッケージ形式の比較

| 形式 | プラットフォーム | サンドボックス | 自動更新 | サイズ | 用途 |
|------|---------------|--------------|---------|-------|------|
| MSIX | Windows 10+ | 部分的 | Store 経由 | 中 | Store 配布 |
| NSIS | Windows | なし | electron-updater | 小 | 直接配布 |
| MSI | Windows | なし | なし(WiX) | 中 | 企業配布 |
| DMG | macOS | なし | 自前 | 大 | 直接配布 |
| pkg (MAS) | macOS | 必須 | Store 経由 | 中 | App Store |
| AppImage | Linux | なし | AppImageUpdate | 大 | 汎用 |
| deb | Debian/Ubuntu | なし | apt repo | 小 | Debian系 |
| snap | Linux | あり(strict) | snapd | 大 | Ubuntu中心 |
| flatpak | Linux | あり | Flathub | 大 | 汎用 |
| RPM | RHEL/Fedora | なし | dnf/yum repo | 小 | RedHat系 |

### 8.1 パッケージ形式の選定ガイド

```
[配布対象は？]
     |
     +--- Windows 一般ユーザー
     |        +--- Store 経由 → MSIX
     |        +--- 直接配布 → NSIS (exe)
     |        +--- 企業内 → MSI (グループポリシー対応)
     |
     +--- macOS 一般ユーザー
     |        +--- App Store → pkg (MAS)
     |        +--- 直接配布 → DMG + Notarization
     |
     +--- Linux
     |        +--- 汎用 → AppImage (依存関係なし)
     |        +--- Ubuntu/GNOME → Snap
     |        +--- Fedora/KDE → Flatpak
     |        +--- Debian系サーバー → deb + APT リポジトリ
     |        +--- RHEL系サーバー → RPM + YUM/DNF リポジトリ
     |
     +--- 全プラットフォーム
              +--- Store + 直接配布 の二本柱を推奨
              +--- CI/CD で全形式を自動ビルド
```

---

## 9. コード署名の包括的ガイド

### 9.1 Windows (Authenticode) 署名

```powershell
# PowerShell: Windows コード署名の完全フロー

# 1. 証明書の取得方法
# EV (Extended Validation) コード署名証明書を推奨
# 発行元: DigiCert, Sectigo, GlobalSign 等

# 2. signtool によるバイナリ署名
$signtoolPath = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\signtool.exe"

# PFX ファイルを使った署名
& $signtoolPath sign `
    /f "certificate.pfx" `
    /p "$env:CERT_PASSWORD" `
    /fd SHA256 `
    /tr http://timestamp.digicert.com `
    /td SHA256 `
    /d "MyApp Desktop Application" `
    /du "https://www.example.com/myapp" `
    "dist\MyApp-Setup-1.2.0.exe"

# 3. 署名の検証
& $signtoolPath verify /pa /v "dist\MyApp-Setup-1.2.0.exe"

# 4. MSIX の署名
& $signtoolPath sign `
    /f "store-certificate.pfx" `
    /p "$env:STORE_CERT_PASSWORD" `
    /fd SHA256 `
    "dist\MyApp-1.2.0-x64.msix"

Write-Host "コード署名完了"
```

### 9.2 macOS (codesign + notarize) 署名

```bash
#!/bin/bash
# scripts/macos-sign-and-notarize.sh
# macOS アプリの完全な署名・Notarization フロー

set -euo pipefail

APP_PATH="$1"
DEVELOPER_ID="Developer ID Application: Example Inc. (XXXXXXXXXX)"
TEAM_ID="${APPLE_TEAM_ID:?APPLE_TEAM_ID が未設定です}"

echo "=== macOS 署名・Notarization フロー ==="

# 1. deep signing (全バイナリを再帰的に署名)
echo "[1/5] コード署名中..."
codesign --deep --force --verify --verbose \
  --sign "$DEVELOPER_ID" \
  --options runtime \
  --entitlements "build/entitlements.mac.plist" \
  --timestamp \
  "$APP_PATH"

# 2. 署名の検証
echo "[2/5] 署名検証中..."
codesign --verify --deep --strict --verbose=2 "$APP_PATH"
spctl --assess --type execute --verbose "$APP_PATH"

# 3. DMG の作成
echo "[3/5] DMG 作成中..."
DMG_PATH="${APP_PATH%.app}.dmg"
hdiutil create -volname "MyApp" \
  -srcfolder "$APP_PATH" \
  -ov -format UDZO \
  "$DMG_PATH"

# DMG にも署名
codesign --sign "$DEVELOPER_ID" --timestamp "$DMG_PATH"

# 4. Notarization 提出
echo "[4/5] Notarization 提出中..."
SUBMISSION_ID=$(xcrun notarytool submit "$DMG_PATH" \
  --apple-id "$APPLE_ID" \
  --password "$APPLE_APP_SPECIFIC_PASSWORD" \
  --team-id "$TEAM_ID" \
  --wait 2>&1 | grep "id:" | head -1 | awk '{print $2}')

echo "Submission ID: $SUBMISSION_ID"

# 5. Stapling (Notarization チケットの埋め込み)
echo "[5/5] Stapling 中..."
xcrun stapler staple "$DMG_PATH"

# 最終検証
xcrun stapler validate "$DMG_PATH"

echo "=== 署名・Notarization 完了: $DMG_PATH ==="
```

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

### アンチパターン 3: Store 審査の考慮不足

```typescript
// NG: Mac App Store サンドボックスで禁止される操作
import { execSync } from 'child_process';

function runShellCommand(cmd: string): string {
  // サンドボックス内では exec が制限される
  return execSync(cmd).toString();
}

function readSystemFile(): string {
  // /etc や /usr 配下はアクセス不可
  return fs.readFileSync('/etc/hosts', 'utf-8');
}

// OK: サンドボックス対応の実装
import { app } from 'electron';
import * as path from 'path';

function getAppDataPath(): string {
  // サンドボックス内の正しいパスを使用
  return app.getPath('userData');
}

function readUserFile(): string {
  // ユーザーが明示的に選択したファイルのみアクセス
  // dialog.showOpenDialog() 経由でパスを取得
  const filePath = path.join(app.getPath('userData'), 'config.json');
  return fs.readFileSync(filePath, 'utf-8');
}
```

**問題点**: Mac App Store のサンドボックス制限を無視した実装は審査で却下される。Microsoft Store の MSIX でも同様に、任意のパスへの書き込みや管理者権限の要求は避けるべき。Store 向けビルドとそれ以外を分岐する設計が必要。

### アンチパターン 4: バージョン不整合

```json
// NG: 複数ファイルのバージョンが不一致
// package.json
{ "version": "1.2.0" }

// tauri.conf.json
{ "version": "1.1.0" }  // <- 古いまま!

// Cargo.toml
// version = "1.0.0"    // <- 更新忘れ!
```

```typescript
// OK: バージョンバンプスクリプトで一括更新
// scripts/bump-version.ts を使って全ファイルを同時に更新する
// (セクション 6.3 参照)
```

**問題点**: バージョン番号が一致しないと、Store 審査で却下されるケースや、自動更新が正しく動作しないケースが発生する。CI/CD パイプラインでバージョンの整合性チェックを入れるか、バンプスクリプトで一括管理する。

---

## FAQ

### Q1: Microsoft Store の審査でよく落ちるポイントは何ですか？

**A**: 最も多い理由は (1) プライバシーポリシーの不備または不適切な URL、(2) アプリの説明文とスクリーンショットの不一致、(3) クラッシュやハングアップなどの品質問題。特に Electron アプリでは、起動時間が遅いと審査に影響することがある。事前に Windows App Certification Kit (WACK) を実行してセルフチェックすることを強く推奨する。また、アプリが特定の API（位置情報、カメラ等）を使用する場合、AppxManifest.xml で対応する Capability を宣言しないと審査で却下される。

### Q2: Mac App Store のサンドボックス制限で困る機能はありますか？

**A**: ファイルシステムへの広範なアクセス、グローバルキーボードショートカット、他アプリとのプロセス間通信、カーネル拡張などがサンドボックスで制限される。特に開発ツールでは Terminal.app の操作や `/usr/local` 配下のファイルアクセスが困難。これらの機能が必須の場合は、Mac App Store 外での直接配布（DMG + Notarization）を検討すべき。なお、Security-Scoped Bookmarks を使えば、ユーザーが一度許可したディレクトリには継続的にアクセスできるため、部分的な回避策となる。

### Q3: GitHub Releases と Store 配布を両方やるべきですか？

**A**: 理想的には両方提供すべき。Store 配布はエンドユーザーにとってインストール・更新が容易で、信頼性も高い。一方 GitHub Releases は開発者コミュニティへのリーチと、Store 審査を待たない即時配布に有利。CI/CD パイプラインで両方のチャネルに同時デプロイする設計にすることで、追加の運用コストを最小化できる。

### Q4: Linux での配布はどの形式を選ぶべきですか？

**A**: ユーザー層と用途による。最も汎用的なのは AppImage で、単一バイナリとしてどの Linux ディストリビューションでも動作する。Ubuntu ユーザーが多い場合は Snap が良い（自動更新サポートあり）。GNOME/KDE デスクトップユーザーには Flatpak + Flathub が発見性に優れる。サーバー環境やシステム管理者向けなら deb/RPM + 自前リポジトリが適切。複数形式を並行提供するのがベストで、electron-builder はこれを1コマンドで実現できる。

### Q5: コード署名証明書の種類と選び方は？

**A**: Windows では Standard コード署名証明書（ソフトウェアトークン）と EV コード署名証明書（ハードウェアトークン必須）がある。EV 証明書は SmartScreen の即時信頼を獲得できるため、ダウンロード数の少ない初期段階では特に重要。価格は年間 $200-500 程度。macOS では Apple Developer Program ($99/年) に含まれる Developer ID 証明書を使用する。CI 環境では証明書の取り扱いに注意が必要で、Base64 エンコードして GitHub Secrets に保管し、ビルド時にデコードして使用するのが一般的なパターン。

### Q6: 企業内配布で Microsoft Store を使わずに MSIX を配布できますか？

**A**: はい、可能。MSIX サイドロード（LOB 配布）は Windows 10 1809 以降でサポートされている。Intune などの MDM ソリューション経由で配布するか、PowerShell の `Add-AppxPackage` コマンドで直接インストールできる。サイドロードの場合、自社の証明書で署名した MSIX を使用する。グループポリシーで開発者モードまたはサイドロードを有効化する必要がある点に注意。Store を介さないため審査は不要だが、自動更新は自前で実装する必要がある。

---

## まとめ

| 項目 | 要点 |
|------|------|
| Microsoft Store | MSIX パッケージング + Partner Center 提出。審査 1-3 日。WACK で事前検証 |
| Mac App Store | サンドボックス対応 + Notarization。審査 1-7 日。Provisioning Profile 管理 |
| GitHub Releases | 即時配布。electron-updater / Tauri updater と連携。Draft → QA → 公開 |
| Linux 配布 | Snap / Flatpak / AppImage / deb の複数形式を CI/CD で自動ビルド |
| 企業配布 | Intune (Windows) / Jamf (macOS) + MDM 連携でサイレントインストール |
| パッケージ形式 | MSIX(Win Store)、NSIS(Win直接)、DMG(Mac直接)、AppImage(Linux) |
| CI/CD | GitHub Actions でマルチプラットフォーム・マルチストア自動ビルド・署名・公開 |
| 署名 | Windows=Authenticode(EV推奨)、macOS=codesign+notarize を CI で自動化 |
| リリース戦略 | Draft リリース → QA → 公開の段階的プロセス。マルチストア統合ワークフロー |
| バージョン管理 | semver + git tag でリリースを自動トリガー。全ファイルの一括バンプ |

## 次に読むべきガイド

- [自動更新](./01-auto-update.md) -- electron-updater / Tauri updater による OTA 更新
- インストーラーのカスタマイズ -- NSIS スクリプトと WiX ツールセットの活用
- マルチアーキテクチャ対応 -- x64 / ARM64 / Universal Binary の戦略

## 参考文献

1. **Microsoft Store アプリの公開ガイド** -- https://learn.microsoft.com/ja-jp/windows/apps/publish/publish-your-app/overview -- Partner Center でのアプリ提出から公開までの公式ガイド
2. **Apple Developer - App Store 配布** -- https://developer.apple.com/distribute/ -- Mac App Store へのアプリ提出と Notarization の公式ドキュメント
3. **electron-builder 公式ドキュメント** -- https://www.electron.build/ -- マルチプラットフォームビルドと署名の包括的リファレンス
4. **GitHub Actions for Tauri** -- https://github.com/tauri-apps/tauri-action -- Tauri アプリのクロスプラットフォームビルドとリリース用 Action
5. **Snapcraft Documentation** -- https://snapcraft.io/docs -- Snap パッケージの作成と Snap Store への公開ガイド
6. **Flatpak Documentation** -- https://docs.flatpak.org/ -- Flatpak マニフェストの作成と Flathub への公開手順
7. **Microsoft Intune - LOB アプリ配布** -- https://learn.microsoft.com/ja-jp/mem/intune/apps/lob-apps-windows -- 企業向け MSIX サイドロード配布の公式ガイド

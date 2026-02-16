# 自動更新 (Auto Update)

> デスクトップアプリケーションの自動更新メカニズムを設計・実装し、ユーザーに透過的かつ安全にアップデートを届ける技術を体系的に学ぶ。

## この章で学ぶこと

1. **electron-updater / Tauri updater の設計思想と実装パターン** -- 主要フレームワークの更新機構を比較しながら、プロダクション品質の自動更新を構築する
2. **更新サーバーの構築と差分更新の最適化** -- 帯域を節約しつつ高速にパッチを配信するためのサーバーアーキテクチャとデルタ更新を理解する
3. **ロールバック戦略と障害復旧** -- 更新失敗時のフォールバック設計により、ユーザー体験を損なわない堅牢な更新パイプラインを構築する
4. **CI/CD パイプラインとの統合** -- GitHub Actions を活用し、ビルドから署名、配信までを完全自動化するワークフローを構築する

---

## 1. 自動更新の全体アーキテクチャ

### 1.1 更新フローの概要

```
+------------------+     (1) チェック要求      +------------------+
|                  | -----------------------> |                  |
|   デスクトップ    |     (2) マニフェスト応答   |   更新サーバー    |
|   アプリ         | <----------------------- |   (S3/GitHub等)  |
|                  |     (3) バイナリDL        |                  |
|                  | -----------------------> |                  |
+------------------+                          +------------------+
        |
        v  (4) 検証 & インストール
+------------------+
|  ローカル展開     |
|  再起動 or       |
|  バックグラウンド  |
+------------------+
```

### 1.2 更新チェックの戦略

```
+-------------------------------------------------------------+
|                  更新チェック戦略の選択                         |
+-------------------------------------------------------------+
| 戦略            | トリガー         | 適用場面               |
|-----------------|-----------------|------------------------|
| 起動時チェック    | アプリ起動        | 一般的なデスクトップアプリ |
| 定期ポーリング   | タイマー(1h等)   | 長時間起動アプリ        |
| プッシュ通知     | WebSocket/SSE   | リアルタイム性が必要     |
| 手動チェック     | ユーザー操作      | 開発者向けツール        |
+-------------------------------------------------------------+
```

### 1.3 更新方式の詳細比較

| 方式 | メリット | デメリット | 適用場面 |
|------|---------|-----------|---------|
| フル置換 | シンプル、確実 | ダウンロード量が大きい | 小規模アプリ |
| 差分パッチ | ダウンロード量が少ない | パッチ生成が複雑 | 大規模アプリ |
| asar 置換 (Electron) | JS 部分のみ更新 | ネイティブ変更不可 | フロントエンド中心 |
| サイドバイサイド | ロールバック容易 | ディスク使用量が増加 | エンタープライズ |
| バックグラウンド DL | UX が良い | メモリ・ディスク使用 | 一般消費者向け |

---

## 2. Electron + electron-updater

### 2.1 基本セットアップ

```typescript
// electron-builder.yml (抜粋)
// publish:
//   provider: github
//   owner: your-org
//   repo: your-app

// main.ts -- メインプロセス
import { autoUpdater } from 'electron-updater';
import { app, BrowserWindow, ipcMain } from 'electron';
import log from 'electron-log';

// ログ設定
autoUpdater.logger = log;
autoUpdater.logger.transports.file.level = 'info';

// 自動ダウンロードを無効化（ユーザーに確認させたい場合）
autoUpdater.autoDownload = false;
autoUpdater.autoInstallOnAppQuit = true;

function setupAutoUpdater(mainWindow: BrowserWindow): void {
  // 更新チェック（起動後 3 秒遅延）
  setTimeout(() => {
    autoUpdater.checkForUpdates();
  }, 3000);

  // イベントハンドラ
  autoUpdater.on('checking-for-update', () => {
    log.info('更新を確認中...');
  });

  autoUpdater.on('update-available', (info) => {
    log.info('更新あり:', info.version);
    mainWindow.webContents.send('update-available', {
      version: info.version,
      releaseDate: info.releaseDate,
      releaseNotes: info.releaseNotes,
    });
  });

  autoUpdater.on('update-not-available', () => {
    log.info('最新版です');
  });

  autoUpdater.on('download-progress', (progress) => {
    mainWindow.webContents.send('download-progress', {
      percent: progress.percent,
      transferred: progress.transferred,
      total: progress.total,
    });
  });

  autoUpdater.on('update-downloaded', (info) => {
    mainWindow.webContents.send('update-downloaded', info.version);
  });

  autoUpdater.on('error', (err) => {
    log.error('更新エラー:', err);
    mainWindow.webContents.send('update-error', err.message);
  });

  // レンダラーからの要求
  ipcMain.handle('start-download', () => autoUpdater.downloadUpdate());
  ipcMain.handle('install-update', () => autoUpdater.quitAndInstall());
}
```

### 2.2 レンダラー側の UI 実装

```typescript
// renderer/update-manager.ts
const { ipcRenderer } = window.require('electron');

class UpdateUI {
  private banner: HTMLElement;

  constructor() {
    this.banner = document.getElementById('update-banner')!;
    this.setupListeners();
  }

  private setupListeners(): void {
    ipcRenderer.on('update-available', (_e, info) => {
      this.showBanner(
        `v${info.version} が利用可能です`,
        'ダウンロード',
        () => ipcRenderer.invoke('start-download')
      );
    });

    ipcRenderer.on('download-progress', (_e, progress) => {
      this.updateProgress(progress.percent);
    });

    ipcRenderer.on('update-downloaded', (_e, version) => {
      this.showBanner(
        `v${version} の準備完了`,
        '再起動して更新',
        () => ipcRenderer.invoke('install-update')
      );
    });

    ipcRenderer.on('update-error', (_e, message) => {
      console.error('更新エラー:', message);
      // 次回起動時に再試行するため、ここではユーザー通知のみ
    });
  }

  private showBanner(text: string, btnLabel: string, onClick: () => void): void {
    this.banner.innerHTML = `
      <span>${text}</span>
      <button id="update-action">${btnLabel}</button>
    `;
    this.banner.style.display = 'flex';
    document.getElementById('update-action')!.addEventListener('click', onClick);
  }

  private updateProgress(percent: number): void {
    const bar = this.banner.querySelector('.progress-bar') as HTMLElement;
    if (bar) bar.style.width = `${percent.toFixed(1)}%`;
  }
}
```

### 2.3 electron-builder の公開設定比較

| 設定項目 | GitHub Releases | S3 / R2 | 自前サーバー |
|---------|----------------|---------|------------|
| `provider` | `github` | `s3` / `generic` | `generic` |
| コスト | 無料(Public) | 従量課金 | サーバー維持費 |
| プライベートリポ | トークン必要 | IAMロール | 任意の認証 |
| CDN | GitHub CDN | CloudFront等 | 自前構築 |
| 帯域制限 | 1GB/月(Free) | 無制限(課金) | 無制限 |
| 差分更新 | 非対応 | 対応可 | 対応可 |
| 設定難易度 | 低 | 中 | 高 |

### 2.4 S3 / CloudFront を使った配信設定

```yaml
# electron-builder.yml — S3 配信設定
publish:
  - provider: s3
    bucket: my-app-releases
    region: ap-northeast-1
    acl: public-read
    path: /releases/${os}/${arch}

# 環境変数で認証情報を設定
# AWS_ACCESS_KEY_ID
# AWS_SECRET_ACCESS_KEY
```

```typescript
// main/auto-updater-s3.ts — S3 からの更新チェック
import { autoUpdater } from 'electron-updater';

// S3 からの更新を設定
autoUpdater.setFeedURL({
  provider: 's3',
  bucket: 'my-app-releases',
  region: 'ap-northeast-1',
  path: `/releases/${process.platform}/${process.arch}`,
});

// Generic サーバーからの更新（自前サーバー）
autoUpdater.setFeedURL({
  provider: 'generic',
  url: 'https://updates.example.com/releases',
  channel: 'latest',
  useMultipleRangeRequest: true, // 差分ダウンロード対応
});
```

### 2.5 定期ポーリングの実装

```typescript
// main/update-scheduler.ts — 定期的な更新チェック
import { autoUpdater } from 'electron-updater';
import log from 'electron-log';

class UpdateScheduler {
  private intervalId: NodeJS.Timeout | null = null;
  private readonly checkIntervalMs: number;

  constructor(intervalHours: number = 4) {
    this.checkIntervalMs = intervalHours * 60 * 60 * 1000;
  }

  start(): void {
    // 起動後 10 秒で初回チェック
    setTimeout(() => this.check(), 10_000);

    // 以降は定期チェック
    this.intervalId = setInterval(() => this.check(), this.checkIntervalMs);
    log.info(`更新チェックスケジューラ開始: ${this.checkIntervalMs / 3600000}時間間隔`);
  }

  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
      log.info('更新チェックスケジューラ停止');
    }
  }

  private async check(): Promise<void> {
    try {
      log.info('定期更新チェック実行');
      const result = await autoUpdater.checkForUpdates();
      if (result?.updateInfo) {
        log.info(`利用可能な更新: v${result.updateInfo.version}`);
      }
    } catch (error) {
      log.warn('更新チェック失敗（次回リトライ）:', error);
      // エラーが連続する場合はチェック間隔を延長
    }
  }
}

export const updateScheduler = new UpdateScheduler(4); // 4時間間隔
```

---

## 3. Tauri Updater

### 3.1 Tauri v2 updater プラグインの設定

```bash
# Tauri v2 では updater はプラグインとして提供される
cargo add tauri-plugin-updater
npm install @tauri-apps/plugin-updater
```

```json
// src-tauri/tauri.conf.json — Tauri v2 の updater 設定
{
  "plugins": {
    "updater": {
      "pubkey": "dW50cnVzdGVkIGNvbW1lbnQ6IG1pbmlzaWduIHB1YmxpYyBr...",
      "endpoints": [
        "https://releases.example.com/{{target}}/{{arch}}/{{current_version}}"
      ],
      "dialog": false
    }
  }
}
```

```json
// src-tauri/capabilities/default.json — updater プラグインの権限
{
  "identifier": "main-capability",
  "windows": ["main"],
  "permissions": [
    "core:default",
    "updater:default"
  ]
}
```

### 3.2 minisign 鍵ペアの生成

```bash
# minisign のインストール
cargo install minisign

# 鍵ペアの生成（パスワードを設定）
minisign -G -p minisign.pub -s minisign.key

# 公開鍵の内容を tauri.conf.json の pubkey に設定
cat minisign.pub

# 秘密鍵は CI/CD の Secrets に保存
# TAURI_SIGNING_PRIVATE_KEY = (minisign.key の内容)
# TAURI_SIGNING_PRIVATE_KEY_PASSWORD = (生成時に設定したパスワード)
```

### 3.3 Rust 側の更新ロジック（v2 プラグイン版）

```rust
// src-tauri/src/main.rs — Tauri v2 updater プラグインの登録
fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            // バックグラウンドで更新チェックを開始
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                if let Err(e) = check_for_updates(handle).await {
                    log::error!("更新チェックエラー: {}", e);
                }
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("Tauri アプリの起動に失敗しました");
}

async fn check_for_updates(app: tauri::AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    use tauri_plugin_updater::UpdaterExt;

    // 5 秒待ってから更新チェック（起動直後を避ける）
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;

    let updater = app.updater()?;
    let response = updater.check().await?;

    if let Some(update) = response {
        log::info!("更新が利用可能: v{}", update.version);

        // フロントエンドに通知
        use tauri::Emitter;
        app.emit("update-available", serde_json::json!({
            "version": update.version,
            "body": update.body,
            "date": update.date,
        }))?;
    } else {
        log::info!("アプリは最新版です");
    }

    Ok(())
}
```

```rust
// src-tauri/src/commands/updater.rs — 更新コマンド
use tauri::AppHandle;
use tauri_plugin_updater::UpdaterExt;

/// 更新を確認するコマンド
#[tauri::command]
pub async fn check_update(app: AppHandle) -> Result<Option<UpdateInfo>, String> {
    let updater = app.updater().map_err(|e| e.to_string())?;
    let response = updater.check().await.map_err(|e| e.to_string())?;

    match response {
        Some(update) => Ok(Some(UpdateInfo {
            version: update.version.clone(),
            body: update.body.clone(),
            date: update.date.clone(),
        })),
        None => Ok(None),
    }
}

/// 更新をダウンロードしてインストールするコマンド
#[tauri::command]
pub async fn install_update(app: AppHandle) -> Result<(), String> {
    use tauri::Emitter;

    let updater = app.updater().map_err(|e| e.to_string())?;
    let response = updater.check().await.map_err(|e| e.to_string())?;

    if let Some(update) = response {
        // ダウンロードの進捗をフロントエンドに送信
        let app_clone = app.clone();
        let mut downloaded: u64 = 0;

        update
            .download_and_install(
                move |chunk_length, content_length| {
                    downloaded += chunk_length as u64;
                    let percent = content_length
                        .map(|total| (downloaded as f64 / total as f64) * 100.0)
                        .unwrap_or(0.0);

                    let _ = app_clone.emit("update-progress", serde_json::json!({
                        "downloaded": downloaded,
                        "total": content_length,
                        "percent": percent,
                    }));
                },
                || {
                    log::info!("ダウンロード完了、インストールを準備中...");
                },
            )
            .await
            .map_err(|e| format!("インストール失敗: {}", e))?;

        log::info!("更新インストール完了。再起動が必要です。");
        app.emit("update-installed", serde_json::json!({
            "version": update.version,
        })).map_err(|e| e.to_string())?;
    }

    Ok(())
}

#[derive(serde::Serialize)]
pub struct UpdateInfo {
    version: String,
    body: Option<String>,
    date: Option<String>,
}
```

### 3.4 フロントエンド (TypeScript) 側

```typescript
// src/lib/updater.ts — Tauri v2 updater プラグインの使用
import { check } from '@tauri-apps/plugin-updater'
import { relaunch } from '@tauri-apps/plugin-process'
import { listen } from '@tauri-apps/api/event'

interface UpdateStatus {
  available: boolean
  version?: string
  body?: string
  date?: string
}

// 更新チェックと状態管理
export class UpdateManager {
  private onStatusChange?: (status: UpdateStatus) => void
  private onProgressChange?: (percent: number) => void

  constructor(
    onStatusChange?: (status: UpdateStatus) => void,
    onProgressChange?: (percent: number) => void
  ) {
    this.onStatusChange = onStatusChange
    this.onProgressChange = onProgressChange
  }

  async checkForUpdate(): Promise<UpdateStatus> {
    try {
      const update = await check()

      if (update) {
        const status: UpdateStatus = {
          available: true,
          version: update.version,
          body: update.body ?? undefined,
          date: update.date ?? undefined,
        }
        this.onStatusChange?.(status)
        return status
      }

      const status: UpdateStatus = { available: false }
      this.onStatusChange?.(status)
      return status
    } catch (error) {
      console.error('更新チェックエラー:', error)
      return { available: false }
    }
  }

  async downloadAndInstall(): Promise<void> {
    try {
      const update = await check()
      if (!update) {
        throw new Error('更新が見つかりません')
      }

      let downloaded = 0
      let contentLength: number | undefined

      await update.downloadAndInstall((event) => {
        switch (event.event) {
          case 'Started':
            contentLength = event.data.contentLength ?? undefined
            console.log(`ダウンロード開始: ${contentLength} bytes`)
            break
          case 'Progress':
            downloaded += event.data.chunkLength
            if (contentLength) {
              const percent = (downloaded / contentLength) * 100
              this.onProgressChange?.(percent)
            }
            break
          case 'Finished':
            console.log('ダウンロード完了')
            this.onProgressChange?.(100)
            break
        }
      })

      // インストール後に再起動
      await relaunch()
    } catch (error) {
      console.error('更新インストールエラー:', error)
      throw error
    }
  }
}
```

```tsx
// src/components/UpdateNotification.tsx — 更新通知コンポーネント
import { useState, useEffect, useCallback } from 'react'
import { UpdateManager } from '../lib/updater'

export function UpdateNotification() {
  const [updateAvailable, setUpdateAvailable] = useState(false)
  const [version, setVersion] = useState('')
  const [releaseNotes, setReleaseNotes] = useState('')
  const [progress, setProgress] = useState(0)
  const [isDownloading, setIsDownloading] = useState(false)
  const [isInstalled, setIsInstalled] = useState(false)

  const manager = useCallback(() => new UpdateManager(
    (status) => {
      setUpdateAvailable(status.available)
      if (status.version) setVersion(status.version)
      if (status.body) setReleaseNotes(status.body)
    },
    (percent) => {
      setProgress(percent)
    }
  ), [])

  useEffect(() => {
    const mgr = manager()
    // 起動後 5 秒で更新チェック
    const timer = setTimeout(() => mgr.checkForUpdate(), 5000)
    return () => clearTimeout(timer)
  }, [manager])

  const handleDownload = async () => {
    setIsDownloading(true)
    try {
      const mgr = manager()
      await mgr.downloadAndInstall()
      setIsInstalled(true)
    } catch {
      setIsDownloading(false)
    }
  }

  if (!updateAvailable) return null

  return (
    <div className="update-notification">
      <div className="update-info">
        <h3>新しいバージョン v{version} が利用可能です</h3>
        {releaseNotes && (
          <div className="release-notes" dangerouslySetInnerHTML={{ __html: releaseNotes }} />
        )}
      </div>

      {isDownloading ? (
        <div className="progress-container">
          <div className="progress-bar" style={{ width: `${progress}%` }} />
          <span>{progress.toFixed(1)}%</span>
        </div>
      ) : isInstalled ? (
        <p>インストール完了。アプリを再起動します...</p>
      ) : (
        <button onClick={handleDownload}>今すぐ更新</button>
      )}
    </div>
  )
}
```

### 3.5 Tauri 更新マニフェスト (JSON) のフォーマット

更新サーバーが返す JSON マニフェストの仕様を理解することが重要である。

```json
// 更新サーバーが返す JSON レスポンスの例
// GET https://releases.example.com/windows-x86_64/1.0.0
{
  "version": "1.1.0",
  "notes": "バグ修正と新機能の追加\n- ファイル検索の高速化\n- ダークモード対応",
  "pub_date": "2025-12-15T10:00:00Z",
  "platforms": {
    "windows-x86_64": {
      "signature": "dW50cnVzdGVkIGNvbW1lbnQ6IHNpZ25hdHVyZSBmcm9tIHRhdXJpLXBsdWdpbi11cGRhdGVy...",
      "url": "https://cdn.example.com/releases/v1.1.0/my-app_1.1.0_x64-setup.nsis.zip"
    },
    "darwin-aarch64": {
      "signature": "dW50cnVzdGVkIGNvbW1lbnQ6IHNpZ25hdHVyZSBmcm9tIHRhdXJpLXBsdWdpbi11cGRhdGVy...",
      "url": "https://cdn.example.com/releases/v1.1.0/my-app_1.1.0_aarch64.app.tar.gz"
    },
    "darwin-x86_64": {
      "signature": "dW50cnVzdGVkIGNvbW1lbnQ6IHNpZ25hdHVyZSBmcm9tIHRhdXJpLXBsdWdpbi11cGRhdGVy...",
      "url": "https://cdn.example.com/releases/v1.1.0/my-app_1.1.0_x64.app.tar.gz"
    },
    "linux-x86_64": {
      "signature": "dW50cnVzdGVkIGNvbW1lbnQ6IHNpZ25hdHVyZSBmcm9tIHRhdXJpLXBsdWdpbi11cGRhdGVy...",
      "url": "https://cdn.example.com/releases/v1.1.0/my-app_1.1.0_amd64.AppImage.tar.gz"
    }
  }
}
```

---

## 4. 更新サーバーの構築

### 4.1 アーキテクチャ

```
+------------------+      +-------------------+      +-----------+
|  CI/CD           |      |  更新サーバー       |      |  CDN      |
|  (GitHub Actions)|----->|  (API + DB)       |----->|  (CF/S3)  |
|  ビルド & 署名    |      |  /update/check    |      |  バイナリ  |
+------------------+      |  /update/download  |      +-----------+
                          +-------------------+            |
                                   ^                       |
                                   |  (1) チェック           |  (3) DL
                                   |                       v
                          +-------------------+
                          |  デスクトップアプリ  |
                          +-------------------+
```

### 4.2 更新サーバー API の実装例

```typescript
// server/routes/update.ts (Express)
import express from 'express';
import semver from 'semver';

const router = express.Router();

interface Release {
  version: string;
  platform: string;
  arch: string;
  url: string;
  signature: string;
  size: number;
  sha256: string;
  releaseDate: string;
  critical: boolean;
  minVersion?: string; // この版未満は強制更新
}

// GET /update/check?platform=win32&arch=x64&version=1.2.0
router.get('/check', async (req, res) => {
  const { platform, arch, version } = req.query;

  const latest = await db.releases.findOne({
    where: { platform, arch, channel: 'stable' },
    order: [['releaseDate', 'DESC']],
  });

  if (!latest || !semver.gt(latest.version, version as string)) {
    return res.status(204).end(); // 更新なし
  }

  // 強制更新チェック
  const forceUpdate =
    latest.critical ||
    (latest.minVersion && semver.lt(version as string, latest.minVersion));

  res.json({
    version: latest.version,
    url: latest.url,
    signature: latest.signature,
    size: latest.size,
    sha256: latest.sha256,
    releaseNotes: latest.releaseNotes,
    forceUpdate,
    releaseDate: latest.releaseDate,
  });
});

// 段階的ロールアウト (カナリア)
router.get('/check/canary', async (req, res) => {
  const { platform, arch, version, userId } = req.query;
  const latest = await db.releases.findOne({
    where: { platform, arch, channel: 'canary' },
    order: [['releaseDate', 'DESC']],
  });

  if (!latest) return res.status(204).end();

  // ユーザーIDのハッシュで段階的に配信 (0-100%)
  const rolloutPercent = latest.rolloutPercent || 100;
  const hash = hashCode(userId as string) % 100;

  if (hash >= rolloutPercent) {
    return res.status(204).end(); // まだこのユーザーには配信しない
  }

  res.json({ version: latest.version, url: latest.url });
});

export default router;
```

### 4.3 Tauri 用の更新サーバー実装例

```typescript
// server/routes/tauri-update.ts — Tauri マニフェスト形式の更新 API
import express from 'express';
import semver from 'semver';

const router = express.Router();

// Tauri updater のエンドポイント
// GET /update/:target/:arch/:current_version
router.get('/:target/:arch/:current_version', async (req, res) => {
  const { target, arch, current_version } = req.params;

  // プラットフォーム名のマッピング
  const platform = `${target}-${arch}`;

  const latest = await db.releases.findOne({
    where: { channel: 'stable' },
    order: [['releaseDate', 'DESC']],
  });

  if (!latest || !semver.gt(latest.version, current_version)) {
    return res.status(204).end(); // 更新なし
  }

  // Tauri マニフェスト形式で返す
  const platformRelease = latest.platforms[platform];
  if (!platformRelease) {
    return res.status(204).end(); // このプラットフォーム向けのリリースなし
  }

  res.json({
    version: latest.version,
    notes: latest.releaseNotes || '',
    pub_date: latest.releaseDate,
    platforms: {
      [platform]: {
        signature: platformRelease.signature,
        url: platformRelease.url,
      },
    },
  });
});

export default router;
```

### 4.4 GitHub Releases をバックエンドとする更新サーバー

```typescript
// server/routes/github-proxy.ts — GitHub Releases をプロキシする更新サーバー
import express from 'express';
import { Octokit } from '@octokit/rest';

const router = express.Router();
const octokit = new Octokit({ auth: process.env.GITHUB_TOKEN });

const OWNER = 'your-org';
const REPO = 'your-app';

router.get('/:target/:arch/:current_version', async (req, res) => {
  const { target, arch, current_version } = req.params;
  const platform = `${target}-${arch}`;

  try {
    // 最新リリースを取得
    const { data: release } = await octokit.repos.getLatestRelease({
      owner: OWNER,
      repo: REPO,
    });

    const latestVersion = release.tag_name.replace('v', '');

    if (!semver.gt(latestVersion, current_version)) {
      return res.status(204).end();
    }

    // プラットフォーム別のアセットを検索
    const signatureAsset = release.assets.find(
      (a) => a.name.endsWith('.sig') && a.name.includes(platform)
    );
    const binaryAsset = release.assets.find(
      (a) => !a.name.endsWith('.sig') && a.name.includes(platform)
    );

    if (!binaryAsset || !signatureAsset) {
      return res.status(204).end();
    }

    // 署名ファイルの内容を取得
    const signatureResponse = await fetch(signatureAsset.browser_download_url);
    const signature = await signatureResponse.text();

    res.json({
      version: latestVersion,
      notes: release.body || '',
      pub_date: release.published_at,
      platforms: {
        [platform]: {
          signature: signature.trim(),
          url: binaryAsset.browser_download_url,
        },
      },
    });
  } catch (error) {
    console.error('GitHub API エラー:', error);
    res.status(500).json({ error: '更新チェックに失敗しました' });
  }
});

export default router;
```

---

## 5. 差分更新 (Delta Update)

### 5.1 差分更新の仕組み

```
+-----------------------------------------------------------+
|              差分更新のフロー                                |
+-----------------------------------------------------------+
|                                                           |
|  v1.0.0 バイナリ (80MB)                                   |
|     |                                                     |
|     v  bsdiff / zstd-delta                                |
|  差分パッチ (2MB)  ← v1.0.0 → v1.1.0 のデルタ             |
|     |                                                     |
|     v  bspatch (クライアント側)                             |
|  v1.1.0 バイナリ (81MB)                                   |
|     |                                                     |
|     v  SHA-256 検証                                       |
|  検証OK → 置換 & 再起動                                    |
|  検証NG → フルダウンロードにフォールバック                    |
|                                                           |
+-----------------------------------------------------------+
```

### 5.2 差分更新の比較

| 方式 | ツール | 圧縮率 | 生成速度 | 適用速度 | 適用場面 |
|------|-------|--------|---------|---------|---------|
| bsdiff/bspatch | bsdiff | 高(95%↑) | 遅い | 中 | Electron |
| courgette | Google製 | 最高(97%↑) | 遅い | 中 | Chrome系 |
| zstd-delta | zstd | 中(80%↑) | 速い | 速い | Tauri / Rust |
| VCDIFF (xdelta3) | xdelta | 中(85%↑) | 中 | 速い | 汎用 |
| Windows Delta | msdelta | 高(90%↑) | 中 | 速い | MSIX専用 |

### 5.3 差分パッチの生成パイプライン

```yaml
# .github/workflows/delta-update.yml — 差分パッチの生成
name: Generate Delta Patches

on:
  release:
    types: [published]

jobs:
  generate-delta:
    runs-on: ubuntu-latest
    steps:
      - name: Download previous release
        run: |
          # 1つ前のリリースを取得
          PREV_TAG=$(gh release list -L 2 --json tagName -q '.[1].tagName')
          gh release download "$PREV_TAG" -D previous/
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Download current release
        run: |
          gh release download "${{ github.event.release.tag_name }}" -D current/
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Generate bsdiff patches
        run: |
          for file in current/*.exe current/*.AppImage; do
            base=$(basename "$file")
            if [ -f "previous/$base" ]; then
              bsdiff "previous/$base" "current/$base" "patches/${base}.patch"
              echo "パッチ生成: $base ($(stat -f%z "patches/${base}.patch") bytes)"
            fi
          done

      - name: Upload patches to release
        run: |
          for patch in patches/*; do
            gh release upload "${{ github.event.release.tag_name }}" "$patch"
          done
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## 6. ロールバック戦略

### 6.1 ロールバックのフロー

```
+------------------------------------------------------------------+
|                   ロールバック判定フロー                             |
+------------------------------------------------------------------+
|                                                                  |
|  更新インストール完了                                              |
|      |                                                           |
|      v                                                           |
|  アプリ起動 → 起動成功?                                           |
|      |            |                                              |
|     YES          NO (クラッシュ or タイムアウト)                    |
|      |            |                                              |
|      v            v                                              |
|  ヘルスチェック   カウンター++                                     |
|  (API応答等)      |                                              |
|      |            v                                              |
|     OK?       3回連続失敗?                                        |
|    / \          /     \                                          |
|  YES  NO      YES     NO                                        |
|   |    |       |       |                                         |
|   v    v       v       v                                         |
| 正常  ロール   ロール   再試行                                     |
| 運用  バック   バック                                              |
|                                                                  |
+------------------------------------------------------------------+
```

### 6.2 ロールバック実装 (Electron)

```typescript
// main/rollback-manager.ts
import { app } from 'electron';
import fs from 'fs';
import path from 'path';

interface UpdateState {
  previousVersion: string;
  currentVersion: string;
  updateDate: string;
  crashCount: number;
  healthCheckPassed: boolean;
}

const STATE_FILE = path.join(app.getPath('userData'), 'update-state.json');
const MAX_CRASH_COUNT = 3;

export class RollbackManager {
  private state: UpdateState;

  constructor() {
    this.state = this.loadState();
  }

  /** アプリ起動時に呼び出す */
  async onAppStart(): Promise<void> {
    if (!this.state.healthCheckPassed) {
      this.state.crashCount++;
      this.saveState();

      if (this.state.crashCount >= MAX_CRASH_COUNT) {
        console.error(`${MAX_CRASH_COUNT}回連続で起動失敗。ロールバックを実行`);
        await this.rollback();
        return;
      }
    }

    // ヘルスチェック (5秒以内に完了しなければ失敗とみなす)
    const healthy = await Promise.race([
      this.performHealthCheck(),
      new Promise<boolean>((resolve) =>
        setTimeout(() => resolve(false), 5000)
      ),
    ]);

    if (healthy) {
      this.state.healthCheckPassed = true;
      this.state.crashCount = 0;
      this.saveState();
    }
  }

  private async performHealthCheck(): Promise<boolean> {
    try {
      // アプリ固有のヘルスチェック
      // 例: DB接続、設定ファイル読み込み、プラグインロード
      return true;
    } catch {
      return false;
    }
  }

  private async rollback(): Promise<void> {
    const backupDir = path.join(app.getPath('userData'), 'backup');
    if (fs.existsSync(backupDir)) {
      // バックアップからの復元ロジック
      // 実際にはインストーラーの仕組みに依存
      console.log(`v${this.state.previousVersion} にロールバック中...`);
    }
  }

  private loadState(): UpdateState {
    try {
      return JSON.parse(fs.readFileSync(STATE_FILE, 'utf-8'));
    } catch {
      return {
        previousVersion: '',
        currentVersion: app.getVersion(),
        updateDate: new Date().toISOString(),
        crashCount: 0,
        healthCheckPassed: true,
      };
    }
  }

  private saveState(): void {
    fs.writeFileSync(STATE_FILE, JSON.stringify(this.state, null, 2));
  }
}
```

### 6.3 Tauri でのロールバック実装

```rust
// src-tauri/src/rollback.rs — Tauri アプリのロールバック管理
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::fs;

#[derive(Serialize, Deserialize, Default)]
pub struct RollbackState {
    previous_version: String,
    current_version: String,
    crash_count: u32,
    health_check_passed: bool,
    update_date: String,
}

const MAX_CRASH_COUNT: u32 = 3;

pub struct RollbackManager {
    state: RollbackState,
    state_file: PathBuf,
}

impl RollbackManager {
    pub fn new(app_data_dir: &PathBuf) -> Self {
        let state_file = app_data_dir.join("rollback-state.json");
        let state = Self::load_state(&state_file);
        Self { state, state_file }
    }

    pub fn on_app_start(&mut self) -> Result<(), String> {
        if !self.state.health_check_passed {
            self.state.crash_count += 1;
            self.save_state()?;

            if self.state.crash_count >= MAX_CRASH_COUNT {
                log::error!("{}回連続で起動失敗。ロールバックを推奨", MAX_CRASH_COUNT);
                return Err("ロールバックが必要です".to_string());
            }
        }

        // ヘルスチェック
        if self.perform_health_check() {
            self.state.health_check_passed = true;
            self.state.crash_count = 0;
            self.save_state()?;
        }

        Ok(())
    }

    pub fn on_update_applied(&mut self, new_version: &str) -> Result<(), String> {
        self.state.previous_version = self.state.current_version.clone();
        self.state.current_version = new_version.to_string();
        self.state.health_check_passed = false;
        self.state.crash_count = 0;
        self.state.update_date = chrono::Utc::now().to_rfc3339();
        self.save_state()
    }

    fn perform_health_check(&self) -> bool {
        // アプリ固有のヘルスチェック
        // DB 接続テスト、設定ファイル読み込みなど
        true
    }

    fn load_state(path: &PathBuf) -> RollbackState {
        fs::read_to_string(path)
            .ok()
            .and_then(|content| serde_json::from_str(&content).ok())
            .unwrap_or_default()
    }

    fn save_state(&self) -> Result<(), String> {
        let content = serde_json::to_string_pretty(&self.state)
            .map_err(|e| format!("シリアライズエラー: {}", e))?;
        fs::write(&self.state_file, content)
            .map_err(|e| format!("ファイル書き込みエラー: {}", e))?;
        Ok(())
    }
}
```

---

## 7. コード署名と検証

### 7.1 プラットフォーム別の署名

| 項目 | Windows (Authenticode) | macOS (codesign) | Tauri (minisign) |
|------|----------------------|-----------------|-----------------|
| ツール | signtool.exe | codesign | minisign |
| 証明書 | EV/OV コード署名証明書 | Developer ID | Ed25519 鍵ペア |
| コスト | 年額$200-500 | Apple Developer $99/年 | 無料 |
| SmartScreen | EV: 即時信頼 | Gatekeeper対応 | 独自検証 |
| タイムスタンプ | RFC 3161 | Apple TS | 手動管理 |
| CI/CD統合 | Azure Key Vault | Keychain | 環境変数 |

### 7.2 GitHub Actions での署名自動化

```yaml
# .github/workflows/sign-and-release.yml — 署名付きリリース
name: Sign and Release

on:
  push:
    tags: ['v*']

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4

      - name: Import certificate
        run: |
          $bytes = [Convert]::FromBase64String("${{ secrets.WIN_CERT_BASE64 }}")
          [IO.File]::WriteAllBytes("cert.pfx", $bytes)

      - name: Build Tauri
        uses: tauri-apps/tauri-action@v0
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          TAURI_SIGNING_PRIVATE_KEY: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY }}
          TAURI_SIGNING_PRIVATE_KEY_PASSWORD: ${{ secrets.TAURI_KEY_PASSWORD }}

      - name: Sign with Authenticode
        run: |
          & "C:\Program Files (x86)\Windows Kits\10\bin\x64\signtool.exe" sign `
            /f cert.pfx `
            /p "${{ secrets.WIN_CERT_PASSWORD }}" `
            /tr http://timestamp.comodoca.com `
            /td sha256 `
            /fd sha256 `
            src-tauri\target\release\bundle\nsis\*.exe

      - name: Clean up certificate
        if: always()
        run: Remove-Item -Force cert.pfx -ErrorAction SilentlyContinue
```

---

## 8. CI/CD パイプライン統合

### 8.1 Tauri アプリの完全なリリースワークフロー

```yaml
# .github/workflows/tauri-release.yml — 完全なリリースパイプライン
name: Release

on:
  push:
    tags: ['v*']

permissions:
  contents: write

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
              tag_name: context.ref.replace('refs/tags/', ''),
              name: `Release ${context.ref.replace('refs/tags/', '')}`,
              draft: true,
              prerelease: false,
              generate_release_notes: true,
            });
            return data.id;

  build-tauri:
    needs: create-release
    strategy:
      fail-fast: false
      matrix:
        include:
          - platform: 'macos-latest'
            args: '--target aarch64-apple-darwin'
          - platform: 'macos-latest'
            args: '--target x86_64-apple-darwin'
          - platform: 'ubuntu-22.04'
            args: ''
          - platform: 'windows-latest'
            args: ''

    runs-on: ${{ matrix.platform }}
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - name: Install Rust stable
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.platform == 'macos-latest' && 'aarch64-apple-darwin,x86_64-apple-darwin' || '' }}

      - name: Install dependencies (Ubuntu)
        if: matrix.platform == 'ubuntu-22.04'
        run: |
          sudo apt-get update
          sudo apt-get install -y libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf

      - run: npm ci

      - name: Build Tauri
        uses: tauri-apps/tauri-action@v0
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          TAURI_SIGNING_PRIVATE_KEY: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY }}
          TAURI_SIGNING_PRIVATE_KEY_PASSWORD: ${{ secrets.TAURI_KEY_PASSWORD }}
        with:
          releaseId: ${{ needs.create-release.outputs.release_id }}
          args: ${{ matrix.args }}

  publish-release:
    needs: [create-release, build-tauri]
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

  update-manifest:
    needs: publish-release
    runs-on: ubuntu-latest
    steps:
      - name: Generate update manifest
        uses: actions/github-script@v7
        with:
          script: |
            // 最新リリースのアセットから更新マニフェストを生成
            const { data: release } = await github.rest.repos.getLatestRelease({
              owner: context.repo.owner,
              repo: context.repo.repo,
            });

            const manifest = {
              version: release.tag_name.replace('v', ''),
              notes: release.body,
              pub_date: release.published_at,
              platforms: {},
            };

            // 署名ファイルとバイナリをペアリング
            for (const asset of release.assets) {
              if (asset.name.endsWith('.sig')) {
                // 署名ファイルの内容を取得
                // プラットフォーム名を抽出してマニフェストに追加
              }
            }

            console.log('Generated manifest:', JSON.stringify(manifest, null, 2));
```

---

## アンチパターン

### アンチパターン 1: 強制即時再起動

```typescript
// NG: ユーザーの作業を中断して強制再起動
autoUpdater.on('update-downloaded', () => {
  autoUpdater.quitAndInstall(); // ユーザーの確認なしに即座に再起動
});

// OK: ユーザーに選択権を与え、適切なタイミングで更新
autoUpdater.on('update-downloaded', (info) => {
  const notification = new Notification({
    title: `v${info.version} の準備完了`,
    body: '次回起動時に自動適用されます。今すぐ再起動することもできます。',
  });
  notification.on('click', () => {
    // ユーザーが明示的にクリックした場合のみ再起動
    autoUpdater.quitAndInstall();
  });
  notification.show();
});
```

**問題点**: ユーザーが未保存の作業中に強制再起動されるとデータ喪失のリスクがある。特にドキュメント編集系アプリでは致命的。

### アンチパターン 2: 署名検証の省略

```typescript
// NG: 開発が面倒だからと署名検証をスキップ
autoUpdater.allowDowngrade = true;
autoUpdater.channel = 'latest';
// 署名なしのバイナリを配布

// OK: 必ず署名を検証し、HTTPS + ピン留めを使用
autoUpdater.allowDowngrade = false;
autoUpdater.autoRunAppAfterInstall = true;
// electron-builder の publish 設定で署名済みバイナリのみ配布
// Tauri: pubkey によるminisign検証を有効化
```

**問題点**: 署名検証なしでは中間者攻撃によりマルウェア入りバイナリに差し替えられるリスクがある。HTTPS だけでは不十分で、バイナリ自体の署名が必要。

### アンチパターン 3: エラーハンドリングの欠如

```typescript
// NG: 更新エラーを無視する
autoUpdater.checkForUpdates(); // エラーハンドリングなし

// OK: 全てのエラーケースを適切にハンドリング
try {
  const result = await autoUpdater.checkForUpdates();
  if (result) {
    log.info(`更新チェック完了: v${result.updateInfo.version}`);
  }
} catch (error) {
  if (error.message.includes('net::ERR_INTERNET_DISCONNECTED')) {
    log.warn('ネットワーク未接続のため更新チェックをスキップ');
  } else if (error.message.includes('ECONNREFUSED')) {
    log.warn('更新サーバーに接続できません。後で再試行します');
  } else {
    log.error('更新チェックで予期しないエラー:', error);
    // Sentry 等のエラー監視サービスに報告
  }
}
```

**問題点**: ネットワークエラー、サーバーダウン、DNS 障害など様々な理由で更新チェックは失敗しうる。エラーを握りつぶすとユーザーが永遠に古いバージョンを使い続ける。

---

## FAQ

### Q1: 更新チェックの頻度はどのくらいが適切ですか？

**A**: 一般的なデスクトップアプリでは「起動時 + 4〜6時間ごと」が推奨される。起動時チェックだけでは長時間起動しっぱなしのアプリで更新が遅れる。ただし、セキュリティ修正を含む緊急更新の場合は、プッシュ通知（WebSocket等）で即時通知する仕組みを別途用意すべき。チェック頻度が高すぎるとサーバー負荷とユーザーの通信量が増加するため、バランスが重要。

### Q2: Electron と Tauri で自動更新の実装難易度はどう違いますか？

**A**: Electron (electron-updater) は成熟しており、GitHub Releases との統合がほぼゼロ設定で動作する。一方 Tauri は minisign による署名が必須で初期設定がやや煩雑だが、セキュリティ面ではデフォルトで強固。Tauri v2 では updater プラグインとして分離され、より柔軟な設定が可能になった。どちらも CI/CD パイプラインの構築が本質的な工数の大半を占める。

### Q3: 段階的ロールアウト（カナリアリリース）はどう実装すればよいですか？

**A**: 更新サーバー側でユーザー ID のハッシュ値を使い、ロールアウト率に基づいて更新を配信する方式が一般的。例えば最初の24時間は 5% のユーザーに配信し、クラッシュレートが閾値以下であれば 25% → 50% → 100% と拡大する。Sentry や Crashlytics と連携してクラッシュレートを自動監視し、異常検知時に自動で配信を停止する仕組みも重要。

### Q4: オフライン環境のユーザーにはどう対応すべきですか？

**A**: オフライン環境では自動更新が機能しないため、以下の対策を講じる。(1) 更新チェック失敗時にサイレントにリトライするスケジューリング、(2) USB メモリ等でのオフラインアップデートパッケージの提供、(3) 企業向けには WSUS や SCCM 等のソフトウェア配布ツールとの連携、(4) 古いバージョンでも最低限の機能が動作するようにバックエンド API のバージョン互換性を維持する。

### Q5: 更新中にアプリがクラッシュした場合はどうなりますか？

**A**: フレームワークによって挙動が異なる。Electron の electron-updater はダウンロード完了後にバックアップを作成し、インストール失敗時は古いバージョンが残る。Tauri は更新バイナリのダウンロードと署名検証が完了してからインストールを実行するため、途中クラッシュしても元のバイナリが破損することはない。ただし、いずれの場合もアプリ側でロールバック機構を実装し、3回連続起動失敗した場合に前バージョンに戻す仕組みを設けることが推奨される。

---

## まとめ

| 項目 | 要点 |
|------|------|
| フレームワーク選択 | Electron は electron-updater、Tauri は updater プラグインが標準 |
| 更新サーバー | GitHub Releases（小規模）、S3+CloudFront（中〜大規模）、自前（完全制御） |
| 差分更新 | bsdiff（Electron）、zstd-delta（Tauri/Rust）で帯域90%以上削減可能 |
| 署名検証 | Windows=Authenticode、macOS=codesign、Tauri=minisign を必ず実施 |
| ロールバック | クラッシュカウンター方式で自動ロールバック、バックアップ保持必須 |
| 段階的配信 | ユーザーIDハッシュによるカナリアリリースでリスク最小化 |
| ユーザー体験 | 強制再起動を避け、バックグラウンドDL + 次回起動時適用が理想 |
| CI/CD | 署名・ビルド・公開を完全自動化し、手動リリースを排除する |
| エラーハンドリング | ネットワークエラー、サーバーダウンに備えたリトライとフォールバック |
| Tauri v2 updater | プラグインベースで柔軟な設定。minisign 署名が必須 |

## 次に読むべきガイド

- [ストア配布 (Microsoft Store / Mac App Store)](./02-store-distribution.md) -- MSIX パッケージングとストア審査の実践
- コード署名の詳細 -- EV 証明書の取得と CI/CD での安全な鍵管理
- CI/CD パイプライン構築 -- GitHub Actions / Azure Pipelines でのビルド自動化

## 参考文献

1. **electron-updater 公式ドキュメント** -- https://www.electron.build/auto-update -- electron-builder の自動更新モジュールの包括的なリファレンス
2. **Tauri Updater Plugin** -- https://tauri.app/plugin/updater/ -- Tauri v2 の updater プラグイン設定と署名ガイド
3. **Squirrel.Windows** -- https://github.com/Squirrel/Squirrel.Windows -- .NET ベースの Windows 自動更新フレームワーク（electron-updater の内部で使用）
4. **bsdiff / bspatch** -- https://www.daemonology.net/bsdiff/ -- Colin Percival によるバイナリ差分アルゴリズムの原論文とツール
5. **minisign** -- https://jedisct1.github.io/minisign/ -- 最小限の署名ツール（Tauri が使用）
6. **tauri-apps/tauri-action** -- https://github.com/tauri-apps/tauri-action -- Tauri の GitHub Actions アクション

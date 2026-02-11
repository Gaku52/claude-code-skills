# 自動更新 (Auto Update)

> デスクトップアプリケーションの自動更新メカニズムを設計・実装し、ユーザーに透過的かつ安全にアップデートを届ける技術を体系的に学ぶ。

## この章で学ぶこと

1. **electron-updater / Tauri updater の設計思想と実装パターン** -- 主要フレームワークの更新機構を比較しながら、プロダクション品質の自動更新を構築する
2. **更新サーバーの構築と差分更新の最適化** -- 帯域を節約しつつ高速にパッチを配信するためのサーバーアーキテクチャとデルタ更新を理解する
3. **ロールバック戦略と障害復旧** -- 更新失敗時のフォールバック設計により、ユーザー体験を損なわない堅牢な更新パイプラインを構築する

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

---

## 3. Tauri Updater

### 3.1 tauri.conf.json の設定

```jsonc
// tauri.conf.json
{
  "tauri": {
    "updater": {
      "active": true,
      "dialog": false,
      "endpoints": [
        "https://releases.example.com/{{target}}/{{arch}}/{{current_version}}"
      ],
      "pubkey": "dW50cnVzdGVkIGNvbW1lbnQ6IG1pbmlzaWduIHB1YmxpYyBr..."
    },
    "bundle": {
      "active": true,
      "targets": ["msi", "nsis", "app", "dmg"]
    }
  }
}
```

### 3.2 Rust 側の更新ロジック

```rust
// src-tauri/src/updater.rs
use tauri::updater::UpdateResponse;
use tauri::{AppHandle, Manager};

#[tauri::command]
pub async fn check_for_update(app: AppHandle) -> Result<Option<String>, String> {
    match app.updater().check().await {
        Ok(update) => match update {
            UpdateResponse::Update { manifest, .. } => {
                Ok(Some(manifest.version.clone()))
            }
            UpdateResponse::UpToDate => Ok(None),
        },
        Err(e) => Err(format!("更新チェック失敗: {}", e)),
    }
}

#[tauri::command]
pub async fn install_update(app: AppHandle) -> Result<(), String> {
    let update = app
        .updater()
        .check()
        .await
        .map_err(|e| e.to_string())?;

    if let UpdateResponse::Update { manifest, .. } = update {
        // フロントエンドに進捗を送信
        let window = app.get_window("main").unwrap();
        window
            .emit("update-installing", &manifest.version)
            .unwrap();

        // ダウンロード & インストール
        update
            .download_and_install()
            .await
            .map_err(|e| format!("インストール失敗: {}", e))?;
    }
    Ok(())
}
```

### 3.3 フロントエンド (TypeScript) 側

```typescript
// src/lib/updater.ts
import { checkUpdate, installUpdate } from '@tauri-apps/api/updater';
import { relaunch } from '@tauri-apps/api/process';
import { listen } from '@tauri-apps/api/event';

export async function handleUpdate(): Promise<void> {
  try {
    const { shouldUpdate, manifest } = await checkUpdate();

    if (shouldUpdate && manifest) {
      const userConfirmed = confirm(
        `v${manifest.version} が利用可能です。更新しますか?\n\n${manifest.body}`
      );

      if (userConfirmed) {
        // ダウンロード進捗の監視
        const unlisten = await listen('tauri://update-download-progress', (event) => {
          const { chunkLength, contentLength } = event.payload as {
            chunkLength: number;
            contentLength: number;
          };
          console.log(`進捗: ${chunkLength}/${contentLength}`);
        });

        await installUpdate();
        unlisten();
        await relaunch();
      }
    }
  } catch (error) {
    console.error('更新エラー:', error);
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

---

## FAQ

### Q1: 更新チェックの頻度はどのくらいが適切ですか？

**A**: 一般的なデスクトップアプリでは「起動時 + 4〜6時間ごと」が推奨される。起動時チェックだけでは長時間起動しっぱなしのアプリで更新が遅れる。ただし、セキュリティ修正を含む緊急更新の場合は、プッシュ通知（WebSocket等）で即時通知する仕組みを別途用意すべき。チェック頻度が高すぎるとサーバー負荷とユーザーの通信量が増加するため、バランスが重要。

### Q2: Electron と Tauri で自動更新の実装難易度はどう違いますか？

**A**: Electron (electron-updater) は成熟しており、GitHub Releases との統合がほぼゼロ設定で動作する。一方 Tauri は minisign による署名が必須で初期設定がやや煩雑だが、セキュリティ面ではデフォルトで強固。Tauri v2 では updater プラグインとして分離され、より柔軟な設定が可能になった。どちらも CI/CD パイプラインの構築が本質的な工数の大半を占める。

### Q3: 段階的ロールアウト（カナリアリリース）はどう実装すればよいですか？

**A**: 更新サーバー側でユーザー ID のハッシュ値を使い、ロールアウト率に基づいて更新を配信する方式が一般的。例えば最初の24時間は 5% のユーザーに配信し、クラッシュレートが閾値以下であれば 25% → 50% → 100% と拡大する。Sentry や Crashlytics と連携してクラッシュレートを自動監視し、異常検知時に自動で配信を停止する仕組みも重要。

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

## 次に読むべきガイド

- [ストア配布 (Microsoft Store / Mac App Store)](./02-store-distribution.md) -- MSIX パッケージングとストア審査の実践
- コード署名の詳細 -- EV 証明書の取得と CI/CD での安全な鍵管理
- CI/CD パイプライン構築 -- GitHub Actions / Azure Pipelines でのビルド自動化

## 参考文献

1. **electron-updater 公式ドキュメント** -- https://www.electron.build/auto-update -- electron-builder の自動更新モジュールの包括的なリファレンス
2. **Tauri Updater Plugin** -- https://tauri.app/plugin/updater/ -- Tauri v2 の updater プラグイン設定と署名ガイド
3. **Squirrel.Windows** -- https://github.com/Squirrel/Squirrel.Windows -- .NET ベースの Windows 自動更新フレームワーク（electron-updater の内部で使用）
4. **bsdiff / bspatch** -- https://www.daemonology.net/bsdiff/ -- Colin Percival によるバイナリ差分アルゴリズムの原論文とツール

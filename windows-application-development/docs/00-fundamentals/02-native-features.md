# ネイティブ機能の活用

> デスクトップアプリの真価はネイティブ機能にある。ファイルシステムアクセス、システム通知、トレイアイコン、自動起動、グローバルショートカット、クリップボード、ドラッグ&ドロップまで、OS との深い統合を解説する。

## この章で学ぶこと

- [ ] ファイルダイアログとファイルシステム操作を実装できる
- [ ] システム通知・トレイアイコンを活用できる
- [ ] グローバルショートカット・自動起動を設定できる

---

## 1. ファイルシステムアクセス

```
ファイル操作フロー:

  ユーザー操作          メインプロセス          OS
  ┌──────┐  IPC      ┌──────────┐  API    ┌────┐
  │開くボタン│ ──────→ │dialog.   │ ─────→ │FS  │
  │クリック │         │showOpen  │        │    │
  └──────┘          │Dialog()  │        │    │
                    └─────┬────┘        └──┬─┘
                          │ パス            │ データ
                          ▼                │
                    ┌──────────┐           │
                    │fs.read   │ ←─────────┘
                    │File()    │
                    └─────┬────┘
                          │ 内容
                    IPC   ▼
  ┌──────┐        ┌──────────┐
  │エディタ│ ←───── │レスポンス  │
  │表示   │        │返却      │
  └──────┘        └──────────┘
```

### Electron 実装

```typescript
// main.ts — ファイルダイアログ
import { dialog, ipcMain } from 'electron';
import fs from 'fs/promises';

ipcMain.handle('file:open', async () => {
  const result = await dialog.showOpenDialog({
    title: 'ファイルを開く',
    filters: [
      { name: 'テキスト', extensions: ['txt', 'md'] },
      { name: 'JSON', extensions: ['json'] },
      { name: 'すべて', extensions: ['*'] },
    ],
    properties: ['openFile'],
  });

  if (result.canceled || !result.filePaths[0]) return null;

  const filePath = result.filePaths[0];
  const content = await fs.readFile(filePath, 'utf-8');
  return { path: filePath, content };
});

ipcMain.handle('file:save', async (_event, filePath: string, content: string) => {
  await fs.writeFile(filePath, content, 'utf-8');
  return true;
});

ipcMain.handle('file:saveAs', async (_event, content: string) => {
  const result = await dialog.showSaveDialog({
    title: '名前を付けて保存',
    filters: [{ name: 'Markdown', extensions: ['md'] }],
  });

  if (result.canceled || !result.filePath) return null;
  await fs.writeFile(result.filePath, content, 'utf-8');
  return result.filePath;
});
```

### Tauri 実装

```typescript
// フロントエンド — Tauri ファイル操作
import { open, save } from '@tauri-apps/plugin-dialog';
import { readTextFile, writeTextFile } from '@tauri-apps/plugin-fs';

async function openFile() {
  const path = await open({
    title: 'ファイルを開く',
    filters: [{ name: 'Text', extensions: ['txt', 'md'] }],
  });
  if (!path) return null;

  const content = await readTextFile(path as string);
  return { path, content };
}

async function saveFile(path: string, content: string) {
  await writeTextFile(path, content);
}
```

---

## 2. システム通知

```typescript
// Electron — 通知
import { Notification } from 'electron';

function showNotification(title: string, body: string) {
  const notification = new Notification({
    title,
    body,
    icon: path.join(__dirname, 'assets/icon.png'),
    silent: false,
  });

  notification.on('click', () => {
    mainWindow?.show();
    mainWindow?.focus();
  });

  notification.show();
}

// Tauri — 通知
import { sendNotification, requestPermission, isPermissionGranted }
  from '@tauri-apps/plugin-notification';

async function notify(title: string, body: string) {
  let granted = await isPermissionGranted();
  if (!granted) {
    const permission = await requestPermission();
    granted = permission === 'granted';
  }
  if (granted) {
    sendNotification({ title, body });
  }
}
```

---

## 3. システムトレイ

```typescript
// Electron — トレイアイコン
import { Tray, Menu, nativeImage } from 'electron';

let tray: Tray | null = null;

function createTray() {
  const icon = nativeImage.createFromPath(
    path.join(__dirname, 'assets/tray-icon.png')
  );
  // macOS: 16x16 or 22x22、Windows: 16x16
  tray = new Tray(icon.resize({ width: 16, height: 16 }));

  const contextMenu = Menu.buildFromTemplate([
    { label: '表示', click: () => mainWindow?.show() },
    { label: '設定', click: () => openSettings() },
    { type: 'separator' },
    { label: '終了', click: () => app.quit() },
  ]);

  tray.setToolTip('My App');
  tray.setContextMenu(contextMenu);

  // クリックでウィンドウ表示
  tray.on('click', () => {
    mainWindow?.isVisible() ? mainWindow.hide() : mainWindow?.show();
  });
}
```

---

## 4. グローバルショートカット

```typescript
// Electron — グローバルショートカット
import { globalShortcut } from 'electron';

app.whenReady().then(() => {
  // Ctrl+Shift+Space でアプリを表示/非表示
  globalShortcut.register('CommandOrControl+Shift+Space', () => {
    if (mainWindow?.isVisible()) {
      mainWindow.hide();
    } else {
      mainWindow?.show();
      mainWindow?.focus();
    }
  });
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
});
```

---

## 5. 自動起動

```typescript
// Electron — ログイン時自動起動
import { app } from 'electron';

function setAutoLaunch(enabled: boolean) {
  app.setLoginItemSettings({
    openAtLogin: enabled,
    openAsHidden: true, // macOS: 非表示で起動
    args: ['--hidden'],  // Windows: 引数
  });
}

function getAutoLaunchStatus(): boolean {
  return app.getLoginItemSettings().openAtLogin;
}
```

---

## 6. クリップボード・ドラッグ&ドロップ

```typescript
// Electron — クリップボード
import { clipboard } from 'electron';

ipcMain.handle('clipboard:read', () => clipboard.readText());
ipcMain.handle('clipboard:write', (_e, text: string) => clipboard.writeText(text));
ipcMain.handle('clipboard:readImage', () => {
  const image = clipboard.readImage();
  return image.isEmpty() ? null : image.toDataURL();
});

// レンダラー — ドラッグ&ドロップ
// React コンポーネント
function DropZone() {
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    files.forEach(file => {
      console.log('Dropped:', file.path, file.name, file.size);
    });
  };

  return (
    <div
      onDragOver={e => e.preventDefault()}
      onDrop={handleDrop}
      style={{ border: '2px dashed #ccc', padding: 40 }}
    >
      ファイルをドロップ
    </div>
  );
}
```

---

## FAQ

### Q1: ファイルアクセスのセキュリティは？
メインプロセスでパス検証を必ず行う。ユーザーが選択したパス以外へのアクセスは拒否する。Tauri は capabilities で制御。

### Q2: macOS と Windows で通知の動作は違う？
macOS は Notification Center 経由、Windows は Action Center 経由。アイコンサイズやアクションボタンの仕様が異なる。

### Q3: トレイアイコンの推奨サイズは？
macOS: 16x16〜22x22（@2x 対応）、Windows: 16x16〜32x32。Template Image（macOS）を使うとダークモード対応。

---

## まとめ

| 機能 | Electron | Tauri |
|------|----------|-------|
| ファイルダイアログ | dialog.showOpenDialog | @tauri-apps/plugin-dialog |
| 通知 | Notification | @tauri-apps/plugin-notification |
| トレイ | Tray | TrayIcon |
| ショートカット | globalShortcut | @tauri-apps/plugin-global-shortcut |
| 自動起動 | app.setLoginItemSettings | @tauri-apps/plugin-autostart |
| クリップボード | clipboard | @tauri-apps/plugin-clipboard |

---

## 次に読むべきガイド
→ [[03-cross-platform.md]] — クロスプラットフォーム対応

---

## 参考文献
1. Electron. "Native File Dialogs." electronjs.org/docs, 2024.
2. Electron. "Tray." electronjs.org/docs/api/tray, 2024.
3. Tauri. "Plugins." tauri.app/plugin, 2024.

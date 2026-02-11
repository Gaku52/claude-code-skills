# クロスプラットフォーム対応

> 1つのコードベースで Windows・macOS・Linux をサポートする。プラットフォーム検出、OS 固有 API の抽象化、パス処理、UI/UX の差異対応まで、クロスプラットフォーム設計を解説する。

## この章で学ぶこと

- [ ] プラットフォーム検出と条件分岐を実装できる
- [ ] OS 固有の UI/UX 差異に対応できる
- [ ] パス・改行コード等の環境差異を適切に処理できる

---

## 1. プラットフォーム検出

```typescript
// プラットフォーム検出
const platform = process.platform;
// 'win32' | 'darwin' | 'linux'

// OS 別処理の抽象化
function getPlatformConfig() {
  switch (process.platform) {
    case 'win32':
      return {
        appDataPath: process.env.APPDATA!,
        separator: '\\',
        lineEnding: '\r\n',
        shortcutModifier: 'Ctrl',
      };
    case 'darwin':
      return {
        appDataPath: `${process.env.HOME}/Library/Application Support`,
        separator: '/',
        lineEnding: '\n',
        shortcutModifier: 'Cmd',
      };
    case 'linux':
      return {
        appDataPath: process.env.XDG_CONFIG_HOME || `${process.env.HOME}/.config`,
        separator: '/',
        lineEnding: '\n',
        shortcutModifier: 'Ctrl',
      };
    default:
      throw new Error(`Unsupported platform: ${process.platform}`);
  }
}
```

---

## 2. パス処理

```
パス処理の注意点:

  Windows: C:\Users\gaku\Documents\file.txt
  macOS:   /Users/gaku/Documents/file.txt
  Linux:   /home/gaku/Documents/file.txt

  解決策: path モジュールを常に使用する
```

```typescript
import path from 'path';
import { app } from 'electron';

// ✓ 正しい: path.join を使用
const configPath = path.join(app.getPath('userData'), 'config.json');

// ✗ 間違い: 文字列結合
const badPath = app.getPath('userData') + '/config.json'; // Windows で壊れる

// アプリデータの保存先
const paths = {
  userData: app.getPath('userData'),      // アプリ設定
  documents: app.getPath('documents'),    // ユーザードキュメント
  downloads: app.getPath('downloads'),    // ダウンロード
  temp: app.getPath('temp'),              // 一時ファイル
  home: app.getPath('home'),              // ホーム
  desktop: app.getPath('desktop'),        // デスクトップ
};
```

---

## 3. メニューバーの OS 差異

```
メニューバーの違い:

  macOS:   画面上部にグローバルメニューバー
           アプリ名メニュー（About/Preferences/Quit）が必須
           Cmd+Q で終了

  Windows: ウィンドウ上部にメニューバー
           File メニューに Exit
           Alt+F4 で終了

  Linux:   ウィンドウ上部（デスクトップ環境による）
           File メニューに Quit
```

```typescript
import { Menu, app } from 'electron';

function createMenu() {
  const isMac = process.platform === 'darwin';

  const template: Electron.MenuItemConstructorOptions[] = [
    // macOS: アプリ名メニュー
    ...(isMac ? [{
      label: app.name,
      submenu: [
        { role: 'about' as const },
        { type: 'separator' as const },
        { label: '設定...', accelerator: 'Cmd+,', click: openSettings },
        { type: 'separator' as const },
        { role: 'quit' as const },
      ],
    }] : []),
    // File メニュー
    {
      label: 'ファイル',
      submenu: [
        { label: '新規', accelerator: 'CmdOrCtrl+N', click: newFile },
        { label: '開く', accelerator: 'CmdOrCtrl+O', click: openFile },
        { label: '保存', accelerator: 'CmdOrCtrl+S', click: saveFile },
        { type: 'separator' },
        ...(isMac ? [] : [
          { label: '設定', click: openSettings },
          { type: 'separator' as const },
          { label: '終了', accelerator: 'Alt+F4', click: () => app.quit() },
        ]),
      ],
    },
    // Edit メニュー
    {
      label: '編集',
      submenu: [
        { role: 'undo' as const },
        { role: 'redo' as const },
        { type: 'separator' as const },
        { role: 'cut' as const },
        { role: 'copy' as const },
        { role: 'paste' as const },
        { role: 'selectAll' as const },
      ],
    },
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}
```

---

## 4. ウィンドウ管理の差異

```typescript
// macOS: ウィンドウを閉じてもアプリは終了しない
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// macOS: Dock アイコンクリックでウィンドウ再作成
app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// タイトルバーのカスタマイズ
const win = new BrowserWindow({
  titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
  // macOS: トラフィックライト（閉じる/最小化/最大化）の位置
  trafficLightPosition: { x: 15, y: 15 },
  // Windows: タイトルバー非表示時のフレーム
  frame: process.platform === 'darwin' ? true : true,
});
```

---

## 5. CI/CD マルチプラットフォームビルド

```yaml
# .github/workflows/build.yml
name: Build
on:
  push:
    tags: ['v*']

jobs:
  build:
    strategy:
      matrix:
        include:
          - os: windows-latest
            target: win
          - os: macos-latest
            target: mac
          - os: ubuntu-latest
            target: linux

    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: pnpm install
      - run: pnpm build
      - run: pnpm package:${{ matrix.target }}
      - uses: actions/upload-artifact@v4
        with:
          name: app-${{ matrix.target }}
          path: out/make/**/*
```

---

## FAQ

### Q1: Windows と macOS でキーボードショートカットをどう統一する？
`CmdOrCtrl` を使用。macOS では Cmd、Windows/Linux では Ctrl に自動マッピングされる。

### Q2: macOS のダークモードにどう対応する？
`nativeTheme.shouldUseDarkColors` で現在のテーマを検出。`nativeTheme.on('updated', ...)` で変更を監視。CSS では `prefers-color-scheme` メディアクエリ。

### Q3: Apple Silicon (arm64) と Intel (x64) の両方に対応するには？
Electron: `--arch=universal` でユニバーサルバイナリを作成。Tauri: `--target aarch64-apple-darwin` と `--target x86_64-apple-darwin` で個別ビルド後、`lipo` で結合。

---

## まとめ

| 項目 | Windows | macOS | Linux |
|------|---------|-------|-------|
| パス区切り | `\` | `/` | `/` |
| 改行コード | `\r\n` | `\n` | `\n` |
| 修飾キー | Ctrl | Cmd | Ctrl |
| メニュー位置 | ウィンドウ内 | 画面上部 | ウィンドウ内 |
| 終了動作 | 全Window閉じ→終了 | Window閉じ→常駐 | 全Window閉じ→終了 |

---

## 次に読むべきガイド
→ [[../01-wpf-and-winui/00-windows-ui-frameworks.md]] — Windows UI フレームワーク

---

## 参考文献
1. Electron. "Platform Considerations." electronjs.org/docs, 2024.
2. Apple. "Human Interface Guidelines." developer.apple.com/design, 2024.
3. Microsoft. "Windows App Design." learn.microsoft.com, 2024.
